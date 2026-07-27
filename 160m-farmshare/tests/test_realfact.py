"""Tests for corpusgen.realfact and scripts.fetch_realfacts (T1) — written
before the implementation. Offline only: PopQA rows are synthesized, never
downloaded; importing scripts.fetch_realfacts must not import `datasets`."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from corpusgen import realfact
from corpusgen.bios import FIRST_NAMES, LAST_NAMES, MIDDLE_NAMES
from corpusgen.records import DB_END, DB_RETRIEVE, DB_START
from scripts.fetch_realfacts import cap_rows, clean_rows, parse_possible_answers

_WRAP = re.compile(r"<\|db_start\|>.*?<\|db_retrieve\|>", re.S)
_KEY = re.compile(r"<\|db_start\|>(.*?)<\|db_retrieve\|>", re.S)


def _unwrap(split_text: str) -> str:
    return _WRAP.sub("", split_text).replace(DB_END, "")


def _key_of(doc) -> str:
    return _KEY.search(doc.split_text()).group(1)


def _mk_facts() -> list:
    facts = []
    for prop, n in (("author", 10), ("capital of", 8), ("director", 5)):
        tag = prop.split()[0].title()
        for i in range(n):
            obj = f"{tag.lower()}-obj-{i}"
            facts.append(realfact.RealFact(
                subj=f"{tag} Entity {i}",
                prop=prop,
                obj=obj,
                question=f"What is the {prop} of {tag} Entity {i}?",
                possible_answers=(obj, f"{obj} alias"),
            ))
    return facts


FACTS = _mk_facts()
N = len(FACTS)  # 23
PROPS = {"author", "capital of", "director"}


# ------------------------------------------------------------ fetch: cleaning


def test_fetch_help_cli_runs_from_repo_root():
    """The documented `python scripts/fetch_realfacts.py --help` invocation
    must not crash on import: launching by file path puts scripts/ (not the
    repo root) on sys.path, so the script must add the repo root itself."""
    repo_root = Path(__file__).resolve().parent.parent
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "scripts/fetch_realfacts.py", "--help"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _row(subj="Ada Lovelace", prop="occupation", obj="mathematician",
         question=None, **extra):
    row = {
        "subj": subj, "prop": prop, "obj": obj,
        "question": question if question is not None else f"What is {subj}'s {prop}?",
        "possible_answers": [obj],
    }
    row.update(extra)
    return row


def test_clean_rows_drops_empty_and_strips():
    rows = [
        _row(subj="  Ada Lovelace  ", obj=" mathematician "),
        _row(subj=None),
        _row(subj=""),
        _row(prop="   "),
        _row(subj="B", obj=None),
        _row(subj="C", question=""),
        {"subj": "D", "prop": "occupation", "obj": "poet"},  # question missing
    ]
    cleaned = clean_rows(rows)
    assert len(cleaned) == 1
    assert cleaned[0]["subj"] == "Ada Lovelace"
    assert cleaned[0]["obj"] == "mathematician"
    assert "possible_answers" in cleaned[0]


def test_clean_rows_dedupes_normalized_key_keep_first():
    rows = [
        _row(subj="John Smith", prop="author", obj="first"),
        _row(subj="john  smith", prop="AUTHOR", obj="second"),
        _row(subj="John Smith", prop="director", obj="third"),
    ]
    cleaned = clean_rows(rows)
    assert [r["obj"] for r in cleaned] == ["first", "third"]


def test_cap_rows_caps_then_sorts_deterministically():
    rows = [_row(subj=f"S{i:02d}", prop=p, obj=f"o{i}")
            for i, p in enumerate(["b", "a", "c"] * 9)]
    before = [dict(r) for r in rows]
    capped = cap_rows(rows, n=10, seed=0)
    assert rows == before  # input not mutated
    assert len(capped) == 10
    assert capped == sorted(capped, key=lambda r: (r["prop"], r["subj"]))
    keys = {(r["subj"], r["prop"]) for r in rows}
    assert all((r["subj"], r["prop"]) in keys for r in capped)
    assert capped == cap_rows(rows, n=10, seed=0)
    assert capped != cap_rows(rows, n=10, seed=1)
    everything = cap_rows(rows, n=100, seed=0)
    assert everything == sorted(rows, key=lambda r: (r["prop"], r["subj"]))


def test_parse_possible_answers():
    assert parse_possible_answers('["a", "b"]', "obj") == ["a", "b"]
    assert parse_possible_answers("not json", "obj") == ["obj"]
    assert parse_possible_answers(None, "obj") == ["obj"]
    assert parse_possible_answers("[]", "obj") == ["obj"]
    assert parse_possible_answers('{"a": 1}', "obj") == ["obj"]
    assert parse_possible_answers(["x"], "obj") == ["x"]


# ------------------------------------------------------------- load and split


def test_load_realfacts_roundtrip(tmp_path):
    path = tmp_path / "popqa_clean.jsonl"
    with open(path, "w") as f:
        for fact in FACTS:
            f.write(json.dumps({
                "subj": fact.subj, "prop": fact.prop, "obj": fact.obj,
                "question": fact.question,
                "possible_answers": list(fact.possible_answers),
            }) + "\n")
    assert realfact.load_realfacts(path) == FACTS


def test_split_by_relation_80_20_disjoint_complete():
    seen, heldout = realfact.split_by_relation(FACTS, frac=0.8, seed=0)
    assert len(seen) + len(heldout) == N
    assert set(seen) | set(heldout) == set(FACTS)
    assert not set(seen) & set(heldout)
    for prop, n_prop in (("author", 10), ("capital of", 8), ("director", 5)):
        n_seen = sum(1 for f in seen if f.prop == prop)
        assert n_seen == int(0.8 * n_prop)
        assert sum(1 for f in heldout if f.prop == prop) == n_prop - n_seen


def test_split_by_relation_deterministic():
    assert (realfact.split_by_relation(FACTS, frac=0.8, seed=0)
            == realfact.split_by_relation(FACTS, frac=0.8, seed=0))


# ------------------------------------------------------------------ rendering


def test_trace_text_exact():
    fact = FACTS[0]
    doc = realfact.render_realfact_docs([fact], n_exposures=1)[0]
    assert doc.kind == "realfact"
    assert doc.split_text() == (
        f"Question: {fact.question}\n"
        f"Reasoning: The {fact.prop} of {fact.subj} is"
        f"{DB_START}{fact.subj}, {fact.prop}{DB_RETRIEVE} {fact.obj}{DB_END}"
        f". So the answer is {fact.obj}.\n"
        f"Answer: {fact.obj}"
    )
    assert doc.dense_text() == (
        f"Question: {fact.question}\n"
        f"Reasoning: The {fact.prop} of {fact.subj} is {fact.obj}."
        f" So the answer is {fact.obj}.\n"
        f"Answer: {fact.obj}"
    )


def test_masking_covers_exactly_the_value():
    docs = realfact.render_realfact_docs(FACTS, n_exposures=2)
    by_subj = {f.subj: f for f in FACTS}
    for doc in docs:
        fact = by_subj[doc.meta["subj"]]
        masked = [t for t, m in doc.split_segments if m]
        assert masked == [f" {fact.obj}"]
        assert (f"{fact.subj}, {fact.prop}", False) in doc.split_segments
        dense = doc.dense_text()
        assert DB_START not in dense and DB_RETRIEVE not in dense and DB_END not in dense
        assert f"\nAnswer: {fact.obj}" in dense
        assert f"\nAnswer: {fact.obj}" in doc.split_text()
        assert dense == _unwrap(doc.split_text())


def test_exposure_rounds_grouped_and_counted():
    docs = realfact.render_realfact_docs(FACTS, n_exposures=3)
    assert len(docs) == 3 * N
    for e in range(3):
        round_docs = docs[e * N:(e + 1) * N]
        assert [d.meta["subj"] for d in round_docs] == [f.subj for f in FACTS]
        assert all(d.meta["exposure"] == e for d in round_docs)
        assert all(d.meta["substituted"] is False for d in round_docs)
    assert docs == realfact.render_realfact_docs(FACTS, n_exposures=3)


# --------------------------------------------------------------- substitution


def test_substitution_full_swaps_question_and_key_together():
    docs = realfact.render_realfact_docs(FACTS, n_exposures=2, seed=0,
                                         substitution_frac=1.0)
    assert len(docs) == 2 * N
    by_subj = {f.subj: f for f in FACTS}
    subjects_by_prop: dict = {}
    for f in FACTS:
        subjects_by_prop.setdefault(f.prop, set()).add(f.subj)
    for doc in docs:
        fact = by_subj[doc.meta["subj"]]
        assert doc.meta["substituted"] is True
        key = _key_of(doc)
        assert key.endswith(f", {fact.prop}")
        name = key[: -len(f", {fact.prop}")]
        assert name in subjects_by_prop[fact.prop]
        question_line = doc.split_text().split("\n", 1)[0]
        assert question_line == f"Question: What is the {fact.prop} of {name}?"
        assert [t for t, m in doc.split_segments if m] == [f" {fact.obj}"]
        assert f"\nAnswer: {fact.obj}" in doc.dense_text()


def test_substitution_permutations_differ_across_exposures():
    docs = realfact.render_realfact_docs(FACTS, n_exposures=2, seed=0,
                                         substitution_frac=1.0)
    round0, round1 = docs[:N], docs[N:]
    assert any(_key_of(a) != _key_of(b) for a, b in zip(round0, round1))


def test_substitution_zero_and_determinism():
    plain = realfact.render_realfact_docs(FACTS, n_exposures=2, seed=0)
    zero = realfact.render_realfact_docs(FACTS, n_exposures=2, seed=0,
                                         substitution_frac=0.0)
    assert zero == plain
    full_a = realfact.render_realfact_docs(FACTS, n_exposures=2, seed=0,
                                           substitution_frac=1.0)
    full_b = realfact.render_realfact_docs(FACTS, n_exposures=2, seed=0,
                                           substitution_frac=1.0)
    assert full_a == full_b
    assert full_a != plain


def test_substitution_skips_when_subj_not_in_question():
    facts = [
        realfact.RealFact(subj="Hidden One", prop="author", obj="obj-a",
                          question="Who wrote the famous book?",
                          possible_answers=("obj-a",)),
        realfact.RealFact(subj="Visible Two", prop="author", obj="obj-b",
                          question="Who is the author of Visible Two?",
                          possible_answers=("obj-b",)),
    ]
    docs = realfact.render_realfact_docs(facts, n_exposures=1, seed=0,
                                         substitution_frac=1.0)
    hidden = docs[0]
    assert hidden.meta["substituted"] is False
    assert hidden.split_text().startswith("Question: Who wrote the famous book?\n")
    assert _key_of(hidden) == "Hidden One, author"


def test_substitution_respects_word_boundaries():
    # subj "Ann" must rewrite only the standalone "Ann", never the "Ann"
    # inside "Annals". With seed=0 the within-relation permutation maps
    # Ann -> Bob, so the standalone token becomes "Bob".
    facts = [
        realfact.RealFact(subj="Ann", prop="author", obj="obj-a",
                          question="Who wrote Annals about Ann?",
                          possible_answers=("obj-a",)),
        realfact.RealFact(subj="Bob", prop="author", obj="obj-b",
                          question="Who is the author of Bob?",
                          possible_answers=("obj-b",)),
    ]
    docs = realfact.render_realfact_docs(facts, n_exposures=1, seed=0,
                                         substitution_frac=1.0)
    ann = docs[0]
    assert ann.meta["substituted"] is True
    assert (ann.split_text().split("\n", 1)[0]
            == "Question: Who wrote Annals about Bob?")
    assert _key_of(ann) == "Bob, author"


def test_substitution_falls_back_when_no_word_boundary_match():
    # "Ann" appears only as a substring of "Annals" (no whole-word match),
    # so substitution must NOT fire and the original text is retained.
    facts = [
        realfact.RealFact(subj="Ann", prop="author", obj="obj-a",
                          question="Who wrote Annals?",
                          possible_answers=("obj-a",)),
        realfact.RealFact(subj="Bob", prop="author", obj="obj-b",
                          question="Who is the author of Bob?",
                          possible_answers=("obj-b",)),
    ]
    docs = realfact.render_realfact_docs(facts, n_exposures=1, seed=0,
                                         substitution_frac=1.0)
    ann = docs[0]
    assert ann.meta["substituted"] is False
    assert (ann.split_text().split("\n", 1)[0]
            == "Question: Who wrote Annals?")
    assert _key_of(ann) == "Ann, author"


# ------------------------------------------------------------------ flooding


def test_fresh_flood_appends_unique_fresh_names():
    k = 9
    docs = realfact.render_realfact_docs(FACTS, n_exposures=1, seed=0,
                                         fresh_flood=k)
    assert len(docs) == N + k
    flood = docs[N:]
    input_subjects = {f.subj for f in FACTS}
    objs_by_prop: dict = {}
    for f in FACTS:
        objs_by_prop.setdefault(f.prop, set()).add(f.obj)
    assert len({d.meta["subj"] for d in flood}) == k
    for doc in flood:
        name, prop = doc.meta["subj"], doc.meta["prop"]
        assert doc.meta.get("fresh") is True
        assert name not in input_subjects
        first, middle, last = name.split(" ")
        assert first in FIRST_NAMES and middle in MIDDLE_NAMES and last in LAST_NAMES
        assert prop in PROPS
        masked = [t for t, m in doc.split_segments if m]
        assert len(masked) == 1
        obj = masked[0][1:]
        assert obj in objs_by_prop[prop]
        assert doc.split_text().startswith(f"Question: What is the {prop} of {name}?\n")
        assert _key_of(doc) == f"{name}, {prop}"
    assert docs == realfact.render_realfact_docs(FACTS, n_exposures=1, seed=0,
                                                 fresh_flood=k)


# ---------------------------------------------------------- eval and organizer


def test_eval_items_fields():
    items = realfact.realfact_eval_items(FACTS, "heldout")
    assert len(items) == N
    for i, (item, fact) in enumerate(zip(items, FACTS)):
        assert item.qid == f"rf-heldout-{i}"
        assert item.task == "realfact"
        assert item.prompt == f"Question: {fact.question}\nReasoning:"
        assert item.answer == fact.obj
        assert item.meta["subj"] == fact.subj
        assert item.meta["prop"] == fact.prop
        assert item.meta["obj"] == fact.obj
        assert item.meta["possible_answers"] == list(fact.possible_answers)
        assert item.meta["split"] == "heldout"


def test_build_real_organizer_serves_every_fact():
    org = realfact.build_real_organizer(FACTS)
    assert len(org) == N
    for fact in FACTS:
        assert org.lookup(f"{fact.subj}, {fact.prop}") == fact.obj
