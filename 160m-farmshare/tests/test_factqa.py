"""Tests for corpusgen.factqa (Task 2) — written before the implementation."""

import datetime
import re

import pytest

from corpusgen import bios, factqa
from corpusgen.records import ATTRIBUTES, DB_END

_WRAP = re.compile(r"<\|db_start\|>.*?<\|db_retrieve\|>", re.S)


def _unwrap(split_text: str) -> str:
    return _WRAP.sub("", split_text).replace(DB_END, "")


RECORDS = bios.generate_records(300, seed=77)
BY_ID = {r.entity_id: r for r in RECORDS}
DOCS = factqa.generate_factqa_docs(RECORDS, 600, seed=101)


def _answer_of(doc) -> str:
    return doc.dense_text().rsplit("\nAnswer: ", 1)[1]


def _docs_of_kind(kind):
    return [d for d in DOCS if d.meta["kind"] == kind]


# ---------------------------------------------------------------- basics


def test_docs_deterministic():
    again = factqa.generate_factqa_docs(RECORDS, 600, seed=101)
    assert again == DOCS
    other = factqa.generate_factqa_docs(RECORDS, 600, seed=102)
    assert other != DOCS


def test_doc_text_format_and_kind():
    fmt = re.compile(r"^Question: .+\nReasoning: .+\nAnswer: .+$", re.S)
    for doc in DOCS:
        assert doc.kind == "factqa"
        assert fmt.match(doc.dense_text()), doc.dense_text()
        assert doc.meta["kind"] in ("extract", "compare_date", "same_city")
        assert doc.meta["entities"]


def test_kind_mix_roughly_30_40_30():
    n = len(DOCS)
    frac = {
        k: len(_docs_of_kind(k)) / n
        for k in ("extract", "compare_date", "same_city")
    }
    assert abs(frac["extract"] - 0.30) < 0.08
    assert abs(frac["compare_date"] - 0.40) < 0.08
    assert abs(frac["same_city"] - 0.30) < 0.08


def test_dense_equals_split_unwrapped():
    for doc in DOCS:
        assert doc.dense_text() == _unwrap(doc.split_text())


# ---------------------------------------------------------------- parse_date


def test_parse_date():
    assert factqa.parse_date("July 4, 1985") == datetime.date(1985, 7, 4)
    assert factqa.parse_date("December 31, 2005") == datetime.date(2005, 12, 31)
    assert factqa.parse_date("January 1, 1930") == datetime.date(1930, 1, 1)
    for rec in RECORDS[:40]:
        d = factqa.parse_date(rec.attrs["birth_date"])
        assert bios.BIRTH_DATE_MIN <= d <= bios.BIRTH_DATE_MAX


# ---------------------------------------------------------------- ground truth


def test_extract_answers_equal_record_values():
    for doc in _docs_of_kind("extract"):
        rec = BY_ID[doc.meta["entities"][0]]
        attr = doc.meta["relation"]
        assert _answer_of(doc) == rec.attrs[attr]
        assert f"What is {rec.name}'s {bios.RELATION_PHRASES[attr]}?" in doc.dense_text()


def test_compare_date_ground_truth_200_items():
    docs = _docs_of_kind("compare_date")
    assert len(docs) >= 200
    for doc in docs[:250]:
        a, b = (BY_ID[e] for e in doc.meta["entities"])
        da = factqa.parse_date(a.attrs["birth_date"])
        db = factqa.parse_date(b.attrs["birth_date"])
        assert da != db  # ties resampled away
        earlier = a if da < db else b
        assert _answer_of(doc) == earlier.name


def test_compare_date_answer_and_dates_plain_in_split():
    for doc in _docs_of_kind("compare_date")[:100]:
        a, b = (BY_ID[e] for e in doc.meta["entities"])
        unmasked = "".join(t for t, m in doc.split_segments if not m)
        # the comparison sentence restates both dates outside any lookup wrap
        assert a.attrs["birth_date"] in unmasked
        assert b.attrs["birth_date"] in unmasked
        # answer (a derived conclusion) is plain text in both renderings
        assert f"\nAnswer: {_answer_of(doc)}" in unmasked


def test_same_city_ground_truth_and_both_classes():
    docs = _docs_of_kind("same_city")
    answers = []
    for doc in docs:
        a, b = (BY_ID[e] for e in doc.meta["entities"])
        want = "yes" if a.attrs["current_city"] == b.attrs["birth_city"] else "no"
        got = _answer_of(doc)
        assert got == want
        answers.append(got)
    frac_yes = answers.count("yes") / len(answers)
    assert frac_yes >= 0.05
    assert 1 - frac_yes >= 0.30


# ---------------------------------------------------------------- split masking


def test_masked_segments_are_exactly_cited_values():
    for doc in DOCS:
        a = BY_ID[doc.meta["entities"][0]]
        b = BY_ID[doc.meta["entities"][-1]]
        masked = [t for t, m in doc.split_segments if m]
        kind = doc.meta["kind"]
        if kind == "extract":
            assert masked == [" " + a.attrs[doc.meta["relation"]]]
        elif kind == "compare_date":
            assert masked == [" " + a.attrs["birth_date"], " " + b.attrs["birth_date"]]
        else:
            assert masked == [" " + a.attrs["current_city"], " " + b.attrs["birth_city"]]


def test_lookup_queries_use_canonical_name_and_raw_key():
    for doc in DOCS[:100]:
        a = BY_ID[doc.meta["entities"][0]]
        b = BY_ID[doc.meta["entities"][-1]]
        split = doc.split_text()
        kind = doc.meta["kind"]
        if kind == "extract":
            assert f"<|db_start|>{a.name}, {doc.meta['relation']}<|db_retrieve|>" in split
        elif kind == "compare_date":
            assert f"<|db_start|>{a.name}, birth_date<|db_retrieve|>" in split
            assert f"<|db_start|>{b.name}, birth_date<|db_retrieve|>" in split
        else:
            assert f"<|db_start|>{a.name}, current_city<|db_retrieve|>" in split
            assert f"<|db_start|>{b.name}, birth_city<|db_retrieve|>" in split


# ---------------------------------------------------------------- eval items


def test_eval_items_fields_and_determinism():
    items = factqa.generate_factqa_eval(RECORDS, 120, seed=5)
    assert items == factqa.generate_factqa_eval(RECORDS, 120, seed=5)
    assert len(items) == 120
    for it in items:
        assert it.task == "factqa"
        assert it.prompt.startswith("Question: ")
        assert it.prompt.endswith("\nReasoning:")
        kind = it.meta["kind"]
        assert it.meta["template"] == f"factqa-{kind}"
        assert it.qid.startswith(f"factqa-{kind}-")
        a = BY_ID[it.meta["entities"][0]]
        b = BY_ID[it.meta["entities"][-1]]
        if kind == "extract":
            assert it.answer == a.attrs[it.meta["relation"]]
        elif kind == "compare_date":
            da = factqa.parse_date(a.attrs["birth_date"])
            db = factqa.parse_date(b.attrs["birth_date"])
            assert it.answer == (a if da < db else b).name
        else:
            same = a.attrs["current_city"] == b.attrs["birth_city"]
            assert it.answer == ("yes" if same else "no")


def test_eval_disjoint_from_train_prompts():
    train_prompts = {
        d.dense_text().split("\nReasoning:", 1)[0] + "\nReasoning:" for d in DOCS
    }
    # same base seed as the training docs, to stress the disjointness machinery
    items = factqa.generate_factqa_eval(RECORDS, 100, seed=101, train_prompts=train_prompts)
    prompts = [it.prompt for it in items]
    assert len(set(prompts)) == 100  # unique within the eval set too
    assert not (set(prompts) & train_prompts)


def test_eval_exhaustion_raises():
    two = RECORDS[:2]
    with pytest.raises(ValueError):
        # only a handful of distinct prompts exist over 2 records; 500 can't be drawn
        factqa.generate_factqa_eval(two, 500, seed=1)


# ---------------------------------------------------------------- fresh entities


def test_fresh_entity_eval():
    fresh = bios.generate_records(340, seed=77)[300:]
    fresh_names = {r.name for r in fresh}
    assert not (fresh_names & {r.name for r in RECORDS})  # prefix-stable generation
    by_id = {r.entity_id: r for r in fresh}
    items = factqa.generate_fresh_entity_eval(fresh, 50, seed=3)
    assert items == factqa.generate_fresh_entity_eval(fresh, 50, seed=3)
    assert len(items) == 50
    for it in items:
        assert it.task == "factqa"
        assert it.meta["kind"] == "extract"
        assert it.meta["fresh"] is True
        assert it.qid.startswith("factqa-fresh-")
        assert it.prompt.endswith("\nReasoning:")
        rec = by_id[it.meta["entities"][0]]
        assert it.answer == rec.attrs[it.meta["relation"]]
        assert rec.name in it.prompt
