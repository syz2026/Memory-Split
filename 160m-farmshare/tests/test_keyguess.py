"""Tests for evals.keyguess: parsing emitted organizer keys, decomposing them
into name/relation halves, and scoring whether the model constructed the
correct key for a held-out fact (per the 2026-07-20 protocol)."""

import pytest

from evals.keyguess import (
    parse_emitted_key,
    score_item,
    score_items,
    split_key,
    wilson,
)
from corpusgen.records import DB_END, DB_RETRIEVE, DB_START, QAItem


# ----------------------------------------------------------------- parse_emitted_key


def test_parse_emitted_key_hit():
    text = f"prefix{DB_START}The Ritual, author{DB_RETRIEVE} Morgan{DB_END}suffix"
    out = parse_emitted_key(text)
    assert out == {"status": "ok", "key": "The Ritual, author"}


def test_parse_emitted_key_no_lookup():
    assert parse_emitted_key("just rambling, no lookup here") == {"status": "no_lookup"}


def test_parse_emitted_key_malformed():
    assert parse_emitted_key(f"{DB_START}The Ritual, author with no close") == {
        "status": "malformed"
    }


def test_parse_emitted_key_first_of_two():
    text = (
        f"{DB_START}First Key, rel{DB_RETRIEVE} v1{DB_END}"
        f"gap{DB_START}Second Key, rel{DB_RETRIEVE} v2{DB_END}"
    )
    assert parse_emitted_key(text) == {"status": "ok", "key": "First Key, rel"}


# ----------------------------------------------------------------------- split_key


def test_split_key_multi_comma():
    assert split_key("Powell, Ohio, capital of") == ("Powell, Ohio", "capital of")


def test_split_key_no_comma():
    assert split_key("no comma here") == ("no comma here", "")


# ------------------------------------------------------------------------ score_item


def _meta(subj="The Ritual", prop="author", obj="Morgan", split="held-out",
          possible_answers=None):
    return {
        "subj": subj,
        "prop": prop,
        "obj": obj,
        "split": split,
        "possible_answers": possible_answers if possible_answers is not None else [obj],
    }


def test_score_item_exact_key():
    prompt = "Who wrote The Ritual? Reasoning:"
    text = f"{DB_START}The Ritual, author{DB_RETRIEVE} Morgan{DB_END}"
    r = score_item(_meta(), prompt, text)
    assert r["status"] == "ok"
    assert r["emitted_key"] == "The Ritual, author"
    assert r["name_ok"] is True
    assert r["rel_ok"] is True
    assert r["full_ok"] is True
    assert r["answer_ok"] is True
    assert r["name_error_class"] is None


def test_score_item_right_name_wrong_relation():
    prompt = "Who wrote The Ritual? Reasoning:"
    text = f"{DB_START}The Ritual, occupation{DB_RETRIEVE} Morgan{DB_END}"
    r = score_item(_meta(), prompt, text)
    assert r["name_ok"] is True
    assert r["rel_ok"] is False
    assert r["full_ok"] is False
    assert r["name_error_class"] is None


def test_score_item_wrong_name_in_context():
    prompt = "Who wrote The Celebrated Jumping Frog? Reasoning:"
    text = f"{DB_START}The Celebrated Jumping Frog, author{DB_RETRIEVE} Morgan{DB_END}"
    r = score_item(_meta(), prompt, text)
    assert r["name_ok"] is False
    assert r["rel_ok"] is True
    assert r["full_ok"] is False
    assert r["name_error_class"] == "in_context_wrong_name"


def test_score_item_wrong_name_out_of_context():
    prompt = "Who wrote The Ritual? Reasoning:"
    text = f"{DB_START}Beselle Quina Sandborn, author{DB_RETRIEVE} Morgan{DB_END}"
    r = score_item(_meta(), prompt, text)
    assert r["name_ok"] is False
    assert r["full_ok"] is False
    assert r["name_error_class"] == "out_of_context_name"


def test_score_item_case_whitespace_insensitive():
    prompt = "  who wrote   the ritual?  Reasoning:"
    text = f"{DB_START}  the ritual, Author   {DB_RETRIEVE} morgan {DB_END}"
    r = score_item(_meta(), prompt, text)
    assert r["name_ok"] is True
    assert r["rel_ok"] is True
    assert r["full_ok"] is True
    assert r["answer_ok"] is True


def test_score_item_space_before_comma_is_not_a_match():
    # normalize collapses whitespace runs but does NOT strip spaces around the
    # comma, so "name , rel" is a different key than "name, rel" (organizer
    # exact-match semantics): full_ok is False even though both halves match.
    prompt = "Who wrote The Ritual? Reasoning:"
    text = f"{DB_START}The Ritual , author{DB_RETRIEVE} Morgan{DB_END}"
    r = score_item(_meta(), prompt, text)
    assert r["name_ok"] is True
    assert r["rel_ok"] is True
    assert r["full_ok"] is False


def test_score_item_answer_uses_possible_answers_fallback_obj():
    prompt = "Who wrote The Ritual? Reasoning:"
    # wrong key but correct answer appears in text; possible_answers empty -> fallback [obj]
    text = f"{DB_START}Wrong, author{DB_RETRIEVE} Morgan{DB_END}"
    r = score_item(_meta(possible_answers=[]), prompt, text)
    assert r["answer_ok"] is True


def test_score_item_answer_substring_any_possible_answer():
    prompt = "Who wrote The Ritual? Reasoning:"
    text = f"{DB_START}The Ritual, author{DB_RETRIEVE} MORGAN the great{DB_END}"
    r = score_item(_meta(possible_answers=["Morgan", "Twain"]), prompt, text)
    assert r["answer_ok"] is True


def test_score_item_no_lookup_status():
    r = score_item(_meta(), "prompt", "no lookup at all")
    assert r["status"] == "no_lookup"
    assert r["name_ok"] is False
    assert r["rel_ok"] is False
    assert r["full_ok"] is False
    assert r["emitted_key"] is None
    assert r["name_error_class"] == "out_of_context_name"


def test_score_item_malformed_status():
    r = score_item(_meta(), "prompt", f"{DB_START}dangling")
    assert r["status"] == "malformed"
    assert r["full_ok"] is False
    assert r["name_error_class"] == "out_of_context_name"


# ----------------------------------------------------------------------- score_items


def _qa(qid, subj, prop, obj, split, prompt, possible_answers=None):
    return QAItem(
        qid=qid,
        task="recall",
        prompt=prompt,
        answer=obj,
        meta={
            "subj": subj,
            "prop": prop,
            "obj": obj,
            "split": split,
            "possible_answers": possible_answers if possible_answers is not None else [obj],
        },
    )


def test_score_items_aggregation_and_splits():
    # 4 items: 2 seen, 2 held-out.
    # seen: 1 full-ok, 1 wrong-name-in-context.
    # held-out: 1 no_lookup, 1 exact-ok.
    items = [
        _qa("s1", "The Ritual", "author", "Morgan", "seen",
            "Who wrote The Ritual? Reasoning:"),
        _qa("s2", "The Ritual", "author", "Morgan", "seen",
            "Who wrote The Celebrated Jumping Frog? Reasoning:"),
        _qa("h1", "The Ritual", "author", "Morgan", "held-out",
            "Who wrote The Ritual? Reasoning:"),
        _qa("h2", "The Ritual", "author", "Morgan", "held-out",
            "Who wrote The Ritual? Reasoning:"),
    ]
    texts = [
        f"{DB_START}The Ritual, author{DB_RETRIEVE} Morgan{DB_END}",
        f"{DB_START}The Celebrated Jumping Frog, author{DB_RETRIEVE} Morgan{DB_END}",
        "no lookup here",
        f"{DB_START}The Ritual, author{DB_RETRIEVE} Morgan{DB_END}",
    ]
    out = score_items(items, texts)
    assert set(out.keys()) == {"seen", "held-out", "all", "records"}
    assert len(out["records"]) == 4
    assert [r["qid"] for r in out["records"]] == ["s1", "s2", "h1", "h2"]

    seen = out["seen"]
    assert seen["n"] == 2
    assert seen["full_key"] == 0.5
    assert seen["name_half"] == 0.5          # s1 name ok, s2 name wrong
    assert seen["relation_half"] == 1.0      # both relations ok
    assert seen["no_lookup_rate"] == 0.0
    assert seen["malformed_rate"] == 0.0
    assert seen["wrong_in_context_rate"] == 0.5

    held = out["held-out"]
    assert held["n"] == 2
    assert held["full_key"] == 0.5            # h1 no_lookup (fail), h2 ok
    assert held["name_half"] == 0.5
    assert held["relation_half"] == 0.5      # h1 no_lookup counts as fail
    assert held["no_lookup_rate"] == 0.5
    assert held["malformed_rate"] == 0.0
    assert held["wrong_in_context_rate"] == 0.0

    allg = out["all"]
    assert allg["n"] == 4
    assert allg["full_key"] == 0.5
    assert allg["no_lookup_rate"] == 0.25
    assert allg["wrong_in_context_rate"] == 0.25


def test_score_items_accepts_dicts_and_missing_qid():
    items = [
        {"meta": _meta(subj="The Ritual", prop="author", obj="Morgan",
                       split="seen"), "prompt": "p"},
    ]
    texts = [f"{DB_START}The Ritual, author{DB_RETRIEVE} Morgan{DB_END}"]
    out = score_items(items, texts)
    assert out["seen"]["full_key"] == 1.0
    assert out["records"][0].get("qid") is None


def test_score_items_wilson_ci_keys_present():
    items = [_qa("q", "n", "r", "v", "seen", "p") for _ in range(4)]
    texts = [f"{DB_START}n, r{DB_RETRIEVE} v{DB_END}"] * 4
    out = score_items(items, texts)
    for metric in ("full_key", "name_half", "relation_half", "answer",
                   "no_lookup_rate", "malformed_rate", "wrong_in_context_rate"):
        assert f"{metric}_ci" in out["seen"]
        lo, hi = out["seen"][f"{metric}_ci"]
        assert 0.0 <= lo <= hi <= 1.0


# --------------------------------------------------------------------------- wilson


def test_wilson_zero_n():
    assert wilson(0, 0) == (0.0, 0.0)


def test_wilson_k15_n601_matches_protocol():
    lo, hi = wilson(15, 601)
    assert lo == pytest.approx(0.015, abs=0.003)
    assert hi == pytest.approx(0.041, abs=0.003)
    assert lo < hi


def test_wilson_all_success_bounds():
    lo, hi = wilson(10, 10)
    assert 0.0 <= lo <= 1.0
    assert hi == pytest.approx(1.0, abs=1e-9)
