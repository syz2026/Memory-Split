"""Fact-use QA: questions whose chain-of-thought cites stored facts and then
reasons over them (extraction, date comparison, city equality).

In the split rendering, each fact's FIRST citation is a lookup call with the
value loss-masked; any restatement after retrieval (the comparison sentence,
the answer line) is plain text in both renderings — copying from context is
allowed and never rewards in-weight memorization.
"""

from __future__ import annotations

import datetime
import random

from corpusgen.bios import MONTH_NAMES, RELATION_PHRASES
from corpusgen.records import BioRecord, Doc, QAItem, Segment, lookup_segments

KIND_WEIGHTS = [("extract", 0.30), ("compare_date", 0.40), ("same_city", 0.30)]
SAME_CITY_YES_FRAC = 0.40


def parse_date(text: str) -> datetime.date:
    month_name, rest = text.split(" ", 1)
    day_str, year_str = rest.split(", ")
    return datetime.date(int(year_str), MONTH_NAMES.index(month_name) + 1, int(day_str))


class _SegBuilder:
    """Assembles the dense and split renderings in lockstep."""

    def __init__(self) -> None:
        self.dense_parts: list[str] = []
        self.split_segs: list[Segment] = []

    def plain(self, text: str) -> None:
        if not text:
            return
        self.dense_parts.append(text)
        if self.split_segs and self.split_segs[-1][1] is False:
            self.split_segs[-1] = (self.split_segs[-1][0] + text, False)
        else:
            self.split_segs.append((text, False))

    def fact(self, name: str, relation: str, value: str) -> None:
        """Cite a fact: plain value in dense; lookup-wrapped masked in split."""
        self.dense_parts.append(" " + value)
        for seg in lookup_segments(name, relation, value):
            self.split_segs.append(seg)

    def dense_text(self) -> str:
        return "".join(self.dense_parts)


def _question_and_cot(kind: str, a: BioRecord, b: BioRecord | None,
                      relation: str | None) -> tuple[str, _SegBuilder, str]:
    """Returns (question_line, builder holding 'Reasoning: ...' body, answer)."""
    sb = _SegBuilder()
    if kind == "extract":
        assert relation is not None
        value = a.attrs[relation]
        question = f"Question: What is {a.name}'s {RELATION_PHRASES[relation]}?"
        sb.plain(f"Reasoning: {a.name}'s {RELATION_PHRASES[relation]} is")
        sb.fact(a.name, relation, value)
        sb.plain(f". So the answer is {value}.")
        return question, sb, value
    if kind == "compare_date":
        assert b is not None
        da, db = parse_date(a.attrs["birth_date"]), parse_date(b.attrs["birth_date"])
        earlier = a if da < db else b
        d_early, d_late = sorted([a.attrs["birth_date"], b.attrs["birth_date"]],
                                 key=parse_date)
        question = f"Question: Who was born earlier, {a.name} or {b.name}?"
        sb.plain(f"Reasoning: {a.name} was born on")
        sb.fact(a.name, "birth_date", a.attrs["birth_date"])
        sb.plain(f". {b.name} was born on")
        sb.fact(b.name, "birth_date", b.attrs["birth_date"])
        sb.plain(f". {d_early} is earlier than {d_late}.")
        return question, sb, earlier.name
    if kind == "same_city":
        assert b is not None
        ca, cb = a.attrs["current_city"], b.attrs["birth_city"]
        same = ca == cb
        question = (f"Question: Is {a.name}'s current city the same as "
                    f"{b.name}'s birth city?")
        sb.plain(f"Reasoning: {a.name}'s current city is")
        sb.fact(a.name, "current_city", ca)
        sb.plain(f". {b.name}'s birth city is")
        sb.fact(b.name, "birth_city", cb)
        sb.plain(f". The two cities are {'the same' if same else 'different'}.")
        return question, sb, "yes" if same else "no"
    raise ValueError(kind)


def _draw(kind: str, records: list[BioRecord], rng: random.Random,
          birth_city_index: dict[str, list[BioRecord]]):
    """Returns (a, b, relation) for one item of the given kind."""
    if kind == "extract":
        a = records[rng.randrange(len(records))]
        relation = rng.choice(list(RELATION_PHRASES))
        return a, None, relation
    if kind == "compare_date":
        while True:
            a, b = rng.sample(records, 2)
            if a.attrs["birth_date"] != b.attrs["birth_date"]:  # ties resampled
                return a, b, None
    # same_city: construct yes-cases deliberately (random pairs match ~1/200)
    if rng.random() < SAME_CITY_YES_FRAC:
        for _ in range(50):
            a = records[rng.randrange(len(records))]
            matches = birth_city_index.get(a.attrs["current_city"], [])
            matches = [m for m in matches if m.entity_id != a.entity_id]
            if matches:
                return a, matches[rng.randrange(len(matches))], None
    while True:
        a, b = rng.sample(records, 2)
        if a.entity_id != b.entity_id:
            return a, b, None


def birth_city_index(records: list[BioRecord]) -> dict[str, list[BioRecord]]:
    """Precompute once and pass to the generators below when calling them
    repeatedly: rebuilding it per call is O(len(records)) and was the
    quadratic blowup that timed out the 4M-entity corpus builds."""
    index: dict[str, list[BioRecord]] = {}
    for rec in records:
        index.setdefault(rec.attrs["birth_city"], []).append(rec)
    return index


_birth_city_index = birth_city_index  # backward-compat alias


def generate_factqa_docs(records: list[BioRecord], n_docs: int, seed: int,
                         index: dict[str, list[BioRecord]] | None = None) -> list[Doc]:
    rng = random.Random(seed)
    index = index if index is not None else birth_city_index(records)
    kinds, weights = zip(*KIND_WEIGHTS)
    docs: list[Doc] = []
    for _ in range(n_docs):
        kind = rng.choices(kinds, weights)[0]
        a, b, relation = _draw(kind, records, rng, index)
        question, sb, answer = _question_and_cot(kind, a, b, relation)
        full = _SegBuilder()
        full.plain(question + "\n")
        full.dense_parts.extend(sb.dense_parts)
        # merge split segments preserving mask structure
        for seg in sb.split_segs:
            if seg[1] is False and full.split_segs and full.split_segs[-1][1] is False:
                full.split_segs[-1] = (full.split_segs[-1][0] + seg[0], False)
            else:
                full.split_segs.append(seg)
        full.plain(f"\nAnswer: {answer}")
        meta = {"kind": kind,
                "entities": [a.entity_id] if b is None else [a.entity_id, b.entity_id]}
        if relation is not None:
            meta["relation"] = relation
        docs.append(Doc(
            kind="factqa",
            dense_segments=[(full.dense_text(), False)],
            split_segments=full.split_segs,
            meta=meta,
        ))
    return docs


def _eval_items(records: list[BioRecord], n_items: int, seed: int,
                train_prompts: set[str] | None, kinds_weights, qid_prefix: str,
                extra_meta: dict | None = None,
                index: dict[str, list[BioRecord]] | None = None) -> list[QAItem]:
    rng = random.Random(seed)
    index = index if index is not None else birth_city_index(records)
    kinds, weights = zip(*kinds_weights)
    items: list[QAItem] = []
    seen_prompts: set[str] = set(train_prompts or set())
    attempts = 0
    max_attempts = max(2000, n_items * 200)
    while len(items) < n_items:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                f"could not draw {n_items} distinct eval prompts "
                f"(got {len(items)} after {attempts} attempts)"
            )
        kind = rng.choices(kinds, weights)[0]
        a, b, relation = _draw(kind, records, rng, index)
        question, _sb, answer = _question_and_cot(kind, a, b, relation)
        prompt = question + "\nReasoning:"
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        meta = {"kind": kind, "template": f"factqa-{kind}",
                "entities": [a.entity_id] if b is None else [a.entity_id, b.entity_id]}
        if relation is not None:
            meta["relation"] = relation
        if extra_meta:
            meta.update(extra_meta)
        items.append(QAItem(
            qid=f"{qid_prefix}-{kind}-{len(items)}" if qid_prefix == "factqa"
            else f"{qid_prefix}-{len(items)}",
            task="factqa",
            prompt=prompt,
            answer=answer,
            meta=meta,
        ))
    return items


def generate_factqa_eval(records: list[BioRecord], n_items: int, seed: int,
                         train_prompts: set[str] | None = None,
                         index: dict[str, list[BioRecord]] | None = None) -> list[QAItem]:
    return _eval_items(records, n_items, seed, train_prompts, KIND_WEIGHTS,
                       "factqa", index=index)


def generate_fresh_entity_eval(fresh_records: list[BioRecord], n_items: int,
                               seed: int) -> list[QAItem]:
    """Extraction QA over entities present ONLY in the organizer (never in
    training text): measures the split arm's lookup-skill generalization."""
    return _eval_items(fresh_records, n_items, seed, None,
                       [("extract", 1.0)], "factqa-fresh", {"fresh": True})
