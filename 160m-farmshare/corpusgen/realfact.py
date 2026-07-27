"""Real-fact traces (PopQA): the held-out key-construction testbed.

Facts are (subj, prop) -> obj Wikidata triples frozen to JSONL by
scripts/fetch_realfacts.py. Each fact renders as a Question/Reasoning/Answer
trace with the value loss-masked inside an organizer lookup, exactly like
factqa: the model must learn to emit the key "{subj}, {prop}", never to
produce the value. Two training-side fixes for the name-copy failure
(protocol 2026-07-20, where memorized keys beat prompt-copying) live here:

- counterfactual substitution: with probability p per (fact, exposure), the
  subject in BOTH the question and the lookup key is swapped to another
  same-relation subject while the retrieved value stays the original fact's,
  so key memorization stops being loss-equivalent to copying the prompt name.
- fresh-name flooding: extra single-exposure traces whose subjects are
  synthetic names appearing nowhere else, trainable only by name-copy.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from corpusgen.bios import FIRST_NAMES, LAST_NAMES, MIDDLE_NAMES
from corpusgen.factqa import _SegBuilder
from corpusgen.records import Doc, QAItem
from organizer.store import Organizer


@dataclass(frozen=True)
class RealFact:
    subj: str
    prop: str
    obj: str
    question: str
    possible_answers: tuple[str, ...]


def load_realfacts(path: str | Path) -> list[RealFact]:
    facts: list[RealFact] = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            facts.append(RealFact(
                subj=row["subj"],
                prop=row["prop"],
                obj=row["obj"],
                question=row["question"],
                possible_answers=tuple(row["possible_answers"]),
            ))
    return facts


def split_by_relation(facts: list[RealFact], frac: float = 0.8,
                      seed: int = 0) -> tuple[list[RealFact], list[RealFact]]:
    """Per relation: seeded shuffle, first floor(frac*n) facts -> seen, rest
    -> heldout. Per-relation string seeding keeps each relation's split
    independent of the others and of input grouping order."""
    by_prop: dict[str, list[RealFact]] = {}
    for fact in facts:
        by_prop.setdefault(fact.prop, []).append(fact)
    seen: list[RealFact] = []
    heldout: list[RealFact] = []
    for prop in sorted(by_prop):
        group = list(by_prop[prop])
        random.Random(f"{seed}:split:{prop}").shuffle(group)
        cut = int(frac * len(group))
        seen.extend(group[:cut])
        heldout.extend(group[cut:])
    return seen, heldout


def _render_trace(question: str, name: str, prop: str, obj: str, meta: dict) -> Doc:
    sb = _SegBuilder()
    sb.plain(f"Question: {question}\n")
    sb.plain(f"Reasoning: The {prop} of {name} is")
    sb.fact(name, prop, obj)
    sb.plain(f". So the answer is {obj}.")
    sb.plain(f"\nAnswer: {obj}")
    return Doc(
        kind="realfact",
        dense_segments=[(sb.dense_text(), False)],
        split_segments=sb.split_segs,
        meta=meta,
    )


def _substitution_partners(facts: list[RealFact], seed: int,
                           exposure: int) -> dict[int, str]:
    """Fact index -> swap subject: a seeded permutation WITHIN each relation,
    fresh per exposure round (identity swaps allowed by chance)."""
    by_prop: dict[str, list[int]] = {}
    for i, fact in enumerate(facts):
        by_prop.setdefault(fact.prop, []).append(i)
    partners: dict[int, str] = {}
    for prop, indices in by_prop.items():
        permuted = list(indices)
        random.Random(f"{seed}:sub:{exposure}:{prop}").shuffle(permuted)
        for i, j in zip(indices, permuted):
            partners[i] = facts[j].subj
    return partners


def _fresh_flood_docs(facts: list[RealFact], k: int, seed: int) -> list[Doc]:
    """k single-exposure traces about never-repeated synthetic subjects; the
    relation and (deliberately mismatched) value come from the real facts."""
    if k <= 0:
        return []
    rng = random.Random(f"{seed}:flood")
    props = sorted({f.prop for f in facts})
    objs_by_prop: dict[str, list[str]] = {}
    for fact in facts:
        objs_by_prop.setdefault(fact.prop, []).append(fact.obj)
    used = {f.subj for f in facts}
    docs: list[Doc] = []
    for _ in range(k):
        while True:
            name = (f"{rng.choice(FIRST_NAMES)} {rng.choice(MIDDLE_NAMES)} "
                    f"{rng.choice(LAST_NAMES)}")
            if name not in used:
                break
        used.add(name)
        prop = rng.choice(props)
        obj = rng.choice(objs_by_prop[prop])
        docs.append(_render_trace(
            question=f"What is the {prop} of {name}?",
            name=name, prop=prop, obj=obj,
            meta={"subj": name, "prop": prop, "exposure": 0,
                  "substituted": False, "fresh": True},
        ))
    return docs


def render_realfact_docs(facts: list[RealFact], n_exposures: int = 6,
                         seed: int = 0, substitution_frac: float = 0.0,
                         fresh_flood: int = 0) -> list[Doc]:
    """One doc per (fact, exposure), emitted in grouped rounds (all facts'
    exposure-0 docs, then exposure-1, ...; input order within a round), plus
    fresh_flood extra docs appended at the end.

    A substituted doc swaps the subject in the question AND the key to the
    fact's permuted same-relation partner while the value/answer stay the
    original fact's obj; if the subject is not a verbatim substring of the
    question the doc renders unsubstituted (meta["substituted"] stays False).
    """
    docs: list[Doc] = []
    for exposure in range(n_exposures):
        partners = (_substitution_partners(facts, seed, exposure)
                    if substitution_frac > 0 else {})
        for i, fact in enumerate(facts):
            name, question, substituted = fact.subj, fact.question, False
            if substitution_frac > 0:
                subj_re = re.compile(rf"(?<!\w){re.escape(fact.subj)}(?!\w)")
                if (random.Random(f"{seed}:{i}:{exposure}").random()
                        < substitution_frac
                        and subj_re.search(fact.question)):
                    name = partners[i]
                    question = subj_re.sub(name, fact.question)
                    substituted = True
            docs.append(_render_trace(
                question=question, name=name, prop=fact.prop, obj=fact.obj,
                meta={"subj": fact.subj, "prop": fact.prop,
                      "exposure": exposure, "substituted": substituted},
            ))
    docs.extend(_fresh_flood_docs(facts, fresh_flood, seed))
    return docs


def realfact_eval_items(facts: list[RealFact], split_label: str) -> list[QAItem]:
    return [
        QAItem(
            qid=f"rf-{split_label}-{i}",
            task="realfact",
            prompt=f"Question: {fact.question}\nReasoning:",
            answer=fact.obj,
            meta={"subj": fact.subj, "prop": fact.prop, "obj": fact.obj,
                  "possible_answers": list(fact.possible_answers),
                  "split": split_label},
        )
        for i, fact in enumerate(facts)
    ]


def build_real_organizer(facts: list[RealFact]) -> Organizer:
    org = Organizer()
    for fact in facts:
        org.add(fact.subj, fact.prop, fact.obj)
    return org
