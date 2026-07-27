"""NR-4 — extractability-vs-capacity probe (no retraining).

Separates "fact not stored" from "fact stored but not generatively extractable"
(Physics-of-LM 3.1) via a multiple-choice RECOGNITION probe on the same
checkpoint: present the true attribute value plus pool-matched distractors and
score by choice log-likelihood. Recognition >> chance while generative recall
≈ 0 ⇒ the fact is in the weights but not extractable, which means
recall-derived bits accounting (H3) understates stored knowledge (ledger L11).

New module; read-only imports from corpusgen.bios and evals.natural.
"""

from __future__ import annotations

import datetime
import random

from corpusgen.bios import (
    BIRTH_DATE_MAX,
    BIRTH_DATE_MIN,
    RELATION_PHRASES,
    VALUE_POOLS,
    format_date,
)
from corpusgen.records import ATTRIBUTES
from evals.natural import loglikelihood_choice_scores

_N_DAYS = (BIRTH_DATE_MAX - BIRTH_DATE_MIN).days + 1


def _distractors(attr: str, gold: str, k: int, rng: random.Random) -> list[str]:
    """k distinct pool-matched values != gold for one attribute."""
    if attr == "birth_date":
        out: set[str] = set()
        while len(out) < k:
            d = format_date(BIRTH_DATE_MIN + datetime.timedelta(days=rng.randrange(_N_DAYS)))
            if d != gold:
                out.add(d)
        return list(out)
    pool = [v for v in VALUE_POOLS[attr] if v != gold]
    return rng.sample(pool, min(k, len(pool)))


def make_mc_recall_items(
    records,
    n_items: int,
    seed: int,
    n_choices: int = 4,
    attributes=ATTRIBUTES,
) -> list[dict]:
    """Multiple-choice recognition probes over (entity, relation).

    Each item: {qid, entity_id, relation, context, choices, gold_index}. context
    is the recall-probe prompt; choices are " {value}" strings (gold + pool
    distractors), shuffled. Deterministic in (records, seed). birth_date has no
    fixed pool, so its distractors are random in-range dates.
    """
    if n_choices < 2:
        raise ValueError("n_choices must be >= 2")
    rng = random.Random(seed)
    attrs = [a for a in attributes if a == "birth_date" or a in VALUE_POOLS]
    items: list[dict] = []
    for _ in range(n_items):
        rec = records[rng.randrange(len(records))]
        attr = attrs[rng.randrange(len(attrs))]
        gold = rec.attrs[attr]
        distract = _distractors(attr, gold, n_choices - 1, rng)
        values = [gold] + distract
        rng.shuffle(values)
        gold_index = values.index(gold)
        items.append({
            "qid": f"mc-{rec.entity_id}-{attr}-{len(items)}",
            "entity_id": rec.entity_id,
            "relation": attr,
            "context": f"{rec.name}'s {RELATION_PHRASES[attr]} is",
            "choices": [" " + v for v in values],
            "gold_index": gold_index,
        })
    return items


def mc_recall_accuracy(model, tok, items, device, score: str = "mean") -> dict:
    """Choice-log-likelihood recognition accuracy over MC recall items.

    score="mean" (default) ranks by per-token logprob (length-normalized, i.e.
    acc_norm) — the fair choice here because pool values differ in token length
    and raw-sum logprob is length-biased. score="sum" ranks by total logprob.
    ``correct_prob`` (softmax mass on the gold choice) uses the SAME score.
    Returns {overall, per_attribute, correct_prob, n, n_choices, chance, score}.
    """
    if score not in ("sum", "mean"):
        raise ValueError("score must be 'sum' or 'mean'")
    idx = 0 if score == "sum" else 1
    n_correct = 0
    prob_mass = 0.0
    per_attr: dict[str, list[int]] = {}
    n_choices_seen = 0
    import math

    for it in items:
        scores = loglikelihood_choice_scores(model, tok, it["context"], it["choices"], device)
        vals = [s[idx] for s in scores]
        pred = max(range(len(vals)), key=vals.__getitem__)
        gold = it["gold_index"]
        correct = int(pred == gold)
        n_correct += correct
        n_choices_seen = max(n_choices_seen, len(vals))
        # softmax mass on the gold choice, over the SAME per-choice score
        mx = max(vals)
        exps = [math.exp(v - mx) for v in vals]
        prob_mass += exps[gold] / sum(exps)
        ht = per_attr.setdefault(it["relation"], [0, 0])
        ht[0] += correct
        ht[1] += 1

    n = len(items)
    return {
        "overall": n_correct / n if n else 0.0,
        "per_attribute": {a: c / t for a, (c, t) in per_attr.items()},
        "correct_prob": prob_mass / n if n else 0.0,
        "n": n,
        "n_choices": n_choices_seen,
        "chance": 1.0 / n_choices_seen if n_choices_seen else 0.0,
        "score": score,
    }
