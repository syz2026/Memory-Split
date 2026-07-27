"""Key-guess metric: did the model *construct* the right organizer key for a
fact it never trained on?

MemorySplit models emit store lookups as ``<|db_start|>{name}, {relation}
<|db_retrieve|>`` inside generated text; the organizer is exact-match on
``normalize("{name}, {relation}")``, so a held-out fact is retrievable only if
the model decodes that exact string. This module captures the emitted query
and scores it against the gold key, decomposed into a name-half and a
relation-half, plus an answer-substring check and an in/out-of-context
classification of name errors (per the 2026-07-20 held-out-key protocol).

Rates count successes over ALL items in a group: a ``no_lookup`` item counts
as a failure for full/name/relation, matching the lost run (which reported
"~10% no_lookup" alongside a 2.5% name-half).
"""

from __future__ import annotations

import math

from corpusgen.records import DB_RETRIEVE, DB_START
from organizer.store import normalize

_Z_95 = 1.96


def wilson(k: int, n: int, z: float = _Z_95) -> tuple[float, float]:
    """Standard Wilson score interval for a binomial proportion.

    Returns ``(0.0, 0.0)`` when ``n == 0``; otherwise the 95% (by default)
    interval, clamped to ``[0, 1]``.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)


def parse_emitted_key(text: str) -> dict:
    """Extract the first emitted organizer query from generated text.

    Looks for the first ``<|db_start|>`` and the next ``<|db_retrieve|>``
    after it. Returns one of:

    - ``{"status": "ok", "key": <stripped query>}``
    - ``{"status": "no_lookup"}``  (no ``<|db_start|>`` anywhere)
    - ``{"status": "malformed"}`` (``<|db_start|>`` present but no closing
      ``<|db_retrieve|>`` follows it)
    """
    start = text.find(DB_START)
    if start == -1:
        return {"status": "no_lookup"}
    inner_lo = start + len(DB_START)
    retrieve = text.find(DB_RETRIEVE, inner_lo)
    if retrieve == -1:
        return {"status": "malformed"}
    return {"status": "ok", "key": text[inner_lo:retrieve].strip()}


def split_key(key: str) -> tuple[str, str]:
    """Split an emitted key into ``(name_half, relation_half)`` on the LAST
    ``", "``. A key with no ``", "`` yields ``(key, "")`` so a degenerate
    key still has a name-half to classify."""
    idx = key.rfind(", ")
    if idx == -1:
        return (key, "")
    return (key[:idx], key[idx + 2 :])


def score_item(item_meta: dict, prompt: str, text: str) -> dict:
    """Score one generated text against the gold key for a fact.

    ``item_meta`` carries ``subj``, ``prop``, ``obj``, ``possible_answers``
    (list; falls back to ``[obj]``), and ``split``. All comparisons use
    organizer ``normalize`` (whitespace-collapse + lowercase) on both sides.
    """
    subj = item_meta["subj"]
    prop = item_meta["prop"]
    obj = item_meta["obj"]
    answers = item_meta.get("possible_answers") or [obj]

    parsed = parse_emitted_key(text)
    status = parsed["status"]
    emitted_key = parsed["key"] if status == "ok" else None

    gold_key = normalize(f"{subj}, {prop}")
    gold_name = normalize(subj)
    gold_rel = normalize(prop)

    if status == "ok":
        name_half, rel_half = split_key(emitted_key)
        norm_name = normalize(name_half)
        norm_rel = normalize(rel_half)
        name_ok = norm_name == gold_name
        rel_ok = norm_rel == gold_rel
        full_ok = normalize(emitted_key) == gold_key
    else:
        name_ok = rel_ok = full_ok = False

    if name_ok:
        name_error_class = None
    else:
        norm_prompt = normalize(prompt)
        emitted_name = normalize(split_key(emitted_key)[0]) if emitted_key is not None else ""
        if emitted_name and emitted_name in norm_prompt and emitted_name != gold_name:
            name_error_class = "in_context_wrong_name"
        else:
            name_error_class = "out_of_context_name"

    norm_text = normalize(text)
    answer_ok = any(normalize(a) in norm_text for a in answers)

    return {
        "status": status,
        "name_ok": name_ok,
        "rel_ok": rel_ok,
        "full_ok": full_ok,
        "answer_ok": answer_ok,
        "name_error_class": name_error_class,
        "emitted_key": emitted_key,
    }


def _meta_of(item) -> dict:
    if isinstance(item, dict):
        return item["meta"]
    return item.meta


def _prompt_of(item) -> str:
    if isinstance(item, dict):
        return item["prompt"]
    return item.prompt


def _qid_of(item):
    if isinstance(item, dict):
        return item.get("qid")
    return getattr(item, "qid", None)


def _rate(records: list[dict], key: str) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r[key]) / len(records)


def _rate_match(records: list[dict], key: str, value) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r[key] == value) / len(records)


def score_items(items: list, texts: list[str]) -> dict:
    """Score a batch of items and aggregate per split (plus ``"all"``).

    ``items`` are QAItem-like (have ``.meta``, ``.prompt``) or dicts with
    ``meta``/``prompt`` keys; ``len(items) == len(texts)``. Each split group
    reports counts and rates over ALL items in the group, each rate paired
    with a 95% Wilson interval under ``"<metric>_ci"``. ``"records"`` holds
    the per-item score dicts (with ``qid`` when available).
    """
    if len(items) != len(texts):
        raise ValueError("items and texts must have equal length")

    records: list[dict] = []
    for item, text in zip(items, texts):
        meta = _meta_of(item)
        row = score_item(meta, _prompt_of(item), text)
        row["qid"] = _qid_of(item)
        row["split"] = meta["split"]
        records.append(row)

    splits: dict[str, list[dict]] = {"all": records}
    for r in records:
        splits.setdefault(r["split"], []).append(r)

    out: dict = {}
    for name, group in splits.items():
        n = len(group)
        metrics = {
            "n": n,
            "full_key": _rate(group, "full_ok"),
            "name_half": _rate(group, "name_ok"),
            "relation_half": _rate(group, "rel_ok"),
            "answer": _rate(group, "answer_ok"),
            "no_lookup_rate": _rate_match(group, "status", "no_lookup"),
            "malformed_rate": _rate_match(group, "status", "malformed"),
            "wrong_in_context_rate": _rate_match(
                group, "name_error_class", "in_context_wrong_name"
            ),
        }
        counts = {
            "full_key": sum(1 for r in group if r["full_ok"]),
            "name_half": sum(1 for r in group if r["name_ok"]),
            "relation_half": sum(1 for r in group if r["rel_ok"]),
            "answer": sum(1 for r in group if r["answer_ok"]),
            "no_lookup_rate": sum(1 for r in group if r["status"] == "no_lookup"),
            "malformed_rate": sum(1 for r in group if r["status"] == "malformed"),
            "wrong_in_context_rate": sum(
                1 for r in group if r["name_error_class"] == "in_context_wrong_name"
            ),
        }
        for metric, value in metrics.items():
            if metric == "n":
                continue
            out_for_split = out.setdefault(name, {})
            out_for_split[metric] = value
            out_for_split[f"{metric}_ci"] = wilson(counts[metric], n)
        out[name]["n"] = n

    out["records"] = records
    return out
