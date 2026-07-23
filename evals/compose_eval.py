"""Two-hop composition scoring: exact-match answer + per-hop lookup accuracy.

The primary endpoint is OOD (P_held) two-hop accuracy, conditioned on both
single-hop facts being individually accessible (Karmim Table-9 style). Answers
are scored by EXACT match (evals.scorers), not the loose substring containment
in evals/recall.py.

For the split arm we also decompose the two lookup queries the model emits:
    hop 1 key : "{x}, {relation}"      -- addressing the bridge
    hop 2 key : "{bridge}, {attr}"     -- requires COPYING hop 1's retrieved
                                          bridge name into the query
Two lookups at accuracy p give ~p^2 end-to-end, so a split miss localises to
retrieval, not reasoning.
"""

from __future__ import annotations

import json
from pathlib import Path

from corpusgen.records import QAItem
from evals.generate import generate_batch_with_stats
from evals.scorers import normalize_answer, parse_answer
from organizer.store import normalize

_DB_START = "<|db_start|>"
_DB_RETRIEVE = "<|db_retrieve|>"


def load_items(path: str | Path) -> list[QAItem]:
    items: list[QAItem] = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            items.append(QAItem(**d))
    return items


def extract_all_keys(text: str) -> list[str]:
    """All well-formed lookup query strings, in emission order."""
    keys: list[str] = []
    pos = 0
    while True:
        i = text.find(_DB_START, pos)
        if i == -1:
            break
        j = text.find(_DB_RETRIEVE, i + len(_DB_START))
        if j == -1:
            break
        keys.append(text[i + len(_DB_START): j].strip())
        pos = j + len(_DB_RETRIEVE)
    return keys


def score_two_hop(gen_text: str, item: QAItem) -> dict:
    """Score one two-hop generation against its gold answer and per-hop keys."""
    pred = parse_answer(gen_text)
    answer_ok = pred is not None and normalize_answer(pred) == normalize_answer(item.answer)
    keys = extract_all_keys(gen_text)
    hop1_ok = len(keys) >= 1 and normalize(keys[0]) == normalize(item.meta["hop1_key"])
    hop2_ok = len(keys) >= 2 and normalize(keys[1]) == normalize(item.meta["hop2_key"])
    return {
        "qid": item.qid,
        "population": item.meta.get("population"),
        "answer_correct": bool(answer_ok),
        "hop1_key_ok": bool(hop1_ok),
        "hop2_key_ok": bool(hop2_ok),
        "both_hops_ok": bool(hop1_ok and hop2_ok),
        "n_lookups_emitted": len(keys),
        "pred": pred,
        "answer": item.answer,
        "meta": item.meta,
    }


def run_two_hop(model, tok, items, organizer, device, max_new: int = 96,
                batch_size: int = 64) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    total = {"n_lookups": 0, "n_hits": 0, "n_misses": 0, "n_malformed": 0}
    for lo in range(0, len(items), batch_size):
        chunk = items[lo: lo + batch_size]
        texts, stats = generate_batch_with_stats(
            model, tok, [it.prompt for it in chunk], max_new, organizer, device)
        for k in total:
            total[k] += stats[k]
        for it, gen in zip(chunk, texts):
            rows.append(score_two_hop(gen, it))
    return rows, total


def run_singlehop(model, tok, items, organizer, device, max_new: int = 24,
                  batch_size: int = 64) -> dict:
    """Single-hop fact-access probes; returns {qid: correct} + per-hop summary.

    Probes end mid-sentence ("{name}'s {relation} is"), so the answer is the
    generated continuation, NOT text after an "Answer:" tag. Score by whether
    the continuation leads with the gold value (dense) or contains it (split,
    where the value arrives inside a forced <|db_*|> lookup span). This matches
    the recall convention in evals/recall.py rather than the Answer:-tag parser.
    """
    correct: dict[str, bool] = {}
    summ: dict[str, dict] = {}
    for lo in range(0, len(items), batch_size):
        chunk = items[lo: lo + batch_size]
        texts, _ = generate_batch_with_stats(
            model, tok, [it.prompt for it in chunk], max_new, organizer, device)
        for it, gen in zip(chunk, texts):
            g, a = normalize_answer(gen), normalize_answer(it.answer)
            ok = bool(a) and (g.startswith(a) or a in g)
            correct[it.qid] = bool(ok)
            key = f"{it.meta.get('population')}-hop{it.meta.get('hop')}"
            s = summ.setdefault(key, {"n": 0, "correct": 0})
            s["n"] += 1
            s["correct"] += int(ok)
    return {"correct": correct, "by_group": {
        k: {"n": v["n"], "accuracy": v["correct"] / v["n"] if v["n"] else 0.0}
        for k, v in summ.items()}}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate_two_hop(rows: list[dict], singlehop_correct: dict | None = None) -> dict:
    """Aggregate answer + per-hop accuracy, and (if single-hop results are
    given) accuracy conditioned on both hops being individually accessible."""
    n = len(rows)
    out = {
        "n": n,
        "answer_accuracy": _mean([r["answer_correct"] for r in rows]),
        "hop1_key_accuracy": _mean([r["hop1_key_ok"] for r in rows]),
        "hop2_key_accuracy": _mean([r["hop2_key_ok"] for r in rows]),
        "both_hops_key_accuracy": _mean([r["both_hops_ok"] for r in rows]),
        "frac_emitting_two_lookups": _mean([r["n_lookups_emitted"] >= 2 for r in rows]),
    }
    if singlehop_correct is not None:
        cond_rows = []
        for r in rows:
            m = r["meta"]
            pop = m.get("population")
            q1 = f"hop1-{pop}-{m['x_id']}-{m['relation']}"
            q2 = f"hop2-{pop}-{m['y_id']}-{m['attr']}"
            if singlehop_correct.get(q1) and singlehop_correct.get(q2):
                cond_rows.append(r)
        out["n_both_hops_accessible"] = len(cond_rows)
        out["fact_access_rate"] = len(cond_rows) / n if n else 0.0
        out["answer_accuracy_conditioned"] = _mean(
            [r["answer_correct"] for r in cond_rows])
    return out
