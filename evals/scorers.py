"""Answer parsing and generative exact-match scoring."""

from __future__ import annotations

import json
from pathlib import Path

from evals.generate import generate_batch

_ANSWER_TAG = "Answer:"
_EOT_MARKER = "<|eot|>"


def parse_answer(text: str) -> str | None:
    """Text after the LAST 'Answer:', up to newline/EOT marker, stripped.

    Returns None when no 'Answer:' tag is present. An empty answer line
    parses to "" (present but blank), distinct from None.
    """
    idx = text.rfind(_ANSWER_TAG)
    if idx == -1:
        return None
    rest = text[idx + len(_ANSWER_TAG) :]
    for stop in ("\n", _EOT_MARKER):
        cut = rest.find(stop)
        if cut != -1:
            rest = rest[:cut]
    return rest.strip()


def normalize_answer(s: str) -> str:
    """Lowercase, collapse whitespace, strip a trailing period."""
    s = " ".join(s.lower().split())
    return s.removesuffix(".").strip()


def score_items(
    model,
    tok,
    items,
    device,
    max_new: int = 384,
    batch_size: int = 16,
) -> list[dict]:
    """Greedy-generate for each QAItem prompt and exact-match the parsed answer.

    Returns one row per item: {qid, task, correct, pred, answer, meta}. Rows
    carry `answer` and `pred` so per-class and per-op breakdowns are
    recoverable post hoc without rerunning generation.
    """
    rows: list[dict] = []
    for lo in range(0, len(items), batch_size):
        chunk = items[lo : lo + batch_size]
        texts = generate_batch(
            model, tok, [it.prompt for it in chunk], max_new, device
        )
        for it, gen in zip(chunk, texts):
            pred = parse_answer(gen)
            correct = pred is not None and (
                normalize_answer(pred) == normalize_answer(it.answer)
            )
            rows.append(
                {
                    "qid": it.qid,
                    "task": it.task,
                    "correct": correct,
                    "pred": pred,
                    "answer": it.answer,
                    "meta": it.meta,
                }
            )
    return rows


def save_results(rows: list[dict], path: str | Path) -> None:
    """Write result rows as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def accuracy_by(rows: list[dict], key: str) -> dict:
    """Accuracy bucketed by a `meta` key, plus the majority-class rate.

    Two endpoints in this project are unreadable as a single number. iGSM
    trains 1/op-weighted and evaluates uniformly, so the aggregate is
    dominated by the least-trained levels; report per `op`. Deduction's eval
    is exactly balanced with a canned NO-branch trace, so a constant "no"
    scores exactly 0.500; report per `answer_class`.

    `majority_rate` is the best constant predictor over this row set. For
    iGSM under MOD=23 it sits near 7.2%, not 1/23 = 4.3%, because `times`
    overproduces zero -- scoring against 1/23 has misreported every iGSM
    number this project has published.
    """
    buckets: dict[object, list[dict]] = {}
    for r in rows:
        buckets.setdefault(r.get("meta", {}).get(key), []).append(r)

    answers: dict[object, int] = {}
    for r in rows:
        answers[r["answer"]] = answers.get(r["answer"], 0) + 1
    majority = max(answers.values()) / len(rows) if rows else 0.0

    return {
        "key": key,
        "n": len(rows),
        "overall": sum(r["correct"] for r in rows) / len(rows) if rows else 0.0,
        "majority_rate": majority,
        "by": {
            str(k): {
                "n": len(v),
                "acc": sum(r["correct"] for r in v) / len(v) if v else 0.0,
            }
            for k, v in sorted(buckets.items(), key=lambda kv: str(kv[0]))
        },
    }
