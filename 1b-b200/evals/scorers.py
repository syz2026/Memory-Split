"""Answer parsing and generative exact-match scoring."""

from __future__ import annotations

import json
from pathlib import Path

from evals.generate import generate_batch_with_stats

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
    organizer,
    device,
    max_new: int = 384,
    batch_size: int = 16,
) -> tuple[list[dict], dict]:
    """Greedy-generate for each QAItem prompt and exact-match the parsed answer.

    Returns (rows, stats): one row per item
    {qid, task, correct, pred, answer, meta} plus the lookup stats
    aggregated over all batches.
    """
    rows: list[dict] = []
    total = {"n_lookups": 0, "n_hits": 0, "n_misses": 0, "n_malformed": 0}
    for lo in range(0, len(items), batch_size):
        chunk = items[lo : lo + batch_size]
        texts, stats = generate_batch_with_stats(
            model, tok, [it.prompt for it in chunk], max_new, organizer, device
        )
        for k in total:
            total[k] += stats[k]
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
    return rows, total


def save_results(rows: list[dict], path: str | Path) -> None:
    """Write result rows as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
