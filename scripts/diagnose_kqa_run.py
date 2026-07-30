#!/usr/bin/env python
"""Explain why a finished KQA evaluation scored at floor.

Runs offline against artifacts a completed evaluation already wrote, so it
needs no GPU and no model. It separates the two failure modes that both look
like "accuracy 0":

  store    the split arm opened lookups whose addresses miss the organizer
  format   the model answered, but generation/scoring discarded the answer

Usage:
  python scripts/diagnose_kqa_run.py --run <run-or-eval-dir> --data-dir <corpus>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evals.scorers import normalize_answer, parse_answer, parse_first_answer
from organizer.store import Organizer, normalize


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _eval_dir(path: str) -> Path:
    root = Path(path)
    return root if (root / "summary.json").exists() else root / "kqa_evals"


def _emitted_queries(row: dict) -> list[str]:
    return [
        event["query"]
        for event in row.get("events", ())
        if event.get("query") is not None
    ]


def diagnose_recall(rows: list[dict], organizer: Organizer) -> dict:
    """Compare the addresses the model wrote against the ones the store holds."""
    opened = [row for row in rows if _emitted_queries(row)]
    exact = 0
    resolvable = 0
    entity_copied = 0
    relation_copied = 0
    malformed = 0
    samples: list[dict] = []

    for row in rows:
        expected = normalize(row["fact_id"])
        expected_entity, _, expected_relation = expected.partition(", ")
        queries = [normalize(query) for query in _emitted_queries(row)]
        malformed += sum(
            1 for event in row.get("events", ()) if event.get("malformed")
        )
        if expected in queries:
            exact += 1
        if any(organizer.lookup(query) is not None for query in queries):
            resolvable += 1
        if any(query.partition(", ")[0] == expected_entity for query in queries):
            entity_copied += 1
        if any(query.partition(", ")[2] == expected_relation for query in queries):
            relation_copied += 1
        if queries and expected not in queries and len(samples) < 12:
            samples.append({"expected": expected, "emitted": queries[0]})

    n = len(rows) or 1
    return {
        "probes": len(rows),
        "opened_a_lookup": len(opened),
        "emitted_expected_address": exact,
        "emitted_any_address_in_store": resolvable,
        "copied_entity_correctly": entity_copied,
        "copied_relation_correctly": relation_copied,
        "malformed_events": malformed,
        "rates": {
            "opened_a_lookup": round(len(opened) / n, 4),
            "emitted_expected_address": round(exact / n, 4),
            "copied_entity_correctly": round(entity_copied / n, 4),
            "copied_relation_correctly": round(relation_copied / n, 4),
        },
        "mismatch_samples": samples,
    }


def diagnose_answers(rows: list[dict]) -> dict:
    """Separate 'answered wrongly' from 'answer discarded by scoring'."""
    tagged = 0
    multi = 0
    first_correct = 0
    last_correct = 0
    empty = 0
    samples: list[dict] = []

    for row in rows:
        generated = row.get("generated", "") or ""
        tags = generated.count("Answer:")
        if tags:
            tagged += 1
        if tags > 1:
            multi += 1
        if not generated.strip():
            empty += 1
        expected = normalize_answer(row["answer"])
        first = parse_first_answer(generated)
        last = parse_answer(generated)
        first_hit = first is not None and normalize_answer(first) == expected
        last_hit = last is not None and normalize_answer(last) == expected
        first_correct += first_hit
        last_correct += last_hit
        if first_hit and not last_hit and len(samples) < 8:
            samples.append(
                {"qid": row["qid"], "answer": row["answer"], "first": first, "last": last}
            )

    n = len(rows) or 1
    return {
        "items": len(rows),
        "emitted_answer_tag": tagged,
        "emitted_multiple_answer_tags": multi,
        "empty_generations": empty,
        "accuracy_first_answer": round(first_correct / n, 5),
        "accuracy_last_answer": round(last_correct / n, 5),
        "recovered_by_first_answer": first_correct - last_correct,
        "recovered_samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    eval_dir = _eval_dir(args.run)
    data_dir = Path(args.data_dir)
    organizer = Organizer.load(data_dir / "organizer.jsonl")

    report: dict = {
        "eval_dir": str(eval_dir),
        "organizer_entries": len(organizer),
    }
    if not organizer:
        report["verdict"] = "organizer is empty; every lookup must miss"

    recall_path = next(
        (
            eval_dir / name
            for name in ("recall_on.jsonl", "recall_closed.jsonl")
            if (eval_dir / name).exists()
        ),
        None,
    )
    if recall_path is not None:
        rows = _load_jsonl(recall_path)
        report["recall"] = {"file": recall_path.name, **diagnose_recall(rows, organizer)}

    for name, label in (
        ("transfer_qa.jsonl", "transfer_qa"),
        ("transfer_qa_oracle_context.jsonl", "oracle_context"),
    ):
        path = eval_dir / name
        if path.exists():
            report[label] = diagnose_answers(_load_jsonl(path))

    verdicts = []
    recall = report.get("recall")
    if recall and recall["opened_a_lookup"] and not recall["emitted_expected_address"]:
        verdicts.append(
            "store: the model opens lookups but never writes an address the "
            "organizer holds, so no fact can reach the answer"
        )
    for label in ("transfer_qa", "oracle_context"):
        section = report.get(label)
        if section and section["recovered_by_first_answer"] > 0:
            verdicts.append(
                f"format: {label} loses "
                f"{section['recovered_by_first_answer']} correct answers to "
                "trailing invented question/answer pairs"
            )
    oracle = report.get("oracle_context")
    if oracle and oracle["accuracy_first_answer"] < 0.05:
        verdicts.append(
            "reasoning: accuracy stays at floor even with every support fact "
            "supplied in the prompt, so the model has not learned the task"
        )
    report["verdicts"] = verdicts

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
