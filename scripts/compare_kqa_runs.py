#!/usr/bin/env python
"""Paired comparison of completed dense and split KQA evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.stats import paired_delta


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _eval_dir(path: str) -> Path:
    root = Path(path)
    return root if (root / "summary.json").exists() else root / "kqa_evals"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-run", required=True)
    parser.add_argument("--split-run", required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()

    dense_dir = _eval_dir(args.dense_run)
    split_dir = _eval_dir(args.split_run)
    dense_rows = _load_jsonl(dense_dir / "transfer_qa.jsonl")
    split_rows = _load_jsonl(split_dir / "transfer_qa.jsonl")
    dense_oracle_rows = _load_jsonl(
        dense_dir / "transfer_qa_oracle_context.jsonl"
    )
    split_oracle_rows = _load_jsonl(
        split_dir / "transfer_qa_oracle_context.jsonl"
    )
    wrong_store_path = split_dir / "transfer_qa_wrong_store.jsonl"
    wrong_store_rows = (
        _load_jsonl(wrong_store_path) if wrong_store_path.exists() else None
    )
    dense_summary = json.loads((dense_dir / "summary.json").read_text())
    split_summary = json.loads((split_dir / "summary.json").read_text())
    all_items = paired_delta(
        split_rows,
        dense_rows,
        cluster_key="template",
        n_boot=args.n_boot,
        seed=args.seed,
    )

    dense_by_id = {row["qid"]: row for row in dense_rows}
    split_by_id = {row["qid"]: row for row in split_rows}
    common_available_ids = {
        qid
        for qid in dense_by_id.keys() & split_by_id.keys()
        if dense_by_id[qid]["facts_available"]
        and split_by_id[qid]["facts_available"]
    }
    dense_available = [
        row for row in dense_rows if row["qid"] in common_available_ids
    ]
    split_available = [
        row for row in split_rows if row["qid"] in common_available_ids
    ]
    conditional = (
        paired_delta(
            split_available,
            dense_available,
            cluster_key="template",
            n_boot=args.n_boot,
            seed=args.seed,
        )
        if common_available_ids
        else None
    )
    result = {
        "contrast": "split - dense",
        "all_test_questions": all_items,
        "oracle_context": {
            "split_minus_dense": paired_delta(
                split_oracle_rows,
                dense_oracle_rows,
                cluster_key="template",
                n_boot=args.n_boot,
                seed=args.seed,
            ),
            "dense_oracle_minus_closed_book": paired_delta(
                dense_oracle_rows,
                dense_rows,
                cluster_key="template",
                n_boot=args.n_boot,
                seed=args.seed,
            ),
            "split_oracle_minus_store_on": paired_delta(
                split_oracle_rows,
                split_rows,
                cluster_key="template",
                n_boot=args.n_boot,
                seed=args.seed,
            ),
        },
        "both_systems_have_all_support_facts": conditional,
        "n_both_available": len(common_available_ids),
        "access_guardrails": {
            "dense_direct_recall": dense_summary["recall"],
            "split_store_recall": split_summary["recall"],
            "split_store_off_recall": split_summary.get("recall_off"),
            "split_store_off_transfer_qa": split_summary.get("transfer_qa_off"),
            "split_wrong_store_transfer_qa": split_summary.get(
                "transfer_qa_wrong_store"
            ),
            "split_store_on_minus_wrong_store": (
                paired_delta(
                    split_rows,
                    wrong_store_rows,
                    cluster_key="template",
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
                if wrong_store_rows is not None
                else None
            ),
            "split_qa_support_query_coverage": {
                "all_support_queried_rate": split_summary["transfer_qa"].get(
                    "all_support_queried_rate"
                ),
                "mean_query_support_coverage": split_summary["transfer_qa"].get(
                    "mean_query_support_coverage"
                ),
            },
        },
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
