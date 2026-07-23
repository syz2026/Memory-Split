#!/usr/bin/env python
"""Create the QA-held-out-fact KQA Pro split and organizer artifacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from corpusgen.kqa_pro import (
    KQAKnowledgeBase,
    SplitConfig,
    build_transfer_split,
    load_questions,
    write_prepared_data,
)
from corpusgen.records import QUERY_TOKEN_CAP
from train.tokenizer import get_tok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=20_000)
    parser.add_argument("--dev-limit", type=int, default=1_000)
    parser.add_argument("--test-limit", type=int, default=2_000)
    parser.add_argument("--min-train-per-skeleton", type=int, default=8)
    parser.add_argument("--max-support-facts", type=int, default=16)
    parser.add_argument("--max-values-per-lookup", type=int, default=16)
    parser.add_argument(
        "--noise-limit",
        type=int,
        default=None,
        help="sample at most this many organizer facts unused by train/test QA",
    )
    parser.add_argument("--allow-find-all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    required = [dataset_dir / name for name in ("kb.json", "train.json", "val.json")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "missing KQA files: "
            + ", ".join(missing)
            + "\nRun scripts/download_kqa.py first."
        )

    started = time.monotonic()
    print("loading KQA knowledge base...", flush=True)
    kb = KQAKnowledgeBase.load(dataset_dir / "kb.json")
    tok = get_tok()
    kb.blocked_keys.update(
        key
        for key, fact in kb.lookup_facts.items()
        if len(tok.encode(f"{fact.name}, {fact.relation}")) >= QUERY_TOKEN_CAP
    )
    print(
        f"projected {len(kb.lookup_facts):,} organizer facts "
        f"({len(kb.ambiguous_keys):,} ambiguous, "
        f"{len(kb.blocked_keys):,} over query cap)",
        flush=True,
    )
    train = load_questions(dataset_dir / "train.json")
    validation = load_questions(dataset_dir / "val.json")
    cfg = SplitConfig(
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        test_limit=args.test_limit,
        min_train_per_skeleton=args.min_train_per_skeleton,
        max_support_facts=args.max_support_facts,
        max_values_per_lookup=args.max_values_per_lookup,
        noise_limit=args.noise_limit,
        exclude_find_all=not args.allow_find_all,
        seed=args.seed,
    )
    print("executing KoPL offline and constructing the transfer split...", flush=True)
    prepared = build_transfer_split(kb, train, validation, cfg)
    report = write_prepared_data(prepared, kb, args.out_dir)
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    with open(Path(args.out_dir) / "report.json", "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if not all(report["checks"].values()):
        raise SystemExit("one or more KQA split invariants failed")


if __name__ == "__main__":
    main()
