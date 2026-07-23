#!/usr/bin/env python
"""Build the legacy pilot corpus.

Current seven-lane builds use ``scripts/build_current_dataset.py`` so the
verified Task 1 source contract cannot be bypassed accidentally.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpusgen.relational_build import (
    RelationalBuildConfig,
    build_relational_corpus,
)
from train.tokenizer import get_tok


def iter_bed_jsonl(path: Path | str):
    path = Path(path)
    while True:
        saw_text = False
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text") if isinstance(row, dict) else None
                if not isinstance(text, str) or not text:
                    raise ValueError(
                        f"{path}:{line_number} requires a non-empty text field"
                    )
                saw_text = True
                yield text
        if not saw_text:
            raise ValueError(f"{path} contains no natural-text records")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the legacy synthetic relational pilot with dense, split, "
            "and matched-random target weights."
        ),
        epilog=(
            "For relational-chinchilla current data, run "
            "scripts/build_current_dataset.py with a verified source root."
        ),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--entities", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--bed-jsonl", required=True)
    parser.add_argument("--route-policy", required=True)
    parser.add_argument("--route-policy-sha256", required=True)
    parser.add_argument("--world-size", type=int, default=64)
    parser.add_argument("--eval-pairs-per-task", type=int, default=10_000)
    parser.add_argument("--eval-pairs-per-world", type=int, default=32)
    parser.add_argument("--guardrail-items", type=int, default=10_000)
    parser.add_argument("--shared-text-eval-count", type=int, default=64)
    args = parser.parse_args(argv)

    cfg = RelationalBuildConfig(
        n_entities=args.entities,
        total_tokens=args.tokens,
        data_seed=args.data_seed,
        world_size=args.world_size,
        eval_pairs_per_task=args.eval_pairs_per_task,
        eval_pairs_per_world=args.eval_pairs_per_world,
        guardrail_items=args.guardrail_items,
        shared_text_eval_count=args.shared_text_eval_count,
    )
    report = build_relational_corpus(
        cfg,
        get_tok(),
        iter_bed_jsonl(args.bed_jsonl),
        Path(args.out),
        route_policy_path=args.route_policy,
        expected_policy_sha256=args.route_policy_sha256,
    )
    print(json.dumps(report["checks"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
