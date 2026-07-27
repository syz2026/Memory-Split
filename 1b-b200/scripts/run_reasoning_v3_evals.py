#!/usr/bin/env python3
"""Run or dry-run one frozen reasoning-v3 checkpoint evaluation."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corpusgen.parallel.canonical import canonical_json_bytes
from evals.reasoning_v3.runner import (
    CHECKPOINT_STEPS,
    _checkpoint_result_key,
    run_frozen_checkpoint_evaluation,
)


def _write(value: object) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one fixed reasoning-v3 checkpoint",
    )
    parser.add_argument("--arm", required=True, choices=("dense", "split90"))
    parser.add_argument("--seed", required=True, type=int, choices=range(10))
    parser.add_argument(
        "--checkpoint-step",
        required=True,
        type=int,
        choices=CHECKPOINT_STEPS,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform CUDA decoding and immutable evaluator-only AWS publication",
    )
    args = parser.parse_args(argv)
    if not args.apply:
        _write(
            {
                "arm": args.arm,
                "checkpoint_step": args.checkpoint_step,
                "mode": "dry_run",
                "mutation": False,
                "result_key": _checkpoint_result_key(
                    args.arm,
                    args.seed,
                    args.checkpoint_step,
                ),
                "seed": args.seed,
            }
        )
        return 0
    published = run_frozen_checkpoint_evaluation(
        args.arm,
        args.seed,
        args.checkpoint_step,
    )
    failed = [
        name
        for name, gate in published.result["validity"]["gates"].items()
        if gate["passed"] is not True
    ]
    _write(
        {
            "failed_gates": failed,
            "mode": "invalid" if failed else "applied",
            "result": asdict(published.object_ref),
        }
    )
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
