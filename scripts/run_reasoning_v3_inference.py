#!/usr/bin/env python3
"""Run or dry-run the complete frozen reasoning-v3 scientific inference."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corpusgen.parallel.canonical import canonical_json_bytes
from evals.reasoning_v3.inference import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_RNG_SEED,
)
from evals.reasoning_v3.reporting import (
    REPORT_KEY,
    run_frozen_scientific_inference,
)
from evals.reasoning_v3.runner import CHECKPOINT_STEPS


def _write(value: object) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze the complete fixed reasoning-v3 checkpoint matrix",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="read exact evaluator-only AWS versions and publish the report",
    )
    args = parser.parse_args(argv)
    if not args.apply:
        _write(
            {
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "checkpoint_count": 100,
                "checkpoint_steps": list(CHECKPOINT_STEPS),
                "mode": "dry_run",
                "mutation": False,
                "report_key": REPORT_KEY,
                "rng_seed": BOOTSTRAP_RNG_SEED,
                "run_count": 20,
            }
        )
        return 0
    published = run_frozen_scientific_inference()
    _write(
        {
            "decision": published.report["decision"],
            "mode": "applied",
            "result": asdict(published.object_ref),
        }
    )
    return 2 if published.report["decision"]["label"] == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
