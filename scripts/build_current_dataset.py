#!/usr/bin/env python
"""Compile one verified current seven-lane dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpusgen.current_dataset import CurrentBuildConfig, build_current_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile the locked relational-chinchilla seven-lane corpus from "
            "an already verified Task 1 source root."
        )
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile", choices=("full",), default="full")
    parser.add_argument("--scale", choices=("29m", "160m", "360m"), required=True)
    parser.add_argument(
        "--fact-load",
        choices=("n50k", "n800k", "n1p8m"),
        required=True,
    )
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    args = parser.parse_args(argv)

    report = build_current_dataset(
        CurrentBuildConfig(
            profile=args.profile,
            scale=args.scale,
            fact_load=args.fact_load,
            data_seed=args.data_seed,
            total_tokens=args.tokens,
        ),
        Path(args.source_root),
        Path(args.out),
    )
    print(
        json.dumps(
            {
                "profile": report["profile"],
                "scientific_result": report["scientific_result"],
                "tokens": report["tokens"]["total"],
                "coverage": report["coverage"],
                "checks": report["checks"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
