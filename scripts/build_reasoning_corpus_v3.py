"""Build or verify the append-only MemorySplit v3 reasoning corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpusgen.reasoning_expansion import (
    DEFAULT_BASE_CORPUS,
    DEFAULT_OUTPUT,
    DEFAULT_RECIPE_PATH,
    DEFAULT_SOURCE_STAGE,
    build_reasoning_corpus,
    verify_reasoning_corpus,
)


def _progress(completed: int, total: int) -> None:
    print(
        json.dumps(
            {
                "event": "reasoning_extension_progress",
                "tokens": completed,
                "total_tokens": total,
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe",
        type=Path,
        default=DEFAULT_RECIPE_PATH,
    )
    parser.add_argument(
        "--source-stage",
        type=Path,
        default=DEFAULT_SOURCE_STAGE,
    )
    parser.add_argument(
        "--base-corpus",
        type=Path,
        default=DEFAULT_BASE_CORPUS,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--publication", type=Path, default=DEFAULT_OUTPUT)
    verify.add_argument("--expected-receipt-sha256")
    args = parser.parse_args(argv)

    if args.command == "build":
        report = build_reasoning_corpus(
            recipe_path=args.recipe,
            source_stage=args.source_stage,
            base_corpus=args.base_corpus,
            destination=args.output,
            progress=_progress,
        )
    else:
        report = verify_reasoning_corpus(
            publication=args.publication,
            recipe_path=args.recipe,
            source_stage=args.source_stage,
            base_corpus=args.base_corpus,
            expected_receipt_sha256=args.expected_receipt_sha256,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
