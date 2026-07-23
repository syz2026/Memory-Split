#!/usr/bin/env python3
"""Build and verify deterministic metadata-first parallel corpora."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from corpusgen.parallel import (  # noqa: E402
    FixtureRenderer,
    InputCatalog,
    ParallelBuildConfig,
    UnsupportedProductionRenderer,
    build_parallel_corpus,
    fixture_catalog,
    verify_parallel_corpus,
)


def _print_receipt(receipt: dict) -> None:
    print(
        json.dumps(
            receipt,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    )


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _lane_weights(values: list[str]) -> tuple[tuple[str, int], ...]:
    weights = []
    for value in values:
        try:
            lane, raw_weight = value.split("=", 1)
            weight = int(raw_weight)
        except ValueError as error:
            raise ValueError("lane weights must use LANE=POSITIVE_INT") from error
        if not lane or weight <= 0:
            raise ValueError("lane weights must use LANE=POSITIVE_INT")
        weights.append((lane, weight))
    return tuple(weights)


def _config(args: argparse.Namespace, lane_weights) -> ParallelBuildConfig:
    return ParallelBuildConfig(
        lane_weights=lane_weights,
        update_tokens=args.update_tokens,
        shard_count=args.shards,
        allow_fewer_shards=args.allow_fewer_shards,
    )


def _add_packing_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update-tokens", type=_positive, required=True)
    parser.add_argument("--shards", type=_positive, default=32)
    parser.add_argument("--allow-fewer-shards", action="store_true")
    parser.add_argument(
        "--workers",
        type=_positive,
        default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fixture = commands.add_parser(
        "build-fixture",
        help="run the archive-free miniature end-to-end path",
    )
    _add_packing_options(fixture)
    fixture.add_argument("--records", type=_positive, default=24)

    production = commands.add_parser(
        "build-production",
        help="exercise the explicit fail-closed production adapter boundary",
    )
    _add_packing_options(production)
    production.add_argument("--catalog", type=Path, required=True)
    production.add_argument("--source", required=True)
    production.add_argument(
        "--lane-weight",
        action="append",
        required=True,
        metavar="LANE=WEIGHT",
    )

    verify = commands.add_parser("verify", help="verify a published corpus")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--expected-build-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        _print_receipt(
            verify_parallel_corpus(
                args.output,
                expected_build_id=args.expected_build_id,
            )
        )
        return 0
    if args.command == "build-fixture":
        catalog = fixture_catalog(args.records)
        renderer = FixtureRenderer()
        lane_weights = (("natural", 1), ("facts", 1), ("reasoning", 1))
    elif args.command == "build-production":
        catalog = InputCatalog.from_bytes(args.catalog.read_bytes())
        renderer = UnsupportedProductionRenderer(args.source)
        lane_weights = _lane_weights(args.lane_weight)
    else:
        raise AssertionError(f"unhandled command: {args.command}")
    receipt = build_parallel_corpus(
        catalog,
        renderer,
        _config(args, lane_weights),
        args.output,
        workers=args.workers,
    )
    _print_receipt(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
