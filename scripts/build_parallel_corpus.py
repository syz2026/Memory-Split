#!/usr/bin/env python3
"""Build and verify deterministic metadata-first parallel corpora."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cluster.aws.corpus_builder.driver import (  # noqa: E402
    load_verified_production_inputs,
)
from corpusgen.parallel import (  # noqa: E402
    FixtureRenderer,
    IncompleteTaskResults,
    ParallelBuildConfig,
    build_parallel_corpus,
    build_parallel_corpus_from_tasks,
    fixture_catalog,
    load_task_results,
    parallel_build_id,
    publish_task_result_via_local_cache,
    publish_verification_receipt,
    render_task_result,
    verify_parallel_corpus,
)
from corpusgen.reasoning_v2.contracts import load_recipe  # noqa: E402


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


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _production_lane_weights() -> tuple[tuple[str, int], ...]:
    recipe = load_recipe(_ROOT / "configs" / "reasoning-dataset-v2.json")
    denominator = math.lcm(
        *(share.denominator for _lane, share in recipe.lane_shares)
    )
    return tuple(
        (lane, share.numerator * (denominator // share.denominator))
        for lane, share in recipe.lane_shares
    )


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
    parser.add_argument("--dense-target-weights", type=Path)
    parser.add_argument("--split90-target-weights", type=Path)
    parser.add_argument(
        "--workers",
        type=_positive,
        default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
    )


def _add_fixture_task_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--scheduler-id", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--records", type=_positive, required=True)
    parser.add_argument("--task-count", type=_positive, required=True)
    parser.add_argument("--update-tokens", type=_positive, required=True)
    parser.add_argument("--shards", type=_positive, default=32)
    parser.add_argument("--allow-fewer-shards", action="store_true")


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
        help="build from verified production source and derived authorities",
    )
    production.add_argument("--source-lock", type=Path, required=True)
    production.add_argument("--source-root", type=Path, required=True)
    production.add_argument("--derived-root", type=Path, required=True)
    production.add_argument("--generator-commit", required=True)
    production.add_argument("--output", type=Path, required=True)
    production.add_argument("--update-tokens", type=_positive, required=True)
    production.add_argument("--shards", type=_positive, required=True)
    production.add_argument("--workers", type=_positive, required=True)
    production.add_argument("--allow-fewer-shards", action="store_true")

    render_task = commands.add_parser(
        "render-fixture-task",
        help="render one disjoint fixture ordinal partition",
    )
    _add_fixture_task_options(render_task)
    render_task.add_argument("--task-index", type=_nonnegative, required=True)
    render_task.add_argument("--local-root", type=Path, required=True)
    render_task.add_argument("--job-id", required=True)
    render_task.add_argument(
        "--workers",
        type=_positive,
        default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
    )

    finalize_tasks = commands.add_parser(
        "finalize-fixture-tasks",
        help="validate every task result and publish one fixture corpus",
    )
    _add_fixture_task_options(finalize_tasks)
    finalize_tasks.add_argument("--allow-incomplete", action="store_true")

    publish_verification = commands.add_parser(
        "publish-verification-receipt",
        help="verify a corpus and atomically publish its canonical receipt",
    )
    publish_verification.add_argument("--output", type=Path, required=True)
    publish_verification.add_argument("--receipt", type=Path, required=True)
    publish_verification.add_argument("--expected-build-id", required=True)

    verify = commands.add_parser("verify", help="verify a published corpus")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--expected-build-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "verify":
        _print_receipt(
            verify_parallel_corpus(
                args.output,
                expected_build_id=args.expected_build_id,
            )
        )
        return 0
    if args.command == "publish-verification-receipt":
        _print_receipt(
            publish_verification_receipt(
                args.output,
                args.receipt,
                expected_build_id=args.expected_build_id,
            )
        )
        return 0
    if args.command in {"render-fixture-task", "finalize-fixture-tasks"}:
        catalog = fixture_catalog(args.records)
        renderer = FixtureRenderer()
        config = _config(
            args,
            (("natural", 1), ("facts", 1), ("reasoning", 1)),
        )
        build_id = parallel_build_id(catalog, renderer.renderer_id, config)
        if args.command == "render-fixture-task":
            result = render_task_result(
                catalog,
                renderer,
                config,
                task_index=args.task_index,
                task_count=args.task_count,
                workers=args.workers,
            )
            result_path = publish_task_result_via_local_cache(
                args.local_root,
                args.shared_root,
                result,
                scheduler_id=args.scheduler_id,
                job_id=args.job_id,
                nonce=args.nonce,
            )
            _print_receipt(
                {
                    "build_id": build_id,
                    "result_path": str(result_path),
                    "task_count": args.task_count,
                    "task_index": args.task_index,
                }
            )
            return 0
        try:
            results = load_task_results(
                args.shared_root,
                build_id,
                scheduler_id=args.scheduler_id,
                nonce=args.nonce,
                expected_task_count=args.task_count,
            )
        except IncompleteTaskResults as error:
            print(str(error), file=sys.stderr)
            return 75 if args.allow_incomplete else 1
        output = args.shared_root / f"fixture-{build_id}"
        _print_receipt(
            build_parallel_corpus_from_tasks(
                catalog,
                renderer.renderer_id,
                config,
                output,
                results,
                expected_task_count=args.task_count,
            )
        )
        return 0
    if args.command == "build-fixture":
        catalog = fixture_catalog(args.records)
        renderer = FixtureRenderer()
        lane_weights = (("natural", 1), ("facts", 1), ("reasoning", 1))
        if (args.dense_target_weights is None) != (
            args.split90_target_weights is None
        ):
            parser.error(
                "--dense-target-weights and --split90-target-weights "
                "must be supplied together"
            )
        sidecar_paths = (
            {
                "dense_target_weights": args.dense_target_weights,
                "split90_target_weights": args.split90_target_weights,
            }
            if (
                args.dense_target_weights is not None
                and args.split90_target_weights is not None
            )
            else None
        )
    elif args.command == "build-production":
        inputs = load_verified_production_inputs(
            source_lock_path=args.source_lock,
            source_root=args.source_root,
            derived_root=args.derived_root,
            expected_generator_commit=args.generator_commit,
        )
        catalog = inputs.catalog
        renderer = inputs.renderer
        lane_weights = _production_lane_weights()
        sidecar_paths = dict(inputs.sidecars)
    else:
        raise AssertionError(f"unhandled command: {args.command}")
    receipt = build_parallel_corpus(
        catalog,
        renderer,
        _config(args, lane_weights),
        args.output,
        workers=args.workers,
        sidecar_paths=sidecar_paths,
    )
    _print_receipt(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
