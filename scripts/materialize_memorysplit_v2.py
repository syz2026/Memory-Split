#!/usr/bin/env python3
"""Materialize, resume, or inspect the frozen MemorySplit v2 source root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpusgen.parallel.production import (
    DEFAULT_RECIPE_PATH,
    ProductionPreflightError,
    load_production_recipe,
)
from corpusgen.v2_materialize import (
    DEFAULT_CHECKPOINT_TOKENS,
    DEFAULT_FINISH_CANDIDATES,
    DEFAULT_FINISH_WINDOW,
    V2MaterializationError,
    materialization_status,
    materialize_v2_smoke,
    materialize_v2_source_root,
)
from corpusgen.v2_sources import (
    DEFAULT_V2_SOURCE_LOCK,
    V2SourceStageError,
)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _emit(value: object, *, stream=sys.stdout) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ),
        file=stream,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    materialize = commands.add_parser(
        "materialize",
        help="build or resume all eight lanes from a verified upstream stage",
    )
    materialize.add_argument("--stage-root", type=Path, required=True)
    materialize.add_argument("--source-root", type=Path, required=True)
    materialize.add_argument("--work-root", type=Path, required=True)
    materialize.add_argument(
        "--source-lock",
        type=Path,
        default=DEFAULT_V2_SOURCE_LOCK,
    )
    materialize.add_argument(
        "--recipe",
        type=Path,
        default=DEFAULT_RECIPE_PATH,
    )
    materialize.add_argument(
        "--objective-python",
        type=Path,
        default=Path(sys.executable),
        help="interpreter containing every pinned objective-generator dependency",
    )
    materialize.add_argument(
        "--checkpoint-tokens",
        type=_positive,
        default=DEFAULT_CHECKPOINT_TOKENS,
    )
    materialize.add_argument(
        "--finish-window",
        type=_positive,
        default=DEFAULT_FINISH_WINDOW,
    )
    materialize.add_argument(
        "--max-finish-candidates",
        type=_positive,
        default=DEFAULT_FINISH_CANDIDATES,
    )

    status = commands.add_parser(
        "status",
        help="print lane checkpoints and routing completion",
    )
    status.add_argument("--work-root", type=Path, required=True)
    status.add_argument(
        "--recipe",
        type=Path,
        default=DEFAULT_RECIPE_PATH,
    )

    smoke = commands.add_parser(
        "smoke",
        help="build a tiny explicitly non-scientific 45-artifact contract",
    )
    smoke.add_argument("--source-root", type=Path, required=True)
    smoke.add_argument("--work-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "status":
            _emit(
                materialization_status(
                    args.work_root,
                    recipe=load_production_recipe(args.recipe),
                )
            )
            return 0
        if args.command == "smoke":
            _emit(materialize_v2_smoke(args.source_root, args.work_root))
            return 0
        _emit(
            materialize_v2_source_root(
                args.stage_root,
                args.source_root,
                args.work_root,
                source_lock_path=args.source_lock,
                recipe_path=args.recipe,
                objective_python=args.objective_python,
                checkpoint_tokens=args.checkpoint_tokens,
                finish_window=args.finish_window,
                max_finish_candidates=args.max_finish_candidates,
            )
        )
        return 0
    except V2MaterializationError as error:
        _emit(
            {
                "error": error.as_dict(),
                "format": "memorysplit-v2-materialization-error-v1",
                "ready": False,
            },
            stream=sys.stderr,
        )
        return 2
    except ProductionPreflightError as error:
        _emit(error.report, stream=sys.stderr)
        return 2
    except (OSError, ValueError, V2SourceStageError) as error:
        _emit(
            {
                "error": {
                    "message": str(error),
                    "type": type(error).__name__,
                },
                "format": "memorysplit-v2-materialization-error-v1",
                "ready": False,
            },
            stream=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
