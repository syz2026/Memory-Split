"""Strict command-line entry point for confirmatory evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from evals.confirmatory.runner import ModelAdapter, evaluate, preflight


class _UsageError(ValueError):
    pass


class _HelpRequested(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _UsageError(message)

    def print_help(self, file=None) -> None:
        super().print_help(file=sys.stderr if file is None else file)

    def exit(self, status: int = 0, message: str | None = None) -> None:
        if status == 0:
            raise _HelpRequested
        raise _UsageError(message or f"parser exited with status {status}")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="python -m evals.confirmatory")
    commands = parser.add_subparsers(dest="command")
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--run", required=True)
    evaluate_parser.add_argument("--sealed-release", required=True)
    evaluate_parser.add_argument(
        "--expected-study-lock-sha256",
        required=True,
    )
    evaluate_parser.add_argument(
        "--device",
        required=True,
        choices=("cpu", "cuda", "mps"),
    )
    evaluate_parser.add_argument("--output-dir")
    return parser


def _write_stdout(payload: dict[str, object]) -> None:
    sys.stdout.write(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    model_adapter: ModelAdapter | None = None,
) -> int:
    """Run the CLI while preserving one-object stdout on every path."""

    try:
        arguments = _parser().parse_args(argv)
        if arguments.command != "evaluate":
            raise _UsageError("the evaluate command is required")
        if arguments.output_dir is None:
            result = preflight(
                run=arguments.run,
                sealed_release=arguments.sealed_release,
                expected_study_lock_sha256=(arguments.expected_study_lock_sha256),
            )
            payload = {
                "status": "dry_run",
                "device": arguments.device,
                "study_lock_sha256": result.study_lock_sha256,
                "checkpoint_sha256": result.checkpoint_sha256,
                "condition_id": result.condition_id,
                "seed": result.seed,
                "item_count": result.item_count,
            }
            print("confirmatory evaluation dry-run verified", file=sys.stderr)
            return_code = 0
        else:
            if model_adapter is None:
                raise RuntimeError(
                    "evaluate with --output-dir requires an injected "
                    "ModelAdapter; no production checkpoint adapter is "
                    "registered"
                )
            result = evaluate(
                run=arguments.run,
                sealed_release=arguments.sealed_release,
                expected_study_lock_sha256=(arguments.expected_study_lock_sha256),
                model_adapter=model_adapter,
                output_dir=arguments.output_dir,
            )
            payload = {
                "status": "published",
                "device": arguments.device,
                "output_dir": str(result.output_dir),
                "study_lock_sha256": result.study_lock_sha256,
                "checkpoint_sha256": result.checkpoint_sha256,
                "condition_id": result.condition_id,
                "seed": result.seed,
                "item_count": result.item_count,
                "report_sha256": result.report_sha256,
            }
            print("confirmatory evidence published", file=sys.stderr)
            return_code = 0
    except _HelpRequested:
        payload = {"status": "help"}
        return_code = 0
    except _UsageError as exc:
        payload = {"status": "error", "error": str(exc)}
        print(f"confirmatory CLI usage error: {exc}", file=sys.stderr)
        return_code = 2
    except Exception as exc:
        payload = {"status": "error", "error": str(exc)}
        print(f"confirmatory evaluation failed: {exc}", file=sys.stderr)
        return_code = 1
    _write_stdout(payload)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
