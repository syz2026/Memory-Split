"""Command-line entry point for protected paired Slurm operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from msctl.cohort import ROLES
from msctl.operations import (
    collect,
    evaluate,
    instantiate,
    resume,
    status,
    submit,
)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=_json_default))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m msctl")
    commands = parser.add_subparsers(dest="command", required=True)
    runs = commands.add_parser("runs")
    run_commands = runs.add_subparsers(dest="runs_command", required=True)
    instantiate_parser = run_commands.add_parser("instantiate")
    instantiate_parser.add_argument("role", choices=tuple(ROLES))
    instantiate_parser.add_argument("--dataset-root", required=True)
    instantiate_parser.add_argument("--pointer", required=True)
    instantiate_parser.add_argument("--source-lock", required=True)
    instantiate_parser.add_argument("--profile", required=True)
    instantiate_parser.add_argument("--runtime-root", required=True)
    instantiate_parser.add_argument("--out-root", required=True)
    instantiate_parser.add_argument("--repository-root", default=".")

    def add_submit_options(command):
        command.add_argument("pair_manifests", nargs="+")
        command.add_argument("--profile", required=True)
        command.add_argument("--venv-root", required=True)
        command.add_argument("--mode", choices=("functional", "resume", "throughput", "protected"), default="protected")
        command.add_argument("--preflight")
        command.add_argument("--apply", action="store_true")

    submit_parser = commands.add_parser("submit")
    add_submit_options(submit_parser)
    evaluate_parser = commands.add_parser("evaluate")
    add_submit_options(evaluate_parser)

    resume_parser = commands.add_parser("resume")
    resume_parser.add_argument("pair_manifest")
    resume_parser.add_argument("--profile", required=True)
    resume_parser.add_argument("--venv-root", required=True)
    resume_parser.add_argument("--preflight", required=True)
    resume_parser.add_argument("--apply", action="store_true")

    status_parser = commands.add_parser("status")
    status_parser.add_argument("pair_manifests", nargs="+")
    status_parser.add_argument("--evidence-root", required=True)
    status_parser.add_argument("--action", choices=("train", "evaluate"), default="train")

    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("pair_manifests", nargs="+")
    collect_parser.add_argument("--evidence-root", required=True)
    collect_parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "runs":
        result = instantiate(
            args.role,
            dataset_root=args.dataset_root,
            pointer_path=args.pointer,
            source_lock_path=args.source_lock,
            profile_path=args.profile,
            runtime_root=args.runtime_root,
            out_root=args.out_root,
            repository_root=args.repository_root,
        )
    elif args.command in {"submit", "evaluate"}:
        function = submit if args.command == "submit" else evaluate
        result = function(
            args.pair_manifests,
            profile_path=args.profile,
            mode=args.mode,
            venv_root=args.venv_root,
            preflight_path=args.preflight,
            apply=args.apply,
        )
    elif args.command == "resume":
        result = resume(
            args.pair_manifest,
            profile_path=args.profile,
            venv_root=args.venv_root,
            preflight_path=args.preflight,
            apply=args.apply,
        )
    elif args.command == "status":
        result = status(
            args.pair_manifests,
            evidence_root=args.evidence_root,
            action=args.action,
        )
    else:
        result = collect(
            args.pair_manifests,
            evidence_root=args.evidence_root,
            output=args.output,
        )
    _print(result)
    return int(result.get("exit_code", 0)) if isinstance(result, dict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
