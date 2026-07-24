"""Command-line entry point for protected paired Slurm operations."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from cluster.aws.reasoning_v3 import (
    stage_from_s3,
    upload_to_s3,
    verify_staged_corpus,
)
from msctl.aws_operations import instantiate_aws
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
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
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

    aws = commands.add_parser("aws")
    aws_commands = aws.add_subparsers(dest="aws_command", required=True)
    upload_parser = aws_commands.add_parser("upload-corpus")
    upload_parser.add_argument("--repository-root", default=".")
    upload_parser.add_argument(
        "--manifest",
        default="cluster/aws/reasoning-v3-corpus-manifest.json",
    )
    upload_parser.add_argument("--s3-uri", required=True)
    upload_parser.add_argument("--kms-key-id", required=True)
    upload_parser.add_argument("--apply", action="store_true")

    stage_parser = aws_commands.add_parser("stage-corpus")
    stage_parser.add_argument(
        "--manifest",
        default="cluster/aws/reasoning-v3-corpus-manifest.json",
    )
    stage_parser.add_argument("--s3-uri", required=True)
    stage_parser.add_argument("--destination", required=True)
    stage_parser.add_argument("--apply", action="store_true")

    verify_parser = aws_commands.add_parser("verify-corpus")
    verify_parser.add_argument(
        "--manifest",
        default="cluster/aws/reasoning-v3-corpus-manifest.json",
    )
    verify_parser.add_argument("--dataset-root", required=True)

    aws_instantiate = aws_commands.add_parser("instantiate")
    aws_instantiate.add_argument("--dataset-root", required=True)
    aws_instantiate.add_argument(
        "--pointer",
        default="DATASET-POINTER-AWS-135M-V3.json",
    )
    aws_instantiate.add_argument(
        "--manifest",
        default="cluster/aws/reasoning-v3-corpus-manifest.json",
    )
    aws_instantiate.add_argument("--profile", required=True)
    aws_instantiate.add_argument("--runtime-root", required=True)
    aws_instantiate.add_argument("--out-root", required=True)
    aws_instantiate.add_argument("--repository-root", default=".")
    aws_instantiate.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "aws":
        if args.aws_command == "upload-corpus":
            result = upload_to_s3(
                args.repository_root,
                args.manifest,
                args.s3_uri,
                kms_key_id=args.kms_key_id,
                apply=args.apply,
            )
        elif args.aws_command == "stage-corpus":
            result = stage_from_s3(
                args.s3_uri,
                args.destination,
                args.manifest,
                apply=args.apply,
            )
        elif args.aws_command == "verify-corpus":
            result = verify_staged_corpus(args.dataset_root, args.manifest)
        else:
            result = instantiate_aws(
                dataset_root=args.dataset_root,
                pointer_path=args.pointer,
                transfer_manifest_path=args.manifest,
                profile_path=args.profile,
                runtime_root=args.runtime_root,
                out_root=args.out_root,
                repository_root=args.repository_root,
                seeds=tuple(args.seeds),
            )
    elif args.command == "runs":
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
