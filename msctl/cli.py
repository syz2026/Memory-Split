"""Single-object JSON command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from cluster.aws.p5.profile import load_aws_p5_profile

from .aws_p5 import build_aws_backend
from .cleanup import apply_cleanup, make_cleanup_plan
from .collect import collect_evidence
from .contracts import load_release
from .dataset import (
    ensure_dataset,
    verify_dataset,
    write_dataset_verification,
)
from .environment import ensure_environment
from .errors import MsctlError
from .jsonutil import canonical_sha256, load_json, require_object
from .operations import (
    cancel_runs,
    check_capacity,
    evaluate_runs,
    instantiate_run_manifest,
    load_bound_inputs,
    render_runs,
    resume_runs,
    status_runs,
    submit_runs,
)
from .profile import (
    AWS_P5_PROFILE,
    SUPPORTED_PROFILE,
    load_profile,
)


SCHEMA_VERSION = 1
DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = (
    DEFAULT_ROOT / "cluster" / "profiles" / "illumina-usfc-prd.json"
)


class _HelpRequested(Exception):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.text = text


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise MsctlError("CLI_USAGE", message)

    def print_help(self, file=None) -> None:
        raise _HelpRequested(self.format_help())

    def exit(self, status: int = 0, message: str | None = None) -> None:
        if status == 0:
            raise _HelpRequested(message or self.format_help())
        raise MsctlError("CLI_USAGE", (message or "invalid arguments").strip())


def _leaf(
    subparsers,
    name: str,
    *,
    help_text: str,
) -> argparse.ArgumentParser:
    return subparsers.add_parser(name, help=help_text)


def build_parser() -> JsonArgumentParser:
    parser = JsonArgumentParser(
        prog="msctl",
        description="Dry-run-first MemorySplit cluster lifecycle control.",
    )
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--repo-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--state-root", default=".msctl-state")
    commands = parser.add_subparsers(dest="command", required=True)

    auth = _leaf(commands, "auth", help_text="authentication checks")
    auth_sub = auth.add_subparsers(dest="action", required=True)
    _leaf(auth_sub, "check", help_text="check local Illumina authorization")

    capacity = _leaf(commands, "capacity", help_text="capacity checks")
    capacity_sub = capacity.add_subparsers(dest="action", required=True)
    _leaf(capacity_sub, "check", help_text="query bounded Slurm capacity")

    env = _leaf(commands, "env", help_text="environment lifecycle")
    env_sub = env.add_subparsers(dest="action", required=True)
    ensure_env = _leaf(env_sub, "ensure", help_text="plan or build environment")
    ensure_env.add_argument("--root")
    ensure_env.add_argument("--lock")
    ensure_env.add_argument("--release")
    ensure_env.add_argument("--apply", action="store_true")

    dataset = _leaf(commands, "dataset", help_text="dataset lifecycle")
    dataset_sub = dataset.add_subparsers(dest="action", required=True)
    ensure_dataset = _leaf(
        dataset_sub, "ensure", help_text="plan or stage immutable dataset"
    )
    ensure_dataset.add_argument("--pointer", default="DATASET-POINTER.json")
    ensure_dataset.add_argument("--receipt")
    ensure_dataset.add_argument("--apply", action="store_true")
    verify_dataset = _leaf(
        dataset_sub, "verify", help_text="verify immutable dataset receipt"
    )
    verify_dataset.add_argument("--pointer", default="DATASET-POINTER.json")
    verify_dataset.add_argument("--receipt")
    verify_dataset.add_argument("--dataset-root")
    verify_dataset.add_argument("--shared-root")
    verify_dataset.add_argument("--release")
    verify_dataset.add_argument("--manifest")
    verify_dataset.add_argument("--verification-out")
    verify_dataset.add_argument("--apply", action="store_true")

    runs = _leaf(commands, "runs", help_text="run plan lifecycle")
    runs_sub = runs.add_subparsers(dest="action", required=True)
    instantiate = _leaf(
        runs_sub,
        "instantiate",
        help_text="bind one provider-owned seed pair after release",
    )
    instantiate.add_argument("--release", required=True)
    instantiate.add_argument("--dataset-receipt", required=True)
    instantiate.add_argument("--sealed-evaluation-release-sha256")
    instantiate.add_argument("--estimated-instance-hours", type=float)
    instantiate.add_argument("--seed", required=True, type=int)
    instantiate.add_argument("--out", required=True)
    instantiate.add_argument("--apply", action="store_true")
    render = _leaf(runs_sub, "render", help_text="render deterministic Slurm argv")
    render.add_argument("--release", required=True)
    render.add_argument("--manifest", required=True)
    render.add_argument("--dataset-pointer")
    render_dataset = render.add_mutually_exclusive_group()
    render.add_argument("--shared-root")
    render_dataset.add_argument("--dataset-root")
    render_dataset.add_argument("--dataset-verification")
    render.add_argument("--environment-receipt")

    for name in ("submit", "resume", "cancel", "evaluate"):
        leaf = _leaf(commands, name, help_text=f"plan or {name} runs")
        leaf.add_argument("--release", required=True)
        leaf.add_argument("--manifest", required=True)
        if name in {"submit", "resume", "evaluate"}:
            leaf.add_argument("--dataset-pointer")
            dataset_binding = leaf.add_mutually_exclusive_group()
            leaf.add_argument("--shared-root")
            dataset_binding.add_argument("--dataset-root")
            dataset_binding.add_argument("--dataset-verification")
            leaf.add_argument("--environment-receipt")
        leaf.add_argument("--approval")
        if name == "submit":
            leaf.add_argument("--instance-id")
            leaf.add_argument("--terminate-at")
        if name == "resume":
            leaf.add_argument("--checkpoint-receipt", required=True)
        leaf.add_argument("--apply", action="store_true")

    status = _leaf(commands, "status", help_text="reconcile run status")
    status.add_argument("--release", required=True)
    status.add_argument("--manifest", required=True)
    status.add_argument("--cached", action="store_true")

    collect = _leaf(commands, "collect", help_text="collect result evidence")
    collect.add_argument("--source", required=True)
    collect.add_argument("--out", required=True)
    collect.add_argument("--apply", action="store_true")

    cleanup = _leaf(commands, "cleanup", help_text="safe cleanup lifecycle")
    cleanup_sub = cleanup.add_subparsers(dest="action", required=True)
    cleanup_plan = _leaf(cleanup_sub, "plan", help_text="render cleanup plan")
    cleanup_plan.add_argument("--root")
    cleanup_plan.add_argument("--release")
    cleanup_plan.add_argument("--manifest")
    cleanup_apply = _leaf(cleanup_sub, "apply", help_text="apply frozen cleanup")
    cleanup_apply.add_argument("--plan")
    cleanup_apply.add_argument("--release", required=True)
    cleanup_apply.add_argument("--manifest")
    cleanup_apply.add_argument("--approval")
    cleanup_apply.add_argument("--apply", action="store_true", required=True)
    return parser


def _command_name(args: argparse.Namespace) -> str:
    action = getattr(args, "action", None)
    return f"{args.command} {action}" if action else str(args.command)


def _unsupported(command: str) -> None:
    raise MsctlError(
        "EXTERNAL_OPERATION_UNSUPPORTED",
        "this external operation has no safe local implementation",
        details={"operation": command},
    )


def _require_cli_values(args: argparse.Namespace, *names: str) -> None:
    missing = [name for name in names if getattr(args, name, None) is None]
    if missing:
        raise MsctlError(
            "CLI_USAGE",
            "missing provider-specific arguments",
            details={
                "missing": [
                    f"--{name.replace('_', '-')}" for name in missing
                ]
            },
        )


def _load_cli_profile(path: Path | str) -> object:
    value = require_object(load_json(path, label="profile"), label="profile")
    if (
        value.get("provider") == AWS_P5_PROFILE
        and value.get("profile_id") == "aws-p5.48xlarge-v3"
    ):
        try:
            return load_aws_p5_profile(path)
        except (OSError, TypeError, ValueError) as error:
            raise MsctlError(
                "PROFILE_INVALID",
                "AWS P5 v3 profile validation failed",
            ) from error
    return load_profile(path)


def _auth_check(profile) -> dict[str, object]:
    value = os.environ.get(profile.shared_root_env)
    if not value:
        raise MsctlError(
            "AUTH_UNAVAILABLE",
            f"{profile.shared_root_env} is not set",
        )
    root = Path(value)
    try:
        root.resolve().relative_to(Path(profile.shared_root_prefix).resolve())
    except ValueError as error:
        raise MsctlError(
            "AUTH_INVALID",
            "shared root is outside the approved /illumina prefix",
        ) from error
    if not root.is_dir() or root.is_symlink() or not os.access(root, os.R_OK):
        raise MsctlError(
            "AUTH_UNAVAILABLE",
            "shared root is not a readable regular directory",
        )
    return {
        "provider": profile.provider,
        "user": os.environ.get("USER"),
        "shared_root": str(root),
        "readable": True,
    }


def dispatch(
    args: argparse.Namespace,
    *,
    profile_loader: Callable[[Path | str], object] | None = None,
    cohort_loader: Callable[[Path | str], object] | None = None,
    aws_backend_factory: Callable[..., object] = build_aws_backend,
    environ: dict[str, str] | None = None,
) -> tuple[bool, dict[str, object]]:
    command = _command_name(args)
    default_profile_loader = (
        _load_cli_profile if command == "runs instantiate" else load_profile
    )
    profile = (profile_loader or default_profile_loader)(args.profile)
    environment = dict(os.environ if environ is None else environ)
    provider = getattr(profile, "provider", None)
    if provider not in {SUPPORTED_PROFILE, AWS_P5_PROFILE}:
        raise MsctlError(
            "PROVIDER_UNSUPPORTED",
            "profile provider is not supported",
            details={"provider": provider},
        )
    if command == "runs instantiate":
        if (
            provider == AWS_P5_PROFILE
            and getattr(profile, "profile_id", None)
            == "aws-p5.48xlarge-v3"
        ):
            _require_cli_values(
                args,
                "sealed_evaluation_release_sha256",
                "estimated_instance_hours",
            )
        return not args.apply, instantiate_run_manifest(
            profile=profile,
            release_path=args.release,
            dataset_receipt=args.dataset_receipt,
            seed=args.seed,
            out=args.out,
            repo_root=args.repo_root,
            apply=args.apply,
            sealed_evaluation_release_sha256=(
                args.sealed_evaluation_release_sha256
            ),
            estimated_instance_hours=args.estimated_instance_hours,
            cohort_loader=cohort_loader,
        )
    if provider == AWS_P5_PROFILE:
        if command == "submit":
            _require_cli_values(args, "instance_id", "terminate_at")
        if command in {"runs render", "submit", "resume", "evaluate"}:
            _require_cli_values(
                args,
                "dataset_pointer",
                "environment_receipt",
            )
            if (args.dataset_root is None) == (
                args.dataset_verification is None
            ):
                raise MsctlError(
                    "CLI_USAGE",
                    "AWS requires exactly one dataset source",
                    details={
                        "required_one_of": [
                            "--dataset-root",
                            "--dataset-verification",
                        ]
                    },
                )
        if command in {"dataset ensure", "dataset verify"}:
            _require_cli_values(args, "receipt")
        backend = aws_backend_factory(
            profile=profile,
            state_root=args.state_root,
            environ=environment,
        )
        return backend.dispatch(command, args)
    if command in {"runs render", "submit", "resume", "evaluate"}:
        _require_cli_values(args, "dataset_pointer", "shared_root")
        if (args.dataset_root is None) == (args.dataset_verification is None):
            raise MsctlError(
                "CLI_USAGE",
                "Illumina requires exactly one dataset source",
                details={
                    "required_one_of": [
                        "--dataset-root",
                        "--dataset-verification",
                    ]
                },
            )
    if command == "dataset verify":
        _require_cli_values(
            args,
            "dataset_root",
            "shared_root",
            "release",
            "manifest",
        )
    if command == "cleanup plan":
        _require_cli_values(args, "root")
    if command == "cleanup apply":
        _require_cli_values(args, "plan", "manifest")
    if command == "auth check":
        return False, _auth_check(profile)
    if command == "capacity check":
        return False, check_capacity(profile, environ=environment)
    if command == "runs render":
        return True, render_runs(
            profile=profile,
            release_path=args.release,
            manifest_path=args.manifest,
            dataset_pointer=args.dataset_pointer,
            shared_root=args.shared_root,
            dataset_root=args.dataset_root,
            dataset_verification=args.dataset_verification,
            environment_receipt=args.environment_receipt,
            repo_root=args.repo_root,
        )
    if command == "submit":
        return not args.apply, submit_runs(
            profile=profile,
            release_path=args.release,
            manifest_path=args.manifest,
            dataset_pointer=args.dataset_pointer,
            shared_root=args.shared_root,
            dataset_root=args.dataset_root,
            dataset_verification=args.dataset_verification,
            environment_receipt=args.environment_receipt,
            repo_root=args.repo_root,
            state_root=args.state_root,
            approval_path=args.approval,
            apply=args.apply,
            environ=environment,
        )
    if command == "env ensure":
        release = load_release(args.release) if args.release else None
        return not args.apply, ensure_environment(
            profile=profile,
            release=release,
            root=args.root,
            lock=args.lock or profile.environment_lock,
            apply=args.apply,
            environ=dict(os.environ),
        )
    if command == "dataset ensure":
        return not args.apply, ensure_dataset(
            profile=profile,
            pointer_path=args.pointer,
            repo_root=args.repo_root,
            apply=args.apply,
            environ=environment,
        )
    if command == "dataset verify":
        release, manifest, _ = load_bound_inputs(
            profile=profile,
            release_path=args.release,
            manifest_path=args.manifest,
            repo_root=args.repo_root,
        )
        result = verify_dataset(
            profile=profile,
            pointer_path=args.pointer,
            dataset_root=args.dataset_root,
            approved_shared_root=args.shared_root,
            release=release,
            manifest=manifest,
            repo_root=args.repo_root,
        )
        if args.verification_out and args.apply:
            write_dataset_verification(args.verification_out, result)
        return bool(args.verification_out and not args.apply), result
    if command == "status":
        return False, status_runs(
            profile=profile,
            release_path=args.release,
            manifest_path=args.manifest,
            repo_root=args.repo_root,
            state_root=args.state_root,
            cached=args.cached,
            environ=environment,
        )
    if command == "resume":
        return not args.apply, resume_runs(
            profile=profile,
            release_path=args.release,
            manifest_path=args.manifest,
            checkpoint_receipt=args.checkpoint_receipt,
            dataset_pointer=args.dataset_pointer,
            shared_root=args.shared_root,
            dataset_root=args.dataset_root,
            dataset_verification=args.dataset_verification,
            environment_receipt=args.environment_receipt,
            repo_root=args.repo_root,
            state_root=args.state_root,
            approval_path=args.approval,
            apply=args.apply,
            environ=environment,
        )
    if command == "cancel":
        return not args.apply, cancel_runs(
            profile=profile,
            release_path=args.release,
            manifest_path=args.manifest,
            repo_root=args.repo_root,
            state_root=args.state_root,
            approval_path=args.approval,
            apply=args.apply,
            environ=environment,
        )
    if command == "evaluate":
        return not args.apply, evaluate_runs(
            profile=profile,
            release_path=args.release,
            manifest_path=args.manifest,
            dataset_pointer=args.dataset_pointer,
            shared_root=args.shared_root,
            dataset_root=args.dataset_root,
            dataset_verification=args.dataset_verification,
            environment_receipt=args.environment_receipt,
            repo_root=args.repo_root,
            state_root=args.state_root,
            approval_path=args.approval,
            apply=args.apply,
            environ=environment,
        )
    if command == "collect":
        return not args.apply, collect_evidence(
            source=args.source,
            out=args.out,
            apply=args.apply,
        )
    if command == "cleanup plan":
        plan = make_cleanup_plan(args.root)
        return True, {
            "operation": command,
            "plan": plan,
            "plan_sha256": canonical_sha256(plan),
        }
    if command == "cleanup apply":
        return False, apply_cleanup(
            profile=profile,
            plan_path=args.plan,
            release_path=args.release,
            manifest_path=args.manifest,
            repo_root=args.repo_root,
            approval_path=args.approval,
            apply=args.apply,
            environ=environment,
        )
    _unsupported(command)
    raise AssertionError("unreachable")


def _emit(value: dict[str, object]) -> None:
    sys.stdout.write(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    command = "unknown"
    try:
        args = build_parser().parse_args(argv)
        command = _command_name(args)
        dry_run, result = dispatch(args)
        report = {
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "command": command,
            "dry_run": dry_run,
            "result": result,
        }
        exit_code = 0
    except _HelpRequested as help_request:
        report = {
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "command": "help",
            "dry_run": True,
            "result": {"help": help_request.text},
        }
        exit_code = 0
    except MsctlError as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "command": command,
            "error": error.as_dict(),
        }
        exit_code = error.exit_code
    except Exception:
        report = {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "command": command,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "unexpected local failure",
                "details": {},
            },
        }
        exit_code = 70
    _emit(report)
    return exit_code
