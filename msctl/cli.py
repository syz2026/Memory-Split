"""Single-object JSON command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
from .jsonutil import canonical_sha256
from .operations import (
    cancel_runs,
    check_capacity,
    evaluate_runs,
    load_bound_inputs,
    render_runs,
    resume_runs,
    status_runs,
    submit_runs,
)
from .profile import load_profile


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
    ensure_env.add_argument("--lock", default="requirements-illumina.lock")
    ensure_env.add_argument("--release")
    ensure_env.add_argument("--apply", action="store_true")

    dataset = _leaf(commands, "dataset", help_text="dataset lifecycle")
    dataset_sub = dataset.add_subparsers(dest="action", required=True)
    ensure_dataset = _leaf(
        dataset_sub, "ensure", help_text="plan or stage immutable dataset"
    )
    ensure_dataset.add_argument("--pointer", default="DATASET-POINTER.json")
    ensure_dataset.add_argument("--apply", action="store_true")
    verify_dataset = _leaf(
        dataset_sub, "verify", help_text="verify immutable dataset receipt"
    )
    verify_dataset.add_argument("--pointer", default="DATASET-POINTER.json")
    verify_dataset.add_argument("--dataset-root", required=True)
    verify_dataset.add_argument("--shared-root", required=True)
    verify_dataset.add_argument("--release", required=True)
    verify_dataset.add_argument("--manifest", required=True)
    verify_dataset.add_argument("--verification-out")
    verify_dataset.add_argument("--apply", action="store_true")

    runs = _leaf(commands, "runs", help_text="run plan lifecycle")
    runs_sub = runs.add_subparsers(dest="action", required=True)
    render = _leaf(runs_sub, "render", help_text="render deterministic Slurm argv")
    render.add_argument("--release", required=True)
    render.add_argument("--manifest", required=True)
    render.add_argument("--dataset-pointer", required=True)
    render.add_argument("--shared-root", required=True)
    render_dataset = render.add_mutually_exclusive_group(required=True)
    render_dataset.add_argument("--dataset-root")
    render_dataset.add_argument("--dataset-verification")
    render.add_argument("--environment-receipt")

    for name in ("submit", "resume", "cancel", "evaluate"):
        leaf = _leaf(commands, name, help_text=f"plan or {name} runs")
        leaf.add_argument("--release", required=True)
        leaf.add_argument("--manifest", required=True)
        if name in {"submit", "resume", "evaluate"}:
            leaf.add_argument("--dataset-pointer", required=True)
            leaf.add_argument("--shared-root", required=True)
            dataset_binding = leaf.add_mutually_exclusive_group(required=True)
            dataset_binding.add_argument("--dataset-root")
            dataset_binding.add_argument("--dataset-verification")
            leaf.add_argument("--environment-receipt")
        leaf.add_argument("--approval")
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
    cleanup_plan.add_argument("--root", required=True)
    cleanup_apply = _leaf(cleanup_sub, "apply", help_text="apply frozen cleanup")
    cleanup_apply.add_argument("--plan", required=True)
    cleanup_apply.add_argument("--release", required=True)
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


def dispatch(args: argparse.Namespace) -> tuple[bool, dict[str, object]]:
    profile = load_profile(args.profile)
    command = _command_name(args)
    if command == "auth check":
        return False, _auth_check(profile)
    if command == "capacity check":
        return False, check_capacity(profile, environ=dict(os.environ))
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
            environ=dict(os.environ),
        )
    if command == "env ensure":
        release = load_release(args.release) if args.release else None
        return not args.apply, ensure_environment(
            profile=profile,
            release=release,
            root=args.root,
            lock=args.lock,
            apply=args.apply,
            environ=dict(os.environ),
        )
    if command == "dataset ensure":
        return not args.apply, ensure_dataset(
            profile=profile,
            pointer_path=args.pointer,
            repo_root=args.repo_root,
            apply=args.apply,
            environ=dict(os.environ),
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
            environ=dict(os.environ),
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
            environ=dict(os.environ),
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
            environ=dict(os.environ),
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
            environ=dict(os.environ),
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
            approval_path=args.approval,
            apply=args.apply,
            environ=dict(os.environ),
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
