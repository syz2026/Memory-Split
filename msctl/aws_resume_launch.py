"""Bind verified checkpoints to the reviewed AWS paired launcher."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster.aws.p5 import launch_seed_pair as reviewed_launcher
from msctl.errors import MsctlError
from msctl.fsutil import open_directory, rename_noreplace_at


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARMS = ("dense", "split90")
_CHECKPOINT_FIELDS = {
    "arm",
    "resume_path",
    "resume_sha256",
    "world_size",
}
_RECEIPT_FIELDS = {
    "schema_version",
    "provider",
    "release_sha256",
    "run_manifest_sha256",
    "dataset_sha256",
    "source_commit",
    "checkpoints",
}
_RECEIPT_CHECKPOINT_FIELDS = {
    "run_id",
    "arm",
    "seed",
    "path",
    "sha256",
    "config_sha256",
    "dataset_sha256",
    "source_commit",
    "step",
    "world_size",
}


class ResumeLaunchError(ValueError):
    """The checkpoint-to-launcher handoff is invalid."""


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ResumeLaunchError(f"checkpoint binding repeats field: {key}")
        value[key] = item
    return value


def _checkpoint_argument(value: str) -> dict[str, object]:
    try:
        binding = json.loads(
            value,
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ResumeLaunchError(
                    f"checkpoint binding contains non-finite {constant}"
                )
            ),
        )
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            "checkpoint binding must be JSON"
        ) from error
    if not isinstance(binding, dict):
        raise argparse.ArgumentTypeError(
            "checkpoint binding must be an object"
        )
    return binding


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ResumeLaunchError(f"{label} must be lowercase SHA-256")
    return value


def _hash_regular(path: Path, *, label: str) -> str:
    try:
        before = path.stat(follow_symlinks=False)
        if (
            path.is_symlink()
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            raise ResumeLaunchError(
                f"{label} must be one singly linked regular file"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        after = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ResumeLaunchError(f"{label} is unavailable") from error
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ResumeLaunchError(f"{label} changed while hashing")
    return digest


def bind_resume_checkpoints(
    plan: reviewed_launcher.LaunchPlan,
    *,
    checkpoint_receipt_sha256: str,
    checkpoints: Sequence[Mapping[str, object]],
    launcher_module: ModuleType = reviewed_launcher,
    make_read_only: bool = False,
) -> reviewed_launcher.LaunchPlan:
    """Return a reviewed launch plan with two receipt-bound resume mounts."""

    receipt_sha256 = _sha256(
        checkpoint_receipt_sha256,
        label="checkpoint receipt",
    )
    if len(checkpoints) != 2:
        raise ResumeLaunchError("resume requires one complete checkpoint pair")
    by_arm: dict[str, Mapping[str, object]] = {}
    verified: list[object] = []
    checkpoint_root = (
        Path(plan.scratch_root) / "staging" / "resume" / receipt_sha256
    ).resolve(strict=True)
    for row in checkpoints:
        if not isinstance(row, Mapping) or set(row) != _CHECKPOINT_FIELDS:
            raise ResumeLaunchError("checkpoint binding fields do not match")
        arm = row["arm"]
        if (
            arm not in _ARMS
            or arm in by_arm
            or type(row["world_size"]) is not int
            or row["world_size"] != 4
        ):
            raise ResumeLaunchError(
                "resume requires distinct Dense/Split90 world-size-4 checkpoints"
            )
        digest = _sha256(
            row["resume_sha256"],
            label=f"{arm} checkpoint",
        )
        if not isinstance(row["resume_path"], str):
            raise ResumeLaunchError("checkpoint path must be a string")
        path = Path(row["resume_path"])
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(checkpoint_root)
        except (OSError, ValueError) as error:
            raise ResumeLaunchError(
                "checkpoint path is outside its receipt-bound staging root"
            ) from error
        if any(character in str(resolved) for character in ",\n\r\x00"):
            raise ResumeLaunchError("checkpoint path is unsafe for a bind mount")
        if _hash_regular(resolved, label=f"{arm} checkpoint") != digest:
            raise ResumeLaunchError(f"{arm} checkpoint SHA-256 does not match")
        if make_read_only:
            os.chmod(resolved, 0o444)
        by_arm[str(arm)] = row
        verified.append(
            launcher_module.VerifiedFile(path=resolved, sha256=digest)
        )
    if set(by_arm) != set(_ARMS):
        raise ResumeLaunchError("resume checkpoint pair is incomplete")

    rebound = []
    for launch in plan.arms:
        binding = by_arm.get(launch.arm)
        if binding is None:
            raise ResumeLaunchError("launcher arm has no checkpoint binding")
        argv = list(launch.argv)
        if (
            len(argv) < 2
            or argv[-2:] != ["--resume", "none"]
            or argv.count(plan.container_image) != 1
        ):
            raise ResumeLaunchError(
                "reviewed launcher does not expose the expected resume boundary"
            )
        image_index = argv.index(plan.container_image)
        source = str(Path(str(binding["resume_path"])).resolve(strict=True))
        mount = (
            f"type=bind,src={source},dst=/resume/checkpoint.pt,readonly"
        )
        argv[image_index:image_index] = ["--mount", mount]
        argv[-2:] = [
            "--resume-path",
            "/resume/checkpoint.pt",
            "--resume-sha256",
            str(binding["resume_sha256"]),
        ]
        rebound.append(dataclasses.replace(launch, argv=tuple(argv)))
    return dataclasses.replace(
        plan,
        arms=tuple(rebound),
        verified_files=tuple(plan.verified_files) + tuple(verified),
    )


def _require_directory(path: Path, *, label: str) -> None:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ResumeLaunchError(f"{label} is unavailable") from error
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ResumeLaunchError(f"{label} must be a real directory")


def _move_directory_noreplace(
    source: Path,
    destination: Path,
    *,
    label: str,
) -> None:
    source_fd = open_directory(source.parent, label=f"{label} source parent")
    destination_fd = open_directory(
        destination.parent,
        label=f"{label} destination parent",
        create=True,
    )
    try:
        try:
            rename_noreplace_at(
                source_fd,
                source.name,
                destination_fd,
                destination.name,
            )
        except FileExistsError as error:
            raise ResumeLaunchError(
                f"{label} archive already exists"
            ) from error
        os.fsync(source_fd)
        os.fsync(destination_fd)
    finally:
        os.close(destination_fd)
        os.close(source_fd)


def prepare_resume_output_roots(
    scratch_root: Path,
    *,
    seed: int,
    checkpoint_receipt_sha256: str,
) -> Path:
    """Atomically archive a prior paired output before a resume attempt."""

    if type(seed) is not int or seed not in {1, 2, 3, 4}:
        raise ResumeLaunchError("resume seed is not assigned to AWS")
    receipt_sha256 = _sha256(
        checkpoint_receipt_sha256,
        label="checkpoint receipt",
    )
    try:
        scratch = Path(scratch_root).resolve(strict=True)
    except OSError as error:
        raise ResumeLaunchError("scratch root is unavailable") from error
    _require_directory(scratch, label="scratch root")
    archive = (
        scratch
        / "resume-history"
        / f"seed-{seed}"
        / receipt_sha256
    )
    try:
        archive_fd = open_directory(
            archive,
            label="resume archive",
            create=True,
        )
    except (MsctlError, OSError, ValueError) as error:
        raise ResumeLaunchError(
            "resume archive path is unsafe"
        ) from error
    try:
        os.fchmod(archive_fd, 0o700)
    finally:
        os.close(archive_fd)
    _require_directory(archive, label="resume archive")

    source_runs = scratch / "runs" / f"seed-{seed}"
    archived_runs = archive / "runs"
    source_exists = source_runs.exists() or source_runs.is_symlink()
    archived_exists = archived_runs.exists() or archived_runs.is_symlink()
    if source_exists and archived_exists:
        raise ResumeLaunchError(
            "both active and archived paired outputs exist"
        )
    if source_exists:
        _require_directory(source_runs, label="paired output root")
        for arm in _ARMS:
            _require_directory(
                source_runs / arm,
                label=f"{arm} output root",
            )
        _move_directory_noreplace(
            source_runs,
            archived_runs,
            label="paired output",
        )
    elif archived_exists:
        _require_directory(archived_runs, label="archived paired output")
        for arm in _ARMS:
            _require_directory(
                archived_runs / arm,
                label=f"archived {arm} output root",
            )

    source_cids = (
        scratch / "staging" / "container-cids" / f"seed-{seed}"
    )
    archived_cids = archive / "container-cids"
    source_cids_exist = source_cids.exists() or source_cids.is_symlink()
    archived_cids_exist = archived_cids.exists() or archived_cids.is_symlink()
    if source_cids_exist and archived_cids_exist:
        raise ResumeLaunchError(
            "both active and archived container identities exist"
        )
    if source_cids_exist:
        _require_directory(source_cids, label="container identity root")
        _move_directory_noreplace(
            source_cids,
            archived_cids,
            label="container identity",
        )
    elif archived_cids_exist:
        _require_directory(
            archived_cids,
            label="archived container identity root",
        )
    return archive


def _verify_checkpoint_receipt(
    path: Path,
    *,
    expected_sha256: str,
    run_manifest_sha256: str,
    plan: reviewed_launcher.LaunchPlan,
    checkpoints: Sequence[Mapping[str, object]],
) -> None:
    expected = _sha256(expected_sha256, label="checkpoint receipt")
    manifest_sha256 = _sha256(
        run_manifest_sha256,
        label="run manifest",
    )
    try:
        if path.is_symlink() or not path.is_file():
            raise ResumeLaunchError(
                "checkpoint receipt must be one regular file"
            )
        payload = path.read_bytes()
    except OSError as error:
        raise ResumeLaunchError("checkpoint receipt is unavailable") from error
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ResumeLaunchError("checkpoint receipt SHA-256 does not match")
    try:
        receipt = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ResumeLaunchError(
                    f"checkpoint receipt contains non-finite {constant}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResumeLaunchError(
            "checkpoint receipt is not valid UTF-8 JSON"
        ) from error
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_FIELDS:
        raise ResumeLaunchError("checkpoint receipt fields do not match")
    if (
        receipt["schema_version"] != 2
        or receipt["provider"] != "aws-p5.48xlarge"
        or receipt["release_sha256"] != plan.release_sha256
        or receipt["run_manifest_sha256"] != manifest_sha256
        or receipt["dataset_sha256"] != plan.corpus_receipt_sha256
        or receipt["source_commit"] != plan.code_commit
    ):
        raise ResumeLaunchError(
            "checkpoint receipt does not bind the reviewed launch plan"
        )
    rows = receipt["checkpoints"]
    if not isinstance(rows, list) or len(rows) != 2:
        raise ResumeLaunchError("checkpoint receipt pair is incomplete")
    receipt_by_arm: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != _RECEIPT_CHECKPOINT_FIELDS:
            raise ResumeLaunchError(
                "checkpoint receipt row fields do not match"
            )
        arm = row["arm"]
        if (
            arm not in _ARMS
            or arm in receipt_by_arm
            or row["seed"] != plan.seed
            or row["dataset_sha256"] != plan.corpus_receipt_sha256
            or row["source_commit"] != plan.code_commit
            or type(row["world_size"]) is not int
            or row["world_size"] != 4
            or type(row["step"]) is not int
            or row["step"] <= 0
        ):
            raise ResumeLaunchError(
                "checkpoint receipt row provenance is invalid"
            )
        _sha256(row["sha256"], label=f"{arm} checkpoint")
        _sha256(row["config_sha256"], label=f"{arm} config")
        receipt_by_arm[str(arm)] = row
    if set(receipt_by_arm) != set(_ARMS):
        raise ResumeLaunchError("checkpoint receipt arms are incomplete")
    binding_by_arm = {str(row.get("arm")): row for row in checkpoints}
    launch_by_arm = {launch.arm: launch for launch in plan.arms}
    for arm in _ARMS:
        row = receipt_by_arm[arm]
        binding = binding_by_arm.get(arm)
        launch = launch_by_arm.get(arm)
        if (
            binding is None
            or launch is None
            or row["sha256"] != binding.get("resume_sha256")
            or row["world_size"] != binding.get("world_size")
            or row["config_sha256"] != launch.config_sha256
        ):
            raise ResumeLaunchError(
                "checkpoint receipt differs from the executable binding"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt-sha256", required=True)
    parser.add_argument("--run-manifest-sha256", required=True)
    parser.add_argument(
        "--checkpoint",
        type=_checkpoint_argument,
        action="append",
        required=True,
    )
    parser.add_argument("--apply", action="store_true")
    return parser


def _verify_launcher_path(launcher: Path, repo_root: Path) -> None:
    expected = (
        repo_root.resolve(strict=True)
        / "cluster"
        / "aws"
        / "p5"
        / "launch_seed_pair.py"
    )
    module_path = Path(reviewed_launcher.__file__).resolve(strict=True)
    try:
        actual = launcher.resolve(strict=True)
    except OSError as error:
        raise ResumeLaunchError("reviewed launcher is unavailable") from error
    if (
        launcher.is_symlink()
        or not launcher.is_file()
        or actual != expected
        or module_path != expected
    ):
        raise ResumeLaunchError(
            "resume must use the authenticated reviewed paired launcher"
        )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        _verify_launcher_path(arguments.launcher, arguments.repo_root)
        plan = reviewed_launcher.load_launch_plan(
            seed=arguments.seed,
            manifest_path=arguments.manifest,
            profile_path=arguments.profile,
            repo_root=arguments.repo_root,
            scratch_root=arguments.scratch_root,
            environment=os.environ,
        )
        _verify_checkpoint_receipt(
            arguments.checkpoint_receipt,
            expected_sha256=arguments.checkpoint_receipt_sha256,
            run_manifest_sha256=arguments.run_manifest_sha256,
            plan=plan,
            checkpoints=arguments.checkpoint,
        )
        plan = bind_resume_checkpoints(
            plan,
            checkpoint_receipt_sha256=arguments.checkpoint_receipt_sha256,
            checkpoints=arguments.checkpoint,
            make_read_only=arguments.apply,
        )
        if not arguments.apply:
            report = reviewed_launcher.render_plan(plan)
            print(json.dumps(report, sort_keys=True, separators=(",", ":")))
            return 0
        prepare_resume_output_roots(
            arguments.scratch_root,
            seed=arguments.seed,
            checkpoint_receipt_sha256=(
                arguments.checkpoint_receipt_sha256
            ),
        )
        client = reviewed_launcher.ImdsV2Client()
        with reviewed_launcher.installed_shutdown_handlers() as shutdown_source:
            result = reviewed_launcher.supervise_pair(
                plan,
                notice_source=client.interruption_notice,
                interruption_handler=(
                    reviewed_launcher._production_interruption_handler
                ),
                shutdown_source=shutdown_source,
            )
        report = {
            "child_pids": dict(sorted(result.child_pids.items())),
            "dry_run": False,
            "failed_arm": result.failed_arm,
            "interruption_receipt": result.interruption_receipt,
            "ok": result.returncode == 0,
            "peer_terminated": result.peer_terminated,
            "resumable": result.resumable,
            "returncode": result.returncode,
            "schema_version": 1,
            "seed": plan.seed,
            "status": result.status,
        }
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return result.returncode
    except (ResumeLaunchError, ValueError, OSError) as error:
        print(
            json.dumps(
                {
                    "error": str(error),
                    "ok": False,
                    "schema_version": 1,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
