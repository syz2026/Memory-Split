#!/usr/bin/env python3
"""Verify and stage one immutable AWS P5 execution environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cluster.aws.p5.interruption_checkpoint import (
    CommandResult,
    ImdsV2Client,
    S3ObjectStore,
)
from cluster.aws.p5.profile import (
    AwsP5Profile,
    AwsP5Runtime,
    load_aws_p5_profile,
    validate_runtime_environment,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_H100_RE = re.compile(r"^NVIDIA H100 80GB(?: HBM3)?$")
_CONTAINER_IMAGE_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9./:_-]*@sha256:[0-9a-f]{64}$"
)


class BootstrapError(ValueError):
    """A fail-closed bootstrap validation error."""


@dataclass(frozen=True)
class BootstrapEvidence:
    instance_id: str
    instance_type: str
    ami_id: str
    gpu_names: tuple[str, ...]
    fabric_manager_active: bool
    instance_store_devices: tuple[str, ...]
    container_image: str


@dataclass(frozen=True)
class BootstrapArtifacts:
    release_sha256: str
    release_receipt_sha256: str
    corpus_receipt_sha256: str
    cohort_assignment_sha256: str
    code_commit: str


def _run_command(
    argv: Sequence[str],
    environment: Mapping[str, str],
) -> CommandResult:
    from cluster.aws.p5.interruption_checkpoint import _default_runner

    return _default_runner(argv, environment)


def _checked(
    runner: Callable[[Sequence[str], Mapping[str, str]], CommandResult],
    argv: list[str],
    environment: Mapping[str, str],
    *,
    operation: str,
) -> CommandResult:
    result = runner(argv, environment)
    if result.returncode != 0:
        raise BootstrapError(f"{operation} failed")
    return result


def _metadata_value(
    metadata_get: Callable[[str], str | None],
    path: str,
) -> str:
    value = metadata_get(path)
    if not isinstance(value, str) or not value:
        raise BootstrapError(f"IMDSv2 did not return {path}")
    return value


def inspect_hardware(
    profile: AwsP5Profile,
    runtime: AwsP5Runtime,
    *,
    metadata_get: Callable[[str], str | None],
    runner: Callable[
        [Sequence[str], Mapping[str, str]], CommandResult
    ] = _run_command,
    command_environment: Mapping[str, str],
    container_image: str,
) -> BootstrapEvidence:
    """Verify actual P5 identity, accelerators, Fabric Manager, NVMe, and image."""

    instance_id = _metadata_value(metadata_get, "meta-data/instance-id")
    if re.fullmatch(r"i-[0-9a-f]{8,17}", instance_id) is None:
        raise BootstrapError("IMDSv2 instance ID is invalid")
    instance_type = _metadata_value(metadata_get, "meta-data/instance-type")
    if instance_type != profile.instance_type:
        raise BootstrapError("instance must be exactly p5.48xlarge")
    ami_id = _metadata_value(metadata_get, "meta-data/ami-id")
    if ami_id != runtime.ami_id:
        raise BootstrapError("running AMI does not match MS_AWS_AMI_ID")

    gpu_result = _checked(
        runner,
        [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
        ],
        command_environment,
        operation="GPU discovery",
    )
    gpu_names = tuple(
        line.strip() for line in gpu_result.stdout.splitlines() if line.strip()
    )
    if len(gpu_names) != profile.allocated_gpus:
        raise BootstrapError("exactly eight H100 devices are required")
    if any(_H100_RE.fullmatch(name) is None for name in gpu_names):
        raise BootstrapError("every accelerator must be an NVIDIA H100 80GB")

    fabric = runner(
        ["systemctl", "is-active", "nvidia-fabricmanager"],
        command_environment,
    )
    if fabric.returncode != 0 or fabric.stdout.strip() != "active":
        raise BootstrapError("NVIDIA Fabric Manager must be active")

    block_result = _checked(
        runner,
        [
            "lsblk",
            "--json",
            "--bytes",
            "--output",
            "PATH,TYPE,MODEL,SIZE,MOUNTPOINTS",
        ],
        command_environment,
        operation="instance-store discovery",
    )
    try:
        block_value = json.loads(block_result.stdout)
    except json.JSONDecodeError as error:
        raise BootstrapError("lsblk returned invalid JSON") from error
    if (
        not isinstance(block_value, dict)
        or not isinstance(block_value.get("blockdevices"), list)
    ):
        raise BootstrapError("lsblk JSON is missing blockdevices")
    devices = []
    for row in block_value["blockdevices"]:
        if not isinstance(row, dict):
            raise BootstrapError("lsblk device rows must be objects")
        if row.get("model") != profile.instance_store_model:
            continue
        path = row.get("path")
        mountpoints = row.get("mountpoints")
        size = row.get("size")
        if (
            row.get("type") != "disk"
            or not isinstance(path, str)
            or not path.startswith("/dev/")
            or any(character in path for character in "\x00\n\r\t ")
            or type(size) is not int
            or size < profile.instance_store_device_bytes
            or not isinstance(mountpoints, list)
            or any(point not in {None, ""} for point in mountpoints)
        ):
            raise BootstrapError("instance-store device metadata is unsafe")
        devices.append(path)
    if len(devices) != profile.instance_store_devices:
        raise BootstrapError(
            "exactly eight unmounted instance-store devices are required"
        )
    if len(set(devices)) != len(devices):
        raise BootstrapError("instance-store device paths must be unique")
    devices.sort()

    if (
        not isinstance(container_image, str)
        or _CONTAINER_IMAGE_RE.fullmatch(container_image) is None
        or not container_image.endswith("@" + runtime.container_digest)
    ):
        raise BootstrapError("container image must use the expected digest")
    image_result = _checked(
        runner,
        [
            "docker",
            "image",
            "inspect",
            "--format={{json .RepoDigests}}",
            container_image,
        ],
        command_environment,
        operation="container image inspection",
    )
    try:
        repo_digests = json.loads(image_result.stdout)
    except json.JSONDecodeError as error:
        raise BootstrapError("container inspection returned invalid JSON") from error
    if not isinstance(repo_digests, list) or container_image not in repo_digests:
        raise BootstrapError("local container does not match the immutable digest")

    return BootstrapEvidence(
        instance_id=instance_id,
        instance_type=instance_type,
        ami_id=ami_id,
        gpu_names=gpu_names,
        fabric_manager_active=True,
        instance_store_devices=tuple(devices),
        container_image=container_image,
    )


def render_bootstrap_commands(
    profile: AwsP5Profile,
    runtime: AwsP5Runtime,
    evidence: BootstrapEvidence,
    *,
    owner_uid: int,
    owner_gid: int,
) -> list[list[str]]:
    """Render the mutation plan as argv arrays only."""

    if type(owner_uid) is not int or owner_uid < 0:
        raise BootstrapError("owner UID must be a nonnegative integer")
    if type(owner_gid) is not int or owner_gid < 0:
        raise BootstrapError("owner GID must be a nonnegative integer")
    if len(evidence.instance_store_devices) != profile.instance_store_devices:
        raise BootstrapError("bootstrap evidence has an incomplete NVMe set")
    scratch = profile.scratch_root
    raid_device = "/dev/md/memorysplit"
    return [
        [
            "mdadm",
            "--create",
            raid_device,
            "--run",
            "--level=0",
            f"--raid-devices={profile.instance_store_devices}",
            *evidence.instance_store_devices,
        ],
        ["mkfs.xfs", "-f", raid_device],
        [
            "install",
            "-d",
            "-m",
            "0700",
            "-o",
            str(owner_uid),
            "-g",
            str(owner_gid),
            scratch,
        ],
        ["mount", "-o", "noatime,nodiratime", raid_device, scratch],
        ["chown", f"{owner_uid}:{owner_gid}", scratch],
        ["chmod", "0700", scratch],
        [
            "install",
            "-d",
            "-m",
            "0700",
            "-o",
            str(owner_uid),
            "-g",
            str(owner_gid),
            f"{scratch}/release",
            f"{scratch}/dataset",
            f"{scratch}/runs",
            f"{scratch}/staging",
        ],
        [
            "aws",
            "s3",
            "sync",
            f"{runtime.s3_root}/releases",
            f"{scratch}/release",
            "--only-show-errors",
            "--no-progress",
            "--region",
            runtime.region,
        ],
        [
            "aws",
            "s3",
            "sync",
            f"{runtime.s3_root}/dataset",
            f"{scratch}/dataset",
            "--only-show-errors",
            "--no-progress",
            "--region",
            runtime.region,
        ],
    ]


def _required_sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise BootstrapError(f"{label} must be a lowercase SHA-256")
    return value


def _regular_digest(
    path: Path,
    *,
    expected: str,
    label: str,
) -> str:
    _required_sha256(expected, label=label)
    if path.is_symlink() or not path.is_file():
        raise BootstrapError(f"{label} must be a regular file")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise BootstrapError(f"{label} SHA-256 mismatch")
    return digest


def verify_bootstrap_artifacts(
    *,
    release_archive: Path,
    release_sha256: str,
    release_receipt: Path,
    release_receipt_sha256: str,
    dataset_receipt: Path,
    dataset_receipt_sha256: str,
    cohort_assignment: Path,
    cohort_assignment_sha256: str,
    code_commit: str,
) -> BootstrapArtifacts:
    """Hash ZIP, release, dataset, and cohort bytes and cross-check bindings."""

    release_digest = _regular_digest(
        release_archive,
        expected=release_sha256,
        label="release archive",
    )
    release_receipt_digest = _regular_digest(
        release_receipt,
        expected=release_receipt_sha256,
        label="release receipt",
    )
    corpus_digest = _regular_digest(
        dataset_receipt,
        expected=dataset_receipt_sha256,
        label="dataset receipt",
    )
    cohort_digest = _regular_digest(
        cohort_assignment,
        expected=cohort_assignment_sha256,
        label="cohort assignment",
    )
    if not isinstance(code_commit, str) or _COMMIT_RE.fullmatch(code_commit) is None:
        raise BootstrapError("code commit must be 40 lowercase hex characters")
    try:
        release_text = release_receipt.read_bytes().decode("utf-8")
        release_value = json.loads(release_text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BootstrapError("release receipt must contain valid JSON") from error
    try:
        bound_archive = release_value["archive"]["sha256"]
        bound_commit = release_value["source"]["commit"]
        bound_cohort = release_value["cohort_assignment_sha256"]
        bound_dataset = release_value["dataset_receipt_sha256"]
    except (KeyError, TypeError) as error:
        raise BootstrapError("release receipt is missing artifact bindings") from error
    if (
        bound_archive != release_digest
        or bound_commit != code_commit
        or bound_cohort != cohort_digest
        or bound_dataset != corpus_digest
    ):
        raise BootstrapError("release receipt artifact bindings do not match")
    return BootstrapArtifacts(
        release_sha256=release_digest,
        release_receipt_sha256=release_receipt_digest,
        corpus_receipt_sha256=corpus_digest,
        cohort_assignment_sha256=cohort_digest,
        code_commit=code_commit,
    )


def build_bootstrap_receipt(
    *,
    profile: AwsP5Profile,
    runtime: AwsP5Runtime,
    evidence: BootstrapEvidence,
    artifacts: BootstrapArtifacts,
    durable_upload_verified: bool,
) -> dict[str, object]:
    if not isinstance(durable_upload_verified, bool):
        raise BootstrapError("durable upload status must be boolean")
    return {
        "ami_id": evidence.ami_id,
        "code_commit": artifacts.code_commit,
        "cohort_assignment_sha256": artifacts.cohort_assignment_sha256,
        "container_digest": runtime.container_digest,
        "corpus_receipt_sha256": artifacts.corpus_receipt_sha256,
        "durable_upload_verified": durable_upload_verified,
        "instance_id": evidence.instance_id,
        "instance_store": {
            "device_bytes": profile.instance_store_device_bytes,
            "devices": len(evidence.instance_store_devices),
            "model": profile.instance_store_model,
            "raid_level": profile.raid_level,
        },
        "instance_type": evidence.instance_type,
        "profile_sha256": profile.sha256,
        "provider": profile.provider,
        "receipt_type": "aws-p5-bootstrap",
        "region": runtime.region,
        "release_sha256": artifacts.release_sha256,
        "schema_version": 1,
        "scratch_root": profile.scratch_root,
    }


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def publish_bootstrap_receipt(
    receipt: Mapping[str, object],
    *,
    receipt_path: Path,
    receipt_uri: str,
    object_store,
) -> bool:
    """Write and independently verify one canonical S3 bootstrap receipt."""

    receipt_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if receipt_path.is_symlink():
        raise BootstrapError("bootstrap receipt path must not be a symlink")
    temporary = receipt_path.with_name(
        f".{receipt_path.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_json(dict(receipt)))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, receipt_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return bool(object_store.put_verified(receipt_path, receipt_uri))


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        type=Path,
        default=root / "profiles" / "aws-p5.48xlarge.json",
    )
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--release-archive", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--release-receipt", type=Path, required=True)
    parser.add_argument("--release-receipt-sha256", required=True)
    parser.add_argument("--dataset-receipt", type=Path, required=True)
    parser.add_argument("--dataset-receipt-sha256", required=True)
    parser.add_argument("--cohort-assignment", type=Path, required=True)
    parser.add_argument("--cohort-assignment-sha256", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--owner-uid", type=int, default=os.getuid())
    parser.add_argument("--owner-gid", type=int, default=os.getgid())
    parser.add_argument("--apply", action="store_true")
    return parser


def _safe_command_environment(
    profile: AwsP5Profile,
    environment: Mapping[str, str],
) -> dict[str, str]:
    return {
        name: environment[name]
        for name in profile.process_env_allowlist
        if name in environment
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        profile = load_aws_p5_profile(arguments.profile)
        runtime = validate_runtime_environment(profile, os.environ)
        command_environment = _safe_command_environment(profile, os.environ)
        client = ImdsV2Client()
        evidence = inspect_hardware(
            profile,
            runtime,
            metadata_get=client.get,
            command_environment=command_environment,
            container_image=arguments.container_image,
        )
        commands = render_bootstrap_commands(
            profile,
            runtime,
            evidence,
            owner_uid=arguments.owner_uid,
            owner_gid=arguments.owner_gid,
        )
        if not arguments.apply:
            print(
                json.dumps(
                    {
                        "commands": commands,
                        "dry_run": True,
                        "ok": True,
                        "schema_version": 1,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        if os.geteuid() != 0:
            raise BootstrapError("bootstrap --apply must run as root")
        for command in commands:
            _checked(
                _run_command,
                command,
                command_environment,
                operation=command[0],
            )
        artifacts = verify_bootstrap_artifacts(
            release_archive=arguments.release_archive,
            release_sha256=arguments.release_sha256,
            release_receipt=arguments.release_receipt,
            release_receipt_sha256=arguments.release_receipt_sha256,
            dataset_receipt=arguments.dataset_receipt,
            dataset_receipt_sha256=arguments.dataset_receipt_sha256,
            cohort_assignment=arguments.cohort_assignment,
            cohort_assignment_sha256=arguments.cohort_assignment_sha256,
            code_commit=arguments.code_commit,
        )
        store = S3ObjectStore(
            region=runtime.region,
            environment=command_environment,
        )
        receipt_path = arguments.receipt or (
            Path(profile.scratch_root) / "staging" / "bootstrap-receipt.json"
        )
        receipt_uri = (
            f"{runtime.s3_root}/receipts/bootstrap/{evidence.instance_id}.json"
        )
        pending = build_bootstrap_receipt(
            profile=profile,
            runtime=runtime,
            evidence=evidence,
            artifacts=artifacts,
            durable_upload_verified=False,
        )
        if not publish_bootstrap_receipt(
            pending,
            receipt_path=receipt_path,
            receipt_uri=receipt_uri,
            object_store=store,
        ):
            raise BootstrapError("bootstrap receipt upload was not verified")
        final = build_bootstrap_receipt(
            profile=profile,
            runtime=runtime,
            evidence=evidence,
            artifacts=artifacts,
            durable_upload_verified=True,
        )
        if not publish_bootstrap_receipt(
            final,
            receipt_path=receipt_path,
            receipt_uri=receipt_uri,
            object_store=store,
        ):
            raise BootstrapError("final bootstrap receipt upload was not verified")
        print(
            json.dumps(
                {
                    "dry_run": False,
                    "ok": True,
                    "receipt": str(receipt_path),
                    "schema_version": 1,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except (BootstrapError, ValueError, OSError) as error:
        print(
            json.dumps(
                {
                    "error": type(error).__name__,
                    "ok": False,
                    "schema_version": 1,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
