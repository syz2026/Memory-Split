#!/usr/bin/env python3
"""Verify and stage one immutable profile-selected AWS GPU environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cluster.aws.p5.interruption_checkpoint import (
    CommandResult,
    ImdsV2Client,
    S3ObjectStore,
)
from cluster.aws.p5.profile import (
    AWS_P5_V3_PROFILE_ID,
    AWS_P6_B300_V3_PROFILE_ID,
    LEGACY_AWS_P5_PROFILE_ID,
    AwsGpuProfile,
    AwsGpuRuntime,
    load_aws_p5_profile,
    validate_runtime_environment,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CONTAINER_IMAGE_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9./:_-]*@sha256:[0-9a-f]{64}$"
)
_V3_PROFILE_IDS = frozenset(
    {AWS_P5_V3_PROFILE_ID, AWS_P6_B300_V3_PROFILE_ID}
)
_V2_COHORT_ID = "memorysplit-confirmatory-v2-360m-n5"
_V3_COHORT_ID = "memorysplit-confirmatory-v3-360m-n10-aws"


class BootstrapError(ValueError):
    """A fail-closed bootstrap validation error."""


@dataclass(frozen=True)
class BootstrapEvidence:
    instance_id: str
    instance_type: str
    ami_id: str
    boot_id: str
    account_id: str
    role_name: str
    role_arn: str
    gpu_names: tuple[str, ...]
    fabric_manager_active: bool
    instance_store_devices: tuple[str, ...]
    container_image: str


@dataclass(frozen=True)
class BootstrapArtifacts:
    release_sha256: str
    release_receipt_sha256: str
    release_members_sha256: str
    release_members: tuple["ReleaseMember", ...]
    corpus_receipt_sha256: str
    corpus_build_id: str
    cohort_assignment_sha256: str
    code_commit: str
    provider: str = LEGACY_AWS_P5_PROFILE_ID
    assigned_seeds: tuple[int, ...] = (1, 2, 3, 4)


@dataclass(frozen=True)
class ReleaseMember:
    path: str
    bytes: int
    sha256: str
    executable: bool


@dataclass(frozen=True)
class PreparedRelease:
    root: Path
    members_sha256: str


def _run_command(
    argv: Sequence[str],
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> CommandResult:
    from cluster.aws.p5.interruption_checkpoint import _default_runner

    return _default_runner(argv, environment, timeout_seconds)


def _checked(
    runner: Callable[
        [Sequence[str], Mapping[str, str], float], CommandResult
    ],
    argv: list[str],
    environment: Mapping[str, str],
    *,
    operation: str,
    timeout_seconds: float = 30.0,
) -> CommandResult:
    result = _bounded_result(
        runner,
        argv,
        environment,
        operation=operation,
        timeout_seconds=timeout_seconds,
    )
    if result.returncode != 0:
        raise BootstrapError(f"{operation} failed")
    return result


def _bounded_result(
    runner: Callable[
        [Sequence[str], Mapping[str, str], float], CommandResult
    ],
    argv: list[str],
    environment: Mapping[str, str],
    *,
    operation: str,
    timeout_seconds: float,
) -> CommandResult:
    try:
        result = runner(argv, environment, timeout_seconds)
    except (subprocess.TimeoutExpired, TimeoutError) as error:
        raise BootstrapError(f"{operation} timed out") from error
    return result


def _metadata_value(
    metadata_get: Callable[[str], str | None],
    path: str,
) -> str:
    value = metadata_get(path)
    if not isinstance(value, str) or not value:
        raise BootstrapError(f"IMDSv2 did not return {path}")
    return value


def _default_boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except (OSError, UnicodeDecodeError) as error:
        raise BootstrapError("kernel boot ID is unavailable") from error


def inspect_hardware(
    profile: AwsGpuProfile,
    runtime: AwsGpuRuntime,
    *,
    metadata_get: Callable[[str], str | None],
    runner: Callable[
        [Sequence[str], Mapping[str, str], float], CommandResult
    ] = _run_command,
    command_environment: Mapping[str, str],
    container_image: str,
    boot_id_get: Callable[[], str] = _default_boot_id,
) -> BootstrapEvidence:
    """Verify actual AWS GPU identity, accelerators, NVMe, and image."""

    instance_id = _metadata_value(metadata_get, "meta-data/instance-id")
    if re.fullmatch(r"i-[0-9a-f]{8,17}", instance_id) is None:
        raise BootstrapError("IMDSv2 instance ID is invalid")
    instance_type = _metadata_value(metadata_get, "meta-data/instance-type")
    if instance_type != profile.instance_type:
        raise BootstrapError(
            f"instance must be exactly {profile.instance_type}"
        )
    ami_id = _metadata_value(metadata_get, "meta-data/ami-id")
    if ami_id != runtime.ami_id:
        raise BootstrapError("running AMI does not match MS_AWS_AMI_ID")
    boot_id = boot_id_get()
    if (
        not isinstance(boot_id, str)
        or re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
            r"[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            boot_id,
        )
        is None
    ):
        raise BootstrapError("kernel boot ID is invalid")
    role_name = _metadata_value(
        metadata_get,
        "meta-data/iam/security-credentials/",
    ).strip()
    if (
        "\n" in role_name
        or re.fullmatch(r"[A-Za-z0-9+=,.@_-]{1,64}", role_name) is None
    ):
        raise BootstrapError("IMDSv2 role name is invalid")
    identity_text = _metadata_value(
        metadata_get,
        "dynamic/instance-identity/document",
    )
    try:
        identity = json.loads(identity_text)
    except json.JSONDecodeError as error:
        raise BootstrapError("IMDSv2 identity document is invalid JSON") from error
    if not isinstance(identity, dict):
        raise BootstrapError("IMDSv2 identity document must be an object")
    account_id = identity.get("accountId")
    if (
        not isinstance(account_id, str)
        or re.fullmatch(r"[0-9]{12}", account_id) is None
        or identity.get("instanceId") != instance_id
        or identity.get("instanceType") != instance_type
        or identity.get("imageId") != ami_id
        or identity.get("region") != runtime.region
    ):
        raise BootstrapError("IMDSv2 identity document does not match runtime")
    sts_result = _checked(
        runner,
        [
            "aws",
            "sts",
            "get-caller-identity",
            "--output",
            "json",
            "--no-cli-pager",
            "--region",
            runtime.region,
        ],
        command_environment,
        operation="STS caller identity",
    )
    try:
        caller = json.loads(sts_result.stdout)
    except json.JSONDecodeError as error:
        raise BootstrapError("STS caller identity returned invalid JSON") from error
    expected_arn_prefix = (
        f"arn:aws:sts::{account_id}:assumed-role/{role_name}/"
    )
    if (
        not isinstance(caller, dict)
        or set(caller) != {"Account", "Arn", "UserId"}
        or caller.get("Account") != account_id
        or not isinstance(caller.get("Arn"), str)
        or not caller["Arn"].startswith(expected_arn_prefix)
        or not isinstance(caller.get("UserId"), str)
        or not caller["UserId"]
    ):
        raise BootstrapError(
            "STS caller identity is not the IMDSv2 instance role"
        )
    role_arn = caller["Arn"]

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
        raise BootstrapError(
            "exactly eight profile accelerator devices are required"
        )
    if any(not profile.matches_gpu_name(name) for name in gpu_names):
        raise BootstrapError(
            f"every accelerator must match {profile.gpu_model}"
        )

    fabric = _bounded_result(
        runner,
        ["systemctl", "is-active", "nvidia-fabricmanager"],
        command_environment,
        operation="NVIDIA Fabric Manager probe",
        timeout_seconds=30.0,
    )
    if fabric.returncode != 0 or fabric.stdout.strip() != "active":
        raise BootstrapError("NVIDIA Fabric Manager must be active")

    block_result = _checked(
        runner,
        [
            "lsblk",
            "--json",
            "--bytes",
            "--tree",
            "--output",
            (
                "NAME,PATH,TYPE,MODEL,SIZE,MOUNTPOINTS,FSTYPE,FSVER,"
                "LABEL,UUID,PTTYPE,PARTTYPE"
            ),
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
        children = row.get("children")
        if (
            row.get("type") != "disk"
            or not isinstance(path, str)
            or not path.startswith("/dev/")
            or any(character in path for character in "\x00\n\r\t ")
            or type(size) is not int
            or size != profile.instance_store_device_bytes
            or not isinstance(mountpoints, list)
            or any(point not in {None, ""} for point in mountpoints)
            or any(
                row.get(field) not in {None, ""}
                for field in (
                    "fstype",
                    "fsver",
                    "label",
                    "uuid",
                    "pttype",
                    "parttype",
                )
            )
            or (
                children is not None
                and children != []
                and children != ()
            )
        ):
            if isinstance(children, list) and children:
                raise BootstrapError(
                    "instance-store device has children or partitions"
                )
            if any(
                row.get(field) not in {None, ""}
                for field in ("fstype", "fsver", "label", "uuid")
            ):
                raise BootstrapError(
                    "instance-store device has a filesystem or signature"
                )
            if row.get("pttype") not in {None, ""} or row.get(
                "parttype"
            ) not in {None, ""}:
                raise BootstrapError(
                    "instance-store device has a partition signature"
                )
            raise BootstrapError("instance-store device metadata is unsafe")
        devices.append(path)
    if len(devices) != profile.instance_store_devices:
        raise BootstrapError(
            "exactly eight unmounted instance-store devices are required"
        )
    if len(set(devices)) != len(devices):
        raise BootstrapError("instance-store device paths must be unique")
    devices.sort()
    swap_result = _checked(
        runner,
        [
            "swapon",
            "--show",
            "--noheadings",
            "--raw",
            "--output",
            "NAME",
        ],
        command_environment,
        operation="swap discovery",
    )
    swap_devices = {
        line.strip()
        for line in swap_result.stdout.splitlines()
        if line.strip()
    }
    if swap_devices & set(devices):
        raise BootstrapError("instance-store device is active swap")
    for device in devices:
        block_name = Path(device).name
        holders = _checked(
            runner,
            ["ls", "-A", f"/sys/class/block/{block_name}/holders"],
            command_environment,
            operation="instance-store holder discovery",
        )
        if holders.stdout.strip():
            raise BootstrapError("instance-store device has active holders")
        md_result = _bounded_result(
            runner,
            ["mdadm", "--examine", "--brief", device],
            command_environment,
            operation="mdadm admission probe",
            timeout_seconds=30.0,
        )
        if md_result.returncode == 0:
            raise BootstrapError(
                "instance-store device already has md/RAID membership"
            )
        if md_result.returncode != 1:
            raise BootstrapError("mdadm admission probe failed")
        wipe_result = _checked(
            runner,
            ["wipefs", "--json", device],
            command_environment,
            operation="instance-store signature discovery",
        )
        try:
            wipe_value = json.loads(wipe_result.stdout)
        except json.JSONDecodeError as error:
            raise BootstrapError("wipefs returned invalid JSON") from error
        if (
            not isinstance(wipe_value, dict)
            or not isinstance(wipe_value.get("signatures"), list)
        ):
            raise BootstrapError("wipefs JSON is missing signatures")
        if wipe_value["signatures"]:
            raise BootstrapError(
                "instance-store device has a filesystem signature"
            )

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
        boot_id=boot_id,
        account_id=account_id,
        role_name=role_name,
        role_arn=role_arn,
        gpu_names=gpu_names,
        fabric_manager_active=True,
        instance_store_devices=tuple(devices),
        container_image=container_image,
    )


def build_aws_command_environment(
    profile: AwsGpuProfile,
    runtime: AwsGpuRuntime,
    *,
    private_home: Path,
) -> dict[str, str]:
    """Build a closed AWS CLI environment backed only by an empty private HOME."""

    del profile
    home = Path(private_home)
    if home.is_symlink() or not home.is_dir():
        raise BootstrapError("AWS private HOME must be a real directory")
    metadata = home.stat()
    if (
        stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
        or any(home.iterdir())
    ):
        raise BootstrapError(
            "AWS private HOME must be owned, mode 0700, and empty"
        )
    return {
        "AWS_REGION": runtime.region,
        "HOME": str(home.resolve(strict=True)),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }


def render_bootstrap_commands(
    profile: AwsGpuProfile,
    runtime: AwsGpuRuntime,
    evidence: BootstrapEvidence,
    *,
    owner_uid: int,
    owner_gid: int,
    apply: bool = False,
    destructive_authorized: bool = False,
) -> list[list[str]]:
    """Render the mutation plan as argv arrays only."""

    if type(owner_uid) is not int or owner_uid <= 0:
        raise BootstrapError("owner UID must be explicitly non-root")
    if type(owner_gid) is not int or owner_gid <= 0:
        raise BootstrapError("owner GID must be explicitly non-root")
    if not isinstance(apply, bool) or not isinstance(
        destructive_authorized, bool
    ):
        raise BootstrapError("bootstrap authorization flags must be booleans")
    if apply and not destructive_authorized:
        raise BootstrapError(
            "destructive instance-store authorization is required for apply"
        )
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
            f"{scratch}/releases",
            f"{scratch}/dataset",
            f"{scratch}/runs",
            f"{scratch}/staging",
            f"{scratch}/staging/releases",
        ],
        [
            "aws",
            "s3",
            "sync",
            f"{runtime.s3_root}/releases",
            f"{scratch}/staging/releases",
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


def _canonical_pretty(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _release_member_path(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or value.startswith("~")
        or "$" in value
    ):
        raise BootstrapError(f"{label} is not a portable release path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(
        part in {"", ".", ".."} for part in value.split("/")
    ):
        raise BootstrapError(f"{label} is not a portable release path")
    return path.as_posix()


def _zip_mode(info: zipfile.ZipInfo) -> int:
    return (info.external_attr >> 16) & 0xFFFF


def _verify_release_archive(
    release_archive: Path,
    *,
    expected_members_sha256: str,
    code_commit: str,
    provider: str = LEGACY_AWS_P5_PROFILE_ID,
    assigned_seeds: Sequence[int] = (1, 2, 3, 4),
) -> tuple[ReleaseMember, ...]:
    _required_sha256(
        expected_members_sha256,
        label="release members manifest",
    )
    try:
        archive = zipfile.ZipFile(release_archive, "r")
    except (OSError, zipfile.BadZipFile) as error:
        raise BootstrapError("release archive must be a valid ZIP") from error
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise BootstrapError("release archive repeats a member")
        regular: dict[str, tuple[zipfile.ZipInfo, bytes]] = {}
        directory_names: set[str] = set()
        for info in infos:
            raw_name = info.filename[:-1] if info.is_dir() else info.filename
            name = _release_member_path(raw_name, label="release member")
            mode = _zip_mode(info)
            file_type = mode & 0o170000
            if file_type == 0o120000:
                raise BootstrapError("release archive contains a symlink")
            if info.is_dir():
                if file_type not in {0, 0o040000}:
                    raise BootstrapError(
                        f"release directory has an unsafe mode: {name}"
                    )
                directory_names.add(name)
                continue
            if file_type not in {0, 0o100000}:
                raise BootstrapError(
                    f"release member is not a regular file: {name}"
                )
            try:
                payload = archive.read(info)
            except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                raise BootstrapError(
                    f"release member could not be read: {name}"
                ) from error
            if len(payload) != info.file_size:
                raise BootstrapError(f"release member byte count drift: {name}")
            regular[name] = (info, payload)
        expected_directories = {
            PurePosixPath(
                *PurePosixPath(name).parts[:depth]
            ).as_posix()
            for name in regular
            for depth in range(1, len(PurePosixPath(name).parts))
        }
        if not directory_names <= expected_directories:
            raise BootstrapError("release archive contains an extra directory")
        if {"SHA256SUMS", "RELEASE-METADATA.json"} - set(regular):
            raise BootstrapError(
                "release archive is missing SHA256SUMS or RELEASE-METADATA.json"
            )
        sums = regular["SHA256SUMS"][1]
        if hashlib.sha256(sums).hexdigest() != expected_members_sha256:
            raise BootstrapError("release SHA256SUMS binding mismatch")
        try:
            sums_text = sums.decode("ascii")
        except UnicodeDecodeError as error:
            raise BootstrapError("release SHA256SUMS must be ASCII") from error
        if not sums_text or not sums_text.endswith("\n"):
            raise BootstrapError("release SHA256SUMS must be newline-terminated")
        checksums: dict[str, str] = {}
        ordered_paths: list[str] = []
        for line in sums_text.splitlines():
            if len(line) < 67 or line[64:66] != "  ":
                raise BootstrapError("release SHA256SUMS line is malformed")
            digest = _required_sha256(
                line[:64],
                label="release member checksum",
            )
            path = _release_member_path(
                line[66:],
                label="release checksum path",
            )
            if path in checksums:
                raise BootstrapError("release SHA256SUMS repeats a path")
            checksums[path] = digest
            ordered_paths.append(path)
        if ordered_paths != sorted(ordered_paths):
            raise BootstrapError("release SHA256SUMS paths must be sorted")
        if set(checksums) != set(regular) - {"SHA256SUMS"}:
            raise BootstrapError(
                "release SHA256SUMS does not enumerate every exact member"
            )
        for name, expected_digest in checksums.items():
            if hashlib.sha256(regular[name][1]).hexdigest() != expected_digest:
                raise BootstrapError(
                    f"release SHA256SUMS member mismatch: {name}"
                )

        metadata_bytes = regular["RELEASE-METADATA.json"][1]
        try:
            metadata = json.loads(metadata_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BootstrapError(
                "release metadata must contain valid UTF-8 JSON"
            ) from error
        assignment = (
            metadata.get("seed_assignment")
            if isinstance(metadata, dict)
            else None
        )
        v3_provider = provider in _V3_PROFILE_IDS
        assignment_providers = {provider}
        if v3_provider:
            assignment_providers.add(AWS_P5_V3_PROFILE_ID)
        source = metadata.get("source") if isinstance(metadata, dict) else None
        valid_source = (
            isinstance(source, dict)
            and source.get("commit") == code_commit
            and source.get("dirty") is False
            and (
                (
                    set(source) == {"commit", "dirty", "tree"}
                    and isinstance(source.get("tree"), str)
                    and re.fullmatch(
                        r"(?:[0-9a-f]{40}|[0-9a-f]{64})",
                        source["tree"],
                    )
                    is not None
                )
                if v3_provider
                else set(source) == {"commit", "dirty"}
            )
        )
        valid_assignment = (
            isinstance(assignment, dict)
            and set(assignment) == {"arms", "cohort_id", "provider", "seeds"}
            and assignment.get("arms") == ["dense", "split90"]
            and assignment.get("cohort_id")
            == (_V3_COHORT_ID if v3_provider else _V2_COHORT_ID)
            and assignment.get("provider") in assignment_providers
            and assignment.get("seeds") == list(assigned_seeds)
        )
        if (
            not isinstance(metadata, dict)
            or _canonical_pretty(metadata) != metadata_bytes
            or metadata.get("schema_version") != 1
            or metadata.get("package_format_version")
            != ("aws-gpu-v3" if v3_provider else 1)
            or metadata.get("provider") != provider
            or (
                v3_provider
                and metadata.get("selected_profile_id") != provider
            )
            or not valid_source
            or not valid_assignment
        ):
            raise BootstrapError("release metadata identity does not match")
        rows = metadata.get("members")
        if not isinstance(rows, list):
            raise BootstrapError("release metadata members must be a list")
        row_paths: list[str] = []
        for row in rows:
            if (
                not isinstance(row, dict)
                or set(row)
                != {"bytes", "git_blob", "git_mode", "path", "sha256"}
            ):
                raise BootstrapError("release metadata member row is invalid")
            path = _release_member_path(
                row["path"], label="release metadata member path"
            )
            row_paths.append(path)
            if path not in regular or path in {
                "RELEASE-METADATA.json",
                "SHA256SUMS",
            }:
                raise BootstrapError(
                    "release metadata member namespace does not match"
                )
            payload = regular[path][1]
            if (
                type(row["bytes"]) is not int
                or row["bytes"] != len(payload)
                or row["sha256"] != hashlib.sha256(payload).hexdigest()
                or row["git_mode"] not in {"100644", "100755"}
                or not isinstance(row["git_blob"], str)
                or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", row["git_blob"])
                is None
            ):
                raise BootstrapError(
                    f"release metadata member binding drift: {path}"
                )
        if (
            row_paths != sorted(row_paths)
            or set(row_paths)
            != set(regular) - {"RELEASE-METADATA.json", "SHA256SUMS"}
        ):
            raise BootstrapError(
                "release metadata does not bind every package member"
            )

        executable_paths = {
            row["path"] for row in rows if row["git_mode"] == "100755"
        }
        return tuple(
            ReleaseMember(
                path=name,
                bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                executable=name in executable_paths,
            )
            for name, (_info, payload) in sorted(regular.items())
        )


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
    profile: AwsGpuProfile | None = None,
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
    try:
        corpus_value = json.loads(
            dataset_receipt.read_bytes().decode("utf-8")
        )
        corpus_build_id = corpus_value["build_id"]
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
    ) as error:
        raise BootstrapError(
            "dataset receipt is missing its canonical build ID"
        ) from error
    _required_sha256(corpus_build_id, label="dataset receipt build ID")
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
        bound_members = release_value["members_sha256"]
    except (KeyError, TypeError) as error:
        raise BootstrapError("release receipt is missing artifact bindings") from error
    v3_package = release_value.get("package_format_version") == "aws-gpu-v3"
    try:
        if v3_package:
            bound_dataset_pointer = _required_sha256(
                release_value["dataset_pointer_sha256"],
                label="release receipt dataset pointer",
            )
            bound_dataset = None
        else:
            bound_dataset = _required_sha256(
                release_value["dataset_receipt_sha256"],
                label="release receipt dataset",
            )
            bound_dataset_pointer = None
    except (KeyError, TypeError) as error:
        raise BootstrapError("release receipt is missing artifact bindings") from error
    if (
        bound_archive != release_digest
        or bound_commit != code_commit
        or bound_cohort != cohort_digest
        or (bound_dataset is not None and bound_dataset != corpus_digest)
    ):
        raise BootstrapError("release receipt artifact bindings do not match")
    release_members = _verify_release_archive(
        release_archive,
        expected_members_sha256=_required_sha256(
            bound_members,
            label="release receipt members",
        ),
        code_commit=code_commit,
        provider=(
            profile.provider
            if profile is not None
            else LEGACY_AWS_P5_PROFILE_ID
        ),
        assigned_seeds=(
            profile.assigned_seeds
            if profile is not None
            else (1, 2, 3, 4)
        ),
    )
    if bound_dataset_pointer is not None and not any(
        member.path == "DATASET-POINTER-AWS.json"
        and member.sha256 == bound_dataset_pointer
        for member in release_members
    ):
        raise BootstrapError(
            "release receipt dataset pointer binding does not match"
        )
    return BootstrapArtifacts(
        release_sha256=release_digest,
        release_receipt_sha256=release_receipt_digest,
        release_members_sha256=bound_members,
        release_members=release_members,
        corpus_receipt_sha256=corpus_digest,
        corpus_build_id=corpus_build_id,
        cohort_assignment_sha256=cohort_digest,
        code_commit=code_commit,
        provider=(
            profile.provider
            if profile is not None
            else LEGACY_AWS_P5_PROFILE_ID
        ),
        assigned_seeds=(
            profile.assigned_seeds
            if profile is not None
            else (1, 2, 3, 4)
        ),
    )


def extract_verified_release(
    *,
    release_archive: Path,
    artifacts: BootstrapArtifacts,
    scratch_root: Path,
    owner_uid: int,
    owner_gid: int,
) -> PreparedRelease:
    """Extract the already verified archive into one digest-named read-only root."""

    if type(owner_uid) is not int or owner_uid <= 0:
        raise BootstrapError("release owner UID must be explicitly non-root")
    if type(owner_gid) is not int or owner_gid <= 0:
        raise BootstrapError("release owner GID must be explicitly non-root")
    _regular_digest(
        release_archive,
        expected=artifacts.release_sha256,
        label="release archive",
    )
    members = _verify_release_archive(
        release_archive,
        expected_members_sha256=artifacts.release_members_sha256,
        code_commit=artifacts.code_commit,
        provider=artifacts.provider,
        assigned_seeds=artifacts.assigned_seeds,
    )
    if members != artifacts.release_members:
        raise BootstrapError("release member evidence changed before extraction")

    scratch = Path(scratch_root)
    releases = scratch / "releases"
    releases.mkdir(parents=True, exist_ok=True, mode=0o700)
    if releases.is_symlink() or not releases.is_dir():
        raise BootstrapError("release parent must be a real directory")
    target = releases / artifacts.release_sha256
    if target.exists() or target.is_symlink():
        raise BootstrapError("digest-named release root already exists")
    staging = releases / (
        f".{artifacts.release_sha256}.{os.getpid()}.extracting"
    )
    if staging.exists() or staging.is_symlink():
        raise BootstrapError("release extraction staging path already exists")
    staging.mkdir(mode=0o700)
    try:
        member_by_path = {member.path: member for member in members}
        with zipfile.ZipFile(release_archive, "r") as archive:
            actual_regular = {
                info.filename
                for info in archive.infolist()
                if not info.is_dir()
            }
            if actual_regular != set(member_by_path):
                raise BootstrapError("release members changed before extraction")
            for relative, member in sorted(member_by_path.items()):
                output = staging.joinpath(*PurePosixPath(relative).parts)
                output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if output.exists() or output.is_symlink():
                    raise BootstrapError(
                        f"release extraction path already exists: {relative}"
                    )
                payload = archive.read(relative)
                if (
                    len(payload) != member.bytes
                    or hashlib.sha256(payload).hexdigest() != member.sha256
                ):
                    raise BootstrapError(
                        f"release member changed during extraction: {relative}"
                    )
                with output.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chown(output, owner_uid, owner_gid)
                os.chmod(output, 0o555 if member.executable else 0o444)
        directories = sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in directories:
            os.chown(directory, owner_uid, owner_gid)
            os.chmod(directory, 0o555)
        os.chown(staging, owner_uid, owner_gid)
        os.chmod(staging, 0o555)
        try:
            os.rename(staging, target)
        except OSError as error:
            raise BootstrapError(
                "digest-named release root could not be published"
            ) from error
    finally:
        if staging.exists():
            os.chmod(staging, 0o700)
            for directory in staging.rglob("*"):
                if directory.is_dir():
                    os.chmod(directory, 0o700)
            shutil.rmtree(staging)
    return PreparedRelease(
        root=target,
        members_sha256=artifacts.release_members_sha256,
    )


def build_bootstrap_receipt(
    *,
    profile: AwsGpuProfile,
    runtime: AwsGpuRuntime,
    evidence: BootstrapEvidence,
    artifacts: BootstrapArtifacts,
    prepared_release: PreparedRelease,
    durable_upload_verified: bool,
) -> dict[str, object]:
    if not isinstance(durable_upload_verified, bool):
        raise BootstrapError("durable upload status must be boolean")
    if (
        prepared_release.members_sha256
        != artifacts.release_members_sha256
        or prepared_release.root
        != Path(profile.scratch_root)
        / "releases"
        / artifacts.release_sha256
        or artifacts.provider != profile.provider
        or artifacts.assigned_seeds != profile.assigned_seeds
    ):
        raise BootstrapError("prepared release evidence does not match bootstrap")
    return {
        "account_id": evidence.account_id,
        "ami_id": evidence.ami_id,
        "boot_id": evidence.boot_id,
        "code_commit": artifacts.code_commit,
        "cohort_assignment_sha256": artifacts.cohort_assignment_sha256,
        "container_image": evidence.container_image,
        "container_digest": runtime.container_digest,
        "corpus_build_id": artifacts.corpus_build_id,
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
        "receipt_type": profile.bootstrap_receipt_type,
        "region": runtime.region,
        "release_members_sha256": prepared_release.members_sha256,
        "release_root": (
            f"releases/{artifacts.release_sha256}"
        ),
        "release_sha256": artifacts.release_sha256,
        "role_arn": evidence.role_arn,
        "role_name": evidence.role_name,
        "runtime_gid": runtime.gid,
        "runtime_uid": runtime.uid,
        "schema_version": 2,
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
    timeout_seconds: float = 120.0,
    monotonic: Callable[[], float] = time.monotonic,
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
    receipt_digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    return bool(
        object_store.put_verified(
            receipt_path,
            receipt_uri,
            expected_sha256=receipt_digest,
            deadline=monotonic() + timeout_seconds,
            monotonic=monotonic,
        )
    )


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
    parser.add_argument("--owner-uid", type=int, required=True)
    parser.add_argument("--owner-gid", type=int, required=True)
    parser.add_argument("--aws-private-home", type=Path, required=True)
    parser.add_argument(
        "--authorize-destructive-instance-store",
        action="store_true",
    )
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        # Keep the historical patch point while the alias loads every closed
        # AWS GPU profile through the neutral implementation.
        profile = load_aws_p5_profile(arguments.profile)
        runtime = validate_runtime_environment(profile, os.environ)
        if (
            arguments.owner_uid != runtime.uid
            or arguments.owner_gid != runtime.gid
        ):
            raise BootstrapError(
                "owner UID/GID must match explicit runtime UID/GID"
            )
        command_environment = build_aws_command_environment(
            profile,
            runtime,
            private_home=arguments.aws_private_home,
        )
        client = ImdsV2Client()
        evidence = inspect_hardware(
            profile,
            runtime,
            metadata_get=client.get,
            runner=_run_command,
            command_environment=command_environment,
            container_image=arguments.container_image,
            boot_id_get=_default_boot_id,
        )
        commands = render_bootstrap_commands(
            profile,
            runtime,
            evidence,
            owner_uid=arguments.owner_uid,
            owner_gid=arguments.owner_gid,
            apply=arguments.apply,
            destructive_authorized=(
                arguments.authorize_destructive_instance_store
            ),
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
                timeout_seconds=900.0,
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
            profile=profile,
        )
        prepared_release = extract_verified_release(
            release_archive=arguments.release_archive,
            artifacts=artifacts,
            scratch_root=Path(profile.scratch_root),
            owner_uid=runtime.uid,
            owner_gid=runtime.gid,
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
            prepared_release=prepared_release,
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
            prepared_release=prepared_release,
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
