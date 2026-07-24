#!/usr/bin/env python3
"""Verify and stage one immutable AWS P5 execution environment."""

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
from cluster.aws.gpu_profile import (
    parse_aws_gpu_profile_bytes,
    read_secure_regular_file,
)
from cluster.aws.p5.profile import (
    AwsP5Profile,
    AwsP5Runtime,
    load_aws_p5_profile,
    validate_runtime_environment,
)
from cluster.aws.qualification import (
    _build_selected_bootstrap_receipt,
    _build_selected_environment_receipt,
    admit_cohort_provider_selection,
)
from msctl.aws_contracts import (
    ARMS,
    COHORT_ASSIGNMENT_PATH,
    COHORT_ID,
    DATASET_POINTER_PATH,
    EXPECTED_CONFIG_PATHS,
    PACKAGE_FORMAT_VERSION,
    PROFILE_PATH,
    PROVIDER,
    SEEDS,
)
from msctl.aws_lifecycle import (
    LIFECYCLE_BINDING_FIELDS,
    ProviderLifecycleBinding,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_H100_RE = re.compile(r"^NVIDIA H100 80GB(?: HBM3)?$")
_CONTAINER_IMAGE_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9./:_-]*@sha256:[0-9a-f]{64}$"
)
BOOTSTRAP_RECEIPT_FIELDS = (
    "account_id",
    "ami_id",
    "boot_id",
    "code_commit",
    "cohort_assignment_sha256",
    "container_image",
    "container_digest",
    "corpus_build_id",
    "corpus_ordered_stream_sha256",
    "corpus_receipt_sha256",
    "durable_upload_verified",
    "instance_id",
    "instance_store",
    "instance_type",
    "profile_sha256",
    "provider",
    "receipt_type",
    "region",
    "release_members_sha256",
    "release_root",
    "release_sha256",
    "role_arn",
    "role_name",
    "runtime_gid",
    "runtime_uid",
    "schema_version",
    "scratch_root",
)
_FULL_BOOTSTRAP_ARGUMENTS = (
    "container_image",
    "release_archive",
    "release_sha256",
    "release_receipt",
    "release_receipt_sha256",
    "dataset_receipt",
    "dataset_receipt_sha256",
    "cohort_assignment",
    "cohort_assignment_sha256",
    "code_commit",
    "owner_uid",
    "owner_gid",
    "aws_private_home",
)


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
    corpus_ordered_stream_sha256: str
    cohort_assignment_sha256: str
    profile_sha256: str
    code_commit: str
    code_tree: str


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
    profile: AwsP5Profile,
    runtime: AwsP5Runtime,
    *,
    metadata_get: Callable[[str], str | None],
    runner: Callable[
        [Sequence[str], Mapping[str, str], float], CommandResult
    ] = _run_command,
    command_environment: Mapping[str, str],
    container_image: str,
    boot_id_get: Callable[[], str] = _default_boot_id,
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
        raise BootstrapError("exactly eight H100 devices are required")
    if any(_H100_RE.fullmatch(name) is None for name in gpu_names):
        raise BootstrapError("every accelerator must be an NVIDIA H100 80GB")

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
    profile: AwsP5Profile,
    runtime: AwsP5Runtime,
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
    profile: AwsP5Profile,
    runtime: AwsP5Runtime,
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


def _exact_fields(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise BootstrapError(
            f"{label} fields do not match; "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _strict_json_object(data: bytes, *, label: str) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise BootstrapError(f"{label} repeats field: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                BootstrapError(f"{label} contains non-finite {constant}")
            ),
        )
    except BootstrapError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BootstrapError(f"{label} must contain valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise BootstrapError(f"{label} must contain a JSON object")
    return value


def _binding(
    value: object,
    *,
    label: str,
    expected_path: str,
) -> str:
    if not isinstance(value, dict):
        raise BootstrapError(f"{label} must be an object")
    _exact_fields(value, {"path", "sha256"}, label=label)
    path = _release_member_path(value["path"], label=f"{label} path")
    if path != expected_path:
        raise BootstrapError(f"{label} path does not match v3")
    return _required_sha256(value["sha256"], label=f"{label} SHA-256")


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
    code_tree: str,
    release_value: Mapping[str, object] | None = None,
) -> tuple[ReleaseMember, ...]:
    _required_sha256(
        expected_members_sha256,
        label="release members manifest",
    )
    if (
        not isinstance(code_commit, str)
        or _COMMIT_RE.fullmatch(code_commit) is None
        or not isinstance(code_tree, str)
        or _COMMIT_RE.fullmatch(code_tree) is None
    ):
        raise BootstrapError(
            "release source commit and tree must be 40-character Git IDs"
        )
    from msctl.contracts import (
        same_typed_value,
        validate_aws_dataset_pointer_contract,
        validate_runtime_attested_contract,
    )
    from msctl.errors import MsctlError

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
        metadata = _strict_json_object(
            metadata_bytes,
            label="release metadata",
        )
        selected_lifecycle = (
            release_value is not None
            and "provider_selection_sha256" in release_value
        )
        selected_provider = (
            release_value["provider"] if release_value is not None else PROVIDER
        )
        selected_profile_path = (
            release_value["profile"]["path"]
            if selected_lifecycle
            and isinstance(release_value.get("profile"), dict)
            else PROFILE_PATH
        )
        metadata_fields = {
            "schema_version",
            "package_format_version",
            "provider",
            "source",
            "seed_assignment",
            "cohort_assignment",
            "profile",
            "environment",
            "dataset_pointer",
            "config_sha256",
            "members",
        }
        if selected_lifecycle:
            metadata_fields |= {
                *LIFECYCLE_BINDING_FIELDS,
                "profile_id",
            }
        _exact_fields(
            metadata,
            metadata_fields,
            label="release metadata",
        )
        metadata_source = metadata["source"]
        if not isinstance(metadata_source, dict):
            raise BootstrapError("release metadata source must be an object")
        _exact_fields(
            metadata_source,
            {"commit", "tree", "dirty"},
            label="release metadata source",
        )
        if (
            _canonical_pretty(metadata) != metadata_bytes
            or type(metadata.get("schema_version")) is not int
            or metadata.get("schema_version") != 1
            or type(metadata.get("package_format_version")) is not int
            or metadata.get("package_format_version") != PACKAGE_FORMAT_VERSION
            or metadata.get("provider") != selected_provider
            or metadata_source["commit"] != code_commit
            or metadata_source["dirty"] is not False
            or metadata_source["tree"] != code_tree
            or not isinstance(metadata_source["tree"], str)
            or _COMMIT_RE.fullmatch(metadata_source["tree"]) is None
            or not same_typed_value(
                metadata.get("seed_assignment"),
                {
                    "arms": list(ARMS),
                    "cohort_id": COHORT_ID,
                    "provider": selected_provider,
                    "seeds": list(SEEDS),
                },
            )
        ):
            raise BootstrapError("release metadata identity does not match")
        if selected_lifecycle and any(
            not same_typed_value(
                metadata.get(field),
                release_value.get(field),
            )
            for field in (*LIFECYCLE_BINDING_FIELDS, "profile_id")
        ):
            raise BootstrapError(
                "release lifecycle metadata differs from receipt"
            )
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

        member_sha256 = {
            row["path"]: row["sha256"]
            for row in rows
        }
        cohort_sha256 = _binding(
            metadata["cohort_assignment"],
            label="release cohort assignment",
            expected_path=COHORT_ASSIGNMENT_PATH,
        )
        profile_sha256 = _binding(
            metadata["profile"],
            label="release profile",
            expected_path=selected_profile_path,
        )
        dataset_pointer_sha256 = _binding(
            metadata["dataset_pointer"],
            label="release dataset pointer",
            expected_path=DATASET_POINTER_PATH,
        )
        for path, digest, label in (
            (COHORT_ASSIGNMENT_PATH, cohort_sha256, "cohort assignment"),
            (selected_profile_path, profile_sha256, "profile"),
            (DATASET_POINTER_PATH, dataset_pointer_sha256, "dataset pointer"),
        ):
            if member_sha256.get(path) != digest:
                raise BootstrapError(
                    f"release {label} is not bound to its exact member"
                )
        config_sha256 = metadata["config_sha256"]
        if not isinstance(config_sha256, dict):
            raise BootstrapError("release config SHA-256 map must be an object")
        if set(config_sha256) != set(EXPECTED_CONFIG_PATHS):
            raise BootstrapError(
                "release config SHA-256 map must contain exact v3 configs"
            )
        for path, digest in config_sha256.items():
            _required_sha256(digest, label=f"release config {path}")
            if member_sha256.get(path) != digest:
                raise BootstrapError(
                    f"release config is not bound to its exact member: {path}"
                )
        try:
            validate_runtime_attested_contract(
                metadata["environment"],
                profile_sha256=profile_sha256,
            )
        except MsctlError as error:
            raise BootstrapError(
                "release runtime-attestation contract is invalid"
            ) from error

        assignment = _strict_json_object(
            regular[COHORT_ASSIGNMENT_PATH][1],
            label="v3 cohort assignment",
        )
        _exact_fields(
            assignment,
            {
                "schema_version",
                "cohort_id",
                "model_parameters",
                "optimizer_steps",
                "provider_seeds",
                "raw_target_tokens",
                "targets_per_update",
            },
            label="v3 cohort assignment",
        )
        provider_seeds = assignment["provider_seeds"]
        if (
            type(assignment["schema_version"]) is not int
            or assignment["schema_version"] != 3
            or assignment["cohort_id"] != COHORT_ID
            or type(assignment["model_parameters"]) is not int
            or assignment["model_parameters"] != 356_033_536
            or type(assignment["optimizer_steps"]) is not int
            or assignment["optimizer_steps"] != 13_582
            or type(assignment["raw_target_tokens"]) is not int
            or assignment["raw_target_tokens"] != 7_120_879_616
            or type(assignment["targets_per_update"]) is not int
            or assignment["targets_per_update"] != 524_288
            or not isinstance(provider_seeds, dict)
            or set(provider_seeds) != {PROVIDER}
            or provider_seeds[PROVIDER] != list(SEEDS)
            or any(type(seed) is not int for seed in provider_seeds[PROVIDER])
        ):
            raise BootstrapError(
                "v3 cohort assignment must contain AWS-only seeds 0 through 9"
            )
        profile = _strict_json_object(
            regular[selected_profile_path][1],
            label="v3 profile",
        )
        if selected_lifecycle:
            try:
                selected_profile = parse_aws_gpu_profile_bytes(
                    regular[selected_profile_path][1]
                )
            except (TypeError, ValueError) as error:
                raise BootstrapError(
                    "release selected profile identity is invalid"
                ) from error
            if (
                selected_profile.provider != selected_provider
                or selected_profile.profile_id != metadata["profile_id"]
                or selected_profile.sha256 != profile_sha256
                or selected_profile.assigned_seeds != tuple(SEEDS)
            ):
                raise BootstrapError(
                    "release selected profile differs from lifecycle"
                )
        elif (
            type(profile.get("schema_version")) is not int
            or profile.get("schema_version") != 1
            or profile.get("profile_id") != "aws-p5.48xlarge-v3"
            or profile.get("provider") != PROVIDER
            or profile.get("assigned_seeds") != list(SEEDS)
            or any(
                type(seed) is not int
                for seed in profile.get("assigned_seeds", ())
            )
        ):
            raise BootstrapError("release v3 profile identity is invalid")
        pointer = _strict_json_object(
            regular[DATASET_POINTER_PATH][1],
            label="dataset pointer",
        )
        try:
            validate_aws_dataset_pointer_contract(pointer)
        except MsctlError as error:
            raise BootstrapError(
                "release dataset pointer contract is invalid"
            ) from error

        if release_value is not None:
            for field in (
                "cohort_assignment",
                "profile",
                "environment",
                "dataset_pointer",
                "config_sha256",
                "seed_assignment",
                "source",
                "package_format_version",
                *LIFECYCLE_BINDING_FIELDS,
                "profile_id",
            ):
                if not selected_lifecycle and field in {
                    *LIFECYCLE_BINDING_FIELDS,
                    "profile_id",
                }:
                    continue
                if not same_typed_value(
                    release_value.get(field),
                    metadata[field],
                ):
                    raise BootstrapError(
                        f"release receipt does not bind metadata field {field}"
                    )
            if (
                release_value.get("cohort_assignment_sha256")
                != cohort_sha256
                or release_value.get("profile_sha256") != profile_sha256
                or release_value.get("dataset_pointer_sha256")
                != dataset_pointer_sha256
            ):
                raise BootstrapError(
                    "release receipt flat hashes do not bind metadata"
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
) -> BootstrapArtifacts:
    """Hash ZIP, release, dataset, and cohort bytes and cross-check bindings."""

    from msctl.contracts import (
        same_typed_value,
        validate_runtime_attested_contract,
    )
    from msctl.errors import MsctlError

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
    release_value = _strict_json_object(
        release_receipt.read_bytes(),
        label="release receipt",
    )
    corpus_digest = _regular_digest(
        dataset_receipt,
        expected=dataset_receipt_sha256,
        label="dataset receipt",
    )
    corpus_value = _strict_json_object(
        dataset_receipt.read_bytes(),
        label="dataset receipt",
    )
    try:
        corpus_build_id = corpus_value["build_id"]
        corpus_ordered_stream_sha256 = corpus_value["ordered_stream_sha256"]
    except KeyError as error:
        raise BootstrapError(
            "dataset receipt is missing build ID or ordered stream SHA-256"
        ) from error
    _required_sha256(corpus_build_id, label="dataset receipt build ID")
    _required_sha256(
        corpus_ordered_stream_sha256,
        label="dataset receipt ordered stream SHA-256",
    )
    cohort_digest = _regular_digest(
        cohort_assignment,
        expected=cohort_assignment_sha256,
        label="cohort assignment",
    )
    if not isinstance(code_commit, str) or _COMMIT_RE.fullmatch(code_commit) is None:
        raise BootstrapError("code commit must be 40 lowercase hex characters")
    selected_lifecycle = "provider_selection_sha256" in release_value
    release_fields = {
        "schema_version",
        "package_format_version",
        "release_id",
        "provider",
        "archive",
        "source",
        "members_sha256",
        "seed_assignment",
        "cohort_assignment",
        "profile",
        "environment",
        "dataset_pointer",
        "cohort_assignment_sha256",
        "profile_sha256",
        "dataset_pointer_sha256",
        "config_sha256",
    }
    if selected_lifecycle:
        release_fields |= {
            *LIFECYCLE_BINDING_FIELDS,
            "profile_id",
        }
    _exact_fields(
        release_value,
        release_fields,
        label="release receipt",
    )
    if (
        type(release_value["schema_version"]) is not int
        or release_value["schema_version"] != 1
        or type(release_value["package_format_version"]) is not int
        or release_value["package_format_version"] != PACKAGE_FORMAT_VERSION
        or release_value["provider"]
        not in (
            {"aws-p5.48xlarge", "aws-p6-b300.48xlarge"}
            if selected_lifecycle
            else {PROVIDER}
        )
        or not isinstance(release_value["release_id"], str)
        or re.fullmatch(
            r"[a-z0-9][a-z0-9._-]{0,126}[a-z0-9]",
            release_value["release_id"],
        )
        is None
    ):
        raise BootstrapError("release receipt package format identity is invalid")
    if selected_lifecycle:
        try:
            lifecycle = ProviderLifecycleBinding(
                cohort_id=release_value["cohort_id"],
                provider=release_value["provider"],
                profile_id=release_value["profile_id"],
                profile_sha256=release_value["profile_sha256"],
                hardware_amendment_sha256=release_value[
                    "hardware_amendment_sha256"
                ],
                provider_selection_sha256=release_value[
                    "provider_selection_sha256"
                ],
                provider_selection_version_id=release_value[
                    "provider_selection_version_id"
                ],
                runtime_lock_sha256=release_value["runtime_lock_sha256"],
                runtime_sbom_sha256=release_value["runtime_sbom_sha256"],
                qualification_evidence_sha256=release_value[
                    "qualification_evidence_sha256"
                ],
                qualification_environment_receipt_sha256=release_value[
                    "qualification_environment_receipt_sha256"
                ],
                qualification_canary_receipt_sha256=release_value[
                    "qualification_canary_receipt_sha256"
                ],
                qualification_approval_receipt_sha256=release_value[
                    "qualification_approval_receipt_sha256"
                ],
                qualification_approval_public_key_sha256=release_value[
                    "qualification_approval_public_key_sha256"
                ],
                objective_controls_contract_sha256=release_value[
                    "objective_controls_contract_sha256"
                ],
                account_id=release_value["account_id"],
                instance_id=release_value["instance_id"],
                boot_id=release_value["boot_id"],
                region=release_value["region"],
                availability_zone=release_value["availability_zone"],
                purchase_model=release_value["purchase_model"],
                seed=release_value["seed"],
                arms=tuple(release_value["arms"]),
            )
        except (TypeError, ValueError) as error:
            raise BootstrapError(
                "release provider lifecycle binding is invalid"
            ) from error
    source = release_value["source"]
    if not isinstance(source, dict):
        raise BootstrapError("release source must be an object")
    _exact_fields(source, {"commit", "tree", "dirty"}, label="release source")
    code_tree = source["tree"]
    if (
        source["commit"] != code_commit
        or source["dirty"] is not False
        or not isinstance(code_tree, str)
        or _COMMIT_RE.fullmatch(code_tree) is None
    ):
        raise BootstrapError(
            "release source commit and tree must be clean 40-character Git IDs"
        )
    archive = release_value["archive"]
    if not isinstance(archive, dict):
        raise BootstrapError("release archive binding must be an object")
    _exact_fields(archive, {"path", "sha256", "bytes"}, label="release archive")
    archive_path = _release_member_path(
        archive["path"],
        label="release archive path",
    )
    bound_archive = _required_sha256(
        archive["sha256"],
        label="release archive",
    )
    if (
        archive_path != release_archive.name
        or type(archive["bytes"]) is not int
        or archive["bytes"] != release_archive.stat().st_size
        or bound_archive != release_digest
    ):
        raise BootstrapError("release receipt archive binding does not match")
    bound_members = _required_sha256(
        release_value["members_sha256"],
        label="release receipt members",
    )
    seed_assignment = release_value["seed_assignment"]
    if not isinstance(seed_assignment, dict):
        raise BootstrapError("release seed assignment must be an object")
    _exact_fields(
        seed_assignment,
        {"cohort_id", "provider", "seeds", "arms"},
        label="release seed assignment",
    )
    if not same_typed_value(
        seed_assignment,
        {
            "arms": list(ARMS),
            "cohort_id": COHORT_ID,
            "provider": release_value["provider"],
            "seeds": list(SEEDS),
        },
    ):
        raise BootstrapError(
            "release seed assignment must contain AWS-only seeds 0 through 9"
        )
    bound_cohort = _binding(
        release_value["cohort_assignment"],
        label="release cohort assignment",
        expected_path=COHORT_ASSIGNMENT_PATH,
    )
    profile_path = (
        {
            "aws-p5.48xlarge-v3": PROFILE_PATH,
            "aws-p6-b300.48xlarge-v3": (
                "cluster/profiles/aws-p6-b300.48xlarge-v3.json"
            ),
        }[release_value["profile_id"]]
        if selected_lifecycle
        else PROFILE_PATH
    )
    profile_sha256 = _binding(
        release_value["profile"],
        label="release profile",
        expected_path=profile_path,
    )
    dataset_pointer_sha256 = _binding(
        release_value["dataset_pointer"],
        label="release dataset pointer",
        expected_path=DATASET_POINTER_PATH,
    )
    if (
        release_value["cohort_assignment_sha256"] != bound_cohort
        or release_value["profile_sha256"] != profile_sha256
        or release_value["dataset_pointer_sha256"] != dataset_pointer_sha256
    ):
        raise BootstrapError("release receipt flat hashes do not bind objects")
    config_sha256 = release_value["config_sha256"]
    if not isinstance(config_sha256, dict) or set(config_sha256) != set(
        EXPECTED_CONFIG_PATHS
    ):
        raise BootstrapError("release receipt config namespace is not exact v3")
    for path, digest in config_sha256.items():
        _required_sha256(digest, label=f"release receipt config {path}")
    try:
        validate_runtime_attested_contract(
            release_value["environment"],
            profile_sha256=profile_sha256,
        )
    except MsctlError as error:
        raise BootstrapError(
            "release receipt runtime-attestation contract is invalid"
        ) from error
    if (
        bound_cohort != cohort_digest
    ):
        raise BootstrapError(
            "release receipt cohort assignment binding does not match"
        )
    release_members = _verify_release_archive(
        release_archive,
        expected_members_sha256=bound_members,
        code_commit=code_commit,
        code_tree=code_tree,
        release_value=release_value,
    )
    return BootstrapArtifacts(
        release_sha256=release_digest,
        release_receipt_sha256=release_receipt_digest,
        release_members_sha256=bound_members,
        release_members=release_members,
        corpus_receipt_sha256=corpus_digest,
        corpus_build_id=corpus_build_id,
        corpus_ordered_stream_sha256=corpus_ordered_stream_sha256,
        cohort_assignment_sha256=cohort_digest,
        profile_sha256=profile_sha256,
        code_commit=code_commit,
        code_tree=code_tree,
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
        code_tree=artifacts.code_tree,
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
    profile: AwsP5Profile,
    runtime: AwsP5Runtime,
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
        or profile.sha256 != artifacts.profile_sha256
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
        "corpus_ordered_stream_sha256": (
            artifacts.corpus_ordered_stream_sha256
        ),
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


def _descriptor_read(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    """Read one descriptor-pinned regular file without following links."""

    if type(max_bytes) is not int or max_bytes <= 0:
        raise BootstrapError(f"{label} byte limit must be positive")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise BootstrapError(f"{label} is missing or linked") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise BootstrapError(f"{label} must be a regular file")
        if metadata.st_size > max_bytes:
            raise BootstrapError(f"{label} exceeds its byte limit")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if len(payload) != metadata.st_size:
        raise BootstrapError(f"{label} changed while being read")
    return payload


def verify_bootstrap_reuse(
    *,
    profile: AwsP5Profile,
    runtime: AwsP5Runtime,
    receipt_path: Path | str,
    expected_receipt_sha256: str,
    metadata_get: Callable[[str], str | None],
    boot_id_get: Callable[[], str] = _default_boot_id,
    scratch_root: Path | str | None = None,
    is_mounted: Callable[[str], bool] = os.path.ismount,
) -> dict[str, object]:
    """Prove one exact bootstrap receipt still matches this boot, mutating
    nothing."""

    _required_sha256(
        expected_receipt_sha256,
        label="expected bootstrap receipt",
    )
    payload = _descriptor_read(
        Path(receipt_path),
        label="bootstrap reuse receipt",
        max_bytes=1024 * 1024,
    )
    if hashlib.sha256(payload).hexdigest() != expected_receipt_sha256:
        raise BootstrapError("bootstrap reuse receipt SHA-256 mismatch")
    value = _strict_json_object(payload, label="bootstrap reuse receipt")
    _exact_fields(
        value,
        set(BOOTSTRAP_RECEIPT_FIELDS),
        label="bootstrap reuse receipt",
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 2
        or value["receipt_type"] != "aws-p5-bootstrap"
        or value["durable_upload_verified"] is not True
    ):
        raise BootstrapError("bootstrap reuse receipt identity is invalid")
    instance_id = _metadata_value(metadata_get, "meta-data/instance-id")
    if value["instance_id"] != instance_id:
        raise BootstrapError(
            "bootstrap reuse receipt instance differs from IMDSv2"
        )
    boot_id = boot_id_get()
    if (
        not isinstance(boot_id, str)
        or not boot_id
        or value["boot_id"] != boot_id
    ):
        raise BootstrapError(
            "bootstrap reuse receipt boot differs from the current kernel"
        )
    if (
        value["profile_sha256"] != profile.sha256
        or value["provider"] != profile.provider
        or value["instance_type"] != profile.instance_type
        or value["scratch_root"] != profile.scratch_root
        or value["region"] != runtime.region
        or value["ami_id"] != runtime.ami_id
        or value["container_image"] != runtime.container_image
        or value["container_digest"] != runtime.container_digest
        or value["runtime_uid"] != runtime.uid
        or value["runtime_gid"] != runtime.gid
    ):
        raise BootstrapError(
            "bootstrap reuse receipt does not match this exact runtime"
        )
    release_sha256 = _required_sha256(
        value["release_sha256"],
        label="bootstrap reuse release",
    )
    if value["release_root"] != f"releases/{release_sha256}":
        raise BootstrapError(
            "bootstrap reuse receipt release root is not content addressed"
        )
    scratch = Path(
        scratch_root if scratch_root is not None else profile.scratch_root
    )
    if scratch.is_symlink() or not scratch.is_dir():
        raise BootstrapError("scratch root must be a real directory")
    if not is_mounted(str(scratch)):
        raise BootstrapError("scratch root is not a mounted filesystem")
    metadata = os.stat(scratch)
    if (
        stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != value["runtime_uid"]
        or metadata.st_gid != value["runtime_gid"]
    ):
        raise BootstrapError(
            "scratch root owner or mode differs from the receipt runtime"
        )
    release_root = scratch / "releases" / release_sha256
    if release_root.is_symlink() or not release_root.is_dir():
        raise BootstrapError(
            "receipted release directory is missing from the scratch root"
        )
    dataset_payload = _descriptor_read(
        scratch / "dataset" / "receipt.json",
        label="bootstrap reuse dataset receipt",
        max_bytes=16 * 1024 * 1024,
    )
    if (
        hashlib.sha256(dataset_payload).hexdigest()
        != value["corpus_receipt_sha256"]
    ):
        raise BootstrapError(
            "dataset receipt hash differs from the bootstrap receipt"
        )
    return {
        "ok": True,
        "mode": "reuse",
        "boot_id": boot_id,
        "instance_id": instance_id,
        "receipt_sha256": expected_receipt_sha256,
        "release_sha256": release_sha256,
        "schema_version": 1,
    }


def bootstrap_authenticated_gpu_environment(
    *,
    authority_root: Path | str,
    repo_root: Path | str,
    runtime_lock_path: Path | str,
    runtime_evidence_path: Path | str,
    runtime_sbom_path: Path | str,
    environment_receipt_path: Path | str,
    store: object,
    account_id: str,
    instance_id: str,
    boot_id: str,
    seed: int,
    expected_selection_version_id: str,
    identity_verifier: object,
    approval_verifier: object,
    trusted_public_key_sha256: str,
    hardware_reader: object,
) -> dict[str, object]:
    """Measure through an injected reader and bind selected bootstrap facts."""

    authority = admit_cohort_provider_selection(
        authority_root=authority_root,
        repo_root=repo_root,
        runtime_lock_path=runtime_lock_path,
        runtime_evidence_path=runtime_evidence_path,
        store=store,
        account_id=account_id,
        instance_id=instance_id,
        boot_id=boot_id,
        seed=seed,
        expected_selection_version_id=expected_selection_version_id,
        identity_verifier=identity_verifier,
        approval_verifier=approval_verifier,
        trusted_public_key_sha256=trusted_public_key_sha256,
    )
    selected_profile = authority.profile
    measure = getattr(hardware_reader, "measure", None)
    if not callable(measure):
        raise BootstrapError("selected hardware reader is unavailable")
    try:
        evidence = measure(
            selected_profile=selected_profile,
            selection_binding=authority.bindings["dense"],
        )
    except BootstrapError:
        raise
    except Exception as error:
        raise BootstrapError("selected hardware measurement failed") from error
    if not isinstance(evidence, Mapping):
        raise BootstrapError("selected hardware measurement is not an object")
    try:
        runtime_lock_data = read_secure_regular_file(
            runtime_lock_path,
            label="selected runtime lock",
            max_bytes=1024 * 1024,
        )
        runtime_sbom_data = read_secure_regular_file(
            runtime_sbom_path,
            label="selected runtime SBOM",
            max_bytes=512 * 1024 * 1024,
        )
        environment_data = read_secure_regular_file(
            environment_receipt_path,
            label="selected environment receipt",
            max_bytes=16 * 1024 * 1024,
        )
        environment_receipt = json.loads(environment_data)
        if not isinstance(environment_receipt, dict):
            raise BootstrapError("selected environment receipt is not an object")
        return _build_selected_bootstrap_receipt(
            selection_authority=authority,
            runtime_lock_data=runtime_lock_data,
            runtime_sbom_data=runtime_sbom_data,
            environment_receipt=environment_receipt,
            hardware_evidence=evidence,
        )
    except (TypeError, ValueError) as error:
        raise BootstrapError("selected bootstrap evidence is invalid") from error


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
        default=root / "profiles" / "aws-p5.48xlarge-v3.json",
    )
    parser.add_argument("--container-image")
    parser.add_argument("--release-archive", type=Path)
    parser.add_argument("--release-sha256")
    parser.add_argument("--release-receipt", type=Path)
    parser.add_argument("--release-receipt-sha256")
    parser.add_argument("--dataset-receipt", type=Path)
    parser.add_argument("--dataset-receipt-sha256")
    parser.add_argument("--cohort-assignment", type=Path)
    parser.add_argument("--cohort-assignment-sha256")
    parser.add_argument("--code-commit")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--owner-uid", type=int)
    parser.add_argument("--owner-gid", type=int)
    parser.add_argument("--aws-private-home", type=Path)
    parser.add_argument(
        "--authorize-destructive-instance-store",
        action="store_true",
    )
    parser.add_argument("--verify-reuse", action="store_true")
    parser.add_argument("--expected-receipt-sha256")
    parser.add_argument("--apply", action="store_true")
    return parser


def _run_verify_reuse(arguments: argparse.Namespace) -> int:
    if (
        arguments.receipt is None
        or arguments.expected_receipt_sha256 is None
    ):
        raise BootstrapError(
            "verify-reuse requires --receipt and --expected-receipt-sha256"
        )
    if arguments.apply or arguments.authorize_destructive_instance_store:
        raise BootstrapError(
            "verify-reuse performs no mutation and forbids apply flags"
        )
    provided = [
        name
        for name in _FULL_BOOTSTRAP_ARGUMENTS
        if getattr(arguments, name) is not None
    ]
    if provided:
        raise BootstrapError(
            "verify-reuse rejects full-bootstrap arguments"
        )
    profile = load_aws_p5_profile(arguments.profile)
    runtime = validate_runtime_environment(profile, os.environ)
    client = ImdsV2Client()
    result = verify_bootstrap_reuse(
        profile=profile,
        runtime=runtime,
        receipt_path=arguments.receipt,
        expected_receipt_sha256=arguments.expected_receipt_sha256,
        metadata_get=client.get,
        boot_id_get=_default_boot_id,
    )
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        if arguments.verify_reuse:
            return _run_verify_reuse(arguments)
        if arguments.expected_receipt_sha256 is not None:
            raise BootstrapError(
                "expected receipt SHA-256 is a verify-reuse argument"
            )
        missing = [
            f"--{name.replace('_', '-')}"
            for name in _FULL_BOOTSTRAP_ARGUMENTS
            if getattr(arguments, name) is None
        ]
        if missing:
            raise BootstrapError(
                "bootstrap requires " + ", ".join(missing)
            )
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
