"""Closed-schema provider profile loading."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .errors import MsctlError
from .fsutil import open_directory, open_regular_at, read_fd
from .jsonutil import (
    canonical_sha256,
    require_exact_keys,
    require_object,
    require_positive_int,
    require_schema_version,
)


SUPPORTED_PROFILE = "illumina-usfc-prd"


@dataclass(frozen=True)
class IlluminaProfile:
    profile_id: str
    provider: str
    cluster: str
    partition: str | None
    account: str | None
    qos: str | None
    gres: str
    allocated_gpus: int
    train_groups: tuple[int, int]
    evaluation_gpus: int
    shared_root_env: str
    shared_root_prefix: str
    seed0_wall_minutes: int
    evaluation_wall_minutes: int
    job_env_allowlist: tuple[str, ...]
    environment_lock: str
    python_implementation: str
    python_version: str | None
    platform: str
    cuda_version: str | None
    environment_status: str
    sha256: str
    source_sha256: str


def _optional_slug(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or any(character in value for character in ",=\n\r\0")
    ):
        raise MsctlError(
            "PROFILE_INVALID",
            f"{label} must be null or a safe Slurm identifier",
        )
    return value


def load_profile(path: Path | str) -> IlluminaProfile:
    candidate = Path(path)
    directory_fd = open_directory(
        candidate.parent,
        label="profile directory",
    )
    try:
        descriptor, parent_fd, _ = open_regular_at(
            directory_fd,
            candidate.name,
            label="profile",
        )
        try:
            before = os.fstat(descriptor)
            data = read_fd(descriptor)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
            os.close(parent_fd)
    finally:
        os.close(directory_fd)
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
    ) or len(data) != after.st_size:
        raise MsctlError(
            "PROFILE_INVALID",
            "profile changed while being read",
        )
    try:
        decoded = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MsctlError(
            "PROFILE_INVALID",
            "profile must contain valid UTF-8 JSON",
        ) from error
    value = require_object(decoded, label="profile")
    require_exact_keys(
        value,
        {
            "schema_version",
            "profile_id",
            "provider",
            "cluster",
            "login",
            "cpu",
            "gpu",
            "storage",
            "slurm",
            "job_env_allowlist",
            "environment",
        },
        label="profile",
    )
    try:
        require_schema_version(
            value["schema_version"],
            label="profile.schema_version",
        )
    except MsctlError as error:
        raise MsctlError(
            "PROFILE_INVALID",
            "unsupported profile schema version",
        ) from error
    if (
        value["profile_id"] != SUPPORTED_PROFILE
        or value["provider"] != SUPPORTED_PROFILE
        or value["cluster"] != "usfc-prd"
    ):
        raise MsctlError(
            "PROVIDER_UNSUPPORTED",
            "only the illumina-usfc-prd provider profile is supported",
        )

    login = require_object(value["login"], label="profile.login")
    require_exact_keys(
        login,
        {"max_vcpus", "heavy_compute_allowed"},
        label="profile.login",
    )
    if login != {"max_vcpus": 4, "heavy_compute_allowed": False}:
        raise MsctlError(
            "PROFILE_INVALID",
            "login profile must prohibit heavy work on the four-vCPU VM",
        )

    cpu = require_object(value["cpu"], label="profile.cpu")
    require_exact_keys(
        cpu,
        {"nodes", "cpus_per_node", "local_ssd_required"},
        label="profile.cpu",
    )
    if cpu != {
        "nodes": 35,
        "cpus_per_node": 56,
        "local_ssd_required": True,
    }:
        raise MsctlError(
            "PROFILE_INVALID",
            "Illumina CPU capacity must match the frozen known facts",
        )

    gpu = require_object(value["gpu"], label="profile.gpu")
    require_exact_keys(
        gpu,
        {
            "model",
            "generic_resource",
            "allocated",
            "seed0_train_groups",
            "evaluation_gpus",
        },
        label="profile.gpu",
    )
    groups = gpu["seed0_train_groups"]
    if (
        gpu["model"] != "NVIDIA A100 80GB"
        or gpu["generic_resource"] != "gpu:a100"
        or gpu["allocated"] != 7
        or groups != [3, 3]
        or gpu["evaluation_gpus"] != 1
    ):
        raise MsctlError(
            "PROFILE_INVALID",
            "GPU profile must be symmetric 3+3 with one evaluation A100",
        )

    storage = require_object(value["storage"], label="profile.storage")
    require_exact_keys(
        storage,
        {
            "shared_root_env",
            "shared_root_prefix",
            "gpu_local_ssd_bytes",
        },
        label="profile.storage",
    )
    if (
        storage["shared_root_env"] != "MS_SHARED_ROOT"
        or storage["shared_root_prefix"] != "/illumina"
        or require_positive_int(
            storage["gpu_local_ssd_bytes"],
            label="profile.storage.gpu_local_ssd_bytes",
        )
        < 10_000_000_000_000
    ):
        raise MsctlError(
            "PROFILE_INVALID",
            "storage profile must use operator-supplied /illumina storage",
        )

    slurm = require_object(value["slurm"], label="profile.slurm")
    require_exact_keys(
        slurm,
        {
            "partition",
            "account",
            "qos",
            "seed0_wall_minutes",
            "evaluation_wall_minutes",
        },
        label="profile.slurm",
    )
    seed0_wall = require_positive_int(
        slurm["seed0_wall_minutes"],
        label="profile.slurm.seed0_wall_minutes",
    )
    evaluation_wall = require_positive_int(
        slurm["evaluation_wall_minutes"],
        label="profile.slurm.evaluation_wall_minutes",
    )

    environment = require_object(
        value["environment"],
        label="profile.environment",
    )
    require_exact_keys(
        environment,
        {"status", "lock_path", "contract"},
        label="profile.environment",
    )
    contract = require_object(
        environment["contract"],
        label="profile.environment.contract",
    )
    require_exact_keys(
        contract,
        {
            "python_implementation",
            "python_version",
            "platform",
            "cuda_version",
        },
        label="profile.environment.contract",
    )
    if (
        environment["status"] not in {"operator_input_required", "pinned"}
        or environment["lock_path"] != "requirements-illumina.lock"
        or contract["python_implementation"] != "CPython"
        or contract["platform"] != "linux_x86_64"
        or (
            contract["python_version"] is not None
            and (
                not isinstance(contract["python_version"], str)
                or not contract["python_version"]
            )
        )
        or (
            contract["cuda_version"] is not None
            and (
                not isinstance(contract["cuda_version"], str)
                or not contract["cuda_version"]
            )
        )
        or (
            environment["status"] == "pinned"
            and (
                contract["python_version"] is None
                or contract["cuda_version"] is None
            )
        )
        or (
            environment["status"] == "operator_input_required"
            and contract["python_version"] is not None
            and contract["cuda_version"] is not None
        )
    ):
        raise MsctlError(
            "PROFILE_INVALID",
            "environment contract must explicitly identify missing site pins",
        )

    allowlist = value["job_env_allowlist"]
    if (
        not isinstance(allowlist, list)
        or not allowlist
        or len(set(allowlist)) != len(allowlist)
        or not all(
            isinstance(item, str)
            and item.startswith("MS_")
            and item.replace("_", "").isalnum()
            for item in allowlist
        )
        or "MSCTL_APPROVAL_KEY" in allowlist
    ):
        raise MsctlError(
            "PROFILE_INVALID",
            "job environment allowlist is invalid",
        )

    return IlluminaProfile(
        profile_id=str(value["profile_id"]),
        provider=str(value["provider"]),
        cluster=str(value["cluster"]),
        partition=_optional_slug(
            slurm["partition"], label="profile.slurm.partition"
        ),
        account=_optional_slug(slurm["account"], label="profile.slurm.account"),
        qos=_optional_slug(slurm["qos"], label="profile.slurm.qos"),
        gres=str(gpu["generic_resource"]),
        allocated_gpus=int(gpu["allocated"]),
        train_groups=(int(groups[0]), int(groups[1])),
        evaluation_gpus=int(gpu["evaluation_gpus"]),
        shared_root_env=str(storage["shared_root_env"]),
        shared_root_prefix=str(storage["shared_root_prefix"]),
        seed0_wall_minutes=seed0_wall,
        evaluation_wall_minutes=evaluation_wall,
        job_env_allowlist=tuple(allowlist),
        environment_lock=str(environment["lock_path"]),
        python_implementation=str(contract["python_implementation"]),
        python_version=(
            str(contract["python_version"])
            if contract["python_version"] is not None
            else None
        ),
        platform=str(contract["platform"]),
        cuda_version=(
            str(contract["cuda_version"])
            if contract["cuda_version"] is not None
            else None
        ),
        environment_status=str(environment["status"]),
        sha256=canonical_sha256(value),
        source_sha256=hashlib.sha256(data).hexdigest(),
    )
