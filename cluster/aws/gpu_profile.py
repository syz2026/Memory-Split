"""Closed AWS GPU hardware profiles and runtime-environment validation."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from msctl.aws_contracts import validate_digest_pinned_oci_image
from msctl.errors import MsctlError
from msctl.fsutil import open_directory


P5_PROFILE_ID = "aws-p5.48xlarge"
P5_PROFILE_ID_V3 = "aws-p5.48xlarge-v3"
P6_PROFILE_ID_V3 = "aws-p6-b300.48xlarge-v3"
P6_SOFTWARE_FLOOR_PROVENANCE = (
    "official AWS P6-B300 requirements: CUDA 13.0, NVIDIA driver R580, "
    "NVLINK 5 R580, kernel 6.1, EFA 1.44.0, and OFI-NCCL 1.17.1"
)
_MAX_PROFILE_BYTES = 65_536
_AMI_RE = re.compile(r"^ami-[0-9a-f]{8,17}$")
_REGION_RE = re.compile(r"^[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+$")
_BUCKET_RE = re.compile(
    r"^(?![0-9]+(?:\.[0-9]+){3}$)(?!-)(?!.*\.\.)(?!.*\.-)(?!.*-\.)"
    r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$"
)
_SECRET_NAME_RE = re.compile(
    r"(?:^|_)(?:CREDENTIALS?|PASSWORD|PRIVATE_KEY|SECRET|TOKEN)(?:_|$)"
)
_STATIC_AWS_CREDENTIALS = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
    }
)
_NON_ROLE_AWS_CREDENTIAL_SOURCES = frozenset(
    {
        "AWS_CONFIG_FILE",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_DEFAULT_PROFILE",
        "AWS_EC2_METADATA_DISABLED",
        "AWS_PROFILE",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    }
)
_BASE_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "profile_id",
        "provider",
        "instance_type",
        "purchase_model",
        "cpu",
        "gpu",
        "storage",
        "runtime",
        "assigned_seeds",
        "process_env_allowlist",
    }
)
_PROCESS_ENV_ALLOWLIST = (
    "AWS_REGION",
    "LANG",
    "LC_ALL",
)


@dataclass(frozen=True)
class _ProfileContract:
    schema_version: int
    profile_id: str
    provider: str
    instance_type: str
    purchase_model: str
    architecture: str
    vcpus: int
    memory_gib: int
    gpu_model: str
    assigned_seeds: tuple[int, ...]
    allowed_regions: tuple[str, ...]
    allowed_availability_zones: tuple[str, ...]
    software_floors: tuple[tuple[str, str], ...] = ()
    extended_location_fields: bool = False


_PROFILE_CONTRACTS = (
    _ProfileContract(
        schema_version=1,
        profile_id=P5_PROFILE_ID,
        provider=P5_PROFILE_ID,
        instance_type="p5.48xlarge",
        purchase_model="on_demand",
        architecture="x86_64",
        vcpus=192,
        memory_gib=2048,
        gpu_model="NVIDIA H100 80GB",
        assigned_seeds=(1, 2, 3, 4),
        allowed_regions=("us-east-1",),
        allowed_availability_zones=(),
    ),
    _ProfileContract(
        schema_version=1,
        profile_id=P5_PROFILE_ID_V3,
        provider=P5_PROFILE_ID,
        instance_type="p5.48xlarge",
        purchase_model="on_demand",
        architecture="x86_64",
        vcpus=192,
        memory_gib=2048,
        gpu_model="NVIDIA H100 80GB",
        assigned_seeds=tuple(range(10)),
        allowed_regions=("us-east-1",),
        allowed_availability_zones=(),
    ),
    _ProfileContract(
        schema_version=1,
        profile_id=P6_PROFILE_ID_V3,
        provider="aws-p6-b300.48xlarge",
        instance_type="p6-b300.48xlarge",
        purchase_model="on_demand",
        architecture="x86_64",
        vcpus=192,
        memory_gib=4096,
        gpu_model="NVIDIA B300",
        assigned_seeds=tuple(range(10)),
        allowed_regions=("us-east-1",),
        allowed_availability_zones=("us-east-1d",),
        software_floors=(
            ("cuda", "13.0"),
            ("efa", "1.44.0"),
            ("kernel", "6.1"),
            ("nvidia_driver", "R580"),
            ("nvlink", "R580"),
            ("ofi_nccl", "1.17.1"),
        ),
        extended_location_fields=True,
    ),
)


@dataclass(frozen=True)
class AwsGpuProfile:
    """Normalized immutable AWS GPU hardware and storage contract."""

    schema_version: int
    profile_id: str
    provider: str
    instance_type: str
    purchase_model: str
    vcpus: int
    memory_gib: int
    gpu_model: str
    allocated_gpus: int
    train_groups: tuple[int, int]
    scratch_root: str
    durable_uri_env: str
    instance_store_model: str
    instance_store_devices: int
    instance_store_device_bytes: int
    raid_level: str
    region_env: str
    ami_id_env: str
    container_digest_env: str
    runtime_uid_env: str
    runtime_gid_env: str
    assigned_seeds: tuple[int, ...]
    process_env_allowlist: tuple[str, ...]
    sha256: str
    architecture: str = "x86_64"
    allowed_regions: tuple[str, ...] = ()
    allowed_availability_zones: tuple[str, ...] = ()
    software_floors: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class AwsGpuRuntime:
    """Validated operator-provided immutable AWS runtime identities."""

    region: str
    s3_root: str
    ami_id: str
    container_image: str
    container_digest: str
    uid: int
    gid: int


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"profile contains duplicate key: {key}")
        result[key] = value
    return result


def _object(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{label} fields do not match schema; "
            f"missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )
    return value


def _exact_int(value: object, expected: int, *, label: str) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(f"{label} must be exactly {expected}")
    return value


def _exact_string(value: object, expected: str, *, label: str) -> str:
    if not isinstance(value, str) or value != expected:
        raise ValueError(f"{label} must be exactly {expected!r}")
    return value


def _profile_contract(value: object) -> tuple[dict[str, object], _ProfileContract]:
    if not isinstance(value, dict):
        raise ValueError("profile must be an object")
    contract = next(
        (
            candidate
            for candidate in _PROFILE_CONTRACTS
            if value.get("schema_version") == candidate.schema_version
            and value.get("profile_id") == candidate.profile_id
        ),
        None,
    )
    if contract is None:
        raise ValueError(
            "profile schema/profile_id identity is not a closed AWS GPU contract"
        )
    fields = _BASE_ROOT_FIELDS
    if contract.extended_location_fields:
        fields = fields | {"offerings", "software_floors"}
    return _object(value, fields=fields, label="profile"), contract


def _parse_profile(raw: object, *, sha256: str) -> AwsGpuProfile:
    value, contract = _profile_contract(raw)
    schema_version = _exact_int(
        value["schema_version"],
        contract.schema_version,
        label="profile.schema_version",
    )
    profile_id = _exact_string(
        value["profile_id"], contract.profile_id, label="profile.profile_id"
    )
    provider = _exact_string(
        value["provider"], contract.provider, label="profile.provider"
    )
    instance_type = _exact_string(
        value["instance_type"],
        contract.instance_type,
        label="profile.instance_type",
    )
    purchase_model = _exact_string(
        value["purchase_model"],
        contract.purchase_model,
        label="profile.purchase_model",
    )

    cpu_fields = frozenset({"vcpus", "memory_gib"})
    if contract.extended_location_fields:
        cpu_fields = cpu_fields | {"architecture"}
    cpu = _object(value["cpu"], fields=cpu_fields, label="profile.cpu")
    if contract.extended_location_fields:
        _exact_string(
            cpu["architecture"],
            contract.architecture,
            label="profile.cpu.architecture",
        )
    vcpus = _exact_int(cpu["vcpus"], contract.vcpus, label="profile.cpu.vcpus")
    memory_gib = _exact_int(
        cpu["memory_gib"],
        contract.memory_gib,
        label="profile.cpu.memory_gib",
    )

    gpu = _object(
        value["gpu"],
        fields=frozenset({"model", "allocated", "seed_train_groups"}),
        label="profile.gpu",
    )
    gpu_model = _exact_string(
        gpu["model"], contract.gpu_model, label="profile.gpu.model"
    )
    allocated_gpus = _exact_int(
        gpu["allocated"], 8, label="profile.gpu.allocated"
    )
    groups = gpu["seed_train_groups"]
    if (
        not isinstance(groups, list)
        or len(groups) != 2
        or any(type(group) is not int for group in groups)
        or groups != [4, 4]
    ):
        raise ValueError("profile.gpu.seed_train_groups must be exactly [4, 4]")

    storage = _object(
        value["storage"],
        fields=frozenset(
            {"scratch_root", "durable_uri_env", "instance_store"}
        ),
        label="profile.storage",
    )
    scratch_root = _exact_string(
        storage["scratch_root"],
        "/mnt/memorysplit",
        label="profile.storage.scratch_root",
    )
    durable_uri_env = _exact_string(
        storage["durable_uri_env"],
        "MS_S3_ROOT",
        label="profile.storage.durable_uri_env",
    )
    instance_store = _object(
        storage["instance_store"],
        fields=frozenset({"model", "devices", "device_bytes", "raid_level"}),
        label="profile.storage.instance_store",
    )
    instance_store_model = _exact_string(
        instance_store["model"],
        "Amazon EC2 NVMe Instance Storage",
        label="profile.storage.instance_store.model",
    )
    instance_store_devices = _exact_int(
        instance_store["devices"],
        8,
        label="profile.storage.instance_store.devices",
    )
    instance_store_device_bytes = _exact_int(
        instance_store["device_bytes"],
        3_840_000_000_000,
        label="profile.storage.instance_store.device_bytes",
    )
    raid_level = _exact_string(
        instance_store["raid_level"],
        "0",
        label="profile.storage.instance_store.raid_level",
    )

    runtime = _object(
        value["runtime"],
        fields=frozenset(
            {
                "region_env",
                "ami_id_env",
                "container_digest_env",
                "runtime_uid_env",
                "runtime_gid_env",
            }
        ),
        label="profile.runtime",
    )
    region_env = _exact_string(
        runtime["region_env"], "AWS_REGION", label="profile.runtime.region_env"
    )
    ami_id_env = _exact_string(
        runtime["ami_id_env"],
        "MS_AWS_AMI_ID",
        label="profile.runtime.ami_id_env",
    )
    container_digest_env = _exact_string(
        runtime["container_digest_env"],
        "MS_CONTAINER_DIGEST",
        label="profile.runtime.container_digest_env",
    )
    runtime_uid_env = _exact_string(
        runtime["runtime_uid_env"],
        "MS_RUNTIME_UID",
        label="profile.runtime.runtime_uid_env",
    )
    runtime_gid_env = _exact_string(
        runtime["runtime_gid_env"],
        "MS_RUNTIME_GID",
        label="profile.runtime.runtime_gid_env",
    )

    seeds = value["assigned_seeds"]
    if (
        not isinstance(seeds, list)
        or any(type(seed) is not int for seed in seeds)
        or seeds != list(contract.assigned_seeds)
    ):
        raise ValueError(
            "profile.assigned_seeds must exactly match its profile identity"
        )
    allowlist = value["process_env_allowlist"]
    if (
        not isinstance(allowlist, list)
        or tuple(allowlist) != _PROCESS_ENV_ALLOWLIST
        or any(not isinstance(name, str) for name in allowlist)
    ):
        raise ValueError(
            "profile.process_env_allowlist must match the closed safe allowlist"
        )

    if contract.extended_location_fields:
        offerings = value["offerings"]
        if not isinstance(offerings, list) or len(offerings) != 1:
            raise ValueError("profile.offerings must contain one closed offering")
        offering = _object(
            offerings[0],
            fields=frozenset({"region", "availability_zone"}),
            label="profile.offerings[0]",
        )
        _exact_string(
            offering["region"],
            contract.allowed_regions[0],
            label="profile.offerings[0].region",
        )
        _exact_string(
            offering["availability_zone"],
            contract.allowed_availability_zones[0],
            label="profile.offerings[0].availability_zone",
        )
        software_floors = _object(
            value["software_floors"],
            fields=frozenset(dict(contract.software_floors)),
            label="profile.software_floors",
        )
        for name, expected in contract.software_floors:
            _exact_string(
                software_floors[name],
                expected,
                label=f"profile.software_floors.{name}",
            )

    return AwsGpuProfile(
        schema_version=schema_version,
        profile_id=profile_id,
        provider=provider,
        instance_type=instance_type,
        purchase_model=purchase_model,
        architecture=contract.architecture,
        vcpus=vcpus,
        memory_gib=memory_gib,
        gpu_model=gpu_model,
        allocated_gpus=allocated_gpus,
        train_groups=(4, 4),
        scratch_root=scratch_root,
        durable_uri_env=durable_uri_env,
        instance_store_model=instance_store_model,
        instance_store_devices=instance_store_devices,
        instance_store_device_bytes=instance_store_device_bytes,
        raid_level=raid_level,
        region_env=region_env,
        ami_id_env=ami_id_env,
        container_digest_env=container_digest_env,
        runtime_uid_env=runtime_uid_env,
        runtime_gid_env=runtime_gid_env,
        assigned_seeds=contract.assigned_seeds,
        process_env_allowlist=_PROCESS_ENV_ALLOWLIST,
        allowed_regions=contract.allowed_regions,
        allowed_availability_zones=contract.allowed_availability_zones,
        sha256=sha256,
        software_floors=contract.software_floors,
    )


def parse_aws_gpu_profile_bytes(data: bytes) -> AwsGpuProfile:
    """Parse pinned bytes under one of the closed AWS GPU identities."""

    if not isinstance(data, bytes):
        raise TypeError("profile bytes must be bytes")
    if len(data) > _MAX_PROFILE_BYTES:
        raise ValueError("profile exceeds 64 KiB")
    try:
        raw = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"profile contains non-finite value: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("profile must contain valid UTF-8 JSON") from error
    return _parse_profile(raw, sha256=hashlib.sha256(data).hexdigest())


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def read_secure_regular_file(
    path: Path | str,
    *,
    label: str,
    max_bytes: int,
    private: bool = False,
) -> bytes:
    """Read one descriptor-pinned owned file without path replacement."""

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("secure file byte limit must be a positive integer")
    candidate = Path(path)
    name = candidate.name
    if not name or name in {".", ".."}:
        raise ValueError(f"{label} path is invalid")
    try:
        parent_fd = open_directory(candidate.parent, label=f"{label} parent")
    except (MsctlError, OSError) as error:
        raise ValueError(f"{label} parent path is unsafe") from error
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        mode = stat.S_IMODE(before.st_mode)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if before.st_nlink != 1:
            raise ValueError(f"{label} must have exactly one link")
        if before.st_uid != os.geteuid():
            raise ValueError(f"{label} owner must be the current user")
        if mode & stat.S_IRUSR == 0:
            raise ValueError(f"{label} mode must permit owner reads")
        if private:
            if mode & 0o077:
                raise ValueError(f"{label} mode must be private")
        elif mode & 0o022:
            raise ValueError(f"{label} must not be group or other writable")
        if before.st_size > max_bytes:
            raise ValueError(f"{label} exceeds its byte limit")

        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"{label} exceeds its byte limit")
        after = os.fstat(descriptor)
        path_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            _file_identity(before) != _file_identity(after)
            or _file_identity(after) != _file_identity(path_after)
            or total != after.st_size
        ):
            raise ValueError(f"{label} changed or was replaced during read")
        return b"".join(chunks)
    except FileNotFoundError as error:
        raise ValueError(f"{label} is not a regular file") from error
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(f"{label} must not be a symlink") from error
        raise ValueError(f"{label} could not be read safely") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def load_aws_gpu_profile(path: Path | str) -> AwsGpuProfile:
    """Load one regular JSON file under a closed AWS GPU profile identity."""

    return parse_aws_gpu_profile_bytes(
        read_secure_regular_file(
            path,
            label="profile",
            max_bytes=_MAX_PROFILE_BYTES,
        )
    )


def _required_environment(
    environment: Mapping[str, str],
    name: str,
) -> str:
    value = environment.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"runtime environment requires {name}")
    return value


def _validate_s3_root(value: str) -> str:
    parsed = urlsplit(value)
    path_parts = parsed.path.removeprefix("/").split("/")
    if (
        parsed.scheme != "s3"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
        or parsed.hostname is None
        or _BUCKET_RE.fullmatch(parsed.hostname) is None
        or len(path_parts) < 1
        or any(part in {"", ".", ".."} for part in path_parts)
        or "\\" in value
    ):
        raise ValueError("MS_S3_ROOT must be an s3:// bucket and durable prefix")
    return value.rstrip("/")


def _nonroot_id(value: str, *, label: str) -> int:
    if re.fullmatch(r"[1-9][0-9]{0,9}", value) is None:
        raise ValueError(f"{label} must be an explicit non-root decimal ID")
    parsed = int(value)
    if parsed > 2_147_483_647:
        raise ValueError(f"{label} exceeds the supported ID range")
    return parsed


def validate_runtime_environment(
    profile: AwsGpuProfile,
    environment: Mapping[str, str],
) -> AwsGpuRuntime:
    """Validate runtime-lock values and absence of ambient credentials."""

    alternate_sources = sorted(
        name
        for name, value in environment.items()
        if value and name.upper() in _NON_ROLE_AWS_CREDENTIAL_SOURCES
    )
    if alternate_sources:
        raise ValueError(
            "runtime environment selects a non-instance-role credential source: "
            + ", ".join(alternate_sources)
        )
    inherited_secrets = sorted(
        name
        for name, value in environment.items()
        if value
        and (
            name.upper() in _STATIC_AWS_CREDENTIALS
            or _SECRET_NAME_RE.search(name.upper()) is not None
        )
    )
    if inherited_secrets:
        raise ValueError(
            "runtime environment contains inherited secret variables: "
            + ", ".join(inherited_secrets)
        )

    region = _required_environment(environment, profile.region_env)
    if _REGION_RE.fullmatch(region) is None:
        raise ValueError("AWS_REGION is not a valid explicit region")
    s3_root = _validate_s3_root(
        _required_environment(environment, profile.durable_uri_env)
    )
    ami_id = _required_environment(environment, profile.ami_id_env)
    if _AMI_RE.fullmatch(ami_id) is None:
        raise ValueError("MS_AWS_AMI_ID must be an immutable AMI ID")
    container_digest = _required_environment(
        environment, profile.container_digest_env
    )
    container_image = _required_environment(environment, "MS_CONTAINER_IMAGE")
    try:
        validate_digest_pinned_oci_image(container_image, container_digest)
    except ValueError as error:
        raise ValueError(
            "MS_CONTAINER_IMAGE must be an exact registry reference pinned "
            "to MS_CONTAINER_DIGEST"
        ) from error
    uid = _nonroot_id(
        _required_environment(environment, profile.runtime_uid_env),
        label="MS_RUNTIME_UID",
    )
    gid = _nonroot_id(
        _required_environment(environment, profile.runtime_gid_env),
        label="MS_RUNTIME_GID",
    )
    return AwsGpuRuntime(
        region=region,
        s3_root=s3_root,
        ami_id=ami_id,
        container_image=container_image,
        container_digest=container_digest,
        uid=uid,
        gid=gid,
    )


__all__ = [
    "AwsGpuProfile",
    "AwsGpuRuntime",
    "P5_PROFILE_ID",
    "P5_PROFILE_ID_V3",
    "P6_PROFILE_ID_V3",
    "P6_SOFTWARE_FLOOR_PROVENANCE",
    "load_aws_gpu_profile",
    "parse_aws_gpu_profile_bytes",
    "read_secure_regular_file",
    "validate_runtime_environment",
]
