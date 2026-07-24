"""Strict closed AWS GPU profile and runtime-environment validation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit


LEGACY_AWS_P5_PROFILE_ID = "aws-p5.48xlarge"
AWS_P5_V3_PROFILE_ID = "aws-p5.48xlarge-v3"
AWS_P6_B300_V3_PROFILE_ID = "aws-p6-b300.48xlarge-v3"
AWS_GPU_PROFILE_IDS = frozenset(
    {
        LEGACY_AWS_P5_PROFILE_ID,
        AWS_P5_V3_PROFILE_ID,
        AWS_P6_B300_V3_PROFILE_ID,
    }
)
# Historical public constant retained for callers that bind the legacy profile.
PROFILE_ID = LEGACY_AWS_P5_PROFILE_ID
_MAX_PROFILE_BYTES = 65_536
_AMI_RE = re.compile(r"^ami-[0-9a-f]{8,17}$")
_CONTAINER_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
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
_ROOT_FIELDS = frozenset(
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
_P5_GPU_NAME_PATTERNS = (r"^NVIDIA H100 80GB(?: HBM3)?$",)
_P6_B300_GPU_NAME_PATTERNS = (r"^NVIDIA B300$",)
_CPU_AFFINITY_HALVES = ((0, 95), (96, 191))
_P5_SOFTWARE_MINIMUMS = {
    "cuda_minimum": "12.1",
    "driver_minimum": "530",
    "efa_minimum": "1.24.1",
    "linux_kernel_minimum": "5.10",
    "ofi_nccl_minimum": "1.7.2",
}
_P6_SOFTWARE_MINIMUMS = {
    "cuda_minimum": "13.0",
    "driver_minimum": "580",
    "efa_minimum": "1.44.0",
    "linux_kernel_minimum": "6.1",
    "ofi_nccl_minimum": "1.17.1",
}


def _v3_profile_value(
    *,
    profile_id: str,
    instance_type: str,
    memory_gib: int,
    gpu_model: str,
    gpu_name_patterns: tuple[str, ...],
    gres: str,
    software_minimums: Mapping[str, str],
) -> dict[str, object]:
    return {
        "assigned_seeds": list(range(10)),
        "cpu": {
            "affinity_halves": [list(group) for group in _CPU_AFFINITY_HALVES],
            "memory_gib": memory_gib,
            "vcpus": 192,
        },
        "gpu": {
            "allocated": 8,
            "gres": gres,
            "model": gpu_model,
            "name_patterns": list(gpu_name_patterns),
            "seed_train_groups": [4, 4],
        },
        "instance_type": instance_type,
        "process_env_allowlist": list(_PROCESS_ENV_ALLOWLIST),
        "profile_id": profile_id,
        "provider": profile_id,
        "purchase_model": "on_demand",
        "runtime": {
            "ami_id_env": "MS_AWS_AMI_ID",
            "container_digest_env": "MS_CONTAINER_DIGEST",
            "region_env": "AWS_REGION",
            "runtime_gid_env": "MS_RUNTIME_GID",
            "runtime_uid_env": "MS_RUNTIME_UID",
        },
        "schema_version": 3,
        "software": dict(software_minimums),
        "storage": {
            "durable_uri_env": "MS_S3_ROOT",
            "instance_store": {
                "device_bytes": 3_840_000_000_000,
                "devices": 8,
                "model": "Amazon EC2 NVMe Instance Storage",
                "raid_level": "0",
            },
            "scratch_root": "/mnt/memorysplit",
        },
    }


_KNOWN_V3_PROFILE_VALUES = {
    AWS_P5_V3_PROFILE_ID: _v3_profile_value(
        profile_id=AWS_P5_V3_PROFILE_ID,
        instance_type="p5.48xlarge",
        memory_gib=2048,
        gpu_model="NVIDIA H100 80GB",
        gpu_name_patterns=_P5_GPU_NAME_PATTERNS,
        gres="gpu:h100:8",
        software_minimums=_P5_SOFTWARE_MINIMUMS,
    ),
    AWS_P6_B300_V3_PROFILE_ID: _v3_profile_value(
        profile_id=AWS_P6_B300_V3_PROFILE_ID,
        instance_type="p6-b300.48xlarge",
        memory_gib=4096,
        gpu_model="NVIDIA B300",
        gpu_name_patterns=_P6_B300_GPU_NAME_PATTERNS,
        gres="gpu:b300:8",
        software_minimums=_P6_SOFTWARE_MINIMUMS,
    ),
}


@dataclass(frozen=True)
class AwsGpuProfile:
    """Normalized, immutable profile for one closed AWS GPU contract."""

    schema_version: int
    profile_id: str
    provider: str
    instance_type: str
    purchase_model: str
    vcpus: int
    memory_gib: int
    gpu_model: str
    gpu_name_patterns: tuple[str, ...]
    allocated_gpus: int
    train_groups: tuple[int, int]
    cpu_affinity_halves: tuple[tuple[int, int], tuple[int, int]]
    gres: str
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
    cuda_minimum: str
    driver_minimum: str
    linux_kernel_minimum: str
    efa_minimum: str
    ofi_nccl_minimum: str
    sha256: str

    @property
    def cpu_affinity(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Compatibility-friendly name for the two disjoint CPU halves."""

        return self.cpu_affinity_halves

    @property
    def software_minimums(self) -> Mapping[str, str]:
        return {
            "cuda": self.cuda_minimum,
            "driver": self.driver_minimum,
            "efa": self.efa_minimum,
            "linux_kernel": self.linux_kernel_minimum,
            "ofi_nccl": self.ofi_nccl_minimum,
        }

    @property
    def bootstrap_receipt_type(self) -> str:
        if self.profile_id == LEGACY_AWS_P5_PROFILE_ID:
            return "aws-p5-bootstrap"
        return "aws-gpu-bootstrap"

    def matches_gpu_name(self, name: str) -> bool:
        return isinstance(name, str) and any(
            re.fullmatch(pattern, name) is not None
            for pattern in self.gpu_name_patterns
        )


@dataclass(frozen=True)
class AwsGpuRuntime:
    """Validated operator-provided immutable AWS identities."""

    region: str
    s3_root: str
    ami_id: str
    container_image: str
    container_digest: str
    uid: int
    gid: int


# Compatibility aliases for the established AWS P5 API.
AwsP5Profile = AwsGpuProfile
AwsP5Runtime = AwsGpuRuntime


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


def _parse_legacy_profile(raw: object, *, sha256: str) -> AwsGpuProfile:
    value = _object(raw, fields=_ROOT_FIELDS, label="profile")
    _exact_int(value["schema_version"], 1, label="profile.schema_version")
    profile_id = _exact_string(
        value["profile_id"], PROFILE_ID, label="profile.profile_id"
    )
    provider = _exact_string(
        value["provider"], PROFILE_ID, label="profile.provider"
    )
    instance_type = _exact_string(
        value["instance_type"], "p5.48xlarge", label="profile.instance_type"
    )
    purchase_model = _exact_string(
        value["purchase_model"], "on_demand", label="profile.purchase_model"
    )

    cpu = _object(
        value["cpu"],
        fields=frozenset({"vcpus", "memory_gib"}),
        label="profile.cpu",
    )
    vcpus = _exact_int(cpu["vcpus"], 192, label="profile.cpu.vcpus")
    memory_gib = _exact_int(
        cpu["memory_gib"], 2048, label="profile.cpu.memory_gib"
    )

    gpu = _object(
        value["gpu"],
        fields=frozenset({"model", "allocated", "seed_train_groups"}),
        label="profile.gpu",
    )
    gpu_model = _exact_string(
        gpu["model"], "NVIDIA H100 80GB", label="profile.gpu.model"
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
        or seeds != [1, 2, 3, 4]
    ):
        raise ValueError("profile.assigned_seeds must be exactly [1, 2, 3, 4]")

    allowlist = value["process_env_allowlist"]
    if (
        not isinstance(allowlist, list)
        or tuple(allowlist) != _PROCESS_ENV_ALLOWLIST
        or any(not isinstance(name, str) for name in allowlist)
    ):
        raise ValueError(
            "profile.process_env_allowlist must match the closed safe allowlist"
        )

    return AwsGpuProfile(
        schema_version=1,
        profile_id=profile_id,
        provider=provider,
        instance_type=instance_type,
        purchase_model=purchase_model,
        vcpus=vcpus,
        memory_gib=memory_gib,
        gpu_model=gpu_model,
        gpu_name_patterns=_P5_GPU_NAME_PATTERNS,
        allocated_gpus=allocated_gpus,
        train_groups=(4, 4),
        cpu_affinity_halves=_CPU_AFFINITY_HALVES,
        gres="gpu:h100:8",
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
        assigned_seeds=(1, 2, 3, 4),
        process_env_allowlist=_PROCESS_ENV_ALLOWLIST,
        cuda_minimum=_P5_SOFTWARE_MINIMUMS["cuda_minimum"],
        driver_minimum=_P5_SOFTWARE_MINIMUMS["driver_minimum"],
        linux_kernel_minimum=_P5_SOFTWARE_MINIMUMS[
            "linux_kernel_minimum"
        ],
        efa_minimum=_P5_SOFTWARE_MINIMUMS["efa_minimum"],
        ofi_nccl_minimum=_P5_SOFTWARE_MINIMUMS["ofi_nccl_minimum"],
        sha256=sha256,
    )


def _same_typed_json(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _same_typed_json(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_typed_json(item, expected_item)
            for item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _parse_v3_profile(
    raw: object,
    *,
    expected: Mapping[str, object],
    sha256: str,
) -> AwsGpuProfile:
    if not _same_typed_json(raw, expected):
        raise ValueError(
            "profile must exactly match one closed known AWS GPU profile"
        )
    value = raw
    cpu = value["cpu"]
    gpu = value["gpu"]
    storage = value["storage"]
    instance_store = storage["instance_store"]
    runtime = value["runtime"]
    software = value["software"]
    return AwsGpuProfile(
        schema_version=3,
        profile_id=value["profile_id"],
        provider=value["provider"],
        instance_type=value["instance_type"],
        purchase_model=value["purchase_model"],
        vcpus=cpu["vcpus"],
        memory_gib=cpu["memory_gib"],
        gpu_model=gpu["model"],
        gpu_name_patterns=tuple(gpu["name_patterns"]),
        allocated_gpus=gpu["allocated"],
        train_groups=tuple(gpu["seed_train_groups"]),
        cpu_affinity_halves=tuple(
            tuple(group) for group in cpu["affinity_halves"]
        ),
        gres=gpu["gres"],
        scratch_root=storage["scratch_root"],
        durable_uri_env=storage["durable_uri_env"],
        instance_store_model=instance_store["model"],
        instance_store_devices=instance_store["devices"],
        instance_store_device_bytes=instance_store["device_bytes"],
        raid_level=instance_store["raid_level"],
        region_env=runtime["region_env"],
        ami_id_env=runtime["ami_id_env"],
        container_digest_env=runtime["container_digest_env"],
        runtime_uid_env=runtime["runtime_uid_env"],
        runtime_gid_env=runtime["runtime_gid_env"],
        assigned_seeds=tuple(value["assigned_seeds"]),
        process_env_allowlist=tuple(value["process_env_allowlist"]),
        cuda_minimum=software["cuda_minimum"],
        driver_minimum=software["driver_minimum"],
        linux_kernel_minimum=software["linux_kernel_minimum"],
        efa_minimum=software["efa_minimum"],
        ofi_nccl_minimum=software["ofi_nccl_minimum"],
        sha256=sha256,
    )


def _parse_profile(raw: object, *, sha256: str) -> AwsGpuProfile:
    if not isinstance(raw, dict):
        raise ValueError("profile must be an object")
    profile_id = raw.get("profile_id")
    if not isinstance(profile_id, str):
        raise ValueError("profile.profile_id must be a known string")
    if profile_id == LEGACY_AWS_P5_PROFILE_ID:
        return _parse_legacy_profile(raw, sha256=sha256)
    expected = _KNOWN_V3_PROFILE_VALUES.get(profile_id)
    if expected is None:
        raise ValueError(
            "profile_id is not one of the closed known AWS GPU profiles"
        )
    return _parse_v3_profile(raw, expected=expected, sha256=sha256)


def load_aws_gpu_profile(path: Path | str) -> AwsGpuProfile:
    """Load one regular JSON file under the closed AWS GPU profile contract."""

    profile_path = Path(path)
    if profile_path.is_symlink():
        raise ValueError(f"profile must not be a symlink: {profile_path}")
    if not profile_path.is_file():
        raise ValueError(f"profile is not a regular file: {profile_path}")
    data = profile_path.read_bytes()
    if len(data) > _MAX_PROFILE_BYTES:
        raise ValueError("profile exceeds 64 KiB")
    try:
        text = data.decode("utf-8")
        raw = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"profile contains non-finite value: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("profile must contain valid UTF-8 JSON") from error
    return _parse_profile(raw, sha256=hashlib.sha256(data).hexdigest())


# Established loader name retained as a true alias.
load_aws_p5_profile = load_aws_gpu_profile


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
    """Validate region, S3 prefix, AMI, digest, and absence of ambient secrets."""

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
    if _CONTAINER_DIGEST_RE.fullmatch(container_digest) is None:
        raise ValueError(
            "MS_CONTAINER_DIGEST must be sha256 followed by 64 lowercase hex"
        )
    container_image = _required_environment(environment, "MS_CONTAINER_IMAGE")
    image_name, separator, image_digest = container_image.partition("@")
    if (
        separator != "@"
        or image_digest != container_digest
        or container_image.count("@") != 1
        or "/" not in image_name
        or "." not in image_name.partition("/")[0]
        or any(character.isspace() for character in container_image)
    ):
        raise ValueError(
            "MS_CONTAINER_IMAGE must be an exact registry reference pinned "
            "to MS_CONTAINER_DIGEST"
        )
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


validate_aws_gpu_runtime_environment = validate_runtime_environment
