"""Strict AWS P5 profile and runtime-environment validation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit


PROFILE_ID = "aws-p5.48xlarge"
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
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
)


@dataclass(frozen=True)
class AwsP5Profile:
    """Normalized, immutable P5 hardware and storage contract."""

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
    assigned_seeds: tuple[int, ...]
    process_env_allowlist: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class AwsP5Runtime:
    """Validated operator-provided immutable AWS identities."""

    region: str
    s3_root: str
    ami_id: str
    container_digest: str


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


def _parse_profile(raw: object, *, sha256: str) -> AwsP5Profile:
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
            {"region_env", "ami_id_env", "container_digest_env"}
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

    return AwsP5Profile(
        schema_version=1,
        profile_id=profile_id,
        provider=provider,
        instance_type=instance_type,
        purchase_model=purchase_model,
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
        assigned_seeds=(1, 2, 3, 4),
        process_env_allowlist=_PROCESS_ENV_ALLOWLIST,
        sha256=sha256,
    )


def load_aws_p5_profile(path: Path | str) -> AwsP5Profile:
    """Load one regular JSON file under the closed P5 v1 schema."""

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


def validate_runtime_environment(
    profile: AwsP5Profile,
    environment: Mapping[str, str],
) -> AwsP5Runtime:
    """Validate region, S3 prefix, AMI, digest, and absence of ambient secrets."""

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
    return AwsP5Runtime(
        region=region,
        s3_root=s3_root,
        ami_id=ami_id,
        container_digest=container_digest,
    )
