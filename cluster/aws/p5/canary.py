"""Pure profile-aware AWS GPU qualification receipt validation.

This module plans and validates qualification only.  It does not invoke Docker,
the AWS CLI, or any other external process.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence

from cluster.aws.p5.profile import AwsGpuProfile, AwsGpuRuntime


CANARY_UPDATES = 100
WARMUP_UPDATES = 10
MEASURED_UPDATES = CANARY_UPDATES - WARMUP_UPDATES
PRODUCTION_UPDATES_PER_ARM = 13_582
TOKENS_PER_UPDATE = 524_288
SEED_PAIRS = 10
_MAX_RECEIPT_BYTES = 1_048_576
_ARMS = ("dense", "split90")
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_type",
        "profile",
        "hardware",
        "software",
        "capabilities",
        "throughput",
    }
)
_V3_ROOT_FIELDS = _ROOT_FIELDS | frozenset(
    {"provenance", "raw_output_sha256"}
)
_PROFILE_FIELDS = frozenset({"profile_id", "provider", "profile_sha256"})
_PROVENANCE_FIELDS = frozenset(
    {
        "instance_id",
        "boot_id",
        "region",
        "ami_id",
        "ami_owner_id",
        "release_sha256",
        "container_image",
        "container_digest",
        "provider_selection_sha256",
        "environment_receipt_sha256",
        "qualified_at",
        "command_plan_sha256",
    }
)
_RAW_PHASES = frozenset(
    {
        "device_topology_software",
        "bf16",
        "sdpa",
        "torch_compile",
        "fused_adamw",
        "simultaneous_4_plus_4_nccl",
        "one_step_training",
        "checkpoint_resume",
        "nvme",
        "throughput",
    }
)
_HARDWARE_FIELDS = frozenset(
    {
        "instance_type",
        "vcpus",
        "memory_gib",
        "gpu_names",
        "gres",
        "cpu_affinity_halves",
        "train_groups",
    }
)
_SOFTWARE_FIELDS = frozenset(
    {"cuda", "driver", "linux_kernel", "efa", "ofi_nccl"}
)
_THROUGHPUT_FIELDS = frozenset(
    {
        "concurrent",
        "updates",
        "warmup_updates",
        "tokens_per_update",
        "arms",
    }
)
_ARM_FIELDS = frozenset({"arm", "gpu_ids", "update_seconds"})
_CAPABILITY_FIELDS = frozenset(
    {
        "fabric_manager_nvlink",
        "bf16",
        "sdpa",
        "torch_compile",
        "fused_adamw",
        "simultaneous_4_plus_4_nccl",
        "one_step_training",
        "checkpoint_resume",
        "nvme_geometry",
    }
)
_FABRIC_MANAGER_NVLINK_FIELDS = frozenset(
    {
        "passed",
        "fabric_manager_active",
        "nvlink_connected",
        "gpu_ids",
    }
)
_BF16_FIELDS = frozenset({"passed", "supported", "dtype", "gpu_ids"})
_SDPA_FIELDS = frozenset(
    {"passed", "forward", "backward", "dtype", "gpu_ids"}
)
_TORCH_COMPILE_FIELDS = frozenset(
    {"passed", "compiled", "backend", "gpu_ids"}
)
_FUSED_ADAMW_FIELDS = frozenset({"passed", "fused", "gpu_ids"})
_SIMULTANEOUS_NCCL_FIELDS = frozenset(
    {"passed", "concurrent", "backend", "groups", "world_sizes"}
)
_ONE_STEP_TRAINING_FIELDS = frozenset(
    {"passed", "arms", "updates", "gpu_groups"}
)
_CHECKPOINT_RESUME_FIELDS = frozenset(
    {"passed", "checkpointed", "resumed", "arms", "gpu_groups"}
)
_NVME_GEOMETRY_FIELDS = frozenset(
    {"passed", "model", "devices", "device_bytes", "raid_level"}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_INSTANCE_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
_AMI_RE = re.compile(r"^ami-[0-9a-f]{8,17}$")
_ACCOUNT_RE = re.compile(r"^[0-9]{12}$")
_REGION_RE = re.compile(r"^us-(?:east-1|west-2)$")
_BOOT_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_ECR_IMAGE_RE = re.compile(
    r"^(?P<account>[0-9]{12})\.dkr\.ecr\.(?P<region>us-(?:east-1|west-2))"
    r"\.amazonaws\.com/(?P<repository>[a-z0-9]+(?:[._/-][a-z0-9]+)*)"
    r"@(?P<digest>sha256:[0-9a-f]{64})$"
)
_VERSION_RE = re.compile(
    r"^R?([0-9]+(?:\.[0-9]+)*)(?:[-+._][A-Za-z0-9._+-]+)?$"
)


class QualificationError(ValueError):
    """The local qualification evidence does not match its closed contract."""


@dataclass(frozen=True)
class ThroughputResult:
    measured_updates_per_arm: int
    warmup_updates_discarded: int
    dense_tokens_per_second: float
    split90_tokens_per_second: float
    concurrent_tokens_per_second: float
    dense_seconds_per_update: float
    split90_seconds_per_update: float

    @property
    def seconds_per_seed_pair(self) -> float:
        return (
            max(
                self.dense_seconds_per_update,
                self.split90_seconds_per_update,
            )
            * PRODUCTION_UPDATES_PER_ARM
        )


@dataclass(frozen=True)
class QualificationReport:
    profile_id: str
    provider: str
    instance_type: str
    gpu_name: str
    throughput: ThroughputResult
    seed_pairs: int
    eta_seconds: float
    receipt_sha256: str | None = None
    provenance: Mapping[str, object] | None = None

    @property
    def eta_hours(self) -> float:
        return self.eta_seconds / 3600.0

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "profile_id": self.profile_id,
            "provider": self.provider,
            "instance_type": self.instance_type,
            "gpu_name": self.gpu_name,
            "geometry": {
                "train_groups": [4, 4],
                "measured_updates_per_arm": (
                    self.throughput.measured_updates_per_arm
                ),
                "warmup_updates_discarded": (
                    self.throughput.warmup_updates_discarded
                ),
            },
            "throughput": {
                "dense_tokens_per_second": (
                    self.throughput.dense_tokens_per_second
                ),
                "split90_tokens_per_second": (
                    self.throughput.split90_tokens_per_second
                ),
                "concurrent_tokens_per_second": (
                    self.throughput.concurrent_tokens_per_second
                ),
            },
            "eta": {
                "production_updates_per_arm": PRODUCTION_UPDATES_PER_ARM,
                "seed_pairs": self.seed_pairs,
                "seconds": self.eta_seconds,
                "hours": self.eta_hours,
            },
        }


def _object(
    value: object,
    fields: frozenset[str],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise QualificationError(f"{label} must be an object")
    actual = set(value)
    if actual != fields:
        raise QualificationError(
            f"{label} fields do not match the closed contract; "
            f"missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )
    return value


def _exact_int(value: object, expected: int, *, label: str) -> int:
    if type(value) is not int or value != expected:
        raise QualificationError(f"{label} must be exactly {expected}")
    return value


def _positive_metric(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise QualificationError(f"{label} must be finite and greater than zero")
    return float(value)


def _metric_series(
    values: object,
    *,
    label: str,
) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != CANARY_UPDATES:
        raise QualificationError(
            f"{label} must contain exactly {CANARY_UPDATES} update durations"
        )
    return tuple(
        _positive_metric(value, label=f"{label}[{index}]")
        for index, value in enumerate(values)
    )


def compute_concurrent_throughput(
    dense_update_seconds: Sequence[float],
    split90_update_seconds: Sequence[float],
    *,
    tokens_per_update: int = TOKENS_PER_UPDATE,
) -> ThroughputResult:
    """Compute conservative concurrent pair throughput after ten warmups."""

    _exact_int(
        tokens_per_update,
        TOKENS_PER_UPDATE,
        label="tokens_per_update",
    )
    dense = _metric_series(dense_update_seconds, label="dense update_seconds")
    split90 = _metric_series(
        split90_update_seconds,
        label="split90 update_seconds",
    )
    dense_measured = dense[WARMUP_UPDATES:]
    split_measured = split90[WARMUP_UPDATES:]
    dense_seconds = math.fsum(dense_measured)
    split_seconds = math.fsum(split_measured)
    measured_tokens = MEASURED_UPDATES * tokens_per_update
    pair_elapsed = max(dense_seconds, split_seconds)
    result = ThroughputResult(
        measured_updates_per_arm=MEASURED_UPDATES,
        warmup_updates_discarded=WARMUP_UPDATES,
        dense_tokens_per_second=measured_tokens / dense_seconds,
        split90_tokens_per_second=measured_tokens / split_seconds,
        concurrent_tokens_per_second=(2 * measured_tokens) / pair_elapsed,
        dense_seconds_per_update=dense_seconds / MEASURED_UPDATES,
        split90_seconds_per_update=split_seconds / MEASURED_UPDATES,
    )
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (
            result.dense_tokens_per_second,
            result.split90_tokens_per_second,
            result.concurrent_tokens_per_second,
            result.dense_seconds_per_update,
            result.split90_seconds_per_update,
        )
    ):
        raise QualificationError("computed throughput must be finite and positive")
    return result


def estimate_seed_pair_eta(
    throughput: ThroughputResult,
    *,
    seed_pairs: int = SEED_PAIRS,
) -> float:
    """Return seconds for the fixed production update count and seed-pair count."""

    _exact_int(seed_pairs, SEED_PAIRS, label="seed_pairs")
    seconds = throughput.seconds_per_seed_pair * seed_pairs
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise QualificationError("seed-pair ETA must be finite and positive")
    return seconds


def _version_tuple(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise QualificationError(f"{label} must be a version string")
    match = _VERSION_RE.fullmatch(value)
    if match is None:
        raise QualificationError(f"{label} is not a supported version string")
    return tuple(int(component) for component in match.group(1).split("."))


def _version_at_least(
    actual: object,
    minimum: str,
    *,
    label: str,
) -> None:
    actual_parts = _version_tuple(actual, label=label)
    minimum_parts = _version_tuple(minimum, label=f"{label} minimum")
    width = max(len(actual_parts), len(minimum_parts))
    if actual_parts + (0,) * (width - len(actual_parts)) < minimum_parts + (
        0,
    ) * (width - len(minimum_parts)):
        raise QualificationError(
            f"{label} {actual!r} is below required minimum {minimum}"
        )


def _validate_software(
    value: object,
    *,
    profile: AwsGpuProfile,
) -> None:
    software = _object(value, _SOFTWARE_FIELDS, label="software")
    minimums = profile.software_minimums
    for field in ("cuda", "driver", "linux_kernel", "efa", "ofi_nccl"):
        _version_at_least(
            software[field],
            minimums[field],
            label=f"software.{field}",
        )


def _same_typed_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _same_typed_value(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_typed_value(item, expected_item)
            for item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _validate_v3_provenance(
    value: object,
    raw_output_sha256: object,
    *,
    expected_instance_id: str | None,
    expected_boot_id: str | None,
    expected_provider_selection_sha256: str | None,
    expected_environment_receipt_sha256: str | None,
    expected_region: str | None,
    expected_ami_id: str | None,
    expected_ami_owner_id: str | None,
    expected_release_sha256: str | None,
    expected_container_image: str | None,
    expected_container_digest: str | None,
    expected_command_plan_sha256: str | None,
) -> Mapping[str, object]:
    provenance = _object(value, _PROVENANCE_FIELDS, label="provenance")
    raw = _object(
        raw_output_sha256,
        _RAW_PHASES,
        label="raw_output_sha256",
    )
    for phase, digest in raw.items():
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise QualificationError(
                f"raw_output_sha256.{phase} must be lowercase SHA-256"
            )
    for field in (
        "release_sha256",
        "provider_selection_sha256",
        "environment_receipt_sha256",
        "command_plan_sha256",
    ):
        digest = provenance[field]
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise QualificationError(f"provenance.{field} must be lowercase SHA-256")
    if (
        not isinstance(provenance["instance_id"], str)
        or _INSTANCE_RE.fullmatch(provenance["instance_id"]) is None
        or not isinstance(provenance["boot_id"], str)
        or _BOOT_ID_RE.fullmatch(provenance["boot_id"]) is None
        or not isinstance(provenance["region"], str)
        or _REGION_RE.fullmatch(provenance["region"]) is None
        or not isinstance(provenance["ami_id"], str)
        or _AMI_RE.fullmatch(provenance["ami_id"]) is None
        or not isinstance(provenance["ami_owner_id"], str)
        or _ACCOUNT_RE.fullmatch(provenance["ami_owner_id"]) is None
        or not isinstance(provenance["container_digest"], str)
        or _DIGEST_RE.fullmatch(provenance["container_digest"]) is None
        or not isinstance(provenance["container_image"], str)
    ):
        raise QualificationError("qualification provenance identity is invalid")
    image = _ECR_IMAGE_RE.fullmatch(str(provenance["container_image"]))
    if (
        image is None
        or image.group("region") != provenance["region"]
        or image.group("digest") != provenance["container_digest"]
    ):
        raise QualificationError(
            "qualification container is not a same-region private ECR digest"
        )
    qualified_at = provenance["qualified_at"]
    if not isinstance(qualified_at, str) or not qualified_at.endswith("Z"):
        raise QualificationError("provenance.qualified_at must be RFC 3339 UTC")
    try:
        parsed = datetime.fromisoformat(qualified_at[:-1] + "+00:00")
    except ValueError as error:
        raise QualificationError(
            "provenance.qualified_at must be RFC 3339 UTC"
        ) from error
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != UTC.utcoffset(parsed)
        or parsed.year < 2020
    ):
        raise QualificationError("provenance.qualified_at must be RFC 3339 UTC")
    expected = {
        "instance_id": expected_instance_id,
        "boot_id": expected_boot_id,
        "provider_selection_sha256": expected_provider_selection_sha256,
        "environment_receipt_sha256": expected_environment_receipt_sha256,
        "region": expected_region,
        "ami_id": expected_ami_id,
        "ami_owner_id": expected_ami_owner_id,
        "release_sha256": expected_release_sha256,
        "container_image": expected_container_image,
        "container_digest": expected_container_digest,
        "command_plan_sha256": expected_command_plan_sha256,
    }
    for field, expected_value in expected.items():
        if expected_value is not None and provenance[field] != expected_value:
            raise QualificationError(
                f"qualification provenance.{field} is stale or cross-runtime"
            )
    return provenance


def _exact_capability(
    value: object,
    *,
    fields: frozenset[str],
    expected: Mapping[str, object],
    label: str,
) -> None:
    capability = _object(value, fields, label=label)
    if capability.get("passed") is not True:
        raise QualificationError(f"{label}.passed must be true")
    if not _same_typed_value(dict(capability), dict(expected)):
        raise QualificationError(
            f"{label} evidence does not match the profile-bound contract"
        )


def _validate_capabilities(
    value: object,
    *,
    profile: AwsGpuProfile,
) -> None:
    capabilities = _object(value, _CAPABILITY_FIELDS, label="capabilities")
    gpu_ids = list(range(profile.allocated_gpus))
    groups = [
        list(range(0, profile.train_groups[0])),
        list(
            range(
                profile.train_groups[0],
                sum(profile.train_groups),
            )
        ),
    ]
    arms = list(_ARMS)
    contracts = {
        "fabric_manager_nvlink": (
            _FABRIC_MANAGER_NVLINK_FIELDS,
            {
                "passed": True,
                "fabric_manager_active": True,
                "nvlink_connected": True,
                "gpu_ids": gpu_ids,
            },
        ),
        "bf16": (
            _BF16_FIELDS,
            {
                "passed": True,
                "supported": True,
                "dtype": "bfloat16",
                "gpu_ids": gpu_ids,
            },
        ),
        "sdpa": (
            _SDPA_FIELDS,
            {
                "passed": True,
                "forward": True,
                "backward": True,
                "dtype": "bfloat16",
                "gpu_ids": gpu_ids,
            },
        ),
        "torch_compile": (
            _TORCH_COMPILE_FIELDS,
            {
                "passed": True,
                "compiled": True,
                "backend": "inductor",
                "gpu_ids": gpu_ids,
            },
        ),
        "fused_adamw": (
            _FUSED_ADAMW_FIELDS,
            {
                "passed": True,
                "fused": True,
                "gpu_ids": gpu_ids,
            },
        ),
        "simultaneous_4_plus_4_nccl": (
            _SIMULTANEOUS_NCCL_FIELDS,
            {
                "passed": True,
                "concurrent": True,
                "backend": "nccl",
                "groups": groups,
                "world_sizes": list(profile.train_groups),
            },
        ),
        "one_step_training": (
            _ONE_STEP_TRAINING_FIELDS,
            {
                "passed": True,
                "arms": arms,
                "updates": 1,
                "gpu_groups": groups,
            },
        ),
        "checkpoint_resume": (
            _CHECKPOINT_RESUME_FIELDS,
            {
                "passed": True,
                "checkpointed": True,
                "resumed": True,
                "arms": arms,
                "gpu_groups": groups,
            },
        ),
        "nvme_geometry": (
            _NVME_GEOMETRY_FIELDS,
            {
                "passed": True,
                "model": profile.instance_store_model,
                "devices": profile.instance_store_devices,
                "device_bytes": profile.instance_store_device_bytes,
                "raid_level": profile.raid_level,
            },
        ),
    }
    for name, (fields, expected) in contracts.items():
        _exact_capability(
            capabilities[name],
            fields=fields,
            expected=expected,
            label=f"capabilities.{name}",
        )


def validate_qualification_receipt(
    receipt: Mapping[str, object],
    profile: AwsGpuProfile,
    *,
    expected_instance_id: str | None = None,
    expected_boot_id: str | None = None,
    expected_provider_selection_sha256: str | None = None,
    expected_environment_receipt_sha256: str | None = None,
    expected_region: str | None = None,
    expected_ami_id: str | None = None,
    expected_ami_owner_id: str | None = None,
    expected_release_sha256: str | None = None,
    expected_container_image: str | None = None,
    expected_container_digest: str | None = None,
    expected_command_plan_sha256: str | None = None,
    receipt_sha256: str | None = None,
) -> QualificationReport:
    """Validate one closed receipt and return throughput plus ten-pair ETA."""

    schema_version = receipt.get("schema_version")
    fields = _V3_ROOT_FIELDS if schema_version == 3 else _ROOT_FIELDS
    root = _object(receipt, fields, label="qualification receipt")
    if schema_version not in {2, 3}:
        raise QualificationError(
            "qualification receipt.schema_version must be exactly 2 or 3"
        )
    if schema_version == 3 and profile.schema_version != 3:
        raise QualificationError("v3 qualification requires a v3 profile")
    if root["receipt_type"] != "aws-gpu-qualification":
        raise QualificationError("qualification receipt type does not match")

    identity = _object(root["profile"], _PROFILE_FIELDS, label="profile")
    if (
        identity["profile_id"] != profile.profile_id
        or identity["provider"] != profile.provider
        or identity["profile_sha256"] != profile.sha256
        or not isinstance(identity["profile_sha256"], str)
        or _SHA256_RE.fullmatch(identity["profile_sha256"]) is None
    ):
        raise QualificationError(
            "qualification receipt is bound to a different profile"
        )

    hardware = _object(root["hardware"], _HARDWARE_FIELDS, label="hardware")
    expected_hardware = {
        "instance_type": profile.instance_type,
        "vcpus": profile.vcpus,
        "memory_gib": profile.memory_gib,
        "gres": profile.gres,
        "cpu_affinity_halves": [
            list(group) for group in profile.cpu_affinity_halves
        ],
        "train_groups": list(profile.train_groups),
    }
    for field, expected in expected_hardware.items():
        if (
            type(hardware[field]) is not type(expected)
            or hardware[field] != expected
        ):
            raise QualificationError(
                f"hardware.{field} does not match profile {profile.profile_id}"
            )
    gpu_names = hardware["gpu_names"]
    if (
        not isinstance(gpu_names, list)
        or len(gpu_names) != profile.allocated_gpus
        or any(not profile.matches_gpu_name(name) for name in gpu_names)
    ):
        raise QualificationError("hardware GPU names do not match the profile")
    if len(set(gpu_names)) != 1:
        raise QualificationError("qualification receipt mixes GPU hardware")

    _validate_software(root["software"], profile=profile)
    _validate_capabilities(root["capabilities"], profile=profile)

    throughput_value = _object(
        root["throughput"],
        _THROUGHPUT_FIELDS,
        label="throughput",
    )
    if throughput_value["concurrent"] is not True:
        raise QualificationError("throughput canary must run both arms concurrently")
    _exact_int(
        throughput_value["updates"],
        CANARY_UPDATES,
        label="throughput.updates",
    )
    _exact_int(
        throughput_value["warmup_updates"],
        WARMUP_UPDATES,
        label="throughput.warmup_updates",
    )
    _exact_int(
        throughput_value["tokens_per_update"],
        TOKENS_PER_UPDATE,
        label="throughput.tokens_per_update",
    )
    arms = throughput_value["arms"]
    if not isinstance(arms, list) or len(arms) != 2:
        raise QualificationError("throughput must contain one complete arm pair")
    by_arm: dict[str, Mapping[str, object]] = {}
    expected_gpu_ids = {
        "dense": list(range(0, profile.train_groups[0])),
        "split90": list(
            range(profile.train_groups[0], sum(profile.train_groups))
        ),
    }
    for index, raw_arm in enumerate(arms):
        arm = _object(raw_arm, _ARM_FIELDS, label=f"throughput.arms[{index}]")
        name = arm["arm"]
        if name not in _ARMS or name in by_arm:
            raise QualificationError(
                "throughput arms must be unique dense and split90 entries"
            )
        if arm["gpu_ids"] != expected_gpu_ids[name]:
            raise QualificationError(
                f"throughput {name} GPU IDs do not form the required 4+4 pair"
            )
        by_arm[name] = arm
    if set(by_arm) != set(_ARMS):
        raise QualificationError(
            "throughput must contain both dense and split90 arms"
        )
    throughput = compute_concurrent_throughput(
        by_arm["dense"]["update_seconds"],
        by_arm["split90"]["update_seconds"],
        tokens_per_update=throughput_value["tokens_per_update"],
    )
    provenance = None
    if schema_version == 3:
        provenance = _validate_v3_provenance(
            root["provenance"],
            root["raw_output_sha256"],
            expected_instance_id=expected_instance_id,
            expected_boot_id=expected_boot_id,
            expected_provider_selection_sha256=(
                expected_provider_selection_sha256
            ),
            expected_environment_receipt_sha256=(
                expected_environment_receipt_sha256
            ),
            expected_region=expected_region,
            expected_ami_id=expected_ami_id,
            expected_ami_owner_id=expected_ami_owner_id,
            expected_release_sha256=expected_release_sha256,
            expected_container_image=expected_container_image,
            expected_container_digest=expected_container_digest,
            expected_command_plan_sha256=expected_command_plan_sha256,
        )
    return QualificationReport(
        profile_id=profile.profile_id,
        provider=profile.provider,
        instance_type=profile.instance_type,
        gpu_name=gpu_names[0],
        throughput=throughput,
        seed_pairs=SEED_PAIRS,
        eta_seconds=estimate_seed_pair_eta(throughput),
        receipt_sha256=receipt_sha256,
        provenance=provenance,
    )


def build_qualification_receipt(
    profile: AwsGpuProfile,
    phase_receipts: Sequence[Mapping[str, object]],
    *,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Assemble only validated runtime phase receipts into one v3 receipt."""

    from cluster.aws.p5.canary_runtime import validate_phase_receipt

    if profile.schema_version != 3:
        raise QualificationError("phase assembly requires one v3 profile")
    expected_keys = {
        ("device-topology-software", None),
        ("bf16", None),
        ("sdpa", None),
        ("torch-compile", None),
        ("fused-adamw", None),
        ("simultaneous-4-plus-4-nccl", None),
        ("nvme", None),
        *(("one-step-training", arm) for arm in _ARMS),
        *(("checkpoint", arm) for arm in _ARMS),
        *(("resume", arm) for arm in _ARMS),
        *(("throughput", arm) for arm in _ARMS),
    }
    by_key: dict[tuple[object, object], dict[str, object]] = {}
    for raw in phase_receipts:
        try:
            receipt = validate_phase_receipt(dict(raw))
        except (TypeError, ValueError) as error:
            raise QualificationError("canary phase receipt is invalid") from error
        key = (receipt["phase"], receipt["arm"])
        if key in by_key:
            raise QualificationError("canary phase receipt is duplicated")
        if (
            receipt["provider"] != profile.provider
            or receipt["instance_type"] != profile.instance_type
            or receipt["profile_sha256"] != profile.sha256
        ):
            raise QualificationError("canary phase receipt is cross-profile")
        by_key[key] = receipt
    if set(by_key) != expected_keys:
        raise QualificationError("canary phase receipt inventory is not exact")

    all_gpu_ids = list(range(profile.allocated_gpus))
    gpu_groups = {
        "dense": list(range(0, profile.train_groups[0])),
        "split90": list(
            range(profile.train_groups[0], sum(profile.train_groups))
        ),
    }
    for (phase, arm), receipt in by_key.items():
        expected_ids = all_gpu_ids if arm is None else gpu_groups[str(arm)]
        if receipt["gpu_ids"] != expected_ids:
            raise QualificationError(
                f"canary {phase} receipt has the wrong GPU identity"
            )
    identity_values = {
        (
            receipt["release_sha256"],
            receipt["container_image"],
            receipt["container_digest"],
            receipt["gres"],
        )
        for receipt in by_key.values()
    }
    if len(identity_values) != 1:
        raise QualificationError("canary phase receipts mix runtime identities")
    (
        phase_release_sha256,
        phase_container_image,
        phase_container_digest,
        phase_gres,
    ) = next(iter(identity_values))
    if (
        phase_release_sha256 != provenance.get("release_sha256")
        or phase_container_image != provenance.get("container_image")
        or phase_container_digest != provenance.get("container_digest")
        or phase_gres != profile.gres
    ):
        raise QualificationError(
            "canary phase receipts conflict with qualification provenance"
        )

    def evidence(phase: str, arm: str | None = None) -> dict[str, object]:
        value = by_key[(phase, arm)]["evidence"]
        if not isinstance(value, dict):
            raise QualificationError("canary phase evidence must be an object")
        return value

    device = evidence("device-topology-software")
    bf16 = evidence("bf16")
    sdpa = evidence("sdpa")
    compiled = evidence("torch-compile")
    adamw = evidence("fused-adamw")
    nccl = evidence("simultaneous-4-plus-4-nccl")
    nvme = evidence("nvme")
    one_step = {arm: evidence("one-step-training", arm) for arm in _ARMS}
    checkpoints = {arm: evidence("checkpoint", arm) for arm in _ARMS}
    resumes = {arm: evidence("resume", arm) for arm in _ARMS}
    throughputs = {arm: evidence("throughput", arm) for arm in _ARMS}
    if any(
        one_step[arm].get("arm") != arm
        or one_step[arm].get("updates") != 1
        or checkpoints[arm].get("arm") != arm
        or checkpoints[arm].get("checkpointed") is not True
        or resumes[arm].get("arm") != arm
        or resumes[arm].get("resumed") is not True
        or checkpoints[arm].get("checkpoint_sha256")
        != resumes[arm].get("checkpoint_sha256")
        or throughputs[arm].get("arm") != arm
        or throughputs[arm].get("updates") != CANARY_UPDATES
        or throughputs[arm].get("warmup_updates") != WARMUP_UPDATES
        or throughputs[arm].get("tokens_per_update") != TOKENS_PER_UPDATE
        or not isinstance(throughputs[arm].get("update_seconds"), list)
        for arm in _ARMS
    ):
        raise QualificationError("paired canary phase evidence is inconsistent")

    def raw_hash(*keys: tuple[str, str | None]) -> str:
        values = {
            f"{phase}:{arm or 'all'}": by_key[(phase, arm)][
                "raw_output_sha256"
            ]
            for phase, arm in keys
        }
        if len(values) == 1:
            return str(next(iter(values.values())))
        return hashlib.sha256(
            (
                json.dumps(
                    values,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("ascii")
        ).hexdigest()

    receipt = {
        "schema_version": 3,
        "receipt_type": "aws-gpu-qualification",
        "profile": {
            "profile_id": profile.profile_id,
            "provider": profile.provider,
            "profile_sha256": profile.sha256,
        },
        "hardware": {
            "instance_type": profile.instance_type,
            "vcpus": profile.vcpus,
            "memory_gib": profile.memory_gib,
            "gpu_names": device.get("gpu_names"),
            "gres": profile.gres,
            "cpu_affinity_halves": [
                list(group) for group in profile.cpu_affinity_halves
            ],
            "train_groups": list(profile.train_groups),
        },
        "software": {
            field: device.get(field)
            for field in _SOFTWARE_FIELDS
        },
        "capabilities": {
            "fabric_manager_nvlink": {
                "passed": True,
                "fabric_manager_active": device.get("fabric_manager_active"),
                "nvlink_connected": device.get("nvlink_connected"),
                "gpu_ids": all_gpu_ids,
            },
            "bf16": {
                "passed": True,
                "supported": bf16.get("supported"),
                "dtype": bf16.get("dtype"),
                "gpu_ids": all_gpu_ids,
            },
            "sdpa": {
                "passed": True,
                "forward": sdpa.get("forward"),
                "backward": sdpa.get("backward"),
                "dtype": sdpa.get("dtype"),
                "gpu_ids": all_gpu_ids,
            },
            "torch_compile": {
                "passed": True,
                "compiled": compiled.get("compiled"),
                "backend": compiled.get("backend"),
                "gpu_ids": all_gpu_ids,
            },
            "fused_adamw": {
                "passed": True,
                "fused": adamw.get("fused"),
                "gpu_ids": all_gpu_ids,
            },
            "simultaneous_4_plus_4_nccl": {
                "passed": True,
                "concurrent": nccl.get("concurrent"),
                "backend": nccl.get("backend"),
                "groups": nccl.get("groups"),
                "world_sizes": nccl.get("world_sizes"),
            },
            "one_step_training": {
                "passed": True,
                "arms": list(_ARMS),
                "updates": 1,
                "gpu_groups": [gpu_groups[arm] for arm in _ARMS],
            },
            "checkpoint_resume": {
                "passed": True,
                "checkpointed": True,
                "resumed": True,
                "arms": list(_ARMS),
                "gpu_groups": [gpu_groups[arm] for arm in _ARMS],
            },
            "nvme_geometry": {
                "passed": True,
                "model": nvme.get("model"),
                "devices": nvme.get("devices"),
                "device_bytes": nvme.get("device_bytes"),
                "raid_level": nvme.get("raid_level"),
            },
        },
        "throughput": {
            "concurrent": True,
            "updates": CANARY_UPDATES,
            "warmup_updates": WARMUP_UPDATES,
            "tokens_per_update": TOKENS_PER_UPDATE,
            "arms": [
                {
                    "arm": arm,
                    "gpu_ids": gpu_groups[arm],
                    "update_seconds": throughputs[arm]["update_seconds"],
                }
                for arm in _ARMS
            ],
        },
        "provenance": dict(provenance),
        "raw_output_sha256": {
            "device_topology_software": raw_hash(
                ("device-topology-software", None)
            ),
            "bf16": raw_hash(("bf16", None)),
            "sdpa": raw_hash(("sdpa", None)),
            "torch_compile": raw_hash(("torch-compile", None)),
            "fused_adamw": raw_hash(("fused-adamw", None)),
            "simultaneous_4_plus_4_nccl": raw_hash(
                ("simultaneous-4-plus-4-nccl", None)
            ),
            "one_step_training": raw_hash(
                *(("one-step-training", arm) for arm in _ARMS)
            ),
            "checkpoint_resume": raw_hash(
                *(("checkpoint", arm) for arm in _ARMS),
                *(("resume", arm) for arm in _ARMS),
            ),
            "nvme": raw_hash(("nvme", None)),
            "throughput": raw_hash(
                *(("throughput", arm) for arm in _ARMS)
            ),
        },
    }
    validate_qualification_receipt(
        receipt,
        profile,
        expected_instance_id=provenance.get("instance_id"),
        expected_provider_selection_sha256=provenance.get(
            "provider_selection_sha256"
        ),
        expected_environment_receipt_sha256=provenance.get(
            "environment_receipt_sha256"
        ),
        expected_region=provenance.get("region"),
        expected_ami_id=provenance.get("ami_id"),
        expected_ami_owner_id=provenance.get("ami_owner_id"),
        expected_release_sha256=provenance.get("release_sha256"),
        expected_container_image=provenance.get("container_image"),
        expected_container_digest=provenance.get("container_digest"),
        expected_command_plan_sha256=provenance.get("command_plan_sha256"),
    )
    return receipt


def _canary_container_argv(
    *,
    profile: AwsGpuProfile,
    runtime: AwsGpuRuntime,
    release_root: str,
    release_sha256: str,
    output_root: str,
    phase: str,
    arm: str | None,
    gpu_ids: Sequence[int],
    output_name: str,
    extra: Sequence[str] = (),
) -> list[str]:
    ids = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    container_output = f"/qualification/{output_name}"
    argv = [
        "/usr/bin/docker",
        "run",
        "--rm",
        "--read-only",
        "--network=host",
        "--ipc=host",
        "--gpus",
        f"device={ids}",
        "--user",
        f"{runtime.uid}:{runtime.gid}",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--mount",
        f"type=bind,src={output_root},dst=/qualification",
        "--mount",
        f"type=bind,src={release_root},dst=/workspace,readonly",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,exec,size=8589934592",
        "--env",
        "HOME=/tmp",
        "--env",
        "PYTHONNOUSERSITE=1",
        runtime.container_image,
        "/opt/venv/bin/python",
        "/workspace/cluster/aws/p5/canary_runtime.py",
        phase,
        "--provider",
        profile.provider,
        "--instance-type",
        profile.instance_type,
        "--profile-sha256",
        profile.sha256,
        "--release-sha256",
        release_sha256,
        "--release-root",
        "/workspace",
        "--container-image",
        runtime.container_image,
        "--container-digest",
        runtime.container_digest,
        "--gres",
        profile.gres,
        "--gpu-ids",
        ids,
    ]
    if arm is not None:
        argv.extend(["--arm", arm])
    argv.extend(["--output", container_output, *extra])
    return argv


def render_canary_command_plan(
    profile: AwsGpuProfile,
    runtime: AwsGpuRuntime,
    *,
    release_sha256: str,
    release_root: str,
) -> dict[str, object]:
    """Render a deterministic remote qualification plan without executing it."""

    if profile.provider != profile.profile_id:
        raise QualificationError("canary profile provider identity is invalid")
    if (
        not isinstance(profile.sha256, str)
        or _SHA256_RE.fullmatch(profile.sha256) is None
    ):
        raise QualificationError("canary profile SHA-256 is invalid")
    digest_prefix, separator, image_digest = runtime.container_digest.partition(":")
    if (
        digest_prefix != "sha256"
        or separator != ":"
        or _SHA256_RE.fullmatch(image_digest) is None
        or not runtime.container_image.endswith("@" + runtime.container_digest)
    ):
        raise QualificationError("canary container image is not digest pinned")
    if (
        not isinstance(release_sha256, str)
        or _SHA256_RE.fullmatch(release_sha256) is None
    ):
        raise QualificationError("canary release SHA-256 is invalid")
    expected_release_root = (
        f"{profile.scratch_root}/releases/{release_sha256}"
    )
    if (
        release_root != expected_release_root
        or not release_root.startswith("/")
        or any(character in release_root for character in ",\n\r\x00")
    ):
        raise QualificationError(
            "canary must execute from the digest-named extracted release"
        )
    dense_ids = list(range(0, profile.train_groups[0]))
    split90_ids = list(
        range(profile.train_groups[0], sum(profile.train_groups))
    )
    all_ids = dense_ids + split90_ids
    if (
        len(all_ids) != profile.allocated_gpus
        or dense_ids != [0, 1, 2, 3]
        or split90_ids != [4, 5, 6, 7]
    ):
        raise QualificationError("canary command plan requires exact GPU groups 0-3 and 4-7")
    output_root = (
        f"{profile.scratch_root}/qualification/{profile.profile_id}/"
        f"release-{release_sha256}/profile-{profile.sha256}/image-{image_digest}"
    )
    serial_commands = [
        _canary_container_argv(
            profile=profile,
            runtime=runtime,
            release_root=release_root,
            release_sha256=release_sha256,
            output_root=output_root,
            phase=phase,
            arm=None,
            gpu_ids=all_ids,
            output_name=f"receipts/{phase}.json",
            extra=extra,
        )
        for phase, extra in (
            ("device-topology-software", ()),
            ("bf16", ()),
            ("sdpa", ()),
            ("torch-compile", ()),
            ("fused-adamw", ()),
            ("simultaneous-4-plus-4-nccl", ()),
            (
                "nvme",
                (
                    "--expected-nvme-model",
                    profile.instance_store_model,
                    "--expected-nvme-devices",
                    str(profile.instance_store_devices),
                    "--expected-nvme-device-bytes",
                    str(profile.instance_store_device_bytes),
                    "--expected-raid-level",
                    profile.raid_level,
                ),
            ),
        )
    ]
    one_step_commands = [
        _canary_container_argv(
            profile=profile,
            runtime=runtime,
            release_root=release_root,
            release_sha256=release_sha256,
            output_root=output_root,
            phase="one-step-training",
            arm=arm,
            gpu_ids=gpu_ids,
            output_name=f"receipts/one-step-training-{arm}.json",
        )
        for arm, gpu_ids in (("dense", dense_ids), ("split90", split90_ids))
    ]
    checkpoint_resume_commands = [
        command
        for arm, gpu_ids in (("dense", dense_ids), ("split90", split90_ids))
        for command in (
            _canary_container_argv(
                profile=profile,
                runtime=runtime,
                release_root=release_root,
                release_sha256=release_sha256,
                output_root=output_root,
                phase="checkpoint",
                arm=arm,
                gpu_ids=gpu_ids,
                output_name=f"receipts/checkpoint-{arm}.json",
                extra=(
                    "--checkpoint",
                    f"/qualification/checkpoints/{arm}.pt",
                ),
            ),
            _canary_container_argv(
                profile=profile,
                runtime=runtime,
                release_root=release_root,
                release_sha256=release_sha256,
                output_root=output_root,
                phase="resume",
                arm=arm,
                gpu_ids=gpu_ids,
                output_name=f"resume/{arm}.json",
                extra=(
                    "--checkpoint",
                    f"/qualification/checkpoints/{arm}.pt",
                ),
            ),
        )
    ]
    throughput_commands = [
        _canary_container_argv(
            profile=profile,
            runtime=runtime,
            release_root=release_root,
            release_sha256=release_sha256,
            output_root=output_root,
            phase="throughput",
            arm=arm,
            gpu_ids=gpu_ids,
            output_name=f"receipts/throughput-{arm}.json",
            extra=(
                "--updates",
                str(CANARY_UPDATES),
                "--warmup-updates",
                str(WARMUP_UPDATES),
                "--tokens-per-update",
                str(TOKENS_PER_UPDATE),
            ),
        )
        for arm, gpu_ids in (("dense", dense_ids), ("split90", split90_ids))
    ]
    return {
        "schema_version": 3,
        "provider": profile.provider,
        "instance_type": profile.instance_type,
        "profile_sha256": profile.sha256,
        "release_sha256": release_sha256,
        "release_root": release_root,
        "container_image": runtime.container_image,
        "container_digest": runtime.container_digest,
        "output_root": output_root,
        "gpu_groups": {
            "dense": dense_ids,
            "split90": split90_ids,
        },
        "phases": {
            "prepare": {
                "concurrent": False,
                "commands": [
                    [
                        "/usr/bin/install",
                        "-d",
                        "-m",
                        "0700",
                        "-o",
                        str(runtime.uid),
                        "-g",
                        str(runtime.gid),
                        output_root,
                        f"{output_root}/receipts",
                        f"{output_root}/checkpoints",
                        f"{output_root}/resume",
                    ]
                ],
            },
            "serial": {
                "concurrent": False,
                "commands": serial_commands,
            },
            "one_step_training": {
                "concurrent": True,
                "commands": one_step_commands,
            },
            "checkpoint_resume": {
                "concurrent": False,
                "commands": checkpoint_resume_commands,
            },
            "throughput": {
                "concurrent": True,
                "commands": throughput_commands,
            },
        },
    }


render_remote_canary_plan = render_canary_command_plan


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise QualificationError(f"receipt repeats JSON field: {key}")
        value[key] = item
    return value


def load_qualification_receipt(
    path: Path | str,
    profile: AwsGpuProfile,
    **kwargs: object,
) -> QualificationReport:
    """Load a bounded regular JSON receipt and validate it without side effects."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise QualificationError("qualification receipt must be a regular file")
    payload = candidate.read_bytes()
    if len(payload) > _MAX_RECEIPT_BYTES:
        raise QualificationError("qualification receipt exceeds 1 MiB")
    try:
        receipt = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                QualificationError(
                    f"qualification receipt contains non-finite {constant}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(
            "qualification receipt must contain valid UTF-8 JSON"
        ) from error
    if receipt.get("schema_version") == 3:
        canonical = (
            json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
        if payload != canonical:
            raise QualificationError(
                "v3 qualification receipt must be canonical JSON plus one newline"
            )
    return validate_qualification_receipt(
        receipt,
        profile,
        receipt_sha256=hashlib.sha256(payload).hexdigest(),
        **kwargs,
    )


build_qualification_report = validate_qualification_receipt
