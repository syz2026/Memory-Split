"""Pure profile-aware AWS GPU qualification receipt validation.

This module plans and validates qualification only.  It does not invoke Docker,
the AWS CLI, or any other external process.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from cluster.aws.p5.profile import AwsGpuProfile


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
        "throughput",
    }
)
_PROFILE_FIELDS = frozenset({"profile_id", "provider", "profile_sha256"})
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
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
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


def validate_qualification_receipt(
    receipt: Mapping[str, object],
    profile: AwsGpuProfile,
) -> QualificationReport:
    """Validate one closed receipt and return throughput plus ten-pair ETA."""

    root = _object(receipt, _ROOT_FIELDS, label="qualification receipt")
    _exact_int(
        root["schema_version"],
        1,
        label="qualification receipt.schema_version",
    )
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
    return QualificationReport(
        profile_id=profile.profile_id,
        provider=profile.provider,
        instance_type=profile.instance_type,
        gpu_name=gpu_names[0],
        throughput=throughput,
        seed_pairs=SEED_PAIRS,
        eta_seconds=estimate_seed_pair_eta(throughput),
    )


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
    return validate_qualification_receipt(receipt, profile)


build_qualification_report = validate_qualification_receipt
