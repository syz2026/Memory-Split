"""Profile-driven AWS GPU qualification contracts.

This module is intentionally side-effect free.  Legacy P5 producers keep their
existing schemas and entry points; selected-profile producers call these
helpers after an authenticated provider selection has been admitted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from cluster.aws.gpu_profile import AwsGpuProfile
from msctl.aws_contracts import (
    AWS_GPU_ATTESTATION_EVIDENCE_V1_FIELDS,
    AWS_RUNTIME_LOCK_FIELDS,
    AWS_RUNTIME_VERSION_FIELDS,
    validate_digest_pinned_oci_image,
    validate_gpu_product_names,
)
from msctl.aws_hardware import AuthenticatedSelectionBinding


P6_PROFILE_ID = "aws-p6-b300.48xlarge-v3"
P6_BASE_AMI_ID = "ami-0260c4d597dcc8641"
P6_BASE_AMI_OWNER_ID = "898082745236"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_ACCOUNT_RE = re.compile(r"^[0-9]{12}$")
_INSTANCE_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
_BOOT_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_CONTAINER_FACT_FIELDS = frozenset({"python", "pytorch", "cuda", "cudnn", "nccl"})
_P6_HOST_FLOORS = {
    "cuda": "13.0",
    "nvidia_driver": "580.0",
    "nvlsm": "580.0",
    "kernel": "6.1",
    "efa": "1.44.0",
    "ofi_nccl": "1.17.1",
}
_SELECTION_FIELD_NAMES = frozenset(
    {
        "cohort_id",
        "hardware_amendment_sha256",
        "provider_selection_sha256",
        "provider_selection_version_id",
        "profile_id",
        "provider",
        "profile_sha256",
        "qualification_evidence_sha256",
        "preselection_environment_receipt_sha256",
        "preselection_canary_receipt_sha256",
        "approval_receipt_sha256",
        "approval_public_key_sha256",
        "account_id",
        "instance_id",
        "boot_id",
        "region",
        "availability_zone",
        "purchase_model",
    }
)
_SELECTED_ENVIRONMENT_FIELDS = _SELECTION_FIELD_NAMES | {
    "schema_version",
    "receipt_type",
    "runtime_lock_sha256",
    "runtime_sbom_sha256",
    "attestation_evidence_sha256",
    "ami_id",
    "ami_owner_id",
    "container_image",
    "container_image_digest",
    "host_facts",
    "container_facts",
}
_SELECTED_BOOTSTRAP_FIELDS = _SELECTION_FIELD_NAMES | {
    "schema_version",
    "receipt_type",
    "runtime_lock_sha256",
    "runtime_sbom_sha256",
    "environment_receipt_sha256",
    "ami_id",
    "ami_owner_id",
    "container_image",
    "container_image_digest",
    "hardware",
}
SELECTED_PHASE_ORDER = (
    "hardware",
    "container",
    "nccl_4x4",
    "training",
    "s3_roundtrip",
    "telemetry_eta",
)
SELECTED_TARGET_TOKENS_PER_ARM = 7_120_879_616
SELECTED_COHORT_PAIRS = 10
SELECTED_THROUGHPUT_UPDATES = 100
SELECTED_THROUGHPUT_WARMUP_UPDATES = 10
_SELECTED_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_type",
        "cohort_id",
        "hardware_amendment_sha256",
        "provider_selection_sha256",
        "provider_selection_version_id",
        "profile_id",
        "provider",
        "profile_sha256",
        "qualification_evidence_sha256",
        "preselection_environment_receipt_sha256",
        "preselection_canary_receipt_sha256",
        "approval_receipt_sha256",
        "approval_public_key_sha256",
        "account_id",
        "instance_id",
        "boot_id",
        "region",
        "availability_zone",
        "purchase_model",
        "runtime_lock_sha256",
        "runtime_sbom_sha256",
        "environment_receipt_sha256",
        "bootstrap_receipt_sha256",
        "ami_id",
        "ami_owner_id",
        "container_image",
        "container_image_digest",
        "phases",
        "hardware",
        "container",
        "nccl_groups",
        "training",
        "s3_roundtrip",
        "telemetry",
        "passed",
        "started_at",
        "ended_at",
        "total_seconds",
    }
)


@dataclass(frozen=True)
class RuntimeQualificationBundle:
    """Validated runtime-lock and SBOM bytes for one selected profile."""

    lock: Mapping[str, object]
    lock_sha256: str
    sbom: Mapping[str, object]
    sbom_sha256: str


@dataclass(frozen=True)
class QualificationCommandSpec:
    """One closed qualification command rendered for an injected runner."""

    name: str
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    cwd: Path
    timeout_seconds: float
    arm: str | None = None
    gpu_ids: tuple[int, ...] = ()
    cpu_affinity: tuple[int, int] | None = None
    world_size: int | None = None
    master_port: int | None = None
    concurrent_group: str | None = None


@dataclass(frozen=True)
class QualificationCommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class QualificationObjectWrite:
    sha256: str
    bytes: int
    version_id: str


@dataclass(frozen=True)
class QualificationObjectRead:
    payload: bytes
    sha256: str
    version_id: str


class QualificationCommandRunner(Protocol):
    def run(self, spec: QualificationCommandSpec) -> QualificationCommandResult: ...

    def run_pair(
        self,
        specs: Sequence[QualificationCommandSpec],
    ) -> Mapping[str, QualificationCommandResult]: ...


class QualificationObjectStore(Protocol):
    def put(
        self,
        uri: str,
        payload: bytes,
        *,
        sha256: str,
    ) -> QualificationObjectWrite: ...

    def get(
        self,
        uri: str,
        *,
        version_id: str,
    ) -> QualificationObjectRead: ...


class QualificationTimeReader(Protocol):
    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...


@dataclass(frozen=True)
class SelectedCanaryPlan:
    """All immutable inputs and profile-derived execution geometry."""

    profile: AwsGpuProfile
    binding: AuthenticatedSelectionBinding
    bundle: RuntimeQualificationBundle
    environment_receipt: Mapping[str, object]
    environment_receipt_sha256: str
    bootstrap_receipt: Mapping[str, object]
    bootstrap_receipt_sha256: str
    scratch_root: Path
    output_path: Path
    s3_root: str
    gpu_groups: Mapping[str, tuple[int, ...]]
    cpu_affinities: Mapping[str, tuple[int, int]]
    master_ports: Mapping[str, int]
    target_tokens_per_arm: int
    cohort_pairs: int


def canonical_qualification_json(value: object) -> bytes:
    """Serialize one qualification object as canonical ASCII JSON."""

    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("qualification value is not canonical JSON") from error


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"qualification JSON repeats field: {key}")
        value[key] = item
    return value


def _canonical_object(data: bytes, *, label: str) -> dict[str, object]:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} input must be bytes")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite {constant}")
            ),
        )
    except ValueError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict) or data != canonical_qualification_json(value):
        raise ValueError(f"{label} must be one canonical JSON object")
    return value


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _fixed_text(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character in value for character in "\x00\n\r")
    ):
        raise ValueError(f"{label} must be one fixed nonempty string")
    return value


def _version_parts(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a measured version")
    normalized = value[1:] if value.startswith("R") else value
    match = re.match(r"^([0-9]+(?:\.[0-9]+)*)", normalized)
    if match is None:
        raise ValueError(f"{label} must start with a numeric version")
    return tuple(int(part) for part in match.group(1).split("."))


def _at_least(actual: object, floor: str, *, label: str) -> None:
    measured = _version_parts(actual, label=label)
    minimum = _version_parts(floor, label=f"{label} floor")
    width = max(len(measured), len(minimum))
    if measured + (0,) * (width - len(measured)) < minimum + (0,) * (
        width - len(minimum)
    ):
        raise ValueError(f"{label} is below the selected profile floor")


def validate_authenticated_selection(
    profile: AwsGpuProfile,
    binding: AuthenticatedSelectionBinding,
) -> AuthenticatedSelectionBinding:
    """Require a binding for exactly this immutable profile and placement."""

    if not isinstance(profile, AwsGpuProfile):
        raise TypeError("selected profile must be an AwsGpuProfile")
    if not isinstance(binding, AuthenticatedSelectionBinding):
        raise TypeError("selection binding must be authenticated")
    if (
        binding.profile_id != profile.profile_id
        or binding.provider != profile.provider
        or binding.profile_sha256 != profile.sha256
        or binding.cohort_id != "memorysplit-confirmatory-v3-360m-n10-aws"
        or binding.purchase_model != profile.purchase_model
        or binding.region not in profile.allowed_regions
        or (
            profile.allowed_availability_zones
            and binding.availability_zone
            not in profile.allowed_availability_zones
        )
    ):
        raise ValueError("authenticated selection binding differs from profile")
    for label, value in (
        ("hardware amendment", binding.amendment_sha256),
        ("provider selection", binding.selection_sha256),
        ("runtime lock", binding.runtime_lock_sha256),
        ("qualification evidence", binding.qualification_evidence_sha256),
        ("environment receipt", binding.environment_receipt_sha256),
        ("canary receipt", binding.canary_receipt_sha256),
        ("approval receipt", binding.approval_receipt_sha256),
        ("approval public key", binding.approval_public_key_sha256),
    ):
        _sha256(value, label=f"selection binding {label}")
    if (
        not isinstance(binding.selection_version_id, str)
        or binding.selection_version_id in {"", "null"}
        or any(
            character in binding.selection_version_id
            for character in "\x00\n\r"
        )
        or _ACCOUNT_RE.fullmatch(binding.account_id) is None
        or _INSTANCE_RE.fullmatch(binding.instance_id) is None
        or _BOOT_RE.fullmatch(binding.boot_id) is None
        or type(binding.seed) is not int
        or binding.seed not in profile.assigned_seeds
        or binding.arm not in {"dense", "split90"}
    ):
        raise ValueError("authenticated selection binding identity is invalid")
    return binding


def _parse_runtime_lock(
    profile: AwsGpuProfile,
    binding: AuthenticatedSelectionBinding,
    data: bytes,
) -> dict[str, object]:
    lock = _canonical_object(data, label="qualification runtime lock")
    if (
        set(lock) != set(AWS_RUNTIME_LOCK_FIELDS)
        or type(lock.get("schema_version")) is not int
        or lock["schema_version"] != 1
        or not isinstance(lock.get("versions"), dict)
        or set(lock["versions"]) != set(AWS_RUNTIME_VERSION_FIELDS)
    ):
        raise ValueError("qualification runtime-lock fields do not match schema")
    for field in ("source_commit", "source_tree"):
        if (
            not isinstance(lock[field], str)
            or _SHA1_RE.fullmatch(lock[field]) is None
        ):
            raise ValueError(f"qualification runtime lock {field} is invalid")
    for field in ("control_bundle_sha256", "profile_sha256"):
        _sha256(lock[field], label=f"qualification runtime lock {field}")
    for field, value in lock["versions"].items():
        _fixed_text(value, label=f"qualification runtime version {field}")
    try:
        validate_digest_pinned_oci_image(
            lock["container_image"],
            lock["container_image_digest"],
        )
    except ValueError as error:
        raise ValueError("qualification runtime image is not digest-pinned") from error
    digest = hashlib.sha256(data).hexdigest()
    if (
        digest != binding.runtime_lock_sha256
        or lock["profile_sha256"] != profile.sha256
    ):
        raise ValueError("qualification runtime lock differs from selection binding")
    if profile.profile_id == P6_PROFILE_ID and (
        lock["ami_id"] != P6_BASE_AMI_ID
        or lock["ami_owner_id"] != P6_BASE_AMI_OWNER_ID
    ):
        raise ValueError("P6 qualification requires the reviewed Base DLAMI")
    return lock


def _parse_runtime_sbom(
    profile: AwsGpuProfile,
    lock: Mapping[str, object],
    lock_sha256: str,
    data: bytes,
) -> dict[str, object]:
    sbom = _canonical_object(data, label="qualification runtime SBOM")
    required = {
        "schema_version",
        "document_type",
        "source",
        "runtime_lock_sha256",
        "image_binding_sha256",
        "project_dependency_lock",
        "host",
        "container",
    }
    if (
        set(sbom) != required
        or type(sbom.get("schema_version")) is not int
        or sbom["schema_version"] != 2
        or sbom["document_type"] != "memorysplit-aws-gpu-sbom-v2"
        or sbom["runtime_lock_sha256"] != lock_sha256
    ):
        raise ValueError("qualification runtime SBOM identity is invalid")
    _sha256(sbom["image_binding_sha256"], label="runtime SBOM image binding")
    source = sbom["source"]
    host = sbom["host"]
    container = sbom["container"]
    if (
        not isinstance(source, dict)
        or source
        != {
            "commit": lock["source_commit"],
            "tree": lock["source_tree"],
        }
        or not isinstance(host, dict)
        or not isinstance(container, dict)
    ):
        raise ValueError("qualification runtime SBOM source or facts are invalid")
    if (
        host.get("ami_id") != lock["ami_id"]
        or host.get("ami_owner_id") != lock["ami_owner_id"]
        or host.get("architecture") != profile.architecture
        or container.get("image") != lock["container_image"]
        or container.get("image_digest") != lock["container_image_digest"]
        or container.get("versions")
        != {
            field: lock["versions"][field]
            for field in sorted(_CONTAINER_FACT_FIELDS)
        }
    ):
        raise ValueError("qualification runtime SBOM differs from runtime lock")
    host_versions = host.get("versions")
    if not isinstance(host_versions, dict):
        raise ValueError("qualification runtime SBOM host versions are invalid")
    for field, value in host_versions.items():
        _fixed_text(value, label=f"runtime SBOM host version {field}")
    if profile.profile_id == P6_PROFILE_ID:
        for field, floor in _P6_HOST_FLOORS.items():
            if field in host_versions:
                _at_least(
                    host_versions[field],
                    floor,
                    label=f"P6 runtime SBOM host {field}",
                )
    return sbom


def load_runtime_qualification_bundle(
    *,
    selected_profile: AwsGpuProfile,
    selection_binding: AuthenticatedSelectionBinding,
    runtime_lock_data: bytes,
    runtime_sbom_data: bytes,
) -> RuntimeQualificationBundle:
    """Validate and bind exact runtime-lock/SBOM bytes to the selection."""

    validate_authenticated_selection(selected_profile, selection_binding)
    lock = _parse_runtime_lock(
        selected_profile,
        selection_binding,
        runtime_lock_data,
    )
    lock_sha256 = hashlib.sha256(runtime_lock_data).hexdigest()
    sbom = _parse_runtime_sbom(
        selected_profile,
        lock,
        lock_sha256,
        runtime_sbom_data,
    )
    return RuntimeQualificationBundle(
        lock=lock,
        lock_sha256=lock_sha256,
        sbom=sbom,
        sbom_sha256=hashlib.sha256(runtime_sbom_data).hexdigest(),
    )


def _selection_fields(
    profile: AwsGpuProfile,
    binding: AuthenticatedSelectionBinding,
) -> dict[str, object]:
    return {
        "cohort_id": binding.cohort_id,
        "hardware_amendment_sha256": binding.amendment_sha256,
        "provider_selection_sha256": binding.selection_sha256,
        "provider_selection_version_id": binding.selection_version_id,
        "profile_id": profile.profile_id,
        "provider": profile.provider,
        "profile_sha256": profile.sha256,
        "qualification_evidence_sha256": (
            binding.qualification_evidence_sha256
        ),
        "preselection_environment_receipt_sha256": (
            binding.environment_receipt_sha256
        ),
        "preselection_canary_receipt_sha256": binding.canary_receipt_sha256,
        "approval_receipt_sha256": binding.approval_receipt_sha256,
        "approval_public_key_sha256": binding.approval_public_key_sha256,
        "account_id": binding.account_id,
        "instance_id": binding.instance_id,
        "boot_id": binding.boot_id,
        "region": binding.region,
        "availability_zone": binding.availability_zone,
        "purchase_model": binding.purchase_model,
    }


def build_selected_environment_receipt(
    *,
    selected_profile: AwsGpuProfile,
    selection_binding: AuthenticatedSelectionBinding,
    runtime_lock_data: bytes,
    runtime_sbom_data: bytes,
    attestation_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Build a post-selection environment receipt from measured attestation."""

    bundle = load_runtime_qualification_bundle(
        selected_profile=selected_profile,
        selection_binding=selection_binding,
        runtime_lock_data=runtime_lock_data,
        runtime_sbom_data=runtime_sbom_data,
    )
    evidence = dict(attestation_evidence)
    if set(evidence) != set(AWS_GPU_ATTESTATION_EVIDENCE_V1_FIELDS):
        raise ValueError("selected environment attestation fields are invalid")
    expected = {
        "profile_id": selected_profile.profile_id,
        "provider": selected_profile.provider,
        "profile_sha256": selected_profile.sha256,
        "runtime_lock_sha256": bundle.lock_sha256,
        "control_bundle_sha256": bundle.lock["control_bundle_sha256"],
        "source_commit": bundle.lock["source_commit"],
        "source_tree": bundle.lock["source_tree"],
        "container_image": bundle.lock["container_image"],
        "container_image_digest": bundle.lock["container_image_digest"],
        "ami_id": bundle.lock["ami_id"],
        "ami_owner_id": bundle.lock["ami_owner_id"],
        "instance_type": selected_profile.instance_type,
        "gpu_count": selected_profile.allocated_gpus,
        "account_id": selection_binding.account_id,
        "instance_id": selection_binding.instance_id,
        "region": selection_binding.region,
        "boot_id": selection_binding.boot_id,
    }
    if any(evidence.get(field) != value for field, value in expected.items()):
        raise ValueError("selected environment evidence differs from binding")
    try:
        validate_gpu_product_names(
            selected_profile.profile_id,
            [evidence.get("gpu_model")] * selected_profile.allocated_gpus,
            expected_count=selected_profile.allocated_gpus,
        )
    except ValueError as error:
        raise ValueError("selected environment GPU identity is invalid") from error
    identity = evidence.get("aws_instance_identity_document")
    host_facts = evidence.get("host_facts")
    container_facts = evidence.get("container_facts")
    if (
        not isinstance(identity, dict)
        or identity.get("accountId") != selection_binding.account_id
        or identity.get("instanceId") != selection_binding.instance_id
        or identity.get("imageId") != bundle.lock["ami_id"]
        or identity.get("instanceType") != selected_profile.instance_type
        or not isinstance(host_facts, dict)
        or not isinstance(container_facts, dict)
        or set(container_facts) != _CONTAINER_FACT_FIELDS
        or container_facts
        != {
            field: bundle.lock["versions"][field]
            for field in sorted(_CONTAINER_FACT_FIELDS)
        }
    ):
        raise ValueError("selected environment measured facts are invalid")
    if selected_profile.profile_id == P6_PROFILE_ID:
        for field, floor in _P6_HOST_FLOORS.items():
            if field not in host_facts:
                raise ValueError(f"P6 environment host fact {field} is missing")
            _at_least(
                host_facts[field],
                floor,
                label=f"P6 environment host {field}",
            )
        branches = {
            _version_parts(host_facts[field], label=field)[0]
            for field in ("nvidia_driver", "fabric_manager", "nvlsm")
        }
        if len(branches) != 1:
            raise ValueError("P6 driver, Fabric Manager, and NVLSM differ")
    return {
        "schema_version": 1,
        "receipt_type": "memorysplit-aws-gpu-environment-selection-v1",
        **_selection_fields(selected_profile, selection_binding),
        "runtime_lock_sha256": bundle.lock_sha256,
        "runtime_sbom_sha256": bundle.sbom_sha256,
        "attestation_evidence_sha256": hashlib.sha256(
            canonical_qualification_json(evidence)
        ).hexdigest(),
        "ami_id": bundle.lock["ami_id"],
        "ami_owner_id": bundle.lock["ami_owner_id"],
        "container_image": bundle.lock["container_image"],
        "container_image_digest": bundle.lock["container_image_digest"],
        "host_facts": dict(host_facts),
        "container_facts": dict(container_facts),
    }


def canonical_selected_environment_receipt(
    receipt: Mapping[str, object],
) -> bytes:
    """Serialize only the closed selection-bound environment schema."""

    if set(receipt) != _SELECTED_ENVIRONMENT_FIELDS:
        raise ValueError("selected environment receipt fields are not closed")
    return canonical_qualification_json(dict(receipt))


def parse_selected_environment_receipt_bytes(
    data: bytes,
    *,
    selected_profile: AwsGpuProfile,
    selection_binding: AuthenticatedSelectionBinding,
    runtime_lock_data: bytes,
    runtime_sbom_data: bytes,
) -> dict[str, object]:
    """Revalidate one canonical post-selection environment receipt."""

    bundle = load_runtime_qualification_bundle(
        selected_profile=selected_profile,
        selection_binding=selection_binding,
        runtime_lock_data=runtime_lock_data,
        runtime_sbom_data=runtime_sbom_data,
    )
    value = _canonical_object(data, label="selected environment receipt")
    if set(value) != _SELECTED_ENVIRONMENT_FIELDS:
        raise ValueError("selected environment receipt fields are not closed")
    expected = {
        "schema_version": 1,
        "receipt_type": "memorysplit-aws-gpu-environment-selection-v1",
        **_selection_fields(selected_profile, selection_binding),
        "runtime_lock_sha256": bundle.lock_sha256,
        "runtime_sbom_sha256": bundle.sbom_sha256,
        "ami_id": bundle.lock["ami_id"],
        "ami_owner_id": bundle.lock["ami_owner_id"],
        "container_image": bundle.lock["container_image"],
        "container_image_digest": bundle.lock["container_image_digest"],
    }
    if any(
        type(value.get(field)) is not type(expected_value)
        or value.get(field) != expected_value
        for field, expected_value in expected.items()
    ):
        raise ValueError("selected environment profile/selection binding differs")
    _sha256(
        value["attestation_evidence_sha256"],
        label="selected environment attestation",
    )
    container = value["container_facts"]
    host = value["host_facts"]
    if (
        not isinstance(container, dict)
        or set(container) != _CONTAINER_FACT_FIELDS
        or container
        != {
            field: bundle.lock["versions"][field]
            for field in sorted(_CONTAINER_FACT_FIELDS)
        }
        or not isinstance(host, dict)
    ):
        raise ValueError("selected environment measured facts are invalid")
    for field, fact in host.items():
        _fixed_text(fact, label=f"selected environment host {field}")
    if selected_profile.profile_id == P6_PROFILE_ID:
        for field, floor in _P6_HOST_FLOORS.items():
            if field not in host:
                raise ValueError(f"P6 environment host {field} is missing")
            _at_least(host[field], floor, label=f"P6 environment host {field}")
        branches = {
            _version_parts(host[field], label=field)[0]
            for field in ("nvidia_driver", "fabric_manager", "nvlsm")
        }
        if len(branches) != 1:
            raise ValueError("P6 environment NVLSM branch differs from driver")
    return value


def _validate_hardware_evidence(
    profile: AwsGpuProfile,
    binding: AuthenticatedSelectionBinding,
    lock: Mapping[str, object],
    evidence: Mapping[str, object],
) -> dict[str, object]:
    value = dict(evidence)
    expected_scalar = {
        "account_id": binding.account_id,
        "instance_id": binding.instance_id,
        "boot_id": binding.boot_id,
        "ami_id": lock["ami_id"],
        "instance_type": profile.instance_type,
        "architecture": profile.architecture,
        "vcpus": profile.vcpus,
        "memory_gib": profile.memory_gib,
    }
    if any(
        type(value.get(field)) is not type(expected)
        or value.get(field) != expected
        for field, expected in expected_scalar.items()
    ):
        raise ValueError("hardware evidence differs from selected profile")
    gpu_names = value.get("gpu_names")
    try:
        validate_gpu_product_names(
            profile.profile_id,
            gpu_names,
            expected_count=profile.allocated_gpus,
        )
    except ValueError as error:
        raise ValueError("hardware GPU evidence differs from profile") from error
    devices = value.get("instance_store")
    if (
        not isinstance(devices, list)
        or len(devices) != profile.instance_store_devices
    ):
        raise ValueError("hardware NVMe evidence has the wrong device count")
    paths: list[str] = []
    for device in devices:
        if (
            not isinstance(device, dict)
            or set(device) != {"path", "model", "bytes"}
            or not isinstance(device["path"], str)
            or not device["path"].startswith("/dev/")
            or any(character in device["path"] for character in "\x00\n\r\t ")
            or device["model"] != profile.instance_store_model
            or type(device["bytes"]) is not int
            or device["bytes"] != profile.instance_store_device_bytes
        ):
            raise ValueError("hardware NVMe evidence differs from profile geometry")
        paths.append(device["path"])
    if len(set(paths)) != len(paths):
        raise ValueError("hardware NVMe paths are duplicated")
    return {
        "instance_type": profile.instance_type,
        "architecture": profile.architecture,
        "vcpus": profile.vcpus,
        "memory_gib": profile.memory_gib,
        "gpu_count": profile.allocated_gpus,
        "gpu_names": list(gpu_names),
        "instance_store_devices": profile.instance_store_devices,
        "instance_store_device_bytes": profile.instance_store_device_bytes,
        "instance_store_model": profile.instance_store_model,
        "instance_store_paths": paths,
        "raid_level": profile.raid_level,
    }


def build_selected_bootstrap_receipt(
    *,
    selected_profile: AwsGpuProfile,
    selection_binding: AuthenticatedSelectionBinding,
    runtime_lock_data: bytes,
    runtime_sbom_data: bytes,
    environment_receipt: Mapping[str, object],
    hardware_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Build one selection-bound bootstrap receipt without mutating a host."""

    bundle = load_runtime_qualification_bundle(
        selected_profile=selected_profile,
        selection_binding=selection_binding,
        runtime_lock_data=runtime_lock_data,
        runtime_sbom_data=runtime_sbom_data,
    )
    environment = dict(environment_receipt)
    if (
        environment.get("receipt_type")
        != "memorysplit-aws-gpu-environment-selection-v1"
        or environment.get("provider_selection_sha256")
        != selection_binding.selection_sha256
        or environment.get("provider_selection_version_id")
        != selection_binding.selection_version_id
        or environment.get("profile_sha256") != selected_profile.sha256
        or environment.get("runtime_lock_sha256") != bundle.lock_sha256
        or environment.get("runtime_sbom_sha256") != bundle.sbom_sha256
        or environment.get("instance_id") != selection_binding.instance_id
        or environment.get("boot_id") != selection_binding.boot_id
    ):
        raise ValueError("bootstrap environment receipt differs from selection")
    hardware = _validate_hardware_evidence(
        selected_profile,
        selection_binding,
        bundle.lock,
        hardware_evidence,
    )
    return {
        "schema_version": 1,
        "receipt_type": "memorysplit-aws-gpu-bootstrap-v1",
        **_selection_fields(selected_profile, selection_binding),
        "runtime_lock_sha256": bundle.lock_sha256,
        "runtime_sbom_sha256": bundle.sbom_sha256,
        "environment_receipt_sha256": hashlib.sha256(
            canonical_qualification_json(environment)
        ).hexdigest(),
        "ami_id": bundle.lock["ami_id"],
        "ami_owner_id": bundle.lock["ami_owner_id"],
        "container_image": bundle.lock["container_image"],
        "container_image_digest": bundle.lock["container_image_digest"],
        "hardware": hardware,
    }


def canonical_selected_bootstrap_receipt(
    receipt: Mapping[str, object],
) -> bytes:
    """Serialize only the closed selection-bound bootstrap schema."""

    if set(receipt) != _SELECTED_BOOTSTRAP_FIELDS:
        raise ValueError("selected bootstrap receipt fields are not closed")
    return canonical_qualification_json(dict(receipt))


def _validate_normalized_hardware(
    profile: AwsGpuProfile,
    hardware: object,
) -> None:
    fields = {
        "instance_type",
        "architecture",
        "vcpus",
        "memory_gib",
        "gpu_count",
        "gpu_names",
        "instance_store_devices",
        "instance_store_device_bytes",
        "instance_store_model",
        "instance_store_paths",
        "raid_level",
    }
    if not isinstance(hardware, dict) or set(hardware) != fields:
        raise ValueError("selected bootstrap hardware fields are invalid")
    expected = {
        "instance_type": profile.instance_type,
        "architecture": profile.architecture,
        "vcpus": profile.vcpus,
        "memory_gib": profile.memory_gib,
        "gpu_count": profile.allocated_gpus,
        "instance_store_devices": profile.instance_store_devices,
        "instance_store_device_bytes": profile.instance_store_device_bytes,
        "instance_store_model": profile.instance_store_model,
        "raid_level": profile.raid_level,
    }
    if any(
        type(hardware.get(field)) is not type(expected_value)
        or hardware.get(field) != expected_value
        for field, expected_value in expected.items()
    ):
        raise ValueError("selected bootstrap hardware differs from profile")
    try:
        validate_gpu_product_names(
            profile.profile_id,
            hardware["gpu_names"],
            expected_count=profile.allocated_gpus,
        )
    except ValueError as error:
        raise ValueError("selected bootstrap GPU evidence is invalid") from error
    paths = hardware["instance_store_paths"]
    if (
        not isinstance(paths, list)
        or len(paths) != profile.instance_store_devices
        or len(set(paths)) != len(paths)
        or any(
            not isinstance(path, str)
            or not path.startswith("/dev/")
            or any(character in path for character in "\x00\n\r\t ")
            for path in paths
        )
    ):
        raise ValueError("selected bootstrap NVMe paths are invalid")


def parse_selected_bootstrap_receipt_bytes(
    data: bytes,
    *,
    selected_profile: AwsGpuProfile,
    selection_binding: AuthenticatedSelectionBinding,
    runtime_lock_data: bytes,
    runtime_sbom_data: bytes,
    environment_receipt_sha256: str,
) -> dict[str, object]:
    """Revalidate one canonical selection-bound bootstrap receipt."""

    bundle = load_runtime_qualification_bundle(
        selected_profile=selected_profile,
        selection_binding=selection_binding,
        runtime_lock_data=runtime_lock_data,
        runtime_sbom_data=runtime_sbom_data,
    )
    expected_environment = _sha256(
        environment_receipt_sha256,
        label="selected bootstrap environment receipt",
    )
    value = _canonical_object(data, label="selected bootstrap receipt")
    if set(value) != _SELECTED_BOOTSTRAP_FIELDS:
        raise ValueError("selected bootstrap receipt fields are not closed")
    expected = {
        "schema_version": 1,
        "receipt_type": "memorysplit-aws-gpu-bootstrap-v1",
        **_selection_fields(selected_profile, selection_binding),
        "runtime_lock_sha256": bundle.lock_sha256,
        "runtime_sbom_sha256": bundle.sbom_sha256,
        "environment_receipt_sha256": expected_environment,
        "ami_id": bundle.lock["ami_id"],
        "ami_owner_id": bundle.lock["ami_owner_id"],
        "container_image": bundle.lock["container_image"],
        "container_image_digest": bundle.lock["container_image_digest"],
    }
    if any(
        type(value.get(field)) is not type(expected_value)
        or value.get(field) != expected_value
        for field, expected_value in expected.items()
    ):
        raise ValueError("selected bootstrap profile/selection binding differs")
    _validate_normalized_hardware(selected_profile, value["hardware"])
    return value


def _profile_groups(
    profile: AwsGpuProfile,
) -> tuple[
    dict[str, tuple[int, ...]],
    dict[str, tuple[int, int]],
]:
    if (
        profile.train_groups != (4, 4)
        or sum(profile.train_groups) != profile.allocated_gpus
        or profile.vcpus % sum(profile.train_groups) != 0
    ):
        raise ValueError("selected profile does not define symmetric 4+4 groups")
    gpu_groups: dict[str, tuple[int, ...]] = {}
    cpu_affinities: dict[str, tuple[int, int]] = {}
    gpu_cursor = 0
    cpu_cursor = 0
    cpus_per_gpu = profile.vcpus // profile.allocated_gpus
    for arm, width in zip(("dense", "split90"), profile.train_groups, strict=True):
        gpu_groups[arm] = tuple(range(gpu_cursor, gpu_cursor + width))
        cpu_width = width * cpus_per_gpu
        cpu_affinities[arm] = (cpu_cursor, cpu_cursor + cpu_width - 1)
        gpu_cursor += width
        cpu_cursor += cpu_width
    if (
        set(gpu_groups["dense"]) & set(gpu_groups["split90"])
        or set(gpu_groups["dense"]) | set(gpu_groups["split90"])
        != set(range(profile.allocated_gpus))
        or cpu_cursor != profile.vcpus
    ):
        raise ValueError("selected profile GPU/CPU group geometry is invalid")
    return gpu_groups, cpu_affinities


def _private_scratch(root: Path, output: Path) -> None:
    try:
        details = root.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError("selected canary scratch root is unavailable") from error
    if (
        root.is_symlink()
        or not stat.S_ISDIR(details.st_mode)
        or details.st_uid != os.geteuid()
        or stat.S_IMODE(details.st_mode) & 0o077
    ):
        raise ValueError("selected canary scratch root must be private and owned")
    try:
        if output.parent.resolve(strict=True) != root.resolve(strict=True):
            raise ValueError("selected canary output must be directly in scratch")
    except OSError as error:
        raise ValueError("selected canary output parent is unavailable") from error
    if output.is_symlink():
        raise ValueError("selected canary output must not be a symlink")


def _s3_root(value: object) -> str:
    parsed = urlsplit(value) if isinstance(value, str) else None
    if (
        parsed is None
        or parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
        or "\\" in value
    ):
        raise ValueError("selected canary S3 root is invalid")
    return value.rstrip("/")


def build_selected_canary_plan(
    *,
    selected_profile: AwsGpuProfile,
    selection_binding: AuthenticatedSelectionBinding,
    runtime_lock_data: bytes,
    runtime_sbom_data: bytes,
    environment_receipt: Mapping[str, object],
    bootstrap_receipt: Mapping[str, object],
    scratch_root: Path | str,
    output_path: Path | str,
    s3_root: str,
    target_tokens_per_arm: int = SELECTED_TARGET_TOKENS_PER_ARM,
    cohort_pairs: int = SELECTED_COHORT_PAIRS,
) -> SelectedCanaryPlan:
    """Build a side-effect-free selected-profile canary plan."""

    bundle = load_runtime_qualification_bundle(
        selected_profile=selected_profile,
        selection_binding=selection_binding,
        runtime_lock_data=runtime_lock_data,
        runtime_sbom_data=runtime_sbom_data,
    )
    environment_data = canonical_selected_environment_receipt(
        environment_receipt
    )
    environment = parse_selected_environment_receipt_bytes(
        environment_data,
        selected_profile=selected_profile,
        selection_binding=selection_binding,
        runtime_lock_data=runtime_lock_data,
        runtime_sbom_data=runtime_sbom_data,
    )
    environment_sha256 = hashlib.sha256(environment_data).hexdigest()
    bootstrap_data = canonical_selected_bootstrap_receipt(bootstrap_receipt)
    bootstrap = parse_selected_bootstrap_receipt_bytes(
        bootstrap_data,
        selected_profile=selected_profile,
        selection_binding=selection_binding,
        runtime_lock_data=runtime_lock_data,
        runtime_sbom_data=runtime_sbom_data,
        environment_receipt_sha256=environment_sha256,
    )
    bootstrap_sha256 = hashlib.sha256(bootstrap_data).hexdigest()
    expected_authority = {
        "provider_selection_sha256": selection_binding.selection_sha256,
        "provider_selection_version_id": selection_binding.selection_version_id,
        "profile_sha256": selected_profile.sha256,
        "runtime_lock_sha256": bundle.lock_sha256,
        "runtime_sbom_sha256": bundle.sbom_sha256,
        "instance_id": selection_binding.instance_id,
        "boot_id": selection_binding.boot_id,
    }
    if (
        environment.get("receipt_type")
        != "memorysplit-aws-gpu-environment-selection-v1"
        or bootstrap.get("receipt_type") != "memorysplit-aws-gpu-bootstrap-v1"
        or any(
            environment.get(field) != expected
            or bootstrap.get(field) != expected
            for field, expected in expected_authority.items()
        )
        or bootstrap.get("environment_receipt_sha256") != environment_sha256
    ):
        raise ValueError(
            "selected canary environment/bootstrap selection binding differs"
        )
    if (
        type(target_tokens_per_arm) is not int
        or target_tokens_per_arm != SELECTED_TARGET_TOKENS_PER_ARM
        or type(cohort_pairs) is not int
        or cohort_pairs != SELECTED_COHORT_PAIRS
    ):
        raise ValueError("selected canary ETA geometry differs from frozen cohort")
    scratch = Path(os.path.abspath(os.fspath(scratch_root)))
    output = Path(os.path.abspath(os.fspath(output_path)))
    _private_scratch(scratch, output)
    gpu_groups, cpu_affinities = _profile_groups(selected_profile)
    port_base = 29_700 + selection_binding.seed * 2
    return SelectedCanaryPlan(
        profile=selected_profile,
        binding=selection_binding,
        bundle=bundle,
        environment_receipt=environment,
        environment_receipt_sha256=environment_sha256,
        bootstrap_receipt=bootstrap,
        bootstrap_receipt_sha256=bootstrap_sha256,
        scratch_root=scratch,
        output_path=output,
        s3_root=_s3_root(s3_root),
        gpu_groups=gpu_groups,
        cpu_affinities=cpu_affinities,
        master_ports={"dense": port_base, "split90": port_base + 1},
        target_tokens_per_arm=target_tokens_per_arm,
        cohort_pairs=cohort_pairs,
    )


_MINIMAL_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "PYTHONNOUSERSITE": "1",
}
_HOST_QUALIFICATION_WORKER = str(
    Path(__file__).with_name("qualification_worker.py").resolve()
)
_CONTAINER_QUALIFICATION_WORKER = (
    "/opt/memorysplit/cluster/aws/qualification_worker.py"
)


def _docker_prefix(
    plan: SelectedCanaryPlan,
    *,
    name: str,
    gpu_ids: tuple[int, ...],
    cpu_affinity: tuple[int, int],
) -> tuple[str, ...]:
    return (
        "/usr/bin/docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--pull",
        "never",
        "--read-only",
        "--ipc=host",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--pids-limit",
        "4096",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=8g",
        "--cpuset-cpus",
        f"{cpu_affinity[0]}-{cpu_affinity[1]}",
        "--gpus",
        "device=" + ",".join(str(gpu) for gpu in gpu_ids),
        plan.bundle.lock["container_image"],
    )


def _selected_specs(
    plan: SelectedCanaryPlan,
) -> tuple[
    QualificationCommandSpec,
    QualificationCommandSpec,
    tuple[QualificationCommandSpec, QualificationCommandSpec],
]:
    full_affinity = (0, plan.profile.vcpus - 1)
    all_gpus = tuple(range(plan.profile.allocated_gpus))
    hardware = QualificationCommandSpec(
        name="hardware-inventory",
        argv=(
            "/usr/bin/python3",
            "-I",
            "-P",
            _HOST_QUALIFICATION_WORKER,
            "--hardware",
            "--profile-id",
            plan.profile.profile_id,
            "--region",
            plan.binding.region,
        ),
        environment=dict(_MINIMAL_ENVIRONMENT),
        cwd=Path("/usr"),
        timeout_seconds=120.0,
    )
    container = QualificationCommandSpec(
        name="container-capabilities",
        argv=(
            *_docker_prefix(
                plan,
                name="memorysplit-selected-container-capabilities",
                gpu_ids=all_gpus,
                cpu_affinity=full_affinity,
            ),
            "/opt/conda/bin/python",
            "-I",
            "-P",
            _CONTAINER_QUALIFICATION_WORKER,
            "--container-capabilities",
        ),
        environment=dict(_MINIMAL_ENVIRONMENT),
        cwd=plan.scratch_root,
        timeout_seconds=1_800.0,
        gpu_ids=all_gpus,
        cpu_affinity=full_affinity,
    )
    groups: list[QualificationCommandSpec] = []
    for arm in ("dense", "split90"):
        gpu_ids = plan.gpu_groups[arm]
        affinity = plan.cpu_affinities[arm]
        port = plan.master_ports[arm]
        groups.append(
            QualificationCommandSpec(
                name=f"{arm}-4rank",
                argv=(
                    *_docker_prefix(
                        plan,
                        name=f"memorysplit-selected-{arm}-4rank",
                        gpu_ids=gpu_ids,
                        cpu_affinity=affinity,
                    ),
                    "/opt/conda/bin/python",
                    "-m",
                    "torch.distributed.run",
                    "--nnodes=1",
                    "--nproc_per_node=4",
                    "--rdzv_backend=c10d",
                    f"--rdzv_endpoint=127.0.0.1:{port}",
                    _CONTAINER_QUALIFICATION_WORKER,
                    "--nccl-train",
                    "--arm",
                    arm,
                    "--gpu-ids",
                    ",".join(str(gpu) for gpu in gpu_ids),
                    "--cpu-affinity",
                    f"{affinity[0]}-{affinity[1]}",
                    "--master-port",
                    str(port),
                    "--updates",
                    str(SELECTED_THROUGHPUT_UPDATES),
                    "--warmup-updates",
                    str(SELECTED_THROUGHPUT_WARMUP_UPDATES),
                ),
                environment=dict(_MINIMAL_ENVIRONMENT),
                cwd=plan.scratch_root,
                timeout_seconds=7_200.0,
                arm=arm,
                gpu_ids=gpu_ids,
                cpu_affinity=affinity,
                world_size=4,
                master_port=port,
                concurrent_group="paired-4rank",
            )
        )
    return hardware, container, (groups[0], groups[1])


def _render_spec(spec: QualificationCommandSpec) -> dict[str, object]:
    value: dict[str, object] = {
        "name": spec.name,
        "argv": list(spec.argv),
        "environment": dict(sorted(spec.environment.items())),
        "cwd": str(spec.cwd),
        "timeout_seconds": spec.timeout_seconds,
    }
    for field, item in (
        ("arm", spec.arm),
        ("world_size", spec.world_size),
        ("master_port", spec.master_port),
        ("concurrent_group", spec.concurrent_group),
    ):
        if item is not None:
            value[field] = item
    if spec.gpu_ids:
        value["gpu_ids"] = list(spec.gpu_ids)
    if spec.cpu_affinity is not None:
        value["cpu_affinity"] = list(spec.cpu_affinity)
    return value


def render_selected_canary_plan(plan: SelectedCanaryPlan) -> dict[str, object]:
    """Render exact commands and immutable authority without executing them."""

    validate_authenticated_selection(plan.profile, plan.binding)
    hardware, container, groups = _selected_specs(plan)
    return {
        "schema_version": 1,
        "receipt_type": "memorysplit-aws-gpu-qualification-plan-v1",
        "profile_id": plan.profile.profile_id,
        "provider": plan.profile.provider,
        "bindings": {
            **_selection_fields(plan.profile, plan.binding),
            "runtime_lock_sha256": plan.bundle.lock_sha256,
            "runtime_sbom_sha256": plan.bundle.sbom_sha256,
            "environment_receipt_sha256": plan.environment_receipt_sha256,
            "bootstrap_receipt_sha256": plan.bootstrap_receipt_sha256,
        },
        "commands": {
            "hardware": [_render_spec(hardware)],
            "container": [_render_spec(container)],
            "nccl_4x4": [_render_spec(spec) for spec in groups],
        },
        "phase_order": list(SELECTED_PHASE_ORDER),
        "output": str(plan.output_path),
    }


def _checked(
    runner: QualificationCommandRunner,
    spec: QualificationCommandSpec,
) -> QualificationCommandResult:
    try:
        result = runner.run(spec)
    except Exception as error:
        raise ValueError(f"qualification {spec.name} command failed") from error
    if (
        not isinstance(result, QualificationCommandResult)
        or type(result.returncode) is not int
        or result.returncode != 0
        or not isinstance(result.stdout, str)
        or not isinstance(result.stderr, str)
        or result.stderr
    ):
        raise ValueError(f"qualification {spec.name} command was not clean")
    return result


def _stdout_object(result: QualificationCommandResult, *, label: str) -> dict[str, object]:
    try:
        data = result.stdout.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} output is not ASCII") from error
    return _canonical_object(data, label=f"{label} output")


def _validate_selected_hardware(
    plan: SelectedCanaryPlan,
    value: Mapping[str, object],
) -> dict[str, object]:
    required = {
        "account_id",
        "instance_id",
        "boot_id",
        "ami_id",
        "instance_type",
        "architecture",
        "vcpus",
        "memory_gib",
        "gpu_names",
        "instance_store",
        "gpu_devices",
        "host_facts",
        "topology",
    }
    if set(value) != required:
        raise ValueError("qualification hardware fields do not match schema")
    _validate_hardware_evidence(
        plan.profile,
        plan.binding,
        plan.bundle.lock,
        value,
    )
    devices = value["gpu_devices"]
    if not isinstance(devices, list) or len(devices) != plan.profile.allocated_gpus:
        raise ValueError("qualification hardware GPU device count is invalid")
    names: list[str] = []
    for index, device in enumerate(devices):
        if (
            not isinstance(device, dict)
            or set(device) != {"index", "name", "memory_mib"}
            or type(device["index"]) is not int
            or device["index"] != index
            or type(device["memory_mib"]) is not int
            or device["memory_mib"] <= 0
        ):
            raise ValueError("qualification hardware GPU evidence is invalid")
        names.append(device["name"])
    try:
        validate_gpu_product_names(
            plan.profile.profile_id,
            names,
            expected_count=plan.profile.allocated_gpus,
        )
    except ValueError as error:
        raise ValueError("qualification hardware GPU names differ") from error
    if plan.profile.profile_id == P6_PROFILE_ID and any(
        not 290_000 <= device["memory_mib"] <= 300_000
        for device in devices
    ):
        raise ValueError("P6 qualification B300 memory evidence is invalid")
    host_facts = value["host_facts"]
    if not isinstance(host_facts, dict):
        raise ValueError("qualification hardware host facts are invalid")
    if plan.profile.profile_id == P6_PROFILE_ID:
        for field, floor in _P6_HOST_FLOORS.items():
            if field not in host_facts:
                raise ValueError(f"P6 qualification host {field} is missing")
            _at_least(
                host_facts[field],
                floor,
                label=f"P6 qualification host {field}",
            )
        branches = {
            _version_parts(host_facts[field], label=field)[0]
            for field in ("nvidia_driver", "fabric_manager", "nvlsm")
        }
        if len(branches) != 1:
            raise ValueError("P6 qualification NVLSM branch differs from driver")
    topology = value["topology"]
    if (
        not isinstance(topology, dict)
        or set(topology)
        != {"matrix_sha256", "nvlink_active", "fully_connected"}
        or _SHA256_RE.fullmatch(str(topology["matrix_sha256"])) is None
        or topology["nvlink_active"] is not True
        or topology["fully_connected"] is not True
    ):
        raise ValueError("qualification topology/NVLink evidence is invalid")
    return dict(value)


def _validate_container_capabilities(
    plan: SelectedCanaryPlan,
    value: Mapping[str, object],
) -> dict[str, object]:
    fields = {
        "versions",
        "bf16",
        "sdpa_forward",
        "sdpa_backward",
        "torch_compile",
        "fused_adamw",
        "one_step_train",
        "checkpoint_resume_exact",
        "checkpoint_sha256",
        "resumed_checkpoint_sha256",
    }
    versions = value.get("versions")
    if (
        set(value) != fields
        or not isinstance(versions, dict)
        or versions
        != {
            field: plan.bundle.lock["versions"][field]
            for field in sorted(_CONTAINER_FACT_FIELDS)
        }
        or any(
            value[field] is not True
            for field in (
                "bf16",
                "sdpa_forward",
                "sdpa_backward",
                "torch_compile",
                "fused_adamw",
                "one_step_train",
                "checkpoint_resume_exact",
            )
        )
        or _SHA256_RE.fullmatch(str(value["checkpoint_sha256"])) is None
        or value["resumed_checkpoint_sha256"] != value["checkpoint_sha256"]
    ):
        raise ValueError(
            "qualification container version or capability evidence is invalid"
        )
    return dict(value)


def _validate_group_result(
    plan: SelectedCanaryPlan,
    value: Mapping[str, object],
    *,
    arm: str,
) -> dict[str, object]:
    fields = {
        "arm",
        "world_size",
        "gpu_ids",
        "cpu_affinity",
        "master_port",
        "all_reduce_sum",
        "all_reduce_latency_seconds",
        "updates",
        "warmup_updates",
        "median_tok_s",
        "peak_memory_bytes",
        "one_step_train",
        "checkpoint_resume_exact",
        "checkpoint_sha256",
        "resumed_checkpoint_sha256",
    }
    numeric = (
        value.get("all_reduce_latency_seconds"),
        value.get("median_tok_s"),
    )
    if (
        set(value) != fields
        or value.get("arm") != arm
        or type(value.get("world_size")) is not int
        or value["world_size"] != 4
        or value.get("gpu_ids") != list(plan.gpu_groups[arm])
        or value.get("cpu_affinity") != list(plan.cpu_affinities[arm])
        or value.get("master_port") != plan.master_ports[arm]
        or type(value.get("all_reduce_sum")) not in (int, float)
        or float(value["all_reduce_sum"]) != 10.0
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) <= 0
            for item in numeric
        )
        or type(value.get("updates")) is not int
        or value["updates"] != SELECTED_THROUGHPUT_UPDATES
        or type(value.get("warmup_updates")) is not int
        or value["warmup_updates"] != SELECTED_THROUGHPUT_WARMUP_UPDATES
        or type(value.get("peak_memory_bytes")) is not int
        or value["peak_memory_bytes"] <= 0
        or value.get("one_step_train") is not True
        or value.get("checkpoint_resume_exact") is not True
        or _SHA256_RE.fullmatch(str(value.get("checkpoint_sha256"))) is None
        or value.get("resumed_checkpoint_sha256")
        != value.get("checkpoint_sha256")
    ):
        raise ValueError(
            f"qualification {arm} independent 4-rank NCCL/checkpoint group "
            "evidence is invalid"
        )
    return dict(value)


def _roundtrip_payload(plan: SelectedCanaryPlan) -> bytes:
    return canonical_qualification_json(
        {
            "boot_id": plan.binding.boot_id,
            "instance_id": plan.binding.instance_id,
            "profile_sha256": plan.profile.sha256,
            "provider_selection_sha256": plan.binding.selection_sha256,
            "provider_selection_version_id": plan.binding.selection_version_id,
            "receipt_type": "memorysplit-aws-gpu-qualification-roundtrip-v1",
            "runtime_lock_sha256": plan.bundle.lock_sha256,
            "runtime_sbom_sha256": plan.bundle.sbom_sha256,
        }
    )


def _s3_roundtrip(
    plan: SelectedCanaryPlan,
    store: QualificationObjectStore,
) -> dict[str, object]:
    payload = _roundtrip_payload(plan)
    digest = hashlib.sha256(payload).hexdigest()
    uri = (
        f"{plan.s3_root}/qualification-roundtrip/"
        f"{plan.binding.selection_sha256}/{digest}.json"
    )
    try:
        written = store.put(uri, payload, sha256=digest)
    except Exception as error:
        raise ValueError("qualification S3 upload failed") from error
    if (
        not isinstance(written, QualificationObjectWrite)
        or written.sha256 != digest
        or type(written.bytes) is not int
        or written.bytes != len(payload)
        or not isinstance(written.version_id, str)
        or written.version_id in {"", "null"}
    ):
        raise ValueError("qualification S3 upload evidence is invalid")
    try:
        read = store.get(uri, version_id=written.version_id)
    except Exception as error:
        raise ValueError("qualification S3 download failed") from error
    if (
        not isinstance(read, QualificationObjectRead)
        or read.payload != payload
        or read.sha256 != digest
        or read.version_id != written.version_id
    ):
        raise ValueError("qualification S3 round-trip bytes differ")
    return {
        "uri": uri,
        "sha256": digest,
        "bytes": len(payload),
        "version_id": written.version_id,
    }


def _telemetry(
    plan: SelectedCanaryPlan,
    groups: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    per_arm = {
        arm: {
            "median_tok_s": float(groups[arm]["median_tok_s"]),
            "peak_memory_bytes": groups[arm]["peak_memory_bytes"],
        }
        for arm in ("dense", "split90")
    }
    pair_eta = max(
        plan.target_tokens_per_arm / per_arm[arm]["median_tok_s"]
        for arm in ("dense", "split90")
    )
    telemetry = {
        "updates": SELECTED_THROUGHPUT_UPDATES,
        "warmup_updates": SELECTED_THROUGHPUT_WARMUP_UPDATES,
        "world_size_per_arm": 4,
        "simultaneous_groups": True,
        "per_arm": per_arm,
        "aggregate_median_tok_s": sum(
            per_arm[arm]["median_tok_s"] for arm in ("dense", "split90")
        ),
        "target_tokens_per_arm": plan.target_tokens_per_arm,
        "projected_pair_eta_seconds": float(pair_eta),
        "cohort_pairs": plan.cohort_pairs,
        "projected_cohort_eta_seconds": float(pair_eta * plan.cohort_pairs),
    }
    if any(
        not math.isfinite(value) or value <= 0
        for value in (
            telemetry["aggregate_median_tok_s"],
            telemetry["projected_pair_eta_seconds"],
            telemetry["projected_cohort_eta_seconds"],
        )
    ):
        raise ValueError("qualification throughput/ETA telemetry is invalid")
    return telemetry


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("qualification time reader returned a naive timestamp")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _phase_record(
    *,
    name: str,
    seconds: float,
    evidence: object,
) -> dict[str, object]:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("qualification phase timer is invalid")
    return {
        "name": name,
        "passed": True,
        "seconds": float(seconds),
        "evidence_sha256": hashlib.sha256(
            canonical_qualification_json(evidence)
        ).hexdigest(),
    }


def canonical_selected_qualification_receipt(
    receipt: Mapping[str, object],
) -> bytes:
    if set(receipt) != _SELECTED_RECEIPT_FIELDS:
        raise ValueError("selected qualification receipt fields do not match")
    return canonical_qualification_json(dict(receipt))


def _atomic_no_replace(path: Path, data: bytes) -> None:
    temporary = path.parent / (
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    descriptor = -1
    installed = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise ValueError("qualification receipt write was short")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        installed = True
        os.unlink(temporary)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise ValueError("selected qualification receipt already exists") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if installed:
            details = path.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(details.st_mode)
                or stat.S_IMODE(details.st_mode) != 0o600
                or path.read_bytes() != data
            ):
                raise ValueError("selected qualification receipt is unsafe")


def execute_selected_canary(
    plan: SelectedCanaryPlan,
    *,
    command_runner: QualificationCommandRunner,
    object_store: QualificationObjectStore,
    time_reader: QualificationTimeReader,
    apply: bool,
) -> dict[str, object]:
    """Run selected-profile qualification through injected side-effect seams."""

    if type(apply) is not bool:
        raise TypeError("selected qualification apply flag must be boolean")
    validate_authenticated_selection(plan.profile, plan.binding)
    _private_scratch(plan.scratch_root, plan.output_path)
    if plan.output_path.exists() or plan.output_path.is_symlink():
        raise ValueError("selected qualification receipt already exists")
    hardware_spec, container_spec, group_specs = _selected_specs(plan)
    started_at = _timestamp(time_reader.now())
    started = time_reader.monotonic()
    phase_rows: list[dict[str, object]] = []

    phase_started = time_reader.monotonic()
    hardware = _validate_selected_hardware(
        plan,
        _stdout_object(
            _checked(command_runner, hardware_spec),
            label="qualification hardware",
        ),
    )
    phase_rows.append(
        _phase_record(
            name="hardware",
            seconds=time_reader.monotonic() - phase_started,
            evidence=hardware,
        )
    )

    phase_started = time_reader.monotonic()
    container = _validate_container_capabilities(
        plan,
        _stdout_object(
            _checked(command_runner, container_spec),
            label="qualification container",
        ),
    )
    phase_rows.append(
        _phase_record(
            name="container",
            seconds=time_reader.monotonic() - phase_started,
            evidence=container,
        )
    )

    phase_started = time_reader.monotonic()
    try:
        pair_results = command_runner.run_pair(group_specs)
    except Exception as error:
        raise ValueError("qualification independent 4-rank pair failed") from error
    if set(pair_results) != {spec.name for spec in group_specs}:
        raise ValueError("qualification independent 4-rank pair is incomplete")
    groups: dict[str, dict[str, object]] = {}
    for spec in group_specs:
        result = pair_results[spec.name]
        if (
            not isinstance(result, QualificationCommandResult)
            or result.returncode != 0
            or result.stderr
        ):
            raise ValueError("qualification independent NCCL group failed")
        groups[str(spec.arm)] = _validate_group_result(
            plan,
            _stdout_object(result, label=f"qualification {spec.arm} group"),
            arm=str(spec.arm),
        )
    group_seconds = time_reader.monotonic() - phase_started
    nccl = [
        {
            "arm": arm,
            "world_size": groups[arm]["world_size"],
            "gpu_ids": groups[arm]["gpu_ids"],
            "cpu_affinity": groups[arm]["cpu_affinity"],
            "master_port": groups[arm]["master_port"],
            "all_reduce_sum": groups[arm]["all_reduce_sum"],
            "all_reduce_latency_seconds": groups[arm][
                "all_reduce_latency_seconds"
            ],
        }
        for arm in ("dense", "split90")
    ]
    training = {
        arm: {
            field: groups[arm][field]
            for field in (
                "updates",
                "warmup_updates",
                "one_step_train",
                "checkpoint_resume_exact",
                "checkpoint_sha256",
                "resumed_checkpoint_sha256",
            )
        }
        for arm in ("dense", "split90")
    }
    phase_rows.append(
        _phase_record(
            name="nccl_4x4",
            seconds=group_seconds,
            evidence=nccl,
        )
    )
    phase_rows.append(
        _phase_record(
            name="training",
            seconds=group_seconds,
            evidence=training,
        )
    )

    phase_started = time_reader.monotonic()
    s3_roundtrip = _s3_roundtrip(plan, object_store)
    phase_rows.append(
        _phase_record(
            name="s3_roundtrip",
            seconds=time_reader.monotonic() - phase_started,
            evidence=s3_roundtrip,
        )
    )

    phase_started = time_reader.monotonic()
    telemetry = _telemetry(plan, groups)
    phase_rows.append(
        _phase_record(
            name="telemetry_eta",
            seconds=time_reader.monotonic() - phase_started,
            evidence=telemetry,
        )
    )
    if tuple(row["name"] for row in phase_rows) != SELECTED_PHASE_ORDER:
        raise ValueError("selected qualification phase order drifted")
    total_seconds = time_reader.monotonic() - started
    if not math.isfinite(total_seconds) or total_seconds < 0:
        raise ValueError("selected qualification total timer is invalid")
    ended_at = _timestamp(time_reader.now())
    receipt = {
        "schema_version": 2,
        "receipt_type": "memorysplit-aws-gpu-qualification-v2",
        **_selection_fields(plan.profile, plan.binding),
        "runtime_lock_sha256": plan.bundle.lock_sha256,
        "runtime_sbom_sha256": plan.bundle.sbom_sha256,
        "environment_receipt_sha256": plan.environment_receipt_sha256,
        "bootstrap_receipt_sha256": plan.bootstrap_receipt_sha256,
        "ami_id": plan.bundle.lock["ami_id"],
        "ami_owner_id": plan.bundle.lock["ami_owner_id"],
        "container_image": plan.bundle.lock["container_image"],
        "container_image_digest": plan.bundle.lock["container_image_digest"],
        "phases": phase_rows,
        "hardware": hardware,
        "container": container,
        "nccl_groups": nccl,
        "training": training,
        "s3_roundtrip": s3_roundtrip,
        "telemetry": telemetry,
        "passed": True,
        "started_at": started_at,
        "ended_at": ended_at,
        "total_seconds": float(total_seconds),
    }
    payload = canonical_selected_qualification_receipt(receipt)
    parse_selected_qualification_receipt_bytes(payload, plan=plan)
    if apply:
        _atomic_no_replace(plan.output_path, payload)
    return receipt


def _validate_phase_bindings(
    phases: object,
    *,
    evidence_by_phase: Mapping[str, object],
) -> None:
    if (
        not isinstance(phases, list)
        or len(phases) != len(SELECTED_PHASE_ORDER)
        or tuple(
            row.get("name") if isinstance(row, dict) else None
            for row in phases
        )
        != SELECTED_PHASE_ORDER
    ):
        raise ValueError("selected qualification phase order is invalid")
    for row in phases:
        expected_sha256 = hashlib.sha256(
            canonical_qualification_json(evidence_by_phase[row["name"]])
        ).hexdigest()
        if (
            set(row) != {"name", "passed", "seconds", "evidence_sha256"}
            or row["passed"] is not True
            or isinstance(row["seconds"], bool)
            or not isinstance(row["seconds"], (int, float))
            or not math.isfinite(float(row["seconds"]))
            or row["seconds"] < 0
            or row["evidence_sha256"] != expected_sha256
        ):
            raise ValueError("selected qualification phase binding is invalid")


def parse_selected_qualification_receipt_bytes(
    data: bytes,
    *,
    plan: SelectedCanaryPlan,
) -> dict[str, object]:
    """Parse and independently revalidate one selected qualification receipt."""

    validate_authenticated_selection(plan.profile, plan.binding)
    value = _canonical_object(data, label="selected qualification receipt")
    if set(value) != _SELECTED_RECEIPT_FIELDS:
        raise ValueError("selected qualification receipt fields do not match")
    expected = {
        "schema_version": 2,
        "receipt_type": "memorysplit-aws-gpu-qualification-v2",
        **_selection_fields(plan.profile, plan.binding),
        "runtime_lock_sha256": plan.bundle.lock_sha256,
        "runtime_sbom_sha256": plan.bundle.sbom_sha256,
        "environment_receipt_sha256": plan.environment_receipt_sha256,
        "bootstrap_receipt_sha256": plan.bootstrap_receipt_sha256,
        "ami_id": plan.bundle.lock["ami_id"],
        "ami_owner_id": plan.bundle.lock["ami_owner_id"],
        "container_image": plan.bundle.lock["container_image"],
        "container_image_digest": plan.bundle.lock["container_image_digest"],
        "passed": True,
    }
    if any(
        type(value.get(field)) is not type(expected_value)
        or value.get(field) != expected_value
        for field, expected_value in expected.items()
    ):
        raise ValueError("selected qualification profile/selection binding differs")
    hardware = _validate_selected_hardware(plan, value["hardware"])
    container = _validate_container_capabilities(plan, value["container"])
    rows = value["nccl_groups"]
    training = value["training"]
    if (
        not isinstance(rows, list)
        or len(rows) != 2
        or not isinstance(training, dict)
        or set(training) != {"dense", "split90"}
    ):
        raise ValueError("selected qualification NCCL/training pair is invalid")
    groups: dict[str, dict[str, object]] = {}
    for arm, nccl_row in zip(("dense", "split90"), rows, strict=True):
        if not isinstance(nccl_row, dict) or not isinstance(training[arm], dict):
            raise ValueError("selected qualification group row is invalid")
        merged = {
            **nccl_row,
            **training[arm],
            "median_tok_s": value["telemetry"]["per_arm"][arm]["median_tok_s"],
            "peak_memory_bytes": value["telemetry"]["per_arm"][arm][
                "peak_memory_bytes"
            ],
        }
        groups[arm] = _validate_group_result(plan, merged, arm=arm)
    expected_s3 = _s3_roundtrip_shape(plan, value["s3_roundtrip"])
    expected_telemetry = _telemetry(plan, groups)
    if value["telemetry"] != expected_telemetry:
        raise ValueError("selected qualification telemetry/ETA binding differs")
    _validate_phase_bindings(
        value["phases"],
        evidence_by_phase={
            "hardware": hardware,
            "container": container,
            "nccl_4x4": rows,
            "training": training,
            "s3_roundtrip": expected_s3,
            "telemetry_eta": expected_telemetry,
        },
    )
    for field in ("started_at", "ended_at"):
        timestamp = value[field]
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            raise ValueError("selected qualification timestamp is invalid")
        try:
            parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
        except ValueError as error:
            raise ValueError("selected qualification timestamp is invalid") from error
        if parsed.utcoffset() != UTC.utcoffset(parsed):
            raise ValueError("selected qualification timestamp is not UTC")
    total = value["total_seconds"]
    if (
        isinstance(total, bool)
        or not isinstance(total, (int, float))
        or not math.isfinite(float(total))
        or total < 0
    ):
        raise ValueError("selected qualification total duration is invalid")
    return value


def _s3_roundtrip_shape(
    plan: SelectedCanaryPlan,
    value: object,
) -> dict[str, object]:
    payload = _roundtrip_payload(plan)
    digest = hashlib.sha256(payload).hexdigest()
    expected_uri = (
        f"{plan.s3_root}/qualification-roundtrip/"
        f"{plan.binding.selection_sha256}/{digest}.json"
    )
    if (
        not isinstance(value, dict)
        or set(value) != {"uri", "sha256", "bytes", "version_id"}
        or value["uri"] != expected_uri
        or value["sha256"] != digest
        or type(value["bytes"]) is not int
        or value["bytes"] != len(payload)
        or not isinstance(value["version_id"], str)
        or value["version_id"] in {"", "null"}
    ):
        raise ValueError("selected qualification S3 round-trip binding is invalid")
    return dict(value)


__all__ = [
    "P6_BASE_AMI_ID",
    "P6_BASE_AMI_OWNER_ID",
    "P6_PROFILE_ID",
    "RuntimeQualificationBundle",
    "SELECTED_COHORT_PAIRS",
    "SELECTED_PHASE_ORDER",
    "SELECTED_TARGET_TOKENS_PER_ARM",
    "SELECTED_THROUGHPUT_UPDATES",
    "SELECTED_THROUGHPUT_WARMUP_UPDATES",
    "SelectedCanaryPlan",
    "QualificationCommandResult",
    "QualificationCommandRunner",
    "QualificationCommandSpec",
    "QualificationObjectRead",
    "QualificationObjectStore",
    "QualificationObjectWrite",
    "QualificationTimeReader",
    "build_selected_canary_plan",
    "build_selected_bootstrap_receipt",
    "build_selected_environment_receipt",
    "canonical_selected_bootstrap_receipt",
    "canonical_selected_environment_receipt",
    "canonical_selected_qualification_receipt",
    "canonical_qualification_json",
    "execute_selected_canary",
    "load_runtime_qualification_bundle",
    "parse_selected_bootstrap_receipt_bytes",
    "parse_selected_environment_receipt_bytes",
    "parse_selected_qualification_receipt_bytes",
    "render_selected_canary_plan",
    "validate_authenticated_selection",
]
