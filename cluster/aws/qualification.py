"""Profile-driven AWS GPU qualification contracts.

This module is intentionally side-effect free.  Legacy P5 producers keep their
existing schemas and entry points; selected-profile producers call these
helpers after an authenticated provider selection has been admitted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import signal
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Protocol
from urllib.parse import urlsplit

from cluster.aws.gpu_profile import (
    AwsGpuProfile,
    load_aws_gpu_profile,
    read_secure_regular_file,
)
from msctl.aws_contracts import (
    AWS_GPU_ATTESTATION_EVIDENCE_V1_FIELDS,
    AWS_RUNTIME_LOCK_FIELDS,
    AWS_RUNTIME_VERSION_FIELDS,
    validate_digest_pinned_oci_image,
    validate_gpu_product_names,
)
from msctl.aws_hardware import (
    P5_PROFILE_PATH,
    P6_PROFILE_PATH,
    AuthenticatedSelectionBinding,
    AwsCliVersionedSelectionStore,
    OpenSslQualificationApprovalVerifier,
    QualificationApprovalVerifier,
    VersionedProviderSelectionStore,
    admit_provider_selection,
    verify_aws_instance_identity_pkcs7,
)


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
        "seed",
        "arms",
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
        "seed",
        "arms",
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
        "processes",
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
class _RuntimeQualificationBundle:
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


class QualificationProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def result(self) -> QualificationCommandResult: ...

    def terminate(self) -> None: ...


class QualificationProcessLauncher(Protocol):
    def start(self, spec: QualificationCommandSpec) -> QualificationProcess: ...


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
class _SelectedCanaryPlan:
    """All immutable inputs and profile-derived execution geometry."""

    authority: CohortSelectionAuthority
    bundle: _RuntimeQualificationBundle
    environment_receipt: Mapping[str, object]
    environment_receipt_sha256: str
    bootstrap_receipt: Mapping[str, object]
    bootstrap_receipt_sha256: str
    qualification_root: Path
    scratch_root: Path
    output_path: Path
    s3_root: str
    gpu_groups: Mapping[str, tuple[int, ...]]
    cpu_affinities: Mapping[str, tuple[int, int]]
    master_ports: Mapping[str, int]
    target_tokens_per_arm: int
    cohort_pairs: int

    @property
    def profile(self) -> AwsGpuProfile:
        return self.authority.profile

    @property
    def binding(self) -> AuthenticatedSelectionBinding:
        return self.authority.bindings["dense"]


@dataclass(frozen=True)
class CohortSelectionAuthority:
    """Both arm-scoped admissions for one seed and immutable selection."""

    profile: AwsGpuProfile
    seed: int
    arms: tuple[str, str]
    bindings: Mapping[str, AuthenticatedSelectionBinding]


def admit_cohort_provider_selection(
    *,
    authority_root: Path | str,
    repo_root: Path | str,
    runtime_lock_path: Path | str,
    runtime_evidence_path: Path | str,
    store: VersionedProviderSelectionStore,
    account_id: str,
    instance_id: str,
    boot_id: str,
    seed: int,
    expected_selection_version_id: str,
    identity_verifier: object,
    approval_verifier: QualificationApprovalVerifier,
    trusted_public_key_sha256: str,
) -> CohortSelectionAuthority:
    """Re-enter fixed selection authority for both protected arms."""

    admitted = {
        arm: admit_provider_selection(
            authority_root=authority_root,
            repo_root=repo_root,
            runtime_lock_path=runtime_lock_path,
            runtime_evidence_path=runtime_evidence_path,
            store=store,
            account_id=account_id,
            instance_id=instance_id,
            boot_id=boot_id,
            seed=seed,
            arm=arm,
            expected_selection_version_id=expected_selection_version_id,
            identity_verifier=identity_verifier,
            approval_verifier=approval_verifier,
            trusted_public_key_sha256=trusted_public_key_sha256,
        )
        for arm in ("dense", "split90")
    }
    dense = admitted["dense"]
    split90 = admitted["split90"]
    if (
        dense.arm != "dense"
        or split90.arm != "split90"
        or dense.seed != seed
        or split90.seed != seed
        or any(
            getattr(dense, field) != getattr(split90, field)
            for field in dense.__dataclass_fields__
            if field != "arm"
        )
    ):
        raise ValueError("cohort arm admissions do not share one selection")
    profile_relative = {
        "aws-p5.48xlarge-v3": P5_PROFILE_PATH,
        P6_PROFILE_ID: P6_PROFILE_PATH,
    }.get(dense.profile_id)
    if profile_relative is None:
        raise ValueError("cohort selection profile is unsupported")
    profile = load_aws_gpu_profile(
        Path(repo_root).joinpath(*profile_relative.split("/"))
    )
    if (
        profile.profile_id != dense.profile_id
        or profile.provider != dense.provider
        or profile.sha256 != dense.profile_sha256
    ):
        raise ValueError("cohort selection profile bytes differ from authority")
    return CohortSelectionAuthority(
        profile=profile,
        seed=seed,
        arms=("dense", "split90"),
        bindings=MappingProxyType(admitted),
    )


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


def _validate_authenticated_selection(
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


def _validate_cohort_authority(
    authority: CohortSelectionAuthority,
) -> CohortSelectionAuthority:
    if not isinstance(authority, CohortSelectionAuthority):
        raise TypeError("cohort selection authority is required")
    if (
        authority.arms != ("dense", "split90")
        or set(authority.bindings) != {"dense", "split90"}
        or type(authority.seed) is not int
        or authority.seed not in authority.profile.assigned_seeds
    ):
        raise ValueError("cohort selection authority scope is invalid")
    dense = _validate_authenticated_selection(
        authority.profile,
        authority.bindings["dense"],
    )
    split90 = _validate_authenticated_selection(
        authority.profile,
        authority.bindings["split90"],
    )
    if (
        dense.arm != "dense"
        or split90.arm != "split90"
        or dense.seed != authority.seed
        or split90.seed != authority.seed
        or any(
            getattr(dense, field) != getattr(split90, field)
            for field in dense.__dataclass_fields__
            if field != "arm"
        )
    ):
        raise ValueError("cohort selection arm/seed authority differs")
    return authority


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
    try:
        sbom = _reviewed_runtime_sbom_parser()(data)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("qualification runtime SBOM provenance is invalid") from error
    source = sbom["source"]
    host = sbom["host"]
    container = sbom["container"]
    project = sbom["project_dependency_lock"]
    build_inputs = container["build_inputs"]
    worker_sha256 = hashlib.sha256(
        Path(__file__).with_name("qualification_worker.py").read_bytes()
    ).hexdigest()
    expected_build_inputs = {
        "dockerfile_sha256",
        "dockerignore_sha256",
        "dependency_lock_sha256",
        "inspection_script_sha256",
        "qualification_worker_sha256",
    }
    if (
        sbom["runtime_lock_sha256"] != lock_sha256
        or source
        != {
            "commit": lock["source_commit"],
            "tree": lock["source_tree"],
        }
        or host["ami_id"] != lock["ami_id"]
        or host["ami_owner_id"] != lock["ami_owner_id"]
        or host["architecture"] != profile.architecture
        or container["image"] != lock["container_image"]
        or container["image_digest"] != lock["container_image_digest"]
        or container["versions"]
        != {
            field: lock["versions"][field]
            for field in sorted(_CONTAINER_FACT_FIELDS)
        }
        or not isinstance(project["packages"], list)
        or not project["packages"]
        or set(build_inputs) != expected_build_inputs
        or build_inputs["qualification_worker_sha256"] != worker_sha256
    ):
        raise ValueError(
            "qualification runtime SBOM closure or worker provenance differs"
        )
    host_versions = host["versions"]
    if profile.profile_id == P6_PROFILE_ID:
        for field, floor in _P6_HOST_FLOORS.items():
            if field not in host_versions:
                raise ValueError(f"qualification runtime SBOM NVLSM/{field} missing")
            _at_least(
                host_versions[field],
                floor,
                label=f"P6 runtime SBOM host {field}",
            )
    return sbom


@lru_cache(maxsize=1)
def _reviewed_runtime_sbom_parser():
    """Load the exact closed parser shipped by the reviewed runtime code."""

    runtime_root = Path(__file__).resolve().parents[2] / "containers" / "aws-gpu"
    build_path = runtime_root / "build_image.py"
    runtime_path = runtime_root / "runtime_lock.py"
    build_spec = importlib.util.spec_from_file_location(
        "_memorysplit_reviewed_build_image",
        build_path,
    )
    runtime_spec = importlib.util.spec_from_file_location(
        "_memorysplit_reviewed_runtime_lock",
        runtime_path,
    )
    if (
        build_spec is None
        or build_spec.loader is None
        or runtime_spec is None
        or runtime_spec.loader is None
    ):
        raise ValueError("reviewed runtime SBOM parser cannot be loaded")
    build_module = importlib.util.module_from_spec(build_spec)
    runtime_module = importlib.util.module_from_spec(runtime_spec)
    prior_build = sys.modules.get("build_image")
    sys.modules[build_spec.name] = build_module
    sys.modules[runtime_spec.name] = runtime_module
    sys.modules["build_image"] = build_module
    try:
        build_spec.loader.exec_module(build_module)
        runtime_spec.loader.exec_module(runtime_module)
    finally:
        if prior_build is None:
            sys.modules.pop("build_image", None)
        else:
            sys.modules["build_image"] = prior_build
    parser = getattr(runtime_module, "parse_runtime_sbom_bytes", None)
    if not callable(parser):
        raise ValueError("reviewed runtime SBOM parser is unavailable")
    return parser


def _load_runtime_qualification_bundle(
    *,
    selection_authority: CohortSelectionAuthority,
    runtime_lock_data: bytes,
    runtime_sbom_data: bytes,
) -> _RuntimeQualificationBundle:
    """Validate and bind exact runtime-lock/SBOM bytes to the selection."""

    authority = _validate_cohort_authority(selection_authority)
    selected_profile = authority.profile
    selection_binding = authority.bindings["dense"]
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
    return _RuntimeQualificationBundle(
        lock=lock,
        lock_sha256=lock_sha256,
        sbom=sbom,
        sbom_sha256=hashlib.sha256(runtime_sbom_data).hexdigest(),
    )


def _selection_fields(
    authority: CohortSelectionAuthority,
) -> dict[str, object]:
    authority = _validate_cohort_authority(authority)
    selected_profile = authority.profile
    selected_binding = authority.bindings["dense"]
    return {
        "cohort_id": selected_binding.cohort_id,
        "hardware_amendment_sha256": selected_binding.amendment_sha256,
        "provider_selection_sha256": selected_binding.selection_sha256,
        "provider_selection_version_id": selected_binding.selection_version_id,
        "profile_id": selected_profile.profile_id,
        "provider": selected_profile.provider,
        "profile_sha256": selected_profile.sha256,
        "qualification_evidence_sha256": (
            selected_binding.qualification_evidence_sha256
        ),
        "preselection_environment_receipt_sha256": (
            selected_binding.environment_receipt_sha256
        ),
        "preselection_canary_receipt_sha256": selected_binding.canary_receipt_sha256,
        "approval_receipt_sha256": selected_binding.approval_receipt_sha256,
        "approval_public_key_sha256": selected_binding.approval_public_key_sha256,
        "account_id": selected_binding.account_id,
        "instance_id": selected_binding.instance_id,
        "boot_id": selected_binding.boot_id,
        "region": selected_binding.region,
        "availability_zone": selected_binding.availability_zone,
        "purchase_model": selected_binding.purchase_model,
        "seed": authority.seed,
        "arms": list(authority.arms),
    }


def _build_selected_environment_receipt(
    *,
    selection_authority: CohortSelectionAuthority,
    runtime_lock_data: bytes,
    runtime_sbom_data: bytes,
    attestation_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Build a post-selection environment receipt from measured attestation."""

    authority = _validate_cohort_authority(selection_authority)
    selected_profile = authority.profile
    selection_binding = authority.bindings["dense"]
    bundle = _load_runtime_qualification_bundle(
        selection_authority=authority,
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
        **_selection_fields(authority),
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


def _canonical_selected_environment_receipt(
    receipt: Mapping[str, object],
) -> bytes:
    """Serialize only the closed selection-bound environment schema."""

    if set(receipt) != _SELECTED_ENVIRONMENT_FIELDS:
        raise ValueError("selected environment receipt fields are not closed")
    return canonical_qualification_json(dict(receipt))


def _parse_selected_environment_receipt_bytes(
    data: bytes,
    *,
    selection_authority: CohortSelectionAuthority,
    runtime_lock_data: bytes,
    runtime_sbom_data: bytes,
) -> dict[str, object]:
    """Revalidate one canonical post-selection environment receipt."""

    authority = _validate_cohort_authority(selection_authority)
    selected_profile = authority.profile
    selection_binding = authority.bindings["dense"]
    bundle = _load_runtime_qualification_bundle(
        selection_authority=authority,
        runtime_lock_data=runtime_lock_data,
        runtime_sbom_data=runtime_sbom_data,
    )
    value = _canonical_object(data, label="selected environment receipt")
    if set(value) != _SELECTED_ENVIRONMENT_FIELDS:
        raise ValueError("selected environment receipt fields are not closed")
    expected = {
        "schema_version": 1,
        "receipt_type": "memorysplit-aws-gpu-environment-selection-v1",
        **_selection_fields(authority),
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


def _build_selected_bootstrap_receipt(
    *,
    selection_authority: CohortSelectionAuthority,
    runtime_lock_data: bytes,
    runtime_sbom_data: bytes,
    environment_receipt: Mapping[str, object],
    hardware_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Build one selection-bound bootstrap receipt without mutating a host."""

    authority = _validate_cohort_authority(selection_authority)
    selected_profile = authority.profile
    selection_binding = authority.bindings["dense"]
    bundle = _load_runtime_qualification_bundle(
        selection_authority=authority,
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
        or environment.get("seed") != authority.seed
        or environment.get("arms") != list(authority.arms)
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
        **_selection_fields(authority),
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


def _canonical_selected_bootstrap_receipt(
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


def _parse_selected_bootstrap_receipt_bytes(
    data: bytes,
    *,
    selection_authority: CohortSelectionAuthority,
    runtime_lock_data: bytes,
    runtime_sbom_data: bytes,
    environment_receipt_sha256: str,
) -> dict[str, object]:
    """Revalidate one canonical selection-bound bootstrap receipt."""

    authority = _validate_cohort_authority(selection_authority)
    selected_profile = authority.profile
    selection_binding = authority.bindings["dense"]
    bundle = _load_runtime_qualification_bundle(
        selection_authority=authority,
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
        **_selection_fields(authority),
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


def _build_selected_canary_plan(
    *,
    selection_authority: CohortSelectionAuthority,
    runtime_lock_data: bytes,
    runtime_sbom_data: bytes,
    environment_receipt: Mapping[str, object],
    bootstrap_receipt: Mapping[str, object],
    qualification_root: Path | str,
    scratch_root: Path | str,
    output_path: Path | str,
    s3_root: str,
    target_tokens_per_arm: int = SELECTED_TARGET_TOKENS_PER_ARM,
    cohort_pairs: int = SELECTED_COHORT_PAIRS,
) -> _SelectedCanaryPlan:
    """Build a side-effect-free selected-profile canary plan."""

    authority = _validate_cohort_authority(selection_authority)
    selected_profile = authority.profile
    selection_binding = authority.bindings["dense"]
    bundle = _load_runtime_qualification_bundle(
        selection_authority=authority,
        runtime_lock_data=runtime_lock_data,
        runtime_sbom_data=runtime_sbom_data,
    )
    environment_data = _canonical_selected_environment_receipt(
        environment_receipt
    )
    environment = _parse_selected_environment_receipt_bytes(
        environment_data,
        selection_authority=authority,
        runtime_lock_data=runtime_lock_data,
        runtime_sbom_data=runtime_sbom_data,
    )
    environment_sha256 = hashlib.sha256(environment_data).hexdigest()
    bootstrap_data = _canonical_selected_bootstrap_receipt(bootstrap_receipt)
    bootstrap = _parse_selected_bootstrap_receipt_bytes(
        bootstrap_data,
        selection_authority=authority,
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
    qualification = Path(
        os.path.abspath(os.fspath(qualification_root))
    ).resolve(strict=True)
    if (
        qualification.is_symlink()
        or not qualification.is_dir()
        or any(
            not (
                qualification
                / "configs"
                / "360m-v3"
                / f"{arm}-s{authority.seed}.yaml"
            ).is_file()
            for arm in authority.arms
        )
    ):
        raise ValueError("selected qualification root lacks reviewed configs")
    _private_scratch(scratch, output)
    gpu_groups, cpu_affinities = _profile_groups(selected_profile)
    port_base = 29_700 + selection_binding.seed * 2
    return _SelectedCanaryPlan(
        authority=authority,
        bundle=bundle,
        environment_receipt=environment,
        environment_receipt_sha256=environment_sha256,
        bootstrap_receipt=bootstrap,
        bootstrap_receipt_sha256=bootstrap_sha256,
        qualification_root=qualification,
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
    plan: _SelectedCanaryPlan,
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
        "--mount",
        (
            f"type=bind,src={plan.qualification_root},"
            "dst=/qualification,readonly"
        ),
        "--cpuset-cpus",
        f"{cpu_affinity[0]}-{cpu_affinity[1]}",
        "--gpus",
        "device=" + ",".join(str(gpu) for gpu in gpu_ids),
        plan.bundle.lock["container_image"],
    )


def _selected_specs(
    plan: _SelectedCanaryPlan,
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
                    "--config",
                    (
                        f"/qualification/configs/360m-v3/"
                        f"{arm}-s{plan.authority.seed}.yaml"
                    ),
                    "--qualification-root",
                    "/qualification",
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


def _validate_group_process_specs(
    plan: _SelectedCanaryPlan,
    specs: Sequence[QualificationCommandSpec],
) -> tuple[QualificationCommandSpec, QualificationCommandSpec]:
    if (
        not isinstance(specs, (list, tuple))
        or len(specs) != 2
        or tuple(spec.arm for spec in specs) != ("dense", "split90")
        or any(spec.world_size != 4 for spec in specs)
        or any(spec.concurrent_group != "paired-4rank" for spec in specs)
        or any(spec.gpu_ids != plan.gpu_groups[str(spec.arm)] for spec in specs)
        or any(
            spec.cpu_affinity != plan.cpu_affinities[str(spec.arm)]
            for spec in specs
        )
        or any(spec.master_port != plan.master_ports[str(spec.arm)] for spec in specs)
    ):
        raise ValueError(
            "simultaneous process GPU/CPU/port geometry differs from profile"
        )
    dense, split90 = specs
    dense_cpus = set(
        range(dense.cpu_affinity[0], dense.cpu_affinity[1] + 1)
    )
    split_cpus = set(
        range(split90.cpu_affinity[0], split90.cpu_affinity[1] + 1)
    )
    if set(dense.gpu_ids) & set(split90.gpu_ids):
        raise ValueError("simultaneous process GPU groups overlap")
    if dense_cpus & split_cpus:
        raise ValueError("simultaneous process CPU groups overlap")
    if dense.master_port == split90.master_port:
        raise ValueError("simultaneous process rendezvous ports overlap")
    return dense, split90


def _run_simultaneous_group_processes(
    plan: _SelectedCanaryPlan,
    specs: Sequence[QualificationCommandSpec],
    *,
    process_launcher: QualificationProcessLauncher,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> tuple[
    dict[str, QualificationCommandResult],
    list[dict[str, object]],
]:
    """Start both four-rank groups before monitoring either to completion."""

    selected = _validate_group_process_specs(plan, specs)
    processes: dict[str, QualificationProcess] = {}
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    statuses: dict[str, int] = {}
    try:
        for spec in selected:
            started = monotonic()
            if not math.isfinite(started):
                raise ValueError("simultaneous process start time is invalid")
            process = process_launcher.start(spec)
            if (
                not isinstance(getattr(process, "pid", None), int)
                or process.pid <= 0
            ):
                raise ValueError("simultaneous process identity is invalid")
            starts[spec.name] = started
            processes[spec.name] = process
        initial = {
            name: process.poll()
            for name, process in processes.items()
        }
        if any(status is not None for status in initial.values()):
            raise ValueError(
                "simultaneous process exited early before both groups ran"
            )
        deadline = max(starts.values()) + max(
            spec.timeout_seconds for spec in selected
        )
        while len(statuses) != len(processes):
            now = monotonic()
            if not math.isfinite(now) or now > deadline:
                raise ValueError("simultaneous process pair timed out")
            for name, process in processes.items():
                if name in statuses:
                    continue
                status = process.poll()
                if status is not None:
                    statuses[name] = status
                    ends[name] = now
            if any(status != 0 for status in statuses.values()):
                raise ValueError("simultaneous process group failed early")
            if len(statuses) != len(processes):
                sleep(0.05)
        if max(starts.values()) >= min(ends.values()):
            raise ValueError("simultaneous process pair has no measured overlap")
        results = {
            name: process.result()
            for name, process in processes.items()
        }
        if any(
            not isinstance(result, QualificationCommandResult)
            or result.returncode != 0
            or result.stderr
            for result in results.values()
        ):
            raise ValueError("simultaneous process result is not clean")
        evidence = [
            {
                "arm": spec.arm,
                "pid": processes[spec.name].pid,
                "started_monotonic": starts[spec.name],
                "ended_monotonic": ends[spec.name],
                "rendezvous": f"127.0.0.1:{spec.master_port}",
                "gpu_ids": list(spec.gpu_ids),
                "cpu_affinity": list(spec.cpu_affinity),
                "returncode": results[spec.name].returncode,
            }
            for spec in selected
        ]
        return results, evidence
    except Exception:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        raise


class _SubprocessQualificationProcess:
    def __init__(
        self,
        process: subprocess.Popen,
        stdout,
        stderr,
    ) -> None:
        self._process = process
        self._stdout = stdout
        self._stderr = stderr
        self.pid = process.pid

    def poll(self) -> int | None:
        return self._process.poll()

    def result(self) -> QualificationCommandResult:
        status = self._process.poll()
        if status is None:
            raise ValueError("qualification process is still running")
        self._stdout.seek(0)
        self._stderr.seek(0)
        return QualificationCommandResult(
            returncode=int(status),
            stdout=self._stdout.read(),
            stderr=self._stderr.read(),
        )

    def terminate(self) -> None:
        if self._process.poll() is None:
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


class SubprocessQualificationProcessLauncher:
    """Production process seam; tests inject deterministic launchers."""

    def start(self, spec: QualificationCommandSpec) -> QualificationProcess:
        stdout = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        stderr = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        try:
            process = subprocess.Popen(
                list(spec.argv),
                cwd=spec.cwd,
                env=dict(spec.environment),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                text=True,
                shell=False,
                start_new_session=True,
            )
        except (OSError, subprocess.SubprocessError):
            stdout.close()
            stderr.close()
            raise
        return _SubprocessQualificationProcess(process, stdout, stderr)


class SubprocessQualificationCommandRunner:
    """Production single-command runner with no shell or inherited env."""

    def run(self, spec: QualificationCommandSpec) -> QualificationCommandResult:
        try:
            completed = subprocess.run(
                list(spec.argv),
                cwd=spec.cwd,
                env=dict(spec.environment),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
                timeout=spec.timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError(
                f"qualification command could not execute: {spec.name}"
            ) from error
        return QualificationCommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


class SystemQualificationTimeReader:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()


class AwsCliQualificationObjectStore:
    """Adapt the reviewed versioned S3 canary store to selected receipts."""

    def __init__(
        self,
        *,
        region: str,
        scratch_root: Path | str,
    ) -> None:
        from cluster.aws.p5.canary import (
            AwsCliObjectStore,
            SubprocessCommandReader,
        )

        self._store = AwsCliObjectStore(
            reader=SubprocessCommandReader(),
            region=region,
            scratch_root=Path(scratch_root),
        )

    def put(
        self,
        uri: str,
        payload: bytes,
        *,
        sha256: str,
    ) -> QualificationObjectWrite:
        written = self._store.put(uri, payload, sha256=sha256)
        return QualificationObjectWrite(
            sha256=written.checksum_sha256,
            bytes=written.byte_count,
            version_id=written.version_id,
        )

    def get(
        self,
        uri: str,
        *,
        version_id: str,
    ) -> QualificationObjectRead:
        read = self._store.get(uri, version_id=version_id)
        return QualificationObjectRead(
            payload=read.payload,
            sha256=read.checksum_sha256,
            version_id=read.version_id,
        )


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


def _render_selected_canary_plan(plan: _SelectedCanaryPlan) -> dict[str, object]:
    """Render exact commands and immutable authority without executing them."""

    _validate_cohort_authority(plan.authority)
    hardware, container, groups = _selected_specs(plan)
    return {
        "schema_version": 1,
        "receipt_type": "memorysplit-aws-gpu-qualification-plan-v1",
        "profile_id": plan.profile.profile_id,
        "provider": plan.profile.provider,
        "bindings": {
            **_selection_fields(plan.authority),
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
    plan: _SelectedCanaryPlan,
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
    plan: _SelectedCanaryPlan,
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
    plan: _SelectedCanaryPlan,
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
        "step_seconds",
        "step_tokens",
        "peak_memory_bytes",
        "geometry",
        "sidecar",
        "capabilities",
        "resume",
    }
    geometry = value.get("geometry")
    sidecar = value.get("sidecar")
    capabilities = value.get("capabilities")
    resume = value.get("resume")
    seconds = value.get("step_seconds")
    tokens = value.get("step_tokens")
    expected_sidecar = (
        "dense_target_weights"
        if arm == "dense"
        else "split90_target_weights"
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
        or isinstance(value.get("all_reduce_latency_seconds"), bool)
        or not isinstance(value.get("all_reduce_latency_seconds"), (int, float))
        or not math.isfinite(float(value["all_reduce_latency_seconds"]))
        or value["all_reduce_latency_seconds"] <= 0
        or type(value.get("updates")) is not int
        or value["updates"] != SELECTED_THROUGHPUT_UPDATES
        or type(value.get("warmup_updates")) is not int
        or value["warmup_updates"] != SELECTED_THROUGHPUT_WARMUP_UPDATES
        or type(value.get("peak_memory_bytes")) is not int
        or value["peak_memory_bytes"] <= 0
        or not isinstance(geometry, dict)
        or geometry
        != {
            "arm": arm,
            "model": "d360m",
            "ctx": 1024,
            "micro_batch_size": 8,
            "tokens_per_step": SELECTED_TARGET_TOKENS_PER_ARM // 13_582,
            "total_tokens": SELECTED_TARGET_TOKENS_PER_ARM,
            "max_steps": 13_582,
            "sidecar_name": expected_sidecar,
            "compile": True,
        }
        or not isinstance(seconds, list)
        or len(seconds) != SELECTED_THROUGHPUT_UPDATES
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or item <= 0
            for item in seconds
        )
        or not isinstance(tokens, list)
        or len(tokens) != SELECTED_THROUGHPUT_UPDATES
        or any(
            type(item) is not int
            or item != geometry.get("tokens_per_step")
            for item in tokens
        )
        or not isinstance(sidecar, dict)
        or set(sidecar)
        != {
            "name",
            "stream_sha256",
            "items",
            "nonzero_targets",
            "target_weight_sum",
        }
        or sidecar.get("name") != expected_sidecar
        or _SHA256_RE.fullmatch(str(sidecar.get("stream_sha256"))) is None
        or type(sidecar.get("items")) is not int
        or sidecar["items"] <= 0
        or type(sidecar.get("nonzero_targets")) is not int
        or not 0 < sidecar["nonzero_targets"] <= sidecar["items"]
        or isinstance(sidecar.get("target_weight_sum"), bool)
        or not isinstance(sidecar.get("target_weight_sum"), (int, float))
        or not math.isfinite(float(sidecar["target_weight_sum"]))
        or sidecar["target_weight_sum"] <= 0
        or not isinstance(capabilities, dict)
        or set(capabilities)
        != {
            "bf16_output_finite",
            "sdpa_forward_finite",
            "sdpa_backward_finite",
            "compiled_model",
            "fused_adamw",
            "gradient_finite",
        }
        or any(value is not True for value in capabilities.values())
        or not isinstance(resume, dict)
        or set(resume)
        != {
            "checkpoint_sha256",
            "checkpoint_step",
            "next_step",
            "model_equivalent",
            "optimizer_equivalent",
            "loss_delta",
            "cursor_before",
            "cursor_after",
            "resumed_cursor_after",
            "rng_restored",
            "tolerance",
        }
        or _SHA256_RE.fullmatch(str(resume.get("checkpoint_sha256"))) is None
        or type(resume.get("checkpoint_step")) is not int
        or resume["checkpoint_step"] < 1
        or type(resume.get("next_step")) is not int
        or resume["next_step"] != resume["checkpoint_step"] + 1
        or resume.get("model_equivalent") is not True
        or resume.get("optimizer_equivalent") is not True
        or resume.get("rng_restored") != ["python", "numpy", "torch", "cuda"]
        or resume.get("tolerance") != 1e-6
        or isinstance(resume.get("loss_delta"), bool)
        or not isinstance(resume.get("loss_delta"), (int, float))
        or not math.isfinite(float(resume["loss_delta"]))
        or not 0 <= resume["loss_delta"] <= resume["tolerance"]
        or type(resume.get("cursor_before")) is not int
        or resume["cursor_before"]
        != resume["checkpoint_step"] * geometry.get("tokens_per_step", 0)
        or type(resume.get("cursor_after")) is not int
        or resume["cursor_after"]
        != resume["next_step"] * geometry.get("tokens_per_step", 0)
        or resume.get("resumed_cursor_after") != resume["cursor_after"]
    ):
        raise ValueError(
            f"qualification {arm} geometry/sidecar/finite gradient/resume "
            "checkpoint RNG evidence is invalid"
        )
    retained_rates = [
        tokens[index] / seconds[index]
        for index in range(
            SELECTED_THROUGHPUT_WARMUP_UPDATES,
            SELECTED_THROUGHPUT_UPDATES,
        )
    ]
    normalized = dict(value)
    normalized["median_tok_s"] = float(statistics.median(retained_rates))
    return normalized


def _validate_paired_group_evidence(
    groups: Mapping[str, Mapping[str, object]],
) -> None:
    if set(groups) != {"dense", "split90"}:
        raise ValueError("paired target-weight sidecar evidence is incomplete")
    dense = groups["dense"]["sidecar"]
    split90 = groups["split90"]["sidecar"]
    if (
        dense["name"] != "dense_target_weights"
        or split90["name"] != "split90_target_weights"
        or dense["stream_sha256"] == split90["stream_sha256"]
        or (
            dense["target_weight_sum"] == split90["target_weight_sum"]
            and dense["nonzero_targets"] == split90["nonzero_targets"]
        )
    ):
        raise ValueError(
            "Dense and Split90 arms do not differ by measured target weights"
        )


def _roundtrip_payload(plan: _SelectedCanaryPlan) -> bytes:
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
    plan: _SelectedCanaryPlan,
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
    plan: _SelectedCanaryPlan,
    groups: Mapping[str, Mapping[str, object]],
    process_evidence: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    evidence = _validate_process_evidence(plan, process_evidence)
    overlap_seconds = min(
        row["ended_monotonic"] for row in evidence
    ) - max(row["started_monotonic"] for row in evidence)
    if not math.isfinite(overlap_seconds) or overlap_seconds <= 0:
        raise ValueError("qualification groups have no measured simultaneous overlap")
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
        "simultaneous_groups": overlap_seconds > 0,
        "measured_overlap_seconds": float(overlap_seconds),
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


def _validate_process_evidence(
    plan: _SelectedCanaryPlan,
    process_evidence: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    fields = {
        "arm",
        "pid",
        "started_monotonic",
        "ended_monotonic",
        "rendezvous",
        "gpu_ids",
        "cpu_affinity",
        "returncode",
    }
    if (
        not isinstance(process_evidence, (list, tuple))
        or len(process_evidence) != 2
    ):
        raise ValueError("qualification process evidence pair is incomplete")
    normalized: list[dict[str, object]] = []
    for expected_arm, raw in zip(
        ("dense", "split90"),
        process_evidence,
        strict=True,
    ):
        if not isinstance(raw, Mapping):
            raise ValueError("qualification process evidence is not an object")
        row = dict(raw)
        start = row.get("started_monotonic")
        end = row.get("ended_monotonic")
        if (
            set(row) != fields
            or row.get("arm") != expected_arm
            or type(row.get("pid")) is not int
            or row["pid"] <= 0
            or isinstance(start, bool)
            or not isinstance(start, (int, float))
            or not math.isfinite(float(start))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
            or not math.isfinite(float(end))
            or end <= start
            or row.get("rendezvous")
            != f"127.0.0.1:{plan.master_ports[expected_arm]}"
            or row.get("gpu_ids") != list(plan.gpu_groups[expected_arm])
            or row.get("cpu_affinity")
            != list(plan.cpu_affinities[expected_arm])
            or type(row.get("returncode")) is not int
            or row["returncode"] != 0
        ):
            raise ValueError(
                "qualification simultaneous process evidence is invalid"
            )
        normalized.append(row)
    if normalized[0]["pid"] == normalized[1]["pid"]:
        raise ValueError("qualification simultaneous process IDs are not distinct")
    return normalized


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


def _canonical_selected_qualification_receipt(
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


def _execute_selected_canary(
    plan: _SelectedCanaryPlan,
    *,
    command_runner: QualificationCommandRunner,
    process_launcher: QualificationProcessLauncher,
    object_store: QualificationObjectStore,
    time_reader: QualificationTimeReader,
    process_sleep: Callable[[float], None],
    apply: bool,
) -> dict[str, object]:
    """Run selected-profile qualification through injected side-effect seams."""

    if type(apply) is not bool:
        raise TypeError("selected qualification apply flag must be boolean")
    _validate_cohort_authority(plan.authority)
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
    pair_results, process_evidence = _run_simultaneous_group_processes(
        plan,
        group_specs,
        process_launcher=process_launcher,
        monotonic=time_reader.monotonic,
        sleep=process_sleep,
    )
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
    _validate_paired_group_evidence(groups)
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
                "step_seconds",
                "step_tokens",
                "peak_memory_bytes",
                "geometry",
                "sidecar",
                "capabilities",
                "resume",
            )
        }
        for arm in ("dense", "split90")
    }
    phase_rows.append(
        _phase_record(
            name="nccl_4x4",
            seconds=group_seconds,
            evidence={"groups": nccl, "processes": process_evidence},
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
    telemetry = _telemetry(plan, groups, process_evidence)
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
        **_selection_fields(plan.authority),
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
        "processes": process_evidence,
        "training": training,
        "s3_roundtrip": s3_roundtrip,
        "telemetry": telemetry,
        "passed": True,
        "started_at": started_at,
        "ended_at": ended_at,
        "total_seconds": float(total_seconds),
    }
    payload = _canonical_selected_qualification_receipt(receipt)
    _parse_selected_qualification_receipt_bytes(payload, plan=plan)
    if apply:
        _atomic_no_replace(plan.output_path, payload)
    return receipt


def run_authenticated_selected_canary(
    *,
    authority_root: Path | str,
    repo_root: Path | str,
    runtime_lock_path: Path | str,
    runtime_evidence_path: Path | str,
    runtime_sbom_path: Path | str,
    environment_receipt_path: Path | str,
    bootstrap_receipt_path: Path | str,
    store: VersionedProviderSelectionStore,
    account_id: str,
    instance_id: str,
    boot_id: str,
    seed: int,
    expected_selection_version_id: str,
    identity_verifier: object,
    approval_verifier: QualificationApprovalVerifier,
    trusted_public_key_sha256: str,
    scratch_root: Path | str,
    output_path: Path | str,
    s3_root: str,
    command_runner: QualificationCommandRunner,
    process_launcher: QualificationProcessLauncher,
    object_store: QualificationObjectStore,
    time_reader: QualificationTimeReader,
    process_sleep: Callable[[float], None] = time.sleep,
    apply: bool,
) -> dict[str, object]:
    """Re-admit cohort authority, then render or execute selected canary."""

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
    environment_receipt = _canonical_object(
        read_secure_regular_file(
            environment_receipt_path,
            label="selected environment receipt",
            max_bytes=16 * 1024 * 1024,
        ),
        label="selected environment receipt",
    )
    bootstrap_receipt = _canonical_object(
        read_secure_regular_file(
            bootstrap_receipt_path,
            label="selected bootstrap receipt",
            max_bytes=16 * 1024 * 1024,
        ),
        label="selected bootstrap receipt",
    )
    plan = _build_selected_canary_plan(
        selection_authority=authority,
        runtime_lock_data=runtime_lock_data,
        runtime_sbom_data=runtime_sbom_data,
        environment_receipt=environment_receipt,
        bootstrap_receipt=bootstrap_receipt,
        qualification_root=repo_root,
        scratch_root=scratch_root,
        output_path=output_path,
        s3_root=s3_root,
    )
    if not apply:
        return _render_selected_canary_plan(plan)
    return _execute_selected_canary(
        plan,
        command_runner=command_runner,
        process_launcher=process_launcher,
        object_store=object_store,
        time_reader=time_reader,
        process_sleep=process_sleep,
        apply=True,
    )


def _selected_canary_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cluster.aws.qualification",
        allow_abbrev=False,
    )
    parser.add_argument("--authority-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--runtime-lock", required=True)
    parser.add_argument("--runtime-evidence", required=True)
    parser.add_argument("--runtime-sbom", required=True)
    parser.add_argument("--environment-receipt", required=True)
    parser.add_argument("--bootstrap-receipt", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--boot-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--selection-version-id", required=True)
    parser.add_argument("--trusted-public-key-sha256", required=True)
    parser.add_argument("--selection-bucket", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--approval-public-key", required=True)
    parser.add_argument("--aws-staging-root", required=True)
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--s3-root", required=True)
    parser.add_argument("--apply", action="store_true")
    return parser


def run_selected_canary_cli(
    argv: Sequence[str] | None = None,
    *,
    store: VersionedProviderSelectionStore,
    identity_verifier: object,
    approval_verifier: QualificationApprovalVerifier,
    command_runner: QualificationCommandRunner,
    process_launcher: QualificationProcessLauncher,
    object_store: QualificationObjectStore,
    time_reader: QualificationTimeReader,
    process_sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Execute the selected canary CLI through explicit injected boundaries."""

    arguments = _selected_canary_parser().parse_args(argv)
    return run_authenticated_selected_canary(
        authority_root=arguments.authority_root,
        repo_root=arguments.repo_root,
        runtime_lock_path=arguments.runtime_lock,
        runtime_evidence_path=arguments.runtime_evidence,
        runtime_sbom_path=arguments.runtime_sbom,
        environment_receipt_path=arguments.environment_receipt,
        bootstrap_receipt_path=arguments.bootstrap_receipt,
        store=store,
        account_id=arguments.account_id,
        instance_id=arguments.instance_id,
        boot_id=arguments.boot_id,
        seed=arguments.seed,
        expected_selection_version_id=arguments.selection_version_id,
        identity_verifier=identity_verifier,
        approval_verifier=approval_verifier,
        trusted_public_key_sha256=arguments.trusted_public_key_sha256,
        scratch_root=arguments.scratch_root,
        output_path=arguments.output,
        s3_root=arguments.s3_root,
        command_runner=command_runner,
        process_launcher=process_launcher,
        object_store=object_store,
        time_reader=time_reader,
        process_sleep=process_sleep,
        apply=bool(arguments.apply),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected canary CLI; dry-run is the default."""

    effective_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        arguments = _selected_canary_parser().parse_args(effective_argv)
        command_environment = {
            "AWS_REGION": arguments.region,
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
        }
        store = AwsCliVersionedSelectionStore(
            bucket=arguments.selection_bucket,
            region=arguments.region,
            environment=command_environment,
            staging_root=arguments.aws_staging_root,
        )
        approval = OpenSslQualificationApprovalVerifier(
            public_key_path=arguments.approval_public_key,
            environment=command_environment,
        )
        result = run_selected_canary_cli(
            effective_argv,
            store=store,
            identity_verifier=verify_aws_instance_identity_pkcs7,
            approval_verifier=approval,
            command_runner=SubprocessQualificationCommandRunner(),
            process_launcher=SubprocessQualificationProcessLauncher(),
            object_store=(
                AwsCliQualificationObjectStore(
                    region=arguments.region,
                    scratch_root=arguments.scratch_root,
                )
                if arguments.apply
                else object()
            ),
            time_reader=SystemQualificationTimeReader(),
        )
        report = {
            "schema_version": 1,
            "ok": True,
            "dry_run": not arguments.apply,
            "result": result,
        }
        status = 0
    except (OSError, TypeError, ValueError) as error:
        report = {
            "schema_version": 1,
            "ok": False,
            "dry_run": "--apply" not in effective_argv,
            "error": str(error),
        }
        status = 2
    sys.stdout.buffer.write(canonical_qualification_json(report))
    return status


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


def _parse_selected_qualification_receipt_bytes(
    data: bytes,
    *,
    plan: _SelectedCanaryPlan,
) -> dict[str, object]:
    """Parse and independently revalidate one selected qualification receipt."""

    _validate_cohort_authority(plan.authority)
    value = _canonical_object(data, label="selected qualification receipt")
    if set(value) != _SELECTED_RECEIPT_FIELDS:
        raise ValueError("selected qualification receipt fields do not match")
    expected = {
        "schema_version": 2,
        "receipt_type": "memorysplit-aws-gpu-qualification-v2",
        **_selection_fields(plan.authority),
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
    processes = _validate_process_evidence(plan, value["processes"])
    training = value["training"]
    telemetry_value = value["telemetry"]
    if (
        not isinstance(rows, list)
        or len(rows) != 2
        or not isinstance(training, dict)
        or set(training) != {"dense", "split90"}
        or not isinstance(telemetry_value, dict)
        or not isinstance(telemetry_value.get("per_arm"), dict)
        or set(telemetry_value["per_arm"]) != {"dense", "split90"}
    ):
        raise ValueError("selected qualification NCCL/training pair is invalid")
    groups: dict[str, dict[str, object]] = {}
    for arm, nccl_row in zip(("dense", "split90"), rows, strict=True):
        if not isinstance(nccl_row, dict) or not isinstance(training[arm], dict):
            raise ValueError("selected qualification group row is invalid")
        merged = {
            **nccl_row,
            **training[arm],
        }
        groups[arm] = _validate_group_result(plan, merged, arm=arm)
    _validate_paired_group_evidence(groups)
    expected_s3 = _s3_roundtrip_shape(plan, value["s3_roundtrip"])
    expected_telemetry = _telemetry(plan, groups, processes)
    if telemetry_value != expected_telemetry:
        raise ValueError("selected qualification telemetry/ETA binding differs")
    _validate_phase_bindings(
        value["phases"],
        evidence_by_phase={
            "hardware": hardware,
            "container": container,
            "nccl_4x4": {"groups": rows, "processes": processes},
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
    plan: _SelectedCanaryPlan,
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
    "CohortSelectionAuthority",
    "P6_BASE_AMI_ID",
    "P6_BASE_AMI_OWNER_ID",
    "P6_PROFILE_ID",
    "SELECTED_COHORT_PAIRS",
    "SELECTED_PHASE_ORDER",
    "SELECTED_TARGET_TOKENS_PER_ARM",
    "SELECTED_THROUGHPUT_UPDATES",
    "SELECTED_THROUGHPUT_WARMUP_UPDATES",
    "QualificationCommandResult",
    "QualificationCommandRunner",
    "QualificationCommandSpec",
    "QualificationObjectRead",
    "QualificationObjectStore",
    "QualificationObjectWrite",
    "QualificationProcessLauncher",
    "QualificationTimeReader",
    "AwsCliQualificationObjectStore",
    "SubprocessQualificationCommandRunner",
    "SubprocessQualificationProcessLauncher",
    "SystemQualificationTimeReader",
    "admit_cohort_provider_selection",
    "canonical_qualification_json",
    "main",
    "run_authenticated_selected_canary",
    "run_selected_canary_cli",
]


if __name__ == "__main__":
    raise SystemExit(main())
