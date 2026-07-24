"""Authenticated selected-provider bindings shared by the AWS v3 lifecycle."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from cluster.aws.gpu_profile import AwsGpuProfile, read_secure_regular_file
from cluster.aws.qualification import (
    CohortSelectionAuthority,
    _load_runtime_qualification_bundle,
    admit_cohort_provider_selection,
)

from .aws_hardware import (
    AuthenticatedSelectionBinding,
    QualificationApprovalVerifier,
    VersionedProviderSelectionStore,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_ACCOUNT_RE = re.compile(r"^[0-9]{12}$")
_INSTANCE_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
_BOOT_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_RUN_ID_RE = re.compile(
    r"^memorysplit-v3-360m-s(?P<seed>[0-9])-(?P<arm>dense|split90)$"
)
_PROFILE_PROVIDERS = {
    "aws-p5.48xlarge-v3": "aws-p5.48xlarge",
    "aws-p6-b300.48xlarge-v3": "aws-p6-b300.48xlarge",
}

LIFECYCLE_BINDING_FIELDS = (
    "account_id",
    "arms",
    "availability_zone",
    "boot_id",
    "cohort_id",
    "hardware_amendment_sha256",
    "instance_id",
    "objective_controls_contract_sha256",
    "profile_id",
    "profile_sha256",
    "provider",
    "provider_selection_sha256",
    "provider_selection_version_id",
    "purchase_model",
    "qualification_approval_public_key_sha256",
    "qualification_approval_receipt_sha256",
    "qualification_canary_receipt_sha256",
    "qualification_environment_receipt_sha256",
    "qualification_evidence_sha256",
    "region",
    "runtime_lock_sha256",
    "runtime_sbom_sha256",
    "seed",
)

OPERATIONAL_METADATA_FIELDS = frozenset(
    {
        *LIFECYCLE_BINDING_FIELDS,
        "arm",
        "config_sha256",
        "dataset_build_id",
        "dataset_receipt_sha256",
        "ordered_stream_sha256",
        "run_id",
        "source_commit",
        "source_tree",
    }
)


def load_objective_controls_contract(path: Path | str):
    """Lazily load YAML-backed objective authority for `python -S` entrypoints."""

    from .objective_controls_v3 import (
        load_objective_controls_contract as load_contract,
    )

    return load_contract(path)


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
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


@dataclass(frozen=True)
class ProviderLifecycleBinding:
    """Canonical provider/runtime/qualification commitments for one seed pair."""

    cohort_id: str
    provider: str
    profile_id: str
    profile_sha256: str
    hardware_amendment_sha256: str
    provider_selection_sha256: str
    provider_selection_version_id: str
    runtime_lock_sha256: str
    runtime_sbom_sha256: str
    qualification_evidence_sha256: str
    qualification_environment_receipt_sha256: str
    qualification_canary_receipt_sha256: str
    qualification_approval_receipt_sha256: str
    qualification_approval_public_key_sha256: str
    objective_controls_contract_sha256: str
    account_id: str
    instance_id: str
    boot_id: str
    region: str
    availability_zone: str
    purchase_model: str
    seed: int
    arms: tuple[str, str]

    def __post_init__(self) -> None:
        for label, value in (
            ("cohort ID", self.cohort_id),
            ("provider", self.provider),
            ("profile ID", self.profile_id),
            ("provider-selection version", self.provider_selection_version_id),
            ("region", self.region),
            ("availability zone", self.availability_zone),
            ("purchase model", self.purchase_model),
        ):
            _fixed_text(value, label=label)
        if self.provider_selection_version_id == "null":
            raise ValueError("provider-selection version must be non-null")
        if _PROFILE_PROVIDERS.get(self.profile_id) != self.provider:
            raise ValueError("provider lifecycle profile and provider differ")
        for label, value in (
            ("profile", self.profile_sha256),
            ("hardware amendment", self.hardware_amendment_sha256),
            ("provider selection", self.provider_selection_sha256),
            ("runtime lock", self.runtime_lock_sha256),
            ("runtime SBOM", self.runtime_sbom_sha256),
            ("qualification evidence", self.qualification_evidence_sha256),
            (
                "qualification environment receipt",
                self.qualification_environment_receipt_sha256,
            ),
            (
                "qualification canary receipt",
                self.qualification_canary_receipt_sha256,
            ),
            (
                "qualification approval receipt",
                self.qualification_approval_receipt_sha256,
            ),
            (
                "qualification approval public key",
                self.qualification_approval_public_key_sha256,
            ),
            (
                "objective-controls contract",
                self.objective_controls_contract_sha256,
            ),
        ):
            _sha256(value, label=label)
        if (
            _ACCOUNT_RE.fullmatch(self.account_id) is None
            or _INSTANCE_RE.fullmatch(self.instance_id) is None
            or _BOOT_RE.fullmatch(self.boot_id) is None
        ):
            raise ValueError("provider lifecycle instance identity is invalid")
        if type(self.seed) is not int or self.seed not in range(10):
            raise ValueError("provider lifecycle seed must be integer 0 through 9")
        if self.arms != ("dense", "split90"):
            raise ValueError("provider lifecycle requires Dense and Split90 scopes")

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "account_id": self.account_id,
            "arms": list(self.arms),
            "availability_zone": self.availability_zone,
            "boot_id": self.boot_id,
            "cohort_id": self.cohort_id,
            "hardware_amendment_sha256": self.hardware_amendment_sha256,
            "instance_id": self.instance_id,
            "objective_controls_contract_sha256": (
                self.objective_controls_contract_sha256
            ),
            "profile_id": self.profile_id,
            "profile_sha256": self.profile_sha256,
            "provider": self.provider,
            "provider_selection_sha256": self.provider_selection_sha256,
            "provider_selection_version_id": (
                self.provider_selection_version_id
            ),
            "purchase_model": self.purchase_model,
            "qualification_approval_public_key_sha256": (
                self.qualification_approval_public_key_sha256
            ),
            "qualification_approval_receipt_sha256": (
                self.qualification_approval_receipt_sha256
            ),
            "qualification_canary_receipt_sha256": (
                self.qualification_canary_receipt_sha256
            ),
            "qualification_environment_receipt_sha256": (
                self.qualification_environment_receipt_sha256
            ),
            "qualification_evidence_sha256": (
                self.qualification_evidence_sha256
            ),
            "region": self.region,
            "runtime_lock_sha256": self.runtime_lock_sha256,
            "runtime_sbom_sha256": self.runtime_sbom_sha256,
            "seed": self.seed,
        }
        if set(value) != set(LIFECYCLE_BINDING_FIELDS):
            raise RuntimeError("provider lifecycle field declaration drifted")
        return value


@dataclass(frozen=True)
class AuthenticatedProviderLifecycle:
    """Authority output retaining profile and both arm-scoped admissions."""

    binding: ProviderLifecycleBinding
    profile: AwsGpuProfile
    arm_bindings: Mapping[str, AuthenticatedSelectionBinding]

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ProviderLifecycleBinding):
            raise TypeError("provider lifecycle binding is required")
        if not isinstance(self.profile, AwsGpuProfile):
            raise TypeError("authenticated provider profile is required")
        if set(self.arm_bindings) != {"dense", "split90"}:
            raise ValueError("provider lifecycle arm admissions are incomplete")
        dense = self.arm_bindings["dense"]
        split90 = self.arm_bindings["split90"]
        if (
            not isinstance(dense, AuthenticatedSelectionBinding)
            or not isinstance(split90, AuthenticatedSelectionBinding)
            or dense.arm != "dense"
            or split90.arm != "split90"
            or any(
                getattr(dense, field) != getattr(split90, field)
                for field in dense.__dataclass_fields__
                if field != "arm"
            )
            or self.binding.provider != self.profile.provider
            or self.binding.profile_id != self.profile.profile_id
            or self.binding.profile_sha256 != self.profile.sha256
        ):
            raise ValueError(
                "provider lifecycle arm/profile authority does not share one selection"
            )
        object.__setattr__(
            self,
            "arm_bindings",
            MappingProxyType(dict(self.arm_bindings)),
        )


def _binding_from_authority(
    authority: CohortSelectionAuthority,
    *,
    runtime_lock_sha256: str,
    runtime_sbom_sha256: str,
    objective_controls_contract_sha256: str,
) -> ProviderLifecycleBinding:
    if not isinstance(authority, CohortSelectionAuthority):
        raise TypeError("cohort provider-selection authority is required")
    if set(authority.bindings) != {"dense", "split90"}:
        raise ValueError("cohort provider-selection arm authority is incomplete")
    dense = authority.bindings["dense"]
    split90 = authority.bindings["split90"]
    if any(
        getattr(dense, field) != getattr(split90, field)
        for field in dense.__dataclass_fields__
        if field != "arm"
    ):
        raise ValueError("cohort arm provider selections differ")
    if runtime_lock_sha256 != dense.runtime_lock_sha256:
        raise ValueError("authenticated runtime lock differs from provider selection")
    return ProviderLifecycleBinding(
        cohort_id=dense.cohort_id,
        provider=dense.provider,
        profile_id=dense.profile_id,
        profile_sha256=dense.profile_sha256,
        hardware_amendment_sha256=dense.amendment_sha256,
        provider_selection_sha256=dense.selection_sha256,
        provider_selection_version_id=dense.selection_version_id,
        runtime_lock_sha256=runtime_lock_sha256,
        runtime_sbom_sha256=runtime_sbom_sha256,
        qualification_evidence_sha256=dense.qualification_evidence_sha256,
        qualification_environment_receipt_sha256=(
            dense.environment_receipt_sha256
        ),
        qualification_canary_receipt_sha256=dense.canary_receipt_sha256,
        qualification_approval_receipt_sha256=dense.approval_receipt_sha256,
        qualification_approval_public_key_sha256=(
            dense.approval_public_key_sha256
        ),
        objective_controls_contract_sha256=(
            objective_controls_contract_sha256
        ),
        account_id=dense.account_id,
        instance_id=dense.instance_id,
        boot_id=dense.boot_id,
        region=dense.region,
        availability_zone=dense.availability_zone,
        purchase_model=dense.purchase_model,
        seed=authority.seed,
        arms=authority.arms,
    )


def admit_provider_lifecycle(
    *,
    authority_root: Path | str,
    repo_root: Path | str,
    runtime_lock_path: Path | str,
    runtime_evidence_path: Path | str,
    runtime_sbom_path: Path | str,
    objective_controls_amendment_path: Path | str,
    store: VersionedProviderSelectionStore,
    account_id: str,
    instance_id: str,
    boot_id: str,
    seed: int,
    expected_selection_version_id: str,
    identity_verifier: Callable[[Mapping[str, object], str, str], bool],
    approval_verifier: QualificationApprovalVerifier,
    trusted_public_key_sha256: str,
) -> AuthenticatedProviderLifecycle:
    """Re-enter every fixed authority needed by the selected AWS v3 lifecycle."""

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
        label="provider lifecycle runtime lock",
        max_bytes=1024 * 1024,
    )
    runtime_sbom_data = read_secure_regular_file(
        runtime_sbom_path,
        label="provider lifecycle runtime SBOM",
        max_bytes=512 * 1024 * 1024,
    )
    bundle = _load_runtime_qualification_bundle(
        selection_authority=authority,
        runtime_lock_data=runtime_lock_data,
        runtime_sbom_data=runtime_sbom_data,
    )
    objective = load_objective_controls_contract(
        objective_controls_amendment_path
    )
    if (
        objective.protected_outcomes_inspected is not False
        or objective.added_360m_controls != ()
    ):
        raise ValueError("objective controls do not preserve the protected cohort")
    binding = _binding_from_authority(
        authority,
        runtime_lock_sha256=bundle.lock_sha256,
        runtime_sbom_sha256=bundle.sbom_sha256,
        objective_controls_contract_sha256=(
            objective.objective_controls_contract_sha256
        ),
    )
    return AuthenticatedProviderLifecycle(
        binding=binding,
        profile=authority.profile,
        arm_bindings=authority.bindings,
    )


def lifecycle_operational_metadata(
    binding: ProviderLifecycleBinding,
    *,
    run_id: str,
    arm: str,
    config_sha256: str,
    dataset_receipt_sha256: str,
    dataset_build_id: str,
    ordered_stream_sha256: str,
    source_commit: str,
    source_tree: str,
) -> dict[str, object]:
    """Create strict run-scoped metadata from authenticated lifecycle fields."""

    if not isinstance(binding, ProviderLifecycleBinding):
        raise TypeError("authenticated provider lifecycle binding is required")
    match = _RUN_ID_RE.fullmatch(run_id) if isinstance(run_id, str) else None
    if (
        arm not in binding.arms
        or match is None
        or int(match.group("seed")) != binding.seed
        or match.group("arm") != arm
    ):
        raise ValueError("run ID and arm do not match provider lifecycle scope")
    for label, digest in (
        ("config", config_sha256),
        ("dataset receipt", dataset_receipt_sha256),
        ("dataset build", dataset_build_id),
        ("ordered stream", ordered_stream_sha256),
    ):
        _sha256(digest, label=label)
    for label, object_id in (
        ("source commit", source_commit),
        ("source tree", source_tree),
    ):
        if not isinstance(object_id, str) or _GIT_SHA1_RE.fullmatch(object_id) is None:
            raise ValueError(f"{label} must be a lowercase Git SHA-1")
    metadata = {
        **binding.to_dict(),
        "arm": arm,
        "config_sha256": config_sha256,
        "dataset_build_id": dataset_build_id,
        "dataset_receipt_sha256": dataset_receipt_sha256,
        "ordered_stream_sha256": ordered_stream_sha256,
        "run_id": run_id,
        "source_commit": source_commit,
        "source_tree": source_tree,
    }
    if set(metadata) != OPERATIONAL_METADATA_FIELDS:
        raise RuntimeError("provider lifecycle operational metadata drifted")
    return metadata


def _same_typed_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _same_typed_value(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_typed_value(left, right)
            for left, right in zip(actual, expected)
        )
    return actual == expected


def validate_lifecycle_operational_metadata(
    value: object,
    *,
    expected: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate one closed operational metadata object and optional identity."""

    if (
        not isinstance(value, dict)
        or set(value) != OPERATIONAL_METADATA_FIELDS
        or not all(isinstance(key, str) for key in value)
    ):
        raise ValueError("provider lifecycle operational metadata fields are invalid")
    arms = value["arms"]
    try:
        binding = ProviderLifecycleBinding(
            cohort_id=value["cohort_id"],
            provider=value["provider"],
            profile_id=value["profile_id"],
            profile_sha256=value["profile_sha256"],
            hardware_amendment_sha256=value[
                "hardware_amendment_sha256"
            ],
            provider_selection_sha256=value[
                "provider_selection_sha256"
            ],
            provider_selection_version_id=value[
                "provider_selection_version_id"
            ],
            runtime_lock_sha256=value["runtime_lock_sha256"],
            runtime_sbom_sha256=value["runtime_sbom_sha256"],
            qualification_evidence_sha256=value[
                "qualification_evidence_sha256"
            ],
            qualification_environment_receipt_sha256=value[
                "qualification_environment_receipt_sha256"
            ],
            qualification_canary_receipt_sha256=value[
                "qualification_canary_receipt_sha256"
            ],
            qualification_approval_receipt_sha256=value[
                "qualification_approval_receipt_sha256"
            ],
            qualification_approval_public_key_sha256=value[
                "qualification_approval_public_key_sha256"
            ],
            objective_controls_contract_sha256=value[
                "objective_controls_contract_sha256"
            ],
            account_id=value["account_id"],
            instance_id=value["instance_id"],
            boot_id=value["boot_id"],
            region=value["region"],
            availability_zone=value["availability_zone"],
            purchase_model=value["purchase_model"],
            seed=value["seed"],
            arms=tuple(arms) if isinstance(arms, list) else arms,
        )
        canonical = lifecycle_operational_metadata(
            binding,
            run_id=value["run_id"],
            arm=value["arm"],
            config_sha256=value["config_sha256"],
            dataset_receipt_sha256=value["dataset_receipt_sha256"],
            dataset_build_id=value["dataset_build_id"],
            ordered_stream_sha256=value["ordered_stream_sha256"],
            source_commit=value["source_commit"],
            source_tree=value["source_tree"],
        )
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError(
            "provider lifecycle operational metadata values are invalid"
        ) from error
    if not _same_typed_value(value, canonical):
        raise ValueError("provider lifecycle operational metadata is not canonical")
    if expected is not None and not _same_typed_value(value, dict(expected)):
        raise ValueError(
            "provider lifecycle operational metadata differs from expected selection"
        )
    return dict(value)


__all__ = [
    "AuthenticatedProviderLifecycle",
    "LIFECYCLE_BINDING_FIELDS",
    "OPERATIONAL_METADATA_FIELDS",
    "ProviderLifecycleBinding",
    "admit_provider_lifecycle",
    "lifecycle_operational_metadata",
    "validate_lifecycle_operational_metadata",
]
