"""Externally rooted study-lock and protected-readiness contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Any, ClassVar

from evals.confirmatory.contracts import (
    CONTRACT_VERSION,
    ConditionId,
    STUDY_CONTRACT_VERSION,
    STUDY_TARGETS_PER_UPDATE,
    StudyArm,
    canonical_sha256,
)
from msctl.aws_contracts import ARMS, SEEDS, SNAPSHOT_STEPS
from msctl.aws_contracts import checkpoint_receipt_key, snapshot_object_key
from msctl.aws_hardware import (
    AWS_HARDWARE_AMENDMENT_SHA256,
    PROVIDER_SELECTION_S3_KEY,
    AuthenticatedSelectionBinding,
)


STUDY_LOCK_SCHEMA = "memorysplit.confirmatory.study-lock.v2"
STUDY_LOCK_SCHEMA_V3 = "memorysplit.confirmatory.study-lock.v3"
VALIDITY_EVIDENCE_SCHEMA = "memorysplit.confirmatory.validity-evidence.v2"
FROZEN_PREREGISTRATION_SHA256 = (
    "fee38e363298d3def46b741320c9d7df4523d0ff3cd249187cf52d54046cbbf0"
)
FROZEN_PREREGISTRATION_SHA256_V3 = (
    "6b2b5da3e3dc3d533498a0aa9d1891f356134ce045b1553b74a0161d94cb81d7"
)
EXPECTED_STUDY_SLOTS_V3 = tuple(
    (seed, StudyArm(arm), optimizer_step)
    for seed in SEEDS
    for arm in ARMS
    for optimizer_step in SNAPSHOT_STEPS
)
REQUIRED_FAMILIES = ("graph", "non_path")
REQUIRED_STRATA = (
    "iid",
    "composition_ood",
    "length_ood",
    "joint_ood",
)
REQUIRED_MEMORY_MODES = ("memory_off", "memory_on")
REQUIRED_CONTROL_IDS = (
    "correct_memory",
    "memory_off",
    "shuffled_returns",
    "relevant_edge_swap",
    "irrelevant_edge_swap",
    "gold_path_replay",
    "no_query",
    "entity_rename",
    "graph_isomorphism",
    "page_order_permutation",
)
REQUIRED_GATE_IDS = (
    "scientific_contract",
    "route_dose",
    "semantic_closure",
    "proof_verification",
    "ood_seal",
    "corpus_identity",
    "paired_training",
    "checkpoint_resume",
    "evaluation_validity",
    "six_29m_diagnostics",
)
REQUIRED_RECEIPTS = (
    *(f"gate:{gate_id}" for gate_id in REQUIRED_GATE_IDS),
    "guardrail:iid",
    *(f"control:{control_id}" for control_id in REQUIRED_CONTROL_IDS),
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_S3_VERSION_ID_LENGTH = 1_024
_STUDY_COHORT_ID_V3 = "memorysplit-confirmatory-v3-360m-n10-aws"
_ELIGIBLE_PROFILE_IDENTITIES_V3 = frozenset(
    {
        (
            "aws-p5.48xlarge-v3",
            "aws-p5.48xlarge",
            "2207bfbad5e8fa9fc804770b582d0b21f8b6ed109b2e3f3b5c0474c732c53543",
        ),
        (
            "aws-p6-b300.48xlarge-v3",
            "aws-p6-b300.48xlarge",
            "6884cd30670214bcecaa105d32b2b5518b1533d9fdbad6ac15327f9a8b7fefa4",
        ),
    }
)


def _strict_fields(
    raw: object,
    expected: frozenset[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise ValueError(f"{name} fields are not exact")
    return raw


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _s3_version_id(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= _MAX_S3_VERSION_ID_LENGTH
        or value == "null"
        or not value.isprintable()
        or any(character.isspace() for character in value)
    ):
        raise ValueError(
            f"{name} must be a bounded printable non-null version candidate"
        )
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _schema(value: object, name: str) -> int:
    if type(value) is not int or value != CONTRACT_VERSION:
        raise ValueError(f"{name} schema_version is invalid")
    return value


def _schema_v3(value: object, name: str) -> int:
    if type(value) is not int or value != STUDY_CONTRACT_VERSION:
        raise ValueError(f"{name} schema_version is invalid")
    return value


def _ordered_strings(
    value: object,
    name: str,
    *,
    expected: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an ordered sequence")
    result = tuple(_string(item, f"{name} item") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicates")
    if expected is not None:
        if result != expected:
            raise ValueError(f"{name} disagrees with the frozen contract")
    elif result != tuple(sorted(result)):
        raise ValueError(f"{name} must use canonical ordering")
    return result


@dataclass(frozen=True)
class ProviderSelectionBinding:
    """Cohort-wide commitment copied from authenticated selection admission."""

    cohort_id: str
    provider_selection_s3_key: str
    provider_selection_sha256: str
    provider_selection_s3_version_id: str
    hardware_amendment_sha256: str
    selected_provider: str
    profile_id: str
    profile_sha256: str
    runtime_lock_sha256: str
    qualification_evidence_sha256: str
    environment_receipt_sha256: str
    canary_receipt_sha256: str
    approval_receipt_sha256: str
    approval_public_key_sha256: str

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "cohort_id",
            "provider_selection_s3_key",
            "provider_selection_sha256",
            "provider_selection_s3_version_id",
            "hardware_amendment_sha256",
            "selected_provider",
            "profile_id",
            "profile_sha256",
            "runtime_lock_sha256",
            "qualification_evidence_sha256",
            "environment_receipt_sha256",
            "canary_receipt_sha256",
            "approval_receipt_sha256",
            "approval_public_key_sha256",
        }
    )

    def __post_init__(self) -> None:
        if self.cohort_id != _STUDY_COHORT_ID_V3:
            raise ValueError("provider selection cohort identity is invalid")
        if self.provider_selection_s3_key != PROVIDER_SELECTION_S3_KEY:
            raise ValueError("provider selection does not use the fixed S3 key")
        selection = _hash(
            self.provider_selection_sha256,
            "provider selection SHA-256",
        )
        version = _s3_version_id(
            self.provider_selection_s3_version_id,
            "provider selection S3 version ID",
        )
        amendment = _hash(
            self.hardware_amendment_sha256,
            "hardware amendment SHA-256",
        )
        if amendment != AWS_HARDWARE_AMENDMENT_SHA256:
            raise ValueError("hardware amendment commitment is invalid")
        provider = _string(self.selected_provider, "selected provider")
        profile_id = _string(self.profile_id, "profile ID")
        profile = _hash(self.profile_sha256, "profile SHA-256")
        if (profile_id, provider, profile) not in _ELIGIBLE_PROFILE_IDENTITIES_V3:
            raise ValueError(
                "selected provider, profile identity, and profile evidence "
                "are crossed"
            )
        runtime = _hash(self.runtime_lock_sha256, "runtime-lock SHA-256")
        qualification = _hash(
            self.qualification_evidence_sha256,
            "qualification evidence SHA-256",
        )
        environment = _hash(
            self.environment_receipt_sha256,
            "environment receipt SHA-256",
        )
        canary = _hash(
            self.canary_receipt_sha256,
            "canary receipt SHA-256",
        )
        approval = _hash(
            self.approval_receipt_sha256,
            "approval receipt SHA-256",
        )
        approval_key = _hash(
            self.approval_public_key_sha256,
            "approval public-key SHA-256",
        )
        object.__setattr__(self, "provider_selection_sha256", selection)
        object.__setattr__(
            self,
            "provider_selection_s3_version_id",
            version,
        )
        object.__setattr__(self, "hardware_amendment_sha256", amendment)
        object.__setattr__(self, "selected_provider", provider)
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "profile_sha256", profile)
        object.__setattr__(self, "runtime_lock_sha256", runtime)
        object.__setattr__(
            self,
            "qualification_evidence_sha256",
            qualification,
        )
        object.__setattr__(
            self,
            "environment_receipt_sha256",
            environment,
        )
        object.__setattr__(self, "canary_receipt_sha256", canary)
        object.__setattr__(self, "approval_receipt_sha256", approval)
        object.__setattr__(
            self,
            "approval_public_key_sha256",
            approval_key,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ProviderSelectionBinding":
        return cls(
            **dict(
                _strict_fields(
                    raw,
                    cls.FIELDS,
                    "provider selection binding",
                )
            )
        )

    @classmethod
    def from_authenticated(
        cls,
        binding: AuthenticatedSelectionBinding,
    ) -> "ProviderSelectionBinding":
        if not isinstance(binding, AuthenticatedSelectionBinding):
            raise TypeError(
                "provider selection requires authenticated admission output"
            )
        return cls(
            cohort_id=binding.cohort_id,
            provider_selection_s3_key=PROVIDER_SELECTION_S3_KEY,
            provider_selection_sha256=binding.selection_sha256,
            provider_selection_s3_version_id=binding.selection_version_id,
            hardware_amendment_sha256=binding.amendment_sha256,
            selected_provider=binding.provider,
            profile_id=binding.profile_id,
            profile_sha256=binding.profile_sha256,
            runtime_lock_sha256=binding.runtime_lock_sha256,
            qualification_evidence_sha256=(
                binding.qualification_evidence_sha256
            ),
            environment_receipt_sha256=(
                binding.environment_receipt_sha256
            ),
            canary_receipt_sha256=binding.canary_receipt_sha256,
            approval_receipt_sha256=binding.approval_receipt_sha256,
            approval_public_key_sha256=(
                binding.approval_public_key_sha256
            ),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "cohort_id": self.cohort_id,
            "provider_selection_s3_key": self.provider_selection_s3_key,
            "provider_selection_sha256": self.provider_selection_sha256,
            "provider_selection_s3_version_id": (
                self.provider_selection_s3_version_id
            ),
            "hardware_amendment_sha256": self.hardware_amendment_sha256,
            "selected_provider": self.selected_provider,
            "profile_id": self.profile_id,
            "profile_sha256": self.profile_sha256,
            "runtime_lock_sha256": self.runtime_lock_sha256,
            "qualification_evidence_sha256": (
                self.qualification_evidence_sha256
            ),
            "environment_receipt_sha256": (
                self.environment_receipt_sha256
            ),
            "canary_receipt_sha256": self.canary_receipt_sha256,
            "approval_receipt_sha256": self.approval_receipt_sha256,
            "approval_public_key_sha256": (
                self.approval_public_key_sha256
            ),
        }


@dataclass(frozen=True)
class StudySnapshotBinding:
    """Candidate S3 identity requiring later versioned HEAD/receipt replay."""

    seed: int
    arm: StudyArm
    optimizer_step: int
    checkpoint_sha256: str
    s3_object_key: str
    s3_version_id: str
    checkpoint_receipt_sha256: str
    checkpoint_receipt_s3_object_key: str
    checkpoint_receipt_s3_version_id: str
    provider_selection_sha256: str
    provider_selection_s3_version_id: str
    snapshot_version: int
    training_run_id: str
    config_fingerprint: str
    model_config_sha256: str
    data_provenance_sha256: str
    world_size: int
    tokens_per_step: int

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "seed",
            "arm",
            "optimizer_step",
            "checkpoint_sha256",
            "s3_object_key",
            "s3_version_id",
            "checkpoint_receipt_sha256",
            "checkpoint_receipt_s3_object_key",
            "checkpoint_receipt_s3_version_id",
            "provider_selection_sha256",
            "provider_selection_s3_version_id",
            "snapshot_version",
            "training_run_id",
            "config_fingerprint",
            "model_config_sha256",
            "data_provenance_sha256",
            "world_size",
            "tokens_per_step",
        }
    )

    def __post_init__(self) -> None:
        digest = _hash(self.checkpoint_sha256, "checkpoint_sha256")
        try:
            arm = StudyArm(self.arm)
        except (TypeError, ValueError) as exc:
            raise ValueError("study snapshot arm must be dense or split90") from exc
        expected_key = snapshot_object_key(
            self.seed,
            arm.value,
            self.optimizer_step,
            digest,
        )
        if (
            type(self.optimizer_step) is not int
            or self.optimizer_step not in SNAPSHOT_STEPS
        ):
            raise ValueError("optimizer_step is not one of the five frozen steps")
        object_key = _string(self.s3_object_key, "S3 object key")
        if object_key != expected_key:
            raise ValueError(
                "S3 object key is not the canonical content-addressed "
                "checkpoint key"
            )
        version_id = _s3_version_id(self.s3_version_id, "S3 version ID")
        receipt_digest = _hash(
            self.checkpoint_receipt_sha256,
            "checkpoint receipt SHA-256",
        )
        expected_receipt_key = checkpoint_receipt_key(
            self.seed,
            receipt_digest,
        )
        receipt_key = _string(
            self.checkpoint_receipt_s3_object_key,
            "checkpoint receipt S3 object key",
        )
        if receipt_key != expected_receipt_key:
            raise ValueError(
                "checkpoint receipt S3 object key is not content-addressed"
            )
        receipt_version_id = _s3_version_id(
            self.checkpoint_receipt_s3_version_id,
            "checkpoint receipt S3 version ID",
        )
        selection_digest = _hash(
            self.provider_selection_sha256,
            "provider selection SHA-256",
        )
        selection_version_id = _s3_version_id(
            self.provider_selection_s3_version_id,
            "provider selection S3 version ID",
        )
        if type(self.snapshot_version) is not int or self.snapshot_version != 2:
            raise ValueError("study snapshot version must be integer 2")
        expected_training_run_id = (
            f"memorysplit-v3-360m-s{self.seed}-{arm.value}"
        )
        training_run_id = _string(self.training_run_id, "training run ID")
        if training_run_id != expected_training_run_id:
            raise ValueError("study snapshot training run identity is invalid")
        config_fingerprint = _hash(
            self.config_fingerprint,
            "snapshot config fingerprint",
        )
        model_config_sha256 = _hash(
            self.model_config_sha256,
            "snapshot model config SHA-256",
        )
        data_provenance_sha256 = _hash(
            self.data_provenance_sha256,
            "snapshot data provenance SHA-256",
        )
        if type(self.world_size) is not int or self.world_size != 4:
            raise ValueError("study snapshot world_size must be exactly 4")
        if (
            type(self.tokens_per_step) is not int
            or self.tokens_per_step != STUDY_TARGETS_PER_UPDATE
        ):
            raise ValueError(
                "study snapshot tokens_per_step must be exactly 524288"
            )
        object.__setattr__(self, "checkpoint_sha256", digest)
        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "s3_object_key", object_key)
        object.__setattr__(self, "s3_version_id", version_id)
        object.__setattr__(
            self,
            "checkpoint_receipt_sha256",
            receipt_digest,
        )
        object.__setattr__(
            self,
            "checkpoint_receipt_s3_object_key",
            receipt_key,
        )
        object.__setattr__(
            self,
            "checkpoint_receipt_s3_version_id",
            receipt_version_id,
        )
        object.__setattr__(
            self,
            "provider_selection_sha256",
            selection_digest,
        )
        object.__setattr__(
            self,
            "provider_selection_s3_version_id",
            selection_version_id,
        )
        object.__setattr__(self, "training_run_id", training_run_id)
        object.__setattr__(
            self,
            "config_fingerprint",
            config_fingerprint,
        )
        object.__setattr__(
            self,
            "model_config_sha256",
            model_config_sha256,
        )
        object.__setattr__(
            self,
            "data_provenance_sha256",
            data_provenance_sha256,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StudySnapshotBinding":
        value = _strict_fields(raw, cls.FIELDS, "study snapshot binding")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "arm": self.arm.value,
            "optimizer_step": self.optimizer_step,
            "checkpoint_sha256": self.checkpoint_sha256,
            "s3_object_key": self.s3_object_key,
            "s3_version_id": self.s3_version_id,
            "checkpoint_receipt_sha256": self.checkpoint_receipt_sha256,
            "checkpoint_receipt_s3_object_key": (
                self.checkpoint_receipt_s3_object_key
            ),
            "checkpoint_receipt_s3_version_id": (
                self.checkpoint_receipt_s3_version_id
            ),
            "provider_selection_sha256": self.provider_selection_sha256,
            "provider_selection_s3_version_id": (
                self.provider_selection_s3_version_id
            ),
            "snapshot_version": self.snapshot_version,
            "training_run_id": self.training_run_id,
            "config_fingerprint": self.config_fingerprint,
            "model_config_sha256": self.model_config_sha256,
            "data_provenance_sha256": self.data_provenance_sha256,
            "world_size": self.world_size,
            "tokens_per_step": self.tokens_per_step,
        }


@dataclass(frozen=True)
class StudyLockV3:
    """No-replace registry for the exact 100 protected AWS snapshots."""

    record_type: str
    schema_version: int
    preregistration_sha256: str
    sealed_evaluation_release_sha256: str
    provider_selection: ProviderSelectionBinding
    snapshots: tuple[StudySnapshotBinding, ...]

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "record_type",
            "schema_version",
            "preregistration_sha256",
            "sealed_evaluation_release_sha256",
            "provider_selection",
            "snapshots",
        }
    )

    def __post_init__(self) -> None:
        if self.record_type != STUDY_LOCK_SCHEMA_V3:
            raise ValueError("v3 study lock record_type is invalid")
        _schema_v3(self.schema_version, "v3 study lock")
        preregistration = _hash(
            self.preregistration_sha256,
            "preregistration_sha256",
        )
        if preregistration != FROZEN_PREREGISTRATION_SHA256_V3:
            raise ValueError(
                "v3 study lock preregistration commitment is invalid"
            )
        release = _hash(
            self.sealed_evaluation_release_sha256,
            "sealed evaluation release SHA-256",
        )
        selection = (
            self.provider_selection
            if isinstance(self.provider_selection, ProviderSelectionBinding)
            else ProviderSelectionBinding.from_dict(self.provider_selection)
        )
        if not isinstance(self.snapshots, (list, tuple)):
            raise ValueError("v3 study lock snapshots must be ordered")
        snapshots = tuple(
            snapshot
            if isinstance(snapshot, StudySnapshotBinding)
            else StudySnapshotBinding.from_dict(snapshot)
            for snapshot in self.snapshots
        )
        slots = tuple(
            (snapshot.seed, snapshot.arm, snapshot.optimizer_step)
            for snapshot in snapshots
        )
        if slots != EXPECTED_STUDY_SLOTS_V3:
            raise ValueError(
                "v3 study lock requires the exact ordered 100 snapshot slots"
            )
        if any(
            snapshot.provider_selection_sha256
            != selection.provider_selection_sha256
            or snapshot.provider_selection_s3_version_id
            != selection.provider_selection_s3_version_id
            for snapshot in snapshots
        ):
            raise ValueError(
                "every snapshot must share the cohort provider selection"
            )
        checkpoint_hashes = tuple(
            snapshot.checkpoint_sha256 for snapshot in snapshots
        )
        object_versions = tuple(
            (snapshot.s3_object_key, snapshot.s3_version_id)
            for snapshot in snapshots
        )
        if (
            len(set(checkpoint_hashes)) != len(checkpoint_hashes)
            or len(set(object_versions)) != len(object_versions)
        ):
            raise ValueError(
                "v3 study lock rejects checkpoint alias or object reuse "
                "across slots"
            )
        receipt_by_seed_step: dict[
            tuple[int, int],
            tuple[str, str, str],
        ] = {}
        for snapshot in snapshots:
            seed_step = snapshot.seed, snapshot.optimizer_step
            receipt = (
                snapshot.checkpoint_receipt_sha256,
                snapshot.checkpoint_receipt_s3_object_key,
                snapshot.checkpoint_receipt_s3_version_id,
            )
            previous = receipt_by_seed_step.setdefault(seed_step, receipt)
            if previous != receipt:
                raise ValueError(
                    "paired snapshot slots disagree on checkpoint receipt"
                )
        receipt_content_identities = tuple(
            (receipt_sha256, receipt_key)
            for receipt_sha256, receipt_key, _version_id
            in receipt_by_seed_step.values()
        )
        if len(set(receipt_content_identities)) != len(
            receipt_content_identities
        ):
            raise ValueError(
                "checkpoint receipt content is reused across seed/step slots"
            )
        object.__setattr__(self, "preregistration_sha256", preregistration)
        object.__setattr__(
            self,
            "sealed_evaluation_release_sha256",
            release,
        )
        object.__setattr__(self, "provider_selection", selection)
        object.__setattr__(self, "snapshots", snapshots)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StudyLockV3":
        value = _strict_fields(raw, cls.FIELDS, "v3 study lock")
        return cls(
            record_type=value["record_type"],
            schema_version=value["schema_version"],
            preregistration_sha256=value["preregistration_sha256"],
            sealed_evaluation_release_sha256=value[
                "sealed_evaluation_release_sha256"
            ],
            provider_selection=value["provider_selection"],
            snapshots=value["snapshots"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "preregistration_sha256": self.preregistration_sha256,
            "sealed_evaluation_release_sha256": (
                self.sealed_evaluation_release_sha256
            ),
            "provider_selection": self.provider_selection.to_dict(),
            "snapshots": [
                snapshot.to_dict() for snapshot in self.snapshots
            ],
        }


@dataclass(frozen=True)
class EvaluationCellIdentity:
    item_id: str
    checkpoint_sha256: str
    seed: int
    condition_id: ConditionId

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"item_id", "checkpoint_sha256", "seed", "condition_id"}
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_id", _string(self.item_id, "item_id"))
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _hash(self.checkpoint_sha256, "checkpoint_sha256"),
        )
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        try:
            condition = ConditionId(self.condition_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("condition_id is invalid") from exc
        object.__setattr__(self, "condition_id", condition)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvaluationCellIdentity":
        return cls(**dict(_strict_fields(raw, cls.FIELDS, "evaluation cell")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "seed": self.seed,
            "condition_id": self.condition_id.value,
        }


@dataclass(frozen=True)
class CheckpointApproval:
    checkpoint_sha256: str
    seed: int
    condition_id: ConditionId
    configuration_sha256: str
    route_dose_sha256: str

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "checkpoint_sha256",
            "seed",
            "condition_id",
            "configuration_sha256",
            "route_dose_sha256",
        }
    )

    def __post_init__(self) -> None:
        for field in (
            "checkpoint_sha256",
            "configuration_sha256",
            "route_dose_sha256",
        ):
            object.__setattr__(
                self,
                field,
                _hash(getattr(self, field), field),
            )
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        try:
            condition = ConditionId(self.condition_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint approval condition_id is invalid") from exc
        object.__setattr__(self, "condition_id", condition)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CheckpointApproval":
        return cls(
            **dict(_strict_fields(raw, cls.FIELDS, "checkpoint approval"))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "seed": self.seed,
            "condition_id": self.condition_id.value,
            "configuration_sha256": self.configuration_sha256,
            "route_dose_sha256": self.route_dose_sha256,
        }


class ReceiptState(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    PENDING = "pending"


@dataclass(frozen=True)
class ReceiptCommitment:
    receipt_id: str
    kind: str
    state: ReceiptState
    receipt_sha256: str

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"receipt_id", "kind", "state", "receipt_sha256"}
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receipt_id",
            _string(self.receipt_id, "receipt_id"),
        )
        if self.kind not in {"gate", "guardrail", "control"}:
            raise ValueError("receipt commitment kind is invalid")
        try:
            state = ReceiptState(self.state)
        except (TypeError, ValueError) as exc:
            raise ValueError("receipt commitment state is invalid") from exc
        object.__setattr__(self, "state", state)
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(self.receipt_sha256, "receipt_sha256"),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ReceiptCommitment":
        return cls(**dict(_strict_fields(raw, cls.FIELDS, "receipt commitment")))

    def to_dict(self) -> dict[str, str]:
        return {
            "receipt_id": self.receipt_id,
            "kind": self.kind,
            "state": self.state.value,
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True)
class ReleaseBinding:
    items_sha256: str
    sealed_gold_sha256: str
    stores_sha256: str
    checkpoints_sha256: str
    item_ids: tuple[str, ...]
    pair_ids: tuple[str, ...]
    world_ids: tuple[str, ...]
    evaluation_cells: tuple[EvaluationCellIdentity, ...]
    item_count: int
    pair_count: int
    world_count: int
    evaluation_cell_count: int
    required_families: tuple[str, ...]
    required_strata: tuple[str, ...]
    required_memory_modes: tuple[str, ...]
    required_controls: tuple[str, ...]

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "items_sha256",
            "sealed_gold_sha256",
            "stores_sha256",
            "checkpoints_sha256",
            "item_ids",
            "pair_ids",
            "world_ids",
            "evaluation_cells",
            "item_count",
            "pair_count",
            "world_count",
            "evaluation_cell_count",
            "required_families",
            "required_strata",
            "required_memory_modes",
            "required_controls",
        }
    )

    def __post_init__(self) -> None:
        for field in (
            "items_sha256",
            "sealed_gold_sha256",
            "stores_sha256",
            "checkpoints_sha256",
        ):
            object.__setattr__(
                self,
                field,
                _hash(getattr(self, field), field),
            )
        for field, count_field in (
            ("item_ids", "item_count"),
            ("pair_ids", "pair_count"),
            ("world_ids", "world_count"),
        ):
            values = _ordered_strings(getattr(self, field), field)
            count = _integer(getattr(self, count_field), count_field)
            if count != len(values):
                raise ValueError(f"{count_field} disagrees with {field}")
            object.__setattr__(self, field, values)
            object.__setattr__(self, count_field, count)
        if not isinstance(self.evaluation_cells, (list, tuple)):
            raise ValueError("evaluation_cells must be ordered")
        cells = tuple(
            cell
            if isinstance(cell, EvaluationCellIdentity)
            else EvaluationCellIdentity.from_dict(cell)
            for cell in self.evaluation_cells
        )
        keys = tuple(
            (
                cell.seed,
                cell.condition_id.value,
                cell.item_id,
                cell.checkpoint_sha256,
            )
            for cell in cells
        )
        if len(set(keys)) != len(keys) or keys != tuple(sorted(keys)):
            raise ValueError("evaluation_cells are duplicated or unordered")
        cell_count = _integer(
            self.evaluation_cell_count,
            "evaluation_cell_count",
        )
        if cell_count != len(cells):
            raise ValueError(
                "evaluation_cell_count disagrees with evaluation_cells"
            )
        object.__setattr__(self, "evaluation_cells", cells)
        object.__setattr__(self, "evaluation_cell_count", cell_count)
        for field, expected in (
            ("required_families", REQUIRED_FAMILIES),
            ("required_strata", REQUIRED_STRATA),
            ("required_memory_modes", REQUIRED_MEMORY_MODES),
            ("required_controls", REQUIRED_CONTROL_IDS),
        ):
            object.__setattr__(
                self,
                field,
                _ordered_strings(getattr(self, field), field, expected=expected),
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ReleaseBinding":
        return cls(**dict(_strict_fields(raw, cls.FIELDS, "release binding")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "items_sha256": self.items_sha256,
            "sealed_gold_sha256": self.sealed_gold_sha256,
            "stores_sha256": self.stores_sha256,
            "checkpoints_sha256": self.checkpoints_sha256,
            "item_ids": list(self.item_ids),
            "pair_ids": list(self.pair_ids),
            "world_ids": list(self.world_ids),
            "evaluation_cells": [
                cell.to_dict() for cell in self.evaluation_cells
            ],
            "item_count": self.item_count,
            "pair_count": self.pair_count,
            "world_count": self.world_count,
            "evaluation_cell_count": self.evaluation_cell_count,
            "required_families": list(self.required_families),
            "required_strata": list(self.required_strata),
            "required_memory_modes": list(self.required_memory_modes),
            "required_controls": list(self.required_controls),
        }


@dataclass(frozen=True)
class StudyLock:
    record_type: str
    schema_version: int
    preregistration_sha256: str
    release: ReleaseBinding
    checkpoints: tuple[CheckpointApproval, ...]
    validity_receipts: tuple[ReceiptCommitment, ...]

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "record_type",
            "schema_version",
            "preregistration_sha256",
            "release",
            "checkpoints",
            "validity_receipts",
        }
    )

    def __post_init__(self) -> None:
        if self.record_type != STUDY_LOCK_SCHEMA:
            raise ValueError("study lock record_type is invalid")
        _schema(self.schema_version, "study lock")
        preregistration = _hash(
            self.preregistration_sha256,
            "preregistration_sha256",
        )
        if preregistration != FROZEN_PREREGISTRATION_SHA256:
            raise ValueError("study lock preregistration commitment is invalid")
        object.__setattr__(self, "preregistration_sha256", preregistration)
        if not isinstance(self.release, ReleaseBinding):
            raise ValueError("study lock release is invalid")
        checkpoints = tuple(
            checkpoint
            if isinstance(checkpoint, CheckpointApproval)
            else CheckpointApproval.from_dict(checkpoint)
            for checkpoint in self.checkpoints
        )
        checkpoint_keys = tuple(
            (checkpoint.seed, checkpoint.condition_id.value)
            for checkpoint in checkpoints
        )
        if checkpoint_keys != tuple(sorted(checkpoint_keys)):
            raise ValueError("study lock checkpoints are not ordered")
        if len(set(checkpoint_keys)) != len(checkpoint_keys):
            raise ValueError("study lock checkpoints contain duplicate slots")
        required_slots = {
            (seed, condition)
            for seed in range(5)
            for condition in (ConditionId.DENSE, ConditionId.SPLIT90)
        }
        if not required_slots <= {
            (checkpoint.seed, checkpoint.condition_id)
            for checkpoint in checkpoints
        }:
            raise ValueError("study lock omits a protected Dense/Split90 slot")
        object.__setattr__(self, "checkpoints", checkpoints)
        commitments = tuple(
            commitment
            if isinstance(commitment, ReceiptCommitment)
            else ReceiptCommitment.from_dict(commitment)
            for commitment in self.validity_receipts
        )
        receipt_ids = tuple(commitment.receipt_id for commitment in commitments)
        if receipt_ids != REQUIRED_RECEIPTS:
            raise ValueError("study lock validity receipts are not exact")
        for commitment in commitments:
            prefix = commitment.receipt_id.split(":", 1)[0]
            if commitment.kind != prefix:
                raise ValueError("receipt commitment kind disagrees with id")
        object.__setattr__(self, "validity_receipts", commitments)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StudyLock":
        value = _strict_fields(raw, cls.FIELDS, "study lock")
        return cls(
            record_type=value["record_type"],
            schema_version=value["schema_version"],
            preregistration_sha256=value["preregistration_sha256"],
            release=ReleaseBinding.from_dict(value["release"]),
            checkpoints=value["checkpoints"],
            validity_receipts=value["validity_receipts"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "preregistration_sha256": self.preregistration_sha256,
            "release": self.release.to_dict(),
            "checkpoints": [
                checkpoint.to_dict() for checkpoint in self.checkpoints
            ],
            "validity_receipts": [
                commitment.to_dict()
                for commitment in self.validity_receipts
            ],
        }


@dataclass(frozen=True)
class ValidityReceipt:
    receipt_id: str
    kind: str
    state: ReceiptState
    evidence_sha256: str

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"receipt_id", "kind", "state", "evidence_sha256"}
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receipt_id",
            _string(self.receipt_id, "receipt_id"),
        )
        if self.kind not in {"gate", "guardrail", "control"}:
            raise ValueError("validity receipt kind is invalid")
        try:
            state = ReceiptState(self.state)
        except (TypeError, ValueError) as exc:
            raise ValueError("validity receipt state is invalid") from exc
        object.__setattr__(self, "state", state)
        object.__setattr__(
            self,
            "evidence_sha256",
            _hash(self.evidence_sha256, "evidence_sha256"),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ValidityReceipt":
        return cls(**dict(_strict_fields(raw, cls.FIELDS, "validity receipt")))

    def to_dict(self) -> dict[str, str]:
        return {
            "receipt_id": self.receipt_id,
            "kind": self.kind,
            "state": self.state.value,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True)
class ValidityEvidence:
    record_type: str
    schema_version: int
    study_lock_sha256: str
    preregistration_sha256: str
    receipts: tuple[ValidityReceipt, ...]

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "record_type",
            "schema_version",
            "study_lock_sha256",
            "preregistration_sha256",
            "receipts",
        }
    )

    def __post_init__(self) -> None:
        if self.record_type != VALIDITY_EVIDENCE_SCHEMA:
            raise ValueError("validity evidence record_type is invalid")
        _schema(self.schema_version, "validity evidence")
        object.__setattr__(
            self,
            "study_lock_sha256",
            _hash(self.study_lock_sha256, "study_lock_sha256"),
        )
        preregistration = _hash(
            self.preregistration_sha256,
            "preregistration_sha256",
        )
        if preregistration != FROZEN_PREREGISTRATION_SHA256:
            raise ValueError("validity evidence preregistration is invalid")
        object.__setattr__(self, "preregistration_sha256", preregistration)
        receipts = tuple(
            receipt
            if isinstance(receipt, ValidityReceipt)
            else ValidityReceipt.from_dict(receipt)
            for receipt in self.receipts
        )
        ids = tuple(receipt.receipt_id for receipt in receipts)
        expected_order = {
            receipt_id: index
            for index, receipt_id in enumerate(REQUIRED_RECEIPTS)
        }
        if (
            len(set(ids)) != len(ids)
            or any(receipt_id not in expected_order for receipt_id in ids)
            or ids
            != tuple(
                sorted(ids, key=lambda receipt_id: expected_order[receipt_id])
            )
        ):
            raise ValueError("validity receipts are duplicated, unknown, or unordered")
        for receipt in receipts:
            if receipt.kind != receipt.receipt_id.split(":", 1)[0]:
                raise ValueError("validity receipt kind disagrees with id")
        object.__setattr__(self, "receipts", receipts)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ValidityEvidence":
        value = _strict_fields(raw, cls.FIELDS, "validity evidence")
        return cls(
            record_type=value["record_type"],
            schema_version=value["schema_version"],
            study_lock_sha256=value["study_lock_sha256"],
            preregistration_sha256=value["preregistration_sha256"],
            receipts=value["receipts"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "study_lock_sha256": self.study_lock_sha256,
            "preregistration_sha256": self.preregistration_sha256,
            "receipts": [receipt.to_dict() for receipt in self.receipts],
        }


@dataclass(frozen=True)
class ReadinessResult:
    complete: bool
    valid: bool


def evaluate_readiness(
    lock: StudyLock,
    evidence: ValidityEvidence,
) -> ReadinessResult:
    if evidence.study_lock_sha256 != canonical_sha256(lock.to_dict()):
        raise ValueError("validity evidence study-lock commitment mismatch")
    if evidence.preregistration_sha256 != lock.preregistration_sha256:
        raise ValueError("validity preregistration commitment mismatch")
    commitments = {
        commitment.receipt_id: commitment
        for commitment in lock.validity_receipts
    }
    observed = {receipt.receipt_id: receipt for receipt in evidence.receipts}
    for receipt_id, receipt in observed.items():
        if canonical_sha256(receipt.to_dict()) != commitments[
            receipt_id
        ].receipt_sha256:
            raise ValueError(
                f"validity receipt {receipt_id} disagrees with commitment"
            )
    failed = any(
        commitment.state is ReceiptState.FAILED
        for commitment in commitments.values()
    ) or any(
        receipt.state is ReceiptState.FAILED for receipt in observed.values()
    )
    complete = (
        set(observed) == set(REQUIRED_RECEIPTS)
        and all(
            receipt.state is ReceiptState.PASSED
            for receipt in observed.values()
        )
    )
    return ReadinessResult(complete=complete, valid=not failed)
