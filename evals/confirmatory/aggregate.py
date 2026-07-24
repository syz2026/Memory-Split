"""Deterministic planning for the 100 protected snapshot evaluations.

This module intentionally stops at planning and per-snapshot binding. Cohort
aggregation and confirmatory statistics belong to a later stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Any, ClassVar

from evals.confirmatory.contracts import (
    STUDY_CONTRACT_VERSION,
    STUDY_TARGETS_PER_UPDATE,
    StudyArm,
    canonical_json_bytes,
    canonical_sha256,
    validate_study_record_identity,
)
from evals.confirmatory.study_lock import (
    EXPECTED_STUDY_SLOTS_V3,
    ProviderSelectionBinding,
    StudyLockV3,
    StudySnapshotBinding,
)


RUN_BINDING_SCHEMA_V3 = "memorysplit.confirmatory.run-binding.v3"
CHECKPOINT_FILE_NAME = "checkpoint.pt"
STUDY_LOCK_FILE_NAME = "study-lock.json"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _strict_fields(
    raw: object,
    expected: frozenset[str],
    name: str,
) -> Mapping[str, Any]:
    if (
        not isinstance(raw, Mapping)
        or any(not isinstance(key, str) for key in raw)
        or set(raw) != expected
    ):
        raise ValueError(f"{name} fields are not exact")
    return raw


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _run_id(
    seed: int,
    arm: StudyArm,
    optimizer_step: int,
    profile_id: str,
) -> str:
    return (
        f"memorysplit-v3-{profile_id}-s{seed:02d}-{arm.value}-"
        f"step{optimizer_step:05d}"
    )


def _output_name(run_id: str) -> str:
    return f"snapshot-evaluation-{run_id}"


@dataclass(frozen=True)
class RunBindingV3:
    """Closed, step-aware identity for one isolated v3 evaluation."""

    record_type: str
    schema_version: int
    run_id: str
    checkpoint_path: str
    study_lock_path: str
    seed: int
    arm: StudyArm
    optimizer_step: int
    checkpoint_sha256: str
    checkpoint_s3_object_key: str
    checkpoint_s3_version_id: str
    checkpoint_receipt_sha256: str
    checkpoint_receipt_s3_object_key: str
    checkpoint_receipt_s3_version_id: str
    sealed_evaluation_release_sha256: str
    study_lock_sha256: str
    provider_selection_s3_key: str
    provider_selection_sha256: str
    provider_selection_s3_version_id: str
    hardware_amendment_sha256: str
    selected_provider: str
    evaluator_profile_id: str
    evaluator_profile_sha256: str
    evaluator_runtime_lock_sha256: str
    evaluator_qualification_evidence_sha256: str
    output_id: str

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "record_type",
            "schema_version",
            "run_id",
            "checkpoint_path",
            "study_lock_path",
            "seed",
            "arm",
            "optimizer_step",
            "checkpoint_sha256",
            "checkpoint_s3_object_key",
            "checkpoint_s3_version_id",
            "checkpoint_receipt_sha256",
            "checkpoint_receipt_s3_object_key",
            "checkpoint_receipt_s3_version_id",
            "sealed_evaluation_release_sha256",
            "study_lock_sha256",
            "provider_selection_s3_key",
            "provider_selection_sha256",
            "provider_selection_s3_version_id",
            "hardware_amendment_sha256",
            "selected_provider",
            "evaluator_profile_id",
            "evaluator_profile_sha256",
            "evaluator_runtime_lock_sha256",
            "evaluator_qualification_evidence_sha256",
            "output_id",
        }
    )

    def __post_init__(self) -> None:
        if self.record_type != RUN_BINDING_SCHEMA_V3:
            raise ValueError("v3 run binding record_type is invalid")
        if (
            type(self.schema_version) is not int
            or self.schema_version != STUDY_CONTRACT_VERSION
        ):
            raise ValueError("v3 run binding schema_version is invalid")
        raw_tokens = (
            self.optimizer_step * STUDY_TARGETS_PER_UPDATE
            if type(self.optimizer_step) is int
            else self.optimizer_step
        )
        seed, arm, _condition, optimizer_step, _raw_tokens = (
            validate_study_record_identity(
                seed=self.seed,
                arm=self.arm,
                condition_id=self.arm,
                optimizer_step=self.optimizer_step,
                raw_token_count=raw_tokens,
            )
        )
        expected_run_id = _run_id(
            seed,
            arm,
            optimizer_step,
            self.evaluator_profile_id,
        )
        if self.run_id != expected_run_id:
            raise ValueError("v3 run binding run identity is invalid")
        if self.checkpoint_path != CHECKPOINT_FILE_NAME:
            raise ValueError("v3 checkpoint path is not the fixed local name")
        if self.study_lock_path != STUDY_LOCK_FILE_NAME:
            raise ValueError("v3 study-lock path is not the fixed local name")

        snapshot = StudySnapshotBinding(
            seed=seed,
            arm=arm,
            optimizer_step=optimizer_step,
            checkpoint_sha256=self.checkpoint_sha256,
            s3_object_key=self.checkpoint_s3_object_key,
            s3_version_id=self.checkpoint_s3_version_id,
            checkpoint_receipt_sha256=self.checkpoint_receipt_sha256,
            checkpoint_receipt_s3_object_key=(
                self.checkpoint_receipt_s3_object_key
            ),
            checkpoint_receipt_s3_version_id=(
                self.checkpoint_receipt_s3_version_id
            ),
            provider_selection_sha256=self.provider_selection_sha256,
            provider_selection_s3_version_id=(
                self.provider_selection_s3_version_id
            ),
        )
        release = _sha256(
            self.sealed_evaluation_release_sha256,
            "sealed evaluation release SHA-256",
        )
        lock = _sha256(self.study_lock_sha256, "study-lock SHA-256")
        selection = ProviderSelectionBinding(
            cohort_id="memorysplit-confirmatory-v3-360m-n10-aws",
            provider_selection_s3_key=self.provider_selection_s3_key,
            provider_selection_sha256=snapshot.provider_selection_sha256,
            provider_selection_s3_version_id=(
                snapshot.provider_selection_s3_version_id
            ),
            hardware_amendment_sha256=self.hardware_amendment_sha256,
            selected_provider=self.selected_provider,
            profile_id=self.evaluator_profile_id,
            profile_sha256=self.evaluator_profile_sha256,
            runtime_lock_sha256=self.evaluator_runtime_lock_sha256,
            qualification_evidence_sha256=(
                self.evaluator_qualification_evidence_sha256
            ),
        )
        expected_output = _output_name(expected_run_id)
        if self.output_id != expected_output:
            raise ValueError("v3 run binding output identity is invalid")

        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "optimizer_step", optimizer_step)
        object.__setattr__(
            self,
            "checkpoint_sha256",
            snapshot.checkpoint_sha256,
        )
        object.__setattr__(
            self,
            "checkpoint_s3_object_key",
            snapshot.s3_object_key,
        )
        object.__setattr__(
            self,
            "checkpoint_s3_version_id",
            snapshot.s3_version_id,
        )
        object.__setattr__(
            self,
            "checkpoint_receipt_sha256",
            snapshot.checkpoint_receipt_sha256,
        )
        object.__setattr__(
            self,
            "checkpoint_receipt_s3_object_key",
            snapshot.checkpoint_receipt_s3_object_key,
        )
        object.__setattr__(
            self,
            "checkpoint_receipt_s3_version_id",
            snapshot.checkpoint_receipt_s3_version_id,
        )
        object.__setattr__(
            self,
            "sealed_evaluation_release_sha256",
            release,
        )
        object.__setattr__(self, "study_lock_sha256", lock)
        object.__setattr__(
            self,
            "provider_selection_s3_key",
            selection.provider_selection_s3_key,
        )
        object.__setattr__(
            self,
            "provider_selection_sha256",
            selection.provider_selection_sha256,
        )
        object.__setattr__(
            self,
            "provider_selection_s3_version_id",
            selection.provider_selection_s3_version_id,
        )
        object.__setattr__(
            self,
            "hardware_amendment_sha256",
            selection.hardware_amendment_sha256,
        )
        object.__setattr__(
            self,
            "selected_provider",
            selection.selected_provider,
        )
        object.__setattr__(
            self,
            "evaluator_profile_id",
            selection.profile_id,
        )
        object.__setattr__(
            self,
            "evaluator_profile_sha256",
            selection.profile_sha256,
        )
        object.__setattr__(
            self,
            "evaluator_runtime_lock_sha256",
            selection.runtime_lock_sha256,
        )
        object.__setattr__(
            self,
            "evaluator_qualification_evidence_sha256",
            selection.qualification_evidence_sha256,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RunBindingV3":
        value = _strict_fields(raw, cls.FIELDS, "v3 run binding")
        return cls(**dict(value))

    @property
    def output_name(self) -> str:
        return self.output_id

    @property
    def condition_id(self) -> str:
        return self.arm.value

    @property
    def raw_token_count(self) -> int:
        return self.optimizer_step * STUDY_TARGETS_PER_UPDATE

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "checkpoint_path": self.checkpoint_path,
            "study_lock_path": self.study_lock_path,
            "seed": self.seed,
            "arm": self.arm.value,
            "optimizer_step": self.optimizer_step,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_s3_object_key": self.checkpoint_s3_object_key,
            "checkpoint_s3_version_id": self.checkpoint_s3_version_id,
            "checkpoint_receipt_sha256": self.checkpoint_receipt_sha256,
            "checkpoint_receipt_s3_object_key": (
                self.checkpoint_receipt_s3_object_key
            ),
            "checkpoint_receipt_s3_version_id": (
                self.checkpoint_receipt_s3_version_id
            ),
            "sealed_evaluation_release_sha256": (
                self.sealed_evaluation_release_sha256
            ),
            "study_lock_sha256": self.study_lock_sha256,
            "provider_selection_s3_key": self.provider_selection_s3_key,
            "provider_selection_sha256": self.provider_selection_sha256,
            "provider_selection_s3_version_id": (
                self.provider_selection_s3_version_id
            ),
            "hardware_amendment_sha256": self.hardware_amendment_sha256,
            "selected_provider": self.selected_provider,
            "evaluator_profile_id": self.evaluator_profile_id,
            "evaluator_profile_sha256": self.evaluator_profile_sha256,
            "evaluator_runtime_lock_sha256": (
                self.evaluator_runtime_lock_sha256
            ),
            "evaluator_qualification_evidence_sha256": (
                self.evaluator_qualification_evidence_sha256
            ),
            "output_id": self.output_id,
        }


@dataclass(frozen=True)
class SnapshotEvaluationPlan:
    """One immutable slot plus its canonical run.json bytes."""

    slot_index: int
    seed: int
    arm: StudyArm
    optimizer_step: int
    run_id: str
    output_name: str
    binding: RunBindingV3
    run_json_bytes: bytes

    def __post_init__(self) -> None:
        if type(self.slot_index) is not int or not 0 <= self.slot_index < 100:
            raise ValueError("snapshot plan slot index is invalid")
        if not isinstance(self.binding, RunBindingV3):
            raise TypeError("snapshot plan binding must be RunBindingV3")
        expected = (
            self.binding.seed,
            self.binding.arm,
            self.binding.optimizer_step,
            self.binding.run_id,
            self.binding.output_name,
        )
        observed = (
            self.seed,
            StudyArm(self.arm),
            self.optimizer_step,
            self.run_id,
            self.output_name,
        )
        if observed != expected:
            raise ValueError("snapshot plan identity disagrees with its binding")
        if self.run_json_bytes != canonical_json_bytes(self.binding.to_dict()):
            raise ValueError("snapshot plan run binding bytes are not canonical")
        object.__setattr__(self, "arm", self.binding.arm)


def _build_snapshot_evaluation_plans(
    *,
    study_lock: StudyLockV3,
    study_lock_sha256: str,
) -> tuple[SnapshotEvaluationPlan, ...]:
    if not isinstance(study_lock, StudyLockV3):
        raise TypeError("snapshot planning requires StudyLockV3")
    claimed_lock = _sha256(study_lock_sha256, "study-lock SHA-256")
    if canonical_sha256(study_lock.to_dict()) != claimed_lock:
        raise ValueError("study-lock hash commitment does not match its bytes")
    selection = study_lock.provider_selection
    plans = []
    for slot_index, snapshot in enumerate(study_lock.snapshots):
        run_id = _run_id(
            snapshot.seed,
            snapshot.arm,
            snapshot.optimizer_step,
            selection.profile_id,
        )
        output_name = _output_name(run_id)
        binding = RunBindingV3(
            record_type=RUN_BINDING_SCHEMA_V3,
            schema_version=STUDY_CONTRACT_VERSION,
            run_id=run_id,
            checkpoint_path=CHECKPOINT_FILE_NAME,
            study_lock_path=STUDY_LOCK_FILE_NAME,
            seed=snapshot.seed,
            arm=snapshot.arm,
            optimizer_step=snapshot.optimizer_step,
            checkpoint_sha256=snapshot.checkpoint_sha256,
            checkpoint_s3_object_key=snapshot.s3_object_key,
            checkpoint_s3_version_id=snapshot.s3_version_id,
            checkpoint_receipt_sha256=(
                snapshot.checkpoint_receipt_sha256
            ),
            checkpoint_receipt_s3_object_key=(
                snapshot.checkpoint_receipt_s3_object_key
            ),
            checkpoint_receipt_s3_version_id=(
                snapshot.checkpoint_receipt_s3_version_id
            ),
            sealed_evaluation_release_sha256=(
                study_lock.sealed_evaluation_release_sha256
            ),
            study_lock_sha256=claimed_lock,
            provider_selection_s3_key=selection.provider_selection_s3_key,
            provider_selection_sha256=(
                selection.provider_selection_sha256
            ),
            provider_selection_s3_version_id=(
                selection.provider_selection_s3_version_id
            ),
            hardware_amendment_sha256=(
                selection.hardware_amendment_sha256
            ),
            selected_provider=selection.selected_provider,
            evaluator_profile_id=selection.profile_id,
            evaluator_profile_sha256=selection.profile_sha256,
            evaluator_runtime_lock_sha256=selection.runtime_lock_sha256,
            evaluator_qualification_evidence_sha256=(
                selection.qualification_evidence_sha256
            ),
            output_id=output_name,
        )
        plans.append(
            SnapshotEvaluationPlan(
                slot_index=slot_index,
                seed=snapshot.seed,
                arm=snapshot.arm,
                optimizer_step=snapshot.optimizer_step,
                run_id=run_id,
                output_name=output_name,
                binding=binding,
                run_json_bytes=canonical_json_bytes(binding.to_dict()),
            )
        )
    return tuple(plans)


def plan_snapshot_evaluations(
    *,
    study_lock: StudyLockV3,
    study_lock_sha256: str,
) -> tuple[SnapshotEvaluationPlan, ...]:
    """Enumerate the exact 100 slots in frozen seed/arm/step order."""

    plans = _build_snapshot_evaluation_plans(
        study_lock=study_lock,
        study_lock_sha256=study_lock_sha256,
    )
    return validate_snapshot_evaluation_plans(
        plans,
        study_lock=study_lock,
        study_lock_sha256=study_lock_sha256,
    )


def validate_snapshot_evaluation_plans(
    plans: Sequence[SnapshotEvaluationPlan],
    *,
    study_lock: StudyLockV3,
    study_lock_sha256: str,
) -> tuple[SnapshotEvaluationPlan, ...]:
    """Reject every missing, extra, reordered, replaced, or aliased slot."""

    if (
        isinstance(plans, (str, bytes))
        or not isinstance(plans, Sequence)
        or any(not isinstance(plan, SnapshotEvaluationPlan) for plan in plans)
    ):
        raise ValueError("snapshot plan registry must be an ordered sequence")
    values = tuple(plans)
    if len(values) != len(EXPECTED_STUDY_SLOTS_V3):
        raise ValueError("snapshot plan requires exactly 100 slots")
    slots = tuple(
        (plan.seed, plan.arm, plan.optimizer_step) for plan in values
    )
    if slots != EXPECTED_STUDY_SLOTS_V3:
        raise ValueError("snapshot plan slots are missing, reordered, or aliased")
    if tuple(plan.slot_index for plan in values) != tuple(range(100)):
        raise ValueError("snapshot plan slot indices are not exact")
    for identity in (
        tuple(plan.run_id for plan in values),
        tuple(plan.output_name for plan in values),
        tuple(plan.run_json_bytes for plan in values),
    ):
        if len(set(identity)) != len(identity):
            raise ValueError("snapshot plan contains an aliased identity")
    expected = _build_snapshot_evaluation_plans(
        study_lock=study_lock,
        study_lock_sha256=study_lock_sha256,
    )
    if values != expected:
        raise ValueError("snapshot plan differs from the exact canonical plan")
    return values


build_snapshot_evaluation_plans = plan_snapshot_evaluations
validate_snapshot_evaluation_plan = validate_snapshot_evaluation_plans


__all__ = [
    "CHECKPOINT_FILE_NAME",
    "RUN_BINDING_SCHEMA_V3",
    "RunBindingV3",
    "STUDY_LOCK_FILE_NAME",
    "SnapshotEvaluationPlan",
    "build_snapshot_evaluation_plans",
    "plan_snapshot_evaluations",
    "validate_snapshot_evaluation_plan",
    "validate_snapshot_evaluation_plans",
]
