"""Deterministic planning, authentication, and v3 cohort aggregation.

Planning stops at per-snapshot binding. Cohort aggregation authenticates the
exact 100 receipts-proven published outputs from source bytes and derives
only replayable frozen confirmatory conclusions; the report's canonical
bytes are its authoritative content address, and publication reuses the
sealing no-replace/quarantine authority.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, ClassVar

from evals.confirmatory.contracts import (
    STUDY_CONTRACT_VERSION,
    STUDY_TARGETS_PER_UPDATE,
    Control,
    ItemRecord,
    MemoryMode,
    ReasoningFamily,
    SealedGoldRecord,
    StoreRecord,
    Stratum,
    StudyArm,
    Twin,
    canonical_json_bytes,
    canonical_sha256,
    validate_study_record_identity,
)
from evals.confirmatory.inference import (
    V3_BOOTSTRAP_CONFIDENCE,
    V3_BOOTSTRAP_DRAWS,
    V3_BOOTSTRAP_RNG_SEED,
    V3_EQUIVALENCE_MARGIN,
    V3_SIGN_ASSIGNMENTS,
    V3_SNAPSHOT_STEPS,
    PairedObservation,
    v3_exact_sign_flip_test,
    v3_practical_equivalence_bounds,
    v3_right_step_aulc,
)
from evals.confirmatory.metrics import (
    PRIMARY_CELL_IDS,
    PRIMARY_CELLS,
    Rate,
    StudyMetricsRecord,
    StudyOutcomeRecord,
)
from evals.confirmatory.sealing import (
    PATH_AUTHORITY,
    SealingError,
    _DirectorySnapshot,
    _PinnedFile,
    _StagedRelease,
    _assert_directory_entry,
    _assert_directory_path,
    _assert_directory_snapshot,
    _assert_final_directory_binding,
    _assert_pinned_file,
    _capture_directory_snapshot,
    _close_staged,
    _lock_output,
    _make_staging,
    _open_directory,
    _quarantine_preserve_directory,
    _quarantine_staged_directory,
    _read_descriptor,
    _regular_file_state,
    _reject_existing_release,
    _reject_quarantine_blockers,
    _rename_noreplace_at,
    _validated_release_from_preregistration_sha256,
    _write_all,
)
from evals.confirmatory.solver import registered_solver, verify_proof_and_answer
from evals.confirmatory.status import classify_status
from evals.confirmatory.study_lock import (
    EXPECTED_STUDY_SLOTS_V3,
    FROZEN_PREREGISTRATION_SHA256_V3,
    REQUIRED_CONTROL_IDS,
    REQUIRED_FAMILIES,
    REQUIRED_STRATA,
    ProviderSelectionBinding,
    StudyLockV3,
    StudySnapshotBinding,
    _git_sha1,
    _positive_bytes,
    _receipt_object_uri,
    _s3_version_id,
)
from msctl.aws_contracts import (
    SEEDS,
    collection_receipt_key,
    run_receipt_key,
)
from msctl.aws_lifecycle import ProviderLifecycleBinding
from msctl.fsutil import open_directory


RUN_BINDING_SCHEMA_V3 = "memorysplit.confirmatory.run-binding.v3"
STUDY_LOCK_FILE_NAME = "study-lock.json"
EVALUATOR_PROFILE_FILE_NAME = "evaluator-profile.json"
EVALUATOR_RUNTIME_LOCK_FILE_NAME = "evaluator-runtime-lock.json"
EVALUATOR_ENVIRONMENT_RECEIPT_FILE_NAME = (
    "evaluator-environment-receipt.json"
)
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


def snapshot_path(optimizer_step: int) -> str:
    return f"snapshots/step{optimizer_step:07d}.pt"


def _run_lifecycle_binding(binding: "RunBindingV3") -> ProviderLifecycleBinding:
    """Reconstruct the training provider lifecycle bound by one run binding."""

    try:
        return ProviderLifecycleBinding(
            cohort_id="memorysplit-confirmatory-v3-360m-n10-aws",
            provider=binding.selected_provider,
            profile_id=binding.evaluator_profile_id,
            profile_sha256=binding.evaluator_profile_sha256,
            hardware_amendment_sha256=binding.hardware_amendment_sha256,
            provider_selection_sha256=binding.provider_selection_sha256,
            provider_selection_version_id=(
                binding.provider_selection_s3_version_id
            ),
            runtime_lock_sha256=binding.evaluator_runtime_lock_sha256,
            runtime_sbom_sha256=binding.runtime_sbom_sha256,
            qualification_evidence_sha256=(
                binding.evaluator_qualification_evidence_sha256
            ),
            qualification_environment_receipt_sha256=(
                binding.evaluator_environment_receipt_sha256
            ),
            qualification_canary_receipt_sha256=(
                binding.evaluator_canary_receipt_sha256
            ),
            qualification_approval_receipt_sha256=(
                binding.evaluator_approval_receipt_sha256
            ),
            qualification_approval_public_key_sha256=(
                binding.evaluator_approval_public_key_sha256
            ),
            objective_controls_contract_sha256=(
                binding.objective_controls_contract_sha256
            ),
            account_id=binding.account_id,
            instance_id=binding.instance_id,
            boot_id=binding.boot_id,
            region=binding.region,
            availability_zone=binding.availability_zone,
            purchase_model=binding.purchase_model,
            seed=binding.seed,
            arms=("dense", "split90"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "v3 run binding cannot reconstruct one authenticated provider "
            "lifecycle"
        ) from exc


@dataclass(frozen=True)
class RunBindingV3:
    """Closed, step-aware identity for one isolated v3 evaluation."""

    record_type: str
    schema_version: int
    run_id: str
    snapshot_path: str
    study_lock_path: str
    seed: int
    arm: StudyArm
    optimizer_step: int
    snapshot_sha256: str
    snapshot_s3_object_key: str
    snapshot_s3_version_id: str
    checkpoint_receipt_sha256: str
    checkpoint_receipt_s3_object_key: str
    checkpoint_receipt_s3_version_id: str
    snapshot_version: int
    training_run_id: str
    config_fingerprint: str
    training_config_sha256: str
    model_config_sha256: str
    model_identity: str
    data_provenance_sha256: str
    data_receipt_sha256: str
    data_build_id: str
    ordered_stream_sha256: str
    world_size: int
    tokens_per_step: int
    sealed_evaluation_release_sha256: str
    study_lock_sha256: str
    provider_selection_s3_key: str
    provider_selection_sha256: str
    provider_selection_s3_version_id: str
    hardware_amendment_sha256: str
    selected_provider: str
    evaluator_profile_id: str
    evaluator_profile_path: str
    evaluator_profile_sha256: str
    evaluator_runtime_lock_path: str
    evaluator_runtime_lock_sha256: str
    evaluator_qualification_evidence_sha256: str
    evaluator_environment_receipt_path: str
    evaluator_environment_receipt_sha256: str
    evaluator_canary_receipt_sha256: str
    evaluator_approval_receipt_sha256: str
    evaluator_approval_public_key_sha256: str
    account_id: str
    availability_zone: str
    boot_id: str
    instance_id: str
    region: str
    purchase_model: str
    runtime_sbom_sha256: str
    objective_controls_contract_sha256: str
    source_commit: str
    source_tree: str
    operational_config_sha256: str
    finalization_receipt_sha256: str
    finalization_receipt_s3_uri: str
    finalization_receipt_bytes: int
    finalization_receipt_s3_version_id: str
    collection_receipt_sha256: str
    collection_receipt_s3_uri: str
    collection_receipt_s3_version_id: str
    output_id: str

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "record_type",
            "schema_version",
            "run_id",
            "snapshot_path",
            "study_lock_path",
            "seed",
            "arm",
            "optimizer_step",
            "snapshot_sha256",
            "snapshot_s3_object_key",
            "snapshot_s3_version_id",
            "checkpoint_receipt_sha256",
            "checkpoint_receipt_s3_object_key",
            "checkpoint_receipt_s3_version_id",
            "snapshot_version",
            "training_run_id",
            "config_fingerprint",
            "training_config_sha256",
            "model_config_sha256",
            "model_identity",
            "data_provenance_sha256",
            "data_receipt_sha256",
            "data_build_id",
            "ordered_stream_sha256",
            "world_size",
            "tokens_per_step",
            "sealed_evaluation_release_sha256",
            "study_lock_sha256",
            "provider_selection_s3_key",
            "provider_selection_sha256",
            "provider_selection_s3_version_id",
            "hardware_amendment_sha256",
            "selected_provider",
            "evaluator_profile_id",
            "evaluator_profile_path",
            "evaluator_profile_sha256",
            "evaluator_runtime_lock_path",
            "evaluator_runtime_lock_sha256",
            "evaluator_qualification_evidence_sha256",
            "evaluator_environment_receipt_path",
            "evaluator_environment_receipt_sha256",
            "evaluator_canary_receipt_sha256",
            "evaluator_approval_receipt_sha256",
            "evaluator_approval_public_key_sha256",
            "account_id",
            "availability_zone",
            "boot_id",
            "instance_id",
            "region",
            "purchase_model",
            "runtime_sbom_sha256",
            "objective_controls_contract_sha256",
            "source_commit",
            "source_tree",
            "operational_config_sha256",
            "finalization_receipt_sha256",
            "finalization_receipt_s3_uri",
            "finalization_receipt_bytes",
            "finalization_receipt_s3_version_id",
            "collection_receipt_sha256",
            "collection_receipt_s3_uri",
            "collection_receipt_s3_version_id",
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
        if self.snapshot_path != snapshot_path(optimizer_step):
            raise ValueError(
                "v3 snapshot path is not the canonical step snapshot"
            )
        if self.study_lock_path != STUDY_LOCK_FILE_NAME:
            raise ValueError("v3 study-lock path is not the fixed local name")
        if self.evaluator_profile_path != EVALUATOR_PROFILE_FILE_NAME:
            raise ValueError("v3 evaluator profile path is not fixed")
        if (
            self.evaluator_runtime_lock_path
            != EVALUATOR_RUNTIME_LOCK_FILE_NAME
        ):
            raise ValueError("v3 evaluator runtime-lock path is not fixed")
        if (
            self.evaluator_environment_receipt_path
            != EVALUATOR_ENVIRONMENT_RECEIPT_FILE_NAME
        ):
            raise ValueError(
                "v3 evaluator environment-receipt path is not fixed"
            )

        snapshot = StudySnapshotBinding(
            seed=seed,
            arm=arm,
            optimizer_step=optimizer_step,
            checkpoint_sha256=self.snapshot_sha256,
            s3_object_key=self.snapshot_s3_object_key,
            s3_version_id=self.snapshot_s3_version_id,
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
            snapshot_version=self.snapshot_version,
            training_run_id=self.training_run_id,
            config_fingerprint=self.config_fingerprint,
            training_config_sha256=self.training_config_sha256,
            model_config_sha256=self.model_config_sha256,
            model_identity=self.model_identity,
            data_provenance_sha256=self.data_provenance_sha256,
            data_receipt_sha256=self.data_receipt_sha256,
            data_build_id=self.data_build_id,
            ordered_stream_sha256=self.ordered_stream_sha256,
            world_size=self.world_size,
            tokens_per_step=self.tokens_per_step,
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
            environment_receipt_sha256=(
                self.evaluator_environment_receipt_sha256
            ),
            canary_receipt_sha256=(
                self.evaluator_canary_receipt_sha256
            ),
            approval_receipt_sha256=(
                self.evaluator_approval_receipt_sha256
            ),
            approval_public_key_sha256=(
                self.evaluator_approval_public_key_sha256
            ),
        )
        expected_output = _output_name(expected_run_id)
        if self.output_id != expected_output:
            raise ValueError("v3 run binding output identity is invalid")

        _run_lifecycle_binding(self)
        _git_sha1(
            self.source_commit,
            "v3 run binding source_commit",
        )
        _git_sha1(
            self.source_tree,
            "v3 run binding source_tree",
        )
        _sha256(
            self.operational_config_sha256,
            "v3 run binding operational config SHA-256",
        )
        finalization_sha256 = _sha256(
            self.finalization_receipt_sha256,
            "v3 run binding finalization receipt SHA-256",
        )
        finalization_root = _receipt_object_uri(
            self.finalization_receipt_s3_uri,
            expected_key=run_receipt_key(seed, finalization_sha256),
            name="v3 run binding finalization receipt",
        )
        _positive_bytes(
            self.finalization_receipt_bytes,
            "v3 run binding finalization receipt",
        )
        _s3_version_id(
            self.finalization_receipt_s3_version_id,
            "v3 run binding finalization receipt S3 version ID",
        )
        collection_sha256 = _sha256(
            self.collection_receipt_sha256,
            "v3 run binding collection receipt SHA-256",
        )
        collection_root = _receipt_object_uri(
            self.collection_receipt_s3_uri,
            expected_key=collection_receipt_key(seed, collection_sha256),
            name="v3 run binding collection receipt",
        )
        _s3_version_id(
            self.collection_receipt_s3_version_id,
            "v3 run binding collection receipt S3 version ID",
        )
        if finalization_root != collection_root:
            raise ValueError(
                "v3 run binding receipts do not share one durable object root"
            )

        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "optimizer_step", optimizer_step)
        object.__setattr__(
            self,
            "snapshot_sha256",
            snapshot.checkpoint_sha256,
        )
        object.__setattr__(
            self,
            "snapshot_s3_object_key",
            snapshot.s3_object_key,
        )
        object.__setattr__(
            self,
            "snapshot_s3_version_id",
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
            "training_run_id",
            snapshot.training_run_id,
        )
        object.__setattr__(
            self,
            "config_fingerprint",
            snapshot.config_fingerprint,
        )
        object.__setattr__(
            self,
            "training_config_sha256",
            snapshot.training_config_sha256,
        )
        object.__setattr__(
            self,
            "model_config_sha256",
            snapshot.model_config_sha256,
        )
        object.__setattr__(
            self,
            "model_identity",
            snapshot.model_identity,
        )
        object.__setattr__(
            self,
            "data_provenance_sha256",
            snapshot.data_provenance_sha256,
        )
        object.__setattr__(
            self,
            "data_receipt_sha256",
            snapshot.data_receipt_sha256,
        )
        object.__setattr__(
            self,
            "data_build_id",
            snapshot.data_build_id,
        )
        object.__setattr__(
            self,
            "ordered_stream_sha256",
            snapshot.ordered_stream_sha256,
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
        object.__setattr__(
            self,
            "evaluator_environment_receipt_sha256",
            selection.environment_receipt_sha256,
        )
        object.__setattr__(
            self,
            "evaluator_canary_receipt_sha256",
            selection.canary_receipt_sha256,
        )
        object.__setattr__(
            self,
            "evaluator_approval_receipt_sha256",
            selection.approval_receipt_sha256,
        )
        object.__setattr__(
            self,
            "evaluator_approval_public_key_sha256",
            selection.approval_public_key_sha256,
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

    def lifecycle_binding(self) -> ProviderLifecycleBinding:
        """Reconstruct this binding's training provider lifecycle authority."""

        return _run_lifecycle_binding(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "snapshot_path": self.snapshot_path,
            "study_lock_path": self.study_lock_path,
            "seed": self.seed,
            "arm": self.arm.value,
            "optimizer_step": self.optimizer_step,
            "snapshot_sha256": self.snapshot_sha256,
            "snapshot_s3_object_key": self.snapshot_s3_object_key,
            "snapshot_s3_version_id": self.snapshot_s3_version_id,
            "checkpoint_receipt_sha256": self.checkpoint_receipt_sha256,
            "checkpoint_receipt_s3_object_key": (
                self.checkpoint_receipt_s3_object_key
            ),
            "checkpoint_receipt_s3_version_id": (
                self.checkpoint_receipt_s3_version_id
            ),
            "snapshot_version": self.snapshot_version,
            "training_run_id": self.training_run_id,
            "config_fingerprint": self.config_fingerprint,
            "training_config_sha256": self.training_config_sha256,
            "model_config_sha256": self.model_config_sha256,
            "model_identity": self.model_identity,
            "data_provenance_sha256": self.data_provenance_sha256,
            "data_receipt_sha256": self.data_receipt_sha256,
            "data_build_id": self.data_build_id,
            "ordered_stream_sha256": self.ordered_stream_sha256,
            "world_size": self.world_size,
            "tokens_per_step": self.tokens_per_step,
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
            "evaluator_profile_path": self.evaluator_profile_path,
            "evaluator_profile_sha256": self.evaluator_profile_sha256,
            "evaluator_runtime_lock_path": (
                self.evaluator_runtime_lock_path
            ),
            "evaluator_runtime_lock_sha256": (
                self.evaluator_runtime_lock_sha256
            ),
            "evaluator_qualification_evidence_sha256": (
                self.evaluator_qualification_evidence_sha256
            ),
            "evaluator_environment_receipt_path": (
                self.evaluator_environment_receipt_path
            ),
            "evaluator_environment_receipt_sha256": (
                self.evaluator_environment_receipt_sha256
            ),
            "evaluator_canary_receipt_sha256": (
                self.evaluator_canary_receipt_sha256
            ),
            "evaluator_approval_receipt_sha256": (
                self.evaluator_approval_receipt_sha256
            ),
            "evaluator_approval_public_key_sha256": (
                self.evaluator_approval_public_key_sha256
            ),
            "account_id": self.account_id,
            "availability_zone": self.availability_zone,
            "boot_id": self.boot_id,
            "instance_id": self.instance_id,
            "region": self.region,
            "purchase_model": self.purchase_model,
            "runtime_sbom_sha256": self.runtime_sbom_sha256,
            "objective_controls_contract_sha256": (
                self.objective_controls_contract_sha256
            ),
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "operational_config_sha256": self.operational_config_sha256,
            "finalization_receipt_sha256": (
                self.finalization_receipt_sha256
            ),
            "finalization_receipt_s3_uri": (
                self.finalization_receipt_s3_uri
            ),
            "finalization_receipt_bytes": self.finalization_receipt_bytes,
            "finalization_receipt_s3_version_id": (
                self.finalization_receipt_s3_version_id
            ),
            "collection_receipt_sha256": self.collection_receipt_sha256,
            "collection_receipt_s3_uri": self.collection_receipt_s3_uri,
            "collection_receipt_s3_version_id": (
                self.collection_receipt_s3_version_id
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


@dataclass(frozen=True)
class CollectionReceiptEvidence:
    """Exact durable bytes and identity of one per-seed collection receipt."""

    payload: bytes
    uri: str
    sha256: str
    version_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError(
                "collection receipt evidence payload bytes are missing"
            )
        if not isinstance(self.uri, str) or not self.uri:
            raise ValueError(
                "collection receipt evidence URI must be a non-empty string"
            )
        _sha256(self.sha256, "collection receipt evidence SHA-256")
        _s3_version_id(
            self.version_id,
            "collection receipt evidence S3 version ID",
        )


def _require_durably_collected_lock(
    *,
    study_lock: StudyLockV3,
    collection_receipts: Sequence[CollectionReceiptEvidence],
) -> None:
    """Prove every locked snapshot slot against ten exact Task 3F receipts."""

    if (
        isinstance(collection_receipts, (str, bytes))
        or not isinstance(collection_receipts, Sequence)
        or any(
            not isinstance(receipt, CollectionReceiptEvidence)
            for receipt in collection_receipts
        )
    ):
        raise ValueError(
            "snapshot planning requires an ordered sequence of collection "
            "receipt evidence"
        )
    receipts = tuple(collection_receipts)
    if len(receipts) != len(SEEDS):
        raise ValueError(
            "snapshot planning requires exactly ten per-seed collection "
            "receipts in ascending seed order"
        )
    from msctl.aws_collect import parse_seed_collection_receipt_bytes

    release_identities = set()
    for seed, evidence in zip(SEEDS, receipts, strict=True):
        lifecycle = study_lock.seed_lifecycles[seed]
        if (
            evidence.uri != lifecycle.collection_receipt_s3_uri
            or evidence.sha256 != lifecycle.collection_receipt_sha256
            or evidence.version_id
            != lifecycle.collection_receipt_s3_version_id
        ):
            raise ValueError(
                f"seed {seed} collection receipt identity differs from "
                "the study lock"
            )
        value = parse_seed_collection_receipt_bytes(
            evidence.payload,
            receipt_uri=evidence.uri,
            receipt_sha256=evidence.sha256,
            receipt_version_id=evidence.version_id,
            expected_binding=study_lock.lifecycle_binding(seed),
        )
        run_reference = value["run_receipt"]
        if (
            run_reference["uri"] != lifecycle.finalization_receipt_s3_uri
            or run_reference["sha256"]
            != lifecycle.finalization_receipt_sha256
            or run_reference["bytes"]
            != lifecycle.finalization_receipt_bytes
            or run_reference["version_id"]
            != lifecycle.finalization_receipt_s3_version_id
        ):
            raise ValueError(
                f"seed {seed} finalization receipt reference differs from "
                "the study lock"
            )
        if value["run_manifest_sha256"] != lifecycle.run_manifest_sha256:
            raise ValueError(
                f"seed {seed} run-manifest identity differs from the "
                "study lock"
            )
        if (
            value["source_commit"] != lifecycle.source_commit
            or value["source_tree"] != lifecycle.source_tree
        ):
            raise ValueError(
                f"seed {seed} source identity differs from the study lock"
            )
        slots = study_lock.snapshots[
            seed * 10 : (seed + 1) * 10
        ]
        if (
            value["dataset_receipt_sha256"] != slots[0].data_receipt_sha256
            or value["dataset_build_id"] != slots[0].data_build_id
            or value["ordered_stream_sha256"]
            != slots[0].ordered_stream_sha256
        ):
            raise ValueError(
                f"seed {seed} dataset identity differs from the locked "
                "snapshots"
            )
        release_identities.add(
            (value["release_sha256"], value["release_receipt_sha256"])
        )
        root = lifecycle.object_root
        rows = value["objects"]
        snapshot_rows = (*rows[0:5], *rows[6:11])
        for slot, row in zip(slots, snapshot_rows, strict=True):
            if (
                row["kind"] != "snapshot"
                or row["arm"] != slot.arm.value
                or row["step"] != slot.optimizer_step
                or row["sha256"] != slot.checkpoint_sha256
                or row["version_id"] != slot.s3_version_id
                or row["uri"] != f"{root}/{slot.s3_object_key}"
            ):
                raise ValueError(
                    f"seed {seed} locked snapshot slot "
                    f"{slot.arm.value}/step-{slot.optimizer_step} was not "
                    "durably collected"
                )
        receipt_row = rows[14]
        if (
            receipt_row["sha256"] != slots[0].checkpoint_receipt_sha256
            or receipt_row["version_id"]
            != slots[0].checkpoint_receipt_s3_version_id
            or receipt_row["uri"]
            != f"{root}/{slots[0].checkpoint_receipt_s3_object_key}"
        ):
            raise ValueError(
                f"seed {seed} locked checkpoint receipt was not durably "
                "collected"
            )
    if len(release_identities) != 1:
        raise ValueError(
            "collection receipts disagree on one training release identity"
        )


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
        lifecycle = study_lock.seed_lifecycles[snapshot.seed]
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
            snapshot_path=snapshot_path(snapshot.optimizer_step),
            study_lock_path=STUDY_LOCK_FILE_NAME,
            seed=snapshot.seed,
            arm=snapshot.arm,
            optimizer_step=snapshot.optimizer_step,
            snapshot_sha256=snapshot.checkpoint_sha256,
            snapshot_s3_object_key=snapshot.s3_object_key,
            snapshot_s3_version_id=snapshot.s3_version_id,
            checkpoint_receipt_sha256=(
                snapshot.checkpoint_receipt_sha256
            ),
            checkpoint_receipt_s3_object_key=(
                snapshot.checkpoint_receipt_s3_object_key
            ),
            checkpoint_receipt_s3_version_id=(
                snapshot.checkpoint_receipt_s3_version_id
            ),
            snapshot_version=snapshot.snapshot_version,
            training_run_id=snapshot.training_run_id,
            config_fingerprint=snapshot.config_fingerprint,
            training_config_sha256=snapshot.training_config_sha256,
            model_config_sha256=snapshot.model_config_sha256,
            model_identity=snapshot.model_identity,
            data_provenance_sha256=snapshot.data_provenance_sha256,
            data_receipt_sha256=snapshot.data_receipt_sha256,
            data_build_id=snapshot.data_build_id,
            ordered_stream_sha256=snapshot.ordered_stream_sha256,
            world_size=snapshot.world_size,
            tokens_per_step=snapshot.tokens_per_step,
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
            evaluator_profile_path=EVALUATOR_PROFILE_FILE_NAME,
            evaluator_profile_sha256=selection.profile_sha256,
            evaluator_runtime_lock_path=(
                EVALUATOR_RUNTIME_LOCK_FILE_NAME
            ),
            evaluator_runtime_lock_sha256=selection.runtime_lock_sha256,
            evaluator_qualification_evidence_sha256=(
                selection.qualification_evidence_sha256
            ),
            evaluator_environment_receipt_path=(
                EVALUATOR_ENVIRONMENT_RECEIPT_FILE_NAME
            ),
            evaluator_environment_receipt_sha256=(
                selection.environment_receipt_sha256
            ),
            evaluator_canary_receipt_sha256=(
                selection.canary_receipt_sha256
            ),
            evaluator_approval_receipt_sha256=(
                selection.approval_receipt_sha256
            ),
            evaluator_approval_public_key_sha256=(
                selection.approval_public_key_sha256
            ),
            account_id=lifecycle.account_id,
            availability_zone=lifecycle.availability_zone,
            boot_id=lifecycle.boot_id,
            instance_id=lifecycle.instance_id,
            region=lifecycle.region,
            purchase_model=lifecycle.purchase_model,
            runtime_sbom_sha256=lifecycle.runtime_sbom_sha256,
            objective_controls_contract_sha256=(
                lifecycle.objective_controls_contract_sha256
            ),
            source_commit=lifecycle.source_commit,
            source_tree=lifecycle.source_tree,
            operational_config_sha256=(
                lifecycle.operational_config_sha256(snapshot.arm)
            ),
            finalization_receipt_sha256=(
                lifecycle.finalization_receipt_sha256
            ),
            finalization_receipt_s3_uri=(
                lifecycle.finalization_receipt_s3_uri
            ),
            finalization_receipt_bytes=(
                lifecycle.finalization_receipt_bytes
            ),
            finalization_receipt_s3_version_id=(
                lifecycle.finalization_receipt_s3_version_id
            ),
            collection_receipt_sha256=(
                lifecycle.collection_receipt_sha256
            ),
            collection_receipt_s3_uri=(
                lifecycle.collection_receipt_s3_uri
            ),
            collection_receipt_s3_version_id=(
                lifecycle.collection_receipt_s3_version_id
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
    collection_receipts: Sequence[CollectionReceiptEvidence],
) -> tuple[SnapshotEvaluationPlan, ...]:
    """Enumerate the exact 100 receipt-proven slots in frozen order."""

    if not isinstance(study_lock, StudyLockV3):
        raise TypeError("snapshot planning requires StudyLockV3")
    _require_durably_collected_lock(
        study_lock=study_lock,
        collection_receipts=collection_receipts,
    )
    plans = _build_snapshot_evaluation_plans(
        study_lock=study_lock,
        study_lock_sha256=study_lock_sha256,
    )
    return validate_snapshot_evaluation_plans(
        plans,
        study_lock=study_lock,
        study_lock_sha256=study_lock_sha256,
        collection_receipts=collection_receipts,
    )


def validate_snapshot_evaluation_plans(
    plans: Sequence[SnapshotEvaluationPlan],
    *,
    study_lock: StudyLockV3,
    study_lock_sha256: str,
    collection_receipts: Sequence[CollectionReceiptEvidence],
) -> tuple[SnapshotEvaluationPlan, ...]:
    """Reject every missing, extra, reordered, replaced, or aliased slot."""

    if not isinstance(study_lock, StudyLockV3):
        raise TypeError("snapshot planning requires StudyLockV3")
    _require_durably_collected_lock(
        study_lock=study_lock,
        collection_receipts=collection_receipts,
    )
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


COHORT_REPORT_SCHEMA_V3 = "memorysplit.confirmatory.cohort-report.v3"
PRIMARY_CONTRAST_ID_V3 = (
    "primary_omnibus_pair_and_proof__graph_non_path__"
    "composition_joint_ood__split90_minus_dense"
)
PRIMARY_TEST_METHOD_V3 = "exact_one_sided_exhaustive_sign_flip"
SECONDARY_CONTRAST_IDS_V3 = (
    "secondary_graph_pair_and_proof__composition_joint_ood__"
    "split90_minus_dense",
    "secondary_non_path_pair_and_proof__composition_joint_ood__"
    "split90_minus_dense",
)
COHORT_REPORT_FILE_NAME = "cohort-report.json"
COHORT_REPORT_ID_PREFIX = "cohort-report-"
COHORT_REPORT_MEMBER_MODE = 0o444
_TERMINAL_STEP = V3_SNAPSHOT_STEPS[-1]
_EQUIVALENCE_MARGIN_EXACT = Fraction(str(V3_EQUIVALENCE_MARGIN))
_PRIMARY_STRATA = (Stratum.COMPOSITION_OOD, Stratum.JOINT_OOD)
_OUTPUT_ARTIFACTS = (
    "inference.json",
    "items.jsonl",
    "metrics.json",
    "outcomes.jsonl",
    "run.json",
    "sealed-gold.jsonl",
    "sealed-release.json",
    "stores.jsonl",
    "study-lock.json",
)
_OUTPUT_NAMES = tuple(sorted((*_OUTPUT_ARTIFACTS, "output.json")))
_MAX_OUTPUT_FILE_BYTES = 2 * 1024 * 1024 * 1024
_OUTPUT_MANIFEST_FIELDS = frozenset(
    {
        "record_type",
        "schema_version",
        "output_id",
        "run_binding_sha256",
        "study_lock_sha256",
        "sealed_evaluation_release_sha256",
        "provider_selection_sha256",
        "provider_selection_s3_version_id",
        "snapshot_sha256",
        "seed",
        "arm",
        "optimizer_step",
        "scope",
        "inference_scope",
        "final_conclusion",
        "production_qualified",
        "publication_class",
        "execution_identity_before",
        "execution_identity_after",
        "artifacts",
    }
)
_OUTPUT_ARTIFACT_FIELDS = frozenset({"path", "sha256", "bytes"})
_EXECUTION_IDENTITY_FIELDS = frozenset(
    {
        "production_qualified",
        "adapter_kind",
        "selected_provider",
        "profile_id",
        "profile_sha256",
        "runtime_lock_sha256",
        "qualification_evidence_sha256",
        "environment_receipt_sha256",
        "canary_receipt_sha256",
        "approval_receipt_sha256",
        "approval_public_key_sha256",
        "account_id",
        "instance_id",
        "boot_id",
        "region",
        "device_type",
        "cuda_available",
        "torch_version",
        "cuda_version",
        "device_count",
        "device_name",
        "device_capability",
    }
)
_INFERENCE_FIELDS_V3 = frozenset(
    {
        "record_type",
        "schema_version",
        "scope",
        "snapshot_scope",
        "output_id",
        "seed",
        "arm",
        "optimizer_step",
        "snapshot_sha256",
        "cohort_aggregation_status",
        "final_conclusion",
    }
)
_REPORT_FIELDS = frozenset(
    {
        "record_type",
        "schema_version",
        "cohort_id",
        "study_lock_sha256",
        "preregistration_sha256",
        "sealed_evaluation_release_sha256",
        "provider_selection",
        "inputs",
        "collection_receipts",
        "snapshots",
        "primary",
        "aulc",
        "secondary",
        "instrument_gates",
        "status",
    }
)
_REPORT_INPUT_FIELDS = frozenset(
    {
        "slot_index",
        "seed",
        "arm",
        "optimizer_step",
        "output_id",
        "output_commitment",
    }
)
_REPORT_COLLECTION_RECEIPT_FIELDS = frozenset(
    {
        "seed",
        "uri",
        "sha256",
        "version_id",
    }
)
_REPORT_SNAPSHOT_FIELDS = frozenset(
    {
        "slot_index",
        "seed",
        "arm",
        "optimizer_step",
        "output_id",
        "output_commitment",
        "snapshot_sha256",
        "primary_cells",
        "primary_accuracy",
    }
)
_STATUS_FIELDS = frozenset(
    {
        "scientific_status",
        "interim_evidence_label",
        "final_inference_conclusion",
    }
)


def _runner_schema_constants() -> tuple[str, str, str]:
    """Read the runner-owned v3 output schemas without a module cycle."""

    from evals.confirmatory.runner import (
        SNAPSHOT_INFERENCE_SCHEMA_V3,
        SNAPSHOT_METRICS_SCHEMA_V3,
        SNAPSHOT_OUTPUT_SCHEMA_V3,
    )

    return (
        SNAPSHOT_OUTPUT_SCHEMA_V3,
        SNAPSHOT_METRICS_SCHEMA_V3,
        SNAPSHOT_INFERENCE_SCHEMA_V3,
    )


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _exact_integer(
    value: object,
    name: str,
    *,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact integer")
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("report mappings require string keys")
        return MappingProxyType(
            {key: _freeze_json(value[key]) for key in sorted(value)}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("report contains a non-canonical value")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _fraction_dict(value: Fraction | int) -> dict[str, int]:
    exact = value if isinstance(value, Fraction) else Fraction(value)
    return {
        "numerator": exact.numerator,
        "denominator": exact.denominator,
    }


def _fraction_from_dict(value: object, name: str) -> Fraction:
    raw = _strict_fields(
        value,
        frozenset({"numerator", "denominator"}),
        name,
    )
    numerator = raw["numerator"]
    denominator = raw["denominator"]
    if type(numerator) is not int or type(denominator) is not int:
        raise ValueError(f"{name} must contain exact integer counts")
    if denominator <= 0:
        raise ValueError(f"{name} denominator must be positive")
    exact = Fraction(numerator, denominator)
    if (
        exact.numerator != numerator
        or exact.denominator != denominator
    ):
        raise ValueError(f"{name} must be a reduced exact fraction")
    return exact


@dataclass(frozen=True)
class SnapshotOutputReference:
    """Informational local path plus authoritative ``output.json`` hash."""

    output_dir: Path
    output_commitment: str

    def __post_init__(self) -> None:
        raw = os.fspath(self.output_dir)
        if not isinstance(raw, str) or not raw or "\0" in raw:
            raise ValueError("output directory path is invalid")
        if any(part == ".." for part in Path(raw).parts):
            raise ValueError("output directory path must not traverse parents")
        object.__setattr__(self, "output_dir", Path(raw))
        object.__setattr__(
            self,
            "output_commitment",
            _sha256(self.output_commitment, "output commitment"),
        )

    @property
    def authoritative_commitment(self) -> str:
        return self.output_commitment

    @property
    def path_authority(self) -> str:
        return PATH_AUTHORITY


@dataclass(frozen=True)
class CohortReport:
    """Closed canonical report whose serialized bytes are the content address."""

    record_type: str
    schema_version: int
    cohort_id: str
    study_lock_sha256: str
    preregistration_sha256: str
    sealed_evaluation_release_sha256: str
    provider_selection: Mapping[str, Any]
    inputs: tuple[Mapping[str, Any], ...]
    collection_receipts: tuple[Mapping[str, Any], ...]
    snapshots: tuple[Mapping[str, Any], ...]
    primary: Mapping[str, Any]
    aulc: tuple[Mapping[str, Any], ...]
    secondary: tuple[Mapping[str, Any], ...]
    instrument_gates: Mapping[str, Any]
    status: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.record_type != COHORT_REPORT_SCHEMA_V3:
            raise ValueError("cohort report record_type is invalid")
        if (
            type(self.schema_version) is not int
            or self.schema_version != STUDY_CONTRACT_VERSION
        ):
            raise ValueError("cohort report schema_version is invalid")
        if self.cohort_id != "memorysplit-confirmatory-v3-360m-n10-aws":
            raise ValueError("cohort report cohort identity is invalid")
        for field in (
            "study_lock_sha256",
            "preregistration_sha256",
            "sealed_evaluation_release_sha256",
        ):
            object.__setattr__(
                self,
                field,
                _sha256(getattr(self, field), field),
            )
        if self.preregistration_sha256 != FROZEN_PREREGISTRATION_SHA256_V3:
            raise ValueError(
                "cohort report preregistration commitment is invalid"
            )
        selection = ProviderSelectionBinding.from_dict(self.provider_selection)
        object.__setattr__(
            self,
            "provider_selection",
            _freeze_json(selection.to_dict()),
        )
        if (
            isinstance(self.inputs, (str, bytes))
            or not isinstance(self.inputs, Sequence)
            or len(self.inputs) != len(EXPECTED_STUDY_SLOTS_V3)
        ):
            raise ValueError("cohort report requires exactly 100 inputs")
        inputs = tuple(_freeze_json(value) for value in self.inputs)
        commitments = []
        for index, raw in enumerate(inputs):
            value = _strict_fields(
                raw,
                _REPORT_INPUT_FIELDS,
                f"cohort report input {index}",
            )
            slot_index = _exact_integer(
                value["slot_index"],
                "input slot_index",
            )
            try:
                arm = StudyArm(value["arm"])
            except (TypeError, ValueError) as exc:
                raise ValueError("cohort report input arm is invalid") from exc
            slot = (
                _exact_integer(value["seed"], "input seed"),
                arm,
                _exact_integer(value["optimizer_step"], "input step"),
            )
            if slot_index != index or slot != EXPECTED_STUDY_SLOTS_V3[index]:
                raise ValueError("cohort report inputs are not in slot order")
            _nonempty_string(value["output_id"], "input output_id")
            commitments.append(
                _sha256(
                    value["output_commitment"],
                    "input output commitment",
                )
            )
        if len(set(commitments)) != len(commitments):
            raise ValueError("cohort report input commitments are duplicated")
        object.__setattr__(self, "inputs", inputs)

        if (
            isinstance(self.collection_receipts, (str, bytes))
            or not isinstance(self.collection_receipts, Sequence)
            or len(self.collection_receipts) != len(SEEDS)
        ):
            raise ValueError(
                "cohort report requires exactly ten collection receipt rows"
            )
        receipt_rows = tuple(
            _freeze_json(value) for value in self.collection_receipts
        )
        receipt_uris = []
        receipt_hashes = []
        for index, raw in enumerate(receipt_rows):
            value = _strict_fields(
                raw,
                _REPORT_COLLECTION_RECEIPT_FIELDS,
                f"cohort report collection receipt {index}",
            )
            seed = _exact_integer(value["seed"], "collection receipt seed")
            if seed != SEEDS[index]:
                raise ValueError(
                    "cohort report collection receipts are not in ascending "
                    "seed order"
                )
            receipt_uris.append(
                _nonempty_string(
                    value["uri"],
                    "cohort report collection receipt URI",
                )
            )
            receipt_hashes.append(
                _sha256(
                    value["sha256"],
                    "cohort report collection receipt SHA-256",
                )
            )
            _s3_version_id(
                value["version_id"],
                "cohort report collection receipt S3 version ID",
            )
        if (
            len(set(receipt_uris)) != len(SEEDS)
            or len(set(receipt_hashes)) != len(SEEDS)
        ):
            raise ValueError(
                "cohort report collection receipts are duplicated"
            )
        object.__setattr__(self, "collection_receipts", receipt_rows)

        if (
            isinstance(self.snapshots, (str, bytes))
            or not isinstance(self.snapshots, Sequence)
            or len(self.snapshots) != len(EXPECTED_STUDY_SLOTS_V3)
        ):
            raise ValueError("cohort report requires exactly 100 snapshots")
        snapshots = tuple(_freeze_json(value) for value in self.snapshots)
        for index, raw in enumerate(snapshots):
            value = _strict_fields(
                raw,
                _REPORT_SNAPSHOT_FIELDS,
                f"cohort report snapshot {index}",
            )
            if value["slot_index"] != index:
                raise ValueError("cohort report snapshot indices are invalid")
            slot = (
                value["seed"],
                StudyArm(value["arm"]),
                value["optimizer_step"],
            )
            if slot != EXPECTED_STUDY_SLOTS_V3[index]:
                raise ValueError("cohort report snapshots are not ordered")
            _sha256(value["output_commitment"], "snapshot output commitment")
            _sha256(value["snapshot_sha256"], "snapshot SHA-256")
            cells = _strict_fields(
                value["primary_cells"],
                frozenset(PRIMARY_CELL_IDS),
                "snapshot primary cells",
            )
            for cell_id in PRIMARY_CELL_IDS:
                _fraction_from_dict(
                    cells[cell_id],
                    f"snapshot primary cell {cell_id}",
                )
            _fraction_from_dict(
                value["primary_accuracy"],
                "snapshot primary accuracy",
            )
        object.__setattr__(self, "snapshots", snapshots)

        object.__setattr__(self, "primary", _freeze_json(self.primary))
        if (
            isinstance(self.aulc, (str, bytes))
            or not isinstance(self.aulc, Sequence)
            or len(self.aulc) != 20
        ):
            raise ValueError("cohort report AULC requires 20 seed/arm rows")
        object.__setattr__(
            self,
            "aulc",
            tuple(_freeze_json(value) for value in self.aulc),
        )
        if (
            isinstance(self.secondary, (str, bytes))
            or not isinstance(self.secondary, Sequence)
            or len(self.secondary) != 2
        ):
            raise ValueError(
                "cohort report requires the two secondary contrasts"
            )
        secondary = tuple(_freeze_json(value) for value in self.secondary)
        if tuple(value.get("contrast_id") for value in secondary) != tuple(
            sorted(SECONDARY_CONTRAST_IDS_V3)
        ):
            raise ValueError(
                "cohort report secondary contrast order is invalid"
            )
        object.__setattr__(self, "secondary", secondary)
        gates = _freeze_json(self.instrument_gates)
        if not isinstance(gates, Mapping) or gates.get("all_passed") is not True:
            raise ValueError("cohort report instrument gates are not complete")
        object.__setattr__(self, "instrument_gates", gates)
        status = _strict_fields(
            self.status,
            _STATUS_FIELDS,
            "cohort report status",
        )
        allowed = {
            "scientific_status": {"incomplete", "invalid", "complete"},
            "interim_evidence_label": {
                "none",
                "directional_only",
                "sign_consistent_only",
            },
            "final_inference_conclusion": {
                "not_evaluated",
                "inconclusive",
                "supports_effect",
                "supports_practical_null",
            },
        }
        for field, values in allowed.items():
            if status[field] not in values:
                raise ValueError(f"cohort report {field} is invalid")
        if (
            status["scientific_status"] != "complete"
            or status["interim_evidence_label"] != "none"
            or status["final_inference_conclusion"] == "not_evaluated"
        ):
            raise ValueError(
                "an exact 100-output cohort report must be terminal"
            )
        object.__setattr__(self, "status", _freeze_json(status))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CohortReport":
        value = _strict_fields(raw, _REPORT_FIELDS, "cohort report")
        return cls(
            record_type=value["record_type"],
            schema_version=value["schema_version"],
            cohort_id=value["cohort_id"],
            study_lock_sha256=value["study_lock_sha256"],
            preregistration_sha256=value["preregistration_sha256"],
            sealed_evaluation_release_sha256=value[
                "sealed_evaluation_release_sha256"
            ],
            provider_selection=value["provider_selection"],
            inputs=tuple(value["inputs"]),
            collection_receipts=tuple(value["collection_receipts"]),
            snapshots=tuple(value["snapshots"]),
            primary=value["primary"],
            aulc=tuple(value["aulc"]),
            secondary=tuple(value["secondary"]),
            instrument_gates=value["instrument_gates"],
            status=value["status"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "cohort_id": self.cohort_id,
            "study_lock_sha256": self.study_lock_sha256,
            "preregistration_sha256": self.preregistration_sha256,
            "sealed_evaluation_release_sha256": (
                self.sealed_evaluation_release_sha256
            ),
            "provider_selection": _thaw_json(self.provider_selection),
            "inputs": _thaw_json(self.inputs),
            "collection_receipts": _thaw_json(self.collection_receipts),
            "snapshots": _thaw_json(self.snapshots),
            "primary": _thaw_json(self.primary),
            "aulc": _thaw_json(self.aulc),
            "secondary": _thaw_json(self.secondary),
            "instrument_gates": _thaw_json(self.instrument_gates),
            "status": _thaw_json(self.status),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def authoritative_commitment(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True)
class PublishedCohortReport:
    report_dir: Path
    report_path: Path
    authoritative_commitment: str
    path_authority: str = PATH_AUTHORITY


@dataclass(frozen=True)
class _OutputFileSnapshot:
    name: str
    state: tuple[int, int, int, int, int, int, int, int, int]
    sha256: str


@dataclass(frozen=True)
class _OutputDirectorySnapshot:
    identity: tuple[int, int, int, int, int, int]
    modified_ns: int
    changed_ns: int
    files: tuple[_OutputFileSnapshot, ...]


@dataclass
class _OpenedOutput:
    reference: SnapshotOutputReference
    descriptor: int
    snapshot: _OutputDirectorySnapshot
    contents: Mapping[str, bytes]


@dataclass(frozen=True)
class _ReleaseData:
    items: Mapping[str, ItemRecord]
    gold: Mapping[str, SealedGoldRecord]
    stores: Mapping[str, StoreRecord]


@dataclass(frozen=True)
class _SnapshotEvidence:
    slot_index: int
    reference: SnapshotOutputReference
    binding: RunBindingV3
    primary_summary: StudyMetricsRecord
    primary_pair_results: Mapping[
        tuple[str, str, str, str],
        bool,
    ]

    @property
    def exact_primary_accuracy(self) -> Fraction:
        return (
            sum(
                (
                    rate.exact_value
                    for rate in self.primary_summary.primary_cells.values()
                ),
                Fraction(),
            )
            / len(PRIMARY_CELL_IDS)
        )


def _output_directory_identity(
    details: os.stat_result,
    *,
    label: str,
) -> tuple[int, int, int, int, int, int]:
    mode = stat.S_IMODE(details.st_mode)
    if (
        not stat.S_ISDIR(details.st_mode)
        or details.st_uid != os.geteuid()
        or details.st_nlink < 1
        or mode != 0o700
    ):
        raise ValueError(f"{label} must be an owned mode-0o700 directory")
    return (
        details.st_dev,
        details.st_ino,
        mode,
        details.st_uid,
        details.st_gid,
        details.st_nlink,
    )


def _output_file_state(
    details: os.stat_result,
    *,
    label: str,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    mode = stat.S_IMODE(details.st_mode)
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_uid != os.geteuid()
        or details.st_nlink != 1
        or mode != 0o600
        or details.st_size < 1
        or details.st_size > _MAX_OUTPUT_FILE_BYTES
    ):
        raise ValueError(f"{label} metadata is unsafe")
    return (
        details.st_dev,
        details.st_ino,
        mode,
        details.st_uid,
        details.st_gid,
        details.st_nlink,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _read_output_file_at(
    directory_fd: int,
    name: str,
) -> tuple[bytes, _OutputFileSnapshot]:
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        initial = _output_file_state(
            os.fstat(descriptor),
            label=f"snapshot output {name}",
        )
        chunks = []
        offset = 0
        while offset < initial[6]:
            chunk = os.pread(
                descriptor,
                min(1 << 20, initial[6] - offset),
                offset,
            )
            if not chunk:
                raise ValueError(f"snapshot output {name} changed while read")
            chunks.append(chunk)
            offset += len(chunk)
        if os.pread(descriptor, 1, initial[6]):
            raise ValueError(f"snapshot output {name} grew while read")
        content = b"".join(chunks)
        final = _output_file_state(
            os.fstat(descriptor),
            label=f"snapshot output {name}",
        )
        named = _output_file_state(
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False),
            label=f"snapshot output {name}",
        )
        if initial != final or final != named:
            raise ValueError(
                f"snapshot output {name} identity changed while read"
            )
        return (
            content,
            _OutputFileSnapshot(
                name=name,
                state=final,
                sha256=hashlib.sha256(content).hexdigest(),
            ),
        )
    finally:
        os.close(descriptor)


def _capture_output_directory(
    descriptor: int,
) -> tuple[_OutputDirectorySnapshot, Mapping[str, bytes]]:
    before = os.fstat(descriptor)
    identity = _output_directory_identity(
        before,
        label="snapshot output directory",
    )
    names = tuple(sorted(os.listdir(descriptor)))
    if names != _OUTPUT_NAMES:
        raise ValueError("snapshot output directory membership is not exact")
    contents = {}
    files = []
    for name in names:
        content, file_snapshot = _read_output_file_at(descriptor, name)
        contents[name] = content
        files.append(file_snapshot)
    after = os.fstat(descriptor)
    if (
        _output_directory_identity(
            after,
            label="snapshot output directory",
        )
        != identity
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise ValueError("snapshot output directory changed while read")
    return (
        _OutputDirectorySnapshot(
            identity=identity,
            modified_ns=after.st_mtime_ns,
            changed_ns=after.st_ctime_ns,
            files=tuple(files),
        ),
        MappingProxyType(contents),
    )


def _assert_output_path_binding(opened: _OpenedOutput) -> None:
    try:
        named = _output_directory_identity(
            os.stat(opened.reference.output_dir, follow_symlinks=False),
            label="snapshot output directory path",
        )
    except OSError as exc:
        raise ValueError(
            "snapshot output informational path was replaced"
        ) from exc
    pinned = _output_directory_identity(
        os.fstat(opened.descriptor),
        label="snapshot output directory",
    )
    if named != opened.snapshot.identity or pinned != opened.snapshot.identity:
        raise ValueError(
            "snapshot output informational path identity was replaced"
        )


def _open_output(reference: SnapshotOutputReference) -> _OpenedOutput:
    try:
        descriptor = open_directory(
            reference.output_dir,
            label="snapshot output directory",
        )
    except Exception as exc:
        raise ValueError(
            "snapshot output directory cannot be descriptor-pinned"
        ) from exc
    try:
        snapshot, contents = _capture_output_directory(descriptor)
        opened = _OpenedOutput(reference, descriptor, snapshot, contents)
        _assert_output_path_binding(opened)
        return opened
    except BaseException:
        os.close(descriptor)
        raise


def _replay_opened_output(opened: _OpenedOutput) -> None:
    observed, _contents = _capture_output_directory(opened.descriptor)
    if observed != opened.snapshot:
        raise ValueError(
            "snapshot output changed before final input replay"
        )
    _assert_output_path_binding(opened)


def _run_aggregation_mutation_hook(event: str, **context: object) -> None:
    """Deterministic race boundary; production intentionally performs no action."""

    del event, context


def _canonical_object(content: bytes, name: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be canonical UTF-8 JSON") from exc
    try:
        canonical = canonical_json_bytes(raw)
    except (TypeError, UnicodeError, ValueError) as exc:
        raise ValueError(f"{name} contains a non-canonical value") from exc
    if not isinstance(raw, Mapping) or canonical != content:
        raise ValueError(f"{name} must be one canonical JSON object")
    return raw


def _canonical_jsonl(
    content: bytes,
    *,
    name: str,
    parser,
    identity,
) -> tuple[Any, ...]:
    lines = content.splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") for line in lines):
        raise ValueError(f"{name} must be non-empty canonical JSONL")
    records = []
    identities = []
    for index, line in enumerate(lines):
        try:
            raw = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{name} line {index} is invalid JSON") from exc
        if not isinstance(raw, Mapping) or canonical_json_bytes(raw) != line:
            raise ValueError(f"{name} line {index} is not canonical")
        try:
            record = parser(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} line {index} is invalid: {exc}") from exc
        records.append(record)
        identities.append(identity(record))
    if len(set(identities)) != len(identities):
        raise ValueError(f"{name} contains a duplicate identity")
    if tuple(identities) != tuple(sorted(identities)):
        raise ValueError(f"{name} is not strictly ordered")
    return tuple(records)


def _artifact_bindings(
    manifest: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    rows = manifest["artifacts"]
    if not isinstance(rows, list):
        raise ValueError("snapshot output artifact bindings must be ordered")
    bindings = {}
    paths = []
    for index, raw in enumerate(rows):
        value = _strict_fields(
            raw,
            _OUTPUT_ARTIFACT_FIELDS,
            f"snapshot output artifact {index}",
        )
        path = _nonempty_string(value["path"], "artifact path")
        if Path(path).name != path or path in {".", ".."}:
            raise ValueError("snapshot output artifact path is unsafe")
        digest = _sha256(value["sha256"], f"artifact {path} SHA-256")
        size = _exact_integer(
            value["bytes"],
            f"artifact {path} byte count",
            minimum=1,
        )
        if path in bindings:
            raise ValueError("snapshot output artifact path is duplicated")
        paths.append(path)
        bindings[path] = MappingProxyType(
            {"path": path, "sha256": digest, "bytes": size}
        )
    if tuple(paths) != tuple(sorted(_OUTPUT_ARTIFACTS)):
        raise ValueError("snapshot output artifact registry is not exact")
    return MappingProxyType(bindings)


def _validate_execution_identity(
    *,
    manifest: Mapping[str, Any],
    binding: RunBindingV3,
) -> None:
    before = _strict_fields(
        manifest["execution_identity_before"],
        _EXECUTION_IDENTITY_FIELDS,
        "evaluator execution identity before",
    )
    after = _strict_fields(
        manifest["execution_identity_after"],
        _EXECUTION_IDENTITY_FIELDS,
        "evaluator execution identity after",
    )
    if canonical_json_bytes(before) != canonical_json_bytes(after):
        raise ValueError("evaluator execution identity changed")
    expected = {
        "selected_provider": binding.selected_provider,
        "profile_id": binding.evaluator_profile_id,
        "profile_sha256": binding.evaluator_profile_sha256,
        "runtime_lock_sha256": binding.evaluator_runtime_lock_sha256,
        "qualification_evidence_sha256": (
            binding.evaluator_qualification_evidence_sha256
        ),
        "environment_receipt_sha256": (
            binding.evaluator_environment_receipt_sha256
        ),
        "canary_receipt_sha256": binding.evaluator_canary_receipt_sha256,
        "approval_receipt_sha256": binding.evaluator_approval_receipt_sha256,
        "approval_public_key_sha256": (
            binding.evaluator_approval_public_key_sha256
        ),
    }
    if any(before[field] != value for field, value in expected.items()):
        raise ValueError(
            "evaluator profile, selection, or evidence identity is crossed"
        )
    for field in (
        "account_id",
        "instance_id",
        "boot_id",
        "region",
        "torch_version",
        "cuda_version",
        "device_name",
    ):
        _nonempty_string(before[field], f"evaluator {field}")
    capability = before["device_capability"]
    if (
        before["production_qualified"] is not True
        or before["adapter_kind"] != "repository"
        or before["device_type"] != "cuda"
        or before["cuda_available"] is not True
        or type(before["device_count"]) is not int
        or before["device_count"] != 8
        or not isinstance(capability, (list, tuple))
        or len(capability) != 2
        or any(type(part) is not int or part < 0 for part in capability)
    ):
        raise ValueError(
            "evaluator execution identity is not provider-qualified"
        )


def _validate_inference_placeholder(
    content: bytes,
    binding: RunBindingV3,
) -> None:
    _output_schema, _metrics_schema, inference_schema = (
        _runner_schema_constants()
    )
    value = _strict_fields(
        _canonical_object(content, "inference.json"),
        _INFERENCE_FIELDS_V3,
        "snapshot inference evidence",
    )
    expected = {
        "record_type": inference_schema,
        "schema_version": STUDY_CONTRACT_VERSION,
        "scope": "cohort",
        "snapshot_scope": "single_snapshot",
        "output_id": binding.output_id,
        "seed": binding.seed,
        "arm": binding.arm.value,
        "optimizer_step": binding.optimizer_step,
        "snapshot_sha256": binding.snapshot_sha256,
        "cohort_aggregation_status": "not_implemented",
        "final_conclusion": None,
    }
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise ValueError(
            "snapshot inference evidence asserts or crosses cohort identity"
        )


def _release_data(
    *,
    contents: Mapping[str, bytes],
    release_sha256: str,
    cache: dict[tuple[str, str, str, str], _ReleaseData],
) -> _ReleaseData:
    items_content = contents["items.jsonl"]
    stores_content = contents["stores.jsonl"]
    gold_content = contents["sealed-gold.jsonl"]
    manifest_content = contents["sealed-release.json"]
    key = (
        hashlib.sha256(items_content).hexdigest(),
        hashlib.sha256(stores_content).hexdigest(),
        hashlib.sha256(gold_content).hexdigest(),
        hashlib.sha256(manifest_content).hexdigest(),
    )
    if key[-1] != release_sha256:
        raise ValueError(
            "sealed-release manifest disagrees with the study commitment"
        )
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        validated = _validated_release_from_preregistration_sha256(
            preregistration_sha256=FROZEN_PREREGISTRATION_SHA256_V3,
            items_content=items_content,
            stores_content=stores_content,
            gold_content=gold_content,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sealed release replay failed: {exc}") from exc
    if validated.manifest_bytes != manifest_content:
        raise ValueError(
            "sealed release source bytes disagree with their manifest"
        )
    items = _canonical_jsonl(
        items_content,
        name="items.jsonl",
        parser=ItemRecord.from_dict,
        identity=lambda record: record.item_id,
    )
    stores = _canonical_jsonl(
        stores_content,
        name="stores.jsonl",
        parser=StoreRecord.from_dict,
        identity=lambda record: record.store_id,
    )
    gold = _canonical_jsonl(
        gold_content,
        name="sealed-gold.jsonl",
        parser=SealedGoldRecord.from_dict,
        identity=lambda record: record.item_id,
    )
    data = _ReleaseData(
        items=MappingProxyType({record.item_id: record for record in items}),
        gold=MappingProxyType({record.item_id: record for record in gold}),
        stores=MappingProxyType(
            {record.store_id: record for record in stores}
        ),
    )
    cache[key] = data
    return data


def _pair_summary(
    rows: Sequence[tuple[StudyOutcomeRecord, Any]],
    *,
    binding: RunBindingV3,
) -> tuple[
    StudyMetricsRecord,
    Mapping[tuple[str, str, str, str], bool],
]:
    grouped: dict[
        tuple[int, str, str],
        dict[Twin, tuple[StudyOutcomeRecord, Any]],
    ] = defaultdict(dict)
    for outcome, verification in rows:
        key = outcome.seed, outcome.world_id, outcome.pair_id
        if outcome.twin in grouped[key]:
            raise ValueError("outcomes contain a duplicate pair twin")
        grouped[key][outcome.twin] = (outcome, verification)
    stratum_successes = {stratum: 0 for stratum in Stratum}
    stratum_totals = {stratum: 0 for stratum in Stratum}
    family_successes = {family: 0 for family in ReasoningFamily}
    family_totals = {family: 0 for family in ReasoningFamily}
    primary_successes = {cell: 0 for cell in PRIMARY_CELLS}
    primary_totals = {cell: 0 for cell in PRIMARY_CELLS}
    primary_pair_results = {}
    pair_metadata = (
        "pair_id",
        "stratum",
        "family",
        "seed",
        "world_id",
        "checkpoint_sha256",
        "arm",
        "condition_id",
        "optimizer_step",
        "raw_token_count",
        "memory_mode",
        "control",
    )
    for pair in grouped.values():
        if set(pair) != {Twin.ORIGINAL, Twin.COUNTERFACTUAL}:
            raise ValueError("every counterfactual pair requires both twins")
        original, original_verification = pair[Twin.ORIGINAL]
        counterfactual, counterfactual_verification = pair[
            Twin.COUNTERFACTUAL
        ]
        if any(
            getattr(original, field) != getattr(counterfactual, field)
            for field in pair_metadata
        ):
            raise ValueError("counterfactual pair metadata is crossed")
        success = (
            original_verification.proof_valid
            and original_verification.answer_valid
            and counterfactual_verification.proof_valid
            and counterfactual_verification.answer_valid
        )
        stratum_successes[original.stratum] += success
        stratum_totals[original.stratum] += 1
        family_successes[original.family] += success
        family_totals[original.family] += 1
        cell = original.family, original.stratum
        if cell in primary_totals:
            primary_successes[cell] += success
            primary_totals[cell] += 1
            primary_pair_results[
                (
                    original.family.value,
                    original.stratum.value,
                    original.world_id,
                    original.pair_id,
                )
            ] = bool(success)
    if any(primary_totals[cell] == 0 for cell in PRIMARY_CELLS):
        raise ValueError("snapshot outcomes omit a frozen primary cell")
    primary_rates = {
        f"{family.value}__{stratum.value}": Rate(
            primary_successes[(family, stratum)],
            primary_totals[(family, stratum)],
        )
        for family, stratum in PRIMARY_CELLS
    }
    summary = StudyMetricsRecord(
        primary_accuracy=(
            sum(rate.value for rate in primary_rates.values())
            / len(PRIMARY_CELLS)
        ),
        primary_cells=primary_rates,
        overall_pair_accuracy=Rate(
            sum(stratum_successes.values()),
            sum(stratum_totals.values()),
        ),
        by_stratum={
            stratum: Rate(stratum_successes[stratum], total)
            for stratum, total in stratum_totals.items()
            if total
        },
        by_family={
            family: Rate(family_successes[family], total)
            for family, total in family_totals.items()
        },
        checkpoint_sha256=binding.snapshot_sha256,
        seed=binding.seed,
        arm=binding.arm,
        condition_id=binding.condition_id,
        optimizer_step=binding.optimizer_step,
        raw_token_count=binding.raw_token_count,
        memory_mode=rows[0][0].memory_mode,
        control=rows[0][0].control,
    )
    return summary, MappingProxyType(primary_pair_results)


def _recompute_metrics(
    *,
    contents: Mapping[str, bytes],
    release: _ReleaseData,
    binding: RunBindingV3,
) -> tuple[
    StudyMetricsRecord,
    Mapping[tuple[str, str, str, str], bool],
]:
    _output_schema, metrics_schema, _inference_schema = (
        _runner_schema_constants()
    )
    outcomes = _canonical_jsonl(
        contents["outcomes.jsonl"],
        name="outcomes.jsonl",
        parser=StudyOutcomeRecord.from_dict,
        identity=lambda record: record.item_id,
    )
    if tuple(record.item_id for record in outcomes) != tuple(release.items):
        raise ValueError(
            "snapshot outcomes are missing, extra, duplicated, or reordered"
        )
    scored = []
    for outcome in outcomes:
        item = release.items[outcome.item_id]
        gold = release.gold[outcome.item_id]
        try:
            store = release.stores[item.store_id]
        except KeyError as exc:
            raise ValueError(
                "outcome item references an unknown store"
            ) from exc
        for field in (
            "item_id",
            "pair_id",
            "twin",
            "stratum",
            "family",
            "world_id",
            "memory_mode",
            "control",
        ):
            if getattr(outcome, field) != getattr(item, field):
                raise ValueError(f"raw outcome item identity mismatch: {field}")
        if (
            outcome.seed != binding.seed
            or outcome.arm != binding.arm
            or outcome.condition_id.value != binding.condition_id
            or outcome.optimizer_step != binding.optimizer_step
            or outcome.raw_token_count != binding.raw_token_count
            or outcome.checkpoint_sha256 != binding.snapshot_sha256
        ):
            raise ValueError(
                "raw outcome checkpoint, arm, or training identity is crossed"
            )
        verification = verify_proof_and_answer(
            item=item,
            store=store,
            gold=gold,
            proof=outcome.submitted_proof,
            answer=outcome.submitted_answer,
            solver=registered_solver(gold.solver_id),
        )
        scored.append((outcome, verification))
    grouped = defaultdict(list)
    for outcome, verification in scored:
        grouped[(outcome.memory_mode, outcome.control)].append(
            (outcome, verification)
        )
    summaries = []
    primary_summary = None
    primary_pair_results = None
    for key in sorted(
        grouped,
        key=lambda value: (value[0].value, value[1].value),
    ):
        summary, pair_results = _pair_summary(
            grouped[key],
            binding=binding,
        )
        summaries.append(summary)
        if key == (MemoryMode.MEMORY_ON, Control.CORRECT):
            primary_summary = summary
            primary_pair_results = pair_results
    if primary_summary is None or primary_pair_results is None:
        raise ValueError("snapshot outcomes omit the correct-memory instrument")
    recomputed = canonical_json_bytes(
        {
            "record_type": metrics_schema,
            "schema_version": STUDY_CONTRACT_VERSION,
            "scope": "single_snapshot",
            "seed": binding.seed,
            "arm": binding.arm.value,
            "optimizer_step": binding.optimizer_step,
            "snapshot_sha256": binding.snapshot_sha256,
            "summaries": [summary.to_dict() for summary in summaries],
        }
    )
    if recomputed != contents["metrics.json"]:
        raise ValueError(
            "metrics artifact disagrees with recomputed raw outcomes"
        )
    return primary_summary, primary_pair_results


def _validate_snapshot_output(
    *,
    slot_index: int,
    opened: _OpenedOutput,
    plan: SnapshotEvaluationPlan,
    lock: StudyLockV3,
    lock_sha256: str,
    release_cache: dict[tuple[str, str, str, str], _ReleaseData],
) -> _SnapshotEvidence:
    output_schema, _metrics_schema, _inference_schema = (
        _runner_schema_constants()
    )
    contents = opened.contents
    manifest_content = contents["output.json"]
    actual_commitment = hashlib.sha256(manifest_content).hexdigest()
    if actual_commitment != opened.reference.output_commitment:
        raise ValueError(
            "snapshot output disagrees with its authoritative commitment"
        )
    manifest = _strict_fields(
        _canonical_object(manifest_content, "output.json"),
        _OUTPUT_MANIFEST_FIELDS,
        "snapshot output manifest",
    )
    if (
        manifest["record_type"] != output_schema
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != STUDY_CONTRACT_VERSION
    ):
        raise ValueError("snapshot output manifest schema identity is invalid")
    bindings = _artifact_bindings(manifest)
    for name in _OUTPUT_ARTIFACTS:
        content = contents[name]
        binding_row = bindings[name]
        if (
            hashlib.sha256(content).hexdigest() != binding_row["sha256"]
            or len(content) != binding_row["bytes"]
        ):
            raise ValueError(
                f"snapshot output artifact {name} hash or size is invalid"
            )
    if (
        hashlib.sha256(contents["run.json"]).hexdigest()
        != manifest["run_binding_sha256"]
    ):
        raise ValueError("run binding artifact commitment is invalid")
    binding = RunBindingV3.from_dict(
        _canonical_object(contents["run.json"], "run.json")
    )
    if contents["run.json"] != canonical_json_bytes(binding.to_dict()):
        raise ValueError("run binding bytes are not canonical")
    if binding != plan.binding:
        raise ValueError(
            "run binding is missing, reordered, cross-profile, or replaced"
        )
    if (
        hashlib.sha256(contents["study-lock.json"]).hexdigest()
        != lock_sha256
        or manifest["study_lock_sha256"] != lock_sha256
        or binding.study_lock_sha256 != lock_sha256
    ):
        raise ValueError("study-lock identity or commitment is crossed")
    output_lock = StudyLockV3.from_dict(
        _canonical_object(contents["study-lock.json"], "study-lock.json")
    )
    if output_lock != lock:
        raise ValueError("snapshot output carries a different study lock")
    release = _release_data(
        contents=contents,
        release_sha256=lock.sealed_evaluation_release_sha256,
        cache=release_cache,
    )
    if (
        manifest["output_id"] != binding.output_id
        or manifest["sealed_evaluation_release_sha256"]
        != binding.sealed_evaluation_release_sha256
        or manifest["provider_selection_sha256"]
        != binding.provider_selection_sha256
        or manifest["provider_selection_s3_version_id"]
        != binding.provider_selection_s3_version_id
        or manifest["snapshot_sha256"] != binding.snapshot_sha256
        or manifest["seed"] != binding.seed
        or manifest["arm"] != binding.arm.value
        or manifest["optimizer_step"] != binding.optimizer_step
        or manifest["scope"] != "single_snapshot"
        or manifest["inference_scope"] != "cohort"
        or manifest["final_conclusion"] is not None
        or manifest["production_qualified"] is not True
        or manifest["publication_class"] != "provider_qualified"
    ):
        raise ValueError(
            "snapshot output run, release, provider, checkpoint, or scope "
            "identity is crossed"
        )
    _validate_execution_identity(manifest=manifest, binding=binding)
    _validate_inference_placeholder(contents["inference.json"], binding)
    primary_summary, pair_results = _recompute_metrics(
        contents=contents,
        release=release,
        binding=binding,
    )
    return _SnapshotEvidence(
        slot_index=slot_index,
        reference=opened.reference,
        binding=binding,
        primary_summary=primary_summary,
        primary_pair_results=pair_results,
    )


def _family_accuracy(
    evidence: _SnapshotEvidence,
    family: ReasoningFamily,
) -> Fraction:
    return (
        sum(
            (
                evidence.primary_summary.primary_cells[
                    f"{family.value}__{stratum.value}"
                ].exact_value
                for stratum in _PRIMARY_STRATA
            ),
            Fraction(),
        )
        / len(_PRIMARY_STRATA)
    )


def _exact_test_dict(result) -> dict[str, Any]:
    if result.assignments != V3_SIGN_ASSIGNMENTS:
        raise ValueError("exact test assignment count is not the frozen 1,024")
    p_value = Fraction(result.extreme_count, result.assignments)
    reject = result.statistic > 0 and p_value <= Fraction(1, 20)
    return {
        "method": PRIMARY_TEST_METHOD_V3,
        "alternative": "greater",
        "inclusive_tail": True,
        "retained_zero_deltas": True,
        "statistic": _fraction_dict(result.statistic),
        "extreme_count": result.extreme_count,
        "assignments": result.assignments,
        "p_value": _fraction_dict(p_value),
        "reject_null": bool(reject),
    }


def _bootstrap_observations(
    terminal: Mapping[tuple[int, StudyArm], _SnapshotEvidence],
) -> tuple[PairedObservation, ...]:
    observations = []
    for seed in SEEDS:
        dense = terminal[(seed, StudyArm.DENSE)]
        split90 = terminal[(seed, StudyArm.SPLIT90)]
        if set(dense.primary_pair_results) != set(
            split90.primary_pair_results
        ):
            raise ValueError(
                "terminal Dense/Split90 primary pair registries are crossed"
            )
        keys = tuple(sorted(dense.primary_pair_results))
        counts_by_cell: dict[tuple[str, str], int] = defaultdict(int)
        counts_by_world: dict[str, int] = defaultdict(int)
        for family, stratum, world_id, _pair_id in keys:
            counts_by_cell[(family, stratum)] += 1
            counts_by_world[world_id] += 1
        expected_cells = {
            (family.value, stratum.value)
            for family, stratum in PRIMARY_CELLS
        }
        if set(counts_by_cell) != expected_cells:
            raise ValueError(
                "terminal primary pair registry omits a frozen cell"
            )
        world_count = len(counts_by_world)
        for family, stratum, world_id, pair_id in keys:
            weight = Fraction(
                world_count * counts_by_world[world_id],
                len(PRIMARY_CELLS) * counts_by_cell[(family, stratum)],
            )
            observations.append(
                PairedObservation(
                    seed=seed,
                    world_id=world_id,
                    pair_id=f"{family}::{stratum}::{pair_id}",
                    treatment=(
                        weight
                        if split90.primary_pair_results[
                            (family, stratum, world_id, pair_id)
                        ]
                        else Fraction()
                    ),
                    control=(
                        weight
                        if dense.primary_pair_results[
                            (family, stratum, world_id, pair_id)
                        ]
                        else Fraction()
                    ),
                )
            )
    return tuple(observations)


def _holm_adjust_exact(
    p_values: Mapping[str, Fraction],
) -> Mapping[str, Fraction]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted = {}
    running = Fraction()
    count = len(ordered)
    for rank, (contrast_id, p_value) in enumerate(ordered):
        running = max(running, (count - rank) * p_value)
        adjusted[contrast_id] = min(Fraction(1), running)
    return MappingProxyType(
        {contrast_id: adjusted[contrast_id] for contrast_id in sorted(adjusted)}
    )


def _build_cohort_report(
    *,
    evidences: Sequence[_SnapshotEvidence],
    lock: StudyLockV3,
    lock_sha256: str,
) -> CohortReport:
    by_slot = {
        (
            evidence.binding.seed,
            evidence.binding.arm,
            evidence.binding.optimizer_step,
        ): evidence
        for evidence in evidences
    }
    if set(by_slot) != set(EXPECTED_STUDY_SLOTS_V3):
        raise ValueError("cohort evidence slots are not exactly complete")
    terminal = {
        (seed, arm): by_slot[(seed, arm, _TERMINAL_STEP)]
        for seed in SEEDS
        for arm in (StudyArm.DENSE, StudyArm.SPLIT90)
    }
    deltas = tuple(
        terminal[(seed, StudyArm.SPLIT90)].exact_primary_accuracy
        - terminal[(seed, StudyArm.DENSE)].exact_primary_accuracy
        for seed in SEEDS
    )
    primary_test = v3_exact_sign_flip_test(deltas)
    observations = _bootstrap_observations(terminal)
    bounds = v3_practical_equivalence_bounds(observations)
    if bounds.bootstrap.seed_effects != deltas:
        raise ValueError(
            "hierarchical bootstrap seed effects disagree with omnibus deltas"
        )
    effect_criterion = (
        primary_test.statistic > 0
        and primary_test.exact_p_value <= Fraction(1, 20)
    )
    equivalence_criterion = bounds.supports_equivalence

    aulc = []
    for seed in SEEDS:
        for arm in (StudyArm.DENSE, StudyArm.SPLIT90):
            points = tuple(
                (
                    step,
                    by_slot[(seed, arm, step)].exact_primary_accuracy,
                )
                for step in V3_SNAPSHOT_STEPS
            )
            raw_area = v3_right_step_aulc(points)
            aulc.append(
                {
                    "seed": seed,
                    "arm": arm.value,
                    "optimizer_steps": list(V3_SNAPSHOT_STEPS),
                    "points": [
                        {
                            "optimizer_step": step,
                            "primary_accuracy": _fraction_dict(value),
                        }
                        for step, value in points
                    ],
                    "interpolation": "none",
                    "integral": "right_step",
                    "raw_area": _fraction_dict(raw_area),
                    "normalized_area": _fraction_dict(
                        raw_area / _TERMINAL_STEP
                    ),
                    "primary": False,
                    "role": "required_secondary_trajectory_evidence",
                }
            )

    family_results = {}
    family_by_contrast = {
        SECONDARY_CONTRAST_IDS_V3[0]: ReasoningFamily.GRAPH,
        SECONDARY_CONTRAST_IDS_V3[1]: ReasoningFamily.NON_PATH,
    }
    for contrast_id, family in family_by_contrast.items():
        family_deltas = tuple(
            _family_accuracy(terminal[(seed, StudyArm.SPLIT90)], family)
            - _family_accuracy(terminal[(seed, StudyArm.DENSE)], family)
            for seed in SEEDS
        )
        result = v3_exact_sign_flip_test(family_deltas)
        family_results[contrast_id] = (family, family_deltas, result)
    raw_p_values = {
        contrast_id: result.exact_p_value
        for contrast_id, (_family, _deltas, result)
        in family_results.items()
    }
    adjusted = _holm_adjust_exact(raw_p_values)
    secondary = []
    for contrast_id in sorted(family_results):
        family, family_deltas, result = family_results[contrast_id]
        secondary.append(
            {
                "contrast_id": contrast_id,
                "family": family.value,
                "optimizer_step": _TERMINAL_STEP,
                "primary": False,
                "role": "nonprimary_secondary",
                "paired_seed_deltas": [
                    {"seed": seed, "delta": _fraction_dict(delta)}
                    for seed, delta in zip(SEEDS, family_deltas, strict=True)
                ],
                "exact_test": _exact_test_dict(result),
                "raw_p_value": _fraction_dict(raw_p_values[contrast_id]),
                "holm": {
                    "method": "holm",
                    "family_size": 2,
                    "tie_break": (
                        "stable_lexicographic_contrast_id_order"
                    ),
                    "adjusted_p_value": _fraction_dict(
                        adjusted[contrast_id]
                    ),
                    "alpha": _fraction_dict(Fraction(1, 20)),
                    "reject_null": adjusted[contrast_id] <= Fraction(1, 20),
                },
            }
        )

    instrument_gates = {
        "production_qualified_outputs": {
            "required": 100,
            "observed": len(evidences),
            "passed": len(evidences) == 100,
        },
        "stable_before_after_evaluator_identity": {
            "required": 100,
            "observed": len(evidences),
            "passed": len(evidences) == 100,
        },
        "required_control_coverage": {
            "required_controls": list(REQUIRED_CONTROL_IDS),
            "required_families": list(REQUIRED_FAMILIES),
            "required_strata": list(REQUIRED_STRATA),
            "passed": True,
        },
        "iid_guardrail_coverage": {
            "required_stratum": Stratum.IID.value,
            "passed": True,
        },
        "all_passed": True,
    }
    gates_pass = instrument_gates["all_passed"]
    supports_effect = bool(effect_criterion and gates_pass)
    supports_practical_null = bool(
        not supports_effect and equivalence_criterion and gates_pass
    )
    axes = classify_status(
        complete=True,
        valid=bool(gates_pass),
        observed_seeds=10,
        required_seeds=10,
        sign_consistent=(
            all(delta > 0 for delta in deltas)
            or all(delta < 0 for delta in deltas)
        ),
        supports_effect=supports_effect,
        supports_practical_null=supports_practical_null,
    )

    inputs = [
        {
            "slot_index": evidence.slot_index,
            "seed": evidence.binding.seed,
            "arm": evidence.binding.arm.value,
            "optimizer_step": evidence.binding.optimizer_step,
            "output_id": evidence.binding.output_id,
            "output_commitment": evidence.reference.output_commitment,
        }
        for evidence in evidences
    ]
    collection_receipts = [
        {
            "seed": lifecycle.seed,
            "uri": lifecycle.collection_receipt_s3_uri,
            "sha256": lifecycle.collection_receipt_sha256,
            "version_id": lifecycle.collection_receipt_s3_version_id,
        }
        for lifecycle in lock.seed_lifecycles
    ]
    snapshots = [
        {
            "slot_index": evidence.slot_index,
            "seed": evidence.binding.seed,
            "arm": evidence.binding.arm.value,
            "optimizer_step": evidence.binding.optimizer_step,
            "output_id": evidence.binding.output_id,
            "output_commitment": evidence.reference.output_commitment,
            "snapshot_sha256": evidence.binding.snapshot_sha256,
            "primary_cells": {
                cell_id: _fraction_dict(
                    evidence.primary_summary.primary_cells[
                        cell_id
                    ].exact_value
                )
                for cell_id in PRIMARY_CELL_IDS
            },
            "primary_accuracy": _fraction_dict(
                evidence.exact_primary_accuracy
            ),
        }
        for evidence in evidences
    ]
    primary = {
        "primary": True,
        "contrast_id": PRIMARY_CONTRAST_ID_V3,
        "estimand": "split90_minus_dense",
        "optimizer_step": _TERMINAL_STEP,
        "cell_weight": _fraction_dict(Fraction(1, 4)),
        "cell_ids": list(PRIMARY_CELL_IDS),
        "paired_seed_deltas": [
            {"seed": seed, "delta": _fraction_dict(delta)}
            for seed, delta in zip(SEEDS, deltas, strict=True)
        ],
        "exact_test": _exact_test_dict(primary_test),
        "bootstrap": {
            "bit_generator": "PCG64",
            "rng_seed": V3_BOOTSTRAP_RNG_SEED,
            "draws": V3_BOOTSTRAP_DRAWS,
            "confidence_percent": int(V3_BOOTSTRAP_CONFIDENCE * 100),
            "resampling_levels": [
                "seed",
                "world",
                "counterfactual_pair",
            ],
            "estimate": _fraction_dict(bounds.bootstrap.estimate),
            "ci_low": _fraction_dict(bounds.bootstrap.ci_low),
            "ci_high": _fraction_dict(bounds.bootstrap.ci_high),
            "margin": _fraction_dict(_EQUIVALENCE_MARGIN_EXACT),
            "strict_bounds": True,
            "supports_practical_equivalence": equivalence_criterion,
            "n_seeds": bounds.bootstrap.n_seeds,
            "n_worlds": bounds.bootstrap.n_worlds,
            "n_pairs": bounds.bootstrap.n_pairs,
        },
        "instrument_gates_required": True,
        "effect_criterion_passed": effect_criterion,
        "practical_equivalence_criterion_passed": equivalence_criterion,
        "supports_effect": supports_effect,
        "supports_practical_null": supports_practical_null,
    }
    if supports_effect and supports_practical_null:
        raise ValueError(
            "effect and practical-null conclusions must be exclusive"
        )
    return CohortReport(
        record_type=COHORT_REPORT_SCHEMA_V3,
        schema_version=STUDY_CONTRACT_VERSION,
        cohort_id="memorysplit-confirmatory-v3-360m-n10-aws",
        study_lock_sha256=lock_sha256,
        preregistration_sha256=lock.preregistration_sha256,
        sealed_evaluation_release_sha256=(
            lock.sealed_evaluation_release_sha256
        ),
        provider_selection=lock.provider_selection.to_dict(),
        inputs=tuple(inputs),
        collection_receipts=tuple(collection_receipts),
        snapshots=tuple(snapshots),
        primary=primary,
        aulc=tuple(aulc),
        secondary=tuple(secondary),
        instrument_gates=instrument_gates,
        status=axes.to_dict(),
    )


def aggregate_cohort_outputs(
    *,
    outputs: Sequence[SnapshotOutputReference],
    collection_receipts: Sequence[CollectionReceiptEvidence],
    expected_study_lock_sha256: str,
) -> CohortReport:
    """Authenticate and aggregate the exact receipts-proven 100-output cohort."""

    expected_lock = _sha256(
        expected_study_lock_sha256,
        "expected study-lock SHA-256",
    )
    if (
        isinstance(outputs, (str, bytes))
        or not isinstance(outputs, Sequence)
        or len(outputs) != len(EXPECTED_STUDY_SLOTS_V3)
        or any(
            not isinstance(value, SnapshotOutputReference)
            for value in outputs
        )
    ):
        raise ValueError(
            "cohort aggregation requires exactly 100 ordered output references"
        )
    references = tuple(outputs)
    commitments = tuple(
        reference.output_commitment for reference in references
    )
    if len(set(commitments)) != len(commitments):
        raise ValueError("cohort output commitments contain a duplicate")
    opened_outputs: list[_OpenedOutput] = []
    try:
        for reference in references:
            opened_outputs.append(_open_output(reference))
        directory_identities = tuple(
            opened.snapshot.identity[:2] for opened in opened_outputs
        )
        if len(set(directory_identities)) != len(directory_identities):
            raise ValueError(
                "cohort outputs must be independently published directories"
            )
        first_lock_content = opened_outputs[0].contents["study-lock.json"]
        if hashlib.sha256(first_lock_content).hexdigest() != expected_lock:
            raise ValueError(
                "cohort study lock disagrees with external commitment"
            )
        lock = StudyLockV3.from_dict(
            _canonical_object(first_lock_content, "study-lock.json")
        )
        plans = plan_snapshot_evaluations(
            study_lock=lock,
            study_lock_sha256=expected_lock,
            collection_receipts=collection_receipts,
        )
        release_cache: dict[
            tuple[str, str, str, str],
            _ReleaseData,
        ] = {}
        evidences = tuple(
            _validate_snapshot_output(
                slot_index=index,
                opened=opened,
                plan=plans[index],
                lock=lock,
                lock_sha256=expected_lock,
                release_cache=release_cache,
            )
            for index, opened in enumerate(opened_outputs)
        )
        report = _build_cohort_report(
            evidences=evidences,
            lock=lock,
            lock_sha256=expected_lock,
        )
        _run_aggregation_mutation_hook(
            "before_final_input_replay",
            outputs=tuple(opened_outputs),
            report=report,
        )
        for opened in opened_outputs:
            _replay_opened_output(opened)
        return report
    finally:
        for opened in reversed(opened_outputs):
            try:
                os.close(opened.descriptor)
            except OSError:
                pass


def _parse_cohort_report(
    value: CohortReport | Mapping[str, Any],
) -> CohortReport:
    if isinstance(value, CohortReport):
        return value
    if isinstance(value, Mapping):
        return CohortReport.from_dict(value)
    raise TypeError("cohort report must be CohortReport or a mapping")


def validate_cohort_report(
    report: CohortReport | Mapping[str, Any],
    *,
    outputs: Sequence[SnapshotOutputReference],
    collection_receipts: Sequence[CollectionReceiptEvidence],
    expected_study_lock_sha256: str,
) -> CohortReport:
    """Recompute every report field from descriptor-pinned source outputs."""

    provided = _parse_cohort_report(report)
    recomputed = aggregate_cohort_outputs(
        outputs=outputs,
        collection_receipts=collection_receipts,
        expected_study_lock_sha256=expected_study_lock_sha256,
    )
    if provided.canonical_bytes != recomputed.canonical_bytes:
        raise ValueError(
            "cohort report disagrees with recomputation from source outputs"
        )
    return recomputed


def _run_report_publication_mutation_hook(
    event: str,
    **context: object,
) -> None:
    """Deterministic race boundary; production intentionally performs no action."""

    del event, context


def _write_report_member(
    directory_fd: int,
    name: str,
    content: bytes,
) -> _PinnedFile:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        _write_all(descriptor, content)
        os.fchmod(descriptor, COHORT_REPORT_MEMBER_MODE)
        os.fsync(descriptor)
        details = os.fstat(descriptor)
        state = _regular_file_state(details, f"staged {name}")
        pinned = _PinnedFile(
            parent_fd=directory_fd,
            name=name,
            descriptor=descriptor,
            content=content,
            identity=(details.st_dev, details.st_ino),
            state=state,
        )
        _assert_report_member(pinned)
        return pinned
    except BaseException:
        os.close(descriptor)
        raise


def _assert_report_member(pinned: _PinnedFile) -> None:
    _assert_pinned_file(pinned, f"staged {pinned.name}")
    details = os.fstat(pinned.descriptor)
    if stat.S_IMODE(details.st_mode) != COHORT_REPORT_MEMBER_MODE:
        raise SealingError(f"staged {pinned.name} has an unsafe mode")
    observed = _read_descriptor(
        pinned.descriptor,
        details.st_size,
        f"staged {pinned.name}",
    )
    if observed != pinned.content:
        raise SealingError(
            f"staged {pinned.name} content verification failed"
        )
    _assert_pinned_file(pinned, f"staged {pinned.name}")


def _stage_report(
    output_fd: int,
    directory_name: str,
    content: bytes,
) -> _StagedRelease:
    staging_name, staging_fd = _make_staging(output_fd, directory_name)
    files: list[_PinnedFile] = []
    try:
        files.append(
            _write_report_member(
                staging_fd,
                COHORT_REPORT_FILE_NAME,
                content,
            )
        )
        for pinned in files:
            _assert_report_member(pinned)
        _assert_directory_entry(
            output_fd,
            staging_name,
            staging_fd,
            "private cohort-report staging",
        )
        os.fsync(staging_fd)
        snapshot = _capture_directory_snapshot(
            staging_fd,
            "private cohort-report staging",
            expected_names=(COHORT_REPORT_FILE_NAME,),
            exact_mode=0o700,
        )
        return _StagedRelease(staging_name, staging_fd, tuple(files), snapshot)
    except BaseException:
        for pinned in reversed(files):
            try:
                os.close(pinned.descriptor)
            except OSError:
                pass
        try:
            _quarantine_preserve_directory(
                output_fd,
                staging_name,
                staging_fd,
                label="failed private cohort-report staging",
            )
        finally:
            os.close(staging_fd)
        raise


def _verify_installed_report(
    output_fd: int,
    directory_name: str,
    staged: _StagedRelease,
) -> _DirectorySnapshot:
    _assert_directory_entry(
        output_fd,
        directory_name,
        staged.descriptor,
        "installed cohort report",
    )
    snapshot = _capture_directory_snapshot(
        staged.descriptor,
        "installed cohort report",
        expected_names=(COHORT_REPORT_FILE_NAME,),
        exact_mode=0o700,
    )
    for pinned in staged.files:
        _assert_report_member(pinned)
    os.fsync(staged.descriptor)
    _assert_directory_entry(
        output_fd,
        directory_name,
        staged.descriptor,
        "installed cohort report",
    )
    _assert_directory_snapshot(
        staged.descriptor,
        snapshot,
        "installed cohort report",
    )
    return snapshot


def publish_cohort_report(
    *,
    output_root: Path,
    report: CohortReport,
    outputs: Sequence[SnapshotOutputReference],
    collection_receipts: Sequence[CollectionReceiptEvidence],
    expected_study_lock_sha256: str,
) -> PublishedCohortReport:
    """Publish one content-addressed report with no-replace sealing authority."""

    typed = validate_cohort_report(
        report,
        outputs=outputs,
        collection_receipts=collection_receipts,
        expected_study_lock_sha256=expected_study_lock_sha256,
    )
    content = typed.canonical_bytes
    commitment = hashlib.sha256(content).hexdigest()
    directory_name = COHORT_REPORT_ID_PREFIX + commitment
    output_path, output_fd = _open_directory(
        output_root,
        "cohort report output root",
    )
    staged: _StagedRelease | None = None
    published = False
    try:
        _lock_output(output_fd)
        _reject_quarantine_blockers(output_fd)
        _reject_existing_release(output_fd, directory_name)
        staged = _stage_report(output_fd, directory_name, content)
        _assert_directory_path(
            output_path,
            output_fd,
            "cohort report output root",
        )
        _assert_directory_entry(
            output_fd,
            staged.name,
            staged.descriptor,
            "private cohort-report staging",
        )
        _assert_directory_snapshot(
            staged.descriptor,
            staged.snapshot,
            "private cohort-report staging",
        )
        for pinned in staged.files:
            _assert_report_member(pinned)
        _rename_noreplace_at(output_fd, staged.name, directory_name)
        installed_snapshot = _verify_installed_report(
            output_fd,
            directory_name,
            staged,
        )
        os.fsync(output_fd)
        result = PublishedCohortReport(
            report_dir=output_path / directory_name,
            report_path=(
                output_path / directory_name / COHORT_REPORT_FILE_NAME
            ),
            authoritative_commitment=commitment,
        )
        _run_report_publication_mutation_hook(
            "publish_before_final_binding",
            release_fd=staged.descriptor,
            parent_fd=output_fd,
            release_name=directory_name,
        )
        for pinned in staged.files:
            _assert_report_member(pinned)
        _assert_final_directory_binding(
            parent_path=output_path,
            parent_fd=output_fd,
            directory_name=directory_name,
            directory_fd=staged.descriptor,
            snapshot=installed_snapshot,
            label="installed cohort report",
        )
        published = True
        return result
    finally:
        if staged is not None:
            try:
                if not published:
                    _quarantine_staged_directory(output_fd, staged)
                    try:
                        os.fsync(output_fd)
                    except OSError:
                        pass
            finally:
                _close_staged(staged)
        os.close(output_fd)


__all__ = [
    "COHORT_REPORT_FILE_NAME",
    "COHORT_REPORT_ID_PREFIX",
    "COHORT_REPORT_MEMBER_MODE",
    "COHORT_REPORT_SCHEMA_V3",
    "CohortReport",
    "CollectionReceiptEvidence",
    "EVALUATOR_ENVIRONMENT_RECEIPT_FILE_NAME",
    "EVALUATOR_PROFILE_FILE_NAME",
    "EVALUATOR_RUNTIME_LOCK_FILE_NAME",
    "PATH_AUTHORITY",
    "PRIMARY_CONTRAST_ID_V3",
    "PRIMARY_TEST_METHOD_V3",
    "PublishedCohortReport",
    "RUN_BINDING_SCHEMA_V3",
    "RunBindingV3",
    "SECONDARY_CONTRAST_IDS_V3",
    "STUDY_LOCK_FILE_NAME",
    "SnapshotEvaluationPlan",
    "SnapshotOutputReference",
    "aggregate_cohort_outputs",
    "build_snapshot_evaluation_plans",
    "plan_snapshot_evaluations",
    "publish_cohort_report",
    "snapshot_path",
    "validate_cohort_report",
    "validate_snapshot_evaluation_plan",
    "validate_snapshot_evaluation_plans",
]
