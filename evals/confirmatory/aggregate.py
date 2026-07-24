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
    SeedLifecycleBinding,
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


__all__ = [
    "EVALUATOR_ENVIRONMENT_RECEIPT_FILE_NAME",
    "EVALUATOR_PROFILE_FILE_NAME",
    "EVALUATOR_RUNTIME_LOCK_FILE_NAME",
    "CollectionReceiptEvidence",
    "RUN_BINDING_SCHEMA_V3",
    "RunBindingV3",
    "STUDY_LOCK_FILE_NAME",
    "SnapshotEvaluationPlan",
    "build_snapshot_evaluation_plans",
    "plan_snapshot_evaluations",
    "snapshot_path",
    "validate_snapshot_evaluation_plan",
    "validate_snapshot_evaluation_plans",
]
