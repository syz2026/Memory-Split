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
    canonical_sha256,
)


STUDY_LOCK_SCHEMA = "memorysplit.confirmatory.study-lock.v2"
VALIDITY_EVIDENCE_SCHEMA = "memorysplit.confirmatory.validity-evidence.v2"
FROZEN_PREREGISTRATION_SHA256 = (
    "fee38e363298d3def46b741320c9d7df4523d0ff3cd249187cf52d54046cbbf0"
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


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _schema(value: object, name: str) -> int:
    if type(value) is not int or value != CONTRACT_VERSION:
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
