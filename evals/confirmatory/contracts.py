"""Strict v2 contracts for confirmatory items, gold, stores, and checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, ClassVar

from evals.confirmatory.actions import ActionSlot, validate_action_slots
from msctl.aws_contracts import ARMS as STUDY_CONDITIONS
from msctl.aws_contracts import SEEDS as STUDY_SEEDS
from msctl.aws_contracts import SNAPSHOT_STEPS


CONTRACT_VERSION = 2
STUDY_CONTRACT_VERSION = 3
ITEM_SCHEMA = "memorysplit.confirmatory.item.v2"
SEALED_GOLD_SCHEMA = "memorysplit.confirmatory.sealed-gold.v2"
STORE_SCHEMA = "memorysplit.confirmatory.store.v2"
CHECKPOINT_SCHEMA = "memorysplit.confirmatory.checkpoint.v2"
STUDY_CHECKPOINT_SCHEMA = "memorysplit.confirmatory.checkpoint.v3"
STUDY_TARGETS_PER_UPDATE = 524_288

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class Stratum(StrEnum):
    IID = "iid"
    COMPOSITION_OOD = "composition_ood"
    LENGTH_OOD = "length_ood"
    JOINT_OOD = "joint_ood"
    COMPOSITION = "composition_ood"
    LENGTH = "length_ood"
    JOINT = "joint_ood"


class ReasoningFamily(StrEnum):
    GRAPH = "graph"
    NON_PATH = "non_path"


class Twin(StrEnum):
    ORIGINAL = "original"
    COUNTERFACTUAL = "counterfactual"


class CompositionSplit(StrEnum):
    SEEN = "seen"
    HELDOUT = "heldout"


class MemoryMode(StrEnum):
    MEMORY_OFF = "memory_off"
    MEMORY_ON = "memory_on"
    OFF = "memory_off"
    ON = "memory_on"


class Control(StrEnum):
    CORRECT = "correct"
    SHUFFLED_RETURNS = "shuffled_returns"
    SHUFFLED_MEMORY = "shuffled_returns"
    RELEVANT_EDGE = "relevant_edge"
    RELEVANT_EDGE_SWAP = "relevant_edge"
    IRRELEVANT_EDGE = "irrelevant_edge"
    IRRELEVANT_EDGE_SWAP = "irrelevant_edge"
    GOLD_PATH = "gold_path"
    GOLD_PATH_REPLAY = "gold_path"
    GOLD_RETURNS = "gold_returns"
    NO_QUERY = "no_query"
    EXPLICIT_MISS = "explicit_miss"
    HANDLE_SWAP = "handle_swap"
    ENTITY_RENAME = "entity_rename"
    ENTITY_RENAMING = "entity_rename"
    GRAPH_ISOMORPHISM = "graph_isomorphism"
    PAGE_ORDER_PERMUTATION = "page_order_permutation"


class Arm(StrEnum):
    DENSE = "dense"
    SPLIT = "split"
    RANDOM = "random"
    RANDOM_MASK = "random"


class StudyArm(StrEnum):
    DENSE = "dense"
    SPLIT90 = "split90"


class ConditionId(StrEnum):
    DENSE = "dense"
    SPLIT90 = "split90"
    RANDOM = "random"


def _strict_fields(
    raw: object,
    expected: frozenset[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in raw):
        raise ValueError(f"{name} keys must be strings")
    missing = expected - set(raw)
    unknown = set(raw) - expected
    if missing:
        raise ValueError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")
    return raw


def _string(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _schema_version(value: object, name: str) -> int:
    if type(value) is not int or value != CONTRACT_VERSION:
        raise ValueError(f"unsupported {name} schema_version")
    return value


def _study_schema_version(value: object, name: str) -> int:
    if type(value) is not int or value != STUDY_CONTRACT_VERSION:
        raise ValueError(f"unsupported {name} schema_version")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _enum(value: object, cls: type[StrEnum], name: str):
    if not isinstance(value, str):
        raise ValueError(f"{name} is not an approved value")
    try:
        return cls(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not an approved value") from exc


def validate_study_record_identity(
    *,
    seed: object,
    arm: object,
    condition_id: object,
    optimizer_step: object,
    raw_token_count: object,
) -> tuple[int, StudyArm, ConditionId, int, int]:
    """Validate one protected AWS N=10 arm/checkpoint identity."""

    if type(seed) is not int or seed not in STUDY_SEEDS:
        raise ValueError("study seed must be an exact integer from 0 to 9")
    typed_arm = _enum(arm, StudyArm, "arm")
    typed_condition = _enum(condition_id, ConditionId, "condition_id")
    if typed_condition.value not in STUDY_CONDITIONS:
        raise ValueError("study condition_id must be dense or split90")
    expected_arm = {
        ConditionId.DENSE: StudyArm.DENSE,
        ConditionId.SPLIT90: StudyArm.SPLIT90,
    }[typed_condition]
    if typed_arm is not expected_arm:
        raise ValueError("study condition_id disagrees with arm")
    if type(optimizer_step) is not int or optimizer_step not in SNAPSHOT_STEPS:
        raise ValueError("optimizer_step is not one of the five frozen steps")
    expected_tokens = optimizer_step * STUDY_TARGETS_PER_UPDATE
    if type(raw_token_count) is not int or raw_token_count != expected_tokens:
        raise ValueError(
            "raw_token_count must equal optimizer_step * 524288"
        )
    return (
        seed,
        typed_arm,
        typed_condition,
        optimizer_step,
        raw_token_count,
    )


def _canonical_value(value: Any, path: str = "$") -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _canonical_value(value.to_dict(), path)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path} contains a non-string key")
        return {
            key: _canonical_value(value[key], f"{path}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [
            _canonical_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{path} contains a non-canonical or non-finite value")


def canonical_json_bytes(value: Any) -> bytes:
    """Return one deterministic UTF-8 JSON record."""

    return (
        json.dumps(
            _canonical_value(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class StoreRow:
    source_id: str
    relation_id: str
    direction: str
    target_kind: str
    target: str
    qualifiers: Mapping[str, str]

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "source_id",
            "relation_id",
            "direction",
            "target_kind",
            "target",
            "qualifiers",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _string(self.source_id, "source_id"))
        object.__setattr__(
            self,
            "relation_id",
            _string(self.relation_id, "relation_id"),
        )
        if self.direction not in {"out", "in"}:
            raise ValueError("store row direction must be out or in")
        if self.target_kind not in {"entity", "literal"}:
            raise ValueError("store row target_kind must be entity or literal")
        object.__setattr__(
            self,
            "target",
            _string(self.target, "target", allow_empty=True),
        )
        if not isinstance(self.qualifiers, Mapping):
            raise ValueError("store row qualifiers must be an object")
        qualifiers = {
            _string(key, "qualifier key"): _string(
                value,
                f"qualifier {key}",
                allow_empty=True,
            )
            for key, value in self.qualifiers.items()
        }
        object.__setattr__(
            self,
            "qualifiers",
            MappingProxyType(dict(sorted(qualifiers.items()))),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StoreRow":
        value = _strict_fields(raw, cls.FIELDS, "StoreRow")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "relation_id": self.relation_id,
            "direction": self.direction,
            "target_kind": self.target_kind,
            "target": self.target,
            "qualifiers": dict(self.qualifiers),
        }

    @property
    def address(self) -> tuple[str, str, str]:
        return self.source_id, self.relation_id, self.direction


def store_content_sha256(
    store_id: str,
    world_id: str,
    rows: Sequence[StoreRow | Mapping[str, Any]],
) -> str:
    typed_rows = tuple(
        row if isinstance(row, StoreRow) else StoreRow.from_dict(row)
        for row in rows
    )
    return canonical_sha256(
        {
            "record_type": "memorysplit.confirmatory.store-content.v2",
            "schema_version": CONTRACT_VERSION,
            "store_id": _string(store_id, "store_id"),
            "world_id": _string(world_id, "world_id"),
            "rows": [row.to_dict() for row in typed_rows],
        }
    )


@dataclass(frozen=True)
class ItemRecord:
    record_type: str
    schema_version: int
    item_id: str
    pair_id: str
    twin: Twin
    stratum: Stratum
    family: ReasoningFamily
    world_id: str
    task: str
    path_length: int
    composition_split: CompositionSplit
    composition_id: str
    prompt: str
    initial_slots: tuple[str | None, ...]
    store_id: str
    memory_mode: MemoryMode
    control: Control

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "record_type",
            "schema_version",
            "item_id",
            "pair_id",
            "twin",
            "stratum",
            "family",
            "world_id",
            "task",
            "path_length",
            "composition_split",
            "composition_id",
            "prompt",
            "initial_slots",
            "store_id",
            "memory_mode",
            "control",
        }
    )

    def __post_init__(self) -> None:
        if self.record_type != ITEM_SCHEMA:
            raise ValueError(f"item record_type must be {ITEM_SCHEMA}")
        _schema_version(self.schema_version, "item")
        for field in (
            "item_id",
            "pair_id",
            "world_id",
            "task",
            "composition_id",
            "prompt",
            "store_id",
        ):
            object.__setattr__(self, field, _string(getattr(self, field), field))
        object.__setattr__(self, "twin", _enum(self.twin, Twin, "twin"))
        object.__setattr__(self, "stratum", _enum(self.stratum, Stratum, "stratum"))
        object.__setattr__(
            self,
            "family",
            _enum(self.family, ReasoningFamily, "family"),
        )
        object.__setattr__(
            self,
            "composition_split",
            _enum(
                self.composition_split,
                CompositionSplit,
                "composition_split",
            ),
        )
        object.__setattr__(
            self,
            "memory_mode",
            _enum(self.memory_mode, MemoryMode, "memory_mode"),
        )
        object.__setattr__(
            self,
            "control",
            _enum(self.control, Control, "control"),
        )
        object.__setattr__(
            self,
            "path_length",
            _integer(self.path_length, "path_length", minimum=2, maximum=10),
        )
        if (
            not isinstance(self.initial_slots, (list, tuple))
            or len(self.initial_slots) != 4
        ):
            raise ValueError("initial_slots must contain exactly four values")
        slots = tuple(
            None if value is None else _string(value, f"initial_slots[{index}]")
            for index, value in enumerate(self.initial_slots)
        )
        if all(value is None for value in slots):
            raise ValueError("initial_slots must bind at least one entity")
        object.__setattr__(self, "initial_slots", slots)

        short = 2 <= self.path_length <= 6
        seen = self.composition_split is CompositionSplit.SEEN
        expected = {
            (True, True): Stratum.IID,
            (True, False): Stratum.COMPOSITION_OOD,
            (False, True): Stratum.LENGTH_OOD,
            (False, False): Stratum.JOINT_OOD,
        }[(short, seen)]
        if self.stratum is not expected:
            raise ValueError(
                "stratum disagrees with path_length and composition_split"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ItemRecord":
        value = _strict_fields(raw, cls.FIELDS, "ItemRecord")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "item_id": self.item_id,
            "pair_id": self.pair_id,
            "twin": self.twin.value,
            "stratum": self.stratum.value,
            "family": self.family.value,
            "world_id": self.world_id,
            "task": self.task,
            "path_length": self.path_length,
            "composition_split": self.composition_split.value,
            "composition_id": self.composition_id,
            "prompt": self.prompt,
            "initial_slots": list(self.initial_slots),
            "store_id": self.store_id,
            "memory_mode": self.memory_mode.value,
            "control": self.control.value,
        }


@dataclass(frozen=True)
class SealedGoldRecord:
    record_type: str
    schema_version: int
    item_id: str
    pair_id: str
    twin: Twin
    answer: str
    proof: tuple[ActionSlot, ...]
    solver_id: str
    store_sha256: str

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "record_type",
            "schema_version",
            "item_id",
            "pair_id",
            "twin",
            "answer",
            "proof",
            "solver_id",
            "store_sha256",
        }
    )

    def __post_init__(self) -> None:
        if self.record_type != SEALED_GOLD_SCHEMA:
            raise ValueError(f"gold record_type must be {SEALED_GOLD_SCHEMA}")
        _schema_version(self.schema_version, "sealed-gold")
        for field in ("item_id", "pair_id", "solver_id"):
            object.__setattr__(self, field, _string(getattr(self, field), field))
        object.__setattr__(
            self,
            "answer",
            _string(self.answer, "answer", allow_empty=True),
        )
        object.__setattr__(self, "twin", _enum(self.twin, Twin, "twin"))
        if not isinstance(self.proof, (list, tuple)) or len(self.proof) != 12:
            raise ValueError("sealed proof must contain exactly 12 action slots")
        proof = validate_action_slots(self.proof)
        object.__setattr__(self, "proof", proof)
        object.__setattr__(
            self,
            "store_sha256",
            _sha256(self.store_sha256, "store_sha256"),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SealedGoldRecord":
        value = _strict_fields(raw, cls.FIELDS, "SealedGoldRecord")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "item_id": self.item_id,
            "pair_id": self.pair_id,
            "twin": self.twin.value,
            "answer": self.answer,
            "proof": [action.to_dict() for action in self.proof],
            "solver_id": self.solver_id,
            "store_sha256": self.store_sha256,
        }


@dataclass(frozen=True)
class StoreRecord:
    record_type: str
    schema_version: int
    store_id: str
    world_id: str
    rows: tuple[StoreRow, ...]
    content_sha256: str

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "record_type",
            "schema_version",
            "store_id",
            "world_id",
            "rows",
            "content_sha256",
        }
    )

    def __post_init__(self) -> None:
        if self.record_type != STORE_SCHEMA:
            raise ValueError(f"store record_type must be {STORE_SCHEMA}")
        _schema_version(self.schema_version, "store")
        object.__setattr__(self, "store_id", _string(self.store_id, "store_id"))
        object.__setattr__(self, "world_id", _string(self.world_id, "world_id"))
        if not isinstance(self.rows, (list, tuple)) or not self.rows:
            raise ValueError("store rows must be a non-empty sequence")
        rows = tuple(
            row if isinstance(row, StoreRow) else StoreRow.from_dict(row)
            for row in self.rows
        )
        addresses = [row.address for row in rows]
        if len(addresses) != len(set(addresses)):
            raise ValueError("store rows contain a duplicate address")
        object.__setattr__(self, "rows", rows)
        claimed = _sha256(self.content_sha256, "content_sha256")
        expected = store_content_sha256(self.store_id, self.world_id, rows)
        if claimed != expected:
            raise ValueError("store content_sha256 does not match its content")
        object.__setattr__(self, "content_sha256", claimed)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StoreRecord":
        value = _strict_fields(raw, cls.FIELDS, "StoreRecord")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "store_id": self.store_id,
            "world_id": self.world_id,
            "rows": [row.to_dict() for row in self.rows],
            "content_sha256": self.content_sha256,
        }

    def lookup(
        self,
        source_id: str,
        relation_id: str,
        direction: str,
    ) -> StoreRow | None:
        address = source_id, relation_id, direction
        return next((row for row in self.rows if row.address == address), None)


@dataclass(frozen=True)
class CheckpointRecord:
    record_type: str
    schema_version: int
    checkpoint_sha256: str
    model_id: str
    arm: Arm
    condition_id: ConditionId
    seed: int
    raw_token_count: int
    configuration_sha256: str
    route_dose_sha256: str
    corpus_sha256: str
    code_sha256: str

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "record_type",
            "schema_version",
            "checkpoint_sha256",
            "model_id",
            "arm",
            "condition_id",
            "seed",
            "raw_token_count",
            "configuration_sha256",
            "route_dose_sha256",
            "corpus_sha256",
            "code_sha256",
        }
    )

    def __post_init__(self) -> None:
        if self.record_type != CHECKPOINT_SCHEMA:
            raise ValueError(
                f"checkpoint record_type must be {CHECKPOINT_SCHEMA}"
            )
        _schema_version(self.schema_version, "checkpoint")
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _sha256(self.checkpoint_sha256, "checkpoint_sha256"),
        )
        object.__setattr__(self, "model_id", _string(self.model_id, "model_id"))
        object.__setattr__(self, "arm", _enum(self.arm, Arm, "arm"))
        object.__setattr__(
            self,
            "condition_id",
            _enum(self.condition_id, ConditionId, "condition_id"),
        )
        expected_arm = {
            ConditionId.DENSE: Arm.DENSE,
            ConditionId.SPLIT90: Arm.SPLIT,
            ConditionId.RANDOM: Arm.RANDOM,
        }[self.condition_id]
        if self.arm is not expected_arm:
            raise ValueError("checkpoint condition_id disagrees with arm")
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        object.__setattr__(
            self,
            "raw_token_count",
            _integer(self.raw_token_count, "raw_token_count", minimum=1),
        )
        for field in (
            "configuration_sha256",
            "route_dose_sha256",
            "corpus_sha256",
            "code_sha256",
        ):
            object.__setattr__(
                self,
                field,
                _sha256(getattr(self, field), field),
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CheckpointRecord":
        value = _strict_fields(raw, cls.FIELDS, "CheckpointRecord")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_id": self.model_id,
            "arm": self.arm.value,
            "condition_id": self.condition_id.value,
            "seed": self.seed,
            "raw_token_count": self.raw_token_count,
            "configuration_sha256": self.configuration_sha256,
            "route_dose_sha256": self.route_dose_sha256,
            "corpus_sha256": self.corpus_sha256,
            "code_sha256": self.code_sha256,
        }


@dataclass(frozen=True)
class StudyCheckpointRecord:
    """Step-aware v3 checkpoint while semantic evaluation records stay v2."""

    record_type: str
    schema_version: int
    checkpoint_sha256: str
    model_id: str
    arm: StudyArm
    condition_id: ConditionId
    seed: int
    optimizer_step: int
    raw_token_count: int
    configuration_sha256: str
    route_dose_sha256: str
    corpus_sha256: str
    code_sha256: str

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "record_type",
            "schema_version",
            "checkpoint_sha256",
            "model_id",
            "arm",
            "condition_id",
            "seed",
            "optimizer_step",
            "raw_token_count",
            "configuration_sha256",
            "route_dose_sha256",
            "corpus_sha256",
            "code_sha256",
        }
    )

    def __post_init__(self) -> None:
        if self.record_type != STUDY_CHECKPOINT_SCHEMA:
            raise ValueError(
                "study checkpoint record_type must be "
                f"{STUDY_CHECKPOINT_SCHEMA}"
            )
        _study_schema_version(self.schema_version, "study checkpoint")
        (
            seed,
            arm,
            condition_id,
            optimizer_step,
            raw_token_count,
        ) = validate_study_record_identity(
            seed=self.seed,
            arm=self.arm,
            condition_id=self.condition_id,
            optimizer_step=self.optimizer_step,
            raw_token_count=self.raw_token_count,
        )
        legacy_arm = {
            StudyArm.DENSE: Arm.DENSE,
            StudyArm.SPLIT90: Arm.SPLIT,
        }[arm]
        base = CheckpointRecord(
            record_type=CHECKPOINT_SCHEMA,
            schema_version=CONTRACT_VERSION,
            checkpoint_sha256=self.checkpoint_sha256,
            model_id=self.model_id,
            arm=legacy_arm,
            condition_id=condition_id,
            seed=seed,
            raw_token_count=raw_token_count,
            configuration_sha256=self.configuration_sha256,
            route_dose_sha256=self.route_dose_sha256,
            corpus_sha256=self.corpus_sha256,
            code_sha256=self.code_sha256,
        )
        for field in (
            "checkpoint_sha256",
            "model_id",
            "configuration_sha256",
            "route_dose_sha256",
            "corpus_sha256",
            "code_sha256",
        ):
            object.__setattr__(self, field, getattr(base, field))
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "condition_id", condition_id)
        object.__setattr__(self, "optimizer_step", optimizer_step)
        object.__setattr__(self, "raw_token_count", raw_token_count)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StudyCheckpointRecord":
        value = _strict_fields(raw, cls.FIELDS, "StudyCheckpointRecord")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_id": self.model_id,
            "arm": self.arm.value,
            "condition_id": self.condition_id.value,
            "seed": self.seed,
            "optimizer_step": self.optimizer_step,
            "raw_token_count": self.raw_token_count,
            "configuration_sha256": self.configuration_sha256,
            "route_dose_sha256": self.route_dose_sha256,
            "corpus_sha256": self.corpus_sha256,
            "code_sha256": self.code_sha256,
        }


@dataclass(frozen=True)
class ContractBundle:
    item: ItemRecord
    gold: SealedGoldRecord
    store: StoreRecord
    checkpoint: CheckpointRecord


def validate_contract_bundle(
    item: ItemRecord,
    gold: SealedGoldRecord,
    store: StoreRecord,
    checkpoint: CheckpointRecord,
) -> ContractBundle:
    """Authenticate cross-record identities without exposing gold to the model."""

    if not isinstance(item, ItemRecord):
        raise TypeError("contract bundle item must be an ItemRecord")
    if not isinstance(gold, SealedGoldRecord):
        raise TypeError("contract bundle gold must be a SealedGoldRecord")
    if not isinstance(store, StoreRecord):
        raise TypeError("contract bundle store must be a StoreRecord")
    if not isinstance(checkpoint, CheckpointRecord):
        raise TypeError("contract bundle checkpoint must be a CheckpointRecord")
    ItemRecord.from_dict(item.to_dict())
    SealedGoldRecord.from_dict(gold.to_dict())
    StoreRecord.from_dict(store.to_dict())
    CheckpointRecord.from_dict(checkpoint.to_dict())

    if (
        item.item_id,
        item.pair_id,
        item.twin,
    ) != (
        gold.item_id,
        gold.pair_id,
        gold.twin,
    ):
        raise ValueError("item and sealed-gold identity mismatch")
    if item.store_id != store.store_id:
        raise ValueError("item store binding mismatch")
    if item.world_id != store.world_id:
        raise ValueError("item and store world binding mismatch")
    if gold.store_sha256 != store.content_sha256:
        raise ValueError("sealed-gold store hash binding mismatch")
    return ContractBundle(item, gold, store, checkpoint)
