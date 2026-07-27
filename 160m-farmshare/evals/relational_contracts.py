"""Strict, versioned result contracts for relational evaluation."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import secrets
import sqlite3
import stat
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

from evals.relational_controls import ControlID, EvalMode
from evals.scorers import normalize_answer


SCHEMA_VERSION = 1
SCHEMA_ID = "relational-result-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TASKS = {
    "path_composition",
    "date_ordering",
    "balanced_equality",
    "factual_recall",
}
_VARIANTS = {"original", "counterfactual"}
_ARMS = {"dense", "split", "random", "selective"}
_COMPOSITION_SPLITS = {"seen", "heldout"}
_ORACLE_EFFECTS = {"unchanged", "changed", "miss"}
_MILESTONE_NAMES = {
    f"{task}_pair_accuracy_0.75"
    for task in (
        "path_composition",
        "date_ordering",
        "balanced_equality",
    )
}
_Z_975 = 1.959963984540054
_VALIDATED_CONSTRUCTION: ContextVar[frozenset[type]] = ContextVar(
    "relational_validated_construction",
    default=frozenset(),
)


def _construct_validated(cls, **values):
    active = _VALIDATED_CONSTRUCTION.get()
    token = _VALIDATED_CONSTRUCTION.set(active | {cls})
    try:
        return cls(**values)
    finally:
        _VALIDATED_CONSTRUCTION.reset(token)


def _validate_typed_instance(instance) -> None:
    cls = type(instance)
    if cls in _VALIDATED_CONSTRUCTION.get():
        return
    validated = cls.from_dict(instance.to_dict())
    for field in fields(instance):
        object.__setattr__(
            instance,
            field.name,
            getattr(validated, field.name),
        )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _deep_freeze(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _deep_thaw(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


_RATE_FIELDS = ("value", "numerator", "denominator")
_SLICE_FIELDS = ("item_accuracy", "pair_accuracy", "exact_action_path")
_PER_HOP_FIELDS = ("relation", "direction", "action", "referent")
_METRIC_FIELDS = (
    "item_accuracy",
    "pair_accuracy",
    "all_six_action_exact",
    "exact_action_path",
    "answer_given_correct_retrieval",
    "gold_path_answer_accuracy",
    "malformed_rate",
    "miss_rate",
    "abstention_rate",
    "excess_read_rate",
    "per_hop",
    "by_hop",
    "by_composition",
    "by_task",
    "store",
    "edit_locality",
    "milestone_crossings",
)

EVAL_ROW_FIELDS = (
    "record_type",
    "schema_version",
    "qid",
    "pair_id",
    "variant",
    "task",
    "world_id",
    "provenance_id",
    "relation_path_hash",
    "template_id",
    "composition_split",
    "hop",
    "seed",
    "model_id",
    "arm",
    "checkpoint_sha256",
    "raw_token_count",
    "memory_mode",
    "control_id",
    "evaluator_sha256",
    "data_sha256",
    "relation_schema_sha256",
    "configuration_sha256",
    "result_schema_sha256",
    "provenance_sha256",
    "cluster_id",
    "prediction",
    "answer",
    "correct",
    "prediction_source",
    "all_actions",
    "gold_all_actions",
    "returned_addresses",
    "gold_addresses",
    "correct_referents",
    "misses",
    "malformed",
    "abstained",
    "excess_reads",
    "halt_step",
    "answer_logits",
    "lookup_latency_ns",
    "lookup_count",
    "store_rows",
    "store_bytes",
    "control_seed",
    "transformation_id",
    "source_store_sha256",
    "transformed_store_sha256",
    "transformation_metadata_sha256",
    "changed_addresses",
    "oracle_before",
    "oracle_after",
    "oracle_effect",
    "edit_locality_correct",
)

CHECKPOINT_SUMMARY_FIELDS = (
    "record_type",
    "schema_version",
    "checkpoint_sha256",
    "model_id",
    "arm",
    "seed",
    "raw_token_count",
    "memory_mode",
    "control_id",
    "evaluator_sha256",
    "data_sha256",
    "relation_schema_sha256",
    "configuration_sha256",
    "result_schema_sha256",
    "provenance_sha256",
    "rows_sha256",
    "n_rows",
    "n_pairs",
    "metrics",
)

GUARDRAIL_REPORT_FIELDS = (
    "record_type",
    "schema_version",
    "split_checkpoint_sha256",
    "dense_checkpoint_sha256",
    "model_id",
    "seed",
    "raw_token_count",
    "evaluator_sha256",
    "data_sha256",
    "relation_schema_sha256",
    "split_configuration_sha256",
    "dense_configuration_sha256",
    "result_schema_sha256",
    "split_result_provenance_sha256",
    "dense_result_provenance_sha256",
    "study_provenance_sha256",
    "pairing_receipt_sha256",
    "split_guardrail_source_sha256",
    "dense_guardrail_source_sha256",
    "split_matrix_manifest_sha256",
    "dense_matrix_manifest_sha256",
    "guards",
    "confirmatory_passed",
)

GUARD_CHECK_FIELDS = (
    "check_id",
    "value",
    "reference_value",
    "threshold",
    "comparison",
    "passed",
    "numerator",
    "denominator",
)

_GUARD_CHECKS = {
    "factual_job": (
        "split_on_recall_floor",
        "split_on_recall_noninferiority",
    ),
    "split_off_leakage": (
        "split_off_recall",
        "split_off_recognition_wilson_hi",
    ),
    "retrieval_procedure": (
        "split_off_first_hop",
        "gold_return_path_noninferiority",
    ),
    "relation_rule": ("rule_noninferiority",),
    "natural_text": ("bpb_noninferiority",),
    "instrument_integrity": ("integrity_conjunction",),
}
_GUARD_RULES = {
    "split_on_recall_floor": (">=", 0.95, None),
    "split_on_recall_noninferiority": (">=", None, ("offset", -0.02)),
    "split_off_recall": ("<", 0.05, None),
    "split_off_recognition_wilson_hi": ("<", 0.30, None),
    "split_off_first_hop": (">=", 0.75, None),
    "gold_return_path_noninferiority": (">=", None, ("offset", -0.05)),
    "rule_noninferiority": (">=", None, ("offset", -0.02)),
    "bpb_noninferiority": ("<=", None, ("scale", 1.01)),
    "integrity_conjunction": ("==", True, None),
}


def _strict_fields(
    value: object,
    fields: Sequence[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    actual = set(value)
    expected = set(fields)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ValueError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _string(value: object, name: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        suffix = "non-empty " if nonempty else ""
        raise ValueError(f"{name} must be a {suffix}string")
    return value


def _enum(value: object, allowed: set[str], name: str) -> str:
    result = _string(value, name)
    if result not in allowed:
        raise ValueError(f"{name} is not an approved enum value")
    return result


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
        bounds = (
            f"[{minimum}, {maximum}]"
            if maximum is not None
            else f">= {minimum}"
        )
        raise ValueError(f"{name} must be {bounds}")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be Boolean")
    return value


def _number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _nullable_string(value: object, name: str) -> str | None:
    return None if value is None else _string(value, name)


def _address(value: object, name: str) -> tuple[int, str, str]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
    ):
        raise ValueError(f"{name} must contain source, relation, direction")
    source = _integer(value[0], f"{name}.source_id")
    relation = _string(value[1], f"{name}.relation_id")
    direction = _enum(value[2], {"out", "in"}, f"{name}.direction")
    return source, relation, direction


def _addresses(
    value: object,
    name: str,
    *,
    expected_length: int | None = None,
    unique: bool = False,
) -> tuple[tuple[int, str, str], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an array")
    if expected_length is not None and len(value) != expected_length:
        raise ValueError(
            f"{name} must contain exactly {expected_length} addresses"
        )
    result = tuple(
        _address(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate addresses")
    return result


def _nullable_addresses(
    value: object,
    name: str,
    *,
    expected_length: int,
) -> tuple[tuple[int, str, str] | None, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != expected_length:
        raise ValueError(
            f"{name} must contain exactly {expected_length} entries"
        )
    return tuple(
        None if item is None else _address(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def _action(value: object, name: str) -> tuple[int, str, str, bool, bool]:
    if not isinstance(value, (list, tuple)) or len(value) != 5:
        raise ValueError(f"{name} must contain five action fields")
    source_slot = _integer(
        value[0],
        f"{name}.source_slot",
        maximum=3,
    )
    relation = _string(value[1], f"{name}.relation_id")
    direction = _enum(value[2], {"out", "in"}, f"{name}.direction")
    read = _boolean(value[3], f"{name}.read")
    halt = _boolean(value[4], f"{name}.halt")
    if read and halt:
        raise ValueError(f"{name} cannot READ and HALT")
    return source_slot, relation, direction, read, halt


def _actions(
    value: object,
    name: str,
) -> tuple[tuple[int, str, str, bool, bool], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise ValueError(f"{name} must contain exactly six action slots")
    return tuple(
        _action(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def _validate_trace(
    actions: tuple[tuple[int, str, str, bool, bool], ...],
    name: str,
    *,
    gold: bool,
) -> int | None:
    halt_positions = [
        index for index, action in enumerate(actions) if action[4]
    ]
    if len(halt_positions) > 1:
        raise ValueError(f"{name} allows at most one HALT")
    halt_step = halt_positions[0] + 1 if halt_positions else None
    if halt_positions:
        halt = halt_positions[0]
        if any(
            action[3] or action[4] for action in actions[halt + 1 :]
        ):
            raise ValueError(f"{name} must contain NOOPs after HALT")
        if gold and (
            halt == 0
            or any(
                not action[3] or action[4] for action in actions[:halt]
            )
        ):
            raise ValueError(f"{name} must contain reads before HALT")
    elif gold and any(not action[3] for action in actions):
        raise ValueError(f"halt-free {name} must contain six reads")
    return halt_step


def cluster_id_for(
    *,
    seed: int,
    world_id: int,
    relation_path_hash: str,
    template_id: str,
) -> str:
    seed = _integer(seed, "seed")
    world_id = _integer(world_id, "world_id")
    relation_path_hash = _sha256(
        relation_path_hash,
        "relation_path_hash",
    )
    template_id = _string(template_id, "template_id")
    return hashlib.sha256(
        canonical_json_bytes(
            [seed, world_id, relation_path_hash, template_id]
        )
    ).hexdigest()


@dataclass(frozen=True)
class EvalRow:
    record_type: str
    schema_version: int
    qid: str
    pair_id: str
    variant: str
    task: str
    world_id: int
    provenance_id: str
    relation_path_hash: str
    template_id: str
    composition_split: str
    hop: int
    seed: int
    model_id: str
    arm: str
    checkpoint_sha256: str
    raw_token_count: int
    memory_mode: str
    control_id: str
    evaluator_sha256: str
    data_sha256: str
    relation_schema_sha256: str
    configuration_sha256: str
    result_schema_sha256: str
    provenance_sha256: str
    cluster_id: str
    prediction: str
    answer: str
    correct: bool
    prediction_source: str
    all_actions: tuple[tuple[int, str, str, bool, bool], ...]
    gold_all_actions: tuple[tuple[int, str, str, bool, bool], ...]
    returned_addresses: tuple[tuple[int, str, str] | None, ...]
    gold_addresses: tuple[tuple[int, str, str], ...]
    correct_referents: tuple[bool, ...]
    misses: int
    malformed: int
    abstained: bool
    excess_reads: int
    halt_step: int | None
    answer_logits: tuple[tuple[float, ...], ...]
    lookup_latency_ns: int
    lookup_count: int
    store_rows: int
    store_bytes: int
    control_seed: int
    transformation_id: str | None
    source_store_sha256: str | None
    transformed_store_sha256: str | None
    transformation_metadata_sha256: str | None
    changed_addresses: tuple[tuple[int, str, str], ...]
    oracle_before: str
    oracle_after: str | None
    oracle_effect: str
    edit_locality_correct: bool | None

    FIELDS: ClassVar[tuple[str, ...]] = EVAL_ROW_FIELDS

    def __post_init__(self) -> None:
        _validate_typed_instance(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvalRow":
        value = _strict_fields(raw, cls.FIELDS, "EvalRow")
        if value["record_type"] != "eval_row":
            raise ValueError("EvalRow record_type must be eval_row")
        version = _integer(value["schema_version"], "schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError("unsupported EvalRow schema_version")
        qid = _string(value["qid"], "qid")
        pair_id = _string(value["pair_id"], "pair_id")
        variant = _enum(value["variant"], _VARIANTS, "variant")
        task = _enum(value["task"], _TASKS, "task")
        world_id = _integer(value["world_id"], "world_id")
        provenance_id = _string(value["provenance_id"], "provenance_id")
        relation_path_hash = _sha256(
            value["relation_path_hash"],
            "relation_path_hash",
        )
        template_id = _string(value["template_id"], "template_id")
        composition_split = _enum(
            value["composition_split"],
            _COMPOSITION_SPLITS,
            "composition_split",
        )
        hop = _integer(value["hop"], "hop", minimum=1, maximum=6)
        seed = _integer(value["seed"], "seed")
        model_id = _string(value["model_id"], "model_id")
        arm = _enum(value["arm"], _ARMS, "arm")
        checkpoint_sha256 = _sha256(
            value["checkpoint_sha256"],
            "checkpoint_sha256",
        )
        raw_token_count = _integer(
            value["raw_token_count"],
            "raw_token_count",
        )
        memory_mode = _enum(
            value["memory_mode"],
            {mode.value for mode in EvalMode},
            "memory_mode",
        )
        control_id = _enum(
            value["control_id"],
            {control.value for control in ControlID},
            "control_id",
        )
        hashes = {
            name: _sha256(value[name], name)
            for name in (
                "evaluator_sha256",
                "data_sha256",
                "relation_schema_sha256",
                "configuration_sha256",
                "result_schema_sha256",
                "provenance_sha256",
            )
        }
        cluster_id = _sha256(value["cluster_id"], "cluster_id")
        expected_cluster = cluster_id_for(
            seed=seed,
            world_id=world_id,
            relation_path_hash=relation_path_hash,
            template_id=template_id,
        )
        if cluster_id != expected_cluster:
            raise ValueError("cluster_id does not match cluster metadata")

        prediction = _string(value["prediction"], "prediction", nonempty=False)
        answer = _string(value["answer"], "answer", nonempty=False)
        correct = _boolean(value["correct"], "correct")
        if correct != (
            normalize_answer(prediction) == normalize_answer(answer)
        ):
            raise ValueError("correct disagrees with prediction and answer")
        if value["prediction_source"] != "model":
            raise ValueError("prediction_source must be model")
        all_actions = _actions(value["all_actions"], "all_actions")
        gold_all_actions = _actions(
            value["gold_all_actions"],
            "gold_all_actions",
        )
        actual_halt = _validate_trace(
            all_actions,
            "all_actions",
            gold=False,
        )
        _validate_trace(
            gold_all_actions,
            "gold_all_actions",
            gold=True,
        )
        if sum(action[3] for action in gold_all_actions) != hop:
            raise ValueError("hop must equal the gold READ count")
        returned_addresses = _nullable_addresses(
            value["returned_addresses"],
            "returned_addresses",
            expected_length=6,
        )
        for index, (action, address) in enumerate(
            zip(all_actions, returned_addresses)
        ):
            if not action[3] and address is not None:
                raise ValueError(
                    f"returned_addresses[{index}] must be null for non-read"
                )
            if address is not None and (
                address[1] != action[1] or address[2] != action[2]
            ):
                raise ValueError(
                    "returned address relation/direction mismatch"
                )
        gold_addresses = _addresses(
            value["gold_addresses"],
            "gold_addresses",
            expected_length=hop,
        )
        gold_reads = [
            action for action in gold_all_actions if action[3]
        ]
        if any(
            address[1] != action[1] or address[2] != action[2]
            for action, address in zip(gold_reads, gold_addresses)
        ):
            raise ValueError(
                "gold addresses do not match gold action relations"
            )
        referents_raw = value["correct_referents"]
        if not isinstance(referents_raw, (list, tuple)) or len(
            referents_raw
        ) != hop:
            raise ValueError(
                "correct_referents must contain one Boolean per hop"
            )
        correct_referents = tuple(
            _boolean(item, f"correct_referents[{index}]")
            for index, item in enumerate(referents_raw)
        )
        misses = _integer(value["misses"], "misses", maximum=6)
        read_attempts = sum(action[3] for action in all_actions)
        expected_misses = sum(
            action[3] and address is None
            for action, address in zip(all_actions, returned_addresses)
        )
        if misses != expected_misses:
            raise ValueError("misses disagree with returned read addresses")
        malformed = _integer(
            value["malformed"],
            "malformed",
            maximum=6,
        )
        abstained = _boolean(value["abstained"], "abstained")
        if abstained != (not bool(prediction.strip())):
            raise ValueError("abstained disagrees with prediction")
        excess_reads = _integer(
            value["excess_reads"],
            "excess_reads",
            maximum=6,
        )
        if excess_reads != max(0, read_attempts - hop):
            raise ValueError("excess_reads disagrees with action trace")
        predicted_read_addresses = [
            address
            for action, address in zip(all_actions, returned_addresses)
            if action[3]
        ]
        expected_referents = tuple(
            (
                predicted_read_addresses[index] == address
                if index < len(predicted_read_addresses)
                else False
            )
            for index, address in enumerate(gold_addresses)
        )
        if correct_referents != expected_referents:
            raise ValueError(
                "correct_referents disagree with returned/gold addresses"
            )
        halt_value = value["halt_step"]
        halt_step = (
            None
            if halt_value is None
            else _integer(
                halt_value,
                "halt_step",
                minimum=1,
                maximum=6,
            )
        )
        if halt_step != actual_halt:
            raise ValueError("halt_step does not match all_actions")
        logits_raw = value["answer_logits"]
        if not isinstance(logits_raw, (list, tuple)) or len(logits_raw) != 6:
            raise ValueError("answer_logits must contain six token sequences")
        answer_logits: list[tuple[float, ...]] = []
        for step, token_values in enumerate(logits_raw):
            if not isinstance(token_values, (list, tuple)) or not token_values:
                raise ValueError(
                    "answer_logits entries must be non-empty arrays"
                )
            answer_logits.append(
                tuple(
                    _number(
                        token_value,
                        f"answer_logits[{step}][{token}]",
                    )
                    for token, token_value in enumerate(token_values)
                )
            )
        lookup_latency_ns = _integer(
            value["lookup_latency_ns"],
            "lookup_latency_ns",
        )
        lookup_count = _integer(
            value["lookup_count"],
            "lookup_count",
            maximum=6,
        )
        if lookup_count > read_attempts:
            raise ValueError("lookup_count cannot exceed read attempts")
        if memory_mode == EvalMode.MEMORY_OFF.value and lookup_count != 0:
            raise ValueError("memory-off lookup_count must be zero")
        if lookup_count == 0 and lookup_latency_ns != 0:
            raise ValueError(
                "lookup latency must be zero when lookup_count is zero"
            )
        store_rows = _integer(value["store_rows"], "store_rows")
        store_bytes = _integer(value["store_bytes"], "store_bytes")
        control_seed = _integer(value["control_seed"], "control_seed")
        transformation_id = (
            None
            if value["transformation_id"] is None
            else _sha256(
                value["transformation_id"],
                "transformation_id",
            )
        )
        transformation_commitments = {
            name: (
                None
                if value[name] is None
                else _sha256(value[name], name)
            )
            for name in (
                "source_store_sha256",
                "transformed_store_sha256",
                "transformation_metadata_sha256",
            )
        }
        changed_addresses = _addresses(
            value["changed_addresses"],
            "changed_addresses",
            unique=True,
        )
        oracle_before = _string(value["oracle_before"], "oracle_before")
        oracle_after = _nullable_string(
            value["oracle_after"],
            "oracle_after",
        )
        oracle_effect = _enum(
            value["oracle_effect"],
            _ORACLE_EFFECTS,
            "oracle_effect",
        )
        expected_effect = (
            "miss"
            if oracle_after is None
            else "unchanged"
            if oracle_after == oracle_before
            else "changed"
        )
        if oracle_effect != expected_effect:
            raise ValueError("oracle_effect disagrees with oracle values")
        locality = value["edit_locality_correct"]
        edit_locality_correct = (
            None
            if locality is None
            else _boolean(locality, "edit_locality_correct")
        )
        changed_count_rules = {
            ControlID.CORRECT.value: 0,
            ControlID.RELEVANT_EDGE.value: 1,
            ControlID.IRRELEVANT_EDGE.value: 1,
            ControlID.GOLD_PATH.value: 0,
            ControlID.GOLD_RETURNS.value: 0,
            ControlID.NO_QUERY.value: 0,
            ControlID.EXPLICIT_MISS.value: 1,
            ControlID.HANDLE_SWAP.value: 0,
        }
        required_count = changed_count_rules.get(control_id)
        if required_count is not None and len(changed_addresses) != required_count:
            raise ValueError(
                f"{control_id} requires {required_count} changed addresses"
            )
        compact_controls = {
            ControlID.SHUFFLED_RETURNS.value,
            ControlID.ENTITY_RENAME.value,
            ControlID.GRAPH_ISOMORPHISM.value,
        }
        if control_id in compact_controls:
            if (
                transformation_id is None
                or changed_addresses
                or any(
                    commitment is None
                    for commitment in transformation_commitments.values()
                )
                or transformation_commitments["source_store_sha256"]
                == transformation_commitments["transformed_store_sha256"]
            ):
                raise ValueError(
                    f"{control_id} requires compact transformation metadata"
                )
        elif transformation_id is not None or any(
            commitment is not None
            for commitment in transformation_commitments.values()
        ):
            raise ValueError(
                f"{control_id} cannot reference a global transformation"
            )
        required_effects = {
            ControlID.CORRECT.value: "unchanged",
            ControlID.RELEVANT_EDGE.value: "changed",
            ControlID.IRRELEVANT_EDGE.value: "unchanged",
            ControlID.GOLD_PATH.value: "unchanged",
            ControlID.GOLD_RETURNS.value: "unchanged",
            ControlID.NO_QUERY.value: "unchanged",
            ControlID.EXPLICIT_MISS.value: "miss",
            ControlID.HANDLE_SWAP.value: "unchanged",
            ControlID.ENTITY_RENAME.value: "unchanged",
            ControlID.GRAPH_ISOMORPHISM.value: "unchanged",
        }
        if oracle_effect != required_effects.get(control_id, oracle_effect):
            raise ValueError(f"{control_id} has an invalid oracle effect")

        return _construct_validated(
            cls,
            record_type="eval_row",
            schema_version=version,
            qid=qid,
            pair_id=pair_id,
            variant=variant,
            task=task,
            world_id=world_id,
            provenance_id=provenance_id,
            relation_path_hash=relation_path_hash,
            template_id=template_id,
            composition_split=composition_split,
            hop=hop,
            seed=seed,
            model_id=model_id,
            arm=arm,
            checkpoint_sha256=checkpoint_sha256,
            raw_token_count=raw_token_count,
            memory_mode=memory_mode,
            control_id=control_id,
            cluster_id=cluster_id,
            prediction=prediction,
            answer=answer,
            correct=correct,
            prediction_source="model",
            all_actions=all_actions,
            gold_all_actions=gold_all_actions,
            returned_addresses=returned_addresses,
            gold_addresses=gold_addresses,
            correct_referents=correct_referents,
            misses=misses,
            malformed=malformed,
            abstained=abstained,
            excess_reads=excess_reads,
            halt_step=halt_step,
            answer_logits=tuple(answer_logits),
            lookup_latency_ns=lookup_latency_ns,
            lookup_count=lookup_count,
            store_rows=store_rows,
            store_bytes=store_bytes,
            control_seed=control_seed,
            transformation_id=transformation_id,
            **transformation_commitments,
            changed_addresses=changed_addresses,
            oracle_before=oracle_before,
            oracle_after=oracle_after,
            oracle_effect=oracle_effect,
            edit_locality_correct=edit_locality_correct,
            **hashes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "qid": self.qid,
            "pair_id": self.pair_id,
            "variant": self.variant,
            "task": self.task,
            "world_id": self.world_id,
            "provenance_id": self.provenance_id,
            "relation_path_hash": self.relation_path_hash,
            "template_id": self.template_id,
            "composition_split": self.composition_split,
            "hop": self.hop,
            "seed": self.seed,
            "model_id": self.model_id,
            "arm": self.arm,
            "checkpoint_sha256": self.checkpoint_sha256,
            "raw_token_count": self.raw_token_count,
            "memory_mode": self.memory_mode,
            "control_id": self.control_id,
            "evaluator_sha256": self.evaluator_sha256,
            "data_sha256": self.data_sha256,
            "relation_schema_sha256": self.relation_schema_sha256,
            "configuration_sha256": self.configuration_sha256,
            "result_schema_sha256": self.result_schema_sha256,
            "provenance_sha256": self.provenance_sha256,
            "cluster_id": self.cluster_id,
            "prediction": self.prediction,
            "answer": self.answer,
            "correct": self.correct,
            "prediction_source": self.prediction_source,
            "all_actions": [list(action) for action in self.all_actions],
            "gold_all_actions": [
                list(action) for action in self.gold_all_actions
            ],
            "returned_addresses": [
                None if address is None else list(address)
                for address in self.returned_addresses
            ],
            "gold_addresses": [
                list(address) for address in self.gold_addresses
            ],
            "correct_referents": list(self.correct_referents),
            "misses": self.misses,
            "malformed": self.malformed,
            "abstained": self.abstained,
            "excess_reads": self.excess_reads,
            "halt_step": self.halt_step,
            "answer_logits": [
                list(values) for values in self.answer_logits
            ],
            "lookup_latency_ns": self.lookup_latency_ns,
            "lookup_count": self.lookup_count,
            "store_rows": self.store_rows,
            "store_bytes": self.store_bytes,
            "control_seed": self.control_seed,
            "transformation_id": self.transformation_id,
            "source_store_sha256": self.source_store_sha256,
            "transformed_store_sha256": self.transformed_store_sha256,
            "transformation_metadata_sha256": (
                self.transformation_metadata_sha256
            ),
            "changed_addresses": [
                list(address) for address in self.changed_addresses
            ],
            "oracle_before": self.oracle_before,
            "oracle_after": self.oracle_after,
            "oracle_effect": self.oracle_effect,
            "edit_locality_correct": self.edit_locality_correct,
        }

    def paired_join_key(self) -> tuple[Any, ...]:
        return (
            self.seed,
            self.qid,
            self.pair_id,
            self.variant,
            self.task,
            self.world_id,
            self.relation_path_hash,
            self.template_id,
        )


_PAIR_METADATA_FIELDS = (
    "task",
    "world_id",
    "provenance_id",
    "relation_path_hash",
    "template_id",
    "composition_split",
    "hop",
    "seed",
    "model_id",
    "arm",
    "checkpoint_sha256",
    "raw_token_count",
    "memory_mode",
    "control_id",
    "evaluator_sha256",
    "data_sha256",
    "relation_schema_sha256",
    "provenance_sha256",
    "cluster_id",
    "control_seed",
    "store_rows",
    "store_bytes",
)


def validate_eval_rows(
    rows: Iterable[EvalRow | Mapping[str, Any]],
    *,
    require_complete_pairs: bool = True,
) -> tuple[EvalRow, ...]:
    materialized = tuple(
        EvalRow.from_dict(
            row.to_dict() if isinstance(row, EvalRow) else row
        )
        for row in rows
    )
    if not materialized:
        raise ValueError("evaluation rows must not be empty")
    seen: set[tuple[str, str, str, str]] = set()
    by_pair: dict[tuple[str, str, str, str], dict[str, EvalRow]] = defaultdict(dict)
    qid_metadata: dict[tuple[str, str], tuple[Any, ...]] = {}
    for row in materialized:
        identity = (
            row.checkpoint_sha256,
            row.memory_mode,
            row.control_id,
            row.qid,
        )
        if identity in seen:
            raise ValueError("duplicate evaluation row identity")
        seen.add(identity)
        qid_key = row.checkpoint_sha256, row.qid
        metadata = (
            row.pair_id,
            row.variant,
            row.task,
            row.world_id,
            row.provenance_id,
            row.relation_path_hash,
            row.template_id,
            row.composition_split,
            row.hop,
            row.seed,
            row.model_id,
            row.arm,
            row.evaluator_sha256,
            row.data_sha256,
            row.relation_schema_sha256,
            row.provenance_sha256,
            row.cluster_id,
        )
        if qid_key in qid_metadata and qid_metadata[qid_key] != metadata:
            raise ValueError("crossed provenance or cluster metadata for qid")
        qid_metadata[qid_key] = metadata
        pair_key = (
            row.checkpoint_sha256,
            row.memory_mode,
            row.control_id,
            row.pair_id,
        )
        if row.variant in by_pair[pair_key]:
            raise ValueError("duplicate variant in evaluation pair")
        by_pair[pair_key][row.variant] = row
    if require_complete_pairs:
        for pair in by_pair.values():
            if set(pair) != _VARIANTS:
                raise ValueError(
                    "every evaluation pair requires both variants"
                )
            original = pair["original"]
            counterfactual = pair["counterfactual"]
            if any(
                getattr(original, name) != getattr(counterfactual, name)
                for name in _PAIR_METADATA_FIELDS
            ):
                raise ValueError("crossed pair metadata or provenance")
            if original.qid == counterfactual.qid:
                raise ValueError("pair variants require distinct qids")
    return materialized


def _rate(value: object, name: str) -> dict[str, int | float | None]:
    measurement = _strict_fields(value, _RATE_FIELDS, name)
    numerator = _integer(
        measurement["numerator"],
        f"{name}.numerator",
    )
    denominator = _integer(
        measurement["denominator"],
        f"{name}.denominator",
    )
    if numerator > denominator:
        raise ValueError(f"{name} numerator cannot exceed denominator")
    raw_value = measurement["value"]
    if denominator == 0:
        if raw_value is not None:
            raise ValueError(f"{name} value must be null for an empty denominator")
        result = None
    else:
        result = _number(
            raw_value,
            f"{name}.value",
            minimum=0.0,
            maximum=1.0,
        )
        if result != numerator / denominator:
            raise ValueError(f"{name} value does not match its counts")
    return {
        "value": result,
        "numerator": numerator,
        "denominator": denominator,
    }


def _slice(value: object, name: str) -> dict[str, Any]:
    measurement = _strict_fields(value, _SLICE_FIELDS, name)
    return {
        field: _rate(measurement[field], f"{name}.{field}")
        for field in _SLICE_FIELDS
    }


def _metrics(value: object) -> dict[str, Any]:
    measurement = _strict_fields(value, _METRIC_FIELDS, "metrics")
    result = {
        field: _rate(measurement[field], f"metrics.{field}")
        for field in _METRIC_FIELDS[:10]
    }
    per_hop = _strict_fields(
        measurement["per_hop"],
        tuple(str(hop) for hop in range(1, 7)),
        "metrics.per_hop",
    )
    result["per_hop"] = {}
    for hop in range(1, 7):
        name = str(hop)
        fields = _strict_fields(
            per_hop[name],
            _PER_HOP_FIELDS,
            f"metrics.per_hop.{name}",
        )
        result["per_hop"][name] = {
            field: _rate(
                fields[field],
                f"metrics.per_hop.{name}.{field}",
            )
            for field in _PER_HOP_FIELDS
        }
    by_hop = _strict_fields(
        measurement["by_hop"],
        tuple(str(hop) for hop in range(1, 7)),
        "metrics.by_hop",
    )
    result["by_hop"] = {
        str(hop): _slice(
            by_hop[str(hop)],
            f"metrics.by_hop.{hop}",
        )
        for hop in range(1, 7)
    }
    by_composition = _strict_fields(
        measurement["by_composition"],
        ("seen", "heldout"),
        "metrics.by_composition",
    )
    result["by_composition"] = {
        name: _slice(
            by_composition[name],
            f"metrics.by_composition.{name}",
        )
        for name in ("seen", "heldout")
    }
    by_task = _strict_fields(
        measurement["by_task"],
        ("path_composition", "date_ordering", "balanced_equality"),
        "metrics.by_task",
    )
    result["by_task"] = {
        name: _slice(
            by_task[name],
            f"metrics.by_task.{name}",
        )
        for name in (
            "path_composition",
            "date_ordering",
            "balanced_equality",
        )
    }
    store = _strict_fields(
        measurement["store"],
        ("rows", "bytes", "lookup_latency_ns", "lookup_count"),
        "metrics.store",
    )
    lookup_count = _integer(store["lookup_count"], "metrics.store.lookup_count")
    latency_raw = store["lookup_latency_ns"]
    if lookup_count == 0:
        if latency_raw is not None:
            raise ValueError(
                "store lookup latency must be null with no lookups"
            )
        latency = None
    else:
        latency = _number(
            latency_raw,
            "metrics.store.lookup_latency_ns",
            minimum=0.0,
        )
    result["store"] = {
        "rows": _integer(store["rows"], "metrics.store.rows"),
        "bytes": _integer(store["bytes"], "metrics.store.bytes"),
        "lookup_latency_ns": latency,
        "lookup_count": lookup_count,
    }
    result["edit_locality"] = _rate(
        measurement["edit_locality"],
        "metrics.edit_locality",
    )
    crossings = measurement["milestone_crossings"]
    if not isinstance(crossings, Mapping) or set(crossings) != _MILESTONE_NAMES:
        raise ValueError(
            "milestone crossing set must match preregistered milestones"
        )
    result["milestone_crossings"] = {
        name: (
            None
            if crossings[name] is None
            else _integer(
                crossings[name],
                f"milestone_crossings.{name}",
            )
        )
        for name in sorted(crossings)
    }
    return result


@dataclass(frozen=True)
class CheckpointSummary:
    record_type: str
    schema_version: int
    checkpoint_sha256: str
    model_id: str
    arm: str
    seed: int
    raw_token_count: int
    memory_mode: str
    control_id: str
    evaluator_sha256: str
    data_sha256: str
    relation_schema_sha256: str
    configuration_sha256: str
    result_schema_sha256: str
    provenance_sha256: str
    rows_sha256: str
    n_rows: int
    n_pairs: int
    metrics: Mapping[str, Any]

    FIELDS: ClassVar[tuple[str, ...]] = CHECKPOINT_SUMMARY_FIELDS

    def __post_init__(self) -> None:
        _validate_typed_instance(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CheckpointSummary":
        value = _strict_fields(raw, cls.FIELDS, "CheckpointSummary")
        if value["record_type"] != "checkpoint_summary":
            raise ValueError(
                "CheckpointSummary record_type must be checkpoint_summary"
            )
        version = _integer(value["schema_version"], "schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError("unsupported CheckpointSummary schema_version")
        metrics = _metrics(value["metrics"])
        n_rows = _integer(value["n_rows"], "n_rows", minimum=2)
        n_pairs = _integer(value["n_pairs"], "n_pairs", minimum=1)
        if n_rows != 2 * n_pairs:
            raise ValueError(
                "checkpoint summaries require exactly two rows per pair"
            )
        memory_mode = _enum(
            value["memory_mode"],
            {mode.value for mode in EvalMode},
            "memory_mode",
        )
        control_id = _enum(
            value["control_id"],
            {control.value for control in ControlID},
            "control_id",
        )
        if metrics["item_accuracy"]["denominator"] != n_rows:
            raise ValueError("item accuracy denominator must equal n_rows")
        if metrics["pair_accuracy"]["denominator"] != n_pairs:
            raise ValueError("pair accuracy denominator must equal n_pairs")
        for name, expected in {
            "all_six_action_exact": n_rows,
            "exact_action_path": n_rows,
            "malformed_rate": 6 * n_rows,
            "abstention_rate": n_rows,
            "excess_read_rate": 6 * n_rows,
        }.items():
            if metrics[name]["denominator"] != expected:
                raise ValueError(
                    f"{name} denominator does not match checkpoint rows"
                )
        expected_gold_path = (
            n_rows if control_id == ControlID.GOLD_PATH.value else 0
        )
        if (
            metrics["gold_path_answer_accuracy"]["denominator"]
            != expected_gold_path
        ):
            raise ValueError(
                "gold_path_answer_accuracy denominator disagrees "
                "with control_id"
            )
        expected_locality = (
            n_rows
            if control_id
            in {
                ControlID.RELEVANT_EDGE.value,
                ControlID.IRRELEVANT_EDGE.value,
            }
            else 0
        )
        if metrics["edit_locality"]["denominator"] != expected_locality:
            raise ValueError(
                "edit_locality denominator disagrees with control_id"
            )
        if (
            metrics["answer_given_correct_retrieval"]["denominator"]
            > n_rows
        ):
            raise ValueError(
                "answer_given_correct_retrieval denominator exceeds n_rows"
            )
        per_hop_denominators: list[int] = []
        for hop in range(1, 7):
            hop_metrics = metrics["per_hop"][str(hop)]
            denominators = {
                hop_metrics[name]["denominator"]
                for name in _PER_HOP_FIELDS
            }
            if len(denominators) != 1:
                raise ValueError(
                    f"per_hop.{hop} metric denominators must match"
                )
            denominator = next(iter(denominators))
            if denominator > n_rows:
                raise ValueError(
                    f"per_hop.{hop} denominator exceeds n_rows"
                )
            per_hop_denominators.append(denominator)
        if per_hop_denominators[0] != n_rows or any(
            current > previous
            for previous, current in zip(
                per_hop_denominators,
                per_hop_denominators[1:],
            )
        ):
            raise ValueError(
                "per_hop denominators must begin at n_rows "
                "and be non-increasing"
            )
        for family in ("by_hop", "by_composition", "by_task"):
            slices = metrics[family].values()
            item_total = 0
            pair_total = 0
            for slice_metrics in slices:
                item_denominator = slice_metrics[
                    "item_accuracy"
                ]["denominator"]
                pair_denominator = slice_metrics[
                    "pair_accuracy"
                ]["denominator"]
                if item_denominator != 2 * pair_denominator:
                    raise ValueError(
                        f"{family} slices require two rows per pair"
                    )
                if (
                    slice_metrics["exact_action_path"]["denominator"]
                    != item_denominator
                ):
                    raise ValueError(
                        f"{family} exact-action denominator mismatch"
                    )
                item_total += item_denominator
                pair_total += pair_denominator
            if item_total != n_rows or pair_total != n_pairs:
                raise ValueError(
                    f"{family} slice denominators must partition the checkpoint"
                )
        lookup_count = metrics["store"]["lookup_count"]
        if memory_mode == EvalMode.MEMORY_ON.value:
            if lookup_count > metrics["miss_rate"]["denominator"]:
                raise ValueError(
                    "memory-on lookup_count cannot exceed read attempts"
                )
        elif lookup_count != 0:
            raise ValueError("memory-off lookup_count must be zero")
        raw_token_count = _integer(
            value["raw_token_count"],
            "raw_token_count",
        )
        if any(
            crossing is not None and crossing > raw_token_count
            for crossing in metrics["milestone_crossings"].values()
        ):
            raise ValueError(
                "milestone crossing cannot name a future raw-token count"
            )
        return _construct_validated(
            cls,
            record_type="checkpoint_summary",
            schema_version=version,
            checkpoint_sha256=_sha256(
                value["checkpoint_sha256"],
                "checkpoint_sha256",
            ),
            model_id=_string(value["model_id"], "model_id"),
            arm=_enum(value["arm"], _ARMS, "arm"),
            seed=_integer(value["seed"], "seed"),
            raw_token_count=raw_token_count,
            memory_mode=memory_mode,
            control_id=control_id,
            evaluator_sha256=_sha256(
                value["evaluator_sha256"],
                "evaluator_sha256",
            ),
            data_sha256=_sha256(value["data_sha256"], "data_sha256"),
            relation_schema_sha256=_sha256(
                value["relation_schema_sha256"],
                "relation_schema_sha256",
            ),
            configuration_sha256=_sha256(
                value["configuration_sha256"],
                "configuration_sha256",
            ),
            result_schema_sha256=_sha256(
                value["result_schema_sha256"],
                "result_schema_sha256",
            ),
            provenance_sha256=_sha256(
                value["provenance_sha256"],
                "provenance_sha256",
            ),
            rows_sha256=_sha256(value["rows_sha256"], "rows_sha256"),
            n_rows=n_rows,
            n_pairs=n_pairs,
            metrics=_deep_freeze(metrics),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_id": self.model_id,
            "arm": self.arm,
            "seed": self.seed,
            "raw_token_count": self.raw_token_count,
            "memory_mode": self.memory_mode,
            "control_id": self.control_id,
            "evaluator_sha256": self.evaluator_sha256,
            "data_sha256": self.data_sha256,
            "relation_schema_sha256": self.relation_schema_sha256,
            "configuration_sha256": self.configuration_sha256,
            "result_schema_sha256": self.result_schema_sha256,
            "provenance_sha256": self.provenance_sha256,
            "rows_sha256": self.rows_sha256,
            "n_rows": self.n_rows,
            "n_pairs": self.n_pairs,
            "metrics": _deep_thaw(self.metrics),
        }

    def validate_rows(
        self,
        rows: Iterable[EvalRow | Mapping[str, Any]],
    ) -> tuple[EvalRow, ...]:
        CheckpointSummary.from_dict(self.to_dict())
        materialized = validate_eval_rows(rows)
        if len(materialized) != self.n_rows:
            raise ValueError("summary n_rows does not match evaluation rows")
        pair_count = len({row.pair_id for row in materialized})
        if pair_count != self.n_pairs:
            raise ValueError("summary n_pairs does not match evaluation rows")
        identities = (
            "checkpoint_sha256",
            "model_id",
            "arm",
            "seed",
            "raw_token_count",
            "memory_mode",
            "control_id",
            "evaluator_sha256",
            "data_sha256",
            "relation_schema_sha256",
            "configuration_sha256",
            "result_schema_sha256",
            "provenance_sha256",
        )
        for row in materialized:
            for name in identities:
                if getattr(row, name) != getattr(self, name):
                    raise ValueError(
                        f"row {name} does not match checkpoint summary"
                    )
        from evals.relational_metrics import compute_checkpoint_metrics

        recomputed = compute_checkpoint_metrics(
            materialized,
            milestone_crossings=self.metrics["milestone_crossings"],
        )
        if recomputed != dict(self.metrics):
            raise ValueError(
                "checkpoint metrics do not match recomputed row metrics"
            )
        if rows_sha256(materialized) != self.rows_sha256:
            raise ValueError("rows_sha256 does not match checkpoint summary")
        return materialized


def _wilson_upper(successes: int, total: int) -> float:
    proportion = successes / total
    denominator = 1 + _Z_975 * _Z_975 / total
    center = (
        proportion + _Z_975 * _Z_975 / (2 * total)
    ) / denominator
    radius = (
        _Z_975
        * (
            proportion * (1 - proportion) / total
            + _Z_975 * _Z_975 / (4 * total * total)
        )
        ** 0.5
        / denominator
    )
    return center + radius


def _guard_check(value: object, expected_id: str, name: str) -> dict[str, Any]:
    check = _strict_fields(value, GUARD_CHECK_FIELDS, name)
    if check["check_id"] != expected_id:
        raise ValueError(f"{name} check_id mismatch")
    expected_comparison, fixed_threshold, formula = _GUARD_RULES[
        expected_id
    ]
    raw_value = check["value"]
    reference_raw = check["reference_value"]
    if formula is None:
        if reference_raw is not None:
            raise ValueError(f"{name} reference_value must be null")
        reference_value = None
    else:
        reference_value = _number(
            reference_raw,
            f"{name}.reference_value",
            minimum=0.0,
            maximum=(
                None
                if expected_id == "bpb_noninferiority"
                else 1.0
            ),
        )
    raw_threshold = check["threshold"]
    if isinstance(raw_value, bool):
        measured: float | bool = raw_value
    else:
        measured = _number(raw_value, f"{name}.value")
    if isinstance(raw_threshold, bool):
        threshold: float | bool = raw_threshold
    else:
        threshold = _number(raw_threshold, f"{name}.threshold")
    comparison = _enum(
        check["comparison"],
        {"<", "<=", ">=", "=="},
        f"{name}.comparison",
    )
    if comparison != expected_comparison:
        raise ValueError(f"{name} comparison is not preregistered")
    if formula is None:
        expected_threshold = fixed_threshold
    elif formula[0] == "offset":
        expected_threshold = reference_value + formula[1]
    else:
        expected_threshold = reference_value * formula[1]
    if threshold != expected_threshold:
        raise ValueError(f"{name} threshold is not preregistered")
    if expected_id == "integrity_conjunction":
        if not isinstance(measured, bool):
            raise ValueError(f"{name}.value must be Boolean")
    elif isinstance(measured, bool):
        raise ValueError(f"{name}.value must be numeric")
    elif expected_id == "bpb_noninferiority" and (
        measured <= 0.0 or reference_value <= 0.0
    ):
        raise ValueError(f"{name} BPB values must be positive")
    elif expected_id != "bpb_noninferiority" and not 0.0 <= measured <= 1.0:
        raise ValueError(f"{name}.value must be a rate")
    passed = _boolean(check["passed"], f"{name}.passed")
    if comparison == "<":
        computed = measured < threshold
    elif comparison == "<=":
        computed = measured <= threshold
    elif comparison == ">=":
        computed = measured >= threshold
    else:
        computed = measured == threshold
    if passed != computed:
        raise ValueError(f"{name} passed flag disagrees with comparison")
    numerator_raw = check["numerator"]
    denominator_raw = check["denominator"]
    if (numerator_raw is None) != (denominator_raw is None):
        raise ValueError(f"{name} numerator/denominator must both be null")
    if numerator_raw is None:
        numerator = denominator = None
    else:
        numerator = _integer(numerator_raw, f"{name}.numerator")
        denominator = _integer(
            denominator_raw,
            f"{name}.denominator",
            minimum=1,
        )
        if numerator > denominator:
            raise ValueError(f"{name} numerator cannot exceed denominator")
    expects_counts = expected_id not in {
        "bpb_noninferiority",
        "integrity_conjunction",
    }
    if expects_counts != (numerator is not None):
        raise ValueError(f"{name} count fields do not match its metric type")
    if expects_counts:
        expected_value = (
            _wilson_upper(numerator, denominator)
            if expected_id == "split_off_recognition_wilson_hi"
            else numerator / denominator
        )
        if measured != expected_value:
            suffix = (
                "Wilson value"
                if expected_id == "split_off_recognition_wilson_hi"
                else "value/counts"
            )
            raise ValueError(f"{name} {suffix} disagree")
    return {
        "check_id": expected_id,
        "value": measured,
        "reference_value": reference_value,
        "threshold": threshold,
        "comparison": comparison,
        "passed": passed,
        "numerator": numerator,
        "denominator": denominator,
    }


@dataclass(frozen=True)
class GuardrailReport:
    record_type: str
    schema_version: int
    split_checkpoint_sha256: str
    dense_checkpoint_sha256: str
    model_id: str
    seed: int
    raw_token_count: int
    evaluator_sha256: str
    data_sha256: str
    relation_schema_sha256: str
    split_configuration_sha256: str
    dense_configuration_sha256: str
    result_schema_sha256: str
    split_result_provenance_sha256: str
    dense_result_provenance_sha256: str
    study_provenance_sha256: str
    pairing_receipt_sha256: str
    split_guardrail_source_sha256: str
    dense_guardrail_source_sha256: str
    split_matrix_manifest_sha256: str
    dense_matrix_manifest_sha256: str
    guards: Mapping[str, Any]
    confirmatory_passed: bool

    FIELDS: ClassVar[tuple[str, ...]] = GUARDRAIL_REPORT_FIELDS

    def __post_init__(self) -> None:
        _validate_typed_instance(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GuardrailReport":
        value = _strict_fields(raw, cls.FIELDS, "GuardrailReport")
        if value["record_type"] != "guardrail_report":
            raise ValueError(
                "GuardrailReport record_type must be guardrail_report"
            )
        version = _integer(value["schema_version"], "schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError("unsupported GuardrailReport schema_version")
        split_checkpoint = _sha256(
            value["split_checkpoint_sha256"],
            "split_checkpoint_sha256",
        )
        dense_checkpoint = _sha256(
            value["dense_checkpoint_sha256"],
            "dense_checkpoint_sha256",
        )
        if split_checkpoint == dense_checkpoint:
            raise ValueError(
                "guardrails require distinct checkpoints for split and dense"
            )
        guard_values = _strict_fields(
            value["guards"],
            tuple(_GUARD_CHECKS),
            "guards",
        )
        guards: dict[str, Any] = {}
        for guard_name, expected_checks in _GUARD_CHECKS.items():
            guard = _strict_fields(
                guard_values[guard_name],
                ("passed", "checks"),
                f"guards.{guard_name}",
            )
            checks_raw = guard["checks"]
            if not isinstance(checks_raw, (list, tuple)) or len(
                checks_raw
            ) != len(expected_checks):
                raise ValueError(
                    f"guards.{guard_name} has the wrong check count"
                )
            checks = [
                _guard_check(
                    check,
                    check_id,
                    f"guards.{guard_name}.checks[{index}]",
                )
                for index, (check, check_id) in enumerate(
                    zip(checks_raw, expected_checks)
                )
            ]
            passed = _boolean(
                guard["passed"],
                f"guards.{guard_name}.passed",
            )
            if passed != all(check["passed"] for check in checks):
                raise ValueError(
                    f"guards.{guard_name} passed must be a conjunction"
                )
            guards[guard_name] = {"passed": passed, "checks": checks}
        confirmatory = _boolean(
            value["confirmatory_passed"],
            "confirmatory_passed",
        )
        if confirmatory != all(
            guard["passed"] for guard in guards.values()
        ):
            raise ValueError(
                "confirmatory verdict must be the six-guard conjunction"
            )
        split_configuration = _sha256(
            value["split_configuration_sha256"],
            "split_configuration_sha256",
        )
        dense_configuration = _sha256(
            value["dense_configuration_sha256"],
            "dense_configuration_sha256",
        )
        if split_configuration == dense_configuration:
            raise ValueError(
                "Split and Dense configuration hashes must be distinct"
            )
        split_result_provenance = _sha256(
            value["split_result_provenance_sha256"],
            "split_result_provenance_sha256",
        )
        dense_result_provenance = _sha256(
            value["dense_result_provenance_sha256"],
            "dense_result_provenance_sha256",
        )
        if split_result_provenance == dense_result_provenance:
            raise ValueError(
                "Split and Dense result provenance hashes must be distinct"
            )
        return _construct_validated(
            cls,
            record_type="guardrail_report",
            schema_version=version,
            split_checkpoint_sha256=split_checkpoint,
            dense_checkpoint_sha256=dense_checkpoint,
            model_id=_string(value["model_id"], "model_id"),
            seed=_integer(value["seed"], "seed"),
            raw_token_count=_integer(
                value["raw_token_count"],
                "raw_token_count",
            ),
            evaluator_sha256=_sha256(
                value["evaluator_sha256"],
                "evaluator_sha256",
            ),
            data_sha256=_sha256(value["data_sha256"], "data_sha256"),
            relation_schema_sha256=_sha256(
                value["relation_schema_sha256"],
                "relation_schema_sha256",
            ),
            split_configuration_sha256=split_configuration,
            dense_configuration_sha256=dense_configuration,
            result_schema_sha256=_sha256(
                value["result_schema_sha256"],
                "result_schema_sha256",
            ),
            split_result_provenance_sha256=split_result_provenance,
            dense_result_provenance_sha256=dense_result_provenance,
            study_provenance_sha256=_sha256(
                value["study_provenance_sha256"],
                "study_provenance_sha256",
            ),
            pairing_receipt_sha256=_sha256(
                value["pairing_receipt_sha256"],
                "pairing_receipt_sha256",
            ),
            split_guardrail_source_sha256=_sha256(
                value["split_guardrail_source_sha256"],
                "split_guardrail_source_sha256",
            ),
            dense_guardrail_source_sha256=_sha256(
                value["dense_guardrail_source_sha256"],
                "dense_guardrail_source_sha256",
            ),
            split_matrix_manifest_sha256=_sha256(
                value["split_matrix_manifest_sha256"],
                "split_matrix_manifest_sha256",
            ),
            dense_matrix_manifest_sha256=_sha256(
                value["dense_matrix_manifest_sha256"],
                "dense_matrix_manifest_sha256",
            ),
            guards=_deep_freeze(guards),
            confirmatory_passed=confirmatory,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "split_checkpoint_sha256": self.split_checkpoint_sha256,
            "dense_checkpoint_sha256": self.dense_checkpoint_sha256,
            "model_id": self.model_id,
            "seed": self.seed,
            "raw_token_count": self.raw_token_count,
            "evaluator_sha256": self.evaluator_sha256,
            "data_sha256": self.data_sha256,
            "relation_schema_sha256": self.relation_schema_sha256,
            "split_configuration_sha256": (
                self.split_configuration_sha256
            ),
            "dense_configuration_sha256": (
                self.dense_configuration_sha256
            ),
            "result_schema_sha256": self.result_schema_sha256,
            "split_result_provenance_sha256": (
                self.split_result_provenance_sha256
            ),
            "dense_result_provenance_sha256": (
                self.dense_result_provenance_sha256
            ),
            "study_provenance_sha256": self.study_provenance_sha256,
            "pairing_receipt_sha256": self.pairing_receipt_sha256,
            "split_guardrail_source_sha256": (
                self.split_guardrail_source_sha256
            ),
            "dense_guardrail_source_sha256": (
                self.dense_guardrail_source_sha256
            ),
            "split_matrix_manifest_sha256": (
                self.split_matrix_manifest_sha256
            ),
            "dense_matrix_manifest_sha256": (
                self.dense_matrix_manifest_sha256
            ),
            "guards": _deep_thaw(self.guards),
            "confirmatory_passed": self.confirmatory_passed,
        }


def validate_result_payload(
    raw: EvalRow
    | CheckpointSummary
    | GuardrailReport
    | Mapping[str, Any],
) -> EvalRow | CheckpointSummary | GuardrailReport:
    """Apply the mandatory semantic validator after structural validation."""

    if isinstance(raw, (EvalRow, CheckpointSummary, GuardrailReport)):
        value = raw.to_dict()
    elif isinstance(raw, Mapping):
        value = raw
    else:
        raise TypeError("relational result payload must be an object")
    record_type = value.get("record_type")
    contract = {
        "eval_row": EvalRow,
        "checkpoint_summary": CheckpointSummary,
        "guardrail_report": GuardrailReport,
    }.get(record_type)
    if contract is None:
        raise ValueError("unknown relational result record_type")
    return contract.from_dict(value)


def _canonicalize(value: Any, path: str = "$") -> Any:
    if isinstance(value, (EvalRow, CheckpointSummary, GuardrailReport)):
        return _canonicalize(
            validate_result_payload(value).to_dict(),
            path,
        )
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _canonicalize(value.to_dict(), path)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path} contains a non-string object key")
        return {
            key: _canonicalize(value[key], f"{path}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [
            _canonicalize(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{path} contains a non-canonical or nonfinite value")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _canonicalize(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )


def rows_sha256(rows: Iterable[EvalRow | Mapping[str, Any]]) -> str:
    materialized = validate_eval_rows(rows)
    digest = hashlib.sha256()
    for row in sorted(
        materialized,
        key=lambda item: (
            item.checkpoint_sha256,
            item.memory_mode,
            item.control_id,
            item.paired_join_key(),
        ),
    ):
        digest.update(canonical_json_bytes(row))
    return digest.hexdigest()


def load_result_schema(path: str | Path | None = None) -> dict[str, Any]:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "relational-result-v1.schema.json"
        if path is None
        else Path(path)
    )
    if schema_path.is_symlink() or not schema_path.is_file():
        raise FileNotFoundError(schema_path)
    try:
        value = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid relational result JSON schema") from exc
    if not isinstance(value, dict):
        raise ValueError("relational result schema must be an object")
    return value


def _ensure_regular_directory(path: Path, name: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{name} cannot be a symlink")
    if not path.is_dir():
        raise ValueError(f"{name} must be a regular directory")


def _mkdir_component(parent: Path, component: str) -> Path:
    if not component or component in {".", ".."} or "/" in component:
        raise ValueError("invalid evaluation path component")
    path = parent / component
    if os.path.lexists(path):
        _ensure_regular_directory(path, "evaluation output parent")
    else:
        path.mkdir()
    return path


def _write_synced(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directory_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError(
            "no-follow directory publication is unsupported on this platform"
        )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _open_directory_at(parent_fd: int, name: str) -> int:
    try:
        return os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(
                "evaluation output parent cannot be a symlink"
            ) from exc
        raise


def _open_or_create_directory_at(parent_fd: int, name: str) -> int:
    if not name or name in {".", ".."} or "/" in name:
        raise ValueError("invalid evaluation path component")
    try:
        os.mkdir(name, mode=0o755, dir_fd=parent_fd)
    except FileExistsError:
        pass
    return _open_directory_at(parent_fd, name)


def _directory_entry_matches(
    parent_fd: int,
    name: str,
    child_fd: int,
) -> bool:
    try:
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    opened = os.fstat(child_fd)
    return (
        stat.S_ISDIR(entry.st_mode)
        and entry.st_dev == opened.st_dev
        and entry.st_ino == opened.st_ino
    )


def _path_matches_directory(path: Path, descriptor: int) -> bool:
    try:
        entry = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return False
    opened = os.fstat(descriptor)
    return (
        stat.S_ISDIR(entry.st_mode)
        and entry.st_dev == opened.st_dev
        and entry.st_ino == opened.st_ino
    )


def _rename_directory_noreplace_between(
    source_parent_fd: int,
    source: str,
    destination_parent_fd: int,
    destination: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        rename = libc.renameatx_np
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_parent_fd,
            source_bytes,
            destination_parent_fd,
            destination_bytes,
            0x00000004,
        )
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_parent_fd,
            source_bytes,
            destination_parent_fd,
            destination_bytes,
            0x1,
        )
    else:
        raise RuntimeError(
            "atomic no-replace directory publication is unsupported"
        )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            f"evaluation output already exists: {destination}"
        )
    raise OSError(error, os.strerror(error), destination)


def _rename_directory_noreplace_at(
    parent_fd: int,
    source: str,
    destination: str,
) -> None:
    _rename_directory_noreplace_between(
        parent_fd,
        source,
        parent_fd,
        destination,
    )


def _write_synced_at(directory_fd: int, name: str, content: bytes) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _remove_staged_directory_at(parent_fd: int, name: str) -> None:
    try:
        descriptor = _open_directory_at(parent_fd, name)
    except FileNotFoundError:
        return
    try:
        for entry in os.listdir(descriptor):
            os.unlink(entry, dir_fd=descriptor)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.rmdir(name, dir_fd=parent_fd)


class StreamingEvaluationPublisher:
    """Disk-spool strict rows and publish without retaining the corpus."""

    def __init__(self, run: str | Path) -> None:
        self.run_path = Path(run)
        _ensure_regular_directory(self.run_path, "run")
        if self.run_path.resolve(strict=True) != self.run_path.absolute():
            raise ValueError("run path cannot traverse symlink components")
        descriptor, name = tempfile.mkstemp(
            dir=self.run_path,
            prefix=".relational-rows-",
            suffix=".sqlite3",
        )
        os.close(descriptor)
        self._database_path = Path(name)
        self._connection = sqlite3.connect(self._database_path)
        self._connection.execute("PRAGMA journal_mode=DELETE")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            """
            CREATE TABLE rows (
                qid TEXT PRIMARY KEY,
                pair_id TEXT NOT NULL,
                variant TEXT NOT NULL,
                sort_key TEXT NOT NULL UNIQUE,
                payload BLOB NOT NULL,
                UNIQUE(pair_id, variant)
            )
            """
        )
        self._closed = False

    @property
    def buffered_rows(self) -> int:
        return 0

    @property
    def closed(self) -> bool:
        return self._closed

    @staticmethod
    def _validated_row(
        row: EvalRow | Mapping[str, Any],
    ) -> EvalRow:
        return EvalRow.from_dict(
            row.to_dict() if isinstance(row, EvalRow) else row
        )

    def add(self, row: EvalRow | Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("streaming publisher is closed")
        value = self._validated_row(row)
        sort_key = json.dumps(
            value.paired_join_key(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            self._connection.execute(
                """
                INSERT INTO rows(qid, pair_id, variant, sort_key, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    value.qid,
                    value.pair_id,
                    value.variant,
                    sort_key,
                    canonical_json_bytes(value),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "duplicate evaluation row identity or pair variant"
            ) from exc

    def add_pair(
        self,
        rows: Iterable[EvalRow | Mapping[str, Any]],
    ) -> None:
        pair = validate_eval_rows(rows)
        if len(pair) != 2 or len({row.pair_id for row in pair}) != 1:
            raise ValueError("streaming publication requires one complete pair")
        with self._connection:
            for row in pair:
                self.add(row)

    def _iter_pair_rows(self):
        cursor = self._connection.execute(
            """
            SELECT pair_id, payload
            FROM rows
            ORDER BY pair_id, variant
            """
        )
        current_id: str | None = None
        current: list[EvalRow] = []
        for pair_id, payload in cursor:
            if current_id is not None and pair_id != current_id:
                yield current
                current = []
            current_id = pair_id
            current.append(EvalRow.from_dict(json.loads(payload)))
        if current:
            yield current

    def _ordered_payloads(self):
        cursor = self._connection.execute(
            "SELECT payload FROM rows ORDER BY sort_key"
        )
        for (payload,) in cursor:
            yield bytes(payload)

    def _computed_summary(
        self,
        milestone_crossings: Mapping[str, int | None] | None,
    ) -> tuple[CheckpointSummary, str]:
        from evals.relational_metrics import CheckpointMetricAccumulator

        accumulator = CheckpointMetricAccumulator(
            milestone_crossings=milestone_crossings
        )
        first: EvalRow | None = None
        n_rows = 0
        n_pairs = 0
        for pair in self._iter_pair_rows():
            accumulator.add_pair(pair)
            first = pair[0] if first is None else first
            n_rows += len(pair)
            n_pairs += 1
        if first is None:
            raise ValueError("evaluation rows must not be empty")
        digest = hashlib.sha256()
        for payload in self._ordered_payloads():
            digest.update(payload)
        summary = CheckpointSummary.from_dict(
            {
                "record_type": "checkpoint_summary",
                "schema_version": SCHEMA_VERSION,
                "checkpoint_sha256": first.checkpoint_sha256,
                "model_id": first.model_id,
                "arm": first.arm,
                "seed": first.seed,
                "raw_token_count": first.raw_token_count,
                "memory_mode": first.memory_mode,
                "control_id": first.control_id,
                "evaluator_sha256": first.evaluator_sha256,
                "data_sha256": first.data_sha256,
                "relation_schema_sha256": first.relation_schema_sha256,
                "configuration_sha256": first.configuration_sha256,
                "result_schema_sha256": first.result_schema_sha256,
                "provenance_sha256": first.provenance_sha256,
                "rows_sha256": digest.hexdigest(),
                "n_rows": n_rows,
                "n_pairs": n_pairs,
                "metrics": accumulator.finalize(),
            }
        )
        return summary, digest.hexdigest()

    def _write_rows(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("xb") as stream:
            for payload in self._ordered_payloads():
                stream.write(payload)
                digest.update(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return digest.hexdigest()

    def _write_rows_at(self, directory_fd: int, name: str) -> str:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(
            name,
            flags,
            0o600,
            dir_fd=directory_fd,
        )
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as stream:
            for payload in self._ordered_payloads():
                stream.write(payload)
                digest.update(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return digest.hexdigest()

    def preview_summary(
        self,
        *,
        milestone_crossings: Mapping[str, int | None] | None = None,
    ) -> CheckpointSummary:
        if self._closed:
            raise RuntimeError("streaming publisher is closed")
        self._connection.commit()
        summary, _ = self._computed_summary(milestone_crossings)
        return summary

    def finish(
        self,
        *,
        milestone_crossings: Mapping[str, int | None] | None = None,
        expected_summary: CheckpointSummary | Mapping[str, Any] | None = None,
    ) -> tuple[Path, CheckpointSummary]:
        if self._closed:
            raise RuntimeError("streaming publisher is closed")
        self._connection.commit()
        summary, expected_rows_hash = self._computed_summary(
            milestone_crossings
        )
        if expected_summary is not None:
            expected = CheckpointSummary.from_dict(
                expected_summary.to_dict()
                if isinstance(expected_summary, CheckpointSummary)
                else expected_summary
            )
            if expected.to_dict() != summary.to_dict():
                raise ValueError(
                    "checkpoint summary does not match streamed row metrics"
                )
            summary = expected

        destination = (
            self.run_path
            / "evals"
            / summary.checkpoint_sha256
            / summary.memory_mode
            / summary.control_id
        )
        run_fd: int | None = None
        evals_fd: int | None = None
        checkpoint_fd: int | None = None
        mode_fd: int | None = None
        lock_owned = False
        temporary_name: str | None = None
        lock_name = f".{summary.control_id}.publish.lock"
        try:
            run_fd = os.open(self.run_path, _directory_flags())
            if not _path_matches_directory(self.run_path, run_fd):
                raise ValueError("run directory changed during publication")
            evals_fd = _open_or_create_directory_at(run_fd, "evals")
            checkpoint_fd = _open_or_create_directory_at(
                evals_fd,
                summary.checkpoint_sha256,
            )
            mode_fd = _open_or_create_directory_at(
                checkpoint_fd,
                summary.memory_mode,
            )
            try:
                os.stat(
                    summary.control_id,
                    dir_fd=mode_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(
                    f"evaluation output already exists: {destination}"
                )
            descriptor = os.open(
                lock_name,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
                dir_fd=mode_fd,
            )
            lock_owned = True
            os.close(descriptor)
            for _ in range(100):
                candidate = (
                    f".{summary.control_id}.tmp-{secrets.token_hex(12)}"
                )
                try:
                    os.mkdir(candidate, mode=0o700, dir_fd=mode_fd)
                except FileExistsError:
                    continue
                temporary_name = candidate
                break
            if temporary_name is None:
                raise FileExistsError(
                    "could not reserve a publication staging directory"
                )
            temporary_fd = _open_directory_at(mode_fd, temporary_name)
            try:
                rows_file_hash = self._write_rows_at(
                    temporary_fd,
                    "rows.jsonl",
                )
                if rows_file_hash != expected_rows_hash:
                    raise AssertionError(
                        "streamed row hash changed during publication"
                    )
                summary_content = canonical_json_bytes(summary)
                _write_synced_at(
                    temporary_fd,
                    "summary.json",
                    summary_content,
                )
                manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "checkpoint_sha256": summary.checkpoint_sha256,
                    "memory_mode": summary.memory_mode,
                    "control_id": summary.control_id,
                    "n_rows": summary.n_rows,
                    "file_sha256": {
                        "rows.jsonl": rows_file_hash,
                        "summary.json": hashlib.sha256(
                            summary_content
                        ).hexdigest(),
                    },
                }
                _write_synced_at(
                    temporary_fd,
                    "manifest.json",
                    canonical_json_bytes(manifest),
                )
                os.fsync(temporary_fd)
            finally:
                os.close(temporary_fd)

            parents_unchanged = (
                _path_matches_directory(self.run_path, run_fd)
                and _directory_entry_matches(run_fd, "evals", evals_fd)
                and _directory_entry_matches(
                    evals_fd,
                    summary.checkpoint_sha256,
                    checkpoint_fd,
                )
                and _directory_entry_matches(
                    checkpoint_fd,
                    summary.memory_mode,
                    mode_fd,
                )
            )
            if not parents_unchanged:
                raise ValueError(
                    "evaluation parent changed during publication"
                )
            _rename_directory_noreplace_at(
                mode_fd,
                temporary_name,
                summary.control_id,
            )
            if not (
                _path_matches_directory(self.run_path, run_fd)
                and _directory_entry_matches(run_fd, "evals", evals_fd)
                and _directory_entry_matches(
                    evals_fd,
                    summary.checkpoint_sha256,
                    checkpoint_fd,
                )
                and _directory_entry_matches(
                    checkpoint_fd,
                    summary.memory_mode,
                    mode_fd,
                )
            ):
                os.rename(
                    summary.control_id,
                    temporary_name,
                    src_dir_fd=mode_fd,
                    dst_dir_fd=mode_fd,
                )
                raise ValueError(
                    "evaluation parent changed during publication"
                )
            temporary_name = None
            os.fsync(mode_fd)
            return destination, summary
        finally:
            if mode_fd is not None and temporary_name is not None:
                _remove_staged_directory_at(mode_fd, temporary_name)
            if mode_fd is not None and lock_owned:
                try:
                    os.unlink(lock_name, dir_fd=mode_fd)
                except FileNotFoundError:
                    pass
            for descriptor in (
                mode_fd,
                checkpoint_fd,
                evals_fd,
                run_fd,
            ):
                if descriptor is not None:
                    os.close(descriptor)
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._connection.close()
        self._database_path.unlink(missing_ok=True)
        self._closed = True


def validate_published_evaluation(
    cell: str | Path,
    *,
    row_consumer: Callable[[EvalRow], None] | None = None,
) -> CheckpointSummary:
    """Stream-verify one immutable cell, including hashes and metrics."""

    path = Path(cell)
    _ensure_regular_directory(path, "evaluation cell")
    if path.resolve(strict=True) != path.absolute():
        raise ValueError("evaluation cell cannot traverse symlink components")
    if {entry.name for entry in path.iterdir()} != {
        "rows.jsonl",
        "summary.json",
        "manifest.json",
    }:
        raise ValueError("evaluation cell file set is not exact")
    rows_path = path / "rows.jsonl"
    summary_path = path / "summary.json"
    manifest_path = path / "manifest.json"
    for file_path in (rows_path, summary_path, manifest_path):
        if file_path.is_symlink() or not file_path.is_file():
            raise ValueError("evaluation cell contains a non-regular file")
    summary = CheckpointSummary.from_dict(
        json.loads(summary_path.read_text(encoding="utf-8"))
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_fields = {
        "schema_version",
        "checkpoint_sha256",
        "memory_mode",
        "control_id",
        "n_rows",
        "file_sha256",
    }
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != expected_manifest_fields
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["checkpoint_sha256"] != summary.checkpoint_sha256
        or manifest["memory_mode"] != summary.memory_mode
        or manifest["control_id"] != summary.control_id
        or manifest["n_rows"] != summary.n_rows
        or not isinstance(manifest["file_sha256"], Mapping)
        or set(manifest["file_sha256"])
        != {"rows.jsonl", "summary.json"}
    ):
        raise ValueError("evaluation cell manifest contract is invalid")
    summary_content = summary_path.read_bytes()
    if summary_content != canonical_json_bytes(summary):
        raise ValueError("checkpoint summary is not canonical")
    summary_hash = hashlib.sha256(summary_content).hexdigest()
    if manifest["file_sha256"]["summary.json"] != summary_hash:
        raise ValueError("checkpoint summary manifest hash mismatch")

    with tempfile.TemporaryDirectory(
        prefix=".relational-history-audit-"
    ) as audit_directory:
        publisher = StreamingEvaluationPublisher(
            Path(audit_directory).resolve()
        )
        digest = hashlib.sha256()
        try:
            with rows_path.open("rb") as stream:
                for line_number, line in enumerate(stream, start=1):
                    digest.update(line)
                    try:
                        raw = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError(
                            f"invalid evaluation row at line {line_number}"
                        ) from exc
                    row = EvalRow.from_dict(raw)
                    if line != canonical_json_bytes(row):
                        raise ValueError(
                            f"non-canonical evaluation row at line {line_number}"
                        )
                    if row_consumer is not None:
                        row_consumer(row)
                    publisher.add(row)
            recomputed = publisher.preview_summary(
                milestone_crossings=summary.metrics[
                    "milestone_crossings"
                ]
            )
        finally:
            publisher.close()
    rows_hash = digest.hexdigest()
    if rows_hash != summary.rows_sha256:
        raise ValueError("rows_sha256 does not match published rows")
    if manifest["file_sha256"]["rows.jsonl"] != rows_hash:
        raise ValueError("evaluation rows manifest hash mismatch")
    if recomputed.to_dict() != summary.to_dict():
        raise ValueError(
            "checkpoint summary does not match recomputed row metrics"
        )
    return summary


def publish_evaluation(
    run: str | Path,
    rows: Iterable[EvalRow | Mapping[str, Any]],
    summary: CheckpointSummary | Mapping[str, Any],
) -> Path:
    """Atomically publish one immutable checkpoint/mode/control result."""
    publisher = StreamingEvaluationPublisher(run)
    try:
        for row in rows:
            publisher.add(row)
        summary_value = CheckpointSummary.from_dict(
            summary.to_dict()
            if isinstance(summary, CheckpointSummary)
            else summary
        )
        destination, _ = publisher.finish(
            milestone_crossings=summary_value.metrics[
                "milestone_crossings"
            ],
            expected_summary=summary_value,
        )
        return destination
    finally:
        publisher.close()
