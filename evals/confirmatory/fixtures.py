"""Small deterministic positive, practical-null, and invalid study fixtures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from types import MappingProxyType

from evals.confirmatory.contracts import (
    CHECKPOINT_SCHEMA,
    CONTRACT_VERSION,
    ITEM_SCHEMA,
    SEALED_GOLD_SCHEMA,
    STORE_SCHEMA,
    Arm,
    CheckpointRecord,
    ConditionId,
    ItemRecord,
    SealedGoldRecord,
    StoreRecord,
    canonical_json_bytes,
    canonical_sha256,
    store_content_sha256,
)
from evals.confirmatory.actions import (
    ACTION_SLOTS,
    validate_action_slots,
)
from evals.confirmatory.inference import PairedObservation
from evals.confirmatory.inference import exact_sign_flip_test
from evals.confirmatory.metrics import (
    ItemOutcome,
    _ScoredItemOutcome,
    balanced_counterfactual_pair_metric,
    score_item_outcome,
)
from evals.confirmatory.reporting import (
    INFERENCE_EVIDENCE_SCHEMA,
    METRICS_SCHEMA,
    PRIMARY_CONTRAST_ID,
    PRIMARY_TEST_METHOD,
    REQUIRED_ARTIFACTS,
)
from evals.confirmatory.status import (
    FinalInferenceConclusion,
    InterimEvidenceLabel,
    ScientificStatus,
    StatusAxes,
)
from evals.confirmatory.study_lock import (
    FROZEN_PREREGISTRATION_SHA256,
    REQUIRED_CONTROL_IDS,
    REQUIRED_FAMILIES,
    REQUIRED_MEMORY_MODES,
    REQUIRED_RECEIPTS,
    REQUIRED_STRATA,
    STUDY_LOCK_SCHEMA,
    VALIDITY_EVIDENCE_SCHEMA,
)


DETERMINISTIC_STUDY_LOCK_SHA256 = (
    "a221a3a6bd21000a24b8a476bdbcae891005d113957860bcc2df7a47c7650340"
)


@dataclass(frozen=True)
class DeterministicFixture:
    name: str
    observations: tuple[PairedObservation, ...]
    complete: bool
    valid: bool
    sign_consistent: bool
    expected_status: StatusAxes
    artifacts: Mapping[str, bytes]
    expected_study_lock_sha256: str

    def __post_init__(self) -> None:
        if self.name not in {
            "positive",
            "null",
            "invalid",
            "pending",
            "shrunken",
            "tiny",
        }:
            raise ValueError("unknown deterministic fixture name")
        if not isinstance(self.expected_status, StatusAxes):
            raise ValueError("fixture expected_status must contain all three axes")
        if not self.observations:
            raise ValueError("deterministic fixture requires observations")
        if set(self.artifacts) != set(REQUIRED_ARTIFACTS) or any(
            not isinstance(content, bytes) or not content
            for content in self.artifacts.values()
        ):
            raise ValueError("deterministic fixture artifacts are not exact")
        object.__setattr__(
            self,
            "artifacts",
            MappingProxyType(dict(self.artifacts)),
        )
        if (
            not isinstance(self.expected_study_lock_sha256, str)
            or len(self.expected_study_lock_sha256) != 64
        ):
            raise ValueError("fixture study-lock commitment is invalid")
        for field in (
            "complete",
            "valid",
            "sign_consistent",
        ):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"{field} must be Boolean")


def _rows(*, treatment: float, control: float) -> tuple[PairedObservation, ...]:
    return tuple(
        PairedObservation(
            seed=seed,
            world_id=f"fixture-world-{world}",
            pair_id=f"seed-{seed}-world-{world}-pair-{pair}",
            treatment=treatment,
            control=control,
        )
        for seed in range(1001, 1006)
        for world in range(2)
        for pair in range(2)
    )


def _noop(op: str = "noop") -> dict:
    return {
        "source_slot": None,
        "relation_id": None,
        "direction": None,
        "op": op,
    }


def _fixture_proof() -> list[dict]:
    return [
        {
            "source_slot": 0,
            "relation_id": "P1",
            "direction": "out",
            "op": "read",
        },
        _noop("halt"),
        *[_noop() for _ in range(ACTION_SLOTS - 2)],
    ]


_CONTROL_VARIANTS = (
    ("correct_memory", "memory_on", "correct"),
    ("memory_off", "memory_off", "correct"),
    ("shuffled_returns", "memory_on", "shuffled_returns"),
    ("relevant_edge_swap", "memory_on", "relevant_edge"),
    ("irrelevant_edge_swap", "memory_on", "irrelevant_edge"),
    ("gold_path_replay", "memory_on", "gold_path"),
    ("no_query", "memory_on", "no_query"),
    ("entity_rename", "memory_on", "entity_rename"),
    ("graph_isomorphism", "memory_on", "graph_isomorphism"),
    ("page_order_permutation", "memory_on", "page_order_permutation"),
)


def _release_records(
    *,
    shrink: bool,
    tiny: bool,
) -> tuple[
    tuple[ItemRecord, ...],
    tuple[SealedGoldRecord, ...],
    tuple[StoreRecord, ...],
]:
    shapes = {
        "iid": (2, "seen"),
        "composition_ood": (2, "heldout"),
        "length_ood": (7, "seen"),
        "joint_ood": (7, "heldout"),
    }
    items = []
    gold_records = []
    stores = []
    for family in REQUIRED_FAMILIES:
        for stratum in REQUIRED_STRATA:
            path_length, composition_split = shapes[stratum]
            for control_id, memory_mode, control in _CONTROL_VARIANTS:
                if tiny and control_id != "correct_memory":
                    continue
                if (
                    shrink
                    and family == "graph"
                    and stratum == "iid"
                    and control_id == "correct_memory"
                ):
                    continue
                pair_id = f"{family}-{stratum}-{control_id}"
                world_id = f"world-{pair_id}"
                store_id = f"store-{pair_id}"
                rows = [
                    {
                        "source_id": "Q1",
                        "relation_id": "P1",
                        "direction": "out",
                        "target_kind": "literal",
                        "target": "done",
                        "qualifiers": {},
                    }
                ]
                content_sha256 = store_content_sha256(
                    store_id,
                    world_id,
                    rows,
                )
                store = StoreRecord.from_dict(
                    {
                        "record_type": STORE_SCHEMA,
                        "schema_version": CONTRACT_VERSION,
                        "store_id": store_id,
                        "world_id": world_id,
                        "rows": rows,
                        "content_sha256": content_sha256,
                    }
                )
                stores.append(store)
                for twin in ("original", "counterfactual"):
                    item_id = f"{pair_id}-{twin}"
                    item = ItemRecord.from_dict(
                        {
                            "record_type": ITEM_SCHEMA,
                            "schema_version": CONTRACT_VERSION,
                            "item_id": item_id,
                            "pair_id": pair_id,
                            "twin": twin,
                            "stratum": stratum,
                            "family": family,
                            "world_id": world_id,
                            "task": "fixture_task",
                            "path_length": path_length,
                            "composition_split": composition_split,
                            "composition_id": f"composition-{pair_id}",
                            "prompt": "Follow P1 from Q1.",
                            "initial_slots": ["Q1", None, None, None],
                            "store_id": store_id,
                            "memory_mode": memory_mode,
                            "control": control,
                        }
                    )
                    items.append(item)
                    gold_records.append(
                        SealedGoldRecord.from_dict(
                            {
                                "record_type": SEALED_GOLD_SCHEMA,
                                "schema_version": CONTRACT_VERSION,
                                "item_id": item_id,
                                "pair_id": pair_id,
                                "twin": twin,
                                "answer": "done",
                                "proof": _fixture_proof(),
                                "solver_id": "lookup-chain-v1",
                                "store_sha256": content_sha256,
                            }
                        )
                    )
    return (
        tuple(sorted(items, key=lambda record: record.item_id)),
        tuple(sorted(gold_records, key=lambda record: record.item_id)),
        tuple(sorted(stores, key=lambda record: record.store_id)),
    )


def _artifact_checkpoints() -> tuple[CheckpointRecord, ...]:
    records = []
    for seed in range(5):
        for condition_id, arm in (
            (ConditionId.DENSE, Arm.DENSE),
            (ConditionId.SPLIT90, Arm.SPLIT),
        ):
            records.append(
                CheckpointRecord.from_dict(
                    {
                        "record_type": CHECKPOINT_SCHEMA,
                        "schema_version": CONTRACT_VERSION,
                        "checkpoint_sha256": hashlib.sha256(
                            f"{seed}:{condition_id.value}".encode()
                        ).hexdigest(),
                        "model_id": "fixture-model",
                        "arm": arm.value,
                        "condition_id": condition_id.value,
                        "seed": seed,
                        "raw_token_count": 1,
                        "configuration_sha256": hashlib.sha256(
                            f"configuration:{seed}:{condition_id.value}".encode()
                        ).hexdigest(),
                        "route_dose_sha256": hashlib.sha256(
                            f"route-dose:{seed}:{condition_id.value}".encode()
                        ).hexdigest(),
                        "corpus_sha256": "c" * 64,
                        "code_sha256": "d" * 64,
                    }
                )
            )
    return tuple(
        sorted(
            records,
            key=lambda record: (record.seed, record.condition_id.value),
        )
    )


def _jsonl(records) -> bytes:
    return b"".join(canonical_json_bytes(record) for record in records)


def _artifact_bundle(
    *,
    split_correct: bool,
    dense_correct: bool,
    receipt_state: str,
    shrink: bool = False,
    tiny: bool = False,
) -> tuple[Mapping[str, bytes], str]:
    items, gold_records, stores = _release_records(
        shrink=shrink,
        tiny=tiny,
    )
    checkpoints = _artifact_checkpoints()
    item_map = {item.item_id: item for item in items}
    gold_map = {gold.item_id: gold for gold in gold_records}
    store_map = {store.store_id: store for store in stores}
    outcomes = []
    summaries = []
    for checkpoint in checkpoints:
        correct = (
            split_correct
            if checkpoint.condition_id is ConditionId.SPLIT90
            else dense_correct
        )
        submissions = [
            ItemOutcome(
                item_id=item.item_id,
                pair_id=item.pair_id,
                twin=item.twin,
                stratum=item.stratum,
                family=item.family,
                seed=checkpoint.seed,
                world_id=item.world_id,
                checkpoint_sha256=checkpoint.checkpoint_sha256,
                arm=checkpoint.arm,
                condition_id=checkpoint.condition_id,
                memory_mode=item.memory_mode,
                control=item.control,
                submitted_answer="done" if correct else "wrong",
                submitted_proof=validate_action_slots(_fixture_proof()),
            )
            for item in items
        ]
        outcomes.extend(submissions)
        scored = [
            score_item_outcome(
                outcome=submission,
                item=item_map[submission.item_id],
                checkpoint=checkpoint,
                gold=gold_map[submission.item_id],
                store=store_map[item_map[submission.item_id].store_id],
            )
            for submission in submissions
        ]
        grouped: dict[
            tuple[object, object],
            list[_ScoredItemOutcome],
        ] = {}
        for row in scored:
            grouped.setdefault((row.memory_mode, row.control), []).append(row)
        for rows in grouped.values():
            row_items = {row.item_id: item_map[row.item_id] for row in rows}
            summaries.append(
                balanced_counterfactual_pair_metric(
                    rows,
                    items=row_items,
                    checkpoints={
                        checkpoint.checkpoint_sha256: checkpoint,
                    },
                )
            )
    checkpoint_map = {
        checkpoint.checkpoint_sha256: checkpoint
        for checkpoint in checkpoints
    }
    summaries.sort(
        key=lambda summary: (
            checkpoint_map[summary.checkpoint_sha256].seed,
            summary.condition_id.value,
            summary.memory_mode.value,
            summary.control.value,
        )
    )
    deltas = []
    for seed in range(5):
        by_condition = {
            summary.condition_id: summary.primary_accuracy
            for summary in summaries
            if checkpoint_map[summary.checkpoint_sha256].seed == seed
            and summary.memory_mode.value == "memory_on"
            and summary.control.value == "correct"
        }
        deltas.append(
            by_condition[ConditionId.SPLIT90]
            - by_condition[ConditionId.DENSE]
        )
    exact = exact_sign_flip_test(deltas, alternative="greater")
    decision = exact.statistic > 0.0 and exact.p_value <= 0.05
    inference = {
        "record_type": INFERENCE_EVIDENCE_SCHEMA,
        "schema_version": CONTRACT_VERSION,
        "primary_test": {
            "contrast_id": PRIMARY_CONTRAST_ID,
            "method": PRIMARY_TEST_METHOD,
            "alternative": "greater",
            "alpha": 0.05,
            "n_pairs": 5,
            "sign_assignments": 32,
            "equality_counted": True,
        },
        "paired_seed_bundle_deltas": deltas,
        "exact_test_result": {
            "statistic": exact.statistic,
            "extreme_count": exact.extreme_count,
            "p_value": exact.p_value,
            "reject_null": decision,
        },
    }
    ordered_outcomes = sorted(
        outcomes,
        key=lambda row: (row.seed, row.condition_id.value, row.item_id),
    )
    items_bytes = _jsonl(item.to_dict() for item in items)
    gold_bytes = _jsonl(gold.to_dict() for gold in gold_records)
    stores_bytes = _jsonl(store.to_dict() for store in stores)
    checkpoint_bytes = _jsonl(
        checkpoint.to_dict() for checkpoint in checkpoints
    )
    receipt_state_by_id = {
        receipt_id: (
            receipt_state
            if receipt_id == "gate:route_dose"
            else "passed"
        )
        for receipt_id in REQUIRED_RECEIPTS
    }
    receipts = [
        {
            "receipt_id": receipt_id,
            "kind": receipt_id.split(":", 1)[0],
            "state": receipt_state_by_id[receipt_id],
            "evidence_sha256": hashlib.sha256(
                f"evidence:{receipt_id}".encode()
            ).hexdigest(),
        }
        for receipt_id in REQUIRED_RECEIPTS
    ]
    approvals = [
        {
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "seed": checkpoint.seed,
            "condition_id": checkpoint.condition_id.value,
            "configuration_sha256": checkpoint.configuration_sha256,
            "route_dose_sha256": checkpoint.route_dose_sha256,
        }
        for checkpoint in checkpoints
    ]
    evaluation_cells = [
        {
            "item_id": item.item_id,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "seed": checkpoint.seed,
            "condition_id": checkpoint.condition_id.value,
        }
        for checkpoint in checkpoints
        for item in items
    ]
    study_lock = {
        "record_type": STUDY_LOCK_SCHEMA,
        "schema_version": CONTRACT_VERSION,
        "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
        "release": {
            "items_sha256": hashlib.sha256(items_bytes).hexdigest(),
            "sealed_gold_sha256": hashlib.sha256(gold_bytes).hexdigest(),
            "stores_sha256": hashlib.sha256(stores_bytes).hexdigest(),
            "checkpoints_sha256": hashlib.sha256(
                checkpoint_bytes
            ).hexdigest(),
            "item_ids": [item.item_id for item in items],
            "pair_ids": sorted({item.pair_id for item in items}),
            "world_ids": sorted({item.world_id for item in items}),
            "evaluation_cells": evaluation_cells,
            "item_count": len(items),
            "pair_count": len({item.pair_id for item in items}),
            "world_count": len({item.world_id for item in items}),
            "evaluation_cell_count": len(evaluation_cells),
            "required_families": list(REQUIRED_FAMILIES),
            "required_strata": list(REQUIRED_STRATA),
            "required_memory_modes": list(REQUIRED_MEMORY_MODES),
            "required_controls": list(REQUIRED_CONTROL_IDS),
        },
        "checkpoints": approvals,
        "validity_receipts": [
            {
                "receipt_id": receipt["receipt_id"],
                "kind": receipt["kind"],
                "state": receipt["state"],
                "receipt_sha256": canonical_sha256(receipt),
            }
            for receipt in receipts
        ],
    }
    study_lock_bytes = canonical_json_bytes(study_lock)
    study_lock_sha256 = hashlib.sha256(study_lock_bytes).hexdigest()
    validity = {
        "record_type": VALIDITY_EVIDENCE_SCHEMA,
        "schema_version": CONTRACT_VERSION,
        "study_lock_sha256": study_lock_sha256,
        "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
        "receipts": receipts,
    }
    artifacts = {
        "checkpoints.jsonl": _jsonl(
            checkpoint.to_dict() for checkpoint in checkpoints
        ),
        "inference.json": canonical_json_bytes(inference),
        "items.jsonl": items_bytes,
        "metrics.json": canonical_json_bytes(
            {
                "record_type": METRICS_SCHEMA,
                "schema_version": CONTRACT_VERSION,
                "summaries": [
                    summary.to_dict() for summary in summaries
                ],
            }
        ),
        "outcomes.jsonl": _jsonl(
            outcome.to_dict() for outcome in ordered_outcomes
        ),
        "sealed-gold.jsonl": gold_bytes,
        "study-lock.json": study_lock_bytes,
        "stores.jsonl": stores_bytes,
        "validity.json": canonical_json_bytes(validity),
    }
    return artifacts, study_lock_sha256


def _make_fixture(
    *,
    name: str,
    split_correct: bool,
    dense_correct: bool,
    receipt_state: str,
    expected_status: StatusAxes,
    shrink: bool = False,
    tiny: bool = False,
) -> DeterministicFixture:
    artifacts, study_lock_sha256 = _artifact_bundle(
        split_correct=split_correct,
        dense_correct=dense_correct,
        receipt_state=receipt_state,
        shrink=shrink,
        tiny=tiny,
    )
    if (
        name in {"positive", "null"}
        and study_lock_sha256 != DETERMINISTIC_STUDY_LOCK_SHA256
    ):
        raise ValueError("deterministic fixture study-lock commitment drifted")
    return DeterministicFixture(
        name=name,
        observations=_rows(
            treatment=0.75 if split_correct and not dense_correct else 0.50,
            control=0.50,
        ),
        complete=receipt_state == "passed",
        valid=receipt_state != "failed",
        sign_consistent=split_correct != dense_correct,
        expected_status=expected_status,
        artifacts=artifacts,
        expected_study_lock_sha256=study_lock_sha256,
    )


@lru_cache(maxsize=1)
def positive_fixture() -> DeterministicFixture:
    return _make_fixture(
        name="positive",
        split_correct=True,
        dense_correct=False,
        receipt_state="passed",
        expected_status=StatusAxes(
            ScientificStatus.COMPLETE,
            InterimEvidenceLabel.NONE,
            FinalInferenceConclusion.SUPPORTS_EFFECT,
        ),
    )


@lru_cache(maxsize=1)
def null_fixture() -> DeterministicFixture:
    return _make_fixture(
        name="null",
        split_correct=True,
        dense_correct=True,
        receipt_state="passed",
        expected_status=StatusAxes(
            ScientificStatus.COMPLETE,
            InterimEvidenceLabel.NONE,
            FinalInferenceConclusion.INCONCLUSIVE,
        ),
    )


@lru_cache(maxsize=1)
def invalid_fixture() -> DeterministicFixture:
    return _make_fixture(
        name="invalid",
        split_correct=True,
        dense_correct=False,
        receipt_state="failed",
        expected_status=StatusAxes(
            ScientificStatus.INVALID,
            InterimEvidenceLabel.NONE,
            FinalInferenceConclusion.NOT_EVALUATED,
        ),
    )


@lru_cache(maxsize=1)
def pending_fixture() -> DeterministicFixture:
    return _make_fixture(
        name="pending",
        split_correct=True,
        dense_correct=False,
        receipt_state="pending",
        expected_status=StatusAxes(
            ScientificStatus.INCOMPLETE,
            InterimEvidenceLabel.SIGN_CONSISTENT_ONLY,
            FinalInferenceConclusion.NOT_EVALUATED,
        ),
    )


@lru_cache(maxsize=1)
def shrunken_fixture() -> DeterministicFixture:
    return _make_fixture(
        name="shrunken",
        split_correct=True,
        dense_correct=False,
        receipt_state="passed",
        shrink=True,
        expected_status=StatusAxes(
            ScientificStatus.COMPLETE,
            InterimEvidenceLabel.NONE,
            FinalInferenceConclusion.SUPPORTS_EFFECT,
        ),
    )


@lru_cache(maxsize=1)
def tiny_fixture() -> DeterministicFixture:
    return _make_fixture(
        name="tiny",
        split_correct=True,
        dense_correct=False,
        receipt_state="passed",
        tiny=True,
        expected_status=StatusAxes(
            ScientificStatus.INCOMPLETE,
            InterimEvidenceLabel.NONE,
            FinalInferenceConclusion.NOT_EVALUATED,
        ),
    )


def fixture_by_name(name: str) -> DeterministicFixture:
    fixtures = {
        "positive": positive_fixture,
        "null": null_fixture,
        "invalid": invalid_fixture,
        "pending": pending_fixture,
        "shrunken": shrunken_fixture,
        "tiny": tiny_fixture,
    }
    try:
        factory = fixtures[name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown deterministic fixture: {name!r}") from exc
    return factory()


practical_null_fixture = null_fixture
