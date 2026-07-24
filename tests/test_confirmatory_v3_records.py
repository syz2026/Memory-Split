from __future__ import annotations

from dataclasses import replace

import pytest

import evals.confirmatory as confirmatory
from evals.confirmatory.contracts import (
    CHECKPOINT_SCHEMA,
    CONTRACT_VERSION,
    ITEM_SCHEMA,
    SEALED_GOLD_SCHEMA,
    STORE_SCHEMA,
    STUDY_CHECKPOINT_SCHEMA,
    STUDY_CONTRACT_VERSION,
    STUDY_TARGETS_PER_UPDATE,
    StudyCheckpointRecord,
)
from evals.confirmatory.metrics import (
    STUDY_METRICS_SCHEMA,
    STUDY_OUTCOME_SCHEMA,
    StudyMetricsRecord,
    StudyOutcomeRecord,
    validate_study_outcome_binding,
)


STEP = 1_358
RAW_TOKENS = STEP * 524_288


def test_confirmatory_package_exports_the_v3_foundation_api():
    assert confirmatory.StudyCheckpointRecord is StudyCheckpointRecord
    assert confirmatory.StudyOutcomeRecord is StudyOutcomeRecord
    assert confirmatory.StudyMetricsRecord is StudyMetricsRecord
    assert confirmatory.STUDY_CONTRACT_VERSION == 3
    assert confirmatory.STUDY_LOCK_SCHEMA_V3.endswith(".study-lock.v3")
    assert callable(confirmatory.v3_exact_sign_flip_test)
    assert callable(confirmatory.v3_practical_equivalence_bounds)
    assert callable(confirmatory.v3_right_step_aulc)


def _proof() -> list[dict]:
    return [
        {
            "source_slot": None,
            "relation_id": None,
            "direction": None,
            "op": "halt",
        },
        *[
            {
                "source_slot": None,
                "relation_id": None,
                "direction": None,
                "op": "noop",
            }
            for _ in range(11)
        ],
    ]


def _checkpoint(**changes) -> dict:
    value = {
        "record_type": STUDY_CHECKPOINT_SCHEMA,
        "schema_version": STUDY_CONTRACT_VERSION,
        "checkpoint_sha256": "a" * 64,
        "model_id": "memorysplit-360m-v3",
        "arm": "split",
        "condition_id": "split90",
        "seed": 0,
        "optimizer_step": STEP,
        "raw_token_count": RAW_TOKENS,
        "configuration_sha256": "b" * 64,
        "route_dose_sha256": "c" * 64,
        "corpus_sha256": "d" * 64,
        "code_sha256": "e" * 64,
    }
    value.update(changes)
    return value


def _outcome(**changes) -> dict:
    value = {
        "record_type": STUDY_OUTCOME_SCHEMA,
        "schema_version": STUDY_CONTRACT_VERSION,
        "item_id": "item-1",
        "pair_id": "pair-1",
        "twin": "original",
        "stratum": "composition_ood",
        "family": "graph",
        "seed": 0,
        "world_id": "world-1",
        "checkpoint_sha256": "a" * 64,
        "arm": "split",
        "condition_id": "split90",
        "optimizer_step": STEP,
        "raw_token_count": RAW_TOKENS,
        "memory_mode": "memory_on",
        "control": "correct",
        "submitted_answer": "answer",
        "submitted_proof": _proof(),
    }
    value.update(changes)
    return value


def _rate(numerator: int, denominator: int) -> dict:
    return {
        "value": numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
    }


def _metrics(**changes) -> dict:
    value = {
        "record_type": STUDY_METRICS_SCHEMA,
        "schema_version": STUDY_CONTRACT_VERSION,
        "primary_accuracy": 1.0,
        "primary_cells": {
            "graph__composition_ood": _rate(1, 1),
            "graph__joint_ood": _rate(1, 1),
            "non_path__composition_ood": _rate(1, 1),
            "non_path__joint_ood": _rate(1, 1),
        },
        "overall_pair_accuracy": _rate(4, 4),
        "by_stratum": {
            "composition_ood": _rate(2, 2),
            "joint_ood": _rate(2, 2),
        },
        "by_family": {
            "graph": _rate(2, 2),
            "non_path": _rate(2, 2),
        },
        "checkpoint_sha256": "a" * 64,
        "seed": 0,
        "arm": "split",
        "condition_id": "split90",
        "optimizer_step": STEP,
        "raw_token_count": RAW_TOKENS,
        "memory_mode": "memory_on",
        "control": "correct",
    }
    value.update(changes)
    return value


def test_v3_study_records_round_trip_without_versioning_semantic_records():
    checkpoint = StudyCheckpointRecord.from_dict(_checkpoint())
    outcome = StudyOutcomeRecord.from_dict(_outcome())
    metrics = StudyMetricsRecord.from_dict(_metrics())

    assert checkpoint.to_dict() == _checkpoint()
    assert outcome.to_dict() == _outcome()
    assert metrics.to_dict() == _metrics()
    assert STUDY_TARGETS_PER_UPDATE == 524_288

    assert CONTRACT_VERSION == 2
    assert ITEM_SCHEMA.endswith(".item.v2")
    assert SEALED_GOLD_SCHEMA.endswith(".sealed-gold.v2")
    assert STORE_SCHEMA.endswith(".store.v2")
    assert CHECKPOINT_SCHEMA.endswith(".checkpoint.v2")


@pytest.mark.parametrize(
    ("factory", "record"),
    [
        (StudyCheckpointRecord.from_dict, _checkpoint()),
        (StudyOutcomeRecord.from_dict, _outcome()),
        (StudyMetricsRecord.from_dict, _metrics()),
    ],
)
def test_v3_study_records_require_an_allowed_step_and_exact_raw_tokens(
    factory,
    record,
):
    with pytest.raises(ValueError, match="optimizer_step"):
        factory({**record, "optimizer_step": 1_357})
    with pytest.raises(ValueError, match="optimizer_step"):
        factory({**record, "optimizer_step": True})
    with pytest.raises(ValueError, match="raw_token_count"):
        factory({**record, "raw_token_count": RAW_TOKENS + 1})


@pytest.mark.parametrize(
    ("factory", "record"),
    [
        (StudyCheckpointRecord.from_dict, _checkpoint()),
        (StudyOutcomeRecord.from_dict, _outcome()),
        (StudyMetricsRecord.from_dict, _metrics()),
    ],
)
def test_v3_study_records_are_restricted_to_n10_dense_and_split90(
    factory,
    record,
):
    with pytest.raises(ValueError, match="seed"):
        factory({**record, "seed": 10})
    with pytest.raises(ValueError, match="condition|arm"):
        factory(
            {
                **record,
                "arm": "random",
                "condition_id": "random",
            }
        )


def test_v3_outcome_binding_authenticates_step_and_raw_token_identity():
    checkpoint = StudyCheckpointRecord.from_dict(_checkpoint())
    outcome = StudyOutcomeRecord.from_dict(_outcome())

    assert (
        validate_study_outcome_binding(
            outcome=outcome,
            checkpoint=checkpoint,
        )
        is outcome
    )

    with pytest.raises(ValueError, match="optimizer_step"):
        validate_study_outcome_binding(
            outcome=replace(outcome, optimizer_step=3_396),
            checkpoint=checkpoint,
        )
    with pytest.raises(ValueError, match="checkpoint"):
        validate_study_outcome_binding(
            outcome=replace(outcome, checkpoint_sha256="f" * 64),
            checkpoint=checkpoint,
        )
