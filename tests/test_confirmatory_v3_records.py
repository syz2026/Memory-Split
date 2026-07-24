from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib

import pytest

import evals.confirmatory as confirmatory
from evals.confirmatory import contracts as contracts_module
from evals.confirmatory import metrics as metrics_module
from evals.confirmatory.contracts import (
    Arm,
    CHECKPOINT_SCHEMA,
    CONTRACT_VERSION,
    ITEM_SCHEMA,
    SEALED_GOLD_SCHEMA,
    STORE_SCHEMA,
    STUDY_CHECKPOINT_SCHEMA,
    STUDY_CONTRACT_VERSION,
    STUDY_TARGETS_PER_UPDATE,
    ItemRecord,
    StudyCheckpointRecord,
)
from evals.confirmatory.metrics import (
    PairMetricSummary,
    Rate,
    STUDY_METRICS_SCHEMA,
    STUDY_OUTCOME_SCHEMA,
    StudyMetricsRecord,
    StudyOutcomeRecord,
    validate_study_outcome_binding,
)
from evals.confirmatory.study_lock import (
    FROZEN_PREREGISTRATION_SHA256_V3,
    STUDY_LOCK_SCHEMA_V3,
    StudyLockV3,
    StudySnapshotBinding,
)
from msctl.aws_contracts import (
    ARMS,
    SEEDS,
    SNAPSHOT_STEPS,
    checkpoint_receipt_key,
    snapshot_object_key,
)
from msctl.aws_hardware import (
    AWS_HARDWARE_AMENDMENT_SHA256,
    PROVIDER_SELECTION_S3_KEY,
)


STEP = 1_358
RAW_TOKENS = STEP * 524_288


def test_confirmatory_package_exports_the_v3_foundation_api():
    assert confirmatory.StudyCheckpointRecord is StudyCheckpointRecord
    assert confirmatory.StudyOutcomeRecord is StudyOutcomeRecord
    assert confirmatory.StudyMetricsRecord is StudyMetricsRecord
    assert confirmatory.StudyArm is contracts_module.StudyArm
    assert confirmatory.STUDY_CONTRACT_VERSION == 3
    assert confirmatory.STUDY_LOCK_SCHEMA_V3.endswith(".study-lock.v3")
    assert callable(confirmatory.v3_exact_sign_flip_test)
    assert callable(confirmatory.v3_practical_equivalence_bounds)
    assert callable(confirmatory.v3_right_step_aulc)
    assert callable(confirmatory.exact_study_paired_delta)


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
        "arm": "split90",
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
        "arm": "split90",
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


def _item(**changes) -> ItemRecord:
    value = {
        "record_type": ITEM_SCHEMA,
        "schema_version": CONTRACT_VERSION,
        "item_id": "item-1",
        "pair_id": "pair-1",
        "twin": "original",
        "stratum": "composition_ood",
        "family": "graph",
        "world_id": "world-1",
        "task": "path_composition",
        "path_length": 2,
        "composition_split": "heldout",
        "composition_id": "P1/P2",
        "prompt": "fixture prompt",
        "initial_slots": ["Q1", None, None, None],
        "store_id": "store-1",
        "memory_mode": "memory_on",
        "control": "correct",
    }
    value.update(changes)
    return ItemRecord.from_dict(value)


def _snapshot_digest(seed: int, arm: str, step: int) -> str:
    if (seed, arm, step) == (0, "split90", STEP):
        return "a" * 64
    return hashlib.sha256(f"{seed}:{arm}:{step}".encode("ascii")).hexdigest()


def _study_lock() -> StudyLockV3:
    selection_sha256 = "6" * 64
    selection_version_id = "provider-selection-version-p5"
    snapshots = []
    for seed in SEEDS:
        for arm in ARMS:
            for step in SNAPSHOT_STEPS:
                digest = _snapshot_digest(seed, arm, step)
                receipt_digest = hashlib.sha256(
                    f"receipt:{seed}:{step}".encode("ascii")
                ).hexdigest()
                snapshots.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "optimizer_step": step,
                        "checkpoint_sha256": digest,
                        "s3_object_key": snapshot_object_key(
                            seed,
                            arm,
                            step,
                            digest,
                        ),
                        "s3_version_id": f"version-{seed}-{arm}-{step}",
                        "checkpoint_receipt_sha256": receipt_digest,
                        "checkpoint_receipt_s3_object_key": (
                            checkpoint_receipt_key(seed, receipt_digest)
                        ),
                        "checkpoint_receipt_s3_version_id": (
                            f"receipt-version-{seed}-{step}"
                        ),
                        "provider_selection_sha256": selection_sha256,
                        "provider_selection_s3_version_id": (
                            selection_version_id
                        ),
                        "snapshot_version": 2,
                        "training_run_id": (
                            f"memorysplit-v3-360m-s{seed}-{arm}"
                        ),
                        "config_fingerprint": hashlib.sha256(
                            f"config:{seed}:{arm}".encode("ascii")
                        ).hexdigest(),
                        "training_config_sha256": hashlib.sha256(
                            f"config-bytes:{seed}:{arm}".encode("ascii")
                        ).hexdigest(),
                        "model_config_sha256": hashlib.sha256(
                            b"model-config"
                        ).hexdigest(),
                        "model_identity": "d360m",
                        "data_provenance_sha256": hashlib.sha256(
                            f"data:{seed}:{arm}".encode("ascii")
                        ).hexdigest(),
                        "data_receipt_sha256": hashlib.sha256(
                            b"data-receipt"
                        ).hexdigest(),
                        "data_build_id": hashlib.sha256(
                            b"data-build"
                        ).hexdigest(),
                        "ordered_stream_sha256": hashlib.sha256(
                            b"ordered-stream"
                        ).hexdigest(),
                        "world_size": 4,
                        "tokens_per_step": 524_288,
                    }
                )
    return StudyLockV3.from_dict(
        {
            "record_type": STUDY_LOCK_SCHEMA_V3,
            "schema_version": STUDY_CONTRACT_VERSION,
            "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256_V3,
            "sealed_evaluation_release_sha256": "f" * 64,
            "provider_selection": {
                "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
                "provider_selection_s3_key": PROVIDER_SELECTION_S3_KEY,
                "provider_selection_sha256": selection_sha256,
                "provider_selection_s3_version_id": selection_version_id,
                "hardware_amendment_sha256": AWS_HARDWARE_AMENDMENT_SHA256,
                "selected_provider": "aws-p5.48xlarge",
                "profile_id": "aws-p5.48xlarge-v3",
                "profile_sha256": (
                    "2207bfbad5e8fa9fc804770b582d0b21f8b6ed109b2e3f3b5c0474c732c53543"
                ),
                "runtime_lock_sha256": "7" * 64,
                "qualification_evidence_sha256": "8" * 64,
                "environment_receipt_sha256": "9" * 64,
                "canary_receipt_sha256": "a" * 64,
                "approval_receipt_sha256": "b" * 64,
                "approval_public_key_sha256": "c" * 64,
            },
            "snapshots": snapshots,
        }
    )


def _rate(numerator: int, denominator: int) -> dict:
    return {
        "value": numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
    }


def test_v2_pair_metric_keeps_legacy_float_aggregation_with_exact_v3_path():
    primary_rates = {
        "graph__composition_ood": Rate(0, 1),
        "graph__joint_ood": Rate(0, 1),
        "non_path__composition_ood": Rate(1, 1),
        "non_path__joint_ood": Rate(2, 3),
    }
    legacy_float = sum(rate.value for rate in primary_rates.values()) / 4
    exact_value = Fraction(5, 12)

    assert legacy_float != float(exact_value)
    summary = PairMetricSummary(
        primary_accuracy=legacy_float,
        primary_cells=primary_rates,
        overall_pair_accuracy=Rate(0, 2),
        by_stratum={contracts_module.Stratum.COMPOSITION_OOD: Rate(0, 2)},
        by_family={
            contracts_module.ReasoningFamily.GRAPH: Rate(0, 1),
            contracts_module.ReasoningFamily.NON_PATH: Rate(0, 1),
        },
        checkpoint_sha256="6" * 64,
        arm=Arm.DENSE,
        condition_id="dense",
        memory_mode="memory_on",
        control="correct",
    )

    assert summary.primary_accuracy == legacy_float
    assert summary.exact_primary_accuracy == exact_value
    assert summary.to_dict()["primary_accuracy"] == legacy_float


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
        "arm": "split90",
        "condition_id": "split90",
        "optimizer_step": STEP,
        "raw_token_count": RAW_TOKENS,
        "memory_mode": "memory_on",
        "control": "correct",
    }
    value.update(changes)
    return value


def _metrics_at_rate(
    numerator: int,
    denominator: int,
    *,
    arm: str,
    condition_id: str,
    checkpoint_sha256: str,
) -> StudyMetricsRecord:
    record = _metrics(
        primary_accuracy=numerator / denominator,
        primary_cells={
            cell: _rate(numerator, denominator)
            for cell in (
                "graph__composition_ood",
                "graph__joint_ood",
                "non_path__composition_ood",
                "non_path__joint_ood",
            )
        },
        overall_pair_accuracy=_rate(4 * numerator, 4 * denominator),
        by_stratum={
            "composition_ood": _rate(2 * numerator, 2 * denominator),
            "joint_ood": _rate(2 * numerator, 2 * denominator),
        },
        by_family={
            "graph": _rate(2 * numerator, 2 * denominator),
            "non_path": _rate(2 * numerator, 2 * denominator),
        },
        arm=arm,
        condition_id=condition_id,
        checkpoint_sha256=checkpoint_sha256,
    )
    return StudyMetricsRecord.from_dict(record)


def test_v3_paired_metric_delta_preserves_exact_rate_counts_at_margin():
    dense = _metrics_at_rate(
        10,
        100,
        arm="dense",
        condition_id="dense",
        checkpoint_sha256="9" * 64,
    )
    split90 = _metrics_at_rate(
        11,
        100,
        arm="split90",
        condition_id="split90",
        checkpoint_sha256="a" * 64,
    )

    delta = metrics_module.exact_study_paired_delta(
        split90=split90,
        dense=dense,
    )

    assert delta == Fraction(1, 100)
    assert not confirmatory.v3_supports_practical_equivalence(-delta, delta)


def test_v3_study_records_round_trip_without_versioning_semantic_records():
    checkpoint = StudyCheckpointRecord.from_dict(_checkpoint())
    outcome = StudyOutcomeRecord.from_dict(_outcome())
    metrics = StudyMetricsRecord.from_dict(_metrics())

    assert checkpoint.to_dict() == _checkpoint()
    assert outcome.to_dict() == _outcome()
    assert metrics.to_dict() == _metrics()
    assert checkpoint.arm is contracts_module.StudyArm.SPLIT90
    assert outcome.arm is contracts_module.StudyArm.SPLIT90
    assert metrics.arm is contracts_module.StudyArm.SPLIT90
    assert STUDY_TARGETS_PER_UPDATE == 524_288

    assert CONTRACT_VERSION == 2
    assert ITEM_SCHEMA.endswith(".item.v2")
    assert SEALED_GOLD_SCHEMA.endswith(".sealed-gold.v2")
    assert STORE_SCHEMA.endswith(".store.v2")
    assert CHECKPOINT_SCHEMA.endswith(".checkpoint.v2")


def test_v3_study_arm_is_explicit_without_changing_legacy_v2_arm():
    assert {arm.value for arm in contracts_module.StudyArm} == {
        "dense",
        "split90",
    }
    assert {arm.value for arm in Arm} == {"dense", "split", "random"}


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


def _binding_records():
    checkpoint = StudyCheckpointRecord.from_dict(_checkpoint())
    outcome = StudyOutcomeRecord.from_dict(_outcome())
    item = _item()
    lock = _study_lock()
    snapshot = next(
        snapshot
        for snapshot in lock.snapshots
        if (
            snapshot.seed,
            snapshot.arm,
            snapshot.optimizer_step,
        )
        == (0, "split90", STEP)
    )
    return item, checkpoint, outcome, snapshot, lock


def test_v3_outcome_binding_authenticates_item_checkpoint_and_lock_snapshot():
    item, checkpoint, outcome, snapshot, lock = _binding_records()

    assert (
        validate_study_outcome_binding(
            outcome=outcome,
            item=item,
            checkpoint=checkpoint,
            snapshot=snapshot,
            lock=lock,
        )
        is outcome
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("item_id", "other-item"),
        ("pair_id", "other-pair"),
        ("twin", "counterfactual"),
        ("world_id", "other-world"),
        ("family", "non_path"),
        ("stratum", "joint_ood"),
        ("memory_mode", "memory_off"),
        ("control", "no_query"),
    ],
)
def test_v3_outcome_binding_rejects_every_crossed_item_identity(field, value):
    item, checkpoint, outcome, snapshot, lock = _binding_records()

    with pytest.raises(ValueError, match=field):
        validate_study_outcome_binding(
            outcome=replace(outcome, **{field: value}),
            item=item,
            checkpoint=checkpoint,
            snapshot=snapshot,
            lock=lock,
        )


def test_v3_outcome_binding_rejects_checkpoint_and_object_replacements():
    item, checkpoint, outcome, snapshot, lock = _binding_records()
    later_step = 3_396

    with pytest.raises(ValueError, match="optimizer_step"):
        validate_study_outcome_binding(
            outcome=replace(
                outcome,
                optimizer_step=later_step,
                raw_token_count=later_step * STUDY_TARGETS_PER_UPDATE,
            ),
            item=item,
            checkpoint=checkpoint,
            snapshot=snapshot,
            lock=lock,
        )

    replacement_hash = "8" * 64
    replacement_receipt_hash = "7" * 64
    replacement = StudySnapshotBinding(
        seed=snapshot.seed,
        arm=snapshot.arm,
        optimizer_step=snapshot.optimizer_step,
        checkpoint_sha256=replacement_hash,
        s3_object_key=snapshot_object_key(
            snapshot.seed,
            snapshot.arm,
            snapshot.optimizer_step,
            replacement_hash,
        ),
        s3_version_id="replacement-version",
        checkpoint_receipt_sha256=replacement_receipt_hash,
        checkpoint_receipt_s3_object_key=checkpoint_receipt_key(
            snapshot.seed,
            replacement_receipt_hash,
        ),
        checkpoint_receipt_s3_version_id="replacement-receipt-version",
        provider_selection_sha256=snapshot.provider_selection_sha256,
        provider_selection_s3_version_id=(
            snapshot.provider_selection_s3_version_id
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
    )
    with pytest.raises(ValueError, match="lock|object|checkpoint"):
        validate_study_outcome_binding(
            outcome=outcome,
            item=item,
            checkpoint=checkpoint,
            snapshot=replacement,
            lock=lock,
        )
