from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

import evals.confirmatory as confirmatory
from evals.confirmatory.aggregate import (
    RUN_BINDING_SCHEMA_V3,
    RunBindingV3,
    SnapshotEvaluationPlan,
    plan_snapshot_evaluations,
    validate_snapshot_evaluation_plans,
)
from evals.confirmatory.contracts import (
    STUDY_CONTRACT_VERSION,
    canonical_json_bytes,
    canonical_sha256,
)
from evals.confirmatory.study_lock import (
    EXPECTED_STUDY_SLOTS_V3,
    FROZEN_PREREGISTRATION_SHA256_V3,
    STUDY_LOCK_SCHEMA_V3,
    StudyLockV3,
)
from msctl.aws_contracts import (
    ARMS,
    SEEDS,
    SNAPSHOT_STEPS,
    checkpoint_object_key,
    checkpoint_receipt_key,
)
from msctl.aws_hardware import (
    AWS_HARDWARE_AMENDMENT_SHA256,
    PROVIDER_SELECTION_S3_KEY,
)


P5_PROFILE_SHA256 = (
    "2207bfbad5e8fa9fc804770b582d0b21f8b6ed109b2e3f3b5c0474c732c53543"
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _lock() -> StudyLockV3:
    selection_sha256 = "a" * 64
    selection_version = "provider-selection-version-p5"
    snapshots = []
    for seed in SEEDS:
        for arm in ARMS:
            for step in SNAPSHOT_STEPS:
                checkpoint_sha256 = _digest(f"checkpoint:{seed}:{arm}:{step}")
                receipt_sha256 = _digest(f"receipt:{seed}:{step}")
                snapshots.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "optimizer_step": step,
                        "checkpoint_sha256": checkpoint_sha256,
                        "s3_object_key": checkpoint_object_key(
                            seed,
                            arm,
                            checkpoint_sha256,
                        ),
                        "s3_version_id": f"checkpoint-version-{seed}-{arm}-{step}",
                        "checkpoint_receipt_sha256": receipt_sha256,
                        "checkpoint_receipt_s3_object_key": (
                            checkpoint_receipt_key(seed, receipt_sha256)
                        ),
                        "checkpoint_receipt_s3_version_id": (
                            f"receipt-version-{seed}-{step}"
                        ),
                        "provider_selection_sha256": selection_sha256,
                        "provider_selection_s3_version_id": selection_version,
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
                "provider_selection_s3_version_id": selection_version,
                "hardware_amendment_sha256": AWS_HARDWARE_AMENDMENT_SHA256,
                "selected_provider": "aws-p5.48xlarge",
                "profile_id": "aws-p5.48xlarge-v3",
                "profile_sha256": P5_PROFILE_SHA256,
                "runtime_lock_sha256": "b" * 64,
                "qualification_evidence_sha256": "c" * 64,
            },
            "snapshots": snapshots,
        }
    )


def test_planner_emits_exact_canonical_100_snapshot_run_bindings():
    lock = _lock()
    lock_sha256 = canonical_sha256(lock.to_dict())

    plans = plan_snapshot_evaluations(
        study_lock=lock,
        study_lock_sha256=lock_sha256,
    )

    assert len(plans) == 100
    assert all(isinstance(plan, SnapshotEvaluationPlan) for plan in plans)
    assert tuple(
        (plan.seed, plan.arm, plan.optimizer_step) for plan in plans
    ) == EXPECTED_STUDY_SLOTS_V3
    assert tuple(plan.slot_index for plan in plans) == tuple(range(100))
    assert len({plan.run_id for plan in plans}) == 100
    assert len({plan.output_name for plan in plans}) == 100
    assert len({plan.run_json_bytes for plan in plans}) == 100

    first = plans[0]
    assert first.run_id == (
        "memorysplit-v3-aws-p5.48xlarge-v3-s00-dense-step01358"
    )
    assert first.output_name == (
        "snapshot-evaluation-memorysplit-v3-aws-p5.48xlarge-v3-"
        "s00-dense-step01358"
    )
    assert first.binding.record_type == RUN_BINDING_SCHEMA_V3
    assert first.binding.schema_version == STUDY_CONTRACT_VERSION
    assert first.binding.arm.value == "dense"
    assert first.binding.checkpoint_path == "checkpoint.pt"
    assert first.binding.study_lock_path == "study-lock.json"
    assert first.binding.study_lock_sha256 == lock_sha256
    assert (
        first.binding.sealed_evaluation_release_sha256
        == lock.sealed_evaluation_release_sha256
    )
    assert first.binding.provider_selection_s3_key == PROVIDER_SELECTION_S3_KEY
    assert first.binding.selected_provider == "aws-p5.48xlarge"
    assert first.binding.evaluator_profile_id == "aws-p5.48xlarge-v3"
    assert first.binding.evaluator_profile_sha256 == P5_PROFILE_SHA256
    assert first.binding.evaluator_runtime_lock_sha256 == "b" * 64
    assert first.binding.evaluator_qualification_evidence_sha256 == "c" * 64
    assert first.run_json_bytes == canonical_json_bytes(first.binding.to_dict())
    assert RunBindingV3.from_dict(
        json.loads(first.run_json_bytes)
    ) == first.binding


def test_confirmatory_package_exports_snapshot_planning_contract():
    assert confirmatory.RunBindingV3 is RunBindingV3
    assert confirmatory.SnapshotEvaluationPlan is SnapshotEvaluationPlan
    assert confirmatory.ProviderSelectionBinding is not None
    assert confirmatory.RUN_BINDING_SCHEMA_V3 == RUN_BINDING_SCHEMA_V3
    assert confirmatory.plan_snapshot_evaluations is plan_snapshot_evaluations
    assert (
        confirmatory.validate_snapshot_evaluation_plans
        is validate_snapshot_evaluation_plans
    )


def test_planner_is_byte_stable_across_repeated_calls():
    lock = _lock()
    lock_sha256 = canonical_sha256(lock.to_dict())

    first = plan_snapshot_evaluations(
        study_lock=lock,
        study_lock_sha256=lock_sha256,
    )
    second = plan_snapshot_evaluations(
        study_lock=StudyLockV3.from_dict(lock.to_dict()),
        study_lock_sha256=lock_sha256,
    )

    assert tuple(plan.run_json_bytes for plan in first) == tuple(
        plan.run_json_bytes for plan in second
    )
    assert tuple(plan.output_name for plan in first) == tuple(
        plan.output_name for plan in second
    )


def test_output_identity_changes_with_the_selected_profile():
    p5 = _lock()
    raw = p5.to_dict()
    raw["provider_selection"].update(
        {
            "provider_selection_sha256": "9" * 64,
            "provider_selection_s3_version_id": "provider-selection-version-p6",
            "selected_provider": "aws-p6-b300.48xlarge",
            "profile_id": "aws-p6-b300.48xlarge-v3",
            "profile_sha256": (
                "6884cd30670214bcecaa105d32b2b5518b1533d9fdbad6ac15327f9a8b7fefa4"
            ),
        }
    )
    for snapshot in raw["snapshots"]:
        snapshot["provider_selection_sha256"] = "9" * 64
        snapshot[
            "provider_selection_s3_version_id"
        ] = "provider-selection-version-p6"
    p6 = StudyLockV3.from_dict(raw)

    p5_plans = plan_snapshot_evaluations(
        study_lock=p5,
        study_lock_sha256=canonical_sha256(p5.to_dict()),
    )
    p6_plans = plan_snapshot_evaluations(
        study_lock=p6,
        study_lock_sha256=canonical_sha256(p6.to_dict()),
    )

    assert p5_plans[0].output_name != p6_plans[0].output_name
    assert "aws-p5.48xlarge-v3" in p5_plans[0].output_name
    assert "aws-p6-b300.48xlarge-v3" in p6_plans[0].output_name


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered", "aliased"])
def test_plan_validator_rejects_non_exact_slot_registries(mutation):
    lock = _lock()
    lock_sha256 = canonical_sha256(lock.to_dict())
    plans = list(
        plan_snapshot_evaluations(
            study_lock=lock,
            study_lock_sha256=lock_sha256,
        )
    )
    if mutation == "missing":
        plans.pop()
    elif mutation == "extra":
        plans.append(plans[-1])
    elif mutation == "reordered":
        plans[0], plans[1] = plans[1], plans[0]
    else:
        plans[1] = plans[0]

    with pytest.raises(ValueError, match="exact|ordered|100|alias|slot"):
        validate_snapshot_evaluation_plans(
            plans,
            study_lock=lock,
            study_lock_sha256=lock_sha256,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("optimizer_step", 1_357, "optimizer_step"),
        ("arm", "split", "arm"),
        ("checkpoint_s3_object_key", "wrong", "checkpoint.*key"),
        ("checkpoint_receipt_s3_object_key", "wrong", "receipt.*key"),
        ("selected_provider", "aws-p6-b300.48xlarge", "provider|profile"),
        ("evaluator_profile_sha256", "0" * 64, "profile"),
        ("output_id", "aliased-output", "output"),
    ],
)
def test_v3_run_binding_rejects_crossed_slot_and_output_identity(
    field,
    value,
    message,
):
    lock = _lock()
    plan = plan_snapshot_evaluations(
        study_lock=lock,
        study_lock_sha256=canonical_sha256(lock.to_dict()),
    )[0]
    raw = plan.binding.to_dict()
    raw[field] = value

    with pytest.raises(ValueError, match=message):
        RunBindingV3.from_dict(raw)


def test_planner_rejects_a_claimed_study_lock_hash():
    lock = _lock()

    with pytest.raises(ValueError, match="study.lock.*hash|commitment"):
        plan_snapshot_evaluations(
            study_lock=lock,
            study_lock_sha256="0" * 64,
        )


def test_plan_value_objects_reject_binding_aliases_at_construction():
    lock = _lock()
    plan = plan_snapshot_evaluations(
        study_lock=lock,
        study_lock_sha256=canonical_sha256(lock.to_dict()),
    )[0]

    with pytest.raises(ValueError, match="binding|identity"):
        replace(plan, run_id="other")
