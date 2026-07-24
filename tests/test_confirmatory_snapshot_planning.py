from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

import evals.confirmatory as confirmatory
from evals.confirmatory.aggregate import (
    RUN_BINDING_SCHEMA_V3,
    CollectionReceiptEvidence,
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
    SeedLifecycleBinding,
    StudyLockV3,
)
from msctl.aws_contracts import (
    ARMS,
    SEEDS,
    SNAPSHOT_STEPS,
    checkpoint_receipt_key,
    run_receipt_key,
    snapshot_object_key,
)
from msctl.aws_hardware import (
    AWS_HARDWARE_AMENDMENT_SHA256,
    PROVIDER_SELECTION_S3_KEY,
)
from tests.study_lock_fixtures import (
    DENSE_OPERATIONAL_CONFIG_SHA256,
    S3_ROOT,
    SPLIT90_OPERATIONAL_CONFIG_SHA256,
    build_seed_lifecycles,
)


P5_PROFILE_SHA256 = (
    "2207bfbad5e8fa9fc804770b582d0b21f8b6ed109b2e3f3b5c0474c732c53543"
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _provider_selection() -> dict:
    return {
        "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
        "provider_selection_s3_key": PROVIDER_SELECTION_S3_KEY,
        "provider_selection_sha256": "a" * 64,
        "provider_selection_s3_version_id": (
            "provider-selection-version-p5"
        ),
        "hardware_amendment_sha256": AWS_HARDWARE_AMENDMENT_SHA256,
        "selected_provider": "aws-p5.48xlarge",
        "profile_id": "aws-p5.48xlarge-v3",
        "profile_sha256": P5_PROFILE_SHA256,
        "runtime_lock_sha256": "b" * 64,
        "qualification_evidence_sha256": "c" * 64,
        "environment_receipt_sha256": "d" * 64,
        "canary_receipt_sha256": "e" * 64,
        "approval_receipt_sha256": "f" * 64,
        "approval_public_key_sha256": "1" * 64,
    }


def _snapshots() -> list[dict]:
    selection_sha256 = "a" * 64
    selection_version = "provider-selection-version-p5"
    snapshots = []
    for seed in SEEDS:
        for arm in ARMS:
            for step in SNAPSHOT_STEPS:
                checkpoint_sha256 = _digest(f"checkpoint:{seed}:{arm}:{step}")
                receipt_sha256 = _digest(f"receipt:{seed}")
                snapshots.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "optimizer_step": step,
                        "checkpoint_sha256": checkpoint_sha256,
                        "s3_object_key": snapshot_object_key(
                            seed,
                            arm,
                            step,
                            checkpoint_sha256,
                        ),
                        "s3_version_id": f"checkpoint-version-{seed}-{arm}-{step}",
                        "checkpoint_receipt_sha256": receipt_sha256,
                        "checkpoint_receipt_s3_object_key": (
                            checkpoint_receipt_key(seed, receipt_sha256)
                        ),
                        "checkpoint_receipt_s3_version_id": (
                            f"receipt-version-{seed}"
                        ),
                        "provider_selection_sha256": selection_sha256,
                        "provider_selection_s3_version_id": selection_version,
                        "snapshot_version": 2,
                        "training_run_id": (
                            f"memorysplit-v3-360m-s{seed}-{arm}"
                        ),
                        "config_fingerprint": _digest(
                            f"config:{seed}:{arm}"
                        ),
                        "training_config_sha256": _digest(
                            f"config-bytes:{seed}:{arm}"
                        ),
                        "model_config_sha256": _digest("model-config"),
                        "model_identity": "d360m",
                        "data_provenance_sha256": _digest(
                            f"data:{seed}:{arm}"
                        ),
                        "data_receipt_sha256": _digest("data-receipt"),
                        "data_build_id": _digest("data-build"),
                        "ordered_stream_sha256": _digest(
                            "ordered-stream"
                        ),
                        "world_size": 4,
                        "tokens_per_step": 524_288,
                    }
                )
    return snapshots


def _lock_and_receipts(
    *,
    selection: dict | None = None,
    snapshots: list[dict] | None = None,
    **lifecycle_kwargs,
) -> tuple[StudyLockV3, tuple[CollectionReceiptEvidence, ...]]:
    provider_selection = (
        _provider_selection() if selection is None else selection
    )
    slots = _snapshots() if snapshots is None else snapshots
    lifecycles, raw_receipts = build_seed_lifecycles(
        snapshots=slots,
        provider_selection=provider_selection,
        **lifecycle_kwargs,
    )
    lock = StudyLockV3.from_dict(
        {
            "record_type": STUDY_LOCK_SCHEMA_V3,
            "schema_version": STUDY_CONTRACT_VERSION,
            "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256_V3,
            "sealed_evaluation_release_sha256": "f" * 64,
            "provider_selection": provider_selection,
            "snapshots": slots,
            "seed_lifecycles": lifecycles,
        }
    )
    receipts = tuple(
        CollectionReceiptEvidence(
            payload=payload,
            uri=uri,
            sha256=sha256,
            version_id=version_id,
        )
        for payload, uri, sha256, version_id in raw_receipts
    )
    return lock, receipts


def test_planner_emits_exact_canonical_100_snapshot_run_bindings():
    lock, receipts = _lock_and_receipts()
    lock_sha256 = canonical_sha256(lock.to_dict())

    plans = plan_snapshot_evaluations(
        study_lock=lock,
        study_lock_sha256=lock_sha256,
        collection_receipts=receipts,
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
    assert first.binding.snapshot_path == "snapshots/step0001358.pt"
    assert first.binding.snapshot_sha256 == (
        lock.snapshots[0].checkpoint_sha256
    )
    assert first.binding.snapshot_s3_object_key.startswith("snapshots/")
    assert "checkpoints/" not in first.binding.snapshot_s3_object_key
    assert first.binding.snapshot_version == 2
    assert first.binding.training_run_id == (
        "memorysplit-v3-360m-s0-dense"
    )
    assert first.binding.world_size == 4
    assert first.binding.tokens_per_step == 524_288
    assert not hasattr(first.binding, "checkpoint_path")
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
    assert first.binding.evaluator_environment_receipt_sha256 == "d" * 64
    assert first.binding.evaluator_profile_path == "evaluator-profile.json"
    assert first.binding.evaluator_runtime_lock_path == (
        "evaluator-runtime-lock.json"
    )
    assert first.binding.evaluator_environment_receipt_path == (
        "evaluator-environment-receipt.json"
    )
    assert first.run_json_bytes == canonical_json_bytes(first.binding.to_dict())
    assert RunBindingV3.from_dict(
        json.loads(first.run_json_bytes)
    ) == first.binding


def test_planner_bindings_carry_the_full_seed_lifecycle_authority():
    lock, receipts = _lock_and_receipts()
    lock_sha256 = canonical_sha256(lock.to_dict())

    plans = plan_snapshot_evaluations(
        study_lock=lock,
        study_lock_sha256=lock_sha256,
        collection_receipts=receipts,
    )

    for plan in plans:
        lifecycle = lock.seed_lifecycles[plan.seed]
        binding = plan.binding
        assert binding.account_id == lifecycle.account_id
        assert binding.availability_zone == lifecycle.availability_zone
        assert binding.boot_id == lifecycle.boot_id
        assert binding.instance_id == lifecycle.instance_id
        assert binding.region == lifecycle.region
        assert binding.purchase_model == lifecycle.purchase_model
        assert binding.runtime_sbom_sha256 == lifecycle.runtime_sbom_sha256
        assert binding.objective_controls_contract_sha256 == (
            lifecycle.objective_controls_contract_sha256
        )
        assert binding.source_commit == lifecycle.source_commit
        assert binding.source_tree == lifecycle.source_tree
        assert binding.operational_config_sha256 == (
            DENSE_OPERATIONAL_CONFIG_SHA256
            if binding.arm.value == "dense"
            else SPLIT90_OPERATIONAL_CONFIG_SHA256
        )
        assert binding.finalization_receipt_sha256 == (
            lifecycle.finalization_receipt_sha256
        )
        assert binding.finalization_receipt_s3_uri == (
            lifecycle.finalization_receipt_s3_uri
        )
        assert binding.finalization_receipt_bytes == (
            lifecycle.finalization_receipt_bytes
        )
        assert binding.finalization_receipt_s3_version_id == (
            lifecycle.finalization_receipt_s3_version_id
        )
        assert binding.collection_receipt_sha256 == (
            lifecycle.collection_receipt_sha256
        )
        assert binding.collection_receipt_s3_uri == (
            lifecycle.collection_receipt_s3_uri
        )
        assert binding.collection_receipt_s3_version_id == (
            lifecycle.collection_receipt_s3_version_id
        )
    dense_configs = {
        plan.binding.operational_config_sha256
        for plan in plans
        if plan.binding.arm.value == "dense"
    }
    split90_configs = {
        plan.binding.operational_config_sha256
        for plan in plans
        if plan.binding.arm.value == "split90"
    }
    assert dense_configs == {DENSE_OPERATIONAL_CONFIG_SHA256}
    assert split90_configs == {SPLIT90_OPERATIONAL_CONFIG_SHA256}
    assert dense_configs != split90_configs


def test_confirmatory_package_exports_snapshot_planning_contract():
    assert confirmatory.RunBindingV3 is RunBindingV3
    assert confirmatory.SnapshotEvaluationPlan is SnapshotEvaluationPlan
    assert confirmatory.ProviderSelectionBinding is not None
    assert confirmatory.SeedLifecycleBinding is SeedLifecycleBinding
    assert confirmatory.CollectionReceiptEvidence is (
        CollectionReceiptEvidence
    )
    assert confirmatory.RUN_BINDING_SCHEMA_V3 == RUN_BINDING_SCHEMA_V3
    assert confirmatory.plan_snapshot_evaluations is plan_snapshot_evaluations
    assert (
        confirmatory.validate_snapshot_evaluation_plans
        is validate_snapshot_evaluation_plans
    )


def test_planner_is_byte_stable_across_repeated_calls():
    lock, receipts = _lock_and_receipts()
    lock_sha256 = canonical_sha256(lock.to_dict())

    first = plan_snapshot_evaluations(
        study_lock=lock,
        study_lock_sha256=lock_sha256,
        collection_receipts=receipts,
    )
    second = plan_snapshot_evaluations(
        study_lock=StudyLockV3.from_dict(lock.to_dict()),
        study_lock_sha256=lock_sha256,
        collection_receipts=tuple(
            CollectionReceiptEvidence(
                payload=bytes(receipt.payload),
                uri=receipt.uri,
                sha256=receipt.sha256,
                version_id=receipt.version_id,
            )
            for receipt in receipts
        ),
    )

    assert tuple(plan.run_json_bytes for plan in first) == tuple(
        plan.run_json_bytes for plan in second
    )
    assert tuple(plan.output_name for plan in first) == tuple(
        plan.output_name for plan in second
    )


def test_output_identity_changes_with_the_selected_profile():
    p6_selection = _provider_selection()
    p6_selection.update(
        {
            "provider_selection_sha256": "9" * 64,
            "provider_selection_s3_version_id": (
                "provider-selection-version-p6"
            ),
            "selected_provider": "aws-p6-b300.48xlarge",
            "profile_id": "aws-p6-b300.48xlarge-v3",
            "profile_sha256": (
                "6884cd30670214bcecaa105d32b2b5518b1533d9fdbad6ac15327f9a8b7fefa4"
            ),
        }
    )
    p6_snapshots = _snapshots()
    for snapshot in p6_snapshots:
        snapshot["provider_selection_sha256"] = "9" * 64
        snapshot[
            "provider_selection_s3_version_id"
        ] = "provider-selection-version-p6"
    p5_lock, p5_receipts = _lock_and_receipts()
    p6_lock, p6_receipts = _lock_and_receipts(
        selection=p6_selection,
        snapshots=p6_snapshots,
    )

    p5_plans = plan_snapshot_evaluations(
        study_lock=p5_lock,
        study_lock_sha256=canonical_sha256(p5_lock.to_dict()),
        collection_receipts=p5_receipts,
    )
    p6_plans = plan_snapshot_evaluations(
        study_lock=p6_lock,
        study_lock_sha256=canonical_sha256(p6_lock.to_dict()),
        collection_receipts=p6_receipts,
    )

    assert p5_plans[0].output_name != p6_plans[0].output_name
    assert "aws-p5.48xlarge-v3" in p5_plans[0].output_name
    assert "aws-p6-b300.48xlarge-v3" in p6_plans[0].output_name


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered", "aliased"])
def test_plan_validator_rejects_non_exact_slot_registries(mutation):
    lock, receipts = _lock_and_receipts()
    lock_sha256 = canonical_sha256(lock.to_dict())
    plans = list(
        plan_snapshot_evaluations(
            study_lock=lock,
            study_lock_sha256=lock_sha256,
            collection_receipts=receipts,
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
            collection_receipts=receipts,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("optimizer_step", 1_357, "optimizer_step"),
        ("arm", "split", "arm"),
        ("snapshot_s3_object_key", "wrong", "snapshot.*key|object key"),
        ("checkpoint_receipt_s3_object_key", "wrong", "receipt.*key"),
        ("snapshot_version", 1, "snapshot.*version"),
        ("training_run_id", "other-run", "training.*run|identity"),
        ("world_size", 8, "world.size"),
        ("tokens_per_step", 1, "tokens.per.step"),
        ("selected_provider", "aws-p6-b300.48xlarge", "provider|profile"),
        ("evaluator_profile_sha256", "0" * 64, "profile"),
        ("output_id", "aliased-output", "output"),
        ("account_id", "bad-account", "lifecycle|account"),
        ("boot_id", "not-a-boot-id", "lifecycle|boot"),
        ("instance_id", "not-an-instance", "lifecycle|instance"),
        ("purchase_model", "", "lifecycle|purchase"),
        ("runtime_sbom_sha256", "not-a-hash", "lifecycle|SBOM|sbom"),
        (
            "objective_controls_contract_sha256",
            "not-a-hash",
            "lifecycle|objective",
        ),
        ("source_commit", "g" * 40, "commit"),
        ("source_tree", "short", "tree"),
        ("operational_config_sha256", "not-a-hash", "operational.*config"),
        (
            "finalization_receipt_s3_uri",
            "s3://memorysplit-prod/other.json",
            "finalization.*(key|URI)",
        ),
        ("finalization_receipt_bytes", 0, "bytes"),
        ("finalization_receipt_s3_version_id", "null", "version"),
        (
            "collection_receipt_s3_uri",
            "s3://memorysplit-prod/other.json",
            "collection.*(key|URI)",
        ),
        ("collection_receipt_sha256", "not-a-hash", "collection"),
        ("collection_receipt_s3_version_id", "", "version"),
    ],
)
def test_v3_run_binding_rejects_crossed_slot_and_output_identity(
    field,
    value,
    message,
):
    lock, receipts = _lock_and_receipts()
    plan = plan_snapshot_evaluations(
        study_lock=lock,
        study_lock_sha256=canonical_sha256(lock.to_dict()),
        collection_receipts=receipts,
    )[0]
    raw = plan.binding.to_dict()
    raw[field] = value

    with pytest.raises(ValueError, match=message):
        RunBindingV3.from_dict(raw)


def test_pre_bridge_44_field_run_binding_fails_closed():
    lock, receipts = _lock_and_receipts()
    plan = plan_snapshot_evaluations(
        study_lock=lock,
        study_lock_sha256=canonical_sha256(lock.to_dict()),
        collection_receipts=receipts,
    )[0]
    raw = plan.binding.to_dict()
    pre_bridge_fields = set(raw) - {
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
    }
    pre_bridge = {field: raw[field] for field in pre_bridge_fields}

    with pytest.raises(ValueError, match="exact"):
        RunBindingV3.from_dict(pre_bridge)


def test_planner_rejects_a_claimed_study_lock_hash():
    lock, receipts = _lock_and_receipts()

    with pytest.raises(ValueError, match="study.lock.*hash|commitment"):
        plan_snapshot_evaluations(
            study_lock=lock,
            study_lock_sha256="0" * 64,
            collection_receipts=receipts,
        )


def test_plan_value_objects_reject_binding_aliases_at_construction():
    lock, receipts = _lock_and_receipts()
    plan = plan_snapshot_evaluations(
        study_lock=lock,
        study_lock_sha256=canonical_sha256(lock.to_dict()),
        collection_receipts=receipts,
    )[0]

    with pytest.raises(ValueError, match="binding|identity"):
        replace(plan, run_id="other")


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "duplicate", "reordered", "foreign-type"],
)
def test_planning_requires_exactly_ten_exact_collection_receipts(mutation):
    lock, receipts = _lock_and_receipts()
    lock_sha256 = canonical_sha256(lock.to_dict())
    evidence = list(receipts)
    if mutation == "missing":
        evidence.pop()
    elif mutation == "extra":
        evidence.append(evidence[-1])
    elif mutation == "duplicate":
        evidence[1] = evidence[0]
    elif mutation == "reordered":
        evidence[0], evidence[1] = evidence[1], evidence[0]
    else:
        evidence[0] = (
            evidence[0].payload,
            evidence[0].uri,
            evidence[0].sha256,
            evidence[0].version_id,
        )

    with pytest.raises(
        ValueError,
        match="ten|receipt|collection|seed",
    ):
        plan_snapshot_evaluations(
            study_lock=lock,
            study_lock_sha256=lock_sha256,
            collection_receipts=evidence,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sha256", "0" * 64, "collection receipt"),
        (
            "version_id",
            "other-collection-version",
            "collection receipt.*(identity|version)",
        ),
        (
            "uri",
            f"{S3_ROOT}/receipts/collections/seed-0/sha256/{'0' * 64}.json",
            "collection receipt|key|URI",
        ),
        ("payload", b"{}\n", "hash|canonical|fields"),
    ],
)
def test_planning_rejects_crossed_collection_receipt_identity(
    field,
    value,
    message,
):
    lock, receipts = _lock_and_receipts()
    lock_sha256 = canonical_sha256(lock.to_dict())
    evidence = list(receipts)
    evidence[0] = replace(evidence[0], **{field: value})

    with pytest.raises(ValueError, match=message):
        plan_snapshot_evaluations(
            study_lock=lock,
            study_lock_sha256=lock_sha256,
            collection_receipts=evidence,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("snapshot-sha", "durably collected"),
        ("snapshot-version", "durably collected"),
        ("checkpoint-receipt-sha", "durably collected|checkpoint receipt"),
        (
            "checkpoint-receipt-version",
            "durably collected|checkpoint receipt",
        ),
    ],
)
def test_planning_row_drift_proves_snapshots_not_durably_collected(
    mutation,
    message,
):
    def mutate(seed: int, collection: dict) -> None:
        if seed != 3:
            return
        if mutation == "snapshot-sha":
            row = collection["objects"][0]
            forged = _digest("forged-snapshot")
            row["sha256"] = forged
            row["uri"] = (
                f"{S3_ROOT}/"
                + snapshot_object_key(3, "dense", 1_358, forged)
            )
        elif mutation == "snapshot-version":
            collection["objects"][6]["version_id"] = "forged-version"
        elif mutation == "checkpoint-receipt-sha":
            forged = _digest("forged-checkpoint-receipt")
            for reference in (
                collection["checkpoint_receipt"],
                collection["objects"][14],
            ):
                reference["sha256"] = forged
                reference["uri"] = (
                    f"{S3_ROOT}/" + checkpoint_receipt_key(3, forged)
                )
        else:
            for reference in (
                collection["checkpoint_receipt"],
                collection["objects"][14],
            ):
                reference["version_id"] = "forged-receipt-version"

    lock, receipts = _lock_and_receipts(mutate_collection=mutate)
    lock_sha256 = canonical_sha256(lock.to_dict())

    with pytest.raises(ValueError, match=message):
        plan_snapshot_evaluations(
            study_lock=lock,
            study_lock_sha256=lock_sha256,
            collection_receipts=receipts,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("finalization-sha", "finalization"),
        ("finalization-bytes", "finalization"),
        ("finalization-version", "finalization"),
        ("run-manifest", "manifest"),
        ("release", "release"),
        ("release-receipt", "release"),
        ("dataset-receipt", "dataset"),
        ("dataset-build", "dataset"),
        ("ordered-stream", "dataset|ordered"),
        ("source-commit", "source"),
        ("source-tree", "source"),
        ("boot", "lifecycle|boot"),
        ("profile", "lifecycle|profile"),
    ],
)
def test_planning_rejects_receipt_reference_and_identity_drift(
    mutation,
    message,
):
    def mutate(seed: int, collection: dict) -> None:
        if seed != 7:
            return
        if mutation == "finalization-sha":
            forged = _digest("forged-finalization")
            for reference in (
                collection["run_receipt"],
                collection["objects"][15],
            ):
                reference["sha256"] = forged
                reference["uri"] = (
                    f"{S3_ROOT}/" + run_receipt_key(7, forged)
                )
        elif mutation == "finalization-bytes":
            for reference in (
                collection["run_receipt"],
                collection["objects"][15],
            ):
                reference["bytes"] = 999_999
        elif mutation == "finalization-version":
            for reference in (
                collection["run_receipt"],
                collection["objects"][15],
            ):
                reference["version_id"] = "forged-finalization-version"
        elif mutation == "run-manifest":
            collection["run_manifest_sha256"] = _digest("forged-manifest")
        elif mutation == "release":
            collection["release_sha256"] = _digest("forged-release")
        elif mutation == "release-receipt":
            collection["release_receipt_sha256"] = _digest(
                "forged-release-receipt"
            )
        elif mutation == "dataset-receipt":
            collection["dataset_receipt_sha256"] = _digest(
                "forged-dataset-receipt"
            )
        elif mutation == "dataset-build":
            collection["dataset_build_id"] = _digest("forged-dataset-build")
        elif mutation == "ordered-stream":
            collection["ordered_stream_sha256"] = _digest(
                "forged-ordered-stream"
            )
        elif mutation == "source-commit":
            collection["source_commit"] = "e" * 40
        elif mutation == "source-tree":
            collection["source_tree"] = "1" * 40
        elif mutation == "boot":
            collection["boot_id"] = (
                "87654321-4321-4cba-8fed-1234567890ab"
            )
        else:
            collection["profile_id"] = "aws-p6-b300.48xlarge-v3"
            collection["provider"] = "aws-p6-b300.48xlarge"

    lock, receipts = _lock_and_receipts(mutate_collection=mutate)
    lock_sha256 = canonical_sha256(lock.to_dict())

    with pytest.raises(ValueError, match=message):
        plan_snapshot_evaluations(
            study_lock=lock,
            study_lock_sha256=lock_sha256,
            collection_receipts=receipts,
        )
