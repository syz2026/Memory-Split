from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

import pytest

import evals.confirmatory as confirmatory
import evals.confirmatory.aggregate as aggregate_module
import evals.confirmatory.runner as runner_module
from evals.confirmatory.aggregate import (
    COHORT_REPORT_FILE_NAME,
    COHORT_REPORT_ID_PREFIX,
    COHORT_REPORT_SCHEMA_V3,
    CohortReport,
    CollectionReceiptEvidence,
    PublishedCohortReport,
    SnapshotOutputReference,
    aggregate_cohort_outputs,
    publish_cohort_report,
    validate_cohort_report,
)
from evals.confirmatory.actions import ActionSlot
from evals.confirmatory.contracts import (
    STUDY_CONTRACT_VERSION,
    Control,
    MemoryMode,
    StudyArm,
    canonical_json_bytes,
    canonical_sha256,
    store_content_sha256,
)
from evals.confirmatory.fixtures import positive_fixture
from evals.confirmatory.runner import Submission
from evals.confirmatory.sealing import (
    PATH_AUTHORITY,
    SealingError,
    _validated_release_from_preregistration_sha256,
)
from evals.confirmatory.study_lock import (
    FROZEN_PREREGISTRATION_SHA256_V3,
    STUDY_LOCK_SCHEMA_V3,
    StudyLockV3,
)
from msctl.aws_contracts import (
    ARMS,
    SEEDS,
    SNAPSHOT_STEPS,
    checkpoint_receipt_key,
    snapshot_object_key,
)
from tests.study_lock_fixtures import (
    build_seed_lifecycles,
    default_provider_selection,
)


PRIMARY_CONTRAST_ID = (
    "primary_omnibus_pair_and_proof__graph_non_path__"
    "composition_joint_ood__split90_minus_dense"
)
GRAPH_CONTRAST_ID = (
    "secondary_graph_pair_and_proof__composition_joint_ood__"
    "split90_minus_dense"
)
NON_PATH_CONTRAST_ID = (
    "secondary_non_path_pair_and_proof__composition_joint_ood__"
    "split90_minus_dense"
)
TERMINAL_STEP = SNAPSHOT_STEPS[-1]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _jsonl(records: list[dict]) -> bytes:
    return b"".join(canonical_json_bytes(record) for record in records)


def _source_bytes() -> tuple[bytes, bytes, bytes]:
    fixture = positive_fixture()
    items = [
        json.loads(line)
        for line in fixture.artifacts["items.jsonl"].splitlines()
    ]
    stores = [
        json.loads(line)
        for line in fixture.artifacts["stores.jsonl"].splitlines()
    ]
    gold = [
        json.loads(line)
        for line in fixture.artifacts["sealed-gold.jsonl"].splitlines()
    ]
    store_hashes = {}
    for store in stores:
        for row in store["rows"]:
            if row["relation_id"] == "P1":
                row["relation_id"] = "r0"
        store["content_sha256"] = store_content_sha256(
            store["store_id"],
            store["world_id"],
            store["rows"],
        )
        store_hashes[store["store_id"]] = store["content_sha256"]
    store_by_item = {item["item_id"]: item["store_id"] for item in items}
    for record in gold:
        for action in record["proof"]:
            if action["relation_id"] == "P1":
                action["relation_id"] = "r0"
        record["store_sha256"] = store_hashes[store_by_item[record["item_id"]]]
    return _jsonl(items), _jsonl(stores), _jsonl(gold)


def _snapshots() -> list[dict]:
    selection_sha256 = "a" * 64
    selection_version = "provider-selection-version-p5"
    snapshots = []
    for seed in SEEDS:
        receipt_sha256 = _digest(f"receipt:{seed}")
        for arm in ARMS:
            for step in SNAPSHOT_STEPS:
                checkpoint_sha256 = _digest(f"checkpoint:{seed}:{arm}:{step}")
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
                        "s3_version_id": (
                            f"checkpoint-version-{seed}-{arm}-{step}"
                        ),
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
                        "ordered_stream_sha256": _digest("ordered-stream"),
                        "world_size": 4,
                        "tokens_per_step": 524_288,
                    }
                )
    return snapshots


def _study_lock_and_receipts(
    release_sha256: str,
) -> tuple[StudyLockV3, str, tuple[CollectionReceiptEvidence, ...]]:
    selection = default_provider_selection()
    snapshots = _snapshots()
    lifecycles, raw_receipts = build_seed_lifecycles(
        snapshots=snapshots,
        provider_selection=selection,
    )
    lock = StudyLockV3.from_dict(
        {
            "record_type": STUDY_LOCK_SCHEMA_V3,
            "schema_version": STUDY_CONTRACT_VERSION,
            "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256_V3,
            "sealed_evaluation_release_sha256": release_sha256,
            "provider_selection": selection,
            "snapshots": snapshots,
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
    return lock, canonical_sha256(lock.to_dict()), receipts


def _execution_identity(binding) -> dict:
    return {
        "production_qualified": True,
        "adapter_kind": "repository",
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
        "account_id": "123456789012",
        "instance_id": "i-0123456789abcdef0",
        "boot_id": "12345678-1234-4abc-8def-1234567890ab",
        "region": "us-east-1",
        "device_type": "cuda",
        "cuda_available": True,
        "torch_version": "2.6.0",
        "cuda_version": "12.8",
        "device_count": 8,
        "device_name": "NVIDIA H100 80GB",
        "device_capability": [9, 0],
    }


def _submissions(
    *,
    items: list[dict],
    gold: list[dict],
    binding,
    mode: str,
) -> dict[str, Submission]:
    item_by_id = {item["item_id"]: item for item in items}
    result = {}
    for record in gold:
        item = item_by_id[record["item_id"]]
        is_primary_control = (
            item["memory_mode"] == MemoryMode.MEMORY_ON.value
            and item["control"] == Control.CORRECT.value
        )
        correct = True
        if is_primary_control:
            if mode == "effect":
                correct = (
                    binding.arm is StudyArm.SPLIT90
                    and binding.optimizer_step == TERMINAL_STEP
                    and item["family"] == "graph"
                )
            elif mode == "null":
                correct = False
            else:
                raise AssertionError(mode)
        result[record["item_id"]] = Submission(
            item_id=record["item_id"],
            answer=record["answer"] if correct else "wrong-answer",
            actions=tuple(
                ActionSlot.from_dict(action) for action in record["proof"]
            ),
        )
    return result


@dataclass(frozen=True)
class _CohortFixture:
    root: Path
    lock: StudyLockV3
    lock_sha256: str
    references: tuple[SnapshotOutputReference, ...]
    receipts: tuple[CollectionReceiptEvidence, ...]


def _build_cohort(root: Path, mode: str) -> _CohortFixture:
    root.chmod(0o700)
    items_bytes, stores_bytes, gold_bytes = _source_bytes()
    validated_release = _validated_release_from_preregistration_sha256(
        preregistration_sha256=FROZEN_PREREGISTRATION_SHA256_V3,
        items_content=items_bytes,
        stores_content=stores_bytes,
        gold_content=gold_bytes,
    )
    release_sha256 = hashlib.sha256(
        validated_release.manifest_bytes
    ).hexdigest()
    lock, lock_sha256, receipts = _study_lock_and_receipts(release_sha256)
    lock_bytes = canonical_json_bytes(lock.to_dict())
    assert hashlib.sha256(lock_bytes).hexdigest() == lock_sha256
    plans = aggregate_module.plan_snapshot_evaluations(
        study_lock=lock,
        study_lock_sha256=lock_sha256,
        collection_receipts=receipts,
    )
    item_records = [json.loads(line) for line in items_bytes.splitlines()]
    gold_records = [json.loads(line) for line in gold_bytes.splitlines()]
    items = {
        record.item_id: record
        for record in runner_module._canonical_jsonl(
            items_bytes,
            name="items.jsonl",
            parser=runner_module.ItemRecord.from_dict,
            identity=lambda record: record.item_id,
        )
    }
    gold = {
        record.item_id: record
        for record in runner_module._canonical_jsonl(
            gold_bytes,
            name="sealed-gold.jsonl",
            parser=runner_module.SealedGoldRecord.from_dict,
            identity=lambda record: record.item_id,
        )
    }
    stores = {
        record.store_id: record
        for record in runner_module._canonical_jsonl(
            stores_bytes,
            name="stores.jsonl",
            parser=runner_module.StoreRecord.from_dict,
            identity=lambda record: record.store_id,
        )
    }
    references = []
    for plan in plans:
        submissions = _submissions(
            items=item_records,
            gold=gold_records,
            binding=plan.binding,
            mode=mode,
        )
        outcomes_bytes, metrics_bytes = runner_module._score_and_summarize_v3(
            items=items,
            gold=gold,
            stores=stores,
            binding=plan.binding,
            submissions=submissions,
        )
        artifacts = {
            "inference.json": runner_module._inference_bytes_v3(plan.binding),
            "items.jsonl": items_bytes,
            "metrics.json": metrics_bytes,
            "outcomes.jsonl": outcomes_bytes,
            "run.json": plan.run_json_bytes,
            "sealed-gold.jsonl": gold_bytes,
            "sealed-release.json": validated_release.manifest_bytes,
            "stores.jsonl": stores_bytes,
            "study-lock.json": lock_bytes,
        }
        identity = _execution_identity(plan.binding)
        manifest = runner_module._snapshot_output_manifest_bytes(
            binding=plan.binding,
            artifacts=artifacts,
            execution_identity_before=identity,
            execution_identity_after=identity,
            production_qualified=True,
        )
        output = root / f"informational-path-{plan.slot_index:03d}"
        output.mkdir(mode=0o700)
        output.chmod(0o700)
        for name, content in {**artifacts, "output.json": manifest}.items():
            path = output / name
            path.write_bytes(content)
            path.chmod(0o600)
        references.append(
            SnapshotOutputReference(
                output_dir=output,
                output_commitment=hashlib.sha256(manifest).hexdigest(),
            )
        )
    return _CohortFixture(
        root,
        lock,
        lock_sha256,
        tuple(references),
        receipts,
    )


@pytest.fixture(scope="module")
def effect_cohort(tmp_path_factory) -> _CohortFixture:
    return _build_cohort(tmp_path_factory.mktemp("cohort-effect"), "effect")


@pytest.fixture(scope="module")
def null_cohort(tmp_path_factory) -> _CohortFixture:
    return _build_cohort(tmp_path_factory.mktemp("cohort-null"), "null")


@pytest.fixture(scope="module")
def effect_report(effect_cohort) -> CohortReport:
    return aggregate_cohort_outputs(
        outputs=effect_cohort.references,
        collection_receipts=effect_cohort.receipts,
        expected_study_lock_sha256=effect_cohort.lock_sha256,
    )


def _exact(raw: dict) -> Fraction:
    return Fraction(raw["numerator"], raw["denominator"])


def test_confirmatory_package_exports_full_v3_cohort_aggregation_api():
    assert confirmatory.CohortReport is CohortReport
    assert confirmatory.PublishedCohortReport is PublishedCohortReport
    assert confirmatory.SnapshotOutputReference is SnapshotOutputReference
    assert confirmatory.COHORT_REPORT_SCHEMA_V3 == COHORT_REPORT_SCHEMA_V3
    assert confirmatory.COHORT_REPORT_FILE_NAME == COHORT_REPORT_FILE_NAME
    assert confirmatory.COHORT_REPORT_ID_PREFIX == COHORT_REPORT_ID_PREFIX
    assert confirmatory.aggregate_cohort_outputs is aggregate_cohort_outputs
    assert confirmatory.validate_cohort_report is validate_cohort_report
    assert confirmatory.publish_cohort_report is publish_cohort_report


def test_schema_constants_remain_equal_to_runner_values():
    assert aggregate_module._runner_schema_constants() == (
        runner_module.SNAPSHOT_OUTPUT_SCHEMA_V3,
        runner_module.SNAPSHOT_METRICS_SCHEMA_V3,
        runner_module.SNAPSHOT_INFERENCE_SCHEMA_V3,
    )
    assert (
        aggregate_module._OUTPUT_ARTIFACTS
        == runner_module._V3_OUTPUT_ARTIFACTS
    )
    assert (
        aggregate_module.PRIMARY_CONTRAST_ID_V3
        == runner_module.PRIMARY_CONTRAST_ID
    )
    assert (
        aggregate_module.PRIMARY_TEST_METHOD_V3
        == runner_module.PRIMARY_TEST_METHOD
    )
    assert COHORT_REPORT_SCHEMA_V3 == (
        "memorysplit.confirmatory.cohort-report.v3"
    )
    assert COHORT_REPORT_FILE_NAME == "cohort-report.json"
    assert COHORT_REPORT_ID_PREFIX == "cohort-report-"


def _replace_output_manifest(
    reference: SnapshotOutputReference,
    mutate,
) -> tuple[SnapshotOutputReference, bytes]:
    path = reference.output_dir / "output.json"
    original = path.read_bytes()
    raw = json.loads(original)
    mutate(raw)
    changed = canonical_json_bytes(raw)
    path.write_bytes(changed)
    path.chmod(0o600)
    return (
        replace(
            reference,
            output_commitment=hashlib.sha256(changed).hexdigest(),
        ),
        original,
    )


def test_full_effect_cohort_recomputes_exact_primary_bootstrap_aulc_and_holm(
    effect_cohort,
    effect_report,
):
    report = effect_report

    assert report.record_type == COHORT_REPORT_SCHEMA_V3
    assert report.schema_version == STUDY_CONTRACT_VERSION
    assert report.cohort_id == "memorysplit-confirmatory-v3-360m-n10-aws"
    assert report.study_lock_sha256 == effect_cohort.lock_sha256
    assert len(report.inputs) == 100
    assert len({row["output_commitment"] for row in report.inputs}) == 100
    assert all(
        "path" not in row and "output_dir" not in row for row in report.inputs
    )
    assert len(report.snapshots) == 100

    primary = report.primary
    assert primary["primary"] is True
    assert primary["contrast_id"] == PRIMARY_CONTRAST_ID
    assert primary["optimizer_step"] == 13_582
    assert [_exact(row["delta"]) for row in primary["paired_seed_deltas"]] == [
        Fraction(1, 2)
    ] * 10
    assert primary["exact_test"] == {
        "method": "exact_one_sided_exhaustive_sign_flip",
        "alternative": "greater",
        "inclusive_tail": True,
        "retained_zero_deltas": True,
        "statistic": {"numerator": 1, "denominator": 2},
        "extreme_count": 1,
        "assignments": 1_024,
        "p_value": {"numerator": 1, "denominator": 1_024},
        "reject_null": True,
    }
    assert primary["bootstrap"]["bit_generator"] == "PCG64"
    assert primary["bootstrap"]["rng_seed"] == 0
    assert primary["bootstrap"]["draws"] == 20_000
    assert primary["bootstrap"]["confidence_percent"] == 90
    assert primary["bootstrap"]["margin"] == {
        "numerator": 1,
        "denominator": 100,
    }
    assert primary["bootstrap"]["supports_practical_equivalence"] is False

    raw_areas = {
        (row["seed"], row["arm"]): _exact(row["raw_area"])
        for row in report.aulc
    }
    assert len(raw_areas) == 20
    assert {raw_areas[(seed, "dense")] for seed in SEEDS} == {Fraction(0)}
    assert {raw_areas[(seed, "split90")] for seed in SEEDS} == {
        Fraction(3_395, 2)
    }
    for row in report.aulc:
        assert row["interpolation"] == "none"
        assert row["integral"] == "right_step"
        assert row["primary"] is False
        assert len(row["points"]) == 5
        assert (
            tuple(point["optimizer_step"] for point in row["points"])
            == SNAPSHOT_STEPS
        )
        assert _exact(row["normalized_area"]) == (
            _exact(row["raw_area"]) / 13_582
        )
    assert {
        _exact(row["normalized_area"])
        for row in report.aulc
        if row["arm"] == "split90"
    } == {Fraction(3_395, 2 * 13_582)}

    secondary = {row["contrast_id"]: row for row in report.secondary}
    assert tuple(row["contrast_id"] for row in report.secondary) == tuple(
        sorted((GRAPH_CONTRAST_ID, NON_PATH_CONTRAST_ID))
    )
    assert secondary[GRAPH_CONTRAST_ID]["primary"] is False
    assert secondary[NON_PATH_CONTRAST_ID]["primary"] is False
    assert secondary[GRAPH_CONTRAST_ID]["holm"]["family_size"] == 2
    assert _exact(secondary[GRAPH_CONTRAST_ID]["raw_p_value"]) == Fraction(
        1, 1_024
    )
    assert _exact(
        secondary[GRAPH_CONTRAST_ID]["holm"]["adjusted_p_value"]
    ) == Fraction(1, 512)
    assert _exact(secondary[NON_PATH_CONTRAST_ID]["raw_p_value"]) == 1
    assert (
        _exact(secondary[NON_PATH_CONTRAST_ID]["holm"]["adjusted_p_value"])
        == 1
    )

    assert report.instrument_gates["all_passed"] is True
    assert report.status == {
        "scientific_status": "complete",
        "interim_evidence_label": "none",
        "final_inference_conclusion": "supports_effect",
    }


def test_effect_report_round_trips_and_content_addresses_canonically(
    effect_report,
):
    assert CohortReport.from_dict(effect_report.to_dict()) == effect_report
    assert effect_report.authoritative_commitment == hashlib.sha256(
        effect_report.canonical_bytes
    ).hexdigest()
    assert json.loads(effect_report.canonical_bytes) == (
        effect_report.to_dict()
    )


def test_report_collection_rows_equal_lock_seed_lifecycle_receipts(
    effect_cohort,
    effect_report,
):
    assert len(effect_report.collection_receipts) == 10
    assert [dict(row) for row in effect_report.collection_receipts] == [
        {
            "seed": lifecycle.seed,
            "uri": lifecycle.collection_receipt_s3_uri,
            "sha256": lifecycle.collection_receipt_sha256,
            "version_id": lifecycle.collection_receipt_s3_version_id,
        }
        for lifecycle in effect_cohort.lock.seed_lifecycles
    ]
    assert [
        (row["seed"], row["uri"], row["sha256"], row["version_id"])
        for row in effect_report.collection_receipts
    ] == [
        (seed, evidence.uri, evidence.sha256, evidence.version_id)
        for seed, evidence in zip(
            SEEDS,
            effect_cohort.receipts,
            strict=True,
        )
    ]


def test_full_null_cohort_supports_only_the_strict_practical_null(null_cohort):
    report = aggregate_cohort_outputs(
        outputs=null_cohort.references,
        collection_receipts=null_cohort.receipts,
        expected_study_lock_sha256=null_cohort.lock_sha256,
    )

    assert report.primary["effect_criterion_passed"] is False
    assert report.primary["practical_equivalence_criterion_passed"] is True
    assert report.primary["supports_effect"] is False
    assert report.primary["supports_practical_null"] is True
    assert _exact(report.primary["bootstrap"]["ci_low"]) == 0
    assert _exact(report.primary["bootstrap"]["ci_high"]) == 0
    assert report.status["final_inference_conclusion"] == (
        "supports_practical_null"
    )
    assert {_exact(row["raw_area"]) for row in report.aulc} == {Fraction(0)}
    assert {_exact(row["normalized_area"]) for row in report.aulc} == {
        Fraction(0)
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "nine",
        "eleven",
        "duplicate",
        "reordered",
        "wrong_identity",
        "mutated_payload",
    ],
)
def test_aggregation_rejects_nonexact_or_unproven_collection_receipts(
    effect_cohort,
    mutation,
):
    receipts = list(effect_cohort.receipts)
    if mutation == "nine":
        receipts.pop()
    elif mutation == "eleven":
        receipts.append(receipts[-1])
    elif mutation == "duplicate":
        receipts[1] = receipts[0]
    elif mutation == "reordered":
        receipts[0], receipts[1] = receipts[1], receipts[0]
    elif mutation == "wrong_identity":
        receipts[3] = replace(
            receipts[3],
            version_id="collection-version-wrong",
        )
    else:
        receipts[5] = replace(
            receipts[5],
            payload=receipts[5].payload + b"\n",
        )

    with pytest.raises(
        ValueError,
        match="ten|receipt|collection|seed|hash|canonical",
    ):
        aggregate_cohort_outputs(
            outputs=effect_cohort.references,
            collection_receipts=receipts,
            expected_study_lock_sha256=effect_cohort.lock_sha256,
        )


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "duplicate", "reordered", "wrong_commitment"],
)
def test_collection_rejects_nonexact_or_unauthenticated_output_registries(
    effect_cohort,
    mutation,
):
    references = list(effect_cohort.references)
    if mutation == "missing":
        references.pop()
    elif mutation == "extra":
        references.append(references[-1])
    elif mutation == "duplicate":
        references[-1] = references[0]
    elif mutation == "reordered":
        references[0], references[1] = references[1], references[0]
    else:
        references[0] = replace(
            references[0],
            output_commitment="0" * 64,
        )

    with pytest.raises(
        ValueError,
        match="100|ordered|slot|duplicate|commitment|independent",
    ):
        aggregate_cohort_outputs(
            outputs=references,
            collection_receipts=effect_cohort.receipts,
            expected_study_lock_sha256=effect_cohort.lock_sha256,
        )


def test_collection_rejects_cross_profile_and_artifact_tampering(
    effect_cohort,
):
    reference = effect_cohort.references[0]
    crossed, original_manifest = _replace_output_manifest(
        reference,
        lambda raw: (
            raw["execution_identity_before"].update(
                {
                    "profile_id": "aws-p6-b300.48xlarge-v3",
                    "profile_sha256": (
                        "6884cd30670214bcecaa105d32b2b5518"
                        "b1533d9fdbad6ac15327f9a8b7fefa4"
                    ),
                }
            ),
            raw["execution_identity_after"].update(
                {
                    "profile_id": "aws-p6-b300.48xlarge-v3",
                    "profile_sha256": (
                        "6884cd30670214bcecaa105d32b2b5518"
                        "b1533d9fdbad6ac15327f9a8b7fefa4"
                    ),
                }
            ),
        ),
    )
    references = list(effect_cohort.references)
    references[0] = crossed
    try:
        with pytest.raises(ValueError, match="profile|evaluator|selection"):
            aggregate_cohort_outputs(
                outputs=references,
                collection_receipts=effect_cohort.receipts,
                expected_study_lock_sha256=effect_cohort.lock_sha256,
            )
    finally:
        reference.output_dir.joinpath("output.json").write_bytes(
            original_manifest
        )
        reference.output_dir.joinpath("output.json").chmod(0o600)

    metrics_path = reference.output_dir / "metrics.json"
    original_metrics = metrics_path.read_bytes()
    metrics_path.write_bytes(original_metrics + b"tamper")
    metrics_path.chmod(0o600)
    try:
        with pytest.raises(ValueError, match="artifact|hash|commitment"):
            aggregate_cohort_outputs(
                outputs=effect_cohort.references,
                collection_receipts=effect_cohort.receipts,
                expected_study_lock_sha256=effect_cohort.lock_sha256,
            )
    finally:
        metrics_path.write_bytes(original_metrics)
        metrics_path.chmod(0o600)


def test_collection_rejects_stale_45_field_run_bindings(effect_cohort):
    reference = effect_cohort.references[0]
    run_path = reference.output_dir / "run.json"
    manifest_path = reference.output_dir / "output.json"
    original_run = run_path.read_bytes()
    original_manifest = manifest_path.read_bytes()
    stale_binding = json.loads(original_run)
    for field in (
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
    ):
        del stale_binding[field]
    assert len(stale_binding) == 45
    stale_run = canonical_json_bytes(stale_binding)
    manifest = json.loads(original_manifest)
    manifest["run_binding_sha256"] = hashlib.sha256(stale_run).hexdigest()
    for row in manifest["artifacts"]:
        if row["path"] == "run.json":
            row["sha256"] = hashlib.sha256(stale_run).hexdigest()
            row["bytes"] = len(stale_run)
    stale_manifest = canonical_json_bytes(manifest)
    run_path.write_bytes(stale_run)
    run_path.chmod(0o600)
    manifest_path.write_bytes(stale_manifest)
    manifest_path.chmod(0o600)
    references = list(effect_cohort.references)
    references[0] = replace(
        reference,
        output_commitment=hashlib.sha256(stale_manifest).hexdigest(),
    )
    try:
        with pytest.raises(ValueError, match="fields are not exact"):
            aggregate_cohort_outputs(
                outputs=references,
                collection_receipts=effect_cohort.receipts,
                expected_study_lock_sha256=effect_cohort.lock_sha256,
            )
    finally:
        run_path.write_bytes(original_run)
        run_path.chmod(0o600)
        manifest_path.write_bytes(original_manifest)
        manifest_path.chmod(0o600)


def test_collection_rejects_submission_tampering_through_metric_replay(
    effect_cohort,
):
    reference = effect_cohort.references[0]
    outcomes_path = reference.output_dir / "outcomes.jsonl"
    manifest_path = reference.output_dir / "output.json"
    original_outcomes = outcomes_path.read_bytes()
    original_manifest = manifest_path.read_bytes()
    lines = original_outcomes.splitlines(keepends=True)
    forged_index = None
    for index, line in enumerate(lines):
        outcome = json.loads(line)
        if outcome["submitted_answer"] != "wrong-answer":
            forged_index = index
            outcome["submitted_answer"] = "forged-answer"
            lines[index] = canonical_json_bytes(outcome)
            break
    assert forged_index is not None
    tampered_outcomes = b"".join(lines)
    manifest = json.loads(original_manifest)
    for row in manifest["artifacts"]:
        if row["path"] == "outcomes.jsonl":
            row["sha256"] = hashlib.sha256(tampered_outcomes).hexdigest()
            row["bytes"] = len(tampered_outcomes)
    tampered_manifest = canonical_json_bytes(manifest)
    outcomes_path.write_bytes(tampered_outcomes)
    outcomes_path.chmod(0o600)
    manifest_path.write_bytes(tampered_manifest)
    manifest_path.chmod(0o600)
    references = list(effect_cohort.references)
    references[0] = replace(
        reference,
        output_commitment=hashlib.sha256(tampered_manifest).hexdigest(),
    )
    try:
        with pytest.raises(
            ValueError,
            match="metrics|recomputed|outcome",
        ):
            aggregate_cohort_outputs(
                outputs=references,
                collection_receipts=effect_cohort.receipts,
                expected_study_lock_sha256=effect_cohort.lock_sha256,
            )
    finally:
        outcomes_path.write_bytes(original_outcomes)
        outcomes_path.chmod(0o600)
        manifest_path.write_bytes(original_manifest)
        manifest_path.chmod(0o600)


def test_collection_replays_descriptor_identity_and_membership_before_return(
    effect_cohort,
    monkeypatch,
):
    target = effect_cohort.references[0].output_dir / "metrics.json"
    original = target.read_bytes()
    fired = False

    def mutate(event, **_context):
        nonlocal fired
        if event == "before_final_input_replay" and not fired:
            fired = True
            target.write_bytes(original + b"late-tamper")
            target.chmod(0o600)

    monkeypatch.setattr(
        aggregate_module,
        "_run_aggregation_mutation_hook",
        mutate,
    )
    try:
        with pytest.raises(
            ValueError,
            match="changed|identity|replay|artifact",
        ):
            aggregate_cohort_outputs(
                outputs=effect_cohort.references,
                collection_receipts=effect_cohort.receipts,
                expected_study_lock_sha256=effect_cohort.lock_sha256,
            )
        assert fired
    finally:
        target.write_bytes(original)
        target.chmod(0o600)


def test_validator_recomputes_every_claim_from_the_source_output_bytes(
    effect_cohort,
    effect_report,
):
    report = effect_report
    assert (
        validate_cohort_report(
            report.to_dict(),
            outputs=effect_cohort.references,
            collection_receipts=effect_cohort.receipts,
            expected_study_lock_sha256=effect_cohort.lock_sha256,
        )
        == report
    )


def _dishonest_delta(value: dict) -> None:
    value["primary"]["paired_seed_deltas"][0]["delta"] = {
        "numerator": 9,
        "denominator": 10,
    }


def _dishonest_receipt(value: dict) -> None:
    value["collection_receipts"][0]["sha256"] = "0" * 64


def _dishonest_aulc(value: dict) -> None:
    value["aulc"][1]["raw_area"] = {"numerator": 13_582, "denominator": 1}
    value["aulc"][1]["normalized_area"] = {"numerator": 1, "denominator": 1}


def _dishonest_status(value: dict) -> None:
    value["status"]["final_inference_conclusion"] = "inconclusive"


@pytest.mark.parametrize(
    "mutate",
    [_dishonest_delta, _dishonest_receipt, _dishonest_aulc, _dishonest_status],
    ids=["delta", "collection_receipt", "aulc", "status"],
)
def test_validator_rejects_dishonest_persisted_reports(
    effect_cohort,
    effect_report,
    mutate,
):
    dishonest = effect_report.to_dict()
    mutate(dishonest)
    with pytest.raises(
        ValueError,
        match="recompute|source|report|receipt",
    ):
        validate_cohort_report(
            dishonest,
            outputs=effect_cohort.references,
            collection_receipts=effect_cohort.receipts,
            expected_study_lock_sha256=effect_cohort.lock_sha256,
        )


def test_report_publication_is_content_addressed_no_replace_and_mode_0444(
    effect_cohort,
    effect_report,
    tmp_path,
):
    report = effect_report
    root = tmp_path / "reports"
    root.mkdir(mode=0o700)
    root.chmod(0o700)

    published = publish_cohort_report(
        output_root=root,
        report=report,
        outputs=effect_cohort.references,
        collection_receipts=effect_cohort.receipts,
        expected_study_lock_sha256=effect_cohort.lock_sha256,
    )

    assert published.authoritative_commitment == (
        report.authoritative_commitment
    )
    assert published.path_authority == PATH_AUTHORITY
    assert published.report_dir.name == (
        COHORT_REPORT_ID_PREFIX + report.authoritative_commitment
    )
    assert published.report_path.name == COHORT_REPORT_FILE_NAME
    assert published.report_path.read_bytes() == report.canonical_bytes
    member_mode = stat.S_IMODE(published.report_path.stat().st_mode)
    assert member_mode == 0o444

    with pytest.raises(SealingError) as collision:
        publish_cohort_report(
            output_root=root,
            report=report,
            outputs=effect_cohort.references,
            collection_receipts=effect_cohort.receipts,
            expected_study_lock_sha256=effect_cohort.lock_sha256,
        )
    assert collision.value.code == "RELEASE_EXISTS"
    assert published.report_path.read_bytes() == report.canonical_bytes

    blocked_root = tmp_path / "blocked-reports"
    blocked_root.mkdir(mode=0o700)
    blocked_root.chmod(0o700)
    blocker = blocked_root / f".sealed-release-quarantine-{'0' * 32}"
    blocker.mkdir(mode=0o700)
    with pytest.raises(SealingError) as blocked:
        publish_cohort_report(
            output_root=blocked_root,
            report=report,
            outputs=effect_cohort.references,
            collection_receipts=effect_cohort.receipts,
            expected_study_lock_sha256=effect_cohort.lock_sha256,
        )
    assert blocked.value.code == "QUARANTINE_BLOCKER"
    assert sorted(path.name for path in blocked_root.iterdir()) == [
        blocker.name
    ]


def test_report_publication_quarantines_post_install_replacement_intact(
    effect_cohort,
    effect_report,
    tmp_path,
    monkeypatch,
):
    report = effect_report
    root = tmp_path / "failed-reports"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    final_name = COHORT_REPORT_ID_PREFIX + report.authoritative_commitment

    def replace_after_install(event, **_context):
        if event == "publish_before_final_binding":
            rogue = root / final_name / "rogue.json"
            rogue.write_bytes(b"{}")
            rogue.chmod(0o600)

    monkeypatch.setattr(
        aggregate_module,
        "_run_report_publication_mutation_hook",
        replace_after_install,
    )

    def forbid(*_args, **_kwargs):
        raise AssertionError("pathname deletion is forbidden")

    monkeypatch.setattr(aggregate_module.os, "unlink", forbid)
    monkeypatch.setattr(aggregate_module.os, "rmdir", forbid)
    monkeypatch.setattr(aggregate_module.os, "remove", forbid)
    try:
        with pytest.raises(SealingError, match="entries|membership|changed"):
            publish_cohort_report(
                output_root=root,
                report=report,
                outputs=effect_cohort.references,
                collection_receipts=effect_cohort.receipts,
                expected_study_lock_sha256=effect_cohort.lock_sha256,
            )
    finally:
        monkeypatch.undo()

    assert not root.joinpath(final_name).exists()
    quarantines = [
        path
        for path in root.iterdir()
        if path.name.startswith(".sealed-release-quarantine-")
    ]
    assert len(quarantines) == 1
    assert quarantines[0].joinpath(COHORT_REPORT_FILE_NAME).read_bytes() == (
        report.canonical_bytes
    )


def test_commitments_are_authoritative_and_paths_are_informational(
    effect_cohort,
    effect_report,
    tmp_path,
):
    reference = effect_cohort.references[0]
    relocated = tmp_path / "unrelated-informational-name"
    shutil.copytree(reference.output_dir, relocated)
    for path in relocated.iterdir():
        path.chmod(0o600)
    relocated.chmod(0o700)
    references = list(effect_cohort.references)
    references[0] = replace(reference, output_dir=relocated)

    report = aggregate_cohort_outputs(
        outputs=references,
        collection_receipts=effect_cohort.receipts,
        expected_study_lock_sha256=effect_cohort.lock_sha256,
    )

    assert report == effect_report
    assert report.inputs[0]["output_commitment"] == (
        reference.output_commitment
    )
    assert (
        "unrelated-informational-name" not in report.canonical_bytes.decode()
    )
