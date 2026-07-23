from __future__ import annotations

import hashlib
import json

import pytest

from evals.confirmatory.contracts import canonical_json_bytes
from evals.confirmatory.fixtures import (
    DETERMINISTIC_STUDY_LOCK_SHA256,
    invalid_fixture,
    positive_fixture,
)
from evals.confirmatory import fixtures as fixtures_module
from evals.confirmatory.reporting import (
    RELEASE_SEALING_LAUNCH_GAP,
    build_artifact_report,
    publish_artifact_report,
    validate_artifact_report,
)


def _trusted_report(fixture):
    return build_artifact_report(
        artifacts=fixture.artifacts,
        expected_study_lock_sha256=fixture.expected_study_lock_sha256,
    )


def test_report_requires_same_external_study_lock_at_build_and_validation():
    fixture = positive_fixture()
    assert (
        fixture.expected_study_lock_sha256
        == DETERMINISTIC_STUDY_LOCK_SHA256
    )
    assert "production release" in RELEASE_SEALING_LAUNCH_GAP
    assert "expected_study_lock_sha256" in RELEASE_SEALING_LAUNCH_GAP

    with pytest.raises((TypeError, ValueError), match="expected_study_lock"):
        build_artifact_report(artifacts=fixture.artifacts)

    report = _trusted_report(fixture)
    assert report.study_lock_sha256 == fixture.expected_study_lock_sha256
    assert (
        validate_artifact_report(
            report,
            fixture.artifacts,
            expected_study_lock_sha256=fixture.expected_study_lock_sha256,
        )
        == report
    )
    with pytest.raises((TypeError, ValueError), match="expected_study_lock"):
        validate_artifact_report(report, fixture.artifacts)
    with pytest.raises(ValueError, match="study lock|commitment"):
        build_artifact_report(
            artifacts=fixture.artifacts,
            expected_study_lock_sha256="0" * 64,
        )


def test_publication_also_requires_external_study_lock(tmp_path):
    fixture = positive_fixture()
    report = _trusted_report(fixture)

    with pytest.raises(TypeError, match="expected_study_lock"):
        publish_artifact_report(
            tmp_path / "report.json",
            report,
            fixture.artifacts,
        )


def test_shrunken_self_consistent_release_cannot_replace_frozen_commitment():
    frozen = positive_fixture()
    shrunken = fixtures_module.shrunken_fixture()

    assert _trusted_report(shrunken).scientific_status == "complete"
    assert (
        shrunken.expected_study_lock_sha256
        != frozen.expected_study_lock_sha256
    )
    with pytest.raises(ValueError, match="study lock|commitment"):
        build_artifact_report(
            artifacts=shrunken.artifacts,
            expected_study_lock_sha256=frozen.expected_study_lock_sha256,
        )


def test_tiny_memory_on_correct_only_release_is_never_complete():
    tiny = fixtures_module.tiny_fixture()

    with pytest.raises(ValueError, match="memory mode|control|required"):
        _trusted_report(tiny)


def test_study_lock_covers_release_registry_and_every_frozen_dimension():
    fixture = positive_fixture()
    lock = json.loads(fixture.artifacts["study-lock.json"])
    release = lock["release"]

    assert release["item_count"] == len(release["item_ids"])
    assert release["pair_count"] == len(release["pair_ids"])
    assert release["world_count"] == len(release["world_ids"])
    assert release["evaluation_cell_count"] == len(
        release["evaluation_cells"]
    )
    assert release["required_families"] == ["graph", "non_path"]
    assert release["required_strata"] == [
        "iid",
        "composition_ood",
        "length_ood",
        "joint_ood",
    ]
    assert release["required_memory_modes"] == [
        "memory_off",
        "memory_on",
    ]
    assert release["required_controls"] == [
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
    ]


@pytest.mark.parametrize("schema_version", [True, 2.0])
def test_study_lock_and_validity_versions_are_strict_integers(schema_version):
    fixture = positive_fixture()
    changed = dict(fixture.artifacts)
    lock = json.loads(changed["study-lock.json"])
    lock["schema_version"] = schema_version
    changed["study-lock.json"] = canonical_json_bytes(lock)
    with pytest.raises(ValueError, match="schema_version"):
        build_artifact_report(
            artifacts=changed,
            expected_study_lock_sha256=hashlib.sha256(
                changed["study-lock.json"]
            ).hexdigest(),
        )

    changed = dict(fixture.artifacts)
    validity = json.loads(changed["validity.json"])
    validity["schema_version"] = schema_version
    changed["validity.json"] = canonical_json_bytes(validity)
    with pytest.raises(ValueError, match="schema_version"):
        build_artifact_report(
            artifacts=changed,
            expected_study_lock_sha256=fixture.expected_study_lock_sha256,
        )


@pytest.mark.parametrize("artifact_name", ["sealed-gold.jsonl", "stores.jsonl"])
def test_gold_and_store_artifacts_are_strict_and_cross_bound(artifact_name):
    fixture = positive_fixture()
    changed = dict(fixture.artifacts)
    changed[artifact_name] = b"not-json\n"

    with pytest.raises(ValueError, match="gold|store|JSON"):
        build_artifact_report(
            artifacts=changed,
            expected_study_lock_sha256=fixture.expected_study_lock_sha256,
        )


def test_forged_correctness_or_submission_cannot_support_effect():
    fixture = positive_fixture()
    changed = dict(fixture.artifacts)
    rows = changed["outcomes.jsonl"].splitlines()
    first = json.loads(rows[0])
    first["submitted_answer"] = "forged"
    first["proof_valid"] = True
    first["answer_valid"] = True
    rows[0] = canonical_json_bytes(first).rstrip(b"\n")
    changed["outcomes.jsonl"] = b"\n".join(rows) + b"\n"

    with pytest.raises(ValueError, match="fields|metrics|outcome"):
        build_artifact_report(
            artifacts=changed,
            expected_study_lock_sha256=fixture.expected_study_lock_sha256,
        )

    changed = dict(fixture.artifacts)
    rows = changed["outcomes.jsonl"].splitlines()
    split_index = next(
        index
        for index, line in enumerate(rows)
        if json.loads(line)["condition_id"] == "split90"
    )
    split_submission = json.loads(rows[split_index])
    split_submission["submitted_answer"] = "wrong"
    rows[split_index] = canonical_json_bytes(split_submission).rstrip(b"\n")
    changed["outcomes.jsonl"] = b"\n".join(rows) + b"\n"
    with pytest.raises(ValueError, match="metrics|outcome"):
        build_artifact_report(
            artifacts=changed,
            expected_study_lock_sha256=fixture.expected_study_lock_sha256,
        )


def test_validity_receipts_are_committed_complete_and_failure_precedes_missing():
    positive = positive_fixture()
    pending = fixtures_module.pending_fixture()
    failed = invalid_fixture()

    assert _trusted_report(positive).scientific_status == "complete"
    assert _trusted_report(pending).scientific_status == "incomplete"
    failed_report = _trusted_report(failed)
    assert failed_report.scientific_status == "invalid"
    assert failed_report.interim_evidence_label == "none"

    omitted = dict(positive.artifacts)
    validity = json.loads(omitted["validity.json"])
    validity["receipts"] = [
        receipt
        for receipt in validity["receipts"]
        if receipt["receipt_id"] != "control:no_query"
    ]
    omitted["validity.json"] = canonical_json_bytes(validity)
    assert (
        build_artifact_report(
            artifacts=omitted,
            expected_study_lock_sha256=positive.expected_study_lock_sha256,
        ).scientific_status
        == "incomplete"
    )
    missing_gate = dict(positive.artifacts)
    validity = json.loads(missing_gate["validity.json"])
    validity["receipts"] = [
        receipt
        for receipt in validity["receipts"]
        if receipt["receipt_id"] != "gate:evaluation_validity"
    ]
    missing_gate["validity.json"] = canonical_json_bytes(validity)
    assert (
        build_artifact_report(
            artifacts=missing_gate,
            expected_study_lock_sha256=positive.expected_study_lock_sha256,
        ).scientific_status
        == "incomplete"
    )

    failed_and_missing = dict(failed.artifacts)
    validity = json.loads(failed_and_missing["validity.json"])
    validity["receipts"] = [
        receipt
        for receipt in validity["receipts"]
        if receipt["receipt_id"] != "control:no_query"
    ]
    failed_and_missing["validity.json"] = canonical_json_bytes(validity)
    assert (
        build_artifact_report(
            artifacts=failed_and_missing,
            expected_study_lock_sha256=failed.expected_study_lock_sha256,
        ).scientific_status
        == "invalid"
    )
    hidden_failure = dict(failed.artifacts)
    validity = json.loads(hidden_failure["validity.json"])
    validity["receipts"] = [
        receipt
        for receipt in validity["receipts"]
        if receipt["state"] != "failed"
    ]
    hidden_failure["validity.json"] = canonical_json_bytes(validity)
    assert (
        build_artifact_report(
            artifacts=hidden_failure,
            expected_study_lock_sha256=failed.expected_study_lock_sha256,
        ).scientific_status
        == "invalid"
    )

    false_pass = dict(failed.artifacts)
    validity = json.loads(false_pass["validity.json"])
    failed_receipt = next(
        receipt
        for receipt in validity["receipts"]
        if receipt["state"] == "failed"
    )
    failed_receipt["state"] = "passed"
    false_pass["validity.json"] = canonical_json_bytes(validity)
    with pytest.raises(ValueError, match="receipt|commitment"):
        build_artifact_report(
            artifacts=false_pass,
            expected_study_lock_sha256=failed.expected_study_lock_sha256,
        )


def test_only_approved_dense_split90_checkpoints_enter_primary_replay():
    fixture = positive_fixture()
    unapproved_model = dict(fixture.artifacts)
    checkpoint_rows = unapproved_model["checkpoints.jsonl"].splitlines()
    first = json.loads(checkpoint_rows[0])
    first["model_id"] = "unapproved-model"
    checkpoint_rows[0] = canonical_json_bytes(first).rstrip(b"\n")
    unapproved_model["checkpoints.jsonl"] = (
        b"\n".join(checkpoint_rows) + b"\n"
    )
    with pytest.raises(ValueError, match="checkpoint|study lock|commitment"):
        build_artifact_report(
            artifacts=unapproved_model,
            expected_study_lock_sha256=fixture.expected_study_lock_sha256,
        )

    changed = dict(fixture.artifacts)
    checkpoint_rows = changed["checkpoints.jsonl"].splitlines()
    first = json.loads(checkpoint_rows[0])
    first["configuration_sha256"] = "f" * 64
    checkpoint_rows[0] = canonical_json_bytes(first).rstrip(b"\n")
    changed["checkpoints.jsonl"] = b"\n".join(checkpoint_rows) + b"\n"

    with pytest.raises(ValueError, match="checkpoint|configuration|study lock"):
        build_artifact_report(
            artifacts=changed,
            expected_study_lock_sha256=fixture.expected_study_lock_sha256,
        )

    first["arm"] = "split"
    first["condition_id"] = "split"
    checkpoint_rows[0] = canonical_json_bytes(first).rstrip(b"\n")
    changed["checkpoints.jsonl"] = b"\n".join(checkpoint_rows) + b"\n"
    with pytest.raises(ValueError, match="condition"):
        build_artifact_report(
            artifacts=changed,
            expected_study_lock_sha256=fixture.expected_study_lock_sha256,
        )
