from __future__ import annotations

import json

import pytest

from evals.confirmatory.contracts import (
    CONTRACT_VERSION,
    canonical_json_bytes,
    canonical_sha256,
)
from evals.confirmatory.fixtures import (
    fixture_by_name,
    invalid_fixture,
    null_fixture,
    positive_fixture,
)
from evals.confirmatory.inference import (
    exact_sign_flip_test,
    hierarchical_paired_bootstrap,
)
from evals.confirmatory.reporting import (
    ARTIFACT_REPORT_SCHEMA,
    REQUIRED_ARTIFACTS,
    ArtifactReport,
    build_artifact_report,
    publish_artifact_report,
    validate_artifact_report,
)
from evals.confirmatory import reporting as reporting_module
from evals.confirmatory.status import classify_status


def _inference(**changes) -> bytes:
    value = {
        "record_type": "memorysplit.confirmatory.inference-evidence.v2",
        "schema_version": CONTRACT_VERSION,
        "terminal_evidence_complete": True,
        "measured_validity_failure": False,
        "observed_valid_seed_pairs": 5,
        "required_seed_pairs": 5,
        "same_sign_preterminal_pairs": 0,
        "supports_effect": True,
        "supports_practical_null": False,
    }
    value.update(changes)
    return canonical_json_bytes(value)


def _artifacts(**inference_changes) -> dict[str, bytes]:
    artifacts = {
        name: canonical_json_bytes(
            {
                "artifact": name,
                "payload": [1, 2, 3],
            }
        )
        for name in REQUIRED_ARTIFACTS
    }
    artifacts["inference.json"] = _inference(**inference_changes)
    return artifacts


def test_artifact_report_round_trips_and_authenticates_every_required_file():
    artifacts = _artifacts()
    report = build_artifact_report(
        artifacts=artifacts,
        expected_cells=35,
        observed_cells=35,
    )

    assert report.record_type == ARTIFACT_REPORT_SCHEMA
    assert report.schema_version == CONTRACT_VERSION
    assert report.scientific_status == "complete"
    assert report.interim_evidence_label == "none"
    assert report.final_inference_conclusion == "supports_effect"
    assert set(report.artifacts) == set(REQUIRED_ARTIFACTS)
    assert ArtifactReport.from_dict(report.to_dict()) == report
    assert validate_artifact_report(report, artifacts) == report
    assert json.loads(canonical_json_bytes(report)) == report.to_dict()


@pytest.mark.parametrize("mutation", ["missing", "extra", "tampered"])
def test_artifact_report_fails_closed_on_missing_extra_or_tampered_files(
    mutation,
):
    artifacts = _artifacts()
    report = build_artifact_report(
        artifacts=artifacts,
        expected_cells=35,
        observed_cells=35,
    )
    changed = dict(artifacts)
    if mutation == "missing":
        changed.pop(next(iter(changed)))
    elif mutation == "extra":
        changed["exploratory.json"] = b"{}\n"
    else:
        name = next(iter(changed))
        changed[name] += b"tamper"

    with pytest.raises(ValueError, match="artifact"):
        validate_artifact_report(report, changed)


def test_artifact_report_rejects_dishonest_status_counts_and_schema_drift():
    artifacts = _artifacts()
    with pytest.raises(ValueError, match="complete|conclusion|evidence"):
        build_artifact_report(
            artifacts=artifacts,
            expected_cells=35,
            observed_cells=34,
        )
    incomplete_artifacts = _artifacts(
        terminal_evidence_complete=False,
        observed_valid_seed_pairs=1,
        same_sign_preterminal_pairs=1,
        supports_effect=False,
    )
    incomplete = build_artifact_report(
        artifacts=incomplete_artifacts,
        expected_cells=35,
        observed_cells=34,
    )
    assert incomplete.scientific_status == "incomplete"
    assert incomplete.interim_evidence_label == "directional_only"
    assert incomplete.final_inference_conclusion == "not_evaluated"

    drifted = incomplete.to_dict()
    drifted["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        ArtifactReport.from_dict(drifted)

    dishonest = incomplete.to_dict()
    dishonest["observed_cells"] = 35
    with pytest.raises(ValueError, match="report_sha256|status"):
        ArtifactReport.from_dict(dishonest)


@pytest.mark.parametrize("schema_version", [True, 2.0])
def test_report_schema_version_is_an_exact_integer(schema_version):
    artifacts = _artifacts()
    report = build_artifact_report(
        artifacts=artifacts,
        expected_cells=35,
        observed_cells=35,
    )
    raw_report = report.to_dict()
    raw_report["schema_version"] = schema_version
    with pytest.raises(ValueError, match="schema"):
        ArtifactReport.from_dict(raw_report)


@pytest.mark.parametrize("schema_version", [True, 2.0])
def test_inference_evidence_schema_version_is_an_exact_integer(
    schema_version,
):
    artifacts = _artifacts()
    changed_artifacts = dict(artifacts)
    raw_evidence = json.loads(changed_artifacts["inference.json"])
    raw_evidence["schema_version"] = schema_version
    changed_artifacts["inference.json"] = canonical_json_bytes(raw_evidence)
    with pytest.raises(ValueError, match="schema"):
        build_artifact_report(
            artifacts=changed_artifacts,
            expected_cells=35,
            observed_cells=35,
        )


def test_measured_invalidity_overrides_missing_cells_and_rejects_strong_claims():
    invalid_artifacts = _artifacts(
        terminal_evidence_complete=False,
        measured_validity_failure=True,
        observed_valid_seed_pairs=2,
        same_sign_preterminal_pairs=2,
        supports_effect=False,
    )
    report = build_artifact_report(
        artifacts=invalid_artifacts,
        expected_cells=35,
        observed_cells=17,
    )

    assert report.scientific_status == "invalid"
    assert report.interim_evidence_label == "directional_only"
    assert report.final_inference_conclusion == "not_evaluated"

    claiming_effect = _artifacts(
        terminal_evidence_complete=False,
        measured_validity_failure=True,
        supports_effect=True,
    )
    with pytest.raises(ValueError, match="supports_effect|conclusion"):
        build_artifact_report(
            artifacts=claiming_effect,
            expected_cells=35,
            observed_cells=17,
        )


def test_report_axes_are_derived_from_strict_hash_bound_inference_evidence():
    artifacts = _artifacts()
    report = build_artifact_report(
        artifacts=artifacts,
        expected_cells=35,
        observed_cells=35,
    )
    dishonest = report.to_dict()
    dishonest.update(
        scientific_status="invalid",
        final_inference_conclusion="not_evaluated",
    )
    payload = {
        key: value
        for key, value in dishonest.items()
        if key != "report_sha256"
    }
    dishonest["report_sha256"] = canonical_sha256(payload)

    with pytest.raises(ValueError, match="evidence|status"):
        validate_artifact_report(dishonest, artifacts)

    drifted = dict(artifacts)
    raw = json.loads(drifted["inference.json"])
    raw["unregistered_claim"] = True
    drifted["inference.json"] = canonical_json_bytes(raw)
    with pytest.raises(ValueError, match="fields"):
        build_artifact_report(
            artifacts=drifted,
            expected_cells=35,
            observed_cells=35,
        )

    assert (
        reporting_module.INFERENCE_EVIDENCE_SCHEMA
        == "memorysplit.confirmatory.inference-evidence.v2"
    )


def test_artifact_report_publication_is_canonical_atomic_and_nonoverwriting(
    tmp_path,
):
    artifacts = _artifacts(
        supports_effect=False,
        supports_practical_null=True,
    )
    report = build_artifact_report(
        artifacts=artifacts,
        expected_cells=35,
        observed_cells=35,
    )
    output = tmp_path / "confirmatory-report.json"

    assert publish_artifact_report(output, report, artifacts) == output
    assert output.read_bytes() == canonical_json_bytes(report)
    with pytest.raises(FileExistsError):
        publish_artifact_report(output, report, artifacts)


def test_publication_pins_parent_across_directory_swap(
    tmp_path,
    monkeypatch,
):
    artifacts = _artifacts()
    report = build_artifact_report(
        artifacts=artifacts,
        expected_cells=35,
        observed_cells=35,
    )
    parent = tmp_path / "validated-parent"
    moved_parent = tmp_path / "pinned-parent"
    attacker = tmp_path / "attacker"
    parent.mkdir()
    attacker.mkdir()
    output = parent / "confirmatory-report.json"
    real_link = reporting_module.os.link
    swapped = False

    def swap_parent_then_link(source, destination, *args, **kwargs):
        nonlocal swapped
        if not swapped:
            reporting_module.os.rename(parent, moved_parent)
            reporting_module.os.symlink(attacker, parent)
            swapped = True
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(
        reporting_module.os,
        "link",
        swap_parent_then_link,
    )

    assert publish_artifact_report(output, report, artifacts) == output
    assert (
        moved_parent.joinpath(output.name).read_bytes()
        == canonical_json_bytes(report)
    )
    assert not attacker.joinpath(output.name).exists()


def test_publication_rejects_symlink_components_and_unsupported_platforms(
    tmp_path,
    monkeypatch,
):
    artifacts = _artifacts()
    report = build_artifact_report(
        artifacts=artifacts,
        expected_cells=35,
        observed_cells=35,
    )
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises((OSError, ValueError), match="symlink|directory"):
        publish_artifact_report(
            alias / "report.json",
            report,
            artifacts,
        )
    assert not real_parent.joinpath("report.json").exists()

    monkeypatch.delattr(reporting_module.os, "O_NOFOLLOW")
    with pytest.raises(RuntimeError, match="unsupported"):
        publish_artifact_report(
            real_parent / "report.json",
            report,
            artifacts,
        )


def test_positive_null_and_invalid_fixtures_are_deterministic_and_decisive():
    positive = positive_fixture()
    practical_null = null_fixture()
    invalid = invalid_fixture()

    assert positive_fixture() == positive
    assert null_fixture() == practical_null
    assert invalid_fixture() == invalid
    assert fixture_by_name("positive") == positive
    assert fixture_by_name("null") == practical_null
    assert fixture_by_name("invalid") == invalid

    for fixture in (positive, practical_null, invalid):
        assert len({row.seed for row in fixture.observations}) == 5
        assert (
            classify_status(
                complete=fixture.complete,
                valid=fixture.valid,
                observed_seeds=5,
                required_seeds=5,
                sign_consistent=fixture.sign_consistent,
                supports_effect=fixture.supports_effect,
                supports_practical_null=fixture.supports_practical_null,
            )
            == fixture.expected_status
        )

    estimate = hierarchical_paired_bootstrap(
        positive.observations,
        n_resamples=50,
        rng_seed=7,
    )
    assert estimate.estimate > 0.0
    assert (
        exact_sign_flip_test(
            estimate.seed_effects,
            alternative="greater",
        ).p_value
        == pytest.approx(1 / 32)
    )
    assert all(row.difference == 0.0 for row in practical_null.observations)
