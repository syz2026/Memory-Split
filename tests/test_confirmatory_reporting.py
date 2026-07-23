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


def _artifacts(name: str = "positive") -> dict[str, bytes]:
    fixtures = {
        "positive": positive_fixture,
        "null": null_fixture,
        "invalid": invalid_fixture,
    }
    return dict(fixtures[name]().artifacts)


def _expected_lock(artifacts) -> str:
    import hashlib

    return hashlib.sha256(artifacts["study-lock.json"]).hexdigest()


def _build(artifacts):
    return build_artifact_report(
        artifacts=artifacts,
        expected_study_lock_sha256=_expected_lock(artifacts),
    )


def _validate(report, artifacts):
    return validate_artifact_report(
        report,
        artifacts,
        expected_study_lock_sha256=_expected_lock(artifacts),
    )


def test_artifact_report_round_trips_and_authenticates_every_required_file():
    artifacts = _artifacts()
    report = _build(artifacts)

    assert report.record_type == ARTIFACT_REPORT_SCHEMA
    assert report.schema_version == CONTRACT_VERSION
    assert report.scientific_status == "complete"
    assert report.interim_evidence_label == "none"
    assert report.final_inference_conclusion == "supports_effect"
    assert set(report.artifacts) == set(REQUIRED_ARTIFACTS)
    assert ArtifactReport.from_dict(report.to_dict()) == report
    assert _validate(report, artifacts) == report
    assert json.loads(canonical_json_bytes(report)) == report.to_dict()


@pytest.mark.parametrize("mutation", ["missing", "extra", "tampered"])
def test_artifact_report_fails_closed_on_missing_extra_or_tampered_files(
    mutation,
):
    artifacts = _artifacts()
    report = _build(artifacts)
    changed = dict(artifacts)
    if mutation == "missing":
        changed.pop(next(iter(changed)))
    elif mutation == "extra":
        changed["exploratory.json"] = b"{}\n"
    else:
        name = next(iter(changed))
        changed[name] += b"tamper"

    with pytest.raises(ValueError, match="artifact"):
        _validate(report, changed)


def test_artifact_report_rejects_dishonest_status_counts_and_schema_drift():
    artifacts = _artifacts()
    report = _build(artifacts)
    drifted = report.to_dict()
    drifted["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        ArtifactReport.from_dict(drifted)

    dishonest = report.to_dict()
    dishonest.update(
        scientific_status="incomplete",
        final_inference_conclusion="not_evaluated",
        observed_cells=79,
    )
    payload = {
        key: value
        for key, value in dishonest.items()
        if key != "report_sha256"
    }
    dishonest["report_sha256"] = canonical_sha256(payload)
    typed = ArtifactReport.from_dict(dishonest)
    with pytest.raises(ValueError, match="counts|status"):
        _validate(typed, artifacts)


@pytest.mark.parametrize("schema_version", [True, 2.0])
def test_report_schema_version_is_an_exact_integer(schema_version):
    artifacts = _artifacts()
    report = _build(artifacts)
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
        _build(changed_artifacts)


def test_measured_invalidity_overrides_missing_cells_and_rejects_strong_claims():
    invalid_artifacts = _artifacts("invalid")
    report = _build(invalid_artifacts)

    assert report.scientific_status == "invalid"
    assert report.interim_evidence_label == "none"
    assert report.final_inference_conclusion == "not_evaluated"

    claiming_effect = dict(invalid_artifacts)
    raw = json.loads(claiming_effect["inference.json"])
    raw["supports_effect"] = True
    claiming_effect["inference.json"] = canonical_json_bytes(raw)
    with pytest.raises(ValueError, match="fields"):
        _build(claiming_effect)


def test_report_axes_are_derived_from_strict_hash_bound_inference_evidence():
    artifacts = _artifacts()
    report = _build(artifacts)
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
        _validate(dishonest, artifacts)

    drifted = dict(artifacts)
    raw = json.loads(drifted["inference.json"])
    raw["unregistered_claim"] = True
    drifted["inference.json"] = canonical_json_bytes(raw)
    with pytest.raises(ValueError, match="fields"):
        _build(drifted)

    assert (
        reporting_module.INFERENCE_EVIDENCE_SCHEMA
        == "memorysplit.confirmatory.inference-evidence.v2"
    )


def test_artifact_report_publication_is_canonical_atomic_and_nonoverwriting(
    tmp_path,
):
    artifacts = _artifacts("null")
    report = _build(artifacts)
    output = tmp_path / "confirmatory-report.json"

    assert (
        publish_artifact_report(
            output,
            report,
            artifacts,
            expected_study_lock_sha256=_expected_lock(artifacts),
        )
        == output
    )
    assert output.read_bytes() == canonical_json_bytes(report)
    with pytest.raises(FileExistsError):
        publish_artifact_report(
            output,
            report,
            artifacts,
            expected_study_lock_sha256=_expected_lock(artifacts),
        )


def test_publication_pins_parent_across_directory_swap(
    tmp_path,
    monkeypatch,
):
    artifacts = _artifacts()
    report = _build(artifacts)
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

    assert (
        publish_artifact_report(
            output,
            report,
            artifacts,
            expected_study_lock_sha256=_expected_lock(artifacts),
        )
        == output
    )
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
    report = _build(artifacts)
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises((OSError, ValueError), match="symlink|directory"):
        publish_artifact_report(
            alias / "report.json",
            report,
            artifacts,
            expected_study_lock_sha256=_expected_lock(artifacts),
        )
    assert not real_parent.joinpath("report.json").exists()

    monkeypatch.delattr(reporting_module.os, "O_NOFOLLOW")
    with pytest.raises(RuntimeError, match="unsupported"):
        publish_artifact_report(
            real_parent / "report.json",
            report,
            artifacts,
            expected_study_lock_sha256=_expected_lock(artifacts),
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
        report = build_artifact_report(
            artifacts=fixture.artifacts,
            expected_study_lock_sha256=fixture.expected_study_lock_sha256,
        )
        assert (
            report.scientific_status,
            report.interim_evidence_label,
            report.final_inference_conclusion,
        ) == (
            fixture.expected_status.scientific_status,
            fixture.expected_status.interim_evidence_label,
            fixture.expected_status.final_inference_conclusion,
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
