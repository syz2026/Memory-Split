from __future__ import annotations

import json

import pytest

from evals.confirmatory.contracts import CONTRACT_VERSION, canonical_json_bytes
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
from evals.confirmatory.status import classify_status


def _artifacts() -> dict[str, bytes]:
    return {
        name: canonical_json_bytes(
            {
                "artifact": name,
                "payload": [1, 2, 3],
            }
        )
        for name in REQUIRED_ARTIFACTS
    }


def test_artifact_report_round_trips_and_authenticates_every_required_file():
    artifacts = _artifacts()
    report = build_artifact_report(
        status="supports_effect",
        artifacts=artifacts,
        expected_cells=35,
        observed_cells=35,
    )

    assert report.record_type == ARTIFACT_REPORT_SCHEMA
    assert report.schema_version == CONTRACT_VERSION
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
        status="supports_effect",
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
    with pytest.raises(ValueError, match="incomplete"):
        build_artifact_report(
            status="supports_effect",
            artifacts=artifacts,
            expected_cells=35,
            observed_cells=34,
        )
    incomplete = build_artifact_report(
        status="incomplete",
        artifacts=artifacts,
        expected_cells=35,
        observed_cells=34,
    )
    assert incomplete.status == "incomplete"

    drifted = incomplete.to_dict()
    drifted["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        ArtifactReport.from_dict(drifted)

    dishonest = incomplete.to_dict()
    dishonest["observed_cells"] = 35
    with pytest.raises(ValueError, match="report_sha256|incomplete"):
        ArtifactReport.from_dict(dishonest)


def test_artifact_report_publication_is_canonical_atomic_and_nonoverwriting(
    tmp_path,
):
    artifacts = _artifacts()
    report = build_artifact_report(
        status="supports_practical_null",
        artifacts=artifacts,
        expected_cells=35,
        observed_cells=35,
    )
    output = tmp_path / "confirmatory-report.json"

    assert publish_artifact_report(output, report, artifacts) == output
    assert output.read_bytes() == canonical_json_bytes(report)
    with pytest.raises(FileExistsError):
        publish_artifact_report(output, report, artifacts)


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
