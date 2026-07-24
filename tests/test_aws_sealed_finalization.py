from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evals.confirmatory.contracts import (
    CHECKPOINT_SCHEMA,
    CONTRACT_VERSION,
    canonical_json_bytes,
)
from evals.confirmatory.fixtures import positive_fixture
from evals.confirmatory.inference import exact_sign_flip_test
from evals.confirmatory.reporting import (
    ARTIFACT_REPORT_SCHEMA_V3,
    INFERENCE_EVIDENCE_SCHEMA_V3,
    METRICS_SCHEMA,
    build_artifact_report,
    validate_artifact_report,
)
from evals.confirmatory.study_lock import (
    STUDY_LOCK_SCHEMA_V3,
    V3_CONTRACT_VERSION,
)
from msctl.aws_sealed_evaluation import (
    load_sealed_evaluation_fixture,
    load_sealed_evaluation_release,
)
from msctl.aws_sealed_finalization import (
    build_finalized_sealed_evaluation,
)
from msctl.cli import build_parser, dispatch
from msctl.errors import MsctlError
from tests.test_v3_hardware_amendment import _sealed_fixture


def _write_inputs(
    root: Path,
) -> tuple[list[Path], list[Path]]:
    checkpoints = root / "checkpoint-records"
    receipts = root / "validity-receipts"
    checkpoints.mkdir()
    receipts.mkdir()
    checkpoint_paths = []
    for seed in range(10):
        for condition_id, arm in (("dense", "dense"), ("split90", "split")):
            value = {
                "record_type": CHECKPOINT_SCHEMA,
                "schema_version": CONTRACT_VERSION,
                "checkpoint_sha256": hashlib.sha256(
                    f"checkpoint:{seed}:{condition_id}".encode()
                ).hexdigest(),
                "model_id": "memorysplit-v3-360m",
                "arm": arm,
                "condition_id": condition_id,
                "seed": seed,
                "raw_token_count": 1,
                "configuration_sha256": hashlib.sha256(
                    f"configuration:{seed}:{condition_id}".encode()
                ).hexdigest(),
                "route_dose_sha256": hashlib.sha256(
                    f"route-dose:{seed}:{condition_id}".encode()
                ).hexdigest(),
                "corpus_sha256": "c" * 64,
                "code_sha256": "d" * 64,
            }
            path = checkpoints / f"{seed}-{condition_id}.json"
            path.write_bytes(canonical_json_bytes(value))
            checkpoint_paths.append(path)

    source_validity = json.loads(
        positive_fixture().artifacts["validity.json"]
    )
    receipt_paths = []
    for index, receipt in enumerate(source_validity["receipts"]):
        path = receipts / f"{index:02d}.json"
        path.write_bytes(canonical_json_bytes(receipt))
        receipt_paths.append(path)
    return checkpoint_paths, receipt_paths


def _jsonl(content: bytes) -> list[dict[str, object]]:
    return [json.loads(line) for line in content.splitlines()]


def _complete_n10_artifacts(
    finalized: dict[str, bytes],
) -> dict[str, bytes]:
    source = positive_fixture().artifacts
    source_checkpoints = {
        (record["seed"], record["condition_id"]): record
        for record in _jsonl(source["checkpoints.jsonl"])
    }
    source_outcomes = _jsonl(source["outcomes.jsonl"])
    source_metrics = json.loads(source["metrics.json"])
    target_checkpoints = _jsonl(finalized["checkpoints.jsonl"])

    outcomes = []
    summaries = []
    for target in target_checkpoints:
        source_checkpoint = source_checkpoints[
            (int(target["seed"]) % 5, target["condition_id"])
        ]
        source_hash = source_checkpoint["checkpoint_sha256"]
        for outcome in source_outcomes:
            if outcome["checkpoint_sha256"] == source_hash:
                outcomes.append(
                    {
                        **outcome,
                        "seed": target["seed"],
                        "checkpoint_sha256": target["checkpoint_sha256"],
                    }
                )
        for summary in source_metrics["summaries"]:
            if summary["checkpoint_sha256"] == source_hash:
                summaries.append(
                    {
                        **summary,
                        "checkpoint_sha256": target["checkpoint_sha256"],
                    }
                )

    outcomes.sort(
        key=lambda row: (
            row["seed"],
            row["condition_id"],
            row["item_id"],
        )
    )
    target_by_hash = {
        checkpoint["checkpoint_sha256"]: checkpoint
        for checkpoint in target_checkpoints
    }
    summaries.sort(
        key=lambda row: (
            target_by_hash[row["checkpoint_sha256"]]["seed"],
            row["condition_id"],
            row["memory_mode"],
            row["control"],
        )
    )
    deltas = tuple(
        json.loads(source["inference.json"])["paired_seed_bundle_deltas"]
    ) * 2
    exact = exact_sign_flip_test(deltas, alternative="greater")
    inference = {
        "record_type": INFERENCE_EVIDENCE_SCHEMA_V3,
        "schema_version": V3_CONTRACT_VERSION,
        "primary_test": {
            "contrast_id": (
                "primary_omnibus_pair_and_proof__graph_non_path__"
                "composition_joint_ood__split90_minus_dense"
            ),
            "method": "exact_one_sided_exhaustive_sign_flip",
            "alternative": "greater",
            "alpha": 0.05,
            "n_pairs": 10,
            "sign_assignments": 1024,
            "equality_counted": True,
        },
        "paired_seed_bundle_deltas": list(deltas),
        "exact_test_result": {
            "statistic": exact.statistic,
            "extreme_count": exact.extreme_count,
            "p_value": exact.p_value,
            "reject_null": exact.statistic > 0.0 and exact.p_value <= 0.05,
        },
    }
    return {
        **finalized,
        "inference.json": canonical_json_bytes(inference),
        "metrics.json": canonical_json_bytes(
            {
                "record_type": METRICS_SCHEMA,
                "schema_version": CONTRACT_VERSION,
                "summaries": summaries,
            }
        ),
        "outcomes.jsonl": b"".join(
            canonical_json_bytes(outcome) for outcome in outcomes
        ),
    }


def test_v3_finalization_and_reporting_complete_all_ten_seeds(tmp_path):
    fixture_root = _sealed_fixture(tmp_path)
    checkpoint_paths, receipt_paths = _write_inputs(tmp_path)
    preregistration_sha256 = "a" * 64

    content, result = build_finalized_sealed_evaluation(
        fixture_root=fixture_root,
        checkpoint_records=checkpoint_paths,
        validity_receipts=receipt_paths,
        preregistration_sha256=preregistration_sha256,
    )

    fixture = load_sealed_evaluation_fixture(fixture_root)
    lock = json.loads(content["study-lock.json"])
    validity = json.loads(content["validity.json"])
    assert result["sealed_fixture_sha256"] == fixture.sha256
    assert lock["record_type"] == STUDY_LOCK_SCHEMA_V3
    assert lock["schema_version"] == V3_CONTRACT_VERSION
    assert validity["schema_version"] == V3_CONTRACT_VERSION
    assert {
        (checkpoint["seed"], checkpoint["condition_id"])
        for checkpoint in lock["checkpoints"]
    } == {
        (seed, condition)
        for seed in range(10)
        for condition in ("dense", "split90")
    }

    artifacts = _complete_n10_artifacts(content)
    report = build_artifact_report(
        artifacts=artifacts,
        expected_study_lock_sha256=str(result["study_lock_sha256"]),
    )
    assert report.record_type == ARTIFACT_REPORT_SCHEMA_V3
    assert report.schema_version == V3_CONTRACT_VERSION
    assert report.scientific_status == "complete"
    assert len(report.paired_seed_bundle_deltas) == 10
    assert json.loads(artifacts["inference.json"])["exact_test_result"][
        "p_value"
    ] == pytest.approx(1 / 1024)
    assert (
        validate_artifact_report(
            report,
            artifacts,
            expected_study_lock_sha256=str(result["study_lock_sha256"]),
        )
        == report
    )


def test_finalization_cli_is_dry_run_first_exclusive_and_fail_closed(tmp_path):
    fixture_root = _sealed_fixture(tmp_path)
    checkpoint_paths, receipt_paths = _write_inputs(tmp_path)
    output = tmp_path / "finalized-sealed-evaluation"
    argv = [
        "sealed-evaluation",
        "finalize",
        "--fixture",
        str(fixture_root),
        *(
            argument
            for path in checkpoint_paths
            for argument in ("--checkpoint-record", str(path))
        ),
        *(
            argument
            for path in receipt_paths
            for argument in ("--validity-receipt", str(path))
        ),
        "--preregistration-sha256",
        "a" * 64,
        "--out",
        str(output),
    ]

    dry_run, planned = dispatch(
        build_parser().parse_args(argv),
        environ={},
        profile_loader=lambda _path: pytest.fail(
            "sealed finalization must not require a compute profile"
        ),
    )
    assert dry_run is True
    assert planned["published"] is False
    assert not output.exists()

    dry_run, published = dispatch(
        build_parser().parse_args([*argv, "--apply"]),
        environ={},
    )
    assert dry_run is False
    assert published["published"] is True
    release = load_sealed_evaluation_release(
        output,
        expected_release_sha256=str(published["sealed_evaluation_sha256"]),
        expected_study_lock_sha256=str(published["study_lock_sha256"]),
        expected_preregistration_sha256="a" * 64,
    )
    assert release.fixture_sha256 == published["sealed_fixture_sha256"]

    with pytest.raises(MsctlError, match="replace"):
        dispatch(
            build_parser().parse_args([*argv, "--apply"]),
            environ={},
        )
    with pytest.raises(MsctlError, match="twenty"):
        build_finalized_sealed_evaluation(
            fixture_root=fixture_root,
            checkpoint_records=checkpoint_paths[:-1],
            validity_receipts=receipt_paths,
            preregistration_sha256="a" * 64,
        )
