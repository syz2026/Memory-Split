from __future__ import annotations

import json

import pytest
import torch

from evals import relational_gates
from evals.relational_controls import ControlID
from evals.relational_stats import ALLOWED_VERDICTS
from scripts.relational_smoke_test import main, run_smoke


def test_smoke_cli_requires_an_explicit_output_directory():
    with pytest.raises(SystemExit):
        main([])


def test_local_pipeline_uses_real_paired_training_and_evaluation(tmp_path):
    report = run_smoke(tmp_path, steps=2, device="cpu")
    expected_fields = {
        "bundle_byte_deterministic",
        "bundle_verified",
        "controls",
        "corpus_builds",
        "corpus_byte_deterministic",
        "corpus_sha256",
        "dense_steps",
        "eval_cells",
        "extracted_bundle_verified",
        "matrix_runs",
        "memory_modes",
        "pairs_complete",
        "resume_compared_next_update",
        "resume_exact",
        "schemas_validated",
        "shared_stream",
        "sidecar_sha256",
        "sidecars",
        "split_steps",
        "synthetic_run_count",
        "verdict_branches",
    }

    assert len(expected_fields) == 21
    assert set(report) == expected_fields
    assert (
        getattr(relational_gates, "SMOKE_REPORT_FIELDS", None)
        == frozenset(expected_fields)
    )
    assert relational_gates.smoke_report_passes(report)
    assert report["shared_stream"] is True
    assert report["dense_steps"] == report["split_steps"] == 2
    assert report["resume_exact"] is True
    assert report["resume_compared_next_update"] is True
    assert report["memory_modes"] == ["off", "on"]
    assert report["pairs_complete"] is True
    assert report["corpus_builds"] == 2
    assert report["corpus_byte_deterministic"] is True
    assert report["sidecars"] == ["dense", "random", "selective", "split"]
    assert report["controls"] == [control.value for control in ControlID]
    assert report["eval_cells"] == 22
    assert report["matrix_runs"] == 35
    assert set(report["verdict_branches"]) == set(ALLOWED_VERDICTS)
    assert report["schemas_validated"] == [
        "freeze-v1.schema.json",
        "relational-asset-receipt-v1.schema.json",
        "relational-result-v1.schema.json",
        "run-config-v1.schema.json",
        "run-manifest-v1.schema.json",
    ]
    assert report["bundle_byte_deterministic"] is True
    assert report["bundle_verified"] is True
    assert report["extracted_bundle_verified"] is True
    assert json.loads((tmp_path / "smoke-report.json").read_text()) == report
    assert (tmp_path / "bundles" / "first.tar.gz").read_bytes() == (
        tmp_path / "bundles" / "second.tar.gz"
    ).read_bytes()

    for arm in ("dense", "split"):
        checkpoint = torch.load(
            tmp_path / "runs" / arm / "ckpt.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["step"] == 2
        config = checkpoint["cfg"]
        assert config["train_bin"] == str(tmp_path / "corpus" / "train.bin")
        assert config["train_weights"] == str(
            tmp_path / "corpus" / f"{arm}.weights.bin"
        )

    for mode in ("off", "on"):
        summary = json.loads(
            (tmp_path / "evals" / f"memory_{mode}" / "summary.json").read_text()
        )
        assert summary["memory"] == mode
        assert summary["n_pairs_per_task"] == 4
        assert all(
            task["n_pairs"] == 4 and task["n_rows"] == 8
            for task in summary["tasks"].values()
        )

    control_report = json.loads(
        (tmp_path / "evals" / "control-matrix.json").read_text()
    )
    assert len(control_report["cells"]) == 22
    assert {
        (cell["control"], cell["memory"])
        for cell in control_report["cells"]
    } == {
        (control.value, mode)
        for control in ControlID
        for mode in ("off", "on")
    }
