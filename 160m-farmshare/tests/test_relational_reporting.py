from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from evals.relational_contracts import canonical_json_bytes
from evals.relational_reporting import (
    REPORT_SECTIONS,
    AnalysisSection,
    build_analysis_document,
    canonical_analysis_bytes,
    publish_analysis_bundle,
    required_report_files,
    validate_analysis_bundle,
)
from evals.relational_stats import CONFIRMATORY_SEEDS
from scripts.analyze_relational import _load_secondary_analyses
from tests.test_relational_stats import _validated_inputs


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _sections(*, wikidata_value=0.61):
    return {
        "paired_deltas": AnalysisSection(
            analysis_role="confirmatory",
            rows=(
                {
                    "contrast": "split_dense_360",
                    "seed": 1001,
                    "delta": 0.03,
                },
                {
                    "contrast": "split_dense_160_high",
                    "seed": 1001,
                    "delta": 0.02,
                },
            ),
        ),
        "dose_interaction": AnalysisSection(
            analysis_role="confirmatory",
            rows=(
                {
                    "seed": 1001,
                    "low_delta": 0.01,
                    "high_delta": 0.03,
                    "interaction": 0.02,
                },
            ),
        ),
        "memory_factorial": AnalysisSection(
            analysis_role="supporting_only",
            rows=(
                {
                    "arm": "split",
                    "memory_mode": "memory_on",
                    "accuracy": 0.70,
                },
                {
                    "arm": "split",
                    "memory_mode": "memory_off",
                    "accuracy": 0.20,
                },
            ),
        ),
        "controls_by_hop_composition": AnalysisSection(
            analysis_role="supporting_only",
            rows=(
                {
                    "arm": "split",
                    "control": "correct",
                    "hop": 2,
                    "composition": "heldout",
                    "accuracy": 0.66,
                },
            ),
        ),
        "guardrails": AnalysisSection(
            analysis_role="instrument_only",
            rows=(
                {
                    "guard": "factual_job",
                    "check": "split_on_recall_floor",
                    "value": 0.96,
                    "passed": True,
                },
            ),
        ),
        "wikidata_robustness": AnalysisSection(
            analysis_role="robustness_only",
            rows=(
                {
                    "dataset": "wikidata5m-inductive",
                    "metric": "pair_accuracy",
                    "value": wikidata_value,
                },
            ),
        ),
    }


def _analysis(*, robustness_value=0.61):
    return build_analysis_document(
        verdict_inputs=_validated_inputs(),
        input_bindings={
            "runs_root_sha256": _sha("runs"),
            "preregistration_sha256": _sha("preregistration"),
            "analysis_code_sha256": _sha("analysis-code"),
            "guardrail_receipt_sha256": [_sha("guardrail")],
        },
        run_matrix=tuple(
            {
                "model": "d160m",
                "arm": "split",
                "load": "n800k",
                "seed": seed,
                "checkpoint_sha256": _sha(f"checkpoint-{seed}"),
            }
            for seed in CONFIRMATORY_SEEDS
        ),
        bootstrap_config={
            "version": "hierarchical-paired-v1",
            "n_boot": 10_000,
            "rng_seed": 20260722,
            "chunk_size": 100,
            "percentile_indices": [249, 9749],
        },
        secondary_analyses={
            "wikidata": {
                "analysis_role": "robustness_only",
                "confirmatory_verdict_eligible": False,
                "value": robustness_value,
            }
        },
    )


def _file_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_reporting_publishes_every_required_deterministic_artifact(tmp_path):
    first = publish_analysis_bundle(
        tmp_path / "first",
        analysis=_analysis(),
        sections=_sections(),
    )
    second = publish_analysis_bundle(
        tmp_path / "second",
        analysis=_analysis(),
        sections=_sections(),
    )

    assert set(_file_bytes(first)) == set(required_report_files())
    assert _file_bytes(first) == _file_bytes(second)
    parsed = validate_analysis_bundle(first)
    assert parsed["verdict"] == "validated"
    assert parsed["seeds"] == list(CONFIRMATORY_SEEDS)
    assert parsed["input_bindings"]["preregistration_sha256"] == _sha(
        "preregistration"
    )
    assert (first / "analysis.json").read_bytes() == canonical_analysis_bytes(
        parsed
    )
    for relative, expected in parsed["artifacts"]["plot_data"].items():
        assert hashlib.sha256((first / relative).read_bytes()).hexdigest() == (
            expected
        )
    assert {
        json.loads((first / relative).read_text())["section"]
        for relative in parsed["artifacts"]["plot_data"]
    } == set(REPORT_SECTIONS)


def test_robustness_changes_cannot_change_verdict_or_decision_hash():
    first = _analysis(robustness_value=0.0)
    second = _analysis(robustness_value=1.0)
    assert first["verdict"] == second["verdict"] == "validated"
    assert first["decision_sha256"] == second["decision_sha256"]
    assert first["secondary_analyses"] != second["secondary_analyses"]
    assert "secondary_analyses" not in first["verdict_inputs"]


def test_selective_or_robustness_data_cannot_enter_confirmatory_sections(
    tmp_path,
):
    sections = _sections()
    sections["paired_deltas"] = AnalysisSection(
        analysis_role="confirmatory",
        rows=(
            {
                "arm": "selective",
                "contrast": "selective_dense",
                "seed": 1001,
                "delta": 1.0,
            },
        ),
    )
    with pytest.raises(ValueError, match="Selective|selective"):
        publish_analysis_bundle(
            tmp_path / "selective",
            analysis=_analysis(),
            sections=sections,
        )

    sections = _sections()
    sections["paired_deltas"] = AnalysisSection(
        analysis_role="robustness_only",
        rows=sections["paired_deltas"].rows,
    )
    with pytest.raises(ValueError, match="analysis role|confirmatory"):
        publish_analysis_bundle(
            tmp_path / "wrong-role",
            analysis=_analysis(),
            sections=sections,
        )


def test_bundle_validation_rejects_malformed_plot_sidecar(tmp_path):
    output = publish_analysis_bundle(
        tmp_path / "bundle",
        analysis=_analysis(),
        sections=_sections(),
    )
    sidecar = (
        output / "figures" / "paired_deltas.plot-data.json"
    )
    value = json.loads(sidecar.read_text())
    value["rows"] = "not-an-array"
    sidecar.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="plot|hash|sidecar"):
        validate_analysis_bundle(output)


def test_bundle_validation_recomputes_verdict_semantics(tmp_path):
    output = publish_analysis_bundle(
        tmp_path / "bundle",
        analysis=_analysis(),
        sections=_sections(),
    )
    path = output / "analysis.json"
    value = json.loads(path.read_bytes())
    value["verdict"] = "practical_null"
    decision = {
        "record_type": "relational_confirmatory_decision",
        "schema_version": 1,
        "seeds": value["seeds"],
        "verdict": value["verdict"],
        "verdict_inputs": value["verdict_inputs"],
        "input_bindings": value["input_bindings"],
        "run_matrix": value["run_matrix"],
        "bootstrap_config": value["bootstrap_config"],
    }
    value["decision_sha256"] = hashlib.sha256(
        canonical_json_bytes(decision)
    ).hexdigest()
    path.write_bytes(canonical_json_bytes(value))

    with pytest.raises(ValueError, match="verdict|decision"):
        validate_analysis_bundle(output)


def test_reporting_rejects_overwrite_traversal_and_symlink_parent(tmp_path):
    output = publish_analysis_bundle(
        tmp_path / "bundle",
        analysis=_analysis(),
        sections=_sections(),
    )
    with pytest.raises(FileExistsError, match="exists"):
        publish_analysis_bundle(
            output,
            analysis=_analysis(),
            sections=_sections(),
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked-parent"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|travers"):
        publish_analysis_bundle(
            link / "escaped",
            analysis=_analysis(),
            sections=_sections(),
        )
    assert not (outside / "escaped").exists()

    traversed = tmp_path / "safe" / ".." / "traversed"
    with pytest.raises(ValueError, match="travers"):
        publish_analysis_bundle(
            traversed,
            analysis=_analysis(),
            sections=_sections(),
        )


def test_reporting_failure_leaves_no_partial_publication(
    tmp_path,
    monkeypatch,
):
    import evals.relational_reporting as reporting

    calls = 0
    original = reporting.render_plot_svg

    def fail_mid_bundle(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected plotting failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(reporting, "render_plot_svg", fail_mid_bundle)
    destination = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected"):
        publish_analysis_bundle(
            destination,
            analysis=_analysis(),
            sections=_sections(),
        )
    assert not os.path.lexists(destination)
    assert not list(tmp_path.glob(".failed.stage-*"))


def test_secondary_robustness_symlink_is_rejected_before_read(
    tmp_path,
    monkeypatch,
):
    outside = tmp_path / "outside.json"
    outside.write_text(
        json.dumps(
            {
                "analysis_role": "robustness_only",
                "confirmatory_verdict_eligible": False,
            }
        )
    )
    linked = tmp_path / "linked.json"
    linked.symlink_to(outside)
    original = Path.read_bytes

    def guarded_read(path):
        if path == linked:
            raise AssertionError("symlink content was read")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    with pytest.raises(ValueError, match="regular|symlink"):
        _load_secondary_analyses([str(linked)])


def test_analysis_document_rejects_missing_hashes_and_ad_hoc_config():
    with pytest.raises(ValueError, match="input binding|hash"):
        build_analysis_document(
            verdict_inputs=_validated_inputs(),
            input_bindings={"preregistration_sha256": "not-a-hash"},
            run_matrix=(),
            bootstrap_config={
                "version": "hierarchical-paired-v1",
                "n_boot": 10_000,
                "rng_seed": 1,
                "chunk_size": 100,
                "percentile_indices": [249, 9749],
            },
            secondary_analyses={},
        )
    with pytest.raises(ValueError, match="n_boot|10,000"):
        build_analysis_document(
            verdict_inputs=_validated_inputs(),
            input_bindings={
                "runs_root_sha256": _sha("runs"),
                "preregistration_sha256": _sha("pre"),
                "analysis_code_sha256": _sha("code"),
                "guardrail_receipt_sha256": [_sha("guard")],
            },
            run_matrix=(),
            bootstrap_config={
                "version": "hierarchical-paired-v1",
                "n_boot": 9_999,
                "rng_seed": 1,
                "chunk_size": 100,
                "percentile_indices": [249, 9749],
            },
            secondary_analyses={},
        )
