from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import subprocess
import sys

import pytest

import scripts.analyze_relational as relational_analysis
from evals.relational_contracts import (
    EvalRow,
    GuardrailReport,
    canonical_json_bytes,
)
from evals.relational_metrics import EXPECTED_TASKS
from evals.relational_stats import (
    ALLOWED_VERDICTS,
    CONFIRMATORY_SEEDS,
    ContrastEstimate,
    VerdictInputs,
    allowed_verdicts,
    bootstrap_trace,
    decide_verdict,
    hierarchical_paired_bootstrap,
    hierarchical_paired_bootstrap_reference,
    hierarchical_paired_contrasts,
    percentile_interval,
    verdict_inputs_to_dict,
)
from scripts.freeze_relational_study import make_fixture_freeze
from scripts.make_relational_manifest import build_manifest
from scripts.analyze_relational import (
    _require_expected_run_matrix,
    analyze_runs,
    expected_run_keys,
)
from tests.task8_helpers import (
    PROTECTED_SEEDS,
    make_rows,
    make_summary,
    replace_row,
)
from tests.test_relational_contracts import _guardrail_report


def _all_false(_label: str, _seed: int, _task: str, _index: int) -> bool:
    return False


def _run_manifest():
    return build_manifest(make_fixture_freeze())


def test_confirmatory_seed_contract_and_run_matrix_are_exact():
    manifest = _run_manifest()
    assert CONFIRMATORY_SEEDS == PROTECTED_SEEDS
    assert set(expected_run_keys(manifest)) == {
        *{
            (model, arm, load, seed)
            for model, load, arms in (
                ("d160m", "n50k", ("dense", "split")),
                ("d160m", "n800k", ("dense", "split", "random")),
                ("d360m", "n1p8m", ("dense", "split")),
            )
            for arm in arms
            for seed in PROTECTED_SEEDS
        }
    }
    assert len(expected_run_keys(manifest)) == 35
    _require_expected_run_matrix(
        {key: {} for key in expected_run_keys(manifest)},
        manifest,
    )

    missing = {key: {} for key in expected_run_keys(manifest)}
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="run matrix mismatch"):
        _require_expected_run_matrix(missing, manifest)


def test_point_estimator_equal_weights_tasks_with_unequal_item_counts():
    pair_counts = {
        "path_composition": 1,
        "date_ordering": 7,
        "balanced_equality": 2,
    }

    def split_success(_label, _seed, task, index):
        if task == "path_composition":
            return True
        if task == "date_ordering":
            return False
        return index == 0

    split = make_rows(
        "split",
        "split",
        pair_counts=pair_counts,
        success=split_success,
    )
    dense = make_rows(
        "dense",
        "dense",
        pair_counts=pair_counts,
        success=_all_false,
    )

    estimate = hierarchical_paired_bootstrap(
        split,
        dense,
        seeds=PROTECTED_SEEDS,
        n_boot=200,
        rng_seed=91,
    )

    assert estimate.mean == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    assert estimate.seed_deltas == pytest.approx((0.5,) * 5)
    assert estimate.cohen_dz is None
    assert estimate.effect_note == "zero_seed_sd"


@pytest.mark.parametrize(
    "seeds",
    [
        PROTECTED_SEEDS[:-1],
        (*PROTECTED_SEEDS[:-1], 1006),
        (*PROTECTED_SEEDS, 1006),
        (True, 1002, 1003, 1004, 1005),
    ],
)
def test_confirmatory_bootstrap_rejects_missing_or_unexpected_seeds(seeds):
    split = make_rows("split", "split")
    dense = make_rows("dense", "dense")
    with pytest.raises(ValueError, match="seeds.*1001.*1005|confirmatory"):
        hierarchical_paired_bootstrap(
            split,
            dense,
            seeds=seeds,
            n_boot=20,
            rng_seed=7,
        )


def test_exact_pairing_rejects_duplicates_missing_twins_and_crossed_metadata():
    split = make_rows("split", "split")
    dense = make_rows("dense", "dense")

    with pytest.raises(ValueError, match="duplicate"):
        hierarchical_paired_bootstrap(
            (*split, split[0]),
            dense,
            seeds=PROTECTED_SEEDS,
            n_boot=20,
            rng_seed=7,
        )
    with pytest.raises(ValueError, match="both variants|complete"):
        hierarchical_paired_bootstrap(
            split[:-1],
            dense,
            seeds=PROTECTED_SEEDS,
            n_boot=20,
            rng_seed=7,
        )

    crossed = list(dense)
    pair_id = crossed[0].pair_id
    for index, row in enumerate(crossed):
        if row.pair_id == pair_id:
            crossed[index] = replace_row(row, composition_split="heldout")
    with pytest.raises(ValueError, match="composition|metadata"):
        hierarchical_paired_bootstrap(
            split,
            crossed,
            seeds=PROTECTED_SEEDS,
            n_boot=20,
            rng_seed=7,
        )

    changed_key = list(dense)
    for index, row in enumerate(changed_key):
        if row.pair_id == pair_id:
            changed_key[index] = replace_row(row, template_id="crossed:v2")
    with pytest.raises(ValueError, match="join|missing|crossed"):
        hierarchical_paired_bootstrap(
            split,
            changed_key,
            seeds=PROTECTED_SEEDS,
            n_boot=20,
            rng_seed=7,
        )


def test_analysis_rejects_non_typed_unknown_task_arm_and_crossed_checkpoint():
    split = make_rows("split", "split")
    dense = make_rows("dense", "dense")

    with pytest.raises(TypeError, match="EvalRow"):
        hierarchical_paired_bootstrap(
            [row.to_dict() for row in split],
            dense,
            seeds=PROTECTED_SEEDS,
            n_boot=20,
            rng_seed=7,
        )

    selective = tuple(replace_row(row, arm="selective") for row in split)
    with pytest.raises(ValueError, match="arm"):
        hierarchical_paired_bootstrap(
            selective,
            dense,
            seeds=PROTECTED_SEEDS,
            n_boot=20,
            rng_seed=7,
        )

    factual = list(split)
    pair_id = factual[0].pair_id
    for index, row in enumerate(factual):
        if row.pair_id == pair_id:
            factual[index] = replace_row(row, task="factual_recall")
    with pytest.raises(ValueError, match="task"):
        hierarchical_paired_bootstrap(
            factual,
            dense,
            seeds=PROTECTED_SEEDS,
            n_boot=20,
            rng_seed=7,
        )

    crossed = list(dense)
    crossed[0] = replace_row(crossed[0], checkpoint_sha256="f" * 64)
    with pytest.raises(ValueError, match="checkpoint"):
        hierarchical_paired_bootstrap(
            split,
            crossed,
            seeds=PROTECTED_SEEDS,
            n_boot=20,
            rng_seed=7,
        )


def test_twins_all_arms_and_clusters_share_every_resample_draw():
    rows_by_arm = {
        "split": make_rows(
            "split",
            "split",
            pair_counts={task: 16 for task in EXPECTED_TASKS},
        ),
        "dense": make_rows(
            "dense",
            "dense",
            pair_counts={task: 16 for task in EXPECTED_TASKS},
        ),
        "random": make_rows(
            "random",
            "random",
            pair_counts={task: 16 for task in EXPECTED_TASKS},
        ),
    }

    trace = bootstrap_trace(
        rows_by_arm,
        seeds=PROTECTED_SEEDS,
        n_boot=20,
        rng_seed=9,
    )

    assert len(trace) == 20
    assert any(
        any(value > 1 for value in draw.pair_multiplicities.values())
        for draw in trace
    )
    for draw in trace:
        baseline = draw.multiplicity("dense", "original")
        assert baseline == draw.multiplicity("split", "original")
        assert baseline == draw.multiplicity("dense", "counterfactual")
        assert baseline == draw.multiplicity("random", "counterfactual")
        assert draw.cluster_multiplicities


def test_bootstrap_is_order_chunk_and_hash_seed_independent():
    split = make_rows(
        "split",
        "split",
        success=lambda _l, seed, task, index: (
            seed + EXPECTED_TASKS.index(task) + index
        )
        % 3
        != 0,
        pair_counts={task: 8 for task in EXPECTED_TASKS},
    )
    dense = make_rows(
        "dense",
        "dense",
        success=lambda _l, seed, task, index: (
            seed + EXPECTED_TASKS.index(task) + index
        )
        % 4
        != 0,
        pair_counts={task: 8 for task in EXPECTED_TASKS},
    )

    expected = hierarchical_paired_bootstrap(
        split,
        dense,
        seeds=PROTECTED_SEEDS,
        n_boot=250,
        rng_seed=123456,
        chunk_size=1,
    )
    for chunk_size in (7, 37, 100):
        actual = hierarchical_paired_bootstrap(
            tuple(reversed(split)),
            tuple(reversed(dense)),
            seeds=PROTECTED_SEEDS,
            n_boot=250,
            rng_seed=123456,
            chunk_size=chunk_size,
        )
        assert actual == expected
    assert (
        hierarchical_paired_bootstrap_reference(
            split,
            dense,
            seeds=PROTECTED_SEEDS,
            n_boot=250,
            rng_seed=123456,
        )
        == expected
    )


def test_paired_did_and_split_random_use_shared_draws():
    pair_counts = {task: 10 for task in EXPECTED_TASKS}

    def first_n(n):
        return lambda _label, _seed, _task, index: index < n

    rows = {
        "split_low": make_rows(
            "split-low", "split", pair_counts=pair_counts, success=first_n(2)
        ),
        "dense_low": make_rows(
            "dense-low", "dense", pair_counts=pair_counts, success=first_n(0)
        ),
        "split_high": make_rows(
            "split-high", "split", pair_counts=pair_counts, success=first_n(8)
        ),
        "dense_high": make_rows(
            "dense-high", "dense", pair_counts=pair_counts, success=first_n(1)
        ),
        "random_high": make_rows(
            "random-high", "random", pair_counts=pair_counts, success=first_n(3)
        ),
    }
    estimates = hierarchical_paired_contrasts(
        rows,
        {
            "split_dense_low": {"split_low": 1.0, "dense_low": -1.0},
            "split_dense_high": {"split_high": 1.0, "dense_high": -1.0},
            "dose_interaction": {
                "split_high": 1.0,
                "dense_high": -1.0,
                "split_low": -1.0,
                "dense_low": 1.0,
            },
            "split_random_high": {
                "split_high": 1.0,
                "random_high": -1.0,
            },
        },
        seeds=PROTECTED_SEEDS,
        n_boot=100,
        rng_seed=71,
    )

    assert estimates["split_dense_low"].mean == pytest.approx(0.2)
    assert estimates["split_dense_high"].mean == pytest.approx(0.7)
    assert estimates["dose_interaction"].mean == pytest.approx(0.5)
    assert estimates["split_random_high"].mean == pytest.approx(0.5)


def test_identity_groups_do_not_relax_shared_evaluator_identity():
    rows = {
        label: make_rows(label, arm)
        for label, arm in (
            ("split_low", "split"),
            ("dense_low", "dense"),
            ("split_high", "split"),
            ("dense_high", "dense"),
            ("random_high", "random"),
        )
    }
    low_evaluator = hashlib.sha256(b"low-evaluator").hexdigest()
    high_evaluator = hashlib.sha256(b"high-evaluator").hexdigest()
    for label in ("split_low", "dense_low"):
        rows[label] = tuple(
            replace_row(row, evaluator_sha256=low_evaluator)
            for row in rows[label]
        )
    for label in ("split_high", "dense_high", "random_high"):
        rows[label] = tuple(
            replace_row(row, evaluator_sha256=high_evaluator)
            for row in rows[label]
        )

    with pytest.raises(
        ValueError,
        match=r"evaluator_sha256.*shared_pair_hierarchy",
    ):
        hierarchical_paired_contrasts(
            rows,
            {
                "split_dense_low": {
                    "split_low": 1.0,
                    "dense_low": -1.0,
                },
                "split_dense_high": {
                    "split_high": 1.0,
                    "dense_high": -1.0,
                },
                "dose_interaction": {
                    "split_high": 1.0,
                    "dense_high": -1.0,
                    "split_low": -1.0,
                    "dense_low": 1.0,
                },
                "split_random_high": {
                    "split_high": 1.0,
                    "random_high": -1.0,
                },
            },
            seeds=PROTECTED_SEEDS,
            n_boot=20,
            rng_seed=71,
            identity_groups={
                "d160m_low": ("dense_low", "split_low"),
                "d160m_high": (
                    "dense_high",
                    "random_high",
                    "split_high",
                ),
            },
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("n_boot", True),
        ("n_boot", 0),
        ("n_boot", -1),
        ("rng_seed", True),
        ("rng_seed", -1),
        ("chunk_size", 0),
        ("chunk_size", 101),
    ],
)
def test_bootstrap_configuration_rejects_bool_and_out_of_range(name, value):
    split = make_rows("split", "split")
    dense = make_rows("dense", "dense")
    kwargs = {
        "seeds": PROTECTED_SEEDS,
        "n_boot": 20,
        "rng_seed": 7,
        "chunk_size": 20,
    }
    kwargs[name] = value
    with pytest.raises(ValueError, match=name.replace("_", " ")):
        hierarchical_paired_bootstrap(split, dense, **kwargs)


def test_percentile_index_convention_is_frozen_at_ten_thousand():
    values = tuple(float(index) for index in reversed(range(10_000)))
    assert percentile_interval(values) == (249.0, 9749.0)
    with pytest.raises(ValueError, match="finite"):
        percentile_interval([0.0, math.nan])


def _estimate(
    mean: float,
    ci_lo: float,
    ci_hi: float,
    *,
    seed_deltas: tuple[float, ...] | None = None,
) -> ContrastEstimate:
    deltas = (mean,) * 5 if seed_deltas is None else seed_deltas
    return ContrastEstimate(
        mean=mean,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        seed_deltas=deltas,
        cohen_dz=None,
        effect_note="test fixture",
    )


def _passing_report() -> GuardrailReport:
    return GuardrailReport.from_dict(_guardrail_report())


def _failing_report() -> GuardrailReport:
    value = _guardrail_report()
    check = value["guards"]["split_off_leakage"]["checks"][0]
    check.update(
        value=0.05,
        numerator=1,
        denominator=20,
        passed=False,
    )
    value["guards"]["split_off_leakage"]["passed"] = False
    value["confirmatory_passed"] = False
    return GuardrailReport.from_dict(value)


def _analysis_runs(
    *,
    mismatched_identity: tuple[str, str, str] | None = None,
) -> dict:
    runs = {}
    manifest = _run_manifest()
    for model, arm, load, seed in sorted(expected_run_keys(manifest)):
        identity = f"analysis-data:{model}:{load}:{seed}"
        if mismatched_identity == (model, arm, load):
            identity += f":{arm}-mismatch"
        data_sha256 = hashlib.sha256(identity.encode()).hexdigest()
        rows = tuple(
            replace_row(row, data_sha256=data_sha256)
            for row in make_rows(
                f"{model}-{arm}-{load}",
                arm,
                seeds=(seed,),
                model_id=model,
                namespace=f"protected-{model}",
                success=lambda label, *_: "-split-" in label,
            )
        )
        summary = make_summary(rows)
        runs[(model, arm, load, seed)] = {
            "on": summary,
            "off": summary,
            "rows": rows,
        }
    return runs


def _analysis_input_bindings(report: GuardrailReport) -> dict:
    return {
        "runs_root_sha256": "a" * 64,
        "preregistration_sha256": "b" * 64,
        "analysis_code_sha256": "c" * 64,
        "guardrail_receipt_sha256": [report.pairing_receipt_sha256],
    }


def test_analyze_runs_uses_row_level_hierarchical_estimates(monkeypatch):
    report = _passing_report()
    monkeypatch.setattr(
        relational_analysis,
        "_collect_guardrails",
        lambda _runs: (report,),
    )
    runs = _analysis_runs()

    result = analyze_runs(
        runs,
        run_manifest=_run_manifest(),
        rng_seed=20260722,
        input_bindings=_analysis_input_bindings(report),
        secondary_analyses={},
    )

    assert result["verdict"] == "inconclusive"
    assert result["verdict_inputs"]["split_dense_360"]["mean"] == 1.0
    assert (
        result["verdict_inputs"]["split_dense_160_high"]["mean"] == 1.0
    )
    assert result["verdict_inputs"]["dose_interaction_160"]["mean"] == 0.0
    assert (
        result["verdict_inputs"]["split_random_160_high"]["mean"] == 1.0
    )
    assert result["bootstrap_config"]["n_boot"] == 10_000
    assert "pooled_seed_sigma" not in result
    assert "paired_360_interval" not in result
    sections = relational_analysis.build_report_sections(
        runs,
        result,
        run_manifest=_run_manifest(),
        guardrails=(report,),
        secondary_analyses={},
    )
    task_rows = [
        row
        for row in sections["paired_deltas"].rows
        if "task" in row
    ]
    assert {row["scale"] for row in task_rows} == {
        "d160m_high",
        "d360m_confirmation",
    }
    assert {row["task"] for row in task_rows} == set(EXPECTED_TASKS)
    assert {row["seed"] for row in task_rows} == set(PROTECTED_SEEDS)


def test_analyze_runs_rejects_identity_mismatch_within_low_load(monkeypatch):
    report = _passing_report()
    monkeypatch.setattr(
        relational_analysis,
        "_collect_guardrails",
        lambda _runs: (report,),
    )
    runs = _analysis_runs(
        mismatched_identity=("d160m", "split", "n50k")
    )

    with pytest.raises(
        ValueError,
        match=r"data_sha256.*d160m_low|d160m_low.*data_sha256",
    ):
        analyze_runs(
            runs,
            run_manifest=_run_manifest(),
            rng_seed=20260722,
            input_bindings=_analysis_input_bindings(report),
            secondary_analyses={},
        )


def _validated_inputs(**changes) -> VerdictInputs:
    values = {
        "split_dense_360": _estimate(0.02, 0.001, 0.04),
        "split_dense_160_high": _estimate(0.02, 0.001, 0.04),
        "dose_interaction_160": _estimate(0.01, 0.001, 0.03),
        "split_random_160_high": _estimate(0.01, 0.001, 0.03),
        "task_means_360": {task: 0.001 for task in EXPECTED_TASKS},
        "task_means_160_high": {
            task: 0.001 for task in EXPECTED_TASKS
        },
        "guardrail_reports": (_passing_report(),),
    }
    values.update(changes)
    return VerdictInputs(**values)


def _practical_null_inputs(**changes) -> VerdictInputs:
    values = {
        "split_dense_360": _estimate(0.0, -0.01, 0.019),
        "split_dense_160_high": _estimate(0.0, -0.01, 0.019),
        "dose_interaction_160": _estimate(-0.01, -0.02, 0.0),
        "split_random_160_high": _estimate(0.0, -0.01, 0.01),
        "task_means_360": {task: 0.0 for task in EXPECTED_TASKS},
        "task_means_160_high": {task: 0.0 for task in EXPECTED_TASKS},
        "guardrail_reports": (_passing_report(),),
    }
    values.update(changes)
    return VerdictInputs(**values)


def test_validated_requires_every_confirmatory_condition():
    assert decide_verdict(_validated_inputs()) == "validated"

    failures = [
        {"split_dense_360": _estimate(0.019, 0.001, 0.04)},
        {"split_dense_360": _estimate(0.02, 0.0, 0.04)},
        {"split_dense_160_high": _estimate(0.02, 0.0, 0.04)},
        {"dose_interaction_160": _estimate(0.01, 0.0, 0.03)},
        {"split_random_160_high": _estimate(0.01, 0.0, 0.03)},
    ]
    failures.extend(
        {
            field: {
                **{task: 0.001 for task in EXPECTED_TASKS},
                task_name: 0.0,
            }
        }
        for field in ("task_means_360", "task_means_160_high")
        for task_name in EXPECTED_TASKS
    )
    for failure in failures:
        assert decide_verdict(_validated_inputs(**failure)) != "validated"

    assert (
        decide_verdict(
            _validated_inputs(guardrail_reports=(_failing_report(),))
        )
        == "invalid"
    )


def test_verdict_inputs_bind_the_full_guardrail_report():
    report = _passing_report()
    serialized = verdict_inputs_to_dict(
        _validated_inputs(guardrail_reports=(report,))
    )
    expected = hashlib.sha256(canonical_json_bytes(report)).hexdigest()

    assert serialized["guardrail_report_sha256"] == [expected]
    assert expected != report.pairing_receipt_sha256


def test_practical_null_boundaries_are_exact_and_nonoverlapping():
    assert decide_verdict(_practical_null_inputs()) == "practical_null"
    assert (
        decide_verdict(
            _practical_null_inputs(
                split_dense_160_high=_estimate(0.0, -0.01, 0.02)
            )
        )
        == "inconclusive"
    )
    assert (
        decide_verdict(
            _practical_null_inputs(
                split_dense_360=_estimate(0.0, -0.01, 0.02)
            )
        )
        == "inconclusive"
    )
    assert (
        decide_verdict(
            _practical_null_inputs(
                dose_interaction_160=_estimate(-0.01, -0.02, 0.000001)
            )
        )
        == "inconclusive"
    )
    assert (
        decide_verdict(
            _practical_null_inputs(
                split_random_160_high=_estimate(
                    0.0, -0.01, 0.010001
                )
            )
        )
        == "inconclusive"
    )
    with pytest.raises(ValueError, match="interval|mean"):
        _estimate(0.02, 0.001, 0.019)


def test_verdict_rejects_nonfinite_missing_unknown_or_adhoc_guardrails():
    with pytest.raises(ValueError, match="finite"):
        ContrastEstimate(
            mean=math.nan,
            ci_lo=0.0,
            ci_hi=1.0,
            seed_deltas=(0.0,) * 5,
            cohen_dz=None,
            effect_note="bad",
        )
    with pytest.raises(ValueError, match="task"):
        decide_verdict(
            _validated_inputs(
                task_means_360={"path_composition": 0.1}
            )
        )
    with pytest.raises((TypeError, ValueError), match="GuardrailReport"):
        decide_verdict(
            _validated_inputs(
                guardrail_reports=({"confirmatory_passed": True},)
            )
        )
    with pytest.raises(TypeError):
        VerdictInputs(
            split_dense_360=_estimate(0.02, 0.001, 0.04),
            split_dense_160_high=_estimate(0.02, 0.001, 0.04),
            dose_interaction_160=_estimate(0.01, 0.001, 0.03),
            split_random_160_high=_estimate(0.01, 0.001, 0.03),
            task_means_360={task: 0.1 for task in EXPECTED_TASKS},
            task_means_160_high={
                task: 0.1 for task in EXPECTED_TASKS
            },
        )


def test_allowed_verdict_enum_has_no_rejected_outcome():
    assert ALLOWED_VERDICTS == (
        "validated",
        "practical_null",
        "inconclusive",
        "invalid",
    )
    assert allowed_verdicts() == ALLOWED_VERDICTS
    assert "rejected" not in allowed_verdicts()


def test_relational_analysis_cli_is_repo_relative_and_strict():
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/analyze_relational.py", "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--runs-root" in completed.stdout
    assert "--run-manifest" in completed.stdout
    assert "--preregistration" in completed.stdout
    assert "--rng-seed" in completed.stdout
    assert "legacy" not in completed.stdout.lower()
