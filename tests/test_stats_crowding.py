"""Seed-level inference, the split gates, the frozen verdict, and an analyzer
that refuses anything but a complete matrix."""

import importlib.util
import json
from pathlib import Path

import pytest

from evals.stats import (
    check_gates,
    min_detectable_effect,
    paired_delta_iid,
    seed_contrast,
    t_ppf,
    t_sf,
    verdict,
)

_SPEC = importlib.util.spec_from_file_location(
    "analyze_crowding",
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_crowding.py",
)
ac = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ac)


# ------------------------------------------------------------ t distribution


@pytest.mark.parametrize("df,want", [(4, 2.13185), (7, 1.89458), (11, 1.79588)])
def test_t_critical_values_match_the_tables(df, want):
    assert abs(t_ppf(0.95, df) - want) < 1e-3


def test_t_survival_is_the_inverse_of_the_quantile():
    for df in (3, 6, 12):
        assert abs(t_sf(t_ppf(0.95, df), df) - 0.05) < 1e-6


# ------------------------------------------------------------ paired_delta_iid


def _rows(vals, key="m"):
    return [{"qid": str(i), key: v} for i, v in enumerate(vals)]


def test_paired_delta_iid_matches_by_qid():
    out = paired_delta_iid(_rows([1, 1, 1]), _rows([0, 0, 0]), metric="m")
    assert out["delta"] == pytest.approx(1.0)
    assert out["n"] == 3


def test_paired_delta_iid_drops_missing_pairwise():
    """The continuous metrics are undefined for items with no worked
    solution; imputing would invent data."""
    a = [{"qid": "1", "m": 1.0}, {"qid": "2", "m": None}, {"qid": "3", "m": 2.0}]
    b = [{"qid": "1", "m": 0.0}, {"qid": "2", "m": 0.0}, {"qid": "3", "m": 0.0}]
    out = paired_delta_iid(a, b, metric="m")
    assert out["n"] == 2 and out["n_dropped"] == 1


def test_lower_is_better_flips_the_sign():
    hi = paired_delta_iid(_rows([2.0]), _rows([1.0]), metric="m")
    lo = paired_delta_iid(_rows([2.0]), _rows([1.0]), metric="m", lower_is_better=True)
    assert hi["delta"] == pytest.approx(-lo["delta"])


# ------------------------------------------------------------ seed_contrast


def test_seed_contrast_reports_a_one_sided_lower_bound():
    out = seed_contrast([2.0, 2.2, 1.8, 2.1, 1.9, 2.0])
    assert out["n"] == 6 and out["mean"] == pytest.approx(2.0)
    assert out["ci_lower"] < out["mean"] < out["ci_upper"]
    assert out["p_one_sided"] < 0.001


def test_sign_test_floor_is_reported_at_small_n():
    """At n=3 the smallest attainable one-sided p is 0.125, so a perfectly
    consistent result still cannot clear 0.05 on sign alone. Saying so beats
    quietly reporting a non-significant p."""
    out = seed_contrast([1.0, 2.0, 3.0])
    assert out["sign_consistent"] is True
    assert out["p_sign"] == pytest.approx(0.125)
    assert out["min_attainable_p_sign"] == pytest.approx(0.125)


def test_seed_contrast_is_honest_about_n_equals_one():
    out = seed_contrast([1.0])
    assert out["underpowered"] is True
    assert out["sd"] != out["sd"]  # nan


def test_min_detectable_effect_replaces_an_assumed_number():
    mde = min_detectable_effect(sd=1.5, n=8)
    assert 1.0 < mde < 2.5
    assert min_detectable_effect(1.5, 24) < mde  # more seeds detect less


# ------------------------------------------------------------ gates


def _good():
    return dict(
        bits={"sup": 0.42, "factmask": 0.01, "randpos": 0.40},
        bits_lower_bound={"sup": 0.35},
        igsm_acc_sup=0.42,
        clip_ratio={"sup": 0.98, "factmask": 0.99, "randpos": 0.98},
        storage_floor=0.30,
        igsm_band=(0.17, 0.85),
    )


def test_all_gates_pass_on_a_healthy_cohort():
    assert all(g["ok"] for g in check_gates(**_good()))


def test_gates_are_split_by_consequence():
    kinds = {g["name"]: g["kind"] for g in check_gates(**_good())}
    assert kinds["sup_carries_a_burden"] == "validity"
    assert kinds["factmask_removed_it"] == "validity"
    assert kinds["endpoint_is_measurable"] == "validity"
    assert kinds["randpos_kept_the_burden"] == "reporting"
    assert kinds["clipping_is_comparable"] == "reporting"


def test_no_burden_is_a_validity_failure():
    kw = _good()
    kw["bits_lower_bound"] = {"sup": 0.001}
    bad = [g for g in check_gates(**kw) if not g["ok"]]
    assert [g["name"] for g in bad] == ["sup_carries_a_burden"]


def test_leaked_facts_are_a_validity_failure():
    kw = _good()
    kw["bits"] = {"sup": 0.42, "factmask": 0.30, "randpos": 0.40}
    names = [g["name"] for g in check_gates(**kw) if not g["ok"]]
    assert "factmask_removed_it" in names


def test_a_floored_endpoint_is_a_validity_failure():
    kw = _good()
    kw["igsm_acc_sup"] = 0.07
    names = [g["name"] for g in check_gates(**kw) if not g["ok"]]
    assert "endpoint_is_measurable" in names


def test_an_unmatched_control_is_disclosed_not_fatal():
    """RANDPOS failing to match is itself a finding; it must not silently
    void a cohort that is otherwise sound."""
    kw = _good()
    kw["bits"] = {"sup": 0.42, "factmask": 0.01, "randpos": 0.05}
    gates = check_gates(**kw)
    v = verdict([2.0] * 6, gates, 1.0)
    assert v["verdict"] != "invalid"
    assert "randpos_kept_the_burden" in v["disclosed_reporting_failures"]


# ------------------------------------------------------------ verdict


def test_validated_requires_the_lower_bound_above_the_effect_of_interest():
    v = verdict([2.0, 2.2, 1.8, 2.1, 1.9, 2.0], check_gates(**_good()), 1.0)
    assert v["verdict"] == "validated"


def test_a_practical_null_is_rejected_not_inconclusive():
    v = verdict([0.05, -0.02, 0.01, 0.0, 0.03, -0.01], check_gates(**_good()), 1.0)
    assert v["verdict"] == "rejected"


def test_a_wide_interval_is_inconclusive():
    v = verdict([3.0, -2.0, 2.5, -1.5, 0.5, 1.0], check_gates(**_good()), 1.0)
    assert v["verdict"] == "inconclusive"


def test_every_gate_failing_yields_invalid():
    kw = dict(
        bits={"sup": 0.001, "factmask": 0.0009, "randpos": 0.0001},
        bits_lower_bound={"sup": 0.0001},
        igsm_acc_sup=0.07,
        clip_ratio={"sup": 0.4, "factmask": 0.9, "randpos": 0.5},
        storage_floor=0.30,
        igsm_band=(0.17, 0.85),
    )
    v = verdict([5.0] * 6, check_gates(**kw), 1.0)
    assert v["verdict"] == "invalid"
    assert len(v["failed_validity_gates"]) == 3


# ------------------------------------------------------------ the analyzer


def _write_cell(root: Path, load: str, seed: int, arm: str, **vals):
    d = root / f"d40m_{load}_{arm}_s{seed}" / "evals"
    d.mkdir(parents=True, exist_ok=True)
    base = {"igsm_acc": 0.40, "recoverable_bits_per_param": 0.40, "clip_ratio": 0.98}
    base.update(vals)
    (d / "summary.json").write_text(json.dumps(base))


def _matrix(root: Path, loads=("high", "low"), seeds=(0, 1, 2)):
    for load in loads:
        for s in seeds:
            _write_cell(root, load, s, "sup", recoverable_bits_per_param=0.42)
            _write_cell(root, load, s, "factmask", igsm_acc=0.44,
                        recoverable_bits_per_param=0.01)
            _write_cell(root, load, s, "randpos", igsm_acc=0.41,
                        recoverable_bits_per_param=0.40)


def test_analyzer_runs_on_a_complete_matrix(tmp_path):
    _matrix(tmp_path)
    out = ac.analyse(tmp_path, "igsm_acc", 0.30, (0.17, 0.85), 0.01,
                     primary_load="high")
    assert out["n_cells"] == 6
    assert out["primary_contrast"] == "Y[factmask] - Y[randpos]"
    assert set(out["per_load"]) == {"high", "low"}


def test_analyzer_refuses_a_missing_arm(tmp_path):
    _matrix(tmp_path)
    import shutil
    shutil.rmtree(tmp_path / "d40m_high_randpos_s1")
    with pytest.raises(ac.IncompleteMatrix, match="missing arms"):
        ac.analyse(tmp_path, "igsm_acc", 0.30, (0.17, 0.85), 0.01)


def test_analyzer_refuses_a_dropped_seed(tmp_path):
    """Analysing, dropping the seed you dislike, then re-analysing is the
    same thing as choosing it."""
    _matrix(tmp_path)
    import shutil
    for arm in ("sup", "factmask", "randpos"):
        shutil.rmtree(tmp_path / f"d40m_high_{arm}_s2")
    with pytest.raises(ac.IncompleteMatrix, match="missing cell"):
        ac.analyse(tmp_path, "igsm_acc", 0.30, (0.17, 0.85), 0.01)


def test_analyzer_refuses_an_unexpected_seed_set(tmp_path):
    _matrix(tmp_path, seeds=(0, 1))
    with pytest.raises(ac.IncompleteMatrix, match="seeds"):
        ac.analyse(tmp_path, "igsm_acc", 0.30, (0.17, 0.85), 0.01,
                   expect_seeds={0, 1, 2})


def test_analyzer_refuses_a_run_without_evaluations(tmp_path):
    _matrix(tmp_path)
    (tmp_path / "d40m_low_sup_s0" / "evals" / "summary.json").unlink()
    with pytest.raises(ac.IncompleteMatrix, match="no evals"):
        ac.analyse(tmp_path, "igsm_acc", 0.30, (0.17, 0.85), 0.01,
                   primary_load="high")


def test_analyzer_refuses_an_empty_root(tmp_path):
    with pytest.raises(ac.IncompleteMatrix, match="no runs"):
        ac.analyse(tmp_path, "igsm_acc", 0.30, (0.17, 0.85), 0.01)


def test_analyzer_states_the_estimand_limit(tmp_path):
    _matrix(tmp_path)
    out = ac.analyse(tmp_path, "igsm_acc", 0.30, (0.17, 0.85), 0.01,
                     primary_load="high")
    assert "NOT" in out["estimand_note"] and "identified" in out["estimand_note"]


# ------------------------------------ seed-blocked inference, added 2026-08-02


def test_analyzer_refuses_to_pick_the_primary_load_itself(tmp_path):
    """Every rule for choosing it from data -- highest storage, largest
    effect, best power -- is choosing the estimand after seeing the outcome."""
    _matrix(tmp_path)
    with pytest.raises(ac.IncompleteMatrix, match="primary-load"):
        ac.analyse(tmp_path, "igsm_acc", 0.30, (0.17, 0.85), 0.01)


def test_analyzer_refuses_a_primary_load_outside_the_matrix(tmp_path):
    _matrix(tmp_path)
    with pytest.raises(ac.IncompleteMatrix, match="not in the matrix"):
        ac.analyse(tmp_path, "igsm_acc", 0.30, (0.17, 0.85), 0.01,
                   primary_load="middling")


def test_inference_is_blocked_by_seed_not_pooled_across_loads(tmp_path):
    """3 loads x 8 seeds is n=8 correlated blocks, not n=24 observations.
    The same initialisations and shard permutations recur at every load, so
    pooling shrinks the interval by roughly sqrt(3) for free."""
    _matrix(tmp_path, loads=("high", "mid", "low"), seeds=(0, 1, 2, 3))
    out = ac.analyse(tmp_path, "igsm_acc", 0.30, (0.17, 0.85), 0.01,
                     primary_load="high")
    assert out["statistic"]["n"] == 4, "one observation per seed, not per cell"
    assert out["pooled_across_loads_NOT_PRIMARY"]["n"] == 12
    assert "seed" in out["inference_unit"]
    assert "NOT" in "".join(out["pooled_across_loads_NOT_PRIMARY"]["note"].upper())


def test_gates_are_evaluated_at_the_primary_load_not_averaged(tmp_path):
    """Loads carry deliberately different fact content, so a mean SUP storage
    describes no corpus that was trained on. Averaging can pass the burden
    gate while the primary load fails it."""
    root = tmp_path
    for s in (0, 1, 2):
        # High load is saturated; low load stores almost nothing.
        _write_cell(root, "high", s, "sup", recoverable_bits_per_param=0.02)
        _write_cell(root, "high", s, "factmask", igsm_acc=0.44,
                    recoverable_bits_per_param=0.001)
        _write_cell(root, "high", s, "randpos", igsm_acc=0.41,
                    recoverable_bits_per_param=0.02)
        _write_cell(root, "low", s, "sup", recoverable_bits_per_param=0.90)
        _write_cell(root, "low", s, "factmask", igsm_acc=0.44,
                    recoverable_bits_per_param=0.01)
        _write_cell(root, "low", s, "randpos", igsm_acc=0.41,
                    recoverable_bits_per_param=0.88)
    # Mean SUP storage is 0.46 and would clear a 0.30 floor; the high load's
    # 0.02 does not.
    out = ac.analyse(root, "igsm_acc", 0.30, (0.17, 0.85), 0.01,
                     primary_load="high")
    assert "sup_carries_a_burden" in out["failed_validity_gates"]
    assert out["verdict"] == "invalid"


def test_dose_response_shape_is_reported(tmp_path):
    """THEORY-CAPACITY.md calls this the only analysis that speaks to capacity
    rather than loss composition, and nothing ran it until now."""
    _matrix(tmp_path, loads=("high", "mid", "low"), seeds=(0, 1))
    out = ac.analyse(tmp_path, "igsm_acc", 0.30, (0.17, 0.85), 0.01,
                     primary_load="high")
    dr = out["dose_response"]
    assert dr["n_loads"] == 3
    assert "signature" in dr
    # All three loads carry the same synthetic effect, so this is the flat case.
    assert "flat" in dr["signature"]


def test_a_flat_nonzero_dose_response_is_called_a_confound(tmp_path):
    """The signature that matters most: an advantage indifferent to fact load
    cannot be capacity reallocation, however significant it is."""
    from theory import capacity
    sig = capacity.dose_response_signature({0.3: 0.05, 1.0: 0.05, 2.0: 0.05},
                                           floor=0.01)
    assert "CONFOUND" in sig


def test_negligible_effects_are_flat_at_zero_not_rising(tmp_path):
    """The old form tested `peak < tol * peak`, true only for negative peak,
    so it could never fire: effects three orders below the noise floor were
    still classified 'consistent with capacity reallocation'."""
    from theory import capacity
    tiny = {0.3: 1e-6, 1.0: 2e-6, 2.0: 3e-6}
    assert "flat at zero" in capacity.dose_response_signature(tiny, floor=0.01)
    # The same shape above the floor is a real rising trend.
    big = {0.3: 0.01, 1.0: 0.05, 2.0: 0.09}
    assert "rising" in capacity.dose_response_signature(big, floor=0.001)
