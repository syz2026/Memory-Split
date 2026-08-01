"""The staged pilot and the GO / NO-GO gate."""

import importlib.util
from pathlib import Path

import pytest
import yaml

OPS = Path(__file__).resolve().parents[1] / "ops" / "crowding"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, OPS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pilot = _load("pilot")
ap = _load("analyze_pilot")


# ------------------------------------------------------------ stage configs


def test_stage_a_crosses_both_architectures():
    cfgs = pilot.stage_a_configs("corpora/a", 800_000_000, 1e-3, 1.0)
    assert {c["model"] for c in cfgs} == {"d40m", "d40m_std"}
    assert all(c["arm"] == "sup" for c in cfgs)


def test_stage_a_covers_every_difficulty_cell():
    """The cells live inside one corpus and are separated at evaluation, so a
    single run measures the whole ladder."""
    mods = {m for m, _, _ in pilot.DIFFICULTY_CELLS}
    bands = {(lo, hi) for _, lo, hi in pilot.DIFFICULTY_CELLS}
    assert mods == {23, 11, 7}
    assert bands == {(1, 4), (5, 8)}


def test_stage_b_pairs_sup_against_nofact():
    cfgs = pilot.stage_b_configs("d40m", "corpora/fact", "corpora/nofact",
                                 5_737_000_000, 1.5e-3, 1.0)
    arms = {c["arm"] for c in cfgs}
    assert arms == {"sup", "nofact"}
    sup = [c for c in cfgs if c["arm"] == "sup"]
    nof = [c for c in cfgs if c["arm"] == "nofact"]
    assert len(sup) == len(nof)
    # Identical everywhere except which stream they read.
    a = {k: v for k, v in sup[0].items() if k not in ("run_id", "arm", "out_rel", "train_bin")}
    b = {k: v for k, v in nof[0].items() if k not in ("run_id", "arm", "out_rel", "train_bin")}
    assert a == b


def test_stage_b_both_arms_probe_the_fact_positions():
    cfgs = pilot.stage_b_configs("d40m", "corpora/fact", "corpora/nofact",
                                 1_000, 1e-3, 1.0)
    assert {c["probe_mask"] for c in cfgs} == {"corpora/fact/factmask.bin"}


def test_stage_c_is_three_complete_triplets():
    cfgs = pilot.stage_c_configs("d40m", "corpora/high", 5_737_000_000, 1.5e-3, 1.0)
    assert len(cfgs) == 9
    assert {c["arm"] for c in cfgs} == {"sup", "factmask", "randpos"}
    assert len({c["seed"] for c in cfgs}) == 3


def test_pilot_seeds_are_disjoint_from_plausible_confirm_seeds():
    """Pilot seeds are burned: they may set the design but not appear in the
    confirmatory matrix."""
    assert all(s >= 9000 for s in pilot.PILOT_SEEDS)
    assert not set(pilot.PILOT_SEEDS) & set(range(0, 100))


def test_configs_round_trip_as_yaml(tmp_path):
    cfgs = pilot.stage_a_configs("corpora/a", 1000, 1e-3, 1.0)
    pilot.write(cfgs, tmp_path)
    for c in cfgs:
        got = yaml.safe_load((tmp_path / f"{c['run_id']}.yaml").read_text())
        assert got == c


# ------------------------------------------------------------ stage A analysis


def _cell(acc, majority=0.072, by_op=None, curve=None, **kw):
    return {"model": "d40m", "mod": 23, "op_lo": 1, "op_hi": 4, "acc": acc,
            "majority_rate": majority, "acc_by_op": by_op or {},
            "acc_over_training": curve or [], **kw}


def test_stage_a_rejects_a_floored_ladder():
    out = ap.stage_a([_cell(0.075), _cell(0.08)])
    assert out["any_cell_clears_floor"] is False
    assert out["chosen"] is None


def test_stage_a_picks_the_largest_lift():
    out = ap.stage_a([_cell(0.30), _cell(0.50), _cell(0.20)])
    assert out["chosen"]["acc"] == 0.50


def test_stage_a_flags_non_monotone_difficulty():
    """Accuracy that does not fall with op is not dependency tracing."""
    good = ap.stage_a([_cell(0.4, by_op={"1": 0.6, "2": 0.4, "3": 0.2})])
    bad = ap.stage_a([_cell(0.4, by_op={"1": 0.2, "2": 0.4, "3": 0.6})])
    assert good["chosen"]["monotone_in_op"] is True
    assert bad["chosen"]["monotone_in_op"] is False


def test_stage_a_flags_no_learning_over_training():
    out = ap.stage_a([_cell(0.4, curve=[0.5, 0.4])])
    assert out["chosen"]["monotone_over_training"] is False


# ------------------------------------------------------------ stage B / C


def test_delta_is_the_reasoning_cost_of_the_fact_load():
    out = ap.stage_b(sup_acc=[0.40, 0.42], nofact_acc=[0.45, 0.46],
                     bits_at_e=0.40, bits_at_3e=0.44)
    assert out["delta"] == pytest.approx(0.045, abs=1e-6)


def test_exposure_saturation_detects_an_exposure_limited_regime():
    limited = ap.stage_b([0.4], [0.4], bits_at_e=0.10, bits_at_3e=0.30)
    saturated = ap.stage_b([0.4], [0.4], bits_at_e=0.40, bits_at_3e=0.44)
    assert limited["parameter_limited"] is False
    assert saturated["parameter_limited"] is True


def test_stage_c_reports_mde_across_candidate_n():
    out = ap.stage_c([1.0, 1.4, 0.8], {"sup": 0.4, "factmask": 0.01}, 0.1, 8)
    assert out["paired_sd"] > 0
    assert out["mde_at_n"][16] < out["mde_at_n"][6]
    assert out["leakage_factmask_over_sup"] == pytest.approx(0.025)


# ------------------------------------------------------------ the gate


def _go_inputs():
    a = ap.stage_a([_cell(0.42, by_op={"1": 0.6, "2": 0.4}, curve=[0.1, 0.42])])
    b = ap.stage_b([0.40, 0.42], [0.48, 0.50], 0.40, 0.44)
    c = ap.stage_c([0.05, 0.06, 0.055], {"sup": 0.4, "factmask": 0.01}, 0.1, 8)
    return a, b, c


def test_a_healthy_pilot_gives_go():
    a, b, c = _go_inputs()
    g = ap.gate(a, b, c, min_interesting_effect=0.02,
                igsm_band=(0.17, 0.85), n_confirm=8)
    assert g["decision"] == "GO", g["checks"]


def test_exposure_limited_pilot_gives_no_go():
    a, _, c = _go_inputs()
    b = ap.stage_b([0.40, 0.42], [0.48, 0.50], 0.10, 0.40)
    g = ap.gate(a, b, c, 0.02, (0.17, 0.85), 8)
    assert g["decision"] == "NO-GO"
    assert not next(x for x in g["checks"]
                    if x["name"] == "parameter_limited_not_exposure_limited")["ok"]


def test_a_delta_smaller_than_the_effect_of_interest_gives_no_go():
    """If the facts cost the supervised arm less than the effect we care
    about, no masking scheme can recover it and the matrix is futile."""
    a, _, c = _go_inputs()
    b = ap.stage_b([0.40, 0.42], [0.404, 0.424], 0.40, 0.44)
    g = ap.gate(a, b, c, 0.02, (0.17, 0.85), 8)
    assert g["decision"] == "NO-GO"
    assert not next(x for x in g["checks"]
                    if x["name"] == "effect_has_room_to_exist")["ok"]


def test_an_underpowered_design_gives_no_go():
    a, b, _ = _go_inputs()
    c = ap.stage_c([0.05, -0.9, 1.2], {"sup": 0.4, "factmask": 0.01}, 0.1, 8)
    g = ap.gate(a, b, c, 0.02, (0.17, 0.85), 8)
    assert g["decision"] == "NO-GO"
    assert not next(x for x in g["checks"] if x["name"] == "design_is_powered")["ok"]


def test_a_floored_endpoint_gives_no_go():
    a = ap.stage_a([_cell(0.075)])
    _, b, c = _go_inputs()
    g = ap.gate(a, b, c, 0.02, (0.17, 0.85), 8)
    assert g["decision"] == "NO-GO"


def test_no_go_points_at_a_paper_written_in_advance():
    a = ap.stage_a([_cell(0.075)])
    _, b, c = _go_inputs()
    g = ap.gate(a, b, c, 0.02, (0.17, 0.85), 8)
    assert "NO-GO-PAPER.md" in g["on_no_go"]
