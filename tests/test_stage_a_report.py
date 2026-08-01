"""Stage A's verdict: the cheapest place this project can end."""

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "report_stage_a",
    Path(__file__).resolve().parents[1] / "ops" / "crowding" / "report_stage_a.py",
)
rs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rs)


def _cell(run, acc, majority=0.072, by_op=None, **kw):
    return {"run": run, "model": "d40m", "mod": 23, "acc": acc,
            "majority_rate": majority, "acc_by_op": by_op or {},
            "ood_acc": 0.1, "deduction_by_class": {}, "clip_frac": 0.0,
            "clip_ratio": 1.0, "gate0": 10.8, "final_loss": 3.0,
            "n_params": 40_560_000, **kw}


def test_a_floored_ladder_stops_the_project():
    r = rs.report([_cell("a", 0.075), _cell("b", 0.09)])
    assert r["decision"] == "STOP"
    assert "NO-GO" in r["rationale"]


def test_a_clearing_cell_continues_and_is_chosen():
    r = rs.report([_cell("a", 0.30), _cell("b", 0.45)])
    assert r["decision"] == "CONTINUE"
    assert r["chosen"]["run"] == "b"


def test_lift_is_measured_against_the_empirical_baseline_not_one_over_23():
    """1/23 = 4.35%, but times overproduces zero so the real floor is ~7.2%.
    A cell at 10% clears 1/23 by 5.7 points and the real floor by only 2.8."""
    r = rs.report([_cell("a", 0.10, majority=0.072)])
    assert r["cells"][0]["lift"] < 0.05
    assert r["decision"] == "STOP"


def test_monotone_in_op_is_reported():
    good = rs.report([_cell("a", 0.4, by_op={"1": 0.6, "2": 0.4, "3": 0.2})])
    bad = rs.report([_cell("a", 0.4, by_op={"1": 0.2, "2": 0.4, "3": 0.6})])
    assert good["cells"][0]["monotone_in_op"] is True
    assert bad["cells"][0]["monotone_in_op"] is False


def test_collect_skips_runs_without_evaluations(tmp_path):
    (tmp_path / "unevaluated").mkdir()
    d = tmp_path / "done" / "evals"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"igsm_acc": 0.4, "model": "d40m"}))
    assert [c["run"] for c in rs.collect(tmp_path)] == ["done"]
