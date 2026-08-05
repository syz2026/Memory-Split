"""The occupancy ladder reader.

Premise 1 of Memory Split — that arbitrary facts occupy parameters — has never
been tested directly by this project. Four generations went straight to premise
3 and measured noise around zero because nothing was ever stored. These tests
pin the four signatures the ladder can return, because the whole point is that
the reading is unambiguous before 1,495 GPU-hours are committed on the strength
of it.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "occupancy", ROOT / "ops" / "crowding" / "occupancy.py")
occ = importlib.util.module_from_spec(_SPEC)
sys.modules["occupancy"] = occ
_SPEC.loader.exec_module(occ)

N_PARAMS = 40_560_000
BPE = 52.96


def _rung(root: Path, exposures: int, entities: int, stored: float):
    d = root / f"occ_d40m_std_e{exposures}"
    (d / "evals").mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump({"run_id": d.name}))
    (d / "evals" / "summary.json").write_text(json.dumps({
        "n_entities": entities,
        "n_params": N_PARAMS,
        "recoverable_bits_per_param": stored,
        "igsm_acc": 0.03,
        "igsm_m1_nll": 2.5,
    }))


def _entities_for(demand: float) -> int:
    return int(demand * N_PARAMS / BPE)


def test_saturating_is_recognised(tmp_path):
    """Storage bends away from linear: capacity is binding and the crowding
    regime has been reached. This is the only signature that licenses spending
    the matrix budget."""
    _rung(tmp_path, 800, _entities_for(0.325), 0.300)   # near-full at low load
    _rung(tmp_path, 400, _entities_for(0.651), 0.520)
    _rung(tmp_path, 200, _entities_for(1.301), 0.620)   # flattening
    r = occ.report(tmp_path)
    assert r["signature"] == "saturating", r
    assert r["bend_vs_linear"] > 0.20
    assert "worth running" in r["reading"]


def test_abandonment_is_recognised(tmp_path):
    """Storage peaks then falls. The model declines to learn rather than
    compressing, so there is no burden to relieve and the arm contrast is not
    worth running -- the shape the null draft claims."""
    _rung(tmp_path, 800, _entities_for(0.325), 0.300)
    _rung(tmp_path, 400, _entities_for(0.651), 0.180)
    _rung(tmp_path, 200, _entities_for(1.301), 0.050)
    r = occ.report(tmp_path)
    assert r["signature"] == "abandonment", r
    assert "cannot occur" in r["reading"]


def test_linear_is_recognised(tmp_path):
    """No bend: nothing binds. Raise the load, not the seed count."""
    _rung(tmp_path, 800, _entities_for(0.325), 0.163)
    _rung(tmp_path, 400, _entities_for(0.651), 0.325)
    _rung(tmp_path, 200, _entities_for(1.301), 0.650)
    r = occ.report(tmp_path)
    assert r["signature"] == "linear", r


def test_inert_is_not_mistaken_for_a_shape(tmp_path):
    """Every corpus this project built before 2026-08-02 was inert. A ladder of
    zeros has no shape and must not be read as one."""
    _rung(tmp_path, 800, _entities_for(0.325), 0.001)
    _rung(tmp_path, 400, _entities_for(0.651), 0.002)
    _rung(tmp_path, 200, _entities_for(1.301), -0.003)
    r = occ.report(tmp_path)
    assert r["signature"] == "inert", r
    assert "below the storage threshold" in r["reading"]


def test_one_rung_is_not_a_ladder(tmp_path):
    _rung(tmp_path, 200, _entities_for(1.301), 0.62)
    r = occ.report(tmp_path)
    assert r["signature"] == "insufficient"


def test_rungs_carry_their_capacity_accounting(tmp_path):
    """Each rung must report demand, achievable capacity and F/C, since the
    signature is only interpretable against where the rung sits."""
    _rung(tmp_path, 800, _entities_for(0.325), 0.30)
    _rung(tmp_path, 200, _entities_for(1.301), 0.62)
    r = occ.report(tmp_path)
    assert r["n_rungs"] == 2
    lo, hi = r["rungs"]
    assert lo["demand_bits_per_param"] < hi["demand_bits_per_param"]
    assert lo["exposures"] == 800 and hi["exposures"] == 200
    # Achievable capacity RISES with exposures, so F/C falls faster than demand.
    assert lo["achievable_bits_per_param"] > hi["achievable_bits_per_param"]
    assert hi["ratio_F_over_C"] == pytest.approx(1.0, abs=0.01)


def test_unscored_runs_are_skipped_not_guessed(tmp_path):
    """A run whose eval has not landed must not silently become a rung."""
    _rung(tmp_path, 800, _entities_for(0.325), 0.30)
    d = tmp_path / "occ_d40m_std_e400"
    (d / "evals").mkdir(parents=True)
    (d / "config.yaml").write_text("run_id: occ_d40m_std_e400\n")
    r = occ.report(tmp_path)
    assert r["n_rungs"] == 1
    assert r["signature"] == "insufficient"


def test_the_denominator_and_bracketing_caveats_are_stated(tmp_path):
    """Two things a reader must not have to infer: occupancy is against total
    parameters, and the ladder cannot bracket F/C = 1 from above."""
    _rung(tmp_path, 800, _entities_for(0.325), 0.30)
    _rung(tmp_path, 200, _entities_for(1.301), 0.62)
    r = occ.report(tmp_path)
    assert "TOTAL parameters" in r["denominator_note"]
    assert "81.2%" in r["denominator_note"]
    assert "from below only" in r["ladder_note"]
