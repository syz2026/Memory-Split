import pytest

from theory import endpoint as E

# Measured, outputs/pilot/stage_a_report.json.
STAGE_A_STD = {1: 0.05397, 2: 0.02895, 3: 0.07216, 4: 0.05000}
STAGE_A_REC = {1: 0.03977, 2: 0.05789, 3: 0.08505, 4: 0.06316}
STAGE_A_N = {1: 352, 2: 380, 3: 388, 4: 380}
CHANCE_23 = 1.0 / 23
MAJORITY = 0.07133


def test_stage_a_step_count():
    assert E.steps_for(800_000_000) == 1525


def test_stage_a_was_undertrained_by_an_order_of_magnitude():
    v = E.step_budget_verdict(E.steps_for(800_000_000), warmup=300)
    assert not v["plausibly_sufficient"]
    assert v["shortfall_vs_low_end"] >= 6
    assert v["warmup_fraction"] > 0.19


def test_the_full_corpus_reaches_a_plausible_budget():
    v = E.step_budget_verdict(E.steps_for(16_400_000_000), warmup=300)
    assert v["plausibly_sufficient"]
    assert v["warmup_fraction"] < 0.02


def test_the_operation_table_is_far_too_small_to_be_capacity_limited():
    # ~38 kbit against 40.56M parameters. Capacity is not the constraint,
    # by six orders of magnitude.
    assert E.table_bits(23, 4) < 50_000
    assert not E.capacity_limited(23, 4, 40_560_000)


def test_both_stage_a_cells_read_as_atomic_operation_not_learned():
    for prof in (STAGE_A_STD, STAGE_A_REC):
        p = E.OpProfile(prof, CHANCE_23, MAJORITY, STAGE_A_N)
        assert "ATOMIC OPERATION" in p.signature()


def test_no_skill_baseline_is_the_majority_rate_not_uniform_chance():
    # Always emitting the modal answer earns 7.1% for free, so reading these
    # cells against 1/23 = 4.3% turns noise into apparent signal.
    p = E.OpProfile(STAGE_A_REC, CHANCE_23, MAJORITY, STAGE_A_N)
    assert p.no_skill == pytest.approx(MAJORITY)
    # op=3 at 8.5% is the highest cell and still within noise of the baseline.
    assert p.near_baseline(3)


def test_tolerance_scales_with_cell_size():
    tight = E.OpProfile(STAGE_A_REC, CHANCE_23, MAJORITY, {o: 100_000 for o in STAGE_A_REC})
    # With 100k per cell the same 8.5% would be a real departure.
    assert not tight.near_baseline(3)


def test_stage_a_falsifies_the_compounding_explanation():
    # Compounding predicts high accuracy at op=1. It is at the baseline.
    p = E.OpProfile(STAGE_A_STD, CHANCE_23, MAJORITY, STAGE_A_N)
    assert p.near_baseline(1)


def test_compounding_is_recognised_when_it_is_real():
    p = E.OpProfile({1: 0.80, 2: 0.64, 3: 0.51, 4: 0.41}, CHANCE_23, MAJORITY)
    sig = p.signature()
    assert "COMPOUNDING" in sig
    # p**k with p=0.8 should invert back to about 0.8 at every op.
    implied = p.implied_per_step
    assert all(0.78 <= v <= 0.82 for v in implied.values())


def test_flat_and_high_is_flagged_as_suspicious():
    p = E.OpProfile({1: 0.70, 2: 0.70, 3: 0.71, 4: 0.70}, CHANCE_23, MAJORITY)
    assert "leakage" in p.signature()


def test_the_live_lr_probe_is_confounded_for_the_endpoint_question():
    # 1,500 steps at MOD=23, against Stage A's 1,525 steps at MOD=23.
    r = E.probe_is_confounded(1_500, 1_525, 23, 23)
    assert r["confounded"]
    assert r["same_step_budget"] and r["same_modulus"]
    assert "CONFOUNDED" in r["reading"]
    assert "choosing a learning rate by training loss" in r["reading"]


def test_a_probe_with_a_longer_budget_is_identifiable():
    r = E.probe_is_confounded(30_000, 1_525, 23, 23)
    assert not r["confounded"]


def test_a_probe_at_an_easier_modulus_is_identifiable():
    r = E.probe_is_confounded(1_500, 1_525, 7, 23)
    assert not r["confounded"]


def test_difficulty_ladder_descends_and_reports_chance():
    rungs = E.difficulty_ladder()
    assert [r["mod"] for r in rungs] == [23, 11, 7, 5]
    assert rungs[0]["chance"] == pytest.approx(1 / 23, abs=1e-3)
    assert rungs[-1]["chance"] == pytest.approx(0.2, abs=1e-3)
    # Easier rungs are strictly cheaper to memorise.
    bits = [r["table_bits"] for r in rungs]
    assert bits == sorted(bits, reverse=True)
