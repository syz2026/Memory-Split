import math

import pytest

from theory import capacity as C

BPE = 52.96
P40 = 40_560_000


def test_capacity_curve_matches_the_cited_anchors():
    assert C.bits_per_param_at(1000) == pytest.approx(2.0, abs=1e-9)
    assert C.bits_per_param_at(100) == pytest.approx(1.0, abs=1e-9)


def test_capacity_is_clamped_not_extrapolated():
    # Extrapolating past the top anchor would invent headroom the model does
    # not have, which would make every over-loaded design look critical.
    assert C.bits_per_param_at(100_000) == pytest.approx(2.0)
    assert C.bits_per_param_at(1) == pytest.approx(0.0, abs=1e-9)
    assert C.bits_per_param_at(0) == 0.0


def test_capacity_is_monotone_in_exposures():
    vals = [C.bits_per_param_at(e) for e in (10, 50, 100, 300, 1000)]
    assert vals == sorted(vals)


def test_the_built_corpus_demands_two_bits_per_param():
    # 1,531,800 entities was chosen to hit the Allen-Zhu & Li saturation
    # figure exactly. This pins that intent so a corpus resize is visible.
    load = C.Load(1.0, 1_531_800, 100, BPE, P40)
    assert load.demand_bits_per_param == pytest.approx(2.0, abs=1e-3)


def test_the_headline_dose_is_over_loaded_not_critical():
    # The 2 bits/param target is a 1000-exposure result. Built at 100
    # exposures, achievable capacity is half that, so the load is 2x
    # over-saturated. This is the central finding of the capacity audit.
    load = C.Load(1.0, 1_531_800, 100, BPE, P40)
    assert load.ratio == pytest.approx(2.0, abs=0.01)
    assert load.regime == "over"


def test_ladder_holds_document_count_constant():
    loads = C.ladder(1_531_800, 100, (1.0, 0.5, 0.25), BPE, P40)
    docs = {l.n_entities * l.exposures for l in loads}
    assert len(docs) == 1, f"token budget differs across doses: {docs}"


def test_ladder_brackets_the_critical_point():
    a = C.audit(1_531_800, 100, (1.0, 0.5, 0.25), BPE, P40)
    assert a.brackets_critical, "inverted-U is untestable if no dose crosses 1"


def test_plateau_onset_is_the_middle_dose_not_the_highest():
    a = C.audit(1_531_800, 100, (1.0, 0.5, 0.25), BPE, P40)
    assert a.plateau_onset.fraction == 0.5
    assert a.has_dose_below_and_above
    assert any("plateau" in n for n in a.report()["notes"])


def test_flat_nonzero_dose_response_is_read_as_a_confound():
    # The discriminator that matters: a capacity story cannot be indifferent
    # to how many facts there are.
    sig = C.dose_response_signature({0.31: 0.040, 0.77: 0.041, 2.0: 0.039})
    assert "CONFOUND" in sig


def test_rising_then_saturating_reads_as_reallocation():
    sig = C.dose_response_signature({0.31: 0.010, 0.77: 0.038, 2.0: 0.041})
    assert "capacity reallocation" in sig


def test_decline_at_the_top_dose_reads_as_abandonment():
    sig = C.dose_response_signature({0.31: 0.010, 0.77: 0.045, 2.0: 0.012})
    assert "abandonment" in sig


def test_audit_flags_a_ladder_that_misses_the_peak():
    a = C.audit(50_000, 100, (1.0, 0.5), BPE, P40)
    assert not a.brackets_critical
    assert any("entirely below" in n for n in a.notes)


def test_entities_for_ratio_round_trips():
    n = C.entities_for_ratio(1.0, 100, BPE, P40)
    assert C.Load(1.0, n, 100, BPE, P40).ratio == pytest.approx(1.0, abs=1e-3)


def test_fixed_doc_solver_hits_the_critical_point():
    docs = 1_531_800 * 100
    n = C.entities_for_ratio_at_fixed_docs(1.0, docs, BPE, P40)
    e = docs // n
    assert C.Load(1.0, n, e, BPE, P40).ratio == pytest.approx(1.0, abs=0.02)


def test_fixed_doc_solver_is_monotone_in_target_ratio():
    docs = 1_531_800 * 100
    ns = [C.entities_for_ratio_at_fixed_docs(r, docs, BPE, P40)
          for r in (0.5, 1.0, 2.0)]
    assert ns == sorted(ns)


def test_three_loads_do_not_fit_a_150gb_scratch_budget():
    per_load = C.bytes_on_disk(16_400_000_000)
    assert per_load / 1e9 == pytest.approx(65.6, abs=0.1)
    assert 3 * per_load / 1e9 > 150, "if this passes, serial builds are unnecessary"


def test_endpoint_at_floor_is_not_responsive():
    # Stage A's STOP. An endpoint at the majority baseline cannot show
    # crowding, and no number of seeds repairs that.
    assert not C.responsive(0.071, floor=0.071)
    assert not C.responsive(0.080, floor=0.071)
    assert C.responsive(0.30, floor=0.071)


def test_endpoint_at_ceiling_is_not_responsive():
    assert not C.responsive(0.99, floor=0.071)


def test_split_cannot_beat_its_own_nofact_ceiling():
    assert C.effect_is_bounded_by_nofact(0.03, 0.05)
    assert C.effect_is_bounded_by_nofact(0.05, 0.05)
    assert not C.effect_is_bounded_by_nofact(0.07, 0.05)


def test_observability_names_the_binding_constraint():
    # Responsiveness fails first: this is the live Stage A situation.
    o = C.observability(0.071, 0.071, ratio=1.0, expected_effect=0.05, mde=0.02)
    assert not o["observable"]
    assert o["binding_constraint"] == "responsiveness"
    assert "nothing was learned" in o["reason"]


def test_observability_passes_when_all_three_hold():
    o = C.observability(0.30, 0.071, ratio=1.0, expected_effect=0.05, mde=0.02)
    assert o["observable"]
    assert o["binding_constraint"] is None


def test_observability_catches_underpowered_designs():
    o = C.observability(0.30, 0.071, ratio=1.0, expected_effect=0.01, mde=0.02)
    assert o["binding_constraint"] == "power"


def test_observability_catches_a_non_binding_load():
    o = C.observability(0.30, 0.071, ratio=0.05, expected_effect=0.05, mde=0.02)
    assert o["binding_constraint"] == "crowding"
