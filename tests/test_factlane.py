"""The fact lane and the RANDPOS control sidecar."""

import random

import numpy as np
import pytest

from corpusgen import bios, factlane, randpos
from corpusgen.records import ATTRIBUTES
from train.tokenizer import get_tok

TOK = get_tok()


# ------------------------------------------------------------ sizing


def test_measured_constants_match_the_generator():
    """plan() is only as good as these two numbers."""
    recs = bios.generate_records(300, 0)
    g = factlane.measure_geometry(recs, TOK)
    assert abs(g["tokens_per_doc"] - factlane.TOKENS_PER_DOC) / factlane.TOKENS_PER_DOC < 0.03
    assert 0.20 < g["masked_fraction"] < 0.30


def test_plan_is_the_exact_tradeoff():
    p = factlane.plan(0.5, 40_560_000, 100)
    assert p["n_entities"] == round(0.5 * 40_560_000 / factlane.BITS_PER_ENTITY)
    assert p["fact_tokens"] == round(
        p["n_entities"] * 100 * factlane.TOKENS_PER_DOC
    )
    assert abs(p["bits_total"] / 40_560_000 - 0.5) < 1e-6


def test_plan_reports_bits_per_nonembedding_parameter_when_asked():
    p = factlane.plan(0.5, 40_560_000, 100, n_nonembed=21_243_264)
    assert p["bits_per_nonembed"] > p["bits_total"] / 40_560_000


def test_pool_entropy_is_52_96():
    r = factlane.occupancy_report(1000, {"d40m": 40_560_000})
    assert abs(r["log2_pool_check"] - 52.96) < 0.01
    assert abs(factlane.BITS_PER_ENTITY - 52.96) < 0.01


# ------------------------------------------------------------ the load axis


def test_load_variants_hold_document_count_fixed():
    """The whole point: loads differ ONLY in unique entropy. If document count
    moved, so would tokens, steps, schedule length and mask mass."""
    vs = factlane.load_variants(300_000, 100, (1.0, 0.3, 0.1))
    docs = {v["n_docs"] for v in vs}
    assert len(docs) == 1, f"document counts differ: {docs}"


def test_load_variants_scale_unique_entropy():
    vs = factlane.load_variants(382_900, 100, (1.0, 0.3))
    hi, lo = vs[0], vs[1]
    assert hi["n_entities"] > lo["n_entities"]
    assert lo["exposures"] > hi["exposures"]
    assert abs(hi["bits_total"] / lo["bits_total"] - 1 / 0.3) < 0.20


def test_choose_entity_count_keeps_the_dose_spacing_tight():
    """load_variants snaps to exact divisors of N*E. A sparse divisor lattice
    can throw a fraction far off -- 300,000 x 100 misses 0.3 by 13% -- so the
    builder picks a count whose divisors bracket every requested fraction."""
    fracs = (1.0, 0.3, 0.1)
    n = factlane.choose_entity_count(300_000, 100, fracs)
    vs = factlane.load_variants(n, 100, fracs)
    assert len({v["n_docs"] for v in vs}) == 1
    for v in vs:
        want = n * v["fraction"]
        assert abs(v["n_entities"] - want) / want < 0.02, v


def test_load_variants_decouple_load_from_exposure():
    """The confound that broke the original sweep: raising entity count at a
    fixed lane budget cut exposures per fact, so load and exposure moved
    together. Here exposures rise as entities fall, by construction."""
    vs = factlane.load_variants(100_000, 200, (1.0, 0.5, 0.25))
    for v in vs:
        assert v["n_entities"] * v["exposures"] == pytest.approx(
            vs[0]["n_docs"], rel=0.01
        )


# ------------------------------------------------------------ emission


def test_exposures_are_spread_not_massed():
    """Grouped rounds: one round is n_entities documents wide, so a fact's
    copies land evenly across the stream rather than in one block."""
    order = list(factlane.doc_order(5, 4))
    positions = [i for i, (e, _) in enumerate(order) if e == 0]
    gaps = np.diff(positions)
    assert len(positions) == 4
    assert (gaps == 5).all(), f"entity 0 not evenly spaced: {positions}"


def test_emit_yields_aligned_ids_and_mask():
    recs = bios.generate_records(4, 0)
    for ids, mask in factlane.emit(recs, 2, TOK):
        assert len(ids) == len(mask)
        assert ids.dtype == np.uint16 and mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})


def test_emit_produces_exactly_n_times_e_documents():
    recs = bios.generate_records(6, 0)
    assert sum(1 for _ in factlane.emit(recs, 3, TOK)) == 18


def test_verify_doc_passes_on_every_sampled_document():
    """Masked spans decode to exactly the six values, and no value leaks
    unmasked. Never asserted by the previous builder in production."""
    recs = bios.generate_records(60, 0)
    for rec in recs:
        for e in range(8):
            factlane.verify_doc(rec, e, TOK)


def test_verify_doc_catches_a_leaked_value():
    rec = bios.generate_records(1, 0)[0]
    tampered = type(rec)(
        entity_id=rec.entity_id,
        name=rec.name,
        attrs={**rec.attrs, "major": rec.name},  # value now equals the name
    )
    with pytest.raises(AssertionError):
        factlane.verify_doc(tampered, 0, TOK)


# ------------------------------------------------------------ RANDPOS


def _doc(entity=0, exposure=0):
    rec = bios.generate_records(4, 0)[entity]
    ids, mask = TOK.encode_segments(bios.render_bio_marked(rec, exposure), add_eot=True)
    return np.asarray(ids, dtype=np.uint16), np.asarray(mask, dtype=np.uint8)


def test_randpos_zeroes_the_same_number_of_targets():
    ids, fm = _doc()
    rp = randpos.build(ids, fm, random.Random(0))
    assert (rp == 0).sum() == (fm == 0).sum()


def test_randpos_never_overlaps_a_value_span():
    for e in range(6):
        ids, fm = _doc(exposure=e)
        rp = randpos.build(ids, fm, random.Random(e))
        assert not ((rp == 0) & (fm == 0)).any()


def test_randpos_matches_the_span_length_histogram():
    ids, fm = _doc()
    rp = randpos.build(ids, fm, random.Random(1))
    f = sorted(ln for _, ln in randpos.spans_of(fm))
    r = sorted(ln for _, ln in randpos.spans_of(rp))
    assert f == r, f"lengths differ: {f} vs {r}"


def test_randpos_matches_relative_position():
    """Count matching alone leaves position free."""
    ids, fm = _doc()
    rp = randpos.build(ids, fm, random.Random(2))
    rep = randpos.match_report(fm, rp)
    gap = abs(
        rep["mean_relative_position_fact"] - rep["mean_relative_position_randpos"]
    )
    assert gap < 0.20, rep


def test_randpos_is_deterministic_given_a_seed():
    ids, fm = _doc()
    a = randpos.build(ids, fm, random.Random(3))
    b = randpos.build(ids, fm, random.Random(3))
    assert np.array_equal(a, b)


def test_cue_window_overlap_is_measured_not_ignored():
    """Length, position and template-only placement leave a small feasible
    set, so spans drift onto the cue phrases that predict each value. That
    biases the control toward the treatment; it must be reported."""
    ids, fm = _doc()
    rp = randpos.build(ids, fm, random.Random(4))
    rep = randpos.match_report(fm, rp)
    assert "cue_window_overlap_frac" in rep
    assert 0.0 <= rep["cue_window_overlap_frac"] <= 1.0
    assert rep["median_distance_to_nearest_value"] is not None


def test_nll_matching_prefers_difficulty_matched_placements():
    """With a pilot NLL table, the chosen span should sit closer in mean
    difficulty to the value span than a difficulty-blind choice does."""
    ids, fm = _doc()
    rng_state = 5
    n = len(fm)
    nll = np.full(n, 0.2, dtype=np.float64)
    nll[fm == 0] = 2.85  # values are hard
    # One template region is made hard too, so a match exists.
    free = np.flatnonzero(fm == 1)
    nll[free[: max(1, len(free) // 3)]] = 2.80

    blind = randpos.build(ids, fm, random.Random(rng_state))
    matched = randpos.build(ids, fm, random.Random(rng_state), token_nll=nll)
    gap_blind = randpos.match_report(fm, blind, nll)["nll_relative_gap"]
    gap_matched = randpos.match_report(fm, matched, nll)["nll_relative_gap"]
    assert gap_matched <= gap_blind


def test_match_report_flags_an_unmatched_control_as_invalid():
    """If template positions cannot be matched within tolerance the control is
    empirically invalid and must say so, not be quietly gated away."""
    ids, fm = _doc()
    n = len(fm)
    nll = np.full(n, 0.05, dtype=np.float64)
    nll[fm == 0] = 5.0  # nothing in the template comes close
    rp = randpos.build(ids, fm, random.Random(6), token_nll=nll)
    rep = randpos.match_report(fm, rp, nll)
    assert rep["nll_within_tolerance"] is False
    assert rep["nll_relative_gap"] > randpos.NLL_TOLERANCE


def test_randpos_leaves_a_document_without_values_untouched():
    ids = np.arange(30, dtype=np.uint16)
    fm = np.ones(30, dtype=np.uint8)
    rp = randpos.build(ids, fm, random.Random(0))
    assert (rp == 1).all()
