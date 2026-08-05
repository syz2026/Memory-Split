"""The corpus builder, rehearsed end to end at small scale."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from corpusgen import randpos

_SPEC = importlib.util.spec_from_file_location(
    "build_corpus",
    Path(__file__).resolve().parents[1] / "ops" / "crowding" / "build_corpus.py",
)
bc = importlib.util.module_from_spec(_SPEC)
# Register before exec: multiprocessing pickles the worker function by
# qualified name, so the module has to be resolvable. Running the file as a
# script (which production does) registers it as __main__ automatically.
sys.modules["build_corpus"] = bc
_SPEC.loader.exec_module(bc)

SHARES = {"fact": 0.5, "igsm": 0.3, "deduction": 0.1, "bed": 0.1}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    # Sized so the fact lane can actually fill its 50% share:
    # 60 entities x 25 exposures x ~75 tokens ~= 112k against a 100k budget.
    out = tmp_path_factory.mktemp("corpus")
    man = bc.build(out, n_entities=60, exposures=25, shares=SHARES,
                   total_tokens=200_000, seed=0)
    return out, man


def test_exact_token_count(built):
    out, man = built
    assert man["n_tokens"] == 200_000
    assert (out / "targets.bin").stat().st_size == 200_000 * 2


def test_all_three_files_are_aligned(built):
    out, _ = built
    n = (out / "targets.bin").stat().st_size // 2
    assert (out / "factmask.bin").stat().st_size == n
    assert (out / "randpos.bin").stat().st_size == n


def test_verify_passes(built):
    out, _ = built
    assert bc.verify(out, expect_tokens=200_000, allow_synthetic_bed=True,
                     require_randpos_nll=False) == []


def test_mask_mass_is_equal_between_the_arms(built):
    out, man = built
    a = man["mask_audit"]
    assert a["mass_matched"], a
    assert a["factmask_zeros"] > 0


def test_randpos_never_overlaps_a_fact_value(built):
    _, man = built
    assert man["mask_audit"]["overlap"] == 0


def test_both_sidecars_are_strict_subsets_of_full_supervision(built):
    """SUP is implicit all-ones, so any value outside {0,1} is corruption and
    any masked position is one SUP supervises."""
    out, _ = built
    for name in ("factmask.bin", "randpos.bin"):
        arr = np.fromfile(out / name, dtype=np.uint8)
        assert set(np.unique(arr)).issubset({0, 1})


def test_every_lane_is_present_and_near_its_share(built):
    _, man = built
    for lane, share in SHARES.items():
        got = man["lane_tokens"][lane] / man["n_tokens"]
        assert got > 0, f"lane {lane} emitted nothing"
        assert abs(got - share) < 0.06, f"{lane}: {got:.3f} vs {share}"


def test_lanes_are_interleaved_not_blocked(built):
    """A blocked layout is what put the previous corpus's reasoning lane in
    the last 12.8% of training and made the endpoint threshold noise."""
    out, _ = built
    fact = np.fromfile(out / "factmask.bin", dtype=np.uint8)
    masked = np.flatnonzero(fact == 0)
    assert masked.size > 0
    # Fact documents must appear in every quarter of the stream.
    quarters = np.array_split(np.arange(len(fact)), 4)
    for i, q in enumerate(quarters):
        assert (fact[q] == 0).any(), f"no fact tokens in quarter {i}"


def test_manifest_records_occupancy_and_hashes(built):
    out, man = built
    assert man["bits_total"] == pytest.approx(60 * 52.96, rel=1e-6)
    assert set(man["bits_per_param"]) == {"d8m", "d40m", "d160m"}
    assert set(man["sha256"]) == {"targets.bin", "factmask.bin", "randpos.bin"}
    for h in man["sha256"].values():
        assert len(h) == 64
    assert json.loads((out / "manifest.json").read_text())["n_tokens"] == 200_000


def test_manifest_flags_a_synthetic_bed(built):
    _, man = built
    assert "SYNTHETIC" in man["bed"], "a rehearsal bed must not look pinned"


def test_builds_are_deterministic(tmp_path):
    a = bc.build(tmp_path / "a", 30, 16, SHARES, 60_000, seed=3)
    b = bc.build(tmp_path / "b", 30, 16, SHARES, 60_000, seed=3)
    assert a["sha256"] == b["sha256"]


def test_a_different_seed_changes_the_stream(tmp_path):
    a = bc.build(tmp_path / "a", 30, 16, SHARES, 60_000, seed=3)
    b = bc.build(tmp_path / "b", 30, 16, SHARES, 60_000, seed=4)
    assert a["sha256"]["targets.bin"] != b["sha256"]["targets.bin"]


# ---------------------------------------------------------- verify fails closed


def test_verify_catches_a_short_corpus(built):
    out, _ = built
    fails = bc.verify(out, expect_tokens=999_999)
    assert any("tokens" in f for f in fails)


def test_verify_catches_a_truncated_sidecar(tmp_path):
    out = tmp_path / "c"
    bc.build(out, 20, 14, SHARES, 40_000, seed=1)
    data = np.fromfile(out / "randpos.bin", dtype=np.uint8)[:-100]
    data.tofile(out / "randpos.bin")
    fails = bc.verify(out, expect_tokens=40_000)
    assert any("randpos sidecar" in f for f in fails)


def test_verify_catches_an_inert_sidecar(tmp_path):
    out = tmp_path / "d"
    bc.build(out, 20, 14, SHARES, 40_000, seed=1)
    n = (out / "targets.bin").stat().st_size // 2
    np.ones(n, dtype=np.uint8).tofile(out / "factmask.bin")
    fails = bc.verify(out, expect_tokens=40_000)
    assert any("masks nothing" in f for f in fails)


def test_verify_catches_unequal_mask_mass(tmp_path):
    out = tmp_path / "e"
    bc.build(out, 20, 14, SHARES, 40_000, seed=1)
    rp = np.fromfile(out / "randpos.bin", dtype=np.uint8)
    rp[np.flatnonzero(rp == 0)[:5]] = 1  # remove five masked targets
    rp.tofile(out / "randpos.bin")
    fails = bc.verify(out, expect_tokens=40_000)
    assert any("mask mass differs" in f for f in fails)


def test_an_underfilled_fact_lane_is_refused_up_front(tmp_path):
    """Stage A's defect, now a build failure.

    20,000 entities at 20 exposures emit ~30M tokens against a 50% share of
    800M, so the bed absorbed 46.3% of the corpus and the fact lane realised
    3.75%. Nothing failed. The build must refuse before writing 65 GB, and the
    message must name the fix.
    """
    with pytest.raises(bc.LaneUnderfilled) as e:
        bc.build(tmp_path / "small", n_entities=10, exposures=4, shares=SHARES,
                 total_tokens=200_000, seed=0)
    msg = str(e.value)
    assert "different experiment" in msg
    assert "exposures" in msg, "the message must say how to fix it"


def test_an_exhausted_lane_still_gives_its_budget_to_the_bed(tmp_path):
    """The reallocation itself is correct and stays: a finite lane must not
    shorten the corpus, because every arm and load shares one step count.
    What changed is that it can no longer happen unnoticed."""
    out = tmp_path / "small"
    man = bc.build(out, n_entities=10, exposures=4, shares=SHARES,
                   total_tokens=200_000, seed=0, allow_short_fact_lane=True)
    assert man["n_tokens"] == 200_000
    assert man["lane_tokens"]["fact"] < SHARES["fact"] * 200_000
    assert man["lane_tokens"]["bed"] > SHARES["bed"] * 200_000
    # Opting out keeps rehearsals working ...
    assert bc.verify(out, expect_tokens=200_000, allow_synthetic_bed=True,
                     allow_share_drift=True, require_randpos_nll=False) == []
    # ... but the default names the substitution.
    fails = bc.verify(out, expect_tokens=200_000, allow_synthetic_bed=True,
                      require_randpos_nll=False)
    assert any("lane 'fact' realised" in f for f in fails)
    assert any("lane 'bed' realised" in f for f in fails)


def test_manifest_records_requested_shares_not_just_realised(tmp_path):
    """`lane_budgets` is mutated by the reallocation, so it cannot answer
    'what was asked for'. Without that, the substitution is invisible."""
    out = tmp_path / "small"
    man = bc.build(out, n_entities=10, exposures=4, shares=SHARES,
                   total_tokens=200_000, seed=0, allow_short_fact_lane=True)
    assert man["requested_shares"]["fact"] == SHARES["fact"]
    assert man["realised_shares"]["fact"] < 0.5 * SHARES["fact"]
    assert man["lane_plan"]["feasible"] is False
    assert man["lane_plan"]["deficit_tokens"] > 0


# ---------------------------------------------------------- config generation

_GSPEC = importlib.util.spec_from_file_location(
    "gen_configs",
    Path(__file__).resolve().parents[1] / "ops" / "crowding" / "gen_configs.py",
)
gc = importlib.util.module_from_spec(_GSPEC)
_GSPEC.loader.exec_module(gc)


def _cohort():
    return gc.cohort({"high": "corpora/high", "low": "corpora/low"},
                     [0, 1], "d40m", 5_737_000_000, 1.5e-3, 1.0)


def test_cohort_shape():
    cfgs = _cohort()
    assert len(cfgs) == 2 * 2 * 3  # loads x seeds x arms
    assert {c["arm"] for c in cfgs} == set(gc.ARMS)


def test_arms_differ_only_in_the_mask():
    gc.assert_arms_match(_cohort())


def test_a_drifted_key_between_arms_is_caught():
    """The difference between the arms IS the experiment; a config key that
    drifts is a confound with no symptom until the analysis."""
    cfgs = _cohort()
    cfgs[1]["lr"] = 9.9e-3
    with pytest.raises(AssertionError, match="differ in"):
        gc.assert_arms_match(cfgs)


def test_sup_has_no_train_mask_and_the_others_do():
    by_arm = {c["arm"]: c for c in _cohort() if c["seed"] == 0
              and c["run_id"].split("_")[1] == "high"}
    assert "train_mask" not in by_arm["sup"]
    assert by_arm["factmask"]["train_mask"].endswith("factmask.bin")
    assert by_arm["randpos"]["train_mask"].endswith("randpos.bin")


def test_every_arm_probes_the_same_positions():
    """Gate 0 is only cross-arm comparable if all arms score the same
    offloaded positions."""
    probes = {c["probe_mask"] for c in _cohort()
              if c["run_id"].split("_")[1] == "high"}
    assert len(probes) == 1 and "factmask.bin" in probes.pop()


# ---------------------------------------------------------- parallel fact lane


def test_randpos_is_a_pure_function_of_its_coordinates():
    """It used to take a shared RNG threaded through the whole fact lane, so a
    document's control mask depended on how far that stream had advanced --
    making the corpus depend on generation order."""
    import random as _r
    a = bc.randpos_seed(7, 3)
    b = bc.randpos_seed(7, 3)
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]
    assert bc.randpos_seed(7, 3).random() != bc.randpos_seed(7, 4).random()


def test_parallel_build_is_byte_identical_to_serial(tmp_path):
    """The whole point. If workers changed the corpus, every run would be
    conditional on how many cores happened to be free."""
    a = bc.build(tmp_path / "serial", 40, 22, SHARES, 120_000, seed=0, workers=1)
    b = bc.build(tmp_path / "par", 40, 22, SHARES, 120_000, seed=0, workers=4)
    assert a["sha256"] == b["sha256"], "parallel output differs from serial"


def test_parallel_build_passes_the_same_verifier(tmp_path):
    out = tmp_path / "par"
    bc.build(out, 40, 22, SHARES, 120_000, seed=0, workers=4)
    assert bc.verify(out, expect_tokens=120_000, allow_synthetic_bed=True,
                     require_randpos_nll=False) == []


def test_chunk_boundaries_do_not_change_the_output():
    """Chunks of 7 docs put boundaries mid-round, where an order-dependent
    generator would diverge."""
    from corpusgen import bios
    from train.tokenizer import get_tok
    tok = get_tok()
    recs = bios.generate_records(9, 1)
    serial = list(bc.fact_stream(recs, 4, tok, workers=1))
    chunked = list(bc.fact_stream(recs, 4, tok, workers=3, chunk_docs=7))
    assert len(serial) == len(chunked) == 36
    for (a_i, a_f, a_r), (b_i, b_f, b_r) in zip(serial, chunked):
        assert np.array_equal(a_i, b_i)
        assert np.array_equal(a_f, b_f)
        assert np.array_equal(a_r, b_r)


def test_doc_index_inverts_the_emission_order():
    from corpusgen import factlane
    order = list(factlane.doc_order(5, 3))
    for k, (entity, exposure) in enumerate(order):
        assert factlane.doc_at(k, 5) == (entity, exposure)


# ----------------------------------------------- the gates added on 2026-08-02


def _flat_nll(value=1.0, vocab=50304):
    return np.full(vocab, value, dtype=np.float32)


def test_plan_lanes_prices_the_fact_lane_before_writing_anything():
    """Stage A's arithmetic, which nothing checked: 20,000 x 20 documents
    cannot fill a 50% share of 800M tokens."""
    from corpusgen import bios
    from train.tokenizer import get_tok
    recs = bios.generate_records(50, 0)
    plan = bc.plan_lanes(recs, 20, get_tok(), SHARES, 10_000_000)
    assert plan["feasible"] is False
    assert plan["deficit_tokens"] > 0
    assert plan["realised_fact_share"] < SHARES["fact"]
    assert plan["exposures_needed_at_this_entity_count"] > 20


def test_plan_lanes_accepts_a_lane_that_can_fill_its_share():
    from corpusgen import bios
    from train.tokenizer import get_tok
    recs = bios.generate_records(60, 0)
    plan = bc.plan_lanes(recs, 25, get_tok(), SHARES, 200_000)
    assert plan["feasible"] is True
    assert plan["deficit_tokens"] == 0


def test_verify_rejects_a_synthetic_bed_by_default(tmp_path):
    """The runbook says treat SYNTHETIC-REHEARSAL-ONLY in a manifest as a
    build failure. Stage A shipped one and nothing noticed."""
    out = tmp_path / "syn"
    bc.build(out, 60, 25, SHARES, 200_000, seed=0)
    fails = bc.verify(out, expect_tokens=200_000, require_randpos_nll=False)
    assert any("SYNTHETIC-REHEARSAL-ONLY" in f for f in fails)


def test_verify_rejects_an_unmeasured_randpos_match_by_default(tmp_path):
    """Every corpus this project has built matched mass but not difficulty,
    because token_nll was never supplied. Silence is no longer an option."""
    out = tmp_path / "nonll"
    man = bc.build(out, 60, 25, SHARES, 200_000, seed=0)
    assert man["randpos_validity"]["status"] == "NOT-MEASURED"
    fails = bc.verify(out, expect_tokens=200_000, allow_synthetic_bed=True)
    assert any("never measured" in f for f in fails)


def test_a_frozen_table_makes_the_match_measurable(tmp_path):
    out = tmp_path / "withnll"
    man = bc.build(out, 60, 25, SHARES, 200_000, seed=0,
                   nll_table=_flat_nll())
    rv = man["randpos_validity"]
    assert rv["status"] == "MEASURED"
    assert rv["n_masked_factmask"] == rv["n_masked_randpos"] > 0
    # A flat table makes every token equally difficult, so the match is exact.
    assert rv["relative_gap"] == pytest.approx(0.0, abs=1e-9)
    assert rv["within_tolerance"] is True
    assert bc.verify(out, expect_tokens=200_000, allow_synthetic_bed=True) == []


def test_an_unmatchable_control_is_reported_not_gated(tmp_path):
    """Preregistration §2: if the control cannot be difficulty-matched, that
    is the result. It must surface loudly and must NOT fail the build."""
    vocab = 50304
    table = np.full(vocab, 0.1, dtype=np.float32)
    # Make the tokens that fact values are drawn from far more surprising than
    # anything else, so no placement can match them.
    table[1000:] = 20.0
    out = tmp_path / "unmatchable"
    man = bc.build(out, 60, 25, SHARES, 200_000, seed=0, nll_table=table)
    rv = man["randpos_validity"]
    assert rv["status"] == "MEASURED"
    if not rv["within_tolerance"]:
        findings = bc.report_findings(out)
        assert any("NOT difficulty-matched" in f for f in findings)
        # Disclosed, not fatal.
        assert bc.verify(out, expect_tokens=200_000,
                         allow_synthetic_bed=True) == []


def test_the_nll_table_does_not_break_parallel_byte_identity(tmp_path):
    """Difficulty-aware placement must stay a pure function of coordinates,
    or the corpus would depend on how many cores were free."""
    table = np.linspace(0.1, 5.0, 50304).astype(np.float32)
    a = bc.build(tmp_path / "s", 40, 22, SHARES, 120_000, seed=0, workers=1,
                 nll_table=table)
    b = bc.build(tmp_path / "p", 40, 22, SHARES, 120_000, seed=0, workers=4,
                 nll_table=table)
    assert a["sha256"] == b["sha256"]


def test_difficulty_matching_actually_moves_randpos(tmp_path):
    """A table with real structure must change where control spans land,
    otherwise the fourth axis is decorative."""
    table = np.linspace(0.1, 5.0, 50304).astype(np.float32)
    plain = bc.build(tmp_path / "plain", 40, 22, SHARES, 120_000, seed=0)
    tuned = bc.build(tmp_path / "tuned", 40, 22, SHARES, 120_000, seed=0,
                     nll_table=table)
    assert plain["sha256"]["targets.bin"] == tuned["sha256"]["targets.bin"], \
        "the token stream must be identical; only the control sidecar moves"
    assert plain["sha256"]["factmask.bin"] == tuned["sha256"]["factmask.bin"]
    assert plain["sha256"]["randpos.bin"] != tuned["sha256"]["randpos.bin"]


def test_mask_mass_is_exact_when_the_tail_truncates_a_document(tmp_path):
    """A fact document cut by the token limit loses its two sidecars at
    different places, so the arms end with unequal target counts -- a
    different effective objective under the fixed-denominator loss. The
    builder spends the tail on bed tokens instead, which are all-ones in both
    sidecars. 100 x 25 at 300k tokens is the shape that exposed this.
    """
    out = tmp_path / "tail"
    man = bc.build(out, 100, 25, SHARES, 300_000, seed=0)
    assert man["n_tokens"] == 300_000
    f = np.fromfile(out / "factmask.bin", dtype=np.uint8)
    r = np.fromfile(out / "randpos.bin", dtype=np.uint8)
    assert int((f == 0).sum()) == int((r == 0).sum())
    assert man["mask_audit"]["mass_matched"]


def test_a_dense_document_scatters_rather_than_dropping_mass(tmp_path):
    """When no contiguous home exists for a control span, mass is preserved
    and the length histogram degrades -- never the other way round."""
    rng = __import__("random").Random(0)
    ids = np.arange(40, dtype=np.uint16)
    # 17 value tokens among 23 free ones, so placement is possible -- but the
    # free positions are fragmented into runs of at most 2, so the six-token
    # span has no contiguous home anywhere.
    fmask = np.ones(40, dtype=np.uint8)
    fmask[0:6] = 0
    for p in range(8, 40, 3):
        fmask[p] = 0
    assert max(ln for _, ln in randpos.spans_of(np.where(fmask == 0, 1, 0))) < 6
    out = randpos.build(ids, fmask, rng)
    assert int((out == 0).sum()) == int((fmask == 0).sum()), \
        "control must zero exactly as many targets as the treatment"
    assert not ((out == 0) & (fmask == 0)).any(), \
        "control must never land on a value token"


def test_the_feasibility_gate_tolerates_a_boundary_sized_lane():
    """Both operating points in the repository fill the fact lane to within
    0.01%, and capacity is a sample mean, so the gate must not refuse the
    intended design on measurement noise. It refuses Stage A's 46% deficit."""
    from corpusgen import bios
    from train.tokenizer import get_tok
    tok, recs = get_tok(), bios.generate_records(200, 0)
    cap, mean_doc = bc.fact_lane_capacity(recs, 20, tok)
    total = int(cap / SHARES["fact"])          # sized to fill exactly
    assert bc.plan_lanes(recs, 20, tok, SHARES, total)["feasible"]
    # A corpus 5% larger leaves the lane 5% short of its share: refused.
    assert not bc.plan_lanes(recs, 20, tok, SHARES,
                             int(total * 1.05))["feasible"]


def test_sup_only_waives_the_difficulty_table_but_not_the_other_gates(tmp_path):
    """A difficulty probe never trains a masked arm, so RANDPOS matching is
    not applicable to it. That must not become a way to smuggle a synthetic
    bed or an underfilled lane past the gates -- which is what bundling it
    into --rehearsal would have done."""
    out = tmp_path / "supon"
    man = bc.build(out, 60, 25, SHARES, 200_000, seed=0, sup_only=True)
    assert man["valid_for"] == "SUP-only"
    # The difficulty gate is waived ...
    fails = bc.verify(out, expect_tokens=200_000, allow_synthetic_bed=True,
                      require_randpos_nll=False)
    assert fails == []
    # ... but the bed gate is not.
    fails = bc.verify(out, expect_tokens=200_000, require_randpos_nll=False)
    assert any("SYNTHETIC" in f for f in fails)
    # And a masked arm must not be able to use it unknowingly.
    fails = bc.verify(out, expect_tokens=200_000, allow_synthetic_bed=True,
                      require_randpos_nll=True)
    assert any("no masked arm may train on it" in f for f in fails)


def test_a_normal_build_is_marked_valid_for_all_arms(tmp_path):
    out = tmp_path / "allarms"
    man = bc.build(out, 60, 25, SHARES, 200_000, seed=0, nll_table=_flat_nll())
    assert man["valid_for"] == "all arms"


# ------------------------- Stage B's NOFACT arm, repaired after the lane gate


NOFACT_SHARES = {"fact": 0.0, "igsm": 0.30, "deduction": 0.10, "bed": 0.60}


def test_the_runbook_nofact_recipe_is_now_refused(tmp_path):
    """`MS_ENTITIES=1 MS_EXPOSURES=1` expressed 'no facts' by starving the lane
    and letting the bed absorb 70% of the corpus -- the same silent mechanism
    that produced the Stage A defect. It must not be the way to say it."""
    with pytest.raises(bc.LaneUnderfilled):
        bc.build(tmp_path / "old", n_entities=1, exposures=1,
                 shares={"fact": .70, "igsm": .123, "deduction": .043,
                         "bed": .134},
                 total_tokens=200_000, seed=0)


def test_a_zero_fact_share_builds_a_clean_nofact_corpus(tmp_path):
    """Stage B's NOFACT arm: same tokens and steps as its twin, no memorisable
    facts. Saying it with a zero share states the intent in the manifest."""
    out = tmp_path / "nofact"
    man = bc.build(out, n_entities=1, exposures=1, shares=NOFACT_SHARES,
                   total_tokens=200_000, seed=0)
    assert man["n_tokens"] == 200_000, "token count must match its twin exactly"
    assert man["lane_profile"] == "NOFACT"
    assert man["lane_tokens"]["fact"] == 0
    assert man["mask_audit"]["factmask_zeros"] == 0
    assert bc.verify(out, expect_tokens=200_000, allow_synthetic_bed=True) == []


def test_a_nofact_corpus_with_leaked_fact_tokens_fails(tmp_path):
    """Empty sidecars are correct only because there are no facts. If any
    masked target appears, the lane leaked and the arm is not NOFACT."""
    out = tmp_path / "leaky"
    bc.build(out, n_entities=1, exposures=1, shares=NOFACT_SHARES,
             total_tokens=200_000, seed=0)
    f = np.fromfile(out / "factmask.bin", dtype=np.uint8)
    f[:5] = 0
    f.tofile(out / "factmask.bin")
    fails = bc.verify(out, expect_tokens=200_000, allow_synthetic_bed=True)
    assert any("NOFACT corpus has masked targets" in x for x in fails)


def test_nofact_does_not_need_a_difficulty_table(tmp_path):
    """No fact values means no control spans to difficulty-match."""
    out = tmp_path / "nofact2"
    bc.build(out, n_entities=1, exposures=1, shares=NOFACT_SHARES,
             total_tokens=200_000, seed=0)
    assert bc.verify(out, expect_tokens=200_000, allow_synthetic_bed=True,
                     require_randpos_nll=True) == []


def test_vectorised_difficulty_matching_is_byte_identical(tmp_path):
    """The sliding-window reduction replaced a per-candidate np.mean loop for
    speed. If it changed a single placement it would change the corpus, so
    this pins it against a straightforward reimplementation of the old path.
    """
    import random as _random
    from corpusgen import bios, factlane
    from train.tokenizer import get_tok

    table = np.linspace(0.05, 6.0, 50304).astype(np.float32)
    tok = get_tok()
    recs = bios.generate_records(40, 0)

    def reference(ids, fmask, rng, token_nll):
        """The pre-optimisation placement rule, transcribed."""
        n = len(fmask)
        out = np.ones(n, dtype=np.uint8)
        spans = randpos.spans_of(fmask)
        if not spans:
            return out
        forbidden = np.zeros(n, dtype=bool)
        for s, ln in spans:
            forbidden[s:s + ln] = True
        taken = np.zeros(n, dtype=bool)
        for i in sorted(range(len(spans)), key=lambda i: -spans[i][1]):
            start, length = spans[i]
            target_rel = start / max(1, n)
            target_nll = float(np.mean(token_nll[start:start + length]))
            cands = [c for c in range(0, n - length + 1)
                     if not forbidden[c:c + length].any()
                     and not taken[c:c + length].any()
                     and abs(c / max(1, n) - target_rel) <= 0.15]
            if not cands:
                cands = [c for c in range(0, n - length + 1)
                         if not forbidden[c:c + length].any()
                         and not taken[c:c + length].any()]
            if not cands:
                free = np.flatnonzero(~forbidden & ~taken)
                if free.size:
                    take = rng.sample(free.tolist(), min(length, free.size))
                    out[take] = 0
                    taken[take] = True
                continue
            best = min(cands, key=lambda c: abs(
                float(np.mean(token_nll[c:c + length])) - target_nll))
            out[best:best + length] = 0
            taken[best:best + length] = True
        return out

    checked = 0
    for k in range(120):
        ids, fmask, entity, exposure = factlane.render_one(recs, k, tok)
        nll = table[np.asarray(ids, dtype=np.int64)]
        fast = randpos.build(ids, fmask, _random.Random(f"x:{k}"), token_nll=nll)
        slow = reference(ids, fmask, _random.Random(f"x:{k}"), nll)
        assert np.array_equal(fast, slow), f"placement diverged on document {k}"
        checked += 1
    assert checked == 120
