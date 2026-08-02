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
    assert bc.verify(out, expect_tokens=200_000) == []


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
    a = bc.build(tmp_path / "a", 30, 4, SHARES, 60_000, seed=3)
    b = bc.build(tmp_path / "b", 30, 4, SHARES, 60_000, seed=3)
    assert a["sha256"] == b["sha256"]


def test_a_different_seed_changes_the_stream(tmp_path):
    a = bc.build(tmp_path / "a", 30, 4, SHARES, 60_000, seed=3)
    b = bc.build(tmp_path / "b", 30, 4, SHARES, 60_000, seed=4)
    assert a["sha256"]["targets.bin"] != b["sha256"]["targets.bin"]


# ---------------------------------------------------------- verify fails closed


def test_verify_catches_a_short_corpus(built):
    out, _ = built
    fails = bc.verify(out, expect_tokens=999_999)
    assert any("tokens" in f for f in fails)


def test_verify_catches_a_truncated_sidecar(tmp_path):
    out = tmp_path / "c"
    bc.build(out, 20, 4, SHARES, 40_000, seed=1)
    data = np.fromfile(out / "randpos.bin", dtype=np.uint8)[:-100]
    data.tofile(out / "randpos.bin")
    fails = bc.verify(out, expect_tokens=40_000)
    assert any("randpos sidecar" in f for f in fails)


def test_verify_catches_an_inert_sidecar(tmp_path):
    out = tmp_path / "d"
    bc.build(out, 20, 4, SHARES, 40_000, seed=1)
    n = (out / "targets.bin").stat().st_size // 2
    np.ones(n, dtype=np.uint8).tofile(out / "factmask.bin")
    fails = bc.verify(out, expect_tokens=40_000)
    assert any("masks nothing" in f for f in fails)


def test_verify_catches_unequal_mask_mass(tmp_path):
    out = tmp_path / "e"
    bc.build(out, 20, 4, SHARES, 40_000, seed=1)
    rp = np.fromfile(out / "randpos.bin", dtype=np.uint8)
    rp[np.flatnonzero(rp == 0)[:5]] = 1  # remove five masked targets
    rp.tofile(out / "randpos.bin")
    fails = bc.verify(out, expect_tokens=40_000)
    assert any("mask mass differs" in f for f in fails)


def test_an_exhausted_lane_gives_its_budget_to_the_bed(tmp_path):
    """A finite lane that cannot fill its share must not shorten the corpus:
    every arm and load in this design shares one step count."""
    out = tmp_path / "small"
    man = bc.build(out, n_entities=10, exposures=4, shares=SHARES,
                   total_tokens=200_000, seed=0)
    assert man["n_tokens"] == 200_000
    assert man["lane_tokens"]["fact"] < SHARES["fact"] * 200_000
    assert man["lane_tokens"]["bed"] > SHARES["bed"] * 200_000
    assert bc.verify(out, expect_tokens=200_000) == []


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
    a = bc.build(tmp_path / "serial", 40, 12, SHARES, 120_000, seed=0, workers=1)
    b = bc.build(tmp_path / "par", 40, 12, SHARES, 120_000, seed=0, workers=4)
    assert a["sha256"] == b["sha256"], "parallel output differs from serial"


def test_parallel_build_passes_the_same_verifier(tmp_path):
    out = tmp_path / "par"
    bc.build(out, 40, 12, SHARES, 120_000, seed=0, workers=4)
    assert bc.verify(out, expect_tokens=120_000) == []


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
