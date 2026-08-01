"""Invariants of the high-exposure fact segment.

The segment exists because reasoning-v3's offloaded spans turned out to be
94.2% inferable from context, so masking them relieved almost nothing. Two
properties have to hold for the replacement to be a fair test, and both are
easy to break silently, so they are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from corpusgen import bios, fact_segment as fs
from corpusgen.records import ATTRIBUTES
from train.tokenizer import get_tok


@pytest.fixture(scope="module")
def tok():
    return get_tok()


@pytest.fixture(scope="module")
def recs():
    return bios.generate_records(60, seed=11)


def test_plan_budget_closes(recs):
    p = fs.plan(4_000_000_000, 200)
    assert p["n_entities"] == 271_739
    assert p["n_docs"] == p["n_entities"] * 200
    # the three quantities trade off exactly
    assert p["n_docs"] * fs.TOKENS_PER_DOC_ESTIMATE <= p["budget_tokens"]


def test_plan_puts_d8m_at_its_storage_ceiling():
    """200 exposures is chosen so the 8M model is saturated and the 40M is not;
    that gradient is the crowding test. ~2 bits/param is the storage rule."""
    p = fs.plan(4_000_000_000, 200)
    assert 1.5 < p["bits_per_param"]["d8m"] < 2.5
    assert p["bits_per_param"]["d40m"] < 0.5
    assert p["bits_per_param"]["d160m"] < 0.2


def test_lower_exposure_overshoots_capacity():
    """The failure mode of the current corpus: huge fact universe, none of it
    memorisable, so dense is never meaningfully burdened."""
    assert fs.plan(4_000_000_000, 6)["bits_per_param"]["d8m"] > 50


def test_order_is_spaced_not_massed():
    order = list(fs.doc_order(4, 3))
    assert order == [(e, x) for x in range(3) for e in range(4)]
    # no entity repeats until every other entity has been seen once
    first_round = order[:4]
    assert len({e for e, _ in first_round}) == 4


def test_masked_spans_are_exactly_the_values(recs, tok):
    for rec in recs[:15]:
        for exposure in (0, 1, 99, 199):
            fs.verify_doc(rec, exposure, tok)


def test_values_never_appear_unmasked(recs, tok):
    """If a value also occurs in plain text the offload leaks and the split arm
    can read the answer it was supposed to have been denied."""
    for rec in recs[:15]:
        for exposure in (0, 7, 199):
            segs = bios.render_bio_marked(rec, exposure)
            plain = "".join(t for t, m in segs if not m)
            for a in ATTRIBUTES:
                assert rec.attrs[a] not in plain


def test_encode_docs_yields_aligned_ids_and_mask(recs, tok):
    n = 0
    for ids, mask in fs.encode_docs(recs, fs.doc_order(10, 3), tok):
        assert ids.dtype == np.uint16 and mask.dtype == np.uint8
        assert len(ids) == len(mask)
        assert ids[-1] == tok.EOT and mask[-1] == 1   # boundary token carries loss
        assert 0 in mask                              # something is offloaded
        n += 1
    assert n == 30


def test_offloaded_fraction_is_material(recs, tok):
    frac = fs.offloaded_fraction(recs, tok)
    assert 0.15 < frac < 0.40, frac


def test_stream_is_deterministic(recs, tok):
    a = [bytes(i) for i, _ in fs.encode_docs(recs, fs.doc_order(8, 4), tok)]
    b = [bytes(i) for i, _ in fs.encode_docs(recs, fs.doc_order(8, 4), tok)]
    assert a == b


def test_builder_interleaves_and_hits_the_exact_budget(tmp_path):
    """End-to-end: the writer must land on the token target exactly (max_steps
    depends on it), keep the arms nested, and actually interleave rather than
    concatenate -- a concatenated stream would mass every fact's exposures."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    src = tmp_path / "v3"
    (src / "base/packed").mkdir(parents=True)
    (src / "base/sidecars").mkdir(parents=True)
    n = 200_000
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 50000, n).astype(np.uint16)
    msk = np.ones(n, dtype=np.uint8)
    for i, s in enumerate(range(0, n, 200)):
        toks[min(s + 199, n - 1)] = 50260
        if i % 2 == 0:
            msk[s + 50 : s + 60] = 0
    toks.tofile(src / "base/packed/targets.bin")
    msk.tofile(src / "base/sidecars/split90_target_weights.bin")
    np.ones(n, np.uint8).tofile(src / "base/sidecars/dense_target_weights.bin")

    out = tmp_path / "v4"
    script = Path(__file__).resolve().parents[1] / "ops" / "corpus-v4" / "build_v4.py"
    r = subprocess.run(
        [sys.executable, str(script), "--src", str(src), "--out", str(out),
         "--fact-tokens", "112000", "--exposures", "20", "--limit-base", str(n)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr

    t = np.fromfile(out / "base/packed/targets.bin", dtype=np.uint16)
    d = np.fromfile(out / "base/sidecars/dense_target_weights.bin", dtype=np.uint8)
    s = np.fromfile(out / "base/sidecars/split90_target_weights.bin", dtype=np.uint8)
    assert len(t) == len(d) == len(s) == 312_000        # exact budget
    assert (d == 1).all()                               # dense supervises everything
    assert int(((d == 0) & (s == 1)).sum()) == 0        # strictly nested
    assert (s == 0).any()                               # something is offloaded

    half = len(t) // 2
    assert (s[:half] == 0).any() and (s[half:] == 0).any(), "streams concatenated"

    m = json.loads((out / "v4-manifest.json").read_text())
    assert m["fact_tokens_emitted"] > 0 and m["base_tokens_emitted"] > 0
    assert m["n_exposures"] == 20
