"""Gradient-mass decomposition: the only capacity-facing measurement here."""

import torch

from evals.gradmass import decompose
from train.model import GPT, GPTConfig

CPU = torch.device("cpu")


def _model():
    torch.manual_seed(0)
    return GPT(GPTConfig(n_layer=1, n_head=2, d_model=32, vocab_size=64, ctx=16))


def _batches(token_lo, token_hi, n=2, B=2, T=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randint(token_lo, token_hi, (B, T), generator=g)
        y = torch.randint(token_lo, token_hi, (B, T), generator=g)
        w = torch.ones_like(y, dtype=torch.float32)
        out.append((x, y, w))
    return out


def test_report_shape():
    out = decompose(_model(), _batches(0, 20), _batches(40, 64, seed=1), CPU)
    assert out["n_params"] > 0
    assert 0.0 <= out["fact_dominant_frac"] <= 1.0
    assert abs(out["fact_dominant_frac"] + out["reasoning_dominant_frac"] - 1.0) < 1e-9
    assert abs(out["fact_mass_share"] + out["reasoning_mass_share"] - 1.0) < 1e-6


def test_disjoint_token_ranges_split_the_embedding_mass():
    """Two objectives touching disjoint vocabulary must not both dominate the
    same embedding rows."""
    out = decompose(_model(), _batches(0, 20), _batches(40, 64, seed=1), CPU)
    assert 0.0 < out["fact_dominant_frac"] < 1.0


def test_include_filter_excludes_the_embedding_table():
    """A tied 40.6M model is 47.6% embedding; crowding in a lookup table is
    not the competition the hypothesis is about, so report both ways."""
    m = _model()
    everything = decompose(m, _batches(0, 20), _batches(40, 64, seed=1), CPU)
    blocks = decompose(m, _batches(0, 20), _batches(40, 64, seed=1), CPU,
                       include="blocks")
    assert blocks["n_params"] < everything["n_params"]
    assert blocks["include"] == "blocks"


def test_identical_objectives_are_near_balanced():
    m = _model()
    a = _batches(0, 30, seed=5)
    out = decompose(m, a, list(a), CPU)
    assert abs(out["fact_mass_share"] - 0.5) < 1e-6


def test_decompose_leaves_no_gradients_behind():
    m = _model()
    decompose(m, _batches(0, 20), _batches(40, 64, seed=1), CPU)
    assert all(p.grad is None for p in m.parameters())
