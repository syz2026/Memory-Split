"""The three architectural confounds, each a live mechanism that can produce
the hypothesised effect with no capacity freed.

1. `n_recurrence=2` on d40m makes it a depth-recurrent weight-shared model, so
   bits-per-parameter against standard-GPT-2 literature is contested.
2. Gradient clipping was hardcoded at 1.0. Fact values carry several times more
   gradient mass per token than template text, so a binding clip scales the
   arms differently.
3. `PackedShards` accepted `seed` and ignored it, so every replicate read the
   identical stream in the identical order.
"""

import json

import numpy as np
import torch

from train.data import PackedShards
from train.model import PRESETS, GPT
from train.trainer import Trainer


# ------------------------------------------------------------ preset


def test_d40m_std_is_d40m_without_weight_sharing():
    a, b = PRESETS["d40m"], PRESETS["d40m_std"]
    assert (a.n_layer, a.n_head, a.d_model) == (b.n_layer, b.n_head, b.d_model)
    assert a.tie_embeddings == b.tie_embeddings
    assert a.n_recurrence == 2 and b.n_recurrence == 1
    assert a.effective_depth == 24 and b.effective_depth == 12


def test_d40m_std_parameter_count_is_unchanged():
    """Recurrence reuses blocks, so dropping it must not move the count."""
    assert GPT(PRESETS["d40m_std"]).num_params() == 40_560_000
    assert GPT(PRESETS["d40m"]).num_params() == 40_560_000


# ------------------------------------------------------------ data order


def _corpus(tmp_path, n=200_000, seed=0):
    p = tmp_path / f"train_{seed}.bin"
    rng = np.random.default_rng(seed)
    rng.integers(0, 50000, size=n, dtype=np.uint16).tofile(p)
    return p


def test_seed_zero_is_the_canonical_unrotated_order(tmp_path):
    p = _corpus(tmp_path)
    ds = PackedShards(p, None, ctx=64, batch_size=4, seed=0)
    assert ds.base_offset == 0


def test_different_seeds_give_different_first_batches(tmp_path):
    p = _corpus(tmp_path)
    xs = []
    for seed in (0, 1, 2):
        ds = PackedShards(p, None, ctx=64, batch_size=4, seed=seed)
        x, _, _ = ds.next_batch()
        xs.append(x)
    assert not torch.equal(xs[0], xs[1])
    assert not torch.equal(xs[0], xs[2])
    assert not torch.equal(xs[1], xs[2])


def test_seed_is_deterministic(tmp_path):
    p = _corpus(tmp_path)
    a = PackedShards(p, None, ctx=64, batch_size=4, seed=7)
    b = PackedShards(p, None, ctx=64, batch_size=4, seed=7)
    xa, _, _ = a.next_batch()
    xb, _, _ = b.next_batch()
    assert torch.equal(xa, xb)


def test_rotation_is_a_pure_rotation_of_the_stream(tmp_path):
    """Reading the whole stream must reproduce it rotated by base_offset, with
    no token dropped or duplicated at the wrap. The loader overlaps successive
    batches by one token by design, so this is asserted on the underlying
    window read rather than on batch contents."""
    n, ctx, bs = 8192, 16, 2
    p = tmp_path / "c.bin"
    original = np.arange(n, dtype=np.uint16)
    original.tofile(p)
    for seed in (0, 3, 11):
        ds = PackedShards(p, None, ctx=ctx, batch_size=bs, seed=seed)
        toks, _ = ds._window(0, n)
        assert len(toks) == n, f"seed {seed}: read {len(toks)} of {n}"
        expected = np.roll(original, -ds.base_offset)
        assert np.array_equal(toks, expected), f"seed {seed}: not a clean rotation"
        assert sorted(toks.tolist()) == list(range(n)), f"seed {seed}: lost tokens"


def test_rotation_is_batch_aligned(tmp_path):
    """An unaligned offset would shift document boundaries relative to context
    windows, which is a data change rather than a data-order change."""
    p = _corpus(tmp_path)
    for seed in (1, 5, 9):
        ds = PackedShards(p, None, ctx=64, batch_size=4, seed=seed)
        assert ds.base_offset % (4 * 64) == 0


# ------------------------------------------------------------ clipping


def _cfg(tmp_path, **over):
    bin_path = tmp_path / "t.bin"
    rng = np.random.default_rng(0)
    rng.integers(0, 50000, size=120_000, dtype=np.uint16).tofile(bin_path)
    cfg = {
        "model": {"n_layer": 2, "n_head": 2, "d_model": 64,
                  "ctx": 64, "vocab_size": 50304},
        "train_bin": str(bin_path), "micro_batch_size": 2,
        "tokens_per_step": 256, "max_steps": 10, "lr": 1e-3,
        "warmup_steps": 1, "seed": 0, "out_dir": str(tmp_path / "run"),
        "log_every": 1, "eval_every": 10_000, "device": "cpu",
        "snap_frac": 0.5, "ckpt_minutes": 999,
    }
    cfg.update(over)
    return cfg


def test_grad_clip_is_configurable(tmp_path):
    assert Trainer(_cfg(tmp_path, grad_clip=7.5)).grad_clip == 7.5


def test_grad_clip_defaults_to_one(tmp_path):
    assert Trainer(_cfg(tmp_path)).grad_clip == 1.0


def test_log_carries_optimization_diagnostics(tmp_path):
    tr = Trainer(_cfg(tmp_path))
    tr.train_steps(2)
    rows = [json.loads(line) for line in open(tr.log_path)]
    assert rows, "no log rows written"
    last = rows[-1]
    for key in ("grad_norm_preclip", "clip_frac", "clip_ratio", "adam_v_mean"):
        assert key in last, f"missing diagnostic {key}"
    assert last["grad_norm_preclip"] > 0
    assert 0.0 <= last["clip_frac"] <= 1.0
    assert 0.0 < last["clip_ratio"] <= 1.0
    assert last["adam_v_mean"] >= 0.0


def test_a_loose_clip_does_not_bind(tmp_path):
    """The pilot raises grad_clip until this holds for the dense arm."""
    tr = Trainer(_cfg(tmp_path, grad_clip=1e6))
    tr.train_steps(2)
    rows = [json.loads(line) for line in open(tr.log_path)]
    assert rows[-1]["clip_frac"] == 0.0
    assert rows[-1]["clip_ratio"] == 1.0
