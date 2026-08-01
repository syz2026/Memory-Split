import json

import numpy as np

from train.trainer import Trainer, cosine_lr


def write_corpus(tmp_path, n=40000, mask_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    # learnable structure: short repeating motifs + noise
    motif = rng.integers(0, 200, size=50)
    toks = np.tile(motif, n // 50 + 1)[:n].astype(np.uint16)
    noise_at = rng.random(n) < 0.05
    toks[noise_at] = rng.integers(0, 200, size=noise_at.sum())
    mask = (rng.random(n) > mask_frac).astype(np.uint8)
    bp, mp = tmp_path / "train.bin", tmp_path / "train.mask.bin"
    toks.tofile(bp)
    mask.tofile(mp)
    return bp, mp


def base_cfg(tmp_path, bp, mp):
    return {
        "model": {"n_layer": 2, "n_head": 2, "d_model": 64, "ctx": 64, "vocab_size": 50304},
        "train_bin": str(bp),
        "train_mask": str(mp),
        "micro_batch_size": 4,
        "tokens_per_step": 4 * 64,
        "max_steps": 80,
        "lr": 3e-3,
        "warmup_steps": 5,
        "seed": 7,
        "out_dir": str(tmp_path / "out"),
        "device": "cpu",
        "log_every": 5,
        "eval_every": 10,
        "snap_frac": 0.5,
        "ckpt_minutes": 999,
    }


def test_loss_decreases_and_logs(tmp_path):
    bp, mp = write_corpus(tmp_path)
    tr = Trainer(base_cfg(tmp_path, bp, mp))
    tr.train_steps()
    rows = [json.loads(l) for l in open(tr.log_path)]
    first, last = rows[0]["loss"], rows[-1]["loss_ema"]
    assert last < first * 0.8, (first, last)
    assert any("loss_masked_values" in r for r in rows)
    assert (tr.out_dir / "ckpt.pt").exists()
    snaps = list((tr.out_dir / "snapshots").glob("*.pt"))
    assert snaps


def test_checkpoint_resume_exact_batches(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["max_steps"] = 10
    a = Trainer(cfg)
    a.train_steps(6)
    a.save_ckpt()
    cursor_after_6 = a.data.state_dict()["cursor"]

    cfg2 = dict(cfg, out_dir=str(tmp_path / "out2"))
    b = Trainer(cfg2)
    b.load_ckpt(a.ckpt_path)
    assert b.step == 6
    assert b.data.state_dict()["cursor"] == cursor_after_6
    xb, _, _ = b.data.next_batch()
    xa, _, _ = a.data.next_batch()
    assert (xa == xb).all()


def test_cosine_schedule():
    assert cosine_lr(0, 1.0, 10, 100) < 0.2
    assert abs(cosine_lr(10, 1.0, 10, 100) - 1.0) < 0.01
    assert cosine_lr(99, 1.0, 10, 100) < 0.2
    assert cosine_lr(150, 1.0, 10, 100) == 0.1


def test_segmented_corpus_config_trains(tmp_path):
    """Every cohort config passes train_bin/train_mask as a base+extension pair.
    Drive that shape through the real Trainer, not just the loader."""
    base, ext = tmp_path / "base", tmp_path / "ext"
    base.mkdir()
    ext.mkdir()
    bp_a, mp_a = write_corpus(base, n=30000, seed=0)
    bp_b, mp_b = write_corpus(ext, n=10000, seed=1)

    cfg = base_cfg(tmp_path, bp_a, mp_a)
    cfg["train_bin"] = [str(bp_a), str(bp_b)]
    cfg["train_mask"] = [str(mp_a), str(mp_b)]
    tr = Trainer(cfg)
    assert tr.data.n_tokens == 40000

    tr.train_steps()
    rows = [json.loads(line) for line in open(tr.log_path)]
    assert rows[-1]["loss_ema"] < rows[0]["loss"] * 0.8
