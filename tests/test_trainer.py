import json

import numpy as np
import pytest

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
    assert (tr.out_dir / "model.pt").exists()
    assert len((tr.out_dir / "model.pt.sha256").read_text().strip()) == 64
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
    xb, _ = b.data.next_batch()
    xa, _ = a.data.next_batch()
    assert (xa == xb).all()


def test_weights_only_initialization_resets_training_state(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["max_steps"] = 3
    source = Trainer(cfg)
    source.train_steps(2)
    source.save_ckpt()

    destination_cfg = dict(cfg, out_dir=str(tmp_path / "continued"))
    destination = Trainer(destination_cfg)
    destination.load_weights(source.ckpt_path)

    assert destination.step == 0
    assert destination.data.state_dict()["cursor"] == 0
    source_state = source.model.state_dict()
    destination_state = destination.model.state_dict()
    assert source_state.keys() == destination_state.keys()
    assert all(
        np.array_equal(source_state[key].detach().numpy(), destination_state[key].detach().numpy())
        for key in source_state
    )


def test_resume_rejects_changed_data_fingerprint(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg.update(max_steps=1, data_fingerprint="first")
    source = Trainer(cfg)
    source.train_steps()

    changed = dict(
        cfg,
        out_dir=str(tmp_path / "changed"),
        data_fingerprint="different",
    )
    destination = Trainer(changed)
    with pytest.raises(ValueError, match="data_fingerprint"):
        destination.load_ckpt(source.ckpt_path)


def test_cosine_schedule():
    assert cosine_lr(0, 1.0, 10, 100) < 0.2
    assert abs(cosine_lr(10, 1.0, 10, 100) - 1.0) < 0.01
    assert cosine_lr(99, 1.0, 10, 100) < 0.2
    assert cosine_lr(150, 1.0, 10, 100) == 0.1
