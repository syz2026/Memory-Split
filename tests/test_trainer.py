import json

import numpy as np
import pytest
import torch

from train.model import GPT, GPTConfig
from train.trainer import Trainer, cosine_lr


def write_corpus(tmp_path, n=40192, mask_frac=0.1, seed=0):
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
        "train_mask": str(mp) if mp is not None else None,
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
    xb, _ = b.data.next_batch()
    xa, _ = a.data.next_batch()
    assert (xa == xb).all()


def test_checkpoint_rejects_world_size_mismatch(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["max_steps"] = 1
    trainer = Trainer(cfg)
    trainer.save_ckpt()
    state = torch.load(trainer.ckpt_path, weights_only=False)
    state["world_size"] = 2
    mismatched = tmp_path / "world-size-mismatch.pt"
    torch.save(state, mismatched)

    with pytest.raises(ValueError, match="world size"):
        trainer.load_ckpt(mismatched)


def test_checkpoint_rejects_data_provenance_mismatch(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["max_steps"] = 1
    trainer = Trainer(cfg)
    trainer.save_ckpt()
    state = torch.load(trainer.ckpt_path, weights_only=False)
    state["data_provenance"] = {"format_version": 2, "tokens": "different"}
    mismatched = tmp_path / "provenance-mismatch.pt"
    torch.save(state, mismatched)

    with pytest.raises(ValueError, match="provenance"):
        trainer.load_ckpt(mismatched)


def test_checkpoint_rejects_training_config_mismatch(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["max_steps"] = 1
    trainer = Trainer(cfg)
    trainer.save_ckpt()
    changed_cfg = dict(
        cfg,
        lr=cfg["lr"] * 2,
        out_dir=str(tmp_path / "changed-out"),
    )
    changed = Trainer(changed_cfg)

    with pytest.raises(ValueError, match="config"):
        changed.load_ckpt(trainer.ckpt_path)


def test_checkpoint_rejects_changed_sidecar_content(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["max_steps"] = 1
    trainer = Trainer(cfg)
    trainer.save_ckpt()
    changed_mask = tmp_path / "changed.mask.bin"
    mask = np.fromfile(mp, dtype=np.uint8)
    mask[0] ^= 1
    mask.tofile(changed_mask)
    changed = Trainer(
        dict(
            cfg,
            train_mask=str(changed_mask),
            out_dir=str(tmp_path / "changed-sidecar-out"),
        )
    )

    with pytest.raises(ValueError, match="provenance"):
        changed.load_ckpt(trainer.ckpt_path)


def test_checkpoint_records_rng_state_by_rank(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["max_steps"] = 1
    trainer = Trainer(cfg)
    trainer.save_ckpt()

    state = torch.load(trainer.ckpt_path, weights_only=False)
    assert len(state["rng_by_rank"]) == 1
    assert set(state["rng_by_rank"][0]) == {
        "python",
        "numpy",
        "torch",
        "cuda",
    }
    assert torch.equal(state["rng_by_rank"][0]["torch"], torch.get_rng_state())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tokens_per_step", 257, "divisible"),
        ("micro_batch_size", 0, "micro_batch_size"),
    ],
)
def test_invalid_update_configs_are_rejected(tmp_path, field, value, message):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg[field] = value

    with pytest.raises(ValueError, match=message):
        Trainer(cfg)


def test_total_tokens_must_be_whole_optimizer_updates(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg.pop("max_steps")
    cfg["total_tokens"] = cfg["tokens_per_step"] * 2 + 1

    with pytest.raises(ValueError, match="total_tokens"):
        Trainer(cfg)


def test_max_steps_must_exactly_match_total_tokens(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["max_steps"] = 2
    cfg["total_tokens"] = cfg["tokens_per_step"] * 3

    with pytest.raises(ValueError, match="max_steps.*total_tokens"):
        Trainer(cfg)


def test_micro_batch_size_is_a_cap_not_a_required_local_row_count(tmp_path):
    bp, mp = write_corpus(tmp_path, n=64)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg.update(
        {
            "model": {
                "n_layer": 1,
                "n_head": 1,
                "d_model": 8,
                "ctx": 4,
                "vocab_size": 256,
            },
            "micro_batch_size": 32,
            "tokens_per_step": 8,
            "max_steps": 1,
        }
    )

    trainer = Trainer(cfg)

    assert trainer.local_sequences == 2
    assert trainer.accum == 1


def test_non_master_rank_does_not_write_snapshot_artifact(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["max_steps"] = 1
    trainer = Trainer(cfg)
    trainer.is_master = False

    trainer.save_snapshot()

    assert not (trainer.out_dir / "snapshots" / "step0000000.pt").exists()


def test_weighted_training_uses_existing_gpt_and_sidecar(tmp_path):
    bp, _ = write_corpus(tmp_path, n=1024)
    weights_path = tmp_path / "train.weights.bin"
    np.zeros(1024, dtype=np.uint8).tofile(weights_path)
    cfg = base_cfg(tmp_path, bp, None)
    cfg.update(
        {
            "model": {
                "n_layer": 2,
                "n_head": 2,
                "d_model": 64,
                "ctx": 64,
                "vocab_size": 256,
            },
            "micro_batch_size": 2,
            "tokens_per_step": 2 * 64,
            "max_steps": 1,
            "train_weights": str(weights_path),
        }
    )
    trainer = Trainer(cfg)
    assert type(trainer.model) is GPT
    assert trainer.data.target_weights is not None
    assert trainer.train_steps(1) == 0.0
    checkpoint = torch.load(trainer.ckpt_path, weights_only=False)
    assert set(checkpoint) == {
        "model",
        "opt",
        "data",
        "step",
        "rng_by_rank",
        "cfg",
        "checkpoint_version",
        "config_fingerprint",
        "world_size",
        "data_provenance",
    }


def test_gpt_can_return_weighted_local_loss_sum_for_global_normalization():
    torch.manual_seed(3)
    model = GPT(
        GPTConfig(
            n_layer=1,
            n_head=1,
            d_model=16,
            ctx=4,
            vocab_size=32,
        )
    )
    x = torch.randint(0, 32, (2, 4))
    targets = torch.randint(0, 32, (2, 4))
    weights = torch.linspace(0.25, 1.0, 8).reshape(2, 4)

    _, mean_loss = model(x, targets, target_weights=weights)
    _, loss_sum = model(
        x,
        targets,
        target_weights=weights,
        loss_reduction="sum",
    )

    assert torch.allclose(loss_sum, mean_loss * targets.numel())


def test_single_process_update_uses_final_partial_microbatch(tmp_path):
    bp, _ = write_corpus(tmp_path, n=1000)
    weights_path = tmp_path / "train.weights.bin"
    np.ones(1000, dtype=np.uint8).tofile(weights_path)
    cfg = base_cfg(tmp_path, bp, None)
    cfg.update(
        {
            "model": {
                "n_layer": 1,
                "n_head": 1,
                "d_model": 16,
                "ctx": 8,
                "vocab_size": 256,
            },
            "micro_batch_size": 3,
            "tokens_per_step": 5 * 8,
            "max_steps": 1,
            "train_weights": str(weights_path),
        }
    )

    trainer = Trainer(cfg)
    trainer.train_steps(1)

    assert trainer.accum == 2
    assert trainer.data.global_cursor == 40


def test_cosine_schedule():
    assert cosine_lr(0, 1.0, 10, 100) < 0.2
    assert abs(cosine_lr(10, 1.0, 10, 100) - 1.0) < 0.01
    assert cosine_lr(99, 1.0, 10, 100) < 0.2
    assert cosine_lr(150, 1.0, 10, 100) == 0.1
