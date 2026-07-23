import copy
import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np
import pytest
import torch

import train.safeio as safeio
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


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fresh_launch_refuses_even_an_empty_existing_output_before_mutation(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    output = tmp_path / "out"
    output.mkdir()
    marker = output / "owner.txt"
    marker.write_text("unchanged")

    with pytest.raises(FileExistsError, match="fresh.*output"):
        Trainer(cfg)

    assert marker.read_text() == "unchanged"
    assert set(output.iterdir()) == {marker}


def test_auto_resume_requires_default_checkpoint_before_output_mutation(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    output = tmp_path / "out"
    output.mkdir()
    marker = output / "owner.txt"
    marker.write_text("unchanged")

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        Trainer(cfg, resume="auto")

    assert marker.read_text() == "unchanged"
    assert set(output.iterdir()) == {marker}


def test_auto_resume_never_starts_fresh_when_output_is_absent(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)

    with pytest.raises(FileNotFoundError, match="checkpoint|missing"):
        Trainer(cfg, resume="auto")

    assert not Path(cfg["out_dir"]).exists()


def test_default_auto_resume_loads_existing_checkpoint_without_rewriting_config(
    tmp_path,
):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["max_steps"] = 2
    first = Trainer(cfg)
    first.train_steps(1)
    config_before = (first.out_dir / "config.yaml").read_bytes()
    first.close()

    resumed = Trainer(cfg, resume="auto")

    assert resumed.step == 1
    assert (resumed.out_dir / "config.yaml").read_bytes() == config_before
    resumed.close()


def test_external_resume_requires_matching_sha_before_creating_output(tmp_path):
    bp, mp = write_corpus(tmp_path)
    source_cfg = base_cfg(tmp_path, bp, mp)
    source_cfg["max_steps"] = 2
    source = Trainer(source_cfg)
    source.train_steps(1)
    checkpoint = source.ckpt_path
    source.close()
    resumed_cfg = dict(source_cfg, out_dir=str(tmp_path / "resumed"))

    with pytest.raises(ValueError, match="resume_sha256"):
        Trainer(
            resumed_cfg,
            resume="auto",
            resume_path=checkpoint,
        )
    assert not (tmp_path / "resumed").exists()

    with pytest.raises(ValueError, match="SHA-256"):
        Trainer(
            resumed_cfg,
            resume="auto",
            resume_path=checkpoint,
            resume_sha256="0" * 64,
        )
    assert not (tmp_path / "resumed").exists()

    resumed = Trainer(
        resumed_cfg,
        resume="auto",
        resume_path=checkpoint,
        resume_sha256=file_sha256(checkpoint),
    )
    assert resumed.step == 1
    assert resumed.data.global_cursor == source_cfg["tokens_per_step"]
    resumed.close()


def test_external_resume_rejects_symlinked_checkpoint_without_output_mutation(
    tmp_path,
):
    bp, mp = write_corpus(tmp_path)
    source_cfg = base_cfg(tmp_path, bp, mp)
    source_cfg["max_steps"] = 1
    source = Trainer(source_cfg)
    source.save_ckpt()
    linked = tmp_path / "linked-checkpoint.pt"
    linked.symlink_to(source.ckpt_path)
    resumed_cfg = dict(source_cfg, out_dir=str(tmp_path / "resumed"))

    with pytest.raises(ValueError, match="symlink|unsafe"):
        Trainer(
            resumed_cfg,
            resume="auto",
            resume_path=linked,
            resume_sha256=file_sha256(source.ckpt_path),
        )

    assert not (tmp_path / "resumed").exists()
    source.close()


def test_external_resume_hashes_and_loads_exactly_the_same_descriptor(
    tmp_path,
    monkeypatch,
):
    bp, mp = write_corpus(tmp_path, n=64)
    source_cfg = base_cfg(tmp_path, bp, mp)
    source_cfg.update(
        {
            "model": {
                "n_layer": 1,
                "n_head": 1,
                "d_model": 8,
                "ctx": 4,
                "vocab_size": 256,
            },
            "tokens_per_step": 8,
            "micro_batch_size": 2,
            "max_steps": 2,
        }
    )
    source = Trainer(source_cfg)
    source.train_steps(1)
    checkpoint = source.ckpt_path
    expected_sha256 = file_sha256(checkpoint)
    source.close()
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"not a checkpoint")
    real_read = safeio._read_and_hash_fd
    swapped = False

    def replace_during_read(fd):
        nonlocal swapped
        if not swapped:
            os.replace(replacement, checkpoint)
            swapped = True
        return real_read(fd)

    monkeypatch.setattr(safeio, "_read_and_hash_fd", replace_during_read)
    resumed_cfg = dict(source_cfg, out_dir=str(tmp_path / "resumed-pinned"))
    resumed = Trainer(
        resumed_cfg,
        resume="auto",
        resume_path=checkpoint,
        resume_sha256=expected_sha256,
    )

    assert resumed.step == 1
    assert checkpoint.read_bytes() == b"not a checkpoint"
    resumed.close()


def test_output_rejects_symlinked_parent_component_before_mutation(tmp_path):
    bp, mp = write_corpus(tmp_path)
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["out_dir"] = str(linked_parent / "out")

    with pytest.raises(ValueError, match="symlink|unsafe"):
        Trainer(cfg)

    assert not (real_parent / "out").exists()


def test_durable_config_checkpoint_and_snapshot_fsync_files_and_directories(
    tmp_path,
    monkeypatch,
):
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
            "tokens_per_step": 8,
            "micro_batch_size": 2,
            "max_steps": 1,
        }
    )
    real_fsync = safeio.os.fsync
    fsync_kinds = []

    def recording_fsync(fd):
        fsync_kinds.append(
            "directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
        )
        return real_fsync(fd)

    monkeypatch.setattr(safeio.os, "fsync", recording_fsync)
    trainer = Trainer(cfg)
    trainer.save_snapshot()
    trainer.save_ckpt()

    assert "file" in fsync_kinds
    assert "directory" in fsync_kinds
    assert not list(trainer.out_dir.rglob("*.tmp-*"))
    trainer.close()


def test_checkpoint_and_snapshot_writes_reject_symlink_swaps(tmp_path):
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
            "tokens_per_step": 8,
            "micro_batch_size": 2,
            "max_steps": 1,
        }
    )
    trainer = Trainer(cfg)
    checkpoint_target = tmp_path / "checkpoint-target"
    checkpoint_target.write_bytes(b"unchanged")
    trainer.ckpt_path.symlink_to(checkpoint_target)

    with pytest.raises(ValueError, match="symlink|unsafe"):
        trainer.save_ckpt()
    assert checkpoint_target.read_bytes() == b"unchanged"
    trainer.ckpt_path.unlink()

    snapshots = trainer.out_dir / "snapshots"
    moved = trainer.out_dir / "moved-snapshots"
    snapshots.rename(moved)
    snapshots.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="changed|unsafe"):
        trainer.save_snapshot()
    trainer.close()


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
    b = Trainer(
        cfg2,
        resume="auto",
        resume_path=a.ckpt_path,
        resume_sha256=file_sha256(a.ckpt_path),
    )
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

    resumed_cfg = dict(cfg, out_dir=str(tmp_path / "world-size-resume"))
    with pytest.raises(ValueError, match="world size"):
        Trainer(
            resumed_cfg,
            resume="auto",
            resume_path=mismatched,
            resume_sha256=file_sha256(mismatched),
        )


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

    resumed_cfg = dict(cfg, out_dir=str(tmp_path / "provenance-resume"))
    with pytest.raises(ValueError, match="provenance"):
        Trainer(
            resumed_cfg,
            resume="auto",
            resume_path=mismatched,
            resume_sha256=file_sha256(mismatched),
        )


def test_checkpoint_rejects_nested_data_provenance_numeric_type_drift(tmp_path):
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
            "tokens_per_step": 8,
            "micro_batch_size": 2,
            "max_steps": 1,
        }
    )
    trainer = Trainer(cfg)
    trainer.save_ckpt()
    state = torch.load(trainer.ckpt_path, weights_only=False)
    state["data_provenance"] = copy.deepcopy(state["data_provenance"])
    byte_count = state["data_provenance"]["tokens"]["bytes"]
    state["data_provenance"]["tokens"]["bytes"] = float(byte_count)
    drifted = tmp_path / "provenance-type-drift.pt"
    torch.save(state, drifted)

    with pytest.raises(ValueError, match="provenance"):
        trainer.load_ckpt(drifted, sha256=file_sha256(drifted))
    trainer.close()


def test_checkpoint_rejects_adamw_parameter_step_drift(tmp_path):
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
            "tokens_per_step": 8,
            "micro_batch_size": 2,
            "max_steps": 2,
        }
    )
    trainer = Trainer(cfg)
    trainer.train_steps(1)
    state = torch.load(trainer.ckpt_path, weights_only=False)
    assert state["step"] == 1
    parameter_state = next(iter(state["opt"]["state"].values()))
    original_step = parameter_state["step"]
    parameter_state["step"] = original_step + 1
    assert parameter_state["step"].dtype == original_step.dtype
    assert parameter_state["step"].device == original_step.device
    drifted = tmp_path / "adamw-step-drift.pt"
    torch.save(state, drifted)

    with pytest.raises(ValueError, match="AdamW step"):
        trainer.load_ckpt(drifted, sha256=file_sha256(drifted))
    trainer.close()


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
    with pytest.raises(ValueError, match="config"):
        Trainer(
            changed_cfg,
            resume="auto",
            resume_path=trainer.ckpt_path,
            resume_sha256=file_sha256(trainer.ckpt_path),
        )


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
    changed_cfg = dict(
        cfg,
        train_mask=str(changed_mask),
        out_dir=str(tmp_path / "changed-sidecar-out"),
    )
    with pytest.raises(ValueError, match="provenance"):
        Trainer(
            changed_cfg,
            resume="auto",
            resume_path=trainer.ckpt_path,
            resume_sha256=file_sha256(trainer.ckpt_path),
        )


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
    "case",
    [
        "top_extra",
        "version_bool",
        "world_bool",
        "step_bool",
        "cfg_extra",
        "model_extra",
        "optimizer_extra",
        "data_extra",
        "data_version_bool",
        "cursor_bool",
        "epoch_bool",
        "rng_extra",
        "cpu_cuda_rng",
    ],
)
def test_checkpoint_schema_is_exact_and_rejects_bool_integers(tmp_path, case):
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
            "tokens_per_step": 8,
            "micro_batch_size": 2,
            "max_steps": 1,
        }
    )
    trainer = Trainer(cfg)
    trainer.save_ckpt()
    state = torch.load(trainer.ckpt_path, weights_only=False)
    assert state["checkpoint_version"] == 3

    if case == "top_extra":
        state["unexpected"] = None
    elif case == "version_bool":
        state["checkpoint_version"] = True
    elif case == "world_bool":
        state["world_size"] = True
    elif case == "step_bool":
        state["step"] = False
    elif case == "cfg_extra":
        state["cfg"]["unexpected"] = None
    elif case == "model_extra":
        state["model"]["unexpected"] = torch.tensor(0)
    elif case == "optimizer_extra":
        state["opt"]["unexpected"] = None
    elif case == "data_extra":
        state["data"]["unexpected"] = None
    elif case == "data_version_bool":
        state["data"]["format_version"] = True
    elif case == "cursor_bool":
        state["data"]["cursor"] = False
    elif case == "epoch_bool":
        state["data"]["epoch"] = False
    elif case == "rng_extra":
        state["rng_by_rank"][0]["unexpected"] = None
    elif case == "cpu_cuda_rng":
        state["rng_by_rank"][0]["cuda"] = torch.zeros(8, dtype=torch.uint8)
    else:
        raise AssertionError(case)

    malformed = tmp_path / f"{case}.pt"
    torch.save(state, malformed)
    model_before = {
        name: value.detach().clone()
        for name, value in trainer._raw_model().state_dict().items()
    }
    with pytest.raises(ValueError):
        trainer.load_ckpt(malformed, sha256=file_sha256(malformed))

    assert trainer.step == 0
    assert trainer.data.global_cursor == 0
    assert all(
        torch.equal(value, model_before[name])
        for name, value in trainer._raw_model().state_dict().items()
    )
    trainer.close()


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
