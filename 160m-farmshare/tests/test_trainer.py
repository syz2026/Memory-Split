import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.run_train import main as run_train_main
from train.model import GPT
from train.trainer import (
    ProvenanceError,
    Trainer,
    cosine_lr,
    train,
    validate_run_start,
)


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


def tree_snapshot(root: Path) -> tuple:
    if not root.exists():
        return ()
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            rows.append((relative, "symlink", str(path.readlink())))
        elif path.is_dir():
            rows.append((relative, "dir", None))
        else:
            rows.append((relative, "file", path.read_bytes()))
    return tuple(rows)


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


def test_weighted_training_uses_existing_gpt_and_sidecar(tmp_path):
    bp, _ = write_corpus(tmp_path, n=1000)
    weights_path = tmp_path / "train.weights.bin"
    np.zeros(1000, dtype=np.uint8).tofile(weights_path)
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
    assert {
        "schema_version",
        "provenance",
        "model",
        "opt",
        "data",
        "step",
        "scheduler",
        "rng_python",
        "rng_numpy",
        "rng_torch",
        "rng_cuda",
        "rng_mps",
        "cfg",
        "running_loss",
        "last_step_loss",
    } <= set(checkpoint)
    assert checkpoint["schema_version"] == 2
    assert checkpoint["provenance"]["corpus_sha256"]
    assert checkpoint["provenance"]["weights_sha256"]


def test_cosine_schedule():
    assert cosine_lr(0, 1.0, 10, 100) < 0.2
    assert abs(cosine_lr(10, 1.0, 10, 100) - 1.0) < 0.01
    assert cosine_lr(99, 1.0, 10, 100) < 0.2
    assert cosine_lr(150, 1.0, 10, 100) == 0.1


def _weighted_asset_config(tmp_path):
    token_path, _ = write_corpus(tmp_path, n=1000)
    weights_path = tmp_path / "train.weights.bin"
    np.ones(1000, dtype=np.uint8).tofile(weights_path)
    cfg = base_cfg(tmp_path, token_path, None)
    cfg["train_weights"] = str(weights_path)
    return cfg, token_path, weights_path


@pytest.mark.parametrize(
    ("field", "path_key", "match"),
    [
        ("stream_sha256", "train_bin", "stream SHA-256.*bytes"),
        ("weights_sha256", "train_weights", "weights SHA-256.*bytes"),
    ],
)
def test_validate_run_start_checks_canonical_asset_hashes(
    tmp_path,
    field,
    path_key,
    match,
):
    cfg, _token_path, _weights_path = _weighted_asset_config(tmp_path)
    cfg[field] = "0" * 64

    with pytest.raises(ProvenanceError, match=match):
        validate_run_start(cfg, resume="none")

    assert not Path(cfg["out_dir"]).exists()


@pytest.mark.parametrize(
    ("canonical", "alias", "path_key"),
    [
        ("stream_sha256", "corpus_sha256", "train_bin"),
        ("weights_sha256", "weights_file_sha256", "train_weights"),
    ],
)
def test_validate_run_start_rejects_conflicting_asset_hash_aliases(
    tmp_path,
    canonical,
    alias,
    path_key,
):
    cfg, _token_path, _weights_path = _weighted_asset_config(tmp_path)
    actual = hashlib.sha256(Path(cfg[path_key]).read_bytes()).hexdigest()
    cfg[canonical] = actual
    cfg[alias] = "f" * 64

    with pytest.raises(ProvenanceError, match="conflicting.*SHA-256"):
        validate_run_start(cfg, resume="none")

    assert not Path(cfg["out_dir"]).exists()


def test_validate_run_start_preserves_explicit_generic_hash_aliases(tmp_path):
    cfg, token_path, weights_path = _weighted_asset_config(tmp_path)
    corpus_sha256 = hashlib.sha256(token_path.read_bytes()).hexdigest()
    weights_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    cfg["corpus_sha256"] = corpus_sha256
    cfg["weights_file_sha256"] = weights_sha256

    start = validate_run_start(cfg, resume="none")

    assert start.provenance.corpus_sha256 == corpus_sha256
    assert start.provenance.weights_sha256 == weights_sha256
    assert not Path(cfg["out_dir"]).exists()


@pytest.mark.parametrize(
    "asset",
    ["stream", "weights"],
)
def test_run_train_rechecks_canonical_asset_hashes_before_output(
    tmp_path,
    monkeypatch,
    asset,
):
    cfg, token_path, weights_path = _weighted_asset_config(tmp_path)
    cfg["stream_sha256"] = hashlib.sha256(token_path.read_bytes()).hexdigest()
    cfg["weights_sha256"] = hashlib.sha256(
        weights_path.read_bytes()
    ).hexdigest()
    config_path = tmp_path / "run-config.json"
    config_path.write_text(json.dumps(cfg))
    selected = token_path if asset == "stream" else weights_path
    selected.write_bytes(b"\xff" * selected.stat().st_size)
    constructed = False

    class ForbiddenTrainer:
        def __init__(self, *_args, **_kwargs):
            nonlocal constructed
            constructed = True
            raise AssertionError("mutated input reached Trainer construction")

    monkeypatch.setattr("train.trainer.Trainer", ForbiddenTrainer)

    assert run_train_main(["--config", str(config_path), "--resume", "none"]) == 1
    assert constructed is False
    assert not Path(cfg["out_dir"]).exists()


def test_missing_configured_weights_fail_before_trainer_or_output_mutation(
    tmp_path,
    monkeypatch,
):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    cfg["train_weights"] = str(tmp_path / "missing.weights.bin")
    out_dir = Path(cfg["out_dir"])
    constructed = False

    class ForbiddenTrainer:
        def __init__(self, *_args, **_kwargs):
            nonlocal constructed
            constructed = True
            raise AssertionError("Trainer must not be constructed before preflight")

    monkeypatch.setattr("train.trainer.Trainer", ForbiddenTrainer)

    with pytest.raises(ProvenanceError, match="weights"):
        train(cfg, resume="auto")

    assert constructed is False
    assert not out_dir.exists()


def test_too_small_corpus_fails_before_trainer_or_output_mutation(
    tmp_path,
    monkeypatch,
):
    bp = tmp_path / "tiny.bin"
    mp = tmp_path / "tiny.mask.bin"
    np.arange(20, dtype=np.uint16).tofile(bp)
    np.ones(20, dtype=np.uint8).tofile(mp)
    cfg = base_cfg(tmp_path, bp, mp)
    constructed = False

    class ForbiddenTrainer:
        def __init__(self, *_args, **_kwargs):
            nonlocal constructed
            constructed = True
            raise AssertionError("Trainer must not see an undersized corpus")

    monkeypatch.setattr("train.trainer.Trainer", ForbiddenTrainer)

    with pytest.raises(ProvenanceError, match="corpus.*batch|small"):
        train(cfg, resume="auto")

    assert constructed is False
    assert not Path(cfg["out_dir"]).exists()


def test_resume_none_rejects_nonempty_output_without_mutation(tmp_path):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir()
    (out_dir / "foreign.txt").write_text("preserve me\n")
    before = tree_snapshot(tmp_path)

    with pytest.raises(ProvenanceError, match="resume=none|nonempty"):
        validate_run_start(cfg, resume="none")

    assert tree_snapshot(tmp_path) == before


@pytest.mark.parametrize("present", ["config", "checkpoint"])
def test_partial_output_is_rejected_without_mutation(tmp_path, present):
    bp, mp = write_corpus(tmp_path)
    cfg = base_cfg(tmp_path, bp, mp)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir()
    if present == "config":
        (out_dir / "config.yaml").write_text("{}\n")
    else:
        torch.save({"partial": True}, out_dir / "ckpt.pt")
    before = tree_snapshot(tmp_path)

    with pytest.raises(ProvenanceError, match="partial|checkpoint|config"):
        validate_run_start(cfg, resume="auto")

    assert tree_snapshot(tmp_path) == before


def test_run_train_returns_nonzero_status_after_terminal_failure(
    tmp_path,
    monkeypatch,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")

    def fail_train(_cfg, *, resume):
        raise RuntimeError(f"terminal failure with resume={resume}")

    monkeypatch.setattr("scripts.run_train.train", fail_train)

    assert run_train_main(["--config", str(config_path)]) != 0
