from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from experiment.ledger import RunLedger
from train.trainer import (
    ProvenanceError,
    Trainer,
    capture_rng_state,
    restore_rng_state,
    train,
    validate_run_start,
)


def _write_pickle_marker(path: str) -> None:
    Path(path).write_text("unsafe checkpoint code executed\n")


class _CheckpointPickleGadget:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (_write_pickle_marker, (str(self.marker),))


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write_training_inputs(root: Path, *, n: int = 512) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    tokens = (np.arange(n) % 61).astype(np.uint16)
    weights = np.ones(n, dtype=np.uint8)
    token_path = root / "train.bin"
    weights_path = root / "train.weights.bin"
    tokens.tofile(token_path)
    weights.tofile(weights_path)
    return token_path, weights_path


def _config(
    out_dir: Path,
    token_path: Path,
    weights_path: Path,
    *,
    max_steps: int = 3,
) -> dict:
    architecture = {
        "n_layer": 1,
        "n_head": 1,
        "d_model": 16,
        "ctx": 8,
        "vocab_size": 64,
    }
    return {
        "run_id": "exact-resume-fixture",
        "freeze_sha256": _digest("freeze"),
        "config_sha256": _digest("config"),
        "source_tree_sha256": _digest("source-tree"),
        "pair_fingerprint": _digest("pair"),
        "model": architecture,
        "architecture": architecture,
        "condition": "dense",
        "train_bin": str(token_path),
        "train_weights": str(weights_path),
        "micro_batch_size": 2,
        "tokens_per_step": 16,
        "max_steps": max_steps,
        "total_tokens": max_steps * 16,
        "actual_raw_positions": max_steps * 16,
        "lr": 1e-3,
        "weight_decay": 0.1,
        "warmup_steps": 1,
        "seed": 17,
        "initialization_seed": 17,
        "data_seed": 17,
        "optimizer": {
            "name": "adamw",
            "lr": 1e-3,
            "betas": [0.9, 0.95],
            "epsilon": 1e-8,
            "weight_decay": 0.1,
            "gradient_clip": 1.0,
        },
        "scheduler": {
            "name": "cosine",
            "warmup_steps": 1,
            "minimum_learning_rate_fraction": 0.1,
        },
        "packing": {
            "format": "packed-u16-v1",
            "context_length": 8,
            "boundary_policy": "fixture",
        },
        "out_dir": str(out_dir),
        "device": "cpu",
        "log_every": 1,
        "eval_every": 99,
        "snap_frac": 1.0,
        "ckpt_minutes": 999,
    }


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert torch.equal(left.cpu(), right.cpu())
        return
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert left.dtype == right.dtype
        assert np.array_equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
        return
    assert left == right


def _tree_snapshot(root: Path) -> tuple:
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            rows.append((relative, "dir", None))
        else:
            rows.append((relative, "file", path.read_bytes()))
    return tuple(rows)


def _last_semantic_log_row(trainer: Trainer) -> dict:
    row = json.loads(trainer.log_path.read_text().splitlines()[-1])
    return {
        key: row[key]
        for key in ("step", "loss", "loss_ema", "lr", "epoch")
    }


def test_resume_matches_uninterrupted_next_update(tmp_path):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")

    random.seed(991)
    np.random.seed(991)
    uninterrupted = Trainer(
        _config(tmp_path / "uninterrupted", token_path, weights_path)
    )
    uninterrupted.train_steps()
    uninterrupted_rng = capture_rng_state()
    uninterrupted_log = _last_semantic_log_row(uninterrupted)

    random.seed(991)
    np.random.seed(991)
    interrupted_cfg = _config(
        tmp_path / "interrupted",
        token_path,
        weights_path,
    )
    interrupted = Trainer(interrupted_cfg)
    interrupted.train_steps(2)
    assert interrupted.step == 2

    random.random()
    np.random.random()
    torch.rand(3)

    resumed = train(interrupted_cfg, resume="auto")
    resumed_rng = capture_rng_state()

    _assert_nested_equal(
        uninterrupted.model.state_dict(),
        resumed.model.state_dict(),
    )
    _assert_nested_equal(
        uninterrupted.opt.state_dict(),
        resumed.opt.state_dict(),
    )
    assert uninterrupted.data.state_dict() == resumed.data.state_dict()
    assert uninterrupted.step == resumed.step == 3
    assert uninterrupted.last_step_loss == resumed.last_step_loss
    assert uninterrupted.running_loss == resumed.running_loss
    assert uninterrupted_log == _last_semantic_log_row(resumed)
    _assert_nested_equal(uninterrupted_rng, resumed_rng)


def test_checkpoint_payload_loads_with_weights_only(tmp_path):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")
    trainer = Trainer(_config(tmp_path / "run", token_path, weights_path))
    trainer.train_steps(1)

    state = torch.load(
        trainer.ckpt_path,
        map_location="cpu",
        weights_only=True,
    )

    assert state["schema_version"] == 2
    assert isinstance(state["rng_numpy"], dict)
    assert isinstance(state["rng_numpy"]["keys"], torch.Tensor)


def test_numpy_rng_safe_encoding_restores_cached_gaussian_exactly():
    np.random.seed(7123)
    np.random.normal()
    state = capture_rng_state()
    expected_next = np.random.normal()
    np.random.seed(999)

    restore_rng_state(state)

    assert isinstance(state["rng_numpy"], dict)
    assert np.random.normal() == expected_next


def test_malicious_pickle_checkpoint_is_rejected_without_execution(tmp_path):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")
    cfg = _config(tmp_path / "run", token_path, weights_path)
    trainer = Trainer(cfg)
    trainer.train_steps(1)
    marker = tmp_path / "pickle-gadget-executed"
    torch.save(
        {"payload": _CheckpointPickleGadget(marker)},
        trainer.ckpt_path,
    )
    before = _tree_snapshot(tmp_path)

    with pytest.raises(ProvenanceError, match="checkpoint"):
        validate_run_start(cfg, resume="auto")

    assert not marker.exists()
    assert _tree_snapshot(tmp_path) == before


def test_checkpoint_replacement_during_open_stream_load_is_rejected(
    tmp_path,
    monkeypatch,
):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")
    cfg = _config(tmp_path / "run", token_path, weights_path)
    trainer = Trainer(cfg)
    trainer.train_steps(1)
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(trainer.ckpt_path.read_bytes())
    real_load = torch.load
    real_open = os.open
    opened_nofollow = False
    loaded_from_stream = False

    def record_checkpoint_open(path, flags, *args, **kwargs):
        nonlocal opened_nofollow
        if Path(path) == trainer.ckpt_path:
            opened_nofollow = bool(flags & os.O_NOFOLLOW)
        return real_open(path, flags, *args, **kwargs)

    def replace_path_before_deserialization(source, *args, **kwargs):
        nonlocal loaded_from_stream
        loaded_from_stream = not isinstance(
            source,
            (str, bytes, os.PathLike),
        )
        os.replace(replacement, trainer.ckpt_path)
        return real_load(source, *args, **kwargs)

    monkeypatch.setattr(os, "open", record_checkpoint_open)
    monkeypatch.setattr(torch, "load", replace_path_before_deserialization)

    with pytest.raises(ProvenanceError, match="checkpoint.*changed|loaded safely"):
        validate_run_start(cfg, resume="auto")
    assert opened_nofollow is True
    assert loaded_from_stream is True


@pytest.mark.parametrize(
    "field",
    [
        "freeze_sha256",
        "config_sha256",
        "corpus_sha256",
        "weights_sha256",
        "source_tree_sha256",
        "initialization_sha256",
        "run_id",
        "pair_fingerprint",
        "architecture",
        "initialization_seed",
        "data_seed",
        "optimizer",
        "scheduler",
        "packing",
    ],
)
def test_resume_rejects_identity_drift_before_writing(tmp_path, field):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")
    cfg = _config(tmp_path / "run", token_path, weights_path)
    interrupted = Trainer(cfg)
    interrupted.train_steps(1)

    state = torch.load(interrupted.ckpt_path, map_location="cpu", weights_only=False)
    if field == "run_id":
        state["provenance"][field] = "different-run"
    elif field in {"initialization_seed", "data_seed"}:
        state["provenance"][field] += 1
    elif field == "architecture":
        state["provenance"][field]["ctx"] += 1
    elif field == "optimizer":
        state["provenance"][field]["weight_decay"] = 0.2
    elif field == "scheduler":
        state["provenance"][field]["minimum_learning_rate_fraction"] = 0.2
    elif field == "packing":
        state["provenance"][field]["boundary_policy"] = "drifted"
    else:
        state["provenance"][field] = _digest(f"drift-{field}")
    torch.save(state, interrupted.ckpt_path)
    before = _tree_snapshot(tmp_path)

    with pytest.raises(ProvenanceError, match=field.replace("_", " ")):
        validate_run_start(cfg, resume="auto")

    assert _tree_snapshot(tmp_path) == before


def test_incompatible_model_state_fails_before_live_trainer_construction(
    tmp_path,
    monkeypatch,
):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")
    cfg = _config(tmp_path / "run", token_path, weights_path)
    interrupted = Trainer(cfg)
    interrupted.train_steps(1)
    state = torch.load(interrupted.ckpt_path, map_location="cpu", weights_only=False)
    first_name = next(iter(state["model"]))
    state["model"][first_name] = state["model"][first_name][:-1]
    torch.save(state, interrupted.ckpt_path)
    before = _tree_snapshot(tmp_path)
    constructed = False

    class ForbiddenTrainer:
        def __init__(self, *_args, **_kwargs):
            nonlocal constructed
            constructed = True
            raise AssertionError("incompatible state reached live Trainer")

    monkeypatch.setattr("train.trainer.Trainer", ForbiddenTrainer)

    with pytest.raises(ProvenanceError, match="model tensor|architecture"):
        train(cfg, resume="auto")

    assert constructed is False
    assert _tree_snapshot(tmp_path) == before


def test_resume_rejects_scheduler_position_drift_before_writing(tmp_path):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")
    cfg = _config(tmp_path / "run", token_path, weights_path)
    interrupted = Trainer(cfg)
    interrupted.train_steps(1)
    state = torch.load(interrupted.ckpt_path, map_location="cpu", weights_only=False)
    state["scheduler"]["last_lr"] = [0.0, 0.0]
    torch.save(state, interrupted.ckpt_path)
    before = _tree_snapshot(tmp_path)

    with pytest.raises(ProvenanceError, match="scheduler.*learning rate"):
        validate_run_start(cfg, resume="auto")

    assert _tree_snapshot(tmp_path) == before


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("cursor", "data cursor"),
        ("raw_positions", "data raw positions"),
    ],
)
def test_resume_rejects_data_position_drift_before_writing(
    tmp_path,
    field,
    match,
):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")
    cfg = _config(tmp_path / "run", token_path, weights_path)
    interrupted = Trainer(cfg)
    interrupted.train_steps(1)
    state = torch.load(interrupted.ckpt_path, map_location="cpu", weights_only=False)
    state["data"][field] = 0
    torch.save(state, interrupted.ckpt_path)
    before = _tree_snapshot(tmp_path)

    with pytest.raises(ProvenanceError, match=match):
        validate_run_start(cfg, resume="auto")

    assert _tree_snapshot(tmp_path) == before


def test_resume_rejects_optimizer_hyperparameter_drift_before_writing(
    tmp_path,
):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")
    cfg = _config(tmp_path / "run", token_path, weights_path)
    interrupted = Trainer(cfg)
    interrupted.train_steps(1)
    state = torch.load(interrupted.ckpt_path, map_location="cpu", weights_only=False)
    state["opt"]["param_groups"][0]["weight_decay"] = 0.2
    torch.save(state, interrupted.ckpt_path)
    before = _tree_snapshot(tmp_path)

    with pytest.raises(ProvenanceError, match="optimizer weight decay"):
        validate_run_start(cfg, resume="auto")

    assert _tree_snapshot(tmp_path) == before


def test_resume_rejects_malformed_available_mps_rng_before_writing(
    tmp_path,
    monkeypatch,
):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")
    cfg = _config(tmp_path / "run", token_path, weights_path)
    interrupted = Trainer(cfg)
    interrupted.train_steps(1)
    state = torch.load(interrupted.ckpt_path, map_location="cpu", weights_only=False)
    state["rng_mps"] = torch.zeros(1, dtype=torch.uint8)
    torch.save(state, interrupted.ckpt_path)
    monkeypatch.setattr("train.trainer.mps_rng_available", lambda: True)
    monkeypatch.setattr(
        torch.mps,
        "get_rng_state",
        lambda: torch.zeros(32, dtype=torch.uint8),
    )
    before = _tree_snapshot(tmp_path)

    with pytest.raises(ProvenanceError, match="MPS RNG state"):
        validate_run_start(cfg, resume="auto")

    assert _tree_snapshot(tmp_path) == before


def test_resume_rejects_malformed_available_cuda_rng_before_writing(
    tmp_path,
    monkeypatch,
):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")
    cfg = _config(tmp_path / "run", token_path, weights_path)
    interrupted = Trainer(cfg)
    interrupted.train_steps(1)
    state = torch.load(interrupted.ckpt_path, map_location="cpu", weights_only=False)
    state["rng_cuda"] = [torch.zeros(1, dtype=torch.uint8)]
    torch.save(state, interrupted.ckpt_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: [torch.zeros(32, dtype=torch.uint8)],
    )
    before = _tree_snapshot(tmp_path)

    with pytest.raises(ProvenanceError, match="CUDA RNG state"):
        validate_run_start(cfg, resume="auto")

    assert _tree_snapshot(tmp_path) == before


def test_resume_accepts_torch_26_adamw_group_schema(tmp_path):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")
    cfg = _config(
        tmp_path / "run",
        token_path,
        weights_path,
        max_steps=2,
    )
    interrupted = Trainer(cfg)
    interrupted.train_steps(1)
    state = torch.load(interrupted.ckpt_path, map_location="cpu", weights_only=False)
    for group in state["opt"]["param_groups"]:
        group.pop("decoupled_weight_decay", None)
    torch.save(state, interrupted.ckpt_path)

    resumed = train(cfg, resume="auto")

    assert resumed.step == 2


def test_training_publishes_complete_immutable_lifecycle(tmp_path):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")
    out_root = tmp_path / "outputs"
    out_root.mkdir()
    cfg = _config(
        out_root / "runs" / "complete",
        token_path,
        weights_path,
        max_steps=1,
    )
    cfg.update(out_root=str(out_root), ledger_root=str(out_root))

    trainer = train(cfg, resume="none")

    ledger = RunLedger(out_root, cfg["run_id"])
    assert [event.event_type for event in ledger.events()] == [
        "planned",
        "preflight_passed",
        "launch_requested",
        "started",
        "checkpointed",
        "completed",
    ]
    assert ledger.summary()["exit_status"] == 0
    assert trainer.step == 1


def test_training_failure_is_retained_with_nonzero_ledger_status(
    tmp_path,
    monkeypatch,
):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")
    out_root = tmp_path / "outputs"
    out_root.mkdir()
    cfg = _config(
        out_root / "runs" / "failed",
        token_path,
        weights_path,
        max_steps=1,
    )
    cfg.update(out_root=str(out_root), ledger_root=str(out_root))

    def fail_after_start(self, n_steps=None):
        raise RuntimeError("simulated worker failure")

    monkeypatch.setattr(Trainer, "train_steps", fail_after_start)

    with pytest.raises(RuntimeError, match="worker failure"):
        train(cfg, resume="none")

    summary = RunLedger(out_root, cfg["run_id"]).summary()
    assert summary["status"] == "failed"
    assert summary["failure_count"] == 1
    assert summary["exit_status"] != 0


def test_resume_recovers_checkpoint_published_before_ledger_event(tmp_path):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")
    out_root = tmp_path / "outputs"
    out_root.mkdir()
    cfg = _config(
        out_root / "runs" / "recovered",
        token_path,
        weights_path,
        max_steps=2,
    )
    cfg.update(out_root=str(out_root), ledger_root=str(out_root))
    start = validate_run_start(cfg, resume="none")
    details = {
        "provenance_sha256": start.provenance.provenance_sha256,
    }
    ledger = RunLedger(out_root, cfg["run_id"])
    ledger.append("planned", event_id="plan", details=details)
    ledger.append(
        "preflight_passed",
        event_id="preflight",
        details=details,
    )
    ledger.append("launch_requested", event_id="launch", details=details)
    ledger.append("started", event_id="start", details=details)

    interrupted = Trainer(cfg)
    interrupted.train_steps(1)
    assert ledger.events()[-1].event_type == "started"

    resumed = train(cfg, resume="auto")

    assert resumed.step == 2
    events = ledger.events()
    assert [event.event_type for event in events[-4:]] == [
        "checkpointed",
        "resumed",
        "checkpointed",
        "completed",
    ]
    assert events[-4].details["imported"] is True
    assert events[-4].details["step"] == 1


def test_training_rejects_existing_ledger_from_different_provenance(tmp_path):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")
    out_root = tmp_path / "outputs"
    out_root.mkdir()
    cfg = _config(
        out_root / "runs" / "bound",
        token_path,
        weights_path,
        max_steps=1,
    )
    cfg.update(out_root=str(out_root), ledger_root=str(out_root))
    first_start = validate_run_start(cfg, resume="none")
    details = {
        "provenance_sha256": first_start.provenance.provenance_sha256,
    }
    ledger = RunLedger(out_root, cfg["run_id"])
    for event_type in (
        "planned",
        "preflight_passed",
        "launch_requested",
        "started",
        "failed",
    ):
        ledger.append(event_type, details=details)

    drifted = {**cfg, "data_seed": cfg["data_seed"] + 1}
    before = _tree_snapshot(tmp_path)

    with pytest.raises(ProvenanceError, match="ledger.*provenance"):
        train(drifted, resume="auto")

    assert _tree_snapshot(tmp_path) == before
    assert not Path(cfg["out_dir"]).exists()


@pytest.mark.parametrize(
    "event_types",
    [
        ("planned",),
        ("planned", "preflight_passed"),
        ("planned", "preflight_passed", "launch_requested"),
        (
            "planned",
            "preflight_passed",
            "launch_requested",
            "started",
        ),
    ],
)
def test_training_recovers_interrupted_initial_ledger_publication(
    tmp_path,
    event_types,
):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")
    out_root = tmp_path / "outputs"
    out_root.mkdir()
    cfg = _config(
        out_root / "runs" / "initial-ledger",
        token_path,
        weights_path,
        max_steps=1,
    )
    cfg.update(out_root=str(out_root), ledger_root=str(out_root))
    start = validate_run_start(cfg, resume="none")
    details = {
        "provenance_sha256": start.provenance.provenance_sha256,
    }
    ledger = RunLedger(out_root, cfg["run_id"])
    for event_type in event_types:
        ledger.append(event_type, details=details)

    trainer = train(cfg, resume="auto")

    assert trainer.step == 1
    assert ledger.events()[-1].event_type == "completed"


@pytest.mark.parametrize("failed_after_checkpoint", [False, True])
def test_resume_imports_checkpoint_newer_than_ledger(
    tmp_path,
    failed_after_checkpoint,
):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")
    out_root = tmp_path / "outputs"
    out_root.mkdir()
    cfg = _config(
        out_root / "runs" / "newer-checkpoint",
        token_path,
        weights_path,
        max_steps=3,
    )
    cfg.update(out_root=str(out_root), ledger_root=str(out_root))
    start = validate_run_start(cfg, resume="none")
    details = {
        "provenance_sha256": start.provenance.provenance_sha256,
    }
    interrupted = Trainer(cfg)
    interrupted.train_steps(2)
    ledger = RunLedger(out_root, cfg["run_id"])
    for event_type in (
        "planned",
        "preflight_passed",
        "launch_requested",
        "started",
    ):
        ledger.append(event_type, details=details)
    ledger.append("checkpointed", details={**details, "step": 1})
    if failed_after_checkpoint:
        ledger.append("failed", details=details)

    resumed = train(cfg, resume="auto")

    assert resumed.step == 3
    imported = [
        event
        for event in ledger.events()
        if event.event_type == "checkpointed"
        and event.details.get("imported") is True
    ]
    assert len(imported) == 1
    assert imported[0].details["step"] == 2


def test_resume_rejects_checkpoint_rollback_behind_ledger(tmp_path):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")
    out_root = tmp_path / "outputs"
    out_root.mkdir()
    cfg = _config(
        out_root / "runs" / "rollback",
        token_path,
        weights_path,
        max_steps=3,
    )
    cfg.update(out_root=str(out_root), ledger_root=str(out_root))
    start = validate_run_start(cfg, resume="none")
    details = {
        "provenance_sha256": start.provenance.provenance_sha256,
    }
    interrupted = Trainer(cfg)
    interrupted.train_steps(1)
    ledger = RunLedger(out_root, cfg["run_id"])
    for event_type in (
        "planned",
        "preflight_passed",
        "launch_requested",
        "started",
    ):
        ledger.append(event_type, details=details)
    ledger.append("checkpointed", details={**details, "step": 2})
    before = _tree_snapshot(tmp_path)

    with pytest.raises(ProvenanceError, match="checkpoint.*behind|rollback"):
        train(cfg, resume="auto")

    assert _tree_snapshot(tmp_path) == before


def test_failed_ledger_with_checkpoint_history_requires_exact_resume(
    tmp_path,
):
    token_path, weights_path = _write_training_inputs(tmp_path / "inputs")
    out_root = tmp_path / "outputs"
    out_root.mkdir()
    cfg = _config(
        out_root / "runs" / "missing-checkpoint",
        token_path,
        weights_path,
        max_steps=3,
    )
    cfg.update(out_root=str(out_root), ledger_root=str(out_root))
    start = validate_run_start(cfg, resume="none")
    details = {
        "provenance_sha256": start.provenance.provenance_sha256,
    }
    ledger = RunLedger(out_root, cfg["run_id"])
    for event_type in (
        "planned",
        "preflight_passed",
        "launch_requested",
        "started",
    ):
        ledger.append(event_type, details=details)
    ledger.append("checkpointed", details={**details, "step": 2})
    ledger.append("failed", details=details)
    before = _tree_snapshot(tmp_path)

    with pytest.raises(ProvenanceError, match="exact checkpoint resume"):
        train(cfg, resume="auto")

    assert _tree_snapshot(tmp_path) == before
    assert not Path(cfg["out_dir"]).exists()
