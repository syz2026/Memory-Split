import hashlib
import json
import os
import signal
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from corpusgen.parallel import (
    FixtureRenderer,
    ParallelBuildConfig,
    build_parallel_corpus,
    fixture_catalog,
    render_metadata,
)
from train.data import BatchSlice, PackedShards, synchronized_rank_batch_plan
from train.trainer import Trainer


def tiny_config(root, token_path):
    return {
        "model": {
            "n_layer": 1,
            "n_head": 1,
            "d_model": 8,
            "ctx": 4,
            "vocab_size": 32,
        },
        "train_bin": str(token_path),
        "train_mask": None,
        "micro_batch_size": 2,
        "tokens_per_step": 20,
        "max_steps": 2,
        "lr": 1e-3,
        "warmup_steps": 1,
        "weight_decay": 0.0,
        "seed": 17,
        "out_dir": str(root),
        "device": "cpu",
        "log_every": 1,
        "eval_every": 100,
        "snap_frac": 1.0,
        "ckpt_minutes": 999,
    }


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _ddp_worker(rank, world_size, port, cfg, resume, steps, error_dir):
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
        }
    )
    trainer = None
    try:
        trainer = Trainer(cfg, resume="auto" if resume else "none")
        collective_counts = [None] * world_size
        torch.distributed.all_gather_object(collective_counts, trainer.accum)
        if len(set(collective_counts)) != 1:
            raise AssertionError(f"unequal backward schedules: {collective_counts}")
        loss = trainer.train_steps(steps)
        Path(error_dir, f"result-{rank}.txt").write_text(repr(loss))
    except BaseException:
        Path(error_dir, f"rank-{rank}.txt").write_text(traceback.format_exc())
        raise
    finally:
        if trainer is not None and hasattr(trainer, "close"):
            trainer.close()
        elif torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def run_ddp(
    cfg,
    tmp_path,
    *,
    resume=False,
    steps=None,
    timeout=45,
    world_size=2,
):
    port = _free_port()
    error_dir = tmp_path / f"errors-{port}"
    error_dir.mkdir()
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_ddp_worker,
            args=(
                rank,
                world_size,
                port,
                cfg,
                resume,
                steps,
                str(error_dir),
            ),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
    for process in alive:
        process.join()
    errors = "\n".join(
        path.read_text() for path in sorted(error_dir.glob("rank-*.txt"))
    )
    assert not alive, f"DDP workers deadlocked\n{errors}"
    assert [process.exitcode for process in processes] == [0] * world_size, errors
    return [
        float(Path(error_dir, f"result-{rank}.txt").read_text())
        for rank in range(world_size)
    ]


def assert_models_close(left, right, *, atol):
    assert set(left) == set(right)
    for name in left:
        assert torch.allclose(left[name], right[name], rtol=0, atol=atol), name


def build_weighted_publication(tmp_path):
    catalog = fixture_catalog(record_count=13)
    renderer = FixtureRenderer()
    logical_tokens = sum(
        record.token_length for record in render_metadata(catalog, renderer)
    )
    dense = np.ones(logical_tokens, dtype=np.uint8)
    split90 = ((np.arange(logical_tokens) % 3) != 0).astype(np.uint8)
    sidecar_paths = {}
    for name, weights in (
        ("dense_target_weights", dense),
        ("split90_target_weights", split90),
    ):
        path = tmp_path / f"{name}.bin"
        weights.tofile(path)
        sidecar_paths[name] = path
    publication = tmp_path / "parallel-v2"
    receipt = build_parallel_corpus(
        catalog,
        renderer,
        ParallelBuildConfig(
            lane_weights=(("natural", 1), ("facts", 1), ("reasoning", 1)),
            update_tokens=32,
            allow_fewer_shards=True,
        ),
        publication,
        sidecar_paths=sidecar_paths,
    )
    return publication, receipt


def test_v2_publication_to_descriptor_loader_to_gloo_ddp_matches_reference(
    tmp_path,
):
    publication, receipt = build_weighted_publication(tmp_path)
    ddp_cfg = tiny_config(tmp_path / "ddp-v2", publication)
    ddp_cfg.pop("train_bin")
    ddp_cfg.pop("train_mask")
    ddp_cfg.update(
        {
            "train_corpus": str(publication),
            "sidecar_name": "split90_target_weights",
            "tokens_per_step": 32,
            "max_steps": 1,
        }
    )
    ddp_cfg["model"]["vocab_size"] = 256

    descriptor_loader = PackedShards.from_parallel_corpus(
        publication,
        ctx=4,
        batch_size=2,
        sidecar_name="split90_target_weights",
    )
    rank_batches = []
    for rank in range(2):
        for batch_slice in synchronized_rank_batch_plan(
            global_cursor=0,
            total_sequences=8,
            ctx=4,
            micro_batch_size=2,
            rank=rank,
            world_size=2,
        ):
            assert batch_slice is not None
            rank_batches.append(
                descriptor_loader.weighted_batch_from_slice(batch_slice)
            )
    _, reference_targets, reference_weights = (
        descriptor_loader.weighted_batch_from_slice(
            BatchSlice(global_start=0, sequence_count=8, ctx=4)
        )
    )
    assert torch.equal(
        torch.cat([batch[1] for batch in rank_batches]),
        reference_targets,
    )
    assert torch.equal(
        torch.cat([batch[2] for batch in rank_batches]),
        reference_weights,
    )

    ddp_losses = run_ddp(ddp_cfg, tmp_path, steps=1)
    ddp_state = torch.load(
        Path(ddp_cfg["out_dir"]) / "ckpt.pt",
        weights_only=False,
    )
    single_cfg = dict(ddp_cfg, out_dir=str(tmp_path / "single-v2"))
    single = Trainer(single_cfg)
    single_loss = single.train_steps(1)
    single_state = torch.load(single.ckpt_path, weights_only=False)

    assert receipt["format"] == "memorysplit-parallel-corpus-v2"
    assert ddp_losses == pytest.approx([single_loss, single_loss], abs=2e-6)
    assert ddp_state["data"]["global_cursor"] == 32
    assert_models_close(ddp_state["model"], single_state["model"], atol=2e-6)
    descriptor_loader.close()
    single.close()


def _rank_local_load_failure_worker(
    rank,
    world_size,
    port,
    cfg,
    valid_checkpoint,
    digest,
    result_dir,
):
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
        }
    )
    try:
        checkpoint = valid_checkpoint if rank == 0 else f"{valid_checkpoint}.missing"
        Trainer(
            cfg,
            resume="auto",
            resume_path=checkpoint,
            resume_sha256=digest,
        )
    except BaseException as error:
        Path(result_dir, f"rank-{rank}.txt").write_text(
            f"{type(error).__name__}: {error}"
        )
    else:
        Path(result_dir, f"rank-{rank}.txt").write_text("unexpected success")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _startup_disagreement_worker(
    rank,
    world_size,
    port,
    cfg,
    mismatch,
    alternate_token_path,
    result_dir,
):
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
        }
    )
    local_cfg = dict(cfg)
    if rank == 1 and mismatch == "config":
        local_cfg["lr"] = cfg["lr"] * 2
    elif rank == 1 and mismatch == "data":
        local_cfg["train_bin"] = alternate_token_path
    try:
        Trainer(local_cfg)
    except BaseException as error:
        Path(result_dir, f"rank-{rank}.txt").write_text(
            f"{type(error).__name__}: {error}"
        )
    else:
        Path(result_dir, f"rank-{rank}.txt").write_text("unexpected success")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _signal_request_worker(
    rank,
    world_size,
    port,
    cfg,
    pid_path,
    result_dir,
):
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "MS_RANK_ZERO_PID_FILE": pid_path,
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
        }
    )
    trainer = None
    try:
        trainer = Trainer(cfg)
        Path(result_dir, f"ready-{rank}").write_text("ready")
        release = Path(result_dir, "release")
        deadline = time.monotonic() + 15
        while not release.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not release.exists():
            raise RuntimeError("signal request test was not released")
        serviced = trainer._service_checkpoint_request()
        Path(result_dir, f"serviced-{rank}").write_text(str(serviced))
    except BaseException:
        Path(result_dir, f"rank-{rank}.txt").write_text(traceback.format_exc())
        raise
    finally:
        if trainer is not None:
            trainer.close()
        elif torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.parametrize("mismatch", ["config", "data"])
def test_ranks_must_agree_on_config_and_data_before_output_mutation(
    tmp_path,
    mismatch,
):
    token_path = tmp_path / "tokens.bin"
    (np.arange(80) % 32).astype(np.uint16).tofile(token_path)
    alternate = tmp_path / "alternate.bin"
    ((np.arange(80) + 1) % 32).astype(np.uint16).tofile(alternate)
    cfg = tiny_config(tmp_path / "disagreement", token_path)
    world_size = 2
    port = _free_port()
    result_dir = tmp_path / f"startup-results-{port}"
    result_dir.mkdir()
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_startup_disagreement_worker,
            args=(
                rank,
                world_size,
                port,
                cfg,
                mismatch,
                str(alternate),
                str(result_dir),
            ),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 45
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join()

    messages = [
        (result_dir / f"rank-{rank}.txt").read_text()
        for rank in range(world_size)
    ]
    assert not alive, messages
    assert [process.exitcode for process in processes] == [0, 0]
    assert messages[0] == messages[1]
    expected = (
        "disagree on training config"
        if mismatch == "config"
        else "data provenance"
    )
    assert expected in messages[0]
    assert not Path(cfg["out_dir"]).exists()


def test_rank_local_checkpoint_read_failure_is_coordinated_without_hang(tmp_path):
    token_path = tmp_path / "tokens.bin"
    (np.arange(80) % 32).astype(np.uint16).tofile(token_path)
    source_cfg = tiny_config(tmp_path / "source", token_path)
    run_ddp(source_cfg, tmp_path, steps=1)
    checkpoint = Path(source_cfg["out_dir"]) / "ckpt.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    resumed_cfg = tiny_config(tmp_path / "failed-resume", token_path)
    world_size = 2
    port = _free_port()
    result_dir = tmp_path / f"load-results-{port}"
    result_dir.mkdir()
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_rank_local_load_failure_worker,
            args=(
                rank,
                world_size,
                port,
                resumed_cfg,
                str(checkpoint),
                digest,
                str(result_dir),
            ),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 45
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join()

    messages = [
        (result_dir / f"rank-{rank}.txt").read_text()
        for rank in range(world_size)
    ]
    assert not alive, messages
    assert [process.exitcode for process in processes] == [0, 0]
    assert messages[0] == messages[1]
    assert "coordinated checkpoint load failed" in messages[0]
    assert "rank 1" in messages[0]
    assert not Path(resumed_cfg["out_dir"]).exists()


def test_cpu_gloo_sigusr1_request_is_serviced_by_all_ranks_at_boundary(tmp_path):
    token_path = tmp_path / "tokens.bin"
    (np.arange(80) % 32).astype(np.uint16).tofile(token_path)
    cfg = tiny_config(tmp_path / "signal-ddp", token_path)
    world_size = 2
    port = _free_port()
    result_dir = tmp_path / f"signal-results-{port}"
    result_dir.mkdir()
    pid_path = tmp_path / "rank-zero.pid"
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_signal_request_worker,
            args=(
                rank,
                world_size,
                port,
                cfg,
                str(pid_path),
                str(result_dir),
            ),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if pid_path.exists() and all(
            (result_dir / f"ready-{rank}").exists()
            for rank in range(world_size)
        ):
            break
        time.sleep(0.01)
    ready = pid_path.exists() and all(
        (result_dir / f"ready-{rank}").exists()
        for rank in range(world_size)
    )
    if ready:
        rank_zero_pid = int(pid_path.read_text(encoding="ascii"))
        assert rank_zero_pid == processes[0].pid
        os.kill(rank_zero_pid, signal.SIGUSR1)
        (result_dir / "release").write_text("release")
    else:
        for process in processes:
            process.terminate()

    for process in processes:
        process.join(15)
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join()
    errors = "\n".join(
        path.read_text() for path in sorted(result_dir.glob("rank-*.txt"))
    )

    assert ready, "rank zero did not atomically publish its PID file"
    assert not alive, f"DDP signal workers deadlocked\n{errors}"
    assert [process.exitcode for process in processes] == [0, 0], errors
    assert [
        (result_dir / f"serviced-{rank}").read_text()
        for rank in range(world_size)
    ] == ["True", "True"]
    checkpoint = torch.load(
        Path(cfg["out_dir"]) / "ckpt.pt",
        weights_only=False,
    )
    assert checkpoint["step"] == 0
    assert len(checkpoint["rng_by_rank"]) == world_size


def test_world_size_four_weighted_zero_quota_ranks_match_single_process(tmp_path):
    token_path = tmp_path / "tokens.bin"
    weight_path = tmp_path / "weights.bin"
    (np.arange(64) % 32).astype(np.uint16).tofile(token_path)
    ((np.arange(64) % 3) != 0).astype(np.uint8).tofile(weight_path)
    ddp_cfg = tiny_config(tmp_path / "ddp-four", token_path)
    ddp_cfg.update(
        {
            "micro_batch_size": 1,
            "tokens_per_step": 8,
            "max_steps": 1,
            "train_weights": str(weight_path),
        }
    )

    run_ddp(ddp_cfg, tmp_path, steps=1, timeout=60, world_size=4)
    ddp_state = torch.load(
        Path(ddp_cfg["out_dir"]) / "ckpt.pt",
        weights_only=False,
    )

    single_cfg = dict(ddp_cfg, out_dir=str(tmp_path / "single-weighted"))
    single = Trainer(single_cfg)
    assert single.local_sequences == 2
    single.train_steps(1)
    single_state = torch.load(single.ckpt_path, weights_only=False)

    assert ddp_state["data"]["global_cursor"] == 8
    assert len(ddp_state["rng_by_rank"]) == 4
    assert_models_close(ddp_state["model"], single_state["model"], atol=2e-6)
    single.close()


def test_cpu_gloo_update_matches_single_process_and_resume_is_exact(tmp_path):
    token_path = tmp_path / "tokens.bin"
    (np.arange(80) % 32).astype(np.uint16).tofile(token_path)

    ddp_out = tmp_path / "ddp-resume"
    ddp_cfg = tiny_config(ddp_out, token_path)
    run_ddp(ddp_cfg, tmp_path, steps=1)
    first = torch.load(ddp_out / "ckpt.pt", weights_only=False)
    assert first["step"] == 1
    assert first["data"]["global_cursor"] == 20
    assert len(first["rng_by_rank"]) == 2

    single_cfg = tiny_config(tmp_path / "single", token_path)
    single = Trainer(single_cfg)
    single.train_steps(1)
    single_state = torch.load(single.ckpt_path, weights_only=False)
    assert_models_close(first["model"], single_state["model"], atol=2e-6)

    run_ddp(ddp_cfg, tmp_path, resume=True, steps=1)
    resumed = torch.load(ddp_out / "ckpt.pt", weights_only=False)
    full_out = tmp_path / "ddp-full"
    full_cfg = tiny_config(full_out, token_path)
    run_ddp(full_cfg, tmp_path, steps=2)
    uninterrupted = torch.load(full_out / "ckpt.pt", weights_only=False)

    assert resumed["step"] == uninterrupted["step"] == 2
    assert resumed["data"] == uninterrupted["data"]
    assert resumed["data"]["global_cursor"] == 40
    assert_models_close(resumed["model"], uninterrupted["model"], atol=0)

    log_rows = [
        json.loads(line) for line in (ddp_out / "log.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in log_rows] == [1, 2]
    assert all(row["tokens_per_step"] == 20 for row in log_rows)
    assert set(path.name for path in ddp_out.iterdir()) == {
        "config.yaml",
        "log.jsonl",
        "snapshots",
        "ckpt.pt",
    }
    assert len(list((ddp_out / "snapshots").glob("*.pt"))) == 1
