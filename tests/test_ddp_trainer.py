import json
import os
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

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
        trainer = Trainer(cfg)
        collective_counts = [None] * world_size
        torch.distributed.all_gather_object(collective_counts, trainer.accum)
        if len(set(collective_counts)) != 1:
            raise AssertionError(f"unequal backward schedules: {collective_counts}")
        if resume:
            trainer.load_ckpt()
        trainer.train_steps(steps)
    except BaseException:
        Path(error_dir, f"rank-{rank}.txt").write_text(traceback.format_exc())
        raise
    finally:
        if trainer is not None and hasattr(trainer, "close"):
            trainer.close()
        elif torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def run_ddp(cfg, tmp_path, *, resume=False, steps=None, timeout=45):
    world_size = 2
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
    assert [process.exitcode for process in processes] == [0, 0], errors


def assert_models_close(left, right, *, atol):
    assert set(left) == set(right)
    for name in left:
        assert torch.allclose(left[name], right[name], rtol=0, atol=atol), name


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
