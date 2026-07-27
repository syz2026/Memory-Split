#!/usr/bin/env python
"""Bare-metal multi-GPU launcher for non-Slurm machines (e.g. an AWS
8xH100 node). Pins one training run per GPU, restarts crashed runs from
their checkpoints, and exits when every run is complete.

Usage:
  PYTHONPATH=. python scripts/run_local_gpus.py --manifest outputs/manifests/confirm.tsv \
      [--gpus 0,1,2,3] [--max-restarts 3]

Each manifest line is a YAML config path (as emitted by make_manifest).
Runs already at max_steps are skipped, so the launcher is idempotent and
safe to rerun after interruptions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def run_finished(cfg: dict) -> bool:
    log = Path(cfg["out_dir"]) / "log.jsonl"
    if not log.exists():
        return False
    try:
        last = json.loads(log.read_text().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return False
    max_steps = cfg.get("max_steps") or int(cfg["total_tokens"] // cfg["tokens_per_step"])
    return last.get("step", 0) >= max_steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gpus", default=None, help="comma list; default: all visible")
    ap.add_argument("--max-restarts", type=int, default=3)
    ap.add_argument("--poll-seconds", type=int, default=60)
    args = ap.parse_args()

    if args.gpus:
        gpus = [g.strip() for g in args.gpus.split(",")]
    else:
        import torch

        gpus = [str(i) for i in range(torch.cuda.device_count())]
    if not gpus:
        sys.exit("no GPUs visible")

    configs = [l.strip() for l in open(args.manifest) if l.strip() and not l.startswith("#")]
    queue: list[str] = []
    for cfg_path in configs:
        cfg = yaml.safe_load(open(cfg_path))
        if run_finished(cfg):
            print(f"[launcher] already complete: {cfg_path}")
        else:
            queue.append(cfg_path)
    print(f"[launcher] {len(queue)} runs across {len(gpus)} GPUs")

    active: dict[str, tuple[subprocess.Popen, str]] = {}  # gpu -> (proc, cfg)
    restarts: dict[str, int] = {}

    def launch(gpu: str, cfg_path: str) -> None:
        out_dir = Path(yaml.safe_load(open(cfg_path))["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        logf = open(out_dir / "launcher.log", "a")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=os.getcwd())
        proc = subprocess.Popen(
            [sys.executable, "-u", "scripts/run_train.py", "--config", cfg_path,
             "--resume", "auto"],
            env=env, stdout=logf, stderr=subprocess.STDOUT,
        )
        active[gpu] = (proc, cfg_path)
        print(f"[launcher] gpu {gpu} <- {cfg_path} (pid {proc.pid})")

    while queue or active:
        for gpu in list(gpus):
            if gpu not in active and queue:
                launch(gpu, queue.pop(0))
        time.sleep(args.poll_seconds)
        for gpu, (proc, cfg_path) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            del active[gpu]
            cfg = yaml.safe_load(open(cfg_path))
            if code == 0 and run_finished(cfg):
                print(f"[launcher] DONE {cfg_path}")
            elif code == 0:
                print(f"[launcher] exited 0 but incomplete, requeueing: {cfg_path}")
                queue.append(cfg_path)
            else:
                n = restarts.get(cfg_path, 0) + 1
                restarts[cfg_path] = n
                if n <= args.max_restarts:
                    print(f"[launcher] crash (exit {code}), restart {n}: {cfg_path}")
                    queue.append(cfg_path)
                else:
                    print(f"[launcher] GAVE UP after {n - 1} restarts: {cfg_path}")
    print("[launcher] all runs complete")


if __name__ == "__main__":
    main()
