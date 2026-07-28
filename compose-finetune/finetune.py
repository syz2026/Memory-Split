#!/usr/bin/env python
"""Finetune a compose arm from a pretrained seed-0 snapshot (init-from).

Reuses the feat/ood-2-hop `Trainer` / `train_steps` unchanged — this is an
*additive* script (no edits to the branch). Loads MODEL WEIGHTS ONLY from the
init snapshot (fresh optimizer + data cursor + step 0 = finetune), then trains.

Colab-drop safe: if an interrupted finetune's `ckpt.pt` already exists in the
out dir, it RESUMES that (takes precedence over init-from). Write the out dir to
persistent storage (mounted Google Drive) so a dropped session resumes.

Usage (dense, v1):
  python finetune.py --config configs/compose_dense_ft.yaml \
      --init-from /content/drive/MyDrive/ms/snapshots/dense_n800k_step0006100.pt \
      --out-dir  /content/drive/MyDrive/ms/runs/dense_n800k_ft
Usage (split, v2 — token-weighted accumulation):
  python finetune.py --config configs/compose_split_ft.yaml --trainer v2 \
      --init-from /content/drive/MyDrive/ms/snapshots/split_n800k_step0006100.pt \
      --out-dir  /content/drive/MyDrive/ms/runs/split_n800k_ft
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import torch
import yaml


def _start_log_tailer(log_path: Path, stop: threading.Event) -> threading.Thread:
    """Background thread: stream new rows from the trainer's log.jsonl to stdout.

    The trainer writes metrics to a FILE, not stdout, so a Colab cell looks idle
    during training. This tails the file so you see live progress regardless of
    which trainer version is on the (cloned) repo.
    """
    def drain(pos: int) -> int:
        if not log_path.exists():
            return pos
        with open(log_path) as f:
            f.seek(pos)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    mv = f" mv {r['loss_masked_values']}" if "loss_masked_values" in r else ""
                    print(f"[step {r.get('step')}] loss {r.get('loss')} "
                          f"ema {r.get('loss_ema')} lr {r.get('lr')} "
                          f"tok/s {r.get('tok_s')} ep {r.get('epoch')}{mv}", flush=True)
                except json.JSONDecodeError:
                    print(line, flush=True)
            return f.tell()

    def run() -> None:
        pos = log_path.stat().st_size if log_path.exists() else 0  # skip pre-existing rows
        while not stop.is_set():
            try:
                pos = drain(pos)
            except Exception:  # noqa: BLE001 - never let logging kill training
                pass
            stop.wait(5)
        try:
            drain(pos)  # final flush of rows written just before stop
        except Exception:  # noqa: BLE001
            pass

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t

# Repo root on sys.path so `train.*` imports work without PYTHONPATH. Works whether
# this file sits at the repo root (Colab copies it there) or in compose-finetune/.
_here = Path(__file__).resolve().parent
for _cand in (_here, _here.parent, Path.cwd()):
    if (_cand / "train").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))


def _load_trainer_cls(which: str):
    if which == "v2":
        from train.trainer_v2 import Trainer  # token-weighted accumulation
    else:
        from train.trainer import Trainer
    return Trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-from", required=True,
                    help="model-only snapshot (or ckpt.pt) whose weights initialize the model")
    ap.add_argument("--trainer", default="v1", choices=["v1", "v2"])
    ap.add_argument("--resume", default="auto", choices=["auto", "none"])
    ap.add_argument("--out-dir", default=None, help="override cfg out_dir (per-base run dir)")
    ap.add_argument("--run-id", default=None, help="override cfg run_id")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override cfg max_steps (e.g. a short smoke test)")
    ap.add_argument("--micro-batch-size", type=int, default=None,
                    help="override cfg micro_batch_size (LOWER it on smaller GPUs; accum "
                         "auto-adjusts so tokens_per_step is unchanged -> identical training). "
                         "40GB A100: use 8; 24GB: use 4.")
    ap.add_argument("--no-compile", action="store_true",
                    help="disable torch.compile (use if compile OOMs or misbehaves on Colab)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.run_id:
        cfg["run_id"] = args.run_id
    if args.max_steps:
        cfg["max_steps"] = args.max_steps
    if args.micro_batch_size:
        cfg["micro_batch_size"] = args.micro_batch_size
    if args.no_compile:
        cfg["compile"] = False
    print(f"[finetune] micro_batch_size={cfg['micro_batch_size']} "
          f"tokens_per_step={cfg['tokens_per_step']} compile={cfg.get('compile', False)}",
          flush=True)

    Trainer = _load_trainer_cls(args.trainer)
    trainer = Trainer(cfg)  # builds model (random), fresh optimizer/data/step, writes config.yaml

    if args.resume == "auto" and trainer.ckpt_path.exists():
        # An interrupted finetune exists -> continue it (Colab-drop safe).
        # Guard the RNG-restore step: some trainer builds save CUDA RNG state that
        # `set_rng_state_all` rejects after a map_location round-trip. model/opt/
        # data/step load BEFORE the RNG step, so a resume that only fails on RNG is
        # still a valid resume — swallow it. Re-raise if nothing actually loaded.
        try:
            trainer.load_ckpt()
        except (TypeError, RuntimeError) as e:
            if getattr(trainer, "step", 0) > 0:
                print(f"[finetune] RESUMED (RNG restore skipped: {e})", flush=True)
            else:
                raise
        print(f"[finetune] RESUMED interrupted finetune at step {trainer.step}", flush=True)
    else:
        # Fresh finetune: load ONLY model weights from the pretrained snapshot.
        state = torch.load(args.init_from, map_location=trainer.device, weights_only=False)
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        raw = getattr(trainer.model, "_orig_mod", trainer.model)
        raw.load_state_dict(sd)  # strict=True: fail loudly on any architecture mismatch
        n = sum(p.numel() for p in raw.parameters())
        print(f"[finetune] init weights from {args.init_from} ({n/1e6:.1f}M params loaded, strict)",
              flush=True)

    print(f"[finetune] training started; metrics stream below every ~20 steps "
          f"(also in {trainer.log_path}). First row in ~1-2 min.", flush=True)
    stop = threading.Event()
    tailer = _start_log_tailer(Path(trainer.log_path), stop)
    try:
        trainer.train_steps()
    finally:
        stop.set()
        tailer.join(timeout=6)
    print(f"[finetune] DONE step={trainer.step} out={trainer.out_dir}", flush=True)


if __name__ == "__main__":
    main()
