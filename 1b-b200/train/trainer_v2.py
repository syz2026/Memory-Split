"""PROPOSED v2 trainer — token-weighted gradient accumulation. NOT USED TO TRAIN ANYTHING HERE.

⚠️  STATUS — READ THIS FIRST:
This file is a **NEW, PROPOSED** alternative to `train/trainer.py`. **Every checkpoint and every
result in this repo was produced by the ORIGINAL `train/trainer.py` (v1)**, which uses equal-
weighted gradient accumulation. This v2 has trained **nothing** here — it is committed *alongside*
v1 only so the proposed fix is visible and reviewable. It is **not** "tested" or "validated"
code; adopting it would require **retraining the split runs**.

WHAT DIFFERS FROM v1 (see `docs/deferred/ESCALATIONS-frozen-code.md` §1):
Each micro-batch's mean loss is weighted by its count of non-`-100` (loss-ON) targets, so the
accumulated gradient equals the TRUE full-batch mean-loss gradient:
    v1:  (loss / accum).backward();          step_loss = mean(micro_losses)
    v2:  (loss * c/total_valid).backward();   step_loss = sum(l*c)/total_valid
- **Dense arm:** every micro-batch has the same valid-token count ⇒ weights are 1/accum ⇒
  **mathematically identical to v1 (a no-op).**
- **Split arm:** masked-value counts vary per micro-batch ⇒ v1 is an arm-asymmetric *biased*
  estimator; v2 is unbiased. **This CHANGES split-arm training**, so v2-trained models are NOT
  comparable to the v1 checkpoints in this repo without a full retrain. Deferred pending owner
  sign-off (frozen preregistered code).

--- original trainer docstring below ---
Training loop: AdamW + cosine, bf16 autocast (CUDA), grad accumulation,
atomic checkpoint/resume (model+opt+data cursor+RNG), model-only snapshots,
JSONL logging including the split-arm mechanism metric `loss_masked_values`.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch

from train.data import PackedShards
from train.model import GPT, GPTConfig, PRESETS


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cosine_lr(step: int, peak: float, warmup: int, total: int, min_frac: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    if step >= total:
        return peak * min_frac
    ratio = (step - warmup) / max(1, total - warmup)
    return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * ratio)))


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = pick_device(cfg.get("device", "auto"))
        torch.manual_seed(cfg["seed"])
        if self.device == "cuda":
            torch.cuda.manual_seed_all(cfg["seed"])
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        model_cfg = (
            PRESETS[cfg["model"]]
            if isinstance(cfg["model"], str)
            else GPTConfig(**cfg["model"])
        )
        if "ctx" in cfg:
            model_cfg.ctx = cfg["ctx"]
        self.model = GPT(model_cfg).to(self.device)
        if cfg.get("compile", False) and self.device == "cuda":
            self.model = torch.compile(self.model)

        self.micro_bs = cfg["micro_batch_size"]
        self.accum = max(1, cfg["tokens_per_step"] // (self.micro_bs * model_cfg.ctx))
        self.data = PackedShards(
            cfg["train_bin"],
            cfg.get("train_mask"),
            ctx=model_cfg.ctx,
            batch_size=self.micro_bs,
            device=self.device,
            seed=cfg["seed"],
        )
        self.max_steps = cfg.get("max_steps") or int(
            cfg["total_tokens"] // cfg["tokens_per_step"]
        )

        decay, no_decay = [], []
        for _, p in self.model.named_parameters():
            (decay if p.dim() >= 2 else no_decay).append(p)
        self.opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": cfg.get("weight_decay", 0.1)},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg["lr"],
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=self.device == "cuda",
        )

        self.step = 0
        self.out_dir = Path(cfg["out_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "snapshots").mkdir(exist_ok=True)
        self.ckpt_path = self.out_dir / "ckpt.pt"
        self.log_path = self.out_dir / "log.jsonl"
        self.snap_every = max(1, int(self.max_steps * cfg.get("snap_frac", 0.10)))
        self.ckpt_seconds = cfg.get("ckpt_minutes", 30) * 60
        self.log_every = cfg.get("log_every", 20)
        self.eval_every = cfg.get("eval_every", 250)
        self._probe = None  # lazy masked-value probe batches

        with open(self.out_dir / "config.yaml", "w") as f:
            import yaml

            yaml.safe_dump(cfg, f, sort_keys=False)

    # --- checkpointing -----------------------------------------------------

    def save_ckpt(self) -> None:
        raw = getattr(self.model, "_orig_mod", self.model)
        state = {
            "model": raw.state_dict(),
            "opt": self.opt.state_dict(),
            "data": self.data.state_dict(),
            "step": self.step,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if self.device == "cuda" else None,
            "cfg": self.cfg,
        }
        tmp = self.ckpt_path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, self.ckpt_path)

    def load_ckpt(self, path: str | Path | None = None) -> None:
        state = torch.load(path or self.ckpt_path, map_location=self.device, weights_only=False)
        raw = getattr(self.model, "_orig_mod", self.model)
        raw.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])
        self.data.load_state_dict(state["data"])
        self.step = state["step"]
        torch.set_rng_state(state["rng_torch"].cpu())
        if self.device == "cuda" and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])

    def save_snapshot(self) -> None:
        raw = getattr(self.model, "_orig_mod", self.model)
        torch.save(
            {"model": raw.state_dict(), "step": self.step, "model_cfg": raw.cfg.__dict__},
            self.out_dir / "snapshots" / f"step{self.step:07d}.pt",
        )

    # --- metrics -----------------------------------------------------------

    @torch.no_grad()
    def loss_masked_values(self) -> float | None:
        """CE at loss-masked positions (fact values). The gate-0 mechanism
        metric: stays high in the split arm, falls in the dense arm's bio text."""
        if self._probe is None:
            self._probe = self.data.masked_value_batch() or "none"
        if self._probe == "none":
            return None
        x, y = self._probe
        losses = []
        was_training = self.model.training
        self.model.eval()
        for i in range(0, x.size(0), self.micro_bs):
            xb = x[i : i + self.micro_bs].to(self.device)
            yb = y[i : i + self.micro_bs].to(self.device)
            with self._autocast():
                _, loss = self.model(xb, yb)
            if loss is not None and torch.isfinite(loss):
                losses.append(loss.item())
        if was_training:
            self.model.train()
        return sum(losses) / len(losses) if losses else None

    def _autocast(self):
        if self.device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        import contextlib

        return contextlib.nullcontext()

    # --- loop ----------------------------------------------------------------

    def train_steps(self, n_steps: int | None = None) -> float:
        target = self.step + n_steps if n_steps is not None else self.max_steps
        target = min(target, self.max_steps)
        self.model.train()
        last_ckpt = time.time()
        t0 = time.time()
        tokens_seen = 0
        running = None
        while self.step < target:
            lr = cosine_lr(
                self.step, self.cfg["lr"], self.cfg.get("warmup_steps", 300), self.max_steps
            )
            for group in self.opt.param_groups:
                group["lr"] = lr
            self.opt.zero_grad(set_to_none=True)
            # Token-weighted gradient accumulation: each micro-batch's mean loss
            # is weighted by its number of non-ignored (loss-ON) targets, so the
            # accumulated gradient equals the TRUE full-batch mean-loss gradient.
            # For the dense arm (no masking) every micro-batch has the same valid
            # count ⇒ weights are 1/accum, identical to the old behavior; the fix
            # only affects the split arm, whose masked-value counts vary per
            # micro-batch — removing an arm-asymmetric weighting bias (L31).
            batches = [self.data.next_batch() for _ in range(self.accum)]
            counts = [int((y != -100).sum().item()) for _, y in batches]
            total_valid = sum(counts) or 1
            micro_losses: list[tuple[float, int]] = []
            for (x, y), c in zip(batches, counts):
                with self._autocast():
                    _, loss = self.model(x, y)
                (loss * (c / total_valid)).backward()
                micro_losses.append((loss.item(), c))
                tokens_seen += x.numel()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.step += 1
            step_loss = sum(l * c for l, c in micro_losses) / total_valid
            running = step_loss if running is None else 0.95 * running + 0.05 * step_loss

            if self.step % self.log_every == 0 or self.step == target:
                row = {
                    "step": self.step,
                    "loss": round(step_loss, 4),
                    "loss_ema": round(running, 4),
                    "lr": lr,
                    "tok_s": round(tokens_seen / max(1e-9, time.time() - t0), 1),
                    "epoch": self.data.epoch,
                }
                if self.step % self.eval_every == 0 or self.step == target:
                    mv = self.loss_masked_values()
                    if mv is not None:
                        row["loss_masked_values"] = round(mv, 4)
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                t0 = time.time()
                tokens_seen = 0
            if self.step % self.snap_every == 0:
                self.save_snapshot()
            if time.time() - last_ckpt > self.ckpt_seconds:
                self.save_ckpt()
                last_ckpt = time.time()
        self.save_ckpt()
        return running if running is not None else float("nan")


def train(cfg: dict, resume: str = "auto") -> Trainer:
    trainer = Trainer(cfg)
    if resume == "auto" and trainer.ckpt_path.exists():
        trainer.load_ckpt()
        print(f"resumed from step {trainer.step}")
    trainer.train_steps()
    return trainer
