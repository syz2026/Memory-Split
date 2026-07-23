"""Training loop: AdamW + cosine, bf16 autocast (CUDA), grad accumulation,
atomic checkpoint/resume (model+opt+data cursor+RNG), model-only snapshots,
JSONL logging including the split-arm mechanism metric `loss_masked_values`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch

from train.data import PackedShards
from train.model import GPT, GPTConfig, PRESETS


_RESUME_IDENTITY_KEYS = (
    "model",
    "ctx",
    "arm",
    "train_bin",
    "train_mask",
    "train_sha256",
    "train_mask_sha256",
    "data_fingerprint",
    "code_commit",
    "source_checkpoint_sha256",
    "source_checkpoint_step",
    "source_pair_id",
    "continuation_id",
    "total_tokens",
    "tokens_per_step",
    "micro_batch_size",
    "max_steps",
    "seed",
    "precision",
    "resolved_precision",
    "lr",
    "warmup_steps",
    "weight_decay",
    "planned_consumed_tokens",
)


def _load_torch_state(path, map_location):
    try:
        return torch.load(
            path, map_location=map_location, weights_only=False, mmap=True
        )
    except (TypeError, RuntimeError):
        return torch.load(path, map_location=map_location, weights_only=False)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file_sha256(path: str | Path, expected: str | None, label: str) -> None:
    if expected is None:
        return
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )


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
        requested_precision = cfg.get("precision", "bf16" if self.device == "cuda" else "fp32")
        if requested_precision == "auto":
            requested_precision = (
                "bf16"
                if self.device == "cuda" and torch.cuda.is_bf16_supported()
                else "fp16" if self.device == "cuda" else "fp32"
            )
        if requested_precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("precision must be one of: auto, bf16, fp16, fp32")
        if self.device != "cuda" and requested_precision != "fp32":
            requested_precision = "fp32"
        self.precision = requested_precision
        self.cfg["resolved_precision"] = self.precision
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
        _verify_file_sha256(
            cfg["train_bin"], cfg.get("train_sha256"), "training shard"
        )
        _verify_file_sha256(
            cfg.get("train_mask"),
            cfg.get("train_mask_sha256"),
            "training mask",
        )
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
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.device == "cuda" and self.precision == "fp16"
        )

        self.step = 0
        self.out_dir = Path(cfg["out_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "snapshots").mkdir(exist_ok=True)
        self.ckpt_path = self.out_dir / "ckpt.pt"
        self.model_path = self.out_dir / "model.pt"
        self.model_sha_path = self.out_dir / "model.pt.sha256"
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
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "cfg": self.cfg,
        }
        tmp = self.ckpt_path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, self.ckpt_path)

    def load_ckpt(self, path: str | Path | None = None) -> None:
        state = _load_torch_state(path or self.ckpt_path, self.device)
        saved_cfg = state.get("cfg", {})
        mismatches = [
            key
            for key in _RESUME_IDENTITY_KEYS
            if saved_cfg.get(key) != self.cfg.get(key)
        ]
        if mismatches:
            raise ValueError(
                "refusing incompatible checkpoint resume; changed config fields: "
                + ", ".join(mismatches)
            )
        raw = getattr(self.model, "_orig_mod", self.model)
        raw.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])
        if self.scaler.is_enabled() and state.get("scaler") is not None:
            self.scaler.load_state_dict(state["scaler"])
        self.data.load_state_dict(state["data"])
        self.step = state["step"]
        torch.set_rng_state(state["rng_torch"].cpu())
        if self.device == "cuda" and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])

    def load_weights(self, path: str | Path) -> None:
        """Initialize model weights from an existing run without resuming it.

        Continued-training datasets require a fresh optimizer, data cursor,
        schedule, and step counter.  Full ``load_ckpt`` is intentionally not
        used because its optimizer/data state belongs to the old corpus.
        """
        _verify_file_sha256(
            path,
            self.cfg.get("source_checkpoint_sha256"),
            "source checkpoint",
        )
        state = _load_torch_state(path, "cpu")
        model_state = state.get("model", state)
        raw = getattr(self.model, "_orig_mod", self.model)
        raw.load_state_dict(model_state)
        del state

    def save_model(self) -> None:
        """Atomically save a model-only artifact for low-RAM eval/continuation."""
        raw = getattr(self.model, "_orig_mod", self.model)
        state = {
            "model": raw.state_dict(),
            "step": self.step,
            "model_cfg": raw.cfg.__dict__,
            "cfg": self.cfg,
        }
        tmp = self.model_path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, self.model_path)
        digest = _sha256_file(self.model_path)
        sha_tmp = self.model_sha_path.with_suffix(".tmp")
        sha_tmp.write_text(digest + "\n")
        os.replace(sha_tmp, self.model_sha_path)

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
            dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
            if self.precision != "fp32":
                return torch.autocast(device_type="cuda", dtype=dtype)
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
            micro_losses = []
            for _ in range(self.accum):
                x, y = self.data.next_batch()
                with self._autocast():
                    _, loss = self.model(x, y)
                scaled_loss = loss / self.accum
                if self.scaler.is_enabled():
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                micro_losses.append(loss.item())
                tokens_seen += x.numel()
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if self.scaler.is_enabled():
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()
            self.step += 1
            step_loss = sum(micro_losses) / len(micro_losses)
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
        self.save_model()
        self.save_ckpt()
        return running if running is not None else float("nan")


def train(
    cfg: dict,
    resume: str = "auto",
    init_from: str | Path | None = None,
) -> Trainer:
    trainer = Trainer(cfg)
    if resume == "auto" and trainer.ckpt_path.exists():
        trainer.load_ckpt()
        print(f"resumed from step {trainer.step}")
    elif init_from or cfg.get("init_from"):
        source = init_from or cfg["init_from"]
        trainer.load_weights(source)
        print(f"initialized model weights from {source}")
    trainer.train_steps()
    return trainer
