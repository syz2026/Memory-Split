"""Single-process or torchrun/NCCL training with exact global updates.

The loop uses AdamW + cosine, bf16 autocast on CUDA, rank-partitioned gradient
accumulation, globally normalized local loss sums, atomic provenance-guarded
checkpoint/resume, rank-zero snapshots, and optional legacy-mask diagnostics.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from train.data import (
    PackedShards,
    rank_sequence_counts,
    synchronized_rank_batch_plan,
)
from train.model import GPT, GPTConfig, PRESETS


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: str
    process_group_initialized: bool = False
    owns_process_group: bool = False
    backend: str | None = None

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_master(self) -> bool:
        return self.rank == 0


def init_distributed(requested_device: str = "auto") -> DistributedContext:
    env_names = ("WORLD_SIZE", "RANK", "LOCAL_RANK")
    present = tuple(name in os.environ for name in env_names)
    if any(present) and not all(present):
        raise ValueError("WORLD_SIZE, RANK, and LOCAL_RANK must be set together")
    if not any(present):
        if torch.distributed.is_initialized():
            raise ValueError("initialized process group requires torchrun rank environment")
        return DistributedContext(
            rank=0,
            world_size=1,
            local_rank=0,
            device=pick_device(requested_device),
        )

    parsed = {}
    for name in env_names:
        try:
            parsed[name] = int(os.environ[name])
        except ValueError as error:
            raise ValueError(f"{name} must be an integer") from error
    world_size = parsed["WORLD_SIZE"]
    rank = parsed["RANK"]
    local_rank = parsed["LOCAL_RANK"]
    if world_size <= 0:
        raise ValueError("WORLD_SIZE must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("RANK must be in [0, WORLD_SIZE)")
    if not 0 <= local_rank < world_size:
        raise ValueError("LOCAL_RANK must be in [0, WORLD_SIZE)")

    wants_cuda = requested_device == "cuda" or requested_device.startswith("cuda:")
    use_cuda = wants_cuda or (requested_device == "auto" and torch.cuda.is_available())
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested_device.startswith("cuda:"):
            try:
                requested_index = int(requested_device.split(":", 1)[1])
            except ValueError as error:
                raise ValueError("explicit CUDA device index is invalid") from error
            if requested_index != local_rank:
                raise ValueError("explicit CUDA device must match LOCAL_RANK")
        if local_rank >= torch.cuda.device_count():
            raise ValueError("LOCAL_RANK exceeds the visible CUDA device count")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        backend = "nccl"
    else:
        if requested_device not in ("auto", "cpu"):
            raise ValueError("distributed training supports only CPU/gloo or CUDA/NCCL")
        device = "cpu"
        backend = "gloo"

    owns_process_group = False
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend)
        owns_process_group = True
    else:
        if (
            torch.distributed.get_world_size() != world_size
            or torch.distributed.get_rank() != rank
        ):
            raise ValueError("initialized process group disagrees with torchrun environment")
        actual_backend = str(torch.distributed.get_backend()).lower()
        if backend not in actual_backend:
            raise ValueError("initialized process group backend is incompatible with device")
    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        process_group_initialized=True,
        owns_process_group=owns_process_group,
        backend=backend,
    )


def normalize_ddp_loss(
    local_loss_sum: torch.Tensor,
    *,
    world_size: int,
    global_target_count: int | float,
) -> torch.Tensor:
    if not isinstance(local_loss_sum, torch.Tensor) or local_loss_sum.ndim != 0:
        raise ValueError("local_loss_sum must be a scalar tensor")
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if (
        type(global_target_count) not in (int, float)
        or not math.isfinite(global_target_count)
        or global_target_count <= 0
    ):
        raise ValueError("global_target_count must be finite and positive")
    return local_loss_sum * (world_size / global_target_count)


def gradient_sync_context(
    model,
    *,
    distributed: bool,
    final_microstep: bool,
):
    if distributed and not final_microstep:
        return model.no_sync()
    return contextlib.nullcontext()


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


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


_DATA_LOCATION_KEYS = {
    "out_dir",
    "train_bin",
    "train_bins",
    "train_corpus",
    "train_mask",
    "train_masks",
    "train_weight_shards",
    "train_weights",
}


def resume_config_fingerprint(cfg: dict, model_cfg: GPTConfig) -> str:
    contract = {
        "config": {
            key: _jsonable(value)
            for key, value in cfg.items()
            if key not in _DATA_LOCATION_KEYS
        },
        "model_config": asdict(model_cfg),
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
    }
    payload = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class Trainer:
    def __init__(self, cfg: dict):
        if not isinstance(cfg, dict):
            raise ValueError("training config must be a dictionary")
        self.cfg = dict(cfg)
        seed = cfg.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.micro_bs = _positive_int(
            cfg.get("micro_batch_size"),
            "micro_batch_size",
        )
        tokens_per_step = _positive_int(
            cfg.get("tokens_per_step"),
            "tokens_per_step",
        )
        raw_model = cfg.get("model")
        if isinstance(raw_model, str):
            if raw_model not in PRESETS:
                raise ValueError(f"unknown model preset: {raw_model}")
            model_cfg = replace(PRESETS[raw_model])
        elif isinstance(raw_model, dict):
            model_cfg = GPTConfig(**raw_model)
        else:
            raise ValueError("model must be a preset name or config dictionary")
        if "ctx" in cfg:
            model_cfg = replace(
                model_cfg,
                ctx=_positive_int(cfg["ctx"], "ctx"),
            )
        _positive_int(model_cfg.ctx, "model context")
        if tokens_per_step % model_cfg.ctx:
            raise ValueError("tokens_per_step must be divisible by model context")
        self.tokens_per_step = tokens_per_step
        self.sequences_per_step = tokens_per_step // model_cfg.ctx
        if cfg.get("max_steps") is not None:
            self.max_steps = _positive_int(cfg["max_steps"], "max_steps")
        else:
            total_tokens = _positive_int(cfg.get("total_tokens"), "total_tokens")
            if total_tokens % tokens_per_step:
                raise ValueError(
                    "total_tokens must be divisible by tokens_per_step"
                )
            self.max_steps = total_tokens // tokens_per_step
        if cfg.get("total_tokens") is not None:
            total_tokens = _positive_int(cfg["total_tokens"], "total_tokens")
            if total_tokens % tokens_per_step:
                raise ValueError(
                    "total_tokens must be divisible by tokens_per_step"
                )
            if self.max_steps * tokens_per_step != total_tokens:
                raise ValueError(
                    "max_steps * tokens_per_step must equal total_tokens"
                )

        requested_device = cfg.get("device", "auto")
        if not isinstance(requested_device, str):
            raise ValueError("device must be a string")
        self.dist = init_distributed(requested_device)
        self.rank = self.dist.rank
        self.world_size = self.dist.world_size
        self.local_rank = self.dist.local_rank
        self.is_master = self.dist.is_master
        self.device = self.dist.device

        torch.manual_seed(seed)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.model = GPT(model_cfg).to(self.device)
        if cfg.get("compile", False) and self.device.startswith("cuda"):
            self.model = torch.compile(self.model)
        if self.dist.distributed:
            ddp_kwargs = {"broadcast_buffers": False}
            if self.device.startswith("cuda"):
                ddp_kwargs.update(
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                )
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                **ddp_kwargs,
            )
        random.seed(seed + self.rank)
        np.random.seed((seed + self.rank) % (2**32))
        torch.manual_seed(seed + self.rank)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed(seed + self.rank)

        sequence_counts = rank_sequence_counts(
            self.sequences_per_step,
            self.world_size,
        )
        self.local_sequences = sequence_counts[self.rank]
        self.accum = math.ceil(max(sequence_counts) / self.micro_bs)

        unsupported = (
            "train_bins",
            "train_masks",
            "train_weight_shards",
        )
        if any(cfg.get(key) is not None for key in unsupported):
            raise ValueError(
                "unbound shard lists are unsupported; use train_corpus publication"
            )
        has_legacy = cfg.get("train_bin") is not None
        has_parallel = cfg.get("train_corpus") is not None
        if has_legacy == has_parallel:
            raise ValueError(
                "config must select exactly one of train_bin or train_corpus"
            )
        if has_parallel:
            self.data = PackedShards.from_parallel_corpus(
                cfg["train_corpus"],
                ctx=model_cfg.ctx,
                batch_size=self.micro_bs,
                device=self.device,
                seed=seed,
                mask_path=cfg.get("train_mask"),
                weights_path=cfg.get("train_weights"),
            )
        else:
            self.data = PackedShards(
                cfg["train_bin"],
                cfg.get("train_mask"),
                ctx=model_cfg.ctx,
                batch_size=self.micro_bs,
                device=self.device,
                seed=seed,
                weights_path=cfg.get("train_weights"),
            )
        self.data.validate_update_alignment(tokens_per_step)
        self.config_fingerprint = resume_config_fingerprint(cfg, model_cfg)

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
            fused=self.device.startswith("cuda"),
        )

        self.step = 0
        self.out_dir = Path(cfg["out_dir"])

        def initialize_output() -> None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "snapshots").mkdir(exist_ok=True)

        self._rank0_action(initialize_output, "output initialization")
        self.ckpt_path = self.out_dir / "ckpt.pt"
        self.log_path = self.out_dir / "log.jsonl"
        self.snap_every = max(1, int(self.max_steps * cfg.get("snap_frac", 0.10)))
        self.ckpt_seconds = cfg.get("ckpt_minutes", 30) * 60
        self.log_every = cfg.get("log_every", 20)
        self.eval_every = cfg.get("eval_every", 250)
        self._probe = None  # lazy masked-value probe batches

        def write_config() -> None:
            import yaml

            config_path = self.out_dir / "config.yaml"
            temporary = config_path.with_suffix(".yaml.tmp")
            with temporary.open("w") as handle:
                yaml.safe_dump(cfg, handle, sort_keys=False)
            os.replace(temporary, config_path)

        self._rank0_action(write_config, "config write")

    def _barrier(self) -> None:
        if self.dist.process_group_initialized:
            torch.distributed.barrier()

    def _global_sum(self, value: float | int) -> float:
        if not self.dist.process_group_initialized:
            return float(value)
        total = torch.tensor(float(value), dtype=torch.float64, device=self.device)
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        return total.item()

    def _broadcast_master_bool(self, value: bool) -> bool:
        if not self.dist.process_group_initialized:
            return value
        flag = torch.tensor(
            int(value if self.is_master else False),
            dtype=torch.uint8,
            device=self.device,
        )
        torch.distributed.broadcast(flag, src=0)
        return bool(flag.item())

    def _rank0_action(self, action, description: str) -> None:
        error = None
        if self.is_master:
            try:
                action()
            except BaseException as caught:
                error = f"{type(caught).__name__}: {caught}"
        if self.dist.process_group_initialized:
            payload = [error]
            torch.distributed.broadcast_object_list(payload, src=0)
            error = payload[0]
        if error is not None:
            raise RuntimeError(f"rank-0 {description} failed: {error}")
        self._barrier()

    def _raw_model(self):
        model = self.model
        while True:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
            elif hasattr(model, "_orig_mod"):
                model = model._orig_mod
            else:
                return model

    def _rng_states_by_rank(self) -> list[dict]:
        local = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state(self.local_rank)
                if self.device.startswith("cuda")
                else None
            ),
        }
        if not self.dist.process_group_initialized:
            return [local]
        states: list[dict | None] = [None] * self.world_size
        torch.distributed.all_gather_object(states, local)
        return [state for state in states if state is not None]

    def _data_states_by_rank(self) -> list[dict]:
        local = self.data.state_dict()
        if not self.dist.process_group_initialized:
            return [local]
        states: list[dict | None] = [None] * self.world_size
        torch.distributed.all_gather_object(states, local)
        result = [state for state in states if state is not None]
        if len(result) != self.world_size or any(
            state != result[0] for state in result[1:]
        ):
            raise RuntimeError("global loader state diverged across ranks")
        return result

    def close(self) -> None:
        if (
            self.dist.owns_process_group
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()

    # --- checkpointing -----------------------------------------------------

    def save_ckpt(self) -> None:
        rng_by_rank = self._rng_states_by_rank()
        data_states = self._data_states_by_rank()

        def write_checkpoint() -> None:
            raw = self._raw_model()
            state = {
                "checkpoint_version": 2,
                "model": raw.state_dict(),
                "opt": self.opt.state_dict(),
                "data": data_states[0],
                "step": self.step,
                "rng_by_rank": rng_by_rank,
                "cfg": self.cfg,
                "config_fingerprint": self.config_fingerprint,
                "world_size": self.world_size,
                "data_provenance": self.data.provenance,
            }
            tmp = self.ckpt_path.with_suffix(".tmp")
            torch.save(state, tmp)
            os.replace(tmp, self.ckpt_path)

        self._rank0_action(write_checkpoint, "checkpoint write")

    def load_ckpt(self, path: str | Path | None = None) -> None:
        self._barrier()
        checkpoint_path = Path(path) if path is not None else self.ckpt_path
        state = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        if state.get("checkpoint_version") != 2:
            raise ValueError("checkpoint version is incompatible")
        if state.get("world_size") != self.world_size:
            raise ValueError(
                "checkpoint world size does not match current world size"
            )
        if state.get("config_fingerprint") != self.config_fingerprint:
            raise ValueError(
                "checkpoint training config does not match current config"
            )
        if state.get("data_provenance") != self.data.provenance:
            raise ValueError(
                "checkpoint data provenance does not match current dataset"
            )
        rng_by_rank = state.get("rng_by_rank")
        if not isinstance(rng_by_rank, list) or len(rng_by_rank) != self.world_size:
            raise ValueError("checkpoint RNG state does not match current world size")
        saved_step = state.get("step")
        if (
            isinstance(saved_step, bool)
            or not isinstance(saved_step, int)
            or not 0 <= saved_step <= self.max_steps
        ):
            raise ValueError("checkpoint step is invalid for current config")
        raw = self._raw_model()
        raw.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])
        self.data.load_state_dict(state["data"])
        self.step = saved_step
        rank_rng = rng_by_rank[self.rank]
        random.setstate(rank_rng["python"])
        np.random.set_state(rank_rng["numpy"])
        torch.set_rng_state(rank_rng["torch"].cpu())
        if self.device.startswith("cuda") and rank_rng.get("cuda") is not None:
            torch.cuda.set_rng_state(rank_rng["cuda"], device=self.local_rank)
        self._barrier()

    def save_snapshot(self) -> None:
        def write_snapshot() -> None:
            raw = self._raw_model()
            path = self.out_dir / "snapshots" / f"step{self.step:07d}.pt"
            temporary = path.with_suffix(".tmp")
            torch.save(
                {
                    "model": raw.state_dict(),
                    "step": self.step,
                    "model_cfg": raw.cfg.__dict__,
                    "world_size": self.world_size,
                    "data_provenance": self.data.provenance,
                },
                temporary,
            )
            os.replace(temporary, path)

        self._rank0_action(write_snapshot, "snapshot write")

    # --- metrics -----------------------------------------------------------

    @torch.no_grad()
    def loss_masked_values(self) -> float | None:
        """CE at targets excluded by an optional legacy binary loss mask.

        Target-weight-only relational runs have no such mask, so this returns
        None and is not a diagnostic of relational sidecar training.
        """
        if self._probe is None:
            self._probe = self.data.masked_value_batch() or "none"
        if self._probe == "none":
            return None
        x, y = self._probe
        losses = []
        raw = self._raw_model()
        was_training = raw.training
        raw.eval()
        for i in range(0, x.size(0), self.micro_bs):
            xb = x[i : i + self.micro_bs].to(self.device)
            yb = y[i : i + self.micro_bs].to(self.device)
            with self._autocast():
                _, loss = raw(xb, yb)
            if loss is not None and torch.isfinite(loss):
                losses.append(loss.item())
        if was_training:
            raw.train()
        return sum(losses) / len(losses) if losses else None

    def _autocast(self):
        if self.device.startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # --- loop ----------------------------------------------------------------

    def train_steps(self, n_steps: int | None = None) -> float:
        if n_steps is not None and (
            isinstance(n_steps, bool)
            or not isinstance(n_steps, int)
            or n_steps < 0
        ):
            raise ValueError("n_steps must be a non-negative integer")
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
            plan = synchronized_rank_batch_plan(
                global_cursor=self.data.global_cursor,
                total_sequences=self.sequences_per_step,
                ctx=self.data.ctx,
                micro_batch_size=self.micro_bs,
                rank=self.rank,
                world_size=self.world_size,
            )
            prepared = []
            denominator = 0
            weighted = self.data.target_weights is not None
            for batch_slice in plan:
                if batch_slice is None:
                    prepared.append(None)
                    continue
                if weighted:
                    x, y, weights = self.data.weighted_batch_from_slice(batch_slice)
                    denominator += y.numel()
                    prepared.append((x, y, weights))
                else:
                    x, y = self.data.batch_from_slice(batch_slice)
                    denominator += int(y.ne(-100).sum())
                    prepared.append((x, y, None))
            global_denominator = self._global_sum(denominator)
            if global_denominator <= 0:
                raise RuntimeError("optimizer update has no active targets")

            loss_numerator = 0.0
            for microstep, batch in enumerate(prepared):
                with gradient_sync_context(
                    self.model,
                    distributed=self.dist.distributed,
                    final_microstep=microstep == len(prepared) - 1,
                ):
                    if batch is None:
                        dummy = torch.zeros(
                            (1, self.data.ctx),
                            dtype=torch.long,
                            device=self.device,
                        )
                        with self._autocast():
                            logits, _ = self.model(dummy)
                        scaled_loss = logits.sum() * 0.0
                    else:
                        x, y, weights = batch
                        with self._autocast():
                            _, loss_sum = self.model(
                                x,
                                y,
                                target_weights=weights,
                                loss_reduction="sum",
                            )
                        scaled_loss = normalize_ddp_loss(
                            loss_sum,
                            world_size=self.world_size,
                            global_target_count=global_denominator,
                        )
                        loss_numerator += loss_sum.detach().item()
                    scaled_loss.backward()
            self.data.advance(self.tokens_per_step)
            tokens_seen += self.tokens_per_step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.step += 1
            step_loss = self._global_sum(loss_numerator) / global_denominator
            running = step_loss if running is None else 0.95 * running + 0.05 * step_loss

            if self.step % self.log_every == 0 or self.step == target:
                self._barrier()
                elapsed = max(1e-9, time.time() - t0)
                throughput = tokens_seen / elapsed
                row = {
                    "step": self.step,
                    "loss": round(step_loss, 4),
                    "loss_ema": round(running, 4),
                    "lr": lr,
                    "tok_s": round(throughput, 1),
                    "global_tok_s": round(throughput, 1),
                    "tokens_per_step": self.tokens_per_step,
                    "global_tokens": self.step * self.tokens_per_step,
                    "epoch": self.data.epoch,
                }
                if self.step % self.eval_every == 0 or self.step == target:
                    mv = self.loss_masked_values()
                    if mv is not None:
                        row["loss_masked_values"] = round(mv, 4)
                def write_log_row() -> None:
                    with open(self.log_path, "a") as f:
                        f.write(json.dumps(row) + "\n")

                self._rank0_action(write_log_row, "log write")
                t0 = time.time()
                tokens_seen = 0
            if self.step % self.snap_every == 0:
                self.save_snapshot()
            checkpoint_due = self._broadcast_master_bool(
                time.time() - last_ckpt > self.ckpt_seconds
            )
            if checkpoint_due:
                self.save_ckpt()
                last_ckpt = time.time()
        self.save_ckpt()
        return running if running is not None else float("nan")


def train(cfg: dict, resume: str = "none") -> Trainer:
    if resume not in {"auto", "none"}:
        raise ValueError("resume must be 'auto' or 'none'")
    trainer = Trainer(cfg)
    checkpoint_exists = trainer._broadcast_master_bool(
        trainer.ckpt_path.exists() if trainer.is_master else False
    )
    if resume == "none" and checkpoint_exists:
        trainer.close()
        raise FileExistsError(
            "fresh launch refused existing checkpoint; use resume='auto'"
        )
    if resume == "auto" and checkpoint_exists:
        trainer.load_ckpt()
        if trainer.is_master:
            print(f"resumed from step {trainer.step}")
    trainer.train_steps()
    return trainer
