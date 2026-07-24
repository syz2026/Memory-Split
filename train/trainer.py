"""Single-process or torchrun/NCCL training with exact global updates.

The loop uses AdamW + cosine, bf16 autocast on CUDA, rank-partitioned gradient
accumulation, globally normalized local loss sums, atomic provenance-guarded
checkpoint/resume, rank-zero snapshots, and optional legacy-mask diagnostics.
"""

from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
import math
import os
import random
import re
import secrets
import signal
import stat
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from train.data import (
    PackedShards,
    rank_sequence_counts,
    strict_json_identity,
    synchronized_rank_batch_plan,
)
from train.model import GPT, GPTConfig, PRESETS
from train.safeio import (
    DurableOutput,
    read_regular_path,
    require_absent_path,
    write_atomic_path,
)


_CHECKPOINT_FIELDS = {
    "checkpoint_version",
    "cfg",
    "config_fingerprint",
    "data",
    "data_provenance",
    "model",
    "opt",
    "rng_by_rank",
    "step",
    "world_size",
}
_RNG_FIELDS = {"cuda", "numpy", "python", "torch"}
_DATA_STATE_FIELDS = {
    "cursor",
    "epoch",
    "format_version",
    "global_cursor",
    "provenance",
}
_LOG_REQUIRED_FIELDS = {
    "epoch",
    "global_tok_s",
    "global_tokens",
    "loss",
    "loss_ema",
    "lr",
    "step",
    "tok_s",
    "tokens_per_step",
}
_LOG_OPTIONAL_FIELDS = {"loss_masked_values"}
_SNAPSHOT_FIELDS = {
    "data_provenance",
    "model",
    "model_cfg",
    "step",
    "world_size",
}
_SNAPSHOT_NAME = re.compile(r"^step([0-9]{7})\.pt$")
_QUARANTINED_SNAPSHOT_NAME = re.compile(
    r"^\.quarantine-step([0-9]{7})\.pt-after-step"
    r"([0-9]{7})-([0-9a-f]{32})$"
)
_ATOMIC_TEMPORARY_NAME = re.compile(
    r"^\.(?P<target>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"\.tmp-(?P<pid>[1-9][0-9]*)-(?P<nonce>[0-9a-f]{16})$"
)
TRAINER_CAPABILITIES = {
    "rank_zero_pid_file": True,
    "receipt_v2": True,
    "resume_sha256": True,
    "sidecar_name": True,
    "sigusr1_checkpoint": True,
}


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


def trainer_capabilities() -> dict[str, bool]:
    return dict(TRAINER_CAPABILITIES)


def resolve_snapshot_steps(
    cfg: dict,
    *,
    max_steps: int,
) -> tuple[int, ...]:
    if type(cfg) is not dict:
        raise ValueError("training config must be a dictionary")
    _positive_int(max_steps, "max_steps")
    schema_version = cfg.get("schema_version", 1)
    if type(schema_version) is not int or schema_version <= 0:
        raise ValueError("schema_version must be a positive integer")
    has_steps = "snapshot_steps" in cfg
    has_fraction = "snap_frac" in cfg
    if has_steps and has_fraction:
        raise ValueError("snapshot_steps and snap_frac cannot both be specified")
    if schema_version == 2 and not has_steps:
        raise ValueError("v2 configs require exact snapshot_steps, not snap_frac")
    if has_steps:
        values = cfg["snapshot_steps"]
        if type(values) is not list or not values:
            raise ValueError("snapshot_steps must be a non-empty list")
        if any(
            type(step) is not int or not 1 <= step <= max_steps
            for step in values
        ):
            raise ValueError(
                "snapshot_steps must contain positive integers within max_steps"
            )
        if any(left >= right for left, right in zip(values, values[1:])):
            raise ValueError("snapshot_steps must be increasing and unique")
        if values[-1] != max_steps:
            raise ValueError("final snapshot_steps entry must equal max_steps")
        return tuple(values)

    snap_frac = cfg.get("snap_frac", 0.10)
    if (
        type(snap_frac) not in (int, float)
        or not math.isfinite(snap_frac)
        or not 0 < snap_frac <= 1
    ):
        raise ValueError("snap_frac must be finite and in (0, 1]")
    snap_every = max(1, int(max_steps * snap_frac))
    return tuple(range(snap_every, max_steps + 1, snap_every))


def _unique_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        value[key] = item
    return value


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json_hash(value) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _config_selector_flags(value: object) -> tuple[bool, bool]:
    if isinstance(value, str):
        normalized = "".join(
            character for character in value.lower() if character.isalnum()
        )
        return "dense" in normalized, "split90" in normalized
    if isinstance(value, dict):
        flags = tuple(_config_selector_flags(item) for item in value.values())
        return (
            any(dense for dense, _ in flags),
            any(split90 for _, split90 in flags),
        )
    if isinstance(value, (list, tuple)):
        flags = tuple(_config_selector_flags(item) for item in value)
        return (
            any(dense for dense, _ in flags),
            any(split90 for _, split90 in flags),
        )
    return False, False


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
    def __init__(
        self,
        cfg: dict,
        *,
        resume: str = "none",
        resume_path: str | Path | None = None,
        resume_sha256: str | None = None,
    ):
        if not isinstance(cfg, dict):
            raise ValueError("training config must be a dictionary")
        if resume not in {"auto", "none"}:
            raise ValueError("resume must be 'auto' or 'none'")
        if resume_path is not None and resume != "auto":
            raise ValueError("resume_path requires resume='auto'")
        if resume_path is not None and resume_sha256 is None:
            raise ValueError("resume_path requires resume_sha256")
        if resume_path is None and resume_sha256 is not None:
            raise ValueError("resume_sha256 requires resume_path")
        self.cfg = copy.deepcopy(cfg)
        cfg = self.cfg
        self._checkpoint_request_generation = 0
        self._checkpoint_request_consumed = 0
        self._previous_sigusr1_handler = None
        self._signal_handler_installed = False
        self.rank_zero_pid_path: Path | None = None
        raw_out_dir = cfg.get("out_dir")
        if not isinstance(raw_out_dir, (str, Path)):
            raise ValueError("out_dir must be a path string")
        self.out_dir = Path(raw_out_dir)
        self.ckpt_path = self.out_dir / "ckpt.pt"
        self.log_path = self.out_dir / "log.jsonl"
        self._output: DurableOutput | None = None
        self._resume_mode = resume
        self._external_resume = resume_path is not None
        self._resume_path = (
            Path(resume_path)
            if resume_path is not None
            else (self.ckpt_path if resume == "auto" else None)
        )
        self._resume_sha256 = resume_sha256
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
        if resume == "none" or self._external_resume:
            self._rank0_action(
                lambda: require_absent_path(self.out_dir, label="output"),
                "fresh output admission",
            )
        self.config_fingerprint = resume_config_fingerprint(cfg, model_cfg)
        self._agree_config_fingerprint()

        torch.manual_seed(seed)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.model_cfg = model_cfg
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
        sidecar_name = cfg.get("sidecar_name")
        if has_legacy and sidecar_name is not None:
            raise ValueError("sidecar_name is unsupported with legacy train_bin")
        selector_flags = tuple(
            _config_selector_flags(cfg.get(field_name))
            for field_name in (
                "arm",
                "condition",
                "intervention",
                "run_id",
            )
        )
        dense_selected = any(dense for dense, _ in selector_flags)
        split90_selected = any(split90 for _, split90 in selector_flags)
        if dense_selected and split90_selected:
            raise ValueError(
                "config has contradictory Dense and Split90 selectors"
            )
        if dense_selected and sidecar_name == "split90_target_weights":
            raise ValueError(
                "Dense requires receipt-v2 sidecar_name dense_target_weights, "
                "not split90_target_weights"
            )
        split90_selected = (
            split90_selected
            or sidecar_name == "split90_target_weights"
        )
        if split90_selected and not has_parallel:
            raise ValueError(
                "Split90 requires a bound receipt-v2 sidecar publication"
            )
        if (
            split90_selected
            and sidecar_name != "split90_target_weights"
        ):
            raise ValueError(
                "Split90 requires receipt-v2 sidecar_name "
                "split90_target_weights"
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
                sidecar_name=sidecar_name,
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
        if split90_selected and (
            self.data.provenance.get("sidecar_name")
            != "split90_target_weights"
            or self.data.target_weights is None
        ):
            self.data.close()
            raise ValueError(
                "Split90 requires a bound receipt-v2 sidecar; "
                "unweighted training is forbidden"
            )
        self.data.validate_update_alignment(tokens_per_step)

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
        self._capture_operational_metrics = False
        self.operational_start_step: int | None = None
        self.operational_step_tok_s: list[float] = []
        self.snapshot_steps = resolve_snapshot_steps(
            cfg,
            max_steps=self.max_steps,
        )
        self._snapshot_step_set = frozenset(self.snapshot_steps)
        self.ckpt_seconds = cfg.get("ckpt_minutes", 30) * 60
        self.log_every = cfg.get("log_every", 20)
        self.eval_every = cfg.get("eval_every", 250)
        self._probe = None  # lazy masked-value probe batches

        self._agree_startup_contract()
        if self._resume_path is not None:
            self.load_ckpt(
                self._resume_path,
                sha256=self._resume_sha256,
                _default_path=not self._external_resume,
            )

        def initialize_output() -> None:
            import yaml

            output = None
            try:
                if resume == "auto" and not self._external_resume:
                    output = DurableOutput.open_existing(self.out_dir)
                    saved_config = yaml.safe_load(
                        output.root.read_regular(
                            "config.yaml",
                            label="resume config",
                        ).payload
                    )
                    if saved_config != _jsonable(cfg):
                        raise ValueError(
                            "resume config.yaml does not match current config"
                        )
                    self._reconcile_resume_artifacts(output)
                else:
                    output = DurableOutput.create(self.out_dir)
                    payload = yaml.safe_dump(
                        _jsonable(cfg),
                        sort_keys=False,
                    ).encode("utf-8")
                    output.root.write_bytes(
                        "config.yaml",
                        payload,
                        replace=False,
                    )
                self._output = output
            except BaseException:
                if output is not None:
                    output.close()
                raise

        self._rank0_action(initialize_output, "output initialization")
        self._rank0_action(
            self._initialize_runtime_controls,
            "runtime control initialization",
        )

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
                error = {
                    "type": type(caught).__name__,
                    "message": str(caught),
                }
        if self.dist.process_group_initialized:
            payload = [error]
            torch.distributed.broadcast_object_list(payload, src=0)
            error = payload[0]
        if error is not None:
            exception_type = {
                "FileExistsError": FileExistsError,
                "FileNotFoundError": FileNotFoundError,
                "ValueError": ValueError,
            }.get(error["type"], RuntimeError)
            raise exception_type(
                f"rank-0 {description} failed: "
                f"{error['type']}: {error['message']}"
            )
        self._barrier()

    def _agree_config_fingerprint(self) -> None:
        if not self.dist.process_group_initialized:
            return
        gathered: list[str | None] = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, self.config_fingerprint)
        if any(item != gathered[0] for item in gathered[1:]):
            raise RuntimeError("ranks disagree on training config")

    def _agree_startup_contract(self) -> None:
        local = {
            "config_fingerprint": self.config_fingerprint,
            "data_provenance_sha256": _canonical_json_hash(self.data.provenance),
        }
        if not self.dist.process_group_initialized:
            return
        gathered: list[dict | None] = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, local)
        if any(item != gathered[0] for item in gathered[1:]):
            raise RuntimeError(
                "ranks disagree on training config or data provenance"
            )

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

    def _initialize_runtime_controls(self) -> None:
        if not hasattr(signal, "SIGUSR1"):
            raise RuntimeError("safe checkpoint requests require SIGUSR1")
        previous = signal.getsignal(signal.SIGUSR1)

        def request_checkpoint(_signum, _frame) -> None:
            self._checkpoint_request_generation += 1

        signal.signal(signal.SIGUSR1, request_checkpoint)
        self._previous_sigusr1_handler = previous
        self._signal_handler_installed = True
        try:
            raw_pid_path = os.environ.get("MS_RANK_ZERO_PID_FILE")
            if raw_pid_path is None:
                return
            if not raw_pid_path or "\x00" in raw_pid_path:
                raise ValueError("MS_RANK_ZERO_PID_FILE must be a safe path")
            pid_path = Path(raw_pid_path)
            write_atomic_path(
                pid_path,
                f"{os.getpid()}\n".encode("ascii"),
                label="rank-zero PID file",
            )
            self.rank_zero_pid_path = pid_path
        except BaseException:
            self._restore_checkpoint_signal_handler()
            raise

    def _restore_checkpoint_signal_handler(self) -> None:
        if not self._signal_handler_installed:
            return
        signal.signal(
            signal.SIGUSR1,
            self._previous_sigusr1_handler,
        )
        self._signal_handler_installed = False
        self._previous_sigusr1_handler = None

    def _service_checkpoint_request(
        self,
        *,
        checkpoint_due: bool = False,
    ) -> bool:
        if type(checkpoint_due) is not bool:
            raise ValueError("checkpoint_due must be boolean")
        master_requested = False
        if self.is_master:
            generation = self._checkpoint_request_generation
            master_requested = generation != self._checkpoint_request_consumed
            self._checkpoint_request_consumed = generation
        requested = self._broadcast_master_bool(master_requested)
        if requested or checkpoint_due:
            self.save_ckpt()
        return requested

    def close(self) -> None:
        if getattr(self, "is_master", False):
            self._restore_checkpoint_signal_handler()
        output = getattr(self, "_output", None)
        if output is not None:
            output.close()
            self._output = None
        if hasattr(self, "data"):
            self.data.close()
        dist = getattr(self, "dist", None)
        if (
            dist is not None
            and dist.owns_process_group
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.destroy_process_group()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass

    # --- checkpointing -----------------------------------------------------

    @staticmethod
    def _validate_atomic_temporary(
        metadata: os.stat_result,
        *,
        name: str,
    ) -> None:
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
            or metadata.st_nlink not in (1, 2)
        ):
            raise ValueError(
                f"resume atomic temporary is foreign or unsafe: {name}"
            )

    def _inspect_atomic_writer_entries(
        self,
        output: DurableOutput,
    ):
        cleanups = []
        snapshot_temporaries = set()
        linked_snapshot_targets = set()
        allowed_root = {
            "ckpt.pt",
            "config.yaml",
            "log.jsonl",
            "snapshots",
        }
        raw_pid_path = os.environ.get("MS_RANK_ZERO_PID_FILE")
        if raw_pid_path:
            pid_path = Path(raw_pid_path).absolute()
            if pid_path.parent == self.out_dir.absolute():
                allowed_root.add(pid_path.name)
        for name in output.root.entries():
            if name in allowed_root:
                continue
            match = _ATOMIC_TEMPORARY_NAME.fullmatch(name)
            if match is None or match.group("target") not in {
                "ckpt.pt",
                "log.jsonl",
            }:
                raise ValueError(
                    f"resume output contains foreign entry: {name}"
                )
            metadata = output.root.entry_metadata(
                name,
                label="resume atomic temporary",
            )
            self._validate_atomic_temporary(metadata, name=name)
            if metadata.st_nlink != 1:
                raise ValueError(
                    f"resume atomic temporary has foreign hard links: {name}"
                )
            cleanups.append((output.root, name, metadata))

        for name in output.snapshots.entries():
            if (
                _SNAPSHOT_NAME.fullmatch(name) is not None
                or _QUARANTINED_SNAPSHOT_NAME.fullmatch(name) is not None
            ):
                continue
            match = _ATOMIC_TEMPORARY_NAME.fullmatch(name)
            target_match = (
                _SNAPSHOT_NAME.fullmatch(match.group("target"))
                if match is not None
                else None
            )
            if match is None or target_match is None:
                raise ValueError(
                    f"resume snapshots contain foreign entry: {name}"
                )
            target_name = match.group("target")
            target_step = int(target_match.group(1))
            if target_step not in self._snapshot_step_set:
                raise ValueError(
                    "resume atomic temporary targets a snapshot step "
                    f"not configured by snapshot_steps: {target_step}"
                )
            metadata = output.snapshots.entry_metadata(
                name,
                label="resume snapshot atomic temporary",
            )
            self._validate_atomic_temporary(metadata, name=name)
            if metadata.st_nlink == 2:
                try:
                    target_metadata = output.snapshots.entry_metadata(
                        target_name,
                        label="resume snapshot atomic target",
                    )
                except FileNotFoundError as error:
                    raise ValueError(
                        "resume snapshot atomic temporary has a foreign "
                        f"hard link: {name}"
                    ) from error
                if (
                    target_metadata.st_nlink != 2
                    or (target_metadata.st_dev, target_metadata.st_ino)
                    != (metadata.st_dev, metadata.st_ino)
                ):
                    raise ValueError(
                        "resume snapshot atomic temporary has a foreign "
                        f"hard link: {name}"
                    )
                linked_snapshot_targets.add(target_name)
            cleanups.append((output.snapshots, name, metadata))
            snapshot_temporaries.add(name)
        return (
            tuple(cleanups),
            frozenset(snapshot_temporaries),
            frozenset(linked_snapshot_targets),
        )

    def _inspect_resume_log(
        self,
        output: DurableOutput,
    ) -> tuple[bytes | None, bool]:
        if "log.jsonl" not in output.root.entries():
            return None, False
        metadata = output.root.entry_metadata(
            "log.jsonl",
            label="resume training log",
        )
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(
                "resume log.jsonl is symlinked, hard-linked, or unsafe"
            )
        payload = output.root.read_regular(
            "log.jsonl",
            label="resume training log",
        ).payload
        durable_end = payload.rfind(b"\n") + 1
        durable_payload = payload[:durable_end]
        partial_tail = durable_end != len(payload)
        lines = (
            durable_payload[:-1].split(b"\n")
            if durable_payload
            else []
        )
        retained = []
        previous_step = 0
        stale_seen = False
        for index, raw_line in enumerate(lines, start=1):
            try:
                row = json.loads(
                    raw_line,
                    object_pairs_hook=_unique_json_object,
                )
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValueError,
            ) as error:
                raise ValueError(
                    f"resume log.jsonl row {index} is malformed"
                ) from error
            if (
                type(row) is not dict
                or set(row) - _LOG_OPTIONAL_FIELDS != _LOG_REQUIRED_FIELDS
                or not set(row) <= _LOG_REQUIRED_FIELDS | _LOG_OPTIONAL_FIELDS
            ):
                raise ValueError(
                    f"resume log.jsonl row {index} fields are foreign"
                )
            step = row["step"]
            if (
                type(step) is not int
                or not 1 <= step <= self.max_steps
                or step <= previous_step
            ):
                raise ValueError(
                    "resume log.jsonl steps must be unique and increasing"
                )
            previous_step = step
            for field_name in ("epoch", "global_tokens", "tokens_per_step"):
                if type(row[field_name]) is not int or row[field_name] < 0:
                    raise ValueError(
                        f"resume log.jsonl {field_name} is invalid"
                    )
            if (
                row["tokens_per_step"] != self.tokens_per_step
                or row["global_tokens"] != step * self.tokens_per_step
            ):
                raise ValueError(
                    "resume log.jsonl token counts do not match training"
                )
            for field_name in (
                "global_tok_s",
                "loss",
                "loss_ema",
                "lr",
                "tok_s",
                *(
                    ("loss_masked_values",)
                    if "loss_masked_values" in row
                    else ()
                ),
            ):
                value = row[field_name]
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        f"resume log.jsonl {field_name} is invalid"
                    )
            if step <= self.step:
                if stale_seen:
                    raise ValueError(
                        "resume log.jsonl stale rows are not a suffix"
                    )
                retained.append(raw_line + b"\n")
            else:
                stale_seen = True
        return b"".join(retained), stale_seen or partial_tail

    def _validate_snapshot_bytes(
        self,
        payload: bytes,
        *,
        expected_step: int,
        name: str,
    ) -> None:
        try:
            state = torch.load(
                io.BytesIO(payload),
                map_location=self.device,
                weights_only=False,
            )
        except BaseException as error:
            raise ValueError(f"resume snapshot is malformed: {name}") from error
        if type(state) is not dict or set(state) != _SNAPSHOT_FIELDS:
            raise ValueError(f"resume snapshot fields are foreign: {name}")
        if (
            type(state["step"]) is not int
            or state["step"] != expected_step
            or not 0 <= expected_step <= self.max_steps
        ):
            raise ValueError(f"resume snapshot step is invalid: {name}")
        if (
            type(state["world_size"]) is not int
            or state["world_size"] != self.world_size
        ):
            raise ValueError(f"resume snapshot world size is invalid: {name}")
        if not strict_json_identity(
            state["data_provenance"],
            self.data.provenance,
        ):
            raise ValueError(f"resume snapshot provenance is invalid: {name}")
        if not strict_json_identity(
            state["model_cfg"],
            self._raw_model().cfg.__dict__,
        ):
            raise ValueError(f"resume snapshot model config is invalid: {name}")
        self._validate_model_state(state["model"])

    def _inspect_resume_snapshots(
        self,
        output: DurableOutput,
        *,
        atomic_temporaries: frozenset[str] = frozenset(),
        linked_atomic_targets: frozenset[str] = frozenset(),
    ) -> tuple[str, ...]:
        stale = []
        for name in output.snapshots.entries():
            if name in atomic_temporaries:
                continue
            active_match = _SNAPSHOT_NAME.fullmatch(name)
            quarantined_match = _QUARANTINED_SNAPSHOT_NAME.fullmatch(name)
            if active_match is None and quarantined_match is None:
                raise ValueError(
                    f"resume snapshots contain foreign entry: {name}"
                )
            metadata = output.snapshots.entry_metadata(
                name,
                label="resume snapshot",
            )
            expected_nlink = 2 if name in linked_atomic_targets else 1
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != expected_nlink
            ):
                raise ValueError(
                    f"resume snapshot is symlinked, linked, or unsafe: {name}"
                )
            expected_step = int(
                (
                    active_match
                    if active_match is not None
                    else quarantined_match
                ).group(1)
            )
            if expected_step not in self._snapshot_step_set:
                raise ValueError(
                    "resume snapshot step is not configured by "
                    f"snapshot_steps: {expected_step}"
                )
            payload = output.snapshots.read_regular(
                name,
                label="resume snapshot",
            ).payload
            self._validate_snapshot_bytes(
                payload,
                expected_step=expected_step,
                name=name,
            )
            if active_match is not None and expected_step > self.step:
                stale.append(name)
        return tuple(stale)

    def _reconcile_resume_artifacts(self, output: DurableOutput) -> None:
        (
            atomic_cleanups,
            snapshot_temporaries,
            linked_snapshot_targets,
        ) = self._inspect_atomic_writer_entries(output)
        retained_log, truncate_log = self._inspect_resume_log(output)
        stale_snapshots = self._inspect_resume_snapshots(
            output,
            atomic_temporaries=snapshot_temporaries,
            linked_atomic_targets=linked_snapshot_targets,
        )
        for directory, name, metadata in atomic_cleanups:
            directory.unlink_regular(
                name,
                expected=metadata,
                label="resume atomic temporary",
            )
        for name in stale_snapshots:
            quarantine_name = (
                f".quarantine-{name}-after-step{self.step:07d}-"
                f"{secrets.token_hex(16)}"
            )
            output.snapshots.quarantine_regular(name, quarantine_name)
        if truncate_log:
            assert retained_log is not None
            output.root.write_bytes(
                "log.jsonl",
                retained_log,
                replace=True,
            )

    def _validate_model_state(self, state: object) -> None:
        if not isinstance(state, dict):
            raise ValueError("checkpoint model state must be a mapping")
        current = self._raw_model().state_dict()
        if tuple(state) != tuple(current):
            raise ValueError("checkpoint model fields do not match the model")
        for name, expected in current.items():
            actual = state[name]
            if (
                not isinstance(actual, torch.Tensor)
                or actual.shape != expected.shape
                or actual.dtype != expected.dtype
            ):
                raise ValueError(f"checkpoint model tensor is incompatible: {name}")

    def _validate_optimizer_state(self, state: object, *, saved_step: int) -> None:
        if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
            raise ValueError("checkpoint optimizer fields do not match AdamW")
        candidate_groups = state["param_groups"]
        candidate_state = state["state"]
        current = self.opt.state_dict()
        current_groups = current["param_groups"]
        if (
            not isinstance(candidate_groups, list)
            or len(candidate_groups) != len(current_groups)
            or not isinstance(candidate_state, dict)
        ):
            raise ValueError("checkpoint optimizer structure is incompatible")

        parameter_ids = []
        for candidate, expected in zip(
            candidate_groups,
            current_groups,
            strict=True,
        ):
            if (
                not isinstance(candidate, dict)
                or set(candidate) != set(expected)
                or not isinstance(candidate.get("params"), list)
                or candidate["params"] != expected["params"]
            ):
                raise ValueError("checkpoint optimizer parameter groups are incompatible")
            for key, expected_value in expected.items():
                if key in {"params", "lr"}:
                    continue
                if candidate[key] != expected_value:
                    raise ValueError(
                        f"checkpoint optimizer setting is incompatible: {key}"
                    )
            learning_rate = candidate["lr"]
            if (
                isinstance(learning_rate, bool)
                or not isinstance(learning_rate, (int, float))
                or not math.isfinite(learning_rate)
                or learning_rate < 0
            ):
                raise ValueError("checkpoint optimizer learning rate is invalid")
            parameter_ids.extend(candidate["params"])
        if (
            any(type(parameter_id) is not int for parameter_id in parameter_ids)
            or len(parameter_ids) != len(set(parameter_ids))
        ):
            raise ValueError("checkpoint optimizer parameter identifiers are invalid")
        if saved_step == 0:
            if candidate_state:
                raise ValueError("step-zero checkpoint optimizer state must be empty")
            return
        if (
            any(type(parameter_id) is not int for parameter_id in candidate_state)
            or set(candidate_state) != set(parameter_ids)
        ):
            raise ValueError("checkpoint optimizer state does not bind every parameter")

        parameters = [
            parameter
            for group in self.opt.param_groups
            for parameter in group["params"]
        ]
        if len(parameters) != len(parameter_ids):
            raise ValueError("checkpoint optimizer parameter count is incompatible")
        parameter_by_id = dict(zip(parameter_ids, parameters, strict=True))
        for parameter_id, record in candidate_state.items():
            if (
                not isinstance(record, dict)
                or set(record) != {"step", "exp_avg", "exp_avg_sq"}
            ):
                raise ValueError("checkpoint AdamW state fields are incompatible")
            parameter = parameter_by_id[parameter_id]
            for field_name in ("exp_avg", "exp_avg_sq"):
                value = record[field_name]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != parameter.shape
                    or value.dtype != parameter.dtype
                ):
                    raise ValueError(
                        f"checkpoint AdamW tensor is incompatible: {field_name}"
                    )
            step_value = record["step"]
            if (
                not isinstance(step_value, torch.Tensor)
                or step_value.ndim != 0
                or step_value.dtype != torch.float32
                or not torch.isfinite(step_value).item()
                or step_value.item() < 0
                or step_value.item() != saved_step
            ):
                raise ValueError("checkpoint AdamW step is invalid")

    def _validate_rng_record(self, record: object, *, rank: int) -> None:
        if type(record) is not dict or set(record) != _RNG_FIELDS:
            raise ValueError(f"checkpoint RNG fields are invalid for rank {rank}")
        if type(record["python"]) is not tuple:
            raise ValueError(
                f"checkpoint Python RNG state is invalid for rank {rank}"
            )
        try:
            random.Random().setstate(record["python"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"checkpoint Python RNG state is invalid for rank {rank}"
            ) from error
        numpy_state = record["numpy"]
        if (
            type(numpy_state) is not tuple
            or len(numpy_state) != 5
            or type(numpy_state[0]) is not str
            or type(numpy_state[1]) is not np.ndarray
            or numpy_state[1].dtype != np.uint32
            or type(numpy_state[2]) is not int
            or type(numpy_state[3]) is not int
            or not isinstance(numpy_state[4], float)
        ):
            raise ValueError(f"checkpoint NumPy RNG state is invalid for rank {rank}")
        try:
            np.random.RandomState().set_state(numpy_state)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"checkpoint NumPy RNG state is invalid for rank {rank}"
            ) from error
        torch_state = record["torch"]
        if (
            not isinstance(torch_state, torch.Tensor)
            or torch_state.dtype != torch.uint8
            or torch_state.ndim != 1
            or torch_state.numel() != torch.get_rng_state().numel()
        ):
            raise ValueError(f"checkpoint torch RNG state is invalid for rank {rank}")
        cuda_state = record["cuda"]
        if self.device.startswith("cuda"):
            expected_cuda_items = torch.cuda.get_rng_state(self.local_rank).numel()
            if (
                not isinstance(cuda_state, torch.Tensor)
                or cuda_state.dtype != torch.uint8
                or cuda_state.ndim != 1
                or cuda_state.numel() != expected_cuda_items
            ):
                raise ValueError(
                    f"checkpoint CUDA RNG state is invalid for rank {rank}"
                )
        elif cuda_state is not None:
            raise ValueError(
                f"CPU checkpoint CUDA RNG state must be null for rank {rank}"
            )

    def _validate_checkpoint_state(self, state: object) -> dict:
        if type(state) is not dict or set(state) != _CHECKPOINT_FIELDS:
            raise ValueError("checkpoint top-level fields do not match the contract")
        version = state["checkpoint_version"]
        if type(version) is not int or version != 3:
            raise ValueError("checkpoint version is incompatible")
        saved_world_size = state["world_size"]
        if type(saved_world_size) is not int or saved_world_size != self.world_size:
            raise ValueError(
                "checkpoint world size does not match current world size"
            )
        saved_step = state["step"]
        if type(saved_step) is not int or not 0 <= saved_step <= self.max_steps:
            raise ValueError("checkpoint step is invalid for current config")
        saved_cfg = state["cfg"]
        if type(saved_cfg) is not dict or set(saved_cfg) != set(self.cfg):
            raise ValueError("checkpoint config fields do not match current config")
        for key in set(self.cfg) - _DATA_LOCATION_KEYS:
            if _jsonable(saved_cfg[key]) != _jsonable(self.cfg[key]):
                raise ValueError(
                    f"checkpoint training config does not match current config: {key}"
                )
        fingerprint = state["config_fingerprint"]
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or fingerprint != self.config_fingerprint
            or resume_config_fingerprint(saved_cfg, self.model_cfg) != fingerprint
        ):
            raise ValueError(
                "checkpoint training config does not match current config"
            )
        provenance = state["data_provenance"]
        if (
            type(provenance) is not dict
            or not strict_json_identity(provenance, self.data.provenance)
        ):
            raise ValueError(
                "checkpoint data provenance does not match current dataset"
            )
        if not isinstance(state["data"], dict) or set(state["data"]) != _DATA_STATE_FIELDS:
            raise ValueError("checkpoint data fields do not match the contract")
        global_cursor = self.data.validate_state_dict(state["data"])
        if global_cursor != saved_step * self.tokens_per_step:
            raise ValueError("checkpoint step and global data cursor are inconsistent")

        self._validate_model_state(state["model"])
        self._validate_optimizer_state(state["opt"], saved_step=saved_step)
        rng_by_rank = state["rng_by_rank"]
        if type(rng_by_rank) is not list or len(rng_by_rank) != self.world_size:
            raise ValueError("checkpoint RNG state does not match current world size")
        for rank, record in enumerate(rng_by_rank):
            self._validate_rng_record(record, rank=rank)
        return state

    def _apply_checkpoint_state(self, state: dict) -> None:
        self._raw_model().load_state_dict(state["model"], strict=True)
        self.opt.load_state_dict(state["opt"])
        self.data.load_state_dict(state["data"])
        self.step = state["step"]
        rank_rng = state["rng_by_rank"][self.rank]
        random.setstate(rank_rng["python"])
        np.random.set_state(rank_rng["numpy"])
        torch.set_rng_state(rank_rng["torch"].cpu())
        if self.device.startswith("cuda"):
            torch.cuda.set_rng_state(
                rank_rng["cuda"].cpu(),
                device=self.local_rank,
            )

    def save_ckpt(self) -> None:
        rng_by_rank = self._rng_states_by_rank()
        data_states = self._data_states_by_rank()

        def write_checkpoint() -> None:
            raw = self._raw_model()
            state = {
                "checkpoint_version": 3,
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
            if self._output is None:
                raise RuntimeError("durable output directory is not initialized")
            self._output.root.write_atomic(
                "ckpt.pt",
                lambda handle: torch.save(state, handle),
                replace=True,
            )

        self._rank0_action(write_checkpoint, "checkpoint write")

    def load_ckpt(
        self,
        path: str | Path | None = None,
        *,
        sha256: str | None = None,
        _default_path: bool = False,
    ) -> None:
        checkpoint_path = Path(path) if path is not None else self.ckpt_path
        if checkpoint_path != self.ckpt_path and sha256 is None and not _default_path:
            raise ValueError("external checkpoint load requires resume_sha256")

        state = None
        local_exception: BaseException | None = None
        checkpoint_sha256 = None
        try:
            pinned = read_regular_path(
                checkpoint_path,
                label="resume checkpoint",
                expected_sha256=sha256,
            )
            checkpoint_sha256 = pinned.sha256
            parsed = torch.load(
                io.BytesIO(pinned.payload),
                map_location=self.device,
                weights_only=False,
            )
            state = self._validate_checkpoint_state(parsed)
            local_status = {
                "ok": True,
                "checkpoint_sha256": checkpoint_sha256,
                "config_fingerprint": state["config_fingerprint"],
                "data_provenance_sha256": _canonical_json_hash(
                    state["data_provenance"]
                ),
            }
        except BaseException as caught:
            local_exception = caught
            local_status = {
                "ok": False,
                "checkpoint_sha256": checkpoint_sha256,
                "error_type": type(caught).__name__,
                "error": str(caught),
            }

        if self.dist.process_group_initialized:
            gathered: list[dict | None] = [None] * self.world_size
            torch.distributed.all_gather_object(gathered, local_status)
        else:
            gathered = [local_status]
        failures = [
            (rank, status)
            for rank, status in enumerate(gathered)
            if status is None or not status.get("ok", False)
        ]
        if failures:
            if not self.dist.process_group_initialized and local_exception is not None:
                raise local_exception
            details = "; ".join(
                (
                    f"rank {rank}: missing status"
                    if status is None
                    else f"rank {rank}: {status['error_type']}: {status['error']}"
                )
                for rank, status in failures
            )
            raise RuntimeError(f"coordinated checkpoint load failed: {details}")
        if any(status != gathered[0] for status in gathered[1:]):
            raise RuntimeError(
                "ranks validated different checkpoint, config, or data hashes"
            )
        assert state is not None
        self._apply_checkpoint_state(state)

    def save_snapshot(self) -> None:
        def write_snapshot() -> None:
            raw = self._raw_model()
            if self._output is None:
                raise RuntimeError("durable output directory is not initialized")
            state = {
                "model": raw.state_dict(),
                "step": self.step,
                "model_cfg": raw.cfg.__dict__,
                "world_size": self.world_size,
                "data_provenance": self.data.provenance,
            }
            self._output.snapshots.write_atomic(
                f"step{self.step:07d}.pt",
                lambda handle: torch.save(state, handle),
                replace=False,
            )

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
            operational_step_started = (
                time.perf_counter()
                if self._capture_operational_metrics
                else None
            )
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
            if operational_step_started is not None:
                operational_elapsed = max(
                    1e-9,
                    time.perf_counter() - operational_step_started,
                )
                self.operational_step_tok_s.append(
                    float(self.tokens_per_step / operational_elapsed)
                )

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
                    if self._output is None:
                        raise RuntimeError(
                            "durable output directory is not initialized"
                        )
                    self._output.root.append_bytes(
                        "log.jsonl",
                        (json.dumps(row) + "\n").encode("utf-8"),
                    )

                self._rank0_action(write_log_row, "log write")
                t0 = time.time()
                tokens_seen = 0
            if self.step in self._snapshot_step_set:
                self.save_snapshot()
            checkpoint_due = self._broadcast_master_bool(
                time.time() - last_ckpt > self.ckpt_seconds
            )
            checkpoint_requested = self._service_checkpoint_request(
                checkpoint_due=checkpoint_due,
            )
            if checkpoint_due or checkpoint_requested:
                last_ckpt = time.time()
        self.save_ckpt()
        return running if running is not None else float("nan")


def train(
    cfg: dict,
    resume: str = "none",
    *,
    resume_path: str | Path | None = None,
    resume_sha256: str | None = None,
    operational_steps: int | None = None,
) -> Trainer:
    if operational_steps is not None and (
        type(operational_steps) is not int
        or operational_steps <= 0
        or operational_steps > sys.maxsize
    ):
        raise ValueError(
            "operational_steps must be a positive exact platform integer"
        )
    trainer = Trainer(
        cfg,
        resume=resume,
        resume_path=resume_path,
        resume_sha256=resume_sha256,
    )
    if operational_steps is not None:
        trainer._capture_operational_metrics = True
        trainer.operational_start_step = trainer.step
    if resume == "auto":
        if trainer.is_master:
            print(f"resumed from step {trainer.step}")
    try:
        trainer.train_steps(operational_steps)
    except BaseException:
        trainer.close()
        raise
    return trainer
