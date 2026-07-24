#!/opt/conda/bin/python
"""Measured host/container workers for selected AWS GPU qualification.

The controller invokes this script only through closed argv rendered by
``cluster.aws.qualification``.  Unit tests inject command results; this module
contains the live probe implementation but does not execute at import time.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np


class QualificationWorkerError(ValueError):
    """A measured worker fact or capability failed closed."""


def load_reviewed_geometry(
    config_path: Path | str,
    *,
    arm: str,
) -> dict[str, object]:
    """Load the exact frozen d360m qualification geometry for one arm."""

    import yaml

    if arm not in {"dense", "split90"}:
        raise QualificationWorkerError("reviewed qualification arm is invalid")
    path = Path(config_path)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise QualificationWorkerError(
            "reviewed qualification config cannot be loaded"
        ) from error
    expected_sidecar = (
        "dense_target_weights"
        if arm == "dense"
        else "split90_target_weights"
    )
    expected = {
        "schema_version": 2,
        "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
        "condition": arm,
        "model": "d360m",
        "ctx": 1024,
        "train_corpus": "dataset/receipt.json",
        "sidecar_name": expected_sidecar,
        "micro_batch_size": 8,
        "tokens_per_step": 524_288,
        "max_steps": 13_582,
        "total_tokens": 7_120_879_616,
        "lr": 0.001,
        "warmup_steps": 300,
        "weight_decay": 0.1,
        "compile": True,
        "device": "cuda",
        "log_every": 20,
        "eval_every": 250,
        "snapshot_steps": [1_358, 3_396, 6_791, 10_187, 13_582],
        "ckpt_minutes": 30,
    }
    if not isinstance(value, dict):
        raise QualificationWorkerError("reviewed qualification config is not an object")
    for field, expected_value in expected.items():
        if type(value.get(field)) is not type(expected_value) or value.get(
            field
        ) != expected_value:
            raise QualificationWorkerError(
                f"reviewed qualification config {field} differs"
            )
    seed = value.get("seed")
    run_id = value.get("run_id")
    out_dir = value.get("out_dir")
    if (
        type(seed) is not int
        or seed not in range(10)
        or run_id != f"memorysplit-v3-360m-s{seed}-{arm}"
        or out_dir != f"runs/seed-{seed}/{arm}"
        or set(value) != {*expected, "seed", "run_id", "out_dir"}
        or value["max_steps"] * value["tokens_per_step"]
        != value["total_tokens"]
    ):
        raise QualificationWorkerError(
            "reviewed qualification config identity or token geometry differs"
        )
    return {
        "arm": arm,
        "model": value["model"],
        "ctx": value["ctx"],
        "micro_batch_size": value["micro_batch_size"],
        "tokens_per_step": value["tokens_per_step"],
        "total_tokens": value["total_tokens"],
        "max_steps": value["max_steps"],
        "sidecar_name": value["sidecar_name"],
        "compile": value["compile"],
    }


def _canonical(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise QualificationWorkerError(
            "qualification worker value is not canonical JSON"
        ) from error


def _run(argv: Sequence[str]) -> bytes:
    if not argv or not argv[0].startswith("/"):
        raise QualificationWorkerError("qualification worker argv is not absolute")
    try:
        completed = subprocess.run(
            list(argv),
            cwd="/",
            env={
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/local/bin:/usr/bin:/bin",
            },
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise QualificationWorkerError("qualification host command failed") from error
    if completed.returncode != 0:
        raise QualificationWorkerError("qualification host command returned error")
    return bytes(completed.stdout if completed.stdout else completed.stderr)


_HOST_COMMANDS = {
    "cuda": ("/usr/local/cuda/bin/nvcc", "--version"),
    "nvidia_driver": (
        "/usr/bin/nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader,nounits",
    ),
    "fabric_manager": ("/usr/bin/nv-fabricmanager", "--version"),
    "docker": ("/usr/bin/docker", "version", "--format", "{{.Server.Version}}"),
    "nvidia_container_runtime": (
        "/usr/bin/nvidia-container-runtime",
        "--version",
    ),
    "aws_cli": ("/usr/local/bin/aws", "--version"),
    "kernel": ("/usr/bin/uname", "--kernel-release"),
    "efa": ("/usr/bin/cat", "/opt/amazon/efa_installed_packages"),
    "ofi_nccl": (
        "/usr/bin/strings",
        "/opt/amazon/ofi-nccl/lib/libnccl-net.so",
    ),
    "nvlsm": ("/usr/bin/nvlsm", "--version"),
}


def _version(field: str, data: bytes) -> str:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise QualificationWorkerError(f"host {field} output is not UTF-8") from error
    patterns = {
        "cuda": r"\brelease\s+([0-9]+(?:\.[0-9]+)+)",
        "efa": r"#\s*EFA installer version:\s*([0-9]+(?:\.[0-9]+)+)",
        "ofi_nccl": r"aws-ofi-nccl\s+([0-9]+(?:\.[0-9]+)+)",
        "aws_cli": r"aws-cli/([0-9]+(?:\.[0-9]+)+)",
    }
    match = re.search(patterns.get(field, r"([0-9]+(?:\.[0-9]+)+)"), text, re.I)
    if match is None:
        raise QualificationWorkerError(f"host {field} has no measured version")
    return match.group(1)


def _memory_gib() -> int:
    try:
        text = Path("/proc/meminfo").read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise QualificationWorkerError("host memory facts are unavailable") from error
    match = re.search(r"(?m)^MemTotal:\s+([0-9]+)\s+kB$", text)
    if match is None:
        raise QualificationWorkerError("host memory facts are malformed")
    return round(int(match.group(1)) / (1024 * 1024))


def _topology(data: bytes) -> dict[str, object]:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise QualificationWorkerError("NVLink topology is not UTF-8") from error
    lines = [line.split() for line in text.splitlines() if line.strip()]
    headers = [f"GPU{index}" for index in range(8)]
    if not lines or lines[0][:8] != headers:
        raise QualificationWorkerError("NVLink topology header is incomplete")
    rows = lines[1:9]
    if len(rows) != 8:
        raise QualificationWorkerError("NVLink topology rows are incomplete")
    for left, row in enumerate(rows):
        if len(row) < 9 or row[0] != f"GPU{left}":
            raise QualificationWorkerError("NVLink topology row is malformed")
        for right, link in enumerate(row[1:9]):
            if (left == right and link != "X") or (
                left != right and re.fullmatch(r"NV[1-9][0-9]*", link) is None
            ):
                raise QualificationWorkerError(
                    "NVLink topology is not fully connected"
                )
    return {
        "matrix_sha256": hashlib.sha256(data).hexdigest(),
        "nvlink_active": True,
        "fully_connected": True,
    }


def collect_hardware(
    profile_id: str,
    *,
    region: str,
    runner: Callable[[Sequence[str]], bytes] = _run,
) -> dict[str, object]:
    """Measure host CPU/RAM, GPUs, NVMe, topology, and software versions."""

    expected = {
        "aws-p5.48xlarge-v3": {
            "instance_type": "p5.48xlarge",
            "memory_gib": 2048,
            "gpu_names": {"NVIDIA H100 80GB", "NVIDIA H100 80GB HBM3"},
        },
        "aws-p6-b300.48xlarge-v3": {
            "instance_type": "p6-b300.48xlarge",
            "memory_gib": 4096,
            "gpu_names": {"NVIDIA B300"},
        },
    }.get(profile_id)
    if expected is None:
        raise QualificationWorkerError("selected profile ID is unsupported")
    gpu_output = runner(
        (
            "/usr/bin/nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        )
    )
    try:
        gpu_lines = gpu_output.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as error:
        raise QualificationWorkerError("GPU inventory is not UTF-8") from error
    gpu_devices: list[dict[str, object]] = []
    for expected_index, line in enumerate(gpu_lines):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise QualificationWorkerError("GPU inventory row is malformed")
        try:
            index = int(parts[0])
            memory_mib = int(parts[2])
        except ValueError as error:
            raise QualificationWorkerError("GPU inventory row is malformed") from error
        if index != expected_index or parts[1] not in expected["gpu_names"]:
            raise QualificationWorkerError("GPU inventory differs from profile")
        gpu_devices.append(
            {"index": index, "name": parts[1], "memory_mib": memory_mib}
        )
    if len(gpu_devices) != 8:
        raise QualificationWorkerError("GPU inventory must contain eight devices")

    try:
        block = json.loads(
            runner(
                (
                    "/usr/bin/lsblk",
                    "--json",
                    "--bytes",
                    "--tree",
                    "--output",
                    "PATH,TYPE,MODEL,SIZE,MOUNTPOINTS,FSTYPE,PTTYPE",
                )
            )
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationWorkerError("NVMe inventory is invalid JSON") from error
    rows = block.get("blockdevices") if isinstance(block, dict) else None
    if not isinstance(rows, list):
        raise QualificationWorkerError("NVMe inventory has no devices")
    instance_store: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("model") != (
            "Amazon EC2 NVMe Instance Storage"
        ):
            continue
        if (
            row.get("type") != "disk"
            or type(row.get("size")) is not int
            or row["size"] != 3_840_000_000_000
            or not isinstance(row.get("path"), str)
            or not row["path"].startswith("/dev/")
            or any(point not in {None, ""} for point in row.get("mountpoints", []))
            or row.get("fstype") not in {None, ""}
            or row.get("pttype") not in {None, ""}
        ):
            raise QualificationWorkerError("NVMe geometry is unsafe")
        instance_store.append(
            {
                "path": row["path"],
                "model": row["model"],
                "bytes": row["size"],
            }
        )
    instance_store.sort(key=lambda item: item["path"])
    if len(instance_store) != 8:
        raise QualificationWorkerError("NVMe inventory must contain eight devices")

    if re.fullmatch(r"[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+", region) is None:
        raise QualificationWorkerError("selected AWS region is invalid")

    def metadata(flag: str, *, label: str) -> str:
        try:
            text = runner(("/usr/bin/ec2-metadata", flag)).decode(
                "utf-8",
                errors="strict",
            )
        except UnicodeDecodeError as error:
            raise QualificationWorkerError(f"{label} is not UTF-8") from error
        value = text.split(":", 1)[-1].strip()
        if not value or any(character in value for character in "\x00\n\r"):
            raise QualificationWorkerError(f"{label} is invalid")
        return value

    instance_id = metadata("-i", label="instance ID")
    instance_type = metadata("-t", label="instance type")
    ami_id = metadata("-a", label="AMI ID")
    architecture = runner(("/usr/bin/uname", "--machine")).decode(
        "ascii",
        errors="strict",
    ).strip()
    account_id = runner(
        (
            "/usr/local/bin/aws",
            "--no-cli-pager",
            "--region",
            region,
            "sts",
            "get-caller-identity",
            "--query",
            "Account",
            "--output",
            "text",
        )
    ).decode("ascii", errors="strict").strip()
    boot_id = runner(
        ("/usr/bin/cat", "/proc/sys/kernel/random/boot_id")
    ).decode("ascii").strip()
    if (
        instance_type != expected["instance_type"]
        or architecture != "x86_64"
        or re.fullmatch(r"[0-9]{12}", account_id) is None
        or re.fullmatch(r"i-[0-9a-f]{8,17}", instance_id) is None
        or re.fullmatch(r"ami-[0-9a-f]{8,17}", ami_id) is None
    ):
        raise QualificationWorkerError("authenticated instance identity differs")
    vcpus = os.cpu_count()
    memory_gib = _memory_gib()
    if vcpus != 192 or memory_gib != expected["memory_gib"]:
        raise QualificationWorkerError("host vCPU/RAM geometry differs from profile")
    host_facts = {
        field: _version(field, runner(argv))
        for field, argv in _HOST_COMMANDS.items()
    }
    return {
        "account_id": account_id,
        "instance_id": instance_id,
        "boot_id": boot_id,
        "ami_id": ami_id,
        "instance_type": instance_type,
        "architecture": architecture,
        "vcpus": vcpus,
        "memory_gib": memory_gib,
        "gpu_names": [item["name"] for item in gpu_devices],
        "instance_store": instance_store,
        "gpu_devices": gpu_devices,
        "host_facts": host_facts,
        "topology": _topology(
            runner(("/usr/bin/nvidia-smi", "topo", "-m"))
        ),
    }


def _torch_versions(torch) -> dict[str, str]:
    nccl = torch.cuda.nccl.version()
    nccl_version = (
        ".".join(map(str, nccl))
        if isinstance(nccl, tuple)
        else str(nccl or "")
    )
    return {
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "cuda": str(torch.version.cuda or ""),
        "cudnn": str(torch.backends.cudnn.version() or ""),
        "nccl": nccl_version,
    }


def _one_step_resume(torch, device) -> tuple[bool, str]:
    random.seed(1731)
    np.random.seed(1731)
    torch.manual_seed(1731)
    torch.cuda.manual_seed_all(1731)
    model = torch.nn.Linear(32, 32, bias=True, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)

    def step(candidate, selected_optimizer):
        selected_optimizer.zero_grad(set_to_none=True)
        scale = random.random() + float(np.random.random())
        inputs = torch.rand(
            (32, 32),
            device=device,
            dtype=torch.float32,
        ) * scale
        loss = candidate(inputs).square().mean()
        loss.backward()
        selected_optimizer.step()
        return loss.detach()

    first_progress_loss = step(model, optimizer)
    if not torch.isfinite(first_progress_loss):
        raise QualificationWorkerError("checkpoint progress loss is not finite")
    checkpoint = io.BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "progress_step": 1,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(device),
            },
        },
        checkpoint,
    )
    checkpoint_bytes = checkpoint.getvalue()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()

    resumed = torch.nn.Linear(32, 32, bias=True, device=device)
    resumed_optimizer = torch.optim.AdamW(
        resumed.parameters(),
        lr=1e-3,
        fused=True,
    )
    state = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location=device,
        weights_only=False,
    )
    resumed.load_state_dict(state["model"])
    resumed_optimizer.load_state_dict(state["optimizer"])
    if state.get("progress_step") != 1 or set(state.get("rng", {})) != {
        "python",
        "numpy",
        "torch",
        "cuda",
    }:
        raise QualificationWorkerError("checkpoint RNG/progress state is invalid")
    first_loss = step(model, optimizer)
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    torch.cuda.set_rng_state(state["rng"]["cuda"], device=device)
    resumed_loss = step(resumed, resumed_optimizer)
    torch.cuda.synchronize(device)
    exact = torch.equal(first_loss, resumed_loss) and all(
        torch.equal(left, right)
        for left, right in zip(model.parameters(), resumed.parameters(), strict=True)
    )
    return bool(exact), checkpoint_sha256


def collect_container_capabilities() -> dict[str, object]:
    """Execute every required framework capability in the running image."""

    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise QualificationWorkerError("CUDA device is unavailable")
    device = torch.device("cuda:0")
    left = torch.randn((64, 64), dtype=torch.bfloat16, device=device)
    right = torch.randn((64, 64), dtype=torch.bfloat16, device=device)
    bf16 = (left @ right).dtype is torch.bfloat16

    query = torch.randn(
        (2, 4, 16, 32),
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    attended = functional.scaled_dot_product_attention(query, query, query)
    sdpa_forward = attended.shape == query.shape and torch.isfinite(attended).all()
    attended.float().sum().backward()
    sdpa_backward = query.grad is not None and torch.isfinite(query.grad).all()

    compiled = torch.compile(lambda value: (value * value).sum())
    torch_compile = torch.isfinite(compiled(left.float()))
    parameter = torch.nn.Parameter(torch.ones(8, device=device))
    fused = torch.optim.AdamW([parameter], lr=1e-3, fused=True)
    fused.zero_grad(set_to_none=True)
    parameter.square().sum().backward()
    fused.step()
    fused_adamw = True
    checkpoint_resume_exact, checkpoint_sha256 = _one_step_resume(torch, device)
    return {
        "versions": _torch_versions(torch),
        "bf16": bool(bf16),
        "sdpa_forward": bool(sdpa_forward),
        "sdpa_backward": bool(sdpa_backward),
        "torch_compile": bool(torch_compile),
        "fused_adamw": fused_adamw,
        "one_step_train": True,
        "checkpoint_resume_exact": checkpoint_resume_exact,
        "checkpoint_sha256": checkpoint_sha256,
        "resumed_checkpoint_sha256": checkpoint_sha256,
    }


def _parse_gpu_ids(value: str) -> list[int]:
    try:
        ids = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise QualificationWorkerError("GPU group IDs are invalid") from error
    if len(ids) != 4 or len(set(ids)) != 4 or any(item not in range(8) for item in ids):
        raise QualificationWorkerError("GPU group must contain four approved IDs")
    return ids


def _parse_affinity(value: str) -> list[int]:
    match = re.fullmatch(r"([0-9]+)-([0-9]+)", value)
    if match is None:
        raise QualificationWorkerError("CPU affinity is invalid")
    start, end = (int(item) for item in match.groups())
    if start > end:
        raise QualificationWorkerError("CPU affinity is reversed")
    return [start, end]


def _trainer_tree_equivalent(
    torch,
    left: object,
    right: object,
    *,
    tolerance: float,
) -> bool:
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and torch.allclose(
                left.detach().cpu(),
                right.detach().cpu(),
                rtol=0.0,
                atol=tolerance,
                equal_nan=False,
            )
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return tuple(left) == tuple(right) and all(
            _trainer_tree_equivalent(
                torch,
                left[key],
                right[key],
                tolerance=tolerance,
            )
            for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _trainer_tree_equivalent(torch, a, b, tolerance=tolerance)
            for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, float):
        return (
            math.isfinite(left)
            and math.isfinite(right)
            and abs(left - right) <= tolerance
        )
    return left == right


def _trainer_state(torch, trainer) -> dict[str, object]:
    return {
        "model": {
            name: tensor.detach().cpu().clone()
            for name, tensor in trainer._raw_model().state_dict().items()
        },
        "optimizer": copy.deepcopy(trainer.opt.state_dict()),
        "data": copy.deepcopy(trainer.data.state_dict()),
        "step": trainer.step,
    }


def _measure_trainer_capabilities(torch, trainer) -> dict[str, bool]:
    from train.data import synchronized_rank_batch_plan

    plan = synchronized_rank_batch_plan(
        global_cursor=trainer.data.global_cursor,
        total_sequences=trainer.sequences_per_step,
        ctx=trainer.data.ctx,
        micro_batch_size=trainer.micro_bs,
        rank=trainer.rank,
        world_size=trainer.world_size,
    )
    batch_slice = next(item for item in plan if item is not None)
    x, y, weights = trainer.data.weighted_batch_from_slice(batch_slice)
    trainer.opt.zero_grad(set_to_none=True)
    with trainer._autocast():
        logits, loss = trainer.model(
            x,
            y,
            target_weights=weights,
            loss_reduction="mean",
        )
    if loss is None:
        raise QualificationWorkerError("reviewed model returned no qualification loss")
    loss.backward()
    gradient_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in trainer.model.parameters()
    )
    compiled = trainer.model
    if isinstance(compiled, torch.nn.parallel.DistributedDataParallel):
        compiled = compiled.module
    result = {
        "bf16_output_finite": (
            logits.dtype is torch.bfloat16
            and torch.isfinite(logits).all().item()
            and torch.isfinite(loss).item()
        ),
        "sdpa_forward_finite": torch.isfinite(logits).all().item(),
        "sdpa_backward_finite": gradient_finite,
        "compiled_model": bool(
            trainer.cfg.get("compile") and hasattr(compiled, "_orig_mod")
        ),
        "fused_adamw": trainer.opt.defaults.get("fused") is True,
        "gradient_finite": gradient_finite,
    }
    trainer.opt.zero_grad(set_to_none=True)
    return {name: bool(value) for name, value in result.items()}


def _sidecar_evidence(torch, trainer) -> dict[str, object]:
    from train.data import synchronized_rank_batch_plan

    plan = synchronized_rank_batch_plan(
        global_cursor=trainer.data.global_cursor,
        total_sequences=trainer.sequences_per_step,
        ctx=trainer.data.ctx,
        micro_batch_size=trainer.micro_bs,
        rank=trainer.rank,
        world_size=trainer.world_size,
    )
    local_sum = 0.0
    local_nonzero = 0
    local_items = 0
    for batch_slice in plan:
        if batch_slice is None:
            continue
        _x, _y, weights = trainer.data.weighted_batch_from_slice(batch_slice)
        local_sum += float(weights.sum().item())
        local_nonzero += int(weights.ne(0).sum().item())
        local_items += weights.numel()
    measured = torch.tensor(
        [local_sum, float(local_nonzero), float(local_items)],
        dtype=torch.float64,
        device=trainer.device,
    )
    torch.distributed.all_reduce(measured)
    weights = trainer.data.provenance.get("weights")
    if not isinstance(weights, dict):
        raise QualificationWorkerError("reviewed sidecar provenance is missing")
    return {
        "name": trainer.data.provenance.get("sidecar_name"),
        "stream_sha256": weights.get("sha256"),
        "items": int(measured[2].item()),
        "nonzero_targets": int(measured[1].item()),
        "target_weight_sum": float(measured[0].item()),
    }


def run_nccl_train(
    *,
    arm: str,
    config: str,
    qualification_root: str,
    gpu_ids: str,
    cpu_affinity: str,
    master_port: int,
    updates: int,
    warmup_updates: int,
) -> dict[str, object] | None:
    """Run one exact reviewed four-rank arm and prove deterministic resume."""

    import torch
    import yaml

    root = Path(qualification_root).resolve(strict=True)
    config_path = Path(config).resolve(strict=True)
    try:
        config_path.relative_to(root)
    except ValueError as error:
        raise QualificationWorkerError(
            "reviewed qualification config escapes its root"
        ) from error
    geometry = load_reviewed_geometry(config_path, arm=arm)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    from train.trainer import Trainer

    if arm not in {"dense", "split90"}:
        raise QualificationWorkerError("qualification arm is invalid")
    selected_gpus = _parse_gpu_ids(gpu_ids)
    selected_affinity = _parse_affinity(cpu_affinity)
    if (
        type(master_port) is not int
        or not 1024 <= master_port <= 65_535
        or updates != 100
        or warmup_updates != 10
    ):
        raise QualificationWorkerError("qualification group geometry is invalid")
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 4 or local_rank not in range(4):
        raise QualificationWorkerError("NCCL group must have exactly four ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    reduced = torch.tensor(float(rank + 1), device=device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    torch.distributed.all_reduce(reduced)
    torch.cuda.synchronize(device)
    all_reduce_seconds = time.perf_counter() - started
    if reduced.item() != 10.0 or not math.isfinite(all_reduce_seconds):
        raise QualificationWorkerError("NCCL all-reduce evidence is invalid")
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        if not isinstance(cfg, dict):
            raise QualificationWorkerError("reviewed training config is not an object")
        output_root = Path(f"/tmp/memorysplit-qualification-{arm}")
        resume_root = Path(f"/tmp/memorysplit-qualification-{arm}-resumed")
        checkpoint_copy = Path(f"/tmp/memorysplit-qualification-{arm}-step.pt")
        if rank == 0:
            for candidate in (output_root, resume_root):
                if candidate.exists():
                    shutil.rmtree(candidate)
            checkpoint_copy.unlink(missing_ok=True)
        torch.distributed.barrier()
        cfg = copy.deepcopy(cfg)
        cfg["out_dir"] = str(output_root)
        torch.cuda.reset_peak_memory_stats(device)
        trainer = Trainer(cfg, resume="none")
        resumed = None
        try:
            capabilities = _measure_trainer_capabilities(torch, trainer)
            sidecar = _sidecar_evidence(torch, trainer)
            trainer._capture_operational_metrics = True
            trainer.operational_start_step = trainer.step
            trainer.train_steps(updates)
            if trainer.step != updates or trainer.data.global_cursor != (
                updates * trainer.tokens_per_step
            ):
                raise QualificationWorkerError(
                    "reviewed training did not make exact progress"
                )
            checkpoint_step = trainer.step
            rates = list(trainer.operational_step_tok_s)
            if rank == 0:
                checkpoint_bytes = trainer.ckpt_path.read_bytes()
                checkpoint_copy.write_bytes(checkpoint_bytes)
            torch.distributed.barrier()
            checkpoint_bytes = checkpoint_copy.read_bytes()
            checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
            checkpoint_hashes: list[str | None] = [None] * world_size
            torch.distributed.all_gather_object(
                checkpoint_hashes,
                checkpoint_sha256,
            )
            checkpoint_sha256 = str(checkpoint_hashes[0])
            if (
                not checkpoint_sha256
                or any(item != checkpoint_sha256 for item in checkpoint_hashes)
            ):
                raise QualificationWorkerError(
                    "ranks disagree on progressed checkpoint"
                )
            checkpoint_state = torch.load(
                checkpoint_copy,
                map_location=device,
                weights_only=False,
            )
            rng_by_rank = checkpoint_state.get("rng_by_rank")
            if (
                checkpoint_state.get("step") != checkpoint_step
                or checkpoint_step < 1
                or not isinstance(rng_by_rank, list)
                or len(rng_by_rank) != world_size
                or any(
                    not isinstance(record, dict)
                    or set(record) != {"python", "numpy", "torch", "cuda"}
                    for record in rng_by_rank
                )
            ):
                raise QualificationWorkerError(
                    "progressed checkpoint RNG/state closure is invalid"
                )
            uninterrupted_loss = trainer.train_steps(1)
            uninterrupted = _trainer_state(torch, trainer)
            if trainer.step != checkpoint_step + 1:
                raise QualificationWorkerError(
                    "uninterrupted matched step did not advance"
                )

            resumed_cfg = copy.deepcopy(cfg)
            resumed_cfg["out_dir"] = str(resume_root)
            resumed = Trainer(
                resumed_cfg,
                resume="auto",
                resume_path=checkpoint_copy,
                resume_sha256=checkpoint_sha256,
            )
            if resumed.step != checkpoint_step:
                raise QualificationWorkerError(
                    "resume did not restore progressed checkpoint step"
                )
            resumed_loss = resumed.train_steps(1)
            resumed_state = _trainer_state(torch, resumed)
            tolerance = 1e-6
            model_equivalent = _trainer_tree_equivalent(
                torch,
                uninterrupted["model"],
                resumed_state["model"],
                tolerance=tolerance,
            )
            optimizer_equivalent = _trainer_tree_equivalent(
                torch,
                uninterrupted["optimizer"],
                resumed_state["optimizer"],
                tolerance=tolerance,
            )
            loss_delta = abs(float(uninterrupted_loss) - float(resumed_loss))
            cursor_before = checkpoint_state["data"]["global_cursor"]
            cursor_after = uninterrupted["data"]["global_cursor"]
            resumed_cursor_after = resumed_state["data"]["global_cursor"]
            if (
                not model_equivalent
                or not optimizer_equivalent
                or not math.isfinite(loss_delta)
                or loss_delta > tolerance
                or cursor_before != checkpoint_step * trainer.tokens_per_step
                or cursor_after != (checkpoint_step + 1) * trainer.tokens_per_step
                or resumed_cursor_after != cursor_after
            ):
                raise QualificationWorkerError(
                    "matched uninterrupted/resumed step is not equivalent"
                )
            if (
                len(rates) != updates
                or any(
                    not math.isfinite(rate) or rate <= 0
                    for rate in rates
                )
            ):
                raise QualificationWorkerError(
                    "reviewed throughput samples are incomplete"
                )
            step_seconds = [
                trainer.tokens_per_step / rate
                for rate in rates
            ]
            peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
            result = {
                "arm": arm,
                "world_size": world_size,
                "gpu_ids": selected_gpus,
                "cpu_affinity": selected_affinity,
                "master_port": master_port,
                "all_reduce_sum": float(reduced.item()),
                "all_reduce_latency_seconds": float(all_reduce_seconds),
                "updates": updates,
                "warmup_updates": warmup_updates,
                "step_seconds": step_seconds,
                "step_tokens": [trainer.tokens_per_step] * updates,
                "peak_memory_bytes": peak_memory_bytes,
                "geometry": geometry,
                "sidecar": sidecar,
                "capabilities": capabilities,
                "resume": {
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_step": checkpoint_step,
                    "next_step": checkpoint_step + 1,
                    "model_equivalent": model_equivalent,
                    "optimizer_equivalent": optimizer_equivalent,
                    "loss_delta": loss_delta,
                    "cursor_before": cursor_before,
                    "cursor_after": cursor_after,
                    "resumed_cursor_after": resumed_cursor_after,
                    "rng_restored": ["python", "numpy", "torch", "cuda"],
                    "tolerance": tolerance,
                },
            }
        finally:
            if resumed is not None:
                resumed.close()
            if "trainer" in locals():
                trainer.close()
        torch.distributed.barrier()
        return result if rank == 0 else None
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--hardware", action="store_true")
    mode.add_argument("--container-capabilities", action="store_true")
    mode.add_argument("--nccl-train", action="store_true")
    parser.add_argument("--profile-id")
    parser.add_argument("--region")
    parser.add_argument("--arm")
    parser.add_argument("--config")
    parser.add_argument("--qualification-root")
    parser.add_argument("--gpu-ids")
    parser.add_argument("--cpu-affinity")
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--warmup-updates", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.hardware:
            if arguments.profile_id is None or arguments.region is None:
                raise QualificationWorkerError(
                    "--hardware requires --profile-id and --region"
                )
            value = collect_hardware(
                arguments.profile_id,
                region=arguments.region,
            )
        elif arguments.container_capabilities:
            value = collect_container_capabilities()
        else:
            required: Mapping[str, object] = {
                "arm": arguments.arm,
                "config": arguments.config,
                "qualification_root": arguments.qualification_root,
                "gpu_ids": arguments.gpu_ids,
                "cpu_affinity": arguments.cpu_affinity,
                "master_port": arguments.master_port,
                "updates": arguments.updates,
                "warmup_updates": arguments.warmup_updates,
            }
            if any(item is None for item in required.values()):
                raise QualificationWorkerError("--nccl-train inputs are incomplete")
            value = run_nccl_train(**required)
            if value is None:
                return 0
    except (OSError, QualificationWorkerError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_canonical(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
