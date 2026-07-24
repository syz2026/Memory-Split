#!/opt/conda/bin/python
"""Measured host/container workers for selected AWS GPU qualification.

The controller invokes this script only through closed argv rendered by
``cluster.aws.qualification``.  Unit tests inject command results; this module
contains the live probe implementation but does not execute at import time.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


class QualificationWorkerError(ValueError):
    """A measured worker fact or capability failed closed."""


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
    torch.manual_seed(1731)
    model = torch.nn.Linear(32, 32, bias=True, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    checkpoint = io.BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
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
    inputs = torch.arange(1024, device=device, dtype=torch.float32).reshape(32, 32)

    def step(candidate, selected_optimizer):
        selected_optimizer.zero_grad(set_to_none=True)
        loss = candidate(inputs).square().mean()
        loss.backward()
        selected_optimizer.step()
        return loss.detach()

    first_loss = step(model, optimizer)
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


def run_nccl_train(
    *,
    arm: str,
    gpu_ids: str,
    cpu_affinity: str,
    master_port: int,
    updates: int,
    warmup_updates: int,
) -> dict[str, object] | None:
    """Run one independent four-rank NCCL/training group."""

    import torch

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

    torch.cuda.reset_peak_memory_stats(device)
    parameter = torch.nn.Parameter(
        torch.randn((512, 512), dtype=torch.bfloat16, device=device)
    )
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, fused=True)
    samples: list[float] = []
    tokens = 16_384
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        batch_started = time.perf_counter()
        loss = (parameter @ parameter).float().square().mean()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - batch_started
        if elapsed <= 0 or not math.isfinite(elapsed):
            raise QualificationWorkerError("training timer is invalid")
        samples.append(tokens / elapsed)
    checkpoint_resume_exact, checkpoint_sha256 = _one_step_resume(torch, device)
    peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    retained = samples[warmup_updates:]
    if not retained or peak_memory_bytes <= 0 or not checkpoint_resume_exact:
        raise QualificationWorkerError("training/checkpoint evidence is invalid")
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
        "median_tok_s": float(statistics.median(retained)),
        "peak_memory_bytes": peak_memory_bytes,
        "one_step_train": True,
        "checkpoint_resume_exact": checkpoint_resume_exact,
        "checkpoint_sha256": checkpoint_sha256,
        "resumed_checkpoint_sha256": checkpoint_sha256,
    }
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return result if rank == 0 else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--hardware", action="store_true")
    mode.add_argument("--container-capabilities", action="store_true")
    mode.add_argument("--nccl-train", action="store_true")
    parser.add_argument("--profile-id")
    parser.add_argument("--region")
    parser.add_argument("--arm")
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
