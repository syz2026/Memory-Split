#!/usr/bin/env python3
"""Execute shell-free AWS GPU qualification phases.

Each phase is independently runnable, writes one closed canonical JSON receipt,
and commits to the exact stdout/stderr bytes produced by any child commands.
Torch is imported lazily so contract tests need neither torch nor a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


PHASE_RECEIPT_TYPE = "memorysplit-aws-gpu-canary-phase-v3"
PHASES = (
    "device-topology-software",
    "bf16",
    "sdpa",
    "torch-compile",
    "fused-adamw",
    "simultaneous-4-plus-4-nccl",
    "one-step-training",
    "checkpoint",
    "resume",
    "nvme",
    "throughput",
)
_SHA256_RE = r"^[0-9a-f]{64}$"
_PHASE_RECEIPT_FIELDS = {
    "schema_version",
    "receipt_type",
    "phase",
    "provider",
    "instance_type",
    "profile_sha256",
    "release_sha256",
    "container_image",
    "container_digest",
    "gres",
    "gpu_ids",
    "arm",
    "passed",
    "evidence",
    "raw_outputs",
    "raw_output_sha256",
}
_RAW_OUTPUT_FIELDS = {
    "argv",
    "returncode",
    "stdout_sha256",
    "stderr_sha256",
}
_EVIDENCE_FIELDS = {
    "device-topology-software": {
        "cuda",
        "driver",
        "linux_kernel",
        "efa",
        "ofi_nccl",
        "gpu_names",
        "fabric_manager_active",
        "nvlink_connected",
        "nvidia_smi_sha256",
        "topology_sha256",
        "kernel_sha256",
        "efa_sha256",
        "fabric_manager_sha256",
        "ofi_nccl_sha256",
    },
    "bf16": {"supported", "dtype", "devices_tested"},
    "sdpa": {"forward", "backward", "dtype", "devices_tested"},
    "torch-compile": {"compiled", "backend", "devices_tested"},
    "fused-adamw": {"fused", "updated", "devices_tested"},
    "simultaneous-4-plus-4-nccl": {
        "concurrent",
        "backend",
        "groups",
        "world_sizes",
    },
    "one-step-training": {
        "arm",
        "updates",
        "finite_loss",
        "world_size",
        "worker_stdout_sha256",
    },
    "checkpoint": {"arm", "checkpointed", "checkpoint_sha256"},
    "resume": {"arm", "resumed", "checkpoint_sha256", "step"},
    "nvme": {"model", "devices", "device_bytes", "raid_level"},
    "throughput": {
        "arm",
        "updates",
        "warmup_updates",
        "tokens_per_update",
        "update_seconds",
    },
}
_ECR_IMAGE_RE = re.compile(
    r"^[0-9]{12}\.dkr\.ecr\.us-(?:east-1|west-2)\.amazonaws\.com/"
    r"[a-z0-9]+(?:[._/-][a-z0-9]+)*@sha256:[0-9a-f]{64}$"
)


class CanaryRuntimeError(ValueError):
    """A canary phase failed or produced unbound evidence."""


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lower_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CanaryRuntimeError(f"{label} must be lowercase SHA-256")
    return value


def _container_digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
    ):
        raise CanaryRuntimeError("container digest must be sha256-pinned")
    _lower_sha256(value.removeprefix("sha256:"), label="container digest")
    return value


def _command(
    argv: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    timeout: float = 60.0,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> tuple[dict[str, object], bytes, bytes]:
    exact = tuple(argv)
    if not exact or any(not isinstance(item, str) or "\x00" in item for item in exact):
        raise CanaryRuntimeError("child argv must contain exact nonempty strings")
    try:
        completed = runner(
            list(exact),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            env=None if environment is None else dict(environment),
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CanaryRuntimeError(f"child command failed: {exact[0]}") from error
    stdout = completed.stdout
    stderr = completed.stderr
    if isinstance(stdout, str):
        stdout = stdout.encode("utf-8")
    if isinstance(stderr, str):
        stderr = stderr.encode("utf-8")
    stdout = bytes(stdout or b"")
    stderr = bytes(stderr or b"")
    record = {
        "argv": list(exact),
        "returncode": int(completed.returncode),
        "stdout_sha256": _sha256(stdout),
        "stderr_sha256": _sha256(stderr),
    }
    if completed.returncode != 0:
        raise CanaryRuntimeError(f"child command returned nonzero: {exact[0]}")
    return record, stdout, stderr


def _torch(module: Any | None = None) -> Any:
    return module if module is not None else importlib.import_module("torch")


def _sync(torch_module: Any) -> None:
    synchronize = getattr(getattr(torch_module, "cuda", None), "synchronize", None)
    if callable(synchronize):
        synchronize()


def _eight_gpu_ids(torch_module: Any) -> range:
    count = int(torch_module.cuda.device_count())
    if count != 8:
        raise CanaryRuntimeError("canary phase requires exactly eight visible GPUs")
    return range(count)


def _reported_version(payload: bytes, *, label: str) -> str:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CanaryRuntimeError(f"{label} output is not UTF-8") from error
    match = re.search(r"(?<![0-9])([0-9]+(?:\.[0-9]+)+)", text)
    if match is None:
        raise CanaryRuntimeError(f"{label} did not report a version")
    return match.group(1)


def phase_device_topology_software(
    *,
    torch_module: Any | None = None,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    torch_value = _torch(torch_module)
    commands = (
        (
            "/usr/bin/nvidia-smi",
            (
                "--query-gpu=index,name,driver_version,"
                "gpu_fabric_info.state,gpu_fabric_info.status"
            ),
            "--format=csv,noheader",
        ),
        ("/usr/bin/nvidia-smi", "topo", "-m"),
        ("/usr/bin/uname", "-r"),
        ("/opt/amazon/efa/bin/fi_info", "--version"),
        (
            "/usr/bin/rpm",
            "--query",
            "--queryformat",
            "%{VERSION}\n",
            "aws-ofi-nccl",
        ),
    )
    raw: list[dict[str, object]] = []
    outputs: list[bytes] = []
    for argv in commands:
        record, stdout, _ = _command(argv, runner=runner)
        raw.append(record)
        outputs.append(stdout)
    cuda = getattr(getattr(torch_value, "version", None), "cuda", None)
    if not isinstance(cuda, str) or not cuda:
        raise CanaryRuntimeError("torch did not report a CUDA runtime")
    try:
        rows = [
            [column.strip() for column in line.split(",")]
            for line in outputs[0].decode("utf-8").splitlines()
            if line.strip()
        ]
    except UnicodeDecodeError as error:
        raise CanaryRuntimeError("nvidia-smi output is not UTF-8") from error
    if (
        len(rows) != 8
        or any(len(row) != 5 for row in rows)
        or [row[0] for row in rows] != [str(index) for index in range(8)]
        or len({row[2] for row in rows}) != 1
    ):
        raise CanaryRuntimeError("device phase requires exactly eight GPUs")
    if any(
        row[3].lower() != "completed" or row[4].lower() != "success"
        for row in rows
    ):
        raise CanaryRuntimeError(
            "NVIDIA Fabric Manager did not complete GPU fabric registration"
        )
    names = [row[1] for row in rows]
    torch_names = [
        str(torch_value.cuda.get_device_name(index))
        for index in range(int(torch_value.cuda.device_count()))
    ]
    if torch_names != names:
        raise CanaryRuntimeError("torch and nvidia-smi GPU identities differ")
    topology = outputs[1].decode("utf-8", errors="strict")
    if "NV" not in topology:
        raise CanaryRuntimeError("topology did not report NVLink connectivity")
    return (
        {
            "cuda": cuda,
            "driver": rows[0][2],
            "linux_kernel": _reported_version(outputs[2], label="kernel"),
            "efa": _reported_version(outputs[3], label="EFA"),
            "ofi_nccl": _reported_version(outputs[4], label="OFI-NCCL"),
            "gpu_names": names,
            "fabric_manager_active": True,
            "nvlink_connected": True,
            "nvidia_smi_sha256": _sha256(outputs[0]),
            "topology_sha256": _sha256(outputs[1]),
            "kernel_sha256": _sha256(outputs[2]),
            "efa_sha256": _sha256(outputs[3]),
            "fabric_manager_sha256": _sha256(outputs[0]),
            "ofi_nccl_sha256": _sha256(outputs[4]),
        },
        raw,
    )


def phase_bf16(
    *,
    torch_module: Any | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    torch_value = _torch(torch_module)
    if torch_value.cuda.is_bf16_supported() is not True:
        raise CanaryRuntimeError("BF16 is not supported")
    outputs = []
    for index in _eight_gpu_ids(torch_value):
        device = f"cuda:{index}"
        left = torch_value.ones(
            (32, 32),
            device=device,
            dtype=torch_value.bfloat16,
        )
        right = torch_value.ones(
            (32, 32),
            device=device,
            dtype=torch_value.bfloat16,
        )
        outputs.append(left @ right)
    _sync(torch_value)
    if any(output.dtype != torch_value.bfloat16 for output in outputs):
        raise CanaryRuntimeError("BF16 operation changed dtype")
    return {"supported": True, "dtype": "bfloat16", "devices_tested": 8}, []


def phase_sdpa(
    *,
    torch_module: Any | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    torch_value = _torch(torch_module)
    queries = []
    for index in _eight_gpu_ids(torch_value):
        query = torch_value.randn(
            (2, 4, 32, 64),
            device=f"cuda:{index}",
            dtype=torch_value.bfloat16,
            requires_grad=True,
        )
        output = torch_value.nn.functional.scaled_dot_product_attention(
            query,
            query,
            query,
            is_causal=True,
        )
        output.float().sum().backward()
        queries.append(query)
    _sync(torch_value)
    if any(query.grad is None for query in queries):
        raise CanaryRuntimeError("SDPA backward did not produce a gradient")
    return {
        "forward": True,
        "backward": True,
        "dtype": "bfloat16",
        "devices_tested": 8,
    }, []


def phase_torch_compile(
    *,
    torch_module: Any | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    torch_value = _torch(torch_module)

    def operation(value: Any) -> Any:
        return (value.sin() + value.cos()).sum()

    compiled = torch_value.compile(operation, backend="inductor", fullgraph=True)
    results = [
        compiled(torch_value.randn((1024,), device=f"cuda:{index}"))
        for index in _eight_gpu_ids(torch_value)
    ]
    _sync(torch_value)
    if any(not bool(torch_value.isfinite(result).item()) for result in results):
        raise CanaryRuntimeError("torch.compile returned a non-finite result")
    return {"compiled": True, "backend": "inductor", "devices_tested": 8}, []


def phase_fused_adamw(
    *,
    torch_module: Any | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    torch_value = _torch(torch_module)
    for index in _eight_gpu_ids(torch_value):
        parameter = torch_value.nn.Parameter(
            torch_value.ones((128,), device=f"cuda:{index}")
        )
        optimizer = torch_value.optim.AdamW([parameter], lr=1e-3, fused=True)
        (parameter.square().mean()).backward()
        optimizer.step()
    _sync(torch_value)
    return {"fused": True, "updated": True, "devices_tested": 8}, []


def _nccl_worker_argv(group: str, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc_per_node=4",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint=127.0.0.1:{port}",
        str(Path(__file__).resolve()),
        "nccl-worker",
        "--group",
        group,
    ]


def phase_simultaneous_nccl(
    *,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    processes: list[tuple[list[str], subprocess.Popen[bytes]]] = []
    for group, devices, port in (
        ("dense", "0,1,2,3", 29620),
        ("split90", "4,5,6,7", 29621),
    ):
        # Preserve the digest-pinned DLC's CUDA, EFA, libfabric, and dynamic
        # loader settings. Replacing the environment here makes the NCCL
        # subprocess unable to discover the image's communication plugins.
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": devices,
                "HOME": "/tmp",
                "NCCL_ASYNC_ERROR_HANDLING": "1",
                "PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
                "PYTHONNOUSERSITE": "1",
            }
        )
        argv = _nccl_worker_argv(group, port)
        try:
            process = popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                env=environment,
            )
        except OSError as error:
            raise CanaryRuntimeError("could not start NCCL worker") from error
        processes.append((argv, process))
    raw: list[dict[str, object]] = []
    for argv, process in processes:
        try:
            stdout, stderr = process.communicate(timeout=180.0)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.communicate()
            raise CanaryRuntimeError("NCCL group timed out") from error
        stdout = bytes(stdout or b"")
        stderr = bytes(stderr or b"")
        raw.append(
            {
                "argv": argv,
                "returncode": int(process.returncode),
                "stdout_sha256": _sha256(stdout),
                "stderr_sha256": _sha256(stderr),
            }
        )
        if process.returncode != 0:
            raise CanaryRuntimeError("NCCL group failed")
    return {
        "concurrent": True,
        "backend": "nccl",
        "groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "world_sizes": [4, 4],
    }, raw


def nccl_worker(*, torch_module: Any | None = None) -> None:
    torch_value = _torch(torch_module)
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except (KeyError, ValueError) as error:
        raise CanaryRuntimeError("NCCL worker local rank is unavailable") from error
    if local_rank not in range(4):
        raise CanaryRuntimeError("NCCL worker local rank is outside 0..3")
    torch_value.cuda.set_device(local_rank)
    distributed = torch_value.distributed
    distributed.init_process_group("nccl")
    try:
        value = torch_value.ones((4096,), device=f"cuda:{local_rank}")
        distributed.all_reduce(value)
        _sync(torch_value)
        if not bool(torch_value.isfinite(value).all().item()):
            raise CanaryRuntimeError("NCCL all-reduce produced non-finite output")
    finally:
        distributed.destroy_process_group()


def _one_step(torch_value: Any, *, device: str = "cuda") -> float:
    model = torch_value.nn.Linear(
        128,
        128,
        device=device,
        dtype=torch_value.bfloat16,
    )
    optimizer = torch_value.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    sample = torch_value.randn(
        (32, 128),
        device=device,
        dtype=torch_value.bfloat16,
    )
    target = torch_value.randn(
        (32, 128),
        device=device,
        dtype=torch_value.bfloat16,
    )
    loss = (model(sample).float() - target.float()).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _sync(torch_value)
    measured = float(loss.detach().cpu().item())
    if not math.isfinite(measured):
        raise CanaryRuntimeError("one-step loss is not finite")
    return measured


def phase_one_step_training(
    *,
    arm: str | None,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if arm not in {"dense", "split90"}:
        raise CanaryRuntimeError("one-step training requires one closed arm")
    argv = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc_per_node=4",
        "--standalone",
        str(Path(__file__).resolve()),
        "training-worker",
        "--arm",
        arm,
    ]
    record, stdout, _stderr = _command(
        argv,
        timeout=180.0,
        runner=runner,
    )
    return {
        "arm": arm,
        "updates": 1,
        "finite_loss": True,
        "world_size": 4,
        "worker_stdout_sha256": _sha256(stdout),
    }, [record]


def training_worker(
    *,
    torch_module: Any | None = None,
) -> None:
    torch_value = _torch(torch_module)
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except (KeyError, ValueError) as error:
        raise CanaryRuntimeError("training worker local rank is unavailable") from error
    if local_rank not in range(4):
        raise CanaryRuntimeError("training worker local rank is outside 0..3")
    torch_value.cuda.set_device(local_rank)
    distributed = torch_value.distributed
    distributed.init_process_group("nccl")
    try:
        loss = _one_step(torch_value, device=f"cuda:{local_rank}")
        result = torch_value.tensor(loss, device=f"cuda:{local_rank}")
        distributed.all_reduce(result)
        if not bool(torch_value.isfinite(result).item()):
            raise CanaryRuntimeError("distributed one-step loss is not finite")
    finally:
        distributed.destroy_process_group()


def phase_checkpoint(
    *,
    checkpoint: Path,
    torch_module: Any | None = None,
    arm: str | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if arm not in {"dense", "split90"}:
        raise CanaryRuntimeError("checkpoint phase requires one closed arm")
    torch_value = _torch(torch_module)
    checkpoint.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if checkpoint.exists() or checkpoint.is_symlink():
        raise CanaryRuntimeError("checkpoint destination already exists")
    state = {
        "arm": arm,
        "step": 1,
        "tensor": torch_value.arange(16, device="cpu"),
    }
    torch_value.save(state, checkpoint)
    digest = _sha256(checkpoint.read_bytes())
    return {
        "arm": arm,
        "checkpointed": True,
        "checkpoint_sha256": digest,
    }, []


def phase_resume(
    *,
    checkpoint: Path,
    torch_module: Any | None = None,
    arm: str | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if arm not in {"dense", "split90"}:
        raise CanaryRuntimeError("resume phase requires one closed arm")
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise CanaryRuntimeError("resume checkpoint must be a regular file")
    digest = _sha256(checkpoint.read_bytes())
    state = _torch(torch_module).load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if (
        not isinstance(state, dict)
        or state.get("arm") != arm
        or state.get("step") != 1
    ):
        raise CanaryRuntimeError("resume checkpoint identity is invalid")
    return {
        "arm": arm,
        "resumed": True,
        "checkpoint_sha256": digest,
        "step": 1,
    }, []


def phase_nvme(
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    expected_model: str,
    expected_devices: int,
    expected_device_bytes: int,
    expected_raid_level: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    argv = (
        "/usr/bin/lsblk",
        "--json",
        "--bytes",
        "--output",
        "NAME,PATH,TYPE,MODEL,SIZE,MOUNTPOINTS",
    )
    record, stdout, _ = _command(argv, runner=runner)
    try:
        value = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanaryRuntimeError("lsblk did not return valid JSON") from error
    rows = value.get("blockdevices") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise CanaryRuntimeError("lsblk did not return blockdevices")
    devices = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("type") == "disk"
        and str(row.get("model", "")).strip()
        == expected_model
    ]
    sizes = [int(row.get("size", 0)) for row in devices]
    if (
        expected_model != "Amazon EC2 NVMe Instance Storage"
        or type(expected_devices) is not int
        or expected_devices <= 0
        or type(expected_device_bytes) is not int
        or expected_device_bytes <= 0
        or expected_raid_level != "0"
        or len(devices) != expected_devices
        or any(size != expected_device_bytes for size in sizes)
    ):
        raise CanaryRuntimeError("NVMe geometry does not match the selected profile")
    return {
        "model": expected_model,
        "devices": len(devices),
        "device_bytes": min(sizes),
        "raid_level": expected_raid_level,
    }, [record]


def phase_throughput(
    *,
    torch_module: Any | None = None,
    arm: str | None,
    updates: int,
    warmup_updates: int,
    tokens_per_update: int,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if (
        arm not in {"dense", "split90"}
        or updates != 100
        or warmup_updates != 10
        or tokens_per_update != 524_288
    ):
        raise CanaryRuntimeError("throughput geometry is not the reviewed canary")
    torch_value = _torch(torch_module)
    durations: list[float] = []
    for _ in range(updates):
        started = clock()
        _one_step(torch_value)
        duration = float(clock() - started)
        if not math.isfinite(duration) or duration <= 0:
            raise CanaryRuntimeError("throughput duration is not positive and finite")
        durations.append(duration)
    return {
        "arm": arm,
        "updates": updates,
        "warmup_updates": warmup_updates,
        "tokens_per_update": tokens_per_update,
        "update_seconds": durations,
    }, []


def build_phase_receipt(
    *,
    phase: str,
    provider: str,
    instance_type: str,
    profile_sha256: str,
    release_sha256: str,
    container_image: str,
    container_digest: str,
    gres: str,
    gpu_ids: Sequence[int],
    arm: str | None,
    evidence: Mapping[str, object],
    raw_outputs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if phase not in PHASES:
        raise CanaryRuntimeError("unknown canary phase")
    _lower_sha256(profile_sha256, label="profile SHA-256")
    _lower_sha256(release_sha256, label="release SHA-256")
    digest = _container_digest(container_digest)
    if (
        not isinstance(container_image, str)
        or not container_image.endswith("@" + digest)
        or any(character.isspace() for character in container_image)
    ):
        raise CanaryRuntimeError("container image does not bind its digest")
    raw = [dict(row) for row in raw_outputs]
    if any(set(row) != _RAW_OUTPUT_FIELDS for row in raw):
        raise CanaryRuntimeError("raw output records do not match the closed schema")
    raw_digest = _sha256(
        _canonical({"evidence": dict(evidence), "raw_outputs": raw})
    )
    receipt = {
        "schema_version": 3,
        "receipt_type": PHASE_RECEIPT_TYPE,
        "phase": phase,
        "provider": provider,
        "instance_type": instance_type,
        "profile_sha256": profile_sha256,
        "release_sha256": release_sha256,
        "container_image": container_image,
        "container_digest": digest,
        "gres": gres,
        "gpu_ids": list(gpu_ids),
        "arm": arm,
        "passed": True,
        "evidence": dict(evidence),
        "raw_outputs": raw,
        "raw_output_sha256": raw_digest,
    }
    validate_phase_receipt(receipt)
    return receipt


def _validate_evidence(
    phase: str,
    arm: object,
    evidence: Mapping[str, object],
) -> None:
    expected_fields = _EVIDENCE_FIELDS[phase]
    if set(evidence) != expected_fields:
        raise CanaryRuntimeError("phase evidence fields do not match")
    if phase == "device-topology-software":
        gpu_names = evidence["gpu_names"]
        if (
            any(
                not isinstance(evidence[field], str) or not evidence[field]
                for field in ("cuda", "driver", "linux_kernel", "efa", "ofi_nccl")
            )
            or not isinstance(gpu_names, list)
            or len(gpu_names) != 8
            or any(not isinstance(name, str) or not name for name in gpu_names)
            or evidence["fabric_manager_active"] is not True
            or evidence["nvlink_connected"] is not True
        ):
            raise CanaryRuntimeError("device phase evidence is incomplete")
        for field in expected_fields:
            if field.endswith("_sha256"):
                _lower_sha256(evidence[field], label=f"device evidence {field}")
        return
    exact = {
        "bf16": {
            "supported": True,
            "dtype": "bfloat16",
            "devices_tested": 8,
        },
        "sdpa": {
            "forward": True,
            "backward": True,
            "dtype": "bfloat16",
            "devices_tested": 8,
        },
        "torch-compile": {
            "compiled": True,
            "backend": "inductor",
            "devices_tested": 8,
        },
        "fused-adamw": {
            "fused": True,
            "updated": True,
            "devices_tested": 8,
        },
        "simultaneous-4-plus-4-nccl": {
            "concurrent": True,
            "backend": "nccl",
            "groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "world_sizes": [4, 4],
        },
    }
    if phase in exact:
        if dict(evidence) != exact[phase]:
            raise CanaryRuntimeError(f"{phase} evidence is not exact")
        return
    if phase == "one-step-training":
        if (
            arm not in {"dense", "split90"}
            or evidence["arm"] != arm
            or evidence["updates"] != 1
            or evidence["finite_loss"] is not True
            or evidence["world_size"] != 4
        ):
            raise CanaryRuntimeError("one-step training evidence is incomplete")
        _lower_sha256(
            evidence["worker_stdout_sha256"],
            label="one-step worker output",
        )
        return
    if phase == "checkpoint":
        if (
            arm not in {"dense", "split90"}
            or evidence["arm"] != arm
            or evidence["checkpointed"] is not True
        ):
            raise CanaryRuntimeError("checkpoint evidence is incomplete")
        _lower_sha256(evidence["checkpoint_sha256"], label="checkpoint evidence")
        return
    if phase == "resume":
        if (
            arm not in {"dense", "split90"}
            or evidence["arm"] != arm
            or evidence["resumed"] is not True
            or evidence["step"] != 1
        ):
            raise CanaryRuntimeError("resume evidence is incomplete")
        _lower_sha256(evidence["checkpoint_sha256"], label="resume checkpoint")
        return
    if phase == "nvme":
        if (
            not isinstance(evidence["model"], str)
            or not evidence["model"]
            or type(evidence["devices"]) is not int
            or evidence["devices"] <= 0
            or type(evidence["device_bytes"]) is not int
            or evidence["device_bytes"] <= 0
            or evidence["raid_level"] != "0"
        ):
            raise CanaryRuntimeError("NVMe evidence is incomplete")
        return
    durations = evidence["update_seconds"]
    if (
        arm not in {"dense", "split90"}
        or evidence["arm"] != arm
        or evidence["updates"] != 100
        or evidence["warmup_updates"] != 10
        or evidence["tokens_per_update"] != 524_288
        or not isinstance(durations, list)
        or len(durations) != 100
        or any(
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
            for duration in durations
        )
    ):
        raise CanaryRuntimeError("throughput evidence is incomplete")


def validate_phase_receipt(receipt: object) -> dict[str, object]:
    """Validate one decoded phase receipt without trusting its producer."""

    if not isinstance(receipt, dict) or set(receipt) != _PHASE_RECEIPT_FIELDS:
        raise CanaryRuntimeError("phase receipt fields do not match")
    if (
        receipt["schema_version"] != 3
        or receipt["receipt_type"] != PHASE_RECEIPT_TYPE
        or receipt["phase"] not in PHASES
        or receipt["passed"] is not True
        or not isinstance(receipt["evidence"], dict)
        or not isinstance(receipt["raw_outputs"], list)
        or not isinstance(receipt["provider"], str)
        or not receipt["provider"]
        or not isinstance(receipt["instance_type"], str)
        or not receipt["instance_type"]
        or not isinstance(receipt["gres"], str)
        or not receipt["gres"].endswith(":8")
        or not isinstance(receipt["gpu_ids"], list)
        or not receipt["gpu_ids"]
        or len(receipt["gpu_ids"]) != len(set(receipt["gpu_ids"]))
        or any(type(gpu_id) is not int or gpu_id not in range(8) for gpu_id in receipt["gpu_ids"])
        or receipt["arm"] not in {None, "dense", "split90"}
    ):
        raise CanaryRuntimeError("phase receipt identity is invalid")
    _lower_sha256(receipt["profile_sha256"], label="profile SHA-256")
    _lower_sha256(receipt["release_sha256"], label="release SHA-256")
    digest = _container_digest(receipt["container_digest"])
    if (
        not isinstance(receipt["container_image"], str)
        or _ECR_IMAGE_RE.fullmatch(receipt["container_image"]) is None
        or not receipt["container_image"].endswith("@" + digest)
        or any(
            not isinstance(row, dict) or set(row) != _RAW_OUTPUT_FIELDS
            for row in receipt["raw_outputs"]
        )
    ):
        raise CanaryRuntimeError("phase receipt runtime binding is invalid")
    for row in receipt["raw_outputs"]:
        argv = row["argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or any(
                not isinstance(argument, str)
                or not argument
                or "\x00" in argument
                for argument in argv
            )
            or row["returncode"] != 0
        ):
            raise CanaryRuntimeError("phase raw command evidence is invalid")
        _lower_sha256(row["stdout_sha256"], label="phase stdout")
        _lower_sha256(row["stderr_sha256"], label="phase stderr")
    _validate_evidence(
        str(receipt["phase"]),
        receipt["arm"],
        receipt["evidence"],
    )
    expected_raw = _sha256(
        _canonical(
            {
                "evidence": receipt["evidence"],
                "raw_outputs": receipt["raw_outputs"],
            }
        )
    )
    if receipt["raw_output_sha256"] != expected_raw:
        raise CanaryRuntimeError("phase raw-output hash does not match")
    return dict(receipt)


def write_receipt(path: Path, receipt: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        payload = _canonical(dict(receipt))
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _gpu_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU IDs must be comma-separated integers") from error
    if (
        not result
        or len(set(result)) != len(result)
        or any(item < 0 or item > 7 for item in result)
    ):
        raise argparse.ArgumentTypeError("GPU IDs must be unique members of 0..7")
    return result


def _verify_release_runtime_root(root: Path, release_sha256: str) -> None:
    _lower_sha256(release_sha256, label="release SHA-256")
    try:
        resolved = root.resolve(strict=True)
        script = Path(__file__).resolve(strict=True)
        relative = script.relative_to(resolved).as_posix()
    except (OSError, ValueError) as error:
        raise CanaryRuntimeError(
            "canary runtime is not executing from the mounted release"
        ) from error
    if relative != "cluster/aws/p5/canary_runtime.py":
        raise CanaryRuntimeError("canary runtime release path is not canonical")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=(*PHASES, "nccl-worker", "training-worker"),
    )
    parser.add_argument("--provider")
    parser.add_argument("--instance-type")
    parser.add_argument("--profile-sha256")
    parser.add_argument("--release-sha256")
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--container-image")
    parser.add_argument("--container-digest")
    parser.add_argument("--gres")
    parser.add_argument("--gpu-ids", type=_gpu_ids)
    parser.add_argument("--arm", choices=("dense", "split90"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--warmup-updates", type=int, default=10)
    parser.add_argument("--tokens-per-update", type=int, default=524_288)
    parser.add_argument("--group", choices=("dense", "split90"))
    parser.add_argument("--expected-nvme-model")
    parser.add_argument("--expected-nvme-devices", type=int)
    parser.add_argument("--expected-nvme-device-bytes", type=int)
    parser.add_argument("--expected-raid-level")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        if arguments.phase == "nccl-worker":
            nccl_worker()
            return 0
        if arguments.phase == "training-worker":
            if arguments.arm is None:
                raise CanaryRuntimeError("training worker arm is required")
            training_worker()
            return 0
        required = (
            arguments.provider,
            arguments.instance_type,
            arguments.profile_sha256,
            arguments.release_sha256,
            arguments.release_root,
            arguments.container_image,
            arguments.container_digest,
            arguments.gres,
            arguments.gpu_ids,
            arguments.output,
        )
        if any(value is None for value in required):
            raise CanaryRuntimeError("phase identity and output arguments are required")
        _verify_release_runtime_root(
            arguments.release_root,
            arguments.release_sha256,
        )
        phase = arguments.phase
        if phase == "device-topology-software":
            evidence, raw = phase_device_topology_software()
        elif phase == "bf16":
            evidence, raw = phase_bf16()
        elif phase == "sdpa":
            evidence, raw = phase_sdpa()
        elif phase == "torch-compile":
            evidence, raw = phase_torch_compile()
        elif phase == "fused-adamw":
            evidence, raw = phase_fused_adamw()
        elif phase == "simultaneous-4-plus-4-nccl":
            evidence, raw = phase_simultaneous_nccl()
        elif phase == "one-step-training":
            evidence, raw = phase_one_step_training(arm=arguments.arm)
        elif phase == "checkpoint":
            if arguments.checkpoint is None:
                raise CanaryRuntimeError("checkpoint path is required")
            evidence, raw = phase_checkpoint(
                checkpoint=arguments.checkpoint,
                arm=arguments.arm,
            )
        elif phase == "resume":
            if arguments.checkpoint is None:
                raise CanaryRuntimeError("checkpoint path is required")
            evidence, raw = phase_resume(
                checkpoint=arguments.checkpoint,
                arm=arguments.arm,
            )
        elif phase == "nvme":
            nvme_required = (
                arguments.expected_nvme_model,
                arguments.expected_nvme_devices,
                arguments.expected_nvme_device_bytes,
                arguments.expected_raid_level,
            )
            if any(value is None for value in nvme_required):
                raise CanaryRuntimeError("expected NVMe geometry is required")
            evidence, raw = phase_nvme(
                expected_model=arguments.expected_nvme_model,
                expected_devices=arguments.expected_nvme_devices,
                expected_device_bytes=arguments.expected_nvme_device_bytes,
                expected_raid_level=arguments.expected_raid_level,
            )
        else:
            evidence, raw = phase_throughput(
                arm=arguments.arm,
                updates=arguments.updates,
                warmup_updates=arguments.warmup_updates,
                tokens_per_update=arguments.tokens_per_update,
            )
        receipt = build_phase_receipt(
            phase=phase,
            provider=arguments.provider,
            instance_type=arguments.instance_type,
            profile_sha256=arguments.profile_sha256,
            release_sha256=arguments.release_sha256,
            container_image=arguments.container_image,
            container_digest=arguments.container_digest,
            gres=arguments.gres,
            gpu_ids=arguments.gpu_ids,
            arm=arguments.arm,
            evidence=evidence,
            raw_outputs=raw,
        )
        validate_phase_receipt(receipt)
        write_receipt(arguments.output, receipt)
        print(_canonical(receipt).decode("ascii"), end="")
        return 0
    except (CanaryRuntimeError, OSError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "schema_version": 3,
                    "ok": False,
                    "error": type(error).__name__,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
