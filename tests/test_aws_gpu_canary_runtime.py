from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cluster.aws.p5 import canary_runtime as runtime


class _Tensor:
    def __init__(self, *, dtype="bfloat16", root=None, value=1.0):
        self.dtype = dtype
        self.root = root
        self.value = value
        self.grad = None

    def _derived(self):
        return _Tensor(dtype=self.dtype, root=self.root or self, value=self.value)

    def __matmul__(self, other):
        del other
        return self._derived()

    def __add__(self, other):
        del other
        return self._derived()

    def __sub__(self, other):
        del other
        return self._derived()

    def sin(self):
        return self._derived()

    def cos(self):
        return self._derived()

    def square(self):
        return self._derived()

    def mean(self):
        return self._derived()

    def sum(self):
        return self._derived()

    def float(self):
        return self._derived()

    def backward(self):
        target = self.root or self
        target.grad = _Tensor()

    def detach(self):
        return self

    def cpu(self):
        return self

    def all(self):
        return self

    def item(self):
        return self.value


class _Cuda:
    def is_bf16_supported(self):
        return True

    def device_count(self):
        return 8

    def get_device_name(self, index):
        return "NVIDIA H100 80GB HBM3"

    def synchronize(self):
        return None

    def set_device(self, index):
        self.device = index


class _Linear:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.parameter = _Tensor()

    def __call__(self, value):
        return value._derived()

    def parameters(self):
        return [self.parameter]


class _AdamW:
    def __init__(self, parameters, **kwargs):
        self.parameters = list(parameters)
        self.kwargs = kwargs

    def step(self):
        return None

    def zero_grad(self, **kwargs):
        del kwargs


class _Torch:
    bfloat16 = "bfloat16"

    def __init__(self):
        self.cuda = _Cuda()
        self.version = SimpleNamespace(cuda="13.0")
        self.nn = SimpleNamespace(
            Parameter=lambda value: value,
            Linear=_Linear,
            functional=SimpleNamespace(
                scaled_dot_product_attention=lambda query, *_args, **_kwargs: (
                    query._derived()
                )
            ),
        )
        self.optim = SimpleNamespace(AdamW=_AdamW)
        self.saved_state = None

    def ones(self, *args, **kwargs):
        del args
        return _Tensor(dtype=kwargs.get("dtype", self.bfloat16))

    def randn(self, *args, **kwargs):
        del args
        return _Tensor(dtype=kwargs.get("dtype", self.bfloat16))

    def arange(self, *args, **kwargs):
        del args, kwargs
        return _Tensor(dtype="int64")

    def tensor(self, value, **kwargs):
        del kwargs
        return _Tensor(value=value)

    def compile(self, operation, **kwargs):
        assert kwargs == {"backend": "inductor", "fullgraph": True}
        return operation

    def isfinite(self, value):
        del value
        return _Tensor(value=True)

    def save(self, state, path):
        self.saved_state = state
        path.write_bytes(b"closed-checkpoint")

    def load(self, path, **kwargs):
        assert path.read_bytes() == b"closed-checkpoint"
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return self.saved_state


def _completed(argv, stdout=b"", stderr=b""):
    return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr=stderr)


def test_device_phase_uses_dlc_argv_and_hashes_raw_outputs():
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        outputs = {
            (
                "/usr/bin/nvidia-smi",
                "--query-gpu=index,name,driver_version,"
                "gpu_fabric_info.state,gpu_fabric_info.status",
            ): (
                "\n".join(
                    f"{index}, NVIDIA H100 80GB HBM3, 580.42, Completed, Success"
                    for index in range(8)
                )
                + "\n"
            ).encode(),
            ("/usr/bin/nvidia-smi", "topo", "-m"): b"GPU0 NV18 GPU1\n",
            ("/usr/bin/uname", "-r"): b"6.8.0-1024-aws\n",
            ("/opt/amazon/efa/bin/fi_info", "--version"): b"libfabric: 1.44.0\n",
            ("/usr/bin/rpm", "--query"): b"1.17.1\n",
        }
        key = (
            tuple(argv[:3])
            if argv[:2] == ["/usr/bin/nvidia-smi", "topo"]
            else tuple(argv[:2])
        )
        return _completed(argv, outputs[key])

    evidence, raw = runtime.phase_device_topology_software(
        torch_module=_Torch(),
        runner=runner,
    )

    assert evidence["cuda"] == "13.0"
    assert evidence["driver"] == "580.42"
    assert evidence["linux_kernel"] == "6.8.0"
    assert evidence["efa"] == "1.44.0"
    assert evidence["ofi_nccl"] == "1.17.1"
    assert evidence["fabric_manager_active"] is True
    assert len(raw) == 5
    assert calls[-1][0] == [
        "/usr/bin/rpm",
        "--query",
        "--queryformat",
        "%{VERSION}\n",
        "aws-ofi-nccl",
    ]
    assert all(call[1]["shell"] is False for call in calls)


def test_torch_phases_cover_all_gpus_checkpoint_resume_and_throughput(tmp_path):
    torch = _Torch()
    assert runtime.phase_bf16(torch_module=torch)[0]["devices_tested"] == 8
    assert runtime.phase_sdpa(torch_module=torch)[0]["devices_tested"] == 8
    assert runtime.phase_torch_compile(torch_module=torch)[0] == {
        "compiled": True,
        "backend": "inductor",
        "devices_tested": 8,
    }
    assert runtime.phase_fused_adamw(torch_module=torch)[0]["updated"] is True

    checkpoint = tmp_path / "checkpoint" / "dense.pt"
    saved, _ = runtime.phase_checkpoint(
        checkpoint=checkpoint,
        torch_module=torch,
        arm="dense",
    )
    resumed, _ = runtime.phase_resume(
        checkpoint=checkpoint,
        torch_module=torch,
        arm="dense",
    )
    assert saved["checkpoint_sha256"] == resumed["checkpoint_sha256"]

    ticks = iter(range(201))
    throughput, _ = runtime.phase_throughput(
        torch_module=torch,
        arm="split90",
        updates=100,
        warmup_updates=10,
        tokens_per_update=524_288,
        clock=lambda: float(next(ticks)),
    )
    assert throughput["update_seconds"] == [1.0] * 100


def test_process_phases_use_exact_argv_without_shell(monkeypatch):
    run_calls = []

    def runner(argv, **kwargs):
        run_calls.append((argv, kwargs))
        return _completed(argv, b'{"finite_loss":true}\n')

    one_step, raw = runtime.phase_one_step_training(
        arm="dense",
        runner=runner,
    )
    assert one_step["world_size"] == 4
    assert raw[0]["argv"][1:3] == ["-m", "torch.distributed.run"]
    assert run_calls[0][1]["shell"] is False

    popen_calls = []
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/aws-ofi-nccl/lib")
    monkeypatch.setenv("FI_PROVIDER_PATH", "/opt/amazon/efa/lib64/libfabric")

    class Process:
        returncode = 0

        def communicate(self, timeout=None):
            assert timeout == 180.0
            return b"ok\n", b""

        def kill(self):
            pytest.fail("successful fake process must not be killed")

    def popen(argv, **kwargs):
        popen_calls.append((argv, kwargs))
        return Process()

    nccl, raw = runtime.phase_simultaneous_nccl(popen=popen)
    assert nccl["groups"] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert len(raw) == 2
    assert [call[1]["env"]["CUDA_VISIBLE_DEVICES"] for call in popen_calls] == [
        "0,1,2,3",
        "4,5,6,7",
    ]
    assert all(
        call[1]["env"]["LD_LIBRARY_PATH"] == "/opt/aws-ofi-nccl/lib"
        and call[1]["env"]["FI_PROVIDER_PATH"]
        == "/opt/amazon/efa/lib64/libfabric"
        for call in popen_calls
    )
    assert all(call[1]["shell"] is False for call in popen_calls)


def test_nvme_and_closed_phase_receipt_are_profile_runtime_bound():
    blockdevices = {
        "blockdevices": [
            {
                "name": f"nvme{index}n1",
                "path": f"/dev/nvme{index}n1",
                "type": "disk",
                "model": "Amazon EC2 NVMe Instance Storage",
                "size": 1_000_000,
                "mountpoints": [None],
            }
            for index in range(8)
        ]
    }

    def runner(argv, **kwargs):
        assert kwargs["shell"] is False
        return _completed(argv, json.dumps(blockdevices).encode())

    evidence, raw = runtime.phase_nvme(
        runner=runner,
        expected_model="Amazon EC2 NVMe Instance Storage",
        expected_devices=8,
        expected_device_bytes=1_000_000,
        expected_raid_level="0",
    )
    digest = "sha256:" + "d" * 64
    receipt = runtime.build_phase_receipt(
        phase="nvme",
        provider="aws-p5.48xlarge-v3",
        instance_type="p5.48xlarge",
        profile_sha256="a" * 64,
        release_sha256="b" * 64,
        container_image=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/"
            f"memorysplit/aws-gpu@{digest}"
        ),
        container_digest=digest,
        gres="gpu:h100:8",
        gpu_ids=range(8),
        arm=None,
        evidence=evidence,
        raw_outputs=raw,
    )
    assert runtime.validate_phase_receipt(receipt) == receipt
    receipt["raw_output_sha256"] = "0" * 64
    with pytest.raises(runtime.CanaryRuntimeError, match="raw-output hash"):
        runtime.validate_phase_receipt(receipt)
