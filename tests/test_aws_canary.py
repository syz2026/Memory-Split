from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import stat
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "cluster" / "aws" / "p5" / "canary.py"
PROFILE = ROOT / "cluster" / "profiles" / "aws-p5.48xlarge-v3.json"
INSTANCE_ID = "i-0123456789abcdef0"
BOOT_ID = "12345678-1234-4abc-8def-1234567890ab"
IMAGE_DIGEST = "sha256:" + "a" * 64
IMAGE = (
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/memorysplit"
    f"@{IMAGE_DIGEST}"
)
VERSIONS = {
    "python": "3.12.4",
    "pytorch": "2.7.1+cu128",
    "cuda": "12.8",
    "cudnn": "9.7.1",
    "nccl": "2.26.2",
    "nvidia_driver": "570.133.20",
    "fabric_manager": "570.133.20",
    "docker": "28.0.4",
    "nvidia_container_runtime": "1.17.8",
    "aws_cli": "2.27.49",
}
RECEIPT_FIELDS = {
    "schema_version",
    "receipt_type",
    "provider",
    "instance_id",
    "boot_id",
    "profile_sha256",
    "runtime_lock_sha256",
    "environment_receipt_sha256",
    "release_sha256",
    "release_receipt_sha256",
    "run_manifest_sha256",
    "dataset_receipt_sha256",
    "dataset_build_id",
    "ordered_stream_sha256",
    "source_commit",
    "source_tree",
    "container_image",
    "container_image_digest",
    "seed",
    "phases",
    "hardware",
    "functional",
    "resume",
    "throughput_4x4",
    "s3_roundtrip",
    "passed",
    "started_at",
    "ended_at",
    "total_seconds",
}


def _load_module():
    assert SCRIPT.is_file(), "AWS P5 qualification canary is missing"
    name = f"aws_canary_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))
    return path


def _runtime_lock(profile_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_commit": "b" * 40,
        "source_tree": "c" * 40,
        "control_bundle_sha256": "d" * 64,
        "profile_sha256": profile_sha256,
        "ami_id": "ami-0123456789abcdef0",
        "ami_owner_id": "210987654321",
        "container_image": IMAGE,
        "container_image_digest": IMAGE_DIGEST,
        "versions": dict(VERSIONS),
    }


def _environment_receipt(
    *,
    profile_sha256: str,
    runtime_lock_sha256: str,
) -> dict[str, object]:
    identity = {
        "accountId": "123456789012",
        "architecture": "x86_64",
        "imageId": "ami-0123456789abcdef0",
        "instanceId": INSTANCE_ID,
        "privateIp": "10.23.45.67",
        "region": "us-east-1",
    }
    return {
        "schema_version": 2,
        "receipt_type": "memorysplit-aws-environment-v2",
        "provider": "aws-p5.48xlarge",
        "profile_sha256": profile_sha256,
        "runtime_lock_sha256": runtime_lock_sha256,
        "control_bundle_sha256": "d" * 64,
        "source_commit": "b" * 40,
        "source_tree": "c" * 40,
        "container_image": IMAGE,
        "container_image_digest": IMAGE_DIGEST,
        "aws_instance_identity_document": identity,
        "aws_instance_identity_pkcs7": "c3ludGhldGljLXNpZ25hdHVyZQ==",
        "account_id": identity["accountId"],
        "instance_id": INSTANCE_ID,
        "region": identity["region"],
        "ami_id": identity["imageId"],
        "boot_id": BOOT_ID,
        "runtime_facts": dict(VERSIONS),
    }


def _make_release_root(root: Path) -> tuple[Path, dict[str, str]]:
    members = [
        "cluster/aws/p5/canary.py",
        "scripts/run_train.py",
        "train/trainer.py",
        "train/data.py",
        "train/model.py",
        "train/safeio.py",
        "msctl/aws_contracts.py",
        "cluster/aws/p5/attest_environment.py",
        "cluster/aws/p5/corpus_contract.py",
        "cluster/aws/p5/profile.py",
        "DATASET-POINTER-AWS.json",
        "configs/cohort-assignment-v3.json",
        "configs/preregistration-v3.yaml",
        "cluster/profiles/aws-p5.48xlarge-v3.json",
        *(
            f"configs/360m-v3/{arm}-s{seed}.yaml"
            for seed in range(10)
            for arm in ("dense", "split90")
        ),
    ]
    for relative in members:
        source = ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    (root / "RELEASE-METADATA.json").write_bytes(
        _canonical({"fixture": "release-metadata"})
    )
    digests = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in root.rglob("*")
        if path.is_file()
    }
    sums = "".join(
        f"{digest}  {relative}\n"
        for relative, digest in sorted(digests.items())
    ).encode("ascii")
    (root / "SHA256SUMS").write_bytes(sums)
    digests["SHA256SUMS"] = hashlib.sha256(sums).hexdigest()
    return root, digests


def _case(tmp_path: Path, module) -> dict[str, object]:
    release_root, member_hashes = _make_release_root(tmp_path / "release-root")
    profile_sha256 = member_hashes[
        "cluster/profiles/aws-p5.48xlarge-v3.json"
    ]
    config_sha256 = {
        f"configs/360m-v3/{arm}-s{seed}.yaml": member_hashes[
            f"configs/360m-v3/{arm}-s{seed}.yaml"
        ]
        for seed in range(10)
        for arm in ("dense", "split90")
    }
    release = {
        "schema_version": 1,
        "package_format_version": 2,
        "release_id": "aws-p5-r1-synthetic",
        "provider": "aws-p5.48xlarge",
        "archive": {
            "path": "ms-aws-p5-r1-synthetic.zip",
            "sha256": "1" * 64,
            "bytes": 12345,
        },
        "source": {
            "commit": "b" * 40,
            "tree": "c" * 40,
            "dirty": False,
        },
        "seed_assignment": {
            "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
            "provider": "aws-p5.48xlarge",
            "seeds": list(range(10)),
            "arms": ["dense", "split90"],
        },
        "cohort_assignment": {
            "path": "configs/cohort-assignment-v3.json",
            "sha256": member_hashes["configs/cohort-assignment-v3.json"],
        },
        "profile": {
            "path": "cluster/profiles/aws-p5.48xlarge-v3.json",
            "sha256": profile_sha256,
        },
        "environment": {
            "mode": "runtime_attested",
            "profile_sha256": profile_sha256,
        },
        "dataset_pointer": {
            "path": "DATASET-POINTER-AWS.json",
            "sha256": member_hashes["DATASET-POINTER-AWS.json"],
        },
        "cohort_assignment_sha256": member_hashes[
            "configs/cohort-assignment-v3.json"
        ],
        "profile_sha256": profile_sha256,
        "dataset_pointer_sha256": member_hashes["DATASET-POINTER-AWS.json"],
        "config_sha256": config_sha256,
        "members_sha256": member_hashes["SHA256SUMS"],
    }
    release_receipt = _write_json(tmp_path / "RELEASE-AWS-P5.json", release)

    dataset = {
        "build_id": "4" * 64,
        "ordered_stream_sha256": "5" * 64,
    }
    dataset_receipt = _write_json(
        tmp_path / "dataset" / "receipt.json",
        dataset,
    )
    runtime_lock = _write_json(
        tmp_path / "runtime-lock.json",
        _runtime_lock(profile_sha256),
    )
    environment = _write_json(
        tmp_path / "environment.json",
        _environment_receipt(
            profile_sha256=profile_sha256,
            runtime_lock_sha256=_sha256(runtime_lock),
        ),
    )
    seed = 0
    manifest = {
        "schema_version": 3,
        "provider": "aws-p5.48xlarge",
        "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
        "seed": seed,
        "release_sha256": release["archive"]["sha256"],
        "release_receipt_sha256": _sha256(release_receipt),
        "profile_sha256": profile_sha256,
        "dataset_pointer_sha256": release["dataset_pointer_sha256"],
        "dataset_receipt_sha256": _sha256(dataset_receipt),
        "dataset_build_id": dataset["build_id"],
        "ordered_stream_sha256": dataset["ordered_stream_sha256"],
        "cohort_assignment_sha256": release["cohort_assignment_sha256"],
        "preregistration_sha256": member_hashes[
            "configs/preregistration-v3.yaml"
        ],
        "sealed_evaluation_release_sha256": "6" * 64,
        "source_commit": release["source"]["commit"],
        "source_tree": release["source"]["tree"],
        "runs": [
            {
                "run_id": f"memorysplit-v3-360m-s{seed}-{arm}",
                "arm": arm,
                "seed": seed,
                "config": f"configs/360m-v3/{arm}-s{seed}.yaml",
                "config_sha256": config_sha256[
                    f"configs/360m-v3/{arm}-s{seed}.yaml"
                ],
                "estimated_gpu_hours": 10.0,
            }
            for arm in ("dense", "split90")
        ],
    }
    manifest_path = _write_json(tmp_path / "run-manifest.json", manifest)
    scratch = tmp_path / "canary-private"
    scratch.mkdir(mode=0o700)

    def verify_dataset(path, *, expected_sha256, expected_ordered_sha256):
        assert Path(path) == dataset_receipt
        assert expected_sha256 == _sha256(dataset_receipt)
        assert expected_ordered_sha256 == dataset["ordered_stream_sha256"]
        return SimpleNamespace(receipt=dataset, files=())

    plan = module.load_canary_plan(
        release_root=release_root,
        release_receipt_path=release_receipt,
        run_manifest_path=manifest_path,
        dataset_receipt_path=dataset_receipt,
        environment_receipt_path=environment,
        runtime_lock_path=runtime_lock,
        instance_id=INSTANCE_ID,
        boot_id=BOOT_ID,
        scratch_root=scratch,
        output_path=scratch / "qualification.json",
        s3_root="s3://memorysplit-prod/confirmatory-v3",
        dataset_verifier=verify_dataset,
    )
    return {
        "plan": plan,
        "release_root": release_root,
        "release_receipt": release_receipt,
        "manifest_path": manifest_path,
        "dataset_receipt": dataset_receipt,
        "environment": environment,
        "runtime_lock": runtime_lock,
        "scratch": scratch,
        "output": scratch / "qualification.json",
        "dataset": dataset,
        "dataset_verifier": verify_dataset,
    }


def _topology() -> str:
    header = "        " + " ".join(f"GPU{i}" for i in range(8))
    rows = []
    for left in range(8):
        cells = ["X" if left == right else "NV18" for right in range(8)]
        rows.append(f"GPU{left}   " + " ".join(cells))
    return "\n".join([header, *rows, "Legend: NV# = NVLink"]) + "\n"


class _Commands:
    def __init__(self, module, plan, *, fail: str | None = None) -> None:
        self.module = module
        self.plan = plan
        self.fail = fail
        self.calls = []
        self.pairs = []
        self.checkpoints: dict[tuple[str, str], Path] = {}

    def _result(self, spec):
        self.calls.append(spec)
        if spec.name == self.fail:
            return self.module.CommandResult(17, "", "synthetic failure")
        if spec.name == "hardware-gpus":
            stdout = "".join(
                f"{index}, NVIDIA H100 80GB HBM3, 81559\n"
                for index in range(8)
            )
            return self.module.CommandResult(0, stdout, "")
        if spec.name == "hardware-fabric-manager":
            return self.module.CommandResult(0, "active\n", "")
        if spec.name == "hardware-topology":
            return self.module.CommandResult(0, _topology(), "")
        if spec.name == "hardware-boot-id":
            return self.module.CommandResult(0, BOOT_ID + "\n", "")
        if spec.name == "nccl-all-reduce":
            return self.module.CommandResult(
                0,
                json.dumps(
                    {
                        "latency_seconds": 0.004,
                        "reduced_sum": 36.0,
                        "world_size": 8,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                "",
            )
        phase, arm, kind = spec.name.split("-", 2)
        if kind == "train":
            work_root = Path(spec.host_work_root)
            output = work_root / f"runs/seed-{self.plan.seed}/{arm}"
            output.mkdir(parents=True, mode=0o700, exist_ok=True)
            checkpoint = output / "ckpt.pt"
            checkpoint.write_bytes(f"{phase}:{arm}:checkpoint".encode("ascii"))
            source_config = self.plan.release_root / (
                f"configs/360m-v3/{arm}-s{self.plan.seed}.yaml"
            )
            (output / "config.yaml").write_bytes(source_config.read_bytes())
            step = 1 if phase == "functional" else 2
            (output / "log.jsonl").write_text(
                json.dumps(
                    {
                        "step": step,
                        "loss": 1.25 if arm == "dense" else 1.5,
                    }
                )
                + "\n",
                encoding="ascii",
            )
            self.checkpoints[(phase, arm)] = checkpoint
            return self.module.CommandResult(0, "done\n", "")
        if kind == "inspect":
            checkpoint = self.checkpoints[(phase, arm)]
            step = 1 if phase == "functional" else 2
            sidecar = (
                "dense_target_weights"
                if arm == "dense"
                else "split90_target_weights"
            )
            evidence = {
                "checkpoint_sha256": _sha256(checkpoint),
                "config_fingerprint": hashlib.sha256(
                    f"config:{arm}".encode("ascii")
                ).hexdigest(),
                "config_matches": True,
                "data": {
                    "build_id": self.plan.dataset_build_id,
                    "global_cursor": step * 524_288,
                    "ordered_stream_sha256": self.plan.ordered_stream_sha256,
                    "receipt_sha256": self.plan.dataset_receipt_sha256,
                    "sidecar_name": sidecar,
                },
                "step": step,
                "world_size": 1,
            }
            return self.module.CommandResult(
                0,
                json.dumps(evidence, sort_keys=True, separators=(",", ":"))
                + "\n",
                "",
            )
        raise AssertionError(spec.name)

    def run(self, spec):
        return self._result(spec)

    def run_pair(self, specs):
        self.pairs.append(tuple(specs))
        results = {}
        for spec in specs:
            if spec.name == self.fail:
                results[spec.name] = self.module.CommandResult(
                    19,
                    "",
                    "synthetic peer failure",
                )
                continue
            arm = "dense" if "dense" in spec.name else "split90"
            base = 1000.0 if arm == "dense" else 900.0
            metrics = {
                "end_step": 100,
                "receipt_type": "memorysplit-operational-training-v1",
                "start_step": 0,
                "step_tok_s": [base + index for index in range(100)],
                "updates": 100,
            }
            results[spec.name] = self.module.CommandResult(
                0,
                json.dumps(metrics, sort_keys=True, separators=(",", ":"))
                + "\n",
                "",
            )
        return results


class _ObjectStore:
    def __init__(self, module, *, mutation: str | None = None) -> None:
        self.module = module
        self.mutation = mutation
        self.puts = []
        self.gets = []
        self.payload = b""
        self.uri = ""

    def put(self, uri, payload, *, sha256):
        self.puts.append((uri, payload, sha256))
        self.payload = payload
        self.uri = uri
        return self.module.ObjectWrite(
            checksum_sha256=("0" * 64 if self.mutation == "put-hash" else sha256),
            byte_count=(len(payload) + 1 if self.mutation == "put-bytes" else len(payload)),
            version_id=(
                "null" if self.mutation == "put-version" else "canary-version-1"
            ),
        )

    def get(self, uri, *, version_id):
        self.gets.append((uri, version_id))
        payload = self.payload + (b"x" if self.mutation == "get-bytes" else b"")
        return self.module.ObjectRead(
            payload=payload,
            checksum_sha256=(
                "0" * 64
                if self.mutation == "get-hash"
                else hashlib.sha256(payload).hexdigest()
            ),
            version_id=(
                "other-version"
                if self.mutation == "get-version"
                else version_id
            ),
        )


class _Time:
    def __init__(self) -> None:
        self.wall = datetime(2026, 7, 23, 20, 0, tzinfo=UTC)
        self.tick = -1.0

    def now(self):
        value = self.wall
        self.wall += timedelta(seconds=1)
        return value

    def monotonic(self):
        self.tick += 1.0
        return self.tick


def _execute(case, module, *, commands=None, store=None, apply=True):
    plan = case["plan"]
    return module.execute_canary(
        plan,
        command_reader=commands or _Commands(module, plan),
        object_store=store or _ObjectStore(module),
        time_reader=_Time(),
        apply=apply,
    )


def test_all_six_phases_render_exact_argv_topology_order_and_receipt(tmp_path):
    module = _load_module()
    case = _case(tmp_path, module)
    plan = case["plan"]

    rendered = module.render_plan(plan)

    assert rendered["phase_order"] == list(module.PHASE_ORDER) == [
        "hardware",
        "nccl_all_reduce",
        "functional",
        "resume",
        "throughput_4x4",
        "s3_roundtrip",
    ]
    phases = {phase["name"]: phase for phase in rendered["phases"]}
    assert [
        row["name"] for row in phases["hardware"]["commands"]
    ] == [
        "hardware-gpus",
        "hardware-fabric-manager",
        "hardware-topology",
        "hardware-boot-id",
    ]
    functional = phases["functional"]["commands"]
    resume = phases["resume"]["commands"]
    throughput = phases["throughput_4x4"]["commands"]
    assert [row["name"] for row in functional] == [
        "functional-dense-train",
        "functional-dense-inspect",
        "functional-split90-train",
        "functional-split90-inspect",
    ]
    assert all(
        command["argv"][command["argv"].index("--operational-steps") + 1]
        == "1"
        for command in functional + resume
        if "--operational-steps" in command["argv"]
    )
    assert len(throughput) == 2
    assert all("--nproc_per_node=4" in row["argv"] for row in throughput)
    assert all(
        row["argv"][row["argv"].index("--operational-steps") + 1] == "100"
        for row in throughput
    )
    dense, split90 = throughput
    assert dense["gpu_ids"] == [0, 1, 2, 3]
    assert split90["gpu_ids"] == [4, 5, 6, 7]
    assert dense["cpu_affinity"] == [0, 95]
    assert split90["cpu_affinity"] == [96, 191]
    assert dense["master_port"] != split90["master_port"]
    assert dense["concurrent_group"] == split90["concurrent_group"]

    commands = _Commands(module, plan)
    store = _ObjectStore(module)
    receipt = _execute(
        case,
        module,
        commands=commands,
        store=store,
        apply=True,
    )

    assert set(receipt) == RECEIPT_FIELDS
    assert receipt["receipt_type"] == "memorysplit-aws-p5-qualification-v1"
    assert receipt["provider"] == "aws-p5.48xlarge"
    assert receipt["instance_id"] == INSTANCE_ID
    assert receipt["boot_id"] == BOOT_ID
    assert receipt["passed"] is True
    assert [phase["name"] for phase in receipt["phases"]] == list(
        module.PHASE_ORDER
    )
    assert all(phase["passed"] is True for phase in receipt["phases"])
    assert receipt["hardware"]["gpu_count"] == 8
    assert receipt["hardware"]["fabric_manager_active"] is True
    assert receipt["hardware"]["nvswitch_fully_connected"] is True
    assert receipt["hardware"]["runtime_facts"] == VERSIONS
    for arm in ("dense", "split90"):
        assert receipt["functional"][arm]["step"] == 1
        assert receipt["functional"][arm]["global_cursor"] == 524_288
        assert receipt["resume"][arm]["step"] == 2
        assert receipt["resume"][arm]["global_cursor"] == 1_048_576
        assert (
            receipt["resume"][arm]["config_fingerprint"]
            == receipt["functional"][arm]["config_fingerprint"]
        )
        assert (
            receipt["resume"][arm]["resumed_from_checkpoint_sha256"]
            == receipt["functional"][arm]["checkpoint_sha256"]
        )
    throughput_receipt = receipt["throughput_4x4"]
    assert throughput_receipt["updates"] == 100
    assert throughput_receipt["warmup_updates"] == 10
    assert throughput_receipt["world_size_per_arm"] == 4
    assert throughput_receipt["aggregate_median_tok_s"] > 0
    assert math.isfinite(throughput_receipt["aggregate_median_tok_s"])
    assert receipt["s3_roundtrip"]["version_id"] == "canary-version-1"
    assert len(store.puts) == len(store.gets) == 1
    assert store.gets[0] == (store.puts[0][0], "canary-version-1")
    assert case["output"].read_bytes() == module.canonical_receipt(receipt)
    assert stat.S_IMODE(case["output"].stat().st_mode) == 0o600
    assert len(commands.pairs) == 1


@pytest.mark.parametrize(
    "values",
    [
        [float(index + 1) for index in range(99)],
        [1.0] * 10 + [0.0] + [1.0] * 89,
        [1.0] * 10 + [float("nan")] + [1.0] * 89,
        [1.0] * 10 + [float("inf")] + [1.0] * 89,
    ],
)
def test_throughput_rejects_missing_zero_and_nonfinite_updates(values):
    module = _load_module()

    with pytest.raises(module.CanaryError, match="throughput"):
        module.compute_throughput(
            {"dense": values, "split90": [2.0] * 100},
            updates=100,
            warmup_updates=10,
        )


def test_throughput_excludes_first_ten_and_computes_deterministic_medians():
    module = _load_module()
    dense = [100_000.0] * 10 + [float(index) for index in range(1, 91)]
    split90 = [200_000.0] * 10 + [float(index * 2) for index in range(1, 91)]

    result = module.compute_throughput(
        {"dense": dense, "split90": split90},
        updates=100,
        warmup_updates=10,
    )

    assert result == {
        "aggregate_median_tok_s": 136.5,
        "per_arm_median_tok_s": {"dense": 45.5, "split90": 91.0},
        "updates": 100,
        "warmup_updates": 10,
        "world_size_per_arm": 4,
    }


@pytest.mark.parametrize(
    "failure",
    [
        "hardware-gpus",
        "nccl-all-reduce",
        "functional-dense-train",
        "resume-split90-inspect",
        "throughput-dense-train",
    ],
)
def test_every_command_phase_failure_emits_no_receipt(tmp_path, failure):
    module = _load_module()
    case = _case(tmp_path, module)
    commands = _Commands(module, case["plan"], fail=failure)

    with pytest.raises(module.CanaryError):
        _execute(case, module, commands=commands)

    assert not case["output"].exists()


def test_hardware_rejects_non_nvswitch_topology_without_receipt(tmp_path):
    module = _load_module()
    case = _case(tmp_path, module)

    class BadTopology(_Commands):
        def _result(self, spec):
            if spec.name == "hardware-topology":
                self.calls.append(spec)
                header = "        " + " ".join(f"GPU{i}" for i in range(8))
                rows = [
                    f"GPU{left}   "
                    + " ".join(
                        "X" if left == right else "PIX"
                        for right in range(8)
                    )
                    for left in range(8)
                ]
                return module.CommandResult(
                    0,
                    "\n".join([header, *rows]) + "\n",
                    "",
                )
            return super()._result(spec)

    with pytest.raises(module.CanaryError, match="NVSwitch"):
        _execute(
            case,
            module,
            commands=BadTopology(module, case["plan"]),
        )

    assert not case["output"].exists()


def test_hardware_rejects_current_boot_drift_without_receipt(tmp_path):
    module = _load_module()
    case = _case(tmp_path, module)

    class WrongBoot(_Commands):
        def _result(self, spec):
            if spec.name == "hardware-boot-id":
                self.calls.append(spec)
                return module.CommandResult(
                    0,
                    "87654321-4321-4abc-8def-1234567890ab\n",
                    "",
                )
            return super()._result(spec)

    with pytest.raises(module.CanaryError, match="boot"):
        _execute(
            case,
            module,
            commands=WrongBoot(module, case["plan"]),
        )

    assert not case["output"].exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "put-hash",
        "put-bytes",
        "put-version",
        "get-bytes",
        "get-hash",
        "get-version",
    ],
)
def test_s3_roundtrip_requires_checksum_bytes_and_version(tmp_path, mutation):
    module = _load_module()
    case = _case(tmp_path, module)
    store = _ObjectStore(module, mutation=mutation)

    with pytest.raises(module.CanaryError, match="S3|object|version|checksum|bytes"):
        _execute(case, module, store=store)

    assert not case["output"].exists()


def test_receipt_is_owner_only_atomic_no_replace_and_failed_rerun_preserves_it(
    tmp_path,
):
    module = _load_module()
    case = _case(tmp_path, module)
    first = _execute(case, module)
    before = case["output"].read_bytes()

    with pytest.raises(module.CanaryError, match="exists|replace"):
        _execute(case, module)

    assert case["output"].read_bytes() == before == module.canonical_receipt(first)
    assert stat.S_IMODE(case["output"].stat().st_mode) == 0o600


def test_atomic_publication_failure_never_leaves_a_passing_receipt(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    output = root / "qualification.json"
    real_fsync = module.os.fsync
    calls = 0

    def fail_directory_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic directory fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", fail_directory_fsync)

    with pytest.raises(OSError):
        module._atomic_no_replace(output, b'{"passed":true}\n')

    assert not output.exists()


def test_receipt_parser_rejects_identity_phase_pair_and_s3_mutations(tmp_path):
    module = _load_module()
    case = _case(tmp_path, module)
    receipt = _execute(case, module, apply=False)
    mutations = [
        lambda value: value.update(instance_id="i-00000000"),
        lambda value: value["phases"].reverse(),
        lambda value: value["hardware"].update(gpu_count=7),
        lambda value: value["functional"]["dense"].update(step=2),
        lambda value: value["resume"]["dense"].update(
            config_fingerprint="0" * 64
        ),
        lambda value: value["throughput_4x4"].update(updates=99),
        lambda value: value["s3_roundtrip"].update(version_id="null"),
        lambda value: value.update(unexpected=True),
        lambda value: value.pop("source_tree"),
    ]

    for mutate in mutations:
        candidate = copy.deepcopy(receipt)
        mutate(candidate)
        with pytest.raises(module.CanaryError):
            module.parse_qualification_receipt_bytes(
                _canonical(candidate),
                plan=case["plan"],
            )


def test_input_identity_mutation_fails_before_any_command(tmp_path):
    module = _load_module()
    case = _case(tmp_path, module)
    environment = json.loads(case["environment"].read_text(encoding="ascii"))
    environment["boot_id"] = str(uuid.uuid4())
    case["environment"].write_bytes(_canonical(environment))
    commands = _Commands(module, case["plan"])

    with pytest.raises(module.CanaryError):
        module.load_canary_plan(
            release_root=case["release_root"],
            release_receipt_path=case["release_receipt"],
            run_manifest_path=case["manifest_path"],
            dataset_receipt_path=case["dataset_receipt"],
            environment_receipt_path=case["environment"],
            runtime_lock_path=case["runtime_lock"],
            instance_id=INSTANCE_ID,
            boot_id=BOOT_ID,
            scratch_root=case["scratch"],
            output_path=case["output"],
            s3_root="s3://memorysplit-prod/confirmatory-v3",
            dataset_verifier=lambda *_args, **_kwargs: pytest.fail(
                "identity drift must fail before dataset or command execution"
            ),
        )

    assert commands.calls == []
    assert not case["output"].exists()


class _NoAwsCalls:
    def __init__(self) -> None:
        self.calls = []

    def run_json(self, argv, *, operation):
        self.calls.append((list(argv), operation))
        raise AssertionError(f"unexpected AWS call during dry run: {operation}")


def _backend(
    tmp_path: Path,
    case,
    *,
    runner=None,
    identity_verifier=None,
    sleep=None,
    monotonic=None,
):
    from cluster.aws.p5.profile import load_aws_p5_profile
    from msctl.aws_p5 import AwsP5Backend

    profile = load_aws_p5_profile(
        case["release_root"]
        / "cluster"
        / "profiles"
        / "aws-p5.48xlarge-v3.json"
    )
    runtime = SimpleNamespace(
        region="us-east-1",
        s3_root="s3://memorysplit-prod/confirmatory-v3",
        ami_id="ami-0123456789abcdef0",
        container_image=IMAGE,
        container_digest=IMAGE_DIGEST,
        uid=1000,
        gid=1000,
    )
    arguments = {
        "profile": profile,
        "runtime": runtime,
        "instance_profile_arn": (
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        "state_root": tmp_path / "controller-state",
        "runner": runner or _NoAwsCalls(),
        "corpus_verifier": case["dataset_verifier"],
        "identity_verifier": identity_verifier or (lambda *_args: True),
        "environ": {},
    }
    if sleep is not None:
        arguments["sleep"] = sleep
    if monotonic is not None:
        arguments["monotonic"] = monotonic
    return AwsP5Backend(
        **arguments,
    )


def _canary_controller_call(backend, case, *, apply):
    return backend.canary_run(
        release_root=case["release_root"],
        release_receipt=case["release_receipt"],
        run_manifest=case["manifest_path"],
        dataset_receipt=case["dataset_receipt"],
        environment_receipt=case["environment"],
        runtime_lock=case["runtime_lock"],
        instance_id=INSTANCE_ID,
        boot_id=BOOT_ID,
        output=case["output"],
        apply=apply,
    )


def test_controller_dry_run_is_local_content_addressed_and_argv_only(tmp_path):
    module = _load_module()
    case = _case(tmp_path, module)
    runner = _NoAwsCalls()
    backend = _backend(tmp_path, case, runner=runner)

    planned = _canary_controller_call(backend, case, apply=False)

    assert runner.calls == []
    assert planned["published"] is False
    assert planned["verified"] is True
    assert planned["qualification_tuple_sha256"] == case["plan"].tuple_sha256
    assert planned["receipt_key_prefix"] == (
        f"canaries/{case['plan'].tuple_sha256}/"
    )
    argv = planned["remote_canary_argv"]
    assert argv[0] == "/usr/bin/python3"
    assert argv[1] == str(
        case["release_root"] / "cluster" / "aws" / "p5" / "canary.py"
    )
    assert argv[-1] == "--apply"
    assert "--operational-steps" not in argv
    intent = planned["operation_intent"]
    assert intent["operation"] == "canary"
    assert intent["instance_id"] == INSTANCE_ID
    assert intent["terminate_at"] is None
    assert intent["steps"] == [{"name": "qualification-canary", "argv": argv}]
    assert intent["operation_id"] == hashlib.sha256(
        module._canonical(
            {
                key: value
                for key, value in intent.items()
                if key
                not in {
                    "operation_id",
                    "ssm_document",
                    "started_receipt_uri",
                    "terminal_receipt_uri",
                }
            },
            newline=False,
        )
    ).hexdigest()
    assert not case["output"].exists()


class _ApplyRunner:
    def __init__(
        self,
        module,
        plan,
        receipt,
        intent,
        *,
        existing_final: bool = False,
        pending_once: bool = False,
    ) -> None:
        self.module = module
        self.plan = plan
        self.receipt = receipt
        self.receipt_bytes = module.canonical_receipt(receipt)
        self.intent = intent
        self.intent_bytes = module._canonical(intent, newline=False)
        self.intent_sha256 = hashlib.sha256(self.intent_bytes).hexdigest()
        self.calls = []
        self.existing_final = existing_final
        self.pending_once = pending_once
        self.consume_calls = 0

    @staticmethod
    def _checksum(digest: str) -> str:
        import base64

        return base64.b64encode(bytes.fromhex(digest)).decode("ascii")

    def run_json(self, argv, *, operation):
        argv = list(argv)
        self.calls.append((argv, operation))
        if operation == "verify canary instance":
            identity = json.loads(
                Path(self.plan.environment_receipt_path).read_text(
                    encoding="ascii"
                )
            )["aws_instance_identity_document"]
            return {
                "instance": {
                    "account_id": "123456789012",
                    "instance_id": INSTANCE_ID,
                    "image_id": "ami-0123456789abcdef0",
                    "instance_type": "p5.48xlarge",
                    "state": "running",
                    "architecture": identity["architecture"],
                    "private_ip": identity["privateIp"],
                }
            }
        if operation == "SSM readiness":
            return {
                "managed_instances": [
                    {"instance_id": INSTANCE_ID, "ping_status": "Online"}
                ]
            }
        if operation == "verify SSM document":
            from msctl.aws_argv import ARGV_DOCUMENT_NAME, ARGV_DOCUMENT_SHA256

            return {
                "documents": [
                    {
                        "name": ARGV_DOCUMENT_NAME,
                        "hash": ARGV_DOCUMENT_SHA256,
                        "status": "Active",
                    }
                ]
            }
        if operation == "publish operation intent":
            return {
                "object": {
                    "checksum_sha256": self._checksum(self.intent_sha256),
                    "version_id": "intent-version-1",
                }
            }
        if operation == "verify operation intent":
            return {
                "object": {
                    "checksum_sha256": self._checksum(self.intent_sha256),
                    "content_length": len(self.intent_bytes),
                    "metadata": {
                        "operation-id": self.intent["operation_id"],
                        "sha256": self.intent_sha256,
                    },
                    "version_id": "intent-version-1",
                }
            }
        if operation == "send canary operation":
            return {"command": {"command_id": "command-12345678"}}
        if operation == "consume canary operation":
            self.consume_calls += 1
            if self.pending_once and self.consume_calls == 1:
                return {
                    "command": {
                        "command_id": "command-12345678",
                        "status": "InProgress",
                        "stdout": "",
                        "stderr": "",
                    }
                }
            return {
                "command": {
                    "command_id": "command-12345678",
                    "status": "Success",
                    "stdout": (
                        self.receipt_bytes.decode("ascii")
                        + '{"executed":true,"schema_version":1}\n'
                    ),
                    "stderr": "",
                }
            }
        if operation == "verify canary roundtrip object":
            digest = self.receipt["s3_roundtrip"]["sha256"]
            return {
                "object": {
                    "checksum_sha256": self._checksum(digest),
                    "content_length": self.receipt["s3_roundtrip"]["bytes"],
                    "version_id": self.receipt["s3_roundtrip"]["version_id"],
                }
            }
        if operation == "download canary roundtrip object":
            destination = Path(argv[argv.index("--checksum-mode") + 2])
            destination.write_bytes(self.module._roundtrip_blob(self.plan))
            digest = self.receipt["s3_roundtrip"]["sha256"]
            return {
                "object": {
                    "checksum_sha256": self._checksum(digest),
                    "version_id": self.receipt["s3_roundtrip"]["version_id"],
                }
            }
        receipt_sha256 = hashlib.sha256(self.receipt_bytes).hexdigest()
        if operation == "publish canary receipt":
            if self.existing_final:
                from msctl.errors import MsctlError

                raise MsctlError("AWS_COMMAND_FAILED", "precondition failed")
            return {
                "object": {
                    "checksum_sha256": self._checksum(receipt_sha256),
                    "version_id": "qualification-version-1",
                }
            }
        if operation == "verify canary receipt publication":
            return {
                "object": {
                    "checksum_sha256": self._checksum(receipt_sha256),
                    "content_length": len(self.receipt_bytes),
                    "metadata": {
                        "qualification-tuple-sha256": self.plan.tuple_sha256,
                        "receipt-sha256": receipt_sha256,
                    },
                    "version_id": "qualification-version-1",
                }
            }
        raise AssertionError(f"unexpected AWS operation: {operation}")


def test_controller_apply_sends_consumes_verifies_and_publishes_no_replace(
    tmp_path,
):
    module = _load_module()
    case = _case(tmp_path, module)
    receipt = _execute(case, module, apply=False)
    dry_backend = _backend(tmp_path / "dry", case)
    planned = _canary_controller_call(dry_backend, case, apply=False)
    runner = _ApplyRunner(
        module,
        case["plan"],
        receipt,
        planned["operation_intent"],
    )
    backend = _backend(tmp_path / "apply", case, runner=runner)

    applied = _canary_controller_call(backend, case, apply=True)

    receipt_sha256 = hashlib.sha256(runner.receipt_bytes).hexdigest()
    assert applied["published"] is True
    assert applied["verified"] is True
    assert applied["receipt_sha256"] == receipt_sha256
    assert applied["receipt_key"] == (
        f"canaries/{case['plan'].tuple_sha256}/{receipt_sha256}.json"
    )
    assert applied["receipt_uri"] == (
        "s3://memorysplit-prod/confirmatory-v3/" + applied["receipt_key"]
    )
    assert applied["version_id"] == "qualification-version-1"
    assert applied["command_id"] == "command-12345678"
    operations = [operation for _argv, operation in runner.calls]
    assert operations == [
        "verify canary instance",
        "SSM readiness",
        "verify SSM document",
        "publish operation intent",
        "verify operation intent",
        "send canary operation",
        "consume canary operation",
        "verify canary roundtrip object",
        "download canary roundtrip object",
        "publish canary receipt",
        "verify canary receipt publication",
    ]
    puts = [argv for argv, _ in runner.calls if "put-object" in argv]
    assert len(puts) == 2
    assert all(argv[argv.index("--if-none-match") + 1] == "*" for argv in puts)
    roundtrip_get = next(
        argv
        for argv, operation in runner.calls
        if operation == "download canary roundtrip object"
    )
    assert roundtrip_get[
        roundtrip_get.index("--version-id") + 1
    ] == receipt["s3_roundtrip"]["version_id"]


def test_controller_existing_identical_receipt_is_verified_without_replacement(
    tmp_path,
):
    module = _load_module()
    case = _case(tmp_path, module)
    receipt = _execute(case, module, apply=False)
    intent = _canary_controller_call(
        _backend(tmp_path / "dry", case),
        case,
        apply=False,
    )["operation_intent"]
    runner = _ApplyRunner(
        module,
        case["plan"],
        receipt,
        intent,
        existing_final=True,
    )
    backend = _backend(tmp_path / "apply", case, runner=runner)

    result = _canary_controller_call(backend, case, apply=True)

    assert result["published"] is True
    assert result["version_id"] == "qualification-version-1"
    final_put = [
        argv
        for argv, operation in runner.calls
        if operation == "publish canary receipt"
    ]
    assert len(final_put) == 1
    assert final_put[0][final_put[0].index("--if-none-match") + 1] == "*"


def test_controller_waits_for_remote_canary_before_consuming_receipt(tmp_path):
    module = _load_module()
    case = _case(tmp_path, module)
    receipt = _execute(case, module, apply=False)
    intent = _canary_controller_call(
        _backend(tmp_path / "dry", case),
        case,
        apply=False,
    )["operation_intent"]
    runner = _ApplyRunner(
        module,
        case["plan"],
        receipt,
        intent,
        pending_once=True,
    )
    sleeps = []
    ticks = iter((0.0, 1.0, 2.0, 3.0))
    backend = _backend(
        tmp_path / "apply",
        case,
        runner=runner,
        sleep=sleeps.append,
        monotonic=lambda: next(ticks),
    )

    result = _canary_controller_call(backend, case, apply=True)

    assert result["published"] is True
    assert runner.consume_calls == 2
    assert sleeps == [5.0]


def test_remote_wrapper_accepts_only_the_exact_canary_operation_boundary(
    tmp_path,
):
    from msctl import aws_argv

    module = _load_module()
    case = _case(tmp_path, module)
    backend = _backend(tmp_path, case)
    intent = _canary_controller_call(
        backend,
        case,
        apply=False,
    )["operation_intent"]
    payload = module._canonical(intent, newline=False)
    digest = hashlib.sha256(payload).hexdigest()

    assert aws_argv._validate_intent(
        payload,
        expected_sha256=digest,
    ) == intent

    mutation = copy.deepcopy(intent)
    mutation["steps"][0]["argv"] = [
        "/usr/bin/python3",
        "/release/scripts/run_train.py",
        "--operational-steps",
        "100",
    ]
    identity = {
        key: value
        for key, value in mutation.items()
        if key
        not in {
            "operation_id",
            "ssm_document",
            "started_receipt_uri",
            "terminal_receipt_uri",
        }
    }
    mutation["operation_id"] = hashlib.sha256(
        module._canonical(identity, newline=False)
    ).hexdigest()
    receipt_root = (
        f"{mutation['environment']['MS_S3_ROOT']}/operations/"
        f"{mutation['operation_id']}/receipts"
    )
    mutation["started_receipt_uri"] = f"{receipt_root}/started.json"
    mutation["terminal_receipt_uri"] = f"{receipt_root}/terminal.json"
    mutated_payload = module._canonical(mutation, newline=False)

    with pytest.raises(aws_argv.RemoteIntentError, match="canary|operational"):
        aws_argv._validate_intent(
            mutated_payload,
            expected_sha256=hashlib.sha256(mutated_payload).hexdigest(),
        )


def test_cli_enables_canary_only_for_v3_and_requires_all_explicit_inputs(
    tmp_path,
):
    from cluster.aws.p5.profile import load_aws_p5_profile
    from msctl.cli import build_parser, dispatch
    from msctl.errors import MsctlError

    module = _load_module()
    case = _case(tmp_path, module)
    profile = load_aws_p5_profile(
        case["release_root"]
        / "cluster"
        / "profiles"
        / "aws-p5.48xlarge-v3.json"
    )
    captured = {}

    class Backend:
        def dispatch(self, command, args):
            captured["command"] = command
            captured["args"] = args
            return not args.apply, {"operation": command}

    arguments = [
        "--profile",
        str(case["release_root"] / PROFILE.relative_to(ROOT)),
        "--repo-root",
        str(case["release_root"]),
        "canary",
        "run",
        "--release",
        str(case["release_receipt"]),
        "--manifest",
        str(case["manifest_path"]),
        "--dataset-receipt",
        str(case["dataset_receipt"]),
        "--environment-receipt",
        str(case["environment"]),
        "--runtime-lock",
        str(case["runtime_lock"]),
        "--instance-id",
        INSTANCE_ID,
        "--boot-id",
        BOOT_ID,
        "--output",
        str(case["output"]),
    ]
    args = build_parser().parse_args(arguments)

    dry_run, result = dispatch(
        args,
        profile_loader=lambda _path: profile,
        aws_backend_factory=lambda **_kwargs: Backend(),
        environ={},
    )

    assert dry_run is True
    assert result == {"operation": "canary run"}
    assert captured["command"] == "canary run"

    required_flags = (
        "--release",
        "--manifest",
        "--dataset-receipt",
        "--environment-receipt",
        "--runtime-lock",
        "--instance-id",
        "--boot-id",
        "--output",
    )
    for flag in required_flags:
        index = arguments.index(flag)
        missing = arguments[:index] + arguments[index + 2 :]
        with pytest.raises(MsctlError) as caught:
            build_parser().parse_args(missing)
        assert caught.value.code == "CLI_USAGE"

    legacy = SimpleNamespace(
        provider="aws-p5.48xlarge",
        profile_id="aws-p5.48xlarge",
    )
    with pytest.raises(MsctlError) as unsupported:
        dispatch(
            args,
            profile_loader=lambda _path: legacy,
            aws_backend_factory=lambda **_kwargs: pytest.fail(
                "non-v3 canary must fail before backend construction"
            ),
            environ={},
        )
    assert unsupported.value.code == "PROVIDER_UNSUPPORTED"

    with pytest.raises(MsctlError) as production_flag:
        build_parser().parse_args(
            [
                "submit",
                "--release",
                "release.json",
                "--manifest",
                "manifest.json",
                "--operational-steps",
                "1",
            ]
        )
    assert production_flag.value.code == "CLI_USAGE"
