from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cluster.aws.gpu_profile import load_aws_gpu_profile
from msctl.aws_hardware import AuthenticatedSelectionBinding


ROOT = Path(__file__).resolve().parents[1]
P5_PROFILE = ROOT / "cluster" / "profiles" / "aws-p5.48xlarge-v3.json"
P6_PROFILE = ROOT / "cluster" / "profiles" / "aws-p6-b300.48xlarge-v3.json"
QUALIFICATION_WORKER = ROOT / "cluster" / "aws" / "qualification_worker.py"
IMAGE_DIGEST = "sha256:" + "a" * 64
IMAGE = (
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/memorysplit/aws-gpu"
    f"@{IMAGE_DIGEST}"
)
AMI_ID = "ami-0260c4d597dcc8641"
AMI_OWNER_ID = "898082745236"
ACCOUNT_ID = "123456789012"
INSTANCE_ID = "i-0123456789abcdef0"
BOOT_ID = "12345678-1234-4abc-8def-1234567890ab"
CONTAINER_FACTS = {
    "python": "3.12.11",
    "pytorch": "2.9.0+cu130",
    "cuda": "13.0",
    "cudnn": "9.10.2",
    "nccl": "2.28.3",
}
HOST_FACTS = {
    "cuda": "13.2",
    "nvidia_driver": "595.71.05",
    "fabric_manager": "595.71.05",
    "docker": "28.5.1",
    "nvidia_container_runtime": "1.18.0",
    "aws_cli": "2.31.7",
    "kernel": "6.17",
    "efa": "1.47.0",
    "ofi_nccl": "1.18.0",
    "nvlsm": "595.71.05",
}


def _canonical(value: object) -> bytes:
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


def _runtime_inputs(profile):
    lock = {
        "schema_version": 1,
        "source_commit": "b" * 40,
        "source_tree": "c" * 40,
        "control_bundle_sha256": "d" * 64,
        "profile_sha256": profile.sha256,
        "ami_id": AMI_ID,
        "ami_owner_id": AMI_OWNER_ID,
        "container_image": IMAGE,
        "container_image_digest": IMAGE_DIGEST,
        "versions": {
            **CONTAINER_FACTS,
            "nvidia_driver": HOST_FACTS["nvidia_driver"],
            "fabric_manager": HOST_FACTS["fabric_manager"],
            "docker": HOST_FACTS["docker"],
            "nvidia_container_runtime": HOST_FACTS[
                "nvidia_container_runtime"
            ],
            "aws_cli": HOST_FACTS["aws_cli"],
        },
    }
    lock_data = _canonical(lock)
    sbom = {
        "schema_version": 2,
        "document_type": "memorysplit-aws-gpu-sbom-v2",
        "source": {
            "commit": lock["source_commit"],
            "tree": lock["source_tree"],
        },
        "runtime_lock_sha256": hashlib.sha256(lock_data).hexdigest(),
        "image_binding_sha256": "e" * 64,
        "project_dependency_lock": {
            "sha256": "f" * 64,
            "packages": [],
        },
        "host": {
            "ami_id": AMI_ID,
            "ami_owner_id": AMI_OWNER_ID,
            "ami_name": (
                "Deep Learning Base AMI with Single CUDA Ubuntu 24.04 "
                "20260523"
            ),
            "architecture": "x86_64",
            "versions": dict(HOST_FACTS),
            "minimum_versions": {
                "cuda": "13.0",
                "nvidia_driver": "580.0",
                "kernel": "6.1",
                "efa": "1.44.0",
                "ofi_nccl": "1.17.1",
            },
        },
        "container": {
            "image": IMAGE,
            "image_digest": IMAGE_DIGEST,
            "versions": dict(CONTAINER_FACTS),
        },
    }
    return lock_data, _canonical(sbom)


def _binding(profile, lock_data):
    return AuthenticatedSelectionBinding(
        cohort_id="memorysplit-confirmatory-v3-360m-n10-aws",
        amendment_sha256="1" * 64,
        selection_sha256="2" * 64,
        selection_version_id="selection-version-1",
        profile_id=profile.profile_id,
        provider=profile.provider,
        profile_sha256=profile.sha256,
        runtime_lock_sha256=hashlib.sha256(lock_data).hexdigest(),
        qualification_evidence_sha256="3" * 64,
        environment_receipt_sha256="4" * 64,
        canary_receipt_sha256="5" * 64,
        approval_receipt_sha256="6" * 64,
        approval_public_key_sha256="7" * 64,
        account_id=ACCOUNT_ID,
        instance_id=INSTANCE_ID,
        boot_id=BOOT_ID,
        region="us-east-1",
        availability_zone="us-east-1d",
        purchase_model="on_demand",
        seed=0,
        arm="dense",
    )


def _attestation(profile, lock_data):
    return {
        "schema_version": 1,
        "evidence_type": "memorysplit-aws-gpu-attestation-v1",
        "profile_id": profile.profile_id,
        "provider": profile.provider,
        "profile_sha256": profile.sha256,
        "runtime_lock_sha256": hashlib.sha256(lock_data).hexdigest(),
        "control_bundle_sha256": "d" * 64,
        "source_commit": "b" * 40,
        "source_tree": "c" * 40,
        "container_image": IMAGE,
        "container_image_digest": IMAGE_DIGEST,
        "ami_id": AMI_ID,
        "ami_owner_id": AMI_OWNER_ID,
        "instance_type": profile.instance_type,
        "gpu_model": profile.gpu_model,
        "gpu_count": profile.allocated_gpus,
        "host_facts": dict(HOST_FACTS),
        "container_facts": dict(CONTAINER_FACTS),
        "aws_instance_identity_document": {
            "accountId": ACCOUNT_ID,
            "architecture": "x86_64",
            "imageId": AMI_ID,
            "instanceId": INSTANCE_ID,
            "instanceType": profile.instance_type,
            "privateIp": "10.23.45.67",
            "region": "us-east-1",
        },
        "aws_instance_identity_pkcs7": "c3ludGhldGlj",
        "account_id": ACCOUNT_ID,
        "instance_id": INSTANCE_ID,
        "region": "us-east-1",
        "boot_id": BOOT_ID,
    }


def _hardware(profile):
    return {
        "account_id": ACCOUNT_ID,
        "instance_id": INSTANCE_ID,
        "boot_id": BOOT_ID,
        "ami_id": AMI_ID,
        "instance_type": profile.instance_type,
        "architecture": profile.architecture,
        "vcpus": profile.vcpus,
        "memory_gib": profile.memory_gib,
        "gpu_names": [profile.gpu_model] * profile.allocated_gpus,
        "instance_store": [
            {
                "path": f"/dev/nvme{index}n1",
                "model": profile.instance_store_model,
                "bytes": profile.instance_store_device_bytes,
            }
            for index in range(profile.instance_store_devices)
        ],
    }


def test_selected_environment_receipt_binds_authenticated_p6_authority():
    from cluster.aws.p5.attest_environment import (
        build_selected_environment_receipt,
    )

    profile = load_aws_gpu_profile(P6_PROFILE)
    lock_data, sbom_data = _runtime_inputs(profile)
    binding = _binding(profile, lock_data)

    receipt = build_selected_environment_receipt(
        selected_profile=profile,
        selection_binding=binding,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        attestation_evidence=_attestation(profile, lock_data),
    )

    assert receipt["receipt_type"] == (
        "memorysplit-aws-gpu-environment-selection-v1"
    )
    assert receipt["hardware_amendment_sha256"] == binding.amendment_sha256
    assert receipt["provider_selection_sha256"] == binding.selection_sha256
    assert (
        receipt["provider_selection_version_id"]
        == binding.selection_version_id
    )
    assert receipt["profile_id"] == "aws-p6-b300.48xlarge-v3"
    assert receipt["runtime_lock_sha256"] == binding.runtime_lock_sha256
    assert receipt["runtime_sbom_sha256"] == hashlib.sha256(sbom_data).hexdigest()
    assert receipt["instance_id"] == binding.instance_id
    assert receipt["ami_id"] == AMI_ID
    assert receipt["container_image"] == IMAGE
    assert receipt["host_facts"] == HOST_FACTS
    assert receipt["container_facts"] == CONTAINER_FACTS

    with pytest.raises(ValueError, match="profile|selection|binding"):
        build_selected_environment_receipt(
            selected_profile=profile,
            selection_binding=replace(
                binding,
                profile_id="aws-p5.48xlarge-v3",
            ),
            runtime_lock_data=lock_data,
            runtime_sbom_data=sbom_data,
            attestation_evidence=_attestation(profile, lock_data),
        )


def test_authenticated_environment_attestation_orchestrates_selected_measurement(
    tmp_path,
    monkeypatch,
):
    import cluster.aws.p5.attest_environment as module

    profile = load_aws_gpu_profile(P6_PROFILE)
    lock_data, sbom_data = _runtime_inputs(profile)
    binding = _binding(profile, lock_data)
    lock = tmp_path / "runtime-lock.json"
    sbom = tmp_path / "runtime-sbom.json"
    control = tmp_path / "control.zip"
    lock.write_bytes(lock_data)
    sbom.write_bytes(sbom_data)
    control.write_bytes(b"control")
    calls = []

    def attest(**kwargs):
        calls.append(kwargs)
        return _attestation(profile, lock_data)

    monkeypatch.setattr(module, "attest_selected_gpu_environment", attest)
    output = tmp_path / "selected-environment.json"
    receipt = module.attest_authenticated_gpu_environment(
        selected_profile=profile,
        selection_binding=binding,
        runtime_lock_path=lock,
        runtime_sbom_path=sbom,
        control_bundle_path=control,
        output_path=output,
        apply=False,
        imds_reader=object(),
        command_reader=object(),
        boot_id_path=tmp_path / "boot-id",
    )

    assert receipt["provider_selection_sha256"] == binding.selection_sha256
    assert len(calls) == 1
    assert calls[0]["selected_profile"] is profile
    assert calls[0]["apply"] is False
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instance_type", "p5.48xlarge"),
        ("vcpus", 191),
        ("memory_gib", 2048),
        ("gpu_names", ["NVIDIA B200"] * 8),
        ("gpu_names", ["NVIDIA B300"] * 7),
        ("instance_store", []),
    ],
)
def test_selected_bootstrap_requires_exact_profile_hardware(field, value):
    from cluster.aws.p5.attest_environment import (
        build_selected_environment_receipt,
    )
    from cluster.aws.p5.bootstrap import build_selected_bootstrap_receipt

    profile = load_aws_gpu_profile(P6_PROFILE)
    lock_data, sbom_data = _runtime_inputs(profile)
    binding = _binding(profile, lock_data)
    environment = build_selected_environment_receipt(
        selected_profile=profile,
        selection_binding=binding,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        attestation_evidence=_attestation(profile, lock_data),
    )
    hardware = _hardware(profile)
    hardware[field] = value

    with pytest.raises(ValueError, match="hardware|profile|GPU|memory|vCPU|NVMe"):
        build_selected_bootstrap_receipt(
            selected_profile=profile,
            selection_binding=binding,
            runtime_lock_data=lock_data,
            runtime_sbom_data=sbom_data,
            environment_receipt=environment,
            hardware_evidence=hardware,
        )


def test_selected_bootstrap_receipt_binds_profile_runtime_and_selection():
    from cluster.aws.p5.attest_environment import (
        build_selected_environment_receipt,
    )
    from cluster.aws.p5.bootstrap import build_selected_bootstrap_receipt

    profile = load_aws_gpu_profile(P6_PROFILE)
    lock_data, sbom_data = _runtime_inputs(profile)
    binding = _binding(profile, lock_data)
    environment = build_selected_environment_receipt(
        selected_profile=profile,
        selection_binding=binding,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        attestation_evidence=_attestation(profile, lock_data),
    )

    receipt = build_selected_bootstrap_receipt(
        selected_profile=profile,
        selection_binding=binding,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        environment_receipt=environment,
        hardware_evidence=_hardware(profile),
    )

    assert receipt["receipt_type"] == "memorysplit-aws-gpu-bootstrap-v1"
    assert receipt["provider"] == profile.provider
    assert receipt["profile_sha256"] == profile.sha256
    assert receipt["runtime_lock_sha256"] == binding.runtime_lock_sha256
    assert receipt["runtime_sbom_sha256"] == hashlib.sha256(sbom_data).hexdigest()
    assert receipt["environment_receipt_sha256"] == hashlib.sha256(
        _canonical(environment)
    ).hexdigest()
    assert receipt["provider_selection_sha256"] == binding.selection_sha256
    assert receipt["provider_selection_version_id"] == (
        binding.selection_version_id
    )
    assert receipt["hardware"]["vcpus"] == 192
    assert receipt["hardware"]["memory_gib"] == 4096
    assert receipt["hardware"]["gpu_count"] == 8
    assert receipt["hardware"]["instance_store_devices"] == 8


def test_selected_environment_and_bootstrap_receipts_are_closed_and_reparsed():
    from cluster.aws.qualification import (
        canonical_selected_bootstrap_receipt,
        canonical_selected_environment_receipt,
        parse_selected_bootstrap_receipt_bytes,
        parse_selected_environment_receipt_bytes,
    )
    from cluster.aws.p5.attest_environment import (
        build_selected_environment_receipt,
    )
    from cluster.aws.p5.bootstrap import build_selected_bootstrap_receipt

    profile = load_aws_gpu_profile(P6_PROFILE)
    lock_data, sbom_data = _runtime_inputs(profile)
    binding = _binding(profile, lock_data)
    environment = build_selected_environment_receipt(
        selected_profile=profile,
        selection_binding=binding,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        attestation_evidence=_attestation(profile, lock_data),
    )
    environment_data = canonical_selected_environment_receipt(environment)
    assert parse_selected_environment_receipt_bytes(
        environment_data,
        selected_profile=profile,
        selection_binding=binding,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
    ) == environment
    bootstrap = build_selected_bootstrap_receipt(
        selected_profile=profile,
        selection_binding=binding,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        environment_receipt=environment,
        hardware_evidence=_hardware(profile),
    )
    bootstrap_data = canonical_selected_bootstrap_receipt(bootstrap)
    assert parse_selected_bootstrap_receipt_bytes(
        bootstrap_data,
        selected_profile=profile,
        selection_binding=binding,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        environment_receipt_sha256=hashlib.sha256(environment_data).hexdigest(),
    ) == bootstrap

    open_environment = dict(environment)
    open_environment["unexpected"] = True
    with pytest.raises(ValueError, match="field|schema|closed"):
        canonical_selected_environment_receipt(open_environment)
    forged_bootstrap = json.loads(bootstrap_data)
    forged_bootstrap["hardware"]["memory_gib"] = 2048
    with pytest.raises(ValueError, match="hardware|profile|memory"):
        parse_selected_bootstrap_receipt_bytes(
            _canonical(forged_bootstrap),
            selected_profile=profile,
            selection_binding=binding,
            runtime_lock_data=lock_data,
            runtime_sbom_data=sbom_data,
            environment_receipt_sha256=hashlib.sha256(
                environment_data
            ).hexdigest(),
        )


def test_authenticated_bootstrap_uses_injected_profile_hardware_reader():
    import cluster.aws.p5.bootstrap as module

    profile = load_aws_gpu_profile(P6_PROFILE)
    lock_data, sbom_data = _runtime_inputs(profile)
    binding = _binding(profile, lock_data)
    environment = module.build_selected_environment_receipt(
        selected_profile=profile,
        selection_binding=binding,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        attestation_evidence=_attestation(profile, lock_data),
    )

    class Reader:
        def __init__(self):
            self.calls = []

        def measure(self, *, selected_profile, selection_binding):
            self.calls.append((selected_profile, selection_binding))
            return _hardware(selected_profile)

    reader = Reader()
    receipt = module.bootstrap_authenticated_gpu_environment(
        selected_profile=profile,
        selection_binding=binding,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        environment_receipt=environment,
        hardware_reader=reader,
    )

    assert reader.calls == [(profile, binding)]
    assert receipt["provider_selection_sha256"] == binding.selection_sha256
    assert receipt["hardware"]["instance_type"] == "p6-b300.48xlarge"


def test_p5_and_p6_authenticated_bindings_cannot_cross_authorize():
    from cluster.aws.qualification import validate_authenticated_selection

    p5 = load_aws_gpu_profile(P5_PROFILE)
    p6 = load_aws_gpu_profile(P6_PROFILE)
    p5_lock, _ = _runtime_inputs(p5)
    p6_lock, _ = _runtime_inputs(p6)

    with pytest.raises(ValueError, match="profile|selection|binding"):
        validate_authenticated_selection(p6, _binding(p5, p5_lock))
    with pytest.raises(ValueError, match="profile|selection|binding"):
        validate_authenticated_selection(p5, _binding(p6, p6_lock))


def _selected_case(tmp_path):
    from cluster.aws.p5.attest_environment import (
        build_selected_environment_receipt,
    )
    from cluster.aws.p5.bootstrap import build_selected_bootstrap_receipt
    from cluster.aws.p5.canary import build_selected_canary_plan

    profile = load_aws_gpu_profile(P6_PROFILE)
    lock_data, sbom_data = _runtime_inputs(profile)
    binding = _binding(profile, lock_data)
    environment = build_selected_environment_receipt(
        selected_profile=profile,
        selection_binding=binding,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        attestation_evidence=_attestation(profile, lock_data),
    )
    bootstrap = build_selected_bootstrap_receipt(
        selected_profile=profile,
        selection_binding=binding,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        environment_receipt=environment,
        hardware_evidence=_hardware(profile),
    )
    scratch = tmp_path / "selected-canary"
    scratch.mkdir(mode=0o700)
    plan = build_selected_canary_plan(
        selected_profile=profile,
        selection_binding=binding,
        runtime_lock_data=lock_data,
        runtime_sbom_data=sbom_data,
        environment_receipt=environment,
        bootstrap_receipt=bootstrap,
        scratch_root=scratch,
        output_path=scratch / "qualification.json",
        s3_root="s3://memorysplit-prod/confirmatory-v3",
    )
    return {
        "profile": profile,
        "lock_data": lock_data,
        "sbom_data": sbom_data,
        "binding": binding,
        "environment": environment,
        "bootstrap": bootstrap,
        "scratch": scratch,
        "plan": plan,
    }


def _canary_hardware(profile):
    return {
        **_hardware(profile),
        "gpu_devices": [
            {
                "index": index,
                "name": profile.gpu_model,
                "memory_mib": 294_912,
            }
            for index in range(8)
        ],
        "host_facts": dict(HOST_FACTS),
        "topology": {
            "matrix_sha256": "8" * 64,
            "nvlink_active": True,
            "fully_connected": True,
        },
    }


def _container_capabilities():
    return {
        "versions": dict(CONTAINER_FACTS),
        "bf16": True,
        "sdpa_forward": True,
        "sdpa_backward": True,
        "torch_compile": True,
        "fused_adamw": True,
        "one_step_train": True,
        "checkpoint_resume_exact": True,
        "checkpoint_sha256": "9" * 64,
        "resumed_checkpoint_sha256": "9" * 64,
    }


def _group_result(plan, arm):
    group = plan.gpu_groups[arm]
    affinity = plan.cpu_affinities[arm]
    throughput = 100_000.0 if arm == "dense" else 80_000.0
    checkpoint = "a" * 64 if arm == "dense" else "b" * 64
    return {
        "arm": arm,
        "world_size": 4,
        "gpu_ids": list(group),
        "cpu_affinity": list(affinity),
        "master_port": plan.master_ports[arm],
        "all_reduce_sum": 10.0,
        "all_reduce_latency_seconds": 0.004,
        "updates": 100,
        "warmup_updates": 10,
        "median_tok_s": throughput,
        "peak_memory_bytes": (
            170_000_000_000 if arm == "dense" else 150_000_000_000
        ),
        "one_step_train": True,
        "checkpoint_resume_exact": True,
        "checkpoint_sha256": checkpoint,
        "resumed_checkpoint_sha256": checkpoint,
    }


class _QualificationRunner:
    def __init__(self, module, plan, *, mutation=None):
        self.module = module
        self.plan = plan
        self.mutation = mutation
        self.calls = []
        self.pairs = []

    def run(self, spec):
        self.calls.append(spec)
        if spec.name == "hardware-inventory":
            value = _canary_hardware(self.plan.profile)
            if self.mutation == "host-floor":
                value["host_facts"]["efa"] = "1.43.9"
            elif self.mutation == "topology":
                value["topology"]["nvlink_active"] = False
            return self.module.QualificationCommandResult(
                0,
                _canonical(value).decode("ascii"),
                "",
            )
        if spec.name == "container-capabilities":
            value = _container_capabilities()
            if self.mutation == "sdpa":
                value["sdpa_backward"] = False
            elif self.mutation == "container-version":
                value["versions"]["cuda"] = "12.9"
            return self.module.QualificationCommandResult(
                0,
                _canonical(value).decode("ascii"),
                "",
            )
        raise AssertionError(f"unexpected command: {spec.name}")

    def run_pair(self, specs):
        self.pairs.append(tuple(specs))
        results = {}
        for spec in specs:
            arm = spec.arm
            value = _group_result(self.plan, arm)
            if self.mutation == "world-size" and arm == "dense":
                value["world_size"] = 8
            elif self.mutation == "gpu-overlap" and arm == "split90":
                value["gpu_ids"] = [0, 1, 2, 3]
            elif self.mutation == "checkpoint" and arm == "split90":
                value["resumed_checkpoint_sha256"] = "0" * 64
            results[spec.name] = self.module.QualificationCommandResult(
                0,
                _canonical(value).decode("ascii"),
                "",
            )
        return results


class _QualificationStore:
    def __init__(self, module):
        self.module = module
        self.objects = {}

    def put(self, uri, payload, *, sha256):
        self.objects[("version-1", uri)] = bytes(payload)
        return self.module.QualificationObjectWrite(
            sha256=sha256,
            bytes=len(payload),
            version_id="version-1",
        )

    def get(self, uri, *, version_id):
        payload = self.objects[(version_id, uri)]
        return self.module.QualificationObjectRead(
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            version_id=version_id,
        )


class _QualificationTime:
    def __init__(self):
        self.origin = datetime(2026, 7, 24, 8, 0, tzinfo=UTC)
        self.tick = 0

    def now(self):
        value = self.origin + timedelta(seconds=self.tick)
        self.tick += 1
        return value

    def monotonic(self):
        value = float(self.tick)
        self.tick += 1
        return value


def test_selected_canary_renders_two_independent_profile_driven_4_rank_groups(
    tmp_path,
):
    from cluster.aws.p5.canary import render_selected_canary_plan

    case = _selected_case(tmp_path)
    plan = case["plan"]
    rendered = render_selected_canary_plan(plan)

    assert plan.gpu_groups == {
        "dense": (0, 1, 2, 3),
        "split90": (4, 5, 6, 7),
    }
    assert plan.cpu_affinities == {
        "dense": (0, 95),
        "split90": (96, 191),
    }
    hardware_argv = rendered["commands"]["hardware"][0]["argv"]
    assert hardware_argv[hardware_argv.index("--region") + 1] == (
        case["binding"].region
    )
    groups = rendered["commands"]["nccl_4x4"]
    assert len(groups) == 2
    assert {row["arm"] for row in groups} == {"dense", "split90"}
    assert all(row["world_size"] == 4 for row in groups)
    assert all(row["concurrent_group"] == "paired-4rank" for row in groups)
    assert groups[0]["master_port"] != groups[1]["master_port"]
    assert not any(
        "--nproc_per_node=8" in argument
        for row in groups
        for argument in row["argv"]
    )
    assert all(
        "--nproc_per_node=4" in row["argv"]
        and IMAGE in row["argv"]
        and "--pull" in row["argv"]
        and "never" in row["argv"]
        for row in groups
    )
    for row in groups:
        arm = row["arm"]
        assert row["argv"][row["argv"].index("--gpu-ids") + 1] == ",".join(
            str(item) for item in plan.gpu_groups[arm]
        )
        assert row["argv"][row["argv"].index("--cpu-affinity") + 1] == (
            f"{plan.cpu_affinities[arm][0]}-{plan.cpu_affinities[arm][1]}"
        )
        assert row["argv"][row["argv"].index("--master-port") + 1] == str(
            plan.master_ports[arm]
        )
    assert rendered["bindings"]["provider_selection_sha256"] == (
        case["binding"].selection_sha256
    )
    assert rendered["bindings"]["provider_selection_version_id"] == (
        "selection-version-1"
    )


def test_selected_canary_executes_measured_p6_qualification_and_eta(tmp_path):
    import cluster.aws.p5.canary as module

    case = _selected_case(tmp_path)
    plan = case["plan"]
    runner = _QualificationRunner(module, plan)
    store = _QualificationStore(module)

    receipt = module.execute_selected_canary(
        plan,
        command_runner=runner,
        object_store=store,
        time_reader=_QualificationTime(),
        apply=False,
    )

    assert len(runner.pairs) == 1
    assert {spec.name for spec in runner.pairs[0]} == {
        "dense-4rank",
        "split90-4rank",
    }
    assert receipt["receipt_type"] == "memorysplit-aws-gpu-qualification-v2"
    assert receipt["profile_id"] == "aws-p6-b300.48xlarge-v3"
    assert receipt["provider_selection_sha256"] == (
        case["binding"].selection_sha256
    )
    assert receipt["provider_selection_version_id"] == "selection-version-1"
    assert receipt["runtime_lock_sha256"] == case["binding"].runtime_lock_sha256
    assert receipt["runtime_sbom_sha256"] == hashlib.sha256(
        case["sbom_data"]
    ).hexdigest()
    assert receipt["environment_receipt_sha256"] == hashlib.sha256(
        _canonical(case["environment"])
    ).hexdigest()
    assert receipt["bootstrap_receipt_sha256"] == hashlib.sha256(
        _canonical(case["bootstrap"])
    ).hexdigest()
    assert receipt["hardware"]["vcpus"] == 192
    assert receipt["hardware"]["memory_gib"] == 4096
    assert receipt["hardware"]["topology"]["nvlink_active"] is True
    assert receipt["container"]["versions"] == CONTAINER_FACTS
    assert all(
        receipt["container"][field] is True
        for field in (
            "bf16",
            "sdpa_forward",
            "sdpa_backward",
            "torch_compile",
            "fused_adamw",
            "one_step_train",
            "checkpoint_resume_exact",
        )
    )
    assert {
        row["arm"]: row["world_size"] for row in receipt["nccl_groups"]
    } == {"dense": 4, "split90": 4}
    assert receipt["telemetry"]["per_arm"]["dense"]["median_tok_s"] == 100_000.0
    assert receipt["telemetry"]["per_arm"]["split90"]["peak_memory_bytes"] == (
        150_000_000_000
    )
    expected_pair_eta = 7_120_879_616 / 80_000.0
    assert receipt["telemetry"]["projected_pair_eta_seconds"] == pytest.approx(
        expected_pair_eta
    )
    assert receipt["telemetry"]["projected_cohort_eta_seconds"] == pytest.approx(
        expected_pair_eta * 10
    )
    assert [row["name"] for row in receipt["phases"]] == list(
        module.SELECTED_PHASE_ORDER
    )
    assert all(
        len(row["evidence_sha256"]) == 64 for row in receipt["phases"]
    )
    payload = module.canonical_selected_qualification_receipt(receipt)
    assert module.parse_selected_qualification_receipt_bytes(
        payload,
        plan=plan,
    ) == receipt
    assert not plan.output_path.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "host-floor",
        "topology",
        "sdpa",
        "container-version",
        "world-size",
        "gpu-overlap",
        "checkpoint",
    ],
)
def test_selected_canary_fails_closed_on_measured_phase_drift(tmp_path, mutation):
    import cluster.aws.p5.canary as module

    case = _selected_case(tmp_path)
    plan = case["plan"]

    with pytest.raises(
        ValueError,
        match=(
            "P6|profile|topology|NVLink|container|capability|NCCL|group|"
            "checkpoint|floor|GPU"
        ),
    ):
        module.execute_selected_canary(
            plan,
            command_runner=_QualificationRunner(
                module,
                plan,
                mutation=mutation,
            ),
            object_store=_QualificationStore(module),
            time_reader=_QualificationTime(),
            apply=False,
        )


def test_selected_qualification_receipt_cannot_cross_authorize_profiles(tmp_path):
    import cluster.aws.p5.canary as module

    case = _selected_case(tmp_path)
    p6_plan = case["plan"]
    receipt = module.execute_selected_canary(
        p6_plan,
        command_runner=_QualificationRunner(module, p6_plan),
        object_store=_QualificationStore(module),
        time_reader=_QualificationTime(),
        apply=False,
    )
    payload = module.canonical_selected_qualification_receipt(receipt)
    p5 = load_aws_gpu_profile(P5_PROFILE)
    p5_lock, p5_sbom = _runtime_inputs(p5)
    p5_binding = _binding(p5, p5_lock)

    with pytest.raises(ValueError, match="profile|provider|selection|binding"):
        module.build_selected_canary_plan(
            selected_profile=p5,
            selection_binding=p5_binding,
            runtime_lock_data=p5_lock,
            runtime_sbom_data=p5_sbom,
            environment_receipt=case["environment"],
            bootstrap_receipt=case["bootstrap"],
            scratch_root=case["scratch"],
            output_path=case["scratch"] / "p5.json",
            s3_root="s3://memorysplit-prod/confirmatory-v3",
        )
    with pytest.raises(ValueError, match="profile|provider|selection|binding"):
        module.parse_selected_qualification_receipt_bytes(payload, plan=replace(
            p6_plan,
            profile=p5,
        ))


def test_legacy_p5_canary_contract_stays_byte_and_name_compatible():
    import cluster.aws.p5.canary as module

    assert module.RECEIPT_TYPE == "memorysplit-aws-p5-qualification-v1"
    assert module.PHASE_ORDER == (
        "hardware",
        "nccl_all_reduce",
        "functional",
        "resume",
        "throughput_4x4",
        "s3_roundtrip",
    )
    assert module.FUNCTIONAL_UPDATES == 1
    assert module.THROUGHPUT_UPDATES == 100


def test_digest_pinned_runtime_build_binds_executable_qualification_worker(
    tmp_path,
):
    build_script = ROOT / "containers" / "aws-gpu" / "build_image.py"
    spec = importlib.util.spec_from_file_location(
        f"p6_runtime_build_{uuid.uuid4().hex}",
        build_script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class Repository:
        def inspect(self, root):
            return {
                "source_commit": "b" * 40,
                "source_tree": "c" * 40,
                "clean": True,
                "command_transcript_sha256": {
                    "head": "1" * 64,
                    "tree": "2" * 64,
                    "status": "3" * 64,
                },
            }

    plan = module.render_build_plan(
        repository_uri=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/"
            "memorysplit/aws-gpu"
        ),
        source_commit="b" * 40,
        repository_root=ROOT,
        docker_config=tmp_path / "docker",
        repository_reader=Repository(),
    )

    assert QUALIFICATION_WORKER.is_file()
    assert plan["inputs"]["qualification_worker_sha256"] == hashlib.sha256(
        QUALIFICATION_WORKER.read_bytes()
    ).hexdigest()
    dockerfile = (
        ROOT / "containers" / "aws-gpu" / "Dockerfile"
    ).read_text(encoding="utf-8")
    dockerignore = (
        ROOT / "containers" / "aws-gpu" / "Dockerfile.dockerignore"
    ).read_text(encoding="utf-8")
    assert (
        "COPY cluster/aws/qualification_worker.py "
        "/opt/memorysplit/cluster/aws/qualification_worker.py"
    ) in dockerfile
    assert "!cluster/aws/qualification_worker.py" in dockerignore

    source = QUALIFICATION_WORKER.read_text(encoding="utf-8")
    for required in (
        "torch.bfloat16",
        "scaled_dot_product_attention",
        "torch.compile",
        "fused=True",
        "init_process_group(\"nccl\")",
        "checkpoint_resume_exact",
        "peak_memory_bytes",
    ):
        assert required in source
