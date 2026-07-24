from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cluster.aws.p5.canary import (
    PRODUCTION_UPDATES_PER_ARM,
    SEED_PAIRS,
    TOKENS_PER_UPDATE,
    QualificationError,
    compute_concurrent_throughput,
    render_canary_command_plan,
    validate_qualification_receipt,
)
from cluster.aws.p5.profile import (
    load_aws_gpu_profile,
    validate_runtime_environment,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = ROOT / "cluster" / "profiles"


def _profile(name: str):
    return load_aws_gpu_profile(PROFILE_ROOT / f"{name}.json")


def _receipt(profile, *, dense_seconds=2.0, split_seconds=4.0):
    gpu_name = (
        "NVIDIA B300"
        if profile.instance_type == "p6-b300.48xlarge"
        else "NVIDIA H100 80GB HBM3"
    )
    software = {
        field: minimum
        for field, minimum in profile.software_minimums.items()
    }
    gpu_ids = list(range(profile.allocated_gpus))
    gpu_groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    return {
        "schema_version": 2,
        "receipt_type": "aws-gpu-qualification",
        "profile": {
            "profile_id": profile.profile_id,
            "provider": profile.provider,
            "profile_sha256": profile.sha256,
        },
        "hardware": {
            "instance_type": profile.instance_type,
            "vcpus": profile.vcpus,
            "memory_gib": profile.memory_gib,
            "gpu_names": [gpu_name] * 8,
            "gres": profile.gres,
            "cpu_affinity_halves": [
                list(group) for group in profile.cpu_affinity_halves
            ],
            "train_groups": [4, 4],
        },
        "software": software,
        "capabilities": {
            "fabric_manager_nvlink": {
                "passed": True,
                "fabric_manager_active": True,
                "nvlink_connected": True,
                "gpu_ids": gpu_ids,
            },
            "bf16": {
                "passed": True,
                "supported": True,
                "dtype": "bfloat16",
                "gpu_ids": gpu_ids,
            },
            "sdpa": {
                "passed": True,
                "forward": True,
                "backward": True,
                "dtype": "bfloat16",
                "gpu_ids": gpu_ids,
            },
            "torch_compile": {
                "passed": True,
                "compiled": True,
                "backend": "inductor",
                "gpu_ids": gpu_ids,
            },
            "fused_adamw": {
                "passed": True,
                "fused": True,
                "gpu_ids": gpu_ids,
            },
            "simultaneous_4_plus_4_nccl": {
                "passed": True,
                "concurrent": True,
                "backend": "nccl",
                "groups": gpu_groups,
                "world_sizes": [4, 4],
            },
            "one_step_training": {
                "passed": True,
                "arms": ["dense", "split90"],
                "updates": 1,
                "gpu_groups": gpu_groups,
            },
            "checkpoint_resume": {
                "passed": True,
                "checkpointed": True,
                "resumed": True,
                "arms": ["dense", "split90"],
                "gpu_groups": gpu_groups,
            },
            "nvme_geometry": {
                "passed": True,
                "model": profile.instance_store_model,
                "devices": profile.instance_store_devices,
                "device_bytes": profile.instance_store_device_bytes,
                "raid_level": profile.raid_level,
            },
        },
        "throughput": {
            "concurrent": True,
            "updates": 100,
            "warmup_updates": 10,
            "tokens_per_update": TOKENS_PER_UPDATE,
            "arms": [
                {
                    "arm": "dense",
                    "gpu_ids": [0, 1, 2, 3],
                    "update_seconds": [999.0] * 10
                    + [dense_seconds] * 90,
                },
                {
                    "arm": "split90",
                    "gpu_ids": [4, 5, 6, 7],
                    "update_seconds": [999.0] * 10
                    + [split_seconds] * 90,
                },
            ],
        },
    }


def _v3_receipt(profile):
    receipt = _receipt(profile)
    digest = "sha256:" + "d" * 64
    receipt["schema_version"] = 3
    receipt["provenance"] = {
        "instance_id": "i-0123456789abcdef0",
        "boot_id": "12345678-1234-4234-9234-123456789abc",
        "region": "us-west-2",
        "ami_id": "ami-0123456789abcdef0",
        "ami_owner_id": "099720109477",
        "release_sha256": "e" * 64,
        "container_image": (
            "123456789012.dkr.ecr.us-west-2.amazonaws.com/"
            f"memorysplit/aws-gpu@{digest}"
        ),
        "container_digest": digest,
        "provider_selection_sha256": "1" * 64,
        "environment_receipt_sha256": "2" * 64,
        "qualified_at": "2026-07-24T01:00:00Z",
        "command_plan_sha256": "3" * 64,
    }
    receipt["raw_output_sha256"] = {
        "device_topology_software": "4" * 64,
        "bf16": "5" * 64,
        "sdpa": "6" * 64,
        "torch_compile": "7" * 64,
        "fused_adamw": "8" * 64,
        "simultaneous_4_plus_4_nccl": "9" * 64,
        "one_step_training": "a" * 64,
        "checkpoint_resume": "b" * 64,
        "nvme": "c" * 64,
        "throughput": "d" * 64,
    }
    return receipt


def test_v3_qualification_requires_complete_cross_runtime_provenance():
    profile = _profile("aws-p5.48xlarge-v3")
    receipt = _v3_receipt(profile)
    provenance = receipt["provenance"]

    report = validate_qualification_receipt(
        receipt,
        profile,
        expected_instance_id=provenance["instance_id"],
        expected_boot_id=provenance["boot_id"],
        expected_provider_selection_sha256=provenance[
            "provider_selection_sha256"
        ],
        expected_environment_receipt_sha256=provenance[
            "environment_receipt_sha256"
        ],
        expected_region=provenance["region"],
        expected_ami_id=provenance["ami_id"],
        expected_ami_owner_id=provenance["ami_owner_id"],
        expected_release_sha256=provenance["release_sha256"],
        expected_container_image=provenance["container_image"],
        expected_container_digest=provenance["container_digest"],
        expected_command_plan_sha256=provenance["command_plan_sha256"],
    )
    assert report.provenance == provenance

    missing = copy.deepcopy(receipt)
    del missing["provenance"]["boot_id"]
    with pytest.raises(QualificationError, match="missing"):
        validate_qualification_receipt(missing, profile)

    with pytest.raises(QualificationError, match="stale or cross-runtime"):
        validate_qualification_receipt(
            receipt,
            profile,
            expected_boot_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        )
    with pytest.raises(QualificationError, match="stale or cross-runtime"):
        validate_qualification_receipt(
            receipt,
            profile,
            expected_release_sha256="f" * 64,
        )

    unknown_raw = copy.deepcopy(receipt)
    unknown_raw["raw_output_sha256"]["self_authored_pass"] = "e" * 64
    with pytest.raises(QualificationError, match="unknown"):
        validate_qualification_receipt(unknown_raw, profile)


@pytest.mark.parametrize(
    "profile_name",
    [
        "aws-p5.48xlarge",
        "aws-p5.48xlarge-v3",
        "aws-p6-b300.48xlarge-v3",
    ],
)
def test_qualification_supports_each_closed_p5_and_p6_profile(profile_name):
    profile = _profile(profile_name)
    report = validate_qualification_receipt(_receipt(profile), profile)

    assert report.profile_id == profile_name
    assert report.throughput.measured_updates_per_arm == 90
    assert report.throughput.warmup_updates_discarded == 10
    assert report.throughput.dense_seconds_per_update == pytest.approx(2.0)
    assert report.throughput.split90_seconds_per_update == pytest.approx(4.0)
    assert report.throughput.concurrent_tokens_per_second == pytest.approx(
        2 * TOKENS_PER_UPDATE / 4.0
    )
    assert report.seed_pairs == 10
    assert report.eta_seconds == pytest.approx(
        4.0 * PRODUCTION_UPDATES_PER_ARM * SEED_PAIRS
    )
    assert report.as_dict()["eta"]["hours"] > 0


def test_throughput_discards_exactly_first_ten_warmup_updates():
    throughput = compute_concurrent_throughput(
        [1_000_000.0] * 10 + [1.0] * 90,
        [1_000_000.0] * 10 + [2.0] * 90,
    )

    assert throughput.dense_seconds_per_update == pytest.approx(1.0)
    assert throughput.split90_seconds_per_update == pytest.approx(2.0)


def test_qualification_rejects_cross_profile_receipt():
    p5 = _profile("aws-p5.48xlarge-v3")
    p6 = _profile("aws-p6-b300.48xlarge-v3")

    with pytest.raises(QualificationError, match="different profile"):
        validate_qualification_receipt(_receipt(p5), p6)


@pytest.mark.parametrize("metric", [0.0, float("nan"), float("inf")])
def test_qualification_rejects_zero_and_nonfinite_metrics(metric):
    profile = _profile("aws-p6-b300.48xlarge-v3")
    receipt = _receipt(profile)
    receipt["throughput"]["arms"][0]["update_seconds"][50] = metric

    with pytest.raises(QualificationError, match="finite|greater than zero"):
        validate_qualification_receipt(receipt, profile)


def test_qualification_rejects_missing_metric_field():
    profile = _profile("aws-p6-b300.48xlarge-v3")
    receipt = _receipt(profile)
    del receipt["throughput"]["arms"][0]["update_seconds"]

    with pytest.raises(QualificationError, match="missing"):
        validate_qualification_receipt(receipt, profile)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda receipt: receipt["hardware"].update(train_groups=[8, 0]),
        lambda receipt: receipt["throughput"]["arms"][0].update(
            gpu_ids=[0, 1, 2]
        ),
        lambda receipt: receipt["throughput"].update(concurrent=False),
        lambda receipt: receipt["throughput"].update(updates=99),
    ],
)
def test_qualification_rejects_wrong_concurrent_4_plus_4_geometry(mutate):
    profile = _profile("aws-p6-b300.48xlarge-v3")
    receipt = _receipt(profile)
    mutate(receipt)

    with pytest.raises(QualificationError):
        validate_qualification_receipt(receipt, profile)


def test_qualification_rejects_mixed_matching_gpu_hardware_names():
    profile = _profile("aws-p5.48xlarge-v3")
    receipt = _receipt(profile)
    receipt["hardware"]["gpu_names"][0] = "NVIDIA H100 80GB"

    with pytest.raises(QualificationError, match="mixes GPU hardware"):
        validate_qualification_receipt(receipt, profile)


@pytest.mark.parametrize(
    ("field", "below"),
    [
        ("cuda", "12.9"),
        ("driver", "579.99"),
        ("linux_kernel", "5.15"),
        ("efa", "1.43.9"),
        ("ofi_nccl", "1.17.0"),
    ],
)
def test_p6_qualification_rejects_runtime_below_required_floor(field, below):
    profile = _profile("aws-p6-b300.48xlarge-v3")
    receipt = _receipt(profile)
    receipt["software"][field] = below

    with pytest.raises(QualificationError, match="below required minimum"):
        validate_qualification_receipt(receipt, profile)


@pytest.mark.parametrize(
    "capability",
    [
        "fabric_manager_nvlink",
        "bf16",
        "sdpa",
        "torch_compile",
        "fused_adamw",
        "simultaneous_4_plus_4_nccl",
        "one_step_training",
        "checkpoint_resume",
        "nvme_geometry",
    ],
)
def test_qualification_requires_every_capability(capability):
    profile = _profile("aws-p6-b300.48xlarge-v3")
    receipt = _receipt(profile)
    del receipt["capabilities"][capability]

    with pytest.raises(QualificationError, match="missing"):
        validate_qualification_receipt(receipt, profile)


@pytest.mark.parametrize(
    ("capability", "field"),
    [
        ("fabric_manager_nvlink", "fabric_manager_active"),
        ("fabric_manager_nvlink", "nvlink_connected"),
        ("bf16", "supported"),
        ("sdpa", "forward"),
        ("sdpa", "backward"),
        ("torch_compile", "compiled"),
        ("fused_adamw", "fused"),
        ("simultaneous_4_plus_4_nccl", "concurrent"),
        ("checkpoint_resume", "checkpointed"),
        ("checkpoint_resume", "resumed"),
    ],
)
def test_qualification_rejects_false_capability_evidence(capability, field):
    profile = _profile("aws-p5.48xlarge-v3")
    receipt = _receipt(profile)
    receipt["capabilities"][capability][field] = False

    with pytest.raises(QualificationError, match="profile-bound contract"):
        validate_qualification_receipt(receipt, profile)


@pytest.mark.parametrize(
    "capability",
    [
        "fabric_manager_nvlink",
        "bf16",
        "sdpa",
        "torch_compile",
        "fused_adamw",
        "simultaneous_4_plus_4_nccl",
        "one_step_training",
        "checkpoint_resume",
        "nvme_geometry",
    ],
)
def test_qualification_rejects_false_pass_and_unknown_capability_fields(
    capability,
):
    profile = _profile("aws-p5.48xlarge-v3")
    receipt = _receipt(profile)
    receipt["capabilities"][capability]["passed"] = False

    with pytest.raises(QualificationError, match="passed must be true"):
        validate_qualification_receipt(receipt, profile)

    receipt = _receipt(profile)
    receipt["capabilities"][capability]["unknown"] = True
    with pytest.raises(QualificationError, match="unknown"):
        validate_qualification_receipt(receipt, profile)


def test_qualification_requires_exact_profile_bound_nvme_geometry():
    profile = _profile("aws-p6-b300.48xlarge-v3")
    receipt = _receipt(profile)
    receipt["capabilities"]["nvme_geometry"]["device_bytes"] -= 1

    with pytest.raises(QualificationError, match="profile-bound contract"):
        validate_qualification_receipt(receipt, profile)


def _runtime(profile):
    digest = "sha256:" + "d" * 64
    return validate_runtime_environment(
        profile,
        {
            "AWS_REGION": "us-west-2",
            "MS_S3_ROOT": "s3://memorysplit-prod/qualification",
            "MS_S3_KMS_KEY_ID": (
                "arn:aws:kms:us-west-2:123456789012:"
                "key/12345678-1234-4234-9234-123456789abc"
            ),
            "MS_AWS_AMI_ID": "ami-0123456789abcdef0",
            "MS_CONTAINER_DIGEST": digest,
            "MS_CONTAINER_IMAGE": (
                "123456789012.dkr.ecr.us-west-2.amazonaws.com/"
                f"memorysplit/aws-gpu@{digest}"
            ),
            "MS_RUNTIME_UID": "10001",
            "MS_RUNTIME_GID": "10002",
        },
    )


@pytest.mark.parametrize(
    "profile_name",
    ["aws-p5.48xlarge-v3", "aws-p6-b300.48xlarge-v3"],
)
def test_canary_command_plan_is_digest_bound_and_shell_free(profile_name):
    profile = _profile(profile_name)
    runtime = _runtime(profile)
    release_sha256 = "b" * 64
    release_root = f"{profile.scratch_root}/releases/{release_sha256}"
    plan = render_canary_command_plan(
        profile,
        runtime,
        release_sha256=release_sha256,
        release_root=release_root,
    )
    image_digest = runtime.container_digest.removeprefix("sha256:")

    assert plan["provider"] == profile.provider
    assert plan["instance_type"] == profile.instance_type
    assert plan["profile_sha256"] == profile.sha256
    assert plan["release_sha256"] == release_sha256
    assert plan["release_root"] == release_root
    assert plan["container_digest"] == runtime.container_digest
    assert plan["output_root"] == (
        f"{profile.scratch_root}/qualification/{profile.profile_id}/"
        f"release-{release_sha256}/profile-{profile.sha256}/image-{image_digest}"
    )
    assert plan["gpu_groups"] == {
        "dense": [0, 1, 2, 3],
        "split90": [4, 5, 6, 7],
    }

    phases = plan["phases"]
    assert set(phases) == {
        "prepare",
        "serial",
        "one_step_training",
        "checkpoint_resume",
        "throughput",
    }
    assert phases["serial"]["concurrent"] is False
    assert phases["one_step_training"]["concurrent"] is True
    assert phases["checkpoint_resume"]["concurrent"] is False
    assert phases["throughput"]["concurrent"] is True
    commands = [
        command
        for phase in phases.values()
        for command in phase["commands"]
    ]
    assert all(
        isinstance(command, list)
        and command
        and all(isinstance(argument, str) for argument in command)
        for command in commands
    )
    assert not any(
        executable in {"/bin/sh", "/bin/bash", "sh", "bash"}
        or "-c" in command
        for command in commands
        for executable in command[:1]
    )

    assert phases["prepare"]["commands"][0][0:4] == [
        "/usr/bin/install",
        "-d",
        "-m",
        "0700",
    ]
    assert len(phases["serial"]["commands"]) == 7

    container_commands = [
        command for command in commands if command[0] == "/usr/bin/docker"
    ]
    for command in container_commands:
        assert runtime.container_image in command
        assert command[command.index("--mount") + 1] == (
            f"type=bind,src={plan['output_root']},dst=/qualification"
        )
        assert (
            f"type=bind,src={release_root},dst=/workspace,readonly"
            in command
        )
        tmpfs = command[command.index("--tmpfs") + 1]
        assert "exec" in tmpfs.split(",")
        assert "noexec" not in tmpfs.split(",")
        assert "/opt/venv/bin/python" in command
        assert "/workspace/cluster/aws/p5/canary_runtime.py" in command
        assert command[command.index("--provider") + 1] == profile.provider
        assert command[command.index("--instance-type") + 1] == (
            profile.instance_type
        )
        assert command[command.index("--profile-sha256") + 1] == (
            profile.sha256
        )
        assert command[command.index("--release-sha256") + 1] == (
            release_sha256
        )
        assert command[command.index("--release-root") + 1] == "/workspace"
        assert command[command.index("--gres") + 1] == profile.gres

    training = phases["one_step_training"]["commands"]
    assert len(training) == 2
    for command, arm, ids in zip(
        training,
        ("dense", "split90"),
        ("0,1,2,3", "4,5,6,7"),
        strict=True,
    ):
        assert command[:6] == [
            "/usr/bin/docker",
            "run",
            "--rm",
            "--read-only",
            "--network=host",
            "--ipc=host",
        ]
        assert command[command.index("--gpus") + 1] == f"device={ids}"
        assert command[command.index("--arm") + 1] == arm
        assert command[command.index("--gpu-ids") + 1] == ids
        assert command[command.index("--output") + 1] == (
            f"/qualification/receipts/one-step-training-{arm}.json"
        )

    throughput = phases["throughput"]["commands"]
    assert len(throughput) == 2
    for command in throughput:
        assert command[command.index("--updates") + 1] == "100"
        assert command[command.index("--warmup-updates") + 1] == "10"

    checkpoints = phases["checkpoint_resume"]["commands"]
    assert [
        (
            command[command.index("--arm") + 1],
            command[
                command.index("/workspace/cluster/aws/p5/canary_runtime.py")
                + 1
            ],
            command[command.index("--gpu-ids") + 1],
        )
        for command in checkpoints
    ] == [
        ("dense", "checkpoint", "0,1,2,3"),
        ("dense", "resume", "0,1,2,3"),
        ("split90", "checkpoint", "4,5,6,7"),
        ("split90", "resume", "4,5,6,7"),
    ]
    assert [
        command[command.index("--output") + 1] for command in checkpoints
    ] == [
        "/qualification/receipts/checkpoint-dense.json",
        "/qualification/resume/dense.json",
        "/qualification/receipts/checkpoint-split90.json",
        "/qualification/resume/split90.json",
    ]
