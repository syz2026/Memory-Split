from __future__ import annotations

from pathlib import Path

import pytest

from cluster.aws.p5.canary import (
    PRODUCTION_UPDATES_PER_ARM,
    SEED_PAIRS,
    TOKENS_PER_UPDATE,
    QualificationError,
    compute_concurrent_throughput,
    validate_qualification_receipt,
)
from cluster.aws.p5.profile import load_aws_gpu_profile


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
    return {
        "schema_version": 1,
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
