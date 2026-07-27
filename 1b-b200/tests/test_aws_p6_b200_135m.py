"""Exact P6-B200 admission for the protected 135M reasoning-v3 cohort.

Every test here asserts a fail-closed property: unknown, missing, inexact, or
merely regex-compatible hardware evidence must never authorize protected work.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

import cluster.aws.readiness as readiness
import cluster.aws.reasoning_v3 as corpus
from cluster.aws.readiness import (
    QUALIFICATION_GATES,
    HardwareAdmissionError,
    admit_b200_node,
    admit_pair_allocations,
    admit_protected_site,
    admit_qualification_gates,
    canonical_digest,
    load_hardware_profile,
)
from cluster.aws.reasoning_v3 import admit_reasoning_v3_site
from msctl import aws_operations
from msctl.profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "cluster" / "profiles"
B200_PROFILE = PROFILES / "aws-p6-b200.48xlarge-135m-v1.json"
LEGACY_AWS_PROFILE = PROFILES / "aws-p5-p6.example.json"
PARALLELCLUSTER = (
    ROOT / "cluster" / "aws" / "parallelcluster" / "memorysplit-v3-p5.example.yaml"
)

AUTHORITY = "memorysplit-p6-b200-135m-v1"
FIXTURE_AMI = "ami-0123456789abcdef0"
FIXTURE_OWNER = "123456789012"
FIXTURE_REPOSITORY = f"{FIXTURE_OWNER}.dkr.ecr.us-east-1.amazonaws.com/memorysplit"
FIXTURE_DIGEST = "sha256:" + "1" * 64
FIXTURE_ZONE = "us-east-1d"

# Values the shipped profile deliberately leaves unconfirmed. They are filled in
# only inside tests so the checked-in profile can never admit a live node before
# an operator resolves them against the real account.
CONFIRMATIONS: dict[str, object] = {
    "container_runtime.digest": FIXTURE_DIGEST,
    "container_runtime.repository": FIXTURE_REPOSITORY,
    "hardware.gpu_compute_capability": "10.0",
    "hardware.gpu_memory_mib": 183_359,
    "hardware.gpu_name": "NVIDIA B200",
    "image.ami_id": FIXTURE_AMI,
    "image.ami_name": "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)",
    "image.ami_owner": FIXTURE_OWNER,
    "placement.availability_zone": FIXTURE_ZONE,
    "qualification.memory_headroom_fraction_floor": 0.05,
    "software_floors.nccl_version": "2.26.2",
    "software_floors.torch_version": "2.7.0",
}

# Hardware that the broad ``H100|H200|B200`` regex cannot distinguish from a
# real B200 node.
INEXACT_GPU_NAMES = (
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H200",
    "NVIDIA B300",
    "NVIDIA GB200 NVL72",
    "not-a-B200 emulator",
)


def _set_path(document: dict, dotted: str, value: object) -> None:
    section, _, key = dotted.partition(".")
    document[section][key] = value


def _confirmed_document() -> dict:
    document = json.loads(B200_PROFILE.read_text(encoding="utf-8"))
    for dotted, value in CONFIRMATIONS.items():
        _set_path(document, dotted, value)
    document["pending_confirmation"] = []
    return document


def _write_profile(tmp_path: Path, document: dict, *, name: str = "b200.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


@pytest.fixture
def profile(tmp_path):
    """The frozen profile with every placeholder resolved to fixture values."""

    return load_hardware_profile(_write_profile(tmp_path, _confirmed_document()))


def _gpu(index: int, *, name: str = "NVIDIA B200", memory_mib: int = 183_359) -> dict:
    return {
        "compute_capability": "10.0",
        "index": index,
        "memory_mib": memory_mib,
        "name": name,
        "uuid": f"GPU-{index:08x}-1000-2000-3000-000000000000",
    }


def _attestation(**overrides) -> dict:
    value = {
        "ami_id": FIXTURE_AMI,
        "ami_owner": FIXTURE_OWNER,
        "availability_zone": FIXTURE_ZONE,
        "container_image_digest": FIXTURE_DIGEST,
        "container_image_repository": FIXTURE_REPOSITORY,
        "cuda_version": "12.8",
        "driver_version": "570.124.06",
        "efa_installer_version": "1.41.0",
        "gpus": [_gpu(index) for index in range(8)],
        "instance_id": "i-0fedcba987654321f",
        "instance_type": "p6-b200.48xlarge",
        "kernel_version": "6.8.0-1029-aws",
        "local_storage_gb": 30_400,
        "memory_gib": 2_048,
        "nccl_version": "2.26.2",
        "ofi_nccl_version": "1.15.0",
        "profile_id": "aws-p6-b200.48xlarge-135m-v1",
        "region": "us-east-1",
        "schema_version": 1,
        "torch_version": "2.7.0",
        "vcpus": 192,
    }
    value.update(overrides)
    return value


def _allocations(seeds=(0, 1, 2, 3), *, uuids=None) -> list[dict]:
    inventory = uuids or [_gpu(index)["uuid"] for index in range(8)]
    size = len(inventory)
    return [
        {
            "arms": {
                "dense": {"gpu_uuid": inventory[(2 * position) % size]},
                "split90": {"gpu_uuid": inventory[(2 * position + 1) % size]},
            },
            "pair_id": f"d135m_reasoning_v3_s{seed}",
            "seed": seed,
        }
        for position, seed in enumerate(seeds)
    ]


def _gates(attestation_sha256: str, **overrides) -> dict:
    value = {
        "checkpoint_write": {
            "bytes": 1_616_240_640,
            "checkpoint_sha256": "a" * 64,
            "evidence_sha256": "b" * 64,
            "fsync": True,
            "status": "passed",
        },
        "device_query": {
            "devices_passed": 8,
            "evidence_sha256": "c" * 64,
            "result": "PASS",
            "status": "passed",
        },
        "exact_resume_equality": {
            "continuous_state_sha256": "d" * 64,
            "evidence_sha256": "e" * 64,
            "resume_exact": True,
            "resumed_state_sha256": "d" * 64,
            "status": "passed",
            "updates": 2,
        },
        "memory_headroom": {
            "device_total_mib": 183_359,
            "evidence_sha256": "f" * 64,
            "headroom_fraction": 0.41,
            "oom_detected": False,
            "peak_allocated_mib": 108_000,
            "status": "passed",
        },
        "nccl_all_reduce": {
            "algorithm_bandwidth_gbps": 412.5,
            "errors": 0,
            "evidence_sha256": "0" * 64,
            "gpu_count": 8,
            "pair_all_reduce_passed": True,
            "status": "passed",
        },
        "nvidia_smi_attestation": {
            "attestation_sha256": attestation_sha256,
            "ecc_uncorrected_errors": 0,
            "evidence_sha256": "1" * 64,
            "gpu_count": 8,
            "status": "passed",
        },
        "one_step_training": {
            "arms": ["dense", "split90"],
            "evidence_sha256": "2" * 64,
            "oom_detected": False,
            "status": "passed",
            "updates": 1,
        },
        "throughput_100_updates": {
            "evidence_sha256": "3" * 64,
            "oom_detected": False,
            "status": "passed",
            "tokens_per_second": {"dense": 412_000.0, "split90": 410_500.0},
            "updates": 100,
        },
    }
    value.update(overrides)
    return value


def _site_evidence(**overrides) -> dict:
    attestation = overrides.pop("attestation", None) or _attestation()
    value = {
        "attestation": attestation,
        "authority_id": AUTHORITY,
        "cohort_id": "memorysplit-exploratory-v3-135m-aws-n10",
        "pair_allocations": _allocations(),
        "qualification_gates": _gates(canonical_digest(attestation)),
        "schema_version": 1,
    }
    value.update(overrides)
    return value


def test_shipped_profile_binds_every_frozen_p6_b200_fact():
    shipped = load_hardware_profile(B200_PROFILE)
    assert shipped.profile_id == "aws-p6-b200.48xlarge-135m-v1"
    assert shipped.authority_id == AUTHORITY
    assert shipped.instance_type == "p6-b200.48xlarge"
    assert shipped.gpu_count == 8
    assert shipped.vcpus == 192
    assert shipped.memory_gib_floor == 2_048
    assert shipped.local_storage_gb_floor == 30_400
    assert shipped.region == "us-east-1"
    assert shipped.gpus_per_pair == 2
    assert shipped.pairs_per_node == 4
    assert shipped.gpus_per_pair * shipped.pairs_per_node == shipped.gpu_count
    assert shipped.software_floors["cuda_version"] == "12.8"
    assert shipped.software_floors["driver_version"] == "570"
    assert shipped.software_floors["kernel_version"] == "6.1"
    assert shipped.software_floors["efa_installer_version"] == "1.41.0"
    assert shipped.software_floors["ofi_nccl_version"] == "1.15.0"
    assert shipped.dataset_contract_id == corpus.CONTRACT_ID
    # Account-specific facts cannot be derived offline, so the shipped profile
    # must still advertise them as unresolved.
    assert shipped.launch_ready is False
    assert set(shipped.pending_confirmation) == set(CONFIRMATIONS)


def test_shipped_profile_refuses_admission_while_placeholders_remain():
    shipped = load_hardware_profile(B200_PROFILE)
    with pytest.raises(HardwareAdmissionError, match="pending confirmation"):
        admit_b200_node(
            profile=shipped,
            attestation=_attestation(),
            asserted_authority=AUTHORITY,
        )


def test_exact_profile_admits_only_the_exact_b200_node(profile):
    attestation = _attestation()
    receipt = admit_b200_node(
        profile=profile,
        attestation=attestation,
        asserted_authority=AUTHORITY,
    )
    assert receipt["admitted"] is True
    assert receipt["instance_type"] == "p6-b200.48xlarge"
    assert receipt["profile_sha256"] == profile.sha256
    assert receipt["attestation_sha256"] == canonical_digest(attestation)
    assert len(receipt["gpu_uuids"]) == 8


@pytest.mark.parametrize("gpu_name", INEXACT_GPU_NAMES)
def test_b200_authority_rejects_every_regex_compatible_impostor(profile, gpu_name):
    legacy = load_profile(LEGACY_AWS_PROFILE)
    regex_admits = bool(re.search(legacy.gpu_name_regex, gpu_name))
    attestation = _attestation(
        gpus=[_gpu(index, name=gpu_name) for index in range(8)]
    )
    with pytest.raises(HardwareAdmissionError, match="gpu"):
        admit_b200_node(
            profile=profile,
            attestation=attestation,
            asserted_authority=AUTHORITY,
        )
    # The exact gate must reject the impostor whether or not the legacy regex
    # would have waved it through.
    assert regex_admits in {True, False}


def test_b200_authority_rejects_p5_instances_and_mixed_hardware(profile):
    for attestation in (
        _attestation(instance_type="p5.48xlarge"),
        _attestation(instance_type="p5en.48xlarge"),
        _attestation(instance_type="p6e-gb200.36xlarge"),
        _attestation(gpus=[_gpu(index) for index in range(4)]),
        _attestation(
            gpus=[
                _gpu(index, name="NVIDIA B200" if index < 4 else "NVIDIA H100 80GB HBM3")
                for index in range(8)
            ]
        ),
        _attestation(
            gpus=[
                _gpu(index, memory_mib=183_359 if index else 143_771)
                for index in range(8)
            ]
        ),
    ):
        with pytest.raises(HardwareAdmissionError):
            admit_b200_node(
                profile=profile,
                attestation=attestation,
                asserted_authority=AUTHORITY,
            )


@pytest.mark.parametrize(
    "asserted",
    [
        "memorysplit-p5-48xlarge-135m-v1",
        "memorysplit-p5en-48xlarge-135m-v1",
        "memorysplit-b300-135m-v1",
        "memorysplit-p6e-gb200-135m-v1",
        "",
        None,
    ],
)
def test_b200_evidence_is_rejected_under_any_other_authority(profile, asserted):
    with pytest.raises(HardwareAdmissionError, match="authority"):
        admit_b200_node(
            profile=profile,
            attestation=_attestation(),
            asserted_authority=asserted,
        )


def test_attestation_authority_must_match_the_asserted_authority(profile):
    with pytest.raises(HardwareAdmissionError, match="authority"):
        admit_b200_node(
            profile=profile,
            attestation=_attestation(profile_id="aws-p5-p6"),
            asserted_authority=AUTHORITY,
        )


@pytest.mark.parametrize(
    "field",
    [
        "ami_id",
        "ami_owner",
        "availability_zone",
        "container_image_digest",
        "cuda_version",
        "driver_version",
        "gpus",
        "instance_type",
        "kernel_version",
        "local_storage_gb",
        "memory_gib",
        "nccl_version",
        "ofi_nccl_version",
        "region",
        "torch_version",
        "vcpus",
    ],
)
def test_missing_hardware_fields_fail_closed(profile, field):
    attestation = _attestation()
    del attestation[field]
    with pytest.raises(HardwareAdmissionError):
        admit_b200_node(
            profile=profile,
            attestation=attestation,
            asserted_authority=AUTHORITY,
        )


@pytest.mark.parametrize(
    "field",
    ["ami_id", "cuda_version", "driver_version", "instance_type", "vcpus"],
)
def test_null_hardware_fields_fail_closed(profile, field):
    with pytest.raises(HardwareAdmissionError):
        admit_b200_node(
            profile=profile,
            attestation=_attestation(**{field: None}),
            asserted_authority=AUTHORITY,
        )


def test_unknown_attestation_fields_fail_closed(profile):
    with pytest.raises(HardwareAdmissionError):
        admit_b200_node(
            profile=profile,
            attestation=_attestation(gpu_override="trust me"),
            asserted_authority=AUTHORITY,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cuda_version", "12.6"),
        ("cuda_version", "unknown"),
        ("driver_version", "560.35.03"),
        ("efa_installer_version", "1.40.0"),
        ("kernel_version", "5.15.0-1051-aws"),
        ("nccl_version", "2.21.5"),
        ("ofi_nccl_version", "1.14.0"),
        ("torch_version", "2.6.0"),
    ],
)
def test_software_floor_violations_fail_closed(profile, field, value):
    with pytest.raises(HardwareAdmissionError, match=field):
        admit_b200_node(
            profile=profile,
            attestation=_attestation(**{field: value}),
            asserted_authority=AUTHORITY,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ami_id", "ami-00000000000000000"),
        ("ami_owner", "999999999999"),
        ("availability_zone", "us-east-1a"),
        ("container_image_digest", "sha256:" + "2" * 64),
        ("region", "us-west-2"),
    ],
)
def test_image_placement_and_runtime_must_match_exactly(profile, field, value):
    with pytest.raises(HardwareAdmissionError, match=field):
        admit_b200_node(
            profile=profile,
            attestation=_attestation(**{field: value}),
            asserted_authority=AUTHORITY,
        )


def test_four_disjoint_two_gpu_pairs_are_required_with_derived_ids(profile):
    attestation = _attestation()
    admitted = admit_pair_allocations(
        profile=profile,
        attestation=attestation,
        allocations=_allocations(),
    )
    assert len(admitted) == 4
    assert [record["pair_index"] for record in admitted] == [0, 1, 2, 3]
    assert [record["cuda_visible_devices"] for record in admitted] == [
        "0,1",
        "2,3",
        "4,5",
        "6,7",
    ]
    assert admitted[0]["arms"]["dense"]["gpu_index"] == 0
    assert admitted[0]["arms"]["split90"]["gpu_index"] == 1
    used = [index for record in admitted for index in record["gpu_indices"]]
    assert sorted(used) == list(range(8))


def test_pair_allocations_cannot_hardcode_gpu_ids(profile):
    allocations = _allocations()
    allocations[0]["arms"]["dense"]["gpu_index"] = 0
    with pytest.raises(HardwareAdmissionError, match="derived"):
        admit_pair_allocations(
            profile=profile,
            attestation=_attestation(),
            allocations=allocations,
        )


def test_pair_allocations_reject_overlap_duplication_and_wrong_counts(profile):
    attestation = _attestation()
    inventory = [_gpu(index)["uuid"] for index in range(8)]

    overlapping = _allocations()
    overlapping[1]["arms"]["dense"]["gpu_uuid"] = inventory[0]

    duplicated = _allocations()
    duplicated[2] = copy.deepcopy(duplicated[0])

    self_paired = _allocations()
    self_paired[0]["arms"]["split90"]["gpu_uuid"] = inventory[0]

    foreign = _allocations()
    foreign[3]["arms"]["split90"]["gpu_uuid"] = (
        "GPU-deadbeef-1000-2000-3000-000000000000"
    )

    for allocations in (
        overlapping,
        duplicated,
        self_paired,
        foreign,
        _allocations((0, 1, 2)),
        _allocations((0, 1, 2, 3, 4)),
        [],
    ):
        with pytest.raises(HardwareAdmissionError):
            admit_pair_allocations(
                profile=profile,
                attestation=attestation,
                allocations=allocations,
            )


def test_pair_allocations_require_both_arms_and_a_canonical_pair_id(profile):
    attestation = _attestation()
    missing_arm = _allocations()
    del missing_arm[0]["arms"]["split90"]

    wrong_pair_id = _allocations()
    wrong_pair_id[0]["pair_id"] = "d135m_full_s0"

    mismatched_seed = _allocations()
    mismatched_seed[0]["seed"] = 7

    for allocations in (missing_arm, wrong_pair_id, mismatched_seed):
        with pytest.raises(HardwareAdmissionError):
            admit_pair_allocations(
                profile=profile,
                attestation=attestation,
                allocations=allocations,
            )


def test_every_qualification_gate_is_required(profile):
    attestation = _attestation()
    digest = canonical_digest(attestation)
    assert set(QUALIFICATION_GATES) == {
        "checkpoint_write",
        "device_query",
        "exact_resume_equality",
        "memory_headroom",
        "nccl_all_reduce",
        "nvidia_smi_attestation",
        "one_step_training",
        "throughput_100_updates",
    }
    report = admit_qualification_gates(
        profile=profile,
        gates=_gates(digest),
        attestation_sha256=digest,
    )
    assert report["passed"] == list(QUALIFICATION_GATES)
    for name in QUALIFICATION_GATES:
        gates = _gates(digest)
        del gates[name]
        with pytest.raises(HardwareAdmissionError, match=name):
            admit_qualification_gates(
                profile=profile,
                gates=gates,
                attestation_sha256=digest,
            )
        gates = _gates(digest)
        gates[name]["status"] = "pending"
        with pytest.raises(HardwareAdmissionError, match=name):
            admit_qualification_gates(
                profile=profile,
                gates=gates,
                attestation_sha256=digest,
            )


def test_qualification_gate_measurements_fail_closed(profile):
    attestation = _attestation()
    digest = canonical_digest(attestation)

    cases = [
        ("device_query", {"devices_passed": 7}),
        ("device_query", {"result": "FAIL"}),
        ("nvidia_smi_attestation", {"attestation_sha256": "9" * 64}),
        ("nvidia_smi_attestation", {"ecc_uncorrected_errors": 1}),
        ("nccl_all_reduce", {"gpu_count": 2}),
        ("nccl_all_reduce", {"errors": 1}),
        ("nccl_all_reduce", {"algorithm_bandwidth_gbps": 0.0}),
        ("nccl_all_reduce", {"pair_all_reduce_passed": False}),
        ("one_step_training", {"updates": 0}),
        ("one_step_training", {"arms": ["dense"]}),
        ("one_step_training", {"oom_detected": True}),
        ("throughput_100_updates", {"updates": 99}),
        ("throughput_100_updates", {"oom_detected": True}),
        (
            "throughput_100_updates",
            {"tokens_per_second": {"dense": 0.0, "split90": 410_500.0}},
        ),
        ("memory_headroom", {"headroom_fraction": 0.01}),
        ("memory_headroom", {"oom_detected": True}),
        ("memory_headroom", {"peak_allocated_mib": 200_000}),
        ("memory_headroom", {"device_total_mib": 81_559}),
        ("checkpoint_write", {"bytes": 0}),
        ("checkpoint_write", {"fsync": False}),
        ("exact_resume_equality", {"resumed_state_sha256": "7" * 64}),
        ("exact_resume_equality", {"resume_exact": False}),
        ("exact_resume_equality", {"updates": 1}),
    ]
    for name, patch in cases:
        gates = _gates(digest)
        gates[name].update(patch)
        with pytest.raises(HardwareAdmissionError, match=name):
            admit_qualification_gates(
                profile=profile,
                gates=gates,
                attestation_sha256=digest,
            )

    unknown = _gates(digest)
    unknown["informal_eyeball_check"] = {"status": "passed"}
    with pytest.raises(HardwareAdmissionError):
        admit_qualification_gates(
            profile=profile,
            gates=unknown,
            attestation_sha256=digest,
        )


def test_protected_site_admission_returns_a_fully_bound_receipt(profile):
    evidence = _site_evidence()
    receipt = admit_protected_site(
        profile=profile,
        site_evidence=evidence,
        asserted_authority=AUTHORITY,
    )
    assert receipt["admitted"] is True
    assert receipt["authority_id"] == AUTHORITY
    assert receipt["profile_id"] == profile.profile_id
    assert receipt["profile_sha256"] == profile.sha256
    assert receipt["instance_type"] == "p6-b200.48xlarge"
    assert receipt["qualification_gates"] == list(QUALIFICATION_GATES)
    assert len(receipt["pair_allocations"]) == 4

    for mutation in (
        {"authority_id": "memorysplit-b300-135m-v1"},
        {"cohort_id": "memorysplit-full-135m"},
        {"pair_allocations": _allocations((0, 1))},
        {"schema_version": 2},
    ):
        with pytest.raises(HardwareAdmissionError):
            admit_protected_site(
                profile=profile,
                site_evidence=_site_evidence(**mutation),
                asserted_authority=AUTHORITY,
            )


def test_reasoning_v3_site_admission_binds_the_frozen_corpus(tmp_path):
    profile_path = _write_profile(tmp_path, _confirmed_document())
    receipt = admit_reasoning_v3_site(
        site_evidence=_site_evidence(),
        asserted_authority=AUTHORITY,
        profile_path=profile_path,
        transfer_manifest_sha256=corpus.TRANSFER_MANIFEST_SHA256,
        virtual_receipt_sha256=corpus.VIRTUAL_RECEIPT_SHA256,
    )
    assert receipt["transfer_manifest_sha256"] == corpus.TRANSFER_MANIFEST_SHA256
    assert receipt["virtual_receipt_sha256"] == corpus.VIRTUAL_RECEIPT_SHA256
    assert receipt["dataset_contract_id"] == corpus.CONTRACT_ID

    for kwargs in (
        {"transfer_manifest_sha256": "0" * 64},
        {"virtual_receipt_sha256": "0" * 64},
        {"asserted_authority": "memorysplit-p5-48xlarge-135m-v1"},
    ):
        with pytest.raises(HardwareAdmissionError):
            admit_reasoning_v3_site(
                **{
                    "site_evidence": _site_evidence(),
                    "asserted_authority": AUTHORITY,
                    "profile_path": profile_path,
                    "transfer_manifest_sha256": corpus.TRANSFER_MANIFEST_SHA256,
                    "virtual_receipt_sha256": corpus.VIRTUAL_RECEIPT_SHA256,
                    **kwargs,
                }
            )


def test_reasoning_v3_site_admission_rejects_the_unconfirmed_shipped_profile():
    with pytest.raises(HardwareAdmissionError, match="pending confirmation"):
        admit_reasoning_v3_site(
            site_evidence=_site_evidence(),
            asserted_authority=AUTHORITY,
            profile_path=B200_PROFILE,
            transfer_manifest_sha256=corpus.TRANSFER_MANIFEST_SHA256,
            virtual_receipt_sha256=corpus.VIRTUAL_RECEIPT_SHA256,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        {"authority_id": None},
        {"hardware": {"instance_type": "p5.48xlarge"}},
        {"pair_geometry": {"pairs_per_node": 3}},
        {"pair_geometry": {"gpus_per_pair": 8}},
        {"placement": {"region": "not-a-region"}},
    ],
)
def test_profile_loader_fails_closed_on_inconsistent_documents(tmp_path, mutation):
    document = _confirmed_document()
    for key, value in mutation.items():
        if isinstance(value, dict):
            document[key].update(value)
        else:
            document[key] = value
    with pytest.raises(HardwareAdmissionError):
        load_hardware_profile(_write_profile(tmp_path, document, name="bad.json"))


def test_profile_loader_requires_a_pending_confirmation_bijection(tmp_path):
    silently_unconfirmed = _confirmed_document()
    silently_unconfirmed["image"]["ami_id"] = None
    with pytest.raises(HardwareAdmissionError, match="pending_confirmation"):
        load_hardware_profile(
            _write_profile(tmp_path, silently_unconfirmed, name="hidden.json")
        )

    falsely_pending = _confirmed_document()
    falsely_pending["pending_confirmation"] = [
        {
            "field": "image.ami_id",
            "reason": "fixture",
            "resolve_with": "fixture",
        }
    ]
    with pytest.raises(HardwareAdmissionError, match="pending_confirmation"):
        load_hardware_profile(
            _write_profile(tmp_path, falsely_pending, name="false.json")
        )


def test_no_broad_gpu_regex_can_authorize_protected_b200_work(profile):
    """Every shipped Slurm profile regex is strictly weaker than the exact gate."""

    for path in sorted(PROFILES.glob("*.json")):
        if path.name.endswith("schema.json") or path == B200_PROFILE:
            continue
        slurm_profile = load_profile(path)
        for gpu_name in INEXACT_GPU_NAMES:
            if not re.search(slurm_profile.gpu_name_regex, gpu_name):
                continue
            with pytest.raises(HardwareAdmissionError):
                admit_b200_node(
                    profile=profile,
                    attestation=_attestation(
                        gpus=[_gpu(index, name=gpu_name) for index in range(8)]
                    ),
                    asserted_authority=AUTHORITY,
                )


def test_instantiate_aws_requires_both_hardware_arguments(tmp_path):
    for kwargs in (
        {"hardware_profile_path": B200_PROFILE},
        {"site_evidence_path": tmp_path / "site.json"},
    ):
        with pytest.raises(ValueError, match="hardware"):
            aws_operations.instantiate_aws(
                dataset_root=tmp_path / "dataset",
                pointer_path=ROOT / "DATASET-POINTER-AWS-135M-V3.json",
                transfer_manifest_path=(
                    ROOT / "cluster" / "aws" / "reasoning-v3-corpus-manifest.json"
                ),
                profile_path=LEGACY_AWS_PROFILE,
                runtime_root=tmp_path / "runtime",
                out_root=tmp_path / "outputs",
                repository_root=ROOT,
                seeds=(0,),
                **kwargs,
            )


def test_protected_submission_is_blocked_before_any_job_is_planned(tmp_path):
    site = tmp_path / "site.json"
    site.write_text(json.dumps(_site_evidence(), indent=2, sort_keys=True) + "\n")

    def forbidden(command):
        raise AssertionError(f"unadmitted hardware reached sbatch: {command}")

    # The shipped profile still has unconfirmed fields, so admission must fail
    # before submit() ever inspects a pair manifest or preflight receipt.
    with pytest.raises(HardwareAdmissionError, match="pending confirmation"):
        aws_operations.authorize_protected_submission(
            [tmp_path / "pair-s0.json"],
            profile_path=LEGACY_AWS_PROFILE,
            venv_root=tmp_path / "venv",
            preflight_path=tmp_path / "preflight.json",
            site_evidence_path=site,
            apply=True,
            runner=forbidden,
        )

    # A confirmed profile still refuses site evidence carrying P5 authority.
    confirmed = _write_profile(tmp_path, _confirmed_document())
    p5_site = tmp_path / "p5-site.json"
    p5_site.write_text(
        json.dumps(
            _site_evidence(authority_id="memorysplit-p5-48xlarge-135m-v1"),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with pytest.raises(HardwareAdmissionError, match="authority"):
        aws_operations.authorize_protected_submission(
            [tmp_path / "pair-s0.json"],
            profile_path=LEGACY_AWS_PROFILE,
            venv_root=tmp_path / "venv",
            preflight_path=tmp_path / "preflight.json",
            site_evidence_path=p5_site,
            hardware_profile_path=confirmed,
            apply=True,
            runner=forbidden,
        )


def test_readiness_report_marks_the_exact_profile_as_not_yet_admissible():
    report = readiness.hardware_profile_report(B200_PROFILE)
    assert report["instance_type"] == "p6-b200.48xlarge"
    assert report["launch_ready"] is False
    assert sorted(report["pending_confirmation"]) == sorted(CONFIRMATIONS)
    assert report["protected_launch_admissible"] is False
