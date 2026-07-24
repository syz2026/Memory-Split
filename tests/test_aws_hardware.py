from __future__ import annotations

import base64
import hashlib
import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
LEGACY_P5_PROFILE = ROOT / "cluster" / "profiles" / "aws-p5.48xlarge.json"
P5_PROFILE = ROOT / "cluster" / "profiles" / "aws-p5.48xlarge-v3.json"
P6_PROFILE = ROOT / "cluster" / "profiles" / "aws-p6-b300.48xlarge-v3.json"
AMENDMENT = ROOT / "configs" / "aws-hardware-amendment-v3.json"
PREREGISTRATION = ROOT / "configs" / "preregistration-v3.yaml"
COHORT_ASSIGNMENT = ROOT / "configs" / "cohort-assignment-v3.json"

P5_PROFILE_SHA256 = "2207bfbad5e8fa9fc804770b582d0b21f8b6ed109b2e3f3b5c0474c732c53543"
PREREGISTRATION_SHA256 = (
    "6b2b5da3e3dc3d533498a0aa9d1891f356134ce045b1553b74a0161d94cb81d7"
)
COHORT_ASSIGNMENT_SHA256 = (
    "47faf6f15e13336f67666081c3d0ebf5be3b981f3d2ac455385faa39e543199c"
)
AMENDMENT_SHA256 = (
    "d4cf13b587c751d27756ad7881e538facb7ea79305a098990a568a7b28b6fb14"
)
P6_SOFTWARE_FLOORS = {
    "cuda": "13.0",
    "efa": "1.44.0",
    "kernel": "6.1",
    "nvidia_driver": "R580",
    "nvlink": "R580",
    "ofi_nccl": "1.17.1",
}
TRUSTED_APPROVAL_PUBLIC_KEY_SHA256 = "9" * 64
SELECTION_LOCAL_PATH = (
    "memorysplit-confirmatory-v3-360m-n10-aws/provider-selection.json"
)
SELECTION_S3_KEY = (
    "cohorts/memorysplit-confirmatory-v3-360m-n10-aws/provider-selection.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return path


def test_neutral_loader_preserves_the_exact_current_p5_profile():
    from cluster.aws.gpu_profile import AwsGpuProfile, load_aws_gpu_profile

    assert _sha256(P5_PROFILE) == P5_PROFILE_SHA256
    profile = load_aws_gpu_profile(P5_PROFILE)

    assert isinstance(profile, AwsGpuProfile)
    assert profile.profile_id == "aws-p5.48xlarge-v3"
    assert profile.provider == "aws-p5.48xlarge"
    assert profile.instance_type == "p5.48xlarge"
    assert profile.architecture == "x86_64"
    assert profile.vcpus == 192
    assert profile.memory_gib == 2048
    assert profile.gpu_model == "NVIDIA H100 80GB"
    assert profile.allocated_gpus == 8
    assert profile.train_groups == (4, 4)
    assert profile.assigned_seeds == tuple(range(10))
    assert profile.allowed_regions == ("us-east-1",)
    assert profile.allowed_availability_zones == ()
    assert profile.sha256 == P5_PROFILE_SHA256


def test_legacy_p5_names_are_aliases_of_the_neutral_contract():
    from cluster.aws.gpu_profile import (
        AwsGpuProfile,
        AwsGpuRuntime,
        load_aws_gpu_profile,
        parse_aws_gpu_profile_bytes,
    )
    from cluster.aws.p5.profile import (
        AwsP5Profile,
        AwsP5Runtime,
        load_aws_p5_profile,
        parse_aws_p5_profile_bytes,
    )

    assert AwsP5Profile is AwsGpuProfile
    assert AwsP5Runtime is AwsGpuRuntime
    assert load_aws_p5_profile is load_aws_gpu_profile
    assert parse_aws_p5_profile_bytes is parse_aws_gpu_profile_bytes
    assert load_aws_p5_profile(P5_PROFILE) == load_aws_gpu_profile(P5_PROFILE)


def _legacy_profile_constructor_values() -> tuple[object, ...]:
    return (
        1,
        "aws-p5.48xlarge",
        "aws-p5.48xlarge",
        "p5.48xlarge",
        "on_demand",
        192,
        2048,
        "NVIDIA H100 80GB",
        8,
        (4, 4),
        "/mnt/memorysplit",
        "MS_S3_ROOT",
        "Amazon EC2 NVMe Instance Storage",
        8,
        3_840_000_000_000,
        "0",
        "AWS_REGION",
        "MS_AWS_AMI_ID",
        "MS_CONTAINER_DIGEST",
        "MS_RUNTIME_UID",
        "MS_RUNTIME_GID",
        (1, 2, 3, 4),
        ("AWS_REGION", "LANG", "LC_ALL"),
        "f" * 64,
    )


def test_legacy_p5_profile_direct_constructor_keeps_old_positional_order():
    from cluster.aws.p5.profile import AwsP5Profile

    profile = AwsP5Profile(*_legacy_profile_constructor_values())

    assert profile.vcpus == 192
    assert profile.sha256 == "f" * 64
    assert profile.architecture == "x86_64"
    assert profile.allowed_regions == ()
    assert profile.allowed_availability_zones == ()
    assert profile.software_floors == ()


def test_legacy_p5_profile_direct_constructor_keeps_old_keyword_surface():
    from cluster.aws.p5.profile import AwsP5Profile

    names = (
        "schema_version",
        "profile_id",
        "provider",
        "instance_type",
        "purchase_model",
        "vcpus",
        "memory_gib",
        "gpu_model",
        "allocated_gpus",
        "train_groups",
        "scratch_root",
        "durable_uri_env",
        "instance_store_model",
        "instance_store_devices",
        "instance_store_device_bytes",
        "raid_level",
        "region_env",
        "ami_id_env",
        "container_digest_env",
        "runtime_uid_env",
        "runtime_gid_env",
        "assigned_seeds",
        "process_env_allowlist",
        "sha256",
    )
    profile = AwsP5Profile(
        **dict(zip(names, _legacy_profile_constructor_values(), strict=True))
    )

    assert profile.profile_id == "aws-p5.48xlarge"
    assert profile.architecture == "x86_64"
    assert profile.software_floors == ()


def test_legacy_p5_runtime_direct_constructor_is_unchanged():
    from cluster.aws.p5.profile import AwsP5Runtime

    runtime = AwsP5Runtime(
        "us-east-1",
        "s3://bucket/prefix",
        "ami-0123456789abcdef0",
        "registry.example/repo@sha256:" + "a" * 64,
        "sha256:" + "a" * 64,
        1000,
        1000,
    )

    assert runtime.uid == 1000
    assert runtime.gid == 1000


def test_legacy_p5_runtime_validation_keeps_region_selection_out_of_profile():
    from cluster.aws.p5.profile import (
        load_aws_p5_profile,
        validate_runtime_environment,
    )

    profile = load_aws_p5_profile(LEGACY_P5_PROFILE)
    runtime = validate_runtime_environment(
        profile,
        {
            "AWS_REGION": "us-west-2",
            "MS_S3_ROOT": "s3://memorysplit-prod/cohort-v2",
            "MS_AWS_AMI_ID": "ami-0123456789abcdef0",
            "MS_CONTAINER_DIGEST": "sha256:" + "a" * 64,
            "MS_CONTAINER_IMAGE": (
                "123456789012.dkr.ecr.us-west-2.amazonaws.com/memorysplit"
                "@sha256:" + "a" * 64
            ),
            "MS_RUNTIME_UID": "1000",
            "MS_RUNTIME_GID": "1000",
        },
    )

    assert runtime.region == "us-west-2"


def test_closed_p6_profile_freezes_b300_hardware_and_one_cohort():
    from cluster.aws.gpu_profile import AwsGpuProfile, load_aws_gpu_profile

    raw = _json(P6_PROFILE)
    profile = load_aws_gpu_profile(P6_PROFILE)

    assert isinstance(profile, AwsGpuProfile)
    assert profile.profile_id == "aws-p6-b300.48xlarge-v3"
    assert profile.provider == "aws-p6-b300.48xlarge"
    assert profile.instance_type == "p6-b300.48xlarge"
    assert profile.purchase_model == "on_demand"
    assert profile.architecture == "x86_64"
    assert profile.vcpus == 192
    assert profile.memory_gib == 4096
    assert profile.gpu_model == "NVIDIA B300"
    assert profile.allocated_gpus == 8
    assert profile.train_groups == (4, 4)
    assert profile.instance_store_devices == 8
    assert profile.instance_store_device_bytes == 3_840_000_000_000
    assert profile.raid_level == "0"
    assert profile.assigned_seeds == tuple(range(10))
    assert profile.allowed_regions == ("us-east-1",)
    assert profile.allowed_availability_zones == ("us-east-1d",)
    assert profile.software_floors == (
        ("cuda", "13.0"),
        ("efa", "1.44.0"),
        ("kernel", "6.1"),
        ("nvidia_driver", "R580"),
        ("nvlink", "R580"),
        ("ofi_nccl", "1.17.1"),
    )
    assert raw["cpu"] == {
        "architecture": "x86_64",
        "memory_gib": 4096,
        "vcpus": 192,
    }
    assert raw["offerings"] == [
        {
            "availability_zone": "us-east-1d",
            "region": "us-east-1",
        }
    ]
    assert raw["software_floors"] == P6_SOFTWARE_FLOORS


def test_p6_nvlink_floor_retains_official_aws_requirement_provenance():
    from cluster.aws.gpu_profile import P6_SOFTWARE_FLOOR_PROVENANCE

    assert P6_SOFTWARE_FLOORS["nvlink"] == "R580"
    assert "official AWS P6-B300 requirements" in (
        P6_SOFTWARE_FLOOR_PROVENANCE
    )
    assert "NVLINK 5 R580" in P6_SOFTWARE_FLOOR_PROVENANCE


def test_hardware_profiles_delegate_ami_and_image_identity_to_runtime_lock():
    p5 = _json(P5_PROFILE)
    p6 = _json(P6_PROFILE)

    for value in (p5, p6):
        runtime = value["runtime"]
        assert runtime == {
            "ami_id_env": "MS_AWS_AMI_ID",
            "container_digest_env": "MS_CONTAINER_DIGEST",
            "region_env": "AWS_REGION",
            "runtime_gid_env": "MS_RUNTIME_GID",
            "runtime_uid_env": "MS_RUNTIME_UID",
        }
        assert "ami_id" not in value
        assert "container_image" not in value
        assert "container_image_digest" not in value
        assert ":latest" not in json.dumps(value)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["cpu"].update(architecture="arm64"),
        lambda value: value["cpu"].update(memory_gib=2048),
        lambda value: value["gpu"].update(model="NVIDIA H100 80GB"),
        lambda value: value["gpu"].update(allocated=4),
        lambda value: value["gpu"].update(seed_train_groups=[8, 0]),
        lambda value: value.update(purchase_model="capacity_block"),
        lambda value: value["offerings"][0].update(
            availability_zone="us-east-1c"
        ),
        lambda value: value.update(
            software_floors=P6_SOFTWARE_FLOORS | {"cuda": "12.9"}
        ),
        lambda value: value.update(
            software_floors=P6_SOFTWARE_FLOORS | {"efa": "1.43.0"}
        ),
        lambda value: value.update(
            software_floors=P6_SOFTWARE_FLOORS | {"kernel": "5.15"}
        ),
        lambda value: value.update(
            software_floors=P6_SOFTWARE_FLOORS | {"nvidia_driver": "R570"}
        ),
        lambda value: value.update(
            software_floors=P6_SOFTWARE_FLOORS | {"nvlink": "R570"}
        ),
        lambda value: value.update(
            software_floors=P6_SOFTWARE_FLOORS | {"ofi_nccl": "1.16.0"}
        ),
        lambda value: value.update(assigned_seeds=list(range(9))),
        lambda value: value.update(ami_id="ami-0123456789abcdef0"),
        lambda value: value["runtime"].update(container_image="repo:latest"),
    ],
    ids=[
        "architecture",
        "memory",
        "gpu-model",
        "gpu-count",
        "asymmetric-groups",
        "purchase-model",
        "availability-zone",
        "cuda-floor",
        "efa-floor",
        "kernel-floor",
        "driver-floor",
        "nvlink-floor",
        "ofi-nccl-floor",
        "seeds",
        "embedded-ami",
        "embedded-image",
    ],
)
def test_p6_profile_rejects_hardware_or_runtime_authority_drift(
    tmp_path,
    mutation,
):
    from cluster.aws.gpu_profile import load_aws_gpu_profile

    value = deepcopy(_json(P6_PROFILE))
    mutation(value)
    path = _write_json(tmp_path / "profile.json", value)

    with pytest.raises(ValueError):
        load_aws_gpu_profile(path)


def test_profile_loader_rejects_hardlinks(tmp_path):
    from cluster.aws.gpu_profile import load_aws_gpu_profile

    source = tmp_path / "source.json"
    source.write_bytes(P5_PROFILE.read_bytes())
    hardlink = tmp_path / "profile.json"
    os.link(source, hardlink)

    with pytest.raises(ValueError, match="link"):
        load_aws_gpu_profile(hardlink)


def test_profile_loader_rejects_group_or_other_writable_mode(tmp_path):
    from cluster.aws.gpu_profile import load_aws_gpu_profile

    path = tmp_path / "profile.json"
    path.write_bytes(P5_PROFILE.read_bytes())
    path.chmod(0o666)

    with pytest.raises(ValueError, match="mode|writable"):
        load_aws_gpu_profile(path)


def test_profile_loader_rejects_wrong_owner(tmp_path, monkeypatch):
    from cluster.aws.gpu_profile import load_aws_gpu_profile

    path = tmp_path / "profile.json"
    path.write_bytes(P5_PROFILE.read_bytes())
    monkeypatch.setattr(os, "geteuid", lambda: path.stat().st_uid + 1)

    with pytest.raises(ValueError, match="owner"):
        load_aws_gpu_profile(path)


def test_profile_loader_detects_path_replacement_during_read(
    tmp_path,
    monkeypatch,
):
    from cluster.aws.gpu_profile import load_aws_gpu_profile

    path = tmp_path / "profile.json"
    path.write_bytes(P5_PROFILE.read_bytes())
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(P5_PROFILE.read_bytes())
    original_read = os.read
    replaced = False

    def replacing_read(descriptor, count):
        nonlocal replaced
        data = original_read(descriptor, count)
        if not replaced:
            replaced = True
            path.unlink()
            replacement.rename(path)
        return data

    monkeypatch.setattr(os, "read", replacing_read)

    with pytest.raises(ValueError, match="changed|replaced"):
        load_aws_gpu_profile(path)
    assert replaced is True


def test_append_only_amendment_binds_frozen_contracts_and_both_profiles():
    from msctl.aws_hardware import load_aws_hardware_amendment

    assert _sha256(PREREGISTRATION) == PREREGISTRATION_SHA256
    assert _sha256(COHORT_ASSIGNMENT) == COHORT_ASSIGNMENT_SHA256
    assert _sha256(P5_PROFILE) == P5_PROFILE_SHA256

    amendment = load_aws_hardware_amendment(AMENDMENT, repo_root=ROOT)
    by_id = {profile.profile_id: profile for profile in amendment.profiles}

    assert amendment.schema_version == 1
    assert (
        amendment.amendment_id
        == "memorysplit-confirmatory-v3-aws-hardware-selection"
    )
    assert amendment.amendment_mode == "append_only"
    assert amendment.prospective is True
    assert amendment.cohort_id == "memorysplit-confirmatory-v3-360m-n10-aws"
    assert amendment.preregistration.path == "configs/preregistration-v3.yaml"
    assert amendment.preregistration.sha256 == PREREGISTRATION_SHA256
    assert (
        amendment.cohort_assignment.path
        == "configs/cohort-assignment-v3.json"
    )
    assert amendment.cohort_assignment.sha256 == COHORT_ASSIGNMENT_SHA256
    assert set(by_id) == {
        "aws-p5.48xlarge-v3",
        "aws-p6-b300.48xlarge-v3",
    }
    assert by_id["aws-p5.48xlarge-v3"].path == (
        "cluster/profiles/aws-p5.48xlarge-v3.json"
    )
    assert by_id["aws-p5.48xlarge-v3"].sha256 == P5_PROFILE_SHA256
    assert by_id["aws-p6-b300.48xlarge-v3"].path == (
        "cluster/profiles/aws-p6-b300.48xlarge-v3.json"
    )
    assert by_id["aws-p6-b300.48xlarge-v3"].sha256 == _sha256(P6_PROFILE)
    assert amendment.supersedes_only == (
        "provider_assignment",
        "hardware_topology",
    )
    assert amendment.seeds == tuple(range(10))
    assert amendment.arms == ("dense", "split90")
    assert amendment.train_groups == (4, 4)
    assert amendment.symmetric_training is True
    assert amendment.one_profile_for_entire_cohort is True
    assert amendment.scientific_invariants == (
        "all_non_provider_fields_remain_frozen"
    )
    assert amendment.protected_outcomes_inspected == ()
    assert amendment.sha256 == _sha256(AMENDMENT)


def test_amendment_is_canonical_pretty_json_with_exact_frozen_scope():
    value = _json(AMENDMENT)

    assert AMENDMENT.read_bytes() == (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    assert value["supersedes_only"] == [
        "provider_assignment",
        "hardware_topology",
    ]
    assert value["frozen_contract"] == {
        "arms": ["dense", "split90"],
        "one_profile_for_entire_cohort": True,
        "protected_outcomes_inspected": [],
        "scientific_invariants": "all_non_provider_fields_remain_frozen",
        "seeds": list(range(10)),
        "symmetric_training": True,
        "train_groups": [4, 4],
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(schema_version=True),
        lambda value: value.update(prospective=False),
        lambda value: value.update(supersedes_only=["scientific_invariants"]),
        lambda value: value["frozen_contract"].update(seeds=list(range(9))),
        lambda value: value["frozen_contract"].update(
            arms=["dense", "split90", "control"]
        ),
        lambda value: value["frozen_contract"].update(train_groups=[3, 5]),
        lambda value: value["frozen_contract"].update(
            protected_outcomes_inspected=["primary_endpoint"]
        ),
        lambda value: value["eligible_profiles"][0].update(sha256="0" * 64),
        lambda value: value.update(unknown=True),
    ],
    ids=[
        "bool-version",
        "retrospective",
        "scientific-supersession",
        "missing-seed",
        "extra-arm",
        "asymmetric-topology",
        "outcome-inspection",
        "profile-hash",
        "unknown-field",
    ],
)
def test_amendment_rejects_scope_or_binding_drift(tmp_path, mutation):
    from msctl.aws_hardware import load_aws_hardware_amendment

    value = deepcopy(_json(AMENDMENT))
    mutation(value)
    path = _write_json(tmp_path / "amendment.json", value)

    with pytest.raises(ValueError):
        load_aws_hardware_amendment(path, repo_root=ROOT)


def test_amendment_rejects_duplicate_keys(tmp_path):
    from msctl.aws_hardware import parse_aws_hardware_amendment_bytes

    data = AMENDMENT.read_bytes().replace(
        b'"schema_version": 1',
        b'"schema_version": 1,\n  "schema_version": 1',
        1,
    )

    with pytest.raises(ValueError, match="duplicate"):
        parse_aws_hardware_amendment_bytes(data)


def test_amendment_loader_rejects_hardlink_and_unsafe_mode(tmp_path):
    from msctl.aws_hardware import load_aws_hardware_amendment

    source = tmp_path / "source.json"
    source.write_bytes(AMENDMENT.read_bytes())
    hardlink = tmp_path / "hardlink.json"
    os.link(source, hardlink)
    with pytest.raises(ValueError, match="link"):
        load_aws_hardware_amendment(hardlink)

    writable = tmp_path / "writable.json"
    writable.write_bytes(AMENDMENT.read_bytes())
    writable.chmod(0o666)
    with pytest.raises(ValueError, match="mode|writable"):
        load_aws_hardware_amendment(writable)


def _profile_path_and_hash(profile_id: str) -> tuple[Path, str]:
    if profile_id == "aws-p6-b300.48xlarge-v3":
        return P6_PROFILE, _sha256(P6_PROFILE)
    if profile_id == "aws-p5.48xlarge-v3":
        return P5_PROFILE, P5_PROFILE_SHA256
    raise AssertionError(profile_id)


def _runtime_lock_value(profile_id: str) -> dict[str, object]:
    _, profile_sha256 = _profile_path_and_hash(profile_id)
    digest = "sha256:" + "b" * 64
    return {
        "ami_id": "ami-0123456789abcdef0",
        "ami_owner_id": "123456789012",
        "container_image": (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/memorysplit"
            f"@{digest}"
        ),
        "container_image_digest": digest,
        "control_bundle_sha256": "c" * 64,
        "profile_sha256": profile_sha256,
        "schema_version": 1,
        "source_commit": "d" * 40,
        "source_tree": "e" * 40,
        "versions": {
            "aws_cli": "2.27.0",
            "cuda": "13.0",
            "cudnn": "9.1.0",
            "docker": "27.0.0",
            "fabric_manager": "580.65.06",
            "nccl": "2.27.0",
            "nvidia_container_runtime": "1.17.8",
            "nvidia_driver": "580.65.06",
            "python": "3.12.0",
            "pytorch": "2.12.1",
        },
    }


def _verified_selection_fixture(
    profile_id: str = "aws-p6-b300.48xlarge-v3",
) -> tuple[dict[str, object], bytes, bytes]:
    evidence, lock_data = _qualification_evidence_value(profile_id)
    evidence_data = _canonical_json(evidence)
    selection = _selection_value(profile_id)
    selection["runtime"] = {
        "ami_id": "ami-0123456789abcdef0",
        "container_image_digest": "sha256:" + "b" * 64,
        "runtime_evidence_sha256": hashlib.sha256(evidence_data).hexdigest(),
        "runtime_lock_sha256": hashlib.sha256(lock_data).hexdigest(),
    }
    return selection, lock_data, evidence_data


@pytest.mark.parametrize(
    "profile_id",
    ["aws-p5.48xlarge-v3", "aws-p6-b300.48xlarge-v3"],
)
def test_verified_selection_parses_exact_runtime_and_attestation_bytes(
    profile_id,
):
    from msctl.aws_hardware import parse_verified_provider_selection_bytes

    profile_path, profile_sha256 = _profile_path_and_hash(profile_id)
    selection, lock_data, evidence_data = _verified_selection_fixture(profile_id)
    receipt = parse_verified_provider_selection_bytes(
        _canonical_json(selection),
        amendment_data=AMENDMENT.read_bytes(),
        profile_data=profile_path.read_bytes(),
        runtime_lock_data=lock_data,
        runtime_evidence_data=evidence_data,
        **_qualification_verification_kwargs(),
    )

    assert receipt.profile.sha256 == profile_sha256
    assert receipt.runtime_lock_sha256 == hashlib.sha256(lock_data).hexdigest()
    assert receipt.runtime_evidence_sha256 == hashlib.sha256(
        evidence_data
    ).hexdigest()
    assert receipt.ami_id == "ami-0123456789abcdef0"
    assert receipt.container_image_digest == "sha256:" + "b" * 64


@pytest.mark.parametrize(
    "mutation",
    [
        "lock-profile",
        "lock-ami",
        "lock-image",
        "evidence-profile",
        "evidence-runtime",
        "evidence-placement",
        "cuda-floor",
        "efa-floor",
        "kernel-floor",
        "driver-floor",
        "nvlink-floor",
        "ofi-nccl-floor",
    ],
)
def test_verified_selection_rejects_runtime_or_floor_drift(mutation):
    from msctl.aws_hardware import parse_verified_provider_selection_bytes

    selection, lock_data, evidence_data = _verified_selection_fixture()
    lock = json.loads(lock_data)
    evidence = json.loads(evidence_data)
    if mutation == "lock-profile":
        lock["profile_sha256"] = P5_PROFILE_SHA256
    elif mutation == "lock-ami":
        lock["ami_id"] = "ami-0fedcba9876543210"
    elif mutation == "lock-image":
        lock["container_image_digest"] = "sha256:" + "a" * 64
        lock["container_image"] = (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/memorysplit@"
            + lock["container_image_digest"]
        )
    elif mutation == "evidence-profile":
        evidence["canary_receipt"]["profile_sha256"] = P5_PROFILE_SHA256
    elif mutation == "evidence-runtime":
        evidence["canary_receipt"]["runtime_lock_sha256"] = "0" * 64
    elif mutation == "evidence-placement":
        evidence["environment_receipt"]["aws_instance_identity_document"][
            "availabilityZone"
        ] = "us-east-1c"
    else:
        field, value = {
            "cuda-floor": ("cuda", "12.9"),
            "efa-floor": ("efa", "1.43.9"),
            "kernel-floor": ("kernel", "5.15"),
            "driver-floor": ("nvidia_driver", "R570"),
            "nvlink-floor": ("nvlink", "R570"),
            "ofi-nccl-floor": ("ofi_nccl", "1.16.9"),
        }[mutation]
        evidence["canary_receipt"]["hardware"]["runtime_facts"][
            field
        ] = value

    lock_data = _canonical_json(lock)
    evidence_data = _canonical_json(evidence)
    selection["runtime"]["runtime_lock_sha256"] = hashlib.sha256(
        lock_data
    ).hexdigest()
    selection["runtime"]["runtime_evidence_sha256"] = hashlib.sha256(
        evidence_data
    ).hexdigest()

    with pytest.raises(ValueError):
        parse_verified_provider_selection_bytes(
            _canonical_json(selection),
            amendment_data=AMENDMENT.read_bytes(),
            profile_data=P6_PROFILE.read_bytes(),
            runtime_lock_data=lock_data,
            runtime_evidence_data=evidence_data,
            **_qualification_verification_kwargs(),
        )


def test_verified_selection_rejects_noncanonical_runtime_lock_bytes():
    from msctl.aws_hardware import parse_verified_provider_selection_bytes

    selection, lock_data, evidence_data = _verified_selection_fixture()
    pretty_lock = (
        json.dumps(json.loads(lock_data), indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    selection["runtime"]["runtime_lock_sha256"] = hashlib.sha256(
        pretty_lock
    ).hexdigest()

    with pytest.raises(ValueError, match="runtime lock.*canonical"):
        parse_verified_provider_selection_bytes(
            _canonical_json(selection),
            amendment_data=AMENDMENT.read_bytes(),
            profile_data=P6_PROFILE.read_bytes(),
            runtime_lock_data=pretty_lock,
            runtime_evidence_data=evidence_data,
            **_qualification_verification_kwargs(),
        )


class _QualificationIdentityVerifier:
    def __init__(self, valid: bool = True) -> None:
        self.valid = valid
        self.calls: list[tuple[dict[str, object], str, str]] = []

    def __call__(self, identity, pkcs7, region):
        self.calls.append((dict(identity), str(pkcs7), str(region)))
        return self.valid


class _QualificationApprovalVerifier:
    def __init__(self, valid: bool = True) -> None:
        self.valid = valid
        self.calls: list[dict[str, object]] = []

    def verify(
        self,
        *,
        payload: bytes,
        signature: str,
        algorithm: str,
        public_key_sha256: str,
    ) -> bool:
        self.calls.append(
            {
                "algorithm": algorithm,
                "payload": payload,
                "public_key_sha256": public_key_sha256,
                "signature": signature,
            }
        )
        return self.valid


def _qualification_verification_kwargs() -> dict[str, object]:
    return {
        "approval_verifier": _QualificationApprovalVerifier(),
        "identity_verifier": _QualificationIdentityVerifier(),
        "trusted_public_key_sha256": TRUSTED_APPROVAL_PUBLIC_KEY_SHA256,
    }


def _qualification_evidence_value(
    profile_id: str = "aws-p6-b300.48xlarge-v3",
    *,
    identity_mode: str = "minimal",
) -> tuple[dict[str, object], bytes]:
    lock_data = _canonical_json(
        _runtime_lock_value(profile_id)
    )
    lock = json.loads(lock_data)
    is_p6 = profile_id.startswith("aws-p6")
    identity = {
        "accountId": "123456789012",
        "architecture": "x86_64",
        "imageId": lock["ami_id"],
        "instanceId": "i-0123456789abcdef0",
        "privateIp": "10.1.2.3",
        "region": "us-east-1",
    }
    availability_zone = "us-east-1d" if is_p6 else "us-east-1c"
    if identity_mode == "full":
        identity.update(
            {
                "availabilityZone": availability_zone,
                "billingProducts": ["bp-6ba54002"],
                "devpayProductCodes": None,
                "instanceType": (
                    "p6-b300.48xlarge" if is_p6 else "p5.48xlarge"
                ),
                "kernelId": None,
                "marketplaceProductCodes": [],
                "pendingTime": "2026-07-24T06:00:00Z",
                "ramdiskId": None,
                "version": "2017-09-30",
            }
        )
    elif identity_mode != "minimal":  # pragma: no cover - helper guard
        raise AssertionError(identity_mode)
    environment = {
        "account_id": identity["accountId"],
        "ami_id": identity["imageId"],
        "aws_instance_identity_document": identity,
        "aws_instance_identity_pkcs7": base64.b64encode(
            b"synthetic-signed-instance-identity"
        ).decode("ascii"),
        "boot_id": "12345678-1234-4abc-8def-1234567890ab",
        "container_image": lock["container_image"],
        "container_image_digest": lock["container_image_digest"],
        "control_bundle_sha256": lock["control_bundle_sha256"],
        "instance_id": identity["instanceId"],
        "profile_sha256": lock["profile_sha256"],
        "provider": (
            "aws-p6-b300.48xlarge" if is_p6 else "aws-p5.48xlarge"
        ),
        "receipt_type": "memorysplit-aws-environment-v2",
        "region": identity["region"],
        "runtime_facts": dict(lock["versions"]),
        "runtime_lock_sha256": hashlib.sha256(lock_data).hexdigest(),
        "schema_version": 2,
        "source_commit": lock["source_commit"],
        "source_tree": lock["source_tree"],
    }
    environment_data = _canonical_json(environment)
    hardware = {
        "gpu_count": 8,
        "runtime_facts": dict(P6_SOFTWARE_FLOORS) if is_p6 else {},
    }
    phases = [
        {"name": name, "passed": True, "seconds": 1.0}
        for name in (
            "hardware",
            "nccl_all_reduce",
            "functional",
            "resume",
            "throughput_4x4",
            "s3_roundtrip",
        )
    ]
    canary = {
        "boot_id": environment["boot_id"],
        "container_image": environment["container_image"],
        "container_image_digest": environment["container_image_digest"],
        "dataset_build_id": "1" * 64,
        "dataset_receipt_sha256": "2" * 64,
        "ended_at": "2026-07-24T06:30:00Z",
        "environment_receipt_sha256": hashlib.sha256(
            environment_data
        ).hexdigest(),
        "functional": {"passed": True},
        "hardware": hardware,
        "instance_id": environment["instance_id"],
        "ordered_stream_sha256": "3" * 64,
        "passed": True,
        "phases": phases,
        "profile_sha256": environment["profile_sha256"],
        "provider": environment["provider"],
        "receipt_type": "memorysplit-aws-gpu-qualification-v1",
        "release_receipt_sha256": "4" * 64,
        "release_sha256": "5" * 64,
        "resume": {"passed": True},
        "run_manifest_sha256": "6" * 64,
        "runtime_lock_sha256": environment["runtime_lock_sha256"],
        "s3_roundtrip": {"passed": True},
        "schema_version": 1,
        "seed": 0,
        "source_commit": lock["source_commit"],
        "source_tree": lock["source_tree"],
        "started_at": "2026-07-24T06:00:00Z",
        "throughput_4x4": {"passed": True},
        "total_seconds": 1800.0,
    }
    canary_data = _canonical_json(canary)
    scope = {
        "account_id": environment["account_id"],
        "ami_id": environment["ami_id"],
        "boot_id": environment["boot_id"],
        "canary_receipt_sha256": hashlib.sha256(canary_data).hexdigest(),
        "container_facts_sha256": hashlib.sha256(
            _canonical_json(environment["runtime_facts"])
        ).hexdigest(),
        "container_image_digest": environment["container_image_digest"],
        "availability_zone": availability_zone,
        "environment_receipt_sha256": hashlib.sha256(
            environment_data
        ).hexdigest(),
        "host_facts_sha256": hashlib.sha256(
            _canonical_json(hardware)
        ).hexdigest(),
        "instance_id": environment["instance_id"],
        "profile_sha256": environment["profile_sha256"],
        "runtime_lock_sha256": environment["runtime_lock_sha256"],
    }
    approval = {
        "algorithm": "RSASSA_PSS_SHA_256",
        "public_key_sha256": TRUSTED_APPROVAL_PUBLIC_KEY_SHA256,
        "scope": scope,
        "scope_sha256": hashlib.sha256(_canonical_json(scope)).hexdigest(),
        "signature": base64.b64encode(b"signed-qualification-scope").decode(
            "ascii"
        ),
    }
    return {
        "approval": approval,
        "availability_zone": availability_zone,
        "canary_receipt": canary,
        "environment_receipt": environment,
        "receipt_type": "memorysplit-aws-qualified-runtime-v2",
        "schema_version": 2,
    }, lock_data


@pytest.mark.parametrize("identity_mode", ["minimal", "full"])
def test_qualification_accepts_realistic_aws_identity_documents(identity_mode):
    from msctl.aws_hardware import (
        parse_authenticated_qualification_evidence_bytes,
    )

    value, lock_data = _qualification_evidence_value(
        identity_mode=identity_mode
    )
    evidence = parse_authenticated_qualification_evidence_bytes(
        _canonical_json(value),
        profile_data=P6_PROFILE.read_bytes(),
        runtime_lock_data=lock_data,
        **_qualification_verification_kwargs(),
    )

    assert evidence.availability_zone == "us-east-1d"
    identity = value["environment_receipt"][
        "aws_instance_identity_document"
    ]
    if identity_mode == "minimal":
        assert set(identity) == {
            "accountId",
            "architecture",
            "imageId",
            "instanceId",
            "privateIp",
            "region",
        }
    else:
        assert "billingProducts" in identity
        assert "version" in identity


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("unknownField", "forbidden"),
        ("billingProducts", "not-a-list"),
        ("kernelId", 42),
        ("instanceType", "p5.48xlarge"),
    ],
)
def test_qualification_rejects_unknown_or_invalid_optional_identity(
    field,
    value,
):
    from msctl.aws_hardware import (
        parse_authenticated_qualification_evidence_bytes,
    )

    evidence, lock_data = _qualification_evidence_value(
        identity_mode="full"
    )
    evidence["environment_receipt"]["aws_instance_identity_document"][
        field
    ] = value

    with pytest.raises(ValueError, match="identity"):
        parse_authenticated_qualification_evidence_bytes(
            _canonical_json(evidence),
            profile_data=P6_PROFILE.read_bytes(),
            runtime_lock_data=lock_data,
            **_qualification_verification_kwargs(),
        )


def test_qualification_evidence_requires_identity_and_approval_verification():
    from msctl.aws_hardware import (
        parse_authenticated_qualification_evidence_bytes,
    )

    value, lock_data = _qualification_evidence_value()
    identity_verifier = _QualificationIdentityVerifier()
    approval_verifier = _QualificationApprovalVerifier()
    evidence = parse_authenticated_qualification_evidence_bytes(
        _canonical_json(value),
        profile_data=P6_PROFILE.read_bytes(),
        runtime_lock_data=lock_data,
        identity_verifier=identity_verifier,
        approval_verifier=approval_verifier,
        trusted_public_key_sha256=TRUSTED_APPROVAL_PUBLIC_KEY_SHA256,
    )

    assert evidence.account_id == "123456789012"
    assert evidence.instance_id == "i-0123456789abcdef0"
    assert evidence.boot_id == "12345678-1234-4abc-8def-1234567890ab"
    assert evidence.profile_sha256 == _sha256(P6_PROFILE)
    assert evidence.runtime_lock_sha256 == hashlib.sha256(lock_data).hexdigest()
    assert evidence.approval_public_key_sha256 == (
        TRUSTED_APPROVAL_PUBLIC_KEY_SHA256
    )
    assert evidence.identity_verified is True
    assert evidence.approval_verified is True
    assert len(identity_verifier.calls) == 1
    assert len(approval_verifier.calls) == 1
    assert approval_verifier.calls[0]["payload"] == _canonical_json(
        value["approval"]["scope"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "unsigned",
        "wrong-public-key",
        "wrong-instance",
        "wrong-boot",
        "wrong-profile",
        "wrong-runtime",
        "wrong-ami",
        "wrong-image",
    ],
)
def test_qualification_evidence_rejects_forgery_or_identity_drift(mutation):
    from msctl.aws_hardware import (
        parse_authenticated_qualification_evidence_bytes,
    )

    value, lock_data = _qualification_evidence_value()
    if mutation == "unsigned":
        value["approval"]["signature"] = ""
    elif mutation == "wrong-public-key":
        value["approval"]["public_key_sha256"] = "8" * 64
    elif mutation == "wrong-instance":
        value["canary_receipt"]["instance_id"] = "i-0fedcba9876543210"
    elif mutation == "wrong-boot":
        value["canary_receipt"]["boot_id"] = (
            "87654321-4321-4abc-8def-1234567890ab"
        )
    elif mutation == "wrong-profile":
        value["canary_receipt"]["profile_sha256"] = P5_PROFILE_SHA256
    elif mutation == "wrong-runtime":
        value["canary_receipt"]["runtime_lock_sha256"] = "7" * 64
    elif mutation == "wrong-ami":
        value["environment_receipt"]["ami_id"] = "ami-0fedcba9876543210"
        value["environment_receipt"]["aws_instance_identity_document"][
            "imageId"
        ] = "ami-0fedcba9876543210"
    elif mutation == "wrong-image":
        value["environment_receipt"]["container_image_digest"] = (
            "sha256:" + "7" * 64
        )
    else:  # pragma: no cover - parameter guard
        raise AssertionError(mutation)

    with pytest.raises(ValueError):
        parse_authenticated_qualification_evidence_bytes(
            _canonical_json(value),
            profile_data=P6_PROFILE.read_bytes(),
            runtime_lock_data=lock_data,
            identity_verifier=_QualificationIdentityVerifier(),
            approval_verifier=_QualificationApprovalVerifier(),
            trusted_public_key_sha256=TRUSTED_APPROVAL_PUBLIC_KEY_SHA256,
        )


@pytest.mark.parametrize(
    ("identity_valid", "approval_valid"),
    [(False, True), (True, False)],
    ids=["forged-instance-signature", "forged-approval-signature"],
)
def test_qualification_evidence_rejects_failed_crypto_verifier(
    identity_valid,
    approval_valid,
):
    from msctl.aws_hardware import (
        parse_authenticated_qualification_evidence_bytes,
    )

    value, lock_data = _qualification_evidence_value()
    with pytest.raises(ValueError, match="signature|verification|approval"):
        parse_authenticated_qualification_evidence_bytes(
            _canonical_json(value),
            profile_data=P6_PROFILE.read_bytes(),
            runtime_lock_data=lock_data,
            identity_verifier=_QualificationIdentityVerifier(identity_valid),
            approval_verifier=_QualificationApprovalVerifier(approval_valid),
            trusted_public_key_sha256=TRUSTED_APPROVAL_PUBLIC_KEY_SHA256,
        )


def _selection_value(
    profile_id: str = "aws-p6-b300.48xlarge-v3",
) -> dict[str, object]:
    if profile_id == "aws-p6-b300.48xlarge-v3":
        profile = {
            "path": "cluster/profiles/aws-p6-b300.48xlarge-v3.json",
            "profile_id": profile_id,
            "provider": "aws-p6-b300.48xlarge",
            "sha256": _sha256(P6_PROFILE),
        }
        placement = {
            "account_id": "123456789012",
            "availability_zone": "us-east-1d",
            "purchase_model": "on_demand",
            "region": "us-east-1",
        }
    elif profile_id == "aws-p5.48xlarge-v3":
        profile = {
            "path": "cluster/profiles/aws-p5.48xlarge-v3.json",
            "profile_id": profile_id,
            "provider": "aws-p5.48xlarge",
            "sha256": P5_PROFILE_SHA256,
        }
        placement = {
            "account_id": "123456789012",
            "availability_zone": "us-east-1c",
            "purchase_model": "on_demand",
            "region": "us-east-1",
        }
    else:  # pragma: no cover - test helper guard
        raise AssertionError(profile_id)
    runtime_evidence, runtime_lock_data = _qualification_evidence_value(
        profile_id
    )
    runtime_evidence_data = _canonical_json(runtime_evidence)
    return {
        "amendment": {
            "path": "configs/aws-hardware-amendment-v3.json",
            "sha256": AMENDMENT_SHA256,
        },
        "authority": {
            "local_path": SELECTION_LOCAL_PATH,
            "s3_key": SELECTION_S3_KEY,
        },
        "aws": placement,
        "cohort": {
            "arms": ["dense", "split90"],
            "cohort_assignment_sha256": COHORT_ASSIGNMENT_SHA256,
            "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "seeds": list(range(10)),
            "train_groups": [4, 4],
        },
        "profile": profile,
        "protected_outcomes_inspected": [],
        "receipt_type": "memorysplit-aws-provider-selection-v1",
        "replacement_policy": "forbidden",
        "runtime": {
            "ami_id": "ami-0123456789abcdef0",
            "container_image_digest": "sha256:" + "b" * 64,
            "runtime_evidence_sha256": hashlib.sha256(
                runtime_evidence_data
            ).hexdigest(),
            "runtime_lock_sha256": hashlib.sha256(
                runtime_lock_data
            ).hexdigest(),
        },
        "schema_version": 1,
        "selected_at": "2026-07-24T05:30:00Z",
        "selection_scope": "entire_cohort",
    }


def _canonical_json(value: object) -> bytes:
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


@pytest.mark.parametrize(
    "profile_id",
    ["aws-p5.48xlarge-v3", "aws-p6-b300.48xlarge-v3"],
)
def test_canonical_selection_receipt_binds_one_profile_and_entire_cohort(
    profile_id,
):
    from msctl.aws_hardware import parse_provider_selection_receipt_bytes

    selection, lock_data, evidence_data = _verified_selection_fixture(profile_id)
    data = _canonical_json(selection)
    receipt = parse_provider_selection_receipt_bytes(
        data,
        amendment_data=AMENDMENT.read_bytes(),
    )

    assert receipt.schema_version == 1
    assert receipt.receipt_type == "memorysplit-aws-provider-selection-v1"
    assert receipt.selection_scope == "entire_cohort"
    assert receipt.replacement_policy == "forbidden"
    assert receipt.authority_local_path == SELECTION_LOCAL_PATH
    assert receipt.authority_s3_key == SELECTION_S3_KEY
    assert receipt.amendment.path == "configs/aws-hardware-amendment-v3.json"
    assert receipt.amendment.sha256 == AMENDMENT_SHA256
    assert receipt.profile.profile_id == profile_id
    assert receipt.seeds == tuple(range(10))
    assert receipt.arms == ("dense", "split90")
    assert receipt.train_groups == (4, 4)
    assert receipt.preregistration_sha256 == PREREGISTRATION_SHA256
    assert receipt.cohort_assignment_sha256 == COHORT_ASSIGNMENT_SHA256
    assert receipt.runtime_lock_sha256 == hashlib.sha256(lock_data).hexdigest()
    assert receipt.runtime_evidence_sha256 == hashlib.sha256(
        evidence_data
    ).hexdigest()
    assert receipt.ami_id == "ami-0123456789abcdef0"
    assert receipt.container_image_digest == "sha256:" + "b" * 64
    assert receipt.account_id == "123456789012"
    assert receipt.region == "us-east-1"
    assert receipt.selected_at == "2026-07-24T05:30:00Z"
    assert receipt.protected_outcomes_inspected == ()
    assert receipt.sha256 == hashlib.sha256(data).hexdigest()
    if profile_id.startswith("aws-p6"):
        assert receipt.availability_zone == "us-east-1d"
        assert receipt.purchase_model == "on_demand"
    else:
        assert receipt.availability_zone == "us-east-1c"
        assert receipt.purchase_model == "on_demand"


def _mutate_selection(value: dict[str, object], mutation: str) -> None:
    if mutation == "bool-version":
        value["schema_version"] = True
    elif mutation == "unknown-field":
        value["unknown"] = True
    elif mutation == "wrong-amendment":
        value["amendment"]["sha256"] = "0" * 64
    elif mutation == "mixed-profile":
        value["profile"]["sha256"] = P5_PROFILE_SHA256
    elif mutation == "subset-seeds":
        value["cohort"]["seeds"] = list(range(9))
    elif mutation == "reordered-arms":
        value["cohort"]["arms"] = ["split90", "dense"]
    elif mutation == "asymmetric-groups":
        value["cohort"]["train_groups"] = [3, 5]
    elif mutation == "replacement":
        value["replacement_policy"] = "replace_allowed"
    elif mutation == "partial-scope":
        value["selection_scope"] = "seed"
    elif mutation == "outcomes":
        value["protected_outcomes_inspected"] = ["primary_endpoint"]
    elif mutation == "runtime-hash":
        value["runtime"]["runtime_lock_sha256"] = "A" * 64
    elif mutation == "mutable-ami":
        value["runtime"]["ami_id"] = "ami-latest"
    elif mutation == "image-digest":
        value["runtime"]["container_image_digest"] = "b" * 64
    elif mutation == "account":
        value["aws"]["account_id"] = "1234"
    elif mutation == "region":
        value["aws"]["region"] = "us-west-2"
        value["aws"]["availability_zone"] = "us-west-2a"
    elif mutation == "availability-zone":
        value["aws"]["availability_zone"] = "us-east-1c"
    elif mutation == "purchase-model":
        value["aws"]["purchase_model"] = "capacity_block"
    elif mutation == "offset-timestamp":
        value["selected_at"] = "2026-07-24T00:30:00-05:00"
    elif mutation == "invalid-timestamp":
        value["selected_at"] = "2026-02-30T05:30:00Z"
    else:  # pragma: no cover - parameterization guard
        raise AssertionError(mutation)


@pytest.mark.parametrize(
    "mutation",
    [
        "bool-version",
        "unknown-field",
        "wrong-amendment",
        "mixed-profile",
        "subset-seeds",
        "reordered-arms",
        "asymmetric-groups",
        "replacement",
        "partial-scope",
        "outcomes",
        "runtime-hash",
        "mutable-ami",
        "image-digest",
        "account",
        "region",
        "availability-zone",
        "purchase-model",
        "offset-timestamp",
        "invalid-timestamp",
    ],
)
def test_selection_receipt_rejects_drift_or_cross_profile_mixing(mutation):
    from msctl.aws_hardware import parse_provider_selection_receipt_bytes

    value = _selection_value()
    _mutate_selection(value, mutation)

    with pytest.raises(ValueError):
        parse_provider_selection_receipt_bytes(
            _canonical_json(value),
            amendment_data=AMENDMENT.read_bytes(),
        )


def test_selection_receipt_requires_its_sole_canonical_byte_encoding():
    from msctl.aws_hardware import parse_provider_selection_receipt_bytes

    pretty = (
        json.dumps(_selection_value(), indent=2, sort_keys=True) + "\n"
    ).encode("ascii")

    with pytest.raises(ValueError, match="canonical"):
        parse_provider_selection_receipt_bytes(
            pretty,
            amendment_data=AMENDMENT.read_bytes(),
        )


def test_selection_receipt_rejects_duplicate_keys():
    from msctl.aws_hardware import parse_provider_selection_receipt_bytes

    duplicate = _canonical_json(_selection_value()).replace(
        b'"schema_version":1',
        b'"schema_version":1,"schema_version":1',
        1,
    )

    with pytest.raises(ValueError, match="duplicate"):
        parse_provider_selection_receipt_bytes(
            duplicate,
            amendment_data=AMENDMENT.read_bytes(),
        )


def test_arbitrary_provider_selection_writer_is_not_public():
    import msctl.aws_hardware as hardware

    assert not hasattr(hardware, "write_provider_selection_receipt")
    assert not hasattr(hardware, "write_aws_provider_selection_receipt")


def test_resume_binding_rejects_cross_profile_or_runtime_tuple_mixing(tmp_path):
    from msctl.aws_hardware import (
        publish_provider_selection,
        validate_resume_hardware_binding,
    )

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    authority_root = tmp_path / "authority"
    store = _MemorySelectionStore()
    published = publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        selection_data=selection_data,
        store=store,
        **_qualification_verification_kwargs(),
    )
    receipt = published.selection
    binding = {
        "account_id": receipt.account_id,
        "amendment_sha256": receipt.amendment.sha256,
        "arm": "split90",
        "authority_root": authority_root,
        "boot_id": receipt.qualification_boot_id,
        "expected_selection_version_id": published.remote.version_id,
        "instance_id": receipt.qualification_instance_id,
        "profile_sha256": receipt.profile.sha256,
        "provider_selection_sha256": receipt.sha256,
        "repo_root": ROOT,
        "runtime_evidence_path": runtime_evidence,
        "runtime_evidence_sha256": receipt.runtime_evidence_sha256,
        "runtime_lock_path": runtime_lock,
        "runtime_lock_sha256": receipt.runtime_lock_sha256,
        "seed": 9,
        "store": store,
        **_qualification_verification_kwargs(),
    }

    validate_resume_hardware_binding(**binding)

    for field, changed in (
        ("profile_sha256", P5_PROFILE_SHA256),
        ("runtime_lock_sha256", "c" * 64),
        ("runtime_evidence_sha256", "d" * 64),
    ):
        mixed = dict(binding)
        mixed[field] = changed
        with pytest.raises(ValueError, match="profile|runtime"):
            validate_resume_hardware_binding(**mixed)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("seed", 10),
        ("arm", "control"),
        ("profile_sha256", P5_PROFILE_SHA256),
        ("runtime_lock_sha256", "2" * 64),
        ("runtime_evidence_sha256", "3" * 64),
        ("amendment_sha256", "0" * 64),
        ("provider_selection_sha256", "1" * 64),
    ],
)
def test_resume_binding_rejects_cohort_or_placement_mixing(
    tmp_path,
    field,
    changed,
):
    from msctl.aws_hardware import (
        publish_provider_selection,
        validate_resume_hardware_binding,
    )
    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    authority_root = tmp_path / "authority"
    store = _MemorySelectionStore()
    published = publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        selection_data=selection_data,
        store=store,
        **_qualification_verification_kwargs(),
    )
    receipt = published.selection
    binding = {
        "account_id": receipt.account_id,
        "amendment_sha256": receipt.amendment.sha256,
        "arm": "dense",
        "authority_root": authority_root,
        "boot_id": receipt.qualification_boot_id,
        "expected_selection_version_id": published.remote.version_id,
        "instance_id": receipt.qualification_instance_id,
        "profile_sha256": receipt.profile.sha256,
        "provider_selection_sha256": receipt.sha256,
        "repo_root": ROOT,
        "runtime_evidence_path": runtime_evidence,
        "runtime_evidence_sha256": receipt.runtime_evidence_sha256,
        "runtime_lock_path": runtime_lock,
        "runtime_lock_sha256": receipt.runtime_lock_sha256,
        "seed": 0,
        "store": store,
        **_qualification_verification_kwargs(),
    }
    binding[field] = changed

    with pytest.raises(ValueError):
        validate_resume_hardware_binding(**binding)


class _MemorySelectionStore:
    def __init__(
        self,
        *,
        lost_put: bool = False,
        corrupt_head: str | None = None,
        history_versions: tuple[str, ...] = (),
        delete_markers: tuple[str, ...] = (),
        post_put_versions: tuple[str, ...] = (),
        post_put_delete_markers: tuple[str, ...] = (),
        fail_at: str | None = None,
    ) -> None:
        self.lost_put = lost_put
        self.corrupt_head = corrupt_head
        self.history_versions = list(history_versions)
        self.delete_markers = list(delete_markers)
        self.post_put_versions = list(post_put_versions)
        self.post_put_delete_markers = list(post_put_delete_markers)
        self.fail_at = fail_at
        self.list_count = 0
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.list_calls: list[str] = []
        self.put_calls: list[dict[str, object]] = []
        self.head_calls: list[dict[str, object]] = []
        self.get_calls: list[dict[str, str]] = []

    def list_versions(self, *, key: str):
        from msctl.aws_hardware import VersionedSelectionHistory

        self.list_calls.append(key)
        self.list_count += 1
        if self.fail_at == f"list-{self.list_count}":
            self.fail_at = None
            raise TimeoutError("synthetic list timeout")
        return VersionedSelectionHistory(
            key=key,
            versions=tuple(self.history_versions),
            delete_markers=tuple(self.delete_markers),
        )

    def put_if_none_match(
        self,
        *,
        key: str,
        data: bytes,
        if_none_match: str,
        checksum_sha256: str,
    ):
        from msctl.aws_hardware import VersionedSelectionObject

        self.put_calls.append(
            {
                "checksum_sha256": checksum_sha256,
                "if_none_match": if_none_match,
                "key": key,
            }
        )
        if key in self.objects:
            return None
        version_id = "version-1"
        self.objects[key] = (data, version_id)
        self.history_versions.append(version_id)
        self.history_versions.extend(self.post_put_versions)
        self.delete_markers.extend(self.post_put_delete_markers)
        if self.fail_at == "put-after-write":
            self.fail_at = None
            raise TimeoutError("synthetic lost PUT response")
        if self.lost_put:
            return None
        return VersionedSelectionObject(
            key=key,
            sha256=hashlib.sha256(data).hexdigest(),
            bytes=len(data),
            version_id=version_id,
        )

    def head(self, *, key: str, version_id: str | None):
        from msctl.aws_hardware import VersionedSelectionObject

        self.head_calls.append({"key": key, "version_id": version_id})
        if self.fail_at == "head":
            self.fail_at = None
            raise TimeoutError("synthetic HEAD timeout")
        stored = self.objects.get(key)
        if stored is None:
            return None
        data, stored_version = stored
        if version_id is not None and version_id != stored_version:
            return None
        sha256 = hashlib.sha256(data).hexdigest()
        byte_count = len(data)
        result_version = stored_version
        if self.corrupt_head == "checksum":
            sha256 = "0" * 64
        elif self.corrupt_head == "bytes":
            byte_count += 1
        elif self.corrupt_head == "version":
            result_version = ""
        return VersionedSelectionObject(
            key=key,
            sha256=sha256,
            bytes=byte_count,
            version_id=result_version,
        )

    def get_exact(self, *, key: str, version_id: str):
        from msctl.aws_hardware import (
            VersionedSelectionObject,
            VersionedSelectionRead,
        )

        self.get_calls.append({"key": key, "version_id": version_id})
        stored = self.objects.get(key)
        if stored is None or stored[1] != version_id:
            return None
        data, stored_version = stored
        return VersionedSelectionRead(
            data=data,
            object=VersionedSelectionObject(
                key=key,
                sha256=hashlib.sha256(data).hexdigest(),
                bytes=len(data),
                version_id=stored_version,
            ),
        )


class _AwsSelectionRunner:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[tuple[list[str], dict[str, str], float]] = []

    def __call__(self, argv, environment, timeout_seconds):
        rendered = list(argv)
        self.calls.append(
            (rendered, dict(environment), float(timeout_seconds))
        )
        operation = rendered[rendered.index("s3api") + 1]
        checksum = base64.b64encode(
            hashlib.sha256(self.payload).digest()
        ).decode("ascii")
        if operation == "list-object-versions":
            value = {"history": {"delete_markers": [], "versions": []}}
        elif operation == "put-object":
            body = Path(rendered[rendered.index("--body") + 1])
            assert body.read_bytes() == self.payload
            value = {
                "object": {
                    "checksum_sha256": checksum,
                    "version_id": "version-7",
                }
            }
        elif operation == "head-object":
            value = {
                "object": {
                    "checksum_sha256": checksum,
                    "content_length": len(self.payload),
                    "version_id": "version-7",
                }
            }
        elif operation == "get-object":
            destination = Path(rendered[rendered.index("--output") - 1])
            destination.write_bytes(self.payload)
            value = {
                "object": {
                    "checksum_sha256": checksum,
                    "content_length": len(self.payload),
                    "version_id": "version-7",
                }
            }
        else:  # pragma: no cover - fake guard
            raise AssertionError(operation)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(value, separators=(",", ":")),
            stderr="",
        )


def test_aws_cli_versioned_selection_store_uses_exact_fixed_key_commands(
    tmp_path,
):
    from msctl.aws_hardware import (
        AwsCliVersionedSelectionStore,
        PROVIDER_SELECTION_S3_KEY,
    )

    payload = _canonical_json(_selection_value())
    runner = _AwsSelectionRunner(payload)
    store = AwsCliVersionedSelectionStore(
        bucket="memorysplit-authority",
        region="us-east-1",
        environment={"AWS_REGION": "us-east-1", "LANG": "C"},
        staging_root=tmp_path,
        runner=runner,
        timeout_seconds=17,
    )
    history = store.list_versions(key=PROVIDER_SELECTION_S3_KEY)
    uploaded = store.put_if_none_match(
        key=PROVIDER_SELECTION_S3_KEY,
        data=payload,
        if_none_match="*",
        checksum_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert uploaded is not None
    headed = store.head(
        key=PROVIDER_SELECTION_S3_KEY,
        version_id=uploaded.version_id,
    )
    downloaded = store.get_exact(
        key=PROVIDER_SELECTION_S3_KEY,
        version_id=uploaded.version_id,
    )

    assert history.versions == ()
    assert history.delete_markers == ()
    assert headed == uploaded
    assert downloaded is not None
    assert downloaded.data == payload
    assert downloaded.object == uploaded
    commands = [argv for argv, _environment, _timeout in runner.calls]
    assert [
        argv[argv.index("s3api") + 1] for argv in commands
    ] == [
        "list-object-versions",
        "put-object",
        "head-object",
        "get-object",
    ]
    assert all(
        argv[argv.index("--bucket") + 1] == "memorysplit-authority"
        and argv[argv.index("--key") + 1] == PROVIDER_SELECTION_S3_KEY
        for argv in commands[1:]
    )
    assert commands[1][commands[1].index("--if-none-match") + 1] == "*"
    assert commands[2][commands[2].index("--version-id") + 1] == "version-7"
    assert commands[3][commands[3].index("--version-id") + 1] == "version-7"
    assert all(timeout == 17 for _argv, _environment, timeout in runner.calls)


@pytest.mark.parametrize(
    ("history_versions", "delete_markers"),
    [(("old-version",), ()), ((), ("delete-marker-1",))],
    ids=["historical-version", "delete-marker"],
)
def test_selection_publication_blocks_any_fixed_key_history(
    tmp_path,
    history_versions,
    delete_markers,
):
    from msctl.aws_hardware import (
        PROVIDER_SELECTION_LOCAL_PATH,
        publish_provider_selection,
    )

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    store = _MemorySelectionStore(
        history_versions=history_versions,
        delete_markers=delete_markers,
    )

    with pytest.raises(ValueError, match="history|version|delete"):
        publish_provider_selection(
            authority_root=tmp_path / "authority",
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            selection_data=selection_data,
            store=store,
            **_qualification_verification_kwargs(),
        )
    assert store.put_calls == []
    assert not (
        tmp_path / "authority" / PROVIDER_SELECTION_LOCAL_PATH
    ).exists()


@pytest.mark.parametrize(
    ("post_put_versions", "post_put_delete_markers"),
    [(("concurrent-version",), ()), ((), ("concurrent-delete",))],
    ids=["concurrent-version", "concurrent-delete-marker"],
)
def test_selection_publication_rechecks_singleton_history_after_put(
    tmp_path,
    post_put_versions,
    post_put_delete_markers,
):
    from msctl.aws_hardware import publish_provider_selection

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    store = _MemorySelectionStore(
        post_put_versions=post_put_versions,
        post_put_delete_markers=post_put_delete_markers,
    )

    with pytest.raises(ValueError, match="history|version|delete"):
        publish_provider_selection(
            authority_root=tmp_path / "authority",
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            selection_data=selection_data,
            store=store,
            **_qualification_verification_kwargs(),
        )
    assert store.list_calls == [SELECTION_S3_KEY, SELECTION_S3_KEY]


@pytest.mark.parametrize("race", ["version", "delete-marker"])
def test_replay_and_admission_recheck_singleton_history(tmp_path, race):
    from msctl.aws_hardware import (
        admit_provider_selection,
        load_versioned_provider_selection_authority,
        publish_provider_selection,
    )

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    authority_root = tmp_path / "authority-history"
    store = _MemorySelectionStore()
    published = publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        selection_data=selection_data,
        store=store,
        **_qualification_verification_kwargs(),
    )
    if race == "version":
        store.history_versions.append("concurrent-version")
    else:
        store.delete_markers.append("concurrent-delete")

    with pytest.raises(ValueError, match="history|version|delete"):
        load_versioned_provider_selection_authority(
            authority_root=authority_root,
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            store=store,
            **_qualification_verification_kwargs(),
        )
    with pytest.raises(ValueError, match="history|version|delete"):
        admit_provider_selection(
            authority_root=authority_root,
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            store=store,
            account_id=published.selection.account_id,
            instance_id=published.selection.qualification_instance_id,
            boot_id=published.selection.qualification_boot_id,
            seed=0,
            arm="dense",
            expected_selection_version_id=published.remote.version_id,
            **_qualification_verification_kwargs(),
        )


def test_pending_intent_is_durable_before_any_remote_mutation(tmp_path):
    from msctl.aws_hardware import (
        PROVIDER_SELECTION_PENDING_ARCHIVE_LOCAL_PATH,
        PROVIDER_SELECTION_PENDING_LOCAL_PATH,
        publish_provider_selection,
    )

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    authority_root = tmp_path / "authority-pending"
    store = _MemorySelectionStore(fail_at="list-1")

    with pytest.raises(ValueError, match="uncertain|publication|timeout"):
        publish_provider_selection(
            authority_root=authority_root,
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            selection_data=selection_data,
            store=store,
            **_qualification_verification_kwargs(),
        )
    pending = authority_root / PROVIDER_SELECTION_PENDING_LOCAL_PATH
    assert json.loads(pending.read_bytes()) == {
        "expected_bytes": len(selection_data),
        "s3_key": SELECTION_S3_KEY,
        "schema_version": 1,
        "selection_sha256": hashlib.sha256(selection_data).hexdigest(),
    }
    assert store.put_calls == []

    published = publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        selection_data=selection_data,
        store=store,
        **_qualification_verification_kwargs(),
    )
    assert published.publication_state == "published"
    assert not pending.exists()
    assert (
        authority_root / PROVIDER_SELECTION_PENDING_ARCHIVE_LOCAL_PATH
    ).is_file()


@pytest.mark.parametrize(
    "fail_at",
    ["put-after-write", "head", "list-2"],
)
def test_retry_recovers_single_exact_remote_version_after_timeout(
    tmp_path,
    fail_at,
):
    from msctl.aws_hardware import (
        PROVIDER_SELECTION_PENDING_LOCAL_PATH,
        publish_provider_selection,
    )

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    authority_root = tmp_path / f"authority-{fail_at}"
    store = _MemorySelectionStore(fail_at=fail_at)

    with pytest.raises(ValueError, match="uncertain|publication|timeout"):
        publish_provider_selection(
            authority_root=authority_root,
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            selection_data=selection_data,
            store=store,
            **_qualification_verification_kwargs(),
        )
    assert (
        authority_root / PROVIDER_SELECTION_PENDING_LOCAL_PATH
    ).is_file()

    recovered = publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        selection_data=selection_data,
        store=store,
        **_qualification_verification_kwargs(),
    )
    assert recovered.publication_state == "recovered"
    assert recovered.remote.version_id == "version-1"
    assert len(store.put_calls) == 1
    assert store.get_calls[-1] == {
        "key": SELECTION_S3_KEY,
        "version_id": "version-1",
    }


@pytest.mark.parametrize(
    "boundary",
    [
        "local-selection",
        "version-binding",
        "pending-archive",
        "pending-archive-midway",
    ],
)
def test_retry_recovers_after_each_local_commit_boundary(
    tmp_path,
    monkeypatch,
    boundary,
):
    import msctl.aws_hardware as hardware

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    authority_root = tmp_path / f"authority-{boundary}"
    store = _MemorySelectionStore()
    target_name = {
        "local-selection": "_publish_local_selection",
        "version-binding": "_publish_version_binding",
        "pending-archive": "_archive_pending_selection",
        "pending-archive-midway": "_archive_pending_selection",
    }[boundary]
    original = getattr(hardware, target_name, None)
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            if boundary == "pending-archive-midway":
                pending = Path(authority_root) / (
                    hardware.PROVIDER_SELECTION_PENDING_LOCAL_PATH
                )
                archive = Path(authority_root) / (
                    hardware.PROVIDER_SELECTION_PENDING_ARCHIVE_LOCAL_PATH
                )
                pending.rename(archive)
            raise TimeoutError(f"synthetic crash at {boundary}")
        assert original is not None
        return original(*args, **kwargs)

    monkeypatch.setattr(hardware, target_name, fail_once, raising=False)

    with pytest.raises(ValueError, match="uncertain|publication|crash"):
        hardware.publish_provider_selection(
            authority_root=authority_root,
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            selection_data=selection_data,
            store=store,
            **_qualification_verification_kwargs(),
        )
    recovered = hardware.publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        selection_data=selection_data,
        store=store,
        **_qualification_verification_kwargs(),
    )

    assert recovered.publication_state == "recovered"
    assert len(store.put_calls) == 1


@pytest.mark.parametrize("conflict", ["multiple-versions", "delete-marker"])
def test_pending_recovery_rejects_conflicting_history(tmp_path, conflict):
    from msctl.aws_hardware import publish_provider_selection

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    authority_root = tmp_path / f"authority-{conflict}"
    store = _MemorySelectionStore(fail_at="list-1")
    with pytest.raises(ValueError):
        publish_provider_selection(
            authority_root=authority_root,
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            selection_data=selection_data,
            store=store,
            **_qualification_verification_kwargs(),
        )
    if conflict == "multiple-versions":
        store.history_versions.extend(["version-a", "version-b"])
    else:
        store.delete_markers.append("delete-a")

    with pytest.raises(ValueError, match="history|version|delete|conflict"):
        publish_provider_selection(
            authority_root=authority_root,
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            selection_data=selection_data,
            store=store,
            **_qualification_verification_kwargs(),
        )
    assert store.put_calls == []


def test_preexisting_single_version_never_becomes_retry_recoverable(tmp_path):
    from msctl.aws_hardware import publish_provider_selection

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    authority_root = tmp_path / "authority-preexisting"
    store = _MemorySelectionStore(history_versions=("preexisting",))
    store.objects[SELECTION_S3_KEY] = (selection_data, "preexisting")

    for _attempt in range(2):
        with pytest.raises(ValueError, match="history|conflict|version"):
            publish_provider_selection(
                authority_root=authority_root,
                repo_root=ROOT,
                runtime_lock_path=runtime_lock,
                runtime_evidence_path=runtime_evidence,
                selection_data=selection_data,
                store=store,
                **_qualification_verification_kwargs(),
            )
    assert store.put_calls == []
    assert store.get_calls == []


def test_completion_record_binds_remote_local_version_and_intents(tmp_path):
    from msctl.aws_hardware import (
        PROVIDER_SELECTION_COMPLETION_LOCAL_PATH,
        PROVIDER_SELECTION_MUTATION_ARCHIVE_LOCAL_PATH,
        PROVIDER_SELECTION_PENDING_ARCHIVE_LOCAL_PATH,
        PROVIDER_SELECTION_VERSION_LOCAL_PATH,
        publish_provider_selection,
    )

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    authority_root = tmp_path / "authority-completion"
    store = _MemorySelectionStore()
    published = publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        selection_data=selection_data,
        store=store,
        **_qualification_verification_kwargs(),
    )
    version_data = (
        authority_root / PROVIDER_SELECTION_VERSION_LOCAL_PATH
    ).read_bytes()
    pending_archive_data = (
        authority_root / PROVIDER_SELECTION_PENDING_ARCHIVE_LOCAL_PATH
    ).read_bytes()
    mutation_archive_data = (
        authority_root / PROVIDER_SELECTION_MUTATION_ARCHIVE_LOCAL_PATH
    ).read_bytes()
    history_data = _canonical_json(
        {
            "delete_markers": [],
            "key": SELECTION_S3_KEY,
            "versions": [published.remote.version_id],
        }
    )
    completion = json.loads(
        (
            authority_root / PROVIDER_SELECTION_COMPLETION_LOCAL_PATH
        ).read_bytes()
    )

    assert completion == {
        "expected_bytes": len(selection_data),
        "history_commitment_sha256": hashlib.sha256(
            history_data
        ).hexdigest(),
        "local_selection_sha256": hashlib.sha256(selection_data).hexdigest(),
        "mutation_intent_sha256": hashlib.sha256(
            mutation_archive_data
        ).hexdigest(),
        "pending_intent_sha256": hashlib.sha256(
            pending_archive_data
        ).hexdigest(),
        "remote_version_id": published.remote.version_id,
        "s3_key": SELECTION_S3_KEY,
        "schema_version": 1,
        "selection_sha256": published.selection.sha256,
        "version_binding_sha256": hashlib.sha256(version_data).hexdigest(),
    }


@pytest.mark.parametrize(
    "boundary",
    [
        "before-completion",
        "after-completion",
        "after-first-archive",
        "after-second-archive",
        "after-final-response",
    ],
)
def test_terminal_completion_crashes_recover_identical_selection(
    tmp_path,
    monkeypatch,
    boundary,
):
    import msctl.aws_hardware as hardware

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    authority_root = tmp_path / f"authority-{boundary}"
    store = _MemorySelectionStore()
    failed = False
    original_completion = getattr(hardware, "_publish_completion_record", None)
    original_archive = getattr(hardware, "_archive_pending_selection")

    if boundary in {"before-completion", "after-completion"}:
        def completion_once(*args, **kwargs):
            nonlocal failed
            if not failed:
                failed = True
                if boundary == "after-completion":
                    assert original_completion is not None
                    original_completion(*args, **kwargs)
                raise TimeoutError(f"synthetic crash {boundary}")
            assert original_completion is not None
            return original_completion(*args, **kwargs)

        monkeypatch.setattr(
            hardware,
            "_publish_completion_record",
            completion_once,
            raising=False,
        )
    elif boundary in {"after-first-archive", "after-second-archive"}:
        def archive_once(*args, **kwargs):
            nonlocal failed
            if not failed:
                failed = True
                if boundary == "after-first-archive":
                    pending = Path(authority_root) / (
                        hardware.PROVIDER_SELECTION_PENDING_LOCAL_PATH
                    )
                    archive = Path(authority_root) / (
                        hardware.PROVIDER_SELECTION_PENDING_ARCHIVE_LOCAL_PATH
                    )
                    pending.rename(archive)
                else:
                    original_archive(*args, **kwargs)
                raise TimeoutError(f"synthetic crash {boundary}")
            return original_archive(*args, **kwargs)

        monkeypatch.setattr(hardware, "_archive_pending_selection", archive_once)

    if boundary == "after-final-response":
        first = hardware.publish_provider_selection(
            authority_root=authority_root,
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            selection_data=selection_data,
            store=store,
            **_qualification_verification_kwargs(),
        )
        assert first.publication_state == "published"
    else:
        with pytest.raises(ValueError, match="uncertain|completion|crash"):
            hardware.publish_provider_selection(
                authority_root=authority_root,
                repo_root=ROOT,
                runtime_lock_path=runtime_lock,
                runtime_evidence_path=runtime_evidence,
                selection_data=selection_data,
                store=store,
                **_qualification_verification_kwargs(),
            )

    recovered = hardware.publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        selection_data=selection_data,
        store=store,
        **_qualification_verification_kwargs(),
    )
    assert recovered.publication_state == "recovered"
    assert len(store.put_calls) == 1


@pytest.mark.parametrize(
    "tamper",
    [
        "missing-completion",
        "forged-completion",
        "missing-pending-archive",
        "missing-mutation-archive",
        "forged-pending-archive",
        "local-byte-drift",
        "version-binding-drift",
    ],
)
def test_admission_rejects_partial_or_forged_completion(tmp_path, tamper):
    from msctl.aws_hardware import (
        PROVIDER_SELECTION_COMPLETION_LOCAL_PATH,
        PROVIDER_SELECTION_MUTATION_ARCHIVE_LOCAL_PATH,
        PROVIDER_SELECTION_PENDING_ARCHIVE_LOCAL_PATH,
        load_versioned_provider_selection_authority,
        publish_provider_selection,
    )

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    authority_root = tmp_path / f"authority-{tamper}"
    store = _MemorySelectionStore()
    publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        selection_data=selection_data,
        store=store,
        **_qualification_verification_kwargs(),
    )
    completion_path = authority_root / PROVIDER_SELECTION_COMPLETION_LOCAL_PATH
    if tamper == "missing-completion":
        completion_path.unlink()
    elif tamper == "forged-completion":
        completion = json.loads(completion_path.read_bytes())
        completion["remote_version_id"] = "forged-version"
        completion_path.write_bytes(_canonical_json(completion))
        completion_path.chmod(0o600)
    elif tamper == "missing-pending-archive":
        (
            authority_root / PROVIDER_SELECTION_PENDING_ARCHIVE_LOCAL_PATH
        ).unlink()
    else:
        if tamper == "missing-mutation-archive":
            (
                authority_root
                / PROVIDER_SELECTION_MUTATION_ARCHIVE_LOCAL_PATH
            ).unlink()
        elif tamper == "forged-pending-archive":
            path = (
                authority_root
                / PROVIDER_SELECTION_PENDING_ARCHIVE_LOCAL_PATH
            )
            path.write_bytes(_canonical_json({"forged": True}))
            path.chmod(0o600)
        elif tamper == "local-byte-drift":
            path = authority_root / SELECTION_LOCAL_PATH
            path.write_bytes(b'{"drift":true}\n')
            path.chmod(0o600)
        elif tamper == "version-binding-drift":
            path = authority_root / (
                "memorysplit-confirmatory-v3-360m-n10-aws/"
                "provider-selection-version.json"
            )
            version = json.loads(path.read_bytes())
            version["version_id"] = "drift-version"
            path.write_bytes(_canonical_json(version))
            path.chmod(0o600)
        else:  # pragma: no cover - parameter guard
            raise AssertionError(tamper)

    with pytest.raises(ValueError, match="completion|archive|authority"):
        load_versioned_provider_selection_authority(
            authority_root=authority_root,
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            store=store,
            **_qualification_verification_kwargs(),
        )


def test_published_version_is_persisted_and_required_for_exact_replay(tmp_path):
    from msctl.aws_hardware import (
        PROVIDER_SELECTION_VERSION_LOCAL_PATH,
        load_versioned_provider_selection_authority,
        publish_provider_selection,
    )

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    authority_root = tmp_path / "authority"
    store = _MemorySelectionStore()
    published = publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        selection_data=selection_data,
        store=store,
        **_qualification_verification_kwargs(),
    )
    version_path = authority_root / PROVIDER_SELECTION_VERSION_LOCAL_PATH
    version_receipt = json.loads(version_path.read_bytes())

    assert version_receipt == {
        "schema_version": 1,
        "selection_sha256": published.selection.sha256,
        "s3_key": published.remote.key,
        "version_id": published.remote.version_id,
    }
    replayed = load_versioned_provider_selection_authority(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        store=store,
        **_qualification_verification_kwargs(),
    )
    assert replayed == published.selection
    assert len(store.get_calls) == 3
    assert all(
        call
        == {
            "key": published.remote.key,
            "version_id": published.remote.version_id,
        }
        for call in store.get_calls
    )

    version_receipt["version_id"] = "other-version"
    version_path.write_bytes(_canonical_json(version_receipt))
    version_path.chmod(0o600)
    with pytest.raises(ValueError, match="version|GET|replay"):
        load_versioned_provider_selection_authority(
            authority_root=authority_root,
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            store=store,
            **_qualification_verification_kwargs(),
        )


def _write_authority_inputs(
    tmp_path: Path,
    profile_id: str = "aws-p6-b300.48xlarge-v3",
) -> tuple[bytes, Path, Path]:
    selection, lock_data, evidence_data = _verified_selection_fixture(profile_id)
    runtime_lock = tmp_path / f"{profile_id}-runtime-lock.json"
    runtime_evidence = tmp_path / f"{profile_id}-runtime-evidence.json"
    runtime_lock.write_bytes(lock_data)
    runtime_evidence.write_bytes(evidence_data)
    runtime_lock.chmod(0o600)
    runtime_evidence.chmod(0o600)
    return _canonical_json(selection), runtime_lock, runtime_evidence


def test_selection_authority_paths_are_fixed_and_cohort_specific():
    from msctl.aws_hardware import (
        PROVIDER_SELECTION_LOCAL_PATH,
        PROVIDER_SELECTION_S3_KEY,
    )

    assert PROVIDER_SELECTION_LOCAL_PATH == SELECTION_LOCAL_PATH
    assert PROVIDER_SELECTION_S3_KEY == SELECTION_S3_KEY
    assert "memorysplit-confirmatory-v3-360m-n10-aws" in (
        PROVIDER_SELECTION_LOCAL_PATH
    )
    assert "memorysplit-confirmatory-v3-360m-n10-aws" in (
        PROVIDER_SELECTION_S3_KEY
    )
    assert "sha256" not in PROVIDER_SELECTION_LOCAL_PATH
    assert "sha256" not in PROVIDER_SELECTION_S3_KEY


def test_public_contract_dataclasses_are_data_only():
    from msctl.aws_hardware import (
        AuthenticatedSelectionBinding,
        ArtifactBinding,
        AwsHardwareAmendment,
        AwsProviderSelectionReceipt,
        AwsRuntimeEvidence,
        AwsRuntimeLock,
        HardwareProfileBinding,
        PublishedProviderSelection,
        VersionedSelectionObject,
        VersionedSelectionHistory,
        VersionedSelectionRead,
    )

    for contract in (
        AuthenticatedSelectionBinding,
        ArtifactBinding,
        AwsHardwareAmendment,
        AwsProviderSelectionReceipt,
        AwsRuntimeEvidence,
        AwsRuntimeLock,
        HardwareProfileBinding,
        PublishedProviderSelection,
        VersionedSelectionObject,
        VersionedSelectionHistory,
        VersionedSelectionRead,
    ):
        authority_methods = [
            name
            for name, value in contract.__dict__.items()
            if not name.startswith("__") and callable(value)
        ]
        assert authority_methods == []


def test_fixed_authority_publishes_local_and_versioned_store_once(tmp_path):
    from msctl.aws_hardware import (
        PROVIDER_SELECTION_LOCAL_PATH,
        PROVIDER_SELECTION_S3_KEY,
        publish_provider_selection,
    )

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    authority_root = tmp_path / "authority"
    store = _MemorySelectionStore()
    published = publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        selection_data=selection_data,
        store=store,
        **_qualification_verification_kwargs(),
    )
    local_path = authority_root / PROVIDER_SELECTION_LOCAL_PATH

    assert local_path.read_bytes() == selection_data
    assert local_path.stat().st_mode & 0o777 == 0o600
    assert published.local_path == local_path
    assert published.remote.key == PROVIDER_SELECTION_S3_KEY
    assert set(store.objects) == {PROVIDER_SELECTION_S3_KEY}
    assert store.put_calls == [
        {
            "checksum_sha256": hashlib.sha256(selection_data).hexdigest(),
            "if_none_match": "*",
            "key": PROVIDER_SELECTION_S3_KEY,
        }
    ]
    assert store.head_calls == [
        {"key": PROVIDER_SELECTION_S3_KEY, "version_id": "version-1"},
        {"key": PROVIDER_SELECTION_S3_KEY, "version_id": "version-1"},
    ]

    with pytest.raises(TypeError):
        publish_provider_selection(
            authority_root=authority_root,
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            selection_data=selection_data,
            store=store,
            path=tmp_path / "alternative.json",
            **_qualification_verification_kwargs(),
        )


def test_fixed_authority_rejects_conflicting_local_profile_selection(tmp_path):
    from msctl.aws_hardware import (
        PROVIDER_SELECTION_LOCAL_PATH,
        publish_provider_selection,
    )

    authority_root = tmp_path / "authority"
    store = _MemorySelectionStore()
    p6_data, p6_lock, p6_evidence = _write_authority_inputs(tmp_path, "aws-p6-b300.48xlarge-v3")
    publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=p6_lock,
        runtime_evidence_path=p6_evidence,
        selection_data=p6_data,
        store=store,
        **_qualification_verification_kwargs(),
    )
    p5_data, p5_lock, p5_evidence = _write_authority_inputs(tmp_path, "aws-p5.48xlarge-v3")

    with pytest.raises(ValueError, match="conflict|different"):
        publish_provider_selection(
            authority_root=authority_root,
            repo_root=ROOT,
            runtime_lock_path=p5_lock,
            runtime_evidence_path=p5_evidence,
            selection_data=p5_data,
            store=store,
            **_qualification_verification_kwargs(),
        )
    assert (authority_root / PROVIDER_SELECTION_LOCAL_PATH).read_bytes() == p6_data


def test_local_conflict_blocks_publication_to_an_empty_remote_store(tmp_path):
    from msctl.aws_hardware import (
        PROVIDER_SELECTION_S3_KEY,
        publish_provider_selection,
    )

    authority_root = tmp_path / "authority"
    p6_data, p6_lock, p6_evidence = _write_authority_inputs(
        tmp_path, "aws-p6-b300.48xlarge-v3"
    )
    publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=p6_lock,
        runtime_evidence_path=p6_evidence,
        selection_data=p6_data,
        store=_MemorySelectionStore(),
        **_qualification_verification_kwargs(),
    )
    p5_data, p5_lock, p5_evidence = _write_authority_inputs(
        tmp_path, "aws-p5.48xlarge-v3"
    )
    empty_store = _MemorySelectionStore()

    with pytest.raises(ValueError, match="conflict|different"):
        publish_provider_selection(
            authority_root=authority_root,
            repo_root=ROOT,
            runtime_lock_path=p5_lock,
            runtime_evidence_path=p5_evidence,
            selection_data=p5_data,
            store=empty_store,
            **_qualification_verification_kwargs(),
        )
    assert PROVIDER_SELECTION_S3_KEY not in empty_store.objects


def test_fixed_store_recovers_only_exact_lost_put(tmp_path):
    from msctl.aws_hardware import (
        PROVIDER_SELECTION_S3_KEY,
        publish_provider_selection,
    )

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    store = _MemorySelectionStore(lost_put=True)
    published = publish_provider_selection(
        authority_root=tmp_path / "authority",
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        selection_data=selection_data,
        store=store,
        **_qualification_verification_kwargs(),
    )

    assert published.remote.version_id == "version-1"
    assert store.head_calls == [
        {"key": PROVIDER_SELECTION_S3_KEY, "version_id": None},
        {"key": PROVIDER_SELECTION_S3_KEY, "version_id": "version-1"},
    ]


@pytest.mark.parametrize("corrupt_head", ["checksum", "bytes", "version"])
def test_fixed_store_rejects_inexact_head_recovery(tmp_path, corrupt_head):
    from msctl.aws_hardware import publish_provider_selection

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    store = _MemorySelectionStore(corrupt_head=corrupt_head)

    with pytest.raises(ValueError, match="HEAD|version|remote"):
        publish_provider_selection(
            authority_root=tmp_path / "authority",
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            selection_data=selection_data,
            store=store,
            **_qualification_verification_kwargs(),
        )


def test_fixed_store_rejects_conflicting_existing_selection(tmp_path):
    from msctl.aws_hardware import (
        PROVIDER_SELECTION_LOCAL_PATH,
        PROVIDER_SELECTION_S3_KEY,
        publish_provider_selection,
    )

    p5_data, _, _ = _write_authority_inputs(
        tmp_path, "aws-p5.48xlarge-v3"
    )
    p6_data, p6_lock, p6_evidence = _write_authority_inputs(
        tmp_path, "aws-p6-b300.48xlarge-v3"
    )
    store = _MemorySelectionStore()
    store.objects[PROVIDER_SELECTION_S3_KEY] = (p5_data, "existing-version")

    with pytest.raises(ValueError, match="conflict|different|remote"):
        publish_provider_selection(
            authority_root=tmp_path / "authority",
            repo_root=ROOT,
            runtime_lock_path=p6_lock,
            runtime_evidence_path=p6_evidence,
            selection_data=p6_data,
            store=store,
            **_qualification_verification_kwargs(),
        )
    assert set(store.objects) == {PROVIDER_SELECTION_S3_KEY}
    assert not (
        tmp_path / "authority" / PROVIDER_SELECTION_LOCAL_PATH
    ).exists()


def test_authority_load_and_resume_reparse_anchored_bytes(tmp_path):
    from msctl.aws_hardware import (
        load_local_provider_selection_authority,
        publish_provider_selection,
        validate_resume_hardware_binding,
    )

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    authority_root = tmp_path / "authority"
    store = _MemorySelectionStore()
    published = publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        selection_data=selection_data,
        store=store,
        **_qualification_verification_kwargs(),
    )
    loaded = load_local_provider_selection_authority(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        **_qualification_verification_kwargs(),
    )
    resumed = validate_resume_hardware_binding(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        store=store,
        account_id=loaded.account_id,
        instance_id=loaded.qualification_instance_id,
        boot_id=loaded.qualification_boot_id,
        expected_selection_version_id=published.remote.version_id,
        amendment_sha256=loaded.amendment.sha256,
        provider_selection_sha256=loaded.sha256,
        profile_sha256=loaded.profile.sha256,
        runtime_lock_sha256=loaded.runtime_lock_sha256,
        runtime_evidence_sha256=loaded.runtime_evidence_sha256,
        seed=9,
        arm="split90",
        **_qualification_verification_kwargs(),
    )

    assert loaded == published.selection
    assert resumed.selection_sha256 == loaded.sha256

    with pytest.raises(ValueError, match="profile"):
        validate_resume_hardware_binding(
            authority_root=authority_root,
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            store=store,
            account_id=loaded.account_id,
            instance_id=loaded.qualification_instance_id,
            boot_id=loaded.qualification_boot_id,
            expected_selection_version_id=published.remote.version_id,
            amendment_sha256=loaded.amendment.sha256,
            provider_selection_sha256=loaded.sha256,
            profile_sha256=P5_PROFILE_SHA256,
            runtime_lock_sha256=loaded.runtime_lock_sha256,
            runtime_evidence_sha256=loaded.runtime_evidence_sha256,
            seed=9,
            arm="split90",
            **_qualification_verification_kwargs(),
        )


def test_forged_dataclasses_cannot_enter_authority_apis(tmp_path):
    from msctl.aws_hardware import (
        load_aws_hardware_amendment,
        parse_provider_selection_receipt_bytes,
        parse_verified_provider_selection_bytes,
        validate_aws_hardware_amendment_files,
        validate_resume_hardware_binding,
    )

    selection, lock_data, evidence_data = _verified_selection_fixture()
    receipt = parse_verified_provider_selection_bytes(
        _canonical_json(selection),
        amendment_data=AMENDMENT.read_bytes(),
        profile_data=P6_PROFILE.read_bytes(),
        runtime_lock_data=lock_data,
        runtime_evidence_data=evidence_data,
        **_qualification_verification_kwargs(),
    )
    forged_receipt = replace(
        receipt,
        profile=replace(
            receipt.profile,
            profile_id="aws-p5.48xlarge-v3",
            sha256=P5_PROFILE_SHA256,
        ),
        sha256="0" * 64,
    )
    amendment = load_aws_hardware_amendment(AMENDMENT, repo_root=ROOT)
    forged_amendment = replace(
        amendment,
        profiles=(forged_receipt.profile,),
        sha256=AMENDMENT_SHA256,
    )

    with pytest.raises(TypeError):
        parse_provider_selection_receipt_bytes(
            _canonical_json(selection),
            amendment=forged_amendment,
        )
    with pytest.raises(TypeError):
        validate_aws_hardware_amendment_files(
            forged_amendment,
            repo_root=ROOT,
        )
    assert (
        validate_aws_hardware_amendment_files(repo_root=ROOT).sha256
        == AMENDMENT_SHA256
    )
    with pytest.raises(TypeError):
        validate_resume_hardware_binding(
            forged_receipt,
            amendment_sha256=AMENDMENT_SHA256,
            provider_selection_sha256="0" * 64,
            profile_sha256=P5_PROFILE_SHA256,
            runtime_lock_sha256=receipt.runtime_lock_sha256,
            seed=0,
            arm="dense",
        )


def test_authority_private_files_reject_mode_and_hardlink_drift(tmp_path):
    from msctl.aws_hardware import (
        PROVIDER_SELECTION_LOCAL_PATH,
        load_local_provider_selection_authority,
        publish_provider_selection,
    )

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    runtime_lock.chmod(0o644)
    with pytest.raises(ValueError, match="private|mode"):
        publish_provider_selection(
            authority_root=tmp_path / "authority",
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            selection_data=selection_data,
            store=_MemorySelectionStore(),
            **_qualification_verification_kwargs(),
        )

    runtime_lock.chmod(0o600)
    authority_root = tmp_path / "authority-valid"
    publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        selection_data=selection_data,
        store=_MemorySelectionStore(),
        **_qualification_verification_kwargs(),
    )
    local_path = authority_root / PROVIDER_SELECTION_LOCAL_PATH
    local_path.chmod(0o644)
    with pytest.raises(ValueError, match="private|mode"):
        load_local_provider_selection_authority(
            authority_root=authority_root,
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            **_qualification_verification_kwargs(),
        )

    local_path.chmod(0o600)
    hardlink = tmp_path / "selection-hardlink.json"
    os.link(local_path, hardlink)
    with pytest.raises(ValueError, match="link"):
        load_local_provider_selection_authority(
            authority_root=authority_root,
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            **_qualification_verification_kwargs(),
        )


def _write_cli_authority_inputs(tmp_path: Path) -> dict[str, Path]:
    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    selection = tmp_path / "selection.json"
    selection.write_bytes(selection_data)
    selection.chmod(0o600)
    public_key = tmp_path / "approval-public-key.pem"
    public_key.write_bytes(b"synthetic public key for injected verifier\n")
    public_key.chmod(0o600)
    return {
        "authority_root": tmp_path / "authority-cli",
        "public_key": public_key,
        "runtime_evidence": runtime_evidence,
        "runtime_lock": runtime_lock,
        "selection": selection,
    }


def _publish_cli_argv(paths: dict[str, Path], *, apply: bool) -> list[str]:
    argv = [
        "publish",
        "--repo-root",
        str(ROOT),
        "--authority-root",
        str(paths["authority_root"]),
        "--runtime-lock",
        str(paths["runtime_lock"]),
        "--qualification-evidence",
        str(paths["runtime_evidence"]),
        "--selection",
        str(paths["selection"]),
        "--bucket",
        "memorysplit-authority",
        "--region",
        "us-east-1",
        "--approval-public-key",
        str(paths["public_key"]),
        "--approval-public-key-sha256",
        TRUSTED_APPROVAL_PUBLIC_KEY_SHA256,
    ]
    if apply:
        argv.append("--apply")
    return argv


def test_authority_cli_is_dry_run_by_default_and_apply_is_explicit(tmp_path):
    from msctl.aws_hardware import run_hardware_authority_cli

    paths = _write_cli_authority_inputs(tmp_path)
    store = _MemorySelectionStore()
    dry_run, planned = run_hardware_authority_cli(
        _publish_cli_argv(paths, apply=False),
        store=store,
        **_qualification_verification_kwargs(),
    )

    assert dry_run is True
    assert planned["operation"] == "publish-provider-selection"
    assert planned["apply"] is False
    assert planned["s3_key"] == SELECTION_S3_KEY
    assert store.list_calls == []
    assert not paths["authority_root"].exists()

    dry_run, applied = run_hardware_authority_cli(
        _publish_cli_argv(paths, apply=True),
        store=store,
        **_qualification_verification_kwargs(),
    )
    assert dry_run is False
    assert applied["apply"] is True
    assert applied["selection_version_id"] == "version-1"
    assert applied["selection_sha256"] == hashlib.sha256(
        paths["selection"].read_bytes()
    ).hexdigest()
    assert store.list_calls == [SELECTION_S3_KEY] * 4


def test_cli_apply_timeout_reports_uncertain_then_retry_reports_recovered(
    tmp_path,
    capsysbinary,
):
    from msctl.aws_hardware import main, run_hardware_authority_cli

    paths = _write_cli_authority_inputs(tmp_path)
    store = _MemorySelectionStore(fail_at="put-after-write")
    exit_code = main(
        _publish_cli_argv(paths, apply=True),
        store=store,
        **_qualification_verification_kwargs(),
    )
    report = json.loads(capsysbinary.readouterr().out)

    assert exit_code == 2
    assert report["ok"] is False
    assert report["dry_run"] is False
    assert report["publication_state"] == "uncertain"

    dry_run, recovered = run_hardware_authority_cli(
        _publish_cli_argv(paths, apply=True),
        store=store,
        **_qualification_verification_kwargs(),
    )
    assert dry_run is False
    assert recovered["publication_state"] == "recovered"
    assert len(store.put_calls) == 1


def test_exact_admission_returns_only_authenticated_selection_binding(tmp_path):
    from msctl.aws_hardware import (
        AuthenticatedSelectionBinding,
        admit_provider_selection,
        publish_provider_selection,
        validate_resume_hardware_binding,
    )

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    authority_root = tmp_path / "authority-admit"
    store = _MemorySelectionStore()
    published = publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        selection_data=selection_data,
        store=store,
        **_qualification_verification_kwargs(),
    )
    binding = admit_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        store=store,
        account_id="123456789012",
        instance_id="i-0123456789abcdef0",
        boot_id="12345678-1234-4abc-8def-1234567890ab",
        seed=0,
        arm="dense",
        expected_selection_version_id=published.remote.version_id,
        **_qualification_verification_kwargs(),
    )

    assert isinstance(binding, AuthenticatedSelectionBinding)
    assert binding.selection_sha256 == published.selection.sha256
    assert binding.selection_version_id == published.remote.version_id
    assert binding.profile_id == "aws-p6-b300.48xlarge-v3"
    assert binding.runtime_lock_sha256 == published.selection.runtime_lock_sha256
    assert binding.qualification_evidence_sha256 == (
        published.selection.runtime_evidence_sha256
    )
    assert binding.account_id == "123456789012"
    assert binding.instance_id == "i-0123456789abcdef0"
    assert binding.boot_id == "12345678-1234-4abc-8def-1234567890ab"
    assert binding.seed == 0
    assert binding.arm == "dense"
    assert not hasattr(binding, "selection")
    validated = validate_resume_hardware_binding(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        store=store,
        account_id=binding.account_id,
        instance_id=binding.instance_id,
        boot_id=binding.boot_id,
        expected_selection_version_id=binding.selection_version_id,
        amendment_sha256=binding.amendment_sha256,
        provider_selection_sha256=binding.selection_sha256,
        profile_sha256=binding.profile_sha256,
        runtime_lock_sha256=binding.runtime_lock_sha256,
        runtime_evidence_sha256=binding.qualification_evidence_sha256,
        seed=0,
        arm="dense",
        **_qualification_verification_kwargs(),
    )
    assert validated == binding


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instance_id", "i-0fedcba9876543210"),
        ("boot_id", "87654321-4321-4abc-8def-1234567890ab"),
        ("account_id", "210987654321"),
        ("expected_selection_version_id", "wrong-version"),
    ],
)
def test_exact_admission_rejects_wrong_authority_identity(
    tmp_path,
    field,
    value,
):
    from msctl.aws_hardware import (
        admit_provider_selection,
        publish_provider_selection,
    )

    selection_data, runtime_lock, runtime_evidence = _write_authority_inputs(
        tmp_path
    )
    authority_root = tmp_path / "authority-admit"
    store = _MemorySelectionStore()
    published = publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        selection_data=selection_data,
        store=store,
        **_qualification_verification_kwargs(),
    )
    values = {
        "account_id": "123456789012",
        "arm": "split90",
        "boot_id": "12345678-1234-4abc-8def-1234567890ab",
        "expected_selection_version_id": published.remote.version_id,
        "instance_id": "i-0123456789abcdef0",
        "seed": 9,
    }
    values[field] = value

    with pytest.raises(ValueError, match="identity|version|authority"):
        admit_provider_selection(
            authority_root=authority_root,
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            store=store,
            **values,
            **_qualification_verification_kwargs(),
        )


def test_openssl_approval_verifier_checks_public_key_commitment_and_signature(
    tmp_path,
):
    from msctl.aws_hardware import OpenSslQualificationApprovalVerifier

    key = tmp_path / "approval-public-key.pem"
    key.write_bytes(b"synthetic-pem-public-key\n")
    key.chmod(0o600)
    calls = []

    def runner(argv, environment, timeout_seconds):
        calls.append((list(argv), dict(environment), timeout_seconds))
        return SimpleNamespace(
            returncode=0,
            stdout="Verified OK\n",
            stderr="",
        )

    verifier = OpenSslQualificationApprovalVerifier(
        public_key_path=key,
        environment={"LANG": "C"},
        runner=runner,
    )
    signature = base64.b64encode(b"synthetic signature").decode("ascii")
    assert verifier.verify(
        payload=b'{"scope":"exact"}\n',
        signature=signature,
        algorithm="RSASSA_PSS_SHA_256",
        public_key_sha256=hashlib.sha256(key.read_bytes()).hexdigest(),
    )
    assert not verifier.verify(
        payload=b'{"scope":"exact"}\n',
        signature=signature,
        algorithm="RSASSA_PSS_SHA_256",
        public_key_sha256="0" * 64,
    )
    assert len(calls) == 1
    assert calls[0][0][:4] == [
        "/usr/bin/openssl",
        "dgst",
        "-sha256",
        "-sigopt",
    ]
