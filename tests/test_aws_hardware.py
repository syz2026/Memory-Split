from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

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


def _runtime_evidence_value(
    profile_id: str,
    runtime_lock_data: bytes,
) -> dict[str, object]:
    _, profile_sha256 = _profile_path_and_hash(profile_id)
    versions = P6_SOFTWARE_FLOORS if profile_id.startswith("aws-p6") else {}
    return {
        "account_id": "123456789012",
        "availability_zone": (
            "us-east-1d" if profile_id.startswith("aws-p6") else "us-east-1c"
        ),
        "profile_sha256": profile_sha256,
        "receipt_type": "memorysplit-aws-runtime-evidence-v1",
        "region": "us-east-1",
        "runtime_lock_sha256": hashlib.sha256(runtime_lock_data).hexdigest(),
        "schema_version": 1,
        "versions": dict(versions),
    }


def _verified_selection_fixture(
    profile_id: str = "aws-p6-b300.48xlarge-v3",
) -> tuple[dict[str, object], bytes, bytes]:
    lock_data = _canonical_json(_runtime_lock_value(profile_id))
    evidence_data = _canonical_json(
        _runtime_evidence_value(profile_id, lock_data)
    )
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
        evidence["profile_sha256"] = P5_PROFILE_SHA256
    elif mutation == "evidence-runtime":
        evidence["runtime_lock_sha256"] = "0" * 64
    elif mutation == "evidence-placement":
        evidence["availability_zone"] = "us-east-1c"
    else:
        field, value = {
            "cuda-floor": ("cuda", "12.9"),
            "efa-floor": ("efa", "1.43.9"),
            "kernel-floor": ("kernel", "5.15"),
            "driver-floor": ("nvidia_driver", "R570"),
            "nvlink-floor": ("nvlink", "R570"),
            "ofi-nccl-floor": ("ofi_nccl", "1.16.9"),
        }[mutation]
        evidence["versions"][field] = value

    lock_data = _canonical_json(lock)
    evidence["runtime_lock_sha256"] = (
        "0" * 64
        if mutation == "evidence-runtime"
        else hashlib.sha256(lock_data).hexdigest()
    )
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
    runtime_lock_data = _canonical_json(_runtime_lock_value(profile_id))
    runtime_evidence_data = _canonical_json(
        _runtime_evidence_value(profile_id, runtime_lock_data)
    )
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

    data = _canonical_json(_selection_value(profile_id))
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
    _, lock_data, evidence_data = _verified_selection_fixture(profile_id)
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
    receipt = publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        selection_data=selection_data,
        store=_MemorySelectionStore(),
    ).selection
    binding = {
        "amendment_sha256": receipt.amendment.sha256,
        "arm": "split90",
        "authority_root": authority_root,
        "profile_sha256": receipt.profile.sha256,
        "provider_selection_sha256": receipt.sha256,
        "repo_root": ROOT,
        "runtime_evidence_path": runtime_evidence,
        "runtime_evidence_sha256": receipt.runtime_evidence_sha256,
        "runtime_lock_path": runtime_lock,
        "runtime_lock_sha256": receipt.runtime_lock_sha256,
        "seed": 9,
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
    receipt = publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        selection_data=selection_data,
        store=_MemorySelectionStore(),
    ).selection
    binding = {
        "amendment_sha256": receipt.amendment.sha256,
        "arm": "dense",
        "authority_root": authority_root,
        "profile_sha256": receipt.profile.sha256,
        "provider_selection_sha256": receipt.sha256,
        "repo_root": ROOT,
        "runtime_evidence_path": runtime_evidence,
        "runtime_evidence_sha256": receipt.runtime_evidence_sha256,
        "runtime_lock_path": runtime_lock,
        "runtime_lock_sha256": receipt.runtime_lock_sha256,
        "seed": 0,
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
    ) -> None:
        self.lost_put = lost_put
        self.corrupt_head = corrupt_head
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.put_calls: list[dict[str, object]] = []
        self.head_calls: list[dict[str, object]] = []

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
        ArtifactBinding,
        AwsHardwareAmendment,
        AwsProviderSelectionReceipt,
        AwsRuntimeEvidence,
        AwsRuntimeLock,
        HardwareProfileBinding,
        PublishedProviderSelection,
        VersionedSelectionObject,
    )

    for contract in (
        ArtifactBinding,
        AwsHardwareAmendment,
        AwsProviderSelectionReceipt,
        AwsRuntimeEvidence,
        AwsRuntimeLock,
        HardwareProfileBinding,
        PublishedProviderSelection,
        VersionedSelectionObject,
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
        {"key": PROVIDER_SELECTION_S3_KEY, "version_id": "version-1"}
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
    )

    assert published.remote.version_id == "version-1"
    assert store.head_calls == [
        {"key": PROVIDER_SELECTION_S3_KEY, "version_id": None}
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
    published = publish_provider_selection(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        selection_data=selection_data,
        store=_MemorySelectionStore(),
    )
    loaded = load_local_provider_selection_authority(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
    )
    resumed = validate_resume_hardware_binding(
        authority_root=authority_root,
        repo_root=ROOT,
        runtime_lock_path=runtime_lock,
        runtime_evidence_path=runtime_evidence,
        amendment_sha256=loaded.amendment.sha256,
        provider_selection_sha256=loaded.sha256,
        profile_sha256=loaded.profile.sha256,
        runtime_lock_sha256=loaded.runtime_lock_sha256,
        runtime_evidence_sha256=loaded.runtime_evidence_sha256,
        seed=9,
        arm="split90",
    )

    assert loaded == published.selection
    assert resumed == loaded

    with pytest.raises(ValueError, match="profile"):
        validate_resume_hardware_binding(
            authority_root=authority_root,
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
            amendment_sha256=loaded.amendment.sha256,
            provider_selection_sha256=loaded.sha256,
            profile_sha256=P5_PROFILE_SHA256,
            runtime_lock_sha256=loaded.runtime_lock_sha256,
            runtime_evidence_sha256=loaded.runtime_evidence_sha256,
            seed=9,
            arm="split90",
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
    )
    local_path = authority_root / PROVIDER_SELECTION_LOCAL_PATH
    local_path.chmod(0o644)
    with pytest.raises(ValueError, match="private|mode"):
        load_local_provider_selection_authority(
            authority_root=authority_root,
            repo_root=ROOT,
            runtime_lock_path=runtime_lock,
            runtime_evidence_path=runtime_evidence,
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
        )
