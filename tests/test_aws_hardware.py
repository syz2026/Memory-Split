from __future__ import annotations

import hashlib
import json
from copy import deepcopy
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
    "9d6bbaedfe2520bd6ce10957e2c8e923a624144c0feb4b19c3764f71040be755"
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
    assert profile.purchase_model == "capacity_block"
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
        lambda value: value.update(purchase_model="on_demand"),
        lambda value: value["offerings"][0].update(
            availability_zone="us-east-1c"
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
            "purchase_model": "capacity_block",
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
    return {
        "amendment": {
            "path": "configs/aws-hardware-amendment-v3.json",
            "sha256": AMENDMENT_SHA256,
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
            "runtime_lock_sha256": "a" * 64,
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
    from msctl.aws_hardware import (
        load_aws_hardware_amendment,
        parse_provider_selection_receipt_bytes,
    )

    amendment = load_aws_hardware_amendment(AMENDMENT, repo_root=ROOT)
    data = _canonical_json(_selection_value(profile_id))
    receipt = parse_provider_selection_receipt_bytes(
        data,
        amendment=amendment,
    )

    assert amendment.sha256 == AMENDMENT_SHA256
    assert receipt.schema_version == 1
    assert receipt.receipt_type == "memorysplit-aws-provider-selection-v1"
    assert receipt.selection_scope == "entire_cohort"
    assert receipt.replacement_policy == "forbidden"
    assert receipt.amendment.path == "configs/aws-hardware-amendment-v3.json"
    assert receipt.amendment.sha256 == AMENDMENT_SHA256
    assert receipt.profile.profile_id == profile_id
    assert receipt.seeds == tuple(range(10))
    assert receipt.arms == ("dense", "split90")
    assert receipt.train_groups == (4, 4)
    assert receipt.preregistration_sha256 == PREREGISTRATION_SHA256
    assert receipt.cohort_assignment_sha256 == COHORT_ASSIGNMENT_SHA256
    assert receipt.runtime_lock_sha256 == "a" * 64
    assert receipt.ami_id == "ami-0123456789abcdef0"
    assert receipt.container_image_digest == "sha256:" + "b" * 64
    assert receipt.account_id == "123456789012"
    assert receipt.region == "us-east-1"
    assert receipt.selected_at == "2026-07-24T05:30:00Z"
    assert receipt.protected_outcomes_inspected == ()
    assert receipt.sha256 == hashlib.sha256(data).hexdigest()
    if profile_id.startswith("aws-p6"):
        assert receipt.availability_zone == "us-east-1d"
        assert receipt.purchase_model == "capacity_block"
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
        value["aws"]["purchase_model"] = "on_demand"
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
    from msctl.aws_hardware import (
        load_aws_hardware_amendment,
        parse_provider_selection_receipt_bytes,
    )

    amendment = load_aws_hardware_amendment(AMENDMENT, repo_root=ROOT)
    value = _selection_value()
    _mutate_selection(value, mutation)

    with pytest.raises(ValueError):
        parse_provider_selection_receipt_bytes(
            _canonical_json(value),
            amendment=amendment,
        )


def test_selection_receipt_requires_its_sole_canonical_byte_encoding():
    from msctl.aws_hardware import (
        load_aws_hardware_amendment,
        parse_provider_selection_receipt_bytes,
    )

    amendment = load_aws_hardware_amendment(AMENDMENT, repo_root=ROOT)
    pretty = (
        json.dumps(_selection_value(), indent=2, sort_keys=True) + "\n"
    ).encode("ascii")

    with pytest.raises(ValueError, match="canonical"):
        parse_provider_selection_receipt_bytes(pretty, amendment=amendment)


def test_selection_receipt_rejects_duplicate_keys():
    from msctl.aws_hardware import (
        load_aws_hardware_amendment,
        parse_provider_selection_receipt_bytes,
    )

    amendment = load_aws_hardware_amendment(AMENDMENT, repo_root=ROOT)
    duplicate = _canonical_json(_selection_value()).replace(
        b'"schema_version":1',
        b'"schema_version":1,"schema_version":1',
        1,
    )

    with pytest.raises(ValueError, match="duplicate"):
        parse_provider_selection_receipt_bytes(
            duplicate,
            amendment=amendment,
        )


def test_provider_selection_publication_is_canonical_and_no_replace(tmp_path):
    from msctl.aws_hardware import (
        load_aws_hardware_amendment,
        write_provider_selection_receipt,
    )

    amendment = load_aws_hardware_amendment(AMENDMENT, repo_root=ROOT)
    path = tmp_path / "provider-selection.json"
    first = _selection_value()
    receipt = write_provider_selection_receipt(
        path,
        first,
        amendment=amendment,
    )
    original = path.read_bytes()

    assert original == _canonical_json(first)
    assert receipt.sha256 == hashlib.sha256(original).hexdigest()

    with pytest.raises(FileExistsError):
        write_provider_selection_receipt(
            path,
            _selection_value("aws-p5.48xlarge-v3"),
            amendment=amendment,
        )
    assert path.read_bytes() == original


def test_resume_binding_rejects_cross_profile_or_runtime_tuple_mixing():
    from msctl.aws_hardware import (
        load_aws_hardware_amendment,
        parse_provider_selection_receipt_bytes,
        validate_resume_hardware_binding,
    )

    amendment = load_aws_hardware_amendment(AMENDMENT, repo_root=ROOT)
    receipt = parse_provider_selection_receipt_bytes(
        _canonical_json(_selection_value()),
        amendment=amendment,
    )
    binding = {
        "account_id": receipt.account_id,
        "amendment_sha256": receipt.amendment.sha256,
        "ami_id": receipt.ami_id,
        "arm": "split90",
        "availability_zone": receipt.availability_zone,
        "container_image_digest": receipt.container_image_digest,
        "profile_sha256": receipt.profile.sha256,
        "provider_selection_sha256": receipt.sha256,
        "purchase_model": receipt.purchase_model,
        "region": receipt.region,
        "runtime_lock_sha256": receipt.runtime_lock_sha256,
        "seed": 9,
    }

    validate_resume_hardware_binding(receipt, **binding)

    mixed = dict(binding)
    mixed["profile_sha256"] = P5_PROFILE_SHA256
    with pytest.raises(ValueError, match="profile"):
        validate_resume_hardware_binding(receipt, **mixed)

    for field, changed in (
        ("runtime_lock_sha256", "c" * 64),
        ("ami_id", "ami-0fedcba9876543210"),
        ("container_image_digest", "sha256:" + "d" * 64),
    ):
        mixed = dict(binding)
        mixed[field] = changed
        with pytest.raises(ValueError, match="runtime|AMI|image"):
            validate_resume_hardware_binding(receipt, **mixed)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("seed", 10),
        ("arm", "control"),
        ("account_id", "210987654321"),
        ("region", "us-west-2"),
        ("availability_zone", "us-east-1c"),
        ("purchase_model", "on_demand"),
        ("amendment_sha256", "0" * 64),
        ("provider_selection_sha256", "1" * 64),
    ],
)
def test_resume_binding_rejects_cohort_or_placement_mixing(field, changed):
    from msctl.aws_hardware import (
        load_aws_hardware_amendment,
        parse_provider_selection_receipt_bytes,
        validate_resume_hardware_binding,
    )

    amendment = load_aws_hardware_amendment(AMENDMENT, repo_root=ROOT)
    receipt = parse_provider_selection_receipt_bytes(
        _canonical_json(_selection_value()),
        amendment=amendment,
    )
    binding = {
        "account_id": receipt.account_id,
        "amendment_sha256": receipt.amendment.sha256,
        "ami_id": receipt.ami_id,
        "arm": "dense",
        "availability_zone": receipt.availability_zone,
        "container_image_digest": receipt.container_image_digest,
        "profile_sha256": receipt.profile.sha256,
        "provider_selection_sha256": receipt.sha256,
        "purchase_model": receipt.purchase_model,
        "region": receipt.region,
        "runtime_lock_sha256": receipt.runtime_lock_sha256,
        "seed": 0,
    }
    binding[field] = changed

    with pytest.raises(ValueError):
        validate_resume_hardware_binding(receipt, **binding)
