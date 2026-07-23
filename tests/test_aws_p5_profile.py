from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest

from cluster.aws.p5.profile import (
    AwsP5Profile,
    load_aws_p5_profile,
    validate_runtime_environment,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "cluster" / "profiles" / "aws-p5.48xlarge.json"
POINTER_PATH = ROOT / "DATASET-POINTER-AWS.json"


def _write_profile(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _production_profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _mutated_profile(tmp_path: Path, mutate) -> Path:
    value = deepcopy(_production_profile())
    mutate(value)
    return _write_profile(tmp_path, value)


def test_production_profile_freezes_exact_p5_hardware_and_seed_assignment():
    profile = load_aws_p5_profile(PROFILE_PATH)

    assert isinstance(profile, AwsP5Profile)
    assert profile.profile_id == "aws-p5.48xlarge"
    assert profile.provider == "aws-p5.48xlarge"
    assert profile.instance_type == "p5.48xlarge"
    assert profile.purchase_model == "on_demand"
    assert profile.vcpus == 192
    assert profile.memory_gib == 2048
    assert profile.gpu_model == "NVIDIA H100 80GB"
    assert profile.allocated_gpus == 8
    assert profile.train_groups == (4, 4)
    assert profile.instance_store_model == "Amazon EC2 NVMe Instance Storage"
    assert profile.instance_store_devices == 8
    assert profile.instance_store_device_bytes == 3_840_000_000_000
    assert profile.scratch_root == "/mnt/memorysplit"
    assert profile.durable_uri_env == "MS_S3_ROOT"
    assert profile.ami_id_env == "MS_AWS_AMI_ID"
    assert profile.container_digest_env == "MS_CONTAINER_DIGEST"
    assert profile.runtime_uid_env == "MS_RUNTIME_UID"
    assert profile.runtime_gid_env == "MS_RUNTIME_GID"
    assert profile.assigned_seeds == (1, 2, 3, 4)
    assert profile.process_env_allowlist == (
        "AWS_REGION",
        "LANG",
        "LC_ALL",
    )
    assert re.fullmatch(r"[0-9a-f]{64}", profile.sha256)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(instance_type="p5.4xlarge"),
        lambda value: value.update(purchase_model="spot"),
        lambda value: value["gpu"].update(allocated=1),
        lambda value: value["gpu"].update(seed_train_groups=[4, 3]),
        lambda value: value["storage"].update(durable_uri_env="MS_SHARED_ROOT"),
        lambda value: value["storage"].update(scratch_root="/tmp/memorysplit"),
        lambda value: value["runtime"].update(ami_id_env="ami-latest"),
        lambda value: value["runtime"].update(
            container_digest_env="memorysplit:latest"
        ),
        lambda value: value["runtime"].update(runtime_uid_env="UID"),
        lambda value: value["runtime"].update(runtime_gid_env="GID"),
        lambda value: value.update(assigned_seeds=[0, 1, 2, 3]),
        lambda value: value["process_env_allowlist"].append(
            "AWS_SECRET_ACCESS_KEY"
        ),
        lambda value: value.update(unexpected=True),
    ],
    ids=[
        "one-gpu-instance",
        "spot-default",
        "one-gpu-allocation",
        "asymmetric-groups",
        "non-s3-root-variable",
        "wrong-scratch-root",
        "mutable-ami-label",
        "mutable-container-tag",
        "implicit-runtime-uid",
        "implicit-runtime-gid",
        "seed-zero",
        "static-aws-key",
        "unknown-field",
    ],
)
def test_profile_parser_rejects_contract_drift(tmp_path, mutate):
    path = _mutated_profile(tmp_path, mutate)

    with pytest.raises(ValueError):
        load_aws_p5_profile(path)


def test_profile_parser_rejects_duplicate_keys(tmp_path):
    value = PROFILE_PATH.read_text(encoding="utf-8")
    duplicate = value.replace(
        '"schema_version": 1,',
        '"schema_version": 1,\n  "schema_version": 1,',
        1,
    )
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_aws_p5_profile(path)


def test_profile_parser_rejects_non_utf8_json(tmp_path):
    path = tmp_path / "utf16-profile.json"
    path.write_bytes(json.dumps(_production_profile()).encode("utf-16"))

    with pytest.raises(ValueError, match="UTF-8"):
        load_aws_p5_profile(path)


def test_profile_parser_rejects_symlink(tmp_path):
    link = tmp_path / "profile-link.json"
    link.symlink_to(PROFILE_PATH)

    with pytest.raises(ValueError, match="symlink"):
        load_aws_p5_profile(link)


def test_runtime_environment_requires_immutable_aws_identity():
    profile = load_aws_p5_profile(PROFILE_PATH)
    runtime = validate_runtime_environment(
        profile,
        {
            "AWS_REGION": "us-east-1",
            "MS_S3_ROOT": "s3://memorysplit-prod/cohort-v2",
            "MS_AWS_AMI_ID": "ami-0123456789abcdef0",
            "MS_CONTAINER_DIGEST": "sha256:" + "a" * 64,
            "MS_CONTAINER_IMAGE": (
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/memorysplit"
                "@sha256:" + "a" * 64
            ),
            "MS_RUNTIME_UID": "1000",
            "MS_RUNTIME_GID": "1000",
        },
    )

    assert runtime.region == "us-east-1"
    assert runtime.s3_root == "s3://memorysplit-prod/cohort-v2"
    assert runtime.ami_id == "ami-0123456789abcdef0"
    assert runtime.container_digest == "sha256:" + "a" * 64
    assert runtime.container_image.endswith(
        "@sha256:" + "a" * 64
    )
    assert runtime.uid == 1000
    assert runtime.gid == 1000


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("AWS_REGION", "not-a-region"),
        ("MS_S3_ROOT", "/mnt/shared"),
        ("MS_S3_ROOT", "s3://bucket"),
        ("MS_S3_ROOT", "s3://user:password@bucket/cohort"),
        ("MS_AWS_AMI_ID", "ami-latest"),
        ("MS_CONTAINER_DIGEST", "memorysplit:latest"),
        ("MS_CONTAINER_DIGEST", "sha256:" + "A" * 64),
        ("MS_CONTAINER_IMAGE", "memorysplit@sha256:" + "a" * 64),
        (
            "MS_CONTAINER_IMAGE",
            "registry.example/memorysplit@sha256:" + "b" * 64,
        ),
        ("MS_RUNTIME_UID", "0"),
        ("MS_RUNTIME_UID", "-1"),
        ("MS_RUNTIME_UID", "01000"),
        ("MS_RUNTIME_GID", "root"),
    ],
)
def test_runtime_environment_rejects_mutable_or_unsafe_values(name, value):
    profile = load_aws_p5_profile(PROFILE_PATH)
    environment = {
        "AWS_REGION": "us-east-1",
        "MS_S3_ROOT": "s3://memorysplit-prod/cohort-v2",
        "MS_AWS_AMI_ID": "ami-0123456789abcdef0",
        "MS_CONTAINER_DIGEST": "sha256:" + "a" * 64,
        "MS_CONTAINER_IMAGE": (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/memorysplit"
            "@sha256:" + "a" * 64
        ),
        "MS_RUNTIME_UID": "1000",
        "MS_RUNTIME_GID": "1000",
    }
    environment[name] = value

    with pytest.raises(ValueError):
        validate_runtime_environment(profile, environment)


@pytest.mark.parametrize(
    "secret_name",
    [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GITHUB_TOKEN",
        "SERVICE_PASSWORD",
        "PRIVATE_KEY",
    ],
)
def test_runtime_environment_rejects_inherited_secrets(secret_name):
    profile = load_aws_p5_profile(PROFILE_PATH)
    environment = {
        "AWS_REGION": "us-east-1",
        "MS_S3_ROOT": "s3://memorysplit-prod/cohort-v2",
        "MS_AWS_AMI_ID": "ami-0123456789abcdef0",
        "MS_CONTAINER_DIGEST": "sha256:" + "a" * 64,
        "MS_CONTAINER_IMAGE": (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/memorysplit"
            "@sha256:" + "a" * 64
        ),
        "MS_RUNTIME_UID": "1000",
        "MS_RUNTIME_GID": "1000",
        secret_name: "must-not-be-inherited",
    }

    with pytest.raises(ValueError, match="secret"):
        validate_runtime_environment(profile, environment)


@pytest.mark.parametrize(
    "source_name",
    [
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_EC2_METADATA_DISABLED",
    ],
)
def test_runtime_environment_rejects_non_instance_role_sources(source_name):
    profile = load_aws_p5_profile(PROFILE_PATH)
    environment = {
        "AWS_REGION": "us-east-1",
        "MS_S3_ROOT": "s3://memorysplit-prod/cohort-v2",
        "MS_AWS_AMI_ID": "ami-0123456789abcdef0",
        "MS_CONTAINER_DIGEST": "sha256:" + "a" * 64,
        "MS_CONTAINER_IMAGE": (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/memorysplit"
            "@sha256:" + "a" * 64
        ),
        "MS_RUNTIME_UID": "1000",
        "MS_RUNTIME_GID": "1000",
        source_name: "configured",
    }

    with pytest.raises(ValueError, match="credential source|instance role"):
        validate_runtime_environment(profile, environment)


def test_aws_dataset_pointer_uses_s3_as_the_durable_boundary():
    pointer = json.loads(POINTER_PATH.read_text(encoding="utf-8"))

    assert pointer == {
        "dataset_id": "memorysplit-v2-20x-reasoning-max-cohort",
        "durable_uri_env": "MS_S3_ROOT",
        "full_corpus_in_release": False,
        "materialization": "s3",
        "provider": "aws-p5.48xlarge",
        "relative_path": "dataset",
        "required_receipt": "dataset/receipt.json",
        "required_sidecars": [
            "dense_target_weights",
            "split90_target_weights",
        ],
        "schema_version": 1,
        "scratch_root": "/mnt/memorysplit",
        "source_lock_manifest": "configs/reasoning-dataset-v2.json",
    }


def test_production_contracts_contain_no_secret_or_placeholder_hash():
    text = (
        PROFILE_PATH.read_text(encoding="utf-8")
        + POINTER_PATH.read_text(encoding="utf-8")
    )

    assert re.search(r"\b[0-9a-fA-F]{64}\b", text) is None
    assert "AWS_SECRET_ACCESS_KEY" not in text
    assert "AWS_ACCESS_KEY_ID" not in text
    assert ":latest" not in text
