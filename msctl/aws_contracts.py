"""Canonical, side-effect-free contract for the AWS-only v3 cohort."""

from __future__ import annotations

import re
from typing import Final


PROVIDER: Final = "aws-p5.48xlarge"
COHORT_ID: Final = "memorysplit-confirmatory-v3-360m-n10-aws"
PREREGISTRATION_ID: Final = "memorysplit-confirmatory-v3"
PREREGISTRATION_PATH: Final = "configs/preregistration-v3.yaml"
COHORT_ASSIGNMENT_PATH: Final = "configs/cohort-assignment-v3.json"
CONFIG_ROOT: Final = "configs/360m-v3"
PROFILE_PATH: Final = "cluster/profiles/aws-p5.48xlarge-v3.json"
DATASET_POINTER_PATH: Final = "DATASET-POINTER-AWS.json"
DATASET_RECEIPT_PATH: Final = "dataset/receipt.json"
SEEDS: Final = tuple(range(10))
ARMS: Final = ("dense", "split90")
SNAPSHOT_STEPS: Final = (1_358, 3_396, 6_791, 10_187, 13_582)
PACKAGE_FORMAT_VERSION: Final = 2
AWS_RUNTIME_VERSION_FIELDS: Final = (
    "python",
    "pytorch",
    "cuda",
    "cudnn",
    "nccl",
    "nvidia_driver",
    "fabric_manager",
    "docker",
    "nvidia_container_runtime",
    "aws_cli",
)
AWS_RUNTIME_LOCK_FIELDS: Final = (
    "schema_version",
    "source_commit",
    "source_tree",
    "control_bundle_sha256",
    "profile_sha256",
    "ami_id",
    "ami_owner_id",
    "container_image",
    "container_image_digest",
    "versions",
)
AWS_ENVIRONMENT_RECEIPT_V2_FIELDS: Final = (
    "schema_version",
    "receipt_type",
    "provider",
    "profile_sha256",
    "runtime_lock_sha256",
    "control_bundle_sha256",
    "source_commit",
    "source_tree",
    "container_image",
    "container_image_digest",
    "aws_instance_identity_document",
    "aws_instance_identity_pkcs7",
    "account_id",
    "instance_id",
    "region",
    "ami_id",
    "boot_id",
    "runtime_facts",
)

EXPECTED_CONFIG_PATHS: Final = tuple(
    f"{CONFIG_ROOT}/{arm}-s{seed}.yaml"
    for seed in SEEDS
    for arm in ARMS
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def expected_config_paths() -> tuple[str, ...]:
    """Return the exact ordered v3 run-config paths."""

    return EXPECTED_CONFIG_PATHS


def validate_sha256(value: object) -> str:
    """Return one canonical SHA-256 digest or reject it."""

    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("SHA-256 must be exactly 64 lowercase hexadecimal characters")
    return value


def release_archive_key(archive_sha256: object) -> str:
    """Derive the content-addressed release archive key."""

    digest = validate_sha256(archive_sha256)
    return f"releases/{digest}/archive.zip"


def release_receipt_key(archive_sha256: object) -> str:
    """Derive the content-addressed release receipt key."""

    digest = validate_sha256(archive_sha256)
    return f"releases/{digest}/release.json"


def release_checksum_key(archive_sha256: object) -> str:
    """Derive the content-addressed release checksum key."""

    digest = validate_sha256(archive_sha256)
    return f"releases/{digest}/archive.sha256"


def dataset_receipt_key() -> str:
    """Return the canonical logical and local dataset receipt key."""

    return DATASET_RECEIPT_PATH


__all__ = [
    "ARMS",
    "AWS_ENVIRONMENT_RECEIPT_V2_FIELDS",
    "AWS_RUNTIME_LOCK_FIELDS",
    "AWS_RUNTIME_VERSION_FIELDS",
    "COHORT_ASSIGNMENT_PATH",
    "COHORT_ID",
    "CONFIG_ROOT",
    "DATASET_POINTER_PATH",
    "DATASET_RECEIPT_PATH",
    "EXPECTED_CONFIG_PATHS",
    "PACKAGE_FORMAT_VERSION",
    "PREREGISTRATION_ID",
    "PREREGISTRATION_PATH",
    "PROFILE_PATH",
    "PROVIDER",
    "SEEDS",
    "SNAPSHOT_STEPS",
    "dataset_receipt_key",
    "expected_config_paths",
    "release_archive_key",
    "release_checksum_key",
    "release_receipt_key",
    "validate_sha256",
]
