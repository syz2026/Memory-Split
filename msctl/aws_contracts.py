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
_OCI_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_OCI_REGISTRY_LABEL_PATTERN = re.compile(
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
)
_OCI_REPOSITORY_COMPONENT_PATTERN = re.compile(
    r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
)


def expected_config_paths() -> tuple[str, ...]:
    """Return the exact ordered v3 run-config paths."""

    return EXPECTED_CONFIG_PATHS


def validate_sha256(value: object) -> str:
    """Return one canonical SHA-256 digest or reject it."""

    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("SHA-256 must be exactly 64 lowercase hexadecimal characters")
    return value


def validate_digest_pinned_oci_image(
    image: object,
    digest: object,
) -> tuple[str, str]:
    """Return one canonical full OCI reference pinned to its exact digest."""

    if (
        not isinstance(digest, str)
        or _OCI_DIGEST_PATTERN.fullmatch(digest) is None
    ):
        raise ValueError(
            "container image digest must be sha256 followed by 64 lowercase hex"
        )
    if (
        not isinstance(image, str)
        or image.count("@") != 1
        or any(character.isspace() for character in image)
        or "\x00" in image
    ):
        raise ValueError(
            "container image must be one canonical digest-pinned OCI reference"
        )
    name, image_digest = image.split("@")
    if image_digest != digest:
        raise ValueError("container image digest must exactly match its lock")
    components = name.split("/")
    if len(components) < 2:
        raise ValueError(
            "container image must include a valid registry and repository"
        )
    registry, *repository = components
    if not registry or not repository:
        raise ValueError(
            "container image must include a valid registry and repository"
        )
    host = registry
    if ":" in registry:
        host, port = registry.rsplit(":", 1)
        if (
            not port.isascii()
            or not port.isdecimal()
            or not 1 <= int(port) <= 65_535
        ):
            raise ValueError("container image registry port is invalid")
    labels = host.split(".")
    if (
        not host
        or host != host.lower()
        or len(host) > 253
        or (
            "." not in host
            and host != "localhost"
        )
        or any(
            _OCI_REGISTRY_LABEL_PATTERN.fullmatch(label) is None
            for label in labels
        )
    ):
        raise ValueError("container image registry host is invalid")
    if any(
        _OCI_REPOSITORY_COMPONENT_PATTERN.fullmatch(component) is None
        for component in repository
    ):
        raise ValueError("container image repository path is invalid")
    return image, digest


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


def checkpoint_object_key(seed: object, arm: object, sha256: object) -> str:
    """Return one content-addressed v3 checkpoint object key."""

    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("checkpoint seed must be an exact integer from 0 to 9")
    if arm not in ARMS:
        raise ValueError("checkpoint arm must be dense or split90")
    digest = validate_sha256(sha256)
    return f"checkpoints/seed-{seed}/{arm}/sha256/{digest}.pt"


def checkpoint_receipt_key(seed: object, sha256: object) -> str:
    """Return one content-addressed v3 pair-receipt object key."""

    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("checkpoint seed must be an exact integer from 0 to 9")
    digest = validate_sha256(sha256)
    return f"receipts/checkpoints/seed-{seed}/sha256/{digest}.json"


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
    "checkpoint_object_key",
    "checkpoint_receipt_key",
    "dataset_receipt_key",
    "expected_config_paths",
    "release_archive_key",
    "release_checksum_key",
    "release_receipt_key",
    "validate_digest_pinned_oci_image",
    "validate_sha256",
]
