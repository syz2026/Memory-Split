"""Prospective AWS hardware amendment and provider-selection contracts."""

from __future__ import annotations

import base64
import argparse
import hashlib
import ipaddress
import json
import math
import os
import re
import secrets
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from cluster.aws.gpu_profile import (
    load_aws_gpu_profile,
    parse_aws_gpu_profile_bytes,
    read_secure_regular_file,
)
from msctl.aws_contracts import (
    AWS_ENVIRONMENT_RECEIPT_V2_FIELDS,
    AWS_RUNTIME_LOCK_FIELDS,
    AWS_RUNTIME_VERSION_FIELDS,
    validate_digest_pinned_oci_image,
)
from msctl.errors import MsctlError
from msctl.fsutil import open_directory, rename_noreplace_at


AWS_HARDWARE_AMENDMENT_PATH = "configs/aws-hardware-amendment-v3.json"
P5_PROFILE_PATH = "cluster/profiles/aws-p5.48xlarge-v3.json"
P6_PROFILE_PATH = "cluster/profiles/aws-p6-b300.48xlarge-v3.json"
PREREGISTRATION_PATH = "configs/preregistration-v3.yaml"
COHORT_ASSIGNMENT_PATH = "configs/cohort-assignment-v3.json"
PROVIDER_SELECTION_LOCAL_PATH = (
    "memorysplit-confirmatory-v3-360m-n10-aws/provider-selection.json"
)
PROVIDER_SELECTION_S3_KEY = (
    "cohorts/memorysplit-confirmatory-v3-360m-n10-aws/provider-selection.json"
)
PROVIDER_SELECTION_VERSION_LOCAL_PATH = (
    "memorysplit-confirmatory-v3-360m-n10-aws/provider-selection-version.json"
)
PROVIDER_SELECTION_PENDING_LOCAL_PATH = (
    "memorysplit-confirmatory-v3-360m-n10-aws/provider-selection-pending.json"
)
PROVIDER_SELECTION_PENDING_ARCHIVE_LOCAL_PATH = (
    "memorysplit-confirmatory-v3-360m-n10-aws/"
    "provider-selection-pending-completed.json"
)
PROVIDER_SELECTION_MUTATION_LOCAL_PATH = (
    "memorysplit-confirmatory-v3-360m-n10-aws/"
    "provider-selection-remote-mutation.json"
)
PROVIDER_SELECTION_MUTATION_ARCHIVE_LOCAL_PATH = (
    "memorysplit-confirmatory-v3-360m-n10-aws/"
    "provider-selection-remote-mutation-completed.json"
)
AWS_HARDWARE_AMENDMENT_SHA256 = (
    "d4cf13b587c751d27756ad7881e538facb7ea79305a098990a568a7b28b6fb14"
)

_MAX_JSON_BYTES = 65_536
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_OCI_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_AMI_RE = re.compile(r"^ami-[0-9a-f]{8,17}$")
_ACCOUNT_RE = re.compile(r"^[0-9]{12}$")
_INSTANCE_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
_BOOT_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_S3_BUCKET_RE = re.compile(
    r"^(?![0-9]+(?:\.[0-9]+){3}$)(?!-)(?!.*\.\.)(?!.*\.-)(?!.*-\.)"
    r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$"
)
_REGION_RE = re.compile(r"^[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+$")
_AVAILABILITY_ZONE_RE = re.compile(
    r"^(?P<region>[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+)[a-z]$"
)
_UTC_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
_P5_PROFILE_SHA256 = (
    "2207bfbad5e8fa9fc804770b582d0b21f8b6ed109b2e3f3b5c0474c732c53543"
)
_P6_PROFILE_SHA256 = (
    "6884cd30670214bcecaa105d32b2b5518b1533d9fdbad6ac15327f9a8b7fefa4"
)
_PREREGISTRATION_SHA256 = (
    "6b2b5da3e3dc3d533498a0aa9d1891f356134ce045b1553b74a0161d94cb81d7"
)
_COHORT_ASSIGNMENT_SHA256 = (
    "47faf6f15e13336f67666081c3d0ebf5be3b981f3d2ac455385faa39e543199c"
)
_EXPECTED_AMENDMENT = {
    "amendment_id": "memorysplit-confirmatory-v3-aws-hardware-selection",
    "amendment_mode": "append_only",
    "cohort_assignment": {
        "path": COHORT_ASSIGNMENT_PATH,
        "sha256": _COHORT_ASSIGNMENT_SHA256,
    },
    "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
    "eligible_profiles": [
        {
            "path": P5_PROFILE_PATH,
            "profile_id": "aws-p5.48xlarge-v3",
            "provider": "aws-p5.48xlarge",
            "sha256": _P5_PROFILE_SHA256,
        },
        {
            "path": P6_PROFILE_PATH,
            "profile_id": "aws-p6-b300.48xlarge-v3",
            "provider": "aws-p6-b300.48xlarge",
            "sha256": _P6_PROFILE_SHA256,
        },
    ],
    "frozen_contract": {
        "arms": ["dense", "split90"],
        "one_profile_for_entire_cohort": True,
        "protected_outcomes_inspected": [],
        "scientific_invariants": "all_non_provider_fields_remain_frozen",
        "seeds": list(range(10)),
        "symmetric_training": True,
        "train_groups": [4, 4],
    },
    "preregistration": {
        "path": PREREGISTRATION_PATH,
        "sha256": _PREREGISTRATION_SHA256,
    },
    "prospective": True,
    "schema_version": 1,
    "supersedes_only": ["provider_assignment", "hardware_topology"],
}
_CANARY_PHASE_ORDER = (
    "hardware",
    "nccl_all_reduce",
    "functional",
    "resume",
    "throughput_4x4",
    "s3_roundtrip",
)
_IDENTITY_REQUIRED_FIELDS = frozenset(
    {
        "accountId",
        "architecture",
        "imageId",
        "instanceId",
        "privateIp",
        "region",
    }
)
_IDENTITY_OPTIONAL_FIELDS = frozenset(
    {
        "availabilityZone",
        "billingProducts",
        "devpayProductCodes",
        "instanceType",
        "kernelId",
        "marketplaceProductCodes",
        "pendingTime",
        "ramdiskId",
        "version",
    }
)
_CANARY_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_type",
        "provider",
        "instance_id",
        "boot_id",
        "profile_sha256",
        "runtime_lock_sha256",
        "environment_receipt_sha256",
        "release_sha256",
        "release_receipt_sha256",
        "run_manifest_sha256",
        "dataset_receipt_sha256",
        "dataset_build_id",
        "ordered_stream_sha256",
        "source_commit",
        "source_tree",
        "container_image",
        "container_image_digest",
        "seed",
        "phases",
        "hardware",
        "functional",
        "resume",
        "throughput_4x4",
        "s3_roundtrip",
        "passed",
        "started_at",
        "ended_at",
        "total_seconds",
    }
)


@dataclass(frozen=True)
class ArtifactBinding:
    path: str
    sha256: str


@dataclass(frozen=True)
class HardwareProfileBinding:
    path: str
    profile_id: str
    provider: str
    sha256: str


@dataclass(frozen=True)
class AwsHardwareAmendment:
    schema_version: int
    amendment_id: str
    amendment_mode: str
    prospective: bool
    cohort_id: str
    preregistration: ArtifactBinding
    cohort_assignment: ArtifactBinding
    profiles: tuple[HardwareProfileBinding, ...]
    supersedes_only: tuple[str, ...]
    seeds: tuple[int, ...]
    arms: tuple[str, ...]
    train_groups: tuple[int, int]
    symmetric_training: bool
    one_profile_for_entire_cohort: bool
    scientific_invariants: str
    protected_outcomes_inspected: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class AwsProviderSelectionReceipt:
    """One immutable hardware/runtime choice for every protected run cell."""

    schema_version: int
    receipt_type: str
    amendment: ArtifactBinding
    authority_local_path: str
    authority_s3_key: str
    cohort_id: str
    preregistration_sha256: str
    cohort_assignment_sha256: str
    profile: HardwareProfileBinding
    seeds: tuple[int, ...]
    arms: tuple[str, ...]
    train_groups: tuple[int, int]
    runtime_lock_sha256: str
    runtime_evidence_sha256: str
    ami_id: str
    container_image_digest: str
    account_id: str
    region: str
    availability_zone: str
    purchase_model: str
    selected_at: str
    selection_scope: str
    replacement_policy: str
    protected_outcomes_inspected: tuple[str, ...]
    sha256: str
    qualification_instance_id: str = ""
    qualification_boot_id: str = ""
    environment_receipt_sha256: str = ""
    canary_receipt_sha256: str = ""
    approval_receipt_sha256: str = ""
    approval_public_key_sha256: str = ""
    identity_verified: bool = False
    approval_verified: bool = False


@dataclass(frozen=True)
class AwsRuntimeLock:
    schema_version: int
    source_commit: str
    source_tree: str
    control_bundle_sha256: str
    profile_sha256: str
    ami_id: str
    ami_owner_id: str
    container_image: str
    container_image_digest: str
    versions: tuple[tuple[str, str], ...]
    sha256: str


@dataclass(frozen=True)
class AwsRuntimeEvidence:
    schema_version: int
    receipt_type: str
    profile_sha256: str
    runtime_lock_sha256: str
    account_id: str
    region: str
    availability_zone: str
    versions: tuple[tuple[str, str], ...]
    sha256: str
    instance_id: str = ""
    boot_id: str = ""
    ami_id: str = ""
    container_image_digest: str = ""
    environment_receipt_sha256: str = ""
    canary_receipt_sha256: str = ""
    approval_receipt_sha256: str = ""
    approval_public_key_sha256: str = ""
    host_facts_sha256: str = ""
    container_facts_sha256: str = ""
    identity_verified: bool = False
    approval_verified: bool = False


@dataclass(frozen=True)
class VersionedSelectionObject:
    key: str
    sha256: str
    bytes: int
    version_id: str


@dataclass(frozen=True)
class PublishedProviderSelection:
    selection: AwsProviderSelectionReceipt
    local_path: Path
    remote: VersionedSelectionObject
    version_path: Path
    publication_state: str


@dataclass(frozen=True)
class VersionedSelectionHistory:
    key: str
    versions: tuple[str, ...]
    delete_markers: tuple[str, ...]


@dataclass(frozen=True)
class VersionedSelectionRead:
    data: bytes
    object: VersionedSelectionObject


@dataclass(frozen=True)
class AuthenticatedSelectionBinding:
    cohort_id: str
    amendment_sha256: str
    selection_sha256: str
    selection_version_id: str
    profile_id: str
    provider: str
    profile_sha256: str
    runtime_lock_sha256: str
    qualification_evidence_sha256: str
    environment_receipt_sha256: str
    canary_receipt_sha256: str
    approval_receipt_sha256: str
    approval_public_key_sha256: str
    account_id: str
    instance_id: str
    boot_id: str
    region: str
    availability_zone: str
    purchase_model: str
    seed: int
    arm: str


class VersionedProviderSelectionStore(Protocol):
    def list_versions(self, *, key: str) -> VersionedSelectionHistory: ...

    def put_if_none_match(
        self,
        *,
        key: str,
        data: bytes,
        if_none_match: str,
        checksum_sha256: str,
    ) -> VersionedSelectionObject | None: ...

    def head(
        self,
        *,
        key: str,
        version_id: str | None,
    ) -> VersionedSelectionObject | None: ...

    def get_exact(
        self,
        *,
        key: str,
        version_id: str,
    ) -> VersionedSelectionRead | None: ...


class QualificationApprovalVerifier(Protocol):
    def verify(
        self,
        *,
        payload: bytes,
        signature: str,
        algorithm: str,
        public_key_sha256: str,
    ) -> bool: ...


class SelectionPublicationError(ValueError):
    def __init__(self, message: str, *, publication_state: str) -> None:
        super().__init__(message)
        self.publication_state = publication_state


def verify_aws_instance_identity_pkcs7(
    identity: Mapping[str, object],
    pkcs7: str,
    region: str,
) -> bool:
    """Use the repository's pinned AWS identity-certificate verifier."""

    from msctl.aws_p5 import _verify_instance_identity_pkcs7

    return _verify_instance_identity_pkcs7(identity, pkcs7, region)


class OpenSslQualificationApprovalVerifier:
    """Verify qualification approval signatures against a pinned public key."""

    def __init__(
        self,
        *,
        public_key_path: Path | str,
        environment: Mapping[str, str],
        runner: Callable[
            [Sequence[str], Mapping[str, str], float], object
        ]
        | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("qualification signature timeout must be positive")
        self.public_key_data = read_secure_regular_file(
            public_key_path,
            label="qualification approval public key",
            max_bytes=64 * 1024,
        )
        self.public_key_sha256 = hashlib.sha256(
            self.public_key_data
        ).hexdigest()
        self.environment = dict(environment)
        self.runner = runner or _default_aws_runner
        self.timeout_seconds = float(timeout_seconds)

    def verify(
        self,
        *,
        payload: bytes,
        signature: str,
        algorithm: str,
        public_key_sha256: str,
    ) -> bool:
        if (
            not isinstance(payload, bytes)
            or not payload
            or algorithm != "RSASSA_PSS_SHA_256"
            or public_key_sha256 != self.public_key_sha256
        ):
            return False
        try:
            signature_data = base64.b64decode(signature, validate=True)
        except (TypeError, ValueError):
            return False
        if not signature_data:
            return False
        with tempfile.TemporaryDirectory(
            prefix="memorysplit-qualification-signature-"
        ) as directory:
            root = Path(directory)
            root.chmod(0o700)
            public_key = root / "approval-public-key.pem"
            message = root / "approval-scope.json"
            signature_path = root / "approval-signature.bin"
            for path, value in (
                (public_key, self.public_key_data),
                (message, payload),
                (signature_path, signature_data),
            ):
                path.write_bytes(value)
                path.chmod(0o600)
            result = self.runner(
                [
                    "/usr/bin/openssl",
                    "dgst",
                    "-sha256",
                    "-sigopt",
                    "rsa_padding_mode:pss",
                    "-verify",
                    str(public_key),
                    "-signature",
                    str(signature_path),
                    str(message),
                ],
                self.environment,
                self.timeout_seconds,
            )
        return (
            getattr(result, "returncode", None) == 0
            and getattr(result, "stdout", None) == "Verified OK\n"
            and getattr(result, "stderr", None) == ""
        )


def _default_aws_runner(
    argv: Sequence[str],
    environment: Mapping[str, str],
    timeout_seconds: float,
):
    return subprocess.run(
        list(argv),
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )


def _aws_command_json(
    result: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, object]:
    if (
        getattr(result, "returncode", None) != 0
        or getattr(result, "stderr", None) != ""
        or not isinstance(getattr(result, "stdout", None), str)
    ):
        raise ValueError(f"{label} AWS command failed")
    try:
        value = json.loads(
            result.stdout,
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value: {constant}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} AWS output is not JSON") from error
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} AWS output fields do not match")
    return value


def _checksum_hex(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} checksum is missing")
    try:
        decoded = base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ValueError(f"{label} checksum is not canonical base64") from error
    if (
        len(decoded) != 32
        or base64.b64encode(decoded).decode("ascii") != value
    ):
        raise ValueError(f"{label} checksum is not SHA-256")
    return decoded.hex()


class AwsCliVersionedSelectionStore:
    """AWS CLI adapter for the sole versioned provider-selection key."""

    def __init__(
        self,
        *,
        bucket: str,
        region: str,
        environment: Mapping[str, str],
        staging_root: Path | str,
        runner: Callable[
            [Sequence[str], Mapping[str, str], float], object
        ] = _default_aws_runner,
        timeout_seconds: float = 120.0,
    ) -> None:
        if (
            not isinstance(bucket, str)
            or _S3_BUCKET_RE.fullmatch(bucket) is None
        ):
            raise ValueError("selection S3 bucket is invalid")
        if not isinstance(region, str) or _REGION_RE.fullmatch(region) is None:
            raise ValueError("selection AWS region is invalid")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("selection AWS timeout must be positive")
        root = Path(staging_root)
        root.mkdir(parents=True, mode=0o700, exist_ok=True)
        metadata = root.stat(follow_symlinks=False)
        if (
            root.is_symlink()
            or not root.is_dir()
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
        ):
            raise ValueError("selection staging root must be private and owned")
        self.bucket = bucket
        self.region = region
        self.environment = dict(environment)
        self.staging_root = root
        self.runner = runner
        self.timeout_seconds = float(timeout_seconds)

    def _argv(self, *arguments: str, query: str) -> list[str]:
        return [
            "aws",
            "--no-cli-pager",
            "--region",
            self.region,
            *arguments,
            "--output",
            "json",
            "--query",
            query,
        ]

    @staticmethod
    def _fixed_key(key: str) -> None:
        if key != PROVIDER_SELECTION_S3_KEY:
            raise ValueError("selection store only permits the fixed cohort key")

    def list_versions(self, *, key: str) -> VersionedSelectionHistory:
        self._fixed_key(key)
        result = self.runner(
            self._argv(
                "s3api",
                "list-object-versions",
                "--bucket",
                self.bucket,
                "--prefix",
                key,
                query=(
                    "{history:{versions:Versions[].{key:Key,"
                    "version_id:VersionId},delete_markers:"
                    "DeleteMarkers[].{key:Key,version_id:VersionId}}}"
                ),
            ),
            self.environment,
            self.timeout_seconds,
        )
        root = _aws_command_json(
            result,
            fields=frozenset({"history"}),
            label="selection version history",
        )
        history = _object(
            root["history"],
            fields=frozenset({"versions", "delete_markers"}),
            label="selection version history",
        )

        def identities(value: object, *, label: str) -> tuple[str, ...]:
            if value is None:
                return ()
            if not isinstance(value, list):
                raise ValueError(f"{label} must be a list")
            found: list[str] = []
            for row in value:
                item = _object(
                    row,
                    fields=frozenset({"key", "version_id"}),
                    label=label,
                )
                version_id = item["version_id"]
                if (
                    item["key"] != key
                    or not isinstance(version_id, str)
                    or version_id in {"", "null"}
                ):
                    raise ValueError(f"{label} contains an invalid identity")
                found.append(version_id)
            return tuple(found)

        return VersionedSelectionHistory(
            key=key,
            versions=identities(history["versions"], label="selection versions"),
            delete_markers=identities(
                history["delete_markers"],
                label="selection delete markers",
            ),
        )

    def put_if_none_match(
        self,
        *,
        key: str,
        data: bytes,
        if_none_match: str,
        checksum_sha256: str,
    ) -> VersionedSelectionObject | None:
        self._fixed_key(key)
        if (
            not isinstance(data, bytes)
            or not data
            or if_none_match != "*"
            or _sha256(checksum_sha256, label="selection PUT SHA-256")
            != hashlib.sha256(data).hexdigest()
        ):
            raise ValueError("selection PUT identity is invalid")
        checksum = base64.b64encode(bytes.fromhex(checksum_sha256)).decode(
            "ascii"
        )
        with tempfile.NamedTemporaryFile(
            prefix="provider-selection-",
            suffix=".json",
            dir=self.staging_root,
            delete=False,
        ) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o600)
            body = Path(handle.name)
        try:
            result = self.runner(
                self._argv(
                    "s3api",
                    "put-object",
                    "--bucket",
                    self.bucket,
                    "--key",
                    key,
                    "--body",
                    str(body),
                    "--content-length",
                    str(len(data)),
                    "--checksum-algorithm",
                    "SHA256",
                    "--checksum-sha256",
                    checksum,
                    "--if-none-match",
                    "*",
                    query=(
                        "{object:{checksum_sha256:ChecksumSHA256,"
                        "version_id:VersionId}}"
                    ),
                ),
                self.environment,
                self.timeout_seconds,
            )
        finally:
            body.unlink(missing_ok=True)
        if getattr(result, "returncode", None) != 0:
            return None
        root = _aws_command_json(
            result,
            fields=frozenset({"object"}),
            label="selection PUT",
        )
        row = _object(
            root["object"],
            fields=frozenset({"checksum_sha256", "version_id"}),
            label="selection PUT",
        )
        version_id = row["version_id"]
        if (
            _checksum_hex(
                row["checksum_sha256"],
                label="selection PUT",
            )
            != checksum_sha256
            or not isinstance(version_id, str)
            or version_id in {"", "null"}
        ):
            raise ValueError("selection PUT response identity differs")
        return VersionedSelectionObject(
            key=key,
            sha256=checksum_sha256,
            bytes=len(data),
            version_id=version_id,
        )

    def head(
        self,
        *,
        key: str,
        version_id: str | None,
    ) -> VersionedSelectionObject | None:
        self._fixed_key(key)
        arguments = [
            "s3api",
            "head-object",
            "--bucket",
            self.bucket,
            "--key",
            key,
        ]
        if version_id is not None:
            if not isinstance(version_id, str) or version_id in {"", "null"}:
                return None
            arguments.extend(["--version-id", version_id])
        arguments.extend(["--checksum-mode", "ENABLED"])
        result = self.runner(
            self._argv(
                *arguments,
                query=(
                    "{object:{checksum_sha256:ChecksumSHA256,"
                    "content_length:ContentLength,version_id:VersionId}}"
                ),
            ),
            self.environment,
            self.timeout_seconds,
        )
        if getattr(result, "returncode", None) != 0:
            return None
        root = _aws_command_json(
            result,
            fields=frozenset({"object"}),
            label="selection HEAD",
        )
        row = _object(
            root["object"],
            fields=frozenset(
                {"checksum_sha256", "content_length", "version_id"}
            ),
            label="selection HEAD",
        )
        returned_version = row["version_id"]
        if (
            type(row["content_length"]) is not int
            or row["content_length"] <= 0
            or not isinstance(returned_version, str)
            or returned_version in {"", "null"}
            or (
                version_id is not None
                and returned_version != version_id
            )
        ):
            return None
        return VersionedSelectionObject(
            key=key,
            sha256=_checksum_hex(
                row["checksum_sha256"],
                label="selection HEAD",
            ),
            bytes=row["content_length"],
            version_id=returned_version,
        )

    def get_exact(
        self,
        *,
        key: str,
        version_id: str,
    ) -> VersionedSelectionRead | None:
        self._fixed_key(key)
        if not isinstance(version_id, str) or version_id in {"", "null"}:
            return None
        destination = self.staging_root / (
            f"provider-selection-get-{secrets.token_hex(12)}.json"
        )
        try:
            result = self.runner(
                self._argv(
                    "s3api",
                    "get-object",
                    "--bucket",
                    self.bucket,
                    "--key",
                    key,
                    "--version-id",
                    version_id,
                    "--checksum-mode",
                    "ENABLED",
                    str(destination),
                    query=(
                        "{object:{checksum_sha256:ChecksumSHA256,"
                        "content_length:ContentLength,version_id:VersionId}}"
                    ),
                ),
                self.environment,
                self.timeout_seconds,
            )
            if getattr(result, "returncode", None) != 0:
                return None
            root = _aws_command_json(
                result,
                fields=frozenset({"object"}),
                label="selection GET",
            )
            row = _object(
                root["object"],
                fields=frozenset(
                    {"checksum_sha256", "content_length", "version_id"}
                ),
                label="selection GET",
            )
            data = read_secure_regular_file(
                destination,
                label="selection GET body",
                max_bytes=_MAX_JSON_BYTES,
            )
        finally:
            destination.unlink(missing_ok=True)
        returned_version = row["version_id"]
        digest = hashlib.sha256(data).hexdigest()
        if (
            type(row["content_length"]) is not int
            or row["content_length"] != len(data)
            or _checksum_hex(
                row["checksum_sha256"],
                label="selection GET",
            )
            != digest
            or returned_version != version_id
        ):
            return None
        return VersionedSelectionRead(
            data=data,
            object=VersionedSelectionObject(
                key=key,
                sha256=digest,
                bytes=len(data),
                version_id=version_id,
            ),
        )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON contains duplicate key: {key}")
        value[key] = item
    return value


def _closed_equal(actual: object, expected: object, *, label: str) -> None:
    if type(actual) is not type(expected):
        raise ValueError(f"{label} has the wrong JSON type")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"{label} has missing or unknown fields")
        for key, expected_item in expected.items():
            _closed_equal(actual[key], expected_item, label=f"{label}.{key}")
        return
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{label} has the wrong number of items")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _closed_equal(
                actual_item,
                expected_item,
                label=f"{label}[{index}]",
            )
        return
    if actual != expected:
        raise ValueError(f"{label} does not match the frozen contract")


def _json_object(data: bytes, *, label: str) -> dict[str, object]:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} bytes must be bytes")
    if len(data) > _MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds 64 KiB")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite value: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must contain valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def parse_aws_hardware_amendment_bytes(data: bytes) -> AwsHardwareAmendment:
    """Parse exact amendment bytes under the closed prospective contract."""

    value = _json_object(data, label="hardware amendment")
    _closed_equal(value, _EXPECTED_AMENDMENT, label="hardware amendment")
    digest = hashlib.sha256(data).hexdigest()
    if digest != AWS_HARDWARE_AMENDMENT_SHA256:
        raise ValueError("hardware amendment bytes do not match the frozen hash")
    profiles = tuple(
        HardwareProfileBinding(
            path=profile["path"],
            profile_id=profile["profile_id"],
            provider=profile["provider"],
            sha256=profile["sha256"],
        )
        for profile in value["eligible_profiles"]
    )
    frozen = value["frozen_contract"]
    return AwsHardwareAmendment(
        schema_version=value["schema_version"],
        amendment_id=value["amendment_id"],
        amendment_mode=value["amendment_mode"],
        prospective=value["prospective"],
        cohort_id=value["cohort_id"],
        preregistration=ArtifactBinding(**value["preregistration"]),
        cohort_assignment=ArtifactBinding(**value["cohort_assignment"]),
        profiles=profiles,
        supersedes_only=tuple(value["supersedes_only"]),
        seeds=tuple(frozen["seeds"]),
        arms=tuple(frozen["arms"]),
        train_groups=tuple(frozen["train_groups"]),
        symmetric_training=frozen["symmetric_training"],
        one_profile_for_entire_cohort=frozen[
            "one_profile_for_entire_cohort"
        ],
        scientific_invariants=frozen["scientific_invariants"],
        protected_outcomes_inspected=tuple(
            frozen["protected_outcomes_inspected"]
        ),
        sha256=digest,
    )


def _regular_bytes(
    path: Path,
    *,
    label: str,
    private: bool = False,
) -> bytes:
    return read_secure_regular_file(
        path,
        label=label,
        max_bytes=_MAX_JSON_BYTES,
        private=private,
    )


def _validate_aws_hardware_amendment_files(
    amendment: AwsHardwareAmendment,
    *,
    repo_root: Path | str,
) -> None:
    """Verify every append-only binding against one repository root."""

    root = Path(repo_root)
    bindings = (
        amendment.preregistration,
        amendment.cohort_assignment,
        *amendment.profiles,
    )
    for binding in bindings:
        candidate = root.joinpath(*binding.path.split("/"))
        data = _regular_bytes(candidate, label=binding.path)
        if hashlib.sha256(data).hexdigest() != binding.sha256:
            raise ValueError(f"amendment binding hash mismatch: {binding.path}")
    for binding in amendment.profiles:
        profile = load_aws_gpu_profile(root / binding.path)
        if (
            profile.profile_id != binding.profile_id
            or profile.provider != binding.provider
            or profile.sha256 != binding.sha256
        ):
            raise ValueError(
                f"amendment profile identity mismatch: {binding.profile_id}"
            )


def load_aws_hardware_amendment(
    path: Path | str,
    *,
    repo_root: Path | str | None = None,
) -> AwsHardwareAmendment:
    """Load a regular amendment and optionally verify all bound repository files."""

    candidate = Path(path)
    amendment = parse_aws_hardware_amendment_bytes(
        _regular_bytes(candidate, label="hardware amendment")
    )
    if repo_root is not None:
        _validate_aws_hardware_amendment_files(amendment, repo_root=repo_root)
    return amendment


def validate_aws_hardware_amendment_files(
    *,
    repo_root: Path | str,
) -> AwsHardwareAmendment:
    """Load the fixed amendment bytes, then verify every bound file."""

    amendment = parse_aws_hardware_amendment_bytes(
        _regular_bytes(
            Path(repo_root).joinpath(*AWS_HARDWARE_AMENDMENT_PATH.split("/")),
            label="hardware amendment",
        )
    )
    _validate_aws_hardware_amendment_files(amendment, repo_root=repo_root)
    return amendment


def _object(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{label} has missing or unknown fields; "
            f"missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )
    return value


def _exact_string(value: object, expected: str, *, label: str) -> str:
    if not isinstance(value, str) or value != expected:
        raise ValueError(f"{label} must be exactly {expected!r}")
    return value


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _canonical_json(value: object) -> bytes:
    try:
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
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("provider selection is not canonical JSON") from error


def _profile_binding(
    value: object,
    *,
    amendment: AwsHardwareAmendment,
) -> HardwareProfileBinding:
    profile = _object(
        value,
        fields=frozenset({"path", "profile_id", "provider", "sha256"}),
        label="provider selection profile",
    )
    for field in ("path", "profile_id", "provider"):
        if not isinstance(profile[field], str):
            raise ValueError(f"provider selection profile {field} must be a string")
    binding = HardwareProfileBinding(
        path=profile["path"],
        profile_id=profile["profile_id"],
        provider=profile["provider"],
        sha256=_sha256(
            profile["sha256"],
            label="provider selection profile SHA-256",
        ),
    )
    if binding not in amendment.profiles:
        raise ValueError(
            "provider selection profile is not one exact eligible profile"
        )
    return binding


def _parse_timestamp(value: object) -> str:
    if (
        not isinstance(value, str)
        or _UTC_TIMESTAMP_RE.fullmatch(value) is None
    ):
        raise ValueError(
            "provider selection timestamp must be UTC to exact seconds"
        )
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError("provider selection timestamp is not a valid date") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError("provider selection timestamp is not canonical")
    return value


def _validate_placement(
    *,
    profile: HardwareProfileBinding,
    account_id: object,
    region: object,
    availability_zone: object,
    purchase_model: object,
) -> tuple[str, str, str, str]:
    if (
        not isinstance(account_id, str)
        or _ACCOUNT_RE.fullmatch(account_id) is None
    ):
        raise ValueError("AWS account ID must be exactly 12 decimal digits")
    if not isinstance(region, str) or _REGION_RE.fullmatch(region) is None:
        raise ValueError("AWS region is invalid")
    if (
        not isinstance(availability_zone, str)
        or (match := _AVAILABILITY_ZONE_RE.fullmatch(availability_zone)) is None
        or match.group("region") != region
    ):
        raise ValueError("AWS availability zone must belong to its region")
    expected_purchase = {
        "aws-p5.48xlarge-v3": "on_demand",
        "aws-p6-b300.48xlarge-v3": "on_demand",
    }[profile.profile_id]
    if purchase_model != expected_purchase:
        raise ValueError("AWS purchase model does not match the selected profile")
    if region != "us-east-1":
        raise ValueError("AWS region is not allowed by the selected profile")
    if (
        profile.profile_id == "aws-p6-b300.48xlarge-v3"
        and availability_zone != "us-east-1d"
    ):
        raise ValueError("P6-B300 selection requires its us-east-1d offering")
    return account_id, region, availability_zone, expected_purchase


def _parse_provider_selection_receipt_bytes(
    data: bytes,
    *,
    amendment: AwsHardwareAmendment,
) -> AwsProviderSelectionReceipt:
    """Parse the sole canonical, cohort-wide provider-selection receipt."""

    if not isinstance(amendment, AwsHardwareAmendment):
        raise TypeError("hardware amendment must be parsed before selection")
    if amendment.sha256 != AWS_HARDWARE_AMENDMENT_SHA256:
        raise ValueError("provider selection requires the frozen amendment")
    value = _json_object(data, label="provider selection receipt")
    if data != _canonical_json(value):
        raise ValueError(
            "provider selection receipt must use its canonical JSON bytes"
        )
    root = _object(
        value,
        fields=frozenset(
            {
                "amendment",
                "authority",
                "aws",
                "cohort",
                "profile",
                "protected_outcomes_inspected",
                "receipt_type",
                "replacement_policy",
                "runtime",
                "schema_version",
                "selected_at",
                "selection_scope",
            }
        ),
        label="provider selection receipt",
    )
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise ValueError("provider selection schema_version must be integer 1")
    receipt_type = _exact_string(
        root["receipt_type"],
        "memorysplit-aws-provider-selection-v1",
        label="provider selection receipt_type",
    )
    selection_scope = _exact_string(
        root["selection_scope"],
        "entire_cohort",
        label="provider selection scope",
    )
    replacement_policy = _exact_string(
        root["replacement_policy"],
        "forbidden",
        label="provider selection replacement policy",
    )
    authority = _object(
        root["authority"],
        fields=frozenset({"local_path", "s3_key"}),
        label="provider selection authority",
    )
    authority_local_path = _exact_string(
        authority["local_path"],
        PROVIDER_SELECTION_LOCAL_PATH,
        label="provider selection local authority path",
    )
    authority_s3_key = _exact_string(
        authority["s3_key"],
        PROVIDER_SELECTION_S3_KEY,
        label="provider selection S3 authority key",
    )

    amendment_value = _object(
        root["amendment"],
        fields=frozenset({"path", "sha256"}),
        label="provider selection amendment",
    )
    amendment_binding = ArtifactBinding(
        path=_exact_string(
            amendment_value["path"],
            AWS_HARDWARE_AMENDMENT_PATH,
            label="provider selection amendment path",
        ),
        sha256=_sha256(
            amendment_value["sha256"],
            label="provider selection amendment SHA-256",
        ),
    )
    if amendment_binding.sha256 != amendment.sha256:
        raise ValueError("provider selection amendment hash does not match")

    cohort = _object(
        root["cohort"],
        fields=frozenset(
            {
                "arms",
                "cohort_assignment_sha256",
                "cohort_id",
                "preregistration_sha256",
                "seeds",
                "train_groups",
            }
        ),
        label="provider selection cohort",
    )
    cohort_id = _exact_string(
        cohort["cohort_id"],
        amendment.cohort_id,
        label="provider selection cohort ID",
    )
    preregistration_sha256 = _sha256(
        cohort["preregistration_sha256"],
        label="provider selection preregistration SHA-256",
    )
    cohort_assignment_sha256 = _sha256(
        cohort["cohort_assignment_sha256"],
        label="provider selection cohort-assignment SHA-256",
    )
    if preregistration_sha256 != amendment.preregistration.sha256:
        raise ValueError("provider selection preregistration hash does not match")
    if cohort_assignment_sha256 != amendment.cohort_assignment.sha256:
        raise ValueError("provider selection cohort-assignment hash does not match")
    for field, expected in (
        ("seeds", list(amendment.seeds)),
        ("arms", list(amendment.arms)),
        ("train_groups", list(amendment.train_groups)),
    ):
        _closed_equal(
            cohort[field],
            expected,
            label=f"provider selection cohort {field}",
        )
    profile = _profile_binding(root["profile"], amendment=amendment)

    runtime = _object(
        root["runtime"],
        fields=frozenset(
            {
                "runtime_lock_sha256",
                "runtime_evidence_sha256",
                "ami_id",
                "container_image_digest",
            }
        ),
        label="provider selection runtime",
    )
    runtime_lock_sha256 = _sha256(
        runtime["runtime_lock_sha256"],
        label="provider selection runtime-lock SHA-256",
    )
    runtime_evidence_sha256 = _sha256(
        runtime["runtime_evidence_sha256"],
        label="provider selection runtime-evidence SHA-256",
    )
    ami_id = runtime["ami_id"]
    if not isinstance(ami_id, str) or _AMI_RE.fullmatch(ami_id) is None:
        raise ValueError("provider selection AMI must be one immutable AMI ID")
    container_image_digest = runtime["container_image_digest"]
    if (
        not isinstance(container_image_digest, str)
        or _OCI_DIGEST_RE.fullmatch(container_image_digest) is None
    ):
        raise ValueError(
            "provider selection image digest must be canonical sha256"
        )

    aws = _object(
        root["aws"],
        fields=frozenset(
            {"account_id", "availability_zone", "purchase_model", "region"}
        ),
        label="provider selection AWS placement",
    )
    account_id, region, availability_zone, purchase_model = _validate_placement(
        profile=profile,
        account_id=aws["account_id"],
        region=aws["region"],
        availability_zone=aws["availability_zone"],
        purchase_model=aws["purchase_model"],
    )
    outcomes = root["protected_outcomes_inspected"]
    if not isinstance(outcomes, list) or outcomes:
        raise ValueError(
            "provider selection requires zero protected outcomes inspected"
        )

    return AwsProviderSelectionReceipt(
        schema_version=1,
        receipt_type=receipt_type,
        amendment=amendment_binding,
        authority_local_path=authority_local_path,
        authority_s3_key=authority_s3_key,
        cohort_id=cohort_id,
        preregistration_sha256=preregistration_sha256,
        cohort_assignment_sha256=cohort_assignment_sha256,
        profile=profile,
        seeds=amendment.seeds,
        arms=amendment.arms,
        train_groups=amendment.train_groups,
        runtime_lock_sha256=runtime_lock_sha256,
        runtime_evidence_sha256=runtime_evidence_sha256,
        ami_id=ami_id,
        container_image_digest=container_image_digest,
        account_id=account_id,
        region=region,
        availability_zone=availability_zone,
        purchase_model=purchase_model,
        selected_at=_parse_timestamp(root["selected_at"]),
        selection_scope=selection_scope,
        replacement_policy=replacement_policy,
        protected_outcomes_inspected=(),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def parse_provider_selection_receipt_bytes(
    data: bytes,
    *,
    amendment_data: bytes,
) -> AwsProviderSelectionReceipt:
    """Parse selection bytes against the immutable amendment bytes."""

    amendment = parse_aws_hardware_amendment_bytes(amendment_data)
    return _parse_provider_selection_receipt_bytes(data, amendment=amendment)


def parse_aws_runtime_lock_bytes(data: bytes) -> AwsRuntimeLock:
    """Parse one exact canonical runtime lock from authenticated bytes."""

    value = _json_object(data, label="runtime lock")
    if data != _canonical_json(value):
        raise ValueError("runtime lock must use canonical JSON bytes")
    lock = _object(
        value,
        fields=frozenset(AWS_RUNTIME_LOCK_FIELDS),
        label="runtime lock",
    )
    if type(lock["schema_version"]) is not int or lock["schema_version"] != 1:
        raise ValueError("runtime lock schema_version must be integer 1")
    for field in ("source_commit", "source_tree"):
        if (
            not isinstance(lock[field], str)
            or _COMMIT_RE.fullmatch(lock[field]) is None
        ):
            raise ValueError(f"runtime lock {field} must be a full Git object")
    control_bundle_sha256 = _sha256(
        lock["control_bundle_sha256"],
        label="runtime lock control-bundle SHA-256",
    )
    profile_sha256 = _sha256(
        lock["profile_sha256"],
        label="runtime lock profile SHA-256",
    )
    ami_id = lock["ami_id"]
    if not isinstance(ami_id, str) or _AMI_RE.fullmatch(ami_id) is None:
        raise ValueError("runtime lock AMI ID is invalid")
    ami_owner_id = lock["ami_owner_id"]
    if (
        not isinstance(ami_owner_id, str)
        or _ACCOUNT_RE.fullmatch(ami_owner_id) is None
    ):
        raise ValueError("runtime lock AMI owner ID is invalid")
    try:
        container_image, container_image_digest = (
            validate_digest_pinned_oci_image(
                lock["container_image"],
                lock["container_image_digest"],
            )
        )
    except ValueError as error:
        raise ValueError("runtime lock container image is not digest pinned") from error
    versions = _object(
        lock["versions"],
        fields=frozenset(AWS_RUNTIME_VERSION_FIELDS),
        label="runtime lock versions",
    )
    normalized_versions: list[tuple[str, str]] = []
    for field in AWS_RUNTIME_VERSION_FIELDS:
        version = versions[field]
        if (
            not isinstance(version, str)
            or not 1 <= len(version) <= 128
            or any(character in version for character in "\x00\n\r")
        ):
            raise ValueError(f"runtime lock versions.{field} is invalid")
        normalized_versions.append((field, version))
    return AwsRuntimeLock(
        schema_version=1,
        source_commit=lock["source_commit"],
        source_tree=lock["source_tree"],
        control_bundle_sha256=control_bundle_sha256,
        profile_sha256=profile_sha256,
        ami_id=ami_id,
        ami_owner_id=ami_owner_id,
        container_image=container_image,
        container_image_digest=container_image_digest,
        versions=tuple(normalized_versions),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _version_parts(value: str, *, label: str) -> tuple[int, ...]:
    normalized = value[1:] if value.startswith("R") else value
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", normalized) is None:
        raise ValueError(f"{label} must be a numeric version or R branch")
    return tuple(int(part) for part in normalized.split("."))


def _version_at_least(value: str, floor: str, *, label: str) -> None:
    actual = _version_parts(value, label=label)
    minimum = _version_parts(floor, label=f"{label} floor")
    width = max(len(actual), len(minimum))
    if actual + (0,) * (width - len(actual)) < minimum + (0,) * (
        width - len(minimum)
    ):
        raise ValueError(f"{label} is below the selected profile floor")


def _canonical_base64(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty canonical base64")
    try:
        decoded = base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ValueError(f"{label} must be canonical base64") from error
    if (
        not decoded
        or base64.b64encode(decoded).decode("ascii") != value
    ):
        raise ValueError(f"{label} must be canonical base64")
    return value


def parse_authenticated_qualification_evidence_bytes(
    data: bytes,
    *,
    profile_data: bytes,
    runtime_lock_data: bytes,
    identity_verifier: Callable[
        [Mapping[str, object], str, str], bool
    ],
    approval_verifier: QualificationApprovalVerifier,
    trusted_public_key_sha256: str,
) -> AwsRuntimeEvidence:
    """Verify signed environment and canary qualification evidence."""

    profile = parse_aws_gpu_profile_bytes(profile_data)
    runtime_lock = parse_aws_runtime_lock_bytes(runtime_lock_data)
    if runtime_lock.profile_sha256 != profile.sha256:
        raise ValueError("qualification runtime lock does not bind profile")
    trusted_key = _sha256(
        trusted_public_key_sha256,
        label="trusted qualification public key",
    )
    value = _json_object(data, label="qualification evidence")
    if data != _canonical_json(value):
        raise ValueError("qualification evidence must use canonical JSON bytes")
    root = _object(
        value,
        fields=frozenset(
            {
                "approval",
                "availability_zone",
                "canary_receipt",
                "environment_receipt",
                "receipt_type",
                "schema_version",
            }
        ),
        label="qualification evidence",
    )
    if (
        type(root["schema_version"]) is not int
        or root["schema_version"] != 2
        or root["receipt_type"] != "memorysplit-aws-qualified-runtime-v2"
    ):
        raise ValueError("qualification evidence identity is invalid")

    environment = _object(
        root["environment_receipt"],
        fields=frozenset(AWS_ENVIRONMENT_RECEIPT_V2_FIELDS),
        label="qualification environment receipt",
    )
    if (
        type(environment["schema_version"]) is not int
        or environment["schema_version"] != 2
        or environment["receipt_type"] != "memorysplit-aws-environment-v2"
        or environment["provider"] != profile.provider
    ):
        raise ValueError("qualification environment receipt identity is invalid")
    environment_profile = _sha256(
        environment["profile_sha256"],
        label="qualification environment profile",
    )
    environment_runtime = _sha256(
        environment["runtime_lock_sha256"],
        label="qualification environment runtime lock",
    )
    if (
        environment_profile != profile.sha256
        or environment_runtime != runtime_lock.sha256
        or environment["control_bundle_sha256"]
        != runtime_lock.control_bundle_sha256
        or environment["source_commit"] != runtime_lock.source_commit
        or environment["source_tree"] != runtime_lock.source_tree
        or environment["ami_id"] != runtime_lock.ami_id
        or environment["container_image"] != runtime_lock.container_image
        or environment["container_image_digest"]
        != runtime_lock.container_image_digest
    ):
        raise ValueError("qualification environment differs from runtime lock")
    identity = environment["aws_instance_identity_document"]
    if not isinstance(identity, dict):
        raise ValueError("qualification instance identity must be an object")
    identity_fields = set(identity)
    if (
        not _IDENTITY_REQUIRED_FIELDS <= identity_fields
        or not identity_fields
        <= _IDENTITY_REQUIRED_FIELDS | _IDENTITY_OPTIONAL_FIELDS
        or not all(
            isinstance(identity[field], str)
            for field in _IDENTITY_REQUIRED_FIELDS
        )
        or _ACCOUNT_RE.fullmatch(identity["accountId"]) is None
        or _INSTANCE_RE.fullmatch(identity["instanceId"]) is None
        or _REGION_RE.fullmatch(identity["region"]) is None
        or identity["architecture"] != profile.architecture
        or identity["imageId"] != runtime_lock.ami_id
    ):
        raise ValueError("qualification instance identity is invalid")
    selected_availability_zone = root["availability_zone"]
    if (
        not isinstance(selected_availability_zone, str)
        or (
            zone_match := _AVAILABILITY_ZONE_RE.fullmatch(
                selected_availability_zone
            )
        )
        is None
        or zone_match.group("region") != identity["region"]
        or (
            profile.allowed_availability_zones
            and selected_availability_zone
            not in profile.allowed_availability_zones
        )
    ):
        raise ValueError("qualification selected availability zone is invalid")
    if "availabilityZone" in identity and (
        not isinstance(identity["availabilityZone"], str)
        or identity["availabilityZone"] != selected_availability_zone
    ):
        raise ValueError("qualification identity availability zone differs")
    if "instanceType" in identity and (
        not isinstance(identity["instanceType"], str)
        or identity["instanceType"] != profile.instance_type
    ):
        raise ValueError("qualification identity instance type differs")
    for field in (
        "billingProducts",
        "devpayProductCodes",
        "marketplaceProductCodes",
    ):
        if field in identity and identity[field] is not None and (
            not isinstance(identity[field], list)
            or any(not isinstance(item, str) for item in identity[field])
        ):
            raise ValueError(f"qualification identity {field} is invalid")
    for field in ("kernelId", "pendingTime", "ramdiskId", "version"):
        if field in identity and identity[field] is not None and (
            not isinstance(identity[field], str) or not identity[field]
        ):
            raise ValueError(f"qualification identity {field} is invalid")
    try:
        address = ipaddress.ip_address(identity["privateIp"])
    except ValueError as error:
        raise ValueError("qualification private IP is invalid") from error
    if not address.is_private or address.version != 4:
        raise ValueError("qualification private IP must be private IPv4")
    account_id = environment["account_id"]
    instance_id = environment["instance_id"]
    region = environment["region"]
    ami_id = environment["ami_id"]
    boot_id = environment["boot_id"]
    availability_zone = selected_availability_zone
    if (
        account_id != identity["accountId"]
        or instance_id != identity["instanceId"]
        or region != identity["region"]
        or ami_id != identity["imageId"]
        or _BOOT_RE.fullmatch(str(boot_id)) is None
        or region not in profile.allowed_regions
    ):
        raise ValueError("qualification environment identity duplicates differ")
    pkcs7 = _canonical_base64(
        environment["aws_instance_identity_pkcs7"],
        label="qualification instance identity signature",
    )
    if not callable(identity_verifier):
        raise TypeError("qualification identity verifier is required")
    try:
        identity_verified = identity_verifier(identity, pkcs7, region)
    except Exception as error:
        raise ValueError(
            "qualification instance signature verification failed"
        ) from error
    if identity_verified is not True:
        raise ValueError("qualification instance signature is invalid")
    container_facts = _object(
        environment["runtime_facts"],
        fields=frozenset(AWS_RUNTIME_VERSION_FIELDS),
        label="qualification container facts",
    )
    if container_facts != dict(runtime_lock.versions):
        raise ValueError("qualification container facts differ from runtime lock")
    environment_data = _canonical_json(environment)
    environment_sha256 = hashlib.sha256(environment_data).hexdigest()

    canary = _object(
        root["canary_receipt"],
        fields=_CANARY_RECEIPT_FIELDS,
        label="qualification canary receipt",
    )
    if (
        type(canary["schema_version"]) is not int
        or canary["schema_version"] != 1
        or canary["receipt_type"]
        not in {
            "memorysplit-aws-p5-qualification-v1",
            "memorysplit-aws-gpu-qualification-v1",
        }
        or canary["provider"] != profile.provider
        or canary["passed"] is not True
        or canary["instance_id"] != instance_id
        or canary["boot_id"] != boot_id
        or canary["profile_sha256"] != profile.sha256
        or canary["runtime_lock_sha256"] != runtime_lock.sha256
        or canary["environment_receipt_sha256"] != environment_sha256
        or canary["container_image"] != runtime_lock.container_image
        or canary["container_image_digest"]
        != runtime_lock.container_image_digest
        or canary["source_commit"] != runtime_lock.source_commit
        or canary["source_tree"] != runtime_lock.source_tree
    ):
        raise ValueError("qualification canary identity is invalid")
    for field in (
        "release_sha256",
        "release_receipt_sha256",
        "run_manifest_sha256",
        "dataset_receipt_sha256",
        "dataset_build_id",
        "ordered_stream_sha256",
    ):
        _sha256(canary[field], label=f"qualification canary {field}")
    phases = canary["phases"]
    if (
        not isinstance(phases, list)
        or [phase.get("name") for phase in phases if isinstance(phase, dict)]
        != list(_CANARY_PHASE_ORDER)
        or any(
            set(phase) != {"name", "passed", "seconds"}
            or phase["passed"] is not True
            or isinstance(phase["seconds"], bool)
            or not isinstance(phase["seconds"], (int, float))
            or not math.isfinite(float(phase["seconds"]))
            or float(phase["seconds"]) < 0
            for phase in phases
            if isinstance(phase, dict)
        )
        or any(not isinstance(phase, dict) for phase in phases)
    ):
        raise ValueError("qualification canary phases are invalid")
    hardware = canary["hardware"]
    if not isinstance(hardware, dict):
        raise ValueError("qualification canary hardware must be an object")
    host_versions = _object(
        hardware.get("runtime_facts"),
        fields=frozenset(dict(profile.software_floors)),
        label="qualification host facts",
    )
    normalized_versions: list[tuple[str, str]] = []
    for field, floor in profile.software_floors:
        version = host_versions[field]
        if not isinstance(version, str):
            raise ValueError(f"qualification host {field} must be a string")
        _version_at_least(
            version,
            floor,
            label=f"qualification host {field}",
        )
        normalized_versions.append((field, version))
    canary_data = _canonical_json(canary)
    canary_sha256 = hashlib.sha256(canary_data).hexdigest()
    host_facts_sha256 = hashlib.sha256(
        _canonical_json(hardware)
    ).hexdigest()
    container_facts_sha256 = hashlib.sha256(
        _canonical_json(container_facts)
    ).hexdigest()

    expected_scope = {
        "account_id": account_id,
        "ami_id": ami_id,
        "availability_zone": availability_zone,
        "boot_id": boot_id,
        "canary_receipt_sha256": canary_sha256,
        "container_facts_sha256": container_facts_sha256,
        "container_image_digest": runtime_lock.container_image_digest,
        "environment_receipt_sha256": environment_sha256,
        "host_facts_sha256": host_facts_sha256,
        "instance_id": instance_id,
        "profile_sha256": profile.sha256,
        "runtime_lock_sha256": runtime_lock.sha256,
    }
    approval = _object(
        root["approval"],
        fields=frozenset(
            {
                "algorithm",
                "public_key_sha256",
                "scope",
                "scope_sha256",
                "signature",
            }
        ),
        label="qualification approval",
    )
    if approval["algorithm"] != "RSASSA_PSS_SHA_256":
        raise ValueError("qualification approval algorithm is invalid")
    approval_key = _sha256(
        approval["public_key_sha256"],
        label="qualification approval public key",
    )
    if approval_key != trusted_key:
        raise ValueError("qualification approval public key is not trusted")
    _closed_equal(
        approval["scope"],
        expected_scope,
        label="qualification approval scope",
    )
    scope_data = _canonical_json(expected_scope)
    scope_sha256 = _sha256(
        approval["scope_sha256"],
        label="qualification approval scope SHA-256",
    )
    if scope_sha256 != hashlib.sha256(scope_data).hexdigest():
        raise ValueError("qualification approval scope hash differs")
    signature = _canonical_base64(
        approval["signature"],
        label="qualification approval signature",
    )
    verify = getattr(approval_verifier, "verify", None)
    if not callable(verify):
        raise TypeError("qualification approval verifier is required")
    try:
        approval_verified = verify(
            payload=scope_data,
            signature=signature,
            algorithm=approval["algorithm"],
            public_key_sha256=approval_key,
        )
    except Exception as error:
        raise ValueError("qualification approval verification failed") from error
    if approval_verified is not True:
        raise ValueError("qualification approval signature is invalid")
    approval_data = _canonical_json(approval)
    return AwsRuntimeEvidence(
        schema_version=2,
        receipt_type=root["receipt_type"],
        profile_sha256=profile.sha256,
        runtime_lock_sha256=runtime_lock.sha256,
        account_id=account_id,
        region=region,
        availability_zone=availability_zone,
        versions=tuple(normalized_versions),
        sha256=hashlib.sha256(data).hexdigest(),
        instance_id=instance_id,
        boot_id=boot_id,
        ami_id=ami_id,
        container_image_digest=runtime_lock.container_image_digest,
        environment_receipt_sha256=environment_sha256,
        canary_receipt_sha256=canary_sha256,
        approval_receipt_sha256=hashlib.sha256(approval_data).hexdigest(),
        approval_public_key_sha256=approval_key,
        host_facts_sha256=host_facts_sha256,
        container_facts_sha256=container_facts_sha256,
        identity_verified=True,
        approval_verified=True,
    )


def parse_verified_provider_selection_bytes(
    data: bytes,
    *,
    amendment_data: bytes,
    profile_data: bytes,
    runtime_lock_data: bytes,
    runtime_evidence_data: bytes,
    identity_verifier: Callable[
        [Mapping[str, object], str, str], bool
    ],
    approval_verifier: QualificationApprovalVerifier,
    trusted_public_key_sha256: str,
) -> AwsProviderSelectionReceipt:
    """Verify selection authority from anchored profile and runtime bytes."""

    amendment = parse_aws_hardware_amendment_bytes(amendment_data)
    profile = parse_aws_gpu_profile_bytes(profile_data)
    eligible = tuple(
        binding
        for binding in amendment.profiles
        if binding.profile_id == profile.profile_id
        and binding.provider == profile.provider
        and binding.sha256 == profile.sha256
    )
    if len(eligible) != 1:
        raise ValueError("selected profile bytes are not amendment eligible")
    receipt = _parse_provider_selection_receipt_bytes(
        data,
        amendment=amendment,
    )
    if receipt.profile != eligible[0]:
        raise ValueError("selection does not match the anchored profile bytes")

    runtime_lock = parse_aws_runtime_lock_bytes(runtime_lock_data)
    if runtime_lock.sha256 != receipt.runtime_lock_sha256:
        raise ValueError("selection runtime-lock hash does not match its bytes")
    if runtime_lock.profile_sha256 != profile.sha256:
        raise ValueError("runtime lock does not bind the selected profile")
    if runtime_lock.ami_id != receipt.ami_id:
        raise ValueError("selection AMI does not match the runtime lock")
    if runtime_lock.container_image_digest != receipt.container_image_digest:
        raise ValueError("selection image digest does not match the runtime lock")

    evidence = parse_authenticated_qualification_evidence_bytes(
        runtime_evidence_data,
        profile_data=profile_data,
        runtime_lock_data=runtime_lock_data,
        identity_verifier=identity_verifier,
        approval_verifier=approval_verifier,
        trusted_public_key_sha256=trusted_public_key_sha256,
    )
    if evidence.sha256 != receipt.runtime_evidence_sha256:
        raise ValueError("selection runtime-evidence hash does not match its bytes")
    if evidence.profile_sha256 != profile.sha256:
        raise ValueError("runtime evidence does not bind the selected profile")
    if evidence.runtime_lock_sha256 != runtime_lock.sha256:
        raise ValueError("runtime evidence does not bind the runtime lock")
    if (
        evidence.account_id != receipt.account_id
        or evidence.region != receipt.region
        or evidence.availability_zone != receipt.availability_zone
    ):
        raise ValueError("runtime evidence placement differs from selection")
    lock_versions = dict(runtime_lock.versions)
    for field, floor in profile.software_floors:
        if field in lock_versions:
            _version_at_least(
                lock_versions[field],
                floor,
                label=f"runtime lock versions.{field}",
            )
    return replace(
        receipt,
        qualification_instance_id=evidence.instance_id,
        qualification_boot_id=evidence.boot_id,
        environment_receipt_sha256=evidence.environment_receipt_sha256,
        canary_receipt_sha256=evidence.canary_receipt_sha256,
        approval_receipt_sha256=evidence.approval_receipt_sha256,
        approval_public_key_sha256=evidence.approval_public_key_sha256,
        identity_verified=evidence.identity_verified,
        approval_verified=evidence.approval_verified,
    )


parse_aws_provider_selection_receipt_bytes = (
    parse_provider_selection_receipt_bytes
)


def canonical_provider_selection_receipt_bytes(
    value: object,
    *,
    amendment_data: bytes,
    profile_data: bytes,
    runtime_lock_data: bytes,
    runtime_evidence_data: bytes,
    identity_verifier: Callable[
        [Mapping[str, object], str, str], bool
    ],
    approval_verifier: QualificationApprovalVerifier,
    trusted_public_key_sha256: str,
) -> bytes:
    """Encode and verify one provider selection from anchored bytes."""

    data = _canonical_json(value)
    parse_verified_provider_selection_bytes(
        data,
        amendment_data=amendment_data,
        profile_data=profile_data,
        runtime_lock_data=runtime_lock_data,
        runtime_evidence_data=runtime_evidence_data,
        identity_verifier=identity_verifier,
        approval_verifier=approval_verifier,
        trusted_public_key_sha256=trusted_public_key_sha256,
    )
    return data


def _load_verified_selection_from_paths(
    selection_data: bytes,
    *,
    repo_root: Path | str,
    runtime_lock_path: Path | str,
    runtime_evidence_path: Path | str,
    identity_verifier: Callable[
        [Mapping[str, object], str, str], bool
    ],
    approval_verifier: QualificationApprovalVerifier,
    trusted_public_key_sha256: str,
) -> AwsProviderSelectionReceipt:
    amendment_data = _regular_bytes(
        Path(repo_root).joinpath(*AWS_HARDWARE_AMENDMENT_PATH.split("/")),
        label="hardware amendment",
    )
    amendment = parse_aws_hardware_amendment_bytes(amendment_data)
    preliminary = _parse_provider_selection_receipt_bytes(
        selection_data,
        amendment=amendment,
    )
    profile_data = _regular_bytes(
        Path(repo_root).joinpath(*preliminary.profile.path.split("/")),
        label="selected hardware profile",
    )
    runtime_lock_data = _regular_bytes(
        Path(runtime_lock_path),
        label="runtime lock",
        private=True,
    )
    runtime_evidence_data = _regular_bytes(
        Path(runtime_evidence_path),
        label="runtime evidence",
        private=True,
    )
    return parse_verified_provider_selection_bytes(
        selection_data,
        amendment_data=amendment_data,
        profile_data=profile_data,
        runtime_lock_data=runtime_lock_data,
        runtime_evidence_data=runtime_evidence_data,
        identity_verifier=identity_verifier,
        approval_verifier=approval_verifier,
        trusted_public_key_sha256=trusted_public_key_sha256,
    )


def _write_selection_noreplace(
    path: Path | str,
    data: bytes,
) -> Path:
    destination = Path(path)
    name = destination.name
    if not name or name in {".", ".."}:
        raise ValueError("provider selection destination name is invalid")
    directory_fd = open_directory(
        destination.parent,
        label="provider selection parent",
        create=True,
        mode=0o700,
    )
    temporary = f".{name}.{secrets.token_hex(12)}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        rename_noreplace_at(
            directory_fd,
            temporary,
            directory_fd,
            name,
        )
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)
    return destination


def _publish_local_selection(
    *,
    authority_root: Path | str,
    data: bytes,
) -> Path:
    destination = Path(authority_root).joinpath(
        *PROVIDER_SELECTION_LOCAL_PATH.split("/")
    )
    try:
        _write_selection_noreplace(destination, data)
    except FileExistsError as error:
        existing = _regular_bytes(
            destination,
            label="provider selection authority",
            private=True,
        )
        if existing != data:
            raise ValueError(
                "fixed local provider selection conflicts with different bytes"
            ) from error
    installed = _regular_bytes(
        destination,
        label="provider selection authority",
        private=True,
    )
    if installed != data:
        raise ValueError("fixed local provider selection differs after publication")
    return destination


def _preflight_local_selection(
    *,
    authority_root: Path | str,
    data: bytes,
) -> None:
    destination = Path(authority_root).joinpath(
        *PROVIDER_SELECTION_LOCAL_PATH.split("/")
    )
    try:
        existing = _regular_bytes(
            destination,
            label="provider selection authority",
            private=True,
        )
    except ValueError as error:
        cause = error.__cause__
        if isinstance(cause, FileNotFoundError) or (
            isinstance(cause, MsctlError) and cause.code == "FILE_NOT_FOUND"
        ):
            return
        raise
    if existing != data:
        raise ValueError(
            "fixed local provider selection conflicts with different bytes"
        )


def _matching_durable_selection_binding(
    *,
    authority_root: Path | str,
    data: bytes,
) -> bool:
    root = Path(authority_root)
    try:
        local_data = _regular_bytes(
            root.joinpath(*PROVIDER_SELECTION_LOCAL_PATH.split("/")),
            label="provider selection authority",
            private=True,
        )
        version_data = _regular_bytes(
            root.joinpath(*PROVIDER_SELECTION_VERSION_LOCAL_PATH.split("/")),
            label="provider selection version authority",
            private=True,
        )
    except ValueError as error:
        cause = error.__cause__
        if isinstance(cause, FileNotFoundError) or (
            isinstance(cause, MsctlError) and cause.code == "FILE_NOT_FOUND"
        ):
            return False
        raise
    if local_data != data:
        raise SelectionPublicationError(
            "durable local selection conflicts with pending candidate",
            publication_state="conflict",
        )
    value = _json_object(version_data, label="provider selection version")
    if (
        version_data != _canonical_json(value)
        or set(value)
        != {"schema_version", "selection_sha256", "s3_key", "version_id"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["selection_sha256"] != hashlib.sha256(data).hexdigest()
        or value["s3_key"] != PROVIDER_SELECTION_S3_KEY
        or not isinstance(value["version_id"], str)
        or value["version_id"] in {"", "null"}
    ):
        raise SelectionPublicationError(
            "durable selection version binding is invalid",
            publication_state="conflict",
        )
    return True


def _pending_selection_data(data: bytes) -> bytes:
    return _canonical_json(
        {
            "expected_bytes": len(data),
            "s3_key": PROVIDER_SELECTION_S3_KEY,
            "schema_version": 1,
            "selection_sha256": hashlib.sha256(data).hexdigest(),
        }
    )


def _publish_pending_selection(
    *,
    authority_root: Path | str,
    data: bytes,
) -> tuple[Path, bool]:
    pending_data = _pending_selection_data(data)
    destination = Path(authority_root).joinpath(
        *PROVIDER_SELECTION_PENDING_LOCAL_PATH.split("/")
    )
    existed = False
    try:
        _write_selection_noreplace(destination, pending_data)
    except FileExistsError as error:
        existed = True
        existing = _regular_bytes(
            destination,
            label="provider selection pending intent",
            private=True,
        )
        if existing != pending_data:
            raise SelectionPublicationError(
                "pending provider selection conflicts with candidate bytes",
                publication_state="conflict",
            ) from error
    installed = _regular_bytes(
        destination,
        label="provider selection pending intent",
        private=True,
    )
    if installed != pending_data:
        raise SelectionPublicationError(
            "pending provider selection differs after publication",
            publication_state="uncertain",
        )
    return destination, existed


def _remote_mutation_marker_path(authority_root: Path | str) -> Path:
    return Path(authority_root).joinpath(
        *PROVIDER_SELECTION_MUTATION_LOCAL_PATH.split("/")
    )


def _remote_mutation_marker_exists(
    *,
    authority_root: Path | str,
    data: bytes,
) -> bool:
    try:
        existing = _regular_bytes(
            _remote_mutation_marker_path(authority_root),
            label="provider selection remote-mutation marker",
            private=True,
        )
    except ValueError as error:
        cause = error.__cause__
        if isinstance(cause, FileNotFoundError) or (
            isinstance(cause, MsctlError) and cause.code == "FILE_NOT_FOUND"
        ):
            return False
        raise
    if existing != _pending_selection_data(data):
        raise SelectionPublicationError(
            "provider selection remote-mutation marker conflicts",
            publication_state="conflict",
        )
    return True


def _publish_remote_mutation_marker(
    *,
    authority_root: Path | str,
    data: bytes,
) -> Path:
    marker_data = _pending_selection_data(data)
    destination = _remote_mutation_marker_path(authority_root)
    try:
        _write_selection_noreplace(destination, marker_data)
    except FileExistsError as error:
        existing = _regular_bytes(
            destination,
            label="provider selection remote-mutation marker",
            private=True,
        )
        if existing != marker_data:
            raise SelectionPublicationError(
                "provider selection remote-mutation marker conflicts",
                publication_state="conflict",
            ) from error
    return destination


def _archive_pending_selection(
    *,
    authority_root: Path | str,
    data: bytes,
) -> Path:
    pending_data = _pending_selection_data(data)
    pending = Path(authority_root).joinpath(
        *PROVIDER_SELECTION_PENDING_LOCAL_PATH.split("/")
    )
    archive = Path(authority_root).joinpath(
        *PROVIDER_SELECTION_PENDING_ARCHIVE_LOCAL_PATH.split("/")
    )
    marker = _remote_mutation_marker_path(authority_root)
    marker_archive = Path(authority_root).joinpath(
        *PROVIDER_SELECTION_MUTATION_ARCHIVE_LOCAL_PATH.split("/")
    )
    parent_fd = open_directory(
        pending.parent,
        label="provider selection authority parent",
    )
    try:
        try:
            rename_noreplace_at(
                parent_fd,
                pending.name,
                parent_fd,
                archive.name,
            )
            os.fsync(parent_fd)
        except FileExistsError:
            archived = _regular_bytes(
                archive,
                label="provider selection pending archive",
                private=True,
            )
            current = _regular_bytes(
                pending,
                label="provider selection pending intent",
                private=True,
            )
            if archived != pending_data or current != pending_data:
                raise SelectionPublicationError(
                    "provider selection pending archive conflicts",
                    publication_state="conflict",
                )
            os.unlink(pending.name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        try:
            rename_noreplace_at(
                parent_fd,
                marker.name,
                parent_fd,
                marker_archive.name,
            )
            os.fsync(parent_fd)
        except FileExistsError:
            archived_marker = _regular_bytes(
                marker_archive,
                label="provider selection remote-mutation archive",
                private=True,
            )
            current_marker = _regular_bytes(
                marker,
                label="provider selection remote-mutation marker",
                private=True,
            )
            if (
                archived_marker != pending_data
                or current_marker != pending_data
            ):
                raise SelectionPublicationError(
                    "provider selection remote-mutation archive conflicts",
                    publication_state="conflict",
                )
            os.unlink(marker.name, dir_fd=parent_fd)
            os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    archived = _regular_bytes(
        archive,
        label="provider selection pending archive",
        private=True,
    )
    if archived != pending_data:
        raise SelectionPublicationError(
            "provider selection pending archive is not durable",
            publication_state="uncertain",
        )
    if _regular_bytes(
        marker_archive,
        label="provider selection remote-mutation archive",
        private=True,
    ) != pending_data:
        raise SelectionPublicationError(
            "provider selection remote-mutation archive is not durable",
            publication_state="uncertain",
        )
    return archive


def _verified_remote_object(
    value: object,
    *,
    expected_sha256: str,
    expected_bytes: int,
    expected_version: str | None,
) -> VersionedSelectionObject:
    if not isinstance(value, VersionedSelectionObject):
        raise ValueError("remote selection HEAD did not return an exact object")
    if (
        value.key != PROVIDER_SELECTION_S3_KEY
        or value.sha256 != expected_sha256
        or value.bytes != expected_bytes
        or not isinstance(value.version_id, str)
        or not value.version_id
        or value.version_id == "null"
        or any(character in value.version_id for character in "\x00\n\r")
        or (
            expected_version is not None
            and value.version_id != expected_version
        )
    ):
        raise ValueError("remote selection HEAD checksum, bytes, or version differs")
    return value


def _publish_remote_selection(
    *,
    authority_root: Path | str,
    data: bytes,
    store: VersionedProviderSelectionStore,
    allow_recovery: bool,
) -> tuple[VersionedSelectionObject, str]:
    digest = hashlib.sha256(data).hexdigest()
    history = store.list_versions(key=PROVIDER_SELECTION_S3_KEY)
    if (
        not isinstance(history, VersionedSelectionHistory)
        or history.key != PROVIDER_SELECTION_S3_KEY
    ):
        raise SelectionPublicationError(
            "fixed-key selection version history is invalid",
            publication_state="uncertain",
        )
    if history.versions or history.delete_markers:
        if (
            not allow_recovery
            or len(history.versions) != 1
            or history.delete_markers
        ):
            raise SelectionPublicationError(
                "fixed-key selection history conflicts with publication; "
                "multiple versions or delete markers are forbidden",
                publication_state="conflict",
            )
        version_id = history.versions[0]
        head = store.head(
            key=PROVIDER_SELECTION_S3_KEY,
            version_id=version_id,
        )
        remote = _verified_remote_object(
            head,
            expected_sha256=digest,
            expected_bytes=len(data),
            expected_version=version_id,
        )
        fetched = store.get_exact(
            key=PROVIDER_SELECTION_S3_KEY,
            version_id=version_id,
        )
        if (
            not isinstance(fetched, VersionedSelectionRead)
            or fetched.data != data
            or fetched.object != remote
        ):
            raise SelectionPublicationError(
                "pending selection recovery GET differs from candidate bytes",
                publication_state="conflict",
            )
        _require_singleton_selection_history(
            store=store,
            version_id=version_id,
        )
        return remote, "recovered"
    _publish_remote_mutation_marker(
        authority_root=authority_root,
        data=data,
    )
    put = store.put_if_none_match(
        key=PROVIDER_SELECTION_S3_KEY,
        data=data,
        if_none_match="*",
        checksum_sha256=digest,
    )
    if put is None:
        head_version = None
    else:
        put = _verified_remote_object(
            put,
            expected_sha256=digest,
            expected_bytes=len(data),
            expected_version=None,
        )
        head_version = put.version_id
    head = store.head(
        key=PROVIDER_SELECTION_S3_KEY,
        version_id=head_version,
    )
    try:
        remote = _verified_remote_object(
            head,
            expected_sha256=digest,
            expected_bytes=len(data),
            expected_version=head_version,
        )
    except ValueError as error:
        raise ValueError(
            "remote fixed-key provider selection conflicts or failed exact HEAD"
        ) from error
    _require_singleton_selection_history(
        store=store,
        version_id=remote.version_id,
    )
    state = "published"
    if put is None:
        fetched = store.get_exact(
            key=PROVIDER_SELECTION_S3_KEY,
            version_id=remote.version_id,
        )
        if (
            not isinstance(fetched, VersionedSelectionRead)
            or fetched.data != data
            or fetched.object != remote
        ):
            raise SelectionPublicationError(
                "lost PUT response cannot recover exact selection bytes",
                publication_state="uncertain",
            )
        _require_singleton_selection_history(
            store=store,
            version_id=remote.version_id,
        )
        state = "recovered"
    return remote, state


def _require_singleton_selection_history(
    *,
    store: VersionedProviderSelectionStore,
    version_id: str,
) -> None:
    history = store.list_versions(key=PROVIDER_SELECTION_S3_KEY)
    if (
        not isinstance(history, VersionedSelectionHistory)
        or history.key != PROVIDER_SELECTION_S3_KEY
        or history.versions != (version_id,)
        or history.delete_markers
    ):
        raise ValueError(
            "fixed-key selection history must contain exactly the selected "
            "version and no delete marker"
        )


def _publish_version_binding(
    *,
    authority_root: Path | str,
    selection: AwsProviderSelectionReceipt,
    remote: VersionedSelectionObject,
) -> Path:
    value = {
        "schema_version": 1,
        "selection_sha256": selection.sha256,
        "s3_key": PROVIDER_SELECTION_S3_KEY,
        "version_id": remote.version_id,
    }
    data = _canonical_json(value)
    destination = Path(authority_root).joinpath(
        *PROVIDER_SELECTION_VERSION_LOCAL_PATH.split("/")
    )
    try:
        _write_selection_noreplace(destination, data)
    except FileExistsError as error:
        existing = _regular_bytes(
            destination,
            label="provider selection version authority",
            private=True,
        )
        if existing != data:
            raise ValueError(
                "fixed provider-selection version authority conflicts"
            ) from error
    return destination


def publish_provider_selection(
    *,
    authority_root: Path | str,
    repo_root: Path | str,
    runtime_lock_path: Path | str,
    runtime_evidence_path: Path | str,
    selection_data: bytes,
    store: VersionedProviderSelectionStore,
    identity_verifier: Callable[
        [Mapping[str, object], str, str], bool
    ],
    approval_verifier: QualificationApprovalVerifier,
    trusted_public_key_sha256: str,
) -> PublishedProviderSelection:
    """Publish the sole local and versioned-S3 cohort selection authority."""

    selection = _load_verified_selection_from_paths(
        selection_data,
        repo_root=repo_root,
        runtime_lock_path=runtime_lock_path,
        runtime_evidence_path=runtime_evidence_path,
        identity_verifier=identity_verifier,
        approval_verifier=approval_verifier,
        trusted_public_key_sha256=trusted_public_key_sha256,
    )
    _preflight_local_selection(
        authority_root=authority_root,
        data=selection_data,
    )
    durable_binding_exists = _matching_durable_selection_binding(
        authority_root=authority_root,
        data=selection_data,
    )
    _pending_path, pending_existed = _publish_pending_selection(
        authority_root=authority_root,
        data=selection_data,
    )
    mutation_started = _remote_mutation_marker_exists(
        authority_root=authority_root,
        data=selection_data,
    )
    try:
        remote, publication_state = _publish_remote_selection(
            authority_root=authority_root,
            data=selection_data,
            store=store,
            allow_recovery=mutation_started
            and (pending_existed or durable_binding_exists),
        )
        local_path = _publish_local_selection(
            authority_root=authority_root,
            data=selection_data,
        )
        version_path = _publish_version_binding(
            authority_root=authority_root,
            selection=selection,
            remote=remote,
        )
        _archive_pending_selection(
            authority_root=authority_root,
            data=selection_data,
        )
    except SelectionPublicationError:
        raise
    except Exception as error:
        raise SelectionPublicationError(
            f"provider selection publication is uncertain: {error}",
            publication_state="uncertain",
        ) from error
    return PublishedProviderSelection(
        selection=selection,
        local_path=local_path,
        remote=remote,
        version_path=version_path,
        publication_state=publication_state,
    )


def load_local_provider_selection_authority(
    *,
    authority_root: Path | str,
    repo_root: Path | str,
    runtime_lock_path: Path | str,
    runtime_evidence_path: Path | str,
    identity_verifier: Callable[
        [Mapping[str, object], str, str], bool
    ],
    approval_verifier: QualificationApprovalVerifier,
    trusted_public_key_sha256: str,
) -> AwsProviderSelectionReceipt:
    """Load and re-verify the sole fixed local selection authority."""

    selection_data = _regular_bytes(
        Path(authority_root).joinpath(*PROVIDER_SELECTION_LOCAL_PATH.split("/")),
        label="provider selection authority",
        private=True,
    )
    return _load_verified_selection_from_paths(
        selection_data,
        repo_root=repo_root,
        runtime_lock_path=runtime_lock_path,
        runtime_evidence_path=runtime_evidence_path,
        identity_verifier=identity_verifier,
        approval_verifier=approval_verifier,
        trusted_public_key_sha256=trusted_public_key_sha256,
    )


def load_versioned_provider_selection_authority(
    *,
    authority_root: Path | str,
    repo_root: Path | str,
    runtime_lock_path: Path | str,
    runtime_evidence_path: Path | str,
    store: VersionedProviderSelectionStore,
    identity_verifier: Callable[
        [Mapping[str, object], str, str], bool
    ],
    approval_verifier: QualificationApprovalVerifier,
    trusted_public_key_sha256: str,
) -> AwsProviderSelectionReceipt:
    """GET and verify the persisted exact S3 selection version."""

    root = Path(authority_root)
    local_data = _regular_bytes(
        root.joinpath(*PROVIDER_SELECTION_LOCAL_PATH.split("/")),
        label="provider selection authority",
        private=True,
    )
    version_data = _regular_bytes(
        root.joinpath(*PROVIDER_SELECTION_VERSION_LOCAL_PATH.split("/")),
        label="provider selection version authority",
        private=True,
    )
    value = _json_object(version_data, label="provider selection version")
    if version_data != _canonical_json(value):
        raise ValueError("provider selection version must be canonical")
    version = _object(
        value,
        fields=frozenset(
            {"schema_version", "selection_sha256", "s3_key", "version_id"}
        ),
        label="provider selection version",
    )
    if type(version["schema_version"]) is not int or version["schema_version"] != 1:
        raise ValueError("provider selection version schema is invalid")
    selection_sha256 = _sha256(
        version["selection_sha256"],
        label="provider selection version selection SHA-256",
    )
    if (
        version["s3_key"] != PROVIDER_SELECTION_S3_KEY
        or not isinstance(version["version_id"], str)
        or version["version_id"] in {"", "null"}
        or selection_sha256 != hashlib.sha256(local_data).hexdigest()
    ):
        raise ValueError("provider selection version identity is invalid")
    _require_singleton_selection_history(
        store=store,
        version_id=version["version_id"],
    )
    fetched = store.get_exact(
        key=PROVIDER_SELECTION_S3_KEY,
        version_id=version["version_id"],
    )
    if (
        not isinstance(fetched, VersionedSelectionRead)
        or fetched.data != local_data
        or fetched.object.key != PROVIDER_SELECTION_S3_KEY
        or fetched.object.sha256 != selection_sha256
        or fetched.object.bytes != len(local_data)
        or fetched.object.version_id != version["version_id"]
    ):
        raise ValueError("provider selection exact-version GET replay failed")
    _require_singleton_selection_history(
        store=store,
        version_id=version["version_id"],
    )
    return _load_verified_selection_from_paths(
        fetched.data,
        repo_root=repo_root,
        runtime_lock_path=runtime_lock_path,
        runtime_evidence_path=runtime_evidence_path,
        identity_verifier=identity_verifier,
        approval_verifier=approval_verifier,
        trusted_public_key_sha256=trusted_public_key_sha256,
    )


def admit_provider_selection(
    *,
    authority_root: Path | str,
    repo_root: Path | str,
    runtime_lock_path: Path | str,
    runtime_evidence_path: Path | str,
    store: VersionedProviderSelectionStore,
    account_id: str,
    instance_id: str,
    boot_id: str,
    seed: int,
    arm: str,
    expected_selection_version_id: str,
    identity_verifier: Callable[
        [Mapping[str, object], str, str], bool
    ],
    approval_verifier: QualificationApprovalVerifier,
    trusted_public_key_sha256: str,
) -> AuthenticatedSelectionBinding:
    """Admit one launch/resume cell from exact authenticated authority."""

    receipt = load_versioned_provider_selection_authority(
        authority_root=authority_root,
        repo_root=repo_root,
        runtime_lock_path=runtime_lock_path,
        runtime_evidence_path=runtime_evidence_path,
        store=store,
        identity_verifier=identity_verifier,
        approval_verifier=approval_verifier,
        trusted_public_key_sha256=trusted_public_key_sha256,
    )
    version_data = _regular_bytes(
        Path(authority_root).joinpath(
            *PROVIDER_SELECTION_VERSION_LOCAL_PATH.split("/")
        ),
        label="provider selection version authority",
        private=True,
    )
    version = _json_object(
        version_data,
        label="provider selection version authority",
    )
    persisted_version = version.get("version_id")
    if (
        not isinstance(expected_selection_version_id, str)
        or expected_selection_version_id in {"", "null"}
        or persisted_version != expected_selection_version_id
    ):
        raise ValueError("selection version authority differs from admission")
    if (
        account_id != receipt.account_id
        or instance_id != receipt.qualification_instance_id
        or boot_id != receipt.qualification_boot_id
        or receipt.identity_verified is not True
        or receipt.approval_verified is not True
    ):
        raise ValueError("selection authority identity differs from admission")
    if type(seed) is not int or seed not in receipt.seeds:
        raise ValueError("selection authority seed differs from admission")
    if arm not in receipt.arms:
        raise ValueError("selection authority arm differs from admission")
    return AuthenticatedSelectionBinding(
        cohort_id=receipt.cohort_id,
        amendment_sha256=receipt.amendment.sha256,
        selection_sha256=receipt.sha256,
        selection_version_id=expected_selection_version_id,
        profile_id=receipt.profile.profile_id,
        provider=receipt.profile.provider,
        profile_sha256=receipt.profile.sha256,
        runtime_lock_sha256=receipt.runtime_lock_sha256,
        qualification_evidence_sha256=receipt.runtime_evidence_sha256,
        environment_receipt_sha256=receipt.environment_receipt_sha256,
        canary_receipt_sha256=receipt.canary_receipt_sha256,
        approval_receipt_sha256=receipt.approval_receipt_sha256,
        approval_public_key_sha256=receipt.approval_public_key_sha256,
        account_id=receipt.account_id,
        instance_id=receipt.qualification_instance_id,
        boot_id=receipt.qualification_boot_id,
        region=receipt.region,
        availability_zone=receipt.availability_zone,
        purchase_model=receipt.purchase_model,
        seed=seed,
        arm=arm,
    )


def validate_resume_hardware_binding(
    *,
    authority_root: Path | str,
    repo_root: Path | str,
    runtime_lock_path: Path | str,
    runtime_evidence_path: Path | str,
    store: VersionedProviderSelectionStore,
    account_id: str,
    instance_id: str,
    boot_id: str,
    expected_selection_version_id: str,
    identity_verifier: Callable[
        [Mapping[str, object], str, str], bool
    ],
    approval_verifier: QualificationApprovalVerifier,
    trusted_public_key_sha256: str,
    amendment_sha256: str,
    provider_selection_sha256: str,
    profile_sha256: str,
    runtime_lock_sha256: str,
    runtime_evidence_sha256: str,
    seed: int,
    arm: str,
) -> AuthenticatedSelectionBinding:
    """Require exact remote version and qualification for resume binding."""

    binding = admit_provider_selection(
        authority_root=authority_root,
        repo_root=repo_root,
        runtime_lock_path=runtime_lock_path,
        runtime_evidence_path=runtime_evidence_path,
        store=store,
        account_id=account_id,
        instance_id=instance_id,
        boot_id=boot_id,
        expected_selection_version_id=expected_selection_version_id,
        seed=seed,
        arm=arm,
        identity_verifier=identity_verifier,
        approval_verifier=approval_verifier,
        trusted_public_key_sha256=trusted_public_key_sha256,
    )
    if amendment_sha256 != binding.amendment_sha256:
        raise ValueError("hardware amendment binding differs from selection")
    if provider_selection_sha256 != binding.selection_sha256:
        raise ValueError("provider-selection receipt binding differs")
    if profile_sha256 != binding.profile_sha256:
        raise ValueError("hardware profile differs from cohort selection")
    if runtime_lock_sha256 != binding.runtime_lock_sha256:
        raise ValueError("runtime-lock binding differs from cohort selection")
    if runtime_evidence_sha256 != binding.qualification_evidence_sha256:
        raise ValueError("runtime-evidence binding differs from cohort selection")
    return binding


validate_provider_selection_binding = validate_resume_hardware_binding
load_provider_selection_receipt = load_local_provider_selection_authority
load_aws_provider_selection_receipt = load_local_provider_selection_authority


def _hardware_authority_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m msctl.aws_hardware",
        description="Dry-run-first AWS GPU provider-selection authority.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser(
        "publish",
        help="verify or publish the sole cohort provider selection",
    )
    publish.add_argument("--repo-root", required=True)
    publish.add_argument("--authority-root", required=True)
    publish.add_argument("--runtime-lock", required=True)
    publish.add_argument("--qualification-evidence", required=True)
    publish.add_argument("--selection", required=True)
    publish.add_argument("--bucket", required=True)
    publish.add_argument("--region", required=True)
    publish.add_argument("--approval-public-key", required=True)
    publish.add_argument("--approval-public-key-sha256", required=True)
    publish.add_argument("--apply", action="store_true")
    return parser


def run_hardware_authority_cli(
    argv: Sequence[str] | None = None,
    *,
    store: VersionedProviderSelectionStore | None = None,
    identity_verifier: Callable[
        [Mapping[str, object], str, str], bool
    ] = verify_aws_instance_identity_pkcs7,
    approval_verifier: QualificationApprovalVerifier | None = None,
    trusted_public_key_sha256: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[bool, dict[str, object]]:
    """Execute the standalone dry-run/apply selection authority CLI."""

    arguments = _hardware_authority_parser().parse_args(argv)
    if arguments.command != "publish":
        raise ValueError("unsupported hardware authority command")
    argument_key = _sha256(
        arguments.approval_public_key_sha256,
        label="CLI trusted approval public key",
    )
    if (
        trusted_public_key_sha256 is not None
        and trusted_public_key_sha256 != argument_key
    ):
        raise ValueError("CLI approval public-key commitment differs")
    trusted_key = argument_key
    environment = dict(os.environ if environ is None else environ)
    command_environment = {
        name: environment[name]
        for name in ("HOME", "LANG", "LC_ALL", "PATH")
        if name in environment
    }
    command_environment["AWS_REGION"] = arguments.region
    verifier = approval_verifier or OpenSslQualificationApprovalVerifier(
        public_key_path=arguments.approval_public_key,
        environment=command_environment,
    )
    selection_data = _regular_bytes(
        Path(arguments.selection),
        label="provider selection candidate",
        private=True,
    )
    verified = _load_verified_selection_from_paths(
        selection_data,
        repo_root=arguments.repo_root,
        runtime_lock_path=arguments.runtime_lock,
        runtime_evidence_path=arguments.qualification_evidence,
        identity_verifier=identity_verifier,
        approval_verifier=verifier,
        trusted_public_key_sha256=trusted_key,
    )
    result: dict[str, object] = {
        "operation": "publish-provider-selection",
        "apply": bool(arguments.apply),
        "publication_state": "planned",
        "cohort_id": verified.cohort_id,
        "profile_id": verified.profile.profile_id,
        "selection_sha256": verified.sha256,
        "s3_bucket": arguments.bucket,
        "s3_key": PROVIDER_SELECTION_S3_KEY,
        "local_path": str(
            Path(arguments.authority_root).joinpath(
                *PROVIDER_SELECTION_LOCAL_PATH.split("/")
            )
        ),
        "qualification_evidence_sha256": verified.runtime_evidence_sha256,
        "approval_public_key_sha256": verified.approval_public_key_sha256,
    }
    if not arguments.apply:
        return True, result
    selected_store = store or AwsCliVersionedSelectionStore(
        bucket=arguments.bucket,
        region=arguments.region,
        environment=command_environment,
        staging_root=Path(arguments.authority_root) / ".selection-s3-staging",
    )
    published = publish_provider_selection(
        authority_root=arguments.authority_root,
        repo_root=arguments.repo_root,
        runtime_lock_path=arguments.runtime_lock,
        runtime_evidence_path=arguments.qualification_evidence,
        selection_data=selection_data,
        store=selected_store,
        identity_verifier=identity_verifier,
        approval_verifier=verifier,
        trusted_public_key_sha256=trusted_key,
    )
    return False, {
        **result,
        "publication_state": published.publication_state,
        "selection_version_id": published.remote.version_id,
        "version_path": str(published.version_path),
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    store: VersionedProviderSelectionStore | None = None,
    identity_verifier: Callable[
        [Mapping[str, object], str, str], bool
    ] = verify_aws_instance_identity_pkcs7,
    approval_verifier: QualificationApprovalVerifier | None = None,
    trusted_public_key_sha256: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    apply_requested = "--apply" in effective_argv
    try:
        dry_run, result = run_hardware_authority_cli(
            effective_argv,
            store=store,
            identity_verifier=identity_verifier,
            approval_verifier=approval_verifier,
            trusted_public_key_sha256=trusted_public_key_sha256,
            environ=environ,
        )
        report = {
            "schema_version": 1,
            "ok": True,
            "dry_run": dry_run,
            "result": result,
        }
        exit_code = 0
    except (OSError, TypeError, ValueError) as error:
        publication_state = getattr(
            error,
            "publication_state",
            "uncertain" if apply_requested else "not-started",
        )
        report = {
            "schema_version": 1,
            "ok": False,
            "dry_run": not apply_requested,
            "publication_state": publication_state,
            "error": str(error),
        }
        exit_code = 2
    sys.stdout.buffer.write(_canonical_json(report))
    return exit_code


__all__ = [
    "AWS_HARDWARE_AMENDMENT_PATH",
    "AWS_HARDWARE_AMENDMENT_SHA256",
    "AuthenticatedSelectionBinding",
    "ArtifactBinding",
    "AwsCliVersionedSelectionStore",
    "AwsHardwareAmendment",
    "AwsProviderSelectionReceipt",
    "AwsRuntimeEvidence",
    "AwsRuntimeLock",
    "COHORT_ASSIGNMENT_PATH",
    "HardwareProfileBinding",
    "P5_PROFILE_PATH",
    "P6_PROFILE_PATH",
    "OpenSslQualificationApprovalVerifier",
    "PREREGISTRATION_PATH",
    "PROVIDER_SELECTION_LOCAL_PATH",
    "PROVIDER_SELECTION_MUTATION_ARCHIVE_LOCAL_PATH",
    "PROVIDER_SELECTION_MUTATION_LOCAL_PATH",
    "PROVIDER_SELECTION_PENDING_ARCHIVE_LOCAL_PATH",
    "PROVIDER_SELECTION_PENDING_LOCAL_PATH",
    "PROVIDER_SELECTION_S3_KEY",
    "PROVIDER_SELECTION_VERSION_LOCAL_PATH",
    "PublishedProviderSelection",
    "QualificationApprovalVerifier",
    "SelectionPublicationError",
    "VersionedProviderSelectionStore",
    "VersionedSelectionHistory",
    "VersionedSelectionObject",
    "VersionedSelectionRead",
    "canonical_provider_selection_receipt_bytes",
    "admit_provider_selection",
    "load_aws_hardware_amendment",
    "load_aws_provider_selection_receipt",
    "load_local_provider_selection_authority",
    "load_provider_selection_receipt",
    "load_versioned_provider_selection_authority",
    "parse_aws_provider_selection_receipt_bytes",
    "parse_aws_hardware_amendment_bytes",
    "parse_aws_runtime_lock_bytes",
    "parse_authenticated_qualification_evidence_bytes",
    "parse_provider_selection_receipt_bytes",
    "parse_verified_provider_selection_bytes",
    "publish_provider_selection",
    "run_hardware_authority_cli",
    "validate_aws_hardware_amendment_files",
    "validate_provider_selection_binding",
    "validate_resume_hardware_binding",
    "verify_aws_instance_identity_pkcs7",
]


if __name__ == "__main__":
    raise SystemExit(main())
