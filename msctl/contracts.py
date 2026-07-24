"""Strict release, run-manifest, and checkpoint contracts."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .aws_contracts import (
    ARMS as AWS_ARMS,
    AWS_ENVIRONMENT_RECEIPT_V2_FIELDS,
    COHORT_ASSIGNMENT_PATH as AWS_COHORT_ASSIGNMENT_PATH,
    COHORT_ID as AWS_COHORT_ID,
    DATASET_POINTER_PATH as AWS_DATASET_POINTER_PATH,
    DATASET_RECEIPT_PATH as AWS_DATASET_RECEIPT_PATH,
    EXPECTED_CONFIG_PATHS as AWS_EXPECTED_CONFIG_PATHS,
    PACKAGE_FORMAT_VERSION as AWS_PACKAGE_FORMAT_VERSION,
    PREREGISTRATION_PATH as AWS_PREREGISTRATION_PATH,
    PROFILE_PATH as AWS_PROFILE_PATH,
    SEEDS as AWS_SEEDS,
)
from .cohort import load_cohort_assignment as load_cohort_assignment
from .errors import MsctlError
from .fsutil import hash_fd, open_directory, open_regular_at, read_fd
from .aws_lifecycle import LIFECYCLE_BINDING_FIELDS, ProviderLifecycleBinding
from .jsonutil import (
    COMMIT_RE,
    RUN_ID_RE,
    canonical_json,
    canonical_sha256,
    load_json,
    portable_relative,
    require_exact_keys,
    require_nonnegative_number,
    require_object,
    require_schema_version,
    require_sha256,
    regular_file,
    resolve_inside,
    sha256_file,
)
from .profile import AWS_P5_PROFILE, SUPPORTED_PROFILE


@dataclass(frozen=True)
class Release:
    release_id: str
    provider: str
    receipt_sha256: str
    archive_sha256: str
    archive_bytes: int
    archive_path: Path
    source_commit: str
    source_tree: str | None
    package_format_version: int | None
    members_sha256: str
    members: dict[str, dict[str, object]]
    archive_files: dict[str, str]
    metadata: dict[str, object]
    value: dict[str, object]


@dataclass(frozen=True)
class Run:
    run_id: str
    arm: str
    seed: int
    config: str
    config_sha256: str
    estimated_gpu_hours: float


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    provider: str
    seed: int
    release_sha256: str
    dataset_sha256: str
    cohort_assignment_sha256: str | None
    study_lock_sha256: str | None
    source_commit: str | None
    runs: tuple[Run, ...]
    sha256: str
    value: dict[str, object]

    @property
    def gpu_hours(self) -> float:
        return sum(run.estimated_gpu_hours for run in self.runs)


@dataclass(frozen=True)
class RunManifestV3:
    schema_version: int
    provider: str
    profile_id: str
    cohort_id: str
    seed: int
    release_sha256: str
    release_receipt_sha256: str
    profile_sha256: str
    hardware_amendment_sha256: str | None
    provider_selection_sha256: str | None
    provider_selection_version_id: str | None
    runtime_lock_sha256: str | None
    runtime_sbom_sha256: str | None
    qualification_evidence_sha256: str | None
    qualification_environment_receipt_sha256: str | None
    qualification_canary_receipt_sha256: str | None
    qualification_approval_receipt_sha256: str | None
    qualification_approval_public_key_sha256: str | None
    objective_controls_contract_sha256: str | None
    account_id: str | None
    instance_id: str | None
    boot_id: str | None
    region: str | None
    availability_zone: str | None
    purchase_model: str | None
    arms: tuple[str, str]
    dataset_pointer_sha256: str
    dataset_receipt_sha256: str
    dataset_build_id: str
    ordered_stream_sha256: str
    cohort_assignment_sha256: str
    preregistration_sha256: str
    sealed_evaluation_release_sha256: str
    source_commit: str
    source_tree: str
    runs: tuple[Run, ...]
    sha256: str
    value: dict[str, object]

    @property
    def gpu_hours(self) -> float:
        return sum(run.estimated_gpu_hours for run in self.runs)


@dataclass(frozen=True)
class Checkpoint:
    run_id: str
    arm: str
    seed: int
    path: Path
    sha256: str
    config_sha256: str
    dataset_sha256: str
    source_commit: str | None
    step: int
    world_size: int


@dataclass(frozen=True)
class CheckpointReceipt:
    schema_version: int
    sha256: str
    checkpoints: tuple[Checkpoint, ...]
    value: dict[str, object]


@dataclass(frozen=True)
class AwsCheckpointDataV3:
    receipt_sha256: str
    build_id: str
    ordered_stream_sha256: str
    global_cursor: int
    sidecar_name: str


@dataclass(frozen=True)
class AwsCheckpointObjectV3:
    uri: str
    sha256: str
    bytes: int
    version_id: str


@dataclass(frozen=True)
class AwsCheckpointV3:
    run_id: str
    arm: str
    seed: int
    checkpoint_version: int
    step: int
    world_size: int
    config_sha256: str
    config_fingerprint: str
    data: AwsCheckpointDataV3
    object: AwsCheckpointObjectV3


@dataclass(frozen=True)
class AwsPairedCheckpointReceiptV3:
    schema_version: int
    receipt_type: str
    provider: str
    profile_id: str
    cohort_id: str
    seed: int
    reason: str
    request_id: str
    instance_id: str
    boot_id: str
    profile_sha256: str
    hardware_amendment_sha256: str | None
    provider_selection_sha256: str | None
    provider_selection_version_id: str | None
    runtime_lock_sha256: str | None
    runtime_sbom_sha256: str | None
    qualification_evidence_sha256: str | None
    environment_receipt_sha256: str
    qualification_environment_receipt_sha256: str | None
    qualification_canary_receipt_sha256: str | None
    qualification_approval_receipt_sha256: str | None
    qualification_approval_public_key_sha256: str | None
    objective_controls_contract_sha256: str | None
    account_id: str | None
    region: str | None
    availability_zone: str | None
    purchase_model: str | None
    arms: tuple[str, str]
    release_sha256: str
    release_receipt_sha256: str
    run_manifest_sha256: str
    dataset_receipt_sha256: str
    dataset_build_id: str
    ordered_stream_sha256: str
    source_commit: str
    source_tree: str
    requested_at: str
    deadline_at: str
    staged_at: str
    checkpoints: tuple[AwsCheckpointV3, AwsCheckpointV3]
    resumable: bool
    uri: str
    sha256: str
    version_id: str
    bytes: int
    value: dict[str, object]


_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_RUNTIME_RECEIPT_FIELDS = list(AWS_ENVIRONMENT_RECEIPT_V2_FIELDS)
_CHECKPOINT_REQUEST_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_CANONICAL_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)


def same_typed_value(actual: object, expected: object) -> bool:
    """Compare nested JSON-like values without Python numeric aliases."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (
            actual.keys() == expected.keys()
            and all(
                same_typed_value(actual[key], expected[key])
                for key in expected
            )
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            same_typed_value(left, right)
            for left, right in zip(actual, expected)
        )
    return actual == expected


def validate_aws_dataset_pointer_contract(
    value: object,
) -> dict[str, object]:
    """Return the exact Task 2A pointer or reject any semantic drift."""

    pointer = require_object(value, label="AWS dataset pointer")
    expected = {
        "dataset_id": "memorysplit-v2-20x-reasoning-max-cohort",
        "durable_uri_env": "MS_S3_ROOT",
        "full_corpus_in_release": False,
        "materialization": "s3",
        "provider": AWS_P5_PROFILE,
        "relative_path": "dataset",
        "required_receipt": AWS_DATASET_RECEIPT_PATH,
        "required_sidecars": [
            "dense_target_weights",
            "split90_target_weights",
        ],
        "schema_version": 1,
        "scratch_root": "/mnt/memorysplit",
        "source_lock_manifest": "configs/reasoning-dataset-v2.json",
    }
    if not same_typed_value(pointer, expected):
        raise MsctlError(
            "RELEASE_INTERNAL_INVALID",
            "AWS dataset pointer does not match the exact Task 2A contract",
        )
    return pointer


def validate_runtime_attested_contract(
    value: object,
    *,
    profile_sha256: str,
) -> dict[str, object]:
    contract = require_object(value, label="AWS runtime environment contract")
    require_exact_keys(
        contract,
        {
            "mode",
            "profile_sha256",
            "container_image_digest_env",
            "container_image_digest_pattern",
            "runtime_environment_receipt",
        },
        label="AWS runtime environment contract",
    )
    receipt = require_object(
        contract["runtime_environment_receipt"],
        label="AWS runtime environment receipt contract",
    )
    require_exact_keys(
        receipt,
        {"required_at_launch", "authentication", "required_fields"},
        label="AWS runtime environment receipt contract",
    )
    if (
        contract["mode"] != "runtime_attested"
        or contract["profile_sha256"] != profile_sha256
        or contract["container_image_digest_env"] != "MS_CONTAINER_DIGEST"
        or contract["container_image_digest_pattern"]
        != "^sha256:[0-9a-f]{64}$"
        or receipt["required_at_launch"] is not True
        or receipt["authentication"]
        != "aws_instance_identity_document_pkcs7"
        or receipt["required_fields"] != _RUNTIME_RECEIPT_FIELDS
    ):
        raise MsctlError(
            "RELEASE_INTERNAL_INVALID",
            "AWS release runtime_attested contract is invalid",
        )
    return contract


def _release_error(
    code: str,
    message: str,
    *,
    path: Path | None = None,
) -> MsctlError:
    details = {"path": str(path)} if path is not None else {}
    return MsctlError(code, message, details=details)


def _read_regular_relative(
    root: Path,
    relative: str,
    *,
    label: str,
) -> bytes:
    root_fd = open_directory(root, label=f"{label} directory")
    try:
        descriptor, parent_fd, _ = open_regular_at(
            root_fd,
            relative,
            label=label,
        )
        try:
            before = os.fstat(descriptor)
            data = read_fd(descriptor)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
            os.close(parent_fd)
    finally:
        os.close(root_fd)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(data) != after.st_size:
        raise _release_error(
            "RELEASE_ARCHIVE_INVALID",
            f"{label} changed while being read",
        )
    return data


def _read_archive(
    release_path: Path,
    relative: str,
) -> tuple[Path, bytes]:
    parent_fd = open_directory(
        release_path.parent,
        label="release directory",
    )
    try:
        descriptor, archive_parent_fd, _ = open_regular_at(
            parent_fd,
            relative,
            label="release archive",
        )
        try:
            before = os.fstat(descriptor)
            data = read_fd(descriptor)
            after = os.fstat(descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) or len(data) != after.st_size:
                raise _release_error(
                    "RELEASE_ARCHIVE_INVALID",
                    "release archive changed while being read",
                )
        finally:
            os.close(descriptor)
            os.close(archive_parent_fd)
    except MsctlError as error:
        if error.code == "RELEASE_ARCHIVE_INVALID":
            raise
        raise _release_error(
            "RELEASE_ARCHIVE_INVALID",
            "release archive must be a non-symlink regular file",
            path=release_path.parent / relative,
        ) from error
    finally:
        os.close(parent_fd)
    return release_path.parent / relative, data


def _parse_internal_json(data: bytes, *, label: str) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise _release_error(
                    "RELEASE_INTERNAL_INVALID",
                    f"{label} contains a duplicate field",
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                _release_error(
                    "RELEASE_INTERNAL_INVALID",
                    f"{label} contains non-finite {constant}",
                )
            ),
        )
    except MsctlError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _release_error(
            "RELEASE_INTERNAL_INVALID",
            f"{label} must be valid UTF-8 JSON",
        ) from error
    return require_object(value, label=label)


def _validate_zip_member(info: zipfile.ZipInfo) -> None:
    name = info.filename
    if (
        not name
        or "\\" in name
        or name.startswith("/")
        or any(part in {"", ".", ".."} for part in name.rstrip("/").split("/"))
    ):
        raise _release_error(
            "RELEASE_INTERNAL_INVALID",
            "release archive contains an unsafe member path",
        )
    mode = info.external_attr >> 16
    if info.is_dir():
        if not mode or not stat.S_ISDIR(mode):
            raise _release_error(
                "RELEASE_INTERNAL_INVALID",
                "release archive directory metadata is invalid",
            )
    elif not mode or not stat.S_ISREG(mode):
        raise _release_error(
            "RELEASE_INTERNAL_INVALID",
            "release archive contains a non-regular member",
        )


def _verify_release_internals(
    archive_data: bytes,
    *,
    release_value: dict[str, object],
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, str],
]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_data))
    except (OSError, zipfile.BadZipFile) as error:
        raise _release_error(
            "RELEASE_ARCHIVE_INVALID",
            "release archive is not a valid ZIP",
        ) from error
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise _release_error(
                "RELEASE_INTERNAL_INVALID",
                "release archive contains duplicate member names",
            )
        for info in infos:
            _validate_zip_member(info)
        file_names = {
            info.filename for info in infos if not info.is_dir()
        }
        if not {"RELEASE-METADATA.json", "SHA256SUMS"} <= file_names:
            raise _release_error(
                "RELEASE_INTERNAL_INVALID",
                "release archive is missing internal verification metadata",
            )
        payload = {name: archive.read(name) for name in file_names}

    sums_bytes = payload["SHA256SUMS"]
    if hashlib.sha256(sums_bytes).hexdigest() != release_value["members_sha256"]:
        raise _release_error(
            "RELEASE_INTERNAL_INVALID",
            "internal checksum manifest does not match RELEASE.json",
        )
    try:
        sums_text = sums_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise _release_error(
            "RELEASE_INTERNAL_INVALID",
            "SHA256SUMS must be ASCII",
        ) from error
    if not sums_text.endswith("\n"):
        raise _release_error(
            "RELEASE_INTERNAL_INVALID",
            "SHA256SUMS must end with one newline",
        )
    checksum_rows: dict[str, str] = {}
    ordered_paths: list[str] = []
    for line in sums_text.splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise _release_error(
                "RELEASE_INTERNAL_INVALID",
                "SHA256SUMS contains a malformed row",
            )
        digest = line[:64]
        relative = line[66:]
        require_sha256(digest, label="SHA256SUMS digest")
        portable_relative(relative, label="SHA256SUMS path")
        if relative in checksum_rows or relative == "SHA256SUMS":
            raise _release_error(
                "RELEASE_INTERNAL_INVALID",
                "SHA256SUMS contains a duplicate or self-reference",
            )
        checksum_rows[relative] = digest
        ordered_paths.append(relative)
    expected_paths = file_names - {"SHA256SUMS"}
    if ordered_paths != sorted(ordered_paths) or set(ordered_paths) != expected_paths:
        raise _release_error(
            "RELEASE_INTERNAL_INVALID",
            "SHA256SUMS is not sorted and hash-complete",
        )
    for relative, digest in checksum_rows.items():
        if hashlib.sha256(payload[relative]).hexdigest() != digest:
            raise _release_error(
                "RELEASE_INTERNAL_INVALID",
                "release member checksum mismatch",
            )

    metadata = _parse_internal_json(
        payload["RELEASE-METADATA.json"],
        label="RELEASE-METADATA.json",
    )
    aws_package_version = (
        release_value.get("package_format_version")
        if release_value["provider"]
        in {"aws-p5.48xlarge", "aws-p6-b300.48xlarge"}
        and "package_format_version" in metadata
        else None
    )
    aws_package = aws_package_version is not None
    selected_lifecycle = (
        aws_package and "provider_selection_sha256" in release_value
    )
    if aws_package:
        metadata_fields = {
            "schema_version",
            "package_format_version",
            "provider",
            "source",
            "seed_assignment",
            "cohort_assignment",
            "profile",
            "environment",
            "dataset_pointer",
            "config_sha256",
            "members",
        }
        if selected_lifecycle:
            metadata_fields |= {
                *LIFECYCLE_BINDING_FIELDS,
                "profile_id",
            }
    elif release_value["provider"] == AWS_P5_PROFILE:
        metadata_fields = {
            "schema_version",
            "provider",
            "source",
            "profile_sha256",
            "environment_hashes",
            "members",
            "seed_assignment",
        }
    else:
        metadata_fields = {
            "schema_version",
            "provider",
            "source",
            "profile_sha256",
            "environment_hashes",
            "members",
        }
        for optional_field in ("preregistration_sha256", "seed_assignment"):
            if optional_field in metadata:
                metadata_fields.add(optional_field)
    require_exact_keys(
        metadata,
        metadata_fields,
        label="RELEASE-METADATA.json",
    )
    try:
        require_schema_version(
            metadata["schema_version"],
            label="RELEASE-METADATA.json.schema_version",
        )
    except MsctlError as error:
        raise _release_error(
            "RELEASE_INTERNAL_INVALID",
            "RELEASE-METADATA.json schema is unsupported",
        ) from error
    metadata_source = require_object(
        metadata["source"],
        label="RELEASE-METADATA.json.source",
    )
    metadata_source_fields = {"commit", "dirty"}
    if aws_package_version == AWS_PACKAGE_FORMAT_VERSION:
        metadata_source_fields.add("tree")
    require_exact_keys(
        metadata_source,
        metadata_source_fields,
        label="RELEASE-METADATA.json.source",
    )
    if (
        not isinstance(metadata_source["commit"], str)
        or COMMIT_RE.fullmatch(metadata_source["commit"]) is None
        or metadata_source["dirty"] is not False
        or (
            aws_package_version == AWS_PACKAGE_FORMAT_VERSION
            and (
                not isinstance(metadata_source["tree"], str)
                or _GIT_SHA1_RE.fullmatch(metadata_source["tree"]) is None
            )
        )
    ):
        raise _release_error(
            "RELEASE_INTERNAL_INVALID",
            "internal release source identity is invalid",
        )
    if (
        metadata["provider"] != release_value["provider"]
        or metadata_source != release_value["source"]
    ):
        raise _release_error(
            "RELEASE_INTERNAL_INVALID",
            "internal metadata does not bind RELEASE.json",
        )
    if aws_package:
        profile_binding = require_object(
            metadata["profile"],
            label="RELEASE-METADATA.json.profile",
        )
        require_exact_keys(
            profile_binding,
            {"path", "sha256"},
            label="RELEASE-METADATA.json.profile",
        )
        profile_hash = require_sha256(
            profile_binding["sha256"],
            label="RELEASE-METADATA.json.profile.sha256",
        )
        validate_runtime_attested_contract(
            metadata["environment"],
            profile_sha256=profile_hash,
        )
    else:
        profile_hash = require_sha256(
            metadata["profile_sha256"],
            label="RELEASE-METADATA.json.profile_sha256",
        )
    raw_members = metadata["members"]
    if not isinstance(raw_members, list) or not raw_members:
        raise _release_error(
            "RELEASE_INTERNAL_INVALID",
            "internal metadata members must be non-empty",
        )
    members: dict[str, dict[str, object]] = {}
    ordered_member_paths: list[str] = []
    for index, raw in enumerate(raw_members):
        row = require_object(raw, label=f"release member[{index}]")
        member_fields = {"path", "bytes", "sha256", "git_blob"}
        if aws_package:
            member_fields.add("git_mode")
        require_exact_keys(row, member_fields, label=f"release member[{index}]")
        relative = portable_relative(
            row["path"],
            label=f"release member[{index}].path",
        )
        size = row["bytes"]
        digest = require_sha256(
            row["sha256"],
            label=f"release member[{index}].sha256",
        )
        if (
            relative in members
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(row["git_blob"], str)
            or _GIT_OBJECT_RE.fullmatch(row["git_blob"]) is None
            or (
                aws_package
                and row["git_mode"] not in {"100644", "100755"}
            )
        ):
            raise _release_error(
                "RELEASE_INTERNAL_INVALID",
                "internal release member row is invalid",
            )
        if (
            relative not in payload
            or len(payload[relative]) != size
            or hashlib.sha256(payload[relative]).hexdigest() != digest
        ):
            raise _release_error(
                "RELEASE_INTERNAL_INVALID",
                "internal release member row does not match ZIP bytes",
            )
        members[relative] = row
        ordered_member_paths.append(relative)
    expected_source_members = file_names - {
        "RELEASE-METADATA.json",
        "SHA256SUMS",
    }
    if (
        ordered_member_paths != sorted(ordered_member_paths)
        or set(ordered_member_paths) != expected_source_members
    ):
        raise _release_error(
            "RELEASE_INTERNAL_INVALID",
            "internal metadata is not sorted and member-complete",
        )
    profile_member_path = (
        "cluster/profiles/illumina-usfc-prd.json"
        if release_value["provider"] == SUPPORTED_PROFILE
        else (
            portable_relative(
                profile_binding["path"],
                label="selected AWS profile path",
            )
            if selected_lifecycle
            else AWS_PROFILE_PATH
            if aws_package_version == AWS_PACKAGE_FORMAT_VERSION
            else "cluster/profiles/aws-p5.48xlarge.json"
        )
    )
    if selected_lifecycle and profile_member_path not in {
        AWS_PROFILE_PATH,
        "cluster/profiles/aws-p6-b300.48xlarge-v3.json",
    }:
        raise _release_error(
            "RELEASE_INTERNAL_INVALID",
            "selected AWS profile path is unsupported",
        )
    profile_member = members.get(profile_member_path)
    if profile_member is None or profile_member["sha256"] != profile_hash:
        raise _release_error(
            "RELEASE_INTERNAL_INVALID",
            "internal profile hash is not bound to its member",
        )
    if aws_package:
        if (
            type(metadata["package_format_version"]) is not int
            or metadata["package_format_version"] not in {1, 2}
            or metadata["package_format_version"]
            != release_value["package_format_version"]
        ):
            raise _release_error(
                "RELEASE_INTERNAL_INVALID",
                "AWS package format identity is invalid",
            )
        if profile_binding["path"] != profile_member_path:
            raise _release_error(
                "RELEASE_INTERNAL_INVALID",
                "AWS release profile path is invalid",
            )
        expected_binding_paths = (
            {
                "cohort_assignment": AWS_COHORT_ASSIGNMENT_PATH,
                "dataset_pointer": AWS_DATASET_POINTER_PATH,
            }
            if aws_package_version == AWS_PACKAGE_FORMAT_VERSION
            else {
                "cohort_assignment": "configs/cohort-assignment-v2.json",
                "dataset_pointer": AWS_DATASET_POINTER_PATH,
            }
        )
        for label, expected_path in expected_binding_paths.items():
            binding = require_object(
                metadata[label],
                label=f"RELEASE-METADATA.json.{label}",
            )
            require_exact_keys(
                binding,
                {"path", "sha256"},
                label=f"RELEASE-METADATA.json.{label}",
            )
            path = portable_relative(
                binding["path"],
                label=f"RELEASE-METADATA.json.{label}.path",
            )
            digest = require_sha256(
                binding["sha256"],
                label=f"RELEASE-METADATA.json.{label}.sha256",
            )
            if (
                path != expected_path
                or path not in members
                or members[path]["sha256"] != digest
            ):
                raise _release_error(
                    "RELEASE_INTERNAL_INVALID",
                    f"AWS {label} is not bound to a release member",
                )
        config_hashes = require_object(
            metadata["config_sha256"],
            label="RELEASE-METADATA.json.config_sha256",
        )
        expected_config_paths = (
            set(AWS_EXPECTED_CONFIG_PATHS)
            if aws_package_version == AWS_PACKAGE_FORMAT_VERSION
            else {
                f"configs/360m-v2/{arm}-s{seed}.yaml"
                for seed in (1, 2, 3, 4)
                for arm in AWS_ARMS
            }
        )
        if set(config_hashes) != expected_config_paths:
            raise _release_error(
                "RELEASE_INTERNAL_INVALID",
                "AWS config hash namespace does not match the package format",
            )
        for path, raw_digest in config_hashes.items():
            relative = portable_relative(path, label="AWS config path")
            digest = require_sha256(raw_digest, label="AWS config SHA-256")
            if relative not in members or members[relative]["sha256"] != digest:
                raise _release_error(
                    "RELEASE_INTERNAL_INVALID",
                    "AWS config hash is not bound to a release member",
                )
        if aws_package_version == AWS_PACKAGE_FORMAT_VERSION:
            assignment_value = _parse_internal_json(
                payload[AWS_COHORT_ASSIGNMENT_PATH],
                label=AWS_COHORT_ASSIGNMENT_PATH,
            )
            require_exact_keys(
                assignment_value,
                {
                    "schema_version",
                    "cohort_id",
                    "model_parameters",
                    "optimizer_steps",
                    "provider_seeds",
                    "raw_target_tokens",
                    "targets_per_update",
                },
                label=AWS_COHORT_ASSIGNMENT_PATH,
            )
            provider_seeds = require_object(
                assignment_value["provider_seeds"],
                label=f"{AWS_COHORT_ASSIGNMENT_PATH}.provider_seeds",
            )
            if (
                type(assignment_value["schema_version"]) is not int
                or assignment_value["schema_version"] != 3
                or assignment_value["cohort_id"] != AWS_COHORT_ID
                or type(assignment_value["model_parameters"]) is not int
                or assignment_value["model_parameters"] != 356_033_536
                or type(assignment_value["optimizer_steps"]) is not int
                or assignment_value["optimizer_steps"] != 13_582
                or type(assignment_value["raw_target_tokens"]) is not int
                or assignment_value["raw_target_tokens"] != 7_120_879_616
                or type(assignment_value["targets_per_update"]) is not int
                or assignment_value["targets_per_update"] != 524_288
                or set(provider_seeds) != {AWS_P5_PROFILE}
                or provider_seeds[AWS_P5_PROFILE] != list(AWS_SEEDS)
                or any(
                    type(seed) is not int
                    for seed in provider_seeds[AWS_P5_PROFILE]
                )
            ):
                raise _release_error(
                    "RELEASE_INTERNAL_INVALID",
                    "AWS v3 cohort assignment identity is invalid",
                )
            pointer = _parse_internal_json(
                payload[AWS_DATASET_POINTER_PATH],
                label=AWS_DATASET_POINTER_PATH,
            )
            validate_aws_dataset_pointer_contract(pointer)
    else:
        environment_hashes = require_object(
            metadata["environment_hashes"],
            label="RELEASE-METADATA.json.environment_hashes",
        )
        for relative, raw_digest in environment_hashes.items():
            digest = require_sha256(
                raw_digest,
                label=f"environment hash {relative}",
            )
            if relative not in members or members[relative]["sha256"] != digest:
                raise _release_error(
                    "RELEASE_INTERNAL_INVALID",
                    "environment hash does not bind a release member",
                )
        if "preregistration_sha256" in metadata:
            preregistration_hash = require_sha256(
                metadata["preregistration_sha256"],
                label="RELEASE-METADATA.json.preregistration_sha256",
            )
            preregistration_member = members.get("configs/preregistration-v2.yaml")
            if (
                preregistration_member is None
                or preregistration_member["sha256"] != preregistration_hash
            ):
                raise _release_error(
                    "RELEASE_INTERNAL_INVALID",
                    "preregistration hash does not bind its release member",
                )
    if "seed_assignment" in metadata:
        assignment = require_object(
            metadata["seed_assignment"],
            label="RELEASE-METADATA.json.seed_assignment",
        )
        require_exact_keys(
            assignment,
            {"cohort_id", "provider", "seeds", "arms"},
            label="RELEASE-METADATA.json.seed_assignment",
        )
        expected_seeds = (
            [0]
            if release_value["provider"] == SUPPORTED_PROFILE
            else (
                list(AWS_SEEDS)
                if aws_package_version == AWS_PACKAGE_FORMAT_VERSION
                else [1, 2, 3, 4]
            )
        )
        expected_cohort_id = (
            AWS_COHORT_ID
            if aws_package_version == AWS_PACKAGE_FORMAT_VERSION
            else "memorysplit-confirmatory-v2-360m-n5"
        )
        expected_assignment = {
            "cohort_id": expected_cohort_id,
            "provider": release_value["provider"],
            "seeds": expected_seeds,
            "arms": ["dense", "split90"],
        }
        if not same_typed_value(assignment, expected_assignment):
            raise _release_error(
                "RELEASE_INTERNAL_INVALID",
                "release seed assignment is invalid",
            )
    archive_files = dict(checksum_rows)
    archive_files["SHA256SUMS"] = hashlib.sha256(sums_bytes).hexdigest()
    return members, metadata, archive_files


def load_release(path: Path | str) -> Release:
    release_path = Path(os.path.abspath(os.fspath(path)))
    value = require_object(load_json(release_path, label="release"), label="release")
    aws_package = (
        value.get("provider")
        in {"aws-p5.48xlarge", "aws-p6-b300.48xlarge"}
        and "package_format_version" in value
    )
    selected_lifecycle = (
        aws_package and "provider_selection_sha256" in value
    )
    package_format_version: int | None = None
    if aws_package:
        raw_package_format = value["package_format_version"]
        if type(raw_package_format) is not int or raw_package_format not in {1, 2}:
            raise MsctlError(
                "RELEASE_INVALID",
                "AWS package format must be the integer 1 or 2",
            )
        package_format_version = raw_package_format
    release_fields = {
        "schema_version",
        "release_id",
        "provider",
        "archive",
        "source",
        "members_sha256",
    }
    if aws_package:
        release_fields |= {
            "package_format_version",
            "seed_assignment",
            "cohort_assignment",
            "profile",
            "environment",
            "dataset_pointer",
            "cohort_assignment_sha256",
            "profile_sha256",
            "dataset_pointer_sha256",
            "config_sha256",
        }
        if selected_lifecycle:
            release_fields |= {
                *LIFECYCLE_BINDING_FIELDS,
                "profile_id",
            }
    require_exact_keys(
        value,
        release_fields,
        label="release",
    )
    try:
        require_schema_version(
            value["schema_version"],
            label="release.schema_version",
        )
    except MsctlError as error:
        raise MsctlError(
            "RELEASE_INVALID",
            "release schema version is unsupported",
        ) from error
    if value["provider"] not in {
        SUPPORTED_PROFILE,
        AWS_P5_PROFILE,
        *(
            {"aws-p6-b300.48xlarge"}
            if selected_lifecycle
            else set()
        ),
    }:
        raise MsctlError(
            "RELEASE_INVALID",
            "release provider is unsupported",
        )
    release_id = value["release_id"]
    if (
        not isinstance(release_id, str)
        or not release_id
        or RUN_ID_RE.fullmatch(release_id) is None
    ):
        raise MsctlError("RELEASE_INVALID", "release_id is invalid")
    archive = require_object(value["archive"], label="release.archive")
    require_exact_keys(
        archive,
        {"path", "sha256", "bytes"},
        label="release.archive",
    )
    archive_relative = portable_relative(
        archive["path"],
        label="release.archive.path",
    )
    if (
        isinstance(archive["bytes"], bool)
        or not isinstance(archive["bytes"], int)
        or archive["bytes"] <= 0
    ):
        raise MsctlError("RELEASE_INVALID", "release archive bytes are invalid")
    source = require_object(value["source"], label="release.source")
    if package_format_version == 1 and "tree" in source:
        raise MsctlError(
            "RELEASE_INVALID",
            "AWS package format 1 cannot carry the v3 source tree",
        )
    source_fields = {"commit", "dirty"}
    if package_format_version == AWS_PACKAGE_FORMAT_VERSION:
        source_fields.add("tree")
    require_exact_keys(source, source_fields, label="release.source")
    if (
        not isinstance(source["commit"], str)
        or COMMIT_RE.fullmatch(source["commit"]) is None
        or source["dirty"] is not False
        or (
            package_format_version == AWS_PACKAGE_FORMAT_VERSION
            and (
                not isinstance(source["tree"], str)
                or _GIT_SHA1_RE.fullmatch(source["tree"]) is None
            )
        )
    ):
        raise MsctlError(
            "RELEASE_INVALID",
            "release source must bind a clean Git commit",
        )
    archive_hash = require_sha256(
        archive["sha256"], label="release.archive.sha256"
    )
    members_hash = require_sha256(
        value["members_sha256"], label="release.members_sha256"
    )
    archive_path, archive_data = _read_archive(
        release_path,
        archive_relative,
    )
    if (
        len(archive_data) != archive["bytes"]
        or hashlib.sha256(archive_data).hexdigest() != archive_hash
    ):
        raise _release_error(
            "RELEASE_ARCHIVE_INVALID",
            "release archive byte count or SHA-256 does not match RELEASE.json",
            path=archive_path,
        )
    try:
        checksum_data = _read_regular_relative(
            release_path.parent,
            f"{archive_relative}.sha256",
            label="external archive checksum",
        )
    except MsctlError as error:
        raise _release_error(
            "RELEASE_ARCHIVE_INVALID",
            "external archive checksum must be a non-symlink regular file",
            path=release_path.parent / f"{archive_relative}.sha256",
        ) from error
    expected_checksum = f"{archive_hash}  {archive_path.name}\n".encode("ascii")
    if checksum_data != expected_checksum:
        raise _release_error(
            "RELEASE_ARCHIVE_INVALID",
            "external archive checksum does not match RELEASE.json",
            path=release_path.parent / f"{archive_relative}.sha256",
        )
    members, metadata, archive_files = _verify_release_internals(
        archive_data,
        release_value=value,
    )
    if selected_lifecycle:
        try:
            lifecycle = ProviderLifecycleBinding(
                cohort_id=value["cohort_id"],
                provider=value["provider"],
                profile_id=value["profile_id"],
                profile_sha256=value["profile_sha256"],
                hardware_amendment_sha256=value[
                    "hardware_amendment_sha256"
                ],
                provider_selection_sha256=value[
                    "provider_selection_sha256"
                ],
                provider_selection_version_id=value[
                    "provider_selection_version_id"
                ],
                runtime_lock_sha256=value["runtime_lock_sha256"],
                runtime_sbom_sha256=value["runtime_sbom_sha256"],
                qualification_evidence_sha256=value[
                    "qualification_evidence_sha256"
                ],
                qualification_environment_receipt_sha256=value[
                    "qualification_environment_receipt_sha256"
                ],
                qualification_canary_receipt_sha256=value[
                    "qualification_canary_receipt_sha256"
                ],
                qualification_approval_receipt_sha256=value[
                    "qualification_approval_receipt_sha256"
                ],
                qualification_approval_public_key_sha256=value[
                    "qualification_approval_public_key_sha256"
                ],
                objective_controls_contract_sha256=value[
                    "objective_controls_contract_sha256"
                ],
                account_id=value["account_id"],
                instance_id=value["instance_id"],
                boot_id=value["boot_id"],
                region=value["region"],
                availability_zone=value["availability_zone"],
                purchase_model=value["purchase_model"],
                seed=value["seed"],
                arms=(
                    tuple(value["arms"])
                    if isinstance(value["arms"], list)
                    else value["arms"]
                ),
            )
        except (TypeError, ValueError) as error:
            raise MsctlError(
                "RELEASE_INVALID",
                "release provider lifecycle binding is invalid",
            ) from error
        if any(
            not same_typed_value(metadata.get(field), expected)
            for field, expected in lifecycle.to_dict().items()
        ) or metadata.get("profile_id") != lifecycle.profile_id:
            raise _release_error(
                "RELEASE_INTERNAL_INVALID",
                "release lifecycle fields differ from internal metadata",
            )
    if aws_package:
        profile_hash = require_sha256(
            value["profile_sha256"],
            label="release.profile_sha256",
        )
        if (
            not same_typed_value(value["profile"], metadata["profile"])
            or not same_typed_value(
                value["environment"],
                metadata["environment"],
            )
            or not same_typed_value(
                value["dataset_pointer"],
                metadata["dataset_pointer"],
            )
            or not same_typed_value(
                value["cohort_assignment"],
                metadata["cohort_assignment"],
            )
            or not same_typed_value(
                value["seed_assignment"],
                metadata["seed_assignment"],
            )
            or not same_typed_value(
                value["config_sha256"],
                metadata["config_sha256"],
            )
            or value["cohort_assignment_sha256"]
            != metadata["cohort_assignment"]["sha256"]
            or profile_hash != metadata["profile"]["sha256"]
            or value["dataset_pointer_sha256"]
            != metadata["dataset_pointer"]["sha256"]
        ):
            raise _release_error(
                "RELEASE_INTERNAL_INVALID",
                "AWS release receipt does not bind internal package metadata",
            )
    return Release(
        release_id=release_id,
        provider=str(value["provider"]),
        receipt_sha256=sha256_file(path),
        archive_sha256=archive_hash,
        archive_bytes=int(archive["bytes"]),
        archive_path=archive_path,
        source_commit=source["commit"],
        source_tree=(
            str(source["tree"])
            if package_format_version == AWS_PACKAGE_FORMAT_VERSION
            else None
        ),
        package_format_version=package_format_version,
        members_sha256=members_hash,
        members=members,
        archive_files=archive_files,
        metadata=metadata,
        value=value,
    )


def load_run_manifest(
    path: Path | str,
    *,
    repo_root: Path | str,
) -> RunManifest | RunManifestV3:
    candidate = regular_file(path, label="run manifest")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise MsctlError(
                    "RUN_MANIFEST_INVALID",
                    "run manifest contains a duplicate field",
                )
            result[key] = item
        return result

    try:
        value = json.loads(
            candidate.read_bytes().decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                MsctlError(
                    "RUN_MANIFEST_INVALID",
                    f"run manifest contains non-finite {constant}",
                )
            ),
        )
    except MsctlError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MsctlError(
            "RUN_MANIFEST_INVALID",
            "run manifest must contain one valid UTF-8 JSON value",
        ) from error
    value = require_object(value, label="run manifest")
    schema_version = value.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in {1, 2, 3}
    ):
        raise MsctlError(
            "RUN_MANIFEST_INVALID",
            "run manifest provider or schema is unsupported",
        )
    selected_lifecycle = (
        schema_version == 3 and "provider_selection_sha256" in value
    )
    root_fields = {
        "schema_version",
        "provider",
        "release_sha256",
        "dataset_sha256",
        "runs",
    }
    if schema_version == 2:
        root_fields |= {
            "seed",
            "cohort_assignment_sha256",
            "study_lock_sha256",
            "source_commit",
        }
    elif schema_version == 3:
        root_fields = {
            *(
                LIFECYCLE_BINDING_FIELDS
                if selected_lifecycle
                else ("provider", "cohort_id", "seed", "profile_sha256")
            ),
            "schema_version",
            "release_sha256",
            "release_receipt_sha256",
            "dataset_pointer_sha256",
            "dataset_receipt_sha256",
            "dataset_build_id",
            "ordered_stream_sha256",
            "cohort_assignment_sha256",
            "preregistration_sha256",
            "sealed_evaluation_release_sha256",
            "source_commit",
            "source_tree",
            "runs",
        }
    require_exact_keys(value, root_fields, label="run manifest")
    provider = value["provider"]
    if (
        not isinstance(provider, str)
        or (schema_version == 1 and provider != SUPPORTED_PROFILE)
        or (
            schema_version == 2
            and provider not in {SUPPORTED_PROFILE, AWS_P5_PROFILE}
        )
        or (
            schema_version == 3
            and (
                value["cohort_id"] != AWS_COHORT_ID
                or provider
                not in (
                    {
                        "aws-p5.48xlarge",
                        "aws-p6-b300.48xlarge",
                    }
                    if selected_lifecycle
                    else {"aws-p5.48xlarge"}
                )
            )
        )
    ):
        raise MsctlError(
            "RUN_MANIFEST_INVALID",
            "run manifest provider or schema is unsupported",
        )
    manifest_seed = 0 if schema_version == 1 else value["seed"]
    owned_seeds = (
        tuple(AWS_SEEDS)
        if schema_version == 3
        else ((0,) if provider == SUPPORTED_PROFILE else (1, 2, 3, 4))
    )
    if (
        isinstance(manifest_seed, bool)
        or not isinstance(manifest_seed, int)
        or manifest_seed not in owned_seeds
    ):
        raise MsctlError(
            "SEED_OWNERSHIP_VIOLATION",
            "run manifest seed is not owned by its provider",
        )
    raw_runs = value["runs"]
    if not isinstance(raw_runs, list) or len(raw_runs) != 2:
        raise MsctlError(
            "RUN_MANIFEST_INVALID",
            "run manifest requires exactly one paired seed",
        )
    runs: list[Run] = []
    for index, item in enumerate(raw_runs):
        row = require_object(item, label=f"run manifest.runs[{index}]")
        run_fields = {
            "run_id",
            "arm",
            "seed",
            "config",
            "config_sha256",
        }
        if schema_version in {1, 3}:
            run_fields.add("estimated_gpu_hours")
        require_exact_keys(row, run_fields, label=f"run manifest.runs[{index}]")
        run_id = row["run_id"]
        if (
            not isinstance(run_id, str)
            or RUN_ID_RE.fullmatch(run_id) is None
        ):
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "run_id is invalid",
                details={"index": index},
            )
        if (
            not isinstance(row["arm"], str)
            or row["arm"] not in {"dense", "split90"}
            or type(row["seed"]) is not int
            or row["seed"] != manifest_seed
        ):
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "paired runs must use the manifest seed and explicit arms",
            )
        if schema_version == 3 and run_id != (
            f"memorysplit-v3-360m-s{manifest_seed}-{row['arm']}"
        ):
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "schema-3 run_id does not match its exact seed and arm",
            )
        relative = portable_relative(
            row["config"], label=f"run manifest.runs[{index}].config"
        )
        if schema_version == 3 and relative != (
            f"configs/360m-v3/{row['arm']}-s{manifest_seed}.yaml"
        ):
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "schema-3 config path does not match its exact seed and arm",
            )
        expected_hash = require_sha256(
            row["config_sha256"],
            label=f"run manifest.runs[{index}].config_sha256",
        )
        config = resolve_inside(
            repo_root,
            relative,
            label=f"run manifest.runs[{index}].config",
        )
        if config.is_symlink() or not config.is_file():
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "run config must be a regular file",
                details={"config": relative},
            )
        if sha256_file(config) != expected_hash:
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "run config hash mismatch",
                details={"config": relative},
            )
        if schema_version == 3:
            raw_gpu_hours = row["estimated_gpu_hours"]
            if (
                isinstance(raw_gpu_hours, bool)
                or not isinstance(raw_gpu_hours, (int, float))
                or not math.isfinite(raw_gpu_hours)
                or raw_gpu_hours <= 0
            ):
                raise MsctlError(
                    "RUN_MANIFEST_INVALID",
                    "schema-3 estimated_gpu_hours must be finite and positive",
                )
        runs.append(
            Run(
                run_id=run_id,
                arm=str(row["arm"]),
                seed=manifest_seed,
                config=relative,
                config_sha256=expected_hash,
                estimated_gpu_hours=(
                    require_nonnegative_number(
                        row["estimated_gpu_hours"],
                        label=(
                            f"run manifest.runs[{index}].estimated_gpu_hours"
                        ),
                    )
                    if schema_version == 1
                    else (
                        float(row["estimated_gpu_hours"])
                        if schema_version == 3
                        else 0.0
                    )
                ),
            )
        )
    if (
        {run.arm for run in runs} != {"dense", "split90"}
        or len({run.run_id for run in runs}) != 2
        or len({run.config for run in runs}) != 2
    ):
        raise MsctlError(
            "RUN_MANIFEST_INVALID",
            "run IDs, configs, and Dense/Split90 arms must be unique",
        )
    source_commit: str | None = None
    if schema_version in {2, 3}:
        raw_commit = value["source_commit"]
        if (
            not isinstance(raw_commit, str)
            or COMMIT_RE.fullmatch(raw_commit) is None
        ):
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "run manifest source commit is invalid",
            )
        source_commit = raw_commit
    if schema_version == 3:
        raw_tree = value["source_tree"]
        if (
            not isinstance(raw_tree, str)
            or _GIT_SHA1_RE.fullmatch(raw_tree) is None
        ):
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "run manifest source tree is invalid",
            )
        try:
            lifecycle = (ProviderLifecycleBinding(
                cohort_id=value["cohort_id"],
                provider=value["provider"],
                profile_id=value["profile_id"],
                profile_sha256=value["profile_sha256"],
                hardware_amendment_sha256=value[
                    "hardware_amendment_sha256"
                ],
                provider_selection_sha256=value[
                    "provider_selection_sha256"
                ],
                provider_selection_version_id=value[
                    "provider_selection_version_id"
                ],
                runtime_lock_sha256=value["runtime_lock_sha256"],
                runtime_sbom_sha256=value["runtime_sbom_sha256"],
                qualification_evidence_sha256=value[
                    "qualification_evidence_sha256"
                ],
                qualification_environment_receipt_sha256=value[
                    "qualification_environment_receipt_sha256"
                ],
                qualification_canary_receipt_sha256=value[
                    "qualification_canary_receipt_sha256"
                ],
                qualification_approval_receipt_sha256=value[
                    "qualification_approval_receipt_sha256"
                ],
                qualification_approval_public_key_sha256=value[
                    "qualification_approval_public_key_sha256"
                ],
                objective_controls_contract_sha256=value[
                    "objective_controls_contract_sha256"
                ],
                account_id=value["account_id"],
                instance_id=value["instance_id"],
                boot_id=value["boot_id"],
                region=value["region"],
                availability_zone=value["availability_zone"],
                purchase_model=value["purchase_model"],
                seed=value["seed"],
                arms=(
                    tuple(value["arms"])
                    if isinstance(value["arms"], list)
                    else value["arms"]
                ),
            ) if selected_lifecycle else None)
        except (TypeError, ValueError) as error:
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "run manifest provider lifecycle binding is invalid",
            ) from error
        return RunManifestV3(
            schema_version=3,
            provider=(
                lifecycle.provider if lifecycle is not None else value["provider"]
            ),
            profile_id=(
                lifecycle.profile_id
                if lifecycle is not None
                else "aws-p5.48xlarge-v3"
            ),
            cohort_id=AWS_COHORT_ID,
            seed=manifest_seed,
            release_sha256=require_sha256(
                value["release_sha256"],
                label="run manifest.release_sha256",
            ),
            release_receipt_sha256=require_sha256(
                value["release_receipt_sha256"],
                label="run manifest.release_receipt_sha256",
            ),
            profile_sha256=require_sha256(
                value["profile_sha256"],
                label="run manifest.profile_sha256",
            ),
            hardware_amendment_sha256=(
                lifecycle.hardware_amendment_sha256
                if lifecycle is not None
                else None
            ),
            provider_selection_sha256=(
                lifecycle.provider_selection_sha256
                if lifecycle is not None
                else None
            ),
            provider_selection_version_id=(
                lifecycle.provider_selection_version_id
                if lifecycle is not None
                else None
            ),
            runtime_lock_sha256=(
                lifecycle.runtime_lock_sha256
                if lifecycle is not None
                else None
            ),
            runtime_sbom_sha256=(
                lifecycle.runtime_sbom_sha256
                if lifecycle is not None
                else None
            ),
            qualification_evidence_sha256=(
                lifecycle.qualification_evidence_sha256
                if lifecycle is not None
                else None
            ),
            qualification_environment_receipt_sha256=(
                lifecycle.qualification_environment_receipt_sha256
                if lifecycle is not None
                else None
            ),
            qualification_canary_receipt_sha256=(
                lifecycle.qualification_canary_receipt_sha256
                if lifecycle is not None
                else None
            ),
            qualification_approval_receipt_sha256=(
                lifecycle.qualification_approval_receipt_sha256
                if lifecycle is not None
                else None
            ),
            qualification_approval_public_key_sha256=(
                lifecycle.qualification_approval_public_key_sha256
                if lifecycle is not None
                else None
            ),
            objective_controls_contract_sha256=(
                lifecycle.objective_controls_contract_sha256
                if lifecycle is not None
                else None
            ),
            account_id=(lifecycle.account_id if lifecycle is not None else None),
            instance_id=(
                lifecycle.instance_id if lifecycle is not None else None
            ),
            boot_id=(lifecycle.boot_id if lifecycle is not None else None),
            region=(lifecycle.region if lifecycle is not None else None),
            availability_zone=(
                lifecycle.availability_zone
                if lifecycle is not None
                else None
            ),
            purchase_model=(
                lifecycle.purchase_model if lifecycle is not None else None
            ),
            arms=(
                lifecycle.arms
                if lifecycle is not None
                else tuple(AWS_ARMS)
            ),
            dataset_pointer_sha256=require_sha256(
                value["dataset_pointer_sha256"],
                label="run manifest.dataset_pointer_sha256",
            ),
            dataset_receipt_sha256=require_sha256(
                value["dataset_receipt_sha256"],
                label="run manifest.dataset_receipt_sha256",
            ),
            dataset_build_id=require_sha256(
                value["dataset_build_id"],
                label="run manifest.dataset_build_id",
            ),
            ordered_stream_sha256=require_sha256(
                value["ordered_stream_sha256"],
                label="run manifest.ordered_stream_sha256",
            ),
            cohort_assignment_sha256=require_sha256(
                value["cohort_assignment_sha256"],
                label="run manifest.cohort_assignment_sha256",
            ),
            preregistration_sha256=require_sha256(
                value["preregistration_sha256"],
                label="run manifest.preregistration_sha256",
            ),
            sealed_evaluation_release_sha256=require_sha256(
                value["sealed_evaluation_release_sha256"],
                label="run manifest.sealed_evaluation_release_sha256",
            ),
            source_commit=raw_commit,
            source_tree=raw_tree,
            runs=tuple(sorted(runs, key=lambda run: run.run_id)),
            sha256=canonical_sha256(value),
            value=value,
        )
    return RunManifest(
        schema_version=schema_version,
        provider=str(provider),
        seed=manifest_seed,
        release_sha256=require_sha256(
            value["release_sha256"],
            label="run manifest.release_sha256",
        ),
        dataset_sha256=require_sha256(
            value["dataset_sha256"],
            label="run manifest.dataset_sha256",
        ),
        cohort_assignment_sha256=(
            require_sha256(
                value["cohort_assignment_sha256"],
                label="run manifest.cohort_assignment_sha256",
            )
            if schema_version == 2
            else None
        ),
        study_lock_sha256=(
            require_sha256(
                value["study_lock_sha256"],
                label="run manifest.study_lock_sha256",
            )
            if schema_version == 2
            else None
        ),
        source_commit=source_commit,
        runs=tuple(sorted(runs, key=lambda run: run.run_id)),
        sha256=canonical_sha256(value),
        value=value,
    )


def bind_release(
    release: Release,
    manifest: RunManifest | RunManifestV3,
) -> None:
    mismatch = False
    if isinstance(manifest, RunManifestV3):
        preregistration = release.members.get(AWS_PREREGISTRATION_PATH)
        config_hashes = release.value.get("config_sha256")
        assignment = release.value.get("seed_assignment")
        expected_assignment = {
            "cohort_id": AWS_COHORT_ID,
            "provider": manifest.provider,
            "seeds": list(AWS_SEEDS),
            "arms": ["dense", "split90"],
        }
        selected_manifest = manifest.provider_selection_sha256 is not None
        lifecycle_matches = (
            all(
                same_typed_value(
                    release.value.get(field),
                    (
                        list(manifest.arms)
                        if field == "arms"
                        else getattr(manifest, field)
                    ),
                )
                for field in LIFECYCLE_BINDING_FIELDS
                if field != "seed"
            )
            and release.value.get("profile_id") == manifest.profile_id
            if selected_manifest
            else True
        )
        mismatch = (
            release.package_format_version != AWS_PACKAGE_FORMAT_VERSION
            or release.provider != manifest.provider
            or release.archive_sha256 != manifest.release_sha256
            or release.receipt_sha256 != manifest.release_receipt_sha256
            or release.value.get("profile_sha256")
            != manifest.profile_sha256
            or release.value.get("dataset_pointer_sha256")
            != manifest.dataset_pointer_sha256
            or release.value.get("cohort_assignment_sha256")
            != manifest.cohort_assignment_sha256
            or preregistration is None
            or preregistration.get("sha256")
            != manifest.preregistration_sha256
            or release.source_commit != manifest.source_commit
            or release.source_tree != manifest.source_tree
            or not lifecycle_matches
            or not same_typed_value(assignment, expected_assignment)
            or not isinstance(config_hashes, dict)
            or any(
                run.config not in release.members
                or release.members[run.config].get("sha256")
                != run.config_sha256
                or config_hashes.get(run.config) != run.config_sha256
                for run in manifest.runs
            )
        )
    else:
        mismatch = (
            release.archive_sha256 != manifest.release_sha256
            or release.provider != manifest.provider
            or (
                manifest.schema_version == 2
                and release.source_commit != manifest.source_commit
            )
            or (
                manifest.provider == AWS_P5_PROFILE
                and release.package_format_version
                == AWS_PACKAGE_FORMAT_VERSION
            )
        )
    if mismatch:
        raise MsctlError(
            "RELEASE_RUN_MISMATCH",
            "run manifest does not bind the supplied release",
        )


def verify_release_member(
    release: Release,
    *,
    member_path: str,
    local_path: Path | str,
    label: str,
) -> str:
    """Bind one local runtime file to its authenticated release member."""

    member = release.members.get(member_path)
    if member is None:
        raise MsctlError(
            "RELEASE_MEMBER_MISMATCH",
            f"{label} is absent from the verified release",
            details={"path": member_path},
        )
    candidate = Path(local_path)
    try:
        directory_fd = open_directory(
            candidate.parent,
            label=f"{label} directory",
        )
        try:
            descriptor, parent_fd, _ = open_regular_at(
                directory_fd,
                candidate.name,
                label=label,
            )
            try:
                size, digest = hash_fd(descriptor)
            finally:
                os.close(descriptor)
                os.close(parent_fd)
        finally:
            os.close(directory_fd)
    except MsctlError as error:
        raise MsctlError(
            "RELEASE_MEMBER_MISMATCH",
            f"{label} is not a safe local release member",
            details={"path": member_path},
        ) from error
    if size != member["bytes"] or digest != member["sha256"]:
        raise MsctlError(
            "RELEASE_MEMBER_MISMATCH",
            f"{label} bytes differ from the verified release",
            details={"path": member_path},
        )
    return digest


def read_release_member(
    release: Release,
    *,
    member_path: str,
    release_root: Path,
    label: str,
) -> bytes:
    """Read the same descriptor whose bytes are authenticated as a member."""

    member = release.members.get(member_path)
    if member is None:
        raise MsctlError(
            "RELEASE_MEMBER_MISMATCH",
            f"{label} is absent from the verified release",
            details={"path": member_path},
        )
    root_fd = open_directory(release_root, label="release extraction root")
    try:
        descriptor, parent_fd, _ = open_regular_at(
            root_fd,
            member_path,
            label=label,
        )
        try:
            before = os.fstat(descriptor)
            data = read_fd(descriptor)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
            os.close(parent_fd)
    except MsctlError as error:
        raise MsctlError(
            "RELEASE_MEMBER_MISMATCH",
            f"{label} is not a safe release member",
            details={"path": member_path},
        ) from error
    finally:
        os.close(root_fd)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if (
        before_identity != after_identity
        or len(data) != member["bytes"]
        or hashlib.sha256(data).hexdigest() != member["sha256"]
    ):
        raise MsctlError(
            "RELEASE_MEMBER_MISMATCH",
            f"{label} bytes differ from the verified release",
            details={"path": member_path},
        )
    return data


def verify_release_extraction(
    release: Release,
    release_root: Path | str,
) -> Path:
    """Authenticate every packaged source byte under one absolute runtime root."""

    root = Path(os.path.abspath(os.fspath(release_root)))
    root_fd = open_directory(root, label="release extraction root")
    os.close(root_fd)
    for relative, expected in sorted(release.archive_files.items()):
        try:
            data = _read_regular_relative(
                root,
                relative,
                label="release extraction member",
            )
        except MsctlError as error:
            raise MsctlError(
                "RELEASE_MEMBER_MISMATCH",
                "release extraction is incomplete or unsafe",
                details={"path": relative},
            ) from error
        if hashlib.sha256(data).hexdigest() != expected:
            raise MsctlError(
                "RELEASE_MEMBER_MISMATCH",
                "release extraction bytes differ from the archive",
                details={"path": relative},
            )
    return root


def verify_checkpoint_receipt(
    path: Path | str,
    *,
    release: Release,
    manifest: RunManifest,
) -> CheckpointReceipt:
    if getattr(manifest, "schema_version", None) == 3:
        raise MsctlError(
            "CHECKPOINT_PROVENANCE_MISMATCH",
            "legacy local checkpoint receipts are audit-only for schema-3 v3 manifests",
        )
    receipt_path = Path(path)
    value = require_object(
        load_json(receipt_path, label="checkpoint receipt"),
        label="checkpoint receipt",
    )
    schema_version = value.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in {1, 2}
        or (
            manifest.provider == AWS_P5_PROFILE
            and schema_version != 2
        )
    ):
        raise MsctlError(
            "CHECKPOINT_PROVENANCE_MISMATCH",
            "checkpoint receipt schema is unsupported",
        )
    receipt_fields = {
        "schema_version",
        "provider",
        "release_sha256",
        "run_manifest_sha256",
        "dataset_sha256",
        "checkpoints",
    }
    if schema_version == 2:
        receipt_fields.add("source_commit")
    require_exact_keys(value, receipt_fields, label="checkpoint receipt")
    if (
        value["provider"] != manifest.provider
        or value["release_sha256"] != release.archive_sha256
        or value["run_manifest_sha256"] != manifest.sha256
        or value["dataset_sha256"] != manifest.dataset_sha256
        or (
            schema_version == 2
            and value["source_commit"] != manifest.source_commit
        )
    ):
        raise MsctlError(
            "CHECKPOINT_PROVENANCE_MISMATCH",
            "checkpoint receipt does not bind the run artifacts",
        )
    raw = value["checkpoints"]
    if not isinstance(raw, list) or len(raw) != len(manifest.runs):
        raise MsctlError(
            "CHECKPOINT_PROVENANCE_MISMATCH",
            "checkpoint receipt must contain the complete paired run",
        )
    by_id = {run.run_id: run for run in manifest.runs}
    seen: set[str] = set()
    checkpoints: list[Checkpoint] = []
    for index, item in enumerate(raw):
        row = require_object(item, label=f"checkpoint[{index}]")
        row_fields = {
            "run_id",
            "path",
            "sha256",
            "config_sha256",
            "step",
            "world_size",
        }
        if schema_version == 2:
            row_fields |= {
                "arm",
                "seed",
                "dataset_sha256",
                "source_commit",
            }
        require_exact_keys(row, row_fields, label=f"checkpoint[{index}]")
        run_id = row["run_id"]
        expected_world_size = (
            4 if manifest.provider == AWS_P5_PROFILE else 3
        )
        if (
            not isinstance(run_id, str)
            or run_id not in by_id
            or run_id in seen
            or row["config_sha256"] != by_id[run_id].config_sha256
            or isinstance(row["world_size"], bool)
            or row["world_size"] != expected_world_size
            or isinstance(row["step"], bool)
            or not isinstance(row["step"], int)
            or row["step"] <= 0
            or (
                schema_version == 2
                and (
                    row["arm"] != by_id[run_id].arm
                    or row["seed"] != by_id[run_id].seed
                    or row["dataset_sha256"] != manifest.dataset_sha256
                    or row["source_commit"] != manifest.source_commit
                )
            )
        ):
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                "checkpoint metadata does not match its run",
                details={"index": index},
            )
        seen.add(run_id)
        relative = portable_relative(
            row["path"], label=f"checkpoint[{index}].path"
        )
        checkpoint = resolve_inside(
            receipt_path.parent,
            relative,
            label=f"checkpoint[{index}].path",
        )
        expected_hash = require_sha256(
            row["sha256"], label=f"checkpoint[{index}].sha256"
        )
        if (
            checkpoint.is_symlink()
            or not checkpoint.is_file()
            or sha256_file(checkpoint) != expected_hash
        ):
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                "checkpoint bytes do not match their receipt",
                details={"run_id": run_id},
            )
        checkpoints.append(
            Checkpoint(
                run_id=run_id,
                arm=by_id[run_id].arm,
                seed=by_id[run_id].seed,
                path=checkpoint,
                sha256=expected_hash,
                config_sha256=str(row["config_sha256"]),
                dataset_sha256=manifest.dataset_sha256,
                source_commit=(
                    str(row["source_commit"])
                    if schema_version == 2
                    else manifest.source_commit
                ),
                step=int(row["step"]),
                world_size=int(row["world_size"]),
            )
        )
    if seen != set(by_id):
        raise MsctlError(
            "CHECKPOINT_PROVENANCE_MISMATCH",
            "checkpoint receipt is missing a paired run",
        )
    if len({checkpoint.step for checkpoint in checkpoints}) != 1:
        raise MsctlError(
            "CHECKPOINT_PROVENANCE_MISMATCH",
            "paired checkpoints must record the same optimizer step",
        )
    return CheckpointReceipt(
        schema_version=schema_version,
        sha256=canonical_sha256(value),
        checkpoints=tuple(
            sorted(checkpoints, key=lambda item: item.run_id)
        ),
        value=value,
    )


def _checkpoint_v3_error(message: str) -> MsctlError:
    return MsctlError("CHECKPOINT_PROVENANCE_MISMATCH", message)


def _checkpoint_v3_time(value: object, *, label: str) -> datetime:
    if (
        not isinstance(value, str)
        or _CANONICAL_UTC_RE.fullmatch(value) is None
    ):
        raise _checkpoint_v3_error(
            f"paired checkpoint {label} is not canonical UTC RFC3339"
        )
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise _checkpoint_v3_error(
            f"paired checkpoint {label} is not canonical UTC RFC3339"
        ) from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise _checkpoint_v3_error(
            f"paired checkpoint {label} is not canonical UTC RFC3339"
        )
    return parsed


def parse_paired_checkpoint_receipt_v3(
    payload: bytes,
    *,
    receipt_uri: str,
    receipt_sha256: str,
    receipt_version_id: str,
) -> AwsPairedCheckpointReceiptV3:
    """Parse exact canonical bytes for one version-pinned AWS v3 pair."""

    if not isinstance(payload, bytes) or not payload:
        raise _checkpoint_v3_error("paired checkpoint receipt bytes are missing")

    def unique_object(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise _checkpoint_v3_error(
                    "paired checkpoint receipt repeats a field"
                )
            result[key] = item
        return result

    try:
        parsed = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                _checkpoint_v3_error(
                    "paired checkpoint receipt contains "
                    f"non-finite {constant}"
                )
            ),
        )
    except MsctlError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _checkpoint_v3_error(
            "paired checkpoint receipt is not valid canonical JSON"
        ) from error
    value = require_object(parsed, label="paired checkpoint receipt")
    if canonical_json(value) + b"\n" != payload:
        raise _checkpoint_v3_error(
            "paired checkpoint receipt bytes are not canonical"
        )
    try:
        expected_sha256 = require_sha256(
            receipt_sha256,
            label="paired checkpoint receipt SHA-256",
        )
    except MsctlError as error:
        raise _checkpoint_v3_error(
            "paired checkpoint receipt SHA-256 is invalid"
        ) from error
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise _checkpoint_v3_error(
            "paired checkpoint receipt hash does not match bytes"
        )
    if (
        not isinstance(receipt_version_id, str)
        or receipt_version_id in {"", "null"}
    ):
        raise _checkpoint_v3_error(
            "paired checkpoint receipt version ID is invalid"
        )
    receipt_suffix = (
        f"/receipts/checkpoints/seed-{value.get('seed')}/sha256/"
        f"{expected_sha256}.json"
    )
    parsed_uri = urlsplit(receipt_uri)
    if (
        not isinstance(receipt_uri, str)
        or parsed_uri.scheme != "s3"
        or not parsed_uri.netloc
        or parsed_uri.query
        or parsed_uri.fragment
        or not receipt_uri.endswith(receipt_suffix)
    ):
        raise _checkpoint_v3_error(
            "paired checkpoint receipt URI is invalid"
        )
    object_root = receipt_uri[: -len(receipt_suffix)]
    selected_lifecycle = "provider_selection_sha256" in value
    top_fields = {
        "boot_id",
        "checkpoints",
        "cohort_id",
        "dataset_build_id",
        "dataset_receipt_sha256",
        "environment_receipt_sha256",
        "freshness",
        "instance_id",
        "ordered_stream_sha256",
        "profile_sha256",
        "provider",
        "reason",
        "receipt_type",
        "release_receipt_sha256",
        "release_sha256",
        "request_id",
        "resumable",
        "run_manifest_sha256",
        "schema_version",
        "seed",
        "source_commit",
        "source_tree",
    }
    if selected_lifecycle:
        top_fields |= set(LIFECYCLE_BINDING_FIELDS)
    require_exact_keys(
        value,
        top_fields,
        label="paired checkpoint receipt",
    )
    seed = value["seed"]
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 3
        or value["receipt_type"]
        != "memorysplit-aws-paired-checkpoint-v3"
        or value["provider"]
        not in (
            {"aws-p5.48xlarge", "aws-p6-b300.48xlarge"}
            if selected_lifecycle
            else {"aws-p5.48xlarge"}
        )
        or value["cohort_id"] != AWS_COHORT_ID
        or type(seed) is not int
        or seed not in AWS_SEEDS
        or value["reason"] not in {"periodic", "interruption"}
        or not isinstance(value["request_id"], str)
        or _CHECKPOINT_REQUEST_ID_RE.fullmatch(value["request_id"]) is None
        or not isinstance(value["instance_id"], str)
        or not value["instance_id"]
        or not isinstance(value["boot_id"], str)
        or not value["boot_id"]
        or value["resumable"] is not True
        or not isinstance(value["source_commit"], str)
        or _GIT_SHA1_RE.fullmatch(value["source_commit"]) is None
        or not isinstance(value["source_tree"], str)
        or _GIT_SHA1_RE.fullmatch(value["source_tree"]) is None
    ):
        raise _checkpoint_v3_error(
            "paired checkpoint receipt identity is invalid"
        )
    try:
        lifecycle = (ProviderLifecycleBinding(
            cohort_id=value["cohort_id"],
            provider=value["provider"],
            profile_id=value["profile_id"],
            profile_sha256=value["profile_sha256"],
            hardware_amendment_sha256=value[
                "hardware_amendment_sha256"
            ],
            provider_selection_sha256=value[
                "provider_selection_sha256"
            ],
            provider_selection_version_id=value[
                "provider_selection_version_id"
            ],
            runtime_lock_sha256=value["runtime_lock_sha256"],
            runtime_sbom_sha256=value["runtime_sbom_sha256"],
            qualification_evidence_sha256=value[
                "qualification_evidence_sha256"
            ],
            qualification_environment_receipt_sha256=value[
                "qualification_environment_receipt_sha256"
            ],
            qualification_canary_receipt_sha256=value[
                "qualification_canary_receipt_sha256"
            ],
            qualification_approval_receipt_sha256=value[
                "qualification_approval_receipt_sha256"
            ],
            qualification_approval_public_key_sha256=value[
                "qualification_approval_public_key_sha256"
            ],
            objective_controls_contract_sha256=value[
                "objective_controls_contract_sha256"
            ],
            account_id=value["account_id"],
            instance_id=value["instance_id"],
            boot_id=value["boot_id"],
            region=value["region"],
            availability_zone=value["availability_zone"],
            purchase_model=value["purchase_model"],
            seed=value["seed"],
            arms=(
                tuple(value["arms"])
                if isinstance(value["arms"], list)
                else value["arms"]
            ),
        ) if selected_lifecycle else None)
    except (TypeError, ValueError) as error:
        raise _checkpoint_v3_error(
            "paired checkpoint provider lifecycle binding is invalid"
        ) from error
    hashes: dict[str, str] = {}
    for field in (
        "profile_sha256",
        "environment_receipt_sha256",
        "release_sha256",
        "release_receipt_sha256",
        "run_manifest_sha256",
        "dataset_receipt_sha256",
        "dataset_build_id",
        "ordered_stream_sha256",
    ):
        try:
            hashes[field] = require_sha256(
                value[field],
                label=f"paired checkpoint receipt {field}",
            )
        except MsctlError as error:
            raise _checkpoint_v3_error(
                f"paired checkpoint receipt {field} is invalid"
            ) from error
    freshness = require_object(
        value["freshness"],
        label="paired checkpoint freshness",
    )
    require_exact_keys(
        freshness,
        {
            "deadline_at",
            "max_age_seconds",
            "requested_at",
            "staged_at",
        },
        label="paired checkpoint freshness",
    )
    requested = _checkpoint_v3_time(
        freshness["requested_at"],
        label="requested_at",
    )
    deadline = _checkpoint_v3_time(
        freshness["deadline_at"],
        label="deadline_at",
    )
    staged = _checkpoint_v3_time(
        freshness["staged_at"],
        label="staged_at",
    )
    if (
        type(freshness["max_age_seconds"]) is not int
        or freshness["max_age_seconds"] != 1200
        or (deadline - requested).total_seconds() != 1200
        or not requested <= staged <= deadline
    ):
        raise _checkpoint_v3_error(
            "paired checkpoint freshness window is invalid"
        )
    raw_checkpoints = value["checkpoints"]
    if not isinstance(raw_checkpoints, list) or len(raw_checkpoints) != 2:
        raise _checkpoint_v3_error(
            "paired checkpoint receipt must contain both arms"
        )
    checkpoints: list[AwsCheckpointV3] = []
    for index, arm in enumerate(AWS_ARMS):
        row = require_object(
            raw_checkpoints[index],
            label=f"paired checkpoint[{index}]",
        )
        require_exact_keys(
            row,
            {
                "arm",
                "checkpoint_version",
                "config_fingerprint",
                "config_sha256",
                "data",
                "object",
                "run_id",
                "seed",
                "step",
                "world_size",
            },
            label=f"paired checkpoint[{index}]",
        )
        step = row["step"]
        if (
            row["arm"] != arm
            or not isinstance(row["run_id"], str)
            or RUN_ID_RE.fullmatch(row["run_id"]) is None
            or type(row["seed"]) is not int
            or row["seed"] != seed
            or type(row["checkpoint_version"]) is not int
            or row["checkpoint_version"] != 3
            or type(step) is not int
            or not 0 <= step <= 13_582
            or type(row["world_size"]) is not int
            or row["world_size"] != 4
        ):
            raise _checkpoint_v3_error(
                f"paired checkpoint[{index}] identity is invalid"
            )
        try:
            config_sha256 = require_sha256(
                row["config_sha256"],
                label=f"paired checkpoint[{index}] config",
            )
            config_fingerprint = require_sha256(
                row["config_fingerprint"],
                label=f"paired checkpoint[{index}] fingerprint",
            )
        except MsctlError as error:
            raise _checkpoint_v3_error(
                f"paired checkpoint[{index}] config binding is invalid"
            ) from error
        data_value = require_object(
            row["data"],
            label=f"paired checkpoint[{index}] data",
        )
        require_exact_keys(
            data_value,
            {
                "build_id",
                "global_cursor",
                "ordered_stream_sha256",
                "receipt_sha256",
                "sidecar_name",
            },
            label=f"paired checkpoint[{index}] data",
        )
        try:
            data_receipt = require_sha256(
                data_value["receipt_sha256"],
                label=f"paired checkpoint[{index}] data receipt",
            )
            data_build = require_sha256(
                data_value["build_id"],
                label=f"paired checkpoint[{index}] data build",
            )
            data_ordered = require_sha256(
                data_value["ordered_stream_sha256"],
                label=f"paired checkpoint[{index}] ordered stream",
            )
        except MsctlError as error:
            raise _checkpoint_v3_error(
                f"paired checkpoint[{index}] data binding is invalid"
            ) from error
        expected_sidecar = (
            "dense_target_weights"
            if arm == "dense"
            else "split90_target_weights"
        )
        if (
            data_receipt != hashes["dataset_receipt_sha256"]
            or data_build != hashes["dataset_build_id"]
            or data_ordered != hashes["ordered_stream_sha256"]
            or type(data_value["global_cursor"]) is not int
            or data_value["global_cursor"] != step * 524_288
            or data_value["sidecar_name"] != expected_sidecar
        ):
            raise _checkpoint_v3_error(
                f"paired checkpoint[{index}] data provenance is invalid"
            )
        object_value = require_object(
            row["object"],
            label=f"paired checkpoint[{index}] object",
        )
        require_exact_keys(
            object_value,
            {"bytes", "sha256", "uri", "version_id"},
            label=f"paired checkpoint[{index}] object",
        )
        try:
            object_sha256 = require_sha256(
                object_value["sha256"],
                label=f"paired checkpoint[{index}] object",
            )
        except MsctlError as error:
            raise _checkpoint_v3_error(
                f"paired checkpoint[{index}] object hash is invalid"
            ) from error
        expected_uri = (
            f"{object_root}/checkpoints/seed-{seed}/{arm}/sha256/"
            f"{object_sha256}.pt"
        )
        if (
            object_value["uri"] != expected_uri
            or type(object_value["bytes"]) is not int
            or object_value["bytes"] <= 0
            or not isinstance(object_value["version_id"], str)
            or object_value["version_id"] in {"", "null"}
        ):
            raise _checkpoint_v3_error(
                f"paired checkpoint[{index}] object identity is invalid"
            )
        checkpoints.append(
            AwsCheckpointV3(
                run_id=row["run_id"],
                arm=arm,
                seed=seed,
                checkpoint_version=3,
                step=step,
                world_size=4,
                config_sha256=config_sha256,
                config_fingerprint=config_fingerprint,
                data=AwsCheckpointDataV3(
                    receipt_sha256=data_receipt,
                    build_id=data_build,
                    ordered_stream_sha256=data_ordered,
                    global_cursor=data_value["global_cursor"],
                    sidecar_name=expected_sidecar,
                ),
                object=AwsCheckpointObjectV3(
                    uri=expected_uri,
                    sha256=object_sha256,
                    bytes=object_value["bytes"],
                    version_id=object_value["version_id"],
                ),
            )
        )
    return AwsPairedCheckpointReceiptV3(
        schema_version=3,
        receipt_type="memorysplit-aws-paired-checkpoint-v3",
        provider=(
            lifecycle.provider if lifecycle is not None else value["provider"]
        ),
        profile_id=(
            lifecycle.profile_id
            if lifecycle is not None
            else "aws-p5.48xlarge-v3"
        ),
        cohort_id=AWS_COHORT_ID,
        seed=seed,
        reason=value["reason"],
        request_id=value["request_id"],
        instance_id=value["instance_id"],
        boot_id=value["boot_id"],
        profile_sha256=hashes["profile_sha256"],
        hardware_amendment_sha256=(
            lifecycle.hardware_amendment_sha256
            if lifecycle is not None
            else None
        ),
        provider_selection_sha256=(
            lifecycle.provider_selection_sha256
            if lifecycle is not None
            else None
        ),
        provider_selection_version_id=(
            lifecycle.provider_selection_version_id
            if lifecycle is not None
            else None
        ),
        runtime_lock_sha256=(
            lifecycle.runtime_lock_sha256
            if lifecycle is not None
            else None
        ),
        runtime_sbom_sha256=(
            lifecycle.runtime_sbom_sha256
            if lifecycle is not None
            else None
        ),
        qualification_evidence_sha256=(
            lifecycle.qualification_evidence_sha256
            if lifecycle is not None
            else None
        ),
        environment_receipt_sha256=hashes[
            "environment_receipt_sha256"
        ],
        qualification_environment_receipt_sha256=(
            lifecycle.qualification_environment_receipt_sha256
            if lifecycle is not None
            else None
        ),
        qualification_canary_receipt_sha256=(
            lifecycle.qualification_canary_receipt_sha256
            if lifecycle is not None
            else None
        ),
        qualification_approval_receipt_sha256=(
            lifecycle.qualification_approval_receipt_sha256
            if lifecycle is not None
            else None
        ),
        qualification_approval_public_key_sha256=(
            lifecycle.qualification_approval_public_key_sha256
            if lifecycle is not None
            else None
        ),
        objective_controls_contract_sha256=(
            lifecycle.objective_controls_contract_sha256
            if lifecycle is not None
            else None
        ),
        account_id=(lifecycle.account_id if lifecycle is not None else None),
        region=(lifecycle.region if lifecycle is not None else None),
        availability_zone=(
            lifecycle.availability_zone if lifecycle is not None else None
        ),
        purchase_model=(
            lifecycle.purchase_model if lifecycle is not None else None
        ),
        arms=(
            lifecycle.arms if lifecycle is not None else tuple(AWS_ARMS)
        ),
        release_sha256=hashes["release_sha256"],
        release_receipt_sha256=hashes["release_receipt_sha256"],
        run_manifest_sha256=hashes["run_manifest_sha256"],
        dataset_receipt_sha256=hashes["dataset_receipt_sha256"],
        dataset_build_id=hashes["dataset_build_id"],
        ordered_stream_sha256=hashes["ordered_stream_sha256"],
        source_commit=value["source_commit"],
        source_tree=value["source_tree"],
        requested_at=freshness["requested_at"],
        deadline_at=freshness["deadline_at"],
        staged_at=freshness["staged_at"],
        checkpoints=(checkpoints[0], checkpoints[1]),
        resumable=True,
        uri=receipt_uri,
        sha256=expected_sha256,
        version_id=receipt_version_id,
        bytes=len(payload),
        value=value,
    )


def verify_aws_checkpoint_receipt_v3(
    receipt: AwsPairedCheckpointReceiptV3,
    *,
    release: object,
    manifest: object,
    environment_receipt_sha256: str,
    instance_id: str | None = None,
    boot_id: str | None = None,
) -> AwsPairedCheckpointReceiptV3:
    """Bind a parsed v3 pair to the exact reviewed launch identities."""

    try:
        environment_sha256 = require_sha256(
            environment_receipt_sha256,
            label="AWS environment receipt",
        )
    except MsctlError as error:
        raise _checkpoint_v3_error(
            "paired checkpoint environment provenance is invalid"
        ) from error
    if (
        not isinstance(receipt, AwsPairedCheckpointReceiptV3)
        or getattr(manifest, "schema_version", None) != 3
        or (
            receipt.provider_selection_sha256 is not None
            and any(
                getattr(receipt, field)
                != (
                    tuple(getattr(manifest, field))
                    if field == "arms"
                    and isinstance(getattr(manifest, field, None), list)
                    else getattr(manifest, field, None)
                )
                for field in LIFECYCLE_BINDING_FIELDS
            )
        )
        or receipt.provider != getattr(manifest, "provider", None)
        or receipt.cohort_id != getattr(manifest, "cohort_id", None)
        or receipt.seed != getattr(manifest, "seed", None)
        or receipt.profile_sha256
        != getattr(manifest, "profile_sha256", None)
        or receipt.environment_receipt_sha256 != environment_sha256
        or receipt.release_sha256
        != getattr(manifest, "release_sha256", None)
        or receipt.release_receipt_sha256
        != getattr(manifest, "release_receipt_sha256", None)
        or receipt.run_manifest_sha256 != getattr(manifest, "sha256", None)
        or receipt.dataset_receipt_sha256
        != getattr(manifest, "dataset_receipt_sha256", None)
        or receipt.dataset_build_id
        != getattr(manifest, "dataset_build_id", None)
        or receipt.ordered_stream_sha256
        != getattr(manifest, "ordered_stream_sha256", None)
        or receipt.source_commit != getattr(manifest, "source_commit", None)
        or receipt.source_tree != getattr(manifest, "source_tree", None)
        or receipt.release_sha256
        != getattr(release, "archive_sha256", None)
        or receipt.release_receipt_sha256
        != getattr(release, "receipt_sha256", None)
        or receipt.source_commit != getattr(release, "source_commit", None)
        or receipt.source_tree != getattr(release, "source_tree", None)
        or (instance_id is not None and receipt.instance_id != instance_id)
        or (boot_id is not None and receipt.boot_id != boot_id)
    ):
        raise _checkpoint_v3_error(
            "paired checkpoint provenance does not match reviewed inputs"
        )
    runs = getattr(manifest, "runs", ())
    if not isinstance(runs, tuple) or len(runs) != 2:
        raise _checkpoint_v3_error(
            "paired checkpoint manifest is not a complete pair"
        )
    by_arm = {getattr(run, "arm", None): run for run in runs}
    if set(by_arm) != set(AWS_ARMS):
        raise _checkpoint_v3_error(
            "paired checkpoint manifest arms are incomplete"
        )
    for checkpoint, arm in zip(receipt.checkpoints, AWS_ARMS, strict=True):
        run = by_arm[arm]
        if (
            checkpoint.arm != arm
            or checkpoint.run_id != getattr(run, "run_id", None)
            or checkpoint.seed != getattr(run, "seed", None)
            or checkpoint.seed != receipt.seed
            or checkpoint.world_size != 4
            or checkpoint.checkpoint_version != 3
            or checkpoint.config_sha256
            != getattr(run, "config_sha256", None)
            or checkpoint.data.receipt_sha256
            != receipt.dataset_receipt_sha256
            or checkpoint.data.build_id != receipt.dataset_build_id
            or checkpoint.data.ordered_stream_sha256
            != receipt.ordered_stream_sha256
        ):
            raise _checkpoint_v3_error(
                f"paired checkpoint {arm} provenance does not match its run"
            )
    return receipt
