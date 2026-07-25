"""Strict release, run-manifest, and checkpoint contracts."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .cohort import load_cohort_assignment as load_cohort_assignment
from .errors import MsctlError
from .fsutil import hash_fd, open_directory, open_regular_at, read_fd
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
    resolve_inside,
    sha256_file,
)
from .profile import (
    AWS_GPU_PROFILES,
    AWS_P5_PROFILE,
    AWS_P5_V3_PROFILE,
    AWS_P6_B300_V3_PROFILE,
    SUPPORTED_PROFILE,
)


@dataclass(frozen=True)
class Release:
    release_id: str
    provider: str
    receipt_sha256: str
    archive_sha256: str
    archive_bytes: int
    archive_path: Path
    source_commit: str
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
    preregistration_sha256: str | None
    hardware_amendment_sha256: str | None
    provider_selection_sha256: str | None
    profile_sha256: str | None
    sealed_fixture_sha256: str | None
    estimated_instance_hours: float
    estimated_gpu_hours: float
    runs: tuple[Run, ...]
    sha256: str
    value: dict[str, object]

    @property
    def gpu_hours(self) -> float:
        if self.schema_version == 3:
            return self.estimated_gpu_hours
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
    checkpoint_uri: str | None = None
    configuration_uri: str | None = None
    run_binding_sha256: str | None = None
    run_binding_uri: str | None = None
    checkpoint_record_sha256: str | None = None
    checkpoint_record_uri: str | None = None


@dataclass(frozen=True)
class CheckpointReceipt:
    schema_version: int
    sha256: str
    checkpoints: tuple[Checkpoint, ...]
    value: dict[str, object]
    cohort_assignment_sha256: str | None = None
    hardware_amendment_sha256: str | None = None
    provider_selection_sha256: str | None = None
    profile_sha256: str | None = None
    preregistration_sha256: str | None = None
    sealed_fixture_sha256: str | None = None


_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_RUNTIME_RECEIPT_FIELDS = [
    "schema_version",
    "profile_sha256",
    "container_image_digest",
    "aws_instance_identity_document",
    "aws_instance_identity_pkcs7",
]


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
    try:
        value = json.loads(data.decode("utf-8"))
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
    aws_provider = release_value["provider"] in AWS_GPU_PROFILES
    aws_package = aws_provider and "package_format_version" in metadata
    aws_gpu_v3_provider = release_value["provider"] in {
        AWS_P5_V3_PROFILE,
        AWS_P6_B300_V3_PROFILE,
    }
    aws_gpu_v3_package = (
        aws_gpu_v3_provider
        and aws_package
        and metadata.get("package_format_version") == "aws-gpu-v3"
    )
    if aws_gpu_v3_provider and not aws_gpu_v3_package:
        raise _release_error(
            "RELEASE_INTERNAL_INVALID",
            "AWS GPU v3 release package format is unsupported",
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
        if aws_gpu_v3_package:
            metadata_fields |= {
                "selected_profile_id",
                "preregistration",
                "hardware_amendment",
                "container_base_lock",
                "contract_locks",
            }
    elif aws_provider:
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
    if (
        metadata["provider"] != release_value["provider"]
        or metadata["source"] != release_value["source"]
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
        else f"cluster/profiles/{release_value['provider']}.json"
    )
    profile_member = members.get(profile_member_path)
    if profile_member is None or profile_member["sha256"] != profile_hash:
        raise _release_error(
            "RELEASE_INTERNAL_INVALID",
            "internal profile hash is not bound to its member",
        )
    if aws_package:
        package_format_version = metadata["package_format_version"]
        if (
            package_format_version != release_value["package_format_version"]
            or (
                aws_gpu_v3_provider
                and package_format_version != "aws-gpu-v3"
            )
            or (
                release_value["provider"] == AWS_P5_PROFILE
                and (
                    type(package_format_version) is not int
                    or package_format_version != 1
                )
            )
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
        for label in ("cohort_assignment", "dataset_pointer"):
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
            if path not in members or members[path]["sha256"] != digest:
                raise _release_error(
                    "RELEASE_INTERNAL_INVALID",
                    f"AWS {label} is not bound to a release member",
                )
        config_hashes = require_object(
            metadata["config_sha256"],
            label="RELEASE-METADATA.json.config_sha256",
        )
        for path, raw_digest in config_hashes.items():
            relative = portable_relative(path, label="AWS config path")
            digest = require_sha256(raw_digest, label="AWS config SHA-256")
            if relative not in members or members[relative]["sha256"] != digest:
                raise _release_error(
                    "RELEASE_INTERNAL_INVALID",
                    "AWS config hash is not bound to a release member",
                )
        if aws_gpu_v3_package:
            if metadata["selected_profile_id"] != release_value["provider"]:
                raise _release_error(
                    "RELEASE_INTERNAL_INVALID",
                    "AWS GPU v3 selected profile identity is invalid",
                )
            for label in ("preregistration", "hardware_amendment"):
                binding = require_object(
                    metadata[label],
                    label=f"RELEASE-METADATA.json.{label}",
                )
                require_exact_keys(
                    binding,
                    {"path", "sha256"},
                    label=f"RELEASE-METADATA.json.{label}",
                )
                relative = portable_relative(
                    binding["path"],
                    label=f"RELEASE-METADATA.json.{label}.path",
                )
                digest = require_sha256(
                    binding["sha256"],
                    label=f"RELEASE-METADATA.json.{label}.sha256",
                )
                if relative not in members or members[relative]["sha256"] != digest:
                    raise _release_error(
                        "RELEASE_INTERNAL_INVALID",
                        f"AWS GPU v3 {label} is not bound to a release member",
                    )
            container_lock = require_object(
                metadata["container_base_lock"],
                label="RELEASE-METADATA.json.container_base_lock",
            )
            require_exact_keys(
                container_lock,
                {"path", "sha256", "base_image", "base_digest"},
                label="RELEASE-METADATA.json.container_base_lock",
            )
            container_path = portable_relative(
                container_lock["path"],
                label="AWS GPU v3 container base lock path",
            )
            container_digest = require_sha256(
                container_lock["sha256"],
                label="AWS GPU v3 container base lock SHA-256",
            )
            if (
                container_path not in members
                or members[container_path]["sha256"] != container_digest
                or not isinstance(container_lock["base_image"], str)
                or not container_lock["base_image"]
                or not isinstance(container_lock["base_digest"], str)
                or not container_lock["base_image"].endswith(
                    "@" + container_lock["base_digest"]
                )
            ):
                raise _release_error(
                    "RELEASE_INTERNAL_INVALID",
                    "AWS GPU v3 container base lock is not immutable and bound",
                )
            contract_locks = require_object(
                metadata["contract_locks"],
                label="RELEASE-METADATA.json.contract_locks",
            )
            require_exact_keys(
                contract_locks,
                {"provider_selection", "fleet", "lifecycle", "canary", "container"},
                label="RELEASE-METADATA.json.contract_locks",
            )
            for lock_name, raw_lock in contract_locks.items():
                lock = require_object(
                    raw_lock,
                    label=f"AWS GPU v3 {lock_name} contract lock",
                )
                require_exact_keys(
                    lock,
                    {"members", "sha256"},
                    label=f"AWS GPU v3 {lock_name} contract lock",
                )
                inventory = require_object(
                    lock["members"],
                    label=f"AWS GPU v3 {lock_name} lock members",
                )
                if not inventory:
                    raise _release_error(
                        "RELEASE_INTERNAL_INVALID",
                        "AWS GPU v3 contract lock inventory is empty",
                    )
                for raw_path, raw_digest in inventory.items():
                    relative = portable_relative(
                        raw_path,
                        label=f"AWS GPU v3 {lock_name} lock path",
                    )
                    digest = require_sha256(
                        raw_digest,
                        label=f"AWS GPU v3 {lock_name} lock member SHA-256",
                    )
                    if (
                        relative not in members
                        or members[relative]["sha256"] != digest
                    ):
                        raise _release_error(
                            "RELEASE_INTERNAL_INVALID",
                            "AWS GPU v3 contract lock is not bound to its members",
                        )
                if require_sha256(
                    lock["sha256"],
                    label=f"AWS GPU v3 {lock_name} aggregate SHA-256",
                ) != canonical_sha256(inventory):
                    raise _release_error(
                        "RELEASE_INTERNAL_INVALID",
                        "AWS GPU v3 contract lock aggregate is invalid",
                    )
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
        seeds = assignment["seeds"]
        if release_value["provider"] == SUPPORTED_PROFILE:
            expected_seeds = [0]
        elif release_value["provider"] == AWS_P5_PROFILE:
            expected_seeds = [1, 2, 3, 4]
        else:
            expected_seeds = list(range(10))
        if (
            not isinstance(assignment["cohort_id"], str)
            or not assignment["cohort_id"]
            or assignment["provider"] != release_value["provider"]
            or not isinstance(seeds, list)
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
            or seeds != expected_seeds
            or assignment["arms"] != ["dense", "split90"]
        ):
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
    aws_gpu_v3_provider = value.get("provider") in {
        AWS_P5_V3_PROFILE,
        AWS_P6_B300_V3_PROFILE,
    }
    aws_package = (
        value.get("provider") in AWS_GPU_PROFILES
        and "package_format_version" in value
    )
    aws_gpu_v3_package = (
        aws_gpu_v3_provider
        and aws_package
        and value.get("package_format_version") == "aws-gpu-v3"
    )
    if aws_gpu_v3_provider and not aws_gpu_v3_package:
        raise MsctlError(
            "RELEASE_INVALID",
            "AWS GPU v3 release package format is unsupported",
        )
    if (
        value.get("provider") == AWS_P5_PROFILE
        and aws_package
        and (
            type(value.get("package_format_version")) is not int
            or value.get("package_format_version") != 1
        )
    ):
        raise MsctlError(
            "RELEASE_INVALID",
            "AWS P5 v2 release package format is unsupported",
        )
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
        if aws_gpu_v3_package:
            release_fields |= {
                "selected_profile_id",
                "preregistration",
                "hardware_amendment",
                "container_base_lock",
                "preregistration_sha256",
                "hardware_amendment_sha256",
                "container_base_lock_sha256",
                "contract_locks",
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
    if value["provider"] not in {SUPPORTED_PROFILE, *AWS_GPU_PROFILES}:
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
    source_fields = {"commit", "dirty"}
    if aws_package:
        source_fields.add("tree")
    require_exact_keys(source, source_fields, label="release.source")
    if (
        not isinstance(source["commit"], str)
        or COMMIT_RE.fullmatch(source["commit"]) is None
        or source["dirty"] is not False
        or (
            aws_package
            and (
                not isinstance(source["tree"], str)
                or _GIT_OBJECT_RE.fullmatch(source["tree"]) is None
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
    if aws_package:
        profile_hash = require_sha256(
            value["profile_sha256"],
            label="release.profile_sha256",
        )
        if (
            value["profile"] != metadata["profile"]
            or value["environment"] != metadata["environment"]
            or value["dataset_pointer"] != metadata["dataset_pointer"]
            or value["cohort_assignment"] != metadata["cohort_assignment"]
            or value["seed_assignment"] != metadata["seed_assignment"]
            or value["config_sha256"] != metadata["config_sha256"]
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
        if aws_gpu_v3_package and (
            value["selected_profile_id"] != value["provider"]
            or value["selected_profile_id"] != metadata["selected_profile_id"]
            or value["preregistration"] != metadata["preregistration"]
            or value["hardware_amendment"] != metadata["hardware_amendment"]
            or value["container_base_lock"] != metadata["container_base_lock"]
            or value["contract_locks"] != metadata["contract_locks"]
            or value["preregistration_sha256"]
            != metadata["preregistration"]["sha256"]
            or value["hardware_amendment_sha256"]
            != metadata["hardware_amendment"]["sha256"]
            or value["container_base_lock_sha256"]
            != metadata["container_base_lock"]["sha256"]
        ):
            raise _release_error(
                "RELEASE_INTERNAL_INVALID",
                "AWS GPU v3 receipt does not bind its closed contracts",
            )
    return Release(
        release_id=release_id,
        provider=str(value["provider"]),
        receipt_sha256=sha256_file(path),
        archive_sha256=archive_hash,
        archive_bytes=int(archive["bytes"]),
        archive_path=archive_path,
        source_commit=source["commit"],
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
) -> RunManifest:
    value = require_object(
        load_json(path, label="run manifest"),
        label="run manifest",
    )
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
        root_fields |= {
            "seed",
            "cohort_assignment_sha256",
            "preregistration_sha256",
            "hardware_amendment_sha256",
            "provider_selection_sha256",
            "profile_sha256",
            "sealed_fixture_sha256",
            "source_commit",
            "estimated_instance_hours",
            "estimated_gpu_hours",
        }
    require_exact_keys(value, root_fields, label="run manifest")
    provider = value["provider"]
    v3_providers = {AWS_P5_V3_PROFILE, AWS_P6_B300_V3_PROFILE}
    valid_provider = (
        (schema_version == 1 and provider == SUPPORTED_PROFILE)
        or (
            schema_version == 2
            and provider in {SUPPORTED_PROFILE, AWS_P5_PROFILE}
        )
        or (schema_version == 3 and provider in v3_providers)
    )
    if not valid_provider:
        raise MsctlError(
            "RUN_MANIFEST_INVALID",
            "run manifest provider or schema is unsupported",
        )
    manifest_seed = 0 if schema_version == 1 else value["seed"]
    if schema_version == 3:
        owned_seeds = tuple(range(10))
    else:
        owned_seeds = (
            (0,) if provider == SUPPORTED_PROFILE else (1, 2, 3, 4)
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
        if schema_version == 1:
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
            row["arm"] not in {"dense", "split90"}
            or isinstance(row["seed"], bool)
            or row["seed"] != manifest_seed
        ):
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "paired runs must use the manifest seed and explicit arms",
            )
        relative = portable_relative(
            row["config"], label=f"run manifest.runs[{index}].config"
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
                    else 0.0
                ),
            )
        )
    if schema_version == 3 and any(
        run.run_id != f"memorysplit-v3-360m-s{manifest_seed}-{run.arm}"
        or run.config
        != f"configs/360m-v3/{run.arm}-s{manifest_seed}.yaml"
        for run in runs
    ):
        raise MsctlError(
            "RUN_MANIFEST_INVALID",
            "v3 run IDs and config paths must identify the frozen seed pair",
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
    estimated_instance_hours = 0.0
    estimated_gpu_hours = 0.0
    if schema_version == 3:
        estimated_instance_hours = require_nonnegative_number(
            value["estimated_instance_hours"],
            label="run manifest.estimated_instance_hours",
        )
        estimated_gpu_hours = require_nonnegative_number(
            value["estimated_gpu_hours"],
            label="run manifest.estimated_gpu_hours",
        )
        if (
            estimated_instance_hours <= 0
            or estimated_gpu_hours <= 0
            or estimated_gpu_hours != estimated_instance_hours * 8
        ):
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "v3 estimated GPU hours must equal eight times instance hours",
            )
    preregistration_sha256 = (
        require_sha256(
            value["preregistration_sha256"],
            label="run manifest.preregistration_sha256",
        )
        if schema_version == 3
        else None
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
            if schema_version in {2, 3}
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
        preregistration_sha256=preregistration_sha256,
        hardware_amendment_sha256=(
            require_sha256(
                value["hardware_amendment_sha256"],
                label="run manifest.hardware_amendment_sha256",
            )
            if schema_version == 3
            else None
        ),
        provider_selection_sha256=(
            require_sha256(
                value["provider_selection_sha256"],
                label="run manifest.provider_selection_sha256",
            )
            if schema_version == 3
            else None
        ),
        profile_sha256=(
            require_sha256(
                value["profile_sha256"],
                label="run manifest.profile_sha256",
            )
            if schema_version == 3
            else None
        ),
        sealed_fixture_sha256=(
            require_sha256(
                value["sealed_fixture_sha256"],
                label="run manifest.sealed_fixture_sha256",
            )
            if schema_version == 3
            else None
        ),
        estimated_instance_hours=estimated_instance_hours,
        estimated_gpu_hours=estimated_gpu_hours,
        runs=tuple(sorted(runs, key=lambda run: run.run_id)),
        sha256=canonical_sha256(value),
        value=value,
    )


def bind_release(release: Release, manifest: RunManifest) -> None:
    if (
        release.archive_sha256 != manifest.release_sha256
        or release.provider != manifest.provider
        or (
            manifest.schema_version in {2, 3}
            and release.source_commit != manifest.source_commit
        )
    ):
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
    require_checkpoint_files: bool = True,
    require_durable_terminal: bool = False,
) -> CheckpointReceipt:
    receipt_path = Path(path)
    value = require_object(
        load_json(receipt_path, label="checkpoint receipt"),
        label="checkpoint receipt",
    )
    schema_version = value.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in {1, 2, 3}
        or (
            manifest.provider == AWS_P5_PROFILE
            and schema_version != 2
        )
        or (
            manifest.schema_version == 3
            and schema_version != 3
        )
        or (
            manifest.schema_version != 3
            and schema_version == 3
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
    if schema_version in {2, 3}:
        receipt_fields.add("source_commit")
    if schema_version == 3:
        receipt_fields |= {
            "cohort_assignment_sha256",
            "preregistration_sha256",
            "hardware_amendment_sha256",
            "provider_selection_sha256",
            "profile_sha256",
            "sealed_fixture_sha256",
        }
    require_exact_keys(value, receipt_fields, label="checkpoint receipt")
    if schema_version == 3:
        try:
            payload = receipt_path.read_bytes()
        except OSError as error:
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                "checkpoint receipt cannot be read",
            ) from error
        if payload != canonical_json(value) + b"\n":
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                "schema-v3 checkpoint receipt must be canonical JSON",
            )
    if (
        value["provider"] != manifest.provider
        or value["release_sha256"] != release.archive_sha256
        or value["run_manifest_sha256"] != manifest.sha256
        or value["dataset_sha256"] != manifest.dataset_sha256
        or (
            schema_version in {2, 3}
            and value["source_commit"] != manifest.source_commit
        )
        or (
            schema_version == 3
            and (
                value["cohort_assignment_sha256"]
                != manifest.cohort_assignment_sha256
                or value["preregistration_sha256"]
                != manifest.preregistration_sha256
                or value["hardware_amendment_sha256"]
                != manifest.hardware_amendment_sha256
                or value["provider_selection_sha256"]
                != manifest.provider_selection_sha256
                or value["profile_sha256"] != manifest.profile_sha256
                or value["sealed_fixture_sha256"]
                != manifest.sealed_fixture_sha256
            )
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
    row_fields = {
        "run_id",
        "path",
        "sha256",
        "config_sha256",
        "step",
        "world_size",
    }
    if schema_version in {2, 3}:
        row_fields |= {
            "arm",
            "seed",
            "dataset_sha256",
            "source_commit",
        }
    durable_fields = {
        "checkpoint_uri",
        "configuration_uri",
        "run_binding_sha256",
        "run_binding_uri",
        "checkpoint_record_sha256",
        "checkpoint_record_uri",
    }
    field_sets = {
        frozenset(item)
        for item in raw
        if isinstance(item, dict)
    }
    allowed_field_sets = {frozenset(row_fields)}
    if schema_version == 3:
        allowed_field_sets.add(frozenset(row_fields | durable_fields))
    if (
        len(field_sets) != 1
        or not all(isinstance(item, dict) for item in raw)
        or not field_sets <= allowed_field_sets
    ):
        raise MsctlError(
            "CHECKPOINT_PROVENANCE_MISMATCH",
            "checkpoint receipt row fields do not match",
        )
    durable_terminal = field_sets == {
        frozenset(row_fields | durable_fields)
    }
    if require_durable_terminal and not durable_terminal:
        raise MsctlError(
            "CHECKPOINT_PROVENANCE_MISMATCH",
            "terminal checkpoint receipt omits durable artifact identities",
        )
    by_id = {run.run_id: run for run in manifest.runs}
    seen: set[str] = set()
    checkpoints: list[Checkpoint] = []
    for index, item in enumerate(raw):
        row = require_object(item, label=f"checkpoint[{index}]")
        require_exact_keys(
            row,
            row_fields | (durable_fields if durable_terminal else set()),
            label=f"checkpoint[{index}]",
        )
        run_id = row["run_id"]
        expected_world_size = 4 if manifest.provider in AWS_GPU_PROFILES else 3
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
                schema_version in {2, 3}
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
            require_exists=require_checkpoint_files,
        )
        expected_hash = require_sha256(
            row["sha256"], label=f"checkpoint[{index}].sha256"
        )
        if require_checkpoint_files and (
            checkpoint.is_symlink()
            or not checkpoint.is_file()
            or sha256_file(checkpoint) != expected_hash
        ):
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                "checkpoint bytes do not match their receipt",
                details={"run_id": run_id},
            )
        durable: dict[str, str | None] = {
            "checkpoint_uri": None,
            "configuration_uri": None,
            "run_binding_sha256": None,
            "run_binding_uri": None,
            "checkpoint_record_sha256": None,
            "checkpoint_record_uri": None,
        }
        if durable_terminal:
            run_binding_sha256 = require_sha256(
                row["run_binding_sha256"],
                label=f"checkpoint[{index}].run_binding_sha256",
            )
            checkpoint_record_sha256 = require_sha256(
                row["checkpoint_record_sha256"],
                label=f"checkpoint[{index}].checkpoint_record_sha256",
            )
            prefix = (
                f"/checkpoints/seed-{row['seed']}/{row['arm']}"
            )
            expected_suffixes = {
                "checkpoint_uri": f"{prefix}/sha256/{expected_hash}.pt",
                "configuration_uri": (
                    f"{prefix}/configuration/sha256/"
                    f"{row['config_sha256']}.yaml"
                ),
                "run_binding_uri": (
                    f"{prefix}/run-binding/sha256/"
                    f"{run_binding_sha256}.json"
                ),
                "checkpoint_record_uri": (
                    f"{prefix}/records/{checkpoint_record_sha256}.json"
                ),
            }
            roots: set[str] = set()
            for field, suffix in expected_suffixes.items():
                uri = row[field]
                if (
                    not isinstance(uri, str)
                    or not uri.startswith("s3://")
                    or not uri.endswith(suffix)
                    or any(character in uri for character in "\n\r\x00")
                ):
                    raise MsctlError(
                        "CHECKPOINT_PROVENANCE_MISMATCH",
                        "checkpoint durable artifact URI is not canonical",
                        details={"run_id": run_id, "field": field},
                    )
                roots.add(uri[: -len(suffix)])
                durable[field] = uri
            if len(roots) != 1:
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    "checkpoint durable artifacts cross object roots",
                    details={"run_id": run_id},
                )
            durable["run_binding_sha256"] = run_binding_sha256
            durable["checkpoint_record_sha256"] = (
                checkpoint_record_sha256
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
                    if schema_version in {2, 3}
                    else manifest.source_commit
                ),
                step=int(row["step"]),
                world_size=int(row["world_size"]),
                checkpoint_uri=durable["checkpoint_uri"],
                configuration_uri=durable["configuration_uri"],
                run_binding_sha256=durable["run_binding_sha256"],
                run_binding_uri=durable["run_binding_uri"],
                checkpoint_record_sha256=durable[
                    "checkpoint_record_sha256"
                ],
                checkpoint_record_uri=durable["checkpoint_record_uri"],
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
        sha256=(
            sha256_file(receipt_path)
            if schema_version == 3
            else canonical_sha256(value)
        ),
        checkpoints=tuple(
            sorted(checkpoints, key=lambda item: item.run_id)
        ),
        value=value,
        cohort_assignment_sha256=getattr(
            manifest,
            "cohort_assignment_sha256",
            None,
        ),
        hardware_amendment_sha256=getattr(
            manifest,
            "hardware_amendment_sha256",
            None,
        ),
        provider_selection_sha256=getattr(
            manifest,
            "provider_selection_sha256",
            None,
        ),
        profile_sha256=getattr(manifest, "profile_sha256", None),
        preregistration_sha256=getattr(
            manifest,
            "preregistration_sha256",
            None,
        ),
        sealed_fixture_sha256=getattr(
            manifest,
            "sealed_fixture_sha256",
            None,
        ),
    )
