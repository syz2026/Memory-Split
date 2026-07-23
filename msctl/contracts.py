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
from .profile import SUPPORTED_PROFILE


@dataclass(frozen=True)
class Release:
    release_id: str
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
    provider: str
    release_sha256: str
    dataset_sha256: str
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
    path: Path
    sha256: str
    config_sha256: str
    step: int
    world_size: int


@dataclass(frozen=True)
class CheckpointReceipt:
    sha256: str
    checkpoints: tuple[Checkpoint, ...]
    value: dict[str, object]


_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


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
    require_exact_keys(
        metadata,
        {
            "schema_version",
            "provider",
            "source",
            "profile_sha256",
            "environment_hashes",
            "members",
        },
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
        require_exact_keys(
            row,
            {"path", "bytes", "sha256", "git_blob"},
            label=f"release member[{index}]",
        )
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
    profile_member = members.get("cluster/profiles/illumina-usfc-prd.json")
    if profile_member is None or profile_member["sha256"] != profile_hash:
        raise _release_error(
            "RELEASE_INTERNAL_INVALID",
            "internal profile hash is not bound to its member",
        )
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
    archive_files = dict(checksum_rows)
    archive_files["SHA256SUMS"] = hashlib.sha256(sums_bytes).hexdigest()
    return members, metadata, archive_files


def load_release(path: Path | str) -> Release:
    release_path = Path(os.path.abspath(os.fspath(path)))
    value = require_object(load_json(release_path, label="release"), label="release")
    require_exact_keys(
        value,
        {
            "schema_version",
            "release_id",
            "provider",
            "archive",
            "source",
            "members_sha256",
        },
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
            "release is not an Illumina v1 release",
        ) from error
    if value["provider"] != SUPPORTED_PROFILE:
        raise MsctlError(
            "RELEASE_INVALID",
            "release is not an Illumina v1 release",
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
    require_exact_keys(
        source,
        {"commit", "dirty"},
        label="release.source",
    )
    if (
        not isinstance(source["commit"], str)
        or COMMIT_RE.fullmatch(source["commit"]) is None
        or source["dirty"] is not False
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
    return Release(
        release_id=release_id,
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
    require_exact_keys(
        value,
        {
            "schema_version",
            "provider",
            "release_sha256",
            "dataset_sha256",
            "runs",
        },
        label="run manifest",
    )
    try:
        require_schema_version(
            value["schema_version"],
            label="run manifest.schema_version",
        )
    except MsctlError as error:
        raise MsctlError(
            "RUN_MANIFEST_INVALID",
            "run manifest provider or schema is unsupported",
        ) from error
    if value["provider"] != SUPPORTED_PROFILE:
        raise MsctlError(
            "RUN_MANIFEST_INVALID",
            "run manifest provider or schema is unsupported",
        )
    raw_runs = value["runs"]
    if not isinstance(raw_runs, list) or len(raw_runs) != 2:
        raise MsctlError(
            "RUN_MANIFEST_INVALID",
            "seed-0 launch requires exactly two runs",
        )
    runs: list[Run] = []
    for index, item in enumerate(raw_runs):
        row = require_object(item, label=f"run manifest.runs[{index}]")
        require_exact_keys(
            row,
            {
                "run_id",
                "arm",
                "seed",
                "config",
                "config_sha256",
                "estimated_gpu_hours",
            },
            label=f"run manifest.runs[{index}]",
        )
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
        if row["arm"] not in {"dense", "split90"} or row["seed"] != 0:
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "seed-0 runs must be Dense and Split90 at seed zero",
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
                seed=0,
                config=relative,
                config_sha256=expected_hash,
                estimated_gpu_hours=require_nonnegative_number(
                    row["estimated_gpu_hours"],
                    label=(
                        f"run manifest.runs[{index}].estimated_gpu_hours"
                    ),
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
    return RunManifest(
        provider=str(value["provider"]),
        release_sha256=require_sha256(
            value["release_sha256"],
            label="run manifest.release_sha256",
        ),
        dataset_sha256=require_sha256(
            value["dataset_sha256"],
            label="run manifest.dataset_sha256",
        ),
        runs=tuple(sorted(runs, key=lambda run: run.run_id)),
        sha256=canonical_sha256(value),
        value=value,
    )


def bind_release(release: Release, manifest: RunManifest) -> None:
    if release.archive_sha256 != manifest.release_sha256:
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
    receipt_path = Path(path)
    value = require_object(
        load_json(receipt_path, label="checkpoint receipt"),
        label="checkpoint receipt",
    )
    require_exact_keys(
        value,
        {
            "schema_version",
            "provider",
            "release_sha256",
            "run_manifest_sha256",
            "dataset_sha256",
            "checkpoints",
        },
        label="checkpoint receipt",
    )
    try:
        require_schema_version(
            value["schema_version"],
            label="checkpoint receipt.schema_version",
        )
    except MsctlError as error:
        raise MsctlError(
            "CHECKPOINT_PROVENANCE_MISMATCH",
            "checkpoint receipt schema is unsupported",
        ) from error
    if (
        value["provider"] != manifest.provider
        or value["release_sha256"] != release.archive_sha256
        or value["run_manifest_sha256"] != manifest.sha256
        or value["dataset_sha256"] != manifest.dataset_sha256
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
        require_exact_keys(
            row,
            {
                "run_id",
                "path",
                "sha256",
                "config_sha256",
                "step",
                "world_size",
            },
            label=f"checkpoint[{index}]",
        )
        run_id = row["run_id"]
        if (
            not isinstance(run_id, str)
            or run_id not in by_id
            or run_id in seen
            or row["config_sha256"] != by_id[run_id].config_sha256
            or row["world_size"] != 3
            or isinstance(row["step"], bool)
            or not isinstance(row["step"], int)
            or row["step"] <= 0
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
                path=checkpoint,
                sha256=expected_hash,
                config_sha256=str(row["config_sha256"]),
                step=int(row["step"]),
                world_size=int(row["world_size"]),
            )
        )
    if seen != set(by_id):
        raise MsctlError(
            "CHECKPOINT_PROVENANCE_MISMATCH",
            "checkpoint receipt is missing a paired run",
        )
    return CheckpointReceipt(
        sha256=canonical_sha256(value),
        checkpoints=tuple(
            sorted(checkpoints, key=lambda item: item.run_id)
        ),
        value=value,
    )
