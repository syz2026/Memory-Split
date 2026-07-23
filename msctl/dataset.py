"""Immutable dataset pointer and receipt verification."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from pathlib import PurePosixPath

from corpusgen.parallel import verify_parallel_corpus

from .contracts import (
    Release,
    RunManifest,
    verify_release_member,
)
from .errors import MsctlError
from .fsutil import (
    atomic_write_json_at,
    hash_fd,
    open_directory,
    open_directory_at,
    open_regular_at,
    read_fd,
)
from .jsonutil import (
    canonical_sha256,
    load_json,
    portable_relative,
    require_exact_keys,
    require_nonnegative_int,
    require_object,
    require_schema_version,
    require_sha256,
    resolve_inside,
)
from .profile import IlluminaProfile
from .slurm import require_tools


POINTER_KEYS = {
    "schema_version",
    "dataset_id",
    "provider",
    "shared_root_env",
    "shared_root_prefix",
    "relative_path",
    "materialization",
    "full_corpus_in_release",
    "source_lock_manifest",
    "required_receipt",
    "receipt_format",
    "identity_scheme",
    "verification_receipt_format",
}
RECEIPT_FORMAT = "memorysplit-parallel-corpus-v1"
IDENTITY_SCHEME = "memorysplit-dataset-binding-v1"
VERIFICATION_FORMAT = "memorysplit-dataset-verification-v1"
VERIFICATION_KEYS = {
    "schema_version",
    "format",
    "provider",
    "release_sha256",
    "run_manifest_sha256",
    "dataset_sha256",
    "dataset_id",
    "dataset_root",
    "publication_root",
    "identity",
    "receipt_sha256",
    "file_identities",
    "verified_files",
    "verified_bytes",
    "verification_sha256",
}
FILE_IDENTITY_KEYS = {
    "path",
    "device",
    "inode",
    "bytes",
    "mtime_ns",
    "ctime_ns",
}


def _publication_root(
    pointer: dict[str, object],
    dataset_root: Path,
    approved_shared_root: Path,
) -> Path:
    relative = PurePosixPath(str(pointer["relative_path"]))
    parts = relative.parts
    prefix = Path(str(pointer["shared_root_prefix"]))
    approved = Path(os.path.abspath(os.fspath(approved_shared_root)))
    try:
        approved.relative_to(prefix)
    except ValueError as error:
        raise MsctlError(
            "DATASET_ROOT_MISMATCH",
            "approved shared root is outside the pointer prefix",
        ) from error
    expected = approved.joinpath(*parts)
    if (
        not dataset_root.is_absolute()
        or dataset_root != expected
    ):
        raise MsctlError(
            "DATASET_ROOT_MISMATCH",
            "dataset root does not match the pointer publication path",
        )
    return approved


def load_pointer(path: Path | str, profile: IlluminaProfile) -> dict[str, object]:
    pointer = require_object(
        load_json(path, label="dataset pointer"),
        label="dataset pointer",
    )
    require_exact_keys(pointer, POINTER_KEYS, label="dataset pointer")
    try:
        require_schema_version(
            pointer["schema_version"],
            label="dataset pointer.schema_version",
        )
    except MsctlError as error:
        raise MsctlError(
            "DATASET_POINTER_INVALID",
            "dataset pointer schema is unsupported",
        ) from error
    if (
        pointer["provider"] != profile.provider
        or pointer["shared_root_env"] != profile.shared_root_env
        or pointer["shared_root_prefix"] != profile.shared_root_prefix
        or pointer["materialization"] != "slurm"
        or pointer["full_corpus_in_release"] is not False
        or pointer["receipt_format"] != RECEIPT_FORMAT
        or pointer["identity_scheme"] != IDENTITY_SCHEME
        or pointer["verification_receipt_format"] != VERIFICATION_FORMAT
    ):
        raise MsctlError(
            "DATASET_POINTER_INVALID",
            "dataset pointer violates the Illumina materialization contract",
        )
    if (
        not isinstance(pointer["dataset_id"], str)
        or not pointer["dataset_id"]
    ):
        raise MsctlError(
            "DATASET_POINTER_INVALID",
            "dataset pointer has an invalid dataset ID",
        )
    portable_relative(pointer["relative_path"], label="dataset relative_path")
    portable_relative(
        pointer["source_lock_manifest"],
        label="dataset source_lock_manifest",
    )
    receipt = portable_relative(
        pointer["required_receipt"],
        label="dataset required_receipt",
    )
    if receipt != "receipt.json":
        raise MsctlError(
            "DATASET_POINTER_INVALID",
            "dataset pointer must use the canonical parallel receipt",
        )
    return pointer


def dataset_identity(
    *,
    pointer: dict[str, object],
    parallel_receipt: dict[str, object],
    receipt_sha256: str,
    source_lock_sha256: str,
) -> dict[str, object]:
    """Derive the sole content identity from recomputed corpus commitments."""

    return {
        "schema_version": 1,
        "identity_scheme": IDENTITY_SCHEME,
        "dataset_id": pointer["dataset_id"],
        "provider": pointer["provider"],
        "source_lock": {
            "path": pointer["source_lock_manifest"],
            "sha256": source_lock_sha256,
        },
        "parallel": {
            "receipt_format": parallel_receipt["format"],
            "receipt_sha256": receipt_sha256,
            "build_id": parallel_receipt["build_id"],
            "ordered_stream_sha256": parallel_receipt[
                "ordered_stream_sha256"
            ],
            "merkle_root_sha256": parallel_receipt[
                "merkle_root_sha256"
            ],
            "packed_stream_sha256": parallel_receipt[
                "packed_stream_sha256"
            ],
        },
    }


def _verify_release_member(
    release: Release,
    *,
    member_path: str,
    local_path: Path,
    label: str,
) -> str:
    try:
        return verify_release_member(
            release,
            member_path=member_path,
            local_path=local_path,
            label=label,
        )
    except MsctlError as error:
        raise MsctlError(
            "DATASET_RELEASE_MISMATCH",
            f"{label} does not match the verified release",
            details={"path": member_path},
        ) from error


def _scan_dataset_files(
    directory_fd: int,
    *,
    prefix: Path = Path(),
) -> set[str]:
    files: set[str] = set()
    for name in sorted(os.listdir(directory_fd)):
        relative = prefix / name
        metadata = os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(metadata.st_mode):
            raise MsctlError(
                "DATASET_RECEIPT_INVALID",
                "dataset must not contain symlinks",
                details={"path": relative.as_posix()},
            )
        if stat.S_ISDIR(metadata.st_mode):
            child_fd = open_directory_at(
                directory_fd,
                name,
                label="dataset root",
            )
            try:
                files.update(
                    _scan_dataset_files(child_fd, prefix=relative)
                )
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise MsctlError(
                "DATASET_RECEIPT_INVALID",
                "dataset contains a non-regular entry",
                details={"path": relative.as_posix()},
            )
        files.add(relative.as_posix())
    return files


def _stable_read_at(root_fd: int, relative: str, *, label: str) -> bytes:
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
        raise MsctlError(
            "DATASET_RECEIPT_INVALID",
            f"{label} changed while being read",
        )
    return data


def _parallel_receipt_at(
    root_fd: int,
    receipt_name: str,
) -> tuple[dict[str, object], bytes]:
    data = _stable_read_at(
        root_fd,
        receipt_name,
        label="parallel corpus receipt",
    )
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MsctlError(
            "DATASET_RECEIPT_INVALID",
            "parallel corpus receipt is not valid UTF-8 JSON",
        ) from error
    if not isinstance(value, dict):
        raise MsctlError(
            "DATASET_RECEIPT_INVALID",
            "parallel corpus receipt must be a JSON object",
        )
    return value, data


def _artifact_rows(
    receipt: dict[str, object],
) -> list[dict[str, object]]:
    raw = receipt.get("artifacts")
    if not isinstance(raw, list):
        raise MsctlError(
            "DATASET_RECEIPT_INVALID",
            "parallel corpus artifacts must be a list",
        )
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        row = require_object(
            item,
            label=f"parallel receipt artifacts[{index}]",
        )
        require_exact_keys(
            row,
            {"path", "bytes", "sha256"},
            label=f"parallel receipt artifacts[{index}]",
        )
        relative = portable_relative(
            row["path"],
            label=f"parallel receipt artifacts[{index}].path",
        )
        if relative in seen:
            raise MsctlError(
                "DATASET_RECEIPT_INVALID",
                "parallel corpus receipt has duplicate artifact paths",
            )
        seen.add(relative)
        rows.append(
            {
                "path": relative,
                "bytes": require_nonnegative_int(
                    row["bytes"],
                    label=f"parallel receipt artifacts[{index}].bytes",
                ),
                "sha256": require_sha256(
                    row["sha256"],
                    label=f"parallel receipt artifacts[{index}].sha256",
                ),
            }
        )
    return sorted(rows, key=lambda row: str(row["path"]))


def _snapshot_dataset(
    root_fd: int,
    *,
    receipt_name: str,
    receipt: dict[str, object],
    receipt_data: bytes,
    verify_hashes: bool,
) -> tuple[list[dict[str, object]], int]:
    artifacts = _artifact_rows(receipt)
    expected_paths = {
        str(row["path"]) for row in artifacts
    } | {receipt_name}
    if _scan_dataset_files(root_fd) != expected_paths:
        raise MsctlError(
            "DATASET_RECEIPT_INVALID",
            "dataset files do not exactly match the canonical receipt",
        )
    identities: list[dict[str, object]] = []
    for row in artifacts:
        relative = str(row["path"])
        descriptor, parent_fd, _ = open_regular_at(
            root_fd,
            relative,
            label="dataset artifact",
        )
        try:
            if verify_hashes:
                size, digest = hash_fd(descriptor)
            else:
                metadata = os.fstat(descriptor)
                size, digest = metadata.st_size, str(row["sha256"])
            metadata = os.fstat(descriptor)
        finally:
            os.close(descriptor)
            os.close(parent_fd)
        if size != row["bytes"] or digest != row["sha256"]:
            raise MsctlError(
                "DATASET_RECEIPT_INVALID",
                "dataset artifact bytes do not match the canonical receipt",
                details={"path": relative},
            )
        identities.append(
            {
                "path": relative,
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "bytes": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
                "ctime_ns": metadata.st_ctime_ns,
            }
        )
    current_receipt_data = _stable_read_at(
        root_fd,
        receipt_name,
        label="parallel corpus receipt",
    )
    if current_receipt_data != receipt_data:
        raise MsctlError(
            "DATASET_RECEIPT_INVALID",
            "parallel corpus receipt changed during verification",
        )
    receipt_descriptor, receipt_parent_fd, _ = open_regular_at(
        root_fd,
        receipt_name,
        label="parallel corpus receipt",
    )
    try:
        receipt_metadata = os.fstat(receipt_descriptor)
    finally:
        os.close(receipt_descriptor)
        os.close(receipt_parent_fd)
    if receipt_metadata.st_size != len(receipt_data):
        raise MsctlError(
            "DATASET_RECEIPT_INVALID",
            "parallel corpus receipt changed during verification",
        )
    identities.append(
        {
            "path": receipt_name,
            "device": receipt_metadata.st_dev,
            "inode": receipt_metadata.st_ino,
            "bytes": receipt_metadata.st_size,
            "mtime_ns": receipt_metadata.st_mtime_ns,
            "ctime_ns": receipt_metadata.st_ctime_ns,
        }
    )
    return (
        sorted(identities, key=lambda row: str(row["path"])),
        sum(int(row["bytes"]) for row in artifacts),
    )


def _bind_dataset_inputs(
    *,
    profile: IlluminaProfile,
    pointer_path: Path | str,
    release: Release,
    manifest: RunManifest,
    repo_root: Path | str,
) -> tuple[dict[str, object], str]:
    pointer = load_pointer(pointer_path, profile)
    if (
        manifest.release_sha256 != release.archive_sha256
        or manifest.provider != profile.provider
    ):
        raise MsctlError(
            "DATASET_RELEASE_MISMATCH",
            "dataset verification inputs do not bind one release and provider",
        )
    _verify_release_member(
        release,
        member_path="DATASET-POINTER.json",
        local_path=Path(pointer_path),
        label="dataset pointer",
    )
    source_lock_relative = str(pointer["source_lock_manifest"])
    source_lock_path = resolve_inside(
        repo_root,
        source_lock_relative,
        label="dataset source lock",
    )
    source_lock_sha256 = _verify_release_member(
        release,
        member_path=source_lock_relative,
        local_path=source_lock_path,
        label="dataset source lock",
    )
    return pointer, source_lock_sha256


def verify_dataset(
    *,
    profile: IlluminaProfile,
    pointer_path: Path | str,
    dataset_root: Path | str,
    approved_shared_root: Path | str,
    release: Release,
    manifest: RunManifest,
    repo_root: Path | str,
) -> dict[str, object]:
    pointer, source_lock_sha256 = _bind_dataset_inputs(
        profile=profile,
        pointer_path=pointer_path,
        release=release,
        manifest=manifest,
        repo_root=repo_root,
    )
    root = Path(os.path.abspath(os.fspath(dataset_root)))
    publication_root = _publication_root(
        pointer,
        root,
        Path(approved_shared_root),
    )
    try:
        root_fd = open_directory(root, label="dataset root")
    except MsctlError as error:
        raise MsctlError(
            "DATASET_MISSING",
            "dataset root must not traverse symlinks or non-directories",
        ) from error
    receipt_name = str(pointer["required_receipt"])
    try:
        receipt_at, receipt_data = _parallel_receipt_at(root_fd, receipt_name)
        try:
            recomputed = verify_parallel_corpus(root)
        except (OSError, ValueError) as error:
            raise MsctlError(
                "DATASET_RECEIPT_INVALID",
                "parallel corpus failed canonical receipt recomputation",
            ) from error
        if receipt_at != recomputed:
            raise MsctlError(
                "DATASET_RECEIPT_INVALID",
                "parallel corpus receipt changed during recomputation",
            )
        if recomputed.get("format") != pointer["receipt_format"]:
            raise MsctlError(
                "DATASET_RECEIPT_INVALID",
                "parallel corpus receipt format does not match the pointer",
            )
        file_identities, verified_bytes = _snapshot_dataset(
            root_fd,
            receipt_name=receipt_name,
            receipt=recomputed,
            receipt_data=receipt_data,
            verify_hashes=True,
        )
    finally:
        os.close(root_fd)
    receipt_sha256 = hashlib.sha256(receipt_data).hexdigest()
    identity = dataset_identity(
        pointer=pointer,
        parallel_receipt=recomputed,
        receipt_sha256=receipt_sha256,
        source_lock_sha256=source_lock_sha256,
    )
    identity_sha256 = canonical_sha256(identity)
    if identity_sha256 != manifest.dataset_sha256:
        raise MsctlError(
            "DATASET_MANIFEST_MISMATCH",
            "recomputed dataset identity does not match the run manifest",
            details={
                "recomputed_dataset_sha256": identity_sha256,
                "manifest_dataset_sha256": manifest.dataset_sha256,
            },
        )
    artifacts = _artifact_rows(recomputed)
    verification: dict[str, object] = {
        "schema_version": 1,
        "format": VERIFICATION_FORMAT,
        "provider": profile.provider,
        "release_sha256": release.archive_sha256,
        "run_manifest_sha256": manifest.sha256,
        "dataset_sha256": identity_sha256,
        "dataset_id": pointer["dataset_id"],
        "dataset_root": str(root),
        "publication_root": str(publication_root),
        "identity": identity,
        "receipt_sha256": receipt_sha256,
        "file_identities": file_identities,
        "verified_files": len(artifacts),
        "verified_bytes": verified_bytes,
    }
    verification["verification_sha256"] = canonical_sha256(verification)
    return verification


def load_dataset_verification(
    path: Path | str,
    *,
    profile: IlluminaProfile,
    pointer_path: Path | str,
    approved_shared_root: Path | str,
    release: Release,
    manifest: RunManifest,
    repo_root: Path | str,
) -> dict[str, object]:
    """Validate a prior full verification using pinned filesystem identities."""

    try:
        verification = require_object(
            load_json(path, label="dataset verification"),
            label="dataset verification",
        )
        require_exact_keys(
            verification,
            VERIFICATION_KEYS,
            label="dataset verification",
        )
        require_schema_version(
            verification["schema_version"],
            label="dataset verification.schema_version",
        )
        verification_sha256 = require_sha256(
            verification["verification_sha256"],
            label="dataset verification.verification_sha256",
        )
        if verification["format"] != VERIFICATION_FORMAT:
            raise MsctlError(
                "SCHEMA_INVALID",
                "dataset verification format is unsupported",
            )
        receipt_sha256 = require_sha256(
            verification["receipt_sha256"],
            label="dataset verification.receipt_sha256",
        )
        for field in (
            "release_sha256",
            "run_manifest_sha256",
            "dataset_sha256",
        ):
            require_sha256(
                verification[field],
                label=f"dataset verification.{field}",
            )
        verified_files = require_nonnegative_int(
            verification["verified_files"],
            label="dataset verification.verified_files",
        )
        verified_bytes = require_nonnegative_int(
            verification["verified_bytes"],
            label="dataset verification.verified_bytes",
        )
    except MsctlError as error:
        raise MsctlError(
            "DATASET_VERIFICATION_INVALID",
            "prior dataset verification schema is invalid",
        ) from error
    unsigned = {
        key: value
        for key, value in verification.items()
        if key != "verification_sha256"
    }
    if canonical_sha256(unsigned) != verification_sha256:
        raise MsctlError(
            "DATASET_VERIFICATION_INVALID",
            "prior dataset verification digest is invalid",
        )
    pointer, source_lock_sha256 = _bind_dataset_inputs(
        profile=profile,
        pointer_path=pointer_path,
        release=release,
        manifest=manifest,
        repo_root=repo_root,
    )
    if (
        verification["provider"] != profile.provider
        or verification["release_sha256"] != release.archive_sha256
        or verification["run_manifest_sha256"] != manifest.sha256
        or verification["dataset_sha256"] != manifest.dataset_sha256
        or verification["dataset_id"] != pointer["dataset_id"]
    ):
        raise MsctlError(
            "DATASET_VERIFICATION_MISMATCH",
            "prior dataset verification does not bind these run inputs",
        )
    root_value = verification["dataset_root"]
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise MsctlError(
            "DATASET_VERIFICATION_INVALID",
            "prior dataset verification root must be absolute",
        )
    publication_value = verification["publication_root"]
    if (
        not isinstance(publication_value, str)
        or not Path(publication_value).is_absolute()
        or _publication_root(
            pointer,
            Path(root_value),
            Path(approved_shared_root),
        )
        != Path(publication_value)
    ):
        raise MsctlError(
            "DATASET_VERIFICATION_INVALID",
            "prior dataset publication root does not match the pointer",
        )
    try:
        root_fd = open_directory(root_value, label="verified dataset root")
    except MsctlError as error:
        raise MsctlError(
            "DATASET_VERIFICATION_STALE",
            "previously verified dataset root is no longer safe",
        ) from error
    receipt_name = str(pointer["required_receipt"])
    try:
        try:
            receipt, receipt_data = _parallel_receipt_at(
                root_fd,
                receipt_name,
            )
            current_identities, current_bytes = _snapshot_dataset(
                root_fd,
                receipt_name=receipt_name,
                receipt=receipt,
                receipt_data=receipt_data,
                verify_hashes=False,
            )
        except MsctlError as error:
            raise MsctlError(
                "DATASET_VERIFICATION_STALE",
                "previously verified dataset layout has changed",
            ) from error
    finally:
        os.close(root_fd)
    if hashlib.sha256(receipt_data).hexdigest() != receipt_sha256:
        raise MsctlError(
            "DATASET_VERIFICATION_STALE",
            "parallel corpus receipt changed after verification",
        )
    if receipt.get("format") != pointer["receipt_format"]:
        raise MsctlError(
            "DATASET_VERIFICATION_STALE",
            "parallel corpus receipt format changed after verification",
        )
    identity = dataset_identity(
        pointer=pointer,
        parallel_receipt=receipt,
        receipt_sha256=receipt_sha256,
        source_lock_sha256=source_lock_sha256,
    )
    if (
        canonical_sha256(verification["identity"]) != canonical_sha256(identity)
        or canonical_sha256(identity) != manifest.dataset_sha256
    ):
        raise MsctlError(
            "DATASET_VERIFICATION_MISMATCH",
            "prior dataset verification identity does not match the manifest",
        )
    recorded_identities = verification["file_identities"]
    if not isinstance(recorded_identities, list):
        raise MsctlError(
            "DATASET_VERIFICATION_INVALID",
            "prior dataset file identities must be a list",
        )
    for index, item in enumerate(recorded_identities):
        try:
            row = require_object(
                item,
                label=f"dataset file identities[{index}]",
            )
            require_exact_keys(
                row,
                FILE_IDENTITY_KEYS,
                label=f"dataset file identities[{index}]",
            )
            portable_relative(
                row["path"],
                label=f"dataset file identities[{index}].path",
            )
            for field in (
                "device",
                "inode",
                "bytes",
                "mtime_ns",
                "ctime_ns",
            ):
                require_nonnegative_int(
                    row[field],
                    label=f"dataset file identities[{index}].{field}",
                )
        except MsctlError as error:
            raise MsctlError(
                "DATASET_VERIFICATION_INVALID",
                "prior dataset file identity schema is invalid",
            ) from error
    artifacts = _artifact_rows(receipt)
    if (
        recorded_identities != current_identities
        or verified_files != len(artifacts)
        or verified_bytes != current_bytes
    ):
        raise MsctlError(
            "DATASET_VERIFICATION_STALE",
            "dataset file identities changed after verification",
        )
    return verification


def write_dataset_verification(
    path: Path | str,
    verification: dict[str, object],
) -> None:
    destination = Path(path)
    parent_fd = open_directory(
        destination.parent,
        label="dataset verification output directory",
    )
    try:
        atomic_write_json_at(
            parent_fd,
            destination.name,
            verification,
            label="dataset verification output",
        )
    finally:
        os.close(parent_fd)


def ensure_dataset(
    *,
    profile: IlluminaProfile,
    pointer_path: Path | str,
    repo_root: Path | str,
    apply: bool,
    environ: dict[str, str] | None = None,
) -> dict[str, object]:
    pointer = load_pointer(pointer_path, profile)
    templates = sorted(
        path.relative_to(Path(repo_root)).as_posix()
        for path in (Path(repo_root) / "cluster" / "slurm").glob(
            "v2_corpus_*.sbatch"
        )
        if path.is_file() and not path.is_symlink()
    )
    plan = {
        "dataset_id": pointer["dataset_id"],
        "relative_path": pointer["relative_path"],
        "source_lock_manifest": pointer["source_lock_manifest"],
        "templates": templates,
        "materialization": "slurm",
    }
    if not apply:
        return plan
    require_tools(["sbatch"], operation="dataset ensure", environ=environ)
    raise MsctlError(
        "EXTERNAL_OPERATION_UNSUPPORTED",
        "dataset submission requires the integrated corpus-stage DAG",
        details={"operation": "dataset ensure", "templates": templates},
    )
