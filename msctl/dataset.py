"""Immutable dataset pointer and receipt verification."""

from __future__ import annotations

from pathlib import Path

from .errors import MsctlError
from .jsonutil import (
    load_json,
    portable_relative,
    require_exact_keys,
    require_object,
    require_sha256,
    resolve_inside,
    sha256_file,
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
}
RECEIPT_KEYS = {
    "schema_version",
    "dataset_id",
    "provider",
    "release_sha256",
    "dataset_sha256",
    "ordered_stream_sha256",
    "merkle_root",
    "files",
}


def load_pointer(path: Path | str, profile: IlluminaProfile) -> dict[str, object]:
    pointer = require_object(
        load_json(path, label="dataset pointer"),
        label="dataset pointer",
    )
    require_exact_keys(pointer, POINTER_KEYS, label="dataset pointer")
    if (
        pointer["schema_version"] != 1
        or pointer["provider"] != profile.provider
        or pointer["shared_root_env"] != profile.shared_root_env
        or pointer["shared_root_prefix"] != profile.shared_root_prefix
        or pointer["materialization"] != "slurm"
        or pointer["full_corpus_in_release"] is not False
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
    if "/" in receipt:
        raise MsctlError(
            "DATASET_POINTER_INVALID",
            "dataset receipt must live at the dataset root",
        )
    return pointer


def verify_dataset(
    *,
    profile: IlluminaProfile,
    pointer_path: Path | str,
    dataset_root: Path | str,
) -> dict[str, object]:
    pointer = load_pointer(pointer_path, profile)
    root = Path(dataset_root)
    if root.is_symlink() or not root.is_dir():
        raise MsctlError(
            "DATASET_MISSING",
            "dataset root must be a regular directory",
        )
    root = root.resolve()
    receipt_name = str(pointer["required_receipt"])
    receipt_path = resolve_inside(
        root,
        receipt_name,
        label="dataset receipt",
    )
    receipt = require_object(
        load_json(receipt_path, label="dataset receipt"),
        label="dataset receipt",
    )
    require_exact_keys(receipt, RECEIPT_KEYS, label="dataset receipt")
    if (
        receipt["schema_version"] != 1
        or receipt["provider"] != profile.provider
        or receipt["dataset_id"] != pointer["dataset_id"]
    ):
        raise MsctlError(
            "DATASET_RECEIPT_INVALID",
            "dataset receipt does not bind the pointer and provider",
        )
    for field in (
        "release_sha256",
        "dataset_sha256",
        "ordered_stream_sha256",
        "merkle_root",
    ):
        require_sha256(receipt[field], label=f"dataset receipt.{field}")
    raw_files = receipt["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise MsctlError(
            "DATASET_RECEIPT_INVALID",
            "dataset receipt must list at least one file",
        )
    seen: set[str] = set()
    receipted_files: set[str] = set()
    total_bytes = 0
    for index, item in enumerate(raw_files):
        row = require_object(item, label=f"dataset receipt.files[{index}]")
        require_exact_keys(
            row,
            {"path", "bytes", "sha256"},
            label=f"dataset receipt.files[{index}]",
        )
        relative = portable_relative(
            row["path"], label=f"dataset receipt.files[{index}].path"
        )
        if relative in seen or relative == receipt_name:
            raise MsctlError(
                "DATASET_RECEIPT_INVALID",
                "dataset receipt contains a duplicate or self-reference",
            )
        seen.add(relative)
        path = resolve_inside(
            root,
            relative,
            label=f"dataset file {relative}",
        )
        if path.is_symlink() or not path.is_file():
            raise MsctlError(
                "DATASET_FILE_UNSAFE",
                "dataset member must be a regular file",
                details={"path": relative},
            )
        expected_bytes = row["bytes"]
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
        ):
            raise MsctlError(
                "DATASET_RECEIPT_INVALID",
                "dataset member byte count is invalid",
            )
        expected_hash = require_sha256(
            row["sha256"],
            label=f"dataset receipt.files[{index}].sha256",
        )
        if path.stat().st_size != expected_bytes or sha256_file(path) != expected_hash:
            raise MsctlError(
                "DATASET_HASH_MISMATCH",
                "dataset member hash or byte count differs from its receipt",
                details={"path": relative},
            )
        total_bytes += expected_bytes
        receipted_files.add(relative)
    all_paths = list(root.rglob("*"))
    unsafe = [
        path.relative_to(root).as_posix()
        for path in all_paths
        if path.is_symlink()
    ]
    if unsafe:
        raise MsctlError(
            "DATASET_FILE_UNSAFE",
            "dataset root contains a symlink",
            details={"path": unsafe[0]},
        )
    actual_files = {
        path.relative_to(root).as_posix()
        for path in all_paths
        if path.is_file()
    }
    expected_files = receipted_files | {receipt_name}
    if actual_files != expected_files:
        raise MsctlError(
            "DATASET_UNKNOWN_FILES",
            "dataset root contains missing or unreceipted files",
            details={
                "missing": sorted(expected_files - actual_files),
                "unknown": sorted(actual_files - expected_files),
            },
        )
    return {
        "dataset_id": receipt["dataset_id"],
        "dataset_sha256": receipt["dataset_sha256"],
        "ordered_stream_sha256": receipt["ordered_stream_sha256"],
        "merkle_root": receipt["merkle_root"],
        "verified_files": len(receipted_files),
        "verified_bytes": total_bytes,
    }


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
