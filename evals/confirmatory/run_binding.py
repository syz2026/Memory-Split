"""Production construction and exclusive publication of evaluator run.json."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import secrets
import stat
from typing import Mapping

from evals.confirmatory.contracts import CONTRACT_VERSION, canonical_json_bytes


RUN_BINDING_SCHEMA = "memorysplit.confirmatory.run-binding.v2"
RUN_BINDING_FIELDS = frozenset(
    {
        "record_type",
        "schema_version",
        "run_id",
        "checkpoint_path",
        "checkpoint_sha256",
        "configuration_path",
        "configuration_sha256",
        "route_dose_sha256",
        "corpus_sha256",
        "code_sha256",
        "seed",
        "condition_id",
    }
)
_HEX = frozenset("0123456789abcdef")


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _directory_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("secure run binding filesystem operations are unsupported")
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_directory(path: Path) -> int:
    flags = _directory_flags()
    if path.is_absolute():
        descriptor = os.open(path.anchor, flags)
        components = path.parts[1:]
    else:
        descriptor = os.open(".", flags)
        components = path.parts
    try:
        for component in components:
            if component in {"", "."}:
                continue
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _relative_regular(root: Path, path: Path, label: str) -> tuple[str, str]:
    root_resolved = root.resolve(strict=True)
    if path.is_absolute():
        candidate = Path(os.path.abspath(path))
        try:
            relative_path = candidate.relative_to(root_resolved)
        except ValueError as error:
            raise ValueError(f"{label} must be inside the run root") from error
    else:
        relative_path = path
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise ValueError(f"{label} path is unsafe")
    relative = relative_path.as_posix()
    parent_fd = _open_directory(root_resolved)
    try:
        for component in relative_path.parts[:-1]:
            child = os.open(component, _directory_flags(), dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = child
        descriptor = os.open(
            relative_path.parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size <= 0
            ):
                raise ValueError(
                    f"{label} must be one non-empty singly linked file"
                )
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)
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
    ) or size != after.st_size:
        raise ValueError(f"{label} changed while being read")
    return relative, digest.hexdigest()


def validate_run_binding(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the evaluator's exact closed run.json record."""

    if not isinstance(value, Mapping) or set(value) != RUN_BINDING_FIELDS:
        raise ValueError("run binding fields are not exact")
    record = dict(value)
    if record["record_type"] != RUN_BINDING_SCHEMA:
        raise ValueError("run binding record_type is invalid")
    if type(record["schema_version"]) is not int or record["schema_version"] != (
        CONTRACT_VERSION
    ):
        raise ValueError("run binding schema_version is invalid")
    if not isinstance(record["run_id"], str) or not record["run_id"]:
        raise ValueError("run_id must be a non-empty string")
    for field in ("checkpoint_path", "configuration_path"):
        path = record[field]
        if (
            not isinstance(path, str)
            or not path
            or path.startswith(("/", "~"))
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise ValueError(f"{field} must be a safe relative path")
    for field in (
        "checkpoint_sha256",
        "configuration_sha256",
        "route_dose_sha256",
        "corpus_sha256",
        "code_sha256",
    ):
        _sha256(record[field], field)
    if type(record["seed"]) is not int or record["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    if record["condition_id"] not in {"dense", "split90"}:
        raise ValueError("condition_id must be dense or split90")
    return record


def build_run_binding(
    *,
    run_root: Path | str,
    run_id: str,
    checkpoint_path: Path | str,
    configuration_path: Path | str,
    route_dose_sha256: str,
    corpus_sha256: str,
    code_sha256: str,
    seed: int,
    condition_id: str,
) -> dict[str, object]:
    """Build evaluator-ready metadata from the exact completed run files."""

    root = Path(run_root)
    status = root.stat(follow_symlinks=False)
    if root.is_symlink() or not stat.S_ISDIR(status.st_mode):
        raise ValueError("run root must be a regular directory")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if condition_id not in {"dense", "split90"}:
        raise ValueError("condition_id must be dense or split90")
    checkpoint_relative, checkpoint_sha256 = _relative_regular(
        root,
        Path(checkpoint_path),
        "checkpoint",
    )
    configuration_relative, configuration_sha256 = _relative_regular(
        root,
        Path(configuration_path),
        "configuration",
    )
    return validate_run_binding({
        "record_type": RUN_BINDING_SCHEMA,
        "schema_version": CONTRACT_VERSION,
        "run_id": run_id,
        "checkpoint_path": checkpoint_relative,
        "checkpoint_sha256": checkpoint_sha256,
        "configuration_path": configuration_relative,
        "configuration_sha256": configuration_sha256,
        "route_dose_sha256": _sha256(
            route_dose_sha256,
            "route_dose_sha256",
        ),
        "corpus_sha256": _sha256(corpus_sha256, "corpus_sha256"),
        "code_sha256": _sha256(code_sha256, "code_sha256"),
        "seed": seed,
        "condition_id": condition_id,
    })


def write_run_binding(
    path: Path | str,
    value: Mapping[str, object],
) -> Path:
    """Publish canonical run.json without replacing existing evidence."""

    record = validate_run_binding(value)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.name in {"", ".", ".."}:
        raise ValueError("run binding destination must name a file")
    parent_fd = _open_directory(destination.parent)
    temporary = f".{destination.name}.{secrets.token_hex(12)}.tmp"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        try:
            payload = canonical_json_bytes(record)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("run binding write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temporary,
                destination.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise FileExistsError(
                f"run binding already exists: {destination}"
            ) from None
        os.fsync(parent_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)
    return destination
