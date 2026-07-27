"""Canonical hashing and crash-safe single-file publication."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, BinaryIO


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _canonicalize(value: Any, path: str = "$") -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path} contains a non-string object key")
        return {
            key: _canonicalize(value[key], f"{path}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [
            _canonicalize(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{path} contains a non-canonical or nonfinite value")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one accepted UTF-8 JSON representation."""

    return (
        json.dumps(
            _canonicalize(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _contains_parent_reference(path: Path) -> bool:
    return any(part == ".." for part in path.parts)


def _absolute_without_resolution(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def require_regular_file(path: str | Path, *, name: str = "artifact") -> Path:
    candidate = Path(path)
    if _contains_parent_reference(candidate):
        raise ValueError(f"{name} path cannot contain traversal")
    absolute = _absolute_without_resolution(candidate)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name} must be a regular file") from exc
    if resolved != absolute:
        raise ValueError(f"{name} path is not canonical or traverses a symlink")
    metadata = candidate.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{name} must be a regular file")
    return absolute


def _require_output_path(path: Path) -> tuple[Path, Path]:
    if _contains_parent_reference(path):
        raise ValueError("output path cannot contain traversal")
    absolute = _absolute_without_resolution(path)
    parent = absolute.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(
            "output parent must be a regular non-symlink directory"
        )
    try:
        resolved_parent = parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError("output parent must be a regular directory") from exc
    if resolved_parent != parent:
        raise ValueError(
            "output parent is not canonical or traverses a symlink"
        )
    if os.path.lexists(absolute):
        if absolute.is_symlink() or not absolute.is_file():
            raise ValueError(
                "output destination must be a regular non-symlink file"
            )
        if absolute.resolve(strict=True) != absolute:
            raise ValueError("output destination path is not canonical")
    return absolute, parent


def atomic_write_stream(
    path: str | Path,
    writer: Callable[[BinaryIO], object],
) -> Path:
    """Stream one artifact into a synced same-directory temporary."""

    if not callable(writer):
        raise TypeError("atomic stream writer must be callable")
    destination, parent = _require_output_path(Path(path))
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{destination.name}.",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        parent_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_bytes(path: str | Path, content: bytes) -> Path:
    """Replace one file via a synced temporary in the same directory."""

    if not isinstance(content, bytes):
        raise TypeError("atomic content must be bytes")
    return atomic_write_stream(path, lambda stream: stream.write(content))


def atomic_write_json(path: str | Path, value: Any) -> Path:
    return atomic_write_bytes(path, canonical_json_bytes(value))


def sha256_file(path: str | Path) -> str:
    source = require_regular_file(path)
    digest = hashlib.sha256()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("artifact must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        current = os.stat(source, follow_symlinks=False)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != opened.st_dev
            or current.st_ino != opened.st_ino
            or current.st_size != opened.st_size
            or current.st_mtime_ns != opened.st_mtime_ns
        ):
            raise ValueError("artifact changed while it was being hashed")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def load_canonical_json(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> Any:
    source = require_regular_file(path, name="JSON artifact")
    content = source.read_bytes()
    if expected_sha256 is not None:
        if (
            not isinstance(expected_sha256, str)
            or _SHA256_RE.fullmatch(expected_sha256) is None
        ):
            raise ValueError("expected SHA-256 must be lowercase hexadecimal")
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected_sha256:
            raise ValueError("JSON artifact SHA-256 hash mismatch")
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("JSON artifact is invalid") from exc
    if content != canonical_json_bytes(value):
        raise ValueError("JSON artifact is not canonical")
    return value


def validate_sha256(value: object, name: str = "SHA-256") -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def validate_relative_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty relative path")
    if "\\" in value:
        raise ValueError(f"{name} must use portable forward slashes")
    path = Path(value)
    if (
        path.is_absolute()
        or _contains_parent_reference(path)
        or value.startswith("./")
        or "//" in value
        or path.as_posix() != value
    ):
        raise ValueError(
            f"{name} must be a canonical relative path without traversal"
        )
    if any(part in {"", "."} for part in path.parts):
        raise ValueError(f"{name} must be a canonical relative path")
    return value
