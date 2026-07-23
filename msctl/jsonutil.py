"""Canonical JSON, hashing, and safe local-file primitives."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Collection, Mapping
from pathlib import Path, PurePosixPath

from .errors import MsctlError


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,126}[a-z0-9]$")


def canonical_json(value: object) -> bytes:
    """Return the sole canonical representation used by hashes/signatures."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise MsctlError(
            "INVALID_JSON_VALUE",
            "value cannot be represented as canonical JSON",
        ) from error


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    candidate = regular_file(path, label="hash input")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file(path: Path | str, *, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise MsctlError(
            "UNSAFE_PATH",
            f"{label} must not be a symlink",
            details={"path": str(candidate)},
        )
    if not candidate.is_file():
        raise MsctlError(
            "FILE_NOT_FOUND",
            f"{label} is not a regular file",
            details={"path": str(candidate)},
        )
    return candidate


def load_json(path: Path | str, *, label: str) -> object:
    candidate = regular_file(path, label=label)
    try:
        data = candidate.read_bytes()
        text = data.decode("utf-8")
        value = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MsctlError(
            "INVALID_JSON",
            f"{label} must contain one valid UTF-8 JSON value",
            details={"path": str(candidate)},
        ) from error
    return value


def require_object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise MsctlError(
            "SCHEMA_INVALID",
            f"{label} must be a JSON object",
        )
    return dict(value)


def require_exact_keys(
    value: Mapping[str, object],
    expected: Collection[str],
    *,
    label: str,
) -> None:
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        raise MsctlError(
            "SCHEMA_INVALID",
            f"{label} has missing or unknown fields",
            details={
                "missing": sorted(expected_set - actual),
                "unknown": sorted(actual - expected_set),
            },
        )


def require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise MsctlError("SCHEMA_INVALID", f"{label} must be lowercase SHA-256")
    return value


def require_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MsctlError("SCHEMA_INVALID", f"{label} must be a positive integer")
    return value


def require_nonnegative_number(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < 0
    ):
        raise MsctlError(
            "SCHEMA_INVALID",
            f"{label} must be a nonnegative finite number",
        )
    number = float(value)
    if number == float("inf") or number != number:
        raise MsctlError(
            "SCHEMA_INVALID",
            f"{label} must be a nonnegative finite number",
        )
    return number


def portable_relative(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise MsctlError(
            "SCHEMA_INVALID",
            f"{label} must be a portable relative path",
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value.startswith("~")
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise MsctlError(
            "SCHEMA_INVALID",
            f"{label} must be a portable relative path",
        )
    return path.as_posix()


def resolve_inside(
    root: Path | str,
    relative: str,
    *,
    label: str,
    require_exists: bool = True,
) -> Path:
    root_path = Path(root).resolve()
    portable = portable_relative(relative, label=label)
    candidate = root_path
    for part in PurePosixPath(portable).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise MsctlError(
                "UNSAFE_PATH",
                f"{label} must not traverse a symlink",
                details={"path": portable},
            )
    try:
        resolved = candidate.resolve(strict=require_exists)
        resolved.relative_to(root_path)
    except (FileNotFoundError, ValueError) as error:
        raise MsctlError(
            "UNSAFE_PATH",
            f"{label} must remain inside its root",
            details={"path": portable},
        ) from error
    return resolved


def atomic_write(
    path: Path | str,
    data: bytes,
    *,
    mode: int = 0o600,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise MsctlError(
            "UNSAFE_PATH",
            "refusing to replace a symlink",
            details={"path": str(destination)},
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def atomic_write_json(path: Path | str, value: object) -> Path:
    return atomic_write(path, canonical_json(value) + b"\n")
