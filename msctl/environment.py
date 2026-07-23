"""Idempotent hash-bound Python environment materialization."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .contracts import Release
from .errors import MsctlError
from .fsutil import load_json_at, open_directory, open_regular_at
from .jsonutil import (
    atomic_write_json,
    canonical_sha256,
    regular_file,
    require_exact_keys,
    require_object,
    require_schema_version,
    require_sha256,
    sha256_file,
)
from .profile import IlluminaProfile


RECEIPT_NAME = "msctl-env-receipt.json"
LOCK_HEADER = "# memorysplit-illumina-lock-v1"


def _run(command: list[str], *, operation: str) -> None:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"HOME", "LANG", "LC_ALL", "PATH", "TMPDIR", "USER"}
    }
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise MsctlError(
            "ENV_BUILD_FAILED",
            f"{operation} did not complete",
        ) from error
    if completed.returncode != 0:
        raise MsctlError(
            "ENV_BUILD_FAILED",
            f"{operation} failed",
            details={"returncode": completed.returncode},
        )


def _read_receipt(path: Path) -> dict[str, object]:
    try:
        parent_fd = open_directory(
            path.parent,
            label="environment receipt directory",
        )
        try:
            raw = load_json_at(
                parent_fd,
                path.name,
                label="environment receipt",
            )
        finally:
            os.close(parent_fd)
        value = require_object(
            raw,
            label="environment receipt",
        )
        require_exact_keys(
            value,
            {
                "schema_version",
                "provider",
                "profile_sha256",
                "lock_sha256",
                "environment_root",
                "python",
                "platform",
                "cuda_version",
                "created_at",
            },
            label="environment receipt",
        )
        require_schema_version(
            value["schema_version"],
            label="environment receipt.schema_version",
        )
        require_sha256(
            value["profile_sha256"],
            label="environment receipt.profile_sha256",
        )
        require_sha256(
            value["lock_sha256"],
            label="environment receipt.lock_sha256",
        )
        if not all(
            isinstance(value[field], str) and value[field]
            for field in (
                "provider",
                "environment_root",
                "python",
                "platform",
                "cuda_version",
                "created_at",
            )
        ):
            raise MsctlError(
                "SCHEMA_INVALID",
                "environment receipt strings must be non-empty",
            )
    except MsctlError as error:
        raise MsctlError(
            "ENV_RECEIPT_INVALID",
            "environment receipt is invalid",
        ) from error
    return value


def verify_environment_receipt(
    path: Path | str | None,
    *,
    profile: IlluminaProfile,
    release: Release,
) -> dict[str, object]:
    """Authenticate the exact runtime environment before paid submission."""

    if path is None:
        raise MsctlError(
            "ENV_RECEIPT_REQUIRED",
            "paid runtime operations require an environment receipt",
        )
    candidate = Path(os.path.abspath(os.fspath(path)))
    if candidate.name != RECEIPT_NAME:
        raise MsctlError(
            "ENV_RECEIPT_INVALID",
            "environment receipt must use the canonical filename",
        )
    receipt = _read_receipt(candidate)
    root = candidate.parent
    if (
        receipt["provider"] != profile.provider
        or receipt["profile_sha256"] != profile.sha256
        or receipt["environment_root"] != str(root)
        or profile.environment_status != "pinned"
        or profile.python_version is None
        or profile.cuda_version is None
        or receipt["python"] != profile.python_version
        or receipt["platform"] != profile.platform
        or receipt["cuda_version"] != profile.cuda_version
    ):
        raise MsctlError(
            "ENV_PROVENANCE_MISMATCH",
            "environment receipt does not bind the provider profile",
        )
    environment_hashes = release.metadata.get("environment_hashes")
    if (
        not isinstance(environment_hashes, dict)
        or environment_hashes.get(profile.environment_lock)
        != receipt["lock_sha256"]
    ):
        raise MsctlError(
            "ENV_PROVENANCE_MISMATCH",
            "environment receipt lock is not authenticated by the release",
        )
    root_fd = open_directory(root, label="environment root")
    try:
        descriptor, parent_fd, _ = open_regular_at(
            root_fd,
            "bin/python",
            label="environment Python",
        )
        try:
            executable = os.fstat(descriptor)
        finally:
            os.close(descriptor)
            os.close(parent_fd)
    except MsctlError as error:
        raise MsctlError(
            "ENV_PROVENANCE_MISMATCH",
            "environment Python is unavailable or unsafe",
        ) from error
    finally:
        os.close(root_fd)
    if executable.st_mode & 0o111 == 0:
        raise MsctlError(
            "ENV_PROVENANCE_MISMATCH",
            "environment Python is not executable",
        )
    return {
        "root": str(root),
        "receipt": str(candidate),
        "receipt_sha256": canonical_sha256(receipt),
        "lock_sha256": receipt["lock_sha256"],
    }


def _validate_lock_contract(path: Path, profile: IlluminaProfile) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise MsctlError(
            "ENV_LOCK_INVALID",
            "environment lock must be readable UTF-8",
        ) from error
    required_headers = {
        LOCK_HEADER,
        f"# platform: {profile.platform}",
        f"# python-implementation: {profile.python_implementation}",
        f"# python-version: {profile.python_version}",
        f"# cuda-version: {profile.cuda_version}",
    }
    lines = {line.strip() for line in text.splitlines()}
    requirement_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "--"))
    ]
    if (
        not required_headers <= lines
        or not requirement_lines
        or "--hash=sha256:" not in text
    ):
        raise MsctlError(
            "ENV_LOCK_INVALID",
            "environment lock does not bind the pinned platform or hashes",
        )


def _detect_cuda_version() -> str:
    executable = shutil.which("nvcc")
    if executable is None:
        raise MsctlError(
            "ENV_CUDA_UNAVAILABLE",
            "pinned CUDA toolkit cannot be verified because nvcc is unavailable",
        )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"HOME", "LANG", "LC_ALL", "PATH", "TMPDIR", "USER"}
    }
    try:
        completed = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise MsctlError(
            "ENV_CUDA_UNAVAILABLE",
            "pinned CUDA toolkit version could not be queried",
        ) from error
    match = re.search(
        r"\brelease\s+([0-9]+\.[0-9]+(?:\.[0-9]+)?)\b",
        completed.stdout + "\n" + completed.stderr,
    )
    if completed.returncode != 0 or match is None:
        raise MsctlError(
            "ENV_CUDA_UNAVAILABLE",
            "nvcc did not report a supported CUDA release",
        )
    return match.group(1)


def ensure_environment(
    *,
    profile: IlluminaProfile,
    root: Path | str | None,
    lock: Path | str,
    apply: bool,
) -> dict[str, object]:
    if root is None:
        raise MsctlError(
            "ENV_ROOT_REQUIRED",
            "env ensure requires --root or an operator-supplied environment root",
        )
    destination = Path(root)
    if destination.is_symlink() or destination.parent.is_symlink():
        raise MsctlError("UNSAFE_PATH", "environment root must not be a symlink")
    lock_candidate = Path(lock)
    lock_path: Path | None = None
    lock_hash: str | None = None
    if lock_candidate.exists() and not lock_candidate.is_symlink():
        lock_path = regular_file(lock_candidate, label="environment lock")
        lock_hash = sha256_file(lock_path)
    missing_inputs = [
        name
        for name, value in (
            ("python_version", profile.python_version),
            ("cuda_version", profile.cuda_version),
        )
        if value is None
    ]
    plan = {
        "provider": profile.provider,
        "root": str(destination),
        "lock": str(lock_candidate),
        "lock_sha256": lock_hash,
        "site_contract": {
            "status": profile.environment_status,
            "python_implementation": profile.python_implementation,
            "python_version": profile.python_version,
            "platform": profile.platform,
            "cuda_version": profile.cuda_version,
        },
        "missing_operator_inputs": missing_inputs,
        "steps": ["venv", "pip-install-require-hashes", "write-receipt"],
    }
    if not apply:
        return {**plan, "created": False}
    if profile.environment_status != "pinned" or missing_inputs:
        raise MsctlError(
            "ENV_CONTRACT_INCOMPLETE",
            "environment apply requires operator-pinned CPython and CUDA versions",
            details={"missing_operator_inputs": missing_inputs},
        )
    if lock_path is None or lock_hash is None:
        raise MsctlError(
            "ENV_LOCK_REQUIRED",
            "environment apply requires the platform-specific hash lock",
            details={"lock": str(lock_candidate)},
        )
    _validate_lock_contract(lock_path, profile)
    actual_platform = (
        f"{platform.system().lower()}_{platform.machine().lower()}"
    )
    actual_python = platform.python_version()
    actual_cuda = _detect_cuda_version()
    if (
        platform.python_implementation() != profile.python_implementation
        or actual_platform != profile.platform
        or actual_python != profile.python_version
        or actual_cuda != profile.cuda_version
    ):
        raise MsctlError(
            "ENV_PLATFORM_MISMATCH",
            "the current interpreter does not match the pinned site contract",
            details={
                "actual_platform": actual_platform,
                "actual_python": actual_python,
                "actual_cuda": actual_cuda,
            },
        )

    receipt_path = destination / RECEIPT_NAME
    if destination.exists():
        if (
            destination.is_dir()
            and not destination.is_symlink()
            and receipt_path.is_file()
            and not receipt_path.is_symlink()
            and (destination / "bin" / "python").is_file()
        ):
            receipt = _read_receipt(receipt_path)
            if (
                receipt.get("schema_version") == 1
                and receipt.get("provider") == profile.provider
                and receipt.get("profile_sha256") == profile.sha256
                and receipt.get("lock_sha256") == lock_hash
                and receipt.get("environment_root")
                == str(destination.absolute())
                and receipt.get("python") == profile.python_version
                and receipt.get("platform") == profile.platform
                and receipt.get("cuda_version") == profile.cuda_version
            ):
                return {
                    **plan,
                    "created": False,
                    "receipt": str(receipt_path),
                }
            raise MsctlError(
                "ENV_PROVENANCE_MISMATCH",
                "existing environment receipt does not match this request",
            )
        raise MsctlError(
            "ENV_ROOT_NOT_EMPTY",
            "refusing to overwrite an unreceipted environment root",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    try:
        _run(
            [sys.executable, "-m", "venv", "--copies", str(temporary)],
            operation="Python venv creation",
        )
        python = temporary / "bin" / "python"
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--require-hashes",
                "-r",
                str(lock_path),
            ],
            operation="hash-locked dependency installation",
        )
        receipt = {
            "schema_version": 1,
            "provider": profile.provider,
            "profile_sha256": profile.sha256,
            "lock_sha256": lock_hash,
            "environment_root": str(destination.absolute()),
            "python": profile.python_version,
            "platform": profile.platform,
            "cuda_version": profile.cuda_version,
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        atomic_write_json(temporary / RECEIPT_NAME, receipt)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        **plan,
        "created": True,
        "receipt": str(destination / RECEIPT_NAME),
    }
