"""Idempotent hash-bound Python environment materialization."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .errors import MsctlError
from .jsonutil import atomic_write_json, regular_file, sha256_file
from .profile import IlluminaProfile


RECEIPT_NAME = "msctl-env-receipt.json"


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
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MsctlError(
            "ENV_RECEIPT_INVALID",
            "environment receipt is invalid",
        ) from error
    if not isinstance(value, dict):
        raise MsctlError(
            "ENV_RECEIPT_INVALID",
            "environment receipt is invalid",
        )
    return value


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
    lock_path = regular_file(lock, label="environment lock")
    lock_hash = sha256_file(lock_path)
    plan = {
        "provider": profile.provider,
        "root": str(destination),
        "lock": str(lock_path),
        "lock_sha256": lock_hash,
        "steps": ["venv", "pip-install-require-hashes", "write-receipt"],
    }
    if not apply:
        return {**plan, "created": False}

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
            [sys.executable, "-m", "venv", str(temporary)],
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
            "python": sys.version.split()[0],
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
