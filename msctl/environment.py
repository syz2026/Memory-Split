"""Idempotent hash-bound Python environment materialization."""

from __future__ import annotations

import os
import json
import platform
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .contracts import Release
from .errors import MsctlError
from .fsutil import (
    hash_fd,
    load_json_at,
    open_directory,
    open_directory_at,
    open_regular_at,
)
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
RECEIPT_SCHEMA_VERSION = 2
LOCK_HEADER = "# memorysplit-illumina-lock-v1"
TREE_FORMAT = "memorysplit-environment-tree-v1"
PYTHON_PROBE = r"""
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

root = Path(sys.argv[1])
site_paths = sorted(
    str(path)
    for lib in (root / "lib", root / "lib64")
    if lib.is_dir()
    for path in lib.glob("python*/site-packages")
    if path.is_dir()
)
distributions = sorted(
    {
        (
            str(item.metadata.get("Name") or "").strip().lower(),
            str(item.version).strip(),
        )
        for site_path in site_paths
        for item in importlib.metadata.distributions(path=[site_path])
    }
)
encoded = json.dumps(
    distributions,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
value = {
    "implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "installed_distributions_sha256": hashlib.sha256(encoded).hexdigest(),
}
print(json.dumps(value, sort_keys=True, separators=(",", ":")))
"""


def _tree_entries(
    descriptor: int,
    *,
    prefix: str = "",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        names = sorted(os.listdir(descriptor))
    except OSError as error:
        raise MsctlError(
            "ENV_RUNTIME_MISMATCH",
            "environment tree cannot be enumerated safely",
        ) from error
    for name in names:
        relative = f"{prefix}/{name}" if prefix else name
        if relative == RECEIPT_NAME:
            continue
        try:
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as error:
            raise MsctlError(
                "ENV_RUNTIME_MISMATCH",
                "environment tree changed during measurement",
            ) from error
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISDIR(metadata.st_mode):
            child = open_directory_at(
                descriptor,
                name,
                label="environment directory",
            )
            try:
                rows.append(
                    {
                        "path": relative,
                        "type": "directory",
                        "mode": mode,
                    }
                )
                rows.extend(_tree_entries(child, prefix=relative))
            finally:
                os.close(child)
            continue
        if stat.S_ISREG(metadata.st_mode):
            file_fd, parent_fd, _ = open_regular_at(
                descriptor,
                name,
                label="environment file",
            )
            try:
                before = os.fstat(file_fd)
                size, digest = hash_fd(file_fd)
                after = os.fstat(file_fd)
            finally:
                os.close(file_fd)
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
            if identity_before != identity_after or size != metadata.st_size:
                raise MsctlError(
                    "ENV_RUNTIME_MISMATCH",
                    "environment tree changed during measurement",
                )
            rows.append(
                {
                    "path": relative,
                    "type": "file",
                    "mode": mode,
                    "bytes": size,
                    "sha256": digest,
                }
            )
            continue
        if stat.S_ISLNK(metadata.st_mode):
            try:
                target = os.readlink(name, dir_fd=descriptor)
            except OSError as error:
                raise MsctlError(
                    "ENV_RUNTIME_MISMATCH",
                    "environment link changed during measurement",
                ) from error
            if (
                not target
                or os.path.isabs(target)
                or ".." in Path(target).parts
            ):
                raise MsctlError(
                    "ENV_RUNTIME_MISMATCH",
                    "environment tree contains an unsafe link",
                )
            rows.append(
                {
                    "path": relative,
                    "type": "symlink",
                    "mode": mode,
                    "target": target,
                }
            )
            continue
        raise MsctlError(
            "ENV_RUNTIME_MISMATCH",
            "environment tree contains a special file",
        )
    return rows


def measure_environment(root: Path | str) -> dict[str, object]:
    """Return a deterministic commitment to the installed environment tree."""

    candidate = Path(os.path.abspath(os.fspath(root)))
    root_fd = open_directory(candidate, label="environment root")
    try:
        rows = _tree_entries(root_fd)
    finally:
        os.close(root_fd)
    leaves = [canonical_sha256(row) for row in rows]
    python_rows = [
        row
        for row in rows
        if row.get("path") == "bin/python" and row.get("type") == "file"
    ]
    if len(python_rows) != 1:
        raise MsctlError(
            "ENV_RUNTIME_MISMATCH",
            "environment Python is unavailable or unsafe",
        )
    return {
        "environment_tree_merkle_sha256": canonical_sha256(
            {
                "format": TREE_FORMAT,
                "leaves": leaves,
            }
        ),
        "environment_tree_entries": len(rows),
        "environment_tree_bytes": sum(
            int(row.get("bytes", 0)) for row in rows
        ),
        "python_executable_sha256": python_rows[0]["sha256"],
    }


def _probe_python(root: Path) -> dict[str, str]:
    root_fd = open_directory(root, label="environment root")
    try:
        descriptor, parent_fd, _ = open_regular_at(
            root_fd,
            "bin/python",
            label="environment Python",
        )
    finally:
        os.close(root_fd)
    probe_link: Path | None = None
    try:
        initial = os.fstat(descriptor)
        if initial.st_mode & 0o111 == 0:
            raise MsctlError(
                "ENV_RUNTIME_MISMATCH",
                "environment Python is not executable",
            )
        proc_path = f"/proc/self/fd/{descriptor}"
        descriptor_bound = Path("/proc/self/fd").is_dir()
        if descriptor_bound:
            executable = proc_path
        else:
            for _ in range(16):
                candidate = root / "bin" / (
                    f".msctl-python-probe-{secrets.token_hex(12)}"
                )
                try:
                    os.link(
                        root / "bin" / "python",
                        candidate,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    continue
                probe_link = candidate
                break
            if probe_link is None:
                raise MsctlError(
                    "ENV_RUNTIME_MISMATCH",
                    "environment Python could not be descriptor-pinned",
                )
            linked_fd = os.open(
                probe_link,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                linked = os.fstat(linked_fd)
            finally:
                os.close(linked_fd)
            if (linked.st_dev, linked.st_ino) != (
                initial.st_dev,
                initial.st_ino,
            ):
                raise MsctlError(
                    "ENV_RUNTIME_MISMATCH",
                    "environment Python changed before its probe",
                )
            executable = str(probe_link)
        metadata = os.fstat(descriptor)
        environment = {
            "LANG": "C.UTF-8",
            "PATH": os.defpath,
            "PYTHONNOUSERSITE": "1",
        }
        try:
            completed = subprocess.run(
                [executable, "-I", "-c", PYTHON_PROBE, str(root)],
                pass_fds=(descriptor,) if descriptor_bound else (),
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise MsctlError(
                "ENV_RUNTIME_MISMATCH",
                "environment Python probe failed",
            ) from error
        after = os.fstat(descriptor)
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise MsctlError(
                "ENV_RUNTIME_MISMATCH",
                "environment Python changed during its probe",
            )
    finally:
        if probe_link is not None:
            try:
                probe_link.unlink()
            except OSError:
                pass
        os.close(descriptor)
        os.close(parent_fd)
    try:
        value = json.loads(completed.stdout)
    except (json.JSONDecodeError, UnboundLocalError) as error:
        raise MsctlError(
            "ENV_RUNTIME_MISMATCH",
            "environment Python returned an invalid probe",
        ) from error
    if (
        completed.returncode != 0
        or not isinstance(value, dict)
        or set(value)
        != {
            "implementation",
            "python_version",
            "installed_distributions_sha256",
        }
        or not all(isinstance(item, str) and item for item in value.values())
        or re.fullmatch(
            r"[0-9a-f]{64}",
            value["installed_distributions_sha256"],
        )
        is None
    ):
        raise MsctlError(
            "ENV_RUNTIME_MISMATCH",
            "environment Python returned an invalid probe",
        )
    return value


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
                "release_sha256",
                "lock_sha256",
                "environment_root",
                "python",
                "platform",
                "cuda_version",
                "cuda_driver_version",
                "environment_tree_merkle_sha256",
                "environment_tree_entries",
                "environment_tree_bytes",
                "python_executable_sha256",
                "python_probe_sha256",
                "installed_distributions_sha256",
                "created_at",
            },
            label="environment receipt",
        )
        require_schema_version(
            value["schema_version"],
            expected=RECEIPT_SCHEMA_VERSION,
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
        for field in (
            "release_sha256",
            "environment_tree_merkle_sha256",
            "python_executable_sha256",
            "python_probe_sha256",
            "installed_distributions_sha256",
        ):
            require_sha256(
                value[field],
                label=f"environment receipt.{field}",
            )
        for field in (
            "environment_tree_entries",
            "environment_tree_bytes",
        ):
            if (
                not isinstance(value[field], int)
                or isinstance(value[field], bool)
                or value[field] < 0
            ):
                raise MsctlError(
                    "SCHEMA_INVALID",
                    "environment receipt tree counts must be nonnegative integers",
                )
        if not all(
            isinstance(value[field], str) and value[field]
            for field in (
                "provider",
                "environment_root",
                "python",
                "platform",
                "cuda_version",
                "cuda_driver_version",
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
    probe_site: bool = False,
    environ: dict[str, str] | None = None,
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
        or receipt["release_sha256"] != release.archive_sha256
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
    try:
        runtime = _runtime_receipt_fields(root, profile=profile)
    except MsctlError as error:
        if error.code == "ENV_RUNTIME_MISMATCH":
            raise
        raise MsctlError(
            "ENV_RUNTIME_MISMATCH",
            "installed environment could not be authenticated",
        ) from error
    if any(receipt[field] != value for field, value in runtime.items()):
        raise MsctlError(
            "ENV_RUNTIME_MISMATCH",
            "installed environment does not match its receipt commitment",
        )
    if probe_site:
        actual_cuda = _detect_cuda_version(environ)
        actual_driver = _detect_cuda_driver(environ)
        if (
            actual_cuda != profile.cuda_version
            or actual_driver != receipt["cuda_driver_version"]
        ):
            raise MsctlError(
                "ENV_RUNTIME_MISMATCH",
                "site CUDA or driver probe does not match the receipt",
            )
    return {
        "root": str(root),
        "receipt": str(candidate),
        "receipt_sha256": canonical_sha256(receipt),
        "lock_sha256": receipt["lock_sha256"],
        "environment_tree_merkle_sha256": receipt[
            "environment_tree_merkle_sha256"
        ],
        "python_probe_sha256": receipt["python_probe_sha256"],
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


def _probe_environment(
    environ: dict[str, str] | None = None,
) -> dict[str, str]:
    source = os.environ if environ is None else environ
    result = {
        key: value
        for key, value in source.items()
        if key in {"HOME", "LANG", "LC_ALL", "PATH", "TMPDIR", "USER"}
    }
    result.setdefault("LANG", "C.UTF-8")
    return result


def _detect_cuda_version(environ: dict[str, str] | None = None) -> str:
    environment = _probe_environment(environ)
    executable = shutil.which("nvcc", path=environment.get("PATH"))
    if executable is None:
        raise MsctlError(
            "ENV_CUDA_UNAVAILABLE",
            "pinned CUDA toolkit cannot be verified because nvcc is unavailable",
        )
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


def _detect_cuda_driver(environ: dict[str, str] | None = None) -> str:
    environment = _probe_environment(environ)
    executable = shutil.which("nvidia-smi", path=environment.get("PATH"))
    if executable is None:
        raise MsctlError(
            "ENV_CUDA_UNAVAILABLE",
            "pinned CUDA driver cannot be verified because nvidia-smi is unavailable",
        )
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise MsctlError(
            "ENV_CUDA_UNAVAILABLE",
            "pinned CUDA driver version could not be queried",
        ) from error
    versions = {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    }
    if (
        completed.returncode != 0
        or len(versions) != 1
        or re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", next(iter(versions), ""))
        is None
    ):
        raise MsctlError(
            "ENV_CUDA_UNAVAILABLE",
            "nvidia-smi did not report one CUDA driver version",
        )
    return next(iter(versions))


def _runtime_receipt_fields(
    root: Path,
    *,
    profile: IlluminaProfile,
) -> dict[str, object]:
    before = measure_environment(root)
    probe = _probe_python(root)
    after = measure_environment(root)
    if before != after:
        raise MsctlError(
            "ENV_RUNTIME_MISMATCH",
            "environment tree changed during its Python probe",
        )
    if (
        probe["implementation"] != profile.python_implementation
        or probe["python_version"] != profile.python_version
    ):
        raise MsctlError(
            "ENV_RUNTIME_MISMATCH",
            "environment Python does not match the pinned contract",
        )
    return {
        **before,
        "python_probe_sha256": canonical_sha256(probe),
        "installed_distributions_sha256": probe[
            "installed_distributions_sha256"
        ],
    }


def ensure_environment(
    *,
    profile: IlluminaProfile,
    release: Release | None,
    root: Path | str | None,
    lock: Path | str,
    apply: bool,
    environ: dict[str, str] | None = None,
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
        "release_sha256": (
            release.archive_sha256 if release is not None else None
        ),
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
    if release is None:
        raise MsctlError(
            "ENV_RELEASE_REQUIRED",
            "environment apply requires an authenticated release",
        )
    if (
        release.value.get("provider") != profile.provider
        or release.metadata.get("profile_sha256") != profile.source_sha256
    ):
        raise MsctlError(
            "PROFILE_RELEASE_MISMATCH",
            "environment release does not authenticate the provider profile",
        )
    environment_hashes = release.metadata.get("environment_hashes")
    if (
        not isinstance(environment_hashes, dict)
        or environment_hashes.get(profile.environment_lock) != lock_hash
    ):
        raise MsctlError(
            "ENV_PROVENANCE_MISMATCH",
            "environment lock is not authenticated by the release",
        )
    actual_platform = (
        f"{platform.system().lower()}_{platform.machine().lower()}"
    )
    actual_python = platform.python_version()
    actual_cuda = _detect_cuda_version(environ)
    actual_driver = _detect_cuda_driver(environ)
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
                receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION
                and receipt.get("provider") == profile.provider
                and receipt.get("profile_sha256") == profile.sha256
                and receipt.get("release_sha256") == release.archive_sha256
                and receipt.get("lock_sha256") == lock_hash
                and receipt.get("environment_root")
                == str(destination.absolute())
                and receipt.get("python") == profile.python_version
                and receipt.get("platform") == profile.platform
                and receipt.get("cuda_version") == profile.cuda_version
                and receipt.get("cuda_driver_version") == actual_driver
            ):
                runtime = _runtime_receipt_fields(
                    destination.absolute(),
                    profile=profile,
                )
                if any(
                    receipt.get(field) != value
                    for field, value in runtime.items()
                ):
                    raise MsctlError(
                        "ENV_RUNTIME_MISMATCH",
                        "existing environment does not match its receipt",
                    )
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
        runtime = _runtime_receipt_fields(temporary, profile=profile)
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "provider": profile.provider,
            "profile_sha256": profile.sha256,
            "release_sha256": release.archive_sha256,
            "lock_sha256": lock_hash,
            "environment_root": str(destination.absolute()),
            "python": profile.python_version,
            "platform": profile.platform,
            "cuda_version": profile.cuda_version,
            "cuda_driver_version": actual_driver,
            **runtime,
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
