#!/usr/bin/env python
"""Offline platform checks for Task-11 relational runs.

The checks inspect only local files, processes, and hardware. They do not
submit jobs, call cloud APIs, install packages, or make network requests.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform as runtime_platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment.artifacts import sha256_file


_EXACT_REQUIREMENT_RE = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.!+_-]*)"
)
_HASH_RE = re.compile(r"--hash=sha256:([0-9a-f]{64})")
_ALLOWED_TARGETS = {
    "macos-arm64": "macos-arm64-py312.lock",
    "linux-x86_64-cuda": "linux-x86_64-cuda-py312.lock",
}
_TARGET_ACCELERATORS = {
    "macos-arm64": "mps",
    "linux-x86_64-cuda": "cu130",
}
_IMPORTS = {
    "datasets": "datasets",
    "huggingface-hub": "huggingface_hub",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pytest": "pytest",
    "pyyaml": "yaml",
    "tiktoken": "tiktoken",
    "torch": "torch",
    "tqdm": "tqdm",
}
_MIN_AWS_MEMORY_BYTES = 128 * 1024**3
_MIN_AWS_DISK_BYTES = 1024**4
_MIN_H100_MEMORY_MIB = 80_000


class LockValidationError(ValueError):
    """Raised when a dependency input is not exact and platform-bound."""


class PreflightError(ValueError):
    """Raised when a local platform is unsafe to launch."""


def _normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _read_regular_text(path: str | Path, *, name: str) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise LockValidationError(f"{name} must be a regular non-symlink file")
    try:
        return candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise LockValidationError(f"{name} is not valid UTF-8") from exc


def validate_base_requirements(path: str | Path) -> dict[str, str]:
    """Return exact direct requirements, rejecting all resolver latitude."""

    content = _read_regular_text(path, name="base requirements")
    requirements: dict[str, str] = {}
    for number, raw in enumerate(content.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ";" in line or " @ " in line:
            raise LockValidationError(
                f"base requirements line {number} is conditional or indirect"
            )
        match = _EXACT_REQUIREMENT_RE.fullmatch(line)
        if match is None:
            raise LockValidationError(
                f"base requirements line {number} is not an exact pin"
            )
        project = _normalized_name(match.group(1))
        if project in requirements:
            raise LockValidationError(f"duplicate base requirement: {project}")
        requirements[project] = match.group(2)
    if not requirements:
        raise LockValidationError("base requirements cannot be empty")
    return dict(sorted(requirements.items()))


def _lock_metadata(content: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw in content.splitlines():
        match = re.fullmatch(r"# ([a-z-]+): (.+)", raw)
        if match is not None:
            key, value = match.groups()
            if key in metadata:
                raise LockValidationError(
                    f"duplicate dependency lock metadata: {key}"
                )
            metadata[key] = value
    return metadata


def _logical_lock_entries(content: str) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    current: list[str] = []
    start = 0
    for number, raw in enumerate(content.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("--"):
            if stripped.startswith("--hash="):
                if not current:
                    raise LockValidationError(
                        f"orphaned lock hash on line {number}"
                    )
            elif current:
                raise LockValidationError(
                    f"lock option interrupts requirement on line {number}"
                )
            else:
                if not re.fullmatch(
                    r"--index-url https://pypi\.org/simple",
                    stripped,
                ):
                    raise LockValidationError(
                        f"unsupported lock option on line {number}"
                    )
                continue
        if not current:
            start = number
        current.append(stripped.removesuffix("\\").strip())
        if not stripped.endswith("\\"):
            entries.append((start, " ".join(current)))
            current = []
    if current:
        raise LockValidationError("lock ends with an incomplete requirement")
    return entries


def validate_lock_file(
    path: str | Path,
    *,
    expected_platform: str | None = None,
) -> dict[str, str]:
    """Validate one fully resolved Python-3.12 hash lock."""

    content = _read_regular_text(path, name="dependency lock")
    metadata = _lock_metadata(content)
    if set(metadata) != {
        "accelerator",
        "lock-format",
        "python-version",
        "target-platform",
    }:
        raise LockValidationError("dependency lock metadata fields are not exact")
    target = metadata.get("target-platform")
    if target not in _ALLOWED_TARGETS:
        raise LockValidationError("dependency lock target platform is unsupported")
    if expected_platform is not None and target != expected_platform:
        raise LockValidationError(
            f"dependency lock targets {target}, not {expected_platform}"
        )
    if metadata.get("python-version") != "3.12":
        raise LockValidationError("dependency lock must target Python 3.12")
    if metadata.get("lock-format") != "uv-requirements-v1":
        raise LockValidationError("dependency lock format metadata is missing")
    if metadata.get("accelerator") != _TARGET_ACCELERATORS[target]:
        raise LockValidationError(
            "dependency lock accelerator disagrees with target platform"
        )
    if content.splitlines().count("--index-url https://pypi.org/simple") > 1:
        raise LockValidationError(
            "dependency lock declares an ambiguous package index"
        )
    expected_name = _ALLOWED_TARGETS[target]
    if Path(path).name != expected_name:
        raise LockValidationError(
            "dependency lock filename is ambiguous for its target platform"
        )

    requirements: dict[str, str] = {}
    for number, entry in _logical_lock_entries(content):
        if ";" in entry or " @ " in entry:
            raise LockValidationError(
                f"lock requirement on line {number} is conditional or indirect"
            )
        hashes = _HASH_RE.findall(entry)
        without_hashes = _HASH_RE.sub("", entry)
        match = _EXACT_REQUIREMENT_RE.fullmatch(without_hashes.strip())
        if match is None:
            raise LockValidationError(
                f"lock requirement on line {number} is not one exact pin"
            )
        if not hashes:
            raise LockValidationError(
                f"lock requirement on line {number} has no SHA-256"
            )
        project = _normalized_name(match.group(1))
        if project in requirements:
            raise LockValidationError(f"duplicate locked requirement: {project}")
        requirements[project] = match.group(2)
    if not requirements:
        raise LockValidationError("dependency lock cannot be empty")
    return dict(sorted(requirements.items()))


def lock_for_runtime(
    requirements_root: str | Path,
    *,
    python_version: tuple[int, ...],
    system: str,
    machine: str,
) -> Path:
    if tuple(python_version[:2]) != (3, 12):
        raise LockValidationError("Task 11 requires Python 3.12 exactly")
    architecture = machine.lower()
    if system == "Darwin" and architecture in {"arm64", "aarch64"}:
        target = "macos-arm64"
    elif system == "Linux" and architecture in {"x86_64", "amd64"}:
        target = "linux-x86_64-cuda"
    else:
        raise LockValidationError(
            f"unsupported Task-11 platform: {system}/{architecture}"
        )
    lock = Path(requirements_root) / _ALLOWED_TARGETS[target]
    validate_lock_file(lock, expected_platform=target)
    return lock


def _required_import_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for project, module in _IMPORTS.items():
        try:
            importlib.import_module(module)
            versions[project] = importlib.metadata.version(project)
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            raise PreflightError(
                f"required dependency cannot be imported: {project}"
            ) from exc
    return dict(sorted(versions.items()))


def _verify_bundle(path: Path, **kwargs: Any) -> dict[str, Any]:
    from scripts.verify_relational_bundle import verify_bundle

    return verify_bundle(path, **kwargs)


def _run_smoke(path: Path, **kwargs: Any) -> dict[str, Any]:
    from scripts.relational_smoke_test import run_smoke

    return run_smoke(path, **kwargs)


def _validate_root(
    path: str | Path,
    *,
    name: str,
    disk_usage: Callable[[str | Path], Any],
    minimum_free: int,
) -> tuple[Path, int]:
    root = Path(path)
    absolute = root if root.is_absolute() else Path.cwd() / root
    if root.is_symlink() or not root.is_dir():
        raise PreflightError(f"{name} must be an existing regular directory")
    if root.resolve(strict=True) != absolute:
        raise PreflightError(f"{name} is not canonical or traverses a symlink")
    if not os.access(root, os.R_OK | os.W_OK | os.X_OK):
        raise PreflightError(f"{name} is not readable and writable")
    free = int(disk_usage(root).free)
    if free < minimum_free:
        raise PreflightError(f"{name} has insufficient free disk space")
    return absolute, free


def _validate_external_assets(
    data_root: str | Path,
    assets: object,
) -> int:
    if not isinstance(assets, list) or not assets:
        raise PreflightError("bundle external asset inventory is invalid")
    root = Path(data_root)
    absolute = root if root.is_absolute() else Path.cwd() / root
    if (
        root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != absolute
    ):
        raise PreflightError(
            "external asset root must be a canonical regular directory"
        )
    verified_builds: dict[Path, str] = {}
    for item in assets:
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {
                "kind",
                "path",
                "commitment_sha256",
                "sha256",
                "bytes",
                "build_manifest_sha256",
            }
            or item["kind"] not in {"corpus", "weights"}
            or not isinstance(item["path"], str)
            or any(
                not isinstance(item[field], str)
                or re.fullmatch(r"[0-9a-f]{64}", item[field]) is None
                for field in (
                    "commitment_sha256",
                    "sha256",
                    "build_manifest_sha256",
                )
            )
            or isinstance(item["bytes"], bool)
            or not isinstance(item["bytes"], int)
            or item["bytes"] < 1
        ):
            raise PreflightError("bundle external asset entry is invalid")
        relative = PurePosixPath(item["path"])
        if (
            relative.is_absolute()
            or "\\" in item["path"]
            or relative.as_posix() != item["path"]
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise PreflightError("bundle external asset path is not portable")
        candidate = absolute.joinpath(*relative.parts)
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.resolve(strict=True) != candidate
        ):
            raise PreflightError(
                f"external asset is missing or unsafe: {item['path']}"
            )
        try:
            digest = sha256_file(candidate)
        except (OSError, ValueError) as exc:
            raise PreflightError(
                f"external asset cannot be read: {item['path']}"
            ) from exc
        if candidate.stat(follow_symlinks=False).st_size != item["bytes"]:
            raise PreflightError(
                f"external asset byte count mismatch: {item['path']}"
            )
        if digest != item["sha256"]:
            raise PreflightError(
                f"external asset hash mismatch: {item['path']}"
            )
        build_manifest = candidate.parent / "manifest.json"
        expected_build_hash = item["build_manifest_sha256"]
        prior_build_hash = verified_builds.get(build_manifest)
        if (
            prior_build_hash is not None
            and prior_build_hash != expected_build_hash
        ):
            raise PreflightError(
                "external asset inventory has inconsistent build manifest hashes"
            )
        if prior_build_hash is None:
            try:
                actual_build_hash = sha256_file(build_manifest)
            except (OSError, ValueError) as exc:
                raise PreflightError(
                    f"external asset build manifest is missing or unsafe: "
                    f"{item['path']}"
                ) from exc
            if actual_build_hash != expected_build_hash:
                raise PreflightError(
                    f"external asset build manifest hash mismatch: "
                    f"{item['path']}"
                )
            verified_builds[build_manifest] = expected_build_hash
    return len(assets)


def validate_farmshare_batch_script(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise PreflightError("FarmShare batch script is missing")
    content = candidate.read_text(encoding="utf-8")

    gpu = re.findall(
        r"^#SBATCH\s+--gres=gpu:([A-Za-z0-9_-]+):([0-9]+)\s*$",
        content,
        flags=re.MULTILINE,
    )
    nodes = re.findall(
        r"^#SBATCH\s+--nodes=([0-9]+)\s*$",
        content,
        flags=re.MULTILINE,
    )
    tasks = re.findall(
        r"^#SBATCH\s+--ntasks=([0-9]+)\s*$",
        content,
        flags=re.MULTILINE,
    )
    resume = re.findall(r"--resume\s+(auto|none)\b", content)
    if (
        gpu != [("L40S", "1")]
        or nodes != ["1"]
        or tasks != ["1"]
        or resume != ["auto"]
    ):
        raise PreflightError(
            "FarmShare batch must request one L40S on one node/task "
            "and use --resume auto"
        )
    return {
        "gpu_type": "L40S",
        "gpu_count": 1,
        "nodes": 1,
        "tasks": 1,
        "resume": "auto",
    }


def parse_h100_inventory(output: str) -> dict[str, Any]:
    rows: list[tuple[str, int]] = []
    for number, raw in enumerate(output.splitlines(), start=1):
        if not raw.strip():
            continue
        fields = [field.strip() for field in raw.rsplit(",", 1)]
        if len(fields) != 2:
            raise PreflightError(
                f"GPU inventory line {number} is not name,memory"
            )
        try:
            memory = int(fields[1])
        except ValueError as exc:
            raise PreflightError("GPU memory inventory is invalid") from exc
        rows.append((fields[0], memory))
    if not rows:
        raise PreflightError("no visible H100 GPUs")
    names = {name for name, _ in rows}
    if len(names) != 1:
        raise PreflightError("GPU inventory is not homogeneous")
    name = rows[0][0]
    if "H100" not in name:
        raise PreflightError("AWS execution requires homogeneous H100 GPUs")
    if any(memory < _MIN_H100_MEMORY_MIB for _, memory in rows):
        raise PreflightError("every H100 must expose at least 80 GB of memory")
    return {
        "count": len(rows),
        "name": name,
        "memory_mib": rows[0][1],
        "homogeneous": True,
    }


def _default_command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PreflightError(f"platform command failed: {command[0]}") from exc
    return completed.stdout


def _default_total_memory() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError) as exc:
        raise PreflightError("system memory cannot be inspected") from exc


def _validate_smoke_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise PreflightError("a passing offline smoke report is required")
    from evals.relational_gates import smoke_report_passes

    try:
        passed = smoke_report_passes(report)
    except ValueError as exc:
        raise PreflightError(f"smoke report is invalid: {exc}") from exc
    if not passed:
        raise PreflightError("smoke report does not match the Task-11 matrix")
    return dict(report)


def run_preflight(
    platform: str,
    *,
    bundle: str | Path,
    requirements_root: str | Path,
    data_root: str | Path | None = None,
    out_root: str | Path | None = None,
    scratch: str | Path | None = None,
    python_version: tuple[int, ...] = tuple(sys.version_info[:3]),
    system: str = runtime_platform.system(),
    machine: str = runtime_platform.machine(),
    run_smoke_check: bool = False,
    smoke_report: Mapping[str, Any] | None = None,
    capacity_mode: str | None = None,
    expected_gpus: int | None = None,
    command_lookup: Callable[[str], str | None] = shutil.which,
    command_output: Callable[[list[str]], str] = _default_command_output,
    total_memory_bytes: int | None = None,
    disk_usage: Callable[[str | Path], Any] = shutil.disk_usage,
) -> dict[str, Any]:
    if platform not in {"local", "farmshare", "aws"}:
        raise PreflightError("platform must be local, farmshare, or aws")
    if platform == "aws":
        if capacity_mode not in {"on-demand", "capacity-block"}:
            raise PreflightError(
                "AWS capacity mode must be on-demand or capacity-block"
            )
        if data_root is None or out_root is None:
            raise PreflightError("AWS data and output roots are required")
    bundle_path = Path(bundle)
    if bundle_path.is_symlink() or not bundle_path.is_file():
        raise PreflightError("bundle must be a regular non-symlink file")

    if platform == "local":
        lock = lock_for_runtime(
            requirements_root,
            python_version=python_version,
            system=system,
            machine=machine,
        )
    else:
        lock = lock_for_runtime(
            requirements_root,
            python_version=python_version,
            system="Linux",
            machine="x86_64",
        )
    locked = validate_lock_file(lock)
    imports = _required_import_versions()
    base = validate_base_requirements(Path(requirements_root) / "base.in")
    for project in base:
        if project not in locked:
            raise PreflightError(f"platform lock omits direct dependency: {project}")
        if imports[project] != base[project]:
            raise PreflightError(
                f"installed dependency differs from exact pin: {project}"
            )

    bundle_report = _verify_bundle(bundle_path)
    checks: dict[str, Any] = {
        "bundle": bundle_report,
        "imports": imports,
        "locked_projects": len(locked),
    }
    if bundle_report.get("external_assets"):
        if data_root is None:
            raise PreflightError(
                "data root is required to hash external bundle assets"
            )
        checks["external_assets"] = _validate_external_assets(
            data_root,
            bundle_report["external_assets"],
        )
    roots: dict[str, Any] = {}
    if data_root is not None:
        _, free = _validate_root(
            data_root,
            name="data root",
            disk_usage=disk_usage,
            minimum_free=1,
        )
        roots["data_writable"] = True
        roots["data_free_bytes"] = free
    if out_root is not None:
        _, free = _validate_root(
            out_root,
            name="output root",
            disk_usage=disk_usage,
            minimum_free=(
                _MIN_AWS_DISK_BYTES if platform == "aws" else 1
            ),
        )
        roots["out_writable"] = True
        roots["out_free_bytes"] = free
    if roots:
        checks["roots"] = roots

    if run_smoke_check:
        with tempfile.TemporaryDirectory(prefix="relational-preflight-") as temp:
            smoke = _run_smoke(
                Path(temp),
                steps=2,
                device="cpu",
                source_root=Path(__file__).resolve().parents[1],
            )
        checks["smoke"] = _validate_smoke_report(smoke)
    elif smoke_report is not None:
        checks["smoke"] = _validate_smoke_report(smoke_report)

    result: dict[str, Any] = {
        "record_type": "relational_platform_preflight",
        "schema_version": 1,
        "platform": platform,
        "passed": True,
        "lock": str(lock),
        "checks": checks,
    }
    if platform == "farmshare":
        missing = [
            command
            for command in ("sbatch", "sinfo", "squeue", "srun")
            if command_lookup(command) is None
        ]
        if missing:
            raise PreflightError(
                "missing FarmShare command(s): " + ", ".join(missing)
            )
        if scratch is None:
            raise PreflightError("FarmShare scratch root is required")
        _, free = _validate_root(
            scratch,
            name="FarmShare scratch",
            disk_usage=disk_usage,
            minimum_free=1,
        )
        slurm = validate_farmshare_batch_script(
            Path(__file__).resolve().parents[1]
            / "cluster"
            / "slurm"
            / "relational_train.sbatch"
        )
        checks["slurm"] = slurm
        checks["scratch_free_bytes"] = free
        if "smoke" not in checks:
            raise PreflightError("FarmShare preflight requires exact-resume smoke")
    elif platform == "aws":
        inventory = parse_h100_inventory(
            command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ]
            )
        )
        if expected_gpus != inventory["count"]:
            raise PreflightError(
                "AWS expected GPU count must equal the visible GPU count"
            )
        memory = (
            _default_total_memory()
            if total_memory_bytes is None
            else int(total_memory_bytes)
        )
        if memory < _MIN_AWS_MEMORY_BYTES:
            raise PreflightError("AWS host memory is below the 128 GiB minimum")
        if "smoke" not in checks:
            raise PreflightError("AWS preflight requires exact-resume smoke")
        checks["gpus"] = inventory
        checks["expected_gpus"] = expected_gpus
        checks["memory_bytes"] = memory
        result["capacity_mode"] = capacity_mode
        result["gpu_count"] = inventory["count"]
    return result


def _load_json(path: str | Path) -> Mapping[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise PreflightError("smoke report must be a regular file")
    try:
        value = json.loads(candidate.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError("smoke report is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise PreflightError("smoke report must be a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("local", "farmshare", "aws"))
    parser.add_argument("--bundle")
    parser.add_argument("--requirements-root")
    parser.add_argument("--data-root")
    parser.add_argument("--out-root")
    parser.add_argument("--scratch")
    parser.add_argument(
        "--capacity-mode",
        choices=("on-demand", "capacity-block"),
    )
    parser.add_argument(
        "--expected-gpus",
        type=int,
        help="visible GPU count expected on an already-provisioned AWS host",
    )
    parser.add_argument("--smoke-report")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--validate-lock")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.validate_lock is not None:
            locked = validate_lock_file(args.validate_lock)
            print(
                json.dumps(
                    {"lock": args.validate_lock, "projects": len(locked)},
                    sort_keys=True,
                )
            )
            return 0
        if args.platform is None or args.bundle is None:
            raise PreflightError("--platform and --bundle are required")
        root = Path(__file__).resolve().parents[1]
        report = run_preflight(
            args.platform,
            bundle=args.bundle,
            requirements_root=(
                Path(args.requirements_root)
                if args.requirements_root
                else root / "requirements"
            ),
            data_root=args.data_root,
            out_root=args.out_root,
            scratch=args.scratch,
            run_smoke_check=not args.skip_smoke and args.smoke_report is None,
            smoke_report=(
                _load_json(args.smoke_report) if args.smoke_report else None
            ),
            capacity_mode=args.capacity_mode,
            expected_gpus=args.expected_gpus,
        )
    except (LockValidationError, PreflightError, ValueError) as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
