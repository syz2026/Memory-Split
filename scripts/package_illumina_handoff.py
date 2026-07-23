#!/usr/bin/env python3
"""Build a deterministic, closed-allowlist Illumina handoff ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_SOURCE_ROOT = str(Path(__file__).resolve().parents[1])
if _SOURCE_ROOT not in sys.path:
    sys.path.insert(0, _SOURCE_ROOT)

from msctl.cohort import load_cohort_assignment
from msctl.errors import MsctlError


PROVIDER = "illumina-usfc-prd"
COHORT_ASSIGNMENT = "configs/cohort-assignment-v2.json"
ILLUMINA_RUN_CONFIGS = frozenset(
    {
        "configs/360m-v2/dense-s0.yaml",
        "configs/360m-v2/split90-s0.yaml",
    }
)
NORMALIZED_TIME = (1980, 1, 1, 0, 0, 0)
PLANNED_DIRECTORIES = (
    ".cursor/skills/memorysplit-cluster/",
    "cluster/profiles/",
    "cluster/slurm/",
    "configs/",
    "configs/360m-v2/",
    "corpusgen/parallel/",
    "corpusgen/reasoning/",
    "evals/confirmatory/",
    "fixtures/current-smoke/",
    "msctl/",
    "scripts/",
    "sources/",
    "tests/",
    "train/",
    "vendor/tiktoken/",
)
REQUIRED_MEMBERS = {
    ".cursor/skills/memorysplit-cluster/SKILL.md",
    "AGENT-START.md",
    "DATASET-POINTER.json",
    "cluster/profiles/illumina-usfc-prd.json",
    "cluster/slurm/v2_evaluate.sbatch",
    "cluster/slurm/v2_seed0.sbatch",
    COHORT_ASSIGNMENT,
    *ILLUMINA_RUN_CONFIGS,
    "msctl/__init__.py",
    "msctl/__main__.py",
    "msctl/cohort.py",
    "scripts/package_illumina_handoff.py",
    "tests/test_msctl.py",
    "tests/test_package_illumina_handoff.py",
}
_ROOT_INCLUDED = {
    "AGENT-START.md",
    "DATASET-POINTER.json",
    "LICENSE",
    "LICENSE.txt",
    "pyproject.toml",
    "pytest.ini",
    "requirements.txt",
    "requirements-illumina.lock",
}
_INCLUDED_SUFFIXES = {
    "corpusgen": {".py"},
    "evals": {".py"},
    "msctl": {".py"},
    "organizer": {".py"},
    "scripts": {".py"},
    "sources": {".json", ".txt"},
    "tests": {".bin", ".json", ".jsonl", ".md", ".py", ".txt", ".yaml", ".yml"},
    "train": {".py"},
    "vendor": None,
}
_KNOWN_EXCLUDED_ROOT_FILES = {
    ".gitignore",
    "HANDOFF-AGENT.md",
    "Memory-split-design.md",
    "README.md",
}
_KNOWN_EXCLUDED_TOP_LEVEL = {
    ".cache",
    ".github",
    ".superpowers",
    "artifacts",
    "data",
    "dist",
    "docs",
    "logs",
    "outputs",
    "paper",
    "wandb",
}
_DISPOSABLE_COMPONENTS = {
    ".cache",
    ".pytest_cache",
    ".tiktoken_cache",
    "__pycache__",
    "cache",
    "caches",
    "checkpoints",
    "logs",
    "snapshots",
}
_SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\b(?:hf|ghp)_[A-Za-z0-9_-]{16,}\b"),
    re.compile(
        rb"(?i)\b(?:AWS_SECRET_ACCESS_KEY|API[_-]?SECRET|PASSWORD)"
        rb"\s*=\s*[\"'][^\"'\r\n]{8,}[\"']"
    ),
)


class PackageError(ValueError):
    """A fail-closed release validation error."""


@dataclass(frozen=True)
class ReleaseArtifacts:
    archive: Path
    sha256_file: Path
    release: Path
    release_id: str
    sha256: str


@dataclass(frozen=True)
class _Tracked:
    path: str
    mode: str
    object_id: str


def _run_git(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PackageError("git repository verification failed")
    return completed.stdout


def _clean_revision(root: Path) -> str:
    status_output = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    if status_output:
        raise PackageError("dirty Git trees cannot produce a release")
    revision = _run_git(root, "rev-parse", "--verify", "HEAD").decode().strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise PackageError("Git HEAD is not a full commit ID")
    return revision


def _tracked_files(root: Path) -> list[_Tracked]:
    output = _run_git(root, "ls-files", "-s", "-z")
    result = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        try:
            metadata, encoded_path = raw.split(b"\t", 1)
            mode, object_id, stage = metadata.decode("ascii").split()
            path = encoded_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise PackageError("Git index contains an unsupported path") from error
        if stage != "0":
            raise PackageError("Git index contains an unresolved stage")
        portable = PurePosixPath(path)
        if (
            portable.is_absolute()
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise PackageError("Git index contains an unsafe path")
        if mode == "120000":
            raise PackageError(f"tracked symlink is forbidden: {path}")
        if mode not in {"100644", "100755"}:
            raise PackageError(f"unsupported tracked file mode: {path}")
        result.append(_Tracked(path=path, mode=mode, object_id=object_id))
    return sorted(result, key=lambda item: item.path)


def _classification(path: str) -> str:
    parts = PurePosixPath(path).parts
    if not parts:
        return "unknown"
    if set(parts) & _DISPOSABLE_COMPONENTS:
        return "excluded"
    if parts[0] == "configs":
        suffix = PurePosixPath(path).suffix.lower()
        if len(parts) == 2 and suffix in {".json", ".tsv", ".yaml", ".yml"}:
            return "included"
        if path in ILLUMINA_RUN_CONFIGS:
            return "included"
        if len(parts) >= 3 and parts[1] in {
            "29m",
            "160m",
            "360m",
            "360m-v2",
        }:
            return "excluded"
        return "unknown"
    if path in _ROOT_INCLUDED:
        return "included"
    if path in _KNOWN_EXCLUDED_ROOT_FILES:
        return "excluded"
    if parts[0] in _KNOWN_EXCLUDED_TOP_LEVEL:
        return "excluded"
    if path == ".cursor/skills/memorysplit-cluster/SKILL.md":
        return "included"
    if parts[0] == ".cursor":
        return "unknown"
    if path == "cluster/profiles/illumina-usfc-prd.json":
        return "included"
    if (
        len(parts) == 3
        and parts[0:2] == ("cluster", "slurm")
        and parts[2].startswith("v2_")
        and parts[2].endswith(".sbatch")
    ):
        return "included"
    if parts[0] == "cluster":
        return "excluded"
    if path == "schemas/mit-cluster-profile-v1.schema.json":
        return "excluded"
    if parts[0] == "schemas":
        return "unknown"
    if (
        len(parts) >= 3
        and parts[0:2] == ("fixtures", "current-smoke")
        and PurePosixPath(path).suffix.lower() in {".bin", ".json", ".jsonl"}
    ):
        return "included"
    if parts[0] == "fixtures":
        return "unknown"
    suffixes = _INCLUDED_SUFFIXES.get(parts[0])
    if parts[0] in _INCLUDED_SUFFIXES:
        if suffixes is None or PurePosixPath(path).suffix.lower() in suffixes:
            return "included"
        return "unknown"
    if (
        len(parts) == 1
        and parts[0].startswith("requirements")
        and parts[0].endswith((".txt", ".lock"))
    ):
        return "included"
    return "unknown"


def _read_member(root: Path, relative: str) -> bytes:
    root_resolved = root.resolve()
    candidate = root
    for part in PurePosixPath(relative).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise PackageError(f"symlink traversal is forbidden: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (FileNotFoundError, ValueError) as error:
        raise PackageError(f"tracked member is missing or unsafe: {relative}") from error
    if not resolved.is_file():
        raise PackageError(f"tracked member is not a regular file: {relative}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(candidate, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise PackageError(
                f"tracked member is not a regular file: {relative}"
            )
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            return handle.read()
    except OSError as error:
        raise PackageError(f"tracked member cannot be read safely: {relative}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _scan_secret(path: str, data: bytes) -> None:
    lower_name = PurePosixPath(path).name.lower()
    if (
        lower_name == ".env"
        or lower_name.startswith("id_rsa")
        or lower_name.endswith((".key", ".pem", ".p12"))
        or "credential" in lower_name
    ):
        raise PackageError(f"secret-like file is forbidden: {path}")
    if any(pattern.search(data) for pattern in _SECRET_PATTERNS):
        raise PackageError(f"secret material detected in: {path}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_pretty(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_new_file(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise PackageError(f"release path already exists: {path.name}") from error
    except OSError as error:
        raise PackageError(f"cannot stage release file: {path.name}") from error


def _zip_info(name: str, *, directory: bool) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=NORMALIZED_TIME)
    info.create_system = 3
    info.compress_type = (
        zipfile.ZIP_STORED if directory else zipfile.ZIP_DEFLATED
    )
    mode = (
        stat.S_IFDIR | 0o755
        if directory
        else stat.S_IFREG
        | (0o755 if name.endswith((".sbatch", ".sh")) else 0o644)
    )
    info.external_attr = mode << 16
    if directory:
        info.external_attr |= 0x10
    info.flag_bits = 0
    return info


def _write_zip(
    path: Path,
    *,
    payload: dict[str, bytes],
    directories: tuple[str, ...],
) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x+b") as handle:
            os.fchmod(handle.fileno(), 0o644)
            with zipfile.ZipFile(
                handle,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
                strict_timestamps=True,
            ) as archive:
                archive.comment = b""
                entries = [(name, True) for name in directories]
                entries.extend((name, False) for name in payload)
                for name, directory in sorted(entries):
                    info = _zip_info(name, directory=directory)
                    archive.writestr(
                        info,
                        b"" if directory else payload[name],
                        compress_type=info.compress_type,
                        compresslevel=9 if not directory else None,
                    )
            handle.flush()
            os.fsync(handle.fileno())
            size = os.fstat(handle.fileno()).st_size
            handle.seek(0)
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            return digest.hexdigest(), size
    except FileExistsError as error:
        raise PackageError(f"release path already exists: {path.name}") from error
    except OSError as error:
        raise PackageError("cannot construct staged release archive") from error


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _publish_staging_set(
    staged: tuple[tuple[Path, Path], ...],
    *,
    output: Path,
) -> None:
    existing = [
        destination.name
        for _, destination in staged
        if _path_exists(destination)
    ]
    if existing:
        raise PackageError(f"release path already exists: {sorted(existing)[0]}")

    created: list[tuple[Path, Path]] = []
    try:
        for source, destination in staged:
            try:
                os.link(source, destination, follow_symlinks=False)
            except FileExistsError as error:
                raise PackageError(
                    f"release path already exists: {destination.name}"
                ) from error
            created.append((source, destination))
        directory_fd = os.open(output, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception as error:
        for source, destination in reversed(created):
            try:
                source_stat = source.stat(follow_symlinks=False)
                destination_stat = destination.stat(follow_symlinks=False)
                if os.path.samestat(source_stat, destination_stat):
                    destination.unlink()
            except FileNotFoundError:
                pass
        if isinstance(error, PackageError):
            raise
        raise PackageError("failed to publish complete release set") from error


def _collect_payload(
    source: Path,
    tracked: list[_Tracked],
    revision: str,
) -> tuple[dict[str, bytes], str]:
    try:
        cohort = load_cohort_assignment(source / COHORT_ASSIGNMENT)
        provider_configs = cohort.configs_for_provider(PROVIDER)
    except MsctlError as error:
        raise PackageError(f"invalid cohort assignment: {error.message}") from error
    selected_paths = {config.path for config in provider_configs}
    selected_cells = {
        (config.seed, config.condition) for config in provider_configs
    }
    if (
        selected_paths != ILLUMINA_RUN_CONFIGS
        or selected_cells != {(0, "dense"), (0, "split90")}
    ):
        raise PackageError("Illumina cohort must contain only seed zero pair")
    expected_config_hashes = {
        config.path: config.sha256 for config in provider_configs
    }

    included: list[_Tracked] = []
    unknown = []
    for item in tracked:
        classification = _classification(item.path)
        if classification == "included":
            included.append(item)
        elif classification == "unknown":
            unknown.append(item.path)
    if unknown:
        raise PackageError(f"unknown tracked path is not allowlisted: {unknown[0]}")
    paths = {item.path for item in included}
    missing = sorted(REQUIRED_MEMBERS - paths)
    if missing:
        raise PackageError(f"required release member is not tracked: {missing[0]}")

    payload: dict[str, bytes] = {}
    member_rows = []
    environment_hashes: dict[str, str] = {}
    for item in included:
        data = _read_member(source, item.path)
        _scan_secret(item.path, data)
        digest = _sha256(data)
        if (
            item.path == COHORT_ASSIGNMENT
            and digest != cohort.assignment_sha256
        ):
            raise PackageError("cohort assignment hash mismatch")
        expected_config_hash = expected_config_hashes.get(item.path)
        if expected_config_hash is not None and digest != expected_config_hash:
            raise PackageError(f"cohort config hash mismatch: {item.path}")
        payload[item.path] = data
        member_rows.append(
            {
                "path": item.path,
                "bytes": len(data),
                "sha256": digest,
                "git_blob": item.object_id,
            }
        )
        if PurePosixPath(item.path).name.startswith("requirements"):
            environment_hashes[item.path] = digest
    profile_hash = _sha256(
        payload["cluster/profiles/illumina-usfc-prd.json"]
    )
    payload["RELEASE-METADATA.json"] = _canonical_pretty(
        {
            "schema_version": 1,
            "provider": PROVIDER,
            "source": {"commit": revision, "dirty": False},
            "profile_sha256": profile_hash,
            "environment_hashes": environment_hashes,
            "seed_assignment": {
                "cohort_id": cohort.cohort_id,
                "provider": PROVIDER,
                "seeds": list(cohort.illumina_seeds),
                "arms": ["dense", "split90"],
            },
            "members": member_rows,
        }
    )
    sums = "".join(
        f"{_sha256(payload[name])}  {name}\n" for name in sorted(payload)
    ).encode("utf-8")
    payload["SHA256SUMS"] = sums
    return payload, _sha256(sums)


def build_handoff(
    *,
    source_root: Path | str,
    out_dir: Path | str,
) -> ReleaseArtifacts:
    """Validate a clean source tree and atomically publish release artifacts."""

    source = Path(source_root)
    if source.is_symlink() or not source.is_dir():
        raise PackageError("source root must be a regular directory")
    revision = _clean_revision(source)
    tracked = _tracked_files(source)
    payload, members_sha256 = _collect_payload(source, tracked, revision)
    release_suffix = members_sha256[:16]
    release_id = f"r1-{release_suffix}"
    archive_name = f"ms-illumina-r1-{release_suffix}.zip"
    output = Path(out_dir)
    if output.is_symlink():
        raise PackageError("output directory must not be a symlink")
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise PackageError("output directory cannot be created") from error
    if not output.is_dir():
        raise PackageError("output path must be a directory")
    archive = output / archive_name
    sha_file = output / f"{archive_name}.sha256"
    release_file = output / "RELEASE.json"
    final_paths = (archive, sha_file, release_file)
    if any(_path_exists(path) for path in final_paths):
        raise PackageError("release path already exists")

    with tempfile.TemporaryDirectory(
        prefix=".ms-illumina-stage-",
        dir=output,
    ) as staging_name:
        staging = Path(staging_name)
        staged_archive = staging / archive.name
        archive_hash, archive_bytes = _write_zip(
            staged_archive,
            payload=payload,
            directories=PLANNED_DIRECTORIES,
        )
        staged_sha = staging / sha_file.name
        _write_new_file(
            staged_sha,
            f"{archive_hash}  {archive_name}\n".encode("ascii"),
        )
        release_value = {
            "schema_version": 1,
            "release_id": release_id,
            "provider": PROVIDER,
            "archive": {
                "path": archive_name,
                "sha256": archive_hash,
                "bytes": archive_bytes,
            },
            "source": {"commit": revision, "dirty": False},
            "members_sha256": members_sha256,
        }
        staged_release = staging / release_file.name
        _write_new_file(staged_release, _canonical_pretty(release_value))
        _publish_staging_set(
            (
                (staged_archive, archive),
                (staged_sha, sha_file),
                (staged_release, release_file),
            ),
            output=output,
        )
    return ReleaseArtifacts(
        archive=archive,
        sha256_file=sha_file,
        release=release_file,
        release_id=release_id,
        sha256=archive_hash,
    )


def _emit(value: dict[str, object]) -> None:
    sys.stdout.write(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic Illumina MemorySplit handoff."
    )
    parser.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--out-dir", default="dist")
    args = parser.parse_args(argv)
    try:
        artifacts = build_handoff(
            source_root=args.source_root,
            out_dir=args.out_dir,
        )
        report = {
            "schema_version": 1,
            "ok": True,
            "release_id": artifacts.release_id,
            "archive": str(artifacts.archive),
            "sha256_file": str(artifacts.sha256_file),
            "release": str(artifacts.release),
            "sha256": artifacts.sha256,
        }
        code = 0
    except PackageError as error:
        report = {
            "schema_version": 1,
            "ok": False,
            "error": {
                "code": "PACKAGE_REJECTED",
                "message": str(error),
            },
        }
        code = 2
    except Exception:
        report = {
            "schema_version": 1,
            "ok": False,
            "error": {
                "code": "PACKAGE_INTERNAL_ERROR",
                "message": "unexpected local packaging failure",
            },
        }
        code = 70
    _emit(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
