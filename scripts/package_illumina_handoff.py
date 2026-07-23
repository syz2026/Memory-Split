#!/usr/bin/env python3
"""Build a deterministic, closed-allowlist Illumina handoff ZIP."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

PROVIDER = "illumina-usfc-prd"
NORMALIZED_TIME = (1980, 1, 1, 0, 0, 0)
PLANNED_DIRECTORIES = (
    ".cursor/skills/memorysplit-cluster/",
    "cluster/profiles/",
    "cluster/slurm/",
    "configs/",
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
    "msctl/__init__.py",
    "msctl/__main__.py",
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
    "configs": {".json", ".tsv", ".yaml", ".yml"},
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
    "AWS-P5-START.md",
    "DATASET-POINTER-AWS.json",
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
_PRIVATE_KEY_PATTERN = re.compile(
    rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY(?: BLOCK)?-----"
)
_AWS_ACCESS_KEY_PATTERN = re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
_SERVICE_TOKEN_PATTERN = re.compile(
    rb"\b(?:hf_[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{16,}|"
    rb"github_pat_[A-Za-z0-9_]{16,}|glpat-[A-Za-z0-9_-]{16,}|"
    rb"npm_[A-Za-z0-9]{16,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}|"
    rb"sk-ant-[A-Za-z0-9_-]{20,})\b"
)
_ASSIGNMENT_NAME_PATTERN = (
    rb"(?:AWS_(?:ACCESS_KEY_ID|SECRET_ACCESS_KEY|SESSION_TOKEN)|"
    rb"(?:HF|HUGGINGFACE|GITHUB|GH)_(?:TOKEN|API_KEY|API_TOKEN|PAT)|"
    rb"(?:OPENAI|ANTHROPIC|WANDB)_(?:API_KEY|TOKEN)|"
    rb"SLACK_(?:TOKEN|BOT_TOKEN)|"
    rb"(?:API|ACCESS|AUTH)_(?:KEY|TOKEN|SECRET)|"
    rb"CLIENT_SECRET|PRIVATE_KEY|PASSWORD)"
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    rb"(?i)\b"
    + _ASSIGNMENT_NAME_PATTERN
    + rb"\s*=\s*(?:[\"'][^\"'\r\n]{8,}[\"']|[^\s#;\"']{8,})"
)
_GENERIC_SECRET_ASSIGNMENT_PATTERN = re.compile(
    rb"\b(?:TOKEN|SECRET|PASSWORD|"
    rb"[A-Z][A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY|PRIVATE_KEY))\s*=\s*"
    rb"(?:[\"'][^\"'\r\n]{8,}[\"']|[^\s#;\"']{8,})"
)
_BEARER_TOKEN_PATTERN = re.compile(
    rb"(?i)\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._~+/-]{16,}"
)
_SECRET_PATTERNS = (
    _PRIVATE_KEY_PATTERN,
    _AWS_ACCESS_KEY_PATTERN,
    _SERVICE_TOKEN_PATTERN,
    _SECRET_ASSIGNMENT_PATTERN,
    _GENERIC_SECRET_ASSIGNMENT_PATTERN,
    _BEARER_TOKEN_PATTERN,
)


class PackageError(ValueError):
    """A fail-closed release validation error."""

    def __init__(self, message: str, *, code: str = "PACKAGE_REJECTED") -> None:
        super().__init__(message)
        self.code = code


class _HelpRequested(Exception):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.text = text


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise PackageError(message, code="CLI_USAGE")

    def print_help(self, file=None) -> None:
        raise _HelpRequested(self.format_help())

    def exit(self, status: int = 0, message: str | None = None) -> None:
        if status == 0:
            raise _HelpRequested(message or self.format_help())
        raise PackageError(
            (message or "invalid arguments").strip(),
            code="CLI_USAGE",
        )


@dataclass(frozen=True)
class ReleaseArtifacts:
    release_dir: Path
    archive: Path
    sha256_file: Path
    release: Path
    release_id: str
    sha256: str
    published: bool


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


def _tracked_files(root: Path, revision: str) -> list[_Tracked]:
    output = _run_git(
        root,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        revision,
    )
    result = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        try:
            metadata, encoded_path = raw.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split()
            path = encoded_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise PackageError("Git tree contains an unsupported path") from error
        if object_type != "blob":
            raise PackageError("Git tree contains an unsupported object")
        portable = PurePosixPath(path)
        if (
            portable.is_absolute()
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise PackageError("Git tree contains an unsafe path")
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


def _read_blob(root: Path, object_id: str) -> bytes:
    if re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", object_id) is None:
        raise PackageError("Git tree contains an invalid blob ID")
    return _run_git(root, "cat-file", "blob", object_id)


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


def _atomic_write(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
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
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _collect_payload(
    source: Path,
    tracked: list[_Tracked],
    revision: str,
) -> tuple[dict[str, bytes], str]:
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
        data = _read_blob(source, item.object_id)
        _scan_secret(item.path, data)
        digest = _sha256(data)
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
            "members": member_rows,
        }
    )
    sums = "".join(
        f"{_sha256(payload[name])}  {name}\n" for name in sorted(payload)
    ).encode("utf-8")
    payload["SHA256SUMS"] = sums
    return payload, _sha256(sums)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise PackageError("private release staging contains an unsafe entry")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directory_fd = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any existing entry."""

    libc = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(encoded_source, encoded_destination, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            -100,
            encoded_source,
            -100,
            encoded_destination,
            0x00000001,
        )
    else:
        raise PackageError(
            "platform lacks atomic no-replace directory publication"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise PackageError(
            "release directory already exists",
            code="RELEASE_EXISTS",
        )
    raise PackageError("atomic release publication failed")


def _verify_staged_release(
    release_file: Path,
    *,
    expected_release_id: str,
    expected_archive_sha256: str,
) -> None:
    try:
        from msctl.contracts import load_release

        verified = load_release(release_file)
    except Exception as error:
        raise PackageError(
            f"staged release checksum verification failed: {error}"
        ) from error
    if (
        verified.release_id != expected_release_id
        or verified.archive_sha256 != expected_archive_sha256
    ):
        raise PackageError("staged release identity verification failed")


def _build_staging(
    staging: Path,
    *,
    payload: dict[str, bytes],
    members_sha256: str,
    revision: str,
    release_id: str,
    archive_name: str,
) -> tuple[str, dict[str, object]]:
    archive = staging / archive_name
    _write_zip(
        archive,
        payload=payload,
        directories=PLANNED_DIRECTORIES,
    )
    archive_data = archive.read_bytes()
    archive_hash = hashlib.sha256(archive_data).hexdigest()
    release_value = {
        "schema_version": 1,
        "release_id": release_id,
        "provider": PROVIDER,
        "archive": {
            "path": archive_name,
            "sha256": archive_hash,
            "bytes": len(archive_data),
        },
        "source": {"commit": revision, "dirty": False},
        "members_sha256": members_sha256,
    }
    _atomic_write(
        staging / f"{archive_name}.sha256",
        f"{archive_hash}  {archive_name}\n".encode("ascii"),
    )
    release_file = staging / "RELEASE.json"
    _atomic_write(release_file, _canonical_pretty(release_value))
    _verify_staged_release(
        release_file,
        expected_release_id=release_id,
        expected_archive_sha256=archive_hash,
    )
    _fsync_tree(staging)
    return archive_hash, release_value


def build_handoff(
    *,
    source_root: Path | str,
    out_dir: Path | str,
    apply: bool = False,
) -> ReleaseArtifacts:
    """Build privately; publish one no-replace release directory only on apply."""

    source = Path(source_root)
    if source.is_symlink() or not source.is_dir():
        raise PackageError("source root must be a regular directory")
    revision = _clean_revision(source)
    tracked = _tracked_files(source, revision)
    payload, members_sha256 = _collect_payload(source, tracked, revision)
    release_suffix = members_sha256[:16]
    release_id = f"r1-{release_suffix}"
    archive_name = f"ms-illumina-r1-{release_suffix}.zip"
    output = Path(out_dir)
    release_dir = output / release_id
    staging: Path
    if apply:
        if output.is_symlink():
            raise PackageError("release output must not be a symlink")
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not output.is_dir():
            raise PackageError("release output must be a directory")
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{release_id}.",
                suffix=".staging",
                dir=output,
            )
        )
    else:
        staging = Path(
            tempfile.mkdtemp(prefix=f".{release_id}.dry-run-")
        ).resolve()
    try:
        os.chmod(staging, 0o700)
        archive_hash, _ = _build_staging(
            staging,
            payload=payload,
            members_sha256=members_sha256,
            revision=revision,
            release_id=release_id,
            archive_name=archive_name,
        )
        if apply:
            _rename_noreplace(staging, release_dir)
            output_fd = os.open(
                output,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                os.fsync(output_fd)
            finally:
                os.close(output_fd)
        return ReleaseArtifacts(
            release_dir=release_dir,
            archive=release_dir / archive_name,
            sha256_file=release_dir / f"{archive_name}.sha256",
            release=release_dir / "RELEASE.json",
            release_id=release_id,
            sha256=archive_hash,
            published=apply,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)


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
    apply = False
    try:
        parser = _JsonArgumentParser(
            description="Build a deterministic Illumina MemorySplit handoff."
        )
        parser.add_argument(
            "--source-root",
            default=str(Path(__file__).resolve().parents[1]),
        )
        parser.add_argument("--out-dir", default="dist")
        parser.add_argument("--apply", action="store_true")
        args = parser.parse_args(argv)
        apply = bool(args.apply)
        artifacts = build_handoff(
            source_root=args.source_root,
            out_dir=args.out_dir,
            apply=args.apply,
        )
        report = {
            "schema_version": 1,
            "ok": True,
            "dry_run": not args.apply,
            "published": artifacts.published,
            "release_id": artifacts.release_id,
            "release_dir": str(artifacts.release_dir),
            "archive": str(artifacts.archive),
            "sha256_file": str(artifacts.sha256_file),
            "release": str(artifacts.release),
            "sha256": artifacts.sha256,
        }
        code = 0
    except _HelpRequested as help_request:
        report = {
            "schema_version": 1,
            "ok": True,
            "dry_run": True,
            "published": False,
            "help": help_request.text,
        }
        code = 0
    except PackageError as error:
        report = {
            "schema_version": 1,
            "ok": False,
            "dry_run": not apply,
            "published": False,
            "error": {
                "code": error.code,
                "message": str(error),
            },
        }
        code = 2
    except Exception:
        report = {
            "schema_version": 1,
            "ok": False,
            "dry_run": not apply,
            "published": False,
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
