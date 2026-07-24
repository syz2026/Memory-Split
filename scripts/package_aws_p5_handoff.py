#!/usr/bin/env python3
"""Build a deterministic, fail-closed AWS P5 MemorySplit handoff."""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Iterator

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msctl.aws_contracts import (
    ARMS,
    AWS_ENVIRONMENT_RECEIPT_V2_FIELDS,
    COHORT_ASSIGNMENT_PATH as COHORT_PATH,
    COHORT_ID,
    CONFIG_ROOT,
    DATASET_POINTER_PATH,
    DATASET_RECEIPT_PATH,
    EXPECTED_CONFIG_PATHS,
    PACKAGE_FORMAT_VERSION,
    PREREGISTRATION_ID,
    PREREGISTRATION_PATH,
    PROFILE_PATH,
    PROVIDER,
    SEEDS as AWS_SEEDS,
    SNAPSHOT_STEPS,
)
if TYPE_CHECKING:
    from msctl.aws_lifecycle import AuthenticatedProviderLifecycle


def admit_provider_lifecycle(**kwargs):
    """Lazily enter selected authority without loading it for legacy CLI use."""

    from msctl.aws_lifecycle import (
        admit_provider_lifecycle as admit_lifecycle,
    )

    return admit_lifecycle(**kwargs)


NORMALIZED_TIME = (1980, 1, 1, 0, 0, 0)
STATIC_ENVIRONMENT_LOCK_PATH = "requirements-aws-p5.lock"
RUNBOOK_PATH = "docs/AWS-P5-360M-RUNBOOK.md"
RELEASE_RECEIPT_NAME = "RELEASE-AWS-P5.json"
CONTAINER_IMAGE_DIGEST_PATTERN = "^sha256:[0-9a-f]{64}$"
EXPECTED_CONFIGS = frozenset(EXPECTED_CONFIG_PATHS)
PROVIDER_BRIDGE_MEMBERS = frozenset(
    {
        "cluster/aws/gpu_profile.py",
        "cluster/aws/qualification.py",
        "cluster/aws/qualification_worker.py",
        "cluster/profiles/aws-p5.48xlarge-v3.json",
        "cluster/profiles/aws-p6-b300.48xlarge-v3.json",
        "configs/aws-hardware-amendment-v3.json",
        "configs/objective-controls-amendment-v3.yaml",
        "configs/29m-v3/manifest.json",
        "configs/29m-v3/full_corpus_dense.yaml",
        "configs/29m-v3/full_corpus_split90.yaml",
        "configs/29m-v3/no_arc_conceptarc_dense.yaml",
        "configs/29m-v3/no_arc_conceptarc_split90.yaml",
        "configs/29m-v3/no_refinement_dense.yaml",
        "configs/29m-v3/no_refinement_split90.yaml",
        "configs/29m-v3/full_corpus_random_fact90.yaml",
        "configs/29m-v3/full_corpus_matched_nonfactual_mask.yaml",
        "containers/aws-gpu/Dockerfile",
        "containers/aws-gpu/Dockerfile.dockerignore",
        "containers/aws-gpu/build_image.py",
        "containers/aws-gpu/host-candidate.json",
        "containers/aws-gpu/inspect_container.py",
        "containers/aws-gpu/requirements.in",
        "containers/aws-gpu/requirements.lock",
        "containers/aws-gpu/runtime_lock.py",
        "msctl/aws_hardware.py",
        "msctl/aws_lifecycle.py",
        "msctl/objective_controls_v3.py",
    }
)
REQUIRED_MEMBERS = EXPECTED_CONFIGS | PROVIDER_BRIDGE_MEMBERS | {
    "AWS-P5-START.md",
    DATASET_POINTER_PATH,
    COHORT_PATH,
    PREREGISTRATION_PATH,
    PROFILE_PATH,
    RUNBOOK_PATH,
    "cluster/aws/p5/attest_environment.py",
    "cluster/aws/p5/bootstrap.sh",
    "cluster/aws/p5/canary.py",
    "cluster/aws/p5/checkpoint_mirror.py",
    "cluster/aws/p5/corpus_contract.py",
    "cluster/aws/p5/interruption_checkpoint.py",
    "cluster/aws/p5/launch_seed_pair.py",
    "cluster/aws/p5/profile.py",
    "corpusgen/parallel/__init__.py",
    "evals/confirmatory/__init__.py",
    "msctl/__init__.py",
    "msctl/__main__.py",
    "msctl/aws_contracts.py",
    "msctl/aws_argv.py",
    "msctl/aws_launch_manifest.py",
    "msctl/aws_p5.py",
    "msctl/aws_resume_launch.py",
    "msctl/contracts.py",
    "msctl/dataset.py",
    "msctl/errors.py",
    "msctl/fsutil.py",
    "msctl/jsonutil.py",
    "msctl/profile.py",
    "msctl/state.py",
    "scripts/build_parallel_corpus.py",
    "scripts/package_aws_p5_handoff.py",
    "scripts/run_train.py",
    "tests/test_package_aws_p5_handoff.py",
    "train/__init__.py",
    "train/data.py",
    "train/model.py",
    "train/safeio.py",
    "train/trainer.py",
}
_ROOT_INCLUDED = {
    "AWS-P5-START.md",
    DATASET_POINTER_PATH,
    "LICENSE",
    "LICENSE.txt",
    "pyproject.toml",
    "pytest.ini",
    "requirements.txt",
}
_ROOT_EXCLUDED = {
    ".gitignore",
    "AGENT-START.md",
    "DATASET-POINTER.json",
    "HANDOFF-AGENT.md",
    "Memory-split-design.md",
    "README.md",
    "requirements-illumina.lock",
}
_EXCLUDED_TOP_LEVEL = {
    ".cursor",
    ".github",
    ".superpowers",
    "artifacts",
    "checkpoints",
    "data",
    "dist",
    "docs",
    "fixtures",
    "logs",
    "outputs",
    "paper",
    "schemas",
    "sealed",
    "wandb",
}
_DISPOSABLE_COMPONENTS = {
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tiktoken_cache",
    "__pycache__",
    "cache",
    "caches",
    "snapshots",
}
_FORBIDDEN_COMPONENT_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"checkpoints?(?:v[0-9]+)?",
        r"credentials?(?:v[0-9]+)?",
        r"data(?:set)?s?(?:v[0-9]+)?",
        r"gold(?:v[0-9]+)?",
        r"logs?(?:v[0-9]+)?",
        r"outputs?(?:v[0-9]+)?",
        r"pass(?:word|phrase)(?:v[0-9]+)?",
        r"privatekeys?(?:v[0-9]+)?",
        r"corpus(?:es)?(?:v[0-9]+)?",
        r"results?(?:v[0-9]+)?",
    )
)
_SHARED_SUFFIXES: dict[str, set[str]] = {
    "corpusgen": {".py"},
    "evals": {".py"},
    "msctl": {".py"},
    "organizer": {".py"},
    "scripts": {".py"},
    "tests": {".py"},
    "train": {".py"},
}
_APPROVED_SOURCES = {
    "sources/Wikidata-CC0-1.0.txt",
    "sources/current-dataset-licenses.json",
    "sources/wikidata5m.lock.json",
}
_APPROVED_VENDOR = {
    "vendor/tiktoken/6c7ea1a7e38e3a7f062df639a5b80947f075ffe6",
    "vendor/tiktoken/6d1cbeee0f20b3d9449abfede4726ed8212e3aee",
}
_SHARED_CONFIGS = {
    "configs/current-dataset-lock.json",
    "configs/reasoning-dataset-v2.json",
    "configs/route-policy.json",
}
_SHARED_TEST_FIXTURES = {
    "tests/fixtures/current_sources/README.md",
    "tests/fixtures/relational-smoke-route-policy.json",
}
_APPROVED_TEST_MODULES = {
    "tests/test_package_aws_p5_handoff.py",
}
_LEGACY_CONFIG_PREFIXES = (
    "configs/29m/",
    "configs/160m/",
    "configs/360m/",
    "configs/360m-v2/",
)
_LEGACY_CONFIG_FILES = {
    "configs/29m.tsv",
    "configs/160m.tsv",
    "configs/360m.tsv",
    "configs/cohort-assignment-v2.json",
    "configs/preregistration-v2.yaml",
}
_PROVIDER_EXCLUDED = {
    "cluster/profiles/aws-p5.48xlarge.json",
    "cluster/profiles/illumina-usfc-prd.json",
    "scripts/package_illumina_handoff.py",
    "tests/test_package_illumina_handoff.py",
}
_PRIVATE_KEY_PATTERN = re.compile(
    rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY"
    rb"(?: BLOCK)?-----"
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
_SENSITIVE_FIELD_EXACT = {
    "credential",
    "credentials",
    "secret",
    "token",
}
_SENSITIVE_FIELD_AFFIXES = (
    "accesskey",
    "accesstoken",
    "apikey",
    "authtoken",
    "clientsecret",
    "credential",
    "credentials",
    "passphrase",
    "password",
    "privatekey",
    "secretkey",
    "sessiontoken",
)
_SENSITIVE_FIELD_VALUE_FORMS = (
    "apitokenvalue",
    "secretvalue",
    "tokenvalue",
)
_TEXT_SUFFIXES = {".json", ".lock", ".md", ".py", ".sh", ".txt", ".yaml", ".yml"}
_OBJECT_ID_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SOURCE_OBJECT_ID_PATTERN = re.compile(r"[0-9a-f]{40}")
_CONFIG_KEYS = {
    "schema_version",
    "cohort_id",
    "run_id",
    "condition",
    "seed",
    "model",
    "ctx",
    "train_corpus",
    "sidecar_name",
    "out_dir",
    "micro_batch_size",
    "tokens_per_step",
    "max_steps",
    "total_tokens",
    "lr",
    "warmup_steps",
    "weight_decay",
    "compile",
    "device",
    "log_every",
    "eval_every",
    "snapshot_steps",
    "ckpt_minutes",
}


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
class _PinnedEntry:
    parent_fd: int
    name: str
    descriptor: int
    kind: str
    sha256: str | None = None


@dataclass(frozen=True)
class _Repository:
    source_fd: int
    source_path: str
    git_dir_fd: int
    git_dir_path: str
    pinned_entries: tuple[_PinnedEntry, ...]
    directory_paths: tuple[tuple[int, str], ...]
    owned_descriptors: tuple[int, ...]

    def close(self) -> None:
        for descriptor in reversed(self.owned_descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


@dataclass(frozen=True)
class _Tracked:
    path: str
    mode: str
    object_id: str


@dataclass(frozen=True)
class _Collected:
    payload: dict[str, bytes]
    modes: dict[str, str]
    tree_id: str
    members_sha256: str
    cohort_sha256: str
    profile_sha256: str
    environment: dict[str, object]
    dataset_pointer_sha256: str
    config_sha256: dict[str, str]
    seed_assignment: dict[str, object]
    provider: str
    profile_id: str
    profile_path: str
    lifecycle_fields: dict[str, object]


@dataclass(frozen=True)
class _Staged:
    archive_fd: int
    archive_sha256: str
    archive_bytes: int
    checksum_bytes: bytes
    release_bytes: bytes


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueSafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise PackageError("YAML mapping contains an invalid key") from error
        if duplicate:
            raise PackageError(f"YAML mapping contains duplicate key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _descriptor_path(descriptor: int) -> str:
    if sys.platform == "darwin":
        try:
            encoded = fcntl.fcntl(descriptor, 50, b"\0" * 1024)
            path = encoded.split(b"\0", 1)[0]
            if path:
                return os.fsdecode(path)
        except OSError as error:
            raise PackageError(
                "platform cannot resolve the pinned source descriptor"
            ) from error
    for prefix in ("/proc/self/fd", "/dev/fd"):
        candidate = f"{prefix}/{descriptor}"
        try:
            return os.readlink(candidate)
        except OSError:
            continue
    raise PackageError("platform cannot expose the pinned source descriptor")


def _assert_directory_descriptor_path(
    descriptor: int,
    path: str,
    *,
    label: str = "directory",
) -> None:
    pinned = os.fstat(descriptor)
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise PackageError(f"pinned {label} path changed") from error
    if (
        not stat.S_ISDIR(current.st_mode)
        or current.st_dev != pinned.st_dev
        or current.st_ino != pinned.st_ino
    ):
        raise PackageError(f"pinned {label} path was replaced")


def _open_existing_directory(path: Path, *, label: str) -> tuple[Path, int]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current_fd = os.open("/", _directory_flags())
    try:
        for component in absolute.parts[1:]:
            try:
                child_fd = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=current_fd,
                )
            except OSError as error:
                raise PackageError(
                    f"{label} must contain no symlink or non-directory component"
                ) from error
            os.close(current_fd)
            current_fd = child_fd
        _assert_directory_descriptor_path(
            current_fd,
            str(absolute),
            label=label,
        )
        return absolute, current_fd
    except Exception:
        os.close(current_fd)
        raise


def _read_pinned_bytes(
    descriptor: int,
    *,
    label: str,
    maximum_bytes: int = 512 * 1024 * 1024,
) -> bytes:
    details = os.fstat(descriptor)
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise PackageError(f"{label} must be a singly-linked regular file")
    if details.st_size < 0 or details.st_size > maximum_bytes:
        raise PackageError(f"{label} has an unsupported size")
    try:
        data = os.pread(descriptor, details.st_size, 0)
    except OSError as error:
        raise PackageError(f"{label} cannot be read safely") from error
    if len(data) != details.st_size:
        raise PackageError(f"{label} changed while being read")
    return data


def _open_entry_at(parent_fd: int, name: str, *, label: str) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise PackageError(f"{label} cannot be opened safely") from error


def _pin_entry(
    parent_fd: int,
    name: str,
    *,
    label: str,
    expected_kind: str,
) -> _PinnedEntry:
    descriptor = _open_entry_at(parent_fd, name, label=label)
    try:
        details = os.fstat(descriptor)
        if expected_kind == "directory":
            if not stat.S_ISDIR(details.st_mode):
                raise PackageError(f"{label} must be a directory")
            digest = None
        elif expected_kind == "file":
            data = _read_pinned_bytes(descriptor, label=label)
            digest = _sha256(data)
        else:
            raise AssertionError(f"unsupported pinned entry kind: {expected_kind}")
        return _PinnedEntry(
            parent_fd=parent_fd,
            name=name,
            descriptor=descriptor,
            kind=expected_kind,
            sha256=digest,
        )
    except Exception:
        os.close(descriptor)
        raise


def _parse_gitdir_file(data: bytes, source_path: Path) -> Path:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PackageError("worktree .git file is not UTF-8") from error
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0].startswith("gitdir: "):
        raise PackageError("worktree .git file has an invalid format")
    raw_path = lines[0][len("gitdir: ") :]
    if not raw_path or "\x00" in raw_path:
        raise PackageError("worktree .git file has an invalid path")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = source_path / candidate
    return Path(os.path.abspath(os.fspath(candidate)))


def _open_repository(path: Path) -> _Repository:
    source_path, source_fd = _open_existing_directory(path, label="source root")
    owned: list[int] = [source_fd]
    pinned_entries: list[_PinnedEntry] = []
    directory_paths: list[tuple[int, str]] = [
        (source_fd, str(source_path)),
    ]
    try:
        dot_git_fd = _open_entry_at(source_fd, ".git", label="source .git")
        owned.append(dot_git_fd)
        dot_git_details = os.fstat(dot_git_fd)
        if stat.S_ISDIR(dot_git_details.st_mode):
            dot_git = _PinnedEntry(
                parent_fd=source_fd,
                name=".git",
                descriptor=dot_git_fd,
                kind="directory",
            )
            git_dir_fd = os.dup(dot_git_fd)
            owned.append(git_dir_fd)
            git_dir_path = Path(_descriptor_path(git_dir_fd))
        elif stat.S_ISREG(dot_git_details.st_mode):
            dot_git_data = _read_pinned_bytes(
                dot_git_fd,
                label="source .git file",
                maximum_bytes=16 * 1024,
            )
            dot_git = _PinnedEntry(
                parent_fd=source_fd,
                name=".git",
                descriptor=dot_git_fd,
                kind="file",
                sha256=_sha256(dot_git_data),
            )
            requested_git_dir = _parse_gitdir_file(dot_git_data, source_path)
            git_dir_path, git_dir_fd = _open_existing_directory(
                requested_git_dir,
                label="worktree Git directory",
            )
            owned.append(git_dir_fd)
        else:
            raise PackageError("source .git must be a file or directory")
        pinned_entries.append(dot_git)
        directory_paths.append((git_dir_fd, str(git_dir_path)))

        try:
            common_link = _pin_entry(
                git_dir_fd,
                "commondir",
                label="Git commondir file",
                expected_kind="file",
            )
        except PackageError as error:
            try:
                os.stat("commondir", dir_fd=git_dir_fd, follow_symlinks=False)
            except FileNotFoundError:
                common_link = None
            else:
                raise error
        if common_link is None:
            common_dir_fd = os.dup(git_dir_fd)
            owned.append(common_dir_fd)
            common_dir_path = git_dir_path
        else:
            owned.append(common_link.descriptor)
            pinned_entries.append(common_link)
            try:
                common_text = _read_pinned_bytes(
                    common_link.descriptor,
                    label="Git commondir file",
                    maximum_bytes=16 * 1024,
                ).decode("utf-8", errors="strict").strip()
            except UnicodeDecodeError as error:
                raise PackageError("Git commondir file is not UTF-8") from error
            if not common_text or "\x00" in common_text:
                raise PackageError("Git commondir file has an invalid path")
            common_candidate = Path(common_text)
            if not common_candidate.is_absolute():
                common_candidate = git_dir_path / common_candidate
            common_dir_path, common_dir_fd = _open_existing_directory(
                Path(os.path.abspath(os.fspath(common_candidate))),
                label="Git common directory",
            )
            owned.append(common_dir_fd)
        directory_paths.append((common_dir_fd, str(common_dir_path)))

        for name, label in (
            ("HEAD", "Git HEAD"),
            ("index", "Git index"),
        ):
            entry = _pin_entry(
                git_dir_fd,
                name,
                label=label,
                expected_kind="file",
            )
            owned.append(entry.descriptor)
            pinned_entries.append(entry)
        config_entry = _pin_entry(
            common_dir_fd,
            "config",
            label="Git repository config",
            expected_kind="file",
        )
        owned.append(config_entry.descriptor)
        pinned_entries.append(config_entry)
        objects_entry = _pin_entry(
            common_dir_fd,
            "objects",
            label="Git object directory",
            expected_kind="directory",
        )
        owned.append(objects_entry.descriptor)
        pinned_entries.append(objects_entry)
        objects_path = _descriptor_path(objects_entry.descriptor)
        directory_paths.append((objects_entry.descriptor, objects_path))

        repository = _Repository(
            source_fd=source_fd,
            source_path=str(source_path),
            git_dir_fd=git_dir_fd,
            git_dir_path=str(git_dir_path),
            pinned_entries=tuple(pinned_entries),
            directory_paths=tuple(directory_paths),
            owned_descriptors=tuple(owned),
        )
        _assert_repository_bindings(repository)
        return repository
    except Exception:
        for descriptor in reversed(owned):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _assert_pinned_entry(entry: _PinnedEntry) -> None:
    pinned = os.fstat(entry.descriptor)
    try:
        current = os.stat(
            entry.name,
            dir_fd=entry.parent_fd,
            follow_symlinks=False,
        )
    except OSError as error:
        raise PackageError(f"pinned Git entry changed: {entry.name}") from error
    expected_type = stat.S_IFDIR if entry.kind == "directory" else stat.S_IFREG
    if (
        stat.S_IFMT(current.st_mode) != expected_type
        or current.st_dev != pinned.st_dev
        or current.st_ino != pinned.st_ino
    ):
        raise PackageError(f"pinned Git entry was replaced: {entry.name}")
    if entry.kind == "file":
        if current.st_nlink != 1:
            raise PackageError(f"pinned Git file is multiply linked: {entry.name}")
        data = _read_pinned_bytes(
            entry.descriptor,
            label=f"pinned Git file {entry.name}",
        )
        if _sha256(data) != entry.sha256:
            raise PackageError(f"pinned Git file changed: {entry.name}")


def _assert_repository_bindings(repository: _Repository) -> None:
    for descriptor, path in repository.directory_paths:
        _assert_directory_descriptor_path(
            descriptor,
            path,
            label="Git repository directory",
        )
    for entry in repository.pinned_entries:
        _assert_pinned_entry(entry)


def _sanitized_git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
    )
    return environment


def _run_git(
    repository: _Repository,
    *arguments: str,
    input_data: bytes | None = None,
) -> bytes:
    _assert_repository_bindings(repository)
    try:
        completed = subprocess.run(
            [
                "git",
                "--no-pager",
                "--no-replace-objects",
                f"--git-dir={repository.git_dir_path}",
                f"--work-tree={repository.source_path}",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                *arguments,
            ],
            input=input_data,
            capture_output=True,
            check=False,
            cwd="/",
            env=_sanitized_git_environment(),
            pass_fds=repository.owned_descriptors,
        )
    except OSError as error:
        raise PackageError("git repository verification failed") from error
    _assert_repository_bindings(repository)
    if completed.returncode != 0:
        raise PackageError("git repository verification failed")
    return completed.stdout


def _head_revision(repository: _Repository) -> str:
    revision = (
        _run_git(repository, "rev-parse", "--verify", "HEAD")
        .decode("ascii", errors="strict")
        .strip()
    )
    if _SOURCE_OBJECT_ID_PATTERN.fullmatch(revision) is None:
        raise PackageError("Git HEAD is not a 40-character lowercase commit ID")
    return revision


def _clean_revision(repository: _Repository) -> str:
    before = _head_revision(repository)
    status = _run_git(
        repository,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    if status:
        raise PackageError("dirty Git trees cannot produce a release")
    after = _head_revision(repository)
    if before != after:
        raise PackageError("Git HEAD changed during clean-tree verification")
    return before


def _commit_tree(repository: _Repository, revision: str) -> str:
    tree_id = (
        _run_git(
            repository,
            "rev-parse",
            "--verify",
            f"{revision}^{{tree}}",
        )
        .decode("ascii", errors="strict")
        .strip()
    )
    if _SOURCE_OBJECT_ID_PATTERN.fullmatch(tree_id) is None:
        raise PackageError(
            "Git commit does not identify a 40-character lowercase tree ID"
        )
    return tree_id


def _assert_repository_unchanged(
    repository: _Repository,
    revision: str,
) -> None:
    before = _head_revision(repository)
    status = _run_git(
        repository,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    after = _head_revision(repository)
    if before != revision or after != revision or status:
        raise PackageError("source Git tree changed during packaging")


def _tracked_files(repository: _Repository, revision: str) -> list[_Tracked]:
    output = _run_git(
        repository,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        revision,
    )
    result: list[_Tracked] = []
    seen: set[str] = set()
    for raw in output.split(b"\0"):
        if not raw:
            continue
        try:
            metadata, encoded_path = raw.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split()
            path = encoded_path.decode("ascii")
        except (UnicodeDecodeError, ValueError) as error:
            raise PackageError("Git tree contains an unsupported path") from error
        portable = PurePosixPath(path)
        if (
            portable.is_absolute()
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise PackageError("Git tree contains an unsafe path")
        if path in seen:
            raise PackageError("Git tree contains a duplicate path")
        seen.add(path)
        if mode == "120000":
            raise PackageError(f"tracked symlink is forbidden: {path}")
        if object_type != "blob":
            raise PackageError(f"unsupported tracked Git object: {path}")
        if mode not in {"100644", "100755"}:
            raise PackageError(f"unsupported tracked file mode: {path}")
        if _OBJECT_ID_PATTERN.fullmatch(object_id) is None:
            raise PackageError("Git tree contains an invalid blob ID")
        result.append(_Tracked(path=path, mode=mode, object_id=object_id))
    return sorted(result, key=lambda item: item.path)


def _canonical_security_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _is_forbidden_component(value: str) -> bool:
    canonical = _canonical_security_name(value)
    if canonical.startswith("sealed"):
        return True
    return any(
        pattern.fullmatch(canonical) is not None
        for pattern in _FORBIDDEN_COMPONENT_PATTERNS
    )


def _is_sensitive_field_name(value: str) -> bool:
    canonical = _canonical_security_name(value)
    if canonical in _SENSITIVE_FIELD_EXACT:
        return True
    if canonical.endswith(("secret", "token")):
        return True
    if any(form in canonical for form in _SENSITIVE_FIELD_VALUE_FORMS):
        return True
    return any(
        concept in canonical for concept in _SENSITIVE_FIELD_AFFIXES
    )


def _path_contains_forbidden_content(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if not parts:
        return False
    if any(_is_forbidden_component(part) for part in parts[:-1]):
        return True
    return _is_forbidden_component(PurePosixPath(parts[-1]).stem)


def _classification(path: str) -> str:
    parts = PurePosixPath(path).parts
    if not parts:
        return "unknown"
    if path == RUNBOOK_PATH:
        return "included"
    if path in {"msctl/dataset.py", "train/data.py"}:
        return "included"
    if path == STATIC_ENVIRONMENT_LOCK_PATH:
        return "forbidden"
    if path in _ROOT_EXCLUDED or path in _PROVIDER_EXCLUDED:
        return "excluded"
    if parts[0] in _EXCLUDED_TOP_LEVEL:
        return "excluded"
    if _path_contains_forbidden_content(path):
        return "forbidden"
    if set(parts) & _DISPOSABLE_COMPONENTS:
        return "excluded"
    if path in PROVIDER_BRIDGE_MEMBERS:
        return "included"
    if parts[0] == "tests":
        if path in _APPROVED_TEST_MODULES or path in _SHARED_TEST_FIXTURES:
            return "included"
        return "excluded"
    if path in _ROOT_INCLUDED:
        return "included"
    if (
        path in {COHORT_PATH, PREREGISTRATION_PATH}
        or path in EXPECTED_CONFIGS
        or path in _SHARED_CONFIGS
    ):
        return "included"
    if path in _LEGACY_CONFIG_FILES or path.startswith(_LEGACY_CONFIG_PREFIXES):
        return "excluded"
    if path.startswith("configs/"):
        return "unknown"
    if path == PROFILE_PATH:
        return "included"
    if parts[0] == "cluster":
        if (
            len(parts) >= 4
            and parts[0:3] == ("cluster", "aws", "p5")
            and PurePosixPath(path).suffix.lower() in {".py", ".sh"}
        ):
            return "included"
        return "excluded"
    if path in _APPROVED_SOURCES or path in _APPROVED_VENDOR:
        return "included"
    if parts[0] in {"sources", "vendor"}:
        return "unknown"
    suffixes = _SHARED_SUFFIXES.get(parts[0])
    if parts[0] in _SHARED_SUFFIXES:
        if PurePosixPath(path).suffix.lower() in suffixes:
            return "included"
        return "unknown"
    return "unknown"


def _read_git_blobs(
    repository: _Repository,
    tracked: list[_Tracked],
) -> dict[str, bytes]:
    queries = b"".join(
        item.object_id.encode("ascii") + b"\n" for item in tracked
    )
    output = _run_git(
        repository,
        "cat-file",
        "--batch",
        input_data=queries,
    )
    cursor = 0
    blobs: dict[str, bytes] = {}
    try:
        for item in tracked:
            header_end = output.index(b"\n", cursor)
            header = output[cursor:header_end].decode("ascii").split()
            if (
                len(header) != 3
                or header[0] != item.object_id
                or header[1] != "blob"
            ):
                raise ValueError
            size = int(header[2])
            data_start = header_end + 1
            data_end = data_start + size
            if size < 0 or output[data_end : data_end + 1] != b"\n":
                raise ValueError
            blobs[item.path] = output[data_start:data_end]
            cursor = data_end + 1
    except (UnicodeDecodeError, ValueError) as error:
        raise PackageError("Git blob snapshot response is malformed") from error
    if cursor != len(output) or len(blobs) != len(tracked):
        raise PackageError("Git blob snapshot response is incomplete")
    return blobs


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
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        try:
            text = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise PackageError(f"text release member is not UTF-8: {path}") from error
        if "\x00" in text:
            raise PackageError(f"text release member contains a NUL byte: {path}")
    if suffix == ".json":
        value = _load_json_value(data, label=path)
        _reject_sensitive_fields(value, label=path)
    elif suffix in {".yaml", ".yml"}:
        value = _load_yaml_value(data, path=path)
        _reject_sensitive_fields(value, label=path)


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


def _unique_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PackageError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise PackageError(f"JSON contains a non-finite number: {value}")


def _load_json_value(data: bytes, *, label: str) -> object:
    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except PackageError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackageError(f"{label} is not canonical JSON data") from error
    return value


def _load_json_object(data: bytes, *, label: str) -> dict[str, object]:
    value = _load_json_value(data, label=label)
    if not isinstance(value, dict):
        raise PackageError(f"{label} must be a JSON object")
    return value


def _load_yaml_value(data: bytes, *, path: str) -> object:
    try:
        text = data.decode("utf-8", errors="strict")
        value = yaml.load(text, Loader=_UniqueSafeLoader)
    except PackageError:
        raise
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise PackageError(f"release member is not valid YAML: {path}") from error
    return value


def _load_yaml_object(data: bytes, *, path: str) -> dict[str, object]:
    value = _load_yaml_value(data, path=path)
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise PackageError(f"run config must be a string-keyed mapping: {path}")
    return value


def _same_typed_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (
            actual.keys() == expected.keys()
            and all(
                _same_typed_value(actual[key], expected[key])
                for key in expected
            )
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_typed_value(left, right)
            for left, right in zip(actual, expected)
        )
    return actual == expected


def _expected_assignment() -> dict[str, object]:
    return {
        "cohort_id": COHORT_ID,
        "model_parameters": 356_033_536,
        "optimizer_steps": 13_582,
        "provider_seeds": {
            PROVIDER: list(AWS_SEEDS),
        },
        "raw_target_tokens": 7_120_879_616,
        "schema_version": 3,
        "targets_per_update": 524_288,
    }


def _validate_assignment(data: bytes) -> dict[str, object]:
    assignment = _load_json_object(data, label="cohort assignment")
    if not _same_typed_value(assignment, _expected_assignment()):
        raise PackageError(
            "cohort assignment must assign exactly seeds 0-9 to AWS P5 "
            "with no other provider"
        )
    if (
        assignment["optimizer_steps"] * assignment["targets_per_update"]
        != assignment["raw_target_tokens"]
    ):
        raise PackageError("cohort assignment token math is inconsistent")
    return assignment


def _expected_config(seed: int, arm: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "cohort_id": COHORT_ID,
        "run_id": f"memorysplit-v3-360m-s{seed}-{arm}",
        "condition": arm,
        "seed": seed,
        "model": "d360m",
        "ctx": 1024,
        "train_corpus": DATASET_RECEIPT_PATH,
        "sidecar_name": (
            "dense_target_weights"
            if arm == "dense"
            else "split90_target_weights"
        ),
        "out_dir": f"runs/seed-{seed}/{arm}",
        "micro_batch_size": 8,
        "tokens_per_step": 524_288,
        "max_steps": 13_582,
        "total_tokens": 7_120_879_616,
        "lr": 0.001,
        "warmup_steps": 300,
        "weight_decay": 0.1,
        "compile": True,
        "device": "cuda",
        "log_every": 20,
        "eval_every": 250,
        "snapshot_steps": list(SNAPSHOT_STEPS),
        "ckpt_minutes": 30,
    }


def _validate_config(path: str, data: bytes, *, seed: int, arm: str) -> None:
    config = _load_yaml_object(data, path=path)
    if set(config) != _CONFIG_KEYS:
        missing = sorted(_CONFIG_KEYS - set(config))
        extra = sorted(set(config) - _CONFIG_KEYS)
        detail = missing[0] if missing else extra[0]
        raise PackageError(f"run config has invalid field {detail}: {path}")
    snapshot_steps = config["snapshot_steps"]
    if not isinstance(snapshot_steps, list) or any(
        type(step) is not int for step in snapshot_steps
    ):
        raise PackageError(
            f"run config snapshot_steps must be an integer list: {path}"
        )
    if not snapshot_steps or snapshot_steps[-1] != config["max_steps"]:
        raise PackageError(
            f"run config snapshot_steps must end at max_steps: {path}"
        )
    if snapshot_steps != list(SNAPSHOT_STEPS):
        raise PackageError(
            f"run config snapshot_steps does not match the frozen schedule: {path}"
        )
    expected = _expected_config(seed, arm)
    for field, expected_value in expected.items():
        if not _same_typed_value(config[field], expected_value):
            raise PackageError(
                f"run config {field} does not match the frozen contract: {path}"
            )
    if config["max_steps"] * config["tokens_per_step"] != config["total_tokens"]:
        raise PackageError(f"run config token math is inconsistent: {path}")


def _nested_value(
    value: dict[str, object],
    path: tuple[str, ...],
    *,
    label: str,
) -> object:
    current: object = value
    for component in path:
        if not isinstance(current, dict) or component not in current:
            raise PackageError(
                f"{label} is missing required field {'.'.join(path)}"
            )
        current = current[component]
    return current


def _reject_sensitive_fields(value: object, *, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise PackageError(f"{label} contains a non-string field")
            if _is_sensitive_field_name(key):
                raise PackageError(
                    f"{label} contains a static credential or secret field"
                )
            _reject_sensitive_fields(child, label=label)
    elif isinstance(value, list):
        for child in value:
            _reject_sensitive_fields(child, label=label)


def _validate_profile(data: bytes) -> None:
    profile = _load_json_object(data, label="AWS P5 profile")
    _reject_sensitive_fields(profile, label="AWS P5 profile")
    runtime = _nested_value(profile, ("runtime",), label="AWS P5 profile")
    if not isinstance(runtime, dict) or set(runtime) != {
        "ami_id_env",
        "container_digest_env",
        "region_env",
        "runtime_gid_env",
        "runtime_uid_env",
    }:
        raise PackageError(
            "AWS P5 profile runtime must not claim a static environment identity"
        )
    expected_values = {
        ("schema_version",): 1,
        ("profile_id",): "aws-p5.48xlarge-v3",
        ("provider",): PROVIDER,
        ("instance_type",): "p5.48xlarge",
        ("purchase_model",): "on_demand",
        ("assigned_seeds",): list(AWS_SEEDS),
        ("gpu", "model"): "NVIDIA H100 80GB",
        ("gpu", "allocated"): 8,
        ("gpu", "seed_train_groups"): [4, 4],
        ("cpu", "vcpus"): 192,
        ("cpu", "memory_gib"): 2048,
        ("storage", "instance_store", "devices"): 8,
        ("storage", "instance_store", "device_bytes"): 3_840_000_000_000,
        ("storage", "instance_store", "model"): (
            "Amazon EC2 NVMe Instance Storage"
        ),
        ("storage", "instance_store", "raid_level"): "0",
        ("storage", "scratch_root"): "/mnt/memorysplit",
        ("storage", "durable_uri_env"): "MS_S3_ROOT",
        ("runtime", "ami_id_env"): "MS_AWS_AMI_ID",
        ("runtime", "container_digest_env"): "MS_CONTAINER_DIGEST",
        ("runtime", "region_env"): "AWS_REGION",
        ("runtime", "runtime_gid_env"): "MS_RUNTIME_GID",
        ("runtime", "runtime_uid_env"): "MS_RUNTIME_UID",
        ("process_env_allowlist",): ["AWS_REGION", "LANG", "LC_ALL"],
    }
    for path, expected in expected_values.items():
        actual = _nested_value(profile, path, label="AWS P5 profile")
        if not _same_typed_value(actual, expected):
            raise PackageError(
                "AWS P5 profile field "
                f"{'.'.join(path)} does not match the frozen contract"
            )


def _runtime_environment_contract(profile_sha256: str) -> dict[str, object]:
    return {
        "mode": "runtime_attested",
        "profile_sha256": profile_sha256,
        "container_image_digest_env": "MS_CONTAINER_DIGEST",
        "container_image_digest_pattern": CONTAINER_IMAGE_DIGEST_PATTERN,
        "runtime_environment_receipt": {
            "required_at_launch": True,
            "authentication": "aws_instance_identity_document_pkcs7",
            "required_fields": list(AWS_ENVIRONMENT_RECEIPT_V2_FIELDS),
        },
    }


def _validate_dataset_pointer(data: bytes) -> None:
    pointer = _load_json_object(data, label="AWS dataset pointer")
    _reject_sensitive_fields(pointer, label="AWS dataset pointer")
    expected = {
        "dataset_id": "memorysplit-v2-20x-reasoning-max-cohort",
        "durable_uri_env": "MS_S3_ROOT",
        "full_corpus_in_release": False,
        "materialization": "s3",
        "provider": PROVIDER,
        "relative_path": "dataset",
        "required_receipt": DATASET_RECEIPT_PATH,
        "required_sidecars": [
            "dense_target_weights",
            "split90_target_weights",
        ],
        "schema_version": 1,
        "scratch_root": "/mnt/memorysplit",
        "source_lock_manifest": "configs/reasoning-dataset-v2.json",
    }
    if not _same_typed_value(pointer, expected):
        raise PackageError("AWS dataset pointer does not match the exact contract")


def _validate_preregistration(data: bytes) -> None:
    preregistration = _load_yaml_object(data, path=PREREGISTRATION_PATH)
    expected = {
        "schema_version": 3,
        "preregistration_id": PREREGISTRATION_ID,
    }
    for field, expected_value in expected.items():
        if field not in preregistration or not _same_typed_value(
            preregistration[field], expected_value
        ):
            raise PackageError(
                f"preregistration {field} does not match the v3 contract"
            )


def _collect_payload(
    repository: _Repository,
    tracked: list[_Tracked],
    revision: str,
    tree_id: str,
    *,
    lifecycle: AuthenticatedProviderLifecycle | None = None,
) -> _Collected:
    included: list[_Tracked] = []
    unknown: list[str] = []
    forbidden: list[str] = []
    for item in tracked:
        classification = _classification(item.path)
        if classification == "included":
            included.append(item)
        elif classification == "unknown":
            unknown.append(item.path)
        elif classification == "forbidden":
            forbidden.append(item.path)
    if forbidden:
        raise PackageError(
            f"forbidden tracked path cannot be released: {forbidden[0]}"
        )
    if unknown:
        raise PackageError(
            f"unknown tracked path is not allowlisted: {unknown[0]}"
        )
    included_paths = {item.path for item in included}
    missing = sorted(REQUIRED_MEMBERS - included_paths)
    if missing:
        raise PackageError(f"required release member is not tracked: {missing[0]}")

    payload: dict[str, bytes] = {}
    modes: dict[str, str] = {}
    member_rows: list[dict[str, object]] = []
    _assert_repository_unchanged(repository, revision)
    snapshot = _read_git_blobs(repository, included)
    for item in included:
        data = snapshot[item.path]
        _scan_secret(item.path, data)
        payload[item.path] = data
        modes[item.path] = item.mode
        member_rows.append(
            {
                "path": item.path,
                "bytes": len(data),
                "sha256": _sha256(data),
                "git_blob": item.object_id,
                "git_mode": item.mode,
            }
        )

    assignment = _validate_assignment(payload[COHORT_PATH])
    _validate_preregistration(payload[PREREGISTRATION_PATH])
    for seed in AWS_SEEDS:
        for arm in ARMS:
            path = f"{CONFIG_ROOT}/{arm}-s{seed}.yaml"
            _validate_config(path, payload[path], seed=seed, arm=arm)
    _validate_profile(payload[PROFILE_PATH])
    _validate_dataset_pointer(payload[DATASET_POINTER_PATH])

    config_sha256 = {
        path: _sha256(payload[path]) for path in sorted(EXPECTED_CONFIGS)
    }
    cohort_sha256 = _sha256(payload[COHORT_PATH])
    if lifecycle is None:
        provider = PROVIDER
        profile_id = "aws-p5.48xlarge-v3"
        profile_path = PROFILE_PATH
        lifecycle_fields: dict[str, object] = {}
    else:
        from msctl.aws_lifecycle import AuthenticatedProviderLifecycle

        if not isinstance(lifecycle, AuthenticatedProviderLifecycle):
            raise PackageError(
                "authenticated provider lifecycle authority is invalid"
            )
        provider = lifecycle.binding.provider
        profile_id = lifecycle.binding.profile_id
        profile_path = {
            "aws-p5.48xlarge-v3": PROFILE_PATH,
            "aws-p6-b300.48xlarge-v3": (
                "cluster/profiles/aws-p6-b300.48xlarge-v3.json"
            ),
        }.get(profile_id, "")
        if not profile_path or profile_path not in payload:
            raise PackageError(
                "selected provider profile is not a reviewed release member"
            )
        lifecycle_fields = lifecycle.binding.to_dict()
    profile_sha256 = _sha256(payload[profile_path])
    if lifecycle is not None and (
        profile_sha256 != lifecycle.binding.profile_sha256
        or lifecycle.profile.provider != provider
        or lifecycle.profile.profile_id != profile_id
        or lifecycle.profile.sha256 != profile_sha256
    ):
        raise PackageError(
            "selected provider profile bytes differ from authenticated authority"
        )
    environment = _runtime_environment_contract(profile_sha256)
    dataset_pointer_sha256 = _sha256(payload[DATASET_POINTER_PATH])
    seed_assignment: dict[str, object] = {
        "cohort_id": assignment["cohort_id"],
        "provider": provider,
        "seeds": list(AWS_SEEDS),
        "arms": list(ARMS),
    }
    payload["RELEASE-METADATA.json"] = _canonical_pretty(
        {
            **lifecycle_fields,
            "schema_version": 1,
            "package_format_version": PACKAGE_FORMAT_VERSION,
            "provider": provider,
            "source": {
                "commit": revision,
                "dirty": False,
                "tree": tree_id,
            },
            "seed_assignment": seed_assignment,
            "cohort_assignment": {
                "path": COHORT_PATH,
                "sha256": cohort_sha256,
            },
            "profile": {
                "path": profile_path,
                "sha256": profile_sha256,
            },
            **({"profile_id": profile_id} if lifecycle_fields else {}),
            "environment": environment,
            "dataset_pointer": {
                "path": DATASET_POINTER_PATH,
                "sha256": dataset_pointer_sha256,
            },
            "config_sha256": config_sha256,
            "members": member_rows,
        }
    )
    modes["RELEASE-METADATA.json"] = "100644"
    sums = "".join(
        f"{_sha256(payload[name])}  {name}\n" for name in sorted(payload)
    ).encode("ascii")
    payload["SHA256SUMS"] = sums
    modes["SHA256SUMS"] = "100644"
    return _Collected(
        payload=payload,
        modes=modes,
        tree_id=tree_id,
        members_sha256=_sha256(sums),
        cohort_sha256=cohort_sha256,
        profile_sha256=profile_sha256,
        environment=environment,
        dataset_pointer_sha256=dataset_pointer_sha256,
        config_sha256=config_sha256,
        seed_assignment=seed_assignment,
        provider=provider,
        profile_id=profile_id,
        profile_path=profile_path,
        lifecycle_fields=lifecycle_fields,
    )


def _planned_directories(paths: Iterator[str]) -> set[str]:
    directories: set[str] = set()
    for path in paths:
        parts = PurePosixPath(path).parts
        for end in range(1, len(parts)):
            directories.add("/".join(parts[:end]) + "/")
    return directories


def _zip_info(name: str, *, mode: int, directory: bool) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=NORMALIZED_TIME)
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.compress_type = (
        zipfile.ZIP_STORED if directory else zipfile.ZIP_DEFLATED
    )
    info.flag_bits = 0
    info.extra = b""
    info.comment = b""
    info.internal_attr = 0
    file_type = stat.S_IFDIR if directory else stat.S_IFREG
    info.external_attr = (file_type | mode) << 16
    if directory:
        info.external_attr |= 0x10
    return info


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise PackageError("short write while staging release")
        written += count


def _write_zip_at(
    directory_fd: int,
    name: str,
    *,
    payload: dict[str, bytes],
    modes: dict[str, str],
) -> None:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    except OSError as error:
        raise PackageError("cannot create private staged archive") from error
    try:
        with os.fdopen(os.dup(descriptor), "w+b") as handle:
            with zipfile.ZipFile(
                handle,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
                strict_timestamps=True,
            ) as archive:
                archive.comment = b""
                entries: list[tuple[str, bool]] = [
                    (directory, True)
                    for directory in _planned_directories(iter(payload))
                ]
                entries.extend((path, False) for path in payload)
                for path, is_directory in sorted(entries):
                    if is_directory:
                        info = _zip_info(path, mode=0o755, directory=True)
                        archive.writestr(info, b"", compress_type=zipfile.ZIP_STORED)
                    else:
                        mode = 0o755 if modes[path] == "100755" else 0o644
                        info = _zip_info(path, mode=mode, directory=False)
                        archive.writestr(
                            info,
                            payload[path],
                            compress_type=zipfile.ZIP_DEFLATED,
                            compresslevel=9,
                        )
            handle.flush()
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_at(
    directory_fd: int,
    name: str,
    data: bytes,
    *,
    mode: int = 0o644,
) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    except OSError as error:
        raise PackageError("cannot create private release receipt") from error
    try:
        _write_all(descriptor, data)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_pinned_regular_at(directory_fd: int, name: str) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise PackageError("staged release member cannot be opened safely") from error
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            raise PackageError(
                "staged release member is not a singly-linked regular file"
            )
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _assert_descriptor_names_entry(
    directory_fd: int,
    name: str,
    descriptor: int,
) -> None:
    pinned = os.fstat(descriptor)
    try:
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise PackageError(
            "staged release path changed after descriptor pinning"
        ) from error
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_dev != pinned.st_dev
        or current.st_ino != pinned.st_ino
        or current.st_size != pinned.st_size
        or current.st_nlink != 1
    ):
        raise PackageError(
            "staged release path was replaced after descriptor pinning"
        )


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _hash_descriptor(descriptor: int) -> tuple[str, int]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
    if total != os.fstat(descriptor).st_size:
        raise PackageError("archive size changed while hashing descriptor")
    return digest.hexdigest(), total


def _verify_zip_descriptor(
    descriptor: int,
    *,
    payload: dict[str, bytes],
    modes: dict[str, str],
) -> None:
    expected_directories = _planned_directories(iter(payload))
    expected_names = expected_directories | set(payload)
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            with zipfile.ZipFile(handle) as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if len(names) != len(set(names)) or set(names) != expected_names:
                    raise PackageError(
                        "staged archive has duplicate or unexpected members"
                    )
                for info in infos:
                    if info.date_time != NORMALIZED_TIME or info.create_system != 3:
                        raise PackageError(
                            "staged archive metadata is not normalized"
                        )
                    raw_mode = info.external_attr >> 16
                    expected_mode = (
                        0o755
                        if info.is_dir()
                        else (0o755 if modes[info.filename] == "100755" else 0o644)
                    )
                    if stat.S_IMODE(raw_mode) != expected_mode:
                        raise PackageError("staged archive mode is not normalized")
                    if not info.is_dir():
                        if archive.read(info.filename) != payload[info.filename]:
                            raise PackageError(
                                "staged archive member checksum verification failed"
                            )
                expected_sums = "".join(
                    f"{_sha256(payload[name])}  {name}\n"
                    for name in sorted(payload)
                    if name != "SHA256SUMS"
                ).encode("ascii")
                if archive.read("SHA256SUMS") != expected_sums:
                    raise PackageError(
                        "staged archive internal checksums are invalid"
                    )
    except PackageError:
        raise
    except (OSError, zipfile.BadZipFile, KeyError, RuntimeError) as error:
        raise PackageError("staged archive verification failed") from error


def _verify_new_file_at(
    directory_fd: int,
    name: str,
    expected: bytes,
) -> None:
    descriptor = _open_pinned_regular_at(directory_fd, name)
    try:
        _assert_descriptor_names_entry(directory_fd, name, descriptor)
        if _read_descriptor(descriptor) != expected:
            raise PackageError("staged release receipt verification failed")
        _assert_descriptor_names_entry(directory_fd, name, descriptor)
    finally:
        os.close(descriptor)


def _build_staging(
    staging_fd: int,
    *,
    collected: _Collected,
    revision: str,
    release_id: str,
    archive_name: str,
) -> _Staged:
    _write_zip_at(
        staging_fd,
        archive_name,
        payload=collected.payload,
        modes=collected.modes,
    )
    archive_fd = _open_pinned_regular_at(staging_fd, archive_name)
    try:
        _assert_descriptor_names_entry(staging_fd, archive_name, archive_fd)
        archive_hash, archive_bytes = _hash_descriptor(archive_fd)
        _verify_zip_descriptor(
            archive_fd,
            payload=collected.payload,
            modes=collected.modes,
        )
        _assert_descriptor_names_entry(staging_fd, archive_name, archive_fd)

        checksum_name = f"{archive_name}.sha256"
        checksum_bytes = f"{archive_hash}  {archive_name}\n".encode("ascii")
        release_value = {
            **collected.lifecycle_fields,
            "schema_version": 1,
            "package_format_version": PACKAGE_FORMAT_VERSION,
            "release_id": release_id,
            "provider": collected.provider,
            "archive": {
                "path": archive_name,
                "sha256": archive_hash,
                "bytes": archive_bytes,
            },
            "source": {
                "commit": revision,
                "dirty": False,
                "tree": collected.tree_id,
            },
            "seed_assignment": collected.seed_assignment,
            "cohort_assignment": {
                "path": COHORT_PATH,
                "sha256": collected.cohort_sha256,
            },
            "profile": {
                "path": collected.profile_path,
                "sha256": collected.profile_sha256,
            },
            **(
                {"profile_id": collected.profile_id}
                if collected.lifecycle_fields
                else {}
            ),
            "environment": collected.environment,
            "dataset_pointer": {
                "path": DATASET_POINTER_PATH,
                "sha256": collected.dataset_pointer_sha256,
            },
            "cohort_assignment_sha256": collected.cohort_sha256,
            "profile_sha256": collected.profile_sha256,
            "dataset_pointer_sha256": collected.dataset_pointer_sha256,
            "config_sha256": collected.config_sha256,
            "members_sha256": collected.members_sha256,
        }
        release_bytes = _canonical_pretty(release_value)
        _write_new_at(staging_fd, checksum_name, checksum_bytes)
        _write_new_at(staging_fd, RELEASE_RECEIPT_NAME, release_bytes)
        _verify_new_file_at(staging_fd, checksum_name, checksum_bytes)
        _verify_new_file_at(staging_fd, RELEASE_RECEIPT_NAME, release_bytes)
        expected_entries = {
            archive_name,
            checksum_name,
            RELEASE_RECEIPT_NAME,
        }
        if set(os.listdir(staging_fd)) != expected_entries:
            raise PackageError("private release staging has unexpected entries")
        _assert_descriptor_names_entry(staging_fd, archive_name, archive_fd)
        os.fsync(staging_fd)
        return _Staged(
            archive_fd=archive_fd,
            archive_sha256=archive_hash,
            archive_bytes=archive_bytes,
            checksum_bytes=checksum_bytes,
            release_bytes=release_bytes,
        )
    except Exception:
        os.close(archive_fd)
        raise


def _verify_installed_release(
    directory_fd: int,
    *,
    staged: _Staged,
    collected: _Collected,
    archive_name: str,
) -> None:
    expected_entries = {
        archive_name,
        f"{archive_name}.sha256",
        RELEASE_RECEIPT_NAME,
    }
    if set(os.listdir(directory_fd)) != expected_entries:
        raise PackageError("installed release has unexpected artifacts")
    _assert_descriptor_names_entry(
        directory_fd,
        archive_name,
        staged.archive_fd,
    )
    digest, size = _hash_descriptor(staged.archive_fd)
    if digest != staged.archive_sha256 or size != staged.archive_bytes:
        raise PackageError("installed archive identity verification failed")
    _verify_zip_descriptor(
        staged.archive_fd,
        payload=collected.payload,
        modes=collected.modes,
    )
    _verify_new_file_at(
        directory_fd,
        f"{archive_name}.sha256",
        staged.checksum_bytes,
    )
    _verify_new_file_at(
        directory_fd,
        RELEASE_RECEIPT_NAME,
        staged.release_bytes,
    )
    _assert_descriptor_names_entry(
        directory_fd,
        archive_name,
        staged.archive_fd,
    )
    os.fsync(directory_fd)


def _open_or_create_output(path: Path) -> tuple[Path, int]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute == Path("/"):
        raise PackageError("release output cannot be the filesystem root")
    current_fd = os.open("/", _directory_flags())
    try:
        for component in absolute.parts[1:]:
            try:
                child_fd = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=current_fd,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=current_fd)
                    child_fd = os.open(
                        component,
                        _directory_flags(),
                        dir_fd=current_fd,
                    )
                except OSError as error:
                    raise PackageError(
                        "release output cannot be created safely"
                    ) from error
            except OSError as error:
                raise PackageError(
                    "release output path contains a symlink or non-directory"
                ) from error
            os.close(current_fd)
            current_fd = child_fd
        _assert_output_control(current_fd)
        return absolute, current_fd
    except Exception:
        os.close(current_fd)
        raise


def _assert_output_control(output_fd: int) -> None:
    details = os.fstat(output_fd)
    if not stat.S_ISDIR(details.st_mode):
        raise PackageError("release output must be a directory")
    if details.st_uid != os.geteuid():
        raise PackageError("release output must be controlled by the current owner")
    if stat.S_IMODE(details.st_mode) & 0o022:
        raise PackageError(
            "release output must not be group- or world-writable"
        )


def _lock_output(output_fd: int) -> None:
    _assert_output_control(output_fd)
    try:
        fcntl.flock(output_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        raise PackageError("release output is locked by another publisher") from error
    _assert_output_control(output_fd)


def _make_staging_at(output_fd: int, release_id: str) -> tuple[str, int]:
    for _ in range(128):
        name = f".{release_id}.{secrets.token_hex(8)}.staging"
        try:
            os.mkdir(name, 0o700, dir_fd=output_fd)
        except FileExistsError:
            continue
        except OSError as error:
            raise PackageError("cannot create private release staging") from error
        try:
            descriptor = os.open(name, _directory_flags(), dir_fd=output_fd)
        except Exception:
            os.rmdir(name, dir_fd=output_fd)
            raise
        _assert_staging_path(output_fd, name, descriptor)
        return name, descriptor
    raise PackageError("cannot allocate a unique private release staging name")


def _assert_staging_path(
    output_fd: int,
    staging_name: str,
    staging_fd: int,
) -> None:
    _assert_output_control(output_fd)
    pinned = os.fstat(staging_fd)
    try:
        current = os.stat(
            staging_name,
            dir_fd=output_fd,
            follow_symlinks=False,
        )
    except OSError as error:
        raise PackageError("private staging pathname changed") from error
    if (
        not stat.S_ISDIR(pinned.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or current.st_dev != pinned.st_dev
        or current.st_ino != pinned.st_ino
        or current.st_uid != os.geteuid()
        or stat.S_IMODE(current.st_mode) & 0o077
    ):
        raise PackageError(
            "private staging pathname was replaced or is not owner-only"
        )


def _remove_staging_at(
    output_fd: int,
    staging_name: str,
    staging_fd: int,
) -> None:
    pinned = os.fstat(staging_fd)
    try:
        for name in os.listdir(staging_fd):
            try:
                details = os.stat(
                    name,
                    dir_fd=staging_fd,
                    follow_symlinks=False,
                )
                if stat.S_ISDIR(details.st_mode):
                    raise PackageError(
                        "private release staging contains an unsafe directory"
                    )
                os.unlink(name, dir_fd=staging_fd)
            except FileNotFoundError:
                continue
        matching_name = None
        for candidate in os.listdir(output_fd):
            try:
                details = os.stat(
                    candidate,
                    dir_fd=output_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            if (
                stat.S_ISDIR(details.st_mode)
                and details.st_dev == pinned.st_dev
                and details.st_ino == pinned.st_ino
            ):
                matching_name = candidate
                break
        if matching_name is not None:
            os.rmdir(matching_name, dir_fd=output_fd)
        try:
            replacement = os.stat(
                staging_name,
                dir_fd=output_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        if stat.S_ISLNK(replacement.st_mode):
            os.unlink(staging_name, dir_fd=output_fd)
    except (FileNotFoundError, OSError):
        return


def _rename_noreplace_at(
    directory_fd: int,
    source_name: str,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source_name)
    encoded_destination = os.fsencode(destination_name)
    ctypes.set_errno(0)
    if sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        rename = libc.renameatx_np
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            directory_fd,
            encoded_source,
            directory_fd,
            encoded_destination,
            0x00000004,
        )
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
            directory_fd,
            encoded_source,
            directory_fd,
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
    raise PackageError("atomic no-replace release publication failed")


def _publish_staging_at(
    output_fd: int,
    staging_name: str,
    staging_fd: int,
    release_id: str,
    *,
    staged: _Staged,
    collected: _Collected,
    archive_name: str,
) -> None:
    _assert_output_control(output_fd)
    _assert_staging_path(output_fd, staging_name, staging_fd)
    _rename_noreplace_at(output_fd, staging_name, release_id)
    try:
        _assert_staging_path(output_fd, release_id, staging_fd)
        _verify_installed_release(
            staging_fd,
            staged=staged,
            collected=collected,
            archive_name=archive_name,
        )
        _assert_staging_path(output_fd, release_id, staging_fd)
    except Exception as install_error:
        try:
            _quarantine_installed_release(output_fd, release_id)
            os.fsync(output_fd)
        except Exception as quarantine_error:
            raise PackageError(
                "unsafe installed release could not be quarantined"
            ) from quarantine_error
        raise PackageError(
            "installed release identity or artifact verification failed"
        ) from install_error


def _quarantine_installed_release(
    output_fd: int,
    release_id: str,
) -> str | None:
    try:
        os.stat(
            release_id,
            dir_fd=output_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    for _ in range(128):
        quarantine_name = (
            f".{release_id}.{secrets.token_hex(8)}.quarantine"
        )
        try:
            _rename_noreplace_at(
                output_fd,
                release_id,
                quarantine_name,
            )
            return quarantine_name
        except PackageError as error:
            if error.code == "RELEASE_EXISTS":
                continue
            raise
    raise PackageError("cannot allocate a unique release quarantine name")


def _assert_external_output(
    repository: _Repository,
    output_path: Path,
) -> Path:
    absolute = Path(os.path.abspath(os.fspath(output_path)))
    source = Path(repository.source_path)
    try:
        within_source = os.path.commonpath((str(source), str(absolute))) == str(
            source
        )
    except ValueError:
        within_source = False
    if within_source:
        raise PackageError(
            "apply output must be outside the source worktree"
        )
    return absolute


def _build_handoff(
    *,
    source_root: Path | str,
    out_dir: Path | str,
    apply: bool = False,
    lifecycle: AuthenticatedProviderLifecycle | None = None,
) -> ReleaseArtifacts:
    """Validate a clean Git snapshot and optionally publish one release set."""

    source = Path(os.path.abspath(os.fspath(source_root)))
    repository = _open_repository(source)
    try:
        output_requested = Path(out_dir)
        if apply:
            _assert_external_output(repository, output_requested)
        revision = _clean_revision(repository)
        tree_id = _commit_tree(repository, revision)
        tracked = _tracked_files(repository, tree_id)
        collected = _collect_payload(
            repository,
            tracked,
            revision,
            tree_id,
            lifecycle=lifecycle,
        )
        release_suffix = collected.members_sha256[:16]
        release_id = f"aws-p5-r1-{release_suffix}"
        archive_name = f"ms-aws-p5-r1-{release_suffix}.zip"
        release_dir = Path(os.path.abspath(os.fspath(output_requested))) / release_id

        if not apply:
            with tempfile.TemporaryDirectory(
                prefix=f".{release_id}.dry-run-"
            ) as temporary:
                staging_fd = os.open(temporary, _directory_flags())
                staged: _Staged | None = None
                try:
                    staged = _build_staging(
                        staging_fd,
                        collected=collected,
                        revision=revision,
                        release_id=release_id,
                        archive_name=archive_name,
                    )
                    _assert_descriptor_names_entry(
                        staging_fd,
                        archive_name,
                        staged.archive_fd,
                    )
                    archive_hash = staged.archive_sha256
                finally:
                    if staged is not None:
                        os.close(staged.archive_fd)
                    os.close(staging_fd)
            return ReleaseArtifacts(
                release_dir=release_dir,
                archive=release_dir / archive_name,
                sha256_file=release_dir / f"{archive_name}.sha256",
                release=release_dir / RELEASE_RECEIPT_NAME,
                release_id=release_id,
                sha256=archive_hash,
                published=False,
            )

        output, output_fd = _open_or_create_output(output_requested)
        staging_name: str | None = None
        staging_fd: int | None = None
        staged = None
        published = False
        try:
            _lock_output(output_fd)
            staging_name, staging_fd = _make_staging_at(output_fd, release_id)
            staged = _build_staging(
                staging_fd,
                collected=collected,
                revision=revision,
                release_id=release_id,
                archive_name=archive_name,
            )
            _assert_descriptor_names_entry(
                staging_fd,
                archive_name,
                staged.archive_fd,
            )
            _assert_staging_path(output_fd, staging_name, staging_fd)
            _publish_staging_at(
                output_fd,
                staging_name,
                staging_fd,
                release_id,
                staged=staged,
                collected=collected,
                archive_name=archive_name,
            )
            os.fsync(output_fd)
            published = True
            release_dir = output / release_id
            return ReleaseArtifacts(
                release_dir=release_dir,
                archive=release_dir / archive_name,
                sha256_file=release_dir / f"{archive_name}.sha256",
                release=release_dir / RELEASE_RECEIPT_NAME,
                release_id=release_id,
                sha256=staged.archive_sha256,
                published=True,
            )
        finally:
            if staged is not None:
                os.close(staged.archive_fd)
            if (
                not published
                and staging_name is not None
                and staging_fd is not None
            ):
                _remove_staging_at(output_fd, staging_name, staging_fd)
            if staging_fd is not None:
                os.close(staging_fd)
            os.close(output_fd)
    finally:
        repository.close()


def build_handoff(
    *,
    source_root: Path | str,
    out_dir: Path | str,
    apply: bool = False,
) -> ReleaseArtifacts:
    """Build the explicit legacy P5-compatible package."""

    return _build_handoff(
        source_root=source_root,
        out_dir=out_dir,
        apply=apply,
    )


def build_authenticated_handoff(
    *,
    source_root: Path | str,
    out_dir: Path | str,
    apply: bool,
    authority_root: Path | str,
    runtime_lock_path: Path | str,
    runtime_evidence_path: Path | str,
    runtime_sbom_path: Path | str,
    objective_controls_amendment_path: Path | str,
    store: object,
    account_id: str,
    instance_id: str,
    boot_id: str,
    seed: int,
    expected_selection_version_id: str,
    identity_verifier: object,
    approval_verifier: object,
    trusted_public_key_sha256: str,
) -> ReleaseArtifacts:
    """Build metadata only after fixed-byte provider authority admission."""

    lifecycle = admit_provider_lifecycle(
        authority_root=authority_root,
        repo_root=source_root,
        runtime_lock_path=runtime_lock_path,
        runtime_evidence_path=runtime_evidence_path,
        runtime_sbom_path=runtime_sbom_path,
        objective_controls_amendment_path=(
            objective_controls_amendment_path
        ),
        store=store,
        account_id=account_id,
        instance_id=instance_id,
        boot_id=boot_id,
        seed=seed,
        expected_selection_version_id=expected_selection_version_id,
        identity_verifier=identity_verifier,
        approval_verifier=approval_verifier,
        trusted_public_key_sha256=trusted_public_key_sha256,
    )
    return _build_handoff(
        source_root=source_root,
        out_dir=out_dir,
        apply=apply,
        lifecycle=lifecycle,
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
    apply = False
    try:
        parser = _JsonArgumentParser(
            description="Build a deterministic AWS P5 MemorySplit handoff."
        )
        parser.add_argument(
            "--source-root",
            default=str(Path(__file__).resolve().parents[1]),
        )
        parser.add_argument(
            "--out-dir",
            default=None,
            help=(
                "external release root; defaults to "
                "../memorysplit-releases/aws-p5 relative to the source"
            ),
        )
        parser.add_argument("--apply", action="store_true")
        args = parser.parse_args(argv)
        apply = bool(args.apply)
        source_root = Path(os.path.abspath(os.fspath(args.source_root)))
        out_dir = (
            Path(args.out_dir)
            if args.out_dir is not None
            else source_root.parent / "memorysplit-releases" / "aws-p5"
        )
        artifacts = build_handoff(
            source_root=source_root,
            out_dir=out_dir,
            apply=apply,
        )
        report = {
            "schema_version": 1,
            "ok": True,
            "provider": PROVIDER,
            "dry_run": not apply,
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
            "provider": PROVIDER,
            "dry_run": True,
            "published": False,
            "help": help_request.text,
        }
        code = 0
    except PackageError as error:
        report = {
            "schema_version": 1,
            "ok": False,
            "provider": PROVIDER,
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
            "provider": PROVIDER,
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
