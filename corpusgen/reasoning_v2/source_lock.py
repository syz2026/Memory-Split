"""Immutable, licensed source resolution and content-addressed staging."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import stat
import subprocess
import tarfile
import tempfile
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from huggingface_hub import HfApi, hf_hub_download

from corpusgen.parallel.canonical import canonical_json_bytes, sha256_hex
from corpusgen.parallel.safeio import (
    atomic_rename_noreplace,
    entry_exists,
    fsync_directory,
    open_directory_at,
    open_directory_path,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "memorysplit-v2-20x-reasoning-max-cohort"
DATASET_CONTRACT_PATH = ROOT / "configs/reasoning-dataset-v2.json"
CURRENT_DATASET_LOCK_PATH = ROOT / "configs/current-dataset-lock.json"
CURRENT_LICENSES_PATH = ROOT / "sources/current-dataset-licenses.json"
WIKIDATA_LOCK_PATH = ROOT / "sources/wikidata5m.lock.json"
WIKIDATA_NOTICE_PATH = ROOT / "sources/Wikidata-CC0-1.0.txt"

_LOCK_FORMAT = "memorysplit-reasoning-v2-source-lock-v1"
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_]*\Z")
_ALLOWED_LICENSES = frozenset(
    {"ODC-By-1.0", "CC0-1.0", "Apache-2.0", "MIT"}
)
_RULETAKER_ARCHIVE = "rule-reasoning-dataset-V2020.2.5.zip"
_RULETAKER_URL = (
    "https://aristo-data-public.s3-us-west-2.amazonaws.com/ruletaker/"
    + _RULETAKER_ARCHIVE
)
_FINEMATH_TARGETS = 1_068_131_943
_FINEWEB_FILES = (
    "sample/10BT/000_00000.parquet",
    "sample/10BT/001_00000.parquet",
    "sample/10BT/002_00000.parquet",
)
_WIKIDATA_FILES = (
    "wikidata5m_alias.tar.gz",
    "wikidata5m_inductive.tar.gz",
    "wikidata5m_transductive.tar.gz",
)

FIXED_WIKIDATA_FILES: dict[str, dict[str, object]] = {
    "wikidata5m_alias.tar.gz": {
        "bytes": 197_449_751,
        "sha256": "0330f580c9f7a57cbad949ac380835fdd2a2e14d96cc0f13fc435401d6b463a8",
    },
    "wikidata5m_inductive.tar.gz": {
        "bytes": 167_247_416,
        "sha256": "955081232cc2de859710bfe3a147f7d8314524010fe5f8c420bb74fdfee4f42a",
    },
    "wikidata5m_transductive.tar.gz": {
        "bytes": 168_258_214,
        "sha256": "383160990b41c0905fc03f4a8afbb9b12be1ca3591e026bde6cdc94a59542597",
    },
}

_FIXED_IDENTITIES: dict[str, tuple[str, str, str, str]] = {
    "fineweb_edu": (
        "huggingface_dataset",
        "HuggingFaceFW/fineweb-edu",
        "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "ODC-By-1.0",
    ),
    "wikidata5m": (
        "huggingface_dataset",
        "intfloat/wikidata5m",
        "6b2b09672129e280c0c9da97ab58154e9d535e6b",
        "CC0-1.0",
    ),
    "arc_agi_1": (
        "git",
        "https://github.com/fchollet/ARC-AGI.git",
        "399030444e0ab0cc8b4e199870fb20b863846f34",
        "Apache-2.0",
    ),
    "arc_agi_2": (
        "git",
        "https://github.com/arcprize/ARC-AGI-2.git",
        "f3283f727488ad98fe575ea6a5ac981e4a188e49",
        "Apache-2.0",
    ),
    "conceptarc": (
        "git",
        "https://github.com/victorvikram/ConceptARC.git",
        "0e67da6af879e4bad3d7cd3c196e8d551b445725",
        "MIT",
    ),
}


def _byte_key(value: str) -> bytes:
    return value.encode("utf-8")


def _safe_relative_path(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != unicodedata.normalize("NFC", value)
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{description} must be a canonical relative POSIX path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"unsafe {description}: {value!r}")
    return value


def _safe_source_id(value: object) -> str:
    if not isinstance(value, str) or _SOURCE_ID_RE.fullmatch(value) is None:
        raise ValueError("source_id must be a lowercase stable identifier")
    return value


def _immutable_revision(kind: object, revision: object) -> tuple[str, str]:
    if kind == "git_commit" and isinstance(revision, str):
        if _COMMIT_RE.fullmatch(revision) is not None:
            return kind, revision
    if kind == "content_sha256" and isinstance(revision, str):
        if _SHA256_RE.fullmatch(revision) is not None:
            return kind, revision
    raise ValueError(
        "source lock requires an immutable revision: a lowercase git commit "
        "or content SHA-256"
    )


def _validate_digest(value: object, description: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


def _validate_commit(value: object, description: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase 40-character commit")
    return value


@dataclass(frozen=True)
class SourceRequest:
    source_id: str
    transport: Literal["huggingface_dataset", "git"]
    repository: str
    required_license_paths: tuple[str, ...] = ()
    required_data_prefixes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _safe_source_id(self.source_id)
        if self.transport not in {"huggingface_dataset", "git"}:
            raise ValueError("source request transport is unsupported")
        if (
            not isinstance(self.repository, str)
            or not self.repository
            or "\x00" in self.repository
        ):
            raise ValueError("source request repository must be nonempty")
        for field_name in ("required_license_paths", "required_data_prefixes"):
            values = getattr(self, field_name)
            if not isinstance(values, tuple):
                raise ValueError(f"{field_name} must be a tuple")
            for value in values:
                _safe_relative_path(value, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} contains duplicate paths")
            if values != tuple(sorted(values, key=_byte_key)):
                raise ValueError(f"{field_name} paths must use bytewise order")


@dataclass(frozen=True)
class SourceFile:
    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _safe_relative_path(self.path, "source file path")
        if type(self.bytes) is not int or self.bytes <= 0:
            raise ValueError("source file bytes must be a positive integer")
        _validate_digest(self.sha256, "source file sha256")

    def as_dict(self) -> dict[str, object]:
        return {
            "bytes": self.bytes,
            "path": self.path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "SourceFile":
        if not isinstance(value, dict) or set(value) != {"bytes", "path", "sha256"}:
            raise ValueError("source file fields do not match the contract")
        return cls(
            path=value["path"],
            bytes=value["bytes"],
            sha256=value["sha256"],
        )


@dataclass(frozen=True)
class SourceEntry:
    source_id: str
    transport: Literal["huggingface_dataset", "git", "https_artifact"]
    repository: str
    revision_kind: Literal["git_commit", "content_sha256"]
    revision: str
    license_spdx: str
    license_files: tuple[str, ...]
    materialized_path: str
    files: tuple[SourceFile, ...]

    def __post_init__(self) -> None:
        _safe_source_id(self.source_id)
        if self.transport not in {
            "huggingface_dataset",
            "git",
            "https_artifact",
        }:
            raise ValueError("source transport is unsupported")
        if (
            not isinstance(self.repository, str)
            or not self.repository
            or "\x00" in self.repository
        ):
            raise ValueError("source repository must be nonempty")
        _immutable_revision(self.revision_kind, self.revision)
        if self.transport in {"huggingface_dataset", "git"}:
            if self.revision_kind != "git_commit":
                raise ValueError("repository source requires an immutable git commit")
        elif self.revision_kind != "content_sha256":
            raise ValueError("HTTPS artifact requires an immutable content SHA-256")
        if self.license_spdx not in _ALLOWED_LICENSES:
            raise ValueError(f"unsupported SPDX license: {self.license_spdx!r}")
        _safe_relative_path(self.materialized_path, "materialized path")
        if not isinstance(self.license_files, tuple) or not self.license_files:
            raise ValueError("source license_files must be a nonempty tuple")
        for path in self.license_files:
            _safe_relative_path(path, "license file path")
        if len(self.license_files) != len(set(self.license_files)):
            raise ValueError("source contains duplicate license file paths")
        if self.license_files != tuple(sorted(self.license_files, key=_byte_key)):
            raise ValueError("source license files must use bytewise path order")
        if not isinstance(self.files, tuple) or not self.files:
            raise ValueError("source files must be a nonempty tuple")
        if not all(isinstance(row, SourceFile) for row in self.files):
            raise ValueError("source files must contain SourceFile values")
        paths = tuple(row.path for row in self.files)
        if len(paths) != len(set(paths)):
            raise ValueError("source contains duplicate file paths")
        if paths != tuple(sorted(paths, key=_byte_key)):
            raise ValueError("source files must use bytewise path order")
        file_parts = {PurePosixPath(path).parts for path in paths}
        for parts in file_parts:
            if any(parts[:depth] in file_parts for depth in range(1, len(parts))):
                raise ValueError(
                    "source file path collides with a directory: "
                    + "/".join(parts)
                )
        missing_licenses = set(self.license_files) - set(paths)
        if missing_licenses:
            raise ValueError(
                "source license file is missing from file inventory: "
                + sorted(missing_licenses, key=_byte_key)[0]
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "files": [row.as_dict() for row in self.files],
            "license_files": list(self.license_files),
            "license_spdx": self.license_spdx,
            "materialized_path": self.materialized_path,
            "repository": self.repository,
            "revision": self.revision,
            "revision_kind": self.revision_kind,
            "source_id": self.source_id,
            "transport": self.transport,
        }

    @classmethod
    def from_dict(cls, value: object) -> "SourceEntry":
        expected = {
            "files",
            "license_files",
            "license_spdx",
            "materialized_path",
            "repository",
            "revision",
            "revision_kind",
            "source_id",
            "transport",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("source entry fields do not match the contract")
        raw_files = value["files"]
        raw_licenses = value["license_files"]
        if not isinstance(raw_files, list) or not isinstance(raw_licenses, list):
            raise ValueError("source entry file inventories must be JSON lists")
        return cls(
            source_id=value["source_id"],
            transport=value["transport"],
            repository=value["repository"],
            revision_kind=value["revision_kind"],
            revision=value["revision"],
            license_spdx=value["license_spdx"],
            license_files=tuple(raw_licenses),
            materialized_path=value["materialized_path"],
            files=tuple(SourceFile.from_dict(row) for row in raw_files),
        )


_EXPECTED_SOURCE_IDS = frozenset(
    {
        "fineweb_edu",
        "finemath",
        "wikidata5m",
        "clrs_text",
        "ruletaker",
        "prontoqa",
        "reasoning_gym_exact_answer",
        "deepmind_mathematics_generator",
        "arc_agi_1",
        "arc_agi_2",
        "conceptarc",
    }
)


@dataclass(frozen=True)
class SourceLock:
    schema_version: Literal[1]
    format: Literal["memorysplit-reasoning-v2-source-lock-v1"]
    dataset_id: str
    dataset_contract_sha256: str
    generator_commit: str
    sources: tuple[SourceEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValueError("source lock schema_version must be integer 1")
        if self.format != _LOCK_FORMAT:
            raise ValueError("source lock format identity mismatch")
        if self.dataset_id != DATASET_ID:
            raise ValueError("source lock dataset identity mismatch")
        _validate_digest(
            self.dataset_contract_sha256,
            "dataset_contract_sha256",
        )
        _validate_commit(self.generator_commit, "generator_commit")
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError("source lock sources must be a nonempty tuple")
        if not all(isinstance(entry, SourceEntry) for entry in self.sources):
            raise ValueError("source lock sources must contain SourceEntry values")
        source_ids = tuple(entry.source_id for entry in self.sources)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source lock contains a duplicate source ID")
        if source_ids != tuple(sorted(source_ids, key=_byte_key)):
            raise ValueError("source lock sources must use bytewise source ID order")
        if set(source_ids) != _EXPECTED_SOURCE_IDS:
            missing = sorted(_EXPECTED_SOURCE_IDS - set(source_ids), key=_byte_key)
            extra = sorted(set(source_ids) - _EXPECTED_SOURCE_IDS, key=_byte_key)
            raise ValueError(
                f"source lock source set mismatch; missing={missing}, extra={extra}"
            )
        materialized = [PurePosixPath(entry.materialized_path) for entry in self.sources]
        if len(materialized) != len(set(materialized)):
            raise ValueError("source lock contains duplicate materialized paths")
        for index, left in enumerate(materialized):
            for right in materialized[index + 1 :]:
                if left in right.parents or right in left.parents:
                    raise ValueError("source lock materialized paths overlap")

    @property
    def sha256(self) -> str:
        return sha256_hex(self.to_bytes())

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_contract_sha256": self.dataset_contract_sha256,
            "dataset_id": self.dataset_id,
            "format": self.format,
            "generator_commit": self.generator_commit,
            "schema_version": self.schema_version,
            "sources": [entry.as_dict() for entry in self.sources],
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    @classmethod
    def from_dict(cls, value: object) -> "SourceLock":
        expected = {
            "dataset_contract_sha256",
            "dataset_id",
            "format",
            "generator_commit",
            "schema_version",
            "sources",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("source lock fields do not match the contract")
        raw_sources = value["sources"]
        if not isinstance(raw_sources, list):
            raise ValueError("source lock sources must be a JSON list")
        return cls(
            schema_version=value["schema_version"],
            format=value["format"],
            dataset_id=value["dataset_id"],
            dataset_contract_sha256=value["dataset_contract_sha256"],
            generator_commit=value["generator_commit"],
            sources=tuple(SourceEntry.from_dict(row) for row in raw_sources),
        )


class SourceResolver(Protocol):
    def resolve(self, request: SourceRequest, download_root: Path) -> SourceEntry:
        raise RuntimeError("SourceResolver implementations must resolve a source")


PUBLIC_REQUESTS = (
    SourceRequest("finemath", "huggingface_dataset", "HuggingFaceTB/finemath"),
    SourceRequest("clrs_text", "git", "https://github.com/google-deepmind/clrs.git"),
    SourceRequest("ruletaker", "git", "https://github.com/allenai/ruletaker.git"),
    SourceRequest("prontoqa", "git", "https://github.com/asaparov/prontoqa.git"),
    SourceRequest(
        "reasoning_gym_exact_answer",
        "git",
        "https://github.com/open-thought/reasoning-gym.git",
    ),
    SourceRequest(
        "deepmind_mathematics_generator",
        "git",
        "https://github.com/google-deepmind/mathematics_dataset.git",
    ),
)

FIXED_REQUESTS = (
    SourceRequest(
        "fineweb_edu",
        "huggingface_dataset",
        "HuggingFaceFW/fineweb-edu",
        required_license_paths=("README.md",),
        required_data_prefixes=_FINEWEB_FILES,
    ),
    SourceRequest(
        "wikidata5m",
        "huggingface_dataset",
        "intfloat/wikidata5m",
        required_license_paths=("Wikidata-CC0-1.0.txt",),
        required_data_prefixes=_WIKIDATA_FILES,
    ),
    SourceRequest(
        "arc_agi_1",
        "git",
        "https://github.com/fchollet/ARC-AGI.git",
        required_license_paths=("LICENSE",),
        required_data_prefixes=("data/training",),
    ),
    SourceRequest(
        "arc_agi_2",
        "git",
        "https://github.com/arcprize/ARC-AGI-2.git",
        required_license_paths=("LICENSE",),
        required_data_prefixes=("data/training",),
    ),
    SourceRequest(
        "conceptarc",
        "git",
        "https://github.com/victorvikram/ConceptARC.git",
        required_license_paths=("LICENSE",),
        required_data_prefixes=("corpus",),
    ),
)

_REQUESTS_BY_ID = {
    request.source_id: request for request in FIXED_REQUESTS + PUBLIC_REQUESTS
}
_RESOLUTION_ORDER = (
    "fineweb_edu",
    "finemath",
    "wikidata5m",
    "clrs_text",
    "ruletaker",
    "prontoqa",
    "reasoning_gym_exact_answer",
    "deepmind_mathematics_generator",
    "arc_agi_1",
    "arc_agi_2",
    "conceptarc",
)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is not allowed: {value}")


def _strict_json_bytes(payload: bytes, description: str) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{description} is not UTF-8 JSON") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} is invalid JSON") from error


def _read_regular_bytes(path: Path, description: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"{description} is missing") from error
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{description} is a symlink")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{description} is not a regular file")
    if before.st_nlink != 1:
        raise ValueError(f"{description} is a hardlink")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    chunks = []
    try:
        opened = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named = path.lstat()
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_nlink)
    if (
        not stat.S_ISREG(opened.st_mode)
        or identity(before) != identity(opened)
        or identity(opened) != identity(after)
        or identity(after) != identity(named)
    ):
        raise ValueError(f"{description} identity changed while reading")
    payload = b"".join(chunks)
    if len(payload) != opened.st_size:
        raise ValueError(f"{description} changed length while reading")
    return payload


def load_source_lock(path: Path) -> SourceLock:
    payload = _read_regular_bytes(Path(path), "source lock")
    value = _strict_json_bytes(payload, "source lock")
    lock = SourceLock.from_dict(value)
    if lock.to_bytes() != payload:
        raise ValueError("source lock JSON is not canonical")
    return lock


def _contract_object(path: Path, description: str) -> dict[str, Any]:
    value = _strict_json_bytes(_read_regular_bytes(path, description), description)
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _require_mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _validate_fixed_contracts() -> None:
    current = _contract_object(CURRENT_DATASET_LOCK_PATH, "current dataset lock")
    sources = _require_mapping(current.get("sources"), "current dataset sources")
    for source_id, (transport, repository, revision, license_spdx) in (
        _FIXED_IDENTITIES.items()
    ):
        source = _require_mapping(
            sources.get(source_id),
            f"fixed {source_id} source",
        )
        actual_revision = source.get(
            "revision" if transport == "huggingface_dataset" else "commit"
        )
        actual_transport = source.get("transport")
        expected_transport = (
            "huggingface" if transport == "huggingface_dataset" else "git"
        )
        if (
            actual_transport != expected_transport
            or source.get("repository") != repository
            or actual_revision != revision
            or source.get("license") != license_spdx
        ):
            label = {
                "fineweb_edu": "FineWeb",
                "wikidata5m": "Wikidata",
                "arc_agi_1": "ARC-AGI",
                "arc_agi_2": "ARC-AGI-2",
                "conceptarc": "ConceptARC",
            }[source_id]
            raise ValueError(f"fixed {label} identity drift in current dataset lock")
    if tuple(sources["fineweb_edu"].get("files", ())) != _FINEWEB_FILES:
        raise ValueError("fixed FineWeb file identity drift")

    wikidata = _contract_object(WIKIDATA_LOCK_PATH, "Wikidata5M source lock")
    if (
        wikidata.get("repo_id") != "intfloat/wikidata5m"
        or wikidata.get("repo_type") != "dataset"
        or wikidata.get("revision") != _FIXED_IDENTITIES["wikidata5m"][2]
        or wikidata.get("files") != FIXED_WIKIDATA_FILES
    ):
        raise ValueError("fixed Wikidata identity drift in source lock")

    licenses = _contract_object(CURRENT_LICENSES_PATH, "current source licenses")
    license_sources = _require_mapping(
        licenses.get("sources"),
        "current source license entries",
    )
    for source_id, (_transport, _repository, _revision, expected) in (
        _FIXED_IDENTITIES.items()
    ):
        row = _require_mapping(
            license_sources.get(source_id),
            f"fixed {source_id} license",
        )
        if row.get("spdx") != expected:
            raise ValueError(f"fixed source license identity drift: {source_id}")
    notice = _read_regular_bytes(WIKIDATA_NOTICE_PATH, "Wikidata CC0 notice")
    if sha256_hex(notice) != (
        "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499"
    ):
        raise ValueError("fixed Wikidata notice identity drift")


def _prefix_present(prefix: str, paths: set[str]) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for path in paths)


def _validate_resolved_entry(
    request: SourceRequest,
    entry: SourceEntry,
) -> None:
    if (
        entry.source_id != request.source_id
        or entry.transport != request.transport
        or entry.repository != request.repository
    ):
        raise ValueError(f"resolved source request identity drift: {request.source_id}")
    paths = {row.path for row in entry.files}
    for path in request.required_license_paths:
        if path not in entry.license_files:
            raise ValueError(
                f"resolved source is missing required license file: "
                f"{request.source_id}:{path}"
            )
    for prefix in request.required_data_prefixes:
        if not _prefix_present(prefix, paths):
            raise ValueError(
                f"resolved source is missing required data prefix: "
                f"{request.source_id}:{prefix}"
            )
    fixed = _FIXED_IDENTITIES.get(request.source_id)
    if fixed is not None:
        transport, repository, revision, license_spdx = fixed
        if (
            entry.transport != transport
            or entry.repository != repository
            or entry.revision_kind != "git_commit"
            or entry.revision != revision
            or entry.license_spdx != license_spdx
        ):
            raise ValueError(
                f"fixed source identity drift: {request.source_id}"
            )
    if request.source_id == "wikidata5m":
        by_path = {row.path: row for row in entry.files}
        for path, expected in FIXED_WIKIDATA_FILES.items():
            row = by_path.get(path)
            if (
                row is None
                or row.bytes != expected["bytes"]
                or row.sha256 != expected["sha256"]
            ):
                raise ValueError(f"fixed source identity drift: wikidata5m:{path}")
    if request.source_id == "finemath":
        if "README.md" not in entry.license_files or not any(
            row.path.startswith("finemath-4plus/train-")
            and row.path.endswith(".parquet")
            for row in entry.files
        ):
            raise ValueError("FineMath source does not prove an ODC-By data snapshot")
    if request.source_id == "ruletaker":
        archive = next(
            (row for row in entry.files if row.path == _RULETAKER_ARCHIVE),
            None,
        )
        if archive is None or not _SHA256_RE.fullmatch(archive.sha256):
            raise ValueError(
                "RuleTaker archive URL has no downloaded content digest"
            )


def _expected_paths(
    lock: SourceLock,
) -> tuple[dict[str, SourceFile], set[str], dict[str, str]]:
    files: dict[str, SourceFile] = {}
    directories = {""}
    licenses: dict[str, str] = {}
    for entry in lock.sources:
        materialized = PurePosixPath(entry.materialized_path)
        for depth in range(1, len(materialized.parts) + 1):
            directories.add("/".join(materialized.parts[:depth]))
        for row in entry.files:
            relative = materialized / PurePosixPath(row.path)
            text = relative.as_posix()
            if text in files:
                raise ValueError(f"duplicate staged source path: {text}")
            files[text] = row
            for depth in range(1, len(relative.parts)):
                directories.add("/".join(relative.parts[:depth]))
        for license_path in entry.license_files:
            licenses[(materialized / license_path).as_posix()] = entry.license_spdx
    return files, directories, licenses


def _read_and_hash_source_file(
    path: Path,
    relative: str,
) -> tuple[int, str]:
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"source file is missing: {relative}") from error
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"source tree contains symlink: {relative}")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"source tree contains special file: {relative}")
    if before.st_nlink != 1:
        raise ValueError(f"source tree contains hardlink: {relative}")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    digest = hashlib.sha256()
    read_bytes = 0
    try:
        opened = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            read_bytes += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named = path.lstat()
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_nlink)
    if (
        not stat.S_ISREG(opened.st_mode)
        or identity(before) != identity(opened)
        or identity(opened) != identity(after)
        or identity(after) != identity(named)
        or read_bytes != opened.st_size
    ):
        raise ValueError(f"source byte drift while reading: {relative}")
    return read_bytes, digest.hexdigest()


def _license_from_bytes(payloads: tuple[bytes, ...]) -> str:
    evidence = b"\n".join(payloads).decode("utf-8", errors="strict")
    lowered = evidence.lower()
    matches = []
    if "license: odc-by" in lowered or "open data commons attribution license" in lowered:
        matches.append("ODC-By-1.0")
    if "creative commons" in lowered and "cc0" in lowered:
        matches.append("CC0-1.0")
    if "apache license" in lowered and "version 2.0" in lowered:
        matches.append("Apache-2.0")
    if "mit license" in lowered or (
        "permission is hereby granted, free of charge" in lowered
        and "the software is provided \"as is\"" in lowered
    ):
        matches.append("MIT")
    if len(matches) != 1:
        raise ValueError("license files do not prove exactly one accepted SPDX value")
    return matches[0]


def _verify_license_files(
    lock: SourceLock,
    source_root: Path,
) -> None:
    for entry in lock.sources:
        payloads = []
        for relative in entry.license_files:
            path = source_root / entry.materialized_path / relative
            metadata = path.lstat()
            if metadata.st_size > 1 << 20:
                raise ValueError(
                    f"license file is unexpectedly large: {entry.source_id}:{relative}"
                )
            payloads.append(
                _read_regular_bytes(
                    path,
                    f"license file {entry.source_id}:{relative}",
                )
            )
        if _license_from_bytes(tuple(payloads)) != entry.license_spdx:
            raise ValueError(f"license file SPDX drift: {entry.source_id}")


def verify_source_tree(
    lock: SourceLock,
    source_root: Path,
) -> dict[str, object]:
    if not isinstance(lock, SourceLock):
        raise TypeError("lock must be a SourceLock")
    root = Path(source_root)
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise ValueError("source root is missing") from error
    if stat.S_ISLNK(root_metadata.st_mode):
        raise ValueError("source root is a symlink")
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("source root is not a directory")
    expected_files, expected_directories, _licenses = _expected_paths(lock)
    actual_files: set[str] = set()
    actual_directories = {""}
    total_bytes = 0

    def visit(directory: Path, prefix: tuple[str, ...]) -> None:
        nonlocal total_bytes
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(
                    iterator,
                    key=lambda entry: _byte_key(entry.name),
                )
        except OSError as error:
            raise ValueError(
                f"source directory cannot be read: {'/'.join(prefix)}"
            ) from error
        for item in entries:
            relative_parts = (*prefix, item.name)
            relative = "/".join(relative_parts)
            metadata = item.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"source tree contains symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                if relative not in expected_directories:
                    raise ValueError(
                        f"source tree contains unlisted directory: {relative}"
                    )
                actual_directories.add(relative)
                visit(Path(item.path), relative_parts)
                continue
            if stat.S_ISREG(metadata.st_mode):
                if relative not in expected_files:
                    raise ValueError(
                        f"source tree contains unlisted file: {relative}"
                    )
                if metadata.st_nlink != 1:
                    raise ValueError(
                        f"source tree contains hardlink: {relative}"
                    )
                expected = expected_files[relative]
                byte_count, digest = _read_and_hash_source_file(
                    Path(item.path),
                    relative,
                )
                if byte_count != expected.bytes or digest != expected.sha256:
                    raise ValueError(f"source byte drift: {relative}")
                actual_files.add(relative)
                total_bytes += byte_count
                continue
            raise ValueError(f"source tree contains special file: {relative}")

    visit(root, ())
    missing_files = sorted(set(expected_files) - actual_files, key=_byte_key)
    if missing_files:
        raise ValueError(f"source tree is missing listed file: {missing_files[0]}")
    missing_directories = sorted(
        expected_directories - actual_directories,
        key=_byte_key,
    )
    if missing_directories:
        raise ValueError(
            f"source tree is missing listed directory: {missing_directories[0]}"
        )
    _verify_license_files(lock, root)
    return {
        "bytes": total_bytes,
        "dataset_id": lock.dataset_id,
        "files": len(actual_files),
        "passed": True,
        "source_lock_sha256": lock.sha256,
    }


def _validate_recipe(recipe: object) -> None:
    if getattr(recipe, "dataset_id", None) != DATASET_ID:
        raise ValueError("recipe dataset identity does not match source-lock contract")
    source_policy = getattr(recipe, "source_policy", None)
    if not isinstance(source_policy, dict):
        raise ValueError("recipe source policy is missing")
    expected = {
        "fineweb_source_lock": "configs/current-dataset-lock.json",
        "wikidata_source_lock": "sources/wikidata5m.lock.json",
    }
    for key, value in expected.items():
        if source_policy.get(key) != value:
            raise ValueError(f"recipe source policy drift: {key}")


def resolve_source_lock(
    recipe: object,
    resolver: SourceResolver,
    download_root: Path,
    generator_commit: str,
) -> SourceLock:
    _validate_recipe(recipe)
    _validate_commit(generator_commit, "generator_commit")
    _validate_fixed_contracts()
    dataset_contract = _read_regular_bytes(
        DATASET_CONTRACT_PATH,
        "reasoning dataset contract",
    )
    root = Path(download_root)
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ValueError("download_root must be a regular directory")
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for source_id in _RESOLUTION_ORDER:
        request = _REQUESTS_BY_ID[source_id]
        try:
            entry = resolver.resolve(request, root)
        except Exception as error:
            raise ValueError(f"unresolved source: {source_id}") from error
        if not isinstance(entry, SourceEntry):
            raise ValueError(f"unresolved source: {source_id}")
        _validate_resolved_entry(request, entry)
        entries.append(entry)
    lock = SourceLock(
        schema_version=1,
        format=_LOCK_FORMAT,
        dataset_id=DATASET_ID,
        dataset_contract_sha256=sha256_hex(dataset_contract),
        generator_commit=generator_commit,
        sources=tuple(sorted(entries, key=lambda entry: _byte_key(entry.source_id))),
    )
    verify_source_tree(lock, root)
    return lock


def _copy_verified_file(source: Path, destination: Path, expected: SourceFile) -> None:
    source_fd = os.open(
        source,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    destination_fd = -1
    digest = hashlib.sha256()
    byte_count = 0
    try:
        source_metadata = os.fstat(source_fd)
        if (
            not stat.S_ISREG(source_metadata.st_mode)
            or source_metadata.st_nlink != 1
        ):
            raise ValueError(f"source staging input is unsafe: {expected.path}")
        destination_fd = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        while True:
            chunk = os.read(source_fd, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
        os.fsync(destination_fd)
        final_source = os.fstat(source_fd)
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        os.close(source_fd)
    if (
        byte_count != expected.bytes
        or digest.hexdigest() != expected.sha256
        or (
            final_source.st_dev,
            final_source.st_ino,
            final_source.st_size,
            final_source.st_nlink,
        )
        != (
            source_metadata.st_dev,
            source_metadata.st_ino,
            source_metadata.st_size,
            source_metadata.st_nlink,
        )
    ):
        raise ValueError(f"source byte drift while staging: {expected.path}")


def _fsync_tree_directories(root: Path, directories: set[str]) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for relative in sorted(
        directories,
        key=lambda value: (-len(PurePosixPath(value).parts), _byte_key(value)),
    ):
        path = root if not relative else root / relative
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def stage_source_lock(
    lock: SourceLock,
    download_root: Path,
    canonical_root: Path,
) -> Path:
    verify_source_tree(lock, Path(download_root))
    canonical = Path(canonical_root)
    canonical_fd = open_directory_path(canonical, create=True, mode=0o700)
    sources_fd = -1
    try:
        sources_fd, _created = open_directory_at(
            canonical_fd,
            "sources",
            create=True,
            mode=0o700,
        )
        final_name = lock.sha256
        final_path = canonical / "sources" / final_name
        if entry_exists(sources_fd, final_name):
            try:
                verify_source_tree(lock, final_path)
            except ValueError as error:
                raise ValueError(
                    f"conflicting source stage: {final_path}"
                ) from error
            return final_path

        stage_name = (
            f".{final_name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
        )
        os.mkdir(stage_name, mode=0o700, dir_fd=sources_fd)
        stage_path = canonical / "sources" / stage_name
        expected_files, directories, _licenses = _expected_paths(lock)
        try:
            for relative in sorted(
                directories - {""},
                key=lambda value: (
                    len(PurePosixPath(value).parts),
                    _byte_key(value),
                ),
            ):
                (stage_path / relative).mkdir(mode=0o700)
            for relative, expected in sorted(
                expected_files.items(),
                key=lambda item: _byte_key(item[0]),
            ):
                _copy_verified_file(
                    Path(download_root) / relative,
                    stage_path / relative,
                    expected,
                )
            _fsync_tree_directories(stage_path, directories)
            verify_source_tree(lock, stage_path)
            try:
                atomic_rename_noreplace(
                    sources_fd,
                    stage_name,
                    sources_fd,
                    final_name,
                )
            except FileExistsError:
                try:
                    verify_source_tree(lock, final_path)
                except ValueError as error:
                    raise ValueError(
                        f"conflicting source stage: {final_path}"
                    ) from error
                shutil.rmtree(stage_path)
            else:
                fsync_directory(sources_fd)
            verify_source_tree(lock, final_path)
            return final_path
        except BaseException:
            if stage_path.exists() and not stage_path.is_symlink():
                shutil.rmtree(stage_path)
            raise
    finally:
        if sources_fd >= 0:
            os.close(sources_fd)
        os.close(canonical_fd)


def _inventory_directory(root: Path) -> tuple[SourceFile, ...]:
    rows = []
    for directory, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        directory_names.sort(key=_byte_key)
        file_names.sort(key=_byte_key)
        for name in tuple(directory_names):
            path = Path(directory) / name
            metadata = path.lstat()
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"resolved source contains symlink: {relative}")
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"resolved source contains special file: {relative}")
        for name in file_names:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            byte_count, digest = _read_and_hash_source_file(path, relative)
            if byte_count <= 0:
                raise ValueError(f"resolved source contains an empty file: {relative}")
            rows.append(SourceFile(relative, byte_count, digest))
    return tuple(sorted(rows, key=lambda row: _byte_key(row.path)))


def _detect_license(root: Path, paths: tuple[str, ...]) -> str:
    payloads = tuple(
        _read_regular_bytes(root / path, f"resolved license file {path}")
        for path in paths
    )
    return _license_from_bytes(payloads)


def _safe_extract_git_archive(archive_path: Path, destination: Path) -> None:
    seen: set[str] = set()
    with tarfile.open(archive_path, mode="r:") as archive:
        for member in archive.getmembers():
            raw_name = member.name[:-1] if member.isdir() else member.name
            relative = _safe_relative_path(raw_name, "git archive member")
            if relative in seen:
                raise ValueError(f"git archive contains duplicate path: {relative}")
            seen.add(relative)
            target = destination / relative
            if member.isdir():
                continue
            if not member.isreg():
                raise ValueError(f"git archive contains link or special file: {relative}")
            if member.size == 0:
                continue
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"git archive member cannot be read: {relative}")
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                while True:
                    chunk = source.read(1 << 20)
                    if not chunk:
                        break
                    view = memoryview(chunk)
                    while view:
                        written = os.write(descriptor, view)
                        view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
                source.close()


def _run_git(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
    }
    return subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )


def _download_url(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MemorySplit-source-freezer/1"},
    )
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    byte_count = 0
    try:
        with urllib.request.urlopen(request) as response:
            final_url = response.geturl()
            if final_url != url:
                raise ValueError("RuleTaker dataset URL redirected unexpectedly")
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                byte_count += len(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if byte_count <= 0:
        raise ValueError("RuleTaker dataset archive is empty")


class PublicSourceResolver:
    """Resolve only the reviewed public catalog, never ambient HF credentials."""

    def __init__(self) -> None:
        self._hf_api = HfApi(token=False)

    def resolve(self, request: SourceRequest, download_root: Path) -> SourceEntry:
        if not isinstance(request, SourceRequest):
            raise TypeError("request must be a SourceRequest")
        root = Path(download_root)
        root.mkdir(parents=True, exist_ok=True)
        materialized = root / request.source_id
        if materialized.exists() or materialized.is_symlink():
            raise ValueError(
                f"source materialization already exists: {request.source_id}"
            )
        materialized.mkdir(mode=0o700)
        try:
            if request.transport == "git":
                entry = self._resolve_git(request, root, materialized)
            elif request.transport == "huggingface_dataset":
                entry = self._resolve_huggingface(request, root, materialized)
            else:
                raise ValueError(f"unsupported source transport: {request.transport}")
            _validate_resolved_entry(request, entry)
            return entry
        except BaseException:
            if materialized.exists() and not materialized.is_symlink():
                shutil.rmtree(materialized)
            raise

    def _resolve_git(
        self,
        request: SourceRequest,
        download_root: Path,
        materialized: Path,
    ) -> SourceEntry:
        fixed = _FIXED_IDENTITIES.get(request.source_id)
        if fixed is None:
            result = _run_git(["ls-remote", request.repository, "HEAD"])
            rows = [line.split() for line in result.stdout.splitlines() if line.strip()]
            if (
                len(rows) != 1
                or len(rows[0]) != 2
                or rows[0][1] != "HEAD"
                or _COMMIT_RE.fullmatch(rows[0][0]) is None
            ):
                raise ValueError(
                    f"git HEAD did not resolve to one immutable commit: "
                    f"{request.source_id}"
                )
            revision = rows[0][0]
        else:
            revision = fixed[2]
        download_root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{download_root.name}-{request.source_id}-git-",
            dir=download_root.parent,
        ) as temporary:
            temporary_root = Path(temporary)
            cache = temporary_root / "cache.git"
            archive = temporary_root / "source.tar"
            _run_git(["init", "--bare", "-q", str(cache)])
            _run_git(
                [
                    "--git-dir",
                    str(cache),
                    "fetch",
                    "--no-tags",
                    "--depth=1",
                    request.repository,
                    revision,
                ]
            )
            fetched = _run_git(
                [
                    "--git-dir",
                    str(cache),
                    "rev-parse",
                    "FETCH_HEAD^{commit}",
                ]
            ).stdout.strip()
            if fetched != revision:
                raise ValueError(f"git fetched commit identity drift: {request.source_id}")
            _run_git(
                [
                    "--git-dir",
                    str(cache),
                    "archive",
                    "--format=tar",
                    f"--output={archive}",
                    revision,
                ]
            )
            _safe_extract_git_archive(archive, materialized)
        if request.source_id == "ruletaker":
            readme = _read_regular_bytes(
                materialized / "README.md",
                "RuleTaker README",
            ).decode("utf-8")
            urls = set(re.findall(re.escape(_RULETAKER_URL), readme))
            if urls != {_RULETAKER_URL}:
                raise ValueError(
                    "RuleTaker README does not contain the one official dataset URL"
                )
            _download_url(_RULETAKER_URL, materialized / _RULETAKER_ARCHIVE)
        license_paths = request.required_license_paths or ("LICENSE",)
        license_spdx = _detect_license(materialized, license_paths)
        files = _inventory_directory(materialized)
        return SourceEntry(
            source_id=request.source_id,
            transport="git",
            repository=request.repository,
            revision_kind="git_commit",
            revision=revision,
            license_spdx=license_spdx,
            license_files=license_paths,
            materialized_path=request.source_id,
            files=files,
        )

    def _hf_info(self, request: SourceRequest, revision: str | None) -> Any:
        arguments: dict[str, object] = {"files_metadata": True}
        if revision is not None:
            arguments["revision"] = revision
        info = self._hf_api.dataset_info(request.repository, **arguments)
        if _COMMIT_RE.fullmatch(getattr(info, "sha", "")) is None:
            raise ValueError(
                f"Hugging Face source did not resolve to an immutable commit: "
                f"{request.source_id}"
            )
        if revision is not None and info.sha != revision:
            raise ValueError(
                f"fixed Hugging Face revision drift: {request.source_id}"
            )
        return info

    def _download_hf_file(
        self,
        request: SourceRequest,
        revision: str,
        relative: str,
        materialized: Path,
        cache: Path,
    ) -> None:
        source_path = Path(
            hf_hub_download(
                repo_id=request.repository,
                filename=relative,
                repo_type="dataset",
                revision=revision,
                token=False,
                cache_dir=cache,
            )
        )
        destination = materialized / relative
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with source_path.open("rb") as source:
                while True:
                    chunk = source.read(1 << 20)
                    if not chunk:
                        break
                    view = memoryview(chunk)
                    while view:
                        written = os.write(descriptor, view)
                        view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _resolve_huggingface(
        self,
        request: SourceRequest,
        download_root: Path,
        materialized: Path,
    ) -> SourceEntry:
        fixed = _FIXED_IDENTITIES.get(request.source_id)
        expected_revision = fixed[2] if fixed is not None else None
        info = self._hf_info(request, expected_revision)
        revision = info.sha
        download_root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{download_root.name}-{request.source_id}-hf-",
            dir=download_root.parent,
        ) as temporary:
            cache = Path(temporary) / "cache"
            if request.source_id == "finemath":
                self._download_finemath(
                    request,
                    info,
                    revision,
                    download_root,
                    materialized,
                    cache,
                )
                license_paths = ("README.md",)
                license_spdx = "ODC-By-1.0"
            else:
                selected = list(request.required_data_prefixes)
                if request.source_id == "fineweb_edu":
                    selected.insert(0, "README.md")
                    license_paths = ("README.md",)
                elif request.source_id == "wikidata5m":
                    license_paths = ("Wikidata-CC0-1.0.txt",)
                else:
                    raise ValueError(
                        f"unreviewed Hugging Face source: {request.source_id}"
                    )
                for relative in selected:
                    self._download_hf_file(
                        request,
                        revision,
                        relative,
                        materialized,
                        cache,
                    )
                if request.source_id == "wikidata5m":
                    shutil.copyfile(
                        WIKIDATA_NOTICE_PATH,
                        materialized / "Wikidata-CC0-1.0.txt",
                    )
                license_spdx = _detect_license(materialized, license_paths)
        files = _inventory_directory(materialized)
        return SourceEntry(
            source_id=request.source_id,
            transport="huggingface_dataset",
            repository=request.repository,
            revision_kind="git_commit",
            revision=revision,
            license_spdx=license_spdx,
            license_files=license_paths,
            materialized_path=request.source_id,
            files=files,
        )

    def _download_finemath(
        self,
        request: SourceRequest,
        info: Any,
        revision: str,
        download_root: Path,
        materialized: Path,
        cache: Path,
    ) -> None:
        card_data = getattr(info, "card_data", None)
        card_license = card_data.get("license") if card_data is not None else None
        if card_license not in {"odc-by", "ODC-By-1.0"}:
            raise ValueError("FineMath metadata does not prove ODC-By-1.0")
        sibling_paths = sorted(
            {
                getattr(sibling, "rfilename", "")
                for sibling in getattr(info, "siblings", ())
            },
            key=_byte_key,
        )
        four_plus = [
            path
            for path in sibling_paths
            if path.startswith("finemath-4plus/train-")
            and path.endswith(".parquet")
        ]
        three_plus = [
            path
            for path in sibling_paths
            if path.startswith("finemath-3plus/train-")
            and path.endswith(".parquet")
        ]
        if "README.md" not in sibling_paths or not four_plus:
            raise ValueError("FineMath repository inventory is incomplete")
        self._download_hf_file(
            request,
            revision,
            "README.md",
            materialized,
            cache,
        )
        if _detect_license(materialized, ("README.md",)) != "ODC-By-1.0":
            raise ValueError("FineMath README does not prove ODC-By-1.0")

        try:
            import pyarrow.parquet as parquet
            from train.tokenizer import get_tok
        except ImportError as error:
            raise RuntimeError(
                "FineMath source resolution requires pyarrow and vendored tokenizer"
            ) from error

        fineweb_root = download_root / "fineweb_edu"
        for relative in _FINEWEB_FILES:
            if not (fineweb_root / relative).is_file():
                raise ValueError(
                    "FineMath resolution requires the fixed FineWeb bytes first"
                )
        with tempfile.TemporaryDirectory(
            prefix=".finemath-proof-",
            dir=download_root.parent,
        ) as proof_directory:
            database = sqlite3.connect(
                str(Path(proof_directory) / "fineweb.sqlite3")
            )
            try:
                database.execute("PRAGMA journal_mode=OFF")
                database.execute("PRAGMA synchronous=OFF")
                database.execute(
                    "CREATE TABLE fineweb ("
                    "digest BLOB NOT NULL, text TEXT NOT NULL, "
                    "PRIMARY KEY (digest, text)"
                    ") WITHOUT ROWID"
                )
                for relative in _FINEWEB_FILES:
                    for text in _iter_parquet_texts(
                        parquet,
                        fineweb_root / relative,
                    ):
                        normalized = unicodedata.normalize("NFC", text)
                        database.execute(
                            "INSERT OR IGNORE INTO fineweb(digest, text) VALUES (?, ?)",
                            (
                                hashlib.sha256(normalized.encode("utf-8")).digest(),
                                normalized,
                            ),
                        )
                    database.commit()
                tokenizer = get_tok()
                token_count = 0
                for relative in [*four_plus, *three_plus]:
                    self._download_hf_file(
                        request,
                        revision,
                        relative,
                        materialized,
                        cache,
                    )
                    for text in _iter_parquet_texts(
                        parquet,
                        materialized / relative,
                    ):
                        normalized = unicodedata.normalize("NFC", text)
                        digest = hashlib.sha256(
                            normalized.encode("utf-8")
                        ).digest()
                        duplicate = database.execute(
                            "SELECT 1 FROM fineweb WHERE digest=? AND text=?",
                            (digest, normalized),
                        ).fetchone()
                        if duplicate is None:
                            token_count += len(tokenizer.encode(normalized)) + 1
                    if token_count > _FINEMATH_TARGETS:
                        break
                if token_count <= _FINEMATH_TARGETS:
                    raise ValueError(
                        "FineMath bytes do not prove more than the frozen quota "
                        "after exact FineWeb cross-deduplication"
                    )
            finally:
                database.close()


def _iter_parquet_texts(parquet_module: Any, path: Path):
    parquet_file = parquet_module.ParquetFile(path)
    if "text" not in parquet_file.schema.names:
        raise ValueError(f"source parquet has no text column: {path}")
    for batch in parquet_file.iter_batches(batch_size=1024, columns=["text"]):
        for value in batch.column(0).to_pylist():
            if not isinstance(value, str):
                raise ValueError(f"source parquet text is not a string: {path}")
            yield value
