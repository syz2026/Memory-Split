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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from huggingface_hub import HfApi, hf_hub_download

from corpusgen.parallel.canonical import canonical_json_bytes, sha256_hex
from corpusgen.parallel.safeio import (
    atomic_rename_noreplace,
    entry_lstat,
    entry_exists,
    fsync_directory,
    list_entries,
    open_directory_at,
    open_parent_directory,
    open_regular_file_at,
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
_FINEMATH_SELECTION_ALGORITHM = "nfc-fineweb-exact-dedup-gpt2-eot-v1"
_FINEMATH_SHARD_RE = re.compile(
    r"train-(?P<index>[0-9]{5})-of-(?P<count>[0-9]{5})\.parquet\Z"
)
_CLEANUP_QUARANTINE_PREFIX = ".memorysplit-source-cleanup-"
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
FIXED_FINEWEB_FILES: dict[str, dict[str, object]] = {
    "sample/10BT/000_00000.parquet": {
        "bytes": 2_152_819_114,
        "sha256": "b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871",
    },
    "sample/10BT/001_00000.parquet": {
        "bytes": 2_152_222_432,
        "sha256": "3fcf2dc69cd52503986276d3d2d26a8c356d0f2ea28a0de4fdbda8cf87755693",
    },
    "sample/10BT/002_00000.parquet": {
        "bytes": 2_151_796_315,
        "sha256": "547ae182d132c9f06b6ce63149567208ea9f57630bfd9b1a2938e504f0c9ebd7",
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
_REVIEWED_LICENSES = {
    "fineweb_edu": "ODC-By-1.0",
    "finemath": "ODC-By-1.0",
    "wikidata5m": "CC0-1.0",
    "clrs_text": "Apache-2.0",
    "ruletaker": "Apache-2.0",
    "prontoqa": "Apache-2.0",
    "reasoning_gym_exact_answer": "Apache-2.0",
    "deepmind_mathematics_generator": "Apache-2.0",
    "arc_agi_1": "Apache-2.0",
    "arc_agi_2": "Apache-2.0",
    "conceptarc": "MIT",
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
    required_data_paths: tuple[str, ...] = ()

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
        for field_name in (
            "required_license_paths",
            "required_data_paths",
            "required_data_prefixes",
        ):
            values = getattr(self, field_name)
            if not isinstance(values, tuple):
                raise ValueError(f"{field_name} must be a tuple")
            for value in values:
                _safe_relative_path(value, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} contains duplicate paths")
            if values != tuple(sorted(values, key=_byte_key)):
                raise ValueError(f"{field_name} paths must use bytewise order")

    def as_dict(self) -> dict[str, object]:
        return {
            "repository": self.repository,
            "required_data_paths": list(self.required_data_paths),
            "required_data_prefixes": list(self.required_data_prefixes),
            "required_license_paths": list(self.required_license_paths),
            "source_id": self.source_id,
            "transport": self.transport,
        }


@dataclass(frozen=True)
class SourceFile:
    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _safe_relative_path(self.path, "source file path")
        if type(self.bytes) is not int or self.bytes < 0:
            raise ValueError("source file bytes must be a non-negative integer")
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


def _validate_finemath_shard_sequence(
    paths: tuple[str, ...],
    subset: str,
) -> None:
    if not paths or paths != tuple(sorted(paths, key=_byte_key)):
        raise ValueError(f"FineMath {subset} shard sequence is not canonical")
    counts = set()
    indices = []
    for path in paths:
        relative = PurePosixPath(path)
        match = (
            _FINEMATH_SHARD_RE.fullmatch(relative.name)
            if relative.parent.as_posix() == subset
            else None
        )
        if match is None:
            raise ValueError(f"FineMath {subset} shard path is invalid: {path}")
        counts.add(int(match.group("count")))
        indices.append(int(match.group("index")))
    if len(counts) != 1:
        raise ValueError(f"FineMath {subset} shard counts disagree")
    count = counts.pop()
    if count <= 0 or len(paths) != count or indices != list(range(count)):
        raise ValueError(f"FineMath {subset} shard sequence is incomplete")


@dataclass(frozen=True)
class FineMathSelectionProof:
    algorithm: Literal["nfc-fineweb-exact-dedup-gpt2-eot-v1"]
    quota: int
    four_plus_paths: tuple[str, ...]
    three_plus_paths: tuple[str, ...]
    selected_paths: tuple[str, ...]
    usable_targets: int
    fineweb_duplicate_rows: int

    def __post_init__(self) -> None:
        if self.algorithm != _FINEMATH_SELECTION_ALGORITHM:
            raise ValueError("FineMath selection algorithm identity drift")
        if type(self.quota) is not int or self.quota < 0:
            raise ValueError("FineMath selection quota must be non-negative")
        _validate_finemath_shard_sequence(
            self.four_plus_paths,
            "finemath-4plus",
        )
        _validate_finemath_shard_sequence(
            self.three_plus_paths,
            "finemath-3plus",
        )
        combined = (*self.four_plus_paths, *self.three_plus_paths)
        if (
            not self.selected_paths
            or self.selected_paths != combined[: len(self.selected_paths)]
        ):
            raise ValueError("FineMath selected shard sequence is not a prefix")
        if type(self.usable_targets) is not int or self.usable_targets <= self.quota:
            raise ValueError("FineMath selection does not exceed its quota")
        if (
            type(self.fineweb_duplicate_rows) is not int
            or self.fineweb_duplicate_rows < 0
        ):
            raise ValueError("FineMath duplicate-row count must be non-negative")

    def as_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "fineweb_duplicate_rows": self.fineweb_duplicate_rows,
            "four_plus_paths": list(self.four_plus_paths),
            "quota": self.quota,
            "selected_paths": list(self.selected_paths),
            "three_plus_paths": list(self.three_plus_paths),
            "usable_targets": self.usable_targets,
        }

    @classmethod
    def from_dict(cls, value: object) -> "FineMathSelectionProof":
        expected = {
            "algorithm",
            "fineweb_duplicate_rows",
            "four_plus_paths",
            "quota",
            "selected_paths",
            "three_plus_paths",
            "usable_targets",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("FineMath selection proof fields do not match")
        for field in ("four_plus_paths", "three_plus_paths", "selected_paths"):
            if not isinstance(value[field], list):
                raise ValueError(f"FineMath selection {field} must be a list")
        return cls(
            algorithm=value["algorithm"],
            quota=value["quota"],
            four_plus_paths=tuple(value["four_plus_paths"]),
            three_plus_paths=tuple(value["three_plus_paths"]),
            selected_paths=tuple(value["selected_paths"]),
            usable_targets=value["usable_targets"],
            fineweb_duplicate_rows=value["fineweb_duplicate_rows"],
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
    finemath_selection: FineMathSelectionProof | None = None

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
        if self.source_id == "finemath":
            if not isinstance(self.finemath_selection, FineMathSelectionProof):
                raise ValueError("FineMath source requires a selection proof")
        elif self.finemath_selection is not None:
            raise ValueError("FineMath selection proof is only valid for FineMath")

    def as_dict(self) -> dict[str, object]:
        return {
            "files": [row.as_dict() for row in self.files],
            "finemath_selection": (
                None
                if self.finemath_selection is None
                else self.finemath_selection.as_dict()
            ),
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
            "finemath_selection",
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
            finemath_selection=(
                None
                if value["finemath_selection"] is None
                else FineMathSelectionProof.from_dict(
                    value["finemath_selection"]
                )
            ),
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
    source_catalog_sha256: str
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
        _validate_digest(self.source_catalog_sha256, "source_catalog_sha256")
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
            "source_catalog_sha256": self.source_catalog_sha256,
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
            "source_catalog_sha256",
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
            source_catalog_sha256=value["source_catalog_sha256"],
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
        required_data_paths=_FINEWEB_FILES,
    ),
    SourceRequest(
        "wikidata5m",
        "huggingface_dataset",
        "intfloat/wikidata5m",
        required_license_paths=("Wikidata-CC0-1.0.txt",),
        required_data_paths=_WIKIDATA_FILES,
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


def _reviewed_license_paths(request: SourceRequest) -> tuple[str, ...]:
    if request.required_license_paths:
        return request.required_license_paths
    if request.source_id == "finemath":
        return ("README.md",)
    return ("LICENSE",)


def _reviewed_fixed_files(source_id: str) -> dict[str, dict[str, object]]:
    if source_id == "fineweb_edu":
        return FIXED_FINEWEB_FILES
    if source_id == "wikidata5m":
        return FIXED_WIKIDATA_FILES
    return {}


def _reviewed_catalog_dict() -> dict[str, object]:
    sources = []
    for source_id in sorted(_REQUESTS_BY_ID, key=_byte_key):
        request = _REQUESTS_BY_ID[source_id]
        fixed = _FIXED_IDENTITIES.get(source_id)
        sources.append(
            {
                "fixed_files": [
                    {
                        "bytes": row["bytes"],
                        "path": path,
                        "sha256": row["sha256"],
                    }
                    for path, row in sorted(
                        _reviewed_fixed_files(source_id).items(),
                        key=lambda item: _byte_key(item[0]),
                    )
                ],
                "fixed_revision": None if fixed is None else fixed[2],
                "finemath_selection_policy": (
                    {
                        "algorithm": _FINEMATH_SELECTION_ALGORITHM,
                        "quota": _FINEMATH_TARGETS,
                    }
                    if source_id == "finemath"
                    else None
                ),
                "license_files": list(_reviewed_license_paths(request)),
                "license_spdx": _REVIEWED_LICENSES[source_id],
                "materialized_path": source_id,
                "request": request.as_dict(),
                "revision_kind": "git_commit",
                "source_id": source_id,
            }
        )
    return {
        "dataset_id": DATASET_ID,
        "format": "memorysplit-reasoning-v2-reviewed-source-catalog-v1",
        "sources": sources,
    }


def reviewed_source_catalog_sha256() -> str:
    return sha256_hex(canonical_json_bytes(_reviewed_catalog_dict()))


def _require_reviewed_request(request: SourceRequest) -> None:
    expected = _REQUESTS_BY_ID.get(request.source_id)
    if expected is None or request != expected:
        raise ValueError(
            f"source request is not in the reviewed source catalog: "
            f"{request.source_id}"
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


def load_source_lock(
    path: Path,
    *,
    expected_generator_commit: str,
) -> SourceLock:
    payload = _read_regular_bytes(Path(path), "source lock")
    value = _strict_json_bytes(payload, "source lock")
    lock = SourceLock.from_dict(value)
    if lock.to_bytes() != payload:
        raise ValueError("source lock JSON is not canonical")
    _authorize_source_lock(
        lock,
        expected_generator_commit=expected_generator_commit,
    )
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
    return any(path.startswith(prefix + "/") for path in paths)


def _is_finemath_train_file(path: str, subset: str) -> bool:
    relative = PurePosixPath(path)
    return (
        relative.parent.as_posix() == subset
        and relative.name.startswith("train-")
        and relative.name.endswith(".parquet")
    )


def _validate_resolved_entry(
    request: SourceRequest,
    entry: SourceEntry,
) -> None:
    _require_reviewed_request(request)
    if (
        entry.source_id != request.source_id
        or entry.transport != request.transport
        or entry.repository != request.repository
    ):
        raise ValueError(
            f"reviewed source catalog identity drift: {request.source_id}"
        )
    if (
        entry.revision_kind != "git_commit"
        or entry.license_spdx != _REVIEWED_LICENSES[request.source_id]
        or entry.license_files != _reviewed_license_paths(request)
        or entry.materialized_path != request.source_id
    ):
        raise ValueError(
            f"reviewed source catalog semantics drift: {request.source_id}"
        )
    paths = {row.path for row in entry.files}
    for path in request.required_license_paths:
        if path not in entry.license_files:
            raise ValueError(
                f"resolved source is missing required license file: "
                f"{request.source_id}:{path}"
            )
    for path in request.required_data_paths:
        if path not in paths:
            raise ValueError(
                f"resolved source is missing required data file: "
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
    by_path = {row.path: row for row in entry.files}
    for path, expected in _reviewed_fixed_files(request.source_id).items():
        row = by_path.get(path)
        if (
            row is None
            or row.bytes != expected["bytes"]
            or row.sha256 != expected["sha256"]
        ):
            raise ValueError(
                f"fixed source file identity drift: "
                f"{request.source_id}:{path}"
            )
    if request.source_id == "finemath":
        proof = entry.finemath_selection
        if (
            proof is None
            or proof.algorithm != _FINEMATH_SELECTION_ALGORITHM
            or proof.quota != _FINEMATH_TARGETS
        ):
            raise ValueError("FineMath source selection proof identity drift")
        allowed_paths = {"README.md", *proof.selected_paths}
        if (
            entry.license_files != ("README.md",)
            or paths != allowed_paths
        ):
            raise ValueError("FineMath source inventory is not exactly selected")
        if not all(
            _is_finemath_train_file(path, "finemath-4plus")
            or _is_finemath_train_file(path, "finemath-3plus")
            for path in proof.selected_paths
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


def _authorize_source_lock(
    lock: SourceLock,
    *,
    expected_generator_commit: str,
) -> None:
    if not isinstance(lock, SourceLock):
        raise TypeError("lock must be a SourceLock")
    _validate_commit(expected_generator_commit, "expected generator commit")
    _validate_fixed_contracts()
    expected_contract_sha256 = sha256_hex(
        _read_regular_bytes(
            DATASET_CONTRACT_PATH,
            "reasoning dataset contract",
        )
    )
    if lock.dataset_contract_sha256 != expected_contract_sha256:
        raise ValueError("source lock dataset contract digest drift")
    if lock.generator_commit != expected_generator_commit:
        raise ValueError("source lock generator commit does not match expectation")
    if lock.source_catalog_sha256 != reviewed_source_catalog_sha256():
        raise ValueError("source lock reviewed source catalog commitment drift")
    by_id = {entry.source_id: entry for entry in lock.sources}
    for source_id in _RESOLUTION_ORDER:
        _validate_resolved_entry(
            _REQUESTS_BY_ID[source_id],
            by_id[source_id],
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


def _namespace_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        stat.S_IMODE(metadata.st_mode),
    )


def _directory_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int, int | None, int | None]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_size,
        getattr(metadata, "st_mtime_ns", None),
        getattr(metadata, "st_ctime_ns", None),
    )


def _file_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int, int | None, int | None]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_size,
        getattr(metadata, "st_mtime_ns", None),
        getattr(metadata, "st_ctime_ns", None),
    )


def _regular_inode_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
    )


def _require_owned_mode(
    metadata: os.stat_result,
    *,
    directory: bool,
    description: str,
) -> None:
    expected_kind = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_kind(metadata.st_mode):
        kind = "directory" if directory else "regular file"
        raise ValueError(f"{description} is not a {kind}")
    if metadata.st_uid != os.geteuid():
        raise ValueError(f"{description} owner drift")
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise ValueError(f"{description} mode is group/world writable")
    if not directory and metadata.st_nlink != 1:
        raise ValueError(f"{description} is a hardlink")


def _open_bound_directory(
    parent_fd: int,
    name: str,
    *,
    description: str,
    create: bool = False,
    mode: int = 0o700,
) -> tuple[int, tuple[int, int, int, int]]:
    try:
        named_before = entry_lstat(parent_fd, name)
    except FileNotFoundError:
        if not create:
            raise
        named_before = None
    directory_fd, _created = open_directory_at(
        parent_fd,
        name,
        create=create,
        mode=mode,
    )
    try:
        opened = os.fstat(directory_fd)
        named_after = entry_lstat(parent_fd, name)
        _require_owned_mode(
            opened,
            directory=True,
            description=description,
        )
        identity = _namespace_identity(opened)
        if (
            (named_before is not None and not stat.S_ISDIR(named_before.st_mode))
            or _namespace_identity(named_after) != identity
            or (
                named_before is not None
                and _namespace_identity(named_before) != identity
            )
        ):
            raise ValueError(f"{description} identity drift")
        return directory_fd, identity
    except BaseException:
        os.close(directory_fd)
        raise


def _require_named_directory_identity(
    parent_fd: int,
    name: str,
    directory_fd: int,
    expected: tuple[int, int, int, int],
    *,
    description: str,
) -> None:
    opened = os.fstat(directory_fd)
    named = entry_lstat(parent_fd, name)
    _require_owned_mode(opened, directory=True, description=description)
    if (
        _namespace_identity(opened) != expected
        or not stat.S_ISDIR(named.st_mode)
        or _namespace_identity(named) != expected
    ):
        raise ValueError(f"{description} identity drift")


def _iter_parquet_texts_from_descriptor(
    descriptor: int,
    description: str,
) -> Iterable[str]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "FineMath verification requires pyarrow"
        ) from error
    duplicate = os.dup(descriptor)
    os.lseek(duplicate, 0, os.SEEK_SET)
    with os.fdopen(duplicate, "rb") as handle:
        parquet_file = parquet.ParquetFile(handle)
        if "text" not in parquet_file.schema.names:
            raise ValueError(f"source parquet has no text column: {description}")
        for batch in parquet_file.iter_batches(
            batch_size=1024,
            columns=["text"],
        ):
            for value in batch.column(0).to_pylist():
                if not isinstance(value, str):
                    raise ValueError(
                        f"source parquet text is not a string: {description}"
                    )
                yield value


def _encode_finemath_text(text: str) -> list[int]:
    from train.tokenizer import get_tok

    return get_tok().encode(text)


def _verify_finemath_selection_descriptors(
    lock: SourceLock,
    descriptors: dict[
        str,
        tuple[int, tuple[int, int, int, int, int, int, int | None, int | None]],
    ],
) -> None:
    entry = next(row for row in lock.sources if row.source_id == "finemath")
    proof = entry.finemath_selection
    if proof is None:
        raise ValueError("FineMath selection proof is missing")
    expected_paths = {
        *(f"fineweb_edu/{path}" for path in _FINEWEB_FILES),
        *(f"finemath/{path}" for path in proof.selected_paths),
    }
    if set(descriptors) != expected_paths:
        raise ValueError("FineMath selection proof descriptors are incomplete")
    with tempfile.TemporaryDirectory(
        prefix=".finemath-verify-",
    ) as temporary:
        try:
            computed = _select_finemath_files(
                four_plus=proof.four_plus_paths,
                three_plus=proof.three_plus_paths,
                fineweb_paths=tuple(
                    f"fineweb_edu/{path}" for path in _FINEWEB_FILES
                ),
                materialize=lambda relative: f"finemath/{relative}",
                iter_texts=lambda relative: _iter_parquet_texts_from_descriptor(
                    descriptors[relative][0],
                    relative,
                ),
                encode=_encode_finemath_text,
                quota=proof.quota,
                database_path=Path(temporary) / "fineweb.sqlite3",
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(
                "FineMath selection proof verification failed"
            ) from error
    if computed != proof:
        raise ValueError("FineMath selection proof drift")
    for relative, (descriptor, identity) in descriptors.items():
        if _file_identity(os.fstat(descriptor)) != identity:
            raise ValueError(
                f"FineMath selection proof file identity drift: {relative}"
            )


@dataclass(frozen=True)
class _SourceTreeSnapshot:
    lock_sha256: str
    directories: tuple[
        tuple[
            str,
            tuple[int, int, int, int, int, int, int | None, int | None],
            tuple[str, ...],
        ],
        ...,
    ]
    files: tuple[
        tuple[
            str,
            tuple[int, int, int, int, int, int, int | None, int | None],
            str,
        ],
        ...,
    ]


def _finemath_proof_hook(
    phase: str,
    root_fd: int,
    descriptors: dict[str, tuple[int, object]],
) -> None:
    del phase, root_fd, descriptors


def _directory_verification_hook(
    phase: str,
    directory_fd: int,
    relative: str,
    names: tuple[str, ...],
) -> None:
    del phase, directory_fd, relative, names


def _verify_source_tree_fd(
    lock: SourceLock,
    root_fd: int,
    *,
    _run_finemath_proof: bool = True,
    _expected_snapshot: _SourceTreeSnapshot | None = None,
) -> dict[str, object]:
    root_metadata = os.fstat(root_fd)
    _require_owned_mode(
        root_metadata,
        directory=True,
        description="source root",
    )
    root_identity = _directory_identity(root_metadata)
    expected_files, expected_directories, license_bindings = _expected_paths(lock)
    expected_children: dict[str, set[str]] = {
        relative: set() for relative in expected_directories
    }
    for relative in expected_directories - {""}:
        path = PurePosixPath(relative)
        parent = path.parent.as_posix()
        if parent == ".":
            parent = ""
        expected_children[parent].add(path.name)
    for relative in expected_files:
        path = PurePosixPath(relative)
        parent = path.parent.as_posix()
        if parent == ".":
            parent = ""
        expected_children[parent].add(path.name)
    actual_files: set[str] = set()
    actual_directories = {""}
    license_payloads: dict[str, bytes] = {}
    finemath_entry = next(
        row for row in lock.sources if row.source_id == "finemath"
    )
    finemath_proof = finemath_entry.finemath_selection
    if finemath_proof is None:
        raise ValueError("FineMath selection proof is missing")
    proof_paths = (
        {
            *(f"fineweb_edu/{path}" for path in _FINEWEB_FILES),
            *(f"finemath/{path}" for path in finemath_proof.selected_paths),
        }
        if _run_finemath_proof
        else set()
    )
    proof_descriptors: dict[
        str,
        tuple[int, tuple[int, int, int, int, int, int, int | None, int | None]],
    ] = {}
    directory_snapshots: dict[
        str,
        tuple[
            tuple[int, int, int, int, int, int, int | None, int | None],
            tuple[str, ...],
        ],
    ] = {}
    file_snapshots: dict[
        str,
        tuple[
            tuple[int, int, int, int, int, int, int | None, int | None],
            str,
        ],
    ] = {}
    total_bytes = 0

    def visit(directory_fd: int, prefix: tuple[str, ...]) -> None:
        nonlocal total_bytes
        directory_before = os.fstat(directory_fd)
        description = "source root" if not prefix else "/".join(prefix)
        _require_owned_mode(
            directory_before,
            directory=True,
            description=description,
        )
        identity_before = _directory_identity(directory_before)
        relative_directory = "/".join(prefix)
        initial_names = list_entries(directory_fd)
        expected_names = tuple(
            sorted(expected_children[relative_directory], key=_byte_key)
        )
        if initial_names != expected_names:
            extras = sorted(set(initial_names) - set(expected_names), key=_byte_key)
            if extras:
                extra_name = extras[0]
                extra = entry_lstat(directory_fd, extra_name)
                extra_relative = "/".join((*prefix, extra_name))
                if stat.S_ISLNK(extra.st_mode):
                    kind = "symlink"
                elif stat.S_ISDIR(extra.st_mode):
                    kind = "unlisted directory"
                elif stat.S_ISREG(extra.st_mode):
                    kind = "unlisted file"
                else:
                    kind = "special file"
                raise ValueError(
                    f"source tree contains {kind}: {extra_relative}"
                )
            missing_name = sorted(
                set(expected_names) - set(initial_names),
                key=_byte_key,
            )[0]
            missing_relative = "/".join((*prefix, missing_name))
            kind = (
                "directory"
                if missing_relative in expected_directories
                else "file"
            )
            raise ValueError(
                f"source tree is missing listed {kind}: {missing_relative}"
            )
        _directory_verification_hook(
            "after_initial_snapshot",
            directory_fd,
            relative_directory,
            initial_names,
        )
        for name in initial_names:
            try:
                metadata = entry_lstat(directory_fd, name)
            except FileNotFoundError as error:
                raise ValueError(
                    f"source directory entry race: {description}/{name}"
                ) from error
            relative_parts = (*prefix, name)
            relative = "/".join(relative_parts)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"source tree contains symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                if relative not in expected_directories:
                    raise ValueError(
                        f"source tree contains unlisted directory: {relative}"
                    )
                child_fd, child_identity = _open_bound_directory(
                    directory_fd,
                    name,
                    description=f"source directory {relative}",
                )
                try:
                    actual_directories.add(relative)
                    visit(child_fd, relative_parts)
                    _require_named_directory_identity(
                        directory_fd,
                        name,
                        child_fd,
                        child_identity,
                        description=f"source directory {relative}",
                    )
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"source tree contains special file: {relative}")
            if relative not in expected_files:
                raise ValueError(f"source tree contains unlisted file: {relative}")
            _require_owned_mode(
                metadata,
                directory=False,
                description=f"source file {relative}",
            )
            descriptor, opened = open_regular_file_at(directory_fd, name)
            digest = hashlib.sha256()
            byte_count = 0
            payload_chunks: list[bytes] | None = (
                [] if relative in license_bindings else None
            )
            try:
                _require_owned_mode(
                    opened,
                    directory=False,
                    description=f"source file {relative}",
                )
                if _file_identity(opened) != _file_identity(metadata):
                    raise ValueError(f"source file identity drift: {relative}")
                while True:
                    chunk = os.read(descriptor, 1 << 20)
                    if not chunk:
                        break
                    digest.update(chunk)
                    byte_count += len(chunk)
                    if payload_chunks is not None:
                        payload_chunks.append(chunk)
                after = os.fstat(descriptor)
                named_after = entry_lstat(directory_fd, name)
                if (
                    _file_identity(after) != _file_identity(opened)
                    or _file_identity(named_after) != _file_identity(opened)
                    or os.read(descriptor, 1)
                ):
                    raise ValueError(f"source file identity drift: {relative}")
                if relative in proof_paths:
                    proof_descriptor = os.dup(descriptor)
                    os.lseek(proof_descriptor, 0, os.SEEK_SET)
                    proof_descriptors[relative] = (
                        proof_descriptor,
                        _file_identity(opened),
                    )
            finally:
                os.close(descriptor)
            expected = expected_files[relative]
            if byte_count != expected.bytes or digest.hexdigest() != expected.sha256:
                raise ValueError(f"source byte drift: {relative}")
            file_snapshots[relative] = (
                _file_identity(opened),
                digest.hexdigest(),
            )
            if payload_chunks is not None:
                if byte_count > 1 << 20:
                    raise ValueError(
                        f"license file is unexpectedly large: {relative}"
                    )
                license_payloads[relative] = b"".join(payload_chunks)
            actual_files.add(relative)
            total_bytes += byte_count
        _directory_verification_hook(
            "before_final_snapshot",
            directory_fd,
            relative_directory,
            initial_names,
        )
        final_names = list_entries(directory_fd)
        directory_after = os.fstat(directory_fd)
        if (
            final_names != initial_names
            or final_names != expected_names
            or _directory_identity(directory_after) != identity_before
        ):
            raise ValueError(f"source directory entry race: {description}")
        directory_snapshots[relative_directory] = (
            identity_before,
            initial_names,
        )

    try:
        visit(root_fd, ())
        if _directory_identity(os.fstat(root_fd)) != root_identity:
            raise ValueError("source directory entry race: source root")
        missing_files = sorted(set(expected_files) - actual_files, key=_byte_key)
        if missing_files:
            raise ValueError(
                f"source tree is missing listed file: {missing_files[0]}"
            )
        missing_directories = sorted(
            expected_directories - actual_directories,
            key=_byte_key,
        )
        if missing_directories:
            raise ValueError(
                f"source tree is missing listed directory: "
                f"{missing_directories[0]}"
            )
        for entry in lock.sources:
            payloads = tuple(
                license_payloads[
                    (
                        PurePosixPath(entry.materialized_path) / relative
                    ).as_posix()
                ]
                for relative in entry.license_files
            )
            if _license_from_bytes(payloads) != entry.license_spdx:
                raise ValueError(f"license file SPDX drift: {entry.source_id}")
        snapshot = _SourceTreeSnapshot(
            lock_sha256=lock.sha256,
            directories=tuple(
                (relative, identity, names)
                for relative, (identity, names) in sorted(
                    directory_snapshots.items(),
                    key=lambda item: _byte_key(item[0]),
                )
            ),
            files=tuple(
                (relative, identity, digest)
                for relative, (identity, digest) in sorted(
                    file_snapshots.items(),
                    key=lambda item: _byte_key(item[0]),
                )
            ),
        )
        if _expected_snapshot is not None and snapshot != _expected_snapshot:
            raise ValueError("source tree changed during FineMath proof")
        result = {
            "bytes": total_bytes,
            "dataset_id": lock.dataset_id,
            "files": len(actual_files),
            "passed": True,
            "source_lock_sha256": lock.sha256,
        }
        if _run_finemath_proof:
            _finemath_proof_hook(
                "during_proof",
                root_fd,
                proof_descriptors,
            )
            try:
                _verify_finemath_selection_descriptors(
                    lock,
                    proof_descriptors,
                )
            except ValueError as error:
                if "file identity drift" in str(error):
                    raise ValueError(
                        "source tree changed during FineMath proof"
                    ) from error
                raise
            try:
                _verify_source_tree_fd(
                    lock,
                    root_fd,
                    _run_finemath_proof=False,
                    _expected_snapshot=snapshot,
                )
            except ValueError as error:
                raise ValueError(
                    "source tree changed during FineMath proof"
                ) from error
        return result
    finally:
        for descriptor, _identity in proof_descriptors.values():
            os.close(descriptor)


def verify_source_tree(
    lock: SourceLock,
    source_root: Path,
    *,
    expected_generator_commit: str,
) -> dict[str, object]:
    _authorize_source_lock(
        lock,
        expected_generator_commit=expected_generator_commit,
    )
    parent_fd = -1
    root_fd = -1
    try:
        parent_fd, root_name = open_parent_directory(Path(source_root))
        root_fd, root_identity = _open_bound_directory(
            parent_fd,
            root_name,
            description="source root",
        )
        result = _verify_source_tree_fd(lock, root_fd)
        _require_named_directory_identity(
            parent_fd,
            root_name,
            root_fd,
            root_identity,
            description="source root",
        )
        return result
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError("source root is missing or unsafe") from error
    finally:
        if root_fd >= 0:
            os.close(root_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


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
        source_catalog_sha256=reviewed_source_catalog_sha256(),
        sources=tuple(sorted(entries, key=lambda entry: _byte_key(entry.source_id))),
    )
    verify_source_tree(
        lock,
        root,
        expected_generator_commit=generator_commit,
    )
    return lock


def _open_directory_map(
    root_fd: int,
    directories: set[str],
    *,
    create: bool,
    description: str,
) -> dict[str, int]:
    descriptors = {"": root_fd}
    opened: dict[str, int] = {}
    try:
        for relative in sorted(
            directories - {""},
            key=lambda value: (
                len(PurePosixPath(value).parts),
                _byte_key(value),
            ),
        ):
            path = PurePosixPath(relative)
            parent = path.parent.as_posix()
            if parent == ".":
                parent = ""
            parent_fd = descriptors[parent]
            if create:
                os.mkdir(path.name, mode=0o700, dir_fd=parent_fd)
            child_fd, _identity = _open_bound_directory(
                parent_fd,
                path.name,
                description=f"{description} {relative}",
            )
            descriptors[relative] = child_fd
            opened[relative] = child_fd
        return descriptors
    except BaseException:
        for descriptor in opened.values():
            os.close(descriptor)
        raise


def _close_directory_map(descriptors: dict[str, int]) -> None:
    for relative, descriptor in descriptors.items():
        if relative:
            os.close(descriptor)


def _copy_verified_file_at(
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
    expected: SourceFile,
) -> None:
    source_named = entry_lstat(source_parent_fd, source_name)
    _require_owned_mode(
        source_named,
        directory=False,
        description=f"source staging input {expected.path}",
    )
    source_fd, source_opened = open_regular_file_at(
        source_parent_fd,
        source_name,
    )
    destination_fd = -1
    digest = hashlib.sha256()
    byte_count = 0
    try:
        _require_owned_mode(
            source_opened,
            directory=False,
            description=f"source staging input {expected.path}",
        )
        if _file_identity(source_named) != _file_identity(source_opened):
            raise ValueError(f"source staging input identity drift: {expected.path}")
        destination_fd = os.open(
            destination_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=destination_parent_fd,
        )
        destination_opened = os.fstat(destination_fd)
        _require_owned_mode(
            destination_opened,
            directory=False,
            description=f"staged source file {expected.path}",
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
        source_after = os.fstat(source_fd)
        source_named_after = entry_lstat(source_parent_fd, source_name)
        destination_after = os.fstat(destination_fd)
        destination_named = entry_lstat(
            destination_parent_fd,
            destination_name,
        )
        _require_owned_mode(
            destination_after,
            directory=False,
            description=f"staged source file {expected.path}",
        )
        _require_owned_mode(
            destination_named,
            directory=False,
            description=f"staged source file {expected.path}",
        )
        if (
            _file_identity(source_after) != _file_identity(source_opened)
            or _file_identity(source_named_after) != _file_identity(source_opened)
            or _regular_inode_identity(destination_after)
            != _regular_inode_identity(destination_opened)
            or _regular_inode_identity(destination_named)
            != _regular_inode_identity(destination_opened)
            or os.read(source_fd, 1)
        ):
            raise ValueError(f"source staging identity drift: {expected.path}")
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        os.close(source_fd)
    if byte_count != expected.bytes or digest.hexdigest() != expected.sha256:
        raise ValueError(f"source byte drift while staging: {expected.path}")


def _fsync_directory_map(descriptors: dict[str, int]) -> None:
    for relative in sorted(
        descriptors,
        key=lambda value: (
            -len(PurePosixPath(value).parts),
            _byte_key(value),
        ),
    ):
        fsync_directory(descriptors[relative])


def _cleanup_quarantine_hook(
    phase: str,
    parent_fd: int,
    name: str,
    quarantine_name: str,
    descriptor: int,
    is_directory: bool,
) -> None:
    del phase, parent_fd, name, quarantine_name, descriptor, is_directory


def _cleanup_quarantine_name(name: str) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return (
        f"{_CLEANUP_QUARANTINE_PREFIX}{digest}-{secrets.token_hex(12)}"
    )


def _restore_cleanup_quarantine(
    parent_fd: int,
    quarantine_name: str,
    original_name: str,
    *,
    description: str,
) -> None:
    try:
        atomic_rename_noreplace(
            parent_fd,
            quarantine_name,
            parent_fd,
            original_name,
        )
    except FileExistsError as error:
        fsync_directory(parent_fd)
        raise ValueError(
            f"{description} cleanup quarantine restore blocked"
        ) from error
    fsync_directory(parent_fd)


def _open_cleanup_child(
    parent_fd: int,
    name: str,
    *,
    description: str,
) -> tuple[int, os.stat_result, bool]:
    metadata = entry_lstat(parent_fd, name)
    if stat.S_ISDIR(metadata.st_mode):
        _require_owned_mode(
            metadata,
            directory=True,
            description=description,
        )
        descriptor, _identity = _open_bound_directory(
            parent_fd,
            name,
            description=description,
        )
        opened = os.fstat(descriptor)
        if _directory_identity(opened) != _directory_identity(metadata):
            os.close(descriptor)
            raise ValueError(f"{description} cleanup identity drift")
        return descriptor, opened, True
    if stat.S_ISREG(metadata.st_mode):
        _require_owned_mode(
            metadata,
            directory=False,
            description=description,
        )
        descriptor, opened = open_regular_file_at(parent_fd, name)
        if _file_identity(opened) != _file_identity(metadata):
            os.close(descriptor)
            raise ValueError(f"{description} cleanup identity drift")
        return descriptor, opened, False
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{description} is a symlink during cleanup")
    raise ValueError(f"{description} is special during cleanup")


def _quarantine_verified_cleanup_entry(
    parent_fd: int,
    name: str,
    pinned_fd: int,
    pinned_metadata: os.stat_result,
    *,
    is_directory: bool,
    description: str,
) -> str:
    expected_identity = (
        _directory_identity(pinned_metadata)
        if is_directory
        else _file_identity(pinned_metadata)
    )
    current_pinned = os.fstat(pinned_fd)
    current_identity = (
        _directory_identity(current_pinned)
        if is_directory
        else _file_identity(current_pinned)
    )
    if current_identity != expected_identity:
        raise ValueError(f"{description} cleanup pinned identity drift")
    named = entry_lstat(parent_fd, name)
    named_identity = (
        _directory_identity(named)
        if is_directory and stat.S_ISDIR(named.st_mode)
        else _file_identity(named)
        if not is_directory and stat.S_ISREG(named.st_mode)
        else None
    )
    if named_identity != expected_identity:
        raise ValueError(f"{description} cleanup named identity drift")

    quarantine_name = _cleanup_quarantine_name(name)
    _cleanup_quarantine_hook(
        "before_quarantine",
        parent_fd,
        name,
        quarantine_name,
        pinned_fd,
        is_directory,
    )
    atomic_rename_noreplace(
        parent_fd,
        name,
        parent_fd,
        quarantine_name,
    )
    fsync_directory(parent_fd)
    _cleanup_quarantine_hook(
        "after_quarantine",
        parent_fd,
        name,
        quarantine_name,
        pinned_fd,
        is_directory,
    )

    quarantine_fd = -1
    try:
        if is_directory:
            quarantine_fd, _identity = _open_bound_directory(
                parent_fd,
                quarantine_name,
                description=f"{description} cleanup quarantine",
            )
            quarantined = os.fstat(quarantine_fd)
            quarantined_identity = _directory_identity(quarantined)
        else:
            quarantine_fd, quarantined = open_regular_file_at(
                parent_fd,
                quarantine_name,
            )
            quarantined_identity = _file_identity(quarantined)
        pinned_after = os.fstat(pinned_fd)
        pinned_after_identity = (
            _directory_identity(pinned_after)
            if is_directory
            else _file_identity(pinned_after)
        )
        if (
            quarantined_identity != pinned_after_identity
            or (
                is_directory
                and _namespace_identity(pinned_after)
                != _namespace_identity(pinned_metadata)
            )
            or (
                not is_directory
                and _regular_inode_identity(pinned_after)
                != _regular_inode_identity(pinned_metadata)
            )
        ):
            os.close(quarantine_fd)
            quarantine_fd = -1
            _restore_cleanup_quarantine(
                parent_fd,
                quarantine_name,
                name,
                description=description,
            )
            raise ValueError(f"{description} cleanup quarantine identity drift")

        _cleanup_quarantine_hook(
            "quarantine_verified",
            parent_fd,
            name,
            quarantine_name,
            pinned_fd,
            is_directory,
        )
        pinned_final = os.fstat(pinned_fd)
        quarantine_final = os.fstat(quarantine_fd)
        named_final = entry_lstat(parent_fd, quarantine_name)
        if is_directory:
            if (
                _namespace_identity(pinned_final)
                != _namespace_identity(pinned_metadata)
                or _namespace_identity(quarantine_final)
                != _namespace_identity(pinned_metadata)
                or _namespace_identity(named_final)
                != _namespace_identity(pinned_metadata)
            ):
                raise ValueError(
                    f"{description} cleanup directory identity drift"
                )
        elif (
            _regular_inode_identity(pinned_final)
            != _regular_inode_identity(pinned_metadata)
            or _file_identity(quarantine_final) != _file_identity(pinned_final)
            or _file_identity(named_final) != _file_identity(pinned_final)
        ):
            raise ValueError(f"{description} cleanup file identity drift")
        fsync_directory(parent_fd)
        return quarantine_name
    finally:
        if quarantine_fd >= 0:
            os.close(quarantine_fd)


def _remove_owned_directory_at(
    parent_fd: int,
    name: str,
    expected_identity: tuple[int, int, int, int],
    *,
    description: str,
) -> str:
    directory_fd, metadata, is_directory = _open_cleanup_child(
        parent_fd,
        name,
        description=description,
    )
    try:
        if not is_directory or _namespace_identity(metadata) != expected_identity:
            raise ValueError(f"{description} identity drift during cleanup")
        return _quarantine_verified_cleanup_entry(
            parent_fd,
            name,
            directory_fd,
            metadata,
            is_directory=True,
            description=description,
        )
    finally:
        os.close(directory_fd)


def stage_source_lock(
    lock: SourceLock,
    download_root: Path,
    canonical_root: Path,
    *,
    expected_generator_commit: str,
) -> Path:
    _authorize_source_lock(
        lock,
        expected_generator_commit=expected_generator_commit,
    )
    canonical = Path(canonical_root)
    final_name = lock.sha256
    final_path = canonical / "sources" / final_name
    source_parent_fd = -1
    source_fd = -1
    canonical_parent_fd = -1
    canonical_fd = -1
    sources_fd = -1
    stage_fd = -1
    stage_name = ""
    stage_identity: tuple[int, int, int, int] | None = None
    try:
        source_parent_fd, source_name = open_parent_directory(
            Path(download_root)
        )
        source_fd, source_identity = _open_bound_directory(
            source_parent_fd,
            source_name,
            description="source root",
        )
        _verify_source_tree_fd(lock, source_fd)

        canonical_parent_fd, canonical_name = open_parent_directory(
            canonical,
            create=True,
            mode=0o700,
        )
        canonical_fd, canonical_identity = _open_bound_directory(
            canonical_parent_fd,
            canonical_name,
            description="canonical directory",
            create=True,
            mode=0o700,
        )
        sources_fd, sources_identity = _open_bound_directory(
            canonical_fd,
            "sources",
            description="canonical sources directory",
            create=True,
            mode=0o700,
        )
        retained_quarantines = tuple(
            name
            for name in list_entries(sources_fd)
            if name.startswith(_CLEANUP_QUARANTINE_PREFIX)
        )
        if retained_quarantines:
            raise ValueError(
                "source stage quarantine requires explicit offline cleanup: "
                f"{retained_quarantines[0]}"
            )
        _require_named_directory_identity(
            canonical_parent_fd,
            canonical_name,
            canonical_fd,
            canonical_identity,
            description="canonical directory",
        )

        if entry_exists(sources_fd, final_name):
            try:
                final_fd, final_identity = _open_bound_directory(
                    sources_fd,
                    final_name,
                    description="published source stage",
                )
                try:
                    _verify_source_tree_fd(lock, final_fd)
                    _require_named_directory_identity(
                        sources_fd,
                        final_name,
                        final_fd,
                        final_identity,
                        description="published source stage",
                    )
                finally:
                    os.close(final_fd)
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"conflicting source stage: {final_path}"
                ) from error
            _require_named_directory_identity(
                canonical_fd,
                "sources",
                sources_fd,
                sources_identity,
                description="canonical sources directory",
            )
            _require_named_directory_identity(
                canonical_parent_fd,
                canonical_name,
                canonical_fd,
                canonical_identity,
                description="canonical directory",
            )
            return final_path

        stage_name = (
            f".{final_name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
        )
        os.mkdir(stage_name, mode=0o700, dir_fd=sources_fd)
        stage_fd, stage_identity = _open_bound_directory(
            sources_fd,
            stage_name,
            description="private source stage",
        )
        expected_files, directories, _licenses = _expected_paths(lock)

        _require_named_directory_identity(
            canonical_parent_fd,
            canonical_name,
            canonical_fd,
            canonical_identity,
            description="canonical directory",
        )
        _require_named_directory_identity(
            canonical_fd,
            "sources",
            sources_fd,
            sources_identity,
            description="canonical sources directory",
        )
        source_directories = _open_directory_map(
            source_fd,
            directories,
            create=False,
            description="source directory",
        )
        destination_directories = _open_directory_map(
            stage_fd,
            directories,
            create=True,
            description="staged source directory",
        )
        try:
            for relative, expected in sorted(
                expected_files.items(),
                key=lambda item: _byte_key(item[0]),
            ):
                path = PurePosixPath(relative)
                parent = path.parent.as_posix()
                if parent == ".":
                    parent = ""
                _copy_verified_file_at(
                    source_directories[parent],
                    path.name,
                    destination_directories[parent],
                    path.name,
                    expected,
                )
            _fsync_directory_map(destination_directories)
        finally:
            _close_directory_map(destination_directories)
            _close_directory_map(source_directories)

        _verify_source_tree_fd(lock, source_fd)
        _require_named_directory_identity(
            source_parent_fd,
            source_name,
            source_fd,
            source_identity,
            description="source root",
        )
        _verify_source_tree_fd(lock, stage_fd)
        _require_named_directory_identity(
            sources_fd,
            stage_name,
            stage_fd,
            stage_identity,
            description="private source stage",
        )
        _require_named_directory_identity(
            canonical_parent_fd,
            canonical_name,
            canonical_fd,
            canonical_identity,
            description="canonical directory",
        )
        _require_named_directory_identity(
            canonical_fd,
            "sources",
            sources_fd,
            sources_identity,
            description="canonical sources directory",
        )

        try:
            atomic_rename_noreplace(
                sources_fd,
                stage_name,
                sources_fd,
                final_name,
            )
        except FileExistsError:
            final_fd, final_identity = _open_bound_directory(
                sources_fd,
                final_name,
                description="published source stage",
            )
            try:
                _verify_source_tree_fd(lock, final_fd)
                _require_named_directory_identity(
                    sources_fd,
                    final_name,
                    final_fd,
                    final_identity,
                    description="published source stage",
                )
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"conflicting source stage: {final_path}"
                ) from error
            finally:
                os.close(final_fd)
            quarantine_name = _remove_owned_directory_at(
                sources_fd,
                stage_name,
                stage_identity,
                description="private source stage",
            )
            stage_name = ""
            raise ValueError(
                "source stage race retained a quarantine for offline cleanup: "
                f"{quarantine_name}"
            )
        else:
            fsync_directory(sources_fd)
            final_fd, final_identity = _open_bound_directory(
                sources_fd,
                final_name,
                description="published source stage",
            )
            try:
                if final_identity != stage_identity:
                    raise ValueError("published source stage identity drift")
                _verify_source_tree_fd(lock, final_fd)
                _require_named_directory_identity(
                    sources_fd,
                    final_name,
                    final_fd,
                    final_identity,
                    description="published source stage",
                )
            finally:
                os.close(final_fd)
            stage_name = ""
        _require_named_directory_identity(
            canonical_parent_fd,
            canonical_name,
            canonical_fd,
            canonical_identity,
            description="canonical directory",
        )
        _require_named_directory_identity(
            canonical_fd,
            "sources",
            sources_fd,
            sources_identity,
            description="canonical sources directory",
        )
        return final_path
    except BaseException:
        if (
            sources_fd >= 0
            and stage_name
            and stage_identity is not None
            and entry_exists(sources_fd, stage_name)
        ):
            _remove_owned_directory_at(
                sources_fd,
                stage_name,
                stage_identity,
                description="private source stage",
            )
            stage_name = ""
        raise
    finally:
        if stage_fd >= 0:
            os.close(stage_fd)
        if sources_fd >= 0:
            os.close(sources_fd)
        if canonical_fd >= 0:
            os.close(canonical_fd)
        if canonical_parent_fd >= 0:
            os.close(canonical_parent_fd)
        if source_fd >= 0:
            os.close(source_fd)
        if source_parent_fd >= 0:
            os.close(source_parent_fd)


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


def _validated_finemath_paths(info: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    card_data = getattr(info, "card_data", None)
    card_license = card_data.get("license") if card_data is not None else None
    if card_license not in {"odc-by", "ODC-By-1.0"}:
        raise ValueError("FineMath metadata does not prove ODC-By-1.0")
    raw_paths = tuple(
        getattr(sibling, "rfilename", "")
        for sibling in getattr(info, "siblings", ())
    )
    if (
        any(not isinstance(path, str) or not path for path in raw_paths)
        or len(raw_paths) != len(set(raw_paths))
    ):
        raise ValueError("FineMath repository inventory is invalid")
    for path in raw_paths:
        _safe_relative_path(path, "FineMath repository path")
    sibling_paths = tuple(sorted(raw_paths, key=_byte_key))

    four_plus = tuple(
        path
        for path in sibling_paths
        if _is_finemath_train_file(path, "finemath-4plus")
    )
    three_plus = tuple(
        path
        for path in sibling_paths
        if _is_finemath_train_file(path, "finemath-3plus")
    )
    if "README.md" not in sibling_paths or not four_plus:
        raise ValueError("FineMath repository inventory is incomplete")
    return four_plus, three_plus


def _select_finemath_files(
    *,
    four_plus: tuple[str, ...],
    three_plus: tuple[str, ...],
    fineweb_paths: tuple[Path, ...],
    materialize: Callable[[str], Path],
    iter_texts: Callable[[Path], Iterable[str]],
    encode: Callable[[str], list[int]],
    quota: int,
    database_path: Path,
) -> FineMathSelectionProof:
    if type(quota) is not int or quota < 0:
        raise ValueError("FineMath quota must be a non-negative integer")
    if len(fineweb_paths) != len(_FINEWEB_FILES):
        raise ValueError("FineMath selection requires all reviewed FineWeb files")
    ordered_four = tuple(sorted(four_plus, key=_byte_key))
    ordered_three = tuple(sorted(three_plus, key=_byte_key))
    if (
        not ordered_four
        or len(ordered_four) != len(set(ordered_four))
        or len(ordered_three) != len(set(ordered_three))
        or set(ordered_four) & set(ordered_three)
    ):
        raise ValueError("FineMath selection inventory is invalid")
    database = sqlite3.connect(str(database_path))
    try:
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA synchronous=OFF")
        database.execute(
            "CREATE TABLE fineweb ("
            "digest BLOB NOT NULL, text TEXT NOT NULL, "
            "PRIMARY KEY (digest, text)"
            ") WITHOUT ROWID"
        )
        for path in fineweb_paths:
            for text in iter_texts(path):
                if not isinstance(text, str):
                    raise ValueError(f"FineWeb text is not a string: {path}")
                normalized = unicodedata.normalize("NFC", text)
                database.execute(
                    "INSERT OR IGNORE INTO fineweb(digest, text) VALUES (?, ?)",
                    (
                        hashlib.sha256(normalized.encode("utf-8")).digest(),
                        normalized,
                    ),
                )
            database.commit()

        selected = []
        token_count = 0
        duplicate_rows = 0
        for relative in (*ordered_four, *ordered_three):
            path = materialize(relative)
            selected.append(relative)
            for text in iter_texts(path):
                if not isinstance(text, str):
                    raise ValueError(
                        f"FineMath text is not a string: {relative}"
                    )
                normalized = unicodedata.normalize("NFC", text)
                digest = hashlib.sha256(normalized.encode("utf-8")).digest()
                duplicate = database.execute(
                    "SELECT 1 FROM fineweb WHERE digest=? AND text=?",
                    (digest, normalized),
                ).fetchone()
                if duplicate is not None:
                    duplicate_rows += 1
                    continue
                token_count += len(encode(normalized)) + 1
            if token_count > quota:
                return FineMathSelectionProof(
                    algorithm=_FINEMATH_SELECTION_ALGORITHM,
                    quota=quota,
                    four_plus_paths=ordered_four,
                    three_plus_paths=ordered_three,
                    selected_paths=tuple(selected),
                    usable_targets=token_count,
                    fineweb_duplicate_rows=duplicate_rows,
                )
        raise ValueError(
            "FineMath bytes do not prove more than the frozen quota "
            "after exact FineWeb cross-deduplication"
        )
    finally:
        database.close()


class PublicSourceResolver:
    """Resolve only the reviewed public catalog, never ambient HF credentials."""

    def __init__(self) -> None:
        self._hf_api = HfApi(token=False)

    def resolve(self, request: SourceRequest, download_root: Path) -> SourceEntry:
        if not isinstance(request, SourceRequest):
            raise TypeError("request must be a SourceRequest")
        _require_reviewed_request(request)
        root = Path(download_root)
        root.mkdir(parents=True, exist_ok=True)
        materialized = root / request.source_id
        if materialized.exists() or materialized.is_symlink():
            raise ValueError(
                f"source materialization already exists: {request.source_id}"
            )
        materialized.mkdir(mode=0o700)
        if request.transport == "git":
            entry = self._resolve_git(request, root, materialized)
        elif request.transport == "huggingface_dataset":
            entry = self._resolve_huggingface(request, root, materialized)
        else:
            raise ValueError(f"unsupported source transport: {request.transport}")
        _validate_resolved_entry(request, entry)
        return entry

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
            finemath_selection = None
            if request.source_id == "finemath":
                finemath_selection = self._download_finemath(
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
                selected = [
                    *request.required_data_paths,
                    *request.required_data_prefixes,
                ]
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
            finemath_selection=finemath_selection,
        )

    def _download_finemath(
        self,
        request: SourceRequest,
        info: Any,
        revision: str,
        download_root: Path,
        materialized: Path,
        cache: Path,
    ) -> FineMathSelectionProof:
        four_plus, three_plus = _validated_finemath_paths(info)
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
            tokenizer = get_tok()

            def materialize(relative: str) -> Path:
                self._download_hf_file(
                    request,
                    revision,
                    relative,
                    materialized,
                    cache,
                )
                return materialized / relative

            return _select_finemath_files(
                four_plus=four_plus,
                three_plus=three_plus,
                fineweb_paths=tuple(
                    fineweb_root / relative for relative in _FINEWEB_FILES
                ),
                materialize=materialize,
                iter_texts=lambda path: _iter_parquet_texts(parquet, path),
                encode=tokenizer.encode,
                quota=_FINEMATH_TARGETS,
                database_path=Path(proof_directory) / "fineweb.sqlite3",
            )


def _iter_parquet_texts(parquet_module: Any, path: Path):
    parquet_file = parquet_module.ParquetFile(path)
    if "text" not in parquet_file.schema.names:
        raise ValueError(f"source parquet has no text column: {path}")
    for batch in parquet_file.iter_batches(batch_size=1024, columns=["text"]):
        for value in batch.column(0).to_pylist():
            if not isinstance(value, str):
                raise ValueError(f"source parquet text is not a string: {path}")
            yield value
