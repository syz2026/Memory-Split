"""Closed receipts and descriptor-pinned Wikidata archive authority."""

from __future__ import annotations

import ctypes
import hashlib
import heapq
import json
import os
import re
import secrets
import stat
import struct
import sys
import tarfile
import unicodedata
import weakref
import zlib
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, BinaryIO, cast

from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.parallel.safeio import (
    atomic_rename_noreplace,
    entry_lstat,
    fsync_directory,
    list_entries,
    open_directory_at,
    open_directory_path,
    open_parent_directory,
    open_regular_file_at,
)
from corpusgen.reasoning_v2 import source_lock as source_lock_module
from corpusgen.reasoning_v2.source_lock import SourceFile, SourceLock
RECEIPT_FORMAT = "memorysplit-reasoning-v2-wikidata-view-v1"
RECEIPT_SCHEMA_VERSION = 1
TRAINING_SPLITS = ("inductive_train", "transductive_train")
ARCHIVE_PATHS = (
    "wikidata5m_alias.tar.gz",
    "wikidata5m_inductive.tar.gz",
    "wikidata5m_transductive.tar.gz",
)

ARCHIVE_MEMBERS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "wikidata5m_alias.tar.gz": (
            "wikidata5m_entity.txt",
            "wikidata5m_relation.txt",
        ),
        "wikidata5m_inductive.tar.gz": (
            "wikidata5m_inductive_test.txt",
            "wikidata5m_inductive_train.txt",
            "wikidata5m_inductive_valid.txt",
        ),
        "wikidata5m_transductive.tar.gz": (
            "wikidata5m_transductive_test.txt",
            "wikidata5m_transductive_train.txt",
            "wikidata5m_transductive_valid.txt",
        ),
    }
)
MEMBER_PATHS = tuple(
    sorted(
        (
            f"members/{member}"
            for archive_path in ARCHIVE_PATHS
            for member in ARCHIVE_MEMBERS[archive_path]
        ),
        key=str.encode,
    )
)
STREAM_PATHS = (
    "streams/aliases.tsv",
    "streams/distinct-edges.tsv",
    "streams/training.tsv",
)
INDEX_PATHS = (
    "indexes/aliases.bin",
    "indexes/inductive-training-offsets.bin",
    "indexes/transductive-training-offsets.bin",
)

_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_QID_RE = re.compile(r"Q[0-9]+\Z")
_PID_RE = re.compile(r"P[0-9]+\Z")
_RECEIPT_FIELDS = {
    "alias_rows",
    "archives",
    "distinct_edges",
    "format",
    "generator_commit",
    "indexes",
    "members",
    "overlap_audit_passed",
    "schema_version",
    "source_lock_sha256",
    "streams",
    "training_rows",
}
_READ_CHUNK_SIZE = 1 << 20
_TAR_BLOCK_SIZE = 512
_TAR_ZERO_BLOCK = b"\0" * _TAR_BLOCK_SIZE
_PAX_HEADER_TYPES = frozenset(
    {tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.SOLARIS_XHDTYPE}
)
_PAX_SIZE_KEYWORDS = frozenset(
    {b"size", b"GNU.sparse.size", b"GNU.sparse.realsize"}
)
_PAX_RECORD_RE = re.compile(rb"(\d+) ([^=]+)=")
_UINT64_MAX = (1 << 64) - 1
_ALIAS_RELATION_BIT = 1 << 63
_ALIAS_NUMERIC_MAX = _ALIAS_RELATION_BIT - 1
_SOURCE_LINE_LIMIT = 64 * 1024 * 1024
_SORT_CHUNK_BYTES = 8 * 1024 * 1024
_SORT_CHUNK_RECORDS = 65_536
_SORT_MERGE_FAN_IN = 32
_SORT_RECORD_LIMIT = 128 * 1024 * 1024
_CONTROL_FILE_LIMIT = 16 * 1024 * 1024
_SORT_HEADER = struct.Struct(">IQ")
_UINT64 = struct.Struct(">Q")
_ALIAS_INDEX_RECORD = struct.Struct(">QQQ")
_EDGE_SORT_KEY = struct.Struct(">QQQBQ")
_RENAME_EXCHANGE = 2
_RENAME_SWAP = 0x00000002
_WRITE_FLAGS = (
    os.O_RDWR
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_VIEW_ROOT_ENTRIES = ("indexes", "members", "receipt.json", "streams")
_TRAINING_SOURCE_ROWS = (
    (
        "inductive_train",
        "wikidata5m_inductive_train.txt",
        "wikidata5m_inductive.tar.gz",
        0,
        True,
    ),
    (
        "transductive_train",
        "wikidata5m_transductive_train.txt",
        "wikidata5m_transductive.tar.gz",
        1,
        True,
    ),
    (
        "inductive_test",
        "wikidata5m_inductive_test.txt",
        "wikidata5m_inductive.tar.gz",
        2,
        False,
    ),
    (
        "inductive_valid",
        "wikidata5m_inductive_valid.txt",
        "wikidata5m_inductive.tar.gz",
        3,
        False,
    ),
    (
        "transductive_test",
        "wikidata5m_transductive_test.txt",
        "wikidata5m_transductive.tar.gz",
        4,
        False,
    ),
    (
        "transductive_valid",
        "wikidata5m_transductive_valid.txt",
        "wikidata5m_transductive.tar.gz",
        5,
        False,
    ),
)
_FileIdentity = tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int | None,
    int | None,
]
_CreationIdentity = tuple[int, int, int, int, int]
_PrivateFileIdentity = tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int | None,
    int | None,
]


def _byte_key(value: str) -> bytes:
    return value.encode("utf-8")


def _validate_sha256(value: object, description: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return cast(str, value)


def _validate_commit(value: object, description: str) -> str:
    if type(value) is not str or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase 40-hex commit")
    return cast(str, value)


def _safe_relative_path(value: object, description: str) -> str:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or "\\" in value
        or unicodedata.normalize("NFC", value) != value
    ):
        raise ValueError(f"{description} must be a canonical relative POSIX path")
    text = cast(str, value)
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or posix.as_posix() != text
        or not posix.parts
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError(f"unsafe {description}: {text!r}")
    return text


def _validate_count(value: object, description: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{description} must be a non-negative integer")
    return cast(int, value)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is not allowed: {value}")


def _strict_json_bytes(payload: bytes, description: str) -> object:
    if not isinstance(payload, bytes):
        raise TypeError(f"{description} must be bytes")
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


def _require_exact_paths(
    records: tuple[Any, ...],
    expected: tuple[str, ...],
    description: str,
    record_type: type | None = None,
) -> None:
    expected_type = ArtifactRecord if record_type is None else record_type
    if not isinstance(records, tuple) or not all(
        isinstance(record, expected_type) for record in records
    ):
        raise ValueError(f"{description} must be a {expected_type.__name__} tuple")
    paths = tuple(record.path for record in records)
    if paths != tuple(sorted(paths, key=_byte_key)):
        raise ValueError(f"{description} must use bytewise path order")
    if paths != expected:
        raise ValueError(f"{description} path inventory does not match the contract")


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _safe_relative_path(self.path, "artifact path")
        _validate_count(self.bytes, "artifact bytes")
        _validate_sha256(self.sha256, "artifact sha256")

    def as_dict(self) -> dict[str, object]:
        return {
            "bytes": self.bytes,
            "path": self.path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ArtifactRecord":
        if (
            not isinstance(value, dict)
            or set(value) != {"bytes", "path", "sha256"}
        ):
            raise ValueError("artifact fields do not match the contract")
        return cls(
            path=value["path"],
            bytes=value["bytes"],
            sha256=value["sha256"],
        )


@dataclass(frozen=True)
class IndexArtifactRecord(ArtifactRecord):
    count: int
    record_width: int

    def __post_init__(self) -> None:
        super().__post_init__()
        count = _validate_count(self.count, "index record count")
        if type(self.record_width) is not int or self.record_width <= 0:
            raise ValueError("index record width must be a positive integer")
        if self.bytes != count * self.record_width:
            raise ValueError("index byte count does not match count and width")

    def as_dict(self) -> dict[str, object]:
        return {
            **super().as_dict(),
            "count": self.count,
            "record_width": self.record_width,
        }

    @classmethod
    def from_dict(cls, value: object) -> "IndexArtifactRecord":
        if (
            not isinstance(value, dict)
            or set(value)
            != {"bytes", "count", "path", "record_width", "sha256"}
        ):
            raise ValueError("index artifact fields do not match the contract")
        return cls(
            path=value["path"],
            bytes=value["bytes"],
            sha256=value["sha256"],
            count=value["count"],
            record_width=value["record_width"],
        )


@dataclass(frozen=True)
class V2TrainingTriple:
    training_split: str
    row: int
    subject: int
    relation: str
    object: int
    member: str
    archive_path: str

    def __post_init__(self) -> None:
        if self.training_split not in TRAINING_SPLITS:
            raise ValueError("training split is not in the frozen contract")
        if type(self.row) is not int or self.row <= 0:
            raise ValueError("training row must be a positive integer")
        for name in ("subject", "object"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.relation) is not str or _PID_RE.fullmatch(self.relation) is None:
            raise ValueError("relation must be a canonical PID")
        member = _safe_relative_path(self.member, "training member")
        if self.archive_path not in ARCHIVE_PATHS:
            raise ValueError("training archive path is not declared")
        if member not in ARCHIVE_MEMBERS[self.archive_path]:
            raise ValueError("training member is not declared by its archive")
        expected_member = f"wikidata5m_{self.training_split}.txt"
        if member != expected_member:
            raise ValueError("training member does not match its split")


@dataclass(frozen=True)
class V2AliasRecord:
    canonical_id: str
    kind: str
    display: str
    aliases: tuple[str, ...]

    def __post_init__(self) -> None:
        expected_kind = (
            "entity"
            if type(self.canonical_id) is str
            and _QID_RE.fullmatch(self.canonical_id) is not None
            else "relation"
            if type(self.canonical_id) is str
            and _PID_RE.fullmatch(self.canonical_id) is not None
            else None
        )
        if expected_kind is None or self.kind != expected_kind:
            raise ValueError("alias canonical ID and kind do not match")
        if (
            type(self.display) is not str
            or not self.display
            or "\x00" in self.display
            or unicodedata.normalize("NFC", self.display) != self.display
        ):
            raise ValueError("alias display must be a nonempty canonical string")
        if not isinstance(self.aliases, tuple):
            raise ValueError("aliases must be a tuple")
        for alias in self.aliases:
            if (
                type(alias) is not str
                or not alias
                or "\x00" in alias
                or unicodedata.normalize("NFC", alias) != alias
            ):
                raise ValueError("alias values must be nonempty canonical strings")


@dataclass(frozen=True)
class WikidataDerivedViewReceipt:
    format: str
    schema_version: int
    source_lock_sha256: str
    generator_commit: str
    archives: tuple[ArtifactRecord, ...]
    members: tuple[ArtifactRecord, ...]
    streams: tuple[ArtifactRecord, ...]
    indexes: tuple[IndexArtifactRecord, ...]
    training_rows: int
    alias_rows: int
    distinct_edges: int
    overlap_audit_passed: bool

    def __post_init__(self) -> None:
        if self.format != RECEIPT_FORMAT:
            raise ValueError("Wikidata receipt format identity mismatch")
        if (
            type(self.schema_version) is not int
            or self.schema_version != RECEIPT_SCHEMA_VERSION
        ):
            raise ValueError("Wikidata receipt schema version identity mismatch")
        _validate_sha256(self.source_lock_sha256, "source lock sha256")
        _validate_commit(self.generator_commit, "generator commit")
        _require_exact_paths(self.archives, ARCHIVE_PATHS, "archive inventory")
        _require_exact_paths(self.members, MEMBER_PATHS, "member inventory")
        _require_exact_paths(self.streams, STREAM_PATHS, "stream inventory")
        _require_exact_paths(
            self.indexes,
            INDEX_PATHS,
            "index inventory",
            IndexArtifactRecord,
        )
        training_rows = _validate_count(self.training_rows, "training rows")
        alias_rows = _validate_count(self.alias_rows, "alias rows")
        distinct_edges = _validate_count(self.distinct_edges, "distinct edges")
        if distinct_edges > training_rows:
            raise ValueError("distinct edge total exceeds training row total")
        if (training_rows == 0) != (distinct_edges == 0):
            raise ValueError(
                "training row and distinct edge totals are inconsistent"
            )
        if type(self.overlap_audit_passed) is not bool:
            raise ValueError("overlap audit result must be a boolean")
        if not self.overlap_audit_passed:
            raise ValueError("training/sealed overlap audit did not pass")

        stream_by_path = {record.path: record for record in self.streams}
        for path, rows in (
            ("streams/training.tsv", training_rows),
            ("streams/aliases.tsv", alias_rows),
            ("streams/distinct-edges.tsv", distinct_edges),
        ):
            byte_count = stream_by_path[path].bytes
            if (rows == 0) != (byte_count == 0):
                raise ValueError(f"stream row/byte totals are inconsistent: {path}")

        index_by_path = {record.path: record for record in self.indexes}
        alias_index = index_by_path["indexes/aliases.bin"]
        inductive_index = index_by_path["indexes/inductive-training-offsets.bin"]
        transductive_index = index_by_path[
            "indexes/transductive-training-offsets.bin"
        ]
        if (
            alias_index.record_width != 24
            or alias_index.count != alias_rows
            or inductive_index.record_width != 8
            or transductive_index.record_width != 8
            or inductive_index.count + transductive_index.count != training_rows
        ):
            raise ValueError("receipt row and index totals are inconsistent")

    def as_dict(self) -> dict[str, object]:
        return {
            "alias_rows": self.alias_rows,
            "archives": [record.as_dict() for record in self.archives],
            "distinct_edges": self.distinct_edges,
            "format": self.format,
            "generator_commit": self.generator_commit,
            "indexes": [record.as_dict() for record in self.indexes],
            "members": [record.as_dict() for record in self.members],
            "overlap_audit_passed": self.overlap_audit_passed,
            "schema_version": self.schema_version,
            "source_lock_sha256": self.source_lock_sha256,
            "streams": [record.as_dict() for record in self.streams],
            "training_rows": self.training_rows,
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    @classmethod
    def from_dict(cls, value: object) -> "WikidataDerivedViewReceipt":
        if not isinstance(value, dict) or set(value) != _RECEIPT_FIELDS:
            raise ValueError("Wikidata receipt fields do not match the contract")
        inventories: dict[str, tuple[Any, ...]] = {}
        for name in ("archives", "members", "streams", "indexes"):
            raw_inventory = value[name]
            if not isinstance(raw_inventory, list):
                raise ValueError(f"Wikidata receipt {name} must be a JSON list")
            record_type = (
                IndexArtifactRecord if name == "indexes" else ArtifactRecord
            )
            inventories[name] = tuple(
                record_type.from_dict(record) for record in raw_inventory
            )
        return cls(
            format=value["format"],
            schema_version=value["schema_version"],
            source_lock_sha256=value["source_lock_sha256"],
            generator_commit=value["generator_commit"],
            archives=inventories["archives"],
            members=inventories["members"],
            streams=inventories["streams"],
            indexes=inventories["indexes"],
            training_rows=value["training_rows"],
            alias_rows=value["alias_rows"],
            distinct_edges=value["distinct_edges"],
            overlap_audit_passed=value["overlap_audit_passed"],
        )

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        *,
        expected_generator_commit: str | None = None,
    ) -> "WikidataDerivedViewReceipt":
        value = _strict_json_bytes(payload, "Wikidata derived-view receipt")
        receipt = cls.from_dict(value)
        if receipt.to_bytes() != payload:
            raise ValueError("Wikidata derived-view receipt is not canonical JSON")
        if expected_generator_commit is not None:
            expected = _validate_commit(
                expected_generator_commit,
                "expected generator commit",
            )
            if receipt.generator_commit != expected:
                raise ValueError("Wikidata receipt generator commit mismatch")
        return receipt


@dataclass(frozen=True)
class _VerifiedViewAuthority:
    root_identity: tuple[int, int, int, int, int, int, int | None, int | None]
    directory_identities: Mapping[
        str,
        tuple[int, int, int, int, int, int, int | None, int | None],
    ]
    file_identities: Mapping[
        str,
        tuple[int, int, int, int, int, int, int | None, int | None],
    ]


@dataclass(frozen=True)
class WikidataDerivedView:
    root: Path
    receipt_sha256: str
    receipt: WikidataDerivedViewReceipt


_VERIFIED_VIEW_AUTHORITIES: dict[
    int,
    tuple[
        weakref.ReferenceType[WikidataDerivedView],
        _VerifiedViewAuthority,
    ],
] = {}


def _register_verified_view(
    view: WikidataDerivedView,
    authority: _VerifiedViewAuthority,
) -> WikidataDerivedView:
    identity = id(view)

    def discard(reference: weakref.ReferenceType[WikidataDerivedView]) -> None:
        current = _VERIFIED_VIEW_AUTHORITIES.get(identity)
        if current is not None and current[0] is reference:
            _VERIFIED_VIEW_AUTHORITIES.pop(identity, None)

    reference = weakref.ref(view, discard)
    _VERIFIED_VIEW_AUTHORITIES[identity] = (reference, authority)
    return view


def _registered_view_authority(
    view: WikidataDerivedView,
) -> _VerifiedViewAuthority:
    current = _VERIFIED_VIEW_AUTHORITIES.get(id(view))
    if current is None or current[0]() is not view:
        raise ValueError("a verified Wikidata derived view is required")
    return current[1]


def parse_wikidata_derived_view_receipt(
    payload: bytes,
    *,
    expected_generator_commit: str | None = None,
) -> WikidataDerivedViewReceipt:
    return WikidataDerivedViewReceipt.from_bytes(
        payload,
        expected_generator_commit=expected_generator_commit,
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


def _creation_identity(metadata: os.stat_result) -> _CreationIdentity:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_uid,
        stat.S_IMODE(metadata.st_mode),
    )


def _private_file_identity(
    metadata: os.stat_result,
) -> _PrivateFileIdentity:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_uid,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_size,
        getattr(metadata, "st_mtime_ns", None),
        getattr(metadata, "st_ctime_ns", None),
    )


def _directory_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int, int | None, int | None]:
    return _file_identity(metadata)


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


def _read_descriptor(
    descriptor: int,
    *,
    expected_identity: tuple[int, int, int, int, int, int, int | None, int | None],
    description: str,
) -> bytes:
    before = os.fstat(descriptor)
    _require_owned_mode(before, directory=False, description=description)
    if _file_identity(before) != expected_identity:
        raise ValueError(f"{description} identity drift")
    if before.st_size > _CONTROL_FILE_LIMIT:
        raise ValueError(f"{description} exceeds the bounded control-file limit")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    byte_count = 0
    while True:
        chunk = os.read(descriptor, _READ_CHUNK_SIZE)
        if not chunk:
            break
        chunks.append(chunk)
        byte_count += len(chunk)
    after = os.fstat(descriptor)
    if (
        _file_identity(after) != expected_identity
        or byte_count != before.st_size
        or os.read(descriptor, 1)
    ):
        raise ValueError(f"{description} changed while reading")
    return b"".join(chunks)


def _digest_descriptor(
    descriptor: int,
    *,
    expected_identity: tuple[int, int, int, int, int, int, int | None, int | None],
    description: str,
) -> tuple[int, str]:
    before = os.fstat(descriptor)
    _require_owned_mode(before, directory=False, description=description)
    if _file_identity(before) != expected_identity:
        raise ValueError(f"{description} identity drift")
    digest = hashlib.sha256()
    byte_count = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, _READ_CHUNK_SIZE)
        if not chunk:
            break
        byte_count += len(chunk)
        digest.update(chunk)
    after = os.fstat(descriptor)
    if (
        _file_identity(after) != expected_identity
        or byte_count != before.st_size
        or os.read(descriptor, 1)
    ):
        raise ValueError(f"{description} changed while hashing")
    return byte_count, digest.hexdigest()


def _open_bound_directory(
    parent_fd: int,
    name: str,
    description: str,
) -> tuple[
    int,
    tuple[int, int, int, int, int, int, int | None, int | None],
]:
    named_before = entry_lstat(parent_fd, name)
    _require_owned_mode(named_before, directory=True, description=description)
    descriptor, _created = open_directory_at(parent_fd, name)
    try:
        opened = os.fstat(descriptor)
        named_after = entry_lstat(parent_fd, name)
        _require_owned_mode(opened, directory=True, description=description)
        identity = _directory_identity(opened)
        if (
            _directory_identity(named_before) != identity
            or _directory_identity(named_after) != identity
        ):
            raise ValueError(f"{description} identity drift")
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _open_bound_file(
    parent_fd: int,
    name: str,
    description: str,
) -> tuple[
    int,
    tuple[int, int, int, int, int, int, int | None, int | None],
]:
    named_before = entry_lstat(parent_fd, name)
    _require_owned_mode(named_before, directory=False, description=description)
    descriptor, opened = open_regular_file_at(parent_fd, name)
    try:
        _require_owned_mode(opened, directory=False, description=description)
        identity = _file_identity(opened)
        named_after = entry_lstat(parent_fd, name)
        if (
            _file_identity(named_before) != identity
            or _file_identity(named_after) != identity
        ):
            raise ValueError(f"{description} identity drift")
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


@dataclass(frozen=True)
class VerifiedArchiveSet:
    source_lock_sha256: str
    generator_commit: str
    archives: tuple[ArtifactRecord, ...]
    members: tuple[ArtifactRecord, ...]
    _archive_descriptors: Mapping[str, int] = field(
        repr=False,
        compare=False,
    )


@dataclass
class _ArchiveAuthorityState:
    lock_parent_fd: int
    lock_name: str
    lock_parent_identity: tuple[
        int, int, int, int, int, int, int | None, int | None
    ]
    lock_fd: int
    lock_identity: tuple[int, int, int, int, int, int, int | None, int | None]
    lock_sha256: str
    root_parent_fd: int
    root_name: str
    root_parent_identity: tuple[
        int, int, int, int, int, int, int | None, int | None
    ]
    root_fd: int
    root_identity: tuple[int, int, int, int, int, int, int | None, int | None]
    wikidata_fd: int
    wikidata_identity: tuple[
        int, int, int, int, int, int, int | None, int | None
    ]
    wikidata_names: tuple[str, ...]
    archive_fds: dict[str, int]
    archive_identities: dict[
        str,
        tuple[int, int, int, int, int, int, int | None, int | None],
    ]
    archive_rows: dict[str, SourceFile]
    verified: VerifiedArchiveSet | None = None


def _archive_authority_hook(
    phase: str,
    archive_name: str | None,
    verified: VerifiedArchiveSet,
) -> None:
    del phase, archive_name, verified


def _register_archive_output(
    planned: dict[str, str],
    path: str,
    kind: str,
) -> None:
    current = PurePosixPath(path)
    previous = planned.get(path)
    if previous is not None:
        raise ValueError(f"duplicate archive output path: {path!r}")
    for other_path, other_kind in planned.items():
        other = PurePosixPath(other_path)
        if current in other.parents or other in current.parents:
            if kind == "file" or other_kind == "file":
                raise ValueError(
                    f"archive file/directory collision: {path!r}, {other_path!r}"
                )
    planned[path] = kind


def _read_member_payload(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    archive_path: str,
) -> ArtifactRecord:
    try:
        stream = archive.extractfile(member)
    except (KeyError, OSError, tarfile.TarError) as error:
        raise ValueError(
            f"archive member payload is truncated: {archive_path}:{member.name}"
        ) from error
    if stream is None:
        raise ValueError(
            f"archive member payload is unreadable: {archive_path}:{member.name}"
        )
    digest = hashlib.sha256()
    byte_count = 0
    try:
        while True:
            chunk = stream.read(_READ_CHUNK_SIZE)
            if not chunk:
                break
            byte_count += len(chunk)
            digest.update(chunk)
        trailing = stream.read(1)
    except (EOFError, OSError, tarfile.TarError) as error:
        raise ValueError(
            f"archive member payload is truncated: {archive_path}:{member.name}"
        ) from error
    finally:
        stream.close()
    if byte_count != member.size or trailing:
        raise ValueError(
            f"archive member size/EOF mismatch: {archive_path}:{member.name}"
        )
    return ArtifactRecord(
        path=f"members/{member.name}",
        bytes=byte_count,
        sha256=digest.hexdigest(),
    )


def _reject_pax_size_override(records: bytes, archive_path: str) -> None:
    """Reject PAX/GNU size overrides and structurally invalid extended records.

    ``tarfile`` lets local (``x``) and global (``g``) PAX extended headers, and
    the equivalent GNU sparse/realsize records, replace the effective payload
    length used to walk subsequent members. The raw envelope walk advances using
    only the regular header ``size``, so any such override would give the two
    layers different structural boundaries. Rejecting every size override before
    authoritative tar parsing keeps both walks byte-identical. Other extended
    metadata is tolerated only when every record is well formed.
    """
    position = 0
    total = len(records)
    while position < total:
        match = _PAX_RECORD_RE.match(records, position)
        if match is None:
            raise ValueError(
                f"archive has a malformed PAX extended header: {archive_path}"
            )
        record_length = int(match.group(1).decode("ascii"))
        if (
            record_length <= 0
            or position + record_length > total
            or records[position + record_length - 1] != 0x0A
        ):
            raise ValueError(
                f"archive has a malformed PAX extended header: {archive_path}"
            )
        if match.group(2) in _PAX_SIZE_KEYWORDS:
            raise ValueError(
                f"archive declares a PAX size override: {archive_path}"
            )
        position += record_length
    if position != total:
        raise ValueError(
            f"archive has a malformed PAX extended header: {archive_path}"
        )


def _validate_archive_envelope_duplicate(
    duplicate: int,
    archive_path: str,
) -> None:
    decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)
    buffered = bytearray()
    pax_records = bytearray()
    payload_blocks = 0
    terminal_blocks = 0
    pax_blocks = 0
    pax_size = 0

    def consume(payload: bytes) -> None:
        nonlocal payload_blocks, terminal_blocks, pax_blocks, pax_size
        buffered.extend(payload)
        offset = 0
        while len(buffered) - offset >= _TAR_BLOCK_SIZE:
            if pax_blocks:
                pax_records.extend(buffered[offset : offset + _TAR_BLOCK_SIZE])
                offset += _TAR_BLOCK_SIZE
                pax_blocks -= 1
                if pax_blocks == 0:
                    _reject_pax_size_override(
                        bytes(pax_records[:pax_size]), archive_path
                    )
                    del pax_records[:]
                continue
            available_blocks = (len(buffered) - offset) // _TAR_BLOCK_SIZE
            if payload_blocks:
                skipped = min(payload_blocks, available_blocks)
                offset += skipped * _TAR_BLOCK_SIZE
                payload_blocks -= skipped
                continue

            block = bytes(buffered[offset : offset + _TAR_BLOCK_SIZE])
            offset += _TAR_BLOCK_SIZE
            if terminal_blocks == 2:
                if block != _TAR_ZERO_BLOCK:
                    raise ValueError(
                        f"archive has trailing decompressed payload: {archive_path}"
                    )
                continue
            if terminal_blocks == 1:
                if block != _TAR_ZERO_BLOCK:
                    raise ValueError(
                        f"archive has an incomplete tar terminator: {archive_path}"
                    )
                terminal_blocks = 2
                continue
            if block == _TAR_ZERO_BLOCK:
                terminal_blocks = 1
                continue
            try:
                header = tarfile.TarInfo.frombuf(
                    block,
                    encoding="utf-8",
                    errors="surrogateescape",
                )
            except (tarfile.HeaderError, UnicodeError, ValueError) as error:
                raise ValueError(
                    f"archive has a malformed tar envelope: {archive_path}"
                ) from error
            if header.size < 0:
                raise ValueError(
                    f"archive has a malformed tar envelope: {archive_path}"
                )
            if header.type == tarfile.GNUTYPE_SPARSE:
                raise ValueError(
                    f"archive declares a GNU sparse size override: {archive_path}"
                )
            block_span = (header.size + _TAR_BLOCK_SIZE - 1) // _TAR_BLOCK_SIZE
            if header.type in _PAX_HEADER_TYPES:
                if header.size > _SOURCE_LINE_LIMIT:
                    raise ValueError(
                        f"archive PAX header exceeds the bounded limit: "
                        f"{archive_path}"
                    )
                if block_span == 0:
                    _reject_pax_size_override(b"", archive_path)
                else:
                    pax_blocks = block_span
                    pax_size = header.size
                continue
            payload_blocks = block_span
        if offset:
            del buffered[:offset]

    try:
        while not decompressor.eof:
            compressed = os.read(duplicate, _READ_CHUNK_SIZE)
            if not compressed:
                break
            remaining = compressed
            while remaining and not decompressor.eof:
                previous_size = len(remaining)
                output = decompressor.decompress(
                    remaining,
                    _READ_CHUNK_SIZE,
                )
                remaining = decompressor.unconsumed_tail
                consume(output)
                if (
                    not output
                    and len(remaining) == previous_size
                    and not decompressor.eof
                ):
                    raise ValueError(
                        f"archive gzip stream made no progress: "
                        f"{archive_path}"
                    )
        if not decompressor.eof:
            raise ValueError(f"archive gzip stream is truncated: {archive_path}")
        consume(decompressor.flush())
        if decompressor.unused_data or os.read(duplicate, 1):
            raise ValueError(
                f"archive has trailing compressed payload: {archive_path}"
            )
    except zlib.error as error:
        raise ValueError(f"archive gzip stream is malformed: {archive_path}") from error

    if buffered:
        raise ValueError(
            f"archive tar envelope is not block-aligned: {archive_path}"
        )
    if pax_blocks:
        raise ValueError(
            f"archive has a truncated PAX extended header: {archive_path}"
        )
    if payload_blocks:
        raise ValueError(f"archive member payload is truncated: {archive_path}")
    if terminal_blocks != 2:
        raise ValueError(
            f"archive tar envelope lacks two terminal zero blocks: {archive_path}"
        )


def _validate_archive_envelope(descriptor: int, archive_path: str) -> None:
    with _open_archive_duplicate_descriptor(
        descriptor,
        f"Wikidata archive envelope {archive_path}",
    ) as duplicate:
        _validate_archive_envelope_duplicate(duplicate, archive_path)


def _parse_archive_descriptor(
    descriptor: int,
    archive_path: str,
    planned: dict[str, str],
) -> tuple[ArtifactRecord, ...]:
    _validate_archive_envelope(descriptor, archive_path)
    records: list[ArtifactRecord] = []
    seen: set[str] = set()
    with _open_archive_duplicate_file(
        descriptor,
        f"Wikidata archive parser {archive_path}",
    ) as handle:
        try:
            with tarfile.open(
                fileobj=cast(BinaryIO, handle),
                mode="r:*",
            ) as archive:
                for member in archive:
                    name = _safe_relative_path(
                        member.name,
                        "archive member path",
                    )
                    kind = "directory" if member.isdir() else "file"
                    _register_archive_output(planned, name, kind)
                    if (
                        member.sparse is not None
                        or member.type == tarfile.GNUTYPE_SPARSE
                    ):
                        raise ValueError(
                            f"unsafe sparse archive member: {name!r}"
                        )
                    if not member.isreg():
                        raise ValueError(
                            f"unsafe archive member type: {name!r}"
                        )
                    if name not in ARCHIVE_MEMBERS[archive_path]:
                        raise ValueError(
                            f"undeclared archive member: "
                            f"{archive_path}:{name}"
                        )
                    seen.add(name)
                    records.append(
                        _read_member_payload(archive, member, archive_path)
                    )
        except (EOFError, OSError, tarfile.TarError) as error:
            raise ValueError(
                f"archive is truncated or malformed: {archive_path}"
            ) from error
    if seen != set(ARCHIVE_MEMBERS[archive_path]):
        missing = sorted(set(ARCHIVE_MEMBERS[archive_path]) - seen, key=_byte_key)
        raise ValueError(
            f"archive member inventory is incomplete: "
            f"{archive_path}:{missing[0] if missing else 'duplicate'}"
        )
    return tuple(records)


def _check_named_file(
    parent_fd: int,
    name: str,
    descriptor: int,
    expected: tuple[int, int, int, int, int, int, int | None, int | None],
    description: str,
) -> None:
    try:
        opened = os.fstat(descriptor)
        named = entry_lstat(parent_fd, name)
        _require_owned_mode(opened, directory=False, description=description)
        _require_owned_mode(named, directory=False, description=description)
    except (OSError, ValueError) as error:
        raise ValueError(f"{description} namespace identity drift") from error
    if _file_identity(opened) != expected or _file_identity(named) != expected:
        raise ValueError(f"{description} namespace identity drift")


def _check_named_directory(
    parent_fd: int,
    name: str,
    descriptor: int,
    expected: tuple[int, int, int, int, int, int, int | None, int | None],
    description: str,
) -> None:
    try:
        opened = os.fstat(descriptor)
        named = entry_lstat(parent_fd, name)
        _require_owned_mode(opened, directory=True, description=description)
        _require_owned_mode(named, directory=True, description=description)
    except (OSError, ValueError) as error:
        raise ValueError(f"{description} identity drift") from error
    if (
        _directory_identity(opened) != expected
        or _directory_identity(named) != expected
    ):
        raise ValueError(f"{description} identity drift")


def _verify_authority_state(
    state: _ArchiveAuthorityState,
    *,
    rehash: bool,
) -> None:
    try:
        if (
            _directory_identity(os.fstat(state.lock_parent_fd))
            != state.lock_parent_identity
        ):
            raise ValueError("source-lock parent identity drift")
        _check_named_file(
            state.lock_parent_fd,
            state.lock_name,
            state.lock_fd,
            state.lock_identity,
            "source lock",
        )
        if rehash:
            lock_payload = _read_descriptor(
                state.lock_fd,
                expected_identity=state.lock_identity,
                description="source lock",
            )
            if hashlib.sha256(lock_payload).hexdigest() != state.lock_sha256:
                raise ValueError("source lock changed during verification")

        if (
            _directory_identity(os.fstat(state.root_parent_fd))
            != state.root_parent_identity
        ):
            raise ValueError("source-root parent identity drift")
        _check_named_directory(
            state.root_parent_fd,
            state.root_name,
            state.root_fd,
            state.root_identity,
            "source root",
        )
        for archive_path in ARCHIVE_PATHS:
            descriptor = state.archive_fds[archive_path]
            identity = state.archive_identities[archive_path]
            _check_named_file(
                state.wikidata_fd,
                archive_path,
                descriptor,
                identity,
                "archive",
            )
            if rehash:
                size, sha256 = _digest_descriptor(
                    descriptor,
                    expected_identity=identity,
                    description=f"Wikidata archive {archive_path}",
                )
                expected = state.archive_rows[archive_path]
                if size != expected.bytes or sha256 != expected.sha256:
                    raise ValueError(
                        f"archive changed during verification: {archive_path}"
                    )
        _check_named_directory(
            state.root_fd,
            "wikidata5m",
            state.wikidata_fd,
            state.wikidata_identity,
            "Wikidata source directory",
        )
        if list_entries(state.wikidata_fd) != state.wikidata_names:
            raise ValueError("Wikidata source directory identity drift")
    except OSError as error:
        raise ValueError("archive authority identity drift") from error


def _load_pinned_source_lock(
    descriptor: int,
    identity: tuple[int, int, int, int, int, int, int | None, int | None],
    *,
    expected_generator_commit: str,
) -> tuple[SourceLock, bytes]:
    payload = _read_descriptor(
        descriptor,
        expected_identity=identity,
        description="source lock",
    )
    value = source_lock_module._strict_json_bytes(payload, "source lock")
    lock = SourceLock.from_dict(value)
    if lock.to_bytes() != payload:
        raise ValueError("source lock JSON is not canonical")
    source_lock_module._authorize_source_lock(
        lock,
        expected_generator_commit=expected_generator_commit,
    )
    return lock, payload


def _precheck_pinned_source_lock(
    source_lock_path: Path,
    *,
    expected_generator_commit: str,
) -> None:
    parent_fd = -1
    descriptor = -1
    try:
        parent_fd, name = open_parent_directory(Path(source_lock_path))
        parent_identity = _directory_identity(os.fstat(parent_fd))
        _require_owned_mode(
            os.fstat(parent_fd),
            directory=True,
            description="source-lock parent",
        )
        descriptor, identity = _open_bound_file(
            parent_fd,
            name,
            "source lock",
        )
        _load_pinned_source_lock(
            descriptor,
            identity,
            expected_generator_commit=expected_generator_commit,
        )
        _check_named_file(
            parent_fd,
            name,
            descriptor,
            identity,
            "source lock",
        )
        if _directory_identity(os.fstat(parent_fd)) != parent_identity:
            raise ValueError("source-lock parent identity drift")
    except OSError as error:
        raise ValueError("source lock authority is missing or unsafe") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_fd >= 0:
            os.close(parent_fd)


@contextmanager
def _open_verified_archives(
    source_lock_path: Path,
    source_root: Path,
    *,
    expected_generator_commit: str,
) -> Iterator[VerifiedArchiveSet]:
    expected_commit = _validate_commit(
        expected_generator_commit,
        "expected generator commit",
    )
    lock_parent_fd = -1
    lock_fd = -1
    root_parent_fd = -1
    root_fd = -1
    wikidata_fd = -1
    archive_fds: dict[str, int] = {}
    state: _ArchiveAuthorityState | None = None
    body_raised = False
    primary_error: BaseException | None = None
    try:
        lock_parent_fd, lock_name = open_parent_directory(Path(source_lock_path))
        lock_parent_metadata = os.fstat(lock_parent_fd)
        _require_owned_mode(
            lock_parent_metadata,
            directory=True,
            description="source-lock parent",
        )
        lock_parent_identity = _directory_identity(lock_parent_metadata)
        lock_fd, lock_identity = _open_bound_file(
            lock_parent_fd,
            lock_name,
            "source lock",
        )
        lock, lock_payload = _load_pinned_source_lock(
            lock_fd,
            lock_identity,
            expected_generator_commit=expected_commit,
        )
        lock_sha256 = hashlib.sha256(lock_payload).hexdigest()
        if lock_sha256 != lock.sha256:
            raise ValueError("source lock digest identity mismatch")

        wikidata_entries = tuple(
            entry for entry in lock.sources if entry.source_id == "wikidata5m"
        )
        if (
            len(wikidata_entries) != 1
            or wikidata_entries[0].materialized_path != "wikidata5m"
        ):
            raise ValueError("source lock has no canonical Wikidata authority")
        wikidata_entry = wikidata_entries[0]
        metadata_paths = {"README.md", *wikidata_entry.license_files}
        archive_rows = {
            row.path: row
            for row in wikidata_entry.files
            if row.path not in metadata_paths
        }
        if tuple(archive_rows) != ARCHIVE_PATHS:
            raise ValueError("source lock archive inventory is not canonical")

        root_parent_fd, root_name = open_parent_directory(Path(source_root))
        root_parent_metadata = os.fstat(root_parent_fd)
        _require_owned_mode(
            root_parent_metadata,
            directory=True,
            description="source-root parent",
        )
        root_parent_identity = _directory_identity(root_parent_metadata)
        root_fd, root_identity = _open_bound_directory(
            root_parent_fd,
            root_name,
            "source root",
        )
        wikidata_fd, wikidata_identity = _open_bound_directory(
            root_fd,
            "wikidata5m",
            "Wikidata source directory",
        )
        wikidata_names = list_entries(wikidata_fd)
        expected_names = tuple(
            sorted(
                (row.path for row in wikidata_entry.files),
                key=_byte_key,
            )
        )
        if wikidata_names != expected_names:
            raise ValueError("Wikidata source directory inventory drift")

        archive_identities = {}
        for archive_path in ARCHIVE_PATHS:
            descriptor, identity = _open_bound_file(
                wikidata_fd,
                archive_path,
                f"Wikidata archive {archive_path}",
            )
            archive_fds[archive_path] = descriptor
            archive_identities[archive_path] = identity

        empty_verified = VerifiedArchiveSet(
            source_lock_sha256=lock_sha256,
            generator_commit=lock.generator_commit,
            archives=(),
            members=(),
            _archive_descriptors=MappingProxyType(archive_fds),
        )
        state = _ArchiveAuthorityState(
            lock_parent_fd=lock_parent_fd,
            lock_name=lock_name,
            lock_parent_identity=lock_parent_identity,
            lock_fd=lock_fd,
            lock_identity=lock_identity,
            lock_sha256=lock_sha256,
            root_parent_fd=root_parent_fd,
            root_name=root_name,
            root_parent_identity=root_parent_identity,
            root_fd=root_fd,
            root_identity=root_identity,
            wikidata_fd=wikidata_fd,
            wikidata_identity=wikidata_identity,
            wikidata_names=wikidata_names,
            archive_fds=archive_fds,
            archive_identities=archive_identities,
            archive_rows=archive_rows,
            verified=empty_verified,
        )
        _archive_authority_hook("after_precheck", None, empty_verified)

        archive_records = []
        member_records = []
        planned_outputs: dict[str, str] = {}
        for archive_path in ARCHIVE_PATHS:
            expected = archive_rows[archive_path]
            size, sha256 = _digest_descriptor(
                archive_fds[archive_path],
                expected_identity=archive_identities[archive_path],
                description=f"Wikidata archive {archive_path}",
            )
            if size != expected.bytes or sha256 != expected.sha256:
                raise ValueError(f"locked archive drift: {archive_path}")
            archive_records.append(
                ArtifactRecord(
                    path=archive_path,
                    bytes=size,
                    sha256=sha256,
                )
            )
            _archive_authority_hook("after_hash", archive_path, empty_verified)
            member_records.extend(
                _parse_archive_descriptor(
                    archive_fds[archive_path],
                    archive_path,
                    planned_outputs,
                )
            )
            _archive_authority_hook("after_parse", archive_path, empty_verified)

        ordered_members = tuple(
            sorted(member_records, key=lambda record: _byte_key(record.path))
        )
        if tuple(record.path for record in ordered_members) != MEMBER_PATHS:
            raise ValueError("decoded archive member inventory drift")
        verified = VerifiedArchiveSet(
            source_lock_sha256=lock_sha256,
            generator_commit=lock.generator_commit,
            archives=tuple(archive_records),
            members=ordered_members,
            _archive_descriptors=MappingProxyType(archive_fds),
        )
        state.verified = verified
        _verify_authority_state(state, rehash=False)
        _archive_authority_hook("before_yield", None, verified)
        _verify_authority_state(state, rehash=False)
        try:
            yield verified
        except BaseException:
            body_raised = True
            raise
        _archive_authority_hook("before_postcheck", None, verified)
        _verify_authority_state(state, rehash=True)
    except OSError as error:
        if body_raised:
            primary_error = error
            raise
        wrapped = ValueError("Wikidata archive authority is missing or unsafe")
        primary_error = wrapped
        raise wrapped from error
    except BaseException as error:
        primary_error = error
        raise
    finally:
        close_error = _close_descriptors_exhaustively(
            (
                *archive_fds.values(),
                wikidata_fd,
                root_fd,
                root_parent_fd,
                lock_fd,
                lock_parent_fd,
            )
        )
        if close_error is not None:
            if primary_error is None:
                raise close_error
            primary_error.add_note(
                f"archive authority close also failed: {close_error!r}"
            )


def _require_derived_mode(
    metadata: os.stat_result,
    *,
    directory: bool,
    description: str,
) -> None:
    _require_owned_mode(
        metadata,
        directory=directory,
        description=description,
    )
    expected_mode = 0o700 if directory else 0o600
    if stat.S_IMODE(metadata.st_mode) != expected_mode:
        raise ValueError(f"{description} mode drift")


def _check_named_created_file(
    parent_fd: int,
    name: str,
    descriptor: int,
    expected_identity: _CreationIdentity,
    description: str,
) -> None:
    try:
        opened = os.fstat(descriptor)
        named = entry_lstat(parent_fd, name)
        _require_derived_mode(
            opened,
            directory=False,
            description=description,
        )
        _require_derived_mode(
            named,
            directory=False,
            description=description,
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"{description} identity drift") from error
    if (
        _creation_identity(opened) != expected_identity
        or _creation_identity(named) != expected_identity
    ):
        raise ValueError(f"{description} identity drift")


def _check_named_private_file(
    parent_fd: int,
    name: str,
    descriptor: int,
    expected_identity: _PrivateFileIdentity,
    description: str,
) -> None:
    try:
        opened = os.fstat(descriptor)
        named = entry_lstat(parent_fd, name)
        _require_derived_mode(
            opened,
            directory=False,
            description=description,
        )
        _require_derived_mode(
            named,
            directory=False,
            description=description,
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"{description} identity drift") from error
    if (
        _private_file_identity(opened) != expected_identity
        or _private_file_identity(named) != expected_identity
    ):
        raise ValueError(f"{description} identity drift")


def _check_named_derived_directory(
    parent_fd: int,
    name: str,
    descriptor: int,
    expected_identity: _CreationIdentity,
    description: str,
) -> None:
    try:
        opened = os.fstat(descriptor)
        named = entry_lstat(parent_fd, name)
        _require_derived_mode(
            opened,
            directory=True,
            description=description,
        )
        _require_derived_mode(
            named,
            directory=True,
            description=description,
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"{description} identity drift") from error
    if (
        _creation_identity(opened) != expected_identity
        or _creation_identity(named) != expected_identity
    ):
        raise ValueError(f"{description} identity drift")


def _close_descriptor(descriptor: int) -> None:
    os.close(descriptor)


def _atomic_exchange_directories(
    directory_fd: int,
    first_name: str,
    second_name: str,
) -> None:
    first = _safe_relative_path(first_name, "exchange entry")
    second = _safe_relative_path(second_name, "exchange entry")
    if "/" in first or "/" in second or first == second:
        raise ValueError("atomic exchange requires distinct sibling names")
    first_bytes = os.fsencode(first)
    second_bytes = os.fsencode(second)
    libc = ctypes.CDLL(None, use_errno=True)
    ctypes.set_errno(0)
    if sys.platform.startswith("linux"):
        try:
            primitive = libc.renameat2
        except AttributeError as error:
            raise RuntimeError(
                "atomic quarantine exchange requires Linux renameat2"
            ) from error
        primitive.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        primitive.restype = ctypes.c_int
        result = primitive(
            directory_fd,
            first_bytes,
            directory_fd,
            second_bytes,
            _RENAME_EXCHANGE,
        )
    elif sys.platform == "darwin":
        try:
            primitive = libc.renameatx_np
        except AttributeError as error:
            raise RuntimeError(
                "atomic quarantine exchange requires macOS renameatx_np"
            ) from error
        primitive.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        primitive.restype = ctypes.c_int
        result = primitive(
            directory_fd,
            first_bytes,
            directory_fd,
            second_bytes,
            _RENAME_SWAP,
        )
    else:
        raise RuntimeError(
            f"no atomic directory exchange for platform {sys.platform!r}"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            f"{first} <-> {second}",
        )


def _append_secondary_error(
    current: BaseException | None,
    error: BaseException,
    description: str,
) -> BaseException:
    if current is None:
        return error
    current.add_note(f"{description}: {error!r}")
    return current


def _close_descriptors_exhaustively(
    descriptors: Iterable[int],
) -> BaseException | None:
    close_error: BaseException | None = None
    for descriptor in descriptors:
        if descriptor < 0:
            continue
        try:
            _close_descriptor(descriptor)
        except BaseException as error:
            close_error = _append_secondary_error(
                close_error,
                error,
                "additional descriptor close failure",
            )
    return close_error


@contextmanager
def _open_archive_duplicate_descriptor(
    descriptor: int,
    description: str,
) -> Iterator[int]:
    duplicate = -1
    primary_error: BaseException | None = None
    try:
        duplicate = os.dup(descriptor)
        os.lseek(duplicate, 0, os.SEEK_SET)
        yield duplicate
    except BaseException as error:
        primary_error = error
        raise
    finally:
        close_error = _close_descriptors_exhaustively((duplicate,))
        if close_error is not None:
            if primary_error is None:
                raise close_error
            primary_error.add_note(
                f"{description} descriptor close also failed: "
                f"{close_error!r}"
            )


@contextmanager
def _open_archive_duplicate_file(
    descriptor: int,
    description: str,
) -> Iterator[BinaryIO]:
    with _open_archive_duplicate_descriptor(
        descriptor,
        description,
    ) as duplicate:
        handle: BinaryIO | None = None
        primary_error: BaseException | None = None
        try:
            handle = cast(
                BinaryIO,
                os.fdopen(duplicate, "rb", closefd=False),
            )
            yield handle
        except BaseException as error:
            primary_error = error
            raise
        finally:
            handle_error: BaseException | None = None
            if handle is not None:
                try:
                    handle.close()
                except BaseException as error:
                    handle_error = error
            if handle_error is not None:
                if primary_error is None:
                    raise handle_error
                primary_error.add_note(
                    f"{description} file handle close also failed: "
                    f"{handle_error!r}"
                )


def _digest_private_descriptor(
    descriptor: int,
    *,
    expected_identity: _PrivateFileIdentity,
    description: str,
) -> tuple[int, str]:
    before = os.fstat(descriptor)
    _require_derived_mode(
        before,
        directory=False,
        description=description,
    )
    if _private_file_identity(before) != expected_identity:
        raise ValueError(f"{description} identity drift")
    digest = hashlib.sha256()
    byte_count = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, _READ_CHUNK_SIZE)
        if not chunk:
            break
        byte_count += len(chunk)
        digest.update(chunk)
    after = os.fstat(descriptor)
    if (
        _private_file_identity(after) != expected_identity
        or byte_count != before.st_size
        or os.read(descriptor, 1)
    ):
        raise ValueError(f"{description} changed while hashing")
    return byte_count, digest.hexdigest()


@dataclass(frozen=True)
class _PrivateFileAuthority:
    identity: _PrivateFileIdentity
    sha256: str


@dataclass
class _PrivateBuildAuthority:
    namespace_fd: int
    name: str
    descriptor: int
    identity: _CreationIdentity
    directory_descriptors: dict[str, int] = field(default_factory=dict)
    directory_identities: dict[str, _CreationIdentity] = field(
        default_factory=dict
    )
    pending_file_identities: dict[str, _CreationIdentity] = field(
        default_factory=dict
    )
    file_authorities: dict[str, _PrivateFileAuthority] = field(
        default_factory=dict
    )
    sealed_root_identity: _PrivateFileIdentity | None = None
    sealed_directory_identities: dict[str, _PrivateFileIdentity] = field(
        default_factory=dict
    )

    def register_directory(
        self,
        relative_path: str,
        descriptor: int,
        identity: _CreationIdentity,
    ) -> None:
        path = _safe_relative_path(
            relative_path,
            "private directory path",
        )
        if (
            "/" in path
            or path in self.directory_identities
            or path in self.pending_file_identities
            or path in self.file_authorities
        ):
            raise ValueError("duplicate private directory authority")
        self.directory_descriptors[path] = descriptor
        self.directory_identities[path] = identity

    def register_file(
        self,
        relative_path: str,
        identity: _CreationIdentity,
    ) -> None:
        path = _safe_relative_path(relative_path, "private file path")
        if (
            path in self.pending_file_identities
            or path in self.file_authorities
            or path in self.directory_identities
        ):
            raise ValueError("duplicate private file authority")
        self.pending_file_identities[path] = identity

    def finalize_file(
        self,
        relative_path: str,
        creation_identity: _CreationIdentity,
        file_authority: _PrivateFileAuthority,
    ) -> None:
        if (
            self.pending_file_identities.get(relative_path)
            != creation_identity
            or relative_path in self.file_authorities
        ):
            raise ValueError("private file authority finalization drift")
        del self.pending_file_identities[relative_path]
        self.file_authorities[relative_path] = file_authority

    def forget_file(
        self,
        relative_path: str,
        file_authority: _PrivateFileAuthority,
    ) -> None:
        if self.file_authorities.get(relative_path) != file_authority:
            raise ValueError("private file authority removal drift")
        del self.file_authorities[relative_path]

    def forget_directory(
        self,
        relative_path: str,
        identity: _CreationIdentity,
    ) -> None:
        if self.directory_identities.get(relative_path) != identity:
            raise ValueError("private directory authority removal drift")
        del self.directory_identities[relative_path]
        del self.directory_descriptors[relative_path]


def _derived_view_build_hook(
    phase: str,
    authority: _PrivateBuildAuthority,
    final_name: str | None,
) -> None:
    del phase, authority, final_name


def _external_sort_hook(
    phase: str,
    work_fd: int,
    run: "_SortRun | _RunExpectation",
) -> None:
    del phase, work_fd, run


def _view_authority_hook(
    phase: str,
    view: WikidataDerivedView,
) -> None:
    del phase, view


def _open_created_file(
    directory_fd: int,
    name: str,
    relative_path: str,
    authority: _PrivateBuildAuthority | None,
) -> tuple[int, _CreationIdentity]:
    descriptor = os.open(
        name,
        _WRITE_FLAGS,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        identity = _creation_identity(os.fstat(descriptor))
        if authority is not None:
            authority.register_file(relative_path, identity)
        _check_named_created_file(
            directory_fd,
            name,
            descriptor,
            identity,
            f"created private file {relative_path}",
        )
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _finalize_private_file(
    directory_fd: int,
    name: str,
    relative_path: str,
    descriptor: int,
    creation_identity: _CreationIdentity,
    sha256: str,
    authority: _PrivateBuildAuthority,
) -> _PrivateFileAuthority:
    expected_sha256 = _validate_sha256(
        sha256,
        f"private file {relative_path} sha256",
    )
    metadata = os.fstat(descriptor)
    _require_derived_mode(
        metadata,
        directory=False,
        description=f"private file {relative_path}",
    )
    if _creation_identity(metadata) != creation_identity:
        raise ValueError(f"private file {relative_path} creation identity drift")
    file_authority = _PrivateFileAuthority(
        identity=_private_file_identity(metadata),
        sha256=expected_sha256,
    )
    _check_named_private_file(
        directory_fd,
        name,
        descriptor,
        file_authority.identity,
        f"private file {relative_path}",
    )
    authority.finalize_file(
        relative_path,
        creation_identity,
        file_authority,
    )
    return file_authority


def _verify_open_private_file(
    parent_fd: int,
    name: str,
    descriptor: int,
    file_authority: _PrivateFileAuthority,
    description: str,
    *,
    rewind: bool,
) -> None:
    _check_named_private_file(
        parent_fd,
        name,
        descriptor,
        file_authority.identity,
        description,
    )
    byte_count, sha256 = _digest_private_descriptor(
        descriptor,
        expected_identity=file_authority.identity,
        description=description,
    )
    if (
        byte_count != file_authority.identity[6]
        or sha256 != file_authority.sha256
    ):
        raise ValueError(f"{description} digest drift")
    _check_named_private_file(
        parent_fd,
        name,
        descriptor,
        file_authority.identity,
        description,
    )
    if rewind:
        os.lseek(descriptor, 0, os.SEEK_SET)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


class _ArtifactWriter:
    def __init__(
        self,
        directory_fd: int,
        name: str,
        relative_path: str,
        *,
        authority: _PrivateBuildAuthority | None = None,
    ) -> None:
        self.relative_path = _safe_relative_path(
            relative_path,
            "derived artifact path",
        )
        self.directory_fd = directory_fd
        self.name = name
        self.authority = authority
        self.descriptor, self.creation_identity = _open_created_file(
            directory_fd,
            name,
            self.relative_path,
            authority,
        )
        self.byte_count = 0
        self.digest = hashlib.sha256()
        self.closed = False

    def write(self, payload: bytes) -> None:
        if self.closed:
            raise ValueError("derived artifact writer is closed")
        if not isinstance(payload, bytes):
            raise TypeError("derived artifact payload must be bytes")
        _write_all(self.descriptor, payload)
        self.byte_count += len(payload)
        self.digest.update(payload)

    def finish(self) -> ArtifactRecord:
        if self.closed:
            raise ValueError("derived artifact writer is closed")
        os.fsync(self.descriptor)
        metadata = os.fstat(self.descriptor)
        if metadata.st_size != self.byte_count:
            raise ValueError(
                f"derived artifact {self.relative_path} size drift"
            )
        if self.authority is None:
            identity = _private_file_identity(metadata)
            _check_named_private_file(
                self.directory_fd,
                self.name,
                self.descriptor,
                identity,
                f"derived artifact {self.relative_path}",
            )
        else:
            _finalize_private_file(
                self.directory_fd,
                self.name,
                self.relative_path,
                self.descriptor,
                self.creation_identity,
                self.digest.hexdigest(),
                self.authority,
            )
        os.close(self.descriptor)
        self.descriptor = -1
        self.closed = True
        return ArtifactRecord(
            path=self.relative_path,
            bytes=self.byte_count,
            sha256=self.digest.hexdigest(),
        )

    def finish_index(
        self,
        *,
        count: int,
        record_width: int,
    ) -> IndexArtifactRecord:
        artifact = self.finish()
        return IndexArtifactRecord(
            path=artifact.path,
            bytes=artifact.bytes,
            sha256=artifact.sha256,
            count=count,
            record_width=record_width,
        )

    def abort(self) -> None:
        if not self.closed:
            os.close(self.descriptor)
            self.descriptor = -1
            self.closed = True


def _read_exact(descriptor: int, size: int, description: str) -> bytes:
    if size < 0 or size > _SORT_RECORD_LIMIT:
        raise ValueError(f"{description} size is outside the bounded contract")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(remaining, _READ_CHUNK_SIZE))
        if not chunk:
            raise ValueError(f"{description} is truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@dataclass
class _RunExpectation:
    name: str
    byte_count: int = 0
    digest: Any = field(default_factory=hashlib.sha256)

    @property
    def sha256(self) -> str:
        return self.digest.hexdigest()

    def write(self, descriptor: int, payload: bytes) -> None:
        _write_all(descriptor, payload)
        self.byte_count += len(payload)
        self.digest.update(payload)


def _write_sort_record(
    descriptor: int,
    key: bytes,
    payload: bytes,
    expectation: _RunExpectation | None = None,
) -> None:
    if (
        not isinstance(key, bytes)
        or not isinstance(payload, bytes)
        or len(key) > 0xFFFFFFFF
        or len(key) + len(payload) > _SORT_RECORD_LIMIT
    ):
        raise ValueError("external-sort record exceeds the bounded contract")
    parts = (
        _SORT_HEADER.pack(len(key), len(payload)),
        key,
        payload,
    )
    for part in parts:
        if expectation is None:
            _write_all(descriptor, part)
        else:
            expectation.write(descriptor, part)


def _read_sort_record(descriptor: int) -> tuple[bytes, bytes] | None:
    first = os.read(descriptor, _SORT_HEADER.size)
    if not first:
        return None
    header = bytearray(first)
    while len(header) < _SORT_HEADER.size:
        chunk = os.read(descriptor, _SORT_HEADER.size - len(header))
        if not chunk:
            raise ValueError("external-sort run has a truncated header")
        header.extend(chunk)
    key_size, payload_size = _SORT_HEADER.unpack(header)
    if key_size + payload_size > _SORT_RECORD_LIMIT:
        raise ValueError("external-sort run record exceeds the bounded contract")
    return (
        _read_exact(descriptor, key_size, "external-sort key"),
        _read_exact(descriptor, payload_size, "external-sort payload"),
    )


@dataclass(frozen=True)
class _SortRun:
    name: str
    identity: _PrivateFileIdentity
    sha256: str


def _open_sort_run_descriptor(work_fd: int, run: _SortRun) -> int:
    _external_sort_hook("before_open", work_fd, run)
    descriptor = -1
    try:
        descriptor, metadata = open_regular_file_at(work_fd, run.name)
        _require_derived_mode(
            metadata,
            directory=False,
            description="external-sort run",
        )
        _verify_open_private_file(
            work_fd,
            run.name,
            descriptor,
            _PrivateFileAuthority(
                identity=run.identity,
                sha256=run.sha256,
            ),
            "external-sort run",
            rewind=True,
        )
        return descriptor
    except BaseException as error:
        close_error = _close_descriptors_exhaustively((descriptor,))
        if close_error is not None:
            error.add_note(
                f"external-sort run reopen close also failed: "
                f"{close_error!r}"
            )
        raise


def _unlink_open_sort_run(
    work_fd: int,
    run: _SortRun,
    descriptor: int,
    authority: _PrivateBuildAuthority,
) -> None:
    file_authority = _PrivateFileAuthority(
        identity=run.identity,
        sha256=run.sha256,
    )
    _check_named_private_file(
        work_fd,
        run.name,
        descriptor,
        run.identity,
        "external-sort run before unlink",
    )
    os.unlink(run.name, dir_fd=work_fd)
    authority.forget_file(f".work/{run.name}", file_authority)


@contextmanager
def _open_sorted_run(
    work_fd: int,
    run: _SortRun | None,
    authority: _PrivateBuildAuthority,
) -> Iterator[Iterator[tuple[bytes, bytes]]]:
    if run is None:
        yield iter(())
        return
    descriptor = _open_sort_run_descriptor(work_fd, run)

    def records() -> Iterator[tuple[bytes, bytes]]:
        previous: bytes | None = None
        while True:
            record = _read_sort_record(descriptor)
            if record is None:
                return
            key, payload = record
            if previous is not None and key < previous:
                raise ValueError("external-sort run ordering drift")
            previous = key
            _external_sort_hook("before_record_yield", work_fd, run)
            yield key, payload

    body_error: BaseException | None = None
    try:
        yield records()
    except BaseException as error:
        body_error = error
        raise
    finally:
        cleanup_error: BaseException | None = None
        try:
            _unlink_open_sort_run(
                work_fd,
                run,
                descriptor,
                authority,
            )
            fsync_directory(work_fd)
        except BaseException as error:
            cleanup_error = error
        close_error = _close_descriptors_exhaustively((descriptor,))
        if close_error is not None:
            cleanup_error = _append_secondary_error(
                cleanup_error,
                close_error,
                "external-sort run close failure",
            )
        if cleanup_error is not None:
            if body_error is None:
                raise cleanup_error
            body_error.add_note(
                f"external-sort cleanup also failed: {cleanup_error!r}"
            )


class _ExternalSorter:
    """Bounded in-memory runs with deterministic external merge reduction."""

    def __init__(
        self,
        work_fd: int,
        prefix: str,
        authority: _PrivateBuildAuthority,
    ) -> None:
        if re.fullmatch(r"[a-z0-9-]+", prefix) is None:
            raise ValueError("external-sort prefix is unsafe")
        self.work_fd = work_fd
        self.prefix = prefix
        self.authority = authority
        self.chunk: list[tuple[bytes, bytes]] = []
        self.chunk_bytes = 0
        self.pending_runs: dict[int, _SortRun] = {}
        self.run_counters: dict[int, int] = {}
        self.finished = False

    def add(self, key: bytes, payload: bytes = b"") -> None:
        if self.finished:
            raise ValueError("external sorter is already finalized")
        record_size = _SORT_HEADER.size + len(key) + len(payload)
        if record_size > _SORT_RECORD_LIMIT:
            raise ValueError("external-sort input record is too large")
        if self.chunk and (
            self.chunk_bytes + record_size > _SORT_CHUNK_BYTES
            or len(self.chunk) >= _SORT_CHUNK_RECORDS
        ):
            self._flush()
        self.chunk.append((key, payload))
        self.chunk_bytes += record_size

    def _new_run_name(self, level: int) -> str:
        run_number = self.run_counters.get(level, 0)
        self.run_counters[level] = run_number + 1
        return f"{self.prefix}-l{level:04d}-r{run_number:08d}.bin"

    def _store_run(self, level: int, run: _SortRun) -> None:
        while level in self.pending_runs:
            older = self.pending_runs.pop(level)
            output_name = self._new_run_name(level + 1)
            run = self._merge_batch([older, run], output_name)
            level += 1
            if level > 64:
                raise ValueError("external-sort run count exceeds uint64")
        self.pending_runs[level] = run

    def _create_run(self, name: str) -> tuple[int, _CreationIdentity]:
        descriptor, creation_identity = _open_created_file(
            self.work_fd,
            name,
            f".work/{name}",
            self.authority,
        )
        return descriptor, creation_identity

    def _finalize_run(
        self,
        name: str,
        descriptor: int,
        creation_identity: _CreationIdentity,
        expectation: _RunExpectation,
        hook_phase: str,
    ) -> _SortRun:
        _external_sort_hook(hook_phase, self.work_fd, expectation)
        metadata = os.fstat(descriptor)
        file_identity = _private_file_identity(metadata)
        observed_bytes, observed_sha256 = _digest_private_descriptor(
            descriptor,
            expected_identity=file_identity,
            description="external-sort output run",
        )
        if (
            observed_bytes != expectation.byte_count
            or observed_sha256 != expectation.sha256
        ):
            raise ValueError(
                "external-sort output run does not match independently "
                "accumulated expected byte count and digest"
            )
        file_authority = _finalize_private_file(
            self.work_fd,
            name,
            f".work/{name}",
            descriptor,
            creation_identity,
            expectation.sha256,
            self.authority,
        )
        return _SortRun(
            name=name,
            identity=file_authority.identity,
            sha256=file_authority.sha256,
        )

    def _flush(self) -> None:
        if not self.chunk:
            return
        self.chunk.sort(key=lambda record: record[0])
        name = self._new_run_name(0)
        descriptor, creation_identity = self._create_run(name)
        expectation = _RunExpectation(name=name)
        run: _SortRun | None = None
        body_error: BaseException | None = None
        try:
            for key, payload in self.chunk:
                _write_sort_record(
                    descriptor,
                    key,
                    payload,
                    expectation,
                )
            os.fsync(descriptor)
            run = self._finalize_run(
                name,
                descriptor,
                creation_identity,
                expectation,
                "before_flush_finalize",
            )
        except BaseException as error:
            body_error = error
            raise
        finally:
            close_error = _close_descriptors_exhaustively((descriptor,))
            if close_error is not None:
                if body_error is None:
                    raise close_error
                body_error.add_note(
                    f"external-sort run writer close also failed: "
                    f"{close_error!r}"
                )
        if run is None:
            raise RuntimeError("external-sort run finalization did not complete")
        self.chunk.clear()
        self.chunk_bytes = 0
        self._store_run(0, run)

    def _merge_batch(
        self,
        runs: list[_SortRun],
        output_name: str,
    ) -> _SortRun:
        inputs: list[tuple[_SortRun, int]] = []
        output = -1
        output_run: _SortRun | None = None
        output_creation_identity: _CreationIdentity | None = None
        output_expectation = _RunExpectation(name=output_name)
        body_error: BaseException | None = None
        try:
            for run in runs:
                inputs.append(
                    (run, _open_sort_run_descriptor(self.work_fd, run))
                )
            output, output_creation_identity = self._create_run(output_name)
            heap: list[tuple[bytes, int, bytes]] = []
            for index, (_run, descriptor) in enumerate(inputs):
                record = _read_sort_record(descriptor)
                if record is not None:
                    key, payload = record
                    heapq.heappush(heap, (key, index, payload))
            previous: bytes | None = None
            while heap:
                key, index, payload = heapq.heappop(heap)
                if previous is not None and key < previous:
                    raise ValueError("external-sort merge ordering drift")
                previous = key
                _write_sort_record(
                    output,
                    key,
                    payload,
                    output_expectation,
                )
                record = _read_sort_record(inputs[index][1])
                if record is not None:
                    next_key, next_payload = record
                    heapq.heappush(
                        heap,
                        (next_key, index, next_payload),
                    )
            os.fsync(output)
            output_run = self._finalize_run(
                output_name,
                output,
                output_creation_identity,
                output_expectation,
                "before_merge_finalize",
            )
            for run, descriptor in inputs:
                _check_named_private_file(
                    self.work_fd,
                    run.name,
                    descriptor,
                    run.identity,
                    "external-sort merge input before unlink",
                )
            for run, descriptor in inputs:
                _unlink_open_sort_run(
                    self.work_fd,
                    run,
                    descriptor,
                    self.authority,
                )
            fsync_directory(self.work_fd)
            return output_run
        except BaseException as error:
            body_error = error
            raise
        finally:
            hook_error: BaseException | None = None
            if output_run is not None:
                try:
                    _external_sort_hook(
                        "before_merge_close",
                        self.work_fd,
                        output_run,
                    )
                except BaseException as error:
                    hook_error = error
            close_error = _close_descriptors_exhaustively(
                (output, *(descriptor for _run, descriptor in inputs))
            )
            secondary_error = hook_error
            if close_error is not None:
                secondary_error = _append_secondary_error(
                    secondary_error,
                    close_error,
                    "external-sort merge close failure",
                )
            if secondary_error is not None:
                if body_error is None:
                    raise secondary_error
                body_error.add_note(
                    f"external-sort merge cleanup also failed: "
                    f"{secondary_error!r}"
                )

    def finish(self) -> _SortRun | None:
        if self.finished:
            raise ValueError("external sorter is already finalized")
        self.finished = True
        self._flush()
        runs = [
            self.pending_runs[level]
            for level in sorted(self.pending_runs, reverse=True)
        ]
        level = 65
        while len(runs) > 1:
            merged: list[_SortRun] = []
            for start in range(0, len(runs), _SORT_MERGE_FAN_IN):
                batch = runs[start : start + _SORT_MERGE_FAN_IN]
                if len(batch) == 1:
                    merged.append(batch[0])
                else:
                    output_name = self._new_run_name(level)
                    merged.append(self._merge_batch(batch, output_name))
            runs = merged
            level += 1
        return runs[0] if runs else None


def _iter_descriptor_lines(
    descriptor: int,
    description: str,
) -> Iterator[tuple[int, bytes]]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    pending = bytearray()
    offset = 0
    while True:
        chunk = os.read(descriptor, _READ_CHUNK_SIZE)
        if chunk:
            pending.extend(chunk)
        while True:
            newline = pending.find(b"\n")
            if newline < 0:
                break
            line = bytes(pending[: newline + 1])
            del pending[: newline + 1]
            if len(line) > _SOURCE_LINE_LIMIT:
                raise ValueError(f"{description} row exceeds the bounded limit")
            yield offset, line
            offset += len(line)
        if len(pending) > _SOURCE_LINE_LIMIT:
            raise ValueError(f"{description} row exceeds the bounded limit")
        if not chunk:
            break
    if pending:
        yield offset, bytes(pending)


def _source_row_bytes(line: bytes) -> bytes:
    row = line[:-1] if line.endswith(b"\n") else line
    if row.endswith(b"\r"):
        row = row[:-1]
    return row


def _numeric_identifier(
    value: bytes,
    prefix: bytes,
    description: str,
    *,
    maximum: int = _UINT64_MAX,
) -> int:
    if (
        len(value) < 2
        or value[:1] != prefix
        or not value[1:].isdigit()
        or (len(value) > 2 and value[1:2] == b"0")
    ):
        raise ValueError(f"{description} is not a canonical identifier")
    number = int(value[1:])
    if number > maximum:
        raise ValueError(f"{description} exceeds the unsigned index range")
    return number


def _alias_key(prefix: bytes, number: int) -> int:
    if number > _ALIAS_NUMERIC_MAX:
        raise ValueError("alias canonical ID exceeds the kind/id key range")
    return number | (_ALIAS_RELATION_BIT if prefix == b"P" else 0)


def _canonical_id_for_alias_key(key: int) -> str:
    prefix = "P" if key & _ALIAS_RELATION_BIT else "Q"
    return f"{prefix}{key & _ALIAS_NUMERIC_MAX}"


def _canonical_json_string(value: str) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _materialize_member_files(
    verified: VerifiedArchiveSet,
    members_fd: int,
    authority: _PrivateBuildAuthority,
) -> tuple[ArtifactRecord, ...]:
    expected = {record.path: record for record in verified.members}
    records: list[ArtifactRecord] = []
    for archive_path in ARCHIVE_PATHS:
        try:
            with _open_archive_duplicate_file(
                verified._archive_descriptors[archive_path],
                f"Wikidata archive materialization {archive_path}",
            ) as handle:
                with tarfile.open(
                    fileobj=cast(BinaryIO, handle),
                    mode="r:*",
                ) as archive:
                    for member in archive:
                        name = _safe_relative_path(
                            member.name,
                            "archive member path",
                        )
                        if (
                            not member.isreg()
                            or name not in ARCHIVE_MEMBERS[archive_path]
                        ):
                            raise ValueError(
                                f"archive member changed during materialization: "
                                f"{archive_path}:{name}"
                            )
                        try:
                            stream = archive.extractfile(member)
                        except (KeyError, OSError, tarfile.TarError) as error:
                            raise ValueError(
                                f"archive member payload is truncated: "
                                f"{archive_path}:{name}"
                            ) from error
                        if stream is None:
                            raise ValueError(
                                f"archive member payload is unreadable: "
                                f"{archive_path}:{name}"
                            )
                        writer = _ArtifactWriter(
                            members_fd,
                            name,
                            f"members/{name}",
                            authority=authority,
                        )
                        try:
                            copied = 0
                            while True:
                                chunk = stream.read(_READ_CHUNK_SIZE)
                                if not chunk:
                                    break
                                copied += len(chunk)
                                writer.write(chunk)
                            if stream.read(1) or copied != member.size:
                                raise ValueError(
                                    f"archive member size/EOF mismatch: "
                                    f"{archive_path}:{name}"
                                )
                            artifact = writer.finish()
                        except BaseException:
                            writer.abort()
                            raise
                        finally:
                            stream.close()
                        locked = expected.get(artifact.path)
                        if locked is None or artifact != locked:
                            raise ValueError(
                                f"decoded member drift: {archive_path}:{name}"
                            )
                        records.append(artifact)
        except (EOFError, OSError, tarfile.TarError) as error:
            raise ValueError(
                f"archive changed during member materialization: {archive_path}"
            ) from error
    ordered = tuple(sorted(records, key=lambda record: _byte_key(record.path)))
    if ordered != verified.members:
        raise ValueError("materialized member inventory drift")
    fsync_directory(members_fd)
    return ordered


@contextmanager
def _open_member_file(
    members_fd: int,
    member_name: str,
    authority: _PrivateBuildAuthority,
) -> Iterator[int]:
    relative_path = f"members/{member_name}"
    file_authority = authority.file_authorities.get(relative_path)
    if file_authority is None:
        raise ValueError(f"decoded member authority is missing: {member_name}")
    descriptor = -1
    body_error: BaseException | None = None
    try:
        descriptor, metadata = open_regular_file_at(members_fd, member_name)
        _require_derived_mode(
            metadata,
            directory=False,
            description=f"decoded member {member_name}",
        )
        _verify_open_private_file(
            members_fd,
            member_name,
            descriptor,
            file_authority,
            f"decoded member {member_name}",
            rewind=True,
        )
        yield descriptor
    except BaseException as error:
        body_error = error
        raise
    finally:
        postcheck_error: BaseException | None = None
        if descriptor >= 0:
            try:
                _check_named_private_file(
                    members_fd,
                    member_name,
                    descriptor,
                    file_authority.identity,
                    f"decoded member {member_name} postcheck",
                )
            except BaseException as error:
                postcheck_error = error
        close_error = _close_descriptors_exhaustively((descriptor,))
        if close_error is not None:
            postcheck_error = _append_secondary_error(
                postcheck_error,
                close_error,
                "decoded member close failure",
            )
        if postcheck_error is not None:
            if body_error is None:
                raise postcheck_error
            body_error.add_note(
                f"decoded member postcheck also failed: "
                f"{postcheck_error!r}"
            )


def _iter_member_triples(
    members_fd: int,
    member_name: str,
    authority: _PrivateBuildAuthority,
) -> Iterator[tuple[int, int, int, int]]:
    with _open_member_file(
        members_fd,
        member_name,
        authority,
    ) as descriptor:
        source_row = 0
        for _offset, line in _iter_descriptor_lines(
            descriptor,
            f"Wikidata member {member_name}",
        ):
            source_row += 1
            fields = _source_row_bytes(line).split(b"\t")
            if len(fields) != 3:
                raise ValueError(
                    f"{member_name}:{source_row}: "
                    "expected 3 tab-separated fields"
                )
            try:
                subject = _numeric_identifier(
                    fields[0],
                    b"Q",
                    "triple subject",
                )
                relation = _numeric_identifier(
                    fields[1],
                    b"P",
                    "triple relation",
                )
                object_id = _numeric_identifier(
                    fields[2],
                    b"Q",
                    "triple object",
                )
            except ValueError as error:
                raise ValueError(
                    f"{member_name}:{source_row}: {error}"
                ) from error
            yield source_row, subject, relation, object_id


def _training_stream_line(
    training_split: str,
    row: int,
    subject: int,
    relation: int,
    object_id: int,
) -> bytes:
    return (
        f"{training_split}\t{row}\tQ{subject}\tP{relation}\tQ{object_id}\n"
    ).encode("ascii")


def _build_training_artifacts(
    members_fd: int,
    streams_fd: int,
    indexes_fd: int,
    work_fd: int,
    authority: _PrivateBuildAuthority,
) -> tuple[
    ArtifactRecord,
    ArtifactRecord,
    tuple[IndexArtifactRecord, IndexArtifactRecord],
    int,
    int,
]:
    training_writer = _ArtifactWriter(
        streams_fd,
        "training.tsv",
        "streams/training.tsv",
        authority=authority,
    )
    distinct_writer = _ArtifactWriter(
        streams_fd,
        "distinct-edges.tsv",
        "streams/distinct-edges.tsv",
        authority=authority,
    )
    index_writers = {
        "inductive_train": _ArtifactWriter(
            indexes_fd,
            "inductive-training-offsets.bin",
            "indexes/inductive-training-offsets.bin",
            authority=authority,
        ),
        "transductive_train": _ArtifactWriter(
            indexes_fd,
            "transductive-training-offsets.bin",
            "indexes/transductive-training-offsets.bin",
            authority=authority,
        ),
    }
    counts = {split: 0 for split in TRAINING_SPLITS}
    edge_sorter = _ExternalSorter(
        work_fd,
        "training-edges",
        authority,
    )
    try:
        for (
            source_name,
            member_name,
            _archive_path,
            source_rank,
            is_training,
        ) in _TRAINING_SOURCE_ROWS:
            for row, subject, relation, object_id in _iter_member_triples(
                members_fd,
                member_name,
                authority,
            ):
                if is_training:
                    training_split = source_name
                    if training_split not in counts:
                        raise ValueError("training source mapping drift")
                    offset = training_writer.byte_count
                    index_writers[training_split].write(_UINT64.pack(offset))
                    training_writer.write(
                        _training_stream_line(
                            training_split,
                            row,
                            subject,
                            relation,
                            object_id,
                        )
                    )
                    counts[training_split] += 1
                edge_sorter.add(
                    _EDGE_SORT_KEY.pack(
                        subject,
                        relation,
                        object_id,
                        source_rank,
                        row,
                    )
                )

        distinct_count = 0
        edge_run = edge_sorter.finish()
        with _open_sorted_run(
            work_fd,
            edge_run,
            authority,
        ) as edge_records:
            current_edge: bytes | None = None
            first_training: tuple[int, int] | None = None
            has_sealed = False

            def finish_edge() -> None:
                nonlocal distinct_count
                if current_edge is None:
                    return
                if first_training is not None and has_sealed:
                    subject, relation, object_id = struct.unpack(
                        ">QQQ",
                        current_edge,
                    )
                    raise ValueError(
                        "training/sealed overlap for "
                        f"Q{subject}/P{relation}/Q{object_id}"
                    )
                if first_training is None:
                    return
                source_rank, row = first_training
                training_split = TRAINING_SPLITS[source_rank]
                subject, relation, object_id = struct.unpack(
                    ">QQQ",
                    current_edge,
                )
                distinct_writer.write(
                    _training_stream_line(
                        training_split,
                        row,
                        subject,
                        relation,
                        object_id,
                    )
                )
                distinct_count += 1

            for key, payload in edge_records:
                if payload or len(key) != _EDGE_SORT_KEY.size:
                    raise ValueError("training edge external-sort drift")
                subject, relation, object_id, source_rank, row = (
                    _EDGE_SORT_KEY.unpack(key)
                )
                edge = key[:24]
                if current_edge != edge:
                    finish_edge()
                    current_edge = edge
                    first_training = None
                    has_sealed = False
                if source_rank < len(TRAINING_SPLITS):
                    if first_training is None:
                        first_training = (source_rank, row)
                else:
                    has_sealed = True
            finish_edge()

        training_artifact = training_writer.finish()
        distinct_artifact = distinct_writer.finish()
        index_artifacts = tuple(
            index_writers[split].finish_index(
                count=counts[split],
                record_width=_UINT64.size,
            )
            for split in TRAINING_SPLITS
        )
    except BaseException:
        training_writer.abort()
        distinct_writer.abort()
        for writer in index_writers.values():
            writer.abort()
        raise
    fsync_directory(streams_fd)
    fsync_directory(indexes_fd)
    return (
        training_artifact,
        distinct_artifact,
        cast(
            tuple[IndexArtifactRecord, IndexArtifactRecord],
            index_artifacts,
        ),
        sum(counts.values()),
        distinct_count,
    )


def _iter_alias_member_rows(
    members_fd: int,
    member_name: str,
    prefix: bytes,
    authority: _PrivateBuildAuthority,
) -> Iterator[tuple[int, int, tuple[tuple[int, bytes, bytes], ...]]]:
    with _open_member_file(
        members_fd,
        member_name,
        authority,
    ) as descriptor:
        source_row = 0
        for _offset, line in _iter_descriptor_lines(
            descriptor,
            f"Wikidata alias member {member_name}",
        ):
            source_row += 1
            fields = _source_row_bytes(line).split(b"\t")
            if len(fields) < 2:
                raise ValueError(
                    f"{member_name}:{source_row}: "
                    "expected a canonical ID and at least one alias"
                )
            try:
                numeric_id = _numeric_identifier(
                    fields[0],
                    prefix,
                    "alias canonical ID",
                    maximum=_ALIAS_NUMERIC_MAX,
                )
            except ValueError as error:
                raise ValueError(
                    f"{member_name}:{source_row}: {error}"
                ) from error
            aliases: list[tuple[int, bytes, bytes]] = []
            for position, raw in enumerate(fields[1:]):
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise ValueError(
                        f"{member_name}:{source_row}: alias is not UTF-8"
                    ) from error
                if "\x00" in text:
                    raise ValueError(
                        f"{member_name}:{source_row}: alias contains NUL"
                    )
                display = " ".join(
                    unicodedata.normalize("NFKC", text).split()
                )
                if not display:
                    continue
                normalized = " ".join(
                    unicodedata.normalize("NFKC", display).split()
                ).casefold()
                normalized_bytes = normalized.encode("utf-8")
                display_bytes = display.encode("utf-8")
                if (
                    len(normalized_bytes) + len(display_bytes)
                    > _SORT_RECORD_LIMIT
                ):
                    raise ValueError(
                        f"{member_name}:{source_row}: alias is too large"
                    )
                aliases.append(
                    (position, normalized_bytes, display_bytes)
                )
            yield source_row, _alias_key(prefix, numeric_id), tuple(aliases)


def _build_alias_artifacts(
    members_fd: int,
    streams_fd: int,
    indexes_fd: int,
    work_fd: int,
    authority: _PrivateBuildAuthority,
) -> tuple[ArtifactRecord, IndexArtifactRecord, int]:
    canonical_sorter = _ExternalSorter(
        work_fd,
        "alias-canonical",
        authority,
    )
    occurrence_sorter = _ExternalSorter(
        work_fd,
        "alias-occurrence",
        authority,
    )
    for member_name, prefix in (
        ("wikidata5m_entity.txt", b"Q"),
        ("wikidata5m_relation.txt", b"P"),
    ):
        for _row, canonical_key, aliases in _iter_alias_member_rows(
            members_fd,
            member_name,
            prefix,
            authority,
        ):
            canonical_sorter.add(_UINT64.pack(canonical_key))
            for position, normalized, display in aliases:
                occurrence_sorter.add(
                    normalized
                    + b"\0"
                    + struct.pack(">QQ", canonical_key, position),
                    display,
                )

    survivor_sorter = _ExternalSorter(
        work_fd,
        "alias-survivor",
        authority,
    )
    occurrence_run = occurrence_sorter.finish()
    with _open_sorted_run(
        work_fd,
        occurrence_run,
        authority,
    ) as occurrences:
        current_normalized: bytes | None = None
        current_owner: int | None = None
        first_key = b""
        first_display = b""
        ambiguous = False

        def finish_normalized() -> None:
            if current_normalized is not None and not ambiguous:
                survivor_sorter.add(first_key, first_display)

        for key, display in occurrences:
            separator = key.find(b"\0")
            if separator < 0 or len(key) - separator - 1 != 16:
                raise ValueError("alias occurrence external-sort drift")
            normalized = key[:separator]
            canonical_key, _position = struct.unpack(
                ">QQ",
                key[separator + 1 :],
            )
            if normalized != current_normalized:
                finish_normalized()
                current_normalized = normalized
                current_owner = canonical_key
                first_key = key[separator + 1 :]
                first_display = display
                ambiguous = False
            elif canonical_key != current_owner:
                ambiguous = True
        finish_normalized()

    canonical_run = canonical_sorter.finish()
    survivor_run = survivor_sorter.finish()
    alias_writer = _ArtifactWriter(
        streams_fd,
        "aliases.tsv",
        "streams/aliases.tsv",
        authority=authority,
    )
    index_writer = _ArtifactWriter(
        indexes_fd,
        "aliases.bin",
        "indexes/aliases.bin",
        authority=authority,
    )
    alias_count = 0
    try:
        with (
            _open_sorted_run(
                work_fd,
                canonical_run,
                authority,
            ) as canonicals,
            _open_sorted_run(
                work_fd,
                survivor_run,
                authority,
            ) as survivors,
        ):
            survivor = next(survivors, None)
            previous_canonical: int | None = None
            for key, payload in canonicals:
                if payload or len(key) != _UINT64.size:
                    raise ValueError("alias canonical external-sort drift")
                canonical_key = _UINT64.unpack(key)[0]
                if (
                    previous_canonical is not None
                    and canonical_key <= previous_canonical
                ):
                    raise ValueError(
                        "duplicate alias canonical ID in source members"
                    )
                previous_canonical = canonical_key
                if (
                    survivor is not None
                    and _UINT64.unpack(survivor[0][:8])[0] < canonical_key
                ):
                    raise ValueError("alias survivor has no canonical owner")

                canonical_id = _canonical_id_for_alias_key(canonical_key)
                row_offset = alias_writer.byte_count
                alias_writer.write(
                    canonical_id.encode("ascii")
                    + b'\t{"aliases":['
                )
                first = True
                display: str | None = None
                previous_position: int | None = None
                while survivor is not None:
                    survivor_key, display_bytes = survivor
                    if len(survivor_key) != 16:
                        raise ValueError("alias survivor index drift")
                    owner, position = struct.unpack(">QQ", survivor_key)
                    if owner != canonical_key:
                        break
                    if (
                        previous_position is not None
                        and position <= previous_position
                    ):
                        raise ValueError("alias survivor ordering drift")
                    previous_position = position
                    try:
                        alias = display_bytes.decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise ValueError(
                            "alias survivor is not UTF-8"
                        ) from error
                    V2AliasRecord(
                        canonical_id=canonical_id,
                        kind=(
                            "relation"
                            if canonical_key & _ALIAS_RELATION_BIT
                            else "entity"
                        ),
                        display=alias,
                        aliases=(alias,),
                    )
                    if not first:
                        alias_writer.write(b",")
                    alias_writer.write(_canonical_json_string(alias))
                    if display is None:
                        display = alias
                    first = False
                    survivor = next(survivors, None)
                if display is None:
                    display = canonical_id
                alias_writer.write(
                    b'],"display":'
                    + _canonical_json_string(display)
                    + b"}\n"
                )
                row_length = alias_writer.byte_count - row_offset
                index_writer.write(
                    _ALIAS_INDEX_RECORD.pack(
                        canonical_key,
                        row_offset,
                        row_length,
                    )
                )
                alias_count += 1
            if survivor is not None:
                raise ValueError("alias survivor has no canonical owner")
        alias_artifact = alias_writer.finish()
        index_artifact = index_writer.finish_index(
            count=alias_count,
            record_width=_ALIAS_INDEX_RECORD.size,
        )
    except BaseException:
        alias_writer.abort()
        index_writer.abort()
        raise
    fsync_directory(streams_fd)
    fsync_directory(indexes_fd)
    return alias_artifact, index_artifact, alias_count


def _derived_row(line: bytes, description: str) -> bytes:
    if not line.endswith(b"\n") or line.endswith(b"\r\n"):
        raise ValueError(f"{description} is not a canonical newline row")
    return line[:-1]


def _positive_decimal(value: bytes, description: str) -> int:
    if (
        not value
        or not value.isdigit()
        or value == b"0"
        or (len(value) > 1 and value[:1] == b"0")
    ):
        raise ValueError(f"{description} must be a canonical positive integer")
    return int(value)


def _parse_training_stream_row(
    line: bytes,
    description: str,
) -> V2TrainingTriple:
    fields = _derived_row(line, description).split(b"\t")
    if len(fields) != 5:
        raise ValueError(f"{description} must have five tab-separated fields")
    try:
        training_split = fields[0].decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{description} split is not ASCII") from error
    if training_split not in TRAINING_SPLITS:
        raise ValueError(f"{description} split is not canonical")
    row = _positive_decimal(fields[1], f"{description} row")
    subject = _numeric_identifier(
        fields[2],
        b"Q",
        f"{description} subject",
    )
    relation_number = _numeric_identifier(
        fields[3],
        b"P",
        f"{description} relation",
    )
    object_id = _numeric_identifier(
        fields[4],
        b"Q",
        f"{description} object",
    )
    archive_path = (
        "wikidata5m_inductive.tar.gz"
        if training_split == "inductive_train"
        else "wikidata5m_transductive.tar.gz"
    )
    return V2TrainingTriple(
        training_split=training_split,
        row=row,
        subject=subject,
        relation=f"P{relation_number}",
        object=object_id,
        member=f"wikidata5m_{training_split}.txt",
        archive_path=archive_path,
    )


def _parse_alias_stream_row(
    line: bytes,
    description: str,
) -> V2AliasRecord:
    row = _derived_row(line, description)
    try:
        canonical_bytes, payload = row.split(b"\t", 1)
    except ValueError as error:
        raise ValueError(
            f"{description} must have two tab-separated fields"
        ) from error
    if canonical_bytes[:1] not in {b"Q", b"P"}:
        raise ValueError(f"{description} canonical ID prefix is invalid")
    prefix = canonical_bytes[:1]
    number = _numeric_identifier(
        canonical_bytes,
        prefix,
        f"{description} canonical ID",
        maximum=_ALIAS_NUMERIC_MAX,
    )
    canonical_id = f"{prefix.decode('ascii')}{number}"
    value = _strict_json_bytes(payload, f"{description} alias payload")
    if (
        not isinstance(value, dict)
        or set(value) != {"aliases", "display"}
        or type(value["display"]) is not str
        or not isinstance(value["aliases"], list)
        or not all(type(alias) is str for alias in value["aliases"])
    ):
        raise ValueError(f"{description} alias payload schema drift")
    if canonical_json_bytes(value) != payload + b"\n":
        raise ValueError(f"{description} alias payload is not canonical JSON")
    return V2AliasRecord(
        canonical_id=canonical_id,
        kind="entity" if prefix == b"Q" else "relation",
        display=value["display"],
        aliases=tuple(value["aliases"]),
    )


def _pread_exact(
    descriptor: int,
    size: int,
    offset: int,
    description: str,
) -> bytes:
    if size < 0 or size > _SOURCE_LINE_LIMIT or offset < 0:
        raise ValueError(f"{description} bounds are invalid")
    chunks: list[bytes] = []
    remaining = size
    position = offset
    while remaining:
        chunk = os.pread(
            descriptor,
            min(remaining, _READ_CHUNK_SIZE),
            position,
        )
        if not chunk:
            raise ValueError(f"{description} is truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
        position += len(chunk)
    return b"".join(chunks)


def _verify_training_logical_files(
    file_fds: Mapping[str, int],
    receipt: WikidataDerivedViewReceipt,
) -> None:
    stream_fd = file_fds["streams/training.tsv"]
    index_records = {
        record.path: record for record in receipt.indexes
    }
    index_fds = {
        "inductive_train": file_fds[
            "indexes/inductive-training-offsets.bin"
        ],
        "transductive_train": file_fds[
            "indexes/transductive-training-offsets.bin"
        ],
    }
    index_artifacts = {
        "inductive_train": index_records[
            "indexes/inductive-training-offsets.bin"
        ],
        "transductive_train": index_records[
            "indexes/transductive-training-offsets.bin"
        ],
    }
    counts = {split: 0 for split in TRAINING_SPLITS}
    current_rank = 0
    total = 0
    for offset, line in _iter_descriptor_lines(
        stream_fd,
        "training stream",
    ):
        triple = _parse_training_stream_row(
            line,
            f"training stream row {total + 1}",
        )
        rank = TRAINING_SPLITS.index(triple.training_split)
        if rank < current_rank:
            raise ValueError("training stream split ordering drift")
        current_rank = rank
        counts[triple.training_split] += 1
        if triple.row != counts[triple.training_split]:
            raise ValueError("training stream row ordering drift")
        if line != _training_stream_line(
            triple.training_split,
            triple.row,
            triple.subject,
            int(triple.relation[1:]),
            triple.object,
        ):
            raise ValueError("training stream canonical row drift")
        index_offset = _pread_exact(
            index_fds[triple.training_split],
            _UINT64.size,
            (counts[triple.training_split] - 1) * _UINT64.size,
            "training offset index record",
        )
        if _UINT64.unpack(index_offset)[0] != offset:
            raise ValueError("training offset index does not match stream")
        total += 1
    if total != receipt.training_rows:
        raise ValueError("training stream row count drift")
    for split in TRAINING_SPLITS:
        if counts[split] != index_artifacts[split].count:
            raise ValueError("training index record count drift")


def _verify_distinct_logical_file(
    descriptor: int,
    receipt: WikidataDerivedViewReceipt,
) -> None:
    previous: tuple[int, int, int] | None = None
    count = 0
    for _offset, line in _iter_descriptor_lines(
        descriptor,
        "distinct-edge stream",
    ):
        triple = _parse_training_stream_row(
            line,
            f"distinct-edge stream row {count + 1}",
        )
        edge = (
            triple.subject,
            int(triple.relation[1:]),
            triple.object,
        )
        if previous is not None and edge <= previous:
            raise ValueError("distinct-edge stream ordering drift")
        previous = edge
        if line != _training_stream_line(
            triple.training_split,
            triple.row,
            triple.subject,
            edge[1],
            triple.object,
        ):
            raise ValueError("distinct-edge stream canonical row drift")
        count += 1
    if count != receipt.distinct_edges:
        raise ValueError("distinct-edge stream row count drift")


def _verify_alias_logical_files(
    file_fds: Mapping[str, int],
    receipt: WikidataDerivedViewReceipt,
) -> None:
    stream_fd = file_fds["streams/aliases.tsv"]
    index_fd = file_fds["indexes/aliases.bin"]
    previous_key: int | None = None
    count = 0
    for offset, line in _iter_descriptor_lines(
        stream_fd,
        "alias stream",
    ):
        alias = _parse_alias_stream_row(
            line,
            f"alias stream row {count + 1}",
        )
        prefix = alias.canonical_id[:1].encode("ascii")
        numeric_id = int(alias.canonical_id[1:])
        key = _alias_key(prefix, numeric_id)
        if previous_key is not None and key <= previous_key:
            raise ValueError("alias stream ordering drift")
        previous_key = key
        normalized: set[str] = set()
        for value in alias.aliases:
            key_value = " ".join(
                unicodedata.normalize("NFKC", value).split()
            ).casefold()
            if not key_value or key_value in normalized:
                raise ValueError("alias stream contains duplicate aliases")
            normalized.add(key_value)
        expected_display = (
            alias.aliases[0] if alias.aliases else alias.canonical_id
        )
        if alias.display != expected_display:
            raise ValueError("alias stream display selection drift")
        index_payload = _pread_exact(
            index_fd,
            _ALIAS_INDEX_RECORD.size,
            count * _ALIAS_INDEX_RECORD.size,
            "alias index record",
        )
        indexed_key, indexed_offset, indexed_length = (
            _ALIAS_INDEX_RECORD.unpack(index_payload)
        )
        if (
            indexed_key != key
            or indexed_offset != offset
            or indexed_length != len(line)
        ):
            raise ValueError("alias index does not match stream")
        count += 1
    if count != receipt.alias_rows:
        raise ValueError("alias stream row count drift")


def _verify_logical_files(
    file_fds: Mapping[str, int],
    receipt: WikidataDerivedViewReceipt,
) -> None:
    _verify_training_logical_files(file_fds, receipt)
    _verify_alias_logical_files(file_fds, receipt)
    _verify_distinct_logical_file(
        file_fds["streams/distinct-edges.tsv"],
        receipt,
    )


def _verify_derived_tree(
    verified_source: VerifiedArchiveSet,
    view_root: Path,
    *,
    expected_generator_commit: str,
    require_namespace: bool,
) -> WikidataDerivedView:
    expected_commit = _validate_commit(
        expected_generator_commit,
        "expected generator commit",
    )
    if verified_source.generator_commit != expected_commit:
        raise ValueError(
            "source lock generator commit does not match expectation"
        )
    root_path = Path(view_root)
    root_parent_fd = -1
    root_fd = -1
    directory_fds: dict[str, int] = {}
    file_fds: dict[str, int] = {}
    directory_identities: dict[
        str,
        tuple[int, int, int, int, int, int, int | None, int | None],
    ] = {}
    file_identities: dict[
        str,
        tuple[int, int, int, int, int, int, int | None, int | None],
    ] = {}
    try:
        root_parent_fd, root_name = open_parent_directory(root_path)
        _require_derived_mode(
            os.fstat(root_parent_fd),
            directory=True,
            description="derived-view parent",
        )
        root_fd, root_identity = _open_bound_directory(
            root_parent_fd,
            root_name,
            "derived-view root",
        )
        _require_derived_mode(
            os.fstat(root_fd),
            directory=True,
            description="derived-view root",
        )
        if list_entries(root_fd) != _VIEW_ROOT_ENTRIES:
            raise ValueError("derived-view root inventory drift")

        receipt_fd, receipt_identity = _open_bound_file(
            root_fd,
            "receipt.json",
            "derived-view receipt",
        )
        file_fds["receipt.json"] = receipt_fd
        _require_derived_mode(
            os.fstat(receipt_fd),
            directory=False,
            description="derived-view receipt",
        )
        receipt_payload = _read_descriptor(
            receipt_fd,
            expected_identity=receipt_identity,
            description="derived-view receipt",
        )
        receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
        receipt = WikidataDerivedViewReceipt.from_bytes(
            receipt_payload,
            expected_generator_commit=expected_commit,
        )
        if require_namespace:
            if root_path.parent.name != "wikidata":
                raise ValueError("derived-view namespace parent drift")
            if root_name != receipt_sha256:
                raise ValueError("derived-view content-address namespace drift")
        if receipt.source_lock_sha256 != verified_source.source_lock_sha256:
            raise ValueError("derived-view source-lock binding drift")
        if receipt.archives != verified_source.archives:
            raise ValueError("derived-view archive binding drift")
        if receipt.members != verified_source.members:
            raise ValueError("derived-view member binding drift")
        file_identities["receipt.json"] = receipt_identity

        inventory = {
            "members": tuple(
                PurePosixPath(record.path).name for record in receipt.members
            ),
            "streams": tuple(
                PurePosixPath(record.path).name for record in receipt.streams
            ),
            "indexes": tuple(
                PurePosixPath(record.path).name for record in receipt.indexes
            ),
        }
        for directory_name in ("members", "streams", "indexes"):
            descriptor, identity = _open_bound_directory(
                root_fd,
                directory_name,
                f"derived-view {directory_name} directory",
            )
            directory_fds[directory_name] = descriptor
            directory_identities[directory_name] = identity
            _require_derived_mode(
                os.fstat(descriptor),
                directory=True,
                description=f"derived-view {directory_name} directory",
            )
            expected_names = tuple(sorted(inventory[directory_name]))
            if list_entries(descriptor) != expected_names:
                raise ValueError(
                    f"derived-view {directory_name} inventory drift"
                )

        records = {
            record.path: record
            for record in (
                *receipt.members,
                *receipt.streams,
                *receipt.indexes,
            )
        }
        for relative_path in sorted(records, key=_byte_key):
            record = records[relative_path]
            directory_name, name = relative_path.split("/", 1)
            descriptor, identity = _open_bound_file(
                directory_fds[directory_name],
                name,
                f"derived artifact {relative_path}",
            )
            file_fds[relative_path] = descriptor
            file_identities[relative_path] = identity
            _require_derived_mode(
                os.fstat(descriptor),
                directory=False,
                description=f"derived artifact {relative_path}",
            )
            byte_count, sha256 = _digest_descriptor(
                descriptor,
                expected_identity=identity,
                description=f"derived artifact {relative_path}",
            )
            if byte_count != record.bytes or sha256 != record.sha256:
                raise ValueError(
                    f"derived artifact digest drift: {relative_path}"
                )

        _verify_logical_files(file_fds, receipt)

        for relative_path, descriptor in file_fds.items():
            parent_fd = (
                root_fd
                if relative_path == "receipt.json"
                else directory_fds[relative_path.split("/", 1)[0]]
            )
            name = (
                relative_path
                if relative_path == "receipt.json"
                else relative_path.split("/", 1)[1]
            )
            _check_named_file(
                parent_fd,
                name,
                descriptor,
                file_identities[relative_path],
                f"derived artifact {relative_path}",
            )
        for directory_name, descriptor in directory_fds.items():
            _check_named_directory(
                root_fd,
                directory_name,
                descriptor,
                directory_identities[directory_name],
                f"derived-view {directory_name} directory",
            )
        _check_named_directory(
            root_parent_fd,
            root_name,
            root_fd,
            root_identity,
            "derived-view root",
        )
        if list_entries(root_fd) != _VIEW_ROOT_ENTRIES:
            raise ValueError("derived-view root inventory drift")

        authority = _VerifiedViewAuthority(
            root_identity=root_identity,
            directory_identities=MappingProxyType(
                dict(directory_identities)
            ),
            file_identities=MappingProxyType(dict(file_identities)),
        )
        return _register_verified_view(
            WikidataDerivedView(
                root=root_path,
                receipt_sha256=receipt_sha256,
                receipt=receipt,
            ),
            authority,
        )
    except OSError as error:
        raise ValueError("derived-view authority is missing or unsafe") from error
    finally:
        for descriptor in file_fds.values():
            os.close(descriptor)
        for descriptor in directory_fds.values():
            os.close(descriptor)
        if root_fd >= 0:
            os.close(root_fd)
        if root_parent_fd >= 0:
            os.close(root_parent_fd)


@contextmanager
def _open_authorized_view_files(
    view: WikidataDerivedView,
    relative_paths: tuple[str, ...],
) -> Iterator[Mapping[str, int]]:
    if not isinstance(view, WikidataDerivedView):
        raise ValueError("a verified Wikidata derived view is required")
    authority = _registered_view_authority(view)
    if (
        _validate_sha256(view.receipt_sha256, "view receipt sha256")
        != view.receipt_sha256
        or view.root.name != view.receipt_sha256
        or view.root.parent.name != "wikidata"
    ):
        raise ValueError("verified view namespace identity drift")
    root_parent_fd = -1
    root_fd = -1
    receipt_fd = -1
    directory_fds: dict[str, int] = {}
    selected_fds: dict[str, int] = {}
    try:
        root_parent_fd, root_name = open_parent_directory(view.root)
        _require_derived_mode(
            os.fstat(root_parent_fd),
            directory=True,
            description="verified-view parent",
        )
        root_fd, root_identity = _open_bound_directory(
            root_parent_fd,
            root_name,
            "verified-view root",
        )
        _require_derived_mode(
            os.fstat(root_fd),
            directory=True,
            description="verified-view root",
        )
        if (
            root_identity != authority.root_identity
            or list_entries(root_fd) != _VIEW_ROOT_ENTRIES
        ):
            raise ValueError("verified view root identity drift")

        receipt_fd, receipt_identity = _open_bound_file(
            root_fd,
            "receipt.json",
            "verified-view receipt",
        )
        if receipt_identity != authority.file_identities["receipt.json"]:
            raise ValueError("verified view receipt identity drift")
        payload = _read_descriptor(
            receipt_fd,
            expected_identity=receipt_identity,
            description="verified-view receipt",
        )
        if hashlib.sha256(payload).hexdigest() != view.receipt_sha256:
            raise ValueError("verified view receipt digest drift")
        receipt = WikidataDerivedViewReceipt.from_bytes(
            payload,
            expected_generator_commit=view.receipt.generator_commit,
        )
        if receipt != view.receipt:
            raise ValueError("verified view receipt object drift")

        expected_names = {
            "members": tuple(
                PurePosixPath(record.path).name
                for record in receipt.members
            ),
            "streams": tuple(
                PurePosixPath(record.path).name
                for record in receipt.streams
            ),
            "indexes": tuple(
                PurePosixPath(record.path).name
                for record in receipt.indexes
            ),
        }
        for directory_name in ("members", "streams", "indexes"):
            descriptor, identity = _open_bound_directory(
                root_fd,
                directory_name,
                f"verified-view {directory_name} directory",
            )
            if (
                identity
                != authority.directory_identities[directory_name]
                or list_entries(descriptor)
                != tuple(sorted(expected_names[directory_name]))
            ):
                os.close(descriptor)
                raise ValueError(
                    f"verified view {directory_name} identity drift"
                )
            directory_fds[directory_name] = descriptor

        records = {
            record.path: record
            for record in (
                *receipt.members,
                *receipt.streams,
                *receipt.indexes,
            )
        }
        requested = set(relative_paths)
        if len(requested) != len(relative_paths) or not requested <= set(records):
            raise ValueError("verified view file request is invalid")
        for relative_path, record in records.items():
            directory_name, name = relative_path.split("/", 1)
            named = entry_lstat(directory_fds[directory_name], name)
            _require_derived_mode(
                named,
                directory=False,
                description=f"verified-view artifact {relative_path}",
            )
            if (
                relative_path not in authority.file_identities
                or _file_identity(named)
                != authority.file_identities[relative_path]
                or named.st_size != record.bytes
            ):
                raise ValueError(
                    f"verified view artifact identity drift: {relative_path}"
                )
            if relative_path not in requested:
                continue
            descriptor, identity = _open_bound_file(
                directory_fds[directory_name],
                name,
                f"verified-view artifact {relative_path}",
            )
            if (
                identity != authority.file_identities[relative_path]
                or os.fstat(descriptor).st_size != record.bytes
            ):
                os.close(descriptor)
                raise ValueError(
                    f"verified view artifact identity drift: {relative_path}"
                )
            selected_fds[relative_path] = descriptor
        yield MappingProxyType(selected_fds)

        _view_authority_hook("before_postcheck", view)
        for directory_name, descriptor in directory_fds.items():
            _check_named_directory(
                root_fd,
                directory_name,
                descriptor,
                authority.directory_identities[directory_name],
                f"verified-view {directory_name} directory",
            )
            if list_entries(descriptor) != tuple(
                sorted(expected_names[directory_name])
            ):
                raise ValueError(
                    f"verified view {directory_name} identity drift"
                )
        for relative_path, descriptor in selected_fds.items():
            directory_name, name = relative_path.split("/", 1)
            _check_named_file(
                directory_fds[directory_name],
                name,
                descriptor,
                authority.file_identities[relative_path],
                f"verified-view artifact {relative_path}",
            )
        _check_named_file(
            root_fd,
            "receipt.json",
            receipt_fd,
            authority.file_identities["receipt.json"],
            "verified-view receipt",
        )
        _check_named_directory(
            root_parent_fd,
            root_name,
            root_fd,
            authority.root_identity,
            "verified-view root",
        )
    except OSError as error:
        raise ValueError("verified view authority is missing or unsafe") from error
    finally:
        for descriptor in selected_fds.values():
            os.close(descriptor)
        for descriptor in directory_fds.values():
            os.close(descriptor)
        if receipt_fd >= 0:
            os.close(receipt_fd)
        if root_fd >= 0:
            os.close(root_fd)
        if root_parent_fd >= 0:
            os.close(root_parent_fd)


def verify_wikidata_derived_view(
    source_lock_path: Path,
    source_root: Path,
    view_root: Path,
    *,
    expected_generator_commit: str,
) -> WikidataDerivedView:
    with _open_verified_archives(
        Path(source_lock_path),
        Path(source_root),
        expected_generator_commit=expected_generator_commit,
    ) as verified:
        return _verify_derived_tree(
            verified,
            Path(view_root),
            expected_generator_commit=expected_generator_commit,
            require_namespace=True,
        )


def iter_v2_training_triples(
    view: WikidataDerivedView,
) -> Iterator[V2TrainingTriple]:
    with _open_authorized_view_files(
        view,
        ("streams/training.tsv",),
    ) as files:
        counts = {split: 0 for split in TRAINING_SPLITS}
        current_rank = 0
        total = 0
        for _offset, line in _iter_descriptor_lines(
            files["streams/training.tsv"],
            "training stream",
        ):
            triple = _parse_training_stream_row(
                line,
                f"training stream row {total + 1}",
            )
            rank = TRAINING_SPLITS.index(triple.training_split)
            counts[triple.training_split] += 1
            if (
                rank < current_rank
                or triple.row != counts[triple.training_split]
            ):
                raise ValueError("training stream ordering drift")
            current_rank = rank
            total += 1
            yield triple
        if total != view.receipt.training_rows:
            raise ValueError("training stream row count drift")


def iter_v2_aliases(
    view: WikidataDerivedView,
) -> Iterator[V2AliasRecord]:
    with _open_authorized_view_files(
        view,
        ("streams/aliases.tsv",),
    ) as files:
        previous_key: int | None = None
        count = 0
        for _offset, line in _iter_descriptor_lines(
            files["streams/aliases.tsv"],
            "alias stream",
        ):
            alias = _parse_alias_stream_row(
                line,
                f"alias stream row {count + 1}",
            )
            key = _alias_key(
                alias.canonical_id[:1].encode("ascii"),
                int(alias.canonical_id[1:]),
            )
            if previous_key is not None and key <= previous_key:
                raise ValueError("alias stream ordering drift")
            previous_key = key
            count += 1
            yield alias
        if count != view.receipt.alias_rows:
            raise ValueError("alias stream row count drift")


def iter_distinct_training_edges(
    view: WikidataDerivedView,
) -> Iterator[V2TrainingTriple]:
    with _open_authorized_view_files(
        view,
        ("streams/distinct-edges.tsv",),
    ) as files:
        previous: tuple[int, int, int] | None = None
        count = 0
        for _offset, line in _iter_descriptor_lines(
            files["streams/distinct-edges.tsv"],
            "distinct-edge stream",
        ):
            triple = _parse_training_stream_row(
                line,
                f"distinct-edge stream row {count + 1}",
            )
            edge = (
                triple.subject,
                int(triple.relation[1:]),
                triple.object,
            )
            if previous is not None and edge <= previous:
                raise ValueError("distinct-edge stream ordering drift")
            previous = edge
            count += 1
            yield triple
        if count != view.receipt.distinct_edges:
            raise ValueError("distinct-edge stream row count drift")


def lookup_training_triple(
    view: WikidataDerivedView,
    training_split: str,
    row: int,
) -> V2TrainingTriple:
    if training_split not in TRAINING_SPLITS:
        raise ValueError("training split is not in the frozen contract")
    if type(row) is not int or row <= 0:
        raise ValueError("training row must be a positive integer")
    index_paths = {
        "inductive_train": "indexes/inductive-training-offsets.bin",
        "transductive_train": "indexes/transductive-training-offsets.bin",
    }
    paths = (
        "streams/training.tsv",
        index_paths["inductive_train"],
        index_paths["transductive_train"],
    )
    with _open_authorized_view_files(view, paths) as files:
        index_by_path = {
            record.path: record for record in view.receipt.indexes
        }
        selected_path = index_paths[training_split]
        selected_index = index_by_path[selected_path]
        if row > selected_index.count:
            raise IndexError("training row is outside the indexed split")
        index_fd = files[selected_path]
        offset = _UINT64.unpack(
            _pread_exact(
                index_fd,
                _UINT64.size,
                (row - 1) * _UINT64.size,
                "training offset index record",
            )
        )[0]
        if row < selected_index.count:
            next_offset = _UINT64.unpack(
                _pread_exact(
                    index_fd,
                    _UINT64.size,
                    row * _UINT64.size,
                    "next training offset index record",
                )
            )[0]
        elif training_split == "inductive_train":
            transductive = index_by_path[index_paths["transductive_train"]]
            if transductive.count:
                next_offset = _UINT64.unpack(
                    _pread_exact(
                        files[index_paths["transductive_train"]],
                        _UINT64.size,
                        0,
                        "first transductive offset index record",
                    )
                )[0]
            else:
                next_offset = os.fstat(
                    files["streams/training.tsv"]
                ).st_size
        else:
            next_offset = os.fstat(files["streams/training.tsv"]).st_size
        stream_size = os.fstat(files["streams/training.tsv"]).st_size
        if (
            offset >= next_offset
            or next_offset > stream_size
            or next_offset - offset > _SOURCE_LINE_LIMIT
        ):
            raise ValueError("training offset index bounds drift")
        line = _pread_exact(
            files["streams/training.tsv"],
            next_offset - offset,
            offset,
            "indexed training row",
        )
        triple = _parse_training_stream_row(line, "indexed training row")
        if triple.training_split != training_split or triple.row != row:
            raise ValueError("indexed training key does not match stream row")
        return triple


def lookup_alias(
    view: WikidataDerivedView,
    canonical_id: str,
) -> V2AliasRecord | None:
    if type(canonical_id) is not str:
        raise TypeError("alias canonical ID must be a string")
    try:
        encoded = canonical_id.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("alias canonical ID is not canonical") from error
    if encoded[:1] not in {b"Q", b"P"}:
        raise ValueError("alias canonical ID is not canonical")
    number = _numeric_identifier(
        encoded,
        encoded[:1],
        "alias canonical ID",
        maximum=_ALIAS_NUMERIC_MAX,
    )
    target = _alias_key(encoded[:1], number)
    with _open_authorized_view_files(
        view,
        ("streams/aliases.tsv", "indexes/aliases.bin"),
    ) as files:
        index = next(
            record
            for record in view.receipt.indexes
            if record.path == "indexes/aliases.bin"
        )
        low = 0
        high = index.count
        index_fd = files["indexes/aliases.bin"]
        while low < high:
            middle = (low + high) // 2
            payload = _pread_exact(
                index_fd,
                _ALIAS_INDEX_RECORD.size,
                middle * _ALIAS_INDEX_RECORD.size,
                "alias index record",
            )
            key, offset, length = _ALIAS_INDEX_RECORD.unpack(payload)
            if key < target:
                low = middle + 1
            elif key > target:
                high = middle
            else:
                stream_size = os.fstat(
                    files["streams/aliases.tsv"]
                ).st_size
                if (
                    length <= 0
                    or length > _SOURCE_LINE_LIMIT
                    or offset > stream_size
                    or length > stream_size - offset
                ):
                    raise ValueError("alias index bounds drift")
                line = _pread_exact(
                    files["streams/aliases.tsv"],
                    length,
                    offset,
                    "indexed alias row",
                )
                alias = _parse_alias_stream_row(line, "indexed alias row")
                if alias.canonical_id != canonical_id:
                    raise ValueError(
                        "indexed alias key does not match stream row"
                    )
                return alias
        return None


def _create_private_directory(
    parent_fd: int,
    name: str,
    *,
    authority: _PrivateBuildAuthority | None = None,
    relative_path: str | None = None,
) -> tuple[int, _CreationIdentity]:
    os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    fsync_directory(parent_fd)
    descriptor = -1
    try:
        descriptor, created = open_directory_at(parent_fd, name)
        if created:
            raise ValueError("private directory creation identity drift")
        identity = _creation_identity(os.fstat(descriptor))
        if authority is not None:
            if relative_path is None:
                raise ValueError("private directory authority path is missing")
            authority.register_directory(
                relative_path,
                descriptor,
                identity,
            )
        _check_named_derived_directory(
            parent_fd,
            name,
            descriptor,
            identity,
            f"private derived directory {name}",
        )
        return descriptor, identity
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _allocate_build_directory(
    wikidata_fd: int,
    source_lock_sha256: str,
    generator_commit: str,
) -> tuple[str, int, _CreationIdentity]:
    prefix = (
        f".build-{source_lock_sha256[:16]}-{generator_commit[:16]}-"
    )
    for _attempt in range(32):
        name = prefix + secrets.token_hex(8)
        try:
            descriptor, identity = _create_private_directory(
                wikidata_fd,
                name,
            )
        except FileExistsError:
            continue
        return name, descriptor, identity
    raise FileExistsError("could not allocate a private derived-view sibling")


def _private_file_parent(
    authority: _PrivateBuildAuthority,
    relative_path: str,
) -> tuple[int, str]:
    parts = relative_path.split("/")
    if len(parts) == 1:
        return authority.descriptor, parts[0]
    if len(parts) == 2 and parts[0] in authority.directory_descriptors:
        return authority.directory_descriptors[parts[0]], parts[1]
    raise ValueError("private file path is outside build authority")


def _expected_private_inventory(
    authority: _PrivateBuildAuthority,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    paths = {
        *authority.pending_file_identities,
        *authority.file_authorities,
    }
    child_entries: dict[str, list[str]] = {
        name: [] for name in authority.directory_descriptors
    }
    root_files: list[str] = []
    for relative_path in paths:
        parts = relative_path.split("/")
        if len(parts) == 1:
            root_files.append(parts[0])
        elif len(parts) == 2 and parts[0] in child_entries:
            child_entries[parts[0]].append(parts[1])
        else:
            raise ValueError("private file inventory is outside authority")
    root_entries = tuple(
        sorted(
            (*authority.directory_descriptors, *root_files),
            key=_byte_key,
        )
    )
    return root_entries, {
        name: tuple(sorted(entries, key=_byte_key))
        for name, entries in child_entries.items()
    }


def _capture_private_directory_identity(
    descriptor: int,
    description: str,
) -> _PrivateFileIdentity:
    metadata = os.fstat(descriptor)
    _require_derived_mode(
        metadata,
        directory=True,
        description=description,
    )
    return _private_file_identity(metadata)


def _verify_private_build_files(
    authority: _PrivateBuildAuthority,
) -> None:
    if authority.pending_file_identities:
        raise ValueError("private candidate has unfinalized files")
    for relative_path in sorted(authority.file_authorities, key=_byte_key):
        parent_fd, name = _private_file_parent(authority, relative_path)
        file_authority = authority.file_authorities[relative_path]
        descriptor = -1
        body_error: BaseException | None = None
        try:
            descriptor, metadata = open_regular_file_at(parent_fd, name)
            _require_derived_mode(
                metadata,
                directory=False,
                description=f"private candidate file {relative_path}",
            )
            _verify_open_private_file(
                parent_fd,
                name,
                descriptor,
                file_authority,
                f"private candidate file {relative_path}",
                rewind=False,
            )
        except BaseException as error:
            body_error = error
            raise
        finally:
            close_error = _close_descriptors_exhaustively((descriptor,))
            if close_error is not None:
                if body_error is None:
                    raise close_error
                body_error.add_note(
                    f"private candidate file close also failed: "
                    f"{close_error!r}"
                )


def _seal_private_build_authority(
    authority: _PrivateBuildAuthority,
) -> None:
    if (
        authority.sealed_root_identity is not None
        or authority.sealed_directory_identities
    ):
        raise ValueError("private build authority is already sealed")
    _check_named_derived_directory(
        authority.namespace_fd,
        authority.name,
        authority.descriptor,
        authority.identity,
        "private candidate root while sealing",
    )
    root_entries, child_entries = _expected_private_inventory(authority)
    if list_entries(authority.descriptor) != root_entries:
        raise ValueError("private candidate root inventory drift")
    before_root = _capture_private_directory_identity(
        authority.descriptor,
        "private candidate root",
    )
    before_children: dict[str, _PrivateFileIdentity] = {}
    for directory_name in sorted(
        authority.directory_descriptors,
        key=_byte_key,
    ):
        descriptor = authority.directory_descriptors[directory_name]
        _check_named_derived_directory(
            authority.descriptor,
            directory_name,
            descriptor,
            authority.directory_identities[directory_name],
            f"private candidate child {directory_name}",
        )
        if list_entries(descriptor) != child_entries[directory_name]:
            raise ValueError(
                f"private candidate {directory_name} inventory drift"
            )
        before_children[directory_name] = (
            _capture_private_directory_identity(
                descriptor,
                f"private candidate child {directory_name}",
            )
        )
    _verify_private_build_files(authority)
    after_root = _capture_private_directory_identity(
        authority.descriptor,
        "private candidate root",
    )
    if after_root != before_root or list_entries(authority.descriptor) != root_entries:
        raise ValueError("private candidate root identity drift")
    for directory_name, expected_identity in before_children.items():
        descriptor = authority.directory_descriptors[directory_name]
        if (
            _capture_private_directory_identity(
                descriptor,
                f"private candidate child {directory_name}",
            )
            != expected_identity
            or list_entries(descriptor) != child_entries[directory_name]
        ):
            raise ValueError(
                f"private candidate {directory_name} identity drift"
            )
    authority.sealed_root_identity = before_root
    authority.sealed_directory_identities = before_children


def _verify_sealed_private_build(
    authority: _PrivateBuildAuthority,
    *,
    root_name: str,
    expected_root_identity: _PrivateFileIdentity,
) -> None:
    if (
        authority.sealed_root_identity is None
        or set(authority.sealed_directory_identities)
        != set(authority.directory_descriptors)
    ):
        raise ValueError("private build authority is not sealed")
    _check_named_derived_directory(
        authority.namespace_fd,
        root_name,
        authority.descriptor,
        authority.identity,
        "sealed private candidate root",
    )
    root_entries, child_entries = _expected_private_inventory(authority)
    if (
        _capture_private_directory_identity(
            authority.descriptor,
            "sealed private candidate root",
        )
        != expected_root_identity
        or list_entries(authority.descriptor) != root_entries
    ):
        raise ValueError("sealed private candidate root identity drift")
    for directory_name, expected_identity in (
        authority.sealed_directory_identities.items()
    ):
        descriptor = authority.directory_descriptors[directory_name]
        _check_named_derived_directory(
            authority.descriptor,
            directory_name,
            descriptor,
            authority.directory_identities[directory_name],
            f"sealed private candidate child {directory_name}",
        )
        if (
            _capture_private_directory_identity(
                descriptor,
                f"sealed private candidate child {directory_name}",
            )
            != expected_identity
            or list_entries(descriptor) != child_entries[directory_name]
        ):
            raise ValueError(
                f"sealed private candidate {directory_name} identity drift"
            )
    _verify_private_build_files(authority)
    if (
        _capture_private_directory_identity(
            authority.descriptor,
            "sealed private candidate root",
        )
        != expected_root_identity
        or list_entries(authority.descriptor) != root_entries
    ):
        raise ValueError("sealed private candidate root identity drift")
    for directory_name, expected_identity in (
        authority.sealed_directory_identities.items()
    ):
        descriptor = authority.directory_descriptors[directory_name]
        if (
            _capture_private_directory_identity(
                descriptor,
                f"sealed private candidate child {directory_name}",
            )
            != expected_identity
            or list_entries(descriptor) != child_entries[directory_name]
        ):
            raise ValueError(
                f"sealed private candidate {directory_name} identity drift"
            )


@dataclass
class _QuarantineMarker:
    namespace_fd: int
    name: str
    descriptor: int
    identity: _CreationIdentity
    entry_name: str | None


def _allocate_quarantine_marker(
    namespace_fd: int,
    published_name: str,
) -> _QuarantineMarker:
    prefix = f".quarantine-{published_name[:16]}-"
    for _attempt in range(32):
        name = prefix + secrets.token_hex(8)
        try:
            descriptor, identity = _create_private_directory(
                namespace_fd,
                name,
            )
        except FileExistsError:
            continue
        if list_entries(descriptor):
            os.close(descriptor)
            raise ValueError("quarantine marker is not empty")
        return _QuarantineMarker(
            namespace_fd=namespace_fd,
            name=name,
            descriptor=descriptor,
            identity=identity,
            entry_name=name,
        )
    raise FileExistsError("could not allocate quarantine marker")


def _remove_quarantine_marker(
    marker: _QuarantineMarker,
    *,
    sync_parent: bool,
) -> None:
    if marker.entry_name is None:
        return
    name = marker.entry_name
    _check_named_derived_directory(
        marker.namespace_fd,
        name,
        marker.descriptor,
        marker.identity,
        "quarantine marker before removal",
    )
    if list_entries(marker.descriptor):
        raise ValueError("quarantine marker is not empty")
    os.rmdir(name, dir_fd=marker.namespace_fd)
    marker.entry_name = None
    if sync_parent:
        fsync_directory(marker.namespace_fd)


def _rollback_quarantine_exchange(
    authority: _PrivateBuildAuthority,
    published_name: str,
    marker: _QuarantineMarker,
) -> None:
    wrong_source_fd = -1
    primary_error: BaseException | None = None
    try:
        wrong_source_before = entry_lstat(
            authority.namespace_fd,
            marker.name,
        )
        _require_derived_mode(
            wrong_source_before,
            directory=True,
            description="exchanged non-candidate source",
        )
        wrong_source_fd, _created = open_directory_at(
            authority.namespace_fd,
            marker.name,
        )
        wrong_source_opened = os.fstat(wrong_source_fd)
        _require_derived_mode(
            wrong_source_opened,
            directory=True,
            description="exchanged non-candidate source",
        )
        wrong_source_identity = _creation_identity(wrong_source_opened)
        if (
            _creation_identity(wrong_source_before)
            != wrong_source_identity
        ):
            raise ValueError(
                "exchanged non-candidate source identity drift"
            )
        _check_named_derived_directory(
            authority.namespace_fd,
            marker.name,
            wrong_source_fd,
            wrong_source_identity,
            "exchanged non-candidate source before rollback",
        )
        _atomic_exchange_directories(
            authority.namespace_fd,
            published_name,
            marker.name,
        )
        marker.entry_name = marker.name
        _check_named_derived_directory(
            marker.namespace_fd,
            marker.name,
            marker.descriptor,
            marker.identity,
            "quarantine marker after exchange rollback",
        )
        _check_named_derived_directory(
            authority.namespace_fd,
            published_name,
            wrong_source_fd,
            wrong_source_identity,
            "restored concurrent winner",
        )
        fsync_directory(authority.namespace_fd)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        close_error = _close_descriptors_exhaustively((wrong_source_fd,))
        if close_error is not None:
            if primary_error is None:
                raise close_error
            primary_error.add_note(
                f"quarantine rollback descriptor close also failed: "
                f"{close_error!r}"
            )


def _named_derived_directory_matches(
    parent_fd: int,
    name: str,
    descriptor: int,
    identity: _CreationIdentity,
    description: str,
) -> bool:
    try:
        _check_named_derived_directory(
            parent_fd,
            name,
            descriptor,
            identity,
            description,
        )
    except (OSError, ValueError):
        return False
    return True


def _exchange_quarantine_published_candidate(
    authority: _PrivateBuildAuthority,
    published_name: str,
    marker: _QuarantineMarker,
) -> str:
    _check_named_derived_directory(
        authority.namespace_fd,
        published_name,
        authority.descriptor,
        authority.identity,
        "published candidate before quarantine exchange",
    )
    _check_named_derived_directory(
        marker.namespace_fd,
        marker.name,
        marker.descriptor,
        marker.identity,
        "quarantine marker before exchange",
    )
    _derived_view_build_hook(
        "before_quarantine_exchange",
        authority,
        published_name,
    )
    _atomic_exchange_directories(
        authority.namespace_fd,
        published_name,
        marker.name,
    )
    candidate_moved = _named_derived_directory_matches(
        authority.namespace_fd,
        marker.name,
        authority.descriptor,
        authority.identity,
        "published candidate after quarantine exchange",
    )
    marker_at_final = _named_derived_directory_matches(
        authority.namespace_fd,
        published_name,
        marker.descriptor,
        marker.identity,
        "quarantine marker at final name",
    )
    marker.entry_name = published_name if marker_at_final else None

    if candidate_moved:
        if marker_at_final:
            _remove_quarantine_marker(marker, sync_parent=True)
        else:
            fsync_directory(authority.namespace_fd)
        return marker.name

    if marker_at_final:
        try:
            _rollback_quarantine_exchange(
                authority,
                published_name,
                marker,
            )
        except BaseException as rollback_error:
            rollback_failure = ValueError(
                "quarantine exchange rollback failed"
            )
            rollback_failure.add_note(
                f"conditional rollback also failed: {rollback_error!r}"
            )
            raise rollback_failure from rollback_error
        raise ValueError(
            "published candidate did not move during quarantine exchange"
        )

    fsync_directory(authority.namespace_fd)
    raise ValueError("quarantine exchange identities are indeterminate")


def _cleanup_private_build(authority: _PrivateBuildAuthority) -> None:
    _check_named_derived_directory(
        authority.namespace_fd,
        authority.name,
        authority.descriptor,
        authority.identity,
        "private build directory before cleanup",
    )
    for directory_name in sorted(authority.directory_identities, key=_byte_key):
        _check_named_derived_directory(
            authority.descriptor,
            directory_name,
            authority.directory_descriptors[directory_name],
            authority.directory_identities[directory_name],
            f"private build child {directory_name} before cleanup",
        )

    for relative_path in sorted(
        tuple(authority.file_authorities),
        key=_byte_key,
    ):
        parent_fd, name = _private_file_parent(authority, relative_path)
        file_authority = authority.file_authorities[relative_path]
        descriptor = -1
        body_error: BaseException | None = None
        try:
            descriptor, metadata = open_regular_file_at(parent_fd, name)
            _require_derived_mode(
                metadata,
                directory=False,
                description=f"private cleanup file {relative_path}",
            )
            _verify_open_private_file(
                parent_fd,
                name,
                descriptor,
                file_authority,
                f"private cleanup file {relative_path}",
                rewind=False,
            )
            os.unlink(name, dir_fd=parent_fd)
            authority.forget_file(relative_path, file_authority)
        except BaseException as error:
            body_error = error
            raise
        finally:
            close_error = _close_descriptors_exhaustively((descriptor,))
            if close_error is not None:
                if body_error is None:
                    raise close_error
                body_error.add_note(
                    f"private cleanup file close also failed: "
                    f"{close_error!r}"
                )

    for relative_path in sorted(
        tuple(authority.pending_file_identities),
        key=_byte_key,
    ):
        parent_fd, name = _private_file_parent(authority, relative_path)
        expected_identity = authority.pending_file_identities[relative_path]
        descriptor = -1
        body_error = None
        try:
            descriptor, metadata = open_regular_file_at(parent_fd, name)
            _require_derived_mode(
                metadata,
                directory=False,
                description=f"pending private cleanup file {relative_path}",
            )
            _check_named_created_file(
                parent_fd,
                name,
                descriptor,
                expected_identity,
                f"pending private cleanup file {relative_path}",
            )
            os.unlink(name, dir_fd=parent_fd)
            del authority.pending_file_identities[relative_path]
        except BaseException as error:
            body_error = error
            raise
        finally:
            close_error = _close_descriptors_exhaustively((descriptor,))
            if close_error is not None:
                if body_error is None:
                    raise close_error
                body_error.add_note(
                    f"pending cleanup close also failed: {close_error!r}"
                )

    for directory_name in sorted(
        tuple(authority.directory_identities),
        key=_byte_key,
        reverse=True,
    ):
        descriptor = authority.directory_descriptors[directory_name]
        identity = authority.directory_identities[directory_name]
        if list_entries(descriptor):
            raise ValueError(
                f"private cleanup directory is not empty: {directory_name}"
            )
        _check_named_derived_directory(
            authority.descriptor,
            directory_name,
            descriptor,
            identity,
            f"private build child {directory_name} before removal",
        )
        os.rmdir(directory_name, dir_fd=authority.descriptor)
        authority.forget_directory(directory_name, identity)

    if list_entries(authority.descriptor):
        raise ValueError("private build cleanup inventory drift")
    _check_named_derived_directory(
        authority.namespace_fd,
        authority.name,
        authority.descriptor,
        authority.identity,
        "private build directory before removal",
    )
    fsync_directory(authority.descriptor)
    os.rmdir(authority.name, dir_fd=authority.namespace_fd)
    fsync_directory(authority.namespace_fd)


def build_wikidata_derived_view(
    source_lock_path: Path,
    source_root: Path,
    output_root: Path,
    *,
    expected_generator_commit: str,
) -> WikidataDerivedView:
    generator_commit = _validate_commit(
        expected_generator_commit,
        "expected generator commit",
    )
    output_fd = -1
    wikidata_fd = -1
    build_fd = -1
    members_fd = -1
    streams_fd = -1
    indexes_fd = -1
    work_fd = -1
    authority: _PrivateBuildAuthority | None = None
    quarantine_marker: _QuarantineMarker | None = None
    candidate_private = False
    published = False
    primary_error: BaseException | None = None
    output_path = Path(output_root)
    try:
        _precheck_pinned_source_lock(
            Path(source_lock_path),
            expected_generator_commit=generator_commit,
        )
        output_fd = open_directory_path(
            output_path,
            create=True,
            mode=0o700,
        )
        _require_owned_mode(
            os.fstat(output_fd),
            directory=True,
            description="derived-view output root",
        )
        wikidata_fd, _created = open_directory_at(
            output_fd,
            "wikidata",
            create=True,
            mode=0o700,
        )
        _require_derived_mode(
            os.fstat(wikidata_fd),
            directory=True,
            description="Wikidata derived-view namespace",
        )

        with _open_verified_archives(
            Path(source_lock_path),
            Path(source_root),
            expected_generator_commit=generator_commit,
        ) as verified:
            if verified.generator_commit != generator_commit:
                raise ValueError(
                    "source lock generator commit does not match expectation"
                )

            build_name, build_fd, build_identity = _allocate_build_directory(
                wikidata_fd,
                verified.source_lock_sha256,
                generator_commit,
            )
            authority = _PrivateBuildAuthority(
                namespace_fd=wikidata_fd,
                name=build_name,
                descriptor=build_fd,
                identity=build_identity,
            )
            candidate_private = True
            members_fd, _members_identity = _create_private_directory(
                build_fd,
                "members",
                authority=authority,
                relative_path="members",
            )
            streams_fd, _streams_identity = _create_private_directory(
                build_fd,
                "streams",
                authority=authority,
                relative_path="streams",
            )
            indexes_fd, _indexes_identity = _create_private_directory(
                build_fd,
                "indexes",
                authority=authority,
                relative_path="indexes",
            )
            work_fd, work_identity = _create_private_directory(
                build_fd,
                ".work",
                authority=authority,
                relative_path=".work",
            )

            members = _materialize_member_files(
                verified,
                members_fd,
                authority,
            )
            _derived_view_build_hook(
                "after_members",
                authority,
                None,
            )
            (
                training_stream,
                distinct_stream,
                training_indexes,
                training_rows,
                distinct_edges,
            ) = _build_training_artifacts(
                members_fd,
                streams_fd,
                indexes_fd,
                work_fd,
                authority,
            )
            alias_stream, alias_index, alias_rows = (
                _build_alias_artifacts(
                    members_fd,
                    streams_fd,
                    indexes_fd,
                    work_fd,
                    authority,
                )
            )
            if list_entries(work_fd):
                raise ValueError("external-sort temporary inventory drift")
            _check_named_derived_directory(
                build_fd,
                ".work",
                work_fd,
                work_identity,
                "external-sort work directory before removal",
            )
            os.rmdir(".work", dir_fd=build_fd)
            authority.forget_directory(".work", work_identity)
            os.close(work_fd)
            work_fd = -1
            fsync_directory(build_fd)

            receipt = WikidataDerivedViewReceipt(
                format=RECEIPT_FORMAT,
                schema_version=RECEIPT_SCHEMA_VERSION,
                source_lock_sha256=verified.source_lock_sha256,
                generator_commit=generator_commit,
                archives=verified.archives,
                members=members,
                streams=tuple(
                    sorted(
                        (
                            training_stream,
                            alias_stream,
                            distinct_stream,
                        ),
                        key=lambda record: _byte_key(record.path),
                    )
                ),
                indexes=tuple(
                    sorted(
                        (
                            alias_index,
                            *training_indexes,
                        ),
                        key=lambda record: _byte_key(record.path),
                    )
                ),
                training_rows=training_rows,
                alias_rows=alias_rows,
                distinct_edges=distinct_edges,
                overlap_audit_passed=True,
            )
            receipt_payload = receipt.to_bytes()
            receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
            receipt_writer = _ArtifactWriter(
                build_fd,
                "receipt.json",
                "receipt.json",
                authority=authority,
            )
            try:
                receipt_writer.write(receipt_payload)
                receipt_writer.finish()
            except BaseException:
                receipt_writer.abort()
                raise

            fsync_directory(members_fd)
            fsync_directory(streams_fd)
            fsync_directory(indexes_fd)
            fsync_directory(build_fd)
            _derived_view_build_hook(
                "before_candidate_verify",
                authority,
                receipt_sha256,
            )
            _check_named_derived_directory(
                wikidata_fd,
                build_name,
                build_fd,
                build_identity,
                "private build directory before candidate verification",
            )
            _verify_derived_tree(
                verified,
                output_path / "wikidata" / build_name,
                expected_generator_commit=generator_commit,
                require_namespace=False,
            )
            _check_named_derived_directory(
                wikidata_fd,
                build_name,
                build_fd,
                build_identity,
                "private build directory after candidate verification",
            )
            _seal_private_build_authority(authority)

        _derived_view_build_hook(
            "before_publish_check",
            authority,
            receipt_sha256,
        )
        quarantine_marker = _allocate_quarantine_marker(
            wikidata_fd,
            receipt_sha256,
        )
        _check_named_derived_directory(
            wikidata_fd,
            quarantine_marker.name,
            quarantine_marker.descriptor,
            quarantine_marker.identity,
            "quarantine marker before final candidate verification",
        )
        _verify_sealed_private_build(
            authority,
            root_name=build_name,
            expected_root_identity=cast(
                _PrivateFileIdentity,
                authority.sealed_root_identity,
            ),
        )
        try:
            atomic_rename_noreplace(
                wikidata_fd,
                build_name,
                wikidata_fd,
                receipt_sha256,
            )
        except FileExistsError:
            winner = _verify_derived_tree(
                verified,
                output_path / "wikidata" / receipt_sha256,
                expected_generator_commit=generator_commit,
                require_namespace=True,
            )
            return winner
        candidate_private = False
        try:
            _derived_view_build_hook(
                "after_publish_rename",
                authority,
                receipt_sha256,
            )
            fsync_directory(wikidata_fd)
            _check_named_derived_directory(
                wikidata_fd,
                receipt_sha256,
                build_fd,
                build_identity,
                "published derived-view target",
            )
            postrename_root_identity = (
                _capture_private_directory_identity(
                    build_fd,
                    "published derived-view target",
                )
            )
            _derived_view_build_hook(
                "before_postpublish_verify",
                authority,
                receipt_sha256,
            )
            _verify_sealed_private_build(
                authority,
                root_name=receipt_sha256,
                expected_root_identity=postrename_root_identity,
            )
            winner = _verify_derived_tree(
                verified,
                output_path / "wikidata" / receipt_sha256,
                expected_generator_commit=generator_commit,
                require_namespace=True,
            )
            _verify_sealed_private_build(
                authority,
                root_name=receipt_sha256,
                expected_root_identity=postrename_root_identity,
            )
            _remove_quarantine_marker(
                quarantine_marker,
                sync_parent=False,
            )
            published = True
            return winner
        except BaseException as error:
            try:
                _exchange_quarantine_published_candidate(
                    authority,
                    receipt_sha256,
                    quarantine_marker,
                )
            except BaseException as quarantine_error:
                error.add_note(
                    f"published candidate quarantine also failed: "
                    f"{quarantine_error!r}"
                )
            raise
    except OSError as error:
        wrapped = ValueError(
            "Wikidata derived-view publication is missing or unsafe"
        )
        primary_error = wrapped
        raise wrapped from error
    except BaseException as error:
        primary_error = error
        raise
    finally:
        secondary_error: BaseException | None = None
        if authority is not None and candidate_private and not published:
            try:
                _derived_view_build_hook(
                    "before_cleanup",
                    authority,
                    None,
                )
                _cleanup_private_build(authority)
            except BaseException as error:
                secondary_error = error
        if (
            quarantine_marker is not None
            and quarantine_marker.entry_name is not None
        ):
            try:
                _remove_quarantine_marker(
                    quarantine_marker,
                    sync_parent=True,
                )
            except BaseException as error:
                secondary_error = _append_secondary_error(
                    secondary_error,
                    error,
                    "quarantine marker cleanup failure",
                )
        close_error = _close_descriptors_exhaustively(
            (
                work_fd,
                indexes_fd,
                streams_fd,
                members_fd,
                build_fd,
                (
                    quarantine_marker.descriptor
                    if quarantine_marker is not None
                    else -1
                ),
                wikidata_fd,
                output_fd,
            )
        )
        if close_error is not None:
            secondary_error = _append_secondary_error(
                secondary_error,
                close_error,
                "outer build descriptor close failure",
            )
        if secondary_error is not None:
            if primary_error is None:
                raise secondary_error
            primary_error.add_note(
                f"private cleanup also failed: {secondary_error!r}"
            )
