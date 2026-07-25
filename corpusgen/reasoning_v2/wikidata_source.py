"""Closed receipts and descriptor-pinned Wikidata archive authority."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import stat
import struct
import sys
import tarfile
import unicodedata
import zlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, BinaryIO, cast

from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.parallel.safeio import (
    atomic_rename_noreplace,
    entry_exists,
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
from corpusgen.wikidata5m import (
    canonicalize_aliases,
    normalize_alias,
    parse_pid,
    parse_qid,
)


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
class IndexArtifactRecord:
    path: str
    bytes: int
    sha256: str
    count: int
    record_width: int

    def __post_init__(self) -> None:
        _safe_relative_path(self.path, "index artifact path")
        _validate_count(self.bytes, "index artifact bytes")
        _validate_sha256(self.sha256, "index artifact sha256")
        _validate_count(self.count, "index artifact count")
        if type(self.record_width) is not int or self.record_width <= 0:
            raise ValueError("index artifact record width must be a positive integer")
        if self.count * self.record_width != self.bytes:
            raise ValueError(
                "index artifact count/record width/bytes are inconsistent"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "bytes": self.bytes,
            "count": self.count,
            "path": self.path,
            "record_width": self.record_width,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "IndexArtifactRecord":
        if (
            not isinstance(value, dict)
            or set(value) != {"bytes", "count", "path", "record_width", "sha256"}
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
            record_type=IndexArtifactRecord,
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
            record_factory = (
                IndexArtifactRecord.from_dict
                if name == "indexes"
                else ArtifactRecord.from_dict
            )
            inventories[name] = tuple(
                record_factory(record) for record in raw_inventory
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


_IdentityTuple = tuple[int, int, int, int, int, int, int | None, int | None]


@dataclass(frozen=True)
class WikidataDerivedViewRef:
    """Data-only pointer to a published derived view.

    A reference never authorizes a read. It carries only the published root path
    and the content-address commitment. To read, a caller must open a live
    verified descriptor session with ``open_wikidata_derived_view``; a bare
    reference (or any hand-constructed object) is inert at every read entry.
    """

    root: Path
    receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path) or not self.root.name:
            raise ValueError("derived view ref root must be a filesystem path")
        _validate_sha256(self.receipt_sha256, "receipt sha256")


@dataclass
class _RetainedFile:
    """A committed file held open for the whole session with its pinned facts."""

    subdir: str
    name: str
    fd: int
    identity: _IdentityTuple


class WikidataDerivedView:
    """Live, context-managed verified descriptor session for reads.

    ``open_wikidata_derived_view`` pins and retains the source-lock, source-root,
    view parent/name/root, receipt, stream, and index descriptors; verifies the
    complete source and receipt bindings; and hashes the same descriptors it
    keeps. Every read uses those retained descriptors and repeats a post-read
    ``fstat`` plus parent/name identity replay before returning or yielding
    bytes. The session cannot be forged: a hand-constructed object has no live
    ``_open`` state or retained descriptors, so every read rejects it. Closing
    the context deterministically closes every retained descriptor and disables
    further reads. There is no importable mint token and no data-only field that
    can bless a read.
    """

    def __init__(
        self,
        *,
        root: Path,
        receipt_sha256: str,
        receipt: "WikidataDerivedViewReceipt",
        source_lock_sha256: str,
        generator_commit: str,
        view_parent_fd: int,
        view_name: str,
        view_parent_identity: _IdentityTuple,
        view_fd: int,
        view_identity: _IdentityTuple,
        receipt_file: "_RetainedFile",
        subdir_fds: "dict[str, tuple[int, _IdentityTuple]]",
        files: "dict[str, _RetainedFile]",
    ) -> None:
        self.root = root
        self.receipt_sha256 = receipt_sha256
        self.receipt = receipt
        self.source_lock_sha256 = source_lock_sha256
        self.generator_commit = generator_commit
        self._view_parent_fd = view_parent_fd
        self._view_name = view_name
        self._view_parent_identity = view_parent_identity
        self._view_fd = view_fd
        self._view_identity = view_identity
        self._receipt_file = receipt_file
        self._subdir_fds = subdir_fds
        self._files = files
        self._open = True

    def _require_open(self) -> None:
        if getattr(self, "_open", False) is not True:
            raise ValueError(
                "WikidataDerivedView session is closed or was never opened"
            )

    def _close(self) -> None:
        self._open = False
        # Deterministic descriptor teardown: committed files, then subdirs,
        # then receipt, then the view root, then its parent.
        errors: list[BaseException] = []
        for retained in self._files.values():
            try:
                os.close(retained.fd)
            except OSError as error:
                errors.append(error)
        for subdir_fd, _identity in self._subdir_fds.values():
            try:
                os.close(subdir_fd)
            except OSError as error:
                errors.append(error)
        for descriptor in (
            self._receipt_file.fd,
            self._view_fd,
            self._view_parent_fd,
        ):
            try:
                os.close(descriptor)
            except OSError as error:
                errors.append(error)
        self._files = {}
        self._subdir_fds = {}
        if errors:
            raise ValueError(
                "derived view session descriptors did not close cleanly"
            ) from errors[0]


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
    # The live, pinned source-root directory descriptor. Build proves output
    # disjointness against this exact descriptor (never a re-stat by path).
    source_root_fd: int = field(default=-1, repr=False, compare=False)


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


def _validate_archive_envelope(descriptor: int, archive_path: str) -> None:
    duplicate = os.dup(descriptor)
    os.lseek(duplicate, 0, os.SEEK_SET)
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
            consume(decompressor.decompress(compressed))
        if not decompressor.eof:
            raise ValueError(f"archive gzip stream is truncated: {archive_path}")
        consume(decompressor.flush())
        if decompressor.unused_data or os.read(duplicate, 1):
            raise ValueError(
                f"archive has trailing compressed payload: {archive_path}"
            )
    except zlib.error as error:
        raise ValueError(f"archive gzip stream is malformed: {archive_path}") from error
    finally:
        os.close(duplicate)

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


def _parse_archive_descriptor(
    descriptor: int,
    archive_path: str,
    planned: dict[str, str],
) -> tuple[ArtifactRecord, ...]:
    _validate_archive_envelope(descriptor, archive_path)
    duplicate = os.dup(descriptor)
    os.lseek(duplicate, 0, os.SEEK_SET)
    records: list[ArtifactRecord] = []
    seen: set[str] = set()
    try:
        with os.fdopen(duplicate, "rb") as handle:
            duplicate = -1
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
    finally:
        if duplicate >= 0:
            os.close(duplicate)
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
        expected_generator_commit=lock.generator_commit,
    )
    return lock, payload


@contextmanager
def _open_verified_archives(
    source_lock_path: Path,
    source_root: Path,
) -> Iterator[VerifiedArchiveSet]:
    lock_parent_fd = -1
    lock_fd = -1
    root_parent_fd = -1
    root_fd = -1
    wikidata_fd = -1
    archive_fds: dict[str, int] = {}
    state: _ArchiveAuthorityState | None = None
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
        lock, lock_payload = _load_pinned_source_lock(lock_fd, lock_identity)
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
            source_root_fd=root_fd,
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
            source_root_fd=root_fd,
        )
        state.verified = verified
        _verify_authority_state(state, rehash=False)
        _archive_authority_hook("before_yield", None, verified)
        _verify_authority_state(state, rehash=False)
        yield verified
        _archive_authority_hook("before_postcheck", None, verified)
        _verify_authority_state(state, rehash=True)
    except OSError as error:
        raise ValueError("Wikidata archive authority is missing or unsafe") from error
    finally:
        for descriptor in archive_fds.values():
            os.close(descriptor)
        if wikidata_fd >= 0:
            os.close(wikidata_fd)
        if root_fd >= 0:
            os.close(root_fd)
        if root_parent_fd >= 0:
            os.close(root_parent_fd)
        if lock_fd >= 0:
            os.close(lock_fd)
        if lock_parent_fd >= 0:
            os.close(lock_parent_fd)


# --- Task 2: canonical streams, indexes, and no-replace publication ----------

SEALED_MEMBERS = (
    "wikidata5m_inductive_test.txt",
    "wikidata5m_inductive_valid.txt",
    "wikidata5m_transductive_test.txt",
    "wikidata5m_transductive_valid.txt",
)
_TRAINING_MEMBER_BY_SPLIT: Mapping[str, str] = MappingProxyType(
    {
        "inductive_train": "wikidata5m_inductive_train.txt",
        "transductive_train": "wikidata5m_transductive_train.txt",
    }
)
_TRAINING_ARCHIVE_BY_SPLIT: Mapping[str, str] = MappingProxyType(
    {
        "inductive_train": "wikidata5m_inductive.tar.gz",
        "transductive_train": "wikidata5m_transductive.tar.gz",
    }
)
_TRAINING_INDEX_BY_SPLIT: Mapping[str, str] = MappingProxyType(
    {
        "inductive_train": "indexes/inductive-training-offsets.bin",
        "transductive_train": "indexes/transductive-training-offsets.bin",
    }
)
_KIND_RANK: Mapping[str, int] = MappingProxyType({"entity": 0, "relation": 1})
_ALIAS_INDEX_WIDTH = 24
_TRAINING_INDEX_WIDTH = 8
_MAX_CANONICAL_NUMERIC = 1 << 63
_LINE_READ_CHUNK = 1 << 16
_VIEW_NAMESPACE = "wikidata"
_TRAINING_STREAM = "streams/training.tsv"
_ALIAS_STREAM = "streams/aliases.tsv"
_DISTINCT_STREAM = "streams/distinct-edges.tsv"
_ALIAS_INDEX = "indexes/aliases.bin"
_BUILD_SCRATCH_NAME = ".build-scratch"
_O_WRITE_FILE = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def _pread(descriptor: int, count: int, offset: int) -> bytes:
    """Positional read used by all bounded indexed-lookup reads."""

    return os.pread(descriptor, count, offset)


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _artifact_record(path: str, payload: bytes) -> "ArtifactRecord":
    return ArtifactRecord(
        path=path,
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _index_artifact_record(
    path: str,
    *,
    byte_count: int,
    sha256: str,
    count: int,
    record_width: int,
) -> "IndexArtifactRecord":
    return IndexArtifactRecord(
        path=path,
        bytes=byte_count,
        sha256=sha256,
        count=count,
        record_width=record_width,
    )


def _canonical_kind(canonical_id: str) -> str:
    if _QID_RE.fullmatch(canonical_id) is not None:
        return "entity"
    if _PID_RE.fullmatch(canonical_id) is not None:
        return "relation"
    raise ValueError(f"invalid canonical ID: {canonical_id!r}")


def _alias_sort_key(canonical_id: str) -> tuple[int, int]:
    kind = _canonical_kind(canonical_id)
    numeric = int(canonical_id[1:])
    if numeric >= _MAX_CANONICAL_NUMERIC:
        raise ValueError(f"canonical numeric id is out of range: {canonical_id!r}")
    return (_KIND_RANK[kind], numeric)


def _alias_index_key(canonical_id: str) -> int:
    kind_rank, numeric = _alias_sort_key(canonical_id)
    return (kind_rank << 63) | numeric


def _encode_training_row(
    training_split: str,
    row: int,
    subject: int,
    relation: str,
    object_: int,
) -> bytes:
    return (
        f"{training_split}\t{row}\tQ{subject}\t{relation}\tQ{object_}\n"
    ).encode("utf-8")


def _encode_alias_row(
    canonical_id: str,
    display: str,
    aliases: tuple[str, ...],
) -> bytes:
    value = _compact_json({"aliases": list(aliases), "display": display})
    return f"{canonical_id}\t{value}\n".encode("utf-8")


def _training_triple_from_row(line: bytes) -> "V2TrainingTriple":
    fields = line.split(b"\t")
    if len(fields) != 5:
        raise ValueError("training row must have five tab-separated fields")
    try:
        training_split = fields[0].decode("utf-8")
        row_text = fields[1].decode("utf-8")
        subject = parse_qid(fields[2].decode("utf-8"))
        relation = parse_pid(fields[3].decode("utf-8"))
        object_ = parse_qid(fields[4].decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError("training row is not UTF-8") from error
    if training_split not in TRAINING_SPLITS:
        raise ValueError("training row split is not declared")
    if not row_text.isascii() or not row_text.isdigit() or (
        len(row_text) > 1 and row_text[0] == "0"
    ):
        raise ValueError("training row index is not canonical")
    row = int(row_text)
    triple = V2TrainingTriple(
        training_split=training_split,
        row=row,
        subject=subject,
        relation=relation,
        object=object_,
        member=_TRAINING_MEMBER_BY_SPLIT[training_split],
        archive_path=_TRAINING_ARCHIVE_BY_SPLIT[training_split],
    )
    if _encode_training_row(training_split, row, subject, relation, object_) != (
        line + b"\n"
    ):
        raise ValueError("training row is not canonical")
    return triple


def _alias_record_from_row(line: bytes) -> "V2AliasRecord":
    tab = line.find(b"\t")
    if tab < 0:
        raise ValueError("alias row must have a canonical id and a value")
    canonical_id = line[:tab].decode("utf-8", "surrogateescape")
    value = _strict_json_bytes(line[tab + 1 :], "alias stream value")
    if not isinstance(value, dict) or set(value) != {"aliases", "display"}:
        raise ValueError("alias value fields do not match the contract")
    raw_aliases = value["aliases"]
    display = value["display"]
    if not isinstance(raw_aliases, list) or not all(
        isinstance(alias, str) for alias in raw_aliases
    ):
        raise ValueError("alias list must be a JSON array of strings")
    if not isinstance(display, str):
        raise ValueError("alias display must be a JSON string")
    record = V2AliasRecord(
        canonical_id=canonical_id,
        kind=_canonical_kind(canonical_id),
        display=display,
        aliases=tuple(raw_aliases),
    )
    if not record.aliases or record.display != record.aliases[0]:
        raise ValueError("alias display must equal the first surface form")
    if _encode_alias_row(canonical_id, display, record.aliases) != line + b"\n":
        raise ValueError("alias row is not canonical")
    return record


_SPLIT_BY_TRAINING_MEMBER: Mapping[str, str] = MappingProxyType(
    {member: split for split, member in _TRAINING_MEMBER_BY_SPLIT.items()}
)
_BUILD_SQLITE_PRAGMAS = (
    "PRAGMA journal_mode=OFF",
    "PRAGMA synchronous=OFF",
    "PRAGMA temp_store=FILE",
    "PRAGMA cache_size=-2048",
    "PRAGMA mmap_size=0",
)
_BUILD_SQLITE_SCHEMA = (
    "CREATE TABLE sealed ("
    "subject INTEGER NOT NULL, rel_num INTEGER NOT NULL, object INTEGER NOT NULL, "
    "PRIMARY KEY (subject, rel_num, object)) WITHOUT ROWID",
    "CREATE TABLE edge_first ("
    "subject INTEGER NOT NULL, rel_num INTEGER NOT NULL, object INTEGER NOT NULL, "
    "row_bytes BLOB NOT NULL, PRIMARY KEY (subject, rel_num, object)) WITHOUT ROWID",
    "CREATE TABLE alias_id (canonical_id TEXT PRIMARY KEY) WITHOUT ROWID",
    "CREATE TABLE alias_form ("
    "canonical_id TEXT NOT NULL, kind_rank INTEGER NOT NULL, numeric INTEGER NOT NULL, "
    "seq INTEGER NOT NULL, normalized TEXT NOT NULL, raw TEXT NOT NULL)",
    "CREATE INDEX alias_form_normalized ON alias_form (normalized)",
    "CREATE INDEX alias_form_order ON alias_form (kind_rank, numeric, seq)",
    "CREATE TABLE ambiguous (normalized TEXT PRIMARY KEY) WITHOUT ROWID",
)


class _StreamSink:
    """Ordered byte sink that hashes incrementally and optionally writes a file.

    Build and verification share one bounded code path: build passes a private
    directory descriptor and the sink writes an ``O_EXCL``/``O_NOFOLLOW`` 0600
    file while accumulating the byte count and SHA-256; verification passes
    ``None`` and the sink only accumulates the byte/hash tallies needed to
    recompute the source-derived receipt. No full stream is ever buffered.
    """

    __slots__ = ("_digest", "_count", "_fd")

    def __init__(self, directory_fd: int | None, name: str) -> None:
        self._digest = hashlib.sha256()
        self._count = 0
        self._fd = -1
        if directory_fd is not None:
            self._fd = os.open(name, _O_WRITE_FILE, 0o600, dir_fd=directory_fd)

    def write(self, chunk: bytes) -> int:
        offset = self._count
        if self._fd >= 0:
            view = memoryview(chunk)
            while view:
                view = view[os.write(self._fd, view):]
        self._digest.update(chunk)
        self._count += len(chunk)
        return offset

    def finalize(self) -> tuple[int, str]:
        if self._fd >= 0:
            os.fsync(self._fd)
            os.close(self._fd)
            self._fd = -1
        return self._count, self._digest.hexdigest()

    def abort(self) -> None:
        if self._fd >= 0:
            try:
                os.close(self._fd)
            finally:
                self._fd = -1


def _iter_member_line_bytes(stream: BinaryIO, sink: "_StreamSink") -> Iterator[bytes]:
    """Yield newline-delimited line bytes while mirroring raw bytes into ``sink``.

    One bounded pass both authenticates the decoded member (its raw bytes are
    hashed, and copied into ``members/`` when building) and feeds the row parser,
    without ever holding the whole member in memory.
    """

    buffer = bytearray()
    start = 0
    while True:
        chunk = stream.read(_READ_CHUNK_SIZE)
        if not chunk:
            break
        sink.write(chunk)
        buffer.extend(chunk)
        while True:
            newline = buffer.find(b"\n", start)
            if newline < 0:
                break
            yield bytes(buffer[start:newline])
            start = newline + 1
        if start:
            del buffer[:start]
            start = 0
    if start < len(buffer):
        yield bytes(buffer[start:])


def _decode_member_line(line: bytes, member: str) -> str:
    try:
        return line.decode("utf-8").rstrip("\r")
    except UnicodeDecodeError as error:
        raise ValueError(f"decoded member is not UTF-8: {member}") from error


def _process_sealed_member(
    lines: Iterator[bytes],
    member: str,
    cursor: "sqlite3.Cursor",
) -> None:
    for line_number, line in enumerate(lines, 1):
        row = _decode_member_line(line, member)
        if not row:
            continue
        fields = row.split("\t")
        if len(fields) != 3:
            raise ValueError(
                f"{member}:{line_number}: expected 3 tab-separated fields"
            )
        try:
            subject = parse_qid(fields[0])
            relation = parse_pid(fields[1])
            object_ = parse_qid(fields[2])
        except ValueError as error:
            raise ValueError(f"{member}:{line_number}: {error}") from error
        cursor.execute(
            "INSERT OR IGNORE INTO sealed (subject, rel_num, object) VALUES (?, ?, ?)",
            (subject, int(relation[1:]), object_),
        )


def _process_training_member(
    lines: Iterator[bytes],
    training_split: str,
    training_sink: "_StreamSink",
    offset_sink: "_StreamSink",
    cursor: "sqlite3.Cursor",
) -> int:
    member = _TRAINING_MEMBER_BY_SPLIT[training_split]
    row_index = 0
    for line_number, line in enumerate(lines, 1):
        row = _decode_member_line(line, member)
        if not row:
            continue
        fields = row.split("\t")
        if len(fields) != 3:
            raise ValueError(
                f"{member}:{line_number}: expected 3 tab-separated fields"
            )
        try:
            subject = parse_qid(fields[0])
            relation = parse_pid(fields[1])
            object_ = parse_qid(fields[2])
        except ValueError as error:
            raise ValueError(f"{member}:{line_number}: {error}") from error
        row_index += 1
        row_bytes = _encode_training_row(
            training_split, row_index, subject, relation, object_
        )
        offset = training_sink.write(row_bytes)
        offset_sink.write(struct.pack(">Q", offset))
        cursor.execute(
            "INSERT OR IGNORE INTO edge_first "
            "(subject, rel_num, object, row_bytes) VALUES (?, ?, ?, ?)",
            (subject, int(relation[1:]), object_, row_bytes),
        )
    return row_index


def _process_alias_member(
    lines: Iterator[bytes],
    prefix: str,
    member: str,
    cursor: "sqlite3.Cursor",
) -> None:
    parse_id = parse_pid if prefix == "P" else parse_qid
    for line_number, line in enumerate(lines, 1):
        row = _decode_member_line(line, member)
        if not row:
            continue
        fields = row.split("\t")
        if len(fields) < 2:
            raise ValueError(
                f"{member}:{line_number}: "
                "expected a canonical ID and at least one alias"
            )
        canonical_id = fields[0]
        try:
            parse_id(canonical_id)
        except ValueError as error:
            raise ValueError(f"{member}:{line_number}: {error}") from error
        try:
            cursor.execute(
                "INSERT INTO alias_id (canonical_id) VALUES (?)", (canonical_id,)
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(
                f"{member}:{line_number}: duplicate canonical ID: {canonical_id}"
            ) from error
        kind_rank, numeric = _alias_sort_key(canonical_id)
        first_seen: set[str] = set()
        seq = 0
        for raw_alias in fields[1:]:
            normalized = normalize_alias(raw_alias)
            if not normalized:
                raise ValueError(f"{member}:{line_number}: empty alias")
            if normalized in first_seen:
                continue
            first_seen.add(normalized)
            cursor.execute(
                "INSERT INTO alias_form "
                "(canonical_id, kind_rank, numeric, seq, normalized, raw) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (canonical_id, kind_rank, numeric, seq, normalized, raw_alias),
            )
            seq += 1


def _emit_alias_entry(
    canonical_id: str,
    key: tuple[int, int],
    forms: list[str],
    aliases_sink: "_StreamSink",
    alias_index_sink: "_StreamSink",
) -> int:
    if not forms:
        return 0
    row_bytes = _encode_alias_row(canonical_id, forms[0], tuple(forms))
    offset = aliases_sink.write(row_bytes)
    alias_index_sink.write(
        struct.pack(">QQQ", (key[0] << 63) | key[1], offset, len(row_bytes))
    )
    return 1


def _stream_view_artifacts(
    verified: VerifiedArchiveSet,
    generator_commit: str,
    *,
    view_fd: int | None,
    scratch_parent_fd: int,
    scratch_parent_path: str,
) -> "WikidataDerivedViewReceipt":
    """Stream canonical streams/indexes from the verified archives.

    Peak memory stays bounded regardless of source size: decoded members are
    streamed straight to ``members/`` (or only hashed, when verifying) while a
    private on-disk temporary SQLite database performs edge dedup/first
    provenance, the global alias ambiguity audit, and the external ordering.
    Every stream and index is written and hashed incrementally, and the receipt
    is assembled from streamed byte/row/SHA-256 tallies. Passing ``view_fd``
    writes the derived tree into that private directory; passing ``None`` only
    recomputes the source-derived receipt for verification.
    """

    member_by_path = {record.path: record for record in verified.members}
    opened_dir_fds: list[int] = []
    sinks: list[_StreamSink] = []
    connection: sqlite3.Connection | None = None
    # External sort state lives inside the exact descriptor-bound private build
    # directory, created descriptor-relative and removed the same way, never a
    # process-global temporary directory.
    os.mkdir(_BUILD_SCRATCH_NAME, 0o700, dir_fd=scratch_parent_fd)
    body_error: BaseException | None = None
    try:
        members_fd: int | None = None
        streams_fd: int | None = None
        indexes_fd: int | None = None
        if view_fd is not None:
            members_fd, _created = open_directory_at(view_fd, "members", create=True)
            opened_dir_fds.append(members_fd)
            streams_fd, _created = open_directory_at(view_fd, "streams", create=True)
            opened_dir_fds.append(streams_fd)
            indexes_fd, _created = open_directory_at(view_fd, "indexes", create=True)
            opened_dir_fds.append(indexes_fd)

        scratch_path = os.path.join(scratch_parent_path, _BUILD_SCRATCH_NAME)
        connection = sqlite3.connect(
            os.path.join(scratch_path, "build.sqlite3"),
            isolation_level=None,
        )
        for pragma in _BUILD_SQLITE_PRAGMAS:
            connection.execute(pragma)
        # Contain any SQLite temporary files inside the same private scratch.
        connection.execute(
            "PRAGMA temp_store_directory = '" + scratch_path.replace("'", "''") + "'"
        )
        for statement in _BUILD_SQLITE_SCHEMA:
            connection.execute(statement)
        # One scratch transaction batches every insert (read-your-writes keeps
        # the audit/ordering queries correct); it is discarded with the DB.
        connection.execute("BEGIN")
        cursor = connection.cursor()

        training_sink = _StreamSink(streams_fd, "training.tsv")
        aliases_sink = _StreamSink(streams_fd, "aliases.tsv")
        distinct_sink = _StreamSink(streams_fd, "distinct-edges.tsv")
        alias_index_sink = _StreamSink(indexes_fd, "aliases.bin")
        inductive_index_sink = _StreamSink(
            indexes_fd, "inductive-training-offsets.bin"
        )
        transductive_index_sink = _StreamSink(
            indexes_fd, "transductive-training-offsets.bin"
        )
        sinks = [
            training_sink,
            aliases_sink,
            distinct_sink,
            alias_index_sink,
            inductive_index_sink,
            transductive_index_sink,
        ]
        offset_sink_by_split = {
            "inductive_train": inductive_index_sink,
            "transductive_train": transductive_index_sink,
        }
        rows_by_split = {"inductive_train": 0, "transductive_train": 0}

        seen_members: set[str] = set()
        for archive_path in ARCHIVE_PATHS:
            wanted = set(ARCHIVE_MEMBERS[archive_path])
            descriptor = verified._archive_descriptors[archive_path]
            duplicate = os.dup(descriptor)
            os.lseek(duplicate, 0, os.SEEK_SET)
            try:
                with os.fdopen(duplicate, "rb") as handle:
                    duplicate = -1
                    with tarfile.open(
                        fileobj=cast(BinaryIO, handle),
                        mode="r:*",
                    ) as archive:
                        for member in archive:
                            name = member.name
                            if name not in wanted:
                                continue
                            member_path = f"members/{name}"
                            stream = archive.extractfile(member)
                            if stream is None:
                                raise ValueError(
                                    f"decoded member is unreadable: "
                                    f"{archive_path}:{name}"
                                )
                            member_sink = _StreamSink(members_fd, name)
                            try:
                                with stream:
                                    lines = _iter_member_line_bytes(
                                        cast(BinaryIO, stream), member_sink
                                    )
                                    if name == "wikidata5m_entity.txt":
                                        _process_alias_member(
                                            lines, "Q", name, cursor
                                        )
                                    elif name == "wikidata5m_relation.txt":
                                        _process_alias_member(
                                            lines, "P", name, cursor
                                        )
                                    elif name in SEALED_MEMBERS:
                                        _process_sealed_member(lines, name, cursor)
                                    elif name in _SPLIT_BY_TRAINING_MEMBER:
                                        split = _SPLIT_BY_TRAINING_MEMBER[name]
                                        rows_by_split[split] = (
                                            _process_training_member(
                                                lines,
                                                split,
                                                training_sink,
                                                offset_sink_by_split[split],
                                                cursor,
                                            )
                                        )
                                    else:
                                        raise ValueError(
                                            f"undeclared decoded member: "
                                            f"{archive_path}:{name}"
                                        )
                                member_bytes, member_sha = member_sink.finalize()
                            except BaseException:
                                member_sink.abort()
                                raise
                            expected = member_by_path.get(member_path)
                            if (
                                expected is None
                                or member_bytes != expected.bytes
                                or member_sha != expected.sha256
                            ):
                                raise ValueError(
                                    f"decoded member payload drift: {member_path}"
                                )
                            seen_members.add(member_path)
            except (EOFError, OSError, tarfile.TarError) as error:
                raise ValueError(
                    f"decoded member extraction failed: {archive_path}"
                ) from error
            finally:
                if duplicate >= 0:
                    os.close(duplicate)

        if seen_members != set(member_by_path):
            raise ValueError("decoded member inventory drift")

        # The train/sealed overlap audit runs once every sealed member is loaded.
        overlap = connection.execute(
            "SELECT 1 FROM edge_first AS e WHERE EXISTS ("
            "SELECT 1 FROM sealed AS s WHERE s.subject = e.subject "
            "AND s.rel_num = e.rel_num AND s.object = e.object) LIMIT 1"
        ).fetchone()
        if overlap is not None:
            raise ValueError(
                "training/sealed edge overlap detected before publication"
            )

        # Distinct edges: first-provenance training rows in numeric edge order.
        distinct_edges = 0
        distinct_cursor = connection.cursor()
        distinct_cursor.execute(
            "SELECT row_bytes FROM edge_first ORDER BY subject, rel_num, object"
        )
        for (row_bytes,) in distinct_cursor:
            distinct_sink.write(row_bytes)
            distinct_edges += 1

        # Global alias ambiguity audit, then entity-then-relation emission.
        connection.execute(
            "INSERT INTO ambiguous (normalized) SELECT normalized FROM alias_form "
            "GROUP BY normalized HAVING COUNT(*) > 1"
        )
        alias_rows = 0
        alias_cursor = connection.cursor()
        alias_cursor.execute(
            "SELECT canonical_id, kind_rank, numeric, raw FROM alias_form "
            "WHERE normalized NOT IN (SELECT normalized FROM ambiguous) "
            "ORDER BY kind_rank, numeric, seq"
        )
        pending_id: str | None = None
        pending_key = (0, 0)
        pending_forms: list[str] = []
        pending_seen: set[str] = set()
        for canonical_id, kind_rank, numeric, raw in alias_cursor:
            if canonical_id != pending_id:
                if pending_id is not None:
                    alias_rows += _emit_alias_entry(
                        pending_id,
                        pending_key,
                        pending_forms,
                        aliases_sink,
                        alias_index_sink,
                    )
                pending_id = canonical_id
                pending_key = (kind_rank, numeric)
                pending_forms = []
                pending_seen = set()
            if "\x00" in raw:
                raise ValueError(
                    f"alias surface contains a NUL byte: {canonical_id}"
                )
            form = unicodedata.normalize("NFC", raw)
            if "\x00" in form:
                raise ValueError(
                    f"alias surface contains a NUL byte after NFC: {canonical_id}"
                )
            if not form or form in pending_seen:
                continue
            pending_seen.add(form)
            pending_forms.append(form)
        if pending_id is not None:
            alias_rows += _emit_alias_entry(
                pending_id,
                pending_key,
                pending_forms,
                aliases_sink,
                alias_index_sink,
            )

        training_rows = (
            rows_by_split["inductive_train"] + rows_by_split["transductive_train"]
        )

        stream_records: dict[str, ArtifactRecord] = {}
        for path, sink in (
            (_ALIAS_STREAM, aliases_sink),
            (_DISTINCT_STREAM, distinct_sink),
            (_TRAINING_STREAM, training_sink),
        ):
            byte_count, sha256 = sink.finalize()
            stream_records[path] = ArtifactRecord(
                path=path, bytes=byte_count, sha256=sha256
            )

        index_records: dict[str, IndexArtifactRecord] = {}
        for path, sink, count, width in (
            (_ALIAS_INDEX, alias_index_sink, alias_rows, _ALIAS_INDEX_WIDTH),
            (
                "indexes/inductive-training-offsets.bin",
                inductive_index_sink,
                rows_by_split["inductive_train"],
                _TRAINING_INDEX_WIDTH,
            ),
            (
                "indexes/transductive-training-offsets.bin",
                transductive_index_sink,
                rows_by_split["transductive_train"],
                _TRAINING_INDEX_WIDTH,
            ),
        ):
            byte_count, sha256 = sink.finalize()
            index_records[path] = IndexArtifactRecord(
                path=path,
                bytes=byte_count,
                sha256=sha256,
                count=count,
                record_width=width,
            )

        receipt = WikidataDerivedViewReceipt(
            format=RECEIPT_FORMAT,
            schema_version=RECEIPT_SCHEMA_VERSION,
            source_lock_sha256=verified.source_lock_sha256,
            generator_commit=generator_commit,
            archives=verified.archives,
            members=verified.members,
            streams=tuple(stream_records[path] for path in STREAM_PATHS),
            indexes=tuple(index_records[path] for path in INDEX_PATHS),
            training_rows=training_rows,
            alias_rows=alias_rows,
            distinct_edges=distinct_edges,
            overlap_audit_passed=True,
        )
        if view_fd is not None:
            for descriptor in opened_dir_fds:
                fsync_directory(descriptor)
        return receipt
    except BaseException as error:
        body_error = error
        for sink in sinks:
            sink.abort()
        raise
    finally:
        if connection is not None:
            connection.close()
        for descriptor in opened_dir_fds:
            os.close(descriptor)
        try:
            _remove_view_tree(scratch_parent_fd, _BUILD_SCRATCH_NAME)
        except OSError as cleanup_error:
            if body_error is None:
                raise ValueError("build scratch cleanup failed") from cleanup_error


def _write_new_file(directory_fd: int, name: str, payload: bytes) -> None:
    descriptor = os.open(name, _O_WRITE_FILE, 0o600, dir_fd=directory_fd)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_view_tree(parent_fd: int, name: str) -> None:
    try:
        metadata = entry_lstat(parent_fd, name)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(metadata.st_mode):
        descriptor, _created = open_directory_at(parent_fd, name)
        try:
            for child in list_entries(descriptor):
                _remove_view_tree(descriptor, child)
        finally:
            os.close(descriptor)
        os.rmdir(name, dir_fd=parent_fd)
    else:
        os.unlink(name, dir_fd=parent_fd)


_O_ANCESTOR_FLAGS = (
    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
)


def _assert_output_outside_source(output_root: Path, source_root: Path) -> None:
    """Cheap path pre-check so no output tree is ever created inside the source.

    This is a fail-fast guard only; the authoritative, TOCTOU-safe proof is
    ``_prove_output_disjoint_from_source`` which walks the retained output
    descriptor's live ancestor chain against the pinned source-root descriptor.
    """

    try:
        source_meta = os.stat(source_root)
    except OSError as error:
        raise ValueError("source root is missing or unreadable") from error
    if not stat.S_ISDIR(source_meta.st_mode):
        raise ValueError("source root is not a directory")
    source_id = (source_meta.st_dev, source_meta.st_ino)

    target = Path(output_root)
    existing: Path | None = None
    for ancestor in (target, *target.parents):
        if ancestor.exists():
            existing = ancestor
            break
    if existing is None:
        raise ValueError("output root has no existing ancestor to authenticate")

    current = open_directory_path(existing)
    try:
        while True:
            metadata = os.fstat(current)
            if (metadata.st_dev, metadata.st_ino) == source_id:
                raise ValueError(
                    "output root is inside the immutable source root"
                )
            parent = os.open("..", _O_ANCESTOR_FLAGS, dir_fd=current)
            parent_metadata = os.fstat(parent)
            if (parent_metadata.st_dev, parent_metadata.st_ino) == (
                metadata.st_dev,
                metadata.st_ino,
            ):
                os.close(parent)
                break
            os.close(current)
            current = parent
    finally:
        os.close(current)


def _ensure_directory(path: Path) -> None:
    """Materialize a directory (and parents) and immediately release it.

    Called before the source authority is pinned, so creating the output tree
    cannot mutate a parent shared with the source lock/root during the pinned
    context (which would trip Task 1's namespace postcheck).
    """

    descriptor = open_directory_path(Path(path), create=True)
    os.close(descriptor)


def _prove_output_disjoint_from_source(output_fd: int, source_root_fd: int) -> None:
    """Prove the retained output descriptor is neither the pinned source root
    nor its descendant, by walking the output's live ancestor descriptors and
    comparing device/inode against the exact pinned source-root descriptor."""

    source_meta = os.fstat(source_root_fd)
    source_id = (source_meta.st_dev, source_meta.st_ino)
    current = os.open(".", _O_ANCESTOR_FLAGS, dir_fd=output_fd)
    try:
        while True:
            metadata = os.fstat(current)
            if (metadata.st_dev, metadata.st_ino) == source_id:
                raise ValueError(
                    "output root is inside the immutable source root"
                )
            parent = os.open("..", _O_ANCESTOR_FLAGS, dir_fd=current)
            parent_metadata = os.fstat(parent)
            if (parent_metadata.st_dev, parent_metadata.st_ino) == (
                metadata.st_dev,
                metadata.st_ino,
            ):
                os.close(parent)
                break
            os.close(current)
            current = parent
    finally:
        os.close(current)


def _build_output_hook(phase: str) -> None:
    """Test seam fired at output-authority phases (e.g. substitution races)."""

    del phase


def _quarantine_and_remove_private(
    namespace_fd: int,
    private_name: str,
    private_identity: tuple[int, int],
) -> None:
    """Descriptor-relative exact-inode quarantine cleanup, reporting errors.

    Never acts on a raced replacement of the private name; renames the exact
    inode to a private quarantine name, re-verifies the moved identity, removes
    that authority, and fsyncs the namespace. Cleanup failures are raised rather
    than swallowed so callers can report them.
    """

    try:
        current = entry_lstat(namespace_fd, private_name)
    except FileNotFoundError:
        return
    if (
        not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != private_identity
    ):
        # Never touch a raced replacement of the private name.
        return
    quarantine_name = f".{_VIEW_NAMESPACE}-view-quarantine-{secrets.token_hex(16)}"
    atomic_rename_noreplace(
        namespace_fd,
        private_name,
        namespace_fd,
        quarantine_name,
    )
    fsync_directory(namespace_fd)
    moved = entry_lstat(namespace_fd, quarantine_name)
    if (
        not stat.S_ISDIR(moved.st_mode)
        or (moved.st_dev, moved.st_ino) != private_identity
    ):
        raise ValueError(
            "private build directory identity drifted during quarantine"
        )
    _remove_view_tree(namespace_fd, quarantine_name)
    fsync_directory(namespace_fd)


def _stream_build_into_output(
    verified: VerifiedArchiveSet,
    commit: str,
    output_root: Path,
    output_fd: int,
    output_parent_fd: int,
    output_name: str,
    output_identity: _IdentityTuple,
) -> str:
    """Stream, authenticate, and no-replace publish the view under ``output_fd``.

    Uses only the retained, proven-disjoint output descriptor. External sort
    state lives inside the exact private build directory and is cleaned up
    descriptor-relative; a losing or failing build quarantines only its own
    private inode and reports cleanup errors.
    """

    namespace_fd, _created = open_directory_at(output_fd, _VIEW_NAMESPACE, create=True)
    try:
        _require_owned_mode(
            os.fstat(namespace_fd),
            directory=True,
            description="output wikidata namespace",
        )
        private_name = f".{_VIEW_NAMESPACE}-view-build-{secrets.token_hex(16)}"
        os.mkdir(private_name, 0o700, dir_fd=namespace_fd)
        created = entry_lstat(namespace_fd, private_name)
        private_identity = (created.st_dev, created.st_ino)
        private_path = Path(output_root) / _VIEW_NAMESPACE / private_name
        published = False
        content_address = ""
        try:
            private_fd, _c = open_directory_at(namespace_fd, private_name)
            try:
                private_meta = os.fstat(private_fd)
                if (private_meta.st_dev, private_meta.st_ino) != private_identity:
                    raise ValueError("private build directory identity drift")
                _require_fixed_mode(
                    private_meta,
                    directory=True,
                    description="private build directory",
                )
                receipt = _stream_view_artifacts(
                    verified,
                    commit,
                    view_fd=private_fd,
                    scratch_parent_fd=private_fd,
                    scratch_parent_path=str(private_path),
                )
                receipt_bytes = receipt.to_bytes()
                content_address = hashlib.sha256(receipt_bytes).hexdigest()
                _write_new_file(private_fd, "receipt.json", receipt_bytes)
                fsync_directory(private_fd)
                # Re-verify every derived byte against the committed receipt.
                _authenticate_view_directory(
                    private_fd,
                    expected_content_address=content_address,
                    expected_commit=commit,
                )
            finally:
                os.close(private_fd)
            # An output-name substitution between the disjointness proof and the
            # write must fail: replay the output name against the retained
            # output descriptor's identity before publishing.
            _build_output_hook("before_publish")
            replay = entry_lstat(output_parent_fd, output_name)
            # Only device/inode substitution matters here; our own namespace
            # creation legitimately bumps the output root's mtime/ctime.
            if (replay.st_dev, replay.st_ino) != (
                output_identity[0],
                output_identity[1],
            ):
                raise ValueError(
                    "output root identity drifted before publication"
                )
            if entry_exists(namespace_fd, content_address):
                # A winner already exists: fully re-verify it against the
                # committed receipt before reuse and never repair it in place;
                # then discard our private build.
                winner_fd, _w = open_directory_at(namespace_fd, content_address)
                try:
                    _require_fixed_mode(
                        os.fstat(winner_fd),
                        directory=True,
                        description="published derived view",
                    )
                    _authenticate_view_directory(
                        winner_fd,
                        expected_content_address=content_address,
                        expected_commit=commit,
                    )
                finally:
                    os.close(winner_fd)
            else:
                try:
                    atomic_rename_noreplace(
                        namespace_fd,
                        private_name,
                        namespace_fd,
                        content_address,
                    )
                except FileExistsError:
                    pass
                else:
                    published = True
                    fsync_directory(namespace_fd)
        finally:
            if not published:
                pending = sys.exc_info()[1]
                try:
                    _quarantine_and_remove_private(
                        namespace_fd,
                        private_name,
                        private_identity,
                    )
                except BaseException:
                    if pending is None:
                        raise
        return content_address
    finally:
        os.close(namespace_fd)


def build_wikidata_derived_view(
    source_lock_path: Path,
    source_root: Path,
    output_root: Path,
    *,
    expected_generator_commit: str,
) -> "WikidataDerivedViewRef":
    commit = _validate_commit(expected_generator_commit, "expected generator commit")
    source_root = Path(source_root)
    output_root = Path(output_root)
    # Fail-fast path pre-check so no output tree is created inside the source.
    _assert_output_outside_source(output_root, source_root)
    # Materialize the output root BEFORE pinning the source authority, so its
    # creation cannot mutate a shared source parent during the pinned context.
    _ensure_directory(output_root)
    with _open_verified_archives(Path(source_lock_path), source_root) as verified:
        if verified.generator_commit != commit:
            raise ValueError(
                "expected generator commit does not match the source lock"
            )
        # Open and retain the actual output-root descriptor AFTER the source is
        # pinned, then prove disjointness against the exact pinned source-root
        # descriptor and use this same descriptor through build and publish.
        output_parent_fd, output_name = open_parent_directory(output_root)
        try:
            output_fd, output_identity = _open_bound_directory(
                output_parent_fd,
                output_name,
                "output root",
            )
            try:
                _prove_output_disjoint_from_source(
                    output_fd, verified.source_root_fd
                )
                _build_output_hook("after_disjointness")
                # A substitution of the output name after the descriptor proof
                # (but before any name-resolved write) is rejected here, before
                # the private build directory or its scratch is touched.
                replay = entry_lstat(output_parent_fd, output_name)
                if (replay.st_dev, replay.st_ino) != (
                    output_identity[0],
                    output_identity[1],
                ):
                    raise ValueError(
                        "output root identity drifted before publication"
                    )
                content_address = _stream_build_into_output(
                    verified,
                    commit,
                    output_root,
                    output_fd,
                    output_parent_fd,
                    output_name,
                    output_identity,
                )
            finally:
                os.close(output_fd)
        finally:
            os.close(output_parent_fd)
    return WikidataDerivedViewRef(
        root=Path(output_root) / _VIEW_NAMESPACE / content_address,
        receipt_sha256=content_address,
    )


def verify_wikidata_derived_view(
    source_lock_path: Path,
    source_root: Path,
    view_root: Path,
    *,
    expected_generator_commit: str,
) -> "WikidataDerivedViewRef":
    commit = _validate_commit(expected_generator_commit, "expected generator commit")
    view_root = Path(view_root)
    with _open_verified_archives(Path(source_lock_path), Path(source_root)) as verified:
        if verified.generator_commit != commit:
            raise ValueError(
                "expected generator commit does not match the source lock"
            )
        # Recompute the source content address with a bounded streaming pass; the
        # SQLite scratch lives inside a descriptor-bound private sibling of the
        # view, never a process-global temporary directory.
        scratch_parent_fd, scratch_name = open_parent_directory(view_root)
        try:
            namespace_path = view_root.parent
            verify_private = f".{_VIEW_NAMESPACE}-view-verify-{secrets.token_hex(16)}"
            os.mkdir(verify_private, 0o700, dir_fd=scratch_parent_fd)
            created = entry_lstat(scratch_parent_fd, verify_private)
            verify_identity = (created.st_dev, created.st_ino)
            try:
                verify_fd, _c = open_directory_at(scratch_parent_fd, verify_private)
                try:
                    receipt = _stream_view_artifacts(
                        verified,
                        commit,
                        view_fd=None,
                        scratch_parent_fd=verify_fd,
                        scratch_parent_path=str(namespace_path / verify_private),
                    )
                finally:
                    os.close(verify_fd)
            finally:
                _quarantine_and_remove_private(
                    scratch_parent_fd, verify_private, verify_identity
                )
        finally:
            os.close(scratch_parent_fd)
    content_address = hashlib.sha256(receipt.to_bytes()).hexdigest()
    if view_root.name != content_address:
        raise ValueError("derived view root is not the source content address")
    parent_fd, name, view_fd = _open_verified_view_directory(view_root)
    try:
        if name != content_address:
            raise ValueError("derived view root is not the content address")
        _authenticate_view_directory(
            view_fd,
            expected_content_address=content_address,
            expected_commit=commit,
        )
        replay = entry_lstat(parent_fd, name)
        if _directory_identity(replay) != _directory_identity(os.fstat(view_fd)):
            raise ValueError("derived view namespace identity drift after verify")
    finally:
        os.close(view_fd)
        os.close(parent_fd)
    return WikidataDerivedViewRef(root=view_root, receipt_sha256=content_address)


def _require_fixed_mode(
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
    mode = stat.S_IMODE(metadata.st_mode)
    if directory:
        if mode != 0o700:
            raise ValueError(f"{description} mode is not 0o700")
    else:
        if mode != 0o600:
            raise ValueError(f"{description} mode is not 0o600")
        if metadata.st_nlink != 1:
            raise ValueError(f"{description} is a hardlink")


def _stream_hash_file(descriptor: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, _READ_CHUNK_SIZE)
        if not chunk:
            break
        byte_count += len(chunk)
        digest.update(chunk)
    return byte_count, digest.hexdigest()


def _open_verified_view_directory(root: Path) -> tuple[int, str, int]:
    parent_fd, name = open_parent_directory(Path(root))
    try:
        named_before = entry_lstat(parent_fd, name)
        view_fd, _created = open_directory_at(parent_fd, name)
    except OSError as error:
        os.close(parent_fd)
        raise ValueError("derived view is missing or unsafe") from error
    except BaseException:
        os.close(parent_fd)
        raise
    try:
        opened = os.fstat(view_fd)
        _require_fixed_mode(opened, directory=True, description="derived view")
        named_after = entry_lstat(parent_fd, name)
        identity = _directory_identity(opened)
        if (
            _directory_identity(named_before) != identity
            or _directory_identity(named_after) != identity
        ):
            raise ValueError("derived view namespace identity drift")
    except BaseException:
        os.close(view_fd)
        os.close(parent_fd)
        raise
    return parent_fd, name, view_fd


def _read_all_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, _READ_CHUNK_SIZE)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _digest_committed_fd(
    descriptor: int,
    directory_fd: int,
    name: str,
    *,
    description: str,
) -> tuple[int, str, _IdentityTuple]:
    """Hash an open committed file with a stable pre/post + namespace replay."""

    before = os.fstat(descriptor)
    _require_fixed_mode(before, directory=False, description=description)
    identity = _file_identity(before)
    byte_count, sha256 = _stream_hash_file(descriptor)
    after = os.fstat(descriptor)
    named = entry_lstat(directory_fd, name)
    if _file_identity(after) != identity or _file_identity(named) != identity:
        raise ValueError(f"{description} identity drift during authentication")
    return byte_count, sha256, identity


def _authenticate_view_directory(
    view_fd: int,
    *,
    expected_content_address: str,
    expected_commit: str,
) -> "WikidataDerivedViewReceipt":
    """Fully authenticate an on-disk view tree against its committed receipt.

    Used for build's pre-publication re-verification and by ``verify``. Every
    committed file is opened, fixed-mode/owner/link checked, and hashed with a
    stable pre/post ``fstat`` plus named-entry replay so a concurrent in-place
    mutation or rename during authentication is rejected. Descriptors are not
    retained; ``open_wikidata_derived_view`` establishes the retained session.
    """

    if set(list_entries(view_fd)) != {
        "receipt.json",
        "members",
        "streams",
        "indexes",
    }:
        raise ValueError("derived view inventory does not match the contract")
    receipt_metadata = entry_lstat(view_fd, "receipt.json")
    _require_fixed_mode(
        receipt_metadata,
        directory=False,
        description="derived view receipt.json",
    )
    receipt_fd, receipt_meta = open_regular_file_at(view_fd, "receipt.json")
    try:
        _require_fixed_mode(
            receipt_meta,
            directory=False,
            description="derived view receipt.json",
        )
        identity = _file_identity(receipt_meta)
        payload = _read_all_fd(receipt_fd)
        after = os.fstat(receipt_fd)
        named = entry_lstat(view_fd, "receipt.json")
        if _file_identity(after) != identity or _file_identity(named) != identity:
            raise ValueError("derived view receipt.json identity drift")
    finally:
        os.close(receipt_fd)
    if hashlib.sha256(payload).hexdigest() != expected_content_address:
        raise ValueError("derived view content address mismatch")
    receipt = WikidataDerivedViewReceipt.from_bytes(
        payload,
        expected_generator_commit=expected_commit,
    )
    for subdir, records_seq in (
        ("members", receipt.members),
        ("streams", receipt.streams),
        ("indexes", receipt.indexes),
    ):
        records = {record.path: record for record in records_seq}
        directory_fd, _created = open_directory_at(view_fd, subdir)
        try:
            _require_fixed_mode(
                os.fstat(directory_fd),
                directory=True,
                description=f"derived view {subdir}/",
            )
            if set(list_entries(directory_fd)) != {
                PurePosixPath(path).name for path in records
            }:
                raise ValueError(
                    f"derived view {subdir} inventory does not match the receipt"
                )
            for path, record in records.items():
                entry_name = PurePosixPath(path).name
                descriptor, metadata = open_regular_file_at(directory_fd, entry_name)
                try:
                    _require_fixed_mode(
                        metadata,
                        directory=False,
                        description=f"derived view {path}",
                    )
                    byte_count, sha256, _identity = _digest_committed_fd(
                        descriptor,
                        directory_fd,
                        entry_name,
                        description=f"derived view {path}",
                    )
                finally:
                    os.close(descriptor)
                if byte_count != record.bytes or sha256 != record.sha256:
                    raise ValueError(
                        f"derived view file does not match the receipt: {path}"
                    )
        finally:
            os.close(directory_fd)
    return receipt


def _establish_view_session(
    verified: VerifiedArchiveSet,
    view: "WikidataDerivedViewRef",
    commit: str,
) -> "WikidataDerivedView":
    """Open, authenticate, and retain every descriptor a read session needs.

    Pins the view parent/name/root, receipt, and each stream/index descriptor;
    binds the receipt to the freshly verified source; hashes the same
    descriptors it keeps; records owner/mode/link/name identities; and replays
    the view-root parent/name after full authentication. The retained
    descriptors are handed to a live ``WikidataDerivedView`` and are the only
    authority any later read trusts.
    """

    parent_fd, name, view_fd = _open_verified_view_directory(Path(view.root))
    subdir_fds: dict[str, tuple[int, _IdentityTuple]] = {}
    files: dict[str, _RetainedFile] = {}
    receipt_fd = -1
    established = False
    try:
        if name != view.receipt_sha256:
            raise ValueError("derived view root name is not the content address")
        view_parent_identity = _directory_identity(os.fstat(parent_fd))
        view_identity = _directory_identity(os.fstat(view_fd))
        if set(list_entries(view_fd)) != {
            "receipt.json",
            "members",
            "streams",
            "indexes",
        }:
            raise ValueError("derived view inventory does not match the contract")

        receipt_metadata = entry_lstat(view_fd, "receipt.json")
        _require_fixed_mode(
            receipt_metadata,
            directory=False,
            description="derived view receipt.json",
        )
        receipt_fd, receipt_meta = open_regular_file_at(view_fd, "receipt.json")
        _require_fixed_mode(
            receipt_meta,
            directory=False,
            description="derived view receipt.json",
        )
        receipt_identity = _file_identity(receipt_meta)
        payload = _read_all_fd(receipt_fd)
        after = os.fstat(receipt_fd)
        named = entry_lstat(view_fd, "receipt.json")
        if _file_identity(after) != receipt_identity or _file_identity(named) != (
            receipt_identity
        ):
            raise ValueError("derived view receipt.json identity drift")
        content_address = hashlib.sha256(payload).hexdigest()
        if content_address != view.receipt_sha256 or content_address != name:
            raise ValueError("derived view content address mismatch")
        receipt = WikidataDerivedViewReceipt.from_bytes(
            payload,
            expected_generator_commit=commit,
        )
        if (
            receipt.source_lock_sha256 != verified.source_lock_sha256
            or receipt.generator_commit != commit
            or receipt.archives != verified.archives
            or receipt.members != verified.members
        ):
            raise ValueError("derived view receipt does not bind the verified source")
        receipt_file = _RetainedFile(
            subdir="",
            name="receipt.json",
            fd=receipt_fd,
            identity=receipt_identity,
        )

        members_fd, _created = open_directory_at(view_fd, "members")
        try:
            _require_fixed_mode(
                os.fstat(members_fd),
                directory=True,
                description="derived view members/",
            )
            member_records = {record.path: record for record in receipt.members}
            if set(list_entries(members_fd)) != {
                PurePosixPath(path).name for path in member_records
            }:
                raise ValueError(
                    "derived view members inventory does not match the receipt"
                )
            for path, record in member_records.items():
                entry_name = PurePosixPath(path).name
                descriptor, metadata = open_regular_file_at(members_fd, entry_name)
                try:
                    _require_fixed_mode(
                        metadata,
                        directory=False,
                        description=f"derived view {path}",
                    )
                    byte_count, sha256, _identity = _digest_committed_fd(
                        descriptor,
                        members_fd,
                        entry_name,
                        description=f"derived view {path}",
                    )
                finally:
                    os.close(descriptor)
                if byte_count != record.bytes or sha256 != record.sha256:
                    raise ValueError(
                        f"derived view file does not match the receipt: {path}"
                    )
        finally:
            os.close(members_fd)

        for subdir, records_seq in (
            ("streams", receipt.streams),
            ("indexes", receipt.indexes),
        ):
            records = {record.path: record for record in records_seq}
            directory_fd, _created = open_directory_at(view_fd, subdir)
            subdir_ok = False
            try:
                _require_fixed_mode(
                    os.fstat(directory_fd),
                    directory=True,
                    description=f"derived view {subdir}/",
                )
                if set(list_entries(directory_fd)) != {
                    PurePosixPath(path).name for path in records
                }:
                    raise ValueError(
                        f"derived view {subdir} inventory does not match the receipt"
                    )
                subdir_fds[subdir] = (
                    directory_fd,
                    _directory_identity(os.fstat(directory_fd)),
                )
                for path, record in records.items():
                    entry_name = PurePosixPath(path).name
                    descriptor, metadata = open_regular_file_at(
                        directory_fd, entry_name
                    )
                    retained = False
                    try:
                        _require_fixed_mode(
                            metadata,
                            directory=False,
                            description=f"derived view {path}",
                        )
                        byte_count, sha256, identity = _digest_committed_fd(
                            descriptor,
                            directory_fd,
                            entry_name,
                            description=f"derived view {path}",
                        )
                        if byte_count != record.bytes or sha256 != record.sha256:
                            raise ValueError(
                                f"derived view file does not match the receipt: {path}"
                            )
                        files[path] = _RetainedFile(
                            subdir=subdir,
                            name=entry_name,
                            fd=descriptor,
                            identity=identity,
                        )
                        retained = True
                    finally:
                        if not retained:
                            os.close(descriptor)
                subdir_ok = True
            finally:
                if not subdir_ok:
                    subdir_fds.pop(subdir, None)
                    os.close(directory_fd)

        # Replay the view-root parent/name after full authentication.
        replay_named = entry_lstat(parent_fd, name)
        if _directory_identity(replay_named) != view_identity or _directory_identity(
            os.fstat(view_fd)
        ) != view_identity:
            raise ValueError(
                "derived view namespace identity drift after authentication"
            )

        session = WikidataDerivedView(
            root=Path(view.root),
            receipt_sha256=content_address,
            receipt=receipt,
            source_lock_sha256=verified.source_lock_sha256,
            generator_commit=commit,
            view_parent_fd=parent_fd,
            view_name=name,
            view_parent_identity=view_parent_identity,
            view_fd=view_fd,
            view_identity=view_identity,
            receipt_file=receipt_file,
            subdir_fds=subdir_fds,
            files=files,
        )
        established = True
        return session
    finally:
        if not established:
            for retained_file in files.values():
                try:
                    os.close(retained_file.fd)
                except OSError:
                    pass
            for subdir_fd, _identity in subdir_fds.values():
                try:
                    os.close(subdir_fd)
                except OSError:
                    pass
            if receipt_fd >= 0:
                try:
                    os.close(receipt_fd)
                except OSError:
                    pass
            try:
                os.close(view_fd)
            except OSError:
                pass
            try:
                os.close(parent_fd)
            except OSError:
                pass


@contextmanager
def open_wikidata_derived_view(
    source_lock_path: Path,
    source_root: Path,
    view: "WikidataDerivedViewRef",
    *,
    expected_generator_commit: str,
) -> Iterator["WikidataDerivedView"]:
    """Open a live, verified descriptor session over a published derived view.

    The session pins the source authority and the view's receipt/stream/index
    descriptors for its whole lifetime, verifies the complete source and receipt
    bindings, and closes every descriptor deterministically on exit. A data-only
    reference never authorizes a read; only the yielded session does.
    """

    if not isinstance(view, WikidataDerivedViewRef):
        raise ValueError("open requires a data-only WikidataDerivedViewRef")
    commit = _validate_commit(expected_generator_commit, "expected generator commit")
    with _open_verified_archives(
        Path(source_lock_path), Path(source_root)
    ) as verified:
        if verified.generator_commit != commit:
            raise ValueError(
                "expected generator commit does not match the source lock"
            )
        session = _establish_view_session(verified, view, commit)
        try:
            yield session
        finally:
            session._close()


def _session_state(view: object) -> "WikidataDerivedView":
    if not isinstance(view, WikidataDerivedView):
        raise ValueError("read requires an open WikidataDerivedView session")
    view._require_open()
    return view


def _session_committed(view: "WikidataDerivedView", relative_path: str) -> "_RetainedFile":
    retained = view._files.get(relative_path)
    if retained is None:
        raise ValueError(f"derived view file is not committed: {relative_path}")
    return retained


def _replay_committed_identity(
    view: "WikidataDerivedView",
    retained: "_RetainedFile",
) -> None:
    """Post-read descriptor + parent/name replay before returning/yielding."""

    current = os.fstat(retained.fd)
    _require_fixed_mode(
        current,
        directory=False,
        description=f"derived view {retained.subdir}/{retained.name}",
    )
    if _file_identity(current) != retained.identity:
        raise ValueError(
            f"derived view file changed since verification: "
            f"{retained.subdir}/{retained.name}"
        )
    subdir_fd, subdir_identity = view._subdir_fds[retained.subdir]
    if _directory_identity(os.fstat(subdir_fd)) != subdir_identity:
        raise ValueError(f"derived view {retained.subdir}/ changed since verification")
    named = entry_lstat(subdir_fd, retained.name)
    if _file_identity(named) != retained.identity:
        raise ValueError(
            f"derived view file namespace drift: {retained.subdir}/{retained.name}"
        )


def _iter_verified_stream(
    view: "WikidataDerivedView",
    relative_path: str,
) -> Iterator[bytes]:
    session = _session_state(view)
    retained = _session_committed(session, relative_path)
    os.lseek(retained.fd, 0, os.SEEK_SET)
    buffer = bytearray()
    start = 0
    while True:
        chunk = os.read(retained.fd, _LINE_READ_CHUNK)
        if not chunk:
            break
        buffer.extend(chunk)
        while True:
            newline = buffer.find(b"\n", start)
            if newline < 0:
                break
            row = bytes(buffer[start:newline])
            session._require_open()
            _replay_committed_identity(session, retained)
            yield row
            start = newline + 1
        if start:
            del buffer[:start]
            start = 0
    if start < len(buffer):
        raise ValueError("derived view stream is not newline terminated")


def iter_v2_training_triples(
    view: "WikidataDerivedView",
) -> Iterator["V2TrainingTriple"]:
    for line in _iter_verified_stream(view, _TRAINING_STREAM):
        yield _training_triple_from_row(line)


def iter_v2_aliases(view: "WikidataDerivedView") -> Iterator["V2AliasRecord"]:
    for line in _iter_verified_stream(view, _ALIAS_STREAM):
        yield _alias_record_from_row(line)


def iter_distinct_training_edges(
    view: "WikidataDerivedView",
) -> Iterator["V2TrainingTriple"]:
    for line in _iter_verified_stream(view, _DISTINCT_STREAM):
        yield _training_triple_from_row(line)


def _read_row_at(descriptor: int, offset: int, size_limit: int) -> bytes:
    chunks = bytearray()
    position = offset
    while position < size_limit:
        chunk = _pread(descriptor, _LINE_READ_CHUNK, position)
        if not chunk:
            break
        newline = chunk.find(b"\n")
        if newline >= 0:
            chunks.extend(chunk[:newline])
            return bytes(chunks)
        chunks.extend(chunk)
        position += len(chunk)
    raise ValueError("derived view row is not newline terminated")


def lookup_training_triple(
    view: "WikidataDerivedView",
    training_split: str,
    row: int,
) -> "V2TrainingTriple":
    if training_split not in TRAINING_SPLITS:
        raise ValueError("training split is not in the frozen contract")
    if type(row) is not int or row <= 0:
        raise ValueError("training row must be a positive integer")
    session = _session_state(view)
    receipt = session.receipt
    index_path = _TRAINING_INDEX_BY_SPLIT[training_split]
    index_record = {record.path: record for record in receipt.indexes}[index_path]
    if index_record.record_width != _TRAINING_INDEX_WIDTH:
        raise ValueError("training offset index record width drift")
    if row > index_record.count:
        raise ValueError("training row is out of range")
    stream_record = {record.path: record for record in receipt.streams}[
        _TRAINING_STREAM
    ]
    index_file = _session_committed(session, index_path)
    stream_file = _session_committed(session, _TRAINING_STREAM)
    raw = _pread(
        index_file.fd,
        _TRAINING_INDEX_WIDTH,
        (row - 1) * _TRAINING_INDEX_WIDTH,
    )
    _replay_committed_identity(session, index_file)
    if len(raw) != _TRAINING_INDEX_WIDTH:
        raise ValueError("training offset index is truncated")
    offset = struct.unpack(">Q", raw)[0]
    if offset >= stream_record.bytes:
        raise ValueError("training offset is out of range")
    line = _read_row_at(stream_file.fd, offset, stream_record.bytes)
    _replay_committed_identity(session, stream_file)
    triple = _training_triple_from_row(line)
    if triple.training_split != training_split or triple.row != row:
        raise ValueError("training row does not match the requested key")
    return triple


def _binary_search_alias(
    index_fd: int,
    count: int,
    stream_fd: int,
    stream_size: int,
    key: int,
) -> bytes | None:
    low = 0
    high = count - 1
    while low <= high:
        mid = (low + high) // 2
        raw = _pread(index_fd, _ALIAS_INDEX_WIDTH, mid * _ALIAS_INDEX_WIDTH)
        if len(raw) != _ALIAS_INDEX_WIDTH:
            raise ValueError("alias index record is truncated")
        record_key, offset, length = struct.unpack(">QQQ", raw)
        if record_key == key:
            if length == 0 or offset + length > stream_size:
                raise ValueError("alias index offset/length is out of range")
            row = _pread(stream_fd, length, offset)
            if len(row) != length or not row.endswith(b"\n"):
                raise ValueError("alias row is truncated")
            return row[:-1]
        if record_key < key:
            low = mid + 1
        else:
            high = mid - 1
    return None


def lookup_alias(
    view: "WikidataDerivedView",
    canonical_id: str,
) -> "V2AliasRecord | None":
    key = _alias_index_key(canonical_id)
    session = _session_state(view)
    receipt = session.receipt
    index_record = {record.path: record for record in receipt.indexes}[_ALIAS_INDEX]
    if index_record.record_width != _ALIAS_INDEX_WIDTH:
        raise ValueError("alias index record width drift")
    count = index_record.count
    if count != receipt.alias_rows:
        raise ValueError("alias index count drift")
    stream_record = {record.path: record for record in receipt.streams}[_ALIAS_STREAM]
    index_file = _session_committed(session, _ALIAS_INDEX)
    stream_file = _session_committed(session, _ALIAS_STREAM)
    found = _binary_search_alias(
        index_file.fd,
        count,
        stream_file.fd,
        stream_record.bytes,
        key,
    )
    _replay_committed_identity(session, index_file)
    _replay_committed_identity(session, stream_file)
    if found is None:
        return None
    record = _alias_record_from_row(found)
    if record.canonical_id != canonical_id:
        raise ValueError("alias row does not match the requested key")
    return record
