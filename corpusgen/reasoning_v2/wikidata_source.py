"""Closed receipts and descriptor-pinned Wikidata archive authority."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
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
    entry_lstat,
    list_entries,
    open_directory_at,
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
    indexes: tuple[ArtifactRecord, ...]
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
        _require_exact_paths(self.indexes, INDEX_PATHS, "index inventory")
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
            alias_index.bytes != alias_rows * 24
            or alias_index.bytes % 24 != 0
            or inductive_index.bytes % 8 != 0
            or transductive_index.bytes % 8 != 0
            or (inductive_index.bytes // 8)
            + (transductive_index.bytes // 8)
            != training_rows
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
            inventories[name] = tuple(
                ArtifactRecord.from_dict(record) for record in raw_inventory
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
class WikidataDerivedView:
    root: Path
    receipt_sha256: str
    receipt: WikidataDerivedViewReceipt


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
