"""Closed receipts and descriptor-pinned Wikidata archive authority."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
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
    atomic_rename_noreplace,
    entry_exists,
    entry_lstat,
    fsync_directory,
    list_entries,
    open_directory_at,
    open_directory_path,
    open_parent_directory,
    open_regular_file_at,
    read_regular_file,
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


_VIEW_MINT_KEY = object()


class _ViewAuthority:
    """Proof, minted only by a verified build/verify, of source-rooted authority.

    The object cannot be constructed without the module-private mint key, so a
    caller cannot fabricate a public ``WikidataDerivedView`` and have the read
    entry points accept it. It carries the source-lock and generator-commit
    binding, the receipt content address, and the mint-time file identities so
    bounded workers can reuse authenticated committed bytes without re-hashing.
    """

    __slots__ = (
        "source_lock_sha256",
        "generator_commit",
        "receipt_sha256",
        "receipt",
        "identities",
    )

    def __init__(
        self,
        mint_key: object,
        *,
        source_lock_sha256: str,
        generator_commit: str,
        receipt_sha256: str,
        receipt: "WikidataDerivedViewReceipt",
        identities: Mapping[
            str, tuple[int, int, int, int, int, int, int | None, int | None]
        ],
    ) -> None:
        if mint_key is not _VIEW_MINT_KEY:
            raise ValueError(
                "verified view authority cannot be constructed externally"
            )
        self.source_lock_sha256 = source_lock_sha256
        self.generator_commit = generator_commit
        self.receipt_sha256 = receipt_sha256
        self.receipt = receipt
        self.identities = MappingProxyType(dict(identities))


@dataclass(frozen=True)
class WikidataDerivedView:
    root: Path
    receipt_sha256: str
    receipt: WikidataDerivedViewReceipt
    _authority: object = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path) or not self.root.name:
            raise ValueError("derived view root must be a filesystem path")
        _validate_sha256(self.receipt_sha256, "receipt sha256")
        if not isinstance(self.receipt, WikidataDerivedViewReceipt):
            raise ValueError("derived view receipt has the wrong type")
        if not isinstance(self._authority, _ViewAuthority):
            raise ValueError(
                "WikidataDerivedView must be minted by a verified build or verify"
            )
        if (
            self._authority.receipt_sha256 != self.receipt_sha256
            or self._authority.receipt is not self.receipt
        ):
            raise ValueError("derived view authority does not match the receipt")


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


def _extract_member_payloads(verified: VerifiedArchiveSet) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
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
                        if member.name not in wanted:
                            continue
                        stream = archive.extractfile(member)
                        if stream is None:
                            raise ValueError(
                                f"decoded member is unreadable: "
                                f"{archive_path}:{member.name}"
                            )
                        with stream:
                            payloads[f"members/{member.name}"] = stream.read()
        except (EOFError, OSError, tarfile.TarError) as error:
            raise ValueError(
                f"decoded member extraction failed: {archive_path}"
            ) from error
        finally:
            if duplicate >= 0:
                os.close(duplicate)
    member_by_path = {record.path: record for record in verified.members}
    if set(payloads) != set(member_by_path):
        raise ValueError("decoded member inventory drift")
    for path, data in payloads.items():
        record = member_by_path[path]
        if len(data) != record.bytes or hashlib.sha256(data).hexdigest() != (
            record.sha256
        ):
            raise ValueError(f"decoded member payload drift: {path}")
    return payloads


def _parse_training_triples(text: str, member: str) -> list[tuple[int, str, int]]:
    rows: list[tuple[int, str, int]] = []
    for line_number, raw_line in enumerate(text.split("\n"), 1):
        row = raw_line.rstrip("\r")
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
        rows.append((subject, relation, object_))
    return rows


def _parse_alias_member(
    text: str,
    prefix: str,
    member: str,
) -> dict[str, tuple[str, ...]]:
    parse_id = parse_pid if prefix == "P" else parse_qid
    aliases: dict[str, tuple[str, ...]] = {}
    for line_number, raw_line in enumerate(text.split("\n"), 1):
        row = raw_line.rstrip("\r")
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
        if canonical_id in aliases:
            raise ValueError(
                f"{member}:{line_number}: duplicate canonical ID: {canonical_id}"
            )
        first_seen: dict[str, str] = {}
        for raw_alias in fields[1:]:
            normalized = normalize_alias(raw_alias)
            if not normalized:
                raise ValueError(f"{member}:{line_number}: empty alias")
            first_seen.setdefault(normalized, raw_alias)
        aliases[canonical_id] = tuple(first_seen.values())
    return aliases


def _compute_view_artifacts(
    verified: VerifiedArchiveSet,
    generator_commit: str,
) -> tuple["WikidataDerivedViewReceipt", dict[str, bytes]]:
    payloads = _extract_member_payloads(verified)

    def member_text(name: str) -> str:
        try:
            return payloads[f"members/{name}"].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"decoded member is not UTF-8: {name}") from error

    sealed_edges: set[tuple[int, str, int]] = set()
    for name in SEALED_MEMBERS:
        for subject, relation, object_ in _parse_training_triples(
            member_text(name), name
        ):
            sealed_edges.add((subject, relation, object_))

    training_bytes = bytearray()
    offsets_by_split: dict[str, list[int]] = {
        split: [] for split in TRAINING_SPLITS
    }
    first_provenance: dict[tuple[int, str, int], bytes] = {}
    training_rows = 0
    for training_split in TRAINING_SPLITS:
        member = _TRAINING_MEMBER_BY_SPLIT[training_split]
        for index, (subject, relation, object_) in enumerate(
            _parse_training_triples(member_text(member), member), 1
        ):
            edge = (subject, relation, object_)
            if edge in sealed_edges:
                raise ValueError(
                    "training/sealed edge overlap: "
                    f"{member} Q{subject} {relation} Q{object_}"
                )
            offsets_by_split[training_split].append(len(training_bytes))
            row_bytes = _encode_training_row(
                training_split, index, subject, relation, object_
            )
            training_bytes.extend(row_bytes)
            if edge not in first_provenance:
                first_provenance[edge] = row_bytes
            training_rows += 1

    distinct_bytes = bytearray()
    for edge in sorted(
        first_provenance,
        key=lambda item: (item[0], int(item[1][1:]), item[2]),
    ):
        distinct_bytes.extend(first_provenance[edge])
    distinct_edges = len(first_provenance)

    merged_aliases = dict(
        _parse_alias_member(
            member_text("wikidata5m_entity.txt"),
            "Q",
            "wikidata5m_entity.txt",
        )
    )
    merged_aliases.update(
        _parse_alias_member(
            member_text("wikidata5m_relation.txt"),
            "P",
            "wikidata5m_relation.txt",
        )
    )
    catalog = canonicalize_aliases(merged_aliases)

    alias_entries: list[tuple[tuple[int, int], str, str, tuple[str, ...]]] = []
    for canonical_id in catalog:
        normalized_forms: list[str] = []
        for surface in catalog[canonical_id]:
            form = unicodedata.normalize("NFC", surface)
            if not form or "\x00" in form or form in normalized_forms:
                continue
            normalized_forms.append(form)
        if not normalized_forms:
            continue
        alias_entries.append(
            (
                _alias_sort_key(canonical_id),
                canonical_id,
                normalized_forms[0],
                tuple(normalized_forms),
            )
        )
    alias_entries.sort(key=lambda entry: entry[0])

    aliases_bytes = bytearray()
    alias_index = bytearray()
    for (kind_rank, numeric), canonical_id, display, aliases in alias_entries:
        offset = len(aliases_bytes)
        row_bytes = _encode_alias_row(canonical_id, display, aliases)
        aliases_bytes.extend(row_bytes)
        alias_index.extend(
            struct.pack(
                ">QQQ",
                (kind_rank << 63) | numeric,
                offset,
                len(row_bytes),
            )
        )
    alias_rows = len(alias_entries)

    stream_files = {
        _ALIAS_STREAM: bytes(aliases_bytes),
        _DISTINCT_STREAM: bytes(distinct_bytes),
        _TRAINING_STREAM: bytes(training_bytes),
    }
    index_files = {
        _ALIAS_INDEX: bytes(alias_index),
        "indexes/inductive-training-offsets.bin": b"".join(
            struct.pack(">Q", offset)
            for offset in offsets_by_split["inductive_train"]
        ),
        "indexes/transductive-training-offsets.bin": b"".join(
            struct.pack(">Q", offset)
            for offset in offsets_by_split["transductive_train"]
        ),
    }
    index_meta = {
        _ALIAS_INDEX: (alias_rows, _ALIAS_INDEX_WIDTH),
        "indexes/inductive-training-offsets.bin": (
            len(offsets_by_split["inductive_train"]),
            _TRAINING_INDEX_WIDTH,
        ),
        "indexes/transductive-training-offsets.bin": (
            len(offsets_by_split["transductive_train"]),
            _TRAINING_INDEX_WIDTH,
        ),
    }

    receipt = WikidataDerivedViewReceipt(
        format=RECEIPT_FORMAT,
        schema_version=RECEIPT_SCHEMA_VERSION,
        source_lock_sha256=verified.source_lock_sha256,
        generator_commit=generator_commit,
        archives=verified.archives,
        members=verified.members,
        streams=tuple(_artifact_record(path, stream_files[path]) for path in STREAM_PATHS),
        indexes=tuple(
            _index_artifact_record(
                path,
                byte_count=len(index_files[path]),
                sha256=hashlib.sha256(index_files[path]).hexdigest(),
                count=index_meta[path][0],
                record_width=index_meta[path][1],
            )
            for path in INDEX_PATHS
        ),
        training_rows=training_rows,
        alias_rows=alias_rows,
        distinct_edges=distinct_edges,
        overlap_audit_passed=True,
    )
    files: dict[str, bytes] = {"receipt.json": receipt.to_bytes()}
    files.update(payloads)
    files.update(stream_files)
    files.update(index_files)
    return receipt, files


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


def _write_view_tree(root_fd: int, files: Mapping[str, bytes]) -> None:
    directories = sorted(
        {path.split("/", 1)[0] for path in files if "/" in path}
    )
    directory_fds: dict[str, int] = {}
    try:
        for name in directories:
            descriptor, _created = open_directory_at(root_fd, name, create=True)
            directory_fds[name] = descriptor
        for path in sorted(files):
            if "/" in path:
                parent, name = path.split("/", 1)
                _write_new_file(directory_fds[parent], name, files[path])
            else:
                _write_new_file(root_fd, path, files[path])
        for descriptor in directory_fds.values():
            fsync_directory(descriptor)
    finally:
        for descriptor in directory_fds.values():
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
    """Descriptor-prove the output is not the source root or a descendant.

    Publication deliberately happens after the source postcheck, so an output
    aliasing the immutable source root (or one of its descendants) could write
    into the source namespace after its final verification. Walk the output's
    nearest existing ancestor chain and reject any ancestor whose device/inode
    equals the source root.
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


def _open_output_namespace(output_root: Path) -> tuple[int, int]:
    output_fd = open_directory_path(Path(output_root), create=True)
    try:
        _require_owned_mode(
            os.fstat(output_fd),
            directory=True,
            description="output root",
        )
        namespace_fd, _created = open_directory_at(
            output_fd,
            _VIEW_NAMESPACE,
            create=True,
        )
    except BaseException:
        os.close(output_fd)
        raise
    try:
        _require_owned_mode(
            os.fstat(namespace_fd),
            directory=True,
            description="output wikidata namespace",
        )
    except BaseException:
        os.close(namespace_fd)
        os.close(output_fd)
        raise
    return output_fd, namespace_fd


def _quarantine_and_remove_private(
    namespace_fd: int,
    private_name: str,
    private_identity: tuple[int, int],
) -> None:
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
    try:
        atomic_rename_noreplace(
            namespace_fd,
            private_name,
            namespace_fd,
            quarantine_name,
        )
    except (FileNotFoundError, FileExistsError, OSError):
        return
    fsync_directory(namespace_fd)
    try:
        moved = entry_lstat(namespace_fd, quarantine_name)
    except FileNotFoundError:
        return
    if (
        not stat.S_ISDIR(moved.st_mode)
        or (moved.st_dev, moved.st_ino) != private_identity
    ):
        return
    _remove_view_tree(namespace_fd, quarantine_name)
    fsync_directory(namespace_fd)


def _publish_view(
    output_root: Path,
    content_address: str,
    files: Mapping[str, bytes],
    *,
    commit: str,
) -> Path:
    view_path = Path(output_root) / _VIEW_NAMESPACE / content_address
    output_fd, namespace_fd = _open_output_namespace(Path(output_root))
    try:
        if entry_exists(namespace_fd, content_address):
            # A pre-existing winner is fully authenticated by the caller's mint.
            return view_path
        private_name = f".{_VIEW_NAMESPACE}-view-build-{secrets.token_hex(16)}"
        os.mkdir(private_name, 0o700, dir_fd=namespace_fd)
        created = entry_lstat(namespace_fd, private_name)
        private_identity = (created.st_dev, created.st_ino)
        published = False
        try:
            private_fd, _created = open_directory_at(namespace_fd, private_name)
            try:
                private_meta = os.fstat(private_fd)
                if (private_meta.st_dev, private_meta.st_ino) != private_identity:
                    raise ValueError("private build directory identity drift")
                _require_fixed_mode(
                    private_meta,
                    directory=True,
                    description="private build directory",
                )
                _write_view_tree(private_fd, files)
                fsync_directory(private_fd)
                # Pre-publication re-verification of every derived byte against
                # the committed receipt before it can become a winner.
                _authenticate_view_directory(
                    private_fd,
                    expected_content_address=content_address,
                    expected_commit=commit,
                )
            finally:
                os.close(private_fd)
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
                _quarantine_and_remove_private(
                    namespace_fd,
                    private_name,
                    private_identity,
                )
        return view_path
    finally:
        os.close(namespace_fd)
        os.close(output_fd)


def build_wikidata_derived_view(
    source_lock_path: Path,
    source_root: Path,
    output_root: Path,
    *,
    expected_generator_commit: str,
) -> "WikidataDerivedView":
    commit = _validate_commit(expected_generator_commit, "expected generator commit")
    source_root = Path(source_root)
    output_root = Path(output_root)
    # Descriptor-prove disjointness before writing any output, so a build can
    # never publish into the immutable source root or one of its descendants.
    _assert_output_outside_source(output_root, source_root)
    with _open_verified_archives(Path(source_lock_path), source_root) as verified:
        if verified.generator_commit != commit:
            raise ValueError(
                "expected generator commit does not match the source lock"
            )
        receipt, files = _compute_view_artifacts(verified, commit)
        content_address = hashlib.sha256(files["receipt.json"]).hexdigest()
    # Publish only after the source authority is fully re-verified and closed,
    # so writing the sibling output tree cannot disturb the source namespace.
    view_path = _publish_view(output_root, content_address, files, commit=commit)
    # Mint a non-forgeable verified handle by reopening and fully authenticating
    # the published view against its committed receipt.
    return _mint_view_from_root(
        view_path,
        content_address=content_address,
        commit=commit,
    )


def verify_wikidata_derived_view(
    source_lock_path: Path,
    source_root: Path,
    view_root: Path,
    *,
    expected_generator_commit: str,
) -> "WikidataDerivedView":
    commit = _validate_commit(expected_generator_commit, "expected generator commit")
    with _open_verified_archives(Path(source_lock_path), Path(source_root)) as verified:
        if verified.generator_commit != commit:
            raise ValueError(
                "expected generator commit does not match the source lock"
            )
        receipt, _files = _compute_view_artifacts(verified, commit)
    content_address = hashlib.sha256(receipt.to_bytes()).hexdigest()
    view_path = Path(view_root)
    if view_path.name != content_address:
        raise ValueError("derived view root is not the source content address")
    # Fully authenticate the on-disk view against the source-derived content
    # address; the collision-resistant address binds the on-disk receipt (and
    # thus every committed stream/index/member hash) to the verified source.
    return _mint_view_from_root(
        view_path,
        content_address=content_address,
        commit=commit,
    )


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


def _open_view_subfile(
    view_fd: int,
    relative_path: str,
) -> tuple[int, os.stat_result]:
    parent, _, name = relative_path.rpartition("/")
    if not parent:
        return open_regular_file_at(view_fd, name)
    directory_fd, _created = open_directory_at(view_fd, parent)
    try:
        _require_fixed_mode(
            os.fstat(directory_fd),
            directory=True,
            description=f"derived view {parent}/",
        )
        return open_regular_file_at(directory_fd, name)
    finally:
        os.close(directory_fd)


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


def _authenticate_view_directory(
    view_fd: int,
    *,
    expected_content_address: str,
    expected_commit: str,
) -> tuple[
    "WikidataDerivedViewReceipt",
    dict[str, tuple[int, int, int, int, int, int, int | None, int | None]],
]:
    receipt_metadata = entry_lstat(view_fd, "receipt.json")
    _require_fixed_mode(
        receipt_metadata,
        directory=False,
        description="derived view receipt.json",
    )
    receipt_payload = read_regular_file(view_fd, "receipt.json")
    receipt = WikidataDerivedViewReceipt.from_bytes(
        receipt_payload,
        expected_generator_commit=expected_commit,
    )
    if hashlib.sha256(receipt_payload).hexdigest() != expected_content_address:
        raise ValueError("derived view content address mismatch")

    committed = {
        "members": {record.path: record for record in receipt.members},
        "streams": {record.path: record for record in receipt.streams},
        "indexes": {record.path: record for record in receipt.indexes},
    }
    if set(list_entries(view_fd)) != {
        "receipt.json",
        "members",
        "streams",
        "indexes",
    }:
        raise ValueError("derived view inventory does not match the contract")

    identities: dict[
        str, tuple[int, int, int, int, int, int, int | None, int | None]
    ] = {}
    for subdir, records in committed.items():
        expected_names = {PurePosixPath(path).name for path in records}
        directory_fd, _created = open_directory_at(view_fd, subdir)
        try:
            _require_fixed_mode(
                os.fstat(directory_fd),
                directory=True,
                description=f"derived view {subdir}/",
            )
            if set(list_entries(directory_fd)) != expected_names:
                raise ValueError(
                    f"derived view {subdir} inventory does not match the receipt"
                )
            for path, record in records.items():
                name = PurePosixPath(path).name
                descriptor, metadata = open_regular_file_at(directory_fd, name)
                try:
                    _require_fixed_mode(
                        metadata,
                        directory=False,
                        description=f"derived view {path}",
                    )
                    byte_count, sha256 = _stream_hash_file(descriptor)
                finally:
                    os.close(descriptor)
                if byte_count != record.bytes or sha256 != record.sha256:
                    raise ValueError(
                        f"derived view file does not match the receipt: {path}"
                    )
                identities[path] = _file_identity(metadata)
        finally:
            os.close(directory_fd)
    return receipt, identities


def _mint_view_from_root(
    root: Path,
    *,
    content_address: str,
    commit: str,
) -> "WikidataDerivedView":
    parent_fd, name, view_fd = _open_verified_view_directory(Path(root))
    try:
        if name != content_address:
            raise ValueError("derived view root is not the content address")
        receipt, identities = _authenticate_view_directory(
            view_fd,
            expected_content_address=content_address,
            expected_commit=commit,
        )
    finally:
        os.close(view_fd)
        os.close(parent_fd)
    authority = _ViewAuthority(
        _VIEW_MINT_KEY,
        source_lock_sha256=receipt.source_lock_sha256,
        generator_commit=commit,
        receipt_sha256=content_address,
        receipt=receipt,
        identities=identities,
    )
    return WikidataDerivedView(
        root=Path(root),
        receipt_sha256=content_address,
        receipt=receipt,
        _authority=authority,
    )


def _reopen_verified_view(
    view: "WikidataDerivedView",
) -> tuple[int, "_ViewAuthority"]:
    if not isinstance(view, WikidataDerivedView) or not isinstance(
        view._authority, _ViewAuthority
    ):
        raise ValueError("read requires a verified WikidataDerivedView handle")
    authority = view._authority
    parent_fd, name, view_fd = _open_verified_view_directory(Path(view.root))
    try:
        payload = read_regular_file(view_fd, "receipt.json")
        content_address = hashlib.sha256(payload).hexdigest()
        if (
            content_address != name
            or content_address != view.receipt_sha256
            or content_address != authority.receipt_sha256
            or payload != authority.receipt.to_bytes()
        ):
            raise ValueError("derived view content address mismatch")
    except OSError as error:
        os.close(view_fd)
        os.close(parent_fd)
        raise ValueError("derived view is missing or unsafe") from error
    except BaseException:
        os.close(view_fd)
        os.close(parent_fd)
        raise
    os.close(parent_fd)
    return view_fd, authority


def _open_authenticated_file(
    view_fd: int,
    authority: "_ViewAuthority",
    relative_path: str,
) -> int:
    if relative_path not in authority.identities:
        raise ValueError(f"derived view file is not committed: {relative_path}")
    descriptor, metadata = _open_view_subfile(view_fd, relative_path)
    try:
        _require_fixed_mode(
            metadata,
            directory=False,
            description=f"derived view {relative_path}",
        )
        if _file_identity(metadata) != authority.identities[relative_path]:
            raise ValueError(
                f"derived view file changed since verification: {relative_path}"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _iter_stream_lines_from_fd(descriptor: int) -> Iterator[bytes]:
    buffer = bytearray()
    start = 0
    while True:
        chunk = os.read(descriptor, _LINE_READ_CHUNK)
        if not chunk:
            break
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
        raise ValueError("derived view stream is not newline terminated")


def _iter_verified_stream(
    view: "WikidataDerivedView",
    relative_path: str,
) -> Iterator[bytes]:
    view_fd, authority = _reopen_verified_view(view)
    try:
        stream_fd = _open_authenticated_file(view_fd, authority, relative_path)
        try:
            os.lseek(stream_fd, 0, os.SEEK_SET)
            yield from _iter_stream_lines_from_fd(stream_fd)
        finally:
            os.close(stream_fd)
    finally:
        os.close(view_fd)


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
    view_fd, authority = _reopen_verified_view(view)
    try:
        receipt = authority.receipt
        index_path = _TRAINING_INDEX_BY_SPLIT[training_split]
        index_record = {record.path: record for record in receipt.indexes}[index_path]
        if index_record.record_width != _TRAINING_INDEX_WIDTH:
            raise ValueError("training offset index record width drift")
        if row > index_record.count:
            raise ValueError("training row is out of range")
        stream_record = {record.path: record for record in receipt.streams}[
            _TRAINING_STREAM
        ]
        index_fd = _open_authenticated_file(view_fd, authority, index_path)
        try:
            raw = _pread(
                index_fd,
                _TRAINING_INDEX_WIDTH,
                (row - 1) * _TRAINING_INDEX_WIDTH,
            )
            if len(raw) != _TRAINING_INDEX_WIDTH:
                raise ValueError("training offset index is truncated")
            offset = struct.unpack(">Q", raw)[0]
        finally:
            os.close(index_fd)
        stream_fd = _open_authenticated_file(view_fd, authority, _TRAINING_STREAM)
        try:
            if offset >= stream_record.bytes:
                raise ValueError("training offset is out of range")
            line = _read_row_at(stream_fd, offset, stream_record.bytes)
        finally:
            os.close(stream_fd)
    finally:
        os.close(view_fd)
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
    view_fd, authority = _reopen_verified_view(view)
    try:
        receipt = authority.receipt
        index_record = {record.path: record for record in receipt.indexes}[
            _ALIAS_INDEX
        ]
        if index_record.record_width != _ALIAS_INDEX_WIDTH:
            raise ValueError("alias index record width drift")
        count = index_record.count
        if count != receipt.alias_rows:
            raise ValueError("alias index count drift")
        stream_record = {record.path: record for record in receipt.streams}[
            _ALIAS_STREAM
        ]
        index_fd = _open_authenticated_file(view_fd, authority, _ALIAS_INDEX)
        try:
            stream_fd = _open_authenticated_file(view_fd, authority, _ALIAS_STREAM)
            try:
                found = _binary_search_alias(
                    index_fd,
                    count,
                    stream_fd,
                    stream_record.bytes,
                    key,
                )
            finally:
                os.close(stream_fd)
        finally:
            os.close(index_fd)
    finally:
        os.close(view_fd)
    if found is None:
        return None
    record = _alias_record_from_row(found)
    if record.canonical_id != canonical_id:
        raise ValueError("alias row does not match the requested key")
    return record
