"""External-memory routing and exact semantic sidecars for reasoning-v2."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import secrets
import sqlite3
import stat
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import zip_longest
from pathlib import Path
from typing import Literal, NoReturn, Self, cast
from urllib.parse import quote

from corpusgen.parallel.canonical import canonical_json_bytes, sha256_hex
from corpusgen.parallel.safeio import (
    atomic_rename_noreplace,
    entry_lstat,
    fsync_directory,
    list_entries,
    open_directory_at,
    open_parent_directory,
    open_regular_file_at,
)
from corpusgen.reasoning import (
    METADATA_SCOPE,
    ROUTE_POLICY,
    TRAIN_ONLY_FEATURES,
    FactMetadata,
    LeakageReport,
    RouteDecision,
    SemanticFact,
    SemanticOccurrence,
    SupervisedField,
    audit_occurrence_closure,
    minimally_rounded_quota,
    plan_occurrence_closure,
    route_score,
)
from corpusgen.reasoning import (
    SemanticLeakageError as _ClosureSemanticLeakageError,
)
from corpusgen.reasoning_v2 import catalog as _catalog_module
from corpusgen.reasoning_v2.catalog import (
    CatalogRecord,
    InputCatalog,
    SemanticFactRow,
)

_ROUTE_MANIFEST_FORMAT = "memorysplit-production-route-manifest-v1"
_ROUTE_DOSE_FORMAT = "memorysplit-production-route-dose-report-v1"
_ROUTE_QUARANTINE_DIRECTORY = ".memorysplit-route-quarantine-v1"
_ROUTE_REDUCER_NAME = ".route-reducer.sqlite3"
_ROUTE_INDEX_NAME = "route-index.sqlite3"
_ROUTE_MANIFEST_NAME = "route-manifest.jsonl"
_DOSE_REPORT_NAME = "dose-report.json"
_DECISION_STREAM_NAME = "route-decisions.jsonl"
_SPLIT = "Split90"
_TARGET_FRACTION = Fraction(9, 10)
_EXTERNAL_SORT_CHUNK_ROWS = 50_000
_EXTERNAL_SORT_MERGE_FAN_IN = 32
_SQLITE_OPEN_LOCK = _catalog_module._SQLITE_OPEN_LOCK
_SEMANTIC_ROLES = frozenset(
    {
        "plain_text",
        "factual_payload",
        "rule",
        "operator",
        "schema",
        "proof",
        "answer_state",
    }
)
_FACT_VALUE_FIELDS = frozenset(
    {
        "expected_hops",
        "expected_reads",
        "fact_id",
        "payload_entropy_bits",
        "record_type",
        "scheduled_exposures",
        "source",
        "surfaces",
    }
)
_FACT_IDENTITY_FIELDS = _FACT_VALUE_FIELDS - {"scheduled_exposures"}


SemanticRole = Literal[
    "plain_text",
    "factual_payload",
    "rule",
    "operator",
    "schema",
    "proof",
    "answer_state",
]


def _route_index_open_hook(
    phase: str,
    parent_fd: int,
    name: str,
    pinned_fd: int,
) -> None:
    del phase, parent_fd, name, pinned_fd


def _route_index_read_hook(
    phase: str,
    parent_fd: int,
    name: str,
    pinned_fd: int,
) -> None:
    del phase, parent_fd, name, pinned_fd


def _artifact_read_hook(
    phase: str,
    directory_fd: int,
    name: str,
    pinned_fd: int,
) -> None:
    del phase, directory_fd, name, pinned_fd


def _strict_text(
    value: object,
    description: str,
    *,
    nonempty: bool = True,
) -> str:
    if type(value) is not str or (nonempty and not value):
        raise ValueError(f"{description} must be a canonical string")
    result = cast(str, value)
    if "\x00" in result or unicodedata.normalize("NFC", result) != result:
        raise ValueError(f"{description} must be NFC-normalized")
    return result


def _strict_fraction(
    value: object,
    description: str,
    *,
    nonnegative: bool = True,
) -> Fraction:
    if type(value) is not Fraction:
        raise ValueError(f"{description} must be an exact finite Fraction")
    result = cast(Fraction, value)
    if nonnegative and result < 0:
        raise ValueError(f"{description} must be non-negative")
    return result


def _fraction_dict(value: Fraction) -> dict[str, int]:
    return {
        "denominator": value.denominator,
        "numerator": value.numerator,
    }


def _fraction_from_dict(value: object, description: str) -> Fraction:
    if type(value) is not dict or set(value) != {"denominator", "numerator"}:
        raise ValueError(f"{description} must be an exact rational")
    denominator = value["denominator"]
    numerator = value["numerator"]
    if (
        type(denominator) is not int
        or type(numerator) is not int
        or denominator <= 0
    ):
        raise ValueError(f"{description} must be an exact rational")
    result = Fraction(numerator, denominator)
    if _fraction_dict(result) != value:
        raise ValueError(f"{description} rational is not canonical")
    return result


def _strict_json_bytes(payload: bytes, description: str) -> object:
    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"{description} contains non-finite value {value}")

    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{description} contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not strict UTF-8 JSON") from error
    return value


def _byte_key(value: str) -> bytes:
    return value.encode("utf-8")


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_size,
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


def _directory_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        stat.S_IMODE(metadata.st_mode),
    )


def _require_owned_regular(metadata: os.stat_result, description: str) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise ValueError(f"{description} is not an owner-controlled regular file")


def _require_owned_directory(metadata: os.stat_result, description: str) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise ValueError(f"{description} is not an owner-controlled directory")


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while materializing route artifact")
        view = view[written:]


def _open_new_regular(directory_fd: int, name: str) -> int:
    return os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )


def _read_regular_at(
    directory_fd: int,
    name: str,
    description: str,
) -> bytes:
    named = entry_lstat(directory_fd, name)
    _require_owned_regular(named, description)
    descriptor, opened = open_regular_file_at(directory_fd, name)
    chunks: list[bytes] = []
    try:
        _require_owned_regular(opened, description)
        if _file_identity(named) != _file_identity(opened):
            raise ValueError(f"{description} identity changed before reading")
        _artifact_read_hook("after_open", directory_fd, name, descriptor)
        if _file_identity(entry_lstat(directory_fd, name)) != _file_identity(opened):
            raise ValueError(f"{description} identity changed before reading")
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        _artifact_read_hook("after_read", directory_fd, name, descriptor)
        after = os.fstat(descriptor)
        named_after = entry_lstat(directory_fd, name)
        if (
            _file_identity(after) != _file_identity(opened)
            or _file_identity(named_after) != _file_identity(opened)
            or os.read(descriptor, 1)
        ):
            raise ValueError(f"{description} identity changed while reading")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if len(payload) != opened.st_size:
        raise ValueError(f"{description} length changed while reading")
    return payload


def _regular_commitment_at(
    directory_fd: int,
    name: str,
    *,
    description: str,
) -> tuple[int, str]:
    named = entry_lstat(directory_fd, name)
    _require_owned_regular(named, description)
    descriptor, opened = open_regular_file_at(directory_fd, name)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        _require_owned_regular(opened, description)
        if _file_identity(named) != _file_identity(opened):
            raise ValueError(f"{description} identity changed before verification")
        _artifact_read_hook("after_open", directory_fd, name, descriptor)
        if _file_identity(entry_lstat(directory_fd, name)) != _file_identity(opened):
            raise ValueError(f"{description} identity changed before verification")
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        _artifact_read_hook("after_read", directory_fd, name, descriptor)
        after = os.fstat(descriptor)
        named_after = entry_lstat(directory_fd, name)
        if (
            _file_identity(after) != _file_identity(opened)
            or _file_identity(named_after) != _file_identity(opened)
            or os.read(descriptor, 1)
        ):
            raise ValueError(f"{description} identity changed while verifying")
    finally:
        os.close(descriptor)
    return byte_count, digest.hexdigest()


def _verify_regular_at(
    directory_fd: int,
    name: str,
    *,
    expected_bytes: int,
    expected_sha256: str,
    description: str,
) -> None:
    byte_count, digest = _regular_commitment_at(
        directory_fd,
        name,
        description=description,
    )
    if byte_count != expected_bytes or digest != expected_sha256:
        raise ValueError(f"{description} content differs")


def _copy_regular_at(
    source_directory_fd: int,
    source_name: str,
    destination_fd: int,
    digest: hashlib._Hash,
) -> int:
    named = entry_lstat(source_directory_fd, source_name)
    _require_owned_regular(named, "route decision stream")
    descriptor, opened = open_regular_file_at(source_directory_fd, source_name)
    byte_count = 0
    try:
        _require_owned_regular(opened, "route decision stream")
        if _file_identity(named) != _file_identity(opened):
            raise ValueError("route decision stream identity changed before copy")
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            _write_all(destination_fd, chunk)
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
        named_after = entry_lstat(source_directory_fd, source_name)
        if (
            _file_identity(after) != _file_identity(opened)
            or _file_identity(named_after) != _file_identity(opened)
            or os.read(descriptor, 1)
        ):
            raise ValueError("route decision stream identity changed while copying")
    finally:
        os.close(descriptor)
    if byte_count != opened.st_size:
        raise ValueError("route decision stream length changed while copying")
    return byte_count


def _new_stage(
    parent_fd: int,
    final_name: str,
) -> tuple[str, int, tuple[int, int, int, int]]:
    for _attempt in range(16):
        stage_name = (
            f".{final_name}.tmp-{os.getpid()}-{secrets.token_hex(12)}"
        )
        try:
            os.mkdir(stage_name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        fsync_directory(parent_fd)
        stage_fd, _created = open_directory_at(parent_fd, stage_name)
        metadata = os.fstat(stage_fd)
        named = entry_lstat(parent_fd, stage_name)
        _require_owned_directory(metadata, "private route stage")
        identity = _directory_identity(metadata)
        if _directory_identity(named) != identity:
            os.close(stage_fd)
            raise ValueError("private route stage identity changed after creation")
        return stage_name, stage_fd, identity
    raise FileExistsError("could not allocate private route stage")


def _open_route_quarantine(
    parent_fd: int,
) -> tuple[int, tuple[int, int, int, int]]:
    quarantine_fd, _created = open_directory_at(
        parent_fd,
        _ROUTE_QUARANTINE_DIRECTORY,
        create=True,
        mode=0o700,
    )
    metadata = os.fstat(quarantine_fd)
    named = entry_lstat(parent_fd, _ROUTE_QUARANTINE_DIRECTORY)
    _require_owned_directory(metadata, "route quarantine directory")
    identity = _directory_identity(metadata)
    if _directory_identity(named) != identity:
        os.close(quarantine_fd)
        raise ValueError("route quarantine directory identity changed")
    return quarantine_fd, identity


def _quarantine_stage(
    parent_fd: int,
    stage_name: str,
    stage_fd: int,
    stage_identity: tuple[int, int, int, int],
    quarantine_fd: int,
    label: str,
) -> str:
    opened = os.fstat(stage_fd)
    named = entry_lstat(parent_fd, stage_name)
    if (
        _directory_identity(opened) != stage_identity
        or _directory_identity(named) != stage_identity
    ):
        raise ValueError(f"{label} route stage quarantine identity changed")
    quarantine_name = ""
    for _attempt in range(16):
        candidate = f"{label}-{secrets.token_hex(16)}"
        try:
            atomic_rename_noreplace(
                parent_fd,
                stage_name,
                quarantine_fd,
                candidate,
            )
        except FileExistsError:
            continue
        quarantine_name = candidate
        break
    if not quarantine_name:
        raise FileExistsError(f"could not allocate {label} route quarantine")
    fsync_directory(parent_fd)
    fsync_directory(quarantine_fd)
    quarantined_fd, _created = open_directory_at(
        quarantine_fd,
        quarantine_name,
    )
    try:
        quarantined = os.fstat(quarantined_fd)
        named_quarantined = entry_lstat(quarantine_fd, quarantine_name)
        pinned_after = os.fstat(stage_fd)
        if (
            _directory_identity(quarantined) != stage_identity
            or _directory_identity(named_quarantined) != stage_identity
            or _directory_identity(pinned_after) != stage_identity
        ):
            raise ValueError(f"quarantined {label} route stage identity changed")
    finally:
        os.close(quarantined_fd)
    return quarantine_name


def _connect_sqlite_at(
    directory_fd: int,
    name: str,
    pinned_fd: int,
    *,
    mode: Literal["ro", "rw"],
    hook: Callable[[str, int, str, int], None] | None = None,
) -> sqlite3.Connection:
    callback = hook or (lambda phase, parent, item, descriptor: None)
    connection: sqlite3.Connection | None = None
    try:
        with _SQLITE_OPEN_LOCK:
            callback("before_sqlite_open", directory_fd, name, pinned_fd)
            callback("before_cwd_snapshot", directory_fd, name, pinned_fd)
            cwd_fd = os.open(
                ".",
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0),
            )
            cwd_changed = False
            try:
                callback("after_cwd_snapshot", directory_fd, name, pinned_fd)
                os.fchdir(directory_fd)
                cwd_changed = True
                callback("after_stage_fchdir", directory_fd, name, pinned_fd)
                connection = sqlite3.connect(
                    f"file:{quote(name, safe='')}?mode={mode}",
                    uri=True,
                )
                callback("sqlite_opened", directory_fd, name, pinned_fd)
            finally:
                try:
                    if cwd_changed:
                        os.fchdir(cwd_fd)
                finally:
                    os.close(cwd_fd)
                if cwd_changed:
                    callback("cwd_restored", directory_fd, name, pinned_fd)
            callback("after_sqlite_open", directory_fd, name, pinned_fd)
    except BaseException:
        if connection is not None:
            connection.close()
        raise
    if connection is None:
        raise ValueError(f"SQLite database failed to open: {name}")
    return connection


@dataclass
class _PinnedDatabase:
    connection: sqlite3.Connection | None
    descriptor: int
    identity: tuple[int, int, int, int, int]
    name: str

    def close_connection(self, directory_fd: int, description: str) -> None:
        if self.connection is None:
            return
        quick_check = self.connection.execute("PRAGMA quick_check").fetchone()
        if quick_check != ("ok",):
            raise ValueError(f"{description} integrity check failed")
        self.connection.commit()
        self.connection.close()
        self.connection = None
        os.fsync(self.descriptor)
        pinned = os.fstat(self.descriptor)
        named = entry_lstat(directory_fd, self.name)
        if (
            _regular_inode_identity(pinned) != self.identity
            or _regular_inode_identity(named) != self.identity
        ):
            raise ValueError(f"{description} identity changed after use")

    def abort_connection(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def close_descriptor(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


def _create_database(
    directory_fd: int,
    name: str,
    description: str,
) -> _PinnedDatabase:
    descriptor = os.open(
        name,
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    connection: sqlite3.Connection | None = None
    try:
        pinned = os.fstat(descriptor)
        named = entry_lstat(directory_fd, name)
        _require_owned_regular(pinned, description)
        identity = _regular_inode_identity(pinned)
        if _regular_inode_identity(named) != identity:
            raise ValueError(f"{description} identity changed before SQLite open")
        connection = _connect_sqlite_at(
            directory_fd,
            name,
            descriptor,
            mode="rw",
        )
        named_after = entry_lstat(directory_fd, name)
        if (
            _regular_inode_identity(os.fstat(descriptor)) != identity
            or _regular_inode_identity(named_after) != identity
        ):
            raise ValueError(f"{description} identity changed during SQLite open")
    except BaseException:
        if connection is not None:
            connection.close()
        os.close(descriptor)
        raise
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    return _PinnedDatabase(
        connection=connection,
        descriptor=descriptor,
        identity=identity,
        name=name,
    )


class SemanticLeakageError(_ClosureSemanticLeakageError):
    """Fail-closed semantic error with token- or text-level details."""

    def __init__(
        self,
        message: str,
        *,
        routed_fact_ids: Sequence[str] = (),
        occurrences: Sequence[SemanticOccurrence] = (),
        metadata_errors: Sequence[str] = (),
        leaks: Sequence[dict[str, object]] = (),
    ) -> None:
        occurrence_rows = tuple(occurrences)
        errors = tuple(metadata_errors) or (message,)
        report = LeakageReport(
            routed_fact_ids=tuple(sorted(set(routed_fact_ids))),
            supervised_occurrences=len(occurrence_rows),
            masked_occurrences=0,
            unmasked_occurrences=occurrence_rows,
            metadata_errors=errors,
        )
        self.leaks = tuple(leaks)
        self.detail = message
        super().__init__(report)
        self.args = (f"{message}: {self.args[0]}",)


@dataclass(frozen=True)
class TokenSemanticSpan:
    token_start: int
    token_end: int
    fact_id: str | None
    role: SemanticRole

    def __post_init__(self) -> None:
        if (
            type(self.token_start) is not int
            or type(self.token_end) is not int
            or self.token_start < 0
            or self.token_end <= self.token_start
        ):
            raise ValueError("token semantic span bounds must be positive integers")
        if self.fact_id is not None:
            _strict_text(self.fact_id, "token semantic fact ID")
        if type(self.role) is not str or self.role not in _SEMANTIC_ROLES:
            raise ValueError("token semantic role is not recognized")
        if self.role == "factual_payload" and self.fact_id is None:
            raise ValueError("factual payload spans require a fact ID")


@dataclass(frozen=True)
class SidecarWeights:
    dense: bytes
    split90: bytes
    routed_payload_targets: int
    leaks: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        if type(self.dense) is not bytes or type(self.split90) is not bytes:
            raise TypeError("sidecar weights must be bytes")
        if len(self.dense) != len(self.split90):
            raise ValueError("Dense and Split90 sidecars must have equal length")
        if any(value != 1 for value in self.dense):
            raise ValueError("Dense sidecar must contain only supervised targets")
        if any(value not in (0, 1) for value in self.split90):
            raise ValueError("Split90 sidecar must be binary")
        if (
            type(self.routed_payload_targets) is not int
            or self.routed_payload_targets < 0
            or self.routed_payload_targets != self.split90.count(0)
        ):
            raise ValueError("routed payload target count disagrees with sidecar")
        if type(self.leaks) is not tuple:
            raise TypeError("semantic leaks must be a tuple")


@dataclass(frozen=True)
class RouteArtifacts:
    manifest_path: Path
    index_path: Path
    dose_report_path: Path
    manifest_sha256: str
    dose_report_sha256: str
    external_fact_count: int
    distinct_fact_fraction: Fraction
    information_burden_fraction: Fraction
    dose_report: dict[str, object]

    def open_index(self) -> RouteIndex:
        return RouteIndex.open(self.index_path)


def _validate_route_index_schema(connection: sqlite3.Connection) -> None:
    objects = tuple(
        connection.execute(
            "SELECT type, name FROM sqlite_schema "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        )
    )
    if objects != (("table", "selected"),):
        raise ValueError("route index schema is not canonical")
    columns = tuple(connection.execute("PRAGMA table_info(selected)"))
    if (
        len(columns) != 1
        or columns[0][1] != "fact_id"
        or str(columns[0][2]).upper() != "TEXT"
        or columns[0][3] != 1
        or columns[0][5] != 1
    ):
        raise ValueError("route index selected table is not canonical")
    if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
        raise ValueError("route index integrity check failed")


class RouteIndex:
    @classmethod
    def open(cls, database_path: Path) -> Self:
        if not isinstance(database_path, Path):
            raise TypeError("route index path must be a pathlib.Path")
        parent_fd = -1
        try:
            parent_fd, name = open_parent_directory(database_path)
            return cls._open_at(parent_fd, name, database_path)
        except OSError as error:
            if parent_fd >= 0:
                os.close(parent_fd)
            raise ValueError("route index is missing or unsafe") from error
        except BaseException:
            if parent_fd >= 0:
                os.close(parent_fd)
            raise

    @classmethod
    def _open_at(
        cls,
        parent_fd: int,
        name: str,
        database_path: Path,
    ) -> Self:
        descriptor = -1
        connection: sqlite3.Connection | None = None
        try:
            named = entry_lstat(parent_fd, name)
            _require_owned_regular(named, "route index")
            descriptor, opened = open_regular_file_at(parent_fd, name)
            _require_owned_regular(opened, "route index")
            identity = _file_identity(opened)
            if _file_identity(named) != identity:
                raise ValueError("route index identity changed before open")
            connection = _connect_sqlite_at(
                parent_fd,
                name,
                descriptor,
                mode="ro",
                hook=_route_index_open_hook,
            )
            named_after = entry_lstat(parent_fd, name)
            if (
                _file_identity(os.fstat(descriptor)) != identity
                or _file_identity(named_after) != identity
            ):
                raise ValueError("route index identity changed during open")
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            _validate_route_index_schema(connection)
            _route_index_open_hook(
                "after_schema_validation",
                parent_fd,
                name,
                descriptor,
            )
            if (
                _file_identity(os.fstat(descriptor)) != identity
                or _file_identity(entry_lstat(parent_fd, name)) != identity
            ):
                raise ValueError(
                    "route index identity changed during schema validation"
                )
            result = cls.__new__(cls)
            result.database_path = database_path
            result._parent_fd = parent_fd
            result._name = name
            result._descriptor = descriptor
            result._identity = identity
            result._connection = connection
            result._closed = False
            result._unsafe = False
            return result
        except BaseException:
            if connection is not None:
                connection.close()
            if descriptor >= 0:
                os.close(descriptor)
            raise

    def __init__(self, database_path: Path) -> None:
        opened = self.open(database_path)
        self.__dict__.update(opened.__dict__)
        opened._parent_fd = -1
        opened._descriptor = -1
        opened._closed = True

    def _replay_identity(self, context: str) -> None:
        if self._closed:
            raise ValueError("route index is closed")
        try:
            pinned = os.fstat(self._descriptor)
            named = entry_lstat(self._parent_fd, self._name)
            _require_owned_regular(pinned, "route index")
            _require_owned_regular(named, "route index")
            if (
                _file_identity(pinned) != self._identity
                or _file_identity(named) != self._identity
            ):
                raise ValueError(f"route index identity changed {context}")
        except BaseException:
            self._unsafe = True
            raise

    def _before_query(self) -> None:
        self._replay_identity("before query")
        _route_index_read_hook(
            "before_query",
            self._parent_fd,
            self._name,
            self._descriptor,
        )
        self._replay_identity("before query")

    def _after_query(self) -> None:
        _route_index_read_hook(
            "after_query",
            self._parent_fd,
            self._name,
            self._descriptor,
        )
        self._replay_identity("after query")

    def is_external(self, fact_id: str) -> bool:
        fact_id = _strict_text(fact_id, "route lookup fact ID")
        self._before_query()
        row = self._connection.execute(
            "SELECT 1 FROM selected WHERE fact_id = ?",
            (fact_id,),
        ).fetchone()
        self._after_query()
        return row is not None

    def iter_external_fact_ids(self) -> Iterator[str]:
        def iterator() -> Iterator[str]:
            self._before_query()
            cursor = self._connection.execute(
                "SELECT fact_id FROM selected ORDER BY fact_id COLLATE BINARY"
            )
            try:
                for row in cursor:
                    if type(row[0]) is not str:
                        raise ValueError("route index fact ID is not a string")
                    yield _strict_text(row[0], "route index fact ID")
            finally:
                cursor.close()
                if not self._unsafe:
                    self._after_query()

        return iterator()

    def close(self) -> None:
        if self._closed:
            return
        identity_error: Exception | None = None
        if not self._unsafe:
            try:
                self._replay_identity("before close")
            except (OSError, ValueError) as error:
                identity_error = error
        self._connection.close()
        if not self._unsafe:
            try:
                self._replay_identity("after close")
            except (OSError, ValueError) as error:
                identity_error = identity_error or error
        os.close(self._descriptor)
        os.close(self._parent_fd)
        self._descriptor = -1
        self._parent_fd = -1
        self._closed = True
        if identity_error is not None:
            raise identity_error

    def __enter__(self) -> Self:
        if self._closed:
            raise ValueError("route index is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
            return
        try:
            self.close()
        except Exception:  # noqa: BLE001 - never mask the active exception
            return


def _validate_semantic_fact_row(fact: object) -> SemanticFactRow:
    if not isinstance(fact, SemanticFactRow):
        raise ValueError(  # noqa: TRY004 - malformed artifact data
            "route semantic facts must contain SemanticFactRow values"
        )
    _strict_text(fact.fact_id, "route semantic fact ID")
    _strict_text(fact.source, "route semantic fact source")
    _strict_text(fact.record_type, "route semantic fact record type")
    _strict_fraction(fact.payload_entropy_bits, "route semantic payload entropy")
    _strict_fraction(fact.expected_reads, "route semantic expected reads")
    _strict_fraction(fact.expected_hops, "route semantic expected hops")
    if type(fact.scheduled_exposures) is not int or fact.scheduled_exposures <= 0:
        raise ValueError(
            "route semantic scheduled exposures must be a positive integer"
        )
    if type(fact.surfaces) is not tuple or not fact.surfaces:
        raise ValueError("route semantic fact surfaces must be a nonempty tuple")
    for surface in fact.surfaces:
        _strict_text(surface, "route semantic fact surface")
    if len(fact.surfaces) != len(set(fact.surfaces)):
        raise ValueError("route semantic fact surfaces must be distinct")
    if fact.surfaces != tuple(sorted(fact.surfaces, key=_byte_key)):
        raise ValueError("route semantic fact surfaces must use canonical order")
    return fact


def _fact_identity_value(fact: SemanticFactRow) -> dict[str, object]:
    return {
        "expected_hops": _fraction_dict(fact.expected_hops),
        "expected_reads": _fraction_dict(fact.expected_reads),
        "fact_id": fact.fact_id,
        "payload_entropy_bits": _fraction_dict(fact.payload_entropy_bits),
        "record_type": fact.record_type,
        "source": fact.source,
        "surfaces": list(fact.surfaces),
    }


def _fact_value(fact: FactMetadata) -> dict[str, object]:
    return {
        "expected_hops": _fraction_dict(fact.expected_hops),
        "expected_reads": _fraction_dict(fact.expected_reads),
        "fact_id": fact.fact_id,
        "payload_entropy_bits": _fraction_dict(fact.payload_entropy_bits),
        "record_type": fact.record_type,
        "scheduled_exposures": fact.scheduled_exposures,
        "source": fact.source,
        "surfaces": list(fact.surfaces),
    }


def _fact_from_value(value: object, description: str) -> FactMetadata:
    if type(value) is not dict or set(value) != _FACT_VALUE_FIELDS:
        raise ValueError(f"{description} has invalid route fact fields")
    fact_id = _strict_text(value["fact_id"], f"{description} fact ID")
    source = _strict_text(value["source"], f"{description} source")
    record_type = _strict_text(
        value["record_type"],
        f"{description} record type",
    )
    entropy = _fraction_from_dict(
        value["payload_entropy_bits"],
        f"{description} payload entropy",
    )
    expected_reads = _fraction_from_dict(
        value["expected_reads"],
        f"{description} expected reads",
    )
    expected_hops = _fraction_from_dict(
        value["expected_hops"],
        f"{description} expected hops",
    )
    if min(entropy, expected_reads, expected_hops) < 0:
        raise ValueError(f"{description} route fractions must be non-negative")
    exposures = value["scheduled_exposures"]
    if type(exposures) is not int or exposures <= 0:
        raise ValueError(f"{description} scheduled exposures must be positive")
    surfaces_value = value["surfaces"]
    if type(surfaces_value) is not list or not surfaces_value:
        raise ValueError(f"{description} surfaces must be a nonempty list")
    surfaces = tuple(
        _strict_text(surface, f"{description} surface")
        for surface in surfaces_value
    )
    if (
        len(surfaces) != len(set(surfaces))
        or surfaces != tuple(sorted(surfaces, key=_byte_key))
    ):
        raise ValueError(f"{description} surfaces are not canonical")
    return FactMetadata(
        fact_id=fact_id,
        source=source,
        record_type=record_type,
        payload_entropy_bits=entropy,
        scheduled_exposures=exposures,
        expected_reads=expected_reads,
        expected_hops=expected_hops,
        surfaces=surfaces,
    )


def _fact_from_reduced(
    metadata: object,
    scheduled_exposures: object,
) -> FactMetadata:
    if type(metadata) is not bytes:
        raise ValueError("route reducer metadata must be bytes")
    if canonical_json_bytes(_strict_json_bytes(metadata, "route reducer metadata")) != (
        metadata
    ):
        raise ValueError("route reducer metadata is not canonical")
    value = _strict_json_bytes(metadata, "route reducer metadata")
    if type(value) is not dict or set(value) != _FACT_IDENTITY_FIELDS:
        raise ValueError("route reducer metadata fields are invalid")
    if (
        type(scheduled_exposures) is not str
        or not scheduled_exposures.isascii()
        or not scheduled_exposures.isdigit()
        or str(int(scheduled_exposures)) != scheduled_exposures
        or int(scheduled_exposures) <= 0
    ):
        raise ValueError("route reducer exposure aggregate is invalid")
    return _fact_from_value(
        {
            **value,
            "scheduled_exposures": int(scheduled_exposures),
        },
        "route reducer row",
    )


def _create_reducer(directory_fd: int) -> _PinnedDatabase:
    database = _create_database(
        directory_fd,
        _ROUTE_REDUCER_NAME,
        "route reducer database",
    )
    assert database.connection is not None
    database.connection.execute(
        "CREATE TABLE facts ("
        "fact_id BLOB NOT NULL PRIMARY KEY, "
        "metadata BLOB NOT NULL, "
        "scheduled_exposures TEXT NOT NULL"
        ") WITHOUT ROWID"
    )
    return database


def _create_route_index(directory_fd: int) -> _PinnedDatabase:
    database = _create_database(
        directory_fd,
        _ROUTE_INDEX_NAME,
        "route index database",
    )
    assert database.connection is not None
    database.connection.execute(
        "CREATE TABLE selected ("
        "fact_id TEXT NOT NULL PRIMARY KEY"
        ") WITHOUT ROWID"
    )
    return database


def _reduce_catalog(
    catalog: InputCatalog,
    connection: sqlite3.Connection,
) -> int:
    iterator_factory = getattr(catalog, "iter_records", None)
    if not callable(iterator_factory):
        raise TypeError("catalog must provide iter_records()")
    for record in iterator_factory():
        if not isinstance(record, CatalogRecord):
            raise ValueError(  # noqa: TRY004 - malformed artifact data
                "route catalog must yield CatalogRecord rows"
            )
        if type(record.semantic_facts) is not tuple:
            raise ValueError("route catalog semantic facts must be a tuple")
        fact_ids: list[str] = []
        for raw_fact in record.semantic_facts:
            fact = _validate_semantic_fact_row(raw_fact)
            fact_ids.append(fact.fact_id)
            metadata = canonical_json_bytes(_fact_identity_value(fact))
            key = _byte_key(fact.fact_id)
            existing = connection.execute(
                "SELECT metadata, scheduled_exposures "
                "FROM facts WHERE fact_id = ?",
                (key,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO facts("
                    "fact_id, metadata, scheduled_exposures"
                    ") VALUES (?, ?, ?)",
                    (key, metadata, str(fact.scheduled_exposures)),
                )
                continue
            if type(existing[0]) is not bytes or existing[0] != metadata:
                raise ValueError(
                    "conflicting metadata for repeated fact: "
                    f"{fact.fact_id}"
                )
            previous = existing[1]
            if type(previous) is not str or not previous.isdigit():
                raise ValueError("route reducer exposure aggregate is invalid")
            total = int(previous) + fact.scheduled_exposures
            connection.execute(
                "UPDATE facts SET scheduled_exposures = ? WHERE fact_id = ?",
                (str(total), key),
            )
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("route CatalogRecord contains duplicate semantic facts")
        if fact_ids != sorted(fact_ids, key=_byte_key):
            raise ValueError("route CatalogRecord semantic facts are not canonical")
    connection.commit()
    count = connection.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    if type(count) is not int or count <= 0:
        raise ValueError("routing requires at least one semantic fact")
    return count


def _iter_reduced_facts(
    connection: sqlite3.Connection,
) -> Iterator[FactMetadata]:
    rows = connection.execute(
        "SELECT metadata, scheduled_exposures "
        "FROM facts ORDER BY fact_id"
    )
    for metadata, scheduled_exposures in rows:
        yield _fact_from_reduced(metadata, scheduled_exposures)


def _route_row_bytes(fact: FactMetadata) -> bytes:
    return canonical_json_bytes(_fact_value(fact))


def _iter_route_run(directory_fd: int, name: str) -> Iterator[FactMetadata]:
    named = entry_lstat(directory_fd, name)
    _require_owned_regular(named, f"external route run {name}")
    descriptor, opened = open_regular_file_at(directory_fd, name)
    _require_owned_regular(opened, f"external route run {name}")
    if _file_identity(named) != _file_identity(opened):
        os.close(descriptor)
        raise ValueError(f"external route run identity changed: {name}")
    try:
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            for raw_line in handle:
                if not raw_line.endswith(b"\n"):
                    raise ValueError(
                        f"external route run is truncated: {name}"
                    )
                value = _strict_json_bytes(raw_line, f"external route run {name}")
                if canonical_json_bytes(value) != raw_line:
                    raise ValueError(
                        f"external route run is not canonical: {name}"
                    )
                yield _fact_from_value(value, f"external route run {name}")
    finally:
        after = os.fstat(descriptor)
        named_after = entry_lstat(directory_fd, name)
        os.close(descriptor)
        if (
            _file_identity(after) != _file_identity(opened)
            or _file_identity(named_after) != _file_identity(opened)
        ):
            raise ValueError(f"external route run identity changed: {name}")


def _write_route_run(
    directory_fd: int,
    name: str,
    rows: Iterable[FactMetadata],
) -> int:
    descriptor = _open_new_regular(directory_fd, name)
    row_count = 0
    try:
        for fact in rows:
            _write_all(descriptor, _route_row_bytes(fact))
            row_count += 1
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return row_count


@dataclass(frozen=True)
class _SortedRouteRows:
    directory_fd: int
    name: str
    row_count: int

    def __iter__(self) -> Iterator[FactMetadata]:
        return _iter_route_run(self.directory_fd, self.name)


def external_sort_rows(
    rows: Iterable[FactMetadata],
    directory_fd: int,
    *,
    sort_name: str,
    sort_key: Callable[[FactMetadata], tuple[object, ...]],
    chunk_rows: int | None = None,
) -> _SortedRouteRows:
    """Sort route rows through bounded canonical runs on a pinned directory."""

    limit = _EXTERNAL_SORT_CHUNK_ROWS if chunk_rows is None else chunk_rows
    if type(limit) is not int or limit <= 0:
        raise ValueError("external route sort chunk size must be positive")
    if (
        type(_EXTERNAL_SORT_MERGE_FAN_IN) is not int
        or _EXTERNAL_SORT_MERGE_FAN_IN < 2
    ):
        raise ValueError("external route sort merge fan-in must be at least two")
    _strict_text(sort_name, "external route sort name")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in sort_name):
        raise ValueError("external route sort name is not filesystem-safe")

    runs: list[tuple[str, int]] = []
    chunk: list[FactMetadata] = []
    total = 0
    chunk_index = 0
    for fact in rows:
        if not isinstance(fact, FactMetadata):
            raise ValueError(  # noqa: TRY004 - malformed artifact data
                "external route sort requires FactMetadata rows"
            )
        chunk.append(fact)
        total += 1
        if len(chunk) < limit:
            continue
        chunk.sort(key=sort_key)
        name = f"{sort_name}-chunk-{chunk_index:08d}.jsonl"
        written = _write_route_run(directory_fd, name, chunk)
        if written != len(chunk):
            raise ValueError("external route sort chunk row count changed")
        runs.append((name, written))
        chunk = []
        chunk_index += 1
    if chunk or not runs:
        chunk.sort(key=sort_key)
        name = f"{sort_name}-chunk-{chunk_index:08d}.jsonl"
        written = _write_route_run(directory_fd, name, chunk)
        if written != len(chunk):
            raise ValueError("external route sort chunk row count changed")
        runs.append((name, written))

    merge_pass = 0
    while len(runs) > 1:
        merged_runs: list[tuple[str, int]] = []
        for group_index, offset in enumerate(
            range(0, len(runs), _EXTERNAL_SORT_MERGE_FAN_IN)
        ):
            group = runs[
                offset : offset + _EXTERNAL_SORT_MERGE_FAN_IN
            ]
            merged_name = (
                f"{sort_name}-merge-{merge_pass:04d}-"
                f"{group_index:08d}.jsonl"
            )
            iterators = tuple(
                _iter_route_run(directory_fd, name) for name, _count in group
            )
            merged = heapq.merge(*iterators, key=sort_key)
            try:
                merged_count = _write_route_run(
                    directory_fd,
                    merged_name,
                    merged,
                )
            finally:
                for iterator in iterators:
                    iterator.close()
            expected = sum(count for _name, count in group)
            if merged_count != expected:
                raise ValueError("external route merge row count changed")
            merged_runs.append((merged_name, merged_count))
        runs = merged_runs
        merge_pass += 1
    return _SortedRouteRows(
        directory_fd=directory_fd,
        name=runs[0][0],
        row_count=total,
    )


def _rank_key(fact: FactMetadata) -> tuple[object, ...]:
    return (-route_score(fact), fact.fact_id)


def _incoming_key(fact: FactMetadata) -> tuple[object, ...]:
    return (
        -fact.information_burden_bits,
        -route_score(fact),
        fact.fact_id,
    )


def _outgoing_key(fact: FactMetadata) -> tuple[object, ...]:
    return (
        fact.information_burden_bits,
        route_score(fact),
        fact.fact_id,
    )


def audit_route_dose(
    *,
    total_facts: int,
    external_facts: int,
    total_information_burden: Fraction,
    external_information_burden: Fraction,
) -> dict[str, object]:
    if (
        type(total_facts) is not int
        or type(external_facts) is not int
        or total_facts <= 0
        or external_facts < 0
        or external_facts > total_facts
    ):
        raise ValueError("route dose fact counts are invalid")
    total_burden = _strict_fraction(
        total_information_burden,
        "total information burden",
    )
    external_burden = _strict_fraction(
        external_information_burden,
        "external information burden",
    )
    if external_burden > total_burden:
        raise ValueError("external information burden exceeds total burden")
    distinct_fraction = Fraction(external_facts, total_facts)
    burden_fraction = (
        Fraction(1)
        if total_burden == 0
        else external_burden / total_burden
    )
    if distinct_fraction < _TARGET_FRACTION:
        raise ValueError("Split90 distinct fact dose is below 9/10")
    if burden_fraction < _TARGET_FRACTION:
        raise ValueError("Split90 information burden dose is below 9/10")
    return {
        "distinct_fact_fraction": _fraction_dict(distinct_fraction),
        "distinct_fact_passed": True,
        "external_fact_count": external_facts,
        "external_information_burden_bits": _fraction_dict(external_burden),
        "format": _ROUTE_DOSE_FORMAT,
        "information_burden_fraction": _fraction_dict(burden_fraction),
        "information_burden_passed": True,
        "passed": True,
        "split": _SPLIT,
        "target_external_fraction": _fraction_dict(_TARGET_FRACTION),
        "total_fact_count": total_facts,
        "total_information_burden_bits": _fraction_dict(total_burden),
    }


def _selected(
    connection: sqlite3.Connection,
    fact_id: str,
) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM selected WHERE fact_id = ?",
            (fact_id,),
        ).fetchone()
        is not None
    )


def _close_iterator(iterator: Iterator[FactMetadata]) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def _repair_burden_selection(
    rank_rows: _SortedRouteRows,
    selected_connection: sqlite3.Connection,
    work_fd: int,
    quota: int,
    selected_burden: Fraction,
    target_burden: Fraction,
) -> Fraction:
    incoming_rows = external_sort_rows(
        (
            fact
            for rank, fact in enumerate(rank_rows)
            if rank >= quota
        ),
        work_fd,
        sort_name="incoming",
        sort_key=_incoming_key,
    )
    outgoing_rows = external_sort_rows(
        (
            fact
            for rank, fact in enumerate(rank_rows)
            if rank < quota
        ),
        work_fd,
        sort_name="outgoing",
        sort_key=_outgoing_key,
    )
    incoming_iterator = iter(incoming_rows)
    outgoing_iterator = iter(outgoing_rows)
    try:
        for incoming, outgoing in zip(
            incoming_iterator,
            outgoing_iterator,
            strict=False,
        ):
            if selected_burden >= target_burden:
                break
            gain = (
                incoming.information_burden_bits
                - outgoing.information_burden_bits
            )
            if gain <= 0:
                break
            deleted = selected_connection.execute(
                "DELETE FROM selected WHERE fact_id = ?",
                (outgoing.fact_id,),
            ).rowcount
            if deleted != 1:
                raise ValueError("route burden repair lost an outgoing fact")
            try:
                selected_connection.execute(
                    "INSERT INTO selected(fact_id) VALUES (?)",
                    (incoming.fact_id,),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(
                    "route burden repair duplicated an incoming fact"
                ) from error
            selected_burden += gain
        selected_connection.commit()
    finally:
        _close_iterator(incoming_iterator)
        _close_iterator(outgoing_iterator)
    return selected_burden


def _write_decisions(
    work_fd: int,
    rank_rows: _SortedRouteRows,
    selected_connection: sqlite3.Connection,
    quota: int,
) -> tuple[int, int, str]:
    descriptor = _open_new_regular(work_fd, _DECISION_STREAM_NAME)
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    try:
        for rank, fact in enumerate(rank_rows):
            external = _selected(selected_connection, fact.fact_id)
            decision = RouteDecision(
                fact=fact,
                score=route_score(fact),
                rank=rank,
                route="external" if external else "internal",
                reason=(
                    "selected by descending train score at fixed quota"
                    if external and rank < quota
                    else "selected to satisfy train-only information-burden quota"
                    if external
                    else "displaced by train-only information-burden quota"
                    if rank < quota
                    else "retained after fixed train-score quota"
                ),
            )
            payload = canonical_json_bytes(decision.as_dict())
            _write_all(descriptor, payload)
            digest.update(payload)
            byte_count += len(payload)
            row_count += 1
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return row_count, byte_count, digest.hexdigest()


def _write_payload(
    directory_fd: int,
    name: str,
    payload: bytes,
) -> None:
    descriptor = _open_new_regular(directory_fd, name)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _route_artifacts(
    output_root: Path,
    *,
    manifest_sha256: str,
    dose_report_sha256: str,
    external_fact_count: int,
    distinct_fact_fraction: Fraction,
    information_burden_fraction: Fraction,
    dose_report: dict[str, object],
) -> RouteArtifacts:
    return RouteArtifacts(
        manifest_path=output_root / _ROUTE_MANIFEST_NAME,
        index_path=output_root / _ROUTE_INDEX_NAME,
        dose_report_path=output_root / _DOSE_REPORT_NAME,
        manifest_sha256=manifest_sha256,
        dose_report_sha256=dose_report_sha256,
        external_fact_count=external_fact_count,
        distinct_fact_fraction=distinct_fact_fraction,
        information_burden_fraction=information_burden_fraction,
        dose_report=dose_report,
    )


def _open_route_index_at(
    directory_fd: int,
    name: str,
    database_path: Path,
) -> RouteIndex:
    duplicated = os.dup(directory_fd)
    try:
        return RouteIndex._open_at(duplicated, name, database_path)
    except BaseException:
        os.close(duplicated)
        raise


def _indexes_match(
    first_directory_fd: int,
    second_directory_fd: int,
) -> bool:
    with _open_route_index_at(
        first_directory_fd,
        _ROUTE_INDEX_NAME,
        Path(_ROUTE_INDEX_NAME),
    ) as first, _open_route_index_at(
        second_directory_fd,
        _ROUTE_INDEX_NAME,
        Path(_ROUTE_INDEX_NAME),
    ) as second:
        sentinel = object()
        for left, right in zip_longest(
            first.iter_external_fact_ids(),
            second.iter_external_fact_ids(),
            fillvalue=sentinel,
        ):
            if left != right:
                return False
    return True


def _verify_output_stage(
    output_fd: int,
    *,
    manifest_bytes: int,
    manifest_sha256: str,
    dose_report_bytes: int,
    dose_report_sha256: str,
    index_bytes: int,
    index_sha256: str,
) -> None:
    if list_entries(output_fd) != (
        _DOSE_REPORT_NAME,
        _ROUTE_INDEX_NAME,
        _ROUTE_MANIFEST_NAME,
    ):
        raise ValueError("route artifact inventory changed")
    _verify_regular_at(
        output_fd,
        _ROUTE_MANIFEST_NAME,
        expected_bytes=manifest_bytes,
        expected_sha256=manifest_sha256,
        description="route manifest",
    )
    _verify_regular_at(
        output_fd,
        _DOSE_REPORT_NAME,
        expected_bytes=dose_report_bytes,
        expected_sha256=dose_report_sha256,
        description="route dose report",
    )
    _verify_regular_at(
        output_fd,
        _ROUTE_INDEX_NAME,
        expected_bytes=index_bytes,
        expected_sha256=index_sha256,
        description="route index",
    )
    with _open_route_index_at(
        output_fd,
        _ROUTE_INDEX_NAME,
        Path(_ROUTE_INDEX_NAME),
    ):
        pass


def _verify_exact_route_winner(
    parent_fd: int,
    final_name: str,
    output_root: Path,
    candidate_fd: int,
    *,
    manifest_bytes: int,
    manifest_sha256: str,
    dose_report_bytes: int,
    dose_report_sha256: str,
    index_bytes: int,
    index_sha256: str,
    external_fact_count: int,
    distinct_fact_fraction: Fraction,
    information_burden_fraction: Fraction,
    dose_report: dict[str, object],
) -> RouteArtifacts:
    winner_fd = -1
    try:
        winner_fd, _created = open_directory_at(parent_fd, final_name)
        opened = os.fstat(winner_fd)
        named = entry_lstat(parent_fd, final_name)
        _require_owned_directory(opened, "route winner")
        identity = _directory_identity(opened)
        if _directory_identity(named) != identity:
            raise ValueError("route winner identity changed before verification")
        _verify_output_stage(
            winner_fd,
            manifest_bytes=manifest_bytes,
            manifest_sha256=manifest_sha256,
            dose_report_bytes=dose_report_bytes,
            dose_report_sha256=dose_report_sha256,
            index_bytes=index_bytes,
            index_sha256=index_sha256,
        )
        if not _indexes_match(winner_fd, candidate_fd):
            raise ValueError("route winner index differs")
        if (
            _directory_identity(os.fstat(winner_fd)) != identity
            or _directory_identity(entry_lstat(parent_fd, final_name))
            != identity
        ):
            raise ValueError("route winner identity changed during verification")
        return _route_artifacts(
            output_root,
            manifest_sha256=manifest_sha256,
            dose_report_sha256=dose_report_sha256,
            external_fact_count=external_fact_count,
            distinct_fact_fraction=distinct_fact_fraction,
            information_burden_fraction=information_burden_fraction,
            dose_report=dose_report,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ValueError(f"conflicting route winner: {output_root}") from error
    finally:
        if winner_fd >= 0:
            os.close(winner_fd)


def build_route_artifacts(
    catalog: InputCatalog,
    work_root: Path,
    output_root: Path,
) -> RouteArtifacts:
    if not isinstance(work_root, Path) or not isinstance(output_root, Path):
        raise TypeError("route work and output roots must be pathlib.Path values")
    if os.fspath(work_root) == os.fspath(output_root):
        raise ValueError("route work and output roots must differ")

    work_parent_fd = -1
    output_parent_fd = -1
    work_quarantine_fd = -1
    output_quarantine_fd = -1
    work_stage_fd = -1
    output_stage_fd = -1
    work_stage_name = ""
    output_stage_name = ""
    work_stage_identity: tuple[int, int, int, int] | None = None
    output_stage_identity: tuple[int, int, int, int] | None = None
    reducer: _PinnedDatabase | None = None
    selected_database: _PinnedDatabase | None = None
    try:
        work_parent_fd, work_name = open_parent_directory(
            work_root,
            create=True,
            mode=0o700,
        )
        output_parent_fd, output_name = open_parent_directory(
            output_root,
            create=True,
            mode=0o700,
        )
        _require_owned_directory(
            os.fstat(work_parent_fd),
            "route work parent",
        )
        _require_owned_directory(
            os.fstat(output_parent_fd),
            "route output parent",
        )
        work_quarantine_fd, _work_quarantine_identity = (
            _open_route_quarantine(work_parent_fd)
        )
        output_quarantine_fd, _output_quarantine_identity = (
            _open_route_quarantine(output_parent_fd)
        )
        (
            work_stage_name,
            work_stage_fd,
            work_stage_identity,
        ) = _new_stage(work_parent_fd, work_name)
        (
            output_stage_name,
            output_stage_fd,
            output_stage_identity,
        ) = _new_stage(output_parent_fd, output_name)

        reducer = _create_reducer(work_stage_fd)
        assert reducer.connection is not None
        total_facts = _reduce_catalog(catalog, reducer.connection)
        rank_rows = external_sort_rows(
            _iter_reduced_facts(reducer.connection),
            work_stage_fd,
            sort_name="rank",
            sort_key=_rank_key,
        )
        if rank_rows.row_count != total_facts:
            raise ValueError("external route rank row count changed")

        quota = minimally_rounded_quota(total_facts, _TARGET_FRACTION)
        selected_database = _create_route_index(output_stage_fd)
        assert selected_database.connection is not None
        total_burden = Fraction()
        selected_burden = Fraction()
        for rank, fact in enumerate(rank_rows):
            burden = fact.information_burden_bits
            total_burden += burden
            if rank < quota:
                selected_database.connection.execute(
                    "INSERT INTO selected(fact_id) VALUES (?)",
                    (fact.fact_id,),
                )
                selected_burden += burden
        selected_database.connection.commit()
        selected_burden = _repair_burden_selection(
            rank_rows,
            selected_database.connection,
            work_stage_fd,
            quota,
            selected_burden,
            total_burden * _TARGET_FRACTION,
        )
        if selected_burden < total_burden * _TARGET_FRACTION:
            raise ValueError(
                "Split90 fact quota cannot meet information burden dose"
            )

        external_fact_count = 0
        replayed_external_burden = Fraction()
        for fact in rank_rows:
            if _selected(selected_database.connection, fact.fact_id):
                external_fact_count += 1
                replayed_external_burden += fact.information_burden_bits
        if external_fact_count != quota:
            raise ValueError("route selection count changed after burden repair")
        if replayed_external_burden != selected_burden:
            raise ValueError("route selected burden changed after repair")
        dose_report = audit_route_dose(
            total_facts=total_facts,
            external_facts=external_fact_count,
            total_information_burden=total_burden,
            external_information_burden=replayed_external_burden,
        )
        distinct_fraction = Fraction(external_fact_count, total_facts)
        information_burden_fraction = (
            Fraction(1)
            if total_burden == 0
            else replayed_external_burden / total_burden
        )

        (
            decision_row_count,
            decision_bytes,
            decision_sha256,
        ) = _write_decisions(
            work_stage_fd,
            rank_rows,
            selected_database.connection,
            quota,
        )
        if decision_row_count != total_facts:
            raise ValueError("route decision row count changed")
        header = {
            "decision_stream_bytes": decision_bytes,
            "decision_stream_sha256": decision_sha256,
            "distinct_fact_fraction": _fraction_dict(distinct_fraction),
            "external_fact_count": external_fact_count,
            "external_information_burden_bits": _fraction_dict(
                replayed_external_burden
            ),
            "format": _ROUTE_MANIFEST_FORMAT,
            "information_burden_fraction": _fraction_dict(
                information_burden_fraction
            ),
            "metadata_scope": METADATA_SCOPE,
            "policy": ROUTE_POLICY,
            "row_count": total_facts,
            "schema_version": 1,
            "split": _SPLIT,
            "target_external_fraction": _fraction_dict(_TARGET_FRACTION),
            "total_information_burden_bits": _fraction_dict(total_burden),
            "train_only_features": list(TRAIN_ONLY_FEATURES),
        }
        header_bytes = canonical_json_bytes(header)
        manifest_fd = _open_new_regular(
            output_stage_fd,
            _ROUTE_MANIFEST_NAME,
        )
        manifest_digest = hashlib.sha256()
        try:
            _write_all(manifest_fd, header_bytes)
            manifest_digest.update(header_bytes)
            copied_decision_bytes = _copy_regular_at(
                work_stage_fd,
                _DECISION_STREAM_NAME,
                manifest_fd,
                manifest_digest,
            )
            if copied_decision_bytes != decision_bytes:
                raise ValueError("route decision stream byte count changed")
            os.fsync(manifest_fd)
        finally:
            os.close(manifest_fd)
        manifest_bytes = len(header_bytes) + decision_bytes
        manifest_sha256 = manifest_digest.hexdigest()

        dose_report_bytes = canonical_json_bytes(dose_report)
        dose_report_sha256 = sha256_hex(dose_report_bytes)
        _write_payload(
            output_stage_fd,
            _DOSE_REPORT_NAME,
            dose_report_bytes,
        )

        selected_database.close_connection(
            output_stage_fd,
            "route index database",
        )
        selected_database.close_descriptor()
        selected_database = None
        index_bytes, index_sha256 = _regular_commitment_at(
            output_stage_fd,
            _ROUTE_INDEX_NAME,
            description="route index",
        )
        reducer.close_connection(work_stage_fd, "route reducer database")
        reducer.close_descriptor()
        reducer = None

        _verify_output_stage(
            output_stage_fd,
            manifest_bytes=manifest_bytes,
            manifest_sha256=manifest_sha256,
            dose_report_bytes=len(dose_report_bytes),
            dose_report_sha256=dose_report_sha256,
            index_bytes=index_bytes,
            index_sha256=index_sha256,
        )
        if (
            _directory_identity(os.fstat(work_stage_fd))
            != work_stage_identity
            or _directory_identity(
                entry_lstat(work_parent_fd, work_stage_name)
            )
            != work_stage_identity
        ):
            raise ValueError("route work stage identity changed")
        _quarantine_stage(
            work_parent_fd,
            work_stage_name,
            work_stage_fd,
            work_stage_identity,
            work_quarantine_fd,
            "work",
        )
        work_stage_name = ""

        fsync_directory(output_stage_fd)
        if (
            _directory_identity(os.fstat(output_stage_fd))
            != output_stage_identity
            or _directory_identity(
                entry_lstat(output_parent_fd, output_stage_name)
            )
            != output_stage_identity
        ):
            raise ValueError("route output stage identity changed")
        try:
            atomic_rename_noreplace(
                output_parent_fd,
                output_stage_name,
                output_parent_fd,
                output_name,
            )
        except FileExistsError:
            winner = _verify_exact_route_winner(
                output_parent_fd,
                output_name,
                output_root,
                output_stage_fd,
                manifest_bytes=manifest_bytes,
                manifest_sha256=manifest_sha256,
                dose_report_bytes=len(dose_report_bytes),
                dose_report_sha256=dose_report_sha256,
                index_bytes=index_bytes,
                index_sha256=index_sha256,
                external_fact_count=external_fact_count,
                distinct_fact_fraction=distinct_fraction,
                information_burden_fraction=information_burden_fraction,
                dose_report=dose_report,
            )
            _quarantine_stage(
                output_parent_fd,
                output_stage_name,
                output_stage_fd,
                output_stage_identity,
                output_quarantine_fd,
                "candidate",
            )
            output_stage_name = ""
            return winner
        output_stage_name = ""
        fsync_directory(output_parent_fd)
        final_named = entry_lstat(output_parent_fd, output_name)
        if _directory_identity(final_named) != output_stage_identity:
            raise ValueError("published route identity changed")
        final_fd, _created = open_directory_at(output_parent_fd, output_name)
        try:
            if _directory_identity(os.fstat(final_fd)) != output_stage_identity:
                raise ValueError("published route identity changed")
            _verify_output_stage(
                final_fd,
                manifest_bytes=manifest_bytes,
                manifest_sha256=manifest_sha256,
                dose_report_bytes=len(dose_report_bytes),
                dose_report_sha256=dose_report_sha256,
                index_bytes=index_bytes,
                index_sha256=index_sha256,
            )
            if (
                _directory_identity(os.fstat(final_fd))
                != output_stage_identity
                or _directory_identity(
                    entry_lstat(output_parent_fd, output_name)
                )
                != output_stage_identity
            ):
                raise ValueError("published route identity changed during replay")
        finally:
            os.close(final_fd)
        return _route_artifacts(
            output_root,
            manifest_sha256=manifest_sha256,
            dose_report_sha256=dose_report_sha256,
            external_fact_count=external_fact_count,
            distinct_fact_fraction=distinct_fraction,
            information_burden_fraction=information_burden_fraction,
            dose_report=dose_report,
        )
    except BaseException as build_error:
        if reducer is not None:
            reducer.abort_connection()
            reducer.close_descriptor()
            reducer = None
        if selected_database is not None:
            selected_database.abort_connection()
            selected_database.close_descriptor()
            selected_database = None
        quarantine_errors: list[BaseException] = []
        if (
            work_parent_fd >= 0
            and work_quarantine_fd >= 0
            and work_stage_fd >= 0
            and work_stage_name
            and work_stage_identity is not None
        ):
            try:
                _quarantine_stage(
                    work_parent_fd,
                    work_stage_name,
                    work_stage_fd,
                    work_stage_identity,
                    work_quarantine_fd,
                    "work",
                )
                work_stage_name = ""
            except BaseException as error:  # noqa: BLE001 - preserve cancellation
                quarantine_errors.append(error)
        if (
            output_parent_fd >= 0
            and output_quarantine_fd >= 0
            and output_stage_fd >= 0
            and output_stage_name
            and output_stage_identity is not None
        ):
            try:
                _quarantine_stage(
                    output_parent_fd,
                    output_stage_name,
                    output_stage_fd,
                    output_stage_identity,
                    output_quarantine_fd,
                    "candidate",
                )
                output_stage_name = ""
            except BaseException as error:  # noqa: BLE001 - preserve cancellation
                quarantine_errors.append(error)
        if quarantine_errors:
            raise ValueError(
                "route build failed and exact-inode quarantine failed"
            ) from build_error
        raise
    finally:
        if reducer is not None:
            reducer.abort_connection()
            reducer.close_descriptor()
        if selected_database is not None:
            selected_database.abort_connection()
            selected_database.close_descriptor()
        if work_stage_fd >= 0:
            os.close(work_stage_fd)
        if output_stage_fd >= 0:
            os.close(output_stage_fd)
        if work_quarantine_fd >= 0:
            os.close(work_quarantine_fd)
        if output_quarantine_fd >= 0:
            os.close(output_quarantine_fd)
        if work_parent_fd >= 0:
            os.close(work_parent_fd)
        if output_parent_fd >= 0:
            os.close(output_parent_fd)


def _validated_spans(
    token_count: int,
    spans: tuple[TokenSemanticSpan, ...],
) -> tuple[TokenSemanticSpan, ...]:
    if type(token_count) is not int or token_count < 0:
        raise ValueError("token count must be a non-negative integer")
    if type(spans) is not tuple:
        raise TypeError("token semantic spans must be an ordered tuple")
    for span in spans:
        if not isinstance(span, TokenSemanticSpan):
            raise TypeError("semantic spans must contain TokenSemanticSpan values")
        if span.token_end > token_count:
            raise ValueError("token semantic span is outside token bounds")
    keys = tuple((span.token_start, span.token_end) for span in spans)
    if keys != tuple(sorted(keys)):
        raise ValueError("token semantic spans must be canonically ordered")

    active: list[TokenSemanticSpan] = []
    for span in spans:
        active = [
            previous
            for previous in active
            if previous.token_end > span.token_start
        ]
        for previous in active:
            if previous.fact_id != span.fact_id:
                raise ValueError(
                    "cross-fact overlapping token semantic spans"
                )
            if previous.role != span.role:
                roles = {previous.role, span.role}
                if "factual_payload" in roles and "proof" in roles:
                    raise SemanticLeakageError(
                        "proof token is also marked as factual payload",
                        routed_fact_ids=tuple(
                            fact_id
                            for fact_id in (previous.fact_id, span.fact_id)
                            if fact_id is not None
                        ),
                    )
                if "factual_payload" in roles and "answer_state" in roles:
                    raise SemanticLeakageError(
                        "answer state token is also marked as factual payload",
                        routed_fact_ids=tuple(
                            fact_id
                            for fact_id in (previous.fact_id, span.fact_id)
                            if fact_id is not None
                        ),
                    )
                raise ValueError(
                    "overlapping token semantic spans have conflicting roles"
                )
        active.append(span)
    return spans


def _expected_split90(
    token_count: int,
    spans: tuple[TokenSemanticSpan, ...],
    routes: RouteIndex,
) -> bytes:
    if not isinstance(routes, RouteIndex):
        raise TypeError("routes must be a read-only RouteIndex")
    expected = bytearray(b"\x01" * token_count)
    route_cache: dict[str, bool] = {}
    for span in spans:
        if span.role != "factual_payload":
            continue
        assert span.fact_id is not None
        external = route_cache.get(span.fact_id)
        if external is None:
            external = routes.is_external(span.fact_id)
            route_cache[span.fact_id] = external
        if external:
            expected[span.token_start : span.token_end] = (
                b"\x00" * (span.token_end - span.token_start)
            )
    return bytes(expected)


def audit_sidecar_weights(
    token_count: int,
    spans: tuple[TokenSemanticSpan, ...],
    routes: RouteIndex,
    *,
    dense: bytes,
    split90: bytes,
) -> tuple[dict[str, object], ...]:
    validated = _validated_spans(token_count, spans)
    if type(dense) is not bytes or type(split90) is not bytes:
        raise SemanticLeakageError("sidecar masks must be bytes")
    if len(dense) != token_count or len(split90) != token_count:
        raise SemanticLeakageError("sidecar mask length differs from token count")
    if any(value not in (0, 1) for value in dense):
        raise SemanticLeakageError("Dense sidecar is not binary")
    if any(value != 1 for value in dense):
        raise SemanticLeakageError("Dense sidecar masks supervised targets")
    if any(value not in (0, 1) for value in split90):
        raise SemanticLeakageError("Split90 sidecar is not binary")
    expected = _expected_split90(token_count, validated, routes)
    leaks: list[dict[str, object]] = []
    for index, (expected_value, actual_value) in enumerate(
        zip(expected, split90, strict=True)
    ):
        if expected_value == actual_value:
            continue
        leaks.append(
            {
                "actual": actual_value,
                "expected": expected_value,
                "kind": (
                    "unmasked_routed_factual_payload"
                    if expected_value == 0
                    else "masked_nonpayload_target"
                ),
                "token_index": index,
            }
        )
    if leaks:
        kinds = {str(leak["kind"]) for leak in leaks}
        if "unmasked_routed_factual_payload" in kinds:
            message = "unmasked routed factual payload target"
        else:
            message = "Split90 masks a nonpayload target"
        raise SemanticLeakageError(
            message,
            leaks=leaks,
            routed_fact_ids=tuple(
                span.fact_id
                for span in validated
                if span.fact_id is not None
            ),
        )
    return ()


def derive_sidecar_weights(
    token_count: int,
    spans: tuple[TokenSemanticSpan, ...],
    routes: RouteIndex,
) -> SidecarWeights:
    validated = _validated_spans(token_count, spans)
    dense = b"\x01" * token_count
    split90 = _expected_split90(token_count, validated, routes)
    leaks = audit_sidecar_weights(
        token_count,
        validated,
        routes,
        dense=dense,
        split90=split90,
    )
    return SidecarWeights(
        dense=dense,
        split90=split90,
        routed_payload_targets=split90.count(0),
        leaks=leaks,
    )


def _audit_forbidden_surface_role(
    text: str,
    routed_facts: tuple[SemanticFact, ...],
    *,
    role: Literal["answer state", "proof"],
) -> tuple[dict[str, object], ...]:
    text = _strict_text(text, role, nonempty=False)
    if type(routed_facts) is not tuple:
        raise TypeError("routed semantic facts must be an ordered tuple")
    if any(not isinstance(fact, SemanticFact) for fact in routed_facts):
        raise TypeError("routed facts must contain SemanticFact values")
    field_id = "answer_state" if role == "answer state" else "proof"
    plan = plan_occurrence_closure(
        routed_facts,
        (SupervisedField(field_id, text),),
    )
    report = audit_occurrence_closure(
        routed_facts,
        plan.fields,
        tuple(fact.fact_id for fact in routed_facts),
        {field_id: (1,) * len(text)},
        fail_closed=False,
    )
    if report.unmasked_occurrences != plan.occurrences:
        raise SemanticLeakageError(
            f"{role} occurrence closure replay disagrees",
            routed_fact_ids=tuple(fact.fact_id for fact in routed_facts),
            metadata_errors=("semantic occurrence replay disagreement",),
        )
    if report.unmasked_occurrences:
        raise SemanticLeakageError(
            f"routed factual surface appears in {role}",
            routed_fact_ids=tuple(fact.fact_id for fact in routed_facts),
            occurrences=report.unmasked_occurrences,
        )
    return ()


def audit_answer_state_surfaces(
    answer_state: str,
    routed_facts: tuple[SemanticFact, ...],
) -> tuple[dict[str, object], ...]:
    return _audit_forbidden_surface_role(
        answer_state,
        routed_facts,
        role="answer state",
    )


def audit_proof_surfaces(
    proof_text: str,
    routed_facts: tuple[SemanticFact, ...],
) -> tuple[dict[str, object], ...]:
    return _audit_forbidden_surface_role(
        proof_text,
        routed_facts,
        role="proof",
    )


__all__ = (
    "RouteArtifacts",
    "RouteIndex",
    "SemanticLeakageError",
    "SemanticRole",
    "SidecarWeights",
    "TokenSemanticSpan",
    "audit_answer_state_surfaces",
    "audit_proof_surfaces",
    "audit_route_dose",
    "audit_sidecar_weights",
    "build_route_artifacts",
    "derive_sidecar_weights",
    "external_sort_rows",
)
