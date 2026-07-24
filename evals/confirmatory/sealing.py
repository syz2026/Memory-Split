"""Deterministic pre-training sealing for confirmatory evaluation releases."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
import ctypes
from dataclasses import dataclass, replace
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from types import MappingProxyType
from typing import Any, TypeVar

from evals.confirmatory.contracts import (
    STUDY_CONTRACT_VERSION,
    Control,
    ItemRecord,
    MemoryMode,
    SealedGoldRecord,
    StoreRecord,
    Twin,
    canonical_json_bytes,
)
from evals.confirmatory.solver import registered_solver, verify_sealed_gold
from evals.confirmatory.study_lock import (
    FROZEN_PREREGISTRATION_SHA256_V3,
    REQUIRED_CONTROL_IDS,
    REQUIRED_FAMILIES,
    REQUIRED_MEMORY_MODES,
    REQUIRED_STRATA,
)


SEALED_RELEASE_SCHEMA = "memorysplit.confirmatory.sealed-release.v3"
SEALED_RELEASE_MANIFEST = "sealed-release.json"
ITEMS_NAME = "items.jsonl"
STORES_NAME = "stores.jsonl"
SEALED_GOLD_NAME = "sealed-gold.jsonl"
RELEASE_ID_PREFIX = "memorysplit-confirmatory-sealed-v3-"

_SOURCE_NAMES = (ITEMS_NAME, STORES_NAME, SEALED_GOLD_NAME)
_RELEASE_NAMES = (
    ITEMS_NAME,
    STORES_NAME,
    SEALED_GOLD_NAME,
    SEALED_RELEASE_MANIFEST,
)
_RELEASE_MODES = MappingProxyType(
    {
        ITEMS_NAME: 0o644,
        STORES_NAME: 0o644,
        SEALED_GOLD_NAME: 0o600,
        SEALED_RELEASE_MANIFEST: 0o644,
    }
)
_CONTROL_IDENTITIES = MappingProxyType(
    {
        (MemoryMode.MEMORY_ON, Control.CORRECT): "correct_memory",
        (MemoryMode.MEMORY_OFF, Control.CORRECT): "memory_off",
        (
            MemoryMode.MEMORY_ON,
            Control.SHUFFLED_RETURNS,
        ): "shuffled_returns",
        (MemoryMode.MEMORY_ON, Control.RELEVANT_EDGE): "relevant_edge_swap",
        (
            MemoryMode.MEMORY_ON,
            Control.IRRELEVANT_EDGE,
        ): "irrelevant_edge_swap",
        (MemoryMode.MEMORY_ON, Control.GOLD_PATH): "gold_path_replay",
        (MemoryMode.MEMORY_ON, Control.NO_QUERY): "no_query",
        (MemoryMode.MEMORY_ON, Control.ENTITY_RENAME): "entity_rename",
        (
            MemoryMode.MEMORY_ON,
            Control.GRAPH_ISOMORPHISM,
        ): "graph_isomorphism",
        (
            MemoryMode.MEMORY_ON,
            Control.PAGE_ORDER_PERMUTATION,
        ): "page_order_permutation",
    }
)
_T = TypeVar("_T")


class SealingError(ValueError):
    """The requested release cannot be sealed safely."""

    def __init__(self, message: str, *, code: str = "SEALING_INVALID") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SealedReleaseResult:
    release_id: str
    release_dir: Path
    release_sha256: str
    manifest_bytes: bytes
    item_count: int
    pair_count: int
    world_count: int
    store_count: int
    published: bool


@dataclass(frozen=True)
class VerifiedSealedRelease:
    release_dir: Path
    release_sha256: str
    manifest_bytes: bytes
    item_count: int
    pair_count: int
    world_count: int
    store_count: int
    sealed_gold_sha256: str


@dataclass(frozen=True)
class ModelVisiblePreflightResult:
    release_dir: Path
    release_sha256: str
    item_count: int
    pair_count: int
    world_count: int
    store_count: int
    sealed_gold_sha256: str


@dataclass(frozen=True)
class _PinnedFile:
    parent_fd: int
    name: str
    descriptor: int
    content: bytes
    identity: tuple[int, int]
    state: tuple[int, int, int, int, int, int, int]


@dataclass(frozen=True)
class _ValidatedRelease:
    manifest: Mapping[str, Any]
    manifest_bytes: bytes
    item_count: int
    pair_count: int
    world_count: int
    store_count: int


@dataclass(frozen=True)
class _ValidatedModelVisible:
    binding: Mapping[str, Any]
    items: tuple[ItemRecord, ...]
    stores: tuple[StoreRecord, ...]
    item_count: int
    pair_count: int
    world_count: int
    store_count: int


@dataclass(frozen=True)
class _StagedRelease:
    name: str
    descriptor: int
    files: tuple[_PinnedFile, ...]


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _file_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _safe_absolute_path(path: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\0" in raw:
        raise SealingError(f"{name} must be a non-empty filesystem path")
    lexical = Path(raw)
    if any(part == ".." for part in lexical.parts):
        raise SealingError(f"{name} must not contain path traversal")
    absolute = Path(os.path.abspath(raw))
    if absolute == Path("/"):
        raise SealingError(f"{name} must not be the filesystem root")
    return absolute


def _assert_owned_directory(details: os.stat_result, name: str) -> None:
    if not stat.S_ISDIR(details.st_mode):
        raise SealingError(f"{name} must be a directory")
    if details.st_uid != os.geteuid():
        raise SealingError(f"{name} must be owned by the current user")
    if stat.S_IMODE(details.st_mode) & 0o022:
        raise SealingError(f"{name} must not be group- or world-writable")


def _open_directory(
    path: str | os.PathLike[str],
    name: str,
) -> tuple[Path, int]:
    absolute = _safe_absolute_path(path, name)
    descriptor = os.open("/", _directory_flags())
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise SealingError(
                    f"{name} contains a symlink, missing component, "
                    "or non-directory"
                ) from exc
            os.close(descriptor)
            descriptor = child
        _assert_owned_directory(os.fstat(descriptor), name)
        _assert_directory_path(absolute, descriptor, name)
        return absolute, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_planned_directory_path(path: Path, name: str) -> None:
    """Inspect every existing component without creating the requested path."""

    descriptor = os.open("/", _directory_flags())
    try:
        components = path.parts[1:]
        for index, component in enumerate(components):
            try:
                details = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return
            except OSError as exc:
                raise SealingError(f"{name} cannot be inspected safely") from exc
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                raise SealingError(
                    f"{name} contains a symlink or non-directory component"
                )
            try:
                child = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise SealingError(
                    f"{name} contains a symlink or non-directory component"
                ) from exc
            os.close(descriptor)
            descriptor = child
            if index == len(components) - 1:
                _assert_owned_directory(os.fstat(descriptor), name)
    finally:
        os.close(descriptor)


def _assert_directory_path(path: Path, descriptor: int, name: str) -> None:
    pinned = os.fstat(descriptor)
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise SealingError(f"{name} changed after descriptor pinning") from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != (pinned.st_dev, pinned.st_ino)
    ):
        raise SealingError(f"{name} was replaced after descriptor pinning")
    _assert_owned_directory(current, name)


def _regular_file_state(
    details: os.stat_result,
    name: str,
) -> tuple[int, int, int, int, int, int, int]:
    if not stat.S_ISREG(details.st_mode):
        raise SealingError(f"{name} must be a regular file")
    if details.st_nlink != 1:
        raise SealingError(f"{name} must not be hard-linked")
    if details.st_uid != os.geteuid():
        raise SealingError(f"{name} must be owned by the current user")
    mode = stat.S_IMODE(details.st_mode)
    if mode & 0o022 or mode & 0o7111:
        raise SealingError(f"{name} has an unsafe writable or executable mode")
    return (
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
        details.st_nlink,
        mode,
    )


def _read_descriptor(descriptor: int, expected_size: int, name: str) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < expected_size:
        chunk = os.pread(
            descriptor,
            min(1024 * 1024, expected_size - offset),
            offset,
        )
        if not chunk:
            raise SealingError(f"{name} changed while being read")
        chunks.append(chunk)
        offset += len(chunk)
    extra = os.pread(descriptor, 1, expected_size)
    if extra:
        raise SealingError(f"{name} grew while being read")
    return b"".join(chunks)


def _open_pinned_file(parent_fd: int, name: str, label: str) -> _PinnedFile:
    if Path(name).name != name or name in {"", ".", ".."}:
        raise SealingError(f"{label} has an unsafe member name")
    try:
        descriptor = os.open(name, _file_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise SealingError(
            f"{label} is missing, a symlink, or cannot be opened safely"
        ) from exc
    try:
        before = os.fstat(descriptor)
        state = _regular_file_state(before, label)
        content = _read_descriptor(descriptor, before.st_size, label)
        after_state = _regular_file_state(os.fstat(descriptor), label)
        if after_state != state:
            raise SealingError(f"{label} changed while being read")
        pinned = _PinnedFile(
            parent_fd=parent_fd,
            name=name,
            descriptor=descriptor,
            content=content,
            identity=(before.st_dev, before.st_ino),
            state=state,
        )
        _assert_pinned_file(pinned, label)
        return pinned
    except BaseException:
        os.close(descriptor)
        raise


def _assert_pinned_file(pinned: _PinnedFile, label: str) -> None:
    descriptor_state = _regular_file_state(os.fstat(pinned.descriptor), label)
    try:
        path_details = os.stat(
            pinned.name,
            dir_fd=pinned.parent_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise SealingError(f"{label} changed after descriptor pinning") from exc
    path_state = _regular_file_state(path_details, label)
    if descriptor_state != pinned.state or path_state != pinned.state:
        raise SealingError(f"{label} changed after descriptor pinning")


def _open_path_file(
    path: str | os.PathLike[str],
    label: str,
) -> tuple[Path, int, _PinnedFile]:
    absolute = _safe_absolute_path(path, label)
    if absolute.name in {"", ".", ".."}:
        raise SealingError(f"{label} must name a file")
    parent_path, parent_fd = _open_directory(absolute.parent, f"{label} parent")
    try:
        pinned = _open_pinned_file(parent_fd, absolute.name, label)
    except BaseException:
        os.close(parent_fd)
        raise
    return parent_path / absolute.name, parent_fd, pinned


def _canonical_jsonl(
    content: bytes,
    *,
    name: str,
    parser: Callable[[Mapping[str, Any]], _T],
    identity: Callable[[_T], str],
) -> tuple[_T, ...]:
    lines = content.splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") for line in lines):
        raise SealingError(f"{name} must be non-empty canonical JSONL")
    records: list[_T] = []
    identities: list[str] = []
    for index, line in enumerate(lines):
        try:
            raw = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SealingError(f"{name} line {index} is invalid JSON") from exc
        try:
            canonical = canonical_json_bytes(raw)
        except (TypeError, UnicodeError, ValueError) as exc:
            raise SealingError(f"{name} line {index} is not canonical") from exc
        if not isinstance(raw, Mapping) or canonical != line:
            raise SealingError(f"{name} line {index} is not canonical")
        try:
            record = parser(raw)
        except (TypeError, ValueError) as exc:
            raise SealingError(f"{name} line {index} is invalid: {exc}") from exc
        records.append(record)
        identities.append(identity(record))
    if len(identities) != len(set(identities)):
        raise SealingError(f"{name} contains duplicate identities")
    if tuple(identities) != tuple(sorted(identities)):
        raise SealingError(f"{name} is not strictly ordered")
    return tuple(records)


def _single_pair_value(
    items: tuple[ItemRecord, ItemRecord],
    field: str,
) -> Any:
    values = {getattr(item, field) for item in items}
    if len(values) != 1:
        raise SealingError(f"pair {items[0].pair_id} disagrees on {field}")
    return next(iter(values))


def _validate_model_visible_records(
    *,
    items_content: bytes,
    stores_content: bytes,
) -> _ValidatedModelVisible:
    items = _canonical_jsonl(
        items_content,
        name=ITEMS_NAME,
        parser=ItemRecord.from_dict,
        identity=lambda record: record.item_id,
    )
    stores = _canonical_jsonl(
        stores_content,
        name=STORES_NAME,
        parser=StoreRecord.from_dict,
        identity=lambda record: record.store_id,
    )
    for store in stores:
        addresses = tuple(row.address for row in store.rows)
        if addresses != tuple(sorted(addresses)):
            raise SealingError(
                f"store {store.store_id} rows are not strictly ordered"
            )

    store_by_id = {record.store_id: record for record in stores}
    referenced_store_ids = {item.store_id for item in items}
    if referenced_store_ids != set(store_by_id):
        raise SealingError("item and store registries are not exactly complete")

    pair_rows: dict[str, list[ItemRecord]] = defaultdict(list)
    for item in items:
        pair_rows[item.pair_id].append(item)
        store = store_by_id[item.store_id]
        if item.world_id != store.world_id:
            raise SealingError("item and store world closure is invalid")

    pair_items: dict[str, tuple[ItemRecord, ItemRecord]] = {}
    for pair_id, rows in pair_rows.items():
        if len(rows) != 2 or {row.twin for row in rows} != {
            Twin.ORIGINAL,
            Twin.COUNTERFACTUAL,
        }:
            raise SealingError(
                f"pair {pair_id} must contain exactly both counterfactual twins"
            )
        typed_rows = (rows[0], rows[1])
        for field in (
            "family",
            "stratum",
            "task",
            "path_length",
            "composition_split",
            "composition_id",
            "world_id",
            "memory_mode",
            "control",
        ):
            _single_pair_value(typed_rows, field)
        pair_items[pair_id] = typed_rows

    if {item.world_id for item in items} != {
        store.world_id for store in stores
    }:
        raise SealingError("item and store world registries are not closed")

    coverage_pairs: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    coverage_worlds: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    observed_controls: set[str] = set()
    for pair_id, rows in pair_items.items():
        row = rows[0]
        try:
            control_id = _CONTROL_IDENTITIES[(row.memory_mode, row.control)]
        except KeyError as exc:
            raise SealingError(
                f"pair {pair_id} does not identify one frozen control"
            ) from exc
        key = row.family.value, row.stratum.value, control_id
        coverage_pairs[key].add(pair_id)
        coverage_worlds[key].add(row.world_id)
        observed_controls.add(control_id)

    observed_families = {item.family.value for item in items}
    observed_strata = {item.stratum.value for item in items}
    observed_memory_modes = {item.memory_mode.value for item in items}
    if observed_families != set(REQUIRED_FAMILIES):
        raise SealingError("reasoning-family coverage is not exact")
    if observed_strata != set(REQUIRED_STRATA):
        raise SealingError("stratum coverage is not exact")
    if observed_memory_modes != set(REQUIRED_MEMORY_MODES):
        raise SealingError("memory-mode coverage is not exact")
    if observed_controls != set(REQUIRED_CONTROL_IDS):
        raise SealingError("required control coverage is not exact")

    cells = []
    for family in REQUIRED_FAMILIES:
        for stratum in REQUIRED_STRATA:
            for control_id in REQUIRED_CONTROL_IDS:
                key = family, stratum, control_id
                pair_ids = tuple(sorted(coverage_pairs.get(key, ())))
                if not pair_ids:
                    raise SealingError(
                        "required coverage cell is empty: "
                        f"{family}/{stratum}/{control_id}"
                    )
                world_ids = tuple(sorted(coverage_worlds[key]))
                memory_mode = next(
                    mode.value
                    for (mode, _control), candidate in _CONTROL_IDENTITIES.items()
                    if candidate == control_id
                )
                cells.append(
                    {
                        "family": family,
                        "stratum": stratum,
                        "control_id": control_id,
                        "memory_mode": memory_mode,
                        "item_count": 2 * len(pair_ids),
                        "pair_count": len(pair_ids),
                        "world_count": len(world_ids),
                        "pair_ids": list(pair_ids),
                        "world_ids": list(world_ids),
                    }
                )

    item_ids = tuple(record.item_id for record in items)
    pair_ids = tuple(sorted(pair_items))
    world_ids = tuple(sorted({record.world_id for record in items}))
    store_ids = tuple(record.store_id for record in stores)
    binding = {
        "items": {
            "path": ITEMS_NAME,
            "sha256": hashlib.sha256(items_content).hexdigest(),
            "bytes": len(items_content),
        },
        "stores": {
            "path": STORES_NAME,
            "sha256": hashlib.sha256(stores_content).hexdigest(),
            "bytes": len(stores_content),
        },
        "item_ids": list(item_ids),
        "pair_ids": list(pair_ids),
        "world_ids": list(world_ids),
        "store_ids": list(store_ids),
        "item_count": len(item_ids),
        "pair_count": len(pair_ids),
        "world_count": len(world_ids),
        "store_count": len(store_ids),
        "coverage": {
            "families": list(REQUIRED_FAMILIES),
            "strata": list(REQUIRED_STRATA),
            "memory_modes": list(REQUIRED_MEMORY_MODES),
            "controls": list(REQUIRED_CONTROL_IDS),
            "cells": cells,
        },
    }
    return _ValidatedModelVisible(
        binding=MappingProxyType(binding),
        items=items,
        stores=stores,
        item_count=len(item_ids),
        pair_count=len(pair_ids),
        world_count=len(world_ids),
        store_count=len(store_ids),
    )


def _validate_records(
    *,
    items_content: bytes,
    stores_content: bytes,
    gold_content: bytes,
) -> _ValidatedModelVisible:
    visible = _validate_model_visible_records(
        items_content=items_content,
        stores_content=stores_content,
    )
    gold = _canonical_jsonl(
        gold_content,
        name=SEALED_GOLD_NAME,
        parser=SealedGoldRecord.from_dict,
        identity=lambda record: record.item_id,
    )
    item_by_id = {record.item_id: record for record in visible.items}
    store_by_id = {record.store_id: record for record in visible.stores}
    gold_by_id = {record.item_id: record for record in gold}
    if tuple(item_by_id) != tuple(gold_by_id):
        raise SealingError("item and sealed-gold registries are not complete")
    for item_id, item in item_by_id.items():
        sealed = gold_by_id[item_id]
        store = store_by_id[item.store_id]
        if (
            item.item_id,
            item.pair_id,
            item.twin,
        ) != (
            sealed.item_id,
            sealed.pair_id,
            sealed.twin,
        ):
            raise SealingError("item and sealed-gold identity closure is invalid")
        if sealed.store_sha256 != store.content_sha256:
            raise SealingError("sealed-gold store commitment is invalid")
        try:
            solver = registered_solver(sealed.solver_id)
        except ValueError as exc:
            raise SealingError(f"sealed gold solver is invalid: {exc}") from exc
        first = verify_sealed_gold(item, store, sealed, solver)
        second = verify_sealed_gold(item, store, sealed, solver)
        if first != second:
            raise SealingError("sealed-gold solver verification is nondeterministic")
        if not first.valid:
            raise SealingError(
                "sealed gold answer or proof cannot be reproduced by its "
                f"registered solver for {item.item_id}: {first.reason}"
            )
    return visible


def _validated_release_from_preregistration_sha256(
    *,
    preregistration_sha256: str,
    items_content: bytes,
    stores_content: bytes,
    gold_content: bytes,
) -> _ValidatedRelease:
    if preregistration_sha256 != FROZEN_PREREGISTRATION_SHA256_V3:
        raise SealingError(
            "preregistration does not match the frozen v3 commitment"
        )
    visible = _validate_records(
        items_content=items_content,
        stores_content=stores_content,
        gold_content=gold_content,
    )
    manifest = {
        "record_type": SEALED_RELEASE_SCHEMA,
        "schema_version": STUDY_CONTRACT_VERSION,
        "preregistration": {
            "path": "configs/preregistration-v3.yaml",
            "sha256": preregistration_sha256,
        },
        "model_visible": visible.binding,
        "sealed_gold": {
            "path": SEALED_GOLD_NAME,
            "sha256": hashlib.sha256(gold_content).hexdigest(),
        },
    }
    return _ValidatedRelease(
        manifest=MappingProxyType(manifest),
        manifest_bytes=canonical_json_bytes(manifest),
        item_count=visible.item_count,
        pair_count=visible.pair_count,
        world_count=visible.world_count,
        store_count=visible.store_count,
    )


def _validated_release(
    *,
    preregistration_content: bytes,
    items_content: bytes,
    stores_content: bytes,
    gold_content: bytes,
) -> _ValidatedRelease:
    return _validated_release_from_preregistration_sha256(
        preregistration_sha256=hashlib.sha256(
            preregistration_content
        ).hexdigest(),
        items_content=items_content,
        stores_content=stores_content,
        gold_content=gold_content,
    )


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError("short write while staging sealed release")
        written += count


def _write_file_at(
    directory_fd: int,
    name: str,
    content: bytes,
    mode: int,
) -> _PinnedFile:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        _write_all(descriptor, content)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        details = os.fstat(descriptor)
        state = _regular_file_state(details, f"staged {name}")
        pinned = _PinnedFile(
            parent_fd=directory_fd,
            name=name,
            descriptor=descriptor,
            content=content,
            identity=(details.st_dev, details.st_ino),
            state=state,
        )
        _assert_staged_file(pinned)
        return pinned
    except BaseException:
        os.close(descriptor)
        try:
            os.unlink(name, dir_fd=directory_fd)
        except OSError:
            pass
        raise


def _assert_staged_file(pinned: _PinnedFile) -> None:
    _assert_pinned_file(pinned, f"staged {pinned.name}")
    details = os.fstat(pinned.descriptor)
    expected_mode = _RELEASE_MODES[pinned.name]
    if stat.S_IMODE(details.st_mode) != expected_mode:
        raise SealingError(f"staged {pinned.name} has an unsafe mode")
    observed = _read_descriptor(
        pinned.descriptor,
        details.st_size,
        f"staged {pinned.name}",
    )
    if observed != pinned.content:
        raise SealingError(f"staged {pinned.name} content verification failed")
    _assert_pinned_file(pinned, f"staged {pinned.name}")


def _assert_directory_entry(
    parent_fd: int,
    name: str,
    descriptor: int,
    label: str,
) -> None:
    pinned = os.fstat(descriptor)
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise SealingError(f"{label} changed after descriptor pinning") from exc
    if (
        not stat.S_ISDIR(pinned.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != (pinned.st_dev, pinned.st_ino)
        or current.st_uid != os.geteuid()
        or stat.S_IMODE(current.st_mode) != 0o700
    ):
        raise SealingError(f"{label} was replaced or has an unsafe mode")


def _make_staging(
    output_fd: int,
    release_id: str,
) -> tuple[str, int]:
    for _ in range(128):
        name = f".{release_id}.{secrets.token_hex(8)}.staging"
        try:
            os.mkdir(name, 0o700, dir_fd=output_fd)
        except FileExistsError:
            continue
        descriptor: int | None = None
        try:
            descriptor = os.open(name, _directory_flags(), dir_fd=output_fd)
            _assert_directory_entry(
                output_fd,
                name,
                descriptor,
                "private release staging",
            )
            return name, descriptor
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.rmdir(name, dir_fd=output_fd)
            except OSError:
                pass
            raise
    raise SealingError("cannot allocate private release staging")


def _remove_staging(
    output_fd: int,
    staging_name: str,
    staging_fd: int,
) -> None:
    try:
        _assert_directory_entry(
            output_fd,
            staging_name,
            staging_fd,
            "private release staging",
        )
    except SealingError:
        return
    try:
        names = os.listdir(staging_fd)
    except OSError:
        return
    for name in names:
        try:
            details = os.stat(name, dir_fd=staging_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.geteuid()
            or details.st_nlink != 1
        ):
            return
        try:
            os.unlink(name, dir_fd=staging_fd)
        except OSError:
            return
    try:
        os.rmdir(staging_name, dir_fd=output_fd)
        os.fsync(output_fd)
    except OSError:
        return


def _stage_release(
    output_fd: int,
    *,
    release_id: str,
    contents: Mapping[str, bytes],
) -> _StagedRelease:
    if tuple(contents) != _RELEASE_NAMES:
        raise SealingError("sealed release staging member order is not exact")
    staging_name, staging_fd = _make_staging(output_fd, release_id)
    files: list[_PinnedFile] = []
    try:
        for name in _RELEASE_NAMES:
            files.append(
                _write_file_at(
                    staging_fd,
                    name,
                    contents[name],
                    _RELEASE_MODES[name],
                )
            )
        if tuple(sorted(os.listdir(staging_fd))) != tuple(sorted(_RELEASE_NAMES)):
            raise SealingError("private release staging has unexpected entries")
        for pinned in files:
            _assert_staged_file(pinned)
        _assert_directory_entry(
            output_fd,
            staging_name,
            staging_fd,
            "private release staging",
        )
        os.fsync(staging_fd)
        return _StagedRelease(staging_name, staging_fd, tuple(files))
    except BaseException:
        for pinned in reversed(files):
            try:
                os.close(pinned.descriptor)
            except OSError:
                pass
        _remove_staging(output_fd, staging_name, staging_fd)
        os.close(staging_fd)
        raise


def _close_staged(staged: _StagedRelease) -> None:
    for pinned in reversed(staged.files):
        try:
            os.close(pinned.descriptor)
        except OSError:
            pass
    try:
        os.close(staged.descriptor)
    except OSError:
        pass


def _lock_output(output_fd: int) -> None:
    _assert_owned_directory(os.fstat(output_fd), "output root")
    try:
        fcntl.flock(output_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise SealingError("output root is locked by another publisher") from exc
    _assert_owned_directory(os.fstat(output_fd), "output root")


def _reject_existing_release(output_fd: int, release_id: str) -> None:
    try:
        os.stat(release_id, dir_fd=output_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SealingError("release destination cannot be inspected safely") from exc
    raise SealingError(
        "content-addressed release directory already exists; "
        "use explicit verification instead of rerunning publication",
        code="RELEASE_EXISTS",
    )


def _rename_noreplace_at(
    directory_fd: int,
    source_name: str,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
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
            source,
            directory_fd,
            destination,
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
            source,
            directory_fd,
            destination,
            0x00000001,
        )
    else:
        raise SealingError(
            "platform lacks atomic no-replace directory publication"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise SealingError(
            "content-addressed release directory already exists",
            code="RELEASE_EXISTS",
        )
    raise SealingError(
        "atomic no-replace release publication failed",
        code="ATOMIC_PUBLICATION_FAILED",
    )


def _verify_staged_release(
    parent_fd: int,
    directory_name: str,
    staged: _StagedRelease,
) -> None:
    _assert_directory_entry(
        parent_fd,
        directory_name,
        staged.descriptor,
        "installed sealed release",
    )
    if tuple(sorted(os.listdir(staged.descriptor))) != tuple(
        sorted(_RELEASE_NAMES)
    ):
        raise SealingError("installed sealed release has unexpected entries")
    for pinned in staged.files:
        _assert_staged_file(pinned)
    os.fsync(staged.descriptor)


def _quarantine_installed(
    output_fd: int,
    release_id: str,
    staged: _StagedRelease,
) -> None:
    for _ in range(128):
        quarantine = f".{release_id}.{secrets.token_hex(8)}.quarantine"
        try:
            _rename_noreplace_at(
                output_fd,
                release_id,
                quarantine,
            )
        except SealingError as exc:
            if exc.code == "RELEASE_EXISTS":
                continue
            return
        _remove_staging(output_fd, quarantine, staged.descriptor)
        return


def _publish_staged(
    output_fd: int,
    *,
    release_id: str,
    staged: _StagedRelease,
) -> None:
    _assert_directory_entry(
        output_fd,
        staged.name,
        staged.descriptor,
        "private release staging",
    )
    for pinned in staged.files:
        _assert_staged_file(pinned)
    _rename_noreplace_at(output_fd, staged.name, release_id)
    try:
        _verify_staged_release(output_fd, release_id, staged)
        os.fsync(output_fd)
    except BaseException:
        _quarantine_installed(output_fd, release_id, staged)
        try:
            os.fsync(output_fd)
        except OSError:
            pass
        raise


def _sha256_commitment(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SealingError(f"{name} must be a lowercase SHA-256 commitment")
    return value


def _strict_mapping(
    value: object,
    expected: frozenset[str],
    name: str,
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise SealingError(f"{name} fields are not exact")
    return value


def _parse_manifest(content: bytes) -> Mapping[str, Any]:
    try:
        raw = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SealingError("sealed-release manifest is invalid JSON") from exc
    try:
        canonical = canonical_json_bytes(raw)
    except (TypeError, UnicodeError, ValueError) as exc:
        raise SealingError("sealed-release manifest is not canonical") from exc
    if not isinstance(raw, Mapping) or canonical != content:
        raise SealingError("sealed-release manifest is not canonical")
    manifest = _strict_mapping(
        raw,
        frozenset(
            {
                "record_type",
                "schema_version",
                "preregistration",
                "model_visible",
                "sealed_gold",
            }
        ),
        "sealed-release manifest",
    )
    if manifest["record_type"] != SEALED_RELEASE_SCHEMA:
        raise SealingError("sealed-release manifest record_type is invalid")
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != STUDY_CONTRACT_VERSION
    ):
        raise SealingError("sealed-release manifest schema_version is invalid")
    preregistration = _strict_mapping(
        manifest["preregistration"],
        frozenset({"path", "sha256"}),
        "preregistration commitment",
    )
    if preregistration["path"] != "configs/preregistration-v3.yaml":
        raise SealingError("preregistration logical path is invalid")
    if (
        _sha256_commitment(
            preregistration["sha256"],
            "preregistration SHA-256",
        )
        != FROZEN_PREREGISTRATION_SHA256_V3
    ):
        raise SealingError("preregistration commitment is not the frozen v3 hash")
    sealed_gold = _strict_mapping(
        manifest["sealed_gold"],
        frozenset({"path", "sha256"}),
        "sealed-gold commitment",
    )
    if sealed_gold["path"] != SEALED_GOLD_NAME:
        raise SealingError("sealed-gold logical path is invalid")
    _sha256_commitment(sealed_gold["sha256"], "sealed-gold SHA-256")
    if not isinstance(manifest["model_visible"], Mapping):
        raise SealingError("model-visible binding must be an object")
    return manifest


def _assert_release_file_mode(pinned: _PinnedFile) -> None:
    expected = _RELEASE_MODES[pinned.name]
    actual = stat.S_IMODE(os.fstat(pinned.descriptor).st_mode)
    if actual != expected:
        raise SealingError(
            f"{pinned.name} mode is not the frozen release mode"
        )


def _manifest_artifact_sha256(
    manifest: Mapping[str, Any],
    name: str,
) -> str:
    if name == SEALED_GOLD_NAME:
        return _sha256_commitment(
            manifest["sealed_gold"]["sha256"],
            "sealed-gold SHA-256",
        )
    key = "items" if name == ITEMS_NAME else "stores"
    model_visible = manifest["model_visible"]
    if not isinstance(model_visible, Mapping):
        raise SealingError("model-visible binding must be an object")
    artifact = _strict_mapping(
        model_visible.get(key),
        frozenset({"path", "sha256", "bytes"}),
        f"model-visible {key} binding",
    )
    if artifact["path"] != name:
        raise SealingError(f"model-visible {key} logical path is invalid")
    if type(artifact["bytes"]) is not int or artifact["bytes"] < 1:
        raise SealingError(f"model-visible {key} byte count is invalid")
    return _sha256_commitment(
        artifact["sha256"],
        f"model-visible {key} SHA-256",
    )


def _assert_manifest_artifact_hash(
    manifest: Mapping[str, Any],
    pinned: _PinnedFile,
) -> None:
    expected = _manifest_artifact_sha256(manifest, pinned.name)
    if hashlib.sha256(pinned.content).hexdigest() != expected:
        raise SealingError(
            f"{pinned.name} hash disagrees with its manifest commitment"
        )


def _unopened_release_file_state(
    release_fd: int,
    name: str,
) -> tuple[int, int, int, int, int, int, int]:
    try:
        details = os.stat(name, dir_fd=release_fd, follow_symlinks=False)
    except OSError as exc:
        raise SealingError(
            f"{name} is missing or cannot be inspected safely"
        ) from exc
    state = _regular_file_state(details, name)
    if stat.S_IMODE(details.st_mode) != _RELEASE_MODES[name]:
        raise SealingError(f"{name} mode is not the frozen release mode")
    return state


def _open_release_manifest(
    *,
    release_dir: str | os.PathLike[str],
    expected_release_sha256: object,
) -> tuple[Path, int, _PinnedFile, Mapping[str, Any], str]:
    expected = _sha256_commitment(
        expected_release_sha256,
        "external sealed-release commitment",
    )
    release_path, release_fd = _open_directory(
        release_dir,
        "sealed release directory",
    )
    try:
        if stat.S_IMODE(os.fstat(release_fd).st_mode) != 0o700:
            raise SealingError(
                "sealed release directory mode is not owner-only"
            )
        if set(os.listdir(release_fd)) != set(_RELEASE_NAMES):
            raise SealingError("sealed release directory entries are not exact")
        manifest_file = _open_pinned_file(
            release_fd,
            SEALED_RELEASE_MANIFEST,
            SEALED_RELEASE_MANIFEST,
        )
        try:
            _assert_release_file_mode(manifest_file)
            actual = hashlib.sha256(manifest_file.content).hexdigest()
            if actual != expected:
                raise SealingError(
                    "sealed-release manifest disagrees with the external "
                    "commitment"
                )
            if release_path.name != RELEASE_ID_PREFIX + actual:
                raise SealingError(
                    "sealed release directory is not the committed "
                    "content-address"
                )
            manifest = _parse_manifest(manifest_file.content)
            _assert_pinned_file(manifest_file, SEALED_RELEASE_MANIFEST)
            _assert_directory_path(
                release_path,
                release_fd,
                "sealed release directory",
            )
            return release_path, release_fd, manifest_file, manifest, actual
        except BaseException:
            os.close(manifest_file.descriptor)
            raise
    except BaseException:
        os.close(release_fd)
        raise


def _close_release_files(
    release_fd: int,
    files: list[_PinnedFile],
) -> None:
    for pinned in reversed(files):
        try:
            os.close(pinned.descriptor)
        except OSError:
            pass
    os.close(release_fd)


def preflight_model_visible_release(
    *,
    release_dir: str | os.PathLike[str],
    expected_release_sha256: str,
) -> ModelVisiblePreflightResult:
    """Validate only committed model-visible bytes, never opening sealed gold."""

    (
        release_path,
        release_fd,
        manifest_file,
        manifest,
        release_sha256,
    ) = _open_release_manifest(
        release_dir=release_dir,
        expected_release_sha256=expected_release_sha256,
    )
    files = [manifest_file]
    try:
        gold_state = _unopened_release_file_state(
            release_fd,
            SEALED_GOLD_NAME,
        )
        items = _open_pinned_file(release_fd, ITEMS_NAME, ITEMS_NAME)
        files.append(items)
        stores = _open_pinned_file(release_fd, STORES_NAME, STORES_NAME)
        files.append(stores)
        _assert_release_file_mode(items)
        _assert_release_file_mode(stores)
        _assert_manifest_artifact_hash(manifest, items)
        _assert_manifest_artifact_hash(manifest, stores)
        visible = _validate_model_visible_records(
            items_content=items.content,
            stores_content=stores.content,
        )
        if manifest["model_visible"] != visible.binding:
            raise SealingError(
                "model-visible artifacts disagree with their manifest commitment"
            )
        for pinned in files:
            _assert_pinned_file(pinned, pinned.name)
        if (
            _unopened_release_file_state(release_fd, SEALED_GOLD_NAME)
            != gold_state
        ):
            raise SealingError(
                "sealed-gold metadata changed during model-visible preflight"
            )
        _assert_directory_path(
            release_path,
            release_fd,
            "sealed release directory",
        )
        sealed_gold = manifest["sealed_gold"]
        return ModelVisiblePreflightResult(
            release_dir=release_path,
            release_sha256=release_sha256,
            item_count=visible.item_count,
            pair_count=visible.pair_count,
            world_count=visible.world_count,
            store_count=visible.store_count,
            sealed_gold_sha256=sealed_gold["sha256"],
        )
    finally:
        _close_release_files(release_fd, files)


def verify_release(
    *,
    release_dir: str | os.PathLike[str],
    expected_release_sha256: str,
) -> VerifiedSealedRelease:
    """Open sealed gold and deterministically replay a published release."""

    (
        release_path,
        release_fd,
        manifest_file,
        manifest,
        release_sha256,
    ) = _open_release_manifest(
        release_dir=release_dir,
        expected_release_sha256=expected_release_sha256,
    )
    files = [manifest_file]
    try:
        for name in _SOURCE_NAMES:
            pinned = _open_pinned_file(release_fd, name, name)
            _assert_release_file_mode(pinned)
            _assert_manifest_artifact_hash(manifest, pinned)
            files.append(pinned)
        contents = {pinned.name: pinned.content for pinned in files[1:]}
        validated = _validated_release_from_preregistration_sha256(
            preregistration_sha256=FROZEN_PREREGISTRATION_SHA256_V3,
            items_content=contents[ITEMS_NAME],
            stores_content=contents[STORES_NAME],
            gold_content=contents[SEALED_GOLD_NAME],
        )
        if validated.manifest_bytes != manifest_file.content:
            raise SealingError(
                "sealed release artifacts disagree with the manifest commitment"
            )
        for pinned in files:
            _assert_pinned_file(pinned, pinned.name)
        _assert_directory_path(
            release_path,
            release_fd,
            "sealed release directory",
        )
        sealed_gold = manifest["sealed_gold"]
        return VerifiedSealedRelease(
            release_dir=release_path,
            release_sha256=release_sha256,
            manifest_bytes=manifest_file.content,
            item_count=validated.item_count,
            pair_count=validated.pair_count,
            world_count=validated.world_count,
            store_count=validated.store_count,
            sealed_gold_sha256=sealed_gold["sha256"],
        )
    finally:
        _close_release_files(release_fd, files)


def seal_release(
    *,
    source_dir: str | os.PathLike[str],
    preregistration_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    apply: bool = False,
) -> SealedReleaseResult:
    """Validate exact v2 records and plan or publish one v3 release."""

    source_path, source_fd = _open_directory(source_dir, "source directory")
    preregistration_parent_fd: int | None = None
    inputs: list[_PinnedFile] = []
    try:
        for name in _SOURCE_NAMES:
            inputs.append(_open_pinned_file(source_fd, name, name))
        (
            _preregistration_path,
            preregistration_parent_fd,
            preregistration,
        ) = _open_path_file(
            preregistration_path,
            "frozen preregistration",
        )
        inputs.append(preregistration)
        identities = [pinned.identity for pinned in inputs]
        if len(set(identities)) != len(identities):
            raise SealingError("release inputs must be distinct physical files")

        content = {pinned.name: pinned.content for pinned in inputs[:3]}
        validated = _validated_release(
            preregistration_content=preregistration.content,
            items_content=content[ITEMS_NAME],
            stores_content=content[STORES_NAME],
            gold_content=content[SEALED_GOLD_NAME],
        )
        for pinned in inputs:
            _assert_pinned_file(pinned, pinned.name)
        _assert_directory_path(source_path, source_fd, "source directory")

        release_sha256 = hashlib.sha256(validated.manifest_bytes).hexdigest()
        release_id = RELEASE_ID_PREFIX + release_sha256
        output = _safe_absolute_path(output_root, "output root")
        _validate_planned_directory_path(output, "output root")
        result = SealedReleaseResult(
            release_id=release_id,
            release_dir=output / release_id,
            release_sha256=release_sha256,
            manifest_bytes=validated.manifest_bytes,
            item_count=validated.item_count,
            pair_count=validated.pair_count,
            world_count=validated.world_count,
            store_count=validated.store_count,
            published=False,
        )
        if not apply:
            return result

        output_path, output_fd = _open_directory(output, "output root")
        staged: _StagedRelease | None = None
        published = False
        try:
            _lock_output(output_fd)
            _reject_existing_release(output_fd, release_id)
            staged = _stage_release(
                output_fd,
                release_id=release_id,
                contents={
                    ITEMS_NAME: content[ITEMS_NAME],
                    STORES_NAME: content[STORES_NAME],
                    SEALED_GOLD_NAME: content[SEALED_GOLD_NAME],
                    SEALED_RELEASE_MANIFEST: validated.manifest_bytes,
                },
            )
            for pinned in inputs:
                _assert_pinned_file(pinned, pinned.name)
            _assert_directory_path(source_path, source_fd, "source directory")
            _assert_directory_path(output_path, output_fd, "output root")
            _publish_staged(
                output_fd,
                release_id=release_id,
                staged=staged,
            )
            published = True
            return replace(result, published=True)
        finally:
            if staged is not None:
                if not published:
                    _remove_staging(
                        output_fd,
                        staged.name,
                        staged.descriptor,
                    )
                _close_staged(staged)
            os.close(output_fd)
    finally:
        for pinned in reversed(inputs):
            try:
                os.close(pinned.descriptor)
            except OSError:
                pass
        if preregistration_parent_fd is not None:
            os.close(preregistration_parent_fd)
        os.close(source_fd)
