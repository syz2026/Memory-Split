"""Rerendering, packing, receipts, resumability, and atomic publication."""

from __future__ import annotations

import hashlib
import json
import os
import fcntl
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from .canonical import canonical_json_bytes, sha256_hex
from .catalog import InputCatalog
from .integrity import ordered_stream_commitments
from .metadata import (
    MetadataRecord,
    Renderer,
    metadata_from_bytes,
    metadata_to_bytes,
    reduce_metadata,
    render_metadata,
    token_bytes,
)
from .schedule import (
    ScheduleRecord,
    ShardAssignment,
    assignments_from_bytes,
    assignments_to_bytes,
    assign_update_aligned_shards,
    largest_deficit_schedule,
    schedule_from_bytes,
    schedule_to_bytes,
)
from .safeio import (
    AtomicFileWriter,
    RetainedTombstoneInventory,
    RetainedTombstoneStore,
    atomic_rename_noreplace,
    atomic_write_or_match,
    clean_owned_temporaries,
    entry_exists,
    entry_lstat,
    fsync_directory,
    is_owned_temporary,
    list_entries,
    open_directory_at,
    open_parent_directory,
    open_regular_file_at,
    open_tombstone_directory,
    read_file_descriptor,
    read_regular_file,
    regular_file_digest,
    retained_tombstone_inventory_fd,
    unlink_regular_if_matches,
)

if TYPE_CHECKING:
    from .tasks import TaskResult

_FORMAT = "memorysplit-parallel-corpus-v1"
_FORMAT_V2 = "memorysplit-parallel-corpus-v2"
_COMPILER_VERSION = "metadata-first-foundation-v1"
_STAGE_OWNER_NAME = ".parallel-owner.json"
_TARGET_WEIGHT_NAMES = (
    "dense_target_weights",
    "split90_target_weights",
)
_FOUNDATION_NAMES = {
    "assignments.jsonl",
    "catalog.jsonl",
    "metadata.jsonl",
    "schedule.jsonl",
}
_RECEIPT_FIELDS = {
    "artifacts",
    "assignments_sha256",
    "build_id",
    "catalog_sha256",
    "compiler_version",
    "config",
    "format",
    "logical_tokens",
    "merkle_root_sha256",
    "metadata_sha256",
    "ordered_stream_sha256",
    "packed_stream_sha256",
    "packed_tokens",
    "padding_tokens",
    "record_count",
    "renderer_id",
    "schedule_sha256",
    "shard_count",
}
_ARTIFACT_FIELDS = {"bytes", "path", "sha256"}
_SIDECAR_SET_FIELDS = {
    "artifacts",
    "dtype",
    "items",
    "name",
    "stream_sha256",
}
_RECEIPT_COUNT_FIELDS = {
    "logical_tokens",
    "packed_tokens",
    "padding_tokens",
    "record_count",
    "shard_count",
}
_RECEIPT_NAMESPACE_KIND = object()


class VerifiedParallelCorpus(dict[str, Any]):
    """Canonical receipt bytes plus out-of-band cleanup visibility."""

    def __init__(
        self,
        receipt: dict[str, Any],
        retained_tombstones: RetainedTombstoneInventory,
    ) -> None:
        super().__init__(receipt)
        self.retained_tombstones = retained_tombstones


def _with_retained_tombstones(
    receipt: dict[str, Any],
    parent_fd: int,
) -> VerifiedParallelCorpus:
    return VerifiedParallelCorpus(
        receipt,
        retained_tombstone_inventory_fd(parent_fd),
    )


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True)
class ParallelBuildConfig:
    lane_weights: tuple[tuple[str, int], ...]
    update_tokens: int
    shard_count: int = 32
    allow_fewer_shards: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.lane_weights, tuple) or not self.lane_weights:
            raise ValueError("lane_weights must be a non-empty tuple")
        lanes = []
        for item in self.lane_weights:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
            ):
                raise ValueError("lane_weights entries must be (lane, weight)")
            _positive_integer(item[1], "lane weight")
            lanes.append(item[0])
        if len(lanes) != len(set(lanes)):
            raise ValueError("lane_weights contain duplicate lanes")
        _positive_integer(self.update_tokens, "update_tokens")
        _positive_integer(self.shard_count, "shard_count")
        if not isinstance(self.allow_fewer_shards, bool):
            raise ValueError("allow_fewer_shards must be boolean")

    def as_dict(self) -> dict[str, object]:
        return {
            "allow_fewer_shards": self.allow_fewer_shards,
            "lane_weights": [
                {"lane": lane, "weight": weight}
                for lane, weight in self.lane_weights
            ],
            "shard_count": self.shard_count,
            "update_tokens": self.update_tokens,
        }

    @classmethod
    def from_dict(cls, value: object) -> ParallelBuildConfig:
        required = {
            "allow_fewer_shards",
            "lane_weights",
            "shard_count",
            "update_tokens",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("parallel build config fields do not match the contract")
        raw_weights = value["lane_weights"]
        if not isinstance(raw_weights, list):
            raise ValueError("lane_weights must be a list")
        weights = []
        for item in raw_weights:
            if not isinstance(item, dict) or set(item) != {"lane", "weight"}:
                raise ValueError("lane weight fields do not match the contract")
            weights.append((item["lane"], item["weight"]))
        return cls(
            lane_weights=tuple(weights),
            update_tokens=value["update_tokens"],
            shard_count=value["shard_count"],
            allow_fewer_shards=value["allow_fewer_shards"],
        )


def parallel_build_id(
    catalog: InputCatalog,
    renderer_id: str,
    config: ParallelBuildConfig,
) -> str:
    return _parallel_build_id(
        catalog,
        renderer_id,
        config,
        publication_format=_FORMAT,
        sidecar_commitments=(),
    )

def _parallel_build_id(
    catalog: InputCatalog,
    renderer_id: str,
    config: ParallelBuildConfig,
    *,
    publication_format: str,
    sidecar_commitments: tuple[dict[str, object], ...],
) -> str:
    if not isinstance(renderer_id, str) or not renderer_id:
        raise ValueError("renderer_id must be non-empty")
    identity = {
        "catalog_sha256": catalog.sha256,
        "compiler_version": _COMPILER_VERSION,
        "config": config.as_dict(),
        "format": publication_format,
        "renderer_id": renderer_id,
    }
    if publication_format == _FORMAT_V2:
        identity["sidecar_sources"] = list(sidecar_commitments)
    return sha256_hex(
        canonical_json_bytes(identity)
    )

@dataclass
class _PinnedSidecarSource:
    name: str
    path: Path
    descriptor: int
    items: int
    sha256: str

    @property
    def commitment(self) -> dict[str, object]:
        return {
            "dtype": "uint8",
            "items": self.items,
            "name": self.name,
            "sha256": self.sha256,
        }

    def read(self, offset: int, length: int) -> bytes:
        payload = os.pread(self.descriptor, length, offset)
        if len(payload) != length:
            raise ValueError(f"sidecar source changed length while pinned: {self.name}")
        return payload

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

def _open_sidecar_sources(
    sidecar_paths: dict[str, Path | str] | None,
) -> tuple[_PinnedSidecarSource, ...]:
    if sidecar_paths is None:
        return ()
    if type(sidecar_paths) is not dict or tuple(sorted(sidecar_paths)) != (
        _TARGET_WEIGHT_NAMES
    ):
        raise ValueError(
            "sidecar_paths must contain exactly dense_target_weights and "
            "split90_target_weights"
        )
    sources = []
    try:
        for name in _TARGET_WEIGHT_NAMES:
            raw_path = sidecar_paths[name]
            if not isinstance(raw_path, (str, Path)):
                raise ValueError(f"sidecar path must be a path string: {name}")
            path = Path(raw_path)
            parent_fd, entry_name = open_parent_directory(path)
            try:
                descriptor, metadata = open_regular_file_at(parent_fd, entry_name)
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"sidecar source is missing, symlinked, or unsafe: {name}"
                ) from error
            finally:
                os.close(parent_fd)
            digest = hashlib.sha256()
            offset = 0
            dense_all_one = True
            try:
                while offset < metadata.st_size:
                    chunk = os.pread(
                        descriptor,
                        min(1 << 20, metadata.st_size - offset),
                        offset,
                    )
                    if not chunk:
                        raise ValueError(
                            f"sidecar source changed length while pinned: {name}"
                        )
                    if any(value not in (0, 1) for value in chunk):
                        raise ValueError(
                            f"sidecar source contains non-binary weights: {name}"
                        )
                    dense_all_one &= all(value == 1 for value in chunk)
                    digest.update(chunk)
                    offset += len(chunk)
                if metadata.st_size <= 0:
                    raise ValueError(f"sidecar source must not be empty: {name}")
                if name == "dense_target_weights" and not dense_all_one:
                    raise ValueError("dense_target_weights must be one at every target")
                sources.append(
                    _PinnedSidecarSource(
                        name=name,
                        path=path,
                        descriptor=descriptor,
                        items=metadata.st_size,
                        sha256=digest.hexdigest(),
                    )
                )
            except BaseException:
                os.close(descriptor)
                raise
        return tuple(sources)
    except BaseException:
        for source in sources:
            source.close()
        raise


def publication_staging_path(
    destination: Path | str,
    build_id: str,
) -> Path:
    if (
        not isinstance(build_id, str)
        or len(build_id) != 64
        or any(char not in "0123456789abcdef" for char in build_id)
    ):
        raise ValueError("build_id must be lowercase SHA-256")
    path = Path(destination)
    if not path.name:
        raise ValueError("destination must name a directory")
    return path.parent / f".{path.name}.parallel-work-{build_id[:16]}"


class _ShardSink:
    def __init__(
        self,
        shards_fd: int,
        assignment: ShardAssignment,
        *,
        owner: str,
        tombstone_fd: RetainedTombstoneStore,
    ) -> None:
        self.assignment = assignment
        self.final_name = f"{assignment.shard_id}.bin"
        self.writer = AtomicFileWriter(
            shards_fd,
            self.final_name,
            owner=owner,
            tombstone_fd=tombstone_fd,
        )
        self.digest = hashlib.sha256()
        self.byte_count = 0

    def write(self, payload: bytes) -> None:
        if len(payload) % 2:
            raise ValueError("packed uint16 payload has odd byte length")
        self.writer.write(payload)
        self.digest.update(payload)
        self.byte_count += len(payload)

    def finish(self) -> dict[str, object]:
        expected = (self.assignment.token_end - self.assignment.token_start) * 2
        if self.byte_count != expected:
            self.abort()
            raise ValueError(f"shard byte count drift: {self.assignment.shard_id}")
        digest = self.digest.hexdigest()
        self.writer.finish(
            expected_bytes=expected,
            expected_sha256=digest,
        )
        return {
            "bytes": expected,
            "path": f"shards/{self.final_name}",
            "sha256": digest,
        }

    def abort(self) -> None:
        self.writer.abort()


class _SidecarShardSink:
    def __init__(
        self,
        sidecar_fd: int,
        assignment: ShardAssignment,
        *,
        sidecar_name: str,
        owner: str,
        tombstone_fd: RetainedTombstoneStore,
    ) -> None:
        self.assignment = assignment
        self.sidecar_name = sidecar_name
        self.final_name = f"{assignment.shard_id}.bin"
        self.writer = AtomicFileWriter(
            sidecar_fd,
            self.final_name,
            owner=owner,
            tombstone_fd=tombstone_fd,
        )
        self.digest = hashlib.sha256()
        self.byte_count = 0

    def write(self, payload: bytes) -> None:
        self.writer.write(payload)
        self.digest.update(payload)
        self.byte_count += len(payload)

    def finish(self) -> dict[str, object]:
        expected = self.assignment.token_end - self.assignment.token_start
        if self.byte_count != expected:
            self.abort()
            raise ValueError(
                f"sidecar shard byte count drift: {self.sidecar_name}/"
                f"{self.assignment.shard_id}"
            )
        digest = self.digest.hexdigest()
        self.writer.finish(
            expected_bytes=expected,
            expected_sha256=digest,
        )
        return {
            "bytes": expected,
            "path": (
                f"sidecars/{self.sidecar_name}/{self.final_name}"
            ),
            "sha256": digest,
        }

    def abort(self) -> None:
        self.writer.abort()

def _pack_sidecar_sets(
    sources: tuple[_PinnedSidecarSource, ...],
    assignments: tuple[ShardAssignment, ...],
    sidecar_fds: dict[str, int],
    *,
    logical_tokens: int,
    packed_tokens: int,
    owner: str,
    tombstone_fd: RetainedTombstoneStore,
) -> list[dict[str, object]]:
    if not sources:
        return []
    for source in sources:
        if source.items != logical_tokens:
            raise ValueError(
                f"sidecar source item count does not match logical tokens: "
                f"{source.name}"
            )

    records = []
    for source in sources:
        stream_digest = hashlib.sha256()
        artifacts = []
        position = 0
        for assignment in assignments:
            sink = _SidecarShardSink(
                sidecar_fds[source.name],
                assignment,
                sidecar_name=source.name,
                owner=owner,
                tombstone_fd=tombstone_fd,
            )
            remaining = assignment.token_end - assignment.token_start
            try:
                while remaining:
                    if position < logical_tokens:
                        take = min(
                            remaining,
                            logical_tokens - position,
                            1 << 20,
                        )
                        chunk = source.read(position, take)
                    else:
                        take = min(remaining, 1 << 20)
                        chunk = bytes(take)
                    sink.write(chunk)
                    stream_digest.update(chunk)
                    position += take
                    remaining -= take
                artifacts.append(sink.finish())
            except BaseException:
                sink.abort()
                raise
        if position != packed_tokens:
            raise ValueError(f"sidecar packed item count drift: {source.name}")
        records.append(
            {
                "artifacts": artifacts,
                "dtype": "uint8",
                "items": packed_tokens,
                "name": source.name,
                "stream_sha256": stream_digest.hexdigest(),
            }
        )
    return records

def _rerender_and_pack_pinned(
    catalog: InputCatalog,
    metadata: tuple[MetadataRecord, ...],
    schedule: tuple[ScheduleRecord, ...],
    assignments: tuple[ShardAssignment, ...],
    renderer: Renderer,
    shards_fd: int,
    *,
    owner: str,
    tombstone_fd: RetainedTombstoneStore,
    cached_payloads: dict[str, bytes] | None = None,
) -> dict[str, object]:
    reduced = reduce_metadata(
        catalog,
        metadata,
        expected_renderer_id=renderer.renderer_id,
    )
    schedule_to_bytes(schedule)
    assignments_to_bytes(assignments)
    ordered_stream_commitments(schedule, reduced)
    by_catalog_id = {record.record_id: record for record in catalog.records}
    by_metadata_id = {record.record_id: record for record in reduced}
    if cached_payloads is not None and set(cached_payloads) != set(by_metadata_id):
        raise ValueError("cached payload namespace does not match metadata")
    if not assignments or assignments[0].token_start != 0:
        raise ValueError("pack assignments must cover the stream from zero")
    logical_tokens = schedule[-1].token_end
    if assignments[-1].token_end < logical_tokens:
        raise ValueError("pack assignments do not cover the logical stream")

    packed_digest = hashlib.sha256()
    shard_artifacts = []
    assignment_index = 0
    sink = _ShardSink(
        shards_fd,
        assignments[assignment_index],
        owner=owner,
        tombstone_fd=tombstone_fd,
    )
    token_position = 0

    def write_packed(payload: bytes) -> None:
        nonlocal assignment_index, sink, token_position
        token_count = len(payload) // 2
        byte_offset = 0
        while token_count:
            assignment = assignments[assignment_index]
            available = assignment.token_end - token_position
            if available <= 0:
                shard_artifacts.append(sink.finish())
                assignment_index += 1
                if assignment_index >= len(assignments):
                    raise ValueError("packed stream exceeds shard assignments")
                sink = _ShardSink(
                    shards_fd,
                    assignments[assignment_index],
                    owner=owner,
                    tombstone_fd=tombstone_fd,
                )
                continue
            take = min(available, token_count)
            chunk = payload[byte_offset : byte_offset + take * 2]
            sink.write(chunk)
            packed_digest.update(chunk)
            token_position += take
            token_count -= take
            byte_offset += take * 2

    try:
        for entry in schedule:
            source = by_catalog_id[entry.record_id]
            expected = by_metadata_id[entry.record_id]
            if source.payload_sha256 != expected.source_sha256:
                raise ValueError(f"source digest drift: {entry.record_id}")
            if cached_payloads is None:
                rendered = renderer.render(source)
                payload = token_bytes(rendered.token_ids)
                if (
                    len(rendered.token_ids) != expected.token_length
                    or rendered.flags != expected.flags
                    or sha256_hex(payload) != expected.render_sha256
                ):
                    raise ValueError(f"rerender drift: {entry.record_id}")
            else:
                payload = cached_payloads[entry.record_id]
                if (
                    not isinstance(payload, bytes)
                    or len(payload) != expected.token_length * 2
                    or sha256_hex(payload) != expected.render_sha256
                ):
                    raise ValueError(f"cached payload drift: {entry.record_id}")
            write_packed(payload)
        if token_position != logical_tokens:
            raise ValueError("rerendered logical token count drift")
        padding_tokens = assignments[-1].token_end - token_position
        if padding_tokens:
            write_packed(bytes(padding_tokens * 2))
        if token_position != assignments[-1].token_end:
            raise ValueError("packed token count drift")
        shard_artifacts.append(sink.finish())
    except BaseException:
        sink.abort()
        raise
    if len(shard_artifacts) != len(assignments):
        raise ValueError("not every shard assignment was published")
    return {
        "packed_stream_sha256": packed_digest.hexdigest(),
        "shards": sorted(shard_artifacts, key=lambda item: item["path"]),
    }


def rerender_and_pack(
    catalog: InputCatalog,
    metadata: tuple[MetadataRecord, ...],
    schedule: tuple[ScheduleRecord, ...],
    assignments: tuple[ShardAssignment, ...],
    renderer: Renderer,
    root: Path | str,
) -> dict[str, object]:
    """Rerender records and install shards through pinned no-follow fds."""

    parent_fd, root_name = open_parent_directory(root)
    root_fd = -1
    tombstone_fd = None
    try:
        root_fd, _created = open_directory_at(parent_fd, root_name)
        tombstone_fd = open_tombstone_directory(parent_fd)
        shards_fd, _created = open_directory_at(root_fd, "shards", create=True)
        try:
            return _rerender_and_pack_pinned(
                catalog,
                metadata,
                schedule,
                assignments,
                renderer,
                shards_fd,
                owner="direct-pack",
                tombstone_fd=tombstone_fd,
            )
        finally:
            os.close(shards_fd)
    finally:
        if tombstone_fd is not None:
            tombstone_fd.close()
        if root_fd >= 0:
            os.close(root_fd)
        os.close(parent_fd)


def _stage_owner_bytes(
    build_id: str,
    publication_format: str = _FORMAT,
) -> bytes:
    return canonical_json_bytes(
        {
            "build_id": build_id,
            "format": publication_format,
            "kind": "publication-staging",
        }
    )


def _require_regular_entry(directory_fd: int, name: str, label: str) -> None:
    try:
        metadata = entry_lstat(directory_fd, name)
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing: {name}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is unsafe: {name}")


def _validate_stage_namespace(
    stage_fd: int,
    build_id: str,
    *,
    sidecar_names: tuple[str, ...] = (),
) -> None:
    regular_names = {*_FOUNDATION_NAMES, _STAGE_OWNER_NAME, "receipt.json"}
    temporary_targets = {*_FOUNDATION_NAMES, "receipt.json"}
    for name in list_entries(stage_fd):
        metadata = entry_lstat(stage_fd, name)
        if name == "shards" or (name == "sidecars" and sidecar_names):
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"parallel corpus {name} entry is unsafe")
        elif name in regular_names:
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"parallel corpus staging entry is unsafe: {name}")
        elif is_owned_temporary(name, temporary_targets, build_id):
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"owned temporary entry is unsafe: {name}")
        else:
            raise ValueError(f"foreign parallel corpus staging entry: {name}")


def _validate_shard_namespace(
    shards_fd: int,
    final_names: set[str],
    build_id: str,
) -> None:
    for name in list_entries(shards_fd):
        metadata = entry_lstat(shards_fd, name)
        if name in final_names:
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"parallel corpus shard is unsafe: {name}")
        elif is_owned_temporary(name, final_names, build_id):
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"owned shard temporary is unsafe: {name}")
        else:
            raise ValueError(f"foreign parallel corpus shard entry: {name}")


def _prepare_staging(
    stage_fd: int,
    *,
    created: bool,
    build_id: str,
    shard_names: set[str],
    publication_format: str,
    sidecar_names: tuple[str, ...],
    tombstone_fd: RetainedTombstoneStore,
) -> tuple[int, int, dict[str, int]]:
    owner_payload = _stage_owner_bytes(build_id, publication_format)
    if created:
        if list_entries(stage_fd):
            raise ValueError("new parallel corpus staging directory is not empty")
        atomic_write_or_match(
            stage_fd,
            _STAGE_OWNER_NAME,
            owner_payload,
            owner=build_id,

            tombstone_fd=tombstone_fd,
        )
    else:
        try:
            actual_owner = read_regular_file(stage_fd, _STAGE_OWNER_NAME)
        except FileNotFoundError as error:
            raise ValueError("parallel corpus staging ownership marker is missing") from error
        if actual_owner != owner_payload:
            raise ValueError("parallel corpus staging ownership marker drift")
    _validate_stage_namespace(
        stage_fd,
        build_id,
        sidecar_names=sidecar_names,
    )
    clean_owned_temporaries(
        stage_fd,
        final_names={*_FOUNDATION_NAMES, "receipt.json"},
        owner=build_id,

        tombstone_fd=tombstone_fd,
    )
    shards_fd, _created = open_directory_at(stage_fd, "shards", create=True)
    sidecars_fd = -1
    opened_sidecars: dict[str, int] = {}
    try:
        _validate_shard_namespace(shards_fd, shard_names, build_id)
        clean_owned_temporaries(
            shards_fd,
            final_names=shard_names,
            owner=build_id,

            tombstone_fd=tombstone_fd,
        )
        if sidecar_names:
            sidecars_fd, _created = open_directory_at(
                stage_fd,
                "sidecars",
                create=True,
            )
            actual_names = set(list_entries(sidecars_fd))
            extras = actual_names - set(sidecar_names)
            if extras:
                raise ValueError(
                    "foreign parallel corpus sidecar set: "
                    f"{sorted(extras)[0]}"
                )
            for sidecar_name in sidecar_names:
                sidecar_fd, _created = open_directory_at(
                    sidecars_fd,
                    sidecar_name,
                    create=True,
                )
                opened_sidecars[sidecar_name] = sidecar_fd
                _validate_shard_namespace(sidecar_fd, shard_names, build_id)
                clean_owned_temporaries(
                    sidecar_fd,
                    final_names=shard_names,
                    owner=build_id,

                    tombstone_fd=tombstone_fd,
                )
    except BaseException:
        os.close(shards_fd)
        for sidecar_fd in opened_sidecars.values():
            os.close(sidecar_fd)
        if sidecars_fd >= 0:
            os.close(sidecars_fd)
        raise
    return shards_fd, sidecars_fd, opened_sidecars


def _artifact_at(
    directory_fd: int,
    name: str,
    *,
    relative_path: str | None = None,
) -> dict[str, object]:
    byte_count, digest = regular_file_digest(directory_fd, name)
    return {
        "bytes": byte_count,
        "path": relative_path or name,
        "sha256": digest,
    }


def _assert_complete_stage(
    stage_fd: int,
    shards_fd: int,
    shard_names: set[str],
    *,
    sidecars_fd: int,
    sidecar_fds: dict[str, int],
) -> None:
    expected_stage = {*_FOUNDATION_NAMES, _STAGE_OWNER_NAME, "shards"}
    if sidecar_fds:
        expected_stage.add("sidecars")
    if set(list_entries(stage_fd)) != expected_stage:
        raise ValueError("parallel corpus staging namespace is incomplete or foreign")
    for name in _FOUNDATION_NAMES | {_STAGE_OWNER_NAME}:
        _require_regular_entry(stage_fd, name, "parallel corpus staging artifact")
    if not stat.S_ISDIR(entry_lstat(stage_fd, "shards").st_mode):
        raise ValueError("parallel corpus shards entry is unsafe")
    if set(list_entries(shards_fd)) != shard_names:
        raise ValueError("parallel corpus shard namespace is incomplete or foreign")
    for name in shard_names:
        _require_regular_entry(shards_fd, name, "parallel corpus shard")
    if sidecar_fds:
        if sidecars_fd < 0 or set(list_entries(sidecars_fd)) != set(sidecar_fds):
            raise ValueError(
                "parallel corpus sidecar namespace is incomplete or foreign"
            )
        for sidecar_name, sidecar_fd in sidecar_fds.items():
            if set(list_entries(sidecar_fd)) != shard_names:
                raise ValueError(
                    f"parallel corpus sidecar set is incomplete: {sidecar_name}"
                )
            for name in shard_names:
                _require_regular_entry(
                    sidecar_fd,
                    name,
                    f"parallel corpus sidecar {sidecar_name}",
                )


def _publish_staging(
    *,
    parent_fd: int,
    stage_fd: int,
    stage_name: str,
    output_name: str,
    build_id: str,
    tombstone_fd: RetainedTombstoneStore,
    publication_format: str,
) -> VerifiedParallelCorpus:
    stage_metadata = os.fstat(stage_fd)
    named_metadata = entry_lstat(parent_fd, stage_name)
    if (
        not stat.S_ISDIR(named_metadata.st_mode)
        or (named_metadata.st_dev, named_metadata.st_ino)
        != (stage_metadata.st_dev, stage_metadata.st_ino)
    ):
        raise ValueError("parallel corpus staging directory identity changed")
    try:
        atomic_rename_noreplace(
            parent_fd,
            stage_name,
            parent_fd,
            output_name,
        )
    except FileExistsError:
        atomic_write_or_match(
            stage_fd,
            _STAGE_OWNER_NAME,
            _stage_owner_bytes(build_id, publication_format),
            owner=build_id,
            tombstone_fd=tombstone_fd,
        )
        fsync_directory(stage_fd)
        try:
            published_fd, _created = open_directory_at(parent_fd, output_name)
            try:
                return _with_retained_tombstones(
                    _verify_parallel_corpus_fd(
                        published_fd,
                        expected_build_id=build_id,
                    ),
                    parent_fd,
                )
            finally:
                os.close(published_fd)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"conflicting parallel corpus output: {output_name}"
            ) from error
    published_fd, _created = open_directory_at(parent_fd, output_name)
    try:
        published_metadata = os.fstat(published_fd)
        if (published_metadata.st_dev, published_metadata.st_ino) != (
            stage_metadata.st_dev,
            stage_metadata.st_ino,
        ):
            raise ValueError("published corpus directory identity changed")
        fsync_directory(parent_fd)
        return _with_retained_tombstones(
            _verify_parallel_corpus_fd(
                published_fd,
                expected_build_id=build_id,
            ),
            parent_fd,
        )
    finally:
        os.close(published_fd)


def _artifact_contract(
    receipt: dict[str, Any],
    *,
    receipt_name: str = "receipt.json",
) -> tuple[dict[str, dict[str, Any]], set[tuple[str, ...]]]:
    primary_artifacts = receipt["artifacts"]
    if not isinstance(primary_artifacts, list) or not primary_artifacts:
        raise ValueError("publication artifacts must be a non-empty list")
    groups = [(primary_artifacts, True)]
    if receipt["format"] == _FORMAT_V2:
        sidecar_sets = receipt["sidecar_sets"]
        if not isinstance(sidecar_sets, list) or len(sidecar_sets) != len(
            _TARGET_WEIGHT_NAMES
        ):
            raise ValueError("sidecar_sets must contain both target-weight sets")
        names = []
        for sidecar_set in sidecar_sets:
            if (
                not isinstance(sidecar_set, dict)
                or set(sidecar_set) != _SIDECAR_SET_FIELDS
            ):
                raise ValueError("sidecar set fields do not match the contract")
            name = sidecar_set["name"]
            names.append(name)
            if sidecar_set["dtype"] != "uint8":
                raise ValueError(f"sidecar dtype must be uint8: {name}")
            if (
                isinstance(sidecar_set["items"], bool)
                or not isinstance(sidecar_set["items"], int)
                or sidecar_set["items"] <= 0
            ):
                raise ValueError(f"sidecar item count is invalid: {name}")
            stream_sha256 = sidecar_set["stream_sha256"]
            if (
                not isinstance(stream_sha256, str)
                or len(stream_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in stream_sha256
                )
            ):
                raise ValueError(f"sidecar stream digest is invalid: {name}")
            artifacts = sidecar_set["artifacts"]
            if not isinstance(artifacts, list) or not artifacts:
                raise ValueError(f"sidecar artifacts must be non-empty: {name}")
            groups.append((artifacts, False))
        if tuple(names) != _TARGET_WEIGHT_NAMES:
            raise ValueError("sidecar set names must be canonical and ordered")

    artifact_by_path = {}
    file_parts = set()
    directory_parts = {()}
    all_paths = []
    for artifacts, require_sorted in groups:
        ordered_paths = []
        for artifact in artifacts:
            if not isinstance(artifact, dict) or set(artifact) != _ARTIFACT_FIELDS:
                raise ValueError("invalid publication artifact record")
            relative_text = artifact["path"]
            if (
                not isinstance(relative_text, str)
                or not relative_text
                or "\\" in relative_text
            ):
                raise ValueError("artifact path must be a safe relative POSIX path")
            relative = PurePosixPath(relative_text)
            parts = relative.parts
            if (
                relative.is_absolute()
                or relative.as_posix() != relative_text
                or not parts
                or any(part in {"", ".", ".."} for part in parts)
                or relative_text
                in {"receipt.json", receipt_name, _STAGE_OWNER_NAME}
            ):
                raise ValueError("artifact path must be a safe relative POSIX path")
            if (
                isinstance(artifact["bytes"], bool)
                or not isinstance(artifact["bytes"], int)
                or artifact["bytes"] < 0
                or not isinstance(artifact["sha256"], str)
                or len(artifact["sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in artifact["sha256"]
                )
            ):
                raise ValueError("invalid publication artifact digest record")
            ordered_paths.append(relative_text)
            all_paths.append(relative_text)
            artifact_by_path[relative_text] = artifact
            file_parts.add(parts)
            for depth in range(1, len(parts)):
                directory_parts.add(parts[:depth])
        if require_sorted and ordered_paths != sorted(ordered_paths):
            raise ValueError("publication artifact paths must be sorted")
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("publication artifact paths must be unique")
    if file_parts & directory_parts:
        raise ValueError("publication artifact path collides with a directory")
    return artifact_by_path, directory_parts


def _unexpected_namespace_entry(
    directory_fd: int,
    name: str,
    relative: str,
) -> ValueError:
    metadata = entry_lstat(directory_fd, name)
    if stat.S_ISLNK(metadata.st_mode):
        kind = "symlink"
    elif stat.S_ISDIR(metadata.st_mode):
        kind = "unexpected directory"
    elif stat.S_ISREG(metadata.st_mode):
        kind = "unexpected file"
    else:
        kind = "special entry"
    return ValueError(f"publication namespace contains {kind}: {relative}")


def _open_artifact_namespace(
    root_fd: int,
    artifact_by_path: dict[str, dict[str, Any]],
    directory_parts: set[tuple[str, ...]],
    receipt_metadata: os.stat_result,
    *,
    receipt_name: str,
    allow_stage_owner: bool,
) -> tuple[dict[str, tuple[int, os.stat_result, int, str]], bytes | None]:
    file_parts = {
        tuple(PurePosixPath(path).parts): path for path in artifact_by_path
    }
    reserved_files = {(receipt_name,): _RECEIPT_NAMESPACE_KIND}
    if allow_stage_owner:
        reserved_files[(_STAGE_OWNER_NAME,)] = _STAGE_OWNER_NAME
    expected_children: dict[tuple[str, ...], dict[str, object]] = {
        directory: {} for directory in directory_parts
    }
    for directory in directory_parts:
        if directory:
            expected_children[directory[:-1]][directory[-1]] = "directory"
    for parts, path in {**file_parts, **reserved_files}.items():
        if parts[:-1] not in expected_children:
            raise ValueError("publication artifact directory namespace is invalid")
        expected_children[parts[:-1]][parts[-1]] = path

    directory_fds = {(): root_fd}
    opened_files: dict[str, tuple[int, os.stat_result, int, str]] = {}
    owner_bytes = None
    try:
        for directory in sorted(directory_parts, key=lambda parts: (len(parts), parts)):
            directory_fd = directory_fds[directory]
            expected = expected_children[directory]
            actual_names = set(list_entries(directory_fd))
            expected_names = set(expected)
            extras = sorted(actual_names - expected_names)
            if extras:
                name = extras[0]
                relative = "/".join((*directory, name))
                raise _unexpected_namespace_entry(directory_fd, name, relative)
            missing = sorted(expected_names - actual_names)
            if missing:
                relative = "/".join((*directory, missing[0]))
                raise ValueError(f"publication namespace is missing: {relative}")
            for name, kind in sorted(expected.items()):
                child_parts = (*directory, name)
                if kind == "directory":
                    child_fd, _created = open_directory_at(directory_fd, name)
                    directory_fds[child_parts] = child_fd
                    continue
                if kind is _RECEIPT_NAMESPACE_KIND:
                    current = entry_lstat(directory_fd, name)
                    if (
                        not stat.S_ISREG(current.st_mode)
                        or (current.st_dev, current.st_ino)
                        != (receipt_metadata.st_dev, receipt_metadata.st_ino)
                    ):
                        raise ValueError("parallel corpus receipt identity changed")
                    continue
                descriptor, metadata = open_regular_file_at(directory_fd, name)
                if kind == _STAGE_OWNER_NAME:
                    try:
                        owner_bytes = read_file_descriptor(descriptor)
                    finally:
                        os.close(descriptor)
                else:
                    try:
                        binding_fd = os.dup(directory_fd)
                    except BaseException:
                        os.close(descriptor)
                        raise
                    opened_files[kind] = (
                        descriptor,
                        metadata,
                        binding_fd,
                        name,
                    )
        return opened_files, owner_bytes
    except BaseException:
        for descriptor, _metadata, binding_fd, _name in opened_files.values():
            os.close(descriptor)
            os.close(binding_fd)
        raise
    finally:
        for directory, descriptor in directory_fds.items():
            if directory:
                os.close(descriptor)


def _read_opened_artifact(
    opened_files: dict[str, tuple[int, os.stat_result, int, str]],
    artifact_by_path: dict[str, dict[str, Any]],
    relative: str,
) -> bytes:
    descriptor, metadata, binding_fd, name = opened_files.pop(relative)
    artifact = artifact_by_path[relative]
    try:
        payload = read_file_descriptor(descriptor)
        final_metadata = os.fstat(descriptor)
        named_metadata = entry_lstat(binding_fd, name)
    finally:
        os.close(descriptor)
        os.close(binding_fd)
    if (
        metadata.st_size != artifact["bytes"]
        or len(payload) != artifact["bytes"]
        or sha256_hex(payload) != artifact["sha256"]
        or not stat.S_ISREG(named_metadata.st_mode)
        or (
            named_metadata.st_dev,
            named_metadata.st_ino,
            named_metadata.st_size,
        )
        != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
        )
        or (
            final_metadata.st_dev,
            final_metadata.st_ino,
            final_metadata.st_size,
            final_metadata.st_nlink,
        )
        != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_nlink,
        )
    ):
        raise ValueError(f"publication artifact digest drift: {relative}")
    return payload


class _PinnedPackedReader:
    def __init__(
        self,
        entries: list[
            tuple[str, int, os.stat_result, int, str, dict[str, Any]]
        ],
    ) -> None:
        self.entries = entries
        self.index = 0
        self.remaining = 0
        self.local_digest = hashlib.sha256()
        self.global_digest = hashlib.sha256()
        try:
            self._begin_entry()
        except BaseException:
            self.close()
            raise

    def _begin_entry(self) -> None:
        if self.index >= len(self.entries):
            self.remaining = 0
            return
        (
            relative,
            _descriptor,
            metadata,
            _binding_fd,
            _name,
            artifact,
        ) = self.entries[self.index]
        if metadata.st_size != artifact["bytes"]:
            raise ValueError(f"publication artifact digest drift: {relative}")
        self.remaining = metadata.st_size
        self.local_digest = hashlib.sha256()

    def _finish_empty_entries(self) -> None:
        while self.index < len(self.entries) and self.remaining == 0:
            (
                relative,
                descriptor,
                metadata,
                binding_fd,
                name,
                artifact,
            ) = self.entries[self.index]
            actual_digest = self.local_digest.hexdigest()
            trailing = os.read(descriptor, 1)
            final_metadata = os.fstat(descriptor)
            named_metadata = entry_lstat(binding_fd, name)
            if (
                trailing
                or not stat.S_ISREG(named_metadata.st_mode)
                or (
                    named_metadata.st_dev,
                    named_metadata.st_ino,
                    named_metadata.st_size,
                )
                != (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_size,
                )
                or (
                    final_metadata.st_dev,
                    final_metadata.st_ino,
                    final_metadata.st_size,
                    final_metadata.st_nlink,
                )
                != (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_size,
                    metadata.st_nlink,
                )
            ):
                raise ValueError(
                    f"publication artifact EOF or identity drift: {relative}"
                )
            os.close(descriptor)
            os.close(binding_fd)
            self.index += 1
            self._begin_entry()
            if actual_digest != artifact["sha256"]:
                raise ValueError(f"publication artifact digest drift: {relative}")

    def consume(
        self,
        byte_count: int,
        *,
        record_digest: Any | None = None,
        require_binary: bool = False,
        require_one: bool = False,
        require_zero: bool = False,
    ) -> None:
        remaining = byte_count
        while remaining:
            self._finish_empty_entries()
            if self.index >= len(self.entries):
                raise ValueError(
                    "packed shards end before their declared token stream"
                )
            (
                _relative,
                descriptor,
                _metadata,
                _binding_fd,
                _name,
                _artifact,
            ) = self.entries[self.index]
            chunk = os.read(descriptor, min(1 << 20, remaining, self.remaining))
            if not chunk:
                raise ValueError("packed shard ends before its declared size")
            self.local_digest.update(chunk)
            self.global_digest.update(chunk)
            if record_digest is not None:
                record_digest.update(chunk)
            if require_binary and any(value not in (0, 1) for value in chunk):
                raise ValueError("target-weight sidecar contains non-binary values")
            if require_one and any(value != 1 for value in chunk):
                raise ValueError("dense target-weight sidecar contains zero weights")
            if require_zero and any(chunk):
                raise ValueError("packed update padding is not zero")
            self.remaining -= len(chunk)
            remaining -= len(chunk)

    def finish(self) -> str:
        self._finish_empty_entries()
        if self.index != len(self.entries):
            raise ValueError("packed shards contain trailing bytes")
        return self.global_digest.hexdigest()

    def close(self) -> None:
        for index in range(self.index, len(self.entries)):
            descriptor = self.entries[index][1]
            binding_fd = self.entries[index][3]
            try:
                os.close(descriptor)
            except OSError:
                pass
            try:
                os.close(binding_fd)
            except OSError:
                pass
        self.index = len(self.entries)


def _verify_sidecar_sets(
    receipt: dict[str, Any],
    assignments: tuple[ShardAssignment, ...],
    opened_files: dict[str, tuple[int, os.stat_result, int, str]],
    artifact_by_path: dict[str, dict[str, Any]],
    *,
    logical_tokens: int,
    packed_tokens: int,
) -> tuple[tuple[dict[str, object], ...], set[str]]:
    if receipt["format"] == _FORMAT:
        return (), set()

    commitments = []
    all_paths = set()
    for sidecar_set in receipt["sidecar_sets"]:
        name = sidecar_set["name"]
        expected_paths = [
            f"sidecars/{name}/{assignment.shard_id}.bin"
            for assignment in assignments
        ]
        actual_paths = [
            artifact["path"] for artifact in sidecar_set["artifacts"]
        ]
        if actual_paths != expected_paths:
            raise ValueError(
                f"sidecar artifacts are partial, reordered, or misnamed: {name}"
            )
        if sidecar_set["items"] != packed_tokens:
            raise ValueError(f"sidecar item count drift: {name}")

        entries = []
        for assignment, relative in zip(
            assignments,
            expected_paths,
            strict=True,
        ):
            expected_bytes = assignment.token_end - assignment.token_start
            if artifact_by_path[relative]["bytes"] != expected_bytes:
                raise ValueError(f"sidecar shard size drift: {relative}")
            descriptor, metadata_stat, binding_fd, entry_name = opened_files.pop(relative)
            entries.append(
                (
                    relative,
                    descriptor,
                    metadata_stat,
                    binding_fd,
                    entry_name,
                    artifact_by_path[relative],
                )
            )
            all_paths.add(relative)

        logical_digest = hashlib.sha256()
        reader = _PinnedPackedReader(entries)
        try:
            reader.consume(
                logical_tokens,
                record_digest=logical_digest,
                require_binary=True,
                require_one=name == "dense_target_weights",
            )
            reader.consume(
                packed_tokens - logical_tokens,
                require_zero=True,
            )
            packed_digest = reader.finish()
        finally:
            reader.close()
        if sidecar_set["stream_sha256"] != packed_digest:
            raise ValueError(f"sidecar stream digest drift: {name}")
        commitments.append(
            {
                "dtype": "uint8",
                "items": logical_tokens,
                "name": name,
                "sha256": logical_digest.hexdigest(),
            }
        )
    return tuple(commitments), all_paths

def _receipt_modification_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int | None, int | None]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        getattr(metadata, "st_mtime_ns", None),
        getattr(metadata, "st_ctime_ns", None),
    )


def _require_receipt_identity(
    expected: tuple[int, int, int, int | None, int | None],
    *observed: os.stat_result,
) -> None:
    if any(
        not stat.S_ISREG(metadata.st_mode)
        or _receipt_modification_identity(metadata) != expected
        for metadata in observed
    ):
        raise ValueError("parallel corpus receipt modification identity drift")


def _verify_receipt_after_verification(
    publication_fd: int,
    receipt_fd: int,
    initial_metadata: os.stat_result,
    initial_bytes: bytes,
    initial_sha256: str,
    *,
    receipt_name: str,
) -> None:
    expected_identity = _receipt_modification_identity(initial_metadata)
    try:
        before_eof = os.fstat(receipt_fd)
        named_before_eof = entry_lstat(publication_fd, receipt_name)
        _require_receipt_identity(
            expected_identity,
            before_eof,
            named_before_eof,
        )
        trailing = os.read(receipt_fd, 1)
        after_eof = os.fstat(receipt_fd)
        named_after_eof = entry_lstat(publication_fd, receipt_name)
        _require_receipt_identity(
            expected_identity,
            after_eof,
            named_after_eof,
        )
        named_before = entry_lstat(publication_fd, receipt_name)
        os.lseek(receipt_fd, 0, os.SEEK_SET)
        before_reread = os.fstat(receipt_fd)
        _require_receipt_identity(
            expected_identity,
            before_reread,
            named_before,
        )
        final_bytes = read_file_descriptor(receipt_fd)
        final_metadata = os.fstat(receipt_fd)
        named_after = entry_lstat(publication_fd, receipt_name)
        _require_receipt_identity(
            expected_identity,
            final_metadata,
            named_after,
        )
        final_eof = os.read(receipt_fd, 1)
        after_final_eof = os.fstat(receipt_fd)
        named_after_final_eof = entry_lstat(publication_fd, receipt_name)
        _require_receipt_identity(
            expected_identity,
            after_final_eof,
            named_after_final_eof,
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            "parallel corpus receipt EOF, identity, or content drift"
        ) from error
    if (
        trailing
        or final_eof
        or initial_metadata.st_size != len(initial_bytes)
        or sha256_hex(initial_bytes) != initial_sha256
        or len(final_bytes) != len(initial_bytes)
        or sha256_hex(final_bytes) != initial_sha256
        or final_bytes != initial_bytes
    ):
        raise ValueError("parallel corpus receipt EOF, identity, or content drift")


def _verify_parallel_corpus_fd(
    publication_fd: int,
    *,
    receipt_name: str = "receipt.json",
    expected_build_id: str | None = None,
    allow_stage_owner: bool = False,
) -> dict[str, Any]:
    if (
        not isinstance(receipt_name, str)
        or not receipt_name
        or receipt_name in {".", ".."}
        or "/" in receipt_name
        or "\x00" in receipt_name
    ):
        raise ValueError("parallel corpus receipt name is unsafe")
    try:
        receipt_fd, receipt_metadata = open_regular_file_at(
            publication_fd,
            receipt_name,
        )
    except (OSError, ValueError) as error:
        raise ValueError("parallel corpus receipt is missing or unsafe") from error
    try:
        return _verify_parallel_corpus_with_receipt_fd(
            publication_fd,
            receipt_fd,
            receipt_metadata,
            receipt_name=receipt_name,
            expected_build_id=expected_build_id,
            allow_stage_owner=allow_stage_owner,
        )
    finally:
        os.close(receipt_fd)


def _verify_parallel_corpus_with_receipt_fd(
    publication_fd: int,
    receipt_fd: int,
    receipt_metadata: os.stat_result,
    *,
    receipt_name: str,
    expected_build_id: str | None = None,
    allow_stage_owner: bool = False,
) -> dict[str, Any]:
    expected_receipt_identity = _receipt_modification_identity(receipt_metadata)
    try:
        named_before_read = entry_lstat(publication_fd, receipt_name)
        before_read = os.fstat(receipt_fd)
        _require_receipt_identity(
            expected_receipt_identity,
            before_read,
            named_before_read,
        )
        receipt_bytes = read_file_descriptor(receipt_fd)
        after_read = os.fstat(receipt_fd)
        named_after_read = entry_lstat(publication_fd, receipt_name)
        _require_receipt_identity(
            expected_receipt_identity,
            after_read,
            named_after_read,
        )
        if os.read(receipt_fd, 1):
            raise ValueError("parallel corpus receipt does not end at EOF")
        after_eof = os.fstat(receipt_fd)
        named_after_eof = entry_lstat(publication_fd, receipt_name)
        _require_receipt_identity(
            expected_receipt_identity,
            after_eof,
            named_after_eof,
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            "parallel corpus receipt EOF, identity, or content drift"
        ) from error
    receipt_sha256 = sha256_hex(receipt_bytes)
    try:
        receipt = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("parallel corpus receipt is invalid JSON") from error
    if not isinstance(receipt, dict) or canonical_json_bytes(receipt) != receipt_bytes:
        raise ValueError("parallel corpus receipt does not match the canonical contract")
    publication_format = receipt.get("format")
    expected_fields = (
        _RECEIPT_FIELDS | {"sidecar_sets"}
        if publication_format == _FORMAT_V2
        else _RECEIPT_FIELDS
    )
    if set(receipt) != expected_fields:
        raise ValueError("parallel corpus receipt does not match the canonical contract")
    if (
        publication_format not in {_FORMAT, _FORMAT_V2}
        or receipt["compiler_version"] != _COMPILER_VERSION
    ):
        raise ValueError("parallel corpus format identity mismatch")
    for field_name in _RECEIPT_COUNT_FIELDS:
        value = receipt[field_name]
        minimum = 0 if field_name == "padding_tokens" else 1
        if type(value) is not int or value < minimum:
            qualifier = "non-negative" if minimum == 0 else "positive"
            raise ValueError(
                f"parallel corpus receipt {field_name} must be a "
                f"{qualifier} integer"
            )
    artifact_by_path, directory_parts = _artifact_contract(
        receipt,
        receipt_name=receipt_name,
    )
    opened_files, owner_bytes = _open_artifact_namespace(
        publication_fd,
        artifact_by_path,
        directory_parts,
        receipt_metadata,
        receipt_name=receipt_name,
        allow_stage_owner=allow_stage_owner,
    )
    try:
        required_paths = {
            "assignments.jsonl",
            "catalog.jsonl",
            "metadata.jsonl",
            "schedule.jsonl",
        }
        if not required_paths <= set(artifact_by_path):
            raise ValueError(
                "parallel corpus is missing required foundation artifacts"
            )
        config = ParallelBuildConfig.from_dict(receipt["config"])
        renderer_id = receipt["renderer_id"]
        if not isinstance(renderer_id, str) or not renderer_id:
            raise ValueError("receipt renderer_id must be non-empty")
        catalog_bytes = _read_opened_artifact(
            opened_files,
            artifact_by_path,
            "catalog.jsonl",
        )
        metadata_bytes = _read_opened_artifact(
            opened_files,
            artifact_by_path,
            "metadata.jsonl",
        )
        schedule_bytes = _read_opened_artifact(
            opened_files,
            artifact_by_path,
            "schedule.jsonl",
        )
        assignment_bytes = _read_opened_artifact(
            opened_files,
            artifact_by_path,
            "assignments.jsonl",
        )
        catalog = InputCatalog.from_bytes(catalog_bytes)
        metadata = metadata_from_bytes(metadata_bytes)
        schedule = schedule_from_bytes(schedule_bytes)
        assignments = assignments_from_bytes(assignment_bytes)
        reduced = reduce_metadata(
            catalog,
            metadata,
            expected_renderer_id=renderer_id,
        )
        expected_schedule = largest_deficit_schedule(
            reduced,
            config.lane_weights,
        )
        if schedule != expected_schedule:
            raise ValueError(
                "published schedule is not the canonical deficit reduction"
            )
        expected_assignments = assign_update_aligned_shards(
            total_tokens=schedule[-1].token_end,
            update_tokens=config.update_tokens,
            shard_count=config.shard_count,
            allow_fewer=config.allow_fewer_shards,
        )
        if assignments != expected_assignments:
            raise ValueError("published shard assignments are not canonical")
        ordered_hash, merkle_root = ordered_stream_commitments(schedule, reduced)
        logical_tokens = schedule[-1].token_end
        packed_tokens = assignments[-1].token_end
        expected_scalars = {
            "assignments_sha256": sha256_hex(assignment_bytes),
            "catalog_sha256": catalog.sha256,
            "logical_tokens": logical_tokens,
            "merkle_root_sha256": merkle_root,
            "metadata_sha256": sha256_hex(metadata_bytes),
            "ordered_stream_sha256": ordered_hash,
            "packed_tokens": packed_tokens,
            "padding_tokens": packed_tokens - logical_tokens,
            "record_count": len(metadata),
            "schedule_sha256": sha256_hex(schedule_bytes),
            "shard_count": len(assignments),
        }
        for field_name, expected in expected_scalars.items():
            if receipt[field_name] != expected:
                raise ValueError(f"parallel corpus receipt drift: {field_name}")

        shard_paths = [
            f"shards/{assignment.shard_id}.bin" for assignment in assignments
        ]
        declared_sidecar_paths = {
            artifact["path"]
            for sidecar_set in receipt.get("sidecar_sets", [])
            for artifact in sidecar_set["artifacts"]
        }
        if (
            set(artifact_by_path) - required_paths - declared_sidecar_paths
            != set(shard_paths)
        ):
            raise ValueError(
                "parallel corpus shard namespace does not match assignments"
            )
        for assignment, relative in zip(
            assignments,
            shard_paths,
            strict=True,
        ):
            expected_bytes = (
                assignment.token_end - assignment.token_start
            ) * 2
            if artifact_by_path[relative]["bytes"] != expected_bytes:
                raise ValueError(f"parallel corpus shard size drift: {relative}")
        entries = [
            (
                relative,
                opened_files[relative][0],
                opened_files[relative][1],
                opened_files[relative][2],
                opened_files[relative][3],
                artifact_by_path[relative],
            )
            for relative in shard_paths
        ]
        for relative in shard_paths:
            del opened_files[relative]
        reader = _PinnedPackedReader(entries)
        try:
            metadata_by_id = {record.record_id: record for record in metadata}
            for entry in schedule:
                digest = hashlib.sha256()
                reader.consume(
                    entry.token_length * 2,
                    record_digest=digest,
                )
                if (
                    digest.hexdigest()
                    != metadata_by_id[entry.record_id].render_sha256
                ):
                    raise ValueError(
                        f"packed record digest drift: {entry.record_id}"
                    )
            reader.consume(
                (packed_tokens - logical_tokens) * 2,
                require_zero=True,
            )
            packed_digest = reader.finish()
        finally:
            reader.close()
        if receipt["packed_stream_sha256"] != packed_digest:
            raise ValueError("packed stream digest drift")
        sidecar_commitments, verified_sidecar_paths = _verify_sidecar_sets(
            receipt,
            assignments,
            opened_files,
            artifact_by_path,
            logical_tokens=logical_tokens,
            packed_tokens=packed_tokens,
        )
        if verified_sidecar_paths != declared_sidecar_paths:
            raise ValueError("parallel corpus sidecar namespace drift")
        build_id = _parallel_build_id(
            catalog,
            renderer_id,
            config,
            publication_format=publication_format,
            sidecar_commitments=sidecar_commitments,
        )
        if receipt["build_id"] != build_id:
            raise ValueError("parallel corpus receipt drift: build_id")
        if expected_build_id is not None and build_id != expected_build_id:
            raise ValueError("parallel corpus build id does not match expectation")
        if allow_stage_owner:
            if owner_bytes != _stage_owner_bytes(build_id, publication_format):
                raise ValueError("parallel corpus staging ownership marker drift")
        elif owner_bytes is not None:
            raise ValueError("published corpus contains a staging owner marker")
        if opened_files:
            raise ValueError("parallel corpus contains unverified artifacts")
        _verify_receipt_after_verification(
            publication_fd,
            receipt_fd,
            receipt_metadata,
            receipt_bytes,
            receipt_sha256,
            receipt_name=receipt_name,
        )
        return receipt
    finally:
        for descriptor, _metadata, binding_fd, _name in opened_files.values():
            os.close(descriptor)
            os.close(binding_fd)


def verify_parallel_corpus(
    root: Path | str,
    *,
    expected_build_id: str | None = None,
    _allow_stage_owner: bool = False,
) -> VerifiedParallelCorpus:
    parent_fd = -1
    publication_fd = -1
    receipt_name = "receipt.json"
    try:
        parent_fd, name = open_parent_directory(root)
        try:
            publication_fd, _created = open_directory_at(parent_fd, name)
        except (OSError, ValueError):
            metadata = entry_lstat(parent_fd, name)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("parallel corpus receipt is not a regular file")
            publication_fd = os.dup(parent_fd)
            receipt_name = name
    except (OSError, ValueError) as error:
        if parent_fd >= 0:
            os.close(parent_fd)
        raise ValueError(
            "parallel corpus publication or receipt is missing or unsafe"
        ) from error
    try:
        return _with_retained_tombstones(
            _verify_parallel_corpus_fd(
                publication_fd,
                receipt_name=receipt_name,
                expected_build_id=expected_build_id,
                allow_stage_owner=_allow_stage_owner,
            ),
            parent_fd,
        )
    finally:
        os.close(publication_fd)
        os.close(parent_fd)


def _verify_parallel_corpus_at(
    parent_fd: int,
    name: str,
    *,
    expected_build_id: str,
) -> VerifiedParallelCorpus:
    publication_fd, _created = open_directory_at(parent_fd, name)
    try:
        return _with_retained_tombstones(
            _verify_parallel_corpus_fd(
                publication_fd,
                expected_build_id=expected_build_id,
            ),
            parent_fd,
        )
    finally:
        os.close(publication_fd)


def build_parallel_corpus(
    catalog: InputCatalog,
    renderer: Renderer,
    config: ParallelBuildConfig,
    destination: Path | str,
    *,
    workers: int = 1,
    sidecar_paths: dict[str, Path | str] | None = None,
    _materialized_metadata: tuple[MetadataRecord, ...] | None = None,
    _cached_payloads: dict[str, bytes] | None = None,
) -> VerifiedParallelCorpus:
    """Build or resume a corpus using pinned, no-replace publication."""

    output = Path(destination)
    sidecar_sources = _open_sidecar_sources(sidecar_paths)
    publication_format = _FORMAT_V2 if sidecar_sources else _FORMAT
    build_id = _parallel_build_id(
        catalog,
        renderer.renderer_id,
        config,
        publication_format=publication_format,
        sidecar_commitments=tuple(
            source.commitment for source in sidecar_sources
        ),
    )
    try:
        parent_fd, output_name = open_parent_directory(output, create=True)
    except BaseException:
        for source in sidecar_sources:
            source.close()
        raise
    stage_path = publication_staging_path(output, build_id)
    stage_name = stage_path.name
    sidecar_names = tuple(source.name for source in sidecar_sources)
    owner_payload = _stage_owner_bytes(build_id, publication_format)
    tombstone_fd = None
    try:
        if entry_exists(parent_fd, output_name):
            try:
                return _verify_parallel_corpus_at(
                    parent_fd,
                    output_name,
                    expected_build_id=build_id,
                )
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"conflicting parallel corpus output: {output}"
                ) from error

        if (_materialized_metadata is None) != (_cached_payloads is None):
            raise ValueError(
                "materialized metadata and cached payloads must be supplied together"
            )
        metadata = (
            render_metadata(catalog, renderer, workers=workers)
            if _materialized_metadata is None
            else reduce_metadata(
                catalog,
                _materialized_metadata,
                expected_renderer_id=renderer.renderer_id,
            )
        )
        schedule = largest_deficit_schedule(metadata, config.lane_weights)
        assignments = assign_update_aligned_shards(
            total_tokens=schedule[-1].token_end,
            update_tokens=config.update_tokens,
            shard_count=config.shard_count,
            allow_fewer=config.allow_fewer_shards,
        )
        shard_names = {
            f"{assignment.shard_id}.bin" for assignment in assignments
        }
        catalog_bytes = catalog.to_bytes()
        metadata_bytes = metadata_to_bytes(metadata)
        schedule_bytes = schedule_to_bytes(schedule)
        assignment_bytes = assignments_to_bytes(assignments)

        fcntl.flock(parent_fd, fcntl.LOCK_EX)
        if entry_exists(parent_fd, output_name):
            try:
                return _verify_parallel_corpus_at(
                    parent_fd,
                    output_name,
                    expected_build_id=build_id,
                )
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"conflicting parallel corpus output: {output}"
                ) from error
        stage_exists = entry_exists(parent_fd, stage_name)
        if stage_exists:
            stage_metadata = entry_lstat(parent_fd, stage_name)
            if not stat.S_ISDIR(stage_metadata.st_mode):
                raise ValueError("parallel corpus staging path is unsafe")
        tombstone_fd = open_tombstone_directory(parent_fd)
        if stage_exists:
            existing_stage_fd, _created = open_directory_at(parent_fd, stage_name)
            try:
                _validate_stage_namespace(
                    existing_stage_fd,
                    build_id,
                    sidecar_names=sidecar_names,
                )
                if entry_exists(existing_stage_fd, "receipt.json"):
                    has_owner = entry_exists(
                        existing_stage_fd,
                        _STAGE_OWNER_NAME,
                    )
                    if has_owner:
                        actual_owner = read_regular_file(
                            existing_stage_fd,
                            _STAGE_OWNER_NAME,
                        )
                        if actual_owner != owner_payload:
                            raise ValueError(
                                "parallel corpus staging ownership marker drift"
                            )
                    _verify_parallel_corpus_fd(
                        existing_stage_fd,
                        expected_build_id=build_id,
                        allow_stage_owner=has_owner,
                    )
                    if has_owner:
                        unlink_regular_if_matches(
                            existing_stage_fd,
                            _STAGE_OWNER_NAME,
                            owner_payload,
                            tombstone_fd=tombstone_fd,
                        )
                    return _publish_staging(
                        parent_fd=parent_fd,
                        stage_fd=existing_stage_fd,
                        stage_name=stage_name,
                        output_name=output_name,
                        build_id=build_id,
                        publication_format=publication_format,
                        tombstone_fd=tombstone_fd,
                    )
                if not entry_exists(existing_stage_fd, _STAGE_OWNER_NAME):
                    raise ValueError(
                        "parallel corpus staging ownership marker is missing"
                    )
                if (
                    read_regular_file(existing_stage_fd, _STAGE_OWNER_NAME)
                    != owner_payload
                ):
                    raise ValueError(
                        "parallel corpus staging ownership marker drift"
                    )
            finally:
                os.close(existing_stage_fd)

        stage_fd, created = open_directory_at(
            parent_fd,
            stage_name,
            create=True,
        )
        shards_fd = -1
        sidecars_fd = -1
        sidecar_fds: dict[str, int] = {}
        try:
            shards_fd, sidecars_fd, sidecar_fds = _prepare_staging(
                stage_fd,
                created=created,
                build_id=build_id,
                shard_names=shard_names,
                publication_format=publication_format,
                sidecar_names=sidecar_names,
                tombstone_fd=tombstone_fd,
            )
            atomic_write_or_match(
                stage_fd,
                "catalog.jsonl",
                catalog_bytes,
                owner=build_id,

                tombstone_fd=tombstone_fd,
            )
            atomic_write_or_match(
                stage_fd,
                "metadata.jsonl",
                metadata_bytes,
                owner=build_id,

                tombstone_fd=tombstone_fd,
            )
            atomic_write_or_match(
                stage_fd,
                "schedule.jsonl",
                schedule_bytes,
                owner=build_id,

                tombstone_fd=tombstone_fd,
            )
            atomic_write_or_match(
                stage_fd,
                "assignments.jsonl",
                assignment_bytes,
                owner=build_id,

                tombstone_fd=tombstone_fd,
            )
            packed = _rerender_and_pack_pinned(
                catalog,
                metadata,
                schedule,
                assignments,
                renderer,
                shards_fd,
                owner=build_id,
                tombstone_fd=tombstone_fd,
                cached_payloads=_cached_payloads,
            )
            logical_tokens = schedule[-1].token_end
            packed_tokens = assignments[-1].token_end
            sidecar_sets = _pack_sidecar_sets(
                sidecar_sources,
                assignments,
                sidecar_fds,
                logical_tokens=logical_tokens,
                packed_tokens=packed_tokens,
                owner=build_id,

                tombstone_fd=tombstone_fd,
            )
            _assert_complete_stage(
                stage_fd,
                shards_fd,
                shard_names,
                sidecars_fd=sidecars_fd,
                sidecar_fds=sidecar_fds,
            )
            artifacts = [
                *(
                    _artifact_at(stage_fd, name)
                    for name in sorted(_FOUNDATION_NAMES)
                ),
                *(
                    _artifact_at(
                        shards_fd,
                        name,
                        relative_path=f"shards/{name}",
                    )
                    for name in sorted(shard_names)
                ),
            ]
            artifacts.sort(key=lambda item: item["path"])
            ordered_hash, merkle_root = ordered_stream_commitments(
                schedule,
                metadata,
            )
            receipt = {
                "artifacts": artifacts,
                "assignments_sha256": sha256_hex(assignment_bytes),
                "build_id": build_id,
                "catalog_sha256": catalog.sha256,
                "compiler_version": _COMPILER_VERSION,
                "config": config.as_dict(),
                "format": publication_format,
                "logical_tokens": logical_tokens,
                "merkle_root_sha256": merkle_root,
                "metadata_sha256": sha256_hex(metadata_bytes),
                "ordered_stream_sha256": ordered_hash,
                "packed_stream_sha256": packed["packed_stream_sha256"],
                "packed_tokens": packed_tokens,
                "padding_tokens": packed_tokens - logical_tokens,
                "record_count": len(metadata),
                "renderer_id": renderer.renderer_id,
                "schedule_sha256": sha256_hex(schedule_bytes),
                "shard_count": len(assignments),
            }
            if sidecar_sets:
                receipt["sidecar_sets"] = sidecar_sets
            atomic_write_or_match(
                stage_fd,
                "receipt.json",
                canonical_json_bytes(receipt),
                owner=build_id,

                tombstone_fd=tombstone_fd,
            )
            fsync_directory(shards_fd)
            for sidecar_fd in sidecar_fds.values():
                fsync_directory(sidecar_fd)
            if sidecars_fd >= 0:
                fsync_directory(sidecars_fd)
            fsync_directory(stage_fd)
            _verify_parallel_corpus_fd(
                stage_fd,
                expected_build_id=build_id,
                allow_stage_owner=True,
            )
            unlink_regular_if_matches(
                stage_fd,
                _STAGE_OWNER_NAME,
                owner_payload,
                tombstone_fd=tombstone_fd,
            )
            _verify_parallel_corpus_fd(
                stage_fd,
                expected_build_id=build_id,
            )
            return _publish_staging(
                parent_fd=parent_fd,
                stage_fd=stage_fd,
                stage_name=stage_name,
                output_name=output_name,
                build_id=build_id,
                publication_format=publication_format,
                tombstone_fd=tombstone_fd,
            )
        finally:
            for sidecar_fd in sidecar_fds.values():
                os.close(sidecar_fd)
            if sidecars_fd >= 0:
                os.close(sidecars_fd)
            if shards_fd >= 0:
                os.close(shards_fd)
            os.close(stage_fd)
    finally:
        if tombstone_fd is not None:
            tombstone_fd.close()
        os.close(parent_fd)
        for source in sidecar_sources:
            source.close()


def build_parallel_corpus_from_tasks(
    catalog: InputCatalog,
    renderer_id: str,
    config: ParallelBuildConfig,
    destination: Path | str,
    task_results: tuple[TaskResult, ...],
    *,
    expected_task_count: int,
) -> VerifiedParallelCorpus:
    """Validate task results and publish directly from their cached token bytes."""

    from .tasks import reduce_task_results

    metadata, payloads = reduce_task_results(
        catalog,
        renderer_id,
        config,
        task_results,
        expected_task_count=expected_task_count,
    )

    class CachedPayloadIdentity:
        def __init__(self) -> None:
            self.renderer_id = renderer_id

        def render(self, record: Any) -> Any:
            raise AssertionError(
                f"cached task finalization must not rerender {record.record_id}"
            )

    return build_parallel_corpus(
        catalog,
        CachedPayloadIdentity(),
        config,
        destination,
        workers=1,
        _materialized_metadata=metadata,
        _cached_payloads=payloads,
    )


def publish_verification_receipt(
    corpus: Path | str,
    destination: Path | str,
    *,
    expected_build_id: str,
) -> VerifiedParallelCorpus:
    """Verify a corpus and install its canonical receipt under a pinned fd."""

    receipt = verify_parallel_corpus(
        corpus,
        expected_build_id=expected_build_id,
    )
    receipt_bytes = canonical_json_bytes(receipt)
    parent_fd, name = open_parent_directory(destination, create=True)
    tombstone_fd = None
    try:
        tombstone_fd = open_tombstone_directory(parent_fd)
        atomic_write_or_match(
            parent_fd,
            name,
            receipt_bytes,
            owner=expected_build_id,
            tombstone_fd=tombstone_fd,
        )
    finally:
        if tombstone_fd is not None:
            tombstone_fd.close()
        os.close(parent_fd)
    return receipt
