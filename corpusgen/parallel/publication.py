"""Rerendering, packing, receipts, resumability, and atomic publication."""

from __future__ import annotations

import hashlib
import json
import os
import fcntl
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

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
    atomic_rename_noreplace,
    atomic_write_or_match,
    clean_owned_temporaries,
    entry_exists,
    entry_lstat,
    fsync_directory,
    is_owned_temporary,
    list_entries,
    open_directory_at,
    open_directory_path,
    open_parent_directory,
    read_regular_file,
    regular_file_digest,
    unlink_regular_if_matches,
)

if TYPE_CHECKING:
    from .tasks import TaskResult

_FORMAT = "memorysplit-parallel-corpus-v1"
_COMPILER_VERSION = "metadata-first-foundation-v1"
_STAGE_OWNER_NAME = ".parallel-owner.json"
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
    if not isinstance(renderer_id, str) or not renderer_id:
        raise ValueError("renderer_id must be non-empty")
    return sha256_hex(
        canonical_json_bytes(
            {
                "catalog_sha256": catalog.sha256,
                "compiler_version": _COMPILER_VERSION,
                "config": config.as_dict(),
                "format": _FORMAT,
                "renderer_id": renderer_id,
            }
        )
    )


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(root: Path, path: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
    }


class _ShardSink:
    def __init__(
        self,
        shards_fd: int,
        assignment: ShardAssignment,
        *,
        owner: str,
    ) -> None:
        self.assignment = assignment
        self.final_name = f"{assignment.shard_id}.bin"
        self.writer = AtomicFileWriter(
            shards_fd,
            self.final_name,
            owner=owner,
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


def _rerender_and_pack_pinned(
    catalog: InputCatalog,
    metadata: tuple[MetadataRecord, ...],
    schedule: tuple[ScheduleRecord, ...],
    assignments: tuple[ShardAssignment, ...],
    renderer: Renderer,
    shards_fd: int,
    *,
    owner: str,
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

    root_fd = open_directory_path(root)
    try:
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
            )
        finally:
            os.close(shards_fd)
    finally:
        os.close(root_fd)


def _stage_owner_bytes(build_id: str) -> bytes:
    return canonical_json_bytes(
        {
            "build_id": build_id,
            "format": _FORMAT,
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


def _validate_stage_namespace(stage_fd: int, build_id: str) -> None:
    regular_names = {*_FOUNDATION_NAMES, _STAGE_OWNER_NAME, "receipt.json"}
    temporary_targets = {*_FOUNDATION_NAMES, "receipt.json"}
    for name in list_entries(stage_fd):
        metadata = entry_lstat(stage_fd, name)
        if name == "shards":
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("parallel corpus shards entry is unsafe")
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
) -> int:
    owner_payload = _stage_owner_bytes(build_id)
    fcntl.flock(stage_fd, fcntl.LOCK_EX)
    if created:
        if list_entries(stage_fd):
            raise ValueError("new parallel corpus staging directory is not empty")
        atomic_write_or_match(
            stage_fd,
            _STAGE_OWNER_NAME,
            owner_payload,
            owner=build_id,
        )
    else:
        try:
            actual_owner = read_regular_file(stage_fd, _STAGE_OWNER_NAME)
        except FileNotFoundError as error:
            raise ValueError("parallel corpus staging ownership marker is missing") from error
        if actual_owner != owner_payload:
            raise ValueError("parallel corpus staging ownership marker drift")
    _validate_stage_namespace(stage_fd, build_id)
    clean_owned_temporaries(
        stage_fd,
        final_names={*_FOUNDATION_NAMES, "receipt.json"},
        owner=build_id,
    )
    shards_fd, _created = open_directory_at(stage_fd, "shards", create=True)
    try:
        _validate_shard_namespace(shards_fd, shard_names, build_id)
        clean_owned_temporaries(
            shards_fd,
            final_names=shard_names,
            owner=build_id,
        )
    except BaseException:
        os.close(shards_fd)
        raise
    return shards_fd


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
) -> None:
    expected_stage = {*_FOUNDATION_NAMES, _STAGE_OWNER_NAME, "shards"}
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


def _publish_staging(
    *,
    parent_fd: int,
    stage_fd: int,
    stage_name: str,
    output_name: str,
    output_path: Path,
    build_id: str,
) -> dict[str, Any]:
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
        try:
            return verify_parallel_corpus(
                output_path,
                expected_build_id=build_id,
            )
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
    finally:
        os.close(published_fd)
    fsync_directory(parent_fd)
    return verify_parallel_corpus(
        output_path,
        expected_build_id=build_id,
    )


class _PackedReader:
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.index = 0
        self.handle: BinaryIO | None = None
        self.digest = hashlib.sha256()

    def _next_handle(self) -> bool:
        if self.handle is not None:
            self.handle.close()
        if self.index >= len(self.paths):
            self.handle = None
            return False
        self.handle = self.paths[self.index].open("rb")
        self.index += 1
        return True

    def consume(
        self,
        byte_count: int,
        *,
        local_digest: Any | None = None,
        require_zero: bool = False,
    ) -> None:
        remaining = byte_count
        while remaining:
            if self.handle is None and not self._next_handle():
                raise ValueError("packed shards end before their declared token stream")
            assert self.handle is not None
            chunk = self.handle.read(min(1 << 20, remaining))
            if not chunk:
                self.handle.close()
                self.handle = None
                continue
            self.digest.update(chunk)
            if local_digest is not None:
                local_digest.update(chunk)
            if require_zero and any(chunk):
                raise ValueError("packed update padding is not zero")
            remaining -= len(chunk)

    def finish(self) -> str:
        if self.handle is not None:
            if self.handle.read(1):
                raise ValueError("packed shards contain trailing bytes")
            self.handle.close()
            self.handle = None
        while self.index < len(self.paths):
            with self.paths[self.index].open("rb") as handle:
                if handle.read(1):
                    raise ValueError("packed shards contain trailing bytes")
            self.index += 1
        return self.digest.hexdigest()


def _validate_artifacts(
    root: Path,
    receipt: dict[str, Any],
    *,
    allow_stage_owner: bool = False,
) -> list[dict[str, Any]]:
    artifacts = receipt["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("publication artifacts must be a non-empty list")
    paths = []
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
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise ValueError("artifact path must be a safe relative POSIX path")
        if relative.as_posix() != relative_text:
            raise ValueError("artifact path is not canonical")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"publication artifact is missing or unsafe: {relative_text}")
        if (
            isinstance(artifact["bytes"], bool)
            or not isinstance(artifact["bytes"], int)
            or artifact["bytes"] < 0
            or path.stat().st_size != artifact["bytes"]
            or _sha256_file(path) != artifact["sha256"]
        ):
            raise ValueError(f"publication artifact digest drift: {relative_text}")
        paths.append(relative_text)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("publication artifact paths must be sorted and unique")
    actual_files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"publication contains a symlink: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    expected_files = {*paths, "receipt.json"}
    if allow_stage_owner:
        expected_files.add(_STAGE_OWNER_NAME)
    if actual_files != expected_files:
        raise ValueError("publication receipt is not hash-complete")
    return artifacts


def verify_parallel_corpus(
    root: Path | str,
    *,
    expected_build_id: str | None = None,
    _allow_stage_owner: bool = False,
) -> dict[str, Any]:
    publication = Path(root)
    if not publication.is_dir() or publication.is_symlink():
        raise ValueError("parallel corpus publication is missing or unsafe")
    receipt_path = publication / "receipt.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError("parallel corpus receipt is missing or unsafe")
    receipt_bytes = receipt_path.read_bytes()
    try:
        receipt = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("parallel corpus receipt is invalid JSON") from error
    if (
        not isinstance(receipt, dict)
        or set(receipt) != _RECEIPT_FIELDS
        or canonical_json_bytes(receipt) != receipt_bytes
    ):
        raise ValueError("parallel corpus receipt does not match the canonical contract")
    if (
        receipt["format"] != _FORMAT
        or receipt["compiler_version"] != _COMPILER_VERSION
    ):
        raise ValueError("parallel corpus format identity mismatch")
    artifacts = _validate_artifacts(
        publication,
        receipt,
        allow_stage_owner=_allow_stage_owner,
    )
    artifact_by_path = {artifact["path"]: artifact for artifact in artifacts}
    required_paths = {
        "assignments.jsonl",
        "catalog.jsonl",
        "metadata.jsonl",
        "schedule.jsonl",
    }
    if not required_paths <= set(artifact_by_path):
        raise ValueError("parallel corpus is missing required foundation artifacts")

    config = ParallelBuildConfig.from_dict(receipt["config"])
    renderer_id = receipt["renderer_id"]
    if not isinstance(renderer_id, str) or not renderer_id:
        raise ValueError("receipt renderer_id must be non-empty")
    catalog_bytes = (publication / "catalog.jsonl").read_bytes()
    metadata_bytes = (publication / "metadata.jsonl").read_bytes()
    schedule_bytes = (publication / "schedule.jsonl").read_bytes()
    assignment_bytes = (publication / "assignments.jsonl").read_bytes()
    catalog = InputCatalog.from_bytes(catalog_bytes)
    metadata = metadata_from_bytes(metadata_bytes)
    schedule = schedule_from_bytes(schedule_bytes)
    assignments = assignments_from_bytes(assignment_bytes)
    reduced = reduce_metadata(
        catalog,
        metadata,
        expected_renderer_id=renderer_id,
    )
    expected_schedule = largest_deficit_schedule(reduced, config.lane_weights)
    if schedule != expected_schedule:
        raise ValueError("published schedule is not the canonical deficit reduction")
    expected_assignments = assign_update_aligned_shards(
        total_tokens=schedule[-1].token_end,
        update_tokens=config.update_tokens,
        shard_count=config.shard_count,
        allow_fewer=config.allow_fewer_shards,
    )
    if assignments != expected_assignments:
        raise ValueError("published shard assignments are not canonical")
    ordered_hash, merkle_root = ordered_stream_commitments(schedule, reduced)
    build_id = parallel_build_id(catalog, renderer_id, config)
    logical_tokens = schedule[-1].token_end
    packed_tokens = assignments[-1].token_end
    expected_scalars = {
        "assignments_sha256": sha256_hex(assignment_bytes),
        "build_id": build_id,
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
    if expected_build_id is not None and build_id != expected_build_id:
        raise ValueError("parallel corpus build id does not match expectation")

    shard_paths = [
        f"shards/{assignment.shard_id}.bin" for assignment in assignments
    ]
    if set(artifact_by_path) - required_paths != set(shard_paths):
        raise ValueError("parallel corpus shard namespace does not match assignments")
    for assignment, relative in zip(assignments, shard_paths, strict=True):
        expected_bytes = (assignment.token_end - assignment.token_start) * 2
        if artifact_by_path[relative]["bytes"] != expected_bytes:
            raise ValueError(f"parallel corpus shard size drift: {relative}")
    reader = _PackedReader([publication / path for path in shard_paths])
    metadata_by_id = {record.record_id: record for record in metadata}
    for entry in schedule:
        digest = hashlib.sha256()
        reader.consume(entry.token_length * 2, local_digest=digest)
        if digest.hexdigest() != metadata_by_id[entry.record_id].render_sha256:
            raise ValueError(f"packed record digest drift: {entry.record_id}")
    reader.consume((packed_tokens - logical_tokens) * 2, require_zero=True)
    packed_digest = reader.finish()
    if receipt["packed_stream_sha256"] != packed_digest:
        raise ValueError("packed stream digest drift")
    return receipt


def build_parallel_corpus(
    catalog: InputCatalog,
    renderer: Renderer,
    config: ParallelBuildConfig,
    destination: Path | str,
    *,
    workers: int = 1,
    _materialized_metadata: tuple[MetadataRecord, ...] | None = None,
    _cached_payloads: dict[str, bytes] | None = None,
) -> dict[str, Any]:
    """Build or resume a corpus using pinned, no-replace publication."""

    output = Path(destination)
    build_id = parallel_build_id(catalog, renderer.renderer_id, config)
    parent_fd, output_name = open_parent_directory(output, create=True)
    stage_path = publication_staging_path(output, build_id)
    stage_name = stage_path.name
    owner_payload = _stage_owner_bytes(build_id)
    try:
        if entry_exists(parent_fd, output_name):
            try:
                return verify_parallel_corpus(
                    output,
                    expected_build_id=build_id,
                )
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"conflicting parallel corpus output: {output}"
                ) from error

        if entry_exists(parent_fd, stage_name):
            stage_metadata = entry_lstat(parent_fd, stage_name)
            if not stat.S_ISDIR(stage_metadata.st_mode):
                raise ValueError("parallel corpus staging path is unsafe")
            existing_stage_fd, _created = open_directory_at(parent_fd, stage_name)
            try:
                fcntl.flock(existing_stage_fd, fcntl.LOCK_EX)
                _validate_stage_namespace(existing_stage_fd, build_id)
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
                    verify_parallel_corpus(
                        stage_path,
                        expected_build_id=build_id,
                        _allow_stage_owner=has_owner,
                    )
                    if has_owner:
                        unlink_regular_if_matches(
                            existing_stage_fd,
                            _STAGE_OWNER_NAME,
                            owner_payload,
                        )
                    return _publish_staging(
                        parent_fd=parent_fd,
                        stage_fd=existing_stage_fd,
                        stage_name=stage_name,
                        output_name=output_name,
                        output_path=output,
                        build_id=build_id,
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

        stage_fd, created = open_directory_at(
            parent_fd,
            stage_name,
            create=True,
        )
        shards_fd = -1
        try:
            shards_fd = _prepare_staging(
                stage_fd,
                created=created,
                build_id=build_id,
                shard_names=shard_names,
            )
            atomic_write_or_match(
                stage_fd,
                "catalog.jsonl",
                catalog_bytes,
                owner=build_id,
            )
            atomic_write_or_match(
                stage_fd,
                "metadata.jsonl",
                metadata_bytes,
                owner=build_id,
            )
            atomic_write_or_match(
                stage_fd,
                "schedule.jsonl",
                schedule_bytes,
                owner=build_id,
            )
            atomic_write_or_match(
                stage_fd,
                "assignments.jsonl",
                assignment_bytes,
                owner=build_id,
            )
            packed = _rerender_and_pack_pinned(
                catalog,
                metadata,
                schedule,
                assignments,
                renderer,
                shards_fd,
                owner=build_id,
                cached_payloads=_cached_payloads,
            )
            _assert_complete_stage(stage_fd, shards_fd, shard_names)
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
            logical_tokens = schedule[-1].token_end
            packed_tokens = assignments[-1].token_end
            receipt = {
                "artifacts": artifacts,
                "assignments_sha256": sha256_hex(assignment_bytes),
                "build_id": build_id,
                "catalog_sha256": catalog.sha256,
                "compiler_version": _COMPILER_VERSION,
                "config": config.as_dict(),
                "format": _FORMAT,
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
            atomic_write_or_match(
                stage_fd,
                "receipt.json",
                canonical_json_bytes(receipt),
                owner=build_id,
            )
            fsync_directory(shards_fd)
            fsync_directory(stage_fd)
            verify_parallel_corpus(
                stage_path,
                expected_build_id=build_id,
                _allow_stage_owner=True,
            )
            unlink_regular_if_matches(
                stage_fd,
                _STAGE_OWNER_NAME,
                owner_payload,
            )
            verify_parallel_corpus(
                stage_path,
                expected_build_id=build_id,
            )
            return _publish_staging(
                parent_fd=parent_fd,
                stage_fd=stage_fd,
                stage_name=stage_name,
                output_name=output_name,
                output_path=output,
                build_id=build_id,
            )
        finally:
            if shards_fd >= 0:
                os.close(shards_fd)
            os.close(stage_fd)
    finally:
        os.close(parent_fd)


def build_parallel_corpus_from_tasks(
    catalog: InputCatalog,
    renderer_id: str,
    config: ParallelBuildConfig,
    destination: Path | str,
    task_results: tuple[TaskResult, ...],
    *,
    expected_task_count: int,
) -> dict[str, Any]:
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
) -> dict[str, Any]:
    """Verify a corpus and install its canonical receipt under a pinned fd."""

    receipt = verify_parallel_corpus(
        corpus,
        expected_build_id=expected_build_id,
    )
    receipt_bytes = canonical_json_bytes(receipt)
    parent_fd, name = open_parent_directory(destination, create=True)
    try:
        atomic_write_or_match(
            parent_fd,
            name,
            receipt_bytes,
            owner=expected_build_id,
        )
    finally:
        os.close(parent_fd)
    return receipt
