"""Rerendering, packing, receipts, resumability, and atomic publication."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

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

_FORMAT = "memorysplit-parallel-corpus-v1"
_COMPILER_VERSION = "metadata-first-foundation-v1"
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


def _atomic_write_or_match(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"resume artifact is missing or unsafe: {path}")
        if path.read_bytes() != payload:
            raise ValueError(f"resume artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            if (
                not path.is_file()
                or path.is_symlink()
                or path.read_bytes() != payload
            ):
                raise ValueError(f"concurrent artifact drift: {path}")
        else:
            temporary.rename(path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


class _ShardSink:
    def __init__(self, root: Path, assignment: ShardAssignment) -> None:
        self.assignment = assignment
        self.final = root / "shards" / f"{assignment.shard_id}.bin"
        self.final.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.final.with_name(
            f".{self.final.name}.tmp-{os.getpid()}"
        )
        if self.temporary.exists() or self.temporary.is_symlink():
            self.temporary.unlink()
        self.handle: BinaryIO = self.temporary.open("xb")
        self.digest = hashlib.sha256()
        self.byte_count = 0

    def write(self, payload: bytes) -> None:
        if len(payload) % 2:
            raise ValueError("packed uint16 payload has odd byte length")
        self.handle.write(payload)
        self.digest.update(payload)
        self.byte_count += len(payload)

    def finish(self) -> dict[str, object]:
        expected = (self.assignment.token_end - self.assignment.token_start) * 2
        try:
            self.handle.flush()
            os.fsync(self.handle.fileno())
        finally:
            self.handle.close()
        if self.byte_count != expected:
            self.abort()
            raise ValueError(f"shard byte count drift: {self.assignment.shard_id}")
        digest = self.digest.hexdigest()
        if self.final.exists() or self.final.is_symlink():
            if (
                not self.final.is_file()
                or self.final.is_symlink()
                or self.final.stat().st_size != expected
                or _sha256_file(self.final) != digest
            ):
                self.abort()
                raise ValueError(f"resume shard drift: {self.assignment.shard_id}")
            self.temporary.unlink()
        else:
            self.temporary.rename(self.final)
        return _artifact(self.final.parents[1], self.final)

    def abort(self) -> None:
        if not self.handle.closed:
            self.handle.close()
        if self.temporary.exists() or self.temporary.is_symlink():
            self.temporary.unlink()


def rerender_and_pack(
    catalog: InputCatalog,
    metadata: tuple[MetadataRecord, ...],
    schedule: tuple[ScheduleRecord, ...],
    assignments: tuple[ShardAssignment, ...],
    renderer: Renderer,
    root: Path | str,
) -> dict[str, object]:
    """Rerender records, verify metadata, and atomically install packed shards."""

    output_root = Path(root)
    if not output_root.is_dir() or output_root.is_symlink():
        raise ValueError("pack root must be an existing regular directory")
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
    if not assignments or assignments[0].token_start != 0:
        raise ValueError("pack assignments must cover the stream from zero")
    logical_tokens = schedule[-1].token_end
    if assignments[-1].token_end < logical_tokens:
        raise ValueError("pack assignments do not cover the logical stream")
    for temporary in (output_root / "shards").glob(".*.tmp-*"):
        if temporary.is_file() or temporary.is_symlink():
            temporary.unlink()

    packed_digest = hashlib.sha256()
    shard_artifacts = []
    assignment_index = 0
    sink = _ShardSink(output_root, assignments[assignment_index])
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
                sink = _ShardSink(output_root, assignments[assignment_index])
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
            rendered = renderer.render(source)
            payload = token_bytes(rendered.token_ids)
            if (
                len(rendered.token_ids) != expected.token_length
                or rendered.flags != expected.flags
                or sha256_hex(payload) != expected.render_sha256
            ):
                raise ValueError(f"rerender drift: {entry.record_id}")
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


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in [*sorted(directories, reverse=True), root]:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def _validate_artifacts(root: Path, receipt: dict[str, Any]) -> list[dict[str, Any]]:
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
    if actual_files != {*paths, "receipt.json"}:
        raise ValueError("publication receipt is not hash-complete")
    return artifacts


def verify_parallel_corpus(
    root: Path | str,
    *,
    expected_build_id: str | None = None,
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
    artifacts = _validate_artifacts(publication, receipt)
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
) -> dict[str, Any]:
    """Build or resume a corpus and publish it with one final directory rename."""

    output = Path(destination)
    build_id = parallel_build_id(catalog, renderer.renderer_id, config)
    if output.exists() or output.is_symlink():
        try:
            return verify_parallel_corpus(output, expected_build_id=build_id)
        except (OSError, ValueError) as error:
            raise ValueError(f"conflicting parallel corpus output: {output}") from error
    parent = output.parent
    if parent.exists() and parent.is_symlink():
        raise ValueError("parallel corpus parent must not be a symlink")
    parent.mkdir(parents=True, exist_ok=True)
    staging = publication_staging_path(output, build_id)
    if staging.exists() or staging.is_symlink():
        if not staging.is_dir() or staging.is_symlink():
            raise ValueError(f"parallel corpus staging path is unsafe: {staging}")
        if (staging / "receipt.json").exists():
            verify_parallel_corpus(staging, expected_build_id=build_id)
            if output.exists() or output.is_symlink():
                raise ValueError("parallel corpus output appeared during resume")
            staging.rename(output)
            _fsync_directory(parent)
            return verify_parallel_corpus(output, expected_build_id=build_id)
    else:
        staging.mkdir()

    metadata = render_metadata(catalog, renderer, workers=workers)
    schedule = largest_deficit_schedule(metadata, config.lane_weights)
    assignments = assign_update_aligned_shards(
        total_tokens=schedule[-1].token_end,
        update_tokens=config.update_tokens,
        shard_count=config.shard_count,
        allow_fewer=config.allow_fewer_shards,
    )
    catalog_bytes = catalog.to_bytes()
    metadata_bytes = metadata_to_bytes(metadata)
    schedule_bytes = schedule_to_bytes(schedule)
    assignment_bytes = assignments_to_bytes(assignments)
    _atomic_write_or_match(staging / "catalog.jsonl", catalog_bytes)
    _atomic_write_or_match(staging / "metadata.jsonl", metadata_bytes)
    _atomic_write_or_match(staging / "schedule.jsonl", schedule_bytes)
    _atomic_write_or_match(staging / "assignments.jsonl", assignment_bytes)
    packed = rerender_and_pack(
        catalog,
        metadata,
        schedule,
        assignments,
        renderer,
        staging,
    )
    ordered_hash, merkle_root = ordered_stream_commitments(schedule, metadata)
    artifact_paths = [
        path
        for path in staging.rglob("*")
        if path.is_file()
        and path.name != "receipt.json"
        and ".tmp-" not in path.name
    ]
    artifacts = sorted(
        (_artifact(staging, path) for path in artifact_paths),
        key=lambda item: item["path"],
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
    _atomic_write_or_match(staging / "receipt.json", canonical_json_bytes(receipt))
    verify_parallel_corpus(staging, expected_build_id=build_id)
    _fsync_tree(staging)
    if output.exists() or output.is_symlink():
        raise ValueError("parallel corpus output appeared during build")
    staging.rename(output)
    _fsync_directory(parent)
    return verify_parallel_corpus(output, expected_build_id=build_id)
