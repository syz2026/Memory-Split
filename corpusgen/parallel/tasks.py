"""Hash-bound, disjoint metadata and payload task results."""

from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .canonical import canonical_json_bytes, sha256_hex
from .catalog import CatalogRecord, InputCatalog
from .metadata import MetadataRecord, Renderer, reduce_metadata, token_bytes

if TYPE_CHECKING:
    from .publication import ParallelBuildConfig

_TASK_FORMAT = "memorysplit-parallel-task-v1"
_TASK_FIELDS = {
    "build_id",
    "catalog_sha256",
    "format",
    "records",
    "renderer_id",
    "result_sha256",
    "task_count",
    "task_index",
}
_CACHED_RECORD_FIELDS = {
    "metadata",
    "payload_base64",
    "payload_sha256",
}


def _sha256_text(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be lowercase SHA-256")
    return value


def _task_bounds(task_index: object, task_count: object) -> tuple[int, int]:
    if (
        isinstance(task_count, bool)
        or not isinstance(task_count, int)
        or task_count <= 0
    ):
        raise ValueError("task_count must be a positive integer")
    if (
        isinstance(task_index, bool)
        or not isinstance(task_index, int)
        or not 0 <= task_index < task_count
    ):
        raise ValueError("task_index must be within task_count")
    return task_index, task_count


def partition_ordinals(
    *,
    record_count: int,
    task_index: int,
    task_count: int,
) -> tuple[int, ...]:
    """Return the stable strided ordinal partition for one task."""

    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count <= 0
    ):
        raise ValueError("record_count must be a positive integer")
    task_index, task_count = _task_bounds(task_index, task_count)
    return tuple(range(task_index, record_count, task_count))


@dataclass(frozen=True)
class CachedTaskRecord:
    metadata: MetadataRecord
    payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, MetadataRecord):
            raise TypeError("cached task metadata must be a MetadataRecord")
        if not isinstance(self.payload, bytes):
            raise TypeError("cached task payload must be bytes")
        if len(self.payload) != self.metadata.token_length * 2:
            raise ValueError(
                f"cached payload token length mismatch: {self.metadata.record_id}"
            )
        if sha256_hex(self.payload) != self.metadata.render_sha256:
            raise ValueError(
                f"cached payload digest mismatch: {self.metadata.record_id}"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "metadata": self.metadata.as_dict(),
            "payload_base64": base64.b64encode(self.payload).decode("ascii"),
            "payload_sha256": self.metadata.render_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> CachedTaskRecord:
        if not isinstance(value, dict) or set(value) != _CACHED_RECORD_FIELDS:
            raise ValueError("cached task record fields do not match the contract")
        encoded = value["payload_base64"]
        if not isinstance(encoded, str):
            raise ValueError("cached payload must be canonical base64")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as error:
            raise ValueError("cached payload must be canonical base64") from error
        if base64.b64encode(payload).decode("ascii") != encoded:
            raise ValueError("cached payload must be canonical base64")
        metadata = MetadataRecord.from_dict(value["metadata"])
        if value["payload_sha256"] != metadata.render_sha256:
            raise ValueError(f"cached payload digest mismatch: {metadata.record_id}")
        return cls(metadata=metadata, payload=payload)


@dataclass(frozen=True)
class TaskResult:
    build_id: str
    catalog_sha256: str
    renderer_id: str
    task_index: int
    task_count: int
    records: tuple[CachedTaskRecord, ...]
    result_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _sha256_text(self.build_id, "task build_id")
        _sha256_text(self.catalog_sha256, "task catalog_sha256")
        _task_bounds(self.task_index, self.task_count)
        if not isinstance(self.renderer_id, str) or not self.renderer_id:
            raise ValueError("task renderer_id must be non-empty")
        if not isinstance(self.records, tuple):
            raise TypeError("task records must be a tuple")
        ordinals = [record.metadata.ordinal for record in self.records]
        if ordinals != sorted(ordinals) or len(ordinals) != len(set(ordinals)):
            raise ValueError("task records must have unique increasing ordinals")
        if any(
            record.metadata.renderer_id != self.renderer_id
            for record in self.records
        ):
            raise ValueError("task record renderer mismatch")
        object.__setattr__(
            self,
            "result_sha256",
            sha256_hex(canonical_json_bytes(self._core_dict())),
        )

    def _core_dict(self) -> dict[str, object]:
        return {
            "build_id": self.build_id,
            "catalog_sha256": self.catalog_sha256,
            "format": _TASK_FORMAT,
            "records": [record.as_dict() for record in self.records],
            "renderer_id": self.renderer_id,
            "task_count": self.task_count,
            "task_index": self.task_index,
        }

    def as_dict(self) -> dict[str, object]:
        return {**self._core_dict(), "result_sha256": self.result_sha256}

    @classmethod
    def from_dict(cls, value: object) -> TaskResult:
        if not isinstance(value, dict) or set(value) != _TASK_FIELDS:
            raise ValueError("task result fields do not match the contract")
        if value["format"] != _TASK_FORMAT:
            raise ValueError("task result format mismatch")
        raw_records = value["records"]
        if not isinstance(raw_records, list):
            raise ValueError("task result records must be a list")
        result = cls(
            build_id=value["build_id"],
            catalog_sha256=value["catalog_sha256"],
            renderer_id=value["renderer_id"],
            task_index=value["task_index"],
            task_count=value["task_count"],
            records=tuple(
                CachedTaskRecord.from_dict(record) for record in raw_records
            ),
        )
        if value["result_sha256"] != result.result_sha256:
            raise ValueError("task result digest mismatch")
        return result


def _render_cached(record: CatalogRecord, renderer: Renderer) -> CachedTaskRecord:
    rendered = renderer.render(record)
    if not set(record.flags) <= set(rendered.flags):
        raise ValueError(f"renderer dropped catalog flags: {record.record_id}")
    payload = token_bytes(rendered.token_ids)
    metadata = MetadataRecord(
        ordinal=record.ordinal,
        record_id=record.record_id,
        lane=record.lane,
        source=record.source,
        source_key=record.source_key,
        renderer_id=renderer.renderer_id,
        token_length=len(rendered.token_ids),
        flags=rendered.flags,
        source_sha256=record.payload_sha256,
        render_sha256=sha256_hex(payload),
    )
    return CachedTaskRecord(metadata=metadata, payload=payload)


def render_task_result(
    catalog: InputCatalog,
    renderer: Renderer,
    config: ParallelBuildConfig,
    *,
    task_index: int,
    task_count: int,
    workers: int = 1,
) -> TaskResult:
    """Render exactly one deterministic ordinal partition."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if not isinstance(renderer.renderer_id, str) or not renderer.renderer_id:
        raise ValueError("renderer_id must be non-empty")
    ordinals = partition_ordinals(
        record_count=len(catalog.records),
        task_index=task_index,
        task_count=task_count,
    )
    selected = [catalog.records[ordinal] for ordinal in ordinals]
    if workers == 1:
        candidates = [_render_cached(record, renderer) for record in selected]
    else:
        candidates = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_render_cached, record, renderer)
                for record in selected
            ]
            for future in as_completed(futures):
                candidates.append(future.result())
    candidates.sort(key=lambda record: record.metadata.ordinal)
    from .publication import parallel_build_id

    return TaskResult(
        build_id=parallel_build_id(catalog, renderer.renderer_id, config),
        catalog_sha256=catalog.sha256,
        renderer_id=renderer.renderer_id,
        task_index=task_index,
        task_count=task_count,
        records=tuple(candidates),
    )


def reduce_task_results(
    catalog: InputCatalog,
    renderer_id: str,
    config: ParallelBuildConfig,
    results: tuple[TaskResult, ...],
    *,
    expected_task_count: int,
) -> tuple[tuple[MetadataRecord, ...], dict[str, bytes]]:
    """Validate complete disjoint results and restore canonical catalog order."""

    _task_bounds(0, expected_task_count)
    if not isinstance(results, tuple):
        raise TypeError("task results must be a tuple")
    indexes = [result.task_index for result in results]
    if len(indexes) != len(set(indexes)):
        raise ValueError("duplicate task result")
    expected_indexes = set(range(expected_task_count))
    actual_indexes = set(indexes)
    missing = sorted(expected_indexes - actual_indexes)
    if missing:
        raise ValueError(
            "missing task results: " + ", ".join(str(index) for index in missing)
        )
    unexpected = sorted(actual_indexes - expected_indexes)
    if unexpected or len(results) != expected_task_count:
        raise ValueError("unexpected task results")
    from .publication import parallel_build_id

    expected_build_id = parallel_build_id(catalog, renderer_id, config)
    cached_records = []
    for result in results:
        if result.build_id != expected_build_id:
            raise ValueError("task result build id mismatch")
        if result.catalog_sha256 != catalog.sha256:
            raise ValueError("task result catalog digest mismatch")
        if result.renderer_id != renderer_id:
            raise ValueError("task result renderer mismatch")
        if result.task_count != expected_task_count:
            raise ValueError("task result task_count mismatch")
        expected_ordinals = partition_ordinals(
            record_count=len(catalog.records),
            task_index=result.task_index,
            task_count=expected_task_count,
        )
        actual_ordinals = tuple(
            record.metadata.ordinal for record in result.records
        )
        if actual_ordinals != expected_ordinals:
            raise ValueError(
                f"task result ordinal partition mismatch: {result.task_index}"
            )
        cached_records.extend(result.records)
    metadata = reduce_metadata(
        catalog,
        tuple(record.metadata for record in cached_records),
        expected_renderer_id=renderer_id,
    )
    payload_by_id = {
        record.metadata.record_id: record.payload for record in cached_records
    }
    if len(payload_by_id) != len(cached_records):
        raise ValueError("duplicate cached task record")
    return metadata, {
        record.record_id: payload_by_id[record.record_id]
        for record in catalog.records
    }


def task_result_to_bytes(result: TaskResult) -> bytes:
    if not isinstance(result, TaskResult):
        raise TypeError("task result must be TaskResult")
    return canonical_json_bytes(result.as_dict())


def task_result_from_bytes(payload: bytes) -> TaskResult:
    if not isinstance(payload, bytes) or not payload.endswith(b"\n"):
        raise ValueError("task result must be newline-terminated bytes")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("task result contains invalid JSON") from error
    if canonical_json_bytes(value) != payload:
        raise ValueError("task result JSON is not canonical")
    return TaskResult.from_dict(value)


def task_result_filename(result: TaskResult) -> str:
    if not isinstance(result, TaskResult):
        raise TypeError("task result must be TaskResult")
    return (
        f"task-{result.task_index:05d}-of-{result.task_count:05d}-"
        f"{result.build_id}.json"
    )
