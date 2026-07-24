"""Pure deterministic schedule and update-aligned shard reducers."""

from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import dataclass
from fractions import Fraction

from .canonical import canonical_json_bytes
from .metadata import MetadataRecord

_SCHEDULE_FIELDS = {
    "lane",
    "metadata_sha256",
    "record_id",
    "sequence",
    "token_end",
    "token_length",
    "token_start",
}
_ASSIGNMENT_FIELDS = {
    "shard_count",
    "shard_index",
    "token_end",
    "token_start",
    "update_end",
    "update_start",
}


@dataclass(frozen=True)
class ScheduleRecord:
    sequence: int
    record_id: str
    lane: str
    metadata_sha256: str
    token_length: int
    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("schedule sequence must be a non-negative integer")
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ValueError("schedule record_id must be non-empty")
        if not isinstance(self.lane, str) or not self.lane:
            raise ValueError("schedule lane must be non-empty")
        if (
            not isinstance(self.metadata_sha256, str)
            or len(self.metadata_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.metadata_sha256)
        ):
            raise ValueError("schedule metadata_sha256 must be lowercase SHA-256")
        if (
            isinstance(self.token_length, bool)
            or not isinstance(self.token_length, int)
            or self.token_length <= 0
        ):
            raise ValueError("schedule token_length must be positive")
        if (
            isinstance(self.token_start, bool)
            or not isinstance(self.token_start, int)
            or self.token_start < 0
            or self.token_end != self.token_start + self.token_length
        ):
            raise ValueError("schedule token span is invalid")

    def as_dict(self) -> dict[str, object]:
        return {
            "lane": self.lane,
            "metadata_sha256": self.metadata_sha256,
            "record_id": self.record_id,
            "sequence": self.sequence,
            "token_end": self.token_end,
            "token_length": self.token_length,
            "token_start": self.token_start,
        }

    @classmethod
    def from_dict(cls, value: object) -> ScheduleRecord:
        if not isinstance(value, dict) or set(value) != _SCHEDULE_FIELDS:
            raise ValueError("schedule record fields do not match the contract")
        return cls(
            sequence=value["sequence"],
            record_id=value["record_id"],
            lane=value["lane"],
            metadata_sha256=value["metadata_sha256"],
            token_length=value["token_length"],
            token_start=value["token_start"],
            token_end=value["token_end"],
        )


@dataclass(frozen=True)
class ShardAssignment:
    shard_index: int
    shard_count: int
    update_start: int
    update_end: int
    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        values = (
            self.shard_index,
            self.shard_count,
            self.update_start,
            self.update_end,
            self.token_start,
            self.token_end,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("shard assignment fields must be integers")
        if (
            self.shard_count <= 0
            or not 0 <= self.shard_index < self.shard_count
            or not 0 <= self.update_start < self.update_end
            or not 0 <= self.token_start < self.token_end
        ):
            raise ValueError("shard assignment bounds are invalid")

    @property
    def shard_id(self) -> str:
        return f"shard-{self.shard_index:05d}-of-{self.shard_count:05d}"

    def as_dict(self) -> dict[str, int]:
        return {
            "shard_count": self.shard_count,
            "shard_index": self.shard_index,
            "token_end": self.token_end,
            "token_start": self.token_start,
            "update_end": self.update_end,
            "update_start": self.update_start,
        }

    @classmethod
    def from_dict(cls, value: object) -> ShardAssignment:
        if not isinstance(value, dict) or set(value) != _ASSIGNMENT_FIELDS:
            raise ValueError("shard assignment fields do not match the contract")
        return cls(
            shard_index=value["shard_index"],
            shard_count=value["shard_count"],
            update_start=value["update_start"],
            update_end=value["update_end"],
            token_start=value["token_start"],
            token_end=value["token_end"],
        )


def _validate_weights(
    weights: tuple[tuple[str, int], ...],
    lanes: set[str],
) -> tuple[tuple[str, int], ...]:
    if not isinstance(weights, tuple) or not weights:
        raise ValueError("lane weights must be a non-empty tuple")
    names = []
    for item in weights:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or isinstance(item[1], bool)
            or not isinstance(item[1], int)
            or item[1] <= 0
        ):
            raise ValueError("lane weights must contain (non-empty lane, positive int)")
        names.append(item[0])
    if len(names) != len(set(names)):
        raise ValueError("lane weights contain duplicate lanes")
    if set(names) != lanes:
        raise ValueError("lane weights must cover exactly the metadata lanes")
    return weights


def largest_deficit_schedule(
    metadata: tuple[MetadataRecord, ...],
    lane_weights: tuple[tuple[str, int], ...],
) -> tuple[ScheduleRecord, ...]:
    """Order whole records by largest remaining exact token deficit."""

    if not isinstance(metadata, tuple) or not metadata:
        raise ValueError("metadata must be a non-empty tuple")
    ordered = tuple(sorted(metadata, key=lambda record: record.ordinal))
    if len({record.ordinal for record in ordered}) != len(ordered):
        raise ValueError("metadata contains duplicate ordinals")
    if len({record.record_id for record in ordered}) != len(ordered):
        raise ValueError("metadata contains duplicate record ids")
    weights = _validate_weights(lane_weights, {record.lane for record in ordered})
    lane_order = {lane: index for index, (lane, _weight) in enumerate(weights)}
    queues = {
        lane: deque(record for record in ordered if record.lane == lane)
        for lane, _weight in weights
    }
    total_tokens = sum(record.token_length for record in ordered)
    total_weight = sum(weight for _lane, weight in weights)
    targets = {
        lane: Fraction(total_tokens * weight, total_weight)
        for lane, weight in weights
    }
    emitted: Counter[str] = Counter()
    active = {lane for lane, queue in queues.items() if queue}
    result = []
    token_start = 0
    while active:
        lane = min(
            active,
            key=lambda name: (
                -(targets[name] - emitted[name]),
                lane_order[name],
            ),
        )
        record = queues[lane].popleft()
        token_end = token_start + record.token_length
        result.append(
            ScheduleRecord(
                sequence=len(result),
                record_id=record.record_id,
                lane=record.lane,
                metadata_sha256=record.metadata_sha256,
                token_length=record.token_length,
                token_start=token_start,
                token_end=token_end,
            )
        )
        emitted[lane] += record.token_length
        token_start = token_end
        if not queues[lane]:
            active.remove(lane)
    return tuple(result)


def schedule_to_bytes(records: tuple[ScheduleRecord, ...]) -> bytes:
    if not records:
        raise ValueError("schedule must not be empty")
    expected_start = 0
    seen = set()
    for sequence, record in enumerate(records):
        if record.sequence != sequence or record.token_start != expected_start:
            raise ValueError("schedule sequence or token spans are not contiguous")
        if record.record_id in seen:
            raise ValueError(f"schedule contains duplicate record: {record.record_id}")
        seen.add(record.record_id)
        expected_start = record.token_end
    return b"".join(canonical_json_bytes(record.as_dict()) for record in records)


def schedule_from_bytes(payload: bytes) -> tuple[ScheduleRecord, ...]:
    if not isinstance(payload, bytes) or not payload or not payload.endswith(b"\n"):
        raise ValueError("schedule must be non-empty newline-terminated bytes")
    records = []
    for line in payload.splitlines(keepends=True):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("schedule contains invalid JSON") from error
        if canonical_json_bytes(value) != line:
            raise ValueError("schedule JSONL is not canonical")
        records.append(ScheduleRecord.from_dict(value))
    result = tuple(records)
    if schedule_to_bytes(result) != payload:
        raise ValueError("schedule bytes are not canonical")
    return result


def assign_update_aligned_shards(
    *,
    total_tokens: int,
    update_tokens: int,
    shard_count: int = 32,
    allow_fewer: bool = False,
) -> tuple[ShardAssignment, ...]:
    """Split the padded stream into contiguous whole-update assignments."""

    for value, name in (
        (total_tokens, "total_tokens"),
        (update_tokens, "update_tokens"),
        (shard_count, "shard_count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(allow_fewer, bool):
        raise ValueError("allow_fewer must be boolean")
    update_count = (total_tokens + update_tokens - 1) // update_tokens
    if update_count < shard_count:
        if not allow_fewer:
            raise ValueError(
                f"stream has fewer updates ({update_count}) than shards ({shard_count})"
            )
        actual_shards = update_count
    else:
        actual_shards = shard_count
    base, remainder = divmod(update_count, actual_shards)
    assignments = []
    update_start = 0
    for shard_index in range(actual_shards):
        updates = base + (1 if shard_index < remainder else 0)
        update_end = update_start + updates
        assignments.append(
            ShardAssignment(
                shard_index=shard_index,
                shard_count=actual_shards,
                update_start=update_start,
                update_end=update_end,
                token_start=update_start * update_tokens,
                token_end=update_end * update_tokens,
            )
        )
        update_start = update_end
    return tuple(assignments)


def assignments_to_bytes(assignments: tuple[ShardAssignment, ...]) -> bytes:
    if not assignments:
        raise ValueError("shard assignments must not be empty")
    expected_token_start = 0
    expected_update_start = 0
    shard_count = len(assignments)
    for shard_index, assignment in enumerate(assignments):
        if (
            assignment.shard_index != shard_index
            or assignment.shard_count != shard_count
            or assignment.token_start != expected_token_start
            or assignment.update_start != expected_update_start
        ):
            raise ValueError("shard assignments are not contiguous")
        expected_token_start = assignment.token_end
        expected_update_start = assignment.update_end
    return b"".join(
        canonical_json_bytes(assignment.as_dict()) for assignment in assignments
    )


def assignments_from_bytes(payload: bytes) -> tuple[ShardAssignment, ...]:
    if not isinstance(payload, bytes) or not payload or not payload.endswith(b"\n"):
        raise ValueError("assignments must be non-empty newline-terminated bytes")
    assignments = []
    for line in payload.splitlines(keepends=True):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("assignments contain invalid JSON") from error
        if canonical_json_bytes(value) != line:
            raise ValueError("assignment JSONL is not canonical")
        assignments.append(ShardAssignment.from_dict(value))
    result = tuple(assignments)
    if assignments_to_bytes(result) != payload:
        raise ValueError("assignment bytes are not canonical")
    return result
