from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

Direction = Literal["out", "in"]
TargetKind = Literal["entity", "literal"]
SegmentRole = Literal[
    "plain",
    "payload",
    "random_control",
    "relation_alias",
    "rule",
    "action",
    "provisional_answer",
    "final_answer",
]
RANDOM_CONTROL_POSITION_BINS = 10


def relative_position_bin(
    start: int,
    end: int,
    document_length: int,
) -> int:
    if not 0 <= start < end <= document_length:
        raise ValueError("span must be non-empty and inside the document")
    return min(
        RANDOM_CONTROL_POSITION_BINS - 1,
        (
            (start + end) * RANDOM_CONTROL_POSITION_BINS
        )
        // (2 * document_length),
    )


@dataclass(frozen=True, order=True)
class GraphAddress:
    source_id: int
    relation_id: str
    direction: Direction

    def __post_init__(self) -> None:
        if self.source_id < 0:
            raise ValueError("source_id must be non-negative")
        if not self.relation_id:
            raise ValueError("relation_id must be non-empty")
        if self.direction not in ("out", "in"):
            raise ValueError(f"invalid direction: {self.direction}")


@dataclass(frozen=True, order=True)
class GraphRow:
    source_id: int
    relation_id: str
    direction: Direction
    target_kind: TargetKind
    target: str
    qualifiers: tuple[tuple[str, str], ...] = ()
    provenance_id: str = ""

    def __post_init__(self) -> None:
        GraphAddress(self.source_id, self.relation_id, self.direction)
        if self.target_kind not in ("entity", "literal"):
            raise ValueError(f"invalid target_kind: {self.target_kind}")
        if not self.target:
            raise ValueError("target must be non-empty")

    @property
    def address(self) -> GraphAddress:
        return GraphAddress(self.source_id, self.relation_id, self.direction)

    def as_json(self) -> dict:
        return {
            "source_id": self.source_id,
            "relation_id": self.relation_id,
            "direction": self.direction,
            "target_kind": self.target_kind,
            "target": self.target,
            "qualifiers": [list(q) for q in self.qualifiers],
            "provenance_id": self.provenance_id,
        }

    @classmethod
    def from_json(cls, value: dict) -> "GraphRow":
        return cls(
            source_id=int(value["source_id"]),
            relation_id=str(value["relation_id"]),
            direction=value["direction"],
            target_kind=value["target_kind"],
            target=str(value["target"]),
            qualifiers=tuple((str(k), str(v)) for k, v in value["qualifiers"]),
            provenance_id=str(value["provenance_id"]),
        )


def stable_fact_id(row: GraphRow) -> str:
    canonical = {
        "provenance_id": row.provenance_id,
        "source_id": row.source_id,
        "relation_id": row.relation_id,
        "direction": row.direction,
        "target_kind": row.target_kind,
        "target": row.target,
        "qualifiers": [list(value) for value in row.qualifiers],
    }
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TaggedSegment:
    text: str
    role: SegmentRole
    fact_id: str | None = None
    payload_field: str | None = None

    def __post_init__(self) -> None:
        if self.role == "payload" and self.fact_id is None:
            raise ValueError("payload segments require fact_id")
        if self.role != "payload" and self.fact_id is not None:
            raise ValueError("only payload segments may carry fact_id")
        if self.role == "payload" and self.payload_field is not None:
            if not self.payload_field:
                raise ValueError("payload fields must be nonempty")
        elif self.payload_field is not None:
            raise ValueError("only payload segments may carry payload_field")


@dataclass(frozen=True)
class ScheduleEntry:
    component: str
    record_id: str
    exposure: int
    curriculum_band: int


@dataclass(frozen=True)
class RenderedRecord:
    segments: tuple[TaggedSegment, ...]
    schedule: ScheduleEntry
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectorFeatures:
    log_exposure: float
    payload_entropy: float
    payload_tokens: float
    expected_queries: float
    path_centrality: float

    def vector(self) -> tuple[float, float, float, float, float]:
        return (
            self.log_exposure,
            self.payload_entropy,
            self.payload_tokens,
            self.expected_queries,
            self.path_centrality,
        )


@dataclass(frozen=True)
class GraphAction:
    source_slot: int
    relation_id: str
    direction: Direction
    read: bool
    halt: bool

    def __post_init__(self) -> None:
        if self.source_slot not in range(4):
            raise ValueError("source_slot must be in [0, 3]")
        if self.halt and self.read:
            raise ValueError("HALT cannot also read")
