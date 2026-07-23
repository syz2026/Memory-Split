from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["out", "in"]
TargetKind = Literal["entity", "literal"]
CanonicalEntityID = int | str
SegmentRole = Literal[
    "plain",
    "payload",
    "random_control",
    "rule",
    "action",
    "query",
    "candidate_state",
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
    source_id: CanonicalEntityID
    relation_id: str
    direction: Direction
    page: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.source_id, bool) or not isinstance(
            self.source_id, (int, str)
        ):
            raise ValueError("source_id must be an integer or canonical identity")
        if isinstance(self.source_id, int) and self.source_id < 0:
            raise ValueError("source_id must be non-negative")
        if isinstance(self.source_id, str) and not self.source_id:
            raise ValueError("source_id must be non-empty")
        if not self.relation_id:
            raise ValueError("relation_id must be non-empty")
        if self.direction not in ("out", "in"):
            raise ValueError(f"invalid direction: {self.direction}")
        if isinstance(self.page, bool) or not isinstance(self.page, int) or self.page < 0:
            raise ValueError("page must be a non-negative integer")

    @property
    def entity_id(self) -> CanonicalEntityID:
        return self.source_id

    def sort_key(self) -> tuple[str, str, str, int]:
        source = (
            f"I{self.source_id:020d}"
            if isinstance(self.source_id, int)
            else f"S{self.source_id}"
        )
        return source, self.relation_id, self.direction, self.page


@dataclass(frozen=True, order=True)
class GraphRow:
    source_id: CanonicalEntityID
    relation_id: str
    direction: Direction
    target_kind: TargetKind
    target: str
    qualifiers: tuple[tuple[str, str], ...] = ()
    provenance_id: str = ""
    page: int = 0
    targets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        GraphAddress(self.source_id, self.relation_id, self.direction, self.page)
        if self.target_kind not in ("entity", "literal"):
            raise ValueError(f"invalid target_kind: {self.target_kind}")
        if not self.target:
            raise ValueError("target must be non-empty")
        if self.targets:
            if self.targets[0] != self.target:
                raise ValueError("target must be the first set-valued member")
            if any(not value for value in self.targets):
                raise ValueError("set-valued targets must be non-empty")
            if len(self.targets) != len(set(self.targets)):
                raise ValueError("set-valued targets must be distinct")

    @property
    def address(self) -> GraphAddress:
        return GraphAddress(
            self.source_id,
            self.relation_id,
            self.direction,
            self.page,
        )

    @property
    def values(self) -> tuple[str, ...]:
        return self.targets or (self.target,)

    def as_json(self) -> dict:
        value = {
            "source_id": self.source_id,
            "relation_id": self.relation_id,
            "direction": self.direction,
            "target_kind": self.target_kind,
            "target": self.target,
            "qualifiers": [list(q) for q in self.qualifiers],
            "provenance_id": self.provenance_id,
        }
        if self.page or self.targets:
            value["page"] = self.page
            value["targets"] = list(self.values)
        return value

    @classmethod
    def from_json(cls, value: dict) -> "GraphRow":
        source_id = value["source_id"]
        if not isinstance(source_id, (int, str)) or isinstance(source_id, bool):
            raise ValueError("source_id must be an integer or canonical identity")
        return cls(
            source_id=source_id,
            relation_id=str(value["relation_id"]),
            direction=value["direction"],
            target_kind=value["target_kind"],
            target=str(value["target"]),
            qualifiers=tuple((str(k), str(v)) for k, v in value["qualifiers"]),
            provenance_id=str(value["provenance_id"]),
            page=int(value.get("page", 0)),
            targets=tuple(str(item) for item in value.get("targets", ())),
        )


@dataclass(frozen=True)
class TaggedSegment:
    text: str
    role: SegmentRole
    fact_id: str | None = None

    def __post_init__(self) -> None:
        if self.role == "payload" and self.fact_id is None:
            raise ValueError("payload segments require fact_id")
        if self.role != "payload" and self.fact_id is not None:
            raise ValueError("only payload segments may carry fact_id")


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
    page: int = 0

    def __post_init__(self) -> None:
        if self.source_slot not in range(4):
            raise ValueError("source_slot must be in [0, 3]")
        if not self.relation_id:
            raise ValueError("relation_id must be non-empty")
        if self.direction not in ("out", "in"):
            raise ValueError(f"invalid direction: {self.direction}")
        if self.halt and self.read:
            raise ValueError("HALT cannot also read")
        if isinstance(self.page, bool) or not isinstance(self.page, int) or self.page < 0:
            raise ValueError("page must be a non-negative integer")
