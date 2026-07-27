"""Shared data model for the memory-split corpus.

Conventions (authoritative; see docs/superpowers/plans/2026-07-17-memory-split.md):
- Segment = (text, masked). masked=True means the segment's tokens receive
  NO loss (fact values in the split arm). masked=False means loss ON.
- A Doc carries both renderings of the same underlying content. For
  knowledge-free docs the two renderings are identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ATTRIBUTES = (
    "birth_date",
    "birth_city",
    "university",
    "major",
    "employer",
    "current_city",
)

Segment = tuple[str, bool]  # (text, masked)


@dataclass(frozen=True)
class BioRecord:
    entity_id: int
    name: str
    attrs: dict[str, str]  # keys == ATTRIBUTES

    def __post_init__(self) -> None:
        missing = [a for a in ATTRIBUTES if a not in self.attrs]
        if missing:
            raise ValueError(f"BioRecord {self.entity_id} missing attributes: {missing}")


@dataclass
class Doc:
    kind: str  # "bed" | "bio" | "igsm" | "deduction" | "factqa"
    dense_segments: list[Segment]
    split_segments: list[Segment]
    meta: dict = field(default_factory=dict)

    def dense_text(self) -> str:
        return "".join(t for t, _ in self.dense_segments)

    def split_text(self) -> str:
        return "".join(t for t, _ in self.split_segments)


@dataclass
class QAItem:
    qid: str
    task: str  # "igsm" | "deduction" | "factqa" | "recall"
    prompt: str  # generative tasks end with "Reasoning:"; recall probes end mid-sentence
    answer: str
    meta: dict = field(default_factory=dict)


DB_START = "<|db_start|>"
DB_RETRIEVE = "<|db_retrieve|>"
DB_END = "<|db_end|>"
EOT = "<|eot|>"


def lookup_segments(name: str, relation: str, value: str) -> list[Segment]:
    """The exact split-arm wrapping of one fact value.

    Loss stays ON for the special tokens and the query (the model must learn
    to ask); loss is OFF only for the value itself, which arrives from the
    organizer at inference time.
    """
    return [
        (DB_START, False),
        (f"{name}, {relation}", False),
        (DB_RETRIEVE, False),
        (f" {value}", True),
        (DB_END, False),
    ]


def plain(text: str) -> Segment:
    return (text, False)
