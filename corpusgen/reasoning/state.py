"""Value-free answer-state pointers for supervised reasoning traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal


AnswerPhase = Literal["candidate", "final"]


@dataclass(frozen=True, order=True)
class AnswerPointer:
    """Reference a returned member without serializing its factual value."""

    slot: int
    read_index: int
    member_index: int = 0

    def __post_init__(self) -> None:
        for field, value in (
            ("slot", self.slot),
            ("read_index", self.read_index),
            ("member_index", self.member_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field} must be an integer")
        if self.slot not in range(4):
            raise ValueError("slot must be in [0, 3]")
        if self.read_index not in range(12):
            raise ValueError("read_index must be in [0, 11]")
        if self.member_index < 0:
            raise ValueError("member_index must be non-negative")

    def as_dict(self, phase: AnswerPhase) -> dict:
        if phase not in ("candidate", "final"):
            raise ValueError("phase must be candidate or final")
        return {
            "member_index": self.member_index,
            "phase": phase,
            "read_index": self.read_index,
            "slot": f"<|slot_{self.slot}|>",
        }


def serialize_answer_state(
    pointer: AnswerPointer,
    *,
    phase: AnswerPhase = "candidate",
) -> str:
    """Serialize only structural coordinates, never a resolved surface."""

    if not isinstance(pointer, AnswerPointer):
        raise TypeError("pointer must be an AnswerPointer")
    payload = json.dumps(
        pointer.as_dict(phase),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return f"<|answer_state|>{payload}"
