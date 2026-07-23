"""Syntax contracts for fixed-budget confirmatory action traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Any


_RELATION_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}")
_ACTION_FIELDS = frozenset({"source_slot", "relation_id", "direction", "op"})
ACTION_SLOTS = 12
MAX_READS = 10


class ActionOp(StrEnum):
    READ = "read"
    NOOP = "noop"
    HALT = "halt"


@dataclass(frozen=True)
class ActionSlot:
    source_slot: int | None
    relation_id: str | None
    direction: str | None
    op: ActionOp

    def __post_init__(self) -> None:
        try:
            op = ActionOp(self.op)
        except (TypeError, ValueError) as exc:
            raise ValueError("action op is invalid") from exc
        object.__setattr__(self, "op", op)
        if op is ActionOp.READ:
            if (
                isinstance(self.source_slot, bool)
                or not isinstance(self.source_slot, int)
                or not 0 <= self.source_slot <= 3
            ):
                raise ValueError("read source_slot must be an integer in [0, 3]")
            if (
                not isinstance(self.relation_id, str)
                or _RELATION_RE.fullmatch(self.relation_id) is None
            ):
                raise ValueError("read relation_id has invalid syntax")
            if self.direction not in {"out", "in"}:
                raise ValueError("read direction must be out or in")
        elif any(
            value is not None
            for value in (self.source_slot, self.relation_id, self.direction)
        ):
            raise ValueError("noop and halt payload fields must be null")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ActionSlot":
        if not isinstance(raw, Mapping):
            raise ValueError("action slot must be an object")
        if any(not isinstance(key, str) for key in raw):
            raise ValueError("action slot keys must be strings")
        missing = _ACTION_FIELDS - set(raw)
        unknown = set(raw) - _ACTION_FIELDS
        if missing:
            raise ValueError(f"action slot missing fields: {sorted(missing)}")
        if unknown:
            raise ValueError(f"action slot has unknown fields: {sorted(unknown)}")
        return cls(
            source_slot=raw["source_slot"],
            relation_id=raw["relation_id"],
            direction=raw["direction"],
            op=raw["op"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_slot": self.source_slot,
            "relation_id": self.relation_id,
            "direction": self.direction,
            "op": self.op.value,
        }


def validate_action_slots(
    raw_actions: Sequence[ActionSlot | Mapping[str, Any]],
) -> tuple[ActionSlot, ...]:
    """Validate only the fixed action grammar, never gold or store candidates."""

    if isinstance(raw_actions, (str, bytes)) or not isinstance(
        raw_actions,
        Sequence,
    ):
        raise ValueError("action trace must be an ordered sequence")
    if len(raw_actions) != ACTION_SLOTS:
        raise ValueError(f"action trace must contain exactly {ACTION_SLOTS} slots")
    actions = tuple(
        action if isinstance(action, ActionSlot) else ActionSlot.from_dict(action)
        for action in raw_actions
    )
    reads = sum(action.op is ActionOp.READ for action in actions)
    if reads > MAX_READS:
        raise ValueError(f"action trace may contain at most {MAX_READS} reads")
    halts = [
        index for index, action in enumerate(actions) if action.op is ActionOp.HALT
    ]
    if len(halts) > 1:
        raise ValueError("action trace may contain at most one HALT")
    if halts and any(
        action.op is not ActionOp.NOOP for action in actions[halts[0] + 1 :]
    ):
        raise ValueError("only NOOP actions are allowed after HALT")
    return actions
