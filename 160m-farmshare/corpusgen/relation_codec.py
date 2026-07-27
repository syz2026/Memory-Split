from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RelationCodec:
    relation_ids: tuple[str, ...]
    _indices: dict[str, int] = field(init=False, repr=False, compare=False)

    def __init__(self, relation_ids: Sequence[str]):
        values = tuple(relation_ids)
        if not values or len(values) != len(set(values)):
            raise ValueError("relation catalog must be non-empty and unique")
        if len(values) > 4096:
            raise ValueError("three-nibble codec supports at most 4096 relations")
        object.__setattr__(self, "relation_ids", values)
        object.__setattr__(
            self,
            "_indices",
            {value: index for index, value in enumerate(values)},
        )

    def encode(self, relation_id: str, tok) -> tuple[int, int, int]:
        index = self._indices[relation_id]
        digits = tok.RELATION_DIGITS
        return (
            digits[(index >> 8) & 15],
            digits[(index >> 4) & 15],
            digits[index & 15],
        )

    def decode(self, token_ids: Sequence[int], tok) -> str:
        if len(token_ids) != 3:
            raise ValueError("relation code requires exactly three tokens")
        reverse = {
            token: index for index, token in enumerate(tok.RELATION_DIGITS)
        }
        try:
            index = (
                (reverse[token_ids[0]] << 8)
                | (reverse[token_ids[1]] << 4)
                | reverse[token_ids[2]]
            )
            return self.relation_ids[index]
        except (KeyError, IndexError) as exc:
            raise ValueError("invalid relation code") from exc

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            list(self.relation_ids),
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()
