from __future__ import annotations

import json

from corpusgen.graph_records import GraphAction, GraphRow, TaggedSegment
from corpusgen.relation_codec import RelationCodec


def serialize_action(
    action: GraphAction,
    tok,
    codec: RelationCodec,
) -> list[int]:
    terminal = (
        tok.GRAPH_HALT
        if action.halt
        else tok.GRAPH_READ
        if action.read
        else tok.GRAPH_NOOP
    )
    direction = tok.DIR_OUT if action.direction == "out" else tok.DIR_IN
    return [
        tok.GRAPH_START,
        tok.SLOTS[action.source_slot],
        *codec.encode(action.relation_id, tok),
        direction,
        terminal,
        tok.GRAPH_END,
    ]


def serialize_return(row: GraphRow | None, fact_id: str | None):
    if row is None:
        return [
            TaggedSegment("<|graph_return|>", "action"),
            TaggedSegment("<|graph_miss|>", "action"),
            TaggedSegment("<|graph_end|>", "action"),
        ]
    if fact_id is None:
        raise ValueError("hit returns require fact_id")
    segments = [
        TaggedSegment("<|graph_return|>", "action"),
        TaggedSegment('{"qualifiers":[', "plain"),
    ]
    for index, (key, value) in enumerate(row.qualifiers):
        prefix = ("," if index else "") + f"[{json.dumps(key)},"
        segments.extend(
            (
                TaggedSegment(prefix, "plain"),
                TaggedSegment(
                    json.dumps(value),
                    "payload",
                    fact_id=fact_id,
                    payload_field=f"qualifier:{key}",
                ),
                TaggedSegment("]", "plain"),
            )
        )
    segments.extend(
        (
            TaggedSegment('],"target":', "plain"),
            TaggedSegment(
                json.dumps(row.target),
                "payload",
                fact_id=fact_id,
                payload_field="target",
            ),
            TaggedSegment(
                ',"target_kind":' + json.dumps(row.target_kind) + "}",
                "plain",
            ),
            TaggedSegment("<|graph_end|>", "action"),
        )
    )
    return segments
