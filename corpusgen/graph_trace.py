from __future__ import annotations

import json

from corpusgen.graph_records import GraphAction, GraphRow, TaggedSegment


def serialize_action(action: GraphAction, tok) -> list[int]:
    terminal = (
        tok.GRAPH_HALT
        if action.halt
        else tok.GRAPH_READ
        if action.read
        else tok.GRAPH_NOOP
    )
    direction = tok.DIR_OUT if action.direction == "out" else tok.DIR_IN
    if action.relation_id not in tok.RELATIONS or action.page:
        relation = tok.encode(action.relation_id)
        page = tok.encode(str(action.page))
        if not relation or not page:
            raise ValueError("relation and page identities must encode non-empty")
        return [
            tok.GRAPH_START,
            tok.SLOTS[action.source_slot],
            tok.RELATION_START,
            *relation,
            tok.RELATION_END,
            direction,
            tok.PAGE_START,
            *page,
            tok.PAGE_END,
            terminal,
            tok.GRAPH_END,
        ]
    return [
        tok.GRAPH_START,
        tok.SLOTS[action.source_slot],
        tok.RELATIONS[action.relation_id],
        direction,
        terminal,
        tok.GRAPH_END,
    ]


def parse_serialized_action(ids, tok) -> GraphAction:
    values = [int(value) for value in ids]
    if len(values) < 6:
        raise ValueError("graph action frame is too short")
    if values[0] != tok.GRAPH_START or values[-1] != tok.GRAPH_END:
        raise ValueError("invalid graph action frame")
    try:
        source_slot = tok.SLOTS.index(values[1])
    except ValueError as error:
        raise ValueError("invalid graph source slot") from error

    terminal = values[-2]
    if terminal not in (tok.GRAPH_READ, tok.GRAPH_NOOP, tok.GRAPH_HALT):
        raise ValueError("invalid graph terminal token")
    if len(values) == 6:
        try:
            relation_id = next(
                name
                for name, token_id in tok.RELATIONS.items()
                if token_id == values[2]
            )
        except StopIteration as error:
            raise ValueError("invalid legacy relation token") from error
        direction_token = values[3]
        page = 0
    else:
        if values[2] != tok.RELATION_START:
            raise ValueError("extended graph action lacks relation delimiter")
        try:
            relation_end = values.index(tok.RELATION_END, 3)
        except ValueError as error:
            raise ValueError("extended graph action lacks relation end") from error
        relation_ids = values[3:relation_end]
        if not relation_ids:
            raise ValueError("graph relation identity must not be empty")
        relation_id = tok.decode(relation_ids)
        direction_position = relation_end + 1
        if (
            direction_position + 2 >= len(values)
            or values[direction_position + 1] != tok.PAGE_START
        ):
            raise ValueError("extended graph action lacks page delimiter")
        try:
            page_end = values.index(tok.PAGE_END, direction_position + 2)
        except ValueError as error:
            raise ValueError("extended graph action lacks page end") from error
        if page_end != len(values) - 3:
            raise ValueError("unexpected tokens after graph page identity")
        page_text = tok.decode(values[direction_position + 2 : page_end])
        if not page_text.isascii() or not page_text.isdigit():
            raise ValueError("graph page identity must be decimal")
        page = int(page_text)
        direction_token = values[direction_position]

    if direction_token == tok.DIR_OUT:
        direction = "out"
    elif direction_token == tok.DIR_IN:
        direction = "in"
    else:
        raise ValueError("invalid graph direction token")
    return GraphAction(
        source_slot=source_slot,
        relation_id=relation_id,
        direction=direction,
        read=terminal == tok.GRAPH_READ,
        halt=terminal == tok.GRAPH_HALT,
        page=page,
    )


def serialize_return(row: GraphRow | None, fact_id: str | None):
    if row is None:
        return [
            TaggedSegment("<|graph_return|>", "action"),
            TaggedSegment("<|graph_miss|>", "action"),
            TaggedSegment("<|graph_end|>", "action"),
        ]
    if fact_id is None:
        raise ValueError("hit returns require fact_id")
    value = {
        "target_kind": row.target_kind,
        "target": row.target,
        "qualifiers": list(row.qualifiers),
    }
    if row.page or row.targets:
        value["page"] = row.page
        value["targets"] = list(row.values)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return [
        TaggedSegment("<|graph_return|>", "action"),
        TaggedSegment(payload, "payload", fact_id=fact_id),
        TaggedSegment("<|graph_end|>", "action"),
    ]
