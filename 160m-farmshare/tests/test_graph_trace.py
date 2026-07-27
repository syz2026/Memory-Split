import pytest

from corpusgen.graph_records import GraphAction, GraphRow
from corpusgen.graph_trace import serialize_action, serialize_return
from train.tokenizer import get_tok


def _codec():
    from corpusgen.relation_codec import RelationCodec

    return RelationCodec(tuple(f"r{i}" for i in range(16)))


def test_graph_action_is_fixed_width_and_atomic():
    tok = get_tok()
    codec = _codec()
    action = GraphAction(2, "r3", "out", read=True, halt=False)
    ids = serialize_action(action, tok, codec)
    assert ids == [
        tok.GRAPH_START,
        tok.SLOTS[2],
        *codec.encode("r3", tok),
        tok.DIR_OUT,
        tok.GRAPH_READ,
        tok.GRAPH_END,
    ]


def test_halt_action_is_exact_fixed_width_frame():
    tok = get_tok()
    codec = _codec()
    action = GraphAction(0, "r0", "out", read=False, halt=True)
    ids = serialize_action(action, tok, codec)
    assert ids == [
        tok.GRAPH_START,
        tok.SLOTS[0],
        *codec.encode("r0", tok),
        tok.DIR_OUT,
        tok.GRAPH_HALT,
        tok.GRAPH_END,
    ]


def test_noop_action_is_exact_fixed_width_frame():
    tok = get_tok()
    codec = _codec()
    action = GraphAction(3, "r15", "in", read=False, halt=False)
    ids = serialize_action(action, tok, codec)
    assert ids == [
        tok.GRAPH_START,
        tok.SLOTS[3],
        *codec.encode("r15", tok),
        tok.DIR_IN,
        tok.GRAPH_NOOP,
        tok.GRAPH_END,
    ]


def test_return_serialization_masks_only_entity_specific_values():
    tok = get_tok()
    row = GraphRow(
        1, "r2", "out", "entity", "9", (("compose", "3"),), "world-1"
    )
    segments = serialize_return(row, "fact-1")
    ids, roles, fact_ids = tok.encode_tagged_segments(segments)

    assert ids[0] == tok.GRAPH_RETURN and ids[-1] == tok.GRAPH_END
    assert "".join(segment.text for segment in segments[1:-1]) == (
        '{"qualifiers":[["compose","3"]],"target":"9",'
        '"target_kind":"entity"}'
    )
    assert [
        segment.text for segment in segments if segment.role == "payload"
    ] == ['"3"', '"9"']
    assert {
        segment.fact_id for segment in segments if segment.role == "payload"
    } == {"fact-1"}
    assert [
        segment.payload_field
        for segment in segments
        if segment.role == "payload"
    ] == ["qualifier:compose", "target"]
    supervised = "".join(
        segment.text for segment in segments if segment.role == "plain"
    )
    assert "qualifiers" in supervised
    assert "compose" in supervised
    assert "target_kind" in supervised
    assert "entity" in supervised
    assert all(
        fact_id == "fact-1"
        for role, fact_id in zip(roles, fact_ids)
        if role == "payload"
    )


def test_relation_alias_is_a_protected_nonpayload_role():
    from corpusgen.graph_records import TaggedSegment

    segment = TaggedSegment("date of birth", "relation_alias")

    assert segment.fact_id is None


def test_return_hit_requires_fact_id():
    row = GraphRow(1, "r2", "out", "entity", "9")

    with pytest.raises(ValueError, match="^hit returns require fact_id$"):
        serialize_return(row, None)


def test_return_miss_has_no_payload():
    tok = get_tok()
    segments = serialize_return(None, None)
    ids, roles, fact_ids = tok.encode_tagged_segments(segments)
    assert tok.GRAPH_MISS in ids
    assert "payload" not in roles
    assert all(f is None for f in fact_ids)


def test_tagged_segment_payload_requires_fact_id():
    from corpusgen.graph_records import TaggedSegment

    with pytest.raises(ValueError, match="payload segments require fact_id"):
        TaggedSegment("data", "payload", fact_id=None)


def test_tagged_segment_non_payload_rejects_fact_id():
    from corpusgen.graph_records import TaggedSegment

    with pytest.raises(ValueError, match="only payload segments may carry fact_id"):
        TaggedSegment("data", "action", fact_id="fact-1")
