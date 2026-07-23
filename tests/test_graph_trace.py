import pytest

from corpusgen.graph_records import GraphAction, GraphRow
from corpusgen.graph_trace import parse_serialized_action, serialize_action, serialize_return
from train.tokenizer import get_tok


def test_graph_action_is_fixed_width_and_atomic():
    tok = get_tok()
    action = GraphAction(2, "r3", "out", read=True, halt=False)
    ids = serialize_action(action, tok)
    assert ids == [
        tok.GRAPH_START,
        tok.SLOTS[2],
        tok.RELATIONS["r3"],
        tok.DIR_OUT,
        tok.GRAPH_READ,
        tok.GRAPH_END,
    ]


def test_halt_action_is_exact_fixed_width_frame():
    tok = get_tok()
    action = GraphAction(0, "r0", "out", read=False, halt=True)
    ids = serialize_action(action, tok)
    assert ids == [
        tok.GRAPH_START,
        tok.SLOTS[0],
        tok.RELATIONS["r0"],
        tok.DIR_OUT,
        tok.GRAPH_HALT,
        tok.GRAPH_END,
    ]


def test_noop_action_is_exact_fixed_width_frame():
    tok = get_tok()
    action = GraphAction(3, "r15", "in", read=False, halt=False)
    ids = serialize_action(action, tok)
    assert ids == [
        tok.GRAPH_START,
        tok.SLOTS[3],
        tok.RELATIONS["r15"],
        tok.DIR_IN,
        tok.GRAPH_NOOP,
        tok.GRAPH_END,
    ]


def test_return_serialization_marks_payload_fact():
    tok = get_tok()
    row = GraphRow(
        1, "r2", "out", "entity", "9", (("compose", "3"),), "world-1"
    )
    segments = serialize_return(row, "fact-1")
    ids, roles, fact_ids = tok.encode_tagged_segments(segments)
    assert ids[0] == tok.GRAPH_RETURN and ids[-1] == tok.GRAPH_END
    assert "payload" in roles
    assert "fact-1" in fact_ids


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


def test_arbitrary_pid_and_page_use_delimited_identity_text():
    tok = get_tok()
    action = GraphAction(
        1,
        "P999999",
        "in",
        read=True,
        halt=False,
        page=37,
    )

    ids = serialize_action(action, tok)

    assert ids[0] == tok.GRAPH_START and ids[-1] == tok.GRAPH_END
    assert tok.RELATION_START in ids and tok.RELATION_END in ids
    assert tok.PAGE_START in ids and tok.PAGE_END in ids
    assert all(0 <= token_id < 50_304 for token_id in ids)
    assert parse_serialized_action(ids, tok) == action


def test_set_valued_return_serializes_all_page_members_as_payload():
    tok = get_tok()
    row = GraphRow(
        "Q1",
        "P31",
        "out",
        "entity",
        "Q2",
        (),
        "wikidata:set",
        page=2,
        targets=("Q2", "Q3", "Q5"),
    )

    segments = serialize_return(row, "wikidata:Q1:P31:page2")
    ids, roles, fact_ids = tok.encode_tagged_segments(segments)
    decoded = tok.decode(ids)

    assert '"page":2' in decoded
    assert '"targets":["Q2","Q3","Q5"]' in decoded
    assert "payload" in roles
    assert "wikidata:Q1:P31:page2" in fact_ids
