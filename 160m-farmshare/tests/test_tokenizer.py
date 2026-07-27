from corpusgen.records import lookup_segments, plain
from train.tokenizer import get_tok


def test_special_tokens_atomic():
    tok = get_tok()
    assert tok.encode("<|db_start|>") == [50257]
    assert tok.encode("<|db_retrieve|>") == [50258]
    assert tok.encode("<|db_end|>") == [50259]
    assert tok.encode("<|eot|>") == [50260]


def test_round_trip_plain_text():
    tok = get_tok()
    text = "Kai Nakamura majored in Communications at Stanford University."
    assert tok.decode(tok.encode(text)) == text


def test_encode_segments_mask_alignment():
    tok = get_tok()
    segs = (
        [plain("Kai Nakamura majored in")]
        + lookup_segments("Kai Nakamura", "major", "Communications")
        + [plain(" at Stanford University.")]
    )
    ids, mask = tok.encode_segments(segs, add_eot=True)
    assert len(ids) == len(mask)
    # the masked positions decode to exactly the value segment
    masked_ids = [i for i, m in zip(ids, mask) if m == 0]
    assert tok.decode(masked_ids) == " Communications"
    # full decode reconstructs the concatenated split text + EOT
    full = "".join(t for t, _ in segs) + "<|eot|>"
    assert tok.decode(ids) == full
    # EOT present, loss ON
    assert ids[-1] == tok.EOT and mask[-1] == 1
    # special tokens all receive loss
    for special in (tok.DB_START, tok.DB_RETRIEVE, tok.DB_END):
        positions = [k for k, i in enumerate(ids) if i == special]
        assert positions and all(mask[k] == 1 for k in positions)


def test_encode_segments_no_eot():
    tok = get_tok()
    ids, mask = tok.encode_segments([plain("hello world")], add_eot=False)
    assert tok.EOT not in ids
    assert all(m == 1 for m in mask)


def test_dense_rendering_fully_unmasked():
    tok = get_tok()
    segs = [plain("Kai Nakamura majored in Communications at Stanford University.")]
    ids, mask = tok.encode_segments(segs)
    assert all(m == 1 for m in mask)


def test_graph_special_token_ids_are_reserved_and_atomic():
    tok = get_tok()
    assert tok.GRAPH_START == 50261
    assert tok.GRAPH_MISS == 50275
    assert tok.RELATION_DIGITS == tuple(range(50276, 50292))
    assert not hasattr(tok, "RELATIONS")
    assert tok.VOCAB_SIZE == 50304
    for text, token_id in tok.graph_special_tokens.items():
        assert tok.encode(text) == [token_id]
