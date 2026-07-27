import hashlib
import json

import pytest

from train.tokenizer import get_tok


def _codec(relation_ids):
    from corpusgen.relation_codec import RelationCodec

    return RelationCodec(relation_ids)


def test_three_nibble_codec_round_trips_full_catalog_without_vocab_growth():
    relation_ids = tuple(f"P{i}" for i in range(1, 823)) + tuple(
        f"SYN_L{i}" for i in range(8)
    )
    codec = _codec(relation_ids)
    tok = get_tok()

    assert len({codec.encode(value, tok) for value in relation_ids}) == 830
    assert all(
        codec.decode(codec.encode(value, tok), tok) == value
        for value in relation_ids
    )
    assert all(
        50276 <= token < 50292
        for value in relation_ids
        for token in codec.encode(value, tok)
    )
    assert tok.VOCAB_SIZE == 50304


@pytest.mark.parametrize("relation_ids", [(), ("P1", "P1")])
def test_relation_catalog_must_be_non_empty_and_unique(relation_ids):
    with pytest.raises(
        ValueError, match="relation catalog must be non-empty and unique"
    ):
        _codec(relation_ids)


def test_three_nibble_codec_rejects_catalogs_larger_than_4096():
    with pytest.raises(
        ValueError, match="three-nibble codec supports at most 4096 relations"
    ):
        _codec(tuple(f"P{i}" for i in range(4097)))


def test_decode_rejects_wrong_width_unknown_digits_and_unused_indices():
    tok = get_tok()
    codec = _codec(("P1",))

    with pytest.raises(
        ValueError, match="relation code requires exactly three tokens"
    ):
        codec.decode(tok.RELATION_DIGITS[:2], tok)
    with pytest.raises(ValueError, match="invalid relation code"):
        codec.decode((tok.RELATION_DIGITS[0], tok.RELATION_DIGITS[0], -1), tok)
    with pytest.raises(ValueError, match="invalid relation code"):
        codec.decode(
            (
                tok.RELATION_DIGITS[0],
                tok.RELATION_DIGITS[0],
                tok.RELATION_DIGITS[1],
            ),
            tok,
        )


def test_catalog_hash_uses_canonical_ordered_utf8_json():
    relation_ids = ("P31", "SYN_é")
    expected = hashlib.sha256(
        json.dumps(
            list(relation_ids),
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()

    assert _codec(relation_ids).sha256() == expected
