"""Tests for corpusgen.bios (Task 2) — written before the implementation."""

import datetime
import math
import re

from corpusgen import bios
from corpusgen.records import ATTRIBUTES, DB_END

_WRAP = re.compile(r"<\|db_start\|>.*?<\|db_retrieve\|>", re.S)


def _unwrap(split_text: str) -> str:
    """Strip <|db_start|>query<|db_retrieve|> and <|db_end|> markers."""
    return _WRAP.sub("", split_text).replace(DB_END, "")


def _sample_records(n: int = 6, seed: int = 42):
    return bios.generate_records(n, seed=seed)


def _surfaces(rec):
    first, _middle, last = rec.name.split(" ")
    return {rec.name, f"{first} {last}", f"{last}, {first}"}


# ---------------------------------------------------------------- records


def test_generate_records_deterministic():
    a = bios.generate_records(200, seed=7)
    b = bios.generate_records(200, seed=7)
    assert a == b
    c = bios.generate_records(200, seed=8)
    assert c != a


def test_generate_records_shape_and_uniqueness_5k():
    recs = bios.generate_records(5000, seed=123)
    assert len(recs) == 5000
    assert [r.entity_id for r in recs] == list(range(5000))
    assert len({r.name for r in recs}) == 5000
    for r in recs[:25]:
        assert tuple(r.attrs.keys()) == ATTRIBUTES
        assert len(r.name.split(" ")) == 3


# ---------------------------------------------------------------- pools


def test_value_pool_sizes_match_spec():
    assert len(bios.VALUE_POOLS["birth_city"]) == 200
    assert len(bios.VALUE_POOLS["university"]) == 300
    assert len(bios.VALUE_POOLS["major"]) == 100
    assert len(bios.VALUE_POOLS["employer"]) == 263
    assert len(bios.VALUE_POOLS["current_city"]) == 200
    assert "birth_date" not in bios.VALUE_POOLS
    for key, pool in bios.VALUE_POOLS.items():
        assert len(set(pool)) == len(pool), f"duplicates in pool {key}"
    assert set(bios.RELATION_PHRASES) == set(ATTRIBUTES)
    assert bios.RELATION_PHRASES["birth_date"] == "birth date"
    assert bios.RELATION_PHRASES["current_city"] == "current city"


def test_name_pools_meet_minimums():
    assert len(set(bios.FIRST_NAMES)) == len(bios.FIRST_NAMES) >= 400
    assert len(set(bios.MIDDLE_NAMES)) == len(bios.MIDDLE_NAMES) >= 200
    assert len(set(bios.LAST_NAMES)) == len(bios.LAST_NAMES) >= 600


def test_birth_date_range_and_format():
    assert bios.BIRTH_DATE_MIN == datetime.date(1930, 1, 1)
    assert bios.BIRTH_DATE_MAX == datetime.date(2005, 12, 31)
    recs = bios.generate_records(300, seed=5)
    # month name, non-zero-padded day, 4-digit year
    pat = re.compile(
        r"^(January|February|March|April|May|June|July|August|September|"
        r"October|November|December) ([1-9]\d?), (\d{4})$"
    )
    for r in recs:
        m = pat.match(r.attrs["birth_date"])
        assert m, r.attrs["birth_date"]
        d = datetime.date(
            int(m.group(3)), bios.MONTH_NAMES.index(m.group(1)) + 1, int(m.group(2))
        )
        assert bios.BIRTH_DATE_MIN <= d <= bios.BIRTH_DATE_MAX


def test_pool_entropy_about_53_bits():
    n_days = (bios.BIRTH_DATE_MAX - bios.BIRTH_DATE_MIN).days + 1
    bits = math.log2(n_days) + sum(
        math.log2(len(bios.VALUE_POOLS[a])) for a in ATTRIBUTES if a != "birth_date"
    )
    assert 51.0 <= bits <= 55.0


# ---------------------------------------------------------------- templates


def test_templates_shape():
    assert set(bios.BIO_TEMPLATES) == set(ATTRIBUTES)
    for attr, templates in bios.BIO_TEMPLATES.items():
        assert len(templates) >= 20, attr
        named = [t for t in templates if "{name}" in t[0] + t[1]]
        assert len(named) >= 5, attr
        for prefix, suffix in templates:
            assert prefix.endswith(" "), (attr, prefix)
            assert suffix, (attr, prefix)
            assert not suffix[0].isalnum(), (attr, suffix)
    assert len(bios.ATTRIBUTE_ORDERINGS) >= 6
    for ordering in bios.ATTRIBUTE_ORDERINGS:
        assert sorted(ordering) == sorted(ATTRIBUTES)


# ---------------------------------------------------------------- rendering


def test_render_bio_doc_deterministic_and_exposure_varies():
    recs = _sample_records()
    d1 = bios.render_bio_doc(recs[0], 3)
    d2 = bios.render_bio_doc(recs[0], 3)
    assert d1 == d2
    for rec in recs:
        assert (
            bios.render_bio_doc(rec, 0).dense_text()
            != bios.render_bio_doc(rec, 1).dense_text()
        )


def test_render_bio_doc_kind_meta_and_dense_unmasked():
    rec = _sample_records()[1]
    doc = bios.render_bio_doc(rec, 5)
    assert doc.kind == "bio"
    assert doc.meta == {"entity_id": rec.entity_id, "exposure": 5}
    assert all(m is False for _, m in doc.dense_segments)


def test_dense_text_equals_split_text_unwrapped():
    for rec in _sample_records():
        for exp in range(4):
            doc = bios.render_bio_doc(rec, exp)
            assert doc.dense_text() == _unwrap(doc.split_text())


def test_masked_segments_are_exactly_attribute_values():
    for rec in _sample_records():
        doc = bios.render_bio_doc(rec, 2)
        masked = [t for t, m in doc.split_segments if m]
        assert len(masked) == 6
        allowed = {" " + v for v in rec.attrs.values()}
        assert set(masked) <= allowed


def test_all_attributes_and_canonical_queries_present():
    for rec in _sample_records():
        doc = bios.render_bio_doc(rec, 7)
        dense = doc.dense_text()
        split = doc.split_text()
        for attr in ATTRIBUTES:
            assert rec.attrs[attr] in dense
            # lookup query always uses canonical full name + raw attribute key
            assert f"<|db_start|>{rec.name}, {attr}<|db_retrieve|>" in split


def test_first_sentence_mentions_full_name():
    for rec in _sample_records():
        for exp in range(3):
            first_sentence = bios.render_bio_doc(rec, exp).dense_text().split(". ")[0]
            assert rec.name in first_sentence


# ---------------------------------------------------------------- recall probes


def test_recall_probes_fields_and_determinism():
    recs = bios.generate_records(40, seed=9)
    probes = bios.recall_probes(recs, 10, seed=3)
    assert len(probes) == 10 * len(ATTRIBUTES)
    assert probes == bios.recall_probes(recs, 10, seed=3)
    by_id = {r.entity_id: r for r in recs}
    for p in probes:
        rec = by_id[p.meta["entity_id"]]
        attr = p.meta["relation"]
        assert p.task == "recall"
        assert p.qid == f"recall-{rec.entity_id}-{attr}"
        assert p.prompt == f"{rec.name}'s {bios.RELATION_PHRASES[attr]} is"
        assert p.answer == rec.attrs[attr]


def test_recall_prompt_not_a_training_template_prefix():
    recs = bios.generate_records(5, seed=11)
    probes = bios.recall_probes(recs, 5, seed=1)
    by_id = {r.entity_id: r for r in recs}
    for p in probes:
        rec = by_id[p.meta["entity_id"]]
        for templates in bios.BIO_TEMPLATES.values():
            for prefix, _suffix in templates:
                for surface in _surfaces(rec):
                    rendered = prefix.format(name=surface).rstrip()
                    assert rendered != p.prompt.rstrip()
