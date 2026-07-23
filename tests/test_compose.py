"""Tests for the OOD two-fact composition generator, builder, and scorer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from corpusgen import bios, compose
from corpusgen.records import DB_END, DB_RETRIEVE, DB_START
from evals.compose_eval import extract_all_keys, score_two_hop, aggregate_two_hop
from corpusgen.records import QAItem
from organizer.store import normalize
from train.tokenizer import get_tok


def _recs(n=40, seed=0):
    return bios.generate_records(n, seed)


# ------------------------------------------------------------- bridge edges


def test_assign_bridges_deterministic_functional_no_self():
    recs = _recs(30)
    e1 = compose.assign_bridges(recs, seed=0)
    e2 = compose.assign_bridges(recs, seed=0)
    assert e1 == e2  # deterministic
    ids = {r.entity_id for r in recs}
    for eid, rels in e1.items():
        assert set(rels) == set(compose.BRIDGE_RELATIONS)
        for rel, tgt in rels.items():
            assert tgt in ids            # within population
            assert tgt != eid            # never self


def test_bridges_stay_within_population():
    recs = _recs(60)
    comp, held = recs[:40], recs[40:]
    ec = compose.assign_bridges(comp, 0)
    eh = compose.assign_bridges(held, 1)
    comp_ids = {r.entity_id for r in comp}
    held_ids = {r.entity_id for r in held}
    assert all(t in comp_ids for rels in ec.values() for t in rels.values())
    assert all(t in held_ids for rels in eh.values() for t in rels.values())


# ------------------------------------------------------------- doc renderings


def test_compose_doc_split_masks_only_values():
    recs = _recs(40)
    x, y = recs[0], recs[7]
    attr = "employer"
    doc = compose.render_compose_doc(x, "mentor", y, attr, exposure=0)
    tok = get_tok()

    # dense: no masked tokens
    _, dmask = tok.encode_segments(doc.dense_segments)
    assert 0 not in dmask
    assert y.name in doc.dense_text()
    assert y.attrs[attr] in doc.dense_text()

    # split: masked tokens are exactly the two retrieved values
    sids, smask = tok.encode_segments(doc.split_segments)
    masked_text = tok.decode([i for i, m in zip(sids, smask) if m == 0])
    assert masked_text == f" {y.name} {y.attrs[attr]}"  # the two masked spans
    assert 0 in smask and 1 in smask


def test_compose_doc_lookup_keys_recoverable():
    recs = _recs(40)
    x, y = recs[2], recs[9]
    doc = compose.render_compose_doc(x, "advisor", y, "university", 0)
    keys = extract_all_keys(doc.split_text())
    assert len(keys) == 2
    assert normalize(keys[0]) == normalize(f"{x.name}, advisor")
    assert normalize(keys[1]) == normalize(f"{y.name}, university")
    # bridge name appears only inside the hop-2 query (must be copied), not in prose
    assert DB_START in doc.split_text() and DB_END in doc.split_text()


def test_eval_prompt_is_prefix_of_training_doc():
    recs = _recs(40)
    x, y = recs[1], recs[3]
    item = compose.compose_item(x, "mentor", y, "major", "comp")
    doc = compose.render_compose_doc(x, "mentor", y, "major", 0)
    assert doc.dense_text().startswith(item.prompt)
    assert doc.split_text().startswith(item.prompt)
    assert item.answer == y.attrs["major"]


def test_compose_doc_deterministic():
    recs = _recs(40)
    x, y = recs[0], recs[1]
    a = compose.render_compose_doc(x, "mentor", y, "employer", 5)
    b = compose.render_compose_doc(x, "mentor", y, "employer", 5)
    assert a.dense_text() == b.dense_text()
    assert a.split_segments == b.split_segments


# ------------------------------------------------------------- scorer


def test_score_two_hop_reads_answer_and_hops():
    recs = _recs(40)
    x, y = recs[4], recs[8]
    item = compose.compose_item(x, "mentor", y, "employer", "held")
    v = y.attrs["employer"]
    gen = (f" {x.name}'s mentor is {DB_START}{x.name}, mentor{DB_RETRIEVE} {y.name}{DB_END}."
           f" Their employer is {DB_START}{y.name}, employer{DB_RETRIEVE} {v}{DB_END}."
           f" So the answer is {v}.\nAnswer: {v}")
    s = score_two_hop(gen, item)
    assert s["answer_correct"] and s["hop1_key_ok"] and s["hop2_key_ok"]
    assert s["both_hops_ok"] and s["n_lookups_emitted"] == 2


def test_score_two_hop_wrong_bridge_fails_hop2():
    recs = _recs(40)
    x, y = recs[4], recs[8]
    item = compose.compose_item(x, "mentor", y, "employer", "held")
    gen = (f"{DB_START}{x.name}, mentor{DB_RETRIEVE} SOMEONE ELSE{DB_END}."
           f" {DB_START}SOMEONE ELSE, employer{DB_RETRIEVE} Wrong{DB_END}.\nAnswer: Wrong")
    s = score_two_hop(gen, item)
    assert s["hop1_key_ok"] and not s["hop2_key_ok"] and not s["answer_correct"]


def test_aggregate_conditioning():
    rows = [
        {"answer_correct": True, "hop1_key_ok": True, "hop2_key_ok": True,
         "both_hops_ok": True, "n_lookups_emitted": 2,
         "meta": {"population": "held", "x_id": 1, "y_id": 2,
                  "relation": "mentor", "attr": "employer"}},
        {"answer_correct": False, "hop1_key_ok": False, "hop2_key_ok": False,
         "both_hops_ok": False, "n_lookups_emitted": 0,
         "meta": {"population": "held", "x_id": 3, "y_id": 4,
                  "relation": "mentor", "attr": "major"}},
    ]
    sh = {"hop1-held-1-mentor": True, "hop2-held-2-employer": True,
          "hop1-held-3-mentor": False, "hop2-held-4-major": True}
    agg = aggregate_two_hop(rows, sh)
    assert agg["n"] == 2
    assert agg["answer_accuracy"] == 0.5
    # only the first item has both hops accessible -> conditioned on 1 item
    assert agg["n_both_hops_accessible"] == 1
    assert agg["answer_accuracy_conditioned"] == 1.0


# ------------------------------------------------------------- end-to-end build


def test_build_compose_smoke(tmp_path: Path):
    import argparse
    from scripts.build_compose import build

    args = argparse.Namespace(
        out=str(tmp_path / "c"), n_entities=60, held_frac=0.25,
        total_tokens=200_000, seed=0, held_per_person=2,
        bio_share=0.2, bridge_share=0.2, compose_share=0.6,
        arms=["dense", "split"], n_eval=50,
    )
    report = build(args)
    assert all(report["checks"].values())

    out = tmp_path / "c"
    # bins exist and ids/mask are aligned per arm
    for arm in ("dense", "split"):
        ids = np.fromfile(out / arm / "train.bin", dtype=np.uint16)
        mask = np.fromfile(out / arm / "train.mask.bin", dtype=np.uint8)
        assert len(ids) == len(mask) and len(ids) > 0
        assert set(np.unique(mask)).issubset({0, 1})
    # dense mask is all ones; split has masked (0) tokens
    dmask = np.fromfile(out / "dense" / "train.mask.bin", dtype=np.uint8)
    smask = np.fromfile(out / "split" / "train.mask.bin", dtype=np.uint8)
    assert dmask.min() == 1
    assert 0 in smask

    # eval files parse and OOD population is truly held
    ood = [json.loads(l) for l in (out / "eval" / "comp_ood.jsonl").read_text().splitlines()]
    assert all(it["meta"]["population"] == "held" for it in ood)
    assert len(ood) > 0
