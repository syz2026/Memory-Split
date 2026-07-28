#!/usr/bin/env python
"""Build the two-hop composition corpus (aligned dense + split arms) for finetuning.

Reconstruction of the missing `scripts/build_compose.py`. It reuses the repo's OWN
bio generator/templates (`corpusgen.bios`) so the single-hop bio facts are rendered
exactly as the base model saw them, then ADDS the two things the composition
experiment needs on top of the bio base:

  * a bridge graph  — each person gets a `mentor` and an `advisor` pointing to
    another person IN THE SAME GROUP (edges closed within group), and
  * two-hop compose documents  — "What is the <attr> of <P>'s <rel>? Reasoning:
    <P>'s <rel> is <B>. Their <attr> is <V>. So the answer is <V>. Answer: <V>".

People are split into P_comp (composed) and P_held (OOD: only atomic single-hop
facts, never composed). Corpus token mix ≈ bio 20% / bridge 13% / compose 67%
(deficit-scheduled). Both arms are rendered from ONE shared plan so they are
aligned; the SPLIT arm wraps every fact VALUE in a loss-masked
`<|db_start|>key<|db_retrieve|> value<|db_end|>` lookup (via `lookup_segments`),
the DENSE arm writes values inline.

Outputs under --out:
  dense/train.bin                 uint16 token stream (all loss ON)
  split/train.bin                 uint16 token stream
  split/train.mask.bin            uint8 (1 = loss ON, 0 = masked fact value)
  eval.json                       held-out two-hop (indist/ood) + single-hop probes + graph
  report.json                     metadata + token/lane accounting

Determinism: fixed seed -> identical graph, plan, and token streams.

Usage:
  python build_compose.py --out data/compose_v1 --n-entities 10000 --held-frac 0.2 \
      --total-tokens 600000000 --n-eval 1000 --seed 1234
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

# Find the repo root (has corpusgen/ + train/) whether this script is run from
# the repo root (Colab copies it there) or from compose-finetune/.
_here = Path(__file__).resolve().parent
for _cand in (_here, _here.parent, Path.cwd()):
    if (_cand / "corpusgen").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from corpusgen import bios
from corpusgen.bios import RELATION_PHRASES, render_bio_doc
from corpusgen.records import Doc, Segment, lookup_segments
from train.tokenizer import get_tok

# Attributes composed in two-hop questions (4 -> 2 rels x 4 attrs = 8 triples/person,
# matching the study's 64k triples / 8k P_comp). birth_date/current_city excluded
# (open pool / duplicate-of-birth_city surface).
COMPOSE_ATTRS = ("employer", "university", "major", "birth_city")
RELS = ("mentor", "advisor")

BRIDGE_TEMPLATES = [
    ("{name}'s {rel} is ", "."),
    ("The {rel} of {name} is ", "."),
    ("According to the registry, {name}'s {rel} is ", "."),
    ("Records list {name}'s {rel} as ", "."),
    ("{name} has long named ", " as their {rel}."),
    ("On file, the {rel} of {name} is ", "."),
    ("It is well known that {name}'s {rel} is ", "."),
    ("The department roster gives {name}'s {rel} as ", "."),
]


# ----------------------------------------------------------------- graph
def build_graph(records, held_frac, seed):
    """Assign each person a mentor + advisor within their group (comp/held)."""
    n = len(records)
    n_held = int(round(n * held_frac))
    held_ids = set(r.entity_id for r in records[n - n_held:])
    comp = [r for r in records if r.entity_id not in held_ids]
    held = [r for r in records if r.entity_id in held_ids]
    by_id = {r.entity_id: r for r in records}
    rng = random.Random(f"graph:{seed}")
    bridges: dict[int, dict[str, int]] = {}
    for group in (comp, held):
        ids = [r.entity_id for r in group]
        for r in group:
            b: dict[str, int] = {}
            for rel in RELS:
                other = r.entity_id
                while other == r.entity_id:
                    other = ids[rng.randrange(len(ids))]
                b[rel] = other
            bridges[r.entity_id] = b
    return by_id, bridges, comp, held, held_ids


# ----------------------------------------------------------------- renderers
def render_bridge_doc(rec, rel, bridge_rec, exposure) -> Doc:
    rng = random.Random(f"bridge:{rec.entity_id}:{rel}:{exposure}")
    prefix, suffix = BRIDGE_TEMPLATES[rng.randrange(len(BRIDGE_TEMPLATES))]
    prefix = prefix.format(name=rec.name, rel=rel)
    suffix = suffix.format(name=rec.name, rel=rel)
    bname = bridge_rec.name
    dense = f"{prefix}{bname}{suffix}"
    split: list[Segment] = [(prefix, False)]
    split += lookup_segments(rec.name, rel, bname)   # value (bridge person) masked
    split += [(suffix, False)]
    return Doc(kind="bridge", dense_segments=[(dense, False)], split_segments=split,
               meta={"entity_id": rec.entity_id, "rel": rel})


def render_compose_doc(rec, rel, attr, bridge_rec, exposure) -> Doc:
    ap = RELATION_PHRASES[attr]
    B = bridge_rec.name
    V = bridge_rec.attrs[attr]
    q = f"What is the {ap} of {rec.name}'s {rel}? Reasoning: "
    mid1 = f"{rec.name}'s {rel} is "
    mid2 = f". Their {ap} is "
    tail = f". So the answer is {V}. Answer: {V}"
    dense = f"{q}{mid1}{B}{mid2}{V}{tail}"
    split: list[Segment] = [(q + mid1, False)]
    split += lookup_segments(rec.name, rel, B)       # hop-1 value masked
    split += [(mid2, False)]
    split += lookup_segments(B, attr, V)             # hop-2 value masked
    split += [(tail, False)]                          # final answer is loss-ON (copy skill)
    return Doc(kind="compose", dense_segments=[(dense, False)], split_segments=split,
               meta={"entity_id": rec.entity_id, "rel": rel, "attr": attr})


# ----------------------------------------------------------------- writer
class ArmWriter:
    def __init__(self, path: Path, with_mask: bool):
        path.mkdir(parents=True, exist_ok=True)
        self.bin = open(path / "train.bin", "wb")
        self.maskf = open(path / "train.mask.bin", "wb") if with_mask else None
        self.tok_buf: list[int] = []
        self.mask_buf: list[int] = []
        self.n = 0

    def add(self, ids, mask):
        self.tok_buf.extend(ids)
        self.mask_buf.extend(mask)
        self.n += len(ids)
        if len(self.tok_buf) >= 2_000_000:
            self.flush()

    def flush(self):
        if self.tok_buf:
            np.asarray(self.tok_buf, dtype=np.uint16).tofile(self.bin)
            if self.maskf is not None:
                np.asarray(self.mask_buf, dtype=np.uint8).tofile(self.maskf)
            self.tok_buf.clear(); self.mask_buf.clear()

    def close(self):
        self.flush(); self.bin.close()
        if self.maskf is not None:
            self.maskf.close()


def emit_eval_and_store(out, records, by_id, bridges, comp, held, indist_holdout,
                        n_eval, seed):
    """Write eval.json + store.json (deterministic in seed). Shared by the full
    build and the --eval-only regeneration path."""
    def bridge_name(eid, rel):
        return by_id[bridges[eid][rel]].name

    def bridge_val(eid, rel, attr):
        return by_id[bridges[eid][rel]].attrs[attr]

    eval_rng = random.Random(f"eval:{seed}")
    indist = [{"person": by_id[eid].name, "entity_id": eid, "rel": rel, "attr": attr,
               "bridge": bridge_name(eid, rel), "answer": bridge_val(eid, rel, attr)}
              for eid, (rel, attr) in indist_holdout.items()]
    eval_rng.shuffle(indist); indist = indist[:n_eval]

    ood_people = eval_rng.sample(held, min(n_eval, len(held)))
    ood = [{"person": r.name, "entity_id": r.entity_id, "rel": rel, "attr": attr,
            "bridge": bridge_name(r.entity_id, rel), "answer": bridge_val(r.entity_id, rel, attr)}
           for r in ood_people for rel in RELS for attr in COMPOSE_ATTRS]
    eval_rng.shuffle(ood); ood = ood[:n_eval]

    def singlehop(group, n):
        sample = eval_rng.sample(group, min(n, len(group)))
        hop1 = [{"person": r.name, "entity_id": r.entity_id, "rel": rel,
                 "bridge": bridge_name(r.entity_id, rel)} for r in sample for rel in RELS]
        hop2 = [{"person": by_id[bridges[r.entity_id][rel]].name, "attr": attr,
                 "value": bridge_val(r.entity_id, rel, attr)}
                for r in sample for rel in RELS for attr in COMPOSE_ATTRS]
        return hop1, hop2

    c_h1, c_h2 = singlehop(comp, n_eval // 2)
    h_h1, h_h2 = singlehop(held, n_eval // 2)
    (out / "eval.json").write_text(json.dumps({
        "seed": seed, "compose_attrs": COMPOSE_ATTRS, "rels": RELS,
        "twohop_indist": indist, "twohop_ood": ood,
        "singlehop": {"comp_hop1": c_h1, "comp_hop2": c_h2,
                      "held_hop1": h_h1, "held_hop2": h_h2},
    }, indent=2))

    from corpusgen.records import ATTRIBUTES
    store = {}
    for r in records:
        for attr in ATTRIBUTES:
            if attr in r.attrs:
                store[f"{r.name}|{attr}"] = r.attrs[attr]
        for rel in RELS:
            store[f"{r.name}|{rel}"] = bridge_name(r.entity_id, rel)
    (out / "store.json").write_text(json.dumps(store))
    print(f"[build] wrote eval.json ({len(indist)} indist, {len(ood)} ood) + "
          f"store.json ({len(store)} keys) -> {out}", flush=True)


def _records_graph_holdout(n_entities, held_frac, seed):
    records = bios.generate_records(n_entities, seed)
    by_id, bridges, comp, held, held_ids = build_graph(records, held_frac, seed)
    rng = random.Random(f"heldout:{seed}")
    all_triples = [(rel, attr) for rel in RELS for attr in COMPOSE_ATTRS]
    indist_holdout = {r.entity_id: all_triples[rng.randrange(len(all_triples))] for r in comp}
    return records, by_id, bridges, comp, held, indist_holdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-entities", type=int, default=10000)
    ap.add_argument("--held-frac", type=float, default=0.2)
    ap.add_argument("--total-tokens", type=int, default=600_000_000)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--eval-only", action="store_true",
                    help="regenerate ONLY eval.json + store.json (no token bins); "
                         "deterministic in seed so it matches an existing corpus.")
    args = ap.parse_args()

    if args.eval_only:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        recs, by_id, bridges, comp, held, indist_holdout = _records_graph_holdout(
            args.n_entities, args.held_frac, args.seed)
        emit_eval_and_store(out, recs, by_id, bridges, comp, held, indist_holdout,
                            args.n_eval, args.seed)
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = get_tok()

    records, by_id, bridges, comp, held, indist_holdout = _records_graph_holdout(
        args.n_entities, args.held_frac, args.seed)
    print(f"[build] {len(records)} entities: {len(comp)} comp, {len(held)} held; seed={args.seed}")

    # ------- lane generators (infinite, exposure-cycling, deterministic) -------
    def bio_gen():
        exp = 0
        while True:
            for r in records:
                yield ("bio", r.entity_id, None, None, exp)
            exp += 1

    def bridge_gen():
        exp = 0
        while True:
            for r in records:
                for rel in RELS:
                    yield ("bridge", r.entity_id, rel, None, exp)
            exp += 1

    def compose_gen():
        exp = 0
        while True:
            for r in comp:
                for rel in RELS:
                    for attr in COMPOSE_ATTRS:
                        if indist_holdout[r.entity_id] == (rel, attr):
                            continue  # reserved for in-distribution eval
                        yield ("compose", r.entity_id, rel, attr, exp)
            exp += 1

    gens = {"bio": bio_gen(), "bridge": bridge_gen(), "compose": compose_gen()}
    targets = {"bio": 0.20, "bridge": 0.13, "compose": 0.67}
    lane_tokens = {k: 0 for k in gens}

    dense_w = ArmWriter(out / "dense", with_mask=False)
    split_w = ArmWriter(out / "split", with_mask=True)

    def render(desc) -> Doc:
        kind, eid, rel, attr, exp = desc
        rec = by_id[eid]
        if kind == "bio":
            return render_bio_doc(rec, exp)
        if kind == "bridge":
            return render_bridge_doc(rec, rel, by_id[bridges[eid][rel]], exp)
        return render_compose_doc(rec, rel, attr, by_id[bridges[eid][rel]], exp)

    total = 0
    n_docs = 0
    while dense_w.n < args.total_tokens:
        denom = max(1, total)
        lane = max(targets, key=lambda k: targets[k] - lane_tokens[k] / denom)
        doc = render(next(gens[lane]))
        d_ids, d_mask = tok.encode_segments(doc.dense_segments, add_eot=True)
        s_ids, s_mask = tok.encode_segments(doc.split_segments, add_eot=True)
        dense_w.add(d_ids, d_mask)
        split_w.add(s_ids, s_mask)
        lane_tokens[lane] += len(d_ids)
        total += len(d_ids)
        n_docs += 1
        if n_docs % 50000 == 0:
            print(f"  {dense_w.n/1e6:.1f}M dense tok | mix "
                  f"{ {k: round(v/denom,3) for k,v in lane_tokens.items()} }", flush=True)

    dense_w.close(); split_w.close()

    # ---------------------------- eval sets + store ----------------------------
    emit_eval_and_store(out, records, by_id, bridges, comp, held, indist_holdout,
                        args.n_eval, args.seed)

    report = {
        "seed": args.seed, "n_entities": args.n_entities, "held_frac": args.held_frac,
        "n_comp": len(comp), "n_held": len(held),
        "dense_tokens": dense_w.n, "split_tokens": split_w.n,
        "lane_tokens": lane_tokens, "lane_fracs": {k: v / total for k, v in lane_tokens.items()},
        "n_docs": n_docs, "compose_attrs": list(COMPOSE_ATTRS), "rels": list(RELS),
        "total_tokens_arg": args.total_tokens,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[build] DONE dense={dense_w.n/1e6:.1f}M split={split_w.n/1e6:.1f}M tok  "
          f"fracs={report['lane_fracs']}  -> {out}")


if __name__ == "__main__":
    main()
