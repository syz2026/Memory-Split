#!/usr/bin/env python
"""Build the OOD two-fact composition corpus (both arms) + organizer + evals.

Streams three doc types into per-arm uint16 token shards with uint8 loss masks:
    bio      atomic single-hop attribute facts  (all people, augmented)
    bridge   atomic single-hop bridge facts     (all people)  -> hop 1 access
    compose  two-hop reasoning docs             (P_comp only) -> the skill

Populations are disjoint with closed bridge edges (Karmim 2606.09338):
    P_comp -- two-hop docs in training; edges stay inside P_comp.
    P_held -- atomic facts only, never composed; edges stay inside P_held.

Writes to <out>/:
    dense/train.bin,  dense/train.mask.bin
    split/train.bin,  split/train.mask.bin
    organizer.jsonl                 (attributes + bridge edges, all people)
    eval/comp_indist.jsonl          held-out P_comp triples (interpolation)
    eval/comp_ood.jsonl             P_held triples (the real OOD test)
    eval/singlehop.jsonl            hop-1 / hop-2 fact-access probes
    report.json                     shares, exposures, disjointness/leakage asserts

Usage (A100 80GB default is tuned for a one-night positive control):
    python scripts/build_compose.py --out data/compose_v1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Run from anywhere without PYTHONPATH: put the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from corpusgen import bios, compose
from corpusgen.records import ATTRIBUTES
from organizer.store import Organizer
from train.tokenizer import get_tok

_T0 = time.monotonic()


def _log(msg: str) -> None:
    print(f"[build_compose +{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


_COMPONENTS = ("bio", "bridge", "compose")


class _Writer:
    """uint16 token ids + uint8 loss mask, bounded RAM. mask=None => all ones."""

    _FLUSH = 1_000_000

    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self._bin = open(out_dir / "train.bin", "wb")
        self._mask = open(out_dir / "train.mask.bin", "wb")
        self._ids: list[np.ndarray] = []
        self._masks: list[np.ndarray | int] = []
        self._buf = 0
        self.component_tokens = {c: 0 for c in _COMPONENTS}
        self.component_docs = {c: 0 for c in _COMPONENTS}
        self.masked_tokens = 0
        self.total = 0

    def add(self, comp: str, ids: np.ndarray, mask: np.ndarray | None) -> None:
        n = len(ids)
        self._ids.append(ids)
        self._masks.append(mask if mask is not None else n)
        self._buf += n
        self.component_tokens[comp] += n
        self.component_docs[comp] += 1
        self.total += n
        if mask is not None:
            self.masked_tokens += n - int(mask.sum())
        if self._buf >= self._FLUSH:
            self.flush()

    def flush(self) -> None:
        if not self._ids:
            return
        np.concatenate(self._ids).tofile(self._bin)
        np.concatenate(
            [m if isinstance(m, np.ndarray) else np.ones(m, dtype=np.uint8)
             for m in self._masks]
        ).tofile(self._mask)
        self._ids, self._masks, self._buf = [], [], 0

    def close(self) -> None:
        self.flush()
        self._bin.close()
        self._mask.close()


def _encode(tok, doc, arm: str) -> tuple[np.ndarray, np.ndarray | None]:
    segs = doc.dense_segments if arm == "dense" else doc.split_segments
    ids, mask = tok.encode_segments(segs, add_eot=True)
    ids_arr = np.asarray(ids, dtype=np.uint16)
    if 0 in mask:
        return ids_arr, np.asarray(mask, dtype=np.uint8)
    return ids_arr, None


def _train_triples(comp_ids: list[int], held_per_person: int, seed: int
                   ) -> tuple[list[tuple[int, str, str]], set[tuple[int, str, str]]]:
    """Partition P_comp's (entity, relation, attr) space into train vs held-out.

    Every person keeps >=1 training triple; ``held_per_person`` triples per
    person are withheld for the interpolation eval. Deterministic in seed.
    """
    all_combos = [(rel, attr) for rel in compose.BRIDGE_RELATIONS
                  for attr in compose.COMPOSE_ATTRS]
    k = max(0, min(held_per_person, len(all_combos) - 1))
    train: list[tuple[int, str, str]] = []
    held: set[tuple[int, str, str]] = set()
    for eid in comp_ids:
        rng = random.Random(f"heldout:{seed}:{eid}")
        combos = list(all_combos)
        rng.shuffle(combos)
        for rel, attr in combos[:k]:
            held.add((eid, rel, attr))
        for rel, attr in combos[k:]:
            train.append((eid, rel, attr))
    return train, held


def build(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = get_tok()

    # ---- records + disjoint populations ----------------------------------
    records = bios.generate_records(args.n_entities, args.seed)
    rec_by_id = {r.entity_id: r for r in records}
    n_comp = int(round(args.n_entities * (1.0 - args.held_frac)))
    P_comp = records[:n_comp]
    P_held = records[n_comp:]
    if len(P_held) < 2:
        raise ValueError("held population too small; lower --held-frac or raise --n-entities")
    comp_ids = [r.entity_id for r in P_comp]
    held_ids = [r.entity_id for r in P_held]
    _log(f"records={len(records)}  P_comp={len(P_comp)}  P_held={len(P_held)}")

    # closed bridge edges within each population
    edges_comp = compose.assign_bridges(P_comp, args.seed)
    edges_held = compose.assign_bridges(P_held, args.seed + 1)
    edges = {**edges_comp, **edges_held}

    # ---- organizer: attributes + bridge edges for ALL people -------------
    org = Organizer()
    for rec in records:
        for attr in ATTRIBUTES:
            org.add(rec.name, attr, rec.attrs[attr])
    for eid, rels in edges.items():
        for rel, tgt in rels.items():
            org.add(rec_by_id[eid].name, rel, rec_by_id[tgt].name)
    org.save(out / "organizer.jsonl")
    _log(f"organizer saved: {len(org)} keys")

    # ---- train vs held-out composition triples ---------------------------
    train_triples, held_triples = _train_triples(comp_ids, args.held_per_person, args.seed)
    train_triple_set = set(train_triples)
    _log(f"compose triples: train={len(train_triples)}  held_out={len(held_triples)}")

    budgets = {
        "bio": int(args.total_tokens * args.bio_share),
        "bridge": int(args.total_tokens * args.bridge_share),
        "compose": int(args.total_tokens * args.compose_share),
    }

    # ---- streams (generated on demand; O(1) memory) ----------------------
    def bio_stream(arm):
        exp = 0
        while True:
            for rec in records:
                yield _encode(tok, bios.render_bio_doc(rec, exp), arm)
            exp += 1

    def bridge_stream(arm):
        exp = 0
        while True:
            for eid in edges:  # all people, both relations
                x = rec_by_id[eid]
                for rel in compose.BRIDGE_RELATIONS:
                    y = rec_by_id[edges[eid][rel]]
                    yield _encode(tok, compose.render_bridge_doc(x, y.name, rel, exp), arm)
            exp += 1

    def compose_stream(arm):
        exp = 0
        while True:
            for eid, rel, attr in train_triples:
                x = rec_by_id[eid]
                y = rec_by_id[edges[eid][rel]]
                yield _encode(tok, compose.render_compose_doc(x, rel, y, attr, exp), arm)
            exp += 1

    # ---- per-arm assembly (largest token-deficit first) ------------------
    report_arms: dict[str, dict] = {}
    for arm in args.arms:
        writer = _Writer(out / arm)
        streams = {"bio": bio_stream(arm), "bridge": bridge_stream(arm),
                   "compose": compose_stream(arm)}
        active = set(_COMPONENTS)
        next_log = args.total_tokens / 8
        while active:
            comp = max(sorted(active),
                       key=lambda c: (budgets[c] - writer.component_tokens[c]) / max(1, budgets[c]))
            if writer.component_tokens[comp] >= budgets[comp]:
                active.discard(comp)
                continue
            ids, mask = next(streams[comp])
            if arm == "dense":
                mask = None
            writer.add(comp, ids, mask)
            if writer.total >= next_log:
                _log(f"{arm}: {writer.total/1e6:.0f}M/{args.total_tokens/1e6:.0f}M tokens")
                next_log += args.total_tokens / 8
        writer.close()
        report_arms[arm] = {
            "total_tokens": writer.total,
            "component_tokens": dict(writer.component_tokens),
            "component_shares": {c: writer.component_tokens[c] / writer.total
                                 for c in _COMPONENTS},
            "component_docs": dict(writer.component_docs),
            "bio_exposures_per_entity": writer.component_docs["bio"] / len(records),
            "compose_exposures_per_triple":
                writer.component_docs["compose"] / max(1, len(train_triples)),
            "masked_token_frac": writer.masked_tokens / writer.total,
        }
        _log(f"{arm}: done ({writer.total/1e6:.0f}M tokens, "
             f"{report_arms[arm]['compose_exposures_per_triple']:.0f} exposures/triple)")

    # ---- eval sets --------------------------------------------------------
    eval_dir = out / "eval"
    eval_dir.mkdir(exist_ok=True)
    rng = random.Random(args.seed * 7 + 1)

    # comp_indist: sample held-out P_comp triples (interpolation control)
    held_list = sorted(held_triples)
    rng.shuffle(held_list)
    comp_indist = [
        compose.compose_item(rec_by_id[eid], rel,
                             rec_by_id[edges[eid][rel]], attr, "comp")
        for eid, rel, attr in held_list[: args.n_eval]
    ]

    # comp_ood: sample P_held two-hop triples (never composed in training)
    ood_space = [(eid, rel, attr) for eid in held_ids
                 for rel in compose.BRIDGE_RELATIONS for attr in compose.COMPOSE_ATTRS]
    rng.shuffle(ood_space)
    comp_ood = [
        compose.compose_item(rec_by_id[eid], rel,
                             rec_by_id[edges[eid][rel]], attr, "held")
        for eid, rel, attr in ood_space[: args.n_eval]
    ]

    # singlehop: the exact hop-1 and hop-2 facts each eval item depends on,
    # so OOD two-hop accuracy can be conditioned on fact-access.
    singlehop: list = []
    seen_probe: set[str] = set()
    for item in comp_indist + comp_ood:
        m = item.meta
        x, y = rec_by_id[m["x_id"]], rec_by_id[m["y_id"]]
        p1 = compose.bridge_probe(x, y, m["relation"], m["population"])
        p2 = compose.attr_probe(y, m["attr"], m["population"])
        for p in (p1, p2):
            if p.qid not in seen_probe:
                seen_probe.add(p.qid)
                singlehop.append(p)

    for stem, items in [("comp_indist", comp_indist), ("comp_ood", comp_ood),
                        ("singlehop", singlehop)]:
        with open(eval_dir / f"{stem}.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps(asdict(it)) + "\n")
    _log(f"evals: comp_indist={len(comp_indist)} comp_ood={len(comp_ood)} "
         f"singlehop={len(singlehop)}")

    # ---- report + integrity checks ---------------------------------------
    n = len(records)
    disjoint = set(comp_ids).isdisjoint(held_ids)
    # every held-out test triple's constituents are trainable and not composed
    leak = [t for t in held_triples if t in train_triple_set]
    # organizer covers every hop of every eval item
    def _covered(items):
        return all(org.lookup(it.meta["hop1_key"]) is not None
                   and org.lookup(it.meta["hop2_key"]) is not None for it in items)
    org_covers = _covered(comp_indist) and _covered(comp_ood)

    report = {
        "cfg": vars(args),
        "populations": {"n_entities": n, "P_comp": len(P_comp), "P_held": len(P_held)},
        "bridge_relations": list(compose.BRIDGE_RELATIONS),
        "compose_attrs": list(compose.COMPOSE_ATTRS),
        "n_train_triples": len(train_triples),
        "n_held_triples": len(held_triples),
        "arms": report_arms,
        "eval_counts": {"comp_indist": len(comp_indist), "comp_ood": len(comp_ood),
                        "singlehop": len(singlehop)},
        "checks": {
            "populations_disjoint": disjoint,
            "no_heldout_leakage_into_training": len(leak) == 0,
            "organizer_covers_all_eval_hops": org_covers,
            "organizer_size_ok": len(org) == n * (len(ATTRIBUTES) + len(compose.BRIDGE_RELATIONS)),
        },
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    _log("report written")
    for k, v in report["checks"].items():
        _log(f"  check {k}: {'OK' if v else 'FAIL'}")
    if not all(report["checks"].values()):
        raise SystemExit("integrity checks FAILED; see report.json")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-entities", type=int, default=10_000)
    ap.add_argument("--held-frac", type=float, default=0.2,
                    help="fraction of entities reserved as OOD P_held")
    ap.add_argument("--total-tokens", type=int, default=600_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--held-per-person", type=int, default=2,
                    help="P_comp triples withheld per person for the interpolation eval")
    ap.add_argument("--bio-share", type=float, default=0.20)
    ap.add_argument("--bridge-share", type=float, default=0.13)
    ap.add_argument("--compose-share", type=float, default=0.67)
    ap.add_argument("--arms", nargs="+", default=["dense", "split"],
                    choices=["dense", "split"])
    ap.add_argument("--n-eval", type=int, default=1000)
    args = ap.parse_args()
    shares = args.bio_share + args.bridge_share + args.compose_share
    if abs(shares - 1.0) > 1e-6:
        ap.error(f"shares must sum to 1.0 (got {shares})")
    build(args)


if __name__ == "__main__":
    main()
