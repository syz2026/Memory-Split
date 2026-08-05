"""Controls for the recoverable-bits probe.

The probe asks a model for a fact using a phrasing absent from training and
converts the answer's NLL into bits against a pool baseline. On the 100-exposure
run it returns about -113 bits per entity, and returns the same for entities the
model never saw. Section 6 of the paper reads that as: no retrievable knowledge.

A reviewer's first move is that the probe may simply not work, and a metric that
cannot go positive proves nothing by going negative. This module runs the two
controls that settle it.

    negative control   the same probe on entities the model never saw. Already
                       reported: -112.29 against -112.90 for trained entities.
                       Rules out the probe rewarding entity familiarity.

    positive control   the same model, same entities, same values, queried in a
                       phrasing the model DID train on. If the metric goes
                       strongly positive here it can detect knowledge, and the
                       held-out null is about accessibility. If it stays
                       negative the metric is broken and the section falls.

Run both. Reporting the held-out null without the positive control is asserting
that an instrument works because it returned the answer you expected, which is
the mistake that cost this project four experiment generations.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# A gap this large between trained and unseen entities counts as the probe
# demonstrating it can see knowledge at all.
DETECTION_FLOOR_BITS = 5.0


def probe_bits(model, tok, records, device, phrasing: str,
               batch_size: int = 16) -> tuple[float, float, int]:
    """Mean recoverable bits per entity under one phrasing.

    `phrasing` is "training" to query with a surface form from the corpus
    generator, or "heldout" to use the probe templates that never appear in
    training.
    """
    from corpusgen import bios
    from evals import storage

    queries = []
    for rec in records:
        for attr in storage.ATTRIBUTES:
            if phrasing == "training":
                prefix = bios.BIO_TEMPLATES[attr][0][0]
            else:
                prefix = storage.PROBE_TEMPLATES[attr][0]
            queries.append({
                "entity_id": rec.entity_id,
                "attr": attr,
                "prompt": prefix.format(name=rec.name),
                "value": rec.attrs[attr],
            })

    nll = storage._value_nll_bits(model, tok, queries, device, batch_size)
    per_entity: dict[object, float] = {}
    for q, n in zip(queries, nll):
        base = storage.baseline_bits(q["attr"], q["value"], tok, "unconditional")
        per_entity[q["entity_id"]] = per_entity.get(q["entity_id"], 0.0) + (base - n)

    vals = list(per_entity.values())
    mean = statistics.fmean(vals)
    se = statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else float("nan")
    return mean, se, len(vals)


def verdict(training_gap: float, heldout_gap: float) -> dict:
    """What the two controls jointly license."""
    if training_gap <= DETECTION_FLOOR_BITS:
        return {
            "probe_detects_knowledge": False,
            "reading": (
                "POSITIVE CONTROL FAILS. The probe cannot separate trained from "
                "unseen entities even in the phrasing the model trained on, so "
                "it cannot detect knowledge at all and a negative result under "
                "held-out phrasing means nothing. Any claim resting on it must "
                "be withdrawn."),
        }
    if heldout_gap > DETECTION_FLOOR_BITS:
        return {
            "probe_detects_knowledge": True,
            "reading": (
                "Knowledge is retrievable under held-out phrasing. The facts are "
                "stored in an addressable form and the capacity argument is "
                "still live."),
        }
    return {
        "probe_detects_knowledge": True,
        "reading": (
            "POSITIVE CONTROL PASSES and the held-out gap does not. The probe "
            "can see knowledge when it is addressable, and finds none under a "
            "phrasing absent from training. The null is about accessibility, "
            "not about the instrument."),
    }


def main() -> int:
    import torch
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-entities", type=int, default=200)
    ap.add_argument("--unseen-seed", type=int, default=987654321)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from corpusgen import bios
    from scripts.run_evals import load_model
    from train.tokenizer import get_tok

    run = Path(args.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    device = torch.device(args.device)
    ckpt = Path(args.ckpt) if args.ckpt else None
    if ckpt is None:
        snaps = sorted((run / "snapshots").glob("step*.pt"))
        ckpt = snaps[-1] if snaps else run / "ckpt.pt"
    model, _ = load_model(run, cfg, device, ckpt)
    tok = get_tok()

    trained = bios.generate_records(args.n_entities, int(cfg.get("corpus_seed", 0)))
    unseen = bios.generate_records(args.n_entities, args.unseen_seed)
    assert {r.name for r in trained}.isdisjoint({r.name for r in unseen})

    cells = {}
    for phrasing in ("training", "heldout"):
        for label, recs in (("trained", trained), ("unseen", unseen)):
            m, se, n = probe_bits(model, tok, recs, device, phrasing,
                                  args.batch_size)
            cells[f"{phrasing}_{label}"] = {"bits_per_entity": m, "se": se, "n": n}
            print(f"  {phrasing:<9} {label:<8} {m:>10.2f} bits/entity  (SE {se:.2f})")

    tg = cells["training_trained"]["bits_per_entity"] - cells["training_unseen"]["bits_per_entity"]
    hg = cells["heldout_trained"]["bits_per_entity"] - cells["heldout_unseen"]["bits_per_entity"]
    out = {
        "checkpoint": str(ckpt),
        "cells": cells,
        "trained_minus_unseen_training_phrasing": tg,
        "trained_minus_unseen_heldout_phrasing": hg,
        "detection_floor_bits": DETECTION_FLOOR_BITS,
        **verdict(tg, hg),
    }
    print(f"\n  trained - unseen, training phrasing : {tg:+.2f} bits/entity")
    print(f"  trained - unseen, held-out phrasing : {hg:+.2f} bits/entity")
    print(f"\n{out['reading']}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    return 0 if out["probe_detects_knowledge"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
