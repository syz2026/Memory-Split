"""Did the model memorise entity-value bindings, or just the value distribution?

`probe_control.py` asks the model for one attribute behind a bare prefix and
finds trained and unseen entities indistinguishable, even in training phrasing.
Two things produce that reading and they have opposite consequences:

    the probe is insensitive   a one-attribute prompt strips the document
                               context the model trained under, so the model
                               cannot use what it stored.

    nothing was stored         the model learned the marginal distribution of
                               values and never bound them to names.

The bare-prefix probe cannot separate these. This one can. It scores value
tokens exactly the way gate 0 does -- full biography document, teacher-forced,
loss read at the marked value positions -- and adds the control gate 0 lacks:
the same measurement on entities the model has never seen.

Under memorisation, trained documents must be cheaper at the value positions
than unseen ones, because the values are predictable only if the binding was
learned. Under distribution-learning, the two cohorts cost the same: knowing
that employers are drawn from a pool of employers helps equally either way.

Gate 0 alone cannot tell these apart, which is why 1.9732 nats was never
evidence of storage.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Below this, the two cohorts are the same distribution and the model did not
# bind values to entities. In nats per value token.
BINDING_FLOOR_NATS = 0.10


def doc_value_nll(model, tok, rec, exposure_idx: int, device) -> tuple[float, int]:
    """Total NLL over value tokens of one rendered biography, and their count.

    Renders through the same function the corpus builder used, so the text is
    byte-identical to what training saw for this (entity, exposure).
    """
    import torch

    from corpusgen import bios

    ids: list[int] = []
    is_value: list[bool] = []
    for text, masked in bios.render_bio_marked(rec, exposure_idx):
        piece = tok.encode(text)
        ids.extend(piece)
        is_value.extend([bool(masked)] * len(piece))

    if not any(is_value) or len(ids) < 2:
        return 0.0, 0

    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(x)
    logp = torch.log_softmax(logits[0].float(), dim=-1)

    # Position p is predicted from p-1, so a value token at p is scored by
    # row p-1. Token 0 has no predictor and is never a value token in practice.
    total = 0.0
    n = 0
    for p in range(1, len(ids)):
        if is_value[p]:
            total += -logp[p - 1, ids[p]].item()
            n += 1
    return total, n


def cohort_nll(model, tok, records, exposures, device) -> dict:
    """Mean NLL per value token over a cohort, with a per-document SE."""
    per_doc = []
    per_person = []
    tok_total = 0
    for rec in records:
        doc_means = []
        for e in exposures:
            total, n = doc_value_nll(model, tok, rec, e, device)
            if n:
                doc_means.append(total / n)
                tok_total += n
        if doc_means:
            per_doc.extend(doc_means)
            per_person.append(statistics.fmean(doc_means))

    mean = statistics.fmean(per_doc)

    # Cluster by person. Each person contributes one document per exposure, and
    # those renderings share a name and six attribute values, so they are not
    # independent draws. Dividing by sqrt(n_documents) would understate the
    # interval by up to sqrt(exposures). The person-level SE is the honest one;
    # the document-level figure is kept only so the difference stays visible.
    def _se(xs):
        return statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else float("nan")

    return {
        "nats_per_value_token": mean,
        "se": _se(per_person),
        "se_by_document_unclustered": _se(per_doc),
        "n_people": len(per_person),
        "n_documents": len(per_doc),
        "n_value_tokens": tok_total,
        "bits_per_value_token": mean / math.log(2),
    }


def verdict(trained: dict, unseen: dict) -> dict:
    """What the contrast licenses."""
    gap = unseen["nats_per_value_token"] - trained["nats_per_value_token"]
    se = (trained["se"] ** 2 + unseen["se"] ** 2) ** 0.5
    stored = gap > BINDING_FLOOR_NATS and gap > 2 * se
    if stored:
        reading = (
            "BINDINGS LEARNED. Trained documents are cheaper at value positions "
            "than unseen ones by more than the floor, so the model did store "
            "entity-value associations. The held-out probe's null is therefore "
            "about accessibility under an unfamiliar query, not about absence, "
            "and the storage section must be rewritten as an accessibility "
            "claim.")
    else:
        reading = (
            "NO BINDINGS. Value tokens cost the same whether or not the model "
            "ever saw the entity, so gate 0's low loss reflects the marginal "
            "distribution of values and not memorisation. The model fitted fact "
            "text without storing facts.")
    return {
        "gap_nats_unseen_minus_trained": gap,
        "gap_se": se,
        "gap_z": gap / se if se else float("nan"),
        "binding_floor_nats": BINDING_FLOOR_NATS,
        "bindings_learned": stored,
        "reading": reading,
    }


def main() -> int:
    import torch
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-entities", type=int, default=200)
    ap.add_argument("--exposures", type=int, default=2,
                    help="renders per entity; exposure_idx 0..k-1")
    ap.add_argument("--unseen-seed", type=int, default=987654321)
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

    corpus_seed = int(cfg.get("corpus_seed", 0))
    trained_recs = bios.generate_records(args.n_entities, corpus_seed)
    unseen_recs = bios.generate_records(args.n_entities, args.unseen_seed)
    assert {r.name for r in trained_recs}.isdisjoint({r.name for r in unseen_recs})

    exposures = list(range(args.exposures))
    print(f"  checkpoint {ckpt.name}, {args.n_entities} entities x "
          f"{args.exposures} exposures per cohort")
    cells = {}
    for label, recs in (("trained", trained_recs), ("unseen", unseen_recs)):
        cells[label] = cohort_nll(model, tok, recs, exposures, device)
        c = cells[label]
        print(f"  {label:<8} {c['nats_per_value_token']:>8.4f} nats/value token  "
              f"(SE {c['se']:.4f}, {c['n_documents']} docs, "
              f"{c['n_value_tokens']:,} value tokens)")

    out = {
        "checkpoint": str(ckpt),
        "corpus_seed": corpus_seed,
        "cells": cells,
        **verdict(cells["trained"], cells["unseen"]),
    }
    print(f"\n  unseen - trained : {out['gap_nats_unseen_minus_trained']:+.4f} nats "
          f"(SE {out['gap_se']:.4f}, z = {out['gap_z']:+.2f})")
    print(f"\n{out['reading']}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
