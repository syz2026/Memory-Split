"""Can the memorization control detect storage when storage is definitely there?

`memorization_control.py` reports that trained and never-seen entities cost the
same at value positions, and Section 6 reads that as the model having learned no
bindings. That reading repeats the mistake the same section is about unless the
instrument has been shown capable of the opposite answer. A measurement that has
only ever returned a null is not evidence of a null.

So this trains a small model to memorise a small entity set and then runs the
same measurement on it. Bindings are present by construction: the model sees
these entities and no others, many times, with nothing else to fit. If the
instrument cannot separate the cohorts here, it cannot separate them anywhere,
and Section 6 falls.

The result is a property of the instrument rather than of the 40M run, so it is
cheap and runs on a CPU. It does not need the cluster.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# A memorising model must beat the never-seen cohort by at least this much, in
# nats per value token, for the instrument to count as able to see storage.
DETECTION_FLOOR_NATS = 0.25


def build_stream(records, exposures: int, tok) -> list[int]:
    """Concatenate rendered biographies exactly as the corpus builder would."""
    from corpusgen import bios

    ids: list[int] = []
    for e in range(exposures):
        for rec in records:
            text = "".join(t for t, _ in bios.render_bio_marked(rec, e))
            ids.extend(tok.encode(text))
            ids.append(tok.EOT)
    return ids


def overfit(stream, cfg, steps: int, lr: float, batch: int, device, log_every: int):
    """Train until the value tokens are memorised, or `steps` runs out."""
    import torch

    from train.model import GPT

    torch.manual_seed(0)
    model = GPT(cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0,
                            betas=(0.9, 0.95))
    data = torch.tensor(stream, dtype=torch.long)
    ctx = cfg.ctx
    g = torch.Generator().manual_seed(1)
    t0 = time.time()
    for step in range(1, steps + 1):
        ix = torch.randint(0, max(1, len(data) - ctx - 1), (batch,), generator=g)
        x = torch.stack([data[i:i + ctx] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + ctx] for i in ix]).to(device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and (step % log_every == 0 or step == 1):
            print(f"    step {step:>5}  loss {loss.item():.4f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    model.eval()
    return model


def main() -> int:
    import torch

    from corpusgen import bios
    from ops.crowding.memorization_control import cohort_nll
    from train.model import GPTConfig
    from train.tokenizer import get_tok

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-entities", type=int, default=64)
    ap.add_argument("--exposures", type=int, default=8,
                    help="renders per entity in the training stream")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--trained-seed", type=int, default=0)
    ap.add_argument("--unseen-seed", type=int, default=987654321)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    tok = get_tok()
    trained = bios.generate_records(args.n_entities, args.trained_seed)
    unseen = bios.generate_records(args.n_entities, args.unseen_seed)
    assert {r.name for r in trained}.isdisjoint({r.name for r in unseen})

    stream = build_stream(trained, args.exposures, tok)
    print(f"  training stream: {len(stream):,} tokens from {args.n_entities} "
          f"entities x {args.exposures} exposures")

    cfg = GPTConfig(n_layer=args.n_layer, n_head=4, d_model=args.d_model,
                    ctx=args.ctx)
    model = overfit(stream, cfg, args.steps, args.lr, args.batch, device,
                    args.log_every)

    # Score with the instrument under test, on exposures it did train on.
    exps = list(range(min(2, args.exposures)))
    cells = {}
    for label, recs in (("trained", trained), ("unseen", unseen)):
        cells[label] = cohort_nll(model, tok, recs, exps, device)
        c = cells[label]
        print(f"  {label:<8} {c['nats_per_value_token']:>8.4f} nats/value token "
              f"(SE {c['se']:.4f}, {c['n_documents']} docs)")

    gap = (cells["unseen"]["nats_per_value_token"]
           - cells["trained"]["nats_per_value_token"])
    se = (cells["trained"]["se"] ** 2 + cells["unseen"]["se"] ** 2) ** 0.5
    detects = gap > DETECTION_FLOOR_NATS and gap > 2 * se

    out = {
        "n_entities": args.n_entities,
        "exposures": args.exposures,
        "steps": args.steps,
        "model": {"n_layer": args.n_layer, "d_model": args.d_model,
                  "ctx": args.ctx},
        "cells": cells,
        "gap_nats_unseen_minus_trained": gap,
        "gap_se": se,
        "gap_z": gap / se if se else float("nan"),
        "detection_floor_nats": DETECTION_FLOOR_NATS,
        "instrument_detects_storage": detects,
        "reading": (
            "POSITIVE CONTROL PASSES. On a model that demonstrably memorised its "
            "entities, the measurement separates trained from never-seen at the "
            "value positions. It can report storage when storage is present, so "
            "its null on the 40M run is informative."
            if detects else
            "POSITIVE CONTROL FAILS. Even against a model trained on nothing but "
            "these entities, the measurement cannot separate them from invented "
            "ones. It cannot detect storage, and the Section 6 null means "
            "nothing. Withdraw it."),
    }
    print(f"\n  unseen - trained : {gap:+.4f} nats (SE {se:.4f}, "
          f"z = {gap / se if se else float('nan'):+.2f})")
    print(f"\n{out['reading']}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\n  wrote {args.out}")
    return 0 if detects else 2


if __name__ == "__main__":
    raise SystemExit(main())
