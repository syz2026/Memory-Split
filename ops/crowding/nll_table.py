"""Freeze the per-token difficulty table that makes RANDPOS a real control.

The primary contrast is FACTMASK minus RANDPOS, so the control has to remove
the same amount of *learnable signal* as the treatment, not merely the same
number of targets. Fact values carry roughly 2.85 bits per token; template text
in a fixed biography frame is under 0.5. A count-matched control removes several
times less loss mass, and with gradient clipping in play that alone can move the
endpoint. Preregistration §2 therefore lists NLL as the fourth matching axis and
makes an unmatched control a reportable result rather than a silent confound.

Until 2026-08-02 the axis existed in `corpusgen/randpos.py` and was never
supplied: `build_corpus.py` called `randpos.build` with three positional
arguments and `token_nll` defaulted to None. Every corpus the project has ever
built matched mass, length and position but not difficulty.

What this produces
------------------
A `float32` array of shape `[vocab_size]`: the mean teacher-forced NLL of each
token id under a supervised checkpoint. Two properties matter.

**It is a marginal, not a contextual, difficulty.** The NLL of a token depends
on its context, and a per-id table averages that away. This is a deliberate
approximation and the only one available, because RANDPOS placement happens at
corpus-build time when no model has seen the document. It is adequate for its
job: separating high-surprisal value tokens from near-deterministic template
tokens, a gap of several nats that survives averaging.

**It must be frozen before it is used.** Choosing control placements with NLL
measured on the run being analysed would leak the outcome into the design. The
table is therefore built from a burned pilot seed (§6: seeds 9001-9003 may not
appear in the matrix) and written with a provenance sidecar recording the
checkpoint, its hash, the seed and the token count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from corpusgen import bios, factlane  # noqa: E402
from train.tokenizer import get_tok  # noqa: E402

# Preregistration §6: burned, and may not appear in the confirmatory matrix.
BURNED_PILOT_SEEDS = (9001, 9002, 9003)


def accumulate(model, docs, vocab_size: int, device, ctx: int,
               batch_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Sum of teacher-forced NLL per token id, and how often each id appeared."""
    sums = np.zeros(vocab_size, dtype=np.float64)
    counts = np.zeros(vocab_size, dtype=np.int64)

    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        width = min(ctx, max(len(d) for d in batch))
        # Right-pad to a rectangle and mask the padding out of the tally.
        idx = np.zeros((len(batch), width), dtype=np.int64)
        keep = np.zeros((len(batch), width), dtype=bool)
        for i, d in enumerate(batch):
            n = min(len(d), width)
            idx[i, :n] = d[:n]
            keep[i, :n] = True

        t = torch.from_numpy(idx).to(device)
        with torch.no_grad():
            logits, _ = model(t)
        # Position t predicts token t+1, so the NLL of target t+1 is read off
        # logits at t. The first token has no predictor and is dropped.
        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        tgt = t[:, 1:]
        nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

        valid = torch.from_numpy(keep[:, 1:]).to(device)
        ids = tgt[valid].cpu().numpy()
        vals = nll[valid].cpu().numpy().astype(np.float64)
        np.add.at(sums, ids, vals)
        np.add.at(counts, ids, 1)
    return sums, counts


def finalise(sums: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, dict]:
    """Mean NLL per id, with unseen ids given the corpus-wide mean.

    An unseen id has no measured difficulty. Filling it with the global mean
    makes it neither preferentially chosen nor preferentially avoided by the
    RANDPOS matcher, which is the neutral choice; filling it with zero would
    make every unseen token look maximally easy and pull control spans onto it.
    """
    seen = counts > 0
    table = np.zeros(len(sums), dtype=np.float32)
    global_mean = float(sums[seen].sum() / max(1, counts[seen].sum())) if seen.any() else 0.0
    table[seen] = (sums[seen] / counts[seen]).astype(np.float32)
    table[~seen] = global_mean
    return table, {
        "vocab_size": int(len(sums)),
        "ids_observed": int(seen.sum()),
        "ids_imputed_at_global_mean": int((~seen).sum()),
        "tokens_scored": int(counts.sum()),
        "global_mean_nll": global_mean,
        "observed_mean_nll": float(table[seen].mean()) if seen.any() else 0.0,
        "observed_p05_nll": float(np.percentile(table[seen], 5)) if seen.any() else 0.0,
        "observed_p95_nll": float(np.percentile(table[seen], 95)) if seen.any() else 0.0,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="a finished SUP run; its config names the architecture")
    ap.add_argument("--ckpt", default=None, help="defaults to <run>/ckpt.pt")
    ap.add_argument("--out", required=True, help="destination .npy")
    ap.add_argument("--seed", type=int, default=BURNED_PILOT_SEEDS[0],
                    help="record seed for the disjoint sample; must be a "
                         "burned pilot seed so the table cannot be built from "
                         "data that appears in the matrix")
    ap.add_argument("--entities", type=int, default=2000)
    ap.add_argument("--exposures", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-unburned-seed", action="store_true")
    args = ap.parse_args()

    if args.seed not in BURNED_PILOT_SEEDS and not args.allow_unburned_seed:
        print(f"seed {args.seed} is not one of the burned pilot seeds "
              f"{BURNED_PILOT_SEEDS}. Building the difficulty table from data "
              "that also appears in the matrix leaks the outcome into the "
              "design. Pass --allow-unburned-seed only if you have a reason.")
        return 1

    from scripts.run_evals import load_model

    run = Path(args.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    device = torch.device(args.device)
    ckpt = Path(args.ckpt) if args.ckpt else run / "ckpt.pt"
    model, mc = load_model(run, cfg, device, ckpt)

    tok = get_tok()
    records = bios.generate_records(args.entities, args.seed)
    docs = [factlane.render_one(records, k, tok)[0]
            for k in range(len(records) * args.exposures)]
    print(f"scoring {len(docs):,} disjoint fact documents at seed {args.seed}")

    sums, counts = accumulate(model, docs, cfg["vocab_size"], device,
                              mc.ctx, args.batch_size)
    table, stats = finalise(sums, counts)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, table)

    provenance = {
        "created": datetime.now(timezone.utc).isoformat(),
        "purpose": "frozen RANDPOS difficulty table, preregistration §2",
        "run": str(run),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": _sha256(ckpt),
        "record_seed": args.seed,
        "seed_is_burned_pilot": args.seed in BURNED_PILOT_SEEDS,
        "entities": args.entities,
        "exposures": args.exposures,
        "documents": len(docs),
        "table_sha256": _sha256(out),
        **stats,
    }
    side = out.with_suffix(".provenance.json")
    side.write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))
    print(f"\nwrote {out} and {side}")
    print("Freeze this now. Rebuilding it after seeing an outcome invalidates "
          "every corpus placed with it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
