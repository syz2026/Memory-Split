"""Why can gate 0 exceed the uniform ceiling?

`loss_masked_values` returned 25.02 nats on a dense arm that trained on those
positions, against ln(50304) = 10.83. A model that had simply not learned the
content would sit at or just below uniform; 25 nats means it puts e^-25 on the
truth, roughly a million times *less* than chance. That is confident and wrong,
not ignorant.

The hypothesis this script tests: the model learned the slot and not the
filler. Biography text is heavily templated, so it knows a city follows "was
born in" and sharpens onto the modal city, while 20 exposures is far below the
threshold at which it could know which city belongs to this entity.

Three readings distinguish that from a measurement bug:

  what is scored   decode the masked positions. If they are not attribute
                   values, the probe is pointed at the wrong tokens.
  how sharp        the model's entropy at those positions. Confidently wrong
                   means low entropy and a wrong argmax; ignorant means
                   entropy near the pool's.
  the shape        the distribution of per-position CE. One pathological
                   outlier is a bug; a broad shift is a phenomenon.

If it is the phenomenon, ln(V) is not an upper bound on gate 0, and "near the
uniform ceiling" cannot be read as "knows nothing".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.data import PackedShards  # noqa: E402
from train.model import PRESETS, GPT, GPTConfig  # noqa: E402
from train.tokenizer import get_tok  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    tok = get_tok()
    device = torch.device(args.device)

    mc = PRESETS[cfg["model"]] if isinstance(cfg["model"], str) else GPTConfig(**cfg["model"])
    if "ctx" in cfg:
        mc.ctx = cfg["ctx"]
    model = GPT(mc).to(device)
    sd = torch.load(run / "ckpt.pt", map_location=device, weights_only=False)
    sd = sd.get("model", sd)
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()})
    model.eval()

    # Rebuild the probe batch exactly as the trainer does.
    data = PackedShards(
        cfg["train_bin"], cfg.get("train_mask"), ctx=mc.ctx,
        batch_size=cfg["micro_batch_size"], device="cpu",
        seed=cfg["seed"], probe_mask_path=cfg.get("probe_mask"),
    )
    batch = data.masked_value_batch()
    if batch is None:
        print("probe found no masked window")
        return 1
    x, y = batch

    scored, ces, ents, argmax_hits = [], [], [], 0
    with torch.no_grad():
        for i in range(0, x.size(0), 8):
            xb, yb = x[i:i + 8].to(device), y[i:i + 8].to(device)
            logits, _ = model(xb)
            lp = torch.log_softmax(logits.float(), dim=-1)
            for b in range(xb.size(0)):
                for t in range(xb.size(1)):
                    tgt = int(yb[b, t])
                    if tgt == -100:
                        continue
                    row = lp[b, t]
                    ce = -float(row[tgt])
                    ces.append(ce)
                    ents.append(float(-(row.exp() * row).sum()))
                    top = int(row.argmax())
                    argmax_hits += int(top == tgt)
                    if len(scored) < 40:
                        scored.append({
                            "true": tok.decode([tgt]),
                            "pred": tok.decode([top]),
                            "ce_nats": round(ce, 2),
                            "p_true": float(row[tgt].exp()),
                            "entropy_nats": round(float(-(row.exp() * row).sum()), 2),
                        })

    ces_a, ents_a = np.array(ces), np.array(ents)
    uniform = math.log(tok.VOCAB_SIZE)
    out = {
        "run": run.name,
        "n_scored_positions": len(ces),
        "uniform_ceiling_nats": round(uniform, 3),
        "mean_ce_nats": round(float(ces_a.mean()), 3),
        "median_ce_nats": round(float(np.median(ces_a)), 3),
        "frac_above_uniform": round(float((ces_a > uniform).mean()), 3),
        "ce_percentiles": {p: round(float(np.percentile(ces_a, p)), 2)
                           for p in (5, 25, 50, 75, 95, 99)},
        "mean_model_entropy_nats": round(float(ents_a.mean()), 3),
        "argmax_accuracy": round(argmax_hits / max(1, len(ces)), 4),
        "sample": scored[:20],
    }

    # The discriminator. Confidently wrong = low entropy, high CE. Ignorant =
    # entropy near uniform, CE near uniform.
    if out["mean_ce_nats"] > uniform and out["mean_model_entropy_nats"] < uniform * 0.6:
        out["verdict"] = (
            "CONFIDENTLY WRONG. The model is sharp at these positions and its "
            "mass is on the wrong token, so it learned the slot and not the "
            "filler. ln(V) is therefore NOT an upper bound on gate 0, and a "
            "reading near the uniform ceiling cannot be read as 'knows "
            "nothing'."
        )
    elif out["mean_ce_nats"] > uniform:
        out["verdict"] = (
            "ABOVE UNIFORM BUT DIFFUSE. High CE without sharpness does not fit "
            "the slot-without-filler account; check the probe alignment."
        )
    else:
        out["verdict"] = "At or below uniform; the 25-nat reading did not reproduce."

    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
