"""Evaluate one finished run and write `<run>/evals/summary.json`.

This is the only producer of the file `scripts/analyze_crowding.py` consumes,
and everything it emits is required by a preregistered gate:

    igsm_acc                    primary endpoint, in-band
    igsm_by_op                  monotone-in-op is a construct check
    igsm_majority_rate          the real baseline, ~7.2%, not 1/23
    igsm_ood_acc                held-out op band, topology-disjoint
    deduction_by_class          a constant "no" scores exactly 0.500
    recoverable_bits_per_param  the storage manipulation check
    clip_ratio / clip_frac      arm comparability
    loss_masked_values          gate 0, from the training log

Accuracy is scored against the empirical majority-class rate computed on the
same item set, never against 1/23. Generation is greedy at temperature 0 with
max_new_tokens=384, as frozen in docs/PREREGISTRATION.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpusgen import bios, deduction, igsm_lite  # noqa: E402
from evals.scorers import accuracy_by, save_results, score_items  # noqa: E402
from evals.storage import recoverable_bits  # noqa: E402
from train.model import GPT, PRESETS, GPTConfig  # noqa: E402
from train.tokenizer import get_tok  # noqa: E402

MAX_NEW = 384


def load_model(run: Path, cfg: dict, device, ckpt: Path | None = None) -> GPT:
    mc = PRESETS[cfg["model"]] if isinstance(cfg["model"], str) else GPTConfig(**cfg["model"])
    if "ctx" in cfg:
        mc.ctx = cfg["ctx"]
    model = GPT(mc).to(device)
    state = torch.load(ckpt or (run / "ckpt.pt"), map_location=device,
                       weights_only=False)
    sd = state.get("model", state)
    # torch.compile prefixes every key; strip it so an uncompiled eval loads.
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, mc


def log_diagnostics(run: Path) -> dict:
    """Optimizer diagnostics and gate 0, averaged over the settled tail."""
    rows = []
    p = run / "log.jsonl"
    if p.exists():
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if not rows:
        return {}
    tail = rows[-max(1, len(rows) // 10):]

    def mean(key):
        vals = [r[key] for r in tail if key in r]
        return sum(vals) / len(vals) if vals else None

    mv = [r["loss_masked_values"] for r in rows if "loss_masked_values" in r]
    return {
        "final_step": rows[-1].get("step"),
        "final_loss_ema": rows[-1].get("loss_ema"),
        "clip_ratio": mean("clip_ratio"),
        "clip_frac": mean("clip_frac"),
        "grad_norm_preclip": mean("grad_norm_preclip"),
        "adam_v_mean": mean("adam_v_mean"),
        "loss_masked_values_final": mv[-1] if mv else None,
        "n_gate0_points": len(mv),
    }


def evaluate(run: Path, device, n_igsm: int, n_ded: int,
             n_storage_entities: int, batch_size: int,
             ckpt: Path | None = None) -> dict:
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    tok = get_tok()
    model, mc = load_model(run, cfg, device, ckpt)

    mod = int(cfg.get("igsm_mod", 23))
    op_lo, op_hi = cfg.get("igsm_op", [1, 4])
    ood_lo, ood_hi = cfg.get("igsm_ood_op", [5, 8])
    seed = int(cfg["seed"])

    out: dict = {"run": run.name, "arm": cfg.get("arm"), "seed": seed,
                 "model": cfg.get("model"), "mod": mod,
                 "ckpt": (ckpt.name if ckpt else "ckpt.pt")}

    # In-band iGSM. Eval seeds are offset far from any training seed so the
    # held-out draw cannot collide with the corpus.
    items = igsm_lite.generate_igsm_eval(n_igsm, op_lo, op_hi, 10_000_000 + seed,
                                         set(), mod=mod)
    rows = score_items(model, tok, items, device, max_new=MAX_NEW,
                       batch_size=batch_size)
    by_op = accuracy_by(rows, "op")
    out["igsm_acc"] = by_op["overall"]
    out["igsm_by_op"] = by_op["by"]
    out["igsm_majority_rate"] = by_op["majority_rate"]
    out["igsm_lift_over_majority"] = by_op["overall"] - by_op["majority_rate"]
    out["igsm_n"] = by_op["n"]
    save_results(rows, run / "evals" / "igsm.jsonl")

    # Out-of-distribution band. Chain length is part of the topology, so a
    # different op band is topology-disjoint from the trained one.
    ood = igsm_lite.generate_igsm_eval(n_igsm // 2, ood_lo, ood_hi,
                                       20_000_000 + seed, set(), mod=mod)
    ood_rows = score_items(model, tok, ood, device, max_new=MAX_NEW,
                           batch_size=batch_size)
    ood_by_op = accuracy_by(ood_rows, "op")
    out["igsm_ood_acc"] = ood_by_op["overall"]
    out["igsm_ood_by_op"] = ood_by_op["by"]
    out["igsm_ood_majority_rate"] = ood_by_op["majority_rate"]
    save_results(ood_rows, run / "evals" / "igsm_ood.jsonl")

    # Deduction, per class only. The eval is exactly balanced and the NO
    # branch's trace is canned, so the aggregate cannot distinguish reasoning
    # from a constant "no".
    ded = deduction.generate_deduction_eval(n_ded, 1, 2, 30_000_000 + seed, set())
    ded_rows = score_items(model, tok, ded, device, max_new=MAX_NEW,
                           batch_size=batch_size)
    by_class = accuracy_by(ded_rows, "answer_class")
    out["deduction_overall"] = by_class["overall"]
    out["deduction_by_class"] = by_class["by"]
    out["deduction_majority_rate"] = by_class["majority_rate"]
    save_results(ded_rows, run / "evals" / "deduction.jsonl")

    # Recoverable bits on the run's own fact set, under held-out paraphrases.
    n_entities = int(cfg.get("n_entities", 0))
    if n_entities:
        recs = bios.generate_records(min(n_entities, n_storage_entities),
                                     int(cfg.get("corpus_seed", seed)))
        for baseline in ("unconditional", "length"):
            rb = recoverable_bits(model, tok, recs, device, baseline=baseline,
                                  batch_size=batch_size,
                                  n_params=model.num_params())
            key = "" if baseline == "unconditional" else "_length"
            out[f"recoverable_bits_per_entity{key}"] = rb["bits_per_entity"]
            out[f"recoverable_bits_per_param{key}"] = (
                rb["bits_per_entity"] * n_entities / model.num_params()
            )
            out[f"recoverable_bits_per_attribute{key}"] = {
                a: v["bits_per_entity"] for a, v in rb["per_attribute"].items()
            }
        out["recoverable_bits_scaled_from"] = len(recs)
        out["n_entities"] = n_entities
    else:
        out["recoverable_bits_per_param"] = 0.0
        out["note_storage"] = "config carries no n_entities; nothing to probe"

    out.update(log_diagnostics(run))
    out["n_params"] = model.num_params()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-igsm", type=int, default=1500)
    ap.add_argument("--n-deduction", type=int, default=400)
    ap.add_argument("--n-storage-entities", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    # Scoring an intermediate snapshot turns one training run into a
    # step ladder at no extra training cost.
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint to score; defaults to the run's ckpt.pt")
    ap.add_argument("--tag", default=None,
                    help="filename stem under evals/; defaults to 'summary'")
    args = ap.parse_args()

    run = Path(args.run)
    (run / "evals").mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt) if args.ckpt else None
    out = evaluate(run, torch.device(args.device), args.n_igsm,
                   args.n_deduction, args.n_storage_entities, args.batch_size,
                   ckpt)
    stem = args.tag or (ckpt.stem if ckpt else "summary")
    (run / "evals" / f"{stem}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(
        {k: out[k] for k in out
         if k.startswith(("igsm_acc", "igsm_lift", "igsm_ood_acc",
                          "deduction_", "recoverable_bits_per_param",
                          "clip_", "loss_masked"))},
        indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
