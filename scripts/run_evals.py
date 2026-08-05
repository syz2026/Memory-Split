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


def continuous_endpoint(model, tok, items, device, batch_size: int,
                        prefix: str) -> dict:
    """A reasoning endpoint that does not floor.

    Exact-match accuracy is a threshold metric: it requires the model to emit a
    whole correct chain of thought *and* terminate with the right token, so it
    reads zero across the entire range where a model is learning. Every run in
    this project has sat at that floor, and a floored endpoint cannot exhibit
    crowding no matter how much capacity is freed -- nothing was learned, so
    nothing could be crowded out.

    Reference-trace NLL is continuous over the same items. It moves while
    accuracy is pinned, which is the difference between an endpoint that can
    resolve a treatment effect and one that cannot.

    `evals/continuous.py` has implemented this since the reasoning-v3 line and
    was called by nothing until 2026-08-03. M1 is the primary: mean per-token
    -log p(gold trace | prompt), which is apples-to-apples across arms wherever
    both read the same rendering -- true for iGSM and deduction, false for the
    fact lane, which is why this runs only on the reasoning tasks.
    """
    from evals.continuous import score_items_continuous

    # M1 teacher-forces prompt+solution in one window, so an item longer than
    # the context is not scoreable and asserts inside the scorer. Drop those
    # rather than lose the metric for the whole run, and record how many, so a
    # silently thinned item set cannot be mistaken for a clean one.
    ctx = getattr(getattr(model, "cfg", None), "ctx", 1024)
    fits, dropped = [], 0
    for it in items:
        sol = (getattr(it, "meta", None) or {}).get("solution", "")
        n = len(tok.encode(getattr(it, "prompt", "") + sol))
        if n < ctx:
            fits.append(it)
        else:
            dropped += 1
    if not fits:
        return {f"{prefix}_continuous_error":
                f"every item exceeds ctx={ctx}; nothing scoreable"}

    try:
        _, agg = score_items_continuous(model, tok, fits, device,
                                        batch_size=batch_size)
    except Exception as e:                      # never lose a run over a metric
        return {f"{prefix}_continuous_error": f"{type(e).__name__}: {e}"}
    # `score_items_continuous` nests by task; flatten to scalars so the frozen
    # analyzer can take one of these as `--endpoint` directly.
    out: dict = {}
    for metric, by_task in agg.items():
        if not isinstance(by_task, dict):
            out[f"{prefix}_{metric}"] = by_task
            continue
        vals = [t["mean"] for t in by_task.values()
                if isinstance(t, dict) and t.get("n")]
        ns = [t["n"] for t in by_task.values()
              if isinstance(t, dict) and t.get("n")]
        if not vals:
            continue
        mean = sum(v * n for v, n in zip(vals, ns)) / sum(ns)
        out[f"{prefix}_{metric}"] = mean
        out[f"{prefix}_{metric}_n"] = sum(ns)
        # The analyzer's verdict is one-sided and assumes larger is better,
        # because the hypothesis predicts a positive effect. NLL runs the other
        # way, so a sign-flipped twin is emitted for use as the estimand and
        # the raw value is kept for reading.
        if metric.endswith("_nll"):
            out[f"{prefix}_{metric}_neg"] = -mean
    out[f"{prefix}_continuous_n_scored"] = len(fits)
    out[f"{prefix}_continuous_n_dropped_over_ctx"] = dropped
    return out


def format_compliance(rows: list[dict], mod: int, prefix: str) -> dict:
    """Separate "could not produce an answer" from "produced a wrong answer".

    Headline accuracy sums two unrelated failures. Measured on the 2026-08-02
    ladder at MOD=5, where only five answers exist: 51.5% of generations
    yielded an answer inside {0..4} at all, and the rest were unparseable, the
    string "no" borrowed from the deduction lane, or raw biography text. Among
    the ones that did land in the answer space, accuracy was 0.2500 against a
    0.2573 majority rate -- the marginal answer distribution, exactly.

    Conflating the two is dangerous in one specific direction: a model that
    merely learns to terminate with "Answer: <digit>" will lift headline
    accuracy toward the majority rate with no arithmetic learned, and that
    would read as the endpoint waking up. The step ladder must be able to tell
    those apart, so both numbers are reported.
    """
    n = len(rows)
    if not n:
        return {}
    answer_space = {str(i) for i in range(mod)}
    parsed = [r for r in rows if r.get("pred") is not None]
    in_space = [r for r in parsed if str(r["pred"]) in answer_space]
    correct = sum(1 for r in rows if r.get("correct"))
    return {
        f"{prefix}_parsed_rate": len(parsed) / n,
        f"{prefix}_in_answer_space_rate": len(in_space) / n,
        f"{prefix}_acc_given_valid_answer": correct / max(1, len(in_space)),
        f"{prefix}_n_in_answer_space": len(in_space),
    }


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
    out.update(format_compliance(rows, mod, prefix="igsm"))
    out.update(continuous_endpoint(model, tok, items, device, batch_size,
                                   prefix="igsm"))
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
    out.update(format_compliance(ood_rows, mod, prefix="igsm_ood"))
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
            scale = n_entities / model.num_params()
            out[f"recoverable_bits_per_param{key}"] = rb["bits_per_entity"] * scale
            # The point estimate is a few hundred entities extrapolated by
            # ~2,000x at the operating point, and `corpus_seed` is pinned so
            # every seed probes the same ones -- the error cannot show up in an
            # across-seed interval. Carry it explicitly.
            if "bits_per_entity_ci95" in rb:
                lo, hi = rb["bits_per_entity_ci95"]
                out[f"recoverable_bits_per_param{key}_ci95"] = [lo * scale, hi * scale]
                out[f"recoverable_bits_per_param{key}_se"] = (
                    rb["bits_per_entity_se"] * scale
                )
                out[f"recoverable_bits_rel_se{key}"] = rb["bits_per_entity_rel_se"]
            out[f"recoverable_bits_per_attribute{key}"] = {
                a: v["bits_per_entity"] for a, v in rb["per_attribute"].items()
            }
        out["recoverable_bits_scaled_from"] = len(recs)
        out["recoverable_bits_extrapolation_factor"] = n_entities / max(1, len(recs))
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
    # 500 of 996,408 entities is a 0.05% sample extrapolated ~2,000x into the
    # figure two validity gates are decided on. 4,000 costs a few extra minutes
    # of forward passes and cuts the sampling SE by ~2.8x.
    ap.add_argument("--n-storage-entities", type=int, default=4000)
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
