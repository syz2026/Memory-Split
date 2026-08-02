"""The staged pilot: measure at the real operating point, stop at the first
failure.

Three stages, each with its own kill decision, about 117 GPU-hours of
insurance on a ~300 GPU-hour bet.

Stage A -- endpoint ladder, ~4 GPU-h.
    One small dense corpus whose iGSM lane cycles through (MOD, op band)
    cells, crossed over n_recurrence in {1, 2}. Answers the two cheapest
    questions that can void everything downstream: can this endpoint move at
    all, and does dropping weight sharing cost the depth the endpoint needs.
    Freezes the architecture and the difficulty.

Stage B -- operating point and the effect ceiling, ~35 GPU-h.
    Two configurations at the exact final high-load settings: SUP with the
    full fact lane, and NOFACT with the fact lane replaced by bed tokens at
    equal token count and equal steps.

    NOFACT is the measurement whose absence was the largest hole in the first
    draft. `delta` -- the reasoning cost of carrying the fact load -- is the
    hard upper bound on any achievable treatment effect. If the facts cost the
    supervised arm 0.5 points, no masking scheme can recover more than 0.5,
    and a 2.0-point target is unachievable in principle. It also gives the
    effect interpretable units: "masking recovers X% of what the facts cost".

Stage C -- pilot triplets, ~78 GPU-h.
    Three complete SUP/FACTMASK/RANDPOS triplets at the high load. Delivers
    the paired SD of the actual primary contrast, which is what sets `n`; an
    early leakage check on FACTMASK; and the RANDPOS difficulty audit.

Pilot seeds are burned. They may set the design, the thresholds and `n`; they
may not appear in the confirmatory matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

# (MOD, op_lo, op_hi) cells for the Stage A ladder.
DIFFICULTY_CELLS = [
    (23, 1, 4), (23, 5, 8),
    (11, 1, 4), (11, 5, 8),
    (7, 1, 4), (7, 5, 8),
]
ARCHITECTURES = ["d40m", "d40m_std"]

PILOT_SEEDS = (9001, 9002, 9003)


def stage_a_configs(
    corpus_rel: str,
    total_tokens: int,
    lr: float,
    grad_clip: float,
) -> list[dict]:
    """One dense run per architecture, at ONE difficulty.

    This does not measure a difficulty ladder, despite an earlier version of
    this docstring saying the cells live inside the corpus and are separated at
    evaluation by meta tags. They do not: `build_corpus.py` takes a single
    `--mod`, so difficulty is fixed for a whole corpus and a ladder needs one
    corpus per rung. Stage A consequently ran two cells at MOD=23 that differed
    only in architecture, and its STOP rests on that single rung.

    The real ladder is `ops/crowding/ladder.py`. See docs/THEORY-ENDPOINT.md.
    """
    out = []
    for model in ARCHITECTURES:
        out.append(
            {
                "run_id": f"pilotA_{model}",
                "stage": "A",
                "arm": "sup",
                "model": model,
                "ctx": 1024,
                "vocab_size": 50304,
                "train_bin": f"{corpus_rel}/targets.bin",
                "micro_batch_size": 32,
                "tokens_per_step": 524_288,
                "total_tokens": total_tokens,
                "lr": lr,
                "warmup_steps": 300,
                "grad_clip": grad_clip,
                "seed": PILOT_SEEDS[0],
                "out_rel": f"runs/pilotA_{model}",
                "device": "cuda",
                "compile": True,
                "log_every": 20,
                "eval_every": 250,
                "snap_frac": 0.25,
            }
        )
    return out


def stage_b_configs(
    model: str,
    fact_corpus_rel: str,
    nofact_corpus_rel: str,
    total_tokens: int,
    lr: float,
    grad_clip: float,
    seeds=PILOT_SEEDS[:2],
) -> list[dict]:
    """SUP with the fact lane, and NOFACT with it replaced by bed at equal
    token count. Identical everywhere else, so their difference is the total
    reasoning cost of carrying the facts."""
    out = []
    for label, corpus in (("sup", fact_corpus_rel), ("nofact", nofact_corpus_rel)):
        for seed in seeds:
            out.append(
                {
                    "run_id": f"pilotB_{label}_s{seed}",
                    "stage": "B",
                    "arm": label,
                    "model": model,
                    "ctx": 1024,
                    "vocab_size": 50304,
                    "train_bin": f"{corpus}/targets.bin",
                    "probe_mask": f"{fact_corpus_rel}/factmask.bin",
                    "micro_batch_size": 32,
                    "tokens_per_step": 524_288,
                    "total_tokens": total_tokens,
                    "lr": lr,
                    "warmup_steps": 300,
                    "grad_clip": grad_clip,
                    "seed": seed,
                    "out_rel": f"runs/pilotB_{label}_s{seed}",
                    "device": "cuda",
                    "compile": True,
                    "log_every": 20,
                    "eval_every": 250,
                    "snap_frac": 0.25,
                }
            )
    return out


def stage_c_configs(
    model: str,
    corpus_rel: str,
    total_tokens: int,
    lr: float,
    grad_clip: float,
    seeds=PILOT_SEEDS,
) -> list[dict]:
    """Three complete triplets at the high load."""
    from importlib import util

    spec = util.spec_from_file_location(
        "gen_configs", Path(__file__).with_name("gen_configs.py")
    )
    gc = util.module_from_spec(spec)
    spec.loader.exec_module(gc)

    cfgs = gc.cohort({"high": corpus_rel}, list(seeds), model,
                     total_tokens, lr, grad_clip)
    for c in cfgs:
        c["stage"] = "C"
        c["run_id"] = "pilotC_" + c["run_id"]
        c["out_rel"] = f"runs/{c['run_id']}"
    gc.assert_arms_match(
        [{**c, "run_id": c["run_id"].replace("pilotC_", "")} for c in cfgs]
    )
    return cfgs


def write(configs: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for c in configs:
        (out / f"{c['run_id']}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    (out / "index.json").write_text(
        json.dumps([c["run_id"] for c in configs], indent=2)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B", "C"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="d40m")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--nofact-corpus", default=None)
    ap.add_argument("--total-tokens", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    args = ap.parse_args()

    if args.stage == "A":
        cfgs = stage_a_configs(args.corpus, args.total_tokens, args.lr, args.grad_clip)
    elif args.stage == "B":
        if not args.nofact_corpus:
            print("stage B needs --nofact-corpus")
            return 1
        cfgs = stage_b_configs(args.model, args.corpus, args.nofact_corpus,
                               args.total_tokens, args.lr, args.grad_clip)
    else:
        cfgs = stage_c_configs(args.model, args.corpus, args.total_tokens,
                               args.lr, args.grad_clip)

    write(cfgs, Path(args.out))
    print(f"stage {args.stage}: {len(cfgs)} configs -> {args.out}")
    for c in cfgs:
        print("  ", c["run_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
