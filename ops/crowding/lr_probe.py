"""Learning-rate probe on the real corpus, dense arm only.

An untuned rate makes a model look capacity-crowded when it is merely badly
optimised, which is the single easiest way to fake this experiment's result.
The retained probe puts d40m at 8.0e-3, but it was measured on a different
corpus, at n_recurrence=2, and under the old mean-reduction loss -- all three
of which move the optimum. d40m_std has never been probed at all.

Selection uses the dense arm only and the winner is applied unchanged to every
arm, so the choice cannot favour one over another. Both edges of the grid must
be worse than the interior, otherwise the optimum is unbracketed and the grid
has to be extended -- reporting a rate at the edge of a grid is reporting that
you did not find the optimum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

GRID = [1.5e-3, 3e-3, 6e-3, 1.2e-2, 2.4e-2]


def configs(corpus: str, model: str, root: str, steps: int, grid: list[float],
            grad_clip: float, n_entities: int) -> list[dict]:
    out = []
    for lr in grid:
        rid = f"lrprobe_{model}_{lr:g}"
        out.append({
            "run_id": rid,
            "stage": "lr-probe",
            "arm": "sup",
            "model": model,
            "ctx": 1024,
            "vocab_size": 50304,
            "train_bin": f"{corpus}/targets.bin",
            "probe_mask": f"{corpus}/factmask.bin",
            "micro_batch_size": 32,
            "tokens_per_step": 524_288,
            "max_steps": steps,
            "total_tokens": steps * 524_288,
            "lr": lr,
            "warmup_steps": 150,
            "weight_decay": 0.1,
            "grad_clip": grad_clip,
            "seed": 0,
            "n_entities": n_entities,
            "corpus_seed": 0,
            "igsm_mod": 23,
            "igsm_op": [1, 4],
            "igsm_ood_op": [5, 8],
            "out_dir": f"{root}/runs/{rid}",
            "device": "cuda",
            "compile": True,
            "log_every": 20,
            "eval_every": 100_000,   # the probe ranks on loss, not on evals
            "snap_frac": 1.0,
            "ckpt_minutes": 999,
        })
    return out


def rank(runs_root: Path, model: str) -> dict:
    """Rank by settled training loss; require the optimum to be bracketed."""
    rows = []
    for d in sorted(runs_root.glob(f"lrprobe_{model}_*")):
        log = d / "log.jsonl"
        if not log.exists():
            continue
        recs = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
        if not recs:
            continue
        tail = recs[-max(1, len(recs) // 10):]
        rows.append({
            "run": d.name,
            "lr": float(d.name.rsplit("_", 1)[1]),
            "final_loss": sum(r["loss_ema"] for r in tail) / len(tail),
            "steps": recs[-1]["step"],
            "clip_frac": sum(r.get("clip_frac", 0) for r in tail) / len(tail),
        })
    if not rows:
        return {"model": model, "error": "no probe runs found"}
    rows.sort(key=lambda r: r["lr"])
    best = min(rows, key=lambda r: r["final_loss"])
    i = rows.index(best)
    bracketed = 0 < i < len(rows) - 1
    return {
        "model": model,
        "grid": rows,
        "chosen_lr": best["lr"],
        "chosen_loss": best["final_loss"],
        "bracketed": bracketed,
        "note": (
            "optimum bracketed on both sides"
            if bracketed else
            "OPTIMUM AT A GRID EDGE -- extend the grid before freezing a rate; "
            "an edge value means the optimum was not found"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gen", "rank"], required=True)
    ap.add_argument("--out")
    ap.add_argument("--corpus")
    ap.add_argument("--root")
    ap.add_argument("--runs")
    ap.add_argument("--model", default="d40m_std")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--n-entities", type=int, default=0)
    ap.add_argument("--grid", type=float, nargs="*", default=GRID)
    args = ap.parse_args()

    if args.mode == "gen":
        cfgs = configs(args.corpus, args.model, args.root, args.steps,
                       args.grid, args.grad_clip, args.n_entities)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for c in cfgs:
            (out / f"{c['run_id']}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
            print("  ", c["run_id"])
        print(f"{len(cfgs)} probe configs -> {out}")
    else:
        r = rank(Path(args.runs), args.model)
        print(json.dumps(r, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(r, indent=2))
        return 0 if r.get("bracketed") else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
