"""Read Stage A and decide whether the endpoint is usable.

Emits the difficulty curve, the architecture comparison, and the one verdict
Stage A exists to produce: whether any cell clears the empirical
majority-class baseline by enough to be worth measuring against.

Everything is scored against the measured majority rate on the same item set,
never 1/23. That correction matters: at MOD 23 the best constant predictor
scores about 7.2% because `times` overproduces zero, and every iGSM number
this project published before was scored against 4.35%.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MIN_LIFT = 0.05


def collect(runs: Path) -> list[dict]:
    cells = []
    for d in sorted(p for p in runs.iterdir() if p.is_dir()):
        s = d / "evals" / "summary.json"
        if not s.exists():
            continue
        j = json.loads(s.read_text())
        cells.append({
            "run": d.name,
            "model": j.get("model"),
            "mod": j.get("mod"),
            "acc": j.get("igsm_acc", 0.0),
            "majority_rate": j.get("igsm_majority_rate", 0.0),
            "acc_by_op": {k: v["acc"] for k, v in (j.get("igsm_by_op") or {}).items()},
            "ood_acc": j.get("igsm_ood_acc"),
            "deduction_by_class": j.get("deduction_by_class"),
            "clip_frac": j.get("clip_frac"),
            "clip_ratio": j.get("clip_ratio"),
            "gate0": j.get("loss_masked_values_final"),
            "final_loss": j.get("final_loss_ema"),
            "n_params": j.get("n_params"),
        })
    return cells


def report(cells: list[dict]) -> dict:
    for c in cells:
        c["lift"] = c["acc"] - c["majority_rate"]
        c["clears_floor"] = c["lift"] >= MIN_LIFT
        ops = sorted(c["acc_by_op"], key=lambda k: int(k))
        c["monotone_in_op"] = (
            all(c["acc_by_op"][a] >= c["acc_by_op"][b] - 1e-9
                for a, b in zip(ops, ops[1:])) if len(ops) > 1 else None
        )
    viable = [c for c in cells if c["clears_floor"]]
    best = max(viable, key=lambda c: c["lift"]) if viable else None
    return {
        "cells": cells,
        "any_clears_floor": bool(viable),
        "chosen": best,
        "decision": "CONTINUE" if viable else "STOP",
        "rationale": (
            f"{best['run']} clears the majority-class baseline by "
            f"{best['lift']:.1%}; freeze this architecture and difficulty."
            if best else
            "No cell clears the majority-class baseline by 5 points. The "
            "endpoint has no power to discriminate anything, so a null from "
            "the matrix would be uninformative. Stop and write the NO-GO "
            "paper -- its outline is in docs/NO-GO-PAPER.md and Stage A "
            "already produced its difficulty-curve figure."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cells = collect(Path(args.runs))
    if not cells:
        print(f"no evaluated runs under {args.runs}")
        return 1
    r = report(cells)

    print(f"{'run':28} {'params':>11} {'acc':>7} {'major':>7} {'lift':>7} "
          f"{'op-mono':>8} {'ood':>7} {'clipfrac':>9}")
    for c in r["cells"]:
        print(f"{c['run'][:28]:28} {c['n_params'] or 0:>11,} "
              f"{c['acc']:>7.3f} {c['majority_rate']:>7.3f} {c['lift']:>+7.3f} "
              f"{str(c['monotone_in_op']):>8} "
              f"{(c['ood_acc'] if c['ood_acc'] is not None else float('nan')):>7.3f} "
              f"{(c['clip_frac'] if c['clip_frac'] is not None else float('nan')):>9.3f}")
    print(f"\ndecision: {r['decision']}")
    print(r["rationale"])

    if args.out:
        Path(args.out).write_text(json.dumps(r, indent=2))
    return 0 if r["decision"] == "CONTINUE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
