"""Read the padding defect's signature off already-committed per-item files.

`docs/RESULTS-2026-08-04.md` retracted every generative number in the project
because `generate_batch` left-padded prompts with EOT and attended over the
pads. Re-measuring needs the checkpoints, which live on FarmShare. This script
needs nothing but `outputs/`, and answers the cheaper question first: does each
committed run actually carry the defect's fingerprint?

Two fingerprints, both from docs/PAPER-MEASUREMENT.md section 3:

1. iGSM accuracy *rising* with operation count. Padding is the gap between a
   prompt and the longest prompt in its batch, and mean prompt length rises
   with operations (100, 136, 206, 245 tokens for op 1-4), so short problems
   are padded hardest. A model that reasons should fall with op count.
2. Deduction collapsing onto one class. The eval is 200 yes / 200 no in strict
   alternation with a canned NO-branch trace, so a constant "no" scores exactly
   0.500 and an aggregate near 0.5 is uninformative on its own.

Usage:  PYTHONPATH=. python3 scripts/diagnose_padding_signature.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "outputs" / "cluster-summaries" / "evals"


def rows(run: str, task: str) -> list[dict]:
    path = EVALS / run / f"{task}.jsonl"
    if not path.exists():
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def rate(items: list[dict]) -> tuple[float, int]:
    if not items:
        return float("nan"), 0
    return sum(bool(r["correct"]) for r in items) / len(items), len(items)


def by_meta(items: list[dict], key: str) -> dict:
    buckets: dict[object, list[dict]] = defaultdict(list)
    for r in items:
        buckets[r.get("meta", {}).get(key)].append(r)
    return dict(sorted(buckets.items(), key=lambda kv: (kv[0] is None, kv[0])))


def main() -> int:
    runs = sorted(p.name for p in EVALS.iterdir() if p.is_dir())

    print("=" * 74)
    print("FINGERPRINT 1 -- iGSM accuracy by operation count")
    print("A reasoning model falls with op count. The padding defect makes it rise,")
    print("because longer prompts sit closer to the batch maximum and are padded less.")
    print("=" * 74)
    print(f"{'run':<34}{'op1':>9}{'op2':>9}{'op3':>9}{'op4':>9}   trend")
    for run in runs:
        items = rows(run, "igsm")
        if not items:
            continue
        cells, accs = [], []
        for op in (1, 2, 3, 4):
            acc, n = rate(by_meta(items, "op").get(op, []))
            cells.append(f"{acc:.3f}" if n else "  -  ")
            if n:
                accs.append(acc)
        trend = "RISING (defect)" if len(accs) > 1 and accs[-1] > accs[0] else "falling/flat"
        print(f"{run:<34}" + "".join(f"{c:>9}" for c in cells) + f"   {trend}")

    print()
    print("=" * 74)
    print("FINGERPRINT 2 -- deduction by class (a constant 'no' scores exactly 0.500)")
    print("=" * 74)
    print(f"{'run':<34}{'yes':>9}{'no':>9}{'overall':>10}   reading")
    for run in runs:
        items = rows(run, "deduction")
        if not items:
            continue
        buckets = by_meta(items, "answer_class")
        if set(buckets) == {None}:  # class not in meta; recover from the gold answer
            buckets = defaultdict(list)
            for r in items:
                buckets[str(r["answer"]).strip().lower()].append(r)
        yes, n_yes = rate(buckets.get("yes", []))
        no, n_no = rate(buckets.get("no", []))
        overall, _ = rate(items)
        if n_yes and n_no:
            skew = abs(no - yes)
            reading = "CONSTANT-'no' signature" if skew > 0.35 else "both classes answered"
        else:
            reading = "class not recoverable"
        print(f"{run:<34}{yes:>9.3f}{no:>9.3f}{overall:>10.3f}   {reading}")

    print()
    print("=" * 74)
    print("Fresh-entity lookup by relation (the mechanism result)")
    print("=" * 74)
    for run in runs:
        items = rows(run, "factqa_fresh")
        if not items:
            continue
        overall, n = rate(items)
        print(f"{run}  overall {overall:.4f} over n={n}")
        for rel, group in by_meta(items, "relation").items():
            acc, k = rate(group)
            print(f"    {str(rel):<14} {acc:.4f}  (n={k})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
