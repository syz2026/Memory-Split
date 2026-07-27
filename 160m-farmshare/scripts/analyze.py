#!/usr/bin/env python
"""Aggregate eval outputs across runs into the dose-response figure, the
paired confirmation contrast, and a markdown summary.

Usage:
  python scripts/analyze.py --runs-root outputs [--stage sweep|confirm|all]
      [--out outputs/analysis]

Expects run dirs named {preset}_{arm}_{load}_s{seed}[...] each containing
evals/summary.json (+ per-task JSONLs for paired stats).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from evals.figures import dose_response_figure
from evals.stats import paired_delta, seed_summary

RUN_RE = re.compile(r"^(?P<preset>d\w+?)_(?P<arm>dense|split)_(?P<load>n\d+k)_s(?P<seed>\d+)$")
LOADS = {"n50k": 50_000, "n200k": 200_000, "n800k": 800_000}


def load_rows(run_dir: Path, task: str) -> list[dict]:
    path = run_dir / "evals" / f"{task}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="outputs")
    ap.add_argument("--out", default="outputs/analysis")
    args = ap.parse_args()

    root = Path(args.runs_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = {}
    for d in sorted(root.iterdir()):
        m = RUN_RE.match(d.name)
        if m and (d / "evals" / "summary.json").exists():
            runs[d.name] = {**m.groupdict(), "dir": d,
                            "summary": json.load(open(d / "evals" / "summary.json"))}
    if not runs:
        print("no completed runs with evals found")
        return

    # dose-response points
    points = []
    for name, r in runs.items():
        comp = r["summary"].get("composite_knowledge_free")
        if comp is None:
            continue
        points.append({
            "n_entities": LOADS[r["load"]],
            "arm": r["arm"],
            "seed": int(r["seed"]),
            "preset": r["preset"],
            "composite": comp,
        })
    fig_points = [p for p in points if p["preset"] == "d160m"]
    if fig_points:
        dose_response_figure(fig_points, out / "dose_response.png")
        print(f"figure -> {out / 'dose_response.png'}")

    # paired contrasts: for every (preset, load, seed) with both arms present
    contrasts = []
    by_key = {}
    for name, r in runs.items():
        by_key[(r["preset"], r["load"], r["seed"], r["arm"])] = r
    for (preset, load, seed, arm), r in sorted(by_key.items()):
        if arm != "split":
            continue
        dense = by_key.get((preset, load, seed, "dense"))
        if not dense:
            continue
        per_task = {}
        for task in ("igsm", "deduction", "factqa"):
            rows_s = load_rows(r["dir"], task)
            rows_d = load_rows(dense["dir"], task)
            if rows_s and rows_d:
                per_task[task] = paired_delta(rows_s, rows_d)
        comp_s = r["summary"].get("composite_knowledge_free")
        comp_d = dense["summary"].get("composite_knowledge_free")
        contrasts.append({
            "preset": preset, "load": load, "seed": int(seed),
            "composite_delta": None if comp_s is None or comp_d is None else comp_s - comp_d,
            "per_task": per_task,
            "recall_split_on": r["summary"].get("recall", {}).get("on"),
            "recall_dense_closed": dense["summary"].get("recall", {}).get("closed"),
            "bits_split_off": r["summary"].get("bits_in_weights"),
            "bits_dense": dense["summary"].get("bits_in_weights"),
        })

    # seed-level summaries per (preset, load)
    groups: dict[tuple, list[float]] = {}
    for c in contrasts:
        if c["composite_delta"] is not None:
            groups.setdefault((c["preset"], c["load"]), []).append(c["composite_delta"])
    seed_stats = {f"{p}_{l}": seed_summary(v) for (p, l), v in groups.items()}

    result = {"contrasts": contrasts, "seed_stats": seed_stats, "n_runs": len(runs)}
    with open(out / "analysis.json", "w") as f:
        json.dump(result, f, indent=2)

    lines = ["# Memory-split analysis\n"]
    for key, st in seed_stats.items():
        lines.append(
            f"- **{key}**: composite delta (split - dense) mean {st['mean']:+.4f}, "
            f"seed sigma {st['seed_sigma']:.4f}, sign-consistent: {st['sign_consistent']}, "
            f"seeds: {st['n_seeds']}"
        )
    for c in contrasts:
        lines.append(
            f"\n## {c['preset']} {c['load']} seed {c['seed']}\n"
            f"- composite delta: {c['composite_delta']}\n"
            f"- guardrail: split(on) {c['recall_split_on']} vs dense(closed) {c['recall_dense_closed']}\n"
            f"- bits-in-weights: split(off) {c['bits_split_off']} vs dense {c['bits_dense']}"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"analysis -> {out / 'analysis.json'}, {out / 'summary.md'}")


if __name__ == "__main__":
    main()
