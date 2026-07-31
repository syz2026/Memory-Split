"""Crowding readout for the tiny cohort.

Primary metric: split90 minus dense mean loss over the settled reasoning
extension, per model size. Within a size both arms share architecture,
recursion, tying, learning rate and data order, so this contrast is clean.

The across-size trend is NOT clean: d160m has n_recurrence=1 and untied
embeddings while d8m and d40m have recursion and tying, so three things vary
together. Report the trend qualitatively, never as a slope.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

BASE_TOKENS = 7_120_879_616
TOKENS_PER_STEP = 524_288
EXT_START = BASE_TOKENS // TOKENS_PER_STEP     # 13582
SETTLED_FROM = 14582
LAST_CLEAN = 15580                             # 15582 wrapped to the corpus head
UNIFORM = 10.8258                              # ln(50304)


def extension_stats(rows: list[dict]) -> dict:
    vals = [r["loss"] for r in rows if SETTLED_FROM <= r["step"] <= LAST_CLEAN]
    if not vals:
        return {"n": 0, "mean": float("nan"), "sd": float("nan")}
    return {"n": len(vals), "mean": st.mean(vals),
            "sd": st.stdev(vals) if len(vals) > 1 else 0.0}


def load(run_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (run_dir / "log.jsonl").open()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="directories holding <model>_<arm>_reasoning_v3_s0/")
    args = ap.parse_args()

    runs: dict[tuple[str, str], list[dict]] = {}
    for d in args.dirs:
        for p in sorted(Path(d).glob("*_reasoning_v3_s0")):
            model, arm = p.name.split("_")[0], p.name.split("_")[1]
            runs[(model, arm)] = load(p)

    order = [m for m in ("d8m", "d40m", "d160m") if (m, "dense") in runs]
    print("Settled reasoning extension, steps "
          f"{SETTLED_FROM}-{LAST_CLEAN} (identical supervision in both arms)\n")
    print(f"{'model':>7} {'n':>4} {'dense':>9} {'split90':>9} {'split-dense':>12}")
    for m in order:
        d, s = extension_stats(runs[(m, "dense")]), extension_stats(runs[(m, "split90")])
        print(f"{m:>7} {d['n']:>4} {d['mean']:>9.4f} {s['mean']:>9.4f} "
              f"{s['mean'] - d['mean']:>+12.4f}")

    print("\nCapacity indicator: dense extension loss by size "
          "(steep rise as size falls = capacity binding)")
    for m in order:
        print(f"  {m:>7} {extension_stats(runs[(m, 'dense')])['mean']:.4f}")

    print(f"\nGate 0: loss_masked_values on the offloaded positions "
          f"(uniform = {UNIFORM:.2f})")
    for m in order:
        for arm in ("dense", "split90"):
            pts = [r["loss_masked_values"] for r in runs[(m, arm)]
                   if "loss_masked_values" in r]
            if not pts:
                print(f"  {m:>7} {arm:>8}: NO PROBE POINTS - gate 0 did not fire")
            else:
                print(f"  {m:>7} {arm:>8}: first {pts[0]:.2f}  last {pts[-1]:.2f}")
    print("\nDense far below uniform means dense absorbed the offloaded content:"
          "\nthat is the memorisation burden the crowding hypothesis assumes.")


if __name__ == "__main__":
    main()
