"""The frozen analysis. Refuses anything but a complete, unaltered matrix.

The primary contrast is written in its reduced form:

    effect_i = Y[FACTMASK, i] - Y[RANDPOS, i]

because (FACTMASK - SUP) - (RANDPOS - SUP) cancels the SUP terms exactly, and
writing it as a double difference implies an identification property this
design does not have. SUP earns its place as the manipulation check and the
load main effect, not as part of the primary estimate.

What the estimate identifies is the effect of masking fact-value targets
rather than matched non-value targets. Capacity reallocation is an
interpretation of that, and it is not identified: gradient interference
predicts the same sign and the same monotone dose trend. The shape of the
effect across loads is what discriminates them -- crowding predicts a
threshold, interference predicts a line -- which is why loads matter more
than seeds beyond the powering minimum.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.stats import check_gates, seed_contrast, verdict  # noqa: E402

ARMS = ("sup", "factmask", "randpos")


class IncompleteMatrix(RuntimeError):
    pass


def discover(runs_root: Path) -> dict[tuple[str, int], dict[str, Path]]:
    """Map (load, seed) -> {arm: run dir}. Run ids are model_load_arm_sN."""
    found: dict[tuple[str, int], dict[str, Path]] = {}
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        parts = d.name.split("_")
        if len(parts) < 4 or not parts[-1].startswith("s"):
            continue
        arm, load = parts[-2], parts[-3]
        try:
            seed = int(parts[-1][1:])
        except ValueError:
            continue
        if arm not in ARMS:
            continue
        cell = found.setdefault((load, seed), {})
        if arm in cell:
            raise IncompleteMatrix(f"duplicate run for {load} s{seed} {arm}")
        cell[arm] = d
    return found


def require_complete(cells: dict[tuple[str, int], dict[str, Path]],
                     expect_loads: set[str] | None = None,
                     expect_seeds: set[int] | None = None) -> None:
    """A partial matrix must not be analysed and then completed. Inspecting
    results and then adding seeds is the same thing as choosing them."""
    if not cells:
        raise IncompleteMatrix("no runs found")
    loads = {k[0] for k in cells}
    seeds = {k[1] for k in cells}
    if expect_loads and loads != expect_loads:
        raise IncompleteMatrix(f"loads {sorted(loads)} != expected {sorted(expect_loads)}")
    if expect_seeds and seeds != expect_seeds:
        raise IncompleteMatrix(f"seeds {sorted(seeds)} != expected {sorted(expect_seeds)}")
    problems = []
    for load in sorted(loads):
        for seed in sorted(seeds):
            cell = cells.get((load, seed))
            if cell is None:
                problems.append(f"missing cell {load} s{seed}")
                continue
            missing = set(ARMS) - set(cell)
            if missing:
                problems.append(f"{load} s{seed} missing arms {sorted(missing)}")
    if problems:
        raise IncompleteMatrix("; ".join(problems))


def load_summary(run_dir: Path) -> dict:
    p = run_dir / "evals" / "summary.json"
    if not p.exists():
        raise IncompleteMatrix(f"{run_dir.name}: no evals/summary.json")
    return json.loads(p.read_text())


def analyse(runs_root: Path, endpoint: str, storage_floor: float,
            igsm_band: tuple[float, float], min_interesting_effect: float,
            expect_loads=None, expect_seeds=None) -> dict:
    cells = discover(runs_root)
    require_complete(cells, expect_loads, expect_seeds)

    per_load: dict[str, list[float]] = {}
    bits: dict[str, list[float]] = {a: [] for a in ARMS}
    clip: dict[str, list[float]] = {a: [] for a in ARMS}
    acc_sup: list[float] = []

    for (load, seed), cell in sorted(cells.items()):
        s = {arm: load_summary(d) for arm, d in cell.items()}
        effect = s["factmask"][endpoint] - s["randpos"][endpoint]
        per_load.setdefault(load, []).append(effect)
        for arm in ARMS:
            bits[arm].append(s[arm]["recoverable_bits_per_param"])
            clip[arm].append(s[arm]["clip_ratio"])
        acc_sup.append(s["sup"][endpoint])

    mean_bits = {a: sum(v) / len(v) for a, v in bits.items()}
    mean_clip = {a: sum(v) / len(v) for a, v in clip.items()}
    sup_stat = seed_contrast(bits["sup"])

    primary_load = max(per_load, key=lambda k: mean_bits["sup"]) if per_load else None
    gates = check_gates(
        bits=mean_bits,
        bits_lower_bound={"sup": sup_stat["ci_lower"]},
        igsm_acc_sup=sum(acc_sup) / len(acc_sup),
        clip_ratio=mean_clip,
        storage_floor=storage_floor,
        igsm_band=igsm_band,
    )
    all_effects = [e for v in per_load.values() for e in v]
    out = verdict(all_effects, gates, min_interesting_effect)
    out["endpoint"] = endpoint
    out["primary_contrast"] = "Y[factmask] - Y[randpos]"
    out["per_load"] = {
        k: {"effects": v, **seed_contrast(v)} for k, v in sorted(per_load.items())
    }
    out["recoverable_bits_per_param"] = mean_bits
    out["clip_ratio"] = mean_clip
    out["n_cells"] = len(cells)
    out["estimand_note"] = (
        "Effect of masking fact-value targets rather than matched non-value "
        "targets. Capacity reallocation is an interpretation and is NOT "
        "identified by this design; gradient interference predicts the same "
        "sign. Discriminate on the shape across loads, not the direction."
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint", default="igsm_acc")
    ap.add_argument("--storage-floor", type=float, required=True)
    ap.add_argument("--igsm-band", type=float, nargs=2, required=True)
    ap.add_argument("--min-interesting-effect", type=float, required=True)
    ap.add_argument("--expect-loads", nargs="*", default=None)
    ap.add_argument("--expect-seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    try:
        out = analyse(
            Path(args.runs_root), args.endpoint, args.storage_floor,
            tuple(args.igsm_band), args.min_interesting_effect,
            set(args.expect_loads) if args.expect_loads else None,
            set(args.expect_seeds) if args.expect_seeds else None,
        )
    except IncompleteMatrix as e:
        print(f"REFUSING TO ANALYSE: {e}")
        print("Do not inspect a partial matrix and then complete it.")
        return 1

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("verdict", "primary_contrast", "endpoint", "statistic",
                       "failed_validity_gates", "disclosed_reporting_failures")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
