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
from theory import capacity  # noqa: E402

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
            expect_loads=None, expect_seeds=None,
            primary_load: str | None = None) -> dict:
    """The frozen analysis, evaluated at one preregistered load.

    `primary_load` must be named by the caller and is not inferred, because
    every rule for picking it from the data -- highest SUP storage, largest
    effect, best-powered cell -- is a choice made after seeing outcomes.

    Inference is blocked by seed within that load. Pooling the three loads
    would treat 3 loads x 8 seeds as n=24 independent observations when the
    same eight initialisations and shard permutations appear at every load, so
    the intervals would be anti-conservative. Pooling is also substantively
    wrong: the theory predicts the effect *rises then saturates* across loads,
    so their mean estimates a quantity no hypothesis is about.

    The across-load structure is not discarded -- it is the mechanism test, and
    it is reported separately as the dose-response shape.
    """
    cells = discover(runs_root)
    require_complete(cells, expect_loads, expect_seeds)

    loads = sorted({k[0] for k in cells})
    if primary_load is None:
        raise IncompleteMatrix(
            f"--primary-load must be named explicitly (one of {loads}). "
            "Choosing it from the data after the fact -- by storage, by effect "
            "size, or by power -- is choosing the estimand after seeing the "
            "outcome. Freeze it in docs/PREREGISTRATION.md §5 first."
        )
    if primary_load not in loads:
        raise IncompleteMatrix(
            f"primary load '{primary_load}' is not in the matrix {loads}")

    per_load: dict[str, list[float]] = {}
    per_load_seeds: dict[str, list[int]] = {}
    bits: dict[str, dict[str, list[float]]] = {a: {} for a in ARMS}
    clip: dict[str, dict[str, list[float]]] = {a: {} for a in ARMS}
    acc_sup: dict[str, list[float]] = {}

    for (load, seed), cell in sorted(cells.items()):
        s = {arm: load_summary(d) for arm, d in cell.items()}
        per_load.setdefault(load, []).append(s["factmask"][endpoint]
                                             - s["randpos"][endpoint])
        per_load_seeds.setdefault(load, []).append(seed)
        for arm in ARMS:
            bits[arm].setdefault(load, []).append(
                s[arm]["recoverable_bits_per_param"])
            clip[arm].setdefault(load, []).append(s[arm]["clip_ratio"])
        acc_sup.setdefault(load, []).append(s["sup"][endpoint])

    def _mean(xs):
        return sum(xs) / len(xs)

    # Gates are evaluated at the primary load, never averaged across loads.
    # The loads carry deliberately different fact content, so a mean SUP
    # storage figure describes no corpus that was actually trained on and could
    # pass the burden gate while the primary load fails it.
    p_bits = {a: _mean(bits[a][primary_load]) for a in ARMS}
    p_clip = {a: _mean(clip[a][primary_load]) for a in ARMS}
    gates = check_gates(
        bits=p_bits,
        bits_lower_bound={"sup": seed_contrast(bits["sup"][primary_load])["ci_lower"]},
        igsm_acc_sup=_mean(acc_sup[primary_load]),
        clip_ratio=p_clip,
        storage_floor=storage_floor,
        igsm_band=igsm_band,
    )

    out = verdict(per_load[primary_load], gates, min_interesting_effect)
    out["endpoint"] = endpoint
    out["primary_contrast"] = "Y[factmask] - Y[randpos]"
    out["primary_load"] = primary_load
    out["inference_unit"] = (
        f"training seed within load '{primary_load}'; "
        f"n={len(per_load[primary_load])} seeds "
        f"{sorted(per_load_seeds[primary_load])}"
    )
    out["per_load"] = {
        k: {"effects": v, "seeds": sorted(per_load_seeds[k]),
            "recoverable_bits_per_param": {a: _mean(bits[a][k]) for a in ARMS},
            **seed_contrast(v)}
        for k, v in sorted(per_load.items())
    }
    out["dose_response"] = dose_response(per_load, bits["sup"],
                                         min_interesting_effect)
    out["recoverable_bits_per_param"] = p_bits
    out["clip_ratio"] = p_clip
    out["n_cells"] = len(cells)
    out["pooled_across_loads_NOT_PRIMARY"] = {
        "note": "reported for completeness only. Seeds recur at every load, so "
                "these observations are correlated and the interval is "
                "anti-conservative. Do not quote this as the result.",
        **seed_contrast([e for v in per_load.values() for e in v]),
    }
    out["estimand_note"] = (
        "Effect of masking fact-value targets rather than matched non-value "
        "targets. Capacity reallocation is an interpretation and is NOT "
        "identified by this design; gradient interference predicts the same "
        "sign. Discriminate on the shape across loads, not the direction."
    )
    return out


def dose_response(per_load: dict[str, list[float]],
                  sup_bits: dict[str, list[float]],
                  min_interesting_effect: float) -> dict:
    """The mechanism test: how the effect moves with fact load.

    THEORY-CAPACITY.md calls this the only analysis in the design that speaks
    to capacity rather than loss composition, and until now it lived in
    `theory/capacity.py` unreferenced by anything that runs. A flat, nonzero
    dose-response is the signature that matters most: an advantage that does
    not track fact load cannot be capacity reallocation, however significant
    it is, because a capacity story cannot be indifferent to load.

    Loads are ordered by measured SUP storage rather than by name, so the
    ordering is the physical one the theory is about.
    """
    if len(per_load) < 2:
        return {"signature": "single load: the shape test needs at least two",
                "n_loads": len(per_load)}
    by_bits = {}
    for load, effects in per_load.items():
        occupancy = sum(sup_bits[load]) / len(sup_bits[load])
        by_bits[occupancy] = sum(effects) / len(effects)
    return {
        "n_loads": len(per_load),
        "ordered_by": "measured SUP recoverable bits per parameter",
        "effect_by_sup_bits_per_param": {f"{k:.4f}": v
                                         for k, v in sorted(by_bits.items())},
        "signature": capacity.dose_response_signature(
            by_bits, floor=min_interesting_effect),
        "caveat": "three loads is the minimum that distinguishes a threshold "
                  "from a line; with two, 'rising' and 'saturating' are the "
                  "same observation.",
    }


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
    ap.add_argument("--primary-load", required=True,
                    help="the load the primary contrast is estimated at, "
                         "frozen in the preregistration before any run. "
                         "Inference is blocked by seed within it; the other "
                         "loads are the dose-response shape test.")
    args = ap.parse_args()

    try:
        out = analyse(
            Path(args.runs_root), args.endpoint, args.storage_floor,
            tuple(args.igsm_band), args.min_interesting_effect,
            set(args.expect_loads) if args.expect_loads else None,
            set(args.expect_seeds) if args.expect_seeds else None,
            primary_load=args.primary_load,
        )
    except IncompleteMatrix as e:
        print(f"REFUSING TO ANALYSE: {e}")
        print("Do not inspect a partial matrix and then complete it.")
        return 1

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("verdict", "primary_contrast", "primary_load",
                       "inference_unit", "endpoint", "statistic",
                       "dose_response", "failed_validity_gates",
                       "disclosed_reporting_failures")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
