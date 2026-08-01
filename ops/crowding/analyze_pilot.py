"""Pilot analysis and the GO / NO-GO gate.

The gate asks four questions, in the order that makes the cheapest failure
happen first. Thresholds are frozen here and may not move once results are
seen -- that is the whole point of writing them before the runs.

1. Parameter-limited, not exposure-limited.
   Going from E to 3E at the operating entity count must buy under 20% more
   recoverable bits. This replaces the first draft's arbitrary bits/param
   threshold. If E is still on the steep part of the exposure curve, the
   ceiling is set by insufficient repetition rather than parameter scarcity,
   freeing parameters cannot help, and a null says nothing about capacity.
   Extrapolation is not permitted: measure at the operating point.

2. The endpoint is a construct, not a floor artifact.
   Accuracy inside the frozen band, AND monotone decreasing in op, AND
   monotone increasing over training. The last two are the signatures of
   dependency tracing and cost nothing extra.

3. The effect has room to exist.
   `delta` from Stage B -- the reasoning cost of carrying the fact load --
   must exceed the minimum interesting effect by at least 2x. Nothing can
   recover more than the facts cost in the first place.

4. The design is powered.
   The Stage C paired SD must give at least 80% power at the affordable n for
   that minimum interesting effect. If not, report the minimum detectable
   effect and accept that `inconclusive` is a likely outcome.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.stats import min_detectable_effect, seed_contrast  # noqa: E402

EXPOSURE_SATURATION_MAX_GAIN = 0.20
POWER_TARGET = 0.80
DELTA_HEADROOM = 2.0


def stage_a(cells: list[dict], min_lift: float = 0.05) -> dict:
    """Pick the architecture and difficulty that put the endpoint in a band.

    `cells` is one record per (model, mod, op_lo, op_hi) with `acc`,
    `majority_rate`, `acc_by_op` and `acc_over_training`.
    """
    scored = []
    for c in cells:
        lift = c["acc"] - c["majority_rate"]
        by_op = c.get("acc_by_op") or {}
        ops = sorted(by_op, key=lambda k: int(k))
        monotone_op = all(
            by_op[a] >= by_op[b] - 1e-9 for a, b in zip(ops, ops[1:])
        ) if len(ops) > 1 else None
        curve = c.get("acc_over_training") or []
        monotone_train = (curve[-1] >= curve[0]) if len(curve) > 1 else None
        scored.append({
            **c,
            "lift_over_majority": lift,
            "clears_floor": lift >= min_lift,
            "monotone_in_op": monotone_op,
            "monotone_over_training": monotone_train,
        })
    viable = [c for c in scored if c["clears_floor"]]
    best = max(viable, key=lambda c: c["lift_over_majority"]) if viable else None
    return {
        "cells": scored,
        "any_cell_clears_floor": bool(viable),
        "chosen": best,
        "note": (
            "Stop here if nothing clears the majority-class baseline by "
            f"{min_lift:.0%}. iGSM floored three times before, twice at models "
            "four times larger, and a floored endpoint has no power to "
            "discriminate anything."
        ),
    }


def stage_b(sup_acc: list[float], nofact_acc: list[float],
            bits_at_e: float, bits_at_3e: float) -> dict:
    """`delta` and the exposure-saturation check."""
    d = [n - s for n, s in zip(nofact_acc, sup_acc)]
    gain = (bits_at_3e - bits_at_e) / max(1e-12, bits_at_e)
    return {
        "delta": sum(d) / len(d) if d else 0.0,
        "delta_per_seed": d,
        "sup_acc_mean": sum(sup_acc) / len(sup_acc) if sup_acc else 0.0,
        "recoverable_bits_at_E": bits_at_e,
        "recoverable_bits_at_3E": bits_at_3e,
        "exposure_gain_from_3E": gain,
        "parameter_limited": gain < EXPOSURE_SATURATION_MAX_GAIN,
        "note": (
            "delta is the total reasoning cost of carrying the fact load and "
            "therefore the hard ceiling on any treatment effect. Report the "
            "effect as a fraction of it."
        ),
    }


def stage_c(effects: list[float], bits: dict[str, float],
            randpos_nll_gap: float, n_confirm: int) -> dict:
    stat = seed_contrast(effects)
    sd = stat["sd"]
    return {
        "paired_sd": sd,
        "per_seed_effects": effects,
        "mde_at_n": {
            n: min_detectable_effect(sd, n, power=POWER_TARGET)
            for n in (6, 8, 12, 16)
        },
        "leakage_factmask_over_sup": (
            bits.get("factmask", 0.0) / max(1e-12, bits.get("sup", 0.0))
        ),
        "randpos_nll_relative_gap": randpos_nll_gap,
        "mde_at_confirm_n": min_detectable_effect(sd, n_confirm,
                                                  power=POWER_TARGET),
    }


def gate(a: dict, b: dict, c: dict, min_interesting_effect: float,
         igsm_band: tuple[float, float], n_confirm: int) -> dict:
    checks = []

    checks.append({
        "name": "parameter_limited_not_exposure_limited",
        "ok": bool(b["parameter_limited"]),
        "observed": f"3E buys {b['exposure_gain_from_3E']:.1%} more bits",
        "required": f"< {EXPOSURE_SATURATION_MAX_GAIN:.0%}",
    })

    chosen = a.get("chosen")
    in_band = bool(chosen) and igsm_band[0] <= b["sup_acc_mean"] <= igsm_band[1]
    checks.append({
        "name": "endpoint_is_a_construct",
        "ok": in_band
        and chosen.get("monotone_in_op") is not False
        and chosen.get("monotone_over_training") is not False,
        "observed": {
            "acc": b["sup_acc_mean"],
            "monotone_in_op": chosen.get("monotone_in_op") if chosen else None,
            "monotone_over_training": (
                chosen.get("monotone_over_training") if chosen else None
            ),
        },
        "required": f"in {igsm_band}, decreasing in op, increasing over training",
    })

    checks.append({
        "name": "effect_has_room_to_exist",
        "ok": b["delta"] >= DELTA_HEADROOM * min_interesting_effect,
        "observed": f"delta = {b['delta']:.4f}",
        "required": f">= {DELTA_HEADROOM}x MIE ({DELTA_HEADROOM * min_interesting_effect:.4f})",
    })

    mde = c["mde_at_confirm_n"]
    checks.append({
        "name": "design_is_powered",
        "ok": mde == mde and mde <= min_interesting_effect,
        "observed": f"MDE at n={n_confirm} is {mde:.4f}",
        "required": f"<= MIE ({min_interesting_effect:.4f})",
    })

    go = all(x["ok"] for x in checks)
    return {
        "decision": "GO" if go else "NO-GO",
        "checks": checks,
        "min_interesting_effect": min_interesting_effect,
        "n_confirm": n_confirm,
        "on_no_go": (
            "Stop and publish the pilot. The exposure-to-storage frontier, the "
            "difficulty curve and delta are contributions independent of "
            "whether the hypothesis ever gets tested. The NO-GO paper's "
            "outline is in docs/NO-GO-PAPER.md and was written before Stage A "
            "ran, so the pilot was built to produce its figures."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-json", required=True,
                    help="measurements collected from the three stages")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-interesting-effect", type=float, required=True)
    ap.add_argument("--igsm-band", type=float, nargs=2, default=(0.17, 0.85))
    ap.add_argument("--n-confirm", type=int, default=8)
    args = ap.parse_args()

    m = json.loads(Path(args.pilot_json).read_text())
    a = stage_a(m["stage_a_cells"])
    b = stage_b(m["stage_b"]["sup_acc"], m["stage_b"]["nofact_acc"],
                m["stage_b"]["bits_at_E"], m["stage_b"]["bits_at_3E"])
    c = stage_c(m["stage_c"]["effects"], m["stage_c"]["bits"],
                m["stage_c"]["randpos_nll_gap"], args.n_confirm)
    g = gate(a, b, c, args.min_interesting_effect, tuple(args.igsm_band),
             args.n_confirm)

    out = {"stage_a": a, "stage_b": b, "stage_c": c, "gate": g}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(g, indent=2))
    return 0 if g["decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
