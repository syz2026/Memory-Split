"""Read the occupancy ladder: does storage saturate before capacity does?

Memory Split's first premise is that arbitrary facts occupy parameters a small
model would otherwise spend on reasoning. Nothing in this project has ever
tested that premise directly. Four experiment generations went straight to the
third premise -- that freeing those parameters buys reasoning -- and every one
of them was measuring noise around zero, because the corpora never stored
anything for masking to free.

This reads the premise off SUP-only runs at a fixed document count with rising
fact demand. No arms, one seed per rung, ~64 GPU-hours against the confirmatory
matrix's 1,495.

The three signatures, and what each licenses:

    saturating   bits/param rises then flattens as demand rises. Capacity binds.
                 The crowding regime has been reached and the arm contrast is
                 finally worth running.

    abandonment  bits/param rises then FALLS. The model declines to learn facts
                 rather than compressing them, so there is no burden to relieve
                 and crowding cannot occur at this scale however clean the
                 contrast. This is the shape the null draft claims; it has never
                 been measured on a corpus that passed its own gates.

    linear       bits/param tracks demand with no bend. Nothing binds; the model
                 is storing what it is shown and has room for more. Raise the
                 load, not the seed count.

    inert        bits/param near zero at every rung. Nothing was stored anywhere,
                 so the ladder says nothing about capacity -- it says the
                 exposure count is still below the storage threshold.

Occupancy is reported against TOTAL parameters, per preregistration §3: the
Allen-Zhu & Li 2 bits/param figure is a 1000-exposure result over total
parameters, and a non-embedding denominator against a 2-bit ceiling would
overstate occupancy. This matters here — at d40m the embedding table is 47.6% of
the model, and at d8m it is 81.2%, which is why d8m cannot carry a capacity
claim at all.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from theory import capacity as C  # noqa: E402

# A rung counts as inert below this many recoverable bits per parameter.
INERT = 0.02
# Fractional drop from the peak that counts as abandonment rather than a plateau.
ABANDON_FRAC = 0.20
# Fractional shortfall against a linear extrapolation that counts as a bend.
BEND_FRAC = 0.20


def read_rungs(runs_root: Path, prefix: str = "occ_") -> list[dict]:
    """One row per occupancy run, ordered by fact demand."""
    rows = []
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        if not d.name.startswith(prefix):
            continue
        summ = d / "evals" / "summary.json"
        cfg = d / "config.yaml"
        if not summ.exists():
            continue
        s = json.loads(summ.read_text())
        ents = int(s.get("n_entities") or 0)
        n_params = int(s.get("n_params") or 0)
        m = re.search(r"e(\d+)$", d.name)
        exposures = int(m.group(1)) if m else 0
        if not (ents and n_params and exposures):
            continue
        demand = ents * C_BITS_PER_ENTITY / n_params
        achievable = C.bits_per_param_at(exposures)
        rows.append({
            "run": d.name,
            "exposures": exposures,
            "entities": ents,
            "demand_bits_per_param": demand,
            "achievable_bits_per_param": achievable,
            "ratio_F_over_C": demand / achievable if achievable else None,
            "stored_bits_per_param": s.get("recoverable_bits_per_param"),
            "stored_ci95": s.get("recoverable_bits_per_param_ci95"),
            "stored_rel_se": s.get("recoverable_bits_rel_se"),
            "entities_probed": s.get("recoverable_bits_scaled_from"),
            "igsm_m1_nll": s.get("igsm_m1_nll"),
            "igsm_acc": s.get("igsm_acc"),
            "config_present": cfg.exists(),
        })
    rows.sort(key=lambda r: r["demand_bits_per_param"])
    return rows


# Imported lazily so the module can be read without corpusgen on the path.
try:
    from corpusgen.factlane import BITS_PER_ENTITY as C_BITS_PER_ENTITY
except Exception:                                            # pragma: no cover
    C_BITS_PER_ENTITY = 52.96


def classify(rows: list[dict]) -> dict:
    """Which of the four signatures the ladder shows."""
    usable = [r for r in rows if r["stored_bits_per_param"] is not None]
    if len(usable) < 2:
        return {"signature": "insufficient",
                "why": f"{len(usable)} scored rung(s); the ladder needs at "
                       "least two to have a shape"}

    stored = [r["stored_bits_per_param"] for r in usable]
    demand = [r["demand_bits_per_param"] for r in usable]
    peak = max(stored)

    if peak < INERT:
        return {
            "signature": "inert",
            "reading": (f"No rung stores more than {INERT} bits/param. Nothing "
                        "was written into the weights at any load, so this says "
                        "nothing about capacity — it says the exposure count is "
                        "still below the storage threshold. Raise exposures "
                        "before raising load."),
            "peak_stored_bits_per_param": peak,
        }

    last, top = stored[-1], max(stored)
    if last < top * (1 - ABANDON_FRAC):
        return {
            "signature": "abandonment",
            "reading": (f"Storage peaks at {top:.4f} bits/param and falls to "
                        f"{last:.4f} at the heaviest load. The model declines to "
                        "learn facts rather than compressing them, so there is "
                        "no burden for masking to relieve and crowding cannot "
                        "occur at this scale. Premise 1 of Memory Split fails, "
                        "and the arm contrast is not worth running."),
            "peak_stored_bits_per_param": top,
            "last_stored_bits_per_param": last,
        }

    # Linear would put the top rung at first-rung efficiency.
    eff0 = stored[0] / demand[0] if demand[0] else 0.0
    predicted = eff0 * demand[-1]
    bend = 1 - (last / predicted) if predicted else 0.0
    if bend > BEND_FRAC:
        return {
            "signature": "saturating",
            "reading": (f"Storage reaches {last:.4f} bits/param against the "
                        f"{predicted:.4f} a linear extrapolation predicts, a "
                        f"{bend:.0%} shortfall. Capacity is beginning to bind: "
                        "the crowding regime has been reached and the arm "
                        "contrast is finally worth running."),
            "bend_vs_linear": bend,
            "last_stored_bits_per_param": last,
            "linear_prediction": predicted,
        }
    return {
        "signature": "linear",
        "reading": (f"Storage tracks demand with no bend ({bend:+.0%} against "
                    "linear). Nothing binds — the model is storing what it is "
                    "shown and has room for more. Raise the load, not the seed "
                    "count."),
        "bend_vs_linear": bend,
    }


def report(runs_root: Path, prefix: str = "occ_") -> dict:
    rows = read_rungs(runs_root, prefix)
    out = {"rungs": rows, "n_rungs": len(rows)}
    out.update(classify(rows))
    out["denominator_note"] = (
        "Occupancy is against TOTAL parameters per preregistration §3. At "
        "d40m the embedding table is 47.6% of the model; at d8m it is 81.2%, "
        "which is why d8m cannot carry a capacity claim."
    )
    out["ladder_note"] = (
        "No rung above F/C = 1 fits the 150 GB scratch budget, so the ladder "
        "brackets the critical point from below only. A plateau at the top "
        "rung is evidence that capacity has begun to bind, not proof that it "
        "has bound hard."
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--prefix", default="occ_")
    ap.add_argument("--out")
    args = ap.parse_args()
    r = report(Path(args.runs), args.prefix)
    print(json.dumps(r, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(r, indent=2))
    return 0 if r.get("signature") == "saturating" else 2


if __name__ == "__main__":
    raise SystemExit(main())
