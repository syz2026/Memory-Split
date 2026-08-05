"""When can a capacity-crowding effect exist at all?

The hypothesis is that offloading facts frees parameters that get reallocated
to reasoning. That can only produce a measurable effect under three conditions,
and each one can be checked before spending GPU-hours.

  crowding        the fact load must actually bind against model capacity
  responsiveness  the endpoint must be off floor and off ceiling in BOTH arms
  power           the expected effect must exceed the minimum detectable one

The staged pilot maps onto them exactly: Stage A tests responsiveness, Stage B
locates the operating point and bounds the magnitude with a NOFACT arm, Stage C
measures the paired SD that sets power.

The central quantitative claim here is about the **shape of the dose-response
curve**, which is a sharper and more falsifiable prediction than any single
dense-versus-split comparison.

  under-load   F << C   dense stores every fact using a small slice of
                        capacity. Freeing that slice buys little, in
                        proportion to F.
  critical     F ~= C   dense is capacity-limited. Freeing the facts returns
                        the whole of C, which is the most available.
  over-load    F >  C   dense is still capacity-limited and still spends all
                        of C on the facts it can fit, so the amount freed
                        stops growing. The effect PLATEAUS rather than rising.
  abandonment  F >>> C  the facts become effectively unlearnable, and the
                        cheapest way to reduce loss is to model the marginal
                        distribution of values instead of the conditional
                        one. Capacity spent on facts falls, and so does the
                        effect.

So the prediction is monotone-increasing-then-saturating, with the plateau
beginning near F/C = 1, and an eventual decline only at loads far past the
design's range. Claiming an inverted U inside F/C <= 2 would be wrong: at twice
capacity the model can still fit half the facts, so it has every reason to
keep spending.

The abandonment regime is not hypothetical. `outputs/pilot/gate0_diagnosis.json`
shows a model at unlearnable fact positions putting its mass on generic English
continuations rather than on values -- the marginal, not the conditional. That
run reached the state by under-exposure rather than over-load, so it evidences
that the behaviour exists, not that it occurs at F/C = 2.

The shape is what discriminates the mechanism from its rivals:

  flat at zero            no effect
  flat and NONZERO        a confound, not reallocation. If masking helps
                          regardless of how many facts there are, the mask is
                          acting as a regulariser or leaking, because a
                          capacity story cannot be indifferent to load.
  rising then saturating  capacity reallocation
  rising then falling     reallocation plus abandonment at the top dose

Capacity figures follow Allen-Zhu & Li 2024, "Physics of Language Models: Part
3.3, Knowledge Capacity Scaling Laws": GPT-2 class models reach 2 bits per
parameter at 1000 exposures, and roughly half that at 100. Their denominator is
TOTAL parameters, so a non-embedding denominator is not comparable and is not
used here. The interpolation between their two anchors is ours, not theirs, and
`ALPHA_ANCHORS` exposes it so the conclusions can be re-derived under a
different curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Allen-Zhu & Li anchors: (exposures, bits per total parameter).
ALPHA_ANCHORS: tuple[tuple[int, float], ...] = ((100, 1.0), (1000, 2.0))

# F/C within this band of 1.0 counts as critical load.
CRITICAL_BAND = (0.5, 1.5)


def bits_per_param_at(exposures: int, anchors=ALPHA_ANCHORS) -> float:
    """Achievable bits per parameter at a given exposure count.

    Log-linear between the anchors, clamped outside them. The clamp at the top
    matters: capacity does not grow without bound in exposures, so extrapolating
    past 1000 would silently manufacture headroom the model does not have.
    """
    (e_lo, a_lo), (e_hi, a_hi) = anchors
    if exposures <= 0:
        return 0.0
    slope = (a_hi - a_lo) / (math.log10(e_hi) - math.log10(e_lo))
    a = a_lo + slope * (math.log10(exposures) - math.log10(e_lo))
    return max(0.0, min(a_hi, a))


@dataclass(frozen=True)
class Load:
    """One point on the dose ladder."""

    fraction: float
    n_entities: int
    exposures: int
    bits_per_entity: float
    n_params: int

    @property
    def fact_bits(self) -> float:
        return self.n_entities * self.bits_per_entity

    @property
    def capacity_bits(self) -> float:
        return bits_per_param_at(self.exposures) * self.n_params

    @property
    def ratio(self) -> float:
        """F/C. The order parameter of the whole design."""
        c = self.capacity_bits
        return float("inf") if c == 0 else self.fact_bits / c

    @property
    def demand_bits_per_param(self) -> float:
        """What the corpus asks of the model, ignoring whether it can comply."""
        return self.fact_bits / self.n_params

    @property
    def regime(self) -> str:
        lo, hi = CRITICAL_BAND
        r = self.ratio
        return "under" if r < lo else ("critical" if r <= hi else "over")

    def report(self) -> dict:
        return {
            "fraction": self.fraction,
            "n_entities": self.n_entities,
            "exposures": self.exposures,
            "fact_bits": round(self.fact_bits),
            "demand_bits_per_param": round(self.demand_bits_per_param, 3),
            "achievable_bits_per_param": round(bits_per_param_at(self.exposures), 3),
            "capacity_bits": round(self.capacity_bits),
            "ratio_F_over_C": round(self.ratio, 3),
            "regime": self.regime,
        }


def ladder(n_entities: int, exposures: int, fractions, bits_per_entity: float,
           n_params: int) -> list[Load]:
    """The dose ladder as `factlane.load_variants` builds it.

    A load at fraction f is f*N entities at E/f exposures, so document count and
    therefore token count, step count and mask mass are identical across doses.
    Only unique entropy moves -- which is the point, but note it also moves the
    *achievable* capacity, because capacity depends on exposure count. Both
    terms of F/C therefore change together along the ladder.
    """
    out = []
    for f in fractions:
        out.append(Load(
            fraction=f,
            n_entities=int(round(n_entities * f)),
            exposures=int(round(exposures / f)),
            bits_per_entity=bits_per_entity,
            n_params=n_params,
        ))
    return out


def entities_for_ratio(ratio: float, exposures: int, bits_per_entity: float,
                       n_params: int) -> int:
    """Entity count that puts the load at a chosen F/C, at fixed exposures."""
    return max(1, int(round(ratio * bits_per_param_at(exposures) * n_params
                            / bits_per_entity)))


def entities_for_ratio_at_fixed_docs(ratio: float, n_docs: int,
                                     bits_per_entity: float, n_params: int,
                                     lo: int = 1_000, hi: int | None = None) -> int:
    """Entity count hitting a chosen F/C while holding document count fixed.

    The ladder trades entities against exposures to keep tokens, steps and mask
    mass constant, so exposures = n_docs / n. That makes both sides of F/C move
    at once: fewer entities means less to store AND more exposures of each,
    which raises achievable capacity too. F/C therefore falls faster than the
    entity count, and no closed form survives the log. Bisect instead.

    Monotonicity holds because F rises with n while C falls with n (exposures
    drop), so the ratio is strictly increasing and bisection is safe.
    """
    hi = hi or max(lo + 1, n_docs // 2)

    def r(n: int) -> float:
        e = max(1, n_docs // n)
        c = bits_per_param_at(e) * n_params
        return float("inf") if c == 0 else (n * bits_per_entity) / c

    if r(lo) > ratio:
        return lo
    if r(hi) < ratio:
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if r(mid) < ratio:
            lo = mid
        else:
            hi = mid
    return hi


def bytes_on_disk(total_tokens: int, n_sidecars: int = 2) -> int:
    """Scratch cost of one built load.

    uint16 targets plus one uint8 byte per sidecar per token. Three loads of a
    16.4B-token corpus is ~197 GB, which does not fit a 150 GB scratch budget,
    so the ladder has to be built, trained and deleted one dose at a time.
    """
    return total_tokens * (2 + n_sidecars)


@dataclass
class Audit:
    loads: list[Load]
    notes: list[str] = field(default_factory=list)

    @property
    def brackets_critical(self) -> bool:
        """Does the ladder span F/C = 1?

        If every dose sits on one side, the inverted-U prediction is untestable
        and the design can only ever report a slope, not a peak.
        """
        rs = [l.ratio for l in self.loads]
        return min(rs) <= 1.0 <= max(rs)

    @property
    def plateau_onset(self) -> Load:
        """The dose nearest F/C = 1, where the effect should stop growing.

        Not the dose with the largest predicted effect -- every dose at or above
        it shares the plateau. It is the elbow, and locating an elbow needs
        doses on both sides of it.
        """
        return min(self.loads, key=lambda l: abs(math.log(max(l.ratio, 1e-9))))

    @property
    def has_dose_below_and_above(self) -> bool:
        """An elbow is only identifiable with doses on both sides."""
        return (any(l.ratio < 1.0 for l in self.loads)
                and any(l.ratio > 1.0 for l in self.loads))

    def report(self) -> dict:
        return {
            "loads": [l.report() for l in self.loads],
            "brackets_critical_point": self.brackets_critical,
            "elbow_identifiable": self.has_dose_below_and_above,
            "plateau_onset_fraction": self.plateau_onset.fraction,
            "plateau_onset_ratio": round(self.plateau_onset.ratio, 3),
            "prediction": (
                "Split-minus-dense reasoning advantage rises with fact load "
                "and saturates near F/C = 1, the elbow falling closest to "
                f"fraction {self.plateau_onset.fraction} "
                f"(F/C = {self.plateau_onset.ratio:.2f}). A dose-response that "
                "is flat but nonzero indicates a confound rather than "
                "reallocation: a capacity mechanism cannot be indifferent to "
                "how many facts there are."
            ),
            "notes": self.notes,
        }


def audit(n_entities: int, exposures: int, fractions, bits_per_entity: float,
          n_params: int) -> Audit:
    loads = ladder(n_entities, exposures, fractions, bits_per_entity, n_params)
    a = Audit(loads)
    if not a.brackets_critical:
        side = "below" if max(l.ratio for l in loads) < 1 else "above"
        a.notes.append(
            f"Ladder sits entirely {side} F/C = 1, so the peak is outside the "
            "design and only a slope is testable. Re-centre the entity count "
            "with entities_for_ratio()."
        )
    over = [l for l in loads if l.regime == "over"]
    if over:
        a.notes.append(
            f"{len(over)} dose(s) in the over-load regime (F/C > "
            f"{CRITICAL_BAND[1]}). Expect these to sit on the plateau, not to "
            "be proportionally stronger: past capacity, raising the load stops "
            "increasing the amount that offloading can free."
        )
    if not a.has_dose_below_and_above:
        a.notes.append(
            "No dose on one side of F/C = 1, so the elbow is not identifiable "
            "and only a slope can be reported."
        )
    return a


def dose_response_signature(effects_by_ratio: dict[float, float],
                            tol: float = 0.2, floor: float = 0.0) -> str:
    """Classify a measured dose-response into the competing explanations.

    `effects_by_ratio` maps F/C to the split-minus-dense advantage. `tol` is the
    fraction of the largest effect below which variation counts as flat.
    `floor` is the absolute size below which an effect counts as zero, and in
    the confirmatory analysis it is the preregistered minimum interesting
    effect.

    `floor` is not cosmetic. "Flat at zero" is a statement about absolute
    magnitude and needs an external scale: the earlier form of this function
    tested `peak < tol * peak`, which is true only for negative `peak` and so
    could never fire. Every negligible dose-response was therefore classified
    as though its shape were meaningful, and a set of effects three orders of
    magnitude below the noise floor could still be reported as "rising then
    saturating: consistent with capacity reallocation".

    The nonzero-flat case is the one worth the trouble: it is the signature of
    the mask helping for reasons that have nothing to do with capacity.
    """
    if not effects_by_ratio:
        return "no data"
    ratios = sorted(effects_by_ratio)
    vals = [effects_by_ratio[r] for r in ratios]
    peak = max(abs(v) for v in vals)
    spread = max(vals) - min(vals)

    if peak <= floor:
        return ("flat at zero: no effect at any load exceeds the minimum "
                "interesting effect")
    if spread < tol * peak:
        return ("flat and nonzero: CONFOUND. The advantage does not track fact "
                "load, so it is not capacity reallocation. Suspect the mask as "
                "regulariser, or leakage.")
    rising = all(b >= a - tol * peak for a, b in zip(vals, vals[1:]))
    if rising:
        return "rising then saturating: consistent with capacity reallocation"
    if vals[-1] < max(vals) - tol * peak:
        return ("rising then falling: reallocation with abandonment at the "
                "top dose")
    return "non-monotone: no clean reading"


# ---------------------------------------------------------------- observability

def responsive(accuracy: float, floor: float, ceiling: float = 0.95,
               margin: float = 0.02) -> bool:
    """Is an endpoint in a regime where a capacity effect could show?

    An endpoint at floor in both arms cannot exhibit crowding: nothing has been
    learned, so nothing was crowded out. This is not a power problem and more
    seeds will not fix it. It is why Stage A gates the experiment rather than
    merely informing it.
    """
    return (accuracy > floor + margin) and (accuracy < ceiling - margin)


def effect_is_bounded_by_nofact(split_gain: float, nofact_gain: float,
                                tol: float = 1e-9) -> bool:
    """Removing SOME fact burden cannot beat removing ALL of it.

    A split arm that outperforms its own NOFACT ceiling is reporting something
    other than capacity reallocation -- most likely the mask acting as a
    regulariser, or a leak. Stage B measures the ceiling for exactly this check.
    """
    return split_gain <= nofact_gain + tol


def observability(endpoint_accuracy: float, floor: float, ratio: float,
                  expected_effect: float, mde: float) -> dict:
    """All three gates in one place, with the binding constraint named."""
    lo, hi = CRITICAL_BAND
    checks = {
        "crowding": lo <= ratio <= hi,
        "responsiveness": responsive(endpoint_accuracy, floor),
        "power": expected_effect > mde,
    }
    failed = [k for k, v in checks.items() if not v]
    return {
        "checks": checks,
        "observable": not failed,
        "binding_constraint": failed[0] if failed else None,
        "reason": {
            "crowding": f"F/C = {ratio:.2f} outside [{lo}, {hi}]; the load does "
                        "not bind, so there is little to free.",
            "responsiveness": f"endpoint at {endpoint_accuracy:.3f} against a "
                              f"floor of {floor:.3f}; nothing was learned, so "
                              "nothing could be crowded out. More seeds will "
                              "not fix this.",
            "power": f"expected effect {expected_effect:.4f} below the minimum "
                     f"detectable {mde:.4f}; the design cannot resolve it.",
        }.get(failed[0]) if failed else None,
    }
