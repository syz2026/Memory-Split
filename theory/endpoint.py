"""Why is iGSM at floor, and what would move it?

Stage A returned STOP: 5.1-6.2% exact match against a 7.1% majority baseline.
The stated rationale was that the endpoint has no power to discriminate
anything. Before accepting that, the accuracy-by-operation-count profile is
worth reading, because the candidate explanations leave different fingerprints
in it.

  capacity        a mod-p operation table is a few kilobits. A 40M-parameter
                  model is not capacity-limited on it by any margin. Ruled
                  out arithmetically, see `table_bits`.
  expressivity    the eval is generative with max_new=384 and the prompt ends
                  at "Reasoning:", so the model may emit chain-of-thought and
                  needs only one operation per step. Ruled out by the eval
                  format.
  output format   deduction scores 55% with parseable output from the same
                  checkpoint, so "Answer:" is being produced. Ruled out.
  compounding     if each step is right with probability p, exact match on a
                  k-step chain is about p**k, so accuracy would be high at
                  op=1 and decay. Testable in the op profile.
  optimization    the operation is learnable but the run had too few steps.
                  Leaves a flat-at-chance profile INCLUDING op=1.

The measured profile is flat at chance at every op count including op=1, which
falsifies compounding and leaves optimization. A model that cannot do one
modular addition has not been crowded out of anything -- there is nothing there
to crowd.

The step budget makes that concrete. Stage A ran 800M tokens at 524,288 tokens
per step: **1,526 optimizer steps**, 300 of them warmup. Modular arithmetic is
a slow task for gradient descent; the grokking literature reports 1e4-1e6 steps
on far smaller models with a fixed table. 1,526 is roughly an order of
magnitude short before any of this project's specifics are considered.

The practical consequence is a warning about the probe now running. It varies
learning rate over 1,500 steps at MOD=23 -- the same budget and same difficulty
as the run whose floor it is meant to explain. A one-factor probe only
identifies its factor when the others are slack, so it will return floor at
every rate and invite the conclusion that the rate is exonerated and the
endpoint is dead. It can choose a learning rate by loss. It cannot speak to the
endpoint. `probe_is_confounded` states the condition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TOKENS_PER_STEP = 524_288

# Optimizer steps reported for learning modular arithmetic in the grokking
# literature (Power et al. 2022; Nanda et al. 2023). Used only as an order of
# magnitude -- those are fixed-table settings and this one streams fresh
# problems, which removes the memorisation phase but not the slowness.
MODULAR_STEPS_RANGE = (10_000, 1_000_000)


def steps_for(total_tokens: int, tokens_per_step: int = TOKENS_PER_STEP) -> int:
    return int(total_tokens // tokens_per_step)


def table_bits(mod: int, n_ops: int) -> float:
    """Bits to memorise the operation table outright.

    The upper bound on what the task could possibly demand of capacity: every
    (a, b, op) cell stored independently, with no structure exploited.
    """
    return mod * mod * n_ops * math.log2(mod)


def capacity_limited(mod: int, n_ops: int, n_params: int,
                     bits_per_param: float = 1.0) -> bool:
    return table_bits(mod, n_ops) > bits_per_param * n_params


@dataclass(frozen=True)
class OpProfile:
    """Accuracy by operation count, read against the no-skill baseline.

    The baseline is `max(chance, majority)`, not uniform chance. Always
    emitting the modal answer earns the majority rate for free, so uniform
    understates what a model achieves without learning anything, and cells a
    few points above 1/mod get mistaken for signal.

    Tolerance is two binomial standard errors when per-cell counts are given,
    and a flat default otherwise. That matters here: at n≈380 the standard
    error near 5% is about 1.1 points, so a fixed 3-point window is roughly
    2.7 sigma at one cell size and something quite different at another.
    """

    acc_by_op: dict[int, float]
    chance: float
    majority: float
    n_by_op: dict[int, int] | None = None

    @property
    def ops(self) -> list[int]:
        return sorted(self.acc_by_op)

    @property
    def at_op1(self) -> float | None:
        return self.acc_by_op.get(min(self.ops)) if self.ops else None

    @property
    def no_skill(self) -> float:
        return max(self.chance, self.majority)

    def _tol(self, op: int, default: float) -> float:
        n = (self.n_by_op or {}).get(op)
        if not n:
            return default
        p = self.no_skill
        return 2.0 * math.sqrt(max(p * (1 - p), 1e-12) / n)

    def near_baseline(self, op: int, default_tol: float = 0.03) -> bool:
        """Is this cell indistinguishable from guessing the modal answer?"""
        return self.acc_by_op[op] <= self.no_skill + self._tol(op, default_tol)

    @property
    def implied_per_step(self) -> dict[int, float]:
        """p such that p**op reproduces the measured exact match.

        Only meaningful if the chain is actually being executed; reported so a
        compounding profile can be read off directly.
        """
        return {op: (a ** (1.0 / op) if a > 0 else 0.0)
                for op, a in self.acc_by_op.items() if op > 0}

    def signature(self, tol: float = 0.03) -> str:
        if not self.ops:
            return "no data"
        vals = [self.acc_by_op[o] for o in self.ops]

        if all(self.near_baseline(o, tol) for o in self.ops):
            return (
                "flat at the no-skill baseline including op=1: the ATOMIC "
                "OPERATION is not learned. Not capacity, not depth, not "
                "compounding. Nothing was learned, so nothing could be crowded "
                "out, and more seeds will not help. Reduce the modulus or "
                "raise the step budget."
            )
        if self.near_baseline(self.ops[0], tol):
            return ("op=1 at baseline but longer chains above it: incoherent, "
                    "suspect leakage or a scoring artefact.")

        # Flatness first: a flat profile also satisfies the decay test, and
        # reading "compounding" off a chain that costs nothing would be wrong.
        if max(vals) - min(vals) < tol:
            return ("flat and above baseline at every op: suspicious. Chain "
                    "length should cost something; check for leakage.")
        if all(b <= a + tol for a, b in zip(vals, vals[1:])):
            return ("high at op=1 and decaying: COMPOUNDING. The chain is "
                    "being executed and errors multiply. Report per-step "
                    "accuracy, which is far more sensitive than exact match.")
        return "non-monotone above baseline: no clean reading"

    def report(self, tol: float = 0.03) -> dict:
        return {
            "acc_by_op": self.acc_by_op,
            "chance": round(self.chance, 4),
            "majority": round(self.majority, 4),
            "no_skill_baseline": round(self.no_skill, 4),
            "acc_at_op1": self.at_op1,
            "op1_near_baseline": self.near_baseline(self.ops[0], tol) if self.ops else None,
            "implied_per_step_accuracy": {k: round(v, 3)
                                          for k, v in self.implied_per_step.items()},
            "signature": self.signature(tol),
        }


def step_budget_verdict(steps: int, warmup: int,
                        needed=MODULAR_STEPS_RANGE) -> dict:
    lo, hi = needed
    return {
        "steps": steps,
        "warmup_steps": warmup,
        "warmup_fraction": round(warmup / max(1, steps), 4),
        "reference_range_for_modular_arithmetic": list(needed),
        "shortfall_vs_low_end": round(lo / max(1, steps), 1),
        "plausibly_sufficient": steps >= lo,
        "note": (
            f"{steps:,} steps is {lo / max(1, steps):.0f}x short of the low end "
            f"of the range reported for modular arithmetic."
            if steps < lo else
            f"{steps:,} steps is within the range reported for modular "
            "arithmetic."
        ),
    }


def probe_is_confounded(probe_steps: int, reference_steps: int,
                        probe_mod: int, reference_mod: int,
                        needed=MODULAR_STEPS_RANGE) -> dict:
    """Can a one-factor probe identify its factor?

    Only if the factors it holds fixed are slack. A learning-rate probe run at
    the same step budget and modulus as the floored run holds the suspected
    binding constraint fixed, so any null it returns is uninformative about the
    endpoint even though it remains valid for picking a rate by loss.
    """
    same_budget = abs(probe_steps - reference_steps) / max(1, reference_steps) < 0.25
    steps_binding = probe_steps < needed[0]
    confounded = same_budget and probe_mod == reference_mod and steps_binding
    return {
        "probe_steps": probe_steps,
        "reference_steps": reference_steps,
        "same_step_budget": same_budget,
        "same_modulus": probe_mod == reference_mod,
        "steps_plausibly_binding": steps_binding,
        "confounded": confounded,
        "reading": (
            "CONFOUNDED for the endpoint question. The probe holds step budget "
            "and modulus at the values suspected of causing the floor, so it "
            "will read floor at every level of its own factor. Still valid for "
            "choosing a learning rate by training loss."
            if confounded else
            "Identifiable: the probe varies its factor against slack elsewhere."
        ),
    }


def difficulty_ladder(mods=(23, 11, 7, 5), n_ops: int = 4) -> list[dict]:
    """The rungs Stage A was supposed to descend.

    `build_corpus.py` takes a single `--mod`, so a ladder needs one corpus per
    rung rather than meta tags inside one corpus. At 800M tokens a rung costs
    about 3.2 GB and a few minutes to build, which is cheap next to the
    conclusion it protects.
    """
    return [
        {
            "mod": m,
            "chance": round(1.0 / m, 4),
            "table_bits": round(table_bits(m, n_ops)),
            "clears_if_op1_above": round(1.0 / m + 0.10, 4),
        }
        for m in mods
    ]
