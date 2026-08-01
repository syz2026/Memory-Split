# The NO-GO paper, outlined before the pilot runs

Written 2026-08-01, before Stage A, on purpose. Two of three independent
reviewers volunteered unprompted that NO-GO at the endpoint criterion is the
single most likely outcome of this programme: iGSM has floored three times,
twice at models four times larger than d40m.

A NO-GO paper written afterwards is a consolation prize and reads like one. A
NO-GO paper outlined first determines what the pilot measures, which is why
Stage B carries a NOFACT run that the first draft of the plan did not have.

**If the pilot returns NO-GO, this is the deliverable.** Do not run the
matrix, and do not weaken a gate to avoid writing it.

---

## Title

*Crowding requires saturation: measuring the precondition for capacity
reallocation in small language models*

## The claim

The belief that factual memorisation competes with reasoning for parameters —
the rationale phi-3 gives for filtering fact-heavy pages — has a precondition
that is measurable and, at accessible scale, unmet. We measure the
exposure-to-storage frontier for a small transformer, quantify the reasoning
cost of a fact load directly, and show what it would take to reach the regime
where the belief could be tested at all.

This is a negative result about a testability precondition, not a claim that
the conjecture is false.

## Figures, and the stage that produces each

**Figure 1 — the exposure-to-storage frontier.** Recoverable bits per
parameter against exposures per fact, at the operating entity count.
*Stage B.* This is the reusable artifact: a partial replication of Allen-Zhu &
Li at a scale others can actually run, and the curve that says whether a
budget is parameter-limited or exposure-limited.

**Figure 2 — the difficulty curve.** iGSM accuracy against MOD and op band,
crossed with architecture, against the empirical majority-class baseline.
*Stage A.* Establishes where the endpoint is a construct rather than a floor
artifact, and includes the correction that the baseline is 7.17% and not 1/23.

**Figure 3 — delta: what the facts actually cost.** Held-out reasoning
accuracy with the fact lane present against the same corpus with it replaced
by bed tokens at equal token count and equal steps. *Stage B.* If this is
small, the treatment effect is bounded small no matter how clean the
execution — which is the quantitative form of the NO-GO.

**Figure 4 — the reachable frontier.** Hours per run to reach a target
occupancy, by model size, against the 47-hour job wall. Shows the scale window
directly: the models large enough to reason cannot be saturated at accessible
budgets, and the ones that saturate cheaply are too small to reason.

**Table 1 — the loss-normalization correction.** The multiplier `1/(1-f)`
applied to surviving targets under mean reduction, at the mask rates used by
this and comparable work.

## Contributions, in the order they survive review

1. **The loss-normalization bug.** `F.cross_entropy(..., ignore_index=-100)`
   with the default `reduction='mean'` averages over surviving targets, so
   masking a fraction `f` multiplies every remaining target's weight by
   `1/(1-f)` — 1.331 at the 24.89% rate measured here. Any masked arm trained
   that way is a different objective from its dense twin over and above the
   masking. This is established independent of any run, it affects anyone
   doing selective-loss training, span-masked LM or LMLM-style offloading, and
   it is the strongest standalone item in the programme.

2. **The exposure-to-storage frontier**, with a measurement that separates
   baseline from ceiling correctly: an unconditional pool baseline gives a
   52.96 bit/entity ceiling, a length-conditioned one gives 45.30, and mixing
   them overstates recovery by the 7.66 bits that token length leaks.

3. **`delta` as a bound.** Measuring the reasoning cost of a fact load
   directly, and using it to bound the achievable treatment effect before
   spending the compute. Generalises: any masking intervention can be bounded
   this way for the price of one extra run.

4. **The scale window.** Quantified, not asserted.

5. **A withdrawal.** The numbers this project previously reported, why they
   have no artifacts, and what replaced them. `docs/RETAINED-RESULTS.md` is
   the audit.

## Venue

A workshop that explicitly wants negative results, methodology, or
science-of-deep-learning work. In a general workshop this reads as a failed
experiment; in the right one the bug and the frontier are the contribution and
the NO-GO is context.

## What this paper must not say

- That crowding does not occur. It says the precondition was not reached.
- Anything about "small language models" generally, or about phi-3.
- That the earlier reported effects were refuted. They were withdrawn for lack
  of artifacts, which is a different and less interesting claim.
