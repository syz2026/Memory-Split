# When can a crowding effect exist?

Derivations in `theory/capacity.py`, assertions in `tests/test_theory_capacity.py`.
Capacity figures follow Allen-Zhu & Li 2024, *Physics of Language Models: Part
3.3, Knowledge Capacity Scaling Laws*. The interpolation between their anchors
is ours; `ALPHA_ANCHORS` exposes it so every conclusion below can be re-derived
under a different curve.

## Three conditions, and the stage that tests each

An effect is observable only if all three hold. Naming them separately matters
because they fail for different reasons and only one of them is fixable with
more compute.

| condition | statement | tested by |
|---|---|---|
| crowding | the fact load binds against capacity, F/C ≈ 1 | Stage B operating point |
| responsiveness | the endpoint is off floor **and** off ceiling in both arms | Stage A |
| power | the expected effect exceeds the minimum detectable one | Stage C paired SD |

Only power is bought with seeds. A responsiveness failure — the endpoint at its
majority baseline in both arms — cannot be repaired by running longer, because
nothing was learned and therefore nothing was crowded out. That is the
theoretical content of Stage A's STOP, and it is why Stage A gates the
experiment instead of merely informing it.

## The corpus in flight is 2× over-saturated

The build uses **1,531,800 entities at 100 exposures**. At 52.96 bits per
entity that is 81.1 Mbit of fact content against d40m's 40.56 M parameters:

**exactly 2.0001 bits per parameter of demand.**

That is not a coincidence — it targets the Allen-Zhu & Li saturation figure.
But 2 bits/param is their **1000-exposure** result. At 100 exposures achievable
capacity is roughly half:

| dose | entities | exposures | demand | achievable | **F/C** | regime |
|---|---:|---:|---:|---:|---:|---|
| f = 1.00 | 1,531,800 | 100 | 2.00 | 1.00 | **2.00** | over |
| f = 0.50 | 765,900 | 200 | 1.00 | 1.30 | **0.77** | critical |
| f = 0.25 | 382,950 | 400 | 0.50 | 1.60 | **0.31** | under |

So the headline dose sits at twice capacity, not at it. This is not a defect —
the ladder spans F/C from 0.31 to 2.00 and therefore brackets the critical
point, which is what makes the shape prediction below testable. It does mean
the top dose should not be expected to be the strongest arm.

Note that both terms of F/C move along the ladder. Holding document count fixed
forces exposures = docs/entities, so fewer entities means less to store *and*
more exposures of each, raising achievable capacity too. F/C therefore falls
faster than the entity count. That is inherent: unique entropy cannot be varied
at fixed tokens without varying exposure.

## The prediction is a shape, not a contrast

The dose-response of split-minus-dense reasoning advantage should **rise with
load and then saturate**, with the elbow near F/C = 1.

Under-load, the dense arm stores every fact in a slice of capacity, and freeing
that slice returns only F. At and above capacity the dense arm is
capacity-limited and spends all of C regardless, so the amount freed stops
growing.

An inverted U *within* this design's range would be the wrong prediction. At
twice capacity the model can still fit half the facts, so it has every reason
to keep spending. A decline requires far heavier load, where the values become
effectively unlearnable and the cheapest way to reduce loss is to model their
marginal distribution rather than the conditional one.

That abandonment regime is not hypothetical.
`outputs/pilot/gate0_diagnosis.json` caught a model doing exactly it —
placing mass on generic English continuations rather than on values. It
arrived there by under-exposure rather than over-load, so it shows the
behaviour exists, not that it occurs at F/C = 2.

### What each shape would mean

| measured shape | reading |
|---|---|
| flat at zero | no effect |
| **flat and nonzero** | **confound, not reallocation** |
| rising then saturating | capacity reallocation |
| rising then falling | reallocation plus abandonment at the top dose |

The second row is the one worth the extra doses. If masking helps by the same
amount regardless of how many facts there are, the gain cannot be capacity
reallocation, because a capacity story cannot be indifferent to load. It points
instead at the mask acting as a regulariser, or at leakage. A single-dose
contrast cannot distinguish that case from the real mechanism no matter how
significant it is — which is the argument for spending the compute on a ladder
rather than on more seeds at one dose.

`dose_response_signature()` performs this classification.

## Two consequences for the design

**The ladder would be better centred.** Solving for target ratios at fixed
document count gives doses at F/C = 0.5, 1.0, 2.0:

| target F/C | entities | exposures |
|---:|---:|---:|
| 0.5 | 552,370 | 277 |
| 1.0 | 930,402 | 164 |
| 2.0 | 1,531,723 | 100 |

The top rung is the corpus already building — 1,531,723 against the actual
1,531,800. Nothing in flight is wasted. Re-centring changes only the two lower
rungs, from 765,900/382,950 to 930,402/552,370, and buys a dose sitting on the
elbow instead of merely straddling it.

**Three loads do not fit scratch.** At uint16 targets plus two uint8 sidecars,
one 16.4B-token load costs 4 bytes per token:

- 65.6 GB per load
- 196.8 GB for three, against a ~150 GB budget
- at most **two** loads resident at once

So the ladder has to be built, trained and deleted one dose at a time, with the
checkpoint and `summary.json` retained and the corpus discarded. Worth
scheduling around: the build is ~2 h and cannot overlap the third dose's
training.

## What the theory does not give

The **magnitude**. The argument above fixes the sign and the shape, not the
effect size, and no amount of bits-per-parameter accounting will produce one,
because the map from freed capacity to reasoning accuracy is unknown.

That is what Stage B's NOFACT arm is for: it measures the gain from removing
*all* fact burden, which upper-bounds the gain from removing some.
`effect_is_bounded_by_nofact()` asserts it. A split arm that beats its own
NOFACT ceiling is reporting something other than capacity reallocation.
