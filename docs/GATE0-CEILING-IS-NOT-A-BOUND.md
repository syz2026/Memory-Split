# The uniform ceiling is not a bound on gate 0

Measured 2026-08-02 on `pilotA_d40m_std`. Artifact:
`outputs/pilot/gate0_diagnosis.json`. Reproduce with
`ops/crowding/diagnose_gate0.py --run <run>`.

## What was observed

`loss_masked_values` — cross-entropy at the offloaded positions — returned
**25.03 nats** on a *dense* arm that trained on those positions, against a
uniform ceiling of ln(50304) = 10.83.

| | |
|---|---:|
| positions scored | 32,517 |
| mean CE | 25.03 nats |
| median CE | 27.78 nats |
| fraction above the uniform ceiling | **98.7%** |
| mean model entropy | **3.83 nats** |
| argmax accuracy | **0.0001** |
| 5th–99th percentile CE | 16.49 – 30.81 |

## Why

The model is *sharp* — 3.83 nats of entropy is roughly 46 effective tokens out
of 50,304 — and essentially never correct. It is confidently wrong, not
ignorant.

The sample says what it is confident about. At positions holding
` Manufacturing`, ` Crane`, ` Springs` it predicts ` g`, ` the`, ` is`:
generic English continuations where a rare proper noun belongs. With the fact
lane at 3.75% of that corpus and 20 exposures per entity, it had not learned
even the slot type, let alone the filler.

The general principle:

> **A language model is never uniform.** It concentrates mass on frequent
> tokens, so at positions requiring rare tokens it necessarily scores *worse*
> than uniform. 1/50,304 is a generously high probability for a rare
> proper-noun subword compared to what a trained model actually assigns it.

Uniform is therefore neither a floor nor a ceiling for this metric. It is an
arbitrary point that a language model has no reason to sit near.

## What this invalidates

`docs/RETAINED-RESULTS.md` described gate 0 as the strongest result in the
project: split arms at 9.64–12.30 nats against ln(50304) = 10.83, read as
"pinned at the uniform ceiling, therefore the offloaded values are absent from
the weights."

That reading does not survive. A model that learned *nothing* at those
positions scores like the one measured here, 25+ nats. A split arm landing
near 10.8 had learned the **slot type** — a city goes here — and was uncertain
*within the pool*, which is log(200) ≈ 5.3 nats plus overhead. Its proximity to
ln(V) was a numerical coincidence that was read as confirmation.

The mechanism claim itself is not overturned: store-off recall of exactly 0%
and near-zero measured extraction are separate evidence that the values are not
retrievable. What is overturned is the *quantitative* gate-0 reading and the
claim that it is the strongest evidence.

## What replaces it

`evals/storage.py`, already built, scores against a pool-conditioned baseline
rather than against ln(V), and reports the unconditional and length-conditioned
versions separately so a conditional baseline is never mixed with an
unconditional ceiling. That was written to fix a specification error; this is
the empirical demonstration of why it was necessary.

Three consequences for reporting:

1. Never quote gate 0 against ln(V). Quote recoverable bits against the pool
   baseline, which is bounded and interpretable.
2. A gate-0 value *above* the uniform ceiling is not a corrupt measurement. It
   is the expected reading for a model that has not learned the content, and
   the diagnostic distinguishes it from a probe bug by entropy and argmax
   accuracy.
3. Cross-arm gate-0 comparisons remain valid — both arms score the same
   positions — but only as a *relative* quantity. The absolute number carries
   no interpretation against uniform.
