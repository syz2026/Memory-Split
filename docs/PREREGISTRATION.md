# Preregistration

**Status: FROZEN pending the pilot.** Sections marked `[PILOT]` carry a value
that the staged pilot measures; every other value is fixed now. Once the pilot
completes, the `[PILOT]` slots are filled, this document is committed and
externally timestamped, and nothing in it may change on the basis of an
outcome.

Written 2026-08-01, before any run of this design.

---

## 1. The question, and what it is not

**Primary estimand.**

```
effect_i = Y[FACTMASK, i] - Y[RANDPOS, i]
```

the effect of removing direct next-token loss from fact-value targets rather
than from an equal, matched mass of non-value targets, on closed-book iGSM
accuracy, in a 40.6M-parameter tied-embedding decoder with 21.2M non-embedding
parameters, trained at roughly 141 tokens per parameter on one controlled
corpus.

Written reduced deliberately. `(FACTMASK − SUP) − (RANDPOS − SUP)` cancels the
SUP terms exactly, so a double-difference notation would imply an
identification property this design does not have. SUP is in the matrix as the
manipulation check and the load main effect, not as part of the estimate.

**What this does not identify.** Capacity reallocation. Masking removes
competing gradients whether or not any parameter was ever occupied, so
gradient interference predicts the same sign and the same monotone dose trend.
No reviewer of this design believed otherwise and neither do we. Capacity
language in the write-up is interpretive and must be labelled as such.

The two things that bear on it are consistency evidence, not identification:
the **shape** of the effect across fact loads (crowding predicts a threshold
near the ceiling, interference predicts a line all the way down) and the
gradient-mass decomposition in §8.

**Out of scope.** Any claim about "small language models" generally, about
phi-3, which is 100x larger, or about LMLM as a deployed retrieval system.
There is no external store at training or evaluation time in this design.

## 2. Arms

One byte-identical `uint16` stream per fact load. Three `uint8` target-weight
sidecars over it:

| Arm | Sidecar | Role |
|---|---|---|
| `SUP` | none (loader defaults to full supervision) | manipulation check, load main effect |
| `FACTMASK` | zero on fact-value targets | treatment |
| `RANDPOS` | zero on matched non-value spans | control |

Within a seed the three configs are byte-identical except `train_mask`, which
`ops/crowding/gen_configs.py` asserts. All three carry `probe_mask =
factmask.bin` so gate 0 scores the same positions in every arm.

RANDPOS matches FACTMASK on count, span length, within-document relative
position, and — using a token-NLL table frozen from a pilot run on disjoint
data — mean per-token difficulty. **If mean masked-span NLL cannot be matched
within 20% relative, the control is empirically invalid and that is reported
as the result, not gated away.**

## 3. Fact load

Load varies by trading entities against exposures at a fixed document count: a
load at fraction `f` is `f·N` entities at `E/f` exposures, snapped to an exact
divisor of `N·E`. Document count, total tokens, optimizer steps, cosine
schedule length, mask mass and mask positions are therefore identical across
loads, and the only thing that moves is the unique-entropy ceiling.

This also decouples load from exposures per fact, which is the confound that
invalidated the earlier dose sweep, where raising the entity count at a fixed
lane budget cut exposures in the same step.

- Loads: `[PILOT]` — two minimum, three preferred. Three is what distinguishes
  a threshold from a line, and that shape test is the only thing in the design
  that speaks to capacity rather than loss composition.
- `N`, `E`: `[PILOT]`
- Occupancy is reported against **total** parameters. The Allen-Zhu & Li
  2 bits/param figure is a 1000-exposure result over total parameters; at
  ~100 exposures their ratio is nearer 1. Do not headline a non-embedding
  denominator against a 2-bit ceiling.

## 4. Corpus

`52.96` bits/entity analytic from the pool sizes; `74.91` tokens/document and
`24.89%` value tokens measured on the shipped generator. Lanes interleave by
running-share deficit and are never blocked — the previous corpus put its
reasoning lane in the last 12.8% of training, which made the endpoint
threshold noise.

- Shares: fact 50%, iGSM 30%, deduction 10%, natural bed 10%
- Bed: a pinned FineWeb-Edu JSONL with a recorded hash. A synthetic bed is
  rehearsal only and the manifest says so.
- Corpus hashes: `[PILOT]`

## 5. Endpoint

**Primary: closed-book iGSM accuracy**, held out by text hash, reported
per `op`.

Scored against the **empirical majority-class baseline, not 1/23**. Measured:
7.17% at op 1–4 under MOD 23, because `times` overproduces zero. Every prior
iGSM number in this project was scored against 4.35% and is misreported.

- `MOD`: 23 primary. A reduction is a declared fallback, triggered only by
  Stage A showing no cell clears majority + 5 percentage points at MOD 23. If
  adopted, the construct is relabelled "dependency tracing under easy modular
  arithmetic" with no continuity claim to Physics-of-LM numbers.
- `op` band: `[PILOT]`
- Architecture, `d40m` vs `d40m_std`: `[PILOT]`. Recurrence buys effective
  depth 24 against 12; dropping it makes bits-per-parameter comparable to
  standard-GPT-2 literature but may cost the endpoint. Stage A decides.
- Generation: greedy, temperature 0, `max_new_tokens = 384`, stop at EOT.
- OOD band: op 5–8, reported separately, topology-hash disjoint from training.

**Secondary: deduction, reported per answer class, never as a single number.**
The eval is exactly 200 yes / 200 no in strict alternation and the NO branch's
trace is a canned sentence, so a constant "no" scores exactly 0.500.

## 6. Inference

- Unit: the training seed. Item bootstraps quantify evaluation noise only.
- `n`: `[PILOT]`, floor 6, target 8. Set from the Stage C paired SD, not from
  a round number.
- One-sided, α = 0.05, Student's t with df = n−1. The exact sign test is
  reported alongside with its floor: at n=3 the smallest attainable one-sided
  p is 0.125.
- Seeds vary both initialization and shard permutation, so inference
  generalizes over corpus realizations rather than being conditional on one.
- Minimum interesting effect: `[PILOT]`, derived from `delta` (§7), never
  chosen as a round number.
- Pilot seeds (9001–9003) are burned and may not appear in the matrix.

**Verdict.** `validated` if the one-sided lower bound exceeds the minimum
interesting effect. `rejected` if the upper bound falls below it — a practical
null, not "no effect of any size". `inconclusive` otherwise. `invalid` if any
validity gate fails.

## 7. Gates

**Validity — failure means the experiment did not happen.**

1. SUP recoverable bits/param, 95% lower bound, above the pilot-measured floor.
2. FACTMASK recoverable bits below 10% of SUP. Values remain in the input
   context and are masked only in the loss, so leakage through the input
   pathway is possible; this is the descendant of the 94.2%-recoverability
   failure that made the previous corpus inert.
3. SUP iGSM accuracy inside the frozen band, monotone decreasing in `op`, and
   monotone increasing over training.

**Reporting — disclosed, not fatal, with the fallback prespecified.**

4. RANDPOS recoverable bits within 20% of SUP.
5. Per-arm mean clip ratio within a frozen absolute band. `grad_clip` is
   raised in the pilot until clipping stops binding for the dense arm, because
   removing the mechanism beats monitoring it.
6. Per-arm mean unmasked-token NLL, so the entropy asymmetry between the arms
   is visible rather than inferred.

All gates use confidence bounds, not point estimates.

## 8. Pilot GO / NO-GO

Evaluated before any confirmatory run; thresholds frozen here.

1. **Parameter-limited, not exposure-limited.** E → 3E buys under 20% more
   recoverable bits at the operating entity count. Measured, not extrapolated.
   If E is still on the steep part of the curve the ceiling is repetition, not
   parameters, and a null says nothing about capacity.
2. **Endpoint is a construct**, per §5 and §7.3.
3. **The effect has room to exist.** `delta`, the reasoning cost of carrying
   the fact load measured by the Stage B NOFACT run, exceeds the minimum
   interesting effect by at least 2x. Nothing can recover more than the facts
   cost.
4. **The design is powered.** Stage C paired SD gives ≥80% power at the
   affordable `n`. If not, the minimum detectable effect is reported and
   `inconclusive` is accepted as a likely outcome.

On NO-GO: stop and publish the pilot. `docs/NO-GO-PAPER.md` was written before
Stage A ran so the pilot produces its figures.

## 9. Analysis, fixed in advance

- `scripts/analyze_crowding.py` refuses a missing arm, a dropped seed, an
  unexpected seed set, a run without evaluations, or an empty root. A partial
  matrix must not be analysed and then completed; inspecting results and then
  adding seeds is the same as choosing them.
- Report the triangulation chain and call it that, not mediation: recoverable
  bits are post-treatment, so this is not causal mediation.
- Gradient-mass decomposition at the final checkpoint: per-parameter gradient
  second moments on held-out fact and held-out iGSM batches separately, and
  the fraction of parameter mass dominated by each. Minutes per checkpoint,
  and the only measurement here that speaks to capacity rather than loss
  composition.
- Per-op, per-class and per-load breakdowns for every endpoint. A single
  aggregate hid a one-family swing in the earlier 1B result.

## 10. Amendments

Anything changed after this document is committed is an amendment: dated,
committed separately, with the reason, and reported in the paper. Values in
`[PILOT]` slots are filled once from pilot data and are not amendments.
