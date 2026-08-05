# Preregistration

**Status: FROZEN. All former `[PILOT]` slots are now closed, 2026-08-05.**
Nothing in this document may change on the basis of an outcome. Changes are
amendments under §10.

Written 2026-08-01, before any run of this design. Slots filled 2026-08-05.

**How the slots were closed.** A slot is closed either by a **measured value**,
where the measurement already exists and is catalogued in
`docs/RETAINED-RESULTS.md`, or by a **frozen rule**, where the measurement does
not exist yet but the decision procedure is fixed now so that the number cannot
be chosen after the outcome is seen. A rule is as binding as a value and is
stated so that a reader can execute it without judgement. Every slot below is
tagged `[MEASURED]` or `[RULE]`, and every `[RULE]` names the exact quantity
that instantiates it.

One slot could not be closed either way and is marked `[PARKED]`: the corpus
hashes live on FarmShare and are unreadable from this machine. Its resume
action is named in §4.

**No rule below is a function of the primary outcome.** `delta`, the Stage C
paired SD and the corpus hashes are all quantities separate from
`Y[FACTMASK] − Y[RANDPOS]`, so instantiating a rule cannot move a threshold
toward the result.

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

- Loads: **three**, at `f` = 1.0, 0.3, 0.1. `[MEASURED]` — three rather than
  two because the shape test is the only thing in the design that speaks to
  capacity rather than loss composition, and two points cannot distinguish a
  threshold from a line. The three fractions are the ones
  `ops/crowding/gen_configs.py` is driven with in `ops/crowding/RUNBOOK.md`.
- `N`, `E` at `f` = 1.0: **N = 996,408 entities, E = 200 exposures.**
  `[MEASURED]` — this is the `high-e200` operating point, built 2026-08-02 as
  job 1673768 and the first corpus in the project to return `VERIFY OK`. It
  places demand at 1.301 bits/param against 1.301 achievable, so **F/C = 1.000**,
  the critical point, computed by `theory/capacity.Load.ratio`. Lower loads
  follow the §3 rule, `f·N` entities at `E/f` exposures snapped to an exact
  divisor of `N·E` = 199,281,600.
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
- Corpus hashes: `[PARKED]`. Not fillable from this machine — the manifests
  live on FarmShare under `$MS_ROOT/corpora/<name>/manifest.json` and the
  Stanford VPN is down. **Resume action:** after
  `bash cluster/connect.sh syz`, record the `sha256` field of each corpus
  manifest here, one line per load, together with the build job id. The values
  are already fixed on disk by the builds themselves, so recording them later
  cannot change them; what is deferred is transcription, not a decision.

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
- **Corpus `op` band: [1, 8].** `[MEASURED]` — widened from [1, 4] because with
  the decoder fixed, the corrected endpoint aggregates to **93.8% over op 1–4**
  and has no headroom for a treatment to move. Per-op on the same checkpoint at
  step 14,076 over 1,500 items: op1 100.0%, op2 98.2%, op3 88.3%, op4 73.1%,
  and OOD op5 25.7%, op6 8.2%, op7 5.3%, op8 5.5%
  (`docs/RETAINED-RESULTS.md` §8). Widening the band is a corpus-construction
  decision taken from a SUP-only difficulty curve with no arm contrast in view,
  which is the procedure §10 permits for a `[PILOT]` slot.
- **Primary scoring band: the contiguous run of `op` values whose SUP accuracy
  falls inside [0.17, 0.85].** `[RULE]` — instantiated by the SUP arm of the
  first run on the [1, 8] corpus, before any FACTMASK or RANDPOS run is scored.
  The interval is the one `scripts/analyze_crowding.py --igsm-band` already
  takes and is not adjustable per load. Ops outside it are reported but excluded
  from the primary estimand, because an op at ceiling or floor cannot move and
  averaging it in only shrinks the effect toward zero. If no contiguous run of
  at least two ops falls inside the interval, that is a **responsiveness
  failure** under §7.3 and the confirmatory matrix does not run.
- **Architecture: `d40m_std`.** `[MEASURED]` — the non-recurrent variant.
  Recurrence buys effective depth 24 against 12, and the concern was that
  dropping it would cost the endpoint. It does not: `stepladder_d40m_std`
  reaches 93.8% in-band once the decoder is fixed, so the endpoint is
  demonstrably learnable without recurrence. `d40m_std` also cuts forward FLOPs
  per token from about 124M to 81M and makes bits-per-parameter comparable to
  the standard-GPT-2 literature. Stage A was to have decided this and cannot:
  its STOP is withdrawn (§10, Amendment 1).
- Generation: greedy, temperature 0, `max_new_tokens = 384`, stop at EOT.
- OOD band: op 5–8, reported separately, topology-hash disjoint from training.

**Secondary: deduction, reported per answer class, never as a single number.**
The eval is exactly 200 yes / 200 no in strict alternation and the NO branch's
trace is a canned sentence, so a constant "no" scores exactly 0.500.

## 6. Inference

- Unit: the training seed. Item bootstraps quantify evaluation noise only.
- `n`: **the smallest integer in [6, 8] whose paired one-sided t-test reaches
  80% power at α = 0.05 against the minimum interesting effect below, using the
  Stage C paired SD as the SD estimate. If 8 does not reach 80%, `n` = 8 and
  the minimum detectable effect is reported alongside the verdict.** `[RULE]` —
  instantiated by the Stage C paired SD, which is not a function of the primary
  outcome. For orientation, at n = 8 and df = 7 the requirement is a paired SD
  no larger than about 1.31 percentage points; the rule, not this figure, is
  what binds.
- One-sided, α = 0.05, Student's t with df = n−1. The exact sign test is
  reported alongside with its floor: at n=3 the smallest attainable one-sided
  p is 0.125.
- Seeds vary both initialization and shard permutation, so inference
  generalizes over corpus realizations rather than being conditional on one.
- Minimum interesting effect: **1.291 percentage points of closed-book iGSM
  accuracy.** `[MEASURED]` — this is the binomial standard error of the frozen
  1,500-item held-out eval at its worst case `p` = 0.5,
  `sqrt(0.25 / 1500)` = 0.012910. An effect smaller than one item-sampling
  standard error cannot be separated from evaluation noise on a single seed,
  before any seed-to-seed variance is added, so it is uninteresting whatever the
  power. Derived from the frozen eval size rather than chosen, and fixed here
  before any arm contrast exists.

  **This replaces the derivation this section originally specified, and the
  replacement is Amendment 2 in §10 rather than a slot fill.** The original text
  read "derived from `delta` (§7)", which is circular: §8.3 passes only if
  `delta` exceeds the minimum interesting effect by 2x, so defining the effect
  as `delta`/2 makes §8.3 true by construction and turns a gate into decoration.
  Anchoring on the eval's own noise floor instead leaves §8.3 able to fail, and
  it now fails unless `delta` ≥ 2.582 percentage points.
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

## 10a. Resource ceiling

The binding resources here are scratch bytes and GPU-hours, not dollars. Both
ceilings are frozen below and both are enforced by
`ops/crowding/guard.py`, which every submission path calls before `sbatch`.
A ceiling that exists only in a runbook is not a ceiling: this account was
already cancelled once mid-battery at 261 GB of scratch, against a documented
150 GB budget that nothing enforced.

The block below is machine-readable and is the single source the guard parses.
Editing a number here changes what the code permits, which is the point — the
ceiling and the document cannot drift apart.

```prereg-ceiling
scratch_bytes_ceiling   = 150000000000
gpu_hours_ceiling       = 1600
soft_stop_fraction      = 0.90
divergence_factor       = 1.5
campaign:fc1            = 40
campaign:stage_a        = 20
campaign:ladder         = 80
campaign:stage_b        = 60
campaign:stage_c        = 220
campaign:matrix         = 1600
```

- **`scratch_bytes_ceiling`** is FarmShare's 150 GB working budget, counted over
  resident corpora plus checkpoints. The guard refuses a submission whose
  projected resident bytes would exceed it, which is why the dose ladder must
  build, train and delete one load at a time: three 21.3B-token loads are
  256 GB.
- **`gpu_hours_ceiling`** is the programme total. Per-campaign sub-ceilings bind
  first, so exhausting `fc1` cannot silently consume the matrix's allocation.
- **`soft_stop_fraction`** holds back the last slice. At 90% of a campaign
  ceiling the guard stops authorising *new* runs while still permitting those
  already claimed to finish, because a ceiling hit at 95% completion leaves
  half-finished runs that are unanalysable and fully paid for.
- **`divergence_factor`** is the breaker. If measured GPU-hours for a completed
  run exceed its projection by more than this factor, the guard halts the
  campaign. That is a scientific signal rather than an accounting one — the cost
  model of the experiment is wrong — so it is filed as a finding and amended
  before anything resumes.

**Projections are made from measured throughput only.** The `d40m_std` rate of
~277,000 tok/s is an extrapolation from `d40m`'s measured 184,671 tok/s via an
assumed 1.5x FLOPs saving, and has never been observed. The guard therefore
budgets at the measured `d40m` rate and treats any faster outturn as headroom
returned. Multiplying a measured rate by an expected speedup and then budgeting
against the product is how a 16.4-hour estimate becomes a 24.7-hour run.

## 10. Amendments

Anything changed after this document is committed is an amendment: dated,
committed separately, with the reason, and reported in the paper. Values in
`[PILOT]` slots are filled once from pilot data and are not amendments.

- **Amendment 1, 2026-08-02** — `docs/AMENDMENT-2026-08-02.md`. Six methods
  corrections filed before any confirmatory run existed: RANDPOS is now
  difficulty-matched and the match measured (§2); a corpus whose lane shares
  drift from the requested ones is refused (§4); equal mask mass is an
  invariant; the confirmatory inference is blocked by seed within one named
  primary load rather than pooled across loads (§6); the dose-response shape
  test is connected to the analysis (§9); every pilot stage carries its entity
  count so recoverable bits are reported (§7). It also disqualifies the Stage A
  corpus, whose fact lane realised 3.75% of a requested 50% on a synthetic bed.
  **Stage A's STOP may not be cited.**

- **Amendment 2, 2026-08-05** — the minimum interesting effect is anchored on
  the eval's item-sampling noise floor instead of on `delta`.

  *Reason.* §6 originally derived the minimum interesting effect from `delta`,
  while §8.3 passes only if `delta` exceeds that same effect by at least 2x.
  The two together are circular: any value satisfying the §6 derivation
  satisfies §8.3 automatically, so §8.3 could not fail and was therefore not a
  gate. This was found while closing the `[PILOT]` slots, before Stage B had
  run and therefore before any value of `delta` existed, so no outcome informed
  the change.

  *Effect on the design.* §8.3 becomes failable, with a threshold of
  `delta` ≥ 2.582 percentage points. The change can only make the pilot harder
  to pass, never easier, which is the direction an amendment filed without
  supervision should go.

  *Not changed.* §8.3's text, the 2x factor, and the verdict rule in §6. Only
  the quantity the effect is derived from.

- **Slot closure, 2026-08-05** — the `[PILOT]` slots in §3, §4, §5 and §6 were
  closed on the date shown, before any FACTMASK or RANDPOS run on the [1, 8]
  corpus existed. Two are frozen rules rather than values and name the quantity
  that instantiates them; one, the corpus hashes in §4, is parked on VPN access
  and is a transcription rather than a decision. This is a slot closure under
  the last paragraph of this section and is not itself an amendment, except for
  Amendment 2 above, which is filed separately because it changed a prescription
  rather than instantiating one.
