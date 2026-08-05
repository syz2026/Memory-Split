# Amendment 1 — 2026-08-02

Filed under `docs/PREREGISTRATION.md` §10: "Anything changed after this
document is committed is an amendment: dated, committed separately, with the
reason, and reported in the paper."

## Why these are methods corrections and not outcome-driven changes

**No confirmatory run exists.** The matrix has not been built, launched or
scored; `outputs/` contains no cell from this design. Every change below was
made to code that had never produced a confirmatory number, so none of them
could have been chosen to favour an outcome. This is the one moment in the
programme when the analysis can be repaired without the repair being suspect,
which is the reason for doing it now rather than after Stage B.

The one place where an outcome *was* visible is Stage A, and Amendment 2 makes
its result weaker rather than stronger: it disqualifies the corpus that
produced the STOP. Amendments that cost you your existing result are the ones
least likely to be motivated reasoning, but the direction is recorded here so a
reader does not have to take that on trust.

---

## 1. RANDPOS is difficulty-matched, and the match is measured

**What was wrong.** §2 requires the control to match the treatment on mean
per-token NLL "using a token-NLL table frozen from a pilot run on disjoint
data", and makes an unmatchable control a reportable result. The machinery
existed in `corpusgen/randpos.py` — `build(..., token_nll=None)` and
`match_report` — and was never supplied: `ops/crowding/build_corpus.py` called
`randpos.build(ids, fmask, rng)` with three positional arguments. **Every
corpus this project has built matched count, span length and relative position
but not difficulty, and no artifact records the gap.**

This is the axis the module's own docstring calls "the one that matters and the
one that can fail", on the grounds that fact values carry ~2.85 bits/token
against under 0.5 for template text, so a count-matched control removes several
times less loss mass than the treatment.

**What changed.** `ops/crowding/nll_table.py` builds a frozen per-token-id NLL
table from a supervised checkpoint over a burned pilot seed, with a provenance
sidecar recording the checkpoint hash, seed and token count. `build_corpus.py`
takes `--nll-table`, threads it through both the serial and parallel fact
lanes, and accumulates the corpus-level match exactly over every fact token.
`manifest.json` gains `randpos_validity`. A build without a table is refused
unless `--rehearsal` is passed.

**What is deliberately not gated.** An out-of-tolerance gap prints a banner and
is recorded, but does not fail the build, because §2 says it "is reported as
the result, not gated away".

**What the first measurement returned, and it is a result.** The
operating-point corpus `high-e200` (996,408 entities x 200 exposures, 21.3B
tokens, built 2026-08-02, `VERIFY OK`) reports FACTMASK's masked spans at
**1.9598 nats against RANDPOS's 1.2712 — a 35.1% relative gap against the 20%
tolerance.** Under §2 the control is empirically invalid and that is reported,
not gated away. It is not a tuning failure: sweeping the placement's position
tolerance from 0.05 to 1.00 moves the gap from 42.2% to 23.5%, where it
asymptotes without ever reaching 20%. Fact values are intrinsically
higher-entropy than anything in a fixed biography template, so a count-matched
non-value control cannot remove equal loss mass at any placement. Catalogued as
`docs/RETAINED-RESULTS.md` §7 and promoted to contribution 2 of the NO-GO
paper. The corpus was deliberately **not** rebuilt to chase a better number,
since no setting clears the gate and re-tuning after seeing the measurement
would be the outcome-driven change this amendment exists to avoid.

**A second confound the same diagnostics exposed.** `match_report` measures
what fraction of control-masked tokens land in a value's *cue window* — the
three tokens immediately before a value, the "was born in" that predicts it.
`randpos.py` warned about this in prose and nothing had ever measured it. The
first build to report it put **30.9% of control-masked tokens inside a cue
window**. If that holds at production scale, RANDPOS is removing fact-relevant
supervision in roughly a third of its mass, which biases the control toward the
treatment and attenuates the primary contrast toward zero. It is now recorded
in every manifest as `randpos_validity.mean_cue_window_overlap_frac`, and
`report_findings` prints a banner above 10%.

That figure came from a 200k-token rehearsal corpus and is not a production
number. It needs re-measuring on the real corpus before it is quoted, and if it
holds, `CUE_WINDOW` exclusion should become a placement constraint rather than
a reported statistic.

**Known approximation, stated for the paper.** The table is marginal, not
contextual: it is the mean NLL of each token id, not of each token in its
context. Nothing better is available, because RANDPOS placement happens at
build time when no model has seen the document. It is adequate for separating
high-surprisal values from near-deterministic template text, a gap of several
nats that survives averaging, and it is not adequate for fine distinctions
within either class.

## 2. A corpus can no longer silently become a different experiment

**What was wrong.** The fact lane is finite — `n_entities × exposures`
documents — while its share is a fraction of the total. When the share asked
for more than the lane could emit, `build` gave the remainder to the bed so the
token count stayed exact. That reallocation is correct and stays: a short
corpus changes `max_steps`, and every arm and load must share a step count.

It was also silent, and Stage A walked into it. 20,000 entities at 20 exposures
emit ~30M tokens against a 50% share of 800M, so:

| lane | requested | realised |
|---|---:|---:|
| fact | 50% | **3.75%** |
| igsm | 30% | 30.00% |
| deduction | 10% | 10.00% |
| bed | 10% | **56.25%** |

The bed absorbed 46.3% of the corpus. `verify()` passed, and
`tests/test_build_corpus.py` contained a test asserting that it should. The
manifest recorded `bits_per_param.d40m = 0.0261` against the 2.0 the design
targets — a factor of 77 — where nobody read it.

**What changed.** `plan_lanes` prices the fact lane before a byte is written
and `build` raises `LaneUnderfilled` with the exposure count that would fix it.
The manifest records `requested_shares`, `realised_shares` and `lane_plan`,
because `lane_budgets` is mutated by the reallocation and cannot answer what
was asked for. `verify` fails closed on a realised share more than 1 point from
the requested one. The test that asserted silence now asserts disclosure.

**Consequence for Stage A.** Its corpus was not the corpus the design
specifies, and its manifest additionally records `bed:
SYNTHETIC-REHEARSAL-ONLY`, which the runbook says to treat as a build failure.
Together with the two defects already recorded in `docs/THEORY-ENDPOINT.md` —
a step budget ~7× short of the literature, and a difficulty ladder that was
never built — **Stage A's STOP now rests on nothing and must not be cited.**
`outputs/pilot/stage_a_report.json` is retained as a record of what was run,
not as a result.

## 3. Equal mask mass is now an invariant, not an aspiration

**What was wrong.** Two paths dropped control mass silently. A document too
dense for a contiguous control span skipped the span entirely
(`randpos.build`), and a fact document truncated by the token limit had its two
sidecars cut at different points. The second was reproducible: at 100 entities
× 25 exposures × 300k tokens, FACTMASK zeroed 37,577 targets and RANDPOS
37,574, with the entire discrepancy in the final 200 tokens.

Unequal mask mass means the arms zero a different number of targets and
therefore train a different effective objective — which is precisely the
confound the fixed-denominator loss was introduced to remove. It would have
reintroduced it through the corpus after being removed from the loss.

**What changed.** A dense document now scatters the span's tokens into free
positions rather than dropping them, preserving mass and degrading the length
histogram instead, which `match_report` records. The builder spends the corpus
tail on bed tokens, which are all-ones in both sidecars, so truncation never
cuts a fact document.

## 4. The confirmatory inference is blocked by seed within one named load

**What was wrong.** `scripts/analyze_crowding.py` flattened every load's
effects into one list and passed it to `verdict()`. Three loads × eight seeds
would have been treated as n=24 independent observations when the same eight
initialisations and shard permutations appear at every load, shrinking the
interval by roughly √3 for free. Validity gates were computed from means taken
across loads, although the loads carry deliberately different fact content, so
a mean SUP storage figure describes no corpus that was trained on and can clear
the burden gate while the primary load fails it. The intended primary load was
selected by `max(per_load, key=lambda k: mean_bits["sup"])`, whose key function
ignores its argument and therefore returned an arbitrary load; the variable was
then unused.

**What changed.** `--primary-load` is required and is not inferred, because
every rule for picking it from data — highest storage, largest effect, best
power — chooses the estimand after seeing the outcome. Inference is blocked by
seed within that load. Gates are evaluated at that load. The pooled figure is
still reported, labelled `pooled_across_loads_NOT_PRIMARY` with its correlation
stated.

**Still to be frozen before the matrix runs.** The identity of the primary
load. §5 does not name it, and this amendment deliberately does not choose it.

## 5. The dose-response shape test is connected to the analysis

**What was wrong.** `theory/capacity.py::dose_response_signature` is described
in `docs/THEORY-CAPACITY.md` as "the only analysis in the design that speaks to
capacity rather than loss composition", and it was referenced by nothing that
runs. It also could not report its most important verdict: the guard for the
zero case read `peak < tol * max(1e-12, peak)`, which is true only for negative
`peak` and therefore never fired. Effects three orders of magnitude below the
noise floor would have been classified "rising then saturating: consistent with
capacity reallocation".

**What changed.** The classifier takes an absolute `floor`, supplied as the
preregistered minimum interesting effect, and the analyzer reports
`dose_response` with loads ordered by measured SUP storage rather than by name.

## 6. Stages B and C carry their entity count

**What was wrong.** No stage in `ops/crowding/pilot.py` set `n_entities`.
`scripts/run_evals.py` responds to its absence by writing `"config carries no
n_entities; nothing to probe"` and emitting no storage figure.

**Scope, checked against the cluster rather than assumed.** Stage A escaped
this: `ops/crowding/stage_a.sh` line 75 patches `c["n_entities"] = entities`
into every config after `pilot.py` writes it, and
`runs/pilotA_d40m_std/config.yaml` on FarmShare duly carries `n_entities:
20000`. `ladder.py` sets it directly. **There is no equivalent wrapper for
Stage B or Stage C** — the runbook invokes `pilot.py --stage B|C` directly —
so the defect is real for those two and only those two. Stage B's
exposure-to-storage frontier is Figure 1 of the NO-GO paper and Stage C's
leakage gate is the descendant of the 94.2%-recoverability failure that made
the previous corpus inert, so the two stages that could not report storage are
the two whose entire purpose is to report it.

**What changed.** `--entities` is required by `pilot.py` and reaches all three
stages from the generator itself, so a stage no longer depends on a shell
wrapper that exists for only one of them.

**What Stage A's probe actually returned, now that it can be read.**
`recoverable_bits_per_param = -0.3190`, and the four ladder runs return
-0.2761 to -0.3301. A negative recoverable-bits figure means the model scores
*worse than the pool baseline* on fact values — the same "confidently wrong"
regime documented in `docs/GATE0-CEILING-IS-NOT-A-BOUND.md`, where a dense arm
reached 25.03 nats against a 10.83-nat uniform ceiling. It is the expected
reading for a corpus carrying 0.0261 bits/param of fact demand, which is to
say almost none.

## 7. The compute budget is restated at the corpus actually in use

**What was wrong.** `RUNBOOK.md` prices every stage at 5.737B tokens per run,
which is 10,943 optimizer steps. `docs/THEORY-ENDPOINT.md` argues the Stage A
floor is a step-budget artifact and `ops/crowding/advance.py` accordingly sets
`FULL_STEPS = 31_280`, which is 16.4B tokens. Both cannot be true, and the
runbook's 410 GPU-hour matrix estimate is the stale one.

**What changed.** The runbook is restated at 16.4B tokens: **~16.4 h per
`d40m_std` run, ~1,184 GPU-hours for 72 runs, ~12.3 days at four concurrent
GPUs**, and 65.6 GB per load against the 150 GB scratch budget, which permits
two resident loads and forces the ladder to be built and deleted one dose at a
time.

### The two designs, and the decision this amendment does not make

The token budget is not an isolated number. Tracing it out, the repository
contains **two internally coherent designs that are not the same experiment**,
and no amendment has ever chosen between them:

| | **A** — prereg §4, RUNBOOK Stages B/C | **B** — `build.sbatch` defaults, THEORY-CAPACITY, `advance.py` |
|---|---:|---:|
| total tokens | 5,737,000,000 | 16,399,769,600 |
| optimizer steps | 10,942 | 31,280 |
| entities x exposures | 382,900 x 100 | 1,531,800 x 100 |
| lane shares (fact/igsm/ded/bed) | 50 / 30 / 10 / 10 | 70 / 12.3 / 4.3 / 13.4 |
| fact demand, d40m | 0.500 bits/param | 2.000 bits/param |
| corpus size | 21.4 GB | 61.1 GB |

Both are exactly self-consistent: in each, the entity count, exposure count and
fact share are chosen so the fact lane fills its share to within 0.01%. Someone
did this arithmetic carefully, twice, and only recorded one of them in the
preregistration.

Design B follows from the two theory documents — `THEORY-CAPACITY.md` targets
F/C = 2.0 at the top dose, and `THEORY-ENDPOINT.md` argues for ~31k steps — so
it is almost certainly the intended one. It has never been written down as an
amendment, so **the preregistration currently describes an experiment nobody
intends to run.**

Two consequences worth knowing before choosing:

- The iGSM share falls from 30% to 12.3%, which looks alarming for a design
  whose primary endpoint is iGSM accuracy and whose diagnosed failure was too
  little iGSM training. In absolute terms it is not: 2.02B iGSM tokens against
  1.72B, **a 17% increase**, spread over 2.86x the optimizer steps.
- Design B costs 2.86x the compute of Design A. The 410 GPU-hour matrix figure
  belongs to A; B's matrix is ~1,184 GPU-hours.

**This amendment deliberately does not pick one.** Choosing the operating point
is a preregistration decision about what experiment is being run, not a bug
fix, and it should be made and dated on its own. `build.sbatch` now carries a
comment saying so at the point where the shares are set.

---

## What this amendment does not change

- The estimand. `effect = Y[FACTMASK] − Y[RANDPOS]` is untouched.
- The claim that capacity reallocation is **not identified** by this design.
- Any threshold, gate value, α, seed floor, or verdict rule.
- Any `[PILOT]` slot. Those are still to be filled once, from pilot data.

---

## Appendix: what the gates found when run against FarmShare

Audited 2026-08-02 on `/scratch/users/syz/crowding`, after the fixes above.

### The difficulty ladder ran, and its four corpora are invalid

`ladder.py --mode rank` returns `any_rung_clears: false` at every modulus:

| mod | no-skill baseline | op=1 accuracy | clears |
|---:|---:|---:|:--|
| 23 | 0.0753 | 0.0376 | no |
| 11 | 0.1327 | 0.0074 | no |
| 7 | 0.2053 | 0.0000 | no |
| 5 | 0.2573 | 0.1293 | no |

Its verdict is "NOT LEARNABLE even at mod 5 ... no longer a difficulty problem".
**That verdict cannot be taken at face value**, because all four ladder corpora
have the §2 defect and it is worse in them than in Stage A:

| lane | requested | realised |
|---|---:|---:|
| fact | 70% | **3.75%** |
| igsm | 12.3% | 12.30% |
| deduction | 4.3% | 4.30% |
| bed | 13.4% | **79.65%** |

`MS_ENTITIES=20000 MS_EXPOSURES=20` against `build.sbatch`'s 70% fact default
leaves a deficit of 66 points, which the bed absorbs. The bed is
`SYNTHETIC-REHEARSAL-ONLY`, so **79.65% of every ladder run's training tokens
were random word salad** drawn from `igsm_lite.ADJECTIVES + NOUNS + PLACES`.
The iGSM lane got 98.4M tokens against Stage A's 240M — 2.4x less — while the
ladder was supposed to hold everything but the modulus fixed against Stage A.

The training loss corroborates it: the ladder runs settle at `loss_ema` 2.31
against Stage A's 1.76, which is what a corpus dominated by high-entropy
filler looks like.

So the ladder varied difficulty, iGSM share and bed composition simultaneously,
and its floor has at least three candidate explanations. It is not the clean
descent `docs/THEORY-ENDPOINT.md` specified, and the NO-GO paper must not cite
it as one. A rerun is cheap — 4 corpus builds plus ~4 GPU-h — and the builder
now refuses the combination that produced these.

### The step-budget hypothesis is still untested, and its corpus is ready

No `stepladder` run exists. The `high` corpus does, and it is sound on every
axis this amendment added a gate for:

| | |
|---|---:|
| tokens | 16,400,000,000 (31,280 steps) |
| entities x exposures | 1,531,800 x 100 |
| realised shares | **exactly 70 / 12.3 / 4.3 / 13.4** |
| bed | pinned `fineweb-edu.jsonl`, not synthetic |
| fact demand, d40m | **2.0001 bits/param** |

That is Design B built correctly, and it is the only artifact in the project
that has ever carried a real fact load. At 31,280 steps it is a 20x step
increase over every run listed above, and it tests the one explanation for the
floor that nothing has yet addressed.

**One defect, and it does not block a SUP run.** `mask_audit.mass_matched` is
`false`: factmask zeroes 2,856,757,563 targets and randpos 2,856,757,560, a
three-token difference — the tail-truncation signature fixed in §3, and the
same magnitude the local rehearsal showed. It fails `verify()`. It is
irrelevant to a SUP-only step-ladder run, which trains with full supervision
and uses `factmask.bin` only as the gate-0 probe, so the corpus can be used
for that today. It must be rebuilt before any FACTMASK/RANDPOS arm trains on
it — which it needs anyway, since it was built without a difficulty table.

## What a reader should conclude

Six defects, all of which would have corrupted or silently voided the
confirmatory matrix, were found and fixed before it ran and before any
confirmatory number existed. Two of them retroactively disqualify results the
project already had: Stage A's STOP, and the difficulty ladder that was run to
check it. The programme's stated position is superseded twice over —
`docs/THEORY-ENDPOINT.md` withdrew the inference from Stage A, and this
amendment withdraws the corpora underneath both Stage A and the ladder.

What survives is narrower and more useful than either: **the step-budget
hypothesis has never been tested, and the corpus to test it with is built,
correct and idle.** Every run in the project to date sits at 1,500–1,525
optimizer steps against a literature that reports 10⁴–10⁶ for modular
arithmetic. That is the experiment to run next, and it costs ~17 GPU-hours
rather than the ~1,184 the matrix would.
