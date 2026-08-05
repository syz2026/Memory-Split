# Findings ledger

The deduplicated, persistent work queue produced by the review swarm. This file
is the interface other agents patch from. Reports under `review-swarm/reports/`
are the raw per-model output and are not authoritative; this file is.

## How to use this as a patching agent

1. Work `OPEN` findings in severity order, S0 first.
2. Before patching, run the finding's **Check**. If it does not reproduce, set
   the status to `NOT-REPRODUCED` with your evidence and move on. Do not patch
   on the strength of the description alone.
3. After patching, set the status to `PATCHED`, add the commit or the file
   changed, and leave the finding in place. Do not delete rows. The next cycle
   re-verifies patched findings and will reopen anything that regressed.
4. If you believe a finding is wrong, set `DISPUTED` and write why. The next
   cycle routes disputed findings to a different model than the one that filed
   them.

## Status values

`OPEN` · `PATCHED` · `NOT-REPRODUCED` · `DISPUTED` · `WONTFIX` · `SUPERSEDED`

## Confirmation count

`Confirmations: n` is the number of distinct models that have independently
raised the finding across all cycles. Three is strong evidence the finding is
real. One from a single cycle is a hypothesis.

**Every finding below was re-verified by the orchestrator before landing here.**
Findings that did not survive verification are in the graveyard at the bottom,
with the reason, so no later cycle re-files them.

---

## Status

**Cycle 1 complete.** All three lanes reported, every finding re-verified.

**Cycle 2 complete.** The repository was unchanged since cycle 1 (same HEAD
`efc338a`, nothing patched), so cycle 2 was redirected from re-scanning to
**adversarial cross-verification**: each model held a lane whose cycle-1 findings
a different model filed, and its primary task was to try to disprove them.

**Result: all ten cross-checked findings survived. None was retired.** Five were
confirmed as filed, five were refined, and three of those five had their severity
reduced. Seven new findings were added, including two S0s in the confirmatory gate
and estimand path that cycle 1 never reached.

| Severity | Open | Change from cycle 1 |
|---|---|---|
| S0 | 7 | +2 new, −1 downgraded |
| S1 | 8 | +3 new, −1 downgraded |
| S2 | 6 | +3 new, +1 downgraded in |
| S3 | 3 | +1 new, +1 downgraded in |
| **Total** | **24** | +9 |

Fifteen findings is above the charter's own guidance and is expected for cycle 1
only. The repository withdrew a large fraction of its numbers on 2026-08-05, the
drafts were written across that boundary, and three modules are untracked and had
never been reviewed. Cycle 2 should be much thinner. If it is not, the lanes are
too broad rather than the repository being that broken.

---

## Open findings

### MS-001 — The preregistration is banner-frozen with the primary load unnamed

- **Severity:** S0 · **Status:** OPEN · **Confirmations:** 2 (Opus 5 cycle 1; Grok 4.5 cycle 2)
- **Cross-check:** Grok 4.5 attacked the "implied primary" escape hatch, that §3
  tagging `f = 1.0` as `[MEASURED]` might make it the de facto primary, and
  rejected it: `f = 1.0` is the ladder's construction reference and the F/C = 1
  point, and nothing converts a construction reference into a frozen estimand.
  Finding stands at S0.
- **Target:** `docs/PREREGISTRATION.md:3` and `:361-368`
- **Defect:** The banner reads "Status: FROZEN. All former `[PILOT]` slots are now
  closed, 2026-08-05", while the identity of the primary load is named nowhere in
  the document. `docs/AMENDMENT-2026-08-02.md:171-172` says so explicitly:
  "**Still to be frozen before the matrix runs.** The identity of the primary
  load. §5 does not name it, and this amendment deliberately does not choose it."
  `scripts/analyze_crowding.py:119-125` still hard-refuses without
  `--primary-load` and directs the caller to freeze it in §5, which never
  mentions load. The estimand is `Y[FACTMASK] − Y[RANDPOS]` **at one load**, so an
  unnamed load leaves the estimand unspecified and whoever runs the matrix will
  supply it after the runs exist.
- **Why checks miss it:** the analyzer refuses to *infer* the load, which is the
  check that exists, and cannot tell whether the value it was handed was frozen
  beforehand or chosen this morning. Nothing compares the argument to the
  document. The narrow claim "all `[PILOT]` slots are closed" is literally true
  because the primary load was never tagged `[PILOT]`, which is what makes the
  banner misleading rather than false.
- **Check:** `rg -n "primary load|primary_load" docs/PREREGISTRATION.md` returns
  only line 336, a backreference to Amendment 1, and no assignment. Fix by naming
  the load with a date, then add a test that parses the declaration out of the
  preregistration and asserts the analyzer was invoked with it, the same
  document-cannot-drift-from-code discipline §10a already applies to the resource
  ceilings.
- **Retires when:** a dated line in `PREREGISTRATION.md` names the load, or the
  design moves to a pooled estimand (which Amendment 1 §4 forecloses).

### MS-002 — PAPER-MEASUREMENT §2 states the wrong corpus and step budget for its own findings

- **Severity:** S0 · **Status:** OPEN, REFINED · **Confirmations:** 2 (Grok 4.5 cycle 1; GPT 5.6 Sol cycle 2)
- **Refinement, GPT 5.6 Sol cycle 2.** The corpus attribution is independently
  reproduced and stays S0. **The arithmetic charge added by the orchestrator was
  too strong and is withdrawn as a standalone error.** The last displayed
  learning-curve point is step 12,512, and 40,695 / 12,512 = 3.25, so "roughly a
  factor of three" has a plausible reading as budget against the demonstrated
  partial curve. The sentence is ambiguous, because its immediate antecedent is
  step 7,820, but it is not false under every reading.

  It must still be recomputed as part of the corpus fix: against the correct
  31,280-step budget neither candidate gives three (31,280/7,820 = 4.0,
  31,280/12,512 = 2.5). Treat the factor as downstream of the S0, not as a second
  defect.
- **Related scope, filed here rather than duplicated:** `README.md:19` and
  `docs/PREREGISTRATION.md:40` both specify roughly 141 tokens per parameter while
  the same preregistration names `high-e200` at 21.3B tokens (`:101,304`), which
  is 526 tokens per 40.56M parameters.
- **Target:** `docs/PAPER-MEASUREMENT.md:60` and `:211`
- **Defect:** §2 Setting says "Runs are 21.3B tokens, 40,695 optimizer steps, on
  one NVIDIA L40S" and presents that as the setting for every finding in the
  paper. But Findings 1 and 2, the per-op tables, and the learning curve at
  `:199-207` all come from `stepladder_d40m_std`, which is **16.4B tokens and
  31,280 steps** (`docs/RETAINED-RESULTS.md` §8). 21.3B / 40,695 is `high-e200`,
  and `docs/PAPER-WORKSHOP.md:44-45` states plainly that **no model has yet been
  trained on it**. The long draft therefore attributes its measurements to a
  corpus that has never been trained.

  The error propagates into an arithmetic claim the paper itself flags as
  load-bearing. Line 211 reads "Accuracy is within four points of its final value
  by step 7,820, so the 40,695-step budget is generous by roughly a factor of
  three — a fact that matters for anyone pricing a confirmatory matrix off it."
  Neither budget yields three: 40,695 / 7,820 = 5.2, and the correct budget gives
  31,280 / 7,820 = **exactly 4.0**. (7,820 is the fifth of twenty snapshots at
  31,280 / 20 = 1,564 apart, which independently confirms the curve belongs to the
  31,280-step run.) A matrix priced off "a factor of three" against the wrong
  budget is mis-sized in both terms at once.
- **Why checks miss it:** no test cross-checks a paper's stated setting against
  the run id whose numbers appear in the same section. `RETAINED-RESULTS.md`
  catalogues both corpora correctly and asserts nothing about draft consistency.
  The workshop draft gets this right, so the defect is confined to the long form.
- **Check:** `rg -n "21\.3B|40,695|16\.4B|31,280|stepladder_d40m_std" docs/PAPER-MEASUREMENT.md docs/PAPER-WORKSHOP.md`
  then `python3 -c "print(31280/7820, 40695/7820)"` → `4.0 5.203...`. Rewrite §2 to
  separate the two corpora the way `PAPER-WORKSHOP.md` §2 does, and recompute the
  step-budget sentence.
- **Retires when:** §2 distinguishes the corpora and the budget sentence uses
  31,280 with a factor that follows from it.

### MS-003 — PAPER-MEASUREMENT asserts "perfect through three" after the same draft withdraws it

- **Severity:** S0 · **Status:** OPEN · **Confirmations:** 2 (Grok 4.5 cycle 1; GPT 5.6 Sol cycle 2)
- **Cross-check:** confirmed as filed. The only perfect op3 cell is 9/9 in a table
  the draft itself calls too small to publish; the larger table has op3 at 88.3%;
  the final-checkpoint table has not landed. "Solves ... perfectly" is an
  unsupported population reading, not a rounding choice.
- **Target:** `docs/PAPER-MEASUREMENT.md:444-446` against `:176-182`; also
  `docs/RESULTS-2026-08-04.md:57-60`
- **Defect:** §4 states "These per-cell counts (8–17) are too small to publish",
  reports that at n≈1,500 mid-training op3 is **88.3% and not at ceiling**, and
  concludes that "'perfect through three, cliff at six' **is not supported at
  large n**". The discussion at `:444-446` then asserts the endpoint "solves
  three-operation dependency chains **perfectly** and reaches 93.8% in-band", with
  no caveat. A live false shape claim sits ~260 lines from its own withdrawal.
  `RESULTS-2026-08-04.md` carries the same unqualified reading ("solves
  three-operation dependency chains perfectly and falls off a cliff at six")
  sourced from the 9/9 and 0/12 cells that §4 disowns.
- **Why checks miss it:** `test_no_document_resurrects_a_withdrawn_number`
  fingerprints the §A/§B ledger strings only. Nothing asserts that a draft's
  discussion is consistent with its own limitations.
- **Check:** `rg -n "perfectly|not supported at large n|88.3" docs/PAPER-MEASUREMENT.md docs/RESULTS-2026-08-04.md`.
  Either drop "perfectly" and the op1–3 ceiling reading, or land the
  final-checkpoint n=1,500 table and cite it (see MS-005).
- **Retires when:** the discussion is scoped to what large-n supports, or a landed
  1,500-item final-checkpoint table shows op1–3 at ceiling.

### MS-004 — HANDOFF.md presents padded-decoder results as science that stands

- **Severity:** ~~S0~~ → **S2** · **Status:** OPEN, REFINED · **Confirmations:** 2 (Grok 4.5 cycle 1; GPT 5.6 Sol cycle 2)
- **Downgrade, GPT 5.6 Sol cycle 2, accepted.** The content is wrong and
  actionable because the file says to read it first, but the file is an explicitly
  dated 2026-08-03 snapshot and the reversal is dated 2026-08-04. It did not
  resurrect a withdrawn result, it became stale when later evidence arrived. By
  the charter's ladder that is an S2 collaborator artifact gap, not an S0 paper
  claim, and a banner pointing at `RESULTS-2026-08-04.md` is a sufficient fix.
- **Sharpening that came with the downgrade, and which matters more than the
  severity:** `RETAINED-RESULTS.md` withdraws all pre-fix batched accuracies by a
  blanket rule, but its *enumerated* table does not list the difficulty-ladder
  cells, and `test_repo_state.py` scans only `docs/*.md` and `*.tex`, never
  `HANDOFF.md`, and holds no ladder fingerprint. That is precisely why this stale
  snapshot is invisible to the guard that exists to catch it. The blanket rule
  covers these numbers semantically and cannot make the test see them.
- **Target:** `HANDOFF.md:72-97`, `:163-164`, `:190-193`, `:248-252`
- **Defect:** Under the heading "The two results that do stand", the file presents
  the difficulty ladder with op=1 accuracies 0.0100, 0.0441, 0.0080, 0.0268 and
  the verdict that the endpoint "is not learnable at 1,500 steps, at any
  difficulty". `docs/RESULTS-2026-08-04.md:80-90` overturns exactly this: "Both
  difficulty ladders, including the 2026-08-02 rebuild on valid corpora and its
  verdict that the task is 'NOT LEARNABLE even at mod 5'." Those cells are
  pre-fix generative accuracies, which `RETAINED-RESULTS.md` withdraws wholesale.
  The file also still says the step ladder "has not been read yet" and names
  `NO-GO-PAPER.md` as the likely deliverable, both false after the reversal.

  Mitigating: `HANDOFF.md` is a dated snapshot (2026-08-03) and is untracked. It
  is nonetheless the file that says "Read this file first; it is the map", so a
  collaborator would build on withdrawn numbers. A supersession banner may be a
  sufficient fix rather than a rewrite.
- **Why checks miss it:** the resurrection fingerprints do not include these
  ladder floats, and nothing greps for post-withdrawal standing-results claims.
- **Check:** `rg -n "results that do stand|not learnable|0\.0100|has not been read yet" HANDOFF.md`
  against `docs/RESULTS-2026-08-04.md:80-90`.
- **Retires when:** the standing-results section is labelled superseded or the
  ladder table is removed, and the Step 0 / NO-GO framing carries the reversal.

### MS-005 — The §10 control table cannot be reproduced from any single treatment mean

- **Severity:** S1 · **Status:** OPEN · **Confirmations:** 2 (Opus 5 lane A + Grok 4.5 lane C, cycle 1)
- **Target:** `docs/RETAINED-RESULTS.md:386-398`, `:255-259`;
  `docs/PAPER-WORKSHOP.md:110-121`; `docs/PAPER-MEASUREMENT.md:266,273`
- **Defect:** Two models arrived at this from opposite directions and the merged
  version is sharper than either. §7 reports the treatment mean as **1.9598**
  (all 21.3B tokens) and §10 as **1.9584** (800-document subsample). The §10 gap
  column is consistent with *neither* denominator across all five rows:

  | control | NLL | printed | vs 1.9584 | vs 1.9598 |
  |---|---:|---:|---:|---:|
  | shipped | 1.2712 | 35.1% | 35.1 ✓ | 35.1 ✓ |
  | position unconstrained | 1.4987 | 23.5% | 23.5 ✓ | 23.5 ✓ |
  | best possible length-matched | 1.5667 | 20.0% | 20.0 ✓ | 20.1 ✗ |
  | scattered, mean-matched | 1.7677 | 9.8% | 9.7 ✗ | 9.8 ✓ |
  | scattered, k hardest | 1.8644 | 4.9% | 4.8 ✗ | 4.9 ✓ |

  Rows 3 and 4–5 require different denominators, so the table was assembled from
  two computations and no reader can reproduce it from the numbers on the page.

  The claim that rests on row 3 survives either way but should be restated. Under
  1.9584 the oracle gap is 20.001%, under 1.9598 it is 20.058%, and
  `corpusgen/randpos.py:211` tests `rel <= 0.20`, so **the oracle fails the
  tolerance under both**. "Lands exactly on the preregistered 20% tolerance"
  (`RETAINED-RESULTS.md:397-398`) and "borderline" (`PAPER-MEASUREMENT.md:266`)
  both read as achieving it. "Even an oracle picking the hardest legal span cannot
  reach the tolerance" is the true statement and the stronger one for the paper's
  own thesis.
- **Why checks miss it:** the tables are prose. `nll_within_tolerance` runs inside
  the builder on the shipped control only and never on the four counterfactual
  rows.
- **Check:** the table above reproduces with
  `python3 -c "d=1.9584; print([round((d-v)/d*100,1) for v in (1.2712,1.4987,1.5667,1.7677,1.8644)])"`
  and again with `d=1.9598`. Fix by stating one denominator per table, labelling
  the §10 treatment row as the 800-document subsample, and recomputing the gap
  column from it.
- **Retires when:** every printed gap follows from the treatment mean printed in
  the same table.

### MS-006 — The 35.1% gap is a lookup-table average, not measured masked-span loss

- **Severity:** S1 · **Status:** OPEN, REFINED · **Confirmations:** 2 (Opus 5 cycle 1; Grok 4.5 cycle 2)
- **Refinement, Grok 4.5 cycle 2.** The scope as originally filed was too wide.
  Preregistration §2 *defines* the fourth matching axis as "using a token-NLL
  table frozen from a pilot run … mean per-token difficulty", and makes failure
  of "mean masked-span NLL" within 20% the reportable result. Under that
  operationalization, printing 1.9598 / 1.2712 under the label "mean masked-span
  NLL" **is entitled**, and the drafts are not misnaming the quantity.

  What survives, and keeps this at S1: no draft scopes the figures as marginal
  per-id table averages, and `PAPER-WORKSHOP.md:106-107`,
  `PAPER-MEASUREMENT.md:241` and `RETAINED-RESULTS.md:290` promote them to
  "learnable signal" removed, which is a contextual claim the table cannot
  support. The self-optimizing matcher point is unaffected and remains
  load-bearing: `build()` places by `argmin |window_mean − target_nll|` over the
  same table `match_report` then averages.
- **Target:** `corpusgen/randpos.py:204-211`, `ops/crowding/build_corpus.py:202-204,253-254`,
  against `docs/PAPER-WORKSHOP.md:110-112` and `docs/PAPER-MEASUREMENT.md` §5
- **Defect:** Both figures are means of a **marginal per-token-id table**, not
  contextual NLL under any model. `_doc_nll(ids, table)` is `table[ids]`, a
  vocabulary lookup, and `match_report` averages that lookup over masked
  positions. `nll_table.py`'s docstring is explicit that this is "a marginal, not
  a contextual, difficulty" and "a deliberate approximation", justified because
  placement happens at build time before any model has seen the document.

  That licenses using the table **to place spans**. It does not license reporting
  the table average as the learnable signal each arm removes, which is contextual
  and is what the confound argument needs. The approximation is worst exactly
  where the claim is loudest: a template token can carry high marginal entropy
  across the corpus and near-zero contextual entropy inside a fixed biography
  frame. Compounding it, `build()` selects placements by
  `argmin |window_mean − target_nll|` over the same table `match_report` then
  averages, so the matcher optimizes the statistic the report scores. That makes
  35.1% a sound measurement of *feasibility under the table* and an unsound one of
  *signal removed*, and both drafts use it as the second.
- **Why checks miss it:** every check in the chain is internally consistent, which
  is the charter's canonical defect shape. Nothing compares the table average
  against a contextual measurement because no contextual measurement exists.
- **Check:** one forward pass, no training. Teacher-force a few hundred held-out
  fact documents under any SUP checkpoint and compute mean contextual NLL at
  FACTMASK positions and at RANDPOS positions. Compare to 1.9598 / 1.2712. If they
  agree within a few percent the finding retires; if not, report the contextual
  pair as the headline and keep the table pair as the placement diagnostic.
- **Retires when:** contextual and marginal masked-span NLL agree, or the drafts
  scope these numbers as table statistics.

### MS-007 — The difficulty table's leak guard protects the data channel, not the model channel

- **Severity:** S1 · **Status:** OPEN · **Confirmations:** 2 (Opus 5 cycle 1; Grok 4.5 cycle 2)
- **Target:** `ops/crowding/nll_table.py:26-32,142-147` against
  `docs/PAPER-WORKSHOP.md:208-211` and `docs/RETAINED-RESULTS.md:298-299`
- **Defect:** `--seed` is validated against `BURNED_PILOT_SEEDS`, but it controls
  only `bios.generate_records(entities, seed)`, which is *which entity records get
  scored*. The checkpoint comes from `--run` / `--ckpt` and is unguarded. So the
  enforced constraint is data disjointness while the admitted violation is model
  non-disjointness: `RETAINED-RESULTS.md:298` records the table as frozen from
  "the step-ladder run's step-1,564 snapshot", and the workshop limitations concede
  it "was frozen from an early checkpoint of the run later analysed". The code's
  error message will read as satisfied in exactly that case.

  Separately, the drafts assert "the gap between arms is unbiased" and do not show
  it. FACTMASK spans are selected by role and their table values are unselected.
  RANDPOS spans are selected to agree with a noisy estimate of difficulty, so they
  carry a winner's-curse component that does not cancel in the difference. State
  the direction rather than asserting unbiasedness.
- **Why checks miss it:** the seed guard fires loudly on the channel it covers,
  which makes the uncovered channel look covered. The provenance sidecar records
  `checkpoint` and `checkpoint_sha256` faithfully and nothing reads them back.
- **Check:** make `nll_table.py` warn or refuse when `--run` resolves to a run
  whose seed is not burned, and read the shipped table's sidecar to record which
  run produced it. For the bias claim, place a second control from a table built
  on a genuinely disjoint checkpoint and compare gaps.
- **Retires when:** the shipped table is shown to come from a burned-seed run, or
  the drafts state the bias direction instead of asserting unbiasedness.

### MS-008 — Workshop limitations describe a table the workshop draft does not contain

- **Severity:** ~~S0~~ → ~~S1~~ → **S3** · **Status:** OPEN, REFINED · **Confirmations:** 2 (Grok 4.5 cycle 1; GPT 5.6 Sol cycle 2)
- **Second downgrade, GPT 5.6 Sol cycle 2, accepted.** The mismatch is real and
  the S1 rationale was overstated, including mine. Section 3 plainly labels its
  displayed table as 1,500 items at 45% of training, and the placeholder plainly
  says the final table is absent, so a reader is not left unaware that the result
  is mid-training or that scoring is pending. The limitations sentence copied the
  8–17 count from the long draft, which is an internal document inconsistency and
  nothing more. Also, on 2026-08-05 a promise to complete on 2026-08-05 is not yet
  provably overdue, so the placeholder half of this finding should not be pressed
  until 2026-08-06.
- **Target:** `docs/PAPER-WORKSHOP.md:87-97`, `:206-207`; `docs/RETAINED-RESULTS.md:328`
- **Defect:** §8 says "Section 3's provisional numbers have 8 to 17 items per cell
  and are replaced by the 1,500-item table when it lands." Section 3 contains no
  8–17-item cells. It reports the 3/64 and 60/64 decoding comparison and a
  mid-training **n=1,500** per-op run (100.0 / 98.2 / 88.3 / 73.1). The 8–17-item
  table is in `PAPER-MEASUREMENT.md` §4. The limitation therefore disclaims a
  weakness the draft does not have while leaving the one it does have
  (mid-training, single checkpoint) undisclaimed.

  Related and still true: the `[PLACEHOLDER: final-checkpoint per-operation table
  ... from the scoring run completing 2026-08-05]` is unresolved, `RETAINED-RESULTS.md`
  §8 still says the final-checkpoint table "is not yet written", and no stepladder
  `evals/step*.json` appears in `outputs/RETAINED-SHA256SUMS`.
- **Severity note:** filed as S0, reduced to S1. The placeholder is explicitly
  labelled and the limitations section is candid, so a reader is warned. The
  defect is that the warning points at the wrong table.
- **Why checks miss it:** no test fails on an unresolved `PLACEHOLDER` past its
  stated date, and nothing checks that a limitation describes content actually
  present in the section it names.
- **Check:** `rg -n "PLACEHOLDER|8 to 17|1,500" docs/PAPER-WORKSHOP.md`. Repoint
  the limitation at the mid-training single-checkpoint caveat, and add a test that
  fails on a `PLACEHOLDER` whose stated completion date has passed.
- **Retires when:** the limitation matches §3's actual content and the placeholder
  is resolved or re-dated.

### MS-009 — The hash manifest does not cover the post-fix headline numbers

- **Severity:** S2 · **Status:** OPEN · **Confirmations:** 2 (Grok 4.5 cycle 1; GPT 5.6 Sol cycle 2)
- **Cross-check:** confirmed as filed, and strengthened.
  `git ls-files 'runs/stepladder_d40m_std/**' 'outputs/**'` filtered for the
  post-fix artifacts returns only `outputs/pilot/stage_a_report.json`, so the run
  tree is not committed at all. FarmShare may hold the artifacts and that would
  not retire the finding, because a replicator with this tree still cannot verify
  the claims from the manifest the catalogue advertises.
- **Target:** `outputs/RETAINED-SHA256SUMS`, against `docs/RETAINED-RESULTS.md`
  §8–§9 and the reproduction sections of both drafts
- **Defect:** All 110 hashed paths exist, and none is under
  `runs/stepladder_d40m_std/**` or contains the post-fix 3/64 vs 60/64
  measurement, the step-14,076 per-op table, Gate 0 at 1.9732, or the
  −112.90 / −112.29 storage cohorts. §7's RANDPOS gap points at FarmShare
  `high-e200` without a hashed artifact embedding 1.9598 / 1.2712. The workshop
  draft asserts "Every number above is catalogued with its artifact", which is
  true of the catalogue rows and not of the manifest they advertise. An outside
  replicator holding only this tree cannot verify the paper's central post-fix
  numbers.
- **Why checks miss it:** manifest integrity is "file present", not "every citable
  row has a hashed producer". Sections 8–11 were written after the hash freeze.
- **Check:** `rg -n "stepladder|1\.9732|112\.90|1\.9598" outputs/RETAINED-SHA256SUMS`
  returns nothing. Pull the artifacts off FarmShare, commit them, and extend the
  manifest, or add a test asserting every RETAINED section names at least one
  hashed path.
- **Retires when:** §7–§9 figures have committed hashed artifacts readable without
  cluster access.

### MS-010 — Superseded endpoint documents carry no banner and are still linked as current

- **Severity:** S2 · **Status:** OPEN, SCOPE NARROWED · **Confirmations:** 2 (Grok 4.5 cycle 1; GPT 5.6 Sol cycle 2)
- **Narrowing, GPT 5.6 Sol cycle 2, accepted.** Only `docs/THEORY-ENDPOINT.md`
  survives as a target. Its first 20 lines carry no supersession notice and it
  opens by presenting the Stage A STOP as measured fact. The rest of the filed
  scope does not hold: `RUNBOOK.md:222` says in bold "That STOP is now withdrawn
  entirely" immediately after its link, and `RUNBOOK.md:335-337` and
  `PREREGISTRATION.md:256` reference `NO-GO-PAPER.md` conditionally for a future
  gate, which is legitimate. `HANDOFF.md` is handled under MS-004.

  **`outputs/pilot/stage_a_report.json` is explicitly removed as a target, and the
  reasoning is worth keeping.** It is a raw measurement artifact and its recorded
  `"decision": "STOP"` should stay byte-faithful. Supersession belongs in a banner
  on the consuming document or in sidecar metadata, never in a rewrite of the
  measurement. Do not patch that file.
- **Target:** `docs/THEORY-ENDPOINT.md:1-31`
- **Defect:** `RESULTS-2026-08-04.md` declares it supersedes `THEORY-ENDPOINT.md`
  "in full" and the STOP in `stage_a_report.json`. Neither carries a
  top-of-file notice. `THEORY-ENDPOINT.md`'s op-profile cells are pre-fix
  generative accuracies. `stage_a_report.json` still reads `"decision": "STOP"`
  and directs the reader to write the NO-GO paper. The handoff reading order and
  the runbook still point at both as live guidance.
- **Why checks miss it:** no test requires a supersession banner when a results
  note names a document as superseded.
- **Check:** `rg -n "THEORY-ENDPOINT|NO-GO-PAPER" HANDOFF.md ops/crowding/RUNBOOK.md docs/PREREGISTRATION.md`
  and read the first 20 lines of `THEORY-ENDPOINT.md`. Add banners; add a test
  that any document named as superseded in a results note opens with a notice.
- **Retires when:** superseded documents open with a pointer to
  `RESULTS-2026-08-04.md`.

### MS-011 — The occupancy ladder relabels prompt accessibility as parameter storage

- **Severity:** S0 · **Status:** OPEN · **Confirmations:** 1 (GPT 5.6 Sol, lane B, cycle 1)
- **Target:** `ops/crowding/occupancy.py:88,107-167`; `evals/storage.py:1-20`;
  `docs/PAPER-MEASUREMENT.md:374-417`
- **Defect:** `evals/storage.py` opens by refusing the word: "It is deliberately
  NOT called 'storage': prompt-conditioned NLL cannot separate what a model has
  stored from what it can be induced to emit, and the previous accuracy-to-bits
  conversion conflated the two badly enough that its ledger had to be withdrawn."
  `occupancy.py:88` then assigns `"stored_bits_per_param": s.get("recoverable_bits_per_param")`
  and `classify()` issues capacity-mechanism verdicts off the shape of the renamed
  quantity, up to and including "The model declines to learn facts rather than
  compressing them ... **Premise 1 of Memory Split fails**, and the arm contrast is
  not worth running."

  This is the charter's defect shape in its purest form, and the docstring it
  contradicts names the exact prior withdrawal being re-committed. A flat
  trained-minus-unseen difference under one held-out phrasing is equally
  consistent with mappings that are present but inaccessible under that phrasing.
  `probe_control.py` exists precisely to settle that with a training-phrasing
  positive control, and no retained artifact shows it has been run, so the long
  paper reaches the abandonment conclusion before its instrument has been
  validated.
- **Why checks miss it:** `tests/test_occupancy.py` writes synthetic values under
  the key `recoverable_bits_per_param` and asserts the renamed storage signatures
  fire, so the test codifies the substitution instead of challenging it.
  `tests/test_storage.py` checks accounting against an oracle stub that reads the
  target from the teacher-forced tensor and never calibrates the held-out prompt
  on a checkpoint known to contain the mappings. `tests/test_probe_control.py`
  hard-codes hypothetical gaps.
- **Check:** on the final checkpoint, run
  `PYTHONPATH=. $MS_PY ops/crowding/probe_control.py --run $MS_ROOT/runs/stepladder_d40m_std --ckpt .../step0031280.pt --n-entities 300 --device cuda`
  and do not read any occupancy signature until it reports the trained-minus-unseen
  gap under both training and held-out phrasing. Separately, either rename
  `stored_bits_per_param` back to `recoverable_bits_per_param` throughout
  `occupancy.py` or gate `classify()` behind a passed positive control.
- **Retires when:** a known-positive memorizing checkpoint separates trained from
  unseen entities under these templates, and the final checkpoint shows no
  separation under training phrasing either.

### MS-012 — RANDPOS silently abandons the length and position matches that define it

- **Severity:** S0 · **Status:** OPEN · **Confirmations:** 1 (GPT 5.6 Sol, lane B, cycle 1)
- **Target:** `corpusgen/randpos.py:94-151,154-212`;
  `ops/crowding/build_corpus.py:291-312,527-657`; `tests/test_factlane.py:157-173`;
  `docs/RETAINED-RESULTS.md:378-402`
- **Defect:** The control is defined by matching on count, span length, and
  within-document relative position. Three separate paths drop two of those three
  without failing anything:

  1. When no placement lies inside `position_tolerance`, `build` falls back to
     **any** legal placement. The comment at `:114` says "nearest in relative
     position", but nothing sorts or selects by position, and on the production
     path (`token_nll` supplied) the choice is `argmin |window_mean − target_nll|`,
     which is NLL only. Position matching is abandoned entirely and silently.
  2. When no contiguous home exists, the span's tokens are scattered.
  3. Even when every span is placed contiguously, `taken` prevents overlap but not
     **adjacency**, so two abutting placements merge into one longer zero run and
     `spans_of` no longer recovers the requested length histogram.

  Executing the real placement on 800 shipped-format biographies with a
  structured surrogate table produced 336/800 documents with a different
  span-length histogram, two above the 0.15 mean position gap, and 768 of 4,800
  fact spans with no exact-length position-valid pairing available. The frozen
  `high-e200` table is not local, so the published row is **unverified rather
  than disproved** — but it is labelled "shipped: contiguous, length- and
  position-matched" in the central control table of the paper.
- **Why checks miss it:** `test_randpos_matches_the_span_length_histogram` and
  `test_randpos_matches_relative_position` each exercise one hard-coded document,
  and the latter compares only the difference of two means with a 0.20 tolerance
  against a 0.15 placement tolerance. Corpus verification asserts equal zero count
  and no value overlap, both of which survive all three paths. The builder does
  compute `frac_documents_length_matched` on a sample and neither `verify` nor
  `report_findings` reads it, which is the manifest-field-nobody-read failure
  again.
- **Check:** with the frozen `high-e200` `nll_table.npy` and the manifest seed,
  regenerate the 800 fact documents underlying §10 and per document assert
  `match_report(...)["length_histogram_matched"]`, count how often the fallback at
  `randpos.py:113` and the scatter at `:119` fire, and require every control span
  within 0.15 relative position of its source. Report all three counts beside the
  §10 table.
- **Retires when:** that run yields zero fallbacks, zero scatters, a 100% length
  match, and zero out-of-tolerance assignments.
- **Interacts with:** MS-005 and MS-006. All three concern the same table, and if
  this one reproduces, the row label in §10 is wrong as well as its denominator.

### MS-013 — Gate 0 averages microbatch means rather than fact-value tokens

- **Severity:** S1 · **Status:** OPEN · **Confirmations:** 1 (GPT 5.6 Sol, lane B, cycle 1)
- **Target:** `train/trainer.py:146-167`; `docs/RETAINED-RESULTS.md:338-345`;
  `docs/PAPER-MEASUREMENT.md:363-366`
- **Defect:** `loss_masked_values` ends `return sum(losses) / len(losses)`, where
  each entry is `self.model(xb, yb)`'s default mean reduction over the live
  targets in that microbatch. Microbatches carrying different numbers of
  fact-value targets therefore get equal weight, so the published 1.9732 nats is a
  mean of microbatch means and not the per-token mean the papers name. A two-chunk
  execution with 1 and 8 live targets returned 4.133627 against 4.150990 for a
  single pass over all nine.

  This is a direct descendant of the `reduction="mean"` defect that withdrew
  generation 3, surviving in the probe path because the fix was applied to the
  training path. The qualitative claim (1.97 against 25.03) very likely survives
  reweighting; the published quantity has not been computed as described.
- **Why checks miss it:** `tests/test_weighted_loss.py` pins the fixed denominator
  for *training* and Gate 0 deliberately uses the unweighted model path.
  `test_loss_decreases_and_logs` checks only that the field exists, and
  `test_masked_value_probe` only that some labels survive. Nothing varies target
  count across probe microbatches.
- **Check:** add a regression test with two probe chunks of unequal live-target
  count asserting `loss_masked_values()` equals `sum(per_token_ce) / n_live`, then
  recompute 1.9732 from one global numerator and denominator on the production
  probe.
- **Retires when:** the token-weighted recomputation agrees with the published
  figure, or the figure is restated.

### MS-014 — Both papers list "save raw generations" as a check they run, and the scorer discards them

- **Severity:** S1 · **Status:** OPEN · **Confirmations:** 1 (GPT 5.6 Sol, lane B, cycle 1), severity raised by orchestrator
- **Target:** `evals/scorers.py:37-72`; `tests/test_run_evals.py:106-113`;
  `docs/PAPER-WORKSHOP.md:220-227`; `docs/PAPER-MEASUREMENT.md:482-486`
- **Defect:** `PAPER-WORKSHOP.md`'s "Appendix: checks we now run" item 2 reads
  "Save generations, not only parsed answers. Ours kept the parse and discarded
  the text, so four rounds of invented documents left no trace in the logs."
  `score_items` computes `gen`, parses it into `pred`, and drops it. The persisted
  row is `{qid, task, correct, pred, answer, meta}` with no continuation text, and
  a retained `igsm.jsonl` row has exactly those keys. The safeguard is asserted in
  the present tense in both drafts and is not implemented, so the invented-document
  failure would again leave no trace.
- **Severity note:** filed S2, raised to S1. This is not an artifact gap, it is a
  false statement in a paper whose central contribution is a checklist of cheap
  checks. A reviewer who runs the checklist against the repository finds item 2
  missing.
- **Why checks miss it:** `test_per_item_rows_are_saved_for_post_hoc_breakdowns`
  requires only `answer` and `pred`, which is the pre-fix artifact shape.
- **Check:** call `score_items` with a scripted continuation containing reasoning
  plus `Answer: 19`, save through `save_results`, and assert the JSONL row carries
  a verbatim `generation` field equal to the whole continuation.
- **Retires when:** generations are persisted under a documented field and the
  appendix matches the code.

### MS-015 — The documented test baseline and virtualenv are both stale

- **Severity:** S3 · **Status:** OPEN · **Confirmations:** 1 (GPT 5.6 Sol, lane B, cycle 1)
- **Target:** `HANDOFF.md:231-242`
- **Defect:** The handoff states "Expected test result in this package: 389 pass,
  4 fail" and "In the live repository ... 392 pass and 1 fail". The suite actually
  collects 434 cases and returns **433 passed with the one intentional failure**.
  It also documents `.venv/bin/python`, and no `.venv` exists in the working tree.
  A reviewer using the documented baseline as a sanity check would conclude
  something had gone wrong.
- **Why checks miss it:** the counts are prose in a document, and nothing asserts
  them against a real run.
- **Check:** `PYTHONPATH=. python3 -m pytest tests -q` and update the two counts,
  or add a test that parses the claimed baseline out of `HANDOFF.md` and compares
  it to `--collect-only -q`.
- **Retires when:** the documented counts match a real run and the setup section
  names an environment that exists.

### MS-016 — Every campaign ceiling was sized at a throughput the guard refuses to use

- **Severity:** S1 · **Status:** OPEN · **Confirmations:** 1 (Opus 5, lane B, cycle 2)
- **Target:** `docs/PREREGISTRATION.md:288-299,318-323`;
  `ops/crowding/guard.py:56,143-148,219-224,240-253`; `HANDOFF.md:307-313`
- **Defect:** §10a mandates that "Projections are made from measured throughput
  only" and that the guard "budgets at the measured `d40m` rate". The guard obeys,
  using `MEASURED_TOK_S = 184_671`, which gives **32.09 GPU-h** for a 21.3B-token
  run. The campaign ceilings in the same frozen block, and the cost table in
  `HANDOFF.md`, were derived at **20.76 h per run**, which is the extrapolated
  ~277,000 tok/s figure that §10a itself says has never been observed. Nothing
  reconciles them, and every remaining campaign is undersized as a result:

  ```
  campaign    runs     need  ceiling    soft   claimable
  stage_b        2       64       60      54    1/2   SHORT by 1
  stage_c        9      289      220     198    6/9   SHORT by 3
  matrix        72     2311     1600    1440   45/72  SHORT by 27
  ```

  Stage B is the immediate next action in both branches of the plan and only one
  of its two runs can be authorised. For the matrix,
  `scripts/analyze_crowding.py::require_complete` refuses a matrix with a dropped
  seed, so the 45 authorised runs are not a partial result but 1,444 GPU-hours of
  unanalysable spend. That is the exact outcome §10a's soft stop was written to
  prevent.

  Every part of this is individually correct and documented. The conservative
  projection rule is right, and the ceiling is frozen where it should be. What is
  missing is any check that the ceilings were derived under the rule the guard
  enforces.
- **Why checks miss it:** `tests/test_guard.py` exercises the mechanism against
  synthetic ceilings and claim sequences. Nothing instantiates the real
  preregistered ceiling against the real campaign shape and asks whether the plan
  fits. From the guard's side both numbers are inputs.
- **Check:**
  `PYTHONPATH=. python3 -c "from ops.crowding.guard import *; c=Ceiling.from_prereg(); print(projected_hours(21_335_900_160), c.for_campaign('stage_b'))"`
  → `32.09  60.0`. Add a test that for each named campaign multiplies its planned
  run count by `projected_hours(total_tokens)` and asserts the product is below
  `soft_stop_fraction * ceiling`.
- **Note on the fix:** §10a is frozen, so correcting the ceilings requires an
  amendment that *raises* resource ceilings after discovering the work does not
  fit. That is defensible here, because the cause is a rate mismatch found before
  any campaign ran rather than an overrun found mid-spend, but it must be filed
  with the derivation shown rather than edited quietly.
- **Retires when:** the ceilings are re-derived at `MEASURED_TOK_S` by amendment,
  or the campaigns are re-planned to fit.

### MS-017 — The guard's release path never returns checkpoint bytes

- **Severity:** S2 · **Status:** OPEN · **Confirmations:** 1 (Opus 5, lane B, cycle 2)
- **Target:** `ops/crowding/guard.py:204-217,283-288,358-361,394-396`
- **Defect:** `release(corpus, checkpoint_bytes=0)` subtracts whatever the ledger
  record carries, but the CLI cannot supply one. The `release` subparser declares
  only `--campaign`, `--corpus` and `--ledger`, and `main` calls
  `guard.release(args.corpus)`, so every CLI release writes `checkpoint_bytes: 0`
  and `resident_bytes`'s `ckpt` accumulator never decreases. Each claim adds 487 MB
  and nothing gives it back: about 35 GB of phantom scratch across 72 matrix runs,
  against a 150 GB ceiling that also holds an 85 GB corpus. The guard will refuse
  a legitimate build before the matrix ends. `release`'s own docstring warns about
  precisely this failure and the CLI makes it unavoidable for checkpoints.
- **Why checks miss it:** `release` is correct as a Python API. No test drives the
  argparse path, so the gap between the signature and the CLI flags is invisible.
- **Check:** add `--checkpoint-bytes` to the subparser, or have `release` look up
  the claim record for that corpus and subtract the bytes it recorded. Assert that
  a claim-then-release cycle returns `resident_bytes()` to its starting value.
- **Retires when:** a CLI claim-then-release round-trips to zero resident bytes.

### MS-018 — Five of six preregistered gates decide on point estimates, and one does not exist

- **Severity:** S0 · **Status:** OPEN · **Confirmations:** 1 (Grok 4.5, lane A, cycle 2)
- **Target:** `docs/PREREGISTRATION.md:237` against `evals/stats.py:295-340` and
  `scripts/analyze_crowding.py:154-163`
- **Defect:** §7 closes with a one-line requirement: "All gates use confidence
  bounds, not point estimates." `check_gates` implements five gates and exactly
  one of them, `sup_carries_a_burden`, receives a bound
  (`bits_lower_bound["sup"]`). The other four decide on cross-seed point means:
  `bits.get("factmask") < 0.10 * bits.get("sup")`,
  `igsm_band[0] <= igsm_acc_sup <= igsm_band[1]`,
  `abs(bits["randpos"] - bits["sup"]) <= 0.20 * bits["sup"]`, and the clip-ratio
  spread.

  Two further omissions in the same function. §7's sixth gate, "Per-arm mean
  unmasked-token NLL, so the entropy asymmetry between the arms is visible rather
  than inferred", is absent entirely, so the gate that would expose the very
  confound Section 4 of the paper is about cannot fire. And §7.3's two monotone
  clauses (decreasing in `op`, increasing over training) exist only in the pilot
  path, not the confirmatory one, so `endpoint_is_measurable` checks the accuracy
  clause alone.

  A matrix can therefore return `validated` or `invalid` on point estimates the
  frozen document forbids, across five of six gates.
- **Why checks miss it:** `tests/test_stats_crowding.py` feeds hand-built point
  dicts and asserts the validity/reporting split and the three-way verdict.
  Nothing asserts that a gate's `observed` field is a bound, and nothing asserts
  the gate count.
- **Check:** `python3 -c "import inspect, evals.stats as s; src=inspect.getsource(s.check_gates); print('bounds:', src.count('lower_bound'), 'gates:', src.count('_gate('), 'unmasked:', 'unmasked' in src.lower())"`
  → one bound, five gates, no unmasked-NLL gate. Fix by threading per-seed vectors
  into every gate and comparing bounds, and by implementing gate 6.
- **Retires when:** every §7 gate decides on a bound computed across seeds, and a
  sixth gate reports per-arm unmasked-token NLL.

### MS-019 — The §5 primary scoring band is never instantiated, so the estimand is not the preregistered one

- **Severity:** S0 · **Status:** OPEN · **Confirmations:** 1 (Grok 4.5, lane A, cycle 2)
- **Target:** `docs/PREREGISTRATION.md:153-161,225-226` against
  `scripts/analyze_crowding.py:136-165` and `evals/stats.py:325-328`
- **Defect:** §5 freezes the primary scoring band as "the contiguous run of `op`
  values whose SUP accuracy falls inside [0.17, 0.85]", tagged `[RULE]` and
  instantiated by the SUP arm of the first run on the [1, 8] corpus. Ops outside
  it "are reported but excluded from the primary estimand, because an op at
  ceiling or floor cannot move and averaging it in only shrinks the effect toward
  zero."

  The analyzer never does this. `rg "igsm_by_op|contiguous|scoring.band" scripts/analyze_crowding.py evals/stats.py`
  returns nothing. `analyse` reads `summary.json["igsm_acc"]`, an aggregate over
  whatever `igsm_op` the run config set, never reads `igsm_by_op` (which
  `run_evals.py` does produce and `ladder.py` does consume), never computes a
  contiguous in-band run, and never restricts `Y[FACTMASK] − Y[RANDPOS]` to those
  ops. `check_gates`'s `endpoint_is_measurable` then compares that aggregate
  against `--igsm-band`, which is §5's *selection interval* being reused as a
  threshold on aggregate accuracy. Those are different quantities.

  The consequence is specific to the decision taken on 2026-08-05. The corpus was
  widened to op [1, 8] precisely because op 1–4 sat at 93.8% with no headroom. An
  aggregate over [1, 8] can sit comfortably inside [0.17, 0.85] while the primary
  estimand still averages the ceiling ops the `[RULE]` exists to exclude, which is
  the exact effect-shrinking the rule was written to prevent.
- **Settles a carried-forward question, in the worse direction.** Cycle 1 left
  open whether §7.3's first clause was circular against §5. It is not circular,
  because the code never instantiates the band from accuracy and then re-checks it.
  The gate and the estimand simply ignore the rule. Unimplemented rather than
  tautological.
- **Why checks miss it:** tests pass synthetic scalar `igsm_acc` values inside
  (0.17, 0.85). Nothing builds a per-op curve, freezes a contiguous band, and
  asserts the contrast uses only those ops.
- **Check:** the `rg` above returns no hits. Fix by deriving the contiguous op set
  from the first SUP matrix seed's `igsm_by_op`, writing it into the
  preregistration or a dated sidecar before any FACTMASK or RANDPOS run is scored,
  and computing both the primary contrast and §7.3 on that set.
- **Retires when:** the analyzer derives, records, and applies the band.

### MS-020 — The pilot's monotone gate treats missing evidence as a pass

- **Severity:** S1 · **Status:** OPEN · **Confirmations:** 1 (Grok 4.5, lane A, cycle 2)
- **Target:** `ops/crowding/analyze_pilot.py:58-62,137-143` against
  `docs/PREREGISTRATION.md:247`
- **Defect:** GO/NO-GO item 2 requires accuracy inside the band **and** monotone
  decreasing in `op` **and** monotone increasing over training. The gate tests
  `monotone_in_op is not False` and `monotone_over_training is not False`. When
  `acc_by_op` has fewer than two ops, or `acc_over_training` fewer than two
  points, Stage A sets those fields to `None`, and `None is not False` is `True`.
  A cell that never measured a curve contributes a vacuous pass on two of three
  conjuncts. Reproduced with `monotone_in_op None`, `monotone_over_training None`,
  `decision GO`. The failure path is fine when the curve exists; the defect is the
  silent pass under missing evidence, which is the charter's shape again.
- **Why checks miss it:** the healthy-GO fixture in `tests/test_pilot.py` supplies
  a two-op decreasing curve and a two-point training curve. Nothing asserts that
  `None` fails.
- **Check:** feed a single-op Stage A cell and expect NO-GO, or an explicit
  `monotone_unmeasured` failure, rather than GO.
- **Retires when:** `None` fails the gate, or Stage A refuses to emit a chosen cell
  without both monotone measurements.

### MS-021 — The occupancy classifier's decision thresholds are not frozen anywhere

- **Severity:** S1 · **Status:** OPEN · **Confirmations:** 1 (GPT 5.6 Sol, lane C, cycle 2)
- **Target:** `ops/crowding/occupancy.py:54-59,119-159`; `docs/THEORY-CAPACITY.md:77-94`;
  `docs/PREREGISTRATION.md:84-110,241-246`; `ops/crowding/RUNBOOK.md:115-126`
- **Defect:** `classify()` decides between "inert", "abandonment", "saturating" and
  "linear" on three constants: `INERT = 0.02`, `ABANDON_FRAC = 0.20`,
  `BEND_FRAC = 0.20`. Each verdict prescribes a different scientific action, up to
  "Premise 1 of Memory Split fails, and the arm contrast is not worth running."
  None of the three appears in the theory note, the runbook, or the frozen
  preregistration, and `occupancy.py` and its tests are **untracked**, so there is
  no committed pre-outcome provenance for any of them. The verdicts are only
  meaningful under the unstated assumption that the cutoffs were fixed before the
  ladder was observed, and nothing in the repository establishes that.
- **Why checks miss it:** `tests/test_occupancy.py` feeds synthetic rows that sit
  comfortably on each side of each threshold and asserts the resulting label. It
  never compares the constants against a dated declaration. The theory tests
  rederive F/C arithmetic, not shape thresholds.
- **Check:** `rg -n "INERT|ABANDON_FRAC|BEND_FRAC" docs/ ops/crowding/RUNBOOK.md`
  returns nothing outside `occupancy.py`. Add the three values and their boundary
  conventions to a dated preregistration amendment, then make a test parse the
  document and compare.
- **Retires when:** a dated frozen document names all three and predates every
  occupancy result.
- **Interacts with:** MS-011. That finding says the occupancy ladder measures the
  wrong quantity; this one says its decision thresholds are unfrozen. Both must
  close before any occupancy verdict is citable.

### MS-022 — The PopQA record advertises a deleted harness as committed and reproducible

- **Severity:** S2 · **Status:** OPEN · **Confirmations:** 1 (GPT 5.6 Sol, lane C, cycle 2)
- **Target:** `docs/POPQA-HELDOUT-KEY.md:3-12,57-138`; `docs/RETAINED-RESULTS.md:211-235`;
  `tests/test_repo_state.py:162-168`
- **Defect:** The document states "The lost harness was recreated ... and is now
  committed" and gives reproduction commands. None of the named files is tracked
  or present:

  ```
  $ git ls-files -- corpusgen/realfact.py evals/keyguess.py evals/constrain.py \
      scripts/run_keyguess_local.py scripts/analyze_keyguess_policy.py \
      cluster/slurm/keyguess_cpu.sbatch
  (no output)
  $ ls corpusgen/realfact.py evals/keyguess.py
  No such file or directory
  ```

  `RETAINED-RESULTS.md` §6 withdraws the experiment for exactly this reason, and
  the document carries no withdrawal banner and still presents dead commands and
  withdrawn measurements as current. The data artifacts and the pinned input are
  tracked and the input hash matches, so the defect is the harness and the framing,
  not the retained arithmetic.
- **Why checks miss it:** `POPQA-HELDOUT-KEY.md` is in `FINGERPRINT_EXEMPT` as the
  primary record of a withdrawn experiment, so the resurrection test is told to
  ignore it. The deleted-module test checks imports in extant Python source, not
  reproduction commands in prose, and `test_docs_holds_only_the_current_documents`
  affirmatively requires the file to remain.
- **Check:** the `git ls-files` above prints nothing. Fix by adding a withdrawal
  banner and removing the claim that the commands reproduce the result, or by
  restoring the harness.
- **Retires when:** the harness is restored and reruns under the fixed decoder, or
  the document opens with a banner and drops the reproduction claim.

### MS-023 — The RUNBOOK builds one corpus and then points Stage B and C at another

- **Severity:** S2 · **Status:** OPEN · **Confirmations:** 1 (GPT 5.6 Sol, lane C, cycle 2)
- **Target:** `ops/crowding/RUNBOOK.md:288-304,320-324,363`; `ops/crowding/pilot.py:106-186,206-238`
- **Defect:** The Stage B block builds `corpora/high-e200` and `corpora/nofact` at
  996,408 entities and 21,335,900,160 tokens. The next command in the same block
  configures Stage B against a different corpus with different numbers, and
  Stage C repeats them:

  ```
  290: MS_OUT=.../corpora/high-e200 MS_ENTITIES=996408 MS_EXPOSURES=200
  291: MS_TOKENS=21335900160
  302:   --corpus corpora/high --nofact-corpus corpora/nofact
  303:   --total-tokens 16399769600 --lr 1.5e-3 --entities 382900
  322:   --out $MS_ROOT/configs/pilotC --corpus corpora/high
  323:   --total-tokens 16399769600 --lr 1.5e-3 --entities 382900
  ```

  `corpora/high` is the *superseded* corpus that `HANDOFF.md` lists for deletion.
  `pilot.py` copies the supplied path, token count, and entity count straight into
  configs without checking them against the corpus manifest. So the commands
  either target a corpus that is about to be deleted, or train the newly built
  NOFACT corpus for a different number of steps, which defeats the equal-token
  comparison Stage B exists to make. Every flag parses and every shell file passes
  `bash -n`.
- **Why checks miss it:** syntax validation is green, and the config generator
  tests argument propagation rather than cross-checking `targets.bin` and
  `manifest.json` against `--total-tokens` and `--entities`. A semantically wrong
  but well-formed command survives.
- **Check:** the `rg` above shows both triples in one block. Fix by having
  `pilot.py` derive token and entity counts from the corpus manifest rather than
  accepting them as flags, which removes the class of error rather than this
  instance.
- **Interacts with MS-016, and changes its Stage B row.** At the runbook's stale
  16.4B tokens the guard projects 24.67 h per run, so Stage B's two runs total
  49.3 h and *do* fit the 60 h ceiling. At the intended 21.3B they do not. Stage
  B's shortfall in MS-016 is therefore contingent on which corpus is actually
  used. The matrix fails under both (72 × 24.67 = 1,776 against 1,600).
- **Retires when:** the Stage B and C commands name the corpus they build and take
  their token and entity counts from its manifest.

### MS-024 — THEORY-CAPACITY presents a superseded ladder as the corpus in flight

- **Severity:** S3 · **Status:** OPEN · **Confirmations:** 1 (GPT 5.6 Sol, lane C, cycle 2)
- **Target:** `docs/THEORY-CAPACITY.md:27-47,98-110` against
  `docs/PREREGISTRATION.md:99-108` and `ops/crowding/RUNBOOK.md:101-126`
- **Defect:** The note says "The corpus in flight is 2x over-saturated" at
  1,531,800 entities and 100 exposures, with rungs at 765,900/200 and 382,950/400,
  and calls the top corpus "already building". The live preregistration and
  runbook define the operating point as 996,408 entities at 200 exposures with
  F/C = 1.000 and rungs 249,102/800, 498,204/400, 996,408/200, and the runbook says
  no rung above F/C = 1 fits. The derivation is arithmetically correct for the
  ladder it describes; that ladder is not the current one.
- **Why checks miss it:** `tests/test_theory_capacity.py` rederives whichever table
  it is given. Nothing binds the prose's entity/exposure tuple to the canonical one
  in the preregistration.
- **Check:** `rg -n "1,531,800|996,408|F/C = 2" docs/THEORY-CAPACITY.md docs/PREREGISTRATION.md`
  shows both ladders. Relabel the note as historical, or update its tuple.
- **Retires when:** the note names the current operating point, or labels its own
  as superseded.

---

## Cross-model disagreements

None yet. MS-005 is the opposite: two models reached different partial readings of
the same table and the merge is stronger than either, which is the intended
behaviour of the rotation.

---

## Graveyard — filed, verified, and rejected

Recorded so no later cycle re-files them.

### A-02 (Opus 5, cycle 1) — "the 20.0% row is misrounded and should be 20.1%"

**Rejected as filed; merged into MS-005.** The finding computed the gap against
§7's full-corpus treatment mean of 1.9598 when §10's table uses its own
800-document mean of 1.9584, against which 20.0% is the correct rounding. The
substantive residue — that the oracle fails `rel <= 0.20` under either
denominator, and that the table is irreproducible from any single one — survives
in MS-005 at S1. Grok 4.5's lane C report surfaced the two-denominator split
that corrected it.

### C-07 (Grok 4.5, cycle 1) — "adjacent RETAINED tables disagree at the second decimal"

**Merged into MS-005**, not rejected. Filed as S3 internal inconsistency; the
merge shows it is the root cause of a reproducibility defect in three documents
and it carries S1 there.
