# Cycle 1 — Lane A, estimand and inference — Opus 5

**Verdict: four findings, one S0. The preregistration declares itself frozen while
the primary load remains unnamed, and Section 4's headline number is a lookup-table
average that the papers describe as measured model loss.**

---

### A-01. The preregistration is banner-frozen with the primary load still unnamed

- **Severity:** S0
- **Target:** `docs/PREREGISTRATION.md:3` and `:361-368`, against
  `docs/AMENDMENT-2026-08-02.md:171-172`
- **Claim under attack:** "**Status: FROZEN. All former `[PILOT]` slots are now
  closed, 2026-08-05.** Nothing in this document may change on the basis of an
  outcome."
- **Defect:** The identity of the primary load is the single most consequential
  free parameter in the design and it is named nowhere in the document. Amendment 1
  says so in as many words: "**Still to be frozen before the matrix runs.** The
  identity of the primary load. §5 does not name it, and this amendment
  deliberately does not choose it." `scripts/analyze_crowding.py:119-125` still
  hard-refuses to run without `--primary-load` and tells the caller to "Freeze it
  in docs/PREREGISTRATION.md §5 first". §5 freezes the primary scoring *band*
  (op values) and never mentions load.

  The narrow reading is defensible, because the primary load was never tagged
  `[PILOT]` and so "all `[PILOT]` slots are closed" is literally true. That is
  what makes this dangerous rather than merely untidy. A reader who checks the
  banner concludes the design is fully specified. The estimand is
  `Y[FACTMASK] - Y[RANDPOS]` *at one load*, so an unnamed load means the estimand
  itself is unspecified, and whoever runs the matrix will supply it from the
  command line after the runs exist.
- **Why existing checks miss it:** `analyze_crowding.py` refuses to *infer* the
  load, which is the check that exists, and it cannot tell whether the value it
  is handed was frozen beforehand or chosen this morning. Nothing compares the
  argument against the document. `tests/test_stats_crowding.py` exercises the
  refusal path, not the provenance of the value.
- **Falsifiable check:**
  `rg -n "primary load|primary_load" docs/PREREGISTRATION.md` returns only §336,
  a backreference to Amendment 1, and no assignment. Then add a test that parses
  a `primary_load = <name>` declaration out of `PREREGISTRATION.md` and asserts
  `analyze_crowding.py` was invoked with that value, so the ceiling and the
  document cannot drift apart in the way §10a already insists on for the
  resource ceilings.
- **What would prove me wrong:** a line in `PREREGISTRATION.md` naming the load,
  dated before the first matrix run, or a decision that the primary estimand is
  pooled across loads (which Amendment 1 §4 forecloses).

---

### A-02. The oracle control fails the 20% tolerance and is reported as meeting it

> **Correction, added in synthesis. This finding was wrong as filed and is
> rejected; see the graveyard entry in `FINDINGS.md` and its replacement MS-005.**
> The gap below is computed against §7's full-corpus treatment mean of 1.9598,
> but the table it attacks is §10's, which uses its own 800-document mean of
> 1.9584. Against 1.9584 the printed 20.0% is the correct rounding. Grok 4.5's
> lane C report surfaced the two-denominator split. What survives is narrower and
> was found only by merging the two reports: no single denominator reproduces all
> five rows of the §10 gap column, and the oracle fails `rel <= 0.20` under
> either. Left unedited below for the audit trail.

- **Severity:** S1
- **Target:** `docs/PAPER-WORKSHOP.md:115` and `:121`,
  `docs/PAPER-MEASUREMENT.md:266` and `:273`, `docs/RETAINED-RESULTS.md:391`
- **Claim under attack:** the "best possible, length-matched" control is tabulated
  at NLL 1.5667 with a gap of **20.0%**, and the prose says an oracle "picks the
  hardest legal span every time [and] lands exactly on the tolerance boundary".
  `PAPER-MEASUREMENT.md:266` labels the row "borderline".
- **Defect:** The gap is 20.058%, not 20.0%. It rounds to 20.1% and it **fails**
  the preregistered tolerance, which `corpusgen/randpos.py:211` implements as
  `rel <= NLL_TOLERANCE` with `NLL_TOLERANCE = 0.20`. The other three rows in the
  same table round correctly under the same definition (35.136 → 35.1, 9.802 →
  9.8, 4.868 → 4.9), so this is not a definitional disagreement, it is one row
  rounded the wrong way toward the threshold it is being compared against. The
  result is robust to the reported precision: taking 1.9598 and 1.5667 at their
  widest four-decimal intervals, the gap ranges over 20.053% to 20.061% and never
  reaches 20%.

  The direction matters and it favours the paper. "Even an oracle that picks the
  hardest legal span every time cannot reach the tolerance" is a cleaner and
  stronger statement of Section 4's thesis than "lands exactly on the boundary",
  which invites the reader to think a slightly better matcher would succeed.
  The paper is currently understating its own finding by way of an arithmetic
  slip.
- **Why existing checks miss it:** the table is prose. No test recomputes a
  published gap from its published operands, and `nll_within_tolerance` is
  evaluated inside the corpus builder on the shipped control only, never on the
  three counterfactual rows.
- **Falsifiable check:**
  `python3 -c "f=1.9598; print((f-1.5667)/f)"` → `0.2005817...`. Then add a doc
  test that parses any three-column NLL/gap table out of `docs/` and asserts the
  gap column equals `(factmask_nll - row_nll) / factmask_nll` to the stated
  precision.
- **What would prove me wrong:** an artifact showing the oracle NLL is ≥ 1.5679
  rather than 1.5667, or a different gap denominator used for that row alone and
  stated somewhere.

---

### A-03. The 35.1% gap is a lookup-table average, not measured masked-span loss

- **Severity:** S1
- **Target:** `corpusgen/randpos.py:204-211`, `ops/crowding/build_corpus.py:202-204,
  253-254`, against `docs/PAPER-WORKSHOP.md:110-112` and
  `docs/PAPER-MEASUREMENT.md` §5
- **Claim under attack:** "Measured over 21.3B tokens, the treatment's masked spans
  average 1.9598 nats and the control's 1.2712", and downstream, "FACTMASK removes
  about 54% more learnable signal than RANDPOS does, by construction."
- **Defect:** Both numbers are means of a **marginal per-token-id table**, not
  contextual NLL of those spans under any model. `_doc_nll(ids, nll_table)` is
  `nll_table[ids]`, a vocabulary lookup, and `match_report` averages that lookup
  over the masked positions. `nll_table.py`'s own docstring is explicit that this
  is "a marginal, not a contextual, difficulty" and calls it "a deliberate
  approximation", justified because placement happens at build time when no model
  has seen the document.

  That justification licenses using the marginal table **to place spans**. It does
  not license reporting the resulting table average as the amount of learnable
  signal each arm removes, which is a contextual quantity and is what the confound
  argument depends on. The approximation is worst exactly where the claim is
  loudest: a template token can carry high marginal entropy across the corpus and
  near-zero contextual entropy inside a fixed biography frame, so the table
  overstates what RANDPOS actually removes and the true contextual gap is probably
  *wider* than 35.1%.

  Compounding it, `build()` selects placements by
  `argmin |window_mean − target_nll|` over the same table that `match_report` then
  averages, so the matcher optimizes the identical statistic the report scores.
  That makes 35.1% a valid measurement of *feasibility under the table* and an
  invalid measurement of *signal removed*, and the papers use it as the second.
- **Why existing checks miss it:** every check in the chain is internally
  consistent, which is the shape the charter warns about. `nll_within_tolerance`
  compares two table averages and is correct as a feasibility test. No check
  compares the table average against a contextual measurement, because no
  contextual measurement exists.
- **Falsifiable check:** one forward pass, no training. Take any SUP checkpoint and
  a few hundred held-out fact documents, teacher-force them, and compute mean
  contextual NLL at the FACTMASK positions and at the RANDPOS positions
  separately. Compare against 1.9598 and 1.2712. If contextual and marginal agree
  within a few percent the finding retires and the papers can keep the wording.
  If they diverge, report the contextual pair as the headline and keep the table
  pair as the placement diagnostic.
- **What would prove me wrong:** contextual and marginal masked-span NLL agreeing
  closely on held-out fact documents, or a sentence in the drafts already scoping
  these numbers as table statistics (I did not find one).

---

### A-04. The difficulty table's leak guard protects the data channel, not the model channel

- **Severity:** S1
- **Target:** `ops/crowding/nll_table.py:26-32, 142-147`, against
  `docs/PAPER-WORKSHOP.md:208-211`
- **Claim under attack:** the table "must be frozen before it is used. Choosing
  control placements with NLL measured on the run being analysed would leak the
  outcome into the design", enforced by requiring `--seed` to be a burned pilot
  seed. And in the draft's limitations: the table "was frozen from an early
  checkpoint of the run later analysed, so the gap between arms is unbiased but
  the absolute values are not independent of the model."
- **Defect:** `--seed` controls only `bios.generate_records(entities, seed)`, which
  is *which entity records get scored*. The checkpoint comes from `--run` and
  `--ckpt` and has no guard at all. So the enforced constraint is data disjointness
  while the admitted violation is model non-disjointness, and the code's error
  message ("Building the difficulty table from data that also appears in the matrix
  leaks the outcome into the design") will read as satisfied in exactly the case
  the draft concedes is not.

  Separately, "the gap between arms is unbiased" is asserted and not shown, and I
  do not think it holds. FACTMASK spans are selected by role and their table values
  are unselected. RANDPOS spans are selected to maximize agreement with a *noisy
  estimate* of difficulty, so their table values carry a winner's-curse component
  that does not cancel in the difference. Whichever direction it runs, the draft
  should state it rather than assert unbiasedness, particularly since Section 4's
  whole argument is about the gap.
- **Why existing checks miss it:** the seed guard fires loudly and correctly on the
  channel it covers, which makes the uncovered channel look covered. The provenance
  sidecar records `checkpoint` and `checkpoint_sha256` faithfully, so the
  information needed to catch this is written down and nothing reads it back.
- **Falsifiable check:** make `nll_table.py` refuse, or at minimum warn on the
  provenance sidecar, when `--run` resolves to a run whose seed is not in
  `BURNED_PILOT_SEEDS`. Then read the sidecar of the table that placed the 21.3B
  corpus and record which run and seed produced it. For the bias claim, place a
  second control using a table built from a genuinely disjoint checkpoint and
  compare the two gaps.
- **What would prove me wrong:** the shipped table having come from a burned-seed
  *run* as well as burned-seed records, which would make the draft's limitation
  sentence over-cautious rather than the code under-protective.

---

## Checked and clean

- **Amendment 2 (§6, minimum interesting effect).** The circularity it fixes is
  real and the fix is correct. Anchoring on `sqrt(0.25/1500) = 0.012910` is
  independent of `delta`, so §8.3 can now fail, and the stated
  `delta ≥ 2.582 pp` threshold is exactly 2 × 1.291 pp. The amendment moves the
  bar in the harder direction, which is the right direction for a change filed
  without supervision.
- **§1's reduced-form estimand.** Writing `Y[FACTMASK] − Y[RANDPOS]` rather than a
  double difference is correct and the stated reason holds: the SUP terms cancel
  exactly, so double-difference notation would imply an identification property
  the design lacks.
- **§3's load construction.** Trading entities against exposures at fixed document
  count does decouple load from exposures per fact, and `theory/capacity.py:140-159`
  implements it as described. `ladder()`'s docstring correctly notes that both
  terms of F/C move together, which is the non-obvious consequence.
- **`dose_response_signature` zero-floor fix.** The old guard
  `peak < tol * max(1e-12, peak)` could indeed only fire for negative `peak`. The
  replacement takes an absolute floor and the "flat and nonzero → confound" branch
  is reachable.
- **`responsive()` fails from the ceiling side at 93.8%.** With `ceiling=0.95,
  margin=0.02` the predicate needs accuracy below 0.93, so the in-band aggregate
  correctly fails, and §5's stated reason for widening the op band to [1, 8]
  follows from the code rather than from narrative.
- **F/C is genuinely invariant to the op-band widening.** `Load.ratio` depends only
  on entities, bits per entity, exposures, and parameters, so holding the fact
  configuration at 996,408 × 200 does preserve F/C = 1.000 as
  `docs/RESULTS-2026-08-04.md:112-115` claims. See the note below for the part of
  that claim I could not close.

## Handed to other lanes

- **Lane B:** the [1, 8] corpus conserves the fact configuration, so with longer
  iGSM documents something else must float, either total tokens (which moves the
  step budget and the scratch projection the §10a guard checks) or the fact lane's
  realised share (which would move exposures and therefore F/C after all). The
  builder is Lane B's surface and the last time a conservation choice went
  unstated it cost a generation of the experiment. Worth confirming which quantity
  floats in job 1675799.
- **Lane C:** `docs/RETAINED-RESULTS.md:391` carries the same 20.0% as the two
  drafts, so A-02 is a three-document fix and the catalogue is one of them.

## Not a finding, recorded so cycle 2 does not re-raise it

§7.3's first clause ("SUP iGSM accuracy inside the frozen band") looks circular
against §5, which *defines* the band as where SUP accuracy falls in [0.17, 0.85].
It is not, provided the gate is evaluated per-seed on the matrix rather than on the
single run that instantiated the band, since later seeds can fall outside a band
frozen from the first. The document does not say which, and the ambiguity is worth
one clarifying sentence, but it is not a defect until the matrix runs.
