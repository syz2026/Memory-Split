# Lane C — cycle 0001 — grok-4.5

**Verdict:** Live drafts and the handoff still mix post-fix claims with pre-fix generative numbers, wrong corpus budgets, and a final-checkpoint table that has not landed.

### PAPER-MEASUREMENT discussion asserts “perfect through three” after the same draft marks that claim unsupported

- **Severity:** S0
- **Target:** docs/PAPER-MEASUREMENT.md:444–446 (also docs/RESULTS-2026-08-04.md:66–67)
- **Claim under attack:** “The reasoning endpoint … solves three-operation dependency chains perfectly and reaches 93.8% in-band.”
- **Defect:** The same draft’s §4 (lines 176–182) states the 48-item per-op table is too small to publish, that mid-training n≈1,500 shows op3 at 88.3% not ceiling, and that “perfect through three, cliff at six” is not supported at large n. `docs/RETAINED-RESULTS.md` §8 explicitly does not support the 48-item table and says the final-checkpoint 1,500-item table is not yet written. `RESULTS-2026-08-04.md` still states the stronger claim with no large-n caveat. A replicator reading the discussion (or the results note) would take a withdrawn shape claim as established.
- **Why existing checks miss it:** `tests/test_repo_state.py::test_no_document_resurrects_a_withdrawn_number` fingerprints only the §A/§B ledger strings (0.8B gates, 1B held-out, PopQA, etc.), not the post-fix 48-item shape claim. Nothing asserts consistency between a draft’s caveats and its discussion.
- **Falsifiable check:** `rg -n "perfectly|perfect through three|48 items|not supported at large n|final-checkpoint" docs/PAPER-MEASUREMENT.md docs/RESULTS-2026-08-04.md docs/RETAINED-RESULTS.md` and confirm discussion/RESULTS still assert the shape that §4 and RETAINED §8 refuse.
- **What would prove me wrong:** Discussion and RESULTS either drop “perfectly” / the op1–3 ceiling reading, or cite a landed final-checkpoint `evals/step*.json` at n=1,500 that actually shows op1–3 at ceiling.

### HANDOFF still presents padded-decoder difficulty-ladder accuracies as one of “two results that do stand”

- **Severity:** S0
- **Target:** HANDOFF.md:72–97 (also HANDOFF.md:163–164, 190–193)
- **Claim under attack:** Under “The two results that do stand,” “The endpoint is not learnable at 1,500 steps, at any difficulty,” with op=1 accuracies 0.0100 / 0.0441 / 0.0080 / 0.0268, and `docs/NO-GO-PAPER.md` as “the likely deliverable.”
- **Defect:** `docs/RESULTS-2026-08-04.md` overturns both difficulty ladders and the NO-GO framing that rested on an unusable endpoint; `docs/RETAINED-RESULTS.md` states that any generative accuracy produced before 2026-08-04 by batched generation is withdrawn. These ladder cells are pre-fix generative accuracies presented as current standing science. The handoff also still says the stepladder “has not been read yet,” which is false after RESULTS-2026-08-04.
- **Why existing checks miss it:** The resurrection test’s fingerprints do not include these ladder floats. HANDOFF is dated 2026-08-03 and is not in `FINGERPRINT_EXEMPT`, but nothing greps for post-withdrawal “results that do stand” claims.
- **Falsifiable check:** `rg -n "results that do stand|not learnable|0\\.0100|likely deliverable|NO-GO-PAPER" HANDOFF.md docs/RESULTS-2026-08-04.md` and verify HANDOFF’s standing-results section still asserts the overturned ladder.
- **What would prove me wrong:** HANDOFF’s standing-results section is rewritten so the ladder table is labeled withdrawn / superseded, and NO-GO is no longer named as the likely deliverable without the padding-reversal caveat.

### PAPER-MEASUREMENT §2 attributes the wrong corpus budget to the stepladder findings

- **Severity:** S0
- **Target:** docs/PAPER-MEASUREMENT.md:60 and :211 (contrast :86, :199–207)
- **Claim under attack:** “Runs are 21.3B tokens, 40,695 optimizer steps”; later, after the stepladder learning curve, “the 40,695-step budget is generous by roughly a factor of three.”
- **Defect:** Finding 1, the 48-item table, the step-14,076 per-op numbers, and the learning-curve steps (1,564…12,512) are from `stepladder_d40m_std` at **16.4B tokens / 31,280 steps** (`RETAINED-RESULTS.md` §8; PAPER-WORKSHOP.md §2 states this correctly). 21.3B / 40,695 is `high-e200`, on which PAPER-WORKSHOP.md §2 says “No model has yet been trained.” The draft therefore mislabels the measured run and prices matrix compute off the wrong step count.
- **Why existing checks miss it:** No test cross-checks paper “Runs are…” lines against the run id in the same section. RETAINED catalogues both corpora but does not assert draft consistency.
- **Falsifiable check:** `rg -n "21\\.3B|40,695|16\\.4B|31,280|stepladder_d40m_std" docs/PAPER-MEASUREMENT.md docs/PAPER-WORKSHOP.md docs/RETAINED-RESULTS.md` and require every generative finding’s budget to match `stepladder_d40m_std`.
- **What would prove me wrong:** §2 distinguishes the two corpora the way PAPER-WORKSHOP §2 does, and the “generous by a factor of three” sentence uses 31,280 (or another budget that actually belongs to the curve).

### Final-checkpoint 1,500-item table is still a placeholder while prose already uses ceiling language

- **Severity:** S0
- **Target:** docs/PAPER-WORKSHOP.md:87–97 and :206–207; docs/RETAINED-RESULTS.md:328
- **Claim under attack:** Workshop §3: “In-band the model is now at ceiling,” followed by `[PLACEHOLDER: final-checkpoint per-operation table, 1,500 items per cell, from the scoring run completing 2026-08-05.]`; limitations say §3’s provisional numbers “have 8 to 17 items per cell and are replaced by the 1,500-item table when it lands.”
- **Defect:** Today is on or after 2026-08-05. In-repo state still has the placeholder; `RETAINED-RESULTS.md` §8 still says “The final-checkpoint 1,500-item table is not yet written”; `PAPER-MEASUREMENT.md:177` still says the 1,500-item evaluation is “in progress.” No `evals/step*.json` for stepladder exists under `outputs/` or in `outputs/RETAINED-SHA256SUMS`. The numbers actually shown in Workshop §3 are mid-training n=1,500 cells (100.0 / 98.2 / 88.3 / 73.1), not 8–17-item cells, so the limitations line describes the wrong table. Ceiling language for “now” outruns the artifact the placeholder is waiting on. (FarmShare job state was not reachable from this checkout; the catalogue and drafts are the authority for whether numbers landed.)
- **Why existing checks miss it:** No test fails on unresolved `PLACEHOLDER` markers or on “completing <date>` after that date. Resurrection fingerprints do not cover missing forward tables.
- **Falsifiable check:** `rg -n "PLACEHOLDER|not yet written|in progress" docs/PAPER-WORKSHOP.md docs/RETAINED-RESULTS.md docs/PAPER-MEASUREMENT.md`; `rg -n "stepladder|step31280|step31_280" outputs/RETAINED-SHA256SUMS`; confirm placeholder still present and no final per-op artifact is hashed.
- **What would prove me wrong:** Placeholder replaced by a table backed by a hashed `evals/step*.json` at the final checkpoint, and “now at ceiling” either cites that table or is narrowed to the mid-training / 64-item quantities that exist.

### Corrected-decoder headline numbers have catalogue rows but no committed hashed artifacts

- **Severity:** S2
- **Target:** docs/RETAINED-RESULTS.md §8–§9 (and citations in docs/PAPER-WORKSHOP.md:214–215, docs/PAPER-MEASUREMENT.md:501–506)
- **Claim under attack:** “Hashes: `outputs/RETAINED-SHA256SUMS` (110 files)”; Workshop: “Every number above is catalogued with its artifact in `RETAINED-RESULTS.md`, sections 7 to 11.”
- **Defect:** All 110 hashed paths exist locally, but none are `runs/stepladder_d40m_std/**` or any file containing the post-fix 3/64 vs 60/64 measurement, the step-14,076 per-op table, Gate 0 1.9732, or the −112.90 / −112.29 storage cohorts. §7’s RANDPOS gap points at FarmShare `high-e200` / `outputs/pilot/` without a hashed NLL-gap artifact that embeds 1.9598 / 1.2712. An outside replicator with only this git tree cannot verify the paper’s central post-fix numbers from the manifest the catalogue advertises.
- **Why existing checks miss it:** `RETAINED-SHA256SUMS` integrity is “file present,” not “every citable RETAINED row has a hashed producer.” Sections 8–11 were reconciled after the 2026-08-01 hash freeze.
- **Falsifiable check:** `wc -l outputs/RETAINED-SHA256SUMS` (110); `rg -n "stepladder|1\\.9732|112\\.90|1\\.9598" outputs/RETAINED-SHA256SUMS`; `python3 -c` loop confirming every SUMS path exists (all do) and none name stepladder evals.
- **What would prove me wrong:** Hashed, committed artifacts for §7–§9 generative and storage figures, with paths listed in `RETAINED-SHA256SUMS` and readable without FarmShare.

### Superseded endpoint docs still lack banners and are still linked as current

- **Severity:** S2
- **Target:** docs/THEORY-ENDPOINT.md:1–31; outputs/pilot/stage_a_report.json:70–71; HANDOFF.md:160–164; ops/crowding/RUNBOOK.md:219–220, 335–336
- **Claim under attack:** THEORY-ENDPOINT presents Stage A STOP and op profile 5.4% / 2.9% / 7.2% / 5.0% as the measured floor; `stage_a_report.json` still has `"decision": "STOP"` and tells the reader to write the NO-GO paper; HANDOFF reading order and RUNBOOK still point at THEORY-ENDPOINT / NO-GO-PAPER as live guidance.
- **Defect:** `docs/RESULTS-2026-08-04.md` declares it supersedes THEORY-ENDPOINT “in full” and the STOP in `stage_a_report.json`. Neither superseded artifact carries a top-of-file withdrawal banner. THEORY-ENDPOINT’s op-profile cells are pre-fix generative accuracies. NO-GO-PAPER.md still opens with “If the pilot returns NO-GO, this is the deliverable” with no note that the endpoint premise is gone.
- **Why existing checks miss it:** No test requires a supersession banner when RESULTS names a document. Resurrection fingerprints do not include Stage A’s 0.054/0.029-style op profile.
- **Falsifiable check:** `head -20 docs/THEORY-ENDPOINT.md`; `rg -n "decision|STOP|NO-GO" outputs/pilot/stage_a_report.json`; `rg -n "THEORY-ENDPOINT|NO-GO-PAPER" HANDOFF.md ops/crowding/RUNBOOK.md docs/PREREGISTRATION.md`.
- **What would prove me wrong:** THEORY-ENDPOINT and `stage_a_report.json` open with an explicit superseded/withdrawn notice pointing at RESULTS-2026-08-04, and live runbooks stop presenting NO-GO as the default deliverable without that notice.

### Adjacent RETAINED tables disagree on the shipped control gap at the second decimal

- **Severity:** S3
- **Target:** docs/RETAINED-RESULTS.md:259 vs :269; also :257 vs :388
- **Claim under attack:** Full-corpus relative gap **35.1%**; position-tolerance row for shipped 0.15 also labeled shipped but **35.0%**. Treatment mean NLL **1.9598** (§7, all 21.3B tokens) vs **1.9584** (§10, 800 documents).
- **Defect:** Same label “shipped” yields 35.1% and 35.0% in one file; treatment NLL differs by 0.0014 across sections without an in-row note that §10 is a subsample. Downstream drafts mostly copy 35.1% / 1.9598, so this is internal catalogue inconsistency rather than a false paper claim.
- **Why existing checks miss it:** No invariant ties the §7 headline gap to the §7 sweep table or the §10 subsample treatment mean.
- **Falsifiable check:** `rg -n "35\\.1%|35\\.0%|1\\.9598|1\\.9584" docs/RETAINED-RESULTS.md` and require one definition per label or an explicit subsample note on the §10 treatment row.
- **What would prove me wrong:** The file states that 35.0% is rounding of the same quantity as 35.1%, and that 1.9584 is the 800-doc subsample of 1.9598, in the table cells themselves.

---

## Checked and found clean

- Fingerprint resurrection outside the known intentional red: only `docs/section-null-without-crowding.md` matches `WITHDRAWN_FINGERPRINTS`; no other live `docs/*.md`, `README.md`, or `HANDOFF.md` co-hosts those ledger sets. `docs/PAPER-BRIEF.md` / `.tex` are absent (already removed).
- `outputs/RETAINED-SHA256SUMS`: 110 lines, 110 paths present, 0 missing files (integrity of the hash list itself is fine; coverage of §8–§9 is the separate S2 above).
- Cross-doc agreement on the retained core: 4.7% = 3/64 and 93.8% = 60/64; FACTMASK/RANDPOS 1.9598 / 1.2712 / 35.1%; scattered-control ladder 1.5667/20.0%, 1.7677/9.8%, 1.8644/4.9%; storage −112.90 / −112.29 / −0.61 ± 0.68; Gate 0 1.9732 vs 25.03; lane substitution 3.75% / 56.25%; entity budgets 996,408×200 vs 1,531,800×100; model size 40.6M / 21.2M; exposure census 1.04–1.55 — consistent across RETAINED, PAPER-WORKSHOP, PAPER-MEASUREMENT, and RESULTS where they cite these.
- `1.000` as F/C vs `1.000` as in-answer-space rate: drafts that use both keep them in separate sentences/tables; no conflation found in the live paper drafts (Workshop’s collapse of 0.9993→1.000 is a rounding of the post-fix rate, not an F/C mix-up).
- Pre-fix / post-fix per-op mixing: live papers do not paste 0.8B-gate or 1B-held-out accuracy tables beside corrected stepladder cells. The live failure mode is the unsupported *post-fix* 48-item shape claim, not a mix with §1 withdrawn gates.
