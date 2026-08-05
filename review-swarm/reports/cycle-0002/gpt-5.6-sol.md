**Verdict:** Two of Grok 4.5's six findings are confirmed as filed; four are real but need narrower scope or lower severity. The new surface adds no S0, one S1, two S2s, and one S3.

## Cross-verification

### MS-002 — REFINED (core S0 remains)

**Check output**

```text
docs/PAPER-MEASUREMENT.md
  60:Runs are 21.3B tokens, 40,695 optimizer steps, on one NVIDIA L40S.
  86:Run `stepladder_d40m_std` at step 31,280, 64 held-out iGSM problems at MOD 23,
  211:of its final value by step 7,820, so the 40,695-step budget is generous by
  233:Measured over all 21.3B tokens of the operating-point corpus:

docs/PAPER-WORKSHOP.md
  41:16.4B-token run of 31,280 steps on one L40S, carrying 1,531,800 entities at 100
  43:that exposure count. Section 4 measures the control on a second corpus of 21.3B
  106:Measured over 21.3B tokens, the treatment's masked spans average 1.9598 nats
```

```text
$ python3 -c "print(31280/7820, 40695/7820, 40695/12512, 31280/12512)"
4.0 5.203964194373402 3.252477621483376 2.5
```

**Reasoning**

The corpus attribution is independently reproduced. The setting says 21.3B
tokens and 40,695 steps, while the named run and its displayed curve are the
16.4B-token, 31,280-step stepladder. That remains an S0 because the paper
attributes measured findings to a different, untrained corpus.

The arithmetic allegation is too strong. The last displayed partial-curve
point is step 12,512, and 40,695 / 12,512 = 3.25, so "roughly a factor of three"
has a plausible reading as budget relative to the demonstrated partial curve.
The sentence is ambiguous because its immediate antecedent is step 7,820, for
which the ratios are 5.20 or 4.00, but it is not unambiguously arithmetically
false under every reading. The finding should retain the wrong-corpus S0 and
drop or narrow the arithmetic charge.

Related scope delta: the budget drift is wider than this paper. `README.md:19`
and `docs/PREREGISTRATION.md:40` still specify roughly 141 tokens per parameter,
while the same preregistration names `high-e200` at 21.3B tokens
(`:101,304`), which is 526.03 tokens per 40.56M parameters. This belongs with
the existing budget-provenance finding rather than as a new duplicate.

### MS-003 — CONFIRMED (S0)

**Check output**

```text
docs/PAPER-MEASUREMENT.md
  179:(n=384), op3 88.3% (n=375) and op4 73.1% (n=342) — so op3 is not at ceiling
  446:chains perfectly and reaches 93.8% in-band. Four generations of this project

docs/RESULTS-2026-08-04.md
  66:The model solves three-operation dependency chains perfectly and falls off a
```

The surrounding source makes the contradiction explicit:

```text
176-**These per-cell counts (8–17) are too small to publish and are reported here
177-only as a provisional shape.** A 1,500-item evaluation is in progress; at step
178-14,076, 45% of the way through training, it gives op1 100.0% (n=399), op2 98.2%
179-(n=384), op3 88.3% (n=375) and op4 73.1% (n=342) — so op3 is not at ceiling
180-mid-training, and "perfect through three, cliff at six" is not supported at
181-large n. The final-checkpoint table at n=1,500 replaces this one before
182-submission.
```

**Reasoning**

The only perfect op3 cell is 9/9 in a table the draft itself says is too small
to publish. The larger table has op3 at 88.3%, and the final-checkpoint
1,500-item table has not landed locally. "Solves ... perfectly" is therefore an
unsupported population reading in both the discussion and the results note,
not merely a rounding choice.

### MS-004 — REFINED (S2 staleness, not S0 resurrection)

**Check output**

```text
HANDOFF.md
  72:## The two results that do stand
  76:### 1. The endpoint is not learnable at 1,500 steps, at any difficulty
  83:| 23 | 0.0753 | 0.0100 | no |
```

```text
HANDOFF.md
  3:Prepared 2026-08-03. Read this file first; it is the map.
  8:> Two negative results do stand, and both are new and artifact-backed.
  9:> The decisive run finished at ~05:00 on 2026-08-03 and has not been read yet;
```

```text
docs/RESULTS-2026-08-04.md
  80:- Stage A's STOP, and its rationale that "the endpoint has no power to
  82:- Both difficulty ladders, including the 2026-08-02 rebuild on valid corpora
  83:  and its verdict that the task is "NOT LEARNABLE even at mod 5".
```

**Reasoning**

The standing-results content is now wrong and actionable because the file says
to read it first. However, the file is an explicitly dated 2026-08-03 snapshot,
and the reversal is dated 2026-08-04. It did not resurrect a result after its
withdrawal; it became stale when later evidence arrived. A top banner pointing
to `RESULTS-2026-08-04.md` is sufficient, which makes this an S2 collaborator
artifact gap rather than an S0 paper claim.

Withdrawal-list delta: `RETAINED-RESULTS.md` gives a blanket rule withdrawing
all pre-fix batched accuracies, but its enumerated table does not list the
difficulty-ladder cells. The resurrection test also scans only
`docs/*.md`/`*.tex`, not `HANDOFF.md`, and has no ladder fingerprint. This is
the precise reason the stale snapshot remains invisible, and it sharpens
MS-004 rather than creating a new finding.

### MS-008 — REFINED (S3)

**Check output**

```text
docs/PAPER-WORKSHOP.md
  87:What the bug was hiding is a working benchmark. At 45% of training, on 1,500
  88-items, the model scores 100.0% at one operation, 98.2% at two, 88.3% at three
  96:[PLACEHOLDER: final-checkpoint per-operation table, 1,500 items per cell, from
  97-the scoring run completing 2026-08-05.]
  206-One model size, one corpus, one seed. Section 3's provisional numbers have 8 to
  207:17 items per cell and are replaced by the 1,500-item table when it lands.
```

**Reasoning**

The mismatch is real, but the S1 rationale is overstated. Section 3 itself
plainly labels the displayed table as 1,500 items at 45% of training, and the
placeholder plainly says the final table is absent. A reader is not left
unaware that the result is mid-training or that final scoring is pending. The
limitations sentence copied the 8–17 count from the long draft, which is an
internal document inconsistency under the charter's S3 definition. Also, on
2026-08-05 a promise to complete on 2026-08-05 is not yet provably overdue.

### MS-009 — CONFIRMED (S2)

**Check output**

```text
$ rg -n "stepladder|1\.9732|112\.90|1\.9598" outputs/RETAINED-SHA256SUMS
No matches found
```

The local run tree is absent:

```text
Glob runs/stepladder_d40m_std/**:
0 files found
```

The tracked-artifact probe returned only the old Stage A report:

```text
$ git ls-files 'runs/stepladder_d40m_std/**' 'outputs/**' |
  rg 'stepladder|step0031280|step31280|nll_table|stage_a_report'
outputs/pilot/stage_a_report.json
```

**Reasoning**

The local manifest does not cover the post-fix headline artifacts, and the run
tree is not committed. FarmShare could contain them, but that would not retire
the finding as written: an outside replicator holding this tree still cannot
verify the claims from the advertised hash manifest.

### MS-010 — REFINED (S2, narrower target)

**Check output**

```text
docs/PREREGISTRATION.md
  256:On NO-GO: stop and publish the pilot. `docs/NO-GO-PAPER.md` was written before

HANDOFF.md
  160:5. `docs/THEORY-ENDPOINT.md` and `docs/THEORY-CAPACITY.md` — why the endpoint
  163:6. `docs/NO-GO-PAPER.md` — the paper outlined *before* the pilot ran, so the
  193:`docs/NO-GO-PAPER.md` is the deliverable. Two of three independent reviewers
  323:`docs/NO-GO-PAPER.md` becomes the deliverable. Its outline predates Stage A on

ops/crowding/RUNBOOK.md
  26:`docs/THEORY-ENDPOINT.md` argues the Stage A floor is a step-budget artifact,
  219:> modular arithmetic is reported to need. Read `docs/THEORY-ENDPOINT.md`
  336:`docs/NO-GO-PAPER.md`, which was outlined before Stage A so the pilot already
```

The first 20 lines of `THEORY-ENDPOINT.md` contain no supersession notice:

```text
1:# Why iGSM is at floor, and what Stage A did not test
3:Derivations in `theory/endpoint.py`, assertions in
4:`tests/test_theory_endpoint.py`, measurements from
5:`outputs/pilot/stage_a_report.json`.
7:Stage A returned **STOP**, on the rationale that "the endpoint has no power to
8:discriminate anything, so a null from the matrix would be uninformative." The
9:floor is real. The rationale does not follow from what was run.
```

**Reasoning**

The missing banner on `THEORY-ENDPOINT.md` is reproduced and remains S2.
The rest of the filed scope is too broad:

- `RUNBOOK.md:222` immediately says, in bold, "That STOP is now withdrawn
  entirely", so its nearby historical link is not current endorsement.
- `RUNBOOK.md:335-337` and `PREREGISTRATION.md:256` use `NO-GO-PAPER.md`
  conditionally for a future gate, which is legitimate.
- `HANDOFF.md` is the dated snapshot addressed under MS-004.
- `stage_a_report.json` is a raw historical artifact. Its recorded `"decision":
  "STOP"` should remain byte-faithful; supersession belongs in a banner on the
  consuming document or in sidecar metadata, not by rewriting the measurement.

Withdrawal-list delta: the Stage A profile and difficulty ladders overturned by
`RESULTS-2026-08-04.md` are not rows in the enumerated withdrawal table and are
not fingerprints in `test_repo_state.py`. The blanket pre-fix rule covers them
semantically but cannot make the resurrection test see them. This supports the
narrow banner fix.

## New findings

### Occupancy verdict thresholds are executable but not frozen or disclosed

- **Severity:** S1
- **Target:** `ops/crowding/occupancy.py:54-59,119-159`; `docs/THEORY-CAPACITY.md:77-94`; `docs/PREREGISTRATION.md:84-110,241-246`; `ops/crowding/RUNBOOK.md:115-126`
- **Claim under attack:** A "saturating" occupancy ladder establishes that capacity has begun to bind and licenses running the arm contrast, while "abandonment", "linear", and "inert" prescribe different scientific decisions.
- **Defect:** The executable classifier defines `INERT = 0.02`, `ABANDON_FRAC = 0.20`, and `BEND_FRAC = 0.20`. Thus a peak below 0.02 bits/parameter is inert, a 20% fall from the peak is abandonment, and a 20% shortfall from first-rung linear efficiency is saturation. None of those decision thresholds appears in the theory note, runbook, or frozen preregistration. Both `occupancy.py` and its tests are untracked, so the repository also has no committed pre-outcome provenance for the cutoffs. The qualitative claim survives only with the unstated assumption that these three cutoffs were fixed before observing the ladder.
- **Why existing checks miss it:** `tests/test_occupancy.py` feeds synthetic rows that sit comfortably on each side and asserts the resulting labels; it does not compare the constants with a dated declaration. The theory tests rederive F/C arithmetic, not occupancy shape thresholds.
- **Falsifiable check:** `rg -n "INERT|ABANDON_FRAC|BEND_FRAC|No rung stores more than|fractional drop|linear extrapolation" docs/THEORY-CAPACITY.md docs/PREREGISTRATION.md ops/crowding/RUNBOOK.md ops/crowding/occupancy.py` currently returns the definitions and uses only in `occupancy.py`. Add the three values and boundary conventions to a dated preregistration amendment, then make a test parse and compare them.
- **What would prove me wrong:** A frozen, dated document or immutable pre-outcome commit that names all three thresholds and predates every occupancy result.

### The PopQA primary record advertises a deleted harness as committed and reproducible

- **Severity:** S2
- **Target:** `docs/POPQA-HELDOUT-KEY.md:3-12,57-138`; `docs/RETAINED-RESULTS.md:211-235`; `tests/test_repo_state.py:162-168`
- **Claim under attack:** "The lost harness was recreated ... and is now committed"; reproduce with `scripts/run_keyguess_local.py --stage all`, run seeds with `cluster/slurm/keyguess_cpu.sbatch`, and regenerate policy tables with `scripts/analyze_keyguess_policy.py`.
- **Defect:** None of the named harness files is tracked or present: `corpusgen/realfact.py`, `evals/keyguess.py`, `evals/constrain.py`, `scripts/run_keyguess_local.py`, `scripts/analyze_keyguess_policy.py`, and `cluster/slurm/keyguess_cpu.sbatch` all produce no `git ls-files` output. `RETAINED-RESULTS.md` consequently withdraws the experiment because its decode path cannot be checked or rerun. The data artifacts and pinned input remain tracked, and the pinned input hash matches, but the document has no withdrawal banner and still presents dead commands and the withdrawn measurements as current results.
- **Why existing checks miss it:** `POPQA-HELDOUT-KEY.md` is explicitly in `FINGERPRINT_EXEMPT` as the primary record of the withdrawn experiment. The deleted-module test checks imports in extant Python source, not reproduction commands in prose, while `test_docs_holds_only_the_current_documents` affirmatively requires this document to remain.
- **Falsifiable check:** `git ls-files -- corpusgen/realfact.py evals/keyguess.py evals/constrain.py scripts/run_keyguess_local.py scripts/analyze_keyguess_policy.py cluster/slurm/keyguess_cpu.sbatch` currently prints nothing. Each named path must exist and `scripts/run_keyguess_local.py --help` must expose `--stage` and `--seed`.
- **What would prove me wrong:** The complete harness is restored and a fixed-decoder rerun reproduces the tables, or the document opens with a withdrawal banner and removes its claim that the deleted commands reproduce the result.

### RUNBOOK Stage B and C recipes silently select stale corpora and budgets

- **Severity:** S2
- **Target:** `ops/crowding/RUNBOOK.md:288-304,320-324`; `ops/crowding/pilot.py:106-186,206-238`
- **Claim under attack:** The Stage B block builds the operating-point and NOFACT corpora at equal tokens and steps, then gives the commands that configure Stage B and Stage C on that high load.
- **Defect:** The block builds `corpora/high-e200` and `corpora/nofact` at 21,335,900,160 tokens, with 996,408 entities in the fact corpus. The very next command points Stage B at `corpora/high`, limits both runs to 16,399,769,600 tokens, and records 382,900 entities. Stage C repeats those stale values. `pilot.py` copies the supplied path, token count, and entity count directly into configs without checking the corpus manifest. The commands therefore either target an old/nonexistent corpus or train the newly built NOFACT corpus for a different number of steps, defeating the stated equal-token comparison while all flags parse successfully.
- **Why existing checks miss it:** Every documented Python flag exists and every shell file passes `bash -n`; syntax validation is green. The config generator tests argument propagation rather than cross-checking `targets.bin` and `manifest.json` against `--total-tokens` and `--entities`, so a semantically wrong but well-formed command survives.
- **Falsifiable check:** `rg -n "high-e200|corpora/high|21335900160|16399769600|996408|382900" ops/crowding/RUNBOOK.md` shows the two incompatible triples in one command block. Generating Stage B configs and comparing `train_bin`, `total_tokens`, and `n_entities` with both corpus manifests resolves it mechanically.
- **What would prove me wrong:** `corpora/high` is documented as an exact alias of `high-e200` with a manifest proving 21,335,900,160 tokens and 996,408 entities, and `pilot.py` derives rather than truncates each run's step count from that manifest.

### THEORY-CAPACITY describes the old F/C=2 ladder as the corpus in flight

- **Severity:** S3
- **Target:** `docs/THEORY-CAPACITY.md:27-47,98-110`; `docs/PREREGISTRATION.md:99-108`; `ops/crowding/RUNBOOK.md:101-126`
- **Claim under attack:** "The corpus in flight is 2x over-saturated": 1,531,800 entities at 100 exposures, with lower rungs at 765,900/200 and 382,950/400; the top corpus is "already building".
- **Defect:** The arithmetic still matches `theory/capacity.py`, but the live preregistration and runbook define the operating point as 996,408 entities at 200 exposures, F/C = 1.000, with occupancy rungs 249,102/800, 498,204/400, and 996,408/200. The runbook explicitly says no rung above F/C=1 fits. The theory note is therefore a correct derivation for a superseded ladder presented as current status.
- **Why existing checks miss it:** `tests/test_theory_capacity.py` can rederive either table from the inputs it is given. Nothing binds the prose's entity/exposure tuple to the canonical tuple in the preregistration or runbook.
- **Falsifiable check:** `rg -n "1,531,800|996,408|249,102|498,204|F/C = 2|F/C = 1" docs/THEORY-CAPACITY.md docs/PREREGISTRATION.md ops/crowding/RUNBOOK.md` displays the incompatible ladders.
- **What would prove me wrong:** The 1,531,800/100 ladder is still an active, separately named design and the theory note labels it as such rather than calling it the corpus in flight.

## Checked and found clean

- `docs/GATE0-CEILING-IS-NOT-A-BOUND.md` matches
  `outputs/pilot/gate0_diagnosis.json`: 32,517 positions, mean 25.026 nats,
  median 27.778, 98.7% above ln(V), entropy 3.826, and argmax accuracy 0.0001.
  `diagnose_gate0.py --help` exits 0 and exposes `--run`.
- The numerical tables in `docs/THEORY-CAPACITY.md` reproduce exactly from
  `theory/capacity.py`: `ALPHA_ANCHORS=((100,1.0),(1000,2.0))`,
  `CRITICAL_BAND=(0.5,1.5)`, ratios 2.000/0.769/0.312, and recentered entity
  counts 552,370/930,402/1,531,723. The current-design status and undisclosed
  occupancy thresholds are the separate findings above.
- Every Python flag quoted by `RUNBOOK.md` for `nll_table.py`, `occupancy.py`,
  `status.py`, `ladder.py`, `pilot.py`, `analyze_pilot.py`, `gen_configs.py`,
  `analyze_crowding.py`, `run_evals.py`, and `diagnose_gate0.py` is present;
  all ten `--help` invocations exited 0. The four shell/Slurm files checked
  pass `bash -n`. The Stage B/C values are semantically wrong despite that.
- `docs/POPQA-HELDOUT-KEY.md`'s pinned PopQA SHA-256 matches
  `data/realfacts/popqa_clean.jsonl`, and its governance-table counts match
  tracked `policy_analysis.json`. The missing executable harness, not the
  retained count arithmetic, is the defect.
- `docs/PAPER-BRIEF.md` and `docs/PAPER-BRIEF.tex` do not exist, are not
  tracked, and have no status entry. Cycle 1 was correct; the claimed conflict
  with git status is not present at `cleanup/core-measurements` / `efc338a`.
- The enumerated withdrawal rows for §1, §5, and §6 agree with the corresponding
  withdrawn sections. The unenumerated difficulty-ladder and Stage A values are
  recorded above as deltas to MS-004 and MS-010 rather than re-filed.
- FarmShare is unreachable from this checkout. I could not inspect remote job
  state, remote final-checkpoint artifacts, or execute the Slurm commands.
  Corpus builds and training were not run because they would write outputs or
  submit jobs. The full suite was unnecessary because no code changed and the
  measured baseline was supplied.
