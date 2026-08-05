# MemorySplit — collaborator handoff

Prepared 2026-08-03. Read this file first; it is the map.

> **State in four lines.**
> The primary estimand has never been validly measured — four experiment
> generations, all withdrawn, none of them failing on the hypothesis.
> Two negative results do stand, and both are new and artifact-backed.
> The decisive run finished at ~05:00 on 2026-08-03 and has not been read yet;
> reading it takes minutes and decides whether there is a paper about crowding
> or a paper about why crowding could not be tested.
> Start at **Next steps → Step 0** at the end of this file if you want the
> action rather than the background.

---

## The question

Does removing factual memorisation from a small language model's weights free
capacity for reasoning? This is the belief the phi-3 team cites when filtering
fact-heavy pages from training data. The training-time mechanism is LMLM-style
(arXiv 2505.15962): fact values sit in the context but are excluded from the
loss.

The primary estimand, stated exactly:

```
effect = Y[FACTMASK] - Y[RANDPOS]
```

the effect of masking fact-value targets rather than an equal, matched mass of
non-value targets, on closed-book iGSM accuracy, in a 40.6M-parameter
tied-embedding decoder, trained on one controlled corpus.

**Capacity reallocation is an interpretation of that quantity, not the quantity
itself, and this design does not identify it.** Masking removes competing
gradients whether or not any parameter was occupied, so gradient interference
predicts the same sign and the same dose trend. What discriminates them is the
*shape* of the effect across fact loads. Nothing here licenses a claim about
"small language models" generally, or about phi-3, which is 100x larger.

---

## Where the project actually stands

**There is no valid measurement of the primary estimand. Not a weak one — none.**
Four experiment generations have run and all four are withdrawn, each for a
different reason, and not one of them failed on the hypothesis:

| generation | when | why it does not count |
|---|---|---|
| 1. original design | Jul 17–22 | reported numbers had no committed artifacts; draft deleted |
| 2. reasoning-v3, 160M | Jul 30–31 | 1.04–1.55 exposures per fact, median 1 — nothing was memorised, so nothing could be crowded out |
| 3. tiny cohort, 8M/40M | Jul 31–Aug 1 | loss used `reduction="mean"`, so masking re-weighted every surviving target by 1/(1−f) = 1.331 and each masked arm trained a different objective from its dense twin |
| 4. Stage A + difficulty ladder | Aug 1 | corpus realised a 3.75% fact share against a requested 50–70%, on a synthetic bed; the fact lane ran dry and the bed silently absorbed the rest |

On 2026-08-02 an audit found seven further defects in the *unrun* pipeline,
before the confirmatory matrix spent any compute. They are catalogued in
`docs/AMENDMENT-2026-08-02.md`. The most consequential:

- **RANDPOS was never difficulty-matched in any corpus ever built.** The
  machinery existed and `build_corpus.py` never passed it the table.
- **Lane shares could be silently substituted.** This is what killed
  generation 4, and nothing detected it.
- **The confirmatory analysis pooled 3 loads x 8 seeds as n=24** when the same
  eight seeds recur at every load.
- **The dose-response shape test** — the only analysis that speaks to capacity
  rather than loss composition — **was wired to nothing and could not fire.**

---

## The two results that do stand

Both are negative, both are new, and both are catalogued with artifacts.

### 1. The endpoint is not learnable at 1,500 steps, at any difficulty

The difficulty ladder, rebuilt on valid corpora and completed 2026-08-02,
scores op=1 accuracy against the majority-class baseline:

| modulus | no-skill baseline | op=1 accuracy | clears? |
|---:|---:|---:|:--|
| 23 | 0.0753 | 0.0100 | no |
| 11 | 0.1327 | 0.0441 | no |
| 7 | 0.2053 | 0.0080 | no |
| 5 | 0.2573 | 0.0268 | no |

Nothing clears, including at modulus 5 where there are five possible answers
and the whole table is 232 bits. Difficulty is eliminated as an explanation.

A caution when reading these: headline accuracy sums two unrelated failures.
Only 51–82% of generations produce an answer inside the answer space at all —
the rest emit the deduction lane's `"no"`, nothing parseable, or raw biography
text. *Conditional on producing a valid answer*, accuracy at modulus 5 is
0.2500 against a 0.2573 majority rate: the marginal answer distribution,
exactly. The model has learned the answer prior and has not learned the
operation.

### 2. The matched control cannot be difficulty-matched, and that is a result

Preregistration §2 requires RANDPOS to match FACTMASK on mean masked-span NLL
within 20% relative, and states that failure "is reported as the result, not
gated away". Measured for the first time on the operating-point corpus:

| | mean masked-span NLL |
|---|---:|
| FACTMASK | 1.9598 nats |
| RANDPOS | 1.2712 nats |
| relative gap | **35.1%** against a 20% tolerance |

This is structural, not a tuning failure. Sweeping the placement's position
tolerance from 0.05 to 1.00 moves the gap from 42.2% to 23.5%, where it
asymptotes without reaching 20%. Fact values are intrinsically higher-entropy
than anything in a fixed biography template, so no set of non-value spans
averages 1.96 nats — those tokens do not exist in these documents.

**Consequence: FACTMASK removes about 54% more learnable signal than RANDPOS
does, by construction. Any FACTMASK − RANDPOS contrast will be confounded by
loss mass.** This generalises to any selective-loss or span-masking design that
pairs a treatment against a count-matched control, it costs one CPU pass to
check, and as far as we know nobody checks it.

---

## The citation rule, which is not optional

**`docs/RETAINED-RESULTS.md` is the sole catalogue of citable numbers. A number
that does not appear there has no artifact behind it and is withdrawn.**

The file ends with an explicit list of withdrawn figures — a held-out deduction
table, a bits-per-entity ledger, a four-way recognition probe, participation
ratios, and a claimed 49-to-196-exposure memorisation threshold. None has a
committed artifact; searches of the repository and the cluster found nothing
that produces them.

> ### Warning about `docs/section-null-without-crowding.md`
>
> That draft is included in this package because it is good writing and its
> Section 6 arithmetic is sound and worth keeping. **But Sections 1, 3 and 4
> rest on four of the withdrawn ledgers.** Do not build on those numbers, and
> do not carry them into a paper, until they have been reproduced and
> catalogued. `tests/test_repo_state.py::test_no_document_resurrects_a_withdrawn_number`
> fails on exactly this and is the one intentionally red test in the suite.
>
> Section 5 is fine: the 94.2% context-recoverability figure is genuinely
> retained.

---

## Reading order

1. `README.md` — the design in two pages.
2. `docs/RETAINED-RESULTS.md` — what may be cited, and what was withdrawn.
   Read before believing any number anywhere else.
3. `docs/AMENDMENT-2026-08-02.md` — the seven defects, why they are methods
   corrections rather than outcome-driven changes, and the appendix of what the
   new gates found when run against the cluster.
4. `docs/PREREGISTRATION.md` — the frozen design: arms, gates, inference,
   GO/NO-GO thresholds.
5. `docs/THEORY-ENDPOINT.md` and `docs/THEORY-CAPACITY.md` — why the endpoint
   floors, and the three conditions under which a crowding effect could exist
   at all.
6. `docs/NO-GO-PAPER.md` — the paper outlined *before* the pilot ran, so the
   pilot would produce its figures. This is the likely deliverable.
7. `docs/GATE0-CEILING-IS-NOT-A-BOUND.md` — a short methods note; a model can
   score far worse than the uniform ceiling by being confidently wrong.
8. `ops/crowding/RUNBOOK.md` — how to actually run any of it.

---

## What is running right now

Submitted 2026-08-02 on Stanford FarmShare (`/scratch/users/syz/crowding`):

- **`stepladder_d40m_std`** — 31,280 optimizer steps against every prior run's
  1,500, on the 16.4B-token corpus. This is the decisive experiment: the step
  budget is the last surviving explanation for the endpoint floor, since
  difficulty has been eliminated. **It should have finished around 05:00 US
  Central on 2026-08-03, and as of writing nobody has read it** — the Stanford
  VPN dropped overnight and took the multiplexed SSH session with it. Reading
  it is Step 0 below and takes minutes. It writes 20 snapshots, so scoring them
  also yields the accuracy-versus-steps curve for free, and that curve sets the
  cheapest sufficient step budget for the confirmatory matrix.
- **`high-e200`** — the new operating-point corpus, complete and `VERIFY OK`.
  996,408 entities x 200 exposures, 21.3B tokens, 40,695 steps, F/C = 1.000.
  The first corpus in the project to clear the full gate set.

### The decision that follows

If the step ladder clears, the endpoint is responsive, Stage B and Stage C
follow, then the confirmatory matrix (~1,495 GPU-h, roughly 13 days at
FarmShare's four-GPU cap). If it does not clear, both defences are spent and
`docs/NO-GO-PAPER.md` is the deliverable. Two of three independent reviewers
called the second outcome the most likely one before any of this ran.

---

## Repository layout

```
corpusgen/   biography records, iGSM-lite and Horn-clause deduction generators
             with independent oracles; randpos.py is the matched control
train/       GPT-2 BPE tokenizer, decoder-only GPT, memmap + sidecar loader,
             AdamW trainer with checkpoint/resume
evals/       greedy decoding, scorers, recoverable-bits accounting, paired
             statistics and the frozen verdict
ops/crowding/ corpus builder, difficulty-table freezer, config generation,
             Slurm submission, the difficulty ladder, status and advance
theory/      capacity and endpoint derivations, with the assertions in tests/
scripts/     training and evaluation drivers, the frozen analyzer, and a
             nine-second offline rehearsal of the whole chain
outputs/     retained artifacts only; see docs/RETAINED-RESULTS.md
```

## Running it

```bash
uv venv .venv --python 3.12
uv pip install -r requirements.txt --python .venv/bin/python
export PYTHONPATH=.
.venv/bin/python -m pytest tests -q
python3 scripts/smoke_pipeline.py            # builds, trains, evaluates, analyses
```

The smoke test rehearses the entire chain on CPU — a corpus build, all three
arms trained from one stream, evaluation of each, the frozen analyzer, and a
check that it refuses an incomplete matrix. `PIPELINE OK` is the expected last
line. It has already caught one bug that would have crashed evaluation partway
through the matrix.

**Expected test result in this package: 389 pass, 4 fail.** All four failures
are expected and none indicates broken code:

| failing test | why |
|---|---|
| `test_no_document_resurrects_a_withdrawn_number` | **intentional.** Fires on `docs/section-null-without-crowding.md`; see the warning above. Do not silence it. |
| `test_no_versioned_or_superseded_source_paths` | shells out to `git`; this package ships without `.git` |
| `test_every_retained_output_is_tracked` | same |
| `test_no_reference_to_a_deleted_module_survives` | same |

In the live repository, where `.git` is present, the count is 392 pass and 1
fail — the intentional one.

---

## Next steps

### Step 0 — read the step ladder (blocked only on a VPN reconnect)

The decisive run finished around 05:00 US Central on 2026-08-03. Nothing has
been read off it yet, because the Stanford VPN dropped overnight and the
8-hour multiplexed SSH session expired with it. The jobs were unaffected;
SLURM does not care whether anyone is watching.

```bash
bash cluster/connect.sh syz              # interactive Duo; then the rest is scriptable
bash cluster/sync_push.sh                # PUSH FIRST -- see the warning below
for s in $MS_ROOT/runs/stepladder_d40m_std/snapshots/step*.pt; do
  $MS_PY scripts/run_evals.py --run $MS_ROOT/runs/stepladder_d40m_std --ckpt $s
done
PYTHONPATH=. $MS_PY ops/crowding/ladder.py --mode steps \
  --run $MS_ROOT/runs/stepladder_d40m_std
```

> **Push before scoring.** The cluster is running code from before the
> format-compliance metric existed. Score the snapshots with the stale copy and
> `gain_attribution` returns `unavailable`, and every snapshot has to be
> rescored.

### Step 1 — the decision, and it is not the obvious one

`--mode steps` now returns a `gain_attribution` field, and **that field, not
headline accuracy, is the reading.** Accuracy can rise purely because the model
learned to terminate with a digit: on the difficulty ladder only 51–82% of
generations produced an answer inside the answer space at all, and among those
that did, accuracy sat at the marginal answer distribution exactly. A model
that learns nothing but formatting will walk headline accuracy up to the
majority rate and trip `first_clearing_step` — which is the number that sizes
the confirmatory matrix.

| `gain_attribution.reading` | meaning | what follows |
|---|---|---|
| **ARITHMETIC** | conditional accuracy pulled above the majority rate | Step 2A: the endpoint is alive |
| **FORMAT ONLY** | compliance rose, conditional accuracy did not | Step 2B. Do **not** size the matrix on `first_clearing_step` |
| **NEITHER** | both flat | Step 2B. Both defences are spent |

### Step 2A — if the endpoint is alive

1. **Freeze the primary load, and date it.** `analyze_crowding.py` now refuses
   to run without `--primary-load` because every rule for picking it from data
   chooses the estimand after seeing the outcome. This is a preregistration
   decision and it is still open.
2. **Read the cheapest sufficient step budget off the ladder.** If the endpoint
   becomes responsive at ~12,000 steps rather than 40,695, every matrix run
   shortens by the same factor and the matrix drops from ~1,495 GPU-h to ~441.
   This is the single largest cost lever in the project and it is free.
3. **Stage B** — SUP against NOFACT on `high-e200`, 2 runs. Measures `delta`,
   the total reasoning cost of carrying the fact load, which upper-bounds any
   achievable treatment effect. Also produces Figure 1, the
   exposure-to-storage frontier.
4. **Stage C** — three SUP/FACTMASK/RANDPOS triplets, 9 runs. Gives the paired
   SD that sets the seed count, plus an early leakage check.
5. **GO/NO-GO gate**, then the confirmatory matrix.

Costs, at the current operating point and 20.8 h per run:

| stage | runs | GPU-h | FarmShare wall (4-GPU cap) | AWS g6e spot |
|---|---:|---:|---|---:|
| Stage B | 2 | 42 | ~11 h | ~$27 |
| Stage C | 9 | 187 | ~2 days | ~$122 |
| Matrix, 3 loads x 3 arms x 8 seeds | 72 | 1,495 | ~13 days | ~$979 |
| Matrix, SUP trimmed to 3 seeds | 57 | 1,184 | ~10 days | ~$776 |
| Matrix, if the ladder allows 12k steps | 72 | 441 | ~4 days | ~$289 |

SUP can be trimmed to 3 seeds per load without cost to the estimand: it is the
manipulation check and the load main effect, not part of
`Y[FACTMASK] − Y[RANDPOS]`. Dropping from three loads to two would save a third
and is **not** advised — the shape test across loads is the only analysis in the
design that speaks to capacity rather than loss composition.

### Step 2B — if it is not

`docs/NO-GO-PAPER.md` becomes the deliverable. Its outline predates Stage A on
purpose, so the pilot was designed to produce its figures. Its contributions,
in the order they survive review:

1. The loss-normalization confound in selective-loss training.
2. **Matched controls are infeasible when the treatment masks the hard
   tokens** — measured, structural, and generalises well beyond this project.
   This is now the strongest standalone item.
3. The exposure-to-storage frontier. **Still needs Stage B**, so Stage B runs
   in this branch too.
4. `delta` as a bound on any masking intervention, for the price of one run.
5. The scale window, quantified: at 200 exposures the design cannot clear the
   preregistered exposure-saturation gate at 40M parameters, and the
   configuration that does clear it needs 259 GB of scratch and a 70-hour run.
6. A withdrawal, with `docs/RETAINED-RESULTS.md` as the audit.

Estimate: 3–6 weeks, most of it writing, plus Stage B's ~42 GPU-h.

### Housekeeping, once the step ladder has been read

- **Delete the old `corpora/high`.** It is held open only by the step-ladder
  run and frees ~28 GB. `high-e200` supersedes it.
- **Benchmark 32 / 64 / 128 build workers on one 800M rung** before building
  three matrix loads. The 128-worker build ran at 41M tok/min against the
  64-worker build's 225M with every core saturated and no memory pressure;
  three loads at that rate is a day of wall clock that may be buying nothing.

---

## Open decisions a second opinion would help with

1. **The primary load.** Still unfrozen; see Step 2A.1.
2. **The RANDPOS finding.** If the matched-control design is genuinely
   infeasible for this corpus, the options are to report it (the path taken,
   and what the preregistration prescribes), to redesign the control, or to
   raise the entropy of the template text until a match becomes possible.
3. **Which operating point is the preregistered one.** The repository contains
   two internally coherent designs that are not the same experiment — 5.737B
   tokens with 50/30/10/10 lane shares, versus 16.4B–21.3B with 70/12.3/4.3/13.4.
   Both fill the fact lane exactly. Only the first is written into
   `docs/PREREGISTRATION.md`, and the second is the one being run.
   `docs/AMENDMENT-2026-08-02.md` §7 lays out the comparison and deliberately
   does not choose.
4. **Scepticism about the endpoint generally.** iGSM has floored at every scale
   and difficulty tried. If it floors at 31,280 steps too, the honest
   conclusion is that the precondition for this experiment is unreachable at
   accessible scale, and the paper is about that.
5. **`paper/` is on the forbidden-directories list** in `test_repo_state.py`,
   left over from the deleted draft. Starting the paper proper needs that guard
   deliberately lifted rather than quietly worked around.
