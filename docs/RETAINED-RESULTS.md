# Retained results

Every artifact-backed result that survives the deletion of the superseded
experiment lines. **Nothing outside this file may be cited in the paper.**
A number without an entry here has no artifact behind it and is withdrawn.

Hashes: `outputs/RETAINED-SHA256SUMS` (110 files).
Written 2026-08-01, before the deletion in Task 0 Step 2.
Reconciled against the decoding defect 2026-08-05; see the next section.

---

## The decoding defect withdraws every generative accuracy number

`evals/generate.py::generate_batch` left-padded each prompt to the batch
maximum with EOT and ran attention over the pads, with no attention mask. EOT
is the document separator, so a prompt behind enough pads read as the start of a
fresh document and the model invented its premises. Corruption is monotone in
pad length: exact at 0–16 pads, broken by 32.

**This is not a small bias, and the affected numbers are not approximations of
the true ones.** The same checkpoint, on the same 64 items, scored **3/64 =
4.7% batched and 60/64 = 93.8% one prompt at a time**, and 60/64 again once the
padding was removed. A number produced by the old path carries no information
about the quantity it was supposed to measure, so it is **withdrawn**, not
"approximate" and not "pending re-measurement". Where such a number was the only
support for a claim, the claim is withdrawn with it.

Fixed 2026-08-04. `generate_batch` now decodes in groups of exactly equal prompt
length, so no padding exists and no mask is required, and
`tests/test_generate.py` asserts that batching never changes a generation.

**What decides whether a section below survives** is the path that produced its
numbers, and there are exactly three:

| path | affected? | why |
|---|---|---|
| batched generation via `generate_batch` | **yes, withdrawn** | unmasked left pads corrupt the prefill |
| teacher-forced NLL via `evals/storage.py` | no | pads are written to the **tail** (`storage.py:135–137`), and under causal attention position `len(p)+j−1` never attends past itself |
| training loss, corpus statistics, wall-clock throughput | no | no decoding involved |

Every section below now carries a **Produced by** line recording which path it
came from. Sections 1, 5 and 6 do not survive.

---

## The loss-normalization caveat applies to every masked run below

`train/model.py` computed `F.cross_entropy(..., ignore_index=-100)` with the
default `reduction='mean'`, which averages over **surviving** targets only.
Masking `f` of a document's targets therefore multiplied every remaining
target's weight by `1/(1-f)` — 1.331 inside fact documents at the measured
24.89% mask rate.

Consequence: **every split/masked arm in this repository was trained under a
different effective objective than its dense twin, over and above the
masking itself.** Those runs remain valid evidence that masking keeps values
out of the weights. They are not valid arm contrasts, and no dense-minus-split
difference below may be reported as a treatment effect.

Fixed in Task 1. Runs produced after that fix are not affected.

---

## 1. 0.8B gate evaluations — WITHDRAWN, produced by the padded decoder

`outputs/cluster-summaries/evals/` — four runs, 24 files.

**Produced by.** Batched generation via `generate_batch`, before the fix.
Confirmed in the artifacts themselves: every row of `igsm.jsonl`,
`deduction.jsonl`, `factqa.jsonl` and `factqa_fresh.jsonl` carries a free-form
`pred` string, and the first `igsm` row reads `pred: None` — the signature of a
corrupted prompt emitting nothing parseable. `evals/scorers.py:54` calls
`generate_batch`.

**Withdrawn.** All four accuracy columns — iGSM, deduction, fact-use QA and
fresh-entity QA — for all four runs. The numbers are deliberately not reproduced
here. `docs/RESULTS-2026-08-04.md` names this section explicitly among what the
defect overturns.

The **99.6% fresh-entity figure goes with them**, and it is worth being
explicit about that, because it was the most quoted number in this section and
it reads as a clean mechanism demonstration. It is a generated answer string
scored for exact match, so it came through the same path as everything else
here, and there is no version of it that survives.

**What survives.** Only facts about the item set, which no model touched: the
deduction eval is exactly 200 yes / 200 no in strict alternation with a canned
NO-branch trace, so a constant "no" scores exactly 0.500; and the empirical
majority-class baseline on iGSM is ~7.2% rather than 1/23 = 4.3%, because
`times` overproduces zero. Both remain true and both are properties of the
generator, not measurements of a model.

**To restore any of it** the four checkpoints must be rescored with the fixed
decoder. The per-item JSONL files are retained so the rescore is a comparison
rather than a fresh baseline.

## 2. 160M dose sweep — training only

`outputs/farmshare-160m-sweep/` — 12 runs, 3 fact loads x 2 arms x 2 seeds, all
at step 6,103 of a 3.2B-token budget, plus two 1B dense calibration gates.
Logs, configs, `ANALYSIS.txt`, and `CHECKPOINTS-SHA256SUMS` for the terminal
checkpoints backed up to FarmShare home storage.

**Produced by.** Training loss only. **Survives the decoding fix** — no
generation was ever run on these checkpoints, which `ANALYSIS.txt` records as
`[FAIL] evaluations present`. The 25.03-nat gate-0 correction below is also a
loss measurement and survives.

**Supports.** That training is highly reproducible — seed spread in final loss
is at most 0.0086 nats. The split-arm cross-entropy at masked positions,
9.64–12.30 nats across all six runs and both seeds, is a reproducible
measurement; what it *means* is corrected below.

**Does not support.** Any endpoint claim. **No evaluations were ever run on
these twelve checkpoints**, and `ANALYSIS.txt` records `[FAIL] evaluations
present`. The dense-minus-split training-loss gap is not a treatment effect,
because the arms score different target sets.

**The 10.83-nat ceiling these numbers were read against is not a bound at
all.** Measured 2026-08-02 on a *dense* arm that trained on those positions:
25.03 nats, with 98.7% of 32,517 positions above the uniform ceiling, model
entropy 3.83 nats and argmax accuracy 0.0001. A language model is never
uniform — it concentrates mass on frequent tokens, so at positions requiring
rare tokens it necessarily scores worse than uniform.

So a split arm landing near 10.8 had learned the slot type and was uncertain
within the pool, not "pinned at the ceiling knowing nothing"; a model that
knows nothing scores 25+. The agreement with ln(V) was a coincidence read as
confirmation. Full write-up and the reproduction command:
docs/GATE0-CEILING-IS-NOT-A-BOUND.md.

## 3. 160M reasoning-v3 seed-0 pair, and the exposure census

`outputs/farmshare-160m-v3-s0/` — one dense/split90 pair at 8.17B tokens, plus
`fact-exposure.json` from a full pass over the corpus sidecar.

**Produced by.** A counting pass over the corpus sidecar, plus training loss.
**Survives the decoding fix** — nothing here was decoded.

**Supports.** The exposure census, which is the most useful number in this
directory: 61,453,327 fact-value occurrences over 1,453,989,440 masked tokens
(20.419% of the base segment), mean span 23.66 tokens, and **1.04 to 1.55
exposures per fact, median 1 under every definition tested**. That is the
corpus-level reason the reasoning-v3 line could not test the hypothesis.

**Does not support.** The arm contrast (+0.0011 nats on the reasoning
extension), for the loss-normalization reason and because it is n=1.

**Known defect in these runs.** Gate 0 never fired. The probe sampled the first
262,400 tokens while the first masked target sits at token 356,253,723, then
cached the miss and disabled itself permanently. Every reasoning-v3 cohort,
including the 1B runs, has zero gate-0 datapoints.

## 4. Tiny crowding cohort

`outputs/farmshare-tiny/` — d8m and d40m, each a dense/split90 pair at 8.17B
tokens, all four at step 15,582. Plus `LR-PROBE.md`, a 9-point learning-rate
grid with both optima bracketed on either side.

**Produced by.** Wall-clock throughput and training loss. **Survives the
decoding fix** — the 94.2% figure is a ratio of cross-entropies at masked
positions, not an accuracy, and no generation was involved.

**Supports.** Measured throughput on one L40S at `ctx=1024`,
`micro_batch_size=32`, compiled: **d8m 462,611 tok/s, d40m 184,671 tok/s**.
These are the numbers all compute planning is sized from. And the gate-0
mechanism result, the first this metric ever produced on reasoning-v3: at 40M
the split arm reaches 5.4264 nats on the offloaded positions against dense's
5.0943, recovering **94.2% of dense's score without ever training there**.
Supervision on those tokens was worth 5.8% of the total.

**Does not support.** The arm contrasts (+0.0028 / +0.0009 / +0.0011 nats),
which are n=1 and smaller than the ±0.0086-nat seed spread measured in the
sweep, on top of the loss-normalization caveat. The across-size trend confounds
parameters with recurrence, embedding tying and learning-rate tuning.

## 5. 1B held-out reasoning evaluation — WITHDRAWN

`outputs/1b-heldout/` — moved out of the deleted paper draft. The seed-0
dense/split90 pair's training logs and configs, the 7,168-item held-out
Reasoning-Gym set (`items.jsonl`), five snapshot scores per arm, a
gemma-3-1b-it-qat-4bit reference, and the measured mask schedule.

**Produced by.** Every eval JSON in `outputs/1b-heldout/evals/` carries
`"scoring": "teacher_forced_greedy_agreement"`, so this did **not** go through
`generate_batch`.

**There is a genuine conflict in the record here, and it resolves against the
numbers.** `docs/RESULTS-2026-08-04.md` lists "the 1B held-out result" among
what the decoding defect overturns, while the artifact metadata says a
different scorer produced it. Both cannot be right. The scorer itself cannot
settle it, because it is one of the modules deleted with the superseded
experiment lines and `tests/test_repo_state.py:118` now forbids referring to
it, so **there is no code in this repository that can reproduce or even inspect
these numbers.** Teacher forcing is not inherently safe either: it is corrupted
by unmasked left pads exactly as sampling is, since the damage is to the
prefill and not to the token-selection step.

**Withdrawn**, on two independent grounds: the project's own retraction names
it, and the harness that produced it is gone. The accuracy figures and the
+5.85-point gap are not reproduced here.

**What survives.** The item set and the generator: `items.jsonl` with 7,168
items across fourteen families, and zero oracle rejections, which is a property
of the generator rather than a measurement of a model. The prequential loss
putting the arms level at +0.0022 nats is a loss measurement and survives the
decoding fix, though it remains subject to the loss-normalization caveat above
and to n=1.

## 6. PopQA held-out key construction — WITHDRAWN, harness deleted

`docs/POPQA-HELDOUT-KEY.md`.

**Produced by.** Generated key strings under a bespoke copy-constrained trie
decoder, on a toy 4-layer / 256-dim ~29M model at ctx 192, CPU, one seed.

**Withdrawn.** Not because `RESULTS-2026-08-04.md` names it — it does not — but
because the decode path cannot be checked and cannot be rerun. The doc names
its harness as `corpusgen/realfact.py`, `evals/keyguess.py`, `evals/constrain.py`
and `scripts/run_keyguess_local.py`, and **none of those files exists in this
repository.** `evals.keyguess` is on the deleted-module list in
`tests/test_repo_state.py:118`, and `keyguess` is on the forbidden-path regex at
line 27. So there is no way to establish whether that decoder padded its batches,
and no way to rescore the checkpoint if it did.

Under this file's own rule — a number with no artifact behind it is withdrawn —
the relation-request, entity-recall and copy-constrained recovery figures are
gone, and they are not reproduced here.

**What survives.** The pinned input snapshot `data/realfacts/popqa_clean.jsonl`
with its SHA-256, and the qualitative mechanism claim that copying a prompt
entity and recalling a memorized key are indistinguishable in loss on this
training distribution. That claim is an argument about the objective, not a
measurement, and it stands on its own reasoning.

## 7. The matched control cannot be difficulty-matched, and that is a result

`outputs/pilot/` and the `high-e200` corpus manifest on FarmShare
(`randpos_validity`), built 2026-08-02, job 1673768, `VERIFY OK`.

**Produced by.** A CPU pass over the corpus computing per-token NLL against a
frozen difficulty table. **Survives the decoding fix** — no model output was
decoded, and the quantity is a property of the corpus and the difficulty table.

Preregistration §2 requires RANDPOS to match FACTMASK on mean per-token NLL
within 20% relative, and states that failure "is reported as the result, not
gated away". This is the first corpus in the project for which the quantity was
ever measured: every earlier corpus matched count, span length and relative
position, because `build_corpus.py` never passed `token_nll`.

Measured over all 21.3B tokens of the operating-point corpus (996,408 entities
x 200 exposures, F/C = 1.000):

| | mean masked-span NLL |
|---|---:|
| FACTMASK | 1.9598 nats |
| RANDPOS | 1.2712 nats |
| relative gap | **35.1%**, against a 20% tolerance |

**Supports.** That the matched-non-value-span control is infeasible for this
corpus construction, and that the infeasibility is structural rather than a
tuning failure. Sweeping the placement's position tolerance on the same corpus
and the same frozen difficulty table:

| position tolerance | NLL gap | cue-window overlap |
|---:|---:|---:|
| 0.05 | 42.2% | 32.5% |
| 0.15 (shipped) | 35.0% | 24.5% |
| 0.40 | 26.8% | 19.4% |
| 0.60 | 24.2% | 18.7% |
| 1.00 (unconstrained) | 23.5% | 18.5% |

The gap asymptotes at 23.5% and never reaches 20% even with positional matching
abandoned entirely. Fact values are intrinsically higher-entropy than anything
else in a fixed biography template, so no set of non-value spans has a mean NLL
of 1.96 — those tokens do not exist in these documents. Relaxing the tolerance
improves both measured axes at once, so there is no trade-off between them,
only a loss of positional matching.

Also measured: **24.5% of control-masked tokens land inside a value's
cue window** (the three tokens preceding a value), so the control removes
fact-relevant supervision across roughly a quarter of its mass and is biased
toward the treatment. On the ladder corpora, built without a difficulty table,
the same figure was 31.1%.

**Does not support.** Any statement about the size or sign of
`Y[FACTMASK] − Y[RANDPOS]`, which has still never been measured. What it
establishes is that when that contrast is measured, it will be confounded by
loss mass: FACTMASK removes about 54% more learnable signal than RANDPOS does,
by construction and unavoidably.

**Generalises beyond this project.** Any selective-loss or span-masking design
that pairs a treatment against a count-matched control inherits this problem
whenever the masked content is intrinsically harder than the material available
to match it. The check costs one pass over the corpus and no GPU time.

**Provenance.** Difficulty table frozen from the step-ladder run's step-1,564
snapshot on burned pilot seed 9001, with `nll_table.provenance.json` recording
the checkpoint hash. Not rebuilt after the measurement was seen.

## 8. The evaluation defect, and what the endpoint does once corrected

`runs/stepladder_d40m_std/` on FarmShare — 1,531,800 entities x 100 exposures
(F/C = 2.0), 16.4B tokens, 31,280 steps, `loss_ema` 1.2512, `clip_frac` 0.0.
Twenty snapshots, scored with the corrected decoder.

**Produced by.** Batched generation via `generate_batch` **after** the fix,
cross-checked against one-prompt-at-a-time decoding. **Survives** — this section
is the measurement of the defect, and it is the only place in this file where a
generative accuracy number is citable.

**Supports.** That batched greedy decoding left-padded prompts to the batch
maximum with EOT and attended over the pads, and that this floored every
generative accuracy number the project produced. On this checkpoint, 64
held-out iGSM items at MOD 23: **3/64 batched, 60/64 unbatched, 60/64 batched
after the fix.** An independent replication at a different seed and
`max_new` obtained **0/64 old, 63/64 unbatched, 63/64 fixed, with 64/64
string-identical between fixed-batched and unbatched**. Corruption sets in
between 16 and 24 pad tokens.

That the endpoint is responsive: at step 14,076, on the full 1,500-item eval,
op1 100.0% (n=399), op2 98.2% (n=384), op3 88.3% (n=375), op4 73.1% (n=342);
OOD op5-8 at 25.7 / 8.2 / 5.3 / 5.5%. Deduction rises 0.533 to 0.783 against a
0.500 constant-answer baseline. In-answer-space rate is 1.000.

**Does not support.** The 48-item per-op table in earlier drafts, whose cells
carry 8-17 items. The final-checkpoint 1,500-item table is not yet written.
Nothing about arm contrasts: this is a SUP-only run.

**Known defect in the artifacts.** `evals/igsm.jsonl` is overwritten by
whichever checkpoint was scored most recently and does not record which. Two
reviewers independently mistook a step-14,076 file for the final checkpoint.
Per-op numbers must be taken from `evals/step*.json`, which are tagged.

## 9. Facts are not retrievable at twice capacity

**Produced by.** Gate 0 is a training-loss measurement. The storage probe is
teacher-forced NLL through `evals/storage.py`, which writes content at the head
of each row and fills the tail with EOT (`storage.py:135–137`), so under causal
attention the value positions at `len(p)+j−1` never attend to a pad.
**Survives the decoding fix**, verified in the source rather than assumed.

Same run. Gate 0, logged once at step 31,280: **1.9732 nats** at fact-value
positions under training phrasing.

The storage probe, under held-out `"Record lookup. {name} has ..."` templates
that `tests/test_storage.py` asserts are disjoint from the training templates:

| cohort | recoverable bits per entity |
|---|---:|
| trained, 100 exposures each | **-112.90** |
| never seen | **-112.29** |
| difference | **-0.61 +/- 0.68 (SE)** |

300 entities per cohort, disjoint generators, final checkpoint, both cohorts
scored in the same call.

**Supports.** That the model's behaviour on facts it saw 100 times is
statistically indistinguishable from facts it never saw. This is the control
that rules out the "unfamiliar query format" reading: a format failure would
depress both cohorts, but knowledge present in any addressable form would
separate them.

**Does not support.** Any claim at or below capacity. This run sits at
**F/C = 2.0**, which `docs/THEORY-CAPACITY.md` predicts is the abandonment
regime, so the result is consistent with the model declining to enter storage
rather than with storage being impossible. The 4,000-entity probe in the
snapshot summaries reports **-126 bits/entity** scaled from `n_entities =
1,531,800`; an earlier draft divided by 996,408 and reported -195, which was
wrong.

**Do not pair gate 0 with a snapshot's recoverable bits.** `eval_every` exceeds
`max_steps`, so gate 0 is logged once at the end and `log_diagnostics` copies
that same value into every snapshot summary. The two numbers in this section
are both from the final checkpoint; any other pairing is confounded.

## 10. Which matching criterion blocks the difficulty match

Computed on `high-e200` with the frozen difficulty table, 800 fact documents,
22,416 value tokens. Reproduce with the placement in `corpusgen/randpos.py`
against `corpora/nll_table.npy`.

**Produced by.** A corpus-level NLL computation. **Survives the decoding fix.**

| control construction | mean masked-span NLL | gap to treatment |
|---|---:|---:|
| treatment (fact values) | 1.9584 | — |
| shipped: contiguous, length- and position-matched | 1.2712 | 35.1% |
| contiguous, length-matched, position unconstrained | 1.4987 | 23.5% |
| best possible contiguous and length-matched | 1.5667 | 20.0% |
| scattered tokens, count- and mean-matched | 1.7677 | 9.8% |
| scattered tokens, the k hardest available | 1.8644 | 4.9% |

**Supports.** That span-length matching, not the corpus and not position, is
what prevents the control from matching difficulty. Tokens hard enough exist;
they are not arranged in contiguous runs. An oracle contiguous control lands
exactly on the preregistered 20% tolerance.

**Does not support.** That no control construction works. Scattered placement
reaches 9.8% and is a valid difficulty match; it trades away the span-structure
match instead, which is a different confound and untested.

## 11. Lane substitution, as realised shares

`outputs/pilot/pilotA-corpus-manifest.json`, Stage A, 800M tokens.

**Produced by.** Realised lane shares read off a corpus manifest, plus
`loss_ema` from training. **Survives the decoding fix.**

| lane | requested | realised |
|---|---:|---:|
| fact | 50% | 3.75% |
| igsm | 30% | 30.00% |
| deduction | 10% | 10.00% |
| bed | 10% | 56.25% |

Bed absorbed 46.3 percentage points over its request. The 2026-08-01 difficulty
ladder was worse, at 3.75% against a requested 70% with a 79.65% synthetic bed;
those manifests are under `invalid-20260802/` with `WHY.md`, and their runs
settled at `loss_ema` 2.31 against Stage A's 1.76.

**Supports.** That a finite lane can exhaust and have its budget silently
reallocated while token count stays exact and every integrity check passes.

**Does not support.** Anything about the rebuilt ladder of 2026-08-02, whose
runs settle near 1.53 and which used a pinned bed.

---

## Withdrawn, listed so nobody re-derives them

### A. Withdrawn 2026-08-05 by the decoding defect or a missing harness

These *do* have committed artifacts. They are withdrawn because the path that
produced them is known to be broken, or because the code that produced them is
no longer in the repository. Do not cite them, and do not treat them as
approximations to be refined.

| # | withdrawn | ground |
|---|---|---|
| §1 | all four accuracy columns across all four 0.8B gate runs, including the 99.6% fresh-entity figure | free-form `pred` strings from `generate_batch` before the fix |
| §5 | the 1B held-out accuracies for both arms, the gemma reference, and the +5.85-point gap | named in `RESULTS-2026-08-04.md`; scorer deleted, so unverifiable and unreproducible |
| §6 | the PopQA relation-request, entity-recall and copy-constrained recovery figures | `evals/keyguess.py` and its siblings do not exist; decode path cannot be checked |

That is **three sections and eleven reported figures**, against eight sections
that survive with their provenance now recorded.

**The general rule this establishes.** Any accuracy in this project produced
before 2026-08-04 by batched generation is withdrawn. Loss measurements, corpus
statistics, throughput, and teacher-forced NLL through `evals/storage.py` are
unaffected, and the reason is structural rather than a judgement call: the
defect is unmasked attention over left pads during prefill, so it reaches
anything that decoded a padded batch and nothing that did not.

### B. Withdrawn earlier, for having no artifact at all

These appear in the deleted paper draft and its supporting documents. None has
a committed artifact; searches of both the repository and the cluster found
nothing that produces them.

- Held-out deduction at 160M / 3.2B: 65.0/63.0, 66.7/69.8, 68.2/62.7. Hardcoded
  in `main.tex`. The only deduction numbers with artifacts are the 0.8B gate
  values in section 1, which do not match.
- Bits stored per entity: 33.1 / 0.25 / 0.20.
- Four-way recognition probe: 0.90 at dense-50k, 0.25 elsewhere.
- Participation ratios: 6.7 / 1.8 / 2.6.
- The claim that memorization appears between 49 and 196 exposures per fact.
  The exposure arithmetic is sound; the storage side of it rests on the
  withdrawn bit ledger.
- The 5.6% split iGSM figure in the documentation-sync note, which reads as
  positive but sits below the ~7.2% majority-class baseline.
