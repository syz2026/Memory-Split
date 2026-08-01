# Retained results

Every artifact-backed result that survives the deletion of the superseded
experiment lines. **Nothing outside this file may be cited in the paper.**
A number without an entry here has no artifact behind it and is withdrawn.

Hashes: `outputs/RETAINED-SHA256SUMS` (110 files).
Written 2026-08-01, before the deletion in Task 0 Step 2.

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

## 1. 0.8B gate evaluations — the only per-item eval artifacts in the project

`outputs/cluster-summaries/evals/` — four runs, 24 files, including per-item
JSONL with `answer` and `pred` on every row.

| Run | iGSM | Deduction | fact-use QA | fresh-entity QA |
|---|---:|---:|---:|---:|
| `d160m_dense_n50k_s0_gate` | 0.0387 | 0.3693 | 0.3087 | 0.002 |
| `d160m_dense_n200k_s0_gate` | 0.0320 | 0.5373 | 0.3120 | 0.004 |
| `d160m_dense_n800k_s0_gate` | 0.0440 | 0.5133 | 0.3080 | 0.008 |
| `d160m_split_n200k_s0_gate` | 0.0560 | 0.5313 | 0.5813 | 0.996 |

160M parameters, 0.8B tokens, 1,500 items per cell, one seed.

**Supports.** That the lookup mechanism works: the split arm answers 99.6% of
questions about entities it never read about, with the store attached, and 0%
with it detached. That iGSM was at floor at this budget.

**Does not support.** Any arm contrast (see the caveat above). Any iGSM claim
scored against 1/23 = 4.3% — the empirical majority-class baseline is ~7.2%
because `times` overproduces zero, so **every iGSM number here is below a
constant predictor**, including the 0.056 split figure that reads as positive.
The fact-use QA gap compares a model with an oracle store against a closed-book
model and is a system comparison, not a capacity one.

**Recoverable from these files without rerunning anything:** per-class
deduction accuracy, which no summary reports and which matters because the eval
is exactly 200 yes / 200 no in strict alternation with a canned NO-branch
trace, so a constant "no" scores exactly 0.500.

## 2. 160M dose sweep — training only

`outputs/farmshare-160m-sweep/` — 12 runs, 3 fact loads x 2 arms x 2 seeds, all
at step 6,103 of a 3.2B-token budget, plus two 1B dense calibration gates.
Logs, configs, `ANALYSIS.txt`, and `CHECKPOINTS-SHA256SUMS` for the terminal
checkpoints backed up to FarmShare home storage.

**Supports.** That the masking mechanism is behaviourally sharp: split-arm
cross-entropy on the masked value positions lands at 9.64–12.30 nats across all
six split runs and both seeds. That training is highly reproducible — seed
spread in final loss is at most 0.0086 nats.

**Does not support.** Any endpoint claim. **No evaluations were ever run on
these twelve checkpoints**, and `ANALYSIS.txt` records `[FAIL] evaluations
present`. The dense-minus-split training-loss gap is not a treatment effect,
because the arms score different target sets.

**Also invalid: the 10.83-nat ceiling these numbers are read against.**
`ln(50304)` assumes uniform over the vocabulary. Values come from pools of
100–3,719 with predictable multi-token continuations and visible preceding
payload tokens, so the correct null is conditional entropy given attribute,
length and context. The gate-0 reading is qualitatively right and
quantitatively wrong.

## 3. 160M reasoning-v3 seed-0 pair, and the exposure census

`outputs/farmshare-160m-v3-s0/` — one dense/split90 pair at 8.17B tokens, plus
`fact-exposure.json` from a full pass over the corpus sidecar.

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

## 5. 1B held-out reasoning evaluation

`outputs/1b-heldout/` — moved out of the deleted paper draft. The seed-0
dense/split90 pair's training logs and configs, the 7,168-item held-out
Reasoning-Gym set (`items.jsonl`), five snapshot scores per arm, a
gemma-3-1b-it-qat-4bit reference, and the measured mask schedule.

Terminal step 15,582: dense 26.87%, split90 32.71%. Reference: gemma 7.70%
zero-shot, 12.11% two-shot.

**Supports.** That the held-out item generator and the teacher-forced greedy
scorer work, with zero oracle rejections across all fourteen families.

**Does not support.** The +5.85-point gap, for four independent reasons: one
seed pair; an unsealed endpoint built after the checkpoints existed; split ahead
on only 6 of 14 families with the aggregate dominated by `spiral_matrix` going
0.0% to 84.8%; and prequential loss on the *same* tokens putting the arms level
at +0.0022 nats, which is only consistent with a few families crossing an
acquisition threshold rather than a capability difference. Plus the
loss-normalization caveat.

## 6. PopQA held-out key construction

`docs/POPQA-HELDOUT-KEY.md`.

**Supports.** A clean negative on transfer to real entities: the model learned
which relation to request (98.5% against a 6.25% baseline) and failed to learn
which entity to ask about (2.5%), because on the training distribution copying
the prompt entity and recalling a memorized key are indistinguishable in loss,
and the model took the memorization route. Copy-constrained decoding repaired
part of it, moving full-key accuracy from 0% to 24.3% on the same checkpoint.

**Does not support.** Anything about capacity. One toy seed at ~29M parameters.
The preregistered threshold was not met.

---

## Withdrawn, listed so nobody re-derives them

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
