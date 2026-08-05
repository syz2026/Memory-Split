# Four ways a small-model reasoning experiment silently measures nothing

Draft, 2026-08-04. Independent verification by two reviewers in progress; every
numeric claim below is marked with the artifact it comes from, and anything
they overturn will be struck rather than quietly edited.

---

## Abstract

We report four defects found in a controlled study of whether masking factual
targets frees model capacity for reasoning. Each defect silently produced a
plausible-looking null, and each is invisible to the checks the study already
performed. The largest is an evaluation bug: batched greedy decoding
left-padded prompts with the document-separator token and attended over the
pads, which put every generative accuracy number in the project at its floor.
Correcting it moves the same checkpoint on the same items from **4.7% to
93.8%**, and overturns four experiment generations' worth of conclusions about
an "unusable" endpoint. The other three are a matched control whose four
matching criteria turn out to be mutually unsatisfiable, a corpus builder that
substitutes one data lane for another without failing, and a divergence between
fitting fact text and retrieving facts. None of these is specific to our hypothesis; all of them
apply to selective-loss training, span-masked language modelling, and
synthetic-corpus reasoning evaluation generally. We give the diagnostic for
each, all of which are cheap, and none of which we found in the literature.
The last defect also yields a substantive result: at twice achievable capacity,
a model shown 1.5 million facts a hundred times each acquires no retrievable
knowledge of any of them, scoring indistinguishably from facts it never saw.
The fact store whose capacity the hypothesis proposes to free does not exist at
this operating point.

---

## 1. Why this is a paper

The study these defects come from asks whether removing factual memorisation
from a small model's weights frees capacity for reasoning — the rationale the
phi-3 team gives for filtering fact-heavy pages from training data. We ran four
generations of the experiment over sixteen days. Every one returned a null, and
every one turned out to have failed on a precondition rather than on the
hypothesis.

That pattern is the contribution. Each failure produced a result that looked
like evidence: a floored endpoint, an inert corpus, a treatment effect
indistinguishable from zero. A reader of any single generation's output would
have concluded the hypothesis was false, or that the scale was wrong. The
defects were only found by instrumenting the things the experiment assumed.

We report them because the assumptions are common and the diagnostics are
cheap. Three of the four cost no GPU time at all.

## 2. Setting

A 40.6M-parameter tied-embedding decoder (21.2M non-embedding), trained from
scratch on a synthetic corpus interleaving four lanes: templated biographies
carrying arbitrary facts, iGSM-style modular-arithmetic word problems with
chain-of-thought traces, Horn-clause deduction, and a natural-text bed of
FineWeb-Edu. The reasoning endpoint is closed-book exact-match accuracy on
held-out iGSM problems, scored against the empirical majority-class baseline.
Runs are 21.3B tokens, 40,695 optimizer steps, on one NVIDIA L40S.

---

## 3. Finding 1 — Left-padding with the separator token floors generative evaluation

### The defect

Batched greedy decoding padded every prompt to the batch maximum so a single
prefill would produce a next-token distribution for each row. Padding was on
the left, with the end-of-text token, and there was no attention mask. The
code's comment asserted two things:

> With RoPE a constant left shift is harmless for greedy decoding at these
> scales; the pads are ordinary EOTs the model has seen as separators.

Both are wrong. Without a mask the pads are attended to as ordinary context.
EOT is the document separator, so a prompt behind enough pads reads as the
beginning of a fresh document, and the model generates a *different, plausible*
document rather than answering the prompt it was given.

The failure is silent in the worst way: output is fluent, on-format, and
on-topic. It parses. It is simply about premises the model invented.

### Magnitude

Run `stepladder_d40m_std` at step 31,280, 64 held-out iGSM problems at MOD 23,
identical items and checkpoint throughout:

| decoding | exact-match accuracy |
|---|---:|
| batched, padded to batch maximum | **3 / 64 = 4.7%** |
| one prompt at a time | **60 / 64 = 93.8%** |
| batched, after the fix | **60 / 64 = 93.8%** |

Corruption is monotone in pad length. Holding one prompt fixed and varying only
the length of a companion prompt in its batch:

| pad tokens | answer correct |
|---:|---|
| 0, 8, 16 | yes |
| 24, 32, 64 | no |

An independent replication placed the threshold at 24 pad tokens on a
63-token prompt, so the transition is somewhere in 16–24 and our coarser
sweep overstated how clean it is.

### Why it looked like a property of the model

Two anomalies had already been observed and attributed to the model. Both are
consequences of the defect.

**The difficulty profile ran backwards.** Accuracy rose with the number of
reasoning operations — the opposite of what a reasoning model should do — and
this was read as the signature of a model emitting a near-constant answer.
Mean prompt length rises with operation count (100, 136, 206 and 245 tokens for
op 1 through 4), and padding is the gap between a prompt and the longest prompt
in its batch. Short problems were padded most and scored worst. In a batch of
64 mixed problems, op-1 items scored 0/13 and op-4 items 2/18.

**Half of all generations fell outside the answer space.** At MOD 5, where only
five answers exist, 51.5% of generations produced an answer in {0..4}; the rest
emitted the deduction lane's `"no"`, nothing parseable, or biography text. This
was diagnosed as a formatting failure and a metric was built to separate it from
arithmetic failure. It was corrupted prompts. The first checkpoint scored after
the fix reports an in-answer-space rate of **0.9993**, and every subsequent one
reports **1.0000**.

### The diagnostic

Decode one prompt alone and compare it to the same prompt in a mixed-length
batch. If they differ, the evaluation is measuring the harness.

We recommend asserting this in a test rather than checking it once. Batching is
a throughput decision; it must never be a semantic one.

### The fix, and its cost

We decode in groups of exactly equal prompt length, so no padding exists and no
mask is required. This is not the only fix — an attention mask over the pads is
the general one — but it is correct by construction and needs no change to the
model. It costs throughput when prompt lengths are diverse: on a 1,500-item iGSM
mix a nominal batch of 32 fragments into roughly 29 length-groups, so batching
is very nearly eliminated. An attention mask would keep both correctness and
throughput and is the better fix for anyone who needs the speed.

### Scope of the damage

Every generative accuracy number this project produced across four experiment
generations, including: a pilot that concluded the endpoint "has no power to
discriminate anything"; a difficulty ladder that concluded the task was "not
learnable" at a modulus with five possible answers; and a step-budget analysis
arguing the endpoint needed 20x more optimizer steps than it was given. The
endpoint was never step-limited, difficulty-limited, or scale-limited.

Measurements that did not pass through generation are unaffected — training
loss, gradient statistics, and teacher-forced probes that pad on the right,
where causal attention cannot see the pads.

---

## 4. Finding 2 — What the endpoint does, once it can be measured

The same checkpoint, 48 items per operation band, MOD 23:

| operations | accuracy | |
|---:|---:|---|
| 1 | 10/10 = 100% | ceiling |
| 2 | 17/17 = 100% | ceiling |
| 3 | 9/9 = 100% | ceiling |
| 4 | 9/12 = 75% | responsive |
| 5 | 5/11 = 45% | responsive |
| 6 | 0/12 = 0% | floor |
| 7 | 0/17 = 0% | floor |
| 8 | 1/8 = 12% | floor |

**These per-cell counts (8–17) are too small to publish and are reported here
only as a provisional shape.** A 1,500-item evaluation is in progress; at step
14,076, 45% of the way through training, it gives op1 100.0% (n=399), op2 98.2%
(n=384), op3 88.3% (n=375) and op4 73.1% (n=342) — so op3 is not at ceiling
mid-training, and "perfect through three, cliff at six" is not supported at
large n. The final-checkpoint table at n=1,500 replaces this one before
submission.

Two independent reviewers flagged this table by comparing it against
`evals/igsm.jsonl`, which is overwritten by whichever checkpoint was scored
most recently — a second instance of an artifact that does not say which model
produced it.

Operations 5–8 were never trained on: the corpus was built with an operation
band of [1,4]. Those columns therefore measure length generalisation, and the
45% at op 5 is generalisation one step beyond the training distribution.

**This creates the opposite problem to the one the project spent four
generations on.** In-band accuracy aggregates to 93.8%, which fails a
responsiveness criterion from the *ceiling* side: there is no headroom for a
treatment to show an effect. An endpoint has to be re-tuned in the direction
opposite to the one everybody assumed.

A partial learning curve, from snapshots of the same run:

| step | iGSM | deduction | reference-trace NLL |
|---:|---:|---:|---:|
| 1,564 | 0.196 | 0.533 | 0.1081 |
| 3,128 | 0.650 | 0.523 | 0.0226 |
| 6,256 | 0.795 | 0.683 | 0.0107 |
| 9,384 | 0.858 | 0.723 | 0.0066 |
| 12,512 | 0.885 | 0.783 | 0.0056 |

Deduction, whose constant-answer baseline is exactly 0.500, is a second
responsive endpoint that had also read as dead. Accuracy is within four points
of its final value by step 7,820, so the 40,695-step budget is generous by
roughly a factor of three — a fact that matters for anyone pricing a
confirmatory matrix off it.

---

## 5. Finding 3 — The axes a matched control must satisfy are incompatible

### The design

The primary contrast masks fact-value targets against an equal mass of matched
non-value targets, so that the comparison isolates *which* content was removed
rather than *how much*. The control matches the treatment on four axes: count,
span length, within-document relative position, and mean per-token difficulty
measured as NLL under a frozen reference model.

The fourth axis is the one that matters. Fact values carry several times the
per-token surprisal of templated biography text, so a control matched only on
count removes far less loss mass and gradient magnitude than the treatment.

### The measurement

Measured over all 21.3B tokens of the operating-point corpus:

| | mean masked-span NLL |
|---|---:|
| treatment (fact values) | 1.9598 nats |
| control (matched non-values) | 1.2712 nats |
| relative gap | **35.1%**, against a preregistered 20% tolerance |

The control removes about 54% less learnable signal than the treatment, by
construction.

### The matching axes are mutually incompatible

Sweeping the position tolerance — the constraint forcing control spans to sit at
similar relative positions — narrows the gap but never closes it:

| position tolerance | NLL gap | cue-window overlap |
|---:|---:|---:|
| 0.05 | 42.2% | 32.5% |
| 0.15 (shipped) | 35.0% | 24.5% |
| 0.40 | 26.8% | 19.4% |
| 0.60 | 24.2% | 18.7% |
| 1.00 (unconstrained) | 23.5% | 18.5% |

Our first reading of this was that the corpus simply contains no non-value
tokens hard enough, making a matched control structurally impossible. **That is
wrong,** and the way it is wrong is the more useful result. Pricing each
constraint separately, over 800 documents and 22,416 value tokens:

| control construction | mean NLL | gap | within 20%? |
|---|---:|---:|---|
| shipped: contiguous, length- and position-matched | 1.2712 | 35.1% | no |
| contiguous, length-matched, position free | 1.4987 | 23.5% | no |
| **best possible** contiguous and length-matched | 1.5667 | **20.0%** | borderline |
| scattered tokens, count- and mean-matched | 1.7677 | **9.8%** | yes |
| scattered tokens, the k hardest | 1.8644 | 4.9% | yes |

Tokens hard enough to match the treatment **do exist** — they are simply not
arranged in contiguous runs. The binding constraint is span-length matching, not
the corpus and not position. Even an oracle contiguous control that picks the
hardest legal span every time lands exactly on the 20% boundary; drop the
contiguity requirement and the gap falls to 9.8%.

### The trilemma this exposes

The design asks the control to match the treatment on count, span length,
relative position, and difficulty. **Those four axes cannot be satisfied
simultaneously in this corpus, and the incompatibility is not incidental.** Fact
values are contiguous high-surprisal runs; the comparable surprisal in
non-value text is distributed as isolated tokens. So a control can preserve the
*shape* of what the treatment removes, or the *amount of learnable signal* it
removes, but not both:

- **Match length and position** — what we shipped — and the control removes 35%
  less loss mass than the treatment. Any measured effect is confounded by
  gradient magnitude.
- **Match count and difficulty** — scattered placement — and the control removes
  the right amount of signal from the wrong shape. Masking five scattered
  tokens is not the same intervention as masking one five-token value, and the
  difference plausibly matters for what the model learns.

We report the trilemma rather than resolving it. A study using a matched-mass
control should state which horn it has taken; we know of none that does, and
until now we had not noticed we had taken one.

**A circularity to disclose.** The frozen difficulty table was built from an
early checkpoint of the same run later analysed. The two arms are scored under
the same table, so the *gap* is not biased by this, but the absolute NLL values
are not independent of the model being studied.

### A second, related bias

24.5% of control-masked tokens land within three tokens of a fact value — the
cue phrases that predict it ("was born in"). The control therefore removes
fact-relevant supervision across roughly a quarter of its mass, biasing it
toward the treatment and attenuating the contrast toward zero. This was
predicted in the design and never measured until now.

### The diagnostic

Compute the mean per-token NLL of treatment-masked and control-masked spans
under any fixed reference model and compare. One CPU pass over the corpus, no
GPU. We know of no selective-loss or span-masking paper that reports it.

---

## 6. Finding 4 — A finite data lane substitutes silently for another

Our corpus interleaves lanes by target share. The fact lane is finite: it can
emit `entities x exposures` documents and no more. When its share asked for
more tokens than it could produce, the builder gave the remainder to the
natural-text bed so the total token count stayed exact — necessary, because
every arm and load in the design must share an optimizer-step count.

That reallocation is correct. It was also silent, and it produced this:

| lane | requested | realised |
|---|---:|---:|
| fact | 50% | **3.75%** |
| iGSM | 30% | 30.00% |
| deduction | 10% | 10.00% |
| bed | 10% | **56.25%** |

The pilot that produced our first STOP decision trained on a corpus that was
3.75% facts, in an experiment about the effect of fact load. A later difficulty
ladder was worse: 3.75% against a requested 70%, with 79.65% of the corpus
being synthetic filler because a bed path was left unset. Its training loss
settled at 2.31 against the earlier run's 1.76, which is what a corpus
dominated by high-entropy filler looks like — the only visible symptom, and one
nobody had reason to compare.

Every integrity check passed. Token count was exact, both mask sidecars were
byte-aligned, mask mass matched, and no sidecar overlapped a value span. The
manifest recorded the realised lane token counts, in a field nobody read.

### The diagnostic

Record the *requested* shares alongside the realised ones and fail closed on a
material difference. The requested shares must be stored separately, because
the reallocation mutates the budget it is derived from. Better still, price the
finite lane before writing anything: our builder now refuses a configuration
whose fact lane cannot fill its share, and reports the exposure count that
would fix it.

---

## 7. Finding 5 — Fitting fact text is not storing facts

Two measurements on the same checkpoint point in opposite directions.

**In the training phrasing**, cross-entropy at fact-value token positions is
**1.9732 nats**. The same metric on an earlier, under-exposed corpus read 25.03
nats, so this corpus is the first in the project where the model fits the fact
text at all.

**Under a held-out paraphrase**, the model scores *worse than the pool prior*.
Recoverable bits — baseline entropy minus the model's NLL on the true value,
queried with a template that never appears in training — comes out at
**−126 bits per entity** against a 52.96-bit ceiling. The model is not
uninformative about the values; it is confidently wrong about them.

### The control that settles it

A negative probe result has an innocent reading: the query format is unfamiliar
and the model cannot be induced to emit what it knows. Our own earlier work on
real entities found exactly that, where constrained decoding recovered a
substantial part of an apparent failure.

The control is to run the identical probe on entities the model **never saw**.
If knowledge is present but unaddressable, trained entities should still
separate from untrained ones; if nothing is retrievable, they will not.
300 entities per cohort, disjoint generators, same templates:

| cohort | recoverable bits per entity |
|---|---:|
| trained, seen 100 times each | **−112.90** |
| never seen | **−112.29** |
| **difference** | **−0.61 ± 0.68 (SE)** |

**The model's behaviour on facts it saw a hundred times is statistically
indistinguishable from its behaviour on facts that do not exist.** Not
attenuated — absent. The large negative value is a property of the metric
against a confidently-wrong model and is not itself interpretable; the
difference is, and it is zero.

### What this licenses

The model fits fact text (1.9732 nats in training phrasing) and acquires no
transferable knowledge of the facts whatsoever. The fact lane consumed **70% of
the training tokens** and produced nothing retrievable.

The operating point matters for how far this generalises. This run carried
1,531,800 entities at 100 exposures: 81 Mbit of fact content against 40.56M
parameters, or **2.0 bits per parameter of demand against roughly 1.0
achievable at that exposure count — twice capacity**. Our own capacity analysis
predicts that in this regime the cheapest way to reduce loss is to model the
marginal distribution of values rather than the conditional one, and to decline
to learn the mapping at all. **That is the abandonment regime, and this is a
direct observation of it.** The model did not compress the facts under pressure;
it opted out of them.

We therefore cannot claim this holds at or below capacity. What we can claim is
sharper than a null: at twice capacity, a model shown 1.5M facts a hundred times
each learns none of them, while learning the text that contains them well enough
to score 1.97 nats on their tokens and to solve reasoning at 93.8%.

Note that the per-parameter form of this metric extrapolates a probed subsample
by the full entity count, which is defensible for a positive quantity and
misleading for a negative one. The per-entity figures above are the
interpretable ones.

---

## 8. What this says about the hypothesis

We set out to measure whether masking fact targets buys reasoning. We have not
measured that. But the findings above are not silent about the hypothesis, and
it would understate them to present this as a methods paper alone.

Memory Split rests on three premises: that arbitrary facts occupy parameters,
that masking frees them, and that the freed parameters go to reasoning.

**Premise 1 fails at the operating point we reached, and fails hard.** At twice
achievable capacity, with 70% of the training budget spent on 1.5M facts shown
a hundred times each, the model acquired no retrievable knowledge of any of
them — indistinguishable, on a held-out query, from facts it never saw. There
is no fact store here whose capacity could be freed. The premise does not fail
because capacity was ample; it fails because the model declined to enter the
storage regime at all, which is what our capacity analysis predicts above
saturation and what nobody had previously observed directly.

**Premise 3 is now testable, which it was not.** The reasoning endpoint we had
concluded was unusable at accessible scale solves three-operation dependency
chains perfectly and reaches 93.8% in-band. Four generations of this project
argued the opposite from an instrument fault. Anyone who cited a floored
reasoning endpoint at this scale as evidence about small-model capacity should
recheck how their generations were batched.

**A methodological consequence for the wider claim.** The rationale for
filtering fact-heavy pages from pretraining data assumes those pages consume
capacity that reasoning would otherwise use. Our measurement is one model, one
corpus, one operating point, and it does not refute that. It does show that the
assumption is measurable, that measuring it is cheap, and that in the one case
we measured it was false — the fact-heavy data consumed 70% of the token budget
and left nothing behind to displace anything.

**What would settle it.** A model trained at F/C ≈ 1.0 rather than 2.0, where
the theory predicts storage rather than abandonment. That corpus is built and
verified; the run is one GPU-day. If facts are acquired there, crowding becomes
testable for the first time and premise 1 survives at capacity. If they are not,
premise 1 fails across the accessible range and the hypothesis has no purchase
at this scale regardless of any treatment effect.

## 9. What we do not claim

- We do not report a treatment effect. The primary contrast has never been
  validly measured, and with the control failing its difficulty match it cannot
  currently be measured cleanly.
- We do not claim the crowding hypothesis is false. Our nulls came from
  experiments that failed preconditions, and the largest of them came from a
  broken instrument.
- We make no claim about "small language models" generally, nor about phi-3,
  which is roughly 100x larger than the model here.
- The difficulty curve in §4 is one seed, one architecture, one corpus.

## 10. A checklist

Every item costs minutes and none needs a GPU.

1. **Assert that batching does not change a generation.** Decode one prompt
   alone and in a mixed-length batch; require equality.
2. **Save raw generations, not just parsed answers.** Our scorer discarded the
   generated text and kept the parse, so four generations of nonsense output
   were invisible in the logs.
3. **Report the difficulty of what your control masks**, not only how much of
   it, and price each matching axis separately. Mean per-token NLL of treatment
   and control spans under any fixed reference model, plus the best value
   achievable under each constraint you impose. Ours were mutually
   unsatisfiable and we did not know.
4. **Record requested data proportions next to realised ones**, and fail on a
   material gap. Do not derive one from a budget the pipeline mutates.
5. **Check an endpoint for ceiling as well as floor.** A saturated endpoint
   cannot show a treatment effect any more than a floored one can.
6. **Query facts in a phrasing absent from training** before making any claim
   about stored knowledge.

## 11. Reproduction

All artifacts, code, and the audit of which prior numbers were withdrawn are in
the repository. `docs/RETAINED-RESULTS.md` is the catalogue of results that have
committed artifacts; anything absent from it has been withdrawn. Findings 1, 2
and 5 are catalogued there as §8 and §9, finding 3 as §7. The padding defect and
its correction are pinned by
`tests/test_generate.py::test_batched_decoding_equals_unbatched_at_mixed_lengths`.

### Independent verification

Two reviewers checked every numeric claim in this draft against the artifacts,
with cluster access to re-run them. Their findings changed the paper:

- The "structurally infeasible" reading of finding 3 was **wrong**; pricing each
  matching constraint separately showed the axes are mutually incompatible
  instead, which is §5 as it now stands.
- The recoverable-bits figure was computed with the wrong entity count. −195
  became **−126**.
- The 48-item per-op table in finding 2 is **not publishable at those cell
  counts** and is marked provisional pending the 1,500-item final-checkpoint
  evaluation.
- One reviewer independently reproduced the padding defect at a different seed
  and decode length: 0/64 old, 63/64 fixed, 64/64 string-identical between
  batched-after-fix and unbatched.
- Both reviewers found that none of these results was catalogued in
  `RETAINED-RESULTS.md`, which the project's own protocol requires before
  citation. They now are.

One disagreement was resolved in the paper's favour. Both reviewers reported
that the committed 1,500-item evaluation contradicted the difficulty curve.
`evals/igsm.jsonl` is overwritten by whichever checkpoint was scored most
recently and records no step; the file they read was step 14,076, not the final
checkpoint. The per-cell sample-size criticism stands regardless, and is why
that table is now marked provisional.
