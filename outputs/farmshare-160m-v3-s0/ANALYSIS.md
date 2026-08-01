# d160m reasoning-v3, seed 0 — run record and reading

Job 1669507 on FarmShare `oat-01`, 2 × L40S 48 GB, submitted 2026-07-30T20:43:33Z,
`COMPLETED` after **16:50:25**. Both arms reached step 15,582 and the supervisor
wrote `.complete-s0`. Pulled 2026-07-31.

Cohort `memorysplit-exploratory-v3-160m-aws-n8`, seed 0 of 8. **n=1.**

| | dense | split90 |
|---|---|---|
| Final step | 15,582 | 15,582 |
| Throughput | 123,795 tok/s | 126,219 tok/s |
| Snapshots | 10 | 10 |

Both arms ran concurrently in one two-GPU allocation, so the pair sat on
identical hardware, as the paired design requires.

## The only region where the arms are comparable

The corpus is base (7,120,879,616 tokens) then reasoning extension
(1,048,576,000), concatenated. At 524,288 tokens/step the extension begins at
**step 13,582** — the last 2,000 steps, 12.8% of training.

Across the base segment the arms are **not** comparable on loss. Split90 zeroes
20.419% of base targets, and `F.cross_entropy` normalises by the count of
non-ignored targets, so the two arms divide by different denominators over
different target sets. Split90's apparent −0.0249 advantage there is the
masking, not learning. The masked targets are the routed factual payloads,
which are the hardest tokens in the segment; dropping them lowers the mean
mechanically.

Over the extension both arms read the *same* sidecar
(`extension/sidecars/shared_target_weights.bin`), so supervision is identical
and the comparison is valid.

## Result: a null, marginally against the hypothesis

Per-step raw loss, split90 minus dense, excluding the final contaminated step:

| Region | n points | dense | split90 | split − dense |
|---|---:|---:|---:|---:|
| base (not comparable) | 679 | 1.4128 | 1.3879 | −0.0249 |
| extension | 100 | 0.8143 | 0.8164 | **+0.0020** |
| extension, settled tail (14,582–15,580) | 50 | 0.7905 | 0.7916 | **+0.0011 ± 0.0007** |

Split90 is worse on 93 of 100 logged extension points. The sign is consistent;
the magnitude is 0.0011 nats, about 0.14% relative.

So on this seed, offloading 20.4% of factual supervision bought **no reasoning
benefit** — it cost a hair. The hypothesis predicts the opposite: that freeing
capacity from fact storage should improve reasoning.

**This does not settle anything, for three reasons.**

The 93/100 sign consistency is not 100 independent samples. Consecutive logged
points 20 steps apart in the same run are heavily autocorrelated, so this
supports "the gap is stable within this run", not a p-value. The design's
terminal statistic is a paired contrast **across the 8 seeds**, and this is
seed 0 of 8.

Next-token loss on the extension is not the deliverable. The question is
downstream reasoning accuracy on held-out Reasoning-Gym items with the
organizer supplying facts to the split arm. That evaluation has not been run.

And a 0.0011-nat gap is within the range that seed noise alone could produce.
For scale, the Setup A sweep's seed spread in final loss was ≤0.0086 nats —
roughly 8× this effect.

## Two defects found, neither fatal, both inherited

### Gate 0 never fired

`loss_masked_values` has **zero** points in both arms. That metric is the
mechanism check — the split arm's cross-entropy on exactly the positions it was
never trained on, which should sit near the ln(50304) = 10.83 uniform ceiling
and demonstrate the offloaded facts are absent from the weights.

Root cause, confirmed against the corpus: `PackedShards.masked_value_batch()`
samples the first `batch_size × (ctx+1) × 8` = **262,400 tokens** of the stream
and returns `None` if it finds no masked position, after which `Trainer` caches
`"none"` and never retries. In this corpus the first masked target is at token
**356,253,723** — step 679. The head of the stream is an unmasked lane, so the
probe looks in the one place the answer cannot be.

This is not specific to this run. The 1B cohort's surviving logs
(`paper/memory-split-hypothesis/data/{dense,split90}.log.jsonl`) also have zero
`loss_masked_values` points, on the same corpus. Any gate-0 evidence in the
project's write-ups comes from **Setup A**, where the probe did fire at
9.64–12.30 nats — not from reasoning-v3.

The fix is to sample the probe window from a masked region rather than offset
0. Until then, reasoning-v3 has no in-training evidence that the split arm is
actually split. The static preflight still proves the *sidecars* differ
correctly (20.419% offloaded, `reverse = 0`); what is missing is the evidence
about the trained weights.

### The final micro-batch wrapped

Both arms end at `epoch: 1` despite this being a single-pass design. It is an
off-by-one, not a second epoch. The loader advances the cursor by
`batch_size × ctx` = 32,768 but reads a window of `batch_size × (ctx+1)` =
32,800, and wraps when `cursor + span >= n_tokens`. Of 249,312 micro-batches,
exactly the last one wraps:

- 32,768 tokens (**0.00040%** of the corpus) re-read from offset 0
- the final 32,768 tokens of the extension were never read

Both arms did this identically, so the pair stays matched and the science is
unaffected. It does contaminate the last logged step: at 15,582 the loss jumps
from 0.79 to 1.15 in both arms because that wrapped batch is drawn from the
corpus head, a different distribution from Reasoning-Gym. **Use step 15,580 or
earlier for any terminal number.**

Recommendation: do *not* patch the loader before seeds 1–7 run. Changing the
wrap condition changes the data order, which would make seed 0 non-comparable
with the rest of its own cohort. Fix it between cohorts, not inside one.

## Fact exposure in the base segment

Measured 2026-07-31 by `ops/cohort-160m/fact_exposure.py`, job 1670818, a full
pass over the base sidecar and token stream (5m29s). Raw output in
`exposure-1670818.out`, machine-readable in `fact-exposure.json`.

A run of zeros in the split90 sidecar is one offloaded fact-value occurrence,
so counting occurrences needs no assumptions:

| | |
|---|---:|
| fact-value occurrences | **61,453,327** |
| masked tokens | 1,453,989,440 (20.419% of base) |
| mean span | 23.66 tokens |

Span length is bimodal — about 42% at 17–18 tokens and 26% at 34–35 — so there
are two payload formats, both far longer than a bare value like a date or a
name. Masking is absent from the first ~5% of the base segment (the first
masked target is at token 356,253,723) and then roughly uniform to the end.

Deciding which occurrences are *the same fact* does need an assumption, because
the v3 base producer is not in this repository. Keying a fact on its value plus
N tokens of preceding context brackets the answer from both sides:

| key | est. distinct facts | mean | median | p90 | p99 | max | seen exactly once |
|---|---:|---:|---:|---:|---:|---:|---:|
| value only | 33.6M | 1.55 | 1 | 2 | 4 | 15,407 | 89.2% |
| ctx 8 + value | 37.2M | 1.40 | 1 | 1 | 4 | 12,494 | 97.1% |
| ctx 16 + value | 41.1M | 1.27 | 1 | 1 | 3 | 12,478 | 97.8% |
| ctx 32 + value | 50.0M | 1.04 | 1 | 1 | 1 | 2,662 | 99.4% |

Value-only merges genuinely different facts that share a value string, so it
undercounts facts and **overcounts** exposures: 1.55 is an upper bound. Context
32 splits one fact rendered in different surroundings, so it overcounts facts
and **undercounts** exposures: 1.04 is a lower bound.

**Each fact is exposed about once — between 1.04 and 1.55 times, median 1 under
every definition.** The distribution is heavily right-skewed: 89–99% of facts
occur exactly once and a thin tail runs to thousands, which at those extremes is
almost certainly recurring boilerplate rather than a fact.

This corroborates the recipe. `docs/1B-CORPUS-CONSTRUCTION.md` states that "the
Wikidata training graph is covered completely once, giving a stable fact
universe" — the measurement is what that constraint implies.

### Why this bears on the null

The hypothesis is that offloading facts frees capacity that would otherwise go
to memorisation. That presupposes the dense arm is *spending* capacity on
memorisation. At roughly one exposure per fact, buried in 7.1B tokens, a 162M
model is unlikely to be memorising much of this material in the first place. If
there is little memorisation burden to relieve, the intervention has little to
free, and the expected effect is close to zero — which is what the extension
measured (+0.0011 nats, split90 marginally worse). Split90 additionally gives
up 20.4% of its supervision signal, which is a plausible source of that small
deficit.

Setup A was built the other way round. `corpusgen/realfact.py` defaults to
`n_exposures=6` and `corpusgen/bios.py` renders each entity at several exposure
indices, so facts there are deliberately repeated — and that is the setup where
a dose-response and a clean gate-0 signal did appear.

So the reasoning-v3 corpus may be poorly matched to the question it is being
used to ask. Before spending eight seeds on it, it is worth deciding whether
the corpus should carry repeated facts, and at what multiplicity. This is a
measurement about corpus construction, not a result about the hypothesis.

## What is on the cluster

`/scratch/users/syz/memorysplit-160m-v3/runs/160m-v3/` — 10.4 GB, the only copy,
on scratch that is not backed up.

| Per arm | |
|---|---|
| `ckpt.pt` | 1.95 GB |
| `snapshots/step*.pt` | 10 × 649 MB, at steps 1558 … 15580 |

The snapshots are what the reasoning evaluation needs. The 1B cohort's
checkpoints were lost with its node; these are exposed the same way.

## Next

1. Run the reasoning evaluation on the seed-0 snapshots — that is the actual
   deliverable and none of it exists yet.
2. Decide on seeds 1–7. One seed cannot support the paired statistic, and a
   full cohort is 8 × ~17 h, which fits the `gpu` partition's 4-GPU cap in four
   sequential pairs.
3. Fix the gate-0 probe window before the next cohort.
