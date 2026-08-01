# Tiny crowding cohort — results

Cohort `memorysplit-exploratory-v3-tiny-crowding-n1`: two models, each a matched
dense/split90 pair, seed 0, on the frozen 8,169,455,616-token reasoning-v3
corpus. FarmShare jobs 1670908 (`d8m`, 4:54:50) and 1670907 (`d40m`, 12:19:31),
both `COMPLETED` 2026-08-01, all four arms at step 15,582.

## Answer

**Crowding does not happen in this range, and Memory Split does not help.**

Dense loss on the reasoning extension is nearly flat from 8M to 160M
parameters — a 20-fold range — so capacity is not the binding constraint. And
the split arm is marginally *worse* at every size, with the smallest model
showing the largest penalty, which is the opposite of the predicted direction.

Gate 0 now says why, and this is the substantive finding: at 40M the split arm
recovers **94.2%** of dense's performance on the offloaded positions *without
ever having trained on them*. Supervision on those tokens is worth only 0.33
nats out of 5.73. The offloaded content is overwhelmingly recoverable from
context, so there is almost no memorisation burden for masking to relieve.

## The question

At 162M parameters the split-minus-dense contrast on the reasoning extension
was **+0.0011** — split90 marginally *worse*, a null. One candidate explanation
is that there was no memorisation burden to relieve: a separate measurement put
fact exposure in this corpus at only 1.04 to 1.55 times per fact, and a model
that never memorised the facts frees nothing by offloading them.

If that is right, shrinking the model should not help either. If instead
capacity genuinely binds at small scale, the split arm should start winning as
size falls. Two sizes below 162M test that.

## The two models

| | d8m | d40m |
|---|---:|---:|
| n_layer x n_head x d_model | 7 x 4 x 128 | 12 x 6 x 384 |
| head_dim / d_ff | 32 / 384 | 64 / 1024 |
| n_recurrence -> effective depth | 3 -> 21 | 2 -> 24 |
| tie_embeddings | yes | yes |
| **parameters** | **7,931,776** | **40,560,000** |
| embedding share | 81.2% | 47.6% |
| tokens per parameter | 1,030 | 201 |

Both counts are asserted exactly against `GPT(PRESETS[name]).num_params()` in
`tests/test_cohort_tiny.py`, and the generator's analytic formula is pinned to
the same values, so a config cannot ship a stale count.

Held fixed against the d160m run: the corpus, the 524,288-token optimizer
batch, 15,582 steps, snapshots at 1558/3896/7791/11687/15582, ctx 1024, warmup
300, weight decay 0.1, single pass with no repetition.

## Setup, measured

**Learning rates** were probed rather than assumed, because an untuned rate
would make a model look capacity-crowded when it is merely badly optimised.
Selection used the dense arm only and the winner was applied to both arms.
Round 1 put both winners on the top edge of the grid, so it was extended until
each optimum was bracketed on both sides. Full grid in `LR-PROBE.md`.

| | chosen | bracketed by |
|---|---:|---|
| d8m | **1.2e-2** | 6e-3 (5.2302) and 2.4e-2 (5.9214), best 5.1476 |
| d40m | **8.0e-3** | 4e-3 (4.7389) and 1.6e-2 (5.4013), best 4.6371 |

**Throughput** on one L40S, compiled, micro_batch 32:

| | tok/s | per arm |
|---|---:|---:|
| d8m | ~466,000 | ~4.9 h |
| d40m | ~183,000 | ~12.4 h |

**Gate 0 now works.** `loss_masked_values` had produced nothing on this corpus
for every previous cohort, the 1B runs included: `masked_value_batch` sampled
the first 262,400 tokens and permanently disabled itself on finding none
masked, while the first masked target sits at token 356,253,723. It now scans
for a masked window. Both arms additionally carry the split90 sidecar as
`probe_mask`, so both score the *same* offloaded positions — which is what lets
the dense number mean something. Confirmed in the smoke run at step 30, both
arms at 10.86 against a ln(50304) = 10.83 ceiling, as expected before either
has learned anything.

## How to read the result, decided in advance

**Primary metric, unconfounded: split90 minus dense over steps 14582-15580.**

The reasoning extension begins at step 13582 (7,120,879,616 / 524,288). Only
there do both arms read the same target-weight sidecar, so only there is a loss
comparison meaningful. Across the base segment split90 masks 20.419% of targets
and cross-entropy normalises by the count of non-ignored targets, so the two
arms divide different numerators by different denominators — any apparent
split90 advantage there is the masking, not learning. Step 15582 is excluded
because its final micro-batch wraps to the corpus head and spikes the loss.

Within a size, both arms share architecture, recurrence, tying, learning rate,
initialisation and data order. That contrast is clean.

**Secondary, confounded: the trend across sizes.** d160m has `n_recurrence=1`
and untied embeddings; d8m and d40m have recursion and tying. Three things vary
together, so report the trend qualitatively and never as a slope.

**Mechanism: gate 0.** If the dense arm sits far below 10.83 on the offloaded
positions, it absorbed that content and there was a real memorisation burden to
relieve. If dense sits near 10.83 too, there was no burden, and a null is the
expected result at every size — which would say the corpus, not the model size,
is what makes the hypothesis untestable here.

**The 8M caveat.** 81% of d8m's parameters are the embedding table; only 1.49M
are transformer. Crowding observed at that size may be embedding-capacity
crowding rather than the schema-versus-fact competition the hypothesis is
about. Separating those needs a variant at smaller `d_model` with more layers,
which this cohort does not include.

**n=1.** One seed cannot support the paired statistic. This sizes a direction
and a magnitude.

## Results

### Primary: split90 minus dense on the reasoning extension

Steps 14582-15580, the settled window where both arms read the same sidecar.
Positive means split90 is worse.

| model | params | dense | split90 | split − dense |
|---|---:|---:|---:|---:|
| d8m | 7,931,776 | 0.8072 | 0.8100 | **+0.0028** |
| d40m | 40,560,000 | 0.7855 | 0.7864 | **+0.0009** |
| d160m | 162,220,800 | 0.7905 | 0.7916 | **+0.0011** |

All three positive, all tiny. There is no drift toward split90 winning as the
model shrinks; the 8M point is the worst for split, not the best. Whatever the
split arm loses by giving up 20.4% of its supervision, it does not get back in
reasoning performance at any size tested.

The penalty also shrinks as the extension is consumed, which suggests a
transient adaptation cost rather than a persistent deficit:

| window | d8m | d40m | d160m |
|---|---:|---:|---:|
| early extension (13600-14000) | +0.0079 | +0.0037 | +0.0047 |
| mid (14000-14582) | +0.0093 | +0.0014 | +0.0017 |
| settled (14582-15580) | +0.0028 | +0.0009 | +0.0011 |

### Capacity is not binding

Dense-arm loss, by region:

| model | base tail (13000-13580) | extension (14582-15580) |
|---|---:|---:|
| d8m | 0.3354 | 0.8072 |
| d40m | 0.3037 | 0.7855 |
| d160m | 0.2960 | 0.7905 |

Across a 20x parameter range the extension loss moves by 0.02 nats, about 2.7%,
and **d40m is slightly better than d160m**. An 8M model with 1.49M transformer
parameters gets within 3% of a 162M model on this material. That is not what a
capacity-limited regime looks like.

The d40m-beats-d160m inversion is not evidence that smaller is better. The
d160m run used the frozen 1.5e-3 tier and was never probed, while d40m got a
probed 8.0e-3; d40m also has effective depth 24 against d160m's 12. The likeliest
reading is that **d160m was under-tuned**, which is worth knowing on its own.

### Gate 0: what supervision on the offloaded positions actually bought

First numbers this metric has ever produced on this corpus. Uniform over the
padded vocabulary is ln(50304) = 10.8258; lower means the arm predicts the
offloaded content better.

| model | arm | final CE | nats below uniform |
|---|---|---:|---:|
| d8m | dense | 10.0652 | 0.7606 |
| d8m | split90 | 10.8899 | −0.0641 |
| d40m | dense | 5.0943 | 5.7315 |
| d40m | split90 | 5.4264 | 5.3994 |

Both arms score the *same* positions, so the difference is exactly what
training on those tokens was worth.

**At 40M, almost nothing.** Dense captured 5.7315 nats; split90 captured 5.3994
of them having never received a gradient there. Supervision was worth 0.3321
nats, **5.8% of the total**. The offloaded spans are 94.2% predictable from
their surroundings. That is the mechanism behind the null: masking them removes
a burden that was never really a burden.

It is consistent with the span structure. These are 17-35 token payloads, not
bare values, so most of their tokens are template and only a small part is the
fact itself. It is also consistent with the earlier finding that each fact is
exposed only 1.04-1.55 times: material seen once and largely inferable is
material a model has little reason to memorise.

**At 8M the model is simply too weak to engage.** split90 sits at −0.06 nats,
which is uniform within noise, and dense captured only 0.7606 — an eighth of
what 40M managed. Neither arm learned this content. That is not crowding; it is
a model below the threshold where the material becomes learnable at all.

## What this does and does not settle

Settled, for this corpus: capacity does not bind between 8M and 160M on the
reasoning extension, Memory Split gives no benefit at any of those sizes, and
the reason is measurable rather than speculative — the offloaded content is
overwhelmingly context-recoverable.

Not settled. This is **n=1** at each size, and the effects are smaller than the
±0.0086-nat seed spread measured in the Setup A sweep, so the sign of any
individual contrast is not secure. It is **training loss, not a reasoning
evaluation**; the preregistered deliverable is downstream accuracy with the
organizer supplying facts, which still has not been run at any scale. The
across-size comparison confounds parameters with recursion, embedding tying and
learning-rate tuning. And at d8m, **81% of parameters are the embedding table**,
so that point speaks to a model with 1.49M transformer parameters rather than a
scaled-down transformer.

The honest summary is that the hypothesis has not been given a fair test by
this corpus, at any size. A corpus where facts are repeated and not inferable
from context — which is what Setup A deliberately built, at `n_exposures=6` —
is the precondition for the question to be answerable. Building that, rather
than running more sizes against reasoning-v3, is the thing that would move this
forward.

## Reproducing

```bash
python3 ops/cohort-tiny/analyze_crowding.py \
  --dirs outputs/farmshare-tiny outputs/farmshare-160m-v3-s0
```

## When the runs land

```bash
export SUNET=syz MS_ROOT=/scratch/users/syz/memorysplit-160m-v3
SOCK="$HOME/.ssh/cm-%r@%h:%p"

# status
ssh -o ControlPath="$SOCK" $SUNET@rice-04.farmshare.stanford.edu \
  "squeue -u $SUNET; for f in \$MS_ROOT/runs/tiny-v3/d*_reasoning_v3_s0/log.jsonl; do
     echo \"\$f: \$(tail -1 \$f)\"; done"

# pull
rsync -az --include '*/' --include 'log.jsonl' --include 'config.yaml' --exclude '*' \
  -e "ssh -o ControlPath=\"$SOCK\"" \
  $SUNET@rice-04.farmshare.stanford.edu:$MS_ROOT/runs/tiny-v3/ outputs/farmshare-tiny/
rm -rf outputs/farmshare-tiny/_smoke_*

# analyse, against the existing d160m point
python3 ops/cohort-tiny/analyze_crowding.py \
  --dirs outputs/farmshare-tiny outputs/farmshare-160m-v3-s0
```

The analyzer prints the split-minus-dense table, dense extension loss by size,
and the gate-0 numbers for both arms at every size. It has been verified
against the d160m result, which it reproduces at +0.0011.

Results directory is tracked (`.gitignore` carries `!outputs/farmshare-tiny/`),
so pulled logs are committed rather than left to be swept.

## Scheduling note

The `gpu` partition is heavily contended. A contiguous 4-GPU block was
estimated at 21 hours of queueing, so the cohort was split into two 2-GPU jobs
via `MS_MODELS`, one per model. Each pair still shares a node, so the primary
contrast is unaffected; only across-model hardware matching is lost, and that
comparison is already confounded. Estimated starts at submission were
2026-08-01 08:35 for d8m and 11:55 for d40m, cluster-local.
