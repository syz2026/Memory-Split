# Tiny crowding cohort — design, setup measurements, and how to read it

Cohort `memorysplit-exploratory-v3-tiny-crowding-n1`: two models, each a matched
dense/split90 pair, seed 0, on the frozen 8,169,455,616-token reasoning-v3
corpus. Submitted 2026-07-31 as FarmShare jobs 1670907 (`d40m`) and 1670908
(`d8m`).

**Training results are not in yet.** Everything below the "Setup, measured"
section is either a measurement already taken or the analysis contract fixed in
advance. The results table is filled by running the command in "When the runs
land".

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
