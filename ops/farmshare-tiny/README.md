# Tiny crowding cohort on FarmShare

Runbook for `memorysplit-exploratory-v3-tiny-crowding-n1`: two models, each as
a matched dense/split90 pair, one seed, on the frozen 8B-token corpus.

The question is whether capacity crowding appears at tiny scale, and whether
loss-masking facts helps in that regime. The d160m point is already measured at
**+0.0011** on the reasoning extension (split90 marginally worse); these two
sizes extend that to three points.

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

Tied embeddings are a requirement, not a preference: at vocab 50,304 an untied
pair costs 12.9M parameters before a single transformer block, so an 8M model
is otherwise impossible. Recurrence is TRM's one transferable idea, effective
depth without parameters. Both apply identically to both arms, so the paired
design is untouched.

**At d8m, 81% of the parameters are the embedding table.** Only 1.49M are
transformer. If crowding shows up at this size it may be embedding-capacity
crowding rather than the schema-versus-fact competition the hypothesis is
about. Say so in any write-up.

## Cost, measured

`d8m` ran at **~460,000 tok/s** on one L40S, compiled, micro_batch 32
(job 1670856, 2026-07-31). That is 8,169,455,616 / 460k = **~4.9 h per arm**.
`d40m` is 484 MFLOP/token against d8m's 98.5, so expect appreciably longer;
the learning-rate probe gives the real figure before the cohort is committed.

All four arms run concurrently, so the pair wall clock is set by d40m.

## Queue reality

The `gpu` partition is heavily contended — a 4-GPU ask has been estimated at
14 h of queueing, behind several 2-day jobs. Two things follow.

The smoke deliberately uses a **different job shape**: one GPU, both arms
sequentially, on the `normal` partition, which carries the same L40S nodes and
whose QoS allows one GPU. It scheduled in under a minute where the 4-GPU ask
had not started after 15.

If the 4-GPU cohort will not schedule, the fallback is **one model at a time
with `--gres=gpu:2`**. That preserves the primary within-pair contrast, since
each pair still shares a node; only the across-model comparison loses
hardware matching, and that comparison is already confounded by recurrence and
tying.

| QoS | GPU cap | Jobs | Partition MaxTime |
|---|---|---|---|
| `gpu` | `gres/gpu=4` | 4 | 2-00:00:00 |
| `normal` | `gres/gpu=1` | 128 | 2-00:00:00 |

## Steps

```bash
export SUNET=syz MS_ROOT=/scratch/users/syz/memorysplit-160m-v3
SOCK="$HOME/.ssh/cm-%r@%h:%p"

# 1. Warm the session (Duo; 8 h persist). Both hosts: sockets are per-host.
bash cluster/connect.sh $SUNET
bash cluster/connect.sh $SUNET dtn.farmshare.stanford.edu

# 2. Ship the repo
rsync -az --exclude '.git' --exclude '_preserved' --exclude 'artifacts' \
  --exclude 'paper' --exclude 'outputs' --exclude 'data' \
  -e "ssh -o ControlPath=\"$SOCK\"" ./ $SUNET@dtn.farmshare.stanford.edu:$MS_ROOT/code/

# 3. Generate configs and stage the sbatch files
ssh -o ControlPath="$SOCK" $SUNET@rice-04.farmshare.stanford.edu "
  MS_CORPUS=$MS_ROOT/corpus MS_CODE=$MS_ROOT/code MS_RUNS=$MS_ROOT/runs/tiny-v3 \
    $MS_ROOT/venv/bin/python $MS_ROOT/code/ops/cohort-tiny/gen_configs_tiny.py
  cp $MS_ROOT/code/ops/farmshare-tiny/run_cohort.sbatch $MS_ROOT/run_cohort_tiny.sbatch
  cp $MS_ROOT/code/ops/farmshare-tiny/smoke.sbatch $MS_ROOT/smoke_tiny.sbatch"

# 4. Smoke. The gate is loss_masked_values in BOTH arms, not just that it runs.
ssh -o ControlPath="$SOCK" $SUNET@rice-04.farmshare.stanford.edu \
  "export MS_ROOT=$MS_ROOT && cd \$MS_ROOT && sbatch smoke_tiny.sbatch"

# 5. Cohort
ssh -o ControlPath="$SOCK" $SUNET@rice-04.farmshare.stanford.edu \
  "export MS_ROOT=$MS_ROOT && cd \$MS_ROOT && sbatch run_cohort_tiny.sbatch"
```

The corpus is **not** re-downloaded. `$MS_ROOT/corpus` was staged and verified
against the three composite stream hashes for the 160M run; the tiny cohort
reads the identical bytes.

## Gate 0, now working

`loss_masked_values` had produced nothing on this corpus for every prior
cohort, including the 1B runs. `masked_value_batch` sampled the first 262,400
tokens and permanently disabled itself when it found none masked; the first
masked target is at token 356,253,723. It now scans for a masked window.

Both arms also carry the split90 sidecar as `probe_mask`, so both score the
**same** offloaded positions. That is the addition that makes the metric answer
the crowding question: the split arm should sit near ln(50304) = 10.83, and the
dense arm's score is how much of the offloaded content dense actually absorbed
— the memorisation burden the hypothesis assumes exists.

Smoke confirmation at step 30, both arms at 10.86, as expected before either
has learned anything:

```
d8m_dense   step 30  loss 9.9646  loss_masked_values 10.8640
d8m_split90 step 30  loss 9.9653  loss_masked_values 10.8641
```

## Monitoring

```bash
squeue -u $SUNET
for f in $MS_ROOT/runs/tiny-v3/*/log.jsonl; do echo "$f: $(tail -1 $f)"; done
```

## Scope

`successor_exploratory_unpreregistered`, n=1. One seed cannot support the
paired statistic; this sizes a direction and a magnitude, nothing more.
