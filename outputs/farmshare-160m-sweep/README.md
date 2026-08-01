# d160m dose-response sweep — pulled from FarmShare 2026-07-30

Training logs and configs for the **Setup A** 160M sweep, pulled from
`/scratch/users/syz/memorysplit/outputs/` on `rice-04`. This is a different
experiment from the reasoning-v3 memory-split cohort in `ops/cohort-160m/` —
same model size, different corpus, different treatment, different question.

Only logs and configs are here. The 194 GB of checkpoints and snapshots stayed
on FarmShare scratch; see "What is still on the cluster" below, because that
matters for whether these runs can still be finished.

## Analysis in this directory

| File | What it does |
|---|---|
| `analyze_sweep.py` | Per-run table, seed spread, PASS/FAIL checks on the claims below |
| `analyze_epoch_confound.py` | Locates the dense epoch flip and sizes the re-read |
| `ANALYSIS.txt` | Committed output of both, so the numbers can be diffed |

Interpretation, next steps, and an open provenance issue affecting the paper's
Table 1: `docs/2026-07-30-160m-sweep-record.md`.

## What this experiment is

The fact-load dose-response from `scripts/make_manifest.py --stage sweep`:
**3 fact loads × 2 arms × 2 seeds = 12 runs**, all complete.

The treatment is the original Setup A one, not the sidecar masking used at
reasoning-v3. Dense trains on biography text where fact values receive ordinary
LM loss. Split trains on a rendering of the same corpus where fact values are
loss-masked and supplied by an external organizer at evaluation. The two arms
therefore read **different token streams**, which is the key difference from
the reasoning-v3 design where both arms read one byte-identical stream.

| | |
|---|---|
| Model | `d160m`, 162,220,800 parameters |
| Budget | 3,200,000,000 tokens, 6,103 steps, `tokens_per_step` 524,288 |
| Loads | `n50k`, `n200k`, `n800k` entities |
| Seeds | 0, 1 |
| `lr` | 1.5e-3, warmup 300, cosine to 10% |
| `micro_batch_size` | 16 (accum 32) |
| Hardware | FarmShare, ~120–127k tok/s per run |
| Completed | seed 0 on 2026-07-21/22, **seed 1 on 2026-07-26**; all 12 at step 6103 |

The two seeds finished five days apart, which matters for reading the paper:
the seed-1 runs landed on 2026-07-26, the same day the write-up is dated, so
`main.tex`'s "one seed per cell" was accurate when written. Seed 1 is genuinely
new evidence relative to it.

Also pulled: two `d1b` **gate** runs (`n800k`, `n4m`, dense only, seed 0) from
the `calib1b` stage, complete at step 2861 of a 1.5B-token budget, at ~18.9k
tok/s. These are the short dense probes used to pick which load binds at 1B.

## Final losses

| Load | Arm | s0 | s1 | mean |
|---|---|---:|---:|---:|
| n50k | dense | 1.9885 | 1.9799 | 1.9842 |
| n50k | split | 1.9576 | 1.9550 | 1.9563 |
| n200k | dense | 2.1045 | 2.1076 | 2.1060 |
| n200k | split | 1.9724 | 1.9671 | 1.9697 |
| n800k | dense | 2.1203 | 2.1231 | 2.1217 |
| n800k | split | 1.9727 | 1.9693 | 1.9710 |

Split-arm `loss_masked_values` at the end: 11.21, 10.22 (n50k), 9.64, 10.69
(n200k), 9.97, 12.30 (n800k).

## Reading these numbers, and how not to

**The dense-minus-split loss gap is not the treatment effect.** It widens with
load — +0.028, +0.136, +0.151 — which looks like a clean dose response, but the
two arms compute loss over *different target sets*. Split masks the fact-value
tokens out of its loss entirely, so it is scored only on the easier residual
text, which is why split sits flat near 1.96 at every load while dense climbs.
Most of that gap is the masking, not learning. Do not report it as a result.

**`loss_masked_values` is a real result.** It is the split arm's cross-entropy
on exactly the positions it was never trained on. Uniform over the padded
50,304 vocabulary would be ln(50304) = 10.83, and the six runs land at 9.6–12.3
— essentially chance. That is gate 0: the offloaded facts did not end up in the
weights. It is the check that the split arm is really split.

**The arms did not see the same number of passes.** Every dense run ends at
`epoch: 1` and every split run at `epoch: 0`. The split rendering is longer in
tokens because it carries the lookup markers, so a fixed 3.2B-token budget
wraps the dense corpus and not the split one. Dense therefore re-read some of
its data.

Measured by `analyze_epoch_confound.py`, all six dense runs flip at step 6,080
of 6,103, putting the dense corpus at ≈3.182B tokens and the re-read at
**0.55% of the budget** — 1.006 passes against split's 1.000. Repetition
inflates memorisation, which is the burden the hypothesis says split is freed
from, so the bias points *toward* the hypothesis and the reported null survives
it. Declare it; it is far too small to move a 5.5-point accuracy difference.

The asymmetry that is *not* yet measured runs the other way: because the split
corpus is longer, the split arm covered a smaller fraction of its documents
than dense covered of its own. The logs cannot size this, since split never
wrapped. `ls -l` on the two `train.bin` files on the cluster settles it.

## What is missing

**No evaluations have been run.** There are no `evals/` directories on the
cluster for any of the 12 runs. The preregistered deliverable is the fact-use
QA delta and the recall/store-off probes, none of which exist yet. The training
is done; the science is not.

Running them needs the checkpoints, which are on the cluster, not here:

```bash
# on rice-04, per run
python scripts/run_evals.py --run outputs/d160m_split_n200k_s0 --load n200k
```

## What is still on the cluster

`/scratch/users/syz/memorysplit/outputs/` — 194 GB total, and the only copy.

| Per run | Size |
|---|---|
| `ckpt.pt` | 1.95 GB (fp32 weights + AdamW state) |
| `snapshots/step*.pt` | 10 × 649 MB |

Plus the two `d1b` gate runs at 50 GB each. Scratch is not backed up. If those
checkpoints are lost, the 12 runs have to be retrained to be evaluated.
