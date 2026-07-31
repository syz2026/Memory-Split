# Learning-rate probe, tiny crowding cohort

Run 2026-07-31 on FarmShare `oat-02` (L40S), jobs 1670865 and 1670876.
400 dense-only steps per rate, everything else identical to the cohort configs.

## Why

The repo's frozen tiers are 1.5e-3 at 160M, 1e-3 at 410M, 6e-4 at 1B. Neither
8M nor 40M has a tier. An untuned rate is exactly the confound that would make
a model look capacity-crowded when it is merely badly optimised, which would
sink the whole crowding readout.

**Selection used the dense arm only**, and the winner is applied unchanged to
both arms. Tuning per-arm would bias the split-minus-dense contrast, which is
the quantity the cohort exists to measure.

## Grid and results

Final `loss_ema` at step 400. Lower is better.

| lr | d8m | d40m |
|---|---:|---:|
| 1.5e-3 | 5.5646 | 4.8455 |
| 2.0e-3 | — | 4.7711 |
| 3.0e-3 | 5.2983 | — |
| 4.0e-3 | — | 4.7389 |
| 6.0e-3 | 5.2302 | — |
| 8.0e-3 | — | **4.6371** |
| 1.2e-2 | **5.1476** | — |
| 1.6e-2 | — | 5.4013 |
| 2.4e-2 | 5.9214 | — |

Round 1 covered 1.5e-3 to 6e-3 for d8m and 1.5e-3 to 4e-3 for d40m. Both
winners landed on the top edge of their grid, so rather than accept an edge
value the grid was extended upward. Round 2 found the turn: both models get
worse at the next rate up, so each optimum is now bracketed on both sides.

## Frozen values

```
d8m   1.2e-2
d40m  8.0e-3
```

Both are far above the 1.5e-3 the 160M tier uses, which is the expected
direction — smaller models tolerate and want higher rates — and both are
higher than a naive width scaling from d160m would predict (3.7e-3 for d8m,
2.1e-3 for d40m). Training 200 to 1,000 tokens per parameter, far past
compute-optimal, plausibly explains the rest.

## Throughput, measured

| model | tok/s per L40S | per arm |
|---|---:|---:|
| d8m | ~466,000 | ~4.9 h |
| d40m | ~183,000 | ~12.4 h |

All four arms run concurrently in one allocation, so d40m sets the cohort wall
clock at roughly 12.4 h against a 47 h job.

## Caveat

400 steps is 210M tokens, 2.6% of the run. It ranks rates during warmup and
early descent, not at convergence. A rate that wins early can lose late,
particularly at the top of the range where instability shows up gradually. The
turn upward at 2.4e-2 and 1.6e-2 gives some confidence the chosen values are
not already unstable, but this is a ranking probe, not a full sweep.
