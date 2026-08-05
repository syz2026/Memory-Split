# Runbook

FarmShare only. One L40S per run, `--partition=gpu`, 47-hour wall, four
concurrent GPUs, and keep total scratch under 150 GB.

Everything up to "submit" is implemented and tested offline. The runs
themselves need the cluster.

## Measured throughput

At `ctx=1024`, `micro_batch_size=32`, compiled, from completed runs:

| Preset | tok/s | source |
|---|---:|---|
| `d8m` | 462,611 | `outputs/farmshare-tiny` |
| `d40m` | 184,671 | `outputs/farmshare-tiny` |
| `d160m` | 134,754 | `outputs/farmshare-160m-v3-s0` |

`d40m_std` drops `n_recurrence` from 2 to 1, which cuts forward FLOPs per
token from about 124M to 81M — expect roughly 1.5x, but **re-measure it in
Stage A** rather than trusting that.

## The budget, at the corpus actually in use

Every figure below that was priced at 5.737B tokens per run is stale.
`docs/THEORY-ENDPOINT.md` argues the Stage A floor is a step-budget artifact,
and `advance.py` accordingly sets `FULL_STEPS = 31_280`. At 524,288 tokens per
step that is **16.4B tokens per run**, not 5.737B, and the two cannot both be
right. Priced at 16.4B:

| quantity | value |
|---|---:|
| tokens per run | 16,399,769,600 |
| hours per run, `d40m_std` at ~277k tok/s | **16.4** |
| hours per run, `d40m` at 184,671 tok/s | 24.7 |
| matrix, 3 loads x 3 arms x 8 seeds = 72 runs | **~1,184 GPU-h** |
| matrix, 2 loads x 3 arms x 8 seeds = 48 runs | ~790 GPU-h |
| wall time at four concurrent GPUs | **~12.3 days** |
| bytes per token, uint16 stream + two uint8 sidecars | 4 |
| **GB per load** | **65.6** |
| loads resident at once against a 150 GB budget | **2** |

A 16.4 h run fits the 47-hour wall with room for a restart. Three loads do not
fit scratch, so the ladder is built, trained and deleted one dose at a time
with the checkpoint and `summary.json` retained; the ~2 h build cannot overlap
the third dose's training.

## Corpus build throughput, measured

| build | workers | tokens | wall | tok/min |
|---|---:|---:|---:|---:|
| `high` (2026-08-01, failed verify) | 64 | 16.4B | ~73 min | 225M |
| ladder rung | 16 | 800M | ~12 min | 67M |
| `high-e200` (2026-08-02) | 128 | 21.3B | ~8.5 h | 41M |

**More workers was not faster, and may have been slower.** The 128-worker build
runs at 41M tok/min against the 64-worker build's 225M, with every worker at
99% CPU, 143 GB of 377 GB memory in use and no swap. Difficulty matching is
not the explanation: it measures at 1.17x the unmatched placement cost, and
after the sliding-window rewrite RANDPOS accounts for ~0.27 h of a 128-worker
build. The remaining suspect is memory-bandwidth contention across 128
concurrent numpy processes.

**Benchmark 32 / 64 / 128 workers on one 800M rung before building three
matrix loads.** Three loads at 8.5 h each is a day of wall clock that may be
buying nothing over 64 workers.

## Before anything else: freeze the RANDPOS difficulty table

The primary contrast is FACTMASK minus RANDPOS, so the control has to remove
the same amount of learnable *signal*, not merely the same number of targets.
Until 2026-08-02 no corpus in this project was difficulty-matched, because
`build_corpus.py` never passed `token_nll`. A build now refuses without one.

```bash
$MS_PY ops/crowding/nll_table.py \
  --run $MS_ROOT/runs/<a finished SUP run> \
  --out $MS_ROOT/corpora/nll_table.npy --seed 9001
```

Seed 9001 is a burned pilot seed (§6) and may not appear in the matrix:
building the table from data that also appears in the matrix leaks the outcome
into the design. Freeze the `.npy` and its `.provenance.json` once. Rebuilding
it after seeing an outcome invalidates every corpus placed with it.

## The occupancy ladder — test premise 1 before premise 3 (~64 GPU-h)

Memory Split has three premises: facts occupy parameters, masking frees them,
and the freed parameters go to reasoning. **Four experiment generations tested
premise 3 while premise 1 was false**, and all four measured noise around zero
because the corpora never stored anything for masking to free. This ladder
tests premise 1 directly, needs no arms, and answers in ~64 GPU-hours rather
than the matrix's 1,495.

```bash
bash ops/crowding/crowding_ladder.sh --plan     # print the plan, submit nothing
bash ops/crowding/crowding_ladder.sh            # build, train, evaluate
PYTHONPATH=. $MS_PY ops/crowding/occupancy.py --runs $MS_ROOT/runs
```

Document count is held fixed, so tokens, steps, schedule and mask mass are
identical across rungs and only the unique-entropy ceiling moves:

| exposures | entities | demand | C(E) | F/C | regime | 3E gain |
|---:|---:|---:|---:|---:|---|---:|
| 800 | 249,102 | 0.325 | 1.903 | 0.171 | under | 5.1% (saturated) |
| 400 | 498,204 | 0.651 | 1.602 | 0.406 | under | 24.8% |
| 200 | 996,408 | 1.301 | 1.301 | **1.000** | critical | 36.7% |

Every rung is 21.3B tokens, 40,695 steps, ~79 GB, ~21.4 h on one L40S. The
200-exposure rung is `high-e200`, already built and `VERIFY OK`, so only two
corpora need building. Build and delete one at a time: three resident is 237 GB
against a 150 GB budget.

| signature | reading |
|---|---|
| **saturating** | capacity binds; the crowding regime is reached and the arm contrast is finally worth running |
| **abandonment** | the model declines to learn rather than compressing; crowding cannot occur at this scale |
| **linear** | nothing binds; raise the load, not the seed count |
| **inert** | nothing stored anywhere; exposures are still below the storage threshold |

**What it cannot do.** No rung above F/C = 1 fits: F/C 1.5 needs 119 GB, F/C 2.0
needs 159 GB, and reaching F/C = 1 while also clearing the exposure-saturation
gate needs 465 GB. The ladder brackets the critical point from below only, so a
plateau at the top rung is evidence that capacity has begun to bind, not proof
that it has bound hard.

## The endpoint that does not floor

Exact-match accuracy is a threshold metric: it requires a whole correct chain
of thought *and* the right terminating token, so it reads zero across the
entire range where a model is learning. Every run in this project has sat at
that floor, and a floored endpoint cannot exhibit crowding however much
capacity is freed.

`evals/continuous.py` has implemented reference-trace NLL since the
reasoning-v3 line and was **called by nothing until 2026-08-03**. It is now
wired into `run_evals.py` and every summary carries:

| key | meaning |
|---|---|
| `igsm_m1_nll` | mean per-token −log p(gold trace \| prompt). Primary. |
| `igsm_m2_nll` | −log p(gold answer \| prompt + gold CoT). Isolates the final step. |
| `igsm_m1_nll_neg` | sign-flipped, for use as `--endpoint` |

The analyzer's verdict is one-sided and assumes larger is better, so pass the
`_neg` twin as the estimand and read the raw value:

```bash
$MS_PY scripts/analyze_crowding.py --endpoint igsm_m1_nll_neg ...
```

## Resuming after a disconnect

SLURM keeps running whether or not anyone is watching, so after a gap the first
question is never "what should I launch" but "what already happened". One call
answers it and names the next action:

```bash
PYTHONPATH=. $MS_PY ops/crowding/status.py --root $MS_ROOT
```

It reports every corpus with its verification state and size against the
150 GB budget, every run with its last step and whether a checkpoint, eval and
snapshots exist, the live queue, and then the single next action. It
specifically catches the case a queue dump hides — trains finished, evals
died, nothing running, so it looks complete.

**SSH to FarmShare needs the Stanford VPN.** If every `*.farmshare.stanford.edu`
host is unreachable on port 22 while `stanford.edu:443` is open, the tunnel has
dropped rather than the cluster being down. Reconnect and re-run the above;
jobs submitted before the drop will have carried on regardless.

## Rehearse the whole chain first

Nine seconds on CPU, and it has already caught one bug that would have
crashed evaluation partway through the matrix.

```bash
PYTHONPATH=. python3 scripts/smoke_pipeline.py
```

Builds a corpus, trains all three arms from one stream, evaluates each, runs
the frozen analyzer, and checks it refuses an incomplete matrix. Expect
`PIPELINE OK`.

## Environment

```bash
export MS_CODE=$SCRATCH/crowding/code
export MS_ROOT=$SCRATCH/crowding
export MS_PY=$SCRATCH/crowding/venv/bin/python
mkdir -p "$MS_ROOT"/{corpora,configs,runs}
```

## Pin the natural bed first

The bed must be a pinned JSONL with a recorded hash. A synthetic bed is
rehearsal only, and the manifest will say `SYNTHETIC-REHEARSAL-ONLY` if you
forget — treat that string in a manifest as a build failure.

```bash
export MS_BED=$MS_ROOT/corpora/fineweb-edu.jsonl
shasum -a 256 "$MS_BED" | tee $MS_ROOT/corpora/bed.sha256
```

---

## Stage A — architecture at one difficulty (~4 GPU-h)

Answers the two cheapest questions that can void everything downstream: can
the endpoint move at all, and does dropping weight sharing cost the depth it
needs. **Stop if nothing clears majority-class + 5 points.**

> **This stage is not a difficulty ladder.** `build_corpus.py` takes a single
> `--mod`, so difficulty is fixed for a whole corpus; both Stage A cells ran
> MOD=23 and varied only architecture. The run of 2026-08-01 returned STOP on
> that one rung, with a step budget of 1,525 — roughly 7x short of what
> modular arithmetic is reported to need. Read `docs/THEORY-ENDPOINT.md`
> before acting on a STOP from here, and run the real ladder below.
>
> **That STOP is now withdrawn entirely.** Its corpus realised a 3.75% fact
> share against a requested 50%, on a `SYNTHETIC-REHEARSAL-ONLY` bed, carrying
> 0.0261 bits/param of fact demand against the 2.0 the design targets — a
> factor of 77. `docs/AMENDMENT-2026-08-02.md` §2. A
> Stage A rerun needs enough documents to fill the fact lane: at 800M tokens a
> 50% share needs ~5.3M fact documents, so 20,000 entities requires ~265
> exposures, not 20. The builder now refuses the old combination.

## The difficulty ladder (~4 GPU-h + 4 short CPU builds)

One corpus per modulus, everything else pinned to Stage A's values so a lift
is attributable to difficulty alone. Judged on **op=1 against the no-skill
baseline**, which is the majority rate rather than `1/mod`.

```bash
bash ops/crowding/ladder.sh                       # builds, trains, evals
$MS_PY ops/crowding/ladder.py --mode rank --runs $MS_ROOT/runs
```

| outcome | reading |
|---|---|
| clears at some rung | operation learnable; rerun the design at a modulus that clears |
| flat even at mod 5 | 5 answers and a 232-bit table — not a difficulty problem |

The step ladder is free: the trainer already writes snapshots, so score them
rather than launching runs.

```bash
for s in $MS_ROOT/runs/<run>/snapshots/step*.pt; do
  $MS_PY scripts/run_evals.py --run $MS_ROOT/runs/<run> --ckpt $s
done
$MS_PY ops/crowding/ladder.py --mode steps --run $MS_ROOT/runs/<run>
```

```bash
MS_OUT=$MS_ROOT/corpora/pilotA MS_ENTITIES=20000 MS_EXPOSURES=265 \
MS_TOKENS=800000000 MS_BED=$MS_BED MS_NLL=$MS_ROOT/corpora/nll_table.npy \
  sbatch ops/crowding/build.sbatch

$MS_PY ops/crowding/pilot.py --stage A \
  --out $MS_ROOT/configs/pilotA --corpus corpora/pilotA \
  --total-tokens 800000000 --lr 1.5e-3 --entities 20000

for c in $MS_ROOT/configs/pilotA/*.yaml; do
  MS_CFG=$c sbatch ops/crowding/train.sbatch
done
```

Then score iGSM per `(MOD, op)` cell and per architecture, and feed the cells
to `analyze_pilot.stage_a`. Freeze the architecture and the difficulty here
and write them into `docs/PREREGISTRATION.md` §5.

## Stage B — operating point and the effect ceiling (~35 GPU-h)

Two corpora at the final high-load settings. The NOFACT one replaces the fact
lane with bed tokens at equal token count and equal steps, which is what
measures `delta`: the total reasoning cost of carrying the facts, and
therefore the hard ceiling on any treatment effect.

> **The old NOFACT recipe no longer builds, and should not.**
> `MS_ENTITIES=1 MS_EXPOSURES=1` said "no facts" by starving the fact lane and
> letting the bed absorb its 70% share -- the same silent reallocation that
> produced the Stage A defect. Say it with a zero fact share instead, which
> puts the intent in the manifest as `lane_profile: NOFACT` and lets `verify`
> require empty sidecars rather than flagging them as an inert-mask bug.

```bash
# operating point: 996,408 x 200 gives >=200 exposures per fact and F/C = 1.000
MS_OUT=$MS_ROOT/corpora/high-e200 MS_ENTITIES=996408 MS_EXPOSURES=200 \
MS_TOKENS=21335900160 MS_BED=$MS_BED MS_NLL=$MS_ROOT/corpora/nll_table.npy \
MS_WORKERS=128 sbatch --cpus-per-task=128 --mem=320G ops/crowding/build.sbatch

# same tokens and steps, no memorisable facts
MS_OUT=$MS_ROOT/corpora/nofact MS_ENTITIES=1 MS_EXPOSURES=1 \
MS_TOKENS=21335900160 MS_BED=$MS_BED MS_WORKERS=128 \
MS_FACT_SHARE=0.0 MS_IGSM_SHARE=0.123 MS_DED_SHARE=0.043 MS_BED_SHARE=0.834 \
sbatch --cpus-per-task=128 --mem=320G ops/crowding/build.sbatch

$MS_PY ops/crowding/pilot.py --stage B --model d40m \
  --out $MS_ROOT/configs/pilotB \
  --corpus corpora/high --nofact-corpus corpora/nofact \
  --total-tokens 16399769600 --lr 1.5e-3 --entities 382900
```

Also build a `3E` variant at the operating entity count for the
exposure-saturation check. That check is the one that decides whether the
model is parameter-limited or merely under-exposed, and extrapolation is not
permitted in its place.

While here: raise `grad_clip` until the dense arm shows `clip_frac` under 1%.
Removing the mechanism beats monitoring it.

## Stage C — pilot triplets (~78 GPU-h)

Three complete SUP/FACTMASK/RANDPOS triplets at the high load. Gives the
paired SD that sets `n`, an early leakage check, and the RANDPOS difficulty
audit.

```bash
$MS_PY ops/crowding/pilot.py --stage C --model d40m \
  --out $MS_ROOT/configs/pilotC --corpus corpora/high \
  --total-tokens 16399769600 --lr 1.5e-3 --entities 382900
```

## The gate

```bash
$MS_PY ops/crowding/analyze_pilot.py \
  --pilot-json $MS_ROOT/runs/pilot-measurements.json \
  --out $MS_ROOT/runs/gate.json \
  --min-interesting-effect <derived from delta> --n-confirm 8
```

Exit 0 is GO, exit 2 is NO-GO. On NO-GO, stop and write
`docs/NO-GO-PAPER.md`, which was outlined before Stage A so the pilot already
produced its figures. Do not weaken a gate to avoid writing it.

On GO: fill every `[PILOT]` slot in `docs/PREREGISTRATION.md`, commit it, and
timestamp it externally. Nothing in it moves afterwards.

---

## The confirmatory matrix

Three arms x loads x `n` fresh seeds, `n` from Stage C with a floor of 6.
Because the load axis trades entities against exposures, every load costs the
same per run.

At `d40m_std` (~16.4 h/run at 16.4B tokens): three loads x 8 seeds is 72
runs and about **1,184 GPU-h**, roughly 12.3 days at four concurrent GPUs. Two
loads x 8 seeds is 48 runs and about 790 GPU-h. Three loads is strongly
preferred — it is the only way to run the threshold-versus-line shape test,
which is the only thing here that speaks to capacity — but it no longer costs
410 GPU-h, and the decision should be made against the real number.

```bash
for f in 1.0 0.3 0.1; do
  MS_OUT=$MS_ROOT/corpora/load_$f ... sbatch ops/crowding/build.sbatch
done

$MS_PY ops/crowding/gen_configs.py --out $MS_ROOT/configs/matrix \
  --model d40m_std --total-tokens 16399769600 --lr 1.5e-3 \
  --seeds 1 2 3 4 5 6 7 8 \
  --load high=corpora/load_1.0 --load mid=corpora/load_0.3 --load low=corpora/load_0.1
```

Submit in **complete seed blocks**, not arm-major waves, so an interruption
leaves a balanced matrix. Runs are resumable; rerunning an unchanged config
picks up its checkpoint.

Before the full submission, run a 500-step probe of one FACTMASK arm and
measure recoverable bits on the fact set. Values stay in the input context and
are masked only in the loss, so the model can still bind them through the
input pathway across many exposures. Learn that for one GPU-hour rather than
four hundred.

## Analysis

```bash
$MS_PY scripts/analyze_crowding.py \
  --runs-root $MS_ROOT/runs --out $MS_ROOT/runs/verdict.json \
  --endpoint igsm_acc --storage-floor <pilot> --igsm-band 0.17 0.85 \
  --min-interesting-effect <pilot> --primary-load <frozen in prereg §5> \
  --expect-loads high mid low --expect-seeds 1 2 3 4 5 6 7 8
```

It refuses a missing arm, a dropped seed, an unexpected seed set, a run
without evaluations, or an empty root. Do not analyse a partial matrix and
then complete it.

## Storage

**65.6 GB per load** at 16.4B tokens and 4 bytes per token, so at most two
loads are resident against the 150 GB budget and the ladder must be built,
trained and deleted one dose at a time. Plus about 487 MB per `d40m` checkpoint
with fp32 Adam state. Prune to two snapshots once a run's evals are written and sane,
and delete optimizer state on completion. Four retained snapshots across 72
runs would be 140 GB on its own, and the account that hit 261 GB was
cancelled mid-battery.
