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

## Stage A — endpoint ladder (~4 GPU-h)

Answers the two cheapest questions that can void everything downstream: can
the endpoint move at all, and does dropping weight sharing cost the depth it
needs. **Stop if nothing clears majority-class + 5 points.**

```bash
MS_OUT=$MS_ROOT/corpora/pilotA MS_ENTITIES=20000 MS_EXPOSURES=20 \
MS_TOKENS=800000000 MS_BED=$MS_BED \
  sbatch ops/crowding/build.sbatch

$MS_PY ops/crowding/pilot.py --stage A \
  --out $MS_ROOT/configs/pilotA --corpus corpora/pilotA \
  --total-tokens 800000000 --lr 1.5e-3

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

```bash
# operating point
MS_OUT=$MS_ROOT/corpora/high MS_ENTITIES=382900 MS_EXPOSURES=100 \
MS_TOKENS=5737000000 MS_BED=$MS_BED sbatch ops/crowding/build.sbatch

# same tokens, no memorisable facts
MS_OUT=$MS_ROOT/corpora/nofact MS_ENTITIES=1 MS_EXPOSURES=1 \
MS_TOKENS=5737000000 MS_BED=$MS_BED sbatch ops/crowding/build.sbatch

$MS_PY ops/crowding/pilot.py --stage B --model d40m \
  --out $MS_ROOT/configs/pilotB \
  --corpus corpora/high --nofact-corpus corpora/nofact \
  --total-tokens 5737000000 --lr 1.5e-3
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
  --total-tokens 5737000000 --lr 1.5e-3
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

At `d40m_std` (~5.7 h/run): three loads x 8 seeds is 72 runs and about
410 GPU-h. Two loads x 8 seeds is 48 runs and about 274 GPU-h. Three loads is
strongly preferred — it is the only way to run the threshold-versus-line shape
test, which is the only thing here that speaks to capacity.

```bash
for f in 1.0 0.3 0.1; do
  MS_OUT=$MS_ROOT/corpora/load_$f ... sbatch ops/crowding/build.sbatch
done

$MS_PY ops/crowding/gen_configs.py --out $MS_ROOT/configs/matrix \
  --model d40m_std --total-tokens 5737000000 --lr 1.5e-3 \
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
  --min-interesting-effect <pilot> \
  --expect-loads high mid low --expect-seeds 1 2 3 4 5 6 7 8
```

It refuses a missing arm, a dropped seed, an unexpected seed set, a run
without evaluations, or an empty root. Do not analyse a partial matrix and
then complete it.

## Storage

Roughly 38 GB of corpora plus about 487 MB per `d40m` checkpoint with fp32
Adam state. Prune to two snapshots once a run's evals are written and sane,
and delete optimizer state on completion. Four retained snapshots across 72
runs would be 140 GB on its own, and the account that hit 261 GB was
cancelled mid-battery.
