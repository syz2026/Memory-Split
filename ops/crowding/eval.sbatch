#!/bin/bash
# Evaluate one finished run. Inference only, so it fits a short GPU slot and
# schedules far faster than a training job.
#
#   MS_RUN=$MS_ROOT/runs/<id> MS_CODE=... sbatch ops/crowding/eval.sbatch

#SBATCH --job-name=crowding-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=02:00:00
#SBATCH --output=crowding-eval-%j.out

set -euo pipefail

: "${MS_RUN:?set MS_RUN to the run directory}"
: "${MS_CODE:?set MS_CODE to the repo}"
PY=${MS_PY:-python3}

say() { echo "[$(date -u +%FT%TZ)] $*"; }
say "eval on $(hostname), job ${SLURM_JOB_ID:-none}"
say "  run $MS_RUN"

PYTHONPATH="$MS_CODE" "$PY" "$MS_CODE/scripts/run_evals.py" \
  --run "$MS_RUN" \
  --device cuda \
  --n-igsm "${MS_N_IGSM:-1500}" \
  --n-deduction "${MS_N_DED:-400}" \
  --n-storage-entities "${MS_N_STORAGE:-500}"

rc=$?
[ $rc -eq 0 ] && say "eval COMPLETE -> $MS_RUN/evals/summary.json" || say "eval FAILED"
exit $rc
