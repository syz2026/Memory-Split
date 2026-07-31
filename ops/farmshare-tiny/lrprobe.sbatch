#!/bin/bash
# Learning-rate probe for the tiny cohort: 400 dense-only steps per rate.
#
# Usage:  MS_ROOT=... sbatch lrprobe.sbatch
#
# The repo's frozen tiers are 1.5e-3 at 160M, 1e-3 at 410M, 6e-4 at 1B. Neither
# 8M nor 40M has a tier, and an untuned rate is exactly the confound that would
# make a model look capacity-crowded when it is merely badly optimised.
#
# Selection uses the DENSE arm only and the winner is then applied to both
# arms, so the choice cannot favour either side of the contrast.
#
# One GPU on the normal partition, sequentially: the normal QoS caps gres/gpu
# at 1, and the gpu partition queues for most of a day.

#SBATCH --job-name=ms-tiny-lr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --output=ms-tiny-lr-%j.out

set -u

: "${MS_ROOT:?MS_ROOT must be exported before sbatch}"
CODE=$MS_ROOT/code
PY=$MS_ROOT/venv/bin/python
CFG_DIR=$CODE/configs/tiny-lrprobe
RUNS=$MS_ROOT/runs/tiny-lrprobe

mkdir -p "$RUNS" "$MS_ROOT/ind-cache-tiny/lr"
cd "$CODE" || exit 1
say() { echo "[$(date -u +%FT%TZ)] $*"; }

say "lr probe on $(hostname), job ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for cfg in "$CFG_DIR"/*.yaml; do
  name=$(basename "$cfg" .yaml)
  say "--- $name ---"
  PYTHONPATH=$CODE OMP_NUM_THREADS=4 \
  TORCHINDUCTOR_CACHE_DIR="$MS_ROOT/ind-cache-tiny/lr" \
    "$PY" "$CODE/scripts/run_train.py" --config "$cfg" --resume none 2>&1 | tail -3
done

say "=== results: lowest final loss_ema wins, per model ==="
for f in "$RUNS"/*/log.jsonl; do
  printf '%-24s %s\n' "$(basename "$(dirname "$f")")" "$(tail -1 "$f")"
done
