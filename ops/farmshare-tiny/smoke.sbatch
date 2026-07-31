#!/bin/bash
# 30-step smoke for the tiny cohort: both d8m arms, sequentially, on one GPU.
#
# Usage:  MS_ROOT=... sbatch smoke.sbatch
#
# Deliberately a separate file from run_cohort.sbatch and a different job
# shape. The gpu partition is usually saturated and a 4-GPU ask can queue for
# most of a day, whereas the normal partition carries the same L40S nodes and
# its QoS allows one GPU, which is all a sequential smoke needs.
#
# The gate is not that training runs. It is that `loss_masked_values` appears
# in BOTH arms' logs: that metric silently produced nothing on this corpus for
# every previous cohort, and a run without it cannot answer the question.

#SBATCH --job-name=ms-tiny-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --output=ms-tiny-smoke-%j.out

set -u

: "${MS_ROOT:?MS_ROOT must be exported before sbatch}"
CODE=$MS_ROOT/code
PY=$MS_ROOT/venv/bin/python
RUNS=$MS_ROOT/runs/tiny-v3
LOGS=$MS_ROOT/logs
CFG_DIR=$CODE/configs/tiny-v3

mkdir -p "$LOGS" "$RUNS" "$MS_ROOT/ind-cache-tiny/smoke"
cd "$CODE" || exit 1
say() { echo "[$(date -u +%FT%TZ)] $*"; }

say "tiny smoke on $(hostname), job ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

rc=0
for arm in dense split90; do
  say "--- d8m $arm ---"
  PYTHONPATH=$CODE OMP_NUM_THREADS=4 \
  TORCHINDUCTOR_CACHE_DIR="$MS_ROOT/ind-cache-tiny/smoke" \
    "$PY" "$CODE/scripts/run_train.py" \
      --config "$CFG_DIR/smoke-d8m-${arm}-s0.yaml" \
      --resume none \
      2>&1 | tail -5
  [ "${PIPESTATUS[0]}" -eq 0 ] || { say "  $arm FAILED"; rc=1; }
done

say "--- logs ---"
probe_seen=0
for arm in dense split90; do
  f=$RUNS/_smoke_d8m_${arm}_reasoning_v3_s0/log.jsonl
  say "d8m_$arm:"
  cat "$f" 2>/dev/null
  grep -q loss_masked_values "$f" 2>/dev/null && probe_seen=$((probe_seen + 1))
done

if [ "$probe_seen" -eq 2 ]; then
  say "GATE PASSED: loss_masked_values present in both arms"
else
  say "GATE FAILED: probe fired in $probe_seen/2 arms - do not launch the cohort"
  rc=1
fi
exit $rc
