#!/bin/bash
# Supervisor for the d1b seed-pair cohort on one 8x B200 node.
#
# Topology: one GPU per run, so a seed pair occupies two adjacent GPUs and
# both arms train simultaneously on symmetric hardware. Four pairs per round,
# two rounds, eight pairs total.
#
# Detached on purpose: this must outlive the SSM session that starts it.

set -u

CODE=/mnt/nvme/code
PY=/opt/ms/venv/bin/python
RUNS=/mnt/nvme/runs/1b-v3
LOGS=/mnt/nvme/logs
S3=s3://${MS_S3_BUCKET}/runs/1b-v3
AWS=/usr/local/bin/aws
STATUS=$RUNS/COHORT-STATUS.txt

# AWS reclaims the instance 30 min before the block ends (11:30Z), so treat
# 11:00Z as destruction and stop with margin for the final upload.
DEADLINE=$(date -u -d '2026-07-27T10:15:00Z' +%s)

mkdir -p "$RUNS" "$LOGS"

say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$STATUS"; }

launch_round() {
  local round=$1; shift
  local seeds=("$@")
  local gpu=0
  local pids=() names=()

  say "ROUND $round starting: seeds ${seeds[*]} (8 runs, 1 GPU each)"
  for s in "${seeds[@]}"; do
    for arm in dense split90; do
      local name="d1b_${arm}_s${s}"
      mkdir -p "/mnt/nvme/ind-cache/g${gpu}"
      CUDA_VISIBLE_DEVICES=$gpu \
      PYTHONPATH=$CODE \
      OMP_NUM_THREADS=8 \
      TORCHINDUCTOR_CACHE_DIR="/mnt/nvme/ind-cache/g${gpu}" \
        nohup "$PY" "$CODE/scripts/run_train.py" \
          --config "$CODE/configs/1b-v3/${arm}-s${s}.yaml" \
          --resume auto \
          >> "$LOGS/${name}.log" 2>&1 &
      pids+=($!); names+=("$name")
      say "  gpu$gpu <- $name (pid ${pids[-1]})"
      gpu=$((gpu + 1))
    done
  done

  local failed=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      say "  OK   ${names[$i]}"
    else
      say "  FAIL ${names[$i]} (see $LOGS/${names[$i]}.log)"
      failed=$((failed + 1))
    fi
  done
  say "ROUND $round finished with $failed failure(s)"
  return 0
}

# Retry any run that died early, as long as the deadline allows it.
# --resume auto restarts from the last 30-minute checkpoint.
retry_round() {
  local round=$1; shift
  local seeds=("$@")
  local gpu=0 pids=() names=()
  for s in "${seeds[@]}"; do
    for arm in dense split90; do
      local name="d1b_${arm}_s${s}"
      local run_id="d1b_${arm}_reasoning_v3_s${s}"
      local done_step
      done_step=$(tail -1 "$RUNS/$run_id/log.jsonl" 2>/dev/null | sed -n 's/.*"step": \([0-9]*\).*/\1/p')
      if [ "${done_step:-0}" -lt 15582 ] && [ "$(date -u +%s)" -lt "$DEADLINE" ]; then
        say "  RETRY $name from step ${done_step:-0}"
        CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$CODE OMP_NUM_THREADS=8 \
        TORCHINDUCTOR_CACHE_DIR="/mnt/nvme/ind-cache/g${gpu}" \
          nohup "$PY" "$CODE/scripts/run_train.py" \
            --config "$CODE/configs/1b-v3/${arm}-s${s}.yaml" --resume auto \
            >> "$LOGS/${name}.log" 2>&1 &
        pids+=($!); names+=("$name")
      fi
      gpu=$((gpu + 1))
    done
  done
  [ ${#pids[@]} -eq 0 ] && return 0
  say "ROUND $round retry: ${#pids[@]} run(s)"
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || say "  RETRY FAIL ${names[$i]}"; done
}

full_sync() {
  say "S3 sync (full) starting"
  "$AWS" s3 sync "$RUNS" "$S3" --exclude "*_smoke*" --only-show-errors \
    && say "S3 sync (full) complete" || say "S3 sync (full) REPORTED ERRORS"
}

say "=== d1b cohort supervisor up (pid $$) ==="
say "deadline for final sync: $(date -u -d @$DEADLINE +%FT%TZ)"

launch_round 1 0 1 2 3
retry_round  1 0 1 2 3
full_sync

now=$(date -u +%s)
remaining=$(( (DEADLINE - now) / 3600 ))
say "round 1 done; ${remaining}h until deadline"

if [ "$remaining" -ge 15 ]; then
  launch_round 2 4 5 6 7
  retry_round  2 4 5 6 7
  full_sync
else
  say "SKIPPING round 2: only ${remaining}h left, a round needs ~15h"
fi

full_sync
say "=== cohort supervisor finished ==="
touch "$RUNS/COHORT-COMPLETE"
"$AWS" s3 cp "$STATUS" "$S3/COHORT-STATUS.txt" --only-show-errors
