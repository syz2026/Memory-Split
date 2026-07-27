#!/bin/bash
# Chained supervisor for the d135m n=10 cohort.
#
# Waits for the running d1b supervisor to finish, then spends the margin left
# in the capacity block on the 135m cohort: 10 seed pairs, 20 runs, one GPU
# each, three rounds of 4 + 4 + 2 pairs.
#
# Deliberately a separate file from run_cohort.sh. That script is executing
# right now and bash reads scripts incrementally, so editing it in place can
# corrupt the live run.
#
# Durability is self-contained for the same reason: sync_daemon.sh only pushes
# runs/1b-v3 and is also running, so this script does its own periodic and
# final sync for runs/135m-v3. It stops at 09:45Z, 30 min ahead of the
# daemon's global pkill at 10:15Z, so the last upload is not racing it.

set -u

CODE=/mnt/nvme/code
PY=/opt/ms/venv/bin/python
RUNS=/mnt/nvme/runs/135m-v3
LOGS=/mnt/nvme/logs
S3=s3://${MS_S3_BUCKET}/runs/135m-v3
AWS=/usr/local/bin/aws
STATUS=$RUNS/COHORT-STATUS.txt
INDCACHE=/mnt/nvme/ind-cache-135m
MAX_STEPS=15582

LEAD_PID=160557
LEAD_FLAG=/mnt/nvme/runs/1b-v3/COHORT-COMPLETE

DEADLINE=$(date -u -d '2026-07-27T09:45:00Z' +%s)
# A round measured 2.78h. Refuse to start one without the round plus a sync
# and slack, so a late finish never leaves a half-trained round for the
# deadline to kill.
ROUND_NEEDED=11400

mkdir -p "$RUNS" "$LOGS" "$INDCACHE"

say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$STATUS"; }

cd "$CODE" || { echo "cannot cd $CODE"; exit 1; }

say "=== d135m chained supervisor up (pid $$) ==="
say "own deadline $(date -u -d @$DEADLINE +%FT%TZ), round budget $((ROUND_NEEDED / 60)) min"

# Refuse to launch the repo's unpatched configs. They point at a relative
# corpus path and write outside the tree this script syncs, so a cohort
# started from them would either die at once or be lost with the node.
for f in "$CODE"/configs/135m-v3/*.yaml; do
  grep -q "^out_dir: $RUNS/" "$f" || { say "ABORT $(basename "$f") out_dir is not under $RUNS"; exit 1; }
  grep -q "^micro_batch_size: 64$" "$f" || { say "ABORT $(basename "$f") is not micro_batch_size 64"; exit 1; }
done
say "config preflight ok: 20 configs, absolute out_dir, micro_batch_size 64"

say "waiting for d1b supervisor (pid $LEAD_PID)"
while kill -0 "$LEAD_PID" 2>/dev/null; do
  if [ "$(date -u +%s)" -ge "$DEADLINE" ]; then
    say "deadline reached while d1b was still running; nothing to do"
    exit 0
  fi
  sleep 120
done

if [ -f "$LEAD_FLAG" ]; then
  say "d1b supervisor completed cleanly"
else
  say "WARN d1b supervisor exited without $LEAD_FLAG (crash or kill); using the remaining block anyway"
fi

# The supervisor waits on its children so this should already hold. Guard
# anyway: two trainers on one GPU would halve both and fit neither. Loop
# against the deadline rather than a fixed count, because if the d1b
# supervisor was ever restarted under a new pid the wait above returns
# immediately and this is the only thing preventing a collision.
waited=0
while true; do
  n=$(pgrep -cf 'scripts/run_train.py' || true)
  [ "${n:-0}" -eq 0 ] && break
  if [ "$(( $(date -u +%s) + ROUND_NEEDED ))" -ge "$DEADLINE" ]; then
    say "ABORT ${n} trainer(s) still running and no room left for a round"
    exit 1
  fi
  [ $((waited % 10)) -eq 0 ] && say "waiting for ${n} trainer(s) to exit"
  waited=$((waited + 1))
  sleep 60
done

watchdog() {
  while [ "$(date -u +%s)" -lt "$DEADLINE" ]; do sleep 60; done
  say "DEADLINE reached - stopping d135m runs"
  pkill -TERM -f 'configs/135m-v3' 2>/dev/null
  sleep 60
  pkill -KILL -f 'configs/135m-v3' 2>/dev/null
}

syncer() {
  while true; do
    sleep 900
    "$AWS" s3 sync "$RUNS" "$S3" --only-show-errors \
      && say "periodic sync ok" || say "periodic sync errors"
  done
}

watchdog & WATCHDOG_PID=$!
syncer & SYNCER_PID=$!

launch_round() {
  local round=$1; shift
  local seeds=("$@")
  local gpu=0 pids=() names=()

  say "ROUND $round starting: seeds ${seeds[*]} ($(( ${#seeds[@]} * 2 )) runs, 1 GPU each)"
  for s in "${seeds[@]}"; do
    for arm in dense split90; do
      local name="d135m_${arm}_s${s}"
      mkdir -p "$INDCACHE/g${gpu}"
      CUDA_VISIBLE_DEVICES=$gpu \
      PYTHONPATH=$CODE \
      OMP_NUM_THREADS=8 \
      TORCHINDUCTOR_CACHE_DIR="$INDCACHE/g${gpu}" \
        nohup "$PY" "$CODE/scripts/run_train.py" \
          --config "$CODE/configs/135m-v3/${arm}-s${s}.yaml" \
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
}

# Restart any run that died early, as long as the deadline allows a full one.
# --resume auto picks up from the last 30-minute checkpoint.
retry_round() {
  local round=$1; shift
  local seeds=("$@")
  local gpu=0 pids=() names=()
  for s in "${seeds[@]}"; do
    for arm in dense split90; do
      local name="d135m_${arm}_s${s}"
      local run_id="d135m_${arm}_reasoning_v3_s${s}"
      local done_step
      done_step=$(tail -1 "$RUNS/$run_id/log.jsonl" 2>/dev/null | sed -n 's/.*"step": \([0-9]*\).*/\1/p')
      if [ "${done_step:-0}" -lt "$MAX_STEPS" ] && \
         [ "$(( $(date -u +%s) + ROUND_NEEDED ))" -lt "$DEADLINE" ]; then
        say "  RETRY $name from step ${done_step:-0}"
        mkdir -p "$INDCACHE/g${gpu}"
        CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$CODE OMP_NUM_THREADS=8 \
        TORCHINDUCTOR_CACHE_DIR="$INDCACHE/g${gpu}" \
          nohup "$PY" "$CODE/scripts/run_train.py" \
            --config "$CODE/configs/135m-v3/${arm}-s${s}.yaml" --resume auto \
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

round_if_time() {
  local round=$1; shift
  local remaining=$(( DEADLINE - $(date -u +%s) ))
  if [ "$remaining" -lt "$ROUND_NEEDED" ]; then
    say "SKIPPING round $round: $((remaining / 60)) min left, a round needs $((ROUND_NEEDED / 60)) min"
    return 1
  fi
  launch_round "$round" "$@"
  retry_round "$round" "$@"
  "$AWS" s3 sync "$RUNS" "$S3" --only-show-errors \
    && say "round $round sync ok" || say "round $round sync errors"
  return 0
}

round_if_time 1 0 1 2 3
round_if_time 2 4 5 6 7
round_if_time 3 8 9

kill "$WATCHDOG_PID" "$SYNCER_PID" 2>/dev/null
"$AWS" s3 sync "$RUNS" "$S3" --only-show-errors \
  && say "final sync ok" || say "final sync REPORTED ERRORS"
touch "$RUNS/COHORT-135M-COMPLETE"
"$AWS" s3 cp "$STATUS" "$S3/COHORT-STATUS.txt" --only-show-errors
say "=== d135m chained supervisor finished ==="
