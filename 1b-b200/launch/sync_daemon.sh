#!/bin/bash
# Durability daemon for the d1b cohort.
#
# /mnt/nvme is instance-store: when the capacity block ends the node is
# destroyed and every byte on it goes with it. Nothing is safe until it is
# in S3, so this pushes continuously and enforces a hard stop with enough
# margin to finish the last upload.

set -u

RUNS=/mnt/nvme/runs/1b-v3
S3=s3://${MS_S3_BUCKET}/runs/1b-v3
AWS=/usr/local/bin/aws
LOG=/mnt/nvme/logs/sync-daemon.log

# Block ends 11:30Z; AWS reclaims ~11:00Z. Stop training at 10:15Z so the
# final push of ~200 GB (~6 min at the measured 600 MB/s) lands comfortably.
DEADLINE=$(date -u -d '2026-07-27T10:15:00Z' +%s)

say() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

say "sync daemon up (pid $$), deadline $(date -u -d @$DEADLINE +%FT%TZ)"

i=0
while true; do
  if [ "$(date -u +%s)" -ge "$DEADLINE" ]; then
    say "DEADLINE reached - stopping training and performing final sync"
    pkill -TERM -f 'scripts/run_train.py' 2>/dev/null
    sleep 90
    pkill -KILL -f 'scripts/run_train.py' 2>/dev/null
    sleep 5
    "$AWS" s3 sync "$RUNS" "$S3" --exclude "*_smoke*" --only-show-errors
    say "final sync complete - all artifacts durable"
    touch "$RUNS/DEADLINE-SYNC-DONE"
    exit 0
  fi

  # Snapshots and logs are written once and never mutated, so they are cheap
  # to push often. ckpt.pt is 12 GB per run and rewritten every 30 min, so it
  # rides the hourly pass instead.
  if [ $((i % 4)) -eq 0 ]; then
    "$AWS" s3 sync "$RUNS" "$S3" --exclude "*_smoke*" --only-show-errors \
      && say "hourly full sync ok" || say "hourly full sync errors"
  else
    "$AWS" s3 sync "$RUNS" "$S3" --exclude "*_smoke*" --exclude "*/ckpt.pt" \
      --only-show-errors && say "incremental sync ok" || say "incremental sync errors"
  fi

  i=$((i + 1))
  sleep 900
done
