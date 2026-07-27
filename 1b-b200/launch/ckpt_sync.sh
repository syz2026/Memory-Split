#!/bin/bash
# Prompt checkpoint durability for runs/1b-v3.
#
# sync_daemon.sh excludes */ckpt.pt on three of every four passes, so a
# checkpoint can sit up to an hour before it reaches S3. That is fine when the
# node has a known end time and is not fine now that the run has to survive
# being shut down at short notice.
#
# This adds a second, narrower loop that pushes every 10 minutes. `aws s3 sync`
# compares size and mtime, so passes where nothing changed cost one LIST and
# upload nothing. Runs alongside sync_daemon rather than replacing it, and
# signals no process at all.

set -u

RUNS=/mnt/nvme/runs/1b-v3
S3=s3://${MS_S3_BUCKET}/runs/1b-v3
AWS=/usr/local/bin/aws
LOG=/mnt/nvme/logs/ckpt-sync.log
INTERVAL=600

# Stop when sync_daemon does its own final upload, so the two are not racing
# on the last pass.
HARD_STOP=$(date -u -d '2026-07-27T10:15:00Z' +%s)

say() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

say "ckpt sync up (pid $$), every ${INTERVAL}s until $(date -u -d @$HARD_STOP +%FT%TZ)"

while [ "$(date -u +%s)" -lt "$HARD_STOP" ]; do
  start=$(date -u +%s)
  if "$AWS" s3 sync "$RUNS" "$S3" --exclude '*_smoke*' --only-show-errors; then
    say "sync ok in $(( $(date -u +%s) - start ))s"
  else
    say "sync REPORTED ERRORS after $(( $(date -u +%s) - start ))s"
  fi
  sleep "$INTERVAL"
done

say "hard stop reached; sync_daemon handles the final upload"
