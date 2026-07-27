#!/bin/bash
# Stage 2 - pull the 31.69 GB corpus to local NVMe and re-verify it end to end.
# Read-only against S3 by construction: the instance role carries an explicit
# Deny on any mutation of corpus/*, so this cannot damage the irreplaceable data.
set -uo pipefail

BUCKET=${MS_S3_BUCKET}
PREFIX=corpus/84142597cebd96e041d47c7c22dd4b42285b71a213b01265728042cb1a8f6fbb
DEST=/mnt/nvme/corpus
EXPECT_BYTES=31689510263
EXPECT_OBJECTS=10

echo "===== STAGE 2: CORPUS STAGING ====="
date -u +%FT%TZ

mountpoint -q /mnt/nvme || { echo "FAIL: /mnt/nvme not mounted, refusing to stage to root volume"; exit 1; }
mkdir -p "$DEST"

# Fail fast on the smallest object first. If SSE-KMS decrypt is not permitted,
# this surfaces it in under a second instead of 30 GB into a doomed transfer.
echo
echo "--- KMS/S3 smoke test (1.5 KB receipt) ---"
if ! aws s3 cp "s3://$BUCKET/$PREFIX/base/receipt.json" /tmp/_smoke.json --region us-east-1; then
  echo "FAIL: cannot read a single small object."
  echo "      Most likely kms:Decrypt on key 0e562bed-fda6-4a2c-9dee-b4c127a99ce3"
  echo "      is denied by the KMS key policy (IAM grant alone is not sufficient"
  echo "      unless the key policy delegates to the account root)."
  exit 1
fi
echo "smoke test OK"

echo
echo "--- transfer ---"
# 192 vCPU and 3200 Gbps of network: the default 10 concurrent requests leaves
# most of that idle, so widen the pool for the large sequential objects.
aws configure set default.s3.max_concurrent_requests 64
aws configure set default.s3.multipart_chunksize 64MB
aws configure set default.s3.max_queue_size 10000

START=$(date +%s)
aws s3 sync "s3://$BUCKET/$PREFIX/" "$DEST/" --region us-east-1 --only-show-errors || {
  echo "FAIL: s3 sync returned non-zero"; exit 1; }
END=$(date +%s)
echo "transfer wall time: $((END-START))s"

echo
echo "--- gross accounting ---"
ACTUAL_BYTES=$(find "$DEST" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
ACTUAL_OBJECTS=$(find "$DEST" -type f | wc -l | tr -d ' ')
echo "objects: ${ACTUAL_OBJECTS} (expect ${EXPECT_OBJECTS})"
echo "bytes:   ${ACTUAL_BYTES} (expect ${EXPECT_BYTES})"
[ "$ACTUAL_OBJECTS" = "$EXPECT_OBJECTS" ] || { echo "FAIL: object count mismatch"; exit 1; }
[ "$ACTUAL_BYTES" = "$EXPECT_BYTES" ]     || { echo "FAIL: byte total mismatch"; exit 1; }

echo
echo "--- cryptographic re-verification ---"
python3 /mnt/nvme/stage/verify_corpus.py \
  --manifest /mnt/nvme/stage/reasoning-v3-corpus-manifest.json \
  --root "$DEST"
RC=$?

echo
if [ "$RC" = "0" ]; then echo "STAGE 2 RESULT: PASS"; else echo "STAGE 2 RESULT: FAIL"; fi
exit $RC
