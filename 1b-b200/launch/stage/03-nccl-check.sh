#!/bin/bash
# Stage 3 - collective communication health, then the exact concurrency shape
# the cohort runs in: four disjoint two-GPU pairs on one node.
set -uo pipefail

STAGE_DIR=/mnt/nvme/stage
LOGS=/mnt/nvme/stage/logs
mkdir -p "$LOGS"

echo "===== STAGE 3: NCCL + PAIR ALLOCATION ====="
date -u +%FT%TZ

FAIL=0

# torchrun lives in the DLAMI's /opt/pytorch venv, which a non-interactive SSM
# shell does not have on PATH.
TORCHRUN=""
for c in /opt/pytorch/bin/torchrun /opt/conda/bin/torchrun "$(command -v torchrun 2>/dev/null)"; do
  [ -n "$c" ] && [ -x "$c" ] && TORCHRUN="$c" && break
done
[ -n "$TORCHRUN" ] || { echo "FAIL: torchrun not found"; exit 1; }
echo "using torchrun: ${TORCHRUN}"

echo
echo "--- 3a: full 8-GPU all-reduce ---"
# Intra-node only: B200 pairs talk over NVLink/NVSwitch, so EFA is deliberately
# not in the path here and no EFA interfaces were attached at launch.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MS_LABEL=all8 \
NCCL_DEBUG=WARN \
"$TORCHRUN" --nproc_per_node=8 --master_port=29400 \
  "$STAGE_DIR/nccl_allreduce.py" 2>&1 | tee "$LOGS/nccl-all8.log"
[ "${PIPESTATUS[0]}" = "0" ] || { echo "FAIL: 8-GPU all-reduce failed"; FAIL=1; }

echo
echo "--- 3b: four disjoint 2-GPU pairs, concurrent ---"
# Each pair gets its own rendezvous port and its own physical GPUs. They are
# started together on purpose: running them serially would not prove they can
# coexist, which is the property the three-wave schedule depends on.
PAIRS=("0,1" "2,3" "4,5" "6,7")
PIDS=()
for i in "${!PAIRS[@]}"; do
  PAIR="${PAIRS[$i]}"
  PORT=$((29500 + i))
  LOG="$LOGS/nccl-pair${i}.log"
  echo "launching pair ${i} on GPUs ${PAIR} (port ${PORT}) -> ${LOG}"
  CUDA_VISIBLE_DEVICES="$PAIR" MS_LABEL="pair${i}_gpus_${PAIR}" \
  NCCL_DEBUG=WARN \
  "$TORCHRUN" --nproc_per_node=2 --master_port="$PORT" \
    "$STAGE_DIR/nccl_allreduce.py" > "$LOG" 2>&1 &
  PIDS+=($!)
done

for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "pair ${i}: exit 0"
  else
    echo "FAIL: pair ${i} exited non-zero"
    FAIL=1
  fi
done

echo
echo "--- pair logs ---"
for i in "${!PAIRS[@]}"; do
  echo "== pair ${i} (GPUs ${PAIRS[$i]}) =="
  cat "$LOGS/nccl-pair${i}.log"
done

echo
echo "--- 3c: disjointness assertion ---"
python3 "$STAGE_DIR/pair_alloc_check.py" --logs "$LOGS" || FAIL=1

echo
if [ "$FAIL" = "0" ]; then echo "STAGE 3 RESULT: PASS"; else echo "STAGE 3 RESULT: FAIL"; fi
exit $FAIL
