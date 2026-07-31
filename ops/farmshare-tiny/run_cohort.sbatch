#!/bin/bash
# Train the tiny crowding cohort: 2 models x 2 arms, one GPU each, one node.
#
# Usage:  MS_ROOT=... sbatch run_cohort.sbatch
#
# By default all four runs share one four-GPU allocation. Both arms of a pair
# must sit on identical hardware because the contrast is within-pair, and
# putting all four on one node also holds hardware constant across the two
# model sizes.
#
# The gpu partition is contended and a contiguous four-GPU block can queue for
# most of a day. Fallback, one model per job:
#
#   MS_MODELS=d8m  sbatch --gres=gpu:2 run_cohort.sbatch
#   MS_MODELS=d40m sbatch --gres=gpu:2 run_cohort.sbatch
#
# That keeps each pair on one node, so the primary within-pair contrast is
# untouched. Only across-model hardware matching is lost, and that comparison
# is already confounded by recurrence and tying. QoS gpu caps gres/gpu at 4,
# so the two halves can still run side by side.

#SBATCH --job-name=ms-tiny
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=47:00:00
#SBATCH --output=ms-tiny-%j.out

set -u

: "${MS_ROOT:?MS_ROOT must be exported before sbatch}"
CODE=$MS_ROOT/code
PY=$MS_ROOT/venv/bin/python
RUNS=$MS_ROOT/runs/tiny-v3
LOGS=$MS_ROOT/logs
CFG_DIR=$CODE/configs/tiny-v3
MAX_STEPS=15582
SMOKE=${SMOKE:-0}
MS_MODELS=${MS_MODELS:-"d8m d40m"}
TAG=$(echo "$MS_MODELS" | tr ' ' '-')

mkdir -p "$LOGS" "$RUNS"
cd "$CODE" || exit 1
say() { echo "[$(date -u +%FT%TZ)] $*"; }

say "tiny cohort starting on $(hostname), job ${SLURM_JOB_ID:-none}"
say "models: $MS_MODELS  smoke=$SMOKE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

pairs=""
for model in $MS_MODELS; do
  pairs="$pairs ${model}:dense ${model}:split90"
done
prefix=""
if [ "$SMOKE" = "1" ]; then
  pairs="d8m:dense d8m:split90"
  MS_MODELS="d8m"
  prefix="smoke-"
fi

# Both arms of a pair must agree on everything but the base sidecar.
for model in $MS_MODELS; do
  for key in max_steps tokens_per_step micro_batch_size lr ctx model model_parameters; do
    a=$(grep "^${key}:" "$CFG_DIR/${prefix}${model}-dense-s0.yaml")
    b=$(grep "^${key}:" "$CFG_DIR/${prefix}${model}-split90-s0.yaml")
    [ "$a" = "$b" ] || { say "ABORT $model arms disagree on $key: '$a' vs '$b'"; exit 1; }
  done
done
say "pair preflight ok"

pids=(); names=(); gpu=0
for spec in $pairs; do
  model=${spec%%:*}; arm=${spec##*:}
  name="${model}_${arm}_s0"
  mkdir -p "$MS_ROOT/ind-cache-tiny/g${gpu}"
  CUDA_VISIBLE_DEVICES=$gpu \
  PYTHONPATH=$CODE \
  OMP_NUM_THREADS=4 \
  TORCHINDUCTOR_CACHE_DIR="$MS_ROOT/ind-cache-tiny/g${gpu}" \
    "$PY" "$CODE/scripts/run_train.py" \
      --config "$CFG_DIR/${prefix}${model}-${arm}-s0.yaml" \
      --resume auto \
      >> "$LOGS/${prefix}${name}.log" 2>&1 &
  pids+=($!); names+=("$name")
  say "  gpu$gpu <- $name (pid ${pids[-1]})"
  gpu=$((gpu + 1))
done

rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then say "  OK   ${names[$i]}"; else say "  ENDED ${names[$i]} non-zero"; rc=1; fi
done

[ "$SMOKE" = "1" ] && { say "smoke done"; exit $rc; }

done_all=1
for model in $MS_MODELS; do
  for arm in dense split90; do
    s=$(tail -1 "$RUNS/${model}_${arm}_reasoning_v3_s0/log.jsonl" 2>/dev/null \
        | sed -n 's/.*"step": \([0-9]*\).*/\1/p')
    say "  ${model}_${arm} at ${s:-0} of $MAX_STEPS"
    [ "${s:-0}" -ge "$MAX_STEPS" ] || done_all=0
  done
done
[ "$done_all" = "1" ] && { say "COMPLETE for $MS_MODELS"; touch "$RUNS/.complete-$TAG"; exit 0; }

# Marker is per model set so the two halves of a split submission cannot
# read each other's progress and wrongly conclude a run has stalled.
MARK=$RUNS/.progress-$TAG
prev=$(cat "$MARK" 2>/dev/null || echo 0)
now=$(for model in $MS_MODELS; do for arm in dense split90; do
        tail -1 "$RUNS/${model}_${arm}_reasoning_v3_s0/log.jsonl" 2>/dev/null \
        | sed -n 's/.*"step": \([0-9]*\).*/\1/p'; done; done | awk '{s+=$1} END{print s+0}')
echo "$now" > "$MARK"
[ "$now" -le "$prev" ] && { say "ABORT chain: no progress ($prev -> $now)"; exit 1; }
say "resubmitting to continue"
ngpu=$(( $(echo "$MS_MODELS" | wc -w) * 2 ))
MS_MODELS="$MS_MODELS" sbatch --gres=gpu:$ngpu "$MS_ROOT/run_cohort_tiny.sbatch"
exit $rc
