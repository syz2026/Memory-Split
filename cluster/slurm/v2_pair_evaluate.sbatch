#!/usr/bin/env bash
#SBATCH --job-name=ms135-pair-eval
#SBATCH --open-mode=append
#SBATCH --output=slurm-%j.out
# Resources are injected from a validated paired-Slurm profile.
set -Eeuo pipefail

: "${PAIR_MANIFEST:?PAIR_MANIFEST is required}"
: "${PROFILE_PATH:?PROFILE_PATH is required}"
: "${RUN_MODE:?RUN_MODE is required}"
: "${MS135_VENV:?MS135_VENV is required}"

repo="$(pwd -P)"
python="$MS135_VENV/bin/python"
evidence_root="$(dirname "$PAIR_MANIFEST")/../evidence"
mkdir -p "$evidence_root"

allocation_gpus="${CUDA_VISIBLE_DEVICES:-}"
IFS=',' read -r -a pair_gpus <<< "$allocation_gpus"
if [ "${#pair_gpus[@]}" -ne 2 ] \
  || [ -z "${pair_gpus[0]}" ] \
  || [ -z "${pair_gpus[1]}" ]; then
  echo "expected exactly two Slurm-assigned GPUs; got: ${allocation_gpus:-<unset>}" >&2
  exit 2
fi
dense_gpu="${pair_gpus[0]}"
split90_gpu="${pair_gpus[1]}"

common_env=(
  "HOME=${HOME:-$repo}"
  "PATH=/usr/bin:/bin"
  "PYTHONPATH=$repo"
  "MS135_VENV=$MS135_VENV"
  "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
  "SLURM_JOB_NAME=${SLURM_JOB_NAME:-ms135-pair-eval}"
  "SLURM_SUBMIT_DIR=$repo"
  "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}"
  "TOKENIZERS_PARALLELISM=false"
)
if [ -n "${TMPDIR:-}" ]; then
  common_env+=("TMPDIR=$TMPDIR")
fi
if [ -n "${NCCL_DEBUG:-}" ]; then
  common_env+=("NCCL_DEBUG=$NCCL_DEBUG")
fi

cd "$repo"
preflight_args=()
if [ "$RUN_MODE" = "protected" ]; then
  : "${PREFLIGHT_PATH:?PREFLIGHT_PATH is required for protected mode}"
  preflight_args=(--preflight "$PREFLIGHT_PATH")
fi
env -i "${common_env[@]}" "$python" scripts/validate_135m_launch.py \
  --manifest "$PAIR_MANIFEST" --profile "$PROFILE_PATH" \
  --mode "$RUN_MODE" "${preflight_args[@]}"

env -i "${common_env[@]}" "CUDA_VISIBLE_DEVICES=$dense_gpu" \
  "$python" scripts/evaluate_135m_pair.py \
  --manifest "$PAIR_MANIFEST" --profile "$PROFILE_PATH" \
  --evidence-root "$evidence_root" --arm dense &
dense_pid=$!
env -i "${common_env[@]}" "CUDA_VISIBLE_DEVICES=$split90_gpu" \
  "$python" scripts/evaluate_135m_pair.py \
  --manifest "$PAIR_MANIFEST" --profile "$PROFILE_PATH" \
  --evidence-root "$evidence_root" --arm split90 &
split_pid=$!

terminate_pair() {
  kill "$dense_pid" "$split_pid" 2>/dev/null || true
}
trap terminate_pair INT TERM

set +e
wait "$dense_pid"
dense_rc=$?
if [ "$dense_rc" -ne 0 ]; then
  kill "$split_pid" 2>/dev/null || true
fi
wait "$split_pid"
split_rc=$?
if [ "$split_rc" -ne 0 ]; then
  kill "$dense_pid" 2>/dev/null || true
fi
set -e

env -i "${common_env[@]}" "$python" scripts/evaluate_135m_pair.py \
  --manifest "$PAIR_MANIFEST" --evidence-root "$evidence_root" --finalize \
  --dense-returncode "$dense_rc" --split-returncode "$split_rc"
