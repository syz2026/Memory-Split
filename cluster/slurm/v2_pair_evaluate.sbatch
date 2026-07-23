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
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"

repo="$SLURM_SUBMIT_DIR"
python="$MS135_VENV/bin/python"
evidence_root="$(dirname "$PAIR_MANIFEST")/../evidence"
mkdir -p "$evidence_root"
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

cd "$repo"
preflight_args=()
if [ "$RUN_MODE" = "protected" ]; then
  : "${PREFLIGHT_PATH:?PREFLIGHT_PATH is required for protected mode}"
  preflight_args=(--preflight "$PREFLIGHT_PATH")
fi
env -i "${common_env[@]}" "$python" scripts/validate_135m_launch.py \
  --manifest "$PAIR_MANIFEST" --profile "$PROFILE_PATH" \
  --mode "$RUN_MODE" "${preflight_args[@]}"

env -i "${common_env[@]}" CUDA_VISIBLE_DEVICES=0 \
  "$python" scripts/evaluate_135m_pair.py \
  --manifest "$PAIR_MANIFEST" --profile "$PROFILE_PATH" \
  --evidence-root "$evidence_root" --arm dense &
dense_pid=$!
env -i "${common_env[@]}" CUDA_VISIBLE_DEVICES=1 \
  "$python" scripts/evaluate_135m_pair.py \
  --manifest "$PAIR_MANIFEST" --profile "$PROFILE_PATH" \
  --evidence-root "$evidence_root" --arm split90 &
split_pid=$!

set +e
wait "$dense_pid"; dense_rc=$?
wait "$split_pid"; split_rc=$?
set -e

env -i "${common_env[@]}" "$python" scripts/evaluate_135m_pair.py \
  --manifest "$PAIR_MANIFEST" --evidence-root "$evidence_root" --finalize \
  --dense-returncode "$dense_rc" --split-returncode "$split_rc"
