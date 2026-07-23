#!/usr/bin/env bash
#SBATCH --job-name=ms-v2-corpus
#SBATCH --partition=normal
#SBATCH --array=0-31%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=v2-corpus-%A_%a.out
#
# Submit with an explicit allowlist, for example:
#   sbatch --export=MS_REPO=/shared/memorysplit,MS_PYTHON=/shared/venv/bin/python,MS_SHARED_ROOT=/shared/corpora,MS_RUN_NONCE=20260723T0202Z-random cluster/slurm/v2_corpus_build.sbatch
#
# Each array task renders one disjoint ordinal partition. Every task then tries
# the exact-N coordinator; all incomplete attempts exit cleanly and the final
# task deterministically packs and atomically publishes the one fixture corpus.
# Production source renderers remain explicitly unsupported.
set -euo pipefail

: "${MS_REPO:?submit MS_REPO explicitly}"
: "${MS_PYTHON:?submit MS_PYTHON explicitly}"
: "${MS_SHARED_ROOT:?submit MS_SHARED_ROOT explicitly}"
: "${MS_RUN_NONCE:?submit a unique scheduler run nonce explicitly}"
: "${SLURM_ARRAY_JOB_ID:?must run as one array job}"
: "${SLURM_ARRAY_TASK_ID:?must run as an array task}"
: "${SLURM_ARRAY_TASK_COUNT:?Slurm must set SLURM_ARRAY_TASK_COUNT}"
: "${SLURM_CPUS_PER_TASK:?Slurm must set SLURM_CPUS_PER_TASK}"
: "${SLURM_JOB_ID:?Slurm must set SLURM_JOB_ID}"

run_python() {
    env -i \
        HOME="${HOME:-/tmp}" \
        LANG=C.UTF-8 \
        PATH=/usr/bin:/bin \
        PYTHONPATH="$MS_REPO" \
        SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
        "$MS_PYTHON" -u "$MS_REPO/scripts/build_parallel_corpus.py" "$@"
}

COMMON_ARGS=(
    --shared-root "$MS_SHARED_ROOT"
    --scheduler-id "$SLURM_ARRAY_JOB_ID"
    --nonce "$MS_RUN_NONCE"
    --records "${MS_RECORDS:-128}"
    --task-count "$SLURM_ARRAY_TASK_COUNT"
    --update-tokens "${MS_UPDATE_TOKENS:-64}"
    --shards "${MS_SHARDS:-32}"
)
if [[ "${MS_ALLOW_FEWER_SHARDS:-0}" == "1" ]]; then
    COMMON_ARGS+=(--allow-fewer-shards)
fi

run_python render-fixture-task \
    "${COMMON_ARGS[@]}" \
    --task-index "$SLURM_ARRAY_TASK_ID" \
    --workers "$SLURM_CPUS_PER_TASK"

if run_python finalize-fixture-tasks \
    "${COMMON_ARGS[@]}" \
    --allow-incomplete; then
    exit 0
else
    status=$?
fi
if [[ "$status" -eq 75 ]]; then
    exit 0
fi
exit "$status"
