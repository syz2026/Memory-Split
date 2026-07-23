#!/usr/bin/env bash
#SBATCH --job-name=ms-v2-corpus
#SBATCH --partition=normal
#SBATCH --array=0-31%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --output=v2-corpus-%A_%a.out
#
# Submit with an explicit allowlist, for example:
#   sbatch --export=MS_REPO=/shared/memorysplit,MS_PYTHON=/shared/venv/bin/python,MS_SHARED_ROOT=/shared/corpora cluster/slurm/v2_corpus_build.sbatch
#
# This archive-free array is the operational smoke template. Production
# catalogs use the same CLI boundary once their source renderer is implemented.
set -euo pipefail

: "${MS_REPO:?submit MS_REPO explicitly}"
: "${MS_PYTHON:?submit MS_PYTHON explicitly}"
: "${MS_SHARED_ROOT:?submit MS_SHARED_ROOT explicitly}"
: "${SLURM_TMPDIR:?cluster must provide node-local SLURM_TMPDIR}"
: "${SLURM_ARRAY_TASK_ID:?must run as an array task}"
: "${SLURM_CPUS_PER_TASK:?Slurm must set SLURM_CPUS_PER_TASK}"
: "${SLURM_JOB_ID:?Slurm must set SLURM_JOB_ID}"

TASK_ID=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
LOCAL_ROOT="${SLURM_TMPDIR%/}/memorysplit-v2-corpus/${SLURM_JOB_ID}/${TASK_ID}"
LOCAL_OUTPUT="$LOCAL_ROOT/corpus"
FINAL_OUTPUT="${MS_SHARED_ROOT%/}/fixture-${TASK_ID}"
SHARED_TMP=""

cleanup() {
    rm -rf "$LOCAL_ROOT"
    if [[ -n "$SHARED_TMP" ]]; then
        rm -rf "$SHARED_TMP"
    fi
}
trap cleanup EXIT INT TERM

run_python() {
    env -i \
        HOME="${HOME:-/tmp}" \
        LANG=C.UTF-8 \
        PATH=/usr/bin:/bin \
        PYTHONPATH="$MS_REPO" \
        SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
        "$MS_PYTHON" -u "$MS_REPO/scripts/build_parallel_corpus.py" "$@"
}

mkdir -p "$LOCAL_ROOT" "$MS_SHARED_ROOT"
if [[ -e "$FINAL_OUTPUT" ]]; then
    run_python verify --output "$FINAL_OUTPUT"
    exit 0
fi

run_python build-fixture \
    --output "$LOCAL_OUTPUT" \
    --records "${MS_RECORDS:-128}" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --update-tokens "${MS_UPDATE_TOKENS:-64}" \
    --shards 32
run_python verify --output "$LOCAL_OUTPUT"

SHARED_TMP="${MS_SHARED_ROOT%/}/.fixture-${TASK_ID}.tmp-${SLURM_JOB_ID}"
if [[ -e "$SHARED_TMP" ]]; then
    echo "shared temporary output already exists: $SHARED_TMP" >&2
    exit 1
fi
cp -a "$LOCAL_OUTPUT" "$SHARED_TMP"
run_python verify --output "$SHARED_TMP"
mv "$SHARED_TMP" "$FINAL_OUTPUT"
SHARED_TMP=""
