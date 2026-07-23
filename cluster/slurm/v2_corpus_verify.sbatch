#!/usr/bin/env bash
#SBATCH --job-name=ms-v2-verify
#SBATCH --partition=normal
#SBATCH --array=0-31%8
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=v2-corpus-verify-%A_%a.out
#
# Submit only the required values:
#   sbatch --export=MS_REPO=/shared/memorysplit,MS_PYTHON=/shared/venv/bin/python,MS_SHARED_ROOT=/shared/corpora cluster/slurm/v2_corpus_verify.sbatch
set -euo pipefail

: "${MS_REPO:?submit MS_REPO explicitly}"
: "${MS_PYTHON:?submit MS_PYTHON explicitly}"
: "${MS_SHARED_ROOT:?submit MS_SHARED_ROOT explicitly}"
: "${SLURM_TMPDIR:?cluster must provide node-local SLURM_TMPDIR}"
: "${SLURM_ARRAY_TASK_ID:?must run as an array task}"
: "${SLURM_CPUS_PER_TASK:?Slurm must set SLURM_CPUS_PER_TASK}"
: "${SLURM_JOB_ID:?Slurm must set SLURM_JOB_ID}"

TASK_ID=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
SOURCE="${MS_SHARED_ROOT%/}/fixture-${TASK_ID}"
LOCAL_ROOT="${SLURM_TMPDIR%/}/memorysplit-v2-verify/${SLURM_JOB_ID}/${TASK_ID}"
LOCAL_CORPUS="$LOCAL_ROOT/corpus"
LOCAL_RECEIPT="$LOCAL_ROOT/verified-receipt.json"
VERIFICATION_ROOT="${MS_SHARED_ROOT%/}/verification"
SHARED_TMP=""

cleanup() {
    rm -rf "$LOCAL_ROOT"
    if [[ -n "$SHARED_TMP" ]]; then
        rm -f "$SHARED_TMP"
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

if [[ ! -d "$SOURCE" ]]; then
    echo "shared corpus is missing: $SOURCE" >&2
    exit 1
fi
mkdir -p "$LOCAL_ROOT" "$VERIFICATION_ROOT"
cp -a "$SOURCE" "$LOCAL_CORPUS"
run_python verify --output "$LOCAL_CORPUS" >"$LOCAL_RECEIPT"

SHARED_TMP="${VERIFICATION_ROOT}/.receipt-${TASK_ID}.tmp-${SLURM_JOB_ID}"
FINAL_RECEIPT="${VERIFICATION_ROOT}/receipt-${TASK_ID}.json"
cp "$LOCAL_RECEIPT" "$SHARED_TMP"
if [[ -e "$FINAL_RECEIPT" ]]; then
    cmp "$SHARED_TMP" "$FINAL_RECEIPT"
    rm "$SHARED_TMP"
else
    mv "$SHARED_TMP" "$FINAL_RECEIPT"
fi
SHARED_TMP=""
