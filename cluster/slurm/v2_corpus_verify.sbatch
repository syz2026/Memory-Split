#!/usr/bin/env bash
#SBATCH --job-name=ms-v2-verify
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=v2-corpus-verify-%j.out
#
# Command-line --export overrides the non-inheriting directive. Submit only:
#   sbatch --export=MS_REPO=/shared/memorysplit,MS_PYTHON=/shared/venv/bin/python,MS_SHARED_ROOT=/shared/corpora,MS_BUILD_ID=<sha256> cluster/slurm/v2_corpus_verify.sbatch
set -euo pipefail

: "${MS_REPO:?submit MS_REPO explicitly}"
: "${MS_PYTHON:?submit MS_PYTHON explicitly}"
: "${MS_SHARED_ROOT:?submit MS_SHARED_ROOT explicitly}"
: "${MS_BUILD_ID:?submit the expected corpus build ID explicitly}"
: "${SLURM_CPUS_PER_TASK:?Slurm must set SLURM_CPUS_PER_TASK}"
: "${SLURM_JOB_ID:?Slurm must set SLURM_JOB_ID}"

SOURCE="${MS_SHARED_ROOT%/}/fixture-${MS_BUILD_ID}"
FINAL_RECEIPT="${MS_SHARED_ROOT%/}/verification/receipt-${MS_BUILD_ID}.json"

run_python() {
    env -i \
        HOME="${HOME:-/tmp}" \
        LANG=C.UTF-8 \
        PATH=/usr/bin:/bin \
        PYTHONPATH="$MS_REPO" \
        SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
        "$MS_PYTHON" -u "$MS_REPO/scripts/build_parallel_corpus.py" "$@"
}

run_python publish-verification-receipt \
    --output "$SOURCE" \
    --receipt "$FINAL_RECEIPT" \
    --expected-build-id "$MS_BUILD_ID"
