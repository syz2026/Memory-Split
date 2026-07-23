#!/usr/bin/env bash
#SBATCH --job-name=ms-v2-evaluate
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --open-mode=append
#SBATCH --output=ms-v2-evaluate-%j.out
#
# `msctl evaluate --apply` verifies a sealed-scoring approval before sbatch.
set -euo pipefail
umask 077

: "${MS_PROVIDER:?}"
: "${MS_PROFILE_SHA256:?}"
: "${MS_RELEASE_SHA256:?}"
: "${MS_RUN_MANIFEST_SHA256:?}"
: "${MS_DATASET_SHA256:?}"
: "${MS_RUN_IDS:?}"
: "${MS_SHARED_ROOT:?}"

[[ "$MS_PROVIDER" == "illumina-usfc-prd" ]]
[[ "$MS_SHARED_ROOT" == /illumina/* ]]
MS_ENV_ROOT="${MS_ENV_ROOT:-$MS_SHARED_ROOT/memorysplit/env}"
MS_DATA_ROOT="${MS_DATA_ROOT:-$MS_SHARED_ROOT/memorysplit/data}"
MS_OUT_ROOT="${MS_OUT_ROOT:-$MS_SHARED_ROOT/memorysplit/runs}"
MS_SEALED_ROOT="${MS_SEALED_ROOT:-$MS_SHARED_ROOT/memorysplit/sealed-eval}"
PYTHON="$MS_ENV_ROOT/bin/python"
[[ -x "$PYTHON" ]]
[[ -d "$MS_SEALED_ROOT" && ! -L "$MS_SEALED_ROOT" ]]

IFS=: read -r -a run_ids <<< "$MS_RUN_IDS"
[[ "${#run_ids[@]}" -gt 0 ]]
cd "${SLURM_SUBMIT_DIR:?}"
for run_id in "${run_ids[@]}"; do
  [[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]+[a-z0-9]$ ]]
  env -i \
    "HOME=$HOME" \
    "PATH=/usr/bin:/bin" \
    "PYTHONPATH=$SLURM_SUBMIT_DIR" \
    "CUDA_VISIBLE_DEVICES=0" \
    "MS_PROVIDER=$MS_PROVIDER" \
    "MS_PROFILE_SHA256=$MS_PROFILE_SHA256" \
    "MS_RELEASE_SHA256=$MS_RELEASE_SHA256" \
    "MS_RUN_MANIFEST_SHA256=$MS_RUN_MANIFEST_SHA256" \
    "MS_DATASET_SHA256=$MS_DATASET_SHA256" \
    "MS_DATA_ROOT=$MS_DATA_ROOT" \
    "MS_OUT_ROOT=$MS_OUT_ROOT" \
    "MS_SEALED_ROOT=$MS_SEALED_ROOT" \
    "$PYTHON" -m evals.confirmatory evaluate \
      --run "$MS_OUT_ROOT/$run_id" \
      --sealed-release "$MS_SEALED_ROOT" \
      --device cuda
done
