#!/usr/bin/env bash
#SBATCH --job-name=ms-v2-seed0
#SBATCH --gres=gpu:a100:7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --open-mode=append
#SBATCH --output=ms-v2-seed0-%j.out
#
# Submitted only through `msctl submit --apply`, whose argv uses
# `sbatch --parsable --export=NONE,<explicit allowlist>`.
set -euo pipefail
umask 077

: "${MS_PROVIDER:?}"
: "${MS_PROFILE_SHA256:?}"
: "${MS_RELEASE_SHA256:?}"
: "${MS_RUN_MANIFEST_SHA256:?}"
: "${MS_DATASET_SHA256:?}"
: "${MS_DENSE_RUN_ID:?}"
: "${MS_DENSE_CONFIG:?}"
: "${MS_DENSE_CONFIG_SHA256:?}"
: "${MS_SPLIT_RUN_ID:?}"
: "${MS_SPLIT_CONFIG:?}"
: "${MS_SPLIT_CONFIG_SHA256:?}"
: "${MS_WORLD_SIZE:?}"
: "${MS_EVAL_GPU:?}"
: "${MS_RESUME:?}"
: "${MS_SHARED_ROOT:?}"

[[ "$MS_PROVIDER" == "illumina-usfc-prd" ]]
[[ "$MS_WORLD_SIZE" == "3" ]]
[[ "$MS_EVAL_GPU" == "6" ]]
[[ "$MS_RESUME" == "0" || "$MS_RESUME" == "1" ]]
[[ "$MS_SHARED_ROOT" == /illumina/* ]]

MS_ENV_ROOT="${MS_ENV_ROOT:-$MS_SHARED_ROOT/memorysplit/env}"
MS_DATA_ROOT="${MS_DATA_ROOT:-$MS_SHARED_ROOT/memorysplit/data}"
MS_OUT_ROOT="${MS_OUT_ROOT:-$MS_SHARED_ROOT/memorysplit/runs}"
PYTHON="$MS_ENV_ROOT/bin/python"
[[ -x "$PYTHON" ]]
cd "${SLURM_SUBMIT_DIR:?}"
[[ -f "$MS_DENSE_CONFIG" && ! -L "$MS_DENSE_CONFIG" ]]
[[ -f "$MS_SPLIT_CONFIG" && ! -L "$MS_SPLIT_CONFIG" ]]
mkdir -p "$MS_OUT_ROOT"

verify_config() {
  env -i "PATH=/usr/bin:/bin" "$PYTHON" - "$1" "$2" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit("config SHA-256 mismatch")
PY
}
verify_config "$MS_DENSE_CONFIG" "$MS_DENSE_CONFIG_SHA256"
verify_config "$MS_SPLIT_CONFIG" "$MS_SPLIT_CONFIG_SHA256"

# Explicit environment allowlist: approval keys, tokens, and ambient shell
# variables never cross into training. GPU 6 stays reserved for independent
# evaluation/checkpoint verification; training is symmetric GPUs 0,1,2 and
# GPUs 3,4,5.
common_env=(
  "HOME=$HOME"
  "PATH=/usr/bin:/bin"
  "PYTHONPATH=$SLURM_SUBMIT_DIR"
  "MS_PROVIDER=$MS_PROVIDER"
  "MS_PROFILE_SHA256=$MS_PROFILE_SHA256"
  "MS_RELEASE_SHA256=$MS_RELEASE_SHA256"
  "MS_RUN_MANIFEST_SHA256=$MS_RUN_MANIFEST_SHA256"
  "MS_DATASET_SHA256=$MS_DATASET_SHA256"
  "MS_DATA_ROOT=$MS_DATA_ROOT"
  "MS_OUT_ROOT=$MS_OUT_ROOT"
)

env -i "${common_env[@]}" CUDA_VISIBLE_DEVICES=0,1,2 \
  "$PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=3 --master_port=29500 \
  scripts/run_train.py --config "$MS_DENSE_CONFIG" --resume auto &
dense_pid=$!

env -i "${common_env[@]}" CUDA_VISIBLE_DEVICES=3,4,5 \
  "$PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=3 --master_port=29510 \
  scripts/run_train.py --config "$MS_SPLIT_CONFIG" --resume auto &
split_pid=$!

set +e
wait "$dense_pid"
dense_status=$?
wait "$split_pid"
split_status=$?
set -e

# GPU 6 is intentionally unused by either training group.
nvidia-smi -i 6 --query-gpu=name,memory.total --format=csv,noheader
[[ "$dense_status" -eq 0 && "$split_status" -eq 0 ]]
