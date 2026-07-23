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
# Submitted only through `msctl submit --apply`, whose argv uses Slurm's
# supported `--export=NAME,NAME=value` explicit allowlist form. It never mixes
# variables with ALL or NONE. The payload is narrowed again with `env -i`.
set -euo pipefail
umask 077

: "${MS_PROVIDER:?}"
: "${MS_PROFILE_SHA256:?}"
: "${MS_RELEASE_ROOT:?}"
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
: "${MS_ENV_ROOT:?}"
: "${MS_DATA_ROOT:?}"
: "${MS_OUT_ROOT:?}"
: "${MS_TRAIN_ENTRYPOINT:?}"
: "${MS_TRAIN_ENTRYPOINT_SHA256:?}"

[[ "$MS_PROVIDER" == "illumina-usfc-prd" ]]
[[ "$MS_WORLD_SIZE" == "3" ]]
[[ "$MS_EVAL_GPU" == "6" ]]
[[ "$MS_RESUME" == "0" || "$MS_RESUME" == "1" ]]
[[ "$MS_SHARED_ROOT" == /illumina/* ]]
if [[ "$MS_RESUME" == "1" ]]; then
  : "${MS_DENSE_RESUME_PATH:?}"
  : "${MS_DENSE_RESUME_SHA256:?}"
  : "${MS_SPLIT_RESUME_PATH:?}"
  : "${MS_SPLIT_RESUME_SHA256:?}"
fi

MS_JOB_HOME="$MS_SHARED_ROOT/memorysplit/job-home"
PYTHON="$MS_ENV_ROOT/bin/python"
[[ -x "$PYTHON" ]]
[[ "$MS_RELEASE_ROOT" == /* && -d "$MS_RELEASE_ROOT" && ! -L "$MS_RELEASE_ROOT" ]]
cd "$MS_RELEASE_ROOT"
[[ -f "$MS_DENSE_CONFIG" && ! -L "$MS_DENSE_CONFIG" ]]
[[ -f "$MS_SPLIT_CONFIG" && ! -L "$MS_SPLIT_CONFIG" ]]
mkdir -p "$MS_OUT_ROOT" "$MS_JOB_HOME"

verify_config() {
  env -i "PATH=/usr/bin:/bin" "$PYTHON" - "$1" "$2" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
digest = hashlib.sha256()
with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
actual = digest.hexdigest()
if actual != expected:
    raise SystemExit("config SHA-256 mismatch")
PY
}
verify_config "$MS_DENSE_CONFIG" "$MS_DENSE_CONFIG_SHA256"
verify_config "$MS_SPLIT_CONFIG" "$MS_SPLIT_CONFIG_SHA256"

TRAIN_ENTRYPOINT="$MS_TRAIN_ENTRYPOINT"
[[ "$TRAIN_ENTRYPOINT" == "$MS_RELEASE_ROOT/scripts/run_train.py" ]]
[[ -f "$TRAIN_ENTRYPOINT" && ! -L "$TRAIN_ENTRYPOINT" ]]
verify_config "$TRAIN_ENTRYPOINT" "$MS_TRAIN_ENTRYPOINT_SHA256"
env -i "PATH=/usr/bin:/bin" "$PYTHON" - "$TRAIN_ENTRYPOINT" <<'PY'
import ast
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
contract = None
for node in tree.body:
    if not isinstance(node, ast.Assign):
        continue
    if any(
        isinstance(target, ast.Name) and target.id == "MSCTL_DDP_CONTRACT"
        for target in node.targets
    ):
        contract = ast.literal_eval(node.value)
if contract != "memorysplit-ddp-v1":
    raise SystemExit(
        "training preflight failed: missing memorysplit-ddp-v1 contract"
    )
PY
train_help="$(
  env -i \
    "HOME=$MS_JOB_HOME" \
    "PATH=/usr/bin:/bin" \
    "PYTHONPATH=$MS_RELEASE_ROOT" \
    "$PYTHON" "$TRAIN_ENTRYPOINT" --help
)"
grep -q -- "--resume-path" <<< "$train_help" || {
  echo "training preflight failed: --resume-path is unavailable" >&2
  exit 64
}

if [[ "$MS_RESUME" == "1" ]]; then
  [[ -f "$MS_DENSE_RESUME_PATH" && ! -L "$MS_DENSE_RESUME_PATH" ]]
  [[ -f "$MS_SPLIT_RESUME_PATH" && ! -L "$MS_SPLIT_RESUME_PATH" ]]
  verify_config "$MS_DENSE_RESUME_PATH" "$MS_DENSE_RESUME_SHA256"
  verify_config "$MS_SPLIT_RESUME_PATH" "$MS_SPLIT_RESUME_SHA256"
fi

# Explicit environment allowlist: approval keys, tokens, and ambient shell
# variables never cross into training. GPU 6 stays reserved for independent
# evaluation/checkpoint verification; training is symmetric GPUs 0,1,2 and
# GPUs 3,4,5.
common_env=(
  "HOME=$MS_JOB_HOME"
  "PATH=/usr/bin:/bin"
  "PYTHONPATH=$MS_RELEASE_ROOT"
  "MS_PROVIDER=$MS_PROVIDER"
  "MS_PROFILE_SHA256=$MS_PROFILE_SHA256"
  "MS_RELEASE_SHA256=$MS_RELEASE_SHA256"
  "MS_RUN_MANIFEST_SHA256=$MS_RUN_MANIFEST_SHA256"
  "MS_DATASET_SHA256=$MS_DATASET_SHA256"
  "MS_DATA_ROOT=$MS_DATA_ROOT"
  "MS_OUT_ROOT=$MS_OUT_ROOT"
)

dense_args=("$TRAIN_ENTRYPOINT" --config "$MS_DENSE_CONFIG")
split_args=("$TRAIN_ENTRYPOINT" --config "$MS_SPLIT_CONFIG")
if [[ "$MS_RESUME" == "1" ]]; then
  dense_args+=(--resume-path "$MS_DENSE_RESUME_PATH")
  split_args+=(--resume-path "$MS_SPLIT_RESUME_PATH")
fi

env -i "${common_env[@]}" CUDA_VISIBLE_DEVICES=0,1,2 \
  "$PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=3 --master_port=29500 \
  "${dense_args[@]}" &
dense_pid=$!

env -i "${common_env[@]}" CUDA_VISIBLE_DEVICES=3,4,5 \
  "$PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=3 --master_port=29510 \
  "${split_args[@]}" &
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
