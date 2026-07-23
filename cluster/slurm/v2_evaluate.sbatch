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
: "${MS_RELEASE_ROOT:?}"
: "${MS_RELEASE_SHA256:?}"
: "${MS_RUN_MANIFEST_SHA256:?}"
: "${MS_DATASET_SHA256:?}"
: "${MS_RUN_IDS:?}"
: "${MS_SHARED_ROOT:?}"
: "${MS_SHARED_ROOT_PREFIX:?}"
: "${MS_ENV_ROOT:?}"
: "${MS_DATA_ROOT:?}"
: "${MS_DATA_RELATIVE_PATH:?}"
: "${MS_OUT_ROOT:?}"
: "${MS_SEALED_ROOT:?}"
: "${MS_EVALUATOR_ENTRYPOINT:?}"
: "${MS_EVALUATOR_ENTRYPOINT_SHA256:?}"

[[ "$MS_PROVIDER" == "illumina-usfc-prd" ]]
[[ "$MS_SHARED_ROOT_PREFIX" == "/illumina" ]]
[[ "$MS_SHARED_ROOT" == "$MS_SHARED_ROOT_PREFIX" || "$MS_SHARED_ROOT" == "$MS_SHARED_ROOT_PREFIX/"* ]]
[[ "$MS_DATA_RELATIVE_PATH" != /* && "$MS_DATA_RELATIVE_PATH" != *".."* ]]
[[ "$MS_DATA_ROOT" == "$MS_SHARED_ROOT/$MS_DATA_RELATIVE_PATH" ]]
MS_JOB_HOME="$MS_SHARED_ROOT/memorysplit/job-home"
PYTHON="$MS_ENV_ROOT/bin/python"
[[ -x "$PYTHON" ]]
[[ -d "$MS_SEALED_ROOT" && ! -L "$MS_SEALED_ROOT" ]]
mkdir -p "$MS_JOB_HOME"

IFS=: read -r -a run_ids <<< "$MS_RUN_IDS"
[[ "${#run_ids[@]}" -gt 0 ]]
[[ "$MS_RELEASE_ROOT" == /* && -d "$MS_RELEASE_ROOT" && ! -L "$MS_RELEASE_ROOT" ]]
cd "$MS_RELEASE_ROOT"
EVALUATOR_ENTRYPOINT="$MS_EVALUATOR_ENTRYPOINT"
[[ "$EVALUATOR_ENTRYPOINT" == "$MS_RELEASE_ROOT/evals/confirmatory/runner.py" ]]
[[ -f "$EVALUATOR_ENTRYPOINT" && ! -L "$EVALUATOR_ENTRYPOINT" ]] || {
  echo "evaluation preflight failed: explicit runner is unavailable" >&2
  exit 64
}
env -i "PATH=/usr/bin:/bin" "$PYTHON" - \
  "$EVALUATOR_ENTRYPOINT" "$MS_EVALUATOR_ENTRYPOINT_SHA256" <<'PY'
import ast
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
digest = hashlib.sha256()
with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != expected:
    raise SystemExit("evaluation preflight failed: runner hash mismatch")
tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
contract = None
for node in tree.body:
    if not isinstance(node, ast.Assign):
        continue
    if any(
        isinstance(target, ast.Name)
        and target.id == "MSCTL_EVALUATOR_CONTRACT"
        for target in node.targets
    ):
        contract = ast.literal_eval(node.value)
if contract != "memorysplit-confirmatory-evaluator-v1":
    raise SystemExit(
        "evaluation preflight failed: runner contract marker is missing"
    )
PY
evaluator_help="$(
  env -i \
    "HOME=$MS_JOB_HOME" \
    "PATH=/usr/bin:/bin" \
    "PYTHONPATH=$MS_RELEASE_ROOT" \
    "$PYTHON" "$EVALUATOR_ENTRYPOINT" --help
)"
for expected in evaluate --run --sealed-release --device; do
  grep -q -- "$expected" <<< "$evaluator_help" || {
    echo "evaluation preflight failed: runner interface is incomplete" >&2
    exit 64
  }
done

for run_id in "${run_ids[@]}"; do
  [[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]+[a-z0-9]$ ]]
  env -i \
    "HOME=$MS_JOB_HOME" \
    "PATH=/usr/bin:/bin" \
    "PYTHONPATH=$MS_RELEASE_ROOT" \
    "CUDA_VISIBLE_DEVICES=0" \
    "MS_PROVIDER=$MS_PROVIDER" \
    "MS_PROFILE_SHA256=$MS_PROFILE_SHA256" \
    "MS_RELEASE_SHA256=$MS_RELEASE_SHA256" \
    "MS_RUN_MANIFEST_SHA256=$MS_RUN_MANIFEST_SHA256" \
    "MS_DATASET_SHA256=$MS_DATASET_SHA256" \
    "MS_DATA_ROOT=$MS_DATA_ROOT" \
    "MS_OUT_ROOT=$MS_OUT_ROOT" \
    "MS_SEALED_ROOT=$MS_SEALED_ROOT" \
    "$PYTHON" "$EVALUATOR_ENTRYPOINT" evaluate \
      --run "$MS_OUT_ROOT/$run_id" \
      --sealed-release "$MS_SEALED_ROOT" \
      --device cuda
done
