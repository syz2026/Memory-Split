#!/usr/bin/env bash
#SBATCH --job-name=memsplit-eval
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
# Evaluate one or more finished runs on a single L40S.
# Usage (from $FS_REPO_DIR):
#   sbatch --export=ALL,RUNS="d160m_dense_n50k_s0_gate d160m_split_n200k_s0_gate",EVAL_ARGS="--limit 1500" \
#       cluster/slurm/eval_runs.sbatch
set -euo pipefail
SCRIPT_DIR="$SLURM_SUBMIT_DIR/cluster"
# shellcheck source=../config.env
source "$SCRIPT_DIR/config.env"
require_sunet
init_modules
module load "$PYTHON_MODULE"
VENV=$(expand_path "$FS_VENV")
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR"
for run in ${RUNS:?export RUNS="run_id [run_id ...]"}; do
    echo "=== evaluating $run"
    "$VENV/bin/python" -u scripts/run_evals.py --run "outputs/$run" ${EVAL_ARGS:-}
done
echo "ALL EVALS DONE"
