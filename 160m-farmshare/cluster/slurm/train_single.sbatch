#!/usr/bin/env bash
#SBATCH --job-name=memsplit-train
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=47:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-%j.out
# Usage (from $FS_REPO_DIR):
#   sbatch --export=ALL,CONFIG=configs/gen/sweep_dense_n200k_s0.yaml cluster/slurm/train_single.sbatch
# Requeue-safe: scripts/train.py --resume auto continues from ckpt.pt.
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
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) config=$CONFIG"
exec "$VENV/bin/python" scripts/run_train.py --config "$CONFIG" --resume auto
