#!/usr/bin/env bash
#SBATCH --job-name=memsplit-smoke
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
# GPU smoke: unit tests + toy end-to-end pipeline on one L40S.
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
"$VENV/bin/python" -m pytest tests -q
exec "$VENV/bin/python" scripts/smoke_test.py --device cuda
