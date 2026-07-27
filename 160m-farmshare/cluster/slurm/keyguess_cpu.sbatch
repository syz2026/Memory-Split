#!/usr/bin/env bash
#SBATCH --job-name=keyguess-seed
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
# One EXP-COPY-01 replication seed bundle on a CPU node (no GPU QOS used).
# Usage (from $FS_REPO_DIR):
#   sbatch --export=ALL,KEYGUESS_SEED=1 cluster/slurm/keyguess_cpu.sbatch
# Requires data/realfacts/popqa_clean.jsonl (shipped in the handoff zip; do
# NOT re-fetch — the frozen snapshot keeps all seeds on identical facts).
# Runs data -> train(a,c) -> eval(A,B,C,D) for the given seed; artifacts land
# in data/keyguess_local/ with _s${KEYGUESS_SEED} suffixes (~3-4 h).
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
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
exec "$VENV/bin/python" -u scripts/run_keyguess_local.py \
    --stage all --device cpu --seed "${KEYGUESS_SEED:?set KEYGUESS_SEED}"
