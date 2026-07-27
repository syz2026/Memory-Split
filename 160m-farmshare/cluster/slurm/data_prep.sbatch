#!/usr/bin/env bash
#SBATCH --job-name=memsplit-data
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
# Usage (from $FS_REPO_DIR):
#   sbatch --export=ALL,BUILD_ARGS="--stage gates" cluster/slurm/data_prep.sbatch
# Downloads the FineWeb-Edu bed sample (compute nodes have egress per recon)
# and builds corpora into $FS_DATA.
set -euo pipefail
SCRIPT_DIR="$SLURM_SUBMIT_DIR/cluster"
# shellcheck source=../config.env
source "$SCRIPT_DIR/config.env"
require_sunet
init_modules
module load "$PYTHON_MODULE"
VENV=$(expand_path "$FS_VENV")
DATA=$(expand_path "$FS_DATA")
mkdir -p "$DATA"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR"
# per-job HF cache: concurrent builds sharing one cache dir contend on
# datasets' file locks
export HF_HOME="$DATA/hf_cache_$SLURM_JOB_ID"
exec "$VENV/bin/python" -u scripts/build_corpus.py --out-root "$DATA" ${BUILD_ARGS:-}
