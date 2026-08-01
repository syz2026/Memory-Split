#!/bin/bash
# One training run on one L40S. Resumable: rerunning an unchanged config
# picks up its atomic checkpoint.
#
#   MS_CFG=path/to/run.yaml MS_CODE=... MS_ROOT=... sbatch ops/crowding/train.sbatch
#
# Submit in complete seed blocks rather than arm-major waves, so an
# interruption leaves a balanced matrix.

#SBATCH --job-name=crowding-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=47:00:00
#SBATCH --requeue
#SBATCH --output=crowding-train-%j.out

set -euo pipefail

: "${MS_CFG:?set MS_CFG to the run config}"
: "${MS_CODE:?set MS_CODE to the repo}"
: "${MS_ROOT:?set MS_ROOT to the data/output root}"
PY=${MS_PY:-python3}

say() { echo "[$(date -u +%FT%TZ)] $*"; }
say "run on $(hostname), job ${SLURM_JOB_ID:-none}"
say "  config $MS_CFG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

cd "$MS_ROOT"
PYTHONPATH="$MS_CODE" "$PY" "$MS_CODE/scripts/run_train.py" --config "$MS_CFG"
rc=$?
[ $rc -eq 0 ] && say "run COMPLETE" || say "run FAILED rc=$rc"
exit $rc
