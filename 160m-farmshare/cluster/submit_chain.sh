#!/usr/bin/env bash
# Submit one training config as a chain of dependent Slurm jobs, for runs
# whose wall-clock exceeds FarmShare's 2-day MaxWall (the 1B arms need
# ~130-160 h). Slurm does NOT requeue on TIMEOUT, so each link is a fresh
# job that resumes from ckpt.pt (scripts/run_train.py --resume auto).
# Links that start after training already finished are cheap no-ops.
#
# Usage (ON FarmShare, from $FS_REPO_DIR):
#   bash cluster/submit_chain.sh configs/gen/confirm_d1b_dense_n800k_s0.yaml 4
set -euo pipefail
CONFIG=${1:?usage: submit_chain.sh <config.yaml> [links=4]}
LINKS=${2:-4}
dep=""
for i in $(seq 1 "$LINKS"); do
    if [ -z "$dep" ]; then
        jid=$(sbatch --parsable --export=ALL,CONFIG="$CONFIG" cluster/slurm/train_single.sbatch)
    else
        jid=$(sbatch --parsable --dependency=afterany:"$dep" \
              --export=ALL,CONFIG="$CONFIG" cluster/slurm/train_single.sbatch)
    fi
    echo "link $i/$LINKS: job $jid  ($CONFIG)"
    dep=$jid
done
