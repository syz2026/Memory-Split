#!/usr/bin/env bash
# Submit eval batteries for every finished run that does not yet have
# evals/summary.json. Safe to run repeatedly (idempotent per run).
#
#   cd /scratch/users/$USER/memorysplit && bash cluster/run_evals_pending.sh
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./config.env
source "$SCRIPT_DIR/config.env"
require_sunet

pending=()
for dir in outputs/d*/; do
    run=$(basename "$dir")
    [ -f "$dir/ckpt.pt" ] || continue
    [ -f "$dir/evals/summary.json" ] && continue
    # skip runs whose training job is still active
    if squeue -u "$SUNET_ID" --noheader -o "%o" 2>/dev/null | grep -q "$run"; then
        continue
    fi
    pending+=("$run")
done

if [ ${#pending[@]} -eq 0 ]; then
    echo "no pending evals"
    exit 0
fi

# batch up to 3 runs per eval job (each full battery ~12-15 min on an L40S)
i=0
batch=()
flush() {
    [ ${#batch[@]} -eq 0 ] && return
    jid=$(sbatch --parsable --exclude=wheat-01 \
        --export=ALL,RUNS="${batch[*]}",EVAL_ARGS="--limit 1500" \
        cluster/slurm/eval_runs.sbatch)
    echo "eval job $jid: ${batch[*]}"
    batch=()
}
for run in "${pending[@]}"; do
    batch+=("$run")
    i=$((i + 1))
    [ $((i % 3)) -eq 0 ] && flush
done
flush
