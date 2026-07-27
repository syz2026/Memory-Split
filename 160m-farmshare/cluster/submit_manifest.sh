#!/usr/bin/env bash
# Submit every config listed in a manifest TSV (one config path per line,
# comments with #) as individual train_single jobs. QOS caps concurrency at
# 4 GPU jobs; MaxSubmitPU=32, so submit at most 32 lines at a time.
# Usage (ON FarmShare, from $FS_REPO_DIR):
#   bash cluster/submit_manifest.sh outputs/manifests/sweep.tsv
set -euo pipefail
MANIFEST=${1:?usage: submit_manifest.sh <manifest.tsv>}
count=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in \#*) continue ;; esac
    sbatch --export=ALL,CONFIG="$line" cluster/slurm/train_single.sbatch
    count=$((count + 1))
done < "$MANIFEST"
echo "submitted $count jobs from $MANIFEST"
