#!/usr/bin/env bash
# Submit the 1B confirmation after applying the PREREGISTERED calib rule
# (docs/superpowers/specs/2026-07-20-preregistration.md): top load = the
# calib1b load with LOWER dense closed-book recall; tie -> n4m.
#
#   cd /scratch/users/$USER/memorysplit && bash cluster/run_confirm.sh n4m
#
# Submits 2 arms x 2 seeds as 4-link dependency chains (runs outlast the
# 2-day MaxWall; each link resumes from ckpt.pt).
set -euo pipefail
TOP=${1:?usage: run_confirm.sh <n800k|n4m>  (per the preregistered calib rule)}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./config.env
source "$SCRIPT_DIR/config.env"
require_sunet
DATA=$(expand_path "$FS_DATA")
VENV=$(expand_path "$FS_VENV")

PYTHONPATH="$PWD" "$VENV/bin/python" scripts/make_manifest.py --stage confirm \
    --top-load "$TOP" --data-root "$DATA"
while IFS= read -r cfg; do
    [ -z "$cfg" ] && continue
    bash cluster/submit_chain.sh "$cfg" 4
done < outputs/manifests/confirm.tsv
squeue -u "$SUNET_ID" -o "%.9i %.13j %.2t %.20E"
