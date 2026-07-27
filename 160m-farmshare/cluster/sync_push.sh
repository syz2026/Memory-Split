#!/usr/bin/env bash
# Push the repo to FarmShare scratch via the DTN (or login host fallback).
# Requires a warm ControlMaster socket: bash cluster/connect.sh <sunetid> [host]
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
# shellcheck source=./config.env
source "$SCRIPT_DIR/config.env"
require_sunet
HOST="${1:-$LOGIN_HOST}"
DEST=$(expand_path "$FS_REPO_DIR")
SOCK=$(cm_socket)
ssh -o ControlPath="$SOCK" "$SUNET_ID@$HOST" "mkdir -p $DEST"
rsync -az --delete \
    --exclude '.git' --exclude '.venv' --exclude 'data' --exclude 'outputs' \
    --exclude '__pycache__' --exclude '.pytest_cache' \
    --exclude 'slurm-*.out' --exclude 'setup_env.log' \
    --exclude 'configs/gen' \
    -e "ssh -o ControlPath=$SOCK" \
    "$REPO_DIR/" "$SUNET_ID@$HOST:$DEST/"
echo "pushed -> $HOST:$DEST"
