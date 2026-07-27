#!/usr/bin/env bash
# Pull run outputs (logs, evals, reports, snapshots metadata — NOT giant ckpts)
# from FarmShare scratch into local outputs/cluster/.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
# shellcheck source=./config.env
source "$SCRIPT_DIR/config.env"
require_sunet
HOST="${1:-$LOGIN_HOST}"
SRC=$(expand_path "$FS_REPO_DIR")
SOCK=$(cm_socket)
mkdir -p "$REPO_DIR/outputs/cluster"
rsync -az \
    --include '*/' \
    --include 'log.jsonl' --include 'config.yaml' --include '*.json' \
    --include 'evals/**' --include 'report*' --include '*.png' \
    --exclude '*' \
    -e "ssh -o ControlPath=$SOCK" \
    "$SUNET_ID@$HOST:$SRC/outputs/" "$REPO_DIR/outputs/cluster/"
echo "pulled <- $HOST:$SRC/outputs"
