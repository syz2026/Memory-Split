#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 SOURCE_ROOT OUTPUT WORK_DIR [BUILD-PRODUCTION-OPTIONS...]" >&2
    exit 64
fi

SOURCE_ROOT=$1
OUTPUT=$2
WORK_DIR=$3
shift 3

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}

source_options=(--source-root "$SOURCE_ROOT")
if [[ -n ${MS_V2_RECIPE:-} ]]; then
    source_options+=(--recipe "$MS_V2_RECIPE")
fi
if [[ -n ${MS_V2_SOURCE_MANIFEST:-} ]]; then
    source_options+=(--source-manifest "$MS_V2_SOURCE_MANIFEST")
fi

"$PYTHON" "$REPO_ROOT/scripts/build_parallel_corpus.py" \
    source-production \
    "${source_options[@]}"

exec "$PYTHON" "$REPO_ROOT/scripts/build_parallel_corpus.py" \
    build-production \
    "${source_options[@]}" \
    --output "$OUTPUT" \
    --work-dir "$WORK_DIR" \
    "$@"
