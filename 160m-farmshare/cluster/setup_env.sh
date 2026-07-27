#!/usr/bin/env bash
# Create/sync the Task-11 Python 3.12 environment from one exact platform
# lock. The uv cache or UV_FIND_LINKS wheelhouse must already be populated:
# installation is deliberately offline and cannot fall back to a network.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
# shellcheck source=./config.env
source "$SCRIPT_DIR/config.env"

if type module >/dev/null 2>&1 || init_modules; then
    module load "$PYTHON_MODULE"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required and must be installed before offline setup." >&2
    exit 1
fi

RUNTIME=$(
    "$PYTHON_BIN" - <<'PY'
import platform
import sys

if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
    raise SystemExit("ERROR: Task 11 requires CPython 3.12 exactly")

system = platform.system()
machine = platform.machine().lower()
if system == "Darwin" and machine in {"arm64", "aarch64"}:
    print("macos-arm64-py312.lock mps")
elif system == "Linux" and machine in {"x86_64", "amd64"}:
    print("linux-x86_64-cuda-py312.lock cu130")
else:
    raise SystemExit(
        f"ERROR: unsupported Task-11 platform: {system}/{machine}"
    )
PY
)
read -r LOCK_NAME TORCH_BACKEND <<<"$RUNTIME"
LOCK="$REPO_ROOT/requirements/$LOCK_NAME"

# Validate exact pins, hashes, and target metadata before touching the venv.
"$PYTHON_BIN" "$REPO_ROOT/scripts/platform_preflight.py" \
    --validate-lock "$LOCK"

VENV="${VENV:-$(expand_path "$FS_VENV")}"
mkdir -p "$(dirname "$VENV")"
if [ ! -x "$VENV/bin/python" ]; then
    uv venv --python "$PYTHON_BIN" --no-python-downloads "$VENV"
fi

uv pip sync \
    --python "$VENV/bin/python" \
    --offline \
    --require-hashes \
    --strict \
    --only-binary :all: \
    --torch-backend "$TORCH_BACKEND" \
    "$LOCK"

"$VENV/bin/python" - <<'PY'
import sys

import numpy
import tiktoken
import torch
import yaml

assert sys.version_info[:2] == (3, 12)
print(
    "env ok:",
    "python",
    sys.version.split()[0],
    "torch",
    torch.__version__,
    "numpy",
    numpy.__version__,
    "cuda",
    torch.cuda.is_available(),
)
PY
echo "venv ready at $VENV from $LOCK_NAME"
