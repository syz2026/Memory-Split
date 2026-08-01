#!/usr/bin/env bash
# Run ON FarmShare (login node), from $FS_REPO_DIR:
#   bash cluster/setup_env.sh
# Builds the venv on scratch, installs torch cu13x wheels + deps, warms the
# tiktoken GPT-2 BPE cache (login nodes have egress; compute nodes verified
# to have egress too, but warm it here to be safe).
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./config.env
source "$SCRIPT_DIR/config.env"
require_sunet
init_modules
module load "$PYTHON_MODULE"
VENV=$(expand_path "$FS_VENV")
mkdir -p "$(dirname "$VENV")"
if [ ! -x "$VENV/bin/python" ]; then
    python -m venv "$VENV"
fi
"$VENV/bin/pip" install --upgrade pip -q
"$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu130 || \
    "$VENV/bin/pip" install -q torch
# From requirements.txt rather than a hardcoded list, so the cluster cannot
# drift from local. That file pins sympy>=1.13.3: torch's _sympy shim fails
# against sympy 1.8 and silently breaks every trainer test.
"$VENV/bin/pip" install -q -r "$SCRIPT_DIR/../requirements.txt"
"$VENV/bin/python" - <<'PY'
import tiktoken, torch
tiktoken.get_encoding("gpt2")
print("env ok: torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
echo "venv ready at $VENV"
