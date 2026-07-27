#!/bin/bash
# Stage 4 - fetch the verified code package and seed the tokenizer cache.
#
# The S3 prefix is itself the package SHA-256, so the digest is checked against
# that rather than a separately transcribed constant. A specific version-id is
# pinned so bucket versioning cannot hand us a different object.
set -uo pipefail

BUCKET=${MS_S3_BUCKET}
PKG_SHA=caa774a88f8b1649cbf3bd9ee298043a389fbb8c69226e31a82a8df3f0d62f72
PKG_KEY="packages/${PKG_SHA}/memorysplit-135m-reasoning-v3-aws.zip"
PKG_VERSION=hHugM8fGzhH5SLRJ4vGUMAmOzHGkwI7i
DEST=/mnt/nvme/code
ZIP=/mnt/nvme/stage/memorysplit-135m-reasoning-v3-aws.zip

echo "===== STAGE 4: CODE PACKAGE ====="
date -u +%FT%TZ

mountpoint -q /mnt/nvme || { echo "FAIL: /mnt/nvme not mounted"; exit 1; }
mkdir -p "$DEST"

echo
echo "--- download (version-pinned) ---"
aws s3api get-object \
  --bucket "$BUCKET" --key "$PKG_KEY" --version-id "$PKG_VERSION" \
  --region us-east-1 "$ZIP" \
  --query '{Len:ContentLength,VersionId:VersionId}' --output json || {
    echo "FAIL: could not download code package"; exit 1; }

echo
echo "--- integrity ---"
ACTUAL=$(sha256sum "$ZIP" | awk '{print $1}')
echo "expected sha256 : ${PKG_SHA}"
echo "actual   sha256 : ${ACTUAL}"
[ "$ACTUAL" = "$PKG_SHA" ] || { echo "FAIL: package digest mismatch"; exit 1; }
echo "digest OK"

echo
echo "--- unpack ---"
rm -rf "${DEST:?}"/* 2>/dev/null || true
unzip -q "$ZIP" -d "$DEST" || { echo "FAIL: unzip failed"; exit 1; }

# The archive may or may not carry a single top-level directory; resolve the
# real repo root by locating a known marker rather than assuming a layout.
ROOT=$(dirname "$(find "$DEST" -maxdepth 3 -name pytest.ini -print -quit)")
[ -n "$ROOT" ] && [ -d "$ROOT" ] || { echo "FAIL: could not locate repo root in package"; exit 1; }
echo "repo root: $ROOT"
echo "$ROOT" > /mnt/nvme/stage/CODE_ROOT
ls -la "$ROOT" | head -30

echo
echo "--- seed tiktoken cache ---"
# train/tokenizer.py defaults TIKTOKEN_CACHE_DIR to <root>/.tiktoken_cache, but
# .tiktoken_cache is gitignored so it is not inside the package. Warm it here
# while the node still has egress; tokenizer-binding tests fail without it.
export TIKTOKEN_CACHE_DIR="${ROOT}/.tiktoken_cache"
mkdir -p "$TIKTOKEN_CACHE_DIR"
echo "TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR}"

# Use the DLAMI's torch venv so the tokenizer is warmed for the same
# interpreter that will later import the training code.
PYBIN=/opt/pytorch/bin/python
[ -x "$PYBIN" ] || PYBIN=$(command -v python3)
echo "using python: ${PYBIN}"

"$PYBIN" -c "import tiktoken" 2>/dev/null || {
  echo "tiktoken absent in this interpreter - installing"
  "$PYBIN" -m pip install -q tiktoken || { echo "FAIL: could not install tiktoken"; exit 1; }
}

"$PYBIN" -c "
import os, tiktoken
enc = tiktoken.get_encoding('gpt2')
ids = enc.encode('memorysplit tokenizer warm')
print('tiktoken version   :', tiktoken.__version__)
print('gpt2 encoding ok, sample ids:', ids[:8])
print('cache dir contents :', os.listdir(os.environ['TIKTOKEN_CACHE_DIR']))
" || { echo "FAIL: could not warm tiktoken gpt2 cache (needs egress)"; exit 1; }

echo
echo "--- tokenizer binding via the package's own code ---"
# Uses the repo's Tok wrapper, which is what the binding tests exercise.
( cd "$ROOT" && TIKTOKEN_CACHE_DIR="$TIKTOKEN_CACHE_DIR" "$PYBIN" -c "
from train.tokenizer import Tok
t = Tok()
print('Tok.VOCAB_SIZE =', Tok.VOCAB_SIZE)
print('specials       =', Tok.EOT, Tok.DB_START, Tok.DB_RETRIEVE, Tok.DB_END)
" ) || echo "WARN: repo Tok import failed (may need deps installed; cache is warmed regardless)"

echo
echo "cache seeded at: ${TIKTOKEN_CACHE_DIR}"
du -sh "$TIKTOKEN_CACHE_DIR"
echo "STAGE 4 RESULT: PASS"
