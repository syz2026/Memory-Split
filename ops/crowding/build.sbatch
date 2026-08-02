#!/bin/bash
# Build one corpus and verify it. CPU only, so this schedules on the normal
# partition immediately rather than queueing behind the contended gpu one.
#
#   MS_OUT=... MS_CODE=... MS_ENTITIES=... MS_EXPOSURES=... MS_TOKENS=... \
#   sbatch ops/crowding/build.sbatch
#
# The verification is the point. A corpus that is silently short, or whose
# arms are not mass-matched, would waste an entire cohort.

#SBATCH --job-name=crowding-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --output=crowding-build-%j.out

set -euo pipefail

: "${MS_OUT:?set MS_OUT to the corpus output root}"
: "${MS_CODE:?set MS_CODE to the repo}"
: "${MS_ENTITIES:?set MS_ENTITIES}"
: "${MS_EXPOSURES:?set MS_EXPOSURES}"
: "${MS_TOKENS:?set MS_TOKENS}"
PY=${MS_PY:-python3}
SEED=${MS_SEED:-0}
MOD=${MS_MOD:-23}
BED=${MS_BED:-}

say() { echo "[$(date -u +%FT%TZ)] $*"; }
say "corpus build on $(hostname), job ${SLURM_JOB_ID:-none}"
say "  out       $MS_OUT"
say "  entities  $MS_ENTITIES x $MS_EXPOSURES exposures"
say "  tokens    $MS_TOKENS"
say "  mod       $MOD  seed $SEED"

BED_ARG=()
if [ -n "$BED" ]; then
  BED_ARG=(--bed-jsonl "$BED")
else
  say "  WARNING: no MS_BED set -- the natural bed will be synthetic filler."
  say "           That is for rehearsal only; the manifest records it."
fi

PYTHONPATH="$MS_CODE" "$PY" "$MS_CODE/ops/crowding/build_corpus.py" \
  --out "$MS_OUT" \
  --entities "$MS_ENTITIES" \
  --exposures "$MS_EXPOSURES" \
  --total-tokens "$MS_TOKENS" \
  --seed "$SEED" \
  --mod "$MOD" \
  --fact-share "${MS_FACT_SHARE:-0.70}" \
  --igsm-share "${MS_IGSM_SHARE:-0.123}" \
  --deduction-share "${MS_DED_SHARE:-0.043}" \
  --bed-share "${MS_BED_SHARE:-0.134}" \
  "${BED_ARG[@]}"

rc=$?
if [ $rc -eq 0 ]; then
  say "corpus build COMPLETE"
  say "manifest:"
  cat "$MS_OUT/manifest.json"
else
  say "corpus build FAILED verification"
fi
exit $rc
