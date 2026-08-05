#!/bin/bash
# Build one corpus and verify it. CPU only, so this schedules on the normal
# partition immediately rather than queueing behind the contended gpu one.
#
#   MS_OUT=... MS_CODE=... MS_ENTITIES=... MS_EXPOSURES=... MS_TOKENS=... \
#   MS_BED=... MS_NLL=... sbatch ops/crowding/build.sbatch
#
# The verification is the point. A corpus that is silently short, or whose
# arms are not mass-matched, would waste an entire cohort.
#
# The lane shares below default to 70/12.3/4.3/13.4, which is the combination
# that lets a 1,531,800-entity fact lane at 100 exposures exactly fill its
# share of 16.4B tokens. docs/PREREGISTRATION.md §4 still says 50/30/10/10,
# which is the combination that fits 382,900 entities into 5.737B tokens.
# BOTH ARE INTERNALLY CONSISTENT AND THEY ARE NOT THE SAME EXPERIMENT.
# See docs/AMENDMENT-2026-08-02.md §7: one of them has to be frozen before the
# matrix runs, and this file is not the place that decision gets made.

#SBATCH --job-name=crowding-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --time=12:00:00
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
NLL=${MS_NLL:-}
REHEARSAL=${MS_REHEARSAL:-}
SUP_ONLY=${MS_SUP_ONLY:-}
# The iGSM operation band. Not exposed until 2026-08-04, so every corpus this
# project built used the [1,4] default. With the generation bug fixed, a d40m
# trained on [1,4] solves op<=3 perfectly and scores 93.8% in-band -- the
# endpoint sits at CEILING, and `capacity.responsive()` rejects it from the
# far side. Widening the band puts the difficulty range inside the training
# distribution so the endpoint can land mid-range honestly.
OP_LO=${MS_OP_LO:-1}
OP_HI=${MS_OP_HI:-4}

say() { echo "[$(date -u +%FT%TZ)] $*"; }
say "corpus build on $(hostname), job ${SLURM_JOB_ID:-none}"
say "  out       $MS_OUT"
say "  entities  $MS_ENTITIES x $MS_EXPOSURES exposures"
say "  tokens    $MS_TOKENS"
WORKERS=${MS_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}
say "  mod       $MOD  seed $SEED  op band [$OP_LO,$OP_HI]"
say "  workers   $WORKERS"
# The fact lane is ~98% of the build: 11.5B tokens at ~1,640 docs/s on one
# core is 26 hours, past any sensible wall clock. Documents are pure
# functions of their coordinates, so chunks parallelise and the output is
# byte-identical to a serial build.

BED_ARG=()
if [ -n "$BED" ]; then
  BED_ARG=(--bed-jsonl "$BED")
else
  say "  WARNING: no MS_BED set -- the natural bed will be synthetic filler."
  say "           That is for rehearsal only; the manifest records it."
fi

# The RANDPOS difficulty table. Without it the control matches mass, span
# length and position but NOT difficulty, which preregistration §2 requires
# and which no corpus in this project had until 2026-08-02. The builder
# refuses to proceed without either this or an explicit rehearsal flag.
NLL_ARG=()
if [ -n "$NLL" ]; then
  test -f "$NLL" || { say "FATAL: MS_NLL set but $NLL does not exist"; exit 1; }
  NLL_ARG=(--nll-table "$NLL")
  say "  nll table $NLL"
elif [ -n "$SUP_ONLY" ]; then
  NLL_ARG=(--sup-only)
  say "  sup-only: no masked arm will train on this corpus, so the RANDPOS"
  say "            difficulty match does not apply. Bed and lane-share gates stay on."
elif [ -n "$REHEARSAL" ]; then
  NLL_ARG=(--rehearsal)
  say "  WARNING: MS_REHEARSAL set -- unmatched control, synthetic bed and an"
  say "           underfilled fact lane are all permitted. Not a scientific corpus."
else
  say "FATAL: set MS_NLL to a frozen difficulty table (ops/crowding/nll_table.py),"
  say "       MS_SUP_ONLY=1 for a probe corpus no masked arm will use,"
  say "       or MS_REHEARSAL=1 if this build is a shakedown."
  exit 1
fi

PYTHONPATH="$MS_CODE" "$PY" "$MS_CODE/ops/crowding/build_corpus.py" \
  --out "$MS_OUT" \
  --entities "$MS_ENTITIES" \
  --exposures "$MS_EXPOSURES" \
  --total-tokens "$MS_TOKENS" \
  --seed "$SEED" \
  --mod "$MOD" \
  --op-lo "$OP_LO" \
  --op-hi "$OP_HI" \
  --fact-share "${MS_FACT_SHARE:-0.70}" \
  --igsm-share "${MS_IGSM_SHARE:-0.123}" \
  --deduction-share "${MS_DED_SHARE:-0.043}" \
  --bed-share "${MS_BED_SHARE:-0.134}" \
  --workers "$WORKERS" \
  "${BED_ARG[@]}" "${NLL_ARG[@]}"

rc=$?
if [ $rc -eq 0 ]; then
  say "corpus build COMPLETE"
  say "manifest:"
  cat "$MS_OUT/manifest.json"
else
  say "corpus build FAILED verification"
fi
exit $rc
