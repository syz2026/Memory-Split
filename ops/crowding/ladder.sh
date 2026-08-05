#!/usr/bin/env bash
# The difficulty ladder Stage A claimed to run but did not. Run ON FarmShare.
#
#   bash ops/crowding/ladder.sh
#
# Stage A concluded "the endpoint has no power to discriminate anything" from
# two cells that both used MOD=23 and differed only in architecture. The
# modulus is baked into the corpus at build time and build_corpus.py takes a
# single --mod, so a real ladder needs one corpus per rung. This builds them.
#
# Everything except the modulus is held at Stage A's values -- same model, same
# 1,500 steps, same learning rate, same seed -- so a lift is attributable to
# difficulty and nothing else. In particular the rate is NOT taken from the
# concurrent LR probe: changing two things at once would forfeit the contrast.
#
# THE 2026-08-01 RUN DID NOT HOLD THAT INVARIANT. It inherited build.sbatch's
# 70% fact share with only 20,000 x 20 documents behind it, so the fact lane
# realised 3.75% and the bed absorbed 79.65% -- and MS_BED was empty, making
# that bed synthetic word salad. The iGSM lane got 98.4M tokens against Stage
# A's 240M, a 2.4x cut, in the one experiment whose whole point was to hold
# iGSM fixed while the modulus moved. Training loss settled at 2.31 against
# Stage A's 1.76, which is what a noise-dominated corpus looks like. Its
# "NOT LEARNABLE even at mod 5" verdict has three candidate explanations and
# cannot be cited. See docs/AMENDMENT-2026-08-02.md.
#
# The shares below therefore pin iGSM at Stage A's 30% and size the fact lane
# to fill its own share exactly, at >=200 exposures per fact. The bed is
# required to be a pinned JSONL. The corpora are built --sup-only, since a
# difficulty probe never trains a masked arm.
#
# Cost: 4 corpora at ~3.2 GB and a few minutes each, then 4 short GPU runs.
#
# Reading the result, on op=1 against the no-skill baseline:
#   clears at some rung   the operation is learnable and 23 was too hard for
#                         the budget. Rerun the design at a modulus that clears.
#   flat even at mod 5    five possible answers and a 232-bit table. No longer
#                         a difficulty problem: check the step ladder, then the
#                         task presentation.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$REPO/cluster/config.env"
require_sunet

VENV=$(expand_path "$FS_VENV")
PY="$VENV/bin/python"
ROOT=$(expand_path "$FS_SCRATCH")/crowding
CFGS="$ROOT/configs/ladder"

MODULI=${MS_MODULI:-"23 11 7 5"}
# 16,000 x 200 documents at ~74.94 tokens is 239.8M, against the 240.0M that a
# 30% fact share of 800M asks for -- a 0.02% deficit, inside the builder's 1%
# tolerance. Change one of these and the other has to move with it, or the bed
# silently absorbs the difference again.
ENTITIES=${MS_ENTITIES:-16000}
EXPOSURES=${MS_EXPOSURES:-200}
TOKENS=${MS_TOKENS:-800000000}
# iGSM pinned at Stage A's 30% so the ladder differs from Stage A in the
# modulus alone, which is the entire premise of the contrast.
FACT_SHARE=${MS_FACT_SHARE:-0.30}
IGSM_SHARE=${MS_IGSM_SHARE:-0.30}
DED_SHARE=${MS_DED_SHARE:-0.10}
BED_SHARE=${MS_BED_SHARE:-0.30}
LR=${MS_LR:-1.5e-3}
GRAD_CLIP=${MS_GRAD_CLIP:-1.0}
MODEL=${MS_MODEL:-d40m_std}
BED=${MS_BED:-$ROOT/corpora/fineweb-edu.jsonl}
PREFIX=${MS_PREFIX:-ladder}

say() { echo "[$(date -u +%FT%TZ)] $*"; }

mkdir -p "$ROOT"/{corpora,configs,runs,logs}
test -f "$BED" || { say "FATAL: no pinned bed at $BED. The 2026-08-01 ladder ran
   on synthetic word salad for 79.65% of its tokens because MS_BED was empty
   and nothing checked. Fetch it with ops/crowding/fetch_bed.py."; exit 1; }
say "root     $ROOT"
say "moduli   $MODULI"
say "corpus   $ENTITIES entities x $EXPOSURES exposures, $TOKENS tokens"
say "shares   fact $FACT_SHARE / igsm $IGSM_SHARE / ded $DED_SHARE / bed $BED_SHARE"
say "bed      $BED"
say "budget   lr $LR, model $MODEL  (Stage A's settings)"

# --- 1. one corpus per rung --------------------------------------------------
declare -A BUILD_JOB
for MOD in $MODULI; do
    C="$ROOT/corpora/$PREFIX-mod$MOD"
    if [ -f "$C/manifest.json" ]; then
        say "mod $MOD corpus present, skipping build"
        continue
    fi
    # build.sbatch is sized for the 16.4B corpus. A 800M rung needs a fraction
    # of that, and asking for 64 CPUs would make the four rungs queue behind
    # each other instead of building in parallel. CLI flags override #SBATCH.
    j=$(MS_OUT="$C" MS_CODE="$REPO" MS_PY="$PY" \
        MS_ENTITIES="$ENTITIES" MS_EXPOSURES="$EXPOSURES" MS_TOKENS="$TOKENS" \
        MS_MOD="$MOD" MS_BED="$BED" MS_WORKERS="${MS_WORKERS:-16}" \
        MS_FACT_SHARE="$FACT_SHARE" MS_IGSM_SHARE="$IGSM_SHARE" \
        MS_DED_SHARE="$DED_SHARE" MS_BED_SHARE="$BED_SHARE" \
        MS_SUP_ONLY=1 \
        sbatch --parsable --cpus-per-task=16 --mem=48G --time=2:00:00 \
        -D "$ROOT/logs" "$REPO/ops/crowding/build.sbatch")
    BUILD_JOB[$MOD]=$j
    say "  mod $MOD build -> job $j"
done

# --- 2. configs --------------------------------------------------------------
say "generating configs"
PYTHONPATH="$REPO" "$PY" "$REPO/ops/crowding/ladder.py" --mode gen \
    --out "$CFGS" --root "$ROOT" --corpus-root "$ROOT/corpora" \
    --model "$MODEL" --lr "$LR" --grad-clip "$GRAD_CLIP" \
    --n-entities "$ENTITIES" --moduli $MODULI

# --- 3. train, each gated on its own corpus ----------------------------------
say "submitting training"
declare -A TRAIN_JOB
for MOD in $MODULI; do
    c="$CFGS/ladder_mod${MOD}_${MODEL}.yaml"
    [ -f "$c" ] || { say "  MISSING $c"; exit 1; }
    DEP=""
    [ -n "${BUILD_JOB[$MOD]:-}" ] && DEP="--dependency=afterok:${BUILD_JOB[$MOD]}"
    guard_claim_cfg ladder "$c"
    j=$(MS_CFG="$c" MS_CODE="$REPO" MS_ROOT="$ROOT" MS_PY="$PY" \
        sbatch --parsable $DEP -D "$ROOT/logs" "$REPO/ops/crowding/train.sbatch")
    TRAIN_JOB[$MOD]=$j
    say "  mod $MOD train -> job $j"
done

# --- 4. evaluate -------------------------------------------------------------
say "submitting evaluations"
for MOD in $MODULI; do
    rid="ladder_mod${MOD}_${MODEL}"
    e=$(MS_RUN="$ROOT/runs/$rid" MS_CODE="$REPO" MS_PY="$PY" \
        sbatch --parsable --dependency=afterok:"${TRAIN_JOB[$MOD]}" \
        -D "$ROOT/logs" "$REPO/ops/crowding/eval.sbatch")
    say "  $rid eval -> job $e"
done

say "ladder submitted. Watch with:  squeue -u $SUNET_ID"
say "When it lands:"
say "  $PY $REPO/ops/crowding/ladder.py --mode rank --runs $ROOT/runs"
