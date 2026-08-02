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
ENTITIES=${MS_ENTITIES:-20000}
EXPOSURES=${MS_EXPOSURES:-20}
TOKENS=${MS_TOKENS:-800000000}
LR=${MS_LR:-1.5e-3}
GRAD_CLIP=${MS_GRAD_CLIP:-1.0}
MODEL=${MS_MODEL:-d40m_std}

say() { echo "[$(date -u +%FT%TZ)] $*"; }

mkdir -p "$ROOT"/{corpora,configs,runs,logs}
say "root     $ROOT"
say "moduli   $MODULI"
say "budget   $TOKENS tokens, lr $LR, model $MODEL  (Stage A's settings)"

# --- 1. one corpus per rung --------------------------------------------------
declare -A BUILD_JOB
for MOD in $MODULI; do
    C="$ROOT/corpora/ladder-mod$MOD"
    if [ -f "$C/manifest.json" ]; then
        say "mod $MOD corpus present, skipping build"
        continue
    fi
    # build.sbatch is sized for the 16.4B corpus. A 800M rung needs a fraction
    # of that, and asking for 64 CPUs would make the four rungs queue behind
    # each other instead of building in parallel. CLI flags override #SBATCH.
    j=$(MS_OUT="$C" MS_CODE="$REPO" MS_PY="$PY" \
        MS_ENTITIES="$ENTITIES" MS_EXPOSURES="$EXPOSURES" MS_TOKENS="$TOKENS" \
        MS_MOD="$MOD" MS_BED="${MS_BED:-}" MS_WORKERS="${MS_WORKERS:-16}" \
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
