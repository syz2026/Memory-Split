#!/usr/bin/env bash
# Stage A: the endpoint learnability ladder. Run ON FarmShare from the repo.
#
#   bash ops/crowding/stage_a.sh
#
# Answers the two cheapest questions that can void everything downstream:
# can the endpoint move at all, and does dropping weight sharing cost the
# depth it needs. Roughly 4 GPU-hours.
#
# Stop if nothing clears the empirical majority-class baseline by 5 points.
# iGSM floored three times before, twice at models four times larger than
# d40m, so this is the most likely place for the project to end.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$REPO/cluster/config.env"
require_sunet

VENV=$(expand_path "$FS_VENV")
PY="$VENV/bin/python"
ROOT=$(expand_path "$FS_SCRATCH")/crowding
CORPUS="$ROOT/corpora/pilotA"
CFGS="$ROOT/configs/pilotA"

# Small on purpose: this stage measures whether the endpoint is learnable,
# not whether facts crowd anything. 20k entities x 20 exposures keeps the
# fact lane honest without spending the budget on it.
ENTITIES=${MS_ENTITIES:-20000}
EXPOSURES=${MS_EXPOSURES:-20}
TOKENS=${MS_TOKENS:-800000000}
LR=${MS_LR:-1.5e-3}
GRAD_CLIP=${MS_GRAD_CLIP:-1.0}

say() { echo "[$(date -u +%FT%TZ)] $*"; }

mkdir -p "$ROOT"/{corpora,configs,runs,logs}
say "root      $ROOT"
say "corpus    $ENTITIES entities x $EXPOSURES exposures, $TOKENS tokens"

# --- 1. rehearse locally before spending a GPU -------------------------------
say "rehearsing the pipeline (CPU, ~10s)"
cd "$REPO"
PYTHONPATH="$REPO" "$PY" scripts/smoke_pipeline.py --out "$ROOT/smoke" >/dev/null
say "  PIPELINE OK"

# --- 2. build the corpus (CPU partition, no GPU queue) -----------------------
if [ -f "$CORPUS/manifest.json" ]; then
    say "corpus already built, skipping"
else
    say "submitting corpus build"
    BUILD_JOB=$(MS_OUT="$CORPUS" MS_CODE="$REPO" MS_PY="$PY" \
        MS_ENTITIES="$ENTITIES" MS_EXPOSURES="$EXPOSURES" MS_TOKENS="$TOKENS" \
        MS_BED="${MS_BED:-}" \
        sbatch --parsable -D "$ROOT/logs" "$REPO/ops/crowding/build.sbatch")
    say "  job $BUILD_JOB"
    DEP="--dependency=afterok:$BUILD_JOB"
fi

# --- 3. configs: one dense run per architecture ------------------------------
say "generating configs"
PYTHONPATH="$REPO" "$PY" "$REPO/ops/crowding/pilot.py" --stage A \
    --out "$CFGS" --corpus "$CORPUS" \
    --total-tokens "$TOKENS" --lr "$LR" --grad-clip "$GRAD_CLIP"

# The ladder lives inside the corpus and is separated at evaluation, so both
# runs read the same stream and differ only in n_recurrence.
"$PY" - "$CFGS" "$CORPUS" "$ENTITIES" <<'PY'
import sys, pathlib, yaml
cfgs, corpus, entities = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
for p in sorted(cfgs.glob("*.yaml")):
    c = yaml.safe_load(p.read_text())
    c["train_bin"] = f"{corpus}/targets.bin"
    c["probe_mask"] = f"{corpus}/factmask.bin"
    c["n_entities"] = entities
    c["corpus_seed"] = 0
    c["igsm_mod"] = 23
    c["igsm_op"] = [1, 4]
    c["igsm_ood_op"] = [5, 8]
    c["out_dir"] = str(pathlib.Path(corpus).parent.parent / "runs" / c["run_id"])
    p.write_text(yaml.safe_dump(c, sort_keys=False))
    print("  ", c["run_id"], c["model"])
PY

# --- 4. submit ---------------------------------------------------------------
say "submitting training"
JOBS=()
for c in "$CFGS"/*.yaml; do
    j=$(MS_CFG="$c" MS_CODE="$REPO" MS_ROOT="$ROOT" MS_PY="$PY" \
        sbatch --parsable ${DEP:-} -D "$ROOT/logs" "$REPO/ops/crowding/train.sbatch")
    JOBS+=("$j")
    say "  $(basename "$c" .yaml) -> job $j"
done

# --- 5. evaluate when each finishes ------------------------------------------
say "submitting evaluations"
i=0
for c in "$CFGS"/*.yaml; do
    rid=$(basename "$c" .yaml)
    e=$(MS_RUN="$ROOT/runs/$rid" MS_CODE="$REPO" MS_PY="$PY" \
        sbatch --parsable --dependency=afterok:"${JOBS[$i]}" \
        -D "$ROOT/logs" "$REPO/ops/crowding/eval.sbatch")
    say "  $rid eval -> job $e"
    i=$((i + 1))
done

say "Stage A submitted. Watch with:  squeue -u $SUNET_ID"
say "When it lands:  $PY $REPO/ops/crowding/report_stage_a.py --runs $ROOT/runs"
