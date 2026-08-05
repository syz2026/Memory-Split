#!/usr/bin/env bash
# Build one fact load, train its three arms across N seeds, evaluate each.
#
#   MS_LOAD=high MS_ENTITIES=1531800 MS_EXPOSURES=100 MS_SEEDS="1 2 3 4" \
#     bash ops/crowding/run_load.sh
#
# Loads are run ONE AT A TIME and the corpus is deleted once its runs are
# evaluated. At 16.4B tokens a corpus is 65.6 GB, so three resident at once
# would be 197 GB -- and the account that reached 261 GB was cancelled
# mid-battery. Sequential keeps the peak at one corpus.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$REPO/cluster/config.env"
require_sunet

VENV=$(expand_path "$FS_VENV"); PY="$VENV/bin/python"
ROOT=$(expand_path "$FS_SCRATCH")/crowding
LOAD=${MS_LOAD:?set MS_LOAD}
ENTITIES=${MS_ENTITIES:?set MS_ENTITIES}
EXPOSURES=${MS_EXPOSURES:?set MS_EXPOSURES}
TOKENS=${MS_TOKENS:-16400000000}
SEEDS=${MS_SEEDS:-"1 2 3 4"}
MODEL=${MS_MODEL:-d40m_std}
LR=${MS_LR:-1.5e-3}
GRAD_CLIP=${MS_GRAD_CLIP:-10.0}
BED=${MS_BED:-$ROOT/corpora/fineweb-edu.jsonl}
NLL=${MS_NLL:-$ROOT/corpora/nll_table.npy}
CORPUS="$ROOT/corpora/$LOAD"
CFGS="$ROOT/configs/$LOAD"

say() { echo "[$(date -u +%FT%TZ)] $*"; }
say "load $LOAD: $ENTITIES entities x $EXPOSURES exposures, $TOKENS tokens"
say "  model $MODEL  seeds [$SEEDS]  grad_clip $GRAD_CLIP"

DEP=""
if [ -f "$CORPUS/manifest.json" ]; then
    say "corpus present, skipping build"
else
    test -f "$BED" || { say "FATAL: no pinned bed at $BED"; exit 1; }
    test -f "$NLL" || { say "FATAL: no frozen RANDPOS difficulty table at $NLL.
       Build one with ops/crowding/nll_table.py before any load. Without it
       the control matches mass but not difficulty, and the primary contrast
       is confounded by loss mass -- preregistration §2."; exit 1; }
    B=$(MS_OUT="$CORPUS" MS_CODE="$REPO" MS_PY="$PY" MS_ENTITIES="$ENTITIES" \
        MS_EXPOSURES="$EXPOSURES" MS_TOKENS="$TOKENS" MS_BED="$BED" \
        MS_NLL="$NLL" \
        sbatch --parsable -D "$ROOT/logs" "$REPO/ops/crowding/build.sbatch")
    say "  build job $B"
    DEP="--dependency=afterok:$B"
fi

PYTHONPATH="$REPO" "$PY" "$REPO/ops/crowding/gen_configs.py" \
    --out "$CFGS" --model "$MODEL" --total-tokens "$TOKENS" \
    --lr "$LR" --grad-clip "$GRAD_CLIP" --seeds $SEEDS \
    --load "$LOAD=$CORPUS" --entities "$LOAD=$ENTITIES" \
    --corpus-seed 0 --igsm-mod 23 --igsm-op 1 4

"$PY" - "$CFGS" "$ROOT" <<'PY'
import sys, pathlib, yaml
cfgs, root = pathlib.Path(sys.argv[1]), sys.argv[2]
for p in sorted(cfgs.glob("*.yaml")):
    c = yaml.safe_load(p.read_text())
    c["out_dir"] = f"{root}/runs/{c['run_id']}"
    p.write_text(yaml.safe_dump(c, sort_keys=False))
PY

say "submitting $(ls "$CFGS"/*.yaml | wc -l) runs"
for c in "$CFGS"/*.yaml; do
    rid=$(basename "$c" .yaml)
    guard_claim_cfg matrix "$c"
    j=$(MS_CFG="$c" MS_CODE="$REPO" MS_ROOT="$ROOT" MS_PY="$PY" \
        sbatch --parsable $DEP -D "$ROOT/logs" "$REPO/ops/crowding/train.sbatch")
    e=$(MS_RUN="$ROOT/runs/$rid" MS_CODE="$REPO" MS_PY="$PY" \
        sbatch --parsable --dependency=afterok:"$j" -D "$ROOT/logs" \
        "$REPO/ops/crowding/eval.sbatch")
    say "  $rid  train $j  eval $e"
done
say "load $LOAD submitted. Delete $CORPUS only after every eval has landed."
