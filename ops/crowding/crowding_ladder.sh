#!/usr/bin/env bash
# The occupancy ladder: does storage saturate before capacity does?
#
#   bash ops/crowding/crowding_ladder.sh          # build + train + evaluate
#   bash ops/crowding/crowding_ladder.sh --plan   # print the plan, submit nothing
#
# WHY THIS AND NOT THE ARM CONTRAST
#
# Memory Split has three premises: facts occupy parameters, masking frees them,
# and the freed parameters go to reasoning. Everything this project has run so
# far failed premise 1 -- the corpora never stored anything, so there was
# nothing to free and the arm contrast was measuring noise around zero. Testing
# premise 3 before premise 1 holds is what wasted the first four generations.
#
# This ladder tests premise 1 directly, and it needs no arms at all: SUP only,
# one seed per rung. If recoverable bits per parameter rises with demand and
# then PLATEAUS, capacity binds and the crowding regime has been reached for
# the first time. If it rises and then FALLS, the model abandons rather than
# compresses, and crowding cannot occur at this scale however clean the
# contrast. Either way the answer arrives in ~64 GPU-hours instead of 1,495.
#
# THE DESIGN INVARIANT
#
# Document count is held fixed across rungs, so tokens, optimizer steps, cosine
# schedule and mask mass are identical and only the unique-entropy ceiling
# moves. Exposures = docs / entities, so both terms of F/C shift together and
# F/C falls faster than the entity count -- see docs/THEORY-CAPACITY.md.
#
#   exposures  entities    demand   C(E)    F/C   regime     3E gain
#         800   249,102     0.325  1.903  0.171   under         5.1%  saturated
#         400   498,204     0.651  1.602  0.406   under        24.8%
#         200   996,408     1.301  1.301  1.000   critical     36.7%
#
# Every rung: 21.3B tokens, 40,695 steps, ~79 GB, ~21.4 h on one L40S.
# The 200-exposure rung is `high-e200`, already built and VERIFY OK, so only
# two corpora need building.
#
# WHAT IT CANNOT DO
#
# No rung above F/C = 1 fits: F/C 1.5 needs 119 GB and F/C 2.0 needs 159 GB
# against a 150 GB scratch budget, and reaching F/C = 1 while also clearing the
# exposure-saturation gate needs 465 GB. So the ladder brackets the critical
# point from below only. A plateau at the top rung is therefore evidence that
# capacity has begun to bind, not proof that it has bound hard.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$REPO/cluster/config.env"
require_sunet

VENV=$(expand_path "$FS_VENV"); PY="$VENV/bin/python"
ROOT=$(expand_path "$FS_SCRATCH")/crowding
CFGS="$ROOT/configs/occupancy"
BED=${MS_BED:-$ROOT/corpora/fineweb-edu.jsonl}
NLL=${MS_NLL:-$ROOT/corpora/nll_table.npy}
MODEL=${MS_MODEL:-d40m_std}
LR=${MS_LR:-1.5e-3}
GRAD_CLIP=${MS_GRAD_CLIP:-10.0}
TOKENS=${MS_TOKENS:-21335900160}
DOCS=199281600
PLAN_ONLY=""
[ "${1:-}" = "--plan" ] && PLAN_ONLY=1

# exposures:entity-count:corpus-name. 200 is the corpus that already exists.
RUNGS=${MS_RUNGS:-"800:249102:occ-e800 400:498204:occ-e400 200:996408:high-e200"}

say() { echo "[$(date -u +%FT%TZ)] $*"; }
say "occupancy ladder, model $MODEL, $TOKENS tokens per rung"
say "rungs: $RUNGS"

if [ -z "$PLAN_ONLY" ]; then
    test -f "$BED" || { say "FATAL: no pinned bed at $BED"; exit 1; }
    test -f "$NLL" || { say "FATAL: no frozen difficulty table at $NLL.
       Build one with ops/crowding/nll_table.py first."; exit 1; }
    mkdir -p "$ROOT"/{corpora,configs,runs,logs} "$CFGS"
fi

# --- 1. one corpus per rung, built and deleted one at a time -----------------
# 79 GB each; three resident would be 237 GB against a 150 GB budget.
declare -A BUILD_JOB
for rung in $RUNGS; do
    IFS=: read -r EXP ENT NAME <<< "$rung"
    C="$ROOT/corpora/$NAME"
    if [ -f "$C/manifest.json" ]; then
        say "  $NAME present, skipping build"
        continue
    fi
    say "  $NAME: $ENT entities x $EXP exposures"
    [ -n "$PLAN_ONLY" ] && continue
    j=$(MS_OUT="$C" MS_CODE="$REPO" MS_PY="$PY" \
        MS_ENTITIES="$ENT" MS_EXPOSURES="$EXP" MS_TOKENS="$TOKENS" \
        MS_BED="$BED" MS_NLL="$NLL" MS_MOD=23 MS_WORKERS=64 \
        sbatch --parsable --cpus-per-task=64 --mem=320G --time=8:00:00 \
        -D "$ROOT/logs" "$REPO/ops/crowding/build.sbatch")
    BUILD_JOB[$NAME]=$j
    say "    build job $j"
done

# --- 2. one SUP run per rung -------------------------------------------------
for rung in $RUNGS; do
    IFS=: read -r EXP ENT NAME <<< "$rung"
    RID="occ_${MODEL}_e${EXP}"
    [ -n "$PLAN_ONLY" ] && { say "  would submit $RID"; continue; }
    "$PY" - "$CFGS/$RID.yaml" "$ROOT" "$NAME" "$ENT" "$RID" "$MODEL" \
           "$TOKENS" "$LR" "$GRAD_CLIP" <<'PY'
import sys, pathlib, yaml
out, root, corpus, ents, rid, model, tokens, lr, clip = sys.argv[1:10]
cfg = {
    "run_id": rid, "stage": "occupancy", "arm": "sup", "model": model,
    "ctx": 1024, "vocab_size": 50304,
    "train_bin": f"{root}/corpora/{corpus}/targets.bin",
    "probe_mask": f"{root}/corpora/{corpus}/factmask.bin",
    "micro_batch_size": 32, "tokens_per_step": 524_288,
    "total_tokens": int(tokens), "max_steps": int(tokens) // 524_288,
    "lr": float(lr), "warmup_steps": 300, "weight_decay": 0.1,
    "grad_clip": float(clip), "seed": 0,
    "n_entities": int(ents), "corpus_seed": 0,
    "igsm_mod": 23, "igsm_op": [1, 4], "igsm_ood_op": [5, 8],
    "out_dir": f"{root}/runs/{rid}",
    "device": "cuda", "compile": True, "log_every": 50,
    "eval_every": 100_000, "snap_frac": 0.10, "ckpt_minutes": 30,
}
p = pathlib.Path(out); p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"  wrote {p}")
PY
    DEP=""
    [ -n "${BUILD_JOB[$NAME]:-}" ] && DEP="--dependency=afterok:${BUILD_JOB[$NAME]}"
    guard_claim_cfg ladder "$CFGS/$RID.yaml"
    t=$(MS_CFG="$CFGS/$RID.yaml" MS_CODE="$REPO" MS_ROOT="$ROOT" MS_PY="$PY" \
        sbatch --parsable $DEP -D "$ROOT/logs" "$REPO/ops/crowding/train.sbatch")
    e=$(MS_RUN="$ROOT/runs/$RID" MS_CODE="$REPO" MS_PY="$PY" \
        sbatch --parsable --dependency=afterok:"$t" -D "$ROOT/logs" \
        "$REPO/ops/crowding/eval.sbatch")
    say "  $RID  train $t  eval $e"
done

[ -n "$PLAN_ONLY" ] && { say "plan only, nothing submitted"; exit 0; }

say "occupancy ladder submitted."
say "When it lands:"
say "  PYTHONPATH=$REPO $PY $REPO/ops/crowding/occupancy.py --runs $ROOT/runs"
say "Delete each corpus once its eval has landed; three will not fit at once."
