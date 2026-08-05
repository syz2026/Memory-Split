#!/usr/bin/env bash
# The decisive run: are facts acquired AT capacity, or only abandoned above it?
#
#   bash ops/crowding/fc1_run.sh            # submit
#   bash ops/crowding/fc1_run.sh --plan     # print, submit nothing
#
# WHY THIS RUN
#
# On the 16.4B corpus the model was shown 1,531,800 facts a hundred times each
# and retrieved none of them: trained entities score -112.90 bits against
# -112.29 for entities that do not exist. That run sits at F/C = 2.0, twice the
# achievable capacity for its exposure count, which is exactly where
# docs/THEORY-CAPACITY.md predicts the model stops trying to store and models
# the marginal distribution of values instead. So the null is consistent with
# abandonment above saturation and says nothing about behaviour at capacity.
#
# `high-e200-op8` fixes that. Same 21.3B tokens and 40,695 steps, but 996,408
# entities at 200 exposures puts demand at 1.301 bits/param against 1.301
# achievable: F/C = 1.000, the critical point. It also widens the reasoning
# band to op 1-8, because at op 1-4 the corrected endpoint sits at 93.8% and
# has no headroom for a treatment to move.
#
# Two outcomes, both publishable:
#
#   facts acquired      premise 1 of Memory Split survives at capacity, the
#                       crowding regime is reachable, and the arm contrast is
#                       finally worth its 187 GPU-hours.
#   facts not acquired  premise 1 fails across the whole accessible range, not
#                       just above saturation, and the hypothesis has no
#                       purchase at this scale whatever the treatment effect.
#
# One run, about 21 hours on one L40S. Nothing else in the programme discriminates
# that cheaply.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$REPO/cluster/config.env"
require_sunet

VENV=$(expand_path "$FS_VENV"); PY="$VENV/bin/python"
ROOT=$(expand_path "$FS_SCRATCH")/crowding
CORPUS=${MS_CORPUS:-$ROOT/corpora/high-e200-op8}
MODEL=${MS_MODEL:-d40m_std}
LR=${MS_LR:-1.5e-3}
GRAD_CLIP=${MS_GRAD_CLIP:-10.0}
ENTITIES=${MS_ENTITIES:-996408}
TOKENS=${MS_TOKENS:-21335900160}
RID=${MS_RID:-fc1_${MODEL}_e200}
PLAN=""; [ "${1:-}" = "--plan" ] && PLAN=1

say() { echo "[$(date -u +%FT%TZ)] $*"; }
say "run      $RID"
say "corpus   $CORPUS"
say "config   $ENTITIES entities x 200 exposures, F/C = 1.000, op band [1,8]"
say "budget   $TOKENS tokens, $((TOKENS / 524288)) steps, ~21 h on one L40S"

if [ -z "$PLAN" ]; then
  test -f "$CORPUS/manifest.json" || { say "FATAL: no corpus at $CORPUS"; exit 1; }
  mkdir -p "$ROOT"/{configs,runs,logs}
fi

CFG="$ROOT/configs/$RID.yaml"
[ -n "$PLAN" ] || "$PY" - "$CFG" "$ROOT" "$CORPUS" "$RID" "$MODEL" "$TOKENS" \
                        "$LR" "$GRAD_CLIP" "$ENTITIES" <<'PY'
import sys, pathlib, yaml
cfg_path, root, corpus, rid, model, tokens, lr, clip, ents = sys.argv[1:10]
cfg = {
    "run_id": rid, "stage": "fc1", "arm": "sup", "model": model,
    "ctx": 1024, "vocab_size": 50304,
    "train_bin": f"{corpus}/targets.bin",
    "probe_mask": f"{corpus}/factmask.bin",
    "micro_batch_size": 32, "tokens_per_step": 524_288,
    "total_tokens": int(tokens), "max_steps": int(tokens) // 524_288,
    "lr": float(lr), "warmup_steps": 300, "weight_decay": 0.1,
    "grad_clip": float(clip), "seed": 0,
    "n_entities": int(ents), "corpus_seed": 0,
    "igsm_mod": 23, "igsm_op": [1, 8], "igsm_ood_op": [9, 12],
    "out_dir": f"{root}/runs/{rid}",
    "device": "cuda", "compile": True, "log_every": 50,
    # Gate 0 must be logged repeatedly, not once. On the previous run
    # eval_every exceeded max_steps, so loss_masked_values appeared a single
    # time at the end and log_diagnostics copied that value into every
    # snapshot summary, which made any gate-0-to-snapshot pairing confounded.
    "eval_every": 2_000,
    "snap_frac": 0.10, "ckpt_minutes": 30,
}
p = pathlib.Path(cfg_path); p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"  wrote {p}")
PY

if [ -n "$PLAN" ]; then
  say "plan only. After the run, the discriminating measurement is:"
  say "  PYTHONPATH=$REPO $PY $REPO/ops/crowding/probe_control.py \\"
  say "      --run $ROOT/runs/$RID --n-entities 200 --device cuda"
  exit 0
fi

# Ceiling first. docs/PREREGISTRATION.md §10a caps campaign fc1 at 40 GPU-h and
# scratch at 150 GB; this refuses the submission rather than reporting it after.
guard_claim_cfg fc1 "$CFG"

T=$(MS_CFG="$CFG" MS_CODE="$REPO" MS_ROOT="$ROOT" MS_PY="$PY" \
    sbatch --parsable -D "$ROOT/logs" "$REPO/ops/crowding/train.sbatch")
E=$(MS_RUN="$ROOT/runs/$RID" MS_CODE="$REPO" MS_PY="$PY" \
    sbatch --parsable --dependency=afterok:"$T" -D "$ROOT/logs" \
    "$REPO/ops/crowding/eval.sbatch")
say "train $T  eval $E"
say ""
say "When it lands, the measurement that decides premise 1:"
say "  PYTHONPATH=$REPO $PY $REPO/ops/crowding/probe_control.py \\"
say "      --run $ROOT/runs/$RID --n-entities 200 --device cuda \\"
say "      --out $ROOT/runs/$RID/probe_control.json"
say ""
say "Read trained_minus_unseen_heldout_phrasing. Above +5 bits per entity and"
say "the facts are addressable at capacity; near zero and premise 1 fails across"
say "the accessible range."
