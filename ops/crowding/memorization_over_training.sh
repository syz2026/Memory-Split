#!/usr/bin/env bash
# Does the storage null hold across training, or only at the final checkpoint?
#
# A reviewer's objection to Section 6: the memorization control is reported at
# one checkpoint, which cannot rule out bindings that formed and then decayed,
# or that were about to appear. The run kept twenty snapshots, so the objection
# is answerable for the cost of a CPU afternoon.
#
# Scores a spread of checkpoints with the same instrument and the same cohorts.
# If the gap sits at zero throughout, "no bindings at the end" becomes "no
# bindings at any point we sampled", which is the claim Section 6 wants.
set -euo pipefail

R=/scratch/users/syz/crowding
RUN=$R/runs/stepladder_d40m_std
OUT=$RUN/memorization_over_training
mkdir -p "$OUT"

# A spread rather than all twenty: early, quarter, half, three-quarter, final.
STEPS=(0001564 0007820 0015640 0023460 0031280)

for s in "${STEPS[@]}"; do
  ckpt=$RUN/snapshots/step${s}.pt
  if [ ! -f "$ckpt" ]; then echo "missing $ckpt, skipping"; continue; fi
  echo "=== step $s ==="
  PYTHONPATH=$R/code OMP_NUM_THREADS=16 \
    /scratch/users/syz/venvs/memorysplit/bin/python \
    $R/code/ops/crowding/memorization_control.py \
      --run "$RUN" --ckpt "$ckpt" \
      --n-entities 200 --exposures 2 --device cpu \
      --out "$OUT/step${s}.json"
done

echo
echo "=== gap across training ==="
/scratch/users/syz/venvs/memorysplit/bin/python - <<'PY'
import glob, json, os
rows = []
for f in sorted(glob.glob("/scratch/users/syz/crowding/runs/stepladder_d40m_std/memorization_over_training/step*.json")):
    d = json.load(open(f))
    step = int("".join(c for c in os.path.basename(f) if c.isdigit()))
    rows.append((step, d["cells"]["trained"]["nats_per_value_token"],
                 d["cells"]["unseen"]["nats_per_value_token"],
                 d["gap_nats_unseen_minus_trained"], d["gap_z"]))
print(f"{'step':>8}{'trained':>10}{'unseen':>10}{'gap':>10}{'z':>8}")
for s, t, u, g, z in rows:
    print(f"{s:>8,}{t:>10.4f}{u:>10.4f}{g:>+10.4f}{z:>+8.2f}")
if rows:
    gaps = [r[3] for r in rows]
    print(f"\nmax |gap| across training: {max(abs(g) for g in gaps):.4f} nats")
    print("verdict:", "no bindings at any sampled checkpoint"
          if max(abs(g) for g in gaps) < 0.10 else
          "a checkpoint shows separation; Section 6 must report the curve")
PY
