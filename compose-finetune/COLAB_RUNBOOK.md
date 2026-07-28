# Colab runbook — finetune dense vs split on two-hop composition

> **Just want to run it?** Use the ready-made notebooks in `notebooks/` (`01_smoke`,
> `02_finetune_dense_n800k`, `03_finetune_split_n800k`) — upload to Colab, set GPU, Run all.
> This file is the annotated reference for what those cells do.


Paste these cells into a Colab notebook (A100 runtime). Run **two notebooks in parallel**
(one `ARM=dense`, one `ARM=split`) for the n800k pair first (value-order, §6.1 of the spec).
All persistent state lives on **Google Drive** so a dropped session resumes.

**Before you start:** the aligned data seed is already confirmed (**1234** — from the n800k
data build's `report.json`) and baked into Cell 2. You only need to (1) upload the
`compose-finetune/` folder and (2) the two n800k snapshots to Drive (see "Weights" at the bottom).

---

## Cell 1 — mount Drive, clone branch, drop in finetune.py + build/eval scripts + configs
```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content
!rm -rf Memory-Split && git clone -q --branch feat/ood-2-hop https://github.com/syz2026/Memory-Split.git
%cd /content/Memory-Split
# finetune.py + build_compose.py + run_compose_eval.py + configs live in your Drive
# (upload the whole compose-finetune/ folder there once).
FT = '/content/drive/MyDrive/ms/compose-finetune'
!cp {FT}/finetune.py .
!mkdir -p scripts && cp {FT}/build_compose.py {FT}/run_compose_eval.py scripts/   # <-- the missing corpus builder + eval
!mkdir -p configs && cp {FT}/configs/compose_dense_ft.yaml {FT}/configs/compose_split_ft.yaml configs/
# 40GB A100? drop the batch (config default 32 is for 80GB). accum auto-compensates -> identical training.
import os
if 'A100' in (os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read()) and '40' in os.popen('nvidia-smi --query-gpu=memory.total --format=csv,noheader').read():
    !sed -i 's/^micro_batch_size:.*/micro_batch_size: 8/' configs/compose_dense_ft.yaml configs/compose_split_ft.yaml
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
!pip -q install tiktoken pyyaml numpy 2>/dev/null; import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
```

> **The two scripts `build_compose.py` and `run_compose_eval.py` are in the
> `compose-finetune/` folder** (upload it to Drive with finetune.py). They rebuild the
> two-hop corpus (aligned dense+split arms) and evaluate it, reusing the repo's own
> `corpusgen`/`train`/`evals`/`organizer` modules — nothing else to install.

## Cell 2 — parameters (SET THESE; differ per notebook)
```python
ARM   = 'dense'          # <-- 'dense' in notebook A, 'split' in notebook B
BASE  = 'n800k'          # do n800k pair first (sharpest); then n50k
ALIGNED_SEED = 1234      # CONFIRMED from the n800k data build (report.json: n_entities=800000,
                         # seed=1234). generate_records is prefix-stable, so --seed 1234 with
                         # 10k entities == the first 10k the base saw. (For n50k base: also 1234.)
SMOKE = True             # True = ~10-min pipeline validation; set False for the real run

DRIVE = '/content/drive/MyDrive/ms'
# Base = the FINAL model-only snapshot of the seed-0 base run. The step number in the
# filename VARIES by load (n800k has more steps than n50k), so glob the highest one.
import os, glob
cands = sorted(glob.glob(f'{DRIVE}/snapshots/{ARM}_{BASE}_step*.pt'))
assert cands, f'no base snapshot at {DRIVE}/snapshots/{ARM}_{BASE}_step*.pt'
INIT_FROM = cands[-1]                                          # highest step = final base weights
OUT_DIR   = f'{DRIVE}/runs/{ARM}_{BASE}_ft'                    # run dir on Drive (resumable)
TRAINER   = 'v2' if ARM == 'split' else 'v1'
print('init from', INIT_FROM)
```

## Cell 3 — build compose ONCE (aligned) + a novel-entity OOD set (shared; skip if present)
```python
import os
ALN = f'{DRIVE}/data/compose_v1'          # aligned corpus (entities the base saw)
NOV = f'{DRIVE}/data/compose_novel'        # novel-entity OOD (different seed -> unseen people)
NENT, TOK, NEV = (500, 3_000_000, 100) if SMOKE else (10000, 600_000_000, 1000)
if not os.path.exists(f'{ALN}/report.json'):
    !python scripts/build_compose.py --out {ALN} --n-entities {NENT} --held-frac 0.2 \
        --total-tokens {TOK} --n-eval {NEV} --seed {ALIGNED_SEED}
if not os.path.exists(f'{NOV}/report.json'):
    !python scripts/build_compose.py --out {NOV} --n-entities {NENT} --held-frac 0.2 \
        --total-tokens {TOK} --n-eval {NEV} --seed {ALIGNED_SEED + 777}   # different seed = novel people
```
*(Note: point the finetune config's `train_bin`/`train_mask` at `{ALN}`; the configs use the
relative `data/compose_v1/...` path, so symlink or edit — see Cell 4.)*

## Cell 4 — finetune (writes ckpts/snapshots to Drive; resumable)
```python
# make the config's relative data path resolve to the Drive-built corpus
!rm -rf data && mkdir -p data && ln -sf {ALN} data/compose_v1
CFG = f'configs/compose_{ARM}_ft.yaml'
# MBS: 8 for a 40GB A100, 16 for 80GB, 4 for 24GB (L4). accum auto-adjusts -> identical training.
MBS = 8
!python finetune.py --config {CFG} --trainer {TRAINER} --init-from {INIT_FROM} \
    --out-dir {OUT_DIR} --run-id {ARM}_{BASE}_ft --micro-batch-size {MBS}
```
*The `--micro-batch-size` flag is the robust OOM fix — it overrides the config's default
(32, sized for an 80GB A100) without editing any file. On a 40GB A100 the config's 32 OOMs
on the `(B,2048,50304)` logits tensor; `MBS=8` fixes it. If it still OOMs, drop to 4 (or add
`--no-compile`).*
*(If the session drops, just re-run this cell — `--resume auto` continues from the Drive ckpt.
For SMOKE, shrink the run by editing `total_tokens` in the config, or just let the small corpus
cap it.)*

## Cell 5 — evaluate: aligned OOD (with the OOD-vs-step curve) + novel-entity OOD
```python
ARMFLAG = '--arm split' if ARM == 'split' else ''
print('=== aligned OOD (P_held) + OOD-vs-step curve ===')
!python scripts/run_compose_eval.py --run {OUT_DIR} --data {ALN} {ARMFLAG}
print('=== novel-entity OOD (unseen people) ===')
!python scripts/run_compose_eval.py --run {OUT_DIR} --data {NOV} {ARMFLAG}
!echo "summary:"; cat {OUT_DIR}/compose_eval/summary.md 2>/dev/null | head -40
```

## Cell 6 — (forgetting guardrail, optional) has finetuning hurt fact recall?
Compare the base vs finetuned on the original bios recall/factQA (needs the seed-0 eval set;
run the seed-0 battery eval on `{INIT_FROM}` vs `{OUT_DIR}/ckpt.pt`). Skip for the first pass.

---

## Recommended order (value-first)
1. **SMOKE=True**, ARM=dense, one notebook → confirm the whole pipeline (~10 min).
2. **SMOKE=False**: notebook A `ARM=dense BASE=n800k`, notebook B `ARM=split BASE=n800k` → run
   in parallel (~30–60 min). Read the OOD-vs-step curves + novel-OOD.
3. Only if warranted: repeat for `BASE=n50k` (change Cell 2; the compose corpora are already
   built and reused).

## Weights (one-time, ~1.2 GB for the n800k pair — off your laptop if possible)
Put the two n800k **model-only** snapshots (the FINAL `snapshots/step*.pt` of each base run)
on Drive as `{DRIVE}/snapshots/dense_n800k_step<NNNNNNN>.pt` and `.../split_n800k_step<NNNNNNN>.pt`
(keep the real step number in the name — Cell 2 globs it). Use the final snapshot, or `ckpt.pt`
if no snapshots were kept.
Ideal path (no laptop): from a machine with cluster access, `rclone`/`gdown` push
`/scratch/users/iriscai/memorysplit/outputs/d160m_{dense,split}_n800k_s0/snapshots/step*.pt`
→ Drive. (Locked out now? A one-time ~1.2 GB via laptop → Drive also works.)

## Interpretation (see the experiment spec §7)
- Compare each finetuned OOD to the from-scratch anchors (dense 91.8 / split 99.2).
- Within-arm finetune-vs-scratch = base-transfer effect; the OOD-vs-step curve gives
  tokens-to-convergence (sample-efficiency) and the keep-best checkpoint.
- Novel-entity OOD: for split (with store) tests procedure transfer to unseen people; for
  dense it's a floor control.
