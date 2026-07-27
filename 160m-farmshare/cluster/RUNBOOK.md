# FarmShare runbook — memory-split battery

Status 2026-07-18 00:30: repo synced to scratch (pre-corpus-package state —
resync first), venv built (torch 2.13.0+cu130), tiktoken cache warmed. The
2026-07-17 bring-up session ended with ALL outbound port-22 traffic blocked
by the local network (github.com:22 and FarmShare both time out; ping fine)
— restore an SSH-capable network (e.g. Stanford VPN), then do the human
step below.

One human step is required whenever the SSH control socket has expired
(FarmShare is password + Duo only):

```bash
cd ~/Documents/MemorySplit
bash cluster/connect.sh syz          # password + Duo push; persists 8h
```

Everything below is scriptable once the socket is warm.

## Day 1 — bring-up (after connect)

```bash
bash cluster/sync_push.sh                          # rsync repo -> scratch
ssh -o ControlPath="$HOME/.ssh/cm-%r@%h:%p" ${SUNETID}@rice-04.farmshare.stanford.edu bash -s <<'EOF'
cd /scratch/users/syz/memorysplit
bash cluster/setup_env.sh                          # idempotent (done 2026-07-17)
sbatch cluster/slurm/smoke_gpu.sbatch              # pytest + toy pipeline on L40S (~15 min)
sbatch --export=ALL,BUILD_ARGS="--stage gates" cluster/slurm/data_prep.sbatch
EOF
```

`data_prep --stage gates` builds all three loads (n50k/n200k/n800k) at the
800M-token gate budget into `/scratch/users/syz/memorysplit_data/`.

## Day 2 — gates A-C (after data_prep finishes)

```bash
ssh ... 'cd /scratch/users/syz/memorysplit && \
  /scratch/users/syz/venvs/memorysplit/bin/python scripts/make_manifest.py \
    --stage gates --data-root /scratch/users/syz/memorysplit_data && \
  bash cluster/submit_manifest.sh outputs/manifests/gates.tsv'
```

Gate criteria (spec §7): A — dense pilot iGSM held-out > 90%;
B — dense recall degrades across N (pick top load; escalate once if flat);
C — split pilot lookup parse rate > 95% and recall ON-OFF gap > 30 pts.
Evaluate finished runs with:

```bash
ssh ... '.../python scripts/run_evals.py --run outputs/<run_id> --limit 2000'
```

Then write and commit `docs/superpowers/specs/2026-07-22-preregistration.md`
(margins = max(2 x pooled pilot seed-sigma, 0.5 pt); freeze before any
confirmation run).

## Days 3-12 — battery (schedule option B, 2026-07-18: 1B confirmation)

```bash
# full-budget corpora: 160M sweep (3.2B tokens/load) AND the 1B corpora
# (10B tokens at loads n800k_1b + n4m_1b) — CPU node, several hours each
sbatch --export=ALL,BUILD_ARGS="--stage full" cluster/slurm/data_prep.sbatch
sbatch --export=ALL,BUILD_ARGS="--stage full1b" cluster/slurm/data_prep.sbatch

# 12 sweep runs (~8-15 L40S-h each), 4 concurrent (QOS cap)
python scripts/make_manifest.py --stage sweep --data-root ...
bash cluster/submit_manifest.sh outputs/manifests/sweep.tsv

# gate B at scale: two short dense-1B calibration runs (~20 h each) pick
# the dose that binds at 1B (n800k vs n4m; 1B has ~6x the 160M capacity)
python scripts/make_manifest.py --stage calib1b --data-root ...
bash cluster/submit_manifest.sh outputs/manifests/calib1b.tsv

# 1B confirmation: 2 arms x 2 seeds, ~130-160 h per run -> submit each as
# a dependency chain (Slurm does not requeue TIMEOUT; each link resumes
# from ckpt.pt automatically)
python scripts/make_manifest.py --stage confirm --top-load <calib winner> --data-root ...
while read cfg; do bash cluster/submit_chain.sh "$cfg" 4; done < outputs/manifests/confirm.tsv

# OPTIONAL 410M tier, only if the calendar allows after the 1B runs land
python scripts/make_manifest.py --stage mid410 --top-load <winner> --data-root ...
```

Evals per finished run: `scripts/run_evals.py --run outputs/<run_id>`
(add `--natural` on the final checkpoint). Pull results home and analyze:

```bash
bash cluster/sync_pull.sh
.venv/bin/python scripts/analyze.py --runs-root outputs/cluster --out outputs/analysis
```

## Kill order (schedule pressure)

1. drop the optional 410M add-back (default off); 2. drop one fact level
from the 160M sweep; 3. drop the 1B confirmation to 1 seed-pair and
restore the second pair on the <= $300 RunPod burst (keep whole pairs on
one platform). The 1B top-load paired contrast is protected last.

## Operational gotchas (learned 2026-07-18 staging)

- **wheat-01 is a bad node**: jobs placed there die in ~4 s with exit
  `0:53` and no output file (three data builds in a row). Add
  `--exclude=wheat-01` if it recurs; healthy wheat/barley nodes work.
- **Always `cd` into $FS_REPO_DIR before sbatch**: the templates resolve
  `cluster/config.env` and write logs via `$SLURM_SUBMIT_DIR`.
- stage.sh submits the gate training runs with `afterok:<gates-data-job>`;
  if the data job is requeued/resubmitted, resubmit the gate runs against
  the new job id (afterok on a FAILED job leaves them pending forever —
  cancel with `scancel` and resubmit).
- **Zombie data jobs**: HF datasets' resource tracker can hang the Python
  interpreter at exit AFTER all work is done (job 1647852 sat RUNNING 3h
  post-completion, stalling its afterok chain). scripts/build_corpus.py now
  ends with `os._exit(0)`; if a data job looks stuck, check whether
  report.json already exists before assuming the build itself is slow.
- **Mixture provenance**: corpora carry the BuildCfg in report.json. After
  the 2026-07-19 gate-A remediation (bed .54 / igsm .12 / ded .08, op 2-6),
  every corpus built earlier (old 62/7/5 mixture) is STALE for battery use
  and must be rebuilt after the gate-A retry verdict: n50k, n200k, n800k,
  n800k_1b, n4m_1b.

## Known facts (recon 2026-07-12/13 + this bring-up)

- QOS gpu: 4 concurrent GPU jobs, 32 submitted max, MaxWall 2 days,
  L40S 48GB (oat-01..06, 4 per node). Login rice-04; scratch
  /scratch/users/syz; egress OK from login and compute nodes.
- venv at /scratch/users/syz/venvs/memorysplit — torch 2.13.0+cu130
  installed 2026-07-17 (login node reports cuda False; GPU nodes have
  driver 595.71.05 / CUDA 13.2).
- train_single.sbatch requeues and `--resume auto` continues from ckpt.pt
  (checkpoint every 30 min), so the 2-day wall is safe for all presets.
- FineWeb-Edu streaming must be anonymous (`token=False`, already in
  scripts/build_corpus.py): a stale ambient HF token turns public-repo
  requests into 401s. Do not export HF_TOKEN in the data-prep job.
