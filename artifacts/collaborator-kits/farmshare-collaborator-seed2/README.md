# FarmShare collaborator 2 — seed 2

This kit owns exactly five protected 160M jobs: Dense/Split at `n50k`, and
Dense/Split/Random at `n800k`, all with model seed 2 and data seed 10002.
`assignment.json` is authoritative.
This ZIP remains a frozen relational-v1 launcher. The included
`2026-07-22-chinchilla-multisource-corpus-design.md` is review material only;
do not substitute v2 data, budgets, or manifests for this assignment.

## Unpack and verify

```bash
export PYTHONDONTWRITEBYTECODE=1
sha256sum -c SHA256SUMS
mkdir source
tar -xzf relational-run.tar.gz -C source
python source/scripts/platform_preflight.py \
  --platform local --bundle relational-run.tar.gz --source-root source
python launch_assignment.py
```

All commands are dry-runs unless `--execute` is present.

## FarmShare environment and data

Upload the unzipped kit to FarmShare, then set account-local absolute roots:

```bash
export RELATIONAL_KIT_ROOT="$PWD"
export RELATIONAL_SOURCE_ROOT="$PWD/source"
export RELATIONAL_VENV="/scratch/users/$USER/venvs/memorysplit"
export DATA_ROOT="/scratch/users/$USER/relational-data-seed2"
export OUT_ROOT="/scratch/users/$USER/relational-runs-seed2"
mkdir -p "$DATA_ROOT" "$OUT_ROOT"
```

Create the pinned three-shard FineWeb-Edu snapshot, then build both owned
corpora. The data jobs start only after the snapshot succeeds:

```bash
bed_job=$(sbatch --parsable --export=ALL \
  farmshare_stage_bed.sbatch)
for data_rel in n50k_ds10002 n800k_ds10002; do
  sbatch --dependency="afterok:$bed_job" \
    --export="ALL,DATA_REL=$data_rel" \
    farmshare_build_corpus.sbatch
done
```

After both data jobs complete, verify every source, bundle, bed, and corpus
hash:

```bash
"$RELATIONAL_VENV/bin/python" verify_assignment.py \
  --data-root "$DATA_ROOT" \
  --bed-jsonl \
    "$DATA_ROOT/fineweb-edu-sample10bt-r87f091-first3.jsonl"
```

## Protected submission

Do not submit protected jobs until the coordinator sends both
`farmshare-29m-gate-authorization.json` and its bound
`farmshare-29m-gate-evidence.json`. Preview again, then submit exactly this
assignment:

```bash
"$RELATIONAL_VENV/bin/python" launch_assignment.py
"$RELATIONAL_VENV/bin/python" launch_assignment.py \
  --authorization farmshare-29m-gate-authorization.json \
  --authorization-evidence farmshare-29m-gate-evidence.json \
  --execute
```

The launcher writes `launch-status.json`. Keep all checkpoints, runtime
configs, Slurm logs, eval outputs, corpus manifests, and hashes. Evaluate each
completed run with `source/scripts/run_relational_evals.py`; do not substitute
a failed or missing seed.
