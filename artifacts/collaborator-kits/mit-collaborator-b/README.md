# MIT collaborator B — three 360M jobs

This kit owns Dense seed 1, Split seed 1, and Split seed 2. It owns the seed-1
and seed-2 corpora. MIT collaborator A owns the shared FineWeb snapshot, the
seed-0 corpus, and the combined six-probe preflight.
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
```

Use collaborator A's reviewed `profile.json`, `relational-run.tar.gz`,
`DATA_ROOT`, and `OUT_ROOT` without changing any bytes. If needed, independently
run the bounded discovery command:

```bash
python source/cluster/mit/probe_cluster.py \
  --out mit-cluster-probe.json \
  --source-root source \
  --bundle relational-run.tar.gz
```

## Shared roots and owned data

```bash
export RELATIONAL_VENV=/absolute/shared/path/relational-venv
export DATA_ROOT=/absolute/shared/path/relational-data
export OUT_ROOT=/absolute/shared/path/relational-runs
```

Wait for collaborator A's verified
`fineweb-edu-sample10bt-r87f091-first3.jsonl`. Inside a scheduler allocation
chosen from the discovery evidence, build and verify the seed-1 and seed-2
corpora:

```bash
"$RELATIONAL_VENV/bin/python" build_owned_corpora.py \
  --data-root "$DATA_ROOT" \
  --bed-jsonl \
    "$DATA_ROOT/fineweb-edu-sample10bt-r87f091-first3.jsonl" \
  --execute
"$RELATIONAL_VENV/bin/python" verify_assignment.py \
  --data-root "$DATA_ROOT" \
  --bed-jsonl \
    "$DATA_ROOT/fineweb-edu-sample10bt-r87f091-first3.jsonl"
```

## Probe, central gate, and full runs

Preview, then submit only this kit's three 200-step probes:

```bash
"$RELATIONAL_VENV/bin/python" launch_assignment.py \
  --profile profile.json --steps 200
"$RELATIONAL_VENV/bin/python" launch_assignment.py \
  --profile profile.json --steps 200 --execute
```

Do not start full runs until collaborator A sends an entirely green
`mit-six-probe-preflight.json`,
`mit-six-probe-authorization.json`, and the unchanged profile. Then submit:

```bash
"$RELATIONAL_VENV/bin/python" launch_assignment.py \
  --profile profile.json \
  --authorization mit-six-probe-authorization.json \
  --authorization-evidence mit-six-probe-preflight.json \
  --execute
```

Evaluate all completed runs and return checkpoints, runtime configs, Slurm
logs, evidence JSON, eval outputs, corpus manifests, profile, bundle, and
SHA-256 files to collaborator A. Never replace a missing or failed seed.
