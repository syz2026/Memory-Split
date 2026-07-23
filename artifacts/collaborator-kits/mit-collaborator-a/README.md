# MIT collaborator A — three 360M jobs

This kit owns Dense seed 0, Split seed 0, and Dense seed 2. It also owns the
shared FineWeb snapshot, the seed-0 corpus, the combined six-probe preflight,
and final artifact synchronization. MIT collaborator B owns the seed-2 corpus
needed by this kit's Dense seed-2 job.
This ZIP is a frozen pilot launcher. The included `Memory-split-design.md`
defines the only current dataset; do not substitute its data, budgets, or
manifests into this historical assignment.

## Unpack and verify

```bash
export PYTHONDONTWRITEBYTECODE=1
sha256sum -c SHA256SUMS
mkdir source
tar -xzf relational-run.tar.gz -C source
python source/scripts/platform_preflight.py \
  --platform local --bundle relational-run.tar.gz --source-root source
```

On the MIT login node, collect bounded discovery evidence:

```bash
python source/cluster/mit/probe_cluster.py \
  --out mit-cluster-probe.json \
  --source-root source \
  --bundle relational-run.tar.gz
```

Create one reviewed `profile.json` from
`source/cluster/mit/profile.example.json`. Both MIT collaborators must use
byte-identical profile and bundle files.

## Shared roots and owned data

Use shared project paths visible to both collaborators:

```bash
export RELATIONAL_VENV=/absolute/shared/path/relational-venv
export DATA_ROOT=/absolute/shared/path/relational-data
export OUT_ROOT=/absolute/shared/path/relational-runs
mkdir -p "$DATA_ROOT" "$OUT_ROOT"
```

Inside a scheduler allocation chosen from the discovery evidence, stage the
bed and build this kit's seed-0 corpus:

```bash
"$RELATIONAL_VENV/bin/python" stage_fineweb.py \
  --out "$DATA_ROOT/fineweb-edu-sample10bt-r87f091-first3.jsonl" \
  --cache-dir "$DATA_ROOT/hf-cache" \
  --execute
"$RELATIONAL_VENV/bin/python" build_owned_corpora.py \
  --data-root "$DATA_ROOT" \
  --bed-jsonl \
    "$DATA_ROOT/fineweb-edu-sample10bt-r87f091-first3.jsonl" \
  --execute
```

Wait until collaborator B's `n1p8m_ds10002` corpus and bed-binding file are
present. Then verify every corpus required by this assignment:

```bash
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

After both collaborators' six probes finish in the shared `OUT_ROOT`, run the
single central authorizer inside a profile-matching one-GPU allocation. It
re-runs the complete preflight against the supplied roots and writes both the
report and root-bound authorization:

```bash
"$RELATIONAL_VENV/bin/python" authorize_mit_full_run.py \
  --profile profile.json \
  --bundle relational-run.tar.gz \
  --source-root source \
  --data-root "$DATA_ROOT" \
  --out-root "$OUT_ROOT" \
  --report-out mit-six-probe-preflight.json \
  --authorization-out mit-six-probe-authorization.json
```

Send the green report, authorization, and unchanged profile to collaborator B.
Only then submit this kit's full resumable runs:

```bash
"$RELATIONAL_VENV/bin/python" launch_assignment.py \
  --profile profile.json \
  --authorization mit-six-probe-authorization.json \
  --authorization-evidence mit-six-probe-preflight.json \
  --execute
```

Evaluate all completed runs and retain checkpoints, runtime configs, Slurm
logs, evidence JSON, eval outputs, corpus manifests, profile, bundle, and
SHA-256 files. Never replace a missing or failed seed.
