# MemorySplit 135M N=10 Slurm runbook

This release runs ten fixed Dense/Split90 pairs on the complete
`memorysplit-parallel-corpus-v2` stream. Each operator owns both arms for two
seeds. A pair always occupies one two-GPU Slurm allocation: Dense uses GPU 0
and Split90 uses GPU 1.

## Frozen ownership

| Role | Site | Seeds |
| --- | --- | --- |
| `farmshare-lead` | FarmShare | 0, 5 |
| `farmshare-collaborator-1` | FarmShare | 1, 6 |
| `farmshare-collaborator-2` | FarmShare | 2, 7 |
| `mit-collaborator-a` | MIT Slurm | 3, 8 |
| `mit-collaborator-b` | MIT Slurm | 4, 9 |

Do not split a seed across people or sites. Do not add a replacement seed,
stop from outcomes, alter checkpoints, or substitute a partial/fixture corpus.
The fixed horizon is 7,120,879,616 targets (13,582 updates of 524,288 targets)
with checkpoints at updates 1,358, 3,396, 6,791, 10,187, and 13,582.

## 1. Verify the release

From the directory containing all five ZIPs and the outer `SHA256SUMS`:

```bash
sha256sum -c SHA256SUMS
python scripts/verify_135m_slurm_releases.py \
  --release-dir artifacts/135m-slurm-releases
```

Extract only your ZIP. Verify its internal `SHA256SUMS` before installing
dependencies. Never add credentials, corpus bytes, checkpoints, or outputs to
the extracted source tree.

## 2. Bind a complete dataset mirror

The checked-in dataset pointer is intentionally unfrozen until the production
Task-4 receipt is supplied. Do not copy hashes out of an untrusted mirror and
treat them as approval: compare the verifier output with the independently
delivered Task-4 receipt identity.

Verify the sharded Task-4 publication:

```bash
python scripts/bridge_135m_dataset.py verify-source \
  /trusted/task4-publication \
  --expected-source-receipt-sha256 TASK4_RECEIPT_SHA256 \
  --expected-ordered-sha256 TASK4_ORDERED_STREAM_SHA256
```

Convert its verified, ordered shards into the flat layout consumed by the 135M
configs. Publication is atomic and refuses an existing destination:

```bash
python scripts/bridge_135m_dataset.py materialize \
  /trusted/task4-publication /site/scratch/memorysplit-v2 \
  --expected-source-receipt-sha256 TASK4_RECEIPT_SHA256 \
  --expected-ordered-sha256 TASK4_ORDERED_STREAM_SHA256
```

Freeze a new pointer only after both the source publication and converted
bytes verify. The checked-in template is never replaced:

```bash
python scripts/bridge_135m_dataset.py freeze-pointer \
  /trusted/task4-publication /site/scratch/memorysplit-v2 \
  /site/path/DATASET-POINTER-SLURM-135M.bound.json \
  --expected-source-receipt-sha256 TASK4_RECEIPT_SHA256 \
  --expected-ordered-sha256 TASK4_ORDERED_STREAM_SHA256
```

Each site may use a different filesystem path, but all bound pointers and
mirrors must produce the same source receipt, bridge receipt, ordered stream,
packed targets, Dense sidecar, Split90 sidecar, lanes, and recipe identity.

Verify an existing flat mirror:

```bash
python scripts/stage_135m_dataset.py /site/path/memorysplit-v2 \
  --pointer /site/path/DATASET-POINTER-SLURM-135M.bound.json
```

Stage a new mirror without replacing anything:

```bash
python scripts/stage_135m_dataset.py /trusted/source/memorysplit-v2 \
  --destination /site/scratch/memorysplit-v2 \
  --pointer /site/path/DATASET-POINTER-SLURM-135M.bound.json
```

Any receipt mismatch, wrong lane quota, misaligned sidecar, noncanonical
Task-4 publication, symlink, extra file, or existing destination aborts.

## 3. Bind the Slurm profile

FarmShare operators use `cluster/profiles/farmshare-l40s.json`. MIT operators
first run the bounded discovery probe:

```bash
python cluster/mit/probe_cluster.py --out mit-cluster-probe.json
```

Copy the matching operator example in `cluster/profiles/`, then fill only the
discovered partition, account, QoS, GRES, GPU regex, CPU, memory, and wall-time
fields. Profiles must request exactly two GPUs and use
`${MS135_VENV}/bin/python`; unsafe paths and shell metacharacters are rejected.

## 4. Instantiate immutable pair manifests

Set site-local absolute roots, then instantiate your role:

```bash
python -m msctl runs instantiate ROLE \
  --dataset-root /site/scratch/memorysplit-v2 \
  --pointer /site/path/DATASET-POINTER-SLURM-135M.bound.json \
  --source-lock configs/reasoning-dataset-v2.json \
  --profile /site/path/profile.json \
  --runtime-root /site/scratch/ms135-runtime/ROLE \
  --out-root /site/scratch/ms135-outputs/ROLE \
  --repository-root .
```

Instantiation verifies the full dataset before writing anything. It writes two
pair manifests and four runtime configs with hashes. It never replaces an
existing manifest; investigate any collision instead of deleting evidence.

## 5. Run all three site canaries

Commands are dry runs unless `--apply` is present. Use both pair manifests for
normal work; one fixed pair is sufficient to qualify a site profile.

```bash
python -m msctl submit runtime/pairs/pair-sSEED.json \
  --profile /site/path/profile.json --venv-root "$MS135_VENV" \
  --mode functional --apply

python -m msctl submit runtime/pairs/pair-sSEED.json \
  --profile /site/path/profile.json --venv-root "$MS135_VENV" \
  --mode resume --apply

python -m msctl submit runtime/pairs/pair-sSEED.json \
  --profile /site/path/profile.json --venv-root "$MS135_VENV" \
  --mode throughput --apply
```

These modes execute respectively one update, an interrupted/resumed two-update
pair, and 100 updates. The pair fails if either arm fails, records an OOM, uses
an unsupported/different GPU model, lacks a checkpoint, or finishes at the
wrong update. Throughput is scheduling evidence only; it cannot change seeds,
data, endpoints, or interpretation.

Freeze the measured evidence:

```bash
python scripts/build_135m_preflight.py \
  --profile /site/path/profile.json \
  --dataset-receipt-sha256 RECEIPT_SHA256 \
  --functional-evidence runtime/evidence/d135m_full_sSEED-functional-train-evidence.json \
  --resume-evidence runtime/evidence/d135m_full_sSEED-resume-train-evidence.json \
  --throughput-evidence runtime/evidence/d135m_full_sSEED-throughput-train-evidence.json \
  --output /site/path/site-preflight.json
```

## 6. Launch and resume protected pairs

Preview:

```bash
python -m msctl submit runtime/pairs/pair-sA.json runtime/pairs/pair-sB.json \
  --profile /site/path/profile.json --venv-root "$MS135_VENV" \
  --mode protected --preflight /site/path/site-preflight.json
```

Submit only after reviewing the exact argv arrays:

```bash
python -m msctl submit runtime/pairs/pair-sA.json runtime/pairs/pair-sB.json \
  --profile /site/path/profile.json --venv-root "$MS135_VENV" \
  --mode protected --preflight /site/path/site-preflight.json --apply
```

Requeueing uses paired checkpoints only. `msctl resume` rejects a one-arm
checkpoint or unequal update/data cursors.

## 7. Evaluate, inspect, and collect

```bash
python -m msctl evaluate runtime/pairs/pair-sA.json runtime/pairs/pair-sB.json \
  --profile /site/path/profile.json --venv-root "$MS135_VENV" \
  --mode protected --preflight /site/path/site-preflight.json --apply

python -m msctl status runtime/pairs/pair-sA.json runtime/pairs/pair-sB.json \
  --evidence-root runtime/evidence --action evaluate

python -m msctl collect runtime/pairs/pair-sA.json runtime/pairs/pair-sB.json \
  --evidence-root runtime/evidence --output handback/collection.json
```

Return the immutable manifests, profile, bound pointer, preflight receipt,
training/evaluation evidence, collection index, logs, checkpoints, and eval
outputs through the agreed secure channel. Do not place results in a release
ZIP.

## Claim boundary

Real FarmShare/MIT GPU canaries and the production corpus receipt are external
launch gates, not evidence completed by this source release. Ten pairs cannot
guarantee a binary verdict. Report `supports_effect`, `supports_practical_null`,
`inconclusive`, or `invalid` using the frozen preregistration and no other
decision rule.
