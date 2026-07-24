# MemorySplit 135M reasoning-v3 on AWS

This path runs the frozen 8,169,455,616-token reasoning-v3 composite as a
separate exploratory N=10 cohort. It does not alter or replace the confirmatory
v2 cohort.

## Cost and safety

The example uses AWS ParallelCluster with `MinCount: 0`, a five-minute idle
scale-down, encrypted storage, IMDSv2, and no credentials in files. As of
2026-07-24, public us-east-1 on-demand prices are approximately:

- `p5.48xlarge`: $55.04/hour (8 H100 GPUs)
- `p5en.48xlarge`: $63.296/hour (8 H200 GPUs)
- `p6-b200.48xlarge`: $113.9328/hour (8 B200 GPUs)

One pair requests two GPUs. Slurm can pack four pairs onto an eight-GPU node.
S3, EFS, the head node, snapshots, and data transfer add separate charges.
Keep GPU minimum capacity at zero and delete the cluster after evidence is
copied out. The example EFS filesystem has `DeletionPolicy: Retain`; delete it
explicitly only after preserving required artifacts.

## 1. AWS prerequisites

Use a private S3 bucket with:

- Block Public Access enabled
- versioning enabled
- default SSE-KMS encryption
- an object-lock or IAM policy that prevents replacement under the corpus
  prefix
- separate read-only corpus and write-only evidence prefixes

The operator identity needs S3 upload access and ParallelCluster create/update
permissions. Cluster nodes should use instance roles. Do not put access keys in
the repository, ZIP, profile, user data, or Slurm exports.

The example head-node role grants read-only access to the frozen corpus and
release prefixes and write access only to the evidence prefix. Replace each
`REPLACE` value with the same bucket/prefix identities used below.

Before creating a cluster, confirm regional P-instance vCPU quota and actual
instance-type offerings. Live quota, capacity, AMI, and GPU canaries cannot be
validated without an attached AWS identity.

```bash
python scripts/check_aws_135m_readiness.py --region us-east-1
```

This check is read-only. Offerings and quota are necessary but do not guarantee
immediate capacity; the measured Slurm canaries remain authoritative.

## 2. Verify and upload frozen corpus objects

Run from the repository that contains `corpus-build/`:

```bash
MANIFEST=cluster/aws/reasoning-v3-corpus-manifest.json
PREFIX=s3://YOUR-BUCKET/corpus/84142597cebd96e041d47c7c22dd4b42285b71a213b01265728042cb1a8f6fbb

# Dry run: hashes every local source and prints argv-only AWS commands.
python -m msctl aws upload-corpus \
  --manifest "$MANIFEST" \
  --s3-uri "$PREFIX" \
  --kms-key-id arn:aws:kms:us-east-1:ACCOUNT:key/KEY-ID

# Explicit mutation.
python -m msctl aws upload-corpus \
  --manifest "$MANIFEST" \
  --s3-uri "$PREFIX" \
  --kms-key-id arn:aws:kms:us-east-1:ACCOUNT:key/KEY-ID \
  --apply
```

Every local object is byte-counted and SHA-256 verified before upload. Existing
objects are accepted only when their immutable `sha256` metadata matches.
Downloads are rehashed locally; S3 ETags are never treated as content hashes.

## 3. Build and transfer the code-only package

Production package creation requires a clean commit:

```bash
python scripts/package_aws_reasoning_v3.py build
python scripts/package_aws_reasoning_v3.py verify \
  artifacts/aws-reasoning-v3/memorysplit-135m-reasoning-v3-aws.zip
aws s3 cp artifacts/aws-reasoning-v3/memorysplit-135m-reasoning-v3-aws.zip \
  s3://YOUR-BUCKET/releases/
```

The ZIP contains code, all 20 frozen configs, tests, pointer, transfer
manifest, profile, and runbook. It deliberately excludes corpus bytes,
credentials, checkpoints, and account identifiers.

## 4. Create or reuse AWS ParallelCluster

Copy and replace every `REPLACE` value in:

`cluster/aws/parallelcluster/memorysplit-v3-p5.example.yaml`

Then validate before creation:

```bash
pcluster validate-cluster-configuration -c memorysplit-v3-p5.yaml
pcluster create-cluster \
  --cluster-name memorysplit-v3 \
  --cluster-configuration memorysplit-v3-p5.yaml
```

The template uses the `gpu` Slurm queue and P5 by default. To use P5en or P6,
change the instance type only after checking quota and availability. Keep the
paired profile's partition equal to the queue name. EFA and placement groups
are disabled because every pair is a single-node job; this avoids an
unnecessary capacity constraint. ParallelCluster 3.15 or newer is recommended.

## 5. Install and stage on the head node

Connect to the head node, then:

```bash
cd /shared
aws s3 cp \
  s3://YOUR-BUCKET/releases/memorysplit-135m-reasoning-v3-aws.zip .
unzip memorysplit-135m-reasoning-v3-aws.zip -d memorysplit
cd memorysplit
sha256sum -c SHA256SUMS

python3 -m venv /shared/memorysplit-venv
/shared/memorysplit-venv/bin/pip install -r requirements.txt
PYTHON=/shared/memorysplit-venv/bin/python

PREFIX=s3://YOUR-BUCKET/corpus/84142597cebd96e041d47c7c22dd4b42285b71a213b01265728042cb1a8f6fbb
"$PYTHON" -m msctl aws stage-corpus \
  --s3-uri "$PREFIX" \
  --destination /shared/memorysplit-dataset \
  --apply
"$PYTHON" -m msctl aws verify-corpus \
  --dataset-root /shared/memorysplit-dataset
```

Staging uses a temporary sibling, verifies the exact closed namespace, hashes
all ten objects and the three logical composite streams, writes a deterministic
stage receipt, then publishes with Linux `renameat2(RENAME_NOREPLACE)`.
Re-running against an identical destination is an idempotent verification;
any divergent existing destination fails closed.

## 6. Instantiate pairs

Start with seed 0:

```bash
"$PYTHON" -m msctl aws instantiate \
  --dataset-root /shared/memorysplit-dataset \
  --profile cluster/profiles/aws-p5-p6.example.json \
  --runtime-root /shared/runtime-v3 \
  --out-root /shared/outputs-v3 \
  --seeds 0
PAIR=/shared/runtime-v3/pairs/pair-s0.json
```

Instantiation is additive and no-replace: an identical seed can be requested
again, while any changed existing runtime file is rejected. After seed 0 passes
every canary, add the other seeds in the same runtime root:

```bash
"$PYTHON" -m msctl aws instantiate \
  --dataset-root /shared/memorysplit-dataset \
  --profile cluster/profiles/aws-p5-p6.example.json \
  --runtime-root /shared/runtime-v3 \
  --out-root /shared/outputs-v3 \
  --seeds 1 2 3 4 5 6 7 8 9
PAIRS=(/shared/runtime-v3/pairs/pair-s{0..9}.json)
```

## 7. Run and freeze site canaries

Commands dry-run unless `--apply` is present:

```bash
for MODE in functional resume throughput; do
  "$PYTHON" -m msctl submit "$PAIR" \
    --profile cluster/profiles/aws-p5-p6.example.json \
    --venv-root /shared/memorysplit-venv \
    --mode "$MODE" --apply
done
```

Wait for Slurm completion and inspect `/shared/runtime-v3/evidence`. Freeze the
measured canaries:

```bash
"$PYTHON" scripts/build_135m_preflight.py \
  --cohort-id memorysplit-exploratory-v3-135m-aws-n10 \
  --profile cluster/profiles/aws-p5-p6.example.json \
  --dataset-receipt-sha256 \
    b1eabb1719f66876ab54cc0791b857ccdbbbddb0ffb8c5986ac2aaa7bf33b80d \
  --functional-evidence \
    /shared/runtime-v3/evidence/d135m_reasoning_v3_s0-functional-train-evidence.json \
  --resume-evidence \
    /shared/runtime-v3/evidence/d135m_reasoning_v3_s0-resume-train-evidence.json \
  --throughput-evidence \
    /shared/runtime-v3/evidence/d135m_reasoning_v3_s0-throughput-train-evidence.json \
  --output /shared/runtime-v3/aws-preflight.json
```

Protected submission is rejected unless all three canaries match the AWS
profile, v3 cohort, and virtual corpus receipt.

## 8. Train, resume, evaluate, and collect

```bash
"$PYTHON" -m msctl submit "${PAIRS[@]}" \
  --profile cluster/profiles/aws-p5-p6.example.json \
  --venv-root /shared/memorysplit-venv \
  --mode protected \
  --preflight /shared/runtime-v3/aws-preflight.json \
  --apply

# Safe after interruption: both arms of each pair must share the exact step
# and cursor.
for PAIR in "${PAIRS[@]}"; do
  "$PYTHON" -m msctl resume "$PAIR" \
    --profile cluster/profiles/aws-p5-p6.example.json \
    --venv-root /shared/memorysplit-venv \
    --preflight /shared/runtime-v3/aws-preflight.json \
    --apply
done

"$PYTHON" -m msctl evaluate "${PAIRS[@]}" \
  --profile cluster/profiles/aws-p5-p6.example.json \
  --venv-root /shared/memorysplit-venv \
  --mode protected \
  --preflight /shared/runtime-v3/aws-preflight.json \
  --apply

"$PYTHON" -m msctl collect "${PAIRS[@]}" \
  --evidence-root /shared/runtime-v3/evidence \
  --output /shared/runtime-v3/collection.json

aws s3 cp /shared/runtime-v3/collection.json \
  s3://YOUR-BUCKET/evidence/YOUR-RUN/
aws s3 cp /shared/runtime-v3/evidence \
  s3://YOUR-BUCKET/evidence/YOUR-RUN/evidence/ --recursive
```

The v3 evaluation is explicitly an operational checkpoint/data-boundary test.
It proves that the terminal checkpoint, segmented base-plus-extension loader,
and target weights work on the GPU. It is not the preregistered v2 scientific
endpoint and must not be reported as confirmatory evidence.

## 9. Run package tests

```bash
"$PYTHON" scripts/generate_aws_reasoning_configs.py
"$PYTHON" -m pytest -q \
  tests/test_aws_reasoning_v3.py \
  tests/test_data.py \
  tests/test_trainer.py
```

Do not launch protected runs when any test, corpus verification, or site
canary fails.
