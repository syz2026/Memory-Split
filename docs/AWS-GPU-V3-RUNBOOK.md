# AWS GPU v3 P5/P6 runbook

This is the paid-run procedure for
`memorysplit-confirmatory-v3-360m-n10-aws`. It is fail-closed. A package,
Capacity Block, running instance, or approval file is not scientific launch
authorization by itself.

## 0. Non-negotiable gates

Before protected work, record evidence that all of the following are true:

1. The frozen cohort, preregistration, hardware amendment, release, dataset,
   provider selection, environment, sealed evaluation, and run manifests are
   hash-bound.
2. The six 29M diagnostics completed: full-corpus Dense/Split90, no-ARC and
   no-ConceptARC Dense/Split90, and no-refinement Dense/Split90. Review all six
   receipts together; do not drop a failed or inconvenient arm.
3. The selected profile passed hardware, software, simultaneous 4+4, one-step
   training, checkpoint/resume, and NVMe qualification.
4. `protected_launch_allowed` can be changed from false only through the
   separately reviewed gate process. This runbook does not change it.

The 29M diagnostic jobs are paid work. Their owning manifest must first be
rendered in dry-run mode, reviewed as one six-run set, and only then submitted
with its explicit apply control. The v3 360M package does not contain diagnostic
outputs or authorize that submission.

## 1. Identity, Region, and network preflight

Use temporary federation into the approved operator role. Do not create, copy,
or export static credentials. The EC2 workers use a different, dedicated
instance role delivered through instance metadata.

Only `us-east-1` and `us-west-2` are in scope:

```bash
set -euo pipefail
REGION=us-east-1
case "$REGION" in
  us-east-1|us-west-2) ;;
  *) echo "region is outside the approved scope" >&2; exit 2 ;;
esac

# Read-only identity and quota review.
aws sts get-caller-identity
aws service-quotas list-service-quotas \
  --service-code ec2 \
  --region "$REGION" \
  --output json > quota-review.json
aws ec2 describe-instance-type-offerings \
  --location-type region \
  --filters Name=instance-type,Values=p5.48xlarge,p6-b300.48xlarge \
  --region "$REGION" \
  --output json > offering-review.json
```

The launch request must select private subnets, disable public IP assignment,
and attach only the dedicated instance profile. Require VPC endpoints (or
reviewed private egress) for SSM, SSM Messages, EC2 Messages, private ECR API
and Docker, S3, KMS, and STS. Inbound SSH is unnecessary; use Session Manager.
Record subnet, security-group, endpoint, and instance-profile identities in the
launch review.

## 2. Package exactly one provider

Choose one literal profile path and keep it unchanged for all following
commands:

```bash
set -euo pipefail
export PROFILE=cluster/profiles/aws-p6-b300.48xlarge-v3.json
OUT_ROOT=../memorysplit-releases/aws-gpu-v3

# DRY RUN: creates no output.
PACKAGE_PLAN="$(python scripts/package_aws_gpu_handoff.py \
  --profile "$PROFILE" \
  --out-dir "$OUT_ROOT")"
printf '%s\n' "$PACKAGE_PLAN" | python -m json.tool

# APPLY only after profile, release ID, and archive hash review.
PACKAGE_RESULT="$(python scripts/package_aws_gpu_handoff.py \
  --profile "$PROFILE" \
  --out-dir "$OUT_ROOT" \
  --apply)"
printf '%s\n' "$PACKAGE_RESULT" | python -m json.tool
```

Capturing stdout in shell variables keeps the clean source tree unchanged; do
not redirect either packaging report into the repository. Resolve
`RELEASE_DIR` from the reviewed JSON rather than a wildcard. Then verify:

```bash
set -euo pipefail
RELEASE_DIR="$(python -c 'import json,sys; print(json.load(sys.stdin)["release_dir"])' <<<"$PACKAGE_RESULT")"
RELEASE_RECEIPT="$RELEASE_DIR/RELEASE-AWS-GPU-V3.json"

python scripts/verify_aws_gpu_v3_release.py \
  --release "$RELEASE_RECEIPT" \
  --profile "$PROFILE" \
  --source-root . > release-verification.json
python -m json.tool release-verification.json
```

The archive must contain one selected profile, all twenty v3 configs, and no
other provider profile, credentials, output, checkpoint, cache, corpus payload,
symlink, or mutable image reference.

## 3. Resolve and pin Ubuntu 24.04 single-CUDA DLAMI

The AWS public SSM alias is discovery-only. Never put the alias in a launch
request; pin the returned AMI ID.

```bash
set -euo pipefail
REGION="${REGION:?}"
DLAMI_PARAMETER=/aws/service/deeplearning/ami/x86_64/base-with-single-cuda-ubuntu-24.04/latest/ami-id

# Read-only resolution.
AMI_ID="$(aws ssm get-parameter \
  --region "$REGION" \
  --name "$DLAMI_PARAMETER" \
  --query Parameter.Value \
  --output text)"
case "$AMI_ID" in
  ami-*) ;;
  *) echo "public parameter did not return an AMI ID" >&2; exit 2 ;;
esac

# Read-only immutable-image review.
aws ec2 describe-images \
  --region "$REGION" \
  --image-ids "$AMI_ID" \
  --owners amazon \
  --output json > dlami-review.json
python -m json.tool dlami-review.json
sha256sum dlami-review.json
```

Review one available x86_64 Amazon-owned image whose name is “Deep Learning
Base AMI with Single CUDA (Ubuntu 24.04)”. Confirm its driver, kernel, EFA,
OFI-NCCL, and CUDA satisfy the selected profile. Record `AMI_ID` and the
describe-images receipt hash. If the public alias later changes, continue using
the reviewed ID or repeat the entire review and provider-selection process.

## 4. Build and pin the private ECR image

The ECR repository must already exist, be private, have tag immutability
enabled, encryption configured, and deny tag replacement. Repository creation
is outside this runbook.

```bash
set -euo pipefail
ECR_REPOSITORY=REPLACE_WITH_APPROVED_PRIVATE_ECR_REPOSITORY

# DRY RUN: prints exact Docker build and push argv; executes neither.
python scripts/build_aws_gpu_image.py \
  --destination "$ECR_REPOSITORY" > image-plan.json
python -m json.tool image-plan.json

# APPLY only after base digest, context hash, immutable build tag, and
# destination repository are reviewed.
python scripts/build_aws_gpu_image.py \
  --destination "$ECR_REPOSITORY" \
  --apply > image-result.json
```

After the push, use read-only ECR calls to resolve and record the manifest
digest. The only runtime form is the private
`repository@sha256:<manifest-digest>` reference; a tag alone is forbidden.

```bash
IMAGE_TAG="$(python -c 'import json; print(json.load(open("image-result.json"))["destination"].rsplit(":", 1)[1])')"
aws ecr describe-images \
  --region "$REGION" \
  --repository-name REPLACE_WITH_REPOSITORY_NAME \
  --image-ids imageTag="$IMAGE_TAG" \
  --output json > ecr-image-review.json
python -m json.tool ecr-image-review.json
```

Set `CONTAINER_DIGEST` from the reviewed `imageDigest`, and set
`CONTAINER_IMAGE` to the same private repository plus `@CONTAINER_DIGEST`.

## 5. Publish the provider-selection receipt

Use one fixed timestamp for dry-run and apply so the reviewed bytes are the
published bytes:

```bash
set -euo pipefail
CONTAINER_DIGEST=sha256:REPLACE_WITH_REVIEWED_MANIFEST_DIGEST
CONTAINER_IMAGE="${ECR_REPOSITORY}@${CONTAINER_DIGEST}"
SELECTED_AT=REPLACE_WITH_APPROVED_UTC_TIMESTAMP
SELECTION=operator/provider-selection-v3.json

# DRY RUN: validates but does not write the receipt.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  provider select \
  --amendment configs/hardware-amendment-v3.json \
  --region "$REGION" \
  --ami-id "$AMI_ID" \
  --container-image "$CONTAINER_IMAGE" \
  --container-digest "$CONTAINER_DIGEST" \
  --selected-at "$SELECTED_AT" \
  --out "$SELECTION" > provider-selection-plan.json
python -m json.tool provider-selection-plan.json

# APPLY only after every hash and selected profile is reviewed.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  provider select \
  --amendment configs/hardware-amendment-v3.json \
  --region "$REGION" \
  --ami-id "$AMI_ID" \
  --container-image "$CONTAINER_IMAGE" \
  --container-digest "$CONTAINER_DIGEST" \
  --selected-at "$SELECTED_AT" \
  --out "$SELECTION" \
  --apply > provider-selection-result.json
```

The receipt must say `mixed_profiles: false`, contain seeds 0–9 exactly once,
and bind the release profile, amendment, cohort, preregistration, AMI ID,
Region, and image digest.

## 6. Acquire capacity outside msctl

`msctl` never calls `RunInstances`, never creates a reservation, and never
purchases capacity. Capacity must exist first, and every later fleet plan uses
explicit instance IDs.

### P6 Capacity Block path

A P6 Capacity Block purchase is separate, paid, and non-cancellable. It is not
covered by ordinary launch approval. First perform read-only discovery:

```bash
set -euo pipefail
aws ec2 describe-capacity-block-offerings \
  --region "$REGION" \
  --instance-type p6-b300.48xlarge \
  --instance-count 1 \
  --start-date-range REPLACE_WITH_EARLIEST_START \
  --end-date-range REPLACE_WITH_LATEST_END \
  --capacity-duration-hours REPLACE_WITH_DURATION \
  --output json > capacity-block-offerings.json
python -m json.tool capacity-block-offerings.json
sha256sum capacity-block-offerings.json
```

An independent cost approver must sign the exact offering ID, start/end,
instance count, currency, and exact total price. Any offering or price change
invalidates approval. Then permission-check the exact purchase without buying:

```bash
set -euo pipefail
CAPACITY_BLOCK_OFFERING_ID=REPLACE_WITH_EXACT_APPROVED_OFFERING

# DRY RUN: expect DryRunOperation; it must not purchase.
aws ec2 purchase-capacity-block \
  --region "$REGION" \
  --capacity-block-offering-id "$CAPACITY_BLOCK_OFFERING_ID" \
  --instance-platform Linux/UNIX \
  --dry-run
```

Only after the expected dry-run result and a second exact-price approval check:

```bash
# PAID APPLY: non-cancellable purchase of the exact approved offering.
aws ec2 purchase-capacity-block \
  --region "$REGION" \
  --capacity-block-offering-id "$CAPACITY_BLOCK_OFFERING_ID" \
  --instance-platform Linux/UNIX \
  --no-dry-run > capacity-block-purchase.json
```

Launch one P6 into that exact block using a reviewed `launch-request.json`.
The request must pin `p6-b300.48xlarge`, `AMI_ID`, private networking, no public
IP, the dedicated instance profile, encrypted volumes, and the Capacity Block
reservation identity:

```bash
# DRY RUN: permission and request review; expect DryRunOperation.
aws ec2 run-instances \
  --region "$REGION" \
  --cli-input-json file://launch-request.json \
  --dry-run

# PAID APPLY only after the launch JSON hash and Capacity Block binding match
# the approval.
aws ec2 run-instances \
  --region "$REGION" \
  --cli-input-json file://launch-request.json \
  --no-dry-run > run-instances-result.json
```

### P5 On-Demand fallback

Use at most four P5 instances. Review current price and quota, then prepare one
immutable `launch-request.json` with count 1–4 and the same private-network and
instance-role controls:

```bash
aws pricing get-products \
  --service-code AmazonEC2 \
  --filters Type=TERM_MATCH,Field=instanceType,Value=p5.48xlarge \
  --region us-east-1 \
  --output json > p5-price-review.json
python -m json.tool p5-price-review.json
sha256sum launch-request.json p5-price-review.json

# DRY RUN: expect DryRunOperation.
aws ec2 run-instances \
  --region "$REGION" \
  --cli-input-json file://launch-request.json \
  --dry-run

# PAID APPLY only after exact hourly price, count, and launch request approval.
aws ec2 run-instances \
  --region "$REGION" \
  --cli-input-json file://launch-request.json \
  --no-dry-run > run-instances-result.json
```

Extract the returned IDs, review them against `DescribeInstances`, and write an
ordered `instance-ids.txt`. Never use tag queries, “all running instances,” or
implicit discovery in a mutating command.

## 7. Stage immutable inputs

The instance role, not operator credentials, reads the cohort-scoped encrypted
S3 prefix and private ECR image. Upload only verified package artifacts and
receipts:

```bash
set -euo pipefail
S3_RELEASE_URI=s3://REPLACE_WITH_COHORT_BUCKET/releases/REPLACE_WITH_RELEASE_ID

# DRY RUN: review every S3 destination and object.
aws s3 sync "$RELEASE_DIR" "$S3_RELEASE_URI" \
  --region "$REGION" \
  --sse aws:kms \
  --sse-kms-key-id REPLACE_WITH_COHORT_KMS_KEY \
  --dryrun

# APPLY after the dry-run object list and KMS key are approved.
aws s3 sync "$RELEASE_DIR" "$S3_RELEASE_URI" \
  --region "$REGION" \
  --sse aws:kms \
  --sse-kms-key-id REPLACE_WITH_COHORT_KMS_KEY
```

Use SSM to reach only the explicit IDs. First perform the read-only
connectivity check and review the exact target before opening a session:

```bash
INSTANCE_ID="$(sed -n '1p' instance-ids.txt)"
aws ssm describe-instance-information \
  --region "$REGION" \
  --filters Key=InstanceIds,Values="$INSTANCE_ID"

# DRY RUN / REVIEW: prints the exact target; opens no session.
printf 'aws ssm start-session --region %q --target %q\n' \
  "$REGION" "$INSTANCE_ID"

# APPLY only after the target ID and Region are reviewed.
aws ssm start-session \
  --region "$REGION" \
  --target "$INSTANCE_ID"
```

## 8. Environment and bootstrap

On each approved instance, set only runtime identities: Region, pinned AMI ID,
digest-pinned private ECR image/digest, cohort S3 root, runtime UID/GID, and the
dedicated instance-profile ARN. Do not inject operator credentials.

Bootstrap inspects IMDSv2 identity and exact hardware before rendering changes.
Run it first without `--apply`:

```bash
set -euo pipefail
export AWS_REGION="$REGION"
export MS_AWS_AMI_ID="$AMI_ID"
export MS_CONTAINER_IMAGE="$CONTAINER_IMAGE"
export MS_CONTAINER_DIGEST="$CONTAINER_DIGEST"
export MS_S3_ROOT=s3://REPLACE_WITH_COHORT_BUCKET/REPLACE_WITH_COHORT_PREFIX
export MS_RUNTIME_UID=10001
export MS_RUNTIME_GID=10001
INSTANCE_ID="${INSTANCE_ID:?set one explicit reviewed instance ID}"

# DRY RUN: hardware checks run, but RAID/extraction changes are only rendered.
sudo -E cluster/aws/p5/bootstrap.sh \
  --profile "$PROFILE" \
  --container-image "$CONTAINER_IMAGE" \
  --release-archive /staging/REPLACE_WITH_RELEASE_ARCHIVE.zip \
  --release-sha256 REPLACE_WITH_ARCHIVE_SHA256 \
  --release-receipt /staging/RELEASE-AWS-GPU-V3.json \
  --release-receipt-sha256 REPLACE_WITH_RECEIPT_SHA256 \
  --dataset-receipt /staging/dataset/receipt.json \
  --dataset-receipt-sha256 REPLACE_WITH_DATASET_SHA256 \
  --cohort-assignment configs/cohort-assignment-v3.json \
  --cohort-assignment-sha256 REPLACE_WITH_COHORT_SHA256 \
  --code-commit REPLACE_WITH_SOURCE_COMMIT \
  --receipt /staging/environment-receipt.json \
  --owner-uid 10001 \
  --owner-gid 10001 \
  --aws-private-home /run/memorysplit-aws
```

Review device model/count/size, RAID0 target, mounts, image digest, source
commit, and receipt paths. Bootstrap destroys instance-store contents, which
are ephemeral:

```bash
# APPLY only on the explicit reviewed instance.
sudo -E cluster/aws/p5/bootstrap.sh \
  --profile "$PROFILE" \
  --container-image "$CONTAINER_IMAGE" \
  --release-archive /staging/REPLACE_WITH_RELEASE_ARCHIVE.zip \
  --release-sha256 REPLACE_WITH_ARCHIVE_SHA256 \
  --release-receipt /staging/RELEASE-AWS-GPU-V3.json \
  --release-receipt-sha256 REPLACE_WITH_RECEIPT_SHA256 \
  --dataset-receipt /staging/dataset/receipt.json \
  --dataset-receipt-sha256 REPLACE_WITH_DATASET_SHA256 \
  --cohort-assignment configs/cohort-assignment-v3.json \
  --cohort-assignment-sha256 REPLACE_WITH_COHORT_SHA256 \
  --code-commit REPLACE_WITH_SOURCE_COMMIT \
  --receipt /staging/environment-receipt.json \
  --owner-uid 10001 \
  --owner-gid 10001 \
  --aws-private-home /run/memorysplit-aws \
  --authorize-destructive-instance-store \
  --apply
```

Bootstrap publishes the final, instance-bound environment receipt under
`$MS_S3_ROOT/receipts/bootstrap/<instance-id>.json`. For every explicit fleet
ID, review and retrieve that exact receipt to the operator checkout:

```bash
ENVIRONMENT_RECEIPT="operator/environment-${INSTANCE_ID}.json"
BOOTSTRAP_RECEIPT_URI="${MS_S3_ROOT}/receipts/bootstrap/${INSTANCE_ID}.json"

# DRY RUN: review the exact source URI and per-instance destination.
aws s3 cp "$BOOTSTRAP_RECEIPT_URI" "$ENVIRONMENT_RECEIPT" \
  --region "$REGION" \
  --dryrun

# APPLY after the instance ID and URI are reviewed; this only writes locally.
aws s3 cp "$BOOTSTRAP_RECEIPT_URI" "$ENVIRONMENT_RECEIPT" \
  --region "$REGION"
sha256sum "$ENVIRONMENT_RECEIPT"
```

Keep one receipt per instance; never reuse a receipt from another fleet ID.
Copy every 30-minute checkpoint and its canonical receipt to durable
cohort-scoped S3. Local NVMe is scratch, never the only checkpoint copy.

## 9. Canary, 100/10 ETA, and launch gate

Render the profile-bound canary plan without executing it:

```bash
python - <<'PY' > canary-plan.json
import json
import os
from pathlib import Path
from cluster.aws.p5.canary import render_canary_command_plan
from cluster.aws.p5.profile import load_aws_gpu_profile, validate_runtime_environment

profile = load_aws_gpu_profile(Path(os.environ["PROFILE"]))
runtime = validate_runtime_environment(profile, os.environ)
print(json.dumps(render_canary_command_plan(profile, runtime), sort_keys=True))
PY
python -m json.tool canary-plan.json
```

The approved SSM executor must show the exact instance ID and argv from this
plan in dry-run/review mode before it executes them. It must not translate them
into implicit fleet discovery. The qualification receipt must prove:

- eight matching GPUs and the profile software floors;
- BF16, SDPA, `torch.compile`, fused AdamW, NVLink/fabric manager, NVMe, and
  checkpoint/resume;
- concurrent Dense on GPUs 0–3 and Split90 on GPUs 4–7;
- exactly 100 updates per arm, discarding exactly the first 10 warmup updates;
- ETA computed from the slower arm’s remaining 90 measured updates, 13,582
  production updates per arm, and all ten pairs.

Validate the returned receipt locally:

```bash
python - <<'PY'
from pathlib import Path
import os
from cluster.aws.p5.canary import load_qualification_receipt
from cluster.aws.p5.profile import load_aws_gpu_profile

profile = load_aws_gpu_profile(Path(os.environ["PROFILE"]))
report = load_qualification_receipt(Path("qualification-receipt.json"), profile)
print(report.as_dict())
PY
```

Do not proceed if any capability is false, a profile/hash differs, hardware is
mixed, an update is missing, or ETA misses the approved deadline/cost window.

## 10. Instantiate manifests and make the explicit fleet plan

For each seed, run `runs instantiate` first without and then with `--apply`.
The example shows seed 0; repeat identically for 0–9:

```bash
SEED=0
MANIFEST="operator/manifests/seed-${SEED}.json"

# DRY RUN: validates release, dataset, amendment, selection, and eval code.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  runs instantiate \
  --release "$RELEASE_RECEIPT" \
  --dataset-receipt /staging/dataset/receipt.json \
  --seed "$SEED" \
  --hardware-amendment configs/hardware-amendment-v3.json \
  --provider-selection "$SELECTION" \
  --sealed-evaluation evals/confirmatory/runner.py \
  --out "$MANIFEST"

# APPLY only after the rendered manifest hashes are reviewed.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  runs instantiate \
  --release "$RELEASE_RECEIPT" \
  --dataset-receipt /staging/dataset/receipt.json \
  --seed "$SEED" \
  --hardware-amendment configs/hardware-amendment-v3.json \
  --provider-selection "$SELECTION" \
  --sealed-evaluation evals/confirmatory/runner.py \
  --out "$MANIFEST" \
  --apply
```

Build the fleet command from all ten explicit manifest paths and the reviewed
instance IDs. Run it once without `--apply`, review, then repeat with `--apply`:

```bash
# DRY RUN.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  fleet plan \
  --amendment configs/hardware-amendment-v3.json \
  --provider-selection "$SELECTION" \
  --manifest operator/manifests/seed-0.json \
  --manifest operator/manifests/seed-1.json \
  --manifest operator/manifests/seed-2.json \
  --manifest operator/manifests/seed-3.json \
  --manifest operator/manifests/seed-4.json \
  --manifest operator/manifests/seed-5.json \
  --manifest operator/manifests/seed-6.json \
  --manifest operator/manifests/seed-7.json \
  --manifest operator/manifests/seed-8.json \
  --manifest operator/manifests/seed-9.json \
  --instance-id REPLACE_WITH_EXPLICIT_INSTANCE_ID \
  --out operator/fleet-plan-v3.json
```

For P6, supply exactly one ID: all ten pairs run sequentially in seed order.
For the P5 fallback, supply at most four ordered IDs. Four IDs deterministically
produce 3/3/2/2 pairs: seeds 0/4/8, 1/5/9, 2/6, and 3/7. Fewer P5 IDs remain
round-robin and sequential per instance. Never run two pairs concurrently on
one instance.

After review, repeat the exact fleet command with `--apply`. Do not alter ID
order between review and publication.

## 11. Submit, monitor, and checkpoint

Every lifecycle command binds the amendment, provider selection, fleet plan,
manifest, explicit instance ID, dataset evidence, environment receipt, release,
and approval. `msctl` sends work to an existing instance through SSM; it cannot
provision one.

For each fleet wave, render before submit:

```bash
MANIFEST=operator/manifests/seed-0.json
INSTANCE_ID=REPLACE_WITH_FLEET_BOUND_INSTANCE_ID
ENVIRONMENT_RECEIPT="operator/environment-${INSTANCE_ID}.json"
TERMINATE_AT=REPLACE_WITH_APPROVED_UTC_DEADLINE

# DRY RUN: no SSM command is sent.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  submit \
  --release "$RELEASE_RECEIPT" \
  --manifest "$MANIFEST" \
  --dataset-pointer DATASET-POINTER-AWS.json \
  --dataset-verification operator/dataset-verification.json \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --hardware-amendment configs/hardware-amendment-v3.json \
  --provider-selection "$SELECTION" \
  --fleet-plan operator/fleet-plan-v3.json \
  --instance-id "$INSTANCE_ID" \
  --terminate-at "$TERMINATE_AT" \
  --approval operator/submit-approval.json

# APPLY only after argv, ID, wave, ETA, cost, and deadline review.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  submit \
  --release "$RELEASE_RECEIPT" \
  --manifest "$MANIFEST" \
  --dataset-pointer DATASET-POINTER-AWS.json \
  --dataset-verification operator/dataset-verification.json \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --hardware-amendment configs/hardware-amendment-v3.json \
  --provider-selection "$SELECTION" \
  --fleet-plan operator/fleet-plan-v3.json \
  --instance-id "$INSTANCE_ID" \
  --terminate-at "$TERMINATE_AT" \
  --approval operator/submit-approval.json \
  --apply
```

Poll with `status --cached` first, then the read-only authoritative status
path. Verify both arms advance together. The trainer’s `ckpt_minutes: 30`
requires a durable paired checkpoint receipt at least every 30 minutes; sync
the checkpoint and receipt only after a dry-run object review:

```bash
# DRY RUN.
aws s3 sync /mnt/memorysplit/runs \
  s3://REPLACE_WITH_COHORT_BUCKET/checkpoints \
  --region "$REGION" \
  --exclude '*' \
  --include '*.pt' \
  --include '*.json' \
  --dryrun

# APPLY after paths, encryption policy, and receipt hashes are reviewed.
aws s3 sync /mnt/memorysplit/runs \
  s3://REPLACE_WITH_COHORT_BUCKET/checkpoints \
  --region "$REGION" \
  --exclude '*' \
  --include '*.pt' \
  --include '*.json'
```

## 12. Resume, evaluate, and collect

Resume only from a verified paired checkpoint receipt:

```bash
# DRY RUN.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  resume \
  --release "$RELEASE_RECEIPT" \
  --manifest "$MANIFEST" \
  --checkpoint-receipt operator/checkpoint-receipt.json \
  --dataset-pointer DATASET-POINTER-AWS.json \
  --dataset-verification operator/dataset-verification.json \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --hardware-amendment configs/hardware-amendment-v3.json \
  --provider-selection "$SELECTION" \
  --fleet-plan operator/fleet-plan-v3.json \
  --approval operator/resume-approval.json

# APPLY after checkpoint hashes, steps, world sizes, and explicit ID review.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  resume \
  --release "$RELEASE_RECEIPT" \
  --manifest "$MANIFEST" \
  --checkpoint-receipt operator/checkpoint-receipt.json \
  --dataset-pointer DATASET-POINTER-AWS.json \
  --dataset-verification operator/dataset-verification.json \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --hardware-amendment configs/hardware-amendment-v3.json \
  --provider-selection "$SELECTION" \
  --fleet-plan operator/fleet-plan-v3.json \
  --approval operator/resume-approval.json \
  --apply
```

Evaluate only after both terminal checkpoints are durable and verified:

```bash
# DRY RUN: validates sealed-evaluation and lifecycle bindings.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  evaluate \
  --release "$RELEASE_RECEIPT" \
  --manifest "$MANIFEST" \
  --dataset-pointer DATASET-POINTER-AWS.json \
  --dataset-verification operator/dataset-verification.json \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --hardware-amendment configs/hardware-amendment-v3.json \
  --provider-selection "$SELECTION" \
  --fleet-plan operator/fleet-plan-v3.json \
  --approval operator/evaluate-approval.json

# APPLY after sealed-release, expected study-lock hash, device, and output review.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  evaluate \
  --release "$RELEASE_RECEIPT" \
  --manifest "$MANIFEST" \
  --dataset-pointer DATASET-POINTER-AWS.json \
  --dataset-verification operator/dataset-verification.json \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --hardware-amendment configs/hardware-amendment-v3.json \
  --provider-selection "$SELECTION" \
  --fleet-plan operator/fleet-plan-v3.json \
  --approval operator/evaluate-approval.json \
  --apply
```

Collect by exact source and exclusive output:

```bash
# DRY RUN.
python -m msctl \
  --profile "$PROFILE" \
  collect \
  --source results/seed-0.json \
  --out operator/collected/seed-0.json

# APPLY after source and destination review.
python -m msctl \
  --profile "$PROFILE" \
  collect \
  --source results/seed-0.json \
  --out operator/collected/seed-0.json \
  --apply
```

Repeat evaluate/collect for every seed and report all ten paired outcomes.

## 13. Stop, restart, and teardown

Stopping an instance does not cancel or refund a P6 Capacity Block. Before any
emergency stop or restart, review the exact ID and checkpoint durability:

```bash
# DRY RUN.
aws ec2 stop-instances \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --dry-run

# APPLY only after durable checkpoint confirmation.
aws ec2 stop-instances \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --no-dry-run

# DRY RUN before a later restart.
aws ec2 start-instances \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --dry-run

# APPLY after revalidating AMI, role, network, and resume plan.
aws ec2 start-instances \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --no-dry-run
```

At completion, verify durable checkpoints, evaluations, logs, operation
receipts, provider selection, fleet plan, and collection hashes. Then terminate
each explicit ID:

```bash
# DRY RUN.
aws ec2 terminate-instances \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --dry-run

# APPLY after final durable-artifact and exact-ID review.
aws ec2 terminate-instances \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --no-dry-run

# Read-only confirmation.
aws ec2 describe-instances \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[].Instances[].State.Name'
```

Remove no S3 evidence, KMS key, ECR digest, approval, or receipt until the
retention owner approves deletion through a separate dry-run/review/apply
process. A Capacity Block remains non-cancellable even after its instance is
terminated.
