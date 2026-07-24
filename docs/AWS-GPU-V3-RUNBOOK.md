# AWS GPU v3 P5/P6 runbook

This is the paid-run procedure for
`memorysplit-confirmatory-v3-360m-n10-aws`. It is fail-closed. A package,
Capacity Block, running instance, or approval file is not scientific launch
authorization by itself.

## 0. Non-negotiable gates

Before protected work, record evidence that all of the following are true:

1. The frozen cohort, preregistration, hardware amendment, release, dataset,
   provider selection, environment, pre-launch sealed fixture, and run
   manifests are hash-bound. The fixture contains only `items.jsonl`,
   `stores.jsonl`, and `sealed-gold.jsonl`; checkpoint and study-lock hashes do
   not exist yet.
2. The six 29M diagnostics completed: full-corpus Dense/Split90, no-ARC and
   no-ConceptARC Dense/Split90, and no-refinement Dense/Split90. Review all six
   receipts together; do not drop a failed or inconvenient arm.
3. The selected profile passed hardware, software, simultaneous 4+4, one-step
   training, checkpoint/resume, and NVMe qualification.
4. `protected_launch_allowed` can be changed from false only through the
   separately reviewed gate process. This runbook does not change it.

After all twenty terminal checkpoints are durable, a separate finalization
step binds those checkpoint records into the N=10 study lock and complete
sealed-evaluation release. Those final hashes authorize evaluation, not
training launch.

The 29M diagnostic jobs are paid work. Their owning manifest must first be
rendered in dry-run mode, reviewed as one six-run set, and only then submitted
with its explicit apply control. The v3 360M package does not contain diagnostic
outputs or authorize that submission.

## 1. Identity, Region, and network preflight

Use temporary federation into the approved operator role. Do not create, copy,
or export static credentials. The EC2 workers use a different, dedicated
instance role delivered through instance metadata.

Keep every generated plan, raw AWS response, approval, receipt, manifest, and
collection outside the reviewed source checkout. The commands below assume one
operator root next to the repository:

```bash
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
OPERATOR_ROOT="$(dirname "$REPO_ROOT")/memorysplit-aws-v3-operator"
REVIEW_ROOT="$OPERATOR_ROOT/review"
mkdir -p \
  "$REVIEW_ROOT" \
  "$OPERATOR_ROOT/approvals" \
  "$OPERATOR_ROOT/collected" \
  "$OPERATOR_ROOT/manifests" \
  "$OPERATOR_ROOT/receipts"
case "$(realpath "$OPERATOR_ROOT")/" in
  "$(realpath "$REPO_ROOT")/"*) echo "operator root is inside source tree" >&2; exit 2 ;;
esac
test -z "$(git -C "$REPO_ROOT" status --short)"
```

Do not place temporary review files under an ignored source-tree directory:
clean-tree gates intentionally inspect the whole checkout, not only tracked
files.

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
  --output json > "$REVIEW_ROOT/quota-review.json"
aws ec2 describe-instance-type-offerings \
  --location-type region \
  --filters Name=instance-type,Values=p5.48xlarge,p6-b300.48xlarge \
  --region "$REGION" \
  --output json > "$REVIEW_ROOT/offering-review.json"
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
  --source-root . > "$REVIEW_ROOT/release-verification.json"
python -m json.tool "$REVIEW_ROOT/release-verification.json"
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
  --output json > "$REVIEW_ROOT/dlami-review.json"
python -m json.tool "$REVIEW_ROOT/dlami-review.json"
sha256sum "$REVIEW_ROOT/dlami-review.json"

AMI_EVIDENCE="$OPERATOR_ROOT/receipts/ami-describe-v1.json"
python - "$REGION" "$AMI_ID" \
  "$REVIEW_ROOT/dlami-review.json" "$AMI_EVIDENCE" <<'PY'
import json
import pathlib
import sys

region, image_id, raw_path, out_path = sys.argv[1:]
raw = json.loads(pathlib.Path(raw_path).read_text(encoding="utf-8"))
images = raw.get("Images")
if not isinstance(images, list) or len(images) != 1:
    raise SystemExit("AMI evidence requires exactly one image")
image = images[0]
receipt = {
    "schema_version": 1,
    "receipt_type": "memorysplit-aws-ami-describe-v1",
    "region": region,
    "image_id": image_id,
    "owner_id": image["OwnerId"],
    "name": image["Name"],
}
pathlib.Path(out_path).write_text(
    json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="ascii",
)
PY
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
IMAGE_PLAN="$REVIEW_ROOT/image-plan.json"
IMAGE_RESULT="$REVIEW_ROOT/image-result.json"

# DRY RUN: prints exact Docker build and push argv; executes neither.
python scripts/build_aws_gpu_image.py \
  --destination "$ECR_REPOSITORY" > "$IMAGE_PLAN"
python -m json.tool "$IMAGE_PLAN"

# APPLY only after base digest, context hash, immutable build tag, and
# destination repository are reviewed.
python scripts/build_aws_gpu_image.py \
  --destination "$ECR_REPOSITORY" \
  --apply > "$IMAGE_RESULT"
```

After the push, use read-only ECR calls to resolve and record the manifest
digest. The only runtime form is the private
`repository@sha256:<manifest-digest>` reference; a tag alone is forbidden.

```bash
IMAGE_TAG="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["destination"].rsplit(":", 1)[1])' "$IMAGE_RESULT")"
aws ecr describe-images \
  --region "$REGION" \
  --repository-name REPLACE_WITH_REPOSITORY_NAME \
  --image-ids imageTag="$IMAGE_TAG" \
  --output json > "$REVIEW_ROOT/ecr-image-review.json"
python -m json.tool "$REVIEW_ROOT/ecr-image-review.json"
```

Set `CONTAINER_DIGEST` from the reviewed `imageDigest`, and set
`CONTAINER_IMAGE` to the same private repository plus `@CONTAINER_DIGEST`.
Then create the closed evidence files consumed by provider selection:

```bash
set -euo pipefail
CONTAINER_DIGEST="$(python -c 'import json,sys; rows=json.load(open(sys.argv[1]))["imageDetails"]; assert len(rows)==1; print(rows[0]["imageDigest"])' "$REVIEW_ROOT/ecr-image-review.json")"
CONTAINER_IMAGE="${ECR_REPOSITORY}@${CONTAINER_DIGEST}"
ECR_EVIDENCE="$OPERATOR_ROOT/receipts/ecr-describe-v1.json"
IMAGE_BUILD_RECEIPT="$OPERATOR_ROOT/receipts/image-build-v1.json"

python - "$REGION" "$ECR_REPOSITORY" "$CONTAINER_IMAGE" \
  "$CONTAINER_DIGEST" "$IMAGE_RESULT" "$ECR_EVIDENCE" \
  "$IMAGE_BUILD_RECEIPT" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    region,
    repository_uri,
    image_uri,
    digest,
    build_result_path,
    ecr_out,
    build_out,
) = sys.argv[1:]
registry, repository_name = repository_uri.split("/", 1)
account_id = registry.split(".", 1)[0]
build = json.loads(pathlib.Path(build_result_path).read_text(encoding="utf-8"))

def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

ecr = {
    "schema_version": 1,
    "receipt_type": "memorysplit-aws-ecr-describe-v1",
    "region": region,
    "registry_id": account_id,
    "repository_name": repository_name,
    "image_digest": digest,
    "image_uri": image_uri,
}
image_build = {
    "schema_version": 1,
    "receipt_type": "memorysplit-aws-gpu-image-build-v1",
    "aws_account_id": account_id,
    "region": region,
    "container_image": image_uri,
    "container_digest": digest,
    "build_context_sha256": build["build_context_sha256"],
    "dockerfile_sha256": sha256("containers/aws-gpu/Dockerfile"),
    "image_lock_sha256": sha256("containers/aws-gpu/image.lock.json"),
    "runtime_dependency_lock_sha256": sha256(
        "containers/aws-gpu/requirements.lock"
    ),
}
for path, value in ((ecr_out, ecr), (build_out, image_build)):
    pathlib.Path(path).write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
PY
```

## 5. Publish the provider-selection receipt

Use one fixed timestamp for dry-run and apply so the reviewed bytes are the
published bytes. P5 uses no capacity arguments. P6 selection is impossible
until the exact Capacity Block offering and resulting reservation IDs exist;
perform section 6 first, then return here with both IDs:

```bash
set -euo pipefail
CONTAINER_DIGEST=sha256:REPLACE_WITH_REVIEWED_MANIFEST_DIGEST
CONTAINER_IMAGE="${ECR_REPOSITORY}@${CONTAINER_DIGEST}"
SELECTED_AT=REPLACE_WITH_APPROVED_UTC_TIMESTAMP
SELECTION="$OPERATOR_ROOT/provider-selection-v3.json"
CAPACITY_ARGS=()
if [ "$PROFILE" = "cluster/profiles/aws-p6-b300.48xlarge-v3.json" ]; then
  CAPACITY_RESERVATION_ID=REPLACE_WITH_EXACT_CAPACITY_RESERVATION_ID
  CAPACITY_BLOCK_OFFERING_ID=REPLACE_WITH_EXACT_APPROVED_OFFERING_ID
  CAPACITY_ARGS=(
    --capacity-reservation-id "$CAPACITY_RESERVATION_ID"
    --capacity-block-offering-id "$CAPACITY_BLOCK_OFFERING_ID"
  )
fi

# DRY RUN: validates but does not write the receipt.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  provider select \
  --amendment configs/hardware-amendment-v3.json \
  --region "$REGION" \
  --ami-id "$AMI_ID" \
  --ami-evidence "$AMI_EVIDENCE" \
  --container-image "$CONTAINER_IMAGE" \
  --container-digest "$CONTAINER_DIGEST" \
  --ecr-evidence "$ECR_EVIDENCE" \
  --image-build-receipt "$IMAGE_BUILD_RECEIPT" \
  "${CAPACITY_ARGS[@]}" \
  --selected-at "$SELECTED_AT" \
  --out "$SELECTION" > "$REVIEW_ROOT/provider-selection-plan.json"
python -m json.tool "$REVIEW_ROOT/provider-selection-plan.json"

# APPLY only after every hash and selected profile is reviewed.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  provider select \
  --amendment configs/hardware-amendment-v3.json \
  --region "$REGION" \
  --ami-id "$AMI_ID" \
  --ami-evidence "$AMI_EVIDENCE" \
  --container-image "$CONTAINER_IMAGE" \
  --container-digest "$CONTAINER_DIGEST" \
  --ecr-evidence "$ECR_EVIDENCE" \
  --image-build-receipt "$IMAGE_BUILD_RECEIPT" \
  "${CAPACITY_ARGS[@]}" \
  --selected-at "$SELECTED_AT" \
  --out "$SELECTION" \
  --apply > "$REVIEW_ROOT/provider-selection-result.json"
```

The receipt must say `mixed_profiles: false`, contain seeds 0–9 exactly once,
and bind the release profile, amendment, cohort, preregistration, AMI describe
evidence, Region, private-ECR describe evidence, image build context,
Dockerfile, dependency lock, image digest, and the selected purchase model.

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
  --output json > "$REVIEW_ROOT/capacity-block-offerings.json"
python -m json.tool "$REVIEW_ROOT/capacity-block-offerings.json"
sha256sum "$REVIEW_ROOT/capacity-block-offerings.json"
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
  --no-dry-run > "$REVIEW_ROOT/capacity-block-purchase.json"
```

Launch one P6 into that exact block using a reviewed `launch-request.json`.
The request must pin `p6-b300.48xlarge`, `AMI_ID`, private networking, no public
IP, the dedicated instance profile, encrypted volumes, and the Capacity Block
reservation identity. In particular, the reviewed JSON must contain these
literal API fields; an ordinary reservation target without the market type is
not a Capacity Block launch:

```json
{
  "MinCount": 1,
  "MaxCount": 1,
  "InstanceType": "p6-b300.48xlarge",
  "InstanceMarketOptions": {
    "MarketType": "capacity-block"
  },
  "CapacityReservationSpecification": {
    "CapacityReservationTarget": {
      "CapacityReservationId": "REPLACE_WITH_EXACT_CAPACITY_RESERVATION_ID"
    }
  }
}
```

Validate those closed fields before either AWS call:

```bash
LAUNCH_REQUEST="$OPERATOR_ROOT/launch-request.json"
python - "$LAUNCH_REQUEST" "$CAPACITY_RESERVATION_ID" <<'PY'
import json
import sys

request = json.load(open(sys.argv[1], encoding="utf-8"))
assert request["MinCount"] == request["MaxCount"] == 1
assert request["InstanceType"] == "p6-b300.48xlarge"
assert request["InstanceMarketOptions"] == {"MarketType": "capacity-block"}
assert request["CapacityReservationSpecification"] == {
    "CapacityReservationTarget": {"CapacityReservationId": sys.argv[2]}
}
PY

# DRY RUN: permission and request review; expect DryRunOperation.
aws ec2 run-instances \
  --region "$REGION" \
  --cli-input-json "file://$LAUNCH_REQUEST" \
  --dry-run

# PAID APPLY only after the launch JSON hash and Capacity Block binding match
# the approval.
aws ec2 run-instances \
  --region "$REGION" \
  --cli-input-json "file://$LAUNCH_REQUEST" \
  --no-dry-run > "$REVIEW_ROOT/run-instances-result.json"
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
  --output json > "$REVIEW_ROOT/p5-price-review.json"
python -m json.tool "$REVIEW_ROOT/p5-price-review.json"
LAUNCH_REQUEST="$OPERATOR_ROOT/launch-request.json"
sha256sum "$LAUNCH_REQUEST" "$REVIEW_ROOT/p5-price-review.json"

# DRY RUN: expect DryRunOperation.
aws ec2 run-instances \
  --region "$REGION" \
  --cli-input-json "file://$LAUNCH_REQUEST" \
  --dry-run

# PAID APPLY only after exact hourly price, count, and launch request approval.
aws ec2 run-instances \
  --region "$REGION" \
  --cli-input-json "file://$LAUNCH_REQUEST" \
  --no-dry-run > "$REVIEW_ROOT/run-instances-result.json"
```

Extract the returned IDs, review them against `DescribeInstances`, and write an
ordered `$OPERATOR_ROOT/instance-ids.txt`. Never use tag queries, “all running
instances,” or implicit discovery in a mutating command.

## 7. Stage immutable inputs

The instance role, not operator credentials, reads the cohort-scoped encrypted
S3 prefix and private ECR image. Upload only verified package artifacts and
receipts:

```bash
set -euo pipefail
export MS_S3_ROOT=s3://REPLACE_WITH_COHORT_BUCKET/REPLACE_WITH_COHORT_PREFIX
export MS_S3_KMS_KEY_ID=arn:aws:kms:REPLACE_WITH_REGION:REPLACE_WITH_ACCOUNT:key/REPLACE_WITH_KEY_UUID
RELEASE_ARCHIVE="$(python -c 'import json,sys; print(json.load(sys.stdin)["archive"])' <<<"$PACKAGE_RESULT")"
RELEASE_SHA256="$(python -c 'import json,sys; print(json.load(sys.stdin)["sha256"])' <<<"$PACKAGE_RESULT")"
S3_RELEASE_PREFIX="${MS_S3_ROOT}/releases/${RELEASE_SHA256}"

# DRY RUN: review each exact source and the runtime's fixed destination name.
aws s3 cp "$RELEASE_ARCHIVE" "$S3_RELEASE_PREFIX/release.zip" \
  --region "$REGION" \
  --sse aws:kms \
  --sse-kms-key-id "$MS_S3_KMS_KEY_ID" \
  --dryrun
aws s3 cp "$RELEASE_RECEIPT" "$S3_RELEASE_PREFIX/RELEASE.json" \
  --region "$REGION" \
  --sse aws:kms \
  --sse-kms-key-id "$MS_S3_KMS_KEY_ID" \
  --dryrun
aws s3 cp configs/cohort-assignment-v3.json \
  "$S3_RELEASE_PREFIX/cohort-assignment-v3.json" \
  --region "$REGION" \
  --sse aws:kms \
  --sse-kms-key-id "$MS_S3_KMS_KEY_ID" \
  --dryrun

# APPLY after all three paths, hashes, and the exact KMS key are approved.
aws s3 cp "$RELEASE_ARCHIVE" "$S3_RELEASE_PREFIX/release.zip" \
  --region "$REGION" \
  --sse aws:kms \
  --sse-kms-key-id "$MS_S3_KMS_KEY_ID"
aws s3 cp "$RELEASE_RECEIPT" "$S3_RELEASE_PREFIX/RELEASE.json" \
  --region "$REGION" \
  --sse aws:kms \
  --sse-kms-key-id "$MS_S3_KMS_KEY_ID"
aws s3 cp configs/cohort-assignment-v3.json \
  "$S3_RELEASE_PREFIX/cohort-assignment-v3.json" \
  --region "$REGION" \
  --sse aws:kms \
  --sse-kms-key-id "$MS_S3_KMS_KEY_ID"
```

Stage the separately controlled pre-launch evaluator fixture under its own
content root, not inside the training archive. It contains exactly
`items.jsonl`, `stores.jsonl`, and `sealed-gold.jsonl`; adding
`checkpoints.jsonl`, `study-lock.json`, or any other file is an error. Compute
its checkpoint-independent root by loading the closed contract:

The separately controlled evaluator release is not created until the
post-training finalization step in section 12.

```bash
set -euo pipefail
SEALED_FIXTURE_ROOT=REPLACE_WITH_EXTERNAL_SEALED_FIXTURE_DIRECTORY
SEALED_FIXTURE_REPORT="$REVIEW_ROOT/sealed-fixture.json"
python - "$SEALED_FIXTURE_ROOT" > "$SEALED_FIXTURE_REPORT" <<'PY'
import json
import sys
from msctl.aws_sealed_evaluation import load_sealed_evaluation_fixture

fixture = load_sealed_evaluation_fixture(sys.argv[1])
print(json.dumps({
    "sealed_fixture_sha256": fixture.sha256,
    "members": dict(fixture.members),
}, sort_keys=True, separators=(",", ":")))
PY
SEALED_FIXTURE_SHA256="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["sealed_fixture_sha256"])' "$SEALED_FIXTURE_REPORT")"
SEALED_FIXTURE_S3_URI="${MS_S3_ROOT}/sealed-fixture/${SEALED_FIXTURE_SHA256}"

# DRY RUN, then APPLY with the same exact external root and KMS key.
aws s3 sync "$SEALED_FIXTURE_ROOT" "$SEALED_FIXTURE_S3_URI" \
  --region "$REGION" \
  --sse aws:kms \
  --sse-kms-key-id "$MS_S3_KMS_KEY_ID" \
  --no-follow-symlinks \
  --dryrun
aws s3 sync "$SEALED_FIXTURE_ROOT" "$SEALED_FIXTURE_S3_URI" \
  --region "$REGION" \
  --sse aws:kms \
  --sse-kms-key-id "$MS_S3_KMS_KEY_ID" \
  --no-follow-symlinks
```

Publish the separately verified dataset receipt and members under
`$MS_S3_ROOT/dataset` before bootstrap. Keep the raw review response and every
generated receipt under `$OPERATOR_ROOT`, never in the source checkout.

Use SSM to reach only the explicit IDs. First perform the read-only
connectivity check and review the exact target before opening a session:

```bash
INSTANCE_ID="$(sed -n '1p' "$OPERATOR_ROOT/instance-ids.txt")"
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
digest-pinned private ECR image/digest, cohort S3 root and KMS key, and runtime
UID/GID. Do not inject operator credentials.

A stock DLAMI has no `/opt/memorysplit` checkout. First build the deterministic
closed control tar, review its hash, and install those exact S3 bytes with the
AWS-managed `AWS-RunShellScript` document. This step must complete before any
custom argv document is used:

```bash
set -euo pipefail
export AWS_REGION="$REGION"
export MS_AWS_AMI_ID="$AMI_ID"
export MS_CONTAINER_IMAGE="$CONTAINER_IMAGE"
export MS_CONTAINER_DIGEST="$CONTAINER_DIGEST"
export MS_RUNTIME_UID=10001
export MS_RUNTIME_GID=10001
INSTANCE_ID="${INSTANCE_ID:?set one explicit reviewed instance ID}"
CONTROL_BUNDLE="$OPERATOR_ROOT/control-bundle.tar"

# DRY RUN and exclusive local APPLY; both derive byte-identical tar bytes.
python -m msctl --profile "$PROFILE" --repo-root . \
  control bundle --out "$CONTROL_BUNDLE" \
  > "$REVIEW_ROOT/control-bundle-plan.json"
python -m msctl --profile "$PROFILE" --repo-root . \
  control bundle --out "$CONTROL_BUNDLE" --apply \
  > "$REVIEW_ROOT/control-bundle-result.json"
CONTROL_BUNDLE_SHA256="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["bundle_sha256"])' "$REVIEW_ROOT/control-bundle-result.json")"

# DRY RUN: renders immutable S3 publication and the stock-document command.
python -m msctl --profile "$PROFILE" --repo-root . \
  --state-root "$OPERATOR_ROOT/state" \
  control install --instance-id "$INSTANCE_ID" \
  > "$REVIEW_ROOT/control-install-plan-${INSTANCE_ID}.json"

# APPLY: publishes with SSE-KMS/no-overwrite, verifies the object, then sends
# exactly one AWS-RunShellScript command to the explicit instance.
python -m msctl --profile "$PROFILE" --repo-root . \
  --state-root "$OPERATOR_ROOT/state" \
  control install --instance-id "$INSTANCE_ID" --apply \
  > "$REVIEW_ROOT/control-install-result-${INSTANCE_ID}.json"
CONTROL_COMMAND_ID="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["command_id"])' "$REVIEW_ROOT/control-install-result-${INSTANCE_ID}.json")"

# Read-only completion check. Do not proceed on Pending/InProgress/Failed.
aws ssm get-command-invocation \
  --region "$REGION" \
  --instance-id "$INSTANCE_ID" \
  --command-id "$CONTROL_COMMAND_ID" \
  --query '{CommandId:CommandId,Status:Status}' \
  --output json > "$REVIEW_ROOT/control-install-status-${INSTANCE_ID}.json"
test "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["Status"])' "$REVIEW_ROOT/control-install-status-${INSTANCE_ID}.json")" = Success
```

The installed root is
`/opt/memorysplit/control/$CONTROL_BUNDLE_SHA256`. On the explicit instance,
bootstrap inspects authenticated IMDSv2 identity and exact hardware before
rendering any mutation. Use the digest root, never `/opt/memorysplit` or an
unverified checkout:

```bash
set -euo pipefail
CONTROL_ROOT="/opt/memorysplit/control/${CONTROL_BUNDLE_SHA256}"
REMOTE_PROFILE="$CONTROL_ROOT/cluster/profiles/$(basename "$PROFILE")"
RELEASE_RECEIPT_SHA256=REPLACE_WITH_REVIEWED_RELEASE_RECEIPT_SHA256
DATASET_RECEIPT_SHA256=REPLACE_WITH_DATASET_RECEIPT_SHA256
COHORT_SHA256=REPLACE_WITH_REVIEWED_COHORT_ASSIGNMENT_SHA256
SOURCE_COMMIT=REPLACE_WITH_REVIEWED_40_HEX_SOURCE_COMMIT
sudo /usr/bin/install -d -m 0700 -o 0 -g 0 /run/memorysplit-aws
BOOTSTRAP_ARGS=(
  "$CONTROL_ROOT/cluster/aws/p5/bootstrap.py"
  --profile "$REMOTE_PROFILE"
  --container-image "$CONTAINER_IMAGE"
  --release-archive "/mnt/memorysplit/staging/releases/$RELEASE_SHA256/release.zip"
  --release-sha256 "$RELEASE_SHA256"
  --release-receipt "/mnt/memorysplit/staging/releases/$RELEASE_SHA256/RELEASE.json"
  --release-receipt-sha256 "$RELEASE_RECEIPT_SHA256"
  --dataset-receipt /mnt/memorysplit/dataset/receipt.json
  --dataset-receipt-sha256 "$DATASET_RECEIPT_SHA256"
  --cohort-assignment "/mnt/memorysplit/staging/releases/$RELEASE_SHA256/cohort-assignment-v3.json"
  --cohort-assignment-sha256 "$COHORT_SHA256"
  --code-commit "$SOURCE_COMMIT"
  --receipt /mnt/memorysplit/staging/bootstrap-receipt.json
  --owner-uid 10001
  --owner-gid 10001
  --aws-private-home /run/memorysplit-aws
)

# DRY RUN: hardware checks run; RAID, extraction, and S3 sync are only rendered.
sudo -E /usr/bin/python3 "${BOOTSTRAP_ARGS[@]}"

# APPLY only after exact device, image, release, dataset, and command review.
sudo -E /usr/bin/python3 "${BOOTSTRAP_ARGS[@]}" \
  --authorize-destructive-instance-store \
  --apply
```

The first apply may create RAID0 and format ephemeral instance store. A repeated
apply with identical inputs must reverify and reuse the exact RAID, mount,
release root, and receipts. It never recreates storage or overwrites a release;
any member, ownership, mount, device, or receipt drift is a hard failure.
Consequently, later protected submit is allowed to rerun this same bootstrap
without failing merely because qualification already prepared the instance.

Bootstrap prints and publishes an authenticated environment receipt under
`$MS_S3_ROOT/receipts/environment/<instance-id>/<boot-id>.json`, separately
from the bootstrap receipt. For every explicit fleet ID, copy the exact URI
printed by bootstrap to the operator root:

```bash
ENVIRONMENT_RECEIPT="$OPERATOR_ROOT/receipts/environment-${INSTANCE_ID}.json"
ENVIRONMENT_RECEIPT_URI=REPLACE_WITH_EXACT_URI_PRINTED_BY_BOOTSTRAP

# DRY RUN: review the exact source URI and per-instance destination.
aws s3 cp "$ENVIRONMENT_RECEIPT_URI" "$ENVIRONMENT_RECEIPT" \
  --region "$REGION" \
  --dryrun

# APPLY after the instance ID and URI are reviewed; this only writes locally.
aws s3 cp "$ENVIRONMENT_RECEIPT_URI" "$ENVIRONMENT_RECEIPT" \
  --region "$REGION"
sha256sum "$ENVIRONMENT_RECEIPT"
```

Keep one receipt per instance; never reuse a receipt from another fleet ID.
The receipt is boot-bound; stopping and starting an instance changes the boot
ID and requires bootstrap, qualification, and readiness to be repeated.

## 9. Canary, 100/10 ETA, and launch gate

Render the authenticated, instance-bound orchestration plan without executing
it, then publish that local plan exclusively:

```bash
CANARY_PLAN="$REVIEW_ROOT/canary-plan-${INSTANCE_ID}.json"

# DRY RUN: verifies the release, provider selection, PKCS7 identity receipt,
# explicit instance ID, and exact release-mounted phase argv.
python3 -m msctl --profile "$PROFILE" --repo-root . \
  canary plan \
  --release "$RELEASE_RECEIPT" \
  --amendment configs/hardware-amendment-v3.json \
  --provider-selection "$SELECTION" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --instance-id "$INSTANCE_ID" \
  --out "$CANARY_PLAN" \
  > "$REVIEW_ROOT/canary-plan-review-${INSTANCE_ID}.json"

# APPLY writes the canonical plan once and refuses replacement.
python3 -m msctl --profile "$PROFILE" --repo-root . \
  canary plan \
  --release "$RELEASE_RECEIPT" \
  --amendment configs/hardware-amendment-v3.json \
  --provider-selection "$SELECTION" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --instance-id "$INSTANCE_ID" \
  --out "$CANARY_PLAN" \
  --apply > "$REVIEW_ROOT/canary-plan-result-${INSTANCE_ID}.json"

# DRY RUN: render the content-addressed SSM intent without AWS mutations.
python3 -m msctl --profile "$PROFILE" --repo-root . \
  canary run \
  --canary-plan "$CANARY_PLAN" \
  --instance-id "$INSTANCE_ID" \
  > "$REVIEW_ROOT/canary-run-review-${INSTANCE_ID}.json"

# APPLY only after reviewing the exact instance and immutable argv.
python3 -m msctl --profile "$PROFILE" --repo-root . \
  canary run \
  --canary-plan "$CANARY_PLAN" \
  --instance-id "$INSTANCE_ID" \
  --apply > "$REVIEW_ROOT/canary-run-result-${INSTANCE_ID}.json"
```

The SSM path validates the one explicit instance and never calls
`RunInstances` or performs implicit fleet discovery. Each phase runs
`cluster/aws/p5/canary_runtime.py` from the mounted, digest-named release using
`/opt/venv/bin/python`; no shell-generated command is accepted. Preserve every
raw-output hash and build one closed qualification receipt. It must bind the
instance ID, current boot ID, Region, AMI ID/owner, private image/digest,
provider selection, environment receipt, timestamp, command-plan hash, release
hash, and every phase raw-output hash, and it must prove:

- eight matching GPUs and the profile software floors;
- BF16, SDPA, `torch.compile`, fused AdamW, NVLink/fabric manager, NVMe, and
  checkpoint/resume;
- concurrent Dense on GPUs 0–3 and Split90 on GPUs 4–7;
- measured process overlap covering at least half of the shorter concurrent
  arm, so incidental startup/teardown overlap cannot qualify;
- exactly 100 updates per arm, discarding exactly the first 10 warmup updates;
- ETA computed from the slower arm’s remaining 90 measured updates, 13,582
  production updates per arm, and all ten pairs.

Validate the returned receipt locally:

```bash
export QUALIFICATION_RECEIPT="$OPERATOR_ROOT/receipts/qualification-${INSTANCE_ID}.json"
python - <<'PY'
from pathlib import Path
import os
from cluster.aws.p5.canary import load_qualification_receipt
from cluster.aws.p5.profile import load_aws_gpu_profile

profile = load_aws_gpu_profile(Path(os.environ["PROFILE"]))
report = load_qualification_receipt(
    Path(os.environ["QUALIFICATION_RECEIPT"]),
    profile,
)
print(report.as_dict())
PY
```

Do not proceed if any capability is false, a profile/hash differs, hardware is
mixed, an update is missing, or ETA misses the approved deadline/cost window.

After all six named 29M diagnostic receipts have independently passed, create
one affirmative, instance-specific protected-launch receipt. The diagnostic
names are closed; substitutions or omissions fail:

```bash
set -euo pipefail
export QUALIFICATION_RECEIPT
test -n "${MSCTL_APPROVAL_KEY:?set the protected receipt HMAC key}"
READINESS="$OPERATOR_ROOT/receipts/readiness-${INSTANCE_ID}.json"
READINESS_REVIEWER=REPLACE_WITH_REVIEWED_IDENTITY
READINESS_REVIEWED_AT=REPLACE_WITH_UTC_TIMESTAMP
READINESS_KEY_ID=REPLACE_WITH_HMAC_KEY_ID
DIAGNOSTIC_ARGS=(
  --diagnostic-receipt "full_corpus_dense=$OPERATOR_ROOT/receipts/diagnostic-full-corpus-dense.json"
  --diagnostic-receipt "full_corpus_split90=$OPERATOR_ROOT/receipts/diagnostic-full-corpus-split90.json"
  --diagnostic-receipt "no_arc_conceptarc_dense=$OPERATOR_ROOT/receipts/diagnostic-no-arc-conceptarc-dense.json"
  --diagnostic-receipt "no_arc_conceptarc_split90=$OPERATOR_ROOT/receipts/diagnostic-no-arc-conceptarc-split90.json"
  --diagnostic-receipt "no_refinement_dense=$OPERATOR_ROOT/receipts/diagnostic-no-refinement-dense.json"
  --diagnostic-receipt "no_refinement_split90=$OPERATOR_ROOT/receipts/diagnostic-no-refinement-split90.json"
)

# DRY RUN: validates every artifact and renders the exact affirmative decision.
python -m msctl --profile "$PROFILE" --repo-root . \
  readiness create \
  --release "$RELEASE_RECEIPT" \
  --amendment configs/hardware-amendment-v3.json \
  --provider-selection "$SELECTION" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --qualification-receipt "$QUALIFICATION_RECEIPT" \
  "${DIAGNOSTIC_ARGS[@]}" \
  --sealed-evaluation-fixture "$SEALED_FIXTURE_ROOT" \
  --reviewer "$READINESS_REVIEWER" \
  --reviewed-at "$READINESS_REVIEWED_AT" \
  --key-id "$READINESS_KEY_ID" \
  --out "$READINESS" \
  > "$REVIEW_ROOT/readiness-plan-${INSTANCE_ID}.json"

# APPLY writes once and refuses replacement.
python -m msctl --profile "$PROFILE" --repo-root . \
  readiness create \
  --release "$RELEASE_RECEIPT" \
  --amendment configs/hardware-amendment-v3.json \
  --provider-selection "$SELECTION" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --qualification-receipt "$QUALIFICATION_RECEIPT" \
  "${DIAGNOSTIC_ARGS[@]}" \
  --sealed-evaluation-fixture "$SEALED_FIXTURE_ROOT" \
  --reviewer "$READINESS_REVIEWER" \
  --reviewed-at "$READINESS_REVIEWED_AT" \
  --key-id "$READINESS_KEY_ID" \
  --out "$READINESS" \
  --apply > "$REVIEW_ROOT/readiness-result-${INSTANCE_ID}.json"
```

Each diagnostic receipt must also contain its relative `artifact_path`,
artifact SHA-256, HMAC `key_id`, and purpose-bound signature; readiness
rehashes the regular artifact and rejects unsigned or symlinked inputs.
The readiness receipt always contains `protected_launch_allowed: true`; the CLI
has no flag that can bypass validation to manufacture that decision. It binds
`sealed_fixture_sha256`, never a not-yet-created checkpoint or study-lock hash.
Do not edit the frozen preregistration, fixture, receipt, or any bound artifact
after review. Missing, false, stale, cross-profile, cross-instance, cross-boot,
unsigned, or artifact-mismatched evidence must fail locally before any paid
lifecycle AWS call.

## 10. Instantiate manifests and make the explicit fleet plan

For each seed, run `runs instantiate` first without and then with `--apply`.
The example shows seed 0; repeat identically for 0–9:

```bash
SEED=0
MANIFEST="$OPERATOR_ROOT/manifests/seed-${SEED}.json"
DATASET_RECEIPT="$OPERATOR_ROOT/receipts/dataset-receipt.json"

# DRY RUN: validates release, dataset, amendment, selection, and the
# checkpoint-independent sealed fixture without writing a manifest.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  runs instantiate \
  --release "$RELEASE_RECEIPT" \
  --dataset-receipt "$DATASET_RECEIPT" \
  --seed "$SEED" \
  --hardware-amendment configs/hardware-amendment-v3.json \
  --provider-selection "$SELECTION" \
  --sealed-evaluation-fixture "$SEALED_FIXTURE_ROOT" \
  --out "$MANIFEST" \
  > "$REVIEW_ROOT/manifest-plan-seed-${SEED}.json"

# APPLY only after the rendered manifest hashes are reviewed.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  runs instantiate \
  --release "$RELEASE_RECEIPT" \
  --dataset-receipt "$DATASET_RECEIPT" \
  --seed "$SEED" \
  --hardware-amendment configs/hardware-amendment-v3.json \
  --provider-selection "$SELECTION" \
  --sealed-evaluation-fixture "$SEALED_FIXTURE_ROOT" \
  --out "$MANIFEST" \
  --apply > "$REVIEW_ROOT/manifest-result-seed-${SEED}.json"
```

Each schema-3 manifest contains `sealed_fixture_sha256` and no
`sealed_evaluation_sha256` or `study_lock_sha256`. The latter two identities
are created only by post-training finalization.

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
  --manifest "$OPERATOR_ROOT/manifests/seed-0.json" \
  --manifest "$OPERATOR_ROOT/manifests/seed-1.json" \
  --manifest "$OPERATOR_ROOT/manifests/seed-2.json" \
  --manifest "$OPERATOR_ROOT/manifests/seed-3.json" \
  --manifest "$OPERATOR_ROOT/manifests/seed-4.json" \
  --manifest "$OPERATOR_ROOT/manifests/seed-5.json" \
  --manifest "$OPERATOR_ROOT/manifests/seed-6.json" \
  --manifest "$OPERATOR_ROOT/manifests/seed-7.json" \
  --manifest "$OPERATOR_ROOT/manifests/seed-8.json" \
  --manifest "$OPERATOR_ROOT/manifests/seed-9.json" \
  --instance-id REPLACE_WITH_EXPLICIT_INSTANCE_ID \
  --out "$OPERATOR_ROOT/fleet-plan-v3.json" \
  > "$REVIEW_ROOT/fleet-plan.json"
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
sealed fixture, qualification, six diagnostics, affirmative readiness, and
approval. Training does not require or accept a finalized checkpoint seal.
`msctl` sends work to an existing instance through SSM; it cannot provision
one.

For each fleet wave, render before submit:

```bash
MANIFEST="$OPERATOR_ROOT/manifests/seed-0.json"
INSTANCE_ID=REPLACE_WITH_FLEET_BOUND_INSTANCE_ID
ENVIRONMENT_RECEIPT="$OPERATOR_ROOT/receipts/environment-${INSTANCE_ID}.json"
QUALIFICATION_RECEIPT="$OPERATOR_ROOT/receipts/qualification-${INSTANCE_ID}.json"
READINESS="$OPERATOR_ROOT/receipts/readiness-${INSTANCE_ID}.json"
TERMINATE_AT=REPLACE_WITH_APPROVED_UTC_DEADLINE
DATASET_VERIFICATION="$OPERATOR_ROOT/receipts/dataset-verification.json"
FLEET_PLAN="$OPERATOR_ROOT/fleet-plan-v3.json"
V3_GATE_ARGS=(
  --hardware-amendment configs/hardware-amendment-v3.json
  --provider-selection "$SELECTION"
  --fleet-plan "$FLEET_PLAN"
  --launch-readiness "$READINESS"
  --qualification-receipt "$QUALIFICATION_RECEIPT"
  "${DIAGNOSTIC_ARGS[@]}"
)
TRAINING_GATE_ARGS=(
  "${V3_GATE_ARGS[@]}"
  --sealed-evaluation-fixture "$SEALED_FIXTURE_ROOT"
)

# DRY RUN: no SSM command is sent.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  --state-root "$OPERATOR_ROOT/state" \
  submit \
  --release "$RELEASE_RECEIPT" \
  --manifest "$MANIFEST" \
  --dataset-pointer DATASET-POINTER-AWS.json \
  --dataset-verification "$DATASET_VERIFICATION" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  "${TRAINING_GATE_ARGS[@]}" \
  --instance-id "$INSTANCE_ID" \
  --terminate-at "$TERMINATE_AT" \
  --approval "$OPERATOR_ROOT/approvals/submit-seed-0.json" \
  > "$REVIEW_ROOT/submit-plan-seed-0.json"

# APPLY only after argv, ID, wave, ETA, cost, and deadline review.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  --state-root "$OPERATOR_ROOT/state" \
  submit \
  --release "$RELEASE_RECEIPT" \
  --manifest "$MANIFEST" \
  --dataset-pointer DATASET-POINTER-AWS.json \
  --dataset-verification "$DATASET_VERIFICATION" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  "${TRAINING_GATE_ARGS[@]}" \
  --instance-id "$INSTANCE_ID" \
  --terminate-at "$TERMINATE_AT" \
  --approval "$OPERATOR_ROOT/approvals/submit-seed-0.json" \
  --apply > "$REVIEW_ROOT/submit-result-seed-0.json"
```

Poll with `status --cached` first, then the read-only authoritative status
path. Verify both arms advance together. The trainer’s `ckpt_minutes: 30`
creates local atomic generations every 30 minutes. Never upload checkpoints
with `s3 sync`, wildcards, or symlink-following tools. The v3 interruption and resume
publishers open one singly linked regular generation with `O_NOFOLLOW`,
snapshot and rehash exact bytes, and use checksum-bound `s3api put-object`
with `--if-none-match '*'`, the exact `MS_S3_KMS_KEY_ID`, and seed-scoped keys
`checkpoints/seed-<seed>/<arm>/sha256/<digest>.pt`. They independently HEAD and
verify size, SHA-256 metadata, version ID, and SSE-KMS identity. Local NVMe is
scratch; do not stop or terminate until the paired canonical receipt and both
objects are durably verified.

## 12. Resume, evaluate, and collect

Resume only from a verified paired checkpoint receipt:

```bash
CHECKPOINT_RECEIPT="$OPERATOR_ROOT/receipts/checkpoint-seed-0.json"

# DRY RUN.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  --state-root "$OPERATOR_ROOT/state" \
  resume \
  --release "$RELEASE_RECEIPT" \
  --manifest "$MANIFEST" \
  --checkpoint-receipt "$CHECKPOINT_RECEIPT" \
  --dataset-pointer DATASET-POINTER-AWS.json \
  --dataset-verification "$DATASET_VERIFICATION" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  "${TRAINING_GATE_ARGS[@]}" \
  --approval "$OPERATOR_ROOT/approvals/resume-seed-0.json" \
  > "$REVIEW_ROOT/resume-plan-seed-0.json"

# APPLY after checkpoint hashes, steps, world sizes, and explicit ID review.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  --state-root "$OPERATOR_ROOT/state" \
  resume \
  --release "$RELEASE_RECEIPT" \
  --manifest "$MANIFEST" \
  --checkpoint-receipt "$CHECKPOINT_RECEIPT" \
  --dataset-pointer DATASET-POINTER-AWS.json \
  --dataset-verification "$DATASET_VERIFICATION" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  "${TRAINING_GATE_ARGS[@]}" \
  --approval "$OPERATOR_ROOT/approvals/resume-seed-0.json" \
  --apply > "$REVIEW_ROOT/resume-result-seed-0.json"
```

The v3 checkpoint receipt must be canonical schema 3 and bind the same
cohort-assignment, preregistration, hardware-amendment, provider-selection,
profile, and sealed-fixture hashes as the manifest. Schema-2 receipts remain
valid only for legacy v2 manifests.

Completion publication must create one evaluator `run.json` per arm from the
actual terminal checkpoint and configuration bytes. Use
`evals.confirmatory.run_binding.build_run_binding(...)`, then
`write_run_binding(<run-root>/run.json, value)`. The helper hashes the files,
enforces the evaluator's closed field set and relative paths, and refuses to
replace existing evidence. The same publication step emits one canonical
`CheckpointRecord` for each arm; do not synthesize these records from mutable
“latest” pointers.

After all twenty checkpoint records and every registered passing validity
receipt are durable, build the complete release. The two inventory files below
must list reviewed, explicit paths in canonical seed/condition and receipt
order; no glob or directory scan is accepted:

```bash
set -euo pipefail
mapfile -t CHECKPOINT_RECORDS < "$REVIEW_ROOT/checkpoint-record-paths.txt"
mapfile -t VALIDITY_RECEIPTS < "$REVIEW_ROOT/validity-receipt-paths.txt"
test "${#CHECKPOINT_RECORDS[@]}" -eq 20
test "${#VALIDITY_RECEIPTS[@]}" -eq 21
CHECKPOINT_RECORD_ARGS=()
for path in "${CHECKPOINT_RECORDS[@]}"; do
  CHECKPOINT_RECORD_ARGS+=(--checkpoint-record "$path")
done
VALIDITY_RECEIPT_ARGS=()
for path in "${VALIDITY_RECEIPTS[@]}"; do
  VALIDITY_RECEIPT_ARGS+=(--validity-receipt "$path")
done

PREREGISTRATION_SHA256=REPLACE_WITH_V3_PREREGISTRATION_SHA256
SEALED_EVALUATION_ROOT="$OPERATOR_ROOT/sealed-evaluation-v3"
SEALED_FINALIZATION_PLAN="$REVIEW_ROOT/sealed-finalization-plan.json"
SEALED_FINALIZATION_RESULT="$REVIEW_ROOT/sealed-finalization-result.json"

# DRY RUN: validates the exact N=10 panel and renders all content roots.
python -m msctl --profile "$PROFILE" --repo-root . \
  sealed-evaluation finalize \
  --fixture "$SEALED_FIXTURE_ROOT" \
  "${CHECKPOINT_RECORD_ARGS[@]}" \
  "${VALIDITY_RECEIPT_ARGS[@]}" \
  --preregistration-sha256 "$PREREGISTRATION_SHA256" \
  --out "$SEALED_EVALUATION_ROOT" > "$SEALED_FINALIZATION_PLAN"

# APPLY: exclusively publishes the same six-member release.
python -m msctl --profile "$PROFILE" --repo-root . \
  sealed-evaluation finalize \
  --fixture "$SEALED_FIXTURE_ROOT" \
  "${CHECKPOINT_RECORD_ARGS[@]}" \
  "${VALIDITY_RECEIPT_ARGS[@]}" \
  --preregistration-sha256 "$PREREGISTRATION_SHA256" \
  --out "$SEALED_EVALUATION_ROOT" \
  --apply > "$SEALED_FINALIZATION_RESULT"

SEALED_EVALUATION_SHA256="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"]["sealed_evaluation_sha256"])' "$SEALED_FINALIZATION_RESULT")"
STUDY_LOCK_SHA256="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"]["study_lock_sha256"])' "$SEALED_FINALIZATION_RESULT")"
test "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"]["sealed_fixture_sha256"])' "$SEALED_FINALIZATION_RESULT")" = "$SEALED_FIXTURE_SHA256"
SEALED_S3_URI="${MS_S3_ROOT}/sealed-evaluation/${SEALED_EVALUATION_SHA256}"

SEALED_MEMBERS=(
  checkpoints.jsonl items.jsonl sealed-gold.jsonl stores.jsonl
  study-lock.json validity.json
)
for member in "${SEALED_MEMBERS[@]}"; do
  aws s3 cp "$SEALED_EVALUATION_ROOT/$member" "$SEALED_S3_URI/$member" \
    --region "$REGION" --sse aws:kms \
    --sse-kms-key-id "$MS_S3_KMS_KEY_ID" --dryrun
done
for member in "${SEALED_MEMBERS[@]}"; do
  aws s3 cp "$SEALED_EVALUATION_ROOT/$member" "$SEALED_S3_URI/$member" \
    --region "$REGION" --sse aws:kms \
    --sse-kms-key-id "$MS_S3_KMS_KEY_ID"
done

EVALUATION_GATE_ARGS=(
  "${V3_GATE_ARGS[@]}"
  --sealed-evaluation-release "$SEALED_EVALUATION_ROOT"
  --expected-sealed-evaluation-sha256 "$SEALED_EVALUATION_SHA256"
  --expected-study-lock-sha256 "$STUDY_LOCK_SHA256"
)
```

Finalization rejects missing, duplicate, random-arm, out-of-range, noncanonical,
or schema-drifted records. It produces exactly Dense/Split90 for seeds 0–9 and
proves that the finalized release retains the launch fixture root.

Evaluate only after the complete finalized release is durable and verified:

```bash
# DRY RUN: validates sealed-evaluation and lifecycle bindings.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  --state-root "$OPERATOR_ROOT/state" \
  evaluate \
  --release "$RELEASE_RECEIPT" \
  --manifest "$MANIFEST" \
  --dataset-pointer DATASET-POINTER-AWS.json \
  --dataset-verification "$DATASET_VERIFICATION" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  "${EVALUATION_GATE_ARGS[@]}" \
  --approval "$OPERATOR_ROOT/approvals/evaluate-seed-0.json" \
  > "$REVIEW_ROOT/evaluate-plan-seed-0.json"

# APPLY after sealed-release, expected study-lock hash, device, and output review.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  --state-root "$OPERATOR_ROOT/state" \
  evaluate \
  --release "$RELEASE_RECEIPT" \
  --manifest "$MANIFEST" \
  --dataset-pointer DATASET-POINTER-AWS.json \
  --dataset-verification "$DATASET_VERIFICATION" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  "${EVALUATION_GATE_ARGS[@]}" \
  --approval "$OPERATOR_ROOT/approvals/evaluate-seed-0.json" \
  --apply > "$REVIEW_ROOT/evaluate-result-seed-0.json"
```

Evaluation stages only
`$MS_S3_ROOT/sealed-evaluation/$SEALED_EVALUATION_SHA256`, revalidates the
external member root and actual `study-lock.json` bytes inside the digest-pinned
container, and invokes the evaluator with `/opt/venv/bin/python`. The training
preregistration hash is never substituted for the external study-lock hash.

Collect by exact source and exclusive output:

```bash
# DRY RUN.
python -m msctl \
  --profile "$PROFILE" \
  collect \
  --source results/seed-0.json \
  --out "$OPERATOR_ROOT/collected/seed-0.json" \
  > "$REVIEW_ROOT/collect-plan-seed-0.json"

# APPLY after source and destination review.
python -m msctl \
  --profile "$PROFILE" \
  collect \
  --source results/seed-0.json \
  --out "$OPERATOR_ROOT/collected/seed-0.json" \
  --apply > "$REVIEW_ROOT/collect-result-seed-0.json"
```

Repeat evaluate/collect for every seed and report all ten paired outcomes. For
a later fleet wave assigned to the same instance, first assemble the closed
collection directory and its `COLLECTION.json`, then explicitly advance:

```bash
NEXT_MANIFEST="$OPERATOR_ROOT/manifests/seed-1.json"
COLLECTION_ROOT="$OPERATOR_ROOT/collected/seed-0"
ADVANCE_APPROVAL="$OPERATOR_ROOT/approvals/fleet-advance-seed-0-to-1.json"

# DRY RUN: verifies paired local terminal state, successful evaluation, exact
# collection bytes, consecutive plan waves, and the proposed exact tag delete.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  --state-root "$OPERATOR_ROOT/state" \
  fleet advance \
  --amendment configs/hardware-amendment-v3.json \
  --provider-selection "$SELECTION" \
  --fleet-plan "$FLEET_PLAN" \
  --to-manifest "$NEXT_MANIFEST" \
  --instance-id "$INSTANCE_ID" \
  --collection-root "$COLLECTION_ROOT" \
  --approval "$ADVANCE_APPROVAL" \
  > "$REVIEW_ROOT/fleet-advance-plan-seed-0-to-1.json"

# APPLY only with a signed approval for the exact rendered resources.
python -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  --state-root "$OPERATOR_ROOT/state" \
  fleet advance \
  --amendment configs/hardware-amendment-v3.json \
  --provider-selection "$SELECTION" \
  --fleet-plan "$FLEET_PLAN" \
  --to-manifest "$NEXT_MANIFEST" \
  --instance-id "$INSTANCE_ID" \
  --collection-root "$COLLECTION_ROOT" \
  --approval "$ADVANCE_APPROVAL" \
  --apply > "$REVIEW_ROOT/fleet-advance-result-seed-0-to-1.json"
```

Apply rechecks authoritative training/evaluation SSM success, immutable terminal
receipt metadata, and the exact currently bound AWS tags before deleting those
exact key/value pairs. It then proves the tags absent and writes an exclusive
local transition receipt bound to the fleet plan, both waves, collection, and
approval. Manual or external tag deletion is never progression evidence and
cannot replace this receipt. One active pair per host remains mandatory.

## 13. Apply-time mutation inventory

Every apply in this runbook has a bounded mutation:

- package, provider-selection, readiness, manifest, fleet-plan, collection, and
  local state applies create exclusive files only under `$OPERATOR_ROOT`;
- image apply performs one local Docker build and one immutable-tag ECR push;
- Capacity Block purchase is the separately approved non-cancellable charge,
  and `run-instances` creates only the reviewed count and launch identity;
- control install conditionally creates one checksum-bound, SSE-KMS S3 object
  and sends one command through `AWS-RunShellScript`;
- the first lifecycle apply may create the fixed hash-checked custom SSM
  document, creates exact cohort tags, publishes a content-addressed operation
  intent, sends one SSM command, and installs an auto-shutdown deadline;
- first bootstrap apply creates/formats RAID0 on ephemeral instance store,
  creates fixed directories, syncs approved S3 inputs, extracts one immutable
  digest root, and publishes bootstrap/environment receipts; repeat applies
  only reverify and reuse the same identity;
- v3 checkpoint publication creates no-overwrite, SSE-KMS, seed-scoped S3
  objects after no-follow snapshots; evaluation materializes and validates the
  external sealed release and writes evaluation output;
- fleet advance deletes only the prior wave's exact reviewed tag key/value
  pairs after all terminal/evaluation/collection evidence is rechecked; and
- stop, start, terminate, session close, and eventual retention deletion are
  the explicit EC2/session/retention mutations shown below.

No dry-run command is launch authorization, and no apply may be inferred from a
read-only AWS response.

## 14. Stop, restart, and teardown

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
