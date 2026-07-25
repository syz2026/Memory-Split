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
instance role delivered through instance metadata. Run all blocks in one
operator shell so the reviewed variables and the `aws` function below persist.

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

FEDERATION_CONFIG="$OPERATOR_ROOT/operator-credential-process.ini"
FEDERATION_HELPER=/absolute/path/to/reviewed-federation-helper
export MSCTL_AWS_CONFIG_FILE="$FEDERATION_CONFIG"
export MSCTL_AWS_PROFILE=memorysplit-v3-operator
export MSCTL_AWS_CONFIG_SHA256=REPLACE_WITH_REVIEWED_CONFIG_SHA256
export MSCTL_AWS_CREDENTIAL_PROCESS_SHA256=REPLACE_WITH_REVIEWED_HELPER_SHA256
test "$(sha256sum "$MSCTL_AWS_CONFIG_FILE" | awk '{print $1}')" = \
  "$MSCTL_AWS_CONFIG_SHA256"
test "$(sha256sum "$FEDERATION_HELPER" | awk '{print $1}')" = \
  "$MSCTL_AWS_CREDENTIAL_PROCESS_SHA256"
python3 - "$REGION" <<'PY'
import os
import sys
from msctl.aws_p5 import load_operator_credential_process

load_operator_credential_process(os.environ, region=sys.argv[1])
PY
AWS_CLI="$(type -P aws)"
case "$AWS_CLI" in
  /*) ;;
  *) echo "AWS CLI must resolve to an absolute executable" >&2; exit 2 ;;
esac

# Every direct operator call uses only the reviewed credential_process.
# Shared credentials and operator-host instance metadata are disabled.
aws() {
  test "$(sha256sum "$MSCTL_AWS_CONFIG_FILE" | awk '{print $1}')" = \
    "$MSCTL_AWS_CONFIG_SHA256"
  test "$(sha256sum "$FEDERATION_HELPER" | awk '{print $1}')" = \
    "$MSCTL_AWS_CREDENTIAL_PROCESS_SHA256"
  env -i \
    AWS_CONFIG_FILE="$MSCTL_AWS_CONFIG_FILE" \
    AWS_PROFILE="$MSCTL_AWS_PROFILE" \
    AWS_SHARED_CREDENTIALS_FILE=/dev/null \
    AWS_EC2_METADATA_DISABLED=true \
    AWS_SDK_LOAD_CONFIG=1 \
    AWS_REGION="$REGION" \
    HOME=/tmp \
    PATH=/usr/local/bin:/usr/bin:/bin \
    "$AWS_CLI" --no-cli-pager "$@"
}

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

The reviewed config is a credential-process-only AWS CLI file with exactly this
shape and no default, source, role-chain, or shared-credential fallback:

```ini
[profile memorysplit-v3-operator]
credential_process = /absolute/path/to/reviewed-federation-helper
region = us-east-1
output = json
```

The helper must emit the AWS `credential_process` version-1 response containing
short-lived operator-role credentials directly to its AWS CLI child. Do not
`eval`, source, log, or export that response. `msctl` independently verifies the
config and helper hashes, invokes AWS CLI under `env -i`, and disables shared
credentials and operator-host instance metadata. The `credential_process` value
must be exactly one absolute executable path with no arguments. The helper must
therefore be self-contained under that minimal environment. Both paths must be
singly linked regular files owned by root or the operator, neither may be a
symlink or group/world-writable, and the helper must be executable. Re-review
both hashes whenever the config, helper, profile, role, or Region changes.

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
PACKAGE_PLAN="$(python3 scripts/package_aws_gpu_handoff.py \
  --profile "$PROFILE" \
  --out-dir "$OUT_ROOT")"
printf '%s\n' "$PACKAGE_PLAN" | python3 -m json.tool

# APPLY only after profile, release ID, and archive hash review.
PACKAGE_RESULT="$(python3 scripts/package_aws_gpu_handoff.py \
  --profile "$PROFILE" \
  --out-dir "$OUT_ROOT" \
  --apply)"
printf '%s\n' "$PACKAGE_RESULT" | python3 -m json.tool
```

Capturing stdout in shell variables keeps the clean source tree unchanged; do
not redirect either packaging report into the repository. Resolve
`RELEASE_DIR` from the reviewed JSON rather than a wildcard. Then verify:

```bash
set -euo pipefail
RELEASE_DIR="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["release_dir"])' <<<"$PACKAGE_RESULT")"
RELEASE_RECEIPT="$RELEASE_DIR/RELEASE-AWS-GPU-V3.json"

python3 scripts/verify_aws_gpu_v3_release.py \
  --release "$RELEASE_RECEIPT" \
  --profile "$PROFILE" \
  --source-root . > "$REVIEW_ROOT/release-verification.json"
python3 -m json.tool "$REVIEW_ROOT/release-verification.json"
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
python3 -m json.tool "$REVIEW_ROOT/dlami-review.json"
sha256sum "$REVIEW_ROOT/dlami-review.json"

AMI_EVIDENCE="$OPERATOR_ROOT/receipts/ami-describe-v1.json"
python3 - "$REGION" "$AMI_ID" \
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
python3 scripts/build_aws_gpu_image.py \
  --destination "$ECR_REPOSITORY" > "$IMAGE_PLAN"
python3 -m json.tool "$IMAGE_PLAN"

# APPLY only after base digest, context hash, immutable build tag, and
# destination repository are reviewed.
ECR_REGISTRY="${ECR_REPOSITORY%%/*}"
DOCKER_CONFIG="$OPERATOR_ROOT/docker-config"
export DOCKER_CONFIG
install -d -m 0700 "$DOCKER_CONFIG"
aws ecr get-login-password --region "$REGION" |
  docker login --username AWS --password-stdin "$ECR_REGISTRY"
trap 'docker logout "$ECR_REGISTRY" >/dev/null 2>&1 || true' EXIT
python3 scripts/build_aws_gpu_image.py \
  --destination "$ECR_REPOSITORY" \
  --apply > "$IMAGE_RESULT"
docker logout "$ECR_REGISTRY"
trap - EXIT
```

The login is performed only after the dry-run build/push argv and target
registry are reviewed. It stores no password in command arguments, and logout
removes the short-lived token from the dedicated operator Docker config. After
the push, use read-only ECR calls to resolve and record the manifest
digest. The only runtime form is the private
`repository@sha256:<manifest-digest>` reference; a tag alone is forbidden.

```bash
IMAGE_TAG="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["destination"].rsplit(":", 1)[1])' "$IMAGE_RESULT")"
aws ecr describe-images \
  --region "$REGION" \
  --repository-name REPLACE_WITH_REPOSITORY_NAME \
  --image-ids imageTag="$IMAGE_TAG" \
  --output json > "$REVIEW_ROOT/ecr-image-review.json"
python3 -m json.tool "$REVIEW_ROOT/ecr-image-review.json"
```

Set `CONTAINER_DIGEST` from the reviewed `imageDigest`, and set
`CONTAINER_IMAGE` to the same private repository plus `@CONTAINER_DIGEST`.
Then create the closed evidence files consumed by provider selection:

```bash
set -euo pipefail
CONTAINER_DIGEST="$(python3 -c 'import json,sys; rows=json.load(open(sys.argv[1]))["imageDetails"]; assert len(rows)==1; print(rows[0]["imageDigest"])' "$REVIEW_ROOT/ecr-image-review.json")"
CONTAINER_IMAGE="${ECR_REPOSITORY}@${CONTAINER_DIGEST}"
ECR_EVIDENCE="$OPERATOR_ROOT/receipts/ecr-describe-v1.json"
IMAGE_BUILD_RECEIPT="$OPERATOR_ROOT/receipts/image-build-v1.json"

python3 - "$REGION" "$ECR_REPOSITORY" "$CONTAINER_IMAGE" \
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
  : "${CAPACITY_RESERVATION_ID:?complete section 6 and retain its exact ID}"
  : "${CAPACITY_BLOCK_OFFERING_ID:?retain the exact approved offering ID}"
  CAPACITY_ARGS=(
    --capacity-reservation-id "$CAPACITY_RESERVATION_ID"
    --capacity-block-offering-id "$CAPACITY_BLOCK_OFFERING_ID"
  )
fi

# DRY RUN: validates but does not write the receipt.
python3 -m msctl \
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
python3 -m json.tool "$REVIEW_ROOT/provider-selection-plan.json"

# APPLY only after every hash and selected profile is reviewed.
python3 -m msctl \
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
python3 -m json.tool "$REVIEW_ROOT/capacity-block-offerings.json"
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
python3 -m json.tool "$REVIEW_ROOT/capacity-block-purchase.json"
CAPACITY_RESERVATION_ID="$(
  python3 - "$REVIEW_ROOT/capacity-block-purchase.json" <<'PY'
import sys
from scripts.validate_aws_gpu_launch_request import extract_capacity_reservation_id

print(extract_capacity_reservation_id(sys.argv[1]))
PY
)"
export CAPACITY_RESERVATION_ID
```

The returned `CapacityReservation.CapacityReservationId` is the only
reservation ID permitted in the launch request and provider-selection receipt;
never copy an ID from discovery or another purchase. Launch one P6 into that
exact block using a reviewed `launch-request.json`.
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

The complete request must contain only the common closed fields enforced by
`validate_aws_gpu_launch_request.py`, plus the two Capacity Block fields shown
above. Set every expected identity explicitly and validate the whole request
before either AWS call:

```bash
LAUNCH_REQUEST="$OPERATOR_ROOT/launch-request.json"
export MS_AWS_INSTANCE_PROFILE_ARN=arn:aws:iam::REPLACE_WITH_ACCOUNT:instance-profile/REPLACE_WITH_DEDICATED_PROFILE
PRIVATE_SUBNET_ID=subnet-REPLACE_WITH_PRIVATE_SUBNET
SECURITY_GROUP_ID=sg-REPLACE_WITH_PRIVATE_SECURITY_GROUP
EBS_KMS_KEY_ID=arn:aws:kms:REPLACE_WITH_REGION:REPLACE_WITH_ACCOUNT:key/REPLACE_WITH_KEY_UUID
ROOT_DEVICE_NAME=/dev/sda1
ROOT_VOLUME_GIB=500
COHORT_ID=memorysplit-confirmatory-v3-360m-n10-aws

python3 scripts/validate_aws_gpu_launch_request.py \
  --request "$LAUNCH_REQUEST" \
  --profile "$PROFILE" \
  --region "$REGION" \
  --ami-id "$AMI_ID" \
  --instance-profile-arn "$MS_AWS_INSTANCE_PROFILE_ARN" \
  --subnet-id "$PRIVATE_SUBNET_ID" \
  --security-group-id "$SECURITY_GROUP_ID" \
  --ebs-kms-key-id "$EBS_KMS_KEY_ID" \
  --root-device-name "$ROOT_DEVICE_NAME" \
  --root-volume-gib "$ROOT_VOLUME_GIB" \
  --cohort-id "$COHORT_ID" \
  --capacity-reservation-id "$CAPACITY_RESERVATION_ID" \
  > "$REVIEW_ROOT/launch-request-validation.json"
python3 -m json.tool "$REVIEW_ROOT/launch-request-validation.json"
LAUNCH_REQUEST_SHA256="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["request_sha256"])' "$REVIEW_ROOT/launch-request-validation.json")"
test "$(sha256sum "$LAUNCH_REQUEST" | awk '{print $1}')" = \
  "$LAUNCH_REQUEST_SHA256"
sha256sum "$LAUNCH_REQUEST" "$REVIEW_ROOT/launch-request-validation.json"

# DRY RUN: permission and request review; expect DryRunOperation.
aws ec2 run-instances \
  --region "$REGION" \
  --cli-input-json "file://$LAUNCH_REQUEST" \
  --dry-run

# PAID APPLY only after the launch JSON hash and Capacity Block binding match
# the approval.
test "$(sha256sum "$LAUNCH_REQUEST" | awk '{print $1}')" = \
  "$LAUNCH_REQUEST_SHA256"
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
python3 -m json.tool "$REVIEW_ROOT/p5-price-review.json"
LAUNCH_REQUEST="$OPERATOR_ROOT/launch-request.json"
export MS_AWS_INSTANCE_PROFILE_ARN=arn:aws:iam::REPLACE_WITH_ACCOUNT:instance-profile/REPLACE_WITH_DEDICATED_PROFILE
PRIVATE_SUBNET_ID=subnet-REPLACE_WITH_PRIVATE_SUBNET
SECURITY_GROUP_ID=sg-REPLACE_WITH_PRIVATE_SECURITY_GROUP
EBS_KMS_KEY_ID=arn:aws:kms:REPLACE_WITH_REGION:REPLACE_WITH_ACCOUNT:key/REPLACE_WITH_KEY_UUID
ROOT_DEVICE_NAME=/dev/sda1
ROOT_VOLUME_GIB=500
COHORT_ID=memorysplit-confirmatory-v3-360m-n10-aws

# Closed local validation: count must be 1-4 and Capacity Block fields are
# forbidden for the On-Demand profile.
python3 scripts/validate_aws_gpu_launch_request.py \
  --request "$LAUNCH_REQUEST" \
  --profile "$PROFILE" \
  --region "$REGION" \
  --ami-id "$AMI_ID" \
  --instance-profile-arn "$MS_AWS_INSTANCE_PROFILE_ARN" \
  --subnet-id "$PRIVATE_SUBNET_ID" \
  --security-group-id "$SECURITY_GROUP_ID" \
  --ebs-kms-key-id "$EBS_KMS_KEY_ID" \
  --root-device-name "$ROOT_DEVICE_NAME" \
  --root-volume-gib "$ROOT_VOLUME_GIB" \
  --cohort-id "$COHORT_ID" \
  > "$REVIEW_ROOT/launch-request-validation.json"
python3 -m json.tool "$REVIEW_ROOT/launch-request-validation.json"
LAUNCH_REQUEST_SHA256="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["request_sha256"])' "$REVIEW_ROOT/launch-request-validation.json")"
test "$(sha256sum "$LAUNCH_REQUEST" | awk '{print $1}')" = \
  "$LAUNCH_REQUEST_SHA256"
sha256sum "$LAUNCH_REQUEST" "$REVIEW_ROOT/p5-price-review.json" \
  "$REVIEW_ROOT/launch-request-validation.json"

# DRY RUN: expect DryRunOperation.
aws ec2 run-instances \
  --region "$REGION" \
  --cli-input-json "file://$LAUNCH_REQUEST" \
  --dry-run

# PAID APPLY only after exact hourly price, count, and launch request approval.
test "$(sha256sum "$LAUNCH_REQUEST" | awk '{print $1}')" = \
  "$LAUNCH_REQUEST_SHA256"
aws ec2 run-instances \
  --region "$REGION" \
  --cli-input-json "file://$LAUNCH_REQUEST" \
  --no-dry-run > "$REVIEW_ROOT/run-instances-result.json"
```

Extract only the returned IDs, write them in lexicographic order, and review
that exact set against `DescribeInstances`. Never use tag queries, “all running
instances,” or implicit discovery in a mutating command:

```bash
RUN_INSTANCES_RESULT="$REVIEW_ROOT/run-instances-result.json"
INSTANCE_IDS_FILE="$OPERATOR_ROOT/instance-ids.txt"
test "$(sha256sum "$LAUNCH_REQUEST" | awk '{print $1}')" = \
  "$LAUNCH_REQUEST_SHA256"
python3 - "$RUN_INSTANCES_RESULT" "$LAUNCH_REQUEST" \
  "$LAUNCH_REQUEST_SHA256" "$INSTANCE_IDS_FILE" <<'PY'
import pathlib
import sys
from scripts.validate_aws_gpu_launch_request import extract_run_instance_ids

result_path, request_path, request_sha256, out_path = sys.argv[1:]
identifiers = extract_run_instance_ids(
    result_path,
    request_path=request_path,
    expected_request_sha256=request_sha256,
)
pathlib.Path(out_path).write_text(
    "".join(f"{value}\n" for value in identifiers),
    encoding="ascii",
)
PY
mapfile -t INSTANCE_IDS < "$INSTANCE_IDS_FILE"
aws ec2 describe-instances \
  --region "$REGION" \
  --instance-ids "${INSTANCE_IDS[@]}" \
  --output json > "$REVIEW_ROOT/launched-instances-review.json"
python3 -m json.tool "$REVIEW_ROOT/launched-instances-review.json"
sha256sum "$INSTANCE_IDS_FILE" "$REVIEW_ROOT/launched-instances-review.json"
```

## 7. Stage immutable inputs

The instance role, not operator credentials, reads the cohort-scoped encrypted
S3 prefix and private ECR image. Upload only verified package artifacts and
receipts:

```bash
set -euo pipefail
export MS_S3_ROOT=s3://REPLACE_WITH_COHORT_BUCKET/REPLACE_WITH_COHORT_PREFIX
export MS_S3_KMS_KEY_ID=arn:aws:kms:REPLACE_WITH_REGION:REPLACE_WITH_ACCOUNT:key/REPLACE_WITH_KEY_UUID
RELEASE_ARCHIVE="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["archive"])' <<<"$PACKAGE_RESULT")"
RELEASE_SHA256="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["sha256"])' <<<"$PACKAGE_RESULT")"
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
python3 - "$SEALED_FIXTURE_ROOT" > "$SEALED_FIXTURE_REPORT" <<'PY'
import json
import sys
from msctl.aws_sealed_evaluation import load_sealed_evaluation_fixture

fixture = load_sealed_evaluation_fixture(sys.argv[1])
print(json.dumps({
    "sealed_fixture_sha256": fixture.sha256,
    "members": dict(fixture.members),
}, sort_keys=True, separators=(",", ":")))
PY
SEALED_FIXTURE_SHA256="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sealed_fixture_sha256"])' "$SEALED_FIXTURE_REPORT")"
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

Publish the separately verified dataset receipt and every receipt-listed member
under the exact prefix that bootstrap syncs to `/mnt/memorysplit/dataset`.
First freeze and review a local upload manifest; it rejects extra files,
symlinks, path traversal, byte-count drift, and hash drift:

```bash
set -euo pipefail
DATASET_ROOT=REPLACE_WITH_VERIFIED_EXTERNAL_DATASET_ROOT
DATASET_RECEIPT="$DATASET_ROOT/receipt.json"
DATASET_UPLOAD_MANIFEST="$REVIEW_ROOT/dataset-upload-manifest.tsv"
python3 - "$DATASET_ROOT" "$DATASET_UPLOAD_MANIFEST" <<'PY'
import hashlib
import json
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
out = pathlib.Path(sys.argv[2])
receipt_path = root / "receipt.json"
assert receipt_path.resolve(strict=True) == receipt_path
receipt_status = receipt_path.stat(follow_symlinks=False)
assert stat.S_ISREG(receipt_status.st_mode)
assert receipt_status.st_nlink == 1
receipt_bytes = receipt_path.read_bytes()
receipt = json.loads(receipt_bytes)
artifacts = receipt.get("artifacts")
assert isinstance(artifacts, list) and artifacts
rows = []
seen = set()
for row in artifacts:
    assert isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"}
    relative = row["path"]
    assert isinstance(relative, str) and relative not in seen
    parts = pathlib.PurePosixPath(relative).parts
    assert parts and not pathlib.PurePosixPath(relative).is_absolute()
    assert all(part not in {"", ".", ".."} for part in parts)
    assert not any(character in relative for character in "\\\t\n\r")
    member = root.joinpath(*parts)
    assert member.resolve(strict=True).is_relative_to(root)
    status = member.stat(follow_symlinks=False)
    assert stat.S_ISREG(status.st_mode) and status.st_nlink == 1
    payload = member.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    assert len(payload) == row["bytes"] and digest == row["sha256"]
    rows.append((relative, digest))
    seen.add(relative)
assert "receipt.json" not in seen
actual = set()
for path in root.rglob("*"):
    status = path.stat(follow_symlinks=False)
    assert not path.is_symlink()
    assert stat.S_ISDIR(status.st_mode) or stat.S_ISREG(status.st_mode)
    if stat.S_ISREG(status.st_mode):
        actual.add(path.relative_to(root).as_posix())
assert actual == seen | {"receipt.json"}
rows.append(("receipt.json", hashlib.sha256(receipt_bytes).hexdigest()))
out.write_text(
    "".join(f"{relative}\t{digest}\n" for relative, digest in sorted(rows)),
    encoding="ascii",
)
PY
python3 - "$DATASET_UPLOAD_MANIFEST" <<'PY'
import pathlib
import re
import sys

rows = pathlib.Path(sys.argv[1]).read_text(encoding="ascii").splitlines()
assert rows and rows == sorted(rows)
for row in rows:
    relative, digest = row.split("\t")
    assert relative and re.fullmatch(r"[0-9a-f]{64}", digest)
PY
sha256sum "$DATASET_UPLOAD_MANIFEST"
DATASET_RECEIPT_SHA256="$(
  awk -F $'\t' '$1 == "receipt.json" {print $2}' \
    "$DATASET_UPLOAD_MANIFEST"
)"
test -n "$DATASET_RECEIPT_SHA256"
```

Render every exact source and bootstrap destination first. The apply loop uses
checksum-bound `put-object` with `If-None-Match: *`, so no existing dataset key
can be replaced, and then verifies checksum metadata and SSE-KMS identity:

```bash
DATASET_BUCKET="$(python3 -c 'from urllib.parse import urlsplit; import os; print(urlsplit(os.environ["MS_S3_ROOT"]).netloc)')"
DATASET_PREFIX="$(python3 -c 'from urllib.parse import urlsplit; import os; print(urlsplit(os.environ["MS_S3_ROOT"]).path.strip("/"))')"

# DRY RUN: no object is written.
while IFS=$'\t' read -r relative digest; do
  aws s3 cp "$DATASET_ROOT/$relative" \
    "${MS_S3_ROOT}/dataset/${relative}" \
    --region "$REGION" \
    --sse aws:kms \
    --sse-kms-key-id "$MS_S3_KMS_KEY_ID" \
    --metadata "sha256=$digest" \
    --checksum-algorithm SHA256 \
    --dryrun
done < "$DATASET_UPLOAD_MANIFEST"

# APPLY only after the complete upload manifest and dry-run paths are approved.
while IFS=$'\t' read -r relative digest; do
  source="$DATASET_ROOT/$relative"
  key="${DATASET_PREFIX}/dataset/${relative}"
  checksum="$(
    python3 -c 'import base64,sys; print(base64.b64encode(bytes.fromhex(sys.argv[1])).decode("ascii"))' \
      "$digest"
  )"
  aws s3api put-object \
    --region "$REGION" \
    --bucket "$DATASET_BUCKET" \
    --key "$key" \
    --body "$source" \
    --checksum-algorithm SHA256 \
    --checksum-sha256 "$checksum" \
    --metadata "sha256=$digest" \
    --server-side-encryption aws:kms \
    --ssekms-key-id "$MS_S3_KMS_KEY_ID" \
    --if-none-match '*' \
    > "$REVIEW_ROOT/dataset-put-${digest}.json"
  aws s3api head-object \
    --region "$REGION" \
    --bucket "$DATASET_BUCKET" \
    --key "$key" \
    --checksum-mode ENABLED \
    --output json > "$REVIEW_ROOT/dataset-head-${digest}.json"
  python3 - "$REVIEW_ROOT/dataset-head-${digest}.json" \
    "$checksum" "$digest" "$MS_S3_KMS_KEY_ID" <<'PY'
import json
import sys

path, checksum, digest, kms_key = sys.argv[1:]
value = json.load(open(path, encoding="utf-8"))
assert value["ChecksumSHA256"] == checksum
assert value["Metadata"] == {"sha256": digest}
assert value["ServerSideEncryption"] == "aws:kms"
assert value["SSEKMSKeyId"] == kms_key
PY
done < "$DATASET_UPLOAD_MANIFEST"
```

Keep the raw review response and every generated receipt under
`$OPERATOR_ROOT`, never in the source checkout.

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
export MS_S3_ROOT="${MS_S3_ROOT:?set the reviewed cohort S3 prefix}"
export MS_S3_KMS_KEY_ID="${MS_S3_KMS_KEY_ID:?set the reviewed KMS key ARN}"
export MS_AWS_INSTANCE_PROFILE_ARN="${MS_AWS_INSTANCE_PROFILE_ARN:?set the dedicated instance profile ARN}"
export MS_RUNTIME_UID=10001
export MS_RUNTIME_GID=10001
INSTANCE_ID="${INSTANCE_ID:?set one explicit reviewed instance ID}"
CONTROL_BUNDLE="$OPERATOR_ROOT/control-bundle.tar"

# DRY RUN and exclusive local APPLY; both derive byte-identical tar bytes.
python3 -m msctl --profile "$PROFILE" --repo-root . \
  control bundle --out "$CONTROL_BUNDLE" \
  > "$REVIEW_ROOT/control-bundle-plan.json"
python3 -m msctl --profile "$PROFILE" --repo-root . \
  control bundle --out "$CONTROL_BUNDLE" --apply \
  > "$REVIEW_ROOT/control-bundle-result.json"
CONTROL_BUNDLE_PLAN_SHA256="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"]["bundle_sha256"])' "$REVIEW_ROOT/control-bundle-plan.json")"
CONTROL_BUNDLE_SHA256="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"]["bundle_sha256"])' "$REVIEW_ROOT/control-bundle-result.json")"
test "$CONTROL_BUNDLE_PLAN_SHA256" = "$CONTROL_BUNDLE_SHA256"
test "$(sha256sum "$CONTROL_BUNDLE" | awk '{print $1}')" = \
  "$CONTROL_BUNDLE_SHA256"

# DRY RUN: verifies the reviewed file/hash and renders immutable S3 publication
# plus the stock-document command.
python3 -m msctl --profile "$PROFILE" --repo-root . \
  --state-root "$OPERATOR_ROOT/state" \
  control install \
  --instance-id "$INSTANCE_ID" \
  --bundle "$CONTROL_BUNDLE" \
  --bundle-sha256 "$CONTROL_BUNDLE_SHA256" \
  > "$REVIEW_ROOT/control-install-plan-${INSTANCE_ID}.json"
test "$(
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"]["control_bundle_sha256"])' \
    "$REVIEW_ROOT/control-install-plan-${INSTANCE_ID}.json"
)" = "$CONTROL_BUNDLE_SHA256"

# APPLY: publishes with SSE-KMS/no-overwrite, verifies the object, then sends
# exactly one AWS-RunShellScript command to the explicit instance.
python3 -m msctl --profile "$PROFILE" --repo-root . \
  --state-root "$OPERATOR_ROOT/state" \
  control install \
  --instance-id "$INSTANCE_ID" \
  --bundle "$CONTROL_BUNDLE" \
  --bundle-sha256 "$CONTROL_BUNDLE_SHA256" \
  --apply \
  > "$REVIEW_ROOT/control-install-result-${INSTANCE_ID}.json"
CONTROL_COMMAND_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"]["command_id"])' "$REVIEW_ROOT/control-install-result-${INSTANCE_ID}.json")"
test "$(
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"]["control_bundle_sha256"])' \
    "$REVIEW_ROOT/control-install-result-${INSTANCE_ID}.json"
)" = "$CONTROL_BUNDLE_SHA256"

# Read-only completion check. Do not proceed on Pending/InProgress/Failed.
aws ssm get-command-invocation \
  --region "$REGION" \
  --instance-id "$INSTANCE_ID" \
  --command-id "$CONTROL_COMMAND_ID" \
  --query '{CommandId:CommandId,Status:Status}' \
  --output json > "$REVIEW_ROOT/control-install-status-${INSTANCE_ID}.json"
test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["Status"])' "$REVIEW_ROOT/control-install-status-${INSTANCE_ID}.json")" = Success
```

The installed root is
`/opt/memorysplit/control/$CONTROL_BUNDLE_SHA256`. On the explicit instance,
bootstrap inspects authenticated IMDSv2 identity and exact hardware before
rendering any mutation. Use the digest root, never `/opt/memorysplit` or an
unverified checkout. Set `PROFILE_BASENAME` to the basename of the one selected
profile (use `aws-p5.48xlarge-v3.json` for the P5 path), and copy every reviewed
hash exactly:

```bash
set -euo pipefail
export AWS_REGION=REPLACE_WITH_REVIEWED_REGION
export MS_AWS_AMI_ID=REPLACE_WITH_REVIEWED_AMI_ID
export MS_CONTAINER_IMAGE=REPLACE_WITH_PRIVATE_REPOSITORY_AT_DIGEST
export MS_CONTAINER_DIGEST=sha256:REPLACE_WITH_64_HEX_DIGEST
export MS_S3_ROOT=s3://REPLACE_WITH_COHORT_BUCKET/REPLACE_WITH_COHORT_PREFIX
export MS_S3_KMS_KEY_ID=arn:aws:kms:REPLACE_WITH_REGION:REPLACE_WITH_ACCOUNT:key/REPLACE_WITH_KEY_UUID
export MS_AWS_INSTANCE_PROFILE_ARN=arn:aws:iam::REPLACE_WITH_ACCOUNT:instance-profile/REPLACE_WITH_DEDICATED_PROFILE
export MS_RUNTIME_UID=10001
export MS_RUNTIME_GID=10001
PROFILE_BASENAME=aws-p6-b300.48xlarge-v3.json
CONTROL_BUNDLE_SHA256=REPLACE_WITH_REVIEWED_CONTROL_BUNDLE_SHA256
RELEASE_SHA256=REPLACE_WITH_REVIEWED_RELEASE_ARCHIVE_SHA256
DATASET_RECEIPT_SHA256=REPLACE_WITH_REVIEWED_DATASET_RECEIPT_SHA256
CONTROL_ROOT="/opt/memorysplit/control/${CONTROL_BUNDLE_SHA256}"
REMOTE_PROFILE="$CONTROL_ROOT/cluster/profiles/$PROFILE_BASENAME"
RELEASE_RECEIPT_SHA256=REPLACE_WITH_REVIEWED_RELEASE_RECEIPT_SHA256
COHORT_SHA256=REPLACE_WITH_REVIEWED_COHORT_ASSIGNMENT_SHA256
SOURCE_COMMIT=REPLACE_WITH_REVIEWED_40_HEX_SOURCE_COMMIT
sudo /usr/bin/install -d -m 0700 -o 0 -g 0 /run/memorysplit-aws

# Read-only digest lookup, then authenticated private-ECR pull using only the
# instance role. The Docker token lives under /run and is removed after pull.
ECR_REGISTRY="${MS_CONTAINER_IMAGE%%/*}"
ECR_REPOSITORY_NAME="${MS_CONTAINER_IMAGE#*/}"
ECR_REPOSITORY_NAME="${ECR_REPOSITORY_NAME%@*}"
sudo /usr/bin/install -d -m 0700 -o 0 -g 0 \
  /run/memorysplit-ecr-aws /run/memorysplit-ecr-docker
sudo /usr/bin/env -i \
  AWS_REGION="$AWS_REGION" \
  HOME=/run/memorysplit-ecr-aws \
  PATH=/usr/local/bin:/usr/bin:/bin \
  aws ecr batch-get-image \
  --region "$AWS_REGION" \
  --repository-name "$ECR_REPOSITORY_NAME" \
  --image-ids "imageDigest=$MS_CONTAINER_DIGEST" \
  --output json
printf 'sudo docker pull %q\n' "$MS_CONTAINER_IMAGE"
sudo /usr/bin/env -i \
  AWS_REGION="$AWS_REGION" \
  HOME=/run/memorysplit-ecr-aws \
  PATH=/usr/local/bin:/usr/bin:/bin \
  aws ecr get-login-password --region "$AWS_REGION" |
  sudo /usr/bin/env -i \
    DOCKER_CONFIG=/run/memorysplit-ecr-docker \
    PATH=/usr/local/bin:/usr/bin:/bin \
    /usr/bin/docker login \
    --username AWS --password-stdin "$ECR_REGISTRY"
sudo /usr/bin/env \
  DOCKER_CONFIG=/run/memorysplit-ecr-docker \
  /usr/bin/docker pull "$MS_CONTAINER_IMAGE"
sudo /usr/bin/env \
  DOCKER_CONFIG=/run/memorysplit-ecr-docker \
  /usr/bin/docker logout "$ECR_REGISTRY"
sudo /usr/bin/docker image inspect \
  --format '{{json .RepoDigests}}' "$MS_CONTAINER_IMAGE" |
  /usr/bin/python3 -c \
    'import json,sys; assert sys.argv[1] in json.load(sys.stdin)' \
    "$MS_CONTAINER_IMAGE"

BOOTSTRAP_ARGS=(
  "$CONTROL_ROOT/cluster/aws/p5/bootstrap.py"
  --profile "$REMOTE_PROFILE"
  --container-image "$MS_CONTAINER_IMAGE"
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
CANARY_APPROVAL="$OPERATOR_ROOT/approvals/canary-${INSTANCE_ID}.json"

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

# APPLY only after reviewing and signing the exact approval_resources object,
# instance, plan digest, and immutable argv from the dry run.
python3 -m msctl --profile "$PROFILE" --repo-root . \
  canary run \
  --canary-plan "$CANARY_PLAN" \
  --instance-id "$INSTANCE_ID" \
  --approval "$CANARY_APPROVAL" \
  --apply > "$REVIEW_ROOT/canary-run-result-${INSTANCE_ID}.json"

CANARY_COMMAND_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"]["command_id"])' "$REVIEW_ROOT/canary-run-result-${INSTANCE_ID}.json")"
QUALIFICATION_RECEIPT_URI="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"]["intent"]["qualification_receipt_uri"])' "$REVIEW_ROOT/canary-run-result-${INSTANCE_ID}.json")"
test "$QUALIFICATION_RECEIPT_URI" = "$(
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"]["qualification_receipt_uri"])' \
    "$REVIEW_ROOT/canary-plan-result-${INSTANCE_ID}.json"
)"

# Read-only wait and terminal status verification.
aws ssm wait command-executed \
  --region "$REGION" \
  --instance-id "$INSTANCE_ID" \
  --command-id "$CANARY_COMMAND_ID"
aws ssm get-command-invocation \
  --region "$REGION" \
  --instance-id "$INSTANCE_ID" \
  --command-id "$CANARY_COMMAND_ID" \
  --query '{CommandId:CommandId,Status:Status}' \
  --output json > "$REVIEW_ROOT/canary-status-${INSTANCE_ID}.json"
test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["Status"])' "$REVIEW_ROOT/canary-status-${INSTANCE_ID}.json")" = Success

# Download from the exact URI returned by the bound canary result.
export QUALIFICATION_RECEIPT="$OPERATOR_ROOT/receipts/qualification-${INSTANCE_ID}.json"
aws s3 cp "$QUALIFICATION_RECEIPT_URI" "$QUALIFICATION_RECEIPT" \
  --region "$REGION" \
  --checksum-mode ENABLED \
  --dryrun
aws s3 cp "$QUALIFICATION_RECEIPT_URI" "$QUALIFICATION_RECEIPT" \
  --region "$REGION" \
  --checksum-mode ENABLED
sha256sum "$QUALIFICATION_RECEIPT"
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
python3 - <<'PY'
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
python3 -m msctl --profile "$PROFILE" --repo-root . \
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
python3 -m msctl --profile "$PROFILE" --repo-root . \
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
: "${DATASET_RECEIPT:?use the exact verified receipt staged in section 7}"

# DRY RUN: validates release, dataset, amendment, selection, and the
# checkpoint-independent sealed fixture without writing a manifest.
python3 -m msctl \
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
python3 -m msctl \
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
python3 -m msctl \
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
For the P5 fallback, supply at most four distinct IDs. `msctl` canonicalizes
them in lexicographic ID order before round-robin assignment, regardless of
argument order. Four IDs deterministically produce 3/3/2/2 pairs: seeds 0/4/8,
1/5/9, 2/6, and 3/7. Fewer P5 IDs remain round-robin and sequential per
instance. Never run two pairs concurrently on one instance.

After review, repeat the exact fleet command with `--apply`. Do not alter the
ID set between review and publication; argument order has no semantic effect.

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
python3 -m msctl \
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

# On the reviewer-controlled signing host, sign only the dry run's exact
# result.approval_resources. MSCTL_APPROVAL_KEYS is a JSON map from key ID to
# a secret of at least 32 bytes; never put it in the checkout.
APPROVAL="$OPERATOR_ROOT/approvals/submit-seed-0.json"
APPROVAL_OPERATION=submit
APPROVAL_KEY_ID=REPLACE_WITH_REVIEWER_KEY_ID
APPROVAL_EXPIRES_AT=REPLACE_WITH_RFC3339_UTC_EXPIRY
: "${MSCTL_APPROVAL_KEYS:?export the reviewer-controlled key map}"
umask 077
python3 - "$REVIEW_ROOT/submit-plan-seed-0.json" "$APPROVAL" \
  "$APPROVAL_OPERATION" "$APPROVAL_KEY_ID" "$APPROVAL_EXPIRES_AT" <<'PY'
import hashlib
import hmac
import json
import os
import sys

from msctl.jsonutil import canonical_json

plan_path, out_path, operation, key_id, expires_at = sys.argv[1:]
with open(plan_path, "rb") as source:
    report = json.load(source)
resources = report["result"]["approval_resources"]
if not isinstance(resources, dict) or resources.get("operation") != operation:
    raise SystemExit("dry run lacks exact approval_resources")
keys = json.loads(os.environ["MSCTL_APPROVAL_KEYS"])
secret = keys[key_id].encode("utf-8")
if len(secret) < 32:
    raise SystemExit("approval key is too short")
unsigned = {
    "schema_version": 1,
    "receipt_id": (
        f"{operation}-{hashlib.sha256(canonical_json(resources)).hexdigest()}"
    ),
    "provider": resources["provider"],
    "operation": operation,
    "release_sha256": resources["release_sha256"],
    "run_manifest_sha256": resources["run_manifest_sha256"],
    "resources": resources,
    "limits": {
        "gpu_hours": resources["gpu_hours"],
        "jobs": resources["jobs"],
    },
    "expires_at": expires_at,
    "key_id": key_id,
}
receipt = {
    **unsigned,
    "signature": hmac.new(
        secret,
        canonical_json(unsigned),
        hashlib.sha256,
    ).hexdigest(),
}
with open(out_path, "xb") as destination:
    destination.write(canonical_json(receipt) + b"\n")
PY

# APPLY only after argv, ID, wave, ETA, cost, and deadline review.
python3 -m msctl \
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
  --approval "$APPROVAL" \
  --apply > "$REVIEW_ROOT/submit-result-seed-0.json"
```

The signed `resources` value must exactly equal the dry run's
`result.approval_resources`; `canonical_json` above defines its signed bytes.
Do not reconstruct or edit it. Repeat the same reviewer-controlled signing flow
for `resume`, `evaluate`, and every other apply after generating that command's
fresh dry-run report. A changed checkpoint hash, deadline, instance, lifecycle
receipt, sealed release, or study lock requires a new dry run and signature.

Poll with `status --cached` first, then the read-only authoritative status
path. Verify both arms advance together. The trainer’s `ckpt_minutes: 30`
creates local atomic generations every 30 minutes, each accompanied by a
canonical `checkpoint-meta.json`. Never upload checkpoints with `s3 sync`,
wildcards, or symlink-following tools. Normal completion, resumed completion,
and interruption publication open one singly linked regular generation with
`O_NOFOLLOW`, snapshot and rehash exact bytes, and use checksum-bound
`s3api put-object` with `--if-none-match '*'`, the exact
`MS_S3_KMS_KEY_ID`, and seed-scoped keys
`checkpoints/seed-<seed>/<arm>/sha256/<digest>.pt`. They independently HEAD and
verify size, SHA-256 metadata, and SSE-KMS identity.

Successful paired completion additionally requires both metadata records to
say `terminal: true` at the configured final step. It snapshots and immutably
publishes both `ckpt.pt` files, both `configuration.yaml` files, both evaluator
`run.json` bindings, and both checkpoint records. One canonical schema-v3 pair
receipt binds every object URI and SHA-256 at
`checkpoints/seed-<seed>/receipts/<receipt-sha256>.json`. A successful
interruption instead publishes the nonterminal schema-v3 resume form from its
stabilized checkpoint pair; it has no terminal run bindings or checkpoint
records and is valid for `resume`, never for training-wave advance or
evaluation. Local NVMe is scratch; do not stop or terminate until the relevant
paired receipt and all receipt-listed objects are durably verified.

There is no prelaunch evaluator or checkpoint-hash binding. At terminal
completion the publisher derives `model_id` and raw token count from the
verified config, route-dose SHA-256 from the config-selected verified corpus
sidecar, corpus SHA-256 from the ordered stream, code SHA-256 from the
authenticated release-member inventory, and checkpoint SHA-256 from the
no-follow snapshot. It writes a canonical `CheckpointRecord` from those exact
values and immutably publishes it under
`checkpoints/seed-<seed>/<arm>/records/<record-sha256>.json`. Interrupted,
nonterminal checkpoints publish only the fixture-bound resume receipt; they
are not finalization records.

## 12. Resume, advance training waves, finalize, evaluate, and collect

Retrieve every checkpoint receipt by its reviewed content hash into the private
operator root, then validate its canonical bytes and complete manifest
provenance before either resume or evaluation. Never use a mutable “latest”
key, `aws s3 sync`, or an unverified local copy:

```bash
set -euo pipefail
umask 077
SEED=0
CHECKPOINT_RECEIPT_SHA256=REPLACE_WITH_LAUNCHER_RECEIPT_SHA256
# Set this to evaluate for the final durable terminal bundle.
CHECKPOINT_PURPOSE=resume
[[ "$SEED" =~ ^(0|[1-9][0-9]*)$ ]]
[[ "$CHECKPOINT_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$CHECKPOINT_PURPOSE" == resume || "$CHECKPOINT_PURPOSE" == evaluate ]]
[[ "$MS_S3_ROOT" == s3://* ]]
CHECKPOINT_RECEIPT="$OPERATOR_ROOT/receipts/checkpoint-seed-${SEED}.json"
CHECKPOINT_TMP="$(mktemp "$OPERATOR_ROOT/receipts/.checkpoint-${SEED}.XXXXXX")"
CHECKPOINT_GET_REPORT="$REVIEW_ROOT/checkpoint-get-seed-${SEED}.json"
S3_LOCATION="${MS_S3_ROOT#s3://}"
if [[ "$S3_LOCATION" == */* ]]; then
  S3_BUCKET="${S3_LOCATION%%/*}"
  S3_PREFIX="${S3_LOCATION#*/}"
else
  S3_BUCKET="$S3_LOCATION"
  S3_PREFIX=
fi
CHECKPOINT_KEY="${S3_PREFIX:+${S3_PREFIX}/}checkpoints/seed-${SEED}/receipts/${CHECKPOINT_RECEIPT_SHA256}.json"
trap 'rm -f "$CHECKPOINT_TMP"' EXIT

aws s3api get-object \
  --region "$REGION" \
  --bucket "$S3_BUCKET" \
  --key "$CHECKPOINT_KEY" \
  --checksum-mode ENABLED \
  "$CHECKPOINT_TMP" > "$CHECKPOINT_GET_REPORT"

python3 - "$CHECKPOINT_TMP" "$CHECKPOINT_RECEIPT" \
  "$RELEASE_RECEIPT" "$MANIFEST" "$CHECKPOINT_RECEIPT_SHA256" \
  "$CHECKPOINT_PURPOSE" <<'PY'
import os
import stat
import sys

from msctl.contracts import (
    load_release,
    load_run_manifest,
    verify_checkpoint_receipt,
)

temporary, destination, release_path, manifest_path, expected, purpose = (
    sys.argv[1:]
)
if purpose not in {"resume", "evaluate"}:
    raise SystemExit("checkpoint purpose must be resume or evaluate")
release = load_release(release_path)
manifest = load_run_manifest(manifest_path, repo_root=".")
receipt = verify_checkpoint_receipt(
    temporary,
    release=release,
    manifest=manifest,
    require_checkpoint_files=False,
    require_durable_terminal=(purpose == "evaluate"),
)
if receipt.sha256 != expected:
    raise SystemExit("checkpoint receipt SHA-256 mismatch")
with open(temporary, "rb") as source:
    payload = source.read()
try:
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
except FileExistsError:
    read_descriptor = os.open(
        destination,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(read_descriptor)
        with os.fdopen(read_descriptor, "rb", closefd=False) as existing:
            existing_payload = existing.read(len(payload) + 1)
        after = os.fstat(read_descriptor)
    finally:
        os.close(read_descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or identity(before) != identity(after)
        or existing_payload != payload
    ):
        raise SystemExit("existing checkpoint receipt conflicts")
else:
    with os.fdopen(descriptor, "wb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
PY
rm -f "$CHECKPOINT_TMP"
trap - EXIT

# DRY RUN.
python3 -m msctl \
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
python3 -m msctl \
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
valid only for legacy v2 manifests. For an interrupted run, use the
`checkpoint_receipt` and `checkpoint_receipt_sha256` emitted by the paired
launcher; the resume path rejects a receipt whose run IDs, arm paths, hashes,
world sizes, or manifest provenance differ from its executable bindings.
Review and sign `resume-plan-seed-0.json` using its exact
`result.approval_resources` before the apply command.

Completion publication must create one evaluator `run.json` per arm from the
actual terminal checkpoint and configuration bytes. Use
`evals.confirmatory.run_binding.build_run_binding(...)` to hash the files and
enforce the evaluator's closed field set and relative paths, then publish those
canonical bytes immutably. The same publication step emits one canonical
`CheckpointRecord` for each arm; do not synthesize these records from mutable
“latest” pointers.

Advance an instance to its next assigned training wave as soon as the prior
pair has successful terminal training and its complete terminal bundle is
durable. Evaluation and collection do not exist yet and are not inputs. Never
assume that the successor is `seed + 1`: the same-instance successor to seed 0
is seed 1 for one instance, seed 2 for two instances, and seed 4 for four
instances. Derive it from the reviewed fleet plan every time:

```bash
CURRENT_SEED="$SEED"
NEXT_SEED="$(python3 - "$FLEET_PLAN" "$CURRENT_SEED" "$INSTANCE_ID" <<'PY'
import json
import sys

plan_path, current_seed_text, instance_id = sys.argv[1:]
with open(plan_path, "rb") as source:
    plan = json.load(source)
current_seed = int(current_seed_text)
rows = plan.get("manifests")
if not isinstance(rows, list):
    raise SystemExit("fleet plan has no manifest schedule")
current = [
    row
    for row in rows
    if row.get("seed") == current_seed and row.get("instance_id") == instance_id
]
if len(current) != 1 or type(current[0].get("wave")) is not int:
    raise SystemExit("completed seed is not uniquely bound to this instance")
successors = [
    row
    for row in rows
    if row.get("instance_id") == instance_id
    and row.get("wave") == current[0]["wave"] + 1
]
if len(successors) != 1 or type(successors[0].get("seed")) is not int:
    raise SystemExit("fleet plan has no unique same-instance successor")
print(successors[0]["seed"])
PY
)"
NEXT_MANIFEST="$OPERATOR_ROOT/manifests/seed-${NEXT_SEED}.json"
CHECKPOINT_RECEIPT="$OPERATOR_ROOT/receipts/checkpoint-seed-${CURRENT_SEED}.json"
ADVANCE_APPROVAL="$OPERATOR_ROOT/approvals/fleet-advance-seed-${CURRENT_SEED}-to-${NEXT_SEED}.json"

# DRY RUN: verifies paired local terminal state, the canonical durable
# schema-v3 checkpoint receipt/records, consecutive plan waves, and the exact
# proposed tag deletion. It does not evaluate or collect.
python3 -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  --state-root "$OPERATOR_ROOT/state" \
  fleet advance \
  --amendment configs/hardware-amendment-v3.json \
  --provider-selection "$SELECTION" \
  --fleet-plan "$FLEET_PLAN" \
  --to-manifest "$NEXT_MANIFEST" \
  --instance-id "$INSTANCE_ID" \
  --checkpoint-receipt "$CHECKPOINT_RECEIPT" \
  --approval "$ADVANCE_APPROVAL" \
  > "$REVIEW_ROOT/fleet-advance-plan-seed-${CURRENT_SEED}-to-${NEXT_SEED}.json"

# APPLY only with signed approval for these exact resources.
python3 -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  --state-root "$OPERATOR_ROOT/state" \
  fleet advance \
  --amendment configs/hardware-amendment-v3.json \
  --provider-selection "$SELECTION" \
  --fleet-plan "$FLEET_PLAN" \
  --to-manifest "$NEXT_MANIFEST" \
  --instance-id "$INSTANCE_ID" \
  --checkpoint-receipt "$CHECKPOINT_RECEIPT" \
  --approval "$ADVANCE_APPROVAL" \
  --apply > "$REVIEW_ROOT/fleet-advance-result-seed-${CURRENT_SEED}-to-${NEXT_SEED}.json"
```

Apply rechecks authoritative training SSM success, rematerializes the receipt,
both checkpoints, both configurations, both `run.json` files, and both
checkpoint records from their immutable S3 URIs, verifies the complete bundle,
then deletes only the prior wave's exact reviewed tags. It proves those tags
absent and writes an exclusive closed
`memorysplit-aws-training-wave-advance-v3` receipt. Manual or external tag
deletion is never progression evidence.

For each target wave, create the required transition receipt for every instance
participating in that wave before submitting any target pair. This all-peer
gate applies unchanged to 1–4 instance fleets, including a shorter final wave.
Repeat train then advance until all ten pairs are terminal. Submit and resume
continue to require the target pair's current exact tags; only post-training
evaluation may use a matching historical advance receipt after prior tags were
removed.

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
python3 -m msctl --profile "$PROFILE" --repo-root . \
  sealed-evaluation finalize \
  --fixture "$SEALED_FIXTURE_ROOT" \
  "${CHECKPOINT_RECORD_ARGS[@]}" \
  "${VALIDITY_RECEIPT_ARGS[@]}" \
  --preregistration-sha256 "$PREREGISTRATION_SHA256" \
  --out "$SEALED_EVALUATION_ROOT" > "$SEALED_FINALIZATION_PLAN"

# APPLY: exclusively publishes the same six-member release.
python3 -m msctl --profile "$PROFILE" --repo-root . \
  sealed-evaluation finalize \
  --fixture "$SEALED_FIXTURE_ROOT" \
  "${CHECKPOINT_RECORD_ARGS[@]}" \
  "${VALIDITY_RECEIPT_ARGS[@]}" \
  --preregistration-sha256 "$PREREGISTRATION_SHA256" \
  --out "$SEALED_EVALUATION_ROOT" \
  --apply > "$SEALED_FINALIZATION_RESULT"

SEALED_EVALUATION_SHA256="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"]["sealed_evaluation_sha256"])' "$SEALED_FINALIZATION_RESULT")"
STUDY_LOCK_SHA256="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"]["study_lock_sha256"])' "$SEALED_FINALIZATION_RESULT")"
test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"]["sealed_fixture_sha256"])' "$SEALED_FINALIZATION_RESULT")" = "$SEALED_FIXTURE_SHA256"
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
python3 -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  --state-root "$OPERATOR_ROOT/state" \
  evaluate \
  --release "$RELEASE_RECEIPT" \
  --manifest "$MANIFEST" \
  --checkpoint-receipt "$CHECKPOINT_RECEIPT" \
  --dataset-pointer DATASET-POINTER-AWS.json \
  --dataset-verification "$DATASET_VERIFICATION" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  "${EVALUATION_GATE_ARGS[@]}" \
  --approval "$OPERATOR_ROOT/approvals/evaluate-seed-0.json" \
  > "$REVIEW_ROOT/evaluate-plan-seed-0.json"

# APPLY after sealed-release, expected study-lock hash, device, and output review.
python3 -m msctl \
  --profile "$PROFILE" \
  --repo-root . \
  --state-root "$OPERATOR_ROOT/state" \
  evaluate \
  --release "$RELEASE_RECEIPT" \
  --manifest "$MANIFEST" \
  --checkpoint-receipt "$CHECKPOINT_RECEIPT" \
  --dataset-pointer DATASET-POINTER-AWS.json \
  --dataset-verification "$DATASET_VERIFICATION" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  "${EVALUATION_GATE_ARGS[@]}" \
  --approval "$OPERATOR_ROOT/approvals/evaluate-seed-0.json" \
  --apply > "$REVIEW_ROOT/evaluate-result-seed-0.json"
```

Evaluation stages
`$MS_S3_ROOT/sealed-evaluation/$SEALED_EVALUATION_SHA256` and the exact
receipt-addressed terminal bundle. It revalidates the external member root and
actual `study-lock.json` bytes inside the digest-pinned container. Before either
arm runs, it rematerializes from immutable S3 and verifies the checkpoint
receipt, both checkpoints, both configurations, both evaluator `run.json`
bindings, and both checkpoint records. It then invokes the evaluator with
`/opt/venv/bin/python`. This works after reboot, NVMe reuse, and prior-wave tag
removal because the historical advance receipt, local successful training
state, fleet binding, and explicit instance ID remain hash-bound. After both
arms succeed, it requires the exact closed confirmatory artifact inventory,
immutably uploads each file, and publishes a closed paired evaluation receipt at
`evaluations/seed-<seed>/receipts/<receipt-sha256>.json` and the canonical
collection source `results/seed-<seed>.json`. The training preregistration hash
is never substituted for the external study-lock hash.

Collect by exact source into one exclusive paired directory:

```bash
# DRY RUN.
python3 -m msctl \
  --profile "$PROFILE" \
  collect \
  --source results/seed-0.json \
  --out "$OPERATOR_ROOT/collected/seed-0" \
  > "$REVIEW_ROOT/collect-plan-seed-0.json"

# APPLY after source and destination review.
python3 -m msctl \
  --profile "$PROFILE" \
  collect \
  --source results/seed-0.json \
  --out "$OPERATOR_ROOT/collected/seed-0" \
  --apply > "$REVIEW_ROOT/collect-result-seed-0.json"
```

Apply downloads the paired checkpoint and evaluation receipts plus every
receipt-listed artifact into a staging directory, verifies S3 checksums,
content hashes, byte counts, canonical URIs, pair provenance, the
`study-lock.json` hash, the finalized six-member sealed-evaluation root, the
three-member launch-fixture root, and the exact inventory, then atomically
installs the directory with canonical `COLLECTION.json`. Repeating the command
accepts only byte-identical verified contents; a partial, extra, symlinked, or
conflicting destination fails closed.

Repeat evaluate/collect for every seed and report all ten paired outcomes.
These closed evaluation receipts and `COLLECTION.json` directories are a
distinct post-finalization completion gate. They do not retroactively authorize
training-wave progression and are never embedded in a training-wave advance
receipt. One active pair per host remains mandatory.

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
  pairs after terminal training and the complete durable checkpoint
  receipt/record bundle are rechecked; and
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
