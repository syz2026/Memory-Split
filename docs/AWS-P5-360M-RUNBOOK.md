# AWS P5 360M paired-cohort runbook

This runbook operates AWS seeds 1–4 of
`memorysplit-confirmatory-v2-360m-n5`. Illumina owns seed 0. Each AWS seed is
one simultaneous Dense/Split90 pair on one `p5.48xlarge`: GPUs 0–3 run Dense
and GPUs 4–7 run Split90. On-Demand is the default. Do not use Spot until the
paired interruption/resume canary has passed.

All mutations and paid actions are shown first as dry runs. A failed dry run,
hash check, receipt check, provider check, or readiness gate is a hard stop.
Never repair a release, manifest, receipt, or checkpoint in place.

## 1. Operator inputs and local preflight

Use an AWS CLI v2 SSO profile. Do not export static AWS access keys.

```bash
set -euo pipefail
umask 077

export AWS_PROFILE=memorysplit-p5-operator
export AWS_REGION=us-east-1
export MS_COHORT_ID=memorysplit-confirmatory-v2-360m-n5
export MS_PROVIDER=aws-p5.48xlarge
export MS_INSTANCE_TYPE=p5.48xlarge
export MS_BUCKET=REPLACE_WITH_GLOBALLY_UNIQUE_BUCKET
export MS_S3_ROOT="s3://${MS_BUCKET}/${MS_COHORT_ID}"
export MS_AWS_AMI_ID=ami-REPLACE_WITH_REGION_PINNED_DLAMI
export MS_CONTAINER_DIGEST="sha256:REPLACE_WITH_64_LOWERCASE_HEX"
export MS_SUBNET_ID=subnet-REPLACE
export MS_SECURITY_GROUP_ID=sg-REPLACE_NO_INBOUND
export MS_INSTANCE_PROFILE_NAME=memorysplit-p5-instance
export MS_RELEASE_ROOT="$PWD/../memorysplit-releases/aws-p5/aws-p5-r1-REPLACE_SUFFIX"
export MS_ILLUMINA_RELEASE="$PWD/dist/illumina/RELEASE.json"
export MS_AWS_RELEASE="$MS_RELEASE_ROOT/RELEASE-AWS-P5.json"
export MS_DATASET_RECEIPT="$PWD/dataset/corpus-receipt.json"
export MS_DATASET_POINTER="$PWD/DATASET-POINTER-AWS.json"
export MS_SEALED_RELEASE="$PWD/evaluation/SEALED-RELEASE.json"
export MS_APPROVAL_ROOT="$PWD/approvals/aws-p5"
export MS_STATE_ROOT="$PWD/.msctl-state/aws-p5"

file_sha256() {
  python - "$1" <<'PY'
from pathlib import Path
import hashlib
import sys

digest = hashlib.sha256()
with Path(sys.argv[1]).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
aws sso login --profile "$AWS_PROFILE"
aws sts get-caller-identity --output json | tee approvals/aws-identity.json
aws configure get region
aws --version
jq --version
python --version

git status --porcelain=v1
test -z "$(git status --porcelain=v1)"
test "$(git rev-parse HEAD)" = \
  "$(jq -r '.source.commit' "$MS_AWS_RELEASE")"

python scripts/verify_cohort_releases.py \
  --illumina "$MS_ILLUMINA_RELEASE" \
  --aws "$MS_AWS_RELEASE" \
  | tee approvals/cohort-verification.json
jq -e '
  .ok == true and
  .illumina.seeds == [0] and
  .aws.seeds == [1,2,3,4] and
  .complete_cohort == [0,1,2,3,4]
' approvals/cohort-verification.json >/dev/null

export MS_RELEASE_SHA256="$(
  jq -er '.archive.sha256' "$MS_AWS_RELEASE"
)"
export MS_ARCHIVE_NAME="$(
  jq -er '.archive.path' "$MS_AWS_RELEASE"
)"
export MS_CODE_COMMIT="$(
  jq -er '.source.commit' "$MS_AWS_RELEASE"
)"
export MS_RELEASE_RECEIPT_SHA256="$(file_sha256 "$MS_AWS_RELEASE")"
export MS_ASSIGNMENT_SHA256="$(
  jq -er '.cohort_assignment_sha256' approvals/cohort-verification.json
)"
export MS_DATASET_SHA256="$(file_sha256 "$MS_DATASET_RECEIPT")"
export MS_EVALUATION_SHA256="$(file_sha256 "$MS_SEALED_RELEASE")"
printf '%s\n' "$MS_CONTAINER_DIGEST" |
  grep -Eq '^sha256:[0-9a-f]{64}$'
```

Confirm the pinned AMI, private networking, no-inbound security group, and P5
offering. The AMI query must return one Amazon-owned, available, x86_64 image.

```bash
aws ec2 describe-images \
  --region "$AWS_REGION" \
  --owners amazon \
  --image-ids "$MS_AWS_AMI_ID" \
  --output json | tee approvals/ami.json
jq -e '
  (.Images | length) == 1 and
  .Images[0].State == "available" and
  .Images[0].Architecture == "x86_64"
' approvals/ami.json >/dev/null

aws ec2 describe-security-groups \
  --region "$AWS_REGION" \
  --group-ids "$MS_SECURITY_GROUP_ID" \
  --output json | tee approvals/security-group.json
jq -e '
  (.SecurityGroups | length) == 1 and
  (.SecurityGroups[0].IpPermissions | length) == 0
' approvals/security-group.json >/dev/null

aws ec2 describe-instance-type-offerings \
  --region "$AWS_REGION" \
  --location-type availability-zone \
  --filters "Name=instance-type,Values=p5.48xlarge" \
  --output json | tee approvals/p5-offerings.json
jq -e '.InstanceTypeOfferings | length > 0' \
  approvals/p5-offerings.json >/dev/null
```

## 2. Quota and current price

One instance needs 192 Running On-Demand P vCPUs; deadline mode needs 768.
Resolve the quota code by its AWS-owned name rather than hard-coding a code.

```bash
export MS_P_QUOTA_CODE="$(
  aws service-quotas list-service-quotas \
    --region "$AWS_REGION" \
    --service-code ec2 \
    --query \
      "Quotas[?QuotaName=='Running On-Demand P instances'].QuotaCode | [0]" \
    --output text
)"
test -n "$MS_P_QUOTA_CODE"
test "$MS_P_QUOTA_CODE" != None
aws service-quotas get-service-quota \
  --region "$AWS_REGION" \
  --service-code ec2 \
  --quota-code "$MS_P_QUOTA_CODE" \
  --output json | tee approvals/p-instance-quota.json
jq -e '.Quota.Value >= 192' approvals/p-instance-quota.json >/dev/null
```

Before choosing four-instance deadline mode, run the stronger gate:

```bash
jq -e '.Quota.Value >= 768' approvals/p-instance-quota.json >/dev/null
```

Query the live Linux/Shared/On-Demand price from the Price List API immediately
before approval. Pricing API calls go to `us-east-1`; `regionCode` selects the
compute region. Record the response, timestamp, and unique hourly USD value.

```bash
aws pricing get-products \
  --region us-east-1 \
  --service-code AmazonEC2 \
  --filters \
    "Type=TERM_MATCH,Field=instanceType,Value=p5.48xlarge" \
    "Type=TERM_MATCH,Field=regionCode,Value=${AWS_REGION}" \
    "Type=TERM_MATCH,Field=operatingSystem,Value=Linux" \
    "Type=TERM_MATCH,Field=tenancy,Value=Shared" \
    "Type=TERM_MATCH,Field=preInstalledSw,Value=NA" \
    "Type=TERM_MATCH,Field=capacitystatus,Value=Used" \
  --output json | tee approvals/p5-current-price.json

jq -r '
  [.PriceList[] | fromjson
   | .terms.OnDemand[]
   | .priceDimensions[]
   | select(.unit == "Hrs")
   | .pricePerUnit.USD] | unique | .[]
' approvals/p5-current-price.json | tee approvals/p5-usd-per-hour.txt
test "$(wc -l < approvals/p5-usd-per-hour.txt | tr -d ' ')" = 1
date -u +%Y-%m-%dT%H:%M:%SZ | tee approvals/p5-price-queried-at.txt
```

## 3. S3 durable boundary and instance role

Use this immutable layout:

```text
s3://$MS_BUCKET/$MS_COHORT_ID/
  releases/aws-p5/
  dataset/
  sealed-evaluation/
  checkpoints/seed-N/{dense,split90}/
  evaluations/seed-N/{dense,split90}/
  evidence/seed-N/
  receipts/
```

Bucket creation and controls are locally rendered first. For a bucket outside
`us-east-1`, add
`--create-bucket-configuration LocationConstraint="$AWS_REGION"`.

```bash
# DRY RUN: local AWS CLI validation, no API mutation.
aws s3api create-bucket \
  --region "$AWS_REGION" \
  --bucket "$MS_BUCKET" \
  --generate-cli-skeleton output >/dev/null

# APPLY once, after confirming the globally unique bucket name.
aws s3api create-bucket \
  --region "$AWS_REGION" \
  --bucket "$MS_BUCKET"

# DRY RUN.
aws s3api put-public-access-block \
  --region "$AWS_REGION" \
  --bucket "$MS_BUCKET" \
  --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true \
  --generate-cli-skeleton output >/dev/null
# APPLY.
aws s3api put-public-access-block \
  --region "$AWS_REGION" \
  --bucket "$MS_BUCKET" \
  --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

# DRY RUN.
aws s3api put-bucket-versioning \
  --region "$AWS_REGION" \
  --bucket "$MS_BUCKET" \
  --versioning-configuration Status=Enabled \
  --generate-cli-skeleton output >/dev/null
# APPLY.
aws s3api put-bucket-versioning \
  --region "$AWS_REGION" \
  --bucket "$MS_BUCKET" \
  --versioning-configuration Status=Enabled

# DRY RUN.
aws s3api put-bucket-encryption \
  --region "$AWS_REGION" \
  --bucket "$MS_BUCKET" \
  --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"aws:kms"}}]}' \
  --generate-cli-skeleton output >/dev/null
# APPLY.
aws s3api put-bucket-encryption \
  --region "$AWS_REGION" \
  --bucket "$MS_BUCKET" \
  --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"aws:kms"}}]}'
```

The EC2 instance profile must contain `AmazonSSMManagedInstanceCore` and a
least-privilege policy scoped to this cohort prefix. It may read releases,
dataset, and sealed evaluation; it may write only checkpoints, evaluations,
evidence, and receipts. It must not contain static credentials. Confirm it:

```bash
aws iam get-instance-profile \
  --instance-profile-name "$MS_INSTANCE_PROFILE_NAME" \
  --output json | tee approvals/instance-profile.json
jq -e '.InstanceProfile.Roles | length == 1' \
  approvals/instance-profile.json >/dev/null
export MS_INSTANCE_PROFILE_ARN="$(
  jq -er '.InstanceProfile.Arn' approvals/instance-profile.json
)"
export MS_AWS_INSTANCE_PROFILE_ARN="$MS_INSTANCE_PROFILE_ARN"
```

Upload only verified inputs. Each paid S3 transfer is listed first with
`--dryrun`.

```bash
# DRY RUN.
aws s3 sync "$MS_RELEASE_ROOT/" \
  "$MS_S3_ROOT/releases/aws-p5/" \
  --region "$AWS_REGION" --no-follow-symlinks --dryrun
# APPLY.
aws s3 sync "$MS_RELEASE_ROOT/" \
  "$MS_S3_ROOT/releases/aws-p5/" \
  --region "$AWS_REGION" --no-follow-symlinks

# DRY RUN: bootstrap independently authenticates the assignment member.
aws s3 cp configs/cohort-assignment-v2.json \
  "$MS_S3_ROOT/releases/aws-p5/configs/cohort-assignment-v2.json" \
  --region "$AWS_REGION" --dryrun
# APPLY.
aws s3 cp configs/cohort-assignment-v2.json \
  "$MS_S3_ROOT/releases/aws-p5/configs/cohort-assignment-v2.json" \
  --region "$AWS_REGION"

# DRY RUN.
aws s3 cp requirements-aws-p5.lock \
  "$MS_S3_ROOT/environment/requirements-aws-p5.lock" \
  --region "$AWS_REGION" --dryrun
# APPLY.
aws s3 cp requirements-aws-p5.lock \
  "$MS_S3_ROOT/environment/requirements-aws-p5.lock" \
  --region "$AWS_REGION"

# DRY RUN.
aws s3 sync "$(dirname "$MS_DATASET_RECEIPT")/" \
  "$MS_S3_ROOT/dataset/" \
  --region "$AWS_REGION" --no-follow-symlinks --dryrun
# APPLY only after the local corpus receipt rehashes every shard and sidecar.
aws s3 sync "$(dirname "$MS_DATASET_RECEIPT")/" \
  "$MS_S3_ROOT/dataset/" \
  --region "$AWS_REGION" --no-follow-symlinks

# DRY RUN.
aws s3 sync "$(dirname "$MS_SEALED_RELEASE")/" \
  "$MS_S3_ROOT/sealed-evaluation/" \
  --region "$AWS_REGION" --no-follow-symlinks --dryrun
# APPLY.
aws s3 sync "$(dirname "$MS_SEALED_RELEASE")/" \
  "$MS_S3_ROOT/sealed-evaluation/" \
  --region "$AWS_REGION" --no-follow-symlinks
```

Record immutable S3 version IDs for the release receipt, ZIP, corpus receipt,
and sealed release before launch:

```bash
aws s3api list-object-versions \
  --region "$AWS_REGION" \
  --bucket "$MS_BUCKET" \
  --prefix "$MS_COHORT_ID/" \
  --output json | tee approvals/input-object-versions.json
jq -e '.Versions | length >= 4' approvals/input-object-versions.json >/dev/null
```

## 4. Instantiate the four immutable pair manifests

Run this on the control host after the archive and corpus receipts exist.
Manifest creation is no-replace. Dry-run all four before writing any.

```bash
MSCTL=(
  python -m msctl
  --profile cluster/profiles/aws-p5.48xlarge.json
  --repo-root "$PWD"
  --state-root "$MS_STATE_ROOT"
)
mkdir -p run-manifests

for seed in 1 2 3 4; do
  # DRY RUN.
  "${MSCTL[@]}" runs instantiate \
    --release "$MS_AWS_RELEASE" \
    --dataset-receipt "$MS_DATASET_RECEIPT" \
    --seed "$seed" \
    --out "run-manifests/aws-p5-s${seed}.json"
done

for seed in 1 2 3 4; do
  # APPLY, atomic and no-replace.
  "${MSCTL[@]}" runs instantiate \
    --release "$MS_AWS_RELEASE" \
    --dataset-receipt "$MS_DATASET_RECEIPT" \
    --seed "$seed" \
    --out "run-manifests/aws-p5-s${seed}.json" \
    --apply
  "${MSCTL[@]}" runs render \
    --release "$MS_AWS_RELEASE" \
    --manifest "run-manifests/aws-p5-s${seed}.json" \
    --dataset-pointer "$MS_DATASET_POINTER" \
    --dataset-root "$(dirname "$MS_DATASET_RECEIPT")"
done
```

Each manifest must bind one seed, both explicit arms, world size 4, release,
dataset, cohort assignment, study lock, code commit, and both config hashes.

## 5. On-Demand launch: choose exactly one mode

The following function writes a launch request with no SSH key, no public IP,
IMDSv2 required, one immutable AMI, the SSM instance profile, and provenance
tags. The subnet must have NAT or VPC endpoints for S3, SSM, `ssmmessages`, and
`ec2messages`.

```bash
write_launch_spec() {
  local mode="$1" seed="$2" name="$3" token="$4" output="$5"
  local manifest="run-manifests/aws-p5-s${seed}.json"
  local manifest_sha256
  manifest_sha256="$(file_sha256 "$manifest")"
  jq -n \
    --arg ami "$MS_AWS_AMI_ID" \
    --arg profile "$MS_INSTANCE_PROFILE_NAME" \
    --arg subnet "$MS_SUBNET_ID" \
    --arg security_group "$MS_SECURITY_GROUP_ID" \
    --arg name "$name" \
    --arg mode "$mode" \
    --arg seed "$seed" \
    --arg cohort "$MS_COHORT_ID" \
    --arg release "$MS_RELEASE_SHA256" \
    --arg assignment "$MS_ASSIGNMENT_SHA256" \
    --arg dataset "$MS_DATASET_SHA256" \
    --arg evaluation "$MS_EVALUATION_SHA256" \
    --arg manifest "$manifest_sha256" \
    --arg token "$token" '
    {
      ImageId: $ami,
      InstanceType: "p5.48xlarge",
      MinCount: 1,
      MaxCount: 1,
      IamInstanceProfile: {Name: $profile},
      EbsOptimized: true,
      InstanceInitiatedShutdownBehavior: "stop",
      NetworkInterfaces: [{
        DeviceIndex: 0,
        SubnetId: $subnet,
        Groups: [$security_group],
        AssociatePublicIpAddress: false,
        DeleteOnTermination: true
      }],
      MetadataOptions: {
        HttpEndpoint: "enabled",
        HttpTokens: "required",
        HttpPutResponseHopLimit: 1,
        InstanceMetadataTags: "enabled"
      },
      TagSpecifications: [{
        ResourceType: "instance",
        Tags: [
          {Key: "Name", Value: $name},
          {Key: "MemorySplitCohort", Value: $cohort},
          {Key: "MemorySplitProvider", Value: "aws-p5.48xlarge"},
          {Key: "MemorySplitMode", Value: $mode},
          {Key: "MemorySplitSeed", Value: $seed},
          {Key: "MemorySplitCohortSHA256", Value: $assignment},
          {Key: "MemorySplitReleaseSHA256", Value: $release},
          {Key: "MemorySplitAssignmentSHA256", Value: $assignment},
          {Key: "MemorySplitDatasetSHA256", Value: $dataset},
          {Key: "MemorySplitEvaluationSHA256", Value: $evaluation},
          {Key: "MemorySplitRunManifestSHA256", Value: $manifest}
        ]
      }],
      ClientToken: $token
    }' > "$output"
}

ec2_launch_dry_run() {
  local spec="$1" output rc
  set +e
  output="$(
    aws ec2 run-instances \
      --region "$AWS_REGION" \
      --cli-input-json "file://${spec}" \
      --dry-run 2>&1
  )"
  rc=$?
  set -e
  test "$rc" -ne 0
  grep -q DryRunOperation <<<"$output"
}
```

### Economy mode: one instance, seeds 1–4 sequentially

```bash
mkdir -p approvals/launch
write_launch_spec \
  sequential 1 "memorysplit-p5-sequential" \
  "ms-n5-sequential-${MS_RELEASE_SHA256:0:16}" \
  approvals/launch/sequential.json

# DRY RUN. A successful authorization check is DryRunOperation.
ec2_launch_dry_run approvals/launch/sequential.json

# APPLY only after a signed approval covers the current hourly price and
# the complete sequential GPU-hour ceiling.
read -r -p 'Type APPLY-SEQUENTIAL-P5: ' answer
test "$answer" = APPLY-SEQUENTIAL-P5
export MS_SEQUENTIAL_INSTANCE_ID="$(
  aws ec2 run-instances \
    --region "$AWS_REGION" \
    --cli-input-json file://approvals/launch/sequential.json \
    --query 'Instances[0].InstanceId' \
    --output text
)"
printf '%s\n' "$MS_SEQUENTIAL_INSTANCE_ID" |
  tee approvals/sequential-instance-id.txt
```

Run seed 1 to collection before seed 2, then seed 3, then seed 4. Never
interleave two pairs on this instance.

### Deadline mode: four instances, one preassigned seed each

Dry-run all four launches before launching any instance.

```bash
mkdir -p approvals/launch
for seed in 1 2 3 4; do
  write_launch_spec \
    deadline "$seed" "memorysplit-p5-seed-${seed}" \
    "ms-n5-s${seed}-${MS_RELEASE_SHA256:0:16}" \
    "approvals/launch/seed-${seed}.json"
  # DRY RUN.
  ec2_launch_dry_run "approvals/launch/seed-${seed}.json"
done

# APPLY only after a signed approval covers 4 x the live hourly price and
# the complete deadline-mode GPU-hour ceiling.
read -r -p 'Type APPLY-FOUR-P5: ' answer
test "$answer" = APPLY-FOUR-P5
: > approvals/deadline-instances.tsv
for seed in 1 2 3 4; do
  instance_id="$(
    aws ec2 run-instances \
      --region "$AWS_REGION" \
      --cli-input-json "file://approvals/launch/seed-${seed}.json" \
      --query 'Instances[0].InstanceId' \
      --output text
  )"
  printf '%s\t%s\n' "$seed" "$instance_id" |
    tee -a approvals/deadline-instances.tsv
done
test "$(wc -l < approvals/deadline-instances.tsv | tr -d ' ')" = 4
```

## 6. SSM, bootstrap, and eight-device NVMe RAID0

There is no inbound SSH. Wait for SSM `PingStatus=Online`, then use Session
Manager.

```bash
export MS_INSTANCE_ID="$MS_SEQUENTIAL_INSTANCE_ID"  # economy mode
# Deadline mode instead selects the instance mapped to one seed:
# export MS_INSTANCE_ID="$(awk '$1 == 1 {print $2}' approvals/deadline-instances.tsv)"

for attempt in $(seq 1 60); do
  ping="$(
    aws ssm describe-instance-information \
      --region "$AWS_REGION" \
      --filters "Key=InstanceIds,Values=${MS_INSTANCE_ID}" \
      --query 'InstanceInformationList[0].PingStatus' \
      --output text
  )"
  test "$ping" = Online && break
  test "$attempt" -lt 60
  sleep 10
done
test "$ping" = Online
aws ssm start-session --region "$AWS_REGION" --target "$MS_INSTANCE_ID"
```

Inside the SSM session, export only non-secret trust roots. The shipped
bootstrap is dry-run by default; `--apply` is the only mutating form.

```bash
set -euo pipefail
umask 077
export AWS_REGION=us-east-1
export MS_S3_ROOT="s3://REPLACE_BUCKET/memorysplit-confirmatory-v2-360m-n5"
export MS_AWS_AMI_ID=ami-REPLACE_WITH_THE_PINNED_ID
export MS_CONTAINER_DIGEST="sha256:REPLACE_WITH_64_LOWERCASE_HEX"
export MS_CONTAINER_IMAGE="REPLACE_REGISTRY/REPLACE_IMAGE@${MS_CONTAINER_DIGEST}"
export MS_RELEASE_SHA256=REPLACE_WITH_64_LOWERCASE_HEX
export MS_RELEASE_RECEIPT_SHA256=REPLACE_WITH_64_LOWERCASE_HEX
export MS_ARCHIVE_NAME=ms-aws-p5-r1-REPLACE_SUFFIX.zip
export MS_CODE_COMMIT=REPLACE_WITH_40_LOWERCASE_HEX
export MS_ASSIGNMENT_SHA256=REPLACE_WITH_64_LOWERCASE_HEX
export MS_DATASET_SHA256=REPLACE_WITH_64_LOWERCASE_HEX
export MS_EVALUATION_SHA256=REPLACE_WITH_64_LOWERCASE_HEX
export MS_OWNER_UID="$(id -u)"
export MS_OWNER_GID="$(id -g)"

# DRY RUN: stage the signed AWS release on the root volume so its bootstrap
# entry point can be invoked before ephemeral NVMe exists.
export MS_BOOTSTRAP_INPUT=/opt/memorysplit-bootstrap-input
export MS_BOOTSTRAP_RELEASE=/opt/memorysplit-bootstrap-release
sudo install -d -m 0700 -o "$MS_OWNER_UID" -g "$MS_OWNER_GID" \
  "$MS_BOOTSTRAP_INPUT"
aws s3 cp \
  "$MS_S3_ROOT/releases/aws-p5/RELEASE-AWS-P5.json" \
  "$MS_BOOTSTRAP_INPUT/RELEASE-AWS-P5.json" \
  --region "$AWS_REGION" --dryrun
aws s3 cp \
  "$MS_S3_ROOT/releases/aws-p5/${MS_ARCHIVE_NAME}" \
  "$MS_BOOTSTRAP_INPUT/${MS_ARCHIVE_NAME}" \
  --region "$AWS_REGION" --dryrun
aws s3 cp \
  "$MS_S3_ROOT/releases/aws-p5/${MS_ARCHIVE_NAME}.sha256" \
  "$MS_BOOTSTRAP_INPUT/${MS_ARCHIVE_NAME}.sha256" \
  --region "$AWS_REGION" --dryrun

# APPLY the three exact downloads.
aws s3 cp \
  "$MS_S3_ROOT/releases/aws-p5/RELEASE-AWS-P5.json" \
  "$MS_BOOTSTRAP_INPUT/RELEASE-AWS-P5.json" \
  --region "$AWS_REGION"
aws s3 cp \
  "$MS_S3_ROOT/releases/aws-p5/${MS_ARCHIVE_NAME}" \
  "$MS_BOOTSTRAP_INPUT/${MS_ARCHIVE_NAME}" \
  --region "$AWS_REGION"
aws s3 cp \
  "$MS_S3_ROOT/releases/aws-p5/${MS_ARCHIVE_NAME}.sha256" \
  "$MS_BOOTSTRAP_INPUT/${MS_ARCHIVE_NAME}.sha256" \
  --region "$AWS_REGION"

test "$(
  sha256sum "$MS_BOOTSTRAP_INPUT/RELEASE-AWS-P5.json" | awk '{print $1}'
)" = "$MS_RELEASE_RECEIPT_SHA256"
(
  cd "$MS_BOOTSTRAP_INPUT"
  sha256sum -c "${MS_ARCHIVE_NAME}.sha256"
)
test "$(
  jq -er '.archive.sha256' \
    "$MS_BOOTSTRAP_INPUT/RELEASE-AWS-P5.json"
)" = "$MS_RELEASE_SHA256"

test ! -e "$MS_BOOTSTRAP_RELEASE"
python - "$MS_BOOTSTRAP_INPUT/$MS_ARCHIVE_NAME" \
  "$MS_BOOTSTRAP_RELEASE" <<'PY'
from pathlib import Path
import stat
import sys
import zipfile

archive_path, output = map(Path, sys.argv[1:])
with zipfile.ZipFile(archive_path) as archive:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise SystemExit("duplicate ZIP member")
    for info in infos:
        name = info.filename
        core = name[:-1] if info.is_dir() else name
        if (
            not core
            or name.startswith("/")
            or "\\" in name
            or any(part in {"", ".", ".."} for part in core.split("/"))
        ):
            raise SystemExit("unsafe ZIP member")
        mode = info.external_attr >> 16
        if info.is_dir() and not stat.S_ISDIR(mode):
            raise SystemExit("unsafe ZIP directory mode")
        if not info.is_dir() and not stat.S_ISREG(mode):
            raise SystemExit("unsafe ZIP file mode")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    archive.extractall(output)
PY
(
  cd "$MS_BOOTSTRAP_RELEASE"
  sha256sum -c SHA256SUMS
)
docker image inspect "$MS_CONTAINER_IMAGE" >/dev/null
cd "$MS_BOOTSTRAP_RELEASE"

TOKEN="$(
  curl -fsS -X PUT \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
    http://169.254.169.254/latest/api/token
)"
test "$(
  curl -fsS \
    -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-type
)" = p5.48xlarge

# DRY RUN: verifies IMDSv2, eight H100s, Fabric Manager, the immutable
# container, eight instance-store devices by model identity, and rendered
# S3/RAID0 argv. It does not mutate disks or download objects.
bash cluster/aws/p5/bootstrap.sh \
  --container-image "$MS_CONTAINER_IMAGE" \
  --release-archive \
    "/mnt/memorysplit/release/aws-p5/${MS_ARCHIVE_NAME}" \
  --release-sha256 "$MS_RELEASE_SHA256" \
  --release-receipt \
    /mnt/memorysplit/release/aws-p5/RELEASE-AWS-P5.json \
  --release-receipt-sha256 "$MS_RELEASE_RECEIPT_SHA256" \
  --dataset-receipt /mnt/memorysplit/dataset/corpus-receipt.json \
  --dataset-receipt-sha256 "$MS_DATASET_SHA256" \
  --cohort-assignment \
    /mnt/memorysplit/release/aws-p5/configs/cohort-assignment-v2.json \
  --cohort-assignment-sha256 "$MS_ASSIGNMENT_SHA256" \
  --code-commit "$MS_CODE_COMMIT" \
  --owner-uid "$MS_OWNER_UID" \
  --owner-gid "$MS_OWNER_GID" |
  tee /tmp/memorysplit-bootstrap-plan.json
jq -e '.ok == true and .dry_run == true' \
  /tmp/memorysplit-bootstrap-plan.json >/dev/null

# Independently inspect device discovery. Device names are never hard-coded.
mapfile -t MS_NVME_DEVICES < <(
  lsblk -dpno NAME,MODEL,TYPE |
    awk '$NF == "disk" &&
         index($0, "Amazon EC2 NVMe Instance Storage") {print $1}' |
    sort -V
)
test "${#MS_NVME_DEVICES[@]}" = 8
printf 'RAID0 plan:'
printf ' %q' sudo mdadm --create /dev/md/memorysplit \
  --level=0 --raid-devices=8 "${MS_NVME_DEVICES[@]}"
printf '\n'
mountpoint -q /mnt/memorysplit && exit 1

# APPLY after inspecting the bootstrap JSON and dynamic eight-device plan.
sudo --preserve-env=AWS_REGION,MS_S3_ROOT,MS_AWS_AMI_ID,MS_CONTAINER_DIGEST,MS_CONTAINER_IMAGE,MS_RELEASE_SHA256,MS_RELEASE_RECEIPT_SHA256,MS_ARCHIVE_NAME,MS_CODE_COMMIT,MS_ASSIGNMENT_SHA256,MS_DATASET_SHA256,MS_OWNER_UID,MS_OWNER_GID \
  bash cluster/aws/p5/bootstrap.sh \
  --container-image "$MS_CONTAINER_IMAGE" \
  --release-archive \
    "/mnt/memorysplit/release/aws-p5/${MS_ARCHIVE_NAME}" \
  --release-sha256 "$MS_RELEASE_SHA256" \
  --release-receipt \
    /mnt/memorysplit/release/aws-p5/RELEASE-AWS-P5.json \
  --release-receipt-sha256 "$MS_RELEASE_RECEIPT_SHA256" \
  --dataset-receipt /mnt/memorysplit/dataset/corpus-receipt.json \
  --dataset-receipt-sha256 "$MS_DATASET_SHA256" \
  --cohort-assignment \
    /mnt/memorysplit/release/aws-p5/configs/cohort-assignment-v2.json \
  --cohort-assignment-sha256 "$MS_ASSIGNMENT_SHA256" \
  --code-commit "$MS_CODE_COMMIT" \
  --owner-uid "$MS_OWNER_UID" \
  --owner-gid "$MS_OWNER_GID" \
  --apply |
  tee /tmp/memorysplit-bootstrap-receipt.json
jq -e '.ok == true and .dry_run == false' \
  /tmp/memorysplit-bootstrap-receipt.json >/dev/null

findmnt --noheadings --output TARGET,FSTYPE,SOURCE /mnt/memorysplit
test "$(stat -c '%a' /mnt/memorysplit)" = 700
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader |
  tee /tmp/memorysplit-gpus.txt
test "$(wc -l < /tmp/memorysplit-gpus.txt | tr -d ' ')" = 8
grep -c 'NVIDIA H100 80GB' /tmp/memorysplit-gpus.txt | grep -qx 8
systemctl is-active nvidia-fabricmanager

# DRY RUN: stage the already authenticated extracted release onto NVMe.
test ! -e /mnt/memorysplit/release/code
rsync -an --delete "$MS_BOOTSTRAP_RELEASE/" \
  /mnt/memorysplit/release/code/
# APPLY.
mkdir -m 0700 /mnt/memorysplit/release/code
rsync -a --delete "$MS_BOOTSTRAP_RELEASE/" \
  /mnt/memorysplit/release/code/
(
  cd /mnt/memorysplit/release/code
  sha256sum -c SHA256SUMS
)
```

`/mnt/memorysplit` is disposable scratch. No checkpoint, receipt, log, or
evidence bundle is complete until an independently hash-verified S3 object
version exists.

## 7. Functional canary and 100-update throughput gate

Canaries are non-claim-bearing. They use copied scratch configs and fresh
output directories; never alter the archived configs or production manifests.
Run this section inside the same SSM session. Create one-update and 100-update
copies from seed 1:

```bash
export MS_RELEASE_DIR=/mnt/memorysplit/release/code
export MS_DATASET_DIR=/mnt/memorysplit/dataset
export MS_CANARY_ROOT=/mnt/memorysplit/staging/canary
export MS_TORCHRUN_BIN="$(command -v torchrun)"
export MS_TASKSET_BIN="$(command -v taskset)"
test -x "$MS_TORCHRUN_BIN"
test -x "$MS_TASKSET_BIN"
mkdir -p "$MS_CANARY_ROOT"

python - "$MS_RELEASE_DIR" "$MS_DATASET_DIR" "$MS_CANARY_ROOT" <<'PY'
from pathlib import Path
import sys
import yaml

release, dataset, root = map(Path, sys.argv[1:])
for label, steps in (("functional", 1), ("throughput-100", 100)):
    target = root / label
    target.mkdir(mode=0o700, parents=True, exist_ok=False)
    for arm in ("dense", "split90"):
        source = release / "configs/360m-v2" / f"{arm}-s1.yaml"
        value = yaml.safe_load(source.read_text())
        value["run_id"] = f"canary-{label}-s1-{arm}"
        value["train_corpus"] = str(dataset / "corpus-receipt.json")
        value["out_dir"] = str(target / arm)
        value["max_steps"] = steps
        value["total_tokens"] = steps * value["tokens_per_step"]
        (target / f"{arm}.yaml").write_text(
            yaml.safe_dump(value, sort_keys=False)
        )
PY
```

Use this paired runner. `plan` prints the exact commands and does not start a
GPU process. `apply` uses symmetric CPU partitions and separate process groups.

```bash
run_canary_pair() {
  local action="$1" label="$2" world="$3"
  local root="$MS_CANARY_ROOT/$label"
  local dense_gpus split_gpus dense_port split_port
  if test "$world" = 1; then
    dense_gpus=0
    split_gpus=4
    dense_port=29600
    split_port=29610
  else
    test "$world" = 4
    dense_gpus=0,1,2,3
    split_gpus=4,5,6,7
    dense_port=29700
    split_port=29710
  fi
  case "$action" in
    plan|apply)
      test ! -e "$root/dense"
      test ! -e "$root/split90"
      ;;
    plan-resume|apply-resume)
      test -f "$root/dense/ckpt.pt"
      test -f "$root/split90/ckpt.pt"
      ;;
    *)
      return 2
      ;;
  esac

  local common_env=(
    env -i
    "HOME=$HOME"
    "PATH=/opt/memorysplit/bin:/usr/local/bin:/usr/bin:/bin"
    "MS_DATA_LOADER_WORKERS=32"
    "MS_DATA_ROOT=$MS_DATASET_DIR"
    "MS_RUN_ROOT=/mnt/memorysplit"
    "NCCL_ASYNC_ERROR_HANDLING=1"
    "NCCL_DEBUG=INFO"
    "OMP_NUM_THREADS=1"
    "PYTHONPATH=$MS_RELEASE_DIR"
    "PYTHONUNBUFFERED=1"
  )
  local dense=(
    "${common_env[@]}"
    "CUDA_VISIBLE_DEVICES=$dense_gpus"
    "$MS_TASKSET_BIN" -c 0-95
    "$MS_TORCHRUN_BIN" --standalone --nproc_per_node="$world"
    --master_port="$dense_port"
    "$MS_RELEASE_DIR/scripts/run_train.py" --config "$root/dense.yaml"
  )
  local split=(
    "${common_env[@]}"
    "CUDA_VISIBLE_DEVICES=$split_gpus"
    "$MS_TASKSET_BIN" -c 96-191
    "$MS_TORCHRUN_BIN" --standalone --nproc_per_node="$world"
    --master_port="$split_port"
    "$MS_RELEASE_DIR/scripts/run_train.py" --config "$root/split90.yaml"
  )

  if test "$action" = plan || test "$action" = plan-resume; then
    printf 'Dense:'
    printf ' %q' "${dense[@]}"
    printf '\nSplit90:'
    printf ' %q' "${split[@]}"
    printf '\n'
    return
  fi
  test "$action" = apply || test "$action" = apply-resume
  (
    date +%s.%N > "$root/dense.started"
    "${dense[@]}" >"$root/dense.stdout" 2>"$root/dense.stderr"
    date +%s.%N > "$root/dense.finished"
  ) &
  dense_pid=$!
  (
    date +%s.%N > "$root/split90.started"
    "${split[@]}" >"$root/split90.stdout" 2>"$root/split90.stderr"
    date +%s.%N > "$root/split90.finished"
  ) &
  split_pid=$!
  set +e
  wait "$dense_pid"; dense_rc=$?
  wait "$split_pid"; split_rc=$?
  set -e
  test "$dense_rc" = 0
  test "$split_rc" = 0
}

# DRY RUN first: two GPUs total, one rank per arm.
run_canary_pair plan functional 1
read -r -p 'Type APPLY-FUNCTIONAL-CANARY: ' answer
test "$answer" = APPLY-FUNCTIONAL-CANARY
run_canary_pair apply functional 1

# Build the paired resume probe without changing either output identity.
python - "$MS_CANARY_ROOT/functional" <<'PY'
from pathlib import Path
import sys
import yaml

root = Path(sys.argv[1])
for arm in ("dense", "split90"):
    path = root / f"{arm}.yaml"
    value = yaml.safe_load(path.read_text())
    value["max_steps"] = 2
    value["total_tokens"] = 2 * value["tokens_per_step"]
    path.write_text(yaml.safe_dump(value, sort_keys=False))
PY

# DRY RUN then APPLY: both arms must resume from their step-1 checkpoints.
run_canary_pair plan-resume functional 1
read -r -p 'Type APPLY-PAIRED-RESUME-CANARY: ' answer
test "$answer" = APPLY-PAIRED-RESUME-CANARY
run_canary_pair apply-resume functional 1
grep -q 'resumed from step 1' "$MS_CANARY_ROOT/functional/dense.stdout"
grep -q 'resumed from step 1' "$MS_CANARY_ROOT/functional/split90.stdout"

# DRY RUN first: production 4+4 topology for exactly 100 updates.
run_canary_pair plan throughput-100 4
read -r -p 'Type APPLY-100-UPDATE-CANARY: ' answer
test "$answer" = APPLY-100-UPDATE-CANARY
run_canary_pair apply throughput-100 4
```

Compute measured per-arm throughput and ETA from the 100-update wall times:

```bash
python - "$MS_CANARY_ROOT/throughput-100" <<'PY' |
  tee /tmp/memorysplit-throughput.json
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
targets = 100 * 524_288
result = {}
for arm in ("dense", "split90"):
    start = float((root / f"{arm}.started").read_text())
    finish = float((root / f"{arm}.finished").read_text())
    seconds = finish - start
    rate = targets / seconds
    result[arm] = {
        "seconds": seconds,
        "targets_per_second": rate,
        "seconds_per_update": seconds / 100,
        "estimated_training_hours": 7_120_879_616 / rate / 3600,
    }
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
PY
jq -e '
  .dense.targets_per_second > 0 and
  .split90.targets_per_second > 0
' /tmp/memorysplit-throughput.json >/dev/null
```

Do not approve production from a projected rate. The measured 100-update
result, clean paired exit, no NCCL error, and successful paired resume canary
are required.

## 8. Production launch, status, and checkpoint mirroring

Return to the control host. Select the exact instance assigned to the seed and
export it for the AWS backend. For economy mode use the same instance ID one
seed at a time. Submit, resume, evaluate, cancel, and cleanup require separate
signed approvals; each approval binds its operation, provider, seed, run
manifest, release, dataset, cohort assignment, requested GPU-hours, and
expiration.

```bash
export MS_INSTANCE_ID="$MS_SEQUENTIAL_INSTANCE_ID"
export MS_AWS_INSTANCE_ID="$MS_INSTANCE_ID"
export MS_SEED=1
export MS_RUN_MANIFEST="$PWD/run-manifests/aws-p5-s${MS_SEED}.json"
export MS_SUBMIT_APPROVAL="$MS_APPROVAL_ROOT/submit-seed-${MS_SEED}.json"

"${MSCTL[@]}" auth check
"${MSCTL[@]}" capacity check
"${MSCTL[@]}" env ensure \
  --root /mnt/memorysplit/environment \
  --lock requirements-aws-p5.lock
"${MSCTL[@]}" env ensure \
  --root /mnt/memorysplit/environment \
  --lock requirements-aws-p5.lock \
  --apply
"${MSCTL[@]}" dataset ensure --pointer DATASET-POINTER-AWS.json
"${MSCTL[@]}" dataset ensure \
  --pointer DATASET-POINTER-AWS.json --apply
"${MSCTL[@]}" dataset verify \
  --pointer "$MS_DATASET_POINTER" \
  --dataset-root "$(dirname "$MS_DATASET_RECEIPT")" \
  --release "$MS_AWS_RELEASE" \
  --manifest "$MS_RUN_MANIFEST"
"${MSCTL[@]}" dataset verify \
  --pointer "$MS_DATASET_POINTER" \
  --dataset-root "$(dirname "$MS_DATASET_RECEIPT")" \
  --release "$MS_AWS_RELEASE" \
  --manifest "$MS_RUN_MANIFEST" \
  --apply

# DRY RUN: renders one supervised Dense/Split90 launch.
"${MSCTL[@]}" submit \
  --release "$MS_AWS_RELEASE" \
  --manifest "$MS_RUN_MANIFEST" \
  --dataset-pointer "$MS_DATASET_POINTER" \
  --dataset-root "$(dirname "$MS_DATASET_RECEIPT")" \
  --approval "$MS_SUBMIT_APPROVAL" |
  tee "approvals/seed-${MS_SEED}-submit-plan.json"
jq -e '.ok == true and .dry_run == true' \
  "approvals/seed-${MS_SEED}-submit-plan.json" >/dev/null
grep -Fq "$MS_INSTANCE_ID" \
  "approvals/seed-${MS_SEED}-submit-plan.json"
! grep -q 'run-instances' \
  "approvals/seed-${MS_SEED}-submit-plan.json"

# APPLY: paid production launch.
"${MSCTL[@]}" submit \
  --release "$MS_AWS_RELEASE" \
  --manifest "$MS_RUN_MANIFEST" \
  --dataset-pointer "$MS_DATASET_POINTER" \
  --dataset-root "$(dirname "$MS_DATASET_RECEIPT")" \
  --approval "$MS_SUBMIT_APPROVAL" \
  --apply

"${MSCTL[@]}" status \
  --release "$MS_AWS_RELEASE" \
  --manifest "$MS_RUN_MANIFEST"
```

The rendered launch must contain exactly these symmetric groups:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 ...
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 ...
```

The launcher must durably mirror both arms at least every 30 minutes. If an
operator performs the transfer directly, run the following inside that
instance's SSM session, not on the control host:

```bash
# DRY RUN.
aws s3 sync \
  "/mnt/memorysplit/runs/seed-${MS_SEED}/" \
  "$MS_S3_ROOT/checkpoints/seed-${MS_SEED}/" \
  --region "$AWS_REGION" --no-follow-symlinks --dryrun
# APPLY only for a launcher-produced paired checkpoint and receipt.
aws s3 sync \
  "/mnt/memorysplit/runs/seed-${MS_SEED}/" \
  "$MS_S3_ROOT/checkpoints/seed-${MS_SEED}/" \
  --region "$AWS_REGION" --no-follow-symlinks
```

In deadline mode, run this section once per row of
`approvals/deadline-instances.tsv`, with `MS_SEED` and `MS_INSTANCE_ID` from
that same row. Never submit the same seed to two active instances.

## 9. Paired resume

Resume only from one receipt binding both checkpoint bytes, both arms, the
same seed, world size 4, config hashes, release/code commit, and corpus hash.

```bash
export MS_CHECKPOINT_ROOT="$PWD/checkpoints/seed-${MS_SEED}"
export MS_CHECKPOINT_RECEIPT="$MS_CHECKPOINT_ROOT/paired-receipt.json"
export MS_RUN_MANIFEST_SHA256="$(file_sha256 "$MS_RUN_MANIFEST")"
export MS_RESUME_APPROVAL="$MS_APPROVAL_ROOT/resume-seed-${MS_SEED}.json"
test ! -e "$MS_CHECKPOINT_ROOT"

# DRY RUN.
aws s3 sync \
  "$MS_S3_ROOT/checkpoints/seed-${MS_SEED}/" \
  "$MS_CHECKPOINT_ROOT/" \
  --region "$AWS_REGION" --dryrun
# APPLY the no-replace local checkpoint bundle.
mkdir -m 0700 -p "$MS_CHECKPOINT_ROOT"
aws s3 sync \
  "$MS_S3_ROOT/checkpoints/seed-${MS_SEED}/" \
  "$MS_CHECKPOINT_ROOT/" \
  --region "$AWS_REGION" --no-follow-symlinks
jq -e '
  .schema_version == 2 and
  .provider == "aws-p5.48xlarge" and
  .release_sha256 == env.MS_RELEASE_SHA256 and
  .run_manifest_sha256 == env.MS_RUN_MANIFEST_SHA256 and
  .dataset_sha256 == env.MS_DATASET_SHA256 and
  .source_commit == env.MS_CODE_COMMIT and
  (.checkpoints | length) == 2 and
  ([.checkpoints[].arm] | sort) == ["dense","split90"] and
  all(.checkpoints[];
    .seed == (env.MS_SEED | tonumber) and
    .world_size == 4)
' "$MS_CHECKPOINT_RECEIPT" >/dev/null

# DRY RUN.
"${MSCTL[@]}" resume \
  --release "$MS_AWS_RELEASE" \
  --manifest "$MS_RUN_MANIFEST" \
  --checkpoint-receipt "$MS_CHECKPOINT_RECEIPT" \
  --dataset-pointer "$MS_DATASET_POINTER" \
  --dataset-root "$(dirname "$MS_DATASET_RECEIPT")" \
  --approval "$MS_RESUME_APPROVAL"

# APPLY: paid paired resume. Never resume one arm alone.
"${MSCTL[@]}" resume \
  --release "$MS_AWS_RELEASE" \
  --manifest "$MS_RUN_MANIFEST" \
  --checkpoint-receipt "$MS_CHECKPOINT_RECEIPT" \
  --dataset-pointer "$MS_DATASET_POINTER" \
  --dataset-root "$(dirname "$MS_DATASET_RECEIPT")" \
  --approval "$MS_RESUME_APPROVAL" \
  --apply
```

If either checkpoint fails validation, mark the receipt non-resumable and
restart the pair from the last earlier valid paired receipt.

## 10. Paired evaluation and evidence collection

Evaluation is allowed only after both arms complete and their terminal
checkpoint receipt validates against the sealed release.

```bash
export MS_EVALUATE_APPROVAL="$MS_APPROVAL_ROOT/evaluate-seed-${MS_SEED}.json"
"${MSCTL[@]}" status \
  --release "$MS_AWS_RELEASE" \
  --manifest "$MS_RUN_MANIFEST" |
  tee "approvals/seed-${MS_SEED}-pre-eval-status.json"
jq -e '
  .ok == true and
  .result.status == "Success" and
  (.result.runs | length) == 2 and
  all(.result.runs[]; .status == "Success")
' "approvals/seed-${MS_SEED}-pre-eval-status.json" >/dev/null

# DRY RUN.
"${MSCTL[@]}" evaluate \
  --release "$MS_AWS_RELEASE" \
  --manifest "$MS_RUN_MANIFEST" \
  --dataset-pointer "$MS_DATASET_POINTER" \
  --dataset-root "$(dirname "$MS_DATASET_RECEIPT")" \
  --approval "$MS_EVALUATE_APPROVAL"
# APPLY: sealed paired evaluation.
"${MSCTL[@]}" evaluate \
  --release "$MS_AWS_RELEASE" \
  --manifest "$MS_RUN_MANIFEST" \
  --dataset-pointer "$MS_DATASET_POINTER" \
  --dataset-root "$(dirname "$MS_DATASET_RECEIPT")" \
  --approval "$MS_EVALUATE_APPROVAL" \
  --apply

# Inside the selected instance's SSM session:
# DRY RUN.
aws s3 sync \
  "/mnt/memorysplit/evaluations/seed-${MS_SEED}/" \
  "$MS_S3_ROOT/evidence/seed-${MS_SEED}/" \
  --region "$AWS_REGION" --no-follow-symlinks --dryrun
# APPLY.
aws s3 sync \
  "/mnt/memorysplit/evaluations/seed-${MS_SEED}/" \
  "$MS_S3_ROOT/evidence/seed-${MS_SEED}/" \
  --region "$AWS_REGION" --no-follow-symlinks

# Back on the control host:
export MS_EVIDENCE_BUNDLE="$PWD/evidence/seed-${MS_SEED}/evidence-bundle.json"
# DRY RUN: source is relative to MS_S3_ROOT, not a local instance path.
"${MSCTL[@]}" collect \
  --source "evidence/seed-${MS_SEED}/evidence-bundle.json" \
  --out "$MS_EVIDENCE_BUNDLE"
# APPLY, local no-replace download.
"${MSCTL[@]}" collect \
  --source "evidence/seed-${MS_SEED}/evidence-bundle.json" \
  --out "$MS_EVIDENCE_BUNDLE" \
  --apply
test -f "$MS_EVIDENCE_BUNDLE"

aws s3api list-object-versions \
  --region "$AWS_REGION" \
  --bucket "$MS_BUCKET" \
  --prefix "$MS_COHORT_ID/evidence/seed-${MS_SEED}/" \
  --output json |
  tee "approvals/seed-${MS_SEED}-evidence-versions.json"
jq -e '.Versions | length > 0' \
  "approvals/seed-${MS_SEED}-evidence-versions.json" >/dev/null
```

Economy mode proceeds to the next seed only after this collection gate. Before
reusing the instance, atomically replace only its seed and run-manifest tags;
all cohort, release, dataset, AMI, profile, and evaluation identities remain
unchanged:

```bash
export MS_NEXT_SEED=2
export MS_NEXT_MANIFEST="run-manifests/aws-p5-s${MS_NEXT_SEED}.json"
export MS_NEXT_MANIFEST_SHA256="$(file_sha256 "$MS_NEXT_MANIFEST")"

# DRY RUN. Success is DryRunOperation.
set +e
retag_plan="$(
  aws ec2 create-tags \
    --region "$AWS_REGION" \
    --resources "$MS_SEQUENTIAL_INSTANCE_ID" \
    --tags \
      "Key=MemorySplitSeed,Value=${MS_NEXT_SEED}" \
      "Key=MemorySplitRunManifestSHA256,Value=${MS_NEXT_MANIFEST_SHA256}" \
    --dry-run 2>&1
)"
retag_rc=$?
set -e
test "$retag_rc" -ne 0
grep -q DryRunOperation <<<"$retag_plan"

# APPLY only after seed 1 has reached the collection gate and no SSM training
# or evaluation command remains active.
aws ec2 create-tags \
  --region "$AWS_REGION" \
  --resources "$MS_SEQUENTIAL_INSTANCE_ID" \
  --tags \
    "Key=MemorySplitSeed,Value=${MS_NEXT_SEED}" \
    "Key=MemorySplitRunManifestSHA256,Value=${MS_NEXT_MANIFEST_SHA256}"
```

Repeat for seeds 3 and 4 only after the preceding seed reaches the same gate.
Deadline mode applies the same gate independently to each preassigned seed.
Observed effect direction never changes whether an assigned seed continues.

## 11. Cancellation and termination

Cancellation is paired and dry-run-first:

```bash
export MS_CANCEL_APPROVAL="$MS_APPROVAL_ROOT/cancel-seed-${MS_SEED}.json"
# DRY RUN.
"${MSCTL[@]}" cancel \
  --release "$MS_AWS_RELEASE" \
  --manifest "$MS_RUN_MANIFEST" \
  --approval "$MS_CANCEL_APPROVAL"
# APPLY only for a preregistered validity or infrastructure stop.
"${MSCTL[@]}" cancel \
  --release "$MS_AWS_RELEASE" \
  --manifest "$MS_RUN_MANIFEST" \
  --approval "$MS_CANCEL_APPROVAL" \
  --apply
```

Before termination, require S3 object versions for both-arm checkpoints,
evaluations, evidence, and receipts. Then dry-run termination before applying:

```bash
aws s3api list-object-versions \
  --region "$AWS_REGION" \
  --bucket "$MS_BUCKET" \
  --prefix "$MS_COHORT_ID/" \
  --output json | tee approvals/final-object-versions.json
jq -e '
  [.Versions[].Key
   | select(test("/(checkpoints|evaluations|evidence|receipts)/"))]
  | length > 0
' approvals/final-object-versions.json >/dev/null

export MS_TERMINATE_IDS="$MS_SEQUENTIAL_INSTANCE_ID"
# Deadline mode:
# export MS_TERMINATE_IDS="$(
#   awk '{print $2}' approvals/deadline-instances.tsv | paste -sd' ' -
# )"

# DRY RUN. Success is DryRunOperation.
set +e
terminate_plan="$(
  aws ec2 terminate-instances \
    --region "$AWS_REGION" \
    --instance-ids $MS_TERMINATE_IDS \
    --dry-run 2>&1
)"
terminate_rc=$?
set -e
test "$terminate_rc" -ne 0
grep -q DryRunOperation <<<"$terminate_plan"

# APPLY after independent S3 rehash and final approval.
read -r -p 'Type TERMINATE-P5: ' answer
test "$answer" = TERMINATE-P5
aws ec2 terminate-instances \
  --region "$AWS_REGION" \
  --instance-ids $MS_TERMINATE_IDS \
  --output json | tee approvals/termination.json
aws ec2 wait instance-terminated \
  --region "$AWS_REGION" \
  --instance-ids $MS_TERMINATE_IDS
```

## 12. Scientific labels

- Seed 0 alone: `scientific_status=incomplete`,
  `interim_evidence_label=directional_only`, completion `1/5`, and
  `final_inference_conclusion=not_evaluated`.
- Any preterminal cohort remains `incomplete`. A same-sign preterminal pattern
  may be labeled only `sign_consistent_only` when the frozen classifier permits
  it; it is not a terminal conclusion.
- Seeds 1–4 continue regardless of seed-0 or interim effect direction unless a
  preregistered measured-validity or infrastructure stop fires.
- After five valid pairs, run the frozen exact one-sided exhaustive sign-flip
  test on the five paired Split90-minus-Dense primary-omnibus deltas.
- At valid terminal N=5, the final label is exactly one of `supports_effect`,
  `supports_practical_null`, or `inconclusive`. The interim label becomes
  `none`.
- `supports_effect` is limited to the narrow primary omnibus. Separate
  superiority in both reasoning families is unsupported at N=5.
- Never report `statistically_significant_single_seed`, `failed_to_reject`, or
  `reasoning_optimal_sprint`.
