# AWS P5 360M paired-cohort runbook

This runbook operates AWS seeds 1–4 of
`memorysplit-confirmatory-v2-360m-n5`. Illumina owns seed 0. Each AWS seed is
one simultaneous Dense/Split90 pair on one On-Demand `p5.48xlarge`: GPUs 0–3
run Dense and GPUs 4–7 run Split90.

All paid or mutating operations have an explicit dry run first. Stop on any
failed hash, identity, readiness, or receipt check. Never edit a release,
manifest, checkpoint receipt, or sealed evaluation artifact in place.

Capacity provisioning is outside this lifecycle. This runbook consumes an
explicitly selected, separately approved instance ID; it never creates
capacity implicitly.

## 0. Integration constants

These are the last committed interfaces used to write this runbook. Re-pin the
constant only after the owning task publishes and its focused tests pass.
Task 3/5 currently has uncommitted launcher review work, and Task 6 does not yet
publish the selected-instance interface used below; those are release blockers,
not reasons to add compatibility branches here.

```bash
set -euo pipefail
umask 077

export MS_ILLUMINA_INTERFACE_REF=a85eb13ff2bf41046ab4a77d4f1924ed7e66c32e
export MS_TASK_3_5_INTERFACE_REF=4b796a5
export MS_TASK_6_INTERFACE_REF=0c9f20b
export MS_TASK_7_INTERFACE_REF=48cfacb
export MS_TASK_8_INTERFACE_REF=f22f6ad

export AWS_REGION=us-east-1
export MS_COHORT_ID=memorysplit-confirmatory-v2-360m-n5
export MS_PROVIDER=aws-p5.48xlarge
export MS_INSTANCE_TYPE=p5.48xlarge
export MS_INSTANCE_ID=i-REPLACE_WITH_SEPARATELY_APPROVED_INSTANCE
export MS_BUCKET=REPLACE_WITH_GLOBALLY_UNIQUE_BUCKET
export MS_S3_ROOT="s3://${MS_BUCKET}/${MS_COHORT_ID}"
export MS_AWS_AMI_ID=ami-REPLACE_WITH_PINNED_DLAMI
export MS_CONTAINER_DIGEST="sha256:REPLACE_WITH_64_LOWERCASE_HEX"
export MS_CONTAINER_IMAGE="REPLACE_WITH_REGISTRY/REPLACE_WITH_IMAGE@${MS_CONTAINER_DIGEST}"
export MS_RUNTIME_UID=1000
export MS_RUNTIME_GID=1000
export MS_AWS_PRIVATE_HOME=/var/lib/memorysplit/aws-home

export MS_RELEASE_ROOT="$PWD/../memorysplit-releases/aws-p5"
export MS_ILLUMINA_RELEASE="$PWD/../memorysplit-releases/illumina/RELEASE.json"
export MS_AWS_RELEASE="$MS_RELEASE_ROOT/RELEASE-AWS-P5.json"
export MS_DATASET_POINTER="$PWD/DATASET-POINTER-AWS.json"
export MS_DATASET_RECEIPT="$PWD/dataset/corpus-receipt.json"
export MS_ENVIRONMENT_RECEIPT="$PWD/environment/aws-p5-environment-receipt.json"
export MS_SEALED_RELEASE="$PWD/evaluation/SEALED-RELEASE.json"
export MS_STUDY_LOCK="$PWD/evaluation/STUDY-LOCK.json"
export MS_TERMINATE_AT=REPLACE_WITH_RFC3339_UTC_DEADLINE
export MS_APPROVAL_ROOT="$PWD/approvals/aws-p5"
export MS_STATE_ROOT="$PWD/.msctl-state/aws-p5"
mkdir -p "$MS_APPROVAL_ROOT" "$MS_STATE_ROOT"

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

case "$MS_INSTANCE_ID" in
  i-[0-9a-f]*) ;;
  *) printf '%s\n' "invalid MS_INSTANCE_ID" >&2; exit 2 ;;
esac
case "$MS_RUNTIME_UID:$MS_RUNTIME_GID" in
  0:*|*:0|*[!0-9:]*|'') printf '%s\n' "runtime UID/GID must be numeric and non-root" >&2; exit 2 ;;
esac
printf '%s\n' "$MS_CONTAINER_IMAGE" |
  grep -Eq '^[A-Za-z0-9][A-Za-z0-9./:_-]*@sha256:[0-9a-f]{64}$'

aws sts get-caller-identity --output json |
  tee "$MS_APPROVAL_ROOT/control-identity.json"
aws --version
jq --version
python --version
test -z "$(git status --porcelain=v1)"

python scripts/verify_cohort_releases.py \
  --illumina "$MS_ILLUMINA_RELEASE" \
  --aws "$MS_AWS_RELEASE" |
  tee "$MS_APPROVAL_ROOT/cohort-verification.json"
jq -e '
  .ok == true and
  .illumina.seeds == [0] and
  .aws.seeds == [1,2,3,4] and
  .complete_cohort == [0,1,2,3,4]
' "$MS_APPROVAL_ROOT/cohort-verification.json" >/dev/null

export MS_RELEASE_SHA256="$(jq -er '.archive.sha256' "$MS_AWS_RELEASE")"
export MS_ARCHIVE_NAME="$(jq -er '.archive.path' "$MS_AWS_RELEASE")"
export MS_RELEASE_RECEIPT_SHA256="$(file_sha256 "$MS_AWS_RELEASE")"
export MS_ASSIGNMENT_SHA256="$(
  jq -er '.cohort_assignment_sha256' "$MS_AWS_RELEASE"
)"
export MS_DATASET_SHA256="$(file_sha256 "$MS_DATASET_RECEIPT")"
export MS_STUDY_LOCK_SHA256="$(file_sha256 "$MS_STUDY_LOCK")"
export MS_CODE_COMMIT="$(jq -er '.source.commit' "$MS_AWS_RELEASE")"
```

## 1. Current price, quota, and selected capacity

One instance consumes 192 Running On-Demand P instances vCPUs. Sequential mode
needs 192; four-instance deadline mode needs 768. Query both quota and live
price immediately before approval.

```bash
export MS_P_QUOTA_CODE="$(
  aws service-quotas list-service-quotas \
    --region "$AWS_REGION" \
    --service-code ec2 \
    --query "Quotas[?QuotaName=='Running On-Demand P instances'].QuotaCode | [0]" \
    --output text
)"
test -n "$MS_P_QUOTA_CODE"
test "$MS_P_QUOTA_CODE" != None
aws service-quotas get-service-quota \
  --region "$AWS_REGION" \
  --service-code ec2 \
  --quota-code "$MS_P_QUOTA_CODE" \
  --output json |
  tee "$MS_APPROVAL_ROOT/p-instance-quota.json"
jq -e '.Quota.Value >= 192' "$MS_APPROVAL_ROOT/p-instance-quota.json" >/dev/null

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
  --output json |
  tee "$MS_APPROVAL_ROOT/p5-current-price.json"
jq -r '
  [.PriceList[] | fromjson
   | .terms.OnDemand[]
   | .priceDimensions[]
   | select(.unit == "Hrs")
   | .pricePerUnit.USD] | unique | .[]
' "$MS_APPROVAL_ROOT/p5-current-price.json" |
  tee "$MS_APPROVAL_ROOT/p5-usd-per-hour.txt"
test "$(wc -l < "$MS_APPROVAL_ROOT/p5-usd-per-hour.txt" | tr -d ' ')" = 1
date -u +%Y-%m-%dT%H:%M:%SZ |
  tee "$MS_APPROVAL_ROOT/p5-price-queried-at.txt"

aws ec2 describe-instances \
  --region "$AWS_REGION" \
  --instance-ids "$MS_INSTANCE_ID" \
  --output json |
  tee "$MS_APPROVAL_ROOT/selected-instance.json"
jq -e \
  --arg instance "$MS_INSTANCE_ID" \
  --arg ami "$MS_AWS_AMI_ID" '
  [.Reservations[].Instances[]] as $instances |
  ($instances | length) == 1 and
  $instances[0].InstanceId == $instance and
  $instances[0].InstanceType == "p5.48xlarge" and
  $instances[0].ImageId == $ami and
  ($instances[0].InstanceLifecycle? == null) and
  ($instances[0].IamInstanceProfile.Arn | type == "string") and
  ($instances[0].PublicIpAddress? == null) and
  ($instances[0].MetadataOptions.HttpTokens == "required") and
  ($instances[0].State.Name | IN("pending","running","stopped"))
' "$MS_APPROVAL_ROOT/selected-instance.json" >/dev/null
```

Before deadline mode, repeat the selected-instance check for four distinct IDs
and require the stronger quota gate:

```bash
jq -e '.Quota.Value >= 768' "$MS_APPROVAL_ROOT/p-instance-quota.json" >/dev/null
```

## 2. Durable S3 boundary and instance role

The immutable layout is:

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

Use an instance role for S3 and SSM. The role may read release, dataset, and
sealed-evaluation prefixes and may write only checkpoints, evaluations,
evidence, and receipts. Do not place long-lived secrets on the control host or
instance.

```bash
# DRY RUN.
aws s3api create-bucket \
  --region "$AWS_REGION" \
  --bucket "$MS_BUCKET" \
  --generate-cli-skeleton output >/dev/null
# APPLY after bucket-name approval.
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

# DRY RUN.
aws s3 sync "$MS_RELEASE_ROOT/" \
  "$MS_S3_ROOT/releases/aws-p5/" \
  --region "$AWS_REGION" --no-follow-symlinks --dryrun
# APPLY.
aws s3 sync "$MS_RELEASE_ROOT/" \
  "$MS_S3_ROOT/releases/aws-p5/" \
  --region "$AWS_REGION" --no-follow-symlinks

# DRY RUN.
aws s3 sync "$(dirname "$MS_DATASET_RECEIPT")/" \
  "$MS_S3_ROOT/dataset/" \
  --region "$AWS_REGION" --no-follow-symlinks --dryrun
# APPLY.
aws s3 sync "$(dirname "$MS_DATASET_RECEIPT")/" \
  "$MS_S3_ROOT/dataset/" \
  --region "$AWS_REGION" --no-follow-symlinks

# DRY RUN.
aws s3 cp "$(dirname "$MS_SEALED_RELEASE")/" \
  "$MS_S3_ROOT/sealed-evaluation/" \
  --recursive --region "$AWS_REGION" --dryrun
# APPLY.
aws s3 cp "$(dirname "$MS_SEALED_RELEASE")/" \
  "$MS_S3_ROOT/sealed-evaluation/" \
  --recursive --region "$AWS_REGION"

aws s3api head-bucket --bucket "$MS_BUCKET" --region "$AWS_REGION"
aws s3api get-bucket-versioning \
  --bucket "$MS_BUCKET" --region "$AWS_REGION" |
  jq -e '.Status == "Enabled"' >/dev/null
```

Record S3 version IDs for the archive, receipt, assignment, dataset receipt,
sealed release, and study lock. The downloaded bytes must reproduce every local
SHA-256 before bootstrap is approved.

## 3. SSM lifecycle for the selected instance

There is no SSH path. Every lifecycle command names the selected instance.
Starting capacity is paid and therefore has a dry run and explicit approval.

```bash
# DRY RUN.
aws ec2 start-instances \
  --region "$AWS_REGION" \
  --instance-ids "$MS_INSTANCE_ID" \
  --dry-run
read -r -p 'Type START-SELECTED-P5: ' MS_START_APPROVAL
test "$MS_START_APPROVAL" = START-SELECTED-P5
# APPLY.
aws ec2 start-instances \
  --region "$AWS_REGION" \
  --instance-ids "$MS_INSTANCE_ID"
aws ec2 wait instance-running \
  --region "$AWS_REGION" \
  --instance-ids "$MS_INSTANCE_ID"
aws ec2 wait instance-status-ok \
  --region "$AWS_REGION" \
  --instance-ids "$MS_INSTANCE_ID"

aws ssm describe-instance-information \
  --region "$AWS_REGION" \
  --filters "Key=InstanceIds,Values=${MS_INSTANCE_ID}" \
  --output json |
  tee "$MS_APPROVAL_ROOT/ssm-ready.json"
jq -e \
  --arg instance "$MS_INSTANCE_ID" '
  [.InstanceInformationList[]
   | select(.InstanceId == $instance and .PingStatus == "Online")]
  | length == 1
' "$MS_APPROVAL_ROOT/ssm-ready.json" >/dev/null

aws ssm start-session \
  --region "$AWS_REGION" \
  --target "$MS_INSTANCE_ID"
```

Run the next three sections inside that SSM session. The instance role and
IMDSv2 are the only AWS identity source.

## 4. Task 3/5 bootstrap and destructive NVMe authorization

Download the authenticated release, receipt, assignment, and dataset receipt
to the root-volume staging directory. Verify each expected digest before
extracting the already cohort-verified archive into a read-only control root.

```bash
set -euo pipefail
umask 077
export AWS_REGION=us-east-1
export MS_S3_ROOT="s3://REPLACE_WITH_BUCKET/memorysplit-confirmatory-v2-360m-n5"
export MS_AWS_AMI_ID=ami-REPLACE_WITH_PINNED_DLAMI
export MS_CONTAINER_DIGEST="sha256:REPLACE_WITH_64_LOWERCASE_HEX"
export MS_CONTAINER_IMAGE="REPLACE_WITH_REGISTRY/REPLACE_WITH_IMAGE@${MS_CONTAINER_DIGEST}"
export MS_RUNTIME_UID=1000
export MS_RUNTIME_GID=1000
export MS_AWS_PRIVATE_HOME=/var/lib/memorysplit/aws-home
export MS_RELEASE_SHA256=REPLACE_WITH_ARCHIVE_SHA256
export MS_RELEASE_RECEIPT_SHA256=REPLACE_WITH_RECEIPT_SHA256
export MS_DATASET_SHA256=REPLACE_WITH_DATASET_RECEIPT_SHA256
export MS_ASSIGNMENT_SHA256=REPLACE_WITH_ASSIGNMENT_SHA256
export MS_CODE_COMMIT=REPLACE_WITH_40_HEX_COMMIT
export MS_ARCHIVE_NAME=REPLACE_WITH_VERIFIED_ARCHIVE_NAME

sudo install -d -m 0700 -o "$MS_RUNTIME_UID" -g "$MS_RUNTIME_GID" \
  /var/tmp/memorysplit-input
sudo install -d -m 0700 -o "$MS_RUNTIME_UID" -g "$MS_RUNTIME_GID" \
  "$MS_AWS_PRIVATE_HOME"
test -z "$(sudo find "$MS_AWS_PRIVATE_HOME" -mindepth 1 -print -quit)"

aws s3 cp "$MS_S3_ROOT/releases/aws-p5/$MS_ARCHIVE_NAME" \
  "/var/tmp/memorysplit-input/$MS_ARCHIVE_NAME" \
  --region "$AWS_REGION" --no-progress
aws s3 cp "$MS_S3_ROOT/releases/aws-p5/RELEASE-AWS-P5.json" \
  /var/tmp/memorysplit-input/RELEASE-AWS-P5.json \
  --region "$AWS_REGION" --no-progress
aws s3 cp "$MS_S3_ROOT/dataset/corpus-receipt.json" \
  /var/tmp/memorysplit-input/corpus-receipt.json \
  --region "$AWS_REGION" --no-progress

test "$(sha256sum "/var/tmp/memorysplit-input/$MS_ARCHIVE_NAME" | cut -d' ' -f1)" = \
  "$MS_RELEASE_SHA256"
test "$(sha256sum /var/tmp/memorysplit-input/RELEASE-AWS-P5.json | cut -d' ' -f1)" = \
  "$MS_RELEASE_RECEIPT_SHA256"
test "$(sha256sum /var/tmp/memorysplit-input/corpus-receipt.json | cut -d' ' -f1)" = \
  "$MS_DATASET_SHA256"

rm -rf /var/tmp/memorysplit-control
mkdir -m 0700 /var/tmp/memorysplit-control
python -m zipfile -e \
  "/var/tmp/memorysplit-input/$MS_ARCHIVE_NAME" \
  /var/tmp/memorysplit-control
chmod -R a-w /var/tmp/memorysplit-control

BOOTSTRAP=(
  sudo
  --preserve-env=AWS_REGION,MS_S3_ROOT,MS_AWS_AMI_ID,MS_CONTAINER_DIGEST,MS_RUNTIME_UID,MS_RUNTIME_GID
  python
  /var/tmp/memorysplit-control/cluster/aws/p5/bootstrap.py
  --profile
  /var/tmp/memorysplit-control/cluster/profiles/aws-p5.48xlarge.json
  --container-image
  "$MS_CONTAINER_IMAGE"
  --release-archive
  "/var/tmp/memorysplit-input/$MS_ARCHIVE_NAME"
  --release-sha256
  "$MS_RELEASE_SHA256"
  --release-receipt
  /var/tmp/memorysplit-input/RELEASE-AWS-P5.json
  --release-receipt-sha256
  "$MS_RELEASE_RECEIPT_SHA256"
  --dataset-receipt
  /var/tmp/memorysplit-input/corpus-receipt.json
  --dataset-receipt-sha256
  "$MS_DATASET_SHA256"
  --cohort-assignment
  /var/tmp/memorysplit-control/configs/cohort-assignment-v2.json
  --cohort-assignment-sha256
  "$MS_ASSIGNMENT_SHA256"
  --code-commit
  "$MS_CODE_COMMIT"
  --owner-uid
  "$MS_RUNTIME_UID"
  --owner-gid
  "$MS_RUNTIME_GID"
  --aws-private-home
  "$MS_AWS_PRIVATE_HOME"
  --receipt
  /var/tmp/memorysplit-input/bootstrap-receipt.json
)

# DRY RUN: bootstrap inspects p5.48xlarge, eight NVIDIA H100 80GB GPUs,
# Fabric Manager, and all eight Amazon EC2 NVMe Instance Storage devices.
"${BOOTSTRAP[@]}" |
  tee /var/tmp/memorysplit-input/bootstrap-plan.json
jq -e '.ok == true and .dry_run == true' \
  /var/tmp/memorysplit-input/bootstrap-plan.json >/dev/null
jq -e '
  [.commands[] | select(.[0] == "mdadm" and .[1] == "--create")]
  | length == 1
' /var/tmp/memorysplit-input/bootstrap-plan.json >/dev/null

read -r -p 'Type DESTROY-SELECTED-INSTANCE-STORE: ' MS_STORAGE_APPROVAL
test "$MS_STORAGE_APPROVAL" = DESTROY-SELECTED-INSTANCE-STORE
# APPLY: the authorization flag is mandatory and deliberately adjacent.
"${BOOTSTRAP[@]}" \
  --authorize-destructive-instance-store \
  --apply |
  tee /var/tmp/memorysplit-input/bootstrap-result.json
jq -e '.ok == true and .dry_run == false' \
  /var/tmp/memorysplit-input/bootstrap-result.json >/dev/null

findmnt -n -o SOURCE,FSTYPE,OPTIONS /mnt/memorysplit
test "$(stat -f -c %T /mnt/memorysplit)" = xfs
test "$(stat -c %u /mnt/memorysplit)" = "$MS_RUNTIME_UID"
test "$(stat -c %g /mnt/memorysplit)" = "$MS_RUNTIME_GID"
```

The bootstrap owns destructive storage admission. Never run `mdadm --create`,
filesystem creation, or mounting by hand. `/mnt/memorysplit` is disposable;
only hash-verified S3 object versions are durable.

## 5. Immutable manifests and selected-instance lifecycle

Return to the control host. Instantiate all four seed-pair manifests before
launch. Creation is no-replace.

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
  # APPLY.
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
    --dataset-root "$(dirname "$MS_DATASET_RECEIPT")" \
    --environment-receipt "$MS_ENVIRONMENT_RECEIPT"
done
```

Every `submit`/`resume` plan must name the selected instance and render the
Task 3/5 launcher:

```text
python cluster/aws/p5/launch_seed_pair.py \
  --seed N \
  --manifest /mnt/memorysplit/.../aws-p5-sN.json \
  --profile /mnt/memorysplit/.../cluster/profiles/aws-p5.48xlarge.json \
  --repo-root /mnt/memorysplit/releases/RELEASE_SHA256 \
  --scratch-root /mnt/memorysplit
```

That launcher executes both distributed workers inside
`$MS_CONTAINER_IMAGE`, which is digest pinned. No distributed training process
runs directly in the host Python environment.

## 6. Functional canary, paired resume probe, and 100-update gate

Canaries are non-claim-bearing copies under scratch. They may reduce update
counts, but must not modify the release or production manifests. The function
below prints the exact digest-pinned Docker argv in plan mode and executes it
only in apply mode. The distributed Python module is therefore inside the
container, never on the host.

```bash
set -euo pipefail
export MS_RELEASE_CODE="/mnt/memorysplit/releases/${MS_RELEASE_SHA256}"
export MS_CANARY_ROOT=/mnt/memorysplit/staging/canary
install -d -m 0700 "$MS_CANARY_ROOT"

make_canary_config() {
  local arm="$1" updates="$2" output="$3"
  python - \
    "$MS_RELEASE_CODE/configs/360m-v2/${arm}-s1.yaml" \
    "$updates" "$output" <<'PY'
from pathlib import Path
import sys
import yaml

source, updates, output = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
value = yaml.safe_load(source.read_text(encoding="utf-8"))
value["max_steps"] = updates
value["total_tokens"] = updates * value["tokens_per_step"]
value["out_dir"] = "/output/run"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
PY
}

run_canary_arm() {
  local mode="$1" label="$2" arm="$3" gpu_ids="$4" nproc="$5"
  local config="$MS_CANARY_ROOT/$label/$arm.yaml"
  local output="$MS_CANARY_ROOT/$label/$arm-output"
  mkdir -p "$output"
  local -a command=(
    docker run --rm
    --read-only
    --network=host
    --ipc=host
    --user "${MS_RUNTIME_UID}:${MS_RUNTIME_GID}"
    --workdir /workspace
    --gpus "device=${gpu_ids}"
    --security-opt no-new-privileges
    --cap-drop ALL
    --mount "type=bind,src=${MS_RELEASE_CODE},dst=/workspace,readonly"
    --mount "type=bind,src=/mnt/memorysplit/dataset,dst=/dataset,readonly"
    --mount "type=bind,src=${config},dst=/runtime/config.yaml,readonly"
    --mount "type=bind,src=${output},dst=/output"
    --env HOME=/tmp/home
    "$MS_CONTAINER_IMAGE"
    /opt/conda/bin/python -m torch.distributed.run
    --nnodes=1
    "--nproc_per_node=${nproc}"
    --rdzv_backend=c10d
    "--rdzv_endpoint=127.0.0.1:$([[ "$arm" = dense ]] && echo 29611 || echo 29612)"
    /workspace/scripts/run_train.py
    --config /runtime/config.yaml
    --resume none
  )
  printf '%q ' "${command[@]}"
  printf '\n'
  if [[ "$mode" = apply ]]; then
    "${command[@]}" >"$output/container.log" 2>&1 &
    printf '%s\n' "$!"
  fi
}

run_canary_pair() {
  local mode="$1" label="$2" updates="$3" nproc="$4"
  make_canary_config dense "$updates" "$MS_CANARY_ROOT/$label/dense.yaml"
  make_canary_config split90 "$updates" "$MS_CANARY_ROOT/$label/split90.yaml"
  if [[ "$nproc" = 1 ]]; then
    dense_gpus=0
    split_gpus=4
  else
    dense_gpus=0,1,2,3
    split_gpus=4,5,6,7
  fi
  if [[ "$mode" = plan ]]; then
    run_canary_arm plan "$label" dense "$dense_gpus" "$nproc"
    run_canary_arm plan "$label" split90 "$split_gpus" "$nproc"
    return
  fi
  dense_pid="$(run_canary_arm apply "$label" dense "$dense_gpus" "$nproc" | tail -1)"
  split_pid="$(run_canary_arm apply "$label" split90 "$split_gpus" "$nproc" | tail -1)"
  wait "$dense_pid"
  wait "$split_pid"
}

# DRY RUN.
run_canary_pair plan functional 1 1
read -r -p 'Type APPLY-FUNCTIONAL-CANARY: ' MS_CANARY_APPROVAL
test "$MS_CANARY_APPROVAL" = APPLY-FUNCTIONAL-CANARY
# APPLY.
run_canary_pair apply functional 1 1

# Paired resume probe: first authenticate both checkpoint receipts and hashes,
# then use the same container function with copied resume configs. A failure in
# either arm invalidates the pair.
for arm in dense split90; do
  test -s "$MS_CANARY_ROOT/functional/${arm}-output/run/checkpoint.pt"
  test -s "$MS_CANARY_ROOT/functional/${arm}-output/run/checkpoint.pt.sha256"
  (
    cd "$MS_CANARY_ROOT/functional/${arm}-output/run"
    sha256sum -c checkpoint.pt.sha256
  )
done
printf '%s\n' "paired resume probe checkpoint hashes verified"

# DRY RUN.
run_canary_pair plan throughput-100 100 4
read -r -p 'Type APPLY-100-UPDATE-CANARY: ' MS_THROUGHPUT_APPROVAL
test "$MS_THROUGHPUT_APPROVAL" = APPLY-100-UPDATE-CANARY
# APPLY.
run_canary_pair apply throughput-100 100 4

# DRY RUN.
aws s3 sync "$MS_CANARY_ROOT/" \
  "$MS_S3_ROOT/evidence/canary/" \
  --region "$AWS_REGION" --no-follow-symlinks --dryrun
# APPLY.
aws s3 sync "$MS_CANARY_ROOT/" \
  "$MS_S3_ROOT/evidence/canary/" \
  --region "$AWS_REGION" --no-follow-symlinks
```

Approve production only after functional exit, the paired resume probe,
measured 100-update throughput, no NCCL error, and verified S3 evidence.

## 7. Sequential and four-instance deadline modes

Sequential mode reuses one selected instance for seeds 1–4. Confirm the prior
pair is complete and durable before changing seed.

```bash
for seed in 1 2 3 4; do
  manifest="run-manifests/aws-p5-s${seed}.json"
  approval="$MS_APPROVAL_ROOT/submit-s${seed}.json"

  # DRY RUN: must render SSM work for exactly the selected instance.
  "${MSCTL[@]}" submit \
    --instance-id "$MS_INSTANCE_ID" \
    --release "$MS_AWS_RELEASE" \
    --manifest "$manifest" \
    --dataset-pointer "$MS_DATASET_POINTER" \
    --dataset-root "$(dirname "$MS_DATASET_RECEIPT")" \
    --environment-receipt "$MS_ENVIRONMENT_RECEIPT" \
    --terminate-at "$MS_TERMINATE_AT" \
    --approval "$approval" |
    tee "$MS_APPROVAL_ROOT/submit-s${seed}-plan.json"
  jq -e --arg instance "$MS_INSTANCE_ID" '
    .dry_run == true and .instance_id == $instance
  ' "$MS_APPROVAL_ROOT/submit-s${seed}-plan.json" >/dev/null

  read -r -p "Type APPLY-SEED-${seed}: " answer
  test "$answer" = "APPLY-SEED-${seed}"
  # APPLY.
  "${MSCTL[@]}" submit \
    --instance-id "$MS_INSTANCE_ID" \
    --release "$MS_AWS_RELEASE" \
    --manifest "$manifest" \
    --dataset-pointer "$MS_DATASET_POINTER" \
    --dataset-root "$(dirname "$MS_DATASET_RECEIPT")" \
    --environment-receipt "$MS_ENVIRONMENT_RECEIPT" \
    --terminate-at "$MS_TERMINATE_AT" \
    --approval "$approval" \
    --apply

  "${MSCTL[@]}" status \
    --release "$MS_AWS_RELEASE" \
    --manifest "$manifest"
done
```

Deadline mode uses four separately approved IDs. Do not let lifecycle code
choose capacity.

```bash
declare -A MS_INSTANCE_BY_SEED=(
  [1]=i-REPLACE_SEED1
  [2]=i-REPLACE_SEED2
  [3]=i-REPLACE_SEED3
  [4]=i-REPLACE_SEED4
)
test "$(printf '%s\n' "${MS_INSTANCE_BY_SEED[@]}" | sort -u | wc -l | tr -d ' ')" = 4

for seed in 1 2 3 4; do
  instance="${MS_INSTANCE_BY_SEED[$seed]}"
  manifest="run-manifests/aws-p5-s${seed}.json"
  approval="$MS_APPROVAL_ROOT/deadline-submit-s${seed}.json"
  # DRY RUN.
  "${MSCTL[@]}" submit \
    --instance-id "$instance" \
    --release "$MS_AWS_RELEASE" \
    --manifest "$manifest" \
    --dataset-pointer "$MS_DATASET_POINTER" \
    --dataset-root "$(dirname "$MS_DATASET_RECEIPT")" \
    --environment-receipt "$MS_ENVIRONMENT_RECEIPT" \
    --terminate-at "$MS_TERMINATE_AT" \
    --approval "$approval"
done

read -r -p 'Type APPLY-FOUR-INSTANCE-DEADLINE: ' MS_DEADLINE_APPROVAL
test "$MS_DEADLINE_APPROVAL" = APPLY-FOUR-INSTANCE-DEADLINE
for seed in 1 2 3 4; do
  instance="${MS_INSTANCE_BY_SEED[$seed]}"
  manifest="run-manifests/aws-p5-s${seed}.json"
  approval="$MS_APPROVAL_ROOT/deadline-submit-s${seed}.json"
  # APPLY.
  "${MSCTL[@]}" submit \
    --instance-id "$instance" \
    --release "$MS_AWS_RELEASE" \
    --manifest "$manifest" \
    --dataset-pointer "$MS_DATASET_POINTER" \
    --dataset-root "$(dirname "$MS_DATASET_RECEIPT")" \
    --environment-receipt "$MS_ENVIRONMENT_RECEIPT" \
    --terminate-at "$MS_TERMINATE_AT" \
    --approval "$approval" \
    --apply
done
```

## 8. Paired resume, evaluation, and collection

Resume only when both arm receipts exist, both checkpoints pass SHA-256, and
the receipt marks the pair resumable.

```bash
seed=1
manifest="run-manifests/aws-p5-s${seed}.json"
checkpoint_receipt="$PWD/checkpoints/seed-${seed}/pair-checkpoint-receipt.json"
jq -e '
  .schema_version == 2 and
  .pair_complete == true and
  .resumable == true and
  (.arms | keys | sort) == ["dense","split90"] and
  ([.arms[].checkpoint_sha256
    | select(type == "string" and test("^[0-9a-f]{64}$"))] | length) == 2
' "$checkpoint_receipt" >/dev/null

# DRY RUN.
"${MSCTL[@]}" resume \
  --release "$MS_AWS_RELEASE" \
  --manifest "$manifest" \
  --dataset-pointer "$MS_DATASET_POINTER" \
  --dataset-root "$(dirname "$MS_DATASET_RECEIPT")" \
  --environment-receipt "$MS_ENVIRONMENT_RECEIPT" \
  --checkpoint-receipt "$checkpoint_receipt" \
  --approval "$MS_APPROVAL_ROOT/resume-s${seed}.json"
read -r -p "Type APPLY-RESUME-${seed}: " answer
test "$answer" = "APPLY-RESUME-${seed}"
# APPLY.
"${MSCTL[@]}" resume \
  --release "$MS_AWS_RELEASE" \
  --manifest "$manifest" \
  --dataset-pointer "$MS_DATASET_POINTER" \
  --dataset-root "$(dirname "$MS_DATASET_RECEIPT")" \
  --environment-receipt "$MS_ENVIRONMENT_RECEIPT" \
  --checkpoint-receipt "$checkpoint_receipt" \
  --approval "$MS_APPROVAL_ROOT/resume-s${seed}.json" \
  --apply
```

The Task 8 evaluator has one canonical argv. Build it as an array and pass it
to the same digest-pinned container; do not execute it in the host environment.

```bash
export MS_RUN="/mnt/memorysplit/runs/seed-${seed}"
export MS_EVAL_OUT="/mnt/memorysplit/evaluations/seed-${seed}"
EVALUATOR=(python -m evals.confirmatory evaluate --run /run --sealed-release /sealed/SEALED-RELEASE.json --expected-study-lock-sha256 "$MS_STUDY_LOCK_SHA256" --device cuda --output-dir /output)
printf '%q ' docker run --rm --read-only --gpus all --network=none \
  --user "${MS_RUNTIME_UID}:${MS_RUNTIME_GID}" --workdir /workspace \
  --mount "type=bind,src=${MS_RELEASE_CODE},dst=/workspace,readonly" \
  --mount "type=bind,src=${MS_RUN},dst=/run,readonly" \
  --mount "type=bind,src=$(dirname "$MS_SEALED_RELEASE"),dst=/sealed,readonly" \
  --mount "type=bind,src=${MS_EVAL_OUT},dst=/output" \
  "$MS_CONTAINER_IMAGE" "${EVALUATOR[@]}"
printf '\n'

# DRY RUN.
"${MSCTL[@]}" evaluate \
  --release "$MS_AWS_RELEASE" \
  --manifest "$manifest" \
  --dataset-pointer "$MS_DATASET_POINTER" \
  --dataset-root "$(dirname "$MS_DATASET_RECEIPT")" \
  --environment-receipt "$MS_ENVIRONMENT_RECEIPT" \
  --approval "$MS_APPROVAL_ROOT/evaluate-s${seed}.json"
read -r -p "Type APPLY-EVALUATE-${seed}: " answer
test "$answer" = "APPLY-EVALUATE-${seed}"
# APPLY.
"${MSCTL[@]}" evaluate \
  --release "$MS_AWS_RELEASE" \
  --manifest "$manifest" \
  --dataset-pointer "$MS_DATASET_POINTER" \
  --dataset-root "$(dirname "$MS_DATASET_RECEIPT")" \
  --environment-receipt "$MS_ENVIRONMENT_RECEIPT" \
  --approval "$MS_APPROVAL_ROOT/evaluate-s${seed}.json" \
  --apply

# DRY RUN.
"${MSCTL[@]}" collect \
  --source "evidence/seed-${seed}/bundle.json" \
  --out "evidence/seed-${seed}/bundle.json"
# APPLY.
"${MSCTL[@]}" collect \
  --source "evidence/seed-${seed}/bundle.json" \
  --out "evidence/seed-${seed}/bundle.json" \
  --apply
```

Verify every collected artifact against its receipt and S3 object version.
Collection is incomplete if either arm, the study lock, validity evidence,
replay report, logs, or checkpoint hashes are missing.

## 9. Stop, restart, and final termination

Stop a sequential instance between approved windows; restart through section 3.

```bash
# DRY RUN.
aws ec2 stop-instances \
  --region "$AWS_REGION" \
  --instance-ids "$MS_INSTANCE_ID" \
  --dry-run
read -r -p 'Type STOP-SELECTED-P5: ' MS_STOP_APPROVAL
test "$MS_STOP_APPROVAL" = STOP-SELECTED-P5
# APPLY.
aws ec2 stop-instances \
  --region "$AWS_REGION" \
  --instance-ids "$MS_INSTANCE_ID"
aws ec2 wait instance-stopped \
  --region "$AWS_REGION" \
  --instance-ids "$MS_INSTANCE_ID"
```

Terminate only after all four pairs and evaluation bundles are durable,
downloaded, hash-verified, and independently reviewed.

```bash
# DRY RUN.
aws ec2 terminate-instances \
  --region "$AWS_REGION" \
  --instance-ids "$MS_INSTANCE_ID" \
  --dry-run
read -r -p 'Type TERMINATE-SELECTED-P5: ' MS_TERMINATE_APPROVAL
test "$MS_TERMINATE_APPROVAL" = TERMINATE-SELECTED-P5
# APPLY.
aws ec2 terminate-instances \
  --region "$AWS_REGION" \
  --instance-ids "$MS_INSTANCE_ID"
aws ec2 wait instance-terminated \
  --region "$AWS_REGION" \
  --instance-ids "$MS_INSTANCE_ID"
```

For deadline mode, repeat this exact dry-run/approval/apply sequence for the
four reviewed instance IDs.

## 10. Frozen scientific labels

Infrastructure state, successful optimization, and statistical interpretation
are separate axes. Use only the preregistered labels:

- `directional_only`
- `sign_consistent_only`
- `supports_effect`
- `supports_practical_null`
- `inconclusive`

Do not claim success from one seed, a canary, throughput, or infrastructure
completion. The confirmatory label is assigned only from the complete sealed
five-seed cohort after study-lock validation and replay.
