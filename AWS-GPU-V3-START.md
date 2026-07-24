# AWS GPU v3 operator start

This handoff supports exactly one closed hardware profile per release:

- `cluster/profiles/aws-p5.48xlarge-v3.json`
- `cluster/profiles/aws-p6-b300.48xlarge-v3.json`

Do not combine profiles within the ten paired seeds. The hardware amendment
permits either profile, but one provider-selection receipt must bind all seeds
0–9 to the same profile, AMI ID, Region, and private ECR image digest.

The protected cohort is currently blocked until the six 29M diagnostics and all
artifact/qualification gates in `configs/preregistration-v3.yaml` are complete.
Packaging and dry-run planning do not authorize a protected launch.

## 1. Select and verify a package

Set one of the two literal paths above:

```bash
set -euo pipefail
PROFILE=cluster/profiles/aws-p6-b300.48xlarge-v3.json
OUT_ROOT=../memorysplit-releases/aws-gpu-v3

# DRY RUN: validates the clean Git snapshot and writes nothing.
python scripts/package_aws_gpu_handoff.py \
  --profile "$PROFILE" \
  --out-dir "$OUT_ROOT"
```

Review the one-object JSON report, especially `provider`, `release_id`, and
`sha256`. Publication is a separate explicit action:

```bash
# APPLY only after the dry-run report is approved.
python scripts/package_aws_gpu_handoff.py \
  --profile "$PROFILE" \
  --out-dir "$OUT_ROOT" \
  --apply
```

Verify the published `RELEASE-AWS-GPU-V3.json` before upload or extraction:

```bash
RELEASE="$OUT_ROOT/<reviewed-release-id>/RELEASE-AWS-GPU-V3.json"
python scripts/verify_aws_gpu_v3_release.py \
  --profile "$PROFILE" \
  --release "$RELEASE" \
  --source-root .
```

Two builds from the same clean commit and profile must have identical archive,
checksum, and receipt bytes. Publication is exclusive and atomic; an existing
release directory is never replaced.

## 2. Security and capacity rules

Use a temporary federated operator role and a dedicated EC2 instance role. Do
not use static credentials. Instances must have no public IP, use private
networking and SSM, and be named by explicit approved instance IDs in every
fleet and lifecycle operation. `msctl` plans and controls existing instances;
it never provisions EC2 capacity and never purchases a Capacity Block.

Resolve the AWS Deep Learning Base AMI with Single CUDA on Ubuntu 24.04 through
the AWS public SSM parameter, then pin and review the returned immutable AMI ID.
Build into a private tag-immutable ECR repository and use only the resulting
`repository@sha256:...` reference.

P6 Capacity Block discovery and purchase are outside `msctl`. Purchase is a
separate, paid, non-cancellable operation requiring approval of the exact
offering ID and exact price before the dry-run and purchase calls. Never infer
purchase approval from package, provider-selection, or launch approval.

Continue with `docs/AWS-GPU-V3-RUNBOOK.md`. The IAM request template is
`docs/AWS-GPU-ACCESS-REQUEST.md`.
