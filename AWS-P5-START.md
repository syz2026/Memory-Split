# AWS P5 handoff start contract

This release is authorized for **seeds 1–4 only**. Seed 0 belongs exclusively
to the Illumina release; any AWS request for seed 0 is forbidden and must be
rejected. Run one matched Dense/Split90 pair at a time on each `p5.48xlarge`:
GPUs 0–3 run Dense and GPUs 4–7 run Split90 (`4+4`).

## Verify before extraction

Work from an empty owner-only directory. Keep the ZIP, its adjacent
`*.sha256` file, and `RELEASE-AWS-P5.json` together.

1. Run `sha256sum -c <archive-name>.sha256`.
2. Confirm the archive name, byte count, SHA-256, source commit, provider,
   cohort hash, profile hash, environment hash, and seeds in
   `RELEASE-AWS-P5.json`.
3. Inspect `RELEASE-METADATA.json` and verify internal `SHA256SUMS` before
   using any archived code.
4. Reject duplicate ZIP names, path traversal, symlinks, an unexpected member,
   either seed-0 config, a missing arm, or any hash mismatch.

The archive is not a credential carrier. Use an EC2 instance role scoped to
the cohort S3 prefix and connect with Systems Manager. Do not export static AWS
keys, tokens, private keys, or passwords into the release or run manifests.

## Stage immutable inputs

Resolve the region-pinned AMI ID and immutable container digest through
`MS_AWS_AMI_ID` and `MS_CONTAINER_DIGEST`. Resolve the durable cohort prefix
through `MS_S3_ROOT`. Verify the AWS profile, environment lock, dataset
pointer, corpus receipt, token shards, and both target-weight sidecars before
paid work.

The eight local NVMe devices are ephemeral scratch only. S3 is the durable
authority for dataset receipts, paired checkpoints, logs, evaluations, and
collection receipts. Never terminate or reuse an instance until uploaded
objects have been independently rehashed from S3.

Sealed gold remains outside this archive and outside model-visible inputs.
Stage its separately authorized release only for the trusted evaluator, and
verify its external digest before evaluation.

## Dry-run, canary, then apply

Build the handoff without publishing:

```bash
python scripts/package_aws_p5_handoff.py \
  --source-root . \
  --out-dir dist/aws-p5
```

Publish one atomic, no-replace release set only after reviewing the JSON:

```bash
python scripts/package_aws_p5_handoff.py \
  --source-root . \
  --out-dir dist/aws-p5 \
  --apply
```

On the P5 host, run the profile/readiness checks, a two-GPU functional canary,
and the measured 100-update `4+4` throughput canary before a full seed. Launch
only an assigned seed in `{1,2,3,4}` and require both arms to start, checkpoint,
resume, evaluate, and collect as one pair.

Stop on a validity, integrity, infrastructure, or pair-completeness failure.
Observed effect direction is never a stop condition for the remaining seeds.
