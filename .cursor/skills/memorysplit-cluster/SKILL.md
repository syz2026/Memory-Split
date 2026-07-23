---
name: memorysplit-cluster
description: Use when planning, launching, monitoring, resuming, evaluating, collecting, or cleaning MemorySplit v2 work on the Illumina usfc-prd Slurm cluster.
---

# MemorySplit cluster operations

Use `python -m msctl`; consume its single JSON object. Illumina
`usfc-prd` is the only supported provider.

## Required workflow

1. Read `AGENT-START.md`, `RELEASE.json`, and `DATASET-POINTER.json`.
2. Run `auth check`, `capacity check`, and the relevant ensure/verify command.
3. Run the requested lifecycle command without `--apply`.
4. Present the dry-run hashes, limits, run IDs, and Slurm argv for review.
5. Mutate only in a separate invocation with explicit `--apply`.
6. Require a valid approval for paid submission/resume, cancellation, sealed
   evaluation, or cleanup.
7. Reconcile status before retrying. Never duplicate or manually resubmit an
   active run ID.

## Stop conditions

Stop on any JSON error, unknown provider, auth failure, expired or mismatched
approval, hash drift, dirty release, symlink, dataset mismatch, unknown Slurm
state, active duplicate, or checkpoint-provenance mismatch. Do not replace
`msctl` with ad hoc `sbatch`, `scancel`, `rm`, or shell retries.

Never expose secrets or pass ambient environment variables into Slurm. In
particular, never print or export `MSCTL_APPROVAL_KEY`.

The login VM is planning-only. Heavy download, hashing, corpus construction,
training, and evaluation must run in Slurm jobs. Seed 0 uses symmetric 3+3
A100 training groups; GPU 6 is reserved for evaluation/verification.
