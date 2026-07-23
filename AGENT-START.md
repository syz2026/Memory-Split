# MemorySplit v2 Illumina agent start

This release supports only the Illumina `usfc-prd` Slurm cluster. The login VM
has four vCPUs: use it only for authentication, release verification, planning,
small metadata reads, and `sbatch`. Run downloads, hashing, corpus builds,
training, and evaluation through Slurm.

## Safety contract

1. Verify the external ZIP SHA-256 and `RELEASE.json` before extraction.
2. Read `DATASET-POINTER.json` and set `MS_SHARED_ROOT` to an operator-approved
   directory under `/illumina`.
3. Invoke `python -m msctl ...` and parse its one JSON object from stdout.
4. Run the dry-run first and inspect every hash, run ID, resource, and command.
5. Use `--apply` only after the required signed approval receipt is present.
6. Stop on any authentication, approval, provenance, dataset, checkpoint,
   Slurm-reconciliation, or hash error. Do not work around a failed gate.
7. Never print, copy into a job, or include in a report any secret, token, or
   `MSCTL_APPROVAL_KEY`.
8. Never duplicate or manually resubmit a run ID. Let `msctl status` reconcile
   `squeue` and `sacct`; use `msctl resume` only with a matching checkpoint
   receipt.

## Ordered bring-up

```text
msctl auth check
msctl capacity check
msctl env ensure
msctl dataset ensure
msctl dataset verify
msctl runs render
msctl submit
msctl status
msctl evaluate
msctl collect
msctl cleanup plan
```

Mutation is always a separate invocation with `--apply`. Paid GPU submission,
resume, cancellation, sealed scoring, and cleanup additionally require an
unexpired approval bound to the exact provider, release SHA-256, run-manifest
SHA-256, job limit, and GPU-hour limit.

Seed 0 is one precommitted Dense/Split90 pair: three A100s per arm, symmetrically
scheduled, with the seventh allocated A100 reserved for evaluation or
verification. Report it only as `directional_only (1/5 complete)`. Its outcome
must not decide whether seeds 1–4 continue.
