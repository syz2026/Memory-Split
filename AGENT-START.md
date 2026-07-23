# MemorySplit v2 Illumina agent start

This release supports only the Illumina `usfc-prd` Slurm cluster. The login VM
has four vCPUs: use it only for authentication, release verification, planning,
small metadata reads, and `sbatch`. Run downloads, hashing, corpus builds,
training, and evaluation through Slurm.

## Safety contract

1. Pass the unmodified `RELEASE.json` to every run lifecycle command. `msctl`
   resolves the ZIP beside that file and authenticates its type, exact byte
   count, external and internal SHA-256 manifests, metadata, and complete member
   set before rendering, submitting, resuming, evaluating, cancelling, or
   reconciling status.
2. This Illumina release may launch only seed 0. Attempting seed 1–4 from this
   archive is a contract violation; use the AWS P5 release for those seeds.
3. Read `DATASET-POINTER.json` and set `MS_SHARED_ROOT` to an operator-approved
   directory under `/illumina`.
4. Invoke `python -m msctl ...` and parse its one JSON object from stdout.
5. Run the dry-run first and inspect every hash, run ID, resource, and command.
6. Use `--apply` only after the required signed approval receipt is present.
7. Stop on any authentication, approval, provenance, dataset, checkpoint,
   Slurm-reconciliation, or hash error. Do not work around a failed gate.
8. Never print, copy into a job, or include in a report any secret, token, or
   `MSCTL_APPROVAL_KEY`.
9. Never duplicate or manually resubmit a run ID. Let `msctl status` reconcile
   `squeue` and `sacct`; use `msctl resume` only with a matching checkpoint
   receipt.
10. If submission is interrupted after intent is recorded, rerun the same
   command. `msctl` searches `squeue` and then `sacct` by its exact deterministic
   comment. Zero or multiple matches remain failed and recoverable; never invoke
   `sbatch` manually to work around that state.
11. Paired submit/resume operations persist one strict pair journal before
    either per-run record and repair an interrupted first record from it.

## Ordered bring-up

```text
msctl auth check
msctl capacity check
msctl env ensure --release RELEASE.json
msctl dataset ensure
msctl dataset verify --shared-root /illumina/... \
  --verification-out DATASET-VERIFICATION.json
msctl dataset verify --shared-root /illumina/... \
  --verification-out DATASET-VERIFICATION.json --apply
msctl runs render --shared-root /illumina/... \
  --dataset-verification DATASET-VERIFICATION.json \
  --environment-receipt /illumina/.../env/msctl-env-receipt.json
msctl submit --shared-root /illumina/... \
  --dataset-verification DATASET-VERIFICATION.json \
  --environment-receipt /illumina/.../env/msctl-env-receipt.json
msctl status --release RELEASE.json
msctl evaluate --shared-root /illumina/... \
  --dataset-verification DATASET-VERIFICATION.json \
  --environment-receipt /illumina/.../env/msctl-env-receipt.json
msctl collect
msctl cleanup plan
```

Supply `--release`, `--manifest`, and `--dataset-pointer` wherever the command
requires them. Run the full `dataset verify` command in a Slurm CPU job: it
recomputes all receipted artifact hashes plus ordered-stream and Merkle
commitments. Its applied output is a strict v1 verification receipt bound to the
release, run manifest, source lock, native corpus receipt, and pinned filesystem
identities. Login-side run commands can use that receipt and recheck only the
small native receipt and filesystem metadata. Alternatively, `--dataset-root`
requests a direct full verification. Exactly one of `--dataset-root` and
`--dataset-verification` is required.

For runtime commands, `--repo-root` is a controller-side extraction used only
for preflight. `msctl` never gives that mutable pathname to Slurm. It sends a
fixed, hash-bound bootstrap on `sbatch` stdin; the compute node opens the
release ZIP without following links, copies and hashes the same bytes into
private `SLURM_TMPDIR`, verifies the authenticated member manifest, safely
extracts to a content-addressed directory, and executes only that local copy.
`--shared-root` must be an approved directory under the pointer's `/illumina`
prefix, and the dataset must equal that root plus the exact pointer
`relative_path`; the bootstrap and extracted job both recheck this equation.
Ambient `MS_DATA_ROOT` and `MS_ENV_ROOT` values are not forwarded. Paid
operations perform environment, DDP, and evaluator checks before creating
state or invoking `sbatch`.

Environment receipts use the strict v2 schema and bind the release and lock
SHA-256, a deterministic Merkle commitment over the installed tree, the exact
Python executable, and installed distribution metadata. Paid submission
recomputes the tree, runs the Python probe, and explicitly rechecks `nvcc` plus
the `nvidia-smi` driver before `sbatch`. The fixed bootstrap requires
`/usr/bin/python3` on compute nodes.

Mutation is always a separate invocation with `--apply`. Paid GPU submission,
resume, cancellation, sealed scoring, and cleanup additionally require an
unexpired approval bound to the exact provider, release SHA-256, run-manifest
SHA-256, operation, rendered resource request, job limit, and GPU-hour limit.
Submit and resume currently request one 7-GPU, 36-hour allocation: approvals
must therefore permit at least 252 GPU-hours, even though only six GPUs train.
Evaluation is independently charged as its rendered GPU count times wall time.
Cancellation and cleanup request zero scheduler resources.

Per Slurm's `sbatch --export` contract, `--export=NAME,NAME=value` exports only
those user variables when `ALL` is omitted (Slurm still supplies its
`SLURM_*`/SPANK variables and documents implicit `--get-user-env` behavior).
The renderer intentionally uses that supported explicit form and never combines
assignments with `ALL` or `NONE`; each payload therefore establishes the final
boundary again with `env -i`. Do not add `#SBATCH --export=NONE`, ambient token
forwarding, or approval-key forwarding.

Seed 0 is one precommitted Dense/Split90 pair: three A100s per arm, symmetrically
scheduled, with the seventh allocated A100 reserved for evaluation or
verification. Report it only as `directional_only (1/5 complete)`. Its outcome
must not decide whether seeds 1–4 continue.

## Deliberate fail-closed integration gates

- Environment creation remains blocked until the operator supplies the exact
  site CPython and CUDA versions in the profile and generates
  `requirements-illumina.lock` for that Linux x86_64 contract. No portable or
  fabricated lock is included. Repackage the release after pinning them: run
  operations reject a local profile, Slurm script, or run config whose bytes
  differ from the authenticated release.
- Training remains blocked until `scripts/run_train.py` declares
  `MSCTL_DDP_CONTRACT = "memorysplit-ddp-v1"`, actually supports one
  three-process DDP writer per arm, and exposes
  `--resume-path <verified-checkpoint>`. Initial submit never uses
  `--resume auto`; resume exports and passes one exact path and SHA-256 per arm.
- Evaluation remains blocked until `evals/confirmatory/runner.py` declares
  `MSCTL_EVALUATOR_CONTRACT = "memorysplit-confirmatory-evaluator-v1"` and
  implements the preflighted `evaluate --run --sealed-release --device`
  interface.
- Production corpus materialization and its canonical parallel-corpus receipt
  must exist under the approved shared root before verification.
