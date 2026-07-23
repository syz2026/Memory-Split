# Task 9 report: split-provider cohort verification

Date: 2026-07-23
Branch: `feat/task-9-cohort-verifier`
Base: `c190429b1542bf8fdb518490e5d2c7b9a45d14d0`

## Outcome

Task 9 is implemented in the requested isolated worktree. The verifier opens
receipts, sidecars, and archives through pinned regular-file descriptors;
checks both external archive bindings and the exact internal `SHA256SUMS`
inventory; rejects unsafe, duplicate, non-regular, replaced, or partially
bound ZIP content; parses the frozen cohort and provider contracts; and emits
one canonical JSON decision.

The accepted cohort is exactly Illumina seed 0 plus AWS seeds 1–4, with one
Dense/Split90 pair per seed, unique run IDs, a clean common source commit, and
byte-identical cohort, corpus, and evaluation identities. The AWS receipt
supports the final richer path-plus-hash bindings for assignment, profile,
environment, dataset pointer, and configs.

The AWS runbook provides dry-run-first commands for current pricing, quota,
versioned S3, private On-Demand P5 launch, SSM, immutable bootstrap inputs,
eight-device NVMe RAID0, functional/paired-resume/100-update canaries,
sequential and four-instance modes, paired lifecycle operations, evidence
collection, termination, and frozen scientific labels.

## Files

- `scripts/verify_cohort_releases.py`
- `tests/test_verify_cohort_releases.py`
- `docs/AWS-P5-360M-RUNBOOK.md`
- `.superpowers/sdd/task-9-report.md`

No packager, config, `msctl`, cluster runtime, trainer, corpus generator,
evaluation, plan, or specification file was modified.

## TDD evidence

RED runs were observed for the missing verifier, an archive-name binding that
was initially accepted, descriptor replacement, the richer AWS receipt
contract, fully rehashed false provider profile and dataset-pointer semantics,
and the initial runbook contract. Each was followed by a focused GREEN run.

## Verification

- `pytest -q tests/test_verify_cohort_releases.py`: `33 passed`.
- Task 9 plus the assigned Illumina packager suite: `53 passed`.
- The prescribed verifier plus both finalized packager suites passed:
  `89 passed in 87.14s`.
- Ruff check, Ruff format check, bytecode compilation, IDE diagnostics, and
  `git diff --check` passed.

## Integration concerns

- Task 6 commit `0c9f20b` launches a new P5 inside `submit` instead of
  consuming the exact prelaunched instance ID. The runbook deliberately
  rejects a submit dry-run that renders a second `run-instances` call.
- Full confirmatory publication still requires the production checkpoint
  adapter identified in Task 8's report.
