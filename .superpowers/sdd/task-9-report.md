# Task 9 report: split-provider cohort verification

Date: 2026-07-23
Branch: `feat/task-9-cohort-verifier`
Base: `c190429b1542bf8fdb518490e5d2c7b9a45d14d0`

## Outcome

Task 9 is implemented in the requested isolated worktree. The verifier opens
receipts, sidecars, and archives through pinned regular-file descriptors;
checks both external archive bindings and the exact internal `SHA256SUMS`
inventory; builds a canonical member trie before reading members; rejects
unsafe, normalized-duplicate, colliding, non-regular, replaced, or partially
bound ZIP content; parses the frozen cohort and provider contracts; and emits
one canonical JSON decision.

The accepted cohort is exactly Illumina seed 0 plus AWS seeds 1–4, with one
Dense/Split90 pair per seed, unique run IDs, a clean common source commit, and
byte-identical cohort, corpus, and evaluation identities. The AWS receipt
supports the final richer path-plus-hash bindings for assignment, profile,
environment, dataset pointer, and configs. Illumina metadata requires and binds
`preregistration_sha256`. The AWS profile is checked against the complete Task
3 contract, and its packaged environment lock must be hash-bound and nonempty.

The AWS runbook provides dry-run-first commands for current pricing, quota,
versioned S3, an explicitly selected On-Demand P5, SSM, private role-only AWS
state, immutable bootstrap inputs, authorized eight-device NVMe RAID0,
digest-pinned container canaries, sequential and four-instance modes, paired
lifecycle operations, canonical confirmatory evaluation, evidence collection,
termination, and frozen scientific labels. Capacity creation is intentionally
outside the lifecycle.

## Files

- `scripts/verify_cohort_releases.py`
- `tests/test_verify_cohort_releases.py`
- `docs/AWS-P5-360M-RUNBOOK.md`
- `.superpowers/sdd/task-9-report.md`

No packager, config, `msctl`, cluster runtime, trainer, corpus generator,
evaluation, plan, or specification file was modified.

## TDD evidence

The review fixes were also test-first. RED runs showed the finalized Illumina
metadata and real `a85eb13` packager output being rejected, a missing
preregistration commitment being accepted, a fully rehashed
t3/zero-GPU/seed-0 profile and empty environment lock being accepted, all three
ZIP collisions reaching member parsing, and both pre-`fdopen` failures leaking
the archive descriptor. The stale runbook contract failed its final-interface
test. Each focused RED was followed by GREEN before the full gate.

## Verification

- Task 9: `43 passed`, including a release built by the exact integrated
  Illumina packager at `a85eb13`.
- Integrated Illumina packager at `a85eb13`: `20 passed`.
- AWS P5 packager at `48cfacb`: `57 passed`.
- Total focused tests: `120 passed`.
- Ruff check/format check, bytecode compilation, and `git diff --check` passed.

## Integration concerns

- Task 3/5 has no clean final review ref: its worktree currently carries
  uncommitted bootstrap, interruption, launcher, and test changes beyond
  `4b796a5`. Its committed report intentionally omits an environment lock.
- Task 7 `48cfacb` still validates the older flat storage/runtime profile
  fixture, not Task 3's strict nested profile. It also requires
  `requirements-aws-p5.lock`; the owning tasks must publish one reconciled
  profile/environment contract before a real AWS release can pass this verifier.
- Task 6 `0c9f20b` is stale. Explicit selected `--instance-id` plus
  `--terminate-at` exists only in a dirty worktree that currently has unresolved
  conflicts, so there is no releasable lifecycle ref for the runbook yet.
- Task 8 `f22f6ad` provides the canonical evaluator arguments and production
  repository checkpoint adapter used by the runbook; no Task 9 shim is needed.
