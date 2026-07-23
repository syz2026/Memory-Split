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
supports the current source-tree bindings and exact strict AWS profile SHA.
Illumina metadata requires and binds `preregistration_sha256`. Every run config
requires exact AULC `snapshot_steps: [1358,3396,6791,10187,13582]`;
`snap_frac` is rejected. AWS releases must declare the exact
profile-bound `runtime_attested` contract from `e01cee3`: the digest
environment name/pattern and an at-launch receipt authenticated by the AWS
instance identity document PKCS7. Static environment locks, static receipt
hashes, malformed receipt requirements, and fabricated profile bindings are
rejected.

The AWS runbook provides dry-run-first commands for current pricing, quota,
versioned S3, an explicitly selected On-Demand P5, SSM, private role-only AWS
state, one Task 6-owned destructive bootstrap, digest-pinned canaries with
executable child-PID waits and checkpoint-bound resume, exact signed approvals,
current `.result.instance_id` parsing, sequential/four-instance modes, and
digest-pinned canonical evaluation. It verifies the exact release environment
contract and immutable `image@sha256` runtime; the selected-instance bootstrap
must emit and authenticate the exact runtime receipt before paired training.
It never fabricates a local/static environment receipt. Capacity creation
remains outside the lifecycle.

## Files

- `scripts/verify_cohort_releases.py`
- `tests/test_verify_cohort_releases.py`
- `docs/AWS-P5-360M-RUNBOOK.md`
- `.superpowers/sdd/task-9-report.md`

No packager, config, `msctl`, cluster runtime, trainer, corpus generator,
evaluation, plan, or specification file was modified.

## TDD evidence

The compatibility fixes were test-first. RED showed exact AULC configs and a
real Illumina package from `70c1951` rejected by the stale `snap_frac` parser.
Current AWS `source.tree` and profile fixtures failed the stale Task 3/5 shape.
Runbook RED covered stale refs, missing profile ARN, wrong JSON nesting,
missing approvals, subshell-owned PIDs, non-resuming probes, duplicate
destructive bootstrap, and printed-but-unexecuted evaluation. The authoritative
environment RED then showed a no-lock `runtime_attested` release rejected by
the stale lock schema; fully rehashed static-lock, missing receipt fields, and
a fabricated environment profile hash remain rejected. A real release built
by the final AWS packager is accepted.

## Verification

- Task 9: `47 passed`, including real packages from cohort/AULC `70c1951` and
  AWS P5 packager `e01cee3`.
- Current Illumina packager branch: `20 passed`.
- Final AWS P5 packager `e01cee3`: `83 passed`.
- All 14 runbook Bash blocks pass `bash -n`.
- `git diff --check` passed. Ruff was unavailable in the active environment.

## Integration concerns

- AWS package `e01cee3` is aligned and tested; its lineage ref is `81543c2`.
- Task 6 commit `a236158` rejects the retired static environment path but has
  not published the selected-instance PKCS7 attestation operation/result
  interface required by this runbook. Production launch remains blocked until
  Task 6 emits and authenticates the exact `e01cee3` receipt before training.
