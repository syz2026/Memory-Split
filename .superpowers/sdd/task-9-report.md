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
supports the current `2629a8e` bindings, including the clean source tree ID.
Illumina metadata requires and binds `preregistration_sha256`. Every run config
requires exact AULC `snapshot_steps: [1358,3396,6791,10187,13582]`;
`snap_frac` is rejected. The AWS profile matches the current package schema,
and every environment entry must be exact-version and SHA-256 pinned.

The AWS runbook provides dry-run-first commands for current pricing, quota,
versioned S3, an explicitly selected On-Demand P5, SSM, private role-only AWS
state, one Task 6-owned destructive bootstrap, digest-pinned canaries with
executable child-PID waits and checkpoint-bound resume, exact signed approvals,
current `.result.instance_id` parsing, sequential/four-instance modes, and
digest-pinned canonical evaluation. Capacity creation remains outside the
lifecycle.

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
destructive bootstrap, and printed-but-unexecuted evaluation.

## Verification

- Task 9: `45 passed`, including a real package from cohort/AULC `70c1951`.
- Current Illumina packager branch: `20 passed`.
- AWS P5 packager at `2629a8e`: `79 passed`.
- Total focused tests: `144 passed`; all runbook Bash blocks pass `bash -n`.
- `git diff --check` passed. Ruff was unavailable in the active environment.

## Integration concerns

- AWS package `2629a8e` has the current receipt/profile/environment envelope,
  but its config validator and fixture still emit `snap_frac`; a raw package is
  rejected until that owner ports `70c1951` `snapshot_steps`.
- Task 6 `84bd93c` still parses AWS `source` without `tree` and renders host
  `runner.py` evaluation. The runbook uses the corrected envelope and executes
  `ac4b5a0` in a digest-pinned container, but Task 6 must publish its pending
  source/evaluator correction before production launch.
