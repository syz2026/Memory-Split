# Task 4C report: receipts-driven v3 cohort aggregation

## Status

`DONE_WITH_CONCERNS` on `integration/provider-aware-lifecycle`, based on clean
review-approved Task 4B head `c92980d`.

Commits:

- `73e6528` — `feat: aggregate provider-aware confirmatory cohort`
- `f659bf8` — `docs: record cohort aggregation progress`

## Recovery note

The implementation session was interrupted after producing the complete
uncommitted aggregation diff and focused tests, without a durable report. No
worker remained. The diff was preserved, inspected, and verified before commit.
The base lacks the public cohort aggregation/report symbols, matching the
brief's import RED, but the original per-slice RED command output is not
recoverable; this report does not invent it.

## Delivered

- Extended `evals/confirmatory/aggregate.py` from receipt-driven snapshot
  planning to full provider-aware cohort aggregation.
- Added and exported:
  - `SnapshotOutputReference`
  - `CohortReport`
  - `PublishedCohortReport`
  - `aggregate_cohort_outputs`
  - `validate_cohort_report`
  - `publish_cohort_report`
- Added `tests/test_confirmatory_v3_aggregate.py` with two payload-real,
  independently committed 100-output cohorts plus mutation/publication tests.

The stale reviewed aggregation branch was reference-only. Its receiptless
planning half, pre-bridge locks/bindings, duplicated runner schema constants,
and hand-rolled publication stack were not ported.

## Source authentication and replay

Aggregation requires exactly 100 ordered, unique
`SnapshotOutputReference` values and ten exact Task 3F
`CollectionReceiptEvidence` values.

For every output:

1. descriptor-pin the owned 0700 directory;
2. require exact ten-member membership and owned 0600 singly linked files;
3. no-follow bounded reads with pre/post identity and final path binding;
4. verify `output.json` SHA-256 commitment;
5. parse the lifecycle-complete StudyLockV3;
6. call current Task 4A `plan_snapshot_evaluations(...,
   collection_receipts=...)`, re-proving all ten Task 3F receipts and all 100
   durable snapshot rows;
7. require canonical `run.json` bytes to equal the planned 63-field binding;
8. replay the sealed release and selected evaluator before/after authority;
9. require the per-snapshot inference placeholder to remain cohort-scoped and
   conclusion-null;
10. replay every raw submission through the trusted solver;
11. recompute metrics and require persisted bytes to match;
12. retain exact pair-level primary/family outcomes only after every binding
    passes.

Before returning, the mutation-hook boundary is followed by a complete second
identity/membership/hash replay of all 100 pinned directories.

## Frozen inference

From the authenticated outputs:

- only optimizer step 13,582 contributes to the primary test;
- ten paired exact Split90-minus-Dense omnibus deltas;
- inclusive one-sided exhaustive sign flip over all 1,024 assignments, zeros
  retained;
- PCG64 seed 0, 20,000 seed→world→pair bootstrap draws, exact nearest-rank
  90% bounds, and strict open ±0.01 practical equivalence;
- supports-effect and supports-practical-null remain mutually exclusive;
- five-point no-interpolation right-step AULC per seed/arm, carrying exact raw
  area and informational exact normalized area (`raw_area / 13582`);
- graph and non-path terminal secondary contrasts with exact
  stable-lexicographic Holm correction for a family of two;
- authenticated instrument gates and frozen status axes.

No inference/statistics primitive was modified.

## Cohort report

The closed canonical `memorysplit.confirmatory.cohort-report.v3` carries:

- lock/preregistration/sealed-release/provider-selection identities;
- all 100 output commitments;
- all 100 exact snapshot-rate rows;
- ten ascending collection receipt identity rows copied from lock seed
  lifecycles;
- primary deltas/test/bounds;
- 20 AULC rows with raw and normalized exact values;
- both secondary tests/Holm values;
- instrument gates, status, and exclusive conclusion.

`validate_cohort_report` recomputes the complete report from source bytes.
Mutating any input commitment, collection identity, delta, bound, AULC,
secondary result, gate, status, or conclusion fails.

## Publication

Publication reuses the reviewed Task 4B/sealing authority pattern:

- full report replay before any write;
- safe existing output root and exclusive flock;
- quarantine-blocker and no-replace collision checks;
- private staging;
- one `cohort-report.json` member, mode 0444, O_EXCL/O_NOFOLLOW, fsynced and
  descriptor-pinned;
- atomic no-replace install to `cohort-report-<sha256>/`;
- installed content/membership verification;
- deterministic post-install mutation hook;
- final parent/name/descriptor/membership/content operation last;
- failed staging/install atomically quarantined intact;
- no pathname unlink/rmdir in the authority path.

Canonical report bytes are the authoritative commitment; paths are
informational and require reopen/reverification.

## Verification

Recovered focused test file:

```text
tests/test_confirmatory_v3_aggregate.py
29 passed
```

Final bounded groups:

```text
focused aggregate/planning/lock/runner: 310 passed
confirmatory v2/v3 regression:          209 passed
lifecycle/hardware authority:           495 passed
DDP provenance:                           8 passed
package handoff:                         106 passed
```

Full suite:

```text
3010 passed, 2 failed, 2 deselected, 1 warning
```

The two failures are the unchanged, base-reproducible v2 cohort-release
packager/runbook tests in `tests/test_verify_cohort_releases.py`, explicitly
outside this task.

Additional checks:

- changed-file `python -m py_compile`: passed;
- `git diff --check`: passed;
- committed-tree AWS package dry-run: `ok=true`, `dry_run=true`;
- frozen-scope checks: only aggregate exports/tests/progress changed;
- Task 4A/4B, runner, lock, trainer, receipts, lifecycle, v2, sealed-gold
  timing, provider authority, configs, corpus, IaC, and runbooks unchanged.

## Self-review and concerns

- Ten collection receipts are threaded through all public aggregation,
  validation, and publication entry points; no receiptless compatibility path
  exists.
- Current lifecycle-complete 63-field bindings and seed lifecycles are the only
  accepted v3 authority.
- Raw AULC remains the frozen authoritative integral; normalized AULC is an
  exact informational derivative.
- No selected evaluate/cleanup enablement, S3 controller/state, operator CLI,
  or live AWS/CUDA operation was added.
- Payload-real fixtures pass production parsers, but real protected outputs do
  not yet exist.
