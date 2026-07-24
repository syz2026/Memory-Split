# Task 3C Atomic Lifecycle Integration Report

## Status

`DONE_WITH_CONCERNS` on branch
`integration/task3c-atomic-lifecycle-clean`.

The branch starts exactly at
`39725404ca480503586c1890edcb2cb8f3d77cbc`. Its pre-report integrated head is
`569b9c71acbca1d19ebee6c793f0e3e8da5a5891`; this report is committed
separately because tracked `.superpowers/sdd/*-report.md` files are the
repository convention.

## Integrated commits

The reviewed atomic chain was fast-forwarded from the exact base, preserving
the original commits and order:

1. `1ee59be37a06dda36a613da5b8e8c69e83b872eb`
2. `b781db561ebe19c030d3fc3323e6e140b9f097e8`
3. `0dbf412982aeb0f0db7fa4f01842d3b6b444dbe2`
4. `b6b0e53bc9610bc5fe1255663297132a31365320`

The remaining commits were applied in the requested order:

5. `2cc4584716319f921e993761bad03b1544e239db` became `1f952ff621f6d3f50f72426b882254bc86fb7834`.
   Both have stable patch ID
   `8311b3b355da356f7820988ad998d2ca823f8dc7`.
6. `96d6223a427b6717c01132977e6c6464856e01f4` became
   `34c8b60b3ad36ac7b0a91f6f5e901ad9c6be13c0`.
7. `ad5ed7b536f4da98ecb8a2a754cf4a4297c3f6b5` became
   `569b9c71acbca1d19ebee6c793f0e3e8da5a5891`. Both have stable patch ID
   `63f7641df14de191d4e7da1f24ea21bae629bc8e`.

## Excluded commit and liveness independence

`5678a7e4ed3f02229a7fbce84228a88fe4cd725f` is not an ancestor of the
integration head. Its stable patch ID,
`97dc5d26c27b5e6bbd1b21d619bbd7b3812af38e`, does not occur in any commit in
the base-to-head range.

The liveness commit is semantically independent of `5678a7e`:

- `5678a7e` does not change `train/safeio.py`, `train/trainer.py`,
  `tests/test_trainer.py`, or `tests/test_ddp_trainer.py`.
- The liveness production files and DDP tests at `96d6223` are byte-identical
  to the requested base; its trainer-test prerequisite is supplied by the
  separately integrated `2cc4584`.
- The only shared path is the historical Task 3C report. The liveness report
  appendix applied cleanly after the exact `96d6223` addition.

Therefore `ad5ed7b` was included.

## Conflict resolution

The atomic chain, `2cc4584`, and `ad5ed7b` applied without state or test
conflicts. The only conflict was
`.superpowers/sdd/task-3c-report.md` while applying report-only `96d6223`:
its parent report contained the excluded `5678a7e` appendix.

Resolution retained the exact base report, omitted the entire excluded
appendix, and appended only `96d6223`'s exact 94 added lines. The expected and
resolved additions both had SHA-256
`e4d1f3d7537a7209b907a7f77cacc0554ef8769b704050bbe18c6bcd79dbcdf0`.
The resulting report commit has a different patch ID only because its hunk
context no longer contains the excluded appendix.

The preserved source-era appendix says `5678a7e` had landed in its original
worktree. That is historical evidence, not this integration branch's ancestry;
the exclusion checks above are authoritative.

## Verification

An exact-base baseline rerun passed `373` tests. Its first sandboxed attempt
had `370` passes and three fixture-only `git init` failures on
`.git/hooks/: Operation not permitted`; the unchanged command passed
unsandboxed.

All final suites used `PYTHONDONTWRITEBYTECODE=1`, disabled pytest's cache,
used dedicated user/group-owned bounded roots, and passed:

- paired lifecycle state: `69`
- checkpoint mirror and failed-attempt cleanup: `75`
- AWS launcher: `148`
- msctl lifecycle: `157`
- trainer and DDP: `103`
- run manifest and package: `172`
- environment, canary, argv, contracts, and contract roundtrip: `173`

Non-overlapping file-suite total: `897 passed`.

The explicit rollback, downgrade, strict token, wrong-parent-GID, abandoned
signal, periodic preservation, and two-rank liveness selection passed
`17` parametrized cases. Changed-file `python -m py_compile`, committed-range
`git diff --check`, and worktree `git diff --check` also passed.

## Concerns

- No live paid AWS, P5, Docker, NCCL, IMDS, or versioned S3 operation was
  performed; those boundaries remain covered by deterministic injected tests.
- `5678a7e` also carried launcher wiring/tests that were intentionally excluded
  with that unapproved patch. This integration does not silently reproduce
  those hunks.
