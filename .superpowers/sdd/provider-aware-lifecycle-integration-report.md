# Provider-Aware Lifecycle Integration Report

## Status

`DONE_WITH_CONCERNS` on `integration/provider-aware-lifecycle`, based exactly on
review-approved `integration/task3c-atomic-lifecycle-clean` at
`b7382e2a4154b21459016b610b937c4c04844ce7`.

The pre-report integrated head is
`a8a177a9b080e35e2f0c4e6cf1da4e4295c72438`. This report is committed
separately because tracked `.superpowers/sdd/*-report.md` files are the
repository convention.

## Integrated commit mappings

The reviewed commits were cherry-picked in the requested order:

1. `708724c539951878dea7ce7dbcc5b21824df5d39` became
   `2e7d8b96af779cc926b720a205c232f898d05631`.
2. `fb0dac439cb054fe3a7f31b21db814e82f6e9925` became
   `ea7b67ad79c5d044bd3930f0c504ea3306234831`.
3. `acb13d613911ec7aa8adaefdf149a9f82569aa57` became
   `ef2dd6ec50013f9b5f33abcda99207f7f82d8f2b`.
4. `123713cc9b40a48336c84e4c7e8f42751ecaeda7` became
   `50904469169382578c9fe32eced366e015ec711f`.
5. `701eec85367b3b84f4265888a6028d4b9916d14c` became
   `6984c51c0eeff92c11192532ccdbcf533300f78b`.
6. `e09da5bbe401182c54ed22b785099e61be0d1950` became
   `7ad563632c0f9315ee8969b4fca4ed3c0a5bbb3d`.
7. `7d95d36e58a194aaf967c9130c705c3920ab0ff9` became
   `adb100e553f9ee027e26997e94f493441dc5e43b`.
8. `29d68ed9b66fa53b2f7a55eb0e8b399fc0f836b8` became
   `a8a177a9b080e35e2f0c4e6cf1da4e4295c72438`.

The first seven mappings preserve stable patch IDs exactly:

```text
62d8b0667dd8bf4d29ce26b66a172837269c2bf4
da510ffc05ac761ac5943a464586d347a76a1689
ebf97d3670619974b233a1e489a0c9e4435e018c
147b452c2af7063626378149efaa54ca1bcd316e
3f46d8a59f557caa527fe3185b438016d6257c30
4085105f5163c0c316dc8a3fd28afe8e2ed524c5
761a08d613ce3c9ee643c238c09d0aa9a41b96af
```

## Conflict resolution

No production, `msctl`, `aws_contracts`, profile, or test conflict occurred.
The hardware and objective implementation trees byte-match their reviewed
source heads on every imported path.

The sole conflict was report-only
`.superpowers/sdd/objective-controls-amendment-report.md`. `29d68ed` appends
review corrections to a report introduced by omitted report-only commit
`46c1595`. The resolution created the final report from `29d68ed`'s exact bytes:
both source and integrated files have Git blob
`47f26cba7560cabfe84db19f69672773311beac8`.

Consequently only that report mapping changes stable patch ID, from
`b0d4a3d46d1078a25f1ab6708c9adfae51141547` to
`736e39f5d47653e74959e4f823584ffa834b69fe`.
`46c1595` is not in branch ancestry.

## Task 3D planning freeze

The intentionally ignored
`.superpowers/sdd/task-3d-provider-brief.md` was revised from the existing Task
3D brief. It now freezes:

- authenticated selected provider, profile ID, and profile hash;
- hardware amendment hash
  `d4cf13b587c751d27756ad7881e538facb7ea79305a098990a568a7b28b6fb14`;
- provider-selection receipt hash and exact non-null version ID;
- objective-controls aggregate hash
  `222844dbf68ad9ca48be2069b5eb3b771b90166252af7eb0a4a5d8db3631adb6`;
- runtime-lock, closed runtime-SBOM, qualification, environment, canary,
  approval-receipt, and approval-key hashes;
- repeated exact-selection admission before evidence, upload, and receipt
  publication;
- fail-closed cross-profile, cross-provider, cross-runtime,
  cross-qualification, and cross-selection behavior.

The receipt provider/profile fields are selected-authority outputs rather than
P5 literals. Task 3D was not implemented.

## Verification

Fresh final verification passed:

- exact-base focused baseline before integration: `472 passed`;
- atomic lifecycle and trainer/DDP regressions: `395 passed`;
- P5/P6 hardware authority: `240 passed`;
- objective-controls contract: `75 passed`;
- v3 run-manifest and AWS package: `172 passed`;
- shared AWS contracts, roundtrip, environment, canary, and argv: `173 passed`;
- changed-file `python -m py_compile`: passed;
- committed-range and worktree `git diff --check`: passed;
- exact base, eight-commit order, source-tree equality, patch identities,
  frozen-science, Task 3C behavior, and exclusions: passed.

The first atomic attempt used `/private/tmp`, whose pytest tree was owned by
UID/GID `501:0`; 33 token-parent security tests correctly rejected the caller
GID mismatch. The unchanged suite passed from the bounded worktree root owned
by `501:20`.

`5678a7e` is neither an ancestor nor patch-equivalent to any integrated commit.
Its stable patch ID
`97dc5d26c27b5e6bbd1b21d619bbd7b3812af38e` is absent. Frozen
preregistration, cohort assignment, all `configs/360m-v3`, Task 3C lifecycle
modules, trainer behavior, and package code have zero diff from the exact base.

## Concerns

- No live AWS, paid capacity, versioned S3, P5/P6 host, Docker, GPU, NCCL, or
  qualification-signature operation was performed; those boundaries remain
  covered by deterministic injected tests.
- The requested `29d68ed` report patch could not retain patch identity without
  its explicitly unrequested parent `46c1595`; exact final reviewed bytes were
  preserved instead.
