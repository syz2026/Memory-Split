# Selected-profile snapshot-evaluation planning report

## Status

`DONE` on `feat/snapshot-evaluation-planning-v3`, based exactly on
`77052ead68e9a384fb8c40fce70cacd4ad6e6038`
(`feat/sealed-eval-release-v3`).

The review-clean hardware authority history was replayed in the requested
order without squashing:

1. `708724c` -> `8c4892c9fad6c4d70f404ced9385cbc572750756`
2. `fb0dac4` -> `a921583516f558a073b34b529aa40b7fe7db9e29`
3. `acb13d6` -> `af307400bdfe604e8c00b58555ab6728a6ff2209`
4. `123713c` -> `39c8d29822393730388888c5e2d2aeb3ed6313b9`
5. `701eec8` -> `8810de4d9ec8497242b1d57223e7386dd496dd22`

Implementation commit:
`1be119c9a09aeee89994335b4c788f8cba04dcc0`.

## Delivered

- `StudyLockV3` now carries one fixed-key provider-selection binding with the
  receipt hash, exact version ID, hardware amendment, selected provider,
  closed P5/P6 profile identity, runtime lock, and qualification evidence.
  Every one of the 100 snapshot rows repeats the same selection hash/version;
  mixed selection, provider, profile, or evidence identities fail closed.
- Added `evals/confirmatory/aggregate.py`. It emits exactly 100 immutable
  `SnapshotEvaluationPlan` values in seed, Dense/Split90, then five-step order.
  Canonical `RunBindingV3` bytes bind the local checkpoint and study lock,
  checkpoint object/receipt evidence, sealed release, provider selection,
  evaluator profile/runtime identity, and profile-aware output identity.
  Missing, extra, reordered, replaced, or aliased plans are rejected.
- Added an isolated v3 runner path while retaining the existing v2 path. A
  v3 run needs only `run.json`, `study-lock.json`, and its exact
  `checkpoint.pt`, plus the four-file content-addressed sealed release. The
  real repository adapter loads architecture/configuration metadata from the
  hash-bound checkpoint, so no separate configuration file is required.
- V3 preflight validates the external/embedded lock hash, selected lock slot,
  every checkpoint/receipt/selection/profile/runtime field, the exact sealed
  release, all model-visible item/store bytes, and the output name. Evaluation
  rechecks mutable control/checkpoint inputs after submissions and before gold.
  Gold is opened only after every submission exists, then the sealed release
  and solver replay are verified.
- Per-snapshot publication is canonical and atomic no-replace. `output.json`
  binds every emitted artifact and run/lock/release/selection/checkpoint
  identity. `inference.json` remains cohort-scoped with
  `final_conclusion: null` and `cohort_aggregation_status: not_implemented`.
- Historical v2 parsing, readiness, runner output, CLI output, scoring, and
  reporting remain unchanged; an explicit regression proves an unrelated v3
  marker cannot switch a v2 run onto the new path.

## Strict TDD evidence

Production slices were preceded by focused failures:

- Study-lock tests failed collection because `ProviderSelectionBinding` did
  not exist.
- Planning tests failed collection because
  `evals.confirmatory.aggregate` did not exist.
- V3 runner tests reached the historical release-local v2 lock lookup and
  failed because no isolated v3 path existed.
- The real no-config checkpoint test failed because the repository adapter
  rejected `RunBindingV3`.
- Checkpoint and run/lock mutation tests first completed or opened gold
  instead of failing before gold access.
- The v2 compatibility test first misclassified a v2 release carrying an
  unrelated `sealed-release.json` marker.
- The selected-profile output test first produced identical P5/P6 names.

Each focused failure passed after its minimal implementation. Final evidence:

```text
planning + study lock + v3/v2 runner + sealing: 155 passed
bounded confirmatory v2/v3 regressions:         139 passed
hardware authority/profile/cohort regressions:  240 passed
python -m py_compile:                            passed
git diff --check:                                passed
```

## Scope audit

No cohort aggregation or confirmatory statistics, S3 fetcher, controller or
`msctl` evaluation dispatch, corpus generation, IaC, or live AWS operation was
added. Per-snapshot pair metrics are local scoring evidence only and cannot
produce a final scientific conclusion.

## Concerns

None.
