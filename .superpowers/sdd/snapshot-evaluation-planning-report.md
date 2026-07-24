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

## Critical/Important review remediation

Review-remediation status: `DONE_WITH_CONCERNS`.

Review base: `f290819cd26218c69ae44c44ea3eab5cef4942f5`.

Remediation commit:
`5269bb4733698f92476f830492866b6dc81f62c4`.

This addendum supersedes the original report's `checkpoint.pt`, marker-based
dispatch, path-read, and unqualified-output claims.

### Findings closed

1. **Production model snapshots.** `Trainer.save_snapshot` now emits an
   additive, model-only study snapshot v2 for the frozen v3 cohort. It contains
   exactly model weights, optimizer step, `model_cfg`, world size, data
   provenance, config fingerprint, and explicit seed/arm/run/model/data
   commitments. Historical snapshot writes and reads retain their prior exact
   schema. Planning binds `snapshots/stepNNNNNNN.pt` and content-addressed
   `snapshots/.../step-NNNNNNN/...` object keys. Evaluation uses
   descriptor-pinned bytes and `torch.load(..., weights_only=True)`, constructs
   the model directly from `model_cfg`, and rejects `cfg`, optimizer, data
   cursor, or RNG-bearing full checkpoints.
2. **Evaluator execution identity.** Provider-selection locks and run bindings
   now carry the approved environment, canary, approval, and public-key
   commitments plus fixed evaluator profile/runtime/environment inputs.
   Provider-qualified publication requires the real repository adapter,
   requested and observed CUDA, the exact selected GPU profile/count/name,
   exact PyTorch/CUDA runtime versions, and a PKCS7-authenticated AWS
   environment receipt. The complete evidence is re-read and re-authenticated
   before inference and after submissions, before gold opens. `output.json`
   binds both observations. CPU, MPS, and injected adapters publish only
   `test_only` evidence with `production_qualified: false`.
3. **Pinned input and output authority.** V3 run bindings, study locks, model
   snapshots, evaluator profiles, runtime locks, and environment receipts use
   bounded `O_NOFOLLOW` descriptor reads with owner, writable-mode, link,
   pre/post descriptor, and final-name checks. The output parent is
   descriptor-pinned; staging writes and atomic no-replace installation are
   directory-relative. Parent replacement fails, and a post-install authority
   failure atomically quarantines the intact output tree.
4. **Contract-driven dispatch.** Dispatch parses canonical `run.json` and its
   exact record type/schema before choosing v2 or v3. V3 remains v3 even when a
   release marker is missing or renamed, while unrelated or dual marker files
   cannot redirect an explicit v2 run. Duplicate keys, crossed schemas, and
   unknown identity fields fail closed.

The exact 100-slot order, cohort-wide provider selection, sealed-gold timing,
atomic output, cohort-only/no-conclusion inference, hardware authority commits,
and historical v2 runner behavior remain intact.

### Strict RED to GREEN evidence

- Trainer snapshot RED lacked version/config/study provenance; GREEN round
  tripped a real `Trainer.save_snapshot` artifact and preserved legacy writes.
- Evaluator RED rejected the production snapshot because it expected invented
  `cfg`; GREEN loads only exact `model_cfg` and rejects full checkpoints plus
  crossed step/arm/config/model/data/world identities.
- Input hardlink, unsafe-mode, same-byte replacement, and oversized-run RED
  cases were accepted or reached JSON parsing; all now fail at pinned reads.
- Marker RED cases routed v2/v3 by filenames; canonical contract identity now
  controls dispatch and rejects forged/ambiguous records.
- Provider-qualified RED accepted no execution contract; GREEN rejects
  CPU/MPS/fixtures, binds CUDA/PyTorch/device/AWS evidence, and fails changed
  post-inference authentication before gold.
- Output-parent and post-install RED races either published through a
  replacement or left a final tree; GREEN rejects the replacement and
  quarantines an intact installed tree.

Final verification:

```text
expanded focused planning/lock/runner/sealing suite: 198 passed
bounded confirmatory v2/v3 regressions:              139 passed
hardware authority/profile/cohort regressions:       240 passed
aws contracts including model snapshot keys:          14 passed
complete Trainer snapshot/compatibility suite:         85 passed
python -m py_compile:                                  passed
git diff --check:                                      passed
```

### Remaining concern

The host has no CUDA GPU and no live AWS environment, so the default production
CUDA/PKCS7 path was not exercised against physical hardware. Tests use the same
injected verifier/probe seams as the hardware authority suites; production
publication itself rejects those test-only adapter paths and requires observed
CUDA plus the authenticated receipt.

## Snapshot-evaluator re-review remediation

Re-review status: `DONE_WITH_CONCERNS`.

Re-review base: `aff46ffc76aa8fd4d4b6431727841e60ce01b2fb`.

Re-review fix:
`c7c39a53c37f7abab773a314da86eb363d69bc65`.

### Important findings closed

1. `StudyLockV3` now records the exact training-config SHA and fingerprint,
   model-config hash and model identity, full data-provenance hash, dataset
   receipt/build/ordered-stream commitments, run/arm identity, world size,
   update size, and provider-selection identity. All five steps of one
   seed/arm must share the complete invariant tuple. Dense/Split90 pairs must
   share model, base data, source-transitive runtime, selection,
   qualification, and evaluator invariants; cohort-wide matched fields are
   also exact. Only the frozen step-derived token count plus snapshot
   object/version and paired-receipt identities may vary across steps.
   `Trainer.save_snapshot` emits and validates the expanded identity while
   historical snapshot schemas remain readable and writable.
2. V3 output publication now hashes and descriptor-validates every staged
   artifact, captures exact installed file content and metadata, then runs one
   composite final authority operation that verifies the pinned parent,
   parent-name-to-installed-directory descriptor, exact membership, file
   identity, metadata, and content. A deterministic hook proves replacing the
   output after all prior checks cannot return success. Every failed pinned
   staging or installed tree is atomically renamed to an intact quarantine;
   the V3 authority path performs no pathname deletion. The returned SHA-256
   of `output.json` is `authoritative_commitment`; `output_dir` is marked
   `informational_reopen_required`.
3. Direct execution bootstraps the repository root before importing project
   modules in both `__main__.py` and `runner.py`. Clean subprocesses with no
   `PYTHONPATH` now produce the same one-object machine contract for
   `python -m evals.confirmatory`, direct package-main execution, and direct
   runner execution from outside the repository.

### Re-review RED to GREEN evidence

- Cross-step config/model/data mutations and cross-arm model/base-data
  mutations were accepted before the new lock-level invariant registries.
- An installed-directory replacement after previous checks returned success,
  and a staged write failure called pathname deletion instead of preserving a
  quarantine.
- Direct package-main and runner subprocesses failed before argument parsing
  with `ModuleNotFoundError` for `evals` or `cluster`.

Final verification:

```text
expanded focused planning/lock/runner/sealing suite: 217 passed
bounded confirmatory v2/v3 regressions:              139 passed
hardware authority/profile/cohort regressions:       240 passed
aws contracts including model snapshot keys:          14 passed
complete Trainer provenance/compatibility suite:      85 passed
clean direct/module CLI subprocess matrix:              3 passed
python -m py_compile:                                  passed
git diff --check:                                      passed
```

The only remaining concern is unchanged: this host has no physical CUDA GPU or
live AWS environment, so production CUDA/PKCS7 enforcement is covered through
the authenticated injected seams rather than a paid hardware execution.
