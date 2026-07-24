# Selected-Provider Lifecycle Bridge Report

## Status

`DONE_WITH_CONCERNS` on `integration/provider-aware-lifecycle`, based exactly on
clean reviewed head `5c0b4b691efb9b84f7856e6f17fc57e1b0240912`.

Implementation commits before this report:

- `fdd080fb5ff9d34d54039de39b331b58f78759fa`
  (`feat: bridge selected provider lifecycle authority`)
- `b9ba606c7d70ed985ab29ab87587d2c7bc57dc2c`
  (`fix: close provider package test scope`)

All pre-bridge atomic lifecycle, hardware-selection, objective-controls,
runtime, and profile-qualification commits remain ancestors unchanged.

## Authority bridge

`msctl/aws_lifecycle.py` adds the sole selected lifecycle authority:

- `admit_provider_lifecycle(...)` accepts fixed authority/repository/runtime
  lock/runtime evidence/runtime SBOM/objective paths, exact selection version,
  versioned store, instance identity, and production identity/approval
  verifiers.
- It re-enters `admit_cohort_provider_selection`, requiring independently
  authenticated Dense and Split90 arm scopes with one selection.
- It authenticates the reviewed runtime lock and closed SBOM parser, then loads
  the fixed objective-controls amendment and aggregate.
- It returns `AuthenticatedProviderLifecycle` and one strict
  `ProviderLifecycleBinding`; no selected public API accepts a caller-built
  binding, provider, profile ID, or profile hash.

The binding closes over selected provider/profile, hardware amendment,
provider-selection hash/version, runtime lock/SBOM, qualification evidence,
environment/canary/approval receipt and approval-key hashes, objective
aggregate, placement/instance identity, exact seed, and both arms.

## Manifest, release, and package

- `RunManifestV3` and selected release/archive metadata carry the complete
  lifecycle binding. Parsers are closed and release binding rejects
  cross-selection, cross-version, runtime-SBOM, profile, and objective drift.
- V3 instantiation calls fixed-path lifecycle admission; direct caller profile
  objects cannot authorize selected manifests. Explicit legacy manifests
  remain readable, while new V3 publication requires authority inputs.
- The package closure explicitly requires both P5/P6 profiles, hardware and
  objective amendments, all eight 29M configs plus their manifest, runtime
  build/lock/SBOM inputs, qualification code/worker, and bridge authority code.
- Unknown production paths remain rejected. Runtime packaging includes only
  the explicitly required package test/fixtures, avoiding test-only secret
  material.
- Selected package metadata is deterministic; the selected double-build test
  byte-matches both archives and release receipts.

## Launch, controller, bootstrap, and state

- `build_authenticated_launcher_manifest`,
  `load_authenticated_launch_plan`, and
  `AwsP5Backend.from_authenticated_selection` accept only fixed
  paths/store/verifiers and re-enter authority.
- Selected P5/P6 instance type and GPU product identity derive from the
  authenticated profile. Selected launch requires the selection-bound
  bootstrap receipt; release/bootstrap parsers accept either reviewed profile
  without weakening the legacy P5 entry points.
- Launch plans carry `lifecycle_binding`. Runtime config and Docker environment
  carry the reviewed operational metadata while scientific config bytes and
  scientific fingerprints remain unchanged.
- Operation intents, approval resources, paired state journals/run records,
  controller results, checkpoint scheduling, and resume all bind the same
  selected provider lifecycle. Selected state remains on the existing atomic
  paired-journal protocol.
- Every selected controller manifest validation re-runs lifecycle admission;
  exact-version resume therefore re-authenticates the persisted selection.

## Checkpoints and model snapshots

- Task 3C checkpoint timing, request-token acknowledgment, freshness,
  content-addressed object keys, exact object versions, lost-response recovery,
  cleanup, and atomic publication behavior are unchanged.
- Selected trainer checkpoint metadata, S3 object metadata, paired receipt,
  parser, verifier, controller HEAD checks, and resume state carry the complete
  lifecycle binding. P5/P6 or selection/runtime/objective cross-binding fails.
- Trainer production snapshots add strict run/arm, config fingerprint/hash,
  dataset receipt/build/order, source commit/tree, and every lifecycle field.
- `parse_model_snapshot_bytes(...)` provides evaluator-grade byte validation.
  Legacy snapshot reads require explicit `allow_legacy=True`.

Task 3D run finalization was not implemented.

## Strict TDD evidence

Observed RED before implementation included:

- missing `msctl.aws_lifecycle`;
- selected run manifest unknown fields and absent fixed-path instantiation;
- missing selected package metadata API and release parser rejection;
- package closure missing runtime/qualification/objective members;
- missing authenticated launcher manifest/controller constructors;
- absent model-snapshot byte parser and absent checkpoint operational metadata;
- checkpoint request constructor, receipt parser, and provenance verifier
  rejecting or failing to bind selected fields;
- selected paired state rejected by the closed state schema;
- committed-tree package dry-run rejecting broadly included runtime tests.

Each focused RED was followed by GREEN before the next slice.

## Verification

Fresh non-overlapping final suites passed:

- provider bridge + run-manifest + package: `197 passed`;
- launcher + checkpoint/resume + cleanup + atomic paired state: `292 passed`;
- trainer + DDP: `104 passed`;
- hardware + objective + runtime + qualification + shared contracts/canary:
  `573 passed`.

Total: `1166 passed`.

The package follow-up reran its full `102`-test suite. The committed repository
dry-run then passed with release ID `aws-p5-r1-aec1e481aa43b3e4` and planned
archive SHA-256
`3b1a006d23780c5b60b5a7d787d52c48ad5ef3571dc0985bd049b3779758152a`.

Changed-file `python -m py_compile`, worktree/range `git diff --check`, exact
base/ancestry, excluded `5678a7e`, and protected-scope checks passed. Frozen
preregistration, cohort assignment, and all `configs/360m-v3` bytes have zero
diff from `5c0b4b6`.

The ignored `.superpowers/sdd/task-3d-provider-brief.md` now names the real
bridge interfaces and explicitly supersedes its obsolete proposed authority
types. It is not committed.

## Concerns

- No live AWS, paid P5/P6, IMDS, versioned S3, Docker image, GPU/NCCL, or
  cryptographic qualification operation was performed. Those boundaries use
  deterministic injected tests.
- The committed-tree dry-run exercises the compatible P5 package entry point;
  selected P5/P6 metadata paths are covered by deterministic authenticated
  package double-build and bootstrap verification tests because no live
  provider-selection authority files exist in the repository.
