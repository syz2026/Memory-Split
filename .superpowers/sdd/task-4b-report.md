# Task 4B report: receipts-driven StudyLockV3 builder and publication

## Status

`DONE` on `integration/provider-aware-lifecycle`, exact base
`9bd99a5` (clean review-approved Task 4A head).

Commits:

1. `c964264` `feat: build and publish receipts-driven v3 study lock`
2. `955ca45` `test: track lock-builder members in package fixture`
3. report commit (`docs: report receipts-driven study lock builder`)

## Delivered

- **`evals/confirmatory/lock_builder.py`** (new): the closed public
  interface from the brief — frozen `SeedCollectionEvidence`,
  `RunStudyIdentity`, and `PublishedStudyLock` dataclasses plus
  `extract_run_study_identities`, `build_study_lock_v3`, and
  `publish_study_lock` — along with `read_local_evidence_bytes` for the
  CLI, `LockBuildError(ValueError)` with a stable `code`, and the
  publication authority. Module import stays light: torch, the trainer,
  `msctl.aws_collect`, `cluster.aws.p5.run_finalization`, and
  `msctl.contracts` are imported only inside functions (subprocess-proven).
- **`scripts/build_confirmatory_study_lock.py`** (new): dry-run-default
  CLI with required `--evidence-index` and
  `--sealed-evaluation-release-sha256`, `--publish` requiring
  `--output-root`, exactly one canonical JSON object on stdout for
  help/success/error, no writes in dry-run, and no network use anywhere.
  The evidence index is a strict schema-1 object (duplicate-key and
  non-finite rejecting) with exactly ten seed rows ordered 0..9 and, per
  seed, ten snapshot rows ordered Dense five steps then Split90 five
  steps; every referenced payload is descriptor-read locally and rehashed
  downstream.
- **`evals/confirmatory/__init__.py`**: exports the three dataclasses and
  three functions.
- **Package closure**: `REQUIRED_MEMBERS` gains exactly
  `evals/confirmatory/lock_builder.py` and
  `scripts/build_confirmatory_study_lock.py` (no broad prefix); the
  synthetic packaging fixture tracks the two new members; a new closure
  test pins both as included.
- **Fixtures**: `tests/study_lock_fixtures.py` gains
  `build_collected_evidence`, which builds ten seeds of *real* canonical
  receipt payloads (Task 3C paired checkpoint, Task 3D finalization,
  Task 3F collection) and 100 real `torch.save` combined 39-field
  snapshots whose hashes/byte counts back every receipt row, plus
  mutation hooks for drift tests. `build_seed_lifecycles` gains an
  optional `evidence_refs` parameter (defaults preserve all Task 4A
  behavior; the 142 pre-existing lock/planning tests pass unchanged).
- **`tests/test_confirmatory_v3_lock_builder.py`** (new): 66 tests
  covering the brief's eleven RED groups (group 11 lives in
  `tests/test_package_aws_p5_handoff.py`).

## Strict TDD evidence

Observed RED before GREEN for every slice:

- Builder + CLI + fixtures: the full focused test file failed collection
  with `ModuleNotFoundError: No module named
  'evals.confirmatory.lock_builder'` (required RED test 1, module absent
  before implementation); the fixture-backward-compatibility run
  (142 passed) was taken before implementation.
- Package closure: `test_study_lock_builder_closure_is_required` failed
  with both members reported as `Extra items in the left set` before the
  `REQUIRED_MEMBERS` additions.
- Packaging fixture closure: after the `REQUIRED_MEMBERS` change, the
  lifecycle/hardware authority group failed with 82 errors
  (`required release member is not tracked:
  evals/confirmatory/lock_builder.py`) until the synthetic
  `_minimal_repo` fixture tracked the two new members (RED → GREEN for
  commit 2).
- GREEN: after implementation, first full run was 65/66 with one test-side
  assertion bug (unsorted expected quarantine listing), then 66/66.

## Evidence derivation and proof ordering

Exactly ten `SeedCollectionEvidence`, seeds 0..9 ascending (position must
equal the parsed receipt seed). Per seed, in order:

1. collection payload rehashed against its declared SHA-256, then parsed
   by `parse_seed_collection_receipt_bytes` (Task 3F) with its
   URI/SHA/version identity and no expected binding;
2. finalization payload hashed against the collection top-level
   `run_receipt` (SHA-256 and byte count; the reviewed Task 3F parser has
   already proven row 16 equals that reference), then parsed by
   `parse_run_finalization_receipt_bytes` (Task 3D);
3. checkpoint payload hashed against the collection top-level
   `checkpoint_receipt` (and, via the same parser guarantee, row 15),
   then parsed by `parse_paired_checkpoint_receipt_v3` (Task 3C);
   `MsctlError` is translated to `LockBuildError`;
4. every provenance/lifecycle field shared by the closed Task 3F and
   Task 3D field sets (computed from `COLLECTION_RECEIPT_FIELDS` ∩
   `_RECEIPT_FIELDS` minus `checkpoint_receipt`/`receipt_type`/
   `request_id`) must be equal, and the finalization checkpoint-receipt
   reference must equal the collection's;
5. finalization arms' five snapshot object rows and log row must equal
   the collection rows pairwise (URI/SHA/bytes/version); run ID and world
   size are enforced seed/arm-scoped and 4 by the reviewed parser;
6. checkpoint receipt must be selected/provider-aware (legacy
   `provider_selection_sha256 is None` fails); its per-arm
   run/config/fingerprint/world rows must equal the finalization arms;
   its terminal checkpoint object rows must equal collection rows 13–14
   (placement); every `_RECEIPT_BINDING_FIELDS` lifecycle field and the
   nine provenance fields must equal the collection receipt;
7. one cohort `ProviderSelectionBinding` is derived from the collection
   receipt (fixed `PROVIDER_SELECTION_S3_KEY`), and all ten must be
   identical; release identity must be uniform;
8. one `SeedLifecycleBinding` is derived with placement
   (account/zone/region/purchase) from the checkpoint receipt,
   boot/instance/SBOM/objective/source/manifest from the collection
   (equal to finalization by step 4), arm config hashes from the
   finalization arms, and the exact finalization/collection receipt
   identities; the checkpoint receipt's embedded lifecycle must equal
   `_reconstructed_lifecycle_binding(selection, lifecycle)` exactly;
9. ten `StudySnapshotBinding` rows per seed use collection snapshot
   SHA/key/version, the seed's one collected checkpoint receipt, the
   dataset triple, the finalized arm config/fingerprint, and the
   extracted study identity.

Snapshot identity extraction (`extract_run_study_identities`) re-admits
the evidence itself (caller data is never trusted), requires the mapping
to cover exactly the 100 canonical `(seed, arm, step)` slots, and per
snapshot: O_NOFOLLOW descriptor open; regular, singly linked
(`st_nlink == 1`), owner-safe, identity-stable across the read; streamed
SHA-256 and byte count against the collection row; expected operational
metadata built with `lifecycle_operational_metadata` from the
receipt-derived `ProviderLifecycleBinding` plus run/arm/config/dataset/
source fields; function-local
`parse_model_snapshot_bytes(payload, expected_operational_metadata=...,
require_study_identity=True)`; slot step/world-size equality; study
identity run/config/dataset equality against the receipts; fingerprint
equality against the finalized arm; then `model_cfg_sha256`,
`model_identity`, `data_provenance_sha256`, and `config_fingerprint` are
extracted and must be invariant across the run's five steps.
Operational-only, study-only, legacy, full-checkpoint, byte/hash drift,
internal step drift, and cross-step identity drift all fail (tested).

`build_study_lock_v3` re-admits the evidence, requires exactly twenty
`RunStudyIdentity` values in frozen seed/arm order with fingerprints
equal to the finalized arms, constructs `StudyLockV3` (frozen v3
preregistration hash, caller's sealed release hash, one provider
selection, ten seed lifecycles, exactly 100 slots in frozen order), and
then proves it: every collection receipt is reparsed with
`expected_binding=lock.lifecycle_binding(seed)`, the canonical lock
SHA-256 is computed, and `plan_snapshot_evaluations(...,
collection_receipts=...)` must yield exactly 100 valid Task 4A plans.
The builder over the canonical payload fixtures reproduces the hand-built
fixture lock byte-for-byte and SHA-for-SHA (tested).

The two config identities remain distinct and explicit:

- each seed lifecycle's per-arm operational config hash comes from the
  finalized arm `config_sha256` (the frozen launch-config file bytes);
- each snapshot slot's `training_config_sha256` comes from the snapshot's
  embedded `study_identity.config_sha256` (the actual runtime config bytes,
  including reviewed launch-time overrides).

Extraction requires the training-config hash to remain invariant across one
run's five snapshots; it no longer incorrectly requires equality with the
different operational-config hash. The evaluator binds and checks both.

## Publication authority

`publish_study_lock` re-proves the lock against the ten receipts through
`plan_snapshot_evaluations` before touching the filesystem, then reuses
the reviewed in-package sealing primitives (`evals.confirmatory.sealing`
is unmodified):

- existing safe output root only (`_open_directory` + owned/mode checks),
  exclusive `flock`, quarantine-blocker rejection, and no-replace
  collision rejection before staging;
- private 0o700 staging via `_make_staging`; the single member
  `study-lock.json` is written O_EXCL/O_NOFOLLOW, fsynced, fchmod 0o444,
  and descriptor-pinned (`_PinnedFile` path/descriptor identity and
  content re-read);
- atomic no-replace rename (`renameatx_np`/`renameat2`); collisions are
  never reused or replaced (`RELEASE_EXISTS`);
- installed verification re-checks entry/membership/content, fsyncs, and
  the last authority operation is the composite parent/name/descriptor/
  membership/content check (`_assert_lock_member` on the pinned member
  plus `_assert_final_directory_binding`), with a deterministic
  `_run_mutation_hook("publish_before_final_binding")` race boundary;
- any failed staging or install is atomically renamed to an intact
  unpredictable quarantine (`.sealed-release-quarantine-<128-bit hex>`)
  through the sealing quarantine machinery; there is no pathname
  `unlink`/`rmdir` anywhere in the authority path (tests patch both to
  raise);
- `PublishedStudyLock.authoritative_commitment` is the lock SHA-256;
  `output_dir`/`study_lock_path` are informational
  (`informational_reopen_and_verify`).

## Verification

```text
RED: focused file collection error (module absent); package closure
  member assertion; 82 authority-group packaging errors pre-fixture
group 1 focused (brief command: lock builder, lock, planning,
  runner, collect, trainer):                            501 passed
tests/test_confirmatory_v3_lock_builder.py alone:        66 passed
group 2 confirmatory v2/v3 regression:                  209 passed
group 3 lifecycle/hardware authority:                   495 passed
group 4 DDP provenance:                                   8 passed
tests/test_package_aws_p5_handoff.py:                   106 passed
full suite (tests/):        2980 passed, 2 failed*, 2 deselected
changed-file python -m py_compile:                       passed
git diff --check (worktree and 9bd99a5..HEAD):           clean
CLI matrix (python evals/confirmatory/runner.py --help,
  python -m evals.confirmatory --help,
  python -m evals.confirmatory.runner --help,
  python scripts/run_train.py --capabilities-json,
  python scripts/build_confirmatory_study_lock.py --help,
  python -m scripts.build_confirmatory_study_lock --help): all exit 0
lazy-import boundary (import lock_builder + CLI leaves torch,
  train.trainer, msctl.aws_collect,
  cluster.aws.p5.run_finalization out; subprocess):      passed
package dry-run at committed HEAD:   ok=True, dry_run=True,
  release_id=aws-p5-r1-8759d18721aa6eed
frozen-science zero diff (configs/, cluster/, msctl/, train/,
  corpus, IaC, runbooks, verifier, v2, sealing, aggregate,
  study_lock, runner, inference): only the 7 scoped files changed
```

*The two full-suite failures are `tests/test_verify_cohort_releases.py::
test_accepts_release_built_by_final_aws_packager` and
`...::test_runbook_is_dry_run_first_and_covers_complete_p5_lifecycle` —
the same two pre-existing v2 cohort-release failures documented at the
Task 4A base and explicitly out of scope for this task; neither file is
touched by this diff. Authority/DDP/package groups and the full suite ran
on the real host (unsandboxed) because fixtures `git init` under `/tmp`.

## Files

- `evals/confirmatory/lock_builder.py` (new)
- `scripts/build_confirmatory_study_lock.py` (new)
- `tests/test_confirmatory_v3_lock_builder.py` (new)
- `evals/confirmatory/__init__.py` (exports)
- `scripts/package_aws_p5_handoff.py` (two `REQUIRED_MEMBERS` entries)
- `tests/study_lock_fixtures.py` (payload-backed evidence builder,
  `evidence_refs`)
- `tests/test_package_aws_p5_handoff.py` (closure test, fixture members)

## Self-review

- Receipt parsing is never duplicated: the only parsers invoked are
  `parse_seed_collection_receipt_bytes`,
  `parse_run_finalization_receipt_bytes`,
  `parse_paired_checkpoint_receipt_v3`, and
  `parse_model_snapshot_bytes`, all function-locally imported; row-16/15
  equality with the top-level references is inherited from the reviewed
  Task 3F parser rather than re-implemented.
- Publication reuses the reviewed sealing primitives inside the same
  package instead of a second copy; only the single-member staging/verify
  helpers are lock-specific because the sealing versions hard-code the
  four release member names/modes and `sealing.py` is out of scope.
  Quarantine names therefore share the `.sealed-release-quarantine-`
  prefix and `*.staging` blockers, which keeps one operator cleanup
  contract.
- Collection receipt S3 `version_id` cannot be falsified from local
  bytes; it flows into the lock verbatim and remains provable only by the
  later versioned HEAD/receipt replay, exactly like Task 4A planning.
- `RunStudyIdentity` carries the snapshot-embedded training-config SHA
  separately from the finalized arm's operational-config SHA. Canonical
  fixtures deliberately use different values and assert the distinction, so
  production runtime overrides cannot be hidden by synthetic equality.
- The `PATH_AUTHORITY` constant is imported from sealing rather than
  redefined (caught in self-review before commit).
- Ruff is not installed on this host; linting relied on `py_compile`
  plus review (same as Task 4A).
- v2 evaluation, sealing/gold timing, Task 3C–3F receipt modules,
  controller/state, trainer, provider authority,
  inference/AULC/statistics, selected evaluate/cleanup blocking, configs,
  preregistration, cohort, corpus, IaC, runbooks, and verifier are all
  unchanged (zero diff outside the seven scoped files).

## Exclusions and blockers

- No cohort aggregation/statistics; no selected evaluate/cleanup
  enablement; no S3 fetcher/controller/state changes; no live AWS/CUDA.
- Dry-run writes nothing (tested against the full observed evidence
  tree); publication is the only writing path and only under
  `--publish --output-root`.
- The two pre-existing v2 cohort-release failures noted above remain out
  of scope for a future owner.
- Real production evidence (ten S3-collected receipt bodies and 100
  snapshot bodies) does not exist yet; the builder is proven over
  payload-real fixtures that admit through the production parsers.

## Review correction: production config identities

The task review proved the original config-hash equality was unsatisfiable for
real trainer output: the finalized arm binds the frozen launch-config file,
while `study_identity.config_sha256` hashes the runtime config containing
operational metadata and runtime path overrides.

Correction:

- `RunStudyIdentity` now includes `training_config_sha256`;
- extraction records that value from each snapshot and enforces only the
  correct five-step run invariant;
- StudyLock snapshot `training_config_sha256` uses the extracted runtime hash;
- SeedLifecycleBinding operational config hashes remain the finalized
  launch-file hashes;
- canonical fixtures use
  `digest("runtime-config:{seed}:{arm}")`, explicitly different from
  `digest("{arm}-operational-config")`.

Review-fix verification:

```text
tests/test_confirmatory_v3_lock_builder.py: 67 passed
focused Task 4B bridge group:               502 passed
```

The added study-only snapshot case also closes the report's prior test-coverage
overclaim. No trainer, receipt, lock, runner, or frozen-science schema was
weakened; the existing Task 4A evaluator already validates the two fields
independently.
