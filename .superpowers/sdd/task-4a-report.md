# Task 4A report: provider-aware snapshot evaluation bridge

## Status

`DONE` on `integration/provider-aware-lifecycle`, exact base
`9fd85f272cf096b10e8f91fbfb20a82826534eb4`.

Commits:

1. `8fbae08` `feat: bridge provider-aware snapshot evaluation`
2. `41531a3` `fix: drop unused planning import`
3. report commit (`docs: report provider-aware evaluation bridge`)

## Delivered

- **Slice 0**: deleted the dead earlier padded `snapshot_object_key`
  definition from `msctl/aws_contracts.py`; the live unpadded Task 3F key
  format is unchanged and a regression pins exactly one definition.
- **Slice 1**: `parse_model_snapshot_bytes` is the one authoritative parser.
  It now loads with `torch.load(..., weights_only=True)`, explicitly rejects
  full-checkpoint fields (`cfg`/`data`/`opt`/`rng_by_rank`), accepts exactly
  four shapes (legacy 5-field behind `allow_legacy=True`, study-only
  8-field, operational 37-field, combined selected-study 39-field), gains
  `require_study_identity` which rejects any shape without
  `snapshot_version`/`study_identity`, and rejects every shape lacking
  operational fields whenever expected operational metadata is supplied
  (including legacy). Study validation still checks snapshot version 2,
  config fingerprint, exact identity fields, canonical model/data hashes,
  and seed/arm/run/cohort/update geometry; flat lifecycle metadata still
  goes through `validate_lifecycle_operational_metadata`.
  `Trainer.save_snapshot` already emitted all four shapes; trainer resume
  now also demands study identity whenever the config declares the
  protected study cohort.
- **Task 3D**: `_admit_arm_evidence` passes `require_study_identity=True`,
  so finalization admits only combined 39-field snapshots for selected
  protected runs.
- **Slice 2**: `evals/confirmatory/study_lock.py` gains
  `SeedLifecycleBinding` (21 fields: placement, boot, SBOM, objective
  controls, source, run manifest, per-arm operational config hashes, and
  finalization/collection receipt identities). `StudyLockV3` gains
  `seed_lifecycles` with exactly ten entries, seeds 0..9 ascending;
  placement/SBOM/objective/source fields are cohort-uniform while boot,
  run-manifest, per-arm config, and receipt identities may vary per seed;
  receipt URIs must sit on one shared durable object root at their
  canonical `run_receipt_key`/`collection_receipt_key` locations with
  non-null version IDs and positive finalization bytes; receipt and
  run-manifest content cannot alias across seeds.
  `StudyLockV3.lifecycle_binding(seed)` reconstructs (and `__post_init__`
  proves) a real `ProviderLifecycleBinding` for every seed.
- **RunBindingV3** gains the slot's 18 lifecycle/receipt fields, validates
  them by reconstructing `ProviderLifecycleBinding`
  (`RunBindingV3.lifecycle_binding()`), and checks source SHA-1s,
  operational-config hash, canonical receipt URIs/bytes/versions, and one
  shared receipt root. Planning fills them from the lock's seed lifecycle
  with the arm-scoped operational config hash.
- **Receipt-driven planning**: `CollectionReceiptEvidence` plus
  `plan_snapshot_evaluations(..., collection_receipts=...)` requiring ten
  exact Task 3F receipts; `validate_snapshot_evaluation_plans` and both
  aliases require the same receipts. Only after full receipt proof are the
  exact 100 byte-stable bindings emitted in frozen slot order.
- **Slice 3**: the runner's duplicated eight-field snapshot parser now
  delegates to `train.trainer.parse_model_snapshot_bytes` through
  function-local imports (no trainer/torch at module import; verified),
  building the exact expected operational metadata with
  `lifecycle_operational_metadata(study_lock.lifecycle_binding(seed), ...)`
  in preflight and `binding.lifecycle_binding()` in the adapter. All
  existing equality checks stay: step, world size, config fingerprint,
  model-config hash, data-provenance hash, exact study identity, nonempty
  model, full-checkpoint rejection, and sealed-gold timing. Selection
  validation now requires every new binding field to equal the lock's seed
  lifecycle plus arm-scoped operational config and whole-lifecycle equality.
  `SeedLifecycleBinding` and `CollectionReceiptEvidence` are exported from
  `evals.confirmatory`.

## Schema migration policy

- `memorysplit.confirmatory.study-lock.v3` and
  `memorysplit.confirmatory.run-binding.v3` keep their record types and
  `schema_version` 3. Strict exact-field parsing makes field growth
  fail-closed: a pre-bridge lock (no `seed_lifecycles`) and the pre-bridge
  44/45-field run binding are both rejected with "fields are not exact".
  No production v3 lock exists, so no migration path is provided.
- V2 lock/run-binding contracts and behavior are unchanged (group 2
  regression run below).
- One deliberate v3 invariant evolution: the lock previously required a
  distinct checkpoint receipt per (seed, step); it now requires the one
  collected per-seed terminal paired checkpoint receipt to be identical
  across all ten slots of a seed and unique across seeds. This is the only
  checkpoint receipt Task 3F durably collects, so slot-level equality with
  collection receipt row 14 is provable; the old shape could never be
  receipt-proven.

## Authority mapping

`ProviderLifecycleBinding` is reconstructed from two sources:

- cohort `ProviderSelectionBinding`: cohort ID, provider, profile ID/hash,
  hardware amendment, selection hash + exact version, runtime lock, and the
  five qualification hashes (evidence, environment, canary, approval,
  approval key);
- per-seed `SeedLifecycleBinding`: runtime SBOM, objective-controls
  contract, account/instance/boot/region/availability-zone/purchase
  placement, and the seed.

`StudyLockV3.lifecycle_binding(seed)` and `RunBindingV3.lifecycle_binding()`
must be equal at preflight; expected snapshot metadata is
`lifecycle_operational_metadata(lifecycle, run_id/arm/config/dataset/source
from the binding)`, and the snapshot must embed it exactly (combined
39-field schema only, `require_study_identity=True`). Placement fields
omitted from collection receipts (account/region/zone/purchase) are proven
by lock↔binding equality plus snapshot-embedded operational metadata.
Evaluator profile/runtime/environment fields stay separate but share the
same cohort provider selection.

## Receipt-planning ordering

1. exactly ten `CollectionReceiptEvidence` values, ascending seed order;
2. evidence URI/SHA-256/version must equal the lock seed lifecycle's
   collection receipt identity;
3. `parse_seed_collection_receipt_bytes(...,
   expected_binding=study_lock.lifecycle_binding(seed))` re-proves canonical
   bytes, payload hash, the closed Task 3F field set, and lifecycle
   equality;
4. cross-binding: top-level `run_receipt` equals the lock finalization
   receipt (URI/SHA/bytes/version); `run_manifest_sha256`,
   `source_commit`, and `source_tree` equal the seed lifecycle; the dataset
   triple equals the locked snapshots; release identity is uniform across
   all ten receipts;
5. the ten snapshot rows must match the lock's seed slots by arm, step,
   SHA, version, and root+key ("not durably collected" otherwise), and the
   checkpoint-receipt row must match the per-seed slot receipt;
6. only then are exactly 100 byte-stable `RunBindingV3` plans emitted in
   frozen seed/arm/step order and re-validated against the canonical plan.

## Strict TDD evidence

Every slice had observed RED before GREEN:

- Slice 0: `assert 2 == 1` on `def snapshot_object_key(` count →
  60 contract tests pass after deleting the dead padded definition.
- Slice 1: five focused failures — non-weights pickle `DID NOT RAISE`,
  full-checkpoint message mismatch, and `TypeError: unexpected keyword
  argument 'require_study_identity'` ×3 → 103 trainer tests pass.
- Task 3D: `DID NOT RAISE FinalizationError` for a selected snapshot
  without study identity → 79 finalization tests pass after the
  `require_study_identity=True` flip and combined-snapshot fixtures.
- Slice 2: study-lock module collection failed with
  `ImportError: cannot import name 'SeedLifecycleBinding'` → 78 lock tests
  pass.
- Planning: module collection failed on missing
  `CollectionReceiptEvidence` → 64 planning tests pass.
- Slice 3: 46 of 72 runner tests failed with the old duplicated parser
  rejecting real combined snapshots (`model snapshot fields are not
  exact`) → 72 runner tests pass after delegation.

## Verification

```text
group 1 focused bridge (trainer, finalization, lock,
  planning, runner, records, collect):                 535 passed
group 2 confirmatory v2/v3 regression:                 209 passed
group 3 lifecycle/hardware authority:                  495 passed
group 4 DDP provenance:                                  8 passed
full suite (tests/):              2913 passed, 2 failed*, 2 deselected
changed-file python -m py_compile:                      passed
git diff --check (worktree and 9fd85f2..HEAD):          clean
CLI matrix (python evals/confirmatory/runner.py --help,
  python -m evals.confirmatory --help,
  python -m evals.confirmatory.runner --help,
  python scripts/run_train.py --capabilities-json):     all exit 0
lazy-import boundary (import runner/aggregate/study_lock
  leaves torch, train.trainer, msctl.aws_collect out):  passed
package dry-run at clean HEAD
  (scripts/package_aws_p5_handoff.py):                  ok=True, dry_run=True
frozen-science zero-diff (configs/, preregistration,
  cohort, corpus, inference/AULC, IaC, runbooks,
  packaging): only the 15 scoped files changed
```

*The two full-suite failures are `tests/test_verify_cohort_releases.py::
test_accepts_release_built_by_final_aws_packager` and
`...::test_runbook_is_dry_run_first_and_covers_complete_p5_lifecycle`; both
fail identically at the exact base `9fd85f2` (v2 cohort-release runbook /
source-commit expectations not satisfiable on this integration branch) and
are untouched by this task. Group 3 needed one unsandboxed rerun because the
sandbox forbids the fixtures' `git init` under `/tmp`; all 495 tests pass on
the real host.

## Files

- `msctl/aws_contracts.py` (dead key removal)
- `train/trainer.py` (unified parser, resume study identity)
- `cluster/aws/p5/run_finalization.py` (`require_study_identity=True`)
- `evals/confirmatory/study_lock.py` (`SeedLifecycleBinding`,
  `seed_lifecycles`, `lifecycle_binding`, per-seed checkpoint receipt)
- `evals/confirmatory/aggregate.py` (`RunBindingV3` growth,
  `CollectionReceiptEvidence`, receipt-driven planning)
- `evals/confirmatory/runner.py` (parser delegation, extended selection
  validation)
- `evals/confirmatory/__init__.py` (exports)
- `tests/study_lock_fixtures.py` (shared lifecycle/receipt fixtures)
- focused tests: `tests/test_aws_contracts.py`, `tests/test_trainer.py`,
  `tests/test_aws_run_finalization.py`,
  `tests/test_confirmatory_v3_study_lock.py`,
  `tests/test_confirmatory_snapshot_planning.py`,
  `tests/test_confirmatory_v3_runner.py`,
  `tests/test_confirmatory_v3_records.py`

## Self-review

- Per-arm operational config hashes are one lock field per seed per arm
  (constant across that arm's five slots, may differ between arms) and may
  differ across seeds, because the twenty frozen launch configs embed their
  seed; making them cohort-uniform would have made every real seed>0
  snapshot unevaluable against its embedded `config_sha256`.
- The runner adapter re-proves the snapshot against
  `binding.lifecycle_binding()`; preflight has already proven that binding
  equals `study_lock.lifecycle_binding(seed)`, so both paths pin one
  authority.
- v2 evaluation, sealing/gold timing/output quarantine, provider
  selection/qualification/runtime authorities, inference/AULC/statistics,
  Task 3C–3F lifecycle/controller/state/receipts, selected
  evaluate/cleanup blocking, configs, preregistration, cohort, corpus,
  IaC, runbooks, and packaging are all unchanged (zero diff outside the
  scoped files).
- An unused import found in self-review was removed in `41531a3`; Ruff is
  not installed on this host, so linting relied on `py_compile` plus
  review.

## Exclusions and blockers

- No cohort aggregation/statistics; `inference.json` stays cohort-scoped
  with `final_conclusion: null` and
  `cohort_aggregation_status: not_implemented`.
- No study-lock builder from real receipts, no S3 fetcher, no selected
  evaluate controller enablement, and no cleanup enablement.
- Production CUDA/AWS/S3 execution remains untested on this host; those
  boundaries are covered by deterministic injected tests.
- Pre-existing (base-reproducible) v2 cohort-release verification failures
  noted above remain for a future owner.
