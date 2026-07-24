# Task 3C Report: Durable Paired Checkpoint Mirroring and V3 Resume

## Status

Complete on `feat/memorysplit-v3-aws-n10`, starting from
`beb38f061c07ebfd63175c955c342e1728a440fd`.

Implementation commit:

- `27370a3` — `feat: add durable v3 checkpoint mirroring`

The session recovered from three PING interruptions by rebuilding state from
the worktree. No completed RED/GREEN work was discarded or restarted.

## Delivered

- Added `cluster/aws/p5/checkpoint_mirror.py` with:
  - strict `memorysplit-trainer-checkpoint-v1` metadata reads;
  - descriptor-pinned concurrent Dense/Split90 staging;
  - content-addressed checkpoint and receipt publication;
  - checksum, byte, metadata, and non-`null` S3 version verification;
  - lost-PUT recovery only through exact HEAD;
  - a monotonic 1080/1200-second scheduler with 120-second killable forked
    attempts and five-second retries; and
  - one publisher shared by periodic and interruption handling.
- Trainer checkpoint writes now atomically publish adjacent `ckpt.meta.json`
  after installing `ckpt.pt`. Metadata binds checkpoint version, step, world
  size, config fingerprint, data identity/cursor/sidecar, and the installed
  inode identity without changing checkpoint schema 3 or scientific config.
- Added strict `AwsPairedCheckpointReceiptV3`, nested checkpoint/data/object
  records, canonical-byte parsing, full provenance verification, and canonical
  checkpoint key helpers.
- Production launch manifests now carry release-receipt, environment-receipt,
  source run-manifest, dataset-receipt, source-tree, instance, and boot
  identities into the runtime mirror.
- Paired supervision polls mirroring without blocking child exit or IMDS.
  Missing the durability deadline returns `CHECKPOINT_STALE`; interruption
  finishes the shared attempt or starts an immediate one and falls back only
  to the last complete pair.
- V3 resume now:
  - requires the URI/SHA-256/version-ID triple and rejects the legacy local
    receipt option;
  - GETs the exact receipt version and HEADs both exact checkpoint versions
    before approval or state mutation;
  - verifies every release, manifest, environment, dataset, config, object,
    checksum, byte, metadata, and version binding;
  - materializes the receipt and both checkpoints with `--version-id`;
  - verifies staged receipt/checkpoint bytes before archiving prior outputs;
  - binds the receipt and both object versions into approval, operation
    identity, remote argv, and schema-2 controller state; and
  - accepts only exact replay while rejecting another version, partial state,
    active/successful prior commands, and uncertain terminal state.
- Legacy schema-1/2 local receipts and schema-1 controller state remain
  readable. Legacy interruption candidate-v3/marker-v1 artifacts remain
  audit-only and are not converted.
- The v3 package requires the checkpoint producer and all controller/resume
  runtime dependencies.
- V3 evaluation, cleanup, finalization, corpus generation, IaC, runbook, and
  verifier behavior were not enabled or implemented.

## RED/GREEN evidence

Every production slice began with a focused failing test. Commands below were
run from the Task 3C worktree with `PYTHONDONTWRITEBYTECODE=1`, pytest cache
disabled, and bounded temporary roots.

### Trainer generation metadata

RED:

```bash
python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-trainer-metadata \
  tests/test_trainer.py::test_checkpoint_metadata_binds_installed_generation
```

Observed: `FileNotFoundError` for missing `ckpt.meta.json`.

GREEN:

```bash
python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-green-trainer-metadata \
  tests/test_trainer.py::test_checkpoint_metadata_binds_installed_generation
```

Observed: `1 passed`.

### Strict metadata reader and legacy absence

RED commands:

```bash
python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-legacy-metadata \
  tests/test_aws_checkpoint_mirror.py::test_explicit_legacy_metadata_absence_is_readable

python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-valid-metadata \
  tests/test_aws_checkpoint_mirror.py::test_trainer_metadata_reads_the_exact_installed_generation

python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-mismatched-metadata \
  tests/test_aws_checkpoint_mirror.py::test_present_mismatched_trainer_metadata_fails
```

Observed respectively: missing module import, unimplemented valid metadata
parse, and `DID NOT RAISE` for a replaced checkpoint inode.

GREEN:

```bash
python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-green-metadata-reader \
  tests/test_aws_checkpoint_mirror.py
```

Observed at that checkpoint: `3 passed`.

### Atomic pair publication and fresh generations

RED commands:

```bash
python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-pair-publisher \
  tests/test_aws_checkpoint_mirror.py::test_distinct_fresh_arm_steps_publish_one_atomic_pair_receipt

python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-fresh-pair \
  tests/test_aws_checkpoint_mirror.py::test_both_arms_require_fresh_post_signal_generations
```

Observed: missing publisher API, then `DID NOT RAISE` while a deliberately
relaxed generation check admitted the stale Split90 checkpoint.

GREEN commands:

```bash
python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-green-pair-publisher \
  tests/test_aws_checkpoint_mirror.py::test_distinct_fresh_arm_steps_publish_one_atomic_pair_receipt

python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-green-fresh-pair \
  tests/test_aws_checkpoint_mirror.py::test_both_arms_require_fresh_post_signal_generations
```

Observed: `1 passed` for each. Dense step 5 and Split90 step 7 formed one
receipt; a stale arm produced no uploads.

### Versioned S3, lost PUT, and scheduler

RED commands:

```bash
python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-scheduler \
  tests/test_aws_checkpoint_mirror.py::test_scheduler_starts_at_1080_and_fail_stops_at_1200

python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-s3-store \
  tests/test_aws_checkpoint_mirror.py::test_s3_lost_put_response_recovers_only_through_exact_head

python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-lost-put-publisher \
  tests/test_aws_checkpoint_mirror.py::test_publisher_recovers_a_lost_put_response_by_exact_head
```

Observed: missing scheduler/store APIs, then explicit failure when a lost PUT
response was not allowed to recover through HEAD.

GREEN commands:

```bash
python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-green-scheduler \
  tests/test_aws_checkpoint_mirror.py::test_scheduler_starts_at_1080_and_fail_stops_at_1200

python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-green-s3-store \
  tests/test_aws_checkpoint_mirror.py::test_s3_lost_put_response_recovers_only_through_exact_head

python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-green-lost-put-publisher \
  tests/test_aws_checkpoint_mirror.py::test_publisher_recovers_a_lost_put_response_by_exact_head
```

Observed: `1 passed` for each. The mutation suite additionally rejects upload,
HEAD, empty version, `"null"` version, and changed-version failures without a
pair receipt.

### Receipt parser, verifier, keys, and legacy audit-only behavior

RED commands:

```bash
python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-receipt-parser \
  tests/test_aws_checkpoint_mirror.py::test_schema_three_pair_receipt_parser_accepts_publisher_bytes

python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-receipt-verifier \
  tests/test_aws_checkpoint_mirror.py::test_v3_receipt_verifier_binds_every_manifest_provenance

python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-checkpoint-keys \
  tests/test_aws_checkpoint_mirror.py::test_checkpoint_keys_accept_seed_bounds_and_reject_seed_ten

python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-legacy-v3 \
  tests/test_aws_checkpoint_mirror.py::test_legacy_local_receipts_are_audit_only_for_v3_manifests
```

Observed: missing parser, verifier, and key APIs; legacy schema-2 verification
then reached an `AttributeError` on a schema-3 manifest instead of rejecting it
as audit-only.

GREEN: all four focused tests passed. The final receipt matrix covers extra and
missing identity, exact numeric types, seed 10, timestamp/freshness drift, row
ordering, step/world-size drift, cursor/sidecar/data drift, object URI drift,
and empty/`"null"` versions.

### Launch, controller fetch, resume intent, and state

RED commands:

```bash
python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-launch-context \
  tests/test_aws_checkpoint_mirror.py::test_v3_launcher_manifest_propagates_complete_mirror_context

python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-launch-builder \
  tests/test_aws_checkpoint_mirror.py::test_launcher_manifest_builder_emits_closed_v3_mirror_context

python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-submit-context \
  tests/test_aws_checkpoint_mirror.py::test_v3_submit_intent_passes_complete_mirror_context_to_launcher

python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-controller-fetch \
  tests/test_aws_checkpoint_mirror.py::test_controller_fetches_exact_receipt_and_heads_both_versions_first

python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-versioned-intent \
  tests/test_aws_checkpoint_mirror.py::test_v3_resume_intent_version_pins_receipt_and_both_checkpoints

python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-state-v2 \
  tests/test_aws_checkpoint_mirror.py::test_schema_two_state_records_receipt_and_object_versions
```

Observed: closed-manifest unknown fields, unexpected builder keywords, access
to the forbidden V3 `dataset_sha256` alias, missing controller fetch API,
missing receipt URI/version intent arguments, and missing schema-2 state
bindings.

GREEN: each focused test passed after implementing the exact context,
GET-before-HEAD controller boundary, three versioned remote downloads, approval
bindings, remote intent validation, and schema-2 state.

### State fail-closed, package, and real resume planning

RED commands:

```bash
python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-red-package-members \
  tests/test_aws_checkpoint_mirror.py::test_v3_package_requires_checkpoint_producer_and_resume_dependencies

python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-partial-state \
  tests/test_aws_checkpoint_mirror.py::test_v3_partial_paired_state_fails_closed

python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-v3-submit \
  tests/test_aws_checkpoint_mirror.py::test_v3_submit_dry_run_is_enabled_with_complete_context

python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-existing-outputs \
  tests/test_aws_checkpoint_mirror.py::test_resume_plan_can_verify_existing_outputs_without_mutating_them
```

Observed: six required runtime members missing, partial V3 state silently
repaired, the dry-run submission plan rebuilt an unbound intent, and the
reviewed launcher rejected the read-only pre-archive output layout.

GREEN: focused tests passed after requiring dependencies, failing closed on
partial schema-2 state, reusing the authenticated submit intent, and adding a
read-only `allow_existing_outputs` planning boundary used only by resume.

### Compatibility regressions discovered during GREEN integration

- DDP layout test initially reported the intentional new `ckpt.meta.json` as
  an extra file; its exact expected layout was updated and the final trainer
  group passed 90 tests.
- Package fixtures initially lacked the newly mandatory producer/dependencies;
  the fixture was expanded and the final manifest/package group passed 172
  tests.
- Legacy resume adapter mocks lacked the new optional launch-plan attributes;
  access was made compatibility-safe and `tests/test_msctl.py` passed 157
  tests.
- One combined package run hit host `ENOSPC`. Only Task 3C-owned temporary
  directories were unlocked and removed, then the required files were run in
  bounded groups.

## Final verification

Required focused files were covered in disk-bounded groups:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-mirror-final-3 \
  tests/test_aws_checkpoint_mirror.py
```

Result: **50 passed**.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_trainer.py tests/test_ddp_trainer.py
```

Durable parent result after the metadata layout update: **90 passed**.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-launcher-final-3 \
  tests/test_aws_p5_launcher.py
```

Result: **135 passed**.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-contract \
  tests/test_aws_contract_roundtrip.py
```

Result: **26 passed**.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_run_manifest_v3.py tests/test_package_aws_p5_handoff.py
```

Durable parent result: **172 passed**. The subsequently hardened contract was
rechecked separately with `tests/test_run_manifest_v3.py`: **72 passed**.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-argv-final \
  tests/test_aws_argv.py
```

Result: **19 passed**.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-env-canary-final \
  tests/test_aws_environment_receipt.py tests/test_aws_canary.py
```

Result: **114 passed**.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-msctl-final-3 \
  tests/test_msctl.py
```

Result: **157 passed**.

Non-overlapping required-suite aggregate: **763 passed**.

Static verification:

```bash
python -m py_compile \
  cluster/aws/p5/checkpoint_mirror.py \
  cluster/aws/p5/launch_seed_pair.py \
  msctl/aws_contracts.py msctl/aws_launch_manifest.py \
  msctl/contracts.py msctl/aws_p5.py msctl/aws_argv.py \
  msctl/aws_resume_launch.py msctl/cli.py msctl/state.py \
  scripts/package_aws_p5_handoff.py train/trainer.py \
  tests/test_aws_checkpoint_mirror.py tests/test_ddp_trainer.py \
  tests/test_package_aws_p5_handoff.py tests/test_trainer.py

git diff --check
```

Result: both passed.

## Files

Created:

- `cluster/aws/p5/checkpoint_mirror.py`
- `tests/test_aws_checkpoint_mirror.py`
- `.superpowers/sdd/task-3c-report.md`

Modified:

- `train/trainer.py`
- `cluster/aws/p5/launch_seed_pair.py`
- `msctl/aws_contracts.py`
- `msctl/aws_launch_manifest.py`
- `msctl/contracts.py`
- `msctl/aws_p5.py`
- `msctl/aws_argv.py`
- `msctl/aws_resume_launch.py`
- `msctl/state.py`
- `msctl/cli.py`
- `scripts/package_aws_p5_handoff.py`
- `tests/test_trainer.py`
- `tests/test_ddp_trainer.py`
- `tests/test_package_aws_p5_handoff.py`

Intentionally unchanged:

- `msctl/operations.py`
- `corpusgen/`
- scientific V2/V3 config bytes and `ckpt_minutes: 30`
- evaluation/finalization/cleanup/IaC/runbook/verifier implementation
- legacy interruption candidate-v3/marker-v1 implementation

## Self-review

### Receipt and generation integrity

- Receipt, freshness, row, data, and object schemas are closed and exact.
- Canonical receipt identity includes the required trailing newline.
- Exact integer checks reject bool/float aliases.
- Dense and Split90 are ordered but may record different optimizer steps.
- Metadata is the commit record for a checkpoint generation and binds the
  singly linked installed inode; the large checkpoint is not read into memory
  before descriptor staging.

### Publication failure behavior

- Both metadata generations are baselined before either signal.
- Both staging workers are concurrent and killable through the attempt process.
- Checkpoint objects are uploaded and exactly HEAD-verified before receipt
  construction; the receipt is uploaded and HEAD-verified last.
- PUT/HEAD/checksum/length/metadata/version drift cannot emit a receipt.
- A lost PUT response has no authority until exact HEAD succeeds.
- Failed attempts leave any content-addressed orphan non-authoritative and do
  not clear `latest`.

### Timing and supervision

- Scheduler constants are exactly 1200/120/1080/5 seconds.
- Freshness starts at training start or the last verified receipt, not at retry
  start.
- Polling is nonblocking in the supervisor; process exit and IMDS continue.
- `CHECKPOINT_STALE` terminates both arms.
- Interruption shares the same scheduler/publisher and falls back only to the
  last complete receipt.

### Resume ordering and replay

- Exact receipt GET and both versioned HEADs occur before `resume()`, approval,
  state lock/mutation, archival, or SSM.
- Remote receipt and checkpoint GETs use exact version IDs.
- Staged receipt and checkpoint bytes are verified before prior output roots
  move.
- Approval, operation identity, and schema-2 state include receipt and both
  object versions.
- Exact replay is deterministic; another receipt/object version, partial state,
  active/successful prior commands, and uncertain status fail closed.

### Compatibility and exclusions

- Legacy local receipt parser remains distinct and rejects schema-3 manifests.
- Legacy state schema remains readable and is not rewritten as V3.
- RunManifestV3 received no `dataset_sha256` or `study_lock_sha256` alias.
- No scientific config, optimizer schedule, checkpoint schema, data cursor,
  evaluation, cleanup, finalization, corpus generation, IaC, runbook, or
  verifier behavior was added.

## Concerns

- No live paid AWS, versioned S3 bucket, P5, NCCL, Docker, or IMDS operation was
  performed. Those boundaries are covered with strict injected command/object
  stores and the existing launcher/canary suites.
- The host volume reached `ENOSPC` during one combined fixture-heavy run.
  Disk-bounded reruns passed after deleting only Task 3C temporary directories.
- The shell emitted an unrelated post-command `dump_zsh_state` warning on some
  successful commands; command exit codes and pytest/static results above were
  unaffected.

Report path:

`/Users/stephenzhang/Documents/MemorySplit/.worktrees/memorysplit-v3-aws-n10/.superpowers/sdd/task-3c-report.md`
