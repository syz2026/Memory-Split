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

## Failed-review remediation from `b3471e0`

Status: complete for the four requested Task 3C findings.

Implementation commit:

- `9e76a88` — `fix: harden Task 3C checkpoint supervision`

No Task 3D, dual-profile, scientific-config, timing-constant, receipt-schema,
or legacy-write behavior was added.

### Fixes

1. V3 resume now accepts exact integer seeds 0 through 9, rejects `bool` and
   10, installs the production paired checkpoint scheduler/request factory,
   and does not expose the legacy interruption publisher as a V3 fallback.
   The production request factory also imports and uses `secrets.token_hex`.
2. Baseline capture now obtains its generation identity from the same
   descriptor-pinned metadata read that validates the bound checkpoint
   descriptor. A pre-signal generation published during baseline validation
   is therefore recorded as the baseline rather than admitted as fresh.
3. `CheckpointMirrorScheduler.poll()` checks the existing 1200-second
   durability deadline before polling or accepting a completed attempt, so a
   late result cannot update `latest` or reset `fresh_at`.
4. Paired supervision owns scheduler cleanup in an unconditional `finally`.
   Success, child failure, requested shutdown, polling failure, stale
   fail-stop, interruption, and `KeyboardInterrupt` all cancel an active
   forked attempt. Scheduler polling exceptions also stop both training arms.

### RED evidence

Each focused test was added before its corresponding production change and
was observed failing for the intended reason:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-review-red-resume \
  tests/test_aws_checkpoint_mirror.py::test_v3_resume_output_archival_accepts_seed_bounds \
  tests/test_aws_checkpoint_mirror.py::test_v3_resume_output_archival_rejects_non_exact_seed \
  tests/test_aws_checkpoint_mirror.py::test_v3_resume_installs_paired_checkpoint_scheduler_without_legacy_fallback
```

Observed: `3 failed, 2 passed`; seeds 0 and 9 were rejected and no
`checkpoint_scheduler_factory` reached supervision.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-review-red-baseline-3 \
  tests/test_aws_checkpoint_mirror.py::test_baseline_pins_the_generation_validated_before_signal
```

Observed: `DID NOT RAISE TimeoutError`; the pre-signal replacement was
incorrectly treated as post-signal. The test's fixed `staged_at` was added
before production code so the RED exercised this race rather than wall time.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-review-red-deadline \
  tests/test_aws_checkpoint_mirror.py::test_scheduler_rejects_completed_attempt_at_durability_deadline
```

Observed: `DID NOT RAISE CheckpointStaleError`; the late pair was accepted.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3c-review-red-cleanup \
  tests/test_aws_p5_launcher.py::test_scheduler_poll_exit_cancels_attempt_and_terminates_pair
```

Observed: `1 failed, 1 passed`; a scheduler polling exception left both arm
processes alive.

The final regression files were also applied without production changes to an
isolated detached worktree at exact `b3471e0`. The combined seed, scheduler
factory, baseline race, late deadline, success/failure/shutdown cleanup,
polling exception, and keyboard-interrupt selection produced `13 failed`.

### GREEN evidence

```text
V3 resume/factory focus:             6 passed
Descriptor-pinned baseline focus:    6 passed
Durability-deadline focus:           3 passed
Supervisor cleanup focus:           10 passed
```

An isolated worktree containing only the staged `9e76a88` patch then passed:

```text
tests/test_aws_checkpoint_mirror.py   57 passed
tests/test_aws_p5_launcher.py         146 passed
changed-file py_compile               passed
git diff --check                      passed
```

### Final disk-bounded verification

All commands used `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, and an
explicit `/tmp/memorysplit-v3-task3c-*` `--basetemp`:

```text
tests/test_aws_checkpoint_mirror.py                         64 passed
tests/test_aws_p5_launcher.py                               146 passed
tests/test_trainer.py tests/test_ddp_trainer.py             90 passed
tests/test_aws_contract_roundtrip.py                        26 passed
tests/test_aws_argv.py                                      19 passed
tests/test_aws_environment_receipt.py tests/test_aws_canary.py
                                                             114 passed
tests/test_msctl.py                                         157 passed
tests/test_run_manifest_v3.py tests/test_package_aws_p5_handoff.py
                                                             172 passed
```

Non-overlapping total: **788 passed**.

Changed-file `python -m py_compile` and `git diff --check` both passed.
Launcher and mirror aggregates were rerun outside the sandbox after sandbox
policy denied fixture-created `.git/hooks` operations and cleanup; those
permission-only attempts were not product failures.

### Remediation concerns

- No paid AWS, versioned S3 bucket, P5, Docker, NCCL, or IMDS operation was
  performed.
- Independent pre-existing/concurrent Task 3C edits remained unstaged in
  `cluster/aws/p5/checkpoint_mirror.py`, `msctl/aws_p5.py`, and
  `tests/test_aws_checkpoint_mirror.py`; `9e76a88` deliberately excluded
  their content-addressed metadata, replay-state, fork-pipe, and associated
  test hunks.
- A first mirror aggregate overlapped one of those controller edits and
  observed one replay-state assertion failure. The immediate focused rerun
  and the stable full rerun passed; the isolated staged snapshot also passed.

## Review-fix appendix: seven-finding remediation from `b3471e0`

Status: complete for all seven findings in
`.superpowers/sdd/task-3c-review-findings.md`, base `b3471e0`.

Commits:

- `9e76a88` — `fix: harden Task 3C checkpoint supervision` (concurrent
  session; see concurrency note below)
- this commit — `fix: complete Task 3C checkpoint review fixes`

Concurrency note: this session and the `9e76a88` session remediated the same
worktree in parallel. `9e76a88` staged and committed whole files, so it
carries this session's `cluster/aws/p5/launch_seed_pair.py` fixes (findings
1, 3, 6, 7) and this session's `tests/test_aws_p5_launcher.py` tests together
with the other session's hardening (baseline generation pinning, scheduler
deadline ordering, v3 resume-launch scheduler wiring, seed-range fix, and
supervisor exception termination). This commit contributes the remaining
finding 2, 4, and 5 hunks that `9e76a88` explicitly excluded. The final
verification below ran on the exact combined tree that `HEAD` plus this
commit produces.

### Fixes and RED/GREEN evidence

All commands ran from this worktree with `PYTHONDONTWRITEBYTECODE=1`,
`-p no:cacheprovider`, and bounded `--basetemp=/tmp/ms3c-fix-*` roots that
were removed after each run.

1. Missing `secrets` import (critical). New
   `tests/test_aws_p5_launcher.py::test_production_checkpoint_request_factory_builds_real_requests`
   constructs `_production_checkpoint_scheduler` against a complete v3
   fixture plan and calls the real request factory (no injected fake).
   RED observed: `NameError: name 'secrets' is not defined` at
   `cluster/aws/p5/launch_seed_pair.py:2581`. GREEN after adding the
   `import secrets`: `1 passed`.

2. `ForkedCheckpointMirrorAttempt` proofs and pipe deadlock. Five focused
   tests exercise the real forked attempt:
   poll-before-exit nonblocking, successful result round-trip, `cancel()`
   kill/reap/descriptor-close (a SIGKILLed child cannot write its sentinel,
   `os.kill`/`os.waitpid` prove reaping, `os.fstat` proves the closed
   descriptor), malformed child payload as a completed failed attempt, and
   an 8 MiB result that exceeds the kernel pipe buffer.
   RED observed: `1 failed, 4 passed`; the large-payload attempt never
   completed in 20 seconds because the parent waited for child exit before
   reading while the child blocked writing. Fix: the parent pipe is now
   nonblocking and `poll()` drains it on every call before and after the
   `WNOHANG` reap; the attempt remains pollable and killable.
   GREEN: `5 passed` in 1.32s.

3. Supervisor durability fail-stop. New
   `test_supervisor_stale_fail_stop_terminates_and_cancels_active_attempt`
   injects a scheduler whose `poll` raises `CheckpointStaleError` without
   self-cancelling. RED observed: status/code/termination passed but the
   active attempt was never cancelled. GREEN after the supervision fix:
   returns `CHECKPOINT_STALE`, code 74, both arms terminated, attempt
   cancelled.

4. V3 resume exact replay at the apply boundary. New
   `tests/test_aws_checkpoint_mirror.py::test_v3_resume_apply_exact_replay_is_idempotent_and_version_pinned`
   publishes a real pair, seeds paired schema-2 submit state, and drives
   `resume(apply=True)` through the injected runner/approval/state boundary.
   First apply succeeds (approval and state bind the receipt URI/hash/
   version and both object versions; one SSM send). Repeating the identical
   triple is idempotent (twice; no further put-object/send-command).
   Reusing the same receipt hash under another receipt version or another
   object version raises `CHECKPOINT_PROVENANCE_MISMATCH` before any state
   write, archival, or SSM call (exactly two EC2 validation reads observed).
   RED observed: the drift case failed `STATE_CORRUPT` instead, because the
   idempotent-replay branch refreshed run states with bare `write_run`,
   leaving the durable pair journal stale so every later resume failed
   closed on journal divergence. Fix: both idempotent resume branches now
   persist through `_write_paired_states`, keeping the journal bound to the
   refreshed states. GREEN: `1 passed`, including a third identical apply.

5. Content-addressed checkpoint object metadata. New
   `test_identical_checkpoint_bytes_recover_by_exact_head_without_progress`
   publishes, then re-publishes identical checkpoint bytes under a new
   request ID at the same optimizer step against a store that models S3
   `If-None-Match: *`. RED observed:
   `ValueError: versioned object HEAD verification failed` because the
   stored object carried the first attempt's `request-id` metadata. Fix:
   `request-id` was removed from checkpoint object metadata in the producer
   (`_checkpoint_metadata`) and from the controller HEAD expectation
   (`AwsP5Backend._v3_checkpoint_metadata`); `request_id` remains in the
   paired receipt body and receipt object metadata. GREEN: the second
   request recovers both objects by exact HEAD (same version IDs, no new
   checkpoint uploads, no step progress) and publishes its own receipt.

6. Cancel on every supervisor exit path. New parametrized
   `test_supervisor_cancels_active_mirror_attempt_on_every_exit_path`
   covers requested shutdown, clean dual-arm exit, peer failure,
   notice-poll failure, and legacy interruption handling. RED observed:
   all five paths returned without cancelling the recorded active attempt.
   Fix: supervision now owns an unconditional `finally` that cancels any
   active attempt, so the attempt cannot continue publishing after the
   supervisor returns. GREEN: `5 passed`.

7. Notice-loop stale handling. New parametrized
   `test_notice_loop_stops_immediately_on_stale_and_falls_back` raises
   `CheckpointStaleError` from the immediate interruption attempt. RED
   observed: the loop kept re-raising (`4` stale raises before the bounded
   fixture stopped it), burning the notice budget in a spin. Fix: the loop
   breaks on the first `CheckpointStaleError`, cancels the attempt, and
   falls back only to the last complete receipt (75 with the receipt URI
   when one exists, otherwise 74, non-resumable). GREEN: `2 passed`,
   exactly one stale raise.

### Final verification on the combined tree

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/ms3c-fix-final-a \
  tests/test_aws_checkpoint_mirror.py tests/test_aws_p5_launcher.py
```

Result: **210 passed**.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/ms3c-fix-final-b tests/test_msctl.py
```

Result: **157 passed**.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/ms3c-fix-final-c \
  tests/test_aws_argv.py tests/test_aws_contract_roundtrip.py \
  tests/test_run_manifest_v3.py tests/test_package_aws_p5_handoff.py
```

Result: **217 passed**.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/ms3c-fix-final-d \
  tests/test_trainer.py tests/test_ddp_trainer.py
```

Result: **90 passed**.

Non-overlapping required-group aggregate: **674 passed**.

Static verification:

```bash
python -m py_compile \
  cluster/aws/p5/checkpoint_mirror.py cluster/aws/p5/launch_seed_pair.py \
  msctl/aws_p5.py msctl/aws_resume_launch.py \
  tests/test_aws_checkpoint_mirror.py tests/test_aws_p5_launcher.py

git diff --check
```

Result: both passed. `git status` confirmed only Task 3C scope files
changed; `msctl/operations.py`, evaluation, finalization, collection,
cleanup, sequential-transition, IaC, runbook, verifier, v2 scientific
files, and `corpusgen/` remain untouched.

### Files in this commit

- `cluster/aws/p5/checkpoint_mirror.py` (fork-pipe drain; content-addressed
  checkpoint object metadata)
- `msctl/aws_p5.py` (controller HEAD metadata expectation; idempotent-replay
  pair-journal persistence)
- `tests/test_aws_checkpoint_mirror.py` (findings 2, 4, 5 proofs)
- `.superpowers/sdd/task-3c-report.md` (this appendix)

Findings 1, 3, 6, and 7 production and test hunks from this session were
carried into `9e76a88` as described in the concurrency note.

### Self-review

- The drained pipe preserves the exact pickled result; partial reads across
  polls accumulate until EOF, and cancellation still SIGKILLs, reaps, and
  closes the descriptor with no publish after cancel.
- Checkpoint object metadata is now a pure function of checkpoint identity;
  the paired receipt keeps `request_id`, and receipt objects remain
  content-addressed per request, so provenance is not weakened.
- The pair journal now always matches the run states after idempotent
  replay, so exact replay is repeatable indefinitely and version drift is
  still refused before mutation.
- Supervisor cancellation is idempotent (`cancel_active` no-ops without an
  attempt) and the `finally` also covers `KeyboardInterrupt`.

### Concerns

- Two remediation sessions edited this worktree concurrently. The combined
  result was reviewed hunk-by-hunk and verified as one tree, but the
  interleaving means `9e76a88`'s message and appendix describe some hunks
  authored here (and vice versa). No work was lost; attribution is blurred.
- The legacy submit idempotent-refresh branch still updates run states with
  bare `write_run` (the same journal-divergence shape fixed for resume).
  No binding finding covers submit, so it was left unchanged; a submit
  status refresh followed by a resume would still fail closed rather than
  corrupt state.
- Sandboxed reruns of the launcher aggregate failed on fixture `git init`
  (`.git/hooks: Operation not permitted`); unsandboxed reruns passed. Not a
  product failure.
- No live paid AWS, versioned S3, P5, NCCL, Docker, or IMDS operation was
  performed; boundaries remain covered by injected stores and runners.

## Signal-acknowledged request-token remediation from `ed6c8d8`

Status: complete for the remaining baseline-to-signal race.

Implementation commit:

- `9b7d52a` — `fix: require signal-acknowledged checkpoint tokens`

No receipt field, receipt timing, checkpoint payload, scientific config,
optimizer schedule, or legacy write path changed.

### Root cause and fix

Descriptor-pinned baselines proved that metadata and checkpoint bytes belonged
to one generation, but did not prove that a later generation acknowledged the
active SIGUSR1. Both baselines were captured before either signal, so a normal
checkpoint published in that interval could satisfy the old generation-only
test.

The production path now uses the existing 32-lowercase-hex `request_id` as a
one-use checkpoint request token:

1. The reviewed launcher binds a distinct host token path to each arm's
   checkpoint sibling and passes the exact container path and arm through
   fixed Docker environment entries. The production request carries those
   paths plus the attested runtime UID/GID; resume uses the same scheduler.
2. Before either signal, the mirror exclusively and durably publishes strict
   canonical per-arm token files. Existing files are never overwritten.
3. The signal handler remains a generation-counter increment only. At the
   safe service boundary, rank zero atomically claims and validates the token
   only when servicing a SIGUSR1 generation, then passes its request ID to
   `save_ckpt`.
4. Adjacent trainer metadata always includes `request_token` (`null` for
   periodic/final/local saves). The reader accepts old metadata without the
   field as read-only compatibility and maps it to `None`.
5. Production staging waits past unrelated generations and accepts only
   metadata whose token equals the active request ID. Missing, stale,
   cross-request, and unacknowledged metadata therefore time out without a
   pair receipt.
6. Token publication/claim/cleanup uses no-follow directory walks, canonical
   payloads, exact owner/mode/link checks, exclusive publication, identity-
   checked quarantine/restore, and directory fsync. Failed signals, staging
   failures, timeout, and fork cancellation remove only the exact request's
   pending files.

### RED evidence

All production slices began with focused failing tests:

```text
Secure token format/publication/consume tests:
  6 failed — checkpoint token APIs did not exist.

Trainer capability/metadata/service tests:
  5 failed — capability and metadata lacked the token, missing tokens did
  not fail closed, and SIGUSR1 saves did not consume/bind a token.

Launcher propagation:
  request.runtime_uid was absent; the token capability was rejected as
  unknown evidence instead of being enforced.

Mirror race/cleanup:
  unrelated gap generation: dense acknowledgment hook was never reached;
  cross-request metadata: DID NOT RAISE TimeoutError;
  failed signal: both observations saw no published token files.
```

The deterministic gap regression installs valid Dense metadata after
baselines but immediately before the Dense signal callback. It then installs
the acknowledged generation only from the staging wait hook. The old code
published step 2 without reaching that hook; the fixed code waits and
publishes acknowledged Dense step 4 with Split90 step 3.

### GREEN evidence

Focused cycles:

```text
Secure token contract and adversarial file checks:      6 passed
Trainer token capability/metadata/service checks:       6 passed
Launcher path/environment/capability checks:            3 passed
Mirror gap/cross-request/failed-signal checks:           3 passed
Mirror compatibility and cancellation focus:            5 passed
Final combined remediation selection:                   14 passed
Resume periodic scheduler plus real token paths:         2 passed
```

### Final disk-bounded verification

Every pytest command used `PYTHONDONTWRITEBYTECODE=1`,
`-p no:cacheprovider`, and a dedicated `/tmp/ms3c-token-*` basetemp:

```text
tests/test_aws_checkpoint_mirror.py                         69 passed
tests/test_aws_p5_launcher.py                               147 passed
tests/test_trainer.py tests/test_ddp_trainer.py              99 passed
tests/test_msctl.py                                         157 passed
tests/test_run_manifest_v3.py tests/test_package_aws_p5_handoff.py
                                                             172 passed
tests/test_aws_argv.py                                       19 passed
```

Non-overlapping required-group aggregate: **663 passed**.

Changed-file `python -m py_compile`, staged and unstaged `git diff --check`,
and the final 14-test focused rerun passed.

### Token-remediation concerns

- No live paid AWS, versioned S3 bucket, P5, Docker, NCCL, or IMDS operation
  was performed. Filesystem, process, object-store, launcher, resume, and
  package boundaries are covered by deterministic injected tests.
