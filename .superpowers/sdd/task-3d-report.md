# Task 3D Report: Provider-aware clean-run finalization

## Status

`DONE_WITH_CONCERNS` — the initial blocked dispatch below is retained as
coordination history and is superseded by the completed implementation record
appended at the end of this report. Task 3D was implemented only after the
provider lifecycle bridge committed at `bec4c79`.

## Dispatch parameters observed

- Worktree: `/Users/stephenzhang/Documents/MemorySplit/.worktrees/provider-aware-lifecycle-integration`
- HEAD at dispatch and at all samples: `5c0b4b691efb9b84f7856e6f17fc57e1b0240912`
  (matches the required base `5c0b4b6`; never advanced during this session).
- Binding brief read in full, including the final profile-qualification
  authority amendment:
  `.superpowers/sdd/task-3d-provider-brief.md`,
  SHA-256 `207c141d9a77faddb5481cb69a966eeeba6cf8e504fcd441dcc1606ee4ee420e`,
  mtime `2026-07-24 04:36:55` (rewritten less than one minute before this
  session began; see conflict analysis for why it is expected to change again).

## Evidence of live concurrent mutation

All times are local (CDT), from three passive samples. This session performed
zero writes to the worktree before this report file.

Sample 1 — 04:39:08:

- dirty: `msctl/contracts.py`, `msctl/operations.py`,
  `tests/test_run_manifest_v3.py`; untracked: `msctl/aws_lifecycle.py`,
  `tests/provider_lifecycle_fixtures.py`,
  `tests/test_provider_lifecycle_bridge.py`.
- `tests/test_run_manifest_v3.py` mtime `04:38:44` — written after this
  session started (~04:37).

Sample 2 — 04:41:35 (after a 75 s passive wait):

- dirty set grew: `scripts/package_aws_p5_handoff.py` (mtime `04:41:31`,
  4 seconds before the sample) and `tests/test_package_aws_p5_handoff.py`
  newly modified.
- working-tree diff SHA-256:
  `717e87a1b4a0d66dc7e5d5fc3ac7072e9d250b5030b7bd618981f4ce0cbd83db`.

Sample 3 — 04:42:47:

- `scripts/package_aws_p5_handoff.py` mtime advanced again to `04:42:13`;
  `tests/test_package_aws_p5_handoff.py` mtime `04:42:32` — 15 seconds before
  the sample.
- working-tree diff SHA-256 changed to
  `8741301fe3f4480d7b2d6f9c203ca5f0c73c85b65849a78e9d7d0fabf0d23d4c`.

The mutating work is identified by the committed-in-worktree plan
`.superpowers/sdd/provider-lifecycle-bridge-plan.md` ("Selected-Provider
Lifecycle Bridge", executing inline in this worktree, explicitly scoped
"without implementing run finalization"). The observed file set matches its
Tasks 1–2 (lifecycle authority module; manifest/release metadata), and its
progression matches the live mtimes. No `provider-lifecycle-bridge-report.md`
exists yet, so that plan is incomplete and still running.

## Conflict analysis

Direct overlap between the bridge plan's remaining tasks and Task 3D:

- `msctl/contracts.py`: bridge Tasks 2 and 4 modify it (currently dirty with
  foreign hunks: `RunManifestV3` gains provider/selection lifecycle fields);
  Task 3D lists it as an allowed-modify file. Any `git add` of this file for a
  Task 3D commit would sweep in half-finished foreign hunks.
- `cluster/aws/p5/launch_seed_pair.py`: bridge Tasks 3, 4, and 5 modify it;
  Task 3D Step 4 (launcher admission/result plumbing) modifies it. The brief's
  amendment premise "The current launcher has no provider-selection admission"
  is about to be falsified by bridge Task 3, which adds authenticated
  selected launch-plan constructors.
- `scripts/package_aws_p5_handoff.py` and
  `tests/test_package_aws_p5_handoff.py`: bridge Tasks 2 and 6 modify them
  (observed changing 15 seconds before Sample 3); Task 3D Step 5 (package
  allowlist closure/round trip) modifies both.
- Task 3D bounded verification groups include
  `tests/test_aws_p5_launcher.py`, `tests/test_trainer.py`,
  `tests/test_aws_checkpoint_mirror.py`, `tests/test_aws_paired_state.py`,
  and `tests/test_package_aws_p5_handoff.py`; bridge Tasks 3–6 modify the
  launcher, checkpoint, trainer, and package tests and their production
  modules, so any RED/GREEN or regression result recorded now is not
  reproducible against a stable tree.
- The binding brief itself: bridge Task 6 explicitly plans to "Update the
  ignored Task3D brief to consume implemented names only". The brief bytes
  hashed above are therefore not final; implementing against them risks
  building to superseded frozen names (e.g. launcher admission shape,
  `SelectionIdentity` derivation, package member list).
- Base identity: bridge Task 6 commits its own work, so the Task 3D
  requirement of exact base/range checks against `5c0b4b6` cannot both hold
  now and after the bridge lands; the dispatch raced the bridge.

## What was verified before blocking

All authority symbols the brief and its amendment require do exist at the
committed base `5c0b4b6`, so Task 3D is implementable once the tree is stable:

- `cluster/aws/qualification.py`: `QualificationTimeReader`,
  `CohortSelectionAuthority`, `admit_cohort_provider_selection`,
  `_load_runtime_qualification_bundle`, `_build_selected_canary_plan`,
  `_parse_selected_qualification_receipt_bytes`.
- `msctl/aws_hardware.py`: `AuthenticatedSelectionBinding`,
  `VersionedProviderSelectionStore`, `QualificationApprovalVerifier`,
  `admit_provider_selection`.
- `cluster/aws/p5/launch_seed_pair.py`: `LaunchPlan`, `SupervisionResult`;
  `cluster/aws/p5/checkpoint_mirror.py`: `VersionedUploadedObject`,
  `CheckpointReceiptRef`.
- `msctl/aws_contracts.py`: `SEEDS`, `ARMS`, `SNAPSHOT_STEPS`
  (`1358, 3396, 6791, 10187, 13582`), `validate_sha256`.
- `msctl/objective_controls_v3.py`: `load_objective_controls_contract`,
  `OBJECTIVE_CONTROLS_CONTRACT_SHA256`.

## RED/GREEN commands, verification, files, commits, authority ordering

None were run and none were created, deliberately: under concurrent writes to
the same production and test files, RED observations are not attributable to
missing behavior, GREEN observations are not attributable to Task 3D changes,
`git diff --check`/exact-scope checks are unstable between invocation and
commit, and any commit of shared files would contain foreign hunks. Recording
such evidence would have violated the brief's freeze and evidence guarantees
rather than satisfied them.

The frozen implementation order (qualified public re-admission API → pure S3
key helpers → finalization publisher/parser → launcher admission/result
plumbing → package closure/round trip) remains correct and was not started.

## Self-review

- No Task 3D scope file, excluded file, or authority contract was touched;
  the only write this session is this ignored report file
  (`.superpowers/` is excluded via `.git/info/exclude`, so it cannot leak
  into the concurrent agent's commits).
- Blocking evidence is three independent timestamped samples with differing
  working-tree diff hashes, not a single observation.
- The alternative of waiting for the bridge to finish was considered and
  rejected: the post-bridge tree invalidates two dispatch parameters (base
  HEAD `5c0b4b6` and the current brief bytes), so a fresh dispatch against
  the post-bridge base and re-frozen brief is required rather than a stale
  in-session continuation.

## Unblocking conditions

Re-dispatch Task 3D only after all of:

1. the provider lifecycle bridge plan completes, commits, and writes its
   report (or is explicitly abandoned and its uncommitted changes reverted);
2. the worktree is clean (`git status --short` empty) at a declared new base
   HEAD;
3. `.superpowers/sdd/task-3d-provider-brief.md` is re-frozen against that
   base (bridge Task 6 edits it) and its hash is quoted in the dispatch.

## Concerns

- Two agents were dispatched into one worktree with overlapping write scopes;
  these tasks are sequential by design (the bridge defers finalization to
  Task 3D and finalizes Task 3D's brief). Parallel dispatch here is a
  coordination failure, not a recoverable in-session condition.
- If the bridge agent commits `tests/test_package_aws_p5_handoff.py` or
  launcher changes that alter `LaunchPlan` construction, the Task 3D brief
  sections on `SelectionIdentity` and package membership will need
  re-freezing before implementation, as anticipated by bridge Task 6.

## Superseding completion record

### Stable base and implementation

Task 3D resumed from clean provider-aware base
`bec4c793d9afcf9611165f26b2d3b92cfd7614f6`.

Implementation commit:

- `e9e2c74` — `feat: finalize provider-aware clean paired runs`

The implementation adds:

- `cluster/aws/p5/run_finalization.py`, with three independent
  `admit_provider_lifecycle(...)` gates before evidence admission, before the
  first upload, and before final receipt publication;
- strict provider-selected finalization receipt parsing and versioned
  snapshot/log/receipt publication;
- pure snapshot/log/run-receipt key helpers;
- clean-pair launcher finalization and success-result references;
- closed package membership and focused finalization tests.

### Interrupted hardening recovery

An interrupted follow-up left additional tests but no running worker. Those
tests were preserved and observed RED against `e9e2c74`:

1. evidence-object metadata omitted authenticated lifecycle fields
   (`KeyError: account-id`);
2. a group-readable snapshot mode was accepted
   (`DID NOT RAISE`);
3. a nonnumeric log loss was accepted
   (`DID NOT RAISE`);
4. a selected-provider clean pair without a finalizer returned `completed`
   instead of `FINALIZATION_FAILED`.

The recovered fixes:

- project every `LIFECYCLE_BINDING_FIELDS` value into snapshot/log S3 metadata;
- require runtime UID, runtime GID, and mode `0600` on local snapshots/logs;
- validate integer token geometry and every finite numeric log field;
- require a finalizer for selected lifecycle plans while preserving legacy
  finalizer-free completion.

Focused GREEN:

```text
tests/test_aws_run_finalization.py
78 passed
```

### Final verification

Non-overlapping bounded groups on the exact recovered code:

```text
tests/test_aws_run_finalization.py
tests/test_aws_p5_launcher.py
tests/test_aws_hardware.py
tests/test_objective_controls_v3.py
tests/test_aws_p6_qualification.py
tests/test_aws_contracts.py
=> 529 passed

tests/test_package_aws_p5_handoff.py
tests/test_trainer.py
tests/test_ddp_trainer.py
tests/test_aws_checkpoint_mirror.py
tests/test_aws_paired_state.py
tests/test_checkpoint_mirror_attempt_cleanup.py
=> 351 passed, 12 pre-existing macOS fork deprecation warnings
```

Total: `880 passed`.

Changed-file `python -m py_compile` and `git diff --check` passed.

### Scope and self-review

- Task 3C checkpoint receipt canonical bytes and timing remain unchanged.
- Selected P5/P6 finalization binds one complete
  `ProviderLifecycleBinding`; no provider is inferred from the legacy module
  path.
- Every evidence object is content-addressed, no-replace, exact-version
  verified, and metadata-bound before the final receipt is published last.
- Finalization failure cannot report selected-provider completion.
- Evaluation, collection, cleanup, sequential transitions, IaC, runbooks,
  release verification, `msctl/operations.py`, and `corpusgen/` remain outside
  Task 3D.

### Remaining concerns

- No live AWS, paid P5/P6, versioned S3, Docker, GPU, NCCL, or cryptographic
  qualification operation was performed.
- The audited 120-second real checkpoint-mirror feasibility risk remains a
  required later qualification gate; Task 3D does not claim to close it.

## Review correction: literal verification roots

The task reviewer reproduced a Darwin-only fixture ownership failure when the
brief's literal `/tmp` basetemps were used. BSD group inheritance made pytest
fixture directories and their files GID `0` while the fixtures declared runtime
GID `20`; the production owner check correctly rejected them.

Strict RED:

```text
test_clean_pair_publishes_twelve_objects_and_one_verified_receipt[p5-seed0]
=> FinalizationError: snapshot owner or mode is not the runtime identity

literal Task 3D package group
=> 33 checkpoint mirror fixture failures at the token-parent GID check
```

The fixtures now explicitly set their generated evidence files and checkpoint
token parent directories to `os.getgid()`. Production UID/GID/mode enforcement
is unchanged.

Exact GREEN using the brief's literal basetemp roots:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3d-finalization \
  tests/test_aws_run_finalization.py tests/test_aws_p5_launcher.py \
  tests/test_aws_hardware.py tests/test_objective_controls_v3.py \
  tests/test_aws_p6_qualification.py
=> 493 passed

PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task3d-package \
  tests/test_package_aws_p5_handoff.py tests/test_trainer.py \
  tests/test_aws_checkpoint_mirror.py tests/test_aws_paired_state.py
=> 337 passed, 6 pre-existing macOS fork deprecation warnings
```

These post-implementation runs supersede the earlier verification wording for
the mandated command forms. The production finalization code is unchanged by
this portability correction.
