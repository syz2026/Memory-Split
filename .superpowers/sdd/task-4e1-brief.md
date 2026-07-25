# Task 4E1: Staged 100-slot evaluation executor

## Base and goal

Start exactly from canonical commit `1fc65e5` on
`feat/task-4e1-staged-executor`. Do not use the archived Task 4D recovery
line.

Implement the local on-instance executor that re-proves the frozen 100-slot
plan, safely materializes all runner inputs, invokes the unchanged v3 runner
once per slot in frozen order, and authenticates the resulting output
directories.

Use strict TDD. Keep selected `evaluate` and `cleanup` blocked.

## Scope

Create:

- `cluster/aws/p5/evaluate_snapshots.py`
- `tests/test_aws_evaluate_snapshots.py`

Modify only:

- `scripts/package_aws_p5_handoff.py`
- `tests/test_package_aws_p5_handoff.py`
- fixture-only additions in `tests/study_lock_fixtures.py` or
  `tests/cohort_output_fixtures.py`
- `.superpowers/sdd/task-4e1-report.md`
- `.superpowers/sdd/progress.md`

Do not modify production code under:

- `evals/confirmatory/`
- `msctl/`
- `train/`
- `corpusgen/`
- `configs/`
- existing `cluster/aws/p5` modules
- IaC, runbooks, or verifiers

No AWS, S3, network, Docker, state, approvals, remote intents, report
aggregation/publication, cleanup, or live CUDA.

## Staged input layout

`staged_input_root` must contain exactly:

```text
study-lock.json
evaluator-profile.json
evaluator-runtime-lock.json
evaluator-environment-receipt.json
seed-collections/seed-0.json ... seed-9.json
snapshots/seed-{0..9}/{dense|split90}/step{step:07d}.pt
```

The sealed release remains a separate content-addressed four-member
directory. Reject unexpected files, links, aliases, unsafe modes, and path
traversal.

## Public interface

```python
class EvaluationExecutionError(ValueError):
    code: str

@dataclass(frozen=True)
class SlotProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes

@dataclass(frozen=True)
class ExecutionPreview:
    study_lock_sha256: str
    slot_count: int
    first_output_id: str
    first_argv: tuple[str, ...]

@dataclass(frozen=True)
class SlotExecutionResult:
    slot_index: int
    seed: int
    arm: str
    optimizer_step: int
    output_id: str
    output_commitment: str
    output_dir: Path

@dataclass(frozen=True)
class CohortExecutionResult:
    study_lock_sha256: str
    outputs: tuple[SlotExecutionResult, ...]
    output_root: Path
    runs_root: Path
    outputs_root: Path
    path_authority: str = "informational_reopen_required"

def preflight_snapshot_evaluations(
    *,
    staged_input_root: Path,
    sealed_release_dir: Path,
    expected_study_lock_sha256: str,
    device: str = "cuda",
    provider_qualified: bool = True,
) -> ExecutionPreview: ...

def execute_snapshot_evaluations(
    *,
    staged_input_root: Path,
    sealed_release_dir: Path,
    expected_study_lock_sha256: str,
    output_root: Path,
    device: str = "cuda",
    provider_qualified: bool = True,
    slot_executor: Callable[
        [tuple[str, ...], Mapping[str, str]], SlotProcessResult
    ] | None = None,
) -> CohortExecutionResult: ...
```

All project imports must be function-local. Importing the module or
requesting CLI help must leave `torch`, `train.trainer`,
`evals.confirmatory`, and `msctl.*` unimported. Execute-time planning may
inherit Task 4A's transitive torch import; document it honestly.

## CLI

Support:

```text
python -m cluster.aws.p5.evaluate_snapshots
  --staged-input-root DIR
  --sealed-release DIR
  --expected-study-lock-sha256 SHA
  [--device {cpu,cuda,mps}]
  [--provider-qualified]
  [--execute --output-root DIR]
```

Requirements:

- dry-run by default;
- `--output-root` only with `--execute`;
- protected execute requires `--provider-qualified` and `--device cuda`;
- `add_help=False`, explicit `--help`, `allow_abbrev=False`;
- exactly one canonical JSON object on stdout for help, dry-run, success,
  and every error;
- exit 2 for argument or declared execution errors; exit 70 for unexpected
  internal errors.

## Frozen operation order

1. Validate arguments and descriptor-read `study-lock.json`.
2. Verify SHA-256, strict canonical bytes, and `StudyLockV3` parse.
3. Build ten `CollectionReceiptEvidence` values from the fixed seed files
   and lock-bound URI/SHA/version identities.
4. Call `plan_snapshot_evaluations`; require exactly 100 plans in
   `EXPECTED_STUDY_SLOTS_V3` order.
5. Verify the staged tree's exact layout. Descriptor-read and hash all 100
   source snapshots against their bindings before any destination write.
6. Call only `preflight_model_visible_release`. The executor must never open
   `sealed-gold.jsonl`; metadata inspection by the frozen preflight is
   permitted.
7. Descriptor-read the three evaluator files. Require their hashes to equal
   the lock selection. Parse generic P5/P6 profile and runtime lock and
   require profile/provider/runtime consistency. The frozen runner fully
   authenticates the environment receipt per slot.
8. Reject an existing/symlinked `output_root`; verify its parent is safe.
9. Create a private sibling staging tree containing `runs/` and `outputs/`,
   all mode 0700.
10. Materialize all 100 run directories before the first invocation. Each
    has exactly six files:
    - canonical `run.json`;
    - copied `snapshots/step{step:07d}.pt`;
    - `study-lock.json`;
    - the three evaluator authority files.

    Use exclusive no-follow writes, never links, immutable owner-readable
    modes, rehash every copy, and revalidate source descriptors.
11. Invoke slots 0-99 sequentially, once each, without retries or
    reordering. Stop at the first failure.

Default child argv:

```text
sys.executable -m evals.confirmatory evaluate
  --run RUN_DIR
  --sealed-release RELEASE_DIR
  --expected-study-lock-sha256 SHA
  --device DEVICE
  --output-dir OUTPUT_DIR
  [--provider-qualified]
```

Task 4E3 supplies Docker/network/IAM containment; 4E1 must not embed it.

12. Remove every `AWS_*` credential variable and `MSCTL_APPROVAL_KEY` from
    the child environment.
13. Require return code zero and one bounded canonical stdout JSON object
    with exact v3 success fields:
    - `status == "published"`;
    - exact seed, condition/arm, step, snapshot, lock, provider, and output
      ID;
    - exact expected output path;
    - `production_qualified` equals requested mode;
    - `report_sha256 == authoritative_commitment`;
    - `path_authority == "informational_reopen_required"`.
14. Reopen each output directory. Require exactly
    `EVALUATION_OUTPUT_MEMBERS`, safe owned singly-linked files, canonical
    `output.json`, exact plan identity, exact nine artifact rows, and matching
    hashes and byte counts. Reading/hashing copied gold is allowed only here,
    after the runner returns.
15. After all outputs pass, replay staging membership and commitments, then
    atomically rename staging no-replace to `output_root`.
16. On any post-staging failure, atomically rename the complete staging tree
    to an unpredictable `.evaluation-quarantine-*` sibling. Never
    pathname-unlink or recursively delete runner outputs or quarantine
    evidence.
17. Return commitments and informational paths. Do not create an evaluation
    manifest.

## Failure codes

- `EVALUATION_INPUT_INVALID`
- `EVALUATION_DESTINATION_EXISTS`
- `EVALUATION_RUN_MATERIALIZATION_FAILED`
- `EVALUATION_SLOT_FAILED`
- `EVALUATION_OUTPUT_INVALID`
- `EVALUATION_PUBLICATION_FAILED`

## Required RED coverage

Observe RED before implementation for each slice:

1. Missing module and CLI.
2. Normal import/help leave torch, trainer, evals, and msctl unloaded.
3. Lock hash, duplicate-key, noncanonical-byte, and schema failures make zero
   writes.
4. Nine/eleven/reordered/drifted collection receipts fail through real
   planning authority.
5. Staged-tree missing/extra/reordered paths, links, aliases, unsafe modes,
   and all snapshot hash mutations.
6. Profile/runtime/environment hash mismatches and crossed
   profile/provider/runtime identities, including generic P5/P6 handling.
7. Audit-hook proof that 4E1 never opens sealed gold; sealed-release
   commitment drift fails before writes.
8. Exact six-file materialization, byte-identical `run.json`,
   copied-not-linked snapshots, copy rehash, and all runs materialized before
   invocation.
9. Exactly 100 sequential calls; failure at slot k makes exactly k+1 calls
   and no retry.
10. Exact argv and credential-scrubbed environment.
11. Nonzero exit, timeout, oversized/multiple/noncanonical stdout,
    status/identity/path/commitment drift, and malformed output
    membership/artifact bindings.
12. Full 100-output payload-real fixture executor followed by
    `aggregate_cohort_outputs`, proving the handoff. A coherent dishonest
    output may pass transport checks but must fail only at frozen Task 4C.
13. Destination collision, parent replacement, and final-rename failure
    preserve quarantine; patched `unlink`, `rmdir`, and `shutil.rmtree`
    must never be called.
14. One-object CLI help/dry-run/execute and argument matrix; dry-run writes
    and invokes nothing and reports only first argv.
15. Package closure and committed-tree archive include the new module.

Do not fabricate a provider-qualified real-runner fixture. The frozen runner
forbids injected adapters in that mode. Use its nonqualified test seam for
direct compatibility and payload-real provider-qualified fixture outputs for
the full aggregation proof.

## Verification

Focused:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task4e1 \
  tests/test_aws_evaluate_snapshots.py \
  tests/test_confirmatory_snapshot_planning.py \
  tests/test_confirmatory_v3_runner.py \
  tests/test_confirmatory_v3_aggregate.py \
  tests/test_confirmatory_v3_lock_builder.py \
  tests/test_package_aws_p5_handoff.py
```

Then run:

- confirmatory v2/v3 regression group;
- lifecycle/hardware authority group unsandboxed;
- DDP provenance group;
- full package-handoff suite;
- full suite, allowing only the two inherited base-reproducible v2 verifier
  failures;
- changed-file `py_compile`;
- `git diff --check`;
- ancestry proving `1fc65e5` is the exact base;
- zero diff over every excluded production path;
- direct/module CLI help matrix;
- subprocess import-boundary checks;
- committed-tree package dry-run;
- frozen-science zero diff.

Self-review, commit implementation, write
`.superpowers/sdd/task-4e1-report.md`, and update progress.

## Handoff

Task 4E2 owns exact-version S3 staging, the thin Task 4C report CLI, immutable
publication of exactly 1,002 post-evaluation objects, and the canonical Task
4D schema-1 evidence index. `collect-cohort` remains a separate operator
action.

Task 4E3 owns selected evaluator routing, state, approval/deadline/lease,
remote intent, containment, and asynchronous recovery. Real apply remains
fail-closed pending independent evaluator qualification and resolution of the
frozen eight-GPU versus planned single-GPU tension.

Task 4F owns eleven-receipt exact-HEAD cleanup gating and safe termination.
