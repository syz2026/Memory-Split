# Task 2B2 report: canonical AWS v3 run manifest and dataset identity

## Status

Implemented, self-reviewed, committed, and verified Task 2B2 on
`feat/memorysplit-v3-aws-n10`, starting from
`9c6e8ecc12fc2831c9e37072d74167bc4158fd84`.

The implementation commit is:

- `0caf23bca3e04199cef573f77a0d0918f4cad939`
  (`feat: add canonical AWS v3 run manifests`)
- This report is committed separately in the commit containing this file.

The generic run-manifest path now has an isolated schema-version-3 AWS
contract. It instantiates a real Task 2A release against a canonical Task 4
dataset publication, carries the exact receipt/build/ordered dataset
identities, and binds all release-owned identities. Valid schema-version-2
manifests continue to load and bind through their explicit legacy contract.

## TDD record

### Baseline

The worktree was confirmed on `feat/memorysplit-v3-aws-n10` at the required
Task 2B1 base. The existing required files passed before the new test file was
introduced:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task2b2-baseline-unrestricted \
  tests/test_aws_contract_roundtrip.py \
  tests/test_cohort_assignment_v3.py \
  tests/test_package_aws_p5_handoff.py \
  tests/test_msctl.py
```

```text
297 passed in 101.26s (0:01:41)
```

The first sandboxed attempt could not create Git hook directories in the
temporary repositories. The baseline was rerun unrestricted as permitted by
the brief. A stale read-only temporary release also prevented reuse of the
first base-temp directory, so subsequent runs used fresh directories.

### RED: schema 3, instantiation, and CLI

`tests/test_run_manifest_v3.py` was created before the production changes. Its
fixture creates a minimal real Git repository, copies the canonical v3
profile, assignment, preregistration, pointer, and all 20 v3 configs, invokes
the real Task 2A packager, and combines that release with the canonical
synthetic Task 4 dataset publication and verifier.

The initial focused tests reached the intended missing contracts:

- `instantiate_run_manifest()` rejected the prospective sealed-evaluation
  argument with `TypeError: unexpected keyword argument`;
- `load_run_manifest()` rejected schema version 3 as unsupported;
- the CLI attempted to load the v3 AWS profile through the legacy generic
  profile loader and returned `PROFILE_INVALID` before reaching the required
  sealed-evaluation argument boundary.

Production behavior was added only after those failures.

### Review RED: nonscalar closed-enum values

Exact-boundary self-review added malformed provider and arm cases before the
corresponding hardening. Both initially escaped the structured contract error
boundary as raw unhashable-list `TypeError` exceptions:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task2b2-enums-red \
  tests/test_run_manifest_v3.py::test_v3_loader_rejects_nonscalar_enum_fields
```

```text
2 failed in 1.72s
```

The loader now type-checks provider and arm before set membership. The enum and
numeric-alias boundary checks then passed:

```text
7 passed in 4.37s
```

Self-review also limited special v3 AWS profile loading to `runs instantiate`.
An integration test proves that another v3 CLI command still stops at
`PROFILE_INVALID`, leaving remote AWS lifecycle enablement to Task 3.

### Focused GREEN

The complete new test file passed after the core implementation:

```text
48 passed in 32.56s
```

The Task 2B1 real-package round trip and focused existing `msctl` tests also
passed:

```text
26 passed in 11.66s
16 passed, 141 deselected in 0.61s
```

### Required final GREEN

The final run used the exact five required files after all self-review changes:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task2b2-final-2 \
  tests/test_run_manifest_v3.py \
  tests/test_aws_contract_roundtrip.py \
  tests/test_cohort_assignment_v3.py \
  tests/test_package_aws_p5_handoff.py \
  tests/test_msctl.py
```

```text
........................................................................ [ 20%]
........................................................................ [ 41%]
........................................................................ [ 61%]
........................................................................ [ 82%]
...............................................................          [100%]
351 passed in 135.84s (0:02:15)
```

The following checks also produced no output and exited `0`:

```bash
git diff --check
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  msctl/contracts.py \
  msctl/operations.py \
  msctl/cli.py \
  tests/test_run_manifest_v3.py
```

The staged implementation diff also passed `git diff --cached --check`.

## Changed files

### `msctl/contracts.py`

- Adds immutable `RunManifestV3` without legacy `dataset_sha256` or
  `study_lock_sha256` properties.
- Parses manifest JSON with duplicate-key and non-finite-number rejection.
- Selects closed schema-specific root and run-row field sets.
- Requires schema 3's provider, cohort, integer seed 0..9, exact run IDs,
  exact v3 config paths, one Dense/Split90 pair, canonical SHA-256 values,
  lowercase Git commit/tree IDs, and finite positive GPU-hour estimates.
- Rejects boolean and integer-valued-float aliases at integer boundaries and
  rejects nonscalar enum values with `MsctlError`.
- Loads valid schema 2 into the existing `RunManifest` type without projecting
  schema-3 fields or compatibility properties.
- Binds schema 3 to package format 2, archive and external receipt hashes,
  profile, pointer, assignment, preregistration, source commit/tree, full v3
  assignment metadata, release member hashes, and config hash map.
- Explicitly rejects schema-2/schema-3 release cross-binding.

### `msctl/operations.py`

- Detects only profile `aws-p5.48xlarge-v3` with exact assigned integer seeds
  0..9 as the AWS v3 instantiation path.
- Requires and validates the prospectively supplied
  `sealed_evaluation_release_sha256`.
- Requires the Task 2A package format and verifies the local v3 profile,
  dataset pointer, assignment, preregistration, and selected pair of configs
  against authenticated release members.
- Uses the real v3 assignment/config paths and emits exact v3 run IDs.
- Independently hashes the selected dataset receipt, invokes the canonical
  Task 4 verifier, rechecks all pinned file identities, and obtains canonical
  `build_id` and `ordered_stream_sha256` values from verified evidence.
- Emits deterministic finite positive `estimated_gpu_hours: 96.0` for each
  four-GPU arm.
- Builds the exact schema-3 object and preserves dry-run-first, no-replace
  publication.
- Leaves the existing non-v3 manifest branch isolated.

### `msctl/cli.py`

- Adds `--sealed-evaluation-release-sha256`.
- Loads the strict v3 AWS profile only for `runs instantiate`.
- Requires the new option at the v3 CLI boundary and forwards it unchanged.
- Keeps non-instantiation v3 AWS commands behind the existing profile gate so
  this task does not enable state, remote AWS lifecycle, or evaluation paths.

### `tests/test_run_manifest_v3.py`

- Builds a real Task 2A release from canonical repository contract files.
- Uses the canonical Task 4 synthetic receipt and semantic verifier.
- Proves exact seed-0 and seed-9 manifests, closed fields, canonical hashing,
  load/reload, release binding, no-replace apply, and CLI forwarding.
- Covers receipt bytes, build ID, ordered stream, release receipt, profile,
  pointer, assignment, preregistration, source commit/tree, config, cohort,
  seed, pair, cross-version, malformed hash, missing/unknown field, duplicate
  JSON, non-finite number, numeric alias, and nonscalar enum mutations.
- Proves explicit valid schema-2 load/bind behavior and both cross-version
  rejection directions.

No state, checkpoint, AWS remote lifecycle, Slurm, packaging, bootstrap,
launcher, evaluation implementation, IaC, verifier, runbook, Illumina asset,
v2 scientific file, or `corpusgen/` file changed.

## Exact contract self-review

The schema-3 root has all and only these 17 fields:

1. `schema_version`
2. `provider`
3. `cohort_id`
4. `seed`
5. `release_sha256`
6. `release_receipt_sha256`
7. `profile_sha256`
8. `dataset_pointer_sha256`
9. `dataset_receipt_sha256`
10. `dataset_build_id`
11. `ordered_stream_sha256`
12. `cohort_assignment_sha256`
13. `preregistration_sha256`
14. `sealed_evaluation_release_sha256`
15. `source_commit`
16. `source_tree`
17. `runs`

Each of the exactly two run rows has all and only:

- `run_id`
- `arm`
- `seed`
- `config`
- `config_sha256`
- `estimated_gpu_hours`

The provider is exactly `aws-p5.48xlarge`; the cohort is exactly
`memorysplit-confirmatory-v3-360m-n10-aws`; seeds are exact integers 0..9; and
the pair is exactly one `dense` and one `split90` run for the root seed.
Boolean and float aliases cannot satisfy schema or seed integers. GPU hours
reject booleans, zero, negatives, and non-finite values and normalize valid
positive numeric values to the in-memory float field.

All ten SHA-256 identity fields require exactly 64 lowercase hexadecimal
characters. `source_commit` and `source_tree` require exactly 40 lowercase
hexadecimal characters. Unknown or missing root/run fields, duplicate JSON
fields at any depth, non-finite JSON constants, v2 run IDs/configs, partial
pairs, duplicate arms, seed 10, and wrong config bytes fail closed.

The code release deliberately binds only the dataset-pointer bytes. It does not
claim a concrete dataset receipt/build/ordered identity. Those identities are
introduced from the caller-selected, fully verified canonical Task 4
publication and then frozen into the run manifest, matching the Task 2B2
boundary.

Schema 2 remains a distinct `RunManifest` object with its historical
`dataset_sha256` and `study_lock_sha256`. Schema 3 is a distinct
`RunManifestV3` object and intentionally has neither field. A schema-2 AWS
manifest cannot bind package format 2, and a schema-3 manifest cannot bind a
legacy release.

## Concerns and deliberate gates

No known Task 2B2 defect remains.

The following are intentional scope gates, not completed lifecycle support:

- `RunManifestV3` has no `dataset_sha256` compatibility property. Old
  state/checkpoint/remote lifecycle code that assumes it must fail until Task
  3 migrates those consumers.
- The sealed evaluation release is caller-supplied and syntax-validated as an
  immutable SHA-256 identity. Loading or verifying that evaluation artifact is
  outside this task.
- The per-arm 96.0 GPU-hour value is deterministic budgeting metadata for this
  manifest. Remote allocation, approval, and accounting propagation remain
  outside this task.
- The v3 CLI profile adapter is intentionally limited to manifest
  instantiation; other v3 AWS commands remain unavailable until their
  contracts are migrated.

## Report path

`/Users/stephenzhang/Documents/MemorySplit/.worktrees/memorysplit-v3-aws-n10/.superpowers/sdd/task-2b2-report.md`
