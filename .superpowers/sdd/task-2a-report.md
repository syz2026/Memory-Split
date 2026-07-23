# Task 2A report: canonical AWS v3 package and shared contracts

## Status

Implemented and verified the canonical AWS-only v3 handoff package on
`feat/memorysplit-v3-aws-n10`, starting from
`8c2eb002f0149370918d7586a0bc4124ad64cb03`.

The implementation commit is:

- `ef4971d2f2c5953aba63576411ab61d810a76897`
  (`feat: package canonical AWS v3 release`)
- This report is committed separately in the commit containing this file.

## TDD record

### RED: missing shared AWS contract

`tests/test_aws_contracts.py` was added before `msctl/aws_contracts.py`.
The first direct-import run produced the expected collection error:

```text
ModuleNotFoundError: No module named 'msctl.aws_contracts'
1 error in 0.42s
```

The test harness was then corrected, still before production code, so the
missing module appeared as ordinary assertion failures rather than a collection
error.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task2a-red-contracts \
  tests/test_aws_contracts.py
```

Expected RED output:

```text
FFFFFFFFFFFFFF                                                           [100%]
...
AssertionError: canonical AWS contract module is missing
14 failed in 0.09s
```

### GREEN: shared AWS contract

After the minimal standard-library-only module was added, the same focused
contract suite passed:

```text
..............                                                           [100%]
14 passed in 0.03s
```

### RED: active package still used v2 paths

The package fixture and assertions were converted to v3 before changing the
packager. The sandboxed run could not initialize its temporary Git repository,
so the command was rerun unrestricted as required by the brief.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task2a-red-package \
  tests/test_package_aws_p5_handoff.py::\
test_double_build_is_byte_identical_and_emits_external_receipts
```

Expected RED output:

```text
PackageError: unknown tracked path is not allowlisted:
configs/360m-v3/dense-s0.yaml
1 failed in 0.48s
```

This failure was caused by the old v2 package allowlist, not by test fixture or
syntax failure.

### GREEN: deterministic v3 package

After the packager consumed the shared v3 constants:

```text
.                                                                        [100%]
1 passed in 0.83s
```

The full packaging-hardening file then passed:

```text
........................................................................ [ 80%]
.................                                                        [100%]
89 passed in 37.13s
```

### RED/GREEN: closed source and dataset-pointer identities

Self-review identified two exact contract checks that needed explicit mutation
tests. Their production checks were removed, the tests were added, and RED was
observed:

```text
FF                                                                       [100%]
...
Failed: DID NOT RAISE ... PackageError
Failed: DID NOT RAISE ... PackageError
2 failed in 1.15s
```

The two mutations were a SHA-256-format Git repository, which would emit
64-character source IDs, and a dataset pointer changed back to
`dataset/corpus-receipt.json`. After restoring the minimal checks:

```text
..                                                                       [100%]
2 passed in 0.72s
```

### Required final GREEN

Command required by the Task 2A brief:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task2a-final \
  tests/test_aws_contracts.py \
  tests/test_package_aws_p5_handoff.py \
  tests/test_cohort_assignment_v3.py \
  tests/test_cohort_assignment_v2.py \
  tests/test_aws_p5_profile.py
```

Output:

```text
........................................................................ [ 33%]
........................................................................ [ 67%]
.....................................................................    [100%]
213 passed in 39.41s
```

## Changed files

- `msctl/aws_contracts.py`
  - exposes immutable provider, cohort, preregistration, path, seed, arm,
    snapshot, and package-format constants;
  - returns the exact ordered 20-config path tuple;
  - validates lowercase 64-character SHA-256 values;
  - derives the three content-addressed release keys and canonical dataset
    receipt key;
  - is pure and standard-library-only.
- `scripts/package_aws_p5_handoff.py`
  - consumes the shared v3 constants;
  - packages the AWS-only seeds `0..9` and all 20 v3 configs;
  - requires v3 assignment, preregistration, profile, and shared contract
    module;
  - validates exact Task 1 assignment, config, profile, dataset-pointer, and
    preregistration identities;
  - emits integer package format `2`;
  - preserves deterministic ZIP, Git-object snapshot, source-cleanliness,
    secret scan, no-replace publication, descriptor pinning, and atomic receipt
    behavior.
- `tests/test_aws_contracts.py`
  - covers exact constants, config paths, key derivation, malformed hashes,
    uppercase hashes, and path-like injection.
- `tests/test_package_aws_p5_handoff.py`
  - converts the release fixture and receipt assertions to v3;
  - retains all prior hardening tests;
  - adds exact closed-field, cross-version, wrong-pointer, wrong-profile,
    source-object, seed-10, Illumina, and v2-exclusion coverage.
- `.superpowers/sdd/task-2a-report.md`
  - records this TDD and verification evidence.

`DATASET-POINTER-AWS.json` was not modified; its existing
`required_receipt: dataset/receipt.json` is canonical.

No bootstrap, launcher, lifecycle, verifier, runbook, Illumina source,
existing v2 contract artifact, or `corpusgen/` file was modified.

## Shared-contract self-review

The shared module freezes:

- provider `aws-p5.48xlarge`;
- cohort `memorysplit-confirmatory-v3-360m-n10-aws`;
- preregistration ID/path `memorysplit-confirmatory-v3` and
  `configs/preregistration-v3.yaml`;
- assignment `configs/cohort-assignment-v3.json`;
- config root `configs/360m-v3`;
- profile `cluster/profiles/aws-p5.48xlarge-v3.json`;
- dataset pointer and receipt paths `DATASET-POINTER-AWS.json` and
  `dataset/receipt.json`;
- seeds `tuple(range(10))`, arms `("dense", "split90")`, snapshots
  `(1358, 3396, 6791, 10187, 13582)`, and integer package format `2`.

The exact config helper returns 20 unique paths in seed/arm order. Hash
validation rejects uppercase, short, long, non-hex, algorithm-prefixed,
whitespace, bytes, null, slash, parent-path, and path-suffixed values before
constructing:

- `releases/<sha256>/archive.zip`;
- `releases/<sha256>/release.json`;
- `releases/<sha256>/archive.sha256`.

## Closed allowlist self-review

The active package requires exactly the v3 assignment, v3 preregistration, v3
profile, shared contract module, and all 20 v3 config paths. Missing half-pairs
fail as missing required members; seed 10 and any other unenumerated v3 config
fail as unknown paths.

The classification review confirmed explicit exclusion of:

- `configs/360m-v2/*.yaml`;
- v2 assignment and preregistration;
- the v2 AWS profile;
- the Illumina profile, packager, packaging tests, pointer, and Slurm tree;
- materialized corpus/data, outputs, checkpoints, logs, sealed evaluation, and
  disposable cache paths.

Unknown roots, unenumerated sources/vendor objects, tracked symlinks,
secret-like fields/material, and forbidden nested path spellings remain
fail-closed. Existing descriptor, race, permissions, deterministic mode,
dry-run/apply, no-replace, and source-substitution tests all remain green.

## Receipt self-review

Tests assert closed top-level field sets for both `RELEASE-METADATA.json` and
external `RELEASE-AWS-P5.json`. Both bind:

- integer package format `2`;
- AWS provider and v3 cohort;
- all seeds `0..9` and both arms;
- exact assignment, profile, dataset-pointer, and 20 config hashes;
- member rows/commitment;
- one source object with exactly `commit`, `tree`, and `dirty`;
- lowercase 40-character commit/tree IDs and `dirty: false`;
- runtime attestation requirements without a fabricated AMI, image digest, or
  environment receipt.

The external receipt additionally binds exact archive path, SHA-256, and byte
count. Neither receipt contains `dataset_receipt_sha256`, `dataset_build_id`, or
`ordered_stream_sha256`; only the dataset-pointer hash is bound.

## V2 byte identity and patch checks

Command:

```bash
git diff --exit-code 8c2eb002f0149370918d7586a0bc4124ad64cb03 -- \
  configs/360m-v2 \
  configs/preregistration-v2.yaml \
  configs/cohort-assignment-v2.json \
  cluster/profiles/aws-p5.48xlarge.json \
  DATASET-POINTER-AWS.json
```

Output: none; exit code `0`.

`tests/test_cohort_assignment_v2.py` and the existing-profile tests are also
included in the 213-test final GREEN.

Patch check:

```bash
git diff --check
```

Output: none; exit code `0`.

## Actual-tree package smoke test

After the implementation commit, the clean real worktree was packaged in
dry-run mode:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/package_aws_p5_handoff.py \
  --source-root . \
  --out-dir /tmp/memorysplit-v3-task2a-package-smoke
```

The command exited `0` with one JSON result reporting `ok: true`,
`dry_run: true`, `published: false`, provider `aws-p5.48xlarge`, and release ID
`aws-p5-r1-4dada6f1fe678679`.

## Concerns

No Task 2A defect remains. As explicitly scoped by the brief, the bootstrap,
paired launcher, lifecycle consumers, verifier, and runbook still encode the
prior runtime contract and must be aligned in Task 2B before this package is
used for an operational v3 launch.

The repository-wide suite was not used as the acceptance gate because the brief
identifies two pre-existing, out-of-scope failures in
`tests/test_verify_cohort_releases.py`; the required focused v3/v2 and packaging
suite is green.

## Review follow-up: closed dataset pointer

Review identified that `_validate_dataset_pointer` checked only four required
values and did not reject additional fields. Consequently, a tracked pointer
could add `dataset_receipt_sha256`, `dataset_build_id`,
`ordered_stream_sha256`, or any unknown field and still be archived and
hash-bound by the code release.

### RED

The mutation test was added before the production change:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task2a-review-red \
  tests/test_package_aws_p5_handoff.py::\
test_packager_rejects_extra_dataset_pointer_fields
```

Output:

```text
FFFF                                                                     [100%]
...
Failed: DID NOT RAISE ... PackageError
4 failed in 2.04s
```

All four mutations reached package construction without rejection, confirming
the review finding.

### GREEN

The validator now compares the parsed pointer against one exact typed object
containing every canonical field and value from `DATASET-POINTER-AWS.json`.
This closes the field allowlist while retaining the canonical pointer unchanged.

Focused command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task2a-review-green \
  tests/test_package_aws_p5_handoff.py::\
test_packager_rejects_extra_dataset_pointer_fields \
  tests/test_package_aws_p5_handoff.py::\
test_packager_rejects_wrong_dataset_receipt_pointer
```

Output:

```text
.....                                                                    [100%]
5 passed in 2.19s
```

### Requested regression suite

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task2a-review-final \
  tests/test_package_aws_p5_handoff.py \
  tests/test_aws_contracts.py
```

Output:

```text
........................................................................ [ 66%]
.....................................                                    [100%]
109 passed in 42.19s
```

`git diff --check` produced no output and exited `0`.
