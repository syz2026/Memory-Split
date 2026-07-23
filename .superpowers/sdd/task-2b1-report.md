# Task 2B1 report: align the AWS v3 package and launch consumers

## Status

Implemented, self-reviewed, and verified Task 2B1 on
`feat/memorysplit-v3-aws-n10`, starting from
`97c248689d3ffbb1be5616832a3172e3ada38550`.

The implementation commit is:

- `bfebf30f60924eb73eddb0ec8c654aa7c2421c82`
  (`feat: align AWS v3 launch contracts`)
- This report is committed separately in the commit containing this file.

The real Task 2A package now round-trips through the generic loader, bootstrap
verification and extraction, launcher-manifest construction, and seed-0 paired
launch planning. Both exact four-rank commands render without hardware or AWS
calls.

## TDD record

### Baseline

The focused pre-change suites passed before Task 2B1 tests were introduced.
The worktree was confirmed to be on `feat/memorysplit-v3-aws-n10` with merge
base `97c248689d3ffbb1be5616832a3172e3ada38550`.

### RED: real v3 package round trip

`tests/test_aws_contract_roundtrip.py` was created before production changes.
It materialized the Task 2A minimal clean Git repository and invoked the real
packager.

The first harness attempt passed an extra path component to the existing
`_minimal_repo()` helper and failed before reaching a product contract. That
test setup error was corrected before recording RED.

The valid contract RED was then run with:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task2b1-red \
  tests/test_aws_contract_roundtrip.py::\
test_real_v3_package_round_trips_to_exact_seed_zero_pair
```

The real package stopped in `load_release()` because the generic loader still
required the historical two-field source object:

```text
MsctlError: release.source has missing or unknown fields
```

This was the expected product RED: Task 2A emitted exact
`{commit, tree, dirty}` source identity, while the loader did not accept it.

### Staged REDs while advancing the same real package

As each earlier boundary was minimally aligned, the unchanged round-trip test
exposed the next stale consumer:

- bootstrap expected the nonexistent
  `release_value["dataset_receipt_sha256"]`;
- the launch-manifest builder rejected seed `0` as unassigned;
- paired launch validation expected the v2 cohort;
- paired launch validation expected v2 config paths, the old profile, the old
  logical corpus path, and `snap_frac`.

These failures showed that package, loader, bootstrap, builder, and launcher
were independently encoding incompatible contracts rather than being bypassed
with hand-shaped fixtures.

Existing focused tests were converted to v3 before the corresponding
production behavior. Mutation coverage was added for package/source fields,
dataset receipt identity, seed ownership, v2 config substitution,
`snap_frac`/snapshot schedules, partial pairs, and wrong config hashes.

### GREEN: focused real round trip and boundary checks

After the minimal consumer changes:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task2b1-focused-final \
  tests/test_aws_contract_roundtrip.py \
  tests/test_aws_p5_launcher.py::test_launcher_rejects_wrong_corpus_identity \
  tests/test_aws_p5_launcher.py::\
test_dry_run_renders_exact_symmetric_four_plus_four_commands \
  tests/test_msctl.py::\
test_load_release_keeps_legacy_source_shape_explicit_and_closed \
  tests/test_msctl.py::\
test_aws_post_bootstrap_builder_emits_exact_task3_launcher_manifest
```

Output:

```text
..................                                                       [100%]
18 passed in 2.43s
```

### Required final GREEN

The brief's `-k 'release or launch_manifest'` expression would filter every
listed file globally and therefore would not exercise the complete launcher,
packager, contract, assignment, and profile files. The full listed files were
run without `-k`:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task2b1-final \
  tests/test_aws_contract_roundtrip.py \
  tests/test_aws_p5_launcher.py \
  tests/test_package_aws_p5_handoff.py \
  tests/test_aws_contracts.py \
  tests/test_cohort_assignment_v3.py \
  tests/test_aws_p5_profile.py \
  tests/test_msctl.py
```

Output:

```text
........................................................................ [ 15%]
........................................................................ [ 30%]
........................................................................ [ 46%]
........................................................................ [ 61%]
........................................................................ [ 76%]
........................................................................ [ 92%]
....................................                                     [100%]
468 passed in 87.57s (0:01:27)
```

## Changed files

- `msctl/contracts.py`
  - extends `Release` with explicit optional `source_tree` and
    `package_format_version`;
  - keeps historical source shape explicit for legacy receipts;
  - validates package format `2`, exact source fields and types, runtime
    attestation, profile/assignment/pointer bindings, all 20 config hashes,
    exact member rows, and v3 assignment/pointer identities;
  - rejects duplicate and non-finite internal JSON values.
- `cluster/aws/p5/bootstrap.py`
  - accepts only the active v3 external and internal release contracts;
  - removes the invalid concrete dataset hash expectation from the code
    release;
  - independently hashes the supplied dataset receipt and cohort assignment;
  - validates and carries dataset `build_id` and `ordered_stream_sha256`;
  - binds the exact profile, assignment, pointer, source tree, config map,
    runtime attestation, and sorted archive members;
  - includes ordered-stream identity in the bootstrap receipt.
- `msctl/aws_launch_manifest.py`
  - uses shared v3 constants for seeds, cohort, provider, config root, and
    dataset receipt path;
  - emits canonical dataset receipt SHA-256, build ID, and ordered-stream
    SHA-256;
  - preserves no-replace publication.
- `cluster/aws/p5/launch_seed_pair.py`
  - admits exact integer seeds `0..9` and rejects negative, `10`, and boolean
    seeds;
  - requires the v3 profile, cohort, package metadata, bootstrap receipt,
    config paths, and logical corpus path;
  - validates dataset hash/build/ordered-stream identity across manifest,
    receipt, semantic corpus evidence, and bootstrap receipt;
  - replaces `snap_frac` with the exact integer snapshot schedule;
  - preserves paired ports, CPU halves, four-rank-per-arm commands, complete
    pair validation, scientific settings, output isolation, and preflight.
- `tests/test_aws_contract_roundtrip.py`
  - builds and applies a real Task 2A package;
  - exercises loader, bootstrap, extraction, manifest builder, and paired
    planner through exact seed-0 command rendering;
  - covers release-format/source/dataset injection, assignment, and dataset
    receipt mutations.
- `tests/test_aws_p5_launcher.py`
  - converts active launcher fixtures and expectations to v3;
  - exercises all ten seeds and exact port schedules;
  - covers v2 path substitution, malformed snapshots, wrong corpus
    hash/build/ordered identity, and real v3 bootstrap archives.
- `tests/test_msctl.py`
  - covers explicit legacy loader compatibility;
  - covers exact v3 post-bootstrap manifest construction and rejection without
    output publication.
- `.superpowers/sdd/task-2b1-report.md`
  - records RED/GREEN, final verification, scope, and self-review evidence.

No packager, Task 2A test, lifecycle/state/operation, evaluation,
infrastructure, diagnostics, verifier, runbook, Illumina, v2 scientific
artifact, or `corpusgen/` file was modified.

## Fail-closed self-review

The active bootstrap and paired-launch paths now require:

- provider `aws-p5.48xlarge`;
- cohort `memorysplit-confirmatory-v3-360m-n10-aws`;
- exact integer package format `2`;
- exact AWS seeds `0..9` and Dense/Split90 arms;
- exact profile `cluster/profiles/aws-p5.48xlarge-v3.json`;
- exact assignment `configs/cohort-assignment-v3.json`;
- all and only the 20 `configs/360m-v3` config bindings;
- exact source keys `commit`, `tree`, and `dirty`, with 40-character lowercase
  Git IDs and identity-typed `dirty: false`;
- exact Task 2A runtime-attestation fields;
- logical dataset receipt `dataset/receipt.json`.

External and internal release objects use closed field sets. Archive
verification rejects unsafe paths, duplicates, symlinks, non-regular members,
unsorted or incomplete checksums/member rows, hash drift, wrong Git modes,
format 1, v2 substitutions, seed 10, Illumina assignment, and binding
disagreement.

The code release still binds only the dataset pointer. It does not contain or
manufacture `dataset_receipt_sha256`, a dataset build ID, or an ordered-stream
hash. Runtime dataset identity enters only through the caller-hashed canonical
dataset receipt, is carried into bootstrap evidence, and is then required by
the generated launch manifest and paired-launch validator.

The strict config reader accepts the repository's exact inline snapshot list
and the exact unindented block serialization emitted by the Task 2A minimal
package fixture. Both forms produce only
`[1358, 3396, 6791, 10187, 13582]`. Booleans, floats, reordering,
duplicates, missing/extra values, aliases, tags, indented/nested lists, and
arbitrary YAML structures are rejected.

`msctl/contracts.py` retains v2 literals only inside the explicit
package-format-1 read-only compatibility branch. The changed active bootstrap,
launch-manifest builder, and paired-launch validator contain no v2 cohort,
config-root, profile, corpus-path, seed-list, or `snap_frac` launch literal.

`git diff --check` produced no output and exited `0` before the implementation
commit.

## Concerns

No known Task 2B1 defect remains.

Broader lifecycle/state/evaluation/IaC/verifier/runbook field alignment was
deliberately not changed. Those areas remain for the separately scoped Task
2B2 or later tasks and must not be treated as v3-operational solely because
this package-to-planner path is now green.

## Review follow-up: canonical pointer and typed numeric equality

Review found two fail-closed gaps in the Task 2B1 implementation:

1. The generic loader, bootstrap verifier, and paired launcher authenticated
   the dataset-pointer bytes but checked only provider, receipt path, and
   `full_corpus_in_release`. Hash-consistent pointers with missing fields,
   unknown fields, concrete dataset identity, or a float schema version could
   pass runtime validation.
2. Several nested comparisons used Python equality. Consequently `false` and
   `0.0` could compare equal to integer seed `0`, and integer-valued floats
   could compare equal to assignment/package integer fields.

The follow-up implementation commit is:

- `4e3c74eea7ef60a65f449ed7b00b74a9c0f76434`
  (`fix: close Task 2B1 typed contracts`)
- This report append is committed separately in the commit containing this
  section.

### Review RED

The tests were added before the review-fix production changes. Package
mutation helpers rebuilt hash-consistent archives, internal metadata,
`SHA256SUMS`, external receipts, and launcher roots so each mutation reached
the intended semantic consumer rather than failing at an earlier hash check.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task2b1-review-red \
  tests/test_aws_contract_roundtrip.py::\
test_real_v3_loader_rejects_noncanonical_dataset_pointer \
  tests/test_aws_contract_roundtrip.py::\
test_real_v3_bootstrap_rejects_noncanonical_dataset_pointer \
  tests/test_aws_contract_roundtrip.py::\
test_real_v3_loader_rejects_outer_seed_numeric_alias \
  tests/test_aws_contract_roundtrip.py::\
test_real_v3_bootstrap_rejects_internal_seed_numeric_alias \
  tests/test_aws_contract_roundtrip.py::\
test_real_v3_loader_rejects_integer_valued_assignment_float \
  tests/test_aws_p5_launcher.py::\
test_launcher_rejects_noncanonical_dataset_pointer \
  tests/test_aws_p5_launcher.py::\
test_launcher_rejects_internal_seed_numeric_alias \
  tests/test_aws_p5_launcher.py::\
test_launcher_rejects_bootstrap_receipt_numeric_alias
```

Output:

```text
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF                                       [100%]
...
34 failed in 10.21s
```

Every case failed with `DID NOT RAISE`, confirming that all mutations passed
the pre-fix consumers:

- pointer field deletion, unknown-field addition, and `schema_version: 1.0`;
- injected `dataset_receipt_sha256`, `dataset_build_id`, `build_id`, and
  `ordered_stream_sha256`;
- package seed `0` replaced by `false` or `0.0` at loader, bootstrap, and
  launcher boundaries;
- integer-valued floats in all four cohort assignment geometry fields;
- bootstrap receipt schema, durable-upload boolean, and instance-store device
  count numeric aliases.

### Review implementation

`msctl/contracts.py` now provides one recursive type-aware comparison for
JSON-like values. It requires identical concrete types before comparing
dictionaries, lists, or scalar values, so Python's `False == 0` and
`0.0 == 0` behavior cannot satisfy an exact contract.

The same module defines the exact canonical Task 2A dataset pointer:

- all and only the 11 canonical fields;
- exact strings, boolean, integer schema version, ordered sidecar list, and
  values;
- no concrete dataset receipt hash, build ID, or ordered-stream identity.

The generic loader invokes this validator on the authenticated pointer member.
Bootstrap and launcher invoke the same validator through runtime-local imports,
preserving their `python -S ... --help` behavior without duplicating the
canonical object.

Type-aware equality now covers:

- v3 internal and external seed assignments;
- loader and bootstrap external/internal release-receipt bindings;
- launcher release metadata and bootstrap receipt bindings;
- nested bootstrap instance-store evidence.

The four integer cohort assignment fields handled by
`msctl/contracts.py` now require exact `int` types, matching the already strict
bootstrap and launcher checks. The historical package-format-1 branch remains
available for valid read-only legacy receipts; only numerically aliased
noncanonical values are newly rejected.

### Review focused GREEN

The unchanged RED command after the production fix produced:

```text
..................................                                       [100%]
34 passed in 9.60s
```

Focused regression files:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task2b1-review-roundtrip \
  tests/test_aws_contract_roundtrip.py
```

```text
26 passed in 12.05s
```

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task2b1-review-launcher \
  tests/test_aws_p5_launcher.py
```

```text
135 passed in 5.34s
```

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task2b1-review-msctl \
  tests/test_msctl.py
```

```text
157 passed in 44.99s
```

### Review required seven-file GREEN

Command:

```bash
git diff --check
python -m py_compile \
  msctl/contracts.py \
  cluster/aws/p5/bootstrap.py \
  cluster/aws/p5/launch_seed_pair.py \
  tests/test_aws_contract_roundtrip.py \
  tests/test_aws_p5_launcher.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task2b1-review-final \
  tests/test_aws_contract_roundtrip.py \
  tests/test_aws_p5_launcher.py \
  tests/test_package_aws_p5_handoff.py \
  tests/test_aws_contracts.py \
  tests/test_cohort_assignment_v3.py \
  tests/test_aws_p5_profile.py \
  tests/test_msctl.py
```

Output:

```text
........................................................................ [ 14%]
........................................................................ [ 28%]
........................................................................ [ 43%]
........................................................................ [ 57%]
........................................................................ [ 71%]
........................................................................ [ 86%]
......................................................................   [100%]
502 passed in 97.07s (0:01:37)
```

`git diff --check` and `py_compile` produced no output and exited `0`.

### Review scope and concerns

Only the three reviewed consumers and their Task 2B1 tests changed in the
implementation commit. No packager, lifecycle/state/operation, evaluation,
IaC, verifier, runbook, Illumina, v2 artifact, or `corpusgen/` file changed.

No known Task 2B1 review finding remains. Broader Task 2B2 work remains
deliberately out of scope.
