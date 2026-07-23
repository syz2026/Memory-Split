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
