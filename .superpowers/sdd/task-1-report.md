# Task 1 report: prospective AWS-only N=10 scientific contract

## Status

Implemented and verified the prospective `memorysplit-confirmatory-v3` contract
on branch `feat/memorysplit-v3-aws-n10`, based on
`33b8c9f5452601cd68f3bf4583a8c78ad674e723`.

The implementation commit is:

- `ade5fce6b935e162d8cf6672b5046cdd24aa6e33`
  (`feat: freeze AWS-only N=10 contract`)
- The report is committed separately in the commit containing this file.

## TDD record

### Focused v2 baseline

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task1-baseline \
  tests/test_cohort_assignment_v2.py \
  tests/test_aws_p5_profile.py \
  tests/test_v2_contract_files.py
```

Output:

```text
........................................................................ [ 75%]
.......................                                                  [100%]
95 passed in 1.35s
```

### RED: v3 cohort contract

`tests/test_cohort_assignment_v3.py` was added before the v3 implementation.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task1-red-cohort \
  tests/test_cohort_assignment_v3.py
```

Expected RED output:

```text
FFFFFFFFFFFFFFFFFFF                                                      [100%]
...
FileNotFoundError: .../configs/cohort-assignment-v3.json
FileNotFoundError: .../configs/preregistration-v3.yaml
FileNotFoundError: .../configs/360m-v3
TypeError: load_cohort_assignment_bytes() got an unexpected keyword argument
           'assignment_filename'
...
19 failed in 0.42s
```

The failures were caused by the absent v3 artifacts and absent explicit bytes
loader identity, not by test syntax or fixture errors unrelated to the feature.

### RED: v3 P5 profile

The v3 profile tests were added before the profile implementation.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task1-red-profile \
  tests/test_aws_p5_profile.py -k v3
```

Expected RED output:

```text
FFFFFFFF                                                                 [100%]
...
ValueError: profile is not a regular file: .../aws-p5.48xlarge-v3.json
FileNotFoundError: .../aws-p5.48xlarge-v3.json
...
8 failed, 48 deselected in 0.20s
```

### Initial GREEN: new v3 behavior

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task1-green-new \
  tests/test_cohort_assignment_v3.py tests/test_aws_p5_profile.py \
  -k 'v3 or bytes_loader_requires_explicit_v3'
```

Output:

```text
...........................                                              [100%]
27 passed, 48 deselected in 0.76s
```

### Self-review correction: unchanged profile schema

Self-review found that the P5 profile shape is unchanged, so its schema remains
`1`; the new contract is selected by the explicit
`(schema_version, profile_id)` identity. The corrected tests were changed first.

RED command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task1-red-profile-schema \
  tests/test_aws_p5_profile.py -k v3
```

RED output:

```text
F.....F.                                                                 [100%]
...
2 failed, 6 passed, 48 deselected in 0.06s
```

GREEN command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task1-green-profile-schema \
  tests/test_aws_p5_profile.py -k v3
```

GREEN output:

```text
........                                                                 [100%]
8 passed, 48 deselected in 0.05s
```

### Required GREEN and v2 regressions

Command required by the brief:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --basetemp=/tmp/memorysplit-v3-task1 \
  tests/test_cohort_assignment_v3.py \
  tests/test_cohort_assignment_v2.py \
  tests/test_aws_p5_profile.py \
  tests/test_v2_contract_files.py
```

Output:

```text
........................................................................ [ 59%]
..................................................                       [100%]
122 passed in 1.99s
```

After the profile-schema self-review correction and strengthened gate
assertions, the same required test set was rerun with the unique base temp
`/tmp/memorysplit-v3-task1-final`:

```text
........................................................................ [ 59%]
..................................................                       [100%]
122 passed in 1.95s
```

## Changed files

Production validation:

- `msctl/cohort.py`
  - adds immutable v2/v3 cohort specifications;
  - binds assignment filename, preregistration identity, config directory,
    provider assignment, cell paths, run IDs, and dataset receipt path;
  - keeps the bytes loader backward-compatible with v2 while requiring the
    explicit v3 assignment filename for v3 snapshots.
- `cluster/aws/p5/profile.py`
  - adds immutable profile identities for the existing and v3 contracts;
  - preserves closed root/nested-field and exact hardware/runtime validation;
  - validates exact profile-specific seed assignments.

Frozen v3 artifacts:

- `configs/preregistration-v3.yaml`
- `configs/cohort-assignment-v3.json`
- `configs/360m-v3/dense-s0.yaml` through `dense-s9.yaml`
- `configs/360m-v3/split90-s0.yaml` through `split90-s9.yaml`
- `cluster/profiles/aws-p5.48xlarge-v3.json`

Behavioral tests:

- `tests/test_cohort_assignment_v3.py`
- `tests/test_aws_p5_profile.py`

Task record:

- `.superpowers/sdd/task-1-report.md`

No `corpusgen/`, packaging, launcher, bootstrap, evaluator, runbook, Illumina,
or existing v2 contract file was modified.

## Scientific contract self-review

Checked the v3 contract against the binding brief:

- IDs are exactly `memorysplit-confirmatory-v3` and
  `memorysplit-confirmatory-v3-360m-n10-aws`.
- Preregistration and assignment schemas are `3`; all 20 trainer configs retain
  schema `2`; the unchanged P5 profile schema remains `1` and is distinguished
  by `profile_id: aws-p5.48xlarge-v3`.
- Provider assignment is exactly AWS seeds `0..9`, with no Illumina entry.
- Conditions are exactly Dense and Split90.
- Model parameters, context, targets/update, updates, raw targets/arm, and
  snapshots are exactly `356033536`, `1024`, `524288`, `13582`,
  `7120879616`, and `[1358,3396,6791,10187,13582]`.
- There are 10 paired units, 20 training cells, and 100 scheduled snapshot
  evaluation cells.
- The profile and preregistration freeze one On-Demand `p5.48xlarge`, eight
  H100 80GB GPUs, Dense ranks `0..3`, Split90 ranks `4..7`, and symmetric
  `4+4` groups.
- Every run config uses `dataset/receipt.json`, the required v3 run ID,
  seed/arm sidecar, and output identity; all other training fields are exact
  and matched.
- The prospective amendment records no inspected seed/arm confirmatory result,
  all-pair continuation independent of effect direction, no extra trained
  controls, and the unchanged scientific dimensions.
- The primary final-step test freezes 10 pairs, all 1024 one-sided exhaustive
  sign assignments, inclusive equality, retained zero deltas, alpha `0.05`,
  and minimum p `0.0009765625`.
- AULC is a required secondary (not a second primary), uses all five exact
  checkpoints with no interpolation, and uses the right-step integral.
- Bootstrap is PCG64, seed `0`, 20,000 draws, 90% interval; practical
  equivalence uses strict `±0.01` bounds.
- Terminal status requires all 10 paired bundles and all validity/evaluation
  evidence.
- The six 29M diagnostics remain protected-launch prerequisites and explicitly
  forbid effect direction as a pass criterion.
- Evaluation strata, controls, route-dose, semantic-closure, proof-verification,
  validity gates, and artifact bindings remain present; protected launch is
  false and all source-controlled production artifact hashes remain null.

## Closed-validation self-review

Mutation tests reject:

- Illumina reintroduction;
- seed 10 and duplicate provider seeds;
- missing and extra cells;
- wrong dataset paths, run IDs, snapshot schedules, and per-file seed identity;
- v2 files substituted into v3;
- cross-version bytes-loader identity;
- missing, duplicate, reordered, negative, and seed-10 P5 assignments;
- mismatched P5 schema/profile identities.

The assignment and run-config validators retain exact closed fields. The P5
parser retains exact closed fields at every level and exact hardware, storage,
runtime, allowlist, purchase-model, topology, and seed values.

## V2 byte identity and patch checks

V2 byte identity command:

```bash
git diff --exit-code 33b8c9f5452601cd68f3bf4583a8c78ad674e723 -- \
  configs/360m-v2 \
  configs/preregistration-v2.yaml \
  configs/cohort-assignment-v2.json \
  cluster/profiles/aws-p5.48xlarge.json
```

Output: none; exit code `0`.

Patch check:

```bash
git diff --check
```

Output: none; exit code `0`.

## Concerns

No Task 1 implementation concern remains. The repository-wide suite was not
used as the acceptance gate because the brief identifies two pre-existing,
out-of-scope failures in `tests/test_verify_cohort_releases.py`; the required
focused v3/v2 suite is green.
