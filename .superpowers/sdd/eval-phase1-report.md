# Sealed evaluation phase-one report

- Status: `DONE_WITH_CONCERNS`.
- Branch: `feat/sealed-eval-foundation-v3`.
- Exact base: `b3471e0969ca2a997d33acf60d2e777720afa1c4`
  (`feat/memorysplit-v3-aws-n10`).
- Implementation commit:
  `a7f48db66f65f5678616313c481aeecaba88fa69`.
- Frozen preregistration SHA-256:
  `6b2b5da3e3dc3d533498a0aa9d1891f356134ce045b1553b74a0161d94cb81d7`.

## Scope delivered

1. Added explicit v3 N=10 inference entry points: the one-sided exhaustive
   1,024-assignment sign flip, PCG64 seed-0 20,000-draw 90% hierarchical
   practical-equivalence bounds, strict open `(-0.01, +0.01)` acceptance, and
   the five-point right-step AULC integral.
2. Preserved item, sealed-gold, and store schemas at v2. Added separate v3
   checkpoint, outcome, and metrics records that require seeds 0–9, Dense or
   Split90, one frozen optimizer step, and
   `raw_token_count == optimizer_step * 524288`.
3. Added an immutable v3 lock over the exact ordered 100
   `(seed, arm, optimizer_step)` slots. Every slot binds a checkpoint hash,
   the canonical key returned by `msctl.aws_contracts.checkpoint_object_key`,
   and a non-null S3 version ID; the lock also binds the frozen
   preregistration and sealed-evaluation release hashes.
4. Did not add `sealing.py` or `aggregate.py`, and did not edit configuration,
   corpus generation, P5 lifecycle, IaC, packaging, runbooks, or v2 historical
   replay behavior.

## TDD evidence

### RED

Before any production edit:

```text
python -m pytest tests/test_confirmatory_v3_inference.py \
  tests/test_confirmatory_v3_records.py \
  tests/test_confirmatory_v3_study_lock.py -q
exit 2: 3 collection errors in 0.33s
ImportError: V3_BOOTSTRAP_CONFIDENCE, STUDY_CHECKPOINT_SCHEMA,
and EXPECTED_STUDY_SLOTS_V3 were absent.
```

The package-export contract was also observed RED before export changes:

```text
python -m pytest \
  tests/test_confirmatory_v3_records.py::test_confirmatory_package_exports_the_v3_foundation_api -q
exit 1: 1 failed in 0.15s
AttributeError: evals.confirmatory had no StudyCheckpointRecord.
```

### GREEN

```text
tests/test_confirmatory_v3_inference.py -q
4 passed in 5.16s

tests/test_confirmatory_v3_records.py -q
8 passed in 0.45s

tests/test_confirmatory_v3_study_lock.py -q
6 passed in 0.39s

three focused v3 files together
19 passed in 5.46s

bounded existing confirmatory + aws_contracts regression suite
156 passed in 34.47s
```

`python -m py_compile` passed for all five changed production modules and all
three focused test modules. `git diff --check` passed.

## Files

- `evals/confirmatory/__init__.py`
- `evals/confirmatory/contracts.py`
- `evals/confirmatory/inference.py`
- `evals/confirmatory/metrics.py`
- `evals/confirmatory/study_lock.py`
- `tests/test_confirmatory_v3_inference.py`
- `tests/test_confirmatory_v3_records.py`
- `tests/test_confirmatory_v3_study_lock.py`
- `.superpowers/sdd/eval-phase1-report.md`

## Self-review

- Confirmed the source worktree stayed at
  `a7a5c1f1f06fb424c2a660dc5642bb0bfbcb085d`; all edits occurred in the
  isolated worktree.
- Confirmed exact slot order is seed, then `dense`/`split90`, then the five
  frozen steps, with no random arm and no missing, duplicate, reordered, or
  replacement slot accepted.
- Confirmed strict integer checks reject booleans and floating-point schema,
  seed, step, and token-count lookalikes.
- Confirmed the v2 semantic constants and focused v2 replay suites remain
  unchanged and green.
- Confirmed no excluded file was changed.

## Concerns

- The host data volume reports 100% utilization with about 1.0 GiB available.
  One large patch write failed with `No space left on device`; it left no
  partial production edit, and all subsequent edits, tests, compilation, diff
  checks, and the implementation commit completed successfully.

## Review-fix addendum

- Review-fix status: `DONE_WITH_CONCERNS`.
- Review-fix commit:
  `b20ca76bbad4261562c85e4437a83cd5fac981b3`.

### Findings resolved

1. Exact v3 inference now carries `Fraction` identities from rate numerators
   and denominators through paired metric deltas, right-step AULC, bootstrap
   values, strict practical-equivalence comparisons, and the exhaustive
   sign-flip statistic. Binary floats are rejected by authoritative v3
   boundary and exact-test APIs; serialized floats remain convenience values.
2. `BootstrapEstimate` now rejects non-finite effects/replicates, booleans and
   non-integer metadata, impossible count hierarchies, length mismatches, and
   estimates or bounds inconsistent with their exact inputs.
3. `StudyLockV3` now rejects checkpoint-hash or versioned-object aliases across
   slots. Each snapshot also binds a content-addressed paired-checkpoint
   receipt hash, object key, and version ID; both arms at one seed/step must
   share that receipt, and receipt evidence cannot be reused across steps.
4. Snapshot and receipt version IDs are treated as opaque candidates and must
   be 1–1,024 printable non-whitespace characters other than literal `null`.
   The contract deliberately does not claim cryptographic reality until later
   versioned HEAD and receipt replay.
5. The v3 outcome binder now authenticates the unchanged v2 item identity,
   v3 checkpoint identity, canonical S3 object identity, evidence references,
   and membership in the exact lock snapshot.
6. Public v3 records and locks now use `StudyArm.DENSE`/`StudyArm.SPLIT90`;
   legacy v2 `Arm.DENSE`/`Arm.SPLIT`/`Arm.RANDOM` values remain unchanged.

### Review RED evidence

```text
exact inference/bootstrap group
7 failed, 9 passed in 0.64s
- Fraction inputs were rejected.
- the exact paired-delta API was absent.
- bootstrap metadata/effects were insufficiently validated.

StudyArm/cross-record binder group
20 failed in 1.52s
- split90 was rejected by the legacy Arm enum.
- the package had no StudyArm.
- the binder did not accept item or lock-snapshot identities.

lock provenance/version group
22 failed, 4 passed in 0.46s
- checkpoint receipt evidence fields were absent.
- malformed version IDs reached only the old field-set rejection.
- a same-object alias across optimizer steps was accepted.
```

### Review GREEN evidence

```text
exact inference/bootstrap group
16 passed in 12.92s

StudyArm/cross-record binder file
20 passed in 0.25s

lock provenance/version group
26 passed in 0.53s

all focused v3 tests
59 passed in 6.25s

bounded v2 confirmatory + aws_contracts regressions
156 passed in 60.59s
```

`python -m py_compile` passed for all five changed production modules and all
three focused test modules. `git diff --check` passed. The review fixes touched
only the original five confirmatory modules and three focused v3 test files;
no sealing, aggregation, lifecycle, provider-selection, configuration,
corpus-generation, IaC, packaging, or runbook file was added or changed.
