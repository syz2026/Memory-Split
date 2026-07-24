# Objective Controls Amendment V3 Report

## Status

DONE on `feat/memorysplit-v3-objective-controls-amendment`.

- Pinned base:
  `b3471e0969ca2a997d33acf60d2e777720afa1c4`
- Implementation commit:
  `e09da5bbe401182c54ed22b785099e61be0d1950`
- Implementation tree:
  `63ff168dde11745ad83ecc276d4ac713c2334c02`

The implementation commit is a direct child of the requested pinned base.

## Scientific decision encoded

- `configs/objective-controls-amendment-v3.yaml` is an append-only,
  prospective amendment.
- It records zero inspected protected outcomes at seed, arm, and aggregate
  levels.
- It enumerates the unchanged 20 protected cells: Dense and Split90 for seeds
  0 through 9.
- It adds no 360M control and disclaims any replicated 360M
  selectivity-vs-random claim.
- It excludes all eight 29M development runs from the N=10 exact test,
  fixed-checkpoint AULC, practical-equivalence test, continuation decisions,
  and any effect-direction gate.
- The 29M matrix preserves the original six diagnostics and adds only
  `full_corpus_random_fact90` and
  `full_corpus_matched_nonfactual_mask` as integrity-only controls.
- The no-ARC replacement remains token-matched `fineweb_edu`; the
  no-refinement replacement remains
  `verified_standard_relational_records`.

## Frozen identities

- `configs/preregistration-v3.yaml`:
  `6b2b5da3e3dc3d533498a0aa9d1891f356134ce045b1553b74a0161d94cb81d7`
- `configs/cohort-assignment-v3.json`:
  `47faf6f15e13336f67666081c3d0ebf5be3b981f3d2ac455385faa39e543199c`
- `configs/objective-controls-amendment-v3.yaml`:
  `a14ad220e8ccd7b1f07859767b5ea759197b1d45a72da03386ec4f95dde2fe92`
- `configs/29m-v3/manifest.json`:
  `a03194591977fccaec4acd0c74c2bc13484f40854208948c7172f2bea4fe8781`

The two frozen parent files and all `configs/360m-v3` files have zero diff
from the pinned base.

## Delivered

- Eight exact 29M YAML configs and one canonical JSON manifest.
- Every run binds:
  - 28,969,216 parameters;
  - seed 0 and `memorysplit-v3-29m-shared-init-s0`;
  - 524,288 targets/update;
  - 1,106 updates;
  - 579,862,528 raw targets;
  - explicit corpus-variant, sidecar, and provenance IDs.
- Both new controls use the full-corpus receipt with distinct random-fact90
  and matched-nonfactual-mask sidecars.
- `msctl/objective_controls_v3.py` is a standalone parser/validator and is not
  imported by the shared lifecycle or CLI.
- Future admission evidence fails closed on:
  - wrong amendment, manifest, config, corpus, sidecar, or provenance binding;
  - missing, extra, duplicate, or reordered runs;
  - bool/float/integer aliases;
  - non-finite optimization;
  - non-exact checkpoint/resume or delta greater than `1e-5`;
  - language degradation greater than 1%;
  - failed route, mask, or semantic-closure audit;
  - any original-run primary-cell accuracy not strictly greater than `0.75`;
  - any directional accuracy payload on the two integrity-only controls.

## TDD evidence

RED:

```text
python -m pytest -q -p no:cacheprovider \
  tests/test_objective_controls_v3.py

ModuleNotFoundError: No module named 'msctl.objective_controls_v3'
```

GREEN after the minimal contract implementation:

```text
47 passed in 3.57s
```

Final focused verification:

```text
python -m pytest -q -p no:cacheprovider \
  tests/test_objective_controls_v3.py tests/test_cohort_assignment_v3.py

66 passed in 6.18s
```

Static and diff verification:

```text
python -m py_compile \
  msctl/objective_controls_v3.py tests/test_objective_controls_v3.py
git diff --check
git diff --exit-code b3471e0969ca2a997d33acf60d2e777720afa1c4 -- \
  configs/preregistration-v3.yaml configs/cohort-assignment-v3.json \
  configs/360m-v3

All exited 0.
```

The pre-edit pinned V3 baseline also passed:

```text
91 passed in 67.77s
```

## Explicit exclusions

No corpus generation, training launcher, evaluator, study lock, AWS profile,
package, IaC, runtime integration, lifecycle registration, or 360M control was
implemented. Actual corpus, sidecar, and provenance content hashes remain
future admission inputs because those artifacts do not yet exist; the
validator requires lowercase SHA-256 values and exact ID/hash consistency
before admission.

## Review correction addendum

Status: DONE on `feat/memorysplit-v3-objective-controls-amendment`.

- Review-fix commit:
  `7d95d36e58a194aaf967c9130c705c3920ab0ff9`
- Review-fix tree:
  `645b5fe909766996fe6dd567ac7bca7df5a9a374`

### Authority boundary

Public admission APIs no longer accept `ObjectiveControlsContract`. Both
`validate_objective_controls_admission` and
`load_objective_controls_admission` require the canonical amendment path and
authenticate the amendment, manifest, all eight configs, frozen
preregistration, and frozen cohort bytes before evidence validation.

`ObjectiveControlsContract`, `ObjectiveControlRun`, and
`ObjectiveControlsAdmission` remain public data-only frozen dataclasses. The
only object-based evidence helper is private and is reached only after the
public path has authenticated canonical bytes.

Regression coverage for caller-forged contracts mutates:

- seed 0 to seed 99;
- amendment and manifest hashes;
- the complete statistical-exclusion set; and
- the zero-outcome-inspection declaration.

All four forged objects had authorized evidence before the fix. None is now
accepted as an authority argument.

### Anchored byte chain

The earlier amendment hash in this report is superseded by the reviewed,
manifest-binding amendment:

- amendment SHA-256:
  `376e6a2234fc89aac3baea52e09d40ac08f841102e33fde9daf473a4ca589bf8`
- canonical manifest SHA-256:
  `a03194591977fccaec4acd0c74c2bc13484f40854208948c7172f2bea4fe8781`
- aggregate objective-controls contract SHA-256:
  `222844dbf68ad9ca48be2069b5eb3b771b90166252af7eb0a4a5d8db3631adb6`

The aggregate is the SHA-256 of canonical JSON binding schema version, source
commit, frozen preregistration and cohort hashes, amendment hash, manifest
hash, and the exact path-to-hash map for all eight configs. The aggregate is
exported as `OBJECTIVE_CONTROLS_CONTRACT_SHA256`, returned in the data-only
contract view, and required in admission evidence.

The amendment now carries the exact canonical manifest hash. The loader checks
reviewed amendment and manifest digests before parsing and checks each config
against both its reviewed constant and manifest row before constructing a
contract. Semantically equivalent amendment drift and self-rehashed
manifest/config drift therefore fail closed.

### Strict RED/GREEN evidence

Authority RED:

```text
test_caller_constructed_contract_cannot_authorize_admission

4 failed: each forged contract produced "DID NOT RAISE".
```

Authority GREEN:

```text
6 passed in 2.51s
```

Anchored-chain RED:

```text
test_contract_exposes_exact_deterministic_aggregate_commitment
test_contract_rejects_semantically_equivalent_amendment_byte_drift
test_contract_rejects_manifest_rehash_drift
test_contract_rejects_config_rehash_drift_with_updated_internal_hash

4 failed: missing aggregate constants plus three "DID NOT RAISE" mutations.
```

Anchored-chain GREEN:

```text
4 passed in 1.36s
```

Final pre-commit verification:

```text
python -m pytest -q -p no:cacheprovider \
  tests/test_objective_controls_v3.py tests/test_cohort_assignment_v3.py

75 passed in 5.30s
```

`python -m py_compile` and `git diff --check` exited 0. The frozen
preregistration, cohort assignment, and all `configs/360m-v3` bytes remain
unchanged from `b3471e0`; the manifest and all eight 29M config bytes remain
unchanged from `46c1595`. The selected eight-run scope and every existing
numeric, finite, audit, language, resume, and learnability admission rule are
unchanged.

### Remaining scope

No runtime, package, run-manifest, evaluator, launcher, corpus generator, AWS
profile, IaC, study lock, or 360M control integration was added. Future
run-manifest/package work can bind the exported aggregate commitment without
changing this amendment.
