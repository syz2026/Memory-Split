# FarmShare v2 Corpus Foundations Integration Report

## Result

- Status: `DONE_WITH_CONCERNS`
- Branch: `integration/farmshare-v2-corpus-foundations`
- Exact base: `0a554760c09c05a2ec39a8049e02415d012a86e5`
- Reviewed implementation head: `4debe60a6f5742035980d40c5ec49855e9e86239`
- Integration method: clean, ordered cherry-picks
- Implementation conflict resolutions: none

## Integrated sequence

| Reviewed commit | Integration commit | Subject |
| --- | --- | --- |
| `d7a81a65081ef71c9d747762bd44842e152078a5` | `cdf30bac451d6fb764409402fc3b089b848ec7d2` | feat: freeze v2 corpus geometry |
| `1706b18586657e66a2062577ff79c915d46c92d5` | `33870adea8e631520cc289982b6af0a800c3f4e0` | fix: enforce immutable v2 recipe contracts |
| `195d46b4a9ce287e4d8e1681295a4a7f471bcdc3` | `2ee4b72001cc3265c8d9c3467725845944c1b29c` | feat: add immutable v2 source freeze |
| `ed3327c3089e0fe5e0583cf2740b24c63c30d814` | `29694ed44bb23de439dafdd68b97bff2853659a8` | fix: harden v2 source freeze authority |
| `53ad2c90121dfe7e0314da81d15f3e614d6cbbc0` | `3a43a43d3df8d25b49b37e3ffe8a08f59412d5b9` | fix: close v2 source-freeze review gaps |
| `18950a49abe30bb6341040518b9977c931855d36` | `ee259cd8fe650b9319de754ffadfa7b647bbc1fd` | fix: eliminate final source-freeze TOCTOU |
| `3ae9ad9ade072d1a2065dd196d1b8f1fb917b7d4` | `20acf37c149b3d4e4d013cfa8a598934e3383a5b` | fix: preserve benign duplicate source stages |
| `8eb2162ba450b50f88a4a4d973caccdec586b74f` | `4debe60a6f5742035980d40c5ec49855e9e86239` | fix: replay concurrent source winners |

Both `git range-diff` checks pair every reviewed commit with `=`. Stable
patch IDs also match one-for-one, subjects are unchanged, and the integrated
parents form the requested linear order.

## Public API and identity audit

- `corpusgen.reasoning_v2.__all__` exposes the nine Task 1 symbols exactly:
  `LANE_ORDER`, `BuildGeometry`, `LaneContract`, `LaneId`,
  `ReasoningV2Recipe`, `balanced_record_lengths`, `geometry_for`,
  `hamilton_quotas`, and `load_recipe`.
- `from corpusgen.reasoning_v2 import source_lock` resolves the Task 2
  submodule normally. No package export/import conflict occurred, so
  `__init__.py` was not changed beyond the reviewed Task 1 patch.
- The loaded recipe and source-lock module share dataset ID
  `memorysplit-v2-20x-reasoning-max-cohort`.
- Both bind `configs/reasoning-dataset-v2.json`; its SHA-256 is
  `c704baf9e07acb93ffab8b33027b310fc9b3d68e0c036a6d8cc1d45b86872ce8`.
- `load_source_lock`, `verify_source_tree`, and `stage_source_lock` require an
  explicit keyword-only `expected_generator_commit`; `resolve_source_lock`
  requires an explicit `generator_commit`. The targeted authority selection
  passed three tests, including rejection of a mismatched generator commit.

## Concern: reviewed recipe/source-lock boundary

The direct reviewed boundary does not compose. Task 1 deliberately deep-freezes
`ReasoningV2Recipe.source_policy` as `mappingproxy`, while Task 2
`_validate_recipe()` requires `isinstance(source_policy, dict)`. Therefore:

```text
source_policy_type=mappingproxy
ValueError: recipe source policy is missing
```

This occurs for the intended
`resolve_source_lock(load_recipe(...), ...)` path before resolver transport.
Dataset and contract identities agree, but end-to-end recipe-to-source-lock
acceptance does not. This is not a package export/import conflict, and changing
the implementation would violate the exact-reviewed-commits constraint, so no
source fix was made. A separately reviewed follow-up should accept
`collections.abc.Mapping` (or otherwise define a shared recipe protocol).

## Verification

- Exact-base full baseline: `1393 passed, 2 failed, 2 deselected`.
  Both pre-existing failures are unrelated AWS cohort-release/runbook checks in
  `tests/test_verify_cohort_releases.py`.
- Task 1, Task 2, current-source, parallel-corpus, and generic v2 regressions:
  `337 passed in 28.27s`.
- Generator-commit authority selection: `3 passed, 84 deselected`.
- Public package API, source submodule import, dataset ID, contract path/digest,
  and required authority signatures: passed.
- Direct immutable recipe to source-lock validation: failed as documented
  above.
- `python -m py_compile` on all seven integrated Python files: passed.
- `git diff --check 0a554760..4debe60`: passed.

The scoped pytest command covered:

```text
tests/test_reasoning_v2_contracts.py
tests/test_reasoning_v2_contracts_strict.py
tests/test_reasoning_v2_source_lock.py
tests/test_current_sources.py
tests/test_parallel_corpus.py
tests/test_reasoning_v2.py
tests/test_v2_contract_files.py
tests/test_v2_readiness.py
tests/test_cohort_assignment_v2.py
```

## Exact-range audit

The reviewed implementation range contains eight commits and only these seven
reviewed files:

```text
corpusgen/reasoning_v2/__init__.py
corpusgen/reasoning_v2/contracts.py
corpusgen/reasoning_v2/source_lock.py
tests/reasoning_v2_fixtures.py
tests/test_reasoning_v2_contracts.py
tests/test_reasoning_v2_contracts_strict.py
tests/test_reasoning_v2_source_lock.py
```

No unrelated report or current seven-lane file was added to the reviewed
implementation range. This report is committed separately as a report-only
convention commit.
