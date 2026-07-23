# Task 4 strict-checkpoint follow-up report

## Result

- Status: `DONE_WITH_CONCERNS`.
- Branch: `fix/task-4-strict-fix-followup`.
- Exact base: `bfb329fe23368f96fd4b76f3c3df9bda815f4aed`.
- Receipt-v2 parent retained: `afd80388399c37f39784e7ea9767a1332cfbf8bc`.
- Reviewed source commit: `93286b8deff24a3d9ec20c82f58a53faec267942`.

The reviewed test patch did not apply mechanically because Task 4 had added
receipt-v2 and geometry coverage in the same test files. Its four test and
production hunks were therefore ported around those additions. The resulting
four-file diff has stable patch ID
`4f5f0da01c7ec717cdd615e87e1b3f3cd96bf6b3`, exactly matching the stable patch
ID of `93286b8`.

## Semantics

- Added recursive `strict_json_identity`; it requires exact JSON container and
  scalar types, rejects bool/int/float coercion, preserves list order, requires
  string dictionary keys, and rejects non-finite floats.
- `PackedShards.validate_state_dict` and trainer checkpoint provenance
  validation both use `strict_json_identity`.
- `_validate_optimizer_state` receives `saved_step`; every nonzero-checkpoint
  AdamW parameter state must contain a finite scalar float32 step exactly equal
  to that saved global step.

## Red/green evidence

Before production changes, the four approved regressions were run as:

```text
python -m pytest -q \
  tests/test_data.py::test_cursor_state_rejects_nested_provenance_numeric_type_drift \
  tests/test_data.py::test_strict_json_identity_distinguishes_types_order_and_nonfinite_values \
  tests/test_trainer.py::test_checkpoint_rejects_nested_data_provenance_numeric_type_drift \
  tests/test_trainer.py::test_checkpoint_rejects_adamw_parameter_step_drift
```

Result: `4 failed in 7.48s`. The failures were the expected two missing
provenance rejections, missing `strict_json_identity`, and missing AdamW-step
rejection. After the production port, the identical command reported
`4 passed in 3.19s`.

## Verification

- Task 4 scoped suite:
  `python -m pytest -q tests/test_parallel_corpus.py tests/test_sharded_loader.py tests/test_ddp_trainer.py tests/test_trainer.py tests/test_data.py tests/test_model.py`
  -> `157 passed in 88.70s`.
- Data/trainer/DDP suites:
  `python -m pytest -q tests/test_data.py tests/test_trainer.py tests/test_ddp_trainer.py`
  -> `80 passed in 55.15s`.
- `python -m py_compile train/data.py train/trainer.py tests/test_data.py tests/test_trainer.py`
  passed.
- `uvx ruff check --ignore E741,F841 train/data.py train/trainer.py tests/test_data.py tests/test_trainer.py`
  passed, and `git diff --check` passed.
- Exact-patch and protected-scope checks passed: only the four reviewed source
  and test files changed before this report; `corpusgen`, `msctl`, evaluators,
  packaging, `scripts/run_train.py`, and `scripts/relational_smoke_test.py`
  remained byte-identical to `bfb329f`.

## Concern

Default Ruff reports two findings that are already present at exact base
`bfb329f`: E741 in the pre-existing log-reader comprehension in
`tests/test_trainer.py`, and F841 for receipt-v2's pre-existing `token_paths`
assignment in `train/data.py`. They were not altered because this follow-up
must remain patch-identical to `93286b8` and must not modify receipt-v2
behavior. Ruff passes after ignoring only those two demonstrated baseline
codes.
