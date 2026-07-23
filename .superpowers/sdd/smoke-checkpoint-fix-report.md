# Smoke checkpoint fix report

- Status: PASS.
- Branch: `fix/relational-smoke-checkpoint-sha256`.
- Exact base: `bfb329fe23368f96fd4b76f3c3df9bda815f4aed`.
- Root cause: each new smoke `Trainer` treated the prior trainer's checkpoint as external, but both resume calls omitted the SHA-256 required by hardened checkpoint loading.
- RED: `pytest -q tests/test_relational_smoke.py` failed `1` test in `3.23s` with `ValueError: external checkpoint load requires resume_sha256`.
- Fix: `scripts/relational_smoke_test.py` now pins `trainer.ckpt_path` with `train.safeio.read_regular_path` and passes the resulting digest to both `load_ckpt` calls.
- GREEN: `pytest -q tests/test_relational_smoke.py` passed `1` test in `4.25s`.
- Compile: `python -m py_compile scripts/relational_smoke_test.py tests/test_relational_smoke.py` passed.
- Diff: `git diff --check` passed.
- Lint: `uvx ruff check scripts/relational_smoke_test.py tests/test_relational_smoke.py` passed.
- Scope: no test change was required; Trainer validation and Task 4 files were not altered.
