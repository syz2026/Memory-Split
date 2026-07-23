# Task 8 report: sealed confirmatory runner

Date: 2026-07-23
Branch: `feat/memorysplit-v2-confirmatory-runner`
Base: `c190429b1542`
Reviewed hardening revision: `198d2953ae2ab41c844a79b79e71c3cf6311f730`

## Outcome

Task 8 is implemented in the requested isolated worktree. The scoped code adds
an injected `ModelAdapter`, deterministic fixture adapter, strict preflight and
evaluation orchestration, and a JSON-only command-line boundary.

Files added:

- `evals/confirmatory/runner.py`
- `evals/confirmatory/__main__.py`
- `tests/test_confirmatory_runner.py`
- `.superpowers/sdd/task-8-report.md`

No reporting, contracts, metrics, solver, or unrelated subsystem file was
modified.

## Trust and replay behavior

- Requires an external lowercase study-lock SHA-256 and authenticates it before
  opening model-visible items.
- Parses model-visible items before model invocation and opens sealed gold only
  after every model submission has been collected.
- Binds checkpoint bytes, configuration bytes, route-dose, corpus, code, seed,
  arm, and explicit `dense`/`split90` condition identity across the run,
  checkpoint registry, and study lock; generic `split` is rejected.
- Requires one answer and exactly 12 validated action slots per item.
- Replays submissions through the registered trusted solver and public metrics
  APIs. Persisted outcomes contain submissions, not model-asserted correctness.
- Produces canonical artifacts and creates a new output directory and files
  with no-replace semantics.
- Requires the externally hash-rooted hardened reporting signatures from
  revision `198d295`; the older API fails closed before publication with the
  exact required interface.
- The CLI defaults to preflight without `--output-dir`, emits exactly one JSON
  object to stdout on success, help, and error paths, and sends diagnostics and
  help text to stderr.

## TDD and verification evidence

Tests were written before implementation. Initial red runs failed because
`runner.evaluate`, the reporting boundary, and `evals.confirmatory.__main__`
did not yet exist. The final help-path regression was also observed red with
`SystemExit: 0` and argparse help on stdout before the parser fix.

Final focused command:

```text
python -m pytest -q tests/test_confirmatory_runner.py \
  tests/test_confirmatory_reporting.py tests/test_confirmatory_validation.py
```

Result: `48 passed in 5.32s`. The runner-only suite reports `20 passed`.

Quality gates:

```text
uvx ruff check evals/confirmatory/runner.py \
  evals/confirmatory/__main__.py tests/test_confirmatory_runner.py
uvx ruff format --check evals/confirmatory/runner.py \
  evals/confirmatory/__main__.py tests/test_confirmatory_runner.py
git diff --check
```

Result: all Ruff checks passed, all three files were already formatted, and
the diff whitespace check passed. IDE diagnostics reported no errors.

A live compatibility replay loaded the runner against committed hardening
revision `198d295`, evaluated 160 fixture items, published the hardened report,
and verified report/study-lock binding. Report SHA-256:
`ebd7592ba1f49d65b631df9a06a504eb23cb3c452ba71165581d1748013599a7`.

## Concerns and limits

- `tests/test_confirmatory_replay.py`, named by the approved brief, does not
  exist at base `c190429`, so that one file could not be run on this isolated
  branch. The available reporting and validation suites passed, and replay
  compatibility was exercised directly against the latest committed hardening
  implementation.
- The base branch's reporting API is intentionally insufficient and full
  publication fails closed until revision `198d295` (or an equivalent hardened
  interface) is integrated.
- A production checkpoint-backed adapter and GPU inference are outside this
  task's scoped files; full CLI publication therefore requires an injected
  adapter. Dry-run preflight is directly executable.
