# Task 8 report: sealed confirmatory runner

Date: 2026-07-23
Branch: `feat/memorysplit-v2-confirmatory-runner`
Base: `c190429b1542`
Validated hardening revision: `f9341246012e83c3647069840218261e860466f6`

## Outcome

The review findings are fixed in the requested isolated worktree. The CLI now
constructs a real checkpoint-backed `RepositoryGPTAdapter` whenever
`--output-dir` is supplied, while tests may still inject the strict
`ModelAdapter` or `DeterministicFixtureAdapter`.

Scoped files:

- `evals/confirmatory/runner.py`
- `evals/confirmatory/__main__.py`
- `tests/test_confirmatory_runner.py`
- `.superpowers/sdd/task-8-report.md`

No reporting, contracts, metrics, solver, or unrelated subsystem file was
modified.

## Trust, model, and replay behavior

- Requires and authenticates an external lowercase study-lock SHA-256 before
  opening model-visible items. The permissive local validator is gone; the
  externally rooted `f934124` study-lock/readiness API is mandatory.
- Rejects malformed frozen preregistration, control registries, receipt
  commitments, and drift in the repository graph-token protocol or `r0`-`r15`
  relation vocabulary.
- Loads hash-bound config/checkpoint bytes with safe PyTorch loading, checks
  seed, explicit `dense`/`split90` identity, architecture metadata, strict
  state-dict shape/keys, and selected device availability.
- Gives the model only a validated `ItemRecord` and its validated `StoreRecord`;
  `memory_off` receives no store. Sealed gold is not opened until every
  submission has been generated.
- Executes the frozen read/return protocol for exactly 12 slots, caps reads at
  10, and pads every post-HALT slot with canonical NOOP. Store targets and
  addresses never become action candidate lists.
- Replays submitted answers/proofs through the trusted solver. Persisted
  outcomes remain submission-only and contain no asserted correctness flags.
- Requires the hardened reporting API and publishes all evidence in an owned
  sibling staging directory. Every artifact and report is fsynced, the staging
  directory is fsynced, and the complete directory is atomically published
  no-replace before the parent is fsynced.
- Cleans or quarantines failed staging trees; injected write/report failures
  leave no final output, and destination collisions preserve the other owner.
- Exposes `MSCTL_EVALUATOR_CONTRACT =
  "memorysplit-confirmatory-evaluator-v1"`. Module and direct-script help emit
  one machine-readable JSON object to stdout; help/log text goes to stderr.

## TDD and verification evidence

RED was observed before each implementation slice: missing production adapter
and contract symbols, permissive validation, non-transactional publication,
and the graph-protocol drift regression all failed first. The final protocol
RED was `DID NOT RAISE`; its focused GREEN result was `1 passed`.

The final cross-worktree run loaded this runner with the committed confirmatory
modules from `f934124` and ran the runner plus every `test_confirmatory*.py`:

```text
136 passed in 41.07s
```

This comprises 34 runner tests and 102 hardening tests, including real hardened
report construction/publication, replay, trust, metrics, validation, contracts,
and inference fixtures. The real tiny GPT/checkpoint CLI test evaluates 160
items across both memory boundaries without mocking reporting.

Additional gates:

```text
module CLI help: evaluate help 1 True
direct CLI help: evaluate help 1 True
ruff check: All checks passed
ruff format --check: 3 files already formatted
py_compile: passed
git diff --check: passed
```

## Concerns and limits

- The branch base intentionally predates the hardened modules. Evaluation and
  publication therefore fail closed until `f934124` (or its exact compatible
  API) is integrated.
- CPU production construction and inference are exercised with a tiny real
  checkpoint. CUDA and MPS availability paths were not executable on this host.
- Atomic no-replace directory publication supports Darwin and Linux and fails
  closed on unsupported platforms.
