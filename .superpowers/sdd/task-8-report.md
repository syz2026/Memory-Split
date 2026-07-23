# Task 8 report: sealed confirmatory runner

Date: 2026-07-23
Branch: `feat/memorysplit-v2-confirmatory-runner`
Base: `c190429b1542`
Merged hardening parent: `f9341246012e83c3647069840218261e860466f6`

## Outcome

Candidate A now contains the hardened confirmatory APIs from `f934124` and runs
the full suite directly from an exact checkout. The runner keeps its packaged
checkpoint-backed `RepositoryGPTAdapter`, injected test adapter boundary,
transactional evidence publication, and canonical CLI.

The merge takes the confirmatory core and its tests exactly from `f934124`,
including `study_lock.py`, sealed replay/reporting, and metrics with no exported
caller-boolean scoring functions. Runner-specific changes remain confined to
`runner.py`, `__main__.py`, `test_confirmatory_runner.py`, and this report. No
provider or launcher was modified.

## Trust, model, and replay behavior

- Requires and authenticates the external study-lock SHA-256, parses
  `validity.json`, verifies every committed gate/control/guardrail receipt, and
  requires readiness to be both complete and valid during preflight. This
  occurs before model construction/inference, sealed-gold access, or scoring.
- Uses the exact externally rooted `f934124` study-lock/readiness types and
  rejects malformed preregistration, registries, receipts, and graph protocol.
- Loads hash-bound config/checkpoint bytes with safe PyTorch loading, checks
  seed, explicit `dense`/`split90` identity, architecture metadata, strict
  state-dict shape/keys, and selected device availability.
- Gives the model only a validated `ItemRecord` and its validated `StoreRecord`;
  `memory_off` receives no store. Store data reaches the model only in the
  serialized return produced by its selected read action.
- Executes the frozen read/return protocol for exactly 12 slots, caps reads at
  10, and pads post-HALT slots with canonical NOOP.
- Decodes the trained ` candidate=VALUE` response from model logits, requires
  exact leading-space/prefix framing, a canonical value, and a strict
  terminator. It strips the field framing before constructing `Submission`.
  The trained final frame is accepted only when `candidate` and `final` agree;
  malformed prefixes, whitespace suffixes, and mismatched finals fail closed.
- Never derives candidates from sealed gold or store rows and never
  teacher-forces `row.target` as candidate tokens. The anti-leak regression
  changes the returned target to `leak-sentinel` while the real checkpoint
  still emits exactly ` candidate=done` and submits only `done`.
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

After merging the exact hardening parent, the unchanged baseline was:

```text
136 passed in 55.60s
```

The new focused RED run had six expected failures: the real checkpoint
submitted `candidate=done`; three malformed/missing frames were accepted; the
anti-leak test saw the unparsed field; and incomplete readiness made 160 model
calls. The focused GREEN run was `6 passed in 9.75s`. A separate trained-final
frame regression failed first on invalid framing and then passed.

Final exact-checkout command:

```text
python -m pytest -q tests/test_confirmatory*.py
```

```text
142 passed in 77.96s
```

This comprises 40 runner tests and 102 hardening tests, including real hardened
report construction/publication, replay, trust, metrics, validation, contracts,
and inference fixtures. The real tiny GPT/checkpoint CLI test evaluates 160
items across both memory boundaries without mocking reporting.

Additional gates:

```text
ruff check: All checks passed
runner scope format check: 3 files already formatted
py_compile: passed
contract marker and hidden metric exports: passed
module and direct CLI smoke tests: passed
git diff --check: passed
```

## Concerns and limits

- CPU production construction and inference are exercised with a tiny real
  checkpoint. CUDA and MPS availability paths were not executable on this host.
- The frozen candidate grammar accepts canonical ASCII identifiers and the four
  slot markers used by the trained corpus. Expanding answer syntax requires a
  versioned protocol change and new checkpoint/evaluator tests.
- Atomic no-replace directory publication supports Darwin and Linux and fails
  closed on unsupported platforms.
