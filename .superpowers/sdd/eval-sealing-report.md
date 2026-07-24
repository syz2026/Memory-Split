# Sealed evaluation release report

- Status: `DONE`.
- Branch: `feat/sealed-eval-release-v3`.
- Exact base: `ed704e1fbf1f56e2384a520a18117934fe2a5948`
  (`feat/sealed-eval-foundation-v3`).
- Implementation commit:
  `dcf9501b4a67809b21ecfd6cc9a5d5b1b54b64de`.
- Frozen preregistration SHA-256:
  `6b2b5da3e3dc3d533498a0aa9d1891f356134ce045b1553b74a0161d94cb81d7`.

## Scope delivered

1. Added `evals/confirmatory/sealing.py` with one canonical v3
   `memorysplit.confirmatory.sealed-release.v3` manifest. Its SHA-256 is the
   release content address; it binds the frozen preregistration, exact
   model-visible item/store bytes and registries, complete family/stratum/
   memory/control coverage, and a separate sealed-gold commitment.
2. Preserved v2 item, store, and sealed-gold bytes verbatim. Sealing validates
   canonical JSONL, ordered unique identities and store rows, item/store/gold
   completeness, pair/twin/world closure, all ten frozen controls, and every
   gold proof/answer through two identical trusted-solver replays.
3. Added model-visible preflight that reads only the manifest, items, and
   stores. It checks sealed-gold metadata and returns only its SHA-256
   commitment; full verification opens and replays gold separately.
4. Added descriptor-pinned publication with no-follow opens, owner/mode/link
   checks, TOCTOU rechecks, per-file and directory fsync, private staging,
   atomic no-replace rename, collision refusal, rollback/quarantine cleanup,
   and exact post-install verification.
5. Added `scripts/seal_confirmatory_release.py`. Dry-run, publish, verify,
   help, and failure paths emit exactly one canonical JSON result. Existing
   releases are never overwritten or implicitly reused; explicit verify is
   the only accepted identical-rerun path.

## Strict TDD evidence

### RED

Before production code, the first focused run failed during collection:

```text
ModuleNotFoundError: No module named 'evals.confirmatory.sealing'
```

Subsequent red cycles established publication, verification/preflight, and
hardening behavior before each implementation:

```text
publication: 5 failed, 6 passed
verification: ImportError for preflight_model_visible_release
dry-run symlink output: 1 failed
counterfactual twin store closure: 1 failed
sealed-gold metadata isolation: 1 failed
filesystem-root output: 1 failed
CLI: ImportError for scripts.seal_confirmatory_release
```

### GREEN

```text
focused sealing tests
23 passed in 4.34s

bounded existing confirmatory v2/v3 regressions
204 passed in 70.44s
```

`python -m py_compile` passed for the sealing module, CLI, and focused tests.
`git diff --check` passed. The exact-base diff for
`evals/confirmatory/contracts.py` and `evals/confirmatory/fixtures.py` was
empty, preserving the existing v2 semantic contract and fixture sources.

## Scope audit

- Added only the sealing module, sealing CLI, focused sealing tests, and this
  report.
- Did not implement aggregation, step-aware `run.json`, study-lock creation,
  S3, lifecycle/msctl, provider selection, packages, IaC, corpus/config
  changes, or real evaluation data generation.
- Publication requires an existing safe output root and never creates or
  replaces the final content-addressed directory non-atomically.

## Concerns

None.

## Filesystem-race review addendum

- Status: `DONE`.
- Review base:
  `a979946e1b156da7eaa4f17fea46a6d8a05306e7`.
- Fix commit:
  `b87aa134bd5328e5b4ab76579353606167e4150c`.

### Findings closed

1. Publication, model-visible preflight, and full verification now hold the
   immediate parent and release directory descriptors, capture exact sorted
   membership plus directory device/inode/mode/owner/link/mtime/ctime state,
   and re-assert parent-name binding and the full snapshot as the final
   operation before returning.
2. Staging and installed-release cleanup now atomically rename only a
   descriptor-pinned directory to an unpredictable private quarantine.
   Recursive deletion descriptor-pins every member, atomically quarantines it,
   compares device/inode/type/mode/owner/link/size after rename, and verifies
   the open descriptor reflects the exact unlink. A mismatched replacement is
   restored or left untouched under a fail-closed error.
3. Deterministic mutation hooks cover insertion, removal, same-name inode
   replacement, and whole-directory swaps after source solver replay, after
   installation, during preflight/verification, and during quarantine cleanup.
   Directory snapshots detect same-name replacement through mtime/ctime even
   when the final name set is unchanged.
4. The CLI now emits `canonical_json_bytes` directly as UTF-8
   (`ensure_ascii=False`) and rejects non-canonical or non-finite values.

### Review RED to GREEN

```text
preflight/verify parent and membership races
RED: 8 failed
GREEN: 8 passed

post-install parent and membership races
RED: 4 failed
GREEN: 4 passed

quarantine directory/member identity races
RED: 4 failed
GREEN: 4 passed

cleanup membership and descriptor-close races
RED: 2 focused failures across the added cycles
GREEN: all focused cases passed

canonical non-ASCII CLI output
RED: 1 failed
GREEN: 1 passed

source membership after solver replay
RED: 3 failed
GREEN: 3 passed
```

Final verification:

```text
original 23 sealing tests plus adversarial race regressions
46 passed in 9.23s

bounded existing confirmatory v2/v3 regressions
204 passed in 149.22s
```

`python -m py_compile` passed for the sealing module, CLI, and focused tests.
`git diff --check` passed, and `contracts.py`/`fixtures.py` remain unchanged
from the review base. The review fix did not change the manifest, closure,
solver, gold-isolation, no-replace, or scope-exclusion contracts. There are no
remaining filesystem-race review concerns.
