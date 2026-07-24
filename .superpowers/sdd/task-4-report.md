# FarmShare v2 corpus producer Task 4 report

## Result

- Status: `DONE_WITH_CONCERNS`.
- Branch: `feat/farmshare-v2-task-4-semantic`.
- Exact base: `0b7bd5657f99c0f4d6ec35671c507afacabc4623`.
- Implementation commit: `f21277c` (`feat: add v2 semantic sidecar routing`).
- Review-fix commit: `944ddb6` (`fix: close Task 4 semantic review findings`).
- Scope: `semantic.py`, semantic tests, and reviewed package exports only.

## Delivered behavior

- Bounded canonical external sort and descriptor-pinned SQLite reduce repeated
  fact occurrences, sum scheduled exposures, preserve exact `Fraction` keys,
  reproduce reviewed Split90 score/quota/burden repair, and emit one decision
  row per reduced fact.
- Route manifest and dose report carry exact rational commitments; the private
  read-only index rejects unsafe identities and replays identity around opens,
  queries, schema checks, and closes.
- Publication uses owner-controlled stages, no-replace rename, exact winner
  byte/content verification, and exact-inode quarantine. Production code has
  no `Path.resolve()` authority or pathname deletion.
- Dense/Split90 audits enforce ordered bounded spans, role-safe overlap, binary
  masks, payload-only zeroing, every declared routed occurrence, and
  answer/proof surface closure.
- Independent occurrence ledgers now bind reviewed character-level closure to
  exact token occurrences. Sidecar derivation rejects missing, extra, or
  mislabeled payload spans and records the plan/ledger hashes and counts.
- SQLite opens use absolute private paths without process-wide cwd changes.
  Descriptor, path, parent, `PRAGMA database_list`, and SQLite inode evidence
  reject file and parent-directory ABA replacement.
- Route artifacts bind index bytes and SHA-256 in the manifest; `open_index()`
  verifies that commitment before opening and requires the exact BINARY schema.
- Dose reports and semantic leak records are recursively immutable, and
  overlapping spans require a common non-null fact ID and compatible role.

## TDD and verification

- Initial RED: semantic test collection failed on the absent Task 4 exports.
- Focused RED/GREEN: exact route-index winner tampering and post-open SQLite
  cleanup each failed for the intended reason before their fixes, then passed.
- Review RED/GREEN cycles covered absent independent authority, omitted and
  mislabeled occurrences, cwd-based opens, file/parent ABA, creation hooks,
  missing index commitments, NOCASE/custom/index schema drift, null-fact
  overlap, and mutable nested evidence.
- Exact-base regression baseline: `158 passed in 16.57s`.
- Review-fixed semantic suite: `60 passed`.
- Required semantic/catalog/foundations/routing/parallel matrix:
  `218 passed in 21.24s`.
- `py_compile`, Ruff, and `git diff --check` passed.

## Concern

- No production-scale 7.12B-target route materialization was run; real-corpus
  generation was explicitly outside Task 4 scope.

Report: `.superpowers/sdd/task-4-report.md`.
