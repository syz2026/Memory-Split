# FarmShare v2 corpus producer Task 4 report

## Result

- Status: `DONE_WITH_CONCERNS`.
- Branch: `feat/farmshare-v2-task-4-semantic`.
- Exact base: `0b7bd5657f99c0f4d6ec35671c507afacabc4623`.
- Implementation commit: `f21277c` (`feat: add v2 semantic sidecar routing`).
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

## TDD and verification

- Initial RED: semantic test collection failed on the absent Task 4 exports.
- Focused RED/GREEN: exact route-index winner tampering and post-open SQLite
  cleanup each failed for the intended reason before their fixes, then passed.
- Exact-base regression baseline: `158 passed in 16.57s`.
- Semantic suite: `42 passed in 0.43s`.
- Required semantic/catalog/foundations/routing/parallel matrix:
  `200 passed in 11.70s`.
- `py_compile`, Ruff, and `git diff --check` passed.

## Concern

- No production-scale 7.12B-target route materialization was run; real-corpus
  generation was explicitly outside Task 4 scope.

Report: `.superpowers/sdd/task-4-report.md`.
