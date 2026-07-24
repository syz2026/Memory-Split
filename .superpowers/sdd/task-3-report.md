# FarmShare v2 corpus producer Task 3 report

## Outcome

- Status: `DONE`
- Branch: `feat/farmshare-v2-task-3-catalog`
- Exact base: `e50ba4ffff0ad440de13158f26cca2f4c832c9d1`
- Initial implementation:
  `49efcc6c4d14f3f37565ec4763459e6ed26f5a2f`
- Review-hardening implementation:
  `4c712bc5a3b728b84207a5e77bed673f7752b005`
- Scope: `corpusgen/reasoning_v2/catalog.py` and
  `tests/test_reasoning_v2_catalog.py`

Task 3 now builds a canonical, external-memory `InputCatalog` from the reviewed
`BuildGeometry`, `SourceLock`, content-addressed verified source root, and eight
lane sources. No Task 4 routing, renderer, cluster, AWS, evaluation, or corpus
materialization code was added.

## Implemented contract

- Represents the exact `balanced_record_lengths()` schedule with a constant-size
  sequence, consumes one distinct draft per requested length, and raises
  `LaneQuotaShortfall` without cycling finite lanes.
- Spools validated drafts, source keys, semantic facts, global generation
  seeds, source-file authority, and Wikidata edge coverage in SQLite, then emits
  `catalog.jsonl` in frozen lane order and bytewise lane-local source-key order.
  Record IDs commit to the canonical draft core plus target length, never the
  global ordinal.
- Writes a canonical `catalog-index.json` binding the source-lock SHA-256,
  catalog SHA-256, lane ordinal ranges, lane counts and target totals, and
  overall counts.
- Verifies the reviewed source tree before and after iterator consumption,
  requires the content-addressed staged path, binds every draft to a listed
  source file and byte hash, and requires an independent expected generator
  commit authority before source iteration. A lock cannot authenticate its own
  generator commitment.
- Uses exact per-lane source identities and exact Wikidata `train` split
  authority. Sealed/evaluation/validation/test/holdout locators are rejected;
  reserved terms can be overridden only by an exact listed path authorized for
  that reviewed source identity, never by an arbitrary `train*` component.
- Rejects missing, unknown, unstable, or identity-mismatched lane mappings;
  duplicate source keys and record IDs; duplicate/conflicting semantic facts;
  missing/aliased/reused generation seeds; seeds on non-generated lanes;
  non-NFC text; non-finite or malformed rational payloads; and noncanonical
  locator, flag, fact, or surface ordering.
- Requires the Wikidata graph source to expose an exact finite edge count and
  streaming edge authority. SQLite proves every authorized edge appears exactly
  once before the first revisit and rejects absent, partial, duplicate, extra,
  or all-revisit schedules.
- Publishes both catalog files as one owner-controlled private directory,
  verifies them through pinned descriptors, fsyncs files/directories, and uses
  atomic no-replace rename. Exact existing/racing winners are fully verified and
  reused; conflicting winners fail unchanged.
- Opens SQLite through a pinned-directory `fchdir` boundary, keeps the original
  file descriptor open, and verifies the spool inode before and after use.
  Namespace replacement fails closed.
- Removes pathname deletion authority. Successful spools and failed/losing
  exact-inode stages are atomically moved into the owner-controlled quarantine
  namespace and retained for offline cleanup; replacement objects are never
  unlinked or removed.

The package `__init__.py` was intentionally unchanged: Task 3 requires module
exports, while the reviewed Task 1 package API remains exactly its nine frozen
symbols.

## TDD and verification evidence

- RED: `python -m pytest -q tests/test_reasoning_v2_catalog.py` failed collection
  with `ImportError: cannot import name 'catalog'`.
- GREEN: catalog suite — `25 passed in 3.03s`.
- Review RED: independent commit authority produced `2 failed`; mandatory
  Wikidata/source-path/compact-length cases produced `9 failed`; publication,
  winner, quarantine, and pinned-spool cases produced `5 failed`.
- Review GREEN: hardened catalog suite — `41 passed in 19.38s`, including a
  2,000-edge bounded-memory build/replay proxy.
- Focused catalog/contract/source/parallel regression —
  `296 passed in 88.01s`.
- Exact-base pre-change focused baseline — `255 passed in 17.32s`.
- `python -m py_compile` passed for the catalog and focused test.
- `git diff --check` passed before both implementation commits.

## Concerns

None within Task 3 scope. Retained quarantine objects intentionally require
explicit offline maintenance and are never reclaimed by the catalog builder.
Per instruction, no real corpus was materialized; the full-scale producer
remains dependent on later lane-source/rendering tasks and FarmShare execution.
