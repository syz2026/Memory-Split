# FarmShare v2 corpus producer Task 3 report

## Outcome

- Status: `DONE`
- Branch: `feat/farmshare-v2-task-3-catalog`
- Exact base: `e50ba4ffff0ad440de13158f26cca2f4c832c9d1`
- Implementation commit: `49efcc6c4d14f3f37565ec4763459e6ed26f5a2f`
- Scope: `corpusgen/reasoning_v2/catalog.py` and
  `tests/test_reasoning_v2_catalog.py`

Task 3 now builds a canonical, external-memory `InputCatalog` from the reviewed
`BuildGeometry`, `SourceLock`, content-addressed verified source root, and eight
lane sources. No Task 4 routing, renderer, cluster, AWS, evaluation, or corpus
materialization code was added.

## Implemented contract

- Calls `balanced_record_lengths()` exactly once per lane, validates each
  length and exact quota sum, consumes one distinct draft per requested length,
  and raises `LaneQuotaShortfall` without cycling finite lanes.
- Spools validated drafts and uniqueness indexes in SQLite, then emits
  `catalog.jsonl` in frozen lane order and bytewise lane-local source-key order.
  Record IDs commit to the canonical draft core plus target length, never the
  global ordinal.
- Writes a canonical `catalog-index.json` binding the source-lock SHA-256,
  catalog SHA-256, lane ordinal ranges, lane counts and target totals, and
  overall counts.
- Verifies the reviewed source tree before and after iterator consumption,
  requires the content-addressed staged path, binds every draft to a listed
  source file and byte hash, and rejects sealed/evaluation/validation/test/
  holdout locators unless the listed path itself explicitly names training.
- Rejects missing, unknown, unstable, or identity-mismatched lane mappings;
  duplicate source keys and record IDs; duplicate/conflicting semantic facts;
  generated-seed reuse; non-NFC text; non-finite or malformed rational payloads;
  and noncanonical locator, flag, fact, or surface ordering.
- Enforces complete training-edge coverage before the canonical Wikidata
  revisit suffix, including exact comparison when the lane source exposes its
  finite training-edge key set.
- Publishes both catalog files as one owner-controlled private directory,
  verifies them through pinned descriptors, fsyncs files/directories, and uses
  atomic no-replace rename. Failures and losing races leave no partial final
  catalog and clean the verified private candidate.

The package `__init__.py` was intentionally unchanged: Task 3 requires module
exports, while the reviewed Task 1 package API remains exactly its nine frozen
symbols.

## TDD and verification evidence

- RED: `python -m pytest -q tests/test_reasoning_v2_catalog.py` failed collection
  with `ImportError: cannot import name 'catalog'`.
- GREEN: catalog suite — `25 passed in 3.03s`.
- Focused catalog/contract/source/parallel regression — `280 passed in 18.58s`.
- Exact-base pre-change focused baseline — `255 passed in 17.32s`.
- `python -m py_compile` passed for the catalog and focused test.
- `git diff --check` passed before the implementation commit.

## Concerns

None within Task 3 scope. Per instruction, no real corpus was materialized; the
full-scale producer remains dependent on later lane-source/rendering tasks and
FarmShare execution.
