# Wikidata prerequisite Task 2 report

Date: 2026-07-24
Branch: `feat/aws-corpus-builder`
Base: `269d7773136ac88adf9331a1436a4fd764ee9338`
Status: `DONE_WITH_CONCERNS`

## Commits

- Implementation:
  `5c4d103e5c59bec7e93eb20a8c6e39d1a2b2ab64` —
  `feat: build bounded Wikidata derived views`.
- This report is committed separately so it can record the implementation
  commit. A commit cannot contain its own hash without changing that hash.

## Changed files

- Modified `corpusgen/reasoning_v2/wikidata_source.py`.
- Modified `tests/test_reasoning_v2_wikidata_source.py`.
- Added `.superpowers/sdd/wikidata-task-2-report.md`.

No source bytes, scientific config, brief, progress file, AWS resource,
network service, remote branch, or other worktree was modified. No push or
amend was performed.

## Implemented contract

- Added canonical training, alias, and distinct-edge streams; per-split
  unsigned 64-bit big-endian training offsets; and fixed-width 24-byte alias
  index records.
- Added closed `IndexArtifactRecord` receipt rows with exact count and record
  width while preserving receipt format and schema version 1.
- Added deterministic build, full source/view verification, verified streaming
  iterators, indexed training/alias lookup, and caller-constructed-view
  rejection.
- Added owner-only sibling construction, exclusive/no-follow file creation,
  file and directory fsync, prepublication source postcheck, SHA-256
  content-addressing, atomic no-replace publication, and full winner
  verification.
- Removed the write-only Task 1 archive-authority generator field. `struct` is
  now live production code for the exact binary index formats.

## Strict TDD evidence

The seven named Task 2 tests were written before production implementation and
run directly. Their two parameterized tests produced nine cases.

### RED

```bash
python -m pytest -q \
  tests/test_reasoning_v2_wikidata_source.py::test_repeated_builds_produce_identical_receipt_streams_and_indexes \
  tests/test_reasoning_v2_wikidata_source.py::test_training_and_alias_order_matches_frozen_contract \
  tests/test_reasoning_v2_wikidata_source.py::test_alias_ambiguity_is_removed_globally \
  tests/test_reasoning_v2_wikidata_source.py::test_indexed_lookup_matches_streaming_without_archive_rescan \
  tests/test_reasoning_v2_wikidata_source.py::test_train_sealed_overlap_and_malformed_rows_fail_before_publish \
  tests/test_reasoning_v2_wikidata_source.py::test_no_replace_publication_reuses_only_a_fully_verified_winner \
  tests/test_reasoning_v2_wikidata_source.py::test_stream_or_index_drift_fails_verification
```

Exact result:

```text
FFFFFFFFF                                                                [100%]
9 failed in 0.66s
```

Every case failed at the intended boundary:

```text
AttributeError: module 'corpusgen.reasoning_v2.wikidata_source'
has no attribute 'build_wikidata_derived_view'
```

### GREEN

The same direct selection after implementation:

```text
.........                                                                [100%]
9 passed in 0.63s
```

The repeated-build case was then tightened to force one-record/64-byte runs,
two-way fan-in, and multiple merge levels; it remained byte-identical:

```text
.                                                                        [100%]
1 passed in 0.35s
```

## Bounded-memory design evidence

- Archive envelopes decompress through a 1 MiB output cap per zlib call; PAX
  metadata is capped at 64 MiB. Member payload hashing and materialization use
  1 MiB reads and write each declared member directly to its exclusive output
  descriptor. No member payload is retained.
- Source rows are streamed with a 64 MiB explicit per-row bound. Control JSON
  is capped at 16 MiB. There is no unbounded `read()`, `read_bytes()`, or
  `read_text()` in production.
- `_ExternalSorter` admits at most 8 MiB or 65,536 records per memory chunk,
  caps one encoded record at 128 MiB, writes deterministic exclusive run
  names, and performs online binary run compaction. At most 65 pending run
  names exist; merge heaps retain one record per bounded fan-in input.
- Alias reduction is external: normalized occurrences are sorted by
  alias/owner/position, globally ambiguous owners are reduced away, survivors
  are externally re-sorted by canonical ID/position, and canonical JSON plus
  the alias index are streamed directly to final files.
- Training and sealed edges are externally sorted by numeric QID/PID/QID plus
  source provenance. The same ordered reduction detects overlap and emits the
  first-provenance distinct stream without an in-memory edge set.
- Training streams and both index families are emitted incrementally.
  Lookups use bounded `pread` calls for one fixed-width index record and one
  selected TSV row; they neither open source archives nor scan a logical
  stream.
- The deterministic test compares default chunking with forced multi-pass
  external sorting and obtains identical receipt, stream, and index bytes.

## Final verification

```bash
python -m pytest -q tests/test_reasoning_v2_wikidata_source.py
```

```text
.......................................                                  [100%]
39 passed in 2.06s
```

```bash
python -m pytest -q \
  tests/test_reasoning_v2_source_lock.py \
  tests/test_current_sources.py
```

```text
........................................................................ [ 64%]
.......................................                                  [100%]
111 passed in 15.16s
```

The second command was rerun outside the filesystem sandbox only because its
local fixtures create Git repositories in the system temporary directory. No
network or AWS access was enabled or used.

```bash
python -m py_compile corpusgen/reasoning_v2/wikidata_source.py
git diff --check
git diff --cached --check
```

All three static checks were silent with exit code zero.

## Self-review

- Public signatures, stream order, canonical bytes, index widths, numeric
  ordering, first provenance, global NFKC/whitespace/casefold ambiguity
  removal, and schema-v1 closure match the brief and binding design.
- Every public consumer reopens canonical receipt/namespace authority.
  Verification hashes every artifact and checks logical rows and indexes;
  lookup checks all file identities but reads only the selected index/row.
- Source descriptors remain pinned throughout construction. Their Task 1
  postcheck completes before the no-replace rename.
- Failure paths remove only the descriptor-relative private sibling they
  created. Published winners are never replaced or repaired.
- The source-root snapshot test proves byte identity across builds. Focused
  drift, malformed row, overlap, ambiguity, deterministic merge, unverified
  view, and corrupted-winner cases all pass.
- Commit scope before this report contained exactly the two authorized code
  and test files.

## Concern

The complete repository suite was already non-green on the exact base before
Task 2: `3 failed, 1747 passed, 2 deselected` in 267.82 seconds. The failures
were the pre-existing SQLite namespace-ABA semantic test and two P5
cohort-release/runbook tests, all outside Task 2 ownership. The requested
39-test Task 2 suite and 111-test focused regression are green.
