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

## Critical/Important review-fix closure

Status: `DONE_WITH_CONCERNS`

Implementation commit:
`4053676a4781b7f0ed976f4ca78a5e58b1c3eb13` —
`fix: harden Wikidata derived-view authority`.

This appendix supersedes the earlier statement that the private archive
authority's generator field was removed. `VerifiedArchiveSet` again retains
the pinned source lock's generator commit because it is security-relevant
authority, not dead state.

### Findings addressed

1. Source-lock authorization now receives the caller's independently supplied
   `expected_generator_commit`; it no longer authorizes a lock against its own
   commit. A descriptor-pinned source-lock precheck completes before creating
   the output root, and both archive opening and derived-tree verification
   require the retained commit to match.
2. The builder retains the private build descriptor plus stable creation
   identity `(device, inode, kind, owner, mode)`. It compares the sibling name
   and open descriptor immediately before descriptor-relative no-replace
   rename, then proves the published name is the same inode before content
   verification.
3. Every external-sort run is a `_SortRun(name, identity)`. Initial runs,
   merge inputs, merge outputs, final readers, and pre-unlink paths compare the
   named entry and open descriptor to that identity. Validation failures close
   every opened input/output descriptor.
4. Private construction maintains an exact ownership ledger for child
   directories and files. Cleanup first pins the original build and child
   directories, removes only ledger entries whose names and descriptors still
   match their creation identities, and never traverses a substituted
   directory. Cleanup and close failures cannot mask the primary build error;
   all retained descriptors are closed in the outer `finally`.
5. Verified public consumers retain `members`, `streams`, and `indexes`
   descriptors and identities, then postcheck all three after successful
   lookup or iteration in addition to the selected files, receipt, and root.
6. Adversarial coverage now includes mismatched and stale generator commits,
   prepublish build-directory substitution, run substitution during external
   reduction, all three child-directory substitutions, cleanup substitution,
   cleanup failure, descriptor closure, postrename target substitution, and
   final-target non-occupation after prepublish drift.
7. Stream formats, schema v1, deterministic ordering, external-memory
   reduction, bounded chunks/fan-in, and all Task 2 public signatures remain
   unchanged.

### Review-fix TDD evidence

The adversarial tests were added before production changes. Direct RED command:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_wikidata_source.py::test_build_rejects_mismatched_source_lock_generator_before_output \
  tests/test_reasoning_v2_wikidata_source.py::test_verify_rejects_stale_source_lock_generator_authority \
  tests/test_reasoning_v2_wikidata_source.py::test_build_directory_swap_fails_before_final_target_occupation \
  tests/test_reasoning_v2_wikidata_source.py::test_external_sort_run_swap_fails_closed_and_closes_opened_descriptor \
  tests/test_reasoning_v2_wikidata_source.py::test_public_consumers_postcheck_child_directory_identity \
  tests/test_reasoning_v2_wikidata_source.py::test_cleanup_never_deletes_substituted_private_build_directory \
  tests/test_reasoning_v2_wikidata_source.py::test_cleanup_failure_preserves_primary_error_and_closes_all_descriptors \
  tests/test_reasoning_v2_wikidata_source.py::test_published_target_inode_swap_fails_before_content_verification
```

Exact RED result:

```text
FFFFFFFFFF                                                               [100%]
10 failed in 0.91s
```

The generator mismatch was accepted, the stale lock failed only later as a
source-lock binding drift, and all hook-driven substitution/cleanup cases
reported `DID NOT RAISE`.

The same direct selection after implementation:

```text
..........                                                               [100%]
10 passed in 0.56s
```

### Review-fix final verification

```bash
python -m pytest -q tests/test_reasoning_v2_wikidata_source.py
```

```text
.................................................                        [100%]
49 passed in 1.71s
```

```bash
python -m pytest -q \
  tests/test_reasoning_v2_source_lock.py \
  tests/test_current_sources.py
```

```text
........................................................................ [ 64%]
.......................................                                  [100%]
111 passed in 16.68s
```

The regression command used local unrestricted filesystem execution only
because its fixtures create temporary Git repositories. No network or AWS
access was used.

```bash
python -m py_compile \
  corpusgen/reasoning_v2/wikidata_source.py \
  tests/test_reasoning_v2_wikidata_source.py
git diff --check
```

Both static checks were silent with exit code zero.

### Review-fix self-review

- Generator equality is established before any output path is created and
  re-established on the fully pinned archive authority; the receipt can no
  longer combine a stale lock with the requested generator.
- Creation identities intentionally exclude mutable size/timestamps while
  retaining device, inode, kind, owner, and exact mode. File hardlinks remain
  rejected independently.
- Sort-run identity is checked on every reopen and while the relevant
  descriptor is still open immediately before unlink. Merge inputs are all
  rechecked before the first input is removed.
- Cleanup validates the original build and every retained child descriptor
  before deleting any file. A same-UID replacement is left untouched, and
  primary exceptions survive cleanup failures with diagnostic notes only.
- Successful publication performs no path-based move: no-replace rename uses
  the already-open namespace descriptor, and the final name is bound back to
  the still-open build descriptor before full verification.
- The forced one-record, two-way multi-level merge remains covered by the
  focused suite, preserving deterministic bounded-memory behavior.
- Commit scope contains only the two Wikidata implementation/test files; this
  appendix is the only additional report change. No amend, push, source
  mutation, AWS operation, network operation, or other worktree edit occurred.

### Remaining concern

The pre-existing full-suite failures recorded above remain outside this task's
ownership. All requested Task 2 and source-lock regression checks are green.
