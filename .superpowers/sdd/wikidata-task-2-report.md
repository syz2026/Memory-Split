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

## Residual integrity closure

Status: `DONE_WITH_CONCERNS`

Implementation commit:
`143820db38de02781657940aeb07f59c976359e8` —
`fix: seal Wikidata derived-view publication`.

### Findings addressed

1. Every finalized private file now has a `_PrivateFileIdentity` containing
   device, inode, kind, owner, exact mode, link count, byte size, `mtime_ns`,
   and `ctime_ns`, plus an expected SHA-256 in `_PrivateFileAuthority`.
   This covers decoded members, external-sort runs, streams, indexes, the
   receipt, and every other candidate file. Incomplete outputs remain in a
   separate creation-identity ledger and cannot enter a sealed candidate.
2. Member and run consumers reopen descriptor-relatively, bind the name to
   the expected open descriptor, hash that descriptor in bounded chunks, read
   from the same descriptor, and postcheck its full identity. Run unlink and
   merge-input checks occur while those descriptors remain open. Same-inode
   modify/restore ABA changes `ctime_ns` and is rejected even when bytes,
   length, mode, and `mtime_ns` are restored.
3. After logical candidate verification, the builder seals the exact root and
   child inventories, retained child-directory identities, and every private
   file identity/hash. Immediately before no-replace rename it rehashes and
   rechecks that sealed authority using retained descriptors.
4. The published root descriptor remains open. Postrename verification
   rechecks the sealed ledger before and after full logical verification. If
   the exact published inode drifts, it is atomically moved from the final
   content-addressed name to a unique owner-only `.quarantine-*` sibling and
   the namespace is fsynced. A substituted concurrent winner fails the inode
   comparison and is never renamed or removed.
5. `_open_sorted_run`, `_merge_batch`, and outer build teardown use exhaustive
   close loops. Every descriptor is attempted after injected failures; a body
   error remains primary, while close failures are raised only after all
   close attempts or attached as diagnostic notes.
6. The added adversarial cases cover member and run ABA, full identity/hash
   ledger closure, stream/index/receipt drift after candidate verification,
   child-directory drift, exact-root quarantine with an absent final target,
   concurrent-winner preservation, and close-failure exhaustion in all three
   required paths.

### Residual-fix TDD evidence

The 10 named tests were added before production changes; the parameterized
candidate-file test produces three cases, for 12 total cases.

Direct RED selection:

```bash
python -m pytest -q tests/test_reasoning_v2_wikidata_source.py \
  -k 'materialized_member_same_inode or run_in_place_modify_restore or \
private_file_authority or candidate_file_drift_after or \
candidate_child_directory_drift or postpublish_drift or \
postpublish_root_swap or open_sorted_run_close or merge_close_failures or \
outer_close_failures'
```

Exact RED result:

```text
FFFFFFFFFFFF                                                             [100%]
12 failed, 49 deselected in 1.26s
```

The member and run ABA attacks reported `DID NOT RAISE`; run identities still
had five fields and no hash; prepublish stream/index/receipt/child drift left
the final name occupied; postpublish hooks were absent; and injected close
failures were never invoked through an exhaustive close primitive.

The same direct selection after implementation:

```text
............                                                             [100%]
12 passed, 49 deselected in 0.67s
```

### Residual-fix final verification

```bash
python -m pytest -q tests/test_reasoning_v2_wikidata_source.py
```

```text
.............................................................            [100%]
61 passed in 2.11s
```

```bash
python -m pytest -q \
  tests/test_reasoning_v2_source_lock.py \
  tests/test_current_sources.py
```

```text
........................................................................ [ 64%]
.......................................                                  [100%]
111 passed in 14.37s
```

The regression command again used unrestricted local filesystem execution
only because its fixtures create temporary Git repositories. No network or
AWS access was enabled or used.

```bash
python -m py_compile \
  corpusgen/reasoning_v2/wikidata_source.py \
  tests/test_reasoning_v2_wikidata_source.py
git diff --check
```

Both static checks were silent with exit code zero.

### Residual-fix bounded-memory evidence

- All new hashing is descriptor-based and streams fixed 1 MiB chunks. It does
  not use unbounded `read()`, `read_bytes()`, or whole-artifact collections.
- The private ledger contains only the fixed candidate file inventory and
  bounded external-run metadata `(name, identity, sha256)`; payload bytes are
  never retained.
- External-sort chunk, record, pending-level, and merge-fan-in bounds remain
  unchanged. Forced one-record multi-level merges remain covered by the
  deterministic focused suite.
- Publication sealing performs repeated streaming passes for integrity, not
  in-memory materialization, so memory remains independent of corpus size.

### Residual-fix self-review

- Public Task 1/Task 2 signatures, receipt format/schema v1, deterministic
  stream/index bytes, numeric ordering, and content-address names are
  unchanged.
- Successful candidates cannot cross publication on creation identity alone:
  all files, children, inventory, receipt, and root are sealed and checked
  immediately before rename. Postrename checks use the retained root and child
  descriptors rather than accepting a path-only candidate.
- Quarantine first proves the final name still denotes the retained published
  root. Therefore candidate drift vacates the target, while a different inode
  at that name is preserved as a possible concurrent winner.
- Close-failure tests call the real close before injecting an exception, prove
  every expected descriptor was attempted, and prove each is closed.
- Commit scope before this appendix contained exactly the two authorized
  Wikidata code/test files. This appendix is the only report change. No amend,
  push, source mutation, AWS operation, network operation, or other worktree
  edit occurred.

### Residual concern

The unrelated pre-existing full-suite failures recorded earlier remain outside
this task's ownership. All requested focused, Task 1/source-lock, compilation,
and diff checks are green.

## Final review closure

Status: `DONE_WITH_CONCERNS`

Implementation commit:
`df5f60bcfffb7cccd1f6f5955c4945aafdef7548` —
`fix: make Wikidata publication transactional`.

### Final findings addressed

1. Initial-run and merge-output writers now accumulate expected byte count and
   SHA-256 directly from each successfully written header, key, and payload.
   `_finalize_run` hashes the pinned output descriptor and requires those
   observed values to equal the independently accumulated expectation before
   registering `_SortRun` authority. It never promotes an observed post-write
   hash into the expected authority.
2. Publication allocates and retains an empty mode-0700 quarantine marker
   directory and descriptor before rename. Failure quarantine atomically
   exchanges the final and marker names with descriptor-relative Linux
   `renameat2(RENAME_EXCHANGE)`; macOS tests use the equivalent
   descriptor-relative `renameatx_np(RENAME_SWAP)`.
3. After exchange, both names are bound to retained descriptors: the quarantine
   name must be the exact published root and the final name must be the exact
   marker. A mismatch triggers a second atomic exchange, then verifies that the
   entry actually swapped out of the final name is restored and that the marker
   is back under its retained name. A race-substituted winner is preserved.
4. A correct exchange removes only the descriptor-bound empty marker at the
   final name and fsyncs the namespace, leaving the failed root under the
   owner-only quarantine name and the content-addressed final name absent.
5. Rename success, the first parent fsync, retained-root binding, full
   postpublication verification, and marker retirement are one guarded
   transaction. The terminal `published` state is set only after they succeed.
   Any prior failure, including the parent fsync, invokes exchange quarantine.
6. Initial-run writer teardown and all archive-authority teardown now use the
   exhaustive close helper. Every close is attempted, body errors remain
   primary, and close errors are surfaced only after all descriptors have been
   attempted or attached as notes.

### Final-review TDD evidence

Five named tests were added before production changes; the run-finalization
test is parameterized over initial and merge outputs, producing six cases.

Direct RED selection:

```bash
python -m pytest -q tests/test_reasoning_v2_wikidata_source.py \
  -k 'run_mutation_before_finalization or run_writer_close_failure or \
archive_teardown_attempts_all or postrename_fsync_failure or \
quarantine_exchange_race'
```

Exact RED result:

```text
FFFFFF                                                                   [100%]
6 failed, 61 deselected in 0.91s
```

Both pre-finalization mutations were accepted, the run-writer hooks did not
fire, archive close injection was bypassed, postrename fsync happened outside
the guarded hook window, and the pre-exchange winner substitution hook never
ran.

The same direct selection after implementation:

```text
......                                                                   [100%]
6 passed, 61 deselected in 0.44s
```

### Final-review verification

```bash
python -m pytest -q tests/test_reasoning_v2_wikidata_source.py
```

```text
...................................................................      [100%]
67 passed in 2.58s
```

```bash
python -m pytest -q \
  tests/test_reasoning_v2_source_lock.py \
  tests/test_current_sources.py
```

```text
........................................................................ [ 64%]
.......................................                                  [100%]
111 passed in 14.55s
```

The regression command used unrestricted local filesystem execution only
because its fixtures create temporary Git repositories. No network or AWS
access was enabled or used.

```bash
python -m py_compile \
  corpusgen/reasoning_v2/wikidata_source.py \
  tests/test_reasoning_v2_wikidata_source.py
git diff --check
```

Both static checks were silent with exit code zero.

### Final-review bounded-memory and self-review

- Expected run authority is updated incrementally from bounded record parts;
  final verification still hashes in fixed 1 MiB chunks. No corpus-sized
  payload or run is retained in memory.
- External-sort chunk, record, pending-level, and merge-fan-in bounds and all
  deterministic stream, index, receipt, and content-address bytes are
  unchanged.
- Exchange quarantine has no check-then-rename decision: the atomic swap occurs
  first, retained descriptor identities adjudicate what moved, and a wrong
  swap is atomically reversed before reporting failure.
- Successful builds remove the still-empty retained marker; failed exact-root
  builds retain only the quarantined root; substituted-winner failures restore
  the winner and remove the marker without deleting either candidate.
- Public Task 1/Task 2 signatures and schema/format v1 remain unchanged.
- Commit scope before this appendix contained exactly the two authorized
  Wikidata implementation/test files. This appendix is the only report change.
  No amend, push, source mutation, AWS operation, network operation, or other
  worktree edit occurred.

### Final concern

The unrelated pre-existing full-suite failures recorded earlier remain outside
this task's ownership. All requested focused, source-lock, compilation, and
diff checks are green.

## Quarantine-race and duplicate-close closure

Status: `DONE_WITH_CONCERNS`

Implementation commit:
`a209ff7c34aa7b1f43fd3b56ca00a5f0fc06a7e4` —
`fix: close Wikidata quarantine race gaps`.

### Remaining findings addressed

1. The retained mode-0700 quarantine marker is now allocated and explicitly
   name-bound before the final sealed-candidate verification. The no-replace
   publication call follows that verification without an intervening
   filesystem mutation.
2. Exchange quarantine name-binds both the final candidate and retained marker
   immediately before the atomic exchange. Afterward, it independently checks
   both names against the retained descriptors.
3. If the failed candidate reached `marker.name` but a substituted entry
   reached the final name, no rollback occurs: the failed candidate stays
   quarantined and the substituted/concurrent final entry is preserved.
4. If the candidate did not move while the exact marker reached the final
   name, rollback first opens and name-binds the exchanged wrong source, then
   atomically exchanges back and verifies both the restored source and marker.
   No other post-exchange state permits rollback.
5. `_open_sort_run_descriptor` now routes validation-failure closure through
   the exhaustive close helper. Archive envelope, parser, and materialization
   duplicates use shared helpers that safely cover duplicate, seek, fdopen,
   body, handle-close, and descriptor-close failures. Primary integrity/body
   errors remain primary; close failures surface only when there is no primary
   error or are attached as notes.

### Review-fix TDD evidence

Six named adversarial tests were written before the production changes.

Initial direct RED selection:

```bash
python -m pytest -q tests/test_reasoning_v2_wikidata_source.py \
  -k 'marker_is_bound_before or marker_substitution_keeps or \
sort_run_open_validation_close or archive_envelope_duplicate_close or \
archive_parser_fdopen_close or archive_materialization_fdopen_close'
```

Initial result:

```text
FF.FFF                                                                   [100%]
5 failed, 1 passed, 67 deselected in 0.68s
```

The five failures directly exposed late marker allocation, unconditional
rollback, and three direct archive closes. The provisional sort-run test passed
because a later descriptor reuse could satisfy its first close assertion. The
test was strengthened, still before production changes, to distinguish a
direct `os.close` of the exact opened inode from helper-mediated closure:

```text
F                                                                        [100%]
1 failed in 0.37s
```

Its expected failure was `assert not direct_close_seen`. After implementation,
the complete six-test selection was GREEN:

```text
......                                                                   [100%]
6 passed, 67 deselected in 0.42s
```

### Final verification

```bash
python -m pytest -q tests/test_reasoning_v2_wikidata_source.py
```

```text
........................................................................ [ 98%]
.                                                                        [100%]
73 passed in 2.61s
```

```bash
python -m pytest -q \
  tests/test_reasoning_v2_source_lock.py \
  tests/test_current_sources.py
```

```text
........................................................................ [ 64%]
.......................................                                  [100%]
111 passed in 14.73s
```

The regression command used unrestricted local filesystem execution only
because its fixtures create temporary Git repositories. No network or AWS
access was enabled or used.

```bash
python -m py_compile \
  corpusgen/reasoning_v2/wikidata_source.py \
  tests/test_reasoning_v2_wikidata_source.py
git diff --check
```

Both static checks were silent with exit code zero.

### Bounded-memory and self-review

- The final verification still streams descriptor hashing in fixed-size chunks;
  marker ordering and exchange adjudication retain only fixed-size identities.
  No corpus-sized read or collection was introduced.
- Archive duplicates remain streaming file objects. `closefd=False` leaves
  descriptor ownership with the exhaustive helper, so a handle-close failure
  cannot skip the underlying descriptor-close attempt.
- The marker-substitution test reproduces the reviewer sequence and checks all
  three inode outcomes: the failed root remains at the quarantine name, the
  substitute remains at the final name, and the displaced retained marker is
  not deleted through a stale name.
- The existing final-substitution race test covers the complementary matrix
  branch: only an exact marker-at-final state permits rollback, and the
  descriptor-bound exchanged winner is restored and verified.
- Task 1/Task 2 public interfaces, schema/format v1, external-sort bounds,
  deterministic stream/index/receipt bytes, and content-address naming are
  unchanged.
- The implementation commit contains exactly the two authorized Wikidata
  code/test files. This appendix is the only report change. No amend, push,
  source mutation, AWS operation, network operation, or other worktree edit
  occurred.

### Concern

The unrelated pre-existing full-suite failures recorded earlier remain outside
this task's ownership. All requested focused, source-lock, compilation, and
diff gates are green.

## Deterministic reusable marker-pool closure

Status: `DONE`

Implementation commit:
`7e4ad91fc9fb484371afcc2f57a12293afe33abb` —
`fix: bound Wikidata marker pool reuse`.

### Resource-bound behavior

1. Every publication operation now has exactly eight deterministic names:
   `.wikidata-marker-<full-receipt-sha256>-0` through `-7`. Production no
   longer creates random quarantine/orphan marker names.
2. A slot index is charged to the operation before its create/open attempt.
   Failed validation, unsafe entries, nonempty candidates, close failures, and
   markers whose pathname binding is later lost cannot make that slot
   available to another allocation or refresh in the same operation.
3. Existing slots are reusable only after no-follow descriptor binding,
   exact owner/mode/kind checks, empty inventory checks, and full
   device/inode/owner/mode/link-count/size/mtime/ctime identity checks before
   and after those inventory checks. Same-inode mutate/restore ABA therefore
   consumes the attempted slot and advances to the next deterministic slot.
4. Successful publication and verified-winner reuse leave the same empty
   marker in its slot. Repeated builds reuse its descriptor-pinned shape and
   do not grow the namespace inventory.
5. Before the no-replace publication, the builder proves an unattempted pool
   slot is absent, performs the final sealed-candidate verification, proves
   detach capacity again, and then publishes without an intervening namespace
   mutation.
6. Failed candidates remain as nonreusable evidence in their pool slot. The
   exact marker exchanged into final is moved, with the reviewed
   no-replace/inspect/restore protocol, into another deterministic slot.
   Seven failed candidates therefore leave seven evidence slots plus one
   reusable empty marker. An eighth failure attempt is rejected before
   publication, with final absent and all descriptors closed.

### TDD evidence

The reusable-inventory, lost-attempt, and seven-plus-one capacity tests were
added before the pool implementation.

Direct RED command:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_wikidata_source.py::test_no_replace_publication_reuses_only_a_fully_verified_winner \
  tests/test_reasoning_v2_wikidata_source.py::test_lost_marker_attempts_exhaust_deterministic_pool \
  tests/test_reasoning_v2_wikidata_source.py::test_seven_failed_candidates_block_eighth_before_publication
```

Exact RED result:

```text
FFF                                                                      [100%]
3 failed in 1.39s
```

The failures showed no deterministic slot after normal success, no retained
pool authority on an allocated marker, and no bounded pool inventory after the
first forced quarantine.

The same selection after implementation:

```text
...                                                                      [100%]
3 passed in 0.61s
```

Self-review then added a same-inode marker-shape ABA test before strengthening
the reusable-slot identity checks.

```bash
python -m pytest -q \
  tests/test_reasoning_v2_wikidata_source.py::test_reusable_marker_slot_rejects_same_inode_aba
```

Exact RED result:

```text
F                                                                        [100%]
1 failed in 0.53s
```

The old shape checks reused slot zero after a temporary child was created and
removed through the pinned directory descriptor.

The same test after identity hardening:

```text
.                                                                        [100%]
1 passed in 0.32s
```

### Final verification

```bash
python -m pytest -q tests/test_reasoning_v2_wikidata_source.py
```

```text
........................................................................ [ 88%]
.........                                                                [100%]
81 passed in 2.83s
```

```bash
python -m pytest -q \
  tests/test_reasoning_v2_source_lock.py \
  tests/test_current_sources.py
```

```text
........................................................................ [ 64%]
.......................................                                  [100%]
111 passed in 15.07s
```

The regression fixtures were run outside the filesystem sandbox because they
create temporary Git repositories. No network or AWS access was enabled or
used.

```bash
python -m py_compile \
  corpusgen/reasoning_v2/wikidata_source.py \
  tests/test_reasoning_v2_wikidata_source.py
git diff --check
```

Both static checks were silent with exit code zero.

### Changed files and self-review

- `corpusgen/reasoning_v2/wikidata_source.py`
- `tests/test_reasoning_v2_wikidata_source.py`
- `.superpowers/sdd/wikidata-task-2-report.md` (this appendix only)
- The allocation-failure test proves all eight failed slot descriptors close
  and a second call creates no additional entry. The lost-authority test moves
  all eight exact markers out of their names and proves the operation still
  cannot allocate a ninth marker.
- The capacity test performs seven independent postpublication failures,
  proves after each that no failed candidate occupies final, observes exactly
  one reusable empty marker, proves every returned marker descriptor closes,
  then proves attempt eight fails before the postpublication hook.
- All prior exchange pre-swap, marker substitution, concurrent-winner,
  final-detach substitution, close-failure, candidate-authority,
  external-sort boundedness, and deterministic-byte tests remain green.
- Marker state is bounded by eight slot indices and constant-size authority
  records; no corpus-sized read or collection was added. Task 1 interfaces and
  schema/format v1 are unchanged.
- The implementation commit contains exactly the authorized module/test
  files. This report is committed separately. No amend, push, source mutation,
  AWS operation, network operation, or other worktree edit occurred.

### Concerns

None.

## Identity-aware quarantine-marker detach closure

Status: `DONE_WITH_CONCERNS`

Implementation commit:
`ad300c7497a1f2c1d9421c5f1db4ab2bc7ebd858` —
`fix: detach Wikidata quarantine markers safely`.

### Remaining race addressed

1. Quarantine-marker cleanup no longer calls `rmdir` on a checked pathname.
   It atomically moves the current source-name entry to a unique
   `.orphan-marker-*` name with no-replace rename before deciding what moved.
2. If the moved entry matches the retained marker descriptor and creation
   identity, the parent is fsynced, the same descriptor/name binding is
   rechecked, and the owner-only marker remains retained at the orphan name.
   The final content-addressed name is therefore absent after exact detach.
3. If a substituted owner-only directory moved, the implementation opens and
   identity-pins that moved directory, restores it to the source name with
   no-replace rename, verifies the restored descriptor/name binding before and
   after parent fsync, closes the temporary descriptor exhaustively, and
   retries only a fixed number of times.
4. Fresh-marker allocation, post-allocation bind failure, retired-marker
   release, successful publication, winner reuse, and failed-candidate
   exchange all use the same no-delete detach path.
5. One shared `_QuarantineOrphanBudget` follows marker rotations and is
   installed before allocation validation. Cleanup may retain at most seven
   orphan markers and reserves the eighth slot for the marker exchanged out of
   the final name. Once the reserve is reached, no additional fresh marker is
   allocated; an exact entry that cannot be detached is left in place.

### TDD evidence

The final-name substitution test and the strengthened repeated fresh-bind
stress test were written before the detach implementation.

Direct RED command:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_wikidata_source.py::test_repeated_fresh_marker_bind_failures_keep_resources_bounded \
  tests/test_reasoning_v2_wikidata_source.py::test_marker_detach_restores_substitute_without_rmdir
```

Exact RED result:

```text
FF                                                                       [100%]
2 failed in 1.78s
```

The existing implementation called marker `rmdir` during repeated fresh-bind
cleanup, and the final-name race never reached an atomic detach, so its
substitution hook remained untriggered.

The same selection after implementation:

```text
..                                                                       [100%]
2 passed in 0.46s
```

Self-review then identified that allocation-stage validation failed before the
active state machine's budget could be attached. A separate shared-budget test
was added before that production correction.

Direct RED command:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_wikidata_source.py::test_quarantine_allocation_failures_share_orphan_budget
```

Exact RED result:

```text
F                                                                        [100%]
1 failed in 0.43s
```

The failure was the missing shared-budget argument:
`TypeError: _allocate_quarantine_marker() takes 2 positional arguments but 3 were given`.

The same test after implementation:

```text
.                                                                        [100%]
1 passed in 0.36s
```

### Final verification

```bash
python -m pytest -q tests/test_reasoning_v2_wikidata_source.py
```

```text
........................................................................ [ 92%]
......                                                                   [100%]
78 passed in 2.99s
```

```bash
python -m pytest -q \
  tests/test_reasoning_v2_source_lock.py \
  tests/test_current_sources.py
```

```text
........................................................................ [ 64%]
.......................................                                  [100%]
111 passed in 15.00s
```

The first sandboxed regression attempt produced 17 fixture setup failures
because macOS denied creation of temporary Git hook directories. The exact
command was rerun outside the filesystem sandbox and passed as shown above.
No network or AWS access was enabled or used.

```bash
python -m py_compile \
  corpusgen/reasoning_v2/wikidata_source.py \
  tests/test_reasoning_v2_wikidata_source.py
git diff --check
```

Both static checks were silent with exit code zero.

### Changed files and self-review

- `corpusgen/reasoning_v2/wikidata_source.py`
- `tests/test_reasoning_v2_wikidata_source.py`
- `.superpowers/sdd/wikidata-task-2-report.md` (this appendix only)
- Marker cleanup contains no pathname deletion. The remaining `rmdir` calls
  remove only descriptor-pinned private build/work directories and are outside
  the quarantine-marker protocol.
- The race test moves an empty replacement at the former check/use point,
  proves the replacement is restored at the final name, proves the failed
  candidate remains quarantined, proves every recorded marker descriptor is
  closed, and checks the retained orphan cap. The stress tests cover exact
  detach (final absent), repeated post-bind failures, allocation-stage
  failures, close-error preservation, and the seven-plus-one reserve.
- Detach state is constant-size and performs no corpus-sized reads or
  collections. External-sort bounds, deterministic output bytes, Task 1
  interfaces, and schema/format v1 are unchanged.
- The implementation commit contains exactly the authorized Wikidata
  module/test files. This report is committed separately. No amend, push,
  source mutation, AWS operation, network operation, or other worktree edit
  occurred.

### Concern

The eight-entry cap is per publication/quarantine state-machine invocation.
Across independent invocations, retained empty owner-only marker directories
intentionally persist because library-side deletion is forbidden by this
review closure; any eventual operator reclamation is outside this task.

## Two-name exchange state-machine closure

Status: `DONE_WITH_CONCERNS`

Implementation commit:
`c5bbbb87c63191130b01549dc1df90e2e996c05d` —
`fix: close Wikidata two-name quarantine race`.

### Remaining race addressed

1. Exchange quarantine now classifies the retained candidate and retained
   marker independently at both the final and quarantine names after every
   exchange. The state is four fixed booleans; no path-only observation is
   promoted to authority.
2. The reviewer sequence that swaps the exact candidate and exact marker before
   the production exchange is distinguishable as `candidate at final, marker at
   marker`. That state receives one descriptor-bound retry. The expected result
   is then `candidate at quarantine, marker at final`; only the exact empty
   marker is removed.
3. Repeating the restored-original state exhausts that marker's retry and
   allocates a freshly name-bound retained marker. The exact old marker is
   removed when still reachable, its descriptor is closed exhaustively, and
   the fresh marker becomes the bounded state machine's active authority.
4. A lost original marker while the candidate remains known at the final name
   takes the same fresh-marker transition. The state machine cannot return or
   raise from a classification that still binds the failed candidate to final.
5. A candidate already at quarantine with a non-marker final entry preserves
   that entry. Rollback remains restricted to the one-sided wrong-source state:
   marker at final, candidate at neither name, and marker absent from its own
   name.
6. Every terminal path fsyncs the namespace, reclassifies all four retained
   descriptor/name relations, and proceeds only when the candidate is no
   longer at final. Continuous hostile interference can delay completion, but
   state and descriptor usage remain bounded.

### State-machine TDD evidence

The exact pre-swap and repeated-state exhaustion tests were added before
production changes.

Direct RED selection:

```bash
python -m pytest -q tests/test_reasoning_v2_wikidata_source.py \
  -k 'two_name_preswap or repeated_original_state'
```

Exact RED result:

```text
FF                                                                       [100%]
2 failed, 73 deselected in 0.65s
```

The first failure showed that the failed candidate remained at the final inode
after the adversary's exact two-name pre-swap. The second showed that no fresh
marker was allocated after the repeated restored-original state.

The same selection after implementation:

```text
..                                                                       [100%]
2 passed, 73 deselected in 0.48s
```

### Final verification

```bash
python -m pytest -q tests/test_reasoning_v2_wikidata_source.py
```

```text
........................................................................ [ 96%]
...                                                                      [100%]
75 passed in 2.92s
```

```bash
python -m pytest -q \
  tests/test_reasoning_v2_source_lock.py \
  tests/test_current_sources.py
```

```text
........................................................................ [ 64%]
.......................................                                  [100%]
111 passed in 16.95s
```

The regression command used unrestricted local filesystem execution only
because its fixtures create temporary Git repositories. No network or AWS
access was enabled or used.

```bash
python -m py_compile \
  corpusgen/reasoning_v2/wikidata_source.py \
  tests/test_reasoning_v2_wikidata_source.py
git diff --check
```

Both static checks were silent with exit code zero.

### Bounded-memory and self-review

- The state machine retains four booleans, one small retry counter, the active
  marker authority, and at most one retired descriptor during marker rotation.
  It does not retain corpus records or add any corpus-sized read.
- Every retry rebinds the candidate and active marker immediately before the
  atomic exchange. Every exchange is followed by a four-location
  descriptor-based classification.
- The new pre-swap test proves the failed candidate leaves final and remains
  under a quarantine name. The exhaustion test forces the original state twice,
  proves a second marker is allocated, proves the old marker is retired, and
  proves exactly one quarantine entry remains with the candidate inode.
- All prior marker-substitution, final-substitution, fsync-failure,
  same-inode-ABA, close-exhaustion, bounded external-sort, and deterministic
  byte tests remain green.
- Task 1/Task 2 public interfaces and schema/format v1 are unchanged.
- The implementation commit contains exactly the two authorized Wikidata
  code/test files. This appendix is the only report change. No amend, push,
  source mutation, AWS operation, network operation, or other worktree edit
  occurred.

### Concern

The unrelated pre-existing full-suite failures recorded earlier remain outside
this task's ownership. All requested focused, source-lock, compilation, and
diff gates are green.

## Fresh-marker ownership closure

Status: `DONE_WITH_CONCERNS`

Implementation commit:
`812133ddfa1472707583407be9feb98ccf7f2efb` —
`fix: release failed Wikidata quarantine markers`.

### Remaining leak addressed

1. `_refresh_quarantine_marker` now owns each newly allocated marker locally
   until its post-allocation descriptor/name check passes and all active-marker
   fields have been transferred. A `finally` path releases every untransferred
   marker.
2. Release first attempts descriptor-bound exact removal. The existing retained
   descriptor and creation identity must still match the named entry, so a
   substituted entry is never removed. Successful exact removal fsyncs the
   namespace.
3. Descriptor closure is attempted exhaustively whether removal succeeds,
   fails, or detects substitution. Removal and close errors are secondary to
   the original allocation/bind validation error and are attached as notes.
4. Initial quarantine-marker allocation now applies the same owned-marker
   release discipline if its post-creation emptiness check fails.
5. Successful ownership transfer reuses the release helper for the retired
   marker. Repeated failed fresh-marker checks therefore retain only the active
   marker descriptor and active marker namespace entry.

### Fresh-marker TDD evidence

The repeated bind-failure stress test was added before production changes.
It forces four post-allocation bind failures, keeps the failed candidate at
final until a fresh marker succeeds, injects one close failure after the real
close, and records live retained-marker descriptors and named quarantine
entries after every failed retry.

Direct RED command:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_wikidata_source.py::test_repeated_fresh_marker_bind_failures_keep_resources_bounded
```

Exact RED result:

```text
F                                                                        [100%]
1 failed in 0.45s
```

The failing resource observations were
`[(2, 2), (3, 3), (4, 4), (5, 5)]`, proving one leaked descriptor and one
leaked namespace entry per retry instead of the required constant
`[(1, 1), (1, 1), (1, 1), (1, 1)]`.

The same test after implementation:

```text
.                                                                        [100%]
1 passed in 0.62s
```

### Final verification

```bash
python -m pytest -q tests/test_reasoning_v2_wikidata_source.py
```

```text
........................................................................ [ 94%]
....                                                                     [100%]
76 passed in 3.21s
```

```bash
python -m pytest -q \
  tests/test_reasoning_v2_source_lock.py \
  tests/test_current_sources.py
```

```text
........................................................................ [ 64%]
.......................................                                  [100%]
111 passed in 15.42s
```

The regression command used unrestricted local filesystem execution only
because its fixtures create temporary Git repositories. No network or AWS
access was enabled or used.

```bash
python -m py_compile \
  corpusgen/reasoning_v2/wikidata_source.py \
  tests/test_reasoning_v2_wikidata_source.py
git diff --check
```

Both static checks were silent with exit code zero.

### Bounded-resource and self-review

- Each failed retry now records exactly one live marker descriptor and one
  `.quarantine-*` entry. After completion, every allocated marker descriptor is
  closed and exactly one quarantine entry remains, bound to the failed
  candidate rather than a marker.
- The injected close error occurs only after the real descriptor close. The
  reported quarantine error remains the first fresh-marker bind `ValueError`;
  the close error does not replace it, and the original postpublication
  `RuntimeError` remains the build's primary exception.
- Exact-removal checks reuse the retained descriptor and creation identity.
  Identity mismatch fails before `rmdir`, closes only the retained descriptor,
  and preserves the substituted name.
- The four-location quarantine state machine, one-sided rollback restriction,
  fresh-marker retry bound, external-sort bounds, deterministic bytes, and all
  Task 1/Task 2 public interfaces and schema/format v1 remain unchanged.
- The implementation commit contains exactly the two authorized Wikidata
  code/test files. This appendix is the only report change. No amend, push,
  source mutation, AWS operation, network operation, or other worktree edit
  occurred.

### Concern

The unrelated pre-existing full-suite failures recorded earlier remain outside
this task's ownership. All requested focused, source-lock, compilation, and
diff gates are green.

## Final report supersession

The deterministic reusable marker-pool closure is the latest appendix in this
report. It supersedes older descriptions of random `.quarantine-*`,
`.orphan-marker-*`, exact-removal, and `rmdir` behavior above. The current
implementation uses only the fixed eight-slot `.wikidata-marker-*` pool:
seven retained failed candidates plus one reusable empty marker.
