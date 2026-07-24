# Task 3 report: exact-version S3 publication and clean-room download

Date: 2026-07-24
Branch: `feat/aws-corpus-builder`
Exact base: `f2d91d0b7fb8dcf932e2afee190a0f9ebb90e905`
Status: `DONE_WITH_INTEGRATION_CONCERNS`

## Commits

- Implementation:
  `3853cb4ed13360f8f6e058e0110d3171b83cd079` —
  `feat: publish versioned corpus artifacts exactly`.
- No-overwrite race hardening:
  `adee2294efc42624b87a0690e3f20aa849a66287` —
  `fix: prevent racing receipt overwrite`.
- The report-only commit is returned in the handoff. A commit cannot record its
  own hash without changing that hash.

## Files changed

- Added `cluster/aws/corpus_builder/s3.py`.
- Added `scripts/aws_corpus_cleanroom_verify.py`.
- Added `tests/test_aws_corpus_builder_s3.py`.
- Replaced the stale unrelated contents of
  `.superpowers/sdd/task-3-report.md` with this report.

No Task 1 contract, Task 2 package, progress file, brief, scientific corpus
file, other worktree, or AWS resource was modified.

## Implemented contract

- Reuses Task 1 `S3ObjectVersion`, `PhaseReceipt`,
  `s3_object_version_from_dict`, `phase_receipt_to_bytes`, and
  `phase_receipt_from_bytes`; no receipt or object contract is duplicated.
- Accepts the dynamically supplied KMS key ARN and validates it through the
  Task 1 contract. No KMS key ARN is hardcoded in production.
- Opens each local artifact with no-follow flags, verifies a positive regular
  file, hashes its pinned descriptor before upload, streams that descriptor
  with an exact `ContentLength`, and rehashes the same descriptor afterward.
- Uploads only with `ServerSideEncryption="aws:kms"`, the exact supplied key
  ARN, and SHA-256 object metadata. A non-empty S3 version ID is mandatory.
- Verifies every published or supplied authority with `head_object` against the
  exact bucket, key, version ID, byte count, normalized ETag, metadata SHA-256,
  encryption algorithm, and KMS key ARN.
- Lists the complete exact-key version history before receipt publication.
  Existing versions are streamed and compared byte-for-byte with canonical
  receipt bytes and exact metadata. Any conflict, duplicate version row, or
  delete marker fails closed without a PUT or delete.
- Uses `IfNoneMatch="*"` for an initially absent receipt key, preventing a
  concurrent current-version winner from being overwritten.
- Downloads only explicit version IDs. Each destination is created with
  `O_EXCL` and no-follow flags, streamed through bounded chunks, fsynced,
  checked for exact bytes and SHA-256, and followed by a second exact HEAD.
  Failure removes only the exact inode created by that call.
- The clean-room CLI accepts the seven required arguments, bootstraps the
  receipt authority from its explicit URI/version/SHA-256, downloads and parses
  the canonical phase receipt, requires one shared dynamic KMS key, maps every
  object safely beneath the build root, rejects foreign files and symlinks,
  checks the final namespace exactly, and calls
  `verify_parallel_corpus(destination, expected_build_id=...)`.
- Live `boto3` import is lazy. All Task 3 tests inject `VersionedFakeS3`; no
  test, verification command, or implementation step contacted AWS or the
  network.

## TDD evidence

### Initial exact-version RED

Command:

```bash
python -m pytest -q tests/test_aws_corpus_builder_s3.py
```

Exact result:

```text
ERROR collecting tests/test_aws_corpus_builder_s3.py
ModuleNotFoundError: No module named 'cluster.aws.corpus_builder.s3'
1 error in 0.13s
```

After the S3 module was added, the same command remained RED at the independent
CLI boundary:

```text
ERROR collecting tests/test_aws_corpus_builder_s3.py
ImportError: cannot import name 'aws_corpus_cleanroom_verify' from 'scripts'
1 error in 0.10s
```

### Review RED/GREEN cycles

- Pinned source namespace replacement initially exposed an over-strict ctime
  identity check: `1 failed, 20 passed in 0.13s`. Removing ctime (while keeping
  device, inode, mode, size, mtime, and pre/post hashes) produced
  `21 passed in 0.08s`.
- A fake GET stream interruption escaped as `FakeClientError`:
  `1 failed in 0.12s`. It was wrapped as `PublicationError`, the exact case
  passed in `0.05s`, and the suite reported `22 passed in 0.09s`.
- A receipt-reuse stream interruption likewise escaped:
  `1 failed in 0.09s`. The exact case passed in `0.04s`, and the suite reported
  `23 passed in 0.07s`.
- The upload-length assertion failed with missing `ContentLength`:
  `1 failed in 0.09s`. The exact case passed in `0.06s`, and the suite reported
  `23 passed in 0.10s`.
- The deterministic racing-winner fake showed receipt publication could PUT
  after an empty list: `1 failed in 0.09s` (`DID NOT RAISE`). Adding the
  conditional PUT made the exact case pass in `0.04s`; the suite reported
  `24 passed in 0.07s`.

### Final GREEN

Command:

```bash
python -m pytest -q tests/test_aws_corpus_builder_s3.py
```

Exact final result:

```text
........................                                                 [100%]
24 passed in 0.07s
```

Focused AWS/corpus regression:

```bash
python -m pytest -q \
  tests/test_aws_corpus_builder_s3.py \
  tests/test_package_aws_corpus_builder.py \
  tests/test_aws_corpus_builder_contracts.py \
  tests/test_parallel_corpus.py \
  tests/test_reasoning_v2_contracts_strict.py \
  tests/test_reasoning_v2_contracts.py \
  tests/test_reasoning_v2_catalog.py \
  tests/test_reasoning_v2_source_lock.py
```

Exact final result:

```text
393 passed in 25.59s
```

The first sandboxed regression attempt produced
`42 failed, 350 passed in 14.85s`; every failure was a local `git init`
receiving `Operation not permitted` under pytest's system temp directory. The
identical local-only command was rerun outside that filesystem sandbox and
first produced `392 passed in 27.34s`, then the final hardened count above.
Unsandboxing was solely for local temp-directory writes; no network-facing
command or AWS call was used.

Static checks:

```bash
python -m py_compile \
  cluster/aws/corpus_builder/s3.py \
  scripts/aws_corpus_cleanroom_verify.py
git diff --check
git diff --cached --check
```

All were silent with exit code zero. The cached check was run before each code
commit so the newly added files, not only tracked modifications, were covered.

## Fake-S3 mutation coverage

The 24-test suite covers:

1. missing PUT version ID;
2. source symlink rejection before S3 mutation;
3. pathname replacement after descriptor pinning;
4. missing exact S3 version;
5. wrong KMS key;
6. wrong byte count;
7. wrong metadata SHA-256;
8. wrong ETag;
9. wrong encryption algorithm;
10. corrupted GET bytes;
11. source authority mutation during GET, caught by the final HEAD;
12. interrupted artifact stream;
13. exact receipt reuse without another PUT;
14. conflicting receipt history alongside an exact older version;
15. delete-marker history;
16. interrupted receipt-reuse stream;
17. a concurrent conditional-PUT winner;
18. existing download destination/no overwrite;
19. foreign clean-room file;
20. clean-room symlink; and
21. phase-receipt/object KMS mismatch.

Every fake HEAD and GET assertion requires an explicit `VersionId`; fake
uploads assert streaming bodies rather than materialized artifact bytes.

## Self-review

- Public signatures match the Task 3 brief and consume Task 1 dataclasses.
- No object bytes or metadata are accepted from an unversioned read: all HEAD
  and GET calls include an exact version ID, and receipt discovery uses
  exact-key version history only to identify versions for exact verification.
- Generic artifact metadata is preserved and HEAD-checked; receipt metadata is
  exactly one canonical SHA-256 binding.
- Receipt publication has no delete authority and cannot overwrite a current
  racing winner.
- Clean-room path derivation rejects traversal, file/directory collisions,
  missing root `receipt.json`, foreign entries, symlinks, and special files.
- Failed exclusive downloads unlink only the still-bound inode they created.
- Imports, tests, and verification were inspected for network/AWS access; the
  only SDK construction path is the CLI's lazy live-client branch, which was
  never executed.
- Commit scope contains only the three Task 3 files plus this report.

## Integration concerns

1. Task 2's closed reviewed-member inventory predates these Task 3 files.
   Under the ownership boundary it was not edited here; the later integration
   task must admit `s3.py`, the clean-room script, and its focused test before a
   real deterministic package can include them.
2. `requirements.txt` does not currently declare `boto3`, and Task 3 was not
   allowed to edit it. Fake-client operation is dependency-free; live CLI use
   requires the later AWS runtime/package task to provide a compatible boto3.

No push, amend, AWS API call, or network call was performed.
