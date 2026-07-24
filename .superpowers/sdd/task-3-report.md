# Task 3 report: exact-version S3 publication and clean-room download

Date: 2026-07-24
Branch: `feat/aws-corpus-builder`
Exact base: `f2d91d0b7fb8dcf932e2afee190a0f9ebb90e905`
Status: `DONE`

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

## Review-fix addendum

This addendum supersedes the two integration concerns above. Task 3 runtime,
CLI, and test authorities are now admitted by Task 2 packaging, and the live
CLI dependency is declared in the authoritative `requirements.txt`.

### Commit and changed files

- Review fixes:
  `36d38777002183832c4243785bf8181149439920` —
  `fix: close exact publication review gaps`.
- Modified only `cluster/aws/corpus_builder/s3.py`,
  `scripts/aws_corpus_cleanroom_verify.py`,
  `tests/test_aws_corpus_builder_s3.py`,
  `cluster/aws/corpus_builder/package.py`,
  `tests/test_package_aws_corpus_builder.py`, and `requirements.txt`.
- This report is committed separately so it can record the implementation
  commit without amending it.

### Review-fix RED/GREEN evidence

Receipt namespace/KMS authority and GET-body closure RED:

```bash
python -m pytest -q \
  tests/test_aws_corpus_builder_s3.py::test_phase_receipt_rejects_key_build_id_before_any_s3_mutation \
  tests/test_aws_corpus_builder_s3.py::test_phase_receipt_rejects_kms_mismatch_before_any_s3_mutation \
  tests/test_aws_corpus_builder_s3.py::test_get_streaming_body_closes_on_every_authority_validation_failure
```

Exact RED result: `8 failed in 0.17s` (both publication cases did not raise,
and all six metadata-drift bodies remained open). After the fix, the same
cases plus the successful-download closure assertion reported
`9 passed in 0.08s`.

Pinned clean-room authority RED:

```bash
python -m pytest -q \
  tests/test_aws_corpus_builder_s3.py::test_cleanroom_cli_downloads_only_receipt_bound_versions_then_verifies \
  tests/test_aws_corpus_builder_s3.py::test_cleanroom_cli_rejects_symlinked_destination_ancestor_before_s3 \
  tests/test_aws_corpus_builder_s3.py::test_cleanroom_cli_rejects_non_final_receipt_before_objects_or_verifier \
  tests/test_aws_corpus_builder_s3.py::test_cleanroom_cli_rechecks_pinned_authority_after_verification
```

Exact RED result: `5 failed in 0.16s`. The first implementation run exposed a
test-only missing `stat` import (`1 failed, 4 passed in 0.10s`); after fixing
the harness, the exact command reported `5 passed in 0.09s`.

Package inventory/dependency RED:

```bash
python -m pytest -q \
  tests/test_package_aws_corpus_builder.py::test_package_allowlist_is_exact \
  tests/test_package_aws_corpus_builder.py::test_task3_runtime_cli_and_tests_are_required_reviewed_members \
  tests/test_package_aws_corpus_builder.py::test_live_cleanroom_dependency_is_declared_once
```

Exact RED result: `3 failed in 0.10s`. After admitting the Task 3 runtime,
CLI, and focused test authorities and declaring `boto3>=1.34`, the exact
command reported `3 passed in 0.09s`. The focused archive-membership test
reported `1 passed in 0.68s`.

### Final verification

Task 3 focused suite:

```bash
python -m pytest -q tests/test_aws_corpus_builder_s3.py
```

Exact result: `36 passed in 0.11s`.

Task 2 package suite:

```bash
python -m pytest -q tests/test_package_aws_corpus_builder.py
```

Exact result: `48 passed in 13.97s`.

Previously reported focused AWS/corpus regression:

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

Exact result: `410 passed in 26.03s`. As before, this local-only regression was
run outside the filesystem sandbox solely because its package tests create Git
repositories under the system temporary directory. No network or AWS path ran.

Static checks:

```bash
python -m py_compile \
  cluster/aws/corpus_builder/s3.py \
  cluster/aws/corpus_builder/package.py \
  scripts/aws_corpus_cleanroom_verify.py \
  scripts/package_aws_corpus_builder.py \
  tests/test_aws_corpus_builder_s3.py \
  tests/test_package_aws_corpus_builder.py
git diff --check
```

Both were silent with exit code zero.

### Added fake-S3 and filesystem mutation coverage

1. Receipt-key build ID mismatch is rejected with unchanged PUT/list counts.
2. Supplied KMS mismatch against the receipt objects' one shared KMS authority
   is rejected with unchanged PUT/list counts.
3. GET response drift in KMS, bytes, SHA-256, ETag, encryption, or version ID
   closes the returned streaming body; the successful stream closes too.
4. A non-final phase receipt performs only its own exact download and invokes
   neither corpus-object downloads nor the verifier.
5. A symlink in any destination ancestor is rejected before S3 reads.
6. Destination-root replacement and phase-receipt name replacement after
   parsing/verifying are detected by final descriptor/name binding checks.
7. The exact phase receipt remains owner-only in a sibling control directory,
   outside the corpus namespace.
8. Task 3 package membership, required focused-test authority, archive
   inclusion, and the live dependency declaration are covered directly.

### Review-fix self-review

- Canonical Task 1 validation completes before publication preflight; key build
  ID and the receipt objects' shared KMS ARN are checked before either LIST or
  PUT can execute.
- GET-body ownership is centralized: every response-authority failure closes
  the body, and both receipt-reuse and object-download success/error paths close
  it in `finally`.
- Clean-room creation walks and pins every real ancestor, exclusively creates
  and pins `0700` corpus/control directories, downloads through directory FDs,
  and never places the phase receipt in the corpus namespace.
- The phase receipt's authority descriptor is opened through the pinned control
  descriptor, parsed from that descriptor, retained, and rechecked after corpus
  verification for inode, size, SHA-256, exact name binding, and exclusive
  control contents. It is never reopened or removed through an unverified
  pathname.
- Task 2 now requires/packages the Task 3 runtime and CLI and requires the Task
  3 focused test without broadening the archive to all tests.
- The existing production-completion gate remains intentionally unchanged for
  Task 4. No other concern remains for Task 3.
- No AWS/network call, package-index access, push, amend, brief edit, progress
  edit, other-worktree edit, or unrelated source change was performed.

## Boto3 compatibility addendum

This addendum supersedes the earlier `boto3>=1.34` floor. The authoritative
dependency is now exactly `boto3>=1.43.54,<2`, whose S3 PutObject service model
exposes the `IfNoneMatch` request member required by no-overwrite publication.

Before any receipt LIST or PUT, `publish_phase_receipt` now obtains
`s3.meta.service_model.operation_model("PutObject")` and requires
`IfNoneMatch` in `input_shape.members`. A client without that service-model
surface, or with an older model lacking the member, fails closed with
`PublicationError` before S3 mutation. The protocol fake exposes an explicit
supported operation model; focused mutations remove the service model or the
member.

### Compatibility RED/GREEN evidence

Command:

```bash
python -m pytest -q \
  tests/test_aws_corpus_builder_s3.py::test_phase_receipt_requires_if_none_match_operation_model_before_mutation \
  tests/test_package_aws_corpus_builder.py::test_live_cleanroom_dependency_is_declared_once
```

Exact RED result: `3 failed in 0.20s`. Both unsupported-client cases performed
receipt publication instead of raising, and the dependency assertion observed
the stale `boto3>=1.34` declaration.

After adding the runtime guard, explicit fake service model, and exact bounded
dependency, the same command reported `3 passed in 0.09s`. Both capability
failures assert unchanged PUT and LIST call counts.

### Compatibility final verification

```bash
python -m pytest -q tests/test_aws_corpus_builder_s3.py
```

Exact result: `38 passed in 0.11s`.

```bash
python -m pytest -q tests/test_package_aws_corpus_builder.py
```

Exact result: `48 passed in 14.32s`.

```bash
python -m py_compile \
  cluster/aws/corpus_builder/s3.py \
  tests/test_aws_corpus_builder_s3.py \
  tests/test_package_aws_corpus_builder.py
git diff --check
```

Both static checks were silent with exit code zero. The package suite again ran
outside the filesystem sandbox only to create local Git repositories under the
system temporary directory. No AWS, network, package-index, push, or amend
operation ran.

### Compatibility scope and self-review

- Modified only `requirements.txt`, Task 3 S3/runtime tests, the Task 2
  dependency assertion/fixture, and this report.
- The capability check is unconditional for receipt publication, including
  exact-history reuse, because an unsupported client cannot safely guarantee
  the absent-key conditional publication branch.
- The existing build-ID and KMS preflight checks still precede all LIST/PUT
  mutations; the capability check follows them and also precedes all LIST/PUT
  mutations.
- The compatibility implementation and report are committed together; the
  resulting new commit hash is returned in the handoff without amending any
  prior commit.
