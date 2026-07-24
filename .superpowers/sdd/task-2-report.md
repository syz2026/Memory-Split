# Task 2 report: deterministic closed-world builder package

Date: 2026-07-24
Branch: `feat/aws-corpus-builder`
Base: `98f43e2241ed5cd4dc45be581f178d8dbc744b2f`
Status: `DONE_WITH_CONCERNS`

## Commits

- Implementation:
  `59e9bdb9c2b6eea70f5f4fcdfc19464a6a82f097` —
  `feat: package deterministic AWS corpus builder`
- The report-only commit is returned in the task handoff. It cannot record its
  own hash without changing that hash.

## Files changed

- Added `cluster/aws/corpus_builder/package.py`.
- Added `scripts/package_aws_corpus_builder.py`.
- Added `tests/test_package_aws_corpus_builder.py`.
- Replaced the stale unrelated contents of
  `.superpowers/sdd/task-2-report.md` with this report.

No Task 1 file, corpus scientific file, progress file, other worktree, or AWS
resource was modified.

## Implemented contract

- Exposes the exact `PackageError`, frozen `BuilderPackage`, and
  `build_corpus_builder_package(source_root, output_dir)` API from the brief.
- Uses the exact `REQUIRED_PREFIXES` and `REQUIRED_FILES` tuples.
- Requires a real clean Git worktree root and a stable full `HEAD` commit before
  and after collection. Untracked files, tracked modifications, and a moving
  revision fail closed.
- Reads selected bytes and Git modes/object IDs from the committed tree through
  `ls-tree` and `cat-file`, never from uncommitted worktree files.
- Requires the checked-in builder contracts/profile, corpus adapters/catalog
  and publication code, reasoning-v2 catalog/source-lock code, frozen recipe
  and source locks, Wikidata notice/lock, both tokenizer assets, executable
  entry points, and focused test authorities.
- Excludes cache, output, checkpoint, and explicit pilot-artifact path
  components. Selected symlinks, submodules, non-blob modes, unsafe paths,
  credential paths, secret-like files, raw token/key material, and structured
  JSON/YAML secret fields are rejected.
- Emits sorted regular-file tar members with Git-derived normalized `0644` or
  `0755` modes, uid/gid zero, empty owner/group names, mtime zero, deterministic
  gzip metadata, and no directory members.
- Emits canonical compact JSON containing the source revision, archive
  identity, and each member's path, mode, byte count, Git object ID, and
  SHA-256. It also emits the conventional archive SHA-256 sidecar.
- The CLI accepts only the required source/output arguments and prints the
  canonical manifest exactly.

## TDD evidence

### Initial RED

Command:

```bash
python -m pytest -q tests/test_package_aws_corpus_builder.py
```

Exact result:

```text
ERROR collecting tests/test_package_aws_corpus_builder.py
ModuleNotFoundError: No module named 'cluster.aws.corpus_builder.package'
1 error in 0.37s
```

### Credential-path review RED

Command:

```bash
python -m pytest -q \
  'tests/test_package_aws_corpus_builder.py::test_builder_package_rejects_dirty_tree_secret_and_unreviewed_member[credential-path]'
```

Exact result before the path-level credential rejection:

```text
F                                                                        [100%]
Failed: DID NOT RAISE <class 'cluster.aws.corpus_builder.package.PackageError'>
1 failed in 0.39s
```

The same command after the fix reported:

```text
.                                                                        [100%]
1 passed in 0.35s
```

### GREEN

Command:

```bash
python -m pytest -q tests/test_package_aws_corpus_builder.py
```

Exact result:

```text
............................                                             [100%]
28 passed in 7.78s
```

The suite covers the exact allowlist, deterministic independent builds,
canonical manifest rows, tar/gzip metadata, normalized executable modes,
authority gates, dirty/untracked trees, tracked secrets, structured secrets,
credential paths, symlinks, non-repositories, internal outputs, and CLI output.

### Focused regression

Command:

```bash
python -m pytest -q \
  tests/test_package_aws_corpus_builder.py \
  tests/test_aws_corpus_builder_contracts.py \
  tests/test_parallel_corpus.py \
  tests/test_reasoning_v2_contracts_strict.py \
  tests/test_reasoning_v2_contracts.py \
  tests/test_reasoning_v2_catalog.py \
  tests/test_reasoning_v2_source_lock.py
```

Exact result:

```text
354 passed in 26.20s
```

Compilation and whitespace commands:

```bash
python -m py_compile \
  cluster/aws/corpus_builder/package.py \
  scripts/package_aws_corpus_builder.py \
  tests/test_package_aws_corpus_builder.py
git diff --check
```

Both were silent with exit code zero before the implementation commit.

## Deterministic committed-tree rebuild

Both builds used the clean implementation commit
`59e9bdb9c2b6eea70f5f4fcdfc19464a6a82f097`.

Commands:

```bash
python scripts/package_aws_corpus_builder.py \
  --source-root . \
  --output-dir /tmp/memorysplit-corpus-package-a
python scripts/package_aws_corpus_builder.py \
  --source-root . \
  --output-dir /tmp/memorysplit-corpus-package-b
cmp \
  /tmp/memorysplit-corpus-package-a/memorysplit-corpus-builder.tar.gz \
  /tmp/memorysplit-corpus-package-b/memorysplit-corpus-builder.tar.gz
cmp \
  /tmp/memorysplit-corpus-package-a/memorysplit-corpus-builder.manifest.json \
  /tmp/memorysplit-corpus-package-b/memorysplit-corpus-builder.manifest.json
cmp \
  /tmp/memorysplit-corpus-package-a/memorysplit-corpus-builder.tar.gz.sha256 \
  /tmp/memorysplit-corpus-package-b/memorysplit-corpus-builder.tar.gz.sha256
```

All three comparisons were silent with exit code zero. Exact shared archive
identity:

```text
sha256: ba4b29017a7cae6deb45b603a751b7c067bf2e7d06011f58faeb1fc6729b1941
bytes: 772286
regular-file members: 97
revision: 59e9bdb9c2b6eea70f5f4fcdfc19464a6a82f097
```

`shasum -a 256` printed the same hash for both archives, `wc -c` printed
`772286` for each, and `tar -tzf ... | wc -l` printed `97`.

## Complete-suite evidence

Command:

```bash
python -m pytest -q
```

Exact summary:

```text
3 failed, 1747 passed, 2 deselected, 1 warning in 267.82s (0:04:27)
```

Failures:

```text
tests/test_reasoning_v2_semantic.py::test_route_index_open_rejects_namespace_aba_to_different_sqlite_inode
tests/test_verify_cohort_releases.py::test_accepts_release_built_by_final_aws_packager
tests/test_verify_cohort_releases.py::test_runbook_is_dry_run_first_and_covers_complete_p5_lifecycle
```

An isolated rerun reported `2 failed, 1 passed in 1.49s`: the SQLite
namespace-ABA case passed, while both cohort-release integration tests failed
again. Their traces point to pre-existing semantic/P5 integration and runbook
files outside the Task 2 diff. Per the ownership boundary, they were not
changed.

## Self-review

- The archive is sourced only from committed blobs under the exact allowlist;
  unrelated tracked files and all untracked files cannot enter it.
- Member order and every tar/gzip metadata field named in the brief are
  explicitly normalized and verified before publication.
- Manifest member rows bind both Git identity and content SHA-256 and are
  emitted in archive order.
- Secret checks reuse the established P5 release patterns and add path-level
  credential rejection; tests prove both raw and structured rejection.
- Required authorities are checked independently from archive selection so
  focused tests must exist without broadening the package to `tests/`.
- Output is required outside the source worktree and each artifact is staged
  with exclusive creation, fsynced, and atomically replaced.
- No network call, AWS command, push, or amendment was performed.

## Concerns

1. The repository has no `pyproject.toml` in this branch or its history.
   `pyproject.toml` remains in the exact required-file allowlist and is packaged
   when committed, but the real 97-member package cannot contain a file absent
   from the committed tree. `requirements.txt` and both required scripts are
   present and gated.
2. Base commit `98f43e2` still exposes `UnsupportedProductionRenderer`; this
   deterministic package does not by itself satisfy the design's separate
   pre-launch requirement for completed production renderers. A later
   integration revision must replace that fail-closed path before AWS launch.
3. The complete repository suite is not green for the three out-of-scope
   failures recorded above. Task 2's 354-test focused regression is green.

## Review-fix addendum

This addendum supersedes concerns 1 and 2 above. `requirements.txt` is now the
sole dependency declaration, and the current branch is deliberately blocked by
the production-pipeline gate until the planned Task 4 completion surface
exists.

### Review-fix commit and scope

- Implementation and focused tests:
  `e3492b64a6f58ebf149cb83307c0192f938b930e` —
  `fix: close corpus package review gaps`.
- Modified only `cluster/aws/corpus_builder/package.py` and
  `tests/test_package_aws_corpus_builder.py` before this report addendum.
- No Task 1 file, brief, progress file, scientific corpus file, other worktree,
  AWS resource, prior commit, or remote branch was changed.

### Review RED evidence

Command:

```bash
python -m pytest -q tests/test_package_aws_corpus_builder.py
```

Exact pre-fix summary:

```text
17 failed, 26 passed in 12.69s
```

The failures exercised removal of `pyproject.toml`, case-normalized artifact
prefixes, missing production modules/symbols, the unsupported renderer, lazy
fetch protection, a committed foreign member, required completion authorities,
physical path aliases/ancestors, and both post-write rechecks.

The coded-error regression was also observed failing before the rendering
change:

```text
1 failed in 0.46s
```

It expected the rendered exception to begin with
`INCOMPLETE_PRODUCTION_PIPELINE:` but received only the uncoded detail.

### Review GREEN and regression evidence

Final Task 2 test command:

```bash
python -m pytest -q tests/test_package_aws_corpus_builder.py
```

Exact result:

```text
43 passed in 16.37s
```

Focused Task 2 regression:

```bash
python -m pytest -q \
  tests/test_package_aws_corpus_builder.py \
  tests/test_aws_corpus_builder_contracts.py \
  tests/test_parallel_corpus.py \
  tests/test_reasoning_v2_contracts_strict.py \
  tests/test_reasoning_v2_contracts.py \
  tests/test_reasoning_v2_catalog.py \
  tests/test_reasoning_v2_source_lock.py
```

Exact result:

```text
369 passed in 29.99s
```

These commands were silent with exit code zero:

```bash
python -m py_compile \
  cluster/aws/corpus_builder/package.py \
  scripts/package_aws_corpus_builder.py \
  tests/test_package_aws_corpus_builder.py
git diff --check
```

### Expected clean-branch gate

At clean commit `e3492b64a6f58ebf149cb83307c0192f938b930e`:

```bash
python scripts/package_aws_corpus_builder.py \
  --source-root . \
  --output-dir /tmp/memorysplit-corpus-package-review-gate-e3492b6
```

exited `1` with:

```text
PackageError: INCOMPLETE_PRODUCTION_PIPELINE: required production pipeline still exposes the unsupported renderer
```

This is the required current-branch result. No new real-tree package, byte
count, member count, or SHA-256 is claimed. The fixture repository contains the
planned Wikidata derived-view, production catalog-adapter, and renderer public
surface; its two independent builds remain byte-identical in the 43-test suite.
Task 9 owns the real-tree two-build identity check after Task 4 removes this
gate.

### Review-fix self-review

- `PackageError.code` and rendered error text identify
  `INCOMPLETE_PRODUCTION_PIPELINE`; the gate scans committed production Python
  blobs for the unsupported renderer before it accepts the exact required
  Wikidata view, catalog-adapter, and renderer symbols.
- `pyproject.toml` is absent from `REQUIRED_FILES`; no replacement dependency
  declaration was added.
- Every Git subprocess uses the centralized sanitized environment containing
  `GIT_NO_LAZY_FETCH=1`; a wrapper test observes all package Git calls.
- Artifact rejection is case-insensitive and recognizes bounded
  artifact/cache/checkpoint/output/pilot prefixes such as `Checkpoint-100` and
  `PILOT-v2`, without rejecting source or tokenizer files merely for `.bin`.
- Package selection is bound to an explicit reviewed member inventory. A
  committed `scripts/unreviewed.py` under an otherwise eligible prefix fails
  with `UNREVIEWED_PACKAGE_MEMBER`.
- Source and output are compared using physical paths in both ancestor
  directions. Output location and source cleanliness are rechecked after all
  three writes and before return.
- Existing sorted-path, normalized tar/gzip metadata, committed-blob reads,
  canonical manifest, checksum, secret scanning, and atomic writes remain
  covered.

### Remaining concerns

1. Task 4 must remove `UnsupportedProductionRenderer` and provide the exact
   reviewed Wikidata derived-view, `WikidataGraphCatalogSource`, and
   `WikidataGraphRenderer` surface before any real package can be emitted.
2. The three previously recorded out-of-scope full-suite failures were not
   rerun in this review-fix wave; the requested 369-test Task 2 matrix is green.
