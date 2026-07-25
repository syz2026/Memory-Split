# AWS Corpus Builder Task 4 Report

## Result

- Branch: `feat/aws-corpus-driver`
- Base: `cb57697b012ce1d8cfbd473c0f5b8c213b27e363`
- Implementation commit: `f65bbac5ed3d41e3fe9237bd03a330070d02b90b`
- No AWS or network calls were made. All S3 behavior in the driver tests used
  `VersionedFakeS3`.
- Owned implementation files only:
  - `cluster/aws/corpus_builder/driver.py`
  - `scripts/build_parallel_corpus.py`
  - `tests/test_aws_corpus_builder_driver.py`
  - `tests/test_parallel_corpus.py`

## Implemented behavior

The driver exposes the required `VerifiedProductionInputs`, `CommandRunner`,
`PHASES`, `CorpusBuildRequest`, `load_verified_production_inputs`, and
`run_corpus_build` interfaces.

The phase state machine:

1. derives each intermediate receipt key from the canonical request seed and
   the SHA-256 of the preceding canonical receipt;
2. treats an absent receipt as incomplete work, but fails closed on delete
   markers, conflicting histories, malformed metadata, or more than one
   receipt version;
3. downloads and parses the exact receipt version, verifies every referenced
   object by version ID, byte count, SHA-256, ETag, SSE algorithm, and KMS key,
   and verifies or restores the exact local phase inventory before reuse;
4. runs non-reused work through `CommandRunner.run`, checks canonical local
   paths, uploads all declared phase objects, and publishes that phase's
   canonical receipt last;
5. publishes every corpus object before the `s3-publish` receipt;
6. creates a new owner-only clean-room root, downloads every receipt-pinned
   corpus version, checks the exact namespace, calls `verify_parallel_corpus`,
   then rehashes the local files and re-HEADs every S3 version;
7. publishes the clean-room receipt only after that verification succeeds; and
8. re-verifies both publication receipts and all objects, requires exact object
   tuple equality, then publishes `receipts/final.json` as the last write.

The production CLI no longer accepts `--source`, `--catalog`, free-form lane
weights, or caller-selected sidecars. `build-production` now requires
`--source-lock`, `--source-root`, `--derived-root`, `--generator-commit`,
`--output`, `--update-tokens`, `--shards`, and `--workers`; it obtains the
catalog, renderer, and sidecars from the verified production factory.

## TDD evidence

RED command, with an explicit disposable pytest root:

```text
python -m pytest -q tests/test_aws_corpus_builder_driver.py \
  --basetemp=/tmp/memorysplit-aws-task4-red
```

Observed RED:

```text
ImportError: cannot import name 'driver' from 'cluster.aws.corpus_builder'
1 error during collection
```

The `/tmp/memorysplit-aws-task4-red` tree was deleted immediately afterward.

Focused GREEN:

```text
python -m pytest -q tests/test_aws_corpus_builder_driver.py \
  --basetemp=/tmp/memorysplit-aws-task4-driver-recheck
.......                                                                  [100%]
7 passed in 0.24s
```

Final producer regression:

```text
python -m pytest -q \
  tests/test_aws_corpus_builder_driver.py \
  tests/test_parallel_corpus.py \
  tests/test_reasoning_v2_wikidata_source.py \
  tests/test_reasoning_v2_catalog.py \
  tests/test_reasoning_v2_renderers.py \
  --basetemp=/tmp/memorysplit-aws-task4-final
........................................................................ [ 31%]
........................................................................ [ 62%]
........................................................................ [ 93%]
................                                                         [100%]
232 passed in 18.29s
```

The `/tmp/memorysplit-aws-task4-driver-recheck` and
`/tmp/memorysplit-aws-task4-final` trees were deleted immediately after their
runs.

Static verification:

```text
python -m py_compile \
  cluster/aws/corpus_builder/driver.py \
  scripts/build_parallel_corpus.py
# silent, exit 0

git diff --check
# silent, exit 0
```

## Resume and recovery reasoning

- The build ID alone is not used as sufficient reuse authority. Each phase
  receipt key commits to the exact preceding canonical receipt bytes, so a
  changed upstream object version changes every downstream expected key.
- An exact valid receipt is authoritative only with all of its exact object
  versions and its complete expected local inventory. Missing or drifted local
  bytes are never silently accepted.
- An invalid existing receipt is not overwritten or repaired. The driver stops
  before invoking the phase runner.
- Work interrupted before receipt publication has no phase authority. A retry
  runs the phase again and relies on deterministic local producers and
  fail-closed receipt publication.
- Work interrupted after a receipt is published can skip that phase only after
  receipt, S3 object, dependency-chain, and local inventory verification.
- A clean-room attempt interrupted before its receipt leaves no success
  authority. The next attempt allocates a different newly created empty root.
- An unpinned newer object version cannot influence verification: the
  clean-room test installs a conflicting latest shard and proves that the
  receipt-pinned older version is downloaded and verified.
- A clean-room object tuple with any changed version ID, byte count, digest,
  ETag, encryption value, KMS key, URI, or build ID is rejected before final
  receipt publication.

## Self-review

- Phase order is exactly the required seven-element `PHASES` tuple.
- Receipt publication is the last operation of every completed phase.
- Corpus object upload precedes the `s3-publish` receipt.
- Clean-room verification uses a newly created owner-only root and exact
  versioned downloads.
- Final publication is ordered after the clean-room phase receipt and checks
  exact dataclass equality of the publication and verification object sets.
- Receipt keys reject conflicting version history and delete markers.
- Runner-produced paths cannot escape the phase or corpus namespace.
- Local files are descriptor-opened with `O_NOFOLLOW`, hashed before upload,
  and checked for inode, size, timestamp, and link-count drift.
- The tests cover full phase order, exact resume, missing-version rejection,
  local dependency drift, latest-version confusion, clean-room disagreement,
  production-factory authority order, verified CLI arguments, and factory-owned
  sidecars.

## Integration concerns and required follow-up

1. At base `cb57697`, `corpusgen.parallel.adapters` still contains only
   `UnsupportedProductionRenderer`; it does not provide the completed
   reasoning-v2-to-parallel adapter required by Task 4. Because that file was
   explicitly outside this task's ownership, the driver keeps the dependency
   behind `_production_inputs_from_verified_view` and requires the adapter to
   expose:

   ```python
   verified_reasoning_v2_inputs(
       *,
       source_lock: SourceLock,
       source_root: Path,
       derived_root: Path,
       wikidata_view: WikidataDerivedView,
       expected_generator_commit: str,
   ) -> object_with_catalog_renderer_and_sidecars
   ```

   Until that integration lands, the verified factory raises
   `ProductionInputsUnavailable` rather than falling back to fixture or
   free-form production identity. This is the principal launch blocker.

2. The same base exposes production catalog/render support for the Wikidata
   path and four exposure renderers, but not a complete parallel adapter for all
   eight frozen lanes. The adapter above must own that composition, route-index
   construction, indexed render cache, and Dense/Split90 stream materialization.
   The driver's coupling to evolving `catalog.py` is intentionally limited to
   that one factory boundary, so the in-flight verified-view authority
   hardening remains a local integration edit.

3. The frozen `PhaseReceipt` contract has no `inputs` or dependency-receipt
   field. The implementation therefore binds dependencies through
   content-addressed receipt keys and enforces final ordering/object equality,
   but `receipts/final.json` cannot itself carry the clean-room receipt version.
   If an auditor must prove that link from final receipt bytes alone,
   `contracts.py` needs a reviewed schema revision. It was explicitly
   out-of-scope here.

4. `CorpusBuildRequest` has no source-object version manifest. The source phase
   consequently expects bootstrap to pre-populate the private
   `<work_root>/source-input` tree before running. Exact S3 source download
   authority must be supplied by the bootstrap/runbook layer or by a future
   request-contract revision.

5. `S3ObjectVersion` forbids zero-byte objects. The current driver therefore
   fails closed if a phase declares an empty file. If the final reviewed source
   lock contains required empty files, source-phase publication needs a
   deterministic archive/manifest representation before launch.

6. Task 2's package allowlist does not yet include `driver.py`, and the
   unsupported marker remains in the out-of-scope adapter module. Packaging is
   expected to update both after the production adapter integration.
