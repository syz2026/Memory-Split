# FarmShare MemorySplit v2 Corpus Producer Design

**Date:** 2026-07-23  
**Status:** approved interactively; written review pending  
**Canonical site:** Stanford FarmShare  
**Canonical root:** `/scratch/users/syz/memorysplit-v2-corpus`  
**Source branch:** `feat/memorysplit-v2-integration@33b8c9f`

## 1. Goal

Produce and freeze the one canonical full MemorySplit v2 training corpus that
currently blocks all protected 360M and 135M runs.

The producer must publish:

- exactly `7,120,879,616` raw causal targets;
- the eight-lane reasoning-maximized mixture frozen in
  `configs/reasoning-dataset-v2.json`;
- one token stream shared byte-for-byte by Dense and Split90;
- binary `dense_target_weights` and `split90_target_weights` sidecars;
- a `memorysplit-parallel-corpus-v2` receipt;
- an outer dataset materialization receipt;
- a frozen source-lock manifest; and
- enough evidence for a byte-identical verified mirror on MIT storage.

No fixture, legacy seven-lane corpus, partial build, mutable source revision, or
smoke corpus may satisfy this contract.

## 2. Ownership and terminal condition

One dedicated **corpus-producer agent** owns the work from implementation
through FarmShare publication. Its terminal condition is not “the code builds.”
It is:

1. the full corpus is present under one immutable release directory;
2. both receipts pass local and independent verification;
3. every source and generated lane is frozen;
4. a second build produces the same logical stream and sidecar commitments;
5. the release is mirrored to MIT and reverified; and
6. the 135M and 360M packages can bind the receipt without modifying it.

The producer reports `BLOCKED` rather than weakening a source, quota, semantic
verification, or provenance requirement.

## 3. Canonical storage layout

FarmShare owns the authoritative materialization:

```text
/scratch/users/syz/memorysplit-v2-corpus/
  sources/
    <source-lock-sha256>/
  work/
    <run-nonce>/
  releases/
    <build-id>/
      receipt.json
      source-lock.json
      dataset/
        corpus-receipt.json
        shards/
          shard-00000.bin
          ...
          shard-00031.bin
        sidecars/
          dense_target_weights/
            shard-00000.bin
            ...
          split90_target_weights/
            shard-00000.bin
            ...
        manifests/
        proofs/
  verification/
    <build-id>.json
```

There is no mutable `current` symlink. Run manifests bind the explicit
`<build-id>` path and receipt SHA-256.

`receipt.json` is the outer materialization receipt used by dataset staging.
`dataset/corpus-receipt.json` is the inner
`memorysplit-parallel-corpus-v2` receipt consumed by training. The outer receipt
binds the inner receipt path and SHA-256.

## 4. Frozen corpus geometry

The producer copies these values exactly from
`configs/reasoning-dataset-v2.json`:

- total targets: `7,120,879,616`;
- targets per optimizer update: `524,288`;
- optimizer updates: `13,582`;
- context length: `1,024`;
- publication shards: `32`;
- allocation method: Hamilton largest remainder;
- tie break: frozen lane order.

The exact lane quotas are:

| Lane | Targets |
| --- | ---: |
| FineWeb-Edu | 1,780,219,904 |
| FineMath | 1,068,131,943 |
| Wikidata graph | 1,424,175,923 |
| Synthetic graph | 712,087,962 |
| Verified synthetic multihop | 1,068,131,942 |
| Wikidata path reasoning | 534,065,971 |
| Relational refinement | 178,021,990 |
| Objective auxiliary | 356,043,981 |

The reducer must reach every quota exactly. Cycling a finite reasoning lane to
fill a quota is forbidden. A lane must generate enough distinct deterministic
records or the build fails.

## 5. Source freeze

Before any production task starts, the producer writes and commits
`configs/reasoning-dataset-v2-source-lock.json`. The lock is strict JSON and
contains repository/revision, file inventory, byte count, SHA-256, license, and
materialized path for every external source.

Already frozen sources retain their existing identities:

- FineWeb-Edu repository `HuggingFaceFW/fineweb-edu`, revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, and the three locked 10BT sample
  parquet files;
- Wikidata5M repository `intfloat/wikidata5m`, revision
  `6b2b09672129e280c0c9da97ab58154e9d535e6b`, with the three archive SHA-256
  values from `sources/wikidata5m.lock.json`;
- ARC-AGI commit `399030444e0ab0cc8b4e199870fb20b863846f34`;
- ARC-AGI-2 commit `f3283f727488ad98fe575ea6a5ac981e4a188e49`;
- ConceptARC commit `0e67da6af879e4bad3d7cd3c196e8d551b445725`.

The source-freeze command resolves and records immutable identities for
FineMath, CLRS, RuleTaker, ProntoQA, and Reasoning Gym before publication. It
does not accept branch names, tags, `main`, `latest`, URLs without a digest, or
an ambient Hugging Face token. If any source cannot be immutably resolved,
licensed, downloaded, and hashed, the full build remains blocked.

Teacher-generated chain-of-thought contributes zero targets in this build.
This satisfies the frozen maximum while avoiding an unfrozen teacher model.

## 6. Compiler architecture

The implementation replaces `UnsupportedProductionRenderer` with a strict
production pipeline while preserving the existing miniature fixture path.

### 6.1 Source catalog

`corpusgen/reasoning_v2/catalog.py` produces a canonical `InputCatalog`.
Every record has:

- a globally unique deterministic record ID;
- lane and source IDs;
- immutable source key;
- source-byte SHA-256;
- deterministic ordinal;
- semantic flags; and
- no evaluation or sealed-test content.

Catalog order is independent of filesystem enumeration, worker count, download
order, and Python hash randomization.

### 6.2 Lane renderers

`corpusgen/reasoning_v2/renderers.py` implements one renderer per lane:

- FineWeb-Edu and FineMath render locked text records.
- Wikidata graph renders the complete frozen training graph once before any
  deterministic revisit permitted by the schedule.
- Synthetic graph uses frozen generator code and explicit seeds.
- Synthetic multihop and Wikidata path lanes emit canonical solver-verified
  proofs.
- Relational refinement emits canonical candidate-state transitions.
- Objective auxiliary emits exact-answer or solver-verified records from the
  locked source set.

All text uses the vendored tokenizer asset already bound by the repository.
Rendered records include token IDs, semantic payload spans, proof commitment,
and flags. The renderer refuses non-finite values, noncanonical Unicode,
unverified answers, duplicate IDs, or records exceeding context without a
frozen split rule.

### 6.3 Semantic routing and sidecars

The compiler reuses the reviewed routing and semantic-closure primitives.

- Dense is `uint8` value `1` for every logical target.
- Split90 is `uint8` and differs only at routed factual payload targets.
- Rules, operators, schemas, and proof procedures remain supervised.
- Every surface occurrence of an offloaded factual payload is masked.
- Split90 meets both frozen 90% dose thresholds.
- Candidate/final answer state cannot repeat routed factual surfaces.
- The exact target budget is update-aligned, so production publication requires
  `padding_tokens == 0`.

The route manifest, dose report, and semantic-leak report are bound into the
outer receipt. Any leak or dose failure invalidates the build.

### 6.4 Two-pass parallel build

Pass 1 renders canonical metadata and proof commitments in disjoint ordinal
partitions. Pass 2 applies the largest-deficit schedule and writes aligned token
and weight shard fragments. The coordinator concatenates fragments only in
frozen ordinal order.

FarmShare uses one 32-task array with at most eight concurrent tasks:

- partition `normal`;
- 16 CPUs per task;
- 64 GiB per task;
- 24-hour wall limit;
- node-local `SLURM_TMPDIR`;
- `--export=NONE` plus an explicit environment allowlist.

Task results publish from node-local storage into an owner-controlled staging
namespace. The final coordinator requires all 32 exact task receipts and
atomically publishes one no-replace release directory.

Worker count and completion order must not affect any receipt or output byte.

## 7. Receipts and verification

The inner receipt binds:

- compiler and renderer versions;
- canonical catalog, metadata, schedule, and assignment hashes;
- exact logical, packed, and padding target counts;
- 32 token shard paths, byte counts, and SHA-256 values;
- token ordered-stream and Merkle commitments;
- Dense and Split90 sidecar shard inventories and stream commitments;
- source-lock SHA-256;
- route-manifest SHA-256;
- proof-manifest SHA-256; and
- build ID.

The outer receipt binds:

- dataset ID `memorysplit-v2-20x-reasoning-max-cohort`;
- source Git commit and source-lock hash;
- inner receipt path and SHA-256;
- exact lane quotas and realized counts;
- graph coverage;
- semantic verification rate `1.0`;
- structural train/evaluation overlap `0.0`;
- route dose and leakage results;
- no-replace publication identity; and
- final scientific status.

Verification opens regular files with no symlink following, rehashes every
artifact, checks EOF and post-read identity, checks sidecar binary semantics,
recomputes all stream commitments, and replays a deterministic proof sample
from every solver-backed lane.

`protected_launch_allowed` becomes true only in the outer receipt after every
gate passes. A repository config cannot override this value.

## 8. FarmShare execution

The local operator first establishes the required Duo-authenticated SSH control
socket:

```bash
bash cluster/connect.sh syz rice-04.farmshare.stanford.edu
bash cluster/connect.sh syz dtn.farmshare.stanford.edu
```

The producer syncs the exact implementation commit to:

```text
/scratch/users/syz/memorysplit
```

It uses the existing environment:

```text
/scratch/users/syz/venvs/memorysplit/bin/python
```

and writes only under:

```text
/scratch/users/syz/memorysplit-v2-corpus
```

A small canary first builds `1,048,576` targets with all eight lanes, both
sidecars, all receipt layers, and deterministic double-build verification.
Only a green canary permits the 7.12B-target array.

The full array and final verifier are separate Slurm jobs. Training jobs never
depend directly on array completion; they depend on the final outer verification
receipt.

## 9. MIT mirror

After FarmShare verification, the producer copies the immutable release through
the FarmShare DTN to an operator-provided MIT storage root. The mirror process
copies into a private staging directory, verifies every hash, and atomically
renames the complete release.

MIT must produce a second verification receipt whose inner receipt bytes,
ordered token commitment, sidecar commitments, source-lock hash, and build ID
match FarmShare exactly. No MIT run may reference a partial or site-repacked
corpus.

## 10. Failure handling

The pipeline fails closed on:

- missing or mutable source identity;
- source byte/hash drift;
- unsupported license;
- duplicate or omitted catalog records;
- quota shortfall or reasoning-lane cycle fill;
- solver failure;
- semantic leakage or insufficient Split90 dose;
- token/sidecar length or binary-value mismatch;
- nonzero logical padding weight;
- missing task receipt;
- task replay disagreement;
- symlink, hardlink, path replacement, or output collision;
- receipt/hash disagreement;
- insufficient FarmShare storage; or
- interrupted publication.

Retries reuse content-addressed source and task artifacts only after their
receipts verify. No failed attempt mutates a published release.

## 11. Testing strategy

Implementation is test-first.

Local tests cover:

- strict source-lock parsing and immutable revision requirements;
- deterministic catalogs for all eight lanes;
- exact Hamilton quotas;
- independent serial/parallel and worker-order identity;
- all semantic routing/leakage/dose gates;
- proof verification and contamination rejection;
- 32-shard update alignment;
- token and sidecar stream identity;
- interrupted/resumed task arrays;
- no-replace publication and filesystem races;
- outer and inner receipt tampering;
- fixture/legacy corpus refusal; and
- FarmShare/MIT mirror identity.

Cluster acceptance adds:

- source staging and hash verification;
- 1,048,576-target canary;
- canary byte-identical rebuild;
- full 7.12B build;
- full independent verification;
- second full logical rebuild commitment match; and
- MIT mirror verification.

## 12. Capacity and schedule

The final packed tokens alone require approximately 13.27 GiB at `uint16`.
The two `uint8` target sidecars together add approximately 13.27 GiB. Source
snapshots, task caches, manifests, proof evidence, and a simultaneous
deterministic rebuild require substantially more. The producer refuses to
start the full array unless the canonical root reports at least 200 GiB free.

Expected elapsed time is several hours after source staging, not minutes. The
producer reports measured canary throughput and a revised completion estimate
before the full build. It does not promise same-day completion without that
measurement.

## 13. Deliverables

The corpus-producer branch delivers:

- the frozen v2 source lock;
- production catalog and renderers;
- deterministic sidecar generation;
- production task/finalization CLI;
- FarmShare build and verify Slurm scripts;
- mirror/verify tooling;
- tests and canary fixtures;
- an operator runbook; and
- the final FarmShare and MIT receipt paths after execution.

The multiseed package remains `BLOCKED_EXTERNAL_CORPUS` until the producer
publishes the final outer receipt.
