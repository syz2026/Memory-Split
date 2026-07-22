# Wikidata5M Real Hashmap Dataset Design

**Date:** 2026-07-22
**Status:** approved design; implementation plan pending written-spec review
**Deliverable:** `memorysplit-wikidata5m-real-hashmap-3000.zip`

## 1. Goal

Build a reproducible 3,000-key hashmap from Wikidata5M rather than from PopQA.
The archive must expose readable entity–relation keys while retaining Wikidata
QIDs and PIDs so homonyms cannot overwrite one another.

The prior `memorysplit-popqa-real-hashmap-3000.zip` is not an acceptable source:
PopQA is a QA benchmark derived from Wikidata tuples and Wikipedia popularity
metadata. It is not the pinned Wikidata5M graph selected for this deliverable.

## 2. Frozen source

Use the Hugging Face dataset `intfloat/wikidata5m` at revision:

`6b2b09672129e280c0c9da97ab58154e9d535e6b`

Only these locked archives are needed:

- `wikidata5m_alias.tar.gz`
  - 197,449,751 bytes
  - SHA-256
    `0330f580c9f7a57cbad949ac380835fdd2a2e14d96cc0f13fc435401d6b463a8`
- `wikidata5m_transductive.tar.gz`
  - 168,258,214 bytes
  - SHA-256
    `383160990b41c0905fc03f4a8afbb9b12be1ca3591e026bde6cdc94a59542597`

The combined download is 365,707,965 bytes. Download by exact revision and
reject either archive if its byte count or SHA-256 differs. Extract only regular
files through the repository's path-traversal-safe archive logic.

Wikidata5M is a pinned third-party derivative of Wikidata, not an official
Wikimedia dump. The README must state this distinction directly.
The packaged structured data is distributed under Wikidata's CC0 1.0 public
domain dedication; the archive must include the CC0 text and link to
`https://www.wikidata.org/wiki/Wikidata:Licensing`.

## 3. Input interpretation

Use the transductive training triples. Each accepted line has exactly:

`subject_QID<TAB>property_PID<TAB>object_QID`

Use the entity and relation alias files for display text. Wikidata5M aliases do
not carry language tags or identify a canonical label. For deterministic
display, use the first raw alias whose NFKC-normalized, whitespace-collapsed
form is non-empty, after within-ID duplicate removal. Preserve the raw alias's
capitalization.

QID/PID identity remains authoritative. A repeated label is safe because every
display key contains the corresponding IDs.

## 4. Relation selection

Compute statistics from the transductive training file and reproduce the
repository's frozen entity-relation selector:

1. require a non-empty relation alias;
2. require support of at least 5,000 triples;
3. require functionality
   `distinct_subjects / support >= 0.95`;
4. order by descending support, then descending exact functionality, then
   ascending numeric PID; and
5. retain the first 32 relations.

Do not add the repository's synthetic literal relations. Wikidata5M triples in
this source contain QID objects only.

## 5. Key and value semantics

Group every eligible source triple by canonical `(subject QID, property PID)`.
One grouped address is one hashmap key. Preserve every distinct object QID for
that address.

Display keys use:

`{subject label} [{QID}], {relation label} [{PID}]`

Example:

`Douglas Adams [Q42], occupation [P106]`

Each key maps to an array, even when there is one target:

```json
[
  {"id": "Q49757", "label": "poet"},
  {"id": "Q6625963", "label": "novelist"}
]
```

Sort values by numeric QID and remove duplicate source edges. A candidate key
is eligible only when its subject, relation, and every retained object have a
non-empty display alias. Record every exclusion count in the build manifest.

## 6. Deterministic 3,000-key sample

The output contains exactly 3,000 unique grouped keys, not exactly 3,000 source
triples. Array-valued keys may preserve more than 3,000 source edges.

Balance the sample across the selected relations:

- 94 keys from each of the first 24 selected relations; and
- 93 keys from each of the remaining 8 relations.

Within each relation, order eligible keys by SHA-256 of the UTF-8 canonical
address `QID<TAB>PID`, with numeric QID and PID as deterministic tie-breakers.
Take the relation's quota from that order. This avoids dependence on archive
line order and avoids a mutable random-number-generator implementation.

Fail the build if any selected relation cannot meet its quota after label
filtering.

## 7. Archive contents

The zip has one root directory,
`memorysplit-wikidata5m-real-hashmap-3000/`, containing:

- `hashmap.json`: one JSON object from display key to value array;
- `hashmap.jsonl`: streaming rows with `key` and `values`;
- `records.jsonl`: canonical subject, relation, values, and display key fields;
- `relation_summary.json`: selected relation IDs, labels, source statistics,
  quotas, and emitted counts;
- `build_manifest.json`: source revision, source hashes, algorithm version
  `wikidata5m-hashmap-v1`, filter counts, source-edge count, and output counts;
- `source/wikidata5m.lock.json`: the exact two-file source lock;
- `README.md`: schema, example, provenance limitations, and loading examples;
- `CITATION.bib`;
- `LICENSES/Wikidata-CC0-1.0.txt`; and
- `SHA256SUMS` for every other packaged file.

The primary `hashmap.json` and streaming `hashmap.jsonl` must encode the same
mapping byte-for-byte after JSON parsing.

Serialize text as UTF-8 with LF endings and a trailing newline. Use deterministic
JSON field ordering. Add zip entries in lexical path order with timestamp
`1980-01-01 00:00:00`, POSIX mode `0644`, and deflate level 9 so repeated builds
from the pinned inputs are byte-identical.

## 8. Failure handling

Abort without publishing a zip when:

- source revision, byte count, or SHA-256 is wrong;
- an archive contains an unsafe member;
- a triple or alias line violates its schema;
- fewer than 32 relations survive the frozen selector;
- a relation cannot supply its sample quota;
- display keys or canonical addresses collide unexpectedly; or
- packaged files disagree during round-trip validation.

Write to a temporary archive and atomically publish the final filename only
after all checks pass. Remove the previously generated PopQA zip only after the
replacement has passed validation.

## 9. Verification contract

Fresh verification must establish:

1. both source archives match the lock;
2. exactly 32 relations and 3,000 unique keys are emitted;
3. relation quotas are 94 for the first 24 and 93 for the last 8;
4. all canonical IDs and display labels are non-empty and well formed;
5. every emitted `(QID, PID, object QID)` edge occurs in the pinned source;
6. every target for each selected grouped key is preserved exactly once;
7. JSON, JSONL, and structured records reconstruct the same mapping;
8. all internal SHA-256 entries verify;
9. the zip has no absolute or parent-traversal paths; and
10. a fresh archive open and decompression test succeeds.

The final handoff reports archive size, archive SHA-256, key count, underlying
edge count, relation count, and the output path.
