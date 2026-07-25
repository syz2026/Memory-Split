# Wikidata V2 Derived View Design

## Decision

Build a deterministic, content-addressed Wikidata derived view outside the
immutable v2 source root. The view is the missing authority boundary between
the three locked Wikidata archives and the production catalog/renderers.

This decision preserves:

- `configs/reasoning-dataset-v2.json`;
- `corpusgen/reasoning_v2/source_lock.py`;
- `configs/current-dataset-lock.json`;
- `sources/wikidata5m.lock.json`;
- the three fixed archive identities; and
- the legacy `corpusgen/current_sources.py` contract.

Direct per-record archive scans are rejected because compressed tar archives
cannot support production random lookup. Expanding the immutable source tree
is rejected because it would change the reviewed source-lock namespace and
conflate external authority with reproducible compiler output.

## Authority flow

```text
v2 SourceLock + exact staged archive bytes
  -> descriptor-pinned archive verification and decoding
  -> deterministic logical streams and indexes
  -> closed WikidataDerivedViewReceipt
  -> no-replace publication at <output-root>/wikidata/<receipt-sha256>
  -> verified read-only view
  -> production catalog adapter
  -> indexed renderer lookup
```

The source root remains byte-for-byte unchanged. A derived view is reusable
only after its receipt, source bindings, complete file inventory, logical
stream commitments, index commitments, and namespace identity all verify.

## Derived view layout

```text
<output-root>/wikidata/<receipt-sha256>/
  receipt.json
  members/
    <declared decoded members>
  streams/
    training.tsv
    aliases.tsv
    distinct-edges.tsv
  indexes/
    inductive-training-offsets.bin
    transductive-training-offsets.bin
    aliases.bin
```

All paths are relative POSIX paths. The receipt contains no timestamp,
hostname, absolute path, inode, or other machine-local value.

`training.tsv` is ordered first by source split (`inductive_train`,
`transductive_train`), then by one-based source row. Each canonical row is:

```text
<training_split>\t<row>\t<Q-subject>\t<P-relation>\t<Q-object>\n
```

Each training offset file contains one unsigned 64-bit big-endian byte offset
per row for its split.

`aliases.tsv` is ordered entities then relations, numerically by canonical ID.
Each canonical row preserves the complete public alias record using this exact
two-column format:

```text
<canonical-id>\t{"aliases":[<canonical JSON strings>],"display":<canonical JSON string>}\n
```

The object uses sorted keys and compact separators. `kind` is derived exactly
from the canonical `Q`/`P` prefix. No alternative scalar-string or open object
representation is accepted in schema version 1.

`aliases.bin` is a sequence of fixed-width 24-byte records sorted by
`(kind_rank, numeric_id)`: one unsigned 64-bit big-endian kind/id key, one
unsigned 64-bit byte offset, and one unsigned 64-bit byte length. Lookup uses
binary search and verifies that the selected TSV row matches the key.

`distinct-edges.tsv` is ordered numerically by
`(subject, relation, object)`. It records the first source provenance in fixed
split/row order and provides the complete-once authority consumed by the
catalog.

## Receipt

`WikidataDerivedViewReceipt` is a closed canonical JSON document with:

- format `memorysplit-reasoning-v2-wikidata-view-v1`;
- schema version `1`;
- source-lock SHA-256;
- generator Git commit;
- exact three archive path/byte/SHA-256 records;
- fixed archive-role and member-role mapping;
- complete decoded-member path/byte/SHA-256 inventory;
- training/sealed overlap-audit result;
- training, alias, and distinct-edge logical-stream byte/row/SHA-256 records;
- every index path/byte/SHA-256/count/record-width record;
- raw training-row, alias-row, and distinct-edge counts; and
- the exact derived-view relative file inventory.

The content address is SHA-256 of canonical `receipt.json` bytes. The receipt
does not contain its own hash.

## Build and verification

The builder:

1. Reopens and authenticates canonical `SourceLock` bytes.
2. Descriptor-pins the source root, Wikidata directory, and all three archive
   files with owner, mode, link-count, regular-file, and pre/post identity
   checks.
3. Hashes and parses the same archive descriptors.
4. Rejects absolute, traversing, Windows-drive, NUL, link, device, FIFO,
   sparse, duplicate, colliding, undeclared, truncated, and size-drifting
   members.
5. Opens and retains the pinned source and output-root descriptors, proves the
   output descriptor is neither the source descriptor nor its descendant, and
   writes only under an owner-controlled private sibling using `O_EXCL`,
   `O_NOFOLLOW`, fixed modes, fsync, and descriptor-relative operations.
6. Applies the existing reviewed TSV/QID/PID parsing and alias normalization.
7. Produces canonical streams and fixed-width indexes deterministically.
8. Computes and writes the closed receipt.
9. Re-verifies all source and derived bytes.
10. Publishes by no-replace rename to the receipt-hash path.

If a winner already exists, the builder verifies it completely before reuse.
No partial or conflicting winner is repaired in place.

External sort state, including SQLite, lives only inside the same pinned
private build directory. It is opened relative to that retained directory,
never through process-global temporary paths. Cleanup first quarantines the
exact private inode by no-replace descriptor-relative rename, verifies the
quarantined identity, removes only that authority, and fsyncs the namespace.

The verifier repeats source-lock, archive, receipt, inventory, stream, index,
namespace, and logical-row checks without trusting caller-constructed receipt
objects.

## Public interfaces

`corpusgen/reasoning_v2/wikidata_source.py` provides:

```python
@dataclass(frozen=True)
class WikidataDerivedViewRef:
    root: Path
    receipt_sha256: str


class WikidataDerivedView:
    """Context-managed verified descriptor session; no data-only authority."""

class WikidataDerivedViewReceipt: ...

def build_wikidata_derived_view(
    source_lock_path: Path,
    source_root: Path,
    output_root: Path,
    *,
    expected_generator_commit: str,
) -> WikidataDerivedViewRef: ...

def verify_wikidata_derived_view(
    source_lock_path: Path,
    source_root: Path,
    view_root: Path,
    *,
    expected_generator_commit: str,
) -> WikidataDerivedViewRef: ...

@contextmanager
def open_wikidata_derived_view(
    source_lock_path: Path,
    source_root: Path,
    view: WikidataDerivedViewRef,
    *,
    expected_generator_commit: str,
) -> Iterator[WikidataDerivedView]: ...

def iter_v2_training_triples(
    view: WikidataDerivedView,
) -> Iterator[V2TrainingTriple]: ...

def iter_v2_aliases(
    view: WikidataDerivedView,
) -> Iterator[V2AliasRecord]: ...

def iter_distinct_training_edges(
    view: WikidataDerivedView,
) -> Iterator[V2TrainingTriple]: ...

def lookup_training_triple(
    view: WikidataDerivedView,
    training_split: str,
    row: int,
) -> V2TrainingTriple: ...

def lookup_alias(
    view: WikidataDerivedView,
    canonical_id: str,
) -> V2AliasRecord | None: ...
```

`WikidataDerivedViewRef` is data-only and never authorizes a read. Public
iteration and lookup require a live `WikidataDerivedView` session returned by
`open_wikidata_derived_view`. The opener descriptor-pins source lock, source
root, view parent/name, receipt, streams, and indexes; verifies complete
receipt and source bindings; hashes the same opened files it retains; and
records owner/mode/link/identity facts. Each read uses those retained
descriptors and repeats post-read descriptor plus named-entry identity checks
before returning or yielding bytes. A caller-constructed object, imported
module-private token, mutable field, or path-only receipt cannot authorize
data. The session closes every descriptor deterministically.

## Catalog integration

`WikidataGraphCatalogSource` implements `LaneCatalogSource` against a verified
view. Its locators bind:

- `path`: locked inductive or transductive archive path;
- `member`: exact decoded training member;
- `split`: literal `train`;
- `training_split`: `inductive_train` or `transductive_train`;
- `row`: one-based source row;
- `training_edge_key`: canonical QID/PID/QID key; and
- `wikidata_view_sha256`: receipt commitment.

The catalog receipt binds the derived-view receipt. Before draft production,
the adapter compares distinct training-edge count with available Wikidata
records. It allocates at least one record per edge or fails with an explicit
capacity error; it never silently drops an edge.

## Renderer integration

`WikidataGraphRenderer` requires a verified `WikidataDerivedView`. It uses
indexed triple and alias lookups, verifies the locator and view commitment,
and never invokes legacy iterators or rescans full streams per record.

The existing shared record contracts, FineWeb renderer, FineMath renderer,
synthetic renderer, fitting logic, and semantic sidecar closure remain.

## Test requirements

Tests use tiny real tar archives and no monkeypatched Wikidata iterators.
They cover:

- deterministic repeated builds and worker/hash-order independence;
- complete source-root immutability;
- every archive path/member/type/collision/truncation attack;
- archive and parent-directory TOCTOU/ABA substitutions;
- malformed QID/PID/TSV and train/sealed overlap;
- exact stream ordering and alias ambiguity removal;
- stream/index lookup equivalence without archive rescans;
- no-replace publication and fully verified winner reuse;
- end-to-end view to catalog to renderer byte identity;
- edge-capacity rejection and complete-once ordering; and
- existing Task 2, Task 3, Task 5A, current-source, tokenizer, and SRGM
  focused regressions.

## Scope

This design does not materialize production archives, change scientific
quotas, launch training, alter the frozen source lock, or modify the legacy
current-source pipeline.
