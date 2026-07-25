# Wikidata V2 Derived View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development to implement this plan task-by-task.
> Steps use checkbox syntax for tracking.

**Goal:** Add the missing verified decoded-source boundary from the three
locked Wikidata archives through production catalog and indexed renderer use.

**Architecture:** Preserve the immutable source root and derive one
content-addressed read-only view under a separate output root. Canonical
logical streams and fixed-width indexes are receipt-bound; catalog and
renderer code consume only a freshly verified view.

**Tech Stack:** Python 3.12, `tarfile`, descriptor-relative POSIX I/O,
canonical JSON/TSV, fixed-width big-endian binary indexes, pytest.

## Global Constraints

- The selected design is
  `docs/superpowers/specs/2026-07-24-wikidata-derived-view-design.md`.
- Do not modify the frozen recipe, source-lock schema, fixed archive hashes,
  immutable source-root namespace, or legacy current-source contract.
- No source download, production extraction, AWS operation, or broad suite.
- Every production behavior begins with a focused failing test.
- Canonical outputs contain no timestamp, hostname, absolute path, inode, or
  filesystem-enumeration order.
- All publication is private-build, fsync, full verification, and no-replace.
- Existing winners are reusable only after complete verification.
- Do not commit unless the controlling user explicitly requests a commit.

---

### Task 1: Closed receipt and descriptor-pinned archive authority

**Files:**
- Create: `corpusgen/reasoning_v2/wikidata_source.py`
- Create: `tests/test_reasoning_v2_wikidata_source.py`

**Interfaces:**
- Consumes: canonical v2 source-lock path, immutable source root, exact three
  Wikidata archive rows, expected 40-hex generator commit.
- Produces:

```python
@dataclass(frozen=True)
class V2TrainingTriple:
    training_split: str
    row: int
    subject: int
    relation: str
    object: int
    member: str
    archive_path: str


@dataclass(frozen=True)
class V2AliasRecord:
    canonical_id: str
    kind: str
    display: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class WikidataDerivedViewReceipt:
    format: str
    schema_version: int
    source_lock_sha256: str
    generator_commit: str
    archives: tuple[ArtifactRecord, ...]
    members: tuple[ArtifactRecord, ...]
    streams: tuple[ArtifactRecord, ...]
    indexes: tuple[IndexArtifactRecord, ...]
    training_rows: int
    alias_rows: int
    distinct_edges: int
    overlap_audit_passed: bool


@dataclass(frozen=True)
class IndexArtifactRecord:
    path: str
    bytes: int
    sha256: str
    count: int
    record_width: int


@dataclass(frozen=True)
class WikidataDerivedView:
    root: Path
    receipt_sha256: str
    receipt: WikidataDerivedViewReceipt
```

- Private authority object:

```python
@contextmanager
def _open_verified_archives(
    source_lock_path: Path,
    source_root: Path,
) -> Iterator[VerifiedArchiveSet]:
    ...
```

- [ ] **Step 1: Write receipt/parser RED tests**

Add tests named:

```python
def test_receipt_parser_rejects_open_missing_and_noncanonical_fields(): ...
def test_archive_authority_hashes_and_parses_the_same_descriptor(): ...
def test_archive_authority_rejects_path_inode_parent_and_aba_drift(): ...
def test_archive_authority_rejects_unsafe_or_undeclared_members(): ...
def test_archive_authority_leaves_source_root_byte_identical(): ...
```

Tiny tar fixtures must include the exact three locked archive roles while
using fixture-local lock hashes. Test hooks replace descriptors/namespaces
between precheck, parse, and postcheck.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_wikidata_source.py
```

Expected: collection fails because
`corpusgen.reasoning_v2.wikidata_source` is absent.

- [ ] **Step 3: Implement the closed data contracts**

Use exact receipt constants:

```python
RECEIPT_FORMAT = "memorysplit-reasoning-v2-wikidata-view-v1"
RECEIPT_SCHEMA_VERSION = 1
TRAINING_SPLITS = ("inductive_train", "transductive_train")
ARCHIVE_PATHS = (
    "wikidata5m_alias.tar.gz",
    "wikidata5m_inductive.tar.gz",
    "wikidata5m_transductive.tar.gz",
)
```

Canonical JSON uses UTF-8, sorted keys, compact separators, `allow_nan=False`,
and exactly one terminal newline. Public parsers reject duplicate JSON keys,
unknown fields, booleans where integers are expected, unsafe paths, invalid
hashes/commits, unsorted inventories, and inconsistent totals.

- [ ] **Step 4: Implement descriptor-pinned archive verification**

Open source-lock bytes, source directories, and archives with `O_NOFOLLOW`;
require owner-controlled regular files/directories, one link, fixed safe modes,
and stable pre/post descriptor plus namespace identity. Hash and parse each
same open archive descriptor. Reject absolute/traversing/Windows/NUL names,
links, devices, FIFOs, sparse members, duplicate outputs, file/directory
collisions, undeclared members, truncated payloads, and size/EOF mismatch.

- [ ] **Step 5: Verify Task 1 GREEN**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_wikidata_source.py
python -m pytest -q tests/test_reasoning_v2_source_lock.py tests/test_current_sources.py
python -m py_compile corpusgen/reasoning_v2/wikidata_source.py
git diff --check
```

Expected: all pass with no warning or diff-check output.

---

### Task 2: Canonical streams, indexes, and no-replace view publication

**Files:**
- Modify: `corpusgen/reasoning_v2/wikidata_source.py`
- Modify: `tests/test_reasoning_v2_wikidata_source.py`

**Interfaces:**

```python
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

- [ ] **Step 1: Write stream/index/publication RED tests**

Add:

```python
def test_repeated_builds_produce_identical_receipt_streams_and_indexes(): ...
def test_training_and_alias_order_matches_frozen_contract(): ...
def test_alias_ambiguity_is_removed_globally(): ...
def test_indexed_lookup_matches_streaming_without_archive_rescan(): ...
def test_train_sealed_overlap_and_malformed_rows_fail_before_publish(): ...
def test_no_replace_publication_reuses_only_a_fully_verified_winner(): ...
def test_stream_or_index_drift_fails_verification(): ...
```

- [ ] **Step 2: Verify RED**

Run the seven tests directly. Expected: absent build/verify/lookup APIs.

- [ ] **Step 3: Implement canonical logical streams**

Produce exact files and formats from the design:

- `streams/training.tsv`;
- `streams/aliases.tsv`;
- `streams/distinct-edges.tsv`;
- per-split unsigned 64-bit big-endian offset indexes; and
- fixed-width 24-byte alias index records.

Each aliases row is exactly
`<canonical-id>\t{"aliases":[...],"display":"..."}\n` using canonical JSON;
`kind` is derived from the `Q`/`P` prefix. Receipt index rows use
`IndexArtifactRecord` and bind each file's exact count and record width.

Training iteration is inductive then transductive, one-based row order.
Distinct edges sort numerically by QID/PID/QID and retain first source
provenance. Aliases sort entities then relations by numeric ID after reviewed
NFKC/whitespace/casefold global ambiguity removal.

- [ ] **Step 4: Implement transactional content-addressed publication**

Build under an owner-only private sibling. Write with `O_EXCL`/`O_NOFOLLOW`,
fsync files/directories, generate and verify `receipt.json`, compute its
SHA-256, then no-replace publish to
`<output_root>/wikidata/<receipt-sha256>`. Verify any concurrent winner in full.

- [ ] **Step 5: Implement verified streaming and indexed lookup**

`WikidataDerivedViewRef` is data-only. Every public read requires a live
context-managed `WikidataDerivedView` session returned by
`open_wikidata_derived_view`. The session retains pinned source, view,
receipt, stream, and index descriptors and identities. Lookup verifies
fixed-width index shape, ordering, uniqueness, offset/length bounds, and
selected TSV key/value bytes, then repeats descriptor and named-entry identity
checks after reading before returning. Iterators repeat the same checks before
each yield. No imported token or caller-constructed object authorizes data, and
no lookup opens an archive or scans an entire logical stream.

The output-root descriptor used for the source-disjointness proof remains open
through private build and publication. Temporary SQLite/external-sort state
lives inside the exact private build directory and follows the same
descriptor-bound quarantine cleanup as other private files.

- [ ] **Step 6: Verify Task 2 GREEN**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_wikidata_source.py
python -m pytest -q tests/test_reasoning_v2_source_lock.py tests/test_current_sources.py
python -m py_compile corpusgen/reasoning_v2/wikidata_source.py
git diff --check
```

Expected: all pass.

---

### Task 3: Production Wikidata catalog adapter and capacity gate

**Files:**
- Modify: `corpusgen/reasoning_v2/catalog.py`
- Modify: `tests/test_reasoning_v2_catalog.py`
- Modify: `tests/test_reasoning_v2_wikidata_source.py`

**Interfaces:**

```python
class WikidataGraphCatalogSource(LaneCatalogSource):
    lane_id: LaneId = "wikidata_graph"

    def __init__(self, view: WikidataDerivedView) -> None: ...
    @property
    def training_edge_count(self) -> int: ...
    def iter_training_edge_keys(self) -> Iterator[str]: ...
    def iter_drafts(self) -> Iterator[CatalogDraft]: ...
```

- [ ] **Step 1: Write production-adapter RED tests**

Use tiny real archives and the built view:

```python
def test_production_adapter_binds_view_archive_member_split_row_and_edge(): ...
def test_catalog_receipt_binds_verified_wikidata_view_sha256(): ...
def test_distinct_edges_appear_once_before_first_revisit(): ...
def test_edge_count_above_available_records_fails_before_draft_output(): ...
def test_end_to_end_catalog_is_byte_identical_across_rebuilds(): ...
```

- [ ] **Step 2: Verify RED**

Run the five tests. Expected: missing `WikidataGraphCatalogSource` and receipt
view binding.

- [ ] **Step 3: Implement adapter and locator authority**

Every Wikidata locator contains:

```python
(
    ("member", decoded_member),
    ("path", locked_archive_path),
    ("row", one_based_row),
    ("split", "train"),
    ("training_edge_key", edge_key),
    ("training_split", source_split),
    ("wikidata_view_sha256", view.receipt_sha256),
)
```

Keys remain canonical bytewise order. `source_byte_sha256` remains the locked
archive hash. Catalog metadata/receipt binds the view commitment.

- [ ] **Step 4: Implement capacity gate**

Before emitting drafts, compare verified distinct-edge count with allocated
Wikidata record count. Require at least one record per edge. If not, raise
`ValueError` containing `Wikidata distinct edges exceed allocated records`;
do not emit a partial catalog.

- [ ] **Step 5: Verify Task 3 GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_wikidata_source.py \
  tests/test_reasoning_v2_catalog.py \
  tests/test_reasoning_v2_contracts.py
git diff --check
```

Expected: all pass.

---

### Task 4: Indexed renderer integration and contract documentation

**Files:**
- Modify: `corpusgen/reasoning_v2/renderers.py`
- Modify: `tests/test_reasoning_v2_renderers.py`
- Modify: `docs/superpowers/specs/2026-07-23-farmshare-v2-corpus-producer-design.md`
- Modify: `docs/superpowers/plans/2026-07-23-farmshare-v2-corpus-producer.md`

**Interfaces:**

```python
class WikidataGraphRenderer(_ExposureRenderer):
    def __init__(
        self,
        source_root: Path,
        wikidata_view: WikidataDerivedView,
    ) -> None: ...
```

- [ ] **Step 1: Replace fixture-only Wikidata tests with RED integration tests**

Remove monkeypatched legacy iterators for Wikidata. Build tiny real archives,
stage/verify source authority, build the view, build the production catalog,
then render:

```python
def test_wikidata_archive_view_catalog_renderer_round_trip(): ...
def test_renderer_rejects_wrong_view_receipt_member_split_row_or_edge(): ...
def test_renderer_uses_indexed_lookups_without_full_stream_or_archive_scan(): ...
def test_end_to_end_rendered_bytes_repeat_across_view_and_catalog_rebuilds(): ...
```

- [ ] **Step 2: Verify RED**

Run the four tests. Expected: legacy iterator construction and locator parsing
cannot satisfy the new authority.

- [ ] **Step 3: Implement verified indexed rendering**

Remove legacy Wikidata iterator imports and full-stream scans. Require a
verified view, `split="train"`, separate `training_split`, exact member/archive
binding, exact view SHA-256, indexed triple lookup, and indexed aliases.
Reverify archive and view authority before and after rendering.

- [ ] **Step 4: Amend existing producer design and plan**

Document the derived-view authority between source staging and catalog, bind
its receipt in catalog/outer receipts, add Task 3W before Task 5A, and state
that Task 5A's Wikidata path consumes indexed view lookups. Do not change
scientific quotas or fixed source identities.

- [ ] **Step 5: Verify Task 4 GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_wikidata_source.py \
  tests/test_reasoning_v2_catalog.py \
  tests/test_reasoning_v2_renderers.py \
  tests/test_reasoning_v2_semantic.py \
  tests/test_reasoning_v2_source_lock.py \
  tests/test_tokenizer.py \
  tests/test_srgm_worlds.py \
  tests/test_current_sources.py
python -m py_compile \
  corpusgen/reasoning_v2/wikidata_source.py \
  corpusgen/reasoning_v2/catalog.py \
  corpusgen/reasoning_v2/renderers.py
git diff --check
```

Expected: all pass with no warnings or diff-check output.

## Plan self-review

- The four tasks cover every selected-design requirement.
- Source authority and derived authority remain separate.
- All later signatures match Task 1/2 definitions.
- Catalog and renderer integration wait for verified indexed view APIs.
- The capacity question fails closed rather than inventing an edge count.
- No production data, AWS action, source-lock mutation, or commit is included.
