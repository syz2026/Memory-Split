# Wikidata5M Real Hashmap Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, and publish `memorysplit-wikidata5m-real-hashmap-3000.zip`, containing exactly 3,000 balanced, collision-safe, array-valued keys derived from the pinned Wikidata5M transductive graph.

**Architecture:** Reuse the committed Wikidata5M lock verification, safe extraction, triple parser, relation statistics, and frozen relation selector. Add one focused builder module that uses sequential disk-backed SQLite passes and bounded per-relation heaps, plus one CLI that atomically builds the archive and independently rebuilds it during verification.

**Tech Stack:** Python 3.12, standard-library `sqlite3`, `hashlib`, `json`, `zipfile`, `tempfile`, and `zlib`; the existing `hf` CLI; pytest 8+.

## Global Constraints

- The approved specification is `docs/superpowers/specs/2026-07-22-wikidata5m-real-hashmap-design.md`.
- Start from commit `45580a514bc103c1434dadcf499fc8ea325e3cf4` on `feat/srgm-implementation`.
- Work in a fresh `.worktrees/wikidata5m-real-hashmap` worktree on branch `feat/wikidata5m-real-hashmap`; do not edit or reuse `.worktrees/srgm-implementation`.
- Do not modify the existing three-archive `sources/wikidata5m.lock.json`; add a dedicated two-archive lock for this deliverable.
- Download only `wikidata5m_alias.tar.gz` and `wikidata5m_transductive.tar.gz`, totaling exactly 365,707,965 bytes.
- Require Hugging Face revision `6b2b09672129e280c0c9da97ab58154e9d535e6b`; reject source revision, size, or SHA-256 drift.
- Extract only regular files through `safe_extract_archives`, always passing the dedicated lock explicitly.
- Read only `wikidata5m_transductive_train.txt`; each accepted row is exactly `QID<TAB>PID<TAB>QID`.
- Alias rows remain strict: malformed IDs, fewer than two tab-separated fields, empty normalized alias fields, and duplicate canonical IDs are schema errors.
- A missing display label means the canonical QID or PID is absent from its alias file.
- Display labels are the first within-ID deduplicated alias after NFKC normalization and whitespace collapse, preserving the original capitalization.
- QIDs and PIDs are authoritative and appear in every display key.
- Reuse the exact minimum support 5,000, exact functionality threshold `distinct_subjects / support >= 0.95`, support/functionality/PID ordering, and 32-relation selector.
- Do not add synthetic literal relations.
- Group by canonical `(subject QID, property PID)`, remove duplicate edges, preserve every distinct object, and sort values by numeric QID.
- Exclude a grouped key if its subject, relation, or any retained object is absent from the corresponding alias file; record every exclusion count.
- Emit 94 keys from selected relation ranks 1–24 and 93 keys from ranks 25–32.
- Rank addresses by SHA-256 of UTF-8 `QID<TAB>PID`, then numeric QID and PID.
- Expose no CLI override for source revision, source lock, algorithm version, relation count, quotas, key count, or sample seed.
- Serialize UTF-8 text with LF endings, a trailing newline, deterministic JSON field ordering, compact separators, and `allow_nan=False`.
- Add ZIP entries in lexical path order with timestamp `1980-01-01 00:00:00`, POSIX mode `0644`, and DEFLATE level 9.
- Guarantee and test byte-identical ZIP output within the same Python 3.12 and zlib environment; record exact Python and zlib versions in `build_manifest.json`.
- Use `TemporaryDirectory` with `dir=work_root` and the exact prefixes `wikidata5m-build-`, `wikidata5m-verify-`, `wikidata5m-relations-`, and `wikidata5m-grouped-`.
- Close and delete the relation-statistics database before creating the grouped-edge database; delete all build-pass temporary state before fresh verification starts.
- Write a sibling temporary archive and publish with `os.replace` only after staged validation succeeds.
- Fresh verification rebuilds the expected dataset and ZIP from the locked source in new temporary directories and requires byte equality.
- Remove `memorysplit-popqa-real-hashmap-3000.zip` only after source-backed verification and `unzip -t` both succeed.
- Require at least 20 GiB free under the production work root; the current machine has approximately 24 GiB free.

---

## Execution Preflight

- [ ] **Create a fresh isolated worktree**

Invoke `superpowers:using-git-worktrees`. If no native worktree tool is available, run:

```bash
ROOT=/Users/stephenzhang/Documents/MemorySplit
BASE=feat/srgm-implementation
BASE_COMMIT=45580a514bc103c1434dadcf499fc8ea325e3cf4
BRANCH=feat/wikidata5m-real-hashmap
WORKTREE="$ROOT/.worktrees/wikidata5m-real-hashmap"
PYTHON="$ROOT/.venv/bin/python"

cd "$ROOT"
test "$(git rev-parse "$BASE")" = "$BASE_COMMIT"
git check-ignore -q .worktrees
test ! -e "$WORKTREE"
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  printf 'refusing existing branch: %s\n' "$BRANCH"
  exit 1
fi
git worktree add "$WORKTREE" -b "$BRANCH" "$BASE"
cd "$WORKTREE"
test "$(git rev-parse HEAD)" = "$BASE_COMMIT"
test -z "$(git status --porcelain)"
test -x "$PYTHON"
printf 'worktree=%s\nbranch=%s\nbase=%s\n' \
  "$WORKTREE" "$(git branch --show-current)" "$(git rev-parse HEAD)"
```

Expected final output:

```text
worktree=/Users/stephenzhang/Documents/MemorySplit/.worktrees/wikidata5m-real-hashmap
branch=feat/wikidata5m-real-hashmap
base=45580a514bc103c1434dadcf499fc8ea325e3cf4
```

- [ ] **Verify the reusable baseline**

```bash
cd /Users/stephenzhang/Documents/MemorySplit/.worktrees/wikidata5m-real-hashmap
PYTHONDONTWRITEBYTECODE=1 \
  /Users/stephenzhang/Documents/MemorySplit/.venv/bin/python \
  -m pytest tests/test_wikidata5m.py tests/test_relation_schema.py \
  -q -p no:cacheprovider

PYTHONDONTWRITEBYTECODE=1 \
  /Users/stephenzhang/Documents/MemorySplit/.venv/bin/python \
  -m pytest tests -q -p no:cacheprovider
```

Expected: both commands exit 0 with no failures. Stop and report any baseline failure.

## File Map

### New committed files

- `sources/wikidata5m_hashmap.lock.json` — exact alias-plus-transductive source lock.
- `sources/Wikidata-CC0-1.0.txt` — vendored official CC0 1.0 legal text.
- `corpusgen/wikidata5m_hashmap.py` — strict aliases, selection, disk-backed grouping, rendering, deterministic ZIP writing, and verification.
- `scripts/build_wikidata5m_hashmap.py` — fixed `build` and `verify` CLI.
- `tests/test_wikidata5m_hashmap.py` — unit, integration, determinism, mutation, cleanup, and CLI tests.

### Existing files modified

- `corpusgen/relation_schema.py` — add a keyword-only temporary-root parameter without changing default callers.
- `tests/test_relation_schema.py` — prove relation-statistics SQLite state is created under and removed from that root.

### Existing files reused unchanged

- `corpusgen/wikidata5m.py`
- `scripts/fetch_wikidata5m.py`
- `tests/test_wikidata5m.py`

### Generated archive members

Under `memorysplit-wikidata5m-real-hashmap-3000/`:

- `hashmap.json`
- `hashmap.jsonl`
- `records.jsonl`
- `relation_summary.json`
- `build_manifest.json`
- `source/wikidata5m.lock.json`
- `README.md`
- `CITATION.bib`
- `LICENSES/Wikidata-CC0-1.0.txt`
- `SHA256SUMS`

---

### Task 1: Freeze Source, Strict Aliases, and Selection

**Files:**
- Create: `sources/wikidata5m_hashmap.lock.json`
- Create: `sources/Wikidata-CC0-1.0.txt`
- Create: `corpusgen/wikidata5m_hashmap.py`
- Create: `tests/test_wikidata5m_hashmap.py`
- Modify: `corpusgen/relation_schema.py:590-612`
- Modify: `tests/test_relation_schema.py`

**Interfaces:**
- Consumes: `WikidataLock`, `parse_qid`, `parse_pid`, `normalize_alias`, `read_aliases`, `compute_relation_stats`, `select_entity_relations`, and `build_download_command`.
- Produces:
  - `HashmapBuildError`
  - `SelectedRelation`
  - `RelationSelection`
  - `normalize_display_alias(value: str) -> str`
  - `first_display_alias(raw_aliases: Sequence[str]) -> str`
  - `iter_strict_alias_rows(path: str | Path, prefix: str) -> Iterator[tuple[int, int | str, str]]`
  - `canonical_address(subject: int, relation_id: str) -> str`
  - `address_sort_key(subject: int, relation_id: str) -> tuple[bytes, int, int]`
  - `relation_quota(rank: int) -> int`
  - `select_hashmap_relations(train_path: str | Path, relation_alias_path: str | Path, *, work_root: str | Path) -> RelationSelection`
  - `compute_relation_stats(transductive_train: str | Path, aliases: Mapping[str, Sequence[str]] | AliasCatalog, *, work_root: str | Path | None = None) -> tuple[RelationStats, ...]`

- [ ] **Step 1: Write failing source and alias tests**

Add these exact assertions to `tests/test_wikidata5m_hashmap.py`:

```python
EXPECTED_HASHMAP_LOCK = {
    "repo_id": "intfloat/wikidata5m",
    "repo_type": "dataset",
    "revision": "6b2b09672129e280c0c9da97ab58154e9d535e6b",
    "files": {
        "wikidata5m_alias.tar.gz": {
            "bytes": 197449751,
            "sha256": "0330f580c9f7a57cbad949ac380835fdd2a2e14d96cc0f13fc435401d6b463a8",
        },
        "wikidata5m_transductive.tar.gz": {
            "bytes": 168258214,
            "sha256": "383160990b41c0905fc03f4a8afbb9b12be1ca3591e026bde6cdc94a59542597",
        },
    },
}


def test_hashmap_lock_is_exactly_two_archives():
    value = json.loads(HASHMAP_LOCK.read_text(encoding="utf-8"))
    assert value == EXPECTED_HASHMAP_LOCK
    assert sum(item["bytes"] for item in value["files"].values()) == 365_707_965


def test_hashmap_download_command_requests_only_locked_archives():
    assert build_download_command(Path("/data"), HASHMAP_LOCK) == [
        "hf",
        "download",
        "intfloat/wikidata5m",
        "--repo-type",
        "dataset",
        "--revision",
        "6b2b09672129e280c0c9da97ab58154e9d535e6b",
        "--include",
        "wikidata5m_alias.tar.gz",
        "--include",
        "wikidata5m_transductive.tar.gz",
        "--local-dir",
        "/data/wikidata5m",
    ]


def test_display_alias_normalization_preserves_original_case():
    assert normalize_display_alias("  Douglas\t  Adams  ") == "Douglas Adams"
    assert first_display_alias((" Douglas   Adams ", "douglas adams")) == (
        "Douglas Adams"
    )


@pytest.mark.parametrize(
    "line",
    [
        "Q1\n",
        "Q1\t\n",
        "Q1\t \tvalid\n",
        "not-a-qid\tname\n",
    ],
)
def test_entity_alias_rows_remain_strict(tmp_path, line):
    path = tmp_path / "wikidata5m_entity.txt"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_strict_alias_rows(path, "Q"))


def test_frozen_quotas_sum_to_three_thousand():
    assert [relation_quota(rank) for rank in range(1, 33)] == (
        [94] * 24 + [93] * 8
    )
    assert sum(relation_quota(rank) for rank in range(1, 33)) == 3_000
```

Also test that an absent QID produces a missing-label exclusion in Task 2; do not represent missing labels with empty alias fields.

- [ ] **Step 2: Write the failing temporary-root test**

In `tests/test_relation_schema.py`, call:

```python
work_root = tmp_path / "sqlite-work"
work_root.mkdir()
stats = compute_relation_stats(train, aliases, work_root=work_root)
assert stats
assert list(work_root.iterdir()) == []
```

Expected behavior: the SQLite directory is created below `work_root` and removed before the function returns.

- [ ] **Step 3: Run the tests to verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
  -m pytest \
  tests/test_wikidata5m_hashmap.py \
  tests/test_relation_schema.py \
  -q -p no:cacheprovider
```

Expected: nonzero exit because the focused module, source files, and `work_root` interface do not exist.

- [ ] **Step 4: Add the exact two-file lock**

Create `sources/wikidata5m_hashmap.lock.json`:

```json
{
  "repo_id": "intfloat/wikidata5m",
  "repo_type": "dataset",
  "revision": "6b2b09672129e280c0c9da97ab58154e9d535e6b",
  "files": {
    "wikidata5m_alias.tar.gz": {
      "bytes": 197449751,
      "sha256": "0330f580c9f7a57cbad949ac380835fdd2a2e14d96cc0f13fc435401d6b463a8"
    },
    "wikidata5m_transductive.tar.gz": {
      "bytes": 168258214,
      "sha256": "383160990b41c0905fc03f4a8afbb9b12be1ca3591e026bde6cdc94a59542597"
    }
  }
}
```

- [ ] **Step 5: Vendor the official CC0 text**

```bash
curl -fsSL \
  https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt |
  "$PYTHON" -c \
    'import sys; b=sys.stdin.buffer.read().replace(b"\r\n", b"\n").replace(b"\r", b"\n"); sys.stdout.buffer.write(b.rstrip(b"\n") + b"\n")' \
  > sources/Wikidata-CC0-1.0.txt
```

Add:

```python
def test_cc0_text_is_complete_lf_terminated_utf8():
    content = CC0_PATH.read_bytes()
    assert content.startswith(b"Creative Commons Legal Code\n")
    assert b"CC0 1.0 Universal" in content
    assert b"4. Limitations and Disclaimers." in content
    assert b"\r" not in content
    assert content.endswith(b"\n")
```

- [ ] **Step 6: Implement strict display helpers**

Use `read_aliases` for the small relation alias file. Implement a streaming entity-alias parser with identical validation: exactly one valid canonical ID followed by at least one non-empty normalized alias, within-ID duplicate removal using `normalize_alias`, and duplicate-ID rejection by its SQLite consumer.

```python
ALGORITHM_VERSION = "wikidata5m-hashmap-v1"
ARCHIVE_ROOT = "memorysplit-wikidata5m-real-hashmap-3000"
KEY_COUNT = 3_000
RELATION_COUNT = 32
ROOT = Path(__file__).resolve().parents[1]
HASHMAP_LOCK_PATH = ROOT / "sources" / "wikidata5m_hashmap.lock.json"
CC0_PATH = ROOT / "sources" / "Wikidata-CC0-1.0.txt"
EXPECTED_HASHMAP_LOCK = {
    "repo_id": "intfloat/wikidata5m",
    "repo_type": "dataset",
    "revision": "6b2b09672129e280c0c9da97ab58154e9d535e6b",
    "files": {
        "wikidata5m_alias.tar.gz": {
            "bytes": 197449751,
            "sha256": "0330f580c9f7a57cbad949ac380835fdd2a2e14d96cc0f13fc435401d6b463a8",
        },
        "wikidata5m_transductive.tar.gz": {
            "bytes": 168258214,
            "sha256": "383160990b41c0905fc03f4a8afbb9b12be1ca3591e026bde6cdc94a59542597",
        },
    },
}


class HashmapBuildError(RuntimeError):
    """Raised when frozen hashmap requirements cannot be satisfied."""


def normalize_display_alias(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("alias must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).split())


def first_display_alias(raw_aliases: Sequence[str]) -> str:
    if not raw_aliases:
        raise ValueError("display aliases must be nonempty")
    display = normalize_display_alias(raw_aliases[0])
    if not display:
        raise ValueError("display alias must be nonempty")
    return display
```

`iter_strict_alias_rows(path, "Q")` yields `(line_number, numeric_qid, first_display_alias)` and fails on every invalid or empty field before yielding that row.

Define the selection contracts:

```python
@dataclass(frozen=True)
class SelectedRelation:
    rank: int
    relation_id: str
    label: str
    support: int
    distinct_subjects: int
    distinct_objects: int
    entity_count: int
    quota: int


@dataclass(frozen=True)
class RelationSelection:
    relations: tuple[SelectedRelation, ...]
    filter_counts: dict[str, int]
```

Implement address and quota functions exactly:

```python
def canonical_address(subject: int, relation_id: str) -> str:
    if isinstance(subject, bool) or not isinstance(subject, int) or subject < 0:
        raise ValueError("subject must be a nonnegative QID number")
    parse_pid(relation_id)
    return f"Q{subject}\t{relation_id}"


def address_sort_key(subject: int, relation_id: str) -> tuple[bytes, int, int]:
    encoded = canonical_address(subject, relation_id).encode("utf-8")
    return hashlib.sha256(encoded).digest(), subject, int(relation_id[1:])


def relation_quota(rank: int) -> int:
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 32:
        raise ValueError("relation rank must be between 1 and 32")
    return 94 if rank <= 24 else 93
```

- [ ] **Step 7: Implement selection and sequential statistics cleanup**

Change `compute_relation_stats` to:

```python
def compute_relation_stats(
    transductive_train: str | Path,
    aliases: Mapping[str, Sequence[str]] | AliasCatalog,
    *,
    work_root: str | Path | None = None,
) -> tuple[RelationStats, ...]:
```

Pass `dir=None if work_root is None else Path(work_root)` to its `TemporaryDirectory`. Keep `build_relation_schema` unchanged so existing callers retain the default behavior.

`select_hashmap_relations` must:

1. Strictly read relation aliases with `read_aliases`.
2. Call `compute_relation_stats(train_path, relation_aliases, work_root=work_root)`.
3. Call `select_entity_relations(stats, count=32)`.
4. Use exact integer arithmetic for functionality.
5. Use the first normalized/collapsed raw alias for display.
6. Assign rank and quota.
7. Return `RelationSelection(relations, filter_counts)`.

Relation exclusion counts use this mutually exclusive order:

1. PID absent from relation alias file;
2. support below 5,000;
3. functionality below 0.95;
4. survived thresholds but ranked after the first 32.

`filter_counts` contains exactly `missing_relation_alias`, `below_min_support`, `below_min_functionality`, `survived_not_selected`, and `selected`; `selected` must equal 32.

- [ ] **Step 8: Verify GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
  -m pytest \
  tests/test_wikidata5m_hashmap.py \
  tests/test_wikidata5m.py \
  tests/test_relation_schema.py \
  -q -p no:cacheprovider
```

Expected: exit 0 with no failures.

- [ ] **Step 9: Commit**

```bash
git add \
  sources/wikidata5m_hashmap.lock.json \
  sources/Wikidata-CC0-1.0.txt \
  corpusgen/wikidata5m_hashmap.py \
  corpusgen/relation_schema.py \
  tests/test_wikidata5m_hashmap.py \
  tests/test_relation_schema.py
git commit -m "feat: freeze Wikidata5M hashmap inputs"
```

---

### Task 2: Build Balanced Array-Valued Records

**Files:**
- Modify: `corpusgen/wikidata5m_hashmap.py`
- Modify: `tests/test_wikidata5m_hashmap.py`

**Interfaces:**
- Consumes: `RelationSelection`, `iter_triples`, strict entity aliases, and `work_root`.
- Produces:
  - `HashmapValue`
  - `HashmapRecord`
  - `RelationSummary`
  - `BuildCounts`
  - `HashmapDataset`
  - `build_hashmap_dataset(train_path, entity_alias_path, selection, *, work_root) -> HashmapDataset`

- [ ] **Step 1: Add a deterministic 32-relation fixture**

Implement `_write_balanced_fixture(root, reverse_triples=False)` with:

1. Selected relations `P1` through `P32`, all labeled `shared relation`.
2. Subjects 1–102 per relation using QID number `relation_index * 10_000 + subject_index`, all labeled `Homonym` except index 101, whose QID is absent from the alias file.
3. One object per address using QID number `1_000_000 + relation_index * 10_000 + subject_index`, except index 102, whose object QID is absent from the alias file.
4. A second distinct object on the lowest-ranked eligible address for each relation.
5. One repeated copy of that address's first edge.
6. One unselected `P999` triple.
7. Reversal of only the complete triple-row order when `reverse_triples=True`.

The expected counts are:

```python
EXPECTED_COUNTS = {
    "source_triples": 3329,
    "selected_relation_triples": 3328,
    "unselected_relation_triples": 1,
    "distinct_selected_edges": 3296,
    "duplicate_selected_edges": 32,
    "selected_grouped_keys": 3264,
    "missing_subject_alias_keys": 32,
    "missing_object_alias_keys": 32,
    "eligible_keys": 3200,
    "eligible_edges": 3232,
    "unsampled_eligible_keys": 200,
    "emitted_keys": 3000,
    "emitted_edges": 3032,
}
```

- [ ] **Step 2: Write failing grouping tests**

```python
def test_builder_is_balanced_array_valued_and_order_independent(tmp_path):
    first_fixture = _write_balanced_fixture(tmp_path / "first")
    second_fixture = _write_balanced_fixture(
        tmp_path / "second",
        reverse_triples=True,
    )
    first_work = tmp_path / "first-work"
    second_work = tmp_path / "second-work"
    first_work.mkdir()
    second_work.mkdir()

    first = build_hashmap_dataset(
        first_fixture.train,
        first_fixture.entities,
        first_fixture.selection,
        work_root=first_work,
    )
    second = build_hashmap_dataset(
        second_fixture.train,
        second_fixture.entities,
        second_fixture.selection,
        work_root=second_work,
    )

    assert first == second
    assert first.counts.to_dict() == EXPECTED_COUNTS
    assert [item.emitted_keys for item in first.relations] == (
        [94] * 24 + [93] * 8
    )
    assert len({record.canonical_address for record in first.records}) == 3000
    assert len({record.display_key for record in first.records}) == 3000
    assert sum(len(record.values) for record in first.records) == 3032
    assert list(first_work.iterdir()) == []
    assert list(second_work.iterdir()) == []


def test_builder_fails_when_one_relation_cannot_meet_quota(tmp_path):
    fixture = _write_balanced_fixture(tmp_path / "fixture")
    fixture.remove_subject_aliases("P32", count=8)
    work_root = tmp_path / "work"
    work_root.mkdir()

    with pytest.raises(
        HashmapBuildError,
        match=r"P32 has 92 eligible keys but requires 93",
    ):
        build_hashmap_dataset(
            fixture.train,
            fixture.entities,
            fixture.selection,
            work_root=work_root,
        )

    assert list(work_root.iterdir()) == []
```

Add:

```python
def test_lowest_ranked_multivalue_addresses_preserve_all_targets(tmp_path):
    fixture = _write_balanced_fixture(tmp_path / "fixture")
    work_root = tmp_path / "work"
    work_root.mkdir()
    dataset = build_hashmap_dataset(
        fixture.train,
        fixture.entities,
        fixture.selection,
        work_root=work_root,
    )
    by_address = {
        record.canonical_address: record
        for record in dataset.records
    }
    for address in fixture.multivalue_addresses:
        values = by_address[address].values
        assert len(values) == 2
        numeric_ids = [int(value.id[1:]) for value in values]
        assert numeric_ids == sorted(set(numeric_ids))


def test_builder_rejects_duplicate_entity_alias_ids(tmp_path):
    fixture = _write_balanced_fixture(tmp_path / "fixture")
    fixture.append_duplicate_entity_alias()
    work_root = tmp_path / "work"
    work_root.mkdir()
    with pytest.raises(ValueError, match="duplicate canonical ID"):
        build_hashmap_dataset(
            fixture.train,
            fixture.entities,
            fixture.selection,
            work_root=work_root,
        )
    assert list(work_root.iterdir()) == []
```

The fixture object implements `remove_subject_aliases(relation_id, count)` by removing that many otherwise eligible QID rows for the named relation, and `append_duplicate_entity_alias()` by appending a second row for its first QID.

- [ ] **Step 3: Run the tests to verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
  -m pytest tests/test_wikidata5m_hashmap.py -q -p no:cacheprovider
```

Expected: nonzero exit because the dataset builder and record contracts do not exist.

- [ ] **Step 4: Add immutable record contracts**

```python
@dataclass(frozen=True)
class HashmapValue:
    id: str
    label: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "label": self.label}


@dataclass(frozen=True)
class HashmapRecord:
    subject_id: str
    subject_label: str
    relation_id: str
    relation_label: str
    values: tuple[HashmapValue, ...]

    @property
    def canonical_address(self) -> str:
        return f"{self.subject_id}\t{self.relation_id}"

    @property
    def display_key(self) -> str:
        return (
            f"{self.subject_label} [{self.subject_id}], "
            f"{self.relation_label} [{self.relation_id}]"
        )
```

`BuildCounts` has exactly the 13 fields in `EXPECTED_COUNTS`.

`RelationSummary` contains rank, PID, display label, support, distinct subjects, distinct objects, exact functionality numerator and denominator, quota, eligible keys, eligible edges, emitted keys, and emitted edges.

`HashmapDataset` contains ordered record and relation tuples, `BuildCounts`, and the relation exclusion counts from `RelationSelection`.

- [ ] **Step 5: Implement one disposable grouped-edge database**

Create the database only inside:

```python
with tempfile.TemporaryDirectory(
    prefix="wikidata5m-grouped-",
    dir=work_root,
) as directory:
    database = Path(directory) / "grouped.sqlite3"
```

Use:

```sql
CREATE TABLE edges (
  relation INTEGER NOT NULL,
  subject INTEGER NOT NULL,
  object INTEGER NOT NULL,
  PRIMARY KEY (relation, subject, object)
) WITHOUT ROWID;

CREATE TABLE needed_entities (
  entity INTEGER PRIMARY KEY
) WITHOUT ROWID;

CREATE TABLE seen_entity_aliases (
  entity INTEGER PRIMARY KEY
) WITHOUT ROWID;

CREATE TABLE entity_labels (
  entity INTEGER PRIMARY KEY,
  label TEXT NOT NULL
) WITHOUT ROWID;
```

For this disposable database set `journal_mode=OFF`, `synchronous=OFF`, `temp_store=FILE`, `locking_mode=EXCLUSIVE`, and `cache_size=-262144`.

Algorithm:

1. Stream triples with `iter_triples`.
2. Count every accepted triple.
3. Batch selected rows in groups of 10,000 into `edges` with `INSERT OR IGNORE`.
4. Derive duplicate counts from attempted and inserted rows.
5. Populate `needed_entities` from both edge columns.
6. Stream strict entity alias rows in batches of 500.
7. Use `seen_entity_aliases` to reject duplicate QID rows.
8. Join each alias batch to `needed_entities` before inserting into `entity_labels`.
9. Never load the complete entity alias file into Python memory.

- [ ] **Step 6: Select candidates with bounded heaps**

Query grouped addresses in relation/subject order. Apply mutually exclusive key exclusions in this order:

1. subject QID absent from `entity_labels`;
2. any retained object QID absent from `entity_labels`;
3. eligible but outside the relation quota.

Use one quota-sized worst-first heap per relation:

```python
@dataclass(frozen=True)
class _WorstFirst:
    rank: tuple[bytes, int, int]
    subject: int

    def __lt__(self, other: "_WorstFirst") -> bool:
        return self.rank > other.rank
```

```python
if len(heap) < quota:
    heapq.heappush(heap, candidate)
elif candidate.rank < heap[0].rank:
    heapq.heapreplace(heap, candidate)
```

After the scan:

1. Fail with PID, eligible count, and quota when undersupplied.
2. Sort retained candidates by normal address rank.
3. Query values ordered by numeric QID.
4. Reject empty value arrays.
5. Reject duplicate canonical address strings.
6. Reject duplicate display keys.
7. Assert 32 relations, 3,000 records, and exact quota totals.
8. Assert all source, edge, group, exclusion, and output count equations.
9. Close SQLite before leaving the `TemporaryDirectory`; verify its directory is removed before returning.

- [ ] **Step 7: Verify GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
  -m pytest \
  tests/test_wikidata5m_hashmap.py \
  tests/test_wikidata5m.py \
  tests/test_relation_schema.py \
  -q -p no:cacheprovider
```

Expected: exit 0 with no failures.

- [ ] **Step 8: Commit**

```bash
git add corpusgen/wikidata5m_hashmap.py tests/test_wikidata5m_hashmap.py
git commit -m "feat: build balanced Wikidata5M hashmap records"
```

---

### Task 3: Package, Verify, and Expose the CLI

**Files:**
- Modify: `corpusgen/wikidata5m_hashmap.py`
- Create: `scripts/build_wikidata5m_hashmap.py`
- Modify: `tests/test_wikidata5m_hashmap.py`

**Interfaces:**
- Consumes: `HashmapDataset`, `RelationSelection`, the dedicated lock, CC0 text, and `work_root`.
- Produces:
  - `ArchiveReport`
  - `render_package_files(dataset, selection, lock, cc0_text) -> dict[str, bytes]`
  - `write_deterministic_zip(path, files) -> None`
  - `build_wikidata5m_hashmap(source_root, out, *, work_root) -> ArchiveReport`
  - `verify_wikidata5m_hashmap(archive, source_root, *, work_root) -> ArchiveReport`
  - CLI subcommands `build` and `verify`

- [ ] **Step 1: Write failing archive and verification tests**

Add tests that establish:

1. Two writes in the same Python 3.12/zlib environment are byte-identical.
2. The manifest records `platform.python_version()`, `zlib.ZLIB_VERSION`, and `zlib.ZLIB_RUNTIME_VERSION`.
3. Physical ZIP member order is lexical and every member has the fixed timestamp, mode, and compression type.
4. `hashmap.json`, `hashmap.jsonl`, and `records.jsonl` reconstruct the same mapping.
5. `SHA256SUMS` covers exactly the other nine members.
6. Lock bytes inside the archive equal the dedicated two-file lock.
7. Unsafe names, duplicate members, metadata drift, checksum drift, malformed IDs, mapping disagreement, and source-edge disagreement are rejected.
8. An injected staged-verification failure leaves a missing destination absent and an existing destination unchanged.
9. Build and fresh verify leave `work_root` empty after each call.
10. `scripts/build_wikidata5m_hashmap.py --help` works repo-relative and exposes no frozen-parameter override.
11. Build and verify reject a Python runtime whose major/minor version is not 3.12.
12. Runtime loading rejects any committed lock content that differs from `EXPECTED_HASHMAP_LOCK`.

For an offline source-backed integration fixture, create production-named tar archives containing 32 relations, 5,000 distinct subjects per relation, one target per address, shared 5,000-subject and 5,000-object pools, 160,000 triples, functionality 1.0, and complete strict aliases.

- [ ] **Step 2: Run the tests to verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
  -m pytest tests/test_wikidata5m_hashmap.py -q -p no:cacheprovider
```

Expected: nonzero exit because package rendering, ZIP writing, source-backed verification, report types, and the CLI do not exist.

- [ ] **Step 3: Render the ten package files**

Use one canonical JSON helper:

```python
def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
```

Render:

- `hashmap.json` as display key to value-array mapping.
- `hashmap.jsonl` rows with `key` and `values`.
- `records.jsonl` rows with `display_key`, `subject`, `relation`, and `values`.
- `relation_summary.json` with 32 ordered relation summaries.
- `source/wikidata5m.lock.json` from `lock.canonical_bytes()`.
- `LICENSES/Wikidata-CC0-1.0.txt` from the committed source file.

`build_manifest.json` contains:

```python
{
    "algorithm_version": "wikidata5m-hashmap-v1",
    "build_environment": {
        "python": platform.python_version(),
        "zlib_compile": zlib.ZLIB_VERSION,
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
    },
    "source": lock.to_dict(),
    "input_split": "wikidata5m_transductive_train.txt",
    "selector": {
        "minimum_support": 5000,
        "minimum_functionality_numerator": 95,
        "minimum_functionality_denominator": 100,
        "relation_count": 32,
        "ordering": [
            "support_descending",
            "exact_functionality_descending",
            "numeric_pid_ascending",
        ],
    },
    "relation_filter_counts": selection.filter_counts,
    "build_counts": dataset.counts.to_dict(),
    "output_counts": {
        "keys": 3000,
        "edges": dataset.counts.emitted_edges,
        "relations": 32,
    },
}
```

The README must directly state:

```text
This dataset was built from the pinned third-party Wikidata5M derivative hosted by intfloat on Hugging Face. It is not an official Wikimedia Foundation dump.

Wikidata5M aliases do not include language tags and do not designate a canonical label. Labels in this archive are deterministic display text only; QIDs and PIDs are authoritative.

The packaged structured data is distributed under the Wikidata CC0 1.0 public-domain dedication. See https://www.wikidata.org/wiki/Wikidata:Licensing.
```

It also documents the ten files, key/value schemas, 32-relation quotas, pinned revision, one array-valued example, and Python loading examples for JSON and JSONL.

Generate `CITATION.bib` with:

```bibtex
@article{wang2021kepler,
  title = {KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation},
  author = {Wang, Xiaozhi and Gao, Tianyu and Zhu, Zhaocheng and Zhang, Zhengyan and Liu, Zhiyuan and Li, Juanzi and Tang, Jian},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {9},
  pages = {176--194},
  year = {2021},
  doi = {10.1162/tacl_a_00360}
}

@misc{intfloat2022wikidata5m,
  author = {{intfloat}},
  title = {Wikidata5M Dataset Snapshot},
  year = {2022},
  url = {https://huggingface.co/datasets/intfloat/wikidata5m},
  note = {Revision 6b2b09672129e280c0c9da97ab58154e9d535e6b}
}
```

Create `SHA256SUMS` last, with one lexical line per other member: lowercase digest, two spaces, relative member path, newline.

- [ ] **Step 4: Write and structurally validate deterministic ZIPs**

For every relative member path in lexical order:

```python
info = zipfile.ZipInfo(
    filename=f"{ARCHIVE_ROOT}/{relative_path}",
    date_time=(1980, 1, 1, 0, 0, 0),
)
info.create_system = 3
info.external_attr = (stat.S_IFREG | 0o644) << 16
info.compress_type = zipfile.ZIP_DEFLATED
archive.writestr(
    info,
    files[relative_path],
    compress_type=zipfile.ZIP_DEFLATED,
    compresslevel=9,
)
```

Do not add explicit directory entries.

Structural validation establishes:

1. Exactly ten unique members under the required root.
2. Lexical physical order and fixed metadata.
3. No absolute, parent-traversal, or backslash path.
4. `ZipFile.testzip()` returns `None`.
5. UTF-8, LF-only, trailing-newline text.
6. Canonical JSON and JSONL.
7. Exact internal SHA-256 coverage.
8. Equivalent JSON, JSONL, and structured mappings.
9. Valid IDs, labels, arrays, counts, and quotas.
10. Manifest Python/zlib versions equal the running environment.

- [ ] **Step 5: Implement sequential build and fresh verification**

`build_wikidata5m_hashmap`:

1. Require `sys.version_info[:2] == (3, 12)`.
2. Load the committed dedicated lock and require equality with the frozen `EXPECTED_HASHMAP_LOCK` mapping.
3. Create one `TemporaryDirectory(prefix="wikidata5m-build-", dir=work_root)`.
4. Verify and safely extract both archives there.
5. Run selection; its relation-statistics directory is deleted before selection returns.
6. Run dataset building; its grouped-edge directory is deleted before the dataset returns.
7. Render and write a sibling temporary ZIP.
8. Structurally validate it against the in-memory dataset.
9. Flush and `fsync`, then publish with `os.replace`.
10. Remove the outer build directory on return or failure.

`verify_wikidata5m_hashmap`:

1. Require `sys.version_info[:2] == (3, 12)` and the exact frozen lock.
2. Require an empty production pass boundary: no build temporary directory remains.
3. Create a new `TemporaryDirectory(prefix="wikidata5m-verify-", dir=work_root)`.
4. Re-verify and re-extract the locked sources.
5. Re-run selection and delete its statistics directory.
6. Re-run dataset building and delete its grouped-edge directory.
7. Render and write a reference ZIP in the verification directory.
8. Structurally validate both archives.
9. Require candidate bytes to equal reference bytes.
10. Remove the verification directory before returning.

`ArchiveReport` has exactly:

```python
archive_bytes: int
archive_sha256: str
edge_count: int
key_count: int
path: str
relation_count: int
```

- [ ] **Step 6: Add the fixed CLI**

`scripts/build_wikidata5m_hashmap.py` exposes:

The `build` subcommand requires `--source-root`, `--out`, and `--work-root`.
The `verify` subcommand requires `--source-root`, `--archive`, and `--work-root`.

Both commands load the committed lock and CC0 file internally and print one compact, sorted JSON report.

- [ ] **Step 7: Verify focused and full suites**

```bash
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
  -m pytest tests/test_wikidata5m_hashmap.py -q -p no:cacheprovider

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
  -m pytest \
  tests/test_wikidata5m_hashmap.py \
  tests/test_wikidata5m.py \
  tests/test_relation_schema.py \
  -q -p no:cacheprovider

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
  -m pytest tests -q -p no:cacheprovider
```

Expected: every command exits 0 with no failures. The determinism test asserts byte equality under the running Python 3.12/zlib environment.

- [ ] **Step 8: Commit**

```bash
git add \
  corpusgen/wikidata5m_hashmap.py \
  scripts/build_wikidata5m_hashmap.py \
  tests/test_wikidata5m_hashmap.py
git commit -m "feat: package and verify Wikidata5M hashmap"
test -z "$(git status --porcelain)"
```

Expected: commit succeeds and the final command emits no output.

---

### Task 4: Build, Verify, Publish, Then Remove PopQA

**Files:**
- Generate: `/Users/stephenzhang/Documents/MemorySplit/memorysplit-wikidata5m-real-hashmap-3000.zip`
- Delete only after verification: `/Users/stephenzhang/Documents/MemorySplit/memorysplit-popqa-real-hashmap-3000.zip`

**Interfaces:**
- Consumes: the committed CLI, exact source archives, Python 3.12/zlib, and at least 20 GiB free work space.
- Produces: the verified final ZIP and JSON build/verification receipts.

- [ ] **Step 1: Establish clean production paths and storage**

```bash
set -o pipefail

ROOT=/Users/stephenzhang/Documents/MemorySplit
WORKTREE="$ROOT/.worktrees/wikidata5m-real-hashmap"
PYTHON="$ROOT/.venv/bin/python"
SOURCE_DATA_ROOT="$ROOT/data/wikidata5m-hashmap"
SOURCE_ROOT="$SOURCE_DATA_ROOT/wikidata5m"
WORK_ROOT="$ROOT/data/wikidata5m-hashmap-work"
FINAL_ZIP="$ROOT/memorysplit-wikidata5m-real-hashmap-3000.zip"
OLD_ZIP="$ROOT/memorysplit-popqa-real-hashmap-3000.zip"
RECEIPT_ROOT="$WORK_ROOT/receipts"

cd "$WORKTREE"
test -z "$(git status --porcelain)"
test ! -e "$FINAL_ZIP"
test ! -e "$SOURCE_DATA_ROOT"
test ! -e "$WORK_ROOT"
test -f "$OLD_ZIP"
mkdir -p "$SOURCE_DATA_ROOT" "$WORK_ROOT/tmp" "$RECEIPT_ROOT"

available_kib="$(df -Pk "$WORK_ROOT" | awk 'NR == 2 {print $4}')"
test "$available_kib" -ge 20971520
printf 'available_kib=%s\n' "$available_kib"
```

Expected: `available_kib` is at least `20971520`.

- [ ] **Step 2: Download and verify only the locked archives**

```bash
/usr/bin/perl -e 'alarm shift; exec @ARGV' 7200 \
  "$PYTHON" scripts/fetch_wikidata5m.py \
  --data-root "$SOURCE_DATA_ROOT" \
  --lock sources/wikidata5m_hashmap.lock.json

"$PYTHON" - "$SOURCE_ROOT" <<'PY'
import sys
from pathlib import Path

from corpusgen.wikidata5m import WikidataLock, verify_archives

root = Path(sys.argv[1])
lock = WikidataLock.from_path("sources/wikidata5m_hashmap.lock.json")
paths = verify_archives(root, lock)
assert {path.name for path in paths} == {
    "wikidata5m_alias.tar.gz",
    "wikidata5m_transductive.tar.gz",
}
assert sum(path.stat().st_size for path in paths) == 365_707_965
print("verified_archives=2")
print("combined_bytes=365707965")
PY
```

Expected final output:

```text
verified_archives=2
combined_bytes=365707965
```

- [ ] **Step 3: Build atomically**

```bash
env TMPDIR="$WORK_ROOT/tmp" \
  /usr/bin/perl -e 'alarm shift; exec @ARGV' 21600 \
  "$PYTHON" scripts/build_wikidata5m_hashmap.py build \
  --source-root "$SOURCE_ROOT" \
  --out "$FINAL_ZIP" \
  --work-root "$WORK_ROOT" |
  tee "$RECEIPT_ROOT/build.json"
```

Expected: exit 0 and one JSON report with `key_count` 3000, `relation_count` 32, `edge_count` at least 3000, and the absolute final path. No `wikidata5m-build-*`, `wikidata5m-relations-*`, or `wikidata5m-grouped-*` directory remains under `WORK_ROOT`.

- [ ] **Step 4: Perform fresh source-backed verification**

```bash
env TMPDIR="$WORK_ROOT/tmp" \
  /usr/bin/perl -e 'alarm shift; exec @ARGV' 21600 \
  "$PYTHON" scripts/build_wikidata5m_hashmap.py verify \
  --source-root "$SOURCE_ROOT" \
  --archive "$FINAL_ZIP" \
  --work-root "$WORK_ROOT" |
  tee "$RECEIPT_ROOT/verify.json"

"$PYTHON" - "$RECEIPT_ROOT/verify.json" "$FINAL_ZIP" <<'PY'
import json
import re
import sys
from pathlib import Path

receipt_path = Path(sys.argv[1])
archive = Path(sys.argv[2]).resolve()
report = json.loads(receipt_path.read_text(encoding="utf-8"))
assert set(report) == {
    "archive_bytes",
    "archive_sha256",
    "edge_count",
    "key_count",
    "path",
    "relation_count",
}
assert report["path"] == str(archive)
assert report["archive_bytes"] == archive.stat().st_size
assert re.fullmatch(r"[0-9a-f]{64}", report["archive_sha256"])
assert report["key_count"] == 3000
assert report["edge_count"] >= 3000
assert report["relation_count"] == 32
print("verification_receipt=accepted")
PY
```

Expected final output:

```text
verification_receipt=accepted
```

The verifier's independent rebuild establishes all ten approved verification-contract items, including exact source edges, complete grouped targets, format agreement, internal hashes, safe paths, and successful decompression.

- [ ] **Step 5: Run an independent decompression test**

```bash
unzip -t "$FINAL_ZIP"
```

Expected: every member reports `OK` and the command reports no compressed-data errors.

- [ ] **Step 6: Remove the incorrect PopQA archive**

Run only after Steps 4 and 5 exit 0:

```bash
test -f "$FINAL_ZIP"
test -s "$RECEIPT_ROOT/verify.json"
test -f "$OLD_ZIP"
rm -- "$OLD_ZIP"
test ! -e "$OLD_ZIP"
printf 'removed=%s\n' "$OLD_ZIP"
```

Expected:

```text
removed=/Users/stephenzhang/Documents/MemorySplit/memorysplit-popqa-real-hashmap-3000.zip
```

- [ ] **Step 7: Print the final handoff**

```bash
"$PYTHON" - "$RECEIPT_ROOT/verify.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"output_path={report['path']}")
print(f"archive_bytes={report['archive_bytes']}")
print(f"archive_sha256={report['archive_sha256']}")
print(f"key_count={report['key_count']}")
print(f"edge_count={report['edge_count']}")
print(f"relation_count={report['relation_count']}")
PY

cd "$WORKTREE"
test -z "$(git status --porcelain)"
```

Expected: six populated handoff lines, `key_count=3000`, `relation_count=32`, a 64-character lowercase SHA-256, and no final Git-status output.
