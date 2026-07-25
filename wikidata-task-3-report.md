# Wikidata derived-view Task 3 report

Status: **DONE**

## Commit identities

- Base: `f098362`
- Implementation and tests:
  `40e51c520ce57f581fc5202fe1b18de961ce51fb`

The implementation commit contains only:

- `corpusgen/reasoning_v2/catalog.py`
- `tests/test_reasoning_v2_catalog.py`

The supplied untracked `wikidata-task-3-brief.md` was not modified or committed.

## RED

Command:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_catalog.py::test_production_adapter_binds_view_archive_member_split_row_and_edge \
  tests/test_reasoning_v2_catalog.py::test_catalog_receipt_binds_verified_wikidata_view_sha256 \
  tests/test_reasoning_v2_catalog.py::test_distinct_edges_appear_once_before_first_revisit \
  tests/test_reasoning_v2_catalog.py::test_edge_count_above_available_records_fails_before_draft_output \
  tests/test_reasoning_v2_catalog.py::test_end_to_end_catalog_is_byte_identical_across_rebuilds
```

Exact terminal summary:

```text
FFFFF                                                                    [100%]
=================================== FAILURES ===================================
_____ test_production_adapter_binds_view_archive_member_split_row_and_edge _____
E           AttributeError: module 'corpusgen.reasoning_v2.catalog' has no attribute 'WikidataGraphCatalogSource'

tests/test_reasoning_v2_catalog.py:550: AttributeError
___________ test_catalog_receipt_binds_verified_wikidata_view_sha256 ___________
E       AttributeError: module 'corpusgen.reasoning_v2.catalog' has no attribute 'WikidataGraphCatalogSource'

tests/test_reasoning_v2_catalog.py:534: AttributeError
_____________ test_distinct_edges_appear_once_before_first_revisit _____________
E       AttributeError: module 'corpusgen.reasoning_v2.catalog' has no attribute 'WikidataGraphCatalogSource'

tests/test_reasoning_v2_catalog.py:624: AttributeError
______ test_edge_count_above_available_records_fails_before_draft_output _______
E       AttributeError: module 'corpusgen.reasoning_v2.catalog' has no attribute 'WikidataGraphCatalogSource'

tests/test_reasoning_v2_catalog.py:534: AttributeError
__________ test_end_to_end_catalog_is_byte_identical_across_rebuilds ___________
E       AttributeError: module 'corpusgen.reasoning_v2.catalog' has no attribute 'WikidataGraphCatalogSource'

tests/test_reasoning_v2_catalog.py:534: AttributeError
=========================== short test summary info ============================
FAILED tests/test_reasoning_v2_catalog.py::test_production_adapter_binds_view_archive_member_split_row_and_edge
FAILED tests/test_reasoning_v2_catalog.py::test_catalog_receipt_binds_verified_wikidata_view_sha256
FAILED tests/test_reasoning_v2_catalog.py::test_distinct_edges_appear_once_before_first_revisit
FAILED tests/test_reasoning_v2_catalog.py::test_edge_count_above_available_records_fails_before_draft_output
FAILED tests/test_reasoning_v2_catalog.py::test_end_to_end_catalog_is_byte_identical_across_rebuilds
5 failed in 1.42s
```

This was the expected RED: all five tests reached real derived-view setup and
failed because the production adapter did not exist.

## GREEN

The same five-test command after the minimal implementation produced:

```text
.....                                                                    [100%]
5 passed in 2.18s
```

The final focused rerun after self-review/refactoring produced:

```text
.....                                                                    [100%]
5 passed in 0.74s
```

## Final regression verification

Command:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_wikidata_source.py \
  tests/test_reasoning_v2_catalog.py \
  tests/test_reasoning_v2_contracts.py \
  tests/test_reasoning_v2_source_lock.py
```

Exact output:

```text
........................................................................ [ 26%]
........................................................................ [ 53%]
........................................................................ [ 80%]
......................................................                   [100%]
270 passed in 20.06s
```

Compile command:

```bash
python -m py_compile \
  corpusgen/reasoning_v2/catalog.py \
  tests/test_reasoning_v2_catalog.py
```

Exact output: none; exit status `0`.

Patch check:

```bash
git diff --check
```

Exact output: none; exit status `0`.

## Bounded-memory evidence

The adapter:

- authenticates/counts distinct edges with a streaming `sum`;
- streams distinct-edge keys and first-pass drafts;
- streams verified raw training triples for revisits;
- stores only scalar commitments and the fixed three-row archive inventory;
- never reads aliases and never accumulates edges or drafts in a Python
  collection.

A real derived view with 10,000 distinct edges was built, then `tracemalloc`
was started immediately before adapter construction. The adapter emitted
10,100 drafts without collecting them.

Command shape:

```bash
python - <<'PY'
# Build a descriptor-verified view from tiny real archives after replacing
# the inductive training member with 10,000 unique rows.
tracemalloc.start()
source = catalog_module.WikidataGraphCatalogSource(view)
_current, constructor_peak = tracemalloc.get_traced_memory()
lengths = catalog_module._BalancedTargetLengths.create(10_100, 1)
draft_count = sum(
    1 for _draft in source.iter_drafts(source_root, lengths)
)
_current, total_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(
    f"distinct_edges={view.receipt.distinct_edges} drafts={draft_count} "
    f"constructor_peak_bytes={constructor_peak} "
    f"total_peak_bytes={total_peak}"
)
PY
```

Exact output:

```text
distinct_edges=10000 drafts=10100 constructor_peak_bytes=1441118 total_peak_bytes=1445195
```

The roughly 1.45 MiB peak while emitting 10,100 drafts is consistent with
streaming behavior; no edge-count-sized Python collection survives or grows
during drafting.

## Self-review

- Verified authority: constructor consumption of
  `iter_distinct_training_edges` rejects a caller-constructed
  `WikidataDerivedView`; subsequent reads use only verified public view APIs.
- Locator closure: every production graph locator has canonical bytewise keys
  `member`, `path`, `row`, `split`, `training_edge_key`, `training_split`, and
  `wikidata_view_sha256`; rows are one-based.
- Byte authority: `source_byte_sha256` comes from the verified receipt's locked
  archive record, not from a decoded member or caller value.
- Complete-once order: numerically ordered distinct edges use an `edge` source
  key prefix and precede all `revisit` source keys; revisits stream the verified
  training sequence deterministically.
- Capacity: the preflight compares verified distinct-edge count with allocated
  graph records after source-tree verification but before creating a catalog
  stage or consuming any lane drafts. Its error includes the required text.
- Receipt closure: `InputCatalog` and canonical `catalog-index.json` bind the
  verified view SHA-256, and the view's source-lock commitment must equal the
  catalog source lock.
- Determinism: independent derived-view and catalog rebuilds are byte-identical.
- Scope: no AWS/network access, no push, no amend, and no out-of-scope source,
  renderer, package, or Wikidata-source file changes.

Concerns: none known.

## Review remediation (reviewed head `2d65589`)

Status: **DONE**

Implementation and tests commit:
`c1533e64ad039c7127eeb50d72483f9d404bd8b6`.

### RED

Command (the explicit basetemp was removed by the command's exit trap):

```bash
BASE="/tmp/memorysplit-wd3-red-$$" &&
trap 'rm -rf -- "$BASE"' EXIT &&
python -m pytest -q \
  tests/test_reasoning_v2_catalog.py::test_catalog_rejects_duck_typed_wikidata_authority \
  tests/test_reasoning_v2_catalog.py::test_catalog_reopens_view_and_rejects_on_disk_drift_before_staging \
  tests/test_reasoning_v2_catalog.py::test_distinct_edges_appear_once_before_first_revisit \
  tests/test_reasoning_v2_catalog.py::test_wikidata_graph_covers_every_training_edge_before_revisit \
  --basetemp="$BASE"
```

Exact terminal summary:

```text
FF..                                                                     [100%]
FAILED tests/test_reasoning_v2_catalog.py::test_catalog_rejects_duck_typed_wikidata_authority
FAILED tests/test_reasoning_v2_catalog.py::test_catalog_reopens_view_and_rejects_on_disk_drift_before_staging
2 failed, 2 passed in 0.90s
```

The first failure was `Failed: DID NOT RAISE <class 'ValueError'>`, proving that
the reviewed implementation accepted a property-compatible look-alike. The
second reached the derived-view identity check only after catalog staging had
started; the public error was `Wikidata graph training-edge authority failed`
rather than the requested pre-staging reopen failure.

### GREEN

The same four-test selection after the implementation produced:

```text
....                                                                     [100%]
4 passed in 0.58s
```

The builder now requires the exact registered
`WikidataGraphCatalogSource`. Before creating an output parent, quarantine, or
stage, it reopens the registered distinct-edge stream through the verified-view
API. The capacity count, view receipt SHA-256, and source-lock binding are
returned only after that fresh verification. Duck-typed properties are not
read for build authority.

### Multi-cardinality process-memory evidence

The measurement used Darwin `resource.getrusage(RUSAGE_SELF).ru_maxrss`, not
`tracemalloc`. Each cardinality ran in a fresh Python process. That process
built a real descriptor-verified derived view, then forked a child and recorded
the child's baseline maximum RSS immediately before adapter construction. The
child constructed the production adapter and streamed all drafts without
collecting them. Thus `rss_growth_bytes` isolates adapter construction and
iteration from archive/view construction.

Command:

```bash
python - <<'PY'
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "tests"))
from test_reasoning_v2_catalog import _measure_production_adapter_rss

root = Path("/tmp/memorysplit-wd3-rss-evidence").resolve()
if root.exists():
    shutil.rmtree(root)
root.mkdir()
try:
    results = [
        _measure_production_adapter_rss(root / str(count), count)
        for count in (1_000, 10_000, 100_000)
    ]
    for result in results:
        print(json.dumps(result, sort_keys=True))
    growth = [row["rss_growth_bytes"] for row in results]
    print(f"rss_growth_spread_bytes={max(growth) - min(growth)}")
finally:
    shutil.rmtree(root)
PY
```

Exact measurements:

| Distinct edges | Drafts | Baseline RSS (bytes) | Peak RSS (bytes) | RSS growth (bytes) |
|---:|---:|---:|---:|---:|
| 1,000 | 1,100 | 6,012,928 | 13,926,400 | 7,913,472 |
| 10,000 | 10,100 | 6,029,312 | 15,089,664 | 9,060,352 |
| 100,000 | 100,100 | 6,012,928 | 17,039,360 | 11,026,432 |

Exact final line:

```text
rss_growth_spread_bytes=3112960
```

Across a 100x edge-count increase, adapter RSS growth rose by 3,112,960 bytes,
not proportionally with cardinality. The committed regression asserts a growth
spread below 8 MiB. Its isolated pytest run produced:

```text
.
1 passed in 11.21s
```

### Final regression verification

Command:

```bash
BASE="/tmp/memorysplit-wd3-required-$$" &&
trap 'rm -rf -- "$BASE"' EXIT &&
python -m pytest -q \
  tests/test_reasoning_v2_catalog.py \
  tests/test_reasoning_v2_source_lock.py \
  --basetemp="$BASE"
```

Exact output:

```text
........................................................................ [ 52%]
.................................................................        [100%]
137 passed in 17.73s
```

Compile command:

```bash
python -m py_compile \
  /Users/stephenzhang/Documents/MemorySplit/.worktrees/wikidata-catalog-adapter/corpusgen/reasoning_v2/catalog.py \
  /Users/stephenzhang/Documents/MemorySplit/.worktrees/wikidata-catalog-adapter/tests/test_reasoning_v2_catalog.py
```

Exact output: none; exit status `0`.

Patch check:

```bash
git diff --check
```

Exact output: none; exit status `0`.

### Remediation self-review

- Build authority: an identity-keyed weak registry holds immutable authority
  captured only after verified-view iteration succeeds. Exact production type
  plus registry membership rejects both look-alikes and uninitialized
  production instances.
- Fresh verification: every build reopens and verifies the distinct-edge
  stream before output staging. On-disk replacement or mutation after adapter
  construction fails closed and emits no stage or quarantine artifact.
- Capacity and receipt: the preflight count, view SHA-256, and source-lock
  binding come from the freshly revalidated registered authority, never from
  caller-supplied properties.
- Revisit provenance: tests assert every locator field and locked archive hash
  exactly for inductive and transductive revisit drafts.
- Final order: the end-to-end catalog test checks that every distinct edge
  occurs exactly once, in verified order, before the first revisit and that no
  training-edge flag appears afterward.
- Fixture isolation: the catalog tests now define their archive fixture helpers
  locally and no longer import underscore-private helpers from another test
  module. Non-production lanes retain their existing fixture adapters.
- Scope: only `catalog.py`, its catalog test, and this report were changed. No
  network or AWS access, push, amend, or out-of-scope file edit occurred.

Concerns: process RSS is allocator- and platform-sensitive, so the committed
test checks the cross-cardinality spread rather than exact byte values. No
known correctness concern remains.
