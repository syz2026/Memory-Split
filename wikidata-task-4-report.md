# Wikidata Task 4 — Indexed Renderer Integration Report

## Status

PASS. Task 4 is implemented on `feat/wikidata-renderer` from base
`2d65589af77d79232bd2091c19241a92641374a8`.

Owned files:

- `corpusgen/reasoning_v2/renderers.py`
- `tests/test_reasoning_v2_renderers.py`
- `wikidata-task-4-report.md`

No source-lock, derived-view, catalog, packaging, CloudFormation, AWS, or
network operation was changed or performed.

## Implementation

`WikidataGraphRenderer` now:

- requires a registry-verified `WikidataDerivedView`; an indexed `Q0` alias
  miss rejects a caller-constructed dataclass before the renderer is usable;
- authenticates and pins all three source archives once at renderer
  construction, then performs metadata/identity checks before and after each
  render without rescanning archive bytes per record;
- requires the exact closed locator fields emitted by
  `WikidataGraphCatalogSource`;
- distinguishes outer `split="train"` from
  `training_split in {"inductive_train", "transductive_train"}`;
- binds exact view receipt SHA-256, source archive path/SHA-256, decoded member,
  one-based indexed row, canonical edge key, and graph phase flag;
- calls only `lookup_training_triple` and `lookup_alias` for record data;
- repeats the indexed triple lookup after token/sidecar construction to
  reverify derived-view authority on both sides of rendering; and
- preserves the existing tokenizer fitting, semantic occurrence closure,
  Dense/Split90 sidecars, and non-Wikidata exposure renderers.

No legacy Wikidata iterator is imported by the renderer.

## TDD evidence

### RED bootstrap

The first required-test invocation against base `2d65589` exited `4` during
collection because `corpusgen.reasoning_v2.renderers` did not yet exist. This
confirmed that the supplied base had no renderer implementation to satisfy the
new integration tests.

After adding the tested shared exposure-renderer baseline, its four
non-Wikidata contract tests passed:

```text
....                                                                     [100%]
4 passed in 0.63s
```

The four named Task 4 tests (the rejection test has seven parameter cases)
then produced a clean behavioral RED against an explicit
`NotImplementedError` stub:

```text
FFFFFFFFFF                                                               [100%]
10 failed in 1.73s
```

Every failure reached `WikidataGraphRenderer.render`; there were no fixture,
collection, or teardown errors in the recorded RED run.

### GREEN

The same four named Task 4 tests after the indexed implementation:

```text
..........                                                               [100%]
10 passed in 2.10s
```

The complete focused renderer file, including shared renderer and
caller-constructed-view coverage:

```text
...............                                                          [100%]
15 passed in 2.35s
```

Catalog and derived-view regressions:

```text
........................................................................ [ 51%]
...................................................................      [100%]
139 passed in 15.13s
```

The final combined invocation of all three requested files was also clean:

```text
........................................................................ [ 46%]
........................................................................ [ 93%]
..........                                                               [100%]
154 passed in 13.40s
```

All pytest runs used an explicit `--basetemp` under `/tmp`.

Required static checks:

```text
python -m py_compile \
  corpusgen/reasoning_v2/wikidata_source.py \
  corpusgen/reasoning_v2/catalog.py \
  corpusgen/reasoning_v2/renderers.py
# exit 0, no output

git diff --check
# exit 0, no output
```

A supplementary Ruff invocation was attempted but Ruff is not installed in
the local Python environment. Ruff was not a required gate; all requested
tests, compilation, and diff checks passed.

## Required behavior coverage

- Real deterministic tar.gz archives flow through source staging, derived-view
  build, the production `WikidataGraphCatalogSource`, and rendering.
- Round trip covers two complete-once edges plus one deterministic revisit.
- Wrong receipt, member, archive, outer split, training split, row, and edge
  key are rejected.
- A caller-constructed view object is rejected.
- The indexed-lookup test forbids all full derived-stream iterators, tar
  reopening, and the archive hashing helper during per-record rendering while
  asserting two indexed triple reads and three indexed alias reads per record.
- Packed uint16 token bytes and both sidecar byte streams are byte-identical
  across independent derived-view and production-catalog rebuilds.

## Self-review

- Authority closure: exact locator-key closure and catalog record-ID replay
  prevent ignored or reordered locator data.
- Source authority: archives are fully hashed once, pinned by inode/owner/mode/
  link-count/size/timestamps, and identity-checked around every render.
- Derived authority: every record is read through verified indexed APIs before
  and after rendering.
- Complexity: rendering performs constant-count indexed reads and no archive,
  training-stream, alias-stream, or distinct-edge iteration per record.
- Semantics: the complete canonical JSON payload is the sole factual surface;
  occurrence closure masks it only when its fact is routed, while padding/EOT
  remain supervised.
- Determinism: canonical JSON, fixed graph padding, catalog-bound target count,
  and rebuilt-view/catalog byte tests all agree.
- Scope: only the two assigned implementation/test files and this report were
  changed. The supplied untracked brief remains untouched.

No blocking defect was found in self-review.

## Commits

- `4483c7297a2ede3f66644209aca50dabf332b467` —
  `feat: add indexed Wikidata renderer`

## Review follow-up

### Status and implementation

PASS. The complete three-archive set is now identity-checked before and after
every Wikidata render. The set is the closed tuple authenticated from the
verified view receipt at renderer construction, so each boundary checks the
alias archive, inductive archive, and transductive archive regardless of which
training split selected the record. The selected archive SHA-256 remains bound
to the catalog record.

The public constructor and render signatures did not change:

```text
WikidataGraphRenderer(source_root: Path, wikidata_view: WikidataDerivedView)
render(record: CatalogRecord, routes: RouteIndex) -> ProductionRenderedRecord
```

FineWeb-Edu, FineMath, synthetic graph, `ProofEnvelope`,
`ProductionRenderedRecord`, and the shared exposure-renderer contracts are new
production code introduced by `4483c72`; they were not preserved from an
earlier committed or reviewed shared renderer. Follow-up behavioral coverage
now proves:

- FineWeb reads all three allowed 10BT paths and rejects an unapproved path
  before opening it;
- FineMath processes FineWeb, 4plus, then 3plus and preserves exact
  cross-deduplicated selection;
- synthetic graph replays the catalog seed and frozen world dimensions;
- non-NFC and non-finite payloads are rejected;
- duplicate synthetic source record IDs and oversized graph cores are
  rejected; and
- source hash and target-count commitment drift are rejected.

### Archive-set RED/GREEN evidence

Pre-render RED, with an inductive record while independently mutating the
alias archive and the non-selected transductive archive:

```text
FF                                                                       [100%]
2 failed in 0.74s
```

Both failures were the expected `Failed: DID NOT RAISE`, proving the old
record-selected check did not cover either archive.

Pre-render GREEN after replacing the selected-archive check with a complete-set
boundary check:

```text
..                                                                       [100%]
2 passed in 0.62s
```

Post-render RED mutated each same archive from inside token/sidecar
construction, after the pre-render boundary:

```text
FF                                                                       [100%]
2 failed in 0.69s
```

Final combined boundary GREEN:

```text
....                                                                     [100%]
4 passed in 1.16s
```

The six added non-Wikidata behavioral characterization cases passed in
`0.40s`. They exercised the already-introduced production behavior and did not
require a production-code change.

The complete focused renderer file now passes:

```text
.........................                                                [100%]
25 passed in 2.45s
```

### Producer design and plan amendments

The producer design now places the content-addressed derived-view authority
between immutable source staging and the production catalog. It requires the
catalog index and every Wikidata locator to bind the view receipt SHA-256, and
requires inner production bindings plus the outer receipt to bind the
inventoried canonical view receipt. The outer receipt additionally binds its
fixed release-relative path.

The implementation plan now records Task 3W before Task 3 and Task 5A. Task 3
consumes `WikidataGraphCatalogSource` over that verified view; Task 5A consumes
`lookup_training_triple()` and `lookup_alias()` and explicitly forbids legacy
iterators and per-record archive/full-stream scans. Task 7 carries the view
receipt through publication. No scientific quota or fixed source identity was
changed.

### Complete regression gate

The required eight-file invocation covered:

```text
tests/test_reasoning_v2_wikidata_source.py
tests/test_reasoning_v2_catalog.py
tests/test_reasoning_v2_renderers.py
tests/test_reasoning_v2_semantic.py
tests/test_reasoning_v2_source_lock.py
tests/test_tokenizer.py
tests/test_srgm_worlds.py
tests/test_current_sources.py
```

The first restricted-sandbox invocation reached `354 passed` and `17 failed`;
every failure had the same environmental root cause: fixture `git init` could
not create `.git/hooks` (`Operation not permitted`). A representative failing
test passed alone outside that filesystem sandbox (`1 passed in 0.45s`), and
the complete local, network-free rerun passed:

```text
........................................................................ [ 19%]
........................................................................ [ 38%]
........................................................................ [ 58%]
........................................................................ [ 77%]
........................................................................ [ 97%]
...........                                                              [100%]
371 passed in 25.50s
```

All pytest invocations used explicit `--basetemp` paths under `/tmp`; those
directories were removed afterward. No AWS or network operation was used.

Final static gates:

```text
python -m py_compile \
  corpusgen/reasoning_v2/wikidata_source.py \
  corpusgen/reasoning_v2/catalog.py \
  corpusgen/reasoning_v2/renderers.py
# exit 0, no output

git diff --check
git diff --check cb57697..HEAD
# both exit 0, no output
```

### Follow-up self-review

- Full-set closure: `_reverify_archives()` traverses the closed three-entry
  receipt-derived tuple at both render boundaries; alias and non-selected
  training drift can no longer pass silently.
- Race resistance: each check opens without symlink following and compares
  named/opened regular-file identity, owner, mode, link count, size, inode, and
  timestamps against the construction-time pin.
- Complexity: each render adds six constant-time metadata checks over a fixed
  three-file set. It does not rehash archive bytes or scan logical streams.
- Authority layering: source archive identity, selected-record SHA-256,
  locator/view commitment, indexed triple replay, and alias lookup remain
  independently checked.
- Compatibility: no public API, renderer version, locator schema, token
  fitting, semantic routing, or sidecar behavior changed.
- Scope: implementation/test changes, the two required producer documents,
  and this report are the only follow-up files changed. The supplied untracked
  brief and review patch remain untouched.

No blocking defect or unresolved concern was found in follow-up self-review.

### Follow-up commits

- `3ac34d216d695f1a7b278586e5c0ff119ef44dd6` —
  `fix: verify complete Wikidata archive set`
- `44fbe732dd54cefb20e0e907f2a9bb398a0e1546` —
  `docs: bind Wikidata view in producer flow`
