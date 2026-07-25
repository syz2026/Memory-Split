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
