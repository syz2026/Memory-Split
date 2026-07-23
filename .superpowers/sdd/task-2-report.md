# Task 2 report: current seven-lane compiler and packaged smoke corpus

## Outcome

Task 2 is implemented on `feat/relational-core` from base `ce203d9`.

Implementation commit:

- `445d090` — `feat: compile current seven-lane dataset`

The implementation does not add or modify cluster manifests, collaborator
assignments, or outer ZIP publication.

## Files changed

Compiler and graph protocol:

- Added `corpusgen/current_dataset.py`.
- Modified `corpusgen/graph_records.py`.
- Modified `corpusgen/graph_trace.py`.
- Modified `corpusgen/relational_build.py`.
- Modified `train/tokenizer.py`.
- Modified `organizer/graph_store.py`.
- Modified `evals/constrain.py`.
- Modified `evals/relational_generate.py`.

Command-line and runtime integration:

- Added `scripts/build_current_dataset.py`.
- Added `scripts/build_current_smoke.py`.
- Modified `scripts/build_relational_corpus.py`.
- Modified `scripts/relational_smoke_test.py`.
- Modified `scripts/run_relational_evals.py`.

Tests:

- Added `tests/test_current_dataset.py`.
- Modified `tests/test_graph_store.py`.
- Modified `tests/test_graph_trace.py`.
- Modified `tests/test_tokenizer.py`.
- Modified `tests/test_constrain.py`.
- Modified `tests/test_relational_generate.py`.
- Modified `tests/test_relational_smoke.py`.

Generated fixture:

- Added `fixtures/current-smoke/manifest.json` and `report.json`.
- Added `train.bin` plus aligned Dense, Split, and Random sidecars.
- Added the schedule, factual-span ledger, mask ledger, graph records,
  Wikidata provenance, source provenance, and their manifests.
- Added the packaged 12-slot evaluation graph and item.

The committed fixture has 17 files, 16 hash-listed artifacts plus
`manifest.json`, and is approximately 1.2 MiB. It contains no source archive.

## Key interface and implementation choices

### Locked scientific contract

- `CurrentBuildConfig(profile, scale, fact_load, data_seed, total_tokens)` is a
  frozen validated dataclass.
- `build_current_dataset(config, sources, out_dir) -> dict` accepts a verified
  Task 1 source root or source-manifest path for full builds. Full builds reject
  the in-memory smoke source adapter.
- `verify_current_dataset(out_dir, expected_profile) -> dict` verifies safe
  relative paths, complete artifact hashes, profile/scientific identity,
  `uint16` token size, aligned binary `uint8` sidecars, mask counts, and full
  scientific checks.
- Lane shares, context length, action/read limits, scale floors, fact-load
  labels, random-mask bins, ARC transform bounds, and Wikidata coverage modes
  are derived from `configs/current-dataset-lock.json` through Task 1's loader.
- Full profiles enforce the locked token floor, exact consumed-prefix maximum
  share deviation of `0.0025`, and complete-once accepted Wikidata coverage.
  Smoke is explicitly `scientific_result: false`.
- The 29M full profile uses a deterministic per-split/per-relation hash-balanced
  sample capped so one complete formatted pass fits the graph-lane budget.
  The 160M and 360M modes retain the complete accepted training graph.

### Streaming and scheduling

- Verified FineWeb, Wikidata training triples, aliases, and puzzle tasks stream
  into an on-disk SQLite spool. Large source collections and emitted-coverage
  bookkeeping do not require an in-memory corpus-sized set.
- Exact duplicate triples collapse on canonical `(QID, PID, QID)` identity.
- FineWeb excludes the locked holdout prefix and cycles verified non-holdout
  rows. Long source text is split only at lossless BPE boundaries.
- The scheduler is largest-token-deficit with stable lane order and measures
  the exact consumed prefix. All records are at most 1,024 tokens.
- Synthetic graph and reasoning lanes reuse deterministic SRGM worlds.
  Wikidata reasoning uses only functional addresses from the accepted training
  graph. Refinement records use deterministic corrupt/correct states.
- Puzzle records include the original and at most 63 unique transforms,
  deduplicated by canonical task hash and ordered by canonical parameter hash.

### Sidecars and ledgers

- `train.bin` is `uint16`; every sidecar is one `uint8` byte per token.
- Dense is all one.
- Split masks only externally routed `payload` spans.
- Every factual payload is recorded in `factual-span-ledger.jsonl`; mask actions
  are recorded in `mask-ledger.jsonl`.
- Random controls are selected deterministically within exact
  `(source, record_type, payload_token_length, packed_position_bin)` strata.
  Position uses ten bins and factual records do not cross a bin boundary.
- Random and Split have exactly equal zero-token counts. Query, rule, action,
  candidate-state, provisional-answer, final-answer, and boundary spans are
  never selected as factual masks.

### Paged graph protocol

- `GraphAddress` adds `page` and accepts legacy integer or canonical text entity
  identities. `GraphRow` adds page-aware set-valued `targets` while preserving
  scalar `target` and legacy page-zero JSON where possible.
- Wikidata objects are distinct and numerically QID-sorted. Stable largest
  prefixes become pages before a formatted record can exceed 1,024 tokens;
  functional rows remain page zero.
- Arbitrary PID text and decimal pages are framed by four reserved delimiters
  in IDs 50292–50295. `VOCAB_SIZE` remains exactly 50,304.
- `GraphActionTrie` constrains finite arbitrary-PID/page candidates. Exact page
  reads flow through the store and decoder.
- Current traces use 12 action slots, one through six training reads,
  deterministic post-HALT no-ops, and a hard ten-read decoding limit. Legacy
  six-slot, fixed-`r0`–`r15`, page-zero pilot records remain accepted.

### Publication and packaged smoke

- Builds write only to `.<name>.partial-<pid>`, fsync the tree, create and verify
  a relative hash-complete manifest, then atomically rename.
- An existing matching output verifies and returns; conflicting, malformed, or
  unsafe output fails closed.
- `fixtures/current-smoke/` contains exactly 131,072 token IDs, all seven lanes,
  all three aligned sidecars, graph/provenance/ledger/manifests, and no source
  archives.
- The smoke runner verifies the packaged fixture before use, trains Dense and
  Split for two CPU steps from the same packaged token stream, checks exact
  resume state and one resumed step, executes memory OFF and ON evaluation, and
  confirms fixture hashes remain unchanged.
- The legacy `SMOKE_FIXTURE` public constant is retained for existing packaging
  and preflight consumers; the current runner itself consumes packaged bytes.

## TDD and verification evidence

Representative red tests were observed before implementation:

- Initial current-dataset tests failed collection because
  `corpusgen.current_dataset` did not exist.
- Paged trace and constraint tests initially failed because
  `parse_serialized_action` and `GraphActionTrie` did not exist.
- The exact-page decoder test exposed string-QID coercion through
  `ValueError: invalid literal for int() with base 10: 'Q1'`.
- The balanced 29M sampling test first failed with
  `AttributeError: '_SourceSpool' object has no attribute
  'select_balanced_training_sample'`.

Final focused command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/test_current_dataset.py tests/test_graph_store.py \
  tests/test_graph_trace.py tests/test_tokenizer.py tests/test_constrain.py \
  tests/test_relational_generate.py tests/test_relational_build.py \
  tests/test_relational_smoke.py tests/test_data.py tests/test_trainer.py -q
```

Result: `118 passed in 12.31s`.

Complete repository command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q
```

Result: `506 passed, 2 deselected, 1 warning in 61.06s`. The warning is an
existing `dateutil` deprecation warning from `tests/test_stats.py`.

Final deterministic fixture check:

```bash
tmpdir=$(mktemp -d)
PYTHONDONTWRITEBYTECODE=1 python scripts/build_current_smoke.py --out "$tmpdir/a"
PYTHONDONTWRITEBYTECODE=1 python scripts/build_current_smoke.py --out "$tmpdir/b"
diff -ru "$tmpdir/a" "$tmpdir/b"
diff -ru fixtures/current-smoke "$tmpdir/a"
```

Result: both fresh outputs and the committed fixture were byte-identical.

Additional checks:

- `git diff --check` passed.
- IDE lint diagnostics reported no errors in the changed source and script
  files.
- The committed smoke manifest reports `profile: smoke`,
  `scientific_result: false`, 131,072 tokens, all seven lanes, complete-once
  fixture coverage, aligned sidecars, Dense all-one, and equal Split/Random
  zero counts.

## Requirement self-review

- Seven exact raw-token lane shares: implemented from the Task 1 lock.
- Full exact-prefix deviation `<= 0.0025`: enforced.
- Every smoke lane and non-scientific identity: enforced and verified.
- `uint16` tokens and three aligned binary `uint8` sidecars: enforced.
- Exact matched-random strata and equal zero mass: enforced and tested.
- Paged, set-valued graph records and 12-slot traces: implemented and tested.
- Arbitrary PID without vocabulary growth: implemented and tested.
- Bounded-memory verified-source build and complete-once before replay:
  implemented with SQLite and enforced for full builds.
- Deterministic source-archive-free smoke: generated, rebuilt twice, and
  byte-compared.
- Transactional/idempotent hash-complete publication: implemented and tested.
- Packaged CPU Dense/Split, resume, memory OFF/ON, fixture immutability:
  implemented and tested.
- Cluster manifests, collaborator assignments, and outer ZIP publication:
  intentionally untouched and excluded from the implementation commit.

## Concerns and follow-up

- A production multi-billion-token full build was not executed locally because
  the verified source root and cluster-scale storage/compute were outside this
  task's local verification environment. The full path is fail-closed on Task 1
  verification and is covered by focused contract tests, but its wall-clock and
  disk requirements should be measured during the first cluster build.
- Concurrent, untracked outer-publication files
  `artifacts/memorysplit-current-smoke-training.zip` and
  `scripts/package_quick_smoke_zip.py` were observed late in the worktree. They
  were not read, modified, staged, or committed as part of Task 2.
