# V2 Puzzle Authority and Objective Planning Design

## Decision

Use a v2-native direct puzzle scan and per-record lookup over the authenticated
content-addressed source tree. Do not create a persistent puzzle derived view
and do not add a legacy manifest or `git/` namespace to the immutable tree.

Source discovery and objective allocation are separate:

```text
SourceLock + direct staged files
  -> PuzzleSourceScan (training/evaluation classification + canonical dedup)
  -> ObjectiveSourceTargetPlan (finite cap + whole-record allocation)
  -> ObjectiveAuxiliaryCatalogSource
  -> ObjectiveAuxiliaryRenderer direct lookup
```

## Puzzle authority

`corpusgen/reasoning_v2/puzzle_source.py` freezes:

- source order: `arc_agi_1`, `arc_agi_2`, `conceptarc`;
- ARC training prefix `data/training/`;
- ARC evaluation prefix `data/evaluation/`;
- ConceptARC training prefix `corpus/` and no evaluation prefix;
- case-sensitive `.json` selection;
- strict UTF-8/duplicate-key/non-finite JSON rejection;
- canonical task hashing as compact sorted-key JSON without a newline;
- evaluation-overlap rejection by canonical task hash;
- cross-source canonical deduplication with first source/path winner; and
- exact-byte duplicate audit.

Discovery iterates only `SourceEntry.files` in frozen source/path order. It
never enumerates the filesystem. `PuzzleSourceScan` contains compact locators,
evaluation hashes, duplicate evidence, policy SHA-256, and scan SHA-256; it
contains no raw tasks or answers.

Each scan/read authenticates the source lock and complete root. A selected JSON
read uses descriptor-relative no-follow opens, owner/mode/link/type checks,
pre/post descriptor and named-entry replay, exact byte count/raw SHA-256,
strict parsing, canonical task SHA-256, and exact test-example count.

## Objective catalog ownership

Corpus-wide allocation belongs in
`corpusgen/reasoning_v2/objective_source.py`, not the renderer.

The shared finite ARC-family ceiling is:

```text
arc_ceiling = floor(full_corpus_total_targets * 25 / 10_000)
```

Finite records are `(accepted puzzle task, test_index)` in frozen
source/path/numeric-test order and are used at most once.

### Whole-record boundary rule

The objective lane receives its compact balanced `TargetLengths` sequence.

1. Traverse finite records and target lengths together from index zero.
2. Assign the next target length to the next finite record only when
   `finite_realized_targets + length <= arc_ceiling`.
3. Stop finite assignment before the first crossing; never skip a length to
   fit a later record and never cycle a finite record.
4. Define
   `finite_unused_targets = arc_ceiling - finite_realized_targets`.
5. Every remaining target length belongs to one of the five unbounded sources
   in exact policy order:
   `deepmind_mathematics_generator`, `clrs_text`, `ruletaker`, `prontoqa`,
   `reasoning_gym_exact_answer`.
6. Compute equal-share Hamilton target quotas over the exact sum of remaining
   lengths.
7. For each remaining length in sequence order, assign it to the source with
   the largest current target deficit
   `quota[source] - assigned[source]`; break ties by policy order.
8. The assigned target total is exact. Each source's absolute deviation from
   its Hamilton target quota must be no greater than the largest assigned
   record length; otherwise fail before emitting drafts.
9. Teacher-generated targets are exactly zero.

The planner records desired and realized per-source targets and the complete
source assignment sequence. These are deterministic commitments, not
approximate logs.

## Puzzle catalog locators

Each selected test example uses bytewise-ordered locator keys:

```python
(
    ("answer_sha256", canonical_task_sha256(expected_output)),
    ("canonical_task_sha256", task.canonical_task_sha256),
    ("path", task.path),
    ("puzzle_policy_sha256", scan.policy_sha256),
    ("puzzle_scan_sha256", scan.sha256),
    ("split", "train"),
    ("test_index", test_index),
)
```

`source_byte_sha256` is the locked JSON file hash. Source ID + path + file hash
+ catalog source-lock hash resolve repository, revision, license, and byte
count without duplicating them in every locator.

The catalog's reserved-path check recognizes only the exact ConceptARC
`conceptarc/corpus/` training authority; unrelated paths containing evaluation
or test-like components still fail.

## Renderer and registry integration

`ObjectiveAuxiliaryRenderer` stores `SourceLock`, source root, expected
generator commit, and the compact scan. It requires every puzzle record to bind
an accepted locator and rereads only that selected JSON. It independently
checks raw hash, canonical task hash, policy, scan, split, test index, and
answer commitment before producing an exact-answer or solver proof.

`build_renderer_registry` retains its live Wikidata-session parameter. It
requires the view and source lock to bind the same source-lock SHA-256 and
generator commit. The live view remains open for registry use.

`iter_supported_reasoning_records` is the sole verified-synthetic-multihop
source sequence. Registry identities or manifests may contain renderer
versions, proof family, proof SHA-256, replay status, answer commitment,
policy commitment, and scan commitment. They never contain raw
`ProofEnvelope.premise_bytes` or raw expected answers.

## Testing

Tests use real staged v2 roots and cover strict source parsing, direct-layout
authority, evaluation overlap, raw/canonical duplicates, deterministic winner
order, ConceptARC policy, descriptor replacement, bounded scan memory, selected
file-only lookup, exact finite cap, no finite cycling, whole-record boundary,
largest-deficit tie breaks, exact totals, bounded deviation, teacher zero,
all eight objective source IDs, catalog acceptance, renderer replay, registry
identity, and live Wikidata-session lifetime.

## Scope

This design does not alter `SourceLock`, source-root bytes, the legacy
`current_sources` contract, fixed source identities, Wikidata view contracts,
or Task 6 implementation.
