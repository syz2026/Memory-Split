# Chinchilla-Scale Multi-Source Relational Corpus v2

**Date:** 2026-07-22  
**Status:** design approved in session; written-spec review pending  
**Scope:** a new corpus and evaluation version for the matched
Dense-versus-Split Transformer experiment

## 1. Goal

This design strengthens the existing Relational MemorySplit experiment with:

1. complete-once coverage of the allowed Wikidata5M training graph;
2. systematic out-of-distribution multi-fact composition;
3. licensed training tasks and augmentation ideas from Tiny Recursive Models
   (TRM); and
4. at least the classic Chinchilla floor of 20 processed target tokens per
   model parameter.

The claim-bearing hypothesis remains:

> At fixed standard-Transformer parameters, initialization, raw processed
> tokens, token order, optimizer schedule, and inference budget, removing
> direct next-token loss from selected arbitrary fact payloads improves
> acquisition of reusable relational reasoning procedures.

The new data version is `relational-v2`. Existing approximately
10-token-per-parameter runs remain valid pilots under their original manifests,
but their results must not be pooled with v2.

Where this document conflicts with
`2026-07-21-selective-recursive-graph-memory-design.md`, it supersedes that
document's corpus mixture, six-slot trace, atomic-only memory-row, token-budget,
OOD-suite, and primary-endpoint requirements. Unchanged model-matching and
platform-control requirements are inherited.

Twenty tokens per parameter is the classic Chinchilla rule of thumb. It is not
proof of the exact compute optimum for a structured, partially masked corpus.
This design guarantees at least that conventional raw-token floor while
preserving equal forward and backward compute between conditions.

## 2. Approved design decisions

The following choices are frozen by the design discussion:

- use a stratified deterministic mixture, not staged training or
  source-size-proportional sampling;
- verify all three pinned Wikidata5M archives;
- train only on official transductive-train and inductive-train triples;
- seal Wikidata5M validation and test splits for evaluation;
- include licensed ARC-AGI and ConceptARC training material plus generated
  TRM-style relational refinement traces;
- exclude Sudoku-Extreme and Maze-Hard because their dataset licenses are not
  declared clearly enough for redistribution;
- train relational traces at one through six hops;
- evaluate unseen relation compositions at two through six hops and unseen
  lengths at seven through ten hops; and
- use update-aligned token budgets of at least 20 tokens per parameter.

## 3. Source contracts

### 3.1 FineWeb-Edu natural text

Use the existing frozen snapshot:

- repository: `HuggingFaceFW/fineweb-edu`;
- revision: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`;
- source shards:
  `sample/10BT/000_00000.parquet`,
  `sample/10BT/001_00000.parquet`, and
  `sample/10BT/002_00000.parquet`;
- materialized rows: 2,182,000;
- materialized JSONL bytes: 10,498,726,596; and
- materialized JSONL SHA-256:
  `f89e844723887daa9714d906bf148bfe6931c2f063e94f791a1e861fefc668ca`.

Retain the first 64 records as the shared natural-text holdout. Training cycles
the remaining records in stable shard and row order while skipping exact
duplicate source strings.

### 3.2 Complete Wikidata5M graph source

The named source scope is `wikidata5m-graph-3archive-v1`. It means the complete
three-archive graph distribution pinned by this project, not an official
Wikimedia dump and not every historical Wikidata field.

Use `intfloat/wikidata5m` at revision
`6b2b09672129e280c0c9da97ab58154e9d535e6b`:

- `wikidata5m_alias.tar.gz`
  - 197,449,751 bytes
  - SHA-256
    `0330f580c9f7a57cbad949ac380835fdd2a2e14d96cc0f13fc435401d6b463a8`
- `wikidata5m_inductive.tar.gz`
  - 167,247,416 bytes
  - SHA-256
    `955081232cc2de859710bfe3a147f7d8314524010fe5f8c420bb74fdfee4f42a`
- `wikidata5m_transductive.tar.gz`
  - 168,258,214 bytes
  - SHA-256
    `383160990b41c0905fc03f4a8afbb9b12be1ca3591e026bde6cdc94a59542597`

Only regular files may be extracted. Archive path traversal, links, duplicate
outputs, unexpected members, byte drift, and hash drift are fatal.

The protected training graph consists of every distinct valid triple in:

- transductive train; and
- inductive train.

Exact duplicate rows are counted in the audit and collapsed because they do not
add a new fact. No other accepted distinct training triple may be dropped.
Inductive validation and test remain sealed. Transductive or other validation
and test files, if present in the verified archives, also remain sealed.

The alias archive supplies deterministic display strings. Use the first
non-empty alias after NFKC normalization and whitespace collapse; fall back to
the QID or PID when no usable alias exists. A missing alias must not cause a
triple to be omitted.

This source does not provide a separately locked description corpus. The design
therefore makes no claim to include Wikidata descriptions. Structured Wikidata
data is handled under CC0 1.0, with the CC0 text and source citation carried
into every portable corpus artifact.

### 3.3 TRM and licensed puzzle sources

Pin the TRM augmentation reference implementation to:

- repository: `SamsungSAILMontreal/TinyRecursiveModels`;
- revision: `c01103738605ba39d1430519b1ee0c62f4c707f8`; and
- code license: MIT.

TRM has no natural-language pretraining corpus. Its data is supervised grid
puzzles. The v2 auxiliary lane uses only sources with a clear redistribution
license:

- ARC-AGI-1, revision
  `399030444e0ab0cc8b4e199870fb20b863846f34`, Apache-2.0;
- ARC-AGI-2, revision
  `f3283f727488ad98fe575ea6a5ac981e4a188e49`, Apache-2.0; and
- ConceptARC, revision
  `0e67da6af879e4bad3d7cd3c196e8d551b445725`, MIT.

Only official training tasks and their outputs may enter the corpus. Canonical
task hashes must remove cross-repository duplicates and exclude any task whose
hash appears in an official evaluation directory. ConceptARC is treated wholly
as training auxiliary data, so v2 must not report ConceptARC benchmark results.
No ARC evaluation solution may be ingested.

Sudoku-Extreme and Maze-Hard are excluded. The source repositories' code
licenses do not establish a redistribution license for all puzzle rows.

## 4. Frozen corpus mixture

Every final consumed prefix uses these raw-target-token shares:

- 40% FineWeb-Edu natural text;
- 20% complete Wikidata5M training-graph serialization;
- 10% controlled synthetic graph exposure;
- 15% synthetic one-to-six-hop composition;
- 7.5% Wikidata functional-path reasoning;
- 2.5% generated TRM-style relational refinement traces; and
- 5% licensed ARC-AGI and ConceptARC tasks.

The mixture retains the current experiment's 30% total graph exposure and 25%
total relational reasoning. The 5% puzzle auxiliary comes from the former 45%
natural-text lane.

Mixture percentages are measured over causal target positions in the exact
prefix consumed by training, before condition-specific target weights. A
deterministic largest-deficit scheduler chooses the next component. Whole
records are never truncated merely to hit a quota. The final consumed prefix
must be within 0.25 percentage points of every target share.

All conditions for a scale, load, and data seed consume the same `train.bin`
bytes, packing boundaries, component schedule, and record order. Condition is
not an input to corpus generation.

### 4.1 Complete-once guarantee

At 160M and 360M, the Wikidata graph lane first emits every accepted distinct
training triple exactly once in deterministic order. It replays records only
after that pass completes. Each grouped or paginated record carries the source
split and source-row provenance needed to prove coverage.

Before corpus freezing, a source-only tokenizer audit must prove that the
complete pass fits within the 160M run's 20% Wikidata lane. If it does not fit,
the build stops and this design must be revised; the builder may not truncate
Wikidata, silently change serialization, or steal tokens from another lane.

The 29M tier is a pipeline and learnability gate, not a complete-Wikidata
scientific result. Its Wikidata lane uses a deterministic hash sample balanced
by source split and relation. Its report must state explicitly that it does not
provide complete-once coverage.

## 5. Record contracts

### 5.1 Wikidata graph records

Parse only strict `QID<TAB>PID<TAB>QID` triples. Group values by canonical
`(subject QID, property PID)` and sort distinct object QIDs numerically.

Serialize compact QID/PID identity and deterministic aliases. The identity,
not an ambiguous label, is authoritative. A grouped record must encode every
member so the coverage ledger can map each accepted triple to an emitted token
span.

Multivalued rows are canonical sorted object lists. An oversized list is split
into deterministic pages with a page count and stable member boundaries; no
member is dropped. The nonparametric organizer exposes the same pages. Path
composition samples only addresses with exactly one object, keeping each path
step deterministic. Set-valued direct retrieval is evaluated separately.

No record may exceed the 1,024-token context after formatting. Records that
would exceed it must be paginated before packing, never silently truncated.

### 5.2 Synthetic and real path records

Each path record contains:

- graph facts or references to graph rows;
- a natural-language query;
- up to 12 fixed action slots;
- exact memory returns;
- a candidate state after each action; and
- a final answer.

The 12-slot protocol supports ten reads followed by `HALT`, with one spare slot.
Training paths use one through six reads. After `HALT`, remaining training
slots are deterministic no-ops. Evaluation may require seven through ten
reads. This extends sequence formatting without adding trainable parameters.

Every path and answer is checked by a deterministic symbolic executor before
publication.

### 5.3 TRM-style relational refinement

Generated refinement records transfer TRM's repeated-state-correction idea into
the existing autoregressive Transformer:

`facts, query, candidate state -> action, memory return, corrected state`.

Corruptions are generated symbolically and have one deterministic correction.
Every action, corrected state, and final answer is supervised. These are
relational language records, not claims that the original TRM puzzle data
contains reasoning traces.

### 5.4 ARC and ConceptARC serialization

Serialize demonstrations, query grids, and output grids with explicit row,
column, and task boundaries. Apply deterministic color permutations,
rotations/reflections, and translations that preserve the task solution.

Emit the original plus at most 63 unique transformations per canonical task,
ordered by transformation-parameter hash. If fewer than 63 unique transforms
exist, emit all of them. Cycle this frozen set deterministically to fill the 5%
lane; source size must never determine mixture weight. All conditions receive
identical full supervision on this lane.

## 6. Split-training intervention

### 6.1 Frozen route manifest

Build one route manifest from training data before model initialization.
Validation/test facts and model outcomes cannot influence routing.

Rules, relation semantics, graph-control syntax, and structural schema are
always internal. Fact payloads use the existing fixed costs:

\[
C_{\mathrm{predict},i} =
\frac{H_i}{\max(E_i,1)}
\]

\[
C_{\mathrm{external},i} =
1.0 + 0.25Q_i + 0.25D_i.
\]

`H_i` is payload self-information under train-only source statistics, `E_i` is
the exact scheduled exposure count, `Q_i` is expected train-only read demand,
and `D_i` is expected hop contribution. Graph centrality affects routing only
through train-only read and hop demand. Fact `i` routes external when
`C_predict,i > C_external,i`.

The constants are not recalibrated from protected outcomes. The canonical
feature table, decision, reason, and route-policy hash are published.

### 6.2 Target-weight sidecars

Tokenize and pack once while writing a factual-span ledger containing:

- source and source-row identity;
- canonical fact or list-member identity;
- route decision;
- record type;
- target-token length; and
- packed target positions.

Generate aligned sidecars:

- **Dense:** every target weight is one.
- **Split:** zero only direct occurrences and memory returns of externally
  routed payloads.
- **Random-mask:** factual payloads remain weighted; equal token mass of
  non-factual spans is masked within the same source, record type,
  payload-length bin, and packed-position bin.

Rules, queries, actions, candidate-state syntax, and final answers remain
supervised in every arm. ARC and ConceptARC have identical all-one target
weights in every arm.

The mask audit must prove:

- byte alignment with `train.bin`;
- no unmasked direct occurrence of an external payload;
- no masked rule, action, query, or final-answer target;
- total random-mask mass within 1% of Split;
- source-conditioned mass within 1%;
- length and position histograms within 1%; and
- neither Wikidata nor synthetic facts account for all Split masking.

### 6.3 Loss and Chinchilla accounting

Use all-position normalized causal loss:

\[
\mathcal L =
\frac{1}{T}
\sum_{t=1}^{T}
w_t[-\log p_\theta(x_t\mid x_{<t})].
\]

`T` is every raw causal target position, including positions whose Split or
Random weight is zero. The Chinchilla token count is the same raw processed
target count. Masking therefore neither increases the weight of remaining
targets nor changes measured training compute.

## 7. OOD composition contract

OOD examples are evaluation-only. Training is strengthened to make the
holdouts meaningful; their answers are not added to the corpus.

### 7.1 Frozen composition partition

Enumerate supported relation bigrams and trigrams from train-only graphs.
Canonicalize each relation sequence and assign it by SHA-256 under a published
partition salt.

Reserve 20% of eligible compositions for OOD evaluation, subject to:

- every individual relation occurs in training reasoning traces;
- every held-out sequence has enough executable paths for its evaluation
  quota;
- no held-out bigram or trigram occurs in a training reasoning trace; and
- direct graph exposure may still contain the component edges.

The last condition makes this a composition holdout rather than a fact
holdout. The partition and all support counts are frozen before model training.

### 7.2 Training curriculum

Within the reasoning lanes:

- first 20%: one- and two-hop paths;
- next 30%: one through four hops; and
- final 50%: one through six hops.

Balance lengths by emitted target tokens rather than record count. Synthetic
and Wikidata path generators obey the same curriculum, but only functional
Wikidata rows may form deterministic paths.

### 7.3 Evaluation strata

Freeze four synthetic path strata, each with 10,000 counterfactual pairs:

1. **IID:** lengths two through six, seen compositions;
2. **composition OOD:** lengths two through six, held-out compositions;
3. **length OOD:** lengths seven through ten, seen relation transitions; and
4. **joint OOD:** lengths seven through ten, held-out compositions.

Every pair contains an original item and a twin in which exactly one supporting
edge changes and the correct answer flips. A pair is correct only when both
members are correct.

Add 5,000 source-backed paths from each usable sealed Wikidata inductive
validation/test split. Real counterfactuals replace one supporting object with
a different alias-valid object from the same relation and degree bucket. The
design does not claim type compatibility because the pinned source has no type
metadata. If a split cannot supply its quota without leakage or ambiguity, its
secondary result is unavailable rather than backfilled from training.

Generate entity-renamed and graph-isomorphic variants from every synthetic
stratum. All evaluation files, gold paths, pair mappings, and hashes are frozen
before protected checkpoints are inspected.

## 8. Token and run budgets

The model sizes and update size remain:

- 29M: 28,969,216 parameters;
- 160M: 162,220,800 parameters;
- 360M: 356,033,536 parameters; and
- 524,288 target tokens per optimizer update.

The minimum update-aligned v2 budgets are:

- **29M:** 1,106 updates and 579,862,528 tokens
  (20.0165 tokens/parameter);
- **160M:** 6,189 updates and 3,244,818,432 tokens
  (20.0025 tokens/parameter); and
- **360M:** 13,582 updates and 7,120,879,616 tokens
  (20.0006 tokens/parameter).

Retain the existing protected run structure:

- 15 runs at 160M across low/high fact load, three seeds, Dense/Split, and the
  high-load Random control; and
- 6 runs at 360M across three seeds and Dense/Split.

Those 21 protected runs process exactly 91,397,554,176 raw target tokens.

The 29M gate runs six matched diagnostics: full-v2 Dense/Split,
no-ARC Dense/Split with token-matched FineWeb replacement, and no-refinement
Dense/Split with token-matched standard relational records. These ablations
measure component contribution; they do not authorize post-hoc mixture
selection. Full v2 must pass its preregistered gates or the protected launch
stops for redesign.

## 9. Evaluation and guardrails

### 9.1 Primary endpoint

The v2 primary endpoint is equal-weight counterfactual pair accuracy over:

- synthetic composition OOD; and
- synthetic joint OOD.

Both Dense and Split receive the same fresh graph and memory-ON evidence. IID,
length-only OOD, the prior date/equality tasks, Wikidata, ARC, and natural-text
metrics are secondary or guardrail endpoints.

Report exact path, per-hop action, referent, malformed action, `MISS`,
excess-read, halt-depth, and gold-path-replay accuracy separately.

### 9.2 Memory and intervention controls

Evaluate the full memory factorial:

1. Dense, memory OFF;
2. Dense, memory ON;
3. Split, memory OFF; and
4. Split, memory ON.

Run these negative controls:

- shuffled memory;
- relevant-edge swap;
- irrelevant-edge swap;
- gold-path replay;
- no-query;
- entity renaming;
- graph isomorphism;
- set-valued page-order permutation; and
- memory OFF.

### 9.3 Required instrument gates

All of the following must pass:

- global external factual-token mass is 40–60%;
- synthetic and Wikidata factual-token mass are each 20–80% external;
- at least 80% of low-use high-information payloads route external;
- at least 80% of rules and top-centrality facts route internal;
- Dense four-way recognition lower 95% bound exceeds 30% at 25% chance;
- Split store-OFF recognition upper 95% bound is below 30%;
- Split memory-ON factual recall is within two points of Dense memory-OFF;
- Split internal-rule accuracy is within two points of Dense;
- shared FineWeb bits per byte is non-inferior within 1% relative;
- every source and mask-ledger audit passes; and
- no training task or triple appears in a sealed evaluation partition.

Failure of an instrument gate makes the corresponding result invalid, not
negative.

### 9.4 Frozen verdict

Let `delta_scale,seed` be Split minus Dense on the v2 primary endpoint.
Validate the hypothesis in this regime only if:

1. all three 360M deltas are positive;
2. mean 360M delta exceeds
   `max(2 percentage points, 2 × pooled seed sigma)`;
3. both primary OOD strata have positive mean 360M deltas;
4. the paired high-load-minus-low-load difference-in-differences has a 95%
   lower bound above zero;
5. Split exceeds Random at 160M high load by more than one point in every
   paired seed; and
6. every required instrument gate passes.

Reject a practically meaningful effect only when every instrument gate passes
and:

1. the upper 95% bound on mean 360M delta is below two points;
2. the upper 95% bound on the load difference-in-differences is at or below
   zero; and
3. selective masking does not beat random masking by more than one point.

All other outcomes are inconclusive.

Real Wikidata scores are secondary because FineWeb can contain the same public
facts. Shared TRM auxiliaries can improve both arms but cannot by themselves
validate MemorySplit.

## 10. Provenance and failure handling

Before any protected build, write one canonical provenance manifest containing:

- source repository, revision, member, byte count, and SHA-256 locks;
- extracted member inventory and hashes;
- raw, valid, malformed, duplicate, and emitted row counts;
- distinct entity, relation, and triple counts by Wikidata split;
- alias normalization and fallback counts;
- train/evaluation overlap audits;
- ARC/ConceptARC canonical task hashes and license notices;
- composition partition salt, sequences, and support counts;
- route-policy inputs, decisions, and SHA-256;
- tokenizer revision and vocabulary hash;
- record-format and scheduler algorithm versions;
- complete-once coverage proof;
- packed token, sidecar, ledger, and evaluation hashes;
- code commit and clean-tree assertion; and
- exact run-manifest hash.

Corpus construction is offline after source staging. Temporary outputs are
written outside final paths and atomically renamed only after verification.
Resume refuses any mismatch in source, data, policy, code, model, optimizer, or
run-manifest provenance.

Fail closed on:

- missing, extra, unsafe, or drifted source files;
- malformed source rows;
- train/evaluation leakage;
- incomplete Wikidata training coverage at 160M/360M;
- mixture quota drift;
- overlength records that were not paginated;
- unsatisfied OOD quotas;
- route or mask guardrail failure;
- sidecar misalignment;
- nondeterministic rebuilds; or
- insufficient disk space for source audit, packing, and checkpoints.

## 11. Execution isolation

All v2 artifacts and jobs use a new namespace such as `relational-v2-*`.
Existing local and FarmShare jobs remain v1 pilots and must not be resumed from
v2 data or checkpoints.

No protected 160M or 360M job may launch until:

1. all source locks and licenses are staged;
2. the complete source audit passes;
3. deterministic rebuild tests pass;
4. the six 29M diagnostics finish;
5. the full-v2 29M pair passes learnability, language, routing, masking, and
   resume gates; and
6. the final corpus, evaluation, and run manifests are signed off by hash.

The portable run bundle contains code, locks, manifests, fixtures, tests, and
launchers, but not the full corpora or checkpoints. Runtime paths remain
relative to `DATA_ROOT` and `OUT_ROOT`.

## 12. Acceptance criteria

The design is implemented only when:

1. the same command rebuilds byte-identical source audits, partitions, corpora,
   sidecars, and evaluation files from pinned inputs;
2. 160M and 360M manifests meet their exact 20× token floors;
3. every allowed distinct Wikidata training triple has an emitted provenance
   entry before replay;
4. no sealed Wikidata or ARC evaluation answer enters training;
5. all four synthetic OOD strata satisfy their pair quotas and symbolic checks;
6. Dense, Split, and Random consume identical tokens and differ only through
   audited target weights;
7. all routing, masking, loss, provenance, resume, and platform tests pass; and
8. the analyzer can return only `validate`, `reject`, `inconclusive`, or
   `invalid` under the frozen rules above.

## 13. Explicit non-goals

This version does not add:

- recurrent neural blocks;
- a trainable selector or router;
- graph neural networks;
- semantic retrieval;
- natural-world entity linking;
- claims of clean closed-book Wikidata memorization;
- ARC, ConceptARC, Sudoku, or maze benchmark claims;
- claims that 20 tokens per parameter is the exact optimum for this mixture;
  or
- mechanistic claims about where any freed capacity resides.

The models remain matched vanilla Transformers. TRM contributes licensed
auxiliary tasks and a data-level repeated-refinement format, not a new neural
architecture.
