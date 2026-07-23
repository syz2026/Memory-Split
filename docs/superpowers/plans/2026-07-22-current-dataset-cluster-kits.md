# Current Dataset Cluster Kits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every collaborator ZIP self-contained for offline smoke training
on the current seven-lane dataset and capable of staging, building, verifying,
and launching the full shared dataset under cluster `DATA_ROOT`.

**Architecture:** Keep full multi-billion-token corpora out of the four ZIPs.
Each ZIP contains one byte-identical compiled smoke corpus, pinned source locks,
the current compiler, and assignment-specific launch helpers. Full sources are
staged once per shared filesystem, full corpora are built before GPU training,
and launchers refuse unverified or historical corpus manifests.

**Tech Stack:** Python 3.11+, NumPy memmaps, SQLite, tiktoken assets vendored in
the repository, Hugging Face `hf download`, Git, Slurm, pytest, deterministic
tar/ZIP publication.

## Global Constraints

- Dataset ID is exactly `relational-chinchilla`; do not introduce numbered
  dataset labels.
- The only normative design is `Memory-split-design.md`.
- FineWeb-Edu uses revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` and the existing frozen
  2,182,000-row JSONL hash.
- Wikidata5M uses all three archives from revision
  `6b2b09672129e280c0c9da97ab58154e9d535e6b`; only transductive-train and
  inductive-train enter training.
- ARC-AGI-1, ARC-AGI-2, and ConceptARC use the exact commits in
  `Memory-split-design.md`; evaluation-task hashes are excluded.
- Token mixture is exactly 40/20/10/15/7.5/2.5/5 percent over raw causal target
  positions.
- Final token floors are 579,862,528 at 29M, 3,244,818,432 at 160M, and
  7,120,879,616 at 360M.
- `train.bin` remains `uint16`; Dense/Split/Random sidecars remain aligned
  `uint8`; all-position loss is unchanged.
- Every condition for one corpus reads identical `train.bin`; only the selected
  weight sidecar differs.
- Full source staging and corpus construction happen before GPU training.
  Training jobs have no network dependency and never mutate dataset bytes.
- The packaged smoke profile is explicitly non-scientific and uses 131,072
  tokens.
- Protected launch remains blocked until full-source, complete-coverage,
  leakage, mask, route, and platform gates pass.

---

### Task 1: Freeze and stage current dataset sources

**Files:**
- Create: `configs/current-dataset-lock.json`
- Create: `sources/wikidata5m.lock.json`
- Create: `sources/current-dataset-licenses.json`
- Create: `sources/Wikidata-CC0-1.0.txt`
- Create: `corpusgen/wikidata5m.py`
- Create: `corpusgen/current_sources.py`
- Create: `scripts/stage_current_sources.py`
- Create: `tests/fixtures/current_sources/`
- Create: `tests/test_current_sources.py`

**Interfaces:**
- Produces:
  `load_dataset_lock(path: Path) -> CurrentDatasetLock`
- Produces:
  `stage_current_sources(lock: CurrentDatasetLock, data_root: Path,
  *, execute: bool) -> dict`
- Produces:
  `verify_current_sources(lock: CurrentDatasetLock,
  source_root: Path) -> dict`
- Produces strict iterators:
  `iter_training_triples`, `iter_aliases`, and `iter_puzzle_tasks`.

- [ ] **Step 1: Write source-contract tests**

Tests must construct tiny local tarballs and Git-style source trees, then assert:

```python
def test_training_source_seals_evaluation_splits(tmp_path):
    manifest = stage_fixture_sources(tmp_path)
    assert manifest["wikidata"]["training_splits"] == [
        "inductive_train",
        "transductive_train",
    ]
    assert manifest["wikidata"]["sealed_splits"] == [
        "inductive_test",
        "inductive_valid",
    ]


def test_arc_evaluation_hash_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="evaluation task"):
        stage_fixture_sources(tmp_path, duplicate_eval_into_train=True)
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/test_current_sources.py -q
```

Expected: collection fails because `corpusgen.current_sources` does not exist.

- [ ] **Step 3: Port strict Wikidata archive handling**

Selectively port `WikidataLock`, `verify_archives`, safe extraction,
`iter_triples`, and alias canonicalization from SRGM merge
`02cb071e104982865b1bb755876ad4023724178e`. Publish extraction only by
atomic rename from a private temporary directory.

- [ ] **Step 4: Implement recursive puzzle-tree locks**

Canonical task identity is SHA-256 over UTF-8 canonical JSON:

```python
json.dumps(task, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
```

Reject a training task if its hash appears in any official evaluation
directory. Record repository, commit, path, bytes, hash, and license for every
accepted file.

- [ ] **Step 5: Implement dry-run-first staging CLI**

The command:

```bash
python scripts/stage_current_sources.py \
  --data-root "$DATA_ROOT" \
  --cache-dir "$DATA_ROOT/hf-cache"
```

prints its pinned `hf download` and Git archive operations without writing.
`--execute` performs the downloads and writes:

```text
$DATA_ROOT/relational-chinchilla/sources/source-manifest.json
```

- [ ] **Step 6: Run focused source tests**

Expected: all current-source fixture tests pass without network.

- [ ] **Step 7: Commit**

```bash
git add configs/current-dataset-lock.json sources corpusgen/wikidata5m.py \
  corpusgen/current_sources.py scripts/stage_current_sources.py \
  tests/fixtures/current_sources tests/test_current_sources.py
git commit -m "feat: add current dataset source staging"
```

---

### Task 2: Compile the seven-lane corpus and packaged smoke data

**Files:**
- Create: `corpusgen/current_dataset.py`
- Create: `scripts/build_current_dataset.py`
- Create: `scripts/build_current_smoke.py`
- Create: `tests/test_current_dataset.py`
- Create: `fixtures/current-smoke/` (generated)
- Modify: `corpusgen/relational_build.py`
- Modify: `corpusgen/graph_records.py`
- Modify: `corpusgen/graph_trace.py`
- Modify: `train/tokenizer.py`
- Modify: `organizer/graph_store.py`
- Modify: `evals/constrain.py`
- Modify: `evals/relational_generate.py`
- Modify: `scripts/build_relational_corpus.py`
- Modify: `scripts/relational_smoke_test.py`
- Modify: `tests/test_graph_store.py`
- Modify: `tests/test_graph_trace.py`
- Modify: `tests/test_tokenizer.py`
- Modify: `tests/test_constrain.py`
- Modify: `tests/test_relational_generate.py`
- Modify: `tests/test_relational_smoke.py`

**Interfaces:**
- Produces:
  `CurrentBuildConfig(profile, scale, fact_load, data_seed, total_tokens)`
- Produces:
  `build_current_dataset(config, sources, out_dir) -> dict`
- Produces:
  `verify_current_dataset(out_dir, expected_profile) -> dict`
- Consumes the verified `source-manifest.json` from Task 1.

- [ ] **Step 1: Write seven-lane and byte-alignment tests**

```python
EXPECTED = {
    "fineweb": 0.40,
    "wikidata_graph": 0.20,
    "synthetic_graph": 0.10,
    "synthetic_reasoning": 0.15,
    "wikidata_reasoning": 0.075,
    "relational_refinement": 0.025,
    "puzzle_auxiliary": 0.05,
}


def test_smoke_build_has_current_lanes_and_aligned_sidecars(tmp_path):
    report = build_fixture_current_dataset(tmp_path, total_tokens=131_072)
    assert report["target_shares"] == EXPECTED
    size = (tmp_path / "train.bin").stat().st_size // 2
    for condition in ("dense", "split", "random"):
        assert (tmp_path / f"{condition}.weights.bin").stat().st_size == size
```

- [ ] **Step 2: Verify tests fail**

Expected: `corpusgen.current_dataset` is missing.

- [ ] **Step 3: Implement deterministic lane scheduler**

Use a largest-token-deficit scheduler with stable lane order. Measure shares on
the exact consumed prefix. Full builds fail above 0.25 percentage-point
deviation; smoke builds report their finite-record deviation but must contain
every lane.

- [ ] **Step 4: Implement source records**

- FineWeb: cycle verified non-holdout rows.
- Wikidata graph: emit every distinct accepted training triple before replay;
  QID/PID fallback prevents alias-based drops.
- Synthetic graph/reasoning: reuse existing deterministic worlds.
- Wikidata reasoning: use only train-graph functional addresses.
- Relational refinement: emit deterministic corrupted/corrected candidate
  states.
- Puzzle auxiliary: original plus at most 63 unique hash-ordered transforms per
  task.

- [ ] **Step 5: Implement paged graph actions and twelve-slot traces**

Represent each graph address as canonical entity ID, relation ID, direction,
and page. Group multivalued rows in stable numeric-QID order and page them
before any record exceeds 1,024 tokens. Unique functional rows remain page
zero. Serialize arbitrary PIDs as delimited identity text while keeping the
50,304-token vocabulary unchanged. Extend constrained generation and the
organizer interface to twelve action slots, exact page reads, deterministic
post-HALT no-ops, and at most ten reads.

- [ ] **Step 6: Implement current sidecar ledger**

Write every factual span to `factual-span-ledger.jsonl`. Split masks only routed
payload spans. Random candidates match source, record type, exact payload-token
length, and one of ten packed-position bins. Rules, actions, queries, and final
answers are never masked.

- [ ] **Step 7: Implement atomic output publication**

Build into `.<name>.partial-<pid>`, fsync files, write and verify
`manifest.json`, then atomically rename. Existing matching output verifies and
returns; conflicting output fails.

- [ ] **Step 8: Build the deterministic smoke fixture**

Run twice and require byte-identical output:

```bash
python scripts/build_current_smoke.py --out /tmp/current-smoke-a
python scripts/build_current_smoke.py --out /tmp/current-smoke-b
diff -ru /tmp/current-smoke-a /tmp/current-smoke-b
```

Copy verified output into `fixtures/current-smoke/`.

- [ ] **Step 9: Run smoke training**

Train Dense and Split for two steps from packaged bytes, resume one step, and
assert memory ON/OFF evaluation paths execute without modifying fixture bytes.

- [ ] **Step 10: Commit**

```bash
git add corpusgen/current_dataset.py corpusgen/relational_build.py \
  corpusgen/graph_records.py corpusgen/graph_trace.py train/tokenizer.py \
  organizer/graph_store.py evals/constrain.py evals/relational_generate.py \
  scripts/build_current_dataset.py scripts/build_current_smoke.py \
  scripts/build_relational_corpus.py scripts/relational_smoke_test.py \
  fixtures/current-smoke tests/test_current_dataset.py \
  tests/test_graph_store.py tests/test_graph_trace.py tests/test_tokenizer.py \
  tests/test_constrain.py tests/test_relational_generate.py \
  tests/test_relational_smoke.py
git commit -m "feat: compile current seven-lane dataset"
```

---

### Task 3: Update Chinchilla manifests and cluster data gates

**Files:**
- Modify: `scripts/make_relational_manifest.py`
- Modify: `scripts/platform_preflight.py`
- Modify: `scripts/run_train.py`
- Modify: `configs/29m/*.yaml`
- Modify: `configs/160m/*.yaml`
- Modify: `configs/360m/*.yaml`
- Modify: `configs/29m.tsv`
- Modify: `configs/160m.tsv`
- Modify: `configs/360m.tsv`
- Modify: `tests/test_relational_manifest.py`
- Modify: `tests/test_platform_preflight.py`

**Interfaces:**
- Every job includes `dataset_id`, `dataset_lock_sha256`,
  `source_manifest_sha256`, `corpus_manifest_sha256`, `profile`, and nested
  `data_rel`.
- `run_train.py` verifies all identities before constructing a model.

- [ ] **Step 1: Write exact-budget tests**

```python
def test_current_token_floors():
    assert SCALE_SETTINGS["29m"]["total_tokens"] == 579_862_528
    assert SCALE_SETTINGS["160m"]["total_tokens"] == 3_244_818_432
    assert SCALE_SETTINGS["360m"]["total_tokens"] == 7_120_879_616
```

Also assert every `data_rel` starts with
`relational-chinchilla/corpora/`.

- [ ] **Step 2: Verify tests fail against historical budgets**

- [ ] **Step 3: Update manifests and YAML generation**

Generate six 29M diagnostics, fifteen 160M jobs, and six 360M jobs. Preserve the
existing seed/load ownership split. Do not hand-edit generated YAML.

- [ ] **Step 4: Add pre-training corpus verification**

Preflight and `run_train.py` must verify the source/build/lock hashes, exact
token count, all sidecars, complete-once flag for protected profiles, and
`scientific_result: false` rejection for the smoke profile.

- [ ] **Step 5: Run focused manifest/preflight tests**

- [ ] **Step 6: Commit**

```bash
git add scripts/make_relational_manifest.py scripts/platform_preflight.py \
  scripts/run_train.py configs tests/test_relational_manifest.py \
  tests/test_platform_preflight.py
git commit -m "feat: gate cluster runs on current dataset"
```

---

### Task 4: Update collaborator assignments and shared-data workflow

**Files:**
- Modify: `artifacts/collaborator-kits/common/build_owned_corpora.py`
- Modify: `artifacts/collaborator-kits/common/verify_assignment.py`
- Modify: `artifacts/collaborator-kits/common/launch_assignment.py`
- Create: `artifacts/collaborator-kits/common/stage_current_sources.sbatch`
- Modify: FarmShare and MIT assignment JSON/READMEs
- Modify: `artifacts/collaborator-kits/WORK-DISTRIBUTION.md`
- Modify: `artifacts/collaborator-kits/tests/test_kit_helpers.py`

**Interfaces:**
- Assignments pin `dataset_id`, lock hash, source-stage role, exact current token
  budget, and owned nested corpus paths.
- All collaborators may read one shared source tree; corpus ownership remains
  seed-balanced.

- [ ] **Step 1: Write assignment rejection tests**

Reject historical token counts, flat `data_rel`, missing source-manifest hash,
and any attempt to launch with the smoke profile.

- [ ] **Step 2: Update source staging and corpus-build helpers**

The source owner stages once. Other collaborators call verification only.
Corpus build jobs use CPU Slurm and atomically publish under:

```text
$DATA_ROOT/relational-chinchilla/corpora/<load>_ds<data_seed>
```

- [ ] **Step 3: Preserve ownership**

- Coordinator FarmShare: all seed-0 29M diagnostics and five seed-0 160M jobs.
- FarmShare ZIP 1: five seed-1 160M jobs.
- FarmShare ZIP 2: five seed-2 160M jobs.
- MIT ZIP A/B: three 360M jobs each, with seed 2 cross-owned as currently.

- [ ] **Step 4: Run assignment helper tests**

- [ ] **Step 5: Commit**

```bash
git add artifacts/collaborator-kits
git commit -m "feat: assign current dataset cluster builds"
```

---

### Task 5: Package deterministic smoke-ready ZIPs

**Files:**
- Create: `scripts/package_collaborator_kits.py`
- Modify: `scripts/package_relational_run.py`
- Modify: `tests/test_relational_bundle.py`
- Modify: `artifacts/collaborator-kits/tests/test_kit_helpers.py`
- Regenerate: `artifacts/relational-run.tar.gz`
- Regenerate: four collaborator ZIPs and checksum files

**Interfaces:**
- `package_collaborator_kits(root: Path, out_dir: Path) -> dict`
- Every ZIP has a flat, closed inventory and includes identical
  `fixtures/current-smoke/` bytes inside the portable bundle.

- [ ] **Step 1: Write bundle inventory tests**

Require source locks, licenses, current configs, source stager, current compiler,
smoke manifest, token stream, three sidecars, graph files, and verifiers.
Continue rejecting full corpora and checkpoints.

- [ ] **Step 2: Permit only namespaced smoke binaries**

`package_relational_run.py` may include `.bin` files only below
`fixtures/current-smoke/`. Every binary must appear in the smoke manifest.

- [ ] **Step 3: Implement deterministic outer ZIP builder**

Use fixed `1980-01-01` timestamps, lexical member order, mode `0644`, raw
deflate, and atomic replacement. Reopen every ZIP, test CRC, verify its closed
inventory and inner `SHA256SUMS`, then write
`artifacts/COLLABORATOR-ZIP-SHA256SUMS`.

- [ ] **Step 4: Build and test packages twice**

```bash
python scripts/package_collaborator_kits.py --out-dir /tmp/kits-a
python scripts/package_collaborator_kits.py --out-dir /tmp/kits-b
diff -ru /tmp/kits-a /tmp/kits-b
```

- [ ] **Step 5: Extract each ZIP and run offline smoke**

For each ZIP:

```bash
sha256sum -c SHA256SUMS
tar -xzf relational-run.tar.gz -C source
python source/scripts/relational_smoke_test.py \
  --fixture source/fixtures/current-smoke --device cpu
```

- [ ] **Step 6: Commit**

```bash
git add scripts/package_collaborator_kits.py scripts/package_relational_run.py \
  tests/test_relational_bundle.py artifacts
git commit -m "chore: publish current dataset cluster kits"
```

---

## Final verification

- [ ] Run source, compiler, smoke, manifest, preflight, bundle, and kit tests.
- [ ] Run `git diff --check`.
- [ ] Verify two independent ZIP builds are byte-identical.
- [ ] Verify each ZIP offline from a fresh temporary directory.
- [ ] Confirm no protected launcher accepts the smoke profile.
- [ ] Confirm full-data commands default to dry-run.
- [ ] Report the kits as **smoke-ready and full-build-capable**.
- [ ] Do not report **protected-training-ready** until cluster-side full source
  staging, complete corpus builds, 29M gates, and platform preflight pass.
