# V2 Puzzle Authority and Objective Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Add authenticated direct puzzle discovery/lookup and deterministic
whole-record objective allocation so Task 5C can consume the v2 source root.

**Architecture:** Puzzle bytes remain in the immutable direct SourceLock tree.
A compact recomputable scan commits training/evaluation/dedup decisions; a
separate catalog-source planner owns finite caps and whole-record allocation.

**Tech Stack:** Python 3.12, descriptor-relative POSIX I/O, canonical JSON and
SHA-256, exact integer Hamilton/deficit allocation, pytest.

## Global Constraints

- Binding design:
  `docs/superpowers/specs/2026-07-24-puzzle-objective-source-design.md`.
- Do not modify SourceLock schemas, source-root bytes, legacy current_sources,
  fixed source identities, Wikidata interfaces, or Task 6 files.
- Discovery uses SourceEntry inventories, never filesystem enumeration.
- No raw tasks, expected answers, or proof premise bytes in compact authority,
  catalog commitments, registry IDs, or receipts.
- All production behavior begins with a focused failing test.
- No download, AWS action, stage, commit, push, merge, or PR.

---

### Task 1: V2-Native Puzzle Scan and Direct Lookup

**Files:**
- Create: `corpusgen/reasoning_v2/puzzle_source.py`
- Create: `tests/test_reasoning_v2_puzzle_source.py`
- Modify: `tests/reasoning_v2_fixtures.py`
- Modify: `tests/test_reasoning_v2_source_lock.py`

**Interfaces:**

```python
PUZZLE_SOURCE_ORDER = ("arc_agi_1", "arc_agi_2", "conceptarc")

@dataclass(frozen=True)
class PuzzleTaskLocator:
    source_id: str
    path: str
    source_bytes: int
    source_sha256: str
    canonical_task_sha256: str
    test_count: int
    policy_sha256: str

@dataclass(frozen=True)
class PuzzleSourceScan:
    policy_sha256: str
    accepted: tuple[PuzzleTaskLocator, ...]
    evaluation_task_sha256s: tuple[str, ...]
    duplicates: tuple[PuzzleDuplicateRecord, ...]
    sha256: str

def scan_v2_puzzle_sources(
    source_lock: SourceLock,
    source_root: Path,
    *,
    expected_generator_commit: str,
) -> PuzzleSourceScan: ...

def iter_v2_puzzle_tasks(
    source_lock: SourceLock,
    source_root: Path,
    scan: PuzzleSourceScan,
    *,
    expected_generator_commit: str,
) -> Iterator[V2PuzzleTask]: ...

def read_v2_puzzle_task(
    source_lock: SourceLock,
    source_root: Path,
    locator: PuzzleTaskLocator,
    *,
    expected_generator_commit: str,
) -> V2PuzzleTask: ...
```

- [ ] **Step 1: Write RED tests**

Add focused tests for exact source/path order, direct layout without a legacy
manifest, repository/revision/license/file/canonical identity, strict JSON and
exact-answer schema, evaluation overlap, raw and canonical duplicates, first
winner, ConceptARC training-only policy, locator mutation, descriptor/name
replacement, deterministic scan, bounded memory, and selected-file-only
lookup.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_puzzle_source.py
```

Expected: collection fails because `puzzle_source` is absent.

- [ ] **Step 3: Implement authority**

Freeze the exact policy and compare it with
`configs/current-dataset-lock.json`. Authenticate the complete SourceLock root,
classify only SourceEntry paths, scan evaluations first, deduplicate by
canonical hash, record raw duplicates, and commit compact scan evidence.
Selected reads use descriptor-relative no-follow opens and pre/post identity,
size, raw hash, strict JSON, canonical hash, and test-count checks.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_puzzle_source.py \
  tests/test_reasoning_v2_source_lock.py \
  tests/test_current_sources.py
python -m py_compile corpusgen/reasoning_v2/puzzle_source.py
git diff --check
```

Expected: all pass; legacy current-source behavior remains unchanged.

---

### Task 2: Objective Source Planner and Catalog Adapter

**Files:**
- Create: `corpusgen/reasoning_v2/objective_source.py`
- Create: `tests/test_reasoning_v2_objective_source.py`
- Modify: `corpusgen/reasoning_v2/catalog.py`
- Modify: `tests/test_reasoning_v2_catalog.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ObjectiveSourceTargetPlan:
    arc_ceiling: int
    finite_realized_targets: int
    finite_unused_targets: int
    unbounded_quotas: tuple[tuple[str, int], ...]
    unbounded_realized: tuple[tuple[str, int], ...]
    assignments: tuple[str, ...]
    teacher_generated_targets: int

def plan_objective_sources(
    geometry: BuildGeometry,
    target_lengths: TargetLengths,
    puzzle_scan: PuzzleSourceScan,
) -> ObjectiveSourceTargetPlan: ...

class ObjectiveAuxiliaryCatalogSource:
    lane_id: LaneId = "objective_auxiliary"
    finite: bool = False
    def iter_drafts(
        self,
        source_root: Path,
        target_lengths: TargetLengths,
    ) -> Iterator[CatalogDraft]: ...
```

- [ ] **Step 1: Write RED tests**

Test exact corpus-wide finite ceiling, one-use finite records, stop-before-
crossing semantics, finite exhaustion/unused accounting, five-source Hamilton
order, largest-deficit assignment/ties, exact assigned total, one-record
deviation bound, teacher zero, all eight source IDs, deterministic keys under
reversed inputs, no evaluation/unknown records, exact locators, ConceptARC
`corpus/` acceptance, and outside-policy rejection.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_objective_source.py
```

Expected: collection fails because `objective_source` is absent.

- [ ] **Step 3: Implement whole-record allocation and adapter**

Apply the exact record-boundary algorithm from the design. Compute equal-share
Hamilton quotas over remaining target units; assign each whole record to the
largest current deficit with source-order ties; require exact total and maximum
deviation no greater than the largest record. Emit finite puzzle drafts once,
then unbounded-source drafts in target-length sequence order. Add only the exact
ConceptARC `corpus/` training override to catalog path validation.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_objective_source.py \
  tests/test_reasoning_v2_puzzle_source.py \
  tests/test_reasoning_v2_catalog.py \
  tests/test_reasoning_v2_contracts.py
python -m py_compile \
  corpusgen/reasoning_v2/puzzle_source.py \
  corpusgen/reasoning_v2/objective_source.py \
  corpusgen/reasoning_v2/catalog.py
git diff --check
```

Expected: all pass with exact target accounting and no partial catalog output.

## Task 5C Resume Gate

After Tasks 1 and 2 each pass independent review, resume
`.superpowers/sdd/task-5c-brief.md` with these binding changes:

- replace legacy `iter_puzzle_tasks` with `read_v2_puzzle_task`;
- remove cap/Hamilton ownership from the renderer;
- construct the objective renderer with SourceLock/root/generator commitment
  and one compact PuzzleSourceScan;
- require exact puzzle locator/answer commitments;
- keep the live Wikidata view registry parameter and eight-lane order;
- use `iter_supported_reasoning_records` for multihop;
- never serialize raw proof premise bytes.

## Plan Self-Review

- Direct-source and allocation responsibilities are separate.
- The whole-record rule is exact and deterministic.
- Every analysis requirement maps to one task/test.
- No placeholder, source-root mutation, derived puzzle view, or Task 6 work is
  included.
