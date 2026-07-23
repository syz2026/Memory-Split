# FarmShare MemorySplit v2 Corpus Producer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, publish, independently verify, mirror, and hand off the one
immutable `7,120,879,616`-target MemorySplit v2 corpus with the exact eight-lane
mixture and byte-aligned Dense/Split90 sidecars.

**Architecture:** A strict source freezer first resolves every mutable upstream
name to downloaded, licensed, hash-complete bytes. An external-memory catalog
and route planner feed eight production renderers into 32 deterministic task
spools; a coordinator schedules those records by largest deficit, writes 32
update-aligned token/sidecar shards, and creates a production extension of the
existing `memorysplit-parallel-corpus-v2` receipt. A separate verifier must
recompute all commitments and proof samples before an outer receipt can
atomically publish the FarmShare release; the same verifier then authenticates
an atomic MIT mirror.

**Tech Stack:** Python 3.11 or newer from the existing FarmShare venv,
standard-library dataclasses/typing/SQLite,
`numpy`, `pyarrow` through the existing `datasets` dependency, vendored GPT-2
`tiktoken`, existing `corpusgen.reasoning` and `corpusgen.parallel` primitives,
canonical JSON/JSONL, SHA-256, Slurm, Bash, `rsync`, Stanford FarmShare, and MIT
POSIX storage.

## Global Constraints

- Canonical site: Stanford FarmShare.
- Canonical root: `/scratch/users/syz/memorysplit-v2-corpus`.
- Repository checkout on FarmShare:
  `/scratch/users/syz/memorysplit`.
- FarmShare Python:
  `/scratch/users/syz/venvs/memorysplit/bin/python`.
- The implementation commit must be a clean descendant of source baseline
  `feat/memorysplit-v2-integration@33b8c9f`; every receipt records the exact
  final implementation commit rather than a branch name.
- Dataset ID: `memorysplit-v2-20x-reasoning-max-cohort`.
- Inner receipt format: `memorysplit-parallel-corpus-v2`.
- Inner receipt path: `dataset/corpus-receipt.json`.
- Outer receipt path: `receipt.json`.
- Raw causal targets: exactly `7,120,879,616`.
- Targets per optimizer update: exactly `524,288`.
- Optimizer updates: exactly `13,582`.
- Context length: exactly `1,024`.
- Full publication shards: exactly `32`.
- Full-build padding targets: exactly `0`.
- Allocation method: Hamilton largest remainder with the frozen lane order as
  the tie break.
- Frozen lane order and quotas:
  `fineweb_edu=1,780,219,904`,
  `finemath=1,068,131,943`,
  `wikidata_graph=1,424,175,923`,
  `synthetic_graph=712,087,962`,
  `verified_synthetic_multihop=1,068,131,942`,
  `wikidata_path_reasoning=534,065,971`,
  `relational_refinement=178,021,990`, and
  `objective_auxiliary=356,043,981`.
- The canary has exactly `1,048,576` targets and all eight lanes. Its Hamilton
  quotas are `262,144`, `157,287`, `209,715`, `104,858`, `157,286`, `78,643`,
  `26,214`, and `52,429` in frozen lane order.
- Dense target weights are `uint8` value `1` for every logical target.
- Split90 target weights are `uint8`, differ only on routed factual payload
  targets, and keep rules, operators, schemas, and proof procedures supervised.
- Every surface occurrence of a routed factual payload must have Split90 weight
  `0`; candidate and final answer states may contain structural pointers but no
  routed factual surface.
- Split90 must route at least `90%` of distinct offloadable facts and at least
  `90%` of train-only information-weighted burden.
- Teacher-generated chain-of-thought contributes exactly zero targets.
- The vendored tokenizer assets and token IDs in `train/tokenizer.py` remain
  unchanged.
- FineWeb-Edu stays at repository `HuggingFaceFW/fineweb-edu`, revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, with the three existing 10BT
  files `sample/10BT/000_00000.parquet`,
  `sample/10BT/001_00000.parquet`, and
  `sample/10BT/002_00000.parquet`.
- Wikidata5M stays at repository `intfloat/wikidata5m`, revision
  `6b2b09672129e280c0c9da97ab58154e9d535e6b`, with
  `wikidata5m_alias.tar.gz` at `197,449,751` bytes and SHA-256
  `0330f580c9f7a57cbad949ac380835fdd2a2e14d96cc0f13fc435401d6b463a8`,
  `wikidata5m_inductive.tar.gz` at `167,247,416` bytes and SHA-256
  `955081232cc2de859710bfe3a147f7d8314524010fe5f8c420bb74fdfee4f42a`,
  and `wikidata5m_transductive.tar.gz` at `168,258,214` bytes and SHA-256
  `383160990b41c0905fc03f4a8afbb9b12be1ca3591e026bde6cdc94a59542597`.
- ARC-AGI stays at commit
  `399030444e0ab0cc8b4e199870fb20b863846f34`; ARC-AGI-2 stays at
  `f3283f727488ad98fe575ea6a5ac981e4a188e49`; ConceptARC stays at
  `0e67da6af879e4bad3d7cd3c196e8d551b445725`.
- The source freezer must resolve FineMath, CLRS, RuleTaker, ProntoQA,
  Reasoning Gym, and the DeepMind mathematics generator from live upstream
  metadata and downloaded bytes. Their revisions must never be guessed or
  copied from this plan.
- The source-freeze CLI accepts source IDs only. It has no revision, tag,
  branch, arbitrary URL, or token option; Hugging Face calls use
  `token=False`. Every emitted revision is a lowercase immutable commit or
  content digest.
- Accepted source licenses are exactly `ODC-By-1.0`, `CC0-1.0`,
  `Apache-2.0`, and `MIT`; a source whose repository and data files do not
  prove one of these values blocks the build.
- `configs/reasoning-dataset-v2-source-lock.json` must be generated, verified,
  pulled back from FarmShare, and committed before the canary or full catalog
  command is allowed to run.
- Any implementation change after source-lock generation invalidates that
  lock's `generator_commit`; resolve, verify, and commit a new source lock
  before resuming production.
- A finite reasoning source may never wrap to its first record. Exhaustion
  raises `LaneQuotaShortfall`; generated records use unique deterministic
  seeds and record IDs.
- Wikidata graph rendering covers the complete frozen training graph once
  before any deterministic graph revisit.
- Evaluation and sealed-test paths are rejected at catalog construction, not
  filtered after rendering.
- All production text entering a renderer is NFC-normalized by the catalog
  canonicalizer; a renderer rejects non-NFC input rather than silently changing
  it.
- Catalog order and every output byte are independent of directory enumeration,
  download order, Python hash randomization, worker count, worker completion
  order, and Slurm task completion order.
- The full FarmShare build uses one `0-31%8` array on partition `normal`, with
  `16` CPUs, `64G`, a `24:00:00` limit, node-local `SLURM_TMPDIR`,
  `#SBATCH --export=NONE`, and an explicit environment allowlist.
- The full build refuses to start unless the canonical filesystem reports at
  least `200 GiB` free.
- Publication is an owner-controlled, no-replace atomic rename. There is no
  mutable `current` symlink and no successful retry mutates a published
  release.
- Verification opens regular files without following symlinks, rejects
  hardlinks and special files, rechecks EOF and post-read identity, and
  recomputes logical streams rather than trusting receipt scalars.
- Existing `build-fixture`, `render-fixture-task`,
  `finalize-fixture-tasks`, v1 receipt, current-dataset smoke, and
  `fixtures/current-smoke` behavior remain unchanged.
- Existing `configs/360m-v2/*.yaml`, `configs/cohort-assignment-v2.json`,
  `DATASET-POINTER.json`, and `DATASET-POINTER-AWS.json` remain byte-identical.
- A fixture, legacy seven-lane corpus, partial build, mutable source, or smoke
  corpus can never produce a production outer receipt.
- A verified canary has an outer receipt with
  `protected_launch_allowed=false`; only the independently rebuilt and
  verified full profile may set it to `true`.
- Code completion is not the terminal condition. Completion requires a
  published FarmShare release, FarmShare verification receipt, matching second
  full-build commitments, an atomically published MIT mirror, an MIT
  verification receipt, and successful corpus binding by the 135M and 360M
  multiseed packages.

---

## Exact File and Responsibility Map

**Frozen contracts and source material**

- Create during source-freeze execution:
  `configs/reasoning-dataset-v2-source-lock.json` — canonical generated lock of
  every external repository, immutable revision, selected file, byte count,
  SHA-256, license, and path relative to
  `sources/<source-lock-sha256>/`.
- Create: `corpusgen/reasoning_v2/contracts.py` — strict parser for
  `configs/reasoning-dataset-v2.json`, Hamilton allocation, full/canary
  geometries, frozen lane identifiers, and package-wide constants.
- Create: `corpusgen/reasoning_v2/source_lock.py` — immutable upstream
  resolution, strict source-lock schema, content-addressed staging, license
  enforcement, and staged-byte verification.
- Create: `tests/reasoning_v2_fixtures.py` — deterministic eight-lane source
  fixture, fake resolver, source-lock builder, and miniature recipe helpers.
- Create: `tests/test_reasoning_v2_contracts.py` — recipe, geometry, and
  Hamilton contract tests.
- Create: `tests/test_reasoning_v2_source_lock.py` — parser, resolver, license,
  staging, drift, and mutable-input rejection tests.

**Catalog, routing, and rendering**

- Create: `corpusgen/reasoning_v2/catalog.py` — external-memory
  `CatalogRecord`/`InputCatalog`, deterministic record IDs, frozen split
  lengths, lane exhaustion checks, source/evaluation provenance, and canonical
  JSONL indexes.
- Create: `corpusgen/reasoning_v2/semantic.py` — production external-sort route
  manifest, route index, dose report, token semantic spans, sidecar derivation,
  and semantic-leak aggregation.
- Create: `corpusgen/reasoning_v2/renderers.py` — one concrete renderer for
  each frozen lane, shared `ProductionRenderedRecord`, tokenizer fitting,
  solver/proof envelopes, objective-source suballocation, and renderer
  registry.
- Create: `tests/test_reasoning_v2_catalog.py` — all-eight-lane catalog,
  ordering, quota, complete-graph, contamination, and no-cycle tests.
- Create: `tests/test_reasoning_v2_semantic.py` — route parity, dose, closure,
  answer-state, and binary sidecar tests.
- Create: `tests/test_reasoning_v2_renderers.py` — exact renderer behavior for
  every lane, proof replay, Unicode, finite-value, context, and duplicate
  rejection tests.

**Parallel build and receipts**

- Create: `corpusgen/reasoning_v2/tasks.py` — 32-way ordinal partitioning,
  node-local token/weight/proof spools, hash-bound task receipts, resume
  validation, external-memory largest-deficit schedule, and aligned shard
  assembly.
- Create: `corpusgen/reasoning_v2/publication.py` — production inner receipt,
  outer materialization receipt, candidate namespace, no-replace release
  publication, build ID, and verification-gated scientific status.
- Create: `corpusgen/reasoning_v2/verification.py` — streaming inner/outer
  verification, pinned filesystem reads, proof sampling, independent rebuild
  comparison, and site verification receipts.
- Create: `corpusgen/reasoning_v2/handoff.py` — read-only consumer requirement
  validation for 135M and 360M multiseed packages.
- Create: `corpusgen/reasoning_v2/__init__.py` — only the reviewed public API
  from these modules.
- Modify: `corpusgen/parallel/publication.py` — dispatch receipts with a strict
  `production_bindings` object to the streaming v2 verifier while preserving
  the existing v1 and fixture-v2 byte contracts.
- Modify: `corpusgen/parallel/__init__.py` — export the production verifier
  dispatch without changing existing exports.
- Modify: `train/data.py` — parse production assignment paths
  `shard-00000.bin` through `shard-00031.bin` when
  `production_bindings` is present while preserving fixture-v2 assignment
  names.
- Create: `tests/test_reasoning_v2_tasks.py` — task partition, spool, replay,
  resume, scheduling, and shard tests.
- Create: `tests/test_reasoning_v2_publication.py` — inner/outer schema,
  no-replace, race, gate, and legacy-refusal tests.
- Create: `tests/test_reasoning_v2_verification.py` — tampering, filesystem,
  proof, stream, and independent rebuild tests.
- Create: `tests/test_reasoning_v2_handoff.py` — existing 360M and generic 135M
  consumer binding tests.
- Modify: `tests/test_parallel_corpus.py` — production-dispatch regression
  coverage while retaining all existing fixture assertions.
- Modify: `tests/test_sharded_loader.py` — open the exact production inner
  receipt and both named sidecars through `PackedShards`.

**CLI, Slurm, mirror, and operations**

- Create: `scripts/build_reasoning_v2_corpus.py` — production-only CLI for
  source freeze/stage, preflight, catalog, route, task render, candidate
  finalization, rebuild, verification, publication, mirror, and handoff.
- Modify: `scripts/build_parallel_corpus.py` — keep fixture commands unchanged;
  remove `UnsupportedProductionRenderer` from the executable production path
  and direct production operators to the dedicated CLI.
- Create: `cluster/slurm/v2_corpus_source_freeze.sbatch` — preproduction
  immutable-resolution job that writes the source-lock candidate outside the
  repository.
- Create: `cluster/slurm/v2_corpus_source_stage.sbatch` — post-commit
  content-addressed source staging and full-byte verification job.
- Modify: `cluster/slurm/v2_corpus_build.sbatch` — real 32-task production
  render array.
- Create: `cluster/slurm/v2_corpus_finalize.sbatch` — exact-N coordinator that
  creates an unpublished release candidate.
- Modify: `cluster/slurm/v2_corpus_verify.sbatch` — independent verifier,
  second-build comparison, outer receipt creation, and no-replace FarmShare
  publication.
- Create: `cluster/submit_v2_corpus.sh` — dry-run-by-default Slurm dependency
  submission for canary and full profiles.
- Create: `corpusgen/reasoning_v2/mirror.py` — private mirror staging, exact
  copy plan, local destination verification, no-replace publication, and MIT
  verification receipt.
- Create: `cluster/mirror_v2_to_mit.sh` — DTN/SSH/rsync driver with explicit
  host/root arguments and no inherited credentials.
- Create: `cluster/FARMSHARE-V2-CORPUS-RUNBOOK.md` — exact source-freeze,
  canary, full build, rebuild, verification, MIT mirror, failure, and handoff
  commands.
- Create: `tests/test_reasoning_v2_cli.py` — production CLI strictness and
  miniature end-to-end tests.
- Create: `tests/test_reasoning_v2_slurm.py` — resource, environment,
  dependency, and local shell tests.
- Create: `tests/test_reasoning_v2_mirror.py` — copy, tamper, collision, and
  FarmShare/MIT identity tests.
- Create: `tests/test_reasoning_v2_regressions.py` — fixture, legacy receipt,
  tokenizer, 360M config, dataset-pointer, and runbook release gates.

---

## Canonical On-Site Layout

All implementation and execution tasks target this exact namespace:

```text
/scratch/users/syz/memorysplit-v2-corpus/
  sources/
    <source-lock-sha256>/
  work/
    <run-nonce>/
  releases/
    <build-id>/
      receipt.json
      source-lock.json
      dataset/
        corpus-receipt.json
        shards/
          shard-<index>.bin
        sidecars/
          dense_target_weights/
            shard-<index>.bin
          split90_target_weights/
            shard-<index>.bin
        manifests/
        proofs/
  verification/
    <build-id>.json
```

Angle-bracket names above are receipt-derived lowercase SHA-256 values or the
operator's validated run nonce; `index` is every zero-padded integer from
`00000` through `00031`, inclusive. They are never mutable aliases.

---

## Part I — Local Implementation Tasks

### Task 1: Freeze recipe parsing and exact geometry

**Files:**
- Create: `corpusgen/reasoning_v2/__init__.py`
- Create: `corpusgen/reasoning_v2/contracts.py`
- Create: `tests/test_reasoning_v2_contracts.py`

**Interfaces:**
- Produces:
  `LaneId`,
  `LaneContract`,
  `BuildGeometry`,
  `ReasoningV2Recipe`,
  `hamilton_quotas(total_targets: int, shares: tuple[tuple[LaneId, Fraction], ...]) -> tuple[tuple[LaneId, int], ...]`,
  `balanced_record_lengths(targets: int, context_length: int) -> tuple[int, ...]`,
  `load_recipe(path: Path) -> ReasoningV2Recipe`, and
  `geometry_for(recipe: ReasoningV2Recipe, profile: Literal["canary", "full"]) -> BuildGeometry`.
- Consumes only `configs/reasoning-dataset-v2.json`.

- [ ] **Step 1: RED — write strict recipe and geometry tests**

```python
from fractions import Fraction
from pathlib import Path

import pytest

from corpusgen.reasoning_v2.contracts import (
    LANE_ORDER,
    balanced_record_lengths,
    geometry_for,
    hamilton_quotas,
    load_recipe,
)

ROOT = Path(__file__).resolve().parents[1]


def test_full_recipe_recomputes_exact_hamilton_quotas():
    recipe = load_recipe(ROOT / "configs/reasoning-dataset-v2.json")
    assert recipe.total_targets == 7_120_879_616
    assert recipe.targets_per_update == 524_288
    assert recipe.optimizer_updates == 13_582
    assert recipe.context_length == 1_024
    assert recipe.shard_count == 32
    assert recipe.lane_quotas == (
        ("fineweb_edu", 1_780_219_904),
        ("finemath", 1_068_131_943),
        ("wikidata_graph", 1_424_175_923),
        ("synthetic_graph", 712_087_962),
        ("verified_synthetic_multihop", 1_068_131_942),
        ("wikidata_path_reasoning", 534_065_971),
        ("relational_refinement", 178_021_990),
        ("objective_auxiliary", 356_043_981),
    )
    assert hamilton_quotas(recipe.total_targets, recipe.lane_shares) == (
        recipe.lane_quotas
    )


def test_canary_is_two_updates_with_all_eight_lanes():
    canary = geometry_for(
        load_recipe(ROOT / "configs/reasoning-dataset-v2.json"),
        "canary",
    )
    assert canary.total_targets == 1_048_576
    assert canary.total_targets // canary.targets_per_update == 2
    assert canary.lane_quotas == tuple(
        zip(
            LANE_ORDER,
            (262_144, 157_287, 209_715, 104_858, 157_286, 78_643, 26_214, 52_429),
            strict=True,
        )
    )


def test_balanced_lengths_close_quota_without_short_tail():
    lengths = balanced_record_lengths(1_068_131_943, 1_024)
    assert sum(lengths) == 1_068_131_943
    assert max(lengths) == 1_024
    assert min(lengths) >= 1_023


@pytest.mark.parametrize("bad", [True, 7_120_879_616.0, -1])
def test_hamilton_rejects_non_integer_or_negative_totals(bad):
    shares = (("fineweb_edu", Fraction(1, 1)),)
    with pytest.raises((TypeError, ValueError)):
        hamilton_quotas(bad, shares)
```

Add mutation cases that duplicate a JSON key, change lane order, change one
quota, change `share_percent`, change the total/update product, use a non-finite
number, or add an unknown contract field. Each must fail before returning a
recipe.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_contracts.py
```

Expected: collection fails with
`ModuleNotFoundError: No module named 'corpusgen.reasoning_v2'`.

- [ ] **Step 3: GREEN — implement the minimal strict contract**

Use these public types and exact allocation rules:

```python
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

LaneId = Literal[
    "fineweb_edu",
    "finemath",
    "wikidata_graph",
    "synthetic_graph",
    "verified_synthetic_multihop",
    "wikidata_path_reasoning",
    "relational_refinement",
    "objective_auxiliary",
]
LANE_ORDER: tuple[LaneId, ...] = (
    "fineweb_edu",
    "finemath",
    "wikidata_graph",
    "synthetic_graph",
    "verified_synthetic_multihop",
    "wikidata_path_reasoning",
    "relational_refinement",
    "objective_auxiliary",
)


@dataclass(frozen=True)
class LaneContract:
    lane_id: LaneId
    share: Fraction
    quota: int
    verification: str


@dataclass(frozen=True)
class BuildGeometry:
    profile: Literal["canary", "full"]
    total_targets: int
    targets_per_update: int
    context_length: int
    shard_count: int
    allow_fewer_shards: bool
    lane_quotas: tuple[tuple[LaneId, int], ...]


@dataclass(frozen=True)
class ReasoningV2Recipe:
    dataset_id: str
    total_targets: int
    targets_per_update: int
    optimizer_updates: int
    context_length: int
    shard_count: int
    lanes: tuple[LaneContract, ...]
    source_policy: dict[str, object]
    intervention: dict[str, object]

    @property
    def lane_shares(self) -> tuple[tuple[LaneId, Fraction], ...]:
        return tuple((lane.lane_id, lane.share) for lane in self.lanes)

    @property
    def lane_quotas(self) -> tuple[tuple[LaneId, int], ...]:
        return tuple((lane.lane_id, lane.quota) for lane in self.lanes)


def hamilton_quotas(
    total_targets: int,
    shares: tuple[tuple[LaneId, Fraction], ...],
) -> tuple[tuple[LaneId, int], ...]:
    if type(total_targets) is not int or total_targets <= 0:
        raise ValueError("total_targets must be a positive integer")
    if tuple(lane for lane, _share in shares) != LANE_ORDER:
        raise ValueError("shares must use the frozen lane order")
    if sum((share for _lane, share in shares), Fraction()) != 1:
        raise ValueError("lane shares must sum exactly to one")
    floors = {
        lane: (total_targets * share).numerator // (total_targets * share).denominator
        for lane, share in shares
    }
    remainder = total_targets - sum(floors.values())
    ranked = sorted(
        shares,
        key=lambda item: (
            -((total_targets * item[1]) - floors[item[0]]),
            LANE_ORDER.index(item[0]),
        ),
    )
    for lane, _share in ranked[:remainder]:
        floors[lane] += 1
    return tuple((lane, floors[lane]) for lane in LANE_ORDER)


def balanced_record_lengths(targets: int, context_length: int) -> tuple[int, ...]:
    if type(targets) is not int or targets <= 0:
        raise ValueError("targets must be a positive integer")
    if type(context_length) is not int or context_length <= 0:
        raise ValueError("context_length must be a positive integer")
    count = (targets + context_length - 1) // context_length
    base, extra = divmod(targets, count)
    return (base + 1,) * extra + (base,) * (count - extra)
```

`load_recipe()` must parse UTF-8 JSON with an `object_pairs_hook` that rejects
duplicate keys and a `parse_constant` callback that rejects `NaN` and
infinities. It must reject unknown fields at the top level, sprint-recipe,
lane, realized-allocation, source-policy, publication-requirements,
intervention, and split90-dose levels. Convert decimal shares with
`Fraction(str(value)) / 100`, recompute Hamilton quotas, and compare the result
to the committed allocation. `geometry_for(recipe, "canary")` recomputes Hamilton
quotas for `1_048_576`; `"full"` copies the committed full geometry.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_contracts.py tests/test_v2_contract_files.py
```

Expected: all tests pass; the existing contract-file tests remain unchanged.

- [ ] **Step 5: Commit**

```bash
git add corpusgen/reasoning_v2/__init__.py \
  corpusgen/reasoning_v2/contracts.py \
  tests/test_reasoning_v2_contracts.py
git commit -m "feat: freeze v2 corpus geometry"
```

---

### Task 2: Implement immutable source freeze and staging

**Files:**
- Create: `corpusgen/reasoning_v2/source_lock.py`
- Create: `tests/reasoning_v2_fixtures.py`
- Create: `tests/test_reasoning_v2_source_lock.py`

**Interfaces:**
- Consumes: `ReasoningV2Recipe` and the fixed source identities in Global
  Constraints.
- Produces:
  `SourceFile`,
  `SourceEntry`,
  `SourceLock`,
  `SourceRequest`,
  `SourceResolver`,
  `PublicSourceResolver`,
  `load_source_lock(path: Path) -> SourceLock`,
  `resolve_source_lock(recipe: ReasoningV2Recipe, resolver: SourceResolver, download_root: Path, generator_commit: str) -> SourceLock`,
  `stage_source_lock(lock: SourceLock, download_root: Path, canonical_root: Path) -> Path`, and
  `verify_source_tree(lock: SourceLock, source_root: Path) -> dict[str, object]`.

- [ ] **Step 1: RED — write resolver, parser, and staging tests**

```python
def test_resolver_emits_only_immutable_licensed_hash_complete_entries(
    tmp_path,
    fake_public_resolver,
    full_recipe,
):
    lock = resolve_source_lock(
        full_recipe,
        fake_public_resolver,
        tmp_path / "downloads",
        generator_commit="a" * 40,
    )
    by_id = {entry.source_id: entry for entry in lock.sources}
    assert {
        "fineweb_edu",
        "finemath",
        "wikidata5m",
        "clrs_text",
        "ruletaker",
        "prontoqa",
        "reasoning_gym_exact_answer",
        "deepmind_mathematics_generator",
        "arc_agi_1",
        "arc_agi_2",
        "conceptarc",
    } <= set(by_id)
    for entry in lock.sources:
        assert entry.revision_kind in {"git_commit", "content_sha256"}
        assert len(entry.revision) in {40, 64}
        assert entry.license_spdx
        assert entry.license_files
        assert entry.files
        assert all(row.bytes > 0 and len(row.sha256) == 64 for row in entry.files)
        assert not Path(entry.materialized_path).is_absolute()


def test_public_huggingface_resolver_never_uses_ambient_token(monkeypatch):
    calls = []

    class RecordingApi:
        def __init__(self, *, token):
            calls.append(token)

    monkeypatch.setenv("HF_TOKEN", "must-not-be-consumed")
    monkeypatch.setattr(source_lock_module, "HfApi", RecordingApi)
    PublicSourceResolver()
    assert calls == [False]


@pytest.mark.parametrize(
    "revision",
    ["main", "master", "latest", "v2.0.1", "refs/heads/main", ""],
)
def test_source_lock_rejects_mutable_revision_text(tmp_path, valid_lock_json, revision):
    value = valid_lock_json
    value["sources"][0]["revision"] = revision
    path = tmp_path / "lock.json"
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(ValueError, match="immutable revision"):
        load_source_lock(path)


def test_staging_is_content_addressed_and_rejects_byte_drift(
    tmp_path,
    fixture_source_lock,
):
    root = stage_source_lock(
        fixture_source_lock.lock,
        fixture_source_lock.download_root,
        tmp_path / "canonical",
    )
    assert root == (
        tmp_path
        / "canonical"
        / "sources"
        / fixture_source_lock.lock.sha256
    )
    assert verify_source_tree(fixture_source_lock.lock, root)["passed"] is True
    victim = next(path for path in root.rglob("*") if path.is_file())
    victim.write_bytes(victim.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="source byte drift"):
        verify_source_tree(fixture_source_lock.lock, root)
```

Also test duplicate source IDs, duplicate file paths, absolute/traversing
materialized paths, bool byte counts, unsupported SPDX values, missing license
files, unlisted files, symlinks, special files, noncanonical JSON, fixed
FineWeb/Wikidata/ARC identity drift, a RuleTaker archive URL without a computed
digest, and an unresolved source.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_source_lock.py
```

Expected: import fails because `corpusgen.reasoning_v2.source_lock` does not
exist.

- [ ] **Step 3: GREEN — implement the strict schema and built-in requests**

Use this schema and resolver boundary:

```python
@dataclass(frozen=True)
class SourceRequest:
    source_id: str
    transport: Literal["huggingface_dataset", "git"]
    repository: str
    required_license_paths: tuple[str, ...] = ()
    required_data_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceFile:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class SourceEntry:
    source_id: str
    transport: Literal["huggingface_dataset", "git", "https_artifact"]
    repository: str
    revision_kind: Literal["git_commit", "content_sha256"]
    revision: str
    license_spdx: str
    license_files: tuple[str, ...]
    materialized_path: str
    files: tuple[SourceFile, ...]


@dataclass(frozen=True)
class SourceLock:
    schema_version: Literal[1]
    format: Literal["memorysplit-reasoning-v2-source-lock-v1"]
    dataset_id: str
    dataset_contract_sha256: str
    generator_commit: str
    sources: tuple[SourceEntry, ...]

    @property
    def sha256(self) -> str:
        return sha256_hex(self.to_bytes())

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())


class SourceResolver(Protocol):
    def resolve(self, request: SourceRequest, download_root: Path) -> SourceEntry:
        raise RuntimeError("SourceResolver implementations must resolve a source")
```

Define built-in requests for these exact repositories:

```python
PUBLIC_REQUESTS = (
    SourceRequest("finemath", "huggingface_dataset", "HuggingFaceTB/finemath"),
    SourceRequest("clrs_text", "git", "https://github.com/google-deepmind/clrs.git"),
    SourceRequest("ruletaker", "git", "https://github.com/allenai/ruletaker.git"),
    SourceRequest("prontoqa", "git", "https://github.com/asaparov/prontoqa.git"),
    SourceRequest(
        "reasoning_gym_exact_answer",
        "git",
        "https://github.com/open-thought/reasoning-gym.git",
    ),
    SourceRequest(
        "deepmind_mathematics_generator",
        "git",
        "https://github.com/google-deepmind/mathematics_dataset.git",
    ),
)
```

`PublicSourceResolver` must:

1. Instantiate `HfApi(token=False)`, query FineMath repository metadata, require
   a 40-character returned SHA, enumerate paths in bytewise path order, and
   download `README.md` plus sorted `finemath-4plus/train-*.parquet` files until
   NFC canonicalization, exact FineWeb cross-deduplication, and the vendored
   tokenizer prove more than the FineMath quota is available. Continue with
   sorted `finemath-3plus/train-*.parquet` files only if the first subset is
   insufficient.
2. For each Git request, run `git ls-remote <repository> HEAD`, require one
   40-character commit, initialize a bare cache, fetch that commit by object
   ID, verify `FETCH_HEAD^{commit}`, and archive the commit without checking out
   a mutable branch.
3. For RuleTaker, parse only the official dataset URL recorded in the resolved
   repository README, download it during freeze, place the archive under the
   `ruletaker` materialized path, and include its byte count and computed
   SHA-256 in that entry's file inventory. Never permit a caller-supplied URL.
4. Inventory regular files in sorted POSIX order, reject links and device
   entries, hash in 1 MiB chunks, and require the repository license file plus
   FineMath's ODC-By metadata.
5. Add the already frozen FineWeb, Wikidata, ARC-AGI, ARC-AGI-2, and ConceptARC
   entries only after their current lock bytes match the fixed identities.

`stage_source_lock()` must copy into a private sibling of
`sources/<lock.sha256>`, fsync every file and directory, call
`verify_source_tree()`, and install with no-replace rename. If the final path
already exists, verify and reuse it; never overwrite it.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_source_lock.py \
  tests/test_current_sources.py
```

Expected: all tests pass, including the existing fixed source identities.

- [ ] **Step 5: Commit**

```bash
git add corpusgen/reasoning_v2/source_lock.py \
  tests/reasoning_v2_fixtures.py \
  tests/test_reasoning_v2_source_lock.py
git commit -m "feat: add immutable v2 source freeze"
```

---

### Task 3: Build the canonical external-memory eight-lane catalog

**Files:**
- Create: `corpusgen/reasoning_v2/catalog.py`
- Create: `tests/test_reasoning_v2_catalog.py`

**Interfaces:**
- Consumes:
  `BuildGeometry`,
  `SourceLock`,
  verified source root, and
  `Mapping[LaneId, LaneCatalogSource]`.
- Produces:
  `SemanticFactRow`,
  `CatalogDraft`,
  `CatalogRecord`,
  `CatalogLaneIndex`,
  `InputCatalog`,
  `LaneCatalogSource`,
  `LaneQuotaShortfall`,
  `catalog_record_id(draft: CatalogDraft, target_count: int) -> str`, and
  `build_input_catalog(geometry: BuildGeometry, source_lock: SourceLock, source_root: Path, lane_sources: Mapping[LaneId, LaneCatalogSource], output_root: Path) -> InputCatalog`.

- [ ] **Step 1: RED — write all-lane, ordering, and exhaustion tests**

```python
def test_catalog_is_canonical_exact_and_filesystem_order_independent(
    tmp_path,
    tiny_geometry,
    staged_fixture_sources,
    fixture_lane_sources,
):
    forward = build_input_catalog(
        tiny_geometry,
        staged_fixture_sources.lock,
        staged_fixture_sources.root,
        fixture_lane_sources(order="forward"),
        tmp_path / "forward",
    )
    reverse = build_input_catalog(
        tiny_geometry,
        staged_fixture_sources.lock,
        staged_fixture_sources.root,
        fixture_lane_sources(order="reverse"),
        tmp_path / "reverse",
    )
    assert forward.sha256 == reverse.sha256
    assert forward.to_bytes() == reverse.to_bytes()
    assert forward.lane_target_counts == tiny_geometry.lane_quotas
    assert tuple(row.lane_id for row in forward.lanes) == LANE_ORDER
    records = tuple(forward.iter_records())
    assert [row.ordinal for row in records] == list(range(len(records)))
    assert len({row.record_id for row in records}) == len(records)
    assert all(row.target_count <= tiny_geometry.context_length for row in records)
    assert all("sealed-test" not in row.semantic_flags for row in records)


def test_finite_reasoning_lane_never_cycles_to_fill_quota(
    tmp_path,
    tiny_geometry,
    staged_fixture_sources,
    fixture_lane_sources,
):
    sources = fixture_lane_sources(order="forward")
    sources["wikidata_path_reasoning"] = FiniteFixtureLane(records=1)
    with pytest.raises(
        LaneQuotaShortfall,
        match="wikidata_path_reasoning.*distinct deterministic records",
    ):
        build_input_catalog(
            tiny_geometry,
            staged_fixture_sources.lock,
            staged_fixture_sources.root,
            sources,
            tmp_path / "short",
        )


def test_wikidata_graph_covers_every_training_edge_before_revisit(catalog_fixture):
    rows = [
        row
        for row in catalog_fixture.iter_records()
        if row.lane_id == "wikidata_graph"
    ]
    first_revisit = next(
        index for index, row in enumerate(rows) if "graph-revisit" in row.semantic_flags
    )
    assert {
        row.source_key for row in rows[:first_revisit]
    } == catalog_fixture.wikidata_training_edge_keys
```

Add tests for duplicate source keys, duplicate semantic fact IDs, a generated
seed reused under another record ID, source-byte hash disagreement, sealed
paths, missing lane source, unknown lane, non-NFC text, non-finite semantic
payload, unstable mapping key order, and target lengths that do not sum to the
lane quota.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_catalog.py
```

Expected: import fails because `corpusgen.reasoning_v2.catalog` is missing.

- [ ] **Step 3: GREEN — implement streaming records and canonical indexes**

Use these exact record fields:

```python
class LaneQuotaShortfall(ValueError):
    def __init__(self, lane_id: LaneId, requested: int, emitted: int) -> None:
        self.lane_id = lane_id
        self.requested = requested
        self.emitted = emitted
        super().__init__(
            f"{lane_id} lacks distinct deterministic records: "
            f"requested={requested}, emitted={emitted}"
        )


@dataclass(frozen=True)
class SemanticFactRow:
    fact_id: str
    source: str
    record_type: str
    payload_entropy_bits: Fraction
    scheduled_exposures: int
    expected_reads: Fraction
    expected_hops: Fraction
    surfaces: tuple[str, ...]


@dataclass(frozen=True)
class CatalogDraft:
    lane_id: LaneId
    source_id: str
    source_key: str
    source_byte_sha256: str
    source_locator: tuple[tuple[str, str | int], ...]
    semantic_flags: tuple[str, ...]
    semantic_facts: tuple[SemanticFactRow, ...]


@dataclass(frozen=True)
class CatalogRecord:
    ordinal: int
    record_id: str
    lane_id: LaneId
    source_id: str
    source_key: str
    source_byte_sha256: str
    source_locator: tuple[tuple[str, str | int], ...]
    target_count: int
    semantic_flags: tuple[str, ...]
    semantic_facts: tuple[SemanticFactRow, ...]


@dataclass(frozen=True)
class CatalogLaneIndex:
    lane_id: LaneId
    first_ordinal: int
    record_count: int
    target_count: int


@dataclass(frozen=True)
class InputCatalog:
    root: Path
    records_path: Path
    index_path: Path
    source_lock_sha256: str
    sha256: str
    record_count: int
    target_count: int
    lanes: tuple[CatalogLaneIndex, ...]

    @property
    def lane_target_counts(self) -> tuple[tuple[LaneId, int], ...]:
        return tuple((lane.lane_id, lane.target_count) for lane in self.lanes)

    def iter_records(self) -> Iterator[CatalogRecord]:
        return records_from_jsonl(self.records_path)

    def to_bytes(self) -> bytes:
        return self.index_path.read_bytes()


class LaneCatalogSource(Protocol):
    lane_id: LaneId
    finite: bool

    def iter_drafts(
        self,
        source_root: Path,
        target_lengths: tuple[int, ...],
    ) -> Iterator[CatalogDraft]:
        raise RuntimeError("lane source must emit deterministic drafts")
```

For each lane, call
`balanced_record_lengths(quota, geometry.context_length)` once. Consume exactly
one unique draft per requested length, set `target_count` from that length, and
raise `LaneQuotaShortfall` if a finite iterator ends. Hash the canonical draft
core plus `target_count` for `record_id`; do not include global ordinal in the
ID. Write `catalog.jsonl` in lane order and lane-local source-key order, then
write `catalog-index.json` with source-lock hash, per-lane ordinal ranges,
counts, target sums, overall record count, target count, and the SHA-256 of
`catalog.jsonl`. Read every source iterator through verified manifest paths;
reject path names containing evaluation, validation, test, sealed, or holdout
unless the source lock explicitly marks the file as training.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_catalog.py \
  tests/test_reasoning_v2_contracts.py
```

Expected: all tests pass and every miniature lane reaches its exact quota.

- [ ] **Step 5: Commit**

```bash
git add corpusgen/reasoning_v2/catalog.py \
  tests/test_reasoning_v2_catalog.py
git commit -m "feat: add canonical v2 input catalog"
```

---

### Task 4: Add production routing and semantic sidecar closure

**Files:**
- Create: `corpusgen/reasoning_v2/semantic.py`
- Create: `tests/test_reasoning_v2_semantic.py`

**Interfaces:**
- Consumes:
  `InputCatalog`,
  `corpusgen.reasoning.FactMetadata`,
  `route_score`,
  `minimally_rounded_quota`,
  `plan_occurrence_closure`, and
  `audit_occurrence_closure`.
- Produces:
  `SemanticRole`,
  `TokenSemanticSpan`,
  `RouteArtifacts`,
  `RouteIndex`,
  `SidecarWeights`,
  `build_route_artifacts(catalog: InputCatalog, work_root: Path, output_root: Path) -> RouteArtifacts`, and
  `derive_sidecar_weights(token_count: int, spans: tuple[TokenSemanticSpan, ...], routes: RouteIndex) -> SidecarWeights`.

- [ ] **Step 1: RED — test external routing parity and token closure**

```python
def test_production_route_manifest_matches_reviewed_in_memory_policy(
    tmp_path,
    routing_catalog,
):
    artifacts = build_route_artifacts(
        routing_catalog,
        tmp_path / "work",
        tmp_path / "routes",
    )
    expected = build_route_manifest(routing_catalog.fact_metadata, "Split90")
    with artifacts.open_index() as index:
        assert tuple(index.iter_external_fact_ids()) == expected.external_fact_ids
    assert artifacts.distinct_fact_fraction >= Fraction(9, 10)
    assert artifacts.information_burden_fraction >= Fraction(9, 10)
    assert artifacts.dose_report["passed"] is True


def test_split90_masks_only_every_routed_factual_payload_token(route_index):
    spans = (
        TokenSemanticSpan(0, 2, "fact-a", "factual_payload"),
        TokenSemanticSpan(2, 4, "fact-a", "answer_state"),
        TokenSemanticSpan(4, 6, None, "rule"),
        TokenSemanticSpan(6, 8, "fact-a", "factual_payload"),
    )
    weights = derive_sidecar_weights(8, spans, route_index({"fact-a"}))
    assert weights.dense == b"\x01" * 8
    assert weights.split90 == b"\x00\x00\x01\x01\x01\x01\x00\x00"
    assert weights.leaks == ()


def test_answer_state_surface_copy_fails_closed(route_index):
    with pytest.raises(SemanticLeakageError, match="answer state"):
        audit_answer_state_surfaces(
            answer_state="<|answer_state|>{\"value\":\"cerulean\"}",
            routed_facts=(SemanticFact("fact-a", ("cerulean",)),),
        )
```

Add tests that reverse fact input, force external-sort chunks of three rows,
exercise burden repair across chunk boundaries, introduce one unmasked token,
overlap two fact surfaces, omit a routed fact, mark a proof token as factual,
use a nonbinary mask, and route less than either 90% threshold. Repeat one fact
across graph-exposure records and require one route row with aggregated
`scheduled_exposures`; repeat it with conflicting source/surface metadata and
require failure.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_semantic.py
```

Expected: import fails because `corpusgen.reasoning_v2.semantic` is absent.

- [ ] **Step 3: GREEN — implement external sort and exact token spans**

Use these public values:

```python
SemanticRole = Literal[
    "plain_text",
    "factual_payload",
    "rule",
    "operator",
    "schema",
    "proof",
    "answer_state",
]


@dataclass(frozen=True)
class TokenSemanticSpan:
    token_start: int
    token_end: int
    fact_id: str | None
    role: SemanticRole


@dataclass(frozen=True)
class SidecarWeights:
    dense: bytes
    split90: bytes
    routed_payload_targets: int
    leaks: tuple[dict[str, object], ...]


class RouteIndex:
    @classmethod
    def open(cls, database_path: Path) -> RouteIndex:
        return cls(database_path)

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        uri = f"file:{database_path.resolve()}?mode=ro"
        self._connection = sqlite3.connect(uri, uri=True)

    def is_external(self, fact_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM selected WHERE fact_id = ?",
            (fact_id,),
        ).fetchone()
        return row is not None

    def iter_external_fact_ids(self) -> Iterator[str]:
        rows = self._connection.execute(
            "SELECT fact_id FROM selected ORDER BY fact_id"
        )
        return (str(row[0]) for row in rows)

    def __enter__(self) -> RouteIndex:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._connection.close()


@dataclass(frozen=True)
class RouteArtifacts:
    manifest_path: Path
    index_path: Path
    dose_report_path: Path
    manifest_sha256: str
    dose_report_sha256: str
    external_fact_count: int
    distinct_fact_fraction: Fraction
    information_burden_fraction: Fraction
    dose_report: dict[str, object]

    def open_index(self) -> RouteIndex:
        return RouteIndex.open(self.index_path)
```

Implement `external_sort_rows()` by sorting bounded chunks with exact
`Fraction` keys, writing canonical chunk JSONL, and merging with
`heapq.merge(*chunk_iterators, key=sort_key)`. Use reviewed `route_score()` and
`minimally_rounded_quota()` unchanged. Store selected IDs in a private SQLite
database with `PRIMARY KEY(fact_id)`, perform the same incoming-highest-burden
and outgoing-lowest-burden swaps as `build_route_manifest()`, and emit one
canonical decision row per ranked fact after the dose passes. The final route
manifest begins with one header row containing format
`memorysplit-production-route-manifest-v1`, policy
`train-score-ranked-quota-v1`, metadata scope `training-only`, exact rational
dose values, row count, and decision-stream SHA-256.

Before sorting, reduce repeated occurrences by `fact_id`. Require source,
record type, entropy, expected reads/hops, and surfaces to be byte-identical,
then sum scheduled exposures. This makes graph revisits contribute burden
without counting one atomic fact more than once.

`derive_sidecar_weights()` initializes both byte arrays to one, validates that
spans are ordered, in bounds, and non-overlapping unless they name the same
fact, then zeros Split90 only for `factual_payload` spans whose fact is in the
read-only `RouteIndex`. It rescans every declared routed occurrence and raises
`SemanticLeakageError` if any target remains one.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_semantic.py \
  tests/test_reasoning_v2.py
```

Expected: all tests pass, proving parity with the reviewed local routing core.

- [ ] **Step 5: Commit**

```bash
git add corpusgen/reasoning_v2/semantic.py \
  tests/test_reasoning_v2_semantic.py
git commit -m "feat: add v2 semantic sidecar routing"
```

---

### Task 5A: Implement renderer contracts and four exposure lanes

**Files:**
- Create: `corpusgen/reasoning_v2/renderers.py`
- Create: `tests/test_reasoning_v2_renderers.py`

**Interfaces:**
- Consumes:
  `CatalogRecord`,
  verified source root,
  `RouteIndex`,
  `get_tok()`,
  `iter_training_triples()`,
  `iter_aliases()`,
  `iter_worlds()`, and
  `iter_graph_records()`.
- Produces:
  `ProofEnvelope`,
  `ProductionRenderedRecord`,
  `ProductionLaneRenderer`,
  `FineWebEduRenderer`,
  `FineMathRenderer`,
  `WikidataGraphRenderer`, and
  `SyntheticGraphRenderer`.

- [ ] **Step 1: RED — test the shared contract and four exposure lanes**

```python
@pytest.mark.parametrize(
    "lane_id",
    ["fineweb_edu", "finemath", "wikidata_graph", "synthetic_graph"],
)
def test_exposure_renderer_is_exact_deterministic_and_closed(
    lane_id,
    renderer_fixture,
):
    renderer, record, routes = renderer_fixture(lane_id)
    first = renderer.render(record, routes)
    second = renderer.render(record, routes)
    assert first == second
    assert len(first.token_ids) == record.target_count
    assert first.dense_target_weights == b"\x01" * record.target_count
    assert len(first.split90_target_weights) == record.target_count
    assert set(first.split90_target_weights) <= {0, 1}
    assert all(0 <= token < 65_536 for token in first.token_ids)
    assert first.proofs == ()
    assert first.semantic_leaks == ()
```

Add four focused tests: FineWeb-Edu reads only the three locked 10BT files;
FineMath consumes `finemath-4plus` before cross-deduplicated
`finemath-3plus`; Wikidata graph emits every training triple before a
`graph-revisit`; synthetic graph reproduces the exact catalog seed and frozen
`WorldConfig`. Reject source hash drift, non-NFC text, a non-finite payload,
duplicate record ID, oversized core, and target-count drift.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_renderers.py
```

Expected: import fails because `corpusgen.reasoning_v2.renderers` is missing.

- [ ] **Step 3: GREEN — implement the common record and exposure renderers**

```python
@dataclass(frozen=True)
class ProofEnvelope:
    family: str
    premise_bytes: bytes
    proof_bytes: bytes
    proof_sha256: str
    replay_verified: bool


@dataclass(frozen=True)
class ProductionRenderedRecord:
    record_id: str
    lane_id: LaneId
    token_ids: tuple[int, ...]
    dense_target_weights: bytes
    split90_target_weights: bytes
    semantic_spans: tuple[TokenSemanticSpan, ...]
    proofs: tuple[ProofEnvelope, ...]
    flags: tuple[str, ...]
    semantic_leaks: tuple[dict[str, object], ...]


class ProductionLaneRenderer(Protocol):
    lane_id: LaneId
    renderer_version: str

    def render(
        self,
        record: CatalogRecord,
        routes: RouteIndex,
    ) -> ProductionRenderedRecord:
        raise RuntimeError("production lane renderer must render one catalog record")


EXPOSURE_RENDERER_TYPES = (
    ("fineweb_edu", FineWebEduRenderer, "fineweb-edu-nfc-gpt2-v1"),
    ("finemath", FineMathRenderer, "finemath-cross-dedup-gpt2-v1"),
    ("wikidata_graph", WikidataGraphRenderer, "wikidata-training-graph-v1"),
    ("synthetic_graph", SyntheticGraphRenderer, "srgm-seeded-graph-v1"),
)
```

Text renderers pack NFC source text to the exact target length and terminate
with EOT. FineMath uses SHA-256 of NFC UTF-8 text for exact FineWeb/4plus/3plus
deduplication. Graph renderers insert only indexed
`<|graph_noop|><|graph_step|>` schema tokens before EOT when the semantic core
is shorter than the requested length. Call `derive_sidecar_weights()` after
fitting and require no leaks.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_renderers.py \
  tests/test_tokenizer.py \
  tests/test_srgm_worlds.py
```

Expected: the four exposure renderers and existing tokenizer/world tests pass.

- [ ] **Step 5: Commit**

```bash
git add corpusgen/reasoning_v2/renderers.py \
  tests/test_reasoning_v2_renderers.py
git commit -m "feat: add v2 exposure renderers"
```

---

### Task 5B: Implement the three solver-backed reasoning lanes

**Files:**
- Modify: `corpusgen/reasoning_v2/renderers.py`
- Modify: `tests/test_reasoning_v2_renderers.py`

**Interfaces:**
- Consumes:
  the Task 5A renderer contract,
  `iter_reasoning_records()`,
  `solve_graph_composition()`,
  `solve_slot_equality()`, and
  `serialize_answer_state()`.
- Produces:
  `VerifiedSyntheticMultihopRenderer`,
  `WikidataPathReasoningRenderer`, and
  `RelationalRefinementRenderer`.

- [ ] **Step 1: RED — test proof replay, sealed-path rejection, and value-free state**

```python
@pytest.mark.parametrize(
    "lane_id",
    [
        "verified_synthetic_multihop",
        "wikidata_path_reasoning",
        "relational_refinement",
    ],
)
def test_reasoning_renderer_has_replayable_proof_and_no_leak(
    lane_id,
    renderer_fixture,
):
    renderer, record, routes = renderer_fixture(lane_id)
    rendered = renderer.render(record, routes)
    assert len(rendered.token_ids) == record.target_count
    assert rendered.proofs
    assert all(proof.replay_verified for proof in rendered.proofs)
    assert rendered.semantic_leaks == ()


def test_refinement_candidate_and_final_states_are_value_free(renderer_fixture):
    renderer, record, routes = renderer_fixture("relational_refinement")
    rendered = renderer.render(record, routes)
    decoded = get_tok().decode(list(rendered.token_ids))
    for fact in record.semantic_facts:
        assert all(surface not in decoded_answer_states(decoded) for surface in fact.surfaces)
```

Add tests that synthetic multihop replays canonical composition/equality,
Wikidata path records use only training triples, a sealed triple fails, a
solver disagreement fails, a premise mutation fails replay, and every
candidate/final state uses `AnswerPointer`.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_renderers.py -k reasoning
```

Expected: imports of the three new renderer classes fail.

- [ ] **Step 3: GREEN — add exact solver renderer versions**

```python
REASONING_RENDERER_TYPES = (
    (
        "verified_synthetic_multihop",
        VerifiedSyntheticMultihopRenderer,
        "srgm-canonical-proof-v1",
    ),
    (
        "wikidata_path_reasoning",
        WikidataPathReasoningRenderer,
        "wikidata-training-path-proof-v1",
    ),
    (
        "relational_refinement",
        RelationalRefinementRenderer,
        "pointer-state-refinement-v1",
    ),
)
```

Each renderer serializes canonical premise bytes and canonical proof bytes into
`ProofEnvelope`, calls `verify_proof()` before returning, and marks proof tokens
with role `proof`. Factual returns alone use role `factual_payload`. Candidate
and final state fields contain only `serialize_answer_state(AnswerPointer(...))`.
Fit the requested target length with indexed schema/proof steps, never repeated
factual surfaces.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_renderers.py \
  tests/test_reasoning_v2.py \
  tests/test_srgm_worlds.py
```

Expected: all exposure and solver-backed renderer tests pass.

- [ ] **Step 5: Commit**

```bash
git add corpusgen/reasoning_v2/renderers.py \
  tests/test_reasoning_v2_renderers.py
git commit -m "feat: add solver-backed v2 renderers"
```

---

### Task 5C: Implement objective auxiliary and freeze the full registry

**Files:**
- Modify: `corpusgen/reasoning_v2/renderers.py`
- Modify: `tests/test_reasoning_v2_renderers.py`

**Interfaces:**
- Consumes:
  Task 5A/5B renderers,
  `iter_puzzle_tasks()`,
  locked DeepMind mathematics, CLRS-Text, RuleTaker, ProntoQA, and Reasoning
  Gym sources.
- Produces:
  `ObjectiveAuxiliaryRenderer`,
  `RendererRegistry`,
  `production_renderer_versions() -> tuple[tuple[LaneId, str], ...]`, and
  `build_renderer_registry(source_lock: SourceLock, source_root: Path) -> RendererRegistry`.

- [ ] **Step 1: RED — test every objective source and final lane order**

```python
def test_objective_renderer_uses_every_locked_exact_source(renderer_fixture):
    renderer, records, routes = objective_renderer_fixture(renderer_fixture)
    rendered = [renderer.render(record, routes) for record in records]
    assert {row.source_id for row in records} == {
        "deepmind_mathematics_generator",
        "clrs_text",
        "ruletaker",
        "prontoqa",
        "reasoning_gym_exact_answer",
        "arc_agi_1",
        "arc_agi_2",
        "conceptarc",
    }
    assert all(item.proofs and item.semantic_leaks == () for item in rendered)


def test_renderer_registry_has_exact_frozen_lane_order(staged_fixture_sources):
    registry = build_renderer_registry(
        staged_fixture_sources.lock,
        staged_fixture_sources.root,
    )
    assert tuple(registry) == LANE_ORDER
    assert len({renderer.renderer_version for renderer in registry.values()}) == 8
```

Add tests for exact answers, mutated answers, ARC/ConceptARC combined cap,
Hamilton reallocation, teacher-generated target count zero, evaluation-task
rejection, unknown source, and a registry version mutation.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_renderers.py -k "objective or registry"
```

Expected: imports of `ObjectiveAuxiliaryRenderer` and `RendererRegistry` fail.

- [ ] **Step 3: GREEN — implement objective scheduling and registry identity**

```python
RENDERER_TYPES = (
    *EXPOSURE_RENDERER_TYPES,
    *REASONING_RENDERER_TYPES,
    ("objective_auxiliary", ObjectiveAuxiliaryRenderer, "locked-exact-objective-v1"),
)


@dataclass(frozen=True)
class RendererRegistry:
    renderers: tuple[tuple[LaneId, ProductionLaneRenderer], ...]

    def __iter__(self) -> Iterator[LaneId]:
        return (lane for lane, _renderer in self.renderers)

    def values(self) -> tuple[ProductionLaneRenderer, ...]:
        return tuple(renderer for _lane, renderer in self.renderers)

    def for_lane(self, lane_id: LaneId) -> ProductionLaneRenderer:
        return dict(self.renderers)[lane_id]

    @property
    def renderer_id(self) -> str:
        return "reasoning-v2-production:" + sha256_hex(
            canonical_json_bytes(production_renderer_versions())
        )
```

Allocate objective targets by Hamilton equally over the five unbounded exact
generators in source-policy order. ARC-AGI, ARC-AGI-2, and ConceptARC share the
exact `floor(full_targets * 0.0025)` ceiling; reallocate unused finite-source
targets over the five unbounded generators by Hamilton. Every objective record
stores an exact-answer or solver-replay proof envelope. The registry requires
exactly `LANE_ORDER`, hashes all eight versions into `renderer_id`, and rejects
extra/missing renderer classes.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_renderers.py \
  tests/test_reasoning_v2.py \
  tests/test_tokenizer.py \
  tests/test_srgm_worlds.py
```

Expected: all eight renderer lanes, objective caps, tokenizer assets, and proof
primitives pass.

- [ ] **Step 5: Commit**

```bash
git add corpusgen/reasoning_v2/renderers.py \
  tests/test_reasoning_v2_renderers.py
git commit -m "feat: complete v2 renderer registry"
```

---

### Task 6A: Build resumable task spools

**Files:**
- Create: `corpusgen/reasoning_v2/tasks.py`
- Create: `tests/test_reasoning_v2_tasks.py`

**Interfaces:**
- Consumes:
  `InputCatalog`,
  `RendererRegistry`,
  `RouteIndex`,
  `BuildGeometry`, and
  safe no-replace primitives from `corpusgen.parallel.safeio`.
- Produces:
  `TaskSpoolArtifact`,
  `TaskRecordIndex`,
  `TaskReceipt`,
  `render_task_spool(catalog: InputCatalog, renderers: RendererRegistry, routes: RouteIndex, geometry: BuildGeometry, *, task_index: int, task_count: int, workers: int, local_root: Path) -> TaskReceipt`,
  `publish_task_spool_via_local_cache(receipt: TaskReceipt, *, local_root: Path, shared_root: Path, scheduler_id: str, job_id: str, nonce: str) -> Path`,
  and `load_task_receipts(shared_root: Path, *, build_key: str, scheduler_id: str, nonce: str, expected_task_count: int) -> tuple[TaskReceipt, ...]`.

- [ ] **Step 1: RED — test task order, resume, and replay**

```python
def test_task_spools_are_disjoint_complete_and_worker_order_independent(
    tmp_path,
    production_fixture,
):
    serial = tuple(
        render_task_spool(
            production_fixture.catalog,
            production_fixture.renderers,
            production_fixture.routes,
            production_fixture.geometry,
            task_index=index,
            task_count=4,
            workers=1,
            local_root=tmp_path / "serial",
        )
        for index in range(4)
    )
    parallel = tuple(
        render_task_spool(
            production_fixture.catalog,
            production_fixture.renderers,
            production_fixture.routes,
            production_fixture.geometry,
            task_index=index,
            task_count=4,
            workers=3,
            local_root=tmp_path / "parallel",
        )
        for index in reversed(range(4))
    )
    assert tuple(row.receipt_sha256 for row in serial) == tuple(
        row.receipt_sha256 for row in reversed(parallel)
    )
    assert set().union(*(set(row.ordinals) for row in serial)) == set(
        range(production_fixture.catalog.record_count)
    )


def test_interrupted_task_is_reused_only_after_full_receipt_verification(
    tmp_path,
    production_fixture,
):
    first = render_and_publish_fixture_task(production_fixture, tmp_path, 0, 4)
    assert render_and_publish_fixture_task(
        production_fixture,
        tmp_path,
        0,
        4,
    ) == first
    first.tokens_path.write_bytes(first.tokens_path.read_bytes() + b"\x00\x00")
    with pytest.raises(ValueError, match="task spool digest drift"):
        render_and_publish_fixture_task(production_fixture, tmp_path, 0, 4)
```

Add tests for missing/duplicate/foreign task receipts, wrong task count, build
key drift, catalog drift, route drift, token/sidecar/proof spool tampering,
record-index offset drift, task replay disagreement, reverse completion order,
one task crash before publication, a symlink in local cache, a hardlink in
shared staging, and a foreign file in a resumed task directory.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_tasks.py
```

Expected: import fails because `corpusgen.reasoning_v2.tasks` is missing.

- [ ] **Step 3: GREEN — implement bounded spools**

Each task directory contains exactly:

```text
record-index.jsonl
tokens.bin
dense_target_weights.bin
split90_target_weights.bin
proofs.jsonl
semantic-spans.jsonl
semantic-leaks.jsonl
task-receipt.json
```

Use these public receipt fields:

```python
@dataclass(frozen=True)
class TaskSpoolArtifact:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class TaskRecordIndex:
    ordinal: int
    record_id: str
    lane_id: LaneId
    token_offset: int
    target_count: int
    token_sha256: str
    dense_sha256: str
    split90_sha256: str
    proof_count: int


@dataclass(frozen=True)
class TaskReceipt:
    format: Literal["memorysplit-v2-task-spool-v1"]
    build_key: str
    catalog_sha256: str
    source_lock_sha256: str
    route_manifest_sha256: str
    renderer_id: str
    task_index: int
    task_count: int
    record_count: int
    target_count: int
    ordinals: tuple[int, ...]
    artifacts: tuple[TaskSpoolArtifact, ...]
    receipt_sha256: str
```

`render_task_spool()` selects records where
`record.ordinal % task_count == task_index`. Process bounded chunks of 4,096
records in a `ThreadPoolExecutor`; sort each completed chunk by ordinal before
appending to the three binary spools and canonical indexes. A task receipt
hashes the canonical unsigned receipt and every artifact. Validate the complete
node-local directory, copy each regular file to a private shared directory,
validate it again, and publish the directory with no-replace rename.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_tasks.py -k "task or spool or receipt"
```

Expected: task partition, worker-order, resume, and tamper tests pass.

- [ ] **Step 5: Commit**

```bash
git add corpusgen/reasoning_v2/tasks.py \
  tests/test_reasoning_v2_tasks.py
git commit -m "feat: add resumable v2 task spools"
```

---

### Task 6B: Reduce the exact schedule into aligned candidate shards

**Files:**
- Modify: `corpusgen/reasoning_v2/tasks.py`
- Modify: `tests/test_reasoning_v2_tasks.py`

**Interfaces:**
- Consumes:
  complete verified `TaskReceipt` values,
  `InputCatalog`, and
  `BuildGeometry`.
- Produces:
  `ProductionShardAssignment`,
  `ScheduleSummary`,
  `CandidateDataset`,
  `production_shard_assignments(geometry: BuildGeometry) -> tuple[ProductionShardAssignment, ...]`,
  `write_largest_deficit_schedule(catalog: InputCatalog, receipts: tuple[TaskReceipt, ...], geometry: BuildGeometry, output_root: Path) -> ScheduleSummary`, and
  `assemble_candidate_dataset(catalog: InputCatalog, receipts: tuple[TaskReceipt, ...], schedule: ScheduleSummary, geometry: BuildGeometry, output_root: Path) -> CandidateDataset`.

- [ ] **Step 1: RED — test exact quotas, schedule, and 32 shard assignments**

```python
def test_full_geometry_assigns_32_whole_update_shards(full_geometry):
    assignments = production_shard_assignments(full_geometry)
    assert len(assignments) == 32
    assert [row.shard_name for row in assignments] == [
        f"shard-{index:05d}.bin" for index in range(32)
    ]
    assert sum(row.update_end - row.update_start for row in assignments) == 13_582
    assert [row.update_end - row.update_start for row in assignments] == (
        [425] * 14 + [424] * 18
    )
    assert assignments[-1].token_end == 7_120_879_616


def test_reverse_task_completion_produces_identical_candidate(production_fixture):
    forward = reduce_fixture_candidate(production_fixture, reverse=False)
    reverse = reduce_fixture_candidate(production_fixture, reverse=True)
    assert forward.ordered_stream_sha256 == reverse.ordered_stream_sha256
    assert forward.dense_stream_sha256 == reverse.dense_stream_sha256
    assert forward.split90_stream_sha256 == reverse.split90_stream_sha256
    assert forward.lane_targets == production_fixture.geometry.lane_quotas
```

Add schedule quota mismatch, schedule record duplication/omission, lane-order
tie, source-offset drift, missing task, sidecar length drift, nonbinary
sidecar, and nonzero full padding tests.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_tasks.py -k "schedule or shard or candidate"
```

Expected: imports of `ProductionShardAssignment` and reducer functions fail.

- [ ] **Step 3: GREEN — implement streaming schedule and shard assembly**

```python

@dataclass(frozen=True)
class ProductionShardAssignment:
    shard_index: int
    shard_count: int
    update_start: int
    update_end: int
    token_start: int
    token_end: int

    @property
    def shard_name(self) -> str:
        return f"shard-{self.shard_index:05d}.bin"


@dataclass(frozen=True)
class ScheduleSummary:
    schedule_path: Path
    assignments_path: Path
    metadata_path: Path
    schedule_sha256: str
    assignments_sha256: str
    metadata_sha256: str
    record_count: int
    logical_tokens: int
    lane_targets: tuple[tuple[LaneId, int], ...]
    assignments: tuple[ProductionShardAssignment, ...]


@dataclass(frozen=True)
class CandidateDataset:
    root: Path
    catalog_path: Path
    metadata_path: Path
    schedule_path: Path
    assignments_path: Path
    token_shards: tuple[Path, ...]
    dense_shards: tuple[Path, ...]
    split90_shards: tuple[Path, ...]
    proof_manifest_path: Path
    semantic_span_manifest_path: Path
    semantic_leak_report_path: Path
    logical_tokens: int
    packed_tokens: int
    lane_targets: tuple[tuple[LaneId, int], ...]
    ordered_stream_sha256: str
    merkle_root_sha256: str
    packed_stream_sha256: str
    dense_stream_sha256: str
    split90_stream_sha256: str
```

`write_largest_deficit_schedule()` must stream lane heads and choose:

```python
lane = min(
    active_lanes,
    key=lambda lane_id: (
        -(quota_by_lane[lane_id] - emitted_by_lane[lane_id]),
        LANE_ORDER.index(lane_id),
    ),
)
```

Within a lane, consume increasing catalog ordinal. Write generic
`ScheduleRecord` JSONL and canonical `ProductionShardAssignment` JSONL. The
reducer opens all 32 verified task indexes, seeks record payloads by offset,
validates each record digest while copying in schedule order, and writes
`shard-00000.bin` through `shard-00031.bin` for tokens and both sidecar sets. It
must require logical targets equal packed targets, making full-build padding
zero.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_tasks.py \
  tests/test_parallel_corpus.py
```

Expected: exact schedule, candidate, and existing fixture task tests pass.

- [ ] **Step 5: Commit**

```bash
git add corpusgen/reasoning_v2/tasks.py \
  tests/test_reasoning_v2_tasks.py
git commit -m "feat: assemble exact v2 corpus shards"
```

---

### Task 7: Publish production inner and outer receipts

**Files:**
- Create: `corpusgen/reasoning_v2/publication.py`
- Create: `tests/test_reasoning_v2_publication.py`

**Interfaces:**
- Consumes:
  `CandidateDataset`,
  source lock bytes,
  route/dose artifacts,
  proof/leak manifests,
  32 `TaskReceipt` values, and
  `ReleaseGateEvidence`.
- Produces:
  `ArtifactReceipt`,
  `SidecarSetReceipt`,
  `ProductionBindings`,
  `InnerCorpusReceipt`,
  `ReleaseGateEvidence`,
  `OuterMaterializationReceipt`,
  `read_inner_receipt(path: Path) -> InnerCorpusReceipt`,
  `production_build_id(bindings: ProductionBindings, *, ordered_stream_sha256: str, merkle_root_sha256: str, packed_stream_sha256: str, dense_stream_sha256: str, split90_stream_sha256: str, artifacts: tuple[ArtifactReceipt, ...]) -> str`,
  `write_release_candidate(candidate_dataset: CandidateDataset, bindings: ProductionBindings, source_lock_bytes: bytes, route_artifacts: RouteArtifacts, proof_manifest_path: Path, semantic_leak_report_path: Path, output_root: Path) -> Path`,
  `write_verified_canary_receipt(candidate: Path, evidence: ReleaseGateEvidence) -> OuterMaterializationReceipt`, and
  `publish_verified_release(candidate: Path, releases_root: Path, evidence: ReleaseGateEvidence) -> OuterMaterializationReceipt`.

- [ ] **Step 1: RED — test exact receipt layering and no-replace gates**

```python
def test_inner_receipt_binds_every_production_commitment(release_candidate):
    receipt = read_inner_receipt(
        release_candidate / "dataset/corpus-receipt.json"
    )
    assert receipt.format == "memorysplit-parallel-corpus-v2"
    assert receipt.logical_tokens == 7_120_879_616
    assert receipt.packed_tokens == receipt.logical_tokens
    assert receipt.padding_tokens == 0
    assert receipt.shard_count == 32
    assert tuple(row.name for row in receipt.sidecar_sets) == (
        "dense_target_weights",
        "split90_target_weights",
    )
    bindings = receipt.production_bindings
    assert bindings.source_lock_sha256 == release_candidate.source_lock_sha256
    assert bindings.route_manifest_sha256 == release_candidate.route_sha256
    assert bindings.proof_manifest_sha256 == release_candidate.proof_sha256
    assert len(bindings.task_receipt_sha256s) == 32


def test_outer_receipt_cannot_exist_before_all_gates_pass(
    tmp_path,
    release_candidate,
    passing_gate_evidence,
):
    failed = replace(
        passing_gate_evidence,
        second_build_commitments_match=False,
    )
    with pytest.raises(ValueError, match="second full build"):
        publish_verified_release(
            release_candidate,
            tmp_path / "releases",
            failed,
        )
    assert not (tmp_path / "releases").exists()


def test_canary_has_both_receipt_layers_but_never_allows_protected_launch(
    canary_candidate,
    passing_canary_evidence,
):
    receipt = write_verified_canary_receipt(
        canary_candidate,
        passing_canary_evidence,
    )
    assert receipt.profile == "canary"
    assert receipt.scientific_status == "verified_canary_only"
    assert receipt.protected_launch_allowed is False


def test_publication_is_atomic_noreplace_and_idempotent(
    tmp_path,
    release_candidate,
    passing_gate_evidence,
):
    first = publish_verified_release(
        release_candidate,
        tmp_path / "releases",
        passing_gate_evidence,
    )
    published = tmp_path / "releases" / first.build_id
    before = tree_file_identities(published)
    second = publish_verified_release(
        release_candidate,
        tmp_path / "releases",
        passing_gate_evidence,
    )
    assert second == first
    assert tree_file_identities(published) == before
```

Also test outer/inner hash mismatch, wrong dataset ID, wrong source commit,
wrong lane quota, realized-count mismatch, graph incomplete, semantic rate below
`1.0`, overlap above `0.0`, leak count nonzero, dose below 90%, missing task,
fixture renderer ID, legacy receipt format, output collision, symlinked
candidate, destination race, interrupted candidate, and a preexisting foreign
release. Set the repository config's `protected_launch_allowed` field to both
boolean values in test copies and prove neither can bypass the full gate
evidence.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_publication.py
```

Expected: import fails because `corpusgen.reasoning_v2.publication` is absent.

- [ ] **Step 3: GREEN — implement both strict receipt schemas**

The inner receipt uses these exact row types and fields:

```python
@dataclass(frozen=True)
class ArtifactReceipt:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class SidecarSetReceipt:
    name: Literal["dense_target_weights", "split90_target_weights"]
    dtype: Literal["uint8"]
    items: int
    stream_sha256: str
    artifacts: tuple[ArtifactReceipt, ...]


@dataclass(frozen=True)
class InnerCorpusReceipt:
    artifacts: tuple[ArtifactReceipt, ...]
    assignments_sha256: str
    build_id: str
    catalog_sha256: str
    compiler_version: str
    config: ParallelBuildConfig
    format: Literal["memorysplit-parallel-corpus-v2"]
    logical_tokens: int
    merkle_root_sha256: str
    metadata_sha256: str
    ordered_stream_sha256: str
    packed_stream_sha256: str
    packed_tokens: int
    padding_tokens: int
    record_count: int
    renderer_id: str
    schedule_sha256: str
    shard_count: int
    sidecar_sets: tuple[SidecarSetReceipt, SidecarSetReceipt]
    production_bindings: ProductionBindings


@dataclass(frozen=True)
class ProductionBindings:
    schema_version: Literal[1]
    dataset_id: str
    source_git_commit: str
    source_lock_sha256: str
    renderer_versions: tuple[tuple[LaneId, str], ...]
    catalog_sha256: str
    metadata_sha256: str
    schedule_sha256: str
    assignments_sha256: str
    route_manifest_sha256: str
    dose_report_sha256: str
    proof_manifest_sha256: str
    semantic_span_manifest_sha256: str
    semantic_leak_report_sha256: str
    lane_quotas: tuple[tuple[LaneId, int], ...]
    realized_lane_targets: tuple[tuple[LaneId, int], ...]
    task_receipt_sha256s: tuple[str, ...]
```

Keep all existing base v2 fields and append only the
`production_bindings` object for production receipts. Inventory
`catalog.jsonl`, `metadata.jsonl`, `schedule.jsonl`, `assignments.jsonl`,
32 token shards, both 32-shard sidecar sets, `manifests/source-lock.json`,
`manifests/route-manifest.jsonl`, `manifests/dose-report.json`,
`manifests/semantic-spans.jsonl`, `manifests/semantic-leaks.json`, and
`proofs/proof-manifest.jsonl`.
`production_build_id()` hashes the canonical bindings, ordered token
commitment, both logical sidecar commitments, and all artifact SHA-256 values.

The independent verifier returns this exact pre-publication gate object:

```python
@dataclass(frozen=True)
class ReleaseGateEvidence:
    profile: Literal["canary", "full"]
    candidate_verification_sha256: str
    inner_verified: bool
    source_lock_verified: bool
    graph_complete_once: bool
    semantic_verification_rate: Fraction
    structural_train_evaluation_overlap: Fraction
    distinct_fact_dose: Fraction
    information_burden_dose: Fraction
    semantic_leak_count: int
    solver_sample_count: int
    solver_verified_count: int
    second_build_commitments_sha256: str
    second_build_commitments_match: bool


@dataclass(frozen=True)
class PublicationIdentity:
    method: Literal["atomic-rename-noreplace"]
    build_id: str
    candidate_verification_sha256: str
    second_build_commitments_sha256: str
    no_replace: Literal[True]
```

`OuterMaterializationReceipt` contains exactly:

```python
@dataclass(frozen=True)
class OuterMaterializationReceipt:
    schema_version: Literal[1]
    format: Literal["memorysplit-v2-materialization-receipt-v1"]
    profile: Literal["canary", "full"]
    dataset_id: str
    source_git_commit: str
    source_lock_path: Literal["source-lock.json"]
    source_lock_sha256: str
    inner_receipt_path: Literal["dataset/corpus-receipt.json"]
    inner_receipt_sha256: str
    build_id: str
    lane_quotas: tuple[tuple[LaneId, int], ...]
    realized_lane_targets: tuple[tuple[LaneId, int], ...]
    graph_coverage: dict[str, object]
    semantic_verification_rate: float
    structural_train_evaluation_overlap: float
    route_dose: dict[str, object]
    semantic_leakage: dict[str, object]
    publication_identity: PublicationIdentity
    scientific_status: Literal["verified_canary_only", "verified_corpus_ready"]
    protected_launch_allowed: bool
```

Require `semantic_verification_rate == 1.0`,
`structural_train_evaluation_overlap == 0.0`, both exact dose fractions at
least `9/10`, zero leaks, complete graph coverage, a passing FarmShare
candidate-verification hash, and matching second-build commitments before
constructing the full outer receipt. `graph_coverage` has exact keys
`training_edges`, `covered_training_edges`, `complete_once`, and
`revisit_started_after_complete`; `route_dose` has exact rational fact/burden
fractions and pass booleans; `semantic_leakage` has `leak_count=0` and the
semantic report SHA-256. A canary outer receipt uses `profile="canary"`,
`scientific_status="verified_canary_only"`, and
`protected_launch_allowed=False`; it remains under `work/` and can never enter
the canonical `releases/` namespace. A full outer receipt uses
`profile="full"`, `scientific_status="verified_corpus_ready"`, and
`protected_launch_allowed=True`. Write it last inside the private candidate,
fsync, verify the candidate, then atomically rename the complete `<build-id>`
directory into `releases/`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_publication.py \
  tests/test_parallel_corpus.py
```

Expected: all tests pass; fixture receipts retain their previous exact field
sets and bytes.

- [ ] **Step 5: Commit**

```bash
git add corpusgen/reasoning_v2/publication.py \
  tests/test_reasoning_v2_publication.py
git commit -m "feat: publish layered v2 corpus receipts"
```

---

### Task 8: Add streaming independent verification and loader dispatch

**Files:**
- Create: `corpusgen/reasoning_v2/verification.py`
- Modify: `corpusgen/parallel/publication.py`
- Modify: `corpusgen/parallel/__init__.py`
- Modify: `train/data.py`
- Create: `tests/test_reasoning_v2_verification.py`
- Modify: `tests/test_parallel_corpus.py`
- Modify: `tests/test_sharded_loader.py`

**Interfaces:**
- Consumes:
  production inner/outer receipts and regular-file release paths.
- Produces:
  `StreamCommitments`,
  `ProofSampleResult`,
  `CandidateVerification`,
  `SiteVerificationReceipt`,
  `verify_inner_corpus(path: Path, expected_build_id: str | None = None) -> InnerCorpusReceipt`,
  `verify_release_candidate(path: Path) -> CandidateVerification`,
  `verify_published_release(path: Path, site: Literal["farmshare", "mit"]) -> SiteVerificationReceipt`,
  `rebuild_commitments(catalog: InputCatalog, renderers: RendererRegistry, routes: RouteIndex, geometry: BuildGeometry, work_root: Path, workers: int) -> StreamCommitments`,
  `compare_rebuild(reference: InnerCorpusReceipt, rebuilt: StreamCommitments) -> bool`, and
  `build_release_gate_evidence(candidate: CandidateVerification, rebuilt: StreamCommitments) -> ReleaseGateEvidence`, and
  `write_site_verification(release_path: Path, site: Literal["farmshare", "mit"], destination: Path) -> SiteVerificationReceipt`.

- [ ] **Step 1: RED — write adversarial verifier and loader tests**

```python
@pytest.mark.parametrize(
    "mutation",
    [
        "token-byte",
        "dense-zero",
        "split90-nonbinary",
        "sidecar-reorder",
        "route-byte",
        "proof-byte",
        "source-lock-byte",
        "outer-inner-hash",
        "receipt-append",
    ],
)
def test_independent_verifier_rejects_hash_consistent_and_raw_tampering(
    release_fixture,
    mutation,
):
    mutate_release_and_rewrite_declared_hashes(release_fixture, mutation)
    with pytest.raises(ValueError):
        verify_published_release(release_fixture, site="farmshare")


def test_verifier_replays_deterministic_sample_from_every_solver_lane(
    release_fixture,
):
    evidence = verify_published_release(release_fixture, site="farmshare")
    assert set(evidence.proof_samples) == {
        "verified_synthetic_multihop",
        "wikidata_path_reasoning",
        "relational_refinement",
        "objective_auxiliary",
    }
    assert all(row.sample_count == 64 for row in evidence.proof_samples.values())
    assert all(row.verified_count == row.sample_count for row in evidence.proof_samples.values())


def test_candidate_verifier_returns_prepublication_gate_evidence(
    candidate_fixture,
):
    assert not (candidate_fixture / "receipt.json").exists()
    evidence = verify_release_candidate(candidate_fixture)
    assert evidence.source_lock_verified is True
    assert evidence.semantic_leak_count == 0
    assert all(
        sample.verified_count == sample.sample_count
        for sample in evidence.proof_samples.values()
    )


def test_training_loader_opens_exact_inner_receipt_and_both_sidecars(
    release_fixture,
):
    receipt = release_fixture / "dataset/corpus-receipt.json"
    dense = PackedShards.from_parallel_corpus(
        receipt,
        ctx=16,
        batch_size=2,
        sidecar_name="dense_target_weights",
    )
    split = PackedShards.from_parallel_corpus(
        receipt,
        ctx=16,
        batch_size=2,
        sidecar_name="split90_target_weights",
    )
    assert dense.provenance["build_id"] == split.provenance["build_id"]
    assert dense.provenance["ordered_stream_sha256"] == (
        split.provenance["ordered_stream_sha256"]
    )
    assert dense.provenance["weights"]["sha256"] != split.provenance["weights"]["sha256"]
```

Add symlinked directory/file, hardlink count greater than one, FIFO, socket,
missing file, extra file/directory, in-place mutation during read, same-size
replacement, append after initial `fstat`, EOF drift, wrong lane count, wrong
task receipt, zero proof sample, proof replay failure, graph-coverage
shortfall, train/evaluation overlap, route threshold failure, logical stream
mismatch, Merkle mismatch, and second-build mismatch cases.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_verification.py \
  tests/test_sharded_loader.py
```

Expected: production verification imports or assertions fail; existing loader
tests still pass.

- [ ] **Step 3: GREEN — implement pinned streaming verification**

Use these commitment fields:

```python
@dataclass(frozen=True)
class StreamCommitments:
    logical_tokens: int
    ordered_stream_sha256: str
    merkle_root_sha256: str
    packed_stream_sha256: str
    dense_stream_sha256: str
    split90_stream_sha256: str
    lane_targets: tuple[tuple[LaneId, int], ...]
    source_lock_sha256: str
    route_manifest_sha256: str
    proof_manifest_sha256: str
    build_id: str


@dataclass(frozen=True)
class ProofSampleResult:
    lane_id: LaneId
    sample_count: int
    verified_count: int
    sample_sha256: str


@dataclass(frozen=True)
class CandidateVerification:
    profile: Literal["canary", "full"]
    candidate_path: str
    inner_receipt_sha256: str
    commitments: StreamCommitments
    source_lock_verified: bool
    graph_complete_once: bool
    semantic_verification_rate: Fraction
    structural_train_evaluation_overlap: Fraction
    distinct_fact_dose: Fraction
    information_burden_dose: Fraction
    semantic_leak_count: int
    proof_samples: dict[LaneId, ProofSampleResult]
    verification_sha256: str


@dataclass(frozen=True)
class SiteVerificationReceipt:
    schema_version: Literal[1]
    format: Literal["memorysplit-v2-site-verification-v1"]
    site: Literal["farmshare", "mit"]
    profile: Literal["full"]
    protected_launch_allowed: Literal[True]
    release_path: str
    outer_receipt_sha256: str
    inner_receipt_sha256: str
    commitments: StreamCommitments
    proof_samples: dict[LaneId, ProofSampleResult]
    verified_files: int
    verified_bytes: int
    passed: Literal[True]
    verification_sha256: str
```

Open the release root one path component at a time with
`O_DIRECTORY|O_NOFOLLOW`; open each file with `O_NOFOLLOW`; require regular
mode and `st_nlink == 1`; hold descriptors while reading; compare initial,
final, and named `(device, inode, size, mtime_ns, ctime_ns, nlink)`; read one
byte past expected EOF; and reject any unreceipted namespace entry.

Stream catalog, metadata, schedule, and semantic-span JSONL one row at a time.
Stream all 32 token shards in assignment order and both sidecar sets in the
same assignment order. Recompute every per-record token digest, ordered token
digest, Merkle root, packed digest, binary semantics, dense-all-one rule,
Split90 closure from declared factual spans plus the route manifest, exact lane
counts, and zero padding. Select proof samples by sorting
`sha256(build_id + "\0" + lane_id + "\0" + record_id)` and taking the first 64
per solver-backed lane; replay every selected proof from stored premises.

`rebuild_commitments()` must not consume first-build task spools or payload
bytes. It independently reopens verified sources, rerenders every catalog
record with a different worker count and reverse bounded-chunk completion
order, reapplies the frozen schedule, and hashes token/Dense/Split90 streams
without publishing shards. This is the required second logical build rather
than a rehash of first-build artifacts.

`build_release_gate_evidence()` copies independently measured candidate fields,
compares `candidate.commitments` to `rebuilt`, and sets
`second_build_commitments_match` only on exact dataclass equality. It hashes the
rebuilt commitments into `second_build_commitments_sha256`; publication never
accepts caller-supplied booleans without these two canonical inputs.

At the start of `verify_parallel_corpus()`, detect the exact
`production_bindings` key. Dispatch that receipt to
`verify_inner_corpus()`; keep the old field-set path unchanged when the key is
absent. Return the same dictionary shape expected by `PackedShards`.
`PackedShards.from_parallel_corpus()` must parse
`ProductionShardAssignment` and require `shard-<index>.bin` paths when
`production_bindings` is present; it continues to parse existing
`ShardAssignment.shard_id` names for fixture-v2 receipts. In the production
branch it also opens and rehashes the exact `manifests/` and `proofs/`
artifacts declared by the receipt, but only token and selected sidecar
descriptors remain mapped for training.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_verification.py \
  tests/test_reasoning_v2_publication.py \
  tests/test_parallel_corpus.py \
  tests/test_sharded_loader.py
```

Expected: all tests pass, including production receipt loading by both arms.

- [ ] **Step 5: Commit**

```bash
git add corpusgen/reasoning_v2/verification.py \
  corpusgen/parallel/publication.py \
  corpusgen/parallel/__init__.py \
  train/data.py \
  tests/test_reasoning_v2_verification.py \
  tests/test_parallel_corpus.py \
  tests/test_sharded_loader.py
git commit -m "feat: independently verify v2 releases"
```

---

### Task 9: Add the production CLI and multiseed handoff gate

**Files:**
- Create: `corpusgen/reasoning_v2/handoff.py`
- Create: `scripts/build_reasoning_v2_corpus.py`
- Modify: `scripts/build_parallel_corpus.py`
- Create: `tests/test_reasoning_v2_cli.py`
- Create: `tests/test_reasoning_v2_handoff.py`

**Interfaces:**
- Consumes all public interfaces from Tasks 1–8.
- Produces:
  `CorpusConsumerRequirement`,
  `sha256_file(path: Path) -> str`,
  `verify_consumer_requirement(release_root: Path, requirement: CorpusConsumerRequirement) -> dict[str, object]`,
  `verify_360m_repository_handoff(release_root: Path, config_root: Path) -> dict[str, object]`,
  and CLI subcommands
  `freeze-sources`, `verify-source-lock`, `stage-sources`, `preflight`,
  `build-catalog`, `build-route`, `render-task`, `finalize-candidate`,
  `rebuild-commitments`, `verify-candidate`, `publish-release`,
  `verify-release`, `compare-builds`, and `verify-handoff`.

- [ ] **Step 1: RED — test strict CLI profiles and package binding**

```python
def test_miniature_cli_runs_all_eight_lanes_through_both_receipts(
    tmp_path,
    production_cli_fixture,
):
    result = run_production_cli_fixture(tmp_path, production_cli_fixture)
    assert result.returncode == 0, result.stderr
    outer = json.loads((result.release / "receipt.json").read_bytes())
    inner = json.loads(
        (result.release / "dataset/corpus-receipt.json").read_bytes()
    )
    assert outer["protected_launch_allowed"] is True
    assert outer["inner_receipt_sha256"] == sha256_file(
        result.release / "dataset/corpus-receipt.json"
    )
    assert inner["production_bindings"]["realized_lane_targets"] == (
        inner["production_bindings"]["lane_quotas"]
    )


def test_full_preflight_requires_200_gib_and_committed_source_lock(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(300 << 30, 150 << 30, 150 << 30),
    )
    result = run_cli("preflight", "--profile", "full", "--canonical-root", tmp_path)
    assert result.returncode != 0
    assert "200 GiB" in result.stderr
    assert "source lock" in result.stderr


def test_360m_configs_bind_inner_receipt_without_receipt_mutation(release_fixture):
    before = (release_fixture / "receipt.json").read_bytes()
    report = verify_360m_repository_handoff(
        release_fixture,
        Path(__file__).resolve().parents[1] / "configs/360m-v2",
    )
    assert report["consumer_cells"] == 10
    assert report["conditions"] == ["dense", "split90"]
    assert (release_fixture / "receipt.json").read_bytes() == before
```

Add tests that every mutating subcommand requires `--execute`, arbitrary
revision/token/URL flags are parser errors, production commands reject
`--allow-fewer-shards` under the full profile, a canary cannot publish as full,
a fixture receipt cannot satisfy `verify-handoff`, a 135M requirement can bind
the same build ID and sidecar names with a smaller update count, wrong
dataset/build/sidecar/token bounds fail, and the old fixture commands retain
their current arguments and output.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_cli.py \
  tests/test_reasoning_v2_handoff.py
```

Expected: script/module imports fail because the production entry points are
missing.

- [ ] **Step 3: GREEN — implement exact subcommands and requirements**

Use this handoff contract:

```python
@dataclass(frozen=True)
class CorpusConsumerRequirement:
    package_id: str
    dataset_id: str
    inner_receipt_path: str
    required_sidecars: tuple[str, ...]
    required_targets: int
    targets_per_update: int
    expected_build_id: str
    expected_outer_receipt_sha256: str
    expected_inner_receipt_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_consumer_requirement(
    release_root: Path,
    requirement: CorpusConsumerRequirement,
) -> dict[str, object]:
    outer = verify_published_release(release_root, site="farmshare")
    if outer.profile != "full" or outer.protected_launch_allowed is not True:
        raise ValueError("consumer requires a protected full corpus release")
    if outer.commitments.build_id != requirement.expected_build_id:
        raise ValueError("consumer build_id does not match the release")
    if outer.outer_receipt_sha256 != requirement.expected_outer_receipt_sha256:
        raise ValueError("consumer outer receipt SHA-256 does not match")
    if outer.inner_receipt_sha256 != requirement.expected_inner_receipt_sha256:
        raise ValueError("consumer inner receipt SHA-256 does not match")
    inner = verify_inner_corpus(
        release_root / requirement.inner_receipt_path,
        expected_build_id=outer.commitments.build_id,
    )
    if requirement.dataset_id != "memorysplit-v2-20x-reasoning-max-cohort":
        raise ValueError("consumer dataset_id does not match the release")
    if requirement.inner_receipt_path != "dataset/corpus-receipt.json":
        raise ValueError("consumer must bind the canonical inner receipt")
    if requirement.required_sidecars != (
        "dense_target_weights",
        "split90_target_weights",
    ):
        raise ValueError("consumer must require both canonical sidecars")
    if requirement.required_targets > inner.logical_tokens:
        raise ValueError("consumer target budget exceeds the corpus")
    if requirement.required_targets % requirement.targets_per_update:
        raise ValueError("consumer target budget is not update aligned")
    return {
        "package_id": requirement.package_id,
        "build_id": inner.build_id,
        "inner_receipt_sha256": sha256_file(
            release_root / requirement.inner_receipt_path
        ),
        "passed": True,
    }


def verify_360m_repository_handoff(
    release_root: Path,
    config_root: Path,
) -> dict[str, object]:
    expected_names = {
        f"{condition}-s{seed}.yaml"
        for condition in ("dense", "split90")
        for seed in range(5)
    }
    paths = tuple(sorted(config_root.glob("*.yaml")))
    if {path.name for path in paths} != expected_names:
        raise ValueError("360M consumer configs must contain exactly ten cells")
    conditions: set[str] = set()
    for path in paths:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if type(value) is not dict:
            raise ValueError(f"360M config must be an object: {path.name}")
        condition = value.get("condition")
        if condition not in {"dense", "split90"}:
            raise ValueError(f"360M condition is invalid: {path.name}")
        expected_sidecar = (
            "dense_target_weights"
            if condition == "dense"
            else "split90_target_weights"
        )
        expected = {
            "train_corpus": "dataset/corpus-receipt.json",
            "sidecar_name": expected_sidecar,
            "total_tokens": 7_120_879_616,
            "tokens_per_step": 524_288,
            "max_steps": 13_582,
        }
        if any(type(value.get(key)) is not type(item) or value.get(key) != item for key, item in expected.items()):
            raise ValueError(f"360M corpus binding is invalid: {path.name}")
        conditions.add(condition)
    site = verify_published_release(release_root, site="farmshare")
    requirement = CorpusConsumerRequirement(
        package_id="memorysplit-confirmatory-v2-360m-n5",
        dataset_id="memorysplit-v2-20x-reasoning-max-cohort",
        inner_receipt_path="dataset/corpus-receipt.json",
        required_sidecars=("dense_target_weights", "split90_target_weights"),
        required_targets=7_120_879_616,
        targets_per_update=524_288,
        expected_build_id=site.commitments.build_id,
        expected_outer_receipt_sha256=site.outer_receipt_sha256,
        expected_inner_receipt_sha256=site.inner_receipt_sha256,
    )
    verified = verify_consumer_requirement(release_root, requirement)
    return {
        **verified,
        "consumer_cells": len(paths),
        "conditions": sorted(conditions),
    }
```

The 360M repository handoff reads all ten `configs/360m-v2/*.yaml` files and
requires `train_corpus: dataset/corpus-receipt.json`, the exact arm sidecar,
`total_tokens: 7120879616`, `tokens_per_step: 524288`, and
`max_steps: 13582`. The generic 135M path takes a strict requirement JSON from
that package containing the expected build ID and both receipt SHA-256 values;
it may request fewer update-aligned targets but cannot request a different
dataset, build, token order, or sidecar.

The production CLI prints one canonical JSON object on stdout and diagnostics
on stderr. Mutating commands default to a dry plan and require `--execute`.
`preflight --profile full` verifies a committed lock, clean Git commit, source
tree, owner-controlled canonical root, no symlink components, no release
collision, and at least `214748364800` free bytes.

Keep `scripts/build_parallel_corpus.py` fixture subcommands unchanged. Its old
`build-production` name exits with code 64 and a deterministic message naming
`scripts/build_reasoning_v2_corpus.py`; no executable path instantiates
`UnsupportedProductionRenderer`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_reasoning_v2_cli.py \
  tests/test_reasoning_v2_handoff.py \
  tests/test_parallel_corpus.py \
  tests/test_cohort_assignment_v2.py
```

Expected: all tests pass and all ten existing 360M configurations bind the
canonical inner receipt.

- [ ] **Step 5: Commit**

```bash
git add corpusgen/reasoning_v2/handoff.py \
  scripts/build_reasoning_v2_corpus.py \
  scripts/build_parallel_corpus.py \
  tests/test_reasoning_v2_cli.py \
  tests/test_reasoning_v2_handoff.py
git commit -m "feat: add v2 corpus production cli"
```

---

### Task 10: Replace fixture Slurm templates with the FarmShare DAG

**Files:**
- Create: `cluster/slurm/v2_corpus_source_freeze.sbatch`
- Create: `cluster/slurm/v2_corpus_source_stage.sbatch`
- Modify: `cluster/slurm/v2_corpus_build.sbatch`
- Create: `cluster/slurm/v2_corpus_finalize.sbatch`
- Modify: `cluster/slurm/v2_corpus_verify.sbatch`
- Create: `cluster/submit_v2_corpus.sh`
- Create: `tests/test_reasoning_v2_slurm.py`

**Interfaces:**
- Consumes: production CLI, canonical root, exact repository/Python paths,
  source-lock SHA-256, profile, run nonce, and Slurm IDs.
- Produces: 32 task receipts, one candidate, one rebuild commitment receipt,
  one FarmShare site receipt, and one no-replace release.

- [ ] **Step 1: RED — assert exact resources, environment, and dependencies**

```python
def test_build_array_has_exact_farmshare_resources(repo_root):
    text = (repo_root / "cluster/slurm/v2_corpus_build.sbatch").read_text()
    assert "#SBATCH --partition=normal" in text
    assert "#SBATCH --array=0-31%8" in text
    assert "#SBATCH --cpus-per-task=16" in text
    assert "#SBATCH --mem=64G" in text
    assert "#SBATCH --time=24:00:00" in text
    assert text.count("#SBATCH --export=NONE") == 1
    assert "SLURM_TMPDIR" in text
    assert "render-task" in text
    assert "render-fixture-task" not in text
    assert "env -i" in text


@pytest.mark.parametrize(
    "name",
    ["v2_corpus_source_freeze.sbatch", "v2_corpus_source_stage.sbatch"],
)
def test_source_jobs_use_normal_partition_and_clean_environment(repo_root, name):
    text = (repo_root / "cluster/slurm" / name).read_text()
    assert "#SBATCH --partition=normal" in text
    assert "#SBATCH --cpus-per-task=16" in text
    assert "#SBATCH --mem=64G" in text
    assert "#SBATCH --time=24:00:00" in text
    assert text.count("#SBATCH --export=NONE") == 1
    assert "env -i" in text


def test_submitter_wires_array_finalize_and_verify_afterok(fake_sbatch, repo_root):
    result = subprocess.run(
        [
            "bash",
            "cluster/submit_v2_corpus.sh",
            "--profile",
            "canary",
            "--nonce",
            "canary-a",
            "--execute",
        ],
        cwd=repo_root,
        env=fake_sbatch.environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = fake_sbatch.calls()
    assert len(calls) == 3
    assert "v2_corpus_build.sbatch" in calls[0]
    assert "--dependency=afterok:1001" in calls[1]
    assert "v2_corpus_finalize.sbatch" in calls[1]
    assert "--dependency=afterok:1002" in calls[2]
    assert "v2_corpus_verify.sbatch" in calls[2]
```

Also test dry-run default, exact environment allowlist, no `--export=ALL`,
missing `SLURM_TMPDIR`, task count other than 32, full profile without source
lock, finalizer missing one receipt, verifier missing rebuild evidence, nonzero
job status propagation, shell syntax, and absence of `rm -rf`, path-based
`mv`, and unverified `cp`.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_slurm.py
```

Expected: assertions fail because the current templates still run fixture
commands and request smaller resources.

- [ ] **Step 3: GREEN — implement the three-job production DAG**

The source-freeze job invokes `freeze-sources` with
`$MS_CANONICAL_ROOT/work/source-freeze/downloads` and writes
`$MS_CANONICAL_ROOT/work/source-freeze/reasoning-dataset-v2-source-lock.json`.
It then runs full-byte `verify-source-lock` and writes
`source-freeze-result.json`. The source-stage job invokes `stage-sources` using
the committed `$MS_SOURCE_LOCK`, runs `verify-source-tree`, and writes
`source-stage-result.json`. Both use the same explicit environment boundary and
resources asserted above; neither runs on a login node.

The array invokes:

```bash
env -i \
  HOME="${HOME:-/tmp}" \
  LANG=C.UTF-8 \
  PATH=/usr/bin:/bin \
  PYTHONPATH="$MS_REPO" \
  SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
  "$MS_PYTHON" -u "$MS_REPO/scripts/build_reasoning_v2_corpus.py" \
  render-task \
  --profile "$MS_PROFILE" \
  --canonical-root "$MS_CANONICAL_ROOT" \
  --source-lock "$MS_SOURCE_LOCK" \
  --nonce "$MS_RUN_NONCE" \
  --scheduler-id "$SLURM_ARRAY_JOB_ID" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --task-count "$SLURM_ARRAY_TASK_COUNT" \
  --local-root "$SLURM_TMPDIR" \
  --workers "$SLURM_CPUS_PER_TASK" \
  --execute
```

The finalizer requires exactly 32 verified task receipts and invokes
`finalize-candidate`; it writes
`work/<nonce>/candidate-result.json` with the candidate path and build ID. The
verifier reads that canonical result, runs `verify-candidate`, runs
`rebuild-commitments` against a separately named rebuild nonce by reopening
sources and rerendering with 16 workers, and compares commitments without
reading first-build task payloads. For `canary` it writes a non-launchable outer
receipt plus `candidate-verification.json` under `work/`. For `full` it invokes
`publish-release` and writes the FarmShare site receipt at
`verification/<build-id>.json`. Both the finalizer and verifier request
partition `normal`, 16 CPUs, 64G, 24 hours, and `--export=NONE`.

`cluster/submit_v2_corpus.sh` accepts only `--profile canary|full`,
`--nonce [A-Za-z0-9._-]+`, and optional `--execute`. It fixes:

```bash
MS_REPO=/scratch/users/syz/memorysplit
MS_PYTHON=/scratch/users/syz/venvs/memorysplit/bin/python
MS_CANONICAL_ROOT=/scratch/users/syz/memorysplit-v2-corpus
MS_SOURCE_LOCK=/scratch/users/syz/memorysplit/configs/reasoning-dataset-v2-source-lock.json
```

It submits with `sbatch --parsable --export=MS_REPO=...,MS_PYTHON=...` so the
template's `--export=NONE` remains the inheritance boundary. Parse each numeric
job ID and use exact `afterok` dependencies.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
bash -n cluster/slurm/v2_corpus_source_freeze.sbatch
bash -n cluster/slurm/v2_corpus_source_stage.sbatch
bash -n cluster/slurm/v2_corpus_build.sbatch
bash -n cluster/slurm/v2_corpus_finalize.sbatch
bash -n cluster/slurm/v2_corpus_verify.sbatch
bash -n cluster/submit_v2_corpus.sh
python -m pytest -q \
  tests/test_reasoning_v2_slurm.py \
  tests/test_parallel_corpus.py
```

Expected: all syntax checks and tests pass.

- [ ] **Step 5: Commit**

```bash
git add cluster/slurm/v2_corpus_build.sbatch \
  cluster/slurm/v2_corpus_source_freeze.sbatch \
  cluster/slurm/v2_corpus_source_stage.sbatch \
  cluster/slurm/v2_corpus_finalize.sbatch \
  cluster/slurm/v2_corpus_verify.sbatch \
  cluster/submit_v2_corpus.sh \
  tests/test_reasoning_v2_slurm.py
git commit -m "feat: add FarmShare v2 corpus dag"
```

---

### Task 11: Implement atomic MIT mirror and second-site verification

**Files:**
- Create: `corpusgen/reasoning_v2/mirror.py`
- Create: `cluster/mirror_v2_to_mit.sh`
- Create: `tests/test_reasoning_v2_mirror.py`

**Interfaces:**
- Consumes:
  verified FarmShare release,
  FarmShare `SiteVerificationReceipt`,
  operator-supplied absolute local relay root,
  operator-supplied MIT SSH host,
  absolute MIT repository/Python paths, and absolute MIT storage root.
- Produces:
  `MirrorPlan`,
  `stage_mirror(source: Path, staging: Path, expected: SiteVerificationReceipt) -> None`,
  `publish_mirror(staging: Path, releases_root: Path, build_id: str) -> Path`, and
  MIT `SiteVerificationReceipt`.

- [ ] **Step 1: RED — test mirror identity, collisions, and tampering**

```python
def test_mit_mirror_is_byte_identical_and_has_second_site_receipt(
    tmp_path,
    farmshare_release,
    farmshare_verification,
):
    staging = tmp_path / "mit/.incoming-build-a"
    stage_mirror(farmshare_release, staging, farmshare_verification)
    published = publish_mirror(
        staging,
        tmp_path / "mit/releases",
        farmshare_verification.commitments.build_id,
    )
    mit = verify_published_release(published, site="mit")
    assert mit.commitments == farmshare_verification.commitments
    assert mit.inner_receipt_sha256 == farmshare_verification.inner_receipt_sha256
    assert mit.site == "mit"


def test_mirror_never_replaces_existing_release(
    tmp_path,
    farmshare_release,
    farmshare_verification,
):
    destination = (
        tmp_path
        / "mit/releases"
        / farmshare_verification.commitments.build_id
    )
    destination.mkdir(parents=True)
    (destination / "foreign").write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        publish_staged_fixture_mirror(
            farmshare_release,
            farmshare_verification,
            destination,
        )
    assert (destination / "foreign").read_bytes() == b"keep"
```

Add partial rsync, source symlink, destination symlink, hardlink, extra file,
changed inner bytes with rewritten outer hash, wrong build ID, wrong FarmShare
verification hash, interrupted staging, stale staging owner, command injection
in host/relay/repository/Python/storage arguments, wrong MIT source commit, and
repeat-idempotence tests.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_mirror.py
```

Expected: import fails because `corpusgen.reasoning_v2.mirror` does not exist.

- [ ] **Step 3: GREEN — implement private staging and DTN driver**

Use:

```python
@dataclass(frozen=True)
class MirrorPlan:
    build_id: str
    source_release: str
    relay_release: str
    staging_release: str
    final_release: str
    expected_farmshare_verification_sha256: str
    file_count: int
    byte_count: int
```

`stage_mirror()` creates an empty owner-marked private directory, copies only
the exact receipted namespace with `O_NOFOLLOW` source and exclusive destination
files, fsyncs, removes the owner marker only after verification, and requires a
MIT verification receipt whose inner receipt bytes, ordered-stream commitment,
Dense/Split90 commitments, source-lock hash, and build ID equal FarmShare.
`publish_mirror()` uses the same no-replace rename primitive as FarmShare.

The shell driver accepts:

```text
--source-release ABSOLUTE_FARMSHARE_PATH
--farmshare-verification ABSOLUTE_FARMSHARE_RECEIPT
--relay-root ABSOLUTE_LOCAL_RELAY_ROOT
--mit-host HOSTNAME
--mit-repo ABSOLUTE_MIT_REPOSITORY
--mit-python ABSOLUTE_MIT_PYTHON
--mit-root ABSOLUTE_MIT_ROOT
--build-id LOWERCASE_SHA256
--execute
```

It defaults to a quoted dry-run, pulls the exact release from the authenticated
FarmShare DTN into
`<relay-root>/.farmshare-<build-id>-<nonce>/`, verifies the relay, then pushes
that verified tree into `<mit-root>/.incoming-<build-id>-<nonce>/`. It runs the
production verifier with `--mit-python` from an `--mit-repo` checkout whose
clean Git HEAD equals the outer receipt's source commit, and only then runs
`publish_mirror`. It requires relay free space greater than the receipted byte
count and never accepts a partial or site-repacked corpus. It retains the relay
until MIT verification succeeds; cleanup is a separate operator maintenance
action.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
bash -n cluster/mirror_v2_to_mit.sh
python -m pytest -q tests/test_reasoning_v2_mirror.py
```

Expected: all tests and shell syntax checks pass.

- [ ] **Step 5: Commit**

```bash
git add corpusgen/reasoning_v2/mirror.py \
  cluster/mirror_v2_to_mit.sh \
  tests/test_reasoning_v2_mirror.py
git commit -m "feat: add verified MIT corpus mirror"
```

---

### Task 12: Add the operator runbook and regression release gate

**Files:**
- Create: `cluster/FARMSHARE-V2-CORPUS-RUNBOOK.md`
- Create: `tests/test_reasoning_v2_regressions.py`

**Interfaces:**
- Consumes every local command and artifact contract from Tasks 1–11.
- Produces an executable runbook and a final local regression gate; it changes
  no production interface.

- [ ] **Step 1: RED — test runbook completeness and frozen compatibility**

```python
def test_runbook_contains_every_external_terminal_gate(repo_root):
    text = (repo_root / "cluster/FARMSHARE-V2-CORPUS-RUNBOOK.md").read_text()
    for required in (
        "cluster/connect.sh syz rice-04.farmshare.stanford.edu",
        "cluster/connect.sh syz dtn.farmshare.stanford.edu",
        "freeze-sources",
        "verify-source-lock",
        "1,048,576",
        "canary-a",
        "canary-b",
        "7,120,879,616",
        "200 GiB",
        "rebuild-commitments",
        "verification/<build-id>.json",
        "mirror_v2_to_mit.sh",
        "BLOCKED_EXTERNAL_CORPUS",
    ):
        assert required in text


def test_existing_release_contracts_remain_exact(repo_root):
    for path in sorted((repo_root / "configs/360m-v2").glob("*.yaml")):
        value = yaml.safe_load(path.read_text())
        assert value["train_corpus"] == "dataset/corpus-receipt.json"
        assert value["total_tokens"] == 7_120_879_616
        assert value["tokens_per_step"] == 524_288
        assert value["max_steps"] == 13_582
    assert json.loads((repo_root / "DATASET-POINTER-AWS.json").read_bytes())[
        "required_receipt"
    ] == "dataset/receipt.json"
```

Add regression tests that build and verify the old parallel fixture, verify
`fixtures/current-smoke`, run the reasoning-v2 smoke fixture, check tokenizer
asset hashes, prove direct `UnsupportedProductionRenderer` remains fail-closed,
and prove the production CLI never imports that sentinel.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tests/test_reasoning_v2_regressions.py
```

Expected: failure because the runbook does not exist.

- [ ] **Step 3: GREEN — write the exact operator runbook**

The runbook must contain the literal commands from Part II, a failure table
mapping every fail-closed condition to `BLOCKED`, the canonical path tree, job
ID/evidence recording instructions, measured canary throughput/ETA formula,
the prohibition on same-day promises before measurement, retry rules for
verified content-addressed artifacts, and the final handoff receipt fields.
It must state that passing local tests, merging code, or finishing the
FarmShare array is not completion.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest -q
python -m compileall -q corpusgen scripts
git diff --exit-code -- \
  configs/360m-v2 \
  configs/cohort-assignment-v2.json \
  DATASET-POINTER.json \
  DATASET-POINTER-AWS.json \
  fixtures/current-smoke \
  train/tokenizer.py
```

Expected: the complete local suite passes, compileall exits zero, and the
frozen compatibility paths have no diff.

- [ ] **Step 5: Commit**

```bash
git add cluster/FARMSHARE-V2-CORPUS-RUNBOOK.md \
  tests/test_reasoning_v2_regressions.py
git commit -m "docs: add v2 corpus production runbook"
```

---

## Part II — External FarmShare and MIT Execution Tasks

These are acceptance operations, not local implementation tasks. Do not weaken
a gate to make an operation pass. Record every command, job ID, exit status,
receipt path, and SHA-256 in the producer log.

### Task 13: Resolve, verify, and commit the real source lock

**Files created by the operation:**
- FarmShare candidate:
  `/scratch/users/syz/memorysplit-v2-corpus/work/source-freeze/reasoning-dataset-v2-source-lock.json`
- Repository:
  `configs/reasoning-dataset-v2-source-lock.json`
- FarmShare staged source root:
  `/scratch/users/syz/memorysplit-v2-corpus/sources/<source-lock-sha256>/`

- [ ] Establish both Duo-authenticated control sockets locally:

```bash
bash cluster/connect.sh syz rice-04.farmshare.stanford.edu
bash cluster/connect.sh syz dtn.farmshare.stanford.edu
```

Expected: both commands print `FarmShare session live`.

- [ ] Transfer the exact clean implementation commit as a Git bundle and verify
  it remotely:

```bash
test -z "$(git status --porcelain)"
git merge-base --is-ancestor 33b8c9f HEAD
IMPLEMENTATION_COMMIT=$(git rev-parse HEAD)
SOURCE_BUNDLE="${TMPDIR:-/tmp}/memorysplit-$IMPLEMENTATION_COMMIT.bundle"
git bundle create "$SOURCE_BUNDLE" HEAD
ssh -o ControlPath="$HOME/.ssh/cm-%r@%h:%p" \
  syz@rice-04.farmshare.stanford.edu \
  'mkdir -p /scratch/users/syz/memorysplit-v2-corpus/work/source-sync'
rsync -a \
  -e "ssh -o ControlPath=$HOME/.ssh/cm-%r@%h:%p" \
  "$SOURCE_BUNDLE" \
  syz@dtn.farmshare.stanford.edu:/scratch/users/syz/memorysplit-v2-corpus/work/source-sync/implementation.bundle
ssh -o ControlPath="$HOME/.ssh/cm-%r@%h:%p" \
  syz@rice-04.farmshare.stanford.edu "
    set -euo pipefail
    cd /scratch/users/syz/memorysplit
    test -z \"\$(git status --porcelain)\"
    git fetch /scratch/users/syz/memorysplit-v2-corpus/work/source-sync/implementation.bundle HEAD
    git checkout --detach FETCH_HEAD
    test \"\$(git rev-parse HEAD)\" = '$IMPLEMENTATION_COMMIT'
    test -z \"\$(git status --porcelain)\"
  "
```

Expected: every command exits zero.

- [ ] Resolve and download the source candidate on FarmShare:

```bash
FREEZE_JOB=$(
  ssh -o ControlPath="$HOME/.ssh/cm-%r@%h:%p" \
    syz@rice-04.farmshare.stanford.edu '
  set -euo pipefail
  cd /scratch/users/syz/memorysplit
  sbatch --wait --parsable \
    --export=MS_REPO=/scratch/users/syz/memorysplit,MS_PYTHON=/scratch/users/syz/venvs/memorysplit/bin/python,MS_CANONICAL_ROOT=/scratch/users/syz/memorysplit-v2-corpus \
    cluster/slurm/v2_corpus_source_freeze.sbatch
  '
)
printf 'source-freeze job: %s\n' "$FREEZE_JOB"
ssh -o ControlPath="$HOME/.ssh/cm-%r@%h:%p" \
  syz@rice-04.farmshare.stanford.edu '
    /scratch/users/syz/venvs/memorysplit/bin/python -c "
import json
from pathlib import Path
path = Path(\"/scratch/users/syz/memorysplit-v2-corpus/work/source-freeze/source-freeze-result.json\")
value = json.loads(path.read_bytes())
assert value[\"passed\"] is True
"
  '
```

Expected: one canonical JSON result with `passed:true`. It must name immutable
FineMath, CLRS, RuleTaker, ProntoQA, Reasoning Gym, and DeepMind mathematics
revisions obtained by the command. If any source, file, or license is missing,
stop and report `BLOCKED_SOURCE_FREEZE`.

- [ ] Pull the generated lock, verify it locally, and commit its actual bytes:

```bash
rsync -a \
  -e "ssh -o ControlPath=$HOME/.ssh/cm-%r@%h:%p" \
  syz@dtn.farmshare.stanford.edu:/scratch/users/syz/memorysplit-v2-corpus/work/source-freeze/reasoning-dataset-v2-source-lock.json \
  configs/reasoning-dataset-v2-source-lock.json
python scripts/build_reasoning_v2_corpus.py verify-source-lock \
  --source-lock configs/reasoning-dataset-v2-source-lock.json \
  --schema-only
git add configs/reasoning-dataset-v2-source-lock.json
git commit -m "data: freeze v2 corpus sources"
SOURCE_LOCK_COMMIT=$(git rev-parse HEAD)
SOURCE_LOCK_SHA256=$(shasum -a 256 configs/reasoning-dataset-v2-source-lock.json | awk '{print $1}')
```

The local schema verifier is sufficient here because the full byte verifier
already passed on FarmShare. Expected: a new clean commit and a 64-character
`SOURCE_LOCK_SHA256`. Do not edit resolved revisions by hand.

- [ ] Transfer the source-lock commit as a second Git bundle, verify the remote
  commit, and stage sources:

```bash
SOURCE_LOCK_BUNDLE="${TMPDIR:-/tmp}/memorysplit-$SOURCE_LOCK_COMMIT.bundle"
git bundle create "$SOURCE_LOCK_BUNDLE" HEAD
rsync -a \
  -e "ssh -o ControlPath=$HOME/.ssh/cm-%r@%h:%p" \
  "$SOURCE_LOCK_BUNDLE" \
  syz@dtn.farmshare.stanford.edu:/scratch/users/syz/memorysplit-v2-corpus/work/source-sync/source-lock.bundle
ssh -o ControlPath="$HOME/.ssh/cm-%r@%h:%p" \
  syz@rice-04.farmshare.stanford.edu "
  set -euo pipefail
  cd /scratch/users/syz/memorysplit
  test -z \"\$(git status --porcelain)\"
  git fetch /scratch/users/syz/memorysplit-v2-corpus/work/source-sync/source-lock.bundle HEAD
  git checkout --detach FETCH_HEAD
  test \"\$(git rev-parse HEAD)\" = '$SOURCE_LOCK_COMMIT'
  test -z \"\$(git status --porcelain)\"
  sbatch --wait --parsable \
    --export=MS_REPO=/scratch/users/syz/memorysplit,MS_PYTHON=/scratch/users/syz/venvs/memorysplit/bin/python,MS_CANONICAL_ROOT=/scratch/users/syz/memorysplit-v2-corpus,MS_SOURCE_LOCK=/scratch/users/syz/memorysplit/configs/reasoning-dataset-v2-source-lock.json \
    cluster/slurm/v2_corpus_source_stage.sbatch
"
ssh -o ControlPath="$HOME/.ssh/cm-%r@%h:%p" \
  syz@rice-04.farmshare.stanford.edu '
    /scratch/users/syz/venvs/memorysplit/bin/python -c "
import json
from pathlib import Path
path = Path(\"/scratch/users/syz/memorysplit-v2-corpus/work/source-freeze/source-stage-result.json\")
value = json.loads(path.read_bytes())
assert value[\"passed\"] is True
"
  '
```

Expected source root:
`/scratch/users/syz/memorysplit-v2-corpus/sources/$SOURCE_LOCK_SHA256`.

---

### Task 14: Run FarmShare source and capacity preflight

- [ ] Verify staged bytes, root ownership, source commit, and capacity:

```bash
ssh syz@rice-04.farmshare.stanford.edu '
  set -euo pipefail
  cd /scratch/users/syz/memorysplit
  /scratch/users/syz/venvs/memorysplit/bin/python \
    scripts/build_reasoning_v2_corpus.py preflight \
    --profile full \
    --canonical-root /scratch/users/syz/memorysplit-v2-corpus \
    --source-lock configs/reasoning-dataset-v2-source-lock.json
'
```

Expected JSON fields:
`source_lock_verified=true`, `source_tree_verified=true`,
`git_clean=true`, `free_bytes>=214748364800`, and `passed=true`.
Otherwise stop with `BLOCKED_PREFLIGHT`; do not submit Slurm.

---

### Task 15: Run and compare two all-lane canaries

- [ ] Submit canary A and record all three job IDs:

```bash
ssh syz@rice-04.farmshare.stanford.edu '
  cd /scratch/users/syz/memorysplit
  bash cluster/submit_v2_corpus.sh \
    --profile canary \
    --nonce canary-a \
    --execute
'
```

- [ ] After canary A's verification job passes, submit canary B:

```bash
ssh syz@rice-04.farmshare.stanford.edu '
  cd /scratch/users/syz/memorysplit
  bash cluster/submit_v2_corpus.sh \
    --profile canary \
    --nonce canary-b \
    --execute
'
```

- [ ] Compare independent canary commitments:

```bash
ssh syz@rice-04.farmshare.stanford.edu '
  cd /scratch/users/syz/memorysplit
  /scratch/users/syz/venvs/memorysplit/bin/python \
    scripts/build_reasoning_v2_corpus.py compare-builds \
    --left /scratch/users/syz/memorysplit-v2-corpus/work/canary-a/candidate-result.json \
    --right /scratch/users/syz/memorysplit-v2-corpus/work/canary-b/candidate-result.json
'
```

Expected: both have exactly `1,048,576` targets, all eight realized quotas,
both sidecars, both receipt layers, zero leaks, passing proof samples, and
identical catalog, schedule, ordered token, Merkle, Dense, Split90, source-lock,
and build-ID commitments. Both outer receipts must have
`profile:"canary"` and `protected_launch_allowed:false`, and neither candidate
may appear under `releases/`. A mismatch is `BLOCKED_CANARY_IDENTITY`.

- [ ] Record measured aggregate targets/second and calculate:

```text
estimated_full_hours =
  7,120,879,616 / measured_canary_targets_per_second / 3600
```

Publish that measurement and estimate before requesting the full build. Do not
claim same-day completion without this evidence.

---

### Task 16: Produce the first full candidate

- [ ] Re-run the full preflight immediately before submission.

- [ ] Submit the full array/finalizer/verifier DAG:

```bash
ssh syz@rice-04.farmshare.stanford.edu '
  cd /scratch/users/syz/memorysplit
  bash cluster/submit_v2_corpus.sh \
    --profile full \
    --nonce full-a \
    --execute
'
```

Expected array: 32 exact task receipts from `0-31%8`; no task receipt may be
missing or replay-disagreeing. Expected finalizer: an unpublished candidate
under `work/full-a/` with exactly 32 token shards and 64 sidecar shards.
Array completion alone is not acceptance.

---

### Task 17: Complete independent rebuild, verification, and FarmShare publication

- [ ] Confirm the verifier job performs a second logical full build under a
  distinct nonce and compares, at minimum:

```text
catalog_sha256
metadata_sha256
schedule_sha256
assignments_sha256
ordered_stream_sha256
merkle_root_sha256
packed_stream_sha256
dense_stream_sha256
split90_stream_sha256
source_lock_sha256
route_manifest_sha256
proof_manifest_sha256
build_id
```

- [ ] Verify the published FarmShare release directly:

```bash
ssh syz@rice-04.farmshare.stanford.edu '
  set -euo pipefail
  cd /scratch/users/syz/memorysplit
  BUILD_ID=$(
    /scratch/users/syz/venvs/memorysplit/bin/python -c "
import json
from pathlib import Path
value = json.loads(Path(
    \"/scratch/users/syz/memorysplit-v2-corpus/work/full-a/candidate-result.json\"
).read_bytes())
print(value[\"build_id\"])
"
  )
  /scratch/users/syz/venvs/memorysplit/bin/python \
    scripts/build_reasoning_v2_corpus.py verify-release \
    --release "/scratch/users/syz/memorysplit-v2-corpus/releases/$BUILD_ID" \
    --site farmshare \
    --verification-receipt \
      "/scratch/users/syz/memorysplit-v2-corpus/verification/$BUILD_ID.json"
'
```

Expected:

```text
/scratch/users/syz/memorysplit-v2-corpus/releases/<build-id>/receipt.json
/scratch/users/syz/memorysplit-v2-corpus/releases/<build-id>/source-lock.json
/scratch/users/syz/memorysplit-v2-corpus/releases/<build-id>/dataset/corpus-receipt.json
/scratch/users/syz/memorysplit-v2-corpus/verification/<build-id>.json
```

The outer receipt must say `protected_launch_allowed:true`; both lane maps must
equal the frozen quotas; semantic rate must be `1.0`; overlap must be `0.0`;
graph coverage must be complete; padding must be zero. Any failure is
`BLOCKED_FARMSHARE_VERIFICATION`, and the candidate must remain outside
`releases/`.

---

### Task 18: Mirror atomically to MIT and verify again

- [ ] Establish the operator-provided MIT login, MIT storage root, clean MIT
  repository checkout, MIT Python path, and absolute local relay root with free
  bytes exceeding the FarmShare release byte count. Record them in the private
  operator log, not in repository files. Transfer the final Git bundle from
  Task 13 to the MIT checkout and require its clean HEAD to equal
  `SOURCE_LOCK_COMMIT`.

- [ ] Re-establish the FarmShare DTN socket immediately before transfer:

```bash
bash cluster/connect.sh syz dtn.farmshare.stanford.edu
```

- [ ] Dry-run, inspect, then execute the mirror:

```bash
bash cluster/mirror_v2_to_mit.sh \
  --source-release "/scratch/users/syz/memorysplit-v2-corpus/releases/$BUILD_ID" \
  --farmshare-verification "/scratch/users/syz/memorysplit-v2-corpus/verification/$BUILD_ID.json" \
  --relay-root "$LOCAL_RELAY_ROOT" \
  --mit-host "$MIT_HOST" \
  --mit-repo "$MIT_REPO" \
  --mit-python "$MIT_PYTHON" \
  --mit-root "$MIT_ROOT" \
  --build-id "$BUILD_ID"

bash cluster/mirror_v2_to_mit.sh \
  --source-release "/scratch/users/syz/memorysplit-v2-corpus/releases/$BUILD_ID" \
  --farmshare-verification "/scratch/users/syz/memorysplit-v2-corpus/verification/$BUILD_ID.json" \
  --relay-root "$LOCAL_RELAY_ROOT" \
  --mit-host "$MIT_HOST" \
  --mit-repo "$MIT_REPO" \
  --mit-python "$MIT_PYTHON" \
  --mit-root "$MIT_ROOT" \
  --build-id "$BUILD_ID" \
  --execute
```

Expected MIT paths:

```text
<MIT_ROOT>/releases/<build-id>/receipt.json
<MIT_ROOT>/releases/<build-id>/dataset/corpus-receipt.json
<MIT_ROOT>/verification/<build-id>.json
```

The MIT verification receipt must match FarmShare on inner receipt bytes,
ordered-stream commitment, Merkle root, Dense commitment, Split90
commitment, source-lock hash, and build ID. A partial copy or mismatch is
`BLOCKED_MIT_MIRROR`.

---

### Task 19: Hand the immutable receipt to both multiseed packages

- [ ] Run the repository 360M handoff gate against FarmShare:

```bash
python scripts/build_reasoning_v2_corpus.py verify-handoff \
  --release "/scratch/users/syz/memorysplit-v2-corpus/releases/$BUILD_ID" \
  --consumer-config-dir configs/360m-v2
```

Expected: ten cells, five Dense/Split90 pairs, exact inner receipt path, exact
sidecar names, and `passed:true`.

- [ ] Give each 135M package the unchanged outer receipt path, outer receipt
  SHA-256, inner receipt path, inner receipt SHA-256, build ID, FarmShare
  verification receipt path/SHA-256, and MIT verification receipt path/SHA-256.
  Run that package's strict requirement JSON through `verify-handoff`; it may
  consume an update-aligned prefix but must bind the same build and receipt
  bytes.

- [ ] Clear `BLOCKED_EXTERNAL_CORPUS` only after both package gates return
  `passed:true`. Do not edit, rewrite, or repack either receipt for a consumer.

---

## Final Terminal Checklist

The producer may report completion only when every item is evidenced:

- [ ] `configs/reasoning-dataset-v2-source-lock.json` contains real resolved
  immutable identities and is committed.
- [ ] FarmShare staged source bytes and licenses verify against that lock.
- [ ] Two independent `1,048,576`-target canaries are commitment-identical.
- [ ] The full candidate contains exactly `7,120,879,616` logical targets,
  exact eight-lane quotas, 32 token shards, both 32-shard sidecars, and zero
  padding.
- [ ] A second full logical build matches every listed stream and provenance
  commitment.
- [ ] The immutable FarmShare release exists at
  `/scratch/users/syz/memorysplit-v2-corpus/releases/<build-id>/`.
- [ ] The FarmShare verification receipt exists at
  `/scratch/users/syz/memorysplit-v2-corpus/verification/<build-id>.json`.
- [ ] The immutable MIT release and MIT verification receipt exist and match
  FarmShare.
- [ ] The 360M five-seed package and every protected 135M multiseed package
  bind the same unchanged outer/inner receipts.

Passing code review, local tests, Slurm array completion, candidate creation, or
FarmShare-only verification does not satisfy this terminal checklist.

---

## Plan Self-Review

- **Spec coverage:** Tasks 1–3 cover geometry and source freeze; Tasks 3–5
  cover all eight lanes, proof-bearing rendering, and semantic sidecars; Tasks
  6–8 cover two-pass parallelism, receipts, publication, and independent
  verification; Tasks 9–12 cover package handoff, CLI, Slurm, MIT tooling, and
  regressions; Tasks 13–19 cover the actual source lock, canary, full build,
  deterministic rebuild, FarmShare publication, MIT mirror, and multiseed
  unblock terminal condition.
- **Unresolved-value review:** The plan intentionally contains no guessed
  FineMath, CLRS, RuleTaker, ProntoQA, Reasoning Gym, or DeepMind mathematics
  revision. Task 13 obtains those values from the implemented resolver,
  verifies downloaded bytes, and commits the generated lock before production.
  Operator-provided MIT host/root values remain runtime arguments and never
  enter a scientific receipt.
- **Type/interface consistency:** `ReasoningV2Recipe -> BuildGeometry ->
  InputCatalog -> RouteArtifacts/RouteIndex -> ProductionRenderedRecord ->
  TaskReceipt/CandidateDataset -> InnerCorpusReceipt ->
  CandidateVerification + StreamCommitments -> ReleaseGateEvidence ->
  OuterMaterializationReceipt -> SiteVerificationReceipt`
  is the single data flow used by all later tasks. Receipt path names, sidecar
  names, lane IDs, build ID, source-lock hash, and stream commitment field names
  are consistent across publication, verification, mirror, and handoff.
- **Compatibility review:** The dedicated production path never uses the
  fixture renderer. Generic fixture behavior remains on the old API; production
  verifier dispatch occurs only when `production_bindings` is present. The
  final local gate checks the frozen tokenizer, fixture corpus, 360M configs,
  cohort assignment, and dataset pointers for zero diff.
