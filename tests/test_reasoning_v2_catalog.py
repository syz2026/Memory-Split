from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import threading
import tracemalloc
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from corpusgen.parallel.canonical import canonical_json_bytes, sha256_hex
from corpusgen.reasoning_v2 import catalog as catalog_module
from corpusgen.reasoning_v2.catalog import (
    CatalogDraft,
    CatalogRecord,
    InputCatalog,
    LaneQuotaShortfall,
    SemanticFactRow,
    build_input_catalog,
    catalog_record_id,
)
from corpusgen.reasoning_v2.contracts import (
    LANE_ORDER,
    BuildGeometry,
    LaneId,
    balanced_record_lengths,
)
from corpusgen.reasoning_v2.source_lock import (
    SourceFile,
    SourceLock,
    load_source_lock,
    stage_source_lock,
)
from corpusgen.reasoning_v2.wikidata_source import (
    build_wikidata_derived_view,
    open_wikidata_derived_view,
)
from reasoning_v2_fixtures import (
    FixtureSourceLock,
    fake_public_resolver,
    fixed_contract_environment,
    fixture_source_lock,
    full_recipe,
)


_LANE_SOURCE_IDS: dict[LaneId, str] = {
    "fineweb_edu": "fineweb_edu",
    "finemath": "finemath",
    "wikidata_graph": "wikidata5m",
    "synthetic_graph": "deepmind_mathematics_generator",
    "verified_synthetic_multihop": "reasoning_gym_exact_answer",
    "wikidata_path_reasoning": "wikidata5m",
    "relational_refinement": "clrs_text",
    "objective_auxiliary": "prontoqa",
}
_GENERATED_LANES = {
    "synthetic_graph",
    "verified_synthetic_multihop",
}
_QUARANTINE_DIRECTORY = ".memorysplit-catalog-quarantine-v1"


def _source_file(lock: SourceLock, source_id: str) -> tuple[str, SourceFile]:
    entry = next(row for row in lock.sources if row.source_id == source_id)
    row = next(item for item in entry.files if item.path not in entry.license_files)
    return entry.materialized_path, row


def _fact(lane_id: LaneId, index: int) -> SemanticFactRow:
    fact_index = index % 2 if lane_id == "wikidata_graph" else index
    return SemanticFactRow(
        fact_id=f"fact:{lane_id}:{fact_index:06d}",
        source=_LANE_SOURCE_IDS[lane_id],
        record_type=f"{lane_id}-fixture-v1",
        payload_entropy_bits=Fraction(fact_index + 1, 2),
        scheduled_exposures=1,
        expected_reads=Fraction(1, fact_index + 1),
        expected_hops=Fraction(fact_index, fact_index + 1),
        surfaces=(f"value:{lane_id}:{fact_index:06d}",),
    )


@dataclass
class FixtureLane:
    lane_id: LaneId
    lock: SourceLock
    order: str = "forward"
    records: int | None = None
    mutation: str | None = None
    finite: bool = True
    started: int = 0
    path_override: str | None = None
    require_compact_lengths: bool = False

    @property
    def training_edge_count(self) -> int:
        if self.lane_id != "wikidata_graph":
            raise AttributeError("only Wikidata graph has training-edge authority")
        return 3 if self.mutation == "authority_extra" else 2

    def iter_training_edge_keys(self, source_root: Path) -> Iterator[str]:
        del source_root
        if self.lane_id != "wikidata_graph":
            raise RuntimeError("only Wikidata graph has training-edge authority")
        if self.mutation == "authority_duplicate":
            yield "edge-000000"
            yield "edge-000000"
            return
        yield "edge-000000"
        yield "edge-000001"
        if self.mutation == "authority_extra":
            yield "edge-000002"

    def iter_drafts(
        self,
        source_root: Path,
        target_lengths: Sequence[int],
    ) -> Iterator[CatalogDraft]:
        self.started += 1
        if self.require_compact_lengths and (
            isinstance(target_lengths, tuple)
            or sys.getsizeof(target_lengths) > 256
        ):
            raise RuntimeError("target lengths were materialized")
        source_id = _LANE_SOURCE_IDS[self.lane_id]
        materialized_path, source_file = _source_file(self.lock, source_id)
        if self.path_override is not None:
            entry = next(row for row in self.lock.sources if row.source_id == source_id)
            source_file = next(
                row for row in entry.files if row.path == self.path_override
            )
        count = len(target_lengths) if self.records is None else self.records
        indices = list(range(count))
        if self.order == "reverse":
            indices.reverse()
        for emitted, index in enumerate(indices):
            if self.mutation == "raise_after_first" and emitted == 1:
                raise RuntimeError("fixture lane stopped unexpectedly")
            source_key = (
                f"edge-{index:06d}"
                if self.lane_id == "wikidata_graph" and index < 2
                else f"revisit-{index - 2:06d}"
                if self.lane_id == "wikidata_graph"
                else f"{self.lane_id}:{index:06d}"
            )
            flags = (
                ("graph-revisit",)
                if self.lane_id == "wikidata_graph" and index >= 2
                else ("graph-training-edge",)
                if self.lane_id == "wikidata_graph"
                else ("generated",)
                if self.lane_id in _GENERATED_LANES
                else ()
            )
            locator_items: list[tuple[str, str | int]] = [
                ("path", source_file.path),
                ("row", index),
            ]
            if source_id == "wikidata5m":
                locator_items.append(("split", "train"))
            if self.lane_id == "wikidata_graph":
                edge_index = index % 2
                locator_items.append(
                    ("training_edge_key", f"edge-{edge_index:06d}")
                )
            if self.lane_id in _GENERATED_LANES:
                lane_seed_base = LANE_ORDER.index(self.lane_id) * 1_000_000
                seed = (
                    lane_seed_base
                    if self.mutation == "reused_seed"
                    else lane_seed_base + index
                )
                if self.mutation == "alias_seed":
                    locator_items.append(("seed", seed))
                elif self.mutation != "missing_seed":
                    locator_items.append(("generation_seed", seed))
            elif self.mutation == "smuggled_seed" and index == 0:
                locator_items.append(("generation_seed", index))
            locator = tuple(
                sorted(locator_items, key=lambda item: item[0].encode("utf-8"))
            )
            semantic_facts = (_fact(self.lane_id, index),)
            draft = CatalogDraft(
                lane_id=self.lane_id,
                source_id=source_id,
                source_key=source_key,
                source_byte_sha256=source_file.sha256,
                source_locator=locator,
                semantic_flags=flags,
                semantic_facts=semantic_facts,
            )
            if self.mutation == "duplicate_source_key" and index == 1:
                draft = replace(
                    draft,
                    source_key=f"{self.lane_id}:000000",
                )
            elif self.mutation == "duplicate_fact_id" and index == 0:
                draft = replace(
                    draft,
                    semantic_facts=(semantic_facts[0], semantic_facts[0]),
                )
            elif self.mutation == "bad_hash" and index == 0:
                draft = replace(draft, source_byte_sha256="0" * 64)
            elif self.mutation == "sealed_path" and index == 0:
                draft = replace(
                    draft,
                    source_locator=tuple(
                        sorted(
                            (*locator, ("member", "evaluation/train/test.json")),
                            key=lambda item: item[0].encode("utf-8"),
                        )
                    ),
                )
            elif self.mutation == "missing_training_split" and index == 0:
                draft = replace(
                    draft,
                    source_locator=tuple(
                        item for item in locator if item[0] != "split"
                    ),
                )
            elif self.mutation == "graph_all_revisit":
                draft = replace(draft, semantic_flags=("graph-revisit",))
            elif (
                self.mutation == "graph_partial_before_revisit"
                and index == 1
            ):
                draft = replace(draft, semantic_flags=("graph-revisit",))
            elif (
                self.mutation == "graph_duplicate_before_revisit"
                and index == 1
            ):
                draft = replace(
                    draft,
                    source_locator=tuple(
                        (
                            key,
                            "edge-000000"
                            if key == "training_edge_key"
                            else value,
                        )
                        for key, value in locator
                    ),
                )
            elif self.mutation == "non_nfc" and index == 0:
                draft = replace(
                    draft,
                    semantic_facts=(
                        replace(
                            semantic_facts[0],
                            surfaces=("e\u0301",),
                        ),
                    ),
                )
            elif self.mutation == "non_finite" and index == 0:
                draft = replace(
                    draft,
                    semantic_facts=(
                        replace(
                            semantic_facts[0],
                            payload_entropy_bits=cast(Fraction, float("nan")),
                        ),
                    ),
                )
            elif self.mutation == "locator_order" and index == 0:
                draft = replace(draft, source_locator=tuple(reversed(locator)))
            elif self.mutation == "flag_order" and index == 0:
                draft = replace(draft, semantic_flags=("zeta", "alpha"))
            elif self.mutation == "fact_order" and index == 0:
                second = replace(
                    semantic_facts[0],
                    fact_id=semantic_facts[0].fact_id + ":z",
                    surfaces=(semantic_facts[0].surfaces[0] + ":z",),
                )
                draft = replace(
                    draft,
                    semantic_facts=(second, semantic_facts[0]),
                )
            elif self.mutation == "surface_order" and index == 0:
                draft = replace(
                    draft,
                    semantic_facts=(
                        replace(
                            semantic_facts[0],
                            surfaces=("zeta", "alpha"),
                        ),
                    ),
                )
            if self.mutation == "source_drift" and emitted == 0:
                path = source_root / materialized_path / source_file.path
                path.write_bytes(path.read_bytes() + b"drift")
            yield draft


@dataclass
class NoWikidataAuthority:
    wrapped: FixtureLane
    lane_id: LaneId = "wikidata_graph"
    finite: bool = True

    def iter_drafts(
        self,
        source_root: Path,
        target_lengths: Sequence[int],
    ) -> Iterator[CatalogDraft]:
        return self.wrapped.iter_drafts(source_root, target_lengths)


@dataclass
class HighCardinalityWikidataLane:
    lock: SourceLock
    training_edge_count: int
    lane_id: LaneId = "wikidata_graph"
    finite: bool = True

    def iter_training_edge_keys(self, source_root: Path) -> Iterator[str]:
        del source_root
        for index in range(self.training_edge_count):
            yield f"edge-{index:08d}"

    def iter_drafts(
        self,
        source_root: Path,
        target_lengths: Sequence[int],
    ) -> Iterator[CatalogDraft]:
        del source_root
        if isinstance(target_lengths, tuple) or sys.getsizeof(target_lengths) > 256:
            raise RuntimeError("high-cardinality target lengths were materialized")
        _materialized, source_file = _source_file(self.lock, "wikidata5m")
        for index in range(len(target_lengths)):
            revisit = index == self.training_edge_count
            edge_index = 0 if revisit else index
            fact = _fact("wikidata_graph", edge_index)
            yield CatalogDraft(
                lane_id="wikidata_graph",
                source_id="wikidata5m",
                source_key=(
                    f"revisit-{edge_index:08d}"
                    if revisit
                    else f"edge-{edge_index:08d}"
                ),
                source_byte_sha256=source_file.sha256,
                source_locator=(
                    ("path", source_file.path),
                    ("row", index),
                    ("split", "train"),
                    ("training_edge_key", f"edge-{edge_index:08d}"),
                ),
                semantic_flags=(
                    ("graph-revisit",)
                    if revisit
                    else ("graph-training-edge",)
                ),
                semantic_facts=(fact,),
            )


class FlippingMapping(Mapping[LaneId, FixtureLane]):
    def __init__(self, values: Mapping[LaneId, FixtureLane]) -> None:
        self._values = dict(values)
        self._iteration = 0

    def __getitem__(self, key: LaneId) -> FixtureLane:
        return self._values[key]

    def __iter__(self) -> Iterator[LaneId]:
        self._iteration += 1
        keys = list(self._values)
        if self._iteration % 2 == 0:
            keys.reverse()
        return iter(keys)

    def __len__(self) -> int:
        return len(self._values)


@pytest.fixture
def tiny_geometry() -> BuildGeometry:
    lane_quotas: tuple[tuple[LaneId, int], ...] = (
        ("fineweb_edu", 9),
        ("finemath", 7),
        ("wikidata_graph", 10),
        ("synthetic_graph", 6),
        ("verified_synthetic_multihop", 5),
        ("wikidata_path_reasoning", 5),
        ("relational_refinement", 4),
        ("objective_auxiliary", 3),
    )
    return BuildGeometry(
        profile="canary",
        total_targets=sum(value for _lane, value in lane_quotas),
        targets_per_update=1,
        context_length=4,
        shard_count=1,
        allow_fewer_shards=True,
        lane_quotas=lane_quotas,
    )


@pytest.fixture
def staged_fixture_sources(
    tmp_path: Path,
    fixture_source_lock: FixtureSourceLock,
):
    root = stage_source_lock(
        fixture_source_lock.lock,
        fixture_source_lock.download_root,
        tmp_path / "canonical",
        expected_generator_commit=fixture_source_lock.lock.generator_commit,
    )
    return SimpleNamespace(lock=fixture_source_lock.lock, root=root)


@pytest.fixture
def fixture_lane_sources(staged_fixture_sources):
    def make(
        *,
        order: str = "forward",
        mutation_lane: LaneId | None = None,
        mutation: str | None = None,
        records: int | None = None,
        require_compact_lengths: bool = False,
    ) -> dict[LaneId, FixtureLane]:
        lanes = LANE_ORDER if order == "forward" else tuple(reversed(LANE_ORDER))
        return {
            lane_id: FixtureLane(
                lane_id=lane_id,
                lock=staged_fixture_sources.lock,
                order=order,
                records=records if lane_id == mutation_lane else None,
                mutation=mutation if lane_id == mutation_lane else None,
                require_compact_lengths=require_compact_lengths,
            )
            for lane_id in lanes
        }

    return make


def _build(
    geometry: BuildGeometry,
    staged_fixture_sources,
    lane_sources: Mapping[LaneId, FixtureLane],
    output_root: Path,
) -> InputCatalog:
    return build_input_catalog(
        geometry,
        staged_fixture_sources.lock,
        staged_fixture_sources.root,
        lane_sources,
        output_root,
        expected_generator_commit=staged_fixture_sources.lock.generator_commit,
    )


def _quarantine_entries(parent: Path) -> tuple[Path, ...]:
    root = parent / _QUARANTINE_DIRECTORY
    assert root.is_dir()
    return tuple(sorted(root.iterdir(), key=lambda path: path.name))


@pytest.mark.parametrize("forged_lock", [False, True])
def test_catalog_requires_independent_generator_commit_authority_before_iteration(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
    forged_lock: bool,
):
    sources = fixture_lane_sources()
    lock = (
        replace(staged_fixture_sources.lock, generator_commit="b" * 40)
        if forged_lock
        else staged_fixture_sources.lock
    )
    expected = (
        staged_fixture_sources.lock.generator_commit
        if forged_lock
        else "b" * 40
    )
    with pytest.raises(ValueError, match="generator commit"):
        build_input_catalog(
            tiny_geometry,
            lock,
            staged_fixture_sources.root,
            sources,
            tmp_path / "stale-lock",
            expected_generator_commit=expected,
        )
    assert all(source.started == 0 for source in sources.values())


def test_catalog_is_canonical_exact_and_filesystem_order_independent(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
):
    forward = _build(
        tiny_geometry,
        staged_fixture_sources,
        fixture_lane_sources(order="forward"),
        tmp_path / "forward",
    )
    reverse = _build(
        tiny_geometry,
        staged_fixture_sources,
        fixture_lane_sources(order="reverse"),
        tmp_path / "reverse",
    )

    assert forward.sha256 == reverse.sha256
    assert forward.to_bytes() == reverse.to_bytes()
    assert forward.records_path.read_bytes() == reverse.records_path.read_bytes()
    assert forward.lane_target_counts == tiny_geometry.lane_quotas
    assert tuple(row.lane_id for row in forward.lanes) == LANE_ORDER
    assert sorted(path.name for path in forward.root.iterdir()) == [
        "catalog-index.json",
        "catalog.jsonl",
    ]

    records = tuple(forward.iter_records())
    assert [row.ordinal for row in records] == list(range(len(records)))
    assert len({row.record_id for row in records}) == len(records)
    assert all(row.target_count <= tiny_geometry.context_length for row in records)
    assert all("sealed-test" not in row.semantic_flags for row in records)
    assert sum(row.target_count for row in records) == tiny_geometry.total_targets

    by_lane = {
        lane_id: tuple(row.target_count for row in records if row.lane_id == lane_id)
        for lane_id in LANE_ORDER
    }
    for lane_id, quota in tiny_geometry.lane_quotas:
        assert by_lane[lane_id] == balanced_record_lengths(
            quota,
            tiny_geometry.context_length,
        )

    index_bytes = forward.to_bytes()
    index = json.loads(index_bytes)
    assert canonical_json_bytes(index) == index_bytes
    assert index["catalog_sha256"] == sha256_hex(forward.records_path.read_bytes())
    assert index["source_lock_sha256"] == staged_fixture_sources.lock.sha256
    assert index["record_count"] == len(records)
    assert index["target_count"] == tiny_geometry.total_targets

    for raw_line, record in zip(
        forward.records_path.read_bytes().splitlines(keepends=True),
        records,
        strict=True,
    ):
        assert canonical_json_bytes(json.loads(raw_line)) == raw_line
        draft = CatalogDraft(
            lane_id=record.lane_id,
            source_id=record.source_id,
            source_key=record.source_key,
            source_byte_sha256=record.source_byte_sha256,
            source_locator=record.source_locator,
            semantic_flags=record.semantic_flags,
            semantic_facts=record.semantic_facts,
        )
        assert record.record_id == catalog_record_id(draft, record.target_count)


def test_finite_reasoning_lane_never_cycles_to_fill_quota(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
):
    sources = fixture_lane_sources(
        mutation_lane="wikidata_path_reasoning",
        records=1,
    )
    with pytest.raises(
        LaneQuotaShortfall,
        match="wikidata_path_reasoning.*distinct deterministic records",
    ) as raised:
        _build(
            tiny_geometry,
            staged_fixture_sources,
            sources,
            tmp_path / "short",
        )
    assert raised.value.requested == 2
    assert raised.value.emitted == 1
    assert not (tmp_path / "short").exists()


def test_wikidata_graph_covers_every_training_edge_before_revisit(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
):
    sources = fixture_lane_sources()
    catalog = _build(
        tiny_geometry,
        staged_fixture_sources,
        sources,
        tmp_path / "catalog",
    )
    rows = [
        row for row in catalog.iter_records() if row.lane_id == "wikidata_graph"
    ]
    first_revisit = next(
        index for index, row in enumerate(rows) if "graph-revisit" in row.semantic_flags
    )
    assert {row.source_key for row in rows[:first_revisit]} == set(
        sources["wikidata_graph"].iter_training_edge_keys(
            staged_fixture_sources.root
        )
    )
    assert all(
        "graph-revisit" in row.semantic_flags for row in rows[first_revisit:]
    )


@pytest.mark.parametrize(
    ("lane_id", "mutation", "message"),
    [
        ("fineweb_edu", "duplicate_source_key", "duplicate source key"),
        ("fineweb_edu", "duplicate_fact_id", "duplicate semantic fact"),
        ("synthetic_graph", "reused_seed", "seed.*reused"),
        ("synthetic_graph", "missing_seed", "generation seed.*required"),
        ("synthetic_graph", "alias_seed", "generation seed.*canonical"),
        ("fineweb_edu", "smuggled_seed", "must not carry.*seed"),
        ("fineweb_edu", "bad_hash", "source byte.*hash"),
        ("wikidata_path_reasoning", "sealed_path", "sealed.*path"),
        ("fineweb_edu", "non_nfc", "NFC"),
        ("fineweb_edu", "non_finite", "finite"),
        ("fineweb_edu", "locator_order", "source locator.*order"),
        ("fineweb_edu", "flag_order", "semantic flags.*order"),
        ("fineweb_edu", "fact_order", "semantic facts.*order"),
        ("fineweb_edu", "surface_order", "surfaces.*order"),
    ],
)
def test_catalog_rejects_noncanonical_or_nonunique_drafts(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
    lane_id: LaneId,
    mutation: str,
    message: str,
):
    sources = fixture_lane_sources(
        mutation_lane=lane_id,
        mutation=mutation,
    )
    output = tmp_path / mutation
    with pytest.raises(ValueError, match=message):
        _build(tiny_geometry, staged_fixture_sources, sources, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("graph_all_revisit", "before.*revisit"),
        ("graph_partial_before_revisit", "every training edge"),
        ("graph_duplicate_before_revisit", "exactly once"),
        ("authority_duplicate", "duplicate.*training edge"),
        ("authority_extra", "every training edge"),
    ],
)
def test_wikidata_graph_requires_exact_complete_once_authority(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
    mutation: str,
    message: str,
):
    sources = fixture_lane_sources(
        mutation_lane="wikidata_graph",
        mutation=mutation,
    )
    with pytest.raises(ValueError, match=message):
        _build(
            tiny_geometry,
            staged_fixture_sources,
            sources,
            tmp_path / mutation,
        )


def test_wikidata_graph_rejects_missing_training_edge_authority(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
):
    sources = fixture_lane_sources()
    sources["wikidata_graph"] = cast(
        FixtureLane,
        NoWikidataAuthority(sources["wikidata_graph"]),
    )
    with pytest.raises(ValueError, match="training-edge authority.*required"):
        _build(
            tiny_geometry,
            staged_fixture_sources,
            sources,
            tmp_path / "missing-edge-authority",
        )


def test_wikidata_source_requires_exact_training_split_authority(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
):
    sources = fixture_lane_sources(
        mutation_lane="wikidata_path_reasoning",
        mutation="missing_training_split",
    )
    with pytest.raises(ValueError, match="Wikidata.*training split"):
        _build(
            tiny_geometry,
            staged_fixture_sources,
            sources,
            tmp_path / "missing-training-split",
        )


def test_reserved_path_cannot_hide_behind_train_component(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    fixture_source_lock: FixtureSourceLock,
):
    relative = "evaluation/train/test.json"
    payload = b"must remain sealed"
    entry = next(
        row for row in fixture_source_lock.lock.sources if row.source_id == "clrs_text"
    )
    materialized = fixture_source_lock.download_root / entry.materialized_path
    path = materialized / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    extra = SourceFile(
        path=relative,
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    changed_entry = replace(
        entry,
        files=tuple(
            sorted((*entry.files, extra), key=lambda row: row.path.encode("utf-8"))
        ),
    )
    changed_lock = replace(
        fixture_source_lock.lock,
        sources=tuple(
            changed_entry if row.source_id == "clrs_text" else row
            for row in fixture_source_lock.lock.sources
        ),
    )
    source_root = stage_source_lock(
        changed_lock,
        fixture_source_lock.download_root,
        tmp_path / "reserved-canonical",
        expected_generator_commit=changed_lock.generator_commit,
    )
    sources = {
        lane_id: FixtureLane(
            lane_id=lane_id,
            lock=changed_lock,
            path_override=relative if lane_id == "relational_refinement" else None,
        )
        for lane_id in LANE_ORDER
    }
    staged = SimpleNamespace(lock=changed_lock, root=source_root)
    with pytest.raises(ValueError, match="sealed or evaluation path"):
        _build(
            tiny_geometry,
            staged,
            sources,
            tmp_path / "reserved-output",
        )


def test_catalog_rejects_missing_and_unknown_lane_sources(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
):
    missing = fixture_lane_sources()
    del missing["objective_auxiliary"]
    with pytest.raises(ValueError, match="missing lane source.*objective_auxiliary"):
        _build(
            tiny_geometry,
            staged_fixture_sources,
            missing,
            tmp_path / "missing",
        )

    unknown = fixture_lane_sources()
    unknown[cast(LaneId, "unknown_lane")] = FixtureLane(
        lane_id=cast(LaneId, "unknown_lane"),
        lock=staged_fixture_sources.lock,
    )
    with pytest.raises(ValueError, match="unknown lane source.*unknown_lane"):
        _build(
            tiny_geometry,
            staged_fixture_sources,
            unknown,
            tmp_path / "unknown",
        )


def test_catalog_rejects_lane_source_identity_mismatch(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
):
    sources = fixture_lane_sources()
    sources["fineweb_edu"] = FixtureLane(
        lane_id="finemath",
        lock=staged_fixture_sources.lock,
    )
    with pytest.raises(ValueError, match="lane source identity mismatch"):
        _build(
            tiny_geometry,
            staged_fixture_sources,
            sources,
            tmp_path / "mismatch",
        )


def test_catalog_rejects_unstable_mapping_key_order(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
):
    sources = FlippingMapping(fixture_lane_sources())
    with pytest.raises(ValueError, match="unstable mapping key order"):
        _build(
            tiny_geometry,
            staged_fixture_sources,
            sources,
            tmp_path / "unstable",
        )


def test_lane_sources_receive_compact_balanced_target_lengths(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
):
    catalog = _build(
        tiny_geometry,
        staged_fixture_sources,
        fixture_lane_sources(require_compact_lengths=True),
        tmp_path / "compact-lengths",
    )
    records = tuple(catalog.iter_records())
    for lane_id, quota in tiny_geometry.lane_quotas:
        assert tuple(
            row.target_count for row in records if row.lane_id == lane_id
        ) == balanced_record_lengths(
            quota,
            tiny_geometry.context_length,
        )


def test_high_cardinality_indexes_and_record_replay_remain_memory_bounded(
    tmp_path: Path,
    staged_fixture_sources,
    fixture_lane_sources,
):
    edge_count = 2_000
    lane_quotas = tuple(
        (
            lane_id,
            edge_count + 1 if lane_id == "wikidata_graph" else 1,
        )
        for lane_id in LANE_ORDER
    )
    geometry = BuildGeometry(
        profile="canary",
        total_targets=sum(quota for _lane, quota in lane_quotas),
        targets_per_update=1,
        context_length=1,
        shard_count=1,
        allow_fewer_shards=True,
        lane_quotas=lane_quotas,
    )
    sources = fixture_lane_sources()
    sources["wikidata_graph"] = cast(
        FixtureLane,
        HighCardinalityWikidataLane(
            lock=staged_fixture_sources.lock,
            training_edge_count=edge_count,
        ),
    )

    tracemalloc.start()
    catalog = _build(
        geometry,
        staged_fixture_sources,
        sources,
        tmp_path / "high-cardinality",
    )
    _current, build_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert build_peak < 32 * 1024 * 1024

    tracemalloc.start()
    replayed = sum(1 for _row in catalog.iter_records())
    _current, replay_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert replayed == edge_count + len(LANE_ORDER)
    assert replay_peak < 8 * 1024 * 1024


def test_source_tree_is_reverified_after_lane_consumption(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
):
    sources = fixture_lane_sources(
        mutation_lane="relational_refinement",
        mutation="source_drift",
    )
    output = tmp_path / "drift"
    with pytest.raises(ValueError, match="source byte drift"):
        _build(tiny_geometry, staged_fixture_sources, sources, output)
    assert not output.exists()


def test_failed_stage_is_exact_inode_quarantined_without_path_deletion(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
):
    sources = fixture_lane_sources(
        mutation_lane="fineweb_edu",
        mutation="raise_after_first",
    )
    output = tmp_path / "transactional"
    with pytest.raises(ValueError, match="lane source.*fineweb_edu"):
        _build(tiny_geometry, staged_fixture_sources, sources, output)
    assert not output.exists()
    assert not tuple(tmp_path.glob(".transactional.tmp-*"))
    retained = _quarantine_entries(tmp_path)
    assert retained
    assert any(
        path.is_dir() and (path / ".catalog-spool.sqlite3").is_file()
        for path in retained
    )
    source = inspect.getsource(catalog_module)
    assert "os.unlink(" not in source
    assert "os.rmdir(" not in source


def test_exact_existing_catalog_is_verified_reused_and_loser_quarantined(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
):
    output = tmp_path / "existing"
    first = _build(
        tiny_geometry,
        staged_fixture_sources,
        fixture_lane_sources(),
        output,
    )
    before = _quarantine_entries(tmp_path)
    second = _build(
        tiny_geometry,
        staged_fixture_sources,
        fixture_lane_sources(order="reverse"),
        output,
    )
    after = _quarantine_entries(tmp_path)
    assert second.sha256 == first.sha256
    assert second.to_bytes() == first.to_bytes()
    assert second.records_path.read_bytes() == first.records_path.read_bytes()
    assert len(after) >= len(before) + 2
    assert sorted(path.name for path in output.iterdir()) == [
        "catalog-index.json",
        "catalog.jsonl",
    ]


def test_conflicting_existing_catalog_is_preserved_and_loser_quarantined(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_bytes(b"keep")
    sources = fixture_lane_sources()
    with pytest.raises(ValueError, match="conflicting catalog winner"):
        _build(
            tiny_geometry,
            staged_fixture_sources,
            sources,
            output,
        )
    assert sentinel.read_bytes() == b"keep"
    assert all(source.started == 1 for source in sources.values())
    retained = _quarantine_entries(tmp_path)
    assert any(path.is_dir() for path in retained)


def test_catalog_publication_losing_race_rejects_empty_winner_and_quarantines_loser(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "race"
    original = catalog_module.atomic_rename_noreplace
    lost = False

    def lose_race(
        source_directory_fd: int,
        source_name: str,
        destination_directory_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal lost
        if destination_name == output.name and not lost:
            lost = True
            os.mkdir(
                destination_name,
                mode=0o700,
                dir_fd=destination_directory_fd,
            )
            raise FileExistsError(destination_name)
        original(
            source_directory_fd,
            source_name,
            destination_directory_fd,
            destination_name,
        )

    monkeypatch.setattr(catalog_module, "atomic_rename_noreplace", lose_race)
    with pytest.raises(ValueError, match="conflicting catalog winner"):
        _build(
            tiny_geometry,
            staged_fixture_sources,
            fixture_lane_sources(),
            output,
        )
    assert output.is_dir()
    assert not tuple(output.iterdir())
    assert not tuple(tmp_path.glob(".race.tmp-*"))
    retained = _quarantine_entries(tmp_path)
    assert any(path.is_dir() for path in retained)


def test_sqlite_spool_namespace_swap_fails_and_preserves_both_inodes(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
    monkeypatch: pytest.MonkeyPatch,
):
    swapped = False

    def swap(
        phase: str,
        stage_fd: int,
        name: str,
        pinned_fd: int,
    ) -> None:
        nonlocal swapped
        del pinned_fd
        if phase != "before_sqlite_open" or swapped:
            return
        swapped = True
        os.rename(
            name,
            ".attacker-original-spool",
            src_dir_fd=stage_fd,
            dst_dir_fd=stage_fd,
        )
        replacement = os.open(
            name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=stage_fd,
        )
        os.close(replacement)

    monkeypatch.setattr(
        catalog_module,
        "_spool_open_hook",
        swap,
        raising=False,
    )
    output = tmp_path / "spool-swap"
    with pytest.raises(ValueError, match="SQLite spool.*identity"):
        _build(
            tiny_geometry,
            staged_fixture_sources,
            fixture_lane_sources(),
            output,
        )
    assert not output.exists()
    retained = _quarantine_entries(tmp_path)
    stage = next(path for path in retained if path.is_dir())
    assert (stage / ".attacker-original-spool").is_file()
    assert (stage / ".catalog-spool.sqlite3").is_file()


def test_concurrent_spool_open_cannot_snapshot_another_stage_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    origin = tmp_path / "origin"
    stage_a = tmp_path / "stage-a"
    stage_b = tmp_path / "stage-b"
    origin.mkdir()
    stage_a.mkdir()
    stage_b.mkdir()
    monkeypatch.chdir(origin)
    origin_fd = os.open(".", os.O_RDONLY | os.O_DIRECTORY)
    stage_fds = {
        "spool-a": os.open(stage_a, os.O_RDONLY | os.O_DIRECTORY),
        "spool-b": os.open(stage_b, os.O_RDONLY | os.O_DIRECTORY),
    }
    original_connect = catalog_module.sqlite3.connect
    original_open = catalog_module.os.open
    first_inside_sqlite = threading.Event()
    release_first = threading.Event()
    second_lock_attempt = threading.Event()
    second_snapshotted_cwd = threading.Event()
    spools: dict[str, object] = {}
    marker_results: dict[str, tuple[str]] = {}
    errors: list[BaseException] = []

    def blocking_connect(*args, **kwargs):
        if threading.current_thread().name == "spool-a":
            first_inside_sqlite.set()
            if not release_first.wait(5):
                raise RuntimeError("timed out releasing first SQLite open")
        return original_connect(*args, **kwargs)

    def tracking_open(path, *args, **kwargs):
        if path == "." and threading.current_thread().name == "spool-b":
            second_snapshotted_cwd.set()
        return original_open(path, *args, **kwargs)

    def worker() -> None:
        name = threading.current_thread().name
        try:
            spool = catalog_module._create_spool(stage_fds[name])
            spool.connection.execute(
                "CREATE TABLE thread_marker(value TEXT NOT NULL)"
            )
            spool.connection.execute(
                "INSERT INTO thread_marker(value) VALUES (?)",
                (name,),
            )
            spool.connection.commit()
            marker_results[name] = spool.connection.execute(
                "SELECT value FROM thread_marker"
            ).fetchone()
            spool.abort_connection()
            spools[name] = spool
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(catalog_module.sqlite3, "connect", blocking_connect)
    monkeypatch.setattr(catalog_module.os, "open", tracking_open)
    monkeypatch.setattr(
        catalog_module,
        "_SQLITE_OPEN_LOCK",
        ObservedThreadLock({"spool-b": second_lock_attempt}),
    )
    first = threading.Thread(target=worker, name="spool-a")
    second = threading.Thread(target=worker, name="spool-b")
    try:
        first.start()
        assert first_inside_sqlite.wait(5)
        second.start()
        assert second_lock_attempt.wait(5)
        assert not second_snapshotted_cwd.is_set()
        release_first.set()
        first.join(5)
        second.join(5)
        assert not first.is_alive()
        assert not second.is_alive()
        assert errors == []
        assert os.fstat(origin_fd).st_ino == os.stat(".").st_ino
        for name, stage_fd in stage_fds.items():
            spool = spools[name]
            assert os.fstat(spool.descriptor).st_ino == os.stat(
                ".catalog-spool.sqlite3",
                dir_fd=stage_fd,
                follow_symlinks=False,
            ).st_ino
            assert marker_results[name] == (name,)
        assert os.fstat(spools["spool-a"].descriptor).st_ino != os.fstat(
            spools["spool-b"].descriptor
        ).st_ino
        Path("relative-after-spools.txt").write_text(
            "origin",
            encoding="utf-8",
        )
        assert (origin / "relative-after-spools.txt").read_text(
            encoding="utf-8"
        ) == "origin"
        assert not (stage_a / "relative-after-spools.txt").exists()
        assert not (stage_b / "relative-after-spools.txt").exists()
    finally:
        release_first.set()
        if first.ident is not None:
            first.join(5)
        if second.ident is not None:
            second.join(5)
        os.fchdir(origin_fd)
        for spool in spools.values():
            spool.abort_connection()
            spool.close_descriptor()
        for descriptor in stage_fds.values():
            os.close(descriptor)
        os.close(origin_fd)


_SQLITE_CWD_HOOK_PHASES = (
    "before_sqlite_open",
    "before_cwd_snapshot",
    "after_cwd_snapshot",
    "after_stage_fchdir",
    "sqlite_opened",
    "cwd_restored",
    "after_sqlite_open",
)


class ObservedThreadLock:
    def __init__(self, attempts: Mapping[str, threading.Event]) -> None:
        self._lock = threading.Lock()
        self._attempts = attempts

    def __enter__(self):
        event = self._attempts.get(threading.current_thread().name)
        if event is not None:
            event.set()
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._lock.release()


@pytest.mark.parametrize("pause_phase", _SQLITE_CWD_HOOK_PHASES)
def test_concurrent_spool_hooks_keep_cwd_transition_serialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pause_phase: str,
):
    origin = tmp_path / f"origin-{pause_phase}"
    stage_a = tmp_path / f"stage-a-{pause_phase}"
    stage_b = tmp_path / f"stage-b-{pause_phase}"
    origin.mkdir()
    stage_a.mkdir()
    stage_b.mkdir()
    monkeypatch.chdir(origin)
    origin_fd = os.open(".", os.O_RDONLY | os.O_DIRECTORY)
    origin_inode = os.fstat(origin_fd).st_ino
    stage_fds = {
        "hook-a": os.open(stage_a, os.O_RDONLY | os.O_DIRECTORY),
        "hook-b": os.open(stage_b, os.O_RDONLY | os.O_DIRECTORY),
    }
    stage_inodes = {
        name: os.fstat(descriptor).st_ino
        for name, descriptor in stage_fds.items()
    }
    paused = threading.Event()
    release = threading.Event()
    second_lock_attempt = threading.Event()
    second_before_open = threading.Event()
    second_before_snapshot = threading.Event()
    records: list[tuple[str, str, int]] = []
    record_lock = threading.Lock()
    spools: dict[str, object] = {}
    errors: list[BaseException] = []

    def hook(phase: str, stage_fd: int, name: str, pinned_fd: int) -> None:
        del stage_fd, name, pinned_fd
        thread_name = threading.current_thread().name
        cwd_inode = os.stat(".").st_ino
        with record_lock:
            records.append((thread_name, phase, cwd_inode))
        if thread_name == "hook-b" and phase == "before_sqlite_open":
            second_before_open.set()
        if thread_name == "hook-b" and phase == "before_cwd_snapshot":
            second_before_snapshot.set()
        if thread_name == "hook-a" and phase == pause_phase:
            paused.set()
            if not release.wait(5):
                raise RuntimeError(f"timed out at hook {phase}")

    def worker() -> None:
        name = threading.current_thread().name
        try:
            spool = catalog_module._create_spool(stage_fds[name])
            spool.abort_connection()
            spools[name] = spool
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(catalog_module, "_spool_open_hook", hook)
    monkeypatch.setattr(
        catalog_module,
        "_SQLITE_OPEN_LOCK",
        ObservedThreadLock({"hook-b": second_lock_attempt}),
    )
    first = threading.Thread(target=worker, name="hook-a")
    second = threading.Thread(target=worker, name="hook-b")
    try:
        first.start()
        assert paused.wait(5)
        second.start()
        assert second_lock_attempt.wait(5)
        assert not second_before_open.is_set()
        assert not second_before_snapshot.is_set()
        release.set()
        assert second_before_open.wait(5)
        first.join(5)
        second.join(5)
        assert not first.is_alive()
        assert not second.is_alive()
        assert errors == []
        assert os.stat(".").st_ino == origin_inode
        for name in ("hook-a", "hook-b"):
            phases = tuple(
                phase
                for thread_name, phase, _cwd_inode in records
                if thread_name == name
            )
            assert phases == _SQLITE_CWD_HOOK_PHASES
            by_phase = {
                phase: cwd_inode
                for thread_name, phase, cwd_inode in records
                if thread_name == name
            }
            assert by_phase["before_sqlite_open"] == origin_inode
            assert by_phase["before_cwd_snapshot"] == origin_inode
            assert by_phase["after_cwd_snapshot"] == origin_inode
            assert by_phase["after_stage_fchdir"] == stage_inodes[name]
            assert by_phase["sqlite_opened"] == stage_inodes[name]
            assert by_phase["cwd_restored"] == origin_inode
            assert by_phase["after_sqlite_open"] == origin_inode
            spool = spools[name]
            assert os.fstat(spool.descriptor).st_ino == os.stat(
                ".catalog-spool.sqlite3",
                dir_fd=stage_fds[name],
                follow_symlinks=False,
            ).st_ino
        Path("relative-hook-result.txt").write_text("origin", encoding="utf-8")
        assert (origin / "relative-hook-result.txt").is_file()
        assert not (stage_a / "relative-hook-result.txt").exists()
        assert not (stage_b / "relative-hook-result.txt").exists()
    finally:
        release.set()
        if first.ident is not None:
            first.join(5)
        if second.ident is not None:
            second.join(5)
        os.fchdir(origin_fd)
        for spool in spools.values():
            spool.abort_connection()
            spool.close_descriptor()
        for descriptor in stage_fds.values():
            os.close(descriptor)
        os.close(origin_fd)


def test_catalog_publication_rejects_symlinked_output_parent(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        _build(
            tiny_geometry,
            staged_fixture_sources,
            fixture_lane_sources(),
            linked_parent / "catalog",
        )
    assert not (real_parent / "catalog").exists()


def test_record_id_binds_draft_core_and_target_count(
    staged_fixture_sources,
):
    source_id = "fineweb_edu"
    _materialized, source_file = _source_file(
        staged_fixture_sources.lock,
        source_id,
    )
    draft = CatalogDraft(
        lane_id="fineweb_edu",
        source_id=source_id,
        source_key="fineweb_edu:record",
        source_byte_sha256=source_file.sha256,
        source_locator=(("path", source_file.path), ("row", 0)),
        semantic_flags=(),
        semantic_facts=(_fact("fineweb_edu", 0),),
    )
    first = catalog_record_id(draft, 3)
    assert first == catalog_record_id(draft, 3)
    assert first != catalog_record_id(draft, 4)
    assert len(first) == hashlib.sha256().digest_size * 2


def test_iter_records_rejects_noncanonical_or_truncated_jsonl(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
):
    catalog = _build(
        tiny_geometry,
        staged_fixture_sources,
        fixture_lane_sources(),
        tmp_path / "catalog",
    )
    first_line = catalog.records_path.read_bytes().splitlines()[0]
    catalog.records_path.write_bytes(first_line)
    with pytest.raises(ValueError, match="canonical|newline|record count|identity"):
        tuple(catalog.iter_records())


# ---------------------------------------------------------------------------
# Task 3: production Wikidata catalog adapter over a verified derived view
# ---------------------------------------------------------------------------


@pytest.fixture
def wikidata_view_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    from test_reasoning_v2_wikidata_source import (
        _base_archive_payloads,
        _install_archive_authority,
    )

    authority = _install_archive_authority(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        _base_archive_payloads(),
    )
    commit = fixture_source_lock.lock.generator_commit
    lock = load_source_lock(
        authority.source_lock_path,
        expected_generator_commit=commit,
    )
    root = stage_source_lock(
        lock,
        authority.source_root,
        tmp_path / "canonical",
        expected_generator_commit=commit,
    )
    view_ref = build_wikidata_derived_view(
        authority.source_lock_path,
        root,
        tmp_path / "views",
        expected_generator_commit=commit,
    )
    # A held-open view session pins the source-lock parent directory identity
    # (inode/link count/mtime). Catalog output therefore has to live under a
    # dedicated directory created before the session opens, so building never
    # mutates the pinned parent. In production the source tree and catalog
    # output are already separate roots; this only matters for the shared
    # tmp_path in tests.
    outputs = tmp_path / "catalogs"
    outputs.mkdir()
    return SimpleNamespace(
        lock=lock,
        root=root,
        source_lock_path=authority.source_lock_path,
        view_ref=view_ref,
        commit=commit,
        outputs=outputs,
    )


def _wikidata_graph_source(view: object):
    from corpusgen.reasoning_v2.catalog import WikidataGraphCatalogSource

    return WikidataGraphCatalogSource(view)


def _wikidata_end_to_end_geometry(wikidata_quota: int) -> BuildGeometry:
    lane_quotas: tuple[tuple[LaneId, int], ...] = tuple(
        (lane_id, wikidata_quota if lane_id == "wikidata_graph" else 1)
        for lane_id in LANE_ORDER
    )
    return BuildGeometry(
        profile="canary",
        total_targets=sum(quota for _lane, quota in lane_quotas),
        targets_per_update=1,
        context_length=1,
        shard_count=1,
        allow_fewer_shards=True,
        lane_quotas=lane_quotas,
    )


def _end_to_end_sources(lock: SourceLock, adapter: object):
    sources = {
        lane_id: FixtureLane(lane_id=lane_id, lock=lock)
        for lane_id in LANE_ORDER
        if lane_id != "wikidata_graph"
    }
    sources[cast(LaneId, "wikidata_graph")] = cast(FixtureLane, adapter)
    return sources


def test_catalog_receipt_binds_verified_wikidata_view_sha256(
    wikidata_view_authority,
):
    auth = wikidata_view_authority
    geometry = _wikidata_end_to_end_geometry(3)
    with open_wikidata_derived_view(
        auth.source_lock_path,
        auth.root,
        auth.view_ref,
        expected_generator_commit=auth.commit,
    ) as view:
        adapter = _wikidata_graph_source(view)
        sources = _end_to_end_sources(auth.lock, adapter)
        catalog = build_input_catalog(
            geometry,
            auth.lock,
            auth.root,
            sources,
            auth.outputs / "catalog",
            expected_generator_commit=auth.commit,
        )
    index = json.loads(catalog.to_bytes())
    assert index["wikidata_view_sha256"] == auth.view_ref.receipt_sha256
    assert canonical_json_bytes(index) == catalog.to_bytes()
    assert getattr(catalog, "wikidata_view_sha256", None) == auth.view_ref.receipt_sha256
    graph_records = [
        row for row in catalog.iter_records() if row.lane_id == "wikidata_graph"
    ]
    assert graph_records
    for record in graph_records:
        assert (
            dict(record.source_locator)["wikidata_view_sha256"]
            == auth.view_ref.receipt_sha256
        )


def test_end_to_end_catalog_is_byte_identical_across_rebuilds(
    wikidata_view_authority,
):
    auth = wikidata_view_authority
    geometry = _wikidata_end_to_end_geometry(3)

    def build(output_name: str) -> InputCatalog:
        with open_wikidata_derived_view(
            auth.source_lock_path,
            auth.root,
            auth.view_ref,
            expected_generator_commit=auth.commit,
        ) as view:
            adapter = _wikidata_graph_source(view)
            sources = _end_to_end_sources(auth.lock, adapter)
            return build_input_catalog(
                geometry,
                auth.lock,
                auth.root,
                sources,
                auth.outputs / output_name,
                expected_generator_commit=auth.commit,
            )

    first = build("first")
    second = build("second")
    assert first.sha256 == second.sha256
    assert first.to_bytes() == second.to_bytes()
    assert first.records_path.read_bytes() == second.records_path.read_bytes()

    graph_records = [
        row for row in first.iter_records() if row.lane_id == "wikidata_graph"
    ]
    assert len(graph_records) == 3
    first_revisit = next(
        (
            index
            for index, row in enumerate(graph_records)
            if "graph-revisit" in row.semantic_flags
        ),
        len(graph_records),
    )
    edge_keys_before_revisit = {
        dict(row.source_locator)["training_edge_key"]
        for row in graph_records[:first_revisit]
    }
    assert edge_keys_before_revisit == {"Q1\tP1\tQ2", "Q3\tP2\tQ4"}
    assert all(
        "graph-revisit" in row.semantic_flags
        for row in graph_records[first_revisit:]
    )
    for record in graph_records:
        assert (
            dict(record.source_locator)["wikidata_view_sha256"]
            == auth.view_ref.receipt_sha256
        )
