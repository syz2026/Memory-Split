from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, cast

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
    stage_source_lock,
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

    @property
    def training_edge_keys(self) -> tuple[str, ...]:
        if self.lane_id != "wikidata_graph":
            return ()
        return ("edge-000000", "edge-000001")

    def iter_drafts(
        self,
        source_root: Path,
        target_lengths: tuple[int, ...],
    ) -> Iterator[CatalogDraft]:
        source_id = _LANE_SOURCE_IDS[self.lane_id]
        materialized_path, source_file = _source_file(self.lock, source_id)
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
            locator: tuple[tuple[str, str | int], ...] = (
                ("path", source_file.path),
                ("row", index),
            )
            if self.lane_id in _GENERATED_LANES:
                seed = 0 if self.mutation == "reused_seed" else index
                locator = (*locator, ("seed", seed))
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
                    source_locator=(*locator, ("split", "evaluation/test")),
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
    ) -> dict[LaneId, FixtureLane]:
        lanes = LANE_ORDER if order == "forward" else tuple(reversed(LANE_ORDER))
        return {
            lane_id: FixtureLane(
                lane_id=lane_id,
                lock=staged_fixture_sources.lock,
                order=order,
                records=records if lane_id == mutation_lane else None,
                mutation=mutation if lane_id == mutation_lane else None,
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
    )


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
        sources["wikidata_graph"].training_edge_keys
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


def test_catalog_rejects_target_lengths_that_do_not_sum_to_lane_quota(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
    monkeypatch: pytest.MonkeyPatch,
):
    original: Callable[[int, int], tuple[int, ...]] = (
        catalog_module.balanced_record_lengths
    )

    def broken(targets: int, context_length: int) -> tuple[int, ...]:
        lengths = original(targets, context_length)
        if targets == tiny_geometry.lane_quotas[0][1]:
            return (*lengths[:-1], lengths[-1] - 1)
        return lengths

    monkeypatch.setattr(catalog_module, "balanced_record_lengths", broken)
    with pytest.raises(ValueError, match="target lengths.*lane quota"):
        _build(
            tiny_geometry,
            staged_fixture_sources,
            fixture_lane_sources(),
            tmp_path / "bad-lengths",
        )


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


def test_catalog_publication_is_transactional_on_iterator_failure(
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


def test_catalog_publication_never_replaces_existing_output(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_bytes(b"keep")
    with pytest.raises((FileExistsError, ValueError), match="already exists|no-replace"):
        _build(
            tiny_geometry,
            staged_fixture_sources,
            fixture_lane_sources(),
            output,
        )
    assert sentinel.read_bytes() == b"keep"


def test_catalog_publication_losing_race_keeps_winner_and_cleans_candidate(
    tmp_path: Path,
    tiny_geometry: BuildGeometry,
    staged_fixture_sources,
    fixture_lane_sources,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "race"

    def lose_race(
        source_directory_fd: int,
        source_name: str,
        destination_directory_fd: int,
        destination_name: str,
    ) -> None:
        del source_directory_fd, source_name
        os.mkdir(destination_name, mode=0o700, dir_fd=destination_directory_fd)
        raise FileExistsError(destination_name)

    monkeypatch.setattr(catalog_module, "atomic_rename_noreplace", lose_race)
    with pytest.raises((FileExistsError, ValueError), match="no-replace|already exists"):
        _build(
            tiny_geometry,
            staged_fixture_sources,
            fixture_lane_sources(),
            output,
        )
    assert output.is_dir()
    assert not tuple(output.iterdir())
    assert not tuple(tmp_path.glob(".race.tmp-*"))


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
