from __future__ import annotations

import copy
import gc
import gzip
import hashlib
import importlib.util
import io
import inspect
import json
import os
import subprocess
import sys
import tarfile
import threading
import tracemalloc
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

from corpusgen.parallel.canonical import canonical_json_bytes, sha256_hex
from corpusgen.reasoning_v2 import catalog as catalog_module
from corpusgen.reasoning_v2 import source_lock as source_lock_module
from corpusgen.reasoning_v2 import wikidata_source as wikidata_source_module
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


EXPECTED_GENERATOR_COMMIT = "a" * 40
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


@dataclass(frozen=True)
class _TarEntry:
    name: str
    payload: bytes


@dataclass(frozen=True)
class _AuthorityFixture:
    source_lock_path: Path
    source_root: Path


def _tar_bytes(entries: list[_TarEntry]) -> bytes:
    compressed = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=compressed,
        mtime=0,
    ) as gzip_stream:
        with tarfile.open(
            fileobj=gzip_stream,
            mode="w",
            format=tarfile.PAX_FORMAT,
        ) as archive:
            for entry in entries:
                member = tarfile.TarInfo(entry.name)
                member.uid = 0
                member.gid = 0
                member.uname = ""
                member.gname = ""
                member.mtime = 0
                member.mode = 0o600
                member.type = tarfile.REGTYPE
                member.size = len(entry.payload)
                archive.addfile(member, io.BytesIO(entry.payload))
    return compressed.getvalue()


def _base_archive_entries() -> dict[str, list[_TarEntry]]:
    return {
        "wikidata5m_alias.tar.gz": [
            _TarEntry("wikidata5m_entity.txt", b"Q1\tAda Lovelace\n"),
            _TarEntry("wikidata5m_relation.txt", b"P1\tknows\n"),
        ],
        "wikidata5m_inductive.tar.gz": [
            _TarEntry("wikidata5m_inductive_test.txt", b"Q9\tP9\tQ10\n"),
            _TarEntry("wikidata5m_inductive_train.txt", b"Q1\tP1\tQ2\n"),
            _TarEntry("wikidata5m_inductive_valid.txt", b"Q7\tP7\tQ8\n"),
        ],
        "wikidata5m_transductive.tar.gz": [
            _TarEntry("wikidata5m_transductive_test.txt", b"Q11\tP11\tQ12\n"),
            _TarEntry("wikidata5m_transductive_train.txt", b"Q3\tP2\tQ4\n"),
            _TarEntry("wikidata5m_transductive_valid.txt", b"Q13\tP13\tQ14\n"),
        ],
    }


def _base_archive_payloads() -> dict[str, bytes]:
    return {
        name: _tar_bytes(entries)
        for name, entries in _base_archive_entries().items()
    }


def _replace_archive_member(
    entries: dict[str, list[_TarEntry]],
    archive_name: str,
    member_name: str,
    payload: bytes,
) -> None:
    archive_entries = entries[archive_name]
    index = next(
        position
        for position, entry in enumerate(archive_entries)
        if entry.name == member_name
    )
    archive_entries[index] = replace(archive_entries[index], payload=payload)


def _install_archive_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
    archive_payloads: dict[str, bytes],
) -> _AuthorityFixture:
    assert tuple(archive_payloads) == wikidata_source_module.ARCHIVE_PATHS
    archive_metadata = {
        name: {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in archive_payloads.items()
    }
    monkeypatch.setattr(
        source_lock_module,
        "FIXED_WIKIDATA_FILES",
        copy.deepcopy(archive_metadata),
    )
    source_lock_module.WIKIDATA_LOCK_PATH.write_bytes(
        canonical_json_bytes(
            {
                "files": archive_metadata,
                "repo_id": "intfloat/wikidata5m",
                "repo_type": "dataset",
                "revision": "6b2b09672129e280c0c9da97ab58154e9d535e6b",
            }
        )
    )

    source_root = fixture_source_lock.download_root
    wikidata_root = source_root / "wikidata5m"
    for name, payload in archive_payloads.items():
        (wikidata_root / name).write_bytes(payload)

    original = fixture_source_lock.lock
    wikidata_entry = next(
        entry for entry in original.sources if entry.source_id == "wikidata5m"
    )
    replacement_rows = {
        name: SourceFile(
            path=name,
            bytes=metadata["bytes"],
            sha256=metadata["sha256"],
        )
        for name, metadata in archive_metadata.items()
    }
    updated_entry = replace(
        wikidata_entry,
        files=tuple(
            replacement_rows.get(row.path, row)
            for row in wikidata_entry.files
        ),
    )
    updated_lock = replace(
        original,
        source_catalog_sha256=source_lock_module.reviewed_source_catalog_sha256(),
        sources=tuple(
            updated_entry if entry.source_id == "wikidata5m" else entry
            for entry in original.sources
        ),
    )
    source_lock_path = tmp_path / "source-lock.json"
    source_lock_path.write_bytes(updated_lock.to_bytes())
    return _AuthorityFixture(
        source_lock_path=source_lock_path,
        source_root=source_root,
    )


def _archive_authority_from_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
    entries: dict[str, list[_TarEntry]],
) -> _AuthorityFixture:
    return _install_archive_authority(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {
            name: _tar_bytes(archive_entries)
            for name, archive_entries in entries.items()
        },
    )


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
class LookalikeWikidataSource:
    wrapped: object
    lane_id: LaneId = "wikidata_graph"
    finite: bool = False

    def __getattr__(self, name: str):
        return getattr(self.wrapped, name)

    def iter_drafts(
        self,
        source_root: Path,
        target_lengths: Sequence[int],
    ) -> Iterator[CatalogDraft]:
        return self.wrapped.iter_drafts(source_root, target_lengths)

    def iter_training_edge_keys(self, source_root: Path) -> Iterator[str]:
        return self.wrapped.iter_training_edge_keys(source_root)


class FlippingMapping(Mapping[LaneId, object]):
    def __init__(self, values: Mapping[LaneId, object]) -> None:
        self._values = dict(values)
        self._iteration = 0

    def __getitem__(self, key: LaneId) -> object:
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
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    authority = _install_archive_authority(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        _base_archive_payloads(),
    )
    lock = SourceLock.from_dict(
        json.loads(authority.source_lock_path.read_bytes())
    )
    root = stage_source_lock(
        lock,
        authority.source_root,
        tmp_path / "canonical",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    view = wikidata_source_module.build_wikidata_derived_view(
        authority.source_lock_path,
        root,
        tmp_path / "derived",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    return SimpleNamespace(
        authority=authority,
        lock=lock,
        root=root,
        source_root=root,
        view=view,
    )


@pytest.fixture
def fixture_lane_sources(staged_fixture_sources):
    def make(
        *,
        order: str = "forward",
        mutation_lane: LaneId | None = None,
        mutation: str | None = None,
        records: int | None = None,
        require_compact_lengths: bool = False,
    ) -> dict[LaneId, object]:
        lanes = LANE_ORDER if order == "forward" else tuple(reversed(LANE_ORDER))
        sources: dict[LaneId, object] = {}
        for lane_id in lanes:
            if lane_id == "wikidata_graph":
                if mutation_lane == lane_id:
                    raise ValueError(
                        "production Wikidata source does not accept fixture mutations"
                    )
                sources[lane_id] = catalog_module.WikidataGraphCatalogSource(
                    staged_fixture_sources.view
                )
                continue
            sources[lane_id] = FixtureLane(
                lane_id=lane_id,
                lock=staged_fixture_sources.lock,
                order=order,
                records=records if lane_id == mutation_lane else None,
                mutation=mutation if lane_id == mutation_lane else None,
                require_compact_lengths=require_compact_lengths,
            )
        return sources

    return make


def _build(
    geometry: BuildGeometry,
    staged_fixture_sources,
    lane_sources: Mapping[LaneId, object],
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


@pytest.fixture
def production_wikidata_catalog_source(
    staged_fixture_sources,
):
    return SimpleNamespace(
        authority=staged_fixture_sources.authority,
        lock=staged_fixture_sources.lock,
        source_root=staged_fixture_sources.root,
        view=staged_fixture_sources.view,
    )


def _production_geometry(wikidata_records: int) -> BuildGeometry:
    lane_quotas = tuple(
        (
            lane_id,
            wikidata_records if lane_id == "wikidata_graph" else 1,
        )
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


def _production_lane_sources(production_wikidata_catalog_source):
    sources: dict[LaneId, object] = {
        lane_id: FixtureLane(
            lane_id=lane_id,
            lock=production_wikidata_catalog_source.lock,
        )
        for lane_id in LANE_ORDER
        if lane_id != "wikidata_graph"
    }
    sources["wikidata_graph"] = catalog_module.WikidataGraphCatalogSource(
        production_wikidata_catalog_source.view
    )
    return sources


def test_production_adapter_binds_view_archive_member_split_row_and_edge(
    production_wikidata_catalog_source,
):
    verified_view = production_wikidata_catalog_source.view
    caller_constructed = wikidata_source_module.WikidataDerivedView(
        root=verified_view.root,
        receipt_sha256=verified_view.receipt_sha256,
        receipt=verified_view.receipt,
    )
    with pytest.raises(ValueError, match="verified"):
        catalog_module.WikidataGraphCatalogSource(caller_constructed)

    source = catalog_module.WikidataGraphCatalogSource(verified_view)
    drafts = tuple(
        source.iter_drafts(
            production_wikidata_catalog_source.source_root,
            (1, 1),
        )
    )
    archive_hashes = {
        record.path: record.sha256
        for record in verified_view.receipt.archives
    }
    assert [
        (
            draft.source_byte_sha256,
            draft.source_locator,
        )
        for draft in drafts
    ] == [
        (
            archive_hashes["wikidata5m_inductive.tar.gz"],
            (
                ("member", "wikidata5m_inductive_train.txt"),
                ("path", "wikidata5m_inductive.tar.gz"),
                ("row", 1),
                ("split", "train"),
                ("training_edge_key", "Q1\tP1\tQ2"),
                ("training_split", "inductive_train"),
                ("wikidata_view_sha256", verified_view.receipt_sha256),
            ),
        ),
        (
            archive_hashes["wikidata5m_transductive.tar.gz"],
            (
                ("member", "wikidata5m_transductive_train.txt"),
                ("path", "wikidata5m_transductive.tar.gz"),
                ("row", 1),
                ("split", "train"),
                ("training_edge_key", "Q3\tP2\tQ4"),
                ("training_split", "transductive_train"),
                ("wikidata_view_sha256", verified_view.receipt_sha256),
            ),
        ),
    ]


def test_catalog_receipt_binds_verified_wikidata_view_sha256(
    tmp_path: Path,
    production_wikidata_catalog_source,
):
    catalog = build_input_catalog(
        _production_geometry(3),
        production_wikidata_catalog_source.lock,
        production_wikidata_catalog_source.source_root,
        _production_lane_sources(production_wikidata_catalog_source),
        tmp_path / "catalog",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    index = json.loads(catalog.to_bytes())

    assert (
        catalog.wikidata_view_sha256
        == production_wikidata_catalog_source.view.receipt_sha256
    )
    assert (
        index["wikidata_view_sha256"]
        == production_wikidata_catalog_source.view.receipt_sha256
    )


def test_distinct_edges_appear_once_before_first_revisit(
    production_wikidata_catalog_source,
):
    source = catalog_module.WikidataGraphCatalogSource(
        production_wikidata_catalog_source.view
    )
    drafts = tuple(
        source.iter_drafts(
            production_wikidata_catalog_source.source_root,
            (1, 1, 1, 1, 1),
        )
    )
    first_revisit = next(
        index
        for index, draft in enumerate(drafts)
        if "graph-revisit" in draft.semantic_flags
    )
    before_revisit = tuple(
        dict(draft.source_locator)["training_edge_key"]
        for draft in drafts[:first_revisit]
    )

    assert before_revisit == tuple(
        source.iter_training_edge_keys(
            production_wikidata_catalog_source.source_root
        )
    )
    assert len(before_revisit) == source.training_edge_count
    assert len(set(before_revisit)) == source.training_edge_count
    assert all(
        draft.semantic_flags == ("graph-training-edge",)
        for draft in drafts[:first_revisit]
    )
    assert all(
        draft.semantic_flags == ("graph-revisit",)
        for draft in drafts[first_revisit:]
    )
    archive_hashes = {
        record.path: record.sha256
        for record in production_wikidata_catalog_source.view.receipt.archives
    }
    assert [
        (draft.source_byte_sha256, draft.source_locator)
        for draft in drafts[first_revisit:]
    ] == [
        (
            archive_hashes["wikidata5m_inductive.tar.gz"],
            (
                ("member", "wikidata5m_inductive_train.txt"),
                ("path", "wikidata5m_inductive.tar.gz"),
                ("row", 1),
                ("split", "train"),
                ("training_edge_key", "Q1\tP1\tQ2"),
                ("training_split", "inductive_train"),
                (
                    "wikidata_view_sha256",
                    production_wikidata_catalog_source.view.receipt_sha256,
                ),
            ),
        ),
        (
            archive_hashes["wikidata5m_transductive.tar.gz"],
            (
                ("member", "wikidata5m_transductive_train.txt"),
                ("path", "wikidata5m_transductive.tar.gz"),
                ("row", 1),
                ("split", "train"),
                ("training_edge_key", "Q3\tP2\tQ4"),
                ("training_split", "transductive_train"),
                (
                    "wikidata_view_sha256",
                    production_wikidata_catalog_source.view.receipt_sha256,
                ),
            ),
        ),
        (
            archive_hashes["wikidata5m_inductive.tar.gz"],
            (
                ("member", "wikidata5m_inductive_train.txt"),
                ("path", "wikidata5m_inductive.tar.gz"),
                ("row", 1),
                ("split", "train"),
                ("training_edge_key", "Q1\tP1\tQ2"),
                ("training_split", "inductive_train"),
                (
                    "wikidata_view_sha256",
                    production_wikidata_catalog_source.view.receipt_sha256,
                ),
            ),
        ),
    ]


def test_edge_count_above_available_records_fails_before_draft_output(
    tmp_path: Path,
    production_wikidata_catalog_source,
):
    sources = _production_lane_sources(production_wikidata_catalog_source)
    output_root = tmp_path / "capacity-failure"

    with pytest.raises(
        ValueError,
        match="Wikidata distinct edges exceed allocated records",
    ):
        build_input_catalog(
            _production_geometry(1),
            production_wikidata_catalog_source.lock,
            production_wikidata_catalog_source.source_root,
            sources,
            output_root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )

    assert not output_root.exists()
    assert all(
        source.started == 0
        for lane_id, source in sources.items()
        if lane_id != "wikidata_graph"
    )


def test_catalog_rejects_duck_typed_wikidata_authority(
    tmp_path: Path,
    production_wikidata_catalog_source,
):
    sources = _production_lane_sources(production_wikidata_catalog_source)
    sources["wikidata_graph"] = LookalikeWikidataSource(
        sources["wikidata_graph"]
    )
    output_root = tmp_path / "lookalike"

    with pytest.raises(ValueError, match="authenticated production"):
        build_input_catalog(
            _production_geometry(3),
            production_wikidata_catalog_source.lock,
            production_wikidata_catalog_source.source_root,
            sources,
            output_root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )

    assert not output_root.exists()


def test_catalog_reopens_view_and_rejects_on_disk_drift_before_staging(
    tmp_path: Path,
    production_wikidata_catalog_source,
):
    sources = _production_lane_sources(production_wikidata_catalog_source)
    stream = (
        production_wikidata_catalog_source.view.root
        / "streams"
        / "distinct-edges.tsv"
    )
    payload = stream.read_bytes()
    mode = stream.stat().st_mode & 0o777
    stream.chmod(0o600)
    stream.write_bytes(payload.replace(b"\tQ1\t", b"\tQ2\t", 1))
    stream.chmod(mode)
    output_root = tmp_path / "drifted-view"

    with pytest.raises(ValueError, match="identity drift|ordering drift"):
        build_input_catalog(
            _production_geometry(3),
            production_wikidata_catalog_source.lock,
            production_wikidata_catalog_source.source_root,
            sources,
            output_root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )

    assert not output_root.exists()
    assert not (tmp_path / _QUARANTINE_DIRECTORY).exists()
    assert all(
        source.started == 0
        for lane_id, source in sources.items()
        if lane_id != "wikidata_graph"
    )


def test_end_to_end_catalog_is_byte_identical_across_rebuilds(
    tmp_path: Path,
    production_wikidata_catalog_source,
):
    second_view = wikidata_source_module.build_wikidata_derived_view(
        production_wikidata_catalog_source.authority.source_lock_path,
        production_wikidata_catalog_source.source_root,
        tmp_path / "derived-second",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    first_sources = _production_lane_sources(production_wikidata_catalog_source)
    second_environment = SimpleNamespace(
        lock=production_wikidata_catalog_source.lock,
        source_root=production_wikidata_catalog_source.source_root,
        view=second_view,
    )
    second_sources = _production_lane_sources(second_environment)
    geometry = _production_geometry(5)

    first = build_input_catalog(
        geometry,
        production_wikidata_catalog_source.lock,
        production_wikidata_catalog_source.source_root,
        first_sources,
        tmp_path / "catalog-first",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    second = build_input_catalog(
        geometry,
        production_wikidata_catalog_source.lock,
        production_wikidata_catalog_source.source_root,
        second_sources,
        tmp_path / "catalog-second",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )

    assert first.sha256 == second.sha256
    assert first.to_bytes() == second.to_bytes()
    assert first.records_path.read_bytes() == second.records_path.read_bytes()


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
    assert all(
        source.started == 0
        for lane_id, source in sources.items()
        if lane_id != "wikidata_graph"
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
    expected_edges = tuple(
        sources["wikidata_graph"].iter_training_edge_keys(
            staged_fixture_sources.root
        )
    )
    assert tuple(
        dict(row.source_locator)["training_edge_key"]
        for row in rows[:first_revisit]
    ) == expected_edges
    assert len(set(expected_edges)) == len(expected_edges)
    assert tuple(row.semantic_flags for row in rows[:first_revisit]) == (
        ("graph-training-edge",),
    ) * len(expected_edges)
    assert tuple(row.semantic_flags for row in rows[first_revisit:]) == (
        ("graph-revisit",),
    ) * (len(rows) - first_revisit)
    assert not any(
        dict(row.source_locator)["training_edge_key"] not in expected_edges
        for row in rows[first_revisit:]
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
    staged_fixture_sources,
):
    relative = "evaluation/train/test.json"
    payload = b"must remain sealed"
    entry = next(
        row
        for row in staged_fixture_sources.lock.sources
        if row.source_id == "clrs_text"
    )
    materialized = (
        staged_fixture_sources.authority.source_root / entry.materialized_path
    )
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
        staged_fixture_sources.lock,
        sources=tuple(
            changed_entry if row.source_id == "clrs_text" else row
            for row in staged_fixture_sources.lock.sources
        ),
    )
    source_root = stage_source_lock(
        changed_lock,
        staged_fixture_sources.authority.source_root,
        tmp_path / "reserved-canonical",
        expected_generator_commit=changed_lock.generator_commit,
    )
    source_lock_path = tmp_path / "reserved-source-lock.json"
    source_lock_path.write_bytes(changed_lock.to_bytes())
    view = wikidata_source_module.build_wikidata_derived_view(
        source_lock_path,
        source_root,
        tmp_path / "reserved-derived",
        expected_generator_commit=changed_lock.generator_commit,
    )
    sources: dict[LaneId, object] = {
        lane_id: FixtureLane(
            lane_id=lane_id,
            lock=changed_lock,
            path_override=relative if lane_id == "relational_refinement" else None,
        )
        for lane_id in LANE_ORDER
        if lane_id != "wikidata_graph"
    }
    sources["wikidata_graph"] = catalog_module.WikidataGraphCatalogSource(view)
    staged = SimpleNamespace(lock=changed_lock, root=source_root, view=view)
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


def _linear_retention_catalog_variant(
    workspace: Path,
) -> tuple[ModuleType, Path, str]:
    catalog_path = Path(cast(str, catalog_module.__file__)).resolve()
    source = catalog_path.read_text(encoding="utf-8")
    class_marker = "\n\nclass WikidataGraphCatalogSource:\n"
    loop_marker = """\
        for edge_index, triple in enumerate(
            iter_distinct_training_edges(authority.view)
        ):
            emitted += 1
            yield self._draft(
"""
    assert source.count(class_marker) == 1
    assert source.count(loop_marker) == 1
    source = source.replace(
        class_marker,
        (
            "\n\n_LINEAR_RETENTION_FOR_MEMORY_TEST: list[bytearray] = []"
            f"{class_marker}"
        ),
    )
    source = source.replace(
        loop_marker,
        loop_marker.replace(
            "            emitted += 1\n",
            (
                "            emitted += 1\n"
                "            _LINEAR_RETENTION_FOR_MEMORY_TEST.append("
                "bytearray(1))\n"
            ),
        ),
    )
    variant_path = workspace / "catalog_linear_retention_variant.py"
    variant_path.write_text(source, encoding="utf-8")
    module_name = (
        f"_memorysplit_catalog_linear_retention_{os.getpid()}_"
        f"{workspace.name}"
    )
    spec = importlib.util.spec_from_file_location(module_name, variant_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module, variant_path, module_name


def _measure_adapter_attributed_live_allocations(
    workspace: Path,
    edge_count: int,
    *,
    linear_retention: bool,
) -> dict[str, int]:
    monkeypatch = pytest.MonkeyPatch()
    variant_module_name: str | None = None
    try:
        workspace.mkdir()
        contract_root = fixed_contract_environment.__wrapped__(
            workspace,
            monkeypatch,
        )
        resolver = fake_public_resolver.__wrapped__(contract_root)
        recipe = full_recipe.__wrapped__()
        fixture = fixture_source_lock.__wrapped__(
            workspace,
            resolver,
            recipe,
        )
        entries = _base_archive_entries()
        training_rows = b"".join(
            (
                f"Q{1_000_000 + index}\tP1\t"
                f"Q{2_000_000 + index}\n"
            ).encode("ascii")
            for index in range(edge_count - 1)
        )
        _replace_archive_member(
            entries,
            "wikidata5m_inductive.tar.gz",
            "wikidata5m_inductive_train.txt",
            training_rows,
        )
        authority = _archive_authority_from_entries(
            workspace,
            monkeypatch,
            fixture,
            entries,
        )
        lock = SourceLock.from_dict(
            json.loads(authority.source_lock_path.read_bytes())
        )
        source_root = stage_source_lock(
            lock,
            authority.source_root,
            workspace / "canonical",
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )
        view = wikidata_source_module.build_wikidata_derived_view(
            authority.source_lock_path,
            source_root,
            workspace / "derived",
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )
        assert view.receipt.distinct_edges == edge_count
        del entries, training_rows, authority, fixture, lock, resolver, recipe

        selected_catalog = cast(ModuleType, catalog_module)
        selected_path = Path(cast(str, selected_catalog.__file__)).resolve()
        if linear_retention:
            (
                selected_catalog,
                selected_path,
                variant_module_name,
            ) = _linear_retention_catalog_variant(workspace)

        tracemalloc.start(1)
        try:
            source = selected_catalog.WikidataGraphCatalogSource(view)
            lengths = selected_catalog._BalancedTargetLengths.create(
                edge_count + 100,
                1,
            )
            checkpoints = {
                1,
                edge_count // 2,
                edge_count,
                edge_count + 100,
            }
            peak_live_blocks = 0
            peak_live_bytes = 0
            drafts = 0
            for drafts, _draft in enumerate(
                source.iter_drafts(
                    source_root,
                    lengths,
                ),
                start=1,
            ):
                if drafts not in checkpoints:
                    continue
                gc.collect()
                snapshot = tracemalloc.take_snapshot().filter_traces(
                    (
                        tracemalloc.Filter(
                            True,
                            str(selected_path),
                            all_frames=False,
                        ),
                    ),
                )
                peak_live_blocks = max(
                    peak_live_blocks,
                    len(snapshot.traces),
                )
                peak_live_bytes = max(
                    peak_live_bytes,
                    sum(trace.size for trace in snapshot.traces),
                )
        finally:
            tracemalloc.stop()

        return {
            "drafts": drafts,
            "edge_count": edge_count,
            "linear_retention": int(linear_retention),
            "peak_live_blocks": peak_live_blocks,
            "peak_live_bytes": peak_live_bytes,
        }
    finally:
        if variant_module_name is not None:
            sys.modules.pop(variant_module_name, None)
        monkeypatch.undo()


def _run_attributed_allocation_probe(
    workspace: Path,
    edge_count: int,
    *,
    linear_retention: bool,
) -> dict[str, int]:
    test_module = Path(__file__).resolve()
    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(test_module.parent)!r})\n"
        "from test_reasoning_v2_catalog import "
        "_measure_adapter_attributed_live_allocations\n"
        "result = _measure_adapter_attributed_live_allocations(\n"
        "    Path(sys.argv[1]), int(sys.argv[2]),\n"
        "    linear_retention=sys.argv[3] == 'linear',\n"
        ")\n"
        "print(json.dumps(result, sort_keys=True))\n"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(workspace),
            str(edge_count),
            "linear" if linear_retention else "streaming",
        ],
        cwd=test_module.parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(dict[str, int], json.loads(completed.stdout))


def _attributed_allocation_growth(
    measurements: tuple[dict[str, int], dict[str, int]],
) -> dict[str, int | Fraction]:
    low, high = measurements
    edge_delta = high["edge_count"] - low["edge_count"]
    assert edge_delta > 0
    byte_growth = high["peak_live_bytes"] - low["peak_live_bytes"]
    block_growth = high["peak_live_blocks"] - low["peak_live_blocks"]
    return {
        "edge_delta": edge_delta,
        "live_block_growth": block_growth,
        "live_byte_growth": byte_growth,
        "marginal_live_bytes_per_edge": Fraction(
            max(0, byte_growth),
            edge_delta,
        ),
    }


def _assert_zero_marginal_attributed_retention(
    measurements: tuple[dict[str, int], dict[str, int]],
) -> dict[str, int | Fraction]:
    growth = _attributed_allocation_growth(measurements)
    assert growth["live_byte_growth"] <= 0, (
        "adapter-attributed live bytes grew with edge cardinality: "
        f"{growth}"
    )
    assert growth["live_block_growth"] <= 0, (
        "adapter-attributed live allocation blocks grew with edge "
        f"cardinality: {growth}"
    )
    return growth


def test_attributed_memory_regression_kills_linear_retention_variant(
    tmp_path: Path,
):
    cardinalities = (1_000, 20_000)
    streaming = tuple(
        _run_attributed_allocation_probe(
            tmp_path / f"streaming-{edge_count}",
            edge_count,
            linear_retention=False,
        )
        for edge_count in cardinalities
    )
    linear = tuple(
        _run_attributed_allocation_probe(
            tmp_path / f"linear-{edge_count}",
            edge_count,
            linear_retention=True,
        )
        for edge_count in cardinalities
    )
    assert tuple(row["drafts"] for row in streaming) == tuple(
        edge_count + 100 for edge_count in cardinalities
    )
    assert tuple(row["drafts"] for row in linear) == tuple(
        edge_count + 100 for edge_count in cardinalities
    )

    streaming_growth = _assert_zero_marginal_attributed_retention(
        cast(tuple[dict[str, int], dict[str, int]], streaming)
    )
    linear_measurements = cast(
        tuple[dict[str, int], dict[str, int]],
        linear,
    )
    linear_growth = _attributed_allocation_growth(linear_measurements)
    with pytest.raises(
        AssertionError,
        match="adapter-attributed live bytes grew",
    ):
        _assert_zero_marginal_attributed_retention(linear_measurements)

    assert streaming_growth["marginal_live_bytes_per_edge"] == 0
    assert linear_growth["live_block_growth"] >= cardinalities[1] - cardinalities[0]
    assert linear_growth["marginal_live_bytes_per_edge"] >= 16


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
    assert all(
        source.started == 1
        for lane_id, source in sources.items()
        if lane_id != "wikidata_graph"
    )
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
