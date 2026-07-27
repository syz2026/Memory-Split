import hashlib
import heapq
import json
import os
import platform
import stat
import subprocess
import sys
import tarfile
import warnings
import zipfile
import zlib
from dataclasses import dataclass, fields, replace
from pathlib import Path

import pytest

import corpusgen.wikidata5m_hashmap as hashmap_module
from corpusgen.relation_schema import RelationStats
from corpusgen.wikidata5m import WikidataLock
from corpusgen.wikidata5m_hashmap import (
    ARCHIVE_ROOT,
    CC0_PATH,
    HASHMAP_LOCK_PATH,
    ArchiveReport,
    HashmapBuildError,
    RelationSelection,
    SelectedRelation,
    address_sort_key,
    build_hashmap_dataset,
    build_wikidata5m_hashmap,
    canonical_json_bytes,
    canonical_address,
    first_display_alias,
    iter_strict_alias_rows,
    normalize_display_alias,
    render_package_files,
    relation_quota,
    select_hashmap_relations,
    verify_wikidata5m_hashmap,
    write_deterministic_zip,
)
from scripts.fetch_wikidata5m import build_download_command


HASHMAP_LOCK = HASHMAP_LOCK_PATH

EXPECTED_HASHMAP_LOCK = {
    "repo_id": "intfloat/wikidata5m",
    "repo_type": "dataset",
    "revision": "6b2b09672129e280c0c9da97ab58154e9d535e6b",
    "files": {
        "wikidata5m_alias.tar.gz": {
            "bytes": 197449751,
            "sha256": "0330f580c9f7a57cbad949ac380835fdd2a2e14d96cc0f13fc435401d6b463a8",
        },
        "wikidata5m_transductive.tar.gz": {
            "bytes": 168258214,
            "sha256": "383160990b41c0905fc03f4a8afbb9b12be1ca3591e026bde6cdc94a59542597",
        },
    },
}


def test_hashmap_lock_is_exactly_two_archives():
    value = json.loads(HASHMAP_LOCK.read_text(encoding="utf-8"))
    assert value == EXPECTED_HASHMAP_LOCK
    assert sum(item["bytes"] for item in value["files"].values()) == 365_707_965


def test_hashmap_download_command_requests_only_locked_archives():
    assert build_download_command(Path("/data"), HASHMAP_LOCK) == [
        "hf",
        "download",
        "intfloat/wikidata5m",
        "--repo-type",
        "dataset",
        "--revision",
        "6b2b09672129e280c0c9da97ab58154e9d535e6b",
        "--include",
        "wikidata5m_alias.tar.gz",
        "--include",
        "wikidata5m_transductive.tar.gz",
        "--local-dir",
        "/data/wikidata5m",
    ]


def test_display_alias_normalization_preserves_original_case():
    assert normalize_display_alias("  Douglas\t  Adams  ") == "Douglas Adams"
    assert first_display_alias((" Douglas   Adams ", "douglas adams")) == (
        "Douglas Adams"
    )


@pytest.mark.parametrize(
    "line",
    [
        "Q1\n",
        "Q1\t\n",
        "Q1\t \tvalid\n",
        "not-a-qid\tname\n",
    ],
)
def test_entity_alias_rows_remain_strict(tmp_path, line):
    path = tmp_path / "wikidata5m_entity.txt"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_strict_alias_rows(path, "Q"))


def test_frozen_quotas_sum_to_three_thousand():
    assert [relation_quota(rank) for rank in range(1, 33)] == (
        [94] * 24 + [93] * 8
    )
    assert sum(relation_quota(rank) for rank in range(1, 33)) == 3_000


def test_cc0_text_is_complete_lf_terminated_utf8():
    content = CC0_PATH.read_bytes()
    assert content.startswith(b"Creative Commons Legal Code\n")
    assert b"CC0 1.0 Universal" in content
    assert b"4. Limitations and Disclaimers." in content
    assert b"\r" not in content
    assert content.endswith(b"\n")


def _write_transductive_train(path, rows):
    path.write_text("".join(rows), encoding="utf-8")


def _write_relation_aliases(path, entries):
    lines = []
    for relation_id in sorted(entries, key=lambda value: int(value[1:])):
        aliases = entries[relation_id]
        lines.append(f"{relation_id}\t" + "\t".join(aliases) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def _valid_stats(count, *, start=4):
    return [
        RelationStats(
            f"P{index}",
            5_000,
            4_750,
            4_000,
            10_000,
            (f"relation {index}",),
        )
        for index in range(start, start + count)
    ]


def test_iter_strict_alias_rows_streams_duplicate_ids_without_memory_tracking(
    tmp_path,
):
    path = tmp_path / "wikidata5m_entity.txt"
    path.write_text("Q1\tFirst\nQ1\tSecond\n", encoding="utf-8")
    assert list(iter_strict_alias_rows(path, "Q")) == [
        (1, 1, "First"),
        (2, 1, "Second"),
    ]


def test_canonical_address_and_sort_key_use_frozen_bytes():
    assert canonical_address(42, "P106") == "Q42\tP106"
    encoded = canonical_address(42, "P106").encode("utf-8")
    assert address_sort_key(42, "P106") == (
        hashlib.sha256(encoded).digest(),
        42,
        106,
    )


def test_hashmap_heap_uses_an_orderable_deterministic_tie_breaker():
    rank = (b"x" * 32, 7, 11)
    lower = hashmap_module._WorstFirst(rank=rank, subject=1)
    higher = hashmap_module._WorstFirst(rank=rank, subject=2)
    heap = [lower]

    heapq.heappush(heap, higher)

    assert higher < lower
    assert heapq.heappop(heap) is higher


def test_select_hashmap_relations_use_raw_relation_aliases_for_labels(
    monkeypatch,
    tmp_path,
):
    stats = [
        RelationStats("P1", 6_000, 5_700, 5_000, 10_000, ("unique",)),
        *_valid_stats(32, start=2),
    ]
    monkeypatch.setattr(
        hashmap_module,
        "compute_relation_stats",
        lambda train_path, relation_aliases, work_root=None: tuple(stats),
    )
    train = tmp_path / "wikidata5m_transductive_train.txt"
    aliases_path = tmp_path / "wikidata5m_relation.txt"
    train.write_text("Q1\tP1\tQ2\n", encoding="utf-8")
    _write_relation_aliases(
        aliases_path,
        {
            "P1": ("Display One", "unique"),
            "P2": ("display one",),
            **{
                f"P{index}": (f"relation {index}",)
                for index in range(3, 34)
            },
        },
    )
    work_root = tmp_path / "work"
    work_root.mkdir()

    selection = select_hashmap_relations(
        train,
        aliases_path,
        work_root=work_root,
    )

    p1 = next(item for item in selection.relations if item.relation_id == "P1")
    assert p1.label == "Display One"


def test_select_hashmap_relations_filter_counts_reconcile(monkeypatch, tmp_path):
    stats = [
        RelationStats("P1", 5_000, 4_750, 4_000, 10_000, ()),
        RelationStats("P2", 4_999, 4_999, 4_000, 10_000, ("low support",)),
        RelationStats("P3", 5_000, 4_749, 4_000, 10_000, ("low functionality",)),
        *_valid_stats(35),
    ]
    monkeypatch.setattr(
        hashmap_module,
        "compute_relation_stats",
        lambda train_path, relation_aliases, work_root=None: tuple(stats),
    )
    train = tmp_path / "wikidata5m_transductive_train.txt"
    aliases_path = tmp_path / "wikidata5m_relation.txt"
    train.write_text("Q1\tP4\tQ2\n", encoding="utf-8")
    _write_relation_aliases(
        aliases_path,
        {item.relation_id: item.aliases for item in stats if item.aliases},
    )
    work_root = tmp_path / "work"
    work_root.mkdir()

    selection = select_hashmap_relations(
        train,
        aliases_path,
        work_root=work_root,
    )

    assert set(selection.filter_counts) == {
        "missing_relation_alias",
        "below_min_support",
        "below_min_functionality",
        "survived_not_selected",
        "selected",
    }
    assert selection.filter_counts["missing_relation_alias"] == 1
    assert selection.filter_counts["below_min_support"] == 1
    assert selection.filter_counts["below_min_functionality"] == 1
    assert selection.filter_counts["survived_not_selected"] == 3
    assert selection.filter_counts["selected"] == 32
    assert sum(selection.filter_counts.values()) == len(stats)


def test_select_hashmap_relations_raises_when_fewer_than_thirty_two_survive(
    monkeypatch,
    tmp_path,
):
    stats = _valid_stats(31)
    monkeypatch.setattr(
        hashmap_module,
        "compute_relation_stats",
        lambda train_path, relation_aliases, work_root=None: tuple(stats),
    )
    train = tmp_path / "wikidata5m_transductive_train.txt"
    aliases_path = tmp_path / "wikidata5m_relation.txt"
    train.write_text("Q1\tP4\tQ2\n", encoding="utf-8")
    _write_relation_aliases(
        aliases_path,
        {item.relation_id: item.aliases for item in stats},
    )
    work_root = tmp_path / "work"
    work_root.mkdir()

    with pytest.raises(HashmapBuildError, match="fewer than 32"):
        select_hashmap_relations(train, aliases_path, work_root=work_root)


EXPECTED_COUNTS = {
    "source_triples": 3329,
    "selected_relation_triples": 3328,
    "unselected_relation_triples": 1,
    "distinct_selected_edges": 3296,
    "duplicate_selected_edges": 32,
    "selected_grouped_keys": 3264,
    "missing_subject_alias_keys": 32,
    "missing_object_alias_keys": 32,
    "eligible_keys": 3200,
    "eligible_edges": 3232,
    "unsampled_eligible_keys": 200,
    "emitted_keys": 3000,
    "emitted_edges": 3032,
}


@dataclass
class _BalancedFixture:
    train: Path
    entities: Path
    selection: RelationSelection
    multivalue_addresses: tuple[str, ...]

    def remove_subject_aliases(self, relation_id: str, count: int) -> None:
        base = int(relation_id[1:]) * 10_000
        removed = {f"Q{base + index}" for index in range(1, count + 1)}
        kept = [
            line
            for line in self.entities.read_text(encoding="utf-8").splitlines(
                keepends=True
            )
            if line.split("\t", 1)[0] not in removed
        ]
        self.entities.write_text("".join(kept), encoding="utf-8")

    def append_duplicate_entity_alias(self) -> None:
        first_id = self.entities.read_text(encoding="utf-8").split("\t", 1)[0]
        with self.entities.open("a", encoding="utf-8") as stream:
            stream.write(f"{first_id}\tHomonym\n")


def _write_balanced_fixture(root, reverse_triples=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train = root / "wikidata5m_transductive_train.txt"
    entities = root / "wikidata5m_entity.txt"

    triple_rows: list[str] = []
    entity_rows: list[str] = []
    multivalue_addresses: list[str] = []
    for relation_index in range(1, 33):
        relation_id = f"P{relation_index}"
        base = relation_index * 10_000
        anchor = min(
            (base + index for index in range(1, 101)),
            key=lambda subject: address_sort_key(subject, relation_id),
        )
        for index in range(1, 103):
            triple_rows.append(
                f"Q{base + index}\t{relation_id}\tQ{1_000_000 + base + index}\n"
            )
        triple_rows.append(
            f"Q{anchor}\t{relation_id}\tQ{2_000_000 + anchor}\n"
        )
        triple_rows.append(
            f"Q{anchor}\t{relation_id}\tQ{1_000_000 + anchor}\n"
        )
        multivalue_addresses.append(canonical_address(anchor, relation_id))
        for index in range(1, 103):
            if index != 101:
                entity_rows.append(f"Q{base + index}\tHomonym\n")
        for index in range(1, 103):
            if index != 102:
                entity_rows.append(f"Q{1_000_000 + base + index}\tHomonym\n")
        entity_rows.append(f"Q{2_000_000 + anchor}\tHomonym\n")
    triple_rows.append("Q10001\tP999\tQ1010001\n")
    if reverse_triples:
        triple_rows.reverse()

    _write_transductive_train(train, triple_rows)
    entities.write_text("".join(entity_rows), encoding="utf-8")
    selection = RelationSelection(
        relations=tuple(
            SelectedRelation(
                rank=rank,
                relation_id=f"P{rank}",
                label="shared relation",
                support=104,
                distinct_subjects=102,
                distinct_objects=103,
                entity_count=6_560,
                quota=relation_quota(rank),
            )
            for rank in range(1, 33)
        ),
        filter_counts={
            "missing_relation_alias": 0,
            "below_min_support": 0,
            "below_min_functionality": 0,
            "survived_not_selected": 0,
            "selected": 32,
        },
    )
    return _BalancedFixture(
        train=train,
        entities=entities,
        selection=selection,
        multivalue_addresses=tuple(multivalue_addresses),
    )


def test_builder_is_balanced_array_valued_and_order_independent(tmp_path):
    first_fixture = _write_balanced_fixture(tmp_path / "first")
    second_fixture = _write_balanced_fixture(
        tmp_path / "second",
        reverse_triples=True,
    )
    first_work = tmp_path / "first-work"
    second_work = tmp_path / "second-work"
    first_work.mkdir()
    second_work.mkdir()

    first = build_hashmap_dataset(
        first_fixture.train,
        first_fixture.entities,
        first_fixture.selection,
        work_root=first_work,
    )
    second = build_hashmap_dataset(
        second_fixture.train,
        second_fixture.entities,
        second_fixture.selection,
        work_root=second_work,
    )

    assert first == second
    assert first.counts.to_dict() == EXPECTED_COUNTS
    assert [item.emitted_keys for item in first.relations] == (
        [94] * 24 + [93] * 8
    )
    assert len({record.canonical_address for record in first.records}) == 3000
    assert len({record.display_key for record in first.records}) == 3000
    assert sum(len(record.values) for record in first.records) == 3032
    assert list(first_work.iterdir()) == []
    assert list(second_work.iterdir()) == []


def test_builder_fails_when_one_relation_cannot_meet_quota(tmp_path):
    fixture = _write_balanced_fixture(tmp_path / "fixture")
    fixture.remove_subject_aliases("P32", count=8)
    work_root = tmp_path / "work"
    work_root.mkdir()

    with pytest.raises(
        HashmapBuildError,
        match=r"P32 has 92 eligible keys but requires 93",
    ):
        build_hashmap_dataset(
            fixture.train,
            fixture.entities,
            fixture.selection,
            work_root=work_root,
        )

    assert list(work_root.iterdir()) == []


def test_lowest_ranked_multivalue_addresses_preserve_all_targets(tmp_path):
    fixture = _write_balanced_fixture(tmp_path / "fixture")
    work_root = tmp_path / "work"
    work_root.mkdir()
    dataset = build_hashmap_dataset(
        fixture.train,
        fixture.entities,
        fixture.selection,
        work_root=work_root,
    )
    by_address = {
        record.canonical_address: record
        for record in dataset.records
    }
    for address in fixture.multivalue_addresses:
        values = by_address[address].values
        assert len(values) == 2
        numeric_ids = [int(value.id[1:]) for value in values]
        assert numeric_ids == sorted(set(numeric_ids))


def test_builder_rejects_duplicate_entity_alias_ids(tmp_path):
    fixture = _write_balanced_fixture(tmp_path / "fixture")
    fixture.append_duplicate_entity_alias()
    work_root = tmp_path / "work"
    work_root.mkdir()
    with pytest.raises(ValueError, match="duplicate canonical ID"):
        build_hashmap_dataset(
            fixture.train,
            fixture.entities,
            fixture.selection,
            work_root=work_root,
        )
    assert list(work_root.iterdir()) == []


PACKAGE_MEMBERS = {
    "CITATION.bib",
    "LICENSES/Wikidata-CC0-1.0.txt",
    "README.md",
    "SHA256SUMS",
    "build_manifest.json",
    "hashmap.json",
    "hashmap.jsonl",
    "records.jsonl",
    "relation_summary.json",
    "source/wikidata5m.lock.json",
}
REQUIRED_README_STATEMENTS = (
    "This dataset was built from the pinned third-party Wikidata5M "
    "derivative hosted by intfloat on Hugging Face. It is not an official "
    "Wikimedia Foundation dump.",
    "Wikidata5M aliases do not include language tags and do not designate a "
    "canonical label. Labels in this archive are deterministic display text "
    "only; QIDs and PIDs are authoritative.",
    "The packaged structured data is distributed under the Wikidata CC0 1.0 "
    "public-domain dedication. See "
    "https://www.wikidata.org/wiki/Wikidata:Licensing.",
)


@dataclass(frozen=True)
class _PackageFixture:
    dataset: object
    selection: RelationSelection
    lock: WikidataLock
    files: dict[str, bytes]


@pytest.fixture(scope="module")
def package_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("hashmap-package")
    fixture = _write_balanced_fixture(root / "fixture")
    selection = RelationSelection(
        relations=tuple(
            replace(
                item,
                support=5_000,
                distinct_subjects=4_750,
                distinct_objects=4_000,
                entity_count=10_000,
            )
            for item in fixture.selection.relations
        ),
        filter_counts=dict(fixture.selection.filter_counts),
    )
    work_root = root / "work"
    work_root.mkdir()
    dataset = build_hashmap_dataset(
        fixture.train,
        fixture.entities,
        selection,
        work_root=work_root,
    )
    lock = WikidataLock.from_path(HASHMAP_LOCK)
    rendered = render_package_files(
        dataset,
        selection,
        lock,
        CC0_PATH.read_text(encoding="utf-8"),
    )
    assert list(work_root.iterdir()) == []
    return _PackageFixture(dataset, selection, lock, rendered)


@pytest.fixture
def rendered_archive(tmp_path, package_fixture):
    path = tmp_path / "package.zip"
    write_deterministic_zip(path, package_fixture.files)
    return path


def _jsonl_rows(content):
    return [json.loads(line) for line in content.decode("utf-8").splitlines()]


def _refresh_sha256sums(files):
    files["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(files[name]).hexdigest()}  {name}\n"
        for name in sorted(set(files) - {"SHA256SUMS"})
    ).encode("utf-8")


def _replace_first_value_id(files, new_id, *, representations):
    mapping = json.loads(files["hashmap.json"])
    key = next(iter(mapping))
    if "json" in representations:
        mapping[key][0]["id"] = new_id
        files["hashmap.json"] = canonical_json_bytes(mapping)

    hashmap_rows = _jsonl_rows(files["hashmap.jsonl"])
    if "jsonl" in representations:
        row = next(item for item in hashmap_rows if item["key"] == key)
        row["values"][0]["id"] = new_id
        files["hashmap.jsonl"] = b"".join(
            canonical_json_bytes(item) for item in hashmap_rows
        )

    record_rows = _jsonl_rows(files["records.jsonl"])
    if "records" in representations:
        row = next(item for item in record_rows if item["display_key"] == key)
        row["values"][0]["id"] = new_id
        files["records.jsonl"] = b"".join(
            canonical_json_bytes(item) for item in record_rows
        )
    _refresh_sha256sums(files)


def _raw_zip_members(files):
    return [
        (
            f"{ARCHIVE_ROOT}/{name}",
            files[name],
            (1980, 1, 1, 0, 0, 0),
            0o644,
        )
        for name in sorted(files)
    ]


def _write_raw_zip(path, members):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w") as archive:
            for name, content, timestamp, mode in members:
                info = zipfile.ZipInfo(name, date_time=timestamp)
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | mode) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(
                    info,
                    content,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )


def _validate_structure(path, package_fixture, *, expected_dataset=True):
    return hashmap_module._validate_archive_structure(
        path,
        expected_dataset=(
            package_fixture.dataset if expected_dataset else None
        ),
        lock=package_fixture.lock,
        cc0_text=CC0_PATH.read_text(encoding="utf-8"),
    )


def test_render_package_files_has_exact_contract_and_documentation(
    package_fixture,
):
    files = package_fixture.files
    assert set(files) == PACKAGE_MEMBERS
    assert all(isinstance(content, bytes) for content in files.values())
    assert files["source/wikidata5m.lock.json"] == (
        package_fixture.lock.canonical_bytes()
    )
    assert files["LICENSES/Wikidata-CC0-1.0.txt"] == CC0_PATH.read_bytes()

    readme = files["README.md"].decode("utf-8")
    for statement in REQUIRED_README_STATEMENTS:
        assert statement in readme
    for name in PACKAGE_MEMBERS:
        assert f"`{name}`" in readme
    assert package_fixture.lock.revision in readme
    assert "24 relations receive 94 keys" in readme
    assert "8 relations receive 93 keys" in readme
    assert '"values": [' in readme
    assert "json.load" in readme
    assert "json.loads" in readme
    assert "reduced exact fraction" in readme
    assert "raw `support` and `distinct_subjects`" in readme
    assert (
        "`missing_relation_alias` counts relations with no usable unambiguous "
        "selector alias after canonicalization"
    ) in readme
    assert (
        "Byte identity is promised only for the exact Python and zlib "
        "versions recorded in `build_manifest.json`."
    ) in readme

    citation = files["CITATION.bib"].decode("utf-8")
    assert "@article{wang2021kepler," in citation
    assert "doi = {10.1162/tacl_a_00360}" in citation
    assert "@misc{intfloat2022wikidata5m," in citation
    assert package_fixture.lock.revision in citation


def test_manifest_records_exact_runtime_and_reduced_functionality_contract(
    package_fixture,
):
    manifest = json.loads(package_fixture.files["build_manifest.json"])
    assert manifest["algorithm_version"] == "wikidata5m-hashmap-v1"
    assert manifest["build_environment"] == {
        "python": platform.python_version(),
        "zlib_compile": zlib.ZLIB_VERSION,
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
    }
    assert manifest["source"] == EXPECTED_HASHMAP_LOCK
    assert manifest["input_split"] == "wikidata5m_transductive_train.txt"
    assert manifest["selector"] == {
        "minimum_support": 5_000,
        "minimum_functionality_numerator": 95,
        "minimum_functionality_denominator": 100,
        "relation_count": 32,
        "ordering": [
            "support_descending",
            "exact_functionality_descending",
            "numeric_pid_ascending",
        ],
    }
    assert manifest["relation_filter_counts"] == (
        package_fixture.selection.filter_counts
    )
    assert manifest["build_counts"] == package_fixture.dataset.counts.to_dict()
    assert manifest["output_counts"] == {
        "keys": 3_000,
        "edges": 3_032,
        "relations": 32,
    }

    summaries = json.loads(package_fixture.files["relation_summary.json"])
    assert len(summaries) == 32
    for summary in summaries:
        assert summary["support"] == 5_000
        assert summary["distinct_subjects"] == 4_750
        assert summary["functionality_numerator"] == 19
        assert summary["functionality_denominator"] == 20
        assert (
            summary["functionality_numerator"] * summary["support"]
            == summary["functionality_denominator"]
            * summary["distinct_subjects"]
        )


def test_three_mapping_representations_reconstruct_identically(package_fixture):
    files = package_fixture.files
    from_json = json.loads(files["hashmap.json"])
    from_jsonl = {
        row["key"]: row["values"] for row in _jsonl_rows(files["hashmap.jsonl"])
    }
    from_records = {
        row["display_key"]: row["values"]
        for row in _jsonl_rows(files["records.jsonl"])
    }
    assert from_json == from_jsonl == from_records
    assert len(from_json) == 3_000
    assert sum(len(values) for values in from_json.values()) == 3_032


def test_sha256sums_covers_exactly_the_other_nine_members(package_fixture):
    lines = package_fixture.files["SHA256SUMS"].decode("utf-8").splitlines()
    parsed = {}
    for line in lines:
        digest, name = line.split("  ", 1)
        assert len(digest) == 64
        assert digest == digest.lower()
        int(digest, 16)
        parsed[name] = digest
    assert set(parsed) == PACKAGE_MEMBERS - {"SHA256SUMS"}
    assert len(lines) == 9
    assert list(parsed) == sorted(parsed)
    for name, digest in parsed.items():
        assert digest == hashlib.sha256(package_fixture.files[name]).hexdigest()


def test_deterministic_zip_writes_are_byte_identical(tmp_path, package_fixture):
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    write_deterministic_zip(first, package_fixture.files)
    write_deterministic_zip(second, package_fixture.files)
    assert first.read_bytes() == second.read_bytes()
    _validate_structure(first, package_fixture)
    _validate_structure(second, package_fixture)


def test_zip_member_order_and_metadata_are_frozen(rendered_archive):
    with zipfile.ZipFile(rendered_archive) as archive:
        members = archive.infolist()
        assert [item.filename for item in members] == [
            f"{ARCHIVE_ROOT}/{name}" for name in sorted(PACKAGE_MEMBERS)
        ]
        assert archive.testzip() is None
        for item in members:
            assert not item.is_dir()
            assert item.date_time == (1980, 1, 1, 0, 0, 0)
            assert item.create_system == 3
            assert item.external_attr == (stat.S_IFREG | 0o644) << 16
            assert item.compress_type == zipfile.ZIP_DEFLATED


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "/absolute.txt",
        f"{ARCHIVE_ROOT}/../escape.txt",
        f"{ARCHIVE_ROOT}\\README.md",
    ],
)
def test_structural_validation_rejects_unsafe_names(
    tmp_path,
    package_fixture,
    unsafe_name,
):
    members = _raw_zip_members(package_fixture.files)
    _, content, timestamp, mode = members[0]
    members[0] = (unsafe_name, content, timestamp, mode)
    archive = tmp_path / "unsafe.zip"
    _write_raw_zip(archive, members)
    with pytest.raises(HashmapBuildError, match="unsafe"):
        _validate_structure(archive, package_fixture)


def test_structural_validation_rejects_duplicate_members(
    tmp_path,
    package_fixture,
):
    members = _raw_zip_members(package_fixture.files)
    members.append(members[0])
    archive = tmp_path / "duplicate.zip"
    _write_raw_zip(archive, members)
    with pytest.raises(HashmapBuildError, match="duplicate"):
        _validate_structure(archive, package_fixture)


def test_structural_validation_rejects_metadata_drift(
    tmp_path,
    package_fixture,
):
    members = _raw_zip_members(package_fixture.files)
    name, content, _, mode = members[0]
    members[0] = (name, content, (1981, 1, 1, 0, 0, 0), mode)
    archive = tmp_path / "metadata.zip"
    _write_raw_zip(archive, members)
    with pytest.raises(HashmapBuildError, match="metadata"):
        _validate_structure(archive, package_fixture)


def test_structural_validation_rejects_checksum_drift(
    tmp_path,
    package_fixture,
):
    files = dict(package_fixture.files)
    files["README.md"] += b"drift\n"
    archive = tmp_path / "checksum.zip"
    write_deterministic_zip(archive, files)
    with pytest.raises(HashmapBuildError, match="checksum"):
        _validate_structure(archive, package_fixture)


def test_structural_validation_rejects_noncanonical_json(
    tmp_path,
    package_fixture,
):
    files = dict(package_fixture.files)
    value = json.loads(files["hashmap.json"])
    files["hashmap.json"] = (
        json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    )
    _refresh_sha256sums(files)
    archive = tmp_path / "noncanonical.zip"
    write_deterministic_zip(archive, files)
    with pytest.raises(HashmapBuildError, match="canonical"):
        _validate_structure(archive, package_fixture)


def test_structural_validation_rejects_malformed_ids(
    tmp_path,
    package_fixture,
):
    files = dict(package_fixture.files)
    _replace_first_value_id(
        files,
        "Qnot-an-id",
        representations={"json", "jsonl", "records"},
    )
    archive = tmp_path / "malformed-id.zip"
    write_deterministic_zip(archive, files)
    with pytest.raises(HashmapBuildError, match="invalid Q id"):
        _validate_structure(archive, package_fixture)


def test_structural_validation_rejects_mapping_disagreement(
    tmp_path,
    package_fixture,
):
    files = dict(package_fixture.files)
    _replace_first_value_id(
        files,
        "Q999999998",
        representations={"jsonl"},
    )
    archive = tmp_path / "mapping-disagreement.zip"
    write_deterministic_zip(archive, files)
    with pytest.raises(HashmapBuildError, match="mapping"):
        _validate_structure(archive, package_fixture)


@dataclass(frozen=True)
class _OfflineSource:
    source_root: Path
    lock_path: Path
    lock: WikidataLock


def _tar_regular_file(archive, source, member_name):
    info = tarfile.TarInfo(member_name)
    info.size = source.stat().st_size
    info.mode = 0o644
    info.mtime = 0
    with source.open("rb") as stream:
        archive.addfile(info, stream)


@pytest.fixture(scope="module")
def offline_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("wikidata-offline-source")
    staging = root / "staging"
    source_root = root / "source"
    staging.mkdir()
    source_root.mkdir()

    train = staging / "wikidata5m_transductive_train.txt"
    with train.open("w", encoding="utf-8", newline="\n") as stream:
        for relation in range(1, 33):
            for subject in range(1, 5_001):
                stream.write(
                    f"Q{subject}\tP{relation}\tQ{10_000 + subject}\n"
                )

    relations = staging / "wikidata5m_relation.txt"
    relations.write_text(
        "".join(
            f"P{relation}\tOffline relation {relation}\n"
            for relation in range(1, 33)
        ),
        encoding="utf-8",
    )
    entities = staging / "wikidata5m_entity.txt"
    with entities.open("w", encoding="utf-8", newline="\n") as stream:
        for subject in range(1, 5_001):
            stream.write(f"Q{subject}\tOffline subject {subject}\n")
        for object_id in range(10_001, 15_001):
            stream.write(f"Q{object_id}\tOffline object {object_id}\n")

    alias_archive = source_root / "wikidata5m_alias.tar.gz"
    with tarfile.open(alias_archive, "w:gz") as archive:
        _tar_regular_file(archive, entities, entities.name)
        _tar_regular_file(archive, relations, relations.name)
    transductive_archive = (
        source_root / "wikidata5m_transductive.tar.gz"
    )
    with tarfile.open(transductive_archive, "w:gz") as archive:
        _tar_regular_file(archive, train, train.name)

    locked_files = {}
    for path in sorted(source_root.iterdir()):
        content = path.read_bytes()
        locked_files[path.name] = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    lock = WikidataLock.from_dict(
        {
            "repo_id": "intfloat/wikidata5m",
            "repo_type": "dataset",
            "revision": "6b2b09672129e280c0c9da97ab58154e9d535e6b",
            "files": locked_files,
        }
    )
    lock_path = root / "wikidata5m_hashmap.lock.json"
    lock_path.write_bytes(lock.canonical_bytes())
    return _OfflineSource(source_root, lock_path, lock)


def _activate_offline_lock(monkeypatch, offline_source):
    monkeypatch.setattr(
        hashmap_module,
        "HASHMAP_LOCK_PATH",
        offline_source.lock_path,
    )
    monkeypatch.setattr(
        hashmap_module,
        "EXPECTED_HASHMAP_LOCK",
        offline_source.lock.to_dict(),
    )


@dataclass(frozen=True)
class _BuiltOfflineArchive:
    archive: Path
    work_root: Path
    report: ArchiveReport


@pytest.fixture(scope="module")
def built_offline_archive(tmp_path_factory, offline_source):
    root = tmp_path_factory.mktemp("wikidata-offline-build")
    work_root = root / "work"
    work_root.mkdir()
    archive = root / "wikidata5m-hashmap.zip"
    monkeypatch = pytest.MonkeyPatch()
    _activate_offline_lock(monkeypatch, offline_source)
    try:
        report = build_wikidata5m_hashmap(
            offline_source.source_root,
            archive,
            work_root=work_root,
        )
    finally:
        monkeypatch.undo()
    return _BuiltOfflineArchive(archive, work_root, report)


def test_source_backed_build_and_fresh_verification_are_sequential_and_clean(
    monkeypatch,
    offline_source,
    built_offline_archive,
):
    assert list(built_offline_archive.work_root.iterdir()) == []
    _activate_offline_lock(monkeypatch, offline_source)
    verified = verify_wikidata5m_hashmap(
        built_offline_archive.archive,
        offline_source.source_root,
        work_root=built_offline_archive.work_root,
    )
    assert verified == built_offline_archive.report
    assert list(built_offline_archive.work_root.iterdir()) == []

    assert [item.name for item in fields(ArchiveReport)] == [
        "archive_bytes",
        "archive_sha256",
        "edge_count",
        "key_count",
        "path",
        "relation_count",
    ]
    assert built_offline_archive.report == ArchiveReport(
        archive_bytes=built_offline_archive.archive.stat().st_size,
        archive_sha256=hashlib.sha256(
            built_offline_archive.archive.read_bytes()
        ).hexdigest(),
        edge_count=3_000,
        key_count=3_000,
        path=str(built_offline_archive.archive),
        relation_count=32,
    )

    with zipfile.ZipFile(built_offline_archive.archive) as archive:
        manifest = json.loads(
            archive.read(f"{ARCHIVE_ROOT}/build_manifest.json")
        )
        summaries = json.loads(
            archive.read(f"{ARCHIVE_ROOT}/relation_summary.json")
        )
    assert manifest["build_counts"]["source_triples"] == 160_000
    assert manifest["relation_filter_counts"] == {
        "missing_relation_alias": 0,
        "below_min_support": 0,
        "below_min_functionality": 0,
        "survived_not_selected": 0,
        "selected": 32,
    }
    assert [item["support"] for item in summaries] == [5_000] * 32
    assert [item["distinct_subjects"] for item in summaries] == [5_000] * 32
    assert [item["functionality_numerator"] for item in summaries] == [1] * 32
    assert [item["functionality_denominator"] for item in summaries] == [1] * 32


def test_source_backed_verification_rejects_coherent_wrong_edges(
    tmp_path,
    monkeypatch,
    offline_source,
    built_offline_archive,
):
    with zipfile.ZipFile(built_offline_archive.archive) as archive:
        files = {
            name.removeprefix(f"{ARCHIVE_ROOT}/"): archive.read(name)
            for name in archive.namelist()
        }
    _replace_first_value_id(
        files,
        "Q999999999",
        representations={"json", "jsonl", "records"},
    )
    candidate = tmp_path / "wrong-edge.zip"
    write_deterministic_zip(candidate, files)
    hashmap_module._validate_archive_structure(
        candidate,
        expected_dataset=None,
        lock=offline_source.lock,
        cc0_text=CC0_PATH.read_text(encoding="utf-8"),
    )

    work_root = tmp_path / "work"
    work_root.mkdir()
    _activate_offline_lock(monkeypatch, offline_source)
    with pytest.raises(HashmapBuildError, match="source-edge disagreement"):
        verify_wikidata5m_hashmap(
            candidate,
            offline_source.source_root,
            work_root=work_root,
        )
    assert list(work_root.iterdir()) == []


@pytest.mark.parametrize("existing_content", [None, b"previous archive"])
def test_staged_validation_failure_preserves_destination_atomically(
    tmp_path,
    monkeypatch,
    package_fixture,
    existing_content,
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    work_root = tmp_path / "work"
    work_root.mkdir()
    destination = tmp_path / "published.zip"
    if existing_content is not None:
        destination.write_bytes(existing_content)

    monkeypatch.setattr(
        hashmap_module,
        "_rebuild_from_sources",
        lambda *args, **kwargs: (
            package_fixture.dataset,
            package_fixture.selection,
        ),
    )

    def reject_staged_archive(path, **kwargs):
        assert Path(path).is_file()
        raise HashmapBuildError("injected staged-verification failure")

    monkeypatch.setattr(
        hashmap_module,
        "_validate_archive_structure",
        reject_staged_archive,
    )
    with pytest.raises(
        HashmapBuildError,
        match="injected staged-verification failure",
    ):
        build_wikidata5m_hashmap(
            source_root,
            destination,
            work_root=work_root,
        )

    if existing_content is None:
        assert not destination.exists()
    else:
        assert destination.read_bytes() == existing_content
    assert list(work_root.iterdir()) == []
    assert list(tmp_path.glob(f".{destination.name}.*")) == []


@pytest.mark.parametrize("operation", ["build", "verify"])
def test_build_and_verify_reject_non_python_312_runtime(
    tmp_path,
    monkeypatch,
    operation,
):
    monkeypatch.setattr(hashmap_module.sys, "version_info", (3, 11, 9))
    source_root = tmp_path / "source"
    source_root.mkdir()
    work_root = tmp_path / "work"
    work_root.mkdir()
    archive = tmp_path / "archive.zip"
    if operation == "build":
        target = lambda: build_wikidata5m_hashmap(
            source_root,
            archive,
            work_root=work_root,
        )
    else:
        target = lambda: verify_wikidata5m_hashmap(
            archive,
            source_root,
            work_root=work_root,
        )
    with pytest.raises(HashmapBuildError, match="Python 3.12"):
        target()
    assert list(work_root.iterdir()) == []
    assert not archive.exists()


def test_runtime_rejects_committed_lock_drift_before_source_access(
    tmp_path,
    monkeypatch,
):
    drifted = json.loads(HASHMAP_LOCK.read_text(encoding="utf-8"))
    drifted["revision"] = "f" * 40
    lock_path = tmp_path / "drifted-lock.json"
    lock_path.write_text(json.dumps(drifted), encoding="utf-8")
    monkeypatch.setattr(hashmap_module, "HASHMAP_LOCK_PATH", lock_path)
    source_root = tmp_path / "missing-source"
    work_root = tmp_path / "work"
    work_root.mkdir()
    destination = tmp_path / "archive.zip"

    with pytest.raises(HashmapBuildError, match="frozen lock"):
        build_wikidata5m_hashmap(
            source_root,
            destination,
            work_root=work_root,
        )
    assert not destination.exists()
    assert list(work_root.iterdir()) == []


def test_fixed_cli_help_is_repo_relative_and_has_no_frozen_overrides():
    repo = Path(__file__).resolve().parents[1]
    environment = {key: value for key, value in os.environ.items()}
    environment.pop("PYTHONPATH", None)
    outputs = []
    for arguments in (
        ["--help"],
        ["build", "--help"],
        ["verify", "--help"],
    ):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/build_wikidata5m_hashmap.py",
                *arguments,
            ],
            cwd=repo,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout)

    assert "build" in outputs[0]
    assert "verify" in outputs[0]
    assert all(
        option in outputs[1]
        for option in ("--source-root", "--out", "--work-root")
    )
    assert all(
        option in outputs[2]
        for option in ("--source-root", "--archive", "--work-root")
    )
    combined = "\n".join(outputs)
    for forbidden in (
        "--minimum-support",
        "--minimum-functionality",
        "--relation-count",
        "--key-count",
        "--revision",
        "--lock",
        "--cc0",
    ):
        assert forbidden not in combined
