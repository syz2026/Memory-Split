from __future__ import annotations

import copy
import gc
import gzip
import hashlib
import io
import json
import os
import shutil
import struct
import tarfile
import tempfile
import tracemalloc
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.reasoning_v2 import source_lock as source_lock_module
from corpusgen.reasoning_v2.source_lock import SourceFile
from corpusgen.reasoning_v2 import wikidata_source as wikidata_source_module
from corpusgen.reasoning_v2.wikidata_source import (
    ARCHIVE_PATHS,
    RECEIPT_FORMAT,
    RECEIPT_SCHEMA_VERSION,
    ArtifactRecord,
    IndexArtifactRecord,
    V2AliasRecord,
    V2TrainingTriple,
    WikidataDerivedView,
    WikidataDerivedViewReceipt,
    WikidataDerivedViewRef,
    _alias_record_from_row,
    _encode_alias_row,
    _open_verified_archives,
    build_wikidata_derived_view,
    iter_distinct_training_edges,
    iter_v2_aliases,
    iter_v2_training_triples,
    lookup_alias,
    lookup_training_triple,
    open_wikidata_derived_view,
    verify_wikidata_derived_view,
)
from reasoning_v2_fixtures import (
    FixtureSourceLock,
    fake_public_resolver,
    fixed_contract_environment,
    fixture_source_lock,
    full_recipe,
)


EXPECTED_GENERATOR_COMMIT = "a" * 40
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
EXPECTED_ARCHIVE_MEMBERS = {
    "wikidata5m_alias.tar.gz": (
        "wikidata5m_entity.txt",
        "wikidata5m_relation.txt",
    ),
    "wikidata5m_inductive.tar.gz": (
        "wikidata5m_inductive_test.txt",
        "wikidata5m_inductive_train.txt",
        "wikidata5m_inductive_valid.txt",
    ),
    "wikidata5m_transductive.tar.gz": (
        "wikidata5m_transductive_test.txt",
        "wikidata5m_transductive_train.txt",
        "wikidata5m_transductive_valid.txt",
    ),
}


@dataclass(frozen=True)
class _TarEntry:
    name: str
    payload: bytes = b""
    kind: str = "file"
    linkname: str = ""
    pax_path: str | None = None


@dataclass(frozen=True)
class _AuthorityFixture:
    source_lock_path: Path
    source_root: Path
    archive_payloads: dict[str, bytes]
    member_payloads: dict[str, bytes]


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
                member.linkname = entry.linkname
                if entry.pax_path is not None:
                    member.pax_headers = {"path": entry.pax_path}
                if entry.kind == "file":
                    member.type = tarfile.REGTYPE
                    member.size = len(entry.payload)
                    archive.addfile(member, io.BytesIO(entry.payload))
                else:
                    member.size = 0
                    member.type = {
                        "directory": tarfile.DIRTYPE,
                        "symlink": tarfile.SYMTYPE,
                        "hardlink": tarfile.LNKTYPE,
                        "fifo": tarfile.FIFOTYPE,
                        "character": tarfile.CHRTYPE,
                        "sparse": tarfile.GNUTYPE_SPARSE,
                    }[entry.kind]
                    archive.addfile(member)
    return compressed.getvalue()


def _tar_body(entries: list[_TarEntry]) -> bytes:
    payload = gzip.decompress(_tar_bytes(entries))
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        members = archive.getmembers()
    assert members
    last = members[-1]
    body_end = last.offset_data + ((last.size + 511) // 512) * 512
    return payload[:body_end]


def _truncated_member_archive() -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        member = tarfile.TarInfo("wikidata5m_entity.txt")
        member.size = 4096
        member.mtime = 0
        archive.addfile(member, io.BytesIO(b"x" * member.size))
    incomplete_payload = raw.getvalue()[: 512 + 127]
    return gzip.compress(incomplete_payload, mtime=0)


def _padded_tar_payload(payload: bytes) -> bytes:
    return payload + b"\0" * (-len(payload) % 512)


def _tar_header(
    name: str,
    *,
    size: int,
    typeflag: bytes = tarfile.REGTYPE,
) -> bytes:
    member = tarfile.TarInfo(name)
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    member.mtime = 0
    member.mode = 0o600
    member.size = size
    member.type = typeflag
    return member.tobuf(
        format=tarfile.USTAR_FORMAT,
        encoding="utf-8",
        errors="strict",
    )


def _pax_record(key: str, value: str) -> bytes:
    suffix = f" {key}={value}\n".encode("utf-8")
    length = len(suffix) + 1
    while True:
        record = str(length).encode("ascii") + suffix
        if len(record) == length:
            return record
        length = len(record)


def _pax_size_mismatch_archive(pax_type: bytes) -> bytes:
    entity_payload = b"Q1\tAda Lovelace\n"
    relation_payload = b"P"
    surprise_payload = b"surprise"
    zero_block = b"\0" * 512
    pax_payload = _pax_record("size", str(len(relation_payload)))
    raw_relation_payload = b"".join(
        (
            _padded_tar_payload(relation_payload),
            zero_block,
            _tar_header("surprise.txt", size=len(surprise_payload)),
            _padded_tar_payload(surprise_payload),
        )
    )
    assert len(raw_relation_payload) == 2048
    raw_tar = b"".join(
        (
            _tar_header(
                "wikidata5m_entity.txt",
                size=len(entity_payload),
            ),
            _padded_tar_payload(entity_payload),
            _tar_header(
                "PaxHeaders/relation",
                size=len(pax_payload),
                typeflag=pax_type,
            ),
            _padded_tar_payload(pax_payload),
            _tar_header("wikidata5m_relation.txt", size=2048),
            raw_relation_payload,
            zero_block * 2,
        )
    )
    return gzip.compress(raw_tar, mtime=0)


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


def _install_archive_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
    archive_payloads: dict[str, bytes],
) -> _AuthorityFixture:
    assert tuple(archive_payloads) == ARCHIVE_PATHS
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

    member_payloads = {
        f"members/{entry.name}": entry.payload
        for entries in _base_archive_entries().values()
        for entry in entries
        if entry.kind == "file" and entry.name in {
            member
            for members in EXPECTED_ARCHIVE_MEMBERS.values()
            for member in members
        }
    }
    return _AuthorityFixture(
        source_lock_path=source_lock_path,
        source_root=source_root,
        archive_payloads=archive_payloads,
        member_payloads=member_payloads,
    )


@pytest.fixture
def archive_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
) -> _AuthorityFixture:
    return _install_archive_authority(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        _base_archive_payloads(),
    )


def _artifact(path: str, *, size: int = 0) -> dict[str, object]:
    return {
        "bytes": size,
        "path": path,
        "sha256": EMPTY_SHA256,
    }


def _index_artifact(
    path: str,
    record_width: int,
    *,
    size: int = 0,
    count: int = 0,
) -> dict[str, object]:
    return {
        "bytes": size,
        "count": count,
        "path": path,
        "record_width": record_width,
        "sha256": EMPTY_SHA256,
    }


def _valid_receipt_dict() -> dict[str, object]:
    members = sorted(
        (
            f"members/{member}"
            for archive_members in EXPECTED_ARCHIVE_MEMBERS.values()
            for member in archive_members
        ),
        key=lambda value: value.encode("utf-8"),
    )
    return {
        "alias_rows": 0,
        "archives": [_artifact(path) for path in ARCHIVE_PATHS],
        "distinct_edges": 0,
        "format": RECEIPT_FORMAT,
        "generator_commit": EXPECTED_GENERATOR_COMMIT,
        "indexes": [
            _index_artifact("indexes/aliases.bin", 24),
            _index_artifact("indexes/inductive-training-offsets.bin", 8),
            _index_artifact("indexes/transductive-training-offsets.bin", 8),
        ],
        "members": [_artifact(path) for path in members],
        "overlap_audit_passed": True,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source_lock_sha256": "b" * 64,
        "streams": [
            _artifact("streams/aliases.tsv"),
            _artifact("streams/distinct-edges.tsv"),
            _artifact("streams/training.tsv"),
        ],
        "training_rows": 0,
    }


def test_receipt_parser_rejects_open_missing_and_noncanonical_fields():
    valid = _valid_receipt_dict()
    payload = canonical_json_bytes(valid)
    receipt = WikidataDerivedViewReceipt.from_bytes(payload)
    assert receipt.to_bytes() == payload
    assert receipt.archives == tuple(
        ArtifactRecord(path=path, bytes=0, sha256=EMPTY_SHA256)
        for path in ARCHIVE_PATHS
    )

    malformed: list[bytes] = []
    unknown = copy.deepcopy(valid)
    unknown["future_field"] = "open"
    malformed.append(canonical_json_bytes(unknown))
    missing = copy.deepcopy(valid)
    missing.pop("streams")
    malformed.append(canonical_json_bytes(missing))
    open_artifact = copy.deepcopy(valid)
    open_artifact["archives"][0]["future_field"] = "open"
    malformed.append(canonical_json_bytes(open_artifact))
    boolean_count = copy.deepcopy(valid)
    boolean_count["training_rows"] = False
    malformed.append(canonical_json_bytes(boolean_count))
    unsafe_path = copy.deepcopy(valid)
    unsafe_path["members"][0]["path"] = "members/../escape"
    malformed.append(canonical_json_bytes(unsafe_path))
    bad_hash = copy.deepcopy(valid)
    bad_hash["source_lock_sha256"] = "B" * 64
    malformed.append(canonical_json_bytes(bad_hash))
    bad_commit = copy.deepcopy(valid)
    bad_commit["generator_commit"] = "main"
    malformed.append(canonical_json_bytes(bad_commit))
    unsorted = copy.deepcopy(valid)
    unsorted["archives"] = list(reversed(unsorted["archives"]))
    malformed.append(canonical_json_bytes(unsorted))
    inconsistent = copy.deepcopy(valid)
    inconsistent["distinct_edges"] = 1
    malformed.append(canonical_json_bytes(inconsistent))
    failed_audit = copy.deepcopy(valid)
    failed_audit["overlap_audit_passed"] = False
    malformed.append(canonical_json_bytes(failed_audit))
    malformed.append(json.dumps(valid, indent=2).encode("utf-8"))
    malformed.append(payload + b"\n")
    malformed.append(
        payload.replace(
            b'{"alias_rows":',
            b'{"alias_rows":0,"alias_rows":',
            1,
        )
    )

    for candidate in malformed:
        with pytest.raises(ValueError):
            WikidataDerivedViewReceipt.from_bytes(candidate)


def test_receipt_parser_rejects_nonempty_training_with_zero_distinct_edges():
    invalid = _valid_receipt_dict()
    invalid["training_rows"] = 1
    invalid["streams"][2]["bytes"] = 1
    invalid["indexes"][1]["bytes"] = 8
    invalid["indexes"][1]["count"] = 1

    with pytest.raises(ValueError, match="training.*distinct edge"):
        WikidataDerivedViewReceipt.from_bytes(canonical_json_bytes(invalid))


def test_receipt_index_rows_bind_exact_count_and_record_width(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    authority = _authority_from_members(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {
            "wikidata5m_entity.txt": b"Q2\tTwo\nQ10\tTen\n",
            "wikidata5m_relation.txt": b"P1\tknows\n",
            "wikidata5m_inductive_train.txt": b"Q1\tP1\tQ2\nQ5\tP3\tQ6\n",
            "wikidata5m_transductive_train.txt": b"Q3\tP2\tQ4\n",
        },
    )
    ref = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    receipt = WikidataDerivedViewReceipt.from_bytes(
        (ref.root / "receipt.json").read_bytes(),
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    assert all(
        isinstance(record, IndexArtifactRecord) for record in receipt.indexes
    )
    indexes = {record.path: record for record in receipt.indexes}
    inductive = indexes["indexes/inductive-training-offsets.bin"]
    transductive = indexes["indexes/transductive-training-offsets.bin"]
    aliases = indexes["indexes/aliases.bin"]
    assert (inductive.record_width, inductive.count) == (8, 2)
    assert (transductive.record_width, transductive.count) == (8, 1)
    assert (aliases.record_width, aliases.count) == (24, 3)
    assert inductive.count + transductive.count == receipt.training_rows == 3
    assert aliases.count == receipt.alias_rows == 3
    for record in receipt.indexes:
        assert record.count * record.record_width == record.bytes

    # The closed receipt round-trips with the index count/width fields.
    assert (
        WikidataDerivedViewReceipt.from_bytes(receipt.to_bytes()) == receipt
    )
    on_disk = json.loads((ref.root / "receipt.json").read_text(encoding="utf-8"))
    for entry in on_disk["indexes"]:
        assert set(entry) == {"bytes", "count", "path", "record_width", "sha256"}

    # The parser rejects an index row that omits the new fields or whose
    # count/record_width are inconsistent with its byte length.
    missing = _valid_receipt_dict()
    missing["indexes"][0].pop("count")
    with pytest.raises(ValueError):
        WikidataDerivedViewReceipt.from_bytes(canonical_json_bytes(missing))

    inconsistent = _valid_receipt_dict()
    inconsistent["indexes"][0]["bytes"] = 24
    inconsistent["indexes"][0]["count"] = 0
    with pytest.raises(ValueError):
        WikidataDerivedViewReceipt.from_bytes(canonical_json_bytes(inconsistent))


def test_aliases_tsv_uses_exact_canonical_object_wire_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    authority = _authority_from_members(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {
            "wikidata5m_entity.txt": b"Q1\tAda\tAda Lovelace\n",
            "wikidata5m_relation.txt": b"P1\tknows\n",
        },
    )
    view = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    aliases_tsv = (view.root / "streams" / "aliases.tsv").read_bytes()
    assert aliases_tsv == (
        b'Q1\t{"aliases":["Ada","Ada Lovelace"],"display":"Ada"}\n'
        b'P1\t{"aliases":["knows"],"display":"knows"}\n'
    )

    # The exact two-column canonical object round-trips and derives kind.
    record = _alias_record_from_row(
        b'Q1\t{"aliases":["Ada","Ada Lovelace"],"display":"Ada"}'
    )
    assert record == V2AliasRecord(
        canonical_id="Q1",
        kind="entity",
        display="Ada",
        aliases=("Ada", "Ada Lovelace"),
    )
    assert _encode_alias_row("Q1", "Ada", ("Ada", "Ada Lovelace")) == (
        b'Q1\t{"aliases":["Ada","Ada Lovelace"],"display":"Ada"}\n'
    )

    # Alternative representations forbidden by schema version 1 are rejected.
    for forbidden in (
        b'Q1\t"Ada"',
        b'Q1\t{"display":"Ada"}',
        b'Q1\t{"aliases":["Ada"],"display":"Ada","kind":"entity"}',
        b'Q1\t{"display":"Ada","aliases":["Ada"]}',
        b'Q1\t{"aliases":["Ada"]}',
    ):
        with pytest.raises(ValueError):
            _alias_record_from_row(forbidden)


def test_archive_authority_hashes_and_parses_the_same_descriptor(
    archive_authority: _AuthorityFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    with _open_verified_archives(
        archive_authority.source_lock_path,
        archive_authority.source_root,
    ) as verified:
        assert tuple(record.path for record in verified.archives) == ARCHIVE_PATHS
        assert {
            record.path: (record.bytes, record.sha256)
            for record in verified.archives
        } == {
            name: (len(payload), hashlib.sha256(payload).hexdigest())
            for name, payload in archive_authority.archive_payloads.items()
        }
        assert {
            record.path: (record.bytes, record.sha256)
            for record in verified.members
        } == {
            path: (len(payload), hashlib.sha256(payload).hexdigest())
            for path, payload in archive_authority.member_payloads.items()
        }

    swapped = False
    victim = (
        archive_authority.source_root
        / "wikidata5m"
        / "wikidata5m_alias.tar.gz"
    )
    displaced = victim.with_name(victim.name + ".displaced")

    def swap_after_hash(phase, archive_name, _verified):
        nonlocal swapped
        if (
            phase == "after_hash"
            and archive_name == victim.name
            and not swapped
        ):
            swapped = True
            victim.rename(displaced)
            victim.write_bytes(b"!" * displaced.stat().st_size)

    monkeypatch.setattr(
        wikidata_source_module,
        "_archive_authority_hook",
        swap_after_hash,
    )
    with pytest.raises(ValueError, match="archive namespace identity drift"):
        with _open_verified_archives(
            archive_authority.source_lock_path,
            archive_authority.source_root,
        ):
            pass


def test_archive_authority_rejects_extra_wikidata_archive_lock_row(
    archive_authority: _AuthorityFixture,
):
    extra_name = "wikidata5m_extra.tar.gz"
    extra_payload = _tar_bytes([_TarEntry("surprise.txt", b"surprise")])
    (
        archive_authority.source_root
        / "wikidata5m"
        / extra_name
    ).write_bytes(extra_payload)

    lock_value = json.loads(
        archive_authority.source_lock_path.read_text(encoding="utf-8")
    )
    wikidata_entry = next(
        entry
        for entry in lock_value["sources"]
        if entry["source_id"] == "wikidata5m"
    )
    wikidata_entry["files"].append(
        {
            "bytes": len(extra_payload),
            "path": extra_name,
            "sha256": hashlib.sha256(extra_payload).hexdigest(),
        }
    )
    wikidata_entry["files"].sort(
        key=lambda row: row["path"].encode("utf-8")
    )
    archive_authority.source_lock_path.write_bytes(
        canonical_json_bytes(lock_value)
    )

    with pytest.raises(ValueError, match="source lock archive inventory"):
        with _open_verified_archives(
            archive_authority.source_lock_path,
            archive_authority.source_root,
        ):
            pass


@pytest.mark.parametrize(
    "attack",
    ["archive_inode", "archive_hardlink", "parent", "parent_aba", "content_aba"],
)
def test_archive_authority_rejects_path_inode_parent_and_aba_drift(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    attack: str,
):
    wikidata_root = archive_authority.source_root / "wikidata5m"
    victim = wikidata_root / "wikidata5m_inductive.tar.gz"
    if attack == "archive_hardlink":
        second_name = tmp_path / "archive-hardlink"
        os.link(victim, second_name)
        with pytest.raises(ValueError, match="hardlink"):
            with _open_verified_archives(
                archive_authority.source_lock_path,
                archive_authority.source_root,
            ):
                pass
        return

    with pytest.raises(ValueError, match="identity drift|changed during verification"):
        with _open_verified_archives(
            archive_authority.source_lock_path,
            archive_authority.source_root,
        ):
            if attack == "archive_inode":
                displaced = victim.with_name(victim.name + ".displaced")
                payload = victim.read_bytes()
                victim.rename(displaced)
                victim.write_bytes(payload)
            elif attack == "parent":
                displaced = wikidata_root.with_name("wikidata5m-displaced")
                wikidata_root.rename(displaced)
                shutil.copytree(displaced, wikidata_root)
            elif attack == "parent_aba":
                displaced = wikidata_root.with_name("wikidata5m-displaced")
                wikidata_root.rename(displaced)
                displaced.rename(wikidata_root)
            else:
                payload = victim.read_bytes()
                changed = bytes((payload[0] ^ 1,)) + payload[1:]
                victim.write_bytes(changed)
                victim.write_bytes(payload)


@pytest.mark.parametrize(
    "attack",
    [
        "absolute",
        "traversal",
        "windows",
        "nul",
        "symlink",
        "hardlink",
        "fifo",
        "device",
        "sparse",
        "duplicate",
        "collision",
        "undeclared",
        "truncated",
    ],
)
def test_archive_authority_rejects_unsafe_or_undeclared_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
    attack: str,
):
    entries = _base_archive_entries()
    alias_entries = entries["wikidata5m_alias.tar.gz"]
    if attack == "absolute":
        alias_entries[0] = replace(alias_entries[0], name="/absolute.txt")
    elif attack == "traversal":
        alias_entries[0] = replace(alias_entries[0], name="../escape.txt")
    elif attack == "windows":
        alias_entries[0] = replace(alias_entries[0], name="C:\\escape.txt")
    elif attack == "nul":
        alias_entries[0] = replace(
            alias_entries[0],
            pax_path="bad\x00name.txt",
        )
    elif attack in {"symlink", "hardlink"}:
        alias_entries[0] = _TarEntry(
            "wikidata5m_entity.txt",
            kind=attack,
            linkname="wikidata5m_relation.txt",
        )
    elif attack == "fifo":
        alias_entries[0] = _TarEntry("wikidata5m_entity.txt", kind="fifo")
    elif attack == "device":
        alias_entries[0] = _TarEntry("wikidata5m_entity.txt", kind="character")
    elif attack == "sparse":
        alias_entries[0] = _TarEntry("wikidata5m_entity.txt", kind="sparse")
    elif attack == "duplicate":
        alias_entries.append(copy.deepcopy(alias_entries[0]))
    elif attack == "collision":
        alias_entries.extend(
            [
                _TarEntry("collision", b"file"),
                _TarEntry("collision/child", b"child"),
            ]
        )
    elif attack == "undeclared":
        alias_entries.append(_TarEntry("surprise.txt", b"surprise"))

    archive_payloads = {
        name: _tar_bytes(archive_entries)
        for name, archive_entries in entries.items()
    }
    if attack == "truncated":
        archive_payloads["wikidata5m_alias.tar.gz"] = (
            _truncated_member_archive()
        )
    authority = _install_archive_authority(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        archive_payloads,
    )
    with pytest.raises(
        ValueError,
        match="archive|member|unsafe|undeclared|duplicate|collision|truncated",
    ):
        with _open_verified_archives(
            authority.source_lock_path,
            authority.source_root,
        ):
            pass


@pytest.mark.parametrize(
    "attack",
    [
        "missing_both_terminal_blocks",
        "only_one_terminal_block",
        "undeclared_member_after_terminator",
        "trailing_decompressed_payload",
        "trailing_compressed_payload",
    ],
)
def test_archive_authority_rejects_open_physical_archive_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
    attack: str,
):
    entries = _base_archive_entries()
    archive_payloads = {
        name: _tar_bytes(archive_entries)
        for name, archive_entries in entries.items()
    }
    alias_name = "wikidata5m_alias.tar.gz"
    alias_body = _tar_body(entries[alias_name])
    zero_block = b"\0" * 512
    if attack == "missing_both_terminal_blocks":
        attacked = gzip.compress(alias_body, mtime=0)
    elif attack == "only_one_terminal_block":
        attacked = gzip.compress(alias_body + zero_block, mtime=0)
    elif attack == "undeclared_member_after_terminator":
        undeclared_body = _tar_body([_TarEntry("surprise.txt", b"surprise")])
        attacked = gzip.compress(
            alias_body + zero_block * 2 + undeclared_body + zero_block * 2,
            mtime=0,
        )
    elif attack == "trailing_decompressed_payload":
        decompressed = gzip.decompress(archive_payloads[alias_name])
        attacked = gzip.compress(decompressed + b"trailing payload", mtime=0)
    else:
        attacked = archive_payloads[alias_name] + gzip.compress(
            b"trailing payload",
            mtime=0,
        )
    archive_payloads[alias_name] = attacked

    authority = _install_archive_authority(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        archive_payloads,
    )
    with pytest.raises(ValueError, match="archive"):
        with _open_verified_archives(
            authority.source_lock_path,
            authority.source_root,
        ):
            pass


def test_archive_authority_rejects_local_pax_size_override_hiding_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    archive_payloads = _base_archive_payloads()
    archive_payloads["wikidata5m_alias.tar.gz"] = (
        _pax_size_mismatch_archive(tarfile.XHDTYPE)
    )
    authority = _install_archive_authority(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        archive_payloads,
    )

    with pytest.raises(ValueError, match="archive.*size"):
        with _open_verified_archives(
            authority.source_lock_path,
            authority.source_root,
        ):
            pass


def test_archive_authority_rejects_global_pax_size_override_hiding_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    archive_payloads = _base_archive_payloads()
    archive_payloads["wikidata5m_alias.tar.gz"] = (
        _pax_size_mismatch_archive(tarfile.XGLTYPE)
    )
    authority = _install_archive_authority(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        archive_payloads,
    )

    with pytest.raises(ValueError, match="archive.*size"):
        with _open_verified_archives(
            authority.source_lock_path,
            authority.source_root,
        ):
            pass


def _source_snapshot(root: Path) -> tuple[tuple[str, str, bytes | None], ...]:
    result = []
    for path in sorted(
        root.rglob("*"),
        key=lambda candidate: candidate.relative_to(root).as_posix().encode(
            "utf-8"
        ),
    ):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            result.append((relative, "directory", None))
        else:
            result.append((relative, "file", path.read_bytes()))
    return tuple(result)


def test_archive_authority_leaves_source_root_byte_identical(
    archive_authority: _AuthorityFixture,
):
    before = _source_snapshot(archive_authority.source_root)
    with _open_verified_archives(
        archive_authority.source_lock_path,
        archive_authority.source_root,
    ) as verified:
        assert len(verified.archives) == 3
        assert len(verified.members) == 8
    assert _source_snapshot(archive_authority.source_root) == before


# --- Task 2: canonical streams, indexes, and no-replace publication ---------


def _authority_from_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
    members: dict[str, bytes],
) -> _AuthorityFixture:
    entries = _base_archive_entries()
    for entry_list in entries.values():
        for index, entry in enumerate(entry_list):
            if entry.name in members:
                entry_list[index] = replace(entry, payload=members[entry.name])
    archive_payloads = {
        name: _tar_bytes(archive_entries)
        for name, archive_entries in entries.items()
    }
    return _install_archive_authority(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        archive_payloads,
    )


def _view_tree(root: Path) -> dict[str, bytes]:
    tree: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            tree[path.relative_to(root).as_posix()] = path.read_bytes()
    return tree


def _alias_value(display: str, aliases: tuple[str, ...]) -> str:
    return json.dumps(
        {"aliases": list(aliases), "display": display},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _open_session(authority: _AuthorityFixture, ref: WikidataDerivedViewRef):
    return open_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        ref,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )


def _load_receipt(ref: WikidataDerivedViewRef) -> WikidataDerivedViewReceipt:
    return WikidataDerivedViewReceipt.from_bytes(
        (Path(ref.root) / "receipt.json").read_bytes(),
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )


def test_repeated_builds_produce_identical_receipt_streams_and_indexes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    authority = _authority_from_members(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {
            "wikidata5m_entity.txt": b"Q10\tTen\nQ2\tTwo\tSecond\n",
            "wikidata5m_relation.txt": b"P1\tknows\n",
            "wikidata5m_inductive_train.txt": b"Q5\tP3\tQ6\nQ1\tP1\tQ2\n",
            "wikidata5m_transductive_train.txt": b"Q3\tP2\tQ4\n",
        },
    )

    first = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out-first",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    second = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out-second",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )

    assert isinstance(first, WikidataDerivedViewRef)
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.root.name == first.receipt_sha256
    assert first.root.parent.name == "wikidata"
    assert _view_tree(first.root) == _view_tree(second.root)

    receipt_bytes = (first.root / "receipt.json").read_bytes()
    assert hashlib.sha256(receipt_bytes).hexdigest() == first.receipt_sha256
    receipt = _load_receipt(first)
    assert receipt.to_bytes() == receipt_bytes
    # The content address must not depend on machine-local values.
    text = receipt_bytes.decode("utf-8")
    assert str(tmp_path) not in text
    inventory = {
        record.path
        for record in (receipt.members + receipt.streams + receipt.indexes)
    }
    assert set(_view_tree(first.root)) == inventory | {"receipt.json"}


def test_training_and_alias_order_matches_frozen_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    authority = _authority_from_members(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {
            "wikidata5m_entity.txt": b"Q10\tTen\nQ2\tTwo\n",
            "wikidata5m_relation.txt": b"P1\tknows\n",
            "wikidata5m_inductive_train.txt": b"Q5\tP3\tQ6\nQ1\tP1\tQ2\n",
            "wikidata5m_transductive_train.txt": b"Q3\tP2\tQ4\n",
        },
    )
    ref = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )

    with _open_session(authority, ref) as view:
        training = list(iter_v2_training_triples(view))
        aliases = list(iter_v2_aliases(view))
        distinct = list(iter_distinct_training_edges(view))

    assert [
        (t.training_split, t.row, t.subject, t.relation, t.object)
        for t in training
    ] == [
        ("inductive_train", 1, 5, "P3", 6),
        ("inductive_train", 2, 1, "P1", 2),
        ("transductive_train", 1, 3, "P2", 4),
    ]
    assert training[0].member == "wikidata5m_inductive_train.txt"
    assert training[0].archive_path == "wikidata5m_inductive.tar.gz"
    assert training[2].member == "wikidata5m_transductive_train.txt"
    assert training[2].archive_path == "wikidata5m_transductive.tar.gz"

    training_tsv = (ref.root / "streams" / "training.tsv").read_bytes()
    assert training_tsv == (
        b"inductive_train\t1\tQ5\tP3\tQ6\n"
        b"inductive_train\t2\tQ1\tP1\tQ2\n"
        b"transductive_train\t1\tQ3\tP2\tQ4\n"
    )

    assert [(a.canonical_id, a.kind) for a in aliases] == [
        ("Q2", "entity"),
        ("Q10", "entity"),
        ("P1", "relation"),
    ]

    assert [(e.subject, e.relation, e.object) for e in distinct] == [
        (1, "P1", 2),
        (3, "P2", 4),
        (5, "P3", 6),
    ]
    assert distinct[0].training_split == "inductive_train"
    assert distinct[0].row == 2


def test_alias_ambiguity_is_removed_globally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    authority = _authority_from_members(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {
            "wikidata5m_entity.txt": b"Q1\tShared\tUnique1\nQ2\tShared\n",
            "wikidata5m_relation.txt": b"P1\tshared\tRel1\n",
        },
    )
    ref = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )

    with _open_session(authority, ref) as view:
        aliases = {a.canonical_id: a for a in iter_v2_aliases(view)}
        # "Shared"/"shared" collapse to the same normalized surface form owned
        # by Q1, Q2, and P1, so it is removed from every record globally.
        assert set(aliases) == {"Q1", "P1"}
        assert aliases["Q1"].aliases == ("Unique1",)
        assert aliases["Q1"].display == "Unique1"
        assert aliases["P1"].aliases == ("Rel1",)
        # Q2 loses its only surface form and disappears entirely.
        assert lookup_alias(view, "Q2") is None
        assert lookup_alias(view, "Q1") == aliases["Q1"]
        assert lookup_alias(view, "P1") == aliases["P1"]


def test_indexed_lookup_matches_streaming_without_archive_rescan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    inductive = b"".join(
        f"Q{i}\tP1\tQ{i + 1000}\n".encode("utf-8") for i in range(1, 31)
    )
    transductive = b"".join(
        f"Q{i}\tP2\tQ{i + 2000}\n".encode("utf-8") for i in range(1, 11)
    )
    entity = b"".join(
        f"Q{i}\tentity number {i}\tE{i}\n".encode("utf-8") for i in range(1, 27)
    )
    relation = b"P1\tknows\nP2\trelated to\n"
    authority = _authority_from_members(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {
            "wikidata5m_entity.txt": entity,
            "wikidata5m_relation.txt": relation,
            "wikidata5m_inductive_train.txt": inductive,
            "wikidata5m_transductive_train.txt": transductive,
        },
    )
    ref = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )

    training_tsv = (ref.root / "streams" / "training.tsv").read_bytes()
    aliases_tsv = (ref.root / "streams" / "aliases.tsv").read_bytes()

    # Reads use only the session's retained decoded-view descriptors; each
    # lookup is seek-bounded (strictly fewer bytes than the full stream) and no
    # read reopens or scans an archive.
    with _open_session(authority, ref) as view:
        streamed = list(iter_v2_training_triples(view))
        for triple in streamed:
            assert (
                lookup_training_triple(view, triple.training_split, triple.row)
                == triple
            )
        alias_records = list(iter_v2_aliases(view))
        for record in alias_records:
            assert lookup_alias(view, record.canonical_id) == record
        assert lookup_alias(view, "Q999999") is None
        assert lookup_alias(view, "P999") is None

        original_pread = wikidata_source_module._pread
        counter = {"bytes": 0}

        def counting_pread(descriptor: int, count: int, offset: int) -> bytes:
            data = original_pread(descriptor, count, offset)
            counter["bytes"] += len(data)
            return data

        monkeypatch.setattr(wikidata_source_module, "_pread", counting_pread)

        counter["bytes"] = 0
        lookup_training_triple(view, "inductive_train", 15)
        assert 0 < counter["bytes"] < len(training_tsv)

        counter["bytes"] = 0
        lookup_alias(view, "Q13")
        assert 0 < counter["bytes"] < len(aliases_tsv)


def test_train_sealed_overlap_and_malformed_rows_fail_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    def _no_published_winner(output_root: Path) -> bool:
        namespace = output_root / "wikidata"
        if not namespace.exists():
            return True
        return not any(len(child.name) == 64 for child in namespace.iterdir())

    overlap = _authority_from_members(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {
            # Q9 P9 Q10 is also the base inductive_test (sealed) edge.
            "wikidata5m_inductive_train.txt": b"Q1\tP1\tQ2\nQ9\tP9\tQ10\n",
        },
    )
    overlap_out = tmp_path / "overlap-out"
    with pytest.raises(ValueError, match="overlap|sealed"):
        build_wikidata_derived_view(
            overlap.source_lock_path,
            overlap.source_root,
            overlap_out,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )
    assert _no_published_winner(overlap_out)

    malformed = _authority_from_members(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {"wikidata5m_transductive_train.txt": b"Q3\tP2\n"},
    )
    malformed_out = tmp_path / "malformed-out"
    with pytest.raises(ValueError):
        build_wikidata_derived_view(
            malformed.source_lock_path,
            malformed.source_root,
            malformed_out,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )
    assert _no_published_winner(malformed_out)


def test_no_replace_publication_reuses_only_a_fully_verified_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    authority = _authority_from_members(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {
            "wikidata5m_entity.txt": b"Q1\tOne\nQ2\tTwo\n",
            "wikidata5m_relation.txt": b"P1\tknows\n",
        },
    )
    output_root = tmp_path / "out"

    first = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        output_root,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    winner_tree = _view_tree(first.root)
    winner_inode = first.root.stat().st_ino

    second = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        output_root,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    assert second.root == first.root
    assert second.root.stat().st_ino == winner_inode
    assert _view_tree(second.root) == winner_tree
    # No leftover private build directories.
    assert sorted(
        child.name for child in (output_root / "wikidata").iterdir()
    ) == [first.receipt_sha256]

    verified = verify_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        first.root,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    assert verified.receipt_sha256 == first.receipt_sha256

    corrupt_target = first.root / "streams" / "training.tsv"
    corrupt_target.write_bytes(corrupt_target.read_bytes() + b"x")

    # A corrupt winner is fully re-verified before reuse, so the build refuses
    # it and never repairs it in place.
    with pytest.raises(ValueError):
        build_wikidata_derived_view(
            authority.source_lock_path,
            authority.source_root,
            output_root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )
    assert corrupt_target.read_bytes().endswith(b"x")


def test_stream_or_index_drift_fails_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    authority = _authority_from_members(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {
            "wikidata5m_entity.txt": b"Q1\tOne\nQ2\tTwo\n",
            "wikidata5m_relation.txt": b"P1\tknows\n",
        },
    )
    ref = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )

    # A clean view verifies and opens.
    verify_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        ref.root,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    with _open_session(authority, ref) as view:
        assert list(iter_v2_aliases(view))

    stream = ref.root / "streams" / "aliases.tsv"
    original = stream.read_bytes()
    stream.write_bytes(original[:-1] + bytes((original[-1] ^ 1,)))
    with pytest.raises(ValueError):
        verify_wikidata_derived_view(
            authority.source_lock_path,
            authority.source_root,
            ref.root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )
    with pytest.raises(ValueError):
        with _open_session(authority, ref):
            pass
    stream.write_bytes(original)

    index = ref.root / "indexes" / "aliases.bin"
    index_bytes = bytearray(index.read_bytes())
    index_bytes[-1] ^= 1
    index.write_bytes(bytes(index_bytes))
    with pytest.raises(ValueError):
        verify_wikidata_derived_view(
            authority.source_lock_path,
            authority.source_root,
            ref.root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )
    with pytest.raises(ValueError):
        with _open_session(authority, ref):
            pass


# --- Descriptor-session read authority --------------------------------------


def test_forged_or_dataonly_reference_and_closed_session_cannot_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    authority = _authority_from_members(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {
            "wikidata5m_entity.txt": b"Q1\tOne\n",
            "wikidata5m_relation.txt": b"P1\tknows\n",
            "wikidata5m_inductive_train.txt": b"Q1\tP1\tQ2\n",
        },
    )
    ref = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )

    # The data-only reference never authorizes a read.
    for reader in (
        lambda v: list(iter_v2_training_triples(v)),
        lambda v: list(iter_v2_aliases(v)),
        lambda v: lookup_alias(v, "Q1"),
        lambda v: lookup_training_triple(v, "inductive_train", 1),
    ):
        with pytest.raises(ValueError):
            reader(ref)

    # A hand-constructed session object has no live retained descriptors.
    forged = object.__new__(WikidataDerivedView)
    for reader in (
        lambda v: list(iter_v2_aliases(v)),
        lambda v: lookup_training_triple(v, "inductive_train", 1),
    ):
        with pytest.raises(ValueError):
            reader(forged)

    # A live session reads; once the context closes every read is refused.
    with _open_session(authority, ref) as view:
        assert lookup_alias(view, "Q1").display == "One"
        assert list(iter_v2_training_triples(view))
    closed = view
    with pytest.raises(ValueError):
        list(iter_v2_aliases(closed))
    with pytest.raises(ValueError):
        lookup_training_triple(closed, "inductive_train", 1)


def test_concurrent_selected_file_mutation_is_detected_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    authority = _authority_from_members(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {
            "wikidata5m_entity.txt": b"Q1\tAlpha\nQ3\tGamma\n",
            "wikidata5m_relation.txt": b"P1\tknows\n",
            "wikidata5m_inductive_train.txt": b"Q1\tP1\tQ2\nQ3\tP1\tQ4\n",
        },
    )
    ref = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    with _open_session(authority, ref) as view:
        assert lookup_training_triple(view, "inductive_train", 1).object == 2

        training = ref.root / "streams" / "training.tsv"
        original = training.read_bytes()
        tampered = original.replace(b"Q1\tP1\tQ2", b"Q1\tP1\tQ8", 1)
        assert len(tampered) == len(original) and tampered != original

        original_pread = wikidata_source_module._pread
        state = {"mutated": False}

        def racing_pread(descriptor: int, count: int, offset: int) -> bytes:
            data = original_pread(descriptor, count, offset)
            if not state["mutated"]:
                state["mutated"] = True
                # Mutate the selected committed file in place after its bytes
                # are read but before the lookup returns.
                training.write_bytes(tampered)
            return data

        monkeypatch.setattr(wikidata_source_module, "_pread", racing_pread)
        with pytest.raises(ValueError):
            lookup_training_triple(view, "inductive_train", 1)
        assert state["mutated"] is True


def test_receipt_mode_link_and_view_root_name_drift_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    authority = _authority_from_members(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {
            "wikidata5m_entity.txt": b"Q1\tOne\n",
            "wikidata5m_relation.txt": b"P1\tknows\n",
        },
    )
    ref = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    receipt_path = ref.root / "receipt.json"

    # Relaxing the receipt's fixed mode is refused when opening a session.
    os.chmod(receipt_path, 0o644)
    with pytest.raises(ValueError):
        with _open_session(authority, ref):
            pass
    os.chmod(receipt_path, 0o600)

    # A second hard link to the receipt is refused.
    link_target = tmp_path / "receipt-link"
    os.link(receipt_path, link_target)
    with pytest.raises(ValueError):
        with _open_session(authority, ref):
            pass
    os.unlink(link_target)

    # The clean view still opens.
    with _open_session(authority, ref) as view:
        assert view.receipt_sha256 == ref.receipt_sha256

    # A reference whose claimed content address is not the view-root name fails.
    wrong = WikidataDerivedViewRef(root=ref.root, receipt_sha256="0" * 64)
    with pytest.raises(ValueError):
        with _open_session(authority, wrong):
            pass


def test_relaxed_committed_mode_or_hardlink_is_refused_on_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    authority = _authority_from_members(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {
            "wikidata5m_entity.txt": b"Q1\tOne\n",
            "wikidata5m_relation.txt": b"P1\tknows\n",
        },
    )
    ref = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    assert os.stat(ref.root).st_mode & 0o777 == 0o700
    for subdir in ("members", "streams", "indexes"):
        assert os.stat(ref.root / subdir).st_mode & 0o777 == 0o700
    assert os.stat(ref.root / "receipt.json").st_mode & 0o777 == 0o600
    assert os.stat(ref.root / "streams" / "training.tsv").st_mode & 0o777 == 0o600

    training = ref.root / "streams" / "training.tsv"
    os.chmod(training, 0o644)
    with pytest.raises(ValueError):
        with _open_session(authority, ref):
            pass
    with pytest.raises(ValueError):
        verify_wikidata_derived_view(
            authority.source_lock_path,
            authority.source_root,
            ref.root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )

    os.chmod(training, 0o600)
    link_target = tmp_path / "extra-hardlink"
    os.link(ref.root / "streams" / "aliases.tsv", link_target)
    with pytest.raises(ValueError):
        with _open_session(authority, ref):
            pass


# --- Output/source descriptor authority -------------------------------------


def test_build_output_must_be_outside_source_and_owner_controlled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    authority = _authority_from_members(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {},
    )
    # Publishing into the immutable source root (or a descendant) is refused
    # before any output is written.
    with pytest.raises(ValueError, match="source|inside"):
        build_wikidata_derived_view(
            authority.source_lock_path,
            authority.source_root,
            authority.source_root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )
    with pytest.raises(ValueError, match="source|inside"):
        build_wikidata_derived_view(
            authority.source_lock_path,
            authority.source_root,
            authority.source_root / "wikidata5m",
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )
    assert not (authority.source_root / "wikidata").exists()

    # A group/world-writable output namespace is refused.
    writable = tmp_path / "writable-out"
    writable.mkdir()
    os.chmod(writable, 0o777)
    with pytest.raises(ValueError, match="writable|owner"):
        build_wikidata_derived_view(
            authority.source_lock_path,
            authority.source_root,
            writable,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )


def test_output_name_substitution_between_check_and_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    authority = _authority_from_members(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {
            "wikidata5m_entity.txt": b"Q1\tOne\n",
            "wikidata5m_relation.txt": b"P1\tknows\n",
        },
    )
    output_root = tmp_path / "out"
    output_root.mkdir()

    state = {"substituted": False}

    def racing_hook(phase: str) -> None:
        if phase == "after_disjointness" and not state["substituted"]:
            state["substituted"] = True
            # Substitute the output name after the descriptor disjointness proof
            # but before the writes/publish resolve it again.
            output_root.rename(tmp_path / "out-substituted")
            (tmp_path / "out").mkdir()

    monkeypatch.setattr(wikidata_source_module, "_build_output_hook", racing_hook)
    with pytest.raises(ValueError, match="output root identity"):
        build_wikidata_derived_view(
            authority.source_lock_path,
            authority.source_root,
            output_root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )
    assert state["substituted"] is True


# --- Private external-sort scratch authority --------------------------------


def test_external_sort_scratch_lives_in_private_build_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    authority = _authority_from_members(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {
            "wikidata5m_entity.txt": b"Q1\tOne\n",
            "wikidata5m_relation.txt": b"P1\tknows\n",
            "wikidata5m_inductive_train.txt": b"Q1\tP1\tQ2\n",
        },
    )
    output_root = tmp_path / "out"
    captured: list[str] = []
    real_connect = wikidata_source_module.sqlite3.connect

    def capturing_connect(path, *args, **kwargs):
        captured.append(str(path))
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(
        wikidata_source_module.sqlite3, "connect", capturing_connect
    )
    build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        output_root,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    assert captured, "the build must open a SQLite scratch database"
    scratch_path = captured[0]
    # The scratch database lives inside the private build sibling under the
    # output namespace, never in a stand-alone process-global temp directory.
    private_prefix = str(output_root / "wikidata" / ".wikidata-view-build-")
    assert scratch_path.startswith(private_prefix)
    assert scratch_path.endswith("/.build-scratch/build.sqlite3")
    # The build creates no stand-alone directory directly in the process-global
    # temporary directory (the old, unsafe mkdtemp behavior).
    assert not any(
        name.startswith("wikidata-view-build-")
        for name in os.listdir(tempfile.gettempdir())
    )


def test_private_scratch_cleanup_failure_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    authority = _authority_from_members(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {
            "wikidata5m_entity.txt": b"Q1\tOne\n",
            "wikidata5m_relation.txt": b"P1\tknows\n",
        },
    )
    output_root = tmp_path / "out"
    build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        output_root,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )

    # On a winner-reuse build the losing private inode is quarantined and
    # removed; a cleanup failure there is reported rather than swallowed.
    real_remove = wikidata_source_module._remove_view_tree

    def failing_remove(parent_fd: int, name: str) -> None:
        if name.startswith(".wikidata-view-quarantine-"):
            raise OSError("injected quarantine cleanup failure")
        return real_remove(parent_fd, name)

    monkeypatch.setattr(wikidata_source_module, "_remove_view_tree", failing_remove)
    with pytest.raises((OSError, ValueError)):
        build_wikidata_derived_view(
            authority.source_lock_path,
            authority.source_root,
            output_root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )


# --- Boundary 4: bounded external-memory streaming build --------------------


def _scaled_members(count: int) -> dict[str, bytes]:
    entity = b"".join(
        f"Q{i}\tentity number {i}\tE{i}\n".encode("utf-8")
        for i in range(1, count + 1)
    )
    relation = b"P1\tknows\nP2\trelated to\n"
    inductive = b"".join(
        f"Q{i}\tP1\tQ{i + 5_000_000}\n".encode("utf-8")
        for i in range(1, count + 1)
    )
    transductive = b"".join(
        f"Q{i}\tP2\tQ{i + 6_000_000}\n".encode("utf-8")
        for i in range(1, count + 1)
    )
    return {
        "wikidata5m_entity.txt": entity,
        "wikidata5m_relation.txt": relation,
        "wikidata5m_inductive_train.txt": inductive,
        "wikidata5m_transductive_train.txt": transductive,
    }


def _peak_build_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
    count: int,
    tag: str,
) -> int:
    authority = _authority_from_members(
        tmp_path, monkeypatch, fixture_source_lock, _scaled_members(count)
    )
    output_root = tmp_path / f"bounded-{tag}"
    gc.collect()
    tracemalloc.start()
    try:
        ref = build_wikidata_derived_view(
            authority.source_lock_path,
            authority.source_root,
            output_root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The whole source scales, so training/alias counts must scale with it
    # (count entity aliases plus the two fixed relation aliases).
    receipt = _load_receipt(ref)
    assert receipt.training_rows == count * 2
    assert receipt.alias_rows == count + 2
    return peak


def test_build_uses_bounded_memory_on_scaled_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    base_count = 4000
    base_peak = _peak_build_bytes(
        tmp_path, monkeypatch, fixture_source_lock, base_count, "base"
    )
    quad_peak = _peak_build_bytes(
        tmp_path, monkeypatch, fixture_source_lock, base_count * 4, "quad"
    )
    # A streamed external-memory build keeps a sub-linear peak: a 4x larger
    # source must not even double the peak Python heap (an in-memory build
    # would grow ~4x). Streamed member decode plus a bounded temporary SQLite
    # external sort/dedup and incremental hashes make the growth near-constant.
    assert quad_peak < base_peak * 2, (base_peak, quad_peak)


# --- Boundary 5: malformed alias surfaces are rejected, not filtered --------


def test_build_rejects_alias_surface_with_nul_instead_of_filtering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    authority = _authority_from_members(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        {
            "wikidata5m_entity.txt": b"Q1\tbad\x00surface\n",
            "wikidata5m_relation.txt": b"P1\tknows\n",
        },
    )
    output_root = tmp_path / "out"
    # A NUL-bearing surface must fail the build rather than be silently dropped.
    with pytest.raises(ValueError, match="NUL"):
        build_wikidata_derived_view(
            authority.source_lock_path,
            authority.source_root,
            output_root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )
    namespace = output_root / "wikidata"
    if namespace.exists():
        assert not any(len(child.name) == 64 for child in namespace.iterdir())


# ---------------------------------------------------------------------------
# Task 3: production Wikidata catalog adapter (WikidataGraphCatalogSource)
# ---------------------------------------------------------------------------


def _wikidata_catalog_adapter(view: object):
    """Construct the production adapter, importing lazily so the absence of the
    symbol only fails the Task 3 tests and never the Task 1/2 collection."""

    from corpusgen.reasoning_v2.catalog import WikidataGraphCatalogSource

    return WikidataGraphCatalogSource(view)


def _build_base_view_ref(
    archive_authority: _AuthorityFixture,
    output_root: Path,
) -> WikidataDerivedViewRef:
    return build_wikidata_derived_view(
        archive_authority.source_lock_path,
        archive_authority.source_root,
        output_root,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )


def test_production_adapter_binds_view_archive_member_split_row_and_edge(
    tmp_path: Path,
    archive_authority: _AuthorityFixture,
):
    ref = _build_base_view_ref(archive_authority, tmp_path / "views")
    inductive_sha = hashlib.sha256(
        archive_authority.archive_payloads["wikidata5m_inductive.tar.gz"]
    ).hexdigest()
    transductive_sha = hashlib.sha256(
        archive_authority.archive_payloads["wikidata5m_transductive.tar.gz"]
    ).hexdigest()
    with open_wikidata_derived_view(
        archive_authority.source_lock_path,
        archive_authority.source_root,
        ref,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    ) as view:
        adapter = _wikidata_catalog_adapter(view)
        assert adapter.lane_id == "wikidata_graph"
        assert adapter.finite is True
        assert adapter.training_edge_count == 2
        assert adapter.wikidata_view_sha256 == ref.receipt_sha256
        assert list(
            adapter.iter_training_edge_keys(archive_authority.source_root)
        ) == ["Q1\tP1\tQ2", "Q3\tP2\tQ4"]
        drafts = list(adapter.iter_drafts(archive_authority.source_root, [1, 1]))
    assert [draft.lane_id for draft in drafts] == ["wikidata_graph", "wikidata_graph"]
    assert [draft.source_id for draft in drafts] == ["wikidata5m", "wikidata5m"]
    first, second = drafts
    assert dict(first.source_locator) == {
        "member": "wikidata5m_inductive_train.txt",
        "path": "wikidata5m_inductive.tar.gz",
        "row": 1,
        "split": "train",
        "training_edge_key": "Q1\tP1\tQ2",
        "training_split": "inductive_train",
        "wikidata_view_sha256": ref.receipt_sha256,
    }
    assert first.source_byte_sha256 == inductive_sha
    assert "graph-training-edge" in first.semantic_flags
    first_keys = [key for key, _value in first.source_locator]
    assert first_keys == sorted(first_keys, key=lambda text: text.encode("utf-8"))
    assert dict(second.source_locator) == {
        "member": "wikidata5m_transductive_train.txt",
        "path": "wikidata5m_transductive.tar.gz",
        "row": 1,
        "split": "train",
        "training_edge_key": "Q3\tP2\tQ4",
        "training_split": "transductive_train",
        "wikidata_view_sha256": ref.receipt_sha256,
    }
    assert second.source_byte_sha256 == transductive_sha
    assert len({draft.source_key for draft in drafts}) == 2


def test_distinct_edges_appear_once_before_first_revisit(
    tmp_path: Path,
    archive_authority: _AuthorityFixture,
):
    ref = _build_base_view_ref(archive_authority, tmp_path / "views")
    with open_wikidata_derived_view(
        archive_authority.source_lock_path,
        archive_authority.source_root,
        ref,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    ) as view:
        adapter = _wikidata_catalog_adapter(view)
        drafts = list(
            adapter.iter_drafts(archive_authority.source_root, [1, 1, 1, 1, 1])
        )
    assert len(drafts) == 5
    edge_positions = [
        index
        for index, draft in enumerate(drafts)
        if "graph-revisit" not in draft.semantic_flags
    ]
    revisit_positions = [
        index
        for index, draft in enumerate(drafts)
        if "graph-revisit" in draft.semantic_flags
    ]
    assert edge_positions == [0, 1]
    assert revisit_positions == [2, 3, 4]
    edge_keys = [
        dict(drafts[index].source_locator)["training_edge_key"]
        for index in edge_positions
    ]
    assert sorted(edge_keys) == ["Q1\tP1\tQ2", "Q3\tP2\tQ4"]
    assert len(set(edge_keys)) == 2
    authorized = set(edge_keys)
    for index in revisit_positions:
        assert (
            dict(drafts[index].source_locator)["training_edge_key"] in authorized
        )
    edge_source_keys = [drafts[index].source_key for index in edge_positions]
    revisit_source_keys = [drafts[index].source_key for index in revisit_positions]
    assert max(key.encode("utf-8") for key in edge_source_keys) < min(
        key.encode("utf-8") for key in revisit_source_keys
    )
    assert len({draft.source_key for draft in drafts}) == len(drafts)


def test_edge_count_above_available_records_fails_before_draft_output(
    tmp_path: Path,
    archive_authority: _AuthorityFixture,
):
    ref = _build_base_view_ref(archive_authority, tmp_path / "views")
    with open_wikidata_derived_view(
        archive_authority.source_lock_path,
        archive_authority.source_root,
        ref,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    ) as view:
        adapter = _wikidata_catalog_adapter(view)
        assert adapter.training_edge_count == 2
        emitted: list[object] = []
        with pytest.raises(
            ValueError,
            match="Wikidata distinct edges exceed allocated records",
        ):
            for draft in adapter.iter_drafts(archive_authority.source_root, [1]):
                emitted.append(draft)
        assert emitted == []


def test_data_only_reference_and_closed_session_cannot_authorize_adapter(
    tmp_path: Path,
    archive_authority: _AuthorityFixture,
):
    ref = _build_base_view_ref(archive_authority, tmp_path / "views")
    with pytest.raises(TypeError, match="WikidataDerivedView"):
        _wikidata_catalog_adapter(ref)
    with open_wikidata_derived_view(
        archive_authority.source_lock_path,
        archive_authority.source_root,
        ref,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    ) as view:
        assert _wikidata_catalog_adapter(view).training_edge_count == 2
    with pytest.raises(ValueError, match="closed|never opened"):
        _wikidata_catalog_adapter(view)


# ---------------------------------------------------------------------------
# Task 4: the verified view's indexed lookups drive the production renderer
# ---------------------------------------------------------------------------


def _empty_route_index(route_dir: Path):
    import sqlite3

    from corpusgen.reasoning_v2.semantic import RouteIndex

    route_dir.mkdir(parents=True, exist_ok=True)
    path = (route_dir / "routes.sqlite3").absolute()
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE selected ("
            "fact_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY"
            ") WITHOUT ROWID"
        )
        connection.commit()
    finally:
        connection.close()
    os.chmod(path, 0o600)
    return RouteIndex.open(path)


def test_view_indexed_lookups_render_wikidata_training_edge(
    tmp_path: Path,
    archive_authority: _AuthorityFixture,
):
    import json as _json

    from corpusgen.reasoning_v2.catalog import (
        CatalogDraft,
        CatalogRecord,
        catalog_record_id,
    )
    from corpusgen.reasoning_v2.renderers import WikidataGraphRenderer
    from train.tokenizer import get_tok

    ref = _build_base_view_ref(archive_authority, tmp_path / "views")
    inductive_sha = hashlib.sha256(
        archive_authority.archive_payloads["wikidata5m_inductive.tar.gz"]
    ).hexdigest()
    routes = _empty_route_index(tmp_path / "routes")
    try:
        with open_wikidata_derived_view(
            archive_authority.source_lock_path,
            archive_authority.source_root,
            ref,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        ) as view:
            triple = lookup_training_triple(view, "inductive_train", 1)
            assert (triple.subject, triple.relation, triple.object) == (1, "P1", 2)
            assert triple.member == "wikidata5m_inductive_train.txt"
            assert triple.archive_path == "wikidata5m_inductive.tar.gz"
            assert lookup_alias(view, "Q1").display == "Ada Lovelace"
            assert lookup_alias(view, "P1").display == "knows"
            assert lookup_alias(view, "Q2") is None

            locator = tuple(
                sorted(
                    (
                        ("member", "wikidata5m_inductive_train.txt"),
                        ("path", "wikidata5m_inductive.tar.gz"),
                        ("row", 1),
                        ("split", "train"),
                        ("training_edge_key", "Q1\tP1\tQ2"),
                        ("training_split", "inductive_train"),
                        ("wikidata_view_sha256", ref.receipt_sha256),
                    ),
                    key=lambda item: item[0].encode("utf-8"),
                )
            )
            draft = CatalogDraft(
                lane_id="wikidata_graph",
                source_id="wikidata5m",
                source_key="wikidata-edge-000000000000",
                source_byte_sha256=inductive_sha,
                source_locator=locator,
                semantic_flags=("graph-training-edge",),
                semantic_facts=(),
            )
            target_count = 96
            record = CatalogRecord(
                ordinal=0,
                record_id=catalog_record_id(draft, target_count),
                lane_id="wikidata_graph",
                source_id="wikidata5m",
                source_key="wikidata-edge-000000000000",
                source_byte_sha256=inductive_sha,
                source_locator=locator,
                target_count=target_count,
                semantic_flags=("graph-training-edge",),
                semantic_facts=(),
            )
            rendered = WikidataGraphRenderer(
                archive_authority.source_root, view
            ).render(record, routes)

            expected = _json.dumps(
                {
                    "object": "Q2",
                    "object_label": "Q2",
                    "relation": "P1",
                    "relation_label": "knows",
                    "split": "inductive_train",
                    "subject": "Q1",
                    "subject_label": "Ada Lovelace",
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
            tok = get_tok()
            core_end = rendered.semantic_spans[0].token_end
            assert tok.decode(list(rendered.token_ids[:core_end])) == expected
            assert rendered.token_ids[-1] == tok.EOT
            assert len(rendered.token_ids) == target_count
            assert rendered.semantic_leaks == ()
            assert rendered.dense_target_weights == b"\x01" * target_count
            assert rendered.split90_target_weights == b"\x01" * target_count
    finally:
        routes.close()
