from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import os
import shutil
import struct
import tarfile
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
    _alias_record_from_row,
    _encode_alias_row,
    _open_verified_archives,
    build_wikidata_derived_view,
    iter_distinct_training_edges,
    iter_v2_aliases,
    iter_v2_training_triples,
    lookup_alias,
    lookup_training_triple,
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
    view = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    assert all(
        isinstance(record, IndexArtifactRecord) for record in view.receipt.indexes
    )
    indexes = {record.path: record for record in view.receipt.indexes}
    inductive = indexes["indexes/inductive-training-offsets.bin"]
    transductive = indexes["indexes/transductive-training-offsets.bin"]
    aliases = indexes["indexes/aliases.bin"]
    assert (inductive.record_width, inductive.count) == (8, 2)
    assert (transductive.record_width, transductive.count) == (8, 1)
    assert (aliases.record_width, aliases.count) == (24, 3)
    assert inductive.count + transductive.count == view.receipt.training_rows == 3
    assert aliases.count == view.receipt.alias_rows == 3
    for record in view.receipt.indexes:
        assert record.count * record.record_width == record.bytes

    # The closed receipt round-trips with the index count/width fields.
    assert (
        WikidataDerivedViewReceipt.from_bytes(view.receipt.to_bytes())
        == view.receipt
    )
    on_disk = json.loads((view.root / "receipt.json").read_text(encoding="utf-8"))
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

    assert first.receipt_sha256 == second.receipt_sha256
    assert first.root.name == first.receipt_sha256
    assert first.root.parent.name == "wikidata"
    assert _view_tree(first.root) == _view_tree(second.root)

    receipt_bytes = (first.root / "receipt.json").read_bytes()
    assert hashlib.sha256(receipt_bytes).hexdigest() == first.receipt_sha256
    assert first.receipt.to_bytes() == receipt_bytes
    # The content address must not depend on machine-local values.
    text = receipt_bytes.decode("utf-8")
    assert str(tmp_path) not in text
    inventory = {
        record.path
        for record in (
            first.receipt.members
            + first.receipt.streams
            + first.receipt.indexes
        )
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
    view = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )

    training = list(iter_v2_training_triples(view))
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

    training_tsv = (view.root / "streams" / "training.tsv").read_bytes()
    assert training_tsv == (
        b"inductive_train\t1\tQ5\tP3\tQ6\n"
        b"inductive_train\t2\tQ1\tP1\tQ2\n"
        b"transductive_train\t1\tQ3\tP2\tQ4\n"
    )

    aliases = list(iter_v2_aliases(view))
    assert [(a.canonical_id, a.kind) for a in aliases] == [
        ("Q2", "entity"),
        ("Q10", "entity"),
        ("P1", "relation"),
    ]

    distinct = list(iter_distinct_training_edges(view))
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
    view = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )

    aliases = {a.canonical_id: a for a in iter_v2_aliases(view)}
    # "Shared"/"shared" collapse to the same normalized surface form owned by
    # Q1, Q2, and P1, so it is removed from every record globally.
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
    view = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )

    training_tsv = (view.root / "streams" / "training.tsv").read_bytes()
    aliases_tsv = (view.root / "streams" / "aliases.tsv").read_bytes()

    # The view is self-contained: destroy the source tree, then every read
    # path must still succeed using only decoded view files (no archive rescan).
    shutil.rmtree(authority.source_root)

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

    with pytest.raises(ValueError):
        build_wikidata_derived_view(
            authority.source_lock_path,
            authority.source_root,
            output_root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )
    # A corrupt winner is never silently repaired in place.
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
    view = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )

    # A clean view verifies.
    verify_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        view.root,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )

    stream = view.root / "streams" / "aliases.tsv"
    original = stream.read_bytes()
    stream.write_bytes(original[:-1] + bytes((original[-1] ^ 1,)))
    with pytest.raises(ValueError):
        verify_wikidata_derived_view(
            authority.source_lock_path,
            authority.source_root,
            view.root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )
    with pytest.raises(ValueError):
        list(iter_v2_aliases(view))
    stream.write_bytes(original)

    index = view.root / "indexes" / "aliases.bin"
    index_bytes = bytearray(index.read_bytes())
    index_bytes[-1] ^= 1
    index.write_bytes(bytes(index_bytes))
    with pytest.raises(ValueError):
        verify_wikidata_derived_view(
            authority.source_lock_path,
            authority.source_root,
            view.root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )


# --- Boundary 2: non-forgeable verified-view read authority ------------------


def _forge_unverified_view(view: WikidataDerivedView) -> WikidataDerivedView:
    forged = object.__new__(WikidataDerivedView)
    object.__setattr__(forged, "root", view.root)
    object.__setattr__(forged, "receipt_sha256", view.receipt_sha256)
    object.__setattr__(forged, "receipt", view.receipt)
    object.__setattr__(forged, "_authority", None)
    return forged


def test_public_reads_require_a_minted_verified_view_handle(
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
    view = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    # A legitimately minted handle reads normally.
    assert lookup_alias(view, "Q1").display == "One"
    assert list(iter_v2_aliases(view))

    # Constructing the public dataclass by hand cannot mint authority.
    with pytest.raises(ValueError):
        WikidataDerivedView(
            root=view.root,
            receipt_sha256=view.receipt_sha256,
            receipt=view.receipt,
        )

    # A view whose minted authority is stripped is refused by every read,
    # even though its on-disk directory and receipt are byte-perfect.
    forged = _forge_unverified_view(view)
    with pytest.raises(ValueError):
        list(iter_v2_training_triples(forged))
    with pytest.raises(ValueError):
        list(iter_v2_aliases(forged))
    with pytest.raises(ValueError):
        lookup_alias(forged, "Q1")
    with pytest.raises(ValueError):
        lookup_training_triple(forged, "inductive_train", 1)


def test_lookup_rejects_committed_stream_and_index_tampering(
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
    view = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    assert lookup_training_triple(view, "inductive_train", 1).object == 2

    # A same-length, still-canonical in-place edit of a committed row must not
    # be able to change a returned value while the receipt hash is unchecked.
    training = view.root / "streams" / "training.tsv"
    original = training.read_bytes()
    tampered = original.replace(b"Q1\tP1\tQ2", b"Q1\tP1\tQ8", 1)
    assert len(tampered) == len(original) and tampered != original
    training.write_bytes(tampered)
    with pytest.raises(ValueError):
        lookup_training_triple(view, "inductive_train", 1)

    # A corrupt/unsorted alias index must fail rather than return a false None.
    index = view.root / "indexes" / "aliases.bin"
    corrupt = bytearray(index.read_bytes())
    corrupt[0] ^= 0xFF
    index.write_bytes(bytes(corrupt))
    with pytest.raises(ValueError):
        lookup_alias(view, "Q1")


# --- Boundary 3: output/source disjointness and filesystem authority --------


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


def test_published_view_uses_fixed_owner_controlled_modes(
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
    view = build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        tmp_path / "out",
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    assert os.stat(view.root).st_mode & 0o777 == 0o700
    for subdir in ("members", "streams", "indexes"):
        assert os.stat(view.root / subdir).st_mode & 0o777 == 0o700
    assert os.stat(view.root / "receipt.json").st_mode & 0o777 == 0o600
    assert os.stat(view.root / "streams" / "training.tsv").st_mode & 0o777 == 0o600

    # Relaxing a committed file's fixed mode is refused by verified reads.
    training = view.root / "streams" / "training.tsv"
    os.chmod(training, 0o644)
    with pytest.raises(ValueError):
        lookup_training_triple(view, "inductive_train", 1)
    with pytest.raises(ValueError):
        verify_wikidata_derived_view(
            authority.source_lock_path,
            authority.source_root,
            view.root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )

    # A second hard link to a committed file is refused.
    os.chmod(training, 0o600)
    link_target = tmp_path / "extra-hardlink"
    os.link(view.root / "streams" / "aliases.tsv", link_target)
    with pytest.raises(ValueError):
        list(iter_v2_aliases(view))
