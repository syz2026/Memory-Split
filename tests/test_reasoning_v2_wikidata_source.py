from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import os
import shutil
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
    WikidataDerivedViewReceipt,
    _open_verified_archives,
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
    *,
    count: int = 0,
    record_width: int,
) -> dict[str, object]:
    return {
        **_artifact(path, size=count * record_width),
        "count": count,
        "record_width": record_width,
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
            _index_artifact("indexes/aliases.bin", record_width=24),
            _index_artifact(
                "indexes/inductive-training-offsets.bin",
                record_width=8,
            ),
            _index_artifact(
                "indexes/transductive-training-offsets.bin",
                record_width=8,
            ),
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
    assert tuple(
        (record.count, record.record_width) for record in receipt.indexes
    ) == ((0, 24), (0, 8), (0, 8))

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
    open_index = copy.deepcopy(valid)
    open_index["indexes"][0]["future_field"] = "open"
    malformed.append(canonical_json_bytes(open_index))
    missing_index_count = copy.deepcopy(valid)
    missing_index_count["indexes"][0].pop("count")
    malformed.append(canonical_json_bytes(missing_index_count))
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


def test_archive_authority_hashes_and_parses_the_same_descriptor(
    archive_authority: _AuthorityFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    with _open_verified_archives(
        archive_authority.source_lock_path,
        archive_authority.source_root,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
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
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
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
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
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
                expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
            ):
                pass
        return

    with pytest.raises(ValueError, match="identity drift|changed during verification"):
        with _open_verified_archives(
            archive_authority.source_lock_path,
            archive_authority.source_root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
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
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
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
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
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
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
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
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
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
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    ) as verified:
        assert len(verified.archives) == 3
        assert len(verified.members) == 8
    assert _source_snapshot(archive_authority.source_root) == before


def _authority_from_entries(
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


def _replace_member(
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


def _build_view(authority: _AuthorityFixture, output_root: Path):
    return wikidata_source_module.build_wikidata_derived_view(
        authority.source_lock_path,
        authority.source_root,
        output_root,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )


def _rewrite_source_lock_generator(
    authority: _AuthorityFixture,
    generator_commit: str,
) -> None:
    value = json.loads(
        authority.source_lock_path.read_text(encoding="utf-8")
    )
    value["generator_commit"] = generator_commit
    authority.source_lock_path.write_bytes(canonical_json_bytes(value))


def _published_view_names(output_root: Path) -> tuple[str, ...]:
    namespace = output_root / "wikidata"
    if not namespace.exists():
        return ()
    return tuple(
        sorted(
            path.name
            for path in namespace.iterdir()
            if len(path.name) == 64
            and all(character in "0123456789abcdef" for character in path.name)
        )
    )


def _view_payloads(view) -> dict[str, bytes]:
    return {
        path: (view.root / path).read_bytes()
        for path in (
            "receipt.json",
            "streams/training.tsv",
            "streams/aliases.tsv",
            "streams/distinct-edges.tsv",
            "indexes/inductive-training-offsets.bin",
            "indexes/transductive-training-offsets.bin",
            "indexes/aliases.bin",
        )
    }


def _alias_stream_row(
    canonical_id: str,
    display: str,
    aliases: list[str],
) -> bytes:
    return (
        canonical_id.encode("utf-8")
        + b"\t"
        + canonical_json_bytes({"aliases": aliases, "display": display})
    )


def test_repeated_builds_produce_identical_receipt_streams_and_indexes(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source_before = _source_snapshot(archive_authority.source_root)

    first = _build_view(archive_authority, tmp_path / "derived-a")
    monkeypatch.setattr(wikidata_source_module, "_SORT_CHUNK_BYTES", 64)
    monkeypatch.setattr(wikidata_source_module, "_SORT_CHUNK_RECORDS", 1)
    monkeypatch.setattr(wikidata_source_module, "_SORT_MERGE_FAN_IN", 2)
    second = _build_view(archive_authority, tmp_path / "derived-b")

    assert first.receipt_sha256 == second.receipt_sha256
    assert first.root == (
        tmp_path / "derived-a" / "wikidata" / first.receipt_sha256
    )
    assert second.root == (
        tmp_path / "derived-b" / "wikidata" / second.receipt_sha256
    )
    assert _view_payloads(first) == _view_payloads(second)
    assert tuple(
        (record.path, record.count, record.record_width)
        for record in first.receipt.indexes
    ) == (
        ("indexes/aliases.bin", 2, 24),
        ("indexes/inductive-training-offsets.bin", 1, 8),
        ("indexes/transductive-training-offsets.bin", 1, 8),
    )
    assert _source_snapshot(archive_authority.source_root) == source_before


def test_training_and_alias_order_matches_frozen_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    entries = _base_archive_entries()
    _replace_member(
        entries,
        "wikidata5m_alias.tar.gz",
        "wikidata5m_entity.txt",
        (
            b"Q10\tTen\n"
            b"Q2\t  Second   Entity \tAlias Two\n"
        ),
    )
    _replace_member(
        entries,
        "wikidata5m_alias.tar.gz",
        "wikidata5m_relation.txt",
        b"P10\tRelation Ten\nP2\tRelation Two\n",
    )
    _replace_member(
        entries,
        "wikidata5m_inductive.tar.gz",
        "wikidata5m_inductive_train.txt",
        b"Q10\tP2\tQ3\nQ2\tP10\tQ1\n",
    )
    _replace_member(
        entries,
        "wikidata5m_transductive.tar.gz",
        "wikidata5m_transductive_train.txt",
        b"Q1\tP1\tQ4\n",
    )
    authority = _authority_from_entries(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        entries,
    )

    view = _build_view(authority, tmp_path / "derived")

    assert (view.root / "streams/training.tsv").read_bytes() == (
        b"inductive_train\t1\tQ10\tP2\tQ3\n"
        b"inductive_train\t2\tQ2\tP10\tQ1\n"
        b"transductive_train\t1\tQ1\tP1\tQ4\n"
    )
    assert (view.root / "streams/aliases.tsv").read_bytes() == b"".join(
        (
            _alias_stream_row(
                "Q2",
                "Second Entity",
                ["Second Entity", "Alias Two"],
            ),
            _alias_stream_row("Q10", "Ten", ["Ten"]),
            _alias_stream_row("P2", "Relation Two", ["Relation Two"]),
            _alias_stream_row("P10", "Relation Ten", ["Relation Ten"]),
        )
    )
    assert (view.root / "streams/distinct-edges.tsv").read_bytes() == (
        b"transductive_train\t1\tQ1\tP1\tQ4\n"
        b"inductive_train\t2\tQ2\tP10\tQ1\n"
        b"inductive_train\t1\tQ10\tP2\tQ3\n"
    )
    assert [
        (
            record.training_split,
            record.row,
            record.subject,
            record.relation,
            record.object,
        )
        for record in wikidata_source_module.iter_v2_training_triples(view)
    ] == [
        ("inductive_train", 1, 10, "P2", 3),
        ("inductive_train", 2, 2, "P10", 1),
        ("transductive_train", 1, 1, "P1", 4),
    ]
    assert [
        (record.canonical_id, record.kind, record.display, record.aliases)
        for record in wikidata_source_module.iter_v2_aliases(view)
    ] == [
        ("Q2", "entity", "Second Entity", ("Second Entity", "Alias Two")),
        ("Q10", "entity", "Ten", ("Ten",)),
        ("P2", "relation", "Relation Two", ("Relation Two",)),
        ("P10", "relation", "Relation Ten", ("Relation Ten",)),
    ]


def test_alias_ambiguity_is_removed_globally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
):
    entries = _base_archive_entries()
    _replace_member(
        entries,
        "wikidata5m_alias.tar.gz",
        "wikidata5m_entity.txt",
        (
            "Q1\t Shared \tUnique One\t unique   one \tＳＯＬＯ\n"
            "Q2\tshared\tUnique Two\n"
            "Q3\tshared\n"
        ).encode("utf-8"),
    )
    _replace_member(
        entries,
        "wikidata5m_alias.tar.gz",
        "wikidata5m_relation.txt",
        b"P1\tSHARED\trelation-only\n",
    )
    authority = _authority_from_entries(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        entries,
    )

    view = _build_view(authority, tmp_path / "derived")

    assert [
        (record.canonical_id, record.display, record.aliases)
        for record in wikidata_source_module.iter_v2_aliases(view)
    ] == [
        ("Q1", "Unique One", ("Unique One", "SOLO")),
        ("Q2", "Unique Two", ("Unique Two",)),
        ("Q3", "Q3", ()),
        ("P1", "relation-only", ("relation-only",)),
    ]


def test_indexed_lookup_matches_streaming_without_archive_rescan(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    view = _build_view(archive_authority, tmp_path / "derived")
    triples = list(wikidata_source_module.iter_v2_training_triples(view))
    aliases = {
        record.canonical_id: record
        for record in wikidata_source_module.iter_v2_aliases(view)
    }

    def reject_archive_rescan(*_args, **_kwargs):
        raise AssertionError("lookup reopened a source archive")

    monkeypatch.setattr(
        wikidata_source_module,
        "_open_verified_archives",
        reject_archive_rescan,
    )

    for triple in triples:
        assert wikidata_source_module.lookup_training_triple(
            view,
            triple.training_split,
            triple.row,
        ) == triple
    for canonical_id, alias in aliases.items():
        assert (
            wikidata_source_module.lookup_alias(view, canonical_id)
            == alias
        )
    assert wikidata_source_module.lookup_alias(view, "Q999999") is None

    caller_constructed = wikidata_source_module.WikidataDerivedView(
        root=view.root,
        receipt_sha256=view.receipt_sha256,
        receipt=view.receipt,
    )
    with pytest.raises(ValueError, match="verified"):
        wikidata_source_module.lookup_alias(caller_constructed, "Q1")


@pytest.mark.parametrize("attack", ["overlap", "malformed"])
def test_train_sealed_overlap_and_malformed_rows_fail_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_source_lock: FixtureSourceLock,
    attack: str,
):
    entries = _base_archive_entries()
    if attack == "overlap":
        _replace_member(
            entries,
            "wikidata5m_inductive.tar.gz",
            "wikidata5m_inductive_test.txt",
            b"Q1\tP1\tQ2\n",
        )
    else:
        _replace_member(
            entries,
            "wikidata5m_inductive.tar.gz",
            "wikidata5m_inductive_train.txt",
            b"Q1\tP1\n",
        )
    authority = _authority_from_entries(
        tmp_path,
        monkeypatch,
        fixture_source_lock,
        entries,
    )
    output_root = tmp_path / "derived"

    with pytest.raises(ValueError, match="overlap|tab-separated"):
        _build_view(authority, output_root)

    wikidata_root = output_root / "wikidata"
    assert not wikidata_root.exists() or not tuple(wikidata_root.iterdir())


def test_no_replace_publication_reuses_only_a_fully_verified_winner(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
):
    output_root = tmp_path / "derived"
    first = _build_view(archive_authority, output_root)
    first_identity = first.root.stat().st_ino

    reused = _build_view(archive_authority, output_root)

    assert reused.root == first.root
    assert reused.root.stat().st_ino == first_identity
    assert tuple(path.name for path in (output_root / "wikidata").iterdir()) == (
        first.receipt_sha256,
    )

    index_path = first.root / "indexes/aliases.bin"
    attacked = bytearray(index_path.read_bytes())
    attacked[-1] ^= 1
    index_path.write_bytes(attacked)
    with pytest.raises(ValueError, match="drift|digest|index"):
        _build_view(archive_authority, output_root)


@pytest.mark.parametrize(
    "relative_path",
    ["streams/training.tsv", "indexes/aliases.bin"],
)
def test_stream_or_index_drift_fails_verification(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    relative_path: str,
):
    view = _build_view(
        archive_authority,
        tmp_path / relative_path.split("/", 1)[0],
    )
    attacked_path = view.root / relative_path
    attacked = bytearray(attacked_path.read_bytes())
    attacked[len(attacked) // 2] ^= 1
    attacked_path.write_bytes(attacked)

    with pytest.raises(ValueError, match="drift|digest|canonical|index"):
        wikidata_source_module.verify_wikidata_derived_view(
            archive_authority.source_lock_path,
            archive_authority.source_root,
            view.root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )


def test_build_rejects_mismatched_source_lock_generator_before_output(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
):
    _rewrite_source_lock_generator(archive_authority, "b" * 40)
    output_root = tmp_path / "derived"

    with pytest.raises(ValueError, match="generator commit"):
        _build_view(archive_authority, output_root)

    assert not output_root.exists()


def test_verify_rejects_stale_source_lock_generator_authority(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
):
    view = _build_view(archive_authority, tmp_path / "derived")
    _rewrite_source_lock_generator(archive_authority, "b" * 40)

    with pytest.raises(ValueError, match="generator commit"):
        wikidata_source_module.verify_wikidata_derived_view(
            archive_authority.source_lock_path,
            archive_authority.source_root,
            view.root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        )


def test_build_directory_swap_fails_before_final_target_occupation(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_root = tmp_path / "derived"
    swapped = False

    def swap_before_publish(phase, authority, _final_name):
        nonlocal swapped
        if phase != "before_publish_check" or swapped:
            return
        swapped = True
        namespace = output_root / "wikidata"
        original = namespace / authority.name
        original.rename(namespace / f"{authority.name}.displaced")
        original.mkdir(mode=0o700)

    monkeypatch.setattr(
        wikidata_source_module,
        "_derived_view_build_hook",
        swap_before_publish,
        raising=False,
    )

    with pytest.raises(ValueError, match="identity drift"):
        _build_view(archive_authority, output_root)

    assert swapped
    assert _published_view_names(output_root) == ()


def test_external_sort_run_swap_fails_closed_and_closes_opened_descriptor(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_root = tmp_path / "derived"
    opened_run_fds: list[int] = []
    swapped = False
    original_open = wikidata_source_module.open_regular_file_at

    def record_run_open(directory_fd, name):
        descriptor, metadata = original_open(directory_fd, name)
        if "-l" in name and name.endswith(".bin"):
            opened_run_fds.append(descriptor)
        return descriptor, metadata

    def swap_run(phase, work_fd, run):
        nonlocal swapped
        if phase != "before_open" or swapped:
            return
        swapped = True
        os.rename(
            run.name,
            f"{run.name}.displaced",
            src_dir_fd=work_fd,
            dst_dir_fd=work_fd,
        )
        replacement = os.open(
            run.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=work_fd,
        )
        os.write(replacement, b"substituted run")
        os.close(replacement)

    monkeypatch.setattr(
        wikidata_source_module,
        "open_regular_file_at",
        record_run_open,
    )
    monkeypatch.setattr(
        wikidata_source_module,
        "_external_sort_hook",
        swap_run,
        raising=False,
    )
    monkeypatch.setattr(wikidata_source_module, "_SORT_CHUNK_RECORDS", 1)

    with pytest.raises(ValueError, match="identity drift"):
        _build_view(archive_authority, output_root)

    assert swapped
    assert opened_run_fds
    for descriptor in opened_run_fds:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert _published_view_names(output_root) == ()


@pytest.mark.parametrize(
    ("consumer", "directory_name"),
    [
        ("lookup", "indexes"),
        ("iteration", "streams"),
        ("iteration", "members"),
    ],
)
def test_public_consumers_postcheck_child_directory_identity(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consumer: str,
    directory_name: str,
):
    view = _build_view(archive_authority, tmp_path / "derived")
    swapped = False

    def swap_child(phase, hooked_view):
        nonlocal swapped
        if phase != "before_postcheck" or swapped:
            return
        swapped = True
        child = hooked_view.root / directory_name
        child.rename(hooked_view.root / f"{directory_name}.displaced")
        child.mkdir(mode=0o700)

    monkeypatch.setattr(
        wikidata_source_module,
        "_view_authority_hook",
        swap_child,
        raising=False,
    )

    with pytest.raises(ValueError, match="identity drift"):
        if consumer == "lookup":
            wikidata_source_module.lookup_alias(view, "Q1")
        else:
            list(wikidata_source_module.iter_v2_aliases(view))

    assert swapped


def test_cleanup_never_deletes_substituted_private_build_directory(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_root = tmp_path / "derived"
    replacement: Path | None = None
    displaced: Path | None = None
    captured_fd = -1

    def fail_then_swap(phase, authority, _final_name):
        nonlocal replacement, displaced, captured_fd
        captured_fd = authority.descriptor
        if phase == "before_candidate_verify":
            raise RuntimeError("primary build failure")
        if phase == "before_cleanup":
            namespace = output_root / "wikidata"
            replacement = namespace / authority.name
            displaced = namespace / f"{authority.name}.displaced"
            replacement.rename(displaced)
            replacement.mkdir(mode=0o700)
            (replacement / "marker").write_bytes(b"do not delete")

    monkeypatch.setattr(
        wikidata_source_module,
        "_derived_view_build_hook",
        fail_then_swap,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="primary build failure"):
        _build_view(archive_authority, output_root)

    assert replacement is not None and (replacement / "marker").is_file()
    assert displaced is not None and displaced.is_dir()
    with pytest.raises(OSError):
        os.fstat(captured_fd)
    assert _published_view_names(output_root) == ()


def test_cleanup_failure_preserves_primary_error_and_closes_all_descriptors(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured_fds: list[int] = []

    def fail_build(phase, authority, _final_name):
        if phase == "before_candidate_verify":
            captured_fds.append(authority.descriptor)
            captured_fds.extend(authority.directory_descriptors.values())
            raise RuntimeError("primary build failure")

    def fail_cleanup(_authority):
        raise OSError("cleanup failure")

    monkeypatch.setattr(
        wikidata_source_module,
        "_derived_view_build_hook",
        fail_build,
        raising=False,
    )
    monkeypatch.setattr(
        wikidata_source_module,
        "_cleanup_private_build",
        fail_cleanup,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="primary build failure") as raised:
        _build_view(archive_authority, tmp_path / "derived")

    assert "cleanup failure" not in str(raised.value)
    assert captured_fds
    for descriptor in captured_fds:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_published_target_inode_swap_fails_before_content_verification(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_root = tmp_path / "derived"
    swapped = False

    def swap_published_target(phase, authority, final_name):
        nonlocal swapped
        if phase != "after_publish_rename" or swapped:
            return
        swapped = True
        namespace = output_root / "wikidata"
        target = namespace / final_name
        displaced = namespace / f"{final_name}.displaced"
        target.rename(displaced)
        shutil.copytree(displaced, target)
        assert target.stat().st_ino != authority.identity[1]

    monkeypatch.setattr(
        wikidata_source_module,
        "_derived_view_build_hook",
        swap_published_target,
        raising=False,
    )

    with pytest.raises(ValueError, match="identity drift"):
        _build_view(archive_authority, output_root)

    assert swapped


def _modify_restore_path_aba(path: Path) -> int:
    metadata = path.stat()
    payload = path.read_bytes()
    assert payload
    attacked = bytearray(payload)
    attacked[len(attacked) // 2] ^= 1
    with path.open("r+b") as stream:
        stream.write(attacked)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
        stream.seek(0)
        stream.write(payload)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o400)
        os.fchmod(stream.fileno(), 0o600)
    os.utime(
        path,
        ns=(metadata.st_atime_ns, metadata.st_mtime_ns),
        follow_symlinks=False,
    )
    assert path.stat().st_ino == metadata.st_ino
    assert path.read_bytes() == payload
    return metadata.st_ino


def _modify_restore_at_aba(directory_fd: int, name: str) -> int:
    descriptor = os.open(name, os.O_RDWR, dir_fd=directory_fd)
    try:
        metadata = os.fstat(descriptor)
        payload = os.pread(descriptor, metadata.st_size, 0)
        assert payload
        attacked = bytearray(payload)
        attacked[len(attacked) // 2] ^= 1
        os.pwrite(descriptor, attacked, 0)
        os.ftruncate(descriptor, len(attacked))
        os.fsync(descriptor)
        os.pwrite(descriptor, payload, 0)
        os.ftruncate(descriptor, len(payload))
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)
    os.utime(
        name,
        ns=(metadata.st_atime_ns, metadata.st_mtime_ns),
        dir_fd=directory_fd,
        follow_symlinks=False,
    )
    restored = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    assert restored.st_ino == metadata.st_ino
    return metadata.st_ino


def test_materialized_member_same_inode_modify_restore_aba_fails(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_root = tmp_path / "derived"
    attacked = False

    def attack_member(phase, authority, _final_name):
        nonlocal attacked
        if phase != "after_members" or attacked:
            return
        attacked = True
        _modify_restore_path_aba(
            output_root
            / "wikidata"
            / authority.name
            / "members"
            / "wikidata5m_inductive_train.txt"
        )

    monkeypatch.setattr(
        wikidata_source_module,
        "_derived_view_build_hook",
        attack_member,
    )

    with pytest.raises(ValueError, match="identity|digest|ABA"):
        _build_view(archive_authority, output_root)

    assert attacked
    assert _published_view_names(output_root) == ()


def test_external_sort_run_in_place_modify_restore_aba_fails(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_root = tmp_path / "derived"
    attacked = False

    def attack_run(phase, work_fd, run):
        nonlocal attacked
        if phase != "before_open" or attacked:
            return
        attacked = True
        _modify_restore_at_aba(work_fd, run.name)

    monkeypatch.setattr(
        wikidata_source_module,
        "_external_sort_hook",
        attack_run,
    )
    monkeypatch.setattr(wikidata_source_module, "_SORT_CHUNK_RECORDS", 1)

    with pytest.raises(ValueError, match="identity|digest|ABA"):
        _build_view(archive_authority, output_root)

    assert attacked
    assert _published_view_names(output_root) == ()


def test_private_file_authority_binds_full_identity_and_sha256(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_checked = False
    candidate_checked = False

    def check_run(phase, _work_fd, run):
        nonlocal run_checked
        if phase == "before_open":
            run_checked = True
            assert len(run.identity) == 9
            assert len(run.sha256) == 64

    def check_candidate(phase, authority, _final_name):
        nonlocal candidate_checked
        if phase != "before_publish_check":
            return
        candidate_checked = True
        assert not authority.pending_file_identities
        assert "receipt.json" in authority.file_authorities
        assert any(
            path.startswith("members/")
            for path in authority.file_authorities
        )
        assert any(
            path.startswith("streams/")
            for path in authority.file_authorities
        )
        assert any(
            path.startswith("indexes/")
            for path in authority.file_authorities
        )
        for file_authority in authority.file_authorities.values():
            assert len(file_authority.identity) == 9
            assert len(file_authority.sha256) == 64

    monkeypatch.setattr(
        wikidata_source_module,
        "_external_sort_hook",
        check_run,
    )
    monkeypatch.setattr(
        wikidata_source_module,
        "_derived_view_build_hook",
        check_candidate,
    )
    monkeypatch.setattr(wikidata_source_module, "_SORT_CHUNK_RECORDS", 1)

    _build_view(archive_authority, tmp_path / "derived")

    assert run_checked
    assert candidate_checked


@pytest.mark.parametrize(
    "relative_path",
    [
        "streams/training.tsv",
        "indexes/aliases.bin",
        "receipt.json",
    ],
)
def test_candidate_file_drift_after_verification_fails_before_publish(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
):
    output_root = tmp_path / "derived"
    attacked = False

    def attack_candidate(phase, authority, _final_name):
        nonlocal attacked
        if phase != "before_publish_check" or attacked:
            return
        attacked = True
        target = output_root / "wikidata" / authority.name / relative_path
        payload = bytearray(target.read_bytes())
        payload[len(payload) // 2] ^= 1
        target.write_bytes(payload)

    monkeypatch.setattr(
        wikidata_source_module,
        "_derived_view_build_hook",
        attack_candidate,
    )

    with pytest.raises(ValueError, match="identity|digest|drift"):
        _build_view(archive_authority, output_root)

    assert attacked
    assert _published_view_names(output_root) == ()


def test_candidate_child_directory_drift_after_verification_fails_before_publish(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_root = tmp_path / "derived"
    attacked = False

    def attack_child(phase, authority, _final_name):
        nonlocal attacked
        if phase != "before_publish_check" or attacked:
            return
        attacked = True
        root = output_root / "wikidata" / authority.name
        child = root / "streams"
        displaced = root / "streams.displaced"
        child.rename(displaced)
        shutil.copytree(displaced, child)

    monkeypatch.setattr(
        wikidata_source_module,
        "_derived_view_build_hook",
        attack_child,
    )

    with pytest.raises(ValueError, match="identity|inventory|drift"):
        _build_view(archive_authority, output_root)

    assert attacked
    assert _published_view_names(output_root) == ()


def test_postpublish_drift_quarantines_exact_root_and_vacates_final_name(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_root = tmp_path / "derived"
    final_name: str | None = None
    published_inode: int | None = None

    def attack_published(phase, authority, receipt_sha256):
        nonlocal final_name, published_inode
        if phase != "before_postpublish_verify":
            return
        final_name = receipt_sha256
        target = output_root / "wikidata" / receipt_sha256
        published_inode = target.stat().st_ino
        stream = target / "streams/training.tsv"
        payload = bytearray(stream.read_bytes())
        payload[len(payload) // 2] ^= 1
        stream.write_bytes(payload)
        assert published_inode == authority.identity[1]

    monkeypatch.setattr(
        wikidata_source_module,
        "_derived_view_build_hook",
        attack_published,
    )

    with pytest.raises(ValueError, match="identity|digest|drift"):
        _build_view(archive_authority, output_root)

    assert final_name is not None
    namespace = output_root / "wikidata"
    assert not (namespace / final_name).exists()
    quarantines = tuple(
        path
        for path in namespace.iterdir()
        if path.name.startswith(".quarantine-")
    )
    assert len(quarantines) == 1
    assert quarantines[0].stat().st_ino == published_inode
    assert quarantines[0].stat().st_mode & 0o777 == 0o700


def test_postpublish_root_swap_preserves_concurrent_winner(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_root = tmp_path / "derived"
    final_name: str | None = None
    winner_inode: int | None = None
    swapped = False

    def install_winner(phase, _authority, receipt_sha256):
        nonlocal final_name, winner_inode, swapped
        if phase != "before_postpublish_verify" or swapped:
            return
        swapped = True
        final_name = receipt_sha256
        namespace = output_root / "wikidata"
        target = namespace / receipt_sha256
        displaced = namespace / f"{receipt_sha256}.published-displaced"
        target.rename(displaced)
        shutil.copytree(displaced, target)
        winner_inode = target.stat().st_ino

    monkeypatch.setattr(
        wikidata_source_module,
        "_derived_view_build_hook",
        install_winner,
    )

    with pytest.raises(ValueError, match="identity|drift"):
        _build_view(archive_authority, output_root)

    assert swapped
    assert final_name is not None
    winner = output_root / "wikidata" / final_name
    assert winner.is_dir()
    assert winner.stat().st_ino == winner_inode
    assert not tuple(
        path
        for path in winner.parent.iterdir()
        if path.name.startswith(".quarantine-")
    )
    verified = wikidata_source_module.verify_wikidata_derived_view(
        archive_authority.source_lock_path,
        archive_authority.source_root,
        winner,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    assert verified.root == winner


def test_open_sorted_run_close_failure_preserves_body_error_and_closes_all(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    active = False
    raised_close = False
    attempted: list[int] = []
    real_close = os.close

    def fail_body(phase, _work_fd, _run):
        nonlocal active
        if phase == "before_record_yield":
            active = True
            raise RuntimeError("sort consumer body failure")

    def injected_close(descriptor):
        nonlocal raised_close
        if not active:
            real_close(descriptor)
            return
        attempted.append(descriptor)
        real_close(descriptor)
        if not raised_close:
            raised_close = True
            raise OSError("injected sorted-run close failure")

    monkeypatch.setattr(
        wikidata_source_module,
        "_external_sort_hook",
        fail_body,
    )
    monkeypatch.setattr(
        wikidata_source_module,
        "_close_descriptor",
        injected_close,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="sort consumer body failure"):
        _build_view(archive_authority, tmp_path / "derived")

    assert raised_close
    assert attempted
    for descriptor in attempted:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_merge_close_failures_attempt_every_input_and_output_descriptor(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    active = False
    failures_remaining = 2
    attempted: list[int] = []
    real_close = os.close

    def start_injection(phase, _work_fd, _run):
        nonlocal active
        if phase == "before_merge_close":
            active = True

    def injected_close(descriptor):
        nonlocal failures_remaining
        if not active:
            real_close(descriptor)
            return
        attempted.append(descriptor)
        real_close(descriptor)
        if failures_remaining:
            failures_remaining -= 1
            raise OSError("injected merge close failure")

    monkeypatch.setattr(
        wikidata_source_module,
        "_external_sort_hook",
        start_injection,
    )
    monkeypatch.setattr(
        wikidata_source_module,
        "_close_descriptor",
        injected_close,
        raising=False,
    )
    monkeypatch.setattr(wikidata_source_module, "_SORT_CHUNK_RECORDS", 1)

    with pytest.raises(ValueError, match="publication|close"):
        _build_view(archive_authority, tmp_path / "derived")

    assert failures_remaining == 0
    assert len(attempted) >= 3
    for descriptor in attempted:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_outer_close_failures_attempt_every_retained_descriptor(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    active = False
    failures_remaining = 2
    retained: list[int] = []
    attempted: list[int] = []
    real_close = os.close

    def fail_build(phase, authority, _final_name):
        nonlocal active
        if phase != "before_candidate_verify":
            return
        retained.append(authority.descriptor)
        retained.extend(authority.directory_descriptors.values())
        active = True
        raise RuntimeError("primary build failure")

    def injected_close(descriptor):
        nonlocal failures_remaining
        if not active:
            real_close(descriptor)
            return
        attempted.append(descriptor)
        real_close(descriptor)
        if failures_remaining:
            failures_remaining -= 1
            raise OSError("injected outer close failure")

    monkeypatch.setattr(
        wikidata_source_module,
        "_derived_view_build_hook",
        fail_build,
    )
    monkeypatch.setattr(
        wikidata_source_module,
        "_close_descriptor",
        injected_close,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="primary build failure") as raised:
        _build_view(archive_authority, tmp_path / "derived")

    assert "injected outer close failure" not in str(raised.value)
    assert failures_remaining == 0
    assert set(retained) <= set(attempted)
    for descriptor in retained:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def _mutate_at_before_finalize(directory_fd: int, name: str) -> None:
    descriptor = os.open(name, os.O_RDWR, dir_fd=directory_fd)
    try:
        payload = os.pread(descriptor, 1, 0)
        assert payload
        os.pwrite(descriptor, bytes((payload[0] ^ 1,)), 0)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    ("target_phase", "force_merges"),
    [
        ("before_flush_finalize", False),
        ("before_merge_finalize", True),
    ],
)
def test_run_mutation_before_finalization_rejects_observed_hash_adoption(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_phase: str,
    force_merges: bool,
):
    attacked = False

    def mutate_pending_run(phase, work_fd, pending_run):
        nonlocal attacked
        if phase != target_phase or attacked:
            return
        attacked = True
        _mutate_at_before_finalize(work_fd, pending_run.name)

    monkeypatch.setattr(
        wikidata_source_module,
        "_external_sort_hook",
        mutate_pending_run,
    )
    if force_merges:
        monkeypatch.setattr(
            wikidata_source_module,
            "_SORT_CHUNK_RECORDS",
            1,
        )

    with pytest.raises(ValueError, match="expected|digest|byte count"):
        _build_view(archive_authority, tmp_path / "derived")

    assert attacked


def test_run_writer_close_failure_preserves_finalize_error(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    active = False
    close_failed = False
    attempted: list[int] = []
    real_close = os.close

    def mutate_before_finalize(phase, work_fd, pending_run):
        nonlocal active
        if phase != "before_flush_finalize" or active:
            return
        _mutate_at_before_finalize(work_fd, pending_run.name)
        active = True

    def injected_close(descriptor):
        nonlocal close_failed
        if not active:
            real_close(descriptor)
            return
        attempted.append(descriptor)
        real_close(descriptor)
        if not close_failed:
            close_failed = True
            raise OSError("injected run-writer close failure")

    monkeypatch.setattr(
        wikidata_source_module,
        "_external_sort_hook",
        mutate_before_finalize,
    )
    monkeypatch.setattr(
        wikidata_source_module,
        "_close_descriptor",
        injected_close,
    )

    with pytest.raises(ValueError, match="expected|digest|byte count") as raised:
        _build_view(archive_authority, tmp_path / "derived")

    assert "run-writer close failure" not in str(raised.value)
    assert close_failed
    assert attempted
    for descriptor in attempted:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_archive_teardown_attempts_all_closes_and_preserves_body_error(
    archive_authority: _AuthorityFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    active = False
    failures_remaining = 2
    archive_descriptors: tuple[int, ...] = ()
    attempted: list[int] = []
    real_close = os.close

    def injected_close(descriptor):
        nonlocal failures_remaining
        if not active:
            real_close(descriptor)
            return
        attempted.append(descriptor)
        real_close(descriptor)
        if failures_remaining:
            failures_remaining -= 1
            raise OSError("injected archive close failure")

    monkeypatch.setattr(
        wikidata_source_module,
        "_close_descriptor",
        injected_close,
    )

    with pytest.raises(RuntimeError, match="archive body failure") as raised:
        with wikidata_source_module._open_verified_archives(
            archive_authority.source_lock_path,
            archive_authority.source_root,
            expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
        ) as verified:
            archive_descriptors = tuple(
                verified._archive_descriptors.values()
            )
            active = True
            raise RuntimeError("archive body failure")

    assert "archive close failure" not in str(raised.value)
    assert failures_remaining == 0
    assert len(attempted) == len(archive_descriptors) + 5
    assert set(archive_descriptors) <= set(attempted)
    for descriptor in attempted:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_postrename_fsync_failure_exchange_quarantines_and_vacates_final(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_root = tmp_path / "derived"
    armed = False
    failed = False
    final_name: str | None = None
    published_inode: int | None = None
    real_fsync_directory = wikidata_source_module.fsync_directory

    def arm_after_rename(phase, authority, receipt_sha256):
        nonlocal armed, final_name, published_inode
        if phase != "after_publish_rename":
            return
        final_name = receipt_sha256
        target = output_root / "wikidata" / receipt_sha256
        published_inode = target.stat().st_ino
        assert published_inode == authority.identity[1]
        armed = True

    def fail_transaction_fsync(directory_fd):
        nonlocal armed, failed
        if armed and not failed:
            armed = False
            failed = True
            raise OSError("injected postrename fsync failure")
        real_fsync_directory(directory_fd)

    monkeypatch.setattr(
        wikidata_source_module,
        "_derived_view_build_hook",
        arm_after_rename,
    )
    monkeypatch.setattr(
        wikidata_source_module,
        "fsync_directory",
        fail_transaction_fsync,
    )

    with pytest.raises(ValueError, match="publication|fsync"):
        _build_view(archive_authority, output_root)

    assert failed
    assert final_name is not None
    namespace = output_root / "wikidata"
    assert not (namespace / final_name).exists()
    quarantines = tuple(
        path
        for path in namespace.iterdir()
        if path.name.startswith(".quarantine-")
    )
    assert len(quarantines) == 1
    assert quarantines[0].stat().st_ino == published_inode
    assert quarantines[0].stat().st_mode & 0o777 == 0o700


def test_quarantine_exchange_race_restores_substituted_winner(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_root = tmp_path / "derived"
    final_name: str | None = None
    winner_inode: int | None = None
    marker_inode: int | None = None
    swapped = False

    def fail_then_install_winner(phase, _authority, receipt_sha256):
        nonlocal final_name, winner_inode, marker_inode, swapped
        if phase == "before_postpublish_verify":
            final_name = receipt_sha256
            raise RuntimeError("forced postpublication failure")
        if phase != "before_quarantine_exchange" or swapped:
            return
        swapped = True
        namespace = output_root / "wikidata"
        markers = tuple(
            path
            for path in namespace.iterdir()
            if path.name.startswith(".quarantine-")
        )
        assert len(markers) == 1
        assert not tuple(markers[0].iterdir())
        assert markers[0].stat().st_mode & 0o777 == 0o700
        marker_inode = markers[0].stat().st_ino
        target = namespace / receipt_sha256
        displaced = namespace / f"{receipt_sha256}.race-displaced"
        target.rename(displaced)
        shutil.copytree(displaced, target)
        winner_inode = target.stat().st_ino

    monkeypatch.setattr(
        wikidata_source_module,
        "_derived_view_build_hook",
        fail_then_install_winner,
    )

    with pytest.raises(RuntimeError, match="forced postpublication failure"):
        _build_view(archive_authority, output_root)

    assert swapped
    assert final_name is not None
    namespace = output_root / "wikidata"
    winner = namespace / final_name
    assert winner.is_dir()
    assert winner.stat().st_ino == winner_inode
    assert all(
        path.stat().st_ino != marker_inode
        for path in namespace.iterdir()
    )
    assert not tuple(
        path
        for path in namespace.iterdir()
        if path.name.startswith(".quarantine-")
    )
    verified = wikidata_source_module.verify_wikidata_derived_view(
        archive_authority.source_lock_path,
        archive_authority.source_root,
        winner,
        expected_generator_commit=EXPECTED_GENERATOR_COMMIT,
    )
    assert verified.root == winner


def test_marker_is_bound_before_final_verification_and_publish_is_immediate(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_root = tmp_path / "derived"
    verified_snapshot: dict[str, tuple[int, int, int]] | None = None
    checked_publish = False
    original_verify = wikidata_source_module._verify_sealed_private_build
    original_rename = wikidata_source_module.atomic_rename_noreplace

    def namespace_snapshot() -> dict[str, tuple[int, int, int]]:
        namespace = output_root / "wikidata"
        return {
            path.name: (
                path.lstat().st_dev,
                path.lstat().st_ino,
                path.lstat().st_mode,
            )
            for path in namespace.iterdir()
        }

    def verify_with_marker(
        authority,
        *,
        root_name,
        expected_root_identity,
    ):
        nonlocal verified_snapshot
        result = original_verify(
            authority,
            root_name=root_name,
            expected_root_identity=expected_root_identity,
        )
        if root_name.startswith(".build-"):
            markers = tuple(
                path
                for path in (output_root / "wikidata").iterdir()
                if path.name.startswith(".quarantine-")
            )
            assert len(markers) == 1
            assert not tuple(markers[0].iterdir())
            assert markers[0].stat().st_mode & 0o777 == 0o700
            verified_snapshot = namespace_snapshot()
        return result

    def publish_without_mutation(
        source_directory_fd,
        source_name,
        destination_directory_fd,
        destination_name,
    ):
        nonlocal checked_publish
        if source_name.startswith(".build-"):
            assert verified_snapshot is not None
            assert namespace_snapshot() == verified_snapshot
            checked_publish = True
        return original_rename(
            source_directory_fd,
            source_name,
            destination_directory_fd,
            destination_name,
        )

    monkeypatch.setattr(
        wikidata_source_module,
        "_verify_sealed_private_build",
        verify_with_marker,
    )
    monkeypatch.setattr(
        wikidata_source_module,
        "atomic_rename_noreplace",
        publish_without_mutation,
    )

    _build_view(archive_authority, output_root)

    assert checked_publish


def test_marker_substitution_keeps_failed_candidate_quarantined(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_root = tmp_path / "derived"
    candidate_inode: int | None = None
    substitute_inode: int | None = None
    marker_inode: int | None = None
    quarantine_name: str | None = None
    substituted = False

    def substitute_marker(phase, authority, receipt_sha256):
        nonlocal candidate_inode, substitute_inode, marker_inode
        nonlocal quarantine_name, substituted
        namespace = output_root / "wikidata"
        if phase == "before_postpublish_verify":
            candidate_inode = (namespace / receipt_sha256).stat().st_ino
            assert candidate_inode == authority.identity[1]
            raise RuntimeError("forced failed candidate")
        if phase != "before_quarantine_exchange" or substituted:
            return
        substituted = True
        markers = tuple(
            path
            for path in namespace.iterdir()
            if path.name.startswith(".quarantine-")
        )
        assert len(markers) == 1
        marker = markers[0]
        quarantine_name = marker.name
        marker_inode = marker.stat().st_ino
        displaced_marker = namespace / "retained-marker-displaced"
        marker.rename(displaced_marker)
        marker.mkdir(mode=0o700)
        substitute_inode = marker.stat().st_ino

    monkeypatch.setattr(
        wikidata_source_module,
        "_derived_view_build_hook",
        substitute_marker,
    )

    with pytest.raises(RuntimeError, match="forced failed candidate"):
        _build_view(archive_authority, output_root)

    assert substituted
    assert quarantine_name is not None
    namespace = output_root / "wikidata"
    final_names = tuple(
        path
        for path in namespace.iterdir()
        if len(path.name) == 64
    )
    assert len(final_names) == 1
    assert final_names[0].stat().st_ino == substitute_inode
    quarantined = namespace / quarantine_name
    assert quarantined.is_dir()
    assert quarantined.stat().st_ino == candidate_inode
    assert final_names[0].stat().st_ino != candidate_inode
    displaced_marker = namespace / "retained-marker-displaced"
    assert displaced_marker.stat().st_ino == marker_inode


def test_sort_run_open_validation_close_failure_preserves_identity_error(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_root = tmp_path / "derived"
    opened: list[int] = []
    opened_identities: dict[int, tuple[int, int]] = {}
    attacked = False
    close_failed = False
    direct_close_seen = False
    original_open = wikidata_source_module.open_regular_file_at
    real_close = os.close

    def record_open(directory_fd, name):
        descriptor, metadata = original_open(directory_fd, name)
        if attacked and "-l" in name and name.endswith(".bin"):
            opened.append(descriptor)
            opened_identities[descriptor] = (
                metadata.st_dev,
                metadata.st_ino,
            )
        return descriptor, metadata

    def substitute_run(phase, work_fd, run):
        nonlocal attacked
        if phase != "before_open" or attacked:
            return
        attacked = True
        os.rename(
            run.name,
            f"{run.name}.displaced",
            src_dir_fd=work_fd,
            dst_dir_fd=work_fd,
        )
        replacement = os.open(
            run.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=work_fd,
        )
        os.write(replacement, b"replacement")
        os.close(replacement)

    def injected_close(descriptor):
        nonlocal close_failed
        closes_target = False
        if descriptor in opened_identities:
            try:
                metadata = os.fstat(descriptor)
            except OSError:
                pass
            else:
                closes_target = (
                    metadata.st_dev,
                    metadata.st_ino,
                ) == opened_identities[descriptor]
        real_close(descriptor)
        if closes_target and not close_failed:
            close_failed = True
            raise OSError("injected sort-open close failure")

    def observe_direct_close(descriptor):
        nonlocal direct_close_seen
        if descriptor in opened_identities:
            try:
                metadata = os.fstat(descriptor)
            except OSError:
                pass
            else:
                if (
                    metadata.st_dev,
                    metadata.st_ino,
                ) == opened_identities[descriptor]:
                    direct_close_seen = True
        real_close(descriptor)

    monkeypatch.setattr(
        wikidata_source_module,
        "open_regular_file_at",
        record_open,
    )
    monkeypatch.setattr(
        wikidata_source_module,
        "_external_sort_hook",
        substitute_run,
    )
    monkeypatch.setattr(
        wikidata_source_module,
        "_close_descriptor",
        injected_close,
    )
    monkeypatch.setattr(
        wikidata_source_module.os,
        "close",
        observe_direct_close,
    )
    monkeypatch.setattr(wikidata_source_module, "_SORT_CHUNK_RECORDS", 1)

    with pytest.raises(ValueError, match="identity drift") as raised:
        _build_view(archive_authority, output_root)

    assert "sort-open close failure" not in str(raised.value)
    assert attacked
    assert not direct_close_seen
    assert close_failed
    assert opened
    for descriptor in opened:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_archive_envelope_duplicate_close_preserves_body_error(
    archive_authority: _AuthorityFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    source = (
        archive_authority.source_root
        / "wikidata5m"
        / wikidata_source_module.ARCHIVE_PATHS[0]
    )
    source_fd = os.open(source, os.O_RDONLY)
    duplicate_fd = -1
    close_failed = False
    real_dup = os.dup
    real_read = os.read
    real_close = os.close

    def record_duplicate(descriptor):
        nonlocal duplicate_fd
        duplicate = real_dup(descriptor)
        if descriptor == source_fd:
            duplicate_fd = duplicate
        return duplicate

    def fail_duplicate_read(descriptor, size):
        if descriptor == duplicate_fd:
            raise RuntimeError("envelope body failure")
        return real_read(descriptor, size)

    def injected_close(descriptor):
        nonlocal close_failed
        real_close(descriptor)
        if descriptor == duplicate_fd and not close_failed:
            close_failed = True
            raise OSError("injected envelope close failure")

    monkeypatch.setattr(wikidata_source_module.os, "dup", record_duplicate)
    monkeypatch.setattr(wikidata_source_module.os, "read", fail_duplicate_read)
    monkeypatch.setattr(
        wikidata_source_module,
        "_close_descriptor",
        injected_close,
    )

    try:
        with pytest.raises(RuntimeError, match="envelope body failure") as raised:
            wikidata_source_module._validate_archive_envelope(
                source_fd,
                wikidata_source_module.ARCHIVE_PATHS[0],
            )
    finally:
        real_close(source_fd)

    assert "envelope close failure" not in str(raised.value)
    assert close_failed
    with pytest.raises(OSError):
        os.fstat(duplicate_fd)


def test_archive_parser_fdopen_close_preserves_body_error(
    archive_authority: _AuthorityFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    archive_path = wikidata_source_module.ARCHIVE_PATHS[0]
    source = archive_authority.source_root / "wikidata5m" / archive_path
    source_fd = os.open(source, os.O_RDONLY)
    armed = False
    duplicate_fd = -1
    close_failed = False
    real_validate = wikidata_source_module._validate_archive_envelope
    real_dup = os.dup
    real_close = os.close

    def validate_then_arm(descriptor, path):
        nonlocal armed
        real_validate(descriptor, path)
        armed = True

    def record_duplicate(descriptor):
        nonlocal duplicate_fd
        duplicate = real_dup(descriptor)
        if armed and descriptor == source_fd:
            duplicate_fd = duplicate
        return duplicate

    def fail_tar_open(*_args, **_kwargs):
        if armed:
            raise RuntimeError("parser body failure")
        raise AssertionError("parser tar hook armed too late")

    def injected_close(descriptor):
        nonlocal close_failed
        real_close(descriptor)
        if descriptor == duplicate_fd and not close_failed:
            close_failed = True
            raise OSError("injected parser close failure")

    monkeypatch.setattr(
        wikidata_source_module,
        "_validate_archive_envelope",
        validate_then_arm,
    )
    monkeypatch.setattr(wikidata_source_module.os, "dup", record_duplicate)
    monkeypatch.setattr(
        wikidata_source_module.tarfile,
        "open",
        fail_tar_open,
    )
    monkeypatch.setattr(
        wikidata_source_module,
        "_close_descriptor",
        injected_close,
    )

    try:
        with pytest.raises(RuntimeError, match="parser body failure") as raised:
            wikidata_source_module._parse_archive_descriptor(
                source_fd,
                archive_path,
                {},
            )
    finally:
        real_close(source_fd)

    assert "parser close failure" not in str(raised.value)
    assert close_failed
    with pytest.raises(OSError):
        os.fstat(duplicate_fd)


def test_archive_materialization_fdopen_close_preserves_body_error(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    tar_calls = 0
    duplicates: list[int] = []
    target_fd = -1
    close_failed = False
    original_tar_open = wikidata_source_module.tarfile.open
    real_dup = os.dup
    real_close = os.close

    def record_duplicate(descriptor):
        duplicate = real_dup(descriptor)
        duplicates.append(duplicate)
        return duplicate

    def fail_materialization_open(*args, **kwargs):
        nonlocal tar_calls, target_fd
        tar_calls += 1
        if tar_calls > len(wikidata_source_module.ARCHIVE_PATHS):
            target_fd = duplicates[-1]
            raise RuntimeError("materialization body failure")
        return original_tar_open(*args, **kwargs)

    def injected_close(descriptor):
        nonlocal close_failed
        real_close(descriptor)
        if descriptor == target_fd and not close_failed:
            close_failed = True
            raise OSError("injected materialization close failure")

    monkeypatch.setattr(wikidata_source_module.os, "dup", record_duplicate)
    monkeypatch.setattr(
        wikidata_source_module.tarfile,
        "open",
        fail_materialization_open,
    )
    monkeypatch.setattr(
        wikidata_source_module,
        "_close_descriptor",
        injected_close,
    )

    with pytest.raises(RuntimeError, match="materialization body failure") as raised:
        _build_view(archive_authority, tmp_path / "derived")

    assert "materialization close failure" not in str(raised.value)
    assert close_failed
    assert target_fd >= 0
    with pytest.raises(OSError):
        os.fstat(target_fd)


def test_two_name_preswap_retries_until_candidate_is_quarantined(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_root = tmp_path / "derived"
    candidate_inode: int | None = None
    marker_inode: int | None = None
    final_name: str | None = None
    adversary_swapped = False

    def preswap_exact_names(phase, authority, receipt_sha256):
        nonlocal candidate_inode, marker_inode, final_name
        nonlocal adversary_swapped
        namespace = output_root / "wikidata"
        if phase == "before_postpublish_verify":
            final_name = receipt_sha256
            candidate_inode = (namespace / receipt_sha256).stat().st_ino
            raise RuntimeError("forced failed candidate")
        if phase != "before_quarantine_exchange" or adversary_swapped:
            return
        adversary_swapped = True
        markers = tuple(
            path
            for path in namespace.iterdir()
            if path.name.startswith(".quarantine-")
        )
        assert len(markers) == 1
        marker_inode = markers[0].stat().st_ino
        wikidata_source_module._atomic_exchange_directories(
            authority.namespace_fd,
            receipt_sha256,
            markers[0].name,
        )
        assert (namespace / receipt_sha256).stat().st_ino == marker_inode
        assert markers[0].stat().st_ino == candidate_inode

    monkeypatch.setattr(
        wikidata_source_module,
        "_derived_view_build_hook",
        preswap_exact_names,
    )

    with pytest.raises(RuntimeError, match="forced failed candidate"):
        _build_view(archive_authority, output_root)

    assert adversary_swapped
    assert final_name is not None
    namespace = output_root / "wikidata"
    final = namespace / final_name
    assert not final.exists() or final.stat().st_ino != candidate_inode
    quarantined_candidates = tuple(
        path
        for path in namespace.iterdir()
        if path.name.startswith(".quarantine-")
        and path.stat().st_ino == candidate_inode
    )
    assert len(quarantined_candidates) == 1
    assert all(
        path.stat().st_ino != marker_inode
        for path in namespace.iterdir()
    )


def test_repeated_original_state_exhaustion_refreshes_marker(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_root = tmp_path / "derived"
    candidate_inode: int | None = None
    final_name: str | None = None
    allocated_markers: list[str] = []
    exchange_calls = 0
    forced_repeats = 2
    original_allocate = wikidata_source_module._allocate_quarantine_marker
    original_exchange = wikidata_source_module._atomic_exchange_directories

    def record_marker(namespace_fd, published_name):
        marker = original_allocate(namespace_fd, published_name)
        allocated_markers.append(marker.name)
        return marker

    def repeat_original_state(directory_fd, first_name, second_name):
        nonlocal exchange_calls
        exchange_calls += 1
        original_exchange(directory_fd, first_name, second_name)
        if exchange_calls <= forced_repeats:
            original_exchange(directory_fd, first_name, second_name)

    def fail_after_publish(phase, _authority, receipt_sha256):
        nonlocal candidate_inode, final_name
        if phase != "before_postpublish_verify":
            return
        final_name = receipt_sha256
        candidate_inode = (
            output_root / "wikidata" / receipt_sha256
        ).stat().st_ino
        raise RuntimeError("forced repeated-state failure")

    monkeypatch.setattr(
        wikidata_source_module,
        "_allocate_quarantine_marker",
        record_marker,
    )
    monkeypatch.setattr(
        wikidata_source_module,
        "_atomic_exchange_directories",
        repeat_original_state,
    )
    monkeypatch.setattr(
        wikidata_source_module,
        "_derived_view_build_hook",
        fail_after_publish,
    )

    with pytest.raises(RuntimeError, match="forced repeated-state failure"):
        _build_view(archive_authority, output_root)

    assert final_name is not None
    assert len(allocated_markers) >= 2
    assert exchange_calls >= forced_repeats + 1
    namespace = output_root / "wikidata"
    final = namespace / final_name
    assert not final.exists() or final.stat().st_ino != candidate_inode
    quarantines = tuple(
        path
        for path in namespace.iterdir()
        if path.name.startswith(".quarantine-")
    )
    assert len(quarantines) == 1
    assert quarantines[0].stat().st_ino == candidate_inode


def test_repeated_fresh_marker_bind_failures_keep_resources_bounded(
    archive_authority: _AuthorityFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_root = tmp_path / "derived"
    candidate_inode: int | None = None
    final_name: str | None = None
    bind_failures = 4
    fresh_checks = 0
    fresh_bind_succeeded = False
    close_failure_injected = False
    allocation_records: list[
        tuple[int, wikidata_source_module._CreationIdentity]
    ] = []
    failed_fresh: dict[
        int,
        wikidata_source_module._CreationIdentity,
    ] = {}
    observations: list[tuple[int, int]] = []
    original_allocate = wikidata_source_module._allocate_quarantine_marker
    original_check = wikidata_source_module._check_named_derived_directory
    original_exchange = wikidata_source_module._atomic_exchange_directories
    real_close = os.close

    def marker_is_open(
        descriptor: int,
        identity: wikidata_source_module._CreationIdentity,
    ) -> bool:
        try:
            metadata = os.fstat(descriptor)
        except OSError:
            return False
        return (
            wikidata_source_module._creation_identity(metadata)
            == identity
        )

    def record_allocation(namespace_fd, published_name):
        marker = original_allocate(namespace_fd, published_name)
        allocation_records.append((marker.descriptor, marker.identity))
        return marker

    def fail_fresh_bind(
        parent_fd,
        name,
        descriptor,
        identity,
        description,
    ):
        nonlocal fresh_checks, fresh_bind_succeeded
        if description == "fresh quarantine marker":
            fresh_checks += 1
            if fresh_checks <= bind_failures:
                failed_fresh[descriptor] = identity
                raise ValueError(
                    f"injected fresh marker bind failure {fresh_checks}"
                )
            fresh_bind_succeeded = True
        return original_check(
            parent_fd,
            name,
            descriptor,
            identity,
            description,
        )

    def injected_close(descriptor):
        nonlocal close_failure_injected
        inject = (
            not close_failure_injected
            and descriptor in failed_fresh
            and marker_is_open(descriptor, failed_fresh[descriptor])
        )
        real_close(descriptor)
        if inject:
            close_failure_injected = True
            raise OSError("injected fresh marker close failure")

    def repeat_until_fresh_marker(directory_fd, first_name, second_name):
        if (
            0 < fresh_checks <= bind_failures
            and len(observations) < fresh_checks
        ):
            namespace = output_root / "wikidata"
            open_markers = sum(
                marker_is_open(descriptor, identity)
                for descriptor, identity in allocation_records
            )
            named_markers = sum(
                path.name.startswith(".quarantine-")
                for path in namespace.iterdir()
            )
            observations.append((open_markers, named_markers))
        original_exchange(directory_fd, first_name, second_name)
        if not fresh_bind_succeeded:
            original_exchange(directory_fd, first_name, second_name)

    def fail_after_publish(phase, _authority, receipt_sha256):
        nonlocal candidate_inode, final_name
        if phase != "before_postpublish_verify":
            return
        final_name = receipt_sha256
        candidate_inode = (
            output_root / "wikidata" / receipt_sha256
        ).stat().st_ino
        raise RuntimeError("forced fresh-marker stress failure")

    monkeypatch.setattr(
        wikidata_source_module,
        "_allocate_quarantine_marker",
        record_allocation,
    )
    monkeypatch.setattr(
        wikidata_source_module,
        "_check_named_derived_directory",
        fail_fresh_bind,
    )
    monkeypatch.setattr(
        wikidata_source_module,
        "_close_descriptor",
        injected_close,
    )
    monkeypatch.setattr(
        wikidata_source_module,
        "_atomic_exchange_directories",
        repeat_until_fresh_marker,
    )
    monkeypatch.setattr(
        wikidata_source_module,
        "_derived_view_build_hook",
        fail_after_publish,
    )

    with pytest.raises(
        RuntimeError,
        match="forced fresh-marker stress failure",
    ) as raised:
        _build_view(archive_authority, output_root)

    assert fresh_bind_succeeded
    assert fresh_checks == bind_failures + 1
    assert observations == [(1, 1)] * bind_failures
    assert close_failure_injected
    notes = getattr(raised.value, "__notes__", ())
    assert any(
        "injected fresh marker bind failure 1" in note
        for note in notes
    )
    assert all(
        "fresh marker close failure" not in note
        for note in notes
    )
    assert not any(
        marker_is_open(descriptor, identity)
        for descriptor, identity in allocation_records
    )
    assert final_name is not None
    namespace = output_root / "wikidata"
    final = namespace / final_name
    assert not final.exists() or final.stat().st_ino != candidate_inode
    quarantines = tuple(
        path
        for path in namespace.iterdir()
        if path.name.startswith(".quarantine-")
    )
    assert len(quarantines) == 1
    assert quarantines[0].stat().st_ino == candidate_inode
