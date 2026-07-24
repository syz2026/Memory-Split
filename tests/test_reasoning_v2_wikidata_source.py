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
