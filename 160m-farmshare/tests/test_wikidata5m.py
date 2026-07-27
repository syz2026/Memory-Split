import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

import corpusgen.wikidata5m as wikidata5m_module
from corpusgen.wikidata5m import (
    SourceDriftError,
    Triple,
    UnsafeArchiveError,
    WikidataLock,
    canonicalize_aliases,
    iter_triples,
    normalize_alias,
    parse_pid,
    parse_qid,
    read_aliases,
    safe_extract_archives,
    verify_archives,
)
from scripts.fetch_wikidata5m import build_download_command


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "wikidata5m"
SOURCE_LOCK = ROOT / "sources" / "wikidata5m.lock.json"
EXPECTED_SOURCE_LOCK = {
    "repo_id": "intfloat/wikidata5m",
    "repo_type": "dataset",
    "revision": "6b2b09672129e280c0c9da97ab58154e9d535e6b",
    "files": {
        "wikidata5m_alias.tar.gz": {
            "bytes": 197449751,
            "sha256": (
                "0330f580c9f7a57cbad949ac380835fdd2a2e14d96cc0f13fc435401d6b463a8"
            ),
        },
        "wikidata5m_inductive.tar.gz": {
            "bytes": 167247416,
            "sha256": (
                "955081232cc2de859710bfe3a147f7d8314524010fe5f8c420bb74fdfee4f42a"
            ),
        },
        "wikidata5m_transductive.tar.gz": {
            "bytes": 168258214,
            "sha256": (
                "383160990b41c0905fc03f4a8afbb9b12be1ca3591e026bde6cdc94a59542597"
            ),
        },
    },
}


def _write_tar(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for info, data in members:
            archive.addfile(info, io.BytesIO(data) if data is not None else None)


def _regular_member(name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    return info, data


def _lock_for_files(root):
    files = {}
    for path in sorted(root.iterdir()):
        if path.is_file():
            content = path.read_bytes()
            files[path.name] = {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
    return WikidataLock.from_dict(
        {
            "repo_id": "example/wikidata",
            "repo_type": "dataset",
            "revision": "a" * 40,
            "files": files,
        }
    )


def test_committed_source_lock_has_exact_claim_bearing_values():
    assert json.loads(SOURCE_LOCK.read_text()) == EXPECTED_SOURCE_LOCK
    assert WikidataLock.from_path(SOURCE_LOCK).to_dict() == EXPECTED_SOURCE_LOCK


def test_fetch_script_builds_the_exact_pinned_hf_command():
    assert build_download_command(Path("/data"), SOURCE_LOCK) == [
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
        "wikidata5m_inductive.tar.gz",
        "--include",
        "wikidata5m_transductive.tar.gz",
        "--local-dir",
        "/data/wikidata5m",
    ]


def test_fixture_manifest_is_canonical_and_locks_every_fixture_byte():
    manifest_path = FIXTURES / "fixture-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest_bytes == (
        json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )

    expected_names = {
        "wikidata5m_relation.txt",
        "wikidata5m_entity.txt",
        "wikidata5m_transductive_train.txt",
        "wikidata5m_inductive_train.txt",
        "wikidata5m_inductive_valid.txt",
        "wikidata5m_inductive_test.txt",
    }
    assert set(manifest["files"]) == expected_names
    for name, expected in manifest["files"].items():
        content = (FIXTURES / name).read_bytes()
        assert expected == {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }


def test_archive_verification_detects_one_byte_drift(tmp_path):
    archive = tmp_path / "tiny.tar.gz"
    archive.write_bytes(b"locked archive")
    lock = _lock_for_files(tmp_path)
    verify_archives(tmp_path, lock)

    archive.write_bytes(archive.read_bytes() + b"!")

    with pytest.raises(SourceDriftError, match="tiny.tar.gz"):
        verify_archives(tmp_path, lock)


def test_safe_extract_rejects_extra_unpinned_archive_before_any_write(tmp_path):
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    _write_tar(
        archive_root / "pinned.tar.gz",
        [_regular_member("pinned.txt", b"pinned")],
    )
    lock = _lock_for_files(archive_root)
    _write_tar(
        archive_root / "extra.tar.gz",
        [_regular_member("extra.txt", b"extra")],
    )
    out = tmp_path / "out"

    with pytest.raises(SourceDriftError, match="unexpected unpinned archive"):
        safe_extract_archives(archive_root, out, lock=lock)

    assert not out.exists()


def test_safe_extract_two_argument_api_loads_the_default_lock(
    monkeypatch, tmp_path
):
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    _write_tar(
        archive_root / "pinned.tar.gz",
        [_regular_member("pinned.txt", b"pinned")],
    )
    lock = _lock_for_files(archive_root)
    monkeypatch.setattr(
        wikidata5m_module,
        "_load_default_lock",
        lambda: lock,
        raising=False,
    )

    extracted = safe_extract_archives(archive_root, tmp_path / "out")

    assert [path.name for path in extracted] == ["pinned.txt"]


def test_safe_extract_fixture_lock_override_is_keyword_only(tmp_path):
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    _write_tar(
        archive_root / "pinned.tar.gz",
        [_regular_member("pinned.txt", b"pinned")],
    )
    lock = _lock_for_files(archive_root)

    with pytest.raises(TypeError, match="takes 2 positional arguments but 3"):
        safe_extract_archives(archive_root, tmp_path / "out", lock)


def test_safe_extract_validates_every_archive_before_writing_any_member(tmp_path):
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    _write_tar(
        archive_root / "a-good.tar.gz",
        [_regular_member("nested/good.txt", b"good")],
    )
    _write_tar(
        archive_root / "z-bad.tar.gz",
        [_regular_member("../escape.txt", b"bad")],
    )
    lock = _lock_for_files(archive_root)
    out = tmp_path / "out"

    with pytest.raises(UnsafeArchiveError, match="unsafe archive member"):
        safe_extract_archives(archive_root, out, lock=lock)

    assert not (out / "nested" / "good.txt").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_rejects_absolute_and_link_members(tmp_path):
    for index, info in enumerate(
        (
            tarfile.TarInfo("/absolute.txt"),
            tarfile.TarInfo("symbolic-link"),
            tarfile.TarInfo("hard-link"),
            tarfile.TarInfo("device"),
        )
    ):
        archive_root = tmp_path / f"archives-{index}"
        archive_root.mkdir()
        if index == 0:
            info.size = 1
            data = b"x"
        else:
            info.type = (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE)[
                index - 1
            ]
            info.linkname = "target"
            data = None
        _write_tar(archive_root / "unsafe.tar.gz", [(info, data)])
        lock = _lock_for_files(archive_root)

        with pytest.raises(UnsafeArchiveError):
            safe_extract_archives(
                archive_root,
                tmp_path / f"out-{index}",
                lock=lock,
            )


def test_safe_extract_writes_regular_files_after_validation(tmp_path):
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    _write_tar(
        archive_root / "tiny.tar.gz",
        [
            _regular_member("one.txt", b"one"),
            _regular_member("nested/two.txt", b"two"),
        ],
    )
    lock = _lock_for_files(archive_root)

    extracted = safe_extract_archives(
        archive_root,
        tmp_path / "out",
        lock=lock,
    )

    assert [path.relative_to(tmp_path / "out").as_posix() for path in extracted] == [
        "nested/two.txt",
        "one.txt",
    ]
    assert (tmp_path / "out" / "one.txt").read_bytes() == b"one"
    assert (tmp_path / "out" / "nested" / "two.txt").read_bytes() == b"two"


@pytest.mark.parametrize("value", ["", "1", "q1", "Q-1", "Q 1", "Q"])
def test_parse_qid_rejects_malformed_ids(value):
    with pytest.raises(ValueError, match="invalid Q id"):
        parse_qid(value)


@pytest.mark.parametrize("value", ["", "1", "p1", "P-1", "P 1", "P"])
def test_parse_pid_rejects_malformed_ids(value):
    with pytest.raises(ValueError, match="invalid P id"):
        parse_pid(value)


def test_triple_parser_reads_qid_pid_qid_and_rejects_bad_rows(tmp_path):
    assert list(iter_triples(FIXTURES / "wikidata5m_transductive_train.txt"))[:2] == [
        Triple(1, "P1", 2),
        Triple(2, "P2", 3),
    ]

    malformed = tmp_path / "triples.txt"
    malformed.write_text("Q1\tR2\tQ3\n")
    with pytest.raises(ValueError, match=r"triples.txt:1: invalid P id"):
        list(iter_triples(malformed))
    malformed.write_text("Q1\tP2\n")
    with pytest.raises(ValueError, match=r"triples.txt:1: expected 3 tab-separated"):
        list(iter_triples(malformed))


def test_alias_reader_deduplicates_normalized_aliases_in_first_seen_raw_order(
    tmp_path,
):
    path = tmp_path / "relations.txt"
    path.write_text(
        "P1\t Head Office \thead   office\tHQ\thq\n"
        "P2\tlocated in\tSituated In\n"
    )

    aliases = read_aliases(path, "P")

    assert aliases == {
        "P1": (" Head Office ", "HQ"),
        "P2": ("located in", "Situated In"),
    }


def test_alias_reader_rejects_duplicate_ids_and_wrong_prefix(tmp_path):
    path = tmp_path / "aliases.txt"
    path.write_text("P1\tone\nP1\ttwo\n")
    with pytest.raises(ValueError, match="duplicate canonical ID"):
        read_aliases(path, "P")

    path.write_text("Q1\tone\n")
    with pytest.raises(ValueError, match="invalid P id"):
        read_aliases(path, "P")


def test_alias_normalization_excludes_cross_relation_collisions():
    aliases = {
        "P1": [" Head Office ", "HQ"],
        "P2": ["head   office", "located in"],
    }

    catalog = canonicalize_aliases(aliases)

    assert normalize_alias(" ℌead   OFFICE ") == "head office"
    assert catalog["P1"] == ("HQ",)
    assert catalog["P2"] == ("located in",)
    assert catalog.ambiguous_normalized == {"head office": ("P1", "P2")}
