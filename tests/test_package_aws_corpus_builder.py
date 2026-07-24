from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from cluster.aws.corpus_builder.package import (
    ARCHIVE_NAME,
    MANIFEST_NAME,
    REQUIRED_FILES,
    REQUIRED_PREFIXES,
    SHA256_NAME,
    BuilderPackage,
    PackageError,
    build_corpus_builder_package,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = ROOT / "cluster" / "aws" / "corpus_builder" / "package.py"
SCRIPT_SOURCE = ROOT / "scripts" / "package_aws_corpus_builder.py"
PROFILE = "cluster/profiles/aws-i4i.16xlarge-corpus-v1.json"
TOKENIZER_ASSETS = (
    "vendor/tiktoken/6c7ea1a7e38e3a7f062df639a5b80947f075ffe6",
    "vendor/tiktoken/6d1cbeee0f20b3d9449abfede4726ed8212e3aee",
)
REQUIRED_AUTHORITIES = (
    "cluster/aws/corpus_builder/contracts.py",
    PROFILE,
    "configs/current-dataset-lock.json",
    "configs/reasoning-dataset-v2.json",
    "corpusgen/parallel/adapters.py",
    "corpusgen/parallel/catalog.py",
    "corpusgen/parallel/publication.py",
    "corpusgen/reasoning_v2/catalog.py",
    "corpusgen/reasoning_v2/source_lock.py",
    "sources/Wikidata-CC0-1.0.txt",
    "sources/current-dataset-licenses.json",
    "sources/wikidata5m.lock.json",
    *TOKENIZER_ASSETS,
    "tests/test_aws_corpus_builder_contracts.py",
    "tests/test_parallel_corpus.py",
    "tests/test_reasoning_v2_catalog.py",
    "tests/test_reasoning_v2_source_lock.py",
)


def _run(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def _git(root: Path, *arguments: str) -> str:
    completed = _run(root, "git", *arguments)
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _write(
    root: Path,
    relative: str,
    data: bytes | str,
    *,
    executable: bool = False,
) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    path.chmod(0o755 if executable else 0o644)


def _commit(root: Path, message: str) -> None:
    _git(root, "add", "--all")
    completed = _run(
        root,
        "git",
        "-c",
        "user.name=Package Test",
        "-c",
        "user.email=package-test@example.invalid",
        "commit",
        "-q",
        "-m",
        message,
    )
    assert completed.returncode == 0, completed.stderr


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _fixture_files() -> dict[str, bytes | str]:
    return {
        "README.md": "# intentionally outside the package allowlist\n",
        "pyproject.toml": "[build-system]\nrequires = []\n",
        "requirements.txt": "pytest\n",
        "cluster/aws/corpus_builder/__init__.py": '"""fixture package"""\n',
        "cluster/aws/corpus_builder/contracts.py": "FORMAT = 'fixture-v1'\n",
        "cluster/aws/corpus_builder/package.py": PACKAGE_SOURCE.read_bytes(),
        PROFILE: '{"profile_id":"aws-i4i.16xlarge-corpus-v1"}\n',
        "cluster/profiles/unrelated-provider.json": '{"provider":"other"}\n',
        "configs/current-dataset-lock.json": '{"schema_version":1}\n',
        "configs/reasoning-dataset-v2.json": '{"schema_version":2}\n',
        "corpusgen/parallel/adapters.py": "PRODUCTION_ADAPTERS = ()\n",
        "corpusgen/parallel/catalog.py": "CATALOG_FORMAT = 1\n",
        "corpusgen/parallel/publication.py": "RECEIPT_FORMAT = 1\n",
        "corpusgen/parallel/tool.sh": "#!/bin/sh\nexit 0\n",
        "corpusgen/reasoning_v2/catalog.py": "WIKIDATA_CATALOG_API = 1\n",
        "corpusgen/reasoning_v2/source_lock.py": "SOURCE_LOCK_FORMAT = 1\n",
        "scripts/build_parallel_corpus.py": "raise SystemExit(0)\n",
        "scripts/package_aws_corpus_builder.py": SCRIPT_SOURCE.read_bytes(),
        "sources/Wikidata-CC0-1.0.txt": "CC0 fixture notice\n",
        "sources/current-dataset-licenses.json": '{"schema_version":1}\n',
        "sources/wikidata5m.lock.json": '{"schema_version":1}\n',
        TOKENIZER_ASSETS[0]: b"tokenizer fixture a\n",
        TOKENIZER_ASSETS[1]: b"tokenizer fixture b\n",
        "tests/test_aws_corpus_builder_contracts.py": "def test_contract(): pass\n",
        "tests/test_parallel_corpus.py": "def test_parallel(): pass\n",
        "tests/test_reasoning_v2_catalog.py": "def test_catalog(): pass\n",
        "tests/test_reasoning_v2_source_lock.py": "def test_source_lock(): pass\n",
    }


def _new_repo(tmp_path: Path, *, name: str = "source") -> Path:
    root = tmp_path / name
    root.mkdir()
    completed = _run(root, "git", "init", "-q")
    assert completed.returncode == 0, completed.stderr
    for relative, data in _fixture_files().items():
        _write(
            root,
            relative,
            data,
            executable=relative == "corpusgen/parallel/tool.sh",
        )
    _write(root, "configs/outputs/old.json", "{}\n")
    _write(root, "scripts/__pycache__/old.pyc", b"cache\n")
    _write(root, "scripts/checkpoints/old.bin", b"checkpoint\n")
    _write(root, "scripts/pilot-artifacts/old.json", "{}\n")
    _commit(root, "fixture")
    return root


def _manifest(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    value = json.loads(payload)
    assert payload == _canonical(value)
    return value


def test_package_allowlist_is_exact():
    assert REQUIRED_PREFIXES == (
        "cluster/aws/corpus_builder/",
        "cluster/profiles/",
        "configs/",
        "corpusgen/parallel/",
        "corpusgen/reasoning_v2/",
        "scripts/",
        "sources/",
        "vendor/tiktoken/",
    )
    assert REQUIRED_FILES == (
        "pyproject.toml",
        "requirements.txt",
        "scripts/build_parallel_corpus.py",
        "scripts/package_aws_corpus_builder.py",
    )


def test_builder_package_is_byte_identical_canonical_and_closed(tmp_path: Path):
    source = _new_repo(tmp_path)

    first = build_corpus_builder_package(source, tmp_path / "a")
    second = build_corpus_builder_package(source, tmp_path / "b")

    assert isinstance(first, BuilderPackage)
    assert first.sha256 == second.sha256
    assert first.archive.read_bytes() == second.archive.read_bytes()
    assert first.archive.name == ARCHIVE_NAME
    assert first.manifest.name == MANIFEST_NAME
    assert first.sha256_file.name == SHA256_NAME
    assert first.bytes == first.archive.stat().st_size
    assert first.sha256 == hashlib.sha256(first.archive.read_bytes()).hexdigest()
    assert first.revision == _git(source, "rev-parse", "--verify", "HEAD")
    assert first.sha256_file.read_text(encoding="ascii") == (
        f"{first.sha256}  {ARCHIVE_NAME}\n"
    )
    assert first.members == tuple(sorted(first.members))
    assert "README.md" not in first.members
    assert not any(path.startswith("tests/") for path in first.members)
    assert "configs/outputs/old.json" not in first.members
    assert "scripts/__pycache__/old.pyc" not in first.members
    assert "scripts/checkpoints/old.bin" not in first.members
    assert "scripts/pilot-artifacts/old.json" not in first.members
    assert set(REQUIRED_FILES).issubset(first.members)
    assert set(REQUIRED_AUTHORITIES) - {
        path for path in REQUIRED_AUTHORITIES if path.startswith("tests/")
    } <= set(first.members)

    archive_bytes = first.archive.read_bytes()
    assert archive_bytes[:2] == b"\x1f\x8b"
    assert archive_bytes[4:8] == b"\0\0\0\0"
    with tarfile.open(first.archive, mode="r:gz") as archive:
        infos = archive.getmembers()
        assert tuple(info.name for info in infos) == first.members
        assert all(info.isreg() for info in infos)
        assert all(info.uid == 0 and info.gid == 0 for info in infos)
        assert all(info.uname == "" and info.gname == "" for info in infos)
        assert all(info.mtime == 0 for info in infos)
        assert all(stat.S_IMODE(info.mode) in {0o644, 0o755} for info in infos)
        assert archive.getmember("corpusgen/parallel/tool.sh").mode == 0o755
        assert archive.getmember("requirements.txt").mode == 0o644

    manifest = _manifest(first.manifest)
    assert manifest["format"] == "memorysplit-corpus-builder-package-v1"
    assert manifest["schema_version"] == 1
    assert manifest["revision"] == first.revision
    assert manifest["archive"] == {
        "bytes": first.bytes,
        "path": ARCHIVE_NAME,
        "sha256": first.sha256,
    }
    rows = manifest["members"]
    assert isinstance(rows, list)
    assert tuple(row["path"] for row in rows) == first.members
    tree_rows = {}
    output = _git(source, "ls-tree", "-r", "--full-tree", first.revision)
    for line in output.splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, object_id = metadata.split()
        tree_rows[path] = (mode, kind, object_id)
    with tarfile.open(first.archive, mode="r:gz") as archive:
        for row in rows:
            data = archive.extractfile(row["path"]).read()
            git_mode, kind, object_id = tree_rows[row["path"]]
            assert kind == "blob"
            assert row == {
                "bytes": len(data),
                "mode": "0755" if git_mode == "100755" else "0644",
                "object_id": object_id,
                "path": row["path"],
                "sha256": hashlib.sha256(data).hexdigest(),
            }


@pytest.mark.parametrize(
    "mutation",
    ("dirty", "secret", "credential-path", "foreign-member", "symlink"),
)
def test_builder_package_rejects_dirty_tree_secret_and_unreviewed_member(
    tmp_path: Path,
    mutation: str,
):
    source = _new_repo(tmp_path)
    if mutation == "dirty":
        _write(source, "requirements.txt", "pytest\nchanged\n")
    elif mutation == "secret":
        _write(
            source,
            "configs/runtime.env",
            "AWS_ACCESS_KEY_ID=AKIA0123456789ABCDEF\n",
        )
        _commit(source, "add secret")
    elif mutation == "credential-path":
        _write(source, "configs/credentials/runtime.json", "{}\n")
        _commit(source, "add credential path")
    elif mutation == "foreign-member":
        _write(source, "scripts/unreviewed.py", "UNREVIEWED = True\n")
    else:
        target = source / "scripts" / "linked.py"
        os.symlink("build_parallel_corpus.py", target)
        _commit(source, "add symlink")

    with pytest.raises(PackageError):
        build_corpus_builder_package(source, tmp_path / "output")


def test_builder_package_rejects_structured_secret_fields(tmp_path: Path):
    source = _new_repo(tmp_path)
    _write(
        source,
        "configs/runtime.json",
        '{"api_token":"this-value-must-not-ship"}\n',
    )
    _commit(source, "add structured secret")

    with pytest.raises(PackageError, match="secret"):
        build_corpus_builder_package(source, tmp_path / "output")


@pytest.mark.parametrize("missing", REQUIRED_AUTHORITIES)
def test_builder_package_requires_reviewed_authorities(tmp_path: Path, missing: str):
    source = _new_repo(tmp_path)
    _git(source, "rm", missing)
    _commit(source, f"remove {Path(missing).name}")

    with pytest.raises(PackageError, match="required"):
        build_corpus_builder_package(source, tmp_path / "output")


def test_builder_package_rejects_non_repository_and_internal_output(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(PackageError):
        build_corpus_builder_package(plain, tmp_path / "plain-output")

    source = _new_repo(tmp_path, name="source")
    with pytest.raises(PackageError, match="outside"):
        build_corpus_builder_package(source, source / "dist")


def test_package_cli_prints_the_canonical_manifest(tmp_path: Path):
    source = _new_repo(tmp_path)
    output = tmp_path / "cli-output"

    completed = _run(
        ROOT,
        sys.executable,
        str(SCRIPT_SOURCE),
        "--source-root",
        str(source),
        "--output-dir",
        str(output),
    )

    assert completed.returncode == 0, completed.stderr
    manifest = output / MANIFEST_NAME
    assert completed.stdout == manifest.read_text(encoding="utf-8")
    assert completed.stderr == ""
