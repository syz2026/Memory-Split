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

import cluster.aws.corpus_builder.package as package_module
from cluster.aws.corpus_builder.package import (
    ARCHIVE_NAME,
    MANIFEST_NAME,
    REQUIRED_FILES,
    REQUIRED_PREFIXES,
    SHA256_NAME,
    BuilderPackage,
    PackageError,
    _is_disposable,
    _sanitized_git_environment,
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
WIKIDATA_SOURCE = "corpusgen/reasoning_v2/wikidata_source.py"
RENDERERS = "corpusgen/reasoning_v2/renderers.py"
WIKIDATA_SOURCE_SURFACE = """\
class WikidataDerivedView:
    pass

class WikidataDerivedViewReceipt:
    pass

def build_wikidata_derived_view(*args, **kwargs):
    pass

def verify_wikidata_derived_view(*args, **kwargs):
    pass

def iter_v2_training_triples(*args, **kwargs):
    pass

def iter_v2_aliases(*args, **kwargs):
    pass

def iter_distinct_training_edges(*args, **kwargs):
    pass

def lookup_training_triple(*args, **kwargs):
    pass

def lookup_alias(*args, **kwargs):
    pass
"""
REQUIRED_AUTHORITIES = (
    "cluster/aws/corpus_builder/contracts.py",
    PROFILE,
    "configs/current-dataset-lock.json",
    "configs/reasoning-dataset-v2.json",
    "corpusgen/wikidata5m.py",
    "corpusgen/parallel/adapters.py",
    "corpusgen/parallel/catalog.py",
    "corpusgen/parallel/publication.py",
    "corpusgen/reasoning_v2/catalog.py",
    RENDERERS,
    "corpusgen/reasoning_v2/source_lock.py",
    WIKIDATA_SOURCE,
    "sources/Wikidata-CC0-1.0.txt",
    "sources/current-dataset-licenses.json",
    "sources/wikidata5m.lock.json",
    *TOKENIZER_ASSETS,
    "tests/test_aws_corpus_builder_contracts.py",
    "tests/test_parallel_corpus.py",
    "tests/test_reasoning_v2_catalog.py",
    "tests/test_reasoning_v2_renderers.py",
    "tests/test_reasoning_v2_source_lock.py",
    "tests/test_reasoning_v2_wikidata_source.py",
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
        "requirements.txt": "pytest\n",
        "cluster/aws/corpus_builder/__init__.py": '"""fixture package"""\n',
        "cluster/aws/corpus_builder/contracts.py": "FORMAT = 'fixture-v1'\n",
        "cluster/aws/corpus_builder/package.py": PACKAGE_SOURCE.read_bytes(),
        PROFILE: '{"profile_id":"aws-i4i.16xlarge-corpus-v1"}\n',
        "configs/current-dataset-lock.json": '{"schema_version":1}\n',
        "configs/reasoning-dataset-v2.json": '{"schema_version":2}\n',
        "corpusgen/parallel/adapters.py": "PRODUCTION_ADAPTERS = ()\n",
        "corpusgen/parallel/catalog.py": "CATALOG_FORMAT = 1\n",
        "corpusgen/parallel/publication.py": "RECEIPT_FORMAT = 1\n",
        "corpusgen/reasoning_v2/catalog.py": (
            "class WikidataGraphCatalogSource:\n"
            "    pass\n"
        ),
        RENDERERS: "class WikidataGraphRenderer:\n    pass\n",
        "corpusgen/reasoning_v2/source_lock.py": "SOURCE_LOCK_FORMAT = 1\n",
        WIKIDATA_SOURCE: WIKIDATA_SOURCE_SURFACE,
        "corpusgen/wikidata5m.py": "WIKIDATA_ARCHIVE_API = 1\n",
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
        "tests/test_reasoning_v2_renderers.py": "def test_renderers(): pass\n",
        "tests/test_reasoning_v2_source_lock.py": "def test_source_lock(): pass\n",
        "tests/test_reasoning_v2_wikidata_source.py": "def test_wikidata(): pass\n",
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
            executable=relative == "scripts/build_parallel_corpus.py",
        )
    _write(root, "configs/outputs/old.json", "{}\n")
    _write(root, "configs/Artifacts-v2/model.bin", b"artifact\n")
    _write(root, "scripts/__pycache__/old.pyc", b"cache\n")
    _write(root, "scripts/checkpoints/old.bin", b"checkpoint\n")
    _write(root, "scripts/Checkpoint-100/model.bin", b"checkpoint\n")
    _write(root, "scripts/pilot-artifacts/old.json", "{}\n")
    _write(root, "scripts/PILOT-v2.json", "{}\n")
    _write(root, "corpusgen/parallel/Output-2026/data.bin", b"output\n")
    _write(root, "vendor/tiktoken/Cache_2/old.bin", b"cache\n")
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
    assert "pyproject.toml" not in first.members
    assert not any(path.startswith("tests/") for path in first.members)
    assert "configs/outputs/old.json" not in first.members
    assert "configs/Artifacts-v2/model.bin" not in first.members
    assert "scripts/__pycache__/old.pyc" not in first.members
    assert "scripts/checkpoints/old.bin" not in first.members
    assert "scripts/Checkpoint-100/model.bin" not in first.members
    assert "scripts/pilot-artifacts/old.json" not in first.members
    assert "scripts/PILOT-v2.json" not in first.members
    assert "corpusgen/parallel/Output-2026/data.bin" not in first.members
    assert "vendor/tiktoken/Cache_2/old.bin" not in first.members
    assert all(path in first.members for path in TOKENIZER_ASSETS)
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
        assert archive.getmember("scripts/build_parallel_corpus.py").mode == 0o755
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


def test_artifact_filter_preserves_reviewed_source_and_tokenizer_binary_names():
    for path in (
        "sources/training.bin",
        "vendor/tiktoken/tokenizer.bin",
        "corpusgen/reasoning_v2/index.bin",
    ):
        assert not _is_disposable(path)


@pytest.mark.parametrize(
    ("path", "missing_symbol"),
    (
        (WIKIDATA_SOURCE, None),
        (WIKIDATA_SOURCE, "build_wikidata_derived_view"),
        ("corpusgen/reasoning_v2/catalog.py", "WikidataGraphCatalogSource"),
        (RENDERERS, "WikidataGraphRenderer"),
    ),
)
def test_builder_package_requires_completed_production_pipeline(
    tmp_path: Path,
    path: str,
    missing_symbol: str | None,
):
    source = _new_repo(tmp_path)
    if missing_symbol is None:
        _git(source, "rm", path)
    else:
        target = source / path
        text = target.read_text(encoding="utf-8")
        target.write_text(
            text.replace(missing_symbol, f"Missing{missing_symbol}"),
            encoding="utf-8",
        )
    _commit(source, "make production pipeline incomplete")

    with pytest.raises(PackageError) as raised:
        build_corpus_builder_package(source, tmp_path / "output")

    assert raised.value.code == "INCOMPLETE_PRODUCTION_PIPELINE"
    assert str(raised.value).startswith("INCOMPLETE_PRODUCTION_PIPELINE:")


def test_builder_package_rejects_unsupported_production_renderer(tmp_path: Path):
    source = _new_repo(tmp_path)
    _write(
        source,
        "corpusgen/parallel/adapters.py",
        "class UnsupportedProductionRenderer:\n    pass\n",
    )
    _commit(source, "restore unsupported renderer")

    with pytest.raises(PackageError) as raised:
        build_corpus_builder_package(source, tmp_path / "output")

    assert raised.value.code == "INCOMPLETE_PRODUCTION_PIPELINE"


def test_all_package_git_subprocesses_disable_lazy_fetch(tmp_path: Path, monkeypatch):
    source = _new_repo(tmp_path)
    assert _sanitized_git_environment()["GIT_NO_LAZY_FETCH"] == "1"
    real_run = subprocess.run
    observed = 0

    def checked_run(*args, **kwargs):
        nonlocal observed
        command = args[0]
        if command and command[0] == "git":
            observed += 1
            assert kwargs["env"]["GIT_NO_LAZY_FETCH"] == "1"
        return real_run(*args, **kwargs)

    monkeypatch.setattr(package_module.subprocess, "run", checked_run)
    build_corpus_builder_package(source, tmp_path / "output")
    assert observed > 0


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
        _commit(source, "add committed unreviewed member")
    else:
        target = source / "scripts" / "linked.py"
        os.symlink("build_parallel_corpus.py", target)
        _commit(source, "add symlink")

    match = "reviewed member inventory" if mutation == "foreign-member" else None
    with pytest.raises(PackageError, match=match):
        build_corpus_builder_package(source, tmp_path / "output")


def test_builder_package_rejects_structured_secret_fields(tmp_path: Path):
    source = _new_repo(tmp_path)
    _write(
        source,
        "configs/current-dataset-lock.json",
        '{"api_token":"this-value-must-not-ship"}\n',
    )
    _commit(source, "replace reviewed config with structured secret")

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


def test_builder_package_rejects_physical_output_aliases_and_ancestors(
    tmp_path: Path,
):
    source = _new_repo(tmp_path, name="source")
    alias = tmp_path / "source-alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(PackageError, match="disjoint"):
        build_corpus_builder_package(source, alias / "dist")

    container = tmp_path / "container"
    container.mkdir()
    nested_source = _new_repo(container, name="source")
    with pytest.raises(PackageError, match="disjoint"):
        build_corpus_builder_package(nested_source, container)


@pytest.mark.parametrize("mutation", ("output-alias", "dirty-source"))
def test_builder_package_rechecks_paths_and_tree_after_writing(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
):
    source = _new_repo(tmp_path)
    output = tmp_path / "output"
    displaced = tmp_path / "displaced-output"
    real_write = package_module._write_atomic
    writes = 0

    def mutating_write(path: Path, data: bytes) -> None:
        nonlocal writes
        real_write(path, data)
        writes += 1
        if writes != 3:
            return
        if mutation == "output-alias":
            output.rename(displaced)
            output.symlink_to(source, target_is_directory=True)
        else:
            _write(source, "untracked-after-write.txt", "changed\n")

    monkeypatch.setattr(package_module, "_write_atomic", mutating_write)
    with pytest.raises(PackageError):
        build_corpus_builder_package(source, output)
    assert writes == 3


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
