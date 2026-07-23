from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "package_illumina_handoff.py"
PLANNED_DIRECTORIES = {
    ".cursor/skills/memorysplit-cluster/",
    "cluster/profiles/",
    "cluster/slurm/",
    "configs/",
    "corpusgen/parallel/",
    "corpusgen/reasoning/",
    "evals/confirmatory/",
    "fixtures/current-smoke/",
    "msctl/",
    "scripts/",
    "sources/",
    "tests/",
    "train/",
    "vendor/tiktoken/",
}


@pytest.fixture
def package_module():
    assert SCRIPT.is_file(), "Illumina packager has not been implemented"
    spec = importlib.util.spec_from_file_location(
        "package_illumina_handoff_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _write(path: Path, data: bytes | str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    path.chmod(0o755 if executable else 0o644)


def _minimal_repo(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    files: dict[str, bytes | str] = {
        "AGENT-START.md": "# Agent\nUse msctl dry-run then --apply.\n",
        "DATASET-POINTER.json": json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "fixture",
                "provider": "illumina-usfc-prd",
            }
        )
        + "\n",
        "requirements.txt": "pytest>=8\n",
        "pytest.ini": "[pytest]\n",
        "msctl/__init__.py": '"""fixture"""\n',
        "msctl/__main__.py": "raise SystemExit(0)\n",
        "corpusgen/__init__.py": (
            "def _hf_download_command():\n"
            "    return ['download']\n"
        ),
        "evals/__init__.py": "",
        "fixtures/current-smoke/train.bin": b"tiny smoke corpus",
        "train/__init__.py": "",
        "scripts/package_illumina_handoff.py": SCRIPT.read_bytes(),
        "tests/test_msctl.py": "def test_fixture():\n    assert True\n",
        "tests/test_package_illumina_handoff.py": "def test_fixture():\n    assert True\n",
        "cluster/profiles/illumina-usfc-prd.json": (
            REPO_ROOT / "cluster" / "profiles" / "illumina-usfc-prd.json"
        ).read_bytes(),
        "cluster/slurm/v2_seed0.sbatch": (
            REPO_ROOT / "cluster" / "slurm" / "v2_seed0.sbatch"
        ).read_bytes(),
        "cluster/slurm/v2_evaluate.sbatch": (
            REPO_ROOT / "cluster" / "slurm" / "v2_evaluate.sbatch"
        ).read_bytes(),
        ".cursor/skills/memorysplit-cluster/SKILL.md": (
            REPO_ROOT
            / ".cursor"
            / "skills"
            / "memorysplit-cluster"
            / "SKILL.md"
        ).read_bytes(),
        "Memory-split-design.md": "known excluded design documentation\n",
        "README.md": "known excluded documentation\n",
        "docs/history.md": "known excluded documentation\n",
        "schemas/mit-cluster-profile-v1.schema.json": "{}\n",
        "sources/Wikidata-CC0-1.0.txt": "CC0 fixture\n",
        "sources/wikidata5m.lock.json": "{}\n",
        "data/full-corpus.bin": b"excluded corpus",
        "outputs/run/checkpoint.pt": b"excluded checkpoint",
        "outputs/run/logs/worker.log": b"excluded log",
        ".cache/compiler.bin": b"excluded cache",
    }
    for relative in (
        "configs/cohort-assignment-v2.json",
        "configs/preregistration-v2.yaml",
        *(
            f"configs/360m-v2/{arm}-s{seed}.yaml"
            for seed in range(5)
            for arm in ("dense", "split90")
        ),
    ):
        files[relative] = (REPO_ROOT / relative).read_bytes()
    for relative, data in files.items():
        _write(
            root / relative,
            data,
            executable=relative.endswith((".sbatch", ".sh")),
        )
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Package Test")
    _git(root, "config", "user.email", "package@example.invalid")
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "add", "-f", ".")
    _git(root, "commit", "-qm", "fixture")
    return root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_production_dataset_pointer_is_nonmaterialized_and_illumina_only():
    pointer = json.loads((REPO_ROOT / "DATASET-POINTER.json").read_text())

    assert pointer["schema_version"] == 1
    assert pointer["provider"] == "illumina-usfc-prd"
    assert pointer["shared_root_env"] == "MS_SHARED_ROOT"
    assert pointer["shared_root_prefix"] == "/illumina"
    assert pointer["materialization"] == "slurm"
    assert pointer["full_corpus_in_release"] is False
    assert ".." not in PurePosixPath(pointer["relative_path"]).parts


def test_double_build_is_byte_identical_with_internal_and_external_hashes(
    tmp_path,
    package_module,
):
    source = _minimal_repo(tmp_path)

    first = package_module.build_handoff(
        source_root=source,
        out_dir=tmp_path / "out-a",
        apply=True,
    )
    second = package_module.build_handoff(
        source_root=source,
        out_dir=tmp_path / "out-b",
        apply=True,
    )

    assert first.archive.name == second.archive.name
    assert first.archive.read_bytes() == second.archive.read_bytes()
    assert first.release.read_bytes() == second.release.read_bytes()
    assert first.sha256_file.read_bytes() == second.sha256_file.read_bytes()
    digest = _sha256(first.archive)
    assert first.sha256_file.read_text() == f"{digest}  {first.archive.name}\n"
    release = json.loads(first.release.read_text())
    assert release == json.loads(second.release.read_text())
    assert release["archive"] == {
        "path": first.archive.name,
        "sha256": digest,
        "bytes": first.archive.stat().st_size,
    }
    assert release["source"] == {
        "commit": _git(source, "rev-parse", "HEAD"),
        "dirty": False,
    }


def test_zip_has_closed_members_normalized_metadata_and_all_planned_directories(
    tmp_path,
    package_module,
):
    source = _minimal_repo(tmp_path)
    artifacts = package_module.build_handoff(
        source_root=source,
        out_dir=tmp_path / "out",
        apply=True,
    )

    with zipfile.ZipFile(artifacts.archive) as archive:
        infos = archive.infolist()
        names = {info.filename for info in infos}
        assert PLANNED_DIRECTORIES <= names
        assert "SHA256SUMS" in names
        assert "RELEASE-METADATA.json" in names
        assert "fixtures/current-smoke/train.bin" in names
        assert "sources/wikidata5m.lock.json" in names
        assert "schemas/mit-cluster-profile-v1.schema.json" not in names
        assert "README.md" not in names
        assert not any(name.startswith("docs/") for name in names)
        assert not any(name.startswith("data/") for name in names)
        assert not any(name.startswith("outputs/") for name in names)
        assert not any("checkpoint" in name.lower() for name in names)
        assert not any("cache" in name.lower() for name in names)
        assert not any("logs/" in name.lower() for name in names)
        for info in infos:
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert not info.filename.startswith("/")
            assert ".." not in PurePosixPath(info.filename).parts
            mode = stat.S_IMODE(info.external_attr >> 16)
            expected = (
                0o755
                if info.is_dir()
                or info.filename.endswith((".sbatch", ".sh"))
                else 0o644
            )
            assert mode == expected

        checksum_lines = archive.read("SHA256SUMS").decode().splitlines()
        checksum_paths = set()
        for line in checksum_lines:
            digest, relative = line.split("  ", 1)
            assert digest == hashlib.sha256(archive.read(relative)).hexdigest()
            checksum_paths.add(relative)
        regular_members = {
            info.filename
            for info in infos
            if not info.is_dir() and info.filename != "SHA256SUMS"
        }
        assert checksum_paths == regular_members


def test_packager_rejects_dirty_git_before_writing(tmp_path, package_module):
    source = _minimal_repo(tmp_path)
    (source / "untracked.py").write_text("print('dirty')\n")
    out = tmp_path / "out"

    with pytest.raises(package_module.PackageError, match="dirty"):
        package_module.build_handoff(source_root=source, out_dir=out)

    assert not out.exists()


def test_packager_rejects_tracked_symlinks(tmp_path, package_module):
    source = _minimal_repo(tmp_path)
    os.symlink("__init__.py", source / "msctl" / "linked.py")
    _git(source, "add", "msctl/linked.py")
    _git(source, "commit", "-qm", "add symlink")

    with pytest.raises(package_module.PackageError, match="symlink"):
        package_module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
        )


def test_packager_rejects_secret_material_without_echoing_it(
    tmp_path,
    package_module,
):
    source = _minimal_repo(tmp_path)
    secret = "this-is-a-real-looking-secret-value"
    secret_name = "AWS_SECRET" + "_ACCESS_KEY"
    (source / "msctl" / "leak.py").write_text(
        f'{secret_name} = "{secret}"\n'
    )
    _git(source, "add", "msctl/leak.py")
    _git(source, "commit", "-qm", "add leak")

    with pytest.raises(package_module.PackageError) as caught:
        package_module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
        )

    assert "secret" in str(caught.value).lower()
    assert secret not in str(caught.value)


def test_packager_rejects_unknown_tracked_paths(tmp_path, package_module):
    source = _minimal_repo(tmp_path)
    _write(source / "mystery" / "tool.py", "print('unknown')\n")
    _git(source, "add", "mystery/tool.py")
    _git(source, "commit", "-qm", "add unknown")

    with pytest.raises(package_module.PackageError, match="unknown"):
        package_module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
        )


def test_packager_cli_emits_one_json_object(tmp_path, package_module):
    source = _minimal_repo(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-root",
            str(source),
            "--out-dir",
            str(tmp_path / "out"),
            "--apply",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    lines = completed.stdout.splitlines()
    assert len(lines) == 1
    report = json.loads(lines[0])
    assert report["ok"] is True
    assert report["sha256"] == _sha256(Path(report["archive"]))


def test_packager_reads_immutable_git_objects_after_snapshot(
    tmp_path,
    package_module,
    monkeypatch,
):
    source = _minimal_repo(tmp_path)
    committed = (source / "AGENT-START.md").read_bytes()
    original_run_git = package_module._run_git
    raced = False

    def racing_git(root, *arguments):
        nonlocal raced
        output = original_run_git(root, *arguments)
        if not raced and arguments and arguments[0] in {"ls-files", "ls-tree"}:
            (source / "AGENT-START.md").write_bytes(b"raced worktree bytes\n")
            raced = True
        return output

    monkeypatch.setattr(package_module, "_run_git", racing_git)
    artifacts = package_module.build_handoff(
        source_root=source,
        out_dir=tmp_path / "out",
        apply=True,
    )

    assert raced is True
    with zipfile.ZipFile(artifacts.archive) as archive:
        assert archive.read("AGENT-START.md") == committed


@pytest.mark.parametrize(
    ("secret_name", "secret_value", "quote"),
    [
        (
            "AWS_" + "SECRET_ACCESS_KEY",
            "abcdefghijklmnopqrstuvwx",
            "",
        ),
        ("HF_" + "TOKEN", "hf_" + "abcdefghijklmnopqrstuvwxyz", "'"),
        (
            "GITHUB_" + "TOKEN",
            "github_" + "pat_abcdefghijklmnopqrstuvwxyz",
            '"',
        ),
        ("api_" + "token", "abcdefghijklmnopqrstuvwxyz012345", ""),
        ("TO" + "KEN", "abcdefghijklmnopqrstuvwxyz012345", '"'),
        (
            "OPENAI_" + "API_KEY",
            "sk-" + "abcdefghijklmnopqrstuvwxyz012345",
            '"',
        ),
    ],
)
def test_packager_rejects_quoted_and_unquoted_common_secret_assignments(
    tmp_path,
    package_module,
    secret_name,
    secret_value,
    quote,
):
    secret_bytes = (
        f"{secret_name} = {quote}{secret_value}{quote}\n".encode()
    )
    source = _minimal_repo(tmp_path)
    leak = source / "msctl" / "leak.py"
    leak.write_bytes(secret_bytes)
    _git(source, "add", "msctl/leak.py")
    _git(source, "commit", "-qm", "add secret fixture")

    with pytest.raises(package_module.PackageError) as caught:
        package_module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
        )

    message = str(caught.value)
    assert "secret" in message.lower()
    assert secret_bytes.strip().decode() not in message


def test_packager_cli_defaults_to_json_dry_run_without_publication(tmp_path):
    source = _minimal_repo(tmp_path)
    out = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-root",
            str(source),
            "--out-dir",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    lines = completed.stdout.splitlines()
    assert len(lines) == 1
    report = json.loads(lines[0])
    assert report["ok"] is True
    assert report["dry_run"] is True
    assert report["published"] is False
    assert not out.exists()


@pytest.mark.parametrize("arguments", [["--help"], ["--not-an-option"]])
def test_packager_help_and_usage_are_one_json_object(arguments):
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.stderr == ""
    lines = completed.stdout.splitlines()
    assert len(lines) == 1
    report = json.loads(lines[0])
    assert isinstance(report, dict)
    assert report["schema_version"] == 1
    if arguments == ["--help"]:
        assert completed.returncode == 0
        assert report["ok"] is True
        assert report["dry_run"] is True
        assert "help" in report
    else:
        assert completed.returncode != 0
        assert report["ok"] is False
        assert report["error"]["code"] == "CLI_USAGE"


def test_packager_apply_atomically_publishes_one_no_replace_release_directory(
    tmp_path,
):
    source = _minimal_repo(tmp_path)
    out = tmp_path / "out"
    command = [
        sys.executable,
        str(SCRIPT),
        "--source-root",
        str(source),
        "--out-dir",
        str(out),
        "--apply",
    ]

    first = subprocess.run(command, capture_output=True, text=True, check=False)
    second = subprocess.run(command, capture_output=True, text=True, check=False)

    assert first.returncode == 0
    first_report = json.loads(first.stdout)
    release_dir = Path(first_report["release_dir"])
    assert release_dir.parent == out
    assert release_dir.is_dir()
    assert {path.name for path in release_dir.iterdir()} == {
        Path(first_report["archive"]).name,
        Path(first_report["sha256_file"]).name,
        "RELEASE.json",
    }
    assert second.returncode != 0
    second_report = json.loads(second.stdout)
    assert second_report["error"]["code"] == "RELEASE_EXISTS"


def test_packager_verifies_internal_checksums_before_publish(
    tmp_path,
    package_module,
    monkeypatch,
):
    source = _minimal_repo(tmp_path)
    original_write_zip = package_module._write_zip

    def corrupting_write_zip(path, *, payload, directories):
        corrupted = dict(payload)
        corrupted["SHA256SUMS"] = b"0" * 64 + b"  AGENT-START.md\n"
        original_write_zip(path, payload=corrupted, directories=directories)

    monkeypatch.setattr(package_module, "_write_zip", corrupting_write_zip)

    with pytest.raises(package_module.PackageError, match="checksum|duplicate"):
        package_module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
        )

    assert not (tmp_path / "out").exists()
