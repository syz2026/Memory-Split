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
import yaml


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
        "msctl/cohort.py": (REPO_ROOT / "msctl" / "cohort.py").read_bytes(),
        "msctl/errors.py": (REPO_ROOT / "msctl" / "errors.py").read_bytes(),
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
    ):
        files[relative] = (REPO_ROOT / relative).read_bytes()
    for config in sorted((REPO_ROOT / "configs" / "360m-v2").iterdir()):
        files[f"configs/360m-v2/{config.name}"] = config.read_bytes()
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


def _commit_mutation(root: Path, message: str) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", message)


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
    )
    second = package_module.build_handoff(
        source_root=source,
        out_dir=tmp_path / "out-b",
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


def test_zip_semantically_proves_exactly_seed_zero_dense_and_split90(
    tmp_path,
    package_module,
):
    source = _minimal_repo(tmp_path)
    artifacts = package_module.build_handoff(
        source_root=source,
        out_dir=tmp_path / "out",
    )

    with zipfile.ZipFile(artifacts.archive) as archive:
        assignment = json.loads(
            archive.read("configs/cohort-assignment-v2.json")
        )
        config_names = {
            name
            for name in archive.namelist()
            if name.startswith("configs/360m-v2/") and not name.endswith("/")
        }
        configs = {
            name: yaml.safe_load(archive.read(name))
            for name in config_names
        }
        metadata = json.loads(archive.read("RELEASE-METADATA.json"))

    assert assignment["provider_seeds"]["illumina-usfc-prd"] == [0]
    assert assignment["provider_seeds"]["aws-p5.48xlarge"] == [1, 2, 3, 4]
    assert config_names == {
        "configs/360m-v2/dense-s0.yaml",
        "configs/360m-v2/split90-s0.yaml",
    }
    assert {
        (config["seed"], config["condition"]) for config in configs.values()
    } == {(0, "dense"), (0, "split90")}
    assert all(
        config["cohort_id"] == assignment["cohort_id"]
        for config in configs.values()
    )
    assert metadata["seed_assignment"] == {
        "cohort_id": "memorysplit-confirmatory-v2-360m-n5",
        "provider": "illumina-usfc-prd",
        "seeds": [0],
        "arms": ["dense", "split90"],
    }
    member_hashes = {
        row["path"]: row["sha256"] for row in metadata["members"]
    }
    preregistration = "configs/preregistration-v2.yaml"
    assert metadata["preregistration_sha256"] == _sha256(
        source / preregistration
    )
    for name in config_names | {
        "configs/cohort-assignment-v2.json",
        preregistration,
    }:
        assert member_hashes[name] == _sha256(source / name)


@pytest.mark.parametrize(
    "mutation",
    [
        "seed_zero_omission",
        "seed_one_inclusion",
        "generic_split",
        "wrong_token_math",
        "partial_pair",
    ],
)
def test_packager_rejects_invalid_provider_cohort_before_output(
    tmp_path,
    package_module,
    mutation,
):
    source = _minimal_repo(tmp_path)
    assignment_path = source / "configs" / "cohort-assignment-v2.json"
    dense_path = source / "configs" / "360m-v2" / "dense-s0.yaml"
    split_path = source / "configs" / "360m-v2" / "split90-s0.yaml"

    if mutation in {"seed_zero_omission", "seed_one_inclusion"}:
        assignment = json.loads(assignment_path.read_text())
        if mutation == "seed_zero_omission":
            assignment["provider_seeds"]["illumina-usfc-prd"] = []
        else:
            assignment["provider_seeds"]["illumina-usfc-prd"] = [0, 1]
            assignment["provider_seeds"]["aws-p5.48xlarge"] = [2, 3, 4]
        assignment_path.write_text(
            json.dumps(assignment, indent=2, sort_keys=True) + "\n"
        )
    elif mutation == "generic_split":
        split = yaml.safe_load(split_path.read_text())
        split["condition"] = "split"
        split_path.write_text(yaml.safe_dump(split, sort_keys=False))
    elif mutation == "wrong_token_math":
        dense = yaml.safe_load(dense_path.read_text())
        dense["total_tokens"] -= 1
        dense_path.write_text(yaml.safe_dump(dense, sort_keys=False))
    elif mutation == "partial_pair":
        split_path.unlink()
    else:  # pragma: no cover - parameterization guard
        raise AssertionError(mutation)
    _commit_mutation(source, mutation)
    out = tmp_path / "out"

    with pytest.raises(package_module.PackageError):
        package_module.build_handoff(source_root=source, out_dir=out)

    assert not out.exists()


def test_packager_rejects_tracked_root_seed_config_without_output(
    tmp_path,
    package_module,
):
    source = _minimal_repo(tmp_path)
    _write(
        source / "configs" / "dense-s1.yaml",
        (source / "configs" / "360m-v2" / "dense-s1.yaml").read_bytes(),
    )
    _commit_mutation(source, "add unallowlisted root seed config")
    out = tmp_path / "out"

    with pytest.raises(package_module.PackageError, match="unknown tracked path"):
        package_module.build_handoff(source_root=source, out_dir=out)

    assert not out.exists()


def test_packager_rejects_assignment_config_hash_mismatch(
    tmp_path,
    package_module,
    monkeypatch,
):
    source = _minimal_repo(tmp_path)
    original = package_module._read_member

    def raced_read(root, relative):
        data = original(root, relative)
        if relative == "configs/360m-v2/dense-s0.yaml":
            return data + b"# raced after cohort validation\n"
        return data

    monkeypatch.setattr(package_module, "_read_member", raced_read)

    with pytest.raises(package_module.PackageError, match="hash"):
        package_module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
        )


def test_packager_rejects_preregistration_replacement_after_snapshot(
    tmp_path,
    package_module,
    monkeypatch,
):
    source = _minimal_repo(tmp_path)
    original = package_module._read_member

    def raced_read(root, relative):
        data = original(root, relative)
        if relative == "configs/preregistration-v2.yaml":
            replaced = data.replace(
                b"model_parameters: 356033536",
                b"model_parameters: 356033535",
                1,
            )
            assert replaced != data
            return replaced
        return data

    monkeypatch.setattr(package_module, "_read_member", raced_read)
    out = tmp_path / "out"

    with pytest.raises(
        package_module.PackageError,
        match="preregistration hash mismatch",
    ):
        package_module.build_handoff(source_root=source, out_dir=out)

    assert not out.exists()


def test_packager_refuses_to_replace_existing_release_set(
    tmp_path,
    package_module,
):
    source = _minimal_repo(tmp_path)
    out = tmp_path / "out"
    first = package_module.build_handoff(source_root=source, out_dir=out)
    originals = {
        path: path.read_bytes()
        for path in (first.archive, first.sha256_file, first.release)
    }

    with pytest.raises(package_module.PackageError, match="exist"):
        package_module.build_handoff(source_root=source, out_dir=out)

    assert {path: path.read_bytes() for path in originals} == originals


def test_packager_publishes_no_partial_set_when_one_final_path_exists(
    tmp_path,
    package_module,
):
    source = _minimal_repo(tmp_path)
    template = package_module.build_handoff(
        source_root=source,
        out_dir=tmp_path / "template",
    )
    out = tmp_path / "out"
    out.mkdir()
    collision = out / template.sha256_file.name
    collision.write_bytes(b"operator-owned\n")

    with pytest.raises(package_module.PackageError, match="exist"):
        package_module.build_handoff(source_root=source, out_dir=out)

    assert collision.read_bytes() == b"operator-owned\n"
    assert not (out / template.archive.name).exists()
    assert not (out / template.release.name).exists()


def test_agent_start_forbids_launching_seed_one_through_four():
    text = (REPO_ROOT / "AGENT-START.md").read_text()

    assert "only seed 0" in text
    assert "seed 1\u20134" in text
    assert "contract violation" in text


def test_zip_has_closed_members_normalized_metadata_and_all_planned_directories(
    tmp_path,
    package_module,
):
    source = _minimal_repo(tmp_path)
    artifacts = package_module.build_handoff(
        source_root=source,
        out_dir=tmp_path / "out",
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
