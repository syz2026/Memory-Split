from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path, PurePosixPath

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "package_aws_p5_handoff.py"
START_GUIDE = REPO_ROOT / "AWS-P5-START.md"
PROVIDER = "aws-p5.48xlarge"
COHORT_ID = "memorysplit-confirmatory-v3-360m-n10-aws"
EXPECTED_CONFIGS = {
    f"configs/360m-v3/{arm}-s{seed}.yaml"
    for seed in range(10)
    for arm in ("dense", "split90")
}
EXPECTED_EXECUTABLES = {
    "cluster/aws/p5/bootstrap.sh",
    "cluster/aws/p5/interruption_checkpoint.py",
    "cluster/aws/p5/launch_seed_pair.py",
    "scripts/build_parallel_corpus.py",
    "scripts/package_aws_p5_handoff.py",
    "scripts/run_train.py",
}
CONFIG_KEYS = {
    "schema_version",
    "cohort_id",
    "run_id",
    "condition",
    "seed",
    "model",
    "ctx",
    "train_corpus",
    "sidecar_name",
    "out_dir",
    "micro_batch_size",
    "tokens_per_step",
    "max_steps",
    "total_tokens",
    "lr",
    "warmup_steps",
    "weight_decay",
    "compile",
    "device",
    "log_every",
    "eval_every",
    "snapshot_steps",
    "ckpt_minutes",
}
SNAPSHOT_STEPS = [1_358, 3_396, 6_791, 10_187, 13_582]


def _load_module():
    assert SCRIPT.is_file(), "AWS P5 packager has not been implemented"
    name = f"package_aws_p5_handoff_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def _cohort(
    *,
    aws_seeds: list[int] | None = None,
    illumina_seeds: list[int] | None = None,
) -> dict[str, object]:
    provider_seeds = {
        PROVIDER: aws_seeds if aws_seeds is not None else list(range(10)),
    }
    if illumina_seeds is not None:
        provider_seeds["illumina-usfc-prd"] = illumina_seeds
    return {
        "cohort_id": COHORT_ID,
        "model_parameters": 356_033_536,
        "optimizer_steps": 13_582,
        "provider_seeds": provider_seeds,
        "raw_target_tokens": 7_120_879_616,
        "schema_version": 3,
        "targets_per_update": 524_288,
    }


def _v2_cohort() -> dict[str, object]:
    return {
        "cohort_id": "memorysplit-confirmatory-v2-360m-n5",
        "model_parameters": 356_033_536,
        "optimizer_steps": 13_582,
        "provider_seeds": {
            PROVIDER: [1, 2, 3, 4],
            "illumina-usfc-prd": [0],
        },
        "raw_target_tokens": 7_120_879_616,
        "schema_version": 2,
        "targets_per_update": 524_288,
    }


def _config(seed: int, arm: str) -> dict[str, object]:
    assert arm in {"dense", "split90"}
    return {
        "schema_version": 2,
        "cohort_id": COHORT_ID,
        "run_id": f"memorysplit-v3-360m-s{seed}-{arm}",
        "condition": arm,
        "seed": seed,
        "model": "d360m",
        "ctx": 1024,
        "train_corpus": "dataset/receipt.json",
        "sidecar_name": (
            "dense_target_weights"
            if arm == "dense"
            else "split90_target_weights"
        ),
        "out_dir": f"runs/seed-{seed}/{arm}",
        "micro_batch_size": 8,
        "tokens_per_step": 524_288,
        "max_steps": 13_582,
        "total_tokens": 7_120_879_616,
        "lr": 0.001,
        "warmup_steps": 300,
        "weight_decay": 0.1,
        "compile": True,
        "device": "cuda",
        "log_every": 20,
        "eval_every": 250,
        "snapshot_steps": list(SNAPSHOT_STEPS),
        "ckpt_minutes": 30,
    }


def _v2_config(seed: int, arm: str) -> dict[str, object]:
    value = _config(seed, arm)
    value["cohort_id"] = "memorysplit-confirmatory-v2-360m-n5"
    value["run_id"] = f"memorysplit-v2-360m-s{seed}-{arm}"
    value["train_corpus"] = "dataset/corpus-receipt.json"
    return value


def _profile() -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile_id": "aws-p5.48xlarge-v3",
        "provider": PROVIDER,
        "instance_type": "p5.48xlarge",
        "purchase_model": "on_demand",
        "gpu": {
            "model": "NVIDIA H100 80GB",
            "allocated": 8,
            "seed_train_groups": [4, 4],
        },
        "cpu": {"vcpus": 192, "memory_gib": 2048},
        "storage": {
            "instance_store": {
                "devices": 8,
                "device_bytes": 3_840_000_000_000,
                "model": "Amazon EC2 NVMe Instance Storage",
                "raid_level": "0",
            },
            "scratch_root": "/mnt/memorysplit",
            "durable_uri_env": "MS_S3_ROOT",
        },
        "runtime": {
            "ami_id_env": "MS_AWS_AMI_ID",
            "container_digest_env": "MS_CONTAINER_DIGEST",
            "region_env": "AWS_REGION",
            "runtime_gid_env": "MS_RUNTIME_GID",
            "runtime_uid_env": "MS_RUNTIME_UID",
        },
        "process_env_allowlist": ["AWS_REGION", "LANG", "LC_ALL"],
        "assigned_seeds": list(range(10)),
    }


def _v2_profile() -> dict[str, object]:
    value = _profile()
    value["profile_id"] = PROVIDER
    value["assigned_seeds"] = [1, 2, 3, 4]
    return value


def _dataset_pointer() -> dict[str, object]:
    return {
        "schema_version": 1,
        "provider": PROVIDER,
        "dataset_id": "memorysplit-v2-20x-reasoning-max-cohort",
        "durable_uri_env": "MS_S3_ROOT",
        "materialization": "s3",
        "relative_path": "dataset",
        "required_receipt": "dataset/receipt.json",
        "required_sidecars": [
            "dense_target_weights",
            "split90_target_weights",
        ],
        "scratch_root": "/mnt/memorysplit",
        "source_lock_manifest": "configs/reasoning-dataset-v2.json",
        "full_corpus_in_release": False,
    }


def _runtime_environment_contract(profile_sha256: str) -> dict[str, object]:
    return {
        "mode": "runtime_attested",
        "profile_sha256": profile_sha256,
        "container_image_digest_env": "MS_CONTAINER_DIGEST",
        "container_image_digest_pattern": "^sha256:[0-9a-f]{64}$",
        "runtime_environment_receipt": {
            "required_at_launch": True,
            "authentication": "aws_instance_identity_document_pkcs7",
            "required_fields": [
                "schema_version",
                "profile_sha256",
                "container_image_digest",
                "aws_instance_identity_document",
                "aws_instance_identity_pkcs7",
            ],
        },
    }


def _minimal_repo(
    tmp_path: Path,
    *,
    name: str = "source",
    object_format: str | None = None,
) -> Path:
    root = tmp_path / name
    root.mkdir()
    files: dict[str, bytes | str] = {
        "AWS-P5-START.md": (
            "# Fixture P5 start\nAWS-only v3 seeds 0-9.\n"
        ),
        "docs/AWS-P5-360M-RUNBOOK.md": "# Fixture AWS P5 runbook\n",
        "DATASET-POINTER-AWS.json": _canonical_json(_dataset_pointer()),
        "requirements.txt": "PyYAML>=6.0\npytest>=8.0\n",
        "pytest.ini": "[pytest]\n",
        "configs/cohort-assignment-v3.json": _canonical_json(_cohort()),
        "configs/preregistration-v3.yaml": (
            "schema_version: 3\n"
            "preregistration_id: memorysplit-confirmatory-v3\n"
        ),
        "configs/cohort-assignment-v2.json": _canonical_json(_v2_cohort()),
        "configs/preregistration-v2.yaml": (
            "schema_version: 2\n"
            "preregistration_id: memorysplit-confirmatory-v2\n"
        ),
        "configs/current-dataset-lock.json": '{"schema_version":1}\n',
        "configs/reasoning-dataset-v2.json": '{"schema_version":2}\n',
        "configs/route-policy.json": '{"schema_version":1}\n',
        "cluster/profiles/aws-p5.48xlarge-v3.json": _canonical_json(_profile()),
        "cluster/profiles/aws-p5.48xlarge.json": _canonical_json(_v2_profile()),
        "cluster/aws/p5/bootstrap.sh": "#!/bin/sh\nset -eu\n",
        "cluster/aws/p5/launch_seed_pair.py": (
            "#!/usr/bin/env python3\nraise SystemExit(0)\n"
        ),
        "cluster/aws/p5/interruption_checkpoint.py": (
            "#!/usr/bin/env python3\nraise SystemExit(0)\n"
        ),
        # A shell library intentionally tracked as non-executable. Packaging
        # must preserve Git mode instead of inferring mode from its suffix.
        "cluster/aws/p5/library.sh": "p5_fixture_library() { :; }\n",
        "msctl/__init__.py": '"""fixture contracts"""\n',
        "msctl/__main__.py": "raise SystemExit(0)\n",
        "msctl/aws_p5.py": "PROVIDER = 'aws-p5.48xlarge'\n",
        "msctl/aws_contracts.py": (
            REPO_ROOT / "msctl" / "aws_contracts.py"
        ).read_bytes(),
        "msctl/dataset.py": "DATASET_CONTRACT = 'receipt-v2'\n",
        "corpusgen/__init__.py": "",
        "corpusgen/parallel/__init__.py": "",
        "corpusgen/parallel/publication.py": "FORMAT = 2\n",
        "train/data.py": "PARALLEL_SIDECAR_V2_CONTRACT = {}\n",
        "evals/__init__.py": "",
        "evals/confirmatory/__init__.py": "",
        "evals/confirmatory/contracts.py": "SCHEMA_VERSION = 2\n",
        "organizer/__init__.py": "",
        "train/__init__.py": "",
        "train/model.py": "MODEL_PARAMETERS = 356_033_536\n",
        "scripts/build_parallel_corpus.py": (
            "#!/usr/bin/env python3\nraise SystemExit(0)\n"
        ),
        "scripts/run_train.py": (
            "#!/usr/bin/env python3\nraise SystemExit(0)\n"
        ),
        "scripts/package_aws_p5_handoff.py": SCRIPT.read_bytes(),
        "tests/test_package_aws_p5_handoff.py": Path(__file__).read_bytes(),
        "tests/test_aws_p5_profile.py": "def test_fixture(): assert True\n",
        "tests/test_aws_p5_launcher.py": "def test_fixture(): assert True\n",
        "tests/test_msctl.py": "def test_fixture(): assert True\n",
        "tests/fixtures/current_sources/README.md": "test fixture only\n",
        "tests/fixtures/relational-smoke-route-policy.json": (
            '{"schema_version":1}\n'
        ),
        "sources/current-dataset-licenses.json": '{"license":"fixture"}\n',
        "sources/wikidata5m.lock.json": '{"sha256":"' + "b" * 64 + '"}\n',
        "vendor/tiktoken/6c7ea1a7e38e3a7f062df639a5b80947f075ffe6": (
            b"tokenizer fixture"
        ),
        # Provider-specific and materialized paths are deliberately tracked
        # so the fixture proves they are excluded rather than merely absent.
        "AGENT-START.md": "# Illumina guide\n",
        "DATASET-POINTER.json": '{"provider":"illumina-usfc-prd"}\n',
        "cluster/profiles/illumina-usfc-prd.json": (
            '{"provider":"illumina-usfc-prd"}\n'
        ),
        "cluster/slurm/v2_seed0.sbatch": "#!/bin/sh\n",
        "scripts/package_illumina_handoff.py": "raise SystemExit(0)\n",
        "tests/test_package_illumina_handoff.py": (
            "def test_fixture(): assert True\n"
        ),
        "configs/360m/legacy.yaml": "seed: 0\n",
        "data/full-corpus.bin": b"excluded corpus",
        "fixtures/current-smoke/train.bin": b"excluded smoke data",
        "artifacts/seed0.zip": b"excluded artifact",
        "docs/history.md": "excluded history\n",
        "outputs/seed-0/checkpoint.pt": b"excluded output",
        "logs/train.log": "excluded log\n",
        "checkpoints/model.pt": b"excluded checkpoint",
        "sealed/gold.json": '{"answer":"excluded"}\n',
        ".cache/compiler.bin": b"excluded cache",
    }
    for seed in range(10):
        for arm in ("dense", "split90"):
            relative = f"configs/360m-v3/{arm}-s{seed}.yaml"
            files[relative] = yaml.safe_dump(
                _config(seed, arm),
                sort_keys=False,
                allow_unicode=False,
            )
    for seed in range(5):
        for arm in ("dense", "split90"):
            relative = f"configs/360m-v2/{arm}-s{seed}.yaml"
            files[relative] = yaml.safe_dump(
                _v2_config(seed, arm),
                sort_keys=False,
                allow_unicode=False,
            )
    for relative, data in files.items():
        _write(
            root / relative,
            data,
            executable=relative in EXPECTED_EXECUTABLES
            or relative == "cluster/slurm/v2_seed0.sbatch",
        )
    init_arguments = ["init", "-q"]
    if object_format is not None:
        init_arguments.append(f"--object-format={object_format}")
    _git(root, *init_arguments)
    _git(root, "config", "user.name", "Package Test")
    _git(root, "config", "user.email", "package@example.invalid")
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "add", "-f", "--all")
    _git(root, "commit", "-qm", "complete AWS P5 fixture")
    return root


def _commit(root: Path, message: str) -> None:
    _git(root, "add", "-f", "--all")
    _git(root, "commit", "-qm", message)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _build(module, source: Path, out: Path):
    return module.build_handoff(
        source_root=source,
        out_dir=out,
        apply=True,
    )


def test_double_build_is_byte_identical_and_emits_external_receipts(tmp_path):
    module = _load_module()
    source = _minimal_repo(tmp_path)

    first = _build(module, source, tmp_path / "out-a")
    second = _build(module, source, tmp_path / "out-b")

    assert first.published is True
    assert first.archive.name == second.archive.name
    assert first.archive.name.startswith("ms-aws-p5-r1-")
    assert first.archive.suffix == ".zip"
    assert first.archive.read_bytes() == second.archive.read_bytes()
    assert first.sha256_file.read_bytes() == second.sha256_file.read_bytes()
    assert first.release.name == "RELEASE-AWS-P5.json"
    assert first.release.read_bytes() == second.release.read_bytes()

    archive_digest = _sha256_path(first.archive)
    assert first.sha256_file.read_text() == (
        f"{archive_digest}  {first.archive.name}\n"
    )
    receipt = json.loads(first.release.read_text())
    assert set(receipt) == {
        "archive",
        "cohort_assignment",
        "cohort_assignment_sha256",
        "config_sha256",
        "dataset_pointer",
        "dataset_pointer_sha256",
        "environment",
        "members_sha256",
        "package_format_version",
        "profile",
        "profile_sha256",
        "provider",
        "release_id",
        "schema_version",
        "seed_assignment",
        "source",
    }
    assert receipt["schema_version"] == 1
    assert receipt["package_format_version"] == 2
    assert type(receipt["package_format_version"]) is int
    assert receipt["provider"] == PROVIDER
    assert receipt["archive"] == {
        "path": first.archive.name,
        "sha256": archive_digest,
        "bytes": first.archive.stat().st_size,
    }
    assert receipt["source"] == {
        "commit": _git(source, "rev-parse", "HEAD"),
        "dirty": False,
        "tree": _git(source, "rev-parse", "HEAD^{tree}"),
    }
    assert set(receipt["source"]) == {"commit", "dirty", "tree"}
    assert re.fullmatch(r"[0-9a-f]{40}", receipt["source"]["commit"])
    assert re.fullmatch(r"[0-9a-f]{40}", receipt["source"]["tree"])
    assert receipt["seed_assignment"] == {
        "cohort_id": COHORT_ID,
        "provider": PROVIDER,
        "seeds": list(range(10)),
        "arms": ["dense", "split90"],
    }
    serialized = json.dumps(receipt, sort_keys=True)
    assert "dataset_receipt_sha256" not in serialized
    assert "dataset_build_id" not in serialized
    assert "ordered_stream_sha256" not in serialized


def test_packager_rejects_non_sha1_source_object_ids(tmp_path):
    module = _load_module()
    source = _minimal_repo(tmp_path, object_format="sha256")

    with pytest.raises(module.PackageError, match="40-character|commit|tree"):
        module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
            apply=True,
        )


def test_archive_contains_only_semantic_seed_pairs_and_hash_bound_metadata(
    tmp_path,
):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    artifacts = _build(module, source, tmp_path / "out")

    with zipfile.ZipFile(artifacts.archive) as archive:
        names = set(archive.namelist())
        config_names = {
            name
            for name in names
            if name.startswith("configs/360m-v3/") and not name.endswith("/")
        }
        assert config_names == EXPECTED_CONFIGS
        assert not any(name.startswith("configs/360m-v2/") for name in names)
        assert "configs/cohort-assignment-v2.json" not in names
        assert "configs/preregistration-v2.yaml" not in names
        assert "cluster/profiles/aws-p5.48xlarge.json" not in names
        assert "configs/preregistration-v3.yaml" in names
        assert "cluster/profiles/aws-p5.48xlarge-v3.json" in names
        assert "msctl/aws_contracts.py" in names
        assert "tests/fixtures/current_sources/README.md" in names
        assert "tests/fixtures/relational-smoke-route-policy.json" in names

        assignment_bytes = archive.read("configs/cohort-assignment-v3.json")
        assignment = json.loads(assignment_bytes)
        assert assignment == _cohort()
        for name in sorted(EXPECTED_CONFIGS):
            arm, raw_seed = PurePosixPath(name).stem.rsplit("-s", 1)
            config = yaml.safe_load(archive.read(name))
            assert set(config) == CONFIG_KEYS
            assert config == _config(int(raw_seed), arm)

        metadata = json.loads(archive.read("RELEASE-METADATA.json"))
        assert set(metadata) == {
            "cohort_assignment",
            "config_sha256",
            "dataset_pointer",
            "environment",
            "members",
            "package_format_version",
            "profile",
            "provider",
            "schema_version",
            "seed_assignment",
            "source",
        }
        assert metadata["schema_version"] == 1
        assert metadata["package_format_version"] == 2
        assert type(metadata["package_format_version"]) is int
        assert metadata["provider"] == PROVIDER
        assert metadata["source"] == {
            "commit": _git(source, "rev-parse", "HEAD"),
            "dirty": False,
            "tree": _git(source, "rev-parse", "HEAD^{tree}"),
        }
        assert set(metadata["source"]) == {"commit", "dirty", "tree"}
        assert re.fullmatch(r"[0-9a-f]{40}", metadata["source"]["commit"])
        assert re.fullmatch(r"[0-9a-f]{40}", metadata["source"]["tree"])
        assert metadata["seed_assignment"] == {
            "cohort_id": COHORT_ID,
            "provider": PROVIDER,
            "seeds": list(range(10)),
            "arms": ["dense", "split90"],
        }
        assert metadata["cohort_assignment"] == {
            "path": "configs/cohort-assignment-v3.json",
            "sha256": _sha256_bytes(assignment_bytes),
        }
        assert metadata["profile"] == {
            "path": "cluster/profiles/aws-p5.48xlarge-v3.json",
            "sha256": _sha256_bytes(
                archive.read("cluster/profiles/aws-p5.48xlarge-v3.json")
            ),
        }
        environment = _runtime_environment_contract(
            metadata["profile"]["sha256"]
        )
        assert metadata["environment"] == environment
        assert "requirements-aws-p5.lock" not in names
        assert "environment_sha256" not in metadata
        assert "credential" not in json.dumps(environment).lower()
        assert metadata["config_sha256"] == {
            name: _sha256_bytes(archive.read(name))
            for name in sorted(EXPECTED_CONFIGS)
        }
        serialized = json.dumps(metadata, sort_keys=True)
        assert "dataset_receipt_sha256" not in serialized
        assert "dataset_build_id" not in serialized
        assert "ordered_stream_sha256" not in serialized

        member_rows = metadata["members"]
        member_paths = {row["path"] for row in member_rows}
        regular_names = {
            info.filename for info in archive.infolist() if not info.is_dir()
        }
        assert member_paths == regular_names - {
            "RELEASE-METADATA.json",
            "SHA256SUMS",
        }
        for row in member_rows:
            assert row["sha256"] == _sha256_bytes(archive.read(row["path"]))
            assert row["bytes"] == len(archive.read(row["path"]))
            assert row["git_mode"] in {"100644", "100755"}
            assert len(row["git_blob"]) in {40, 64}

        checksum_paths: set[str] = set()
        for line in archive.read("SHA256SUMS").decode("ascii").splitlines():
            digest, relative = line.split("  ", 1)
            assert digest == _sha256_bytes(archive.read(relative))
            checksum_paths.add(relative)
        assert checksum_paths == regular_names - {"SHA256SUMS"}
        members_sha256 = _sha256_bytes(archive.read("SHA256SUMS"))

    receipt = json.loads(artifacts.release.read_text())
    assert receipt["cohort_assignment_sha256"] == metadata[
        "cohort_assignment"
    ]["sha256"]
    assert receipt["profile_sha256"] == metadata["profile"]["sha256"]
    assert "environment_sha256" not in receipt
    assert receipt["dataset_pointer_sha256"] == metadata["dataset_pointer"][
        "sha256"
    ]
    assert receipt["config_sha256"] == metadata["config_sha256"]
    assert receipt["members_sha256"] == members_sha256
    assert receipt["cohort_assignment"] == metadata["cohort_assignment"]
    assert receipt["profile"] == metadata["profile"]
    assert receipt["environment"] == metadata["environment"]
    assert receipt["dataset_pointer"] == metadata["dataset_pointer"]


def test_packager_rejects_claimed_static_aws_environment_identity(tmp_path):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    _write(
        source / "requirements-aws-p5.lock",
        _canonical_json(
            {
                "schema_version": 1,
                "profile_sha256": "0" * 64,
                "container_image_digest": "sha256:" + "1" * 64,
                "runtime_environment_receipt_sha256": "2" * 64,
            }
        ),
    )
    _commit(source, "add fabricated static AWS environment identity")

    with pytest.raises(module.PackageError, match="environment|lock|forbidden"):
        module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
            apply=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ami_id", "ami-fabricated"),
        ("container_image_digest", "sha256:" + "1" * 64),
        (
            "runtime_environment_receipt",
            {"profile_sha256": "2" * 64},
        ),
    ],
)
def test_packager_rejects_fabricated_profile_runtime_identity(
    tmp_path,
    field,
    value,
):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    path = source / "cluster/profiles/aws-p5.48xlarge-v3.json"
    profile = json.loads(path.read_text())
    profile["runtime"][field] = value
    path.write_text(_canonical_json(profile))
    _commit(source, f"add fabricated runtime identity: {field}")

    with pytest.raises(module.PackageError, match="runtime|environment|identity"):
        module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
            apply=True,
        )


def test_packager_rejects_wrong_v3_profile_seeds(tmp_path):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    path = source / "cluster/profiles/aws-p5.48xlarge-v3.json"
    profile = json.loads(path.read_text())
    profile["assigned_seeds"] = list(range(9))
    path.write_text(_canonical_json(profile))
    _commit(source, "remove one assigned v3 seed")

    with pytest.raises(module.PackageError, match="assigned_seeds|seed|profile"):
        module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
            apply=True,
        )


def test_packager_rejects_wrong_dataset_receipt_pointer(tmp_path):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    path = source / "DATASET-POINTER-AWS.json"
    pointer = json.loads(path.read_text())
    pointer["required_receipt"] = "dataset/corpus-receipt.json"
    path.write_text(_canonical_json(pointer))
    _commit(source, "change canonical dataset receipt pointer")

    with pytest.raises(module.PackageError, match="pointer|required_receipt|receipt"):
        module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
            apply=True,
        )


@pytest.mark.parametrize(
    ("substitution", "source_relative", "target_relative"),
    [
        (
            "assignment",
            "configs/cohort-assignment-v2.json",
            "configs/cohort-assignment-v3.json",
        ),
        (
            "preregistration",
            "configs/preregistration-v2.yaml",
            "configs/preregistration-v3.yaml",
        ),
        (
            "run-config",
            "configs/360m-v2/dense-s0.yaml",
            "configs/360m-v3/dense-s0.yaml",
        ),
        (
            "profile",
            "cluster/profiles/aws-p5.48xlarge.json",
            "cluster/profiles/aws-p5.48xlarge-v3.json",
        ),
    ],
)
def test_packager_rejects_cross_version_substitution(
    tmp_path,
    substitution,
    source_relative,
    target_relative,
):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    (source / target_relative).write_bytes((source / source_relative).read_bytes())
    _commit(source, f"substitute v2 {substitution}")

    with pytest.raises(
        module.PackageError,
        match="assignment|preregistration|config|profile|contract|v3",
    ):
        module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
            apply=True,
        )


def test_archive_excludes_materialized_provider_and_sealed_content(tmp_path):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    artifacts = _build(module, source, tmp_path / "out")

    with zipfile.ZipFile(artifacts.archive) as archive:
        names = set(archive.namelist())

    forbidden_roots = {
        ".cache",
        "artifacts",
        "checkpoints",
        "data",
        "fixtures",
        "logs",
        "outputs",
        "sealed",
    }
    assert not any(
        PurePosixPath(name).parts
        and PurePosixPath(name).parts[0] in forbidden_roots
        for name in names
    )
    assert {
        name for name in names if PurePosixPath(name).parts[:1] == ("docs",)
    } == {"docs/", "docs/AWS-P5-360M-RUNBOOK.md"}
    assert "AGENT-START.md" not in names
    assert "DATASET-POINTER.json" not in names
    assert "cluster/profiles/illumina-usfc-prd.json" not in names
    assert "cluster/profiles/aws-p5.48xlarge.json" not in names
    assert "cluster/slurm/v2_seed0.sbatch" not in names
    assert "scripts/package_illumina_handoff.py" not in names
    assert "tests/test_package_illumina_handoff.py" not in names
    assert not any(name.startswith("configs/360m/") for name in names)
    assert not any(name.startswith("configs/360m-v2/") for name in names)


def test_zip_metadata_is_normalized_and_preserves_git_executable_modes(tmp_path):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    artifacts = _build(module, source, tmp_path / "out")

    with zipfile.ZipFile(artifacts.archive) as archive:
        infos = archive.infolist()
        assert len({info.filename for info in infos}) == len(infos)
        for info in infos:
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.create_system == 3
            assert info.flag_bits == 0
            assert not info.filename.startswith("/")
            assert "\\" not in info.filename
            assert ".." not in PurePosixPath(info.filename).parts
            raw_mode = info.external_attr >> 16
            assert stat.S_IFMT(raw_mode) in {stat.S_IFDIR, stat.S_IFREG}
            expected_mode = (
                0o755
                if info.is_dir() or info.filename in EXPECTED_EXECUTABLES
                else 0o644
            )
            assert stat.S_IMODE(raw_mode) == expected_mode
        assert (
            stat.S_IMODE(
                archive.getinfo("cluster/aws/p5/library.sh").external_attr >> 16
            )
            == 0o644
        )


def test_packager_rejects_dirty_tree_before_writing(tmp_path):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    (source / "untracked.txt").write_text("dirty\n")
    out = tmp_path / "out"

    with pytest.raises(module.PackageError, match="dirty"):
        module.build_handoff(source_root=source, out_dir=out, apply=True)

    assert not out.exists()


def test_inherited_git_controls_cannot_redirect_a_dirty_source(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    source = _minimal_repo(tmp_path, name="dirty-source")
    alternate = _minimal_repo(tmp_path, name="alternate-clean-source")
    (source / "untracked.txt").write_text("dirty source must remain authoritative\n")
    out = tmp_path / "out"
    injected = {
        "GIT_DIR": str(alternate / ".git"),
        "GIT_WORK_TREE": str(alternate),
        "GIT_INDEX_FILE": str(alternate / ".git" / "index"),
        "GIT_OBJECT_DIRECTORY": str(alternate / ".git" / "objects"),
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.worktree",
        "GIT_CONFIG_VALUE_0": str(alternate),
    }
    for key, value in injected.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(module.PackageError, match="dirty"):
        module.build_handoff(source_root=source, out_dir=out, apply=True)

    assert not out.exists()


def test_snapshot_packages_requested_commit_after_source_path_substitution(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    source = _minimal_repo(tmp_path, name="source")
    alternate = _minimal_repo(tmp_path, name="alternate")
    _write(
        alternate / "AWS-P5-START.md",
        "alternate repository bytes must never be packaged\n",
    )
    _commit(alternate, "change alternate package member")
    expected_member = subprocess.check_output(
        ["git", "-C", str(source), "show", "HEAD:AWS-P5-START.md"]
    )
    expected_commit = _git(source, "rev-parse", "HEAD")
    expected_tree = _git(source, "rev-parse", "HEAD^{tree}")
    original_snapshot = module._read_git_blobs
    captured_source = tmp_path / "captured-source"
    substituted = False

    def substituting_snapshot(*args, **kwargs):
        nonlocal substituted
        snapshot = original_snapshot(*args, **kwargs)
        os.rename(source, captured_source)
        os.rename(alternate, source)
        substituted = True
        return snapshot

    monkeypatch.setattr(module, "_read_git_blobs", substituting_snapshot)

    artifacts = module.build_handoff(
        source_root=source,
        out_dir=tmp_path / "out",
        apply=True,
    )

    assert substituted is True
    with zipfile.ZipFile(artifacts.archive) as archive:
        assert archive.read("AWS-P5-START.md") == expected_member
        metadata = json.loads(archive.read("RELEASE-METADATA.json"))
    assert metadata["source"] == {
        "commit": expected_commit,
        "dirty": False,
        "tree": expected_tree,
    }


def test_snapshot_loads_release_members_in_one_batched_object_read(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    original_run = subprocess.run
    cat_file_commands: list[tuple[str, ...]] = []

    def recording_run(command, *args, **kwargs):
        rendered = tuple(os.fspath(part) for part in command)
        if "cat-file" in rendered:
            cat_file_commands.append(rendered)
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", recording_run)

    _build(module, source, tmp_path / "out")

    assert len(cat_file_commands) == 1
    assert cat_file_commands[0][-2:] == ("cat-file", "--batch")


def test_packager_rejects_any_tracked_symlink(tmp_path):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    os.symlink("history.md", source / "docs" / "linked-history.md")
    _commit(source, "add excluded-path symlink")

    with pytest.raises(module.PackageError, match="symlink"):
        module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
            apply=True,
        )


def test_packager_rejects_secret_without_echoing_it(tmp_path):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    secret = "this-is-a-real-looking-secret-value"
    secret_name = "AWS_SECRET" + "_ACCESS_KEY"
    (source / "msctl" / "aws_p5.py").write_text(
        f'{secret_name} = "{secret}"\n'
    )
    _commit(source, "add secret")

    with pytest.raises(module.PackageError) as caught:
        module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
            apply=True,
        )

    assert "secret" in str(caught.value).lower()
    assert secret not in str(caught.value)


def test_packager_rejects_static_credential_fields_in_profile(tmp_path):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    profile = _profile()
    profile["aws_access_key_id"] = "static-credential-must-not-ship"
    (source / "cluster/profiles/aws-p5.48xlarge-v3.json").write_text(
        _canonical_json(profile)
    )
    _commit(source, "add static profile credential")

    with pytest.raises(module.PackageError, match="credential|secret"):
        module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
            apply=True,
        )


@pytest.mark.parametrize(
    "relative",
    [
        "vendor/tiktoken/data/payload.bin",
        "sources/archive/outputs/result.json",
        "tests/helpers/checkpoints/state.py",
        "msctl/runtime/logs/helper.py",
        "evals/confirmatory/sealed/gold.py",
        "train/private/credentials/key.py",
        "tests/helpers/sealed_gold_v2/answers.py",
        "msctl/private/sealedGold/answers.py",
        "evals/private/da-ta/corpus.py",
        "train/private/out-puts/result.py",
        "scripts/private/lo-gs/worker.py",
        "tests/private/check_points/state.py",
        "msctl/private/creden-tials/key.py",
        "evals/private/result/answers.py",
        "evals/private/results/answers.py",
        "evals/private/RESULTS/answers.py",
        "evals/private/re-sults/answers.py",
        "evals/private/re_sult/answers.py",
        "evals/private/Re-SuLtS_v2/answers.py",
    ],
)
def test_packager_rejects_forbidden_content_nested_in_allowed_trees(
    tmp_path,
    relative,
):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    _write(source / relative, b"forbidden nested content\n")
    _commit(source, "add nested forbidden content")

    with pytest.raises(module.PackageError, match="forbidden"):
        module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
            apply=True,
        )


@pytest.mark.parametrize(
    "relative",
    [
        "vendor/tiktoken/unapproved-asset",
        "sources/unapproved-source.json",
    ],
)
def test_packager_rejects_unenumerated_vendor_and_source_paths(
    tmp_path,
    relative,
):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    _write(source / relative, b'{"fixture":true}\n')
    _commit(source, "add unapproved source path")

    with pytest.raises(module.PackageError, match="unknown|allowlist"):
        module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
            apply=True,
        )


@pytest.mark.parametrize(
    "relative",
    [
        "configs/current-dataset-lock.json",
        "configs/preregistration-v3.yaml",
    ],
)
def test_packager_recursively_rejects_secret_keys_without_echoing_values(
    tmp_path,
    relative,
):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    secret_key = "pass" + "word"
    secret_value = "nested-json-or-yaml-secret-must-not-echo"
    if relative.endswith(".json"):
        content = _canonical_json(
            {
                "schema_version": 1,
                "nested": {secret_key: secret_value},
            }
        )
    else:
        content = (
            "schema_version: 2\n"
            "nested:\n"
            f"  {secret_key}: {secret_value}\n"
        )
    (source / relative).write_text(content)
    _commit(source, "add nested structured secret")

    with pytest.raises(module.PackageError) as caught:
        module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
            apply=True,
        )

    assert "secret" in str(caught.value).lower()
    assert secret_value not in str(caught.value)


@pytest.mark.parametrize(
    ("relative", "sensitive_key"),
    [
        ("configs/current-dataset-lock.json", "client-secret"),
        ("configs/preregistration-v3.yaml", "clientSecret"),
        ("configs/current-dataset-lock.json", "apiKey"),
        ("configs/preregistration-v3.yaml", "api-key"),
        ("configs/current-dataset-lock.json", "accessKey"),
        ("configs/preregistration-v3.yaml", "privateKey"),
        ("configs/current-dataset-lock.json", "password"),
        ("configs/preregistration-v3.yaml", "passphrase"),
        ("configs/current-dataset-lock.json", "token"),
        ("configs/preregistration-v3.yaml", "credentials"),
        ("configs/current-dataset-lock.json", "secretValue"),
        ("configs/preregistration-v3.yaml", "secret_value"),
        ("configs/current-dataset-lock.json", "tokenValue"),
        ("configs/preregistration-v3.yaml", "token_value"),
        ("configs/current-dataset-lock.json", "apiToken"),
        ("configs/preregistration-v3.yaml", "api_token_value"),
    ],
)
def test_structured_credential_key_variants_are_rejected_without_values(
    tmp_path,
    relative,
    sensitive_key,
):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    credential_value = "structured-value-must-never-be-echoed"
    if relative.endswith(".json"):
        content = _canonical_json(
            {
                "schema_version": 1,
                "nested": {sensitive_key: credential_value},
            }
        )
    else:
        content = (
            "schema_version: 2\n"
            "nested:\n"
            f"  {sensitive_key}: {credential_value}\n"
        )
    (source / relative).write_text(content)
    _commit(source, "add structured credential key variant")

    with pytest.raises(module.PackageError) as caught:
        module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
            apply=True,
        )

    assert "secret" in str(caught.value).lower()
    assert credential_value not in str(caught.value)


def test_structured_scanner_allows_benign_token_measurement_keys(tmp_path):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    benign = {
        "schema_version": 1,
        "token_count": 1024,
        "tokenizer_name": "fixture-tokenizer",
        "tokenization_method": "fixture",
        "token_type_count": 2,
        "api_token_count": 0,
        "secretary_name": "fixture-role",
        "tokens_per_step": 524_288,
        "raw_target_tokens": 7_120_879_616,
    }
    (source / "configs/current-dataset-lock.json").write_text(
        _canonical_json(benign)
    )
    _commit(source, "add benign token measurement keys")

    artifacts = _build(module, source, tmp_path / "out")

    with zipfile.ZipFile(artifacts.archive) as archive:
        assert json.loads(
            archive.read("configs/current-dataset-lock.json")
        ) == benign


def test_packager_rejects_unknown_tracked_path(tmp_path):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    _write(source / "mystery" / "tool.py", "print('unknown')\n")
    _commit(source, "add unknown path")

    with pytest.raises(module.PackageError, match="unknown"):
        module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
            apply=True,
        )


def test_packager_rejects_illumina_reintroduction_in_v3_assignment(tmp_path):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    assignment = _cohort(illumina_seeds=[0])
    (source / "configs/cohort-assignment-v3.json").write_text(
        _canonical_json(assignment)
    )
    _commit(source, "reintroduce Illumina provider")

    with pytest.raises(module.PackageError, match="provider|assignment|Illumina"):
        module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
            apply=True,
        )


@pytest.mark.parametrize(
    ("relative", "field", "value", "message"),
    [
        (
            "configs/360m-v3/dense-s0.yaml",
            "seed",
            10,
            "seed",
        ),
        (
            "configs/360m-v3/split90-s2.yaml",
            "condition",
            "split",
            "condition|split90",
        ),
        (
            "configs/360m-v3/dense-s3.yaml",
            "max_steps",
            13_581,
            "max_steps|token",
        ),
        (
            "configs/360m-v3/split90-s9.yaml",
            "sidecar_name",
            "dense_target_weights",
            "sidecar",
        ),
        (
            "configs/360m-v3/dense-s4.yaml",
            "train_corpus",
            "dataset/corpus-receipt.json",
            "train_corpus|dataset",
        ),
    ],
)
def test_packager_rejects_semantically_invalid_provider_config(
    tmp_path,
    relative,
    field,
    value,
    message,
):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    path = source / relative
    config = yaml.safe_load(path.read_text())
    config[field] = value
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    _commit(source, f"break {field}")

    with pytest.raises(module.PackageError, match=message):
        module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
            apply=True,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("bool", "snapshot_steps"),
        ("float", "snapshot_steps"),
        ("missing-field", "snapshot_steps"),
        ("missing-element", "snapshot_steps"),
        ("extra-element", "snapshot_steps"),
        ("reordered", "snapshot_steps"),
        ("duplicate", "snapshot_steps"),
        ("drift", "snapshot_steps"),
        ("final-drift", "snapshot_steps"),
        ("snap-frac", "snap_frac"),
    ],
)
def test_packager_rejects_invalid_snapshot_schedule(
    tmp_path,
    case,
    message,
):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    path = source / "configs/360m-v3/dense-s0.yaml"
    config = yaml.safe_load(path.read_text())
    if case == "bool":
        config["snapshot_steps"][2] = True
    elif case == "float":
        config["snapshot_steps"][2] = 6_791.0
    elif case == "missing-field":
        del config["snapshot_steps"]
    elif case == "missing-element":
        config["snapshot_steps"] = [1_358, 3_396, 6_791, 13_582]
    elif case == "extra-element":
        config["snapshot_steps"] = [
            1_358,
            3_396,
            6_791,
            9_000,
            10_187,
            13_582,
        ]
    elif case == "reordered":
        config["snapshot_steps"] = [3_396, 1_358, 6_791, 10_187, 13_582]
    elif case == "duplicate":
        config["snapshot_steps"] = [1_358, 3_396, 6_791, 6_791, 13_582]
    elif case == "drift":
        config["snapshot_steps"][3] = 10_188
    elif case == "final-drift":
        config["snapshot_steps"][-1] = 13_581
    elif case == "snap-frac":
        config["snap_frac"] = 0.1
    else:
        raise AssertionError(f"unknown test case: {case}")
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    _commit(source, f"break snapshot schedule: {case}")

    with pytest.raises(module.PackageError, match=message):
        module.build_handoff(
            source_root=source,
            out_dir=tmp_path / "out",
            apply=True,
        )


def test_packager_rejects_partial_pair_and_unassigned_provider_config(tmp_path):
    module = _load_module()
    partial_source = _minimal_repo(tmp_path, name="partial")
    _git(
        partial_source,
        "rm",
        "-q",
        "configs/360m-v3/split90-s9.yaml",
    )
    _git(partial_source, "commit", "-qm", "remove half of pair")

    with pytest.raises(module.PackageError, match="required|pair|split90"):
        module.build_handoff(
            source_root=partial_source,
            out_dir=tmp_path / "partial-out",
            apply=True,
        )

    extra_source = _minimal_repo(tmp_path, name="extra")
    _write(
        extra_source / "configs/360m-v3/dense-s10.yaml",
        yaml.safe_dump(_config(10, "dense"), sort_keys=False),
    )
    _commit(extra_source, "add unassigned config")

    with pytest.raises(module.PackageError, match="unknown|unassigned|seed"):
        module.build_handoff(
            source_root=extra_source,
            out_dir=tmp_path / "extra-out",
            apply=True,
        )


def test_apply_publishes_one_atomic_no_replace_release_set(tmp_path):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    out = tmp_path / "out"

    first = _build(module, source, out)
    before = {
        path.name: path.read_bytes()
        for path in first.release_dir.iterdir()
        if path.is_file()
    }

    with pytest.raises(module.PackageError) as caught:
        _build(module, source, out)

    assert caught.value.code == "RELEASE_EXISTS"
    assert {
        path.name: path.read_bytes()
        for path in first.release_dir.iterdir()
        if path.is_file()
    } == before
    assert set(before) == {
        first.archive.name,
        first.sha256_file.name,
        "RELEASE-AWS-P5.json",
    }
    assert [path for path in out.iterdir() if path.name.startswith(".")] == []


def test_descriptor_pinning_detects_archive_path_replacement(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    out = tmp_path / "out"
    original_open = module._open_pinned_regular_at
    raced = False

    def racing_open(directory_fd, name):
        nonlocal raced
        descriptor = original_open(directory_fd, name)
        if not raced and name.endswith(".zip"):
            replacement = ".replacement-archive"
            replacement_fd = os.open(
                replacement,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            try:
                os.write(replacement_fd, b"not the archive that was hashed")
                os.fsync(replacement_fd)
            finally:
                os.close(replacement_fd)
            os.rename(
                replacement,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            raced = True
        return descriptor

    monkeypatch.setattr(module, "_open_pinned_regular_at", racing_open)

    with pytest.raises(module.PackageError, match="changed|replaced|descriptor"):
        module.build_handoff(source_root=source, out_dir=out, apply=True)

    assert raced is True
    assert not out.exists() or list(out.iterdir()) == []


@pytest.mark.parametrize("replacement_kind", ["directory", "symlink"])
def test_staging_path_replacement_cannot_publish_a_release(
    tmp_path,
    monkeypatch,
    replacement_kind,
):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    out = tmp_path / "out"
    original_make = module._make_staging_at
    raced = False

    def racing_make(output_fd, release_id):
        nonlocal raced
        name, descriptor = original_make(output_fd, release_id)
        moved = f"{name}.moved"
        os.rename(
            name,
            moved,
            src_dir_fd=output_fd,
            dst_dir_fd=output_fd,
        )
        if replacement_kind == "directory":
            os.mkdir(name, 0o700, dir_fd=output_fd)
        else:
            os.symlink(moved, name, dir_fd=output_fd)
        raced = True
        return name, descriptor

    monkeypatch.setattr(module, "_make_staging_at", racing_make)

    with pytest.raises(
        module.PackageError,
        match="staging|symlink|replaced|descriptor",
    ):
        module.build_handoff(source_root=source, out_dir=out, apply=True)

    assert raced is True
    if out.exists():
        assert not any(
            not path.name.startswith(".") or path.name == "RELEASE-AWS-P5.json"
            for path in out.iterdir()
        )


def test_last_boundary_staging_replacement_is_quarantined(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    out = tmp_path / "out"
    original_rename = module._rename_noreplace_at
    raced = False

    def racing_rename(directory_fd, source_name, destination_name):
        nonlocal raced
        if not raced and source_name.endswith(".staging"):
            moved = f"{source_name}.last-boundary"
            os.rename(
                source_name,
                moved,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.mkdir(source_name, 0o700, dir_fd=directory_fd)
            raced = True
        return original_rename(directory_fd, source_name, destination_name)

    monkeypatch.setattr(module, "_rename_noreplace_at", racing_rename)

    with pytest.raises(
        module.PackageError,
        match="installed|staging|identity|replaced|descriptor",
    ):
        module.build_handoff(source_root=source, out_dir=out, apply=True)

    assert raced is True
    if out.exists():
        assert not any(not path.name.startswith(".") for path in out.iterdir())


def test_apply_rejects_group_or_world_writable_output_parent(tmp_path):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    out = tmp_path / "shared-output"
    out.mkdir(mode=0o700)
    out.chmod(0o777)

    with pytest.raises(module.PackageError, match="owner|writable|permission"):
        module.build_handoff(source_root=source, out_dir=out, apply=True)

    assert list(out.iterdir()) == []


def test_packager_rejects_symlink_output_directory(tmp_path):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    linked_output = tmp_path / "linked-output"
    os.symlink(real_output, linked_output)

    with pytest.raises(module.PackageError, match="symlink|output"):
        module.build_handoff(
            source_root=source,
            out_dir=linked_output,
            apply=True,
        )

    assert list(real_output.iterdir()) == []


def test_cli_defaults_to_dry_run_and_emits_one_json_object(tmp_path):
    _load_module()
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
    assert len(completed.stdout.splitlines()) == 1
    report = json.loads(completed.stdout)
    assert report["ok"] is True
    assert report["dry_run"] is True
    assert report["published"] is False
    assert report["provider"] == PROVIDER
    assert not out.exists()


def test_apply_requires_output_outside_source_worktree(tmp_path):
    module = _load_module()
    source = _minimal_repo(tmp_path)
    internal_out = source / "dist" / "aws-p5"

    with pytest.raises(module.PackageError, match="outside|external"):
        module.build_handoff(
            source_root=source,
            out_dir=internal_out,
            apply=True,
        )

    assert not internal_out.exists()
    assert _git(source, "status", "--porcelain") == ""


def test_cli_apply_uses_documented_external_output_by_default(tmp_path):
    _load_module()
    source = _minimal_repo(tmp_path)
    expected_output = tmp_path / "memorysplit-releases" / "aws-p5"
    command = [
        sys.executable,
        str(source / "scripts" / "package_aws_p5_handoff.py"),
        "--source-root",
        ".",
        "--apply",
    ]

    completed = subprocess.run(
        command,
        cwd=source,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert len(completed.stdout.splitlines()) == 1
    report = json.loads(completed.stdout)
    assert report["ok"] is True
    assert report["dry_run"] is False
    assert report["published"] is True
    release_dir = Path(report["release_dir"])
    assert release_dir.parent == expected_output
    assert Path(report["archive"]).is_file()
    assert Path(report["sha256_file"]).is_file()
    assert Path(report["release"]).name == "RELEASE-AWS-P5.json"
    assert _git(source, "status", "--porcelain") == ""


def test_operator_start_guide_states_fail_closed_seed_and_storage_contract():
    assert START_GUIDE.is_file(), "AWS P5 operator guide has not been implemented"
    text = START_GUIDE.read_text().lower()

    assert "seeds 1–4 only" in text or "seeds 1-4 only" in text
    assert "seed 0" in text and ("forbidden" in text or "reject" in text)
    assert "4+4" in text
    assert "instance role" in text
    assert "nvme" in text and "ephemeral" in text
    assert "s3" in text and "durable" in text
    assert "release-aws-p5.json" in text
    assert "sha-256" in text or "sha256" in text
    assert "sealed gold" in text and "outside" in text
    assert "../memorysplit-releases/aws-p5" in text
