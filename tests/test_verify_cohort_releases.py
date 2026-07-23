from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tarfile
import warnings
import zipfile
import uuid

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "verify_cohort_releases.py"
RUNBOOK = REPO_ROOT / "docs" / "AWS-P5-360M-RUNBOOK.md"

# Final interface refs. Update only after the owning task publishes a commit.
ILLUMINA_INTEGRATION_REF = "a85eb13ff2bf41046ab4a77d4f1924ed7e66c32e"
TASK_3_5_REF = "4b796a5"
TASK_6_REF = "0c9f20b"
TASK_7_REF = "48cfacb"
TASK_8_REF = "f22f6ad"

COHORT_ID = "memorysplit-confirmatory-v2-360m-n5"
ILLUMINA = "illumina-usfc-prd"
AWS = "aws-p5.48xlarge"
SOURCE_COMMIT = "a" * 40
ASSIGNMENT_PATH = "configs/cohort-assignment-v2.json"
CORPUS_IDENTITY_PATH = "configs/reasoning-dataset-v2.json"
EVALUATION_IDENTITY_PATH = "configs/preregistration-v2.yaml"
NORMALIZED_TIME = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class FixtureRelease:
    receipt: Path
    archive: Path
    sha256_file: Path
    release_id: str
    archive_sha256: str
    assignment_sha256: str
    corpus_sha256: str
    evaluation_sha256: str


def _canonical_pretty(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _assignment() -> dict[str, object]:
    return {
        "schema_version": 2,
        "cohort_id": COHORT_ID,
        "model_parameters": 356_033_536,
        "optimizer_steps": 13_582,
        "provider_seeds": {
            AWS: [1, 2, 3, 4],
            ILLUMINA: [0],
        },
        "raw_target_tokens": 7_120_879_616,
        "targets_per_update": 524_288,
    }


def _git_bytes(revision: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{revision}:{path}"],
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8",
        errors="replace",
    )
    return completed.stdout


def _aws_profile() -> dict[str, object]:
    return json.loads(
        _git_bytes(
            TASK_3_5_REF,
            "cluster/profiles/aws-p5.48xlarge.json",
        )
    )


def _run_config(
    seed: int,
    arm: str,
    *,
    run_id: str | None = None,
) -> bytes:
    value = {
        "schema_version": 2,
        "cohort_id": COHORT_ID,
        "run_id": run_id or f"memorysplit-v2-360m-s{seed}-{arm}",
        "condition": arm,
        "seed": seed,
        "model": "d360m",
        "ctx": 1024,
        "train_corpus": "dataset/corpus-receipt.json",
        "sidecar_name": (
            "dense_target_weights" if arm == "dense" else "split90_target_weights"
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
        "snap_frac": 0.1,
        "ckpt_minutes": 30,
    }
    return yaml.safe_dump(value, sort_keys=False).encode("utf-8")


def _zip_info(name: str, *, mode: int = 0o644) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=NORMALIZED_TIME)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    file_type = stat.S_IFDIR if name.endswith("/") else stat.S_IFREG
    permissions = 0o755 if name.endswith("/") else mode
    info.external_attr = (file_type | permissions) << 16
    return info


def _write_zip(path: Path, entries: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name, content in entries:
            archive.writestr(
                _zip_info(name),
                content,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _build_release(
    root: Path,
    *,
    provider: str,
    seeds: tuple[int, ...],
    source_commit: str = SOURCE_COMMIT,
    source_dirty: bool = False,
    metadata_provider: str | None = None,
    assignment_bytes: bytes | None = None,
    corpus_bytes: bytes | None = None,
    evaluation_bytes: bytes | None = None,
    environment_bytes: bytes | None = None,
    omitted_cells: frozenset[tuple[int, str]] = frozenset(),
    run_ids: dict[tuple[int, str], str] | None = None,
) -> FixtureRelease:
    root.mkdir(parents=True)
    assignment_content = assignment_bytes or _canonical_pretty(_assignment())
    corpus_content = corpus_bytes or _canonical_pretty(
        {
            "schema_version": 2,
            "contract_id": "memorysplit-reasoning-dataset-v2",
            "raw_target_tokens": 7_120_879_616,
        }
    )
    evaluation_content = evaluation_bytes or (
        b"schema_version: 2\n"
        b"preregistration_id: memorysplit-confirmatory-v2\n"
        b"frozen: true\n"
    )
    profile_path = (
        "cluster/profiles/illumina-usfc-prd.json"
        if provider == ILLUMINA
        else "cluster/profiles/aws-p5.48xlarge.json"
    )
    profile_content = _canonical_pretty(
        (
            {
                "schema_version": 1,
                "profile_id": ILLUMINA,
                "provider": ILLUMINA,
            }
            if provider == ILLUMINA
            else _aws_profile()
        )
    )
    requirements_content = (
        b"pytest==8.4.1\npyyaml==6.0.2\n"
        if environment_bytes is None
        else environment_bytes
    )
    environment_path = (
        "requirements.txt" if provider == ILLUMINA else "requirements-aws-p5.lock"
    )
    payload = {
        ASSIGNMENT_PATH: assignment_content,
        CORPUS_IDENTITY_PATH: corpus_content,
        EVALUATION_IDENTITY_PATH: evaluation_content,
        profile_path: profile_content,
        environment_path: requirements_content,
    }
    dataset_pointer_path = "DATASET-POINTER-AWS.json"
    dataset_pointer_content = _canonical_pretty(
        {
            "schema_version": 1,
            "provider": AWS,
            "dataset_id": "memorysplit-parallel-corpus-v2",
            "durable_uri_env": "MS_S3_ROOT",
            "receipt_relative_path": "dataset/corpus-receipt.json",
            "full_corpus_in_release": False,
        }
    )
    if provider == AWS:
        payload[dataset_pointer_path] = dataset_pointer_content
    selected_run_ids = run_ids or {}
    for seed in seeds:
        for arm in ("dense", "split90"):
            if (seed, arm) in omitted_cells:
                continue
            payload[f"configs/360m-v2/{arm}-s{seed}.yaml"] = _run_config(
                seed,
                arm,
                run_id=selected_run_ids.get((seed, arm)),
            )

    member_rows = []
    for name in sorted(payload):
        row = {
            "path": name,
            "bytes": len(payload[name]),
            "sha256": _sha256(payload[name]),
            "git_blob": "b" * 40,
        }
        if provider == AWS:
            row["git_mode"] = "100644"
        member_rows.append(row)
    seed_assignment = {
        "cohort_id": COHORT_ID,
        "provider": provider,
        "seeds": list(seeds),
        "arms": ["dense", "split90"],
    }
    if provider == ILLUMINA:
        metadata = {
            "schema_version": 1,
            "provider": metadata_provider or provider,
            "source": {
                "commit": source_commit,
                "dirty": source_dirty,
            },
            "profile_sha256": _sha256(profile_content),
            "preregistration_sha256": _sha256(evaluation_content),
            "environment_hashes": {
                environment_path: _sha256(requirements_content),
            },
            "members": member_rows,
            "seed_assignment": seed_assignment,
        }
    else:
        config_sha256 = {
            name: _sha256(content)
            for name, content in payload.items()
            if name.startswith("configs/360m-v2/")
        }
        metadata = {
            "schema_version": 1,
            "package_format_version": 1,
            "provider": metadata_provider or provider,
            "source": {
                "commit": source_commit,
                "dirty": source_dirty,
            },
            "seed_assignment": seed_assignment,
            "cohort_assignment": {
                "path": ASSIGNMENT_PATH,
                "sha256": _sha256(assignment_content),
            },
            "profile": {
                "path": profile_path,
                "sha256": _sha256(profile_content),
            },
            "environment": {
                "path": environment_path,
                "sha256": _sha256(requirements_content),
            },
            "dataset_pointer": {
                "path": dataset_pointer_path,
                "sha256": _sha256(dataset_pointer_content),
            },
            "config_sha256": config_sha256,
            "members": member_rows,
        }
    payload["RELEASE-METADATA.json"] = _canonical_pretty(metadata)
    sums = "".join(
        f"{_sha256(payload[name])}  {name}\n" for name in sorted(payload)
    ).encode("utf-8")
    entries = sorted(payload.items())
    entries.append(("SHA256SUMS", sums))

    members_sha256 = _sha256(sums)
    release_id = (
        f"r1-{members_sha256[:16]}"
        if provider == ILLUMINA
        else f"aws-p5-r1-{members_sha256[:16]}"
    )
    archive_name = (
        f"ms-illumina-r1-{members_sha256[:16]}.zip"
        if provider == ILLUMINA
        else f"ms-aws-p5-r1-{members_sha256[:16]}.zip"
    )
    archive_path = root / archive_name
    _write_zip(archive_path, entries)
    archive_content = archive_path.read_bytes()
    archive_sha256 = _sha256(archive_content)
    receipt_name = "RELEASE.json" if provider == ILLUMINA else "RELEASE-AWS-P5.json"
    receipt_path = root / receipt_name
    receipt = {
        "schema_version": 1,
        "release_id": release_id,
        "provider": provider,
        "archive": {
            "path": archive_name,
            "sha256": archive_sha256,
            "bytes": len(archive_content),
        },
        "source": {
            "commit": source_commit,
            "dirty": source_dirty,
        },
        "members_sha256": members_sha256,
    }
    if provider == AWS:
        receipt.update(
            {
                "package_format_version": 1,
                "seed_assignment": seed_assignment,
                "cohort_assignment": metadata["cohort_assignment"],
                "cohort_assignment_sha256": _sha256(assignment_content),
                "profile": metadata["profile"],
                "profile_sha256": _sha256(profile_content),
                "environment": metadata["environment"],
                "environment_sha256": _sha256(requirements_content),
                "dataset_pointer": metadata["dataset_pointer"],
                "dataset_pointer_sha256": _sha256(dataset_pointer_content),
                "config_sha256": metadata["config_sha256"],
            }
        )
    receipt_path.write_bytes(_canonical_pretty(receipt))
    sha256_path = root / f"{archive_name}.sha256"
    sha256_path.write_text(
        f"{archive_sha256}  {archive_name}\n",
        encoding="ascii",
    )
    return FixtureRelease(
        receipt=receipt_path,
        archive=archive_path,
        sha256_file=sha256_path,
        release_id=release_id,
        archive_sha256=archive_sha256,
        assignment_sha256=_sha256(assignment_content),
        corpus_sha256=_sha256(corpus_content),
        evaluation_sha256=_sha256(evaluation_content),
    )


def _invoke(illumina: Path, aws: Path) -> subprocess.CompletedProcess[str]:
    assert SCRIPT.is_file(), "cohort release verifier has not been implemented"
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--illumina",
            str(illumina),
            "--aws",
            str(aws),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _load_verifier_module():
    name = f"verify_cohort_releases_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_module(path: Path, prefix: str):
    name = f"{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _materialize_integration_ref(root: Path, revision: str) -> Path:
    archive_path = root / "integration.tar"
    checkout = root / "integration"
    root.mkdir()
    checkout.mkdir()
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "archive",
            "--format=tar",
            f"--output={archive_path}",
            revision,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    with tarfile.open(archive_path, mode="r") as archive:
        archive.extractall(checkout, filter="data")
    return checkout


def _assert_rejected(completed: subprocess.CompletedProcess[str]) -> None:
    assert completed.returncode == 2
    assert completed.stderr == ""
    assert len(completed.stdout.splitlines()) == 1
    report = json.loads(completed.stdout)
    assert report["schema_version"] == 1
    assert report["ok"] is False
    assert report["error"]["code"] == "COHORT_RELEASES_REJECTED"
    assert completed.stdout == (
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def _rename_bound_archive(release: FixtureRelease, new_name: str) -> None:
    renamed = release.archive.with_name(new_name)
    release.archive.rename(renamed)
    receipt = json.loads(release.receipt.read_text(encoding="utf-8"))
    receipt["archive"]["path"] = new_name
    release.receipt.write_bytes(_canonical_pretty(receipt))
    renamed.with_name(f"{new_name}.sha256").write_text(
        f"{receipt['archive']['sha256']}  {new_name}\n",
        encoding="ascii",
    )


def _bound_archive(release: FixtureRelease) -> Path:
    receipt = json.loads(release.receipt.read_text(encoding="utf-8"))
    return release.receipt.parent / receipt["archive"]["path"]


def _archive_entries(release: FixtureRelease) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(_bound_archive(release)) as archive:
        return [(info.filename, archive.read(info)) for info in archive.infolist()]


def _rewrite_release(
    release: FixtureRelease,
    entries: list[tuple[str, bytes]],
    *,
    bind_new_sums: bool = False,
) -> Path:
    receipt = json.loads(release.receipt.read_text(encoding="utf-8"))
    archive_path = release.receipt.parent / receipt["archive"]["path"]
    _write_zip(archive_path, entries)
    if bind_new_sums:
        sum_entries = [content for name, content in entries if name == "SHA256SUMS"]
        assert len(sum_entries) == 1
        members_sha256 = _sha256(sum_entries[0])
        receipt["members_sha256"] = members_sha256
        receipt["release_id"] = (
            f"r1-{members_sha256[:16]}"
            if receipt["provider"] == ILLUMINA
            else f"aws-p5-r1-{members_sha256[:16]}"
        )
        prefix = "ms-illumina-r1" if receipt["provider"] == ILLUMINA else "ms-aws-p5-r1"
        rebound = archive_path.with_name(f"{prefix}-{members_sha256[:16]}.zip")
        archive_path.rename(rebound)
        archive_path = rebound
        receipt["archive"]["path"] = rebound.name
    archive_content = archive_path.read_bytes()
    archive_sha256 = _sha256(archive_content)
    receipt["archive"]["sha256"] = archive_sha256
    receipt["archive"]["bytes"] = len(archive_content)
    release.receipt.write_bytes(_canonical_pretty(receipt))
    archive_path.with_name(f"{archive_path.name}.sha256").write_text(
        f"{archive_sha256}  {archive_path.name}\n",
        encoding="ascii",
    )
    return archive_path


def _refresh_external_binding(release: FixtureRelease) -> None:
    receipt = json.loads(release.receipt.read_text(encoding="utf-8"))
    archive_path = release.receipt.parent / receipt["archive"]["path"]
    archive_content = archive_path.read_bytes()
    archive_sha256 = _sha256(archive_content)
    receipt["archive"]["sha256"] = archive_sha256
    receipt["archive"]["bytes"] = len(archive_content)
    release.receipt.write_bytes(_canonical_pretty(receipt))
    archive_path.with_name(f"{archive_path.name}.sha256").write_text(
        f"{archive_sha256}  {archive_path.name}\n",
        encoding="ascii",
    )


def _replace_entry(
    entries: list[tuple[str, bytes]],
    name: str,
    content: bytes,
) -> list[tuple[str, bytes]]:
    assert sum(entry_name == name for entry_name, _ in entries) == 1
    return [
        (entry_name, content if entry_name == name else entry_content)
        for entry_name, entry_content in entries
    ]


def _resign_sums(
    entries: list[tuple[str, bytes]],
) -> list[tuple[str, bytes]]:
    payload = {name: content for name, content in entries if name != "SHA256SUMS"}
    sums = "".join(
        f"{_sha256(payload[name])}  {name}\n" for name in sorted(payload)
    ).encode("utf-8")
    return sorted(payload.items()) + [("SHA256SUMS", sums)]


def _replace_aws_bound_member(
    release: FixtureRelease,
    *,
    path: str,
    content: bytes,
    metadata_field: str,
    receipt_field: str,
) -> None:
    entries = dict(_archive_entries(release))
    entries.pop("SHA256SUMS")
    entries[path] = content
    digest = _sha256(content)

    metadata = json.loads(entries["RELEASE-METADATA.json"])
    metadata[metadata_field]["sha256"] = digest
    for member in metadata["members"]:
        if member["path"] == path:
            member["bytes"] = len(content)
            member["sha256"] = digest
            break
    else:
        raise AssertionError(f"metadata does not list {path}")
    entries["RELEASE-METADATA.json"] = _canonical_pretty(metadata)

    receipt = json.loads(release.receipt.read_text(encoding="utf-8"))
    receipt[receipt_field] = digest
    receipt[metadata_field]["sha256"] = digest
    release.receipt.write_bytes(_canonical_pretty(receipt))
    _rewrite_release(
        release,
        _resign_sums(list(entries.items())),
        bind_new_sums=True,
    )


def _replace_metadata(
    release: FixtureRelease,
    mutate,
) -> None:
    entries = dict(_archive_entries(release))
    entries.pop("SHA256SUMS")
    metadata = json.loads(entries["RELEASE-METADATA.json"])
    mutate(metadata)
    entries["RELEASE-METADATA.json"] = _canonical_pretty(metadata)
    _rewrite_release(
        release,
        _resign_sums(list(entries.items())),
        bind_new_sums=True,
    )


def _pair(tmp_path: Path) -> tuple[FixtureRelease, FixtureRelease]:
    return (
        _build_release(
            tmp_path / "illumina",
            provider=ILLUMINA,
            seeds=(0,),
        ),
        _build_release(
            tmp_path / "aws",
            provider=AWS,
            seeds=(1, 2, 3, 4),
        ),
    )


def test_accepts_release_built_by_integrated_illumina_packager(
    tmp_path: Path,
) -> None:
    integration_root = _materialize_integration_ref(
        tmp_path / "integrated-ref",
        ILLUMINA_INTEGRATION_REF,
    )
    package_tests = _load_module(
        integration_root / "tests/test_package_illumina_handoff.py",
        "integrated_illumina_package_tests",
    )
    package_module = _load_module(
        integration_root / "scripts/package_illumina_handoff.py",
        "integrated_illumina_packager",
    )
    fixture_root = tmp_path / "real-illumina"
    fixture_root.mkdir()
    source = package_tests._minimal_repo(fixture_root)
    reasoning_identity = (integration_root / CORPUS_IDENTITY_PATH).read_bytes()
    package_tests._write(source / CORPUS_IDENTITY_PATH, reasoning_identity)
    package_tests._commit_mutation(source, "add reasoning identity")
    illumina = package_module.build_handoff(
        source_root=source,
        out_dir=tmp_path / "real-illumina-release",
    )
    with zipfile.ZipFile(illumina.archive) as archive:
        assignment = archive.read(ASSIGNMENT_PATH)
        corpus = archive.read(CORPUS_IDENTITY_PATH)
        evaluation = archive.read(EVALUATION_IDENTITY_PATH)
        metadata = json.loads(archive.read("RELEASE-METADATA.json"))
    assert metadata["preregistration_sha256"] == _sha256(evaluation)
    source_commit = json.loads(illumina.release.read_text())["source"]["commit"]
    aws = _build_release(
        tmp_path / "aws",
        provider=AWS,
        seeds=(1, 2, 3, 4),
        source_commit=source_commit,
        assignment_bytes=assignment,
        corpus_bytes=corpus,
        evaluation_bytes=evaluation,
    )

    completed = _invoke(illumina.release, aws.receipt)

    assert completed.returncode == 0, completed.stdout
    assert json.loads(completed.stdout)["ok"] is True


def test_rejects_missing_illumina_preregistration_commitment(
    tmp_path: Path,
) -> None:
    illumina, aws = _pair(tmp_path)
    _replace_metadata(
        illumina,
        lambda metadata: metadata.pop("preregistration_sha256"),
    )

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_unbound_illumina_preregistration_commitment(
    tmp_path: Path,
) -> None:
    illumina, aws = _pair(tmp_path)
    _replace_metadata(
        illumina,
        lambda metadata: metadata.__setitem__(
            "preregistration_sha256",
            "0" * 64,
        ),
    )

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_accepts_exact_disjoint_five_seed_cohort_and_emits_canonical_json(
    tmp_path: Path,
) -> None:
    illumina = _build_release(
        tmp_path / "illumina",
        provider=ILLUMINA,
        seeds=(0,),
    )
    aws = _build_release(
        tmp_path / "aws",
        provider=AWS,
        seeds=(1, 2, 3, 4),
    )

    completed = _invoke(illumina.receipt, aws.receipt)

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert len(completed.stdout.splitlines()) == 1
    report = json.loads(completed.stdout)
    assert completed.stdout == (
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    assert report == {
        "schema_version": 1,
        "ok": True,
        "cohort_id": COHORT_ID,
        "source_commit": SOURCE_COMMIT,
        "cohort_assignment_sha256": illumina.assignment_sha256,
        "corpus_identity_sha256": illumina.corpus_sha256,
        "evaluation_identity_sha256": illumina.evaluation_sha256,
        "arms": ["dense", "split90"],
        "complete_cohort": [0, 1, 2, 3, 4],
        "illumina": {
            "provider": ILLUMINA,
            "release_id": illumina.release_id,
            "archive_sha256": illumina.archive_sha256,
            "seeds": [0],
        },
        "aws": {
            "provider": AWS,
            "release_id": aws.release_id,
            "archive_sha256": aws.archive_sha256,
            "seeds": [1, 2, 3, 4],
        },
    }


def test_rejects_archive_name_not_bound_to_provider_and_member_hash(
    tmp_path: Path,
) -> None:
    illumina = _build_release(
        tmp_path / "illumina",
        provider=ILLUMINA,
        seeds=(0,),
    )
    aws = _build_release(
        tmp_path / "aws",
        provider=AWS,
        seeds=(1, 2, 3, 4),
    )
    _rename_bound_archive(aws, "renamed-but-rehashed.zip")

    completed = _invoke(illumina.receipt, aws.receipt)

    _assert_rejected(completed)


def test_rejects_seed_overlap(tmp_path: Path) -> None:
    illumina = _build_release(
        tmp_path / "illumina",
        provider=ILLUMINA,
        seeds=(0, 1),
    )
    aws = _build_release(
        tmp_path / "aws",
        provider=AWS,
        seeds=(1, 2, 3, 4),
    )

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_missing_seed(tmp_path: Path) -> None:
    illumina = _build_release(
        tmp_path / "illumina",
        provider=ILLUMINA,
        seeds=(0,),
    )
    aws = _build_release(
        tmp_path / "aws",
        provider=AWS,
        seeds=(1, 2, 3),
    )

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_different_source_commits(tmp_path: Path) -> None:
    illumina = _build_release(
        tmp_path / "illumina",
        provider=ILLUMINA,
        seeds=(0,),
    )
    aws = _build_release(
        tmp_path / "aws",
        provider=AWS,
        seeds=(1, 2, 3, 4),
        source_commit="c" * 40,
    )

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_dirty_source_receipt_and_metadata(tmp_path: Path) -> None:
    illumina = _build_release(
        tmp_path / "illumina",
        provider=ILLUMINA,
        seeds=(0,),
        source_dirty=True,
    )
    aws = _build_release(
        tmp_path / "aws",
        provider=AWS,
        seeds=(1, 2, 3, 4),
    )

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_semantically_equal_assignment_with_different_bytes(
    tmp_path: Path,
) -> None:
    illumina = _build_release(
        tmp_path / "illumina",
        provider=ILLUMINA,
        seeds=(0,),
    )
    compact_assignment = (
        json.dumps(
            _assignment(),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    aws = _build_release(
        tmp_path / "aws",
        provider=AWS,
        seeds=(1, 2, 3, 4),
        assignment_bytes=compact_assignment,
    )

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_semantically_wrong_assignment(tmp_path: Path) -> None:
    wrong = _assignment()
    wrong["model_parameters"] = 356_033_535
    illumina = _build_release(
        tmp_path / "illumina",
        provider=ILLUMINA,
        seeds=(0,),
        assignment_bytes=_canonical_pretty(wrong),
    )
    aws = _build_release(
        tmp_path / "aws",
        provider=AWS,
        seeds=(1, 2, 3, 4),
    )

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_different_corpus_identity(tmp_path: Path) -> None:
    illumina = _build_release(
        tmp_path / "illumina",
        provider=ILLUMINA,
        seeds=(0,),
    )
    aws = _build_release(
        tmp_path / "aws",
        provider=AWS,
        seeds=(1, 2, 3, 4),
        corpus_bytes=_canonical_pretty(
            {
                "schema_version": 2,
                "contract_id": "different-corpus",
                "raw_target_tokens": 7_120_879_616,
            }
        ),
    )

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_different_evaluation_identity(tmp_path: Path) -> None:
    illumina = _build_release(
        tmp_path / "illumina",
        provider=ILLUMINA,
        seeds=(0,),
    )
    aws = _build_release(
        tmp_path / "aws",
        provider=AWS,
        seeds=(1, 2, 3, 4),
        evaluation_bytes=(
            b"schema_version: 2\n"
            b"preregistration_id: different-evaluation\n"
            b"frozen: true\n"
        ),
    )

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_partial_dense_split90_pair(tmp_path: Path) -> None:
    illumina = _build_release(
        tmp_path / "illumina",
        provider=ILLUMINA,
        seeds=(0,),
    )
    aws = _build_release(
        tmp_path / "aws",
        provider=AWS,
        seeds=(1, 2, 3, 4),
        omitted_cells=frozenset({(3, "split90")}),
    )

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_duplicate_run_ids(tmp_path: Path) -> None:
    duplicate = "memorysplit-v2-360m-s1-dense"
    illumina = _build_release(
        tmp_path / "illumina",
        provider=ILLUMINA,
        seeds=(0,),
    )
    aws = _build_release(
        tmp_path / "aws",
        provider=AWS,
        seeds=(1, 2, 3, 4),
        run_ids={(2, "dense"): duplicate},
    )

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_inconsistent_provider_metadata(tmp_path: Path) -> None:
    illumina = _build_release(
        tmp_path / "illumina",
        provider=ILLUMINA,
        seeds=(0,),
        metadata_provider=AWS,
    )
    aws = _build_release(
        tmp_path / "aws",
        provider=AWS,
        seeds=(1, 2, 3, 4),
    )

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_inconsistent_aws_receipt_path_hash_binding(
    tmp_path: Path,
) -> None:
    illumina, aws = _pair(tmp_path)
    receipt = json.loads(aws.receipt.read_text(encoding="utf-8"))
    receipt["profile"]["sha256"] = "0" * 64
    aws.receipt.write_bytes(_canonical_pretty(receipt))

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_semantically_wrong_provider_profile_when_fully_rehashed(
    tmp_path: Path,
) -> None:
    illumina, aws = _pair(tmp_path)
    _replace_aws_bound_member(
        aws,
        path="cluster/profiles/aws-p5.48xlarge.json",
        content=_canonical_pretty(
            {
                "schema_version": 1,
                "provider": ILLUMINA,
            }
        ),
        metadata_field="profile",
        receipt_field="profile_sha256",
    )

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_t3_zero_gpu_seed0_profile_when_fully_rehashed(
    tmp_path: Path,
) -> None:
    illumina, aws = _pair(tmp_path)
    hostile = _aws_profile()
    hostile["instance_type"] = "t3.micro"
    hostile["purchase_model"] = "spot"
    hostile["gpu"] = {
        "model": "none",
        "allocated": 0,
        "seed_train_groups": [0, 0],
    }
    hostile["assigned_seeds"] = [0]
    _replace_aws_bound_member(
        aws,
        path="cluster/profiles/aws-p5.48xlarge.json",
        content=_canonical_pretty(hostile),
        metadata_field="profile",
        receipt_field="profile_sha256",
    )

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_empty_hash_bound_aws_environment_lock(
    tmp_path: Path,
) -> None:
    illumina, aws = _pair(tmp_path)
    _replace_aws_bound_member(
        aws,
        path="requirements-aws-p5.lock",
        content=b"",
        metadata_field="environment",
        receipt_field="environment_sha256",
    )

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_semantically_wrong_dataset_pointer_when_fully_rehashed(
    tmp_path: Path,
) -> None:
    illumina, aws = _pair(tmp_path)
    _replace_aws_bound_member(
        aws,
        path="DATASET-POINTER-AWS.json",
        content=_canonical_pretty(
            {
                "schema_version": 1,
                "provider": AWS,
                "dataset_id": "memorysplit-parallel-corpus-v2",
                "durable_uri_env": "MS_S3_ROOT",
                "receipt_relative_path": "dataset/corpus-receipt.json",
                "full_corpus_in_release": True,
            }
        ),
        metadata_field="dataset_pointer",
        receipt_field="dataset_pointer_sha256",
    )

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_external_archive_hash_mismatch(tmp_path: Path) -> None:
    illumina, aws = _pair(tmp_path)
    with _bound_archive(aws).open("ab") as stream:
        stream.write(b"tamper")

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_external_sha256_sidecar_mismatch(tmp_path: Path) -> None:
    illumina, aws = _pair(tmp_path)
    aws.sha256_file.write_text(
        f"{'0' * 64}  {aws.archive.name}\n",
        encoding="ascii",
    )

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_internal_member_hash_mismatch(tmp_path: Path) -> None:
    illumina, aws = _pair(tmp_path)
    entries = _archive_entries(aws)
    entries = _replace_entry(
        entries,
        "configs/360m-v2/dense-s1.yaml",
        b"tampered: true\n",
    )
    _rewrite_release(aws, entries)

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_sha256sums_omission_even_when_externally_rehashed(
    tmp_path: Path,
) -> None:
    illumina, aws = _pair(tmp_path)
    entries = _archive_entries(aws)
    sums = dict(entries)["SHA256SUMS"].decode("utf-8").splitlines()
    shortened = ("\n".join(sums[1:]) + "\n").encode("utf-8")
    entries = _replace_entry(entries, "SHA256SUMS", shortened)
    _rewrite_release(aws, entries, bind_new_sums=True)

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_duplicate_zip_members(tmp_path: Path) -> None:
    illumina, aws = _pair(tmp_path)
    entries = _archive_entries(aws)
    duplicate = next(entry for entry in entries if entry[0] == ASSIGNMENT_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _rewrite_release(aws, entries + [duplicate])

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


@pytest.mark.parametrize(
    "colliding_entries",
    [
        [("collision", b"file"), ("collision/child", b"child")],
        [("collision/", b""), ("collision", b"file")],
        [("caf\u00e9", b"composed"), ("cafe\u0301", b"decomposed")],
    ],
    ids=("file-ancestor", "file-directory", "normalized-duplicate"),
)
def test_rejects_unsafe_zip_topology_before_member_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    colliding_entries: list[tuple[str, bytes]],
) -> None:
    illumina, _ = _pair(tmp_path)
    entries = _archive_entries(illumina)
    _rewrite_release(illumina, entries + colliding_entries)
    module = _load_verifier_module()

    def unexpected_member_read(*_args, **_kwargs):
        raise AssertionError("unsafe topology reached member parsing")

    monkeypatch.setattr(module, "_member_bytes", unexpected_member_read)

    with pytest.raises(module.VerificationError, match="collision|topology"):
        module._verify_release(
            illumina.receipt,
            expected_provider=module.ILLUMINA_PROVIDER,
        )


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../escape",
        "/absolute",
        "configs//double.yaml",
        "configs\\windows.yaml",
        "C:/drive.yaml",
    ],
)
def test_rejects_unsafe_zip_member_names(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    illumina, aws = _pair(tmp_path)
    entries = _archive_entries(aws)
    _rewrite_release(aws, entries + [(unsafe_name, b"unsafe\n")])

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_zip_symlink_member(tmp_path: Path) -> None:
    illumina, aws = _pair(tmp_path)
    archive_path = _bound_archive(aws)
    with zipfile.ZipFile(archive_path, mode="a") as archive:
        info = _zip_info("linked-member")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, b"target")
    _refresh_external_binding(aws)

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_symlinked_archive_even_when_target_hash_matches(
    tmp_path: Path,
) -> None:
    illumina, aws = _pair(tmp_path)
    archive_path = _bound_archive(aws)
    target = archive_path.with_name("actual-archive.zip")
    archive_path.rename(target)
    archive_path.symlink_to(target.name)

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_rejects_duplicate_json_keys_in_release_receipt(tmp_path: Path) -> None:
    illumina, aws = _pair(tmp_path)
    receipt = aws.receipt.read_text(encoding="utf-8")
    aws.receipt.write_text(
        receipt.replace(
            '"schema_version": 1,',
            '"schema_version": 1,\n  "schema_version": 1,',
            1,
        ),
        encoding="utf-8",
    )

    _assert_rejected(_invoke(illumina.receipt, aws.receipt))


def test_cli_argument_failure_is_one_canonical_json_object() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )

    _assert_rejected(completed)


def test_descriptor_pinning_rejects_archive_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    illumina, aws = _pair(tmp_path)
    module = _load_verifier_module()
    original_open = module._open_regular_at
    raced = False

    def racing_open(directory_fd: int, name: str, label: str) -> int:
        nonlocal raced
        descriptor = original_open(directory_fd, name, label)
        if not raced and name.endswith(".zip"):
            replacement = ".replacement-archive"
            replacement_fd = os.open(
                replacement,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            try:
                os.write(replacement_fd, b"not-the-authenticated-archive")
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

    monkeypatch.setattr(module, "_open_regular_at", racing_open)

    with pytest.raises(module.VerificationError, match="changed|replaced"):
        module.verify_cohort_releases(
            illumina_release=illumina.receipt,
            aws_release=aws.receipt,
        )
    assert raced is True


@pytest.mark.parametrize("failure_point", ["descriptor-check", "fdopen"])
def test_archive_descriptor_is_closed_on_pre_fdopen_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    illumina, _ = _pair(tmp_path)
    module = _load_verifier_module()
    original_open = module._open_regular_at
    opened_archives: list[int] = []

    def capturing_open(directory_fd: int, name: str, label: str) -> int:
        descriptor = original_open(directory_fd, name, label)
        if name.endswith(".zip"):
            opened_archives.append(descriptor)
        return descriptor

    monkeypatch.setattr(module, "_open_regular_at", capturing_open)
    if failure_point == "descriptor-check":
        original_assert = module._assert_descriptor_names_entry

        def failing_assert(directory_fd, name, descriptor, label):
            if name.endswith(".zip"):
                raise module.VerificationError("injected pre-fdopen failure")
            return original_assert(directory_fd, name, descriptor, label)

        monkeypatch.setattr(module, "_assert_descriptor_names_entry", failing_assert)
        expected_error = module.VerificationError
    else:

        def failing_fdopen(*_args, **_kwargs):
            raise OSError("injected fdopen failure")

        monkeypatch.setattr(module.os, "fdopen", failing_fdopen)
        expected_error = OSError

    with pytest.raises(expected_error):
        module._verify_release(
            illumina.receipt,
            expected_provider=module.ILLUMINA_PROVIDER,
        )

    assert len(opened_archives) == 1
    with pytest.raises(OSError):
        os.fstat(opened_archives[0])


def test_runbook_is_dry_run_first_and_covers_complete_p5_lifecycle() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    lowered = text.lower()

    for required in (
        "integration constants",
        ILLUMINA_INTEGRATION_REF,
        TASK_3_5_REF,
        TASK_6_REF,
        TASK_7_REF,
        TASK_8_REF,
        "p5.48xlarge",
        "aws pricing get-products",
        "running on-demand p instances",
        "s3://",
        "aws ssm start-session",
        "amazon ec2 nvme instance storage",
        "mdadm --create",
        "functional canary",
        "100-update",
        "paired resume probe",
        "sequential",
        "deadline mode",
        "runs instantiate",
        "--dataset-pointer",
        "--dataset-root",
        "--environment-receipt",
        "ms_runtime_uid",
        "ms_runtime_gid",
        "--owner-uid",
        "--owner-gid",
        "--aws-private-home",
        "--authorize-destructive-instance-store",
        "--instance-id",
        "--terminate-at",
        "cluster/aws/p5/bootstrap.py",
        "cluster/aws/p5/launch_seed_pair.py",
        "submit",
        "resume",
        "evaluate",
        "collect",
        "terminate-instances",
        "directional_only",
        "sign_consistent_only",
        "supports_effect",
        "supports_practical_null",
        "inconclusive",
    ):
        assert required in lowered
    for forbidden in (
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "run-instances",
    ):
        assert forbidden not in text
    evaluator = re.compile(
        r"python -m evals\.confirmatory evaluate\s+"
        r"--run .+?--sealed-release .+?"
        r"--expected-study-lock-sha256 .+?"
        r"--device .+?--output-dir ",
        flags=re.DOTALL,
    )
    assert evaluator.search(text)
    assert text.count('--instance-id "$MS_INSTANCE_ID"') >= 2
    assert text.index("# DRY RUN: bootstrap") < text.index(
        "--authorize-destructive-instance-store"
    )
    assert "# DRY RUN." in text
    assert "--dry-run" in text
    assert "--dryrun" in text
    assert text.count("--apply") >= 7


def test_every_runbook_bash_block_is_syntactically_valid() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)\n```", text, flags=re.DOTALL)

    assert blocks
    for index, block in enumerate(blocks):
        completed = subprocess.run(
            ["bash", "-n"],
            input=block,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, (
            f"bash block {index} is invalid: {completed.stderr}"
        )
