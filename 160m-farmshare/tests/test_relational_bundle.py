from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from experiment.artifacts import canonical_json_bytes, canonical_sha256
from scripts.freeze_relational_study import (
    make_fixture_freeze,
    write_freeze_manifest,
)
from scripts.make_relational_manifest import build_manifest, write_run_manifest
from scripts.package_relational_run import package_run
from scripts.verify_relational_bundle import (
    BundleVerificationError,
    verify_bundle,
    verify_extracted_bundle,
)
from tests.relational_asset_fixtures import stage_launchable_relational_assets


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CONTRACTS = {
    "cluster/RELATIONAL-RUNBOOK.md",
    "cluster/aws/run_relational_manifest.py",
    "cluster/config.env",
    "cluster/setup_env.sh",
    "cluster/slurm/relational_train.sbatch",
    "cluster/submit_relational_manifest.sh",
    "contracts/freeze.json",
    "contracts/run-manifest.json",
    "contracts/preregistration.md",
    "contracts/schemas/freeze-v1.schema.json",
    "contracts/schemas/relational-asset-receipt-v1.schema.json",
    "contracts/schemas/run-config-v1.schema.json",
    "contracts/schemas/run-manifest-v1.schema.json",
    "contracts/schemas/relational-result-v1.schema.json",
    "requirements/base.in",
    "requirements/macos-arm64-py312.lock",
    "requirements/linux-x86_64-cuda-py312.lock",
    "fixtures/relational-smoke.json",
    "offline_tests/verify_contracts.py",
}


def _fixture_contracts(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    freeze = make_fixture_freeze()
    manifest = build_manifest(freeze)
    freeze_path = tmp_path / "freeze.json"
    manifest_path = tmp_path / "run-manifest.json"
    write_freeze_manifest(freeze_path, freeze)
    write_run_manifest(manifest_path, manifest)
    return freeze_path, manifest_path


def _build(tmp_path: Path, name: str = "relational-run.tar.gz") -> Path:
    freeze, manifest = _fixture_contracts(tmp_path)
    return package_run(
        tmp_path / name,
        source_root=REPO_ROOT,
        freeze_path=freeze,
        run_manifest_path=manifest,
        require_clean=False,
    )


def _archive_files(path: Path) -> tuple[list[tarfile.TarInfo], dict[str, bytes]]:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        files = {
            member.name: archive.extractfile(member).read()
            for member in members
        }
    return members, files


def _write_archive(path: Path, entries: list[tuple[tarfile.TarInfo, bytes]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(
                fileobj=zipped,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as archive:
                for member, payload in entries:
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))


def test_bundle_is_byte_deterministic_hash_complete_and_current(tmp_path):
    first = _build(tmp_path / "first")
    second = _build(tmp_path / "second")

    assert first.read_bytes() == second.read_bytes()
    members, files = _archive_files(first)
    names = set(files)
    assert REQUIRED_CONTRACTS <= names
    assert sum(
        name.startswith("contracts/configs/") and name.endswith(".json")
        for name in names
    ) == 35
    assert not any(
        name.endswith((".bin", ".pt", ".pth", ".safetensors"))
        for name in names
    )
    assert [member.name for member in members] == sorted(names)
    assert all(
        member.isfile()
        and member.uid == 0
        and member.gid == 0
        and member.uname == ""
        and member.gname == ""
        and member.mtime == 0
        and member.mode == 0o644
        for member in members
    )

    bundle_manifest = json.loads(files["BUNDLE-MANIFEST.json"])
    indexed = {item["path"]: item for item in bundle_manifest["members"]}
    assert set(indexed) == names - {"BUNDLE-MANIFEST.json"}
    for name, item in indexed.items():
        assert item["bytes"] == len(files[name])
        assert item["sha256"] == hashlib.sha256(files[name]).hexdigest()
    assert bundle_manifest["run_count"] == 35
    assert bundle_manifest["eval_control_count"] == 11
    assert bundle_manifest["eval_cell_count"] == 22
    assert bundle_manifest["external_assets_only"] is True
    assert bundle_manifest["external_assets"] == []


def test_independent_verifier_extracts_and_runs_offline_contracts(tmp_path):
    bundle = _build(tmp_path / "build")
    extraction = tmp_path / "extracted"

    report = verify_bundle(bundle, extract_to=extraction)
    second = verify_extracted_bundle(extraction)

    assert report["verified"] is True
    assert report["offline_tests_passed"] is True
    assert report["member_count"] > 35
    assert second["verified"] is True
    assert (extraction / "contracts" / "run-manifest.json").is_file()


def test_verifier_rejects_member_hash_tampering(tmp_path):
    bundle = _build(tmp_path / "build")
    members, files = _archive_files(bundle)
    entries = []
    for member in members:
        payload = files[member.name]
        if member.name == "fixtures/relational-smoke.json":
            payload += b" "
        entries.append((member, payload))
    tampered = tmp_path / "tampered.tar.gz"
    _write_archive(tampered, entries)

    with pytest.raises(BundleVerificationError, match="hash|size"):
        verify_bundle(tampered, run_tests=False)


def test_verifier_rejects_external_assets_not_derived_from_receipt(tmp_path):
    bundle = _build(tmp_path / "build")
    members, files = _archive_files(bundle)
    manifest = json.loads(files["BUNDLE-MANIFEST.json"])
    manifest["external_assets"].append(
        {
            "kind": "corpus",
            "path": "relational/forged/train.bin",
            "commitment_sha256": "0" * 64,
            "sha256": "0" * 64,
            "bytes": 1,
            "build_manifest_sha256": "0" * 64,
        }
    )
    material = dict(manifest)
    material.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(material)
    files["BUNDLE-MANIFEST.json"] = canonical_json_bytes(manifest)
    tampered = tmp_path / "asset-index-tampered.tar.gz"
    _write_archive(
        tampered,
        [(member, files[member.name]) for member in members],
    )

    with pytest.raises(BundleVerificationError, match="external asset"):
        verify_bundle(tampered, run_tests=False)


@pytest.mark.parametrize(
    ("name", "member_type"),
    [
        ("../escape", tarfile.REGTYPE),
        ("/absolute", tarfile.REGTYPE),
        (r"windows\\escape", tarfile.REGTYPE),
        ("symlink", tarfile.SYMTYPE),
    ],
)
def test_verifier_rejects_unsafe_archive_members(tmp_path, name, member_type):
    member = tarfile.TarInfo(name)
    member.type = member_type
    member.linkname = "../../outside" if member_type == tarfile.SYMTYPE else ""
    member.uid = member.gid = member.mtime = 0
    member.uname = member.gname = ""
    member.mode = 0o644
    malicious = tmp_path / "malicious.tar.gz"
    _write_archive(malicious, [(member, b"unsafe")])

    with pytest.raises(
        BundleVerificationError,
        match="unsafe|regular|portable|absolute|traversal",
    ):
        verify_bundle(malicious, run_tests=False)


def test_verifier_rejects_duplicate_archive_members(tmp_path):
    first = tarfile.TarInfo("duplicate")
    second = tarfile.TarInfo("duplicate")
    for member in (first, second):
        member.uid = member.gid = member.mtime = 0
        member.uname = member.gname = ""
        member.mode = 0o644
    malicious = tmp_path / "duplicate.tar.gz"
    _write_archive(malicious, [(first, b"one"), (second, b"two")])

    with pytest.raises(BundleVerificationError, match="duplicate"):
        verify_bundle(malicious, run_tests=False)


def test_production_packaging_requires_clean_source_tree(tmp_path):
    freeze, manifest = _fixture_contracts(tmp_path)
    probe = REPO_ROOT / ".task-11-dirty-probe"
    probe.write_text("dirty\n")
    try:
        with pytest.raises(ValueError, match="clean source tree"):
            package_run(
                tmp_path / "bundle.tar.gz",
                source_root=REPO_ROOT,
                freeze_path=freeze,
                run_manifest_path=manifest,
            )
    finally:
        probe.unlink(missing_ok=True)


def test_launchable_bundle_embeds_and_independently_verifies_asset_receipt(
    tmp_path,
    monkeypatch,
):
    import scripts.package_relational_run as packager

    freeze, receipt, manifest, _data_root = stage_launchable_relational_assets(
        tmp_path / "assets"
    )
    freeze_path = tmp_path / "freeze.json"
    manifest_path = tmp_path / "run-manifest.json"
    write_freeze_manifest(freeze_path, freeze)
    write_run_manifest(manifest_path, manifest)
    monkeypatch.setattr(
        packager,
        "_source_revision",
        lambda *_args, **_kwargs: {
            "record_type": "relational_source_revision",
            "schema_version": 1,
            "git_revision": freeze.source_provenance.git_revision,
            "git_tree": "2" * 40,
            "clean_tree": True,
        },
    )
    monkeypatch.setattr(
        packager,
        "tracked_source_tree_sha256",
        lambda _root: freeze.source_provenance.source_tree_sha256,
    )

    bundle = package_run(
        tmp_path / "launchable.tar.gz",
        source_root=REPO_ROOT,
        freeze_path=freeze_path,
        run_manifest_path=manifest_path,
        require_clean=False,
    )
    report = verify_bundle(bundle, run_tests=False)
    _members, files = _archive_files(bundle)

    assert json.loads(files["contracts/asset-receipt.json"]) == receipt.to_dict()
    assert len(report["external_assets"]) == 50
    assert all(
        set(asset)
        == {
            "kind",
            "path",
            "commitment_sha256",
            "sha256",
            "bytes",
            "build_manifest_sha256",
        }
        for asset in report["external_assets"]
    )


def test_launchable_bundle_must_match_frozen_source_revision(tmp_path):
    freeze, _receipt, manifest, _data_root = (
        stage_launchable_relational_assets(tmp_path / "assets")
    )
    freeze_path = tmp_path / "freeze.json"
    manifest_path = tmp_path / "run-manifest.json"
    write_freeze_manifest(freeze_path, freeze)
    write_run_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="source revision"):
        package_run(
            tmp_path / "bundle.tar.gz",
            source_root=REPO_ROOT,
            freeze_path=freeze_path,
            run_manifest_path=manifest_path,
            require_clean=False,
        )
