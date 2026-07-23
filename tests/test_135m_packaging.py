from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from pathlib import Path

import pytest

from msctl.cohort import EXPECTED_CONFIG_PATHS, ROLES
from scripts.package_135m_slurm_cohort import AWS_BUNDLE_NAME, package_all
from scripts.verify_135m_slurm_releases import (
    ReleaseVerificationError,
    verify_aws_bundle,
    verify_release_set,
)


ROOT = Path(__file__).resolve().parents[1]


def _build(tmp_path: Path, name: str):
    out = tmp_path / name
    archives = package_all(out, source_root=ROOT, require_clean=False)
    return out, archives


def test_five_role_archives_are_deterministic_and_partition_all_cells(tmp_path):
    first_dir, first = _build(tmp_path, "first")
    second_dir, second = _build(tmp_path, "second")
    assert set(first) == set(ROLES)
    assert {
        role: first[role].read_bytes()
        for role in ROLES
    } == {
        role: second[role].read_bytes()
        for role in ROLES
    }
    assert (first_dir / "SHA256SUMS").read_bytes() == (
        second_dir / "SHA256SUMS"
    ).read_bytes()
    assert (first_dir / AWS_BUNDLE_NAME).read_bytes() == (
        second_dir / AWS_BUNDLE_NAME
    ).read_bytes()

    report = verify_release_set(
        list(first.values()),
        source_root=ROOT,
        outer_index=first_dir / "SHA256SUMS",
    )
    assert report["roles"] == list(ROLES)
    assert set(tuple(cell) for cell in report["cells"]) == {
        (arm, seed)
        for seed in range(10)
        for arm in ("dense", "split90")
    }
    bundle_report = verify_aws_bundle(
        first_dir / AWS_BUNDLE_NAME,
        source_root=ROOT,
    )
    assert bundle_report["verified"] is True
    assert bundle_report["roles"] == list(ROLES)


def test_each_archive_contains_only_its_four_configs_and_no_corpus(tmp_path):
    _, archives = _build(tmp_path, "releases")
    all_config_paths = set(EXPECTED_CONFIG_PATHS)
    for role, archive in archives.items():
        with zipfile.ZipFile(archive) as release:
            names = set(release.namelist())
            assignment = json.loads(release.read("assignment.json"))
            owned = set(assignment["config_paths"])
            assert len(owned) == 4
            assert names & all_config_paths == owned
            assert not any(
                name.endswith((".bin", ".pt", ".npy"))
                or name.startswith(("data/", "dataset/", "outputs/"))
                for name in names
            )


def test_aws_bundle_contains_five_role_archives_and_run_instructions(tmp_path):
    release_dir, archives = _build(tmp_path, "releases")
    bundle = release_dir / AWS_BUNDLE_NAME
    with zipfile.ZipFile(bundle) as release:
        names = set(release.namelist())
        assert names == {
            "README-AWS.md",
            "SHA256SUMS",
            "bundle-manifest.json",
            "roles/SHA256SUMS",
            *(f"roles/{role}.zip" for role in ROLES),
        }
        manifest = json.loads(release.read("bundle-manifest.json"))
        assert manifest["bundle_format"] == "memorysplit-135m-n10-all-roles-aws-v1"
        assert [entry["role"] for entry in manifest["role_archives"]] == list(
            ROLES
        )
        assert {
            entry["role"]: entry["sha256"]
            for entry in manifest["role_archives"]
        } == {
            role: hashlib.sha256(path.read_bytes()).hexdigest()
            for role, path in archives.items()
        }
        readme = release.read("README-AWS.md").decode()
        assert "aws s3 cp" in readme
        assert "production corpus" in readme
        assert "does not turn them into AWS-native jobs" in readme


def _copy_with_extra(
    source: Path,
    destination: Path,
    name: str,
    data: bytes,
    mode=0o100644,
):
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED
    ) as new:
        for info in old.infolist():
            new.writestr(info, old.read(info.filename))
        extra = zipfile.ZipInfo(name)
        extra.date_time = (1980, 1, 1, 0, 0, 0)
        extra.create_system = 3
        extra.external_attr = mode << 16
        new.writestr(extra, data)


@pytest.mark.parametrize(
    ("name", "data", "mode", "match"),
    [
        ("extra.txt", b"extra", 0o100644, "extra"),
        ("secrets.txt", b"github_pat_example", 0o100644, "secret"),
        ("escape", b"target", stat.S_IFLNK | 0o777, "symlink"),
    ],
)
def test_verifier_rejects_extra_secret_and_symlink_members(
    tmp_path, name, data, mode, match
):
    _, archives = _build(tmp_path, "releases")
    role = next(iter(ROLES))
    tampered = tmp_path / f"tampered-{name}.zip"
    _copy_with_extra(archives[role], tampered, name, data, mode)
    paths = [tampered if item == archives[role] else item for item in archives.values()]
    with pytest.raises(ReleaseVerificationError, match=match):
        verify_release_set(paths, source_root=ROOT)
