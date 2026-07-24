from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
P5_PROFILE = "cluster/profiles/aws-p5.48xlarge-v3.json"
P6_PROFILE = "cluster/profiles/aws-p6-b300.48xlarge-v3.json"
PROFILES = (P5_PROFILE, P6_PROFILE)
EXCLUDED_FIXTURES = {
    ".cache/compiler.bin": b"cache payload",
    "credentials/operator.txt": b"credential payload",
    "data/corpus.bin": b"corpus payload",
    "outputs/seed-0/checkpoint.pt": b"output payload",
    "images/mutable-reference.txt": b"example.invalid/team/image:latest\n",
}


def _load_script(name: str):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PACKAGER = _load_script("package_aws_gpu_handoff")
VERIFIER = _load_script("verify_aws_gpu_v3_release")


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(0o644)


def _commit(root: Path, message: str) -> None:
    _git(root, "add", "-f", "--all")
    _git(root, "commit", "-qm", message)


def _source_repo(tmp_path: Path, profile: str, *, name: str = "source") -> Path:
    root = tmp_path / name
    root.mkdir()
    for relative in sorted(PACKAGER.required_members(profile)):
        source = REPO_ROOT / relative
        assert source.is_file() and not source.is_symlink(), relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    other_profile = next(candidate for candidate in PROFILES if candidate != profile)
    destination = root / other_profile
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / other_profile, destination)
    for relative, data in EXCLUDED_FIXTURES.items():
        _write(root / relative, data)

    _git(root, "init", "-q")
    _git(root, "config", "user.name", "AWS GPU Package Test")
    _git(root, "config", "user.email", "package@example.invalid")
    _git(root, "config", "commit.gpgsign", "false")
    _commit(root, "complete AWS GPU v3 package fixture")
    return root


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build(source: Path, out: Path, profile: str):
    return PACKAGER.build_handoff(
        source_root=source,
        out_dir=out,
        profile=profile,
        apply=True,
    )


@pytest.mark.parametrize("profile", PROFILES)
def test_profile_selected_package_is_deterministic_and_exact(
    tmp_path: Path,
    profile: str,
) -> None:
    source = _source_repo(tmp_path, profile)

    first = _build(source, tmp_path / "out-a", profile)
    second = _build(source, tmp_path / "out-b", profile)

    assert first.published is True
    assert first.archive.name == second.archive.name
    assert first.archive.read_bytes() == second.archive.read_bytes()
    assert first.sha256_file.read_bytes() == second.sha256_file.read_bytes()
    assert first.release.read_bytes() == second.release.read_bytes()
    assert first.sha256 == _sha256(first.archive.read_bytes())

    expected_source = PACKAGER.required_members(profile)
    with zipfile.ZipFile(first.archive) as archive:
        infos = archive.infolist()
        regular_names = {
            info.filename for info in infos if not info.is_dir()
        }
        assert regular_names == expected_source | {
            PACKAGER.METADATA_PATH,
            PACKAGER.SUMS_PATH,
        }
        assert (set(PROFILES) - {profile}).isdisjoint(regular_names)
        assert set(EXCLUDED_FIXTURES).isdisjoint(regular_names)
        assert {
            name
            for name in regular_names
            if name.startswith("configs/360m-v3/")
        } == set(PACKAGER.EXPECTED_CONFIGS)
        assert {
            "scripts/run_train.py",
            "train/trainer.py",
            "evals/confirmatory/runner.py",
            PACKAGER.AMENDMENT_PATH,
            PACKAGER.SELECTION_SCHEMA_PATH,
            PACKAGER.CONTAINER_LOCK_PATH,
        } <= regular_names
        for info in infos:
            mode = info.external_attr >> 16
            assert stat.S_IFMT(mode) in {stat.S_IFDIR, stat.S_IFREG}
            assert info.date_time == PACKAGER._core.NORMALIZED_TIME

        metadata = json.loads(archive.read(PACKAGER.METADATA_PATH))
        sums_bytes = archive.read(PACKAGER.SUMS_PATH)
        sums = {
            name: digest
            for digest, name in (
                row.split("  ", 1)
                for row in sums_bytes.decode("ascii").splitlines()
            )
        }
        assert metadata["selected_profile_id"] == PACKAGER.PROFILE_SPECS[
            profile
        ]["provider"]
        assert metadata["profile"] == {
            "path": profile,
            "sha256": PACKAGER.PROFILE_SPECS[profile]["sha256"],
        }
        assert metadata["seed_assignment"]["seeds"] == list(range(10))
        assert set(metadata["contract_locks"]) == set(PACKAGER.CONTRACT_GROUPS)
        assert sums == {
            name: _sha256(archive.read(name))
            for name in regular_names - {PACKAGER.SUMS_PATH}
        }

    receipt = json.loads(first.release.read_text(encoding="utf-8"))
    assert receipt["package_format_version"] == "aws-gpu-v3"
    assert receipt["selected_profile_id"] == metadata["selected_profile_id"]
    assert receipt["source"] == {
        "commit": _git(source, "rev-parse", "HEAD"),
        "dirty": False,
        "tree": _git(source, "rev-parse", "HEAD^{tree}"),
    }
    assert receipt["members_sha256"] == _sha256(sums_bytes)
    assert receipt["preregistration_sha256"] == metadata["preregistration"][
        "sha256"
    ]
    assert receipt["hardware_amendment_sha256"] == metadata[
        "hardware_amendment"
    ]["sha256"]
    assert receipt["container_base_lock_sha256"] == metadata[
        "container_base_lock"
    ]["sha256"]

    from msctl.contracts import load_release

    loaded = load_release(first.release)
    assert loaded.provider == receipt["provider"]
    assert loaded.archive_sha256 == receipt["archive"]["sha256"]


@pytest.mark.parametrize("profile", PROFILES)
def test_published_v3_package_is_bootstrap_compatible(
    tmp_path: Path,
    profile: str,
) -> None:
    from cluster.aws.p5.bootstrap import verify_bootstrap_artifacts
    from cluster.aws.p5.profile import load_aws_gpu_profile

    source = _source_repo(tmp_path, profile)
    artifacts = _build(source, tmp_path / "out", profile)
    dataset_receipt = tmp_path / "dataset-receipt.json"
    dataset_receipt.write_bytes(
        PACKAGER._core._canonical_pretty({"build_id": "a" * 64})
    )
    cohort = source / PACKAGER.COHORT_PATH

    verified = verify_bootstrap_artifacts(
        release_archive=artifacts.archive,
        release_sha256=_sha256(artifacts.archive.read_bytes()),
        release_receipt=artifacts.release,
        release_receipt_sha256=_sha256(artifacts.release.read_bytes()),
        dataset_receipt=dataset_receipt,
        dataset_receipt_sha256=_sha256(dataset_receipt.read_bytes()),
        cohort_assignment=cohort,
        cohort_assignment_sha256=_sha256(cohort.read_bytes()),
        code_commit=_git(source, "rev-parse", "HEAD"),
        profile=load_aws_gpu_profile(source / profile),
    )

    assert verified.provider == PACKAGER.PROFILE_SPECS[profile]["provider"]
    assert verified.assigned_seeds == tuple(PACKAGER.SEEDS)


def test_cli_is_dry_run_by_default_and_profile_is_closed(tmp_path: Path) -> None:
    source = _source_repo(tmp_path, P5_PROFILE)
    out = tmp_path / "out"
    script = source / "scripts" / "package_aws_gpu_handoff.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source-root",
            str(source),
            "--out-dir",
            str(out),
            "--profile",
            P5_PROFILE,
        ],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout
    assert completed.stderr == ""
    assert len(completed.stdout.splitlines()) == 1
    report = json.loads(completed.stdout)
    assert report["ok"] is True
    assert report["dry_run"] is True
    assert report["published"] is False
    assert not out.exists()

    rejected = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source-root",
            str(source),
            "--profile",
            "cluster/profiles/aws-p4-v3.json",
        ],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert rejected.stderr == ""
    assert json.loads(rejected.stdout)["error"]["code"] == "CLI_USAGE"


def test_msctl_accepts_legacy_v2_and_rejects_unknown_v3_format(
    tmp_path: Path,
) -> None:
    from msctl.contracts import load_release
    from msctl.errors import MsctlError
    from tests.test_package_aws_p5_handoff import _minimal_repo

    legacy_source = _minimal_repo(tmp_path)
    legacy = PACKAGER._core.build_handoff(
        source_root=legacy_source,
        out_dir=tmp_path / "legacy-out",
        apply=True,
    )
    assert load_release(legacy.release).provider == "aws-p5.48xlarge"

    source = _source_repo(tmp_path, P6_PROFILE, name="v3-source")
    artifacts = _build(source, tmp_path / "v3-out", P6_PROFILE)
    receipt = json.loads(artifacts.release.read_text(encoding="utf-8"))
    receipt["package_format_version"] = "unknown-aws-package"
    invalid = artifacts.release.with_name("INVALID-RELEASE.json")
    invalid.write_bytes(PACKAGER._core._canonical_pretty(receipt))

    with pytest.raises(MsctlError, match="package format is unsupported"):
        load_release(invalid)


def test_packager_rejects_symlinks_and_mutable_required_images(
    tmp_path: Path,
) -> None:
    linked = _source_repo(tmp_path, P5_PROFILE, name="linked")
    os.symlink("checkpoint.pt", linked / "outputs" / "linked-checkpoint")
    _commit(linked, "add forbidden tracked symlink")
    with pytest.raises(PACKAGER.PackageError, match="symlink"):
        _build(linked, tmp_path / "linked-out", P5_PROFILE)
    assert not (tmp_path / "linked-out").exists()

    mutable = _source_repo(tmp_path, P5_PROFILE, name="mutable")
    start = mutable / "AWS-GPU-V3-START.md"
    start.write_text(
        start.read_text(encoding="utf-8")
        + "\nForbidden example: registry.example/team/image:latest\n",
        encoding="utf-8",
    )
    _commit(mutable, "add mutable runtime image reference")
    with pytest.raises(PACKAGER.PackageError, match="mutable image tag"):
        _build(mutable, tmp_path / "mutable-out", P5_PROFILE)
    assert not (tmp_path / "mutable-out").exists()

    reformatted = _source_repo(tmp_path, P5_PROFILE, name="reformatted")
    config = reformatted / "configs" / "360m-v3" / "dense-s0.yaml"
    config.write_text(
        config.read_text(encoding="utf-8") + "# semantic no-op\n",
        encoding="utf-8",
    )
    _commit(reformatted, "reformat one frozen config")
    with pytest.raises(PACKAGER.PackageError, match="config byte inventory"):
        _build(reformatted, tmp_path / "reformatted-out", P5_PROFILE)
    assert not (tmp_path / "reformatted-out").exists()


def _refresh_external_archive_binding(artifacts) -> None:
    content = artifacts.archive.read_bytes()
    digest = _sha256(content)
    receipt = json.loads(artifacts.release.read_text(encoding="utf-8"))
    receipt["archive"]["sha256"] = digest
    receipt["archive"]["bytes"] = len(content)
    artifacts.release.write_bytes(PACKAGER._core._canonical_pretty(receipt))
    artifacts.sha256_file.write_text(
        f"{digest}  {artifacts.archive.name}\n",
        encoding="ascii",
    )


def test_verifier_rejects_cross_profile_mixed_and_stale_releases(
    tmp_path: Path,
) -> None:
    source = _source_repo(tmp_path, P5_PROFILE)
    artifacts = _build(source, tmp_path / "out", P5_PROFILE)

    report = VERIFIER.verify_release(
        release=artifacts.release,
        profile=P5_PROFILE,
        source_root=source,
    )
    assert report["ok"] is True
    assert report["mixed_profiles"] is False

    with pytest.raises(VERIFIER.VerificationError, match="different.*profile"):
        VERIFIER.verify_release(
            release=artifacts.release,
            profile=P6_PROFILE,
            source_root=source,
        )

    with zipfile.ZipFile(artifacts.archive, mode="a") as archive:
        info = PACKAGER._core._zip_info(
            P6_PROFILE,
            mode=0o644,
            directory=False,
        )
        archive.writestr(
            info,
            (source / P6_PROFILE).read_bytes(),
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )
    _refresh_external_archive_binding(artifacts)
    with pytest.raises(VERIFIER.VerificationError, match="allowlist|mix"):
        VERIFIER.verify_release(
            release=artifacts.release,
            profile=P5_PROFILE,
            source_root=source,
        )

    stale_source = _source_repo(tmp_path, P5_PROFILE, name="stale-source")
    stale = _build(stale_source, tmp_path / "stale-out", P5_PROFILE)
    excluded = stale_source / "outputs" / "seed-0" / "checkpoint.pt"
    excluded.write_bytes(b"new excluded output")
    _commit(stale_source, "advance source after release")
    with pytest.raises(VERIFIER.VerificationError, match="stale"):
        VERIFIER.verify_release(
            release=stale.release,
            profile=P5_PROFILE,
            source_root=stale_source,
        )


def test_required_member_paths_are_portable_and_exist() -> None:
    for profile in PROFILES:
        members = PACKAGER.required_members(profile)
        assert profile in members
        assert (set(PROFILES) - {profile}).isdisjoint(members)
        for relative in members:
            path = PurePosixPath(relative)
            assert not path.is_absolute()
            assert "\\" not in relative
            assert ".." not in path.parts
            assert (REPO_ROOT / relative).is_file()
