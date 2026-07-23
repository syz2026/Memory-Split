from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from cluster.aws.p5.bootstrap import (
    BootstrapError,
    BootstrapEvidence,
    PreparedRelease,
    build_bootstrap_receipt,
    extract_verified_release,
    verify_bootstrap_artifacts,
)
from cluster.aws.p5.launch_seed_pair import load_launch_plan, render_plan
from cluster.aws.p5.profile import (
    load_aws_p5_profile,
    validate_runtime_environment,
)
from msctl.aws_contracts import (
    COHORT_ASSIGNMENT_PATH,
    CONFIG_ROOT,
    PROFILE_PATH,
)
from msctl.aws_launch_manifest import build_launcher_manifest
from msctl.contracts import load_release
from msctl.errors import MsctlError
from tests.test_aws_p5_launcher import (
    BOOT_ID,
    H100_NAMES,
    SAFE_ENVIRONMENT,
    _launcher_fixture,
)
from tests.test_package_aws_p5_handoff import (
    _build,
    _git,
    _load_module,
    _minimal_repo,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    return path


def _build_real_package(tmp_path: Path):
    source = _minimal_repo(tmp_path)
    packaged = _build(_load_module(), source, tmp_path / "published")
    return source, packaged


def test_real_v3_package_round_trips_to_exact_seed_zero_pair(tmp_path):
    source, packaged = _build_real_package(tmp_path)

    release = load_release(packaged.release)

    assert release.package_format_version == 2
    assert release.source_commit == _git(source, "rev-parse", "HEAD")
    assert release.source_tree == _git(source, "rev-parse", "HEAD^{tree}")

    launcher = _launcher_fixture(tmp_path / "runtime", seed=0)
    scratch_root = launcher["scratch_root"]
    dataset_receipt = launcher["corpus_path"]
    cohort_assignment = source / COHORT_ASSIGNMENT_PATH
    artifacts = verify_bootstrap_artifacts(
        release_archive=packaged.archive,
        release_sha256=release.archive_sha256,
        release_receipt=packaged.release,
        release_receipt_sha256=release.receipt_sha256,
        dataset_receipt=dataset_receipt,
        dataset_receipt_sha256=_sha256(dataset_receipt),
        cohort_assignment=cohort_assignment,
        cohort_assignment_sha256=_sha256(cohort_assignment),
        code_commit=release.source_commit,
    )

    assert artifacts.corpus_build_id == launcher["corpus"]["build_id"]
    assert (
        artifacts.corpus_ordered_stream_sha256
        == launcher["corpus"]["ordered_stream_sha256"]
    )

    prepared = extract_verified_release(
        release_archive=packaged.archive,
        artifacts=artifacts,
        scratch_root=scratch_root,
        owner_uid=os.getuid(),
        owner_gid=os.getgid(),
    )
    profile = load_aws_p5_profile(prepared.root / PROFILE_PATH)
    runtime = validate_runtime_environment(profile, SAFE_ENVIRONMENT)
    evidence = BootstrapEvidence(
        instance_id="i-0123456789abcdef0",
        instance_type="p5.48xlarge",
        ami_id=runtime.ami_id,
        boot_id=BOOT_ID,
        account_id="123456789012",
        role_name="MemorySplitP5Role",
        role_arn=(
            "arn:aws:sts::123456789012:"
            "assumed-role/MemorySplitP5Role/i-0123456789abcdef0"
        ),
        gpu_names=H100_NAMES,
        fabric_manager_active=True,
        instance_store_devices=tuple(
            f"/dev/nvme{index}n1" for index in range(8)
        ),
        container_image=runtime.container_image,
    )
    bootstrap_receipt = build_bootstrap_receipt(
        profile=profile,
        runtime=runtime,
        evidence=evidence,
        artifacts=artifacts,
        prepared_release=PreparedRelease(
            root=(
                Path(profile.scratch_root)
                / "releases"
                / artifacts.release_sha256
            ),
            members_sha256=artifacts.release_members_sha256,
        ),
        durable_upload_verified=True,
    )
    bootstrap_path = _write_canonical_json(
        scratch_root / "staging" / "roundtrip-bootstrap.json",
        bootstrap_receipt,
    )
    runs = [
        {
            "arm": arm,
            "config": f"{CONFIG_ROOT}/{arm}-s0.yaml",
            "config_sha256": _sha256(
                prepared.root / CONFIG_ROOT / f"{arm}-s0.yaml"
            ),
        }
        for arm in ("dense", "split90")
    ]
    manifest_path = scratch_root / "staging" / "roundtrip-launch.json"
    manifest = build_launcher_manifest(
        out=manifest_path,
        scratch_root=scratch_root,
        seed=0,
        profile_sha256=profile.sha256,
        release_sha256=release.archive_sha256,
        release_members_sha256=release.members_sha256,
        cohort_assignment_sha256=_sha256(cohort_assignment),
        code_commit=release.source_commit,
        bootstrap_receipt=bootstrap_path,
        corpus_receipt=dataset_receipt,
        runs=runs,
    )

    plan = load_launch_plan(
        seed=0,
        manifest_path=manifest_path,
        profile_path=prepared.root / PROFILE_PATH,
        repo_root=prepared.root,
        scratch_root=scratch_root,
        environment=SAFE_ENVIRONMENT,
        observed_instance_type="p5.48xlarge",
        observed_instance_id=evidence.instance_id,
        observed_boot_id=evidence.boot_id,
        gpu_names=H100_NAMES,
        port_available=lambda _port: True,
        semantic_corpus_verifier=lambda _root: launcher["corpus"],
        enforce_profile_scratch=False,
    )
    rendered = render_plan(plan)

    assert manifest["cohort_id"] == "memorysplit-confirmatory-v3-360m-n10-aws"
    assert [command["arm"] for command in rendered["commands"]] == [
        "dense",
        "split90",
    ]
    assert all(
        "--nproc_per_node=4" in command["argv"]
        for command in rendered["commands"]
    )


def test_real_v3_loader_rejects_format_source_and_dataset_mutations(tmp_path):
    _source, packaged = _build_real_package(tmp_path)
    original = json.loads(packaged.release.read_text())
    mutations = []

    format_one = json.loads(packaged.release.read_text())
    format_one["package_format_version"] = 1
    mutations.append(("package format", format_one))

    missing_tree = json.loads(packaged.release.read_text())
    del missing_tree["source"]["tree"]
    mutations.append(("source", missing_tree))

    extra_source = json.loads(packaged.release.read_text())
    extra_source["source"]["unexpected"] = "forbidden"
    mutations.append(("source", extra_source))

    concrete_dataset = json.loads(packaged.release.read_text())
    concrete_dataset["dataset_receipt_sha256"] = "d" * 64
    mutations.append(("field|dataset", concrete_dataset))

    for message, value in mutations:
        _write_canonical_json(packaged.release, value)
        with pytest.raises(MsctlError, match=message):
            load_release(packaged.release)
    _write_canonical_json(packaged.release, original)


def test_real_v3_bootstrap_rejects_release_identity_mutations(tmp_path):
    source, packaged = _build_real_package(tmp_path)
    dataset_receipt = _write_canonical_json(
        tmp_path / "dataset" / "receipt.json",
        {
            "build_id": "b" * 64,
            "ordered_stream_sha256": "c" * 64,
        },
    )
    cohort_assignment = source / COHORT_ASSIGNMENT_PATH
    original = json.loads(packaged.release.read_text())
    cases = []

    format_one = json.loads(packaged.release.read_text())
    format_one["package_format_version"] = 1
    cases.append(("package format", format_one))

    missing_tree = json.loads(packaged.release.read_text())
    del missing_tree["source"]["tree"]
    cases.append(("source|tree", missing_tree))

    wrong_tree = json.loads(packaged.release.read_text())
    wrong_tree["source"]["tree"] = "not-a-git-tree"
    cases.append(("tree", wrong_tree))

    concrete_dataset = json.loads(packaged.release.read_text())
    concrete_dataset["dataset_receipt_sha256"] = _sha256(dataset_receipt)
    cases.append(("dataset|field", concrete_dataset))

    for message, value in cases:
        _write_canonical_json(packaged.release, value)
        with pytest.raises(BootstrapError, match=message):
            verify_bootstrap_artifacts(
                release_archive=packaged.archive,
                release_sha256=_sha256(packaged.archive),
                release_receipt=packaged.release,
                release_receipt_sha256=_sha256(packaged.release),
                dataset_receipt=dataset_receipt,
                dataset_receipt_sha256=_sha256(dataset_receipt),
                cohort_assignment=cohort_assignment,
                cohort_assignment_sha256=_sha256(cohort_assignment),
                code_commit=original["source"]["commit"],
            )
    _write_canonical_json(packaged.release, original)


def test_real_v3_bootstrap_rejects_assignment_and_dataset_mutations(tmp_path):
    source, packaged = _build_real_package(tmp_path)
    cohort_assignment = source / COHORT_ASSIGNMENT_PATH
    valid_dataset = {
        "build_id": "b" * 64,
        "ordered_stream_sha256": "c" * 64,
    }
    dataset_receipt = _write_canonical_json(
        tmp_path / "dataset" / "receipt.json",
        valid_dataset,
    )
    receipt = json.loads(packaged.release.read_text())
    arguments = {
        "release_archive": packaged.archive,
        "release_sha256": _sha256(packaged.archive),
        "release_receipt": packaged.release,
        "release_receipt_sha256": _sha256(packaged.release),
        "dataset_receipt": dataset_receipt,
        "dataset_receipt_sha256": _sha256(dataset_receipt),
        "cohort_assignment": cohort_assignment,
        "cohort_assignment_sha256": _sha256(cohort_assignment),
        "code_commit": receipt["source"]["commit"],
    }

    wrong_assignment = _write_canonical_json(
        tmp_path / "wrong-assignment.json",
        {"cohort_id": "wrong"},
    )
    with pytest.raises(BootstrapError, match="cohort|assignment"):
        verify_bootstrap_artifacts(
            **{
                **arguments,
                "cohort_assignment": wrong_assignment,
                "cohort_assignment_sha256": _sha256(wrong_assignment),
            }
        )

    with pytest.raises(BootstrapError, match="dataset.*SHA-256"):
        verify_bootstrap_artifacts(
            **{**arguments, "dataset_receipt_sha256": "0" * 64}
        )

    for field in ("build_id", "ordered_stream_sha256"):
        malformed = dict(valid_dataset)
        malformed[field] = "not-a-sha256"
        _write_canonical_json(dataset_receipt, malformed)
        with pytest.raises(BootstrapError, match=field.replace("_", " ") + "|SHA-256"):
            verify_bootstrap_artifacts(
                **{
                    **arguments,
                    "dataset_receipt_sha256": _sha256(dataset_receipt),
                }
            )
