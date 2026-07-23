from __future__ import annotations

import hashlib
import json
import os
import zipfile
from copy import deepcopy
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
    DATASET_POINTER_PATH,
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


def _canonical_pretty_json(value: object) -> bytes:
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


def _rewrite_real_package(
    packaged,
    *,
    member_values: dict[str, object] | None = None,
    mutate_metadata=None,
) -> None:
    member_values = member_values or {}
    with zipfile.ZipFile(packaged.archive, "r") as source:
        infos = source.infolist()
        payloads = {
            info.filename: source.read(info)
            for info in infos
            if not info.is_dir()
        }

    metadata = json.loads(payloads["RELEASE-METADATA.json"])
    binding_names = {
        COHORT_ASSIGNMENT_PATH: "cohort_assignment",
        DATASET_POINTER_PATH: "dataset_pointer",
        PROFILE_PATH: "profile",
    }
    changed_bindings = set()
    for relative, value in member_values.items():
        payload = _canonical_pretty_json(value)
        payloads[relative] = payload
        row = next(
            item for item in metadata["members"] if item["path"] == relative
        )
        row["bytes"] = len(payload)
        row["sha256"] = hashlib.sha256(payload).hexdigest()
        binding_name = binding_names.get(relative)
        if binding_name is not None:
            metadata[binding_name]["sha256"] = row["sha256"]
            changed_bindings.add(binding_name)
    if mutate_metadata is not None:
        mutate_metadata(metadata)
    payloads["RELEASE-METADATA.json"] = _canonical_pretty_json(metadata)
    checksum_paths = sorted(set(payloads) - {"SHA256SUMS"})
    payloads["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(payloads[path]).hexdigest()}  {path}\n"
        for path in checksum_paths
    ).encode("ascii")

    rewritten = packaged.archive.with_name("rewritten.zip")
    with zipfile.ZipFile(rewritten, "w") as destination:
        for info in infos:
            destination.writestr(
                info,
                b"" if info.is_dir() else payloads[info.filename],
            )
    os.replace(rewritten, packaged.archive)

    release = json.loads(packaged.release.read_text())
    release["archive"]["bytes"] = packaged.archive.stat().st_size
    release["archive"]["sha256"] = _sha256(packaged.archive)
    release["members_sha256"] = hashlib.sha256(
        payloads["SHA256SUMS"]
    ).hexdigest()
    for binding_name in changed_bindings:
        release[binding_name] = deepcopy(metadata[binding_name])
        release[f"{binding_name}_sha256"] = metadata[binding_name]["sha256"]
    _write_canonical_json(packaged.release, release)
    packaged.archive.with_name(packaged.archive.name + ".sha256").write_text(
        f"{_sha256(packaged.archive)}  {packaged.archive.name}\n",
        encoding="ascii",
    )


def _pointer_mutation(value: dict[str, object], mutation: str) -> None:
    if mutation == "missing":
        del value["dataset_id"]
    elif mutation == "extra":
        value["unexpected"] = "forbidden"
    elif mutation == "schema-float":
        value["schema_version"] = 1.0
    else:
        value[mutation] = "d" * 64


_POINTER_MUTATIONS = (
    "missing",
    "extra",
    "schema-float",
    "dataset_receipt_sha256",
    "dataset_build_id",
    "build_id",
    "ordered_stream_sha256",
)


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


@pytest.mark.parametrize("mutation", _POINTER_MUTATIONS)
def test_real_v3_loader_rejects_noncanonical_dataset_pointer(
    tmp_path,
    mutation,
):
    source, packaged = _build_real_package(tmp_path)
    pointer = json.loads((source / DATASET_POINTER_PATH).read_text())
    _pointer_mutation(pointer, mutation)
    _rewrite_real_package(
        packaged,
        member_values={DATASET_POINTER_PATH: pointer},
    )

    with pytest.raises(MsctlError, match="dataset|pointer|contract"):
        load_release(packaged.release)


@pytest.mark.parametrize("mutation", _POINTER_MUTATIONS)
def test_real_v3_bootstrap_rejects_noncanonical_dataset_pointer(
    tmp_path,
    mutation,
):
    source, packaged = _build_real_package(tmp_path)
    pointer = json.loads((source / DATASET_POINTER_PATH).read_text())
    _pointer_mutation(pointer, mutation)
    _rewrite_real_package(
        packaged,
        member_values={DATASET_POINTER_PATH: pointer},
    )
    dataset_receipt = _write_canonical_json(
        tmp_path / "dataset" / "receipt.json",
        {
            "build_id": "b" * 64,
            "ordered_stream_sha256": "c" * 64,
        },
    )
    cohort_assignment = source / COHORT_ASSIGNMENT_PATH
    release = json.loads(packaged.release.read_text())

    with pytest.raises(BootstrapError, match="dataset|pointer|contract"):
        verify_bootstrap_artifacts(
            release_archive=packaged.archive,
            release_sha256=_sha256(packaged.archive),
            release_receipt=packaged.release,
            release_receipt_sha256=_sha256(packaged.release),
            dataset_receipt=dataset_receipt,
            dataset_receipt_sha256=_sha256(dataset_receipt),
            cohort_assignment=cohort_assignment,
            cohort_assignment_sha256=_sha256(cohort_assignment),
            code_commit=release["source"]["commit"],
        )


@pytest.mark.parametrize("numeric_alias", [False, 0.0])
def test_real_v3_loader_rejects_outer_seed_numeric_alias(
    tmp_path,
    numeric_alias,
):
    _source, packaged = _build_real_package(tmp_path)
    release = json.loads(packaged.release.read_text())
    release["seed_assignment"]["seeds"][0] = numeric_alias
    _write_canonical_json(packaged.release, release)

    with pytest.raises(MsctlError, match="seed|metadata|bind"):
        load_release(packaged.release)


@pytest.mark.parametrize("numeric_alias", [False, 0.0])
def test_real_v3_bootstrap_rejects_internal_seed_numeric_alias(
    tmp_path,
    numeric_alias,
):
    source, packaged = _build_real_package(tmp_path)

    def mutate(metadata):
        metadata["seed_assignment"]["seeds"][0] = numeric_alias

    _rewrite_real_package(packaged, mutate_metadata=mutate)
    dataset_receipt = _write_canonical_json(
        tmp_path / "dataset" / "receipt.json",
        {
            "build_id": "b" * 64,
            "ordered_stream_sha256": "c" * 64,
        },
    )
    cohort_assignment = source / COHORT_ASSIGNMENT_PATH
    release = json.loads(packaged.release.read_text())

    with pytest.raises(BootstrapError, match="seed|metadata|identity"):
        verify_bootstrap_artifacts(
            release_archive=packaged.archive,
            release_sha256=_sha256(packaged.archive),
            release_receipt=packaged.release,
            release_receipt_sha256=_sha256(packaged.release),
            dataset_receipt=dataset_receipt,
            dataset_receipt_sha256=_sha256(dataset_receipt),
            cohort_assignment=cohort_assignment,
            cohort_assignment_sha256=_sha256(cohort_assignment),
            code_commit=release["source"]["commit"],
        )


@pytest.mark.parametrize(
    "field",
    [
        "model_parameters",
        "optimizer_steps",
        "raw_target_tokens",
        "targets_per_update",
    ],
)
def test_real_v3_loader_rejects_integer_valued_assignment_float(
    tmp_path,
    field,
):
    source, packaged = _build_real_package(tmp_path)
    assignment = json.loads((source / COHORT_ASSIGNMENT_PATH).read_text())
    assignment[field] = float(assignment[field])
    _rewrite_real_package(
        packaged,
        member_values={COHORT_ASSIGNMENT_PATH: assignment},
    )

    with pytest.raises(MsctlError, match="assignment|identity"):
        load_release(packaged.release)
