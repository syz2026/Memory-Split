from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from msctl.aws_selection import (
    create_provider_selection,
    load_hardware_amendment,
    validate_provider_selection,
)
from msctl.aws_sealed_evaluation import load_sealed_evaluation_fixture
from msctl.cohort import load_cohort_assignment
from msctl.contracts import load_run_manifest
from msctl.jsonutil import canonical_json
from msctl.profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / "configs" / "hardware-amendment-v3.json"
P5 = ROOT / "cluster" / "profiles" / "aws-p5.48xlarge-v3.json"
P6 = ROOT / "cluster" / "profiles" / "aws-p6-b300.48xlarge-v3.json"
CONTAINER_DIGEST = "sha256:" + "a" * 64
CONTAINER_IMAGE = (
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/memorysplit/aws-gpu@"
    + CONTAINER_DIGEST
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selection(profile_path: Path):
    amendment = load_hardware_amendment(AMENDMENT)
    profile = load_profile(profile_path)
    with tempfile.TemporaryDirectory() as directory:
        ami, ecr, build = _write_provider_evidence(Path(directory))
        capacity = (
            {
                "capacity_reservation_id": "cr-12345678",
                "capacity_block_offering_id": "cbo-12345678",
            }
            if profile.purchase_model == "capacity_block"
            else {
                "capacity_reservation_id": None,
                "capacity_block_offering_id": None,
            }
        )
        value = create_provider_selection(
            profile=profile,
            amendment=amendment,
            region="us-east-1",
            ami_id="ami-0123456789abcdef0",
            container_image=CONTAINER_IMAGE,
            container_digest=CONTAINER_DIGEST,
            ami_evidence=ami,
            ecr_evidence=ecr,
            image_build_receipt=build,
            selected_at="2026-07-24T00:00:00Z",
            **capacity,
        )
    return profile, amendment, validate_provider_selection(
        value,
        amendment=amendment,
        profile=profile,
    )


def _write_provider_evidence(root: Path) -> tuple[Path, Path, Path]:
    from scripts.build_aws_gpu_image import build_context_sha256

    values = (
        (
            "ami.json",
            {
                "schema_version": 1,
                "receipt_type": "memorysplit-aws-ami-describe-v1",
                "region": "us-east-1",
                "image_id": "ami-0123456789abcdef0",
                "owner_id": "099720109477",
                "name": (
                    "Deep Learning Base AMI with Single CUDA "
                    "(Ubuntu 24.04)"
                ),
            },
        ),
        (
            "ecr.json",
            {
                "schema_version": 1,
                "receipt_type": "memorysplit-aws-ecr-describe-v1",
                "region": "us-east-1",
                "registry_id": "123456789012",
                "repository_name": "memorysplit/aws-gpu",
                "image_digest": CONTAINER_DIGEST,
                "image_uri": CONTAINER_IMAGE,
            },
        ),
        (
            "build.json",
            {
                "schema_version": 1,
                "receipt_type": "memorysplit-aws-gpu-image-build-v1",
                "aws_account_id": "123456789012",
                "region": "us-east-1",
                "container_image": CONTAINER_IMAGE,
                "container_digest": CONTAINER_DIGEST,
                "build_context_sha256": build_context_sha256(
                    ROOT / "containers" / "aws-gpu"
                ),
                "dockerfile_sha256": _sha256(
                    ROOT / "containers" / "aws-gpu" / "Dockerfile"
                ),
                "image_lock_sha256": _sha256(
                    ROOT / "containers" / "aws-gpu" / "image.lock.json"
                ),
                "runtime_dependency_lock_sha256": _sha256(
                    ROOT / "containers" / "aws-gpu" / "requirements.lock"
                ),
            },
        ),
    )
    paths: list[Path] = []
    for name, value in values:
        path = root / name
        path.write_bytes(canonical_json(value) + b"\n")
        paths.append(path)
    return paths[0], paths[1], paths[2]


def _sealed_release(root: Path) -> Path:
    from evals.confirmatory.fixtures import positive_fixture
    from msctl.aws_sealed_evaluation import REQUIRED_SEALED_MEMBERS

    release = root / "sealed-evaluation"
    release.mkdir()
    artifacts = positive_fixture().artifacts
    for name in sorted(REQUIRED_SEALED_MEMBERS):
        (release / name).write_bytes(artifacts[name])
    return release


def _sealed_fixture(root: Path) -> Path:
    from evals.confirmatory.fixtures import positive_fixture
    from evals.confirmatory.sealing import SEALED_FIXTURE_MEMBERS

    fixture = root / "sealed-fixture"
    fixture.mkdir()
    artifacts = positive_fixture().artifacts
    for name in SEALED_FIXTURE_MEMBERS:
        (fixture / name).write_bytes(artifacts[name])
    return fixture


def _manifests(
    tmp_path: Path,
    *,
    profile_path: Path = P5,
) -> tuple[object, object, list[Path]]:
    profile, amendment, selection = _selection(profile_path)
    paths: list[Path] = []
    for seed in range(10):
        runs = []
        for arm in ("dense", "split90"):
            relative = f"configs/360m-v3/{arm}-s{seed}.yaml"
            runs.append(
                {
                    "arm": arm,
                    "config": relative,
                    "config_sha256": _sha256(ROOT / relative),
                    "run_id": f"memorysplit-v3-360m-s{seed}-{arm}",
                    "seed": seed,
                }
            )
        value = {
            "schema_version": 3,
            "provider": profile.provider,
            "seed": seed,
            "source_commit": "1" * 40,
            "release_sha256": "2" * 64,
            "dataset_sha256": "3" * 64,
            "cohort_assignment_sha256": amendment.cohort_assignment_sha256,
            "preregistration_sha256": amendment.preregistration_sha256,
            "hardware_amendment_sha256": amendment.sha256,
            "provider_selection_sha256": selection.sha256,
            "profile_sha256": profile.sha256,
            "sealed_fixture_sha256": "4" * 64,
            "estimated_instance_hours": 24.0,
            "estimated_gpu_hours": 192.0,
            "runs": runs,
        }
        path = tmp_path / f"runs-s{seed}.json"
        path.write_bytes(canonical_json(value) + b"\n")
        paths.append(path)
    return profile, selection, paths


def test_canonical_v3_cohort_and_amendment_bind_exact_bytes():
    cohort = load_cohort_assignment(
        ROOT / "configs" / "cohort-assignment-v3.json"
    )
    amendment = load_hardware_amendment(AMENDMENT)

    assert cohort.schema_version == 3
    assert cohort.cohort_id == "memorysplit-confirmatory-v3-360m-n10-aws"
    assert len(cohort.configs) == 20
    assert {config.seed for config in cohort.configs} == set(range(10))
    assert {config.condition for config in cohort.configs} == {
        "dense",
        "split90",
    }
    assert amendment.cohort_assignment_sha256 == cohort.assignment_sha256
    assert amendment.preregistration_sha256 == cohort.preregistration_sha256
    assert [profile.profile_id for profile in amendment.allowed_profiles] == [
        "aws-p5.48xlarge-v3",
        "aws-p6-b300.48xlarge-v3",
    ]


@pytest.mark.parametrize("profile_path", [P5, P6])
def test_provider_selection_and_v3_manifest_roundtrip(profile_path, tmp_path):
    profile, amendment, selection = _selection(profile_path)
    assert selection.provider == profile.provider
    assert selection.profile_sha256 == profile.sha256
    assert selection.amendment_sha256 == amendment.sha256
    assert selection.seeds == tuple(range(10))

    _, _, paths = _manifests(tmp_path, profile_path=profile_path)
    manifest = load_run_manifest(paths[7], repo_root=ROOT)
    assert manifest.schema_version == 3
    assert manifest.provider == profile.provider
    assert manifest.seed == 7
    assert manifest.provider_selection_sha256 == selection.sha256
    assert manifest.profile_sha256 == profile.sha256
    assert manifest.gpu_hours == 192.0


@pytest.mark.parametrize("profile_path", [P5, P6])
def test_v3_manifest_instantiation_binds_selected_profile(
    profile_path,
    tmp_path,
    monkeypatch,
):
    import msctl.operations as operations
    from msctl.aws_selection import write_provider_selection

    profile, _, selection = _selection(profile_path)
    selection_path = tmp_path / "provider-selection.json"
    write_provider_selection(selection_path, selection.value)
    dataset_receipt = tmp_path / "receipt.json"
    dataset_receipt.write_bytes(
        canonical_json(
            {
                "schema_version": 2,
                "ordered_stream_sha256": "8" * 64,
            }
        )
        + b"\n"
    )
    dataset_sha256 = _sha256(dataset_receipt)
    release = SimpleNamespace(
        provider=profile.provider,
        archive_sha256="1" * 64,
        source_commit="2" * 40,
        metadata={
            "profile": {"sha256": profile.sha256},
            "seed_assignment": {
                "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
                "provider": "aws-p5.48xlarge-v3",
                "seeds": list(range(10)),
                "arms": ["dense", "split90"],
            },
        },
    )
    monkeypatch.setattr(operations, "load_release", lambda _path: release)
    monkeypatch.setattr(
        operations,
        "verify_release_member",
        lambda _release, *, local_path, **_kwargs: _sha256(Path(local_path)),
    )
    sealed_fixture = _sealed_fixture(tmp_path)
    evidence = SimpleNamespace(
        files=(
            SimpleNamespace(
                path=dataset_receipt,
                sha256=dataset_sha256,
            ),
        )
    )
    out = tmp_path / "runs-s7.json"
    arguments = {
        "profile": profile,
        "release_path": tmp_path / "RELEASE.json",
        "dataset_receipt": dataset_receipt,
        "seed": 7,
        "out": out,
        "repo_root": ROOT,
        "dataset_verifier": lambda *_args, **_kwargs: evidence,
        "hardware_amendment": AMENDMENT,
        "provider_selection": selection_path,
        "sealed_evaluation_fixture": sealed_fixture,
    }
    rendered = operations.instantiate_run_manifest(
        apply=False,
        **arguments,
    )
    assert rendered["published"] is False
    assert not out.exists()

    published = operations.instantiate_run_manifest(
        apply=True,
        **arguments,
    )
    assert published["published"] is True
    manifest = load_run_manifest(out, repo_root=ROOT)
    assert manifest.schema_version == 3
    assert manifest.seed == 7
    assert manifest.provider == profile.provider
    assert manifest.provider_selection_sha256 == selection.sha256
    assert manifest.profile_sha256 == profile.sha256
    assert manifest.hardware_amendment_sha256 == selection.amendment_sha256
    assert manifest.study_lock_sha256 is None
    assert manifest.sealed_fixture_sha256 == load_sealed_evaluation_fixture(
        sealed_fixture
    ).sha256
    assert manifest.estimated_instance_hours == 24.0
    assert manifest.estimated_gpu_hours == 192.0


def test_provider_selection_cli_is_dry_run_first_and_fails_closed(tmp_path):
    from msctl.aws_selection import (
        load_provider_selection,
        write_provider_selection,
    )
    from msctl.cli import build_parser, dispatch

    profile, amendment, selection = _selection(P5)
    ami_evidence, ecr_evidence, image_build_receipt = (
        _write_provider_evidence(tmp_path)
    )
    out = tmp_path / "provider-selection.json"
    argv = [
        "--profile",
        str(P5),
        "provider",
        "select",
        "--amendment",
        str(AMENDMENT),
        "--region",
        selection.region,
        "--ami-id",
        selection.ami_id,
        "--container-image",
        selection.container_image,
        "--container-digest",
        selection.container_digest,
        "--ami-evidence",
        str(ami_evidence),
        "--ecr-evidence",
        str(ecr_evidence),
        "--image-build-receipt",
        str(image_build_receipt),
        "--selected-at",
        selection.selected_at,
        "--out",
        str(out),
    ]
    dry_run, rendered = dispatch(
        build_parser().parse_args(argv),
        environ={},
    )
    assert dry_run is True
    assert rendered["published"] is False
    assert not out.exists()

    dry_run, published = dispatch(
        build_parser().parse_args([*argv, "--apply"]),
        environ={},
    )
    assert dry_run is False
    assert published["published"] is True
    loaded = load_provider_selection(
        out,
        amendment=amendment,
        profile=profile,
    )
    assert loaded.sha256 == selection.sha256
    with pytest.raises(Exception) as existing:
        dispatch(
            build_parser().parse_args([*argv, "--apply"]),
            environ={},
        )
    assert getattr(existing.value, "code", None) == "PROVIDER_SELECTION_EXISTS"

    p6 = load_profile(P6)
    with pytest.raises(Exception) as cross_profile:
        validate_provider_selection(
            selection.value,
            amendment=amendment,
            profile=p6,
        )
    assert (
        getattr(cross_profile.value, "code", None)
        == "PROVIDER_SELECTION_INVALID"
    )

    unknown = dict(selection.value)
    unknown["unexpected"] = True
    with pytest.raises(Exception) as unknown_field:
        validate_provider_selection(
            unknown,
            amendment=amendment,
            profile=profile,
        )
    assert getattr(unknown_field.value, "code", None) == "PROVIDER_SELECTION_INVALID"

    duplicate = tmp_path / "duplicate-selection.json"
    duplicate.write_text(
        '{"schema_version":3,"schema_version":3}\n',
        encoding="utf-8",
    )
    with pytest.raises(Exception) as duplicate_field:
        load_provider_selection(
            duplicate,
            amendment=amendment,
            profile=profile,
        )
    assert (
        getattr(duplicate_field.value, "code", None)
        == "PROVIDER_SELECTION_INVALID"
    )

    stale = dict(selection.value)
    stale["amendment_sha256"] = "0" * 64
    stale_path = tmp_path / "stale-selection.json"
    write_provider_selection(stale_path, stale)
    with pytest.raises(Exception) as stale_amendment:
        load_provider_selection(
            stale_path,
            amendment=amendment,
            profile=profile,
        )
    assert (
        getattr(stale_amendment.value, "code", None)
        == "PROVIDER_SELECTION_INVALID"
    )


@pytest.mark.parametrize(
    ("instance_count", "expected"),
    [
        (1, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
        (2, [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]),
        (3, [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]),
        (4, [[0, 4, 8], [1, 5, 9], [2, 6], [3, 7]]),
    ],
)
def test_fleet_plan_is_deterministic_round_robin(
    tmp_path,
    instance_count,
    expected,
):
    from msctl.aws_fleet import create_fleet_plan, validate_fleet_plan

    profile, selection, paths = _manifests(tmp_path)
    instance_ids = [
        f"i-{index:017x}" for index in range(1, instance_count + 1)
    ]
    value = create_fleet_plan(
        profile=profile,
        selection=selection,
        manifest_paths=paths,
        instance_ids=reversed(instance_ids),
        repo_root=ROOT,
    )
    plan = validate_fleet_plan(
        value,
        profile=profile,
        selection=selection,
        manifest_paths=paths,
        repo_root=ROOT,
    )

    assert [list(instance.seeds) for instance in plan.instances] == expected
    assert [instance.instance_id for instance in plan.instances] == instance_ids
    assert all(instance.max_active_pairs == 1 for instance in plan.instances)
    assert plan.sha256 == hashlib.sha256(canonical_json(value)).hexdigest()
    for seed in range(10):
        binding = plan.binding_for_seed(seed)
        assert binding.wave == seed // instance_count


def test_fleet_plan_cli_is_dry_run_first_and_exclusive(tmp_path):
    from msctl.aws_fleet import load_fleet_plan
    from msctl.aws_selection import write_provider_selection
    from msctl.cli import build_parser, dispatch

    profile, selection, paths = _manifests(tmp_path)
    selection_path = tmp_path / "provider-selection.json"
    out = tmp_path / "fleet-plan.json"
    instance_ids = [
        "i-00000000000000001",
        "i-00000000000000002",
    ]
    write_provider_selection(selection_path, selection.value)
    argv = [
        "--profile",
        str(P5),
        "--repo-root",
        str(ROOT),
        "fleet",
        "plan",
        "--amendment",
        str(AMENDMENT),
        "--provider-selection",
        str(selection_path),
        *(
            argument
            for path in paths
            for argument in ("--manifest", str(path))
        ),
        *(
            argument
            for instance_id in reversed(instance_ids)
            for argument in ("--instance-id", instance_id)
        ),
        "--out",
        str(out),
    ]

    dry_run, rendered = dispatch(build_parser().parse_args(argv), environ={})
    assert dry_run is True
    assert rendered["published"] is False
    assert not out.exists()

    dry_run, published = dispatch(
        build_parser().parse_args([*argv, "--apply"]),
        environ={},
    )
    assert dry_run is False
    assert published["published"] is True
    plan = load_fleet_plan(
        out,
        profile=profile,
        selection=selection,
        manifest_paths=paths,
        repo_root=ROOT,
    )
    assert [instance.instance_id for instance in plan.instances] == instance_ids
    assert published["plan_sha256"] == plan.sha256

    with pytest.raises(Exception) as existing:
        dispatch(
            build_parser().parse_args([*argv, "--apply"]),
            environ={},
        )
    assert getattr(existing.value, "code", None) == "FLEET_PLAN_EXISTS"


def test_fleet_plan_rejects_duplicate_instances_and_manifest_substitution(
    tmp_path,
):
    from msctl.aws_fleet import create_fleet_plan

    profile, selection, paths = _manifests(tmp_path)
    with pytest.raises(Exception) as duplicate:
        create_fleet_plan(
            profile=profile,
            selection=selection,
            manifest_paths=paths,
            instance_ids=[
                "i-0123456789abcdef0",
                "i-0123456789abcdef0",
            ],
            repo_root=ROOT,
        )
    assert getattr(duplicate.value, "code", None) == "FLEET_PLAN_INVALID"

    value = json.loads(paths[-1].read_text(encoding="utf-8"))
    value["provider_selection_sha256"] = "f" * 64
    paths[-1].write_bytes(canonical_json(value) + b"\n")
    with pytest.raises(Exception) as substituted:
        create_fleet_plan(
            profile=profile,
            selection=selection,
            manifest_paths=paths,
            instance_ids=["i-0123456789abcdef0"],
            repo_root=ROOT,
        )
    assert getattr(substituted.value, "code", None) == "FLEET_PLAN_INVALID"


def test_fleet_plan_rejects_bounds_missing_seeds_and_unknown_fields(tmp_path):
    from msctl.aws_fleet import create_fleet_plan, validate_fleet_plan

    profile, selection, paths = _manifests(tmp_path)
    with pytest.raises(Exception) as too_many:
        create_fleet_plan(
            profile=profile,
            selection=selection,
            manifest_paths=paths,
            instance_ids=[
                f"i-{index:017x}" for index in range(1, 6)
            ],
            repo_root=ROOT,
        )
    assert getattr(too_many.value, "code", None) == "FLEET_PLAN_INVALID"

    with pytest.raises(Exception) as malformed:
        create_fleet_plan(
            profile=profile,
            selection=selection,
            manifest_paths=paths,
            instance_ids=["not-an-instance"],
            repo_root=ROOT,
        )
    assert getattr(malformed.value, "code", None) == "FLEET_PLAN_INVALID"

    with pytest.raises(Exception) as missing_seed:
        create_fleet_plan(
            profile=profile,
            selection=selection,
            manifest_paths=paths[:-1],
            instance_ids=["i-0123456789abcdef0"],
            repo_root=ROOT,
        )
    assert getattr(missing_seed.value, "code", None) == "FLEET_PLAN_INVALID"

    duplicate_seed = tmp_path / "duplicate-seed.json"
    duplicate_seed.write_bytes(paths[0].read_bytes())
    with pytest.raises(Exception) as duplicated_seed:
        create_fleet_plan(
            profile=profile,
            selection=selection,
            manifest_paths=[*paths[:-1], duplicate_seed],
            instance_ids=["i-0123456789abcdef0"],
            repo_root=ROOT,
        )
    assert getattr(duplicated_seed.value, "code", None) == "FLEET_PLAN_INVALID"

    value = create_fleet_plan(
        profile=profile,
        selection=selection,
        manifest_paths=paths,
        instance_ids=["i-0123456789abcdef0"],
        repo_root=ROOT,
    )
    value["unexpected"] = True
    with pytest.raises(Exception) as unknown:
        validate_fleet_plan(
            value,
            profile=profile,
            selection=selection,
        )
    assert getattr(unknown.value, "code", None) == "FLEET_PLAN_INVALID"


def test_later_fleet_wave_requires_local_terminal_advance_receipt(tmp_path):
    from msctl.aws_fleet import (
        create_fleet_advance,
        create_fleet_plan,
        fleet_transition_for_target,
        load_fleet_advance,
        validate_fleet_advance,
        validate_fleet_plan,
        write_fleet_advance,
    )
    from msctl.aws_p5 import AwsP5Backend, V3LifecycleContext

    profile, selection, paths = _manifests(tmp_path)
    instance_id = "i-0123456789abcdef0"
    plan = validate_fleet_plan(
        create_fleet_plan(
            profile=profile,
            selection=selection,
            manifest_paths=paths,
            instance_ids=[instance_id],
            repo_root=ROOT,
        ),
        profile=profile,
        selection=selection,
        manifest_paths=paths,
        repo_root=ROOT,
    )
    target = load_run_manifest(paths[1], repo_root=ROOT)
    source, target_binding = fleet_transition_for_target(
        plan,
        instance_id=instance_id,
        target=target,
    )
    evidence = {
        "training_state_sha256": "1" * 64,
        "training_command_id": "training-command-12345678",
        "training_terminal_receipt_uri": (
            "s3://memorysplit-prod/training/receipts/terminal.json"
        ),
        "checkpoint_receipt_sha256": "8" * 64,
        "checkpoint_receipt_uri": (
            "s3://memorysplit-prod/checkpoints/seed-0/receipts/"
            f"{'8' * 64}.json"
        ),
        "checkpoint_records": [
            {
                "run_id": f"run-{arm}-0",
                "arm": arm,
                "sha256": digest,
                "uri": (
                    "s3://memorysplit-prod/checkpoints/seed-0/"
                    f"{arm}/records/{digest}.json"
                ),
            }
            for arm, digest in (
                ("dense", "2" * 64),
                ("split90", "3" * 64),
            )
        ],
        "aws_bound_tags_sha256": "4" * 64,
        "aws_unbound_tags_sha256": "5" * 64,
    }
    value = create_fleet_advance(
        plan=plan,
        instance_id=instance_id,
        from_binding=source,
        to_binding=target_binding,
        evidence=evidence,
        approval_sha256="6" * 64,
        advanced_at="2026-07-24T02:00:00Z",
    )
    validated = validate_fleet_advance(
        value,
        plan=plan,
        to_binding=target_binding,
    )
    assert validated.from_binding.seed == 0
    assert validated.to_binding.seed == 1
    assert validated.evidence == evidence

    changed = json.loads(canonical_json(value))
    changed["decision"]["unbound"] = False
    with pytest.raises(Exception) as false_unbind:
        validate_fleet_advance(
            changed,
            plan=plan,
            to_binding=target_binding,
        )
    assert getattr(false_unbind.value, "code", None) == "FLEET_ADVANCE_INVALID"

    cross_seed = json.loads(canonical_json(value))
    cross_seed["evidence"]["checkpoint_records"][0]["uri"] = (
        cross_seed["evidence"]["checkpoint_records"][0]["uri"].replace(
            "seed-0",
            "seed-1",
        )
    )
    with pytest.raises(Exception) as cross_seed_record:
        validate_fleet_advance(
            cross_seed,
            plan=plan,
            to_binding=target_binding,
        )
    assert getattr(cross_seed_record.value, "code", None) == (
        "FLEET_ADVANCE_INVALID"
    )

    amendment = load_hardware_amendment(AMENDMENT)
    context = V3LifecycleContext(
        amendment=amendment,
        selection=selection,
        fleet_plan=plan,
        fleet_binding=target_binding,
    )
    backend = AwsP5Backend(
        profile=profile,
        runtime=SimpleNamespace(
            region=selection.region,
            s3_root="s3://memorysplit-prod/cohort-v3",
            kms_key_id=(
                "arn:aws:kms:us-east-1:123456789012:"
                "key/12345678-1234-4234-8234-123456789012"
            ),
            ami_id=selection.ami_id,
            container_image=selection.container_image,
            container_digest=selection.container_digest,
        ),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-v3"
        ),
        state_root=tmp_path / "state",
        runner=SimpleNamespace(
            run_json=lambda *_args, **_kwargs: pytest.fail(
                "local fleet admission must not use missing AWS tags as proof"
            )
        ),
    )
    with pytest.raises(Exception) as absent:
        backend._require_fleet_advance(target, context)
    assert getattr(absent.value, "code", None) == "FLEET_ADVANCE_REQUIRED"

    receipt_path = write_fleet_advance(
        backend.state_root,
        plan=plan,
        to_binding=target_binding,
        value=value,
    )
    assert receipt_path.is_file()
    assert load_fleet_advance(
        backend.state_root,
        plan=plan,
        to_binding=target_binding,
    ) is not None
    backend._require_fleet_advance(target, context)


def test_fleet_advance_dry_run_then_apply_verifies_and_unbinds_exact_tags(
    tmp_path,
    monkeypatch,
):
    from contextlib import nullcontext

    import msctl.aws_p5 as aws_p5
    from msctl.aws_fleet import (
        create_fleet_plan,
        validate_fleet_plan,
        write_fleet_plan,
    )
    from msctl.aws_selection import write_provider_selection

    profile, selection, paths = _manifests(tmp_path)
    instance_id = "i-0123456789abcdef0"
    plan_value = create_fleet_plan(
        profile=profile,
        selection=selection,
        manifest_paths=paths,
        instance_ids=[instance_id],
        repo_root=ROOT,
    )
    plan = validate_fleet_plan(
        plan_value,
        profile=profile,
        selection=selection,
        manifest_paths=paths,
        repo_root=ROOT,
    )
    plan_path = tmp_path / "fleet-plan.json"
    selection_path = tmp_path / "selection.json"
    write_fleet_plan(plan_path, plan_value)
    write_provider_selection(selection_path, selection.value)
    previous = load_run_manifest(paths[0], repo_root=ROOT)
    target = load_run_manifest(paths[1], repo_root=ROOT)
    calls = []
    approval_calls = []
    backend = aws_p5.AwsP5Backend(
        profile=profile,
        runtime=SimpleNamespace(
            region=selection.region,
            s3_root="s3://memorysplit-prod/cohort-v3",
            kms_key_id=(
                "arn:aws:kms:us-east-1:123456789012:"
                "key/12345678-1234-4234-8234-123456789012"
            ),
            ami_id=selection.ami_id,
            container_image=selection.container_image,
            container_digest=selection.container_digest,
        ),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-v3"
        ),
        state_root=tmp_path / "state",
        runner=SimpleNamespace(
            run_json=lambda argv, **kwargs: (
                calls.append((argv, kwargs["operation"])) or {}
            )
        ),
        approval_verifier=lambda **kwargs: approval_calls.append(kwargs),
    )
    terminate_at = "2026-07-25T00:00:00Z"
    common = {
        "status": "Success",
        "command_id": "training-command-12345678",
        "operation_id": "1" * 64,
        "intent_sha256": "2" * 64,
        "launch_readiness_sha256": "3" * 64,
        "provider": profile.provider,
        "instance_type": profile.instance_type,
        "seed": previous.seed,
        "release_sha256": previous.release_sha256,
        "dataset_sha256": previous.dataset_sha256,
        "run_manifest_sha256": previous.sha256,
        "profile_sha256": previous.profile_sha256,
        "runtime_sha256": "4" * 64,
        "container_digest": selection.container_digest,
        "gres": profile.gres,
        "terminate_at": terminate_at,
        "preregistration_sha256": previous.preregistration_sha256,
        "hardware_amendment_sha256": previous.hardware_amendment_sha256,
        "provider_selection_sha256": previous.provider_selection_sha256,
        "sealed_fixture_sha256": previous.sealed_fixture_sha256,
        "fleet_plan_sha256": plan.sha256,
        "fleet_wave": 0,
        "control_bundle_sha256": backend.control_bundle.sha256,
    }
    states = {
        run.run_id: {**common, "run_id": run.run_id, "arm": run.arm}
        for run in previous.runs
    }
    evaluation = {
        "status": "Success",
        "operation": "evaluate",
        "instance_id": instance_id,
        "run_manifest_sha256": previous.sha256,
        "fleet_plan_sha256": plan.sha256,
        "fleet_wave": 0,
        "launch_readiness_sha256": "3" * 64,
        "command_id": "evaluation-command-12345678",
        "operation_id": "5" * 64,
        "intent_sha256": "6" * 64,
            "sealed_evaluation_sha256": "a" * 64,
            "study_lock_sha256": "b" * 64,
    }

    class Store:
        def locked(self):
            return nullcontext()

        def read_run(self, run_id):
            return dict(states[run_id])

        def read_evaluation(self, manifest_sha256):
            assert manifest_sha256 == previous.sha256
            return dict(evaluation)

    monkeypatch.setattr(aws_p5, "StateStore", lambda _root: Store())
    monkeypatch.setattr(
        backend,
        "_terminal_checkpoint_identity",
        lambda *_args, **_kwargs: {
            "receipt": {},
            "checkpoint_receipt_sha256": "8" * 64,
            "sha256": "8" * 64,
            "uri": (
                "s3://memorysplit-prod/checkpoints/seed-0/receipts/"
                f"{'8' * 64}.json"
            ),
            "records": [
                {
                    "run_id": f"run-{arm}-0",
                    "arm": arm,
                    "sha256": digest,
                    "uri": (
                        "s3://memorysplit-prod/checkpoints/seed-0/"
                        f"{arm}/records/{digest}.json"
                    ),
                }
                for arm, digest in (
                    ("dense", "9" * 64),
                    ("split90", "a" * 64),
                )
            ],
        },
    )
    monkeypatch.setattr(
        backend,
        "_materialize_terminal_checkpoint_identity",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(backend, "_repair_paired_states", lambda *_args: None)
    monkeypatch.setattr(
        backend,
        "_same_state_binding",
        lambda *_args, **_kwargs: True,
    )
    arguments = {
        "amendment_path": AMENDMENT,
        "provider_selection_path": selection_path,
        "fleet_plan_path": plan_path,
        "target_manifest_path": paths[1],
        "repo_root": ROOT,
        "instance_id": instance_id,
        "checkpoint_receipt_path": tmp_path / "checkpoint-receipt.json",
    }
    dry_run = backend.fleet_advance(
        **arguments,
        approval_path=None,
        apply=False,
    )
    assert dry_run["advanced"] == 0
    assert dry_run["from_seed"] == 0
    assert dry_run["to_seed"] == 1
    assert calls == []

    approval = tmp_path / "fleet-advance-approval.json"
    approval.write_bytes(b'{"approved":true}\n')
    monkeypatch.setattr(
        backend,
        "_command_status",
        lambda *_args: "Success",
    )
    monkeypatch.setattr(
        backend,
        "_verify_state_terminal_receipt",
        lambda state: (
            "s3://memorysplit-prod/"
            f"{state['command_id']}/receipts/terminal.json"
        ),
    )
    monkeypatch.setattr(
        backend,
        "_selected_instance_argv",
        lambda *_args: ["describe-instance"],
    )
    observed_bindings = []

    def parse_selected(*_args, **kwargs):
        observed_bindings.append(dict(kwargs["expected_binding"]))
        return {}

    monkeypatch.setattr(backend, "_parse_selected_instance", parse_selected)
    applied = backend.fleet_advance(
        **arguments,
        approval_path=approval,
        apply=True,
    )
    assert applied["advanced"] == 1
    assert applied["advance_receipt_sha256"]
    assert len(approval_calls) == 1
    assert approval_calls[0]["operation"] == "fleet-advance"
    assert approval_calls[0]["scope_sha256"] == plan.sha256
    assert all(value is not None for value in observed_bindings[0].values())
    assert all(value is None for value in observed_bindings[1].values())
    assert [operation for _argv, operation in calls] == [
        "verify fleet advance binding",
        "unbind completed fleet wave",
        "verify fleet wave unbound",
    ]


@pytest.mark.parametrize("profile_path", [P5, P6])
def test_v3_profiles_support_full_dry_run_lifecycle(
    profile_path,
    tmp_path,
    monkeypatch,
):
    from msctl.aws_fleet import (
        create_fleet_plan,
        validate_fleet_plan,
        write_fleet_plan,
    )
    from msctl.aws_p5 import AwsP5Backend, V3LifecycleContext
    from msctl.aws_readiness import LaunchReadiness
    from msctl.aws_control_bundle import write_control_bundle
    from msctl.aws_sealed_evaluation import (
        SealedEvaluationFixture,
        SealedEvaluationRelease,
    )
    from msctl.aws_selection import write_provider_selection
    from msctl.aws_argv import _receipt, _validate_intent, _validate_receipt
    from msctl.contracts import verify_checkpoint_receipt
    from msctl.state import StateStore
    profile, selection, paths = _manifests(
        tmp_path,
        profile_path=profile_path,
    )
    instance_id = "i-0123456789abcdef0"
    plan_value = create_fleet_plan(
        profile=profile,
        selection=selection,
        manifest_paths=paths,
        instance_ids=[instance_id],
        repo_root=ROOT,
    )
    plan = validate_fleet_plan(
        plan_value,
        profile=profile,
        selection=selection,
        manifest_paths=paths,
        repo_root=ROOT,
    )
    manifest = load_run_manifest(paths[0], repo_root=ROOT)
    selection_path = tmp_path / "provider-selection.json"
    plan_path = tmp_path / "fleet-plan.json"
    write_provider_selection(selection_path, selection.value)
    write_fleet_plan(plan_path, plan_value)
    runtime = SimpleNamespace(
        region=selection.region,
        s3_root="s3://memorysplit-prod/cohort-v3",
        kms_key_id=(
            "arn:aws:kms:us-east-1:123456789012:"
            "key/12345678-1234-4234-8234-123456789012"
        ),
        ami_id=selection.ami_id,
        container_image=selection.container_image,
        container_digest=selection.container_digest,
    )
    backend = AwsP5Backend(
        profile=profile,
        runtime=runtime,
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-v3"
        ),
        state_root=tmp_path / "state",
        runner=SimpleNamespace(
            run_json=lambda *_args, **_kwargs: pytest.fail(
                "dry-run lifecycle must not call AWS"
            )
        ),
    )
    amendment = load_hardware_amendment(AMENDMENT)
    sealed_fixture = SealedEvaluationFixture(
        root=tmp_path / "sealed-fixture",
        sha256=manifest.sealed_fixture_sha256,
        members={},
    )
    sealed = SealedEvaluationRelease(
        root=tmp_path / "sealed-evaluation",
        sha256="9" * 64,
        fixture_sha256=manifest.sealed_fixture_sha256,
        study_lock_sha256="a" * 64,
        preregistration_sha256=manifest.preregistration_sha256,
        members={},
    )
    readiness = LaunchReadiness(
        profile_id=profile.profile_id,
        sha256="8" * 64,
        bindings={
            "release_sha256": manifest.release_sha256,
            "provider_selection_sha256": selection.sha256,
            "hardware_amendment_sha256": amendment.sha256,
            "sealed_fixture_sha256": manifest.sealed_fixture_sha256,
            "environment_receipt_sha256": "7" * 64,
        },
        decision={"protected_launch_allowed": True},
        path=None,
        value={},
    )
    context = V3LifecycleContext(
        amendment=amendment,
        selection=selection,
        fleet_plan=plan,
        fleet_binding=plan.binding_for_seed(manifest.seed),
        readiness=readiness,
        sealed_fixture=sealed_fixture,
    )
    evaluation_context = V3LifecycleContext(
        amendment=amendment,
        selection=selection,
        fleet_plan=plan,
        fleet_binding=plan.binding_for_seed(manifest.seed),
        readiness=readiness,
        sealed_evaluation=sealed,
    )
    mismatched_evaluation_context = V3LifecycleContext(
        amendment=amendment,
        selection=selection,
        fleet_plan=plan,
        fleet_binding=plan.binding_for_seed(manifest.seed),
        readiness=readiness,
        sealed_evaluation=SealedEvaluationRelease(
            root=sealed.root,
            sha256=sealed.sha256,
            fixture_sha256="f" * 64,
            study_lock_sha256=sealed.study_lock_sha256,
            preregistration_sha256=sealed.preregistration_sha256,
            members=sealed.members,
        ),
    )
    with pytest.raises(Exception) as fixture_mismatch:
        backend._v3_evaluation_bindings(
            manifest,
            mismatched_evaluation_context,
        )
    assert getattr(fixture_mismatch.value, "code", None) == (
        "PROVIDER_SELECTION_MISMATCH"
    )
    assert context is not None
    release = SimpleNamespace(
        provider=profile.provider,
        archive_sha256=manifest.release_sha256,
        source_commit=manifest.source_commit,
    )
    evidence = {
        "dataset_pointer_sha256": "5" * 64,
        "dataset_verification_sha256": "6" * 64,
        "environment_receipt_sha256": "7" * 64,
    }
    reviewed_bundle = tmp_path / "control-bundle.tar"
    write_control_bundle(reviewed_bundle, backend.control_bundle)
    control_install = backend.control_install(
        instance_id=instance_id,
        bundle_path=reviewed_bundle,
        bundle_sha256=backend.control_bundle.sha256,
        apply=False,
    )
    assert control_install["ssm_document"] == "AWS-RunShellScript"
    assert control_install["control_bundle_sha256"] == (
        backend.control_bundle.sha256
    )
    send_command = control_install["commands"][2]
    assert send_command[send_command.index("--document-name") + 1] == (
        "AWS-RunShellScript"
    )
    terminate_at = (
        datetime.now(UTC) + timedelta(hours=1)
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    checkpoint_rows = []
    for run in manifest.runs:
        checkpoint = tmp_path / f"{run.arm}.pt"
        checkpoint.write_bytes(f"{profile.provider}:{run.run_id}".encode())
        checkpoint_sha256 = _sha256(checkpoint)
        run_binding_sha256 = (
            "b" * 64 if run.arm == "dense" else "c" * 64
        )
        checkpoint_record_sha256 = (
            "d" * 64 if run.arm == "dense" else "e" * 64
        )
        artifact_prefix = (
            f"{runtime.s3_root}/checkpoints/seed-{run.seed}/{run.arm}"
        )
        checkpoint_rows.append(
            {
                "run_id": run.run_id,
                "arm": run.arm,
                "seed": run.seed,
                "path": checkpoint.name,
                "sha256": checkpoint_sha256,
                "checkpoint_uri": (
                    f"{artifact_prefix}/sha256/{checkpoint_sha256}.pt"
                ),
                "config_sha256": run.config_sha256,
                "configuration_uri": (
                    f"{artifact_prefix}/configuration/sha256/"
                    f"{run.config_sha256}.yaml"
                ),
                "run_binding_sha256": run_binding_sha256,
                "run_binding_uri": (
                    f"{artifact_prefix}/run-binding/sha256/"
                    f"{run_binding_sha256}.json"
                ),
                "checkpoint_record_sha256": checkpoint_record_sha256,
                "checkpoint_record_uri": (
                    f"{artifact_prefix}/records/"
                    f"{checkpoint_record_sha256}.json"
                ),
                "dataset_sha256": manifest.dataset_sha256,
                "source_commit": manifest.source_commit,
                "step": 1358,
                "world_size": 4,
            }
        )
    checkpoint_value = {
        "schema_version": 3,
        "provider": manifest.provider,
        "release_sha256": manifest.release_sha256,
        "run_manifest_sha256": manifest.sha256,
        "dataset_sha256": manifest.dataset_sha256,
        "source_commit": manifest.source_commit,
        "cohort_assignment_sha256": manifest.cohort_assignment_sha256,
        "preregistration_sha256": manifest.preregistration_sha256,
        "hardware_amendment_sha256": manifest.hardware_amendment_sha256,
        "provider_selection_sha256": manifest.provider_selection_sha256,
        "profile_sha256": manifest.profile_sha256,
        "sealed_fixture_sha256": manifest.sealed_fixture_sha256,
        "checkpoints": checkpoint_rows,
    }
    checkpoint_path = tmp_path / "checkpoint-receipt.json"
    checkpoint_path.write_bytes(canonical_json(checkpoint_value) + b"\n")
    checkpoint_receipt = verify_checkpoint_receipt(
        checkpoint_path,
        release=release,
        manifest=manifest,
    )

    rendered = backend.render(
        release=release,
        manifest=manifest,
        evidence=evidence,
        context=context,
    )
    submitted = backend.submit(
        release=release,
        manifest=manifest,
        instance_id=None,
        terminate_at=terminate_at,
        approval_path=None,
        apply=False,
        evidence=evidence,
        context=context,
    )
    resumed = backend.resume(
        release=release,
        manifest=manifest,
        checkpoint_receipt=checkpoint_receipt,
        approval_path=None,
        apply=False,
        evidence=evidence,
        context=context,
    )
    cancelled = backend.cancel(
        release=release,
        manifest=manifest,
        approval_path=None,
        apply=False,
        context=context,
    )
    evaluated = backend.evaluate(
        release=release,
        manifest=manifest,
        approval_path=None,
        apply=False,
        evidence=evidence,
        context=evaluation_context,
        checkpoint_receipt=checkpoint_receipt,
    )
    cleaned = backend.cleanup(
        release=release,
        manifest=manifest,
        approval_path=None,
        apply=False,
        context=context,
    )
    collected = backend.collect(
        source=f"results/seed-{manifest.seed}.json",
        out=tmp_path / "collected.json",
        apply=False,
    )

    intent = submitted["operation_intent"]
    intent_sha256 = hashlib.sha256(canonical_json(intent)).hexdigest()
    validated_intent = _validate_intent(
        canonical_json(intent),
        expected_sha256=intent_sha256,
        expected_control_bundle_sha256=backend.control_bundle.sha256,
    )
    for protected in (
        resumed["operation_intent"],
        evaluated["operation_intent"],
    ):
        protected_intent = backend._operation_envelope(
            protected,
            instance_id=instance_id,
            terminate_at=terminate_at,
        )
        protected_payload = canonical_json(protected_intent)
        assert _validate_intent(
            protected_payload,
            expected_sha256=hashlib.sha256(protected_payload).hexdigest(),
            expected_control_bundle_sha256=backend.control_bundle.sha256,
        ) == protected_intent
    started_receipt = _receipt(
        validated_intent,
        intent_sha256=intent_sha256,
        kind="started",
        nonce="a" * 32,
    )
    validated_receipt = _validate_receipt(
        started_receipt,
        validated_intent,
        intent_sha256=intent_sha256,
        kind="started",
    )
    assert (
        validated_receipt["provider_selection_sha256"]
        == manifest.provider_selection_sha256
    )
    published = {
        "intent_sha256": intent_sha256,
        "intent_uri": (
            f"{runtime.s3_root}/operations/intents/sha256/"
            f"{intent_sha256}.json"
        ),
    }
    states = [
        backend._new_aws_run_state(
            run=run,
            manifest=manifest,
            operation="submit",
            instance_id=instance_id,
            terminate_at=terminate_at,
            intent=intent,
            published=published,
            attempt=1,
            context=context,
        )
        for run in manifest.runs
    ]
    for state in states:
        state["command_id"] = "command-12345678"
        state["status"] = "Pending"
        state["send_attempted"] = True
    store = StateStore(backend.state_root)
    with store.locked():
        backend._write_paired_states(store, manifest, states, context)
    status = backend.status(
        release=release,
        manifest=manifest,
        cached=True,
        context=context,
    )
    for state in states:
        state["status"] = "Cancelling"
    with store.locked():
        backend._write_paired_states(store, manifest, states, context)
    approval_calls = []
    monkeypatch.setattr(
        backend,
        "_verify_approval",
        lambda _path, *, operation, release, manifest, resources=None: (
            approval_calls.append((operation, resources))
        ),
    )
    idempotent_cancel = backend.cancel(
        release=release,
        manifest=manifest,
        approval_path=tmp_path / "approval.json",
        apply=True,
        context=context,
    )

    expected_bindings = {
        "cohort_assignment_sha256": manifest.cohort_assignment_sha256,
        "preregistration_sha256": manifest.preregistration_sha256,
        "hardware_amendment_sha256": manifest.hardware_amendment_sha256,
        "provider_selection_sha256": manifest.provider_selection_sha256,
        "profile_sha256": manifest.profile_sha256,
        "sealed_fixture_sha256": manifest.sealed_fixture_sha256,
        "fleet_plan_sha256": plan.sha256,
        "fleet_wave": context.fleet_binding.wave,
        "launch_readiness_sha256": readiness.sha256,
        "control_bundle_sha256": backend.control_bundle.sha256,
    }
    for lifecycle_intent in (
        rendered["operation_intent"],
        submitted["operation_intent"],
        resumed["operation_intent"],
    ):
        assert lifecycle_intent["schema_version"] == 3
        assert {
            key: lifecycle_intent[key] for key in expected_bindings
        } == expected_bindings
        assert "sealed_evaluation_sha256" not in lifecycle_intent
        assert "study_lock_sha256" not in lifecycle_intent
    assert {
        key: evaluated["operation_intent"][key] for key in expected_bindings
    } == expected_bindings
    assert evaluated["operation_intent"]["sealed_evaluation_sha256"] == sealed.sha256
    assert evaluated["operation_intent"]["study_lock_sha256"] == (
        sealed.study_lock_sha256
    )
    for result in (cancelled, cleaned):
        assert {key: result[key] for key in expected_bindings} == expected_bindings
    assert submitted["instance_id"] == instance_id
    assert status["instance_id"] == instance_id
    assert status["authoritative"] is False
    assert idempotent_cancel["idempotent"] is True
    assert approval_calls[0][0] == "cancel"
    assert approval_calls[0][1]["provider_selection_sha256"] == (
        manifest.provider_selection_sha256
    )
    assert resumed["submitted"] == 0
    assert evaluated["submitted"] == 0
    from msctl.aws_sealed_evaluation import REQUIRED_SEALED_MEMBERS

    evaluation_steps = evaluated["operation_intent"]["steps"]
    materializations = [
        step
        for step in evaluation_steps
        if step["name"].startswith("materialize-sealed-evaluation-")
    ]
    assert len(materializations) == len(REQUIRED_SEALED_MEMBERS)
    assert {
        step["argv"][-1].rsplit("/", 1)[-1] for step in materializations
    } == set(REQUIRED_SEALED_MEMBERS)
    assert all(
        "s3api" in step["argv"]
        and "get-object" in step["argv"]
        and "sync" not in step["argv"]
        and step["argv"][step["argv"].index("--checksum-mode") + 1]
        == "ENABLED"
        for step in materializations
    )
    terminal_materializations = [
        step
        for step in evaluation_steps
        if step["name"].startswith("materialize-terminal-")
    ]
    assert len(terminal_materializations) == 9
    assert all(
        "s3api" in step["argv"]
        and "get-object" in step["argv"]
        and "sync" not in step["argv"]
        and step["argv"][step["argv"].index("--checksum-mode") + 1]
        == "ENABLED"
        for step in terminal_materializations
    )
    evaluation_envelope = backend._operation_envelope(
        evaluated["operation_intent"],
        instance_id=instance_id,
        terminate_at=terminate_at,
    )
    evaluation_intent_sha256 = hashlib.sha256(
        canonical_json(evaluation_envelope)
    ).hexdigest()
    validated_evaluation = _validate_intent(
        canonical_json(evaluation_envelope),
        expected_sha256=evaluation_intent_sha256,
        expected_control_bundle_sha256=backend.control_bundle.sha256,
    )
    assert validated_evaluation["operation"] == "evaluate"
    assert validated_evaluation["checkpoint_receipt"]["sha256"] == (
        checkpoint_receipt.sha256
    )
    assert cancelled["cancelled"] == 0
    assert cleaned["terminated"] == 0
    assert collected["collected"] == 0
    assert submitted["gres"] == profile.gres
    assert all(
        forbidden not in json.dumps(result)
        for result in (
            rendered,
            submitted,
            resumed,
            cancelled,
            evaluated,
            cleaned,
            collected,
        )
        for forbidden in (
            "run-instances",
            "purchase-reserved-instances-offering",
            "create-capacity-reservation",
        )
    )


def test_control_install_binds_reviewed_file_hash_and_running_source(tmp_path):
    from msctl.aws_control_bundle import (
        CONTROL_BUNDLE_MEMBERS,
        build_control_bundle_bytes,
        write_control_bundle,
    )
    from msctl.aws_p5 import AwsP5Backend

    profile, _amendment, selection = _selection(P5)
    runner_calls = []
    backend = AwsP5Backend(
        profile=profile,
        runtime=SimpleNamespace(
            region=selection.region,
            s3_root="s3://memorysplit-prod/cohort-v3",
            kms_key_id=(
                "arn:aws:kms:us-east-1:123456789012:"
                "key/12345678-1234-4234-8234-123456789012"
            ),
            ami_id=selection.ami_id,
            container_image=selection.container_image,
            container_digest=selection.container_digest,
        ),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-v3"
        ),
        state_root=tmp_path / "state",
        runner=SimpleNamespace(
            run_json=lambda *args, **kwargs: runner_calls.append((args, kwargs))
        ),
    )
    reviewed = tmp_path / "reviewed-control.tar"
    write_control_bundle(reviewed, backend.control_bundle)

    with pytest.raises(Exception) as wrong_hash:
        backend.control_install(
            instance_id="i-0123456789abcdef0",
            bundle_path=reviewed,
            bundle_sha256="0" * 64,
            apply=False,
        )
    assert getattr(wrong_hash.value, "code", None) == "CONTROL_BUNDLE_INVALID"

    alternate_root = tmp_path / "alternate-source"
    for relative in CONTROL_BUNDLE_MEMBERS:
        destination = alternate_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    changed = alternate_root / "msctl" / "errors.py"
    changed.write_bytes(changed.read_bytes() + b"\n")
    alternate = build_control_bundle_bytes(alternate_root)
    alternate_path = tmp_path / "alternate-control.tar"
    write_control_bundle(alternate_path, alternate)

    with pytest.raises(Exception) as wrong_source:
        backend.control_install(
            instance_id="i-0123456789abcdef0",
            bundle_path=alternate_path,
            bundle_sha256=alternate.sha256,
            apply=False,
        )
    assert getattr(wrong_source.value, "code", None) == "CONTROL_BUNDLE_INVALID"
    assert runner_calls == []


def test_control_install_apply_waits_for_verified_ssm_success(tmp_path, monkeypatch):
    import base64

    import msctl.aws_p5 as aws_p5

    profile, _amendment, selection = _selection(P5)
    kms_key_id = (
        "arn:aws:kms:us-east-1:123456789012:"
        "key/12345678-1234-4234-8234-123456789012"
    )
    statuses = iter(("InProgress", "Success"))
    calls: list[tuple[str, ...]] = []
    backend_ref: list[aws_p5.AwsP5Backend] = []

    def run_json(argv, *, operation):
        del operation
        exact = tuple(argv)
        calls.append(exact)
        if "describe-instance-information" in exact:
            return {
                "managed_instances": [
                    {
                        "instance_id": "i-0123456789abcdef0",
                        "ping_status": "Online",
                    }
                ]
            }
        if "put-object" in exact:
            return {
                "object": {
                    "checksum_sha256": exact[
                        exact.index("--checksum-sha256") + 1
                    ],
                    "version_id": "version-1",
                }
            }
        if "head-object" in exact:
            bundle = backend_ref[0].control_bundle
            return {
                "object": {
                    "checksum_sha256": base64.b64encode(
                        bytes.fromhex(bundle.sha256)
                    ).decode("ascii"),
                    "content_length": bundle.bytes,
                    "metadata": {
                        "bundle-type": "memorysplit-aws-control-bundle-v1",
                        "sha256": bundle.sha256,
                    },
                    "server_side_encryption": "aws:kms",
                    "sse_kms_key_id": kms_key_id,
                    "version_id": "version-1",
                }
            }
        if "send-command" in exact:
            return {
                "command": {
                    "command_id": "control-command-12345678",
                    "status": "Pending",
                }
            }
        if "get-command-invocation" in exact:
            return {
                "command": {
                    "command_id": "control-command-12345678",
                    "status": next(statuses),
                }
            }
        pytest.fail(f"unexpected AWS argv: {exact}")

    backend = aws_p5.AwsP5Backend(
        profile=profile,
        runtime=SimpleNamespace(
            region=selection.region,
            s3_root="s3://memorysplit-prod/cohort-v3",
            kms_key_id=kms_key_id,
            ami_id=selection.ami_id,
            container_image=selection.container_image,
            container_digest=selection.container_digest,
        ),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-v3"
        ),
        state_root=tmp_path / "state",
        runner=SimpleNamespace(run_json=run_json),
    )
    backend_ref.append(backend)
    monkeypatch.setattr(aws_p5.time, "sleep", lambda _seconds: None)
    reviewed_bundle = tmp_path / "control-bundle.tar"
    from msctl.aws_control_bundle import write_control_bundle

    write_control_bundle(reviewed_bundle, backend.control_bundle)

    result = backend.control_install(
        instance_id="i-0123456789abcdef0",
        bundle_path=reviewed_bundle,
        bundle_sha256=backend.control_bundle.sha256,
        apply=True,
    )

    assert result["installed"] == 1
    assert result["status"] == "Success"
    assert sum("get-command-invocation" in call for call in calls) == 2


@pytest.mark.parametrize(
    ("profile_path", "instance_type", "gpu_names"),
    [
        (P5, "p5.48xlarge", ("NVIDIA H100 80GB HBM3",) * 8),
        (P6, "p6-b300.48xlarge", ("NVIDIA B300",) * 8),
    ],
)
def test_v3_launcher_manifest_is_consumable_by_remote_dry_run(
    profile_path,
    instance_type,
    gpu_names,
    tmp_path,
):
    from cluster.aws.p5.launch_seed_pair import load_launch_plan, render_plan
    from cluster.aws.p5.profile import load_aws_gpu_profile
    from msctl.aws_launch_manifest import build_launcher_manifest
    from tests.test_aws_p5_launcher import (
        BOOT_ID,
        CODE_COMMIT,
        HEX,
        SAFE_ENVIRONMENT,
        _launcher_fixture,
    )

    seed = 3
    fixture = _launcher_fixture(tmp_path, seed=seed)
    repo_root = fixture["repo_root"]
    scratch_root = fixture["scratch_root"]
    for directory in sorted(
        (path for path in repo_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
    ):
        directory.chmod(0o755)
    repo_root.chmod(0o755)
    for path in repo_root.rglob("*"):
        if path.is_file():
            path.chmod(0o644)

    configs = {}
    for arm in ("dense", "split90"):
        old = fixture["configs"][arm]
        old.unlink()
        config = (
            repo_root
            / "configs"
            / "360m-v3"
            / f"{arm}-s{seed}.yaml"
        )
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_bytes(
            (ROOT / "configs" / "360m-v3" / config.name).read_bytes()
        )
        configs[arm] = config

    profile = load_aws_gpu_profile(profile_path)
    member_paths = sorted(
        path.relative_to(repo_root).as_posix()
        for path in repo_root.rglob("*")
        if path.is_file()
        and path.name not in {"RELEASE-METADATA.json", "SHA256SUMS"}
    )
    metadata = {
        "members": [
            {
                "bytes": (repo_root / relative).stat().st_size,
                "git_blob": "5" * 40,
                "git_mode": "100644",
                "path": relative,
                "sha256": _sha256(repo_root / relative),
            }
            for relative in member_paths
        ],
        "package_format_version": "aws-gpu-v3",
        "provider": profile.provider,
        "selected_profile_id": profile.profile_id,
        "schema_version": 1,
        "seed_assignment": {
            "arms": ["dense", "split90"],
            "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
            "provider": "aws-p5.48xlarge-v3",
            "seeds": list(range(10)),
        },
        "source": {
            "commit": CODE_COMMIT,
            "dirty": False,
            "tree": "6" * 40,
        },
        "cohort_assignment": {},
        "preregistration": {},
        "hardware_amendment": {},
        "profile": {"sha256": profile.sha256},
        "environment": {},
        "dataset_pointer": {},
        "container_base_lock": {},
        "config_sha256": {},
        "contract_locks": {},
    }
    metadata_path = repo_root / "RELEASE-METADATA.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    sums_path = repo_root / "SHA256SUMS"
    sums_path.write_text(
        "".join(
            f"{_sha256(repo_root / relative)}  {relative}\n"
            for relative in sorted([*member_paths, "RELEASE-METADATA.json"])
        ),
        encoding="ascii",
    )
    release_members_sha256 = _sha256(sums_path)

    bootstrap_path = fixture["bootstrap_path"]
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap.update(
        {
            "instance_type": profile.instance_type,
            "profile_sha256": profile.sha256,
            "provider": profile.provider,
            "receipt_type": profile.bootstrap_receipt_type,
            "release_members_sha256": release_members_sha256,
        }
    )
    bootstrap_path.write_bytes(canonical_json(bootstrap) + b"\n")

    for path in repo_root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for directory in sorted(
        (path for path in repo_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    repo_root.chmod(0o555)

    fixture["manifest_path"].unlink()
    v3_hashes = {
        "run_manifest_sha256": "a" * 64,
        "preregistration_sha256": "b" * 64,
        "hardware_amendment_sha256": "c" * 64,
        "provider_selection_sha256": "d" * 64,
        "sealed_fixture_sha256": "e" * 64,
        "fleet_plan_sha256": "f" * 64,
        "launch_readiness_sha256": "2" * 64,
        "control_bundle_sha256": "3" * 64,
    }
    build_launcher_manifest(
        out=fixture["manifest_path"],
        scratch_root=scratch_root,
        seed=seed,
        profile_sha256=profile.sha256,
        release_sha256=HEX["release"],
        release_members_sha256=release_members_sha256,
        cohort_assignment_sha256=HEX["cohort"],
        code_commit=CODE_COMMIT,
        bootstrap_receipt=bootstrap_path,
        corpus_receipt=fixture["corpus_path"],
        runs=[
            {
                "arm": arm,
                "config": config.relative_to(repo_root).as_posix(),
                "config_sha256": _sha256(config),
            }
            for arm, config in sorted(configs.items())
        ],
        profile=profile,
        fleet_wave=1,
        **v3_hashes,
    )
    environment = {
        **SAFE_ENVIRONMENT,
        "MS_S3_ROOT": "s3://memorysplit-prod/cohort-v3",
    }
    plan = load_launch_plan(
        seed=seed,
        manifest_path=fixture["manifest_path"],
        profile_path=profile_path,
        repo_root=repo_root,
        scratch_root=scratch_root,
        environment=environment,
        observed_instance_type=instance_type,
        observed_instance_id="i-0123456789abcdef0",
        observed_boot_id=BOOT_ID,
        gpu_names=gpu_names,
        port_available=lambda _port: True,
        semantic_corpus_verifier=lambda _root: fixture["corpus"],
        enforce_profile_scratch=False,
    )
    rendered = render_plan(plan)

    assert rendered["provider"] == profile.provider
    assert rendered["gres"] == profile.gres
    assert {
        tuple(command["runtime_config"]["snapshot_steps"])
        for command in rendered["commands"]
    } == {(1358, 3396, 6791, 10187, 13582)}
    sidecar_hashes = {
        row["name"]: row["stream_sha256"]
        for row in fixture["corpus"]["sidecar_sets"]
    }
    assert plan.ordered_stream_sha256 == fixture["corpus"][
        "ordered_stream_sha256"
    ]
    assert plan.release_members_sha256 == release_members_sha256
    for launch in plan.arms:
        assert launch.model_id == "d360m"
        assert launch.raw_token_count == 7_120_879_616
        assert launch.route_dose_sha256 == sidecar_hashes[
            f"{launch.arm}_target_weights"
        ]
