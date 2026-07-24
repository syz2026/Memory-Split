from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from msctl.aws_selection import (
    create_provider_selection,
    load_hardware_amendment,
    validate_provider_selection,
)
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
    value = create_provider_selection(
        profile=profile,
        amendment=amendment,
        region="us-east-1",
        ami_id="ami-0123456789abcdef0",
        container_image=CONTAINER_IMAGE,
        container_digest=CONTAINER_DIGEST,
        selected_at="2026-07-24T00:00:00Z",
    )
    return profile, amendment, validate_provider_selection(
        value,
        amendment=amendment,
        profile=profile,
    )


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
            "sealed_evaluation_sha256": "4" * 64,
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
        "sealed_evaluation": ROOT / "evals" / "confirmatory" / "runner.py",
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
    assert manifest.estimated_instance_hours == 24.0
    assert manifest.estimated_gpu_hours == 192.0


def test_provider_selection_cli_is_dry_run_first_and_fails_closed(tmp_path):
    from msctl.aws_selection import (
        load_provider_selection,
        write_provider_selection,
    )
    from msctl.cli import build_parser, dispatch

    profile, amendment, selection = _selection(P5)
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
    from msctl.aws_p5 import AwsP5Backend
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
    manifest = load_run_manifest(paths[3], repo_root=ROOT)
    selection_path = tmp_path / "provider-selection.json"
    plan_path = tmp_path / "fleet-plan.json"
    write_provider_selection(selection_path, selection.value)
    write_fleet_plan(plan_path, plan_value)
    runtime = SimpleNamespace(
        region=selection.region,
        s3_root="s3://memorysplit-prod/cohort-v3",
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
    context = backend._load_v3_context(
        manifest,
        amendment_path=AMENDMENT,
        provider_selection_path=selection_path,
        fleet_plan_path=plan_path,
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
    terminate_at = (
        datetime.now(UTC) + timedelta(hours=1)
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    checkpoint_rows = []
    for run in manifest.runs:
        checkpoint = tmp_path / f"{run.arm}.pt"
        checkpoint.write_bytes(f"{profile.provider}:{run.run_id}".encode())
        checkpoint_rows.append(
            {
                "run_id": run.run_id,
                "arm": run.arm,
                "seed": run.seed,
                "path": checkpoint.name,
                "sha256": _sha256(checkpoint),
                "config_sha256": run.config_sha256,
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
        "sealed_evaluation_sha256": manifest.sealed_evaluation_sha256,
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
        context=context,
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
    )
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
        "sealed_evaluation_sha256": manifest.sealed_evaluation_sha256,
        "fleet_plan_sha256": plan.sha256,
        "fleet_wave": context.fleet_binding.wave,
    }
    for lifecycle_intent in (
        rendered["operation_intent"],
        submitted["operation_intent"],
        resumed["operation_intent"],
        evaluated["operation_intent"],
    ):
        assert lifecycle_intent["schema_version"] == 3
        assert {
            key: lifecycle_intent[key] for key in expected_bindings
        } == expected_bindings
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
        "package_format_version": 1,
        "provider": profile.provider,
        "schema_version": 1,
        "seed_assignment": {
            "arms": ["dense", "split90"],
            "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
            "provider": "aws-p5.48xlarge-v3",
            "seeds": list(range(10)),
        },
        "source": {"commit": CODE_COMMIT, "dirty": False},
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
        "sealed_evaluation_sha256": "e" * 64,
        "fleet_plan_sha256": "f" * 64,
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
