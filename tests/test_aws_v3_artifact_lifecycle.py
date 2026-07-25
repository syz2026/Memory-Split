from __future__ import annotations

import base64
import hashlib
import json
import shutil
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from cluster.aws.p5.interruption_checkpoint import UploadedObject
from cluster.aws.p5.terminal_artifacts import (
    EVALUATION_ARTIFACTS,
    _snapshot_regular,
    publish_evaluation_pair,
    publish_terminal_pair,
    verify_durable_terminal,
)
from msctl.aws_fleet import (
    create_fleet_advance,
    create_fleet_plan,
    fleet_transition_for_target,
    validate_fleet_plan,
    verify_fleet_collection,
    write_fleet_advance,
    write_fleet_plan,
)
from msctl.aws_p5 import AwsP5Backend, V3LifecycleContext
from msctl.aws_sealed_evaluation import (
    REQUIRED_SEALED_MEMBERS,
    load_sealed_evaluation_fixture,
    load_sealed_evaluation_release,
)
from msctl.aws_selection import (
    load_hardware_amendment,
    write_provider_selection,
)
from msctl.contracts import load_run_manifest, verify_checkpoint_receipt
from msctl.jsonutil import canonical_json
from tests.test_v3_hardware_amendment import (
    AMENDMENT,
    ROOT,
    _manifests,
    _sealed_fixture,
    _sealed_release,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class _MemoryObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_verified(
        self,
        path,
        uri,
        *,
        expected_sha256,
        deadline,
        monotonic,
        wall_deadline=None,
        wall_monotonic=None,
    ):
        del deadline, monotonic, wall_deadline, wall_monotonic
        payload = Path(path).read_bytes()
        if _digest(payload) != expected_sha256:
            return None
        existing = self.objects.get(uri)
        if existing is not None and existing != payload:
            return None
        self.objects[uri] = payload
        return UploadedObject(
            uri=uri,
            sha256=expected_sha256,
            bytes=len(payload),
        )


class _MemoryAwsRunner:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.calls: list[tuple[str, ...]] = []

    def run_json(self, argv, *, operation):
        del operation
        exact = tuple(argv)
        self.calls.append(exact)
        if "get-object" not in exact:
            pytest.fail(f"unexpected AWS operation: {exact}")
        bucket = exact[exact.index("--bucket") + 1]
        key = exact[exact.index("--key") + 1]
        destination = Path(exact[exact.index("--checksum-mode") + 2])
        uri = f"s3://{bucket}/{key}"
        payload = self.objects[uri]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return {
            "object": {
                "checksum_sha256": base64.b64encode(
                    bytes.fromhex(_digest(payload))
                ).decode("ascii"),
                "version_id": "version-1",
            }
        }


def _runtime(selection):
    return SimpleNamespace(
        region=selection.region,
        s3_root="s3://memorysplit-prod/cohort-v3",
        kms_key_id=(
            "arn:aws:kms:us-east-1:123456789012:"
            "key/12345678-1234-4234-8234-123456789012"
        ),
        ami_id=selection.ami_id,
        container_image=selection.container_image,
        container_digest=selection.container_digest,
        uid=1000,
        gid=1000,
    )


def _backend(profile, selection, tmp_path, runner):
    return AwsP5Backend(
        profile=profile,
        runtime=_runtime(selection),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-v3"
        ),
        state_root=tmp_path / "state",
        runner=runner,
    )


def _advance_evidence(seed: int) -> dict[str, object]:
    return {
        "training_state_sha256": "1" * 64,
        "training_command_id": f"training-command-{seed:08d}",
        "training_terminal_receipt_uri": (
            f"s3://memorysplit-prod/training/{seed}/receipts/terminal.json"
        ),
        "checkpoint_receipt_sha256": "4" * 64,
        "checkpoint_receipt_uri": (
            f"s3://memorysplit-prod/checkpoints/seed-{seed}/receipts/"
            f"{'4' * 64}.json"
        ),
        "checkpoint_records": [
            {
                "run_id": f"run-{arm}-{seed}",
                "arm": arm,
                "sha256": digest,
                "uri": (
                    f"s3://memorysplit-prod/checkpoints/seed-{seed}/{arm}/"
                    f"records/{digest}.json"
                ),
            }
            for arm, digest in (
                ("dense", "5" * 64),
                ("split90", "8" * 64),
            )
        ],
        "aws_bound_tags_sha256": "6" * 64,
        "aws_unbound_tags_sha256": "7" * 64,
    }


def _publish_terminal_manifest(
    *,
    manifest,
    profile,
    runtime,
    root: Path,
    object_store: _MemoryObjectStore,
):
    scratch = root / f"training-seed-{manifest.seed}"
    scratch.mkdir()
    launches = []
    for run in sorted(manifest.runs, key=lambda item: item.arm):
        run_root = scratch / "runs" / f"seed-{manifest.seed}" / run.arm / "run"
        run_root.mkdir(parents=True)
        checkpoint_payload = f"terminal:{run.run_id}".encode("ascii")
        checkpoint = run_root / "ckpt.pt"
        checkpoint.write_bytes(checkpoint_payload)
        metadata = {
            "schema_version": 1,
            "receipt_type": "memorysplit-training-checkpoint-v1",
            "run_id": run.run_id,
            "condition": run.arm,
            "seed": manifest.seed,
            "step": 1358,
            "max_steps": 1358,
            "world_size": 4,
            "config_fingerprint": run.config_sha256,
            "checkpoint_path": "ckpt.pt",
            "checkpoint_sha256": _digest(checkpoint_payload),
            "checkpoint_bytes": len(checkpoint_payload),
            "terminal": True,
        }
        (run_root / "checkpoint-meta.json").write_bytes(
            canonical_json(metadata) + b"\n"
        )
        launches.append(
            SimpleNamespace(
                arm=run.arm,
                run_id=run.run_id,
                condition_id=run.arm,
                max_steps=1358,
                checkpoint_path=checkpoint,
                config_path=ROOT / run.config,
                config_sha256=run.config_sha256,
                model_id="d360m",
                raw_token_count=7_120_879_616,
                route_dose_sha256=_digest(
                    f"route-dose:{run.arm}".encode("ascii")
                ),
            )
        )
    publication = publish_terminal_pair(
        SimpleNamespace(
            seed=manifest.seed,
            profile=profile,
            runtime=runtime,
            scratch_root=scratch,
            corpus_receipt_sha256=manifest.dataset_sha256,
            ordered_stream_sha256="c" * 64,
            release_members_sha256="d" * 64,
            code_commit=manifest.source_commit,
            release_sha256=manifest.release_sha256,
            arms=tuple(launches),
            launch_manifest={
                "schema_version": 3,
                "run_manifest_sha256": manifest.sha256,
                "cohort_assignment_sha256": (
                    manifest.cohort_assignment_sha256
                ),
                "preregistration_sha256": manifest.preregistration_sha256,
                "hardware_amendment_sha256": (
                    manifest.hardware_amendment_sha256
                ),
                "provider_selection_sha256": (
                    manifest.provider_selection_sha256
                ),
                "sealed_fixture_sha256": manifest.sealed_fixture_sha256,
            },
        ),
        object_store=object_store,
        operation_id=_digest(
            f"terminal-operation:{manifest.seed}".encode("ascii")
        ),
    )
    return scratch, publication


def test_mocked_v3_terminal_evaluate_collect_advance_lifecycle(
    tmp_path,
    monkeypatch,
):
    import msctl.aws_p5 as aws_p5

    profile, selection, manifest_paths = _manifests(tmp_path)
    sealed = load_sealed_evaluation_release(_sealed_release(tmp_path))
    assert sealed.fixture_sha256 is not None
    for path in manifest_paths:
        value = json.loads(path.read_bytes())
        value["sealed_fixture_sha256"] = sealed.fixture_sha256
        path.write_bytes(canonical_json(value) + b"\n")
    instance_id = "i-0123456789abcdef0"
    plan_value = create_fleet_plan(
        profile=profile,
        selection=selection,
        manifest_paths=manifest_paths,
        instance_ids=[instance_id],
        repo_root=ROOT,
    )
    fleet_plan = validate_fleet_plan(
        plan_value,
        profile=profile,
        selection=selection,
        manifest_paths=manifest_paths,
        repo_root=ROOT,
    )
    manifest = load_run_manifest(manifest_paths[0], repo_root=ROOT)
    target = load_run_manifest(manifest_paths[1], repo_root=ROOT)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    runtime = _runtime(selection)
    launches = []
    for run in sorted(manifest.runs, key=lambda item: item.arm):
        run_root = scratch / "runs" / "seed-0" / run.arm / "run"
        run_root.mkdir(parents=True)
        checkpoint_payload = f"terminal:{run.run_id}".encode("ascii")
        checkpoint = run_root / "ckpt.pt"
        checkpoint.write_bytes(checkpoint_payload)
        checkpoint_sha256 = _digest(checkpoint_payload)
        metadata = {
            "schema_version": 1,
            "receipt_type": "memorysplit-training-checkpoint-v1",
            "run_id": run.run_id,
            "condition": run.arm,
            "seed": 0,
            "step": 1358,
            "max_steps": 1358,
            "world_size": 4,
            "config_fingerprint": run.config_sha256,
            "checkpoint_path": "ckpt.pt",
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_bytes": len(checkpoint_payload),
            "terminal": True,
        }
        (run_root / "checkpoint-meta.json").write_bytes(
            canonical_json(metadata) + b"\n"
        )
        launches.append(
            SimpleNamespace(
                arm=run.arm,
                run_id=run.run_id,
                condition_id=run.arm,
                max_steps=1358,
                checkpoint_path=checkpoint,
                config_path=ROOT / run.config,
                config_sha256=run.config_sha256,
                model_id="d360m",
                raw_token_count=7_120_879_616,
                route_dose_sha256=(
                    "b" * 64 if run.arm == "dense" else "e" * 64
                ),
            )
        )
    publication_plan = SimpleNamespace(
        seed=0,
        profile=profile,
        runtime=runtime,
        scratch_root=scratch,
        corpus_receipt_sha256=manifest.dataset_sha256,
        ordered_stream_sha256="c" * 64,
        release_members_sha256="d" * 64,
        code_commit=manifest.source_commit,
        release_sha256=manifest.release_sha256,
        arms=tuple(launches),
        launch_manifest={
            "schema_version": 3,
            "run_manifest_sha256": manifest.sha256,
            "cohort_assignment_sha256": manifest.cohort_assignment_sha256,
            "preregistration_sha256": manifest.preregistration_sha256,
            "hardware_amendment_sha256": (
                manifest.hardware_amendment_sha256
            ),
            "provider_selection_sha256": (
                manifest.provider_selection_sha256
            ),
            "sealed_fixture_sha256": manifest.sealed_fixture_sha256,
        },
    )
    object_store = _MemoryObjectStore()
    terminal = publish_terminal_pair(
        publication_plan,
        object_store=object_store,
        operation_id="9" * 64,
    )
    durable = verify_durable_terminal(
        receipt_path=terminal.receipt_path,
        run_root=scratch / "runs" / "seed-0",
        s3_root=runtime.s3_root,
        object_store=object_store,
        expected={
            "provider": profile.provider,
            "run_manifest_sha256": manifest.sha256,
            "sealed_fixture_sha256": manifest.sealed_fixture_sha256,
        },
    )
    assert durable["sha256"] == terminal.receipt_sha256
    for launch in launches:
        binding = json.loads(
            (launch.checkpoint_path.parent / "run.json").read_text()
        )
        assert binding["checkpoint_sha256"] == _digest(
            launch.checkpoint_path.read_bytes()
        )
        assert binding["configuration_path"] == "configuration.yaml"
        assert binding["route_dose_sha256"] == launch.route_dose_sha256
        assert binding["corpus_sha256"] == publication_plan.ordered_stream_sha256
        assert binding["code_sha256"] == publication_plan.release_members_sha256
        record = json.loads(
            terminal.checkpoint_record_paths[launch.arm].read_bytes()
        )
        assert record["checkpoint_sha256"] == binding["checkpoint_sha256"]
        assert record["model_id"] == "d360m"
        assert record["raw_token_count"] == 7_120_879_616
        assert record["arm"] == (
            "dense" if launch.arm == "dense" else "split"
        )
        assert object_store.objects[
            terminal.checkpoint_record_uris[launch.arm]
        ] == terminal.checkpoint_record_paths[launch.arm].read_bytes()

    evaluation_root = scratch / "evaluations"
    for launch in launches:
        output = evaluation_root / launch.run_id
        output.mkdir(parents=True)
        for name in EVALUATION_ARTIFACTS:
            payload = (
                (sealed.root / name).read_bytes()
                if name in REQUIRED_SEALED_MEMBERS
                else f"{launch.arm}:{name}\n".encode("ascii")
            )
            (output / name).write_bytes(payload)
    evaluation = publish_evaluation_pair(
        evaluation_root=evaluation_root,
        checkpoint_receipt_path=terminal.receipt_path,
        s3_root=runtime.s3_root,
        operation_id="7" * 64,
        sealed_evaluation_sha256=sealed.sha256,
        study_lock_sha256=sealed.study_lock_sha256,
        fleet_plan_sha256=fleet_plan.sha256,
        fleet_wave=0,
        launch_readiness_sha256="8" * 64,
        object_store=object_store,
        expected={
            "provider": profile.provider,
            "run_manifest_sha256": manifest.sha256,
            "sealed_fixture_sha256": manifest.sealed_fixture_sha256,
        },
    )
    runner = _MemoryAwsRunner(object_store.objects)
    backend = _backend(profile, selection, tmp_path, runner)
    collection_root = tmp_path / "collection"
    collected = backend.collect(
        source="results/seed-0.json",
        out=collection_root,
        apply=True,
    )
    assert collected["collected"] == 2 + 2 * len(EVALUATION_ARTIFACTS)
    assert verify_fleet_collection(
        collection_root,
        manifest=manifest,
    ) == collected["collection_receipt_sha256"]
    collection = backend._verify_v3_lifecycle_collection(
        collection_root,
        manifest=manifest,
        fleet_plan_sha256=fleet_plan.sha256,
        fleet_wave=0,
    )
    assert collection["evaluation_receipt_sha256"] == evaluation.receipt_sha256
    assert (
        backend.collect(
            source="results/seed-0.json",
            out=collection_root,
            apply=True,
        )["idempotent"]
        is True
    )

    selection_path = tmp_path / "selection.json"
    plan_path = tmp_path / "fleet-plan.json"
    write_provider_selection(selection_path, selection.value)
    write_fleet_plan(plan_path, plan_value)
    terminate_at = "2026-07-25T00:00:00Z"
    common = {
        "status": "Success",
        "command_id": "training-command-12345678",
        "operation_id": "1" * 64,
        "intent_sha256": "2" * 64,
        "launch_readiness_sha256": "8" * 64,
        "provider": profile.provider,
        "instance_type": profile.instance_type,
        "seed": 0,
        "release_sha256": manifest.release_sha256,
        "dataset_sha256": manifest.dataset_sha256,
        "run_manifest_sha256": manifest.sha256,
        "profile_sha256": manifest.profile_sha256,
        "runtime_sha256": "4" * 64,
        "container_digest": selection.container_digest,
        "gres": profile.gres,
        "terminate_at": terminate_at,
        "preregistration_sha256": manifest.preregistration_sha256,
        "hardware_amendment_sha256": manifest.hardware_amendment_sha256,
        "provider_selection_sha256": manifest.provider_selection_sha256,
        "sealed_fixture_sha256": manifest.sealed_fixture_sha256,
        "fleet_plan_sha256": fleet_plan.sha256,
        "fleet_wave": 0,
        "control_bundle_sha256": backend.control_bundle.sha256,
    }
    states = {
        run.run_id: {**common, "run_id": run.run_id, "arm": run.arm}
        for run in manifest.runs
    }
    class Store:
        def locked(self):
            return nullcontext()

        def read_run(self, run_id):
            return dict(states[run_id])

    monkeypatch.setattr(aws_p5, "StateStore", lambda _root: Store())
    monkeypatch.setattr(backend, "_repair_paired_states", lambda *_args: None)
    monkeypatch.setattr(
        backend,
        "_same_state_binding",
        lambda *_args, **_kwargs: True,
    )
    advanced = backend.fleet_advance(
        amendment_path=AMENDMENT,
        provider_selection_path=selection_path,
        fleet_plan_path=plan_path,
        target_manifest_path=manifest_paths[1],
        repo_root=ROOT,
        instance_id=instance_id,
        checkpoint_receipt_path=terminal.receipt_path,
        approval_path=None,
        apply=False,
    )
    assert advanced["from_seed"] == manifest.seed
    assert advanced["to_seed"] == target.seed
    assert advanced["resources"]["checkpoint_receipt_sha256"] == (
        terminal.receipt_sha256
    )
    for field, digest in (
        ("sealed_evaluation_sha256", "f" * 64),
        ("study_lock_sha256", "6" * 64),
    ):
        wrong_final = tmp_path / f"collection-wrong-{field}"
        shutil.copytree(collection_root, wrong_final)
        wrong_evaluation = json.loads(
            (wrong_final / "EVALUATION.json").read_bytes()
        )
        wrong_evaluation[field] = digest
        wrong_evaluation_payload = canonical_json(wrong_evaluation) + b"\n"
        (wrong_final / "EVALUATION.json").write_bytes(wrong_evaluation_payload)
        wrong_collection = json.loads(
            (wrong_final / "COLLECTION.json").read_bytes()
        )
        evaluation_row = next(
            row
            for row in wrong_collection["files"]
            if row["path"] == "EVALUATION.json"
        )
        evaluation_row["bytes"] = len(wrong_evaluation_payload)
        evaluation_row["sha256"] = _digest(wrong_evaluation_payload)
        (wrong_final / "COLLECTION.json").write_bytes(
            canonical_json(wrong_collection) + b"\n"
        )
        with pytest.raises(Exception) as wrong_final_error:
            backend._verify_v3_lifecycle_collection(
                wrong_final,
                manifest=manifest,
                fleet_plan_sha256=fleet_plan.sha256,
                fleet_wave=0,
            )
        assert getattr(wrong_final_error.value, "code", None) == (
            "FLEET_COLLECTION_INVALID"
        )
    (collection_root / "unexpected-link").symlink_to(
        collection_root / "CHECKPOINT.json"
    )
    with pytest.raises(Exception) as invalid_collection:
        backend._verify_v3_lifecycle_collection(
            collection_root,
            manifest=manifest,
            fleet_plan_sha256=fleet_plan.sha256,
            fleet_wave=0,
        )
    assert getattr(invalid_collection.value, "code", None) == (
        "FLEET_COLLECTION_INVALID"
    )


@pytest.mark.parametrize("instance_count", [1, 2, 4])
def test_mocked_all_v3_waves_finalize_then_evaluate_and_collect(
    tmp_path,
    instance_count,
):
    from evals.confirmatory.contracts import canonical_json_bytes
    from evals.confirmatory.fixtures import positive_fixture
    from msctl.aws_readiness import LaunchReadiness
    from msctl.aws_sealed_finalization import finalize_sealed_evaluation

    profile, selection, manifest_paths = _manifests(tmp_path)
    fixture = load_sealed_evaluation_fixture(_sealed_fixture(tmp_path))
    for path in manifest_paths:
        value = json.loads(path.read_bytes())
        value["sealed_fixture_sha256"] = fixture.sha256
        path.write_bytes(canonical_json(value) + b"\n")
    instance_ids = [
        f"i-{index + 1:017x}" for index in range(instance_count)
    ]
    plan = validate_fleet_plan(
        create_fleet_plan(
            profile=profile,
            selection=selection,
            manifest_paths=manifest_paths,
            instance_ids=instance_ids,
            repo_root=ROOT,
        ),
        profile=profile,
        selection=selection,
        manifest_paths=manifest_paths,
        repo_root=ROOT,
    )
    runtime = _runtime(selection)
    object_store = _MemoryObjectStore()
    runner = _MemoryAwsRunner(object_store.objects)
    backend = _backend(profile, selection, tmp_path, runner)
    amendment = load_hardware_amendment(AMENDMENT)
    terminal: dict[int, dict[str, object]] = {}
    checkpoint_records: list[Path] = []
    advance_count = 0
    maximum_wave = max(binding.wave for binding in plan.manifests)

    for wave in range(maximum_wave + 1):
        wave_bindings = sorted(
            (
                binding
                for binding in plan.manifests
                if binding.wave == wave
            ),
            key=lambda binding: binding.instance_id,
        )
        if wave:
            for index, target_binding in enumerate(wave_bindings):
                target = load_run_manifest(
                    target_binding.path,
                    repo_root=ROOT,
                )
                source_binding, checked_target = fleet_transition_for_target(
                    plan,
                    instance_id=target_binding.instance_id,
                    target=target,
                )
                assert checked_target == target_binding
                source = terminal[source_binding.seed]
                identity = source["identity"]
                assert isinstance(identity, dict)
                evidence = {
                    "training_state_sha256": _digest(
                        f"training-state:{source_binding.seed}".encode(
                            "ascii"
                        )
                    ),
                    "training_command_id": (
                        f"training-command-{source_binding.seed:08d}"
                    ),
                    "training_terminal_receipt_uri": (
                        f"{runtime.s3_root}/operations/"
                        f"{source_binding.sha256}/receipts/terminal.json"
                    ),
                    "checkpoint_receipt_sha256": identity["sha256"],
                    "checkpoint_receipt_uri": identity["uri"],
                    "checkpoint_records": identity["records"],
                    "aws_bound_tags_sha256": _digest(
                        f"bound-tags:{source_binding.seed}".encode("ascii")
                    ),
                    "aws_unbound_tags_sha256": _digest(
                        f"unbound-tags:{source_binding.seed}".encode("ascii")
                    ),
                }
                assert not {
                    "evaluation_state_sha256",
                    "evaluation_receipt_sha256",
                    "collection_receipt_sha256",
                } & set(evidence)
                value = create_fleet_advance(
                    plan=plan,
                    instance_id=target_binding.instance_id,
                    from_binding=source_binding,
                    to_binding=target_binding,
                    evidence=evidence,
                    approval_sha256=_digest(
                        f"advance-approval:{source_binding.seed}".encode(
                            "ascii"
                        )
                    ),
                    advanced_at=(
                        f"2026-07-24T02:{wave:02d}:{index:02d}Z"
                    ),
                )
                write_fleet_advance(
                    backend.state_root,
                    plan=plan,
                    to_binding=target_binding,
                    value=value,
                )
                advance_count += 1
                if index + 1 < len(wave_bindings):
                    first = wave_bindings[0]
                    first_manifest = load_run_manifest(
                        first.path,
                        repo_root=ROOT,
                    )
                    first_context = V3LifecycleContext(
                        amendment=amendment,
                        selection=selection,
                        fleet_plan=plan,
                        fleet_binding=first,
                    )
                    with pytest.raises(Exception) as peer_gate:
                        backend._require_fleet_advance(
                            first_manifest,
                            first_context,
                        )
                    assert getattr(peer_gate.value, "code", None) == (
                        "FLEET_ADVANCE_REQUIRED"
                    )
                    assert len(peer_gate.value.details["missing"]) == (
                        len(wave_bindings) - index - 1
                    )
            for target_binding in wave_bindings:
                target = load_run_manifest(
                    target_binding.path,
                    repo_root=ROOT,
                )
                backend._require_fleet_advance(
                    target,
                    V3LifecycleContext(
                        amendment=amendment,
                        selection=selection,
                        fleet_plan=plan,
                        fleet_binding=target_binding,
                    ),
                )

        assert not any(
            "/evaluations/" in uri or "/results/" in uri
            for uri in object_store.objects
        )
        for binding in wave_bindings:
            manifest = load_run_manifest(binding.path, repo_root=ROOT)
            scratch, publication = _publish_terminal_manifest(
                manifest=manifest,
                profile=profile,
                runtime=runtime,
                root=tmp_path,
                object_store=object_store,
            )
            operator_receipt = (
                tmp_path
                / "operator-receipts"
                / f"checkpoint-seed-{manifest.seed}.json"
            )
            operator_receipt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(publication.receipt_path, operator_receipt)
            copied_records = []
            for arm in ("dense", "split90"):
                record = (
                    tmp_path
                    / "finalization-records"
                    / f"seed-{manifest.seed}-{arm}.json"
                )
                record.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(
                    publication.checkpoint_record_paths[arm],
                    record,
                )
                checkpoint_records.append(record)
                copied_records.append(record)
            shutil.rmtree(scratch)
            assert not (operator_receipt.parent / "dense.pt").exists()
            identity = backend._terminal_checkpoint_identity(
                operator_receipt,
                manifest=manifest,
            )
            rematerialized = (
                backend._materialize_terminal_checkpoint_identity(
                    identity,
                    manifest=manifest,
                )
            )
            assert len(rematerialized) == 9
            terminal[manifest.seed] = {
                "identity": identity,
                "receipt": operator_receipt,
                "records": copied_records,
            }

    assert len(terminal) == 10
    assert len(checkpoint_records) == 20
    assert advance_count == 10 - instance_count
    assert not any(
        "/evaluations/" in uri or "/results/" in uri
        for uri in object_store.objects
    )

    validity_root = tmp_path / "validity-receipts"
    validity_root.mkdir()
    validity = json.loads(positive_fixture().artifacts["validity.json"])
    validity_receipts = []
    for index, receipt in enumerate(validity["receipts"]):
        path = validity_root / f"{index:02d}.json"
        path.write_bytes(canonical_json_bytes(receipt))
        validity_receipts.append(path)
    first_manifest = load_run_manifest(manifest_paths[0], repo_root=ROOT)
    finalized_root = tmp_path / "finalized-sealed-evaluation"
    finalized_result = finalize_sealed_evaluation(
        fixture_root=fixture.root,
        checkpoint_records=checkpoint_records,
        validity_receipts=validity_receipts,
        preregistration_sha256=first_manifest.preregistration_sha256,
        out=finalized_root,
        apply=True,
    )
    assert finalized_result["published"] is True
    finalized = load_sealed_evaluation_release(finalized_root)
    assert finalized.fixture_sha256 == fixture.sha256

    readiness = LaunchReadiness(
        profile_id=profile.profile_id,
        sha256="8" * 64,
        bindings={
            "release_sha256": first_manifest.release_sha256,
            "provider_selection_sha256": selection.sha256,
            "hardware_amendment_sha256": amendment.sha256,
            "sealed_fixture_sha256": fixture.sha256,
            "environment_receipt_sha256": "7" * 64,
        },
        decision={"protected_launch_allowed": True},
        path=None,
        value={},
    )
    release = SimpleNamespace(
        provider=profile.provider,
        archive_sha256=first_manifest.release_sha256,
        receipt_sha256="9" * 64,
        members_sha256="a" * 64,
        source_commit=first_manifest.source_commit,
    )
    collected_seeds = []
    for manifest_path in manifest_paths:
        manifest = load_run_manifest(manifest_path, repo_root=ROOT)
        binding = plan.binding_for_seed(manifest.seed)
        checkpoint = verify_checkpoint_receipt(
            terminal[manifest.seed]["receipt"],
            release=release,
            manifest=manifest,
            require_checkpoint_files=False,
            require_durable_terminal=True,
        )
        context = V3LifecycleContext(
            amendment=amendment,
            selection=selection,
            fleet_plan=plan,
            fleet_binding=binding,
            readiness=readiness,
            sealed_evaluation=finalized,
        )
        terminate_at = "2097-12-31T23:00:00Z"
        submit_core = backend._training_operation_intent(
            operation="submit",
            release=release,
            manifest=manifest,
            terminate_at=terminate_at,
            evidence={
                "dataset_pointer_sha256": "5" * 64,
                "dataset_verification_sha256": "6" * 64,
                "environment_receipt_sha256": "7" * 64,
            },
            context=context,
        )
        submit_intent = backend._operation_envelope(
            submit_core,
            instance_id=binding.instance_id,
            terminate_at=terminate_at,
        )
        submit_intent_sha256 = _digest(canonical_json(submit_intent))
        persisted = [
            backend._new_aws_run_state(
                run=run,
                manifest=manifest,
                operation="submit",
                instance_id=binding.instance_id,
                terminate_at=terminate_at,
                intent=submit_intent,
                published={
                    "intent_sha256": submit_intent_sha256,
                    "intent_uri": (
                        f"{runtime.s3_root}/operations/intents/sha256/"
                        f"{submit_intent_sha256}.json"
                    ),
                },
                attempt=1,
                context=context,
            )
            for run in manifest.runs
        ]
        for state in persisted:
            state["command_id"] = f"training-command-{manifest.seed:08d}"
            state["status"] = "Success"
            state["send_attempted"] = True
        from msctl.state import StateStore

        store = StateStore(backend.state_root)
        with store.locked():
            backend._write_paired_states(store, manifest, persisted, context)

        evaluation_plan = backend.evaluate(
            release=release,
            manifest=manifest,
            approval_path=None,
            apply=False,
            evidence={
                "dataset_pointer_sha256": "5" * 64,
                "dataset_verification_sha256": "6" * 64,
                "environment_receipt_sha256": "7" * 64,
            },
            context=context,
            checkpoint_receipt=checkpoint,
        )
        assert evaluation_plan["approval_resources"]["instance_id"] == (
            binding.instance_id
        )
        assert evaluation_plan["approval_resources"]["terminate_at"] == (
            terminate_at
        )
        assert evaluation_plan["approval_resources"][
            "checkpoint_receipt_sha256"
        ] == checkpoint.sha256
        terminal_steps = [
            step
            for step in evaluation_plan["operation_intent"]["steps"]
            if step["name"].startswith("materialize-terminal-")
        ]
        assert len(terminal_steps) == 9
        assert all(
            "get-object" in step["argv"]
            and "--checksum-mode" in step["argv"]
            and "run-instances" not in step["argv"]
            for step in terminal_steps
        )

        evaluation_root = tmp_path / "evaluations" / f"seed-{manifest.seed}"
        for run in manifest.runs:
            output = evaluation_root / run.run_id
            output.mkdir(parents=True)
            for name in EVALUATION_ARTIFACTS:
                payload = (
                    (finalized.root / name).read_bytes()
                    if name in REQUIRED_SEALED_MEMBERS
                    else f"{run.arm}:{name}\n".encode("ascii")
                )
                (output / name).write_bytes(payload)
        publish_evaluation_pair(
            evaluation_root=evaluation_root,
            checkpoint_receipt_path=terminal[manifest.seed]["receipt"],
            s3_root=runtime.s3_root,
            operation_id=_digest(
                f"evaluation-operation:{manifest.seed}".encode("ascii")
            ),
            sealed_evaluation_sha256=finalized.sha256,
            study_lock_sha256=finalized.study_lock_sha256,
            fleet_plan_sha256=plan.sha256,
            fleet_wave=binding.wave,
            launch_readiness_sha256=readiness.sha256,
            object_store=object_store,
            expected={
                "provider": profile.provider,
                "run_manifest_sha256": manifest.sha256,
                "sealed_fixture_sha256": fixture.sha256,
            },
        )
        collection_root = (
            tmp_path / "collections" / f"seed-{manifest.seed}"
        )
        collected = backend.collect(
            source=f"results/seed-{manifest.seed}.json",
            out=collection_root,
            apply=True,
        )
        assert collected["collected"] == 2 + 2 * len(EVALUATION_ARTIFACTS)
        assert verify_fleet_collection(
            collection_root,
            manifest=manifest,
        )
        lifecycle_collection = backend._verify_v3_lifecycle_collection(
            collection_root,
            manifest=manifest,
            fleet_plan_sha256=plan.sha256,
            fleet_wave=binding.wave,
        )
        assert lifecycle_collection["evaluation"][
            "sealed_evaluation_sha256"
        ] == (
            finalized.sha256
        )
        collected_seeds.append(manifest.seed)

    assert collected_seeds == list(range(10))
    assert all(
        binding.instance_id in instance_ids
        for binding in plan.manifests
    )
    assert profile.provider.endswith("-v3")
    assert all("run-instances" not in call for call in runner.calls)


def test_evaluation_uses_historical_advance_after_training_tags_are_removed(
    tmp_path,
):
    import msctl.aws_p5 as aws_p5
    from msctl.aws_readiness import LaunchReadiness
    from msctl.aws_sealed_evaluation import SealedEvaluationFixture

    profile, selection, manifest_paths = _manifests(tmp_path)
    instance_id = "i-0123456789abcdef0"
    plan = validate_fleet_plan(
        create_fleet_plan(
            profile=profile,
            selection=selection,
            manifest_paths=manifest_paths,
            instance_ids=[instance_id],
            repo_root=ROOT,
        ),
        profile=profile,
        selection=selection,
        manifest_paths=manifest_paths,
        repo_root=ROOT,
    )

    class Runner:
        row = None

        def run_json(self, argv, *, operation):
            del operation
            if "describe-instances" in argv:
                return {"instances": [dict(self.row)]}
            if "describe-instance-attribute" in argv:
                return {
                    "attribute": {
                        "instance_id": instance_id,
                        "shutdown_behavior": "terminate",
                    }
                }
            pytest.fail(f"unexpected AWS operation: {argv}")

    runner = Runner()
    backend = _backend(profile, selection, tmp_path, runner)
    manifest = load_run_manifest(manifest_paths[0], repo_root=ROOT)
    target = load_run_manifest(manifest_paths[1], repo_root=ROOT)
    source_binding, target_binding = fleet_transition_for_target(
        plan,
        instance_id=instance_id,
        target=target,
    )
    write_fleet_advance(
        backend.state_root,
        plan=plan,
        to_binding=target_binding,
        value=create_fleet_advance(
            plan=plan,
            instance_id=instance_id,
            from_binding=source_binding,
            to_binding=target_binding,
            evidence=_advance_evidence(manifest.seed),
            approval_sha256="8" * 64,
            advanced_at="2026-07-24T02:00:00Z",
        ),
    )
    context = V3LifecycleContext(
        amendment=load_hardware_amendment(AMENDMENT),
        selection=selection,
        fleet_plan=plan,
        fleet_binding=source_binding,
        readiness=LaunchReadiness(
            profile_id=profile.profile_id,
            sha256="8" * 64,
            bindings={
                "release_sha256": manifest.release_sha256,
                "provider_selection_sha256": selection.sha256,
                "hardware_amendment_sha256": (
                    manifest.hardware_amendment_sha256
                ),
                "sealed_fixture_sha256": manifest.sealed_fixture_sha256,
                "environment_receipt_sha256": "7" * 64,
            },
            decision={"protected_launch_allowed": True},
            path=None,
            value={},
        ),
        sealed_fixture=SealedEvaluationFixture(
            root=tmp_path / "sealed-fixture",
            sha256=manifest.sealed_fixture_sha256,
            members={},
        ),
    )
    terminate_at = "2026-07-25T00:00:00Z"
    runner.row = {
        **{
            field: None
            for field in (
                aws_p5._SELECTED_INSTANCE_FIELDS
                | aws_p5._V3_INSTANCE_FIELDS
            )
        },
        "instance_id": instance_id,
        "instance_type": profile.instance_type,
        "state": "running",
        "instance_profile_arn": backend.instance_profile_arn,
        "ami_id": selection.ami_id,
    }

    with pytest.raises(Exception) as strict_binding:
        backend._validate_selected_instance_binding(
            manifest,
            instance_id=instance_id,
            terminate_at=terminate_at,
            operation="resume",
            context=context,
        )
    assert getattr(strict_binding.value, "code", None) == (
        "INSTANCE_BINDING_MISMATCH"
    )

    with pytest.raises(Exception) as wrong_checkpoint:
        backend._validate_evaluation_instance_binding(
            manifest,
            instance_id=instance_id,
            terminate_at=terminate_at,
            checkpoint_receipt_sha256="9" * 64,
            context=context,
        )
    assert getattr(wrong_checkpoint.value, "code", None) == (
        "FLEET_ADVANCE_REQUIRED"
    )

    runner.row["provider"] = "recommitted-arbitrary-binding"
    with pytest.raises(Exception) as conflicting_historical_binding:
        backend._validate_evaluation_instance_binding(
            manifest,
            instance_id=instance_id,
            terminate_at=terminate_at,
            checkpoint_receipt_sha256="4" * 64,
            context=context,
        )
    assert getattr(conflicting_historical_binding.value, "code", None) == (
        "INSTANCE_BINDING_MISMATCH"
    )
    runner.row["provider"] = None

    validated = backend._validate_evaluation_instance_binding(
        manifest,
        instance_id=instance_id,
        terminate_at=terminate_at,
        checkpoint_receipt_sha256="4" * 64,
        context=context,
    )
    assert validated["instance_id"] == instance_id


def test_multi_instance_wave_gate_requires_every_parallel_advance(tmp_path):
    profile, selection, manifest_paths = _manifests(tmp_path)
    instance_ids = (
        "i-0123456789abcdef0",
        "i-1123456789abcdef0",
    )
    plan = validate_fleet_plan(
        create_fleet_plan(
            profile=profile,
            selection=selection,
            manifest_paths=manifest_paths,
            instance_ids=instance_ids,
            repo_root=ROOT,
        ),
        profile=profile,
        selection=selection,
        manifest_paths=manifest_paths,
        repo_root=ROOT,
    )
    backend = _backend(
        profile,
        selection,
        tmp_path,
        SimpleNamespace(
            run_json=lambda *_args, **_kwargs: pytest.fail(
                "local wave admission must not call AWS"
            )
        ),
    )
    amendment = load_hardware_amendment(AMENDMENT)
    targets = []
    for seed in (2, 3):
        target = load_run_manifest(manifest_paths[seed], repo_root=ROOT)
        source, target_binding = fleet_transition_for_target(
            plan,
            instance_id=target.binding_instance
            if hasattr(target, "binding_instance")
            else plan.binding_for_seed(seed).instance_id,
            target=target,
        )
        targets.append((target, source, target_binding))
    for index, (_target, source, target_binding) in enumerate(targets):
        value = create_fleet_advance(
            plan=plan,
            instance_id=target_binding.instance_id,
            from_binding=source,
            to_binding=target_binding,
            evidence=_advance_evidence(source.seed),
            approval_sha256="8" * 64,
            advanced_at=f"2026-07-24T02:00:0{index}Z",
        )
        if index == 0:
            write_fleet_advance(
                backend.state_root,
                plan=plan,
                to_binding=target_binding,
                value=value,
            )
            context = V3LifecycleContext(
                amendment=amendment,
                selection=selection,
                fleet_plan=plan,
                fleet_binding=target_binding,
            )
            with pytest.raises(Exception) as incomplete:
                backend._require_fleet_advance(targets[0][0], context)
            assert getattr(incomplete.value, "code", None) == (
                "FLEET_ADVANCE_REQUIRED"
            )
            assert incomplete.value.details["missing"] == [
                {
                    "instance_id": targets[1][2].instance_id,
                    "seed": targets[1][2].seed,
                    "wave": 1,
                }
            ]
        else:
            write_fleet_advance(
                backend.state_root,
                plan=plan,
                to_binding=target_binding,
                value=value,
            )
    context = V3LifecycleContext(
        amendment=amendment,
        selection=selection,
        fleet_plan=plan,
        fleet_binding=targets[0][2],
    )
    backend._require_fleet_advance(targets[0][0], context)


def test_terminal_snapshot_refuses_linked_sources_and_parents(tmp_path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"checkpoint")
    linked_source = tmp_path / "linked-source.pt"
    linked_source.symlink_to(source)

    with pytest.raises(ValueError, match="unsafe|regular"):
        _snapshot_regular(
            linked_source,
            tmp_path / "snapshot.pt",
            label="terminal checkpoint",
        )
    assert not (tmp_path / "snapshot.pt").exists()

    destination_root = tmp_path / "destination"
    real_destination_root = tmp_path / "real-destination"
    real_destination_root.mkdir()
    destination_root.symlink_to(real_destination_root, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        _snapshot_regular(
            source,
            destination_root / "snapshot.pt",
            label="terminal checkpoint",
        )
    assert not (real_destination_root / "snapshot.pt").exists()

    hard_link = tmp_path / "hard-link.pt"
    hard_link.hardlink_to(source)
    with pytest.raises(ValueError, match="regular"):
        _snapshot_regular(
            source,
            tmp_path / "hard-linked-snapshot.pt",
            label="terminal checkpoint",
        )
