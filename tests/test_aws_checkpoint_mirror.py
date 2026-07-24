from __future__ import annotations

import json
from pathlib import Path
import base64
import copy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Mapping

import pytest

from cluster.aws.p5 import checkpoint_mirror as checkpoint_mirror_module
from cluster.aws.p5.checkpoint_mirror import (
    CheckpointMirrorScheduler,
    CheckpointMirrorRequest,
    CheckpointStaleError,
    CheckpointReceiptRef,
    PublishedCheckpointPair,
    S3VersionedObjectStore,
    VersionedUploadedObject,
    publish_paired_checkpoint,
    read_trainer_checkpoint_metadata,
)
from msctl import aws_resume_launch as aws_resume_launch_module
from msctl.aws_contracts import checkpoint_object_key, checkpoint_receipt_key
from msctl.contracts import (
    parse_paired_checkpoint_receipt_v3,
    verify_checkpoint_receipt,
    verify_aws_checkpoint_receipt_v3,
)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_metadata(
    checkpoint: Path,
    *,
    step: int,
    world_size: int = 4,
    sidecar_name: str = "dense_target_weights",
    receipt_sha256: str = "d" * 64,
    build_id: str = "b" * 64,
    ordered_stream_sha256: str = "e" * 64,
) -> dict[str, object]:
    installed = checkpoint.stat(follow_symlinks=False)
    value = {
        "checkpoint_version": 3,
        "config_fingerprint": "f" * 64,
        "data": {
            "build_id": build_id,
            "global_cursor": step * 524_288,
            "ordered_stream_sha256": ordered_stream_sha256,
            "receipt_sha256": receipt_sha256,
            "sidecar_name": sidecar_name,
        },
        "installed": {
            "bytes": installed.st_size,
            "ctime_ns": installed.st_ctime_ns,
            "device": installed.st_dev,
            "gid": installed.st_gid,
            "inode": installed.st_ino,
            "links": installed.st_nlink,
            "mode": installed.st_mode,
            "mtime_ns": installed.st_mtime_ns,
            "uid": installed.st_uid,
        },
        "receipt_type": "memorysplit-trainer-checkpoint-v1",
        "schema_version": 1,
        "step": step,
        "world_size": world_size,
    }
    checkpoint.with_name("ckpt.meta.json").write_bytes(_canonical_json(value))
    return value


def test_explicit_legacy_metadata_absence_is_readable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"legacy-checkpoint")

    assert (
        read_trainer_checkpoint_metadata(
            checkpoint,
            allow_legacy_absent=True,
        )
        is None
    )
    with pytest.raises(ValueError, match="metadata.*missing"):
        read_trainer_checkpoint_metadata(checkpoint)


def test_trainer_metadata_reads_the_exact_installed_generation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"checkpoint-generation")
    expected = _write_metadata(checkpoint, step=17)

    actual = read_trainer_checkpoint_metadata(checkpoint)

    assert actual is not None
    assert actual.checkpoint_version == expected["checkpoint_version"]
    assert actual.step == expected["step"]
    assert actual.world_size == expected["world_size"]
    assert actual.config_fingerprint == expected["config_fingerprint"]
    assert actual.data == expected["data"]
    assert actual.installed == expected["installed"]


def test_present_mismatched_trainer_metadata_fails(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"first-generation")
    _write_metadata(checkpoint, step=1)
    checkpoint.write_bytes(b"replacement-generation")

    with pytest.raises(ValueError, match="does not match installed"):
        read_trainer_checkpoint_metadata(
            checkpoint,
            allow_legacy_absent=True,
        )


class _MemoryVersionedStore:
    def __init__(self) -> None:
        self.objects: dict[
            str, tuple[bytes, str, Mapping[str, str], str]
        ] = {}
        self.put_order: list[str] = []

    def put_if_absent(
        self,
        path: Path,
        uri: str,
        *,
        sha256: str,
        byte_count: int,
        metadata: Mapping[str, str],
    ) -> VersionedUploadedObject | None:
        payload = path.read_bytes()
        assert len(payload) == byte_count
        assert uri not in self.objects
        version_id = f"version-{len(self.objects) + 1}"
        self.objects[uri] = (payload, sha256, dict(metadata), version_id)
        self.put_order.append(uri)
        return VersionedUploadedObject(
            uri=uri,
            sha256=sha256,
            bytes=byte_count,
            version_id=version_id,
        )

    def head_exact(
        self,
        uri: str,
        *,
        sha256: str,
        byte_count: int,
        metadata: Mapping[str, str],
        version_id: str | None,
    ) -> VersionedUploadedObject | None:
        payload, stored_sha256, stored_metadata, stored_version = self.objects[uri]
        if (
            stored_sha256 != sha256
            or len(payload) != byte_count
            or stored_metadata != dict(metadata)
            or (version_id is not None and stored_version != version_id)
        ):
            return None
        return VersionedUploadedObject(
            uri=uri,
            sha256=stored_sha256,
            bytes=len(payload),
            version_id=stored_version,
        )


class _LostFirstPutStore(_MemoryVersionedStore):
    def __init__(self) -> None:
        super().__init__()
        self.lost = True

    def put_if_absent(self, *args, **kwargs):
        uploaded = super().put_if_absent(*args, **kwargs)
        if self.lost:
            self.lost = False
            return None
        return uploaded


class _FaultyVersionedStore(_MemoryVersionedStore):
    def __init__(self, fault: str) -> None:
        super().__init__()
        self.fault = fault

    def put_if_absent(self, path, uri, **kwargs):
        if self.fault == "upload" and "/split90/" in uri:
            return None
        uploaded = super().put_if_absent(path, uri, **kwargs)
        if "/split90/" in uri and self.fault in {
            "missing-version",
            "null-version",
        }:
            return VersionedUploadedObject(
                uri=uploaded.uri,
                sha256=uploaded.sha256,
                bytes=uploaded.bytes,
                version_id=(
                    "" if self.fault == "missing-version" else "null"
                ),
            )
        return uploaded

    def head_exact(self, uri, **kwargs):
        if "/split90/" in uri and self.fault in {"upload", "head"}:
            return None
        verified = super().head_exact(uri, **kwargs)
        if (
            verified is not None
            and "/split90/" in uri
            and self.fault == "changed-version"
        ):
            return VersionedUploadedObject(
                uri=verified.uri,
                sha256=verified.sha256,
                bytes=verified.bytes,
                version_id="changed-version",
            )
        return verified


def _request(tmp_path: Path) -> CheckpointMirrorRequest:
    checkpoint_paths = {
        arm: tmp_path / arm / "ckpt.pt"
        for arm in ("dense", "split90")
    }
    for arm, path in checkpoint_paths.items():
        path.parent.mkdir()
        path.write_bytes(f"baseline-{arm}".encode("ascii"))
        _write_metadata(
            path,
            step=1,
            sidecar_name=f"{arm}_target_weights",
        )
    return CheckpointMirrorRequest(
        seed=0,
        reason="periodic",
        request_id="a" * 32,
        requested_at="2026-07-23T12:00:00Z",
        deadline_at="2026-07-23T12:20:00Z",
        instance_id="i-0123456789abcdef0",
        boot_id="11111111-2222-4333-8444-555555555555",
        profile_sha256="1" * 64,
        environment_receipt_sha256="2" * 64,
        release_sha256="3" * 64,
        release_receipt_sha256="4" * 64,
        run_manifest_sha256="5" * 64,
        dataset_receipt_sha256="d" * 64,
        dataset_build_id="b" * 64,
        ordered_stream_sha256="e" * 64,
        source_commit="9" * 40,
        source_tree="a" * 40,
        rank_zero_pids={"dense": 101, "split90": 102},
        checkpoint_paths=checkpoint_paths,
        config_sha256={"dense": "b" * 64, "split90": "c" * 64},
        run_ids={
            "dense": "memorysplit-v3-360m-dense-s0",
            "split90": "memorysplit-v3-360m-split90-s0",
        },
        s3_root="s3://memorysplit-test/prefix",
    )


def test_distinct_fresh_arm_steps_publish_one_atomic_pair_receipt(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    store = _MemoryVersionedStore()
    step_by_pid = {101: 5, 102: 7}
    arm_by_pid = {101: "dense", 102: "split90"}
    signals: list[tuple[int, int]] = []

    def signal_process(pid: int, signum: int) -> None:
        signals.append((pid, signum))
        arm = arm_by_pid[pid]
        checkpoint = request.checkpoint_paths[arm]
        checkpoint.write_bytes(f"fresh-{arm}-{step_by_pid[pid]}".encode("ascii"))
        _write_metadata(
            checkpoint,
            step=step_by_pid[pid],
            sidecar_name=f"{arm}_target_weights",
        )

    published = publish_paired_checkpoint(
        request,
        object_store=store,
        signal_process=signal_process,
        staging_root=tmp_path / "staging",
        staged_at=lambda: "2026-07-23T12:01:00Z",
    )

    assert [pid for pid, _signum in signals] == [101, 102]
    assert len(store.put_order) == 3
    assert store.put_order[-1] == published.receipt.uri
    assert published.receipt.uri.endswith(
        f"/receipts/checkpoints/seed-0/sha256/"
        f"{published.receipt.sha256}.json"
    )
    assert [row["arm"] for row in published.value["checkpoints"]] == [
        "dense",
        "split90",
    ]
    assert [row["step"] for row in published.value["checkpoints"]] == [5, 7]
    assert [
        row["data"]["global_cursor"]
        for row in published.value["checkpoints"]
    ] == [5 * 524_288, 7 * 524_288]


def test_both_arms_require_fresh_post_signal_generations(tmp_path: Path) -> None:
    request = _request(tmp_path)
    store = _MemoryVersionedStore()
    now = [0.0]

    def signal_process(pid: int, _signum: int) -> None:
        if pid != 101:
            return
        checkpoint = request.checkpoint_paths["dense"]
        checkpoint.write_bytes(b"fresh-dense")
        _write_metadata(
            checkpoint,
            step=2,
            sidecar_name="dense_target_weights",
        )

    def sleep(seconds: float) -> None:
        now[0] += max(seconds, 0.01)

    with pytest.raises(TimeoutError, match="split90.*fresh"):
        publish_paired_checkpoint(
            request,
            object_store=store,
            signal_process=signal_process,
            staging_root=tmp_path / "staging",
            monotonic=lambda: now[0],
            sleep=sleep,
        )

    assert store.put_order == []


def test_baseline_pins_the_generation_validated_before_signal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    store = _MemoryVersionedStore()
    dense_checkpoint = request.checkpoint_paths["dense"]
    dense_metadata = dense_checkpoint.with_name("ckpt.meta.json")
    original_read_regular = checkpoint_mirror_module._read_regular
    replaced = False

    def replace_before_metadata_validation(path: Path, *, label: str):
        nonlocal replaced
        if Path(path) == dense_metadata and not replaced:
            replaced = True
            dense_checkpoint.write_bytes(b"published-before-baseline-validation")
            _write_metadata(
                dense_checkpoint,
                step=22,
                sidecar_name="dense_target_weights",
            )
        return original_read_regular(path, label=label)

    monkeypatch.setattr(
        checkpoint_mirror_module,
        "_read_regular",
        replace_before_metadata_validation,
    )
    now = [0.0]

    def signal_process(pid: int, _signum: int) -> None:
        if pid != 102:
            return
        split_checkpoint = request.checkpoint_paths["split90"]
        split_checkpoint.write_bytes(b"fresh-split90")
        _write_metadata(
            split_checkpoint,
            step=2,
            sidecar_name="split90_target_weights",
        )

    def sleep(seconds: float) -> None:
        now[0] += max(seconds, 0.01)

    with pytest.raises(TimeoutError, match="dense.*fresh"):
        publish_paired_checkpoint(
            request,
            object_store=store,
            signal_process=signal_process,
            staging_root=tmp_path / "staging",
            monotonic=lambda: now[0],
            sleep=sleep,
            staged_at=lambda: "2026-07-23T12:01:00Z",
        )

    assert replaced is True
    assert store.put_order == []


def test_publisher_recovers_a_lost_put_response_by_exact_head(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    store = _LostFirstPutStore()
    arm_by_pid = {101: "dense", 102: "split90"}

    def signal_process(pid: int, _signum: int) -> None:
        arm = arm_by_pid[pid]
        checkpoint = request.checkpoint_paths[arm]
        checkpoint.write_bytes(f"fresh-{arm}".encode("ascii"))
        _write_metadata(
            checkpoint,
            step=3,
            sidecar_name=f"{arm}_target_weights",
        )

    published = publish_paired_checkpoint(
        request,
        object_store=store,
        signal_process=signal_process,
        staging_root=tmp_path / "staging",
        staged_at=lambda: "2026-07-23T12:01:00Z",
    )

    assert published.receipt.version_id == "version-3"
    assert len(store.objects) == 3


@pytest.mark.parametrize(
    "fault",
    [
        "upload",
        "head",
        "missing-version",
        "null-version",
        "changed-version",
    ],
)
def test_partial_or_unversioned_publication_emits_no_pair_receipt(
    tmp_path: Path,
    fault: str,
) -> None:
    request = _request(tmp_path)
    store = _FaultyVersionedStore(fault)
    arm_by_pid = {101: "dense", 102: "split90"}

    def signal_process(pid: int, _signum: int) -> None:
        arm = arm_by_pid[pid]
        checkpoint = request.checkpoint_paths[arm]
        checkpoint.write_bytes(f"fresh-{arm}".encode("ascii"))
        _write_metadata(
            checkpoint,
            step=4,
            sidecar_name=f"{arm}_target_weights",
        )

    with pytest.raises(ValueError):
        publish_paired_checkpoint(
            request,
            object_store=store,
            signal_process=signal_process,
            staging_root=tmp_path / "staging",
        )

    assert not any(
        "/receipts/checkpoints/" in uri for uri in store.put_order
    )


def test_schema_three_pair_receipt_parser_accepts_publisher_bytes(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    store = _MemoryVersionedStore()
    arm_by_pid = {101: "dense", 102: "split90"}

    def signal_process(pid: int, _signum: int) -> None:
        arm = arm_by_pid[pid]
        checkpoint = request.checkpoint_paths[arm]
        checkpoint.write_bytes(f"fresh-{arm}".encode("ascii"))
        _write_metadata(
            checkpoint,
            step=6 if arm == "dense" else 8,
            sidecar_name=f"{arm}_target_weights",
        )

    published = publish_paired_checkpoint(
        request,
        object_store=store,
        signal_process=signal_process,
        staging_root=tmp_path / "staging",
        staged_at=lambda: "2026-07-23T12:01:00Z",
    )
    receipt_bytes = store.objects[published.receipt.uri][0]

    parsed = parse_paired_checkpoint_receipt_v3(
        receipt_bytes,
        receipt_uri=published.receipt.uri,
        receipt_sha256=published.receipt.sha256,
        receipt_version_id=published.receipt.version_id,
    )

    assert parsed.seed == 0
    assert parsed.request_id == request.request_id
    assert [checkpoint.step for checkpoint in parsed.checkpoints] == [6, 8]
    assert [
        checkpoint.object.version_id for checkpoint in parsed.checkpoints
    ] == ["version-1", "version-2"]


def test_v3_receipt_verifier_binds_every_manifest_provenance(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    store = _MemoryVersionedStore()
    arm_by_pid = {101: "dense", 102: "split90"}

    def signal_process(pid: int, _signum: int) -> None:
        arm = arm_by_pid[pid]
        checkpoint = request.checkpoint_paths[arm]
        checkpoint.write_bytes(f"verified-{arm}".encode("ascii"))
        _write_metadata(
            checkpoint,
            step=9,
            sidecar_name=f"{arm}_target_weights",
        )

    published = publish_paired_checkpoint(
        request,
        object_store=store,
        signal_process=signal_process,
        staging_root=tmp_path / "staging",
        staged_at=lambda: "2026-07-23T12:01:00Z",
    )
    receipt = parse_paired_checkpoint_receipt_v3(
        store.objects[published.receipt.uri][0],
        receipt_uri=published.receipt.uri,
        receipt_sha256=published.receipt.sha256,
        receipt_version_id=published.receipt.version_id,
    )
    release = SimpleNamespace(
        archive_sha256=request.release_sha256,
        receipt_sha256=request.release_receipt_sha256,
        source_commit=request.source_commit,
        source_tree=request.source_tree,
    )
    manifest = SimpleNamespace(
        schema_version=3,
        provider="aws-p5.48xlarge",
        cohort_id="memorysplit-confirmatory-v3-360m-n10-aws",
        seed=0,
        profile_sha256=request.profile_sha256,
        release_sha256=request.release_sha256,
        release_receipt_sha256=request.release_receipt_sha256,
        dataset_receipt_sha256=request.dataset_receipt_sha256,
        dataset_build_id=request.dataset_build_id,
        ordered_stream_sha256=request.ordered_stream_sha256,
        source_commit=request.source_commit,
        source_tree=request.source_tree,
        sha256=request.run_manifest_sha256,
        runs=tuple(
            SimpleNamespace(
                run_id=request.run_ids[arm],
                arm=arm,
                seed=0,
                config_sha256=request.config_sha256[arm],
            )
            for arm in ("dense", "split90")
        ),
    )

    assert (
        verify_aws_checkpoint_receipt_v3(
            receipt,
            release=release,
            manifest=manifest,
            environment_receipt_sha256=request.environment_receipt_sha256,
            instance_id=request.instance_id,
            boot_id=request.boot_id,
        )
        is receipt
    )
    with pytest.raises(Exception, match="provenance|environment"):
        verify_aws_checkpoint_receipt_v3(
            receipt,
            release=release,
            manifest=manifest,
            environment_receipt_sha256="0" * 64,
            instance_id=request.instance_id,
            boot_id=request.boot_id,
        )


def test_checkpoint_keys_accept_seed_bounds_and_reject_seed_ten() -> None:
    assert checkpoint_object_key(0, "dense", "a" * 64) == (
        "checkpoints/seed-0/dense/sha256/" + "a" * 64 + ".pt"
    )
    assert checkpoint_receipt_key(9, "b" * 64) == (
        "receipts/checkpoints/seed-9/sha256/" + "b" * 64 + ".json"
    )
    with pytest.raises(ValueError, match="seed"):
        checkpoint_object_key(10, "split90", "c" * 64)


def test_legacy_local_receipts_are_audit_only_for_v3_manifests(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_bytes(
        _canonical_json(
            {
                "checkpoints": [],
                "dataset_sha256": "d" * 64,
                "provider": "aws-p5.48xlarge",
                "release_sha256": "e" * 64,
                "run_manifest_sha256": "f" * 64,
                "schema_version": 2,
                "source_commit": "a" * 40,
            }
        )
    )
    manifest = SimpleNamespace(
        schema_version=3,
        provider="aws-p5.48xlarge",
    )
    release = SimpleNamespace(archive_sha256="e" * 64)

    with pytest.raises(Exception, match="legacy|schema-3|v3"):
        verify_checkpoint_receipt(
            legacy,
            release=release,
            manifest=manifest,
        )


def test_v3_launcher_manifest_propagates_complete_mirror_context(
    tmp_path: Path,
) -> None:
    from tests.test_aws_p5_launcher import (
        _launcher_fixture,
        _load_fixture_plan,
        _write_json,
    )

    fixture = _launcher_fixture(tmp_path, seed=9)
    fixture["manifest"].update(
        {
            "dataset_receipt_sha256": "1" * 64,
            "environment_receipt_sha256": "2" * 64,
            "release_receipt_sha256": "3" * 64,
            "run_manifest_sha256": "4" * 64,
            "schema_version": 2,
            "source_tree": "5" * 40,
        }
    )
    fixture["manifest"]["dataset_receipt_sha256"] = fixture["manifest"][
        "corpus_receipt"
    ]["sha256"]
    _write_json(fixture["manifest_path"], fixture["manifest"])

    plan = _load_fixture_plan(fixture)

    assert plan.seed == 9
    assert plan.environment_receipt_sha256 == "2" * 64
    assert plan.release_receipt_sha256 == "3" * 64
    assert plan.run_manifest_sha256 == "4" * 64
    assert plan.dataset_receipt_sha256 == fixture["manifest"][
        "corpus_receipt"
    ]["sha256"]
    assert plan.dataset_build_id == fixture["manifest"]["corpus_receipt"][
        "build_id"
    ]
    assert plan.ordered_stream_sha256 == fixture["manifest"][
        "corpus_receipt"
    ]["ordered_stream_sha256"]
    assert plan.source_tree == "5" * 40


def test_resume_plan_can_verify_existing_outputs_without_mutating_them(
    tmp_path: Path,
) -> None:
    from cluster.aws.p5.launch_seed_pair import load_launch_plan
    from tests.test_aws_p5_launcher import (
        H100_NAMES,
        BOOT_ID,
        PROFILE_PATH,
        SAFE_ENVIRONMENT,
        _launcher_fixture,
    )

    fixture = _launcher_fixture(tmp_path, seed=0)
    output_paths = [
        fixture["scratch_root"] / f"runs/seed-0/{arm}"
        for arm in ("dense", "split90")
    ]
    for output in output_paths:
        (output / "run").mkdir(parents=True)
        (output / "run" / "sentinel").write_text("prior")

    plan = load_launch_plan(
        seed=0,
        manifest_path=fixture["manifest_path"],
        profile_path=PROFILE_PATH,
        repo_root=fixture["repo_root"],
        scratch_root=fixture["scratch_root"],
        environment=SAFE_ENVIRONMENT,
        observed_instance_type="p5.48xlarge",
        observed_instance_id="i-0123456789abcdef0",
        observed_boot_id=BOOT_ID,
        gpu_names=H100_NAMES,
        port_available=lambda _port: True,
        semantic_corpus_verifier=lambda _root: fixture["corpus"],
        enforce_profile_scratch=False,
        allow_existing_outputs=True,
    )

    assert plan.seed == 0
    assert all((path / "run" / "sentinel").read_text() == "prior" for path in output_paths)


def test_launcher_manifest_builder_emits_closed_v3_mirror_context(
    tmp_path: Path,
) -> None:
    from msctl.aws_launch_manifest import build_launcher_manifest
    from tests.test_aws_p5_launcher import _launcher_fixture

    fixture = _launcher_fixture(tmp_path, seed=0)
    source = fixture["manifest"]
    out = fixture["scratch_root"] / "staging" / "v3-launcher.json"
    manifest = build_launcher_manifest(
        out=out,
        scratch_root=fixture["scratch_root"],
        seed=0,
        profile_sha256=source["profile_sha256"],
        release_sha256=source["release_sha256"],
        release_members_sha256=source["release_members_sha256"],
        release_receipt_sha256="1" * 64,
        environment_receipt_sha256="2" * 64,
        run_manifest_sha256="3" * 64,
        cohort_assignment_sha256=source["cohort_assignment_sha256"],
        code_commit=source["code_commit"],
        source_tree="4" * 40,
        bootstrap_receipt=fixture["bootstrap_path"],
        corpus_receipt=fixture["corpus_path"],
        runs=[
            {
                "arm": row["arm"],
                "config": row["config"],
                "config_sha256": row["config_sha256"],
            }
            for row in source["runs"]
        ],
    )

    assert manifest["schema_version"] == 2
    assert manifest["release_receipt_sha256"] == "1" * 64
    assert manifest["environment_receipt_sha256"] == "2" * 64
    assert manifest["run_manifest_sha256"] == "3" * 64
    assert manifest["source_tree"] == "4" * 40
    assert manifest["dataset_receipt_sha256"] == source[
        "corpus_receipt"
    ]["sha256"]


def test_v3_submit_intent_passes_complete_mirror_context_to_launcher(
    tmp_path: Path,
) -> None:
    from msctl.contracts import load_run_manifest
    from tests.test_aws_canary import _backend, _case, _load_module

    case = _case(tmp_path, _load_module())
    backend = _backend(tmp_path, case)
    manifest = load_run_manifest(
        case["manifest_path"],
        repo_root=case["release_root"],
    )
    release = SimpleNamespace(
        archive_sha256=manifest.release_sha256,
        receipt_sha256=manifest.release_receipt_sha256,
        members_sha256="a" * 64,
        source_commit=manifest.source_commit,
        source_tree=manifest.source_tree,
    )
    evidence = {
        "dataset_pointer_sha256": manifest.dataset_pointer_sha256,
        "dataset_verification_sha256": manifest.dataset_receipt_sha256,
        "environment_receipt_sha256": "e" * 64,
        "instance_id": "i-0123456789abcdef0",
        "boot_id": "11111111-2222-4333-8444-555555555555",
    }

    intent = backend._training_operation_intent(
        operation="submit",
        release=release,
        manifest=manifest,
        terminate_at="2026-07-24T12:00:00Z",
        evidence=evidence,
    )

    argv = next(
        step["argv"]
        for step in intent["steps"]
        if step["name"] == "build-launcher-manifest"
    )
    expected = {
        "--release-receipt-sha256": release.receipt_sha256,
        "--environment-receipt-sha256": "e" * 64,
        "--run-manifest-sha256": manifest.sha256,
        "--source-tree": manifest.source_tree,
    }
    for option, value in expected.items():
        assert argv[argv.index(option) + 1] == value
    assert intent["dataset_sha256"] == manifest.dataset_receipt_sha256


def test_v3_submit_dry_run_is_enabled_with_complete_context(
    tmp_path: Path,
) -> None:
    from msctl.contracts import load_run_manifest
    from tests.test_aws_canary import _backend, _case, _load_module

    case = _case(tmp_path, _load_module())
    backend = _backend(tmp_path, case)
    manifest = load_run_manifest(
        case["manifest_path"],
        repo_root=case["release_root"],
    )
    release = SimpleNamespace(
        provider="aws-p5.48xlarge",
        archive_sha256=manifest.release_sha256,
        receipt_sha256=manifest.release_receipt_sha256,
        members_sha256="a" * 64,
        source_commit=manifest.source_commit,
        source_tree=manifest.source_tree,
    )
    terminate_at = (
        datetime.now(timezone.utc).replace(microsecond=0)
        + timedelta(hours=1)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    planned = backend.submit(
        release=release,
        manifest=manifest,
        instance_id="i-0123456789abcdef0",
        terminate_at=terminate_at,
        approval_path=None,
        apply=False,
        evidence={
            "dataset_pointer_sha256": manifest.dataset_pointer_sha256,
            "dataset_verification_sha256": (
                manifest.dataset_receipt_sha256
            ),
            "environment_receipt_sha256": "e" * 64,
            "instance_id": "i-0123456789abcdef0",
            "boot_id": "11111111-2222-4333-8444-555555555555",
        },
    )

    assert planned["operation_intent"]["schema_version"] == 2
    assert planned["operation_intent"]["boot_id"].endswith(
        "555555555555"
    )
    assert planned["operation_intent"]["checkpoint_receipt"] is None


def test_v3_lifecycle_rejects_unauthenticated_environment_before_submit(
    tmp_path: Path,
) -> None:
    from msctl.contracts import load_run_manifest
    from tests.test_aws_canary import _backend, _case, _load_module

    case = _case(tmp_path, _load_module())
    backend = _backend(
        tmp_path,
        case,
        identity_verifier=lambda *_args: False,
    )
    manifest = load_run_manifest(
        case["manifest_path"],
        repo_root=case["release_root"],
    )
    verification = {
        "receipt_sha256": manifest.dataset_receipt_sha256,
    }
    verification["verification_sha256"] = __import__("hashlib").sha256(
        json.dumps(
            verification,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    verification_path = tmp_path / "dataset-verification.json"
    verification_path.write_bytes(_canonical_json(verification))

    with pytest.raises(Exception, match="environment|authenticated"):
        backend._load_lifecycle_evidence(
            manifest=manifest,
            dataset_pointer=(
                case["release_root"] / "DATASET-POINTER-AWS.json"
            ),
            dataset_root=None,
            dataset_verification=verification_path,
            environment_receipt=case["environment"],
            expected_instance_id="i-0123456789abcdef0",
        )


def test_controller_fetches_exact_receipt_and_heads_both_versions_first(
    tmp_path: Path,
) -> None:
    from tests.test_aws_canary import _backend, _case, _load_module

    (tmp_path / "producer").mkdir()
    request = replace(
        _request(tmp_path / "producer"),
        s3_root="s3://memorysplit-prod/confirmatory-v3",
    )
    store = _MemoryVersionedStore()
    arm_by_pid = {101: "dense", 102: "split90"}

    def signal_process(pid: int, _signum: int) -> None:
        arm = arm_by_pid[pid]
        checkpoint = request.checkpoint_paths[arm]
        checkpoint.write_bytes(f"controller-{arm}".encode("ascii"))
        _write_metadata(
            checkpoint,
            step=10,
            sidecar_name=f"{arm}_target_weights",
        )

    published = publish_paired_checkpoint(
        request,
        object_store=store,
        signal_process=signal_process,
        staging_root=tmp_path / "producer-staging",
        staged_at=lambda: "2026-07-23T12:01:00Z",
    )
    case = _case(tmp_path / "controller", _load_module())

    class ReceiptRunner:
        def __init__(self):
            self.calls = []

        def run_json(self, argv, *, operation):
            self.calls.append((list(argv), operation))
            if "get-object" in argv:
                destination = Path(argv[argv.index("--checksum-mode") + 2])
                destination.write_bytes(
                    store.objects[published.receipt.uri][0]
                )
                return {
                    "receipt": {
                        "checksum_sha256": base64.b64encode(
                            bytes.fromhex(published.receipt.sha256)
                        ).decode("ascii"),
                        "version_id": published.receipt.version_id,
                    }
                }
            key = argv[argv.index("--key") + 1]
            uri = "s3://memorysplit-prod/" + key
            payload, digest, metadata, version_id = store.objects[uri]
            return {
                "object": {
                    "checksum_sha256": base64.b64encode(
                        bytes.fromhex(digest)
                    ).decode("ascii"),
                    "content_length": len(payload),
                    "metadata": metadata,
                    "version_id": version_id,
                }
            }

    runner = ReceiptRunner()
    backend = _backend(tmp_path / "backend", case, runner=runner)
    manifest = SimpleNamespace(
        schema_version=3,
        provider="aws-p5.48xlarge",
        cohort_id="memorysplit-confirmatory-v3-360m-n10-aws",
        seed=0,
        profile_sha256=request.profile_sha256,
        release_sha256=request.release_sha256,
        release_receipt_sha256=request.release_receipt_sha256,
        dataset_receipt_sha256=request.dataset_receipt_sha256,
        dataset_build_id=request.dataset_build_id,
        ordered_stream_sha256=request.ordered_stream_sha256,
        source_commit=request.source_commit,
        source_tree=request.source_tree,
        sha256=request.run_manifest_sha256,
        runs=tuple(
            SimpleNamespace(
                run_id=request.run_ids[arm],
                arm=arm,
                seed=0,
                config_sha256=request.config_sha256[arm],
            )
            for arm in ("dense", "split90")
        ),
    )
    release = SimpleNamespace(
        archive_sha256=request.release_sha256,
        receipt_sha256=request.release_receipt_sha256,
        source_commit=request.source_commit,
        source_tree=request.source_tree,
    )

    fetched = backend._fetch_checkpoint_receipt_v3(
        receipt_uri=published.receipt.uri,
        receipt_sha256=published.receipt.sha256,
        receipt_version_id=published.receipt.version_id,
        release=release,
        manifest=manifest,
        evidence={
            "environment_receipt_sha256": (
                request.environment_receipt_sha256
            ),
            "instance_id": request.instance_id,
            "boot_id": request.boot_id,
        },
    )

    assert fetched.version_id == published.receipt.version_id
    assert ["get-object" in call[0] for call in runner.calls] == [
        True,
        False,
        False,
    ]
    assert all(
        "--version-id" in argv for argv, _operation in runner.calls
    )


def test_v3_resume_dry_run_consumes_the_versioned_pair(
    tmp_path: Path,
) -> None:
    from msctl.contracts import load_run_manifest
    from tests.test_aws_canary import _backend, _case, _load_module

    case = _case(tmp_path / "controller", _load_module())
    backend = _backend(tmp_path / "backend", case)
    manifest = load_run_manifest(
        case["manifest_path"],
        repo_root=case["release_root"],
    )
    release = SimpleNamespace(
        provider="aws-p5.48xlarge",
        archive_sha256=manifest.release_sha256,
        receipt_sha256=manifest.release_receipt_sha256,
        members_sha256="a" * 64,
        source_commit=manifest.source_commit,
        source_tree=manifest.source_tree,
    )
    producer = tmp_path / "producer"
    producer.mkdir()
    environment_sha256 = "e" * 64
    request = replace(
        _request(producer),
        profile_sha256=manifest.profile_sha256,
        environment_receipt_sha256=environment_sha256,
        release_sha256=manifest.release_sha256,
        release_receipt_sha256=manifest.release_receipt_sha256,
        run_manifest_sha256=manifest.sha256,
        dataset_receipt_sha256=manifest.dataset_receipt_sha256,
        dataset_build_id=manifest.dataset_build_id,
        ordered_stream_sha256=manifest.ordered_stream_sha256,
        source_commit=manifest.source_commit,
        source_tree=manifest.source_tree,
        config_sha256={
            run.arm: run.config_sha256 for run in manifest.runs
        },
        run_ids={run.arm: run.run_id for run in manifest.runs},
        s3_root="s3://memorysplit-prod/confirmatory-v3",
    )
    store = _MemoryVersionedStore()
    arm_by_pid = {101: "dense", 102: "split90"}

    def signal_process(pid: int, _signum: int) -> None:
        arm = arm_by_pid[pid]
        checkpoint = request.checkpoint_paths[arm]
        checkpoint.write_bytes(f"resume-{arm}".encode("ascii"))
        _write_metadata(
            checkpoint,
            step=14,
            sidecar_name=f"{arm}_target_weights",
            receipt_sha256=request.dataset_receipt_sha256,
            build_id=request.dataset_build_id,
            ordered_stream_sha256=request.ordered_stream_sha256,
        )

    published = publish_paired_checkpoint(
        request,
        object_store=store,
        signal_process=signal_process,
        staging_root=tmp_path / "producer-staging",
        staged_at=lambda: "2026-07-23T12:01:00Z",
    )
    receipt = parse_paired_checkpoint_receipt_v3(
        store.objects[published.receipt.uri][0],
        receipt_uri=published.receipt.uri,
        receipt_sha256=published.receipt.sha256,
        receipt_version_id=published.receipt.version_id,
    )

    planned = backend.resume(
        release=release,
        manifest=manifest,
        checkpoint_receipt=receipt,
        approval_path=None,
        apply=False,
        evidence={
            "dataset_pointer_sha256": manifest.dataset_pointer_sha256,
            "dataset_verification_sha256": (
                manifest.dataset_receipt_sha256
            ),
            "environment_receipt_sha256": environment_sha256,
            "instance_id": request.instance_id,
            "boot_id": request.boot_id,
        },
    )

    assert planned["checkpoint_receipt_sha256"] == receipt.sha256
    assert planned["checkpoint_receipt_version_id"] == receipt.version_id
    assert planned["operation_intent"]["schema_version"] == 2
    assert [
        row["version_id"]
        for row in planned["operation_intent"]["checkpoint_receipt"][
            "checkpoints"
        ]
    ] == ["version-1", "version-2"]


def test_v3_resume_intent_version_pins_receipt_and_both_checkpoints(
    tmp_path: Path,
) -> None:
    from tests.test_aws_canary import _backend, _case, _load_module

    case = _case(tmp_path, _load_module())
    backend = _backend(tmp_path, case)
    manifest = SimpleNamespace(
        schema_version=3,
        seed=0,
        sha256="1" * 64,
        dataset_pointer_sha256="2" * 64,
        dataset_receipt_sha256="3" * 64,
        source_tree="4" * 40,
        cohort_assignment_sha256="5" * 64,
        runs=(),
    )
    release = SimpleNamespace(
        archive_sha256="6" * 64,
        receipt_sha256="7" * 64,
        members_sha256="8" * 64,
    )
    checkpoints = {
        arm: SimpleNamespace(
            arm=arm,
            world_size=4,
            step=11,
            config_sha256=("e" if arm == "dense" else "f") * 64,
            config_fingerprint=("1" if arm == "dense" else "2") * 64,
            object=SimpleNamespace(
                uri=(
                    "s3://memorysplit-prod/confirmatory-v3/checkpoints/"
                    f"seed-0/{arm}/sha256/{digest}.pt"
                ),
                sha256=digest,
                bytes=123,
                version_id=f"{arm}-version",
            ),
        )
        for arm, digest in (
            ("dense", "a" * 64),
            ("split90", "b" * 64),
        )
    }

    intent = backend._training_operation_intent(
        operation="resume",
        release=release,
        manifest=manifest,
        checkpoints=checkpoints,
        checkpoint_receipt_sha256="c" * 64,
        checkpoint_receipt_uri=(
            "s3://memorysplit-prod/confirmatory-v3/receipts/checkpoints/"
            "seed-0/sha256/" + "c" * 64 + ".json"
        ),
        checkpoint_receipt_version_id="receipt-version",
        checkpoint_receipt_bytes=456,
        evidence={
            "dataset_pointer_sha256": "2" * 64,
            "dataset_verification_sha256": "3" * 64,
            "environment_receipt_sha256": "d" * 64,
            "instance_id": "i-0123456789abcdef0",
            "boot_id": "11111111-2222-4333-8444-555555555555",
        },
    )

    downloads = [
        step["argv"]
        for step in intent["steps"]
        if step["name"].startswith("materialize-resume-")
        and step["name"] != "materialize-resume-staging"
    ]
    assert len(downloads) == 3
    assert [
        argv[argv.index("--version-id") + 1] for argv in downloads
    ] == ["receipt-version", "dense-version", "split90-version"]
    binding = intent["checkpoint_receipt"]
    assert binding["uri"].endswith("c" * 64 + ".json")
    assert binding["version_id"] == "receipt-version"
    assert [
        row["version_id"] for row in binding["checkpoints"]
    ] == ["dense-version", "split90-version"]
    from msctl import aws_argv
    from msctl.jsonutil import canonical_json

    envelope = backend._operation_envelope(
        intent,
        instance_id="i-0123456789abcdef0",
        terminate_at="2026-07-24T12:00:00Z",
    )
    payload = canonical_json(envelope)
    assert aws_argv._validate_intent(
        payload,
        expected_sha256=__import__("hashlib").sha256(payload).hexdigest(),
    ) == envelope
    from msctl.aws_argv import _validate_intent

    envelope = backend._operation_envelope(
        intent,
        instance_id="i-0123456789abcdef0",
        terminate_at="2026-07-24T12:00:00Z",
    )
    payload = _canonical_json(envelope)[:-1]
    assert _validate_intent(
        payload,
        expected_sha256=__import__("hashlib").sha256(payload).hexdigest(),
    ) == envelope


def test_schema_two_state_records_receipt_and_object_versions(
    tmp_path: Path,
) -> None:
    from tests.test_aws_canary import _backend, _case, _load_module

    case = _case(tmp_path, _load_module())
    backend = _backend(tmp_path, case)
    run = SimpleNamespace(
        run_id="memorysplit-v3-360m-s0-dense",
        arm="dense",
        seed=0,
        config_sha256="1" * 64,
    )
    manifest = SimpleNamespace(
        schema_version=3,
        release_sha256="2" * 64,
        release_receipt_sha256="3" * 64,
        sha256="4" * 64,
        dataset_pointer_sha256="5" * 64,
        dataset_receipt_sha256="6" * 64,
        dataset_build_id="7" * 64,
        ordered_stream_sha256="8" * 64,
        cohort_assignment_sha256="9" * 64,
        preregistration_sha256="a" * 64,
        source_commit="b" * 40,
        source_tree="c" * 40,
    )
    intent = {
        "dataset_pointer_sha256": "5" * 64,
        "dataset_verification_sha256": "6" * 64,
        "environment_receipt_sha256": "d" * 64,
        "operation_id": "e" * 64,
    }
    receipt = SimpleNamespace(
        uri="s3://bucket/receipt.json",
        sha256="f" * 64,
        version_id="receipt-version",
    )
    checkpoints = {
        arm: SimpleNamespace(
            object=SimpleNamespace(
                uri=f"s3://bucket/{arm}.pt",
                sha256=("1" if arm == "dense" else "2") * 64,
                version_id=f"{arm}-version",
                bytes=123,
            )
        )
        for arm in ("dense", "split90")
    }

    state = backend._new_aws_run_state(
        run=run,
        manifest=manifest,
        operation="resume",
        instance_id="i-0123456789abcdef0",
        terminate_at="2026-07-24T12:00:00Z",
        intent=intent,
        published={
            "intent_sha256": "f" * 64,
            "intent_uri": "s3://bucket/intent.json",
        },
        attempt=2,
        checkpoint_receipt=receipt,
        checkpoints=checkpoints,
        prior_command_ids=["command-12345678"],
    )

    assert state["schema_version"] == 2
    assert "dataset_sha256" not in state
    assert "study_lock_sha256" not in state
    assert state["checkpoint_receipt"] == {
        "sha256": "f" * 64,
        "uri": "s3://bucket/receipt.json",
        "version_id": "receipt-version",
    }
    assert {
        row["version_id"] for row in state["checkpoint_objects"]
    } == {"dense-version", "split90-version"}
    from msctl.state import StateStore

    store = StateStore(tmp_path / "state")
    with store.locked():
        store.write_run(run.run_id, state)
        assert store.read_run(run.run_id) == state


def test_v3_partial_paired_state_fails_closed(tmp_path: Path) -> None:
    from msctl.state import StateStore
    from tests.test_aws_canary import _backend, _case, _load_module

    case = _case(tmp_path, _load_module())
    backend = _backend(tmp_path, case)
    runs = tuple(
        SimpleNamespace(
            run_id=f"memorysplit-v3-360m-s0-{arm}",
            arm=arm,
            seed=0,
            config_sha256=("1" if arm == "dense" else "2") * 64,
        )
        for arm in ("dense", "split90")
    )
    manifest = SimpleNamespace(
        schema_version=3,
        release_sha256="3" * 64,
        release_receipt_sha256="4" * 64,
        sha256="5" * 64,
        dataset_pointer_sha256="6" * 64,
        dataset_receipt_sha256="7" * 64,
        dataset_build_id="8" * 64,
        ordered_stream_sha256="9" * 64,
        cohort_assignment_sha256="a" * 64,
        preregistration_sha256="b" * 64,
        source_commit="c" * 40,
        source_tree="d" * 40,
        runs=runs,
    )
    intent = {
        "dataset_pointer_sha256": "6" * 64,
        "dataset_verification_sha256": "7" * 64,
        "environment_receipt_sha256": "e" * 64,
        "operation_id": "f" * 64,
    }
    published = {
        "intent_sha256": "1" * 64,
        "intent_uri": "s3://bucket/intent.json",
    }
    states = [
        backend._new_aws_run_state(
            run=run,
            manifest=manifest,
            operation="submit",
            instance_id="i-0123456789abcdef0",
            terminate_at="2026-07-24T12:00:00Z",
            intent=intent,
            published=published,
            attempt=1,
        )
        for run in runs
    ]
    store = StateStore(tmp_path / "partial-state")
    with store.locked():
        store.write_aws_pair(
            manifest.sha256,
            {
                "schema_version": 2,
                "provider": "aws-p5.48xlarge",
                "run_manifest_sha256": manifest.sha256,
                "operation_id": intent["operation_id"],
                "states": states,
            },
        )
        store.write_run(runs[0].run_id, states[0])
        with pytest.raises(Exception, match="partial|incomplete"):
            backend._repair_paired_states(store, manifest)


def test_interruption_uses_scheduler_publisher_and_keeps_last_complete_pair(
    tmp_path: Path,
) -> None:
    from cluster.aws.p5.launch_seed_pair import supervise_pair
    from tests.test_aws_p5_launcher import (
        _FakeProcess,
        _FakeSpawner,
        _launcher_fixture,
        _load_fixture_plan,
        _pass_rank_zero_resolver,
        _pass_trainer_preflight,
    )

    plan = _load_fixture_plan(_launcher_fixture(tmp_path))
    published = PublishedCheckpointPair(
        receipt=CheckpointReceiptRef(
            uri="s3://bucket/receipt.json",
            sha256="a" * 64,
            version_id="receipt-version",
            bytes=123,
        ),
        checkpoints=(
            VersionedUploadedObject(
                "s3://bucket/dense.pt",
                "b" * 64,
                10,
                "dense-version",
            ),
            VersionedUploadedObject(
                "s3://bucket/split90.pt",
                "c" * 64,
                10,
                "split90-version",
            ),
        ),
        value={},
    )
    reasons = []

    class Scheduler:
        latest = None
        active = False

        def maybe_start(self, request, **_kwargs):
            reasons.append(request.reason)
            self.active = request.reason == "interruption"
            return self.active

        def poll(self, **_kwargs):
            if self.active:
                self.latest = published
                self.active = False
                return published
            return None

        def cancel_active(self):
            self.active = False

    scheduler = Scheduler()

    result = supervise_pair(
        plan,
        spawner=_FakeSpawner(
            {
                "dense": _FakeProcess(101, [None]),
                "split90": _FakeProcess(202, [None]),
            }
        ),
        sleep=lambda _delay: None,
        notice_source=lambda: "spot-interruption",
        trainer_preflight=_pass_trainer_preflight,
        rank_zero_resolver=_pass_rank_zero_resolver,
        checkpoint_scheduler_factory=lambda _plan, _pids: (
            scheduler,
            lambda reason: SimpleNamespace(reason=reason),
        ),
    )

    assert reasons == ["periodic", "interruption"]
    assert result.resumable is True
    assert result.interruption_receipt == published.receipt.uri


def test_v3_package_requires_checkpoint_producer_and_resume_dependencies() -> None:
    import scripts.package_aws_p5_handoff as packager

    assert {
        "cluster/aws/p5/checkpoint_mirror.py",
        "msctl/aws_argv.py",
        "msctl/aws_launch_manifest.py",
        "msctl/aws_resume_launch.py",
        "msctl/contracts.py",
        "msctl/state.py",
    } <= packager.REQUIRED_MEMBERS


@pytest.mark.parametrize("seed", [0, 9])
def test_v3_resume_output_archival_accepts_seed_bounds(
    tmp_path: Path,
    seed: int,
) -> None:
    scratch = tmp_path / f"seed-{seed}"
    scratch.mkdir()

    archive = aws_resume_launch_module.prepare_resume_output_roots(
        scratch,
        seed=seed,
        checkpoint_receipt_sha256="a" * 64,
    )

    assert archive == (
        scratch.resolve()
        / "resume-history"
        / f"seed-{seed}"
        / ("a" * 64)
    )


@pytest.mark.parametrize("seed", [True, 10])
def test_v3_resume_output_archival_rejects_non_exact_seed(
    tmp_path: Path,
    seed: object,
) -> None:
    scratch = tmp_path / "invalid-seed"
    scratch.mkdir()

    with pytest.raises(
        aws_resume_launch_module.ResumeLaunchError,
        match="seed",
    ):
        aws_resume_launch_module.prepare_resume_output_roots(
            scratch,
            seed=seed,
            checkpoint_receipt_sha256="a" * 64,
        )


def test_v3_resume_installs_paired_checkpoint_scheduler_without_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = SimpleNamespace(seed=0, run_manifest_sha256="b" * 64)
    production_scheduler_factory = object()
    legacy_interruption_handler = object()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        aws_resume_launch_module,
        "_verify_launcher_path",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        aws_resume_launch_module,
        "_verify_checkpoint_receipt",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        aws_resume_launch_module,
        "bind_resume_checkpoints",
        lambda launch_plan, **_kwargs: launch_plan,
    )
    monkeypatch.setattr(
        aws_resume_launch_module,
        "prepare_resume_output_roots",
        lambda *_args, **_kwargs: tmp_path / "archive",
    )
    monkeypatch.setattr(
        aws_resume_launch_module.reviewed_launcher,
        "load_launch_plan",
        lambda **_kwargs: plan,
    )
    monkeypatch.setattr(
        aws_resume_launch_module.reviewed_launcher,
        "_production_checkpoint_scheduler",
        production_scheduler_factory,
    )
    monkeypatch.setattr(
        aws_resume_launch_module.reviewed_launcher,
        "_production_interruption_handler",
        legacy_interruption_handler,
    )

    class Client:
        def interruption_notice(self) -> None:
            return None

    class ShutdownHandlers:
        def __enter__(self):
            return lambda: None

        def __exit__(self, *_args) -> None:
            return None

    def supervise_pair(_plan, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            child_pids={"dense": 101, "split90": 102},
            failed_arm=None,
            interruption_receipt=None,
            peer_terminated=False,
            resumable=False,
            returncode=0,
            status="completed",
        )

    monkeypatch.setattr(
        aws_resume_launch_module.reviewed_launcher,
        "ImdsV2Client",
        Client,
    )
    monkeypatch.setattr(
        aws_resume_launch_module.reviewed_launcher,
        "installed_shutdown_handlers",
        lambda: ShutdownHandlers(),
    )
    monkeypatch.setattr(
        aws_resume_launch_module.reviewed_launcher,
        "supervise_pair",
        supervise_pair,
    )

    result = aws_resume_launch_module.main(
        [
            "--seed",
            "0",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--profile",
            str(tmp_path / "profile.toml"),
            "--repo-root",
            str(tmp_path),
            "--scratch-root",
            str(tmp_path),
            "--launcher",
            str(tmp_path / "launch_seed_pair.py"),
            "--checkpoint-receipt",
            str(tmp_path / "receipt.json"),
            "--checkpoint-receipt-sha256",
            "a" * 64,
            "--checkpoint-receipt-uri",
            "s3://bucket/receipt.json",
            "--checkpoint-receipt-version-id",
            "receipt-version",
            "--run-manifest-sha256",
            "b" * 64,
            "--checkpoint",
            "{}",
            "--apply",
        ]
    )

    assert result == 0
    assert observed["checkpoint_scheduler_factory"] is production_scheduler_factory
    assert observed["interruption_handler"] is None
    capsys.readouterr()


def test_receipt_seed_nine_passes_and_seed_ten_fails(tmp_path: Path) -> None:
    request = replace(
        _request(tmp_path),
        seed=9,
        run_ids={
            "dense": "memorysplit-v3-360m-s9-dense",
            "split90": "memorysplit-v3-360m-s9-split90",
        },
    )
    store = _MemoryVersionedStore()
    arm_by_pid = {101: "dense", 102: "split90"}

    def signal_process(pid: int, _signum: int) -> None:
        arm = arm_by_pid[pid]
        checkpoint = request.checkpoint_paths[arm]
        checkpoint.write_bytes(f"seed-nine-{arm}".encode("ascii"))
        _write_metadata(
            checkpoint,
            step=12,
            sidecar_name=f"{arm}_target_weights",
        )

    published = publish_paired_checkpoint(
        request,
        object_store=store,
        signal_process=signal_process,
        staging_root=tmp_path / "staging",
        staged_at=lambda: "2026-07-23T12:01:00Z",
    )
    parsed = parse_paired_checkpoint_receipt_v3(
        store.objects[published.receipt.uri][0],
        receipt_uri=published.receipt.uri,
        receipt_sha256=published.receipt.sha256,
        receipt_version_id=published.receipt.version_id,
    )
    assert parsed.seed == 9
    with pytest.raises(ValueError, match="seed"):
        replace(request, seed=10)


@pytest.mark.parametrize(
    "mutation",
    [
        "top-extra",
        "seed-bool",
        "seed-ten",
        "reason",
        "request-id",
        "freshness-extra",
        "freshness-window",
        "timestamp",
        "row-order",
        "step-bool",
        "step-range",
        "world-size",
        "data-cursor",
        "sidecar",
        "data-provenance",
        "object-uri",
        "object-version-empty",
        "object-version-null",
    ],
)
def test_receipt_field_type_and_provenance_mutations_fail(
    tmp_path: Path,
    mutation: str,
) -> None:
    request = _request(tmp_path)
    store = _MemoryVersionedStore()
    arm_by_pid = {101: "dense", 102: "split90"}

    def signal_process(pid: int, _signum: int) -> None:
        arm = arm_by_pid[pid]
        checkpoint = request.checkpoint_paths[arm]
        checkpoint.write_bytes(f"matrix-{arm}".encode("ascii"))
        _write_metadata(
            checkpoint,
            step=13,
            sidecar_name=f"{arm}_target_weights",
        )

    published = publish_paired_checkpoint(
        request,
        object_store=store,
        signal_process=signal_process,
        staging_root=tmp_path / "staging",
        staged_at=lambda: "2026-07-23T12:01:00Z",
    )
    value = copy.deepcopy(published.value)
    if mutation == "top-extra":
        value["unexpected"] = None
    elif mutation == "seed-bool":
        value["seed"] = True
    elif mutation == "seed-ten":
        value["seed"] = 10
    elif mutation == "reason":
        value["reason"] = "manual"
    elif mutation == "request-id":
        value["request_id"] = "A" * 32
    elif mutation == "freshness-extra":
        value["freshness"]["unexpected"] = None
    elif mutation == "freshness-window":
        value["freshness"]["max_age_seconds"] = 1199
    elif mutation == "timestamp":
        value["freshness"]["staged_at"] = "2026-07-23T12:01:00+00:00"
    elif mutation == "row-order":
        value["checkpoints"].reverse()
    elif mutation == "step-bool":
        value["checkpoints"][0]["step"] = True
    elif mutation == "step-range":
        value["checkpoints"][0]["step"] = 13_583
    elif mutation == "world-size":
        value["checkpoints"][0]["world_size"] = 8
    elif mutation == "data-cursor":
        value["checkpoints"][0]["data"]["global_cursor"] += 1
    elif mutation == "sidecar":
        value["checkpoints"][0]["data"][
            "sidecar_name"
        ] = "split90_target_weights"
    elif mutation == "data-provenance":
        value["checkpoints"][0]["data"]["receipt_sha256"] = "0" * 64
    elif mutation == "object-uri":
        value["checkpoints"][0]["object"]["uri"] += ".changed"
    elif mutation == "object-version-empty":
        value["checkpoints"][0]["object"]["version_id"] = ""
    elif mutation == "object-version-null":
        value["checkpoints"][0]["object"]["version_id"] = "null"
    else:
        raise AssertionError(mutation)
    payload = _canonical_json(value)
    digest = __import__("hashlib").sha256(payload).hexdigest()
    uri = (
        "s3://memorysplit-test/prefix/receipts/checkpoints/"
        f"seed-{value['seed']}/sha256/{digest}.json"
    )

    with pytest.raises(Exception):
        parse_paired_checkpoint_receipt_v3(
            payload,
            receipt_uri=uri,
            receipt_sha256=digest,
            receipt_version_id="receipt-version",
        )


class _ControlledAttempt:
    def __init__(self) -> None:
        self.complete = False
        self.result = None
        self.cancelled = False

    def poll(self):
        return self.complete, self.result

    def cancel(self) -> None:
        self.cancelled = True


def test_scheduler_starts_at_1080_and_fail_stops_at_1200(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    attempts: list[_ControlledAttempt] = []

    def start_attempt(_request):
        attempt = _ControlledAttempt()
        attempts.append(attempt)
        return attempt

    scheduler = CheckpointMirrorScheduler(
        start_attempt,
        started_at=0.0,
    )

    assert scheduler.maybe_start(request, now=1079.999) is False
    assert scheduler.maybe_start(request, now=1080.0) is True
    assert scheduler.poll(now=1199.999) is None
    with pytest.raises(CheckpointStaleError, match="CHECKPOINT_STALE"):
        scheduler.poll(now=1200.0)
    assert attempts[0].cancelled is True
    assert scheduler.latest is None


def test_scheduler_resets_freshness_only_after_verified_pair(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    attempt = _ControlledAttempt()
    pair = PublishedCheckpointPair(
        receipt=CheckpointReceiptRef(
            uri="s3://bucket/receipt.json",
            sha256="1" * 64,
            version_id="receipt-version",
            bytes=10,
        ),
        checkpoints=(
            VersionedUploadedObject(
                "s3://bucket/dense.pt",
                "2" * 64,
                5,
                "dense-version",
            ),
            VersionedUploadedObject(
                "s3://bucket/split90.pt",
                "3" * 64,
                5,
                "split90-version",
            ),
        ),
        value={},
    )
    scheduler = CheckpointMirrorScheduler(
        lambda _request: attempt,
        started_at=0.0,
    )

    assert scheduler.maybe_start(request, now=1080.0) is True
    assert scheduler.fresh_at == 0.0
    attempt.complete = True
    attempt.result = pair
    assert scheduler.poll(now=1081.0) is pair
    assert scheduler.fresh_at == 1081.0
    assert scheduler.latest is pair
    assert scheduler.maybe_start(request, now=2160.999) is False


def test_scheduler_rejects_completed_attempt_at_durability_deadline(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    attempt = _ControlledAttempt()
    pair = PublishedCheckpointPair(
        receipt=CheckpointReceiptRef(
            uri="s3://bucket/late-receipt.json",
            sha256="4" * 64,
            version_id="late-receipt-version",
            bytes=10,
        ),
        checkpoints=(
            VersionedUploadedObject(
                "s3://bucket/late-dense.pt",
                "5" * 64,
                5,
                "late-dense-version",
            ),
            VersionedUploadedObject(
                "s3://bucket/late-split90.pt",
                "6" * 64,
                5,
                "late-split90-version",
            ),
        ),
        value={},
    )
    scheduler = CheckpointMirrorScheduler(
        lambda _request: attempt,
        started_at=0.0,
    )

    assert scheduler.maybe_start(request, now=1080.0) is True
    attempt.complete = True
    attempt.result = pair

    with pytest.raises(CheckpointStaleError, match="CHECKPOINT_STALE"):
        scheduler.poll(now=1200.0)

    assert attempt.cancelled is True
    assert scheduler.latest is None
    assert scheduler.fresh_at == 0.0


def test_s3_lost_put_response_recovers_only_through_exact_head(
    tmp_path: Path,
) -> None:
    payload = b"versioned-checkpoint"
    digest = __import__("hashlib").sha256(payload).hexdigest()
    checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
    body = tmp_path / "checkpoint.pt"
    body.write_bytes(payload)
    metadata = {"arm": "dense", "sha256": digest}
    calls: list[list[str]] = []
    responses = [
        type("Result", (), {"returncode": 1, "stdout": "", "stderr": "lost"})(),
        type(
            "Result",
            (),
            {
                "returncode": 0,
                "stdout": json.dumps(
                    {
                        "object": {
                            "checksum_sha256": checksum,
                            "content_length": len(payload),
                            "metadata": metadata,
                            "version_id": "version-9",
                        }
                    }
                ),
                "stderr": "",
            },
        )(),
    ]

    def runner(argv, _environment, _timeout):
        calls.append(list(argv))
        return responses.pop(0)

    store = S3VersionedObjectStore(
        region="us-east-1",
        environment={"PATH": "/usr/bin:/bin"},
        runner=runner,
    )

    assert (
        store.put_if_absent(
            body,
            "s3://bucket/checkpoint.pt",
            sha256=digest,
            byte_count=len(payload),
            metadata=metadata,
        )
        is None
    )
    recovered = store.head_exact(
        "s3://bucket/checkpoint.pt",
        sha256=digest,
        byte_count=len(payload),
        metadata=metadata,
        version_id=None,
    )

    assert recovered is not None
    assert recovered.version_id == "version-9"
    assert "--if-none-match" in calls[0]
    assert "--version-id" not in calls[1]
