from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pytest
import torch

import cluster.aws.p5.launch_seed_pair as launch_module
import cluster.aws.p5.run_finalization as finalization
from cluster.aws.gpu_profile import load_aws_gpu_profile
from cluster.aws.p5.checkpoint_mirror import (
    CheckpointReceiptRef,
    PublishedCheckpointPair,
    VersionedUploadedObject,
)
from msctl.aws_contracts import (
    ARMS,
    COHORT_ID,
    SNAPSHOT_STEPS,
    checkpoint_receipt_key,
    log_object_key,
    run_receipt_key,
    snapshot_object_key,
)
from msctl.aws_lifecycle import (
    AuthenticatedProviderLifecycle,
    LIFECYCLE_BINDING_FIELDS,
    ProviderLifecycleBinding,
    lifecycle_operational_metadata,
)
from tests.provider_lifecycle_fixtures import provider_lifecycle
from tests.test_aws_p5_launcher import (
    H100_NAMES,
    SAFE_ENVIRONMENT,
    _launcher_fixture,
    _sha256,
    _write_json,
)


ROOT = Path(__file__).resolve().parents[1]
P5_PROFILE = ROOT / "cluster" / "profiles" / "aws-p5.48xlarge-v3.json"
P6_PROFILE = ROOT / "cluster" / "profiles" / "aws-p6-b300.48xlarge-v3.json"
P6_MEMBER_PATH = "cluster/profiles/aws-p6-b300.48xlarge-v3.json"
GPU_NAMES = {
    "aws-p5.48xlarge-v3": H100_NAMES,
    "aws-p6-b300.48xlarge-v3": ("NVIDIA B300",) * 8,
}
FINGERPRINT = "f" * 64
REQUEST_ID = "d" * 32
FINALIZED_AT = "2026-07-24T02:00:00Z"
TERMINAL_STEP = 13_582
_SIDECARS = {
    "dense": "dense_target_weights",
    "split90": "split90_target_weights",
}


def _canonical(value: object) -> bytes:
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


class _TimeReader:
    """Deterministic UTC/monotonic reader for bounded finalization."""

    def __init__(self, *, now_text: str = FINALIZED_AT, ticks=None) -> None:
        self._now = datetime.strptime(
            now_text, "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
        self._ticks = list(ticks) if ticks is not None else None
        self._current = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        if self._ticks:
            self._current = self._ticks.pop(0)
        else:
            self._current += 1.0
        return self._current


class _MemoryStore:
    def __init__(self) -> None:
        self.objects: dict[
            str, tuple[bytes, str, Mapping[str, str], str]
        ] = {}
        self.put_order: list[str] = []

    def put_if_absent(self, path, uri, *, sha256, byte_count, metadata):
        payload = Path(path).read_bytes()
        assert len(payload) == byte_count
        assert uri not in self.put_order
        version_id = f"version-{len(self.put_order) + 1}"
        self.objects[uri] = (payload, sha256, dict(metadata), version_id)
        self.put_order.append(uri)
        return VersionedUploadedObject(
            uri=uri,
            sha256=sha256,
            bytes=byte_count,
            version_id=version_id,
        )

    def head_exact(self, uri, *, sha256, byte_count, metadata, version_id):
        stored = self.objects.get(uri)
        if stored is None:
            return None
        payload, stored_sha256, stored_metadata, stored_version = stored
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


class _FaultyStore(_MemoryStore):
    """Injects one publication fault for URIs containing a target."""

    def __init__(self, *, target: str, fault: str) -> None:
        super().__init__()
        self.target = target
        self.fault = fault

    def put_if_absent(self, path, uri, **kwargs):
        if self.target in uri and self.fault == "partial-upload":
            return None
        uploaded = super().put_if_absent(path, uri, **kwargs)
        if self.target in uri and self.fault in {
            "missing-version",
            "null-version",
        }:
            return dataclasses.replace(
                uploaded,
                version_id="" if self.fault == "missing-version" else "null",
            )
        return uploaded

    def head_exact(self, uri, **kwargs):
        if self.target in uri and self.fault in {
            "partial-upload",
            "head-missing",
        }:
            return None
        verified = super().head_exact(uri, **kwargs)
        if verified is None or self.target not in uri:
            return verified
        if self.fault == "head-version-drift":
            return dataclasses.replace(verified, version_id="drifted-version")
        if self.fault == "head-sha-drift":
            return dataclasses.replace(verified, sha256="e" * 64)
        if self.fault == "head-bytes-drift":
            return dataclasses.replace(verified, bytes=verified.bytes + 1)
        return verified


class _LostPutStore(_MemoryStore):
    """Loses the first PUT response, optionally losing the object too."""

    def __init__(self, *, recoverable: bool) -> None:
        super().__init__()
        self.recoverable = recoverable
        self.lost = False

    def put_if_absent(self, path, uri, **kwargs):
        if not self.lost:
            self.lost = True
            uploaded = super().put_if_absent(path, uri, **kwargs)
            if not self.recoverable:
                del self.objects[uri]
                self.put_order.remove(uri)
                assert uploaded is not None
            return None
        return super().put_if_absent(path, uri, **kwargs)


def _drifted_lifecycle(
    lifecycle: AuthenticatedProviderLifecycle,
    **fields,
) -> AuthenticatedProviderLifecycle:
    return AuthenticatedProviderLifecycle(
        binding=replace(lifecycle.binding, **fields),
        profile=lifecycle.profile,
        arm_bindings=dict(lifecycle.arm_bindings),
    )


def _unlock_release(repo_root: Path) -> None:
    for path in [repo_root, *repo_root.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)


def _lock_release(repo_root: Path) -> None:
    for path in repo_root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    repo_root.chmod(0o555)


def _selected_fixture(tmp_path: Path, *, profile_path: Path, seed: int = 1):
    fixture = _launcher_fixture(tmp_path, seed=seed)
    profile = load_aws_gpu_profile(profile_path)
    lifecycle = provider_lifecycle(profile, seed=seed)
    binding = lifecycle.binding
    repo_root = fixture["repo_root"]

    _unlock_release(repo_root)
    metadata = json.loads(fixture["metadata_path"].read_text())
    profile_member = {
        "aws-p5.48xlarge-v3": "cluster/profiles/aws-p5.48xlarge-v3.json",
        "aws-p6-b300.48xlarge-v3": P6_MEMBER_PATH,
    }[profile.profile_id]
    if profile.profile_id == "aws-p6-b300.48xlarge-v3":
        member_path = repo_root / P6_MEMBER_PATH
        member_path.parent.mkdir(parents=True, exist_ok=True)
        member_path.write_bytes((ROOT / P6_MEMBER_PATH).read_bytes())
        metadata["members"].append(
            {
                "bytes": member_path.stat().st_size,
                "git_blob": "5" * 40,
                "git_mode": "100644",
                "path": P6_MEMBER_PATH,
                "sha256": _sha256(member_path),
            }
        )
        metadata["members"].sort(key=lambda row: row["path"])
    metadata["provider"] = profile.provider
    metadata["seed_assignment"]["provider"] = profile.provider
    metadata["profile"] = {
        "path": profile_member,
        "sha256": profile.sha256,
    }
    metadata["environment"]["profile_sha256"] = profile.sha256
    metadata.update(binding.to_dict())
    metadata["profile_id"] = binding.profile_id
    fixture["metadata_path"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    checksum_paths = sorted(
        path.relative_to(repo_root).as_posix()
        for path in repo_root.rglob("*")
        if path.is_file()
        and path.relative_to(repo_root).as_posix() != "SHA256SUMS"
    )
    fixture["sums_path"].write_text(
        "".join(
            f"{_sha256(repo_root / relative)}  {relative}\n"
            for relative in checksum_paths
        ),
        encoding="ascii",
    )
    _lock_release(repo_root)
    release_members_sha256 = _sha256(fixture["sums_path"])
    fixture["release_members_sha256"] = release_members_sha256

    selected_bootstrap = {
        "receipt_type": "memorysplit-aws-gpu-bootstrap-v1",
        "container_image": SAFE_ENVIRONMENT["MS_CONTAINER_IMAGE"],
    }
    _write_json(fixture["bootstrap_path"], selected_bootstrap)
    fixture["manifest"].update(
        {
            **binding.to_dict(),
            "schema_version": 3,
            "profile_sha256": binding.profile_sha256,
            "release_members_sha256": release_members_sha256,
            "dataset_receipt_sha256": fixture["manifest"][
                "corpus_receipt"
            ]["sha256"],
            "environment_receipt_sha256": (
                binding.qualification_environment_receipt_sha256
            ),
            "release_receipt_sha256": "b" * 64,
            "run_manifest_sha256": "c" * 64,
            "source_tree": "5" * 40,
        }
    )
    fixture["manifest"]["bootstrap_receipt"]["sha256"] = _sha256(
        fixture["bootstrap_path"]
    )
    _write_json(fixture["manifest_path"], fixture["manifest"])
    fixture["lifecycle"] = lifecycle
    fixture["profile"] = profile
    fixture["selected_bootstrap"] = selected_bootstrap
    fixture["seed"] = seed
    fixture["tmp_path"] = tmp_path
    runtime_lock = tmp_path / "runtime-lock.json"
    runtime_sbom = tmp_path / "runtime-sbom.json"
    runtime_lock.write_bytes(b"runtime-lock\n")
    runtime_sbom.write_bytes(b"runtime-sbom\n")
    fixture["runtime_lock"] = runtime_lock
    fixture["runtime_sbom"] = runtime_sbom
    return fixture


def _authenticated_plan(fixture, monkeypatch):
    lifecycle = fixture["lifecycle"]
    binding = lifecycle.binding
    monkeypatch.setattr(
        launch_module,
        "admit_provider_lifecycle",
        lambda **kwargs: lifecycle,
    )
    monkeypatch.setattr(
        launch_module,
        "_parse_selected_bootstrap_receipt_bytes",
        lambda *args, **kwargs: fixture["selected_bootstrap"],
    )
    return launch_module.load_authenticated_launch_plan(
        seed=fixture["seed"],
        manifest_path=fixture["manifest_path"],
        repo_root=fixture["repo_root"],
        scratch_root=fixture["scratch_root"],
        environment=SAFE_ENVIRONMENT,
        authority_root=fixture["tmp_path"] / "authority",
        runtime_lock_path=fixture["runtime_lock"],
        runtime_evidence_path=fixture["tmp_path"] / "runtime-evidence.json",
        runtime_sbom_path=fixture["runtime_sbom"],
        objective_controls_amendment_path=(
            fixture["repo_root"]
            / "configs"
            / "objective-controls-amendment-v3.yaml"
        ),
        store=object(),
        account_id=binding.account_id,
        instance_id=binding.instance_id,
        boot_id=binding.boot_id,
        expected_selection_version_id=(
            binding.provider_selection_version_id
        ),
        identity_verifier=object(),
        approval_verifier=object(),
        trusted_public_key_sha256=(
            binding.qualification_approval_public_key_sha256
        ),
        observed_instance_type=fixture["profile"].instance_type,
        observed_instance_id=binding.instance_id,
        observed_boot_id=binding.boot_id,
        gpu_names=GPU_NAMES[fixture["profile"].profile_id],
        port_available=lambda _port: True,
        semantic_corpus_verifier=lambda _root: fixture["corpus"],
        enforce_profile_scratch=False,
    )


def _expected_metadata(plan, launch) -> dict[str, object]:
    return lifecycle_operational_metadata(
        plan.lifecycle_binding,
        run_id=str(launch.runtime_config["run_id"]),
        arm=launch.arm,
        config_sha256=launch.config_sha256,
        dataset_receipt_sha256=plan.dataset_receipt_sha256,
        dataset_build_id=plan.dataset_build_id,
        ordered_stream_sha256=plan.ordered_stream_sha256,
        source_commit=plan.code_commit,
        source_tree=str(plan.source_tree),
    )


def _snapshot_state(
    plan,
    launch,
    step: int,
    *,
    fingerprint: str = FINGERPRINT,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "model": {"weight": [0.0]},
        "model_cfg": {"ctx": 1024},
        "data_provenance": {"receipt_sha256": plan.dataset_receipt_sha256},
        "step": step,
        "world_size": 4,
        "config_fingerprint": fingerprint,
        **(metadata if metadata is not None else _expected_metadata(plan, launch)),
    }


def _log_row(step: int) -> dict[str, object]:
    return {
        "step": step,
        "loss": 1.5,
        "loss_ema": 1.4,
        "lr": 0.001,
        "tok_s": 10.0,
        "global_tok_s": 10.0,
        "tokens_per_step": 524_288,
        "global_tokens": step * 524_288,
        "epoch": 0,
    }


def _write_run_evidence(plan, *, log_steps=(20, 1_358, 6_791, TERMINAL_STEP)):
    for launch in plan.arms:
        run_dir = Path(launch.checkpoint_path).parent
        snapshots = run_dir / "snapshots"
        snapshots.mkdir(parents=True, exist_ok=True)
        for step in SNAPSHOT_STEPS:
            snapshot_path = snapshots / f"step{step:07d}.pt"
            with snapshot_path.open("wb") as handle:
                torch.save(_snapshot_state(plan, launch, step), handle)
            snapshot_path.chmod(0o600)
        log_path = run_dir / "log.jsonl"
        log_path.write_bytes(
            b"".join(
                (json.dumps(_log_row(step)) + "\n").encode("ascii")
                for step in log_steps
            )
        )
        log_path.chmod(0o600)


def _checkpoint_pair(
    plan,
    *,
    step: int = 6_791,
    request_id: str = "c" * 32,
    fingerprint: str = FINGERPRINT,
    mutate=None,
) -> PublishedCheckpointPair:
    binding = plan.lifecycle_binding
    launches = {launch.arm: launch for launch in plan.arms}
    rows = []
    objects = []
    for arm in ARMS:
        launch = launches[arm]
        sha = hashlib.sha256(f"{arm}-checkpoint".encode("ascii")).hexdigest()
        uri = (
            f"{plan.runtime.s3_root}/checkpoints/seed-{plan.seed}/{arm}/"
            f"sha256/{sha}.pt"
        )
        rows.append(
            {
                "arm": arm,
                "checkpoint_version": 3,
                "config_fingerprint": fingerprint,
                "config_sha256": launch.config_sha256,
                "data": {
                    "build_id": plan.dataset_build_id,
                    "global_cursor": step * 524_288,
                    "ordered_stream_sha256": plan.ordered_stream_sha256,
                    "receipt_sha256": plan.dataset_receipt_sha256,
                    "sidecar_name": _SIDECARS[arm],
                },
                "object": {
                    "bytes": 10,
                    "sha256": sha,
                    "uri": uri,
                    "version_id": f"{arm}-version",
                },
                "run_id": str(launch.runtime_config["run_id"]),
                "seed": plan.seed,
                "step": step,
                "world_size": 4,
            }
        )
        objects.append(
            VersionedUploadedObject(
                uri=uri,
                sha256=sha,
                bytes=10,
                version_id=f"{arm}-version",
            )
        )
    value = {
        **binding.to_dict(),
        "boot_id": plan.boot_id,
        "checkpoints": rows,
        "cohort_id": COHORT_ID,
        "dataset_build_id": plan.dataset_build_id,
        "dataset_receipt_sha256": plan.dataset_receipt_sha256,
        "environment_receipt_sha256": plan.environment_receipt_sha256,
        "freshness": {
            "deadline_at": "2026-07-24T01:20:00Z",
            "max_age_seconds": 1_200,
            "requested_at": "2026-07-24T01:00:00Z",
            "staged_at": "2026-07-24T01:05:00Z",
        },
        "instance_id": plan.instance_id,
        "ordered_stream_sha256": plan.ordered_stream_sha256,
        "profile_sha256": binding.profile_sha256,
        "provider": binding.provider,
        "reason": "periodic",
        "receipt_type": "memorysplit-aws-paired-checkpoint-v3",
        "release_receipt_sha256": plan.release_receipt_sha256,
        "release_sha256": plan.release_sha256,
        "request_id": request_id,
        "resumable": True,
        "run_manifest_sha256": plan.run_manifest_sha256,
        "schema_version": 3,
        "seed": plan.seed,
        "source_commit": plan.code_commit,
        "source_tree": plan.source_tree,
    }
    if mutate is not None:
        mutate(value)
    payload = _canonical(value)
    sha = hashlib.sha256(payload).hexdigest()
    return PublishedCheckpointPair(
        receipt=CheckpointReceiptRef(
            uri=(
                f"{plan.runtime.s3_root}/receipts/checkpoints/"
                f"seed-{value['seed']}/sha256/{sha}.json"
            ),
            sha256=sha,
            version_id="checkpoint-receipt-version",
            bytes=len(payload),
        ),
        checkpoints=(objects[0], objects[1]),
        value=value,
    )


def _seed_checkpoint_receipt(store, pair) -> None:
    payload = _canonical(pair.value)
    store.objects[pair.receipt.uri] = (
        payload,
        pair.receipt.sha256,
        {
            "receipt-sha256": pair.receipt.sha256,
            "receipt-type": "memorysplit-aws-paired-checkpoint-v3",
            "request-id": str(pair.value["request_id"]),
            "seed": str(pair.value["seed"]),
        },
        pair.receipt.version_id,
    )


def _authority_kwargs(fixture) -> dict[str, object]:
    binding = fixture["lifecycle"].binding
    return {
        "authority_root": fixture["tmp_path"] / "authority",
        "repo_root": fixture["repo_root"],
        "runtime_lock_path": fixture["runtime_lock"],
        "runtime_evidence_path": (
            fixture["tmp_path"] / "runtime-evidence.json"
        ),
        "runtime_sbom_path": fixture["runtime_sbom"],
        "objective_controls_amendment_path": (
            fixture["repo_root"]
            / "configs"
            / "objective-controls-amendment-v3.yaml"
        ),
        "store": object(),
        "account_id": binding.account_id,
        "instance_id": binding.instance_id,
        "boot_id": binding.boot_id,
        "expected_selection_version_id": (
            binding.provider_selection_version_id
        ),
        "identity_verifier": object(),
        "approval_verifier": object(),
        "trusted_public_key_sha256": (
            binding.qualification_approval_public_key_sha256
        ),
    }


def _install_admit(monkeypatch, lifecycles) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    results = list(lifecycles)

    def admit(**kwargs):
        calls.append(kwargs)
        result = results[min(len(calls), len(results)) - 1]
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(finalization, "admit_provider_lifecycle", admit)
    return calls


def _finalize(fixture, plan, *, pair, store, time_reader=None, **overrides):
    arguments = {
        "checkpoint_pair": pair,
        "object_store": store,
        "s3_root": plan.runtime.s3_root,
        "seed": plan.seed,
        "request_id": REQUEST_ID,
        "time_reader": time_reader or _TimeReader(),
        "staging_root": (
            fixture["scratch_root"] / "staging" / "finalization"
        ),
        **_authority_kwargs(fixture),
    }
    arguments.update(overrides)
    return finalization.finalize_paired_run(plan, **arguments)


def _prepared(tmp_path, monkeypatch, *, profile_path=P5_PROFILE, seed=1):
    fixture = _selected_fixture(tmp_path, profile_path=profile_path, seed=seed)
    plan = _authenticated_plan(fixture, monkeypatch)
    _write_run_evidence(plan)
    pair = _checkpoint_pair(plan)
    store = _MemoryStore()
    _seed_checkpoint_receipt(store, pair)
    return fixture, plan, pair, store


def _receipt_uris(store) -> list[str]:
    return [uri for uri in store.put_order if "/receipts/runs/" in uri]


@pytest.mark.parametrize(
    ("profile_path", "seed"),
    [(P5_PROFILE, 0), (P6_PROFILE, 9)],
    ids=["p5-seed0", "p6-seed9"],
)
def test_clean_pair_publishes_twelve_objects_and_one_verified_receipt(
    tmp_path,
    monkeypatch,
    profile_path,
    seed,
):
    fixture, plan, pair, store = _prepared(
        tmp_path,
        monkeypatch,
        profile_path=profile_path,
        seed=seed,
    )
    lifecycle = fixture["lifecycle"]
    binding = lifecycle.binding
    calls = _install_admit(monkeypatch, [lifecycle])

    result = _finalize(fixture, plan, pair=pair, store=store)

    assert len(calls) == 3
    for call in calls:
        assert call["expected_selection_version_id"] == (
            binding.provider_selection_version_id
        )
        assert call["seed"] == seed
        assert call["account_id"] == binding.account_id
        assert call["instance_id"] == binding.instance_id
        assert call["boot_id"] == binding.boot_id
        assert call["authority_root"] == fixture["tmp_path"] / "authority"
        assert call["runtime_lock_path"] == fixture["runtime_lock"]
        assert call["runtime_sbom_path"] == fixture["runtime_sbom"]
        assert call["trusted_public_key_sha256"] == (
            binding.qualification_approval_public_key_sha256
        )

    launches = {launch.arm: launch for launch in plan.arms}
    expected_uris = []
    for arm in ARMS:
        run_dir = Path(launches[arm].checkpoint_path).parent
        for step in SNAPSHOT_STEPS:
            payload = (run_dir / "snapshots" / f"step{step:07d}.pt").read_bytes()
            expected_uris.append(
                f"{plan.runtime.s3_root}/"
                + snapshot_object_key(
                    seed,
                    arm,
                    step,
                    hashlib.sha256(payload).hexdigest(),
                )
            )
        log_payload = (run_dir / "log.jsonl").read_bytes()
        expected_uris.append(
            f"{plan.runtime.s3_root}/"
            + log_object_key(
                seed,
                arm,
                hashlib.sha256(log_payload).hexdigest(),
            )
        )
    assert [item.uri for item in result.evidence] == expected_uris
    assert len(result.evidence) == 12
    assert store.put_order[:12] == expected_uris
    assert len(store.put_order) == 13
    assert store.put_order[-1] == result.receipt.uri
    expected_lifecycle_metadata = {
        field.replace("_", "-"): (
            ",".join(value) if field == "arms" else str(value)
        )
        for field, value in binding.to_dict().items()
        if field in LIFECYCLE_BINDING_FIELDS
    }
    assert set(expected_lifecycle_metadata) == {
        field.replace("_", "-") for field in LIFECYCLE_BINDING_FIELDS
    }
    for uri in expected_uris:
        metadata = store.objects[uri][2]
        for key, value in expected_lifecycle_metadata.items():
            assert metadata[key] == value

    payload, stored_sha256, _metadata, stored_version = store.objects[
        result.receipt.uri
    ]
    assert stored_sha256 == result.receipt.sha256
    assert stored_version == result.receipt.version_id
    assert hashlib.sha256(payload).hexdigest() == result.receipt.sha256
    assert result.receipt.bytes == len(payload)
    assert result.receipt.uri == (
        f"{plan.runtime.s3_root}/"
        + run_receipt_key(seed, result.receipt.sha256)
    )
    assert result.provider_selection_sha256 == (
        binding.provider_selection_sha256
    )
    assert result.provider_selection_version_id == (
        binding.provider_selection_version_id
    )

    value = finalization.parse_run_finalization_receipt_bytes(
        payload,
        receipt_uri=result.receipt.uri,
        receipt_sha256=result.receipt.sha256,
        receipt_version_id=result.receipt.version_id,
        expected_binding=binding,
    )
    assert value["schema_version"] == 3
    assert value["receipt_type"] == (
        "memorysplit-aws-paired-run-finalization-v3"
    )
    assert value["provider"] == fixture["profile"].provider
    assert value["profile_id"] == fixture["profile"].profile_id
    assert value["environment_receipt_sha256"] == (
        binding.qualification_environment_receipt_sha256
    )
    assert value["canary_receipt_sha256"] == (
        binding.qualification_canary_receipt_sha256
    )
    assert value["objective_controls_contract_sha256"] == (
        binding.objective_controls_contract_sha256
    )
    assert value["cohort_id"] == COHORT_ID
    assert value["seed"] == seed
    assert value["request_id"] == REQUEST_ID
    assert value["finalized_at"] == FINALIZED_AT
    assert value["release_sha256"] == plan.release_sha256
    assert value["release_receipt_sha256"] == plan.release_receipt_sha256
    assert value["run_manifest_sha256"] == plan.run_manifest_sha256
    assert value["dataset_receipt_sha256"] == plan.dataset_receipt_sha256
    assert value["dataset_build_id"] == plan.dataset_build_id
    assert value["ordered_stream_sha256"] == plan.ordered_stream_sha256
    assert value["source_commit"] == plan.code_commit
    assert value["source_tree"] == plan.source_tree
    assert value["checkpoint_receipt"] == {
        "uri": pair.receipt.uri,
        "sha256": pair.receipt.sha256,
        "version_id": pair.receipt.version_id,
    }
    assert value["complete"] is True
    arms = value["arms"]
    assert [row["arm"] for row in arms] == list(ARMS)
    for row in arms:
        launch = launches[row["arm"]]
        assert row["run_id"] == launch.runtime_config["run_id"]
        assert row["final_step"] == TERMINAL_STEP
        assert row["world_size"] == 4
        assert row["config_sha256"] == launch.config_sha256
        assert row["config_fingerprint"] == FINGERPRINT
        assert [item["step"] for item in row["snapshots"]] == list(
            SNAPSHOT_STEPS
        )
        for item in row["snapshots"]:
            assert set(item["object"]) == {
                "uri",
                "sha256",
                "bytes",
                "version_id",
            }
            assert item["object"]["version_id"] not in {"", "null"}
        assert set(row["log"]) == {"uri", "sha256", "bytes", "version_id"}


def test_finalization_requires_selected_provider_plan(tmp_path, monkeypatch):
    from tests.test_aws_p5_launcher import _load_fixture_plan

    fixture = _selected_fixture(tmp_path, profile_path=P5_PROFILE, seed=1)
    legacy = _launcher_fixture(tmp_path / "legacy", seed=1)
    plan = _load_fixture_plan(legacy)
    _install_admit(monkeypatch, [fixture["lifecycle"]])
    store = _MemoryStore()

    with pytest.raises(finalization.FinalizationError, match="selected"):
        _finalize(fixture, plan, pair=None, store=store)
    assert store.put_order == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hardware_amendment_sha256", "b" * 64),
        ("provider_selection_sha256", "b" * 64),
        ("provider_selection_version_id", "selection-version-2"),
        ("runtime_lock_sha256", "b" * 64),
        ("runtime_sbom_sha256", "b" * 64),
        ("qualification_evidence_sha256", "b" * 64),
        ("qualification_environment_receipt_sha256", "b" * 64),
        ("qualification_canary_receipt_sha256", "b" * 64),
        ("qualification_approval_receipt_sha256", "b" * 64),
        ("qualification_approval_public_key_sha256", "b" * 64),
        ("objective_controls_contract_sha256", "b" * 64),
    ],
)
def test_authority_mutation_fails_closed_before_any_upload(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    fixture, plan, pair, store = _prepared(tmp_path, monkeypatch)
    drifted = _drifted_lifecycle(fixture["lifecycle"], **{field: value})
    _install_admit(monkeypatch, [drifted])

    with pytest.raises(finalization.FinalizationError, match="drift|admission"):
        _finalize(fixture, plan, pair=pair, store=store)
    assert store.put_order == []
    assert _receipt_uris(store) == []


def test_cross_profile_admission_fails_closed(tmp_path, monkeypatch):
    fixture, plan, pair, store = _prepared(
        tmp_path,
        monkeypatch,
        profile_path=P5_PROFILE,
    )
    foreign = provider_lifecycle(load_aws_gpu_profile(P6_PROFILE), seed=1)
    _install_admit(monkeypatch, [foreign])

    with pytest.raises(finalization.FinalizationError, match="drift"):
        _finalize(fixture, plan, pair=pair, store=store)
    assert store.put_order == []


@pytest.mark.parametrize("gate", ["gate-b", "gate-c"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider_selection_version_id", "selection-version-2"),
        ("provider_selection_sha256", "b" * 64),
        ("runtime_sbom_sha256", "b" * 64),
        ("objective_controls_contract_sha256", "b" * 64),
    ],
)
def test_selection_drift_between_gates_fails_closed(
    tmp_path,
    monkeypatch,
    gate,
    field,
    value,
):
    fixture, plan, pair, store = _prepared(tmp_path, monkeypatch)
    lifecycle = fixture["lifecycle"]
    drifted = _drifted_lifecycle(lifecycle, **{field: value})
    sequence = (
        [lifecycle, drifted, drifted]
        if gate == "gate-b"
        else [lifecycle, lifecycle, drifted]
    )
    calls = _install_admit(monkeypatch, sequence)

    with pytest.raises(finalization.FinalizationError, match="drift"):
        _finalize(fixture, plan, pair=pair, store=store)

    assert _receipt_uris(store) == []
    if gate == "gate-b":
        assert len(calls) == 2
        assert store.put_order == []
    else:
        assert len(calls) == 3
        assert len(store.put_order) == 12


def test_admission_failure_between_gates_emits_no_receipt(
    tmp_path,
    monkeypatch,
):
    fixture, plan, pair, store = _prepared(tmp_path, monkeypatch)
    lifecycle = fixture["lifecycle"]
    _install_admit(
        monkeypatch,
        [lifecycle, ValueError("selection version has drifted"), lifecycle],
    )

    with pytest.raises(finalization.FinalizationError, match="admission"):
        _finalize(fixture, plan, pair=pair, store=store)
    assert store.put_order == []
    assert _receipt_uris(store) == []


def test_finalization_is_bounded_to_1200_monotonic_seconds(
    tmp_path,
    monkeypatch,
):
    fixture, plan, pair, store = _prepared(tmp_path, monkeypatch)
    _install_admit(monkeypatch, [fixture["lifecycle"]])
    reader = _TimeReader(ticks=[0.0] + [1_300.0] * 64)

    assert finalization.FINALIZATION_ATTEMPT_SECONDS == 1_200
    with pytest.raises(finalization.FinalizationError, match="deadline"):
        _finalize(fixture, plan, pair=pair, store=store, time_reader=reader)
    assert _receipt_uris(store) == []


@pytest.mark.parametrize(
    ("seed_override", "message"),
    [(10, "seed"), (True, "seed"), (2, "seed")],
    ids=["seed-ten", "seed-bool", "seed-mismatch"],
)
def test_foreign_or_mismatched_seed_fails_closed(
    tmp_path,
    monkeypatch,
    seed_override,
    message,
):
    fixture, plan, pair, store = _prepared(tmp_path, monkeypatch)
    _install_admit(monkeypatch, [fixture["lifecycle"]])

    with pytest.raises(finalization.FinalizationError, match=message):
        _finalize(fixture, plan, pair=pair, store=store, seed=seed_override)
    assert store.put_order == []


def test_wrong_s3_root_fails_closed(tmp_path, monkeypatch):
    fixture, plan, pair, store = _prepared(tmp_path, monkeypatch)
    _install_admit(monkeypatch, [fixture["lifecycle"]])

    with pytest.raises(finalization.FinalizationError, match="S3 root"):
        _finalize(
            fixture,
            plan,
            pair=pair,
            store=store,
            s3_root="s3://memorysplit-prod/other-root",
        )
    assert store.put_order == []


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-snapshot",
        "missing-terminal-snapshot",
        "extra-snapshot",
        "foreign-snapshot",
        "wrong-step-snapshot",
        "hardlinked-snapshot",
        "symlinked-snapshot",
        "wrong-mode-snapshot",
        "malformed-snapshot",
        "cross-arm-snapshot",
        "cross-selection-snapshot",
        "missing-log",
        "malformed-log",
        "nonfinite-log",
        "overflow-log",
        "truncated-log",
        "empty-log",
        "post-terminal-log",
        "nonmonotonic-log",
        "missing-terminal-log",
        "nonnumeric-log",
        "foreign-log-fields",
    ],
)
def test_local_evidence_admission_fails_closed(
    tmp_path,
    monkeypatch,
    mutation,
):
    fixture, plan, pair, store = _prepared(tmp_path, monkeypatch)
    _install_admit(monkeypatch, [fixture["lifecycle"]])
    launches = {launch.arm: launch for launch in plan.arms}
    dense_run = Path(launches["dense"].checkpoint_path).parent
    split_run = Path(launches["split90"].checkpoint_path).parent
    dense_snapshots = dense_run / "snapshots"
    split_snapshots = split_run / "snapshots"

    if mutation == "missing-snapshot":
        (dense_snapshots / "step0006791.pt").unlink()
    elif mutation == "missing-terminal-snapshot":
        (split_snapshots / "step0013582.pt").unlink()
    elif mutation == "extra-snapshot":
        (dense_snapshots / "step0000001.pt").write_bytes(b"extra")
    elif mutation == "foreign-snapshot":
        with (dense_snapshots / "step0001358.pt").open("wb") as handle:
            torch.save({"weights": {}}, handle)
    elif mutation == "wrong-step-snapshot":
        with (dense_snapshots / "step0001358.pt").open("wb") as handle:
            torch.save(
                _snapshot_state(plan, launches["dense"], 1_359),
                handle,
            )
    elif mutation == "hardlinked-snapshot":
        os.link(
            dense_snapshots / "step0001358.pt",
            dense_snapshots / "hardlink-copy",
        )
    elif mutation == "symlinked-snapshot":
        original = dense_snapshots / "step0001358.pt"
        moved = dense_snapshots / "aside.bin"
        original.rename(moved)
        original.symlink_to(moved)
    elif mutation == "wrong-mode-snapshot":
        (dense_snapshots / "step0001358.pt").chmod(0o640)
    elif mutation == "malformed-snapshot":
        (dense_snapshots / "step0001358.pt").write_bytes(b"not-a-snapshot")
    elif mutation == "cross-arm-snapshot":
        with (split_snapshots / "step0003396.pt").open("wb") as handle:
            torch.save(
                _snapshot_state(plan, launches["dense"], 3_396),
                handle,
            )
    elif mutation == "cross-selection-snapshot":
        foreign_binding = replace(
            plan.lifecycle_binding,
            provider_selection_sha256="b" * 64,
        )
        launch = launches["split90"]
        metadata = lifecycle_operational_metadata(
            foreign_binding,
            run_id=str(launch.runtime_config["run_id"]),
            arm="split90",
            config_sha256=launch.config_sha256,
            dataset_receipt_sha256=plan.dataset_receipt_sha256,
            dataset_build_id=plan.dataset_build_id,
            ordered_stream_sha256=plan.ordered_stream_sha256,
            source_commit=plan.code_commit,
            source_tree=str(plan.source_tree),
        )
        with (split_snapshots / "step0003396.pt").open("wb") as handle:
            torch.save(
                _snapshot_state(
                    plan,
                    launch,
                    3_396,
                    metadata=metadata,
                ),
                handle,
            )
    elif mutation == "missing-log":
        (dense_run / "log.jsonl").unlink()
    elif mutation == "malformed-log":
        with (dense_run / "log.jsonl").open("ab") as handle:
            handle.write(b'{"step": }\n')
    elif mutation == "nonfinite-log":
        row = json.dumps(_log_row(TERMINAL_STEP)).replace("1.5", "NaN")
        with (dense_run / "log.jsonl").open("ab") as handle:
            handle.write(row.encode("ascii") + b"\n")
    elif mutation == "overflow-log":
        row = json.dumps(_log_row(TERMINAL_STEP)).replace("1.5", "1e999")
        with (dense_run / "log.jsonl").open("ab") as handle:
            handle.write(row.encode("ascii") + b"\n")
    elif mutation == "truncated-log":
        payload = (dense_run / "log.jsonl").read_bytes()
        (dense_run / "log.jsonl").write_bytes(payload[:-1])
    elif mutation == "empty-log":
        (dense_run / "log.jsonl").write_bytes(b"")
    elif mutation == "post-terminal-log":
        with (dense_run / "log.jsonl").open("ab") as handle:
            handle.write(
                (json.dumps(_log_row(TERMINAL_STEP + 1)) + "\n").encode()
            )
    elif mutation == "nonmonotonic-log":
        with (dense_run / "log.jsonl").open("ab") as handle:
            handle.write((json.dumps(_log_row(6_791)) + "\n").encode())
            handle.write(
                (json.dumps(_log_row(TERMINAL_STEP)) + "\n").encode()
            )
    elif mutation == "missing-terminal-log":
        (dense_run / "log.jsonl").write_bytes(
            b"".join(
                (json.dumps(_log_row(step)) + "\n").encode("ascii")
                for step in (20, 1_358, 6_791)
            )
        )
    elif mutation == "nonnumeric-log":
        row = dict(_log_row(TERMINAL_STEP), loss="not-a-number")
        with (dense_run / "log.jsonl").open("ab") as handle:
            handle.write((json.dumps(row) + "\n").encode("ascii"))
    elif mutation == "foreign-log-fields":
        row = dict(_log_row(TERMINAL_STEP), gpu_temperature=80)
        with (dense_run / "log.jsonl").open("ab") as handle:
            handle.write((json.dumps(row) + "\n").encode("ascii"))
    else:  # pragma: no cover - parametrization is closed
        raise AssertionError(mutation)

    with pytest.raises(finalization.FinalizationError):
        _finalize(fixture, plan, pair=pair, store=store)
    assert store.put_order == []
    assert _receipt_uris(store) == []


def test_local_evidence_owner_must_match_runtime_identity(
    tmp_path,
    monkeypatch,
):
    fixture, plan, pair, store = _prepared(tmp_path, monkeypatch)
    _install_admit(monkeypatch, [fixture["lifecycle"]])
    foreign_owner_plan = replace(plan, runtime_uid=plan.runtime_uid + 1)

    with pytest.raises(finalization.FinalizationError, match="owner|mode"):
        _finalize(
            fixture,
            foreign_owner_plan,
            pair=pair,
            store=store,
        )
    assert store.put_order == []
    assert _receipt_uris(store) == []


def test_snapshot_fingerprint_must_match_the_checkpoint_receipt(
    tmp_path,
    monkeypatch,
):
    fixture, plan, pair, store = _prepared(tmp_path, monkeypatch)
    _install_admit(monkeypatch, [fixture["lifecycle"]])
    launches = {launch.arm: launch for launch in plan.arms}
    dense_snapshots = Path(launches["dense"].checkpoint_path).parent / "snapshots"
    with (dense_snapshots / "step0003396.pt").open("wb") as handle:
        torch.save(
            _snapshot_state(
                plan,
                launches["dense"],
                3_396,
                fingerprint="e" * 64,
            ),
            handle,
        )

    with pytest.raises(finalization.FinalizationError, match="fingerprint"):
        _finalize(fixture, plan, pair=pair, store=store)
    assert store.put_order == []


def test_checkpoint_receipt_is_required_and_identity_bound(
    tmp_path,
    monkeypatch,
):
    fixture, plan, pair, store = _prepared(tmp_path, monkeypatch)
    _install_admit(monkeypatch, [fixture["lifecycle"]])

    with pytest.raises(
        finalization.FinalizationError,
        match="checkpoint receipt",
    ):
        _finalize(fixture, plan, pair=None, store=store)

    stale = PublishedCheckpointPair(
        receipt=dataclasses.replace(pair.receipt, sha256="9" * 64),
        checkpoints=pair.checkpoints,
        value=pair.value,
    )
    with pytest.raises(
        finalization.FinalizationError,
        match="checkpoint receipt",
    ):
        _finalize(fixture, plan, pair=stale, store=store)

    for field, value in (
        ("run_manifest_sha256", "9" * 64),
        ("release_sha256", "9" * 64),
        ("boot_id", "9" * 32),
        ("provider_selection_version_id", "selection-version-2"),
    ):
        mutated = _checkpoint_pair(
            plan,
            mutate=lambda receipt, field=field, value=value: receipt.update(
                {field: value}
            ),
        )
        seeded = _MemoryStore()
        _seed_checkpoint_receipt(seeded, mutated)
        with pytest.raises(finalization.FinalizationError):
            _finalize(fixture, plan, pair=mutated, store=seeded)
        assert _receipt_uris(seeded) == []

    assert store.put_order == []


def test_unverified_checkpoint_receipt_object_prevents_finalization(
    tmp_path,
    monkeypatch,
):
    fixture, plan, pair, _seeded = _prepared(tmp_path, monkeypatch)
    _install_admit(monkeypatch, [fixture["lifecycle"]])
    empty_store = _MemoryStore()

    with pytest.raises(finalization.FinalizationError, match="unverified"):
        _finalize(fixture, plan, pair=pair, store=empty_store)
    assert empty_store.put_order == []


def test_legacy_checkpoint_provider_token_is_schema_only(
    tmp_path,
    monkeypatch,
):
    from msctl.contracts import parse_paired_checkpoint_receipt_v3
    from msctl.errors import MsctlError

    fixture, plan, pair, store = _prepared(tmp_path, monkeypatch)
    _install_admit(monkeypatch, [fixture["lifecycle"]])

    lifecycle_fields = set(plan.lifecycle_binding.to_dict())
    legacy_value = {
        field: value
        for field, value in pair.value.items()
        if field not in lifecycle_fields
        or field
        in {
            "boot_id",
            "cohort_id",
            "instance_id",
            "profile_sha256",
            "provider",
            "seed",
        }
    }
    legacy_value["provider"] = "aws-p5.48xlarge"
    legacy_payload = _canonical(legacy_value)
    legacy_sha256 = hashlib.sha256(legacy_payload).hexdigest()
    legacy_uri = (
        f"{plan.runtime.s3_root}/receipts/checkpoints/"
        f"seed-{plan.seed}/sha256/{legacy_sha256}.json"
    )

    parsed = parse_paired_checkpoint_receipt_v3(
        legacy_payload,
        receipt_uri=legacy_uri,
        receipt_sha256=legacy_sha256,
        receipt_version_id="legacy-version",
    )
    assert parsed.value["provider"] == "aws-p5.48xlarge"
    assert parsed.provider_selection_sha256 is None

    foreign_value = dict(legacy_value, provider="aws-p6-b300.48xlarge")
    foreign_payload = _canonical(foreign_value)
    foreign_sha256 = hashlib.sha256(foreign_payload).hexdigest()
    with pytest.raises(MsctlError):
        parse_paired_checkpoint_receipt_v3(
            foreign_payload,
            receipt_uri=(
                f"{plan.runtime.s3_root}/receipts/checkpoints/"
                f"seed-{plan.seed}/sha256/{foreign_sha256}.json"
            ),
            receipt_sha256=foreign_sha256,
            receipt_version_id="legacy-version",
        )

    legacy_pair = PublishedCheckpointPair(
        receipt=CheckpointReceiptRef(
            uri=legacy_uri,
            sha256=legacy_sha256,
            version_id="legacy-version",
            bytes=len(legacy_payload),
        ),
        checkpoints=pair.checkpoints,
        value=legacy_value,
    )
    with pytest.raises(
        finalization.FinalizationError,
        match="provider-aware",
    ):
        _finalize(fixture, plan, pair=legacy_pair, store=store)
    assert store.put_order == []


@pytest.mark.parametrize(
    "fault",
    [
        "partial-upload",
        "head-missing",
        "head-version-drift",
        "head-sha-drift",
        "head-bytes-drift",
        "missing-version",
        "null-version",
    ],
)
def test_publication_faults_emit_no_receipt(tmp_path, monkeypatch, fault):
    fixture, plan, pair, _store = _prepared(tmp_path, monkeypatch)
    _install_admit(monkeypatch, [fixture["lifecycle"]])
    store = _FaultyStore(target="/split90/", fault=fault)
    _seed_checkpoint_receipt(store, pair)

    with pytest.raises(finalization.FinalizationError):
        _finalize(fixture, plan, pair=pair, store=store)
    assert _receipt_uris(store) == []
    assert any("/dense/" in uri for uri in store.put_order)


def test_metadata_drift_on_head_emits_no_receipt(tmp_path, monkeypatch):
    fixture, plan, pair, _store = _prepared(tmp_path, monkeypatch)
    _install_admit(monkeypatch, [fixture["lifecycle"]])

    class _MetadataDriftStore(_MemoryStore):
        def put_if_absent(self, path, uri, **kwargs):
            uploaded = super().put_if_absent(path, uri, **kwargs)
            if "/split90/" in uri:
                payload, sha256, metadata, version = self.objects[uri]
                metadata = dict(metadata, arm="dense")
                self.objects[uri] = (payload, sha256, metadata, version)
            return uploaded

    store = _MetadataDriftStore()
    _seed_checkpoint_receipt(store, pair)

    with pytest.raises(finalization.FinalizationError):
        _finalize(fixture, plan, pair=pair, store=store)
    assert _receipt_uris(store) == []


def test_lost_put_recovers_only_through_exact_head(tmp_path, monkeypatch):
    fixture, plan, pair, _store = _prepared(tmp_path, monkeypatch)
    lifecycle = fixture["lifecycle"]
    _install_admit(monkeypatch, [lifecycle])
    recovered = _LostPutStore(recoverable=True)
    _seed_checkpoint_receipt(recovered, pair)

    result = _finalize(fixture, plan, pair=pair, store=recovered)
    assert len(result.evidence) == 12
    assert len(_receipt_uris(recovered)) == 1

    _install_admit(monkeypatch, [lifecycle])
    lost = _LostPutStore(recoverable=False)
    _seed_checkpoint_receipt(lost, pair)
    with pytest.raises(finalization.FinalizationError):
        _finalize(fixture, plan, pair=pair, store=lost)
    assert _receipt_uris(lost) == []


def test_config_rehash_failures_fail_closed(tmp_path, monkeypatch):
    fixture, plan, pair, store = _prepared(tmp_path, monkeypatch)
    lifecycle = fixture["lifecycle"]
    _install_admit(monkeypatch, [lifecycle])
    config_path = Path(plan.arms[0].config_path)
    config_path.chmod(0o644)
    config_path.write_bytes(config_path.read_bytes() + b"# drift\n")

    with pytest.raises(finalization.FinalizationError, match="config"):
        _finalize(fixture, plan, pair=pair, store=store)
    assert store.put_order == []


def test_gate_c_config_rewrite_emits_no_receipt(tmp_path, monkeypatch):
    fixture, plan, pair, store = _prepared(tmp_path, monkeypatch)
    lifecycle = fixture["lifecycle"]
    config_path = Path(plan.arms[0].config_path)
    calls: list[dict[str, object]] = []

    def admit(**kwargs):
        calls.append(kwargs)
        if len(calls) == 3:
            config_path.chmod(0o644)
            config_path.write_bytes(config_path.read_bytes() + b"# drift\n")
        return lifecycle

    monkeypatch.setattr(finalization, "admit_provider_lifecycle", admit)

    with pytest.raises(finalization.FinalizationError, match="config"):
        _finalize(fixture, plan, pair=pair, store=store)
    assert len(calls) == 3
    assert len(store.put_order) == 12
    assert _receipt_uris(store) == []


def test_run_manifest_drift_after_admission_fails_closed(
    tmp_path,
    monkeypatch,
):
    fixture, plan, pair, store = _prepared(tmp_path, monkeypatch)
    _install_admit(monkeypatch, [fixture["lifecycle"]])
    manifest = dict(fixture["manifest"])
    manifest["provider_selection_version_id"] = "selection-version-2"
    _write_json(fixture["manifest_path"], manifest)

    with pytest.raises(finalization.FinalizationError, match="manifest"):
        _finalize(fixture, plan, pair=pair, store=store)
    assert store.put_order == []


def test_release_metadata_drift_after_admission_fails_closed(
    tmp_path,
    monkeypatch,
):
    fixture, plan, pair, store = _prepared(tmp_path, monkeypatch)
    _install_admit(monkeypatch, [fixture["lifecycle"]])
    metadata_path = fixture["metadata_path"]
    metadata_path.chmod(0o644)
    metadata = json.loads(metadata_path.read_text())
    metadata["provider_selection_sha256"] = "b" * 64
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    with pytest.raises(finalization.FinalizationError, match="release"):
        _finalize(fixture, plan, pair=pair, store=store)
    assert store.put_order == []


def _published_receipt(tmp_path, monkeypatch):
    fixture, plan, pair, store = _prepared(tmp_path, monkeypatch)
    binding = fixture["lifecycle"].binding
    _install_admit(monkeypatch, [fixture["lifecycle"]])
    result = _finalize(fixture, plan, pair=pair, store=store)
    payload = store.objects[result.receipt.uri][0]
    value = json.loads(payload.decode("ascii"))
    return plan, binding, result, value


def test_receipt_parser_rejects_every_field_and_type_mutation(
    tmp_path,
    monkeypatch,
):
    plan, binding, result, clean = _published_receipt(tmp_path, monkeypatch)
    s3_root = plan.runtime.s3_root

    def receipt_key(seed, sha256):
        try:
            return run_receipt_key(seed, sha256)
        except ValueError:
            return f"receipts/runs/seed-{seed}/sha256/{sha256}.json"

    def reparse(value, *, uri=None, version_id=None, binding_check=binding):
        payload = _canonical(value)
        sha256 = hashlib.sha256(payload).hexdigest()
        return finalization.parse_run_finalization_receipt_bytes(
            payload,
            receipt_uri=(
                uri
                if uri is not None
                else f"{s3_root}/" + receipt_key(value["seed"], sha256)
            ),
            receipt_sha256=sha256,
            receipt_version_id=(
                version_id if version_id is not None else "receipt-version"
            ),
            expected_binding=binding_check,
        )

    assert reparse(clean)["complete"] is True

    def drop(field):
        def mutate(value):
            del value[field]

        return mutate

    def put(field, new_value):
        def mutate(value):
            value[field] = new_value

        return mutate

    def arm_row(index, field, new_value):
        def mutate(value):
            value["arms"][index][field] = new_value

        return mutate

    mutations = [
        ("drop-schema", drop("schema_version")),
        ("extra-field", put("extra", "value")),
        ("schema-two", put("schema_version", 2)),
        ("schema-text", put("schema_version", "3")),
        ("foreign-receipt-type", put("receipt_type", "memorysplit-x")),
        ("foreign-cohort", put("cohort_id", "memorysplit-v2")),
        ("foreign-provider", put("provider", "aws-p7")),
        (
            "provider-profile-mismatch",
            put("provider", "aws-p6-b300.48xlarge"),
        ),
        ("seed-text", put("seed", "1")),
        ("seed-ten", put("seed", 10)),
        ("request-id-short", put("request_id", "d" * 31)),
        (
            "noncanonical-time",
            put("finalized_at", "2026-07-24T02:00:00+00:00"),
        ),
        ("incomplete", put("complete", False)),
        ("complete-text", put("complete", "true")),
        ("release-invalid", put("release_sha256", "Z" * 64)),
        ("commit-short", put("source_commit", "4" * 39)),
        ("amendment-drift", put("hardware_amendment_sha256", "b" * 64)),
        ("selection-drift", put("provider_selection_sha256", "b" * 64)),
        (
            "selection-version-drift",
            put("provider_selection_version_id", "selection-version-2"),
        ),
        ("sbom-drift", put("runtime_sbom_sha256", "b" * 64)),
        (
            "objective-drift",
            put("objective_controls_contract_sha256", "b" * 64),
        ),
        (
            "checkpoint-missing-field",
            lambda value: value["checkpoint_receipt"].pop("uri"),
        ),
        (
            "checkpoint-extra-field",
            lambda value: value["checkpoint_receipt"].update({"bytes": 10}),
        ),
        (
            "checkpoint-null-version",
            lambda value: value["checkpoint_receipt"].update(
                {"version_id": "null"}
            ),
        ),
        (
            "checkpoint-foreign-root",
            lambda value: value["checkpoint_receipt"].update(
                {
                    "uri": (
                        "s3://foreign-bucket/receipts/checkpoints/seed-1/"
                        "sha256/"
                        + value["checkpoint_receipt"]["sha256"]
                        + ".json"
                    )
                }
            ),
        ),
        (
            "arms-reordered",
            lambda value: value.update({"arms": value["arms"][::-1]}),
        ),
        (
            "arms-single",
            lambda value: value.update({"arms": value["arms"][:1]}),
        ),
        ("arm-extra-field", arm_row(0, "extra", "value")),
        (
            "arm-missing-field",
            lambda value: value["arms"][0].pop("final_step"),
        ),
        ("arm-nonterminal", arm_row(0, "final_step", 13_581)),
        ("arm-step-text", arm_row(0, "final_step", "13582")),
        ("arm-world-size", arm_row(1, "world_size", 8)),
        ("arm-fingerprint", arm_row(0, "config_fingerprint", "Z" * 64)),
        (
            "log-bytes-zero",
            lambda value: value["arms"][0]["log"].update({"bytes": 0}),
        ),
        (
            "log-bytes-float",
            lambda value: value["arms"][0]["log"].update({"bytes": 1.0}),
        ),
        (
            "log-version-empty",
            lambda value: value["arms"][0]["log"].update(
                {"version_id": ""}
            ),
        ),
        (
            "log-uri-cross-arm",
            lambda value: value["arms"][0]["log"].update(
                {"uri": value["arms"][1]["log"]["uri"]}
            ),
        ),
        (
            "snapshots-reordered",
            lambda value: value["arms"][0].update(
                {"snapshots": value["arms"][0]["snapshots"][::-1]}
            ),
        ),
        (
            "snapshots-missing-step",
            lambda value: value["arms"][0].update(
                {"snapshots": value["arms"][0]["snapshots"][:4]}
            ),
        ),
        (
            "snapshots-duplicate-step",
            lambda value: value["arms"][0].update(
                {
                    "snapshots": value["arms"][0]["snapshots"][:4]
                    + value["arms"][0]["snapshots"][3:4]
                }
            ),
        ),
        (
            "snapshot-off-schedule",
            lambda value: value["arms"][0]["snapshots"][0].update(
                {"step": 1_359}
            ),
        ),
        (
            "snapshot-uri-sha-mismatch",
            lambda value: value["arms"][0]["snapshots"][0]["object"].update(
                {"sha256": "b" * 64}
            ),
        ),
        (
            "snapshot-null-version",
            lambda value: value["arms"][0]["snapshots"][0]["object"].update(
                {"version_id": "null"}
            ),
        ),
        (
            "snapshot-negative-bytes",
            lambda value: value["arms"][0]["snapshots"][0]["object"].update(
                {"bytes": -1}
            ),
        ),
        (
            "snapshot-extra-object-field",
            lambda value: value["arms"][0]["snapshots"][0]["object"].update(
                {"etag": "abc"}
            ),
        ),
    ]

    for label, mutate in mutations:
        value = json.loads(json.dumps(clean))
        mutate(value)
        with pytest.raises(ValueError, match="finalization"):
            reparse(value)
        del label

    with pytest.raises(ValueError, match="finalization"):
        reparse(clean, version_id="null")
    with pytest.raises(ValueError, match="finalization"):
        reparse(
            clean,
            uri=f"s3://foreign-bucket/"
            + run_receipt_key(
                clean["seed"],
                hashlib.sha256(_canonical(clean)).hexdigest(),
            ),
        )
    with pytest.raises(ValueError, match="finalization"):
        finalization.parse_run_finalization_receipt_bytes(
            _canonical(clean),
            receipt_uri=result.receipt.uri,
            receipt_sha256="b" * 64,
            receipt_version_id="receipt-version",
        )
    with pytest.raises(ValueError, match="finalization"):
        reparse(
            clean,
            binding_check=replace(
                binding,
                provider_selection_sha256="b" * 64,
            ),
        )
    noncanonical = _canonical(clean)[:-1] + b" \n"
    with pytest.raises(ValueError, match="finalization"):
        finalization.parse_run_finalization_receipt_bytes(
            noncanonical,
            receipt_uri=result.receipt.uri,
            receipt_sha256=hashlib.sha256(noncanonical).hexdigest(),
            receipt_version_id="receipt-version",
        )


class _RecordingFinalizer:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls: list[object] = []
        self.error = error
        self.result = finalization.FinalizationResult(
            receipt=finalization.PublishedRunReceipt(
                uri="s3://memorysplit-prod/cohort-v3/receipts/runs/"
                "seed-1/sha256/" + "a" * 64 + ".json",
                sha256="a" * 64,
                bytes=321,
                version_id="run-receipt-version",
            ),
            provider_selection_sha256="2" * 64,
            provider_selection_version_id="selection-version-1",
            evidence=(),
        )

    def __call__(self, latest):
        self.calls.append(latest)
        if self.error is not None:
            raise self.error
        return self.result


class _LatestOnlyScheduler:
    """Mirror scheduler stub carrying one complete published pair."""

    def __init__(self, *, latest=None) -> None:
        self.latest = latest
        self.active = False
        self.cancelled = False

    def maybe_start(self, _request, *, now=None, immediate=False):
        del now, immediate
        return False

    def poll(self, *, now=None):
        del now
        return None

    def cancel_active(self):
        self.cancelled = True
        self.active = False


def _latest_pair_sentinel() -> PublishedCheckpointPair:
    return PublishedCheckpointPair(
        receipt=CheckpointReceiptRef(
            uri="s3://bucket/receipts/checkpoints/seed-1/sha256/"
            + "b" * 64
            + ".json",
            sha256="b" * 64,
            version_id="checkpoint-version",
            bytes=99,
        ),
        checkpoints=(
            VersionedUploadedObject("s3://bucket/dense.pt", "c" * 64, 10, "v1"),
            VersionedUploadedObject(
                "s3://bucket/split90.pt", "d" * 64, 10, "v2"
            ),
        ),
        value={},
    )


def _supervise(plan, *, finalizer, processes, **kwargs):
    from tests.test_aws_p5_launcher import (
        _FakeSpawner,
        _pass_rank_zero_resolver,
        _pass_trainer_preflight,
    )

    return launch_module.supervise_pair(
        plan,
        spawner=_FakeSpawner(processes),
        sleep=lambda _delay: None,
        trainer_preflight=_pass_trainer_preflight,
        rank_zero_resolver=_pass_rank_zero_resolver,
        finalizer=finalizer,
        **kwargs,
    )


def test_launcher_success_requires_finalization_and_exposes_references(
    tmp_path,
):
    from tests.test_aws_p5_launcher import _FakeProcess, _load_fixture_plan

    plan = _load_fixture_plan(_launcher_fixture(tmp_path))
    latest = _latest_pair_sentinel()
    scheduler = _LatestOnlyScheduler(latest=latest)
    finalizer = _RecordingFinalizer()

    result = _supervise(
        plan,
        finalizer=finalizer,
        processes={
            "dense": _FakeProcess(101, [None, 0]),
            "split90": _FakeProcess(202, [None, 0]),
        },
        checkpoint_scheduler_factory=lambda _plan, _pids: (
            scheduler,
            lambda reason: SimpleNamespace(reason=reason),
        ),
    )

    assert result.status == "completed"
    assert result.returncode == 0
    assert finalizer.calls == [latest]
    assert finalizer.calls[0] is latest
    assert scheduler.cancelled is True
    assert result.finalization_receipt_uri == finalizer.result.receipt.uri
    assert result.finalization_receipt_version_id == (
        finalizer.result.receipt.version_id
    )
    assert result.provider_selection_sha256 == "2" * 64
    assert result.provider_selection_version_id == "selection-version-1"


def test_selected_provider_supervision_requires_finalizer(
    tmp_path,
    monkeypatch,
):
    from tests.test_aws_p5_launcher import _FakeProcess

    fixture = _selected_fixture(tmp_path, profile_path=P5_PROFILE, seed=1)
    plan = _authenticated_plan(fixture, monkeypatch)

    result = _supervise(
        plan,
        finalizer=None,
        processes={
            "dense": _FakeProcess(101, [None, 0]),
            "split90": _FakeProcess(202, [None, 0]),
        },
        checkpoint_scheduler_factory=lambda _plan, _pids: (
            _LatestOnlyScheduler(latest=_latest_pair_sentinel()),
            lambda reason: SimpleNamespace(reason=reason),
        ),
    )

    assert result.status == "FINALIZATION_FAILED"
    assert result.status != "completed"
    assert result.returncode != 0
    assert result.finalization_receipt_uri is None
    assert result.finalization_receipt_version_id is None
    assert result.provider_selection_sha256 is None
    assert result.provider_selection_version_id is None


def test_failing_finalization_is_reported_and_never_completed(tmp_path):
    from tests.test_aws_p5_launcher import _FakeProcess, _load_fixture_plan

    plan = _load_fixture_plan(_launcher_fixture(tmp_path))
    finalizer = _RecordingFinalizer(
        error=finalization.FinalizationError(
            "run finalization provider authority drifted at gate C"
        )
    )

    result = _supervise(
        plan,
        finalizer=finalizer,
        processes={
            "dense": _FakeProcess(101, [None, 0]),
            "split90": _FakeProcess(202, [None, 0]),
        },
        checkpoint_scheduler_factory=lambda _plan, _pids: (
            _LatestOnlyScheduler(latest=_latest_pair_sentinel()),
            lambda reason: SimpleNamespace(reason=reason),
        ),
    )

    assert result.status == "FINALIZATION_FAILED"
    assert result.status != "completed"
    assert result.returncode != 0
    assert len(finalizer.calls) == 1
    assert result.finalization_receipt_uri is None
    assert result.finalization_receipt_version_id is None
    assert result.provider_selection_sha256 is None
    assert result.provider_selection_version_id is None


@pytest.mark.parametrize(
    "exit_path",
    ["arm-failure", "interruption", "checkpoint-stale", "shutdown"],
)
def test_failure_paths_never_invoke_finalization(tmp_path, exit_path):
    from tests.test_aws_p5_launcher import _FakeProcess, _load_fixture_plan

    plan = _load_fixture_plan(_launcher_fixture(tmp_path))
    finalizer = _RecordingFinalizer()

    class _StaleScheduler(_LatestOnlyScheduler):
        def poll(self, *, now=None):
            del now
            from cluster.aws.p5.checkpoint_mirror import CheckpointStaleError

            raise CheckpointStaleError(
                "CHECKPOINT_STALE: paired durability deadline expired"
            )

    kwargs: dict[str, object] = {}
    processes = {
        "dense": _FakeProcess(101, [None, 0]),
        "split90": _FakeProcess(202, [None, 0]),
    }
    if exit_path == "arm-failure":
        processes["dense"] = _FakeProcess(101, [None, 1])
        processes["split90"] = _FakeProcess(202, [None, None])
    elif exit_path == "interruption":
        kwargs["notice_source"] = lambda: "spot-interruption"
        kwargs["interruption_handler"] = None
    elif exit_path == "checkpoint-stale":
        kwargs["checkpoint_scheduler_factory"] = lambda _plan, _pids: (
            _StaleScheduler(),
            lambda reason: SimpleNamespace(reason=reason),
        )
    elif exit_path == "shutdown":
        kwargs["shutdown_source"] = lambda: 15

    result = _supervise(
        plan,
        finalizer=finalizer,
        processes=processes,
        **kwargs,
    )

    assert result.status in {
        "failed",
        "interrupted",
        "CHECKPOINT_STALE",
        "terminated",
    }
    assert result.returncode != 0
    assert finalizer.calls == []
    assert result.finalization_receipt_uri is None
    assert result.finalization_receipt_version_id is None
    assert result.provider_selection_sha256 is None
    assert result.provider_selection_version_id is None


def test_supervision_without_finalizer_keeps_legacy_result_shape(tmp_path):
    from tests.test_aws_p5_launcher import (
        _FakeProcess,
        _FakeSpawner,
        _load_fixture_plan,
    )

    plan = _load_fixture_plan(_launcher_fixture(tmp_path))
    result = launch_module.supervise_pair(
        plan,
        spawner=_FakeSpawner(
            {
                "dense": _FakeProcess(101, [None, 0]),
                "split90": _FakeProcess(202, [None, 0]),
            }
        ),
        sleep=lambda _delay: None,
        trainer_preflight=lambda _plan: None,
        rank_zero_resolver=lambda _plan, pids: dict(pids),
    )

    assert result.status == "completed"
    assert result.returncode == 0
    assert result.finalization_receipt_uri is None
    assert result.finalization_receipt_version_id is None
    assert result.provider_selection_sha256 is None
    assert result.provider_selection_version_id is None
