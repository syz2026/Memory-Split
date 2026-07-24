from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

import evals.confirmatory as confirmatory
import evals.confirmatory.lock_builder as lock_builder
from evals.confirmatory.aggregate import (
    CollectionReceiptEvidence,
    plan_snapshot_evaluations,
)
from evals.confirmatory.contracts import (
    STUDY_CONTRACT_VERSION,
    canonical_json_bytes,
    canonical_sha256,
)
from evals.confirmatory.lock_builder import (
    STUDY_LOCK_DIRECTORY_PREFIX,
    LockBuildError,
    PublishedStudyLock,
    RunStudyIdentity,
    SeedCollectionEvidence,
    build_study_lock_v3,
    extract_run_study_identities,
    publish_study_lock,
)
from evals.confirmatory.study_lock import (
    FROZEN_PREREGISTRATION_SHA256_V3,
    STUDY_LOCK_SCHEMA_V3,
    StudyLockV3,
)
from msctl.aws_contracts import ARMS, SEEDS, SNAPSHOT_STEPS, snapshot_object_key
from msctl.aws_lifecycle import LIFECYCLE_BINDING_FIELDS
from scripts import build_confirmatory_study_lock as lock_cli
from tests.study_lock_fixtures import (
    S3_ROOT,
    build_collected_evidence,
    digest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEALED_RELEASE_SHA256 = "f" * 64
LOCK_MEMBER = "study-lock.json"


@pytest.fixture(scope="module")
def canonical_bundle():
    return build_collected_evidence()


@pytest.fixture(scope="module")
def canonical_lock(canonical_bundle, tmp_path_factory):
    root = tmp_path_factory.mktemp("canonical-lock")
    evidence = _evidence_rows(canonical_bundle)
    identities = extract_run_study_identities(
        evidence,
        snapshot_paths=_snapshot_paths(canonical_bundle, root),
    )
    return build_study_lock_v3(
        sealed_evaluation_release_sha256=SEALED_RELEASE_SHA256,
        evidence=evidence,
        run_study_identities=identities,
    )


def _evidence_rows(bundle) -> list[SeedCollectionEvidence]:
    return [
        SeedCollectionEvidence(
            collection=CollectionReceiptEvidence(
                payload=payload,
                uri=uri,
                sha256=sha256,
                version_id=version_id,
            ),
            finalization_payload=bundle.finalization_payloads[seed],
            checkpoint_receipt_payload=(
                bundle.checkpoint_receipt_payloads[seed]
            ),
        )
        for seed, (payload, uri, sha256, version_id) in enumerate(
            bundle.collection_receipts
        )
    ]


def _collection_evidence(bundle) -> tuple[CollectionReceiptEvidence, ...]:
    return tuple(row.collection for row in _evidence_rows(bundle))


def _snapshot_paths(bundle, root: Path) -> dict[tuple[int, str, int], Path]:
    directory = Path(root) / "snapshot-files"
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[tuple[int, str, int], Path] = {}
    for (seed, arm, step), payload in bundle.snapshot_payloads.items():
        path = directory / f"seed-{seed}-{arm}-step-{step}.pt"
        path.write_bytes(payload)
        paths[(seed, arm, step)] = path
    return paths


def _fixture_lock(bundle) -> StudyLockV3:
    return StudyLockV3.from_dict(
        {
            "record_type": STUDY_LOCK_SCHEMA_V3,
            "schema_version": STUDY_CONTRACT_VERSION,
            "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256_V3,
            "sealed_evaluation_release_sha256": SEALED_RELEASE_SHA256,
            "provider_selection": bundle.provider_selection,
            "snapshots": bundle.snapshots,
            "seed_lifecycles": bundle.seed_lifecycles,
        }
    )


def _built_lock(bundle, root: Path) -> StudyLockV3:
    evidence = _evidence_rows(bundle)
    identities = extract_run_study_identities(
        evidence,
        snapshot_paths=_snapshot_paths(bundle, root),
    )
    return build_study_lock_v3(
        sealed_evaluation_release_sha256=SEALED_RELEASE_SHA256,
        evidence=evidence,
        run_study_identities=identities,
    )


def _publish_root(tmp_path: Path, name: str = "locks") -> Path:
    root = (Path(tmp_path) / name).resolve()
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    return root


def _quarantine_directories(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.name.startswith(".sealed-release-quarantine-")
        or path.name.endswith(".staging")
    )


def test_lock_builder_module_exports_its_closed_public_interface():
    assert confirmatory.SeedCollectionEvidence is SeedCollectionEvidence
    assert confirmatory.RunStudyIdentity is RunStudyIdentity
    assert confirmatory.PublishedStudyLock is PublishedStudyLock
    assert (
        confirmatory.extract_run_study_identities
        is extract_run_study_identities
    )
    assert confirmatory.build_study_lock_v3 is build_study_lock_v3
    assert confirmatory.publish_study_lock is publish_study_lock
    assert issubclass(LockBuildError, ValueError)

    identity = RunStudyIdentity(
        seed=0,
        arm="dense",
        model_config_sha256="a" * 64,
        model_identity="d360m",
        data_provenance_sha256="b" * 64,
        config_fingerprint="c" * 64,
    )
    with pytest.raises(FrozenInstanceError):
        identity.seed = 1
    with pytest.raises(ValueError, match="seed|arm|SHA-256|identity"):
        RunStudyIdentity(
            seed=0,
            arm="split",
            model_config_sha256="a" * 64,
            model_identity="d360m",
            data_provenance_sha256="b" * 64,
            config_fingerprint="c" * 64,
        )
    with pytest.raises(ValueError, match="collection|evidence"):
        SeedCollectionEvidence(
            collection=("not", "collection", "evidence"),
            finalization_payload=b"x",
            checkpoint_receipt_payload=b"y",
        )


def test_extraction_returns_twenty_receipt_bound_run_identities(
    canonical_bundle,
    tmp_path,
):
    evidence = _evidence_rows(canonical_bundle)

    identities = extract_run_study_identities(
        evidence,
        snapshot_paths=_snapshot_paths(canonical_bundle, tmp_path),
    )

    assert len(identities) == 20
    assert [
        (identity.seed, identity.arm) for identity in identities
    ] == [(seed, arm) for seed in SEEDS for arm in ARMS]
    assert all(
        identity.config_fingerprint
        == digest(f"config:{identity.seed}:{identity.arm}")
        for identity in identities
    )
    assert {
        identity.model_config_sha256 for identity in identities
    } == {canonical_bundle.snapshots[0]["model_config_sha256"]}
    assert {identity.model_identity for identity in identities} == {"d360m"}
    assert len(
        {identity.data_provenance_sha256 for identity in identities}
    ) == 20


def test_builder_reproduces_the_canonical_fixture_lock_byte_for_byte(
    canonical_bundle,
    canonical_lock,
):
    fixture_lock = _fixture_lock(canonical_bundle)

    assert canonical_lock == fixture_lock
    assert canonical_json_bytes(canonical_lock.to_dict()) == (
        canonical_json_bytes(fixture_lock.to_dict())
    )
    assert canonical_sha256(canonical_lock.to_dict()) == (
        canonical_sha256(fixture_lock.to_dict())
    )


@pytest.mark.parametrize(
    "mutation",
    ["nine", "eleven", "duplicate", "descending", "foreign-type"],
)
def test_builder_requires_exactly_ten_ascending_seed_collections(
    canonical_bundle,
    mutation,
):
    evidence = _evidence_rows(canonical_bundle)
    if mutation == "nine":
        evidence.pop()
    elif mutation == "eleven":
        evidence.append(evidence[-1])
    elif mutation == "duplicate":
        evidence[1] = evidence[0]
    elif mutation == "descending":
        evidence.reverse()
    else:
        evidence[0] = (
            evidence[0].collection,
            evidence[0].finalization_payload,
            evidence[0].checkpoint_receipt_payload,
        )

    with pytest.raises(ValueError, match="ten|seed|evidence"):
        build_study_lock_v3(
            sealed_evaluation_release_sha256=SEALED_RELEASE_SHA256,
            evidence=evidence,
            run_study_identities=(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sha256", "0" * 64, "hash|receipt"),
        ("payload", b"{}\n", "hash|canonical|fields"),
        (
            "uri",
            None,
            "root|canonical key",
        ),
    ],
)
def test_builder_rejects_crossed_collection_receipt_identity(
    canonical_bundle,
    field,
    value,
    message,
):
    evidence = _evidence_rows(canonical_bundle)
    if field == "uri":
        value = evidence[5].collection.uri.replace(
            S3_ROOT,
            "s3://memorysplit-other/confirmatory-v3",
        )
        evidence[5] = SeedCollectionEvidence(
            collection=replace(evidence[5].collection, uri=value),
            finalization_payload=evidence[5].finalization_payload,
            checkpoint_receipt_payload=(
                evidence[5].checkpoint_receipt_payload
            ),
        )
    else:
        evidence[0] = SeedCollectionEvidence(
            collection=replace(evidence[0].collection, **{field: value}),
            finalization_payload=evidence[0].finalization_payload,
            checkpoint_receipt_payload=(
                evidence[0].checkpoint_receipt_payload
            ),
        )

    with pytest.raises(ValueError, match=message):
        build_study_lock_v3(
            sealed_evaluation_release_sha256=SEALED_RELEASE_SHA256,
            evidence=evidence,
            run_study_identities=(),
        )


@pytest.mark.parametrize("payload", ["finalization", "checkpoint"])
def test_builder_rejects_receipt_payload_hash_drift(
    canonical_bundle,
    payload,
):
    evidence = _evidence_rows(canonical_bundle)
    evidence[0] = SeedCollectionEvidence(
        collection=evidence[0].collection,
        finalization_payload=(
            evidence[1].finalization_payload
            if payload == "finalization"
            else evidence[0].finalization_payload
        ),
        checkpoint_receipt_payload=(
            evidence[1].checkpoint_receipt_payload
            if payload == "checkpoint"
            else evidence[0].checkpoint_receipt_payload
        ),
    )

    with pytest.raises(ValueError, match="hash|bytes"):
        build_study_lock_v3(
            sealed_evaluation_release_sha256=SEALED_RELEASE_SHA256,
            evidence=evidence,
            run_study_identities=(),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("run-id", "run ID|seed and arm"),
        ("world-size", "world-size-4|world"),
        ("config-sha", "config"),
        ("fingerprint", "fingerprint"),
        ("snapshot-row", "snapshot"),
        ("provenance", "manifest|differs"),
    ],
)
def test_builder_rejects_finalization_arm_and_provenance_drift(
    mutation,
    message,
):
    def mutate(seed: int, value: dict) -> None:
        if seed != 4:
            return
        if mutation == "run-id":
            value["arms"][0]["run_id"] = "memorysplit-v3-360m-s5-dense"
        elif mutation == "world-size":
            value["arms"][0]["world_size"] = 8
        elif mutation == "config-sha":
            value["arms"][0]["config_sha256"] = digest("forged-config")
        elif mutation == "fingerprint":
            value["arms"][0]["config_fingerprint"] = digest(
                "forged-fingerprint"
            )
        elif mutation == "snapshot-row":
            forged = digest("forged-snapshot")
            row = value["arms"][0]["snapshots"][0]["object"]
            row["sha256"] = forged
            row["uri"] = (
                f"{S3_ROOT}/"
                + snapshot_object_key(4, "dense", 1_358, forged)
            )
        else:
            value["run_manifest_sha256"] = digest("forged-manifest")

    bundle = build_collected_evidence(mutate_finalization=mutate)

    with pytest.raises(ValueError, match=message):
        build_study_lock_v3(
            sealed_evaluation_release_sha256=SEALED_RELEASE_SHA256,
            evidence=_evidence_rows(bundle),
            run_study_identities=(),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("legacy", "selected|provider-aware"),
        ("placement-object", "checkpoint"),
        ("cross-lifecycle", "lifecycle|SBOM|sbom"),
        ("per-arm", "fingerprint|arm"),
    ],
)
def test_builder_rejects_checkpoint_receipt_drift(mutation, message):
    def mutate(seed: int, value: dict) -> None:
        if seed != 2:
            return
        if mutation == "legacy":
            for field in sorted(
                set(LIFECYCLE_BINDING_FIELDS)
                - {
                    "boot_id",
                    "cohort_id",
                    "instance_id",
                    "profile_sha256",
                    "provider",
                    "seed",
                }
            ):
                del value[field]
        elif mutation == "placement-object":
            value["checkpoints"][0]["object"][
                "version_id"
            ] = "forged-checkpoint-version"
        elif mutation == "cross-lifecycle":
            value["runtime_sbom_sha256"] = digest("forged-sbom")
        else:
            value["checkpoints"][0]["config_fingerprint"] = digest(
                "forged-arm-fingerprint"
            )

    bundle = build_collected_evidence(mutate_checkpoint=mutate)

    with pytest.raises(ValueError, match=message):
        build_study_lock_v3(
            sealed_evaluation_release_sha256=SEALED_RELEASE_SHA256,
            evidence=_evidence_rows(bundle),
            run_study_identities=(),
        )


def test_builder_rejects_cross_seed_checkpoint_placement_drift():
    def mutate(seed: int, value: dict) -> None:
        if seed != 6:
            return
        value["region"] = "us-west-2"
        value["availability_zone"] = "us-west-2a"

    bundle = build_collected_evidence(mutate_checkpoint=mutate)
    evidence = _evidence_rows(bundle)
    identities = tuple(
        RunStudyIdentity(
            seed=seed,
            arm=arm,
            model_config_sha256=bundle.snapshots[0]["model_config_sha256"],
            model_identity="d360m",
            data_provenance_sha256=digest(f"unused:{seed}:{arm}"),
            config_fingerprint=digest(f"config:{seed}:{arm}"),
        )
        for seed in SEEDS
        for arm in ARMS
    )

    with pytest.raises(ValueError, match="uniform|cohort"):
        build_study_lock_v3(
            sealed_evaluation_release_sha256=SEALED_RELEASE_SHA256,
            evidence=evidence,
            run_study_identities=identities,
        )


def test_extraction_rejects_checkpoint_placement_drift_against_snapshots(
    tmp_path,
):
    def mutate(seed: int, value: dict) -> None:
        if seed != 0:
            return
        value["region"] = "us-west-2"
        value["availability_zone"] = "us-west-2a"

    bundle = build_collected_evidence(mutate_checkpoint=mutate)

    with pytest.raises(ValueError, match="metadata|snapshot"):
        extract_run_study_identities(
            _evidence_rows(bundle),
            snapshot_paths=_snapshot_paths(bundle, tmp_path),
        )


@pytest.mark.parametrize("mutation", ["body", "bytes", "missing"])
def test_extraction_rejects_snapshot_body_and_byte_drift(
    canonical_bundle,
    tmp_path,
    mutation,
):
    paths = _snapshot_paths(canonical_bundle, tmp_path)
    target = paths[(0, "dense", 1_358)]
    if mutation == "body":
        payload = bytearray(target.read_bytes())
        payload[-1] ^= 0x01
        target.write_bytes(bytes(payload))
    elif mutation == "bytes":
        target.write_bytes(target.read_bytes() + b"\x00")
    else:
        renamed = target.with_name("missing.pt")
        target.rename(renamed)

    with pytest.raises(ValueError, match="hash|byte|snapshot"):
        extract_run_study_identities(
            _evidence_rows(canonical_bundle),
            snapshot_paths=paths,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("step", "step"),
        ("identity-config", "config"),
        ("cross-step-identity", "invariant|cross-step"),
        ("operational-only", "study identity"),
        ("full-checkpoint", "full optimizer|checkpoint"),
        ("legacy", "legacy|metadata"),
    ],
)
def test_extraction_rejects_snapshot_content_drift(
    tmp_path,
    mutation,
    message,
):
    def mutate(seed: int, arm: str, step: int, state: dict) -> None:
        if (seed, arm, step) != (1, "dense", 3_396):
            return
        if mutation == "step":
            state["step"] = 1_358
        elif mutation == "identity-config":
            state["study_identity"]["config_sha256"] = digest(
                "forged-runtime-config"
            )
        elif mutation == "cross-step-identity":
            state["study_identity"]["model_identity"] = "other-model"
        elif mutation == "operational-only":
            del state["snapshot_version"]
            del state["study_identity"]
        elif mutation == "full-checkpoint":
            state["cfg"] = {}
            state["opt"] = {}
            state["rng_by_rank"] = []
        else:
            for field in sorted(
                set(state)
                - {
                    "model",
                    "model_cfg",
                    "data_provenance",
                    "step",
                    "world_size",
                }
            ):
                del state[field]

    bundle = build_collected_evidence(mutate_snapshot_state=mutate)

    with pytest.raises(ValueError, match=message):
        extract_run_study_identities(
            _evidence_rows(bundle),
            snapshot_paths=_snapshot_paths(bundle, tmp_path),
        )


@pytest.mark.parametrize("mutation", ["missing-slot", "extra-slot", "type"])
def test_extraction_requires_the_exact_100_canonical_snapshot_slots(
    canonical_bundle,
    tmp_path,
    mutation,
):
    paths = _snapshot_paths(canonical_bundle, tmp_path)
    if mutation == "missing-slot":
        del paths[(9, "split90", 13_582)]
    elif mutation == "extra-slot":
        paths[(9, "split90", 13_583)] = paths[(9, "split90", 13_582)]
    else:
        paths = [(key, value) for key, value in paths.items()]

    with pytest.raises((ValueError, TypeError), match="snapshot|slot|path"):
        extract_run_study_identities(
            _evidence_rows(canonical_bundle),
            snapshot_paths=paths,
        )


@pytest.mark.parametrize(
    "mutation",
    ["nineteen", "reordered", "foreign-fingerprint", "foreign-type"],
)
def test_build_rejects_unbound_or_misordered_run_identities(
    canonical_bundle,
    tmp_path,
    mutation,
):
    evidence = _evidence_rows(canonical_bundle)
    identities = list(
        extract_run_study_identities(
            evidence,
            snapshot_paths=_snapshot_paths(canonical_bundle, tmp_path),
        )
    )
    if mutation == "nineteen":
        identities.pop()
    elif mutation == "reordered":
        identities[0], identities[1] = identities[1], identities[0]
    elif mutation == "foreign-fingerprint":
        identities[0] = replace(
            identities[0],
            config_fingerprint=digest("forged-fingerprint"),
        )
    else:
        identities[0] = (
            identities[0].seed,
            identities[0].arm,
        )

    with pytest.raises(ValueError, match="identit|fingerprint|twenty"):
        build_study_lock_v3(
            sealed_evaluation_release_sha256=SEALED_RELEASE_SHA256,
            evidence=evidence,
            run_study_identities=identities,
        )


def test_built_lock_yields_exactly_100_receipt_proven_plans(
    canonical_bundle,
    canonical_lock,
):
    plans = plan_snapshot_evaluations(
        study_lock=canonical_lock,
        study_lock_sha256=canonical_sha256(canonical_lock.to_dict()),
        collection_receipts=_collection_evidence(canonical_bundle),
    )

    assert len(plans) == 100
    assert tuple(
        (plan.seed, plan.arm.value, plan.optimizer_step) for plan in plans
    ) == tuple(
        (seed, arm, step)
        for seed in SEEDS
        for arm in ARMS
        for step in SNAPSHOT_STEPS
    )
    assert len({plan.run_json_bytes for plan in plans}) == 100


def test_mutated_lock_fails_receipt_planning(
    canonical_bundle,
    canonical_lock,
):
    raw = canonical_lock.to_dict()
    raw["snapshots"][0]["s3_version_id"] = "forged-snapshot-version"
    mutated = StudyLockV3.from_dict(raw)

    with pytest.raises(ValueError, match="durably collected"):
        plan_snapshot_evaluations(
            study_lock=mutated,
            study_lock_sha256=canonical_sha256(mutated.to_dict()),
            collection_receipts=_collection_evidence(canonical_bundle),
        )


def test_publish_installs_exactly_one_lock_member(
    canonical_bundle,
    canonical_lock,
    tmp_path,
):
    root = _publish_root(tmp_path)
    lock_bytes = canonical_json_bytes(canonical_lock.to_dict())
    lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()

    published = publish_study_lock(
        canonical_lock,
        collection_receipts=_collection_evidence(canonical_bundle),
        output_root=root,
    )

    assert isinstance(published, PublishedStudyLock)
    assert published.authoritative_commitment == lock_sha256
    assert lock_sha256 == canonical_sha256(canonical_lock.to_dict())
    directory = root / f"{STUDY_LOCK_DIRECTORY_PREFIX}{lock_sha256}"
    assert published.output_dir == directory
    assert published.study_lock_path == directory / LOCK_MEMBER
    assert sorted(path.name for path in root.iterdir()) == [directory.name]
    assert [path.name for path in directory.iterdir()] == [LOCK_MEMBER]
    assert stat.S_IMODE(os.lstat(directory).st_mode) == 0o700
    assert stat.S_IMODE(os.lstat(directory / LOCK_MEMBER).st_mode) == 0o444

    observed = (directory / LOCK_MEMBER).read_bytes()
    assert observed == lock_bytes
    assert hashlib.sha256(observed).hexdigest() == (
        published.authoritative_commitment
    )
    assert StudyLockV3.from_dict(json.loads(observed)) == canonical_lock


def test_publish_rejects_collisions_without_reuse_or_replacement(
    canonical_bundle,
    canonical_lock,
    tmp_path,
):
    root = _publish_root(tmp_path)
    receipts = _collection_evidence(canonical_bundle)
    published = publish_study_lock(
        canonical_lock,
        collection_receipts=receipts,
        output_root=root,
    )
    before = published.study_lock_path.read_bytes()

    with pytest.raises(ValueError, match="already exists"):
        publish_study_lock(
            canonical_lock,
            collection_receipts=receipts,
            output_root=root,
        )

    assert published.study_lock_path.read_bytes() == before
    assert sorted(path.name for path in root.iterdir()) == [
        published.output_dir.name
    ]

    foreign_root = _publish_root(tmp_path, "foreign")
    conflict = (
        foreign_root
        / f"{STUDY_LOCK_DIRECTORY_PREFIX}"
        f"{canonical_sha256(canonical_lock.to_dict())}"
    )
    conflict.mkdir(mode=0o700)
    sentinel = conflict / "sentinel"
    sentinel.write_text("untouched", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        publish_study_lock(
            canonical_lock,
            collection_receipts=receipts,
            output_root=foreign_root,
        )

    assert sentinel.read_text(encoding="utf-8") == "untouched"


def test_publish_requires_receipt_proof_and_existing_output_root(
    canonical_bundle,
    canonical_lock,
    tmp_path,
):
    root = _publish_root(tmp_path)
    receipts = _collection_evidence(canonical_bundle)

    with pytest.raises(ValueError, match="ten|receipt|collection"):
        publish_study_lock(
            canonical_lock,
            collection_receipts=receipts[:9],
            output_root=root,
        )
    assert sorted(root.iterdir()) == []

    with pytest.raises(ValueError, match="output root|directory"):
        publish_study_lock(
            canonical_lock,
            collection_receipts=receipts,
            output_root=root / "missing",
        )


def test_failed_staging_is_quarantined_without_pathname_deletion(
    canonical_bundle,
    canonical_lock,
    tmp_path,
    monkeypatch,
):
    root = _publish_root(tmp_path)

    def forbid_pathname_deletion(*_args, **_kwargs):
        raise AssertionError("authority path must never unlink or rmdir")

    def fail_member_write(*_args, **_kwargs):
        raise OSError("injected staged lock write failure")

    monkeypatch.setattr(lock_builder, "_write_lock_member", fail_member_write)
    monkeypatch.setattr(lock_builder.os, "unlink", forbid_pathname_deletion)
    monkeypatch.setattr(lock_builder.os, "rmdir", forbid_pathname_deletion)

    with pytest.raises(OSError, match="injected staged lock write failure"):
        publish_study_lock(
            canonical_lock,
            collection_receipts=_collection_evidence(canonical_bundle),
            output_root=root,
        )

    quarantines = _quarantine_directories(root)
    assert len(quarantines) == 1
    assert sorted(quarantines[0].iterdir()) == []

    with pytest.raises(ValueError, match="quarantine|blocker"):
        publish_study_lock(
            canonical_lock,
            collection_receipts=_collection_evidence(canonical_bundle),
            output_root=root,
        )


def test_failed_install_verification_is_quarantined(
    canonical_bundle,
    canonical_lock,
    tmp_path,
    monkeypatch,
):
    root = _publish_root(tmp_path)

    def forbid_pathname_deletion(*_args, **_kwargs):
        raise AssertionError("authority path must never unlink or rmdir")

    def fail_install(*_args, **_kwargs):
        raise LockBuildError(
            "injected installed study lock verification failure"
        )

    monkeypatch.setattr(lock_builder, "_verify_installed_lock", fail_install)
    monkeypatch.setattr(lock_builder.os, "unlink", forbid_pathname_deletion)
    monkeypatch.setattr(lock_builder.os, "rmdir", forbid_pathname_deletion)

    with pytest.raises(
        ValueError,
        match="installed study lock verification",
    ):
        publish_study_lock(
            canonical_lock,
            collection_receipts=_collection_evidence(canonical_bundle),
            output_root=root,
        )

    quarantines = _quarantine_directories(root)
    assert len(quarantines) == 1
    assert [path.name for path in quarantines[0].iterdir()] == [LOCK_MEMBER]
    assert quarantines[0].joinpath(LOCK_MEMBER).read_bytes() == (
        canonical_json_bytes(canonical_lock.to_dict())
    )


def test_membership_mutation_after_install_is_quarantined(
    canonical_bundle,
    canonical_lock,
    tmp_path,
    monkeypatch,
):
    root = _publish_root(tmp_path)
    lock_sha256 = canonical_sha256(canonical_lock.to_dict())
    directory = root / f"{STUDY_LOCK_DIRECTORY_PREFIX}{lock_sha256}"

    def forbid_pathname_deletion(*_args, **_kwargs):
        raise AssertionError("authority path must never unlink or rmdir")

    def mutate_membership(event, **_context):
        if event != "publish_before_final_binding":
            return
        directory.joinpath("smuggled-extra-member").write_text(
            "attack",
            encoding="utf-8",
        )

    monkeypatch.setattr(lock_builder, "_run_mutation_hook", mutate_membership)
    monkeypatch.setattr(lock_builder.os, "unlink", forbid_pathname_deletion)
    monkeypatch.setattr(lock_builder.os, "rmdir", forbid_pathname_deletion)

    with pytest.raises(ValueError, match="entries|membership|changed"):
        publish_study_lock(
            canonical_lock,
            collection_receipts=_collection_evidence(canonical_bundle),
            output_root=root,
        )

    quarantines = _quarantine_directories(root)
    assert len(quarantines) == 1
    assert sorted(path.name for path in quarantines[0].iterdir()) == [
        "smuggled-extra-member",
        LOCK_MEMBER,
    ]


def test_post_install_replacement_is_detected_and_quarantined(
    canonical_bundle,
    canonical_lock,
    tmp_path,
    monkeypatch,
):
    root = _publish_root(tmp_path)
    lock_sha256 = canonical_sha256(canonical_lock.to_dict())
    directory = root / f"{STUDY_LOCK_DIRECTORY_PREFIX}{lock_sha256}"
    displaced = root / "displaced-installed-lock"
    sentinel = directory / "replacement-sentinel"

    def swap_directory(event, **_context):
        if event != "publish_before_final_binding":
            return
        directory.rename(displaced)
        directory.mkdir(mode=0o700)
        directory.chmod(0o700)
        sentinel.write_text("replacement", encoding="utf-8")

    monkeypatch.setattr(lock_builder, "_run_mutation_hook", swap_directory)

    with pytest.raises(
        ValueError,
        match="changed|replaced|identity|entries",
    ):
        publish_study_lock(
            canonical_lock,
            collection_receipts=_collection_evidence(canonical_bundle),
            output_root=root,
        )

    assert sentinel.read_text(encoding="utf-8") == "replacement"
    assert not displaced.exists()
    quarantines = _quarantine_directories(root)
    assert len(quarantines) == 1
    assert quarantines[0].joinpath(LOCK_MEMBER).read_bytes() == (
        canonical_json_bytes(canonical_lock.to_dict())
    )


def test_lock_builder_and_cli_import_boundary_keeps_torch_lazy():
    code = (
        "import sys\n"
        "import evals.confirmatory.lock_builder\n"
        "import scripts.build_confirmatory_study_lock\n"
        "forbidden = [\n"
        "    'torch',\n"
        "    'train.trainer',\n"
        "    'msctl.aws_collect',\n"
        "    'cluster.aws.p5.run_finalization',\n"
        "]\n"
        "loaded = [name for name in forbidden if name in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def _write_evidence_index(bundle, root: Path) -> Path:
    evidence_dir = (Path(root) / "evidence").resolve()
    snapshot_dir = evidence_dir / "snapshots"
    snapshot_dir.mkdir(parents=True)
    seeds = []
    for seed, (payload, uri, sha256, version_id) in enumerate(
        bundle.collection_receipts
    ):
        (evidence_dir / f"collection-{seed}.json").write_bytes(payload)
        (evidence_dir / f"finalization-{seed}.json").write_bytes(
            bundle.finalization_payloads[seed]
        )
        (evidence_dir / f"checkpoint-{seed}.json").write_bytes(
            bundle.checkpoint_receipt_payloads[seed]
        )
        rows = []
        for arm in ARMS:
            for step in SNAPSHOT_STEPS:
                name = f"snapshots/seed-{seed}-{arm}-step-{step}.pt"
                (evidence_dir / name).write_bytes(
                    bundle.snapshot_payloads[(seed, arm, step)]
                )
                rows.append({"arm": arm, "step": step, "path": name})
        seeds.append(
            {
                "seed": seed,
                "collection": {
                    "uri": uri,
                    "sha256": sha256,
                    "version_id": version_id,
                    "payload_path": f"collection-{seed}.json",
                },
                "finalization_payload_path": f"finalization-{seed}.json",
                "checkpoint_receipt_payload_path": (
                    f"checkpoint-{seed}.json"
                ),
                "snapshots": rows,
            }
        )
    index_path = evidence_dir / "evidence-index.json"
    index_path.write_text(
        json.dumps({"schema_version": 1, "seeds": seeds}, indent=2),
        encoding="utf-8",
    )
    return index_path


def _tree_state(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in sorted(Path(root).rglob("*"))
    }


def test_cli_dry_run_emits_one_canonical_json_and_writes_nothing(
    canonical_bundle,
    canonical_lock,
    tmp_path,
    capsys,
):
    index_path = _write_evidence_index(canonical_bundle, tmp_path)
    observed_root = index_path.parent
    before = _tree_state(observed_root)

    code = lock_cli.main(
        [
            "--evidence-index",
            str(index_path),
            "--sealed-evaluation-release-sha256",
            SEALED_RELEASE_SHA256,
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert len(captured.out.splitlines()) == 1
    result = json.loads(captured.out)
    assert captured.out == canonical_json_bytes(result).decode("utf-8")
    assert result["ok"] is True
    assert result["mode"] == "dry-run"
    assert result["published"] is False
    assert result["study_lock_sha256"] == canonical_sha256(
        canonical_lock.to_dict()
    )
    assert result["slot_count"] == 100
    assert result["seed_count"] == 10
    assert result["run_identity_count"] == 20
    assert _tree_state(observed_root) == before


def test_cli_publish_installs_the_lock_and_reports_its_commitment(
    canonical_bundle,
    canonical_lock,
    tmp_path,
    capsys,
):
    index_path = _write_evidence_index(canonical_bundle, tmp_path)
    root = _publish_root(tmp_path)

    code = lock_cli.main(
        [
            "--evidence-index",
            str(index_path),
            "--sealed-evaluation-release-sha256",
            SEALED_RELEASE_SHA256,
            "--publish",
            "--output-root",
            str(root),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert len(captured.out.splitlines()) == 1
    result = json.loads(captured.out)
    assert result["ok"] is True
    assert result["mode"] == "publish"
    assert result["published"] is True
    lock_sha256 = canonical_sha256(canonical_lock.to_dict())
    assert result["study_lock_sha256"] == lock_sha256
    assert result["path_authority"] == "informational_reopen_and_verify"
    directory = root / f"{STUDY_LOCK_DIRECTORY_PREFIX}{lock_sha256}"
    assert Path(result["output_dir"]) == directory
    assert Path(result["study_lock_path"]) == directory / LOCK_MEMBER
    assert directory.joinpath(LOCK_MEMBER).read_bytes() == (
        canonical_json_bytes(canonical_lock.to_dict())
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        [],
        ["--evidence-index", "missing.json"],
        ["--sealed-evaluation-release-sha256", "f" * 64],
        [
            "--evidence-index",
            "missing.json",
            "--sealed-evaluation-release-sha256",
            "f" * 64,
            "--publish",
        ],
        [
            "--evidence-index",
            "missing.json",
            "--sealed-evaluation-release-sha256",
            "f" * 64,
            "--output-root",
            "somewhere",
        ],
        [
            "--evidence-index",
            "missing.json",
            "--sealed-evaluation-release-sha256",
            "f" * 64,
            "--publish",
            "--dry-run",
            "--output-root",
            "somewhere",
        ],
        ["--unknown"],
    ],
)
def test_cli_help_and_errors_preserve_one_json_object_stdout(
    arguments,
    capsys,
):
    code = lock_cli.main(arguments)

    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    result = json.loads(captured.out)
    if arguments == ["--help"]:
        assert code == 0
        assert result["ok"] is True
        assert result["mode"] == "help"
        assert "usage" in result
    else:
        assert code != 0
        assert result["ok"] is False
        assert result["published"] is False
        assert result["error"]["code"]
        assert result["error"]["message"]


@pytest.mark.parametrize(
    "mutation",
    ["nine-seeds", "snapshot-order", "declared-sha", "schema", "fields"],
)
def test_cli_rejects_invalid_evidence_indexes_with_one_json_error(
    canonical_bundle,
    tmp_path,
    capsys,
    mutation,
):
    index_path = _write_evidence_index(canonical_bundle, tmp_path)
    value = json.loads(index_path.read_text(encoding="utf-8"))
    if mutation == "nine-seeds":
        value["seeds"].pop()
    elif mutation == "snapshot-order":
        rows = value["seeds"][0]["snapshots"]
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "declared-sha":
        value["seeds"][0]["collection"]["sha256"] = "0" * 64
    elif mutation == "schema":
        value["schema_version"] = 2
    else:
        value["extra"] = True
    index_path.write_text(json.dumps(value), encoding="utf-8")

    code = lock_cli.main(
        [
            "--evidence-index",
            str(index_path),
            "--sealed-evaluation-release-sha256",
            SEALED_RELEASE_SHA256,
        ]
    )

    captured = capsys.readouterr()
    assert code != 0
    assert len(captured.out.splitlines()) == 1
    result = json.loads(captured.out)
    assert result["ok"] is False
    assert result["error"]["code"]
    assert result["error"]["message"]
