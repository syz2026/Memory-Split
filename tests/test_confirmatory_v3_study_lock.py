from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib

import pytest

from evals.confirmatory.study_lock import (
    EXPECTED_STUDY_SLOTS_V3,
    FROZEN_PREREGISTRATION_SHA256_V3,
    STUDY_LOCK_SCHEMA_V3,
    ProviderSelectionBinding,
    StudyLockV3,
)
from msctl.aws_hardware import (
    AWS_HARDWARE_AMENDMENT_SHA256,
    PROVIDER_SELECTION_S3_KEY,
    AuthenticatedSelectionBinding,
)
from msctl.aws_contracts import (
    ARMS,
    SEEDS,
    SNAPSHOT_STEPS,
    checkpoint_receipt_key,
    snapshot_object_key,
)


def _digest(seed: int, arm: str, step: int) -> str:
    return hashlib.sha256(f"{seed}:{arm}:{step}".encode("ascii")).hexdigest()


def _provider_selection(**changes) -> dict:
    value = {
        "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
        "provider_selection_s3_key": PROVIDER_SELECTION_S3_KEY,
        "provider_selection_sha256": "a" * 64,
        "provider_selection_s3_version_id": "selection-version-p5",
        "hardware_amendment_sha256": AWS_HARDWARE_AMENDMENT_SHA256,
        "selected_provider": "aws-p5.48xlarge",
        "profile_id": "aws-p5.48xlarge-v3",
        "profile_sha256": (
            "2207bfbad5e8fa9fc804770b582d0b21f8b6ed109b2e3f3b5c0474c732c53543"
        ),
        "runtime_lock_sha256": "b" * 64,
        "qualification_evidence_sha256": "c" * 64,
        "environment_receipt_sha256": "d" * 64,
        "canary_receipt_sha256": "e" * 64,
        "approval_receipt_sha256": "f" * 64,
        "approval_public_key_sha256": "1" * 64,
    }
    value.update(changes)
    return value


def _snapshots() -> list[dict]:
    snapshots = []
    for seed in SEEDS:
        for arm in ARMS:
            for step in SNAPSHOT_STEPS:
                digest = _digest(seed, arm, step)
                receipt_digest = hashlib.sha256(
                    f"receipt:{seed}:{step}".encode("ascii")
                ).hexdigest()
                snapshots.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "optimizer_step": step,
                        "checkpoint_sha256": digest,
                        "s3_object_key": snapshot_object_key(
                            seed,
                            arm,
                            step,
                            digest,
                        ),
                        "s3_version_id": f"version-{seed}-{arm}-{step}",
                        "checkpoint_receipt_sha256": receipt_digest,
                        "checkpoint_receipt_s3_object_key": (
                            checkpoint_receipt_key(seed, receipt_digest)
                        ),
                        "checkpoint_receipt_s3_version_id": (
                            f"receipt-version-{seed}-{step}"
                        ),
                        "provider_selection_sha256": "a" * 64,
                        "provider_selection_s3_version_id": (
                            "selection-version-p5"
                        ),
                        "snapshot_version": 2,
                        "training_run_id": (
                            f"memorysplit-v3-360m-s{seed}-{arm}"
                        ),
                        "config_fingerprint": hashlib.sha256(
                            f"config:{seed}:{arm}".encode("ascii")
                        ).hexdigest(),
                        "training_config_sha256": hashlib.sha256(
                            f"config-bytes:{seed}:{arm}".encode("ascii")
                        ).hexdigest(),
                        "model_config_sha256": hashlib.sha256(
                            b"model-config"
                        ).hexdigest(),
                        "model_identity": "d360m",
                        "data_provenance_sha256": hashlib.sha256(
                            f"data:{seed}:{arm}".encode("ascii")
                        ).hexdigest(),
                        "data_receipt_sha256": hashlib.sha256(
                            b"data-receipt"
                        ).hexdigest(),
                        "data_build_id": hashlib.sha256(
                            b"data-build"
                        ).hexdigest(),
                        "ordered_stream_sha256": hashlib.sha256(
                            b"ordered-stream"
                        ).hexdigest(),
                        "world_size": 4,
                        "tokens_per_step": 524_288,
                    }
                )
    return snapshots


def _lock(**changes) -> dict:
    value = {
        "record_type": STUDY_LOCK_SCHEMA_V3,
        "schema_version": 3,
        "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256_V3,
        "sealed_evaluation_release_sha256": "f" * 64,
        "provider_selection": _provider_selection(),
        "snapshots": _snapshots(),
    }
    value.update(changes)
    return value


def test_v3_study_lock_binds_the_exact_ordered_100_snapshot_cohort():
    lock = StudyLockV3.from_dict(_lock())

    assert lock.to_dict() == _lock()
    assert FROZEN_PREREGISTRATION_SHA256_V3 == (
        "6b2b5da3e3dc3d533498a0aa9d1891f356134ce045b1553b74a0161d94cb81d7"
    )
    assert len(lock.snapshots) == len(EXPECTED_STUDY_SLOTS_V3) == 100
    assert tuple(
        (snapshot.seed, snapshot.arm, snapshot.optimizer_step)
        for snapshot in lock.snapshots
    ) == EXPECTED_STUDY_SLOTS_V3
    assert not any(snapshot.arm == "random" for snapshot in lock.snapshots)
    assert isinstance(lock.provider_selection, ProviderSelectionBinding)
    assert {
        snapshot.provider_selection_sha256 for snapshot in lock.snapshots
    } == {lock.provider_selection.provider_selection_sha256}
    assert {
        snapshot.provider_selection_s3_version_id for snapshot in lock.snapshots
    } == {lock.provider_selection.provider_selection_s3_version_id}

    first = lock.snapshots[0]
    assert first.s3_object_key == snapshot_object_key(
        first.seed,
        first.arm,
        first.optimizer_step,
        first.checkpoint_sha256,
    )
    assert first.s3_version_id
    assert first.checkpoint_receipt_s3_object_key == checkpoint_receipt_key(
        first.seed,
        first.checkpoint_receipt_sha256,
    )
    assert first.checkpoint_receipt_s3_version_id
    assert first.snapshot_version == 2
    assert first.training_run_id == "memorysplit-v3-360m-s0-dense"
    assert first.world_size == 4
    assert first.tokens_per_step == 524_288

    with pytest.raises(FrozenInstanceError):
        first.checkpoint_sha256 = "0" * 64


@pytest.mark.parametrize("mutation", ["missing", "reordered", "aliased"])
def test_v3_study_lock_is_exact_and_slots_cannot_be_replaced(mutation):
    snapshots = _snapshots()
    if mutation == "missing":
        snapshots.pop()
    elif mutation == "reordered":
        snapshots[0], snapshots[1] = snapshots[1], snapshots[0]
    else:
        assert snapshots[0]["seed"] == snapshots[1]["seed"]
        assert snapshots[0]["arm"] == snapshots[1]["arm"]
        snapshots[1]["checkpoint_sha256"] = snapshots[0][
            "checkpoint_sha256"
        ]
        snapshots[1]["s3_object_key"] = snapshot_object_key(
            snapshots[1]["seed"],
            snapshots[1]["arm"],
            snapshots[1]["optimizer_step"],
            snapshots[0]["checkpoint_sha256"],
        )
        snapshots[1]["s3_version_id"] = snapshots[0]["s3_version_id"]

    with pytest.raises(ValueError, match="exact|ordered|100|slot|alias|reuse"):
        StudyLockV3.from_dict(_lock(snapshots=snapshots))


def test_v3_study_lock_rejects_random_arms_wrong_keys_and_null_versions():
    random_arm = _snapshots()
    random_arm[0]["arm"] = "random"
    with pytest.raises(ValueError, match="arm"):
        StudyLockV3.from_dict(_lock(snapshots=random_arm))

    wrong_key = _snapshots()
    wrong_key[0]["s3_object_key"] = (
        f"checkpoints/{wrong_key[0]['checkpoint_sha256']}.pt"
    )
    with pytest.raises(ValueError, match="content-addressed|object key"):
        StudyLockV3.from_dict(_lock(snapshots=wrong_key))

    for version_id in (None, ""):
        null_version = _snapshots()
        null_version[0]["s3_version_id"] = version_id
        with pytest.raises(ValueError, match="version"):
            StudyLockV3.from_dict(_lock(snapshots=null_version))


@pytest.mark.parametrize(
    "field",
    ["s3_version_id", "checkpoint_receipt_s3_version_id"],
)
@pytest.mark.parametrize(
    "version_id",
    [
        None,
        "",
        "null",
        "has space",
        "has\ttab",
        "has\nnewline",
        "control-\x7f",
        "x" * 1_025,
    ],
)
def test_v3_study_lock_rejects_non_real_version_id_candidates(
    field,
    version_id,
):
    snapshots = _snapshots()
    snapshots[0][field] = version_id

    with pytest.raises(ValueError, match="version"):
        StudyLockV3.from_dict(_lock(snapshots=snapshots))


def test_v3_study_lock_accepts_bounded_printable_opaque_version_ids():
    snapshots = _snapshots()
    snapshots[0]["s3_version_id"] = "opaque-A+/=._~:1"
    snapshots[0][
        "checkpoint_receipt_s3_version_id"
    ] = "opaque-receipt-A+/=._~:1"
    snapshots[len(SNAPSHOT_STEPS)][
        "checkpoint_receipt_s3_version_id"
    ] = "opaque-receipt-A+/=._~:1"

    lock = StudyLockV3.from_dict(_lock(snapshots=snapshots))

    assert lock.snapshots[0].s3_version_id == "opaque-A+/=._~:1"
    assert (
        lock.snapshots[0].checkpoint_receipt_s3_version_id
        == "opaque-receipt-A+/=._~:1"
    )


def test_v3_study_lock_rejects_wrong_checkpoint_receipt_reference():
    snapshots = _snapshots()
    snapshots[0]["checkpoint_receipt_s3_object_key"] = (
        "receipts/checkpoints/wrong.json"
    )

    with pytest.raises(ValueError, match="receipt.*key"):
        StudyLockV3.from_dict(_lock(snapshots=snapshots))


def test_v3_study_lock_rejects_receipt_content_alias_with_new_version():
    snapshots = _snapshots()
    source_dense = 0
    source_split90 = len(SNAPSHOT_STEPS)
    target_dense = 1
    target_split90 = len(SNAPSHOT_STEPS) + 1
    assert (
        snapshots[source_dense]["checkpoint_receipt_sha256"]
        == snapshots[source_split90]["checkpoint_receipt_sha256"]
    )

    for target in (target_dense, target_split90):
        snapshots[target]["checkpoint_receipt_sha256"] = snapshots[
            source_dense
        ]["checkpoint_receipt_sha256"]
        snapshots[target]["checkpoint_receipt_s3_object_key"] = snapshots[
            source_dense
        ]["checkpoint_receipt_s3_object_key"]
        snapshots[target][
            "checkpoint_receipt_s3_version_id"
        ] = "different-version-cannot-disguise-content-alias"

    with pytest.raises(ValueError, match="receipt.*(reuse|alias|content)"):
        StudyLockV3.from_dict(_lock(snapshots=snapshots))


def test_v3_study_lock_binds_frozen_preregistration_and_sealed_release_hashes():
    with pytest.raises(ValueError, match="preregistration"):
        StudyLockV3.from_dict(
            _lock(preregistration_sha256="0" * 64)
        )
    with pytest.raises(ValueError, match="sealed.*release"):
        StudyLockV3.from_dict(
            _lock(sealed_evaluation_release_sha256=None)
        )


@pytest.mark.parametrize(
    "field",
    [
        "training_run_id",
        "config_fingerprint",
        "training_config_sha256",
        "model_config_sha256",
        "model_identity",
        "data_provenance_sha256",
        "data_receipt_sha256",
        "data_build_id",
        "ordered_stream_sha256",
    ],
)
def test_v3_study_lock_rejects_cross_step_provenance_drift(field):
    snapshots = _snapshots()
    assert snapshots[0]["seed"] == snapshots[1]["seed"] == 0
    assert snapshots[0]["arm"] == snapshots[1]["arm"] == "dense"
    snapshots[1][field] = (
        "other-model"
        if field in {"training_run_id", "model_identity"}
        else "0" * 64
    )

    with pytest.raises(
        ValueError,
        match="cross.step|provenance|invariant|training run identity",
    ):
        StudyLockV3.from_dict(_lock(snapshots=snapshots))


@pytest.mark.parametrize(
    "field",
    [
        "model_config_sha256",
        "model_identity",
        "data_receipt_sha256",
        "data_build_id",
        "ordered_stream_sha256",
    ],
)
def test_v3_study_lock_rejects_cross_arm_matched_invariant_drift(field):
    snapshots = _snapshots()
    split90_index = len(SNAPSHOT_STEPS)
    assert snapshots[0]["seed"] == snapshots[split90_index]["seed"] == 0
    assert snapshots[0]["arm"] == "dense"
    assert snapshots[split90_index]["arm"] == "split90"
    snapshots[split90_index][field] = (
        "other-model" if field == "model_identity" else "0" * 64
    )

    with pytest.raises(ValueError, match="cross.arm|matched|invariant"):
        StudyLockV3.from_dict(_lock(snapshots=snapshots))


def test_v3_study_lock_requires_one_selection_for_every_snapshot():
    snapshots = _snapshots()
    snapshots[-1]["provider_selection_sha256"] = "d" * 64

    with pytest.raises(ValueError, match="cohort|selection"):
        StudyLockV3.from_dict(_lock(snapshots=snapshots))

    snapshots = _snapshots()
    snapshots[-1][
        "provider_selection_s3_version_id"
    ] = "cross-profile-version"
    with pytest.raises(ValueError, match="cohort|selection|version"):
        StudyLockV3.from_dict(_lock(snapshots=snapshots))


@pytest.mark.parametrize(
    "changes",
    [
        {"hardware_amendment_sha256": "0" * 64},
        {"selected_provider": "aws-p6-b300.48xlarge"},
        {"profile_id": "aws-p6-b300.48xlarge-v3"},
        {
            "profile_sha256": (
                "6884cd30670214bcecaa105d32b2b5518b1533d9fdbad6ac15327f9a8b7fefa4"
            )
        },
        {"runtime_lock_sha256": None},
        {"qualification_evidence_sha256": None},
        {"environment_receipt_sha256": None},
    ],
)
def test_v3_study_lock_rejects_cross_profile_or_unbound_evidence(changes):
    with pytest.raises(
        ValueError,
        match="amendment|profile|provider|runtime|evidence|environment",
    ):
        StudyLockV3.from_dict(
            _lock(provider_selection=_provider_selection(**changes))
        )


def test_provider_selection_binding_copies_authenticated_admission_fields():
    authenticated = AuthenticatedSelectionBinding(
        cohort_id="memorysplit-confirmatory-v3-360m-n10-aws",
        amendment_sha256=AWS_HARDWARE_AMENDMENT_SHA256,
        selection_sha256="a" * 64,
        selection_version_id="selection-version-p5",
        profile_id="aws-p5.48xlarge-v3",
        provider="aws-p5.48xlarge",
        profile_sha256=(
            "2207bfbad5e8fa9fc804770b582d0b21f8b6ed109b2e3f3b5c0474c732c53543"
        ),
        runtime_lock_sha256="b" * 64,
        qualification_evidence_sha256="c" * 64,
        environment_receipt_sha256="d" * 64,
        canary_receipt_sha256="e" * 64,
        approval_receipt_sha256="f" * 64,
        approval_public_key_sha256="1" * 64,
        account_id="123456789012",
        instance_id="i-0123456789abcdef0",
        boot_id="01234567-89ab-4def-8123-456789abcdef",
        region="us-east-1",
        availability_zone="us-east-1a",
        purchase_model="on_demand",
        seed=0,
        arm="dense",
    )

    binding = ProviderSelectionBinding.from_authenticated(authenticated)

    assert binding.to_dict() == _provider_selection()
