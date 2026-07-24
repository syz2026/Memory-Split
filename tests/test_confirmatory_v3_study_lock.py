from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib

import pytest

from evals.confirmatory.study_lock import (
    EXPECTED_STUDY_SLOTS_V3,
    FROZEN_PREREGISTRATION_SHA256_V3,
    STUDY_LOCK_SCHEMA_V3,
    StudyLockV3,
)
from msctl.aws_contracts import (
    ARMS,
    SEEDS,
    SNAPSHOT_STEPS,
    checkpoint_object_key,
)


def _digest(seed: int, arm: str, step: int) -> str:
    return hashlib.sha256(f"{seed}:{arm}:{step}".encode("ascii")).hexdigest()


def _snapshots() -> list[dict]:
    snapshots = []
    for seed in SEEDS:
        for arm in ARMS:
            for step in SNAPSHOT_STEPS:
                digest = _digest(seed, arm, step)
                snapshots.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "optimizer_step": step,
                        "checkpoint_sha256": digest,
                        "s3_object_key": checkpoint_object_key(
                            seed,
                            arm,
                            digest,
                        ),
                        "s3_version_id": f"version-{seed}-{arm}-{step}",
                    }
                )
    return snapshots


def _lock(**changes) -> dict:
    value = {
        "record_type": STUDY_LOCK_SCHEMA_V3,
        "schema_version": 3,
        "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256_V3,
        "sealed_evaluation_release_sha256": "f" * 64,
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

    first = lock.snapshots[0]
    assert first.s3_object_key == checkpoint_object_key(
        first.seed,
        first.arm,
        first.checkpoint_sha256,
    )
    assert first.s3_version_id

    with pytest.raises(FrozenInstanceError):
        first.checkpoint_sha256 = "0" * 64


@pytest.mark.parametrize("mutation", ["missing", "reordered", "replaced"])
def test_v3_study_lock_is_exact_and_slots_cannot_be_replaced(mutation):
    snapshots = _snapshots()
    if mutation == "missing":
        snapshots.pop()
    elif mutation == "reordered":
        snapshots[0], snapshots[1] = snapshots[1], snapshots[0]
    else:
        replacement = dict(snapshots[0])
        replacement["checkpoint_sha256"] = "0" * 64
        replacement["s3_object_key"] = checkpoint_object_key(
            replacement["seed"],
            replacement["arm"],
            replacement["checkpoint_sha256"],
        )
        snapshots[-1] = replacement

    with pytest.raises(ValueError, match="exact|ordered|100|slot"):
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


def test_v3_study_lock_binds_frozen_preregistration_and_sealed_release_hashes():
    with pytest.raises(ValueError, match="preregistration"):
        StudyLockV3.from_dict(
            _lock(preregistration_sha256="0" * 64)
        )
    with pytest.raises(ValueError, match="sealed.*release"):
        StudyLockV3.from_dict(
            _lock(sealed_evaluation_release_sha256=None)
        )
