from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib

import pytest

from evals.confirmatory.study_lock import (
    EXPECTED_STUDY_SLOTS_V3,
    FROZEN_PREREGISTRATION_SHA256_V3,
    STUDY_LOCK_SCHEMA_V3,
    ProviderSelectionBinding,
    SeedLifecycleBinding,
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
from msctl.aws_lifecycle import ProviderLifecycleBinding
from tests.study_lock_fixtures import (
    build_seed_lifecycles,
    expected_lifecycle_binding,
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
                    f"receipt:{seed}".encode("ascii")
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
                            f"receipt-version-{seed}"
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


def _seed_lifecycles(
    selection: dict | None = None,
    **kwargs,
) -> list[dict]:
    lifecycles, _receipts = build_seed_lifecycles(
        snapshots=_snapshots(),
        provider_selection=(
            _provider_selection() if selection is None else selection
        ),
        **kwargs,
    )
    return lifecycles


def _lock(**changes) -> dict:
    snapshots = changes.pop("snapshots", None)
    selection = changes.pop("provider_selection", None)
    if snapshots is None:
        snapshots = _snapshots()
    if selection is None:
        selection = _provider_selection()
    value = {
        "record_type": STUDY_LOCK_SCHEMA_V3,
        "schema_version": 3,
        "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256_V3,
        "sealed_evaluation_release_sha256": "f" * 64,
        "provider_selection": selection,
        "snapshots": snapshots,
        "seed_lifecycles": _seed_lifecycles(selection=selection),
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
    for slot in snapshots:
        if slot["seed"] == 0:
            slot[
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


def test_v3_study_lock_requires_one_collected_receipt_per_seed():
    diverging = _snapshots()
    diverging[1]["checkpoint_receipt_s3_version_id"] = (
        "different-version-cannot-split-the-seed-receipt"
    )
    with pytest.raises(
        ValueError,
        match="disagree.*per-seed checkpoint receipt",
    ):
        StudyLockV3.from_dict(_lock(snapshots=diverging))

    cross_seed = _snapshots()
    per_seed = 2 * len(SNAPSHOT_STEPS)
    for slot in cross_seed[per_seed : 2 * per_seed]:
        assert slot["seed"] == 1
        slot["checkpoint_receipt_sha256"] = cross_seed[0][
            "checkpoint_receipt_sha256"
        ]
        slot["checkpoint_receipt_s3_object_key"] = checkpoint_receipt_key(
            1,
            cross_seed[0]["checkpoint_receipt_sha256"],
        )
        slot[
            "checkpoint_receipt_s3_version_id"
        ] = "different-version-cannot-disguise-content-alias"
    with pytest.raises(ValueError, match="receipt.*(reuse|alias|content)"):
        StudyLockV3.from_dict(_lock(snapshots=cross_seed))


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


def test_v3_study_lock_binds_ten_seed_lifecycles_and_reconstructs_authority():
    lock = StudyLockV3.from_dict(_lock())

    assert lock.to_dict() == _lock()
    assert len(lock.seed_lifecycles) == 10
    assert all(
        isinstance(lifecycle, SeedLifecycleBinding)
        for lifecycle in lock.seed_lifecycles
    )
    assert tuple(
        lifecycle.seed for lifecycle in lock.seed_lifecycles
    ) == tuple(range(10))
    for seed in SEEDS:
        binding = lock.lifecycle_binding(seed)
        assert isinstance(binding, ProviderLifecycleBinding)
        assert binding == expected_lifecycle_binding(
            _provider_selection(),
            seed,
        )
    with pytest.raises(ValueError, match="seed"):
        lock.lifecycle_binding(10)
    with pytest.raises(FrozenInstanceError):
        lock.seed_lifecycles[0].boot_id = "0" * 32


def test_v3_study_lock_fails_closed_for_pre_bridge_lock_shapes():
    pre_bridge = _lock()
    del pre_bridge["seed_lifecycles"]

    with pytest.raises(ValueError, match="exact"):
        StudyLockV3.from_dict(pre_bridge)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "reordered", "foreign-seed"],
)
def test_v3_study_lock_requires_exactly_ten_ascending_seed_lifecycles(
    mutation,
):
    lifecycles = _seed_lifecycles()
    if mutation == "missing":
        lifecycles.pop()
    elif mutation == "duplicate":
        lifecycles[1] = dict(lifecycles[0])
    elif mutation == "reordered":
        lifecycles[0], lifecycles[1] = lifecycles[1], lifecycles[0]
    else:
        lifecycles[9] = dict(lifecycles[9], seed=0)

    with pytest.raises(ValueError, match="seed|ten|ascending|exact"):
        StudyLockV3.from_dict(_lock(seed_lifecycles=lifecycles))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("account_id", "210987654321"),
        ("availability_zone", "us-east-1a"),
        ("instance_id", "i-0fedcba9876543210"),
        ("region", "us-west-2"),
        ("purchase_model", "spot"),
        ("runtime_sbom_sha256", "8" * 64),
        ("objective_controls_contract_sha256", "7" * 64),
        ("source_commit", "e" * 40),
        ("source_tree", "1" * 40),
    ],
)
def test_v3_study_lock_rejects_cohort_uniform_lifecycle_drift(field, value):
    lifecycles = _seed_lifecycles()
    lifecycles[3] = dict(lifecycles[3], **{field: value})

    with pytest.raises(ValueError, match="uniform|cohort|constant"):
        StudyLockV3.from_dict(_lock(seed_lifecycles=lifecycles))


def test_v3_study_lock_accepts_boot_only_lifecycle_drift():
    reboot = "87654321-4321-4cba-8fed-1234567890ab"
    lifecycles = _seed_lifecycles(
        boot_ids={4: reboot},
    )

    lock = StudyLockV3.from_dict(_lock(seed_lifecycles=lifecycles))

    assert lock.seed_lifecycles[4].boot_id == reboot
    assert lock.lifecycle_binding(4).boot_id == reboot
    assert lock.lifecycle_binding(3).boot_id != reboot


def test_v3_study_lock_allows_arm_scoped_operational_config_hashes():
    lifecycles = _seed_lifecycles()
    assert all(
        lifecycle["dense_operational_config_sha256"]
        != lifecycle["split90_operational_config_sha256"]
        for lifecycle in lifecycles
    )
    for seed, lifecycle in enumerate(lifecycles):
        lifecycle["dense_operational_config_sha256"] = hashlib.sha256(
            f"dense-config:{seed}".encode("ascii")
        ).hexdigest()
        lifecycle["split90_operational_config_sha256"] = hashlib.sha256(
            f"split90-config:{seed}".encode("ascii")
        ).hexdigest()

    lock = StudyLockV3.from_dict(_lock(seed_lifecycles=lifecycles))

    assert (
        lock.seed_lifecycles[0].dense_operational_config_sha256
        != lock.seed_lifecycles[0].split90_operational_config_sha256
    )
    assert (
        lock.seed_lifecycles[0].dense_operational_config_sha256
        != lock.seed_lifecycles[1].dense_operational_config_sha256
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "finalization_receipt_s3_uri",
            "s3://memorysplit-prod/receipts/other.json",
            "finalization.*(key|URI)",
        ),
        (
            "collection_receipt_s3_uri",
            "s3://memorysplit-prod/receipts/other.json",
            "collection.*(key|URI)",
        ),
        ("finalization_receipt_sha256", "0" * 64, "finalization"),
        ("collection_receipt_sha256", "0" * 64, "collection"),
        ("finalization_receipt_s3_version_id", "null", "version"),
        ("collection_receipt_s3_version_id", "", "version"),
        ("finalization_receipt_bytes", 0, "bytes"),
        ("finalization_receipt_bytes", -3, "bytes"),
        ("boot_id", "not-a-boot-id", "lifecycle|boot"),
        ("run_manifest_sha256", "not-a-hash", "manifest"),
        ("source_commit", "z" * 40, "commit"),
    ],
)
def test_v3_seed_lifecycle_rejects_invalid_receipt_and_identity_fields(
    field,
    value,
    message,
):
    lifecycles = _seed_lifecycles()
    lifecycles[0] = dict(lifecycles[0], **{field: value})

    with pytest.raises(ValueError, match=message):
        StudyLockV3.from_dict(_lock(seed_lifecycles=lifecycles))


def test_v3_seed_lifecycle_rejects_receipt_identity_aliases_across_seeds():
    lifecycles = _seed_lifecycles()
    lifecycles[2] = dict(
        lifecycles[2],
        collection_receipt_sha256=lifecycles[1][
            "collection_receipt_sha256"
        ],
        collection_receipt_s3_uri=(
            "s3://memorysplit-prod/confirmatory-v3/"
            "receipts/collections/seed-2/sha256/"
            + lifecycles[1]["collection_receipt_sha256"]
            + ".json"
        ),
    )

    with pytest.raises(ValueError, match="alias|reuse|unique"):
        StudyLockV3.from_dict(_lock(seed_lifecycles=lifecycles))


def test_v3_seed_lifecycle_rejects_crossed_object_roots():
    lifecycles = _seed_lifecycles()
    foreign_root = "s3://memorysplit-other/confirmatory-v3"
    lifecycles[5] = dict(
        lifecycles[5],
        collection_receipt_s3_uri=(
            f"{foreign_root}/receipts/collections/seed-5/sha256/"
            + lifecycles[5]["collection_receipt_sha256"]
            + ".json"
        ),
    )

    with pytest.raises(ValueError, match="root"):
        StudyLockV3.from_dict(_lock(seed_lifecycles=lifecycles))


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
