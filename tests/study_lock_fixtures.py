"""Shared v3 study-lock seed-lifecycle and collection-receipt fixtures."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from msctl.aws_contracts import (
    ARMS,
    SEEDS,
    SNAPSHOT_STEPS,
    checkpoint_object_key,
    checkpoint_receipt_key,
    collection_receipt_key,
    log_object_key,
    run_receipt_key,
    snapshot_object_key,
)
from msctl.aws_hardware import (
    AWS_HARDWARE_AMENDMENT_SHA256,
    PROVIDER_SELECTION_S3_KEY,
)
from msctl.aws_lifecycle import (
    ProviderLifecycleBinding,
    lifecycle_operational_metadata,
)


S3_ROOT = "s3://memorysplit-prod/confirmatory-v3"
ACCOUNT_ID = "123456789012"
INSTANCE_ID = "i-0123456789abcdef0"
BOOT_ID = "12345678-1234-4abc-8def-1234567890ab"
REGION = "us-east-1"
AVAILABILITY_ZONE = "us-east-1d"
PURCHASE_MODEL = "on_demand"
RUNTIME_SBOM_SHA256 = "9" * 64
OBJECTIVE_CONTROLS_SHA256 = "a" * 64
SOURCE_COMMIT = "f" * 40
SOURCE_TREE = "0" * 40
P5_PROFILE_SHA256 = (
    "2207bfbad5e8fa9fc804770b582d0b21f8b6ed109b2e3f3b5c0474c732c53543"
)
STUDY_COHORT_ID = "memorysplit-confirmatory-v3-360m-n10-aws"
MODEL_IDENTITY = "d360m"
MODEL_CFG = {
    "ctx": 16,
    "d_model": 8,
    "n_head": 2,
    "n_layer": 1,
    "vocab_size": 64,
}
TERMINAL_STEP = 13_582
TOKENS_PER_STEP = 524_288


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


RELEASE_SHA256 = digest("training-release")
RELEASE_RECEIPT_SHA256 = digest("training-release-receipt")
DENSE_OPERATIONAL_CONFIG_SHA256 = digest("dense-operational-config")
SPLIT90_OPERATIONAL_CONFIG_SHA256 = digest("split90-operational-config")
DATA_RECEIPT_SHA256 = digest("data-receipt")
DATA_BUILD_ID = digest("data-build")
ORDERED_STREAM_SHA256 = digest("ordered-stream")


def default_provider_selection(**changes) -> dict:
    value = {
        "cohort_id": STUDY_COHORT_ID,
        "provider_selection_s3_key": PROVIDER_SELECTION_S3_KEY,
        "provider_selection_sha256": "a" * 64,
        "provider_selection_s3_version_id": "provider-selection-version-p5",
        "hardware_amendment_sha256": AWS_HARDWARE_AMENDMENT_SHA256,
        "selected_provider": "aws-p5.48xlarge",
        "profile_id": "aws-p5.48xlarge-v3",
        "profile_sha256": P5_PROFILE_SHA256,
        "runtime_lock_sha256": "b" * 64,
        "qualification_evidence_sha256": "c" * 64,
        "environment_receipt_sha256": "d" * 64,
        "canary_receipt_sha256": "e" * 64,
        "approval_receipt_sha256": "f" * 64,
        "approval_public_key_sha256": "1" * 64,
    }
    value.update(changes)
    return value


def canonical_receipt_bytes(value: object) -> bytes:
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


def lifecycle_core(**overrides) -> dict:
    value = {
        "account_id": ACCOUNT_ID,
        "availability_zone": AVAILABILITY_ZONE,
        "boot_id": BOOT_ID,
        "instance_id": INSTANCE_ID,
        "objective_controls_contract_sha256": OBJECTIVE_CONTROLS_SHA256,
        "purchase_model": PURCHASE_MODEL,
        "region": REGION,
        "runtime_sbom_sha256": RUNTIME_SBOM_SHA256,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
    }
    value.update(overrides)
    return value


def expected_lifecycle_binding(
    provider_selection: dict,
    seed: int,
    *,
    core: dict | None = None,
    boot_id: str | None = None,
) -> ProviderLifecycleBinding:
    values = lifecycle_core() if core is None else core
    return ProviderLifecycleBinding(
        cohort_id=provider_selection["cohort_id"],
        provider=provider_selection["selected_provider"],
        profile_id=provider_selection["profile_id"],
        profile_sha256=provider_selection["profile_sha256"],
        hardware_amendment_sha256=provider_selection[
            "hardware_amendment_sha256"
        ],
        provider_selection_sha256=provider_selection[
            "provider_selection_sha256"
        ],
        provider_selection_version_id=provider_selection[
            "provider_selection_s3_version_id"
        ],
        runtime_lock_sha256=provider_selection["runtime_lock_sha256"],
        runtime_sbom_sha256=values["runtime_sbom_sha256"],
        qualification_evidence_sha256=provider_selection[
            "qualification_evidence_sha256"
        ],
        qualification_environment_receipt_sha256=provider_selection[
            "environment_receipt_sha256"
        ],
        qualification_canary_receipt_sha256=provider_selection[
            "canary_receipt_sha256"
        ],
        qualification_approval_receipt_sha256=provider_selection[
            "approval_receipt_sha256"
        ],
        qualification_approval_public_key_sha256=provider_selection[
            "approval_public_key_sha256"
        ],
        objective_controls_contract_sha256=values[
            "objective_controls_contract_sha256"
        ],
        account_id=values["account_id"],
        instance_id=values["instance_id"],
        boot_id=values["boot_id"] if boot_id is None else boot_id,
        region=values["region"],
        availability_zone=values["availability_zone"],
        purchase_model=values["purchase_model"],
        seed=seed,
        arms=("dense", "split90"),
    )


def _collection_rows(
    slots: list[dict],
    *,
    seed: int,
    checkpoint_receipt: dict,
    run_receipt: dict,
    snapshot_bytes: dict | None = None,
) -> list[dict]:
    by_slot = {
        (slot["arm"], slot["optimizer_step"]): slot for slot in slots
    }
    rows: list[dict] = []
    for arm in ("dense", "split90"):
        for step in SNAPSHOT_STEPS:
            slot = by_slot[(arm, step)]
            rows.append(
                {
                    "arm": arm,
                    "bytes": (
                        4_096 + step
                        if snapshot_bytes is None
                        else snapshot_bytes[(arm, step)]
                    ),
                    "kind": "snapshot",
                    "sha256": slot["checkpoint_sha256"],
                    "step": step,
                    "uri": f"{S3_ROOT}/" + slot["s3_object_key"],
                    "version_id": slot["s3_version_id"],
                }
            )
        log_sha256 = digest(f"training-log-{seed}-{arm}")
        rows.append(
            {
                "arm": arm,
                "bytes": 2_048,
                "kind": "log",
                "sha256": log_sha256,
                "step": None,
                "uri": (
                    f"{S3_ROOT}/" + log_object_key(seed, arm, log_sha256)
                ),
                "version_id": f"log-version-{seed}-{arm}",
            }
        )
    for arm in ("dense", "split90"):
        checkpoint_sha256 = digest(f"terminal-checkpoint-{seed}-{arm}")
        rows.append(
            {
                "arm": arm,
                "bytes": 8_192,
                "kind": "checkpoint",
                "sha256": checkpoint_sha256,
                "step": None,
                "uri": (
                    f"{S3_ROOT}/"
                    + checkpoint_object_key(seed, arm, checkpoint_sha256)
                ),
                "version_id": f"checkpoint-version-{seed}-{arm}",
            }
        )
    rows.append(
        {
            "arm": None,
            "kind": "checkpoint_receipt",
            "step": None,
            **checkpoint_receipt,
        }
    )
    rows.append(
        {
            "arm": None,
            "kind": "run_receipt",
            "step": None,
            **run_receipt,
        }
    )
    return rows


def build_seed_lifecycles(
    *,
    snapshots: list[dict],
    provider_selection: dict,
    core: dict | None = None,
    boot_ids: dict[int, str] | None = None,
    dense_operational_config_sha256: str = (
        DENSE_OPERATIONAL_CONFIG_SHA256
    ),
    split90_operational_config_sha256: str = (
        SPLIT90_OPERATIONAL_CONFIG_SHA256
    ),
    release_sha256: str = RELEASE_SHA256,
    release_receipt_sha256: str = RELEASE_RECEIPT_SHA256,
    evidence_refs: dict[int, dict] | None = None,
    mutate_collection=None,
    mutate_lifecycle=None,
) -> tuple[list[dict], list[tuple[bytes, str, str, str]]]:
    """Build ten lock seed lifecycles plus matching collection receipts.

    Returns ``(seed_lifecycles, receipts)`` where each receipt is the tuple
    ``(payload, uri, sha256, version_id)`` for one seed in ascending order.
    ``evidence_refs`` optionally pins per-seed real finalization/checkpoint
    receipt identities and snapshot byte counts for payload-backed fixtures.
    """

    values = lifecycle_core() if core is None else core
    lifecycles: list[dict] = []
    receipts: list[tuple[bytes, str, str, str]] = []
    for seed in SEEDS:
        slots = [slot for slot in snapshots if slot["seed"] == seed]
        boot_id = (boot_ids or {}).get(seed, values["boot_id"])
        refs = (evidence_refs or {}).get(seed, {})
        run_manifest_sha256 = digest(f"run-manifest-{seed}")
        finalization_sha256 = refs.get(
            "finalization_sha256",
            digest(f"finalization-receipt-{seed}"),
        )
        finalization_bytes = refs.get("finalization_bytes", 2_048 + seed)
        finalization_version = f"finalization-version-{seed}"
        checkpoint_receipt = {
            "bytes": refs.get("checkpoint_receipt_bytes", 1_000 + seed),
            "sha256": slots[0]["checkpoint_receipt_sha256"],
            "uri": (
                f"{S3_ROOT}/"
                + slots[0]["checkpoint_receipt_s3_object_key"]
            ),
            "version_id": slots[0]["checkpoint_receipt_s3_version_id"],
        }
        run_receipt = {
            "bytes": finalization_bytes,
            "sha256": finalization_sha256,
            "uri": (
                f"{S3_ROOT}/" + run_receipt_key(seed, finalization_sha256)
            ),
            "version_id": finalization_version,
        }
        collection = {
            "boot_id": boot_id,
            "canary_receipt_sha256": provider_selection[
                "canary_receipt_sha256"
            ],
            "checkpoint_receipt": dict(checkpoint_receipt),
            "cohort_id": provider_selection["cohort_id"],
            "collected_at": "2026-07-24T02:00:00Z",
            "complete": True,
            "dataset_build_id": slots[0]["data_build_id"],
            "dataset_receipt_sha256": slots[0]["data_receipt_sha256"],
            "environment_receipt_sha256": provider_selection[
                "environment_receipt_sha256"
            ],
            "hardware_amendment_sha256": provider_selection[
                "hardware_amendment_sha256"
            ],
            "instance_id": values["instance_id"],
            "objective_controls_contract_sha256": values[
                "objective_controls_contract_sha256"
            ],
            "objects": _collection_rows(
                slots,
                seed=seed,
                checkpoint_receipt=checkpoint_receipt,
                run_receipt=run_receipt,
                snapshot_bytes=refs.get("snapshot_bytes"),
            ),
            "ordered_stream_sha256": slots[0]["ordered_stream_sha256"],
            "profile_id": provider_selection["profile_id"],
            "profile_sha256": provider_selection["profile_sha256"],
            "provider": provider_selection["selected_provider"],
            "provider_selection_sha256": provider_selection[
                "provider_selection_sha256"
            ],
            "provider_selection_version_id": provider_selection[
                "provider_selection_s3_version_id"
            ],
            "qualification_approval_public_key_sha256": provider_selection[
                "approval_public_key_sha256"
            ],
            "qualification_approval_receipt_sha256": provider_selection[
                "approval_receipt_sha256"
            ],
            "qualification_evidence_sha256": provider_selection[
                "qualification_evidence_sha256"
            ],
            "receipt_type": "memorysplit-aws-seed-collection-v3",
            "release_receipt_sha256": release_receipt_sha256,
            "release_sha256": release_sha256,
            "request_id": f"{seed:x}".zfill(32),
            "run_manifest_sha256": run_manifest_sha256,
            "run_receipt": dict(run_receipt),
            "runtime_lock_sha256": provider_selection[
                "runtime_lock_sha256"
            ],
            "runtime_sbom_sha256": values["runtime_sbom_sha256"],
            "schema_version": 3,
            "seed": seed,
            "source_commit": values["source_commit"],
            "source_tree": values["source_tree"],
        }
        if mutate_collection is not None:
            mutate_collection(seed, collection)
        payload = canonical_receipt_bytes(collection)
        collection_sha256 = hashlib.sha256(payload).hexdigest()
        collection_uri = (
            f"{S3_ROOT}/" + collection_receipt_key(seed, collection_sha256)
        )
        collection_version = f"collection-version-{seed}"
        lifecycle = {
            "account_id": values["account_id"],
            "availability_zone": values["availability_zone"],
            "boot_id": boot_id,
            "collection_receipt_s3_uri": collection_uri,
            "collection_receipt_s3_version_id": collection_version,
            "collection_receipt_sha256": collection_sha256,
            "dense_operational_config_sha256": (
                dense_operational_config_sha256
            ),
            "finalization_receipt_bytes": finalization_bytes,
            "finalization_receipt_s3_uri": run_receipt["uri"],
            "finalization_receipt_s3_version_id": finalization_version,
            "finalization_receipt_sha256": finalization_sha256,
            "instance_id": values["instance_id"],
            "objective_controls_contract_sha256": values[
                "objective_controls_contract_sha256"
            ],
            "purchase_model": values["purchase_model"],
            "region": values["region"],
            "run_manifest_sha256": run_manifest_sha256,
            "runtime_sbom_sha256": values["runtime_sbom_sha256"],
            "seed": seed,
            "source_commit": values["source_commit"],
            "source_tree": values["source_tree"],
            "split90_operational_config_sha256": (
                split90_operational_config_sha256
            ),
        }
        if mutate_lifecycle is not None:
            mutate_lifecycle(seed, lifecycle)
        lifecycles.append(lifecycle)
        receipts.append(
            (payload, collection_uri, collection_sha256, collection_version)
        )
    return lifecycles, receipts


@dataclass(frozen=True)
class CollectedEvidenceFixture:
    """One payload-backed canonical Task 3F evidence set for ten seeds."""

    provider_selection: dict
    snapshots: list[dict]
    seed_lifecycles: list[dict]
    collection_receipts: list[tuple[bytes, str, str, str]]
    finalization_payloads: dict[int, bytes]
    checkpoint_receipt_payloads: dict[int, bytes]
    snapshot_payloads: dict[tuple[int, str, int], bytes]


def _snapshot_state_bytes(state: dict) -> bytes:
    import io

    import torch

    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def build_collected_evidence(
    *,
    provider_selection: dict | None = None,
    core: dict | None = None,
    boot_ids: dict[int, str] | None = None,
    dense_operational_config_sha256: str = (
        DENSE_OPERATIONAL_CONFIG_SHA256
    ),
    split90_operational_config_sha256: str = (
        SPLIT90_OPERATIONAL_CONFIG_SHA256
    ),
    release_sha256: str = RELEASE_SHA256,
    release_receipt_sha256: str = RELEASE_RECEIPT_SHA256,
    mutate_snapshot_state=None,
    mutate_checkpoint=None,
    mutate_finalization=None,
    mutate_collection=None,
    mutate_lifecycle=None,
) -> CollectedEvidenceFixture:
    """Build ten seeds of real receipt payloads plus 100 snapshot payloads.

    Every hash and byte count in the returned collection receipts is derived
    from the actual finalization/checkpoint receipt payloads and the actual
    ``torch.save`` snapshot payloads, so the set admits through the real
    Task 3C/3D/3F parsers.
    """

    import torch

    from train.trainer import _canonical_json_hash

    selection = (
        default_provider_selection()
        if provider_selection is None
        else provider_selection
    )
    values = lifecycle_core() if core is None else core
    arm_configs = {
        "dense": dense_operational_config_sha256,
        "split90": split90_operational_config_sha256,
    }
    model_cfg_sha256 = _canonical_json_hash(MODEL_CFG)

    snapshots: list[dict] = []
    snapshot_payloads: dict[tuple[int, str, int], bytes] = {}
    finalization_payloads: dict[int, bytes] = {}
    checkpoint_receipt_payloads: dict[int, bytes] = {}
    evidence_refs: dict[int, dict] = {}
    for seed in SEEDS:
        boot_id = (boot_ids or {}).get(seed, values["boot_id"])
        binding = expected_lifecycle_binding(
            selection,
            seed,
            core=values,
            boot_id=boot_id,
        )
        seed_slots: list[dict] = []
        arm_rows: dict[str, dict] = {}
        snapshot_bytes: dict[tuple[str, int], int] = {}
        for arm in ARMS:
            run_id = f"memorysplit-v3-360m-s{seed}-{arm}"
            config_sha256 = arm_configs[arm]
            training_config_sha256 = digest(
                f"runtime-config:{seed}:{arm}"
            )
            config_fingerprint = digest(f"config:{seed}:{arm}")
            metadata = lifecycle_operational_metadata(
                binding,
                run_id=run_id,
                arm=arm,
                config_sha256=config_sha256,
                dataset_receipt_sha256=DATA_RECEIPT_SHA256,
                dataset_build_id=DATA_BUILD_ID,
                ordered_stream_sha256=ORDERED_STREAM_SHA256,
                source_commit=values["source_commit"],
                source_tree=values["source_tree"],
            )
            data_provenance = {
                "arm": arm,
                "seed": seed,
                "source": "fixture-provenance",
            }
            data_provenance_sha256 = _canonical_json_hash(data_provenance)
            study_identity = {
                "arm": arm,
                "cohort_id": STUDY_COHORT_ID,
                "config_sha256": training_config_sha256,
                "data_build_id": DATA_BUILD_ID,
                "data_provenance_sha256": data_provenance_sha256,
                "data_receipt_sha256": DATA_RECEIPT_SHA256,
                "model_cfg_sha256": model_cfg_sha256,
                "model_identity": MODEL_IDENTITY,
                "ordered_stream_sha256": ORDERED_STREAM_SHA256,
                "run_id": run_id,
                "seed": seed,
                "tokens_per_step": TOKENS_PER_STEP,
            }
            arm_rows[arm] = {
                "run_id": run_id,
                "config_sha256": config_sha256,
                "config_fingerprint": config_fingerprint,
            }
            for step in SNAPSHOT_STEPS:
                state = {
                    "model": {"weight": torch.zeros(2, 2)},
                    "model_cfg": dict(MODEL_CFG),
                    "data_provenance": dict(data_provenance),
                    "step": step,
                    "world_size": 4,
                    "config_fingerprint": config_fingerprint,
                    "snapshot_version": 2,
                    "study_identity": dict(study_identity),
                    **metadata,
                }
                if mutate_snapshot_state is not None:
                    mutate_snapshot_state(seed, arm, step, state)
                payload = _snapshot_state_bytes(state)
                snapshot_payloads[(seed, arm, step)] = payload
                snapshot_bytes[(arm, step)] = len(payload)
                snapshot_sha256 = hashlib.sha256(payload).hexdigest()
                seed_slots.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "optimizer_step": step,
                        "checkpoint_sha256": snapshot_sha256,
                        "s3_object_key": snapshot_object_key(
                            seed,
                            arm,
                            step,
                            snapshot_sha256,
                        ),
                        "s3_version_id": (
                            f"checkpoint-version-{seed}-{arm}-{step}"
                        ),
                        "checkpoint_receipt_sha256": "",
                        "checkpoint_receipt_s3_object_key": "",
                        "checkpoint_receipt_s3_version_id": (
                            f"receipt-version-{seed}"
                        ),
                        "provider_selection_sha256": selection[
                            "provider_selection_sha256"
                        ],
                        "provider_selection_s3_version_id": selection[
                            "provider_selection_s3_version_id"
                        ],
                        "snapshot_version": 2,
                        "training_run_id": run_id,
                        "config_fingerprint": config_fingerprint,
                        "training_config_sha256": training_config_sha256,
                        "model_config_sha256": model_cfg_sha256,
                        "model_identity": MODEL_IDENTITY,
                        "data_provenance_sha256": data_provenance_sha256,
                        "data_receipt_sha256": DATA_RECEIPT_SHA256,
                        "data_build_id": DATA_BUILD_ID,
                        "ordered_stream_sha256": ORDERED_STREAM_SHA256,
                        "world_size": 4,
                        "tokens_per_step": TOKENS_PER_STEP,
                    }
                )

        checkpoint_value = {
            "account_id": values["account_id"],
            "arms": ["dense", "split90"],
            "availability_zone": values["availability_zone"],
            "boot_id": boot_id,
            "checkpoints": [
                {
                    "arm": arm,
                    "checkpoint_version": 3,
                    "config_fingerprint": arm_rows[arm][
                        "config_fingerprint"
                    ],
                    "config_sha256": arm_rows[arm]["config_sha256"],
                    "data": {
                        "build_id": DATA_BUILD_ID,
                        "global_cursor": TERMINAL_STEP * TOKENS_PER_STEP,
                        "ordered_stream_sha256": ORDERED_STREAM_SHA256,
                        "receipt_sha256": DATA_RECEIPT_SHA256,
                        "sidecar_name": f"{arm}_target_weights",
                    },
                    "object": {
                        "bytes": 8_192,
                        "sha256": digest(
                            f"terminal-checkpoint-{seed}-{arm}"
                        ),
                        "uri": (
                            f"{S3_ROOT}/"
                            + checkpoint_object_key(
                                seed,
                                arm,
                                digest(
                                    f"terminal-checkpoint-{seed}-{arm}"
                                ),
                            )
                        ),
                        "version_id": f"checkpoint-version-{seed}-{arm}",
                    },
                    "run_id": arm_rows[arm]["run_id"],
                    "seed": seed,
                    "step": TERMINAL_STEP,
                    "world_size": 4,
                }
                for arm in ARMS
            ],
            "cohort_id": selection["cohort_id"],
            "dataset_build_id": DATA_BUILD_ID,
            "dataset_receipt_sha256": DATA_RECEIPT_SHA256,
            "environment_receipt_sha256": selection[
                "environment_receipt_sha256"
            ],
            "freshness": {
                "deadline_at": "2026-07-24T01:20:00Z",
                "max_age_seconds": 1_200,
                "requested_at": "2026-07-24T01:00:00Z",
                "staged_at": "2026-07-24T01:10:00Z",
            },
            "hardware_amendment_sha256": selection[
                "hardware_amendment_sha256"
            ],
            "instance_id": values["instance_id"],
            "objective_controls_contract_sha256": values[
                "objective_controls_contract_sha256"
            ],
            "ordered_stream_sha256": ORDERED_STREAM_SHA256,
            "profile_id": selection["profile_id"],
            "profile_sha256": selection["profile_sha256"],
            "provider": selection["selected_provider"],
            "provider_selection_sha256": selection[
                "provider_selection_sha256"
            ],
            "provider_selection_version_id": selection[
                "provider_selection_s3_version_id"
            ],
            "purchase_model": values["purchase_model"],
            "qualification_approval_public_key_sha256": selection[
                "approval_public_key_sha256"
            ],
            "qualification_approval_receipt_sha256": selection[
                "approval_receipt_sha256"
            ],
            "qualification_canary_receipt_sha256": selection[
                "canary_receipt_sha256"
            ],
            "qualification_environment_receipt_sha256": selection[
                "environment_receipt_sha256"
            ],
            "qualification_evidence_sha256": selection[
                "qualification_evidence_sha256"
            ],
            "reason": "periodic",
            "receipt_type": "memorysplit-aws-paired-checkpoint-v3",
            "region": values["region"],
            "release_receipt_sha256": release_receipt_sha256,
            "release_sha256": release_sha256,
            "request_id": digest(f"checkpoint-request-{seed}")[:32],
            "resumable": True,
            "run_manifest_sha256": digest(f"run-manifest-{seed}"),
            "runtime_lock_sha256": selection["runtime_lock_sha256"],
            "runtime_sbom_sha256": values["runtime_sbom_sha256"],
            "schema_version": 3,
            "seed": seed,
            "source_commit": values["source_commit"],
            "source_tree": values["source_tree"],
        }
        if mutate_checkpoint is not None:
            mutate_checkpoint(seed, checkpoint_value)
        checkpoint_payload = canonical_receipt_bytes(checkpoint_value)
        checkpoint_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()
        checkpoint_receipt_payloads[seed] = checkpoint_payload
        for slot in seed_slots:
            slot["checkpoint_receipt_sha256"] = checkpoint_sha256
            slot["checkpoint_receipt_s3_object_key"] = (
                checkpoint_receipt_key(seed, checkpoint_sha256)
            )

        finalization_value = {
            "arms": [
                {
                    "arm": arm,
                    "config_fingerprint": arm_rows[arm][
                        "config_fingerprint"
                    ],
                    "config_sha256": arm_rows[arm]["config_sha256"],
                    "final_step": TERMINAL_STEP,
                    "log": {
                        "bytes": 2_048,
                        "sha256": digest(f"training-log-{seed}-{arm}"),
                        "uri": (
                            f"{S3_ROOT}/"
                            + log_object_key(
                                seed,
                                arm,
                                digest(f"training-log-{seed}-{arm}"),
                            )
                        ),
                        "version_id": f"log-version-{seed}-{arm}",
                    },
                    "run_id": arm_rows[arm]["run_id"],
                    "snapshots": [
                        {
                            "object": {
                                "bytes": snapshot_bytes[(arm, step)],
                                "sha256": hashlib.sha256(
                                    snapshot_payloads[(seed, arm, step)]
                                ).hexdigest(),
                                "uri": (
                                    f"{S3_ROOT}/"
                                    + snapshot_object_key(
                                        seed,
                                        arm,
                                        step,
                                        hashlib.sha256(
                                            snapshot_payloads[
                                                (seed, arm, step)
                                            ]
                                        ).hexdigest(),
                                    )
                                ),
                                "version_id": (
                                    f"checkpoint-version-{seed}-{arm}-{step}"
                                ),
                            },
                            "step": step,
                        }
                        for step in SNAPSHOT_STEPS
                    ],
                    "world_size": 4,
                }
                for arm in ARMS
            ],
            "boot_id": boot_id,
            "canary_receipt_sha256": selection["canary_receipt_sha256"],
            "checkpoint_receipt": {
                "sha256": checkpoint_sha256,
                "uri": (
                    f"{S3_ROOT}/"
                    + checkpoint_receipt_key(seed, checkpoint_sha256)
                ),
                "version_id": f"receipt-version-{seed}",
            },
            "cohort_id": selection["cohort_id"],
            "complete": True,
            "dataset_build_id": DATA_BUILD_ID,
            "dataset_receipt_sha256": DATA_RECEIPT_SHA256,
            "environment_receipt_sha256": selection[
                "environment_receipt_sha256"
            ],
            "finalized_at": "2026-07-24T01:30:00Z",
            "hardware_amendment_sha256": selection[
                "hardware_amendment_sha256"
            ],
            "instance_id": values["instance_id"],
            "objective_controls_contract_sha256": values[
                "objective_controls_contract_sha256"
            ],
            "ordered_stream_sha256": ORDERED_STREAM_SHA256,
            "profile_id": selection["profile_id"],
            "profile_sha256": selection["profile_sha256"],
            "provider": selection["selected_provider"],
            "provider_selection_sha256": selection[
                "provider_selection_sha256"
            ],
            "provider_selection_version_id": selection[
                "provider_selection_s3_version_id"
            ],
            "qualification_approval_public_key_sha256": selection[
                "approval_public_key_sha256"
            ],
            "qualification_approval_receipt_sha256": selection[
                "approval_receipt_sha256"
            ],
            "qualification_evidence_sha256": selection[
                "qualification_evidence_sha256"
            ],
            "receipt_type": "memorysplit-aws-paired-run-finalization-v3",
            "release_receipt_sha256": release_receipt_sha256,
            "release_sha256": release_sha256,
            "request_id": digest(f"finalization-request-{seed}")[:32],
            "run_manifest_sha256": digest(f"run-manifest-{seed}"),
            "runtime_lock_sha256": selection["runtime_lock_sha256"],
            "runtime_sbom_sha256": values["runtime_sbom_sha256"],
            "schema_version": 3,
            "seed": seed,
            "source_commit": values["source_commit"],
            "source_tree": values["source_tree"],
        }
        if mutate_finalization is not None:
            mutate_finalization(seed, finalization_value)
        finalization_payload = canonical_receipt_bytes(finalization_value)
        finalization_payloads[seed] = finalization_payload
        evidence_refs[seed] = {
            "finalization_sha256": hashlib.sha256(
                finalization_payload
            ).hexdigest(),
            "finalization_bytes": len(finalization_payload),
            "checkpoint_receipt_bytes": len(checkpoint_payload),
            "snapshot_bytes": snapshot_bytes,
        }
        snapshots.extend(seed_slots)

    seed_lifecycles, collection_receipts = build_seed_lifecycles(
        snapshots=snapshots,
        provider_selection=selection,
        core=values,
        boot_ids=boot_ids,
        dense_operational_config_sha256=dense_operational_config_sha256,
        split90_operational_config_sha256=(
            split90_operational_config_sha256
        ),
        release_sha256=release_sha256,
        release_receipt_sha256=release_receipt_sha256,
        evidence_refs=evidence_refs,
        mutate_collection=mutate_collection,
        mutate_lifecycle=mutate_lifecycle,
    )
    return CollectedEvidenceFixture(
        provider_selection=selection,
        snapshots=snapshots,
        seed_lifecycles=seed_lifecycles,
        collection_receipts=collection_receipts,
        finalization_payloads=finalization_payloads,
        checkpoint_receipt_payloads=checkpoint_receipt_payloads,
        snapshot_payloads=snapshot_payloads,
    )
