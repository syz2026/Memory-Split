"""Shared v3 study-lock seed-lifecycle and collection-receipt fixtures."""

from __future__ import annotations

import hashlib
import json

from msctl.aws_contracts import (
    SEEDS,
    SNAPSHOT_STEPS,
    checkpoint_object_key,
    collection_receipt_key,
    log_object_key,
    run_receipt_key,
)
from msctl.aws_lifecycle import ProviderLifecycleBinding


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


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


RELEASE_SHA256 = digest("training-release")
RELEASE_RECEIPT_SHA256 = digest("training-release-receipt")
DENSE_OPERATIONAL_CONFIG_SHA256 = digest("dense-operational-config")
SPLIT90_OPERATIONAL_CONFIG_SHA256 = digest("split90-operational-config")


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
                    "bytes": 4_096 + step,
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
    mutate_collection=None,
    mutate_lifecycle=None,
) -> tuple[list[dict], list[tuple[bytes, str, str, str]]]:
    """Build ten lock seed lifecycles plus matching collection receipts.

    Returns ``(seed_lifecycles, receipts)`` where each receipt is the tuple
    ``(payload, uri, sha256, version_id)`` for one seed in ascending order.
    """

    values = lifecycle_core() if core is None else core
    lifecycles: list[dict] = []
    receipts: list[tuple[bytes, str, str, str]] = []
    for seed in SEEDS:
        slots = [slot for slot in snapshots if slot["seed"] == seed]
        boot_id = (boot_ids or {}).get(seed, values["boot_id"])
        run_manifest_sha256 = digest(f"run-manifest-{seed}")
        finalization_sha256 = digest(f"finalization-receipt-{seed}")
        finalization_bytes = 2_048 + seed
        finalization_version = f"finalization-version-{seed}"
        checkpoint_receipt = {
            "bytes": 1_000 + seed,
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
