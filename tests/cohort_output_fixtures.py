"""Shared payload-real v3 100-output cohort builder (factored from Task 4C).

Every artifact produced here admits through the production confirmatory
parsers (runner scoring, sealed-release replay, aggregate authentication),
so both the aggregation tests and the cohort evidence-collection tests can
build the same effect/practical-null cohorts. The builder is torch-free:
aggregation never opens snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import evals.confirmatory.aggregate as aggregate_module
import evals.confirmatory.runner as runner_module
from evals.confirmatory.actions import ActionSlot
from evals.confirmatory.aggregate import (
    CollectionReceiptEvidence,
    SnapshotOutputReference,
)
from evals.confirmatory.contracts import (
    STUDY_CONTRACT_VERSION,
    Control,
    MemoryMode,
    StudyArm,
    canonical_json_bytes,
    canonical_sha256,
    store_content_sha256,
)
from evals.confirmatory.fixtures import positive_fixture
from evals.confirmatory.runner import Submission
from evals.confirmatory.sealing import (
    _validated_release_from_preregistration_sha256,
)
from evals.confirmatory.study_lock import (
    FROZEN_PREREGISTRATION_SHA256_V3,
    STUDY_LOCK_SCHEMA_V3,
    StudyLockV3,
)
from msctl.aws_contracts import (
    ARMS,
    SEEDS,
    SNAPSHOT_STEPS,
    checkpoint_receipt_key,
    snapshot_object_key,
)
from tests.study_lock_fixtures import (
    build_seed_lifecycles,
    default_provider_selection,
)


TERMINAL_STEP = SNAPSHOT_STEPS[-1]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _jsonl(records: list[dict]) -> bytes:
    return b"".join(canonical_json_bytes(record) for record in records)


def _source_bytes() -> tuple[bytes, bytes, bytes]:
    fixture = positive_fixture()
    items = [
        json.loads(line)
        for line in fixture.artifacts["items.jsonl"].splitlines()
    ]
    stores = [
        json.loads(line)
        for line in fixture.artifacts["stores.jsonl"].splitlines()
    ]
    gold = [
        json.loads(line)
        for line in fixture.artifacts["sealed-gold.jsonl"].splitlines()
    ]
    store_hashes = {}
    for store in stores:
        for row in store["rows"]:
            if row["relation_id"] == "P1":
                row["relation_id"] = "r0"
        store["content_sha256"] = store_content_sha256(
            store["store_id"],
            store["world_id"],
            store["rows"],
        )
        store_hashes[store["store_id"]] = store["content_sha256"]
    store_by_item = {item["item_id"]: item["store_id"] for item in items}
    for record in gold:
        for action in record["proof"]:
            if action["relation_id"] == "P1":
                action["relation_id"] = "r0"
        record["store_sha256"] = store_hashes[store_by_item[record["item_id"]]]
    return _jsonl(items), _jsonl(stores), _jsonl(gold)


def _snapshots() -> list[dict]:
    selection_sha256 = "a" * 64
    selection_version = "provider-selection-version-p5"
    snapshots = []
    for seed in SEEDS:
        receipt_sha256 = _digest(f"receipt:{seed}")
        for arm in ARMS:
            for step in SNAPSHOT_STEPS:
                checkpoint_sha256 = _digest(f"checkpoint:{seed}:{arm}:{step}")
                snapshots.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "optimizer_step": step,
                        "checkpoint_sha256": checkpoint_sha256,
                        "s3_object_key": snapshot_object_key(
                            seed,
                            arm,
                            step,
                            checkpoint_sha256,
                        ),
                        "s3_version_id": (
                            f"checkpoint-version-{seed}-{arm}-{step}"
                        ),
                        "checkpoint_receipt_sha256": receipt_sha256,
                        "checkpoint_receipt_s3_object_key": (
                            checkpoint_receipt_key(seed, receipt_sha256)
                        ),
                        "checkpoint_receipt_s3_version_id": (
                            f"receipt-version-{seed}"
                        ),
                        "provider_selection_sha256": selection_sha256,
                        "provider_selection_s3_version_id": selection_version,
                        "snapshot_version": 2,
                        "training_run_id": (
                            f"memorysplit-v3-360m-s{seed}-{arm}"
                        ),
                        "config_fingerprint": _digest(
                            f"config:{seed}:{arm}"
                        ),
                        "training_config_sha256": _digest(
                            f"config-bytes:{seed}:{arm}"
                        ),
                        "model_config_sha256": _digest("model-config"),
                        "model_identity": "d360m",
                        "data_provenance_sha256": _digest(
                            f"data:{seed}:{arm}"
                        ),
                        "data_receipt_sha256": _digest("data-receipt"),
                        "data_build_id": _digest("data-build"),
                        "ordered_stream_sha256": _digest("ordered-stream"),
                        "world_size": 4,
                        "tokens_per_step": 524_288,
                    }
                )
    return snapshots


def _study_lock_and_receipts(
    release_sha256: str,
) -> tuple[StudyLockV3, str, tuple[CollectionReceiptEvidence, ...]]:
    selection = default_provider_selection()
    snapshots = _snapshots()
    lifecycles, raw_receipts = build_seed_lifecycles(
        snapshots=snapshots,
        provider_selection=selection,
    )
    lock = StudyLockV3.from_dict(
        {
            "record_type": STUDY_LOCK_SCHEMA_V3,
            "schema_version": STUDY_CONTRACT_VERSION,
            "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256_V3,
            "sealed_evaluation_release_sha256": release_sha256,
            "provider_selection": selection,
            "snapshots": snapshots,
            "seed_lifecycles": lifecycles,
        }
    )
    receipts = tuple(
        CollectionReceiptEvidence(
            payload=payload,
            uri=uri,
            sha256=sha256,
            version_id=version_id,
        )
        for payload, uri, sha256, version_id in raw_receipts
    )
    return lock, canonical_sha256(lock.to_dict()), receipts


def _execution_identity(binding) -> dict:
    return {
        "production_qualified": True,
        "adapter_kind": "repository",
        "selected_provider": binding.selected_provider,
        "profile_id": binding.evaluator_profile_id,
        "profile_sha256": binding.evaluator_profile_sha256,
        "runtime_lock_sha256": binding.evaluator_runtime_lock_sha256,
        "qualification_evidence_sha256": (
            binding.evaluator_qualification_evidence_sha256
        ),
        "environment_receipt_sha256": (
            binding.evaluator_environment_receipt_sha256
        ),
        "canary_receipt_sha256": binding.evaluator_canary_receipt_sha256,
        "approval_receipt_sha256": binding.evaluator_approval_receipt_sha256,
        "approval_public_key_sha256": (
            binding.evaluator_approval_public_key_sha256
        ),
        "account_id": "123456789012",
        "instance_id": "i-0123456789abcdef0",
        "boot_id": "12345678-1234-4abc-8def-1234567890ab",
        "region": "us-east-1",
        "device_type": "cuda",
        "cuda_available": True,
        "torch_version": "2.6.0",
        "cuda_version": "12.8",
        "device_count": 8,
        "device_name": "NVIDIA H100 80GB",
        "device_capability": [9, 0],
    }


def _submissions(
    *,
    items: list[dict],
    gold: list[dict],
    binding,
    mode: str,
) -> dict[str, Submission]:
    item_by_id = {item["item_id"]: item for item in items}
    result = {}
    for record in gold:
        item = item_by_id[record["item_id"]]
        is_primary_control = (
            item["memory_mode"] == MemoryMode.MEMORY_ON.value
            and item["control"] == Control.CORRECT.value
        )
        correct = True
        if is_primary_control:
            if mode == "effect":
                correct = (
                    binding.arm is StudyArm.SPLIT90
                    and binding.optimizer_step == TERMINAL_STEP
                    and item["family"] == "graph"
                )
            elif mode == "null":
                correct = False
            else:
                raise AssertionError(mode)
        result[record["item_id"]] = Submission(
            item_id=record["item_id"],
            answer=record["answer"] if correct else "wrong-answer",
            actions=tuple(
                ActionSlot.from_dict(action) for action in record["proof"]
            ),
        )
    return result


@dataclass(frozen=True)
class _CohortFixture:
    root: Path
    lock: StudyLockV3
    lock_sha256: str
    references: tuple[SnapshotOutputReference, ...]
    receipts: tuple[CollectionReceiptEvidence, ...]


def _build_cohort(root: Path, mode: str) -> _CohortFixture:
    root.chmod(0o700)
    items_bytes, stores_bytes, gold_bytes = _source_bytes()
    validated_release = _validated_release_from_preregistration_sha256(
        preregistration_sha256=FROZEN_PREREGISTRATION_SHA256_V3,
        items_content=items_bytes,
        stores_content=stores_bytes,
        gold_content=gold_bytes,
    )
    release_sha256 = hashlib.sha256(
        validated_release.manifest_bytes
    ).hexdigest()
    lock, lock_sha256, receipts = _study_lock_and_receipts(release_sha256)
    lock_bytes = canonical_json_bytes(lock.to_dict())
    assert hashlib.sha256(lock_bytes).hexdigest() == lock_sha256
    plans = aggregate_module.plan_snapshot_evaluations(
        study_lock=lock,
        study_lock_sha256=lock_sha256,
        collection_receipts=receipts,
    )
    item_records = [json.loads(line) for line in items_bytes.splitlines()]
    gold_records = [json.loads(line) for line in gold_bytes.splitlines()]
    items = {
        record.item_id: record
        for record in runner_module._canonical_jsonl(
            items_bytes,
            name="items.jsonl",
            parser=runner_module.ItemRecord.from_dict,
            identity=lambda record: record.item_id,
        )
    }
    gold = {
        record.item_id: record
        for record in runner_module._canonical_jsonl(
            gold_bytes,
            name="sealed-gold.jsonl",
            parser=runner_module.SealedGoldRecord.from_dict,
            identity=lambda record: record.item_id,
        )
    }
    stores = {
        record.store_id: record
        for record in runner_module._canonical_jsonl(
            stores_bytes,
            name="stores.jsonl",
            parser=runner_module.StoreRecord.from_dict,
            identity=lambda record: record.store_id,
        )
    }
    references = []
    for plan in plans:
        submissions = _submissions(
            items=item_records,
            gold=gold_records,
            binding=plan.binding,
            mode=mode,
        )
        outcomes_bytes, metrics_bytes = runner_module._score_and_summarize_v3(
            items=items,
            gold=gold,
            stores=stores,
            binding=plan.binding,
            submissions=submissions,
        )
        artifacts = {
            "inference.json": runner_module._inference_bytes_v3(plan.binding),
            "items.jsonl": items_bytes,
            "metrics.json": metrics_bytes,
            "outcomes.jsonl": outcomes_bytes,
            "run.json": plan.run_json_bytes,
            "sealed-gold.jsonl": gold_bytes,
            "sealed-release.json": validated_release.manifest_bytes,
            "stores.jsonl": stores_bytes,
            "study-lock.json": lock_bytes,
        }
        identity = _execution_identity(plan.binding)
        manifest = runner_module._snapshot_output_manifest_bytes(
            binding=plan.binding,
            artifacts=artifacts,
            execution_identity_before=identity,
            execution_identity_after=identity,
            production_qualified=True,
        )
        output = root / f"informational-path-{plan.slot_index:03d}"
        output.mkdir(mode=0o700)
        output.chmod(0o700)
        for name, content in {**artifacts, "output.json": manifest}.items():
            path = output / name
            path.write_bytes(content)
            path.chmod(0o600)
        references.append(
            SnapshotOutputReference(
                output_dir=output,
                output_commitment=hashlib.sha256(manifest).hexdigest(),
            )
        )
    return _CohortFixture(
        root,
        lock,
        lock_sha256,
        tuple(references),
        receipts,
    )
