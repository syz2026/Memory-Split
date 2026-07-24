#!/usr/bin/env python3
"""Build one deterministic, profile-selected AWS GPU v3 handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import package_aws_p5_handoff as _core


PackageError = _core.PackageError
ReleaseArtifacts = _core.ReleaseArtifacts

PACKAGE_FORMAT_VERSION = "aws-gpu-v3"
COHORT_ID = "memorysplit-confirmatory-v3-360m-n10-aws"
COHORT_PATH = "configs/cohort-assignment-v3.json"
PREREGISTRATION_PATH = "configs/preregistration-v3.yaml"
AMENDMENT_PATH = "configs/hardware-amendment-v3.json"
DATASET_POINTER_PATH = "DATASET-POINTER-AWS.json"
CONTAINER_LOCK_PATH = "containers/aws-gpu/image.lock.json"
CONTAINER_DOCKERFILE_PATH = "containers/aws-gpu/Dockerfile"
RUNTIME_DEPENDENCY_LOCK_PATH = "containers/aws-gpu/requirements.lock"
SELECTION_SCHEMA_PATH = "schemas/aws-gpu-provider-selection-v3.schema.json"
RELEASE_RECEIPT_NAME = "RELEASE-AWS-GPU-V3.json"
METADATA_PATH = "RELEASE-METADATA.json"
SUMS_PATH = "SHA256SUMS"
ARMS = ("dense", "split90")
SEEDS = tuple(range(10))
SNAPSHOT_STEPS = (1_358, 3_396, 6_791, 10_187, 13_582)
FROZEN_CONFIGS_SHA256 = (
    "42b650aee789c9861ab5df3638b38cc8c1e661181b8b09423691bab737483a30"
)
FROZEN_HASHES = {
    COHORT_PATH: "2fc8bae1343fa0ff65c2dd6548be20c5b34b009dae10906070b39cc48d14dd5d",
    PREREGISTRATION_PATH: "de7f2213cc4918665252b3c469778c1caa72066c84c7bae6ebe92c28b070a25a",
    AMENDMENT_PATH: "22e201991c56b447b772ece0b9e1667d6787a305b19983bda9a4e8d8eca6cc3b",
    CONTAINER_LOCK_PATH: "3cea8a17349b9c17822eabee9f49ad9d40623b2c6ae76752ec8c1004efd6bbf8",
    RUNTIME_DEPENDENCY_LOCK_PATH: "236ac66261c7dad568c7f236d99b82338c741cf2a8fc9fc4a848acd481a74d32",
    SELECTION_SCHEMA_PATH: "cf0ac0f57c7b6ebf68d54220d35a2c6bbe9f832d2eae5863c7945da44ff7aee2",
}
PROFILE_SPECS = {
    "cluster/profiles/aws-p5.48xlarge-v3.json": {
        "provider": "aws-p5.48xlarge-v3",
        "instance_type": "p5.48xlarge",
        "gpu_model": "NVIDIA H100 80GB",
        "sha256": "0969f70b2d10fb9f2065a7b6f10904658c514fa88cde6b818ae56020826d96bd",
        "purchase_model": "on_demand",
        "slug": "p5",
    },
    "cluster/profiles/aws-p6-b300.48xlarge-v3.json": {
        "provider": "aws-p6-b300.48xlarge-v3",
        "instance_type": "p6-b300.48xlarge",
        "gpu_model": "NVIDIA B300",
        "sha256": "f4b3fc95d5f035decbebb5150854dbe81c7440af4a2097697b106246b8db375f",
        "purchase_model": "capacity_block",
        "slug": "p6-b300",
    },
}
PROFILE_PATHS = tuple(PROFILE_SPECS)
EXPECTED_CONFIGS = frozenset(
    f"configs/360m-v3/{arm}-s{seed}.yaml"
    for seed in SEEDS
    for arm in ARMS
)
_CONFIG_FIELDS = {
    "schema_version",
    "cohort_id",
    "run_id",
    "condition",
    "seed",
    "model",
    "ctx",
    "train_corpus",
    "sidecar_name",
    "out_dir",
    "micro_batch_size",
    "tokens_per_step",
    "max_steps",
    "total_tokens",
    "lr",
    "warmup_steps",
    "weight_decay",
    "compile",
    "device",
    "log_every",
    "eval_every",
    "snapshot_steps",
    "ckpt_minutes",
}
_STATIC_REQUIRED_MEMBERS = frozenset(
    {
        "AWS-GPU-V3-START.md",
        DATASET_POINTER_PATH,
        "requirements.txt",
        "docs/AWS-GPU-ACCESS-REQUEST.md",
        "docs/AWS-GPU-V3-RUNBOOK.md",
        COHORT_PATH,
        PREREGISTRATION_PATH,
        AMENDMENT_PATH,
        "configs/current-dataset-lock.json",
        "configs/reasoning-dataset-v2.json",
        "configs/route-policy.json",
        CONTAINER_DOCKERFILE_PATH,
        CONTAINER_LOCK_PATH,
        RUNTIME_DEPENDENCY_LOCK_PATH,
        SELECTION_SCHEMA_PATH,
        "cluster/aws/p5/bootstrap.py",
        "cluster/aws/p5/bootstrap.sh",
        "cluster/aws/p5/canary.py",
        "cluster/aws/p5/canary_runtime.py",
        "cluster/aws/p5/corpus_contract.py",
        "cluster/aws/p5/interruption_checkpoint.py",
        "cluster/aws/p5/launch_seed_pair.py",
        "cluster/aws/p5/profile.py",
        "corpusgen/__init__.py",
        "corpusgen/bios.py",
        "corpusgen/build.py",
        "corpusgen/current_dataset.py",
        "corpusgen/current_sources.py",
        "corpusgen/deduction.py",
        "corpusgen/factqa.py",
        "corpusgen/graph_records.py",
        "corpusgen/graph_trace.py",
        "corpusgen/igsm_lite.py",
        "corpusgen/records.py",
        "corpusgen/realfact.py",
        "corpusgen/relational_build.py",
        "corpusgen/srgm_worlds.py",
        "corpusgen/wikidata5m.py",
        "corpusgen/parallel/__init__.py",
        "corpusgen/parallel/adapters.py",
        "corpusgen/parallel/canonical.py",
        "corpusgen/parallel/catalog.py",
        "corpusgen/parallel/integrity.py",
        "corpusgen/parallel/metadata.py",
        "corpusgen/parallel/publication.py",
        "corpusgen/parallel/safeio.py",
        "corpusgen/parallel/schedule.py",
        "corpusgen/parallel/tasks.py",
        "corpusgen/parallel/workspace.py",
        "corpusgen/reasoning/__init__.py",
        "corpusgen/reasoning/closure.py",
        "corpusgen/reasoning/proofs.py",
        "corpusgen/reasoning/routing.py",
        "corpusgen/reasoning/smoke.py",
        "corpusgen/reasoning/state.py",
        "evals/__init__.py",
        "evals/confirmatory/__init__.py",
        "evals/confirmatory/__main__.py",
        "evals/confirmatory/actions.py",
        "evals/confirmatory/contracts.py",
        "evals/confirmatory/fixtures.py",
        "evals/confirmatory/inference.py",
        "evals/confirmatory/metrics.py",
        "evals/confirmatory/reporting.py",
        "evals/confirmatory/run_binding.py",
        "evals/confirmatory/runner.py",
        "evals/confirmatory/sealing.py",
        "evals/confirmatory/solver.py",
        "evals/confirmatory/status.py",
        "evals/confirmatory/study_lock.py",
        "msctl/__init__.py",
        "msctl/__main__.py",
        "msctl/approval.py",
        "msctl/aws_argv.py",
        "msctl/aws_control_bundle.py",
        "msctl/aws_fleet.py",
        "msctl/aws_launch_manifest.py",
        "msctl/aws_p5.py",
        "msctl/aws_readiness.py",
        "msctl/aws_resume_launch.py",
        "msctl/aws_sealed_evaluation.py",
        "msctl/aws_sealed_finalization.py",
        "msctl/aws_selection.py",
        "msctl/bootstrap.py",
        "msctl/cleanup.py",
        "msctl/cli.py",
        "msctl/cohort.py",
        "msctl/collect.py",
        "msctl/contracts.py",
        "msctl/dataset.py",
        "msctl/environment.py",
        "msctl/errors.py",
        "msctl/fsutil.py",
        "msctl/jsonutil.py",
        "msctl/operations.py",
        "msctl/profile.py",
        "msctl/slurm.py",
        "msctl/state.py",
        "organizer/__init__.py",
        "organizer/graph_store.py",
        "organizer/store.py",
        "scripts/build_aws_gpu_image.py",
        "scripts/build_parallel_corpus.py",
        "scripts/package_aws_gpu_handoff.py",
        "scripts/package_aws_p5_handoff.py",
        "scripts/run_train.py",
        "scripts/verify_aws_gpu_v3_release.py",
        "sources/Wikidata-CC0-1.0.txt",
        "sources/current-dataset-licenses.json",
        "sources/wikidata5m.lock.json",
        "train/__init__.py",
        "train/data.py",
        "train/model.py",
        "train/safeio.py",
        "train/tokenizer.py",
        "train/trainer.py",
        "vendor/tiktoken/6c7ea1a7e38e3a7f062df639a5b80947f075ffe6",
        "vendor/tiktoken/6d1cbeee0f20b3d9449abfede4726ed8212e3aee",
    }
) | EXPECTED_CONFIGS
CONTRACT_GROUPS = {
    "provider_selection": (
        SELECTION_SCHEMA_PATH,
        "msctl/aws_selection.py",
    ),
    "fleet": (
        "msctl/aws_fleet.py",
    ),
    "lifecycle": (
        "msctl/approval.py",
        "msctl/aws_argv.py",
        "msctl/aws_control_bundle.py",
        "msctl/aws_launch_manifest.py",
        "msctl/aws_p5.py",
        "msctl/aws_readiness.py",
        "msctl/aws_sealed_evaluation.py",
        "msctl/aws_sealed_finalization.py",
        "msctl/cli.py",
        "msctl/contracts.py",
        "evals/confirmatory/run_binding.py",
        "evals/confirmatory/sealing.py",
        "msctl/operations.py",
        "msctl/state.py",
    ),
    "canary": (
        "cluster/aws/p5/canary.py",
        "cluster/aws/p5/canary_runtime.py",
    ),
    "container": (
        CONTAINER_DOCKERFILE_PATH,
        CONTAINER_LOCK_PATH,
        RUNTIME_DEPENDENCY_LOCK_PATH,
        "scripts/build_aws_gpu_image.py",
    ),
}
_MUTABLE_IMAGE_TAG = re.compile(
    rb"(?i)(?:^|[\"'\s])(?:[a-z0-9._-]+(?:/[a-z0-9._/-]+)+):latest"
)


@dataclass(frozen=True)
class _Collected:
    payload: dict[str, bytes]
    modes: dict[str, str]
    tree_id: str
    members_sha256: str
    config_sha256: dict[str, str]
    bindings: dict[str, dict[str, object]]
    contract_locks: dict[str, dict[str, object]]
    environment: dict[str, object]
    seed_assignment: dict[str, object]


def required_members(profile_path: str) -> frozenset[str]:
    """Return the exact regular-file inventory for one selected profile."""

    if profile_path not in PROFILE_SPECS:
        raise PackageError("profile is not one of the two closed v3 profiles")
    return _STATIC_REQUIRED_MEMBERS | {profile_path}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _binding(path: str, payload: dict[str, bytes]) -> dict[str, object]:
    return {"path": path, "sha256": _sha256(payload[path])}


def _same_typed_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _same_typed_value(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_typed_value(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _expected_config(seed: int, arm: str) -> dict[str, object]:
    return {
        "schema_version": 3,
        "cohort_id": COHORT_ID,
        "run_id": f"memorysplit-v3-360m-s{seed}-{arm}",
        "condition": arm,
        "seed": seed,
        "model": "d360m",
        "ctx": 1024,
        "train_corpus": "dataset/corpus-receipt.json",
        "sidecar_name": (
            "dense_target_weights"
            if arm == "dense"
            else "split90_target_weights"
        ),
        "out_dir": f"runs/seed-{seed}/{arm}",
        "micro_batch_size": 8,
        "tokens_per_step": 524_288,
        "max_steps": 13_582,
        "total_tokens": 7_120_879_616,
        "lr": 0.001,
        "warmup_steps": 300,
        "weight_decay": 0.1,
        "compile": True,
        "device": "cuda",
        "log_every": 20,
        "eval_every": 250,
        "snapshot_steps": list(SNAPSHOT_STEPS),
        "ckpt_minutes": 30,
    }


def _validate_config(path: str, data: bytes, *, seed: int, arm: str) -> None:
    value = _core._load_yaml_object(data, path=path)
    if set(value) != _CONFIG_FIELDS:
        raise PackageError(f"v3 run config fields are not exact: {path}")
    if not _same_typed_value(value, _expected_config(seed, arm)):
        raise PackageError(f"v3 run config violates the frozen cohort: {path}")


def _validate_cohort(payload: dict[str, bytes]) -> None:
    for path, expected in FROZEN_HASHES.items():
        if _sha256(payload[path]) != expected:
            raise PackageError(f"frozen v3 contract hash changed: {path}")
    assignment = _core._load_json_object(
        payload[COHORT_PATH],
        label="v3 cohort assignment",
    )
    expected_assignment = {
        "cohort_id": COHORT_ID,
        "model_parameters": 356_033_536,
        "optimizer_steps": 13_582,
        "provider_seeds": {"aws-p5.48xlarge-v3": list(SEEDS)},
        "raw_target_tokens": 7_120_879_616,
        "schema_version": 3,
        "targets_per_update": 524_288,
    }
    if not _same_typed_value(assignment, expected_assignment):
        raise PackageError("v3 cohort assignment is not the frozen ten-pair cohort")
    for seed in SEEDS:
        for arm in ARMS:
            path = f"configs/360m-v3/{arm}-s{seed}.yaml"
            _validate_config(path, payload[path], seed=seed, arm=arm)
    config_inventory = {
        path: _sha256(payload[path]) for path in sorted(EXPECTED_CONFIGS)
    }
    if _hash_inventory(config_inventory) != FROZEN_CONFIGS_SHA256:
        raise PackageError("canonical v3 config byte inventory changed")


def _validate_profile_and_amendment(
    payload: dict[str, bytes],
    profile_path: str,
) -> dict[str, object]:
    spec = PROFILE_SPECS[profile_path]
    if _sha256(payload[profile_path]) != spec["sha256"]:
        raise PackageError("selected profile bytes differ from the closed profile")
    profile = _core._load_json_object(
        payload[profile_path],
        label="selected AWS GPU v3 profile",
    )
    expected_profile_fields = {
        "schema_version": 3,
        "profile_id": spec["provider"],
        "provider": spec["provider"],
        "instance_type": spec["instance_type"],
        "purchase_model": spec["purchase_model"],
        "assigned_seeds": list(SEEDS),
    }
    for field, expected in expected_profile_fields.items():
        if field not in profile or not _same_typed_value(profile[field], expected):
            raise PackageError(f"selected profile has invalid field: {field}")
    gpu = profile.get("gpu")
    if (
        not isinstance(gpu, dict)
        or gpu.get("model") != spec["gpu_model"]
        or gpu.get("allocated") != 8
        or gpu.get("seed_train_groups") != [4, 4]
    ):
        raise PackageError("selected profile does not have the frozen 4+4 GPU geometry")

    amendment = _core._load_json_object(
        payload[AMENDMENT_PATH],
        label="v3 hardware amendment",
    )
    expected_bindings = {
        "cohort_assignment": {
            "path": COHORT_PATH,
            "sha256": FROZEN_HASHES[COHORT_PATH],
        },
        "preregistration": {
            "path": PREREGISTRATION_PATH,
            "sha256": FROZEN_HASHES[PREREGISTRATION_PATH],
        },
    }
    if (
        amendment.get("schema_version") != 3
        or amendment.get("amendment_id")
        != "memorysplit-confirmatory-v3-hardware-amendment"
        or amendment.get("cohort_id") != COHORT_ID
        or amendment.get("bindings") != expected_bindings
        or amendment.get("original_scientific_provider_assignment")
        != "aws-p5.48xlarge-v3"
        or amendment.get("supersedes_hardware_selection") is not True
        or amendment.get("protected_outcomes_inspected") is not False
    ):
        raise PackageError("hardware amendment does not bind the frozen v3 cohort")
    rows = amendment.get("allowed_profiles")
    if not isinstance(rows, list) or len(rows) != 2:
        raise PackageError("hardware amendment must name exactly two closed profiles")
    expected_rows = [
        {
            "gpu_model": value["gpu_model"],
            "instance_type": value["instance_type"],
            "path": path,
            "profile_id": value["provider"],
            "provider": value["provider"],
            "sha256": value["sha256"],
        }
        for path, value in PROFILE_SPECS.items()
    ]
    if rows != expected_rows:
        raise PackageError("hardware amendment allowed-profile inventory changed")
    rule = amendment.get("selection_rule")
    if rule != {
        "mixed_profiles_forbidden": True,
        "same_profile_for_all_pairs": True,
        "seeds": list(SEEDS),
        "selected_profile_count": 1,
    }:
        raise PackageError("hardware amendment no longer forbids mixed profiles")
    return profile


def _validate_container_lock(payload: dict[str, bytes]) -> dict[str, object]:
    lock = _core._load_json_object(
        payload[CONTAINER_LOCK_PATH],
        label="AWS GPU container base lock",
    )
    expected = {
        "base": {
            "digest": "sha256:3bfc0b4c9561561b4cd578cb78611ec6258e1791854000b58a279d5880afc7d6",
            "image": (
                "public.ecr.aws/deep-learning-containers/"
                "pytorch:2.12.1-cu130-amzn2023"
            ),
        },
        "runtime": {
            "cuda": "13.0",
            "gid": 10001,
            "operating_system": "Amazon Linux 2023",
            "python": "3.11",
            "pytorch": "2.12.1",
            "requirements_sha256": (
                "236ac66261c7dad568c7f236d99b82338c741cf2a8fc9fc4a848acd481a74d32"
            ),
            "uid": 10001,
            "user": "memorysplit",
        },
        "schema_version": 1,
    }
    if not _same_typed_value(lock, expected):
        raise PackageError("container base lock differs from the reviewed contract")
    dockerfile = payload[CONTAINER_DOCKERFILE_PATH]
    if (
        b"FROM ${BASE_IMAGE}@${BASE_DIGEST}" not in dockerfile
        or b"FROM ${BASE_IMAGE}\n" in dockerfile
    ):
        raise PackageError("container Dockerfile does not consume the base digest")
    return lock


def _validate_dataset_pointer(data: bytes) -> None:
    pointer = _core._load_json_object(data, label="AWS dataset pointer")
    if (
        pointer.get("schema_version") != 1
        or pointer.get("provider") != "aws-p5.48xlarge"
        or pointer.get("materialization") != "s3"
        or pointer.get("durable_uri_env") != "MS_S3_ROOT"
        or pointer.get("full_corpus_in_release") is not False
    ):
        raise PackageError("AWS dataset pointer is not the immutable external pointer")


def _validate_selection_schema(data: bytes) -> None:
    schema = _core._load_json_object(data, label="provider-selection schema")
    properties = schema.get("properties")
    profile_pairs = [
        {
            "properties": {
                "gpu_model": {"const": spec["gpu_model"]},
                "instance_type": {"const": spec["instance_type"]},
                "purchase_model": {"const": spec["purchase_model"]},
                "capacity_reservation_id": (
                    {"type": "null"}
                    if spec["purchase_model"] == "on_demand"
                    else {
                        "pattern": "^[a-z]{2,4}-[A-Za-z0-9-]{8,64}$",
                        "type": "string",
                    }
                ),
                "capacity_block_offering_id": (
                    {"type": "null"}
                    if spec["purchase_model"] == "on_demand"
                    else {
                        "pattern": "^[a-z]{2,4}-[A-Za-z0-9-]{8,64}$",
                        "type": "string",
                    }
                ),
                "provider": {"const": spec["provider"]},
                "selected_profile_id": {"const": spec["provider"]},
            },
            "required": [
                "selected_profile_id",
                "provider",
                "instance_type",
                "gpu_model",
                "purchase_model",
                "capacity_reservation_id",
                "capacity_block_offering_id",
            ],
        }
        for spec in PROFILE_SPECS.values()
    ]
    if (
        schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
        or schema.get("oneOf") != profile_pairs
        or not isinstance(properties, dict)
        or properties.get("schema_version") != {"const": 3}
        or properties.get("mixed_profiles") != {"const": False}
        or properties.get("provider", {}).get("enum")
        != [spec["provider"] for spec in PROFILE_SPECS.values()]
    ):
        raise PackageError("provider-selection schema is not closed to P5/P6 v3")


def _runtime_environment_contract(profile_sha256: str) -> dict[str, object]:
    return {
        "mode": "runtime_attested",
        "profile_sha256": profile_sha256,
        "container_image_digest_env": "MS_CONTAINER_DIGEST",
        "container_image_digest_pattern": "^sha256:[0-9a-f]{64}$",
        "runtime_environment_receipt": {
            "required_at_launch": True,
            "authentication": "aws_instance_identity_document_pkcs7",
            "required_fields": [
                "schema_version",
                "profile_sha256",
                "container_image_digest",
                "aws_instance_identity_document",
                "aws_instance_identity_pkcs7",
            ],
        },
    }


def _hash_inventory(inventory: dict[str, str]) -> str:
    canonical = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return _sha256(canonical)


def _contract_locks(payload: dict[str, bytes]) -> dict[str, dict[str, object]]:
    locks: dict[str, dict[str, object]] = {}
    for name, paths in CONTRACT_GROUPS.items():
        inventory = {path: _sha256(payload[path]) for path in sorted(paths)}
        locks[name] = {
            "members": inventory,
            "sha256": _hash_inventory(inventory),
        }
    return locks


def _collect_payload(
    repository: _core._Repository,
    tracked: list[_core._Tracked],
    revision: str,
    tree_id: str,
    profile_path: str,
) -> _Collected:
    expected = required_members(profile_path)
    by_path = {item.path: item for item in tracked}
    missing = sorted(expected - set(by_path))
    if missing:
        raise PackageError(f"required v3 release member is not tracked: {missing[0]}")
    other_profiles = set(PROFILE_PATHS) - {profile_path}
    if other_profiles & expected:
        raise PackageError("release inventory mixes P5 and P6 profiles")

    selected = [by_path[path] for path in sorted(expected)]
    _core._assert_repository_unchanged(repository, revision)
    snapshot = _core._read_git_blobs(repository, selected)
    payload: dict[str, bytes] = {}
    modes: dict[str, str] = {}
    member_rows: list[dict[str, object]] = []
    for item in selected:
        data = snapshot[item.path]
        _core._scan_secret(item.path, data)
        if _MUTABLE_IMAGE_TAG.search(data):
            raise PackageError(f"mutable image tag is forbidden: {item.path}")
        payload[item.path] = data
        modes[item.path] = item.mode
        member_rows.append(
            {
                "path": item.path,
                "bytes": len(data),
                "sha256": _sha256(data),
                "git_blob": item.object_id,
                "git_mode": item.mode,
            }
        )

    _validate_cohort(payload)
    _validate_profile_and_amendment(payload, profile_path)
    container_lock = _validate_container_lock(payload)
    _validate_dataset_pointer(payload[DATASET_POINTER_PATH])
    _validate_selection_schema(payload[SELECTION_SCHEMA_PATH])

    spec = PROFILE_SPECS[profile_path]
    config_sha256 = {
        path: _sha256(payload[path]) for path in sorted(EXPECTED_CONFIGS)
    }
    bindings: dict[str, dict[str, object]] = {
        "cohort_assignment": _binding(COHORT_PATH, payload),
        "preregistration": _binding(PREREGISTRATION_PATH, payload),
        "hardware_amendment": _binding(AMENDMENT_PATH, payload),
        "profile": _binding(profile_path, payload),
        "dataset_pointer": _binding(DATASET_POINTER_PATH, payload),
        "container_base_lock": {
            **_binding(CONTAINER_LOCK_PATH, payload),
            "base_digest": container_lock["base"]["digest"],
            "base_image": (
                f"{container_lock['base']['image']}@"
                f"{container_lock['base']['digest']}"
            ),
        },
    }
    contract_locks = _contract_locks(payload)
    environment = _runtime_environment_contract(str(spec["sha256"]))
    seed_assignment = {
        "cohort_id": COHORT_ID,
        "provider": spec["provider"],
        "seeds": list(SEEDS),
        "arms": list(ARMS),
    }
    metadata = {
        "schema_version": 1,
        "package_format_version": PACKAGE_FORMAT_VERSION,
        "provider": spec["provider"],
        "selected_profile_id": spec["provider"],
        "source": {
            "commit": revision,
            "dirty": False,
            "tree": tree_id,
        },
        "seed_assignment": seed_assignment,
        "cohort_assignment": bindings["cohort_assignment"],
        "preregistration": bindings["preregistration"],
        "hardware_amendment": bindings["hardware_amendment"],
        "profile": bindings["profile"],
        "environment": environment,
        "dataset_pointer": bindings["dataset_pointer"],
        "container_base_lock": bindings["container_base_lock"],
        "config_sha256": config_sha256,
        "contract_locks": contract_locks,
        "members": member_rows,
    }
    payload[METADATA_PATH] = _core._canonical_pretty(metadata)
    modes[METADATA_PATH] = "100644"
    sums = "".join(
        f"{_sha256(payload[name])}  {name}\n" for name in sorted(payload)
    ).encode("ascii")
    payload[SUMS_PATH] = sums
    modes[SUMS_PATH] = "100644"
    return _Collected(
        payload=payload,
        modes=modes,
        tree_id=tree_id,
        members_sha256=_sha256(sums),
        config_sha256=config_sha256,
        bindings=bindings,
        contract_locks=contract_locks,
        environment=environment,
        seed_assignment=seed_assignment,
    )


def _receipt_value(
    *,
    collected: _Collected,
    revision: str,
    release_id: str,
    archive_name: str,
    archive_sha256: str,
    archive_bytes: int,
    profile_path: str,
) -> dict[str, object]:
    spec = PROFILE_SPECS[profile_path]
    bindings = collected.bindings
    return {
        "schema_version": 1,
        "package_format_version": PACKAGE_FORMAT_VERSION,
        "release_id": release_id,
        "provider": spec["provider"],
        "selected_profile_id": spec["provider"],
        "archive": {
            "path": archive_name,
            "sha256": archive_sha256,
            "bytes": archive_bytes,
        },
        "source": {
            "commit": revision,
            "dirty": False,
            "tree": collected.tree_id,
        },
        "seed_assignment": collected.seed_assignment,
        "cohort_assignment": bindings["cohort_assignment"],
        "preregistration": bindings["preregistration"],
        "hardware_amendment": bindings["hardware_amendment"],
        "profile": bindings["profile"],
        "environment": collected.environment,
        "dataset_pointer": bindings["dataset_pointer"],
        "container_base_lock": bindings["container_base_lock"],
        "cohort_assignment_sha256": bindings["cohort_assignment"]["sha256"],
        "preregistration_sha256": bindings["preregistration"]["sha256"],
        "hardware_amendment_sha256": bindings["hardware_amendment"]["sha256"],
        "profile_sha256": bindings["profile"]["sha256"],
        "dataset_pointer_sha256": bindings["dataset_pointer"]["sha256"],
        "container_base_lock_sha256": bindings["container_base_lock"]["sha256"],
        "config_sha256": collected.config_sha256,
        "contract_locks": collected.contract_locks,
        "members_sha256": collected.members_sha256,
    }


def _build_staging(
    staging_fd: int,
    *,
    collected: _Collected,
    revision: str,
    release_id: str,
    archive_name: str,
    profile_path: str,
) -> _core._Staged:
    _core._write_zip_at(
        staging_fd,
        archive_name,
        payload=collected.payload,
        modes=collected.modes,
    )
    archive_fd = _core._open_pinned_regular_at(staging_fd, archive_name)
    try:
        _core._assert_descriptor_names_entry(staging_fd, archive_name, archive_fd)
        archive_hash, archive_bytes = _core._hash_descriptor(archive_fd)
        _core._verify_zip_descriptor(
            archive_fd,
            payload=collected.payload,
            modes=collected.modes,
        )
        release_bytes = _core._canonical_pretty(
            _receipt_value(
                collected=collected,
                revision=revision,
                release_id=release_id,
                archive_name=archive_name,
                archive_sha256=archive_hash,
                archive_bytes=archive_bytes,
                profile_path=profile_path,
            )
        )
        checksum_bytes = f"{archive_hash}  {archive_name}\n".encode("ascii")
        _core._write_new_at(
            staging_fd,
            f"{archive_name}.sha256",
            checksum_bytes,
        )
        _core._write_new_at(
            staging_fd,
            RELEASE_RECEIPT_NAME,
            release_bytes,
        )
        _core._verify_new_file_at(
            staging_fd,
            f"{archive_name}.sha256",
            checksum_bytes,
        )
        _core._verify_new_file_at(
            staging_fd,
            RELEASE_RECEIPT_NAME,
            release_bytes,
        )
        if set(os.listdir(staging_fd)) != {
            archive_name,
            f"{archive_name}.sha256",
            RELEASE_RECEIPT_NAME,
        }:
            raise PackageError("private v3 release staging has unexpected entries")
        _core._assert_descriptor_names_entry(
            staging_fd,
            archive_name,
            archive_fd,
        )
        os.fsync(staging_fd)
        return _core._Staged(
            archive_fd=archive_fd,
            archive_sha256=archive_hash,
            archive_bytes=archive_bytes,
            checksum_bytes=checksum_bytes,
            release_bytes=release_bytes,
        )
    except Exception:
        os.close(archive_fd)
        raise


def _verify_installed(
    directory_fd: int,
    *,
    staged: _core._Staged,
    collected: _Collected,
    archive_name: str,
) -> None:
    if set(os.listdir(directory_fd)) != {
        archive_name,
        f"{archive_name}.sha256",
        RELEASE_RECEIPT_NAME,
    }:
        raise PackageError("installed v3 release has unexpected artifacts")
    _core._assert_descriptor_names_entry(
        directory_fd,
        archive_name,
        staged.archive_fd,
    )
    digest, size = _core._hash_descriptor(staged.archive_fd)
    if digest != staged.archive_sha256 or size != staged.archive_bytes:
        raise PackageError("installed v3 archive identity changed")
    _core._verify_zip_descriptor(
        staged.archive_fd,
        payload=collected.payload,
        modes=collected.modes,
    )
    _core._verify_new_file_at(
        directory_fd,
        f"{archive_name}.sha256",
        staged.checksum_bytes,
    )
    _core._verify_new_file_at(
        directory_fd,
        RELEASE_RECEIPT_NAME,
        staged.release_bytes,
    )
    os.fsync(directory_fd)


def _publish(
    output_fd: int,
    staging_name: str,
    staging_fd: int,
    release_id: str,
    *,
    staged: _core._Staged,
    collected: _Collected,
    archive_name: str,
) -> None:
    _core._assert_output_control(output_fd)
    _core._assert_staging_path(output_fd, staging_name, staging_fd)
    _core._rename_noreplace_at(output_fd, staging_name, release_id)
    try:
        _core._assert_staging_path(output_fd, release_id, staging_fd)
        _verify_installed(
            staging_fd,
            staged=staged,
            collected=collected,
            archive_name=archive_name,
        )
        _core._assert_staging_path(output_fd, release_id, staging_fd)
    except Exception as install_error:
        try:
            _core._quarantine_installed_release(output_fd, release_id)
            os.fsync(output_fd)
        except Exception as quarantine_error:
            raise PackageError(
                "unsafe installed v3 release could not be quarantined"
            ) from quarantine_error
        raise PackageError(
            "installed v3 release identity verification failed"
        ) from install_error


def _profile_relative(source: Path, requested: Path | str) -> str:
    raw = os.fspath(requested)
    if raw in PROFILE_SPECS:
        return raw
    candidate = Path(os.path.abspath(os.fspath(requested)))
    try:
        relative = candidate.relative_to(source).as_posix()
    except ValueError as error:
        raise PackageError(
            "profile must be one of the two closed v3 profile JSONs"
        ) from error
    if relative not in PROFILE_SPECS:
        raise PackageError("profile must be one of the two closed v3 profile JSONs")
    return relative


def build_handoff(
    *,
    source_root: Path | str,
    out_dir: Path | str,
    profile: Path | str,
    apply: bool = False,
) -> ReleaseArtifacts:
    """Validate a clean Git snapshot and optionally publish one v3 release."""

    source = Path(os.path.abspath(os.fspath(source_root)))
    profile_path = _profile_relative(source, profile)
    repository = _core._open_repository(source)
    try:
        output_requested = Path(out_dir)
        if apply:
            _core._assert_external_output(repository, output_requested)
        revision = _core._clean_revision(repository)
        tree_id = _core._commit_tree(repository, revision)
        tracked = _core._tracked_files(repository, tree_id)
        collected = _collect_payload(
            repository,
            tracked,
            revision,
            tree_id,
            profile_path,
        )
        spec = PROFILE_SPECS[profile_path]
        suffix = collected.members_sha256[:16]
        release_id = f"aws-gpu-v3-{spec['slug']}-{suffix}"
        archive_name = f"memorysplit-aws-gpu-v3-{spec['slug']}-{suffix}.zip"
        release_dir = (
            Path(os.path.abspath(os.fspath(output_requested))) / release_id
        )

        if not apply:
            with tempfile.TemporaryDirectory(
                prefix=f".{release_id}.dry-run-"
            ) as temporary:
                staging_fd = os.open(temporary, _core._directory_flags())
                staged: _core._Staged | None = None
                try:
                    staged = _build_staging(
                        staging_fd,
                        collected=collected,
                        revision=revision,
                        release_id=release_id,
                        archive_name=archive_name,
                        profile_path=profile_path,
                    )
                    archive_hash = staged.archive_sha256
                finally:
                    if staged is not None:
                        os.close(staged.archive_fd)
                    os.close(staging_fd)
            return ReleaseArtifacts(
                release_dir=release_dir,
                archive=release_dir / archive_name,
                sha256_file=release_dir / f"{archive_name}.sha256",
                release=release_dir / RELEASE_RECEIPT_NAME,
                release_id=release_id,
                sha256=archive_hash,
                published=False,
            )

        output, output_fd = _core._open_or_create_output(output_requested)
        staging_name: str | None = None
        staging_fd: int | None = None
        staged = None
        published = False
        try:
            _core._lock_output(output_fd)
            staging_name, staging_fd = _core._make_staging_at(
                output_fd,
                release_id,
            )
            staged = _build_staging(
                staging_fd,
                collected=collected,
                revision=revision,
                release_id=release_id,
                archive_name=archive_name,
                profile_path=profile_path,
            )
            _publish(
                output_fd,
                staging_name,
                staging_fd,
                release_id,
                staged=staged,
                collected=collected,
                archive_name=archive_name,
            )
            os.fsync(output_fd)
            published = True
            release_dir = output / release_id
            return ReleaseArtifacts(
                release_dir=release_dir,
                archive=release_dir / archive_name,
                sha256_file=release_dir / f"{archive_name}.sha256",
                release=release_dir / RELEASE_RECEIPT_NAME,
                release_id=release_id,
                sha256=staged.archive_sha256,
                published=True,
            )
        finally:
            if staged is not None:
                os.close(staged.archive_fd)
            if (
                not published
                and staging_name is not None
                and staging_fd is not None
            ):
                _core._remove_staging_at(output_fd, staging_name, staging_fd)
            if staging_fd is not None:
                os.close(staging_fd)
            os.close(output_fd)
    finally:
        repository.close()


def _emit(value: dict[str, object]) -> None:
    sys.stdout.write(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    apply = False
    requested_profile: str | None = None
    try:
        parser = _core._JsonArgumentParser(
            description=(
                "Build one deterministic AWS P5-v3 or P6-v3 MemorySplit handoff."
            )
        )
        parser.add_argument(
            "--profile",
            required=True,
            choices=PROFILE_PATHS,
            help="select exactly one closed v3 profile JSON",
        )
        parser.add_argument(
            "--source-root",
            default=str(Path(__file__).resolve().parents[1]),
        )
        parser.add_argument(
            "--out-dir",
            default=None,
            help=(
                "external release root; defaults to "
                "../memorysplit-releases/aws-gpu-v3 relative to the source"
            ),
        )
        parser.add_argument("--apply", action="store_true")
        args = parser.parse_args(argv)
        apply = bool(args.apply)
        requested_profile = str(args.profile)
        source_root = Path(os.path.abspath(os.fspath(args.source_root)))
        out_dir = (
            Path(args.out_dir)
            if args.out_dir is not None
            else source_root.parent / "memorysplit-releases" / "aws-gpu-v3"
        )
        artifacts = build_handoff(
            source_root=source_root,
            out_dir=out_dir,
            profile=requested_profile,
            apply=apply,
        )
        spec = PROFILE_SPECS[requested_profile]
        report = {
            "schema_version": 1,
            "ok": True,
            "provider": spec["provider"],
            "profile": requested_profile,
            "dry_run": not apply,
            "published": artifacts.published,
            "release_id": artifacts.release_id,
            "release_dir": str(artifacts.release_dir),
            "archive": str(artifacts.archive),
            "sha256_file": str(artifacts.sha256_file),
            "release": str(artifacts.release),
            "sha256": artifacts.sha256,
        }
        code = 0
    except _core._HelpRequested as help_request:
        report = {
            "schema_version": 1,
            "ok": True,
            "dry_run": True,
            "published": False,
            "help": help_request.text,
        }
        code = 0
    except PackageError as error:
        report = {
            "schema_version": 1,
            "ok": False,
            "profile": requested_profile,
            "dry_run": not apply,
            "published": False,
            "error": {"code": error.code, "message": str(error)},
        }
        code = 2
    except Exception:
        report = {
            "schema_version": 1,
            "ok": False,
            "profile": requested_profile,
            "dry_run": not apply,
            "published": False,
            "error": {
                "code": "PACKAGE_INTERNAL_ERROR",
                "message": "unexpected local v3 packaging failure",
            },
        }
        code = 70
    _emit(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
