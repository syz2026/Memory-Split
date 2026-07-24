"""Instantiate the frozen reasoning-v3 cohort on AWS ParallelCluster."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cluster.aws.reasoning_v3 import verify_staged_corpus
from cluster.corpus_contract import sha256_file
from msctl.adapters.slurm import load_pair_manifest
from msctl.manifest import canonical_json_bytes, write_json_no_replace
from msctl.operations import _write_yaml_no_replace
from msctl.profile import load_profile
from msctl.reasoning_cohort import (
    ARMS,
    COHORT_ID,
    COMPOSITE_STREAM_SHA256,
    PROVIDER,
    RAW_TARGETS,
    ROLE,
    ROLES,
    SCIENTIFIC_SCOPE,
    SEEDS,
    config_path,
    load_cohort_assignment,
    load_dataset_pointer,
    load_run_config,
    pair_id,
)


def _runtime_paths(dataset_root: Path, values: list[str]) -> list[str]:
    result = []
    for value in values:
        if not isinstance(value, str) or not value.startswith("dataset/"):
            raise ValueError("reasoning-v3 stream paths must begin with dataset/")
        result.append(str(dataset_root / value.removeprefix("dataset/")))
    return result


def _write_json_no_replace_or_identical(path: Path, value: object) -> Path:
    expected = canonical_json_bytes(value)
    try:
        return write_json_no_replace(path, value)
    except FileExistsError as error:
        if path.is_file() and not path.is_symlink() and path.read_bytes() == expected:
            return path
        raise FileExistsError(f"existing AWS manifest differs: {path}") from error


def _write_yaml_no_replace_or_identical(path: Path, value: object) -> Path:
    expected = yaml.safe_dump(value, sort_keys=False).encode()
    try:
        return _write_yaml_no_replace(path, value)
    except FileExistsError as error:
        if path.is_file() and not path.is_symlink() and path.read_bytes() == expected:
            return path
        raise FileExistsError(f"existing AWS runtime config differs: {path}") from error


def instantiate_aws(
    *,
    dataset_root: Path | str,
    pointer_path: Path | str,
    transfer_manifest_path: Path | str,
    profile_path: Path | str,
    runtime_root: Path | str,
    out_root: Path | str,
    repository_root: Path | str,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    """Verify frozen bytes and write immutable pair manifests for selected seeds."""

    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or tuple(sorted(seeds)) != seeds
        or any(isinstance(seed, bool) or seed not in SEEDS for seed in seeds)
    ):
        raise ValueError("AWS seeds must be a non-empty sorted unique v3 subset")
    root = Path(repository_root).resolve()
    dataset = Path(dataset_root).resolve()
    runtime = Path(runtime_root).resolve()
    outputs = Path(out_root).resolve()
    profile = load_profile(profile_path)
    if profile.platform != "aws":
        raise ValueError("reasoning-v3 AWS instantiation requires an AWS profile")
    pointer = load_dataset_pointer(pointer_path)
    if pointer["transfer_manifest"] != Path(transfer_manifest_path).resolve().relative_to(
        root
    ).as_posix():
        raise ValueError("AWS transfer manifest path differs from the frozen pointer")
    evidence = verify_staged_corpus(dataset, transfer_manifest_path)
    assignment = load_cohort_assignment(
        root / "configs" / "cohort-assignment-135m-v3-aws-n10.json"
    )
    # The role manifest always binds the complete cohort, independent of the
    # selected launch wave. This makes seed-by-seed instantiation additive
    # without weakening no-replace semantics for any existing file.
    configs = []
    for seed in SEEDS:
        for arm in ARMS:
            relative = config_path(arm, seed)
            path = root / relative
            cfg = load_run_config(path, root=root)
            configs.append(
                {
                    "arm": arm,
                    "config_path": relative,
                    "config_sha256": sha256_file(path),
                    "pair_id": cfg["pair_id"],
                    "run_id": cfg["run_id"],
                    "seed": seed,
                }
            )
    role_manifest = {
        "cohort_id": COHORT_ID,
        "configs": configs,
        "dataset": {
            "composite_stream_sha256": dict(COMPOSITE_STREAM_SHA256),
            "contract_id": pointer["contract_id"],
            "manifest_sha256": evidence.manifest_sha256,
            "ordered_token_stream_sha256": COMPOSITE_STREAM_SHA256["packed_targets"],
            "raw_target_tokens": RAW_TARGETS,
            "receipt_sha256": evidence.virtual_receipt_sha256,
            "scientific_scope": SCIENTIFIC_SCOPE,
        },
        "operator": ROLE,
        "platform": ROLES[ROLE]["platform"],
        "provider": PROVIDER,
        "schema_version": 1,
        "seeds": list(SEEDS),
    }
    if assignment["scientific_scope"] != SCIENTIFIC_SCOPE:
        raise ValueError("AWS cohort assignment scientific scope differs")
    role_manifest_path = _write_json_no_replace_or_identical(
        runtime / "role-manifest.json",
        role_manifest,
    )
    pair_paths = []
    for seed in seeds:
        arm_records = []
        for arm in ARMS:
            relative = config_path(arm, seed)
            static_path = root / relative
            cfg = load_run_config(static_path, root=root)
            runtime_cfg = dict(cfg)
            runtime_cfg["train_bin"] = _runtime_paths(dataset, cfg["train_bin"])
            runtime_cfg["train_mask"] = _runtime_paths(dataset, cfg["train_mask"])
            runtime_cfg["out_dir"] = str(outputs / cfg["run_id"])
            runtime_cfg["dataset_receipt_sha256"] = evidence.virtual_receipt_sha256
            runtime_cfg["ordered_token_stream_sha256"] = (
                COMPOSITE_STREAM_SHA256["packed_targets"]
            )
            runtime_cfg["transfer_manifest_sha256"] = evidence.manifest_sha256
            runtime_cfg["profile_sha256"] = profile.sha256
            runtime_path = runtime / "configs" / f"{cfg['run_id']}.yaml"
            _write_yaml_no_replace_or_identical(runtime_path, runtime_cfg)
            arm_records.append(
                {
                    "arm": arm,
                    "config_path": relative,
                    "config_sha256": sha256_file(static_path),
                    "out_dir": runtime_cfg["out_dir"],
                    "run_id": cfg["run_id"],
                    "runtime_config": str(runtime_path),
                    "runtime_config_sha256": sha256_file(runtime_path),
                }
            )
        pair = {
            "arms": arm_records,
            "cohort_id": COHORT_ID,
            "dataset": {
                "ordered_token_stream_sha256": COMPOSITE_STREAM_SHA256[
                    "packed_targets"
                ],
                "receipt_sha256": evidence.virtual_receipt_sha256,
            },
            "operator": ROLE,
            "pair_id": pair_id(seed),
            "profile_sha256": profile.sha256,
            "provider": PROVIDER,
            "schema_version": 1,
            "seed": seed,
        }
        pair_path = _write_json_no_replace_or_identical(
            runtime / "pairs" / f"pair-s{seed}.json",
            pair,
        )
        load_pair_manifest(pair_path)
        pair_paths.append(pair_path)
    return {
        "dataset_manifest_sha256": evidence.manifest_sha256,
        "dataset_receipt_sha256": evidence.virtual_receipt_sha256,
        "pair_manifests": pair_paths,
        "profile_sha256": profile.sha256,
        "role_manifest": role_manifest_path,
    }
