"""Create hash-bound role manifests only after complete-corpus verification."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from cluster.corpus_contract import sha256_file, verify_dataset_root
from msctl.cohort import (
    COHORT_ID,
    ROLES,
    load_cohort_assignment,
    load_run_config,
    role_config_paths,
)


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_json_no_replace(path: Path | str, value: object) -> Path:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def build_role_manifest(
    role: str,
    *,
    dataset_root: Path | str,
    pointer_path: Path | str,
    source_lock_path: Path | str,
    repository_root: Path | str,
) -> dict[str, Any]:
    """Build one four-cell manifest; corpus verification always happens first."""

    if role not in ROLES:
        raise ValueError(f"unknown frozen role: {role!r}")
    root = Path(repository_root).resolve()
    evidence = verify_dataset_root(
        dataset_root,
        pointer_path=pointer_path,
        source_lock_path=source_lock_path,
    )
    assignment_path = root / "configs" / "cohort-assignment-135m-n10.json"
    load_cohort_assignment(assignment_path)
    configs = []
    for relative in role_config_paths(role):
        path = root / relative
        cfg = load_run_config(path, root=root)
        configs.append(
            {
                "arm": cfg["arm"],
                "config_path": relative,
                "config_sha256": sha256_file(path),
                "pair_id": cfg["pair_id"],
                "run_id": cfg["run_id"],
                "seed": cfg["seed"],
            }
        )
    role_spec = ROLES[role]
    return {
        "cohort_id": COHORT_ID,
        "configs": configs,
        "dataset": {
            "contract_id": evidence.contract_id,
            "lane_ids": list(evidence.lane_ids),
            "ordered_token_stream_sha256": evidence.ordered_token_stream_sha256,
            "raw_target_tokens": evidence.raw_target_tokens,
            "receipt_sha256": evidence.receipt_sha256,
            "semantic_verification_sha256": evidence.semantic_verification_sha256,
            "stream_sha256": dict(evidence.stream_sha256),
        },
        "operator": role,
        "platform": role_spec["platform"],
        "provider": role_spec["provider"],
        "schema_version": 1,
        "seeds": list(role_spec["seeds"]),
    }


def create_role_manifest(
    role: str,
    output: Path | str,
    **kwargs,
) -> Path:
    return write_json_no_replace(output, build_role_manifest(role, **kwargs))
