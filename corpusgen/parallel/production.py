"""Fail-closed production adapter for the frozen MemorySplit v2 corpus.

The adapter consumes already materialized, hash-locked lane token streams.  It
does not synthesize absent sources, infer revisions, or cycle reasoning data.
The source bundle is deliberately separate from the repository because the
authoritative recipe still has unfrozen upstream artifact bindings.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
from array import array
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Self

from corpusgen.reasoning.proofs import (
    SOLVER_ID,
    CompositionPremise,
    EqualityPremise,
    GraphTraversalPremise,
    solve_graph_composition,
    solve_graph_traversal,
    solve_slot_equality,
)

from .canonical import canonical_json_bytes, sha256_hex
from .catalog import CatalogRecord, InputCatalog
from .metadata import (
    MetadataRecord,
    RenderedRecord,
    metadata_from_bytes,
    render_metadata,
)
from .publication import (
    ParallelBuildConfig,
    VerifiedParallelCorpus,
    build_parallel_corpus,
    verify_parallel_corpus,
)
from .schedule import largest_deficit_schedule

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECIPE_PATH = _ROOT / "configs" / "reasoning-dataset-v2.json"
DEFAULT_SOURCE_MANIFEST = "source-manifest.json"

PRODUCTION_SOURCE_FORMAT = "memorysplit-v2-production-sources-v1"
PRODUCTION_LOCK_FORMAT = "memorysplit-v2-source-lock-v1"
HUGGINGFACE_SOURCE_LOCK_FORMAT = "memorysplit-v2-huggingface-source-lock"
OBJECTIVE_SOURCE_LOCK_FORMAT = "memorysplit-v2-objective-auxiliary-lock"
PRODUCTION_SLICE_FORMAT = "memorysplit-v2-production-slice-v1"
PRODUCTION_RENDERER_VERSION = "memorysplit-v2-production-u16-v1"

FROZEN_LANES = (
    "fineweb_edu",
    "finemath",
    "wikidata_graph",
    "synthetic_graph",
    "verified_synthetic_multihop",
    "wikidata_path_reasoning",
    "relational_refinement",
    "objective_auxiliary",
)
FROZEN_TOKEN_QUOTAS = (
    ("fineweb_edu", 1_780_219_904),
    ("finemath", 1_068_131_943),
    ("wikidata_graph", 1_424_175_923),
    ("synthetic_graph", 712_087_962),
    ("verified_synthetic_multihop", 1_068_131_942),
    ("wikidata_path_reasoning", 534_065_971),
    ("relational_refinement", 178_021_990),
    ("objective_auxiliary", 356_043_981),
)
FROZEN_TOTAL_TOKENS = 7_120_879_616
FROZEN_UPDATE_TOKENS = 524_288
FROZEN_OPTIMIZER_STEPS = 13_582
REASONING_LANES = frozenset(
    {
        "verified_synthetic_multihop",
        "wikidata_path_reasoning",
        "relational_refinement",
    }
)
OBJECTIVE_LANE = "objective_auxiliary"
FROZEN_OBJECTIVE_SOURCE_IDS = (
    "deepmind_mathematics_generator",
    "clrs_text",
    "ruletaker",
    "prontoqa",
    "reasoning_gym_exact_answer",
    "arc_agi_training",
    "conceptarc_training",
)

# These names describe the evidence a real source bundle must carry.  A lock is
# a hash-committed provenance document, not a claim that this repository
# currently contains the corresponding data.
FROZEN_REQUIRED_SOURCE_LOCKS = (
    ("fineweb_edu", ("fineweb_edu",)),
    ("finemath", ("finemath",)),
    ("wikidata_graph", ("wikidata5m",)),
    ("synthetic_graph", ("synthetic_graph_generator",)),
    (
        "verified_synthetic_multihop",
        ("verified_synthetic_multihop_generator", "reasoning_solver"),
    ),
    (
        "wikidata_path_reasoning",
        ("wikidata5m", "wikidata_path_reasoning_generator", "reasoning_solver"),
    ),
    (
        "relational_refinement",
        ("relational_refinement_generator", "reasoning_solver"),
    ),
    (
        "objective_auxiliary",
        ("objective_auxiliary",),
    ),
)

_HEX_40_TO_64 = re.compile(r"[0-9a-f]{40,64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ARTIFACT_FIELDS = {"bytes", "path", "sha256"}
_LOCK_DESCRIPTOR_FIELDS = {"bytes", "id", "path", "sha256"}
_LANE_FIELDS = {
    "cycle_fill",
    "id",
    "mask_ledger",
    "route_ledger",
    "source_locks",
    "split90_target_weights",
    "tokens",
    "verification_ledger",
}
_SOURCE_MANIFEST_FIELDS = {
    "contract_id",
    "format",
    "lanes",
    "recipe_sha256",
    "sidecar_dtype",
    "token_dtype",
}
_SLICE_FIELDS = {
    "format",
    "lane",
    "mask_ledger_sha256",
    "route_ledger_sha256",
    "source_lock_sha256s",
    "source_manifest_sha256",
    "split90_sha256",
    "token_count",
    "token_offset",
    "tokens_sha256",
    "verification_ledger_sha256",
}
_ROUTE_FIELDS = {"burden_bits", "external", "fact_id"}
_BURDEN_FIELDS = {"denominator", "numerator"}
_MASK_FIELDS = {"end", "fact_id", "start"}
_VERIFICATION_ROW_FIELDS = {
    "record_id",
    "source_id",
    "token_end",
    "token_start",
    "verification",
}


class ProductionPreflightError(RuntimeError):
    """A source bundle cannot satisfy the frozen production contract."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        issues = self.report.get("issues", [])
        detail = "; ".join(
            str(issue.get("message", "production preflight failed"))
            for issue in issues[:4]
        )
        if len(issues) > 4:
            detail += f"; and {len(issues) - 4} more issue(s)"
        super().__init__(detail or "production preflight failed")


@dataclass(frozen=True)
class ProductionRecipe:
    contract_id: str
    recipe_sha256: str
    lanes: tuple[str, ...]
    token_quotas: tuple[tuple[str, int], ...]
    total_tokens: int
    update_tokens: int
    optimizer_steps: int
    required_source_locks: tuple[tuple[str, tuple[str, ...]], ...]
    reasoning_lanes: frozenset[str]
    objective_lane: str

    def __post_init__(self) -> None:
        _required_text(self.contract_id, label="production contract_id")
        _digest(self.recipe_sha256, label="production recipe_sha256")
        if self.lanes != FROZEN_LANES:
            raise ValueError("production recipe must retain the eight frozen lanes")
        if (
            not isinstance(self.token_quotas, tuple)
            or tuple(lane for lane, _quota in self.token_quotas) != self.lanes
        ):
            raise ValueError("production token quotas must follow frozen lane order")
        for lane, quota in self.token_quotas:
            _required_text(lane, label="production quota lane")
            _positive_int(quota, label=f"{lane} production token quota")
        if _positive_int(self.total_tokens, label="production total_tokens") != sum(
            quota for _lane, quota in self.token_quotas
        ):
            raise ValueError("production token quotas do not sum to total_tokens")
        update_tokens = _positive_int(
            self.update_tokens, label="production update_tokens"
        )
        optimizer_steps = _positive_int(
            self.optimizer_steps, label="production optimizer_steps"
        )
        if update_tokens * optimizer_steps != self.total_tokens:
            raise ValueError(
                "production total_tokens must be an exact optimizer-step budget"
            )
        if (
            not isinstance(self.required_source_locks, tuple)
            or tuple(lane for lane, _locks in self.required_source_locks) != self.lanes
        ):
            raise ValueError(
                "production source locks must follow the frozen lane order"
            )
        for lane, locks in self.required_source_locks:
            if (
                not isinstance(locks, tuple)
                or not locks
                or len(locks) != len(set(locks))
            ):
                raise ValueError(f"{lane} source locks must be a non-empty tuple")
            for source_id in locks:
                _required_text(source_id, label=f"{lane} source lock id")
        if (
            not isinstance(self.reasoning_lanes, frozenset)
            or not self.reasoning_lanes <= frozenset(self.lanes)
            or self.objective_lane not in self.lanes
            or self.objective_lane in self.reasoning_lanes
        ):
            raise ValueError("production reasoning/objective lane identity drift")

    @property
    def quota_by_lane(self) -> dict[str, int]:
        return dict(self.token_quotas)

    @property
    def locks_by_lane(self) -> dict[str, tuple[str, ...]]:
        return dict(self.required_source_locks)

    @classmethod
    def for_testing(
        cls,
        token_quotas: Mapping[str, int],
        *,
        update_tokens: int,
        required_source_locks: Mapping[str, tuple[str, ...]],
        reasoning_lanes: frozenset[str] = REASONING_LANES,
        objective_lane: str = OBJECTIVE_LANE,
    ) -> ProductionRecipe:
        lanes = tuple(token_quotas)
        if lanes != FROZEN_LANES:
            raise ValueError("test recipe must retain the eight authoritative lanes")
        total = sum(token_quotas.values())
        if total % update_tokens:
            raise ValueError("test recipe total must be divisible by update_tokens")
        return cls(
            contract_id="memorysplit-reasoning-dataset-v2",
            recipe_sha256=sha256_hex(
                canonical_json_bytes(
                    {
                        "testing": True,
                        "token_quotas": dict(token_quotas),
                        "update_tokens": update_tokens,
                    }
                )
            ),
            lanes=lanes,
            token_quotas=tuple(token_quotas.items()),
            total_tokens=total,
            update_tokens=update_tokens,
            optimizer_steps=total // update_tokens,
            required_source_locks=tuple(required_source_locks.items()),
            reasoning_lanes=reasoning_lanes,
            objective_lane=objective_lane,
        )


@dataclass(frozen=True)
class ProductionSourceManifest:
    root: Path
    path: Path
    value: dict[str, Any]
    sha256: str

    @property
    def lanes(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.value["lanes"])

    @property
    def lane_by_id(self) -> dict[str, dict[str, Any]]:
        return {lane["id"]: lane for lane in self.lanes}


def _issue(
    code: str,
    message: str,
    *,
    action: str,
    lane: str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "action": action,
        "code": code,
        "message": message,
    }
    if lane is not None:
        value["lane"] = lane
    if path is not None:
        value["path"] = path
    return value


def _strict_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON value {value}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _required_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _relative_path(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or value.startswith("~")
        or "$" in value
    ):
        raise ValueError(f"{label} must be a portable relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a portable relative POSIX path")
    return path.as_posix()


def _safe_path(root: Path, relative: str, *, label: str) -> Path:
    _relative_path(relative, label=label)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"source root is missing or unsafe: {root}")
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as error:
            raise ValueError(f"{label} is missing: {relative}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} crosses a symlink: {relative}")
    if not current.is_file():
        raise ValueError(f"{label} is not a regular file: {relative}")
    return current


def _size_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"source artifact is not regular: {path}")
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(handle.fileno())
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ) or size != after.st_size:
        raise ValueError(f"source artifact changed while hashing: {path}")
    return size, digest.hexdigest()


def _artifact_descriptor(root: Path, relative: str) -> dict[str, Any]:
    path = _safe_path(root, relative, label="source artifact")
    size, digest = _size_sha256(path)
    return {"bytes": size, "path": relative, "sha256": digest}


def _validate_artifact_descriptor(
    value: object,
    *,
    label: str,
    allow_empty: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _ARTIFACT_FIELDS:
        raise ValueError(f"{label} fields do not match the contract")
    byte_count = _nonnegative_int(value["bytes"], label=f"{label} bytes")
    if not allow_empty and byte_count == 0:
        raise ValueError(f"{label} must not be empty")
    return {
        "bytes": byte_count,
        "path": _relative_path(value["path"], label=f"{label} path"),
        "sha256": _digest(value["sha256"], label=f"{label} sha256"),
    }


def _verify_artifact(
    root: Path,
    descriptor: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    path = _safe_path(root, str(descriptor["path"]), label=label)
    size, digest = _size_sha256(path)
    if size != descriptor["bytes"] or digest != descriptor["sha256"]:
        raise ValueError(
            f"{label} digest drift: expected {descriptor['sha256']}, got {digest}"
        )
    return path


def load_production_recipe(
    path: Path | str = DEFAULT_RECIPE_PATH,
) -> ProductionRecipe:
    recipe_path = Path(path)
    raw = recipe_path.read_bytes()
    value = _strict_object(raw, label="MemorySplit v2 recipe")
    if value.get("schema_version") != 2:
        raise ValueError("MemorySplit v2 recipe schema_version must be 2")
    if value.get("contract_id") != "memorysplit-reasoning-dataset-v2":
        raise ValueError("unexpected MemorySplit v2 contract_id")
    sprint = value.get("sprint_recipe")
    if not isinstance(sprint, dict):
        raise TypeError("MemorySplit v2 sprint_recipe must be an object")
    lanes = sprint.get("lanes")
    if not isinstance(lanes, list):
        raise TypeError("MemorySplit v2 lanes must be a list")
    lane_ids = tuple(
        _required_text(lane.get("id"), label="recipe lane id")
        if isinstance(lane, dict)
        else ""
        for lane in lanes
    )
    allocation = sprint.get("realized_token_allocation")
    if not isinstance(allocation, dict):
        raise TypeError("frozen realized_token_allocation is missing")
    raw_quotas = allocation.get("token_quotas")
    if not isinstance(raw_quotas, dict):
        raise TypeError("frozen token_quotas are missing")
    quotas = tuple((lane, raw_quotas.get(lane)) for lane in FROZEN_LANES)
    publication = sprint.get("publication_requirements")
    expected_publication = {
        "complete_once_wikidata_training_graph": True,
        "stable_fact_universe_and_exposure_burden": True,
        "solver_verification_rate": 1.0,
        "structural_train_evaluation_overlap_max": 0.0,
        "deterministic_rebuild_required": True,
        "reasoning_lane_cycle_fill_forbidden": True,
    }
    if (
        lane_ids != FROZEN_LANES
        or tuple(allocation.get("lane_order", ())) != FROZEN_LANES
        or allocation.get("method") != "hamilton_largest_remainder"
        or allocation.get("tie_break") != "stable_lane_order"
        or tuple(quotas) != FROZEN_TOKEN_QUOTAS
        or allocation.get("total_tokens") != FROZEN_TOTAL_TOKENS
        or sprint.get("raw_target_tokens") != FROZEN_TOTAL_TOKENS
        or sprint.get("targets_per_update") != FROZEN_UPDATE_TOKENS
        or sprint.get("optimizer_steps") != FROZEN_OPTIMIZER_STEPS
        or publication != expected_publication
    ):
        raise ValueError(
            "authoritative MemorySplit v2 lane, quota, update, or publication "
            "contract drifted"
        )
    if sum(dict(FROZEN_TOKEN_QUOTAS).values()) != FROZEN_TOTAL_TOKENS:
        raise AssertionError("frozen token quota constant drift")
    if FROZEN_UPDATE_TOKENS * FROZEN_OPTIMIZER_STEPS != FROZEN_TOTAL_TOKENS:
        raise AssertionError("frozen optimizer budget constant drift")
    policy = sprint.get("source_policy")
    if not isinstance(policy, dict):
        raise TypeError("frozen source_policy is missing")
    if (
        policy.get("finemath_repository") != "HuggingFaceTB/finemath"
        or policy.get("finemath_subsets_in_order")
        != [
            "finemath-4plus",
            "finemath-3plus-cross-deduplicated-remainder",
        ]
        or policy.get("cross_deduplicate_finemath_against_fineweb") is not True
        or tuple(policy.get("objective_auxiliary_sources", ()))
        != FROZEN_OBJECTIVE_SOURCE_IDS
    ):
        raise ValueError("frozen production source policy drifted")
    return ProductionRecipe(
        contract_id=value["contract_id"],
        recipe_sha256=sha256_hex(raw),
        lanes=FROZEN_LANES,
        token_quotas=FROZEN_TOKEN_QUOTAS,
        total_tokens=FROZEN_TOTAL_TOKENS,
        update_tokens=FROZEN_UPDATE_TOKENS,
        optimizer_steps=FROZEN_OPTIMIZER_STEPS,
        required_source_locks=FROZEN_REQUIRED_SOURCE_LOCKS,
        reasoning_lanes=REASONING_LANES,
        objective_lane=OBJECTIVE_LANE,
    )


def _validate_source_manifest_value(
    value: dict[str, Any],
    *,
    recipe: ProductionRecipe,
) -> None:
    if set(value) != _SOURCE_MANIFEST_FIELDS:
        raise ValueError("production source manifest fields do not match")
    if (
        value["format"] != PRODUCTION_SOURCE_FORMAT
        or value["contract_id"] != recipe.contract_id
        or value["recipe_sha256"] != recipe.recipe_sha256
        or value["token_dtype"] != "uint16_le"
        or value["sidecar_dtype"] != "uint8"
    ):
        raise ValueError("production source manifest identity mismatch")
    lanes = value["lanes"]
    if not isinstance(lanes, list) or [
        lane.get("id") if isinstance(lane, dict) else None for lane in lanes
    ] != list(recipe.lanes):
        raise ValueError("production source manifest lanes are missing or reordered")
    for lane in lanes:
        if not isinstance(lane, dict) or set(lane) != _LANE_FIELDS:
            raise ValueError("production source lane fields do not match")
        lane_id = lane["id"]
        if lane["cycle_fill"] is not False:
            raise ValueError(f"production lane enables cycle fill: {lane_id}")
        _validate_artifact_descriptor(
            lane["tokens"], label=f"{lane_id} tokens", allow_empty=False
        )
        _validate_artifact_descriptor(
            lane["split90_target_weights"],
            label=f"{lane_id} Split90 weights",
            allow_empty=False,
        )
        _validate_artifact_descriptor(
            lane["route_ledger"],
            label=f"{lane_id} route ledger",
            allow_empty=True,
        )
        _validate_artifact_descriptor(
            lane["mask_ledger"],
            label=f"{lane_id} mask ledger",
            allow_empty=True,
        )
        verification = lane["verification_ledger"]
        required = lane_id in recipe.reasoning_lanes or lane_id == recipe.objective_lane
        if required:
            _validate_artifact_descriptor(
                verification,
                label=f"{lane_id} verification ledger",
                allow_empty=False,
            )
        elif verification is not None:
            raise ValueError(
                f"{lane_id} must not claim an unrequired verification ledger"
            )
        locks = lane["source_locks"]
        required_locks = recipe.locks_by_lane[lane_id]
        if not isinstance(locks, list) or [
            lock.get("id") if isinstance(lock, dict) else None for lock in locks
        ] != list(required_locks):
            raise ValueError(
                f"{lane_id} source locks must be exactly {list(required_locks)}"
            )
        for lock in locks:
            if not isinstance(lock, dict) or set(lock) != _LOCK_DESCRIPTOR_FIELDS:
                raise ValueError(f"{lane_id} source lock descriptor fields drifted")
            _required_text(lock["id"], label=f"{lane_id} source lock id")
            _relative_path(lock["path"], label=f"{lane_id} source lock path")
            _positive_int(lock["bytes"], label=f"{lane_id} source lock bytes")
            _digest(lock["sha256"], label=f"{lane_id} source lock sha256")


def load_production_source_manifest(
    source_root: Path | str,
    *,
    recipe: ProductionRecipe,
    manifest_path: Path | str | None = None,
) -> ProductionSourceManifest:
    root = Path(source_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"production source root is missing or unsafe: {root}")
    root = root.resolve()
    path = (
        Path(manifest_path)
        if manifest_path is not None
        else root / DEFAULT_SOURCE_MANIFEST
    )
    if not path.is_absolute():
        path = root / path
    if path.parent != root:
        raise ValueError(
            "production source manifest must be a direct child of source_root"
        )
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"production source manifest is missing or unsafe: {path}")
    raw = path.read_bytes()
    value = _strict_object(raw, label="production source manifest")
    if raw != canonical_json_bytes(value):
        raise ValueError("production source manifest is not canonical JSON")
    _validate_source_manifest_value(value, recipe=recipe)
    return ProductionSourceManifest(
        root=root,
        path=path,
        value=value,
        sha256=sha256_hex(raw),
    )


def expected_production_source_paths(
    recipe: ProductionRecipe,
) -> tuple[str, ...]:
    paths = []
    for lane in recipe.lanes:
        paths.extend(
            (
                f"materialized/{lane}.tokens.bin",
                f"materialized/{lane}.split90.weights.bin",
                f"ledgers/{lane}.routes.jsonl",
                f"ledgers/{lane}.masks.jsonl",
            )
        )
        if lane in recipe.reasoning_lanes or lane == recipe.objective_lane:
            paths.append(f"ledgers/{lane}.verification.jsonl")
    lock_ids = {
        lock_id
        for _lane, lane_locks in recipe.required_source_locks
        for lock_id in lane_locks
    }
    paths.extend(f"locks/{lock_id}.lock.json" for lock_id in sorted(lock_ids))
    return tuple(paths)


def _generic_lock_artifacts(
    value: object,
    *,
    source_id: str,
    allow_empty: bool = False,
) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{source_id} lock artifacts must be a non-empty list")
    paths = []
    for artifact in value:
        if not isinstance(artifact, dict) or set(artifact) != _ARTIFACT_FIELDS:
            raise ValueError(f"{source_id} lock artifact fields do not match")
        paths.append(
            _relative_path(artifact["path"], label=f"{source_id} locked artifact path")
        )
        byte_validator = _nonnegative_int if allow_empty else _positive_int
        byte_validator(artifact["bytes"], label=f"{source_id} locked artifact bytes")
        _digest(artifact["sha256"], label=f"{source_id} locked artifact sha256")
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError(f"{source_id} lock artifacts must be uniquely path-sorted")


def _validate_huggingface_files(
    value: object,
    *,
    source_id: str,
    with_members: bool,
) -> tuple[int, tuple[str, ...]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{source_id} source lock files must be non-empty")
    paths = []
    total_bytes = 0
    expected_fields = {
        "bytes",
        "git_blob_sha1",
        "path",
        "sha256",
        *(("members",) if with_members else ()),
    }
    for file_record in value:
        if not isinstance(file_record, dict) or set(file_record) != expected_fields:
            raise ValueError(f"{source_id} source file fields do not match")
        path = _relative_path(
            file_record["path"], label=f"{source_id} source file path"
        )
        byte_count = _positive_int(
            file_record["bytes"], label=f"{source_id} source file bytes"
        )
        _digest(file_record["sha256"], label=f"{source_id} source file sha256")
        if (
            not isinstance(file_record["git_blob_sha1"], str)
            or re.fullmatch(r"[0-9a-f]{40}", file_record["git_blob_sha1"]) is None
        ):
            raise ValueError(f"{source_id} source file git_blob_sha1 is invalid")
        if with_members:
            members = file_record["members"]
            if not isinstance(members, list) or not members:
                raise ValueError(f"{source_id} archive members must be non-empty")
            member_paths = []
            for member in members:
                if not isinstance(member, dict) or set(member) != {
                    "bytes",
                    "path",
                    "sha256",
                }:
                    raise ValueError(f"{source_id} archive member fields do not match")
                member_paths.append(
                    _relative_path(
                        member["path"],
                        label=f"{source_id} archive member path",
                    )
                )
                _positive_int(
                    member["bytes"],
                    label=f"{source_id} archive member bytes",
                )
                _digest(
                    member["sha256"],
                    label=f"{source_id} archive member sha256",
                )
            if member_paths != sorted(member_paths) or len(member_paths) != len(
                set(member_paths)
            ):
                raise ValueError(
                    f"{source_id} archive members must be uniquely path-sorted"
                )
        paths.append(path)
        total_bytes += byte_count
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError(f"{source_id} source files must be uniquely path-sorted")
    return total_bytes, tuple(paths)


def _validate_huggingface_source_lock(
    source_id: str,
    value: Mapping[str, Any],
) -> None:
    common_fields = {
        "files",
        "format",
        "license",
        "license_evidence",
        "repo_type",
        "repository",
        "revision",
        "schema_version",
        "source_id",
        "total_bytes",
    }
    extra_fields = {
        "fineweb_edu": {"selection"},
        "finemath": {"deduplication", "selection", "subset_metadata"},
        "wikidata5m": {"complete_once"},
    }[source_id]
    if set(value) != common_fields | extra_fields:
        raise ValueError(f"{source_id} Hugging Face lock fields do not match")
    repositories = {
        "fineweb_edu": "HuggingFaceFW/fineweb-edu",
        "finemath": "HuggingFaceTB/finemath",
        "wikidata5m": "intfloat/wikidata5m",
    }
    if (
        value["format"] != HUGGINGFACE_SOURCE_LOCK_FORMAT
        or value["schema_version"] != 1
        or value["source_id"] != source_id
        or value["repository"] != repositories[source_id]
        or value["repo_type"] != "dataset"
        or not isinstance(value["revision"], str)
        or _HEX_40_TO_64.fullmatch(value["revision"]) is None
        or not isinstance(value["license_evidence"], dict)
    ):
        raise ValueError(f"{source_id} Hugging Face lock identity is invalid")
    _required_text(value["license"], label=f"{source_id} source license")
    total_bytes, paths = _validate_huggingface_files(
        value["files"],
        source_id=source_id,
        with_members=source_id == "wikidata5m",
    )
    if value["total_bytes"] != total_bytes:
        raise ValueError(f"{source_id} source lock total_bytes drifted")

    if source_id == "fineweb_edu":
        if (
            value["selection"]
            != {
                "config": "sample-10BT",
                "file_order": "lexicographic_path",
                "scope": "complete_config",
            }
            or len(paths) != 14
            or any(not path.startswith("sample/10BT/") for path in paths)
        ):
            raise ValueError(
                "FineWeb lock must bind the complete expanded sample-10BT config"
            )
    elif source_id == "finemath":
        expected_deduplication = {
            "digest": "sha256",
            "encoding": "utf-8",
            "field": "text",
            "identity": "exact_text_bytes",
            "priority": [
                "fineweb_edu",
                "finemath-4plus",
                "finemath-3plus",
            ],
            "sidecar_encoding": ("one_unsigned_byte_per_source_row_1_keep_0_drop"),
        }
        if (
            value["selection"]
            != {
                "file_order": "lexicographic_path_within_subset",
                "subsets_in_order": [
                    "finemath-4plus",
                    "finemath-3plus-cross-deduplicated-remainder",
                ],
            }
            or value["deduplication"] != expected_deduplication
            or len(paths) != 192
            or sum(path.startswith("finemath-4plus/") for path in paths) != 64
            or sum(path.startswith("finemath-3plus/") for path in paths) != 128
            or not isinstance(value["subset_metadata"], dict)
            or set(value["subset_metadata"]) != {"finemath-3plus", "finemath-4plus"}
        ):
            raise ValueError(
                "FineMath lock does not implement the frozen source policy"
            )
        for subset, metadata in value["subset_metadata"].items():
            if not isinstance(metadata, dict) or set(metadata) != {
                "download_bytes",
                "rows",
            }:
                raise ValueError(f"FineMath {subset} metadata fields drifted")
            _positive_int(
                metadata["download_bytes"],
                label=f"FineMath {subset} download_bytes",
            )
            _positive_int(metadata["rows"], label=f"FineMath {subset} rows")
    else:
        expected_paths = (
            "wikidata5m_alias.tar.gz",
            "wikidata5m_inductive.tar.gz",
            "wikidata5m_transductive.tar.gz",
        )
        if paths != expected_paths or value["complete_once"] != {
            "sidecar_encoding": ("one_unsigned_byte_per_source_row_1_keep_0_drop"),
            "split_order": [
                "wikidata5m_inductive_train.txt",
                "wikidata5m_transductive_train.txt",
            ],
            "triple_identity": "exact_subject_relation_object",
        }:
            raise ValueError(
                "Wikidata5M lock does not bind the frozen complete-once policy"
            )


def _validate_objective_source_lock(value: Mapping[str, Any]) -> None:
    if (
        set(value) != {"format", "schema_version", "sources"}
        or value["format"] != OBJECTIVE_SOURCE_LOCK_FORMAT
        or value["schema_version"] != 1
        or not isinstance(value["sources"], list)
        or [
            source.get("id") if isinstance(source, dict) else None
            for source in value["sources"]
        ]
        != list(FROZEN_OBJECTIVE_SOURCE_IDS)
    ):
        raise ValueError("objective auxiliary source lock identity is invalid")
    for source in value["sources"]:
        if not isinstance(source, dict) or set(source) != {
            "components",
            "id",
            "training_use",
        }:
            raise ValueError("objective auxiliary source fields do not match")
        source_id = _required_text(source["id"], label="objective auxiliary source id")
        _required_text(
            source["training_use"],
            label=f"{source_id} objective training_use",
        )
        components = source["components"]
        if not isinstance(components, list) or not components:
            raise ValueError(f"{source_id} objective components must be non-empty")
        component_ids = []
        for component in components:
            if not isinstance(component, dict) or set(component) != {
                "archive",
                "commit",
                "files",
                "id",
                "license",
                "license_paths",
                "repository",
            }:
                raise ValueError(f"{source_id} component fields do not match")
            component_id = _required_text(
                component["id"], label=f"{source_id} component id"
            )
            component_ids.append(component_id)
            _required_text(
                component["repository"],
                label=f"{component_id} component repository",
            )
            _required_text(
                component["license"],
                label=f"{component_id} component license",
            )
            if (
                not isinstance(component["commit"], str)
                or re.fullmatch(r"[0-9a-f]{40}", component["commit"]) is None
            ):
                raise ValueError(f"{component_id} component commit is not immutable")
            archive = component["archive"]
            if not isinstance(archive, dict) or set(archive) != {
                "bytes",
                "root",
                "sha256",
                "url",
            }:
                raise ValueError(f"{component_id} archive fields do not match")
            _positive_int(archive["bytes"], label=f"{component_id} archive bytes")
            _required_text(archive["root"], label=f"{component_id} archive root")
            _digest(archive["sha256"], label=f"{component_id} archive sha256")
            _required_text(archive["url"], label=f"{component_id} archive URL")
            files = component["files"]
            _generic_lock_artifacts(
                files,
                source_id=component_id,
                allow_empty=True,
            )
            file_paths = {file_record["path"] for file_record in files}
            license_paths = component["license_paths"]
            if (
                not isinstance(license_paths, list)
                or not license_paths
                or len(license_paths) != len(set(license_paths))
            ):
                raise ValueError(
                    f"{component_id} license_paths must be non-empty and unique"
                )
            for license_path in license_paths:
                relative = _relative_path(
                    license_path,
                    label=f"{component_id} license path",
                )
                if relative not in file_paths:
                    raise ValueError(
                        f"{component_id} locked license file is missing: {relative}"
                    )
        if len(component_ids) != len(set(component_ids)):
            raise ValueError(f"{source_id} repeats an objective component")


def _validate_source_lock(source_id: str, payload: bytes) -> None:
    value = _strict_object(payload, label=f"{source_id} source lock")
    if source_id == "wikidata5m" and set(value) == {
        "files",
        "repo_id",
        "repo_type",
        "revision",
    }:
        if (
            value["repo_id"] != "intfloat/wikidata5m"
            or value["repo_type"] != "dataset"
            or not isinstance(value["revision"], str)
            or _HEX_40_TO_64.fullmatch(value["revision"]) is None
            or not isinstance(value["files"], dict)
            or tuple(sorted(value["files"]))
            != (
                "wikidata5m_alias.tar.gz",
                "wikidata5m_inductive.tar.gz",
                "wikidata5m_transductive.tar.gz",
            )
        ):
            raise ValueError("Wikidata5M legacy lock identity is invalid")
        for name, artifact in value["files"].items():
            _relative_path(name, label="Wikidata5M archive path")
            if not isinstance(artifact, dict) or set(artifact) != {
                "bytes",
                "sha256",
            }:
                raise ValueError("Wikidata5M archive lock fields do not match")
            _positive_int(artifact["bytes"], label="Wikidata5M archive bytes")
            _digest(artifact["sha256"], label="Wikidata5M archive sha256")
        return

    if source_id in {"fineweb_edu", "finemath", "wikidata5m"}:
        _validate_huggingface_source_lock(source_id, value)
        return
    if source_id == "objective_auxiliary":
        _validate_objective_source_lock(value)
        return

    expected = {
        "artifacts",
        "format",
        "kind",
        "policy",
        "repository",
        "revision",
        "source_id",
    }
    if set(value) != expected or value["format"] != PRODUCTION_LOCK_FORMAT:
        raise ValueError(
            f"{source_id} requires a {PRODUCTION_LOCK_FORMAT} provenance document"
        )
    if value["source_id"] != source_id:
        raise ValueError(f"{source_id} lock source_id mismatch")
    if value["kind"] not in {"dataset", "generator", "git", "solver"}:
        raise ValueError(f"{source_id} lock kind is not immutable-source capable")
    _required_text(value["repository"], label=f"{source_id} lock repository")
    revision = _required_text(value["revision"], label=f"{source_id} lock revision")
    if _HEX_40_TO_64.fullmatch(revision) is None:
        raise ValueError(f"{source_id} lock revision must be an immutable hex digest")
    if not isinstance(value["policy"], dict):
        raise TypeError(f"{source_id} lock policy must be an object")
    _generic_lock_artifacts(value["artifacts"], source_id=source_id)

    if source_id == "reasoning_solver" and (
        value["kind"] != "solver" or value["policy"].get("solver_id") != SOLVER_ID
    ):
        raise ValueError("reasoning solver lock does not bind the frozen solver")


def _iter_canonical_jsonl(
    path: Path,
    *,
    label: str,
    allow_empty: bool,
) -> Iterator[dict[str, Any]]:
    count = 0
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith(b"\n"):
                raise ValueError(f"{label}:{line_number} is not newline-terminated")
            value = _strict_object(line, label=f"{label}:{line_number}")
            if canonical_json_bytes(value) != line:
                raise ValueError(f"{label}:{line_number} is not canonical JSON")
            count += 1
            yield value
    if count == 0 and not allow_empty:
        raise ValueError(f"{label} must contain at least one row")


def _burden(value: object, *, label: str) -> Fraction:
    if not isinstance(value, dict) or set(value) != _BURDEN_FIELDS:
        raise ValueError(f"{label} burden fields do not match")
    numerator = _nonnegative_int(value["numerator"], label=f"{label} numerator")
    denominator = _positive_int(value["denominator"], label=f"{label} denominator")
    return Fraction(numerator, denominator)


def _load_route_ledger(
    path: Path,
    *,
    lane: str,
    database: sqlite3.Connection,
) -> dict[str, Any]:
    database.execute("DELETE FROM lane_routes")
    route_rows = 0
    new_facts = 0
    new_external_facts = 0
    new_total_burden = Fraction()
    new_external_burden = Fraction()
    for row in _iter_canonical_jsonl(
        path, label=f"{lane} route ledger", allow_empty=True
    ):
        if set(row) != _ROUTE_FIELDS:
            raise ValueError(f"{lane} route ledger fields do not match")
        fact_id = _required_text(row["fact_id"], label=f"{lane} fact_id")
        if not isinstance(row["external"], bool):
            raise TypeError(f"{lane} route ledger external must be boolean")
        external = row["external"]
        burden = _burden(row["burden_bits"], label=f"{lane} fact {fact_id}")
        try:
            database.execute(
                "INSERT INTO lane_routes(fact_id, external, masked) VALUES (?, ?, 0)",
                (fact_id, int(external)),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(f"{lane} route ledger repeats fact {fact_id!r}") from error
        previous = database.execute(
            """
            SELECT external, burden_numerator, burden_denominator, first_lane
            FROM global_routes
            WHERE fact_id = ?
            """,
            (fact_id,),
        ).fetchone()
        if previous is None:
            database.execute(
                """
                INSERT INTO global_routes(
                    fact_id, external, burden_numerator, burden_denominator,
                    first_lane
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    fact_id,
                    int(external),
                    str(burden.numerator),
                    str(burden.denominator),
                    lane,
                ),
            )
            new_facts += 1
            new_external_facts += int(external)
            new_total_burden += burden
            if external:
                new_external_burden += burden
        else:
            previous_burden = Fraction(int(previous[1]), int(previous[2]))
            if bool(previous[0]) != external or previous_burden != burden:
                raise ValueError(
                    f"fact {fact_id!r} has inconsistent global routing evidence "
                    f"across {previous[3]} and {lane}"
                )
        route_rows += 1
        if route_rows % 100_000 == 0:
            database.commit()
    database.commit()
    return {
        "new_external_burden": new_external_burden,
        "new_external_facts": new_external_facts,
        "new_facts": new_facts,
        "new_total_burden": new_total_burden,
        "route_rows": route_rows,
    }


def _new_preflight_database() -> sqlite3.Connection:
    # An empty SQLite filename creates a temporary, file-backed database that
    # is removed on close.  SQLITE_TMPDIR/TMPDIR can point this external-memory
    # verifier at scratch storage for full 7.12B-token source roots.
    database = sqlite3.connect("")
    database.execute("PRAGMA journal_mode=OFF")
    database.execute("PRAGMA synchronous=OFF")
    database.execute("PRAGMA temp_store=FILE")
    database.executescript(
        """
        CREATE TABLE global_routes (
            fact_id TEXT PRIMARY KEY,
            external INTEGER NOT NULL CHECK(external IN (0, 1)),
            burden_numerator TEXT NOT NULL,
            burden_denominator TEXT NOT NULL,
            first_lane TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE lane_routes (
            fact_id TEXT PRIMARY KEY,
            external INTEGER NOT NULL CHECK(external IN (0, 1)),
            masked INTEGER NOT NULL CHECK(masked IN (0, 1))
        ) WITHOUT ROWID;
        CREATE TABLE verification_seen (
            record_id TEXT PRIMARY KEY
        ) WITHOUT ROWID;
        """
    )
    return database


def _require_repeated_byte(handle: Any, length: int, value: int, *, label: str) -> None:
    remaining = length
    expected_byte = bytes((value,))
    while remaining:
        chunk = handle.read(min(1 << 20, remaining))
        if not chunk or chunk != expected_byte * len(chunk):
            raise ValueError(label)
        remaining -= len(chunk)


def _verify_mask_ledger(
    split_path: Path,
    mask_path: Path,
    *,
    lane: str,
    token_count: int,
    database: sqlite3.Connection,
) -> dict[str, int]:
    position = 0
    zero_tokens = 0
    mask_rows = 0
    with split_path.open("rb") as weights:
        for row in _iter_canonical_jsonl(
            mask_path, label=f"{lane} mask ledger", allow_empty=True
        ):
            if set(row) != _MASK_FIELDS:
                raise ValueError(f"{lane} mask ledger fields do not match")
            start = _nonnegative_int(row["start"], label=f"{lane} mask start")
            end = _positive_int(row["end"], label=f"{lane} mask end")
            fact_id = _required_text(row["fact_id"], label=f"{lane} mask fact_id")
            if not position <= start < end <= token_count:
                raise ValueError(f"{lane} mask spans overlap, reorder, or exceed quota")
            route = database.execute(
                "SELECT external FROM lane_routes WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if route is None or not bool(route[0]):
                raise ValueError(
                    f"{lane} mask span references a missing or internal fact: {fact_id}"
                )
            _require_repeated_byte(
                weights,
                start - position,
                1,
                label=f"{lane} Split90 has zeros outside its mask ledger",
            )
            _require_repeated_byte(
                weights,
                end - start,
                0,
                label=f"{lane} Split90 leaves a routed occurrence supervised",
            )
            position = end
            zero_tokens += end - start
            database.execute(
                "UPDATE lane_routes SET masked = 1 WHERE fact_id = ?",
                (fact_id,),
            )
            mask_rows += 1
            if mask_rows % 100_000 == 0:
                database.commit()
        _require_repeated_byte(
            weights,
            token_count - position,
            1,
            label=f"{lane} Split90 has zeros outside its mask ledger",
        )
        if weights.read(1):
            raise ValueError(f"{lane} Split90 sidecar exceeds its token quota")
    missing = database.execute(
        """
        SELECT fact_id FROM lane_routes
        WHERE external = 1 AND masked = 0
        ORDER BY fact_id
        LIMIT 1
        """
    ).fetchone()
    if missing is not None:
        raise ValueError(f"{lane} routed fact has no masked occurrence: {missing[0]}")
    database.commit()
    masked_facts = int(
        database.execute(
            "SELECT COUNT(*) FROM lane_routes WHERE masked = 1"
        ).fetchone()[0]
    )
    return {
        "masked_facts": masked_facts,
        "zero_tokens": zero_tokens,
    }


def _parse_solver_premises(family: str, values: object) -> tuple[Any, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError("solver premises must be a non-empty list")
    premises = []
    for value in values:
        if not isinstance(value, dict):
            raise TypeError("solver premise must be an object")
        if family == "graph_composition_mod4":
            if set(value) != {"compose_code", "fact_id", "hop", "type"}:
                raise ValueError("composition premise fields do not match")
            if value["type"] != "composition":
                raise ValueError("composition premise type mismatch")
            premises.append(
                CompositionPremise(
                    fact_id=value["fact_id"],
                    hop=value["hop"],
                    compose_code=value["compose_code"],
                )
            )
        elif family == "slot_equality":
            if set(value) != {"fact_id", "slot", "type", "value"}:
                raise ValueError("equality premise fields do not match")
            if value["type"] != "equality":
                raise ValueError("equality premise type mismatch")
            premises.append(
                EqualityPremise(
                    fact_id=value["fact_id"],
                    slot=value["slot"],
                    value=value["value"],
                )
            )
        elif family == "graph_path_traversal":
            if set(value) != {
                "fact_id",
                "hop",
                "relation",
                "source",
                "target",
                "type",
            }:
                raise ValueError("graph traversal premise fields do not match")
            if value["type"] != "graph_traversal":
                raise ValueError("graph traversal premise type mismatch")
            premises.append(
                GraphTraversalPremise(
                    fact_id=value["fact_id"],
                    hop=value["hop"],
                    source=value["source"],
                    relation=value["relation"],
                    target=value["target"],
                )
            )
        else:
            raise ValueError(f"unsupported production solver family: {family}")
    return tuple(premises)


def _verify_solver_bundle(value: Mapping[str, Any]) -> None:
    if set(value) != {"family", "kind", "premises", "proof"}:
        raise ValueError("solver verification fields do not match")
    if value["kind"] != "solver":
        raise ValueError("reasoning verification kind must be solver")
    family = _required_text(value["family"], label="solver family")
    premises = _parse_solver_premises(family, value["premises"])
    if family == "graph_composition_mod4":
        expected = solve_graph_composition(premises)
    elif family == "slot_equality":
        expected = solve_slot_equality(premises)
    elif family == "graph_path_traversal":
        expected = solve_graph_traversal(premises)
    else:
        raise ValueError(f"unsupported production solver family: {family}")
    if value["proof"] != expected.as_dict():
        raise ValueError(f"{family} proof failed deterministic solver replay")


def _verify_objective_bundle(value: Mapping[str, Any]) -> None:
    if set(value) != {
        "answer",
        "kind",
        "reference_answer",
        "validator",
    }:
        raise ValueError("objective verification fields do not match")
    if (
        value["kind"] != "objective_answer"
        or value["validator"] != "canonical_exact_match"
        or canonical_json_bytes(value["answer"])
        != canonical_json_bytes(value["reference_answer"])
    ):
        raise ValueError("objective answer failed canonical exact validation")


def _verify_verification_ledger(
    path: Path,
    *,
    lane: str,
    token_count: int,
    source_ids: frozenset[str],
    recipe: ProductionRecipe,
    database: sqlite3.Connection,
) -> dict[str, int]:
    position = 0
    solver_rows = 0
    objective_rows = 0
    verified_rows = 0
    database.execute("DELETE FROM verification_seen")
    for row in _iter_canonical_jsonl(
        path, label=f"{lane} verification ledger", allow_empty=False
    ):
        if set(row) != _VERIFICATION_ROW_FIELDS:
            raise ValueError(f"{lane} verification ledger fields do not match")
        record_id = _required_text(
            row["record_id"], label=f"{lane} verification record_id"
        )
        try:
            database.execute(
                "INSERT INTO verification_seen(record_id) VALUES (?)",
                (record_id,),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(
                f"{lane} verification ledger repeats record {record_id!r}; "
                "reasoning cycle-fill is forbidden"
            ) from error
        source_id = _required_text(
            row["source_id"], label=f"{lane} verification source_id"
        )
        if source_id not in source_ids:
            raise ValueError(
                f"{lane} verification row uses unlocked source {source_id!r}"
            )
        start = _nonnegative_int(
            row["token_start"], label=f"{lane} verification token_start"
        )
        end = _positive_int(row["token_end"], label=f"{lane} verification token_end")
        if start != position or end <= start or end > token_count:
            raise ValueError(
                f"{lane} verification ranges must cover the token stream "
                "exactly once in source order"
            )
        verification = row["verification"]
        if not isinstance(verification, dict):
            raise TypeError(f"{lane} verification must be an object")
        if verification.get("kind") == "solver":
            _verify_solver_bundle(verification)
            solver_rows += 1
        elif lane == recipe.objective_lane:
            _verify_objective_bundle(verification)
            objective_rows += 1
        else:
            raise ValueError(f"{lane} requires solver verification for every record")
        position = end
        verified_rows += 1
        if verified_rows % 100_000 == 0:
            database.commit()
    if position != token_count:
        raise ValueError(
            f"{lane} verification ledger covers {position} of {token_count} tokens"
        )
    if lane in recipe.reasoning_lanes and solver_rows != verified_rows:
        raise ValueError(f"{lane} solver verification rate is below 1.0")
    database.commit()
    return {
        "objective_verified_records": objective_rows,
        "solver_verified_records": solver_rows,
        "verified_records": verified_rows,
    }


def production_preflight(
    source_root: Path | str,
    *,
    recipe: ProductionRecipe | None = None,
    recipe_path: Path | str = DEFAULT_RECIPE_PATH,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    recipe_value = recipe or load_production_recipe(recipe_path)
    root = Path(source_root)
    report: dict[str, Any] = {
        "contract_id": recipe_value.contract_id,
        "issues": [],
        "lane_order": list(recipe_value.lanes),
        "ready": False,
        "recipe_sha256": recipe_value.recipe_sha256,
        "required_total_tokens": recipe_value.total_tokens,
        "token_quotas": recipe_value.quota_by_lane,
    }
    issues: list[dict[str, Any]] = report["issues"]
    try:
        manifest = load_production_source_manifest(
            root,
            recipe=recipe_value,
            manifest_path=manifest_path,
        )
    except (OSError, ValueError) as error:
        issues.append(
            _issue(
                "source_manifest_unavailable",
                str(error),
                action=(
                    "materialize every required lane and immutable source lock, "
                    "then run source-production to seal source-manifest.json"
                ),
                path=str(
                    manifest_path
                    if manifest_path is not None
                    else root / DEFAULT_SOURCE_MANIFEST
                ),
            )
        )
        report["required_paths"] = list(expected_production_source_paths(recipe_value))
        return report

    root = manifest.root
    report["source_manifest_path"] = str(manifest.path)
    report["source_manifest_sha256"] = manifest.sha256
    lane_reports: dict[str, Any] = {}
    all_source_locks: dict[str, str] = {}
    total_tokens = 0
    total_zero_tokens = 0
    total_solver_rows = 0
    total_objective_rows = 0
    total_facts = 0
    external_facts = 0
    total_burden = Fraction()
    external_burden = Fraction()
    preflight_database = _new_preflight_database()

    for lane_id in recipe_value.lanes:
        lane = manifest.lane_by_id[lane_id]
        lane_report: dict[str, Any] = {"cycle_fill": lane["cycle_fill"]}
        lane_reports[lane_id] = lane_report
        try:
            token_descriptor = _validate_artifact_descriptor(
                lane["tokens"],
                label=f"{lane_id} tokens",
                allow_empty=False,
            )
            split_descriptor = _validate_artifact_descriptor(
                lane["split90_target_weights"],
                label=f"{lane_id} Split90",
                allow_empty=False,
            )
            route_descriptor = _validate_artifact_descriptor(
                lane["route_ledger"],
                label=f"{lane_id} route ledger",
                allow_empty=True,
            )
            mask_descriptor = _validate_artifact_descriptor(
                lane["mask_ledger"],
                label=f"{lane_id} mask ledger",
                allow_empty=True,
            )
            _verify_artifact(root, token_descriptor, label=f"{lane_id} token artifact")
            split_path = _verify_artifact(
                root, split_descriptor, label=f"{lane_id} Split90 artifact"
            )
            route_path = _verify_artifact(
                root, route_descriptor, label=f"{lane_id} route ledger"
            )
            mask_path = _verify_artifact(
                root, mask_descriptor, label=f"{lane_id} mask ledger"
            )
            if token_descriptor["bytes"] % 2:
                raise ValueError(f"{lane_id} uint16 token artifact has an odd byte")
            token_count = token_descriptor["bytes"] // 2
            expected_quota = recipe_value.quota_by_lane[lane_id]
            if token_count != expected_quota:
                raise ValueError(
                    f"{lane_id} has {token_count} tokens; frozen quota is "
                    f"{expected_quota} (deficit {expected_quota - token_count:+d})"
                )
            if split_descriptor["bytes"] != token_count:
                raise ValueError(
                    f"{lane_id} Split90 has {split_descriptor['bytes']} weights "
                    f"for {token_count} tokens"
                )
            total_tokens += token_count
            lane_report["tokens"] = token_count

            source_ids: set[str] = set()
            lock_digests: dict[str, str] = {}
            for lock in lane["source_locks"]:
                lock_path = _verify_artifact(
                    root, lock, label=f"{lane_id} source lock {lock['id']}"
                )
                _validate_source_lock(lock["id"], lock_path.read_bytes())
                previous_digest = all_source_locks.get(lock["id"])
                if previous_digest is not None and previous_digest != lock["sha256"]:
                    raise ValueError(
                        f"source lock {lock['id']!r} differs across production lanes"
                    )
                all_source_locks[lock["id"]] = lock["sha256"]
                source_ids.add(lock["id"])
                lock_digests[lock["id"]] = lock["sha256"]
            lane_report["source_locks"] = lock_digests

            routes = _load_route_ledger(
                route_path,
                lane=lane_id,
                database=preflight_database,
            )
            total_facts += routes["new_facts"]
            external_facts += routes["new_external_facts"]
            total_burden += routes["new_total_burden"]
            external_burden += routes["new_external_burden"]
            masks = _verify_mask_ledger(
                split_path,
                mask_path,
                lane=lane_id,
                token_count=token_count,
                database=preflight_database,
            )
            total_zero_tokens += masks["zero_tokens"]
            lane_report["route_facts"] = routes["route_rows"]
            lane_report.update(masks)

            verification_descriptor = lane["verification_ledger"]
            if verification_descriptor is not None:
                descriptor = _validate_artifact_descriptor(
                    verification_descriptor,
                    label=f"{lane_id} verification ledger",
                    allow_empty=False,
                )
                verification_path = _verify_artifact(
                    root,
                    descriptor,
                    label=f"{lane_id} verification ledger",
                )
                verification = _verify_verification_ledger(
                    verification_path,
                    lane=lane_id,
                    token_count=token_count,
                    source_ids=frozenset(source_ids),
                    recipe=recipe_value,
                    database=preflight_database,
                )
                lane_report.update(verification)
                total_solver_rows += verification["solver_verified_records"]
                total_objective_rows += verification["objective_verified_records"]
        except (OSError, TypeError, ValueError, sqlite3.Error) as error:
            issues.append(
                _issue(
                    "lane_preflight_failed",
                    str(error),
                    action=(
                        f"re-materialize {lane_id} from its pinned sources and "
                        "regenerate its route, mask, and verification ledgers"
                    ),
                    lane=lane_id,
                )
            )
    preflight_database.close()

    report["lanes"] = lane_reports
    report["observed_total_tokens"] = total_tokens
    report["solver_verified_records"] = total_solver_rows
    report["objective_verified_records"] = total_objective_rows
    report["source_locks"] = dict(sorted(all_source_locks.items()))
    report["split90_zero_tokens"] = total_zero_tokens
    if total_tokens != recipe_value.total_tokens:
        issues.append(
            _issue(
                "total_token_quota_mismatch",
                f"observed {total_tokens} of {recipe_value.total_tokens} tokens",
                action="satisfy every frozen lane quota; no lane may borrow or cycle",
            )
        )

    report["route_dose"] = {
        "distinct_external_facts": external_facts,
        "distinct_facts": total_facts,
        "distinct_fraction": (
            f"{external_facts}/{total_facts}" if total_facts else None
        ),
        "external_burden": (
            {
                "denominator": external_burden.denominator,
                "numerator": external_burden.numerator,
            }
            if total_burden
            else None
        ),
        "total_burden": (
            {
                "denominator": total_burden.denominator,
                "numerator": total_burden.numerator,
            }
            if total_burden
            else None
        ),
    }
    if total_facts <= 0 or total_burden <= 0:
        issues.append(
            _issue(
                "route_ledger_empty",
                "production route ledgers contain no positive-burden facts",
                action="emit the stable fact universe and train-only burden ledger",
            )
        )
    else:
        if Fraction(external_facts, total_facts) < Fraction(9, 10):
            issues.append(
                _issue(
                    "split90_distinct_dose_below_90_percent",
                    f"Split90 routes {external_facts}/{total_facts} distinct facts",
                    action="recompute the frozen Split90 route manifest",
                )
            )
        if external_burden / total_burden < Fraction(9, 10):
            issues.append(
                _issue(
                    "split90_burden_dose_below_90_percent",
                    "Split90 routes less than 90% of information-weighted burden",
                    action="recompute the frozen Split90 route manifest",
                )
            )
    report["ready"] = not issues
    return report


def require_production_preflight(
    source_root: Path | str,
    *,
    recipe: ProductionRecipe | None = None,
    recipe_path: Path | str = DEFAULT_RECIPE_PATH,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    report = production_preflight(
        source_root,
        recipe=recipe,
        recipe_path=recipe_path,
        manifest_path=manifest_path,
    )
    if report["ready"] is not True:
        raise ProductionPreflightError(report)
    return report


def seal_production_sources(
    source_root: Path | str,
    *,
    recipe: ProductionRecipe | None = None,
    recipe_path: Path | str = DEFAULT_RECIPE_PATH,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """Hash and seal a complete pre-materialized source layout.

    This operation never downloads, generates, substitutes, or cycles data.
    Missing inputs are returned as actionable preflight issues.
    """

    recipe_value = recipe or load_production_recipe(recipe_path)
    root = Path(source_root)
    if not root.is_dir() or root.is_symlink():
        report = {
            "contract_id": recipe_value.contract_id,
            "issues": [
                _issue(
                    "source_root_missing",
                    f"source root is missing or unsafe: {root}",
                    action="create a regular source root and materialize locked inputs",
                    path=str(root),
                )
            ],
            "ready": False,
            "required_paths": list(expected_production_source_paths(recipe_value)),
        }
        raise ProductionPreflightError(report)
    root = root.resolve()
    missing = [
        relative
        for relative in expected_production_source_paths(recipe_value)
        if not (root / relative).is_file() or (root / relative).is_symlink()
    ]
    if missing:
        report = {
            "contract_id": recipe_value.contract_id,
            "issues": [
                _issue(
                    "required_source_artifact_missing",
                    f"required source artifact is missing: {relative}",
                    action="materialize this artifact from an immutable source lock",
                    path=relative,
                )
                for relative in missing
            ],
            "ready": False,
            "required_paths": list(expected_production_source_paths(recipe_value)),
        }
        raise ProductionPreflightError(report)

    lanes = []
    try:
        for lane in recipe_value.lanes:
            lock_descriptors = []
            for source_id in recipe_value.locks_by_lane[lane]:
                relative = f"locks/{source_id}.lock.json"
                descriptor = _artifact_descriptor(root, relative)
                _validate_source_lock(source_id, (root / relative).read_bytes())
                lock_descriptors.append({"id": source_id, **descriptor})
            verification = (
                _artifact_descriptor(root, f"ledgers/{lane}.verification.jsonl")
                if lane in recipe_value.reasoning_lanes
                or lane == recipe_value.objective_lane
                else None
            )
            lanes.append(
                {
                    "cycle_fill": False,
                    "id": lane,
                    "mask_ledger": _artifact_descriptor(
                        root, f"ledgers/{lane}.masks.jsonl"
                    ),
                    "route_ledger": _artifact_descriptor(
                        root, f"ledgers/{lane}.routes.jsonl"
                    ),
                    "source_locks": lock_descriptors,
                    "split90_target_weights": _artifact_descriptor(
                        root, f"materialized/{lane}.split90.weights.bin"
                    ),
                    "tokens": _artifact_descriptor(
                        root, f"materialized/{lane}.tokens.bin"
                    ),
                    "verification_ledger": verification,
                }
            )
    except (OSError, TypeError, ValueError) as error:
        raise ProductionPreflightError(
            {
                "contract_id": recipe_value.contract_id,
                "issues": [
                    _issue(
                        "source_evidence_invalid",
                        str(error),
                        action=(
                            "repair the named materialized artifact or immutable "
                            "source lock, then rerun source-production"
                        ),
                    )
                ],
                "ready": False,
                "required_paths": list(expected_production_source_paths(recipe_value)),
            }
        ) from error
    value = {
        "contract_id": recipe_value.contract_id,
        "format": PRODUCTION_SOURCE_FORMAT,
        "lanes": lanes,
        "recipe_sha256": recipe_value.recipe_sha256,
        "sidecar_dtype": "uint8",
        "token_dtype": "uint16_le",
    }
    payload = canonical_json_bytes(value)
    destination = (
        Path(manifest_path)
        if manifest_path is not None
        else root / DEFAULT_SOURCE_MANIFEST
    )
    if not destination.is_absolute():
        destination = root / destination
    if destination.parent != root:
        raise ValueError(
            "source manifest destination must be a direct child of source_root"
        )
    candidate = destination.with_name(f".{destination.name}.candidate-{os.getpid()}")
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f"stale source manifest candidate: {candidate}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    try:
        with candidate.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        report = production_preflight(
            root,
            recipe=recipe_value,
            manifest_path=candidate,
        )
        if report["ready"] is not True:
            raise ProductionPreflightError(report)
        try:
            os.link(candidate, destination)
        except FileExistsError:
            if (
                not destination.is_file()
                or destination.is_symlink()
                or destination.read_bytes() != payload
            ):
                raise ValueError(
                    f"conflicting production source manifest: {destination}"
                )
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        candidate.unlink(missing_ok=True)
    # The candidate and destination contain identical canonical bytes (the
    # latter is hard-linked or was already byte-identical), so repeating the
    # external-memory route and verification scan would add no evidence and
    # doubles full-corpus sealing time.
    report = dict(report)
    report["source_manifest_path"] = str(destination)
    return report


def _slice_payload(
    lane: Mapping[str, Any],
    *,
    manifest_sha256: str,
    offset: int,
    count: int,
) -> bytes:
    verification = lane["verification_ledger"]
    return canonical_json_bytes(
        {
            "format": PRODUCTION_SLICE_FORMAT,
            "lane": lane["id"],
            "mask_ledger_sha256": lane["mask_ledger"]["sha256"],
            "route_ledger_sha256": lane["route_ledger"]["sha256"],
            "source_lock_sha256s": [lock["sha256"] for lock in lane["source_locks"]],
            "source_manifest_sha256": manifest_sha256,
            "split90_sha256": lane["split90_target_weights"]["sha256"],
            "token_count": count,
            "token_offset": offset,
            "tokens_sha256": lane["tokens"]["sha256"],
            "verification_ledger_sha256": (
                verification["sha256"] if verification is not None else None
            ),
        }
    )


def build_production_catalog(
    manifest: ProductionSourceManifest,
    *,
    recipe: ProductionRecipe,
    chunk_tokens: int = 262_144,
) -> InputCatalog:
    _positive_int(chunk_tokens, label="production catalog chunk_tokens")
    records = []
    ordinal = 0
    for lane_id in recipe.lanes:
        lane = manifest.lane_by_id[lane_id]
        quota = recipe.quota_by_lane[lane_id]
        offset = 0
        while offset < quota:
            count = min(chunk_tokens, quota - offset)
            flags = {
                f"lane:{lane_id}",
                "production",
                "source-locked",
                f"source-manifest:{manifest.sha256}",
            }
            if lane_id in recipe.reasoning_lanes:
                flags.update({"no-cycle-fill", "solver-verified"})
            elif lane_id == recipe.objective_lane:
                flags.add("objective-verified")
            records.append(
                CatalogRecord(
                    ordinal=ordinal,
                    record_id=f"{lane_id}-{offset:010d}-{count:07d}",
                    lane=lane_id,
                    source=lane_id,
                    source_key=f"tokens:{offset}:{count}",
                    payload=_slice_payload(
                        lane,
                        manifest_sha256=manifest.sha256,
                        offset=offset,
                        count=count,
                    ),
                    flags=tuple(sorted(flags)),
                )
            )
            ordinal += 1
            offset += count
    return InputCatalog(tuple(records))


def production_renderer_id(
    recipe: ProductionRecipe,
    source_manifest_sha256: str,
) -> str:
    _digest(source_manifest_sha256, label="source manifest sha256")
    return (
        f"{PRODUCTION_RENDERER_VERSION}:recipe={recipe.recipe_sha256}:"
        f"sources={source_manifest_sha256}"
    )


def _modification_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


class ProductionRenderer:
    """Read pinned uint16 lane slices and reproduce their exact bytes."""

    def __init__(
        self,
        manifest: ProductionSourceManifest,
        recipe: ProductionRecipe,
    ) -> None:
        self.manifest = manifest
        self.recipe = recipe
        self.renderer_id = production_renderer_id(recipe, manifest.sha256)
        self._files: dict[str, tuple[int, os.stat_result, int, os.stat_result]] = {}
        try:
            for lane_id in recipe.lanes:
                lane = manifest.lane_by_id[lane_id]
                token_fd, token_stat = self._open_verified(
                    lane["tokens"], label=f"{lane_id} token artifact"
                )
                try:
                    split_fd, split_stat = self._open_verified(
                        lane["split90_target_weights"],
                        label=f"{lane_id} Split90 artifact",
                    )
                except BaseException:
                    os.close(token_fd)
                    raise
                self._files[lane_id] = (
                    token_fd,
                    token_stat,
                    split_fd,
                    split_stat,
                )
        except BaseException:
            self.close()
            raise

    def _open_verified(
        self,
        descriptor: Mapping[str, Any],
        *,
        label: str,
    ) -> tuple[int, os.stat_result]:
        path = _safe_path(self.manifest.root, descriptor["path"], label=label)
        descriptor_fd = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            before = os.fstat(descriptor_fd)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{label} is not regular")
            digest = hashlib.sha256()
            offset = 0
            while offset < before.st_size:
                chunk = os.pread(
                    descriptor_fd,
                    min(1 << 20, before.st_size - offset),
                    offset,
                )
                if not chunk:
                    raise ValueError(f"{label} changed length while pinned")
                digest.update(chunk)
                offset += len(chunk)
            after = os.fstat(descriptor_fd)
            if (
                _modification_identity(before) != _modification_identity(after)
                or offset != descriptor["bytes"]
                or digest.hexdigest() != descriptor["sha256"]
            ):
                raise ValueError(f"{label} differs from its source manifest")
            return descriptor_fd, after
        except BaseException:
            os.close(descriptor_fd)
            raise

    def close(self) -> None:
        for token_fd, _token_stat, split_fd, _split_stat in getattr(
            self, "_files", {}
        ).values():
            for descriptor in (token_fd, split_fd):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        if hasattr(self, "_files"):
            self._files.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _descriptor(self, record: CatalogRecord) -> dict[str, Any]:
        value = _strict_object(
            record.payload, label=f"production catalog record {record.record_id}"
        )
        if (
            set(value) != _SLICE_FIELDS
            or canonical_json_bytes(value) != record.payload
            or value["format"] != PRODUCTION_SLICE_FORMAT
            or value["lane"] != record.lane
            or value["source_manifest_sha256"] != self.manifest.sha256
        ):
            raise ValueError(f"production catalog descriptor drift: {record.record_id}")
        lane = self.manifest.lane_by_id.get(record.lane)
        if lane is None:
            raise ValueError(f"unknown production lane: {record.lane}")
        verification = lane["verification_ledger"]
        expected = {
            "format": PRODUCTION_SLICE_FORMAT,
            "lane": record.lane,
            "mask_ledger_sha256": lane["mask_ledger"]["sha256"],
            "route_ledger_sha256": lane["route_ledger"]["sha256"],
            "source_lock_sha256s": [lock["sha256"] for lock in lane["source_locks"]],
            "source_manifest_sha256": self.manifest.sha256,
            "split90_sha256": lane["split90_target_weights"]["sha256"],
            "token_count": value["token_count"],
            "token_offset": value["token_offset"],
            "tokens_sha256": lane["tokens"]["sha256"],
            "verification_ledger_sha256": (
                verification["sha256"] if verification is not None else None
            ),
        }
        if value != expected:
            raise ValueError(
                f"production catalog source binding drift: {record.record_id}"
            )
        offset = _nonnegative_int(
            value["token_offset"], label="production token_offset"
        )
        count = _positive_int(value["token_count"], label="production token_count")
        if offset + count > self.recipe.quota_by_lane[record.lane]:
            raise ValueError(
                f"production catalog slice exceeds lane quota: {record.record_id}"
            )
        return value

    def _pread(
        self,
        lane: str,
        descriptor_index: int,
        expected_stat_index: int,
        offset: int,
        length: int,
        *,
        label: str,
    ) -> bytes:
        values = self._files[lane]
        file_descriptor = values[descriptor_index]
        expected_stat = values[expected_stat_index]
        before = os.fstat(file_descriptor)
        payload = os.pread(file_descriptor, length, offset)
        after = os.fstat(file_descriptor)
        if (
            _modification_identity(before) != _modification_identity(expected_stat)
            or _modification_identity(after) != _modification_identity(expected_stat)
            or len(payload) != length
        ):
            raise ValueError(f"{label} changed while pinned")
        return payload

    def render(self, record: CatalogRecord) -> RenderedRecord:
        descriptor = self._descriptor(record)
        offset = descriptor["token_offset"]
        count = descriptor["token_count"]
        payload = self._pread(
            record.lane,
            0,
            1,
            offset * 2,
            count * 2,
            label=f"{record.lane} token artifact",
        )
        token_ids = array("H")
        token_ids.frombytes(payload)
        if sys.byteorder != "little":
            token_ids.byteswap()
        flags = tuple(sorted({*record.flags, f"renderer:{self.renderer_id}"}))
        return RenderedRecord(tuple(token_ids), flags)

    def split90_weights(self, record: CatalogRecord) -> bytes:
        descriptor = self._descriptor(record)
        payload = self._pread(
            record.lane,
            2,
            3,
            descriptor["token_offset"],
            descriptor["token_count"],
            label=f"{record.lane} Split90 artifact",
        )
        if any(value not in (0, 1) for value in payload):
            raise ValueError(f"{record.lane} Split90 contains non-binary weights")
        return payload


def production_build_config(
    recipe: ProductionRecipe,
    *,
    shard_count: int = 32,
) -> ParallelBuildConfig:
    return ParallelBuildConfig(
        lane_weights=recipe.token_quotas,
        update_tokens=recipe.update_tokens,
        shard_count=shard_count,
        allow_fewer_shards=False,
    )


def _lane_token_counts(
    metadata: tuple[MetadataRecord, ...],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in metadata:
        counts[record.lane] += record.token_length
    return dict(counts)


def materialize_production_sidecars(
    catalog: InputCatalog,
    renderer: ProductionRenderer,
    config: ParallelBuildConfig,
    metadata: tuple[MetadataRecord, ...],
    work_dir: Path | str,
    *,
    recipe: ProductionRecipe,
) -> dict[str, Path]:
    root = Path(work_dir)
    if root.is_symlink():
        raise ValueError(f"production work directory is a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "dense_target_weights": root / "dense_target_weights.bin",
        "split90_target_weights": root / "split90_target_weights.bin",
    }
    temporary = {
        name: path.with_name(f".{path.name}.partial-{os.getpid()}")
        for name, path in paths.items()
    }
    if any(path.exists() or path.is_symlink() for path in temporary.values()):
        raise FileExistsError("stale production sidecar temporary file")
    schedule = largest_deficit_schedule(metadata, config.lane_weights)
    by_id = {record.record_id: record for record in catalog.records}
    logical_tokens = 0
    try:
        with (
            temporary["dense_target_weights"].open("xb") as dense,
            temporary["split90_target_weights"].open("xb") as split,
        ):
            for entry in schedule:
                record = by_id[entry.record_id]
                weights = renderer.split90_weights(record)
                if len(weights) != entry.token_length:
                    raise ValueError(f"Split90/catalog length drift: {entry.record_id}")
                dense.write(b"\x01" * entry.token_length)
                split.write(weights)
                logical_tokens += entry.token_length
            for handle in (dense, split):
                handle.flush()
                os.fsync(handle.fileno())
        if logical_tokens != recipe.total_tokens:
            raise ValueError(
                f"sidecars cover {logical_tokens} of {recipe.total_tokens} tokens"
            )
        for name in ("dense_target_weights", "split90_target_weights"):
            expected_size, expected_digest = _size_sha256(temporary[name])
            try:
                os.link(temporary[name], paths[name])
            except FileExistsError:
                existing = paths[name]
                if (
                    not existing.is_file()
                    or existing.is_symlink()
                    or _size_sha256(existing) != (expected_size, expected_digest)
                ):
                    raise ValueError(
                        f"conflicting production sidecar work file: {existing}"
                    )
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
    return paths


def verify_production_corpus(
    root: Path | str,
    *,
    recipe: ProductionRecipe,
    source_manifest_sha256: str,
    expected_build_id: str | None = None,
) -> VerifiedParallelCorpus:
    receipt = verify_parallel_corpus(root, expected_build_id=expected_build_id)
    expected_renderer = production_renderer_id(recipe, source_manifest_sha256)
    expected_config = production_build_config(
        recipe, shard_count=receipt["config"]["shard_count"]
    )
    if (
        receipt["format"] != "memorysplit-parallel-corpus-v2"
        or receipt["renderer_id"] != expected_renderer
        or receipt["logical_tokens"] != recipe.total_tokens
        or receipt["packed_tokens"] != recipe.total_tokens
        or receipt["padding_tokens"] != 0
        or receipt["config"] != expected_config.as_dict()
        or [item["name"] for item in receipt["sidecar_sets"]]
        != ["dense_target_weights", "split90_target_weights"]
    ):
        raise ValueError("published corpus does not match frozen production identity")
    publication = Path(root)
    metadata = metadata_from_bytes((publication / "metadata.jsonl").read_bytes())
    lane_tokens = _lane_token_counts(metadata)
    if lane_tokens != recipe.quota_by_lane:
        raise ValueError(f"published production lane quotas drifted: {lane_tokens}")
    for record in metadata:
        flags = set(record.flags)
        required = {
            f"lane:{record.lane}",
            "production",
            "source-locked",
            f"source-manifest:{source_manifest_sha256}",
            f"renderer:{expected_renderer}",
        }
        if not required <= flags:
            raise ValueError(f"published production flags drifted: {record.record_id}")
        if (
            record.lane in recipe.reasoning_lanes
            and not {
                "no-cycle-fill",
                "solver-verified",
            }
            <= flags
        ):
            raise ValueError(f"reasoning proof/cycle flags drifted: {record.record_id}")
        if record.lane == recipe.objective_lane and "objective-verified" not in flags:
            raise ValueError(f"objective verification flag drifted: {record.record_id}")
    return receipt


def build_production_corpus(
    source_root: Path | str,
    destination: Path | str,
    work_dir: Path | str,
    *,
    recipe: ProductionRecipe | None = None,
    recipe_path: Path | str = DEFAULT_RECIPE_PATH,
    manifest_path: Path | str | None = None,
    workers: int = 1,
    shard_count: int = 32,
    chunk_tokens: int = 262_144,
) -> VerifiedParallelCorpus:
    recipe_value = recipe or load_production_recipe(recipe_path)
    preflight = require_production_preflight(
        source_root,
        recipe=recipe_value,
        manifest_path=manifest_path,
    )
    manifest = load_production_source_manifest(
        source_root,
        recipe=recipe_value,
        manifest_path=manifest_path,
    )
    if manifest.sha256 != preflight["source_manifest_sha256"]:
        raise ValueError("source manifest changed after production preflight")
    catalog = build_production_catalog(
        manifest,
        recipe=recipe_value,
        chunk_tokens=chunk_tokens,
    )
    config = production_build_config(recipe_value, shard_count=shard_count)
    output = Path(destination)
    if output.exists() or output.is_symlink():
        existing = verify_production_corpus(
            output,
            recipe=recipe_value,
            source_manifest_sha256=manifest.sha256,
        )
        if (
            existing["catalog_sha256"] != catalog.sha256
            or existing["config"] != config.as_dict()
        ):
            raise ValueError(
                "existing production corpus conflicts with requested build options"
            )
        return existing
    with ProductionRenderer(manifest, recipe_value) as renderer:
        metadata = render_metadata(catalog, renderer, workers=workers)
        if _lane_token_counts(metadata) != recipe_value.quota_by_lane:
            raise ValueError("rendered metadata does not meet exact frozen lane quotas")
        sidecars = materialize_production_sidecars(
            catalog,
            renderer,
            config,
            metadata,
            work_dir,
            recipe=recipe_value,
        )
        receipt = build_parallel_corpus(
            catalog,
            renderer,
            config,
            destination,
            workers=workers,
            sidecar_paths=sidecars,
            _materialized_metadata=metadata,
        )
    return verify_production_corpus(
        destination,
        recipe=recipe_value,
        source_manifest_sha256=manifest.sha256,
        expected_build_id=receipt["build_id"],
    )


__all__ = [
    "DEFAULT_RECIPE_PATH",
    "DEFAULT_SOURCE_MANIFEST",
    "FROZEN_LANES",
    "FROZEN_OBJECTIVE_SOURCE_IDS",
    "FROZEN_OPTIMIZER_STEPS",
    "FROZEN_REQUIRED_SOURCE_LOCKS",
    "FROZEN_TOKEN_QUOTAS",
    "FROZEN_TOTAL_TOKENS",
    "FROZEN_UPDATE_TOKENS",
    "HUGGINGFACE_SOURCE_LOCK_FORMAT",
    "OBJECTIVE_LANE",
    "OBJECTIVE_SOURCE_LOCK_FORMAT",
    "PRODUCTION_LOCK_FORMAT",
    "PRODUCTION_RENDERER_VERSION",
    "PRODUCTION_SOURCE_FORMAT",
    "ProductionPreflightError",
    "ProductionRecipe",
    "ProductionRenderer",
    "ProductionSourceManifest",
    "build_production_catalog",
    "build_production_corpus",
    "expected_production_source_paths",
    "load_production_recipe",
    "load_production_source_manifest",
    "materialize_production_sidecars",
    "production_build_config",
    "production_preflight",
    "production_renderer_id",
    "require_production_preflight",
    "seal_production_sources",
    "verify_production_corpus",
]
