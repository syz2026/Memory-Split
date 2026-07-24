"""Strict Task-4-to-Slurm compatibility for the complete MemorySplit v2 corpus."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

from cluster.aws.p5.corpus_contract import (
    CorpusContractError as Task4CorpusContractError,
)
from cluster.aws.p5.corpus_contract import (
    verify_canonical_corpus as verify_task4_canonical_corpus,
)
from corpusgen.parallel import atomic_rename_noreplace, schedule_from_bytes
from corpusgen.parallel.canonical import canonical_json_bytes
from msctl.cohort import (
    DATASET_CONTRACT_ID,
    RAW_TARGETS,
    TARGETS_PER_UPDATE,
    TERMINAL_UPDATES,
)


POINTER_SCHEMA_VERSION = 2
BRIDGE_RECEIPT_SCHEMA_VERSION = 1
BRIDGE_FORMAT = "memorysplit-slurm-135m-flat-v1"
TASK4_FORMAT = "memorysplit-parallel-corpus-v2"
DATASET_ID = "memorysplit-v2-20x-reasoning-max-cohort"
AUTHORITATIVE_RECIPE_SHA256 = (
    "c704baf9e07acb93ffab8b33027b310fc9b3d68e0c036a6d8cc1d45b86872ce8"
)
SIDECAR_NAMES = ("dense_target_weights", "split90_target_weights")
STREAM_PATHS = {
    "packed_targets": "packed/targets.bin",
    "dense_target_weights": "sidecars/dense_target_weights.bin",
    "split90_target_weights": "sidecars/split90_target_weights.bin",
}
SOURCE_LOCK_MANIFEST = "configs/reasoning-dataset-v2.json"
DATASET_RELATIVE_PATH = "dataset"
MIRROR_ROOT_ENVIRONMENT = {
    "farmshare": "MS135_FARMSHARE_DATASET_ROOT",
    "mit": "MS135_MIT_DATASET_ROOT",
}
AUTHORITATIVE_LANES = (
    ("fineweb_edu", Fraction(25)),
    ("finemath", Fraction(15)),
    ("wikidata_graph", Fraction(20)),
    ("synthetic_graph", Fraction(10)),
    ("verified_synthetic_multihop", Fraction(15)),
    ("wikidata_path_reasoning", Fraction(15, 2)),
    ("relational_refinement", Fraction(5, 2)),
    ("objective_auxiliary", Fraction(5)),
)

_HEX = frozenset("0123456789abcdef")
_POINTER_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "dataset_id",
        "materialization",
        "layout_format",
        "relative_path",
        "receipt_relative_path",
        "required_sidecars",
        "source_lock_manifest",
        "expected_source_lock_sha256",
        "expected_receipt_sha256",
        "expected_source_receipt_sha256",
        "expected_ordered_token_stream_sha256",
        "expected_packed_stream_sha256",
        "launch_gate_status",
        "mirror_root_environment",
    }
)
_DYNAMIC_POINTER_FIELDS = (
    "expected_receipt_sha256",
    "expected_source_receipt_sha256",
    "expected_ordered_token_stream_sha256",
    "expected_packed_stream_sha256",
)
_BRIDGE_FIELDS = frozenset(
    {
        "schema_version",
        "format",
        "contract_id",
        "source_recipe_sha256",
        "raw_target_tokens",
        "lane_ids",
        "task4_publication",
        "artifacts",
    }
)
_TASK4_BINDING_FIELDS = frozenset(
    {
        "format",
        "receipt_sha256",
        "build_id",
        "ordered_stream_sha256",
        "merkle_root_sha256",
        "packed_stream_sha256",
        "sidecar_stream_sha256",
    }
)
_ARTIFACT_FIELDS = frozenset({"bytes", "path", "sha256"})


class CorpusContractError(ValueError):
    """The corpus cannot satisfy the protected 135M launch contract."""


@dataclass(frozen=True)
class DatasetPointer:
    dataset_id: str
    contract_id: str
    layout_format: str
    relative_path: str
    receipt_relative_path: str
    source_lock_manifest: str
    expected_source_lock_sha256: str
    expected_receipt_sha256: str
    expected_source_receipt_sha256: str
    expected_ordered_token_stream_sha256: str
    expected_packed_stream_sha256: str
    mirror_root_environment: Mapping[str, str]


@dataclass(frozen=True)
class RecipeContract:
    sha256: str
    raw_target_tokens: int
    update_tokens: int
    lane_ids: tuple[str, ...]
    lane_shares: tuple[Fraction, ...]
    token_quotas: Mapping[str, int]


@dataclass(frozen=True)
class Task4PublicationEvidence:
    root: Path
    receipt: Mapping[str, Any]
    receipt_sha256: str
    ordered_stream_sha256: str
    packed_stream_sha256: str
    sidecar_stream_sha256: Mapping[str, str]
    lane_ids: tuple[str, ...]
    raw_target_tokens: int
    source_recipe_sha256: str


@dataclass(frozen=True)
class CorpusEvidence:
    contract_id: str
    receipt_path: str
    receipt_sha256: str
    source_receipt_sha256: str
    ordered_token_stream_sha256: str
    packed_stream_sha256: str
    stream_sha256: Mapping[str, str]
    raw_target_tokens: int
    lane_ids: tuple[str, ...]
    semantic_verification_sha256: str
    semantic_verification_passed: bool

    def identity(self) -> tuple[Any, ...]:
        """Return the content identity, excluding the site-local receipt path."""

        return (
            self.contract_id,
            self.receipt_sha256,
            self.source_receipt_sha256,
            self.ordered_token_stream_sha256,
            self.packed_stream_sha256,
            tuple(sorted(self.stream_sha256.items())),
            self.raw_target_tokens,
            self.lane_ids,
            self.semantic_verification_sha256,
            self.semantic_verification_passed,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "lane_ids": list(self.lane_ids),
            "ordered_token_stream_sha256": self.ordered_token_stream_sha256,
            "packed_stream_sha256": self.packed_stream_sha256,
            "raw_target_tokens": self.raw_target_tokens,
            "receipt_path": self.receipt_path,
            "receipt_sha256": self.receipt_sha256,
            "semantic_verification_passed": self.semantic_verification_passed,
            "semantic_verification_sha256": self.semantic_verification_sha256,
            "source_receipt_sha256": self.source_receipt_sha256,
            "stream_sha256": dict(self.stream_sha256),
        }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _HEX
    )


def _require_sha256(value: object, *, label: str) -> str:
    if not _is_sha256(value):
        raise CorpusContractError(f"{label} must be a lowercase SHA-256 digest")
    return str(value)


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CorpusContractError(f"{label} must be a positive integer")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CorpusContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _exact_mapping(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CorpusContractError(f"{label} must be a JSON object")
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise CorpusContractError(
            f"{label} fields differ from the protected schema; "
            f"missing={missing}, unknown={unknown}"
        )
    return value


def _portable_relative(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CorpusContractError(f"{label} must be a portable relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CorpusContractError(f"{label} must be a portable relative path")
    return value


def _read_regular_bytes(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> bytes:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise CorpusContractError(f"{label} is missing, symlinked, or unsafe") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise CorpusContractError(f"{label} must be a singly linked regular file")
        if before.st_size > maximum_bytes:
            raise CorpusContractError(f"{label} exceeds the maximum supported size")
        chunks = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1 << 20, maximum_bytes + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > maximum_bytes:
                raise CorpusContractError(
                    f"{label} exceeds the maximum supported size"
                )
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after) or size != after.st_size:
            raise CorpusContractError(f"{label} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_json(
    path: Path,
    *,
    label: str,
    maximum_bytes: int = 16 << 20,
) -> tuple[Any, bytes]:
    data = _read_regular_bytes(path, label=label, maximum_bytes=maximum_bytes)
    try:
        value = json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                CorpusContractError(f"{label} contains non-finite JSON: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusContractError(f"{label} must be valid UTF-8 JSON") from error
    return value, data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    _stream_regular(path, label=str(path), consumer=digest.update)
    return digest.hexdigest()


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stream_regular(
    path: Path,
    *,
    label: str,
    consumer: Callable[[bytes], None] | None = None,
) -> tuple[int, str]:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise CorpusContractError(f"{label} is missing, symlinked, or unsafe") from error
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise CorpusContractError(f"{label} must be a singly linked regular file")
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            if consumer is not None:
                consumer(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after) or size != after.st_size:
            raise CorpusContractError(f"{label} changed while being read")
    finally:
        os.close(descriptor)
    return size, digest.hexdigest()


def _safe_member(root: Path, relative: str, *, label: str) -> Path:
    _portable_relative(relative, label=label)
    resolved_root = root.resolve(strict=True)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise CorpusContractError(f"{label} is missing: {relative}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise CorpusContractError(f"{label} traverses a symlink: {relative}")
    try:
        current.resolve(strict=True).relative_to(resolved_root)
    except (FileNotFoundError, ValueError) as error:
        raise CorpusContractError(f"{label} escapes its root: {relative}") from error
    return current


def _share(value: object, *, label: str) -> Fraction:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise CorpusContractError(f"{label} must be finite and positive")
    return Fraction(str(value))


def _hamilton_quotas(
    total: int,
    lanes: tuple[tuple[str, Fraction], ...],
) -> dict[str, int]:
    ideals = [(lane, Fraction(total) * share / 100) for lane, share in lanes]
    quotas = {lane: int(ideal) for lane, ideal in ideals}
    remaining = total - sum(quotas.values())
    order = sorted(
        range(len(ideals)),
        key=lambda index: (-(ideals[index][1] - int(ideals[index][1])), index),
    )
    for index in order[:remaining]:
        quotas[ideals[index][0]] += 1
    return quotas


def _load_source_recipe(path: Path, *, expected_sha256: str) -> RecipeContract:
    expected = _require_sha256(expected_sha256, label="source recipe")
    raw, data = _load_json(path, label="authoritative reasoning dataset recipe")
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise CorpusContractError(
            "reasoning dataset recipe does not match its authoritative SHA-256"
        )
    if not isinstance(raw, Mapping):
        raise CorpusContractError("reasoning dataset recipe must be an object")
    if (
        type(raw.get("schema_version")) is not int
        or raw.get("schema_version") != 2
        or raw.get("contract_id") != "memorysplit-reasoning-dataset-v2"
    ):
        raise CorpusContractError("reasoning dataset recipe identity is not v2")
    recipe = raw.get("sprint_recipe")
    if not isinstance(recipe, Mapping):
        raise CorpusContractError("reasoning dataset sprint recipe is missing")
    expected_geometry = {
        "label": "20tpp_reasoning_maximized_sprint",
        "model_parameters": 356_033_536,
        "nominal_tokens_per_parameter": 20,
        "targets_per_update": TARGETS_PER_UPDATE,
        "optimizer_steps": TERMINAL_UPDATES,
        "raw_target_tokens": RAW_TARGETS,
        "mixture_unit": "raw_causal_target_tokens_before_condition_weights",
        "share_semantics": "target_percentages",
    }
    for field, value in expected_geometry.items():
        if type(recipe.get(field)) is not type(value) or recipe.get(field) != value:
            raise CorpusContractError(
                f"reasoning dataset recipe {field} differs from the 135M contract"
            )
    if RAW_TARGETS != TARGETS_PER_UPDATE * TERMINAL_UPDATES:
        raise CorpusContractError("135M target geometry is not integral")

    lanes = recipe.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != len(AUTHORITATIVE_LANES):
        raise CorpusContractError("reasoning dataset recipe must contain eight lanes")
    parsed_lanes = []
    for index, (lane, authoritative) in enumerate(
        zip(lanes, AUTHORITATIVE_LANES, strict=True)
    ):
        if not isinstance(lane, Mapping):
            raise CorpusContractError(f"reasoning lane {index} must be an object")
        if "source_lock_sha256" in lane:
            raise CorpusContractError(
                "reasoning lanes must not contain invented per-lane digests"
            )
        lane_id = lane.get("id")
        lane_share = _share(
            lane.get("share_percent"),
            label=f"reasoning lane {index} share",
        )
        if (lane_id, lane_share) != authoritative:
            raise CorpusContractError(
                "reasoning dataset lane ids or shares differ from the "
                "authoritative upstream recipe"
            )
        parsed_lanes.append((str(lane_id), lane_share))
    if sum((share for _lane, share in parsed_lanes), Fraction()) != 100:
        raise CorpusContractError("reasoning dataset lane shares must total 100")

    allocation = recipe.get("realized_token_allocation")
    if not isinstance(allocation, Mapping):
        raise CorpusContractError("realized token allocation is missing")
    expected_quotas = _hamilton_quotas(RAW_TARGETS, tuple(parsed_lanes))
    if (
        allocation.get("method") != "hamilton_largest_remainder"
        or allocation.get("tie_break") != "stable_lane_order"
        or allocation.get("lane_order") != [lane for lane, _share in parsed_lanes]
        or type(allocation.get("total_tokens")) is not int
        or allocation.get("total_tokens") != RAW_TARGETS
        or allocation.get("token_quotas") != expected_quotas
    ):
        raise CorpusContractError(
            "reasoning dataset realized token allocation is not authoritative"
        )
    publication = recipe.get("publication_requirements")
    if publication != {
        "complete_once_wikidata_training_graph": True,
        "stable_fact_universe_and_exposure_burden": True,
        "solver_verification_rate": 1.0,
        "structural_train_evaluation_overlap_max": 0.0,
        "deterministic_rebuild_required": True,
        "reasoning_lane_cycle_fill_forbidden": True,
    }:
        raise CorpusContractError(
            "reasoning dataset publication requirements differ from upstream"
        )
    intervention = raw.get("intervention")
    if (
        not isinstance(intervention, Mapping)
        or intervention.get("primary_arms") != ["dense", "split90"]
        or intervention.get("only_claim_bearing_difference")
        != "direct_target_weights_on_routed_factual_payloads"
        or intervention.get("semantic_mask_closure_required") is not True
    ):
        raise CorpusContractError("reasoning dataset intervention is not authoritative")
    invariants = intervention.get("matched_pair_invariants")
    required_invariants = {
        "raw_target_tokens",
        "token_bytes",
        "token_order",
        "packing_boundaries",
        "targets_per_update",
        "training_steps",
    }
    if not isinstance(invariants, list) or not required_invariants <= set(invariants):
        raise CorpusContractError("reasoning dataset matched-pair invariants are incomplete")
    return RecipeContract(
        sha256=actual,
        raw_target_tokens=RAW_TARGETS,
        update_tokens=TARGETS_PER_UPDATE,
        lane_ids=tuple(lane for lane, _share in parsed_lanes),
        lane_shares=tuple(share for _lane, share in parsed_lanes),
        token_quotas=expected_quotas,
    )


def _validate_publication_recipe(
    root: Path,
    receipt: Mapping[str, Any],
    recipe: RecipeContract,
) -> tuple[dict[str, str], tuple[str, ...]]:
    if receipt.get("format") != TASK4_FORMAT:
        raise CorpusContractError("Task-4 publication is not receipt v2")
    if (
        type(receipt.get("logical_tokens")) is not int
        or receipt["logical_tokens"] != recipe.raw_target_tokens
        or type(receipt.get("packed_tokens")) is not int
        or receipt["packed_tokens"] != recipe.raw_target_tokens
        or type(receipt.get("padding_tokens")) is not int
        or receipt["padding_tokens"] != 0
    ):
        raise CorpusContractError(
            "Task-4 publication does not contain the exact complete 135M horizon"
        )
    config = receipt.get("config")
    if not isinstance(config, Mapping):
        raise CorpusContractError("Task-4 publication config is missing")
    if (
        type(config.get("update_tokens")) is not int
        or config["update_tokens"] != recipe.update_tokens
    ):
        raise CorpusContractError(
            "Task-4 publication update geometry differs from the 135M contract"
        )
    weights = config.get("lane_weights")
    if not isinstance(weights, list) or len(weights) != len(recipe.lane_ids):
        raise CorpusContractError("Task-4 publication lane weights are incomplete")
    parsed_weights = []
    for index, (item, lane_id) in enumerate(
        zip(weights, recipe.lane_ids, strict=True)
    ):
        if (
            not isinstance(item, Mapping)
            or set(item) != {"lane", "weight"}
            or item.get("lane") != lane_id
        ):
            raise CorpusContractError(
                "Task-4 publication lane order differs from the recipe"
            )
        parsed_weights.append(
            _positive_int(item.get("weight"), label=f"lane weight {index}")
        )
    expected_weights = [
        recipe.token_quotas[lane_id] for lane_id in recipe.lane_ids
    ]
    if parsed_weights != expected_weights:
        raise CorpusContractError(
            "Task-4 publication lane weights do not match the authoritative "
            "Hamilton quotas"
        )

    schedule_path = _safe_member(
        root,
        "schedule.jsonl",
        label="Task-4 schedule",
    )
    schedule_bytes = _read_regular_bytes(
        schedule_path,
        label="Task-4 schedule",
        maximum_bytes=256 << 20,
    )
    if hashlib.sha256(schedule_bytes).hexdigest() != receipt.get("schedule_sha256"):
        raise CorpusContractError("Task-4 schedule changed after canonical verification")
    try:
        schedule = schedule_from_bytes(schedule_bytes)
    except ValueError as error:
        raise CorpusContractError("Task-4 schedule is not canonical") from error
    realized = Counter()
    for record in schedule:
        realized[record.lane] += record.token_length
    if dict(realized) != dict(recipe.token_quotas):
        raise CorpusContractError(
            "Task-4 publication does not realize the authoritative lane quotas"
        )

    sidecar_sets = receipt.get("sidecar_sets")
    if not isinstance(sidecar_sets, list):
        raise CorpusContractError("Task-4 publication has no bound sidecar sets")
    sidecar_digests = {}
    for index, name in enumerate(SIDECAR_NAMES):
        try:
            item = sidecar_sets[index]
        except IndexError as error:
            raise CorpusContractError(
                "Task-4 publication sidecar sets are incomplete"
            ) from error
        if not isinstance(item, Mapping) or item.get("name") != name:
            raise CorpusContractError(
                "Task-4 publication sidecar sets are not canonical and ordered"
            )
        sidecar_digests[name] = _require_sha256(
            item.get("stream_sha256"),
            label=f"Task-4 {name} stream",
        )
    if len(sidecar_sets) != len(SIDECAR_NAMES):
        raise CorpusContractError("Task-4 publication contains extra sidecar sets")
    return sidecar_digests, recipe.lane_ids


def verify_task4_publication(
    publication_root: Path | str,
    *,
    source_lock_path: Path | str,
    expected_source_lock_sha256: str,
    expected_receipt_sha256: str | None = None,
    expected_ordered_sha256: str | None = None,
) -> Task4PublicationEvidence:
    """Run the exact upstream semantic verifier, then bind it to the recipe."""

    root = Path(publication_root)
    if root.is_symlink() or not root.is_dir():
        raise CorpusContractError("Task-4 publication root is missing or unsafe")
    recipe = _load_source_recipe(
        Path(source_lock_path),
        expected_sha256=expected_source_lock_sha256,
    )
    receipt_path = root / "receipt.json"
    raw, receipt_bytes = _load_json(receipt_path, label="Task-4 corpus receipt")
    if not isinstance(raw, Mapping):
        raise CorpusContractError("Task-4 corpus receipt must be an object")
    actual_receipt = hashlib.sha256(receipt_bytes).hexdigest()
    receipt_lock = (
        actual_receipt
        if expected_receipt_sha256 is None
        else _require_sha256(
            expected_receipt_sha256,
            label="expected Task-4 receipt",
        )
    )
    ordered = _require_sha256(
        raw.get("ordered_stream_sha256"),
        label="Task-4 ordered stream",
    )
    ordered_lock = (
        ordered
        if expected_ordered_sha256 is None
        else _require_sha256(
            expected_ordered_sha256,
            label="expected Task-4 ordered stream",
        )
    )
    try:
        upstream = verify_task4_canonical_corpus(
            receipt_path,
            expected_sha256=receipt_lock,
            expected_ordered_sha256=ordered_lock,
        )
    except (Task4CorpusContractError, OSError, ValueError) as error:
        raise CorpusContractError(
            "authoritative Task-4 verifier rejected the publication"
        ) from error
    receipt = dict(upstream.receipt)
    if receipt != raw:
        raise CorpusContractError(
            "Task-4 receipt changed across canonical verification"
        )
    if sha256_file(receipt_path) != actual_receipt:
        raise CorpusContractError(
            "Task-4 receipt changed after canonical verification"
        )
    sidecars, lane_ids = _validate_publication_recipe(root, receipt, recipe)
    return Task4PublicationEvidence(
        root=root.resolve(strict=True),
        receipt=receipt,
        receipt_sha256=actual_receipt,
        ordered_stream_sha256=ordered,
        packed_stream_sha256=_require_sha256(
            receipt.get("packed_stream_sha256"),
            label="Task-4 packed stream",
        ),
        sidecar_stream_sha256=sidecars,
        lane_ids=lane_ids,
        raw_target_tokens=recipe.raw_target_tokens,
        source_recipe_sha256=recipe.sha256,
    )


def _bridge_receipt(evidence: Task4PublicationEvidence) -> dict[str, Any]:
    artifacts = [
        {
            "bytes": evidence.raw_target_tokens * 2,
            "path": STREAM_PATHS["packed_targets"],
            "sha256": evidence.packed_stream_sha256,
        },
        *[
            {
                "bytes": evidence.raw_target_tokens,
                "path": STREAM_PATHS[name],
                "sha256": evidence.sidecar_stream_sha256[name],
            }
            for name in SIDECAR_NAMES
        ],
    ]
    artifacts.sort(key=lambda item: item["path"])
    return {
        "artifacts": artifacts,
        "contract_id": DATASET_CONTRACT_ID,
        "format": BRIDGE_FORMAT,
        "lane_ids": list(evidence.lane_ids),
        "raw_target_tokens": evidence.raw_target_tokens,
        "schema_version": BRIDGE_RECEIPT_SCHEMA_VERSION,
        "source_recipe_sha256": evidence.source_recipe_sha256,
        "task4_publication": {
            "build_id": evidence.receipt["build_id"],
            "format": TASK4_FORMAT,
            "merkle_root_sha256": evidence.receipt["merkle_root_sha256"],
            "ordered_stream_sha256": evidence.ordered_stream_sha256,
            "packed_stream_sha256": evidence.packed_stream_sha256,
            "receipt_sha256": evidence.receipt_sha256,
            "sidecar_stream_sha256": dict(evidence.sidecar_stream_sha256),
        },
    }


def _publication_artifacts(
    evidence: Task4PublicationEvidence,
) -> dict[str, list[Mapping[str, Any]]]:
    primary = [
        item
        for item in evidence.receipt["artifacts"]
        if str(item["path"]).startswith("shards/")
    ]
    sidecars = {
        item["name"]: list(item["artifacts"])
        for item in evidence.receipt["sidecar_sets"]
    }
    return {"packed_targets": primary, **sidecars}


def _concatenate_artifacts(
    root: Path,
    records: list[Mapping[str, Any]],
    destination: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    stream_digest = hashlib.sha256()
    stream_bytes = 0
    try:
        with destination.open("xb") as output:
            for record in records:
                item = _exact_mapping(
                    record,
                    fields=_ARTIFACT_FIELDS,
                    label="Task-4 shard artifact",
                )
                relative = _portable_relative(
                    item["path"],
                    label="Task-4 shard path",
                )
                expected_item_bytes = _positive_int(
                    item["bytes"],
                    label=f"Task-4 shard bytes {relative}",
                )
                expected_item_sha = _require_sha256(
                    item["sha256"],
                    label=f"Task-4 shard digest {relative}",
                )
                item_digest = hashlib.sha256()

                def consume(chunk: bytes) -> None:
                    output.write(chunk)
                    item_digest.update(chunk)
                    stream_digest.update(chunk)

                size, digest = _stream_regular(
                    _safe_member(root, relative, label="Task-4 shard"),
                    label=f"Task-4 shard {relative}",
                    consumer=consume,
                )
                if (
                    size != expected_item_bytes
                    or digest != expected_item_sha
                    or item_digest.hexdigest() != expected_item_sha
                ):
                    raise CorpusContractError(
                        f"Task-4 shard changed during conversion: {relative}"
                    )
                stream_bytes += size
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if stream_bytes != expected_bytes or stream_digest.hexdigest() != expected_sha256:
        raise CorpusContractError(
            f"converted stream does not match Task-4 identity: {destination.name}"
        )


def _canonical_pretty(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _atomic_write_no_replace(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace: {path}")
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        descriptor = os.open(
            temporary.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o644,
            dir_fd=parent_fd,
        )
        try:
            remaining = memoryview(data)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("short write while publishing immutable JSON")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        atomic_rename_noreplace(
            parent_fd,
            temporary.name,
            parent_fd,
            path.name,
        )
        os.fsync(parent_fd)
    except BaseException:
        try:
            os.unlink(temporary.name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(parent_fd)


def _publish_directory_no_replace(temporary: Path, destination: Path) -> None:
    parent_fd = os.open(
        destination.parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        atomic_rename_noreplace(
            parent_fd,
            temporary.name,
            parent_fd,
            destination.name,
        )
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def materialize_135m_layout(
    publication_root: Path | str,
    destination_root: Path | str,
    *,
    source_lock_path: Path | str,
    expected_source_lock_sha256: str = AUTHORITATIVE_RECIPE_SHA256,
    expected_source_receipt_sha256: str | None = None,
    expected_ordered_sha256: str | None = None,
) -> CorpusEvidence:
    """Verify Task 4, concatenate its ordered shards, and publish no-replace."""

    destination = Path(destination_root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination already exists: {destination}")
    evidence = verify_task4_publication(
        publication_root,
        source_lock_path=source_lock_path,
        expected_source_lock_sha256=expected_source_lock_sha256,
        expected_receipt_sha256=expected_source_receipt_sha256,
        expected_ordered_sha256=expected_ordered_sha256,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.task4-bridge-{os.getpid()}"
    )
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"conversion staging path already exists: {temporary}")
    try:
        (temporary / "packed").mkdir(parents=True)
        (temporary / "sidecars").mkdir()
        groups = _publication_artifacts(evidence)
        _concatenate_artifacts(
            evidence.root,
            groups["packed_targets"],
            temporary / STREAM_PATHS["packed_targets"],
            expected_bytes=evidence.raw_target_tokens * 2,
            expected_sha256=evidence.packed_stream_sha256,
        )
        for name in SIDECAR_NAMES:
            _concatenate_artifacts(
                evidence.root,
                groups[name],
                temporary / STREAM_PATHS[name],
                expected_bytes=evidence.raw_target_tokens,
                expected_sha256=evidence.sidecar_stream_sha256[name],
            )
        receipt_bytes = canonical_json_bytes(_bridge_receipt(evidence))
        _atomic_write_no_replace(temporary / "receipt.json", receipt_bytes)
        converted = _verify_flat_layout(
            temporary,
            source_lock_path=Path(source_lock_path),
            expected_source_lock_sha256=expected_source_lock_sha256,
            expected_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
            expected_source_receipt_sha256=evidence.receipt_sha256,
            expected_ordered_sha256=evidence.ordered_stream_sha256,
            expected_packed_sha256=evidence.packed_stream_sha256,
            expected_source_publication=evidence,
        )
        _publish_directory_no_replace(temporary, destination)
    except BaseException:
        if temporary.is_dir() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    published = _verify_flat_layout(
        destination,
        source_lock_path=Path(source_lock_path),
        expected_source_lock_sha256=expected_source_lock_sha256,
        expected_receipt_sha256=converted.receipt_sha256,
        expected_source_receipt_sha256=evidence.receipt_sha256,
        expected_ordered_sha256=evidence.ordered_stream_sha256,
        expected_packed_sha256=evidence.packed_stream_sha256,
        expected_source_publication=evidence,
    )
    return published


def _namespace(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()

    def visit(path: Path, prefix: PurePosixPath) -> None:
        try:
            entries = list(os.scandir(path))
        except OSError as error:
            raise CorpusContractError("dataset namespace cannot be enumerated") from error
        for entry in entries:
            relative = (prefix / entry.name).as_posix()
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise CorpusContractError(
                    f"dataset namespace contains a symlink: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative)
                visit(Path(entry.path), prefix / entry.name)
            elif stat.S_ISREG(metadata.st_mode):
                files.add(relative)
            else:
                raise CorpusContractError(
                    f"dataset namespace contains a special entry: {relative}"
                )

    visit(root, PurePosixPath())
    return files, directories


def _artifact_map(raw: object) -> dict[str, Mapping[str, Any]]:
    if not isinstance(raw, list) or len(raw) != len(STREAM_PATHS):
        raise CorpusContractError("bridge receipt must bind exactly three streams")
    result = {}
    ordered_paths = []
    for item in raw:
        record = _exact_mapping(
            item,
            fields=_ARTIFACT_FIELDS,
            label="bridge artifact",
        )
        path = _portable_relative(record["path"], label="bridge artifact path")
        _positive_int(record["bytes"], label=f"bridge artifact bytes {path}")
        _require_sha256(record["sha256"], label=f"bridge artifact digest {path}")
        if path in result:
            raise CorpusContractError("bridge receipt repeats an artifact path")
        result[path] = record
        ordered_paths.append(path)
    if ordered_paths != sorted(ordered_paths):
        raise CorpusContractError("bridge artifact paths must be sorted")
    if set(result) != set(STREAM_PATHS.values()):
        raise CorpusContractError("bridge receipt stream paths are not canonical")
    return result


def _verify_flat_layout(
    dataset_root: Path,
    *,
    source_lock_path: Path,
    expected_source_lock_sha256: str,
    expected_receipt_sha256: str,
    expected_source_receipt_sha256: str,
    expected_ordered_sha256: str,
    expected_packed_sha256: str,
    expected_source_publication: Task4PublicationEvidence | None = None,
) -> CorpusEvidence:
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise CorpusContractError(f"dataset root is missing or unsafe: {dataset_root}")
    recipe = _load_source_recipe(
        source_lock_path,
        expected_sha256=expected_source_lock_sha256,
    )
    receipt_path = dataset_root / "receipt.json"
    raw, receipt_bytes = _load_json(receipt_path, label="135M bridge receipt")
    receipt = _exact_mapping(
        raw,
        fields=_BRIDGE_FIELDS,
        label="135M bridge receipt",
    )
    if canonical_json_bytes(raw) != receipt_bytes:
        raise CorpusContractError("135M bridge receipt is not canonical JSON")
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    if receipt_sha != _require_sha256(
        expected_receipt_sha256,
        label="expected 135M bridge receipt",
    ):
        raise CorpusContractError("135M bridge receipt SHA-256 mismatch")
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != BRIDGE_RECEIPT_SCHEMA_VERSION
        or receipt["format"] != BRIDGE_FORMAT
        or receipt["contract_id"] != DATASET_CONTRACT_ID
        or receipt["source_recipe_sha256"] != recipe.sha256
        or type(receipt["raw_target_tokens"]) is not int
        or receipt["raw_target_tokens"] != recipe.raw_target_tokens
        or receipt["lane_ids"] != list(recipe.lane_ids)
    ):
        raise CorpusContractError("135M bridge receipt identity or geometry differs")
    task4 = _exact_mapping(
        receipt["task4_publication"],
        fields=_TASK4_BINDING_FIELDS,
        label="Task-4 publication binding",
    )
    source_receipt = _require_sha256(
        task4["receipt_sha256"],
        label="Task-4 source receipt",
    )
    ordered = _require_sha256(
        task4["ordered_stream_sha256"],
        label="Task-4 ordered stream",
    )
    packed = _require_sha256(
        task4["packed_stream_sha256"],
        label="Task-4 packed stream",
    )
    if (
        task4["format"] != TASK4_FORMAT
        or source_receipt
        != _require_sha256(
            expected_source_receipt_sha256,
            label="expected Task-4 source receipt",
        )
        or ordered
        != _require_sha256(
            expected_ordered_sha256,
            label="expected ordered stream",
        )
        or packed
        != _require_sha256(
            expected_packed_sha256,
            label="expected packed stream",
        )
    ):
        raise CorpusContractError("Task-4 source binding differs from the frozen lock")
    for field in ("build_id", "merkle_root_sha256"):
        _require_sha256(task4[field], label=f"Task-4 {field}")
    sidecars = _exact_mapping(
        task4["sidecar_stream_sha256"],
        fields=frozenset(SIDECAR_NAMES),
        label="Task-4 sidecar streams",
    )
    sidecar_digests = {
        name: _require_sha256(sidecars[name], label=f"Task-4 {name}")
        for name in SIDECAR_NAMES
    }
    if expected_source_publication is not None:
        expected_binding = {
            "build_id": expected_source_publication.receipt["build_id"],
            "format": TASK4_FORMAT,
            "merkle_root_sha256": expected_source_publication.receipt[
                "merkle_root_sha256"
            ],
            "ordered_stream_sha256": (
                expected_source_publication.ordered_stream_sha256
            ),
            "packed_stream_sha256": expected_source_publication.packed_stream_sha256,
            "receipt_sha256": expected_source_publication.receipt_sha256,
            "sidecar_stream_sha256": dict(
                expected_source_publication.sidecar_stream_sha256
            ),
        }
        if dict(task4) != expected_binding:
            raise CorpusContractError(
                "135M bridge does not bind every Task-4 source identity"
            )
    artifacts = _artifact_map(receipt["artifacts"])
    expected_artifacts = {
        STREAM_PATHS["packed_targets"]: (recipe.raw_target_tokens * 2, packed),
        **{
            STREAM_PATHS[name]: (recipe.raw_target_tokens, sidecar_digests[name])
            for name in SIDECAR_NAMES
        },
    }
    if any(
        artifacts[path]["bytes"] != byte_count
        or artifacts[path]["sha256"] != digest
        for path, (byte_count, digest) in expected_artifacts.items()
    ):
        raise CorpusContractError("135M bridge artifact declarations differ")
    actual_files, actual_directories = _namespace(dataset_root)
    if actual_files != {"receipt.json", *STREAM_PATHS.values()}:
        raise CorpusContractError("135M dataset contains missing or extra files")
    if actual_directories != {"packed", "sidecars"}:
        raise CorpusContractError("135M dataset contains missing or extra directories")

    stream_digests = {}
    for name, relative in STREAM_PATHS.items():
        require_one = name == "dense_target_weights"
        require_binary = name in SIDECAR_NAMES

        def validate(
            chunk: bytes,
            *,
            stream_name: str = name,
            binary: bool = require_binary,
            one: bool = require_one,
        ) -> None:
            if one and chunk.strip(b"\x01"):
                raise CorpusContractError(
                    "dense_target_weights must be one at every target"
                )
            if binary and not one and chunk.translate(None, b"\x00\x01"):
                raise CorpusContractError(
                    f"{stream_name} contains non-binary target weights"
                )

        size, digest = _stream_regular(
            _safe_member(dataset_root, relative, label=f"{name} stream"),
            label=f"{name} stream",
            consumer=validate,
        )
        expected_size, expected_digest = expected_artifacts[relative]
        if size != expected_size or digest != expected_digest:
            raise CorpusContractError(f"{name} stream differs from its actual hash")
        stream_digests[name] = digest
    return CorpusEvidence(
        contract_id=DATASET_CONTRACT_ID,
        receipt_path=str(receipt_path.resolve(strict=True)),
        receipt_sha256=receipt_sha,
        source_receipt_sha256=source_receipt,
        ordered_token_stream_sha256=ordered,
        packed_stream_sha256=packed,
        stream_sha256=stream_digests,
        raw_target_tokens=recipe.raw_target_tokens,
        lane_ids=recipe.lane_ids,
        semantic_verification_sha256=source_receipt,
        semantic_verification_passed=True,
    )


def load_dataset_pointer(
    path: Path | str,
    *,
    require_frozen: bool = True,
) -> DatasetPointer:
    pointer_path = Path(path)
    raw, _data = _load_json(pointer_path, label="dataset pointer")
    pointer = _exact_mapping(raw, fields=_POINTER_FIELDS, label="dataset pointer")
    if (
        type(pointer["schema_version"]) is not int
        or pointer["schema_version"] != POINTER_SCHEMA_VERSION
        or pointer["contract_id"] != DATASET_CONTRACT_ID
        or pointer["dataset_id"] != DATASET_ID
        or pointer["materialization"] != "filesystem-mirror"
        or pointer["layout_format"] != BRIDGE_FORMAT
        or pointer["relative_path"] != DATASET_RELATIVE_PATH
        or pointer["receipt_relative_path"] != "receipt.json"
        or pointer["required_sidecars"] != list(SIDECAR_NAMES)
        or pointer["source_lock_manifest"] != SOURCE_LOCK_MANIFEST
    ):
        raise CorpusContractError("dataset pointer is not the Slurm v2 bridge")
    for field in ("relative_path", "receipt_relative_path", "source_lock_manifest"):
        _portable_relative(pointer[field], label=f"dataset pointer {field}")
    source_lock_sha = _require_sha256(
        pointer["expected_source_lock_sha256"],
        label="dataset pointer source recipe",
    )
    if source_lock_sha != AUTHORITATIVE_RECIPE_SHA256:
        raise CorpusContractError(
            "dataset pointer does not bind the authoritative upstream recipe"
        )
    mirrors = pointer["mirror_root_environment"]
    if not isinstance(mirrors, Mapping) or dict(mirrors) != MIRROR_ROOT_ENVIRONMENT:
        raise CorpusContractError("dataset pointer mirror environment is invalid")
    status = pointer["launch_gate_status"]
    dynamic = [pointer[field] for field in _DYNAMIC_POINTER_FIELDS]
    if status == "unfrozen":
        if any(value is not None for value in dynamic):
            raise CorpusContractError(
                "unfrozen dataset pointer must not contain speculative hashes"
            )
        if require_frozen:
            raise CorpusContractError(
                "dataset pointer is unfrozen; protected manifests cannot be created"
            )
        return DatasetPointer(
            dataset_id=DATASET_ID,
            contract_id=DATASET_CONTRACT_ID,
            layout_format=BRIDGE_FORMAT,
            relative_path=str(pointer["relative_path"]),
            receipt_relative_path="receipt.json",
            source_lock_manifest=str(pointer["source_lock_manifest"]),
            expected_source_lock_sha256=source_lock_sha,
            expected_receipt_sha256="",
            expected_source_receipt_sha256="",
            expected_ordered_token_stream_sha256="",
            expected_packed_stream_sha256="",
            mirror_root_environment=dict(mirrors),
        )
    if status != "frozen":
        raise CorpusContractError("dataset pointer launch status is invalid")
    locked = {
        field: _require_sha256(pointer[field], label=f"dataset pointer {field}")
        for field in _DYNAMIC_POINTER_FIELDS
    }
    return DatasetPointer(
        dataset_id=DATASET_ID,
        contract_id=DATASET_CONTRACT_ID,
        layout_format=BRIDGE_FORMAT,
        relative_path=str(pointer["relative_path"]),
        receipt_relative_path="receipt.json",
        source_lock_manifest=str(pointer["source_lock_manifest"]),
        expected_source_lock_sha256=source_lock_sha,
        expected_receipt_sha256=locked["expected_receipt_sha256"],
        expected_source_receipt_sha256=locked[
            "expected_source_receipt_sha256"
        ],
        expected_ordered_token_stream_sha256=locked[
            "expected_ordered_token_stream_sha256"
        ],
        expected_packed_stream_sha256=locked["expected_packed_stream_sha256"],
        mirror_root_environment=dict(mirrors),
    )


def verify_dataset_root(
    dataset_root: Path | str,
    *,
    pointer_path: Path | str,
    source_lock_path: Path | str | None = None,
) -> CorpusEvidence:
    pointer_file = Path(pointer_path)
    pointer = load_dataset_pointer(pointer_file)
    lock_path = (
        Path(source_lock_path)
        if source_lock_path is not None
        else pointer_file.parent / pointer.source_lock_manifest
    )
    return _verify_flat_layout(
        Path(dataset_root),
        source_lock_path=lock_path,
        expected_source_lock_sha256=pointer.expected_source_lock_sha256,
        expected_receipt_sha256=pointer.expected_receipt_sha256,
        expected_source_receipt_sha256=pointer.expected_source_receipt_sha256,
        expected_ordered_sha256=pointer.expected_ordered_token_stream_sha256,
        expected_packed_sha256=pointer.expected_packed_stream_sha256,
    )


def freeze_dataset_pointer(
    publication_root: Path | str,
    dataset_root: Path | str,
    *,
    template_path: Path | str,
    output_path: Path | str,
    expected_source_receipt_sha256: str,
    expected_ordered_sha256: str,
    source_lock_path: Path | str | None = None,
) -> Path:
    """Freeze a new pointer only after verifying source and converted bytes."""

    template_file = Path(template_path)
    template = load_dataset_pointer(template_file, require_frozen=False)
    raw, _data = _load_json(template_file, label="dataset pointer template")
    if raw["launch_gate_status"] != "unfrozen":
        raise CorpusContractError("pointer template must be unfrozen")
    lock_path = (
        Path(source_lock_path)
        if source_lock_path is not None
        else template_file.parent / template.source_lock_manifest
    )
    source = verify_task4_publication(
        publication_root,
        source_lock_path=lock_path,
        expected_source_lock_sha256=template.expected_source_lock_sha256,
        expected_receipt_sha256=expected_source_receipt_sha256,
        expected_ordered_sha256=expected_ordered_sha256,
    )
    receipt_path = Path(dataset_root) / "receipt.json"
    receipt_sha = sha256_file(receipt_path)
    converted = _verify_flat_layout(
        Path(dataset_root),
        source_lock_path=lock_path,
        expected_source_lock_sha256=template.expected_source_lock_sha256,
        expected_receipt_sha256=receipt_sha,
        expected_source_receipt_sha256=source.receipt_sha256,
        expected_ordered_sha256=source.ordered_stream_sha256,
        expected_packed_sha256=source.packed_stream_sha256,
        expected_source_publication=source,
    )
    frozen = dict(raw)
    frozen.update(
        {
            "expected_ordered_token_stream_sha256": (
                converted.ordered_token_stream_sha256
            ),
            "expected_packed_stream_sha256": converted.packed_stream_sha256,
            "expected_receipt_sha256": converted.receipt_sha256,
            "expected_source_receipt_sha256": converted.source_receipt_sha256,
            "launch_gate_status": "frozen",
        }
    )
    destination = Path(output_path)
    _atomic_write_no_replace(destination, _canonical_pretty(frozen))
    try:
        load_dataset_pointer(destination)
        verify_dataset_root(
            dataset_root,
            pointer_path=destination,
            source_lock_path=lock_path,
        )
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def stage_dataset_no_replace(
    source_root: Path | str,
    destination_root: Path | str,
    *,
    pointer_path: Path | str,
    source_lock_path: Path | str | None = None,
) -> CorpusEvidence:
    """Copy one frozen flat mirror atomically; never merge or replace a path."""

    source = Path(source_root)
    destination = Path(destination_root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination already exists: {destination}")
    evidence = verify_dataset_root(
        source,
        pointer_path=pointer_path,
        source_lock_path=source_lock_path,
    )
    for member in source.rglob("*"):
        if member.is_symlink():
            raise CorpusContractError(
                f"source mirror contains a symlink: {member.relative_to(source)}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.staging-{os.getpid()}"
    )
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"staging path already exists: {temporary}")
    try:
        shutil.copytree(source, temporary, symlinks=False)
        staged = verify_dataset_root(
            temporary,
            pointer_path=pointer_path,
            source_lock_path=source_lock_path,
        )
        if staged.identity() != evidence.identity():
            raise CorpusContractError("staged mirror identity changed during copy")
        _publish_directory_no_replace(temporary, destination)
    except BaseException:
        if temporary.is_dir() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    return verify_dataset_root(
        destination,
        pointer_path=pointer_path,
        source_lock_path=source_lock_path,
    )
