"""Provider-neutral verification for the complete MemorySplit v2 corpus."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from msctl.cohort import DATASET_CONTRACT_ID, RAW_TARGETS


RECEIPT_SCHEMA_VERSION = 2
POINTER_SCHEMA_VERSION = 1
STREAM_NAMES = (
    "packed_targets",
    "dense_target_weights",
    "split90_target_weights",
)
SIDECAR_NAMES = ("dense_target_weights", "split90_target_weights")
SEMANTIC_ALGORITHM = "memorysplit-v2-semantic-v1"
_HEX = frozenset("0123456789abcdef")
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "production",
        "frozen",
        "complete",
        "raw_target_tokens",
        "ordered_token_stream_sha256",
        "streams",
        "lanes",
        "wikidata",
        "semantic_verification",
    }
)
_POINTER_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "dataset_id",
        "materialization",
        "relative_path",
        "receipt_relative_path",
        "required_sidecars",
        "source_lock_manifest",
        "expected_receipt_sha256",
        "expected_ordered_token_stream_sha256",
        "launch_gate_status",
    }
)
_POINTER_OPTIONAL_FIELDS = frozenset({"mirror_root_environment"})


class CorpusContractError(ValueError):
    """The corpus cannot satisfy the protected launch contract."""


@dataclass(frozen=True)
class DatasetPointer:
    dataset_id: str
    contract_id: str
    relative_path: str
    receipt_relative_path: str
    source_lock_manifest: str
    expected_receipt_sha256: str
    expected_ordered_token_stream_sha256: str
    mirror_root_environment: Mapping[str, str]


@dataclass(frozen=True)
class CorpusEvidence:
    contract_id: str
    receipt_path: str
    receipt_sha256: str
    ordered_token_stream_sha256: str
    stream_sha256: Mapping[str, str]
    raw_target_tokens: int
    lane_ids: tuple[str, ...]
    semantic_verification_sha256: str
    semantic_verification_passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "lane_ids": list(self.lane_ids),
            "ordered_token_stream_sha256": self.ordered_token_stream_sha256,
            "raw_target_tokens": self.raw_target_tokens,
            "receipt_path": self.receipt_path,
            "receipt_sha256": self.receipt_sha256,
            "semantic_verification_passed": self.semantic_verification_passed,
            "semantic_verification_sha256": self.semantic_verification_sha256,
            "stream_sha256": dict(self.stream_sha256),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CorpusContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, *, label: str, maximum_bytes: int = 16 << 20) -> Any:
    if not path.is_file():
        raise CorpusContractError(f"{label} is missing: {path}")
    if path.is_symlink():
        raise CorpusContractError(f"{label} must not be a symlink: {path}")
    data = path.read_bytes()
    if len(data) > maximum_bytes:
        raise CorpusContractError(f"{label} exceeds the maximum supported size")
    try:
        return json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CorpusContractError(f"{label} contains non-finite JSON: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusContractError(f"{label} must be valid UTF-8 JSON") from error


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
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != value
    ):
        raise CorpusContractError(f"{label} must be a portable relative path")
    return value


def _safe_member(root: Path, relative: str, *, label: str) -> Path:
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise CorpusContractError(f"{label} traverses a symlink: {relative}")
    try:
        current.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as error:
        raise CorpusContractError(
            f"{label} is missing or escapes the dataset root: {relative}"
        ) from error
    if not current.is_file():
        raise CorpusContractError(f"{label} is not a regular file: {relative}")
    return current


def _load_source_lock(path: Path) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    raw = _load_json(path, label="reasoning dataset source lock")
    if not isinstance(raw, Mapping):
        raise CorpusContractError("reasoning dataset source lock must be an object")
    if raw.get("schema_version") != 2:
        raise CorpusContractError("reasoning dataset source lock schema is not v2")
    if raw.get("contract_id") != "memorysplit-reasoning-dataset-v2":
        raise CorpusContractError("reasoning dataset source lock id is wrong")
    recipe = raw.get("sprint_recipe")
    if not isinstance(recipe, Mapping):
        raise CorpusContractError("reasoning dataset sprint recipe is missing")
    if (
        recipe.get("raw_target_tokens") != RAW_TARGETS
        or recipe.get("all_lanes_required") is not True
        or recipe.get("wikidata_coverage") != "complete_once"
        or recipe.get("sidecars") != list(SIDECAR_NAMES)
    ):
        raise CorpusContractError("reasoning dataset recipe is not the complete v2 recipe")
    lanes = recipe.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != 8:
        raise CorpusContractError("reasoning dataset recipe must freeze all eight lanes")
    normalized = []
    seen = set()
    for index, lane in enumerate(lanes):
        if not isinstance(lane, Mapping):
            raise CorpusContractError(f"source lane {index} is not an object")
        allowed = {"id", "share_percent", "source_lock_sha256", "complete_once"}
        if set(lane) - allowed or not {"id", "share_percent", "source_lock_sha256"} <= set(lane):
            raise CorpusContractError(f"source lane {index} has invalid fields")
        lane_id = lane["id"]
        if (
            not isinstance(lane_id, str)
            or not lane_id
            or lane_id in seen
        ):
            raise CorpusContractError("source lane ids must be unique strings")
        seen.add(lane_id)
        share = lane["share_percent"]
        if (
            isinstance(share, bool)
            or not isinstance(share, (int, float))
            or share <= 0
        ):
            raise CorpusContractError(f"source lane {lane_id} has invalid share")
        digest = _require_sha256(
            lane["source_lock_sha256"],
            label=f"source lane {lane_id} lock",
        )
        if lane_id == "wikidata_graph" and lane.get("complete_once") is not True:
            raise CorpusContractError("Wikidata source lane is not complete-once")
        normalized.append(
            {
                "id": lane_id,
                "share_percent": float(share),
                "source_lock_sha256": digest,
            }
        )
    if abs(sum(item["share_percent"] for item in normalized) - 100.0) > 1e-9:
        raise CorpusContractError("reasoning dataset lane shares must total 100")
    return normalized, tuple(item["id"] for item in normalized)


def _validate_streams(
    root: Path,
    raw_streams: object,
    *,
    ordered_sha256: str,
) -> dict[str, str]:
    streams = _exact_mapping(
        raw_streams,
        fields=frozenset(STREAM_NAMES),
        label="receipt streams",
    )
    digests = {}
    for name in STREAM_NAMES:
        stream = _exact_mapping(
            streams[name],
            fields=frozenset({"path", "sha256", "targets"}),
            label=f"{name} stream",
        )
        relative = _portable_relative(stream["path"], label=f"{name} path")
        expected_path = {
            "packed_targets": "packed/targets.bin",
            "dense_target_weights": "sidecars/dense_target_weights.bin",
            "split90_target_weights": "sidecars/split90_target_weights.bin",
        }[name]
        if relative != expected_path:
            raise CorpusContractError(f"{name} path is not canonical")
        if stream["targets"] != RAW_TARGETS or isinstance(stream["targets"], bool):
            raise CorpusContractError(f"{name} target count is not the complete horizon")
        expected = _require_sha256(stream["sha256"], label=f"{name} stream")
        path = _safe_member(root, relative, label=f"{name} stream")
        actual = sha256_file(path)
        if actual != expected:
            raise CorpusContractError(f"{name} stream SHA-256 does not match receipt")
        digests[name] = actual
    if digests["packed_targets"] != ordered_sha256:
        raise CorpusContractError(
            "packed stream does not match the ordered token-stream identity"
        )
    return digests


def _validate_semantic_evidence(
    root: Path,
    raw: object,
    *,
    lane_ids: tuple[str, ...],
    ordered_sha256: str,
) -> str:
    semantic = _exact_mapping(
        raw,
        fields=frozenset(
            {
                "algorithm",
                "evidence_path",
                "evidence_sha256",
                "passed",
            }
        ),
        label="semantic verification",
    )
    if (
        semantic["algorithm"] != SEMANTIC_ALGORITHM
        or semantic["passed"] is not True
    ):
        raise CorpusContractError("semantic verification is not frozen and passing")
    relative = _portable_relative(
        semantic["evidence_path"],
        label="semantic evidence path",
    )
    if relative != "semantic-verification.json":
        raise CorpusContractError("semantic evidence path is not canonical")
    path = _safe_member(root, relative, label="semantic evidence")
    expected_sha = _require_sha256(
        semantic["evidence_sha256"],
        label="semantic evidence",
    )
    if sha256_file(path) != expected_sha:
        raise CorpusContractError("semantic evidence SHA-256 does not match")
    evidence = _load_json(path, label="semantic evidence")
    expected = {
        "contract_id": DATASET_CONTRACT_ID,
        "lane_ids": list(lane_ids),
        "ordered_token_stream_sha256": ordered_sha256,
        "packed_target_count": RAW_TARGETS,
        "passed": True,
        "schema_version": 1,
        "sidecar_alignment": {
            "dense_target_weights": RAW_TARGETS,
            "split90_target_weights": RAW_TARGETS,
        },
        "source_locks_verified": True,
        "wikidata_complete_once": True,
    }
    if evidence != expected:
        raise CorpusContractError(
            "semantic evidence does not prove the complete deterministic corpus"
        )
    return expected_sha


def verify_canonical_corpus(
    receipt_path: Path | str,
    *,
    expected_sha256: str,
    expected_ordered_sha256: str,
    source_lock_path: Path | str,
    semantic_verifier: Callable[[Path], Mapping[str, object]] | None = None,
) -> CorpusEvidence:
    """Verify a production receipt and every protected stream, fail closed."""

    expected_receipt = _require_sha256(
        expected_sha256,
        label="expected receipt",
    )
    expected_ordered = _require_sha256(
        expected_ordered_sha256,
        label="expected ordered token stream",
    )
    receipt = Path(receipt_path)
    raw = _exact_mapping(
        _load_json(receipt, label="corpus receipt"),
        fields=_RECEIPT_FIELDS,
        label="corpus receipt",
    )
    actual_receipt = sha256_file(receipt)
    if actual_receipt != expected_receipt:
        raise CorpusContractError("corpus receipt SHA-256 does not match its frozen lock")
    if (
        raw["schema_version"] != RECEIPT_SCHEMA_VERSION
        or isinstance(raw["schema_version"], bool)
        or raw["contract_id"] != DATASET_CONTRACT_ID
    ):
        raise CorpusContractError("corpus receipt is not the canonical v2 contract")
    for field in ("production", "frozen", "complete"):
        if raw[field] is not True:
            raise CorpusContractError(f"corpus receipt {field} flag must be true")
    if raw["raw_target_tokens"] != RAW_TARGETS or isinstance(
        raw["raw_target_tokens"], bool
    ):
        raise CorpusContractError("corpus receipt target horizon is not complete")
    ordered = _require_sha256(
        raw["ordered_token_stream_sha256"],
        label="receipt ordered token stream",
    )
    if ordered != expected_ordered:
        raise CorpusContractError("ordered token-stream identity is not frozen")

    expected_lanes, lane_ids = _load_source_lock(Path(source_lock_path))
    raw_lanes = raw["lanes"]
    if not isinstance(raw_lanes, list):
        raise CorpusContractError("corpus receipt lanes must be a list")
    normalized_lanes = []
    for lane in raw_lanes:
        item = _exact_mapping(
            lane,
            fields=frozenset({"id", "share_percent", "source_lock_sha256"}),
            label="receipt lane",
        )
        normalized_lanes.append(
            {
                "id": item["id"],
                "share_percent": float(item["share_percent"]),
                "source_lock_sha256": item["source_lock_sha256"],
            }
        )
    if normalized_lanes != expected_lanes:
        raise CorpusContractError(
            "corpus receipt does not contain all eight frozen source lanes"
        )
    wikidata = _exact_mapping(
        raw["wikidata"],
        fields=frozenset({"lane_id", "complete_once", "duplicate_entities"}),
        label="Wikidata receipt",
    )
    if wikidata != {
        "lane_id": "wikidata_graph",
        "complete_once": True,
        "duplicate_entities": 0,
    }:
        raise CorpusContractError("Wikidata coverage is not complete exactly once")

    root = receipt.parent
    stream_digests = _validate_streams(
        root,
        raw["streams"],
        ordered_sha256=ordered,
    )
    semantic_sha = _validate_semantic_evidence(
        root,
        raw["semantic_verification"],
        lane_ids=lane_ids,
        ordered_sha256=ordered,
    )
    if semantic_verifier is not None:
        replay = semantic_verifier(root)
        if not isinstance(replay, Mapping) or replay.get("passed") is not True:
            raise CorpusContractError("semantic verifier replay did not pass")
        if replay.get("ordered_token_stream_sha256") != ordered:
            raise CorpusContractError("semantic verifier replay stream identity differs")

    return CorpusEvidence(
        contract_id=DATASET_CONTRACT_ID,
        receipt_path=str(receipt.resolve()),
        receipt_sha256=actual_receipt,
        ordered_token_stream_sha256=ordered,
        stream_sha256=stream_digests,
        raw_target_tokens=RAW_TARGETS,
        lane_ids=lane_ids,
        semantic_verification_sha256=semantic_sha,
        semantic_verification_passed=True,
    )


def load_dataset_pointer(
    path: Path | str,
    *,
    require_frozen: bool = True,
) -> DatasetPointer:
    pointer_path = Path(path)
    raw = _load_json(pointer_path, label="dataset pointer")
    if not isinstance(raw, Mapping):
        raise CorpusContractError("dataset pointer must be an object")
    fields = set(raw)
    missing = sorted(_POINTER_REQUIRED_FIELDS - fields)
    unknown = sorted(
        fields - _POINTER_REQUIRED_FIELDS - _POINTER_OPTIONAL_FIELDS
    )
    if missing or unknown:
        raise CorpusContractError(
            f"dataset pointer schema mismatch; missing={missing}, unknown={unknown}"
        )
    if (
        raw["schema_version"] != POINTER_SCHEMA_VERSION
        or isinstance(raw["schema_version"], bool)
        or raw["contract_id"] != DATASET_CONTRACT_ID
        or raw["materialization"] != "filesystem-mirror"
        or raw["required_sidecars"] != list(SIDECAR_NAMES)
    ):
        raise CorpusContractError("dataset pointer is not the Slurm v2 contract")
    for field in ("relative_path", "receipt_relative_path", "source_lock_manifest"):
        _portable_relative(raw[field], label=f"dataset pointer {field}")
    if raw["receipt_relative_path"] != "receipt.json":
        raise CorpusContractError("dataset pointer receipt path is not canonical")
    mirrors = raw.get("mirror_root_environment", {})
    if not isinstance(mirrors, Mapping) or any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or not value
        for key, value in mirrors.items()
    ):
        raise CorpusContractError("dataset pointer mirror environment is invalid")
    if require_frozen and raw["launch_gate_status"] != "frozen":
        raise CorpusContractError(
            "dataset pointer is unfrozen; protected manifests cannot be created"
        )
    receipt_sha = _require_sha256(
        raw["expected_receipt_sha256"],
        label="dataset pointer receipt",
    )
    ordered_sha = _require_sha256(
        raw["expected_ordered_token_stream_sha256"],
        label="dataset pointer ordered stream",
    )
    return DatasetPointer(
        dataset_id=str(raw["dataset_id"]),
        contract_id=DATASET_CONTRACT_ID,
        relative_path=str(raw["relative_path"]),
        receipt_relative_path=str(raw["receipt_relative_path"]),
        source_lock_manifest=str(raw["source_lock_manifest"]),
        expected_receipt_sha256=receipt_sha,
        expected_ordered_token_stream_sha256=ordered_sha,
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
    root = Path(dataset_root)
    if not root.is_dir() or root.is_symlink():
        raise CorpusContractError(f"dataset root is missing or unsafe: {root}")
    lock_path = (
        Path(source_lock_path)
        if source_lock_path is not None
        else pointer_file.parent / pointer.source_lock_manifest
    )
    return verify_canonical_corpus(
        root / pointer.receipt_relative_path,
        expected_sha256=pointer.expected_receipt_sha256,
        expected_ordered_sha256=pointer.expected_ordered_token_stream_sha256,
        source_lock_path=lock_path,
    )


def stage_dataset_no_replace(
    source_root: Path | str,
    destination_root: Path | str,
    *,
    pointer_path: Path | str,
    source_lock_path: Path | str | None = None,
) -> CorpusEvidence:
    """Copy one verified mirror atomically; never merge with or replace a path."""

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
        if staged.receipt_sha256 != evidence.receipt_sha256:
            raise CorpusContractError("staged mirror identity changed during copy")
        os.rename(temporary, destination)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    return verify_dataset_root(
        destination,
        pointer_path=pointer_path,
        source_lock_path=source_lock_path,
    )
