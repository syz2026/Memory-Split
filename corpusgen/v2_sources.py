"""Deterministic source staging for the authoritative MemorySplit v2 corpus."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import mmap
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import tarfile
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from corpusgen.wikidata5m import SourceDriftError, iter_triples, size_and_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V2_SOURCE_LOCK = (
    ROOT / "sources" / "memorysplit-v2" / "source-set.lock.json"
)
DEFAULT_RESERVE_BYTES = 8 * 1024**3
RECEIPT_NAME = "source-stage-receipt.json"

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_EXPECTED_LOCK_IDS = (
    "fineweb_edu",
    "finemath",
    "wikidata5m",
    "objective_auxiliary",
)
_EXPECTED_AUXILIARY_IDS = (
    "deepmind_mathematics_generator",
    "clrs_text",
    "ruletaker",
    "prontoqa",
    "reasoning_gym_exact_answer",
    "arc_agi_training",
    "conceptarc_training",
)
_EXPECTED_MISSING_SOURCE_LOCKS = (
    "synthetic_graph_generator",
    "verified_synthetic_multihop_generator",
    "wikidata_path_reasoning_generator",
    "relational_refinement_generator",
    "reasoning_solver",
)
_EXPECTED_MISSING_MATERIALIZED_LANES = (
    "fineweb_edu",
    "finemath",
    "wikidata_graph",
    "synthetic_graph",
    "verified_synthetic_multihop",
    "wikidata_path_reasoning",
    "relational_refinement",
    "objective_auxiliary",
)
_EXPECTED_HF_PINS = {
    "fineweb_edu": (
        "HuggingFaceFW/fineweb-edu",
        "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "ODC-By-1.0",
    ),
    "finemath": (
        "HuggingFaceTB/finemath",
        "e92b25a616738fe95dc186b64dfb19f9c8525594",
        "ODC-By-1.0",
    ),
    "wikidata5m": (
        "intfloat/wikidata5m",
        "6b2b09672129e280c0c9da97ab58154e9d535e6b",
        "CC0-1.0",
    ),
}
_WIKIDATA_SEALED_SPLITS = (
    "wikidata5m_inductive_test.txt",
    "wikidata5m_inductive_valid.txt",
    "wikidata5m_transductive_test.txt",
    "wikidata5m_transductive_valid.txt",
)
_WIKIDATA_TRAINING_INDEX_STRUCT = struct.Struct(">QQQ")
_WIKIDATA_TRAINING_INDEX_ENCODING = "three_big_endian_uint64_subject_relation_object"
_WIKIDATA_SEALED_POLICY = (
    "retain_official_sources_exclude_exact_training_overlaps_from_evaluation"
)


class V2SourceStageError(RuntimeError):
    """The v2 source set cannot be staged without violating its lock."""


class InsufficientDiskError(V2SourceStageError):
    """The destination filesystem cannot hold the locked source stage."""


@dataclass(frozen=True)
class V2SourceSetLock:
    path: Path
    sha256: str
    raw: Mapping[str, Any]
    sources: Mapping[str, Mapping[str, Any]]
    source_paths: Mapping[str, Path]
    scientific_requirements: Mapping[str, Any]

    @property
    def dataset_id(self) -> str:
        return str(self.raw["dataset_id"])

    @property
    def contract_id(self) -> str:
        return str(self.raw["contract_id"])

    @property
    def production_readiness(self) -> Mapping[str, Any]:
        return _mapping(
            self.raw["production_readiness"],
            "v2 source-set production_readiness",
        )

    @property
    def download_bytes(self) -> int:
        total = sum(
            int(self.sources[name]["total_bytes"])
            for name in ("fineweb_edu", "finemath", "wikidata5m")
        )
        auxiliary = self.sources["objective_auxiliary"]
        return total + sum(
            int(component["archive"]["bytes"])
            for source in auxiliary["sources"]
            for component in source["components"]
        )


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, *, maximum_bytes: int = 32 << 20) -> Any:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing regular JSON file: {path}")
    content = path.read_bytes()
    if len(content) > maximum_bytes:
        raise ValueError(f"JSON file exceeds {maximum_bytes} bytes: {path}")
    try:
        return json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid UTF-8 JSON: {path}") from error


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(
            f"{label} fields differ from lock schema; "
            f"missing={missing}, extra={extra}"
        )


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _sha1(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 40-hex object id")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _relative_path(value: Any, label: str, *, allow_parent: bool = False) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} must be a nonempty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts:
        raise ValueError(f"unsafe {label}: {value!r}")
    if not allow_parent and ".." in path.parts:
        raise ValueError(f"unsafe {label}: {value!r}")
    return path.as_posix()


def _validate_file_records(
    value: Any,
    label: str,
    *,
    require_blob: bool,
    extra_fields: set[str] | None = None,
) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    expected_fields = {"bytes", "path", "sha256"}
    if require_blob:
        expected_fields.add("git_blob_sha1")
    expected_fields.update(extra_fields or set())
    records: list[Mapping[str, Any]] = []
    previous = ""
    for index, raw in enumerate(value):
        record = _mapping(raw, f"{label}[{index}]")
        _exact_fields(record, expected_fields, f"{label}[{index}]")
        path = _relative_path(record["path"], f"{label}[{index}].path")
        if path <= previous:
            raise ValueError(f"{label} paths must be unique and sorted")
        previous = path
        _nonnegative_int(record["bytes"], f"{label}[{index}].bytes")
        _sha256(record["sha256"], f"{label}[{index}].sha256")
        if require_blob:
            _sha1(record["git_blob_sha1"], f"{label}[{index}].git_blob_sha1")
        records.append(record)
    return records


def _validate_huggingface_lock(
    source_id: str,
    value: Mapping[str, Any],
) -> None:
    common = {
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
    extra = (
        {"selection"}
        if source_id == "fineweb_edu"
        else {"deduplication", "selection", "subset_metadata"}
        if source_id == "finemath"
        else {"complete_once"}
    )
    _exact_fields(value, common | extra, f"{source_id} lock")
    if (
        value["format"] != "memorysplit-v2-huggingface-source-lock"
        or value["schema_version"] != 1
        or value["source_id"] != source_id
        or value["repo_type"] != "dataset"
    ):
        raise ValueError(f"{source_id} lock identity is invalid")
    expected_repository, expected_revision, expected_license = _EXPECTED_HF_PINS[
        source_id
    ]
    if (
        not Path(str(value["repository"])).is_dir()
        and (
            value["repository"] != expected_repository
            or value["revision"] != expected_revision
        )
    ) or value["license"] != expected_license:
        raise ValueError(f"{source_id} upstream pin or license is invalid")
    _sha1(value["revision"], f"{source_id} revision")
    license_evidence = _mapping(
        value["license_evidence"],
        f"{source_id} license evidence",
    )
    if source_id in {"fineweb_edu", "finemath"}:
        _exact_fields(
            license_evidence,
            {"dataset_card_value", "terms_url"},
            f"{source_id} license evidence",
        )
        if license_evidence != {
            "dataset_card_value": "odc-by",
            "terms_url": "https://commoncrawl.org/terms-of-use",
        }:
            raise ValueError(f"{source_id} ODC-By evidence is invalid")
    else:
        _exact_fields(
            license_evidence,
            {"notice_path", "notice_sha256"},
            "Wikidata license evidence",
        )
        _relative_path(
            license_evidence["notice_path"],
            "Wikidata license notice path",
            allow_parent=True,
        )
        _sha256(
            license_evidence["notice_sha256"],
            "Wikidata license notice sha256",
        )
    files = _validate_file_records(
        value["files"],
        f"{source_id}.files",
        require_blob=True,
        extra_fields={"members"} if source_id == "wikidata5m" else None,
    )
    if sum(int(item["bytes"]) for item in files) != _positive_int(
        value["total_bytes"],
        f"{source_id}.total_bytes",
    ):
        raise ValueError(f"{source_id} total_bytes does not match its file records")

    if source_id == "fineweb_edu":
        selection = _mapping(value["selection"], "FineWeb selection")
        if (
            selection.get("config") != "sample-10BT"
            or selection.get("scope") != "complete_config"
            or selection.get("file_order") != "lexicographic_path"
            or len(files) != 14
            or any(
                not item["path"].startswith("sample/10BT/") for item in files
            )
        ):
            raise ValueError("FineWeb lock is not the complete sample-10BT config")
    elif source_id == "finemath":
        selection = _mapping(value["selection"], "FineMath selection")
        if selection.get("subsets_in_order") != [
            "finemath-4plus",
            "finemath-3plus-cross-deduplicated-remainder",
        ]:
            raise ValueError("FineMath subset order differs from the v2 requirement")
        deduplication = _mapping(value["deduplication"], "FineMath deduplication")
        if deduplication != {
            "digest": "sha256",
            "encoding": "utf-8",
            "field": "text",
            "identity": "exact_text_bytes",
            "priority": [
                "fineweb_edu",
                "finemath-4plus",
                "finemath-3plus",
            ],
            "sidecar_encoding": (
                "one_unsigned_byte_per_source_row_1_keep_0_drop"
            ),
        }:
            raise ValueError("FineMath deduplication contract is invalid")
        metadata = _mapping(value["subset_metadata"], "FineMath subset metadata")
        if set(metadata) != {"finemath-3plus", "finemath-4plus"}:
            raise ValueError("FineMath subset metadata is incomplete")
        for name, item in metadata.items():
            details = _mapping(item, f"FineMath subset {name}")
            _exact_fields(details, {"download_bytes", "rows"}, f"FineMath {name}")
            _positive_int(details["download_bytes"], f"FineMath {name} bytes")
            _positive_int(details["rows"], f"FineMath {name} rows")
    else:
        complete_once = _mapping(value["complete_once"], "Wikidata complete_once")
        if complete_once != {
            "sidecar_encoding": (
                "one_unsigned_byte_per_source_row_1_keep_0_drop"
            ),
            "split_order": [
                "wikidata5m_inductive_train.txt",
                "wikidata5m_transductive_train.txt",
            ],
            "triple_identity": "exact_subject_relation_object",
        }:
            raise ValueError("Wikidata complete-once contract is invalid")
        archive_paths = {
            "wikidata5m_alias.tar.gz",
            "wikidata5m_inductive.tar.gz",
            "wikidata5m_transductive.tar.gz",
        }
        if {item["path"] for item in files} != archive_paths:
            raise ValueError("Wikidata lock must cover exactly the three graph archives")
        for item in value["files"]:
            # Wikidata records add a member inventory secured by the archive hash.
            if set(item) != {
                "bytes",
                "git_blob_sha1",
                "members",
                "path",
                "sha256",
            }:
                raise ValueError("Wikidata archive record fields are invalid")
            members = item["members"]
            if not isinstance(members, list) or not members:
                raise ValueError("Wikidata archive members are missing")
            previous = ""
            for member in members:
                member = _mapping(member, "Wikidata archive member")
                _exact_fields(
                    member,
                    {"bytes", "path", "sha256"},
                    "Wikidata archive member",
                )
                path = _relative_path(member["path"], "Wikidata archive member")
                if path <= previous:
                    raise ValueError("Wikidata archive members must be sorted")
                previous = path
                _positive_int(member["bytes"], f"Wikidata member {path} bytes")
                _sha256(member["sha256"], f"Wikidata member {path} sha256")


def _validate_auxiliary_lock(
    value: Mapping[str, Any],
    expected_ids: Sequence[str],
) -> None:
    _exact_fields(value, {"format", "schema_version", "sources"}, "auxiliary lock")
    if (
        value["format"] != "memorysplit-v2-objective-auxiliary-lock"
        or value["schema_version"] != 1
    ):
        raise ValueError("objective auxiliary lock identity is invalid")
    sources = value["sources"]
    if not isinstance(sources, list) or [
        item.get("id") if isinstance(item, Mapping) else None for item in sources
    ] != list(expected_ids):
        raise ValueError("objective auxiliary sources differ from scientific policy")
    component_ids: set[str] = set()
    for source_index, raw_source in enumerate(sources):
        source = _mapping(raw_source, f"auxiliary source {source_index}")
        _exact_fields(
            source,
            {"components", "id", "training_use"},
            f"auxiliary source {source_index}",
        )
        if not isinstance(source["training_use"], str) or not source["training_use"]:
            raise ValueError("objective auxiliary training_use must be explicit")
        components = source["components"]
        if not isinstance(components, list) or not components:
            raise ValueError("objective auxiliary source has no components")
        for component_index, raw_component in enumerate(components):
            label = f"auxiliary {source['id']} component {component_index}"
            component = _mapping(raw_component, label)
            _exact_fields(
                component,
                {
                    "archive",
                    "commit",
                    "files",
                    "id",
                    "license",
                    "license_paths",
                    "repository",
                },
                label,
            )
            component_id = component["id"]
            if (
                not isinstance(component_id, str)
                or not component_id
                or component_id in component_ids
            ):
                raise ValueError("objective component ids must be unique strings")
            component_ids.add(component_id)
            _sha1(component["commit"], f"{label}.commit")
            if (
                not isinstance(component["repository"], str)
                or not component["repository"].startswith("https://github.com/")
            ):
                raise ValueError(f"{label}.repository must be a GitHub HTTPS URL")
            if not isinstance(component["license"], str) or not component["license"]:
                raise ValueError(f"{label}.license must be explicit")
            archive = _mapping(component["archive"], f"{label}.archive")
            _exact_fields(
                archive,
                {"bytes", "root", "sha256", "url"},
                f"{label}.archive",
            )
            _positive_int(archive["bytes"], f"{label}.archive.bytes")
            _sha256(archive["sha256"], f"{label}.archive.sha256")
            _relative_path(archive["root"], f"{label}.archive.root")
            if (
                not isinstance(archive["url"], str)
                or not (
                    archive["url"].startswith("https://codeload.github.com/")
                    or archive["url"].startswith("file://")
                )
            ):
                raise ValueError(
                    f"{label}.archive.url must be a codeload or local fixture URL"
                )
            records = _validate_file_records(
                component["files"],
                f"{label}.files",
                require_blob=False,
            )
            license_paths = component["license_paths"]
            if not isinstance(license_paths, list) or not license_paths:
                raise ValueError(f"{label}.license_paths must be nonempty")
            known = {record["path"] for record in records}
            for index, path in enumerate(license_paths):
                path = _relative_path(path, f"{label}.license_paths[{index}]")
                if path not in known:
                    raise ValueError(f"{label} does not lock license file {path}")


def load_v2_source_lock(
    path: Path | str = DEFAULT_V2_SOURCE_LOCK,
) -> V2SourceSetLock:
    lock_path = Path(path).resolve()
    raw = _mapping(_read_json(lock_path), "v2 source-set lock")
    _exact_fields(
        raw,
        {
            "contract_id",
            "dataset_id",
            "format",
            "production_readiness",
            "schema_version",
            "scientific_requirements",
            "source_locks",
        },
        "v2 source-set lock",
    )
    if (
        raw["format"] != "memorysplit-v2-source-set-lock"
        or raw["schema_version"] != 1
        or raw["contract_id"] != "memorysplit-reasoning-dataset-v2"
        or raw["dataset_id"] != "memorysplit-v2-frozen-upstream-sources"
    ):
        raise ValueError("v2 source-set lock identity is invalid")
    production_readiness = _mapping(
        raw["production_readiness"],
        "v2 source-set production_readiness",
    )
    _exact_fields(
        production_readiness,
        {
            "missing_materialized_lanes",
            "missing_source_locks",
            "ready",
            "reason",
        },
        "v2 source-set production_readiness",
    )
    if (
        production_readiness["ready"] is not False
        or production_readiness["missing_source_locks"]
        != list(_EXPECTED_MISSING_SOURCE_LOCKS)
        or production_readiness["missing_materialized_lanes"]
        != list(_EXPECTED_MISSING_MATERIALIZED_LANES)
        or not isinstance(production_readiness["reason"], str)
        or not production_readiness["reason"]
    ):
        raise ValueError(
            "v2 source-set must explicitly remain production-incomplete"
        )

    scientific_reference = _mapping(
        raw["scientific_requirements"],
        "scientific requirements reference",
    )
    _exact_fields(
        scientific_reference,
        {"path", "sha256"},
        "scientific requirements reference",
    )
    scientific_relative = _relative_path(
        scientific_reference["path"],
        "scientific requirements path",
        allow_parent=True,
    )
    scientific_path = (lock_path.parent / scientific_relative).resolve()
    expected_scientific_sha = _sha256(
        scientific_reference["sha256"],
        "scientific requirements sha256",
    )
    if size_and_sha256(scientific_path)[1] != expected_scientific_sha:
        raise ValueError("scientific requirements changed after source lock freeze")
    scientific = _mapping(_read_json(scientific_path), "scientific requirements")
    try:
        policy = scientific["sprint_recipe"]["source_policy"]
    except (KeyError, TypeError) as error:
        raise ValueError("scientific source policy is missing") from error
    policy = _mapping(policy, "scientific source policy")
    if policy.get("finemath_subsets_in_order") != [
        "finemath-4plus",
        "finemath-3plus-cross-deduplicated-remainder",
    ] or policy.get("cross_deduplicate_finemath_against_fineweb") is not True:
        raise ValueError("scientific FineMath source requirement is not frozen")
    expected_auxiliary = policy.get("objective_auxiliary_sources")
    if expected_auxiliary != list(_EXPECTED_AUXILIARY_IDS):
        raise ValueError("scientific objective auxiliary source list is not frozen")

    references = raw["source_locks"]
    if not isinstance(references, list) or [
        item.get("id") if isinstance(item, Mapping) else None for item in references
    ] != list(_EXPECTED_LOCK_IDS):
        raise ValueError("source-set child locks must use the frozen order")
    sources: dict[str, Mapping[str, Any]] = {}
    source_paths: dict[str, Path] = {}
    for index, raw_reference in enumerate(references):
        reference = _mapping(raw_reference, f"source lock reference {index}")
        _exact_fields(
            reference,
            {"id", "path", "sha256"},
            f"source lock reference {index}",
        )
        source_id = reference["id"]
        relative = _relative_path(reference["path"], f"{source_id} lock path")
        child_path = lock_path.parent.joinpath(*PurePosixPath(relative).parts)
        expected = _sha256(reference["sha256"], f"{source_id} lock sha256")
        actual = size_and_sha256(child_path)[1]
        if actual != expected:
            raise ValueError(
                f"{source_id} child lock drift: expected {expected}, got {actual}"
            )
        child = _mapping(_read_json(child_path), f"{source_id} child lock")
        if source_id == "objective_auxiliary":
            _validate_auxiliary_lock(child, expected_auxiliary)
        else:
            _validate_huggingface_lock(source_id, child)
        sources[source_id] = child
        source_paths[source_id] = child_path.resolve()

    content = lock_path.read_bytes()
    return V2SourceSetLock(
        path=lock_path,
        sha256=hashlib.sha256(content).hexdigest(),
        raw=raw,
        sources=sources,
        source_paths=source_paths,
        scientific_requirements={
            "path": scientific_path,
            "sha256": expected_scientific_sha,
        },
    )


def _nearest_existing(path: Path) -> Path:
    candidate = path.resolve(strict=False)
    while not candidate.exists():
        if candidate == candidate.parent:
            raise V2SourceStageError(f"no existing parent for disk preflight: {path}")
        candidate = candidate.parent
    return candidate


def _locked_derived_bytes(lock: V2SourceSetLock) -> int:
    wikidata = sum(
        int(member["bytes"])
        for archive in lock.sources["wikidata5m"]["files"]
        for member in archive["members"]
    )
    auxiliary = sum(
        int(record["bytes"])
        for source in lock.sources["objective_auxiliary"]["sources"]
        for component in source["components"]
        for record in component["files"]
    )
    finemath_sidecars = sum(
        int(item["rows"])
        for item in lock.sources["finemath"]["subset_metadata"].values()
    )
    # Reserve a second uncompressed-Wikidata allowance for complete-once
    # sidecars, the fixed-width exact training index, and sealed-exclusion
    # evidence. It is deliberately conservative because row counts are learned
    # only after the pinned archives have been parsed.
    return (2 * wikidata) + auxiliary + finemath_sidecars


def _staging_paths(
    lock: V2SourceSetLock,
    data_root: Path,
) -> dict[str, Path]:
    stage_root = data_root / ".memorysplit-v2-source-stage" / lock.sha256
    return {
        "final": data_root / lock.dataset_id,
        "stage": stage_root,
        "payload": stage_root / "payload",
        "downloads": stage_root / "downloads",
        "state": stage_root / "state",
    }


def _hf_command(
    source: Mapping[str, Any],
    destination: Path,
    hf_command: str,
    max_workers: int,
) -> list[str]:
    return [
        hf_command,
        "download",
        str(source["repository"]),
        *[str(item["path"]) for item in source["files"]],
        "--repo-type",
        str(source["repo_type"]),
        "--revision",
        str(source["revision"]),
        "--local-dir",
        str(destination),
        "--max-workers",
        str(max_workers),
    ]


def _path_is_complete(path: Path, record: Mapping[str, Any]) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == record["bytes"]
    )


def _remaining_download_bytes(
    lock: V2SourceSetLock,
    paths: Mapping[str, Path],
) -> int:
    remaining = 0
    for source_id in ("fineweb_edu", "finemath", "wikidata5m"):
        source = lock.sources[source_id]
        destination = paths["payload"] / source_id
        for record in source["files"]:
            path = destination.joinpath(*PurePosixPath(record["path"]).parts)
            if not _path_is_complete(path, record):
                remaining += int(record["bytes"])
    for source in lock.sources["objective_auxiliary"]["sources"]:
        for component in source["components"]:
            archive = component["archive"]
            path = paths["downloads"] / "objective" / f"{component['id']}.tar.gz"
            if not _path_is_complete(path, archive):
                remaining += int(archive["bytes"])
    return remaining


def plan_v2_source_stage(
    lock: V2SourceSetLock,
    data_root: Path | str,
    *,
    cache_dir: Path | str | None = None,
    hf_command: str = "hf",
    max_workers: int = 8,
    reserve_bytes: int = DEFAULT_RESERVE_BYTES,
    disk_free_bytes: int | None = None,
) -> dict[str, Any]:
    _positive_int(max_workers, "max_workers")
    _nonnegative_int(reserve_bytes, "reserve_bytes")
    if disk_free_bytes is not None:
        _nonnegative_int(disk_free_bytes, "disk_free_bytes")
    if not isinstance(hf_command, str) or not hf_command:
        raise ValueError("hf_command must be a nonempty string")
    root = Path(data_root).resolve()
    cache = (
        Path(cache_dir).resolve()
        if cache_dir is not None
        else root / ".cache" / "memorysplit-v2"
    )
    paths = _staging_paths(lock, root)
    remaining = _remaining_download_bytes(lock, paths)
    derived = _locked_derived_bytes(lock)
    safety_margin = max(4 * 1024**3, (lock.download_bytes + 9) // 10)
    required = remaining + derived + reserve_bytes + safety_margin
    available = (
        int(disk_free_bytes)
        if disk_free_bytes is not None
        else shutil.disk_usage(_nearest_existing(root)).free
    )
    operations: list[dict[str, Any]] = []
    for source_id in ("fineweb_edu", "finemath", "wikidata5m"):
        source = lock.sources[source_id]
        destination = paths["payload"] / source_id
        if Path(str(source["repository"])).is_dir():
            operations.append(
                {
                    "destination": str(destination),
                    "files": len(source["files"]),
                    "kind": "local_copy",
                    "source": source_id,
                }
            )
        else:
            operations.append(
                {
                    "argv": _hf_command(
                        source,
                        destination,
                        hf_command,
                        max_workers,
                    ),
                    "env": {
                        "HF_HOME": str(cache / "huggingface"),
                        "HF_HUB_DISABLE_TELEMETRY": "1",
                    },
                    "kind": "huggingface_download",
                    "source": source_id,
                }
            )
    for source in lock.sources["objective_auxiliary"]["sources"]:
        for component in source["components"]:
            operations.append(
                {
                    "bytes": component["archive"]["bytes"],
                    "destination": str(
                        paths["downloads"]
                        / "objective"
                        / f"{component['id']}.tar.gz"
                    ),
                    "kind": "http_download",
                    "sha256": component["archive"]["sha256"],
                    "source": source["id"],
                    "url": component["archive"]["url"],
                }
            )
    return {
        "available_disk_bytes": available,
        "cache_dir": str(cache),
        "contract_id": lock.contract_id,
        "dataset_id": lock.dataset_id,
        "disk_preflight": "passed" if available >= required else "failed",
        "download_bytes": lock.download_bytes,
        "execute": False,
        "final_root": str(paths["final"]),
        "locked_derived_bytes": derived,
        "operations": operations,
        "production_readiness": dict(lock.production_readiness),
        "remaining_download_bytes": remaining,
        "required_free_bytes": required,
        "reserve_bytes": reserve_bytes,
        "safety_margin_bytes": safety_margin,
        "source_set_lock_sha256": lock.sha256,
        "stage_root": str(paths["stage"]),
    }


def _safe_destination(root: Path, relative: str) -> Path:
    _relative_path(relative, "staged file path")
    path = root.joinpath(*PurePosixPath(relative).parts)
    current = path
    while current != root and current != current.parent:
        if current.is_symlink():
            raise SourceDriftError(f"staged path crosses a symlink: {relative}")
        current = current.parent
    return path


def _verify_file(path: Path, record: Mapping[str, Any], label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise SourceDriftError(f"missing regular {label}: {path}")
    size, digest = size_and_sha256(path)
    if size != record["bytes"] or digest != record["sha256"]:
        raise SourceDriftError(
            f"{label} drift for {path}: expected "
            f"{record['bytes']} bytes/{record['sha256']}, got {size}/{digest}"
        )


def _copy_local_locked_source(
    source: Mapping[str, Any],
    destination: Path,
) -> None:
    repository = Path(str(source["repository"]))
    for record in source["files"]:
        relative = str(record["path"])
        source_path = _safe_destination(repository, relative)
        _verify_file(source_path, record, "local source file")
        target = _safe_destination(destination, relative)
        if target.exists() or target.is_symlink():
            _verify_file(target, record, "resumed source file")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.partial")
        with source_path.open("rb") as input_stream, temporary.open("wb") as output:
            shutil.copyfileobj(input_stream, output, 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(target)


def _verify_hf_source(source: Mapping[str, Any], destination: Path) -> None:
    for record in source["files"]:
        path = _safe_destination(destination, str(record["path"]))
        _verify_file(path, record, "Hugging Face source file")


def _run_hf_download(
    source: Mapping[str, Any],
    destination: Path,
    *,
    cache_dir: Path,
    hf_command: str,
    max_workers: int,
    run: Callable[..., subprocess.CompletedProcess[Any]],
) -> None:
    local = Path(str(source["repository"]))
    if local.is_dir():
        _copy_local_locked_source(source, destination)
        return
    executable = hf_command
    if os.sep not in executable and shutil.which(executable) is None:
        adjacent = Path(os.sys.executable).with_name(executable)
        if adjacent.is_file():
            executable = str(adjacent)
        else:
            raise V2SourceStageError(
                f"Hugging Face CLI is not installed or not on PATH: {hf_command}"
            )
    command = _hf_command(source, destination, executable, max_workers)
    environment = {
        **os.environ,
        "HF_HOME": str(cache_dir / "huggingface"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    # huggingface_hub >= 1.0 rejects --local-dir together with --cache-dir.
    # HF_HOME preserves an explicit cache location without that invalid pair.
    run(command, check=True, env=environment)
    _verify_hf_source(source, destination)


def _download_http(
    url: str,
    destination: Path,
    expected: Mapping[str, Any],
) -> None:
    if destination.exists() or destination.is_symlink():
        _verify_file(destination, expected, "objective source archive")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    expected_size = int(expected["bytes"])
    if partial.exists() and (
        not partial.is_file()
        or partial.is_symlink()
        or partial.stat().st_size > expected_size
    ):
        raise SourceDriftError(f"unsafe or oversized partial download: {partial}")
    if partial.exists() and partial.stat().st_size == expected_size:
        try:
            _verify_file(partial, expected, "completed partial objective archive")
        except SourceDriftError:
            partial.unlink()
        else:
            partial.replace(destination)
            return

    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "memorysplit-v2-source-stager/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", None)
                append = bool(offset and status == 206)
                mode = "ab" if append else "wb"
                with partial.open(mode) as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        if output.tell() > expected_size:
                            raise SourceDriftError(
                                f"download exceeds locked size: {url}"
                            )
                    output.flush()
                    os.fsync(output.fileno())
            if partial.stat().st_size != expected_size:
                raise OSError(
                    f"short download: expected {expected_size}, "
                    f"got {partial.stat().st_size}"
                )
            _verify_file(partial, expected, "partial objective source archive")
            partial.replace(destination)
            return
        except (OSError, urllib.error.URLError):
            if attempt == 2:
                raise


def _safe_tar_member(member: tarfile.TarInfo) -> PurePosixPath:
    name = member.name.rstrip("/")
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if (
        not name
        or "\x00" in name
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
        or "." in posix.parts
        or not (member.isdir() or member.isreg())
    ):
        raise SourceDriftError(f"unsafe archive member: {member.name!r}")
    return posix


def _verify_tree(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    expected = {str(item["path"]): item for item in records}
    actual: set[str] = set()
    if not root.is_dir() or root.is_symlink():
        raise SourceDriftError(f"missing regular {label} tree: {root}")
    for directory, directories, files in os.walk(root):
        directories.sort()
        files.sort()
        directory_path = Path(directory)
        for name in directories:
            path = directory_path / name
            if path.is_symlink():
                raise SourceDriftError(f"{label} tree contains symlink: {path}")
        for name in files:
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or not path.is_file():
                raise SourceDriftError(f"{label} tree contains unsafe file: {path}")
            actual.add(relative)
    if actual != set(expected):
        raise SourceDriftError(
            f"{label} tree namespace drift: "
            f"missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected))}"
        )
    for relative in sorted(expected):
        _verify_file(root / relative, expected[relative], f"{label} file")


def _extract_auxiliary_component(
    archive_path: Path,
    source_id: str,
    component: Mapping[str, Any],
    destination: Path,
) -> None:
    records = component["files"]
    if destination.exists() or destination.is_symlink():
        _verify_tree(destination, records, label=f"{source_id}/{component['id']}")
        return
    expected = {str(item["path"]): item for item in records}
    private = destination.with_name(f".{destination.name}.partial")
    if private.exists() or private.is_symlink():
        shutil.rmtree(private, ignore_errors=True)
    private.mkdir(parents=True)
    seen: set[str] = set()
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                path = _safe_tar_member(member)
                if len(path.parts) == 1:
                    if path.as_posix() != component["archive"]["root"]:
                        raise SourceDriftError("objective archive root drift")
                    continue
                if path.parts[0] != component["archive"]["root"]:
                    raise SourceDriftError("objective archive has multiple roots")
                relative = PurePosixPath(*path.parts[1:]).as_posix()
                if member.isdir() or relative not in expected:
                    continue
                if relative in seen:
                    raise SourceDriftError(
                        f"duplicate objective archive member: {relative}"
                    )
                seen.add(relative)
                stream = archive.extractfile(member)
                if stream is None:
                    raise SourceDriftError(f"unreadable archive member: {relative}")
                target = _safe_destination(private, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with stream, target.open("xb") as output:
                    while chunk := stream.read(1024 * 1024):
                        size += len(chunk)
                        digest.update(chunk)
                        output.write(chunk)
                record = expected[relative]
                if size != record["bytes"] or digest.hexdigest() != record["sha256"]:
                    raise SourceDriftError(
                        f"objective archive member drift: {relative}"
                    )
                target.chmod(0o644)
        if seen != set(expected):
            raise SourceDriftError(
                f"objective archive missing selected files: "
                f"{sorted(set(expected) - seen)}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        private.replace(destination)
    except BaseException:
        shutil.rmtree(private, ignore_errors=True)
        raise


def _extract_wikidata(
    source: Mapping[str, Any],
    archive_root: Path,
    destination: Path,
) -> list[dict[str, Any]]:
    expected: dict[str, Mapping[str, Any]] = {}
    for archive in source["files"]:
        for member in archive["members"]:
            path = str(member["path"])
            if path in expected:
                raise SourceDriftError(f"duplicate Wikidata member lock: {path}")
            expected[path] = member
    if destination.exists() or destination.is_symlink():
        if not destination.is_dir() or destination.is_symlink():
            raise SourceDriftError("Wikidata extraction destination is unsafe")
        actual = {
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        if actual != set(expected):
            raise SourceDriftError("resumed Wikidata extraction namespace drift")
    else:
        private = destination.with_name(f".{destination.name}.partial")
        if private.exists() or private.is_symlink():
            shutil.rmtree(private, ignore_errors=True)
        private.mkdir(parents=True)
        seen: set[str] = set()
        try:
            for archive_record in source["files"]:
                archive_path = archive_root / archive_record["path"]
                _verify_file(archive_path, archive_record, "Wikidata archive")
                archive_expected = {
                    str(item["path"]): item for item in archive_record["members"]
                }
                with tarfile.open(archive_path, "r:gz") as archive:
                    for member in archive.getmembers():
                        path = _safe_tar_member(member)
                        relative = path.as_posix()
                        if member.isdir():
                            continue
                        if relative not in archive_expected:
                            raise SourceDriftError(
                                f"unexpected Wikidata archive member: {relative}"
                            )
                        if relative in seen:
                            raise SourceDriftError(
                                f"duplicate Wikidata archive member: {relative}"
                            )
                        seen.add(relative)
                        stream = archive.extractfile(member)
                        if stream is None:
                            raise SourceDriftError(
                                f"unreadable Wikidata member: {relative}"
                            )
                        target = _safe_destination(private, relative)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        digest = hashlib.sha256()
                        size = 0
                        with stream, target.open("xb") as output:
                            while chunk := stream.read(1024 * 1024):
                                size += len(chunk)
                                digest.update(chunk)
                                output.write(chunk)
                        expected_member = archive_expected[relative]
                        if (
                            size != expected_member["bytes"]
                            or digest.hexdigest() != expected_member["sha256"]
                        ):
                            raise SourceDriftError(
                                f"Wikidata member content drift: {relative}"
                            )
                        target.chmod(0o644)
            if seen != set(expected):
                raise SourceDriftError(
                    f"Wikidata archives missing members: {sorted(set(expected) - seen)}"
                )
            private.replace(destination)
        except BaseException:
            shutil.rmtree(private, ignore_errors=True)
            raise
    records = []
    for relative in sorted(expected):
        path = destination / relative
        size, digest = size_and_sha256(path)
        if (
            size != expected[relative]["bytes"]
            or digest != expected[relative]["sha256"]
        ):
            raise SourceDriftError(f"Wikidata extracted file drift: {relative}")
        records.append({"bytes": size, "path": relative, "sha256": digest})
    return records


def _open_state_database(path: Path, kind: str, lock_sha256: str) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS processed (
            phase TEXT NOT NULL,
            path TEXT NOT NULL,
            input_sha256 TEXT NOT NULL,
            rows INTEGER NOT NULL,
            kept INTEGER NOT NULL,
            sidecar_path TEXT,
            sidecar_sha256 TEXT,
            PRIMARY KEY (phase, path)
        ) WITHOUT ROWID
        """
    )
    existing = dict(connection.execute("SELECT key, value FROM metadata"))
    wanted = {"kind": kind, "source_set_lock_sha256": lock_sha256}
    if existing and existing != wanted:
        connection.close()
        raise SourceDriftError(f"resume database metadata drift: {path}")
    if not existing:
        with connection:
            connection.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                sorted(wanted.items()),
            )
    return connection


def _parquet_text_batches(path: Path, batch_size: int = 8192) -> Iterator[list[str]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise V2SourceStageError(
            "pyarrow is required for production FineMath cross-deduplication"
        ) from error
    parquet_file = parquet.ParquetFile(path)
    if "text" not in parquet_file.schema_arrow.names:
        raise SourceDriftError(f"parquet source lacks text column: {path}")
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["text"],
        use_threads=False,
    ):
        values = batch.column(0).to_pylist()
        if any(not isinstance(value, str) for value in values):
            raise SourceDriftError(f"parquet text column contains non-string: {path}")
        yield values


def _existing_digests(
    connection: sqlite3.Connection,
    digests: Sequence[bytes],
) -> set[bytes]:
    unique = list(dict.fromkeys(digests))
    result: set[bytes] = set()
    for start in range(0, len(unique), 500):
        chunk = unique[start : start + 500]
        placeholders = ",".join("?" for _ in chunk)
        result.update(
            row[0]
            for row in connection.execute(
                f"SELECT digest FROM seen WHERE digest IN ({placeholders})",
                chunk,
            )
        )
    return result


def _sidecar_relative(prefix: str, source_path: str) -> str:
    return f"{prefix}/{source_path}.keep.u8"


def _process_text_file(
    connection: sqlite3.Connection,
    *,
    phase: str,
    source_path: str,
    source_sha256: str,
    path: Path,
    selection_root: Path,
    emit_sidecar: bool,
) -> Mapping[str, Any]:
    existing = connection.execute(
        """
        SELECT input_sha256, rows, kept, sidecar_path, sidecar_sha256
        FROM processed WHERE phase = ? AND path = ?
        """,
        (phase, source_path),
    ).fetchone()
    if existing is not None:
        input_sha, rows, kept, sidecar_path, sidecar_sha = existing
        if input_sha != source_sha256:
            raise SourceDriftError(f"resumed input hash drift: {source_path}")
        if emit_sidecar:
            target = selection_root.parent / sidecar_path
            _verify_file(
                target,
                {"bytes": rows, "sha256": sidecar_sha},
                "FineMath selection sidecar",
            )
        return {
            "dropped_rows": rows - kept,
            "input_path": source_path,
            "input_sha256": input_sha,
            "kept_rows": kept,
            "rows": rows,
            **(
                {
                    "sidecar": {
                        "bytes": rows,
                        "path": sidecar_path,
                        "sha256": sidecar_sha,
                    }
                }
                if emit_sidecar
                else {}
            ),
        }

    sidecar_relative = (
        _sidecar_relative("selection", source_path) if emit_sidecar else None
    )
    sidecar = (
        selection_root.parent / sidecar_relative if sidecar_relative else None
    )
    temporary = (
        sidecar.with_name(f".{sidecar.name}.partial") if sidecar is not None else None
    )
    if sidecar is not None:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        if sidecar.exists() or sidecar.is_symlink():
            sidecar.unlink()
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()

    rows = 0
    kept = 0
    published = False
    connection.execute("BEGIN IMMEDIATE")
    try:
        output = temporary.open("xb") if temporary is not None else None
        try:
            for texts in _parquet_text_batches(path):
                digests = [
                    hashlib.sha256(text.encode("utf-8")).digest() for text in texts
                ]
                known = _existing_digests(connection, digests)
                newly_seen: set[bytes] = set()
                keep = bytearray()
                insert: list[tuple[bytes]] = []
                for digest in digests:
                    accepted = digest not in known and digest not in newly_seen
                    if accepted:
                        newly_seen.add(digest)
                        insert.append((digest,))
                        kept += 1
                    keep.append(int(accepted))
                connection.executemany(
                    "INSERT INTO seen (digest) VALUES (?)",
                    insert,
                )
                rows += len(digests)
                if output is not None:
                    output.write(keep)
            if output is not None:
                output.flush()
                os.fsync(output.fileno())
        finally:
            if output is not None:
                output.close()

        sidecar_sha = None
        if temporary is not None and sidecar is not None:
            sidecar_size, sidecar_sha = size_and_sha256(temporary)
            if sidecar_size != rows:
                raise SourceDriftError("FineMath sidecar row alignment failure")
            temporary.replace(sidecar)
            published = True
        connection.execute(
            """
            INSERT INTO processed
                (phase, path, input_sha256, rows, kept, sidecar_path, sidecar_sha256)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                phase,
                source_path,
                source_sha256,
                rows,
                kept,
                sidecar_relative,
                sidecar_sha,
            ),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if published and sidecar is not None:
            sidecar.unlink(missing_ok=True)
        raise
    return {
        "dropped_rows": rows - kept,
        "input_path": source_path,
        "input_sha256": source_sha256,
        "kept_rows": kept,
        "rows": rows,
        **(
            {
                "sidecar": {
                    "bytes": rows,
                    "path": sidecar_relative,
                    "sha256": sidecar_sha,
                }
            }
            if emit_sidecar
            else {}
        ),
    }


def _build_finemath_selection(
    lock: V2SourceSetLock,
    payload: Path,
    state_root: Path,
) -> Mapping[str, Any]:
    fineweb = lock.sources["fineweb_edu"]
    finemath = lock.sources["finemath"]
    database = state_root / "finemath-exact-text.sqlite3"
    connection = _open_state_database(
        database,
        "finemath-exact-text-v1",
        lock.sha256,
    )
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS seen "
            "(digest BLOB PRIMARY KEY) WITHOUT ROWID"
        )
        connection.commit()
        for record in fineweb["files"]:
            source_path = str(record["path"])
            _process_text_file(
                connection,
                phase="fineweb_edu",
                source_path=source_path,
                source_sha256=str(record["sha256"]),
                path=payload / "fineweb_edu" / source_path,
                selection_root=payload / "finemath" / "selection",
                emit_sidecar=False,
            )

        records: list[Mapping[str, Any]] = []
        for subset in ("finemath-4plus", "finemath-3plus"):
            for record in finemath["files"]:
                source_path = str(record["path"])
                if not source_path.startswith(f"{subset}/"):
                    continue
                records.append(
                    _process_text_file(
                        connection,
                        phase=subset,
                        source_path=source_path,
                        source_sha256=str(record["sha256"]),
                        path=payload / "finemath" / source_path,
                        selection_root=payload / "finemath" / "selection",
                        emit_sidecar=True,
                    )
                )
    finally:
        connection.close()

    expected_rows = {
        name: int(value["rows"])
        for name, value in finemath["subset_metadata"].items()
    }
    actual_rows = {
        name: sum(
            int(record["rows"])
            for record in records
            if str(record["input_path"]).startswith(f"{name}/")
        )
        for name in expected_rows
    }
    if actual_rows != expected_rows:
        raise SourceDriftError(
            f"FineMath row totals differ from the upstream lock: "
            f"expected={expected_rows}, actual={actual_rows}"
        )
    result = {
        "algorithm": finemath["deduplication"],
        "files": records,
        "format": "memorysplit-v2-finemath-selection",
        "schema_version": 1,
        "subset_order": finemath["selection"]["subsets_in_order"],
        "totals": {
            "dropped_rows": sum(int(item["dropped_rows"]) for item in records),
            "kept_rows": sum(int(item["kept_rows"]) for item in records),
            "rows": sum(int(item["rows"]) for item in records),
        },
    }
    manifest = payload / "finemath" / "selection-manifest.json"
    manifest.write_bytes(_canonical_bytes(result))
    return result


def _existing_triples(
    connection: sqlite3.Connection,
    triples: Sequence[tuple[int, int, int]],
) -> set[tuple[int, int, int]]:
    unique = list(dict.fromkeys(triples))
    result: set[tuple[int, int, int]] = set()
    for start in range(0, len(unique), 250):
        chunk = unique[start : start + 250]
        placeholders = ",".join("(?,?,?)" for _ in chunk)
        parameters = [value for triple in chunk for value in triple]
        result.update(
            (int(row[0]), int(row[1]), int(row[2]))
            for row in connection.execute(
                f"""
                SELECT subject, relation, object
                FROM seen
                WHERE (subject, relation, object) IN ({placeholders})
                """,
                parameters,
            )
        )
    return result


def _process_wikidata_split(
    connection: sqlite3.Connection,
    *,
    split: str,
    path: Path,
    input_sha256: str,
    selection_root: Path,
) -> Mapping[str, Any]:
    existing = connection.execute(
        """
        SELECT input_sha256, rows, kept, sidecar_path, sidecar_sha256
        FROM processed WHERE phase = 'wikidata_training' AND path = ?
        """,
        (split,),
    ).fetchone()
    if existing is not None:
        input_sha, rows, kept, sidecar_path, sidecar_sha = existing
        if input_sha != input_sha256:
            raise SourceDriftError(f"resumed Wikidata split hash drift: {split}")
        _verify_file(
            selection_root.parent / sidecar_path,
            {"bytes": rows, "sha256": sidecar_sha},
            "Wikidata complete-once sidecar",
        )
        return {
            "dropped_rows": rows - kept,
            "input_path": split,
            "input_sha256": input_sha,
            "kept_rows": kept,
            "rows": rows,
            "sidecar": {
                "bytes": rows,
                "path": sidecar_path,
                "sha256": sidecar_sha,
            },
        }

    sidecar_relative = f"selection/{split}.keep.u8"
    sidecar = selection_root.parent / sidecar_relative
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    if sidecar.exists() or sidecar.is_symlink():
        sidecar.unlink()
    temporary = sidecar.with_name(f".{sidecar.name}.partial")
    temporary.unlink(missing_ok=True)
    rows = 0
    kept = 0
    published = False
    connection.execute("BEGIN IMMEDIATE")
    try:
        with temporary.open("xb") as output:
            batch: list[tuple[int, int, int]] = []

            def flush() -> None:
                nonlocal kept
                known = _existing_triples(connection, batch)
                newly_seen: set[tuple[int, int, int]] = set()
                mask = bytearray()
                insert: list[tuple[int, int, int]] = []
                for triple in batch:
                    accepted = triple not in known and triple not in newly_seen
                    if accepted:
                        newly_seen.add(triple)
                        insert.append(triple)
                        kept += 1
                    mask.append(int(accepted))
                connection.executemany(
                    "INSERT INTO seen (subject, relation, object) VALUES (?, ?, ?)",
                    insert,
                )
                output.write(mask)
                batch.clear()

            for triple in iter_triples(path):
                rows += 1
                batch.append(
                    (triple.subject, int(triple.relation[1:]), triple.object)
                )
                if len(batch) == 4096:
                    flush()
            if batch:
                flush()
            output.flush()
            os.fsync(output.fileno())
        size, sidecar_sha = size_and_sha256(temporary)
        if size != rows:
            raise SourceDriftError("Wikidata sidecar row alignment failure")
        temporary.replace(sidecar)
        published = True
        connection.execute(
            """
            INSERT INTO processed
                (phase, path, input_sha256, rows, kept, sidecar_path, sidecar_sha256)
            VALUES ('wikidata_training', ?, ?, ?, ?, ?, ?)
            """,
            (
                split,
                input_sha256,
                rows,
                kept,
                sidecar_relative,
                sidecar_sha,
            ),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        temporary.unlink(missing_ok=True)
        if published:
            sidecar.unlink(missing_ok=True)
        raise
    return {
        "dropped_rows": rows - kept,
        "input_path": split,
        "input_sha256": input_sha256,
        "kept_rows": kept,
        "rows": rows,
        "sidecar": {
            "bytes": rows,
            "path": sidecar_relative,
            "sha256": sidecar_sha,
        },
    }


def _wikidata_triple_tuple(triple: Any) -> tuple[int, int, int]:
    return (triple.subject, int(triple.relation[1:]), triple.object)


def _wikidata_triple_bytes(triple: tuple[int, int, int]) -> bytes:
    subject, relation, object_id = triple
    return f"Q{subject}\tP{relation}\tQ{object_id}\n".encode("ascii")


def _wikidata_triple_sequence_sha256(
    triples: Sequence[tuple[int, int, int]],
) -> str:
    digest = hashlib.sha256()
    for triple in triples:
        digest.update(_wikidata_triple_bytes(triple))
    return digest.hexdigest()


def _pack_wikidata_training_rows(
    rows: Sequence[Sequence[Any]],
) -> bytes:
    content = bytearray()
    try:
        for row in rows:
            content.extend(
                _WIKIDATA_TRAINING_INDEX_STRUCT.pack(
                    int(row[0]),
                    int(row[1]),
                    int(row[2]),
                )
            )
    except (IndexError, OverflowError, struct.error, TypeError, ValueError) as error:
        raise SourceDriftError(
            "Wikidata training triple exceeds index encoding"
        ) from error
    return bytes(content)


def _build_wikidata_training_index(
    connection: sqlite3.Connection,
    selection_root: Path,
) -> Mapping[str, Any]:
    """Publish a deterministic, exact index supporting independent overlap checks."""

    destination = selection_root / "training-triples.u64be"
    selection_root.mkdir(parents=True, exist_ok=True)
    count = int(connection.execute("SELECT COUNT(*) FROM seen").fetchone()[0])
    expected_size = count * _WIKIDATA_TRAINING_INDEX_STRUCT.size
    query = """
        SELECT subject, relation, object
        FROM seen
        ORDER BY subject, relation, object
    """
    digest = hashlib.sha256()

    if destination.exists() or destination.is_symlink():
        if (
            not destination.is_file()
            or destination.is_symlink()
            or destination.stat().st_size != expected_size
        ):
            raise SourceDriftError("resumed Wikidata training index drift")
        with destination.open("rb") as stream:
            cursor = connection.execute(query)
            while rows := cursor.fetchmany(8192):
                expected = _pack_wikidata_training_rows(rows)
                actual = stream.read(len(expected))
                if actual != expected:
                    raise SourceDriftError(
                        "resumed Wikidata training index content drift"
                    )
                digest.update(actual)
            if stream.read(1):
                raise SourceDriftError("resumed Wikidata training index is too long")
    else:
        temporary = destination.with_name(f".{destination.name}.partial")
        if temporary.exists() or temporary.is_symlink():
            if temporary.is_symlink() or not temporary.is_file():
                raise SourceDriftError(
                    f"unsafe Wikidata training index partial: {temporary}"
                )
            temporary.unlink()
        try:
            with temporary.open("xb") as stream:
                cursor = connection.execute(query)
                while rows := cursor.fetchmany(8192):
                    content = _pack_wikidata_training_rows(rows)
                    stream.write(content)
                    digest.update(content)
                stream.flush()
                os.fsync(stream.fileno())
            if temporary.stat().st_size != expected_size:
                raise SourceDriftError("Wikidata training index size drift")
            temporary.chmod(0o644)
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    return {
        "bytes": expected_size,
        "distinct_triples": count,
        "encoding": _WIKIDATA_TRAINING_INDEX_ENCODING,
        "ordering": "ascending_numeric_subject_relation_object",
        "path": destination.relative_to(selection_root.parent).as_posix(),
        "sha256": digest.hexdigest(),
    }


@contextlib.contextmanager
def _open_wikidata_training_index(
    wikidata_root: Path,
    raw_record: Mapping[str, Any],
    *,
    verify_hash: bool,
) -> Iterator[tuple[mmap.mmap, int]]:
    record = _mapping(raw_record, "Wikidata training index")
    if set(record) != {
        "bytes",
        "distinct_triples",
        "encoding",
        "ordering",
        "path",
        "sha256",
    }:
        raise SourceDriftError("Wikidata training index descriptor drift")
    count = record["distinct_triples"]
    size = record["bytes"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size != count * _WIKIDATA_TRAINING_INDEX_STRUCT.size
        or record["encoding"] != _WIKIDATA_TRAINING_INDEX_ENCODING
        or record["ordering"] != "ascending_numeric_subject_relation_object"
        or record["path"] != "selection/training-triples.u64be"
        or not isinstance(record["sha256"], str)
        or not _SHA256_RE.fullmatch(record["sha256"])
    ):
        raise SourceDriftError("Wikidata training index descriptor is invalid")
    path = _safe_destination(wikidata_root, str(record["path"]))
    if verify_hash:
        _verify_file(path, record, "Wikidata training triple index")
    elif not path.is_file() or path.is_symlink() or path.stat().st_size != size:
        raise SourceDriftError(f"missing regular Wikidata training index: {path}")
    with (
        path.open("rb") as stream,
        mmap.mmap(stream.fileno(), length=0, access=mmap.ACCESS_READ) as index,
    ):
        yield index, count


def _wikidata_training_index_contains(
    index: mmap.mmap,
    count: int,
    triple: tuple[int, int, int],
) -> bool:
    try:
        key = _WIKIDATA_TRAINING_INDEX_STRUCT.pack(*triple)
    except (OverflowError, struct.error) as error:
        raise SourceDriftError(
            "Wikidata sealed triple exceeds index encoding"
        ) from error
    low = 0
    high = count
    width = _WIKIDATA_TRAINING_INDEX_STRUCT.size
    while low < high:
        middle = (low + high) // 2
        candidate = index[middle * width : (middle + 1) * width]
        if candidate < key:
            low = middle + 1
        else:
            high = middle
    return low < count and index[low * width : (low + 1) * width] == key


def _write_exact_derived_file(path: Path, content: bytes, label: str) -> None:
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != content:
            raise SourceDriftError(f"resumed {label} drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.exists() or temporary.is_symlink():
        if temporary.is_symlink() or not temporary.is_file():
            raise SourceDriftError(f"unsafe {label} partial: {temporary}")
        temporary.unlink()
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _derive_wikidata_sealed_split(
    *,
    split: str,
    path: Path,
    input_sha256: str,
    training_index_sha256: str,
    index: mmap.mmap,
    training_count: int,
) -> tuple[
    Mapping[str, Any],
    bytes,
    bytes,
    list[tuple[int, int, int]],
    list[tuple[int, int, int]],
    list[tuple[int, int, int]],
]:
    triples = [_wikidata_triple_tuple(item) for item in iter_triples(path)]
    eligible: list[tuple[int, int, int]] = []
    excluded: list[tuple[int, int, int]] = []
    evidence_records = []
    mask = bytearray()
    for row_index, triple in enumerate(triples):
        overlaps_training = _wikidata_training_index_contains(
            index,
            training_count,
            triple,
        )
        mask.append(0 if overlaps_training else 1)
        if overlaps_training:
            excluded.append(triple)
            evidence_records.append(
                {
                    "canonical_triple_sha256": hashlib.sha256(
                        _wikidata_triple_bytes(triple)
                    ).hexdigest(),
                    "row_index": row_index,
                }
            )
        else:
            eligible.append(triple)

    sidecar_relative = f"selection/{split}.eligible.u8"
    evidence_relative = f"selection/{split}.training-overlap-exclusions.json"
    evidence = {
        "format": "memorysplit-v2-wikidata-sealed-exclusions",
        "input_path": split,
        "input_sha256": input_sha256,
        "policy": _WIKIDATA_SEALED_POLICY,
        "reason": "exact_triple_present_in_complete_once_training",
        "records": evidence_records,
        "row_indexing": "zero_based_nonempty_parsed_source_rows",
        "schema_version": 1,
        "training_index_sha256": training_index_sha256,
    }
    sidecar_content = bytes(mask)
    evidence_content = _canonical_bytes(evidence)
    record = {
        "eligibility_sidecar": {
            "bytes": len(sidecar_content),
            "encoding": "one_unsigned_byte_per_source_row_1_eligible_0_excluded",
            "path": sidecar_relative,
            "sha256": hashlib.sha256(sidecar_content).hexdigest(),
        },
        "eligible_distinct_triples": len(set(eligible)),
        "eligible_duplicate_rows": len(eligible) - len(set(eligible)),
        "eligible_rows": len(eligible),
        "eligible_triples_sha256": _wikidata_triple_sequence_sha256(eligible),
        "excluded_training_overlap_distinct_triples": len(set(excluded)),
        "excluded_training_overlap_rows": len(excluded),
        "excluded_training_overlap_triples_sha256": (
            _wikidata_triple_sequence_sha256(excluded)
        ),
        "exclusion_evidence": {
            "bytes": len(evidence_content),
            "path": evidence_relative,
            "records": len(evidence_records),
            "sha256": hashlib.sha256(evidence_content).hexdigest(),
        },
        "input_path": split,
        "input_sha256": input_sha256,
        "source_distinct_triples": len(set(triples)),
        "source_duplicate_rows": len(triples) - len(set(triples)),
        "source_rows": len(triples),
    }
    return record, sidecar_content, evidence_content, triples, eligible, excluded


def _synchronize_wikidata_sealed_artifact(
    path: Path,
    expected: bytes,
    *,
    label: str,
    publish: bool,
    sidecar: bool = False,
) -> None:
    if not path.is_file() or path.is_symlink():
        if not publish:
            raise SourceDriftError(f"missing regular {label}: {path}")
        _write_exact_derived_file(path, expected, label)
        return
    actual = path.read_bytes()
    if actual == expected:
        return
    if sidecar and len(actual) == len(expected):
        for expected_marker, actual_marker in zip(expected, actual, strict=True):
            if actual_marker == 1 and expected_marker == 0:
                raise SourceDriftError(
                    "eligible Wikidata sealed subset contains a training triple"
                )
            if actual_marker not in (0, 1):
                raise SourceDriftError(
                    "Wikidata sealed eligibility sidecar contains invalid bytes"
                )
    raise SourceDriftError(f"{label} drift: {path}")


def _derive_wikidata_sealed_audit(
    *,
    files_root: Path,
    extracted: Mapping[str, Mapping[str, Any]],
    selection_root: Path,
    training_index: Mapping[str, Any],
    publish: bool,
    verify_index_hash: bool,
) -> Mapping[str, Any]:
    split_records = []
    source_triples: list[tuple[int, int, int]] = []
    eligible_triples: list[tuple[int, int, int]] = []
    excluded_triples: list[tuple[int, int, int]] = []
    with _open_wikidata_training_index(
        selection_root.parent,
        training_index,
        verify_hash=verify_index_hash,
    ) as (index, training_count):
        for split in _WIKIDATA_SEALED_SPLITS:
            if split not in extracted:
                raise SourceDriftError(
                    f"Wikidata sealed file is not inventoried: {split}"
                )
            record, sidecar, evidence, source, eligible, excluded = (
                _derive_wikidata_sealed_split(
                    split=split,
                    path=files_root / split,
                    input_sha256=str(extracted[split]["sha256"]),
                    training_index_sha256=str(training_index["sha256"]),
                    index=index,
                    training_count=training_count,
                )
            )
            _synchronize_wikidata_sealed_artifact(
                _safe_destination(
                    selection_root.parent, record["eligibility_sidecar"]["path"]
                ),
                sidecar,
                label="Wikidata sealed eligibility sidecar",
                publish=publish,
                sidecar=True,
            )
            _synchronize_wikidata_sealed_artifact(
                _safe_destination(
                    selection_root.parent, record["exclusion_evidence"]["path"]
                ),
                evidence,
                label="Wikidata sealed exclusion evidence",
                publish=publish,
            )
            split_records.append(record)
            source_triples.extend(source)
            eligible_triples.extend(eligible)
            excluded_triples.extend(excluded)

    return {
        "eligibility_sidecar_encoding": (
            "one_unsigned_byte_per_source_row_1_eligible_0_excluded"
        ),
        "format": "memorysplit-v2-wikidata-sealed-evaluation-audit",
        "policy": _WIKIDATA_SEALED_POLICY,
        "schema_version": 1,
        "source_files_retained": True,
        "splits": split_records,
        "totals": {
            "eligible_distinct_triples": len(set(eligible_triples)),
            "eligible_duplicate_rows": (
                len(eligible_triples) - len(set(eligible_triples))
            ),
            "eligible_rows": len(eligible_triples),
            "eligible_training_overlap_rows": 0,
            "excluded_training_overlap_distinct_triples": len(set(excluded_triples)),
            "excluded_training_overlap_rows": len(excluded_triples),
            "official_source_distinct_triples": len(set(source_triples)),
            "official_source_duplicate_rows": (
                len(source_triples) - len(set(source_triples))
            ),
            "official_source_rows": len(source_triples),
        },
        "training_index": dict(training_index),
    }


def _audit_sealed_wikidata(
    connection: sqlite3.Connection,
    files_root: Path,
    extracted: Mapping[str, Mapping[str, Any]],
    selection_root: Path,
) -> Mapping[str, Any]:
    training_index = _build_wikidata_training_index(connection, selection_root)
    return _derive_wikidata_sealed_audit(
        files_root=files_root,
        extracted=extracted,
        selection_root=selection_root,
        training_index=training_index,
        publish=True,
        verify_index_hash=False,
    )


def _verify_wikidata_sealed_audit(
    wikidata_root: Path,
    extracted: Mapping[str, Mapping[str, Any]],
    raw_audit: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        audit = _mapping(raw_audit, "Wikidata sealed evaluation audit")
        training_index = _mapping(
            audit["training_index"],
            "Wikidata sealed training index",
        )
        expected = _derive_wikidata_sealed_audit(
            files_root=wikidata_root / "files",
            extracted=extracted,
            selection_root=wikidata_root / "selection",
            training_index=training_index,
            publish=False,
            verify_index_hash=True,
        )
    except SourceDriftError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise SourceDriftError(
            "Wikidata sealed evaluation audit schema drift"
        ) from error
    if audit != expected:
        raise SourceDriftError(
            "Wikidata sealed evaluation audit does not match eligible rows"
        )
    if expected["totals"]["eligible_training_overlap_rows"] != 0:
        raise SourceDriftError(
            "eligible Wikidata sealed subset overlaps complete-once training"
        )
    return expected


def _build_wikidata_selection(
    lock: V2SourceSetLock,
    payload: Path,
    state_root: Path,
    extracted_records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    source = lock.sources["wikidata5m"]
    files_root = payload / "wikidata5m" / "files"
    extracted = {str(item["path"]): item for item in extracted_records}
    database = state_root / "wikidata-complete-once.sqlite3"
    connection = _open_state_database(
        database,
        "wikidata-complete-once-v1",
        lock.sha256,
    )
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS seen (
                subject INTEGER NOT NULL,
                relation INTEGER NOT NULL,
                object INTEGER NOT NULL,
                PRIMARY KEY (subject, relation, object)
            ) WITHOUT ROWID
            """
        )
        connection.commit()
        records = []
        for split in source["complete_once"]["split_order"]:
            records.append(
                _process_wikidata_split(
                    connection,
                    split=split,
                    path=files_root / split,
                    input_sha256=str(extracted[split]["sha256"]),
                    selection_root=payload / "wikidata5m" / "selection",
                )
            )
        sealed_audit = _audit_sealed_wikidata(
            connection,
            files_root,
            extracted,
            payload / "wikidata5m" / "selection",
        )
        distinct = int(connection.execute("SELECT COUNT(*) FROM seen").fetchone()[0])
    finally:
        connection.close()
    if distinct != sum(int(record["kept_rows"]) for record in records):
        raise SourceDriftError("Wikidata complete-once distinct count drift")
    result = {
        "algorithm": source["complete_once"],
        "files": records,
        "format": "memorysplit-v2-wikidata-complete-once-selection",
        "schema_version": 1,
        "sealed_evaluation_audit": sealed_audit,
        "totals": {
            "distinct_training_triples": distinct,
            "dropped_duplicate_rows": sum(
                int(item["dropped_rows"]) for item in records
            ),
            "training_rows": sum(int(item["rows"]) for item in records),
        },
    }
    manifest = payload / "wikidata5m" / "selection-manifest.json"
    manifest.write_bytes(_canonical_bytes(result))
    return result


def _lock_contract_files(
    lock: V2SourceSetLock,
    payload: Path,
) -> list[tuple[Path, Path]]:
    inputs = [lock.path, *[lock.source_paths[name] for name in _EXPECTED_LOCK_IDS]]
    repository = payload / "source-contract"
    contract_locks = repository / "sources" / "memorysplit-v2"
    scientific_destination = (
        contract_locks / str(lock.raw["scientific_requirements"]["path"])
    ).resolve()
    if repository.resolve() not in scientific_destination.parents:
        raise SourceDriftError("scientific contract destination escapes staged root")
    notice_reference = lock.sources["wikidata5m"]["license_evidence"]
    notice_path = (
        lock.source_paths["wikidata5m"].parent / notice_reference["notice_path"]
    ).resolve()
    notice_destination = (
        contract_locks / str(notice_reference["notice_path"])
    ).resolve()
    if repository.resolve() not in notice_destination.parents:
        raise SourceDriftError("license notice destination escapes staged root")
    return [
        *[(path, payload / "locks" / path.name) for path in inputs],
        *[(path, contract_locks / path.name) for path in inputs],
        (
            Path(lock.scientific_requirements["path"]),
            scientific_destination,
        ),
        (
            notice_path,
            notice_destination,
        ),
        (notice_path, payload / "licenses" / "Wikidata-CC0-1.0.txt"),
    ]


def _verify_staged_lock_contract(lock: V2SourceSetLock, payload: Path) -> None:
    for source, destination in _lock_contract_files(lock, payload):
        if (
            not destination.is_file()
            or destination.is_symlink()
            or destination.read_bytes() != source.read_bytes()
        ):
            raise SourceDriftError(f"staged source contract drift: {destination}")
    staged_master = (
        payload
        / "source-contract"
        / "sources"
        / "memorysplit-v2"
        / lock.path.name
    )
    staged_lock = load_v2_source_lock(staged_master)
    if staged_lock.sha256 != lock.sha256:
        raise SourceDriftError("staged source-set lock identity drift")


def _copy_lock_contract(lock: V2SourceSetLock, payload: Path) -> None:
    for source, destination in _lock_contract_files(lock, payload):
        content = source.read_bytes()
        if destination.exists() or destination.is_symlink():
            if (
                not destination.is_file()
                or destination.is_symlink()
                or destination.read_bytes() != content
            ):
                raise SourceDriftError(f"staged source contract drift: {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.partial")
        if temporary.exists() or temporary.is_symlink():
            if temporary.is_symlink() or not temporary.is_file():
                raise SourceDriftError(
                    f"unsafe staged contract partial: {temporary}"
                )
            temporary.unlink()
        with temporary.open("xb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(0o644)
        temporary.replace(destination)

    notice_reference = lock.sources["wikidata5m"]["license_evidence"]
    notice_path = (
        lock.source_paths["wikidata5m"].parent / notice_reference["notice_path"]
    ).resolve()
    _verify_file(
        notice_path,
        {
            "bytes": notice_path.stat().st_size,
            "sha256": notice_reference["notice_sha256"],
        },
        "Wikidata CC0 notice",
    )
    _verify_staged_lock_contract(lock, payload)


def _inventory(root: Path, *, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = exclude or set()
    records = []
    for directory, directories, files in os.walk(root):
        directories.sort()
        files.sort()
        directory_path = Path(directory)
        for name in directories:
            path = directory_path / name
            if path.is_symlink():
                raise SourceDriftError(f"source stage contains symlink: {path}")
        for name in files:
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            if relative in excluded:
                continue
            if path.is_symlink() or not path.is_file():
                raise SourceDriftError(f"source stage contains unsafe file: {path}")
            size, digest = size_and_sha256(path)
            records.append({"bytes": size, "path": relative, "sha256": digest})
    return records


def _tree_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_bytes(list(records))).hexdigest()


def _build_receipt(
    lock: V2SourceSetLock,
    payload: Path,
    finemath_selection: Mapping[str, Any],
    wikidata_selection: Mapping[str, Any],
    extracted_wikidata: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    files = _inventory(payload, exclude={RECEIPT_NAME})
    auxiliary = []
    for source in lock.sources["objective_auxiliary"]["sources"]:
        components = []
        for component in source["components"]:
            records = component["files"]
            components.append(
                {
                    "archive": component["archive"],
                    "commit": component["commit"],
                    "file_count": len(records),
                    "id": component["id"],
                    "license": component["license"],
                    "selected_bytes": sum(int(item["bytes"]) for item in records),
                    "selected_tree_sha256": _tree_digest(records),
                }
            )
        auxiliary.append(
            {
                "components": components,
                "id": source["id"],
                "training_use": source["training_use"],
            }
        )
    return {
        "contract_id": lock.contract_id,
        "dataset_id": lock.dataset_id,
        "download_bytes": lock.download_bytes,
        "files": files,
        "format": "memorysplit-v2-source-stage-receipt",
        "inventory_sha256": _tree_digest(files),
        "missing_materialized_lanes": list(
            lock.production_readiness["missing_materialized_lanes"]
        ),
        "missing_source_locks": list(
            lock.production_readiness["missing_source_locks"]
        ),
        "production_corpus_ready": False,
        "schema_version": 1,
        "source_set_lock_sha256": lock.sha256,
        "stage_complete": True,
        "sources": {
            "finemath": {
                "license": lock.sources["finemath"]["license"],
                "repository": lock.sources["finemath"]["repository"],
                "revision": lock.sources["finemath"]["revision"],
                "selection_manifest_sha256": hashlib.sha256(
                    _canonical_bytes(finemath_selection)
                ).hexdigest(),
                "selection_totals": finemath_selection["totals"],
            },
            "fineweb_edu": {
                "file_count": len(lock.sources["fineweb_edu"]["files"]),
                "license": lock.sources["fineweb_edu"]["license"],
                "repository": lock.sources["fineweb_edu"]["repository"],
                "revision": lock.sources["fineweb_edu"]["revision"],
                "scope": "complete sample-10BT",
            },
            "objective_auxiliary": auxiliary,
            "wikidata5m": {
                "complete_once": True,
                "extracted_files": list(extracted_wikidata),
                "license": lock.sources["wikidata5m"]["license"],
                "repository": lock.sources["wikidata5m"]["repository"],
                "revision": lock.sources["wikidata5m"]["revision"],
                "sealed_evaluation": wikidata_selection["sealed_evaluation_audit"],
                "selection_manifest_sha256": hashlib.sha256(
                    _canonical_bytes(wikidata_selection)
                ).hexdigest(),
                "selection_totals": wikidata_selection["totals"],
            },
        },
    }


def _read_canonical_mapping(path: Path, label: str) -> Mapping[str, Any]:
    value = _mapping(_read_json(path), label)
    if path.read_bytes() != _canonical_bytes(value):
        raise SourceDriftError(f"{label} is not canonical JSON")
    return value


def _read_canonical_receipt(path: Path) -> Mapping[str, Any]:
    return _read_canonical_mapping(path, "source stage receipt")


def verify_v2_source_stage(
    lock: V2SourceSetLock,
    source_root: Path | str,
) -> Mapping[str, Any]:
    root = Path(source_root)
    if not root.is_dir() or root.is_symlink():
        raise SourceDriftError(f"missing regular v2 source root: {root}")
    receipt_path = root / RECEIPT_NAME
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise SourceDriftError(f"missing regular v2 source receipt: {receipt_path}")
    receipt = _read_canonical_receipt(receipt_path)
    required = {
        "contract_id",
        "dataset_id",
        "download_bytes",
        "files",
        "format",
        "inventory_sha256",
        "missing_materialized_lanes",
        "missing_source_locks",
        "production_corpus_ready",
        "schema_version",
        "source_set_lock_sha256",
        "stage_complete",
        "sources",
    }
    if set(receipt) != required:
        raise SourceDriftError("source stage receipt fields differ from schema")
    if (
        receipt["format"] != "memorysplit-v2-source-stage-receipt"
        or receipt["schema_version"] != 1
        or receipt["contract_id"] != lock.contract_id
        or receipt["dataset_id"] != lock.dataset_id
        or receipt["source_set_lock_sha256"] != lock.sha256
        or receipt["stage_complete"] is not True
        or receipt["production_corpus_ready"] is not False
        or receipt["missing_source_locks"]
        != lock.production_readiness["missing_source_locks"]
        or receipt["missing_materialized_lanes"]
        != lock.production_readiness["missing_materialized_lanes"]
        or receipt["download_bytes"] != lock.download_bytes
    ):
        raise SourceDriftError("source stage receipt identity or completion drift")
    recorded_files = receipt["files"]
    if not isinstance(recorded_files, list):
        raise SourceDriftError("source stage receipt file inventory is invalid")
    actual_files = _inventory(root, exclude={RECEIPT_NAME})
    if actual_files != recorded_files:
        raise SourceDriftError("source stage file inventory drift")
    if _tree_digest(actual_files) != receipt["inventory_sha256"]:
        raise SourceDriftError("source stage inventory digest drift")
    _verify_staged_lock_contract(lock, root)
    for source_id in ("fineweb_edu", "finemath", "wikidata5m"):
        _verify_hf_source(lock.sources[source_id], root / source_id)
    for source in lock.sources["objective_auxiliary"]["sources"]:
        for component in source["components"]:
            _verify_tree(
                root
                / "objective_auxiliary"
                / str(source["id"])
                / str(component["id"]),
                component["files"],
                label=f"{source['id']}/{component['id']}",
            )
    extracted_wikidata = _extract_wikidata(
        lock.sources["wikidata5m"],
        root / "wikidata5m",
        root / "wikidata5m" / "files",
    )
    finemath_selection = _read_canonical_mapping(
        root / "finemath" / "selection-manifest.json",
        "FineMath selection manifest",
    )
    wikidata_selection = _read_canonical_mapping(
        root / "wikidata5m" / "selection-manifest.json",
        "Wikidata selection manifest",
    )
    _verify_wikidata_sealed_audit(
        root / "wikidata5m",
        {str(item["path"]): item for item in extracted_wikidata},
        wikidata_selection.get("sealed_evaluation_audit"),
    )
    expected_receipt = _build_receipt(
        lock,
        root,
        finemath_selection,
        wikidata_selection,
        extracted_wikidata,
    )
    if receipt != expected_receipt:
        raise SourceDriftError("source stage receipt does not match locked sources")
    return receipt


@contextlib.contextmanager
def _stage_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise V2SourceStageError(
                f"another source staging process holds {path}"
            ) from error
        stream.seek(0)
        stream.truncate()
        stream.write(f"{os.getpid()}\n")
        stream.flush()
        yield


def stage_v2_sources(
    lock: V2SourceSetLock,
    data_root: Path | str,
    *,
    execute: bool,
    cache_dir: Path | str | None = None,
    hf_command: str = "hf",
    max_workers: int = 8,
    reserve_bytes: int = DEFAULT_RESERVE_BYTES,
    restart: bool = False,
    disk_free_bytes: int | None = None,
    run: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> Mapping[str, Any]:
    root = Path(data_root).resolve()
    cache = (
        Path(cache_dir).resolve()
        if cache_dir is not None
        else root / ".cache" / "memorysplit-v2"
    )
    plan = plan_v2_source_stage(
        lock,
        root,
        cache_dir=cache,
        hf_command=hf_command,
        max_workers=max_workers,
        reserve_bytes=reserve_bytes,
        disk_free_bytes=disk_free_bytes,
    )
    if restart:
        clean_remaining = lock.download_bytes
        plan = {
            **plan,
            "remaining_download_bytes": clean_remaining,
            "required_free_bytes": (
                int(plan["required_free_bytes"])
                + clean_remaining
                - int(plan["remaining_download_bytes"])
            ),
        }
        plan["disk_preflight"] = (
            "passed"
            if int(plan["available_disk_bytes"]) >= int(plan["required_free_bytes"])
            else "failed"
        )
    if not execute:
        return plan
    paths = _staging_paths(lock, root)
    final_root = paths["final"]
    if final_root.exists() or final_root.is_symlink():
        return verify_v2_source_stage(lock, final_root)
    if plan["disk_preflight"] != "passed":
        raise InsufficientDiskError(
            "source staging disk preflight failed: "
            f"required={plan['required_free_bytes']}, "
            f"available={plan['available_disk_bytes']}"
        )
    if root.is_symlink():
        raise V2SourceStageError(f"data root must not be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    lock_path = paths["stage"].parent / f".{lock.sha256}.lock"
    # Keep the lock outside the disposable stage directory so --restart can
    # never remove another process's live lock inode.
    with _stage_lock(lock_path):
        if final_root.exists() or final_root.is_symlink():
            return verify_v2_source_stage(lock, final_root)
        if restart and (paths["stage"].exists() or paths["stage"].is_symlink()):
            if paths["stage"].is_symlink() or not paths["stage"].is_dir():
                raise SourceDriftError(f"unsafe restart stage root: {paths['stage']}")
            shutil.rmtree(paths["stage"])
        paths["payload"].mkdir(parents=True, exist_ok=True)
        paths["downloads"].mkdir(parents=True, exist_ok=True)
        paths["state"].mkdir(parents=True, exist_ok=True)

        _copy_lock_contract(lock, paths["payload"])
        for source_id in ("fineweb_edu", "finemath", "wikidata5m"):
            destination = paths["payload"] / source_id
            destination.mkdir(parents=True, exist_ok=True)
            _run_hf_download(
                lock.sources[source_id],
                destination,
                cache_dir=cache,
                hf_command=hf_command,
                max_workers=max_workers,
                run=run,
            )

        objective_root = paths["payload"] / "objective_auxiliary"
        for source in lock.sources["objective_auxiliary"]["sources"]:
            for component in source["components"]:
                archive_path = (
                    paths["downloads"]
                    / "objective"
                    / f"{component['id']}.tar.gz"
                )
                _download_http(
                    str(component["archive"]["url"]),
                    archive_path,
                    component["archive"],
                )
                _extract_auxiliary_component(
                    archive_path,
                    str(source["id"]),
                    component,
                    objective_root / str(source["id"]) / str(component["id"]),
                )

        extracted_wikidata = _extract_wikidata(
            lock.sources["wikidata5m"],
            paths["payload"] / "wikidata5m",
            paths["payload"] / "wikidata5m" / "files",
        )
        finemath_selection = _build_finemath_selection(
            lock,
            paths["payload"],
            paths["state"],
        )
        wikidata_selection = _build_wikidata_selection(
            lock,
            paths["payload"],
            paths["state"],
            extracted_wikidata,
        )

        for source_id in ("fineweb_edu", "finemath", "wikidata5m"):
            shutil.rmtree(
                paths["payload"] / source_id / ".cache",
                ignore_errors=True,
            )
        receipt = _build_receipt(
            lock,
            paths["payload"],
            finemath_selection,
            wikidata_selection,
            extracted_wikidata,
        )
        receipt_path = paths["payload"] / RECEIPT_NAME
        receipt_path.write_bytes(_canonical_bytes(receipt))
        verify_v2_source_stage(lock, paths["payload"])
        if final_root.exists() or final_root.is_symlink():
            raise FileExistsError(f"v2 source root appeared during staging: {final_root}")
        paths["payload"].replace(final_root)

        verified = verify_v2_source_stage(lock, final_root)
        shutil.rmtree(paths["stage"], ignore_errors=True)
        return verified


def receipt_summary(
    lock: V2SourceSetLock,
    source_root: Path | str,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(source_root).resolve()
    receipt_path = root / RECEIPT_NAME
    return {
        "dataset_id": receipt["dataset_id"],
        "download_bytes": receipt["download_bytes"],
        "inventory_sha256": receipt["inventory_sha256"],
        "missing_materialized_lanes": receipt["missing_materialized_lanes"],
        "missing_source_locks": receipt["missing_source_locks"],
        "production_corpus_ready": receipt["production_corpus_ready"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": size_and_sha256(receipt_path)[1],
        "source_root": str(root),
        "source_set_lock_sha256": lock.sha256,
        "stage_complete": receipt["stage_complete"],
    }
