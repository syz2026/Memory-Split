"""Pinned, offline-verifiable sources for the current relational dataset."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from corpusgen.wikidata5m import (
    ArchiveLock,
    SourceDriftError,
    Triple,
    WikidataLock,
    canonicalize_aliases,
    iter_triples,
    normalize_alias,
    parse_pid,
    parse_qid,
    safe_extract_archives,
    size_and_sha256,
    verify_archives,
)


_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SOURCE_NAMES = {
    "fineweb_edu",
    "wikidata5m",
    "tiny_recursive_models",
    "arc_agi_1",
    "arc_agi_2",
    "conceptarc",
}
_PUZZLE_SOURCE_ORDER = ("arc_agi_1", "arc_agi_2", "conceptarc")

_REQUIRED_IMPLEMENTATION = {
    "context_tokens": 1024,
    "raw_target_lane_shares": {
        "fineweb_edu": 0.4,
        "wikidata_graph": 0.2,
        "synthetic_graph": 0.1,
        "synthetic_reasoning": 0.15,
        "wikidata_reasoning": 0.075,
        "relational_refinement": 0.025,
        "puzzle_auxiliary": 0.05,
    },
    "token_floors": {
        "29m": {
            "parameters": 28969216,
            "updates": 1106,
            "raw_target_tokens": 579862528,
        },
        "160m": {
            "parameters": 162220800,
            "updates": 6189,
            "raw_target_tokens": 3244818432,
        },
        "360m": {
            "parameters": 356033536,
            "updates": 13582,
            "raw_target_tokens": 7120879616,
        },
    },
    "synthetic_fact_load": {
        "applies_to": "synthetic_entity_count_only",
        "wikidata_training_graph": "constant_complete_for_160m_and_360m",
        "wikidata_varies_by_fact_load": False,
        "wikidata_coverage_by_scale": {
            "29m": "deterministic_hash_sample_balanced_by_split_and_relation",
            "160m": "complete_training_graph",
            "360m": "complete_training_graph",
        },
        "labels": {"n50k": 50000, "n800k": 800000, "n1p8m": 1800000},
    },
    "reasoning_protocol": {
        "action_slots": 12,
        "max_reads": 10,
        "training_reads": {"min": 1, "max": 6},
        "held_out_length_reads": {"min": 7, "max": 10},
        "post_halt_slots": "deterministic_noop",
    },
    "wikidata_pages": {
        "address_fields": ["entity_qid", "property_pid", "direction", "page"],
        "object_order": "distinct_numeric_qid_ascending",
        "page_index_base": 0,
        "functional_row_page": 0,
        "page_rule": "largest_stable_prefix_with_formatted_record_at_most_context",
        "truncate": False,
    },
    "random_mask_matching": {
        "equal_mass_unit": "target_tokens",
        "strata": [
            "source",
            "record_type",
            "exact_payload_token_length",
            "packed_position_bin",
        ],
        "packed_position_bins": 10,
        "candidate_payload_class": "non_factual",
    },
    "composition_partition": {
        "salt": "relational-chinchilla/composition-partition",
        "sequence_lengths": [2, 3],
        "canonical_sequence": "comma_joined_pid_sequence",
        "hash": "sha256",
        "hash_input_parts": [
            "salt_utf8",
            "nul_byte",
            "canonical_sequence_utf8",
        ],
        "digest_integer_byte_order": "big",
        "evaluation_modulus": 5,
        "evaluation_remainders": [0],
    },
    "arc_tasks": {
        "canonical_hash": {
            "algorithm": "sha256",
            "encoding": "utf-8",
            "json": {
                "sort_keys": True,
                "separators": [",", ":"],
                "ensure_ascii": False,
            },
        },
        "transforms": {
            "include_original": True,
            "max_unique_excluding_original": 63,
            "spatial_symmetries": [
                "identity",
                "rotate_90",
                "rotate_180",
                "rotate_270",
                "reflect_horizontal",
                "reflect_vertical",
                "reflect_main_diagonal",
                "reflect_anti_diagonal",
            ],
            "color_permutations": "global_foreground_1_to_9_with_zero_fixed",
            "translations": "all_common_in_bounds_nonzero_shifts_with_zero_fill",
            "deduplicate_by": "canonical_task_sha256",
            "order_by": "sha256_of_canonical_parameter_json",
        },
    },
}


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_json_bytes(content: bytes, *, description: str) -> Any:
    try:
        text = content.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON in {description}") from exc


def _strict_json_path(path: Path) -> Any:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing regular JSON file: {path}")
    return _strict_json_bytes(path.read_bytes(), description=str(path))


def _require_mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _require_keys(
    value: Mapping[str, Any],
    expected: set[str],
    description: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(
            f"{description} fields mismatch; missing={missing}, extra={extra}"
        )


def _validate_commit(value: Any, description: str) -> str:
    if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
        raise ValueError(f"{description} must be a 40-character lowercase commit")
    return value


def _validate_relative_path(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{description} must be a nonempty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"unsafe {description}: {value!r}")
    return value


def _validate_contract_reference(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{description} must be a nonempty POSIX path")
    if PurePosixPath(value).is_absolute():
        raise ValueError(f"{description} must be relative to the dataset lock")
    return value


def _validate_string_list(value: Any, description: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{description} must be a nonempty list")
    result = []
    for index, item in enumerate(value):
        result.append(_validate_relative_path(item, f"{description}[{index}]"))
    if len(result) != len(set(result)):
        raise ValueError(f"{description} contains duplicates")
    return result


@dataclass(frozen=True)
class CurrentDatasetLock:
    path: Path
    dataset_id: str
    sources: dict[str, dict[str, Any]]
    implementation: dict[str, Any]
    licenses_path: Path
    licenses: dict[str, Any]
    wikidata_lock: WikidataLock
    raw: dict[str, Any]

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.raw)


def _validate_huggingface_source(
    name: str,
    source: dict[str, Any],
) -> None:
    common = {
        "transport",
        "repository",
        "repo_type",
        "revision",
        "license",
    }
    expected = (
        common
        | {
            "files",
            "materialized_jsonl",
            "holdout_records",
        }
        if name == "fineweb_edu"
        else common
        | {
            "lock_path",
            "scope",
            "training_splits",
            "sealed_splits",
            "split_files",
            "alias_files",
            "notice_path",
        }
    )
    _require_keys(source, expected, f"{name} source")
    if source["transport"] != "huggingface":
        raise ValueError(f"{name} transport must be huggingface")
    if not isinstance(source["repository"], str) or not source["repository"]:
        raise ValueError(f"{name} repository must be nonempty")
    if source["repo_type"] != "dataset":
        raise ValueError(f"{name} repo_type must be dataset")
    _validate_commit(source["revision"], f"{name} revision")
    if not isinstance(source["license"], str) or not source["license"]:
        raise ValueError(f"{name} license must be nonempty")

    if name == "fineweb_edu":
        _validate_string_list(source["files"], "fineweb_edu files")
        materialized = _require_mapping(
            source["materialized_jsonl"],
            "fineweb_edu materialized_jsonl",
        )
        _require_keys(
            materialized,
            {"path", "rows", "bytes", "sha256"},
            "fineweb_edu materialized_jsonl",
        )
        _validate_relative_path(materialized["path"], "materialized JSONL path")
        for field in ("rows", "bytes"):
            if (
                isinstance(materialized[field], bool)
                or not isinstance(materialized[field], int)
                or materialized[field] <= 0
            ):
                raise ValueError(f"materialized JSONL {field} must be positive")
        if (
            not isinstance(materialized["sha256"], str)
            or not _SHA256_RE.fullmatch(materialized["sha256"])
        ):
            raise ValueError("materialized JSONL sha256 must be lowercase SHA-256")
        if source["holdout_records"] != 64:
            raise ValueError("fineweb_edu holdout_records must be 64")
        return

    _validate_contract_reference(source["lock_path"], "Wikidata lock path")
    _validate_contract_reference(source["notice_path"], "Wikidata notice path")
    if source["scope"] != "wikidata5m-graph-3archive":
        raise ValueError("unexpected Wikidata source scope")
    if source["training_splits"] != [
        "inductive_train",
        "transductive_train",
    ]:
        raise ValueError("unexpected Wikidata training splits")
    if source["sealed_splits"] != ["inductive_test", "inductive_valid"]:
        raise ValueError("unexpected Wikidata sealed splits")
    split_files = _require_mapping(source["split_files"], "Wikidata split_files")
    expected_splits = set(source["training_splits"] + source["sealed_splits"])
    if set(split_files) != expected_splits:
        raise ValueError("Wikidata split_files must cover train and sealed splits")
    for split, path in split_files.items():
        _validate_relative_path(path, f"Wikidata split {split}")
    aliases = _require_mapping(source["alias_files"], "Wikidata alias_files")
    _require_keys(aliases, {"entities", "relations"}, "Wikidata alias_files")
    for kind, path in aliases.items():
        _validate_relative_path(path, f"Wikidata {kind} aliases")


def _validate_git_source(name: str, source: dict[str, Any]) -> None:
    common = {
        "transport",
        "repository",
        "commit",
        "license",
        "license_paths",
        "role",
    }
    expected = (
        common
        if name == "tiny_recursive_models"
        else common | {"training_paths", "evaluation_paths", "include_glob"}
    )
    _require_keys(source, expected, f"{name} source")
    if source["transport"] != "git":
        raise ValueError(f"{name} transport must be git")
    if not isinstance(source["repository"], str) or not source["repository"]:
        raise ValueError(f"{name} repository must be nonempty")
    _validate_commit(source["commit"], f"{name} commit")
    if not isinstance(source["license"], str) or not source["license"]:
        raise ValueError(f"{name} license must be nonempty")
    _validate_string_list(source["license_paths"], f"{name} license_paths")
    expected_role = (
        "augmentation_reference"
        if name == "tiny_recursive_models"
        else "puzzle"
    )
    if source["role"] != expected_role:
        raise ValueError(f"{name} role must be {expected_role}")
    if name == "tiny_recursive_models":
        return
    _validate_string_list(source["training_paths"], f"{name} training_paths")
    evaluation = source["evaluation_paths"]
    if not isinstance(evaluation, list):
        raise ValueError(f"{name} evaluation_paths must be a list")
    for index, path in enumerate(evaluation):
        _validate_relative_path(path, f"{name} evaluation_paths[{index}]")
    if len(evaluation) != len(set(evaluation)):
        raise ValueError(f"{name} evaluation_paths contains duplicates")
    if source["include_glob"] != "**/*.json":
        raise ValueError(f"{name} include_glob must be **/*.json")


def _validate_license_registry(
    value: Any,
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    registry = _require_mapping(value, "license registry")
    _require_keys(
        registry,
        {"format", "dataset_id", "sources"},
        "license registry",
    )
    if registry["format"] != "memorysplit-current-dataset-licenses":
        raise ValueError("unexpected license registry format")
    if registry["dataset_id"] != "relational-chinchilla":
        raise ValueError("license registry dataset_id mismatch")
    entries = _require_mapping(registry["sources"], "license registry sources")
    if set(entries) != _SOURCE_NAMES:
        raise ValueError("license registry must cover every current source")
    for name, source in sources.items():
        entry = _require_mapping(entries[name], f"{name} license entry")
        if entry.get("spdx") != source["license"]:
            raise ValueError(f"{name} license registry mismatch")
    return registry


def load_dataset_lock(path: Path) -> CurrentDatasetLock:
    lock_path = Path(path)
    raw = _require_mapping(_strict_json_path(lock_path), "dataset lock")
    _require_keys(
        raw,
        {
            "format",
            "dataset_id",
            "licenses_path",
            "sources",
            "implementation",
        },
        "dataset lock",
    )
    if raw["format"] != "memorysplit-current-dataset-lock":
        raise ValueError("unexpected dataset lock format")
    if raw["dataset_id"] != "relational-chinchilla":
        raise ValueError("dataset_id must be exactly relational-chinchilla")
    if raw["implementation"] != _REQUIRED_IMPLEMENTATION:
        raise ValueError("implementation contract differs from the approved lock")

    raw_sources = _require_mapping(raw["sources"], "dataset sources")
    if set(raw_sources) != _SOURCE_NAMES:
        raise ValueError("dataset lock must contain exactly the current sources")
    sources: dict[str, dict[str, Any]] = {}
    for name, raw_source in raw_sources.items():
        source = _require_mapping(raw_source, f"{name} source")
        if name in {"fineweb_edu", "wikidata5m"}:
            _validate_huggingface_source(name, source)
        else:
            _validate_git_source(name, source)
        sources[name] = source

    wikidata_source = sources["wikidata5m"]
    wikidata_lock_path = (
        lock_path.parent / wikidata_source["lock_path"]
    ).resolve()
    wikidata_lock = WikidataLock.from_path(wikidata_lock_path)
    if (
        wikidata_lock.repo_id != wikidata_source["repository"]
        or wikidata_lock.repo_type != wikidata_source["repo_type"]
        or wikidata_lock.revision != wikidata_source["revision"]
    ):
        raise ValueError("Wikidata archive lock does not match dataset lock")

    licenses_relative = _validate_contract_reference(
        raw["licenses_path"],
        "licenses_path",
    )
    licenses_path = (lock_path.parent / licenses_relative).resolve()
    licenses = _validate_license_registry(
        _strict_json_path(licenses_path),
        sources,
    )
    notice_path = (
        lock_path.parent / wikidata_source["notice_path"]
    ).resolve()
    if not notice_path.is_file() or notice_path.is_symlink():
        raise ValueError(f"missing Wikidata license notice: {notice_path}")

    return CurrentDatasetLock(
        path=lock_path.resolve(),
        dataset_id=raw["dataset_id"],
        sources=sources,
        implementation=raw["implementation"],
        licenses_path=licenses_path,
        licenses=licenses,
        wikidata_lock=wikidata_lock,
        raw=raw,
    )


def canonical_task_sha256(task: object) -> str:
    canonical = json.dumps(
        task,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _source_cache(data_root: Path) -> Path:
    configured = os.environ.get("MEMORYSPLIT_SOURCE_CACHE")
    return Path(configured) if configured else data_root / "hf-cache"


def _hf_download_command(
    source: Mapping[str, Any],
    files: list[str],
    destination: Path,
    cache_dir: Path,
) -> list[str]:
    command = [
        "hf",
        "download",
        source["repository"],
        "--repo-type",
        source["repo_type"],
        "--revision",
        source["revision"],
    ]
    for path in files:
        command.extend(["--include", path])
    command.extend(
        [
            "--local-dir",
            str(destination),
            "--cache-dir",
            str(cache_dir),
        ]
    )
    return command


def _git_operations(
    name: str,
    source: Mapping[str, Any],
    cache_dir: Path,
    archive_path: Path,
) -> list[dict[str, Any]]:
    repository_cache = cache_dir / "git" / f"{name}.git"
    return [
        {
            "source": name,
            "kind": "git_init",
            "argv": ["git", "init", "--bare", str(repository_cache)],
        },
        {
            "source": name,
            "kind": "git_fetch",
            "argv": [
                "git",
                "-C",
                str(repository_cache),
                "fetch",
                "--depth",
                "1",
                source["repository"],
                source["commit"],
            ],
        },
        {
            "source": name,
            "kind": "git_archive",
            "argv": [
                "git",
                "-C",
                str(repository_cache),
                "archive",
                "--format=tar.gz",
                f"--output={archive_path}",
                source["commit"],
            ],
        },
    ]


def _staging_plan(lock: CurrentDatasetLock, data_root: Path) -> dict[str, Any]:
    cache_dir = _source_cache(data_root)
    placeholder = data_root / lock.dataset_id / ".sources-stage"
    downloads = placeholder / ".downloads"
    archives = placeholder / ".git-archives"
    operations: list[dict[str, Any]] = []

    fineweb = lock.sources["fineweb_edu"]
    operations.append(
        {
            "source": "fineweb_edu",
            "kind": "huggingface_download",
            "argv": _hf_download_command(
                fineweb,
                fineweb["files"],
                downloads / "fineweb_edu",
                cache_dir,
            ),
        }
    )
    wikidata = lock.sources["wikidata5m"]
    operations.append(
        {
            "source": "wikidata5m",
            "kind": "huggingface_download",
            "argv": _hf_download_command(
                wikidata,
                list(lock.wikidata_lock.files),
                downloads / "wikidata5m",
                cache_dir,
            ),
        }
    )
    for name in (
        "tiny_recursive_models",
        "arc_agi_1",
        "arc_agi_2",
        "conceptarc",
    ):
        operations.extend(
            _git_operations(
                name,
                lock.sources[name],
                cache_dir,
                archives / f"{name}.tar.gz",
            )
        )

    return {
        "format": "memorysplit-current-source-staging-plan",
        "dataset_id": lock.dataset_id,
        "execute": False,
        "source_root": str(data_root / lock.dataset_id / "sources"),
        "manifest_path": str(
            data_root / lock.dataset_id / "sources" / "source-manifest.json"
        ),
        "cache_dir": str(cache_dir),
        "operations": operations,
    }


def _safe_local_file(root: Path, relative: str) -> Path:
    _validate_relative_path(relative, "source file")
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise SourceDriftError(f"source file crosses symlink: {relative}")
    if not current.is_file():
        raise SourceDriftError(f"missing source file: {relative}")
    return current


def _copy_selected_files(
    source_root: Path,
    destination: Path,
    files: list[str],
) -> None:
    for relative in files:
        source = _safe_local_file(source_root, relative)
        target = destination.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as input_stream, target.open("xb") as output:
            shutil.copyfileobj(input_stream, output)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def _download_or_copy_huggingface(
    source: Mapping[str, Any],
    files: list[str],
    download_root: Path,
    destination: Path,
    cache_dir: Path,
) -> None:
    local_repository = Path(source["repository"])
    if local_repository.is_dir():
        _copy_selected_files(local_repository, destination, files)
        return

    _run(
        _hf_download_command(
            source,
            files,
            download_root,
            cache_dir,
        )
    )
    _copy_selected_files(download_root, destination, files)


def _stage_git_source(
    name: str,
    source: Mapping[str, Any],
    cache_dir: Path,
    archive_root: Path,
    destination: Path,
) -> None:
    repository_cache = cache_dir / "git" / f"{name}.git"
    if repository_cache.is_symlink():
        raise SourceDriftError(f"Git cache is a symlink: {repository_cache}")
    if not repository_cache.exists():
        repository_cache.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", "--bare", str(repository_cache)])
    elif not repository_cache.is_dir():
        raise SourceDriftError(f"Git cache is not a directory: {repository_cache}")

    _run(
        [
            "git",
            "-C",
            str(repository_cache),
            "fetch",
            "--depth",
            "1",
            source["repository"],
            source["commit"],
        ]
    )
    fetched = _run(
        [
            "git",
            "-C",
            str(repository_cache),
            "rev-parse",
            "FETCH_HEAD^{commit}",
        ]
    ).stdout.strip()
    if fetched != source["commit"]:
        raise SourceDriftError(
            f"{name} fetched commit drift: expected {source['commit']}, got {fetched}"
        )

    source_archive_root = archive_root / name
    source_archive_root.mkdir(parents=True)
    archive_path = source_archive_root / f"{name}.tar.gz"
    _run(
        [
            "git",
            "-C",
            str(repository_cache),
            "archive",
            "--format=tar.gz",
            f"--output={archive_path}",
            source["commit"],
        ]
    )
    archive_bytes, archive_sha256 = size_and_sha256(archive_path)
    archive_lock = WikidataLock(
        repo_id=source["repository"],
        repo_type="git",
        revision=source["commit"],
        files={
            archive_path.name: ArchiveLock(
                bytes=archive_bytes,
                sha256=archive_sha256,
            )
        },
    )
    safe_extract_archives(
        source_archive_root,
        destination,
        lock=archive_lock,
    )


def _inventory_files(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise SourceDriftError(f"missing regular source directory: {root}")
    records: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
        for entry in ordered:
            path = Path(entry.path)
            if entry.is_symlink():
                raise SourceDriftError(f"source tree contains symlink: {path}")
            if entry.is_dir(follow_symlinks=False):
                visit(path)
            elif entry.is_file(follow_symlinks=False):
                size, sha256 = size_and_sha256(path)
                records.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "bytes": size,
                        "sha256": sha256,
                    }
                )
            else:
                raise SourceDriftError(
                    f"source tree contains non-regular entry: {path}"
                )

    visit(root)
    return records


def _record_by_path(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {record["path"]: record for record in records}


def _puzzle_files(
    tree: Path,
    roots: list[str],
    include_glob: str,
) -> list[Path]:
    found: dict[str, Path] = {}
    for relative_root in roots:
        root = tree.joinpath(*PurePosixPath(relative_root).parts)
        if not root.is_dir() or root.is_symlink():
            raise SourceDriftError(f"missing puzzle source directory: {relative_root}")
        for path in root.glob(include_glob):
            if path.is_symlink() or not path.is_file():
                raise SourceDriftError(f"unsafe puzzle source entry: {path}")
            relative = path.relative_to(tree).as_posix()
            found[relative] = path
    return [found[path] for path in sorted(found)]


def _task_record(
    name: str,
    source: Mapping[str, Any],
    tree: Path,
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    content = path.read_bytes()
    task = _strict_json_bytes(
        content,
        description=f"{name}:{path.relative_to(tree).as_posix()}",
    )
    if not isinstance(task, dict):
        raise ValueError(
            f"puzzle task must be a JSON object: "
            f"{name}:{path.relative_to(tree).as_posix()}"
        )
    return task, {
        "source": name,
        "repository": source["repository"],
        "commit": source["commit"],
        "path": path.relative_to(tree).as_posix(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "canonical_task_sha256": canonical_task_sha256(task),
        "license": source["license"],
    }


def _build_puzzle_manifest(
    lock: CurrentDatasetLock,
    source_root: Path,
) -> dict[str, Any]:
    evaluation_hashes: set[str] = set()
    evaluation_files: list[dict[str, Any]] = []
    for name in _PUZZLE_SOURCE_ORDER:
        source = lock.sources[name]
        tree = source_root / "git" / name
        for path in _puzzle_files(
            tree,
            source["evaluation_paths"],
            source["include_glob"],
        ):
            _, record = _task_record(name, source, tree, path)
            evaluation_hashes.add(record["canonical_task_sha256"])
            evaluation_files.append(
                {
                    "source": name,
                    "path": record["path"],
                    "canonical_task_sha256": record[
                        "canonical_task_sha256"
                    ],
                }
            )

    accepted: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    first_by_hash: dict[str, str] = {}
    for name in _PUZZLE_SOURCE_ORDER:
        source = lock.sources[name]
        tree = source_root / "git" / name
        for path in _puzzle_files(
            tree,
            source["training_paths"],
            source["include_glob"],
        ):
            _, record = _task_record(name, source, tree, path)
            task_hash = record["canonical_task_sha256"]
            identity = f"{name}:{record['path']}"
            if task_hash in evaluation_hashes:
                raise ValueError(
                    f"training file is an official evaluation task: {identity}"
                )
            if task_hash in first_by_hash:
                duplicates.append(
                    {
                        "canonical_task_sha256": task_hash,
                        "duplicate_of": first_by_hash[task_hash],
                        "path": record["path"],
                        "source": name,
                    }
                )
                continue
            first_by_hash[task_hash] = identity
            accepted.append(record)

    return {
        "canonicalization": lock.implementation["arc_tasks"]["canonical_hash"],
        "accepted_files": accepted,
        "duplicate_training_files": duplicates,
        "evaluation_files": evaluation_files,
        "evaluation_task_hashes": sorted(evaluation_hashes),
    }


def _audit_wikidata_splits(
    source: Mapping[str, Any],
    files_root: Path,
) -> dict[str, Any]:
    split_rows: dict[str, dict[str, int]] = {}
    training_rows: list[Triple] = []
    sealed_rows: list[Triple] = []
    for split in source["training_splits"] + source["sealed_splits"]:
        path = files_root / source["split_files"][split]
        rows = list(iter_triples(path))
        distinct = set(rows)
        split_rows[split] = {
            "rows": len(rows),
            "distinct_triples": len(distinct),
            "duplicate_rows": len(rows) - len(distinct),
        }
        if split in source["training_splits"]:
            training_rows.extend(rows)
        else:
            sealed_rows.extend(rows)

    training_distinct = set(training_rows)
    sealed_distinct = set(sealed_rows)
    overlap = training_distinct & sealed_distinct
    if overlap:
        triple = min(
            overlap,
            key=lambda item: (item.subject, int(item.relation[1:]), item.object),
        )
        raise ValueError(
            "Wikidata evaluation triple appears in training: "
            f"Q{triple.subject}\t{triple.relation}\tQ{triple.object}"
        )
    return {
        "split_rows": split_rows,
        "training_distinct_triples": len(training_distinct),
        "training_duplicate_rows": len(training_rows) - len(training_distinct),
        "sealed_distinct_triples": len(sealed_distinct),
        "train_sealed_overlap": 0,
    }


def _build_manifest(
    lock: CurrentDatasetLock,
    source_root: Path,
) -> dict[str, Any]:
    fineweb_source = lock.sources["fineweb_edu"]
    fineweb_root = source_root / "fineweb_edu"
    fineweb_files = _inventory_files(fineweb_root)
    if set(_record_by_path(fineweb_files)) != set(fineweb_source["files"]):
        raise SourceDriftError("FineWeb staged files differ from the source lock")

    wikidata_source = lock.sources["wikidata5m"]
    wikidata_archive_root = source_root / "wikidata5m" / "archives"
    verified_archives = verify_archives(
        wikidata_archive_root,
        lock.wikidata_lock,
    )
    wikidata_files_root = source_root / "wikidata5m" / "files"
    wikidata_files = _inventory_files(wikidata_files_root)
    expected_wikidata_files = set(wikidata_source["split_files"].values()) | set(
        wikidata_source["alias_files"].values()
    )
    if set(_record_by_path(wikidata_files)) != expected_wikidata_files:
        raise SourceDriftError(
            "Wikidata extracted members differ from the source lock"
        )
    wikidata_audit = _audit_wikidata_splits(
        wikidata_source,
        wikidata_files_root,
    )
    alias_rows: dict[str, int] = {}
    for kind, prefix in (("entities", "Q"), ("relations", "P")):
        alias_rows[kind] = len(
            _read_optional_aliases(
                wikidata_files_root / wikidata_source["alias_files"][kind],
                prefix,
            )
        )

    git_sources: dict[str, Any] = {}
    for name in sorted(_SOURCE_NAMES - {"fineweb_edu", "wikidata5m"}):
        source = lock.sources[name]
        tree = source_root / "git" / name
        files = _inventory_files(tree)
        inventory = _record_by_path(files)
        for license_path in source["license_paths"]:
            if license_path not in inventory:
                raise SourceDriftError(
                    f"{name} is missing source license file {license_path}"
                )
        git_sources[name] = {
            "repository": source["repository"],
            "commit": source["commit"],
            "role": source["role"],
            "license": source["license"],
            "license_paths": source["license_paths"],
            "files": files,
        }

    licenses_root = source_root / "licenses"
    license_files = _inventory_files(licenses_root)
    staged_registry = _strict_json_path(
        licenses_root / "current-dataset-licenses.json"
    )
    if staged_registry != lock.licenses:
        raise SourceDriftError("staged license registry differs from lock")

    lock_sha256 = hashlib.sha256(lock.canonical_bytes()).hexdigest()
    return {
        "format": "memorysplit-current-source-manifest",
        "dataset_id": lock.dataset_id,
        "dataset_lock_sha256": lock_sha256,
        "fineweb_edu": {
            "repository": fineweb_source["repository"],
            "revision": fineweb_source["revision"],
            "files": fineweb_files,
            "materialized_jsonl_lock": fineweb_source["materialized_jsonl"],
            "holdout_records": fineweb_source["holdout_records"],
            "license": fineweb_source["license"],
        },
        "wikidata": {
            "scope": wikidata_source["scope"],
            "repository": wikidata_source["repository"],
            "revision": wikidata_source["revision"],
            "archives": [
                {
                    "path": path.name,
                    "bytes": lock.wikidata_lock.files[path.name].bytes,
                    "sha256": lock.wikidata_lock.files[path.name].sha256,
                }
                for path in verified_archives
            ],
            "files": wikidata_files,
            "training_splits": wikidata_source["training_splits"],
            "sealed_splits": wikidata_source["sealed_splits"],
            "split_files": wikidata_source["split_files"],
            "alias_files": wikidata_source["alias_files"],
            "alias_rows": alias_rows,
            "license": wikidata_source["license"],
            **wikidata_audit,
        },
        "git_sources": git_sources,
        "puzzles": _build_puzzle_manifest(lock, source_root),
        "licenses": {
            "registry": "licenses/current-dataset-licenses.json",
            "files": license_files,
        },
    }


def _read_canonical_manifest(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = _require_mapping(
        _strict_json_bytes(raw, description=str(path)),
        "source manifest",
    )
    if raw != _canonical_bytes(value):
        raise SourceDriftError("source manifest is not canonical JSON")
    return value


def verify_current_sources(
    lock: CurrentDatasetLock,
    source_root: Path,
) -> dict:
    root = Path(source_root)
    if not root.is_dir() or root.is_symlink():
        raise SourceDriftError(f"missing current source root: {root}")
    manifest_path = root / "source-manifest.json"
    recorded: dict[str, Any] | None = None
    if manifest_path.exists() or manifest_path.is_symlink():
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise SourceDriftError("source manifest is not a regular file")
        recorded = _read_canonical_manifest(manifest_path)

    try:
        computed = _build_manifest(lock, root)
    except (OSError, ValueError, SourceDriftError) as exc:
        if recorded is not None:
            raise SourceDriftError(f"source manifest drift: {exc}") from exc
        raise

    if recorded is not None and recorded != computed:
        raise SourceDriftError("source manifest drift: staged bytes changed")
    return computed


def _copy_license_contract(
    lock: CurrentDatasetLock,
    source_root: Path,
) -> None:
    licenses_root = source_root / "licenses"
    licenses_root.mkdir(parents=True)
    shutil.copyfile(
        lock.licenses_path,
        licenses_root / "current-dataset-licenses.json",
    )
    wikidata = lock.sources["wikidata5m"]
    notice_path = (lock.path.parent / wikidata["notice_path"]).resolve()
    shutil.copyfile(notice_path, licenses_root / notice_path.name)


def _execute_staging(
    lock: CurrentDatasetLock,
    data_root: Path,
    private_root: Path,
) -> dict[str, Any]:
    cache_dir = _source_cache(data_root)
    download_root = private_root / ".downloads"

    fineweb = lock.sources["fineweb_edu"]
    _download_or_copy_huggingface(
        fineweb,
        fineweb["files"],
        download_root / "fineweb_edu",
        private_root / "fineweb_edu",
        cache_dir,
    )

    wikidata = lock.sources["wikidata5m"]
    wikidata_archives = private_root / "wikidata5m" / "archives"
    _download_or_copy_huggingface(
        wikidata,
        list(lock.wikidata_lock.files),
        download_root / "wikidata5m",
        wikidata_archives,
        cache_dir,
    )
    safe_extract_archives(
        wikidata_archives,
        private_root / "wikidata5m" / "files",
        lock=lock.wikidata_lock,
    )

    git_archive_root = private_root / ".git-archives"
    for name in (
        "tiny_recursive_models",
        "arc_agi_1",
        "arc_agi_2",
        "conceptarc",
    ):
        _stage_git_source(
            name,
            lock.sources[name],
            cache_dir,
            git_archive_root,
            private_root / "git" / name,
        )

    shutil.rmtree(download_root, ignore_errors=True)
    shutil.rmtree(git_archive_root, ignore_errors=True)
    _copy_license_contract(lock, private_root)
    manifest = verify_current_sources(lock, private_root)
    (private_root / "source-manifest.json").write_bytes(
        _canonical_bytes(manifest)
    )
    return verify_current_sources(lock, private_root)


def stage_current_sources(
    lock: CurrentDatasetLock,
    data_root: Path,
    *,
    execute: bool,
) -> dict:
    root = Path(data_root)
    if not execute:
        return _staging_plan(lock, root)

    dataset_root = root / lock.dataset_id
    final_root = dataset_root / "sources"
    if final_root.exists() or final_root.is_symlink():
        return verify_current_sources(lock, final_root)
    if dataset_root.is_symlink():
        raise SourceDriftError(f"dataset root is a symlink: {dataset_root}")
    dataset_root.mkdir(parents=True, exist_ok=True)
    private_root = Path(
        tempfile.mkdtemp(
            prefix=".sources.partial-",
            dir=dataset_root,
        )
    )
    try:
        manifest = _execute_staging(lock, root, private_root)
        if final_root.exists() or final_root.is_symlink():
            raise FileExistsError(f"source root appeared during staging: {final_root}")
        private_root.rename(final_root)
    except BaseException:
        shutil.rmtree(private_root, ignore_errors=True)
        raise

    verified = verify_current_sources(lock, final_root)
    if verified != manifest:
        raise SourceDriftError("source manifest changed during atomic publication")
    return verified


def _manifest_for_iteration(source_root: Path) -> dict[str, Any]:
    root = Path(source_root)
    manifest = _read_canonical_manifest(root / "source-manifest.json")
    if (
        manifest.get("format") != "memorysplit-current-source-manifest"
        or manifest.get("dataset_id") != "relational-chinchilla"
    ):
        raise SourceDriftError("iterator requires a current source manifest")
    return manifest


def _verify_manifest_file(
    path: Path,
    expected: Mapping[str, Any],
) -> None:
    if not path.is_file() or path.is_symlink():
        raise SourceDriftError(f"missing manifested source file: {path}")
    size, sha256 = size_and_sha256(path)
    if size != expected["bytes"] or sha256 != expected["sha256"]:
        raise SourceDriftError(f"manifested source file drift: {path}")


@dataclass(frozen=True)
class TrainingTriple:
    split: str
    row: int
    subject: int
    relation: str
    object: int


def iter_training_triples(source_root: Path) -> Iterator[TrainingTriple]:
    root = Path(source_root)
    manifest = _manifest_for_iteration(root)
    wikidata = manifest["wikidata"]
    inventory = _record_by_path(wikidata["files"])
    for split in wikidata["training_splits"]:
        relative = wikidata["split_files"][split]
        path = root / "wikidata5m" / "files" / relative
        _verify_manifest_file(path, inventory[relative])
        for row_number, triple in enumerate(iter_triples(path), 1):
            yield TrainingTriple(
                split=split,
                row=row_number,
                subject=triple.subject,
                relation=triple.relation,
                object=triple.object,
            )


@dataclass(frozen=True)
class AliasRecord:
    canonical_id: str
    kind: str
    display: str
    aliases: tuple[str, ...]


def _display_alias(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _read_optional_aliases(
    path: Path,
    prefix: str,
) -> dict[str, tuple[str, ...]]:
    parse_id = parse_pid if prefix == "P" else parse_qid
    result: dict[str, tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            row = line.rstrip("\r\n")
            if not row:
                continue
            fields = row.split("\t")
            if len(fields) < 2:
                raise ValueError(
                    f"{path.name}:{line_number}: expected ID and aliases"
                )
            canonical_id = fields[0]
            try:
                parse_id(canonical_id)
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}:{line_number}: {exc}"
                ) from exc
            if canonical_id in result:
                raise ValueError(
                    f"{path.name}:{line_number}: duplicate canonical ID"
                )
            aliases: dict[str, str] = {}
            for raw_alias in fields[1:]:
                display = _display_alias(raw_alias)
                if display:
                    aliases.setdefault(normalize_alias(display), display)
            result[canonical_id] = tuple(aliases.values())
    return result


def iter_aliases(source_root: Path) -> Iterator[AliasRecord]:
    root = Path(source_root)
    manifest = _manifest_for_iteration(root)
    wikidata = manifest["wikidata"]
    inventory = _record_by_path(wikidata["files"])
    for kind, prefix in (("entities", "Q"), ("relations", "P")):
        relative = wikidata["alias_files"][kind]
        path = root / "wikidata5m" / "files" / relative
        _verify_manifest_file(path, inventory[relative])
        raw_aliases = _read_optional_aliases(path, prefix)
        nonempty = {
            canonical_id: aliases
            for canonical_id, aliases in raw_aliases.items()
            if aliases
        }
        catalog = canonicalize_aliases(nonempty)
        ordered_ids = sorted(
            raw_aliases,
            key=lambda value: int(value[1:]),
        )
        for canonical_id in ordered_ids:
            aliases = catalog[canonical_id] if canonical_id in catalog else ()
            yield AliasRecord(
                canonical_id=canonical_id,
                kind="entity" if prefix == "Q" else "relation",
                display=aliases[0] if aliases else canonical_id,
                aliases=aliases,
            )


@dataclass(frozen=True)
class PuzzleTask:
    source: str
    repository: str
    commit: str
    path: str
    sha256: str
    canonical_task_sha256: str
    license: str
    task: dict[str, Any]


def iter_puzzle_tasks(source_root: Path) -> Iterator[PuzzleTask]:
    root = Path(source_root)
    manifest = _manifest_for_iteration(root)
    for record in manifest["puzzles"]["accepted_files"]:
        relative = _validate_relative_path(record["path"], "puzzle task path")
        source = record["source"]
        if source not in _PUZZLE_SOURCE_ORDER:
            raise SourceDriftError(f"unexpected puzzle source: {source}")
        path = root / "git" / source
        path = path.joinpath(*PurePosixPath(relative).parts)
        _verify_manifest_file(path, record)
        task = _strict_json_path(path)
        if not isinstance(task, dict):
            raise ValueError(f"puzzle task must be an object: {source}:{relative}")
        task_hash = canonical_task_sha256(task)
        if task_hash != record["canonical_task_sha256"]:
            raise SourceDriftError(f"canonical puzzle task drift: {source}:{relative}")
        yield PuzzleTask(
            source=source,
            repository=record["repository"],
            commit=record["commit"],
            path=relative,
            sha256=record["sha256"],
            canonical_task_sha256=task_hash,
            license=record["license"],
            task=task,
        )
