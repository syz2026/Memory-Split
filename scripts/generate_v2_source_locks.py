#!/usr/bin/env python3
"""Generate reviewed MemorySplit v2 source locks from pinned upstream bytes.

This is a networked maintainer tool, not part of normal corpus staging.  Every
revision is intentionally hard-coded.  Changing a revision or accepting a
changed archive checksum therefore requires a source-control review.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import HfApi, hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "sources" / "memorysplit-v2"

HF_PINS = {
    "fineweb_edu": {
        "repository": "HuggingFaceFW/fineweb-edu",
        "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "license": "ODC-By-1.0",
    },
    "finemath": {
        "repository": "HuggingFaceTB/finemath",
        "revision": "e92b25a616738fe95dc186b64dfb19f9c8525594",
        "license": "ODC-By-1.0",
    },
    "wikidata5m": {
        "repository": "intfloat/wikidata5m",
        "revision": "6b2b09672129e280c0c9da97ab58154e9d535e6b",
        "license": "CC0-1.0",
    },
}

_AUXILIARY_SOURCES = (
    {
        "id": "deepmind_mathematics_generator",
        "training_use": "deterministically_generated_objective_question_answer_pairs",
        "components": (
            {
                "id": "mathematics_dataset",
                "repository": "https://github.com/google-deepmind/mathematics_dataset",
                "commit": "427f45075f84b8b9774950196ad63867ca20ffb3",
                "archive_name": "mathematics_dataset.tar.gz",
                "license": "Apache-2.0",
                "license_paths": ("LICENSE",),
                "include": ("*",),
            },
        ),
    },
    {
        "id": "clrs_text",
        "training_use": "clrs_text_generator_outputs_with_objective_traces",
        "components": (
            {
                "id": "clrs",
                "repository": "https://github.com/google-deepmind/clrs",
                "commit": "b846b4a2cb01fb3dde2bbed4039b6a05dc45e171",
                "archive_name": "clrs.tar.gz",
                "license": "Apache-2.0",
                "license_paths": ("LICENSE",),
                "include": (
                    "LICENSE",
                    "README.md",
                    "MANIFEST.in",
                    "setup.py",
                    "requirements/*",
                    "clrs/*",
                ),
                # The colab directory contains published benchmark metrics and
                # model outputs, not generator inputs.
                "exclude": ("clrs/_src/clrs_text/colabs/*",),
            },
        ),
    },
    {
        "id": "ruletaker",
        "training_use": "fresh_generator_outputs_with_theorem_prover_labels",
        "components": (
            {
                "id": "ruletaker",
                "repository": "https://github.com/allenai/ruletaker",
                "commit": "abaacec9364992eff5ec4555b837e20fee2f2ff0",
                "archive_name": "ruletaker.tar.gz",
                "license": "Apache-2.0",
                "license_paths": ("LICENSE",),
                "include": ("*",),
            },
        ),
    },
    {
        "id": "prontoqa",
        "training_use": "fresh_generator_outputs_with_formal_proofs",
        "components": (
            {
                "id": "prontoqa",
                "repository": "https://github.com/asaparov/prontoqa",
                "commit": "0a6412b6fddf46324a1cb96e066dd7b3d89b87d6",
                "archive_name": "prontoqa.tar.gz",
                "license": "Apache-2.0",
                "license_paths": ("LICENSE",),
                # Published benchmark/model-output ZIPs are deliberately not
                # staged into the training-eligible source tree.
                "include": ("LICENSE", "README.md", "*.py", "bad_patterns.txt"),
            },
        ),
    },
    {
        "id": "reasoning_gym_exact_answer",
        "training_use": "seeded_generators_with_exact_answer_or_solver_validation",
        "components": (
            {
                "id": "reasoning_gym",
                "repository": "https://github.com/open-thought/reasoning-gym",
                "commit": "49b07130b3fcd12f2d064bba7c43869543a0e7e7",
                "archive_name": "reasoning-gym.tar.gz",
                "license": "Apache-2.0",
                "license_paths": ("LICENSE", "NOTICE.txt"),
                "include": (
                    "LICENSE",
                    "NOTICE.txt",
                    "README.md",
                    "pyproject.toml",
                    "reasoning_gym/*",
                ),
            },
        ),
    },
    {
        "id": "arc_agi_training",
        "training_use": "official_training_tasks_only",
        "components": (
            {
                "id": "arc_agi_1",
                "repository": "https://github.com/fchollet/ARC-AGI",
                "commit": "399030444e0ab0cc8b4e199870fb20b863846f34",
                "archive_name": "arc-agi.tar.gz",
                "license": "Apache-2.0",
                "license_paths": ("LICENSE",),
                "include": ("LICENSE", "README.md", "data/training/*.json"),
            },
            {
                "id": "arc_agi_2",
                "repository": "https://github.com/arcprize/ARC-AGI-2",
                "commit": "f3283f727488ad98fe575ea6a5ac981e4a188e49",
                "archive_name": "arc-agi-2.tar.gz",
                "license": "Apache-2.0",
                "license_paths": ("LICENSE",),
                "include": ("LICENSE", "readme.md", "data/training/*.json"),
            },
        ),
    },
    {
        "id": "conceptarc_training",
        "training_use": "conceptarc_corpus_training_tasks_only",
        "components": (
            {
                "id": "conceptarc",
                "repository": "https://github.com/victorvikram/ConceptARC",
                "commit": "0e67da6af879e4bad3d7cd3c196e8d551b445725",
                "archive_name": "conceptarc.tar.gz",
                "license": "MIT",
                "license_paths": ("LICENSE",),
                "include": ("LICENSE", "README.md", "corpus/*.json"),
            },
        ),
    },
)

_WIKIDATA_MEMBERS = {
    "wikidata5m_alias.tar.gz": (
        "wikidata5m_entity.txt",
        "wikidata5m_relation.txt",
    ),
    "wikidata5m_inductive.tar.gz": (
        "wikidata5m_inductive_test.txt",
        "wikidata5m_inductive_train.txt",
        "wikidata5m_inductive_valid.txt",
    ),
    "wikidata5m_transductive.tar.gz": (
        "wikidata5m_transductive_test.txt",
        "wikidata5m_transductive_train.txt",
        "wikidata5m_transductive_valid.txt",
    ),
}
_MISSING_PRODUCTION_LOCK_IDS = (
    "synthetic_graph_generator",
    "verified_synthetic_multihop_generator",
    "wikidata_path_reasoning_generator",
    "relational_refinement_generator",
    "reasoning_solver",
)
_MISSING_MATERIALIZED_LANES = (
    "fineweb_edu",
    "finemath",
    "wikidata_graph",
    "synthetic_graph",
    "verified_synthetic_multihop",
    "wikidata_path_reasoning",
    "relational_refinement",
    "objective_auxiliary",
)


def _json_bytes(value: object) -> bytes:
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


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _size_and_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _write_new_or_replace(path: Path, content: bytes, *, replace: bool) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"refusing to replace source lock: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _lfs_sha256(sibling: object) -> str:
    lfs = getattr(sibling, "lfs", None)
    digest = getattr(lfs, "sha256", None)
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(
            f"upstream file lacks immutable LFS SHA-256: "
            f"{getattr(sibling, 'rfilename', '<unknown>')}"
        )
    return digest


def _hf_records(
    api: HfApi,
    source_id: str,
    prefixes: tuple[str, ...],
) -> tuple[str, list[dict[str, Any]], Mapping[str, Any]]:
    pin = HF_PINS[source_id]
    info = api.dataset_info(
        pin["repository"],
        revision=pin["revision"],
        files_metadata=True,
    )
    if info.sha != pin["revision"]:
        raise RuntimeError(
            f"{source_id} revision drift: expected {pin['revision']}, got {info.sha}"
        )
    records = []
    for sibling in info.siblings:
        path = sibling.rfilename
        if not any(path.startswith(prefix) for prefix in prefixes):
            continue
        size = sibling.size
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise RuntimeError(f"invalid upstream byte count for {path}")
        records.append(
            {
                "bytes": size,
                "git_blob_sha1": sibling.blob_id,
                "path": path,
                "sha256": _lfs_sha256(sibling),
            }
        )
    records.sort(key=lambda item: item["path"])
    card = info.card_data.to_dict() if info.card_data else {}
    return info.sha, records, card


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "memorysplit-v2-source-lock-generator/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def _safe_relative(path: str) -> str:
    value = PurePosixPath(path)
    if (
        not path
        or value.is_absolute()
        or ".." in value.parts
        or "." in value.parts
        or "\\" in path
    ):
        raise RuntimeError(f"unsafe upstream archive path: {path!r}")
    return value.as_posix()


def _selected(
    path: str,
    patterns: Iterable[str],
    excluded: Iterable[str] = (),
) -> bool:
    return any(
        fnmatch.fnmatchcase(path, pattern) for pattern in patterns
    ) and not any(fnmatch.fnmatchcase(path, pattern) for pattern in excluded)


def _component_lock(
    component: Mapping[str, Any],
    *,
    archive_cache: Path | None,
    temporary: Path,
) -> dict[str, Any]:
    commit = component["commit"]
    repository = component["repository"]
    url = (
        repository.replace("https://github.com/", "https://codeload.github.com/")
        + f"/tar.gz/{commit}"
    )
    cached = (
        archive_cache / component["archive_name"]
        if archive_cache is not None
        else None
    )
    archive_path = temporary / component["archive_name"]
    if cached is not None and cached.is_file():
        shutil.copyfile(cached, archive_path)
    else:
        _download(url, archive_path)

    archive_content = archive_path.read_bytes()
    members: list[dict[str, Any]] = []
    roots: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        seen: set[str] = set()
        for member in archive.getmembers():
            full = _safe_relative(member.name.rstrip("/"))
            root, separator, relative = full.partition("/")
            roots.add(root)
            if not separator:
                if not member.isdir():
                    raise RuntimeError("archive root must be a directory")
                continue
            if member.isdir():
                continue
            if not member.isreg():
                raise RuntimeError(f"unsupported archive member type: {member.name}")
            relative = _safe_relative(relative)
            if not _selected(
                relative,
                component["include"],
                component.get("exclude", ()),
            ):
                continue
            if relative in seen:
                raise RuntimeError(f"duplicate selected archive member: {relative}")
            seen.add(relative)
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            content = stream.read()
            if len(content) != member.size:
                raise RuntimeError(f"short archive member: {member.name}")
            members.append(
                {
                    "bytes": len(content),
                    "path": relative,
                    "sha256": _sha256_bytes(content),
                }
            )
    if len(roots) != 1 or not members:
        raise RuntimeError(f"unexpected archive roots or empty selection for {repository}")
    members.sort(key=lambda item: item["path"])
    by_path = {item["path"]: item for item in members}
    for license_path in component["license_paths"]:
        if license_path not in by_path:
            raise RuntimeError(
                f"{component['id']} is missing locked license {license_path}"
            )

    return {
        "archive": {
            "bytes": len(archive_content),
            "root": next(iter(roots)),
            "sha256": _sha256_bytes(archive_content),
            "url": url,
        },
        "commit": commit,
        "files": members,
        "id": component["id"],
        "license": component["license"],
        "license_paths": list(component["license_paths"]),
        "repository": repository,
    }


def _fineweb_lock(api: HfApi) -> dict[str, Any]:
    revision, files, card = _hf_records(
        api,
        "fineweb_edu",
        ("sample/10BT/",),
    )
    if len(files) != 14:
        raise RuntimeError(f"FineWeb-Edu sample-10BT expected 14 shards, got {len(files)}")
    return {
        "files": files,
        "format": "memorysplit-v2-huggingface-source-lock",
        "license": HF_PINS["fineweb_edu"]["license"],
        "license_evidence": {
            "dataset_card_value": card.get("license"),
            "terms_url": "https://commoncrawl.org/terms-of-use",
        },
        "repo_type": "dataset",
        "repository": HF_PINS["fineweb_edu"]["repository"],
        "revision": revision,
        "schema_version": 1,
        "selection": {
            "config": "sample-10BT",
            "file_order": "lexicographic_path",
            "scope": "complete_config",
        },
        "source_id": "fineweb_edu",
        "total_bytes": sum(item["bytes"] for item in files),
    }


def _finemath_lock(api: HfApi) -> dict[str, Any]:
    revision, files, card = _hf_records(
        api,
        "finemath",
        ("finemath-3plus/", "finemath-4plus/"),
    )
    subset_metadata = {}
    for item in card.get("dataset_info", []):
        name = item.get("config_name")
        if name in {"finemath-3plus", "finemath-4plus"}:
            split = item["splits"][0]
            subset_metadata[name] = {
                "download_bytes": item["download_size"],
                "rows": split["num_examples"],
            }
    expected_counts = {"finemath-3plus": 128, "finemath-4plus": 64}
    actual_counts = {
        name: sum(item["path"].startswith(f"{name}/") for item in files)
        for name in expected_counts
    }
    if actual_counts != expected_counts:
        raise RuntimeError(f"unexpected FineMath shard counts: {actual_counts}")
    if set(subset_metadata) != set(expected_counts):
        raise RuntimeError("FineMath dataset card is missing pinned subset metadata")
    return {
        "deduplication": {
            "digest": "sha256",
            "encoding": "utf-8",
            "field": "text",
            "identity": "exact_text_bytes",
            "priority": [
                "fineweb_edu",
                "finemath-4plus",
                "finemath-3plus",
            ],
            "sidecar_encoding": "one_unsigned_byte_per_source_row_1_keep_0_drop",
        },
        "files": files,
        "format": "memorysplit-v2-huggingface-source-lock",
        "license": HF_PINS["finemath"]["license"],
        "license_evidence": {
            "dataset_card_value": card.get("license"),
            "terms_url": "https://commoncrawl.org/terms-of-use",
        },
        "repo_type": "dataset",
        "repository": HF_PINS["finemath"]["repository"],
        "revision": revision,
        "schema_version": 1,
        "selection": {
            "file_order": "lexicographic_path_within_subset",
            "subsets_in_order": [
                "finemath-4plus",
                "finemath-3plus-cross-deduplicated-remainder",
            ],
        },
        "source_id": "finemath",
        "subset_metadata": subset_metadata,
        "total_bytes": sum(item["bytes"] for item in files),
    }


def _wikidata_lock(
    api: HfApi,
    *,
    archive_cache: Path | None,
) -> dict[str, Any]:
    revision, files, _ = _hf_records(
        api,
        "wikidata5m",
        tuple(f"{name}" for name in _WIKIDATA_MEMBERS),
    )
    wanted = set(_WIKIDATA_MEMBERS)
    if {item["path"] for item in files} != wanted:
        raise RuntimeError("Wikidata5M archive selection differs from the frozen scope")
    for item in files:
        archive_name = item["path"]
        cached = archive_cache / archive_name if archive_cache is not None else None
        archive_path = (
            cached
            if cached is not None and cached.is_file()
            else Path(
                hf_hub_download(
                    repo_id=HF_PINS["wikidata5m"]["repository"],
                    filename=archive_name,
                    repo_type="dataset",
                    revision=revision,
                )
            )
        )
        archive_size, archive_sha256 = _size_and_sha256(archive_path)
        if archive_size != item["bytes"] or archive_sha256 != item["sha256"]:
            raise RuntimeError(f"cached Wikidata archive drift: {archive_name}")
        members = []
        with tarfile.open(archive_path, "r:gz") as archive:
            seen: set[str] = set()
            for member in archive.getmembers():
                relative = _safe_relative(member.name.rstrip("/"))
                if member.isdir():
                    continue
                if not member.isreg() or relative in seen:
                    raise RuntimeError(
                        f"unsafe or duplicate Wikidata member: {member.name}"
                    )
                seen.add(relative)
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"cannot read Wikidata member: {member.name}")
                digest = hashlib.sha256()
                size = 0
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    digest.update(chunk)
                if size != member.size:
                    raise RuntimeError(f"short Wikidata member: {member.name}")
                members.append(
                    {
                        "bytes": size,
                        "path": relative,
                        "sha256": digest.hexdigest(),
                    }
                )
        if seen != set(_WIKIDATA_MEMBERS[archive_name]):
            raise RuntimeError(
                f"Wikidata member inventory drift for {archive_name}: {sorted(seen)}"
            )
        item["members"] = sorted(members, key=lambda member: member["path"])
    return {
        "complete_once": {
            "sidecar_encoding": "one_unsigned_byte_per_source_row_1_keep_0_drop",
            "split_order": [
                "wikidata5m_inductive_train.txt",
                "wikidata5m_transductive_train.txt",
            ],
            "triple_identity": "exact_subject_relation_object",
        },
        "files": files,
        "format": "memorysplit-v2-huggingface-source-lock",
        "license": HF_PINS["wikidata5m"]["license"],
        "license_evidence": {
            "notice_path": "../Wikidata-CC0-1.0.txt",
            "notice_sha256": (
                "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499"
            ),
        },
        "repo_type": "dataset",
        "repository": HF_PINS["wikidata5m"]["repository"],
        "revision": revision,
        "schema_version": 1,
        "source_id": "wikidata5m",
        "total_bytes": sum(item["bytes"] for item in files),
    }


def generate_locks(
    output_dir: Path,
    *,
    archive_cache: Path | None,
    replace: bool,
) -> dict[str, str]:
    api = HfApi()
    locks = {
        "fineweb_edu": ("fineweb-edu.lock.json", _fineweb_lock(api)),
        "finemath": ("finemath.lock.json", _finemath_lock(api)),
        "wikidata5m": (
            "wikidata5m-complete-once.lock.json",
            _wikidata_lock(api, archive_cache=archive_cache),
        ),
    }
    with tempfile.TemporaryDirectory(prefix="memorysplit-v2-locks-") as temporary:
        temp_root = Path(temporary)
        auxiliary = {
            "format": "memorysplit-v2-objective-auxiliary-lock",
            "schema_version": 1,
            "sources": [],
        }
        for source in _AUXILIARY_SOURCES:
            auxiliary["sources"].append(
                {
                    "components": [
                        _component_lock(
                            component,
                            archive_cache=archive_cache,
                            temporary=temp_root,
                        )
                        for component in source["components"]
                    ],
                    "id": source["id"],
                    "training_use": source["training_use"],
                }
            )
    locks["objective_auxiliary"] = (
        "objective-auxiliaries.lock.json",
        auxiliary,
    )

    digests: dict[str, str] = {}
    rendered: dict[str, bytes] = {}
    for source_id, (filename, value) in locks.items():
        content = _json_bytes(value)
        rendered[filename] = content
        digests[source_id] = _sha256_bytes(content)

    scientific_contract = ROOT / "configs" / "reasoning-dataset-v2.json"
    scientific = json.loads(scientific_contract.read_text(encoding="utf-8"))
    policy = scientific["sprint_recipe"]["source_policy"]
    expected_auxiliary = [source["id"] for source in _AUXILIARY_SOURCES]
    if policy["objective_auxiliary_sources"] != expected_auxiliary:
        raise RuntimeError("objective auxiliary lock differs from scientific source policy")
    if policy["finemath_subsets_in_order"] != [
        "finemath-4plus",
        "finemath-3plus-cross-deduplicated-remainder",
    ]:
        raise RuntimeError("FineMath lock differs from scientific source policy")

    master = {
        "contract_id": "memorysplit-reasoning-dataset-v2",
        "dataset_id": "memorysplit-v2-frozen-upstream-sources",
        "format": "memorysplit-v2-source-set-lock",
        "production_readiness": {
            "missing_materialized_lanes": list(_MISSING_MATERIALIZED_LANES),
            "missing_source_locks": list(_MISSING_PRODUCTION_LOCK_IDS),
            "ready": False,
            "reason": (
                "upstream data locks are frozen, but production generator and "
                "solver locks plus materialized lane artifacts remain absent"
            ),
        },
        "schema_version": 1,
        "scientific_requirements": {
            "path": "../../configs/reasoning-dataset-v2.json",
            "sha256": _sha256_bytes(scientific_contract.read_bytes()),
        },
        "source_locks": [
            {
                "id": source_id,
                "path": locks[source_id][0],
                "sha256": digests[source_id],
            }
            for source_id in (
                "fineweb_edu",
                "finemath",
                "wikidata5m",
                "objective_auxiliary",
            )
        ],
    }
    rendered["source-set.lock.json"] = _json_bytes(master)
    digests["source_set"] = _sha256_bytes(rendered["source-set.lock.json"])

    for filename in sorted(rendered):
        _write_new_or_replace(
            output_dir / filename,
            rendered[filename],
            replace=replace,
        )
    return digests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--archive-cache",
        type=Path,
        help="optional directory containing the pinned codeload archive names",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace existing locks (review the resulting diff)",
    )
    args = parser.parse_args(argv)
    digests = generate_locks(
        args.output_dir,
        archive_cache=args.archive_cache,
        replace=args.replace,
    )
    print(json.dumps(digests, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
