"""Verified source extraction and schema binding for Wikidata robustness."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from corpusgen.relation_schema import RelationSchema, build_relation_schema
from corpusgen.wikidata5m import (
    WikidataLock,
    read_aliases,
    safe_extract_archives,
    verify_archives,
)


_REQUIRED_EXTRACTED_FILES = (
    "wikidata5m_relation.txt",
    "wikidata5m_entity.txt",
    "wikidata5m_transductive_train.txt",
    "wikidata5m_inductive_train.txt",
    "wikidata5m_inductive_valid.txt",
    "wikidata5m_inductive_test.txt",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def wikidata_lock_sha256(lock: WikidataLock) -> str:
    return hashlib.sha256(lock.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class VerifiedWikidataSource:
    split_path: Path
    source_sha256: str
    source_lock_sha256: str
    source_archive_sha256: Mapping[str, str]
    recomputed_schema: RelationSchema

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_archive_sha256",
            MappingProxyType(dict(sorted(self.source_archive_sha256.items()))),
        )


def _require_extracted_file(root: Path, name: str) -> Path:
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"verified extraction is missing {name}")
    return path


def verify_extract_and_bind_schema(
    archive_root: str | Path,
    extraction_root: str | Path,
    supplied_schema: RelationSchema,
    split: str,
    *,
    lock: WikidataLock,
) -> VerifiedWikidataSource:
    """Verify locked archives, extract privately, and rebuild schema exactly."""

    if split not in {"valid", "test"}:
        raise ValueError("Wikidata robustness split must be valid or test")
    source = Path(archive_root)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Wikidata source must be a regular archive directory")
    if not isinstance(lock, WikidataLock):
        raise TypeError("lock must be a WikidataLock")
    if not isinstance(supplied_schema, RelationSchema):
        raise TypeError("schema must be RelationSchema v2")

    verified_archives = verify_archives(source, lock)
    safe_extract_archives(source, extraction_root, lock=lock)
    extracted = Path(extraction_root)
    files = {
        name: _require_extracted_file(extracted, name)
        for name in _REQUIRED_EXTRACTED_FILES
    }

    selected_name = f"wikidata5m_inductive_{split}.txt"
    selected_path = files[selected_name]
    selected_hash = _sha256_file(selected_path)
    other_split = "test" if split == "valid" else "valid"
    comparison_names = (
        "wikidata5m_inductive_train.txt",
        f"wikidata5m_inductive_{other_split}.txt",
    )
    if selected_hash in {
        _sha256_file(files[name]) for name in comparison_names
    }:
        raise ValueError(
            "selected inductive split content duplicates another split"
        )

    aliases = read_aliases(files["wikidata5m_relation.txt"], "P")
    recomputed = build_relation_schema(
        files["wikidata5m_transductive_train.txt"],
        aliases,
    )
    if (
        recomputed.sha256() != supplied_schema.sha256()
        or recomputed.canonical_bytes() != supplied_schema.canonical_bytes()
        or recomputed.codec_catalog != supplied_schema.codec_catalog
        or recomputed.path_relation_ids != supplied_schema.path_relation_ids
    ):
        raise ValueError(
            "supplied relation schema does not match recomputed relation schema"
        )

    actual_archive_hashes = {
        path.name: _sha256_file(path) for path in verified_archives
    }
    expected_archive_hashes = {
        name: item.sha256 for name, item in lock.files.items()
    }
    if actual_archive_hashes != expected_archive_hashes:
        raise AssertionError("verified archive hashes drifted after verification")
    return VerifiedWikidataSource(
        split_path=selected_path,
        source_sha256=selected_hash,
        source_lock_sha256=wikidata_lock_sha256(lock),
        source_archive_sha256=actual_archive_hashes,
        recomputed_schema=recomputed,
    )
