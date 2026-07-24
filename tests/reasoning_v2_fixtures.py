from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.reasoning_v2 import load_recipe, source_lock as source_lock_module
from corpusgen.reasoning_v2.source_lock import (
    FIXED_REQUESTS,
    PUBLIC_REQUESTS,
    SourceEntry,
    SourceFile,
    SourceLock,
    SourceRequest,
    resolve_source_lock,
)


ROOT = Path(__file__).resolve().parents[1]

_FIXED_REVISIONS = {
    "fineweb_edu": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    "wikidata5m": "6b2b09672129e280c0c9da97ab58154e9d535e6b",
    "arc_agi_1": "399030444e0ab0cc8b4e199870fb20b863846f34",
    "arc_agi_2": "f3283f727488ad98fe575ea6a5ac981e4a188e49",
    "conceptarc": "0e67da6af879e4bad3d7cd3c196e8d551b445725",
}
_LICENSES = {
    "fineweb_edu": ("ODC-By-1.0", "README.md", b"license: odc-by\n"),
    "finemath": ("ODC-By-1.0", "README.md", b"license: odc-by\n"),
    "wikidata5m": (
        "CC0-1.0",
        "Wikidata-CC0-1.0.txt",
        b"Creative Commons CC0 1.0\n",
    ),
    "conceptarc": ("MIT", "LICENSE", b"MIT License\n"),
}
_APACHE = b"Apache License\nVersion 2.0, January 2004\n"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_files(root: Path, files: dict[str, bytes]) -> tuple[SourceFile, ...]:
    rows = []
    for relative, payload in sorted(
        files.items(),
        key=lambda item: item[0].encode("utf-8"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        rows.append(
            SourceFile(
                path=relative,
                bytes=len(payload),
                sha256=_sha256(payload),
            )
        )
    return tuple(rows)


class FakePublicResolver:
    def __init__(
        self,
        *,
        revision_overrides: dict[str, str] | None = None,
        omit_ruletaker_archive: bool = False,
        unresolved_source: str | None = None,
    ) -> None:
        self.revision_overrides = revision_overrides or {}
        self.omit_ruletaker_archive = omit_ruletaker_archive
        self.unresolved_source = unresolved_source

    def resolve(self, request: SourceRequest, download_root: Path) -> SourceEntry:
        if request.source_id == self.unresolved_source:
            raise RuntimeError("upstream unavailable")
        source_id = request.source_id
        license_spdx, license_path, license_bytes = _LICENSES.get(
            source_id,
            ("Apache-2.0", "LICENSE", _APACHE),
        )
        files = {
            license_path: license_bytes,
        }
        if license_path != "README.md":
            files["README.md"] = f"# {source_id}\n".encode()
        if source_id == "fineweb_edu":
            files.update(
                {
                    "sample/10BT/000_00000.parquet": b"fineweb zero",
                    "sample/10BT/001_00000.parquet": b"fineweb one",
                    "sample/10BT/002_00000.parquet": b"fineweb two",
                }
            )
        elif source_id == "finemath":
            files["finemath-4plus/train-00000-of-00064.parquet"] = b"math"
        elif source_id == "wikidata5m":
            files.update(
                {
                    "wikidata5m_alias.tar.gz": b"aliases",
                    "wikidata5m_inductive.tar.gz": b"inductive",
                    "wikidata5m_transductive.tar.gz": b"transductive",
                }
            )
        elif source_id == "ruletaker" and not self.omit_ruletaker_archive:
            files["rule-reasoning-dataset-V2020.2.5.zip"] = b"dataset archive"
        elif source_id == "arc_agi_1":
            files["data/training/a.json"] = b"{}\n"
        elif source_id == "arc_agi_2":
            files["data/training/b.json"] = b"{}\n"
        elif source_id == "conceptarc":
            files["corpus/concept/a.json"] = b"{}\n"
        else:
            files["src/data.txt"] = source_id.encode()

        materialized = download_root / source_id
        rows = _source_files(materialized, files)
        revision = self.revision_overrides.get(
            source_id,
            _FIXED_REVISIONS.get(
                source_id,
                hashlib.sha256(source_id.encode()).hexdigest()[:40],
            ),
        )
        finemath_selection = None
        if source_id == "finemath":
            four_plus_paths = tuple(
                f"finemath-4plus/train-{index:05d}-of-00064.parquet"
                for index in range(64)
            )
            three_plus_paths = tuple(
                f"finemath-3plus/train-{index:05d}-of-00128.parquet"
                for index in range(128)
            )
            finemath_selection = source_lock_module.FineMathSelectionProof(
                algorithm="nfc-fineweb-exact-dedup-gpt2-eot-v1",
                quota=source_lock_module._FINEMATH_TARGETS,
                four_plus_paths=four_plus_paths,
                three_plus_paths=three_plus_paths,
                selected_paths=(four_plus_paths[0],),
                usable_targets=source_lock_module._FINEMATH_TARGETS + 1,
                fineweb_duplicate_rows=0,
            )
        return SourceEntry(
            source_id=source_id,
            transport=request.transport,
            repository=request.repository,
            revision_kind="git_commit",
            revision=revision,
            license_spdx=license_spdx,
            license_files=(license_path,),
            materialized_path=source_id,
            files=rows,
            finemath_selection=finemath_selection,
        )


@dataclass(frozen=True)
class FixtureSourceLock:
    lock: SourceLock
    download_root: Path


@pytest.fixture
def full_recipe():
    return load_recipe(ROOT / "configs/reasoning-dataset-v2.json")


@pytest.fixture
def fixed_contract_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    contract_root = tmp_path / "contracts"
    contract_root.mkdir()
    current = json.loads(
        (ROOT / "configs/current-dataset-lock.json").read_text(encoding="utf-8")
    )
    current_path = contract_root / "current-dataset-lock.json"
    current_path.write_bytes(canonical_json_bytes(current))

    wikidata_files = {
        "wikidata5m_alias.tar.gz": {
            "bytes": len(b"aliases"),
            "sha256": _sha256(b"aliases"),
        },
        "wikidata5m_inductive.tar.gz": {
            "bytes": len(b"inductive"),
            "sha256": _sha256(b"inductive"),
        },
        "wikidata5m_transductive.tar.gz": {
            "bytes": len(b"transductive"),
            "sha256": _sha256(b"transductive"),
        },
    }
    fineweb_files = {
        path: {
            "bytes": len(payload),
            "sha256": _sha256(payload),
        }
        for path, payload in {
            "sample/10BT/000_00000.parquet": b"fineweb zero",
            "sample/10BT/001_00000.parquet": b"fineweb one",
            "sample/10BT/002_00000.parquet": b"fineweb two",
        }.items()
    }
    wikidata_path = contract_root / "wikidata5m.lock.json"
    wikidata_path.write_bytes(
        canonical_json_bytes(
            {
                "files": wikidata_files,
                "repo_id": "intfloat/wikidata5m",
                "repo_type": "dataset",
                "revision": _FIXED_REVISIONS["wikidata5m"],
            }
        )
    )
    licenses_path = ROOT / "sources/current-dataset-licenses.json"
    notice_path = ROOT / "sources/Wikidata-CC0-1.0.txt"
    monkeypatch.setattr(source_lock_module, "CURRENT_DATASET_LOCK_PATH", current_path)
    monkeypatch.setattr(source_lock_module, "WIKIDATA_LOCK_PATH", wikidata_path)
    monkeypatch.setattr(source_lock_module, "CURRENT_LICENSES_PATH", licenses_path)
    monkeypatch.setattr(source_lock_module, "WIKIDATA_NOTICE_PATH", notice_path)
    monkeypatch.setattr(
        source_lock_module,
        "FIXED_WIKIDATA_FILES",
        copy.deepcopy(wikidata_files),
    )
    monkeypatch.setattr(
        source_lock_module,
        "FIXED_FINEWEB_FILES",
        copy.deepcopy(fineweb_files),
    )
    monkeypatch.setattr(source_lock_module, "_FINEMATH_TARGETS", 4)

    def iter_fixture_texts(descriptor, _description):
        metadata = os.fstat(descriptor)
        payload = os.pread(descriptor, metadata.st_size, 0)
        return iter((payload.decode("utf-8"),))

    monkeypatch.setattr(
        source_lock_module,
        "_iter_parquet_texts_from_descriptor",
        iter_fixture_texts,
        raising=False,
    )
    monkeypatch.setattr(
        source_lock_module,
        "_encode_finemath_text",
        lambda text: list(text),
        raising=False,
    )
    return contract_root


@pytest.fixture
def fake_public_resolver(fixed_contract_environment):
    return FakePublicResolver()


@pytest.fixture
def fixture_source_lock(
    tmp_path: Path,
    fake_public_resolver: FakePublicResolver,
    full_recipe,
) -> FixtureSourceLock:
    download_root = tmp_path / "downloads"
    lock = resolve_source_lock(
        full_recipe,
        fake_public_resolver,
        download_root,
        generator_commit="a" * 40,
    )
    return FixtureSourceLock(lock=lock, download_root=download_root)


@pytest.fixture
def valid_lock_json(fixture_source_lock: FixtureSourceLock) -> dict[str, object]:
    return copy.deepcopy(fixture_source_lock.lock.as_dict())


ALL_REQUESTS = FIXED_REQUESTS + PUBLIC_REQUESTS
