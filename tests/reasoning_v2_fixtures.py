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


def _compact_task_bytes(task: dict[str, object]) -> bytes:
    return (
        json.dumps(task, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )


def _reformatted_task_bytes(task: dict[str, object]) -> bytes:
    return json.dumps(task, ensure_ascii=False, indent=2, sort_keys=True).encode(
        "utf-8"
    ) + b"\n"


# Distinct exact-answer ARC training tasks. Their canonical hashes differ; the
# raw and reformatted variants of task one drive the raw/canonical duplicate
# audit exercised by the finite puzzle scan.
_PUZZLE_TASK_ONE = {
    "train": [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}],
    "test": [{"input": [[2, 2], [2, 2]], "output": [[3, 3], [3, 3]]}],
}
_PUZZLE_TASK_TWO = {
    "train": [{"input": [[4]], "output": [[5]]}],
    "test": [
        {"input": [[6]], "output": [[7]]},
        {"input": [[8]], "output": [[9]]},
    ],
}
_PUZZLE_TASK_THREE = {
    "train": [{"input": [[1, 1]], "output": [[2, 2]]}],
    "test": [{"input": [[3, 3]], "output": [[4, 4]]}],
}
# Evaluation tasks withhold their test outputs (as the real ARC evaluation
# split does); the scan only needs their canonical hashes for overlap checks.
_PUZZLE_EVAL_ONE = {
    "train": [{"input": [[7]], "output": [[7]]}],
    "test": [{"input": [[8]]}],
}
_PUZZLE_EVAL_TWO = {
    "train": [{"input": [[5, 5]], "output": [[6, 6]]}],
    "test": [{"input": [[9, 9]]}],
}

_PUZZLE_TASK_ONE_BYTES = _compact_task_bytes(_PUZZLE_TASK_ONE)
_PUZZLE_TASK_ONE_REFORMATTED_BYTES = _reformatted_task_bytes(_PUZZLE_TASK_ONE)

# Direct, listed, byte-bound finite puzzle layout for the v2 fixture: one
# accepted training task per source, an exact-byte cross-source duplicate, a
# canonical (reformatted) cross-source duplicate, and ARC evaluation files.
DEFAULT_PUZZLE_FILES: dict[str, dict[str, bytes]] = {
    "arc_agi_1": {
        "data/training/a.json": _PUZZLE_TASK_ONE_BYTES,
        "data/evaluation/e1.json": _compact_task_bytes(_PUZZLE_EVAL_ONE),
    },
    "arc_agi_2": {
        "data/training/a_rawdup.json": _PUZZLE_TASK_ONE_BYTES,
        "data/training/b.json": _compact_task_bytes(_PUZZLE_TASK_TWO),
        "data/evaluation/e2.json": _compact_task_bytes(_PUZZLE_EVAL_TWO),
    },
    "conceptarc": {
        "corpus/concept/a.json": _PUZZLE_TASK_ONE_REFORMATTED_BYTES,
        "corpus/concept/c.json": _compact_task_bytes(_PUZZLE_TASK_THREE),
    },
}


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
        puzzle_files: dict[str, dict[str, bytes]] | None = None,
    ) -> None:
        self.revision_overrides = revision_overrides or {}
        self.omit_ruletaker_archive = omit_ruletaker_archive
        self.unresolved_source = unresolved_source
        self.puzzle_files = puzzle_files

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
        elif source_id in DEFAULT_PUZZLE_FILES:
            override = (
                None if self.puzzle_files is None else self.puzzle_files.get(source_id)
            )
            files.update(
                DEFAULT_PUZZLE_FILES[source_id] if override is None else override
            )
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
