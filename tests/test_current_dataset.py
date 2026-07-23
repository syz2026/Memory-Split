from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from corpusgen.current_dataset import (
    CurrentBuildConfig,
    _SourceSpool,
    _puzzle_variants,
    _token_chunks,
    build_current_dataset,
    build_fixture_current_dataset,
    fixture_current_sources,
    verify_current_dataset,
)
from train.tokenizer import get_tok


EXPECTED = {
    "fineweb": 0.40,
    "wikidata_graph": 0.20,
    "synthetic_graph": 0.10,
    "synthetic_reasoning": 0.15,
    "wikidata_reasoning": 0.075,
    "relational_refinement": 0.025,
    "puzzle_auxiliary": 0.05,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_smoke_build_has_current_lanes_and_aligned_sidecars(tmp_path):
    report = build_fixture_current_dataset(tmp_path, total_tokens=131_072)

    assert report["target_shares"] == EXPECTED
    assert report["profile"] == "smoke"
    assert report["scientific_result"] is False
    assert report["tokens"]["total"] == 131_072
    assert set(report["tokens"]["lanes"]) == set(EXPECTED)
    assert all(
        lane["tokens"] > 0 and lane["records"] > 0
        for lane in report["tokens"]["lanes"].values()
    )

    size = (tmp_path / "train.bin").stat().st_size // 2
    assert size == 131_072
    for condition in ("dense", "split", "random"):
        assert (tmp_path / f"{condition}.weights.bin").stat().st_size == size

    dense = np.fromfile(tmp_path / "dense.weights.bin", dtype=np.uint8)
    split = np.fromfile(tmp_path / "split.weights.bin", dtype=np.uint8)
    random = np.fromfile(tmp_path / "random.weights.bin", dtype=np.uint8)
    assert dense.tolist() == [1] * size
    assert int((split == 0).sum()) > 0
    assert int((split == 0).sum()) == int((random == 0).sum())


def test_smoke_manifest_is_relative_hash_complete_and_verifiable(tmp_path):
    built = build_fixture_current_dataset(tmp_path)
    verified = verify_current_dataset(tmp_path, "smoke")

    assert verified == built
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["profile"] == "smoke"
    assert manifest["scientific_result"] is False
    manifested = {item["path"] for item in manifest["artifacts"]}
    actual = {
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert manifested == actual
    for artifact in manifest["artifacts"]:
        relative = Path(artifact["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = tmp_path / relative
        assert artifact["bytes"] == path.stat().st_size
        assert artifact["sha256"] == _sha256(path)


def test_smoke_build_is_byte_deterministic_and_idempotent(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_report = build_fixture_current_dataset(first)
    second_report = build_fixture_current_dataset(second)

    assert first_report == second_report
    assert _tree_bytes(first) == _tree_bytes(second)
    assert build_fixture_current_dataset(first) == first_report


def test_existing_conflicting_output_is_rejected(tmp_path):
    build_fixture_current_dataset(tmp_path, total_tokens=131_072)
    config = CurrentBuildConfig(
        profile="smoke",
        scale="29m",
        fact_load="n50k",
        data_seed=9,
        total_tokens=65_536,
    )

    with pytest.raises(ValueError, match="conflicting current dataset output"):
        build_current_dataset(config, fixture_current_sources(), tmp_path)


def test_factual_ledger_and_random_masks_use_exact_locked_strata(tmp_path):
    build_fixture_current_dataset(tmp_path)
    factual = [
        json.loads(line)
        for line in (tmp_path / "factual-span-ledger.jsonl").read_text().splitlines()
    ]
    masks = [
        json.loads(line)
        for line in (tmp_path / "mask-ledger.jsonl").read_text().splitlines()
    ]

    assert factual
    assert all(
        {
            "source",
            "source_row",
            "fact_id",
            "route",
            "record_type",
            "payload_token_length",
            "position_bin",
            "start",
            "end",
        }
        <= set(row)
        for row in factual
    )
    split = Counter(
        (
            row["source"],
            row["record_type"],
            row["payload_token_length"],
            row["position_bin"],
        )
        for row in masks
        if row["condition"] == "split"
    )
    random = Counter(
        (
            row["source"],
            row["record_type"],
            row["payload_token_length"],
            row["position_bin"],
        )
        for row in masks
        if row["condition"] == "random"
    )
    assert split
    assert split == random
    assert all(row["role"] == "random_control" for row in masks if row["condition"] == "random")


def test_smoke_graph_is_paged_set_valued_and_tracks_complete_once(tmp_path):
    report = build_fixture_current_dataset(tmp_path)
    rows = [
        json.loads(line)
        for line in (tmp_path / "graph.jsonl").read_text().splitlines()
    ]
    wikidata = [row for row in rows if row["provenance_id"].startswith("wikidata:")]

    assert wikidata
    assert any(len(row["targets"]) > 1 for row in wikidata)
    assert all(row["page"] >= 0 for row in wikidata)
    assert all(
        row["targets"]
        == sorted(set(row["targets"]), key=lambda value: int(value[1:]))
        for row in wikidata
    )
    assert report["coverage"]["distinct_training_triples"] > 0
    assert report["coverage"]["emitted_before_replay"] == report["coverage"][
        "distinct_training_triples"
    ]
    assert report["coverage"]["complete_once"] is True


def test_config_rejects_unknown_contract_values():
    with pytest.raises(ValueError, match="profile"):
        CurrentBuildConfig("preview", "29m", "n50k", 0, 1)
    with pytest.raises(ValueError, match="scale"):
        CurrentBuildConfig("smoke", "tiny", "n50k", 0, 1)
    with pytest.raises(ValueError, match="fact_load"):
        CurrentBuildConfig("smoke", "29m", "all", 0, 1)


def test_puzzle_variants_include_hash_ordered_color_and_translation_transforms():
    task = {
        "train": [
            {
                "input": [[0, 1, 0], [0, 2, 0], [0, 0, 0]],
                "output": [[0, 0, 0], [0, 1, 0], [0, 2, 0]],
            }
        ],
        "test": [
            {
                "input": [[0, 3, 0], [0, 4, 0], [0, 0, 0]],
                "output": [[0, 0, 0], [0, 3, 0], [0, 4, 0]],
            }
        ],
    }

    variants = _puzzle_variants(task)

    assert 9 < len(variants) <= 64
    assert variants[0][1] == task
    assert [parameter_hash for parameter_hash, _ in variants[1:]] == sorted(
        parameter_hash for parameter_hash, _ in variants[1:]
    )
    assert len(
        {
            hashlib.sha256(
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            for _, value in variants
        }
    ) == len(variants)


def test_long_source_rows_are_split_without_dropping_tokens():
    tok = get_tok()
    text = " relational" * 3_000

    chunks = list(_token_chunks(tok, text))

    assert len(chunks) > 1
    assert all(0 < len(tok.encode(chunk)) <= 768 for chunk in chunks)
    assert [
        token_id
        for chunk in chunks
        for token_id in tok.encode(chunk)
    ] == tok.encode(text)


def test_full_profile_refuses_unverified_smoke_sources(tmp_path):
    with pytest.raises(ValueError, match="verified Task 1 source root"):
        build_current_dataset(
            CurrentBuildConfig(
                profile="full",
                scale="160m",
                fact_load="n50k",
                data_seed=1,
                total_tokens=3_244_818_432,
            ),
            fixture_current_sources(),
            tmp_path,
        )


def test_29m_wikidata_sample_is_hash_stable_and_balanced(tmp_path):
    selections = []
    for name in ("one.sqlite3", "two.sqlite3"):
        spool = _SourceSpool(tmp_path / name, fixture_current_sources())
        spool.select_balanced_training_sample(5)
        rows = spool.connection.execute(
            """
            SELECT split, relation, subject, object
            FROM triples
            ORDER BY split, relation, subject, object
            """
        ).fetchall()
        strata = Counter((split, relation) for split, relation, _, _ in rows)
        selections.append(rows)
        spool.close()

        assert set(strata.values()) == {1}

    assert selections[0] == selections[1]
