from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from fractions import Fraction
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
    build_reasoning_v2_smoke_fixture,
    fixture_current_sources,
    verify_current_dataset,
    verify_reasoning_v2_smoke_fixture,
)
from corpusgen.reasoning.routing import build_route_manifests
from corpusgen.reasoning.smoke import (
    _encode_fields,
    _fixture_facts,
    _fixture_fields,
    _fixture_state_bundles,
    _mask_for_manifest,
    _route_dose,
    _semantic_facts,
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


def _refresh_manifest_artifact(root: Path, relative: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifact = next(
        item for item in manifest["artifacts"] if item["path"] == relative
    )
    path = root / relative
    artifact["bytes"] = path.stat().st_size
    artifact["sha256"] = _sha256(path)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )


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


def test_v2_smoke_proves_route_dose_and_zero_supervised_semantic_copies(
    tmp_path,
):
    built = build_reasoning_v2_smoke_fixture(tmp_path)
    verified = verify_reasoning_v2_smoke_fixture(tmp_path)

    assert verified == built
    assert built["profile"] == "smoke"
    assert built["scientific_result"] is False
    assert built["scientific_readiness"] is False
    assert built["route_dose"]["Split50"]["external_facts"] == 5
    assert built["route_dose"]["Split90"]["external_facts"] == 9
    assert all(
        dose["external_facts"] == dose["quota_facts"]
        and dose["information_burden_quota_met"]
        for dose in built["route_dose"].values()
    )
    assert all(
        closure["passed"]
        and closure["unmasked_supervised_occurrences"] == 0
        for closure in built["semantic_closure"].values()
    )
    assert built["answer_states"]["surface_value_copies"] == 0
    assert built["answer_states"]["phases"] == ["candidate", "final"]
    assert built["proofs"] == {
        "families": ["graph_composition_mod4", "slot_equality"],
        "verified": True,
    }

    token_count = built["tokens"]
    dense = np.fromfile(tmp_path / "dense.weights.bin", dtype=np.uint8)
    assert dense.tolist() == [1] * token_count
    for split in ("Split50", "Split90"):
        weights = np.fromfile(
            tmp_path / f"{split.lower()}.weights.bin",
            dtype=np.uint8,
        )
        assert len(weights) == token_count
        assert int((weights == 0).sum()) > 0


def test_v2_verifier_rejects_rehashed_overmasked_sidecar(tmp_path):
    build_reasoning_v2_smoke_fixture(tmp_path)
    sidecar = tmp_path / "split50.weights.bin"
    sidecar.write_bytes(bytes(sidecar.stat().st_size))
    _refresh_manifest_artifact(tmp_path, sidecar.name)

    with pytest.raises(ValueError, match="Split50 sidecar"):
        verify_reasoning_v2_smoke_fixture(tmp_path)


def test_v2_verifier_rejects_rehashed_false_route_claims(tmp_path):
    build_reasoning_v2_smoke_fixture(tmp_path)
    route_path = tmp_path / "split90-route-manifest.json"
    route = json.loads(route_path.read_text())
    route["total_facts"] = 999
    route["total_information_burden_bits"] = {
        "numerator": 0,
        "denominator": 1,
    }
    route["external_information_burden_bits"] = {
        "numerator": 0,
        "denominator": 1,
    }
    route["information_burden_fraction"] = {
        "numerator": 0,
        "denominator": 1,
    }
    route_path.write_text(
        json.dumps(route, sort_keys=True, separators=(",", ":")) + "\n"
    )
    _refresh_manifest_artifact(tmp_path, route_path.name)

    with pytest.raises(ValueError, match="Split90 route"):
        verify_reasoning_v2_smoke_fixture(tmp_path)


def test_v2_verifier_binds_routes_to_fixed_fixture_metadata(tmp_path):
    build_reasoning_v2_smoke_fixture(tmp_path)
    fixed_facts = _fixture_facts()
    tampered_facts = tuple(
        replace(
            fact,
            payload_entropy_bits=(
                1
                if fact.fact_id == "fact-00"
                else 1_000
                if fact.fact_id == "fact-09"
                else fact.payload_entropy_bits
            ),
        )
        for fact in fixed_facts
    )
    manifests = build_route_manifests(tampered_facts)
    fields = _fixture_fields(_fixture_state_bundles())
    train = np.fromfile(tmp_path / "train.bin", dtype=np.uint16)
    _, slices = _encode_fields(_semantic_facts(tampered_facts), fields)
    report_path = tmp_path / "report.json"
    report = json.loads(report_path.read_text())

    for split, manifest in manifests.items():
        stem = split.lower()
        route_path = tmp_path / f"{stem}-route-manifest.json"
        sidecar_path = tmp_path / f"{stem}.weights.bin"
        route_path.write_bytes(manifest.to_bytes())
        sidecar_path.write_bytes(_mask_for_manifest(len(train), slices, manifest))
        report["route_dose"][split] = _route_dose(manifest)
        _refresh_manifest_artifact(tmp_path, route_path.name)
        _refresh_manifest_artifact(tmp_path, sidecar_path.name)

    report_path.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    )
    _refresh_manifest_artifact(tmp_path, report_path.name)
    true_burdens = {
        fact.fact_id: fact.information_burden_bits
        for fact in fixed_facts
    }
    split90_external = manifests["Split90"].external_fact_ids
    true_fraction = sum(
        (true_burdens[fact_id] for fact_id in split90_external),
        Fraction(),
    ) / sum(true_burdens.values(), Fraction())
    assert true_fraction < Fraction(9, 10)

    with pytest.raises(ValueError, match="fixed fixture route"):
        verify_reasoning_v2_smoke_fixture(tmp_path)


def test_v2_verifier_binds_records_to_fixed_supervised_fields(tmp_path):
    build_reasoning_v2_smoke_fixture(tmp_path)
    facts = _fixture_facts()
    fields = tuple(
        replace(field, supervised=False)
        for field in _fixture_fields(_fixture_state_bundles())
    )
    records_path = tmp_path / "records.jsonl"
    _write_jsonl(
        records_path,
        [
            {
                "field_id": field.field_id,
                "text": field.text,
                "supervised": field.supervised,
            }
            for field in fields
        ],
    )
    token_ids, slices = _encode_fields(_semantic_facts(facts), fields)
    train_path = tmp_path / "train.bin"
    dense_path = tmp_path / "dense.weights.bin"
    token_ids.tofile(train_path)
    dense_path.write_bytes(bytes([1]) * len(token_ids))
    manifests = build_route_manifests(facts)
    for split, manifest in manifests.items():
        sidecar_path = tmp_path / f"{split.lower()}.weights.bin"
        sidecar_path.write_bytes(_mask_for_manifest(len(token_ids), slices, manifest))
        _refresh_manifest_artifact(tmp_path, sidecar_path.name)
    report_path = tmp_path / "report.json"
    report = json.loads(report_path.read_text())
    report["tokens"] = len(token_ids)
    report_path.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    )
    for path in (records_path, train_path, dense_path, report_path):
        _refresh_manifest_artifact(tmp_path, path.name)

    with pytest.raises(ValueError, match="fixed supervised records"):
        verify_reasoning_v2_smoke_fixture(tmp_path)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (
            "leaks",
            [
                {
                    "field_id": "record-00",
                    "start": 0,
                    "end": 1,
                    "fact_id": "fact-00",
                    "surface": "forged",
                }
            ],
        ),
        ("metadata_errors", ["forged metadata error"]),
        ("routed_fact_ids", []),
        ("supervised_occurrences", -1),
        ("masked_occurrences", -1),
    ],
)
def test_v2_verifier_replays_complete_semantic_leakage_artifact(
    tmp_path,
    field,
    replacement,
):
    build_reasoning_v2_smoke_fixture(tmp_path)
    leakage_path = tmp_path / "split50-semantic-leakage.json"
    leakage = json.loads(leakage_path.read_text())
    leakage[field] = replacement
    leakage_path.write_text(
        json.dumps(leakage, sort_keys=True, separators=(",", ":")) + "\n"
    )
    _refresh_manifest_artifact(tmp_path, leakage_path.name)

    with pytest.raises(ValueError, match="semantic leakage artifact"):
        verify_reasoning_v2_smoke_fixture(tmp_path)


def test_v2_verifier_derives_complete_report_from_replayed_evidence(tmp_path):
    build_reasoning_v2_smoke_fixture(tmp_path)
    report_path = tmp_path / "report.json"
    report = json.loads(report_path.read_text())
    report["facts"] = 999
    report["semantic_closure"]["Split50"]["supervised_occurrences"] = -1
    report["semantic_closure"]["Split50"]["masked_occurrences"] = -1
    report_path.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    )
    _refresh_manifest_artifact(tmp_path, report_path.name)

    with pytest.raises(ValueError, match="complete report"):
        verify_reasoning_v2_smoke_fixture(tmp_path)


def test_v2_smoke_persists_typed_proof_and_answer_state_inputs(tmp_path):
    build_reasoning_v2_smoke_fixture(tmp_path)

    proofs = _read_jsonl(tmp_path / "proofs.jsonl")
    states = _read_jsonl(tmp_path / "answer-states.jsonl")

    assert all(set(bundle) == {"family", "premises", "proof"} for bundle in proofs)
    assert {bundle["family"] for bundle in proofs} == {
        "graph_composition_mod4",
        "slot_equality",
    }
    assert all(
        set(row) == {"phase", "pointer", "state"}
        and set(row["pointer"]) == {"member_index", "read_index", "slot"}
        for row in states
    )


def test_v2_verifier_rejects_rehashed_proof_change(tmp_path):
    build_reasoning_v2_smoke_fixture(tmp_path)
    proof_path = tmp_path / "proofs.jsonl"
    bundles = _read_jsonl(proof_path)
    published_proof = bundles[0].get("proof", bundles[0])
    published_proof["conclusion"]["relation"] = "r0"
    _write_jsonl(proof_path, bundles)
    _refresh_manifest_artifact(tmp_path, proof_path.name)

    with pytest.raises(ValueError, match="proof bundle"):
        verify_reasoning_v2_smoke_fixture(tmp_path)


def test_v2_verifier_rejects_rehashed_answer_state_change(tmp_path):
    build_reasoning_v2_smoke_fixture(tmp_path)
    state_path = tmp_path / "answer-states.jsonl"
    states = _read_jsonl(state_path)
    states[0]["state"] += " value-00-cerulean"
    _write_jsonl(state_path, states)
    _refresh_manifest_artifact(tmp_path, state_path.name)

    with pytest.raises(ValueError, match="answer-state"):
        verify_reasoning_v2_smoke_fixture(tmp_path)


def test_v2_verifier_rejects_broken_symlink_anywhere_in_artifact_tree(
    tmp_path,
):
    build_reasoning_v2_smoke_fixture(tmp_path)
    (tmp_path / "broken-link").symlink_to("missing-target")

    with pytest.raises(ValueError, match="non-regular"):
        verify_reasoning_v2_smoke_fixture(tmp_path)
