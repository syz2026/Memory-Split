from __future__ import annotations

import hashlib
import heapq
import json
import random
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import torch

from corpusgen import relational_build as relational
from corpusgen.bed_snapshot import SourceDriftError, lock_bed_snapshot
from corpusgen.mask_ledger import LeakageError, RandomMaskUndersupplyError
from corpusgen.publication_audit import (
    verify_published_artifacts,
    write_published_artifact_audit,
)
from corpusgen.relational_build import (
    EncodedSpan,
    FactCost,
    RelationalBuildConfig,
    RoutePolicy,
    _encode_record,
    build_relational_corpus,
    calibrate_write_cost,
    derive_weights,
)
from corpusgen.graph_records import (
    GraphRow,
    RenderedRecord,
    ScheduleEntry,
    TaggedSegment,
    stable_fact_id,
)
from corpusgen.relation_schema import (
    LITERAL_RELATIONS,
    RelationSchema,
    RelationSpec,
)
from corpusgen.world_splits import (
    SplitArtifactExpectations,
    build_split_plan,
    composition_hash,
)
from organizer.graph_store import AtomicGraphStore
from organizer.packed_graph_store import PackedGraphStore
from train.tokenizer import get_tok


def test_factual_reservoir_heap_never_compares_payload_objects():
    candidate_type = getattr(relational, "_FactualCandidate", None)
    assert candidate_type is not None

    first = candidate_type(
        priority=-7,
        fact_id="same-fact",
        ordinal=3,
        world=object(),
        fact=object(),
        route="external",
    )
    second = candidate_type(
        priority=-7,
        fact_id="same-fact",
        ordinal=3,
        world=object(),
        fact=object(),
        route="external",
    )
    heap = [first]

    heapq.heappush(heap, second)

    assert len(heap) == 2


def test_write_cost_is_selected_without_semantic_labels():
    facts = [
        FactCost("a", entropy=12, exposures=1, expected_reads=1, expected_hops=1),
        FactCost("b", entropy=8, exposures=1, expected_reads=1, expected_hops=1),
        FactCost(
            "c",
            entropy=1,
            exposures=16,
            expected_reads=20,
            expected_hops=2,
        ),
        FactCost(
            "d",
            entropy=1,
            exposures=16,
            expected_reads=20,
            expected_hops=2,
        ),
    ]

    policy = calibrate_write_cost(facts)

    assert 0.40 <= policy.route_rate(facts) <= 0.60
    assert policy.is_external(facts[0])
    assert not policy.is_external(facts[-1])


def test_split_and_random_weights_mask_only_their_allowed_spans():
    internal = FactCost(
        "internal",
        entropy=1,
        exposures=32,
        expected_reads=20,
        expected_hops=4,
    )
    external = FactCost(
        "external",
        entropy=12,
        exposures=1,
        expected_reads=0,
        expected_hops=0,
    )
    spans = [
        EncodedSpan(0, 3, "action"),
        EncodedSpan(3, 6, "payload", "internal", internal),
        EncodedSpan(6, 9, "random_control"),
        EncodedSpan(9, 10, "action"),
        EncodedSpan(10, 13, "payload", "external", external),
        EncodedSpan(13, 14, "relation_alias"),
        EncodedSpan(14, 17, "random_control"),
        EncodedSpan(17, 25, "rule"),
        EncodedSpan(25, 35, "provisional_answer"),
        EncodedSpan(35, 50, "action"),
        EncodedSpan(50, 53, "plain"),
        EncodedSpan(53, 100, "final_answer"),
    ]
    policy = RoutePolicy(write_cost=1)

    split = derive_weights("split", spans, policy, random.Random(7))
    selective = derive_weights("selective", spans, policy, random.Random(7))
    random_control = derive_weights("random", spans, policy, random.Random(7))

    assert np.flatnonzero(split == 0).tolist() == [3, 4, 5, 10, 11, 12]
    assert np.flatnonzero(selective == 0).tolist() == [10, 11, 12]
    assert np.flatnonzero(random_control == 0).tolist() == [
        6,
        7,
        8,
        14,
        15,
        16,
    ]
    assert random_control[50:53].tolist() == [1, 1, 1]


def test_random_matching_samples_deterministically_within_exact_key_pool():
    external = FactCost(
        "external",
        entropy=12,
        exposures=1,
        expected_reads=0,
        expected_hops=0,
    )
    spans = [
        EncodedSpan(0, 10, "action"),
        EncodedSpan(10, 13, "payload", "external", external),
        EncodedSpan(13, 15, "action"),
        EncodedSpan(15, 18, "random_control"),
        EncodedSpan(18, 21, "random_control"),
        EncodedSpan(21, 100, "action"),
    ]
    policy = RoutePolicy(write_cost=1)

    first = derive_weights("random", spans, policy, random.Random(23))
    second = derive_weights("random", spans, policy, random.Random(23))

    assert np.array_equal(first, second)
    assert np.flatnonzero(first == 0).tolist() in (
        [15, 16, 17],
        [18, 19, 20],
    )


def test_expected_external_ranges_are_collected_before_weight_derivation():
    external = FactCost(
        "external",
        entropy=12,
        exposures=1,
        expected_reads=0,
        expected_hops=0,
    )
    spans = [
        EncodedSpan(0, 5, "action"),
        EncodedSpan(5, 8, "payload", "external", external),
        EncodedSpan(8, 12, "plain"),
    ]

    expected = relational.collect_expected_external_ranges(
        spans,
        RoutePolicy(write_cost=1),
    )

    assert [(item.start, item.end, item.fact_id) for item in expected] == [
        (5, 8, "external")
    ]


def test_split_coverage_validation_rejects_missing_or_extra_zeros():
    external = FactCost(
        "external",
        entropy=12,
        exposures=1,
        expected_reads=0,
        expected_hops=0,
    )
    spans = [
        EncodedSpan(0, 5, "action"),
        EncodedSpan(5, 8, "payload", "external", external),
        EncodedSpan(8, 12, "plain"),
    ]
    expected = relational.collect_expected_external_ranges(
        spans,
        RoutePolicy(write_cost=1),
    )

    with pytest.raises(ValueError, match="expected external payload"):
        relational.validate_split_coverage(
            expected,
            spans,
            np.ones(12, dtype=np.uint8),
            [],
        )

    weights = np.ones(12, dtype=np.uint8)
    weights[5:8] = 0
    weights[9] = 0
    actual = [(5, 8, spans[1])]
    with pytest.raises(ValueError, match="protected nonpayload"):
        relational.validate_split_coverage(expected, spans, weights, actual)


def test_record_encoding_rejects_token_ids_outside_uint16():
    class OversizedTokenizer:
        EOT = 1

        @staticmethod
        def encode_tagged_segments(segments):
            del segments
            return [70_000], ["plain"], [None]

    record = RenderedRecord(
        segments=(TaggedSegment("ignored", "plain"),),
        schedule=ScheduleEntry("bed", "bed-0", 0, 0),
    )

    with pytest.raises(ValueError, match="token id does not fit uint16"):
        _encode_record(OversizedTokenizer(), record, {})


def test_eval_pair_count_must_be_positive():
    with pytest.raises(
        ValueError,
        match="eval_pairs_per_task must be positive",
    ):
        RelationalBuildConfig(
            n_entities=64,
            total_tokens=60_000,
            data_seed=17,
            eval_pairs_per_task=0,
        )


def test_default_mixture_and_ordered_development_fallbacks_are_frozen():
    base = {
        "n_entities": 64,
        "total_tokens": 60_000,
        "data_seed": 17,
    }
    assert RelationalBuildConfig(**base).component_shares == {
        "bed": 0.70,
        "graph": 0.15,
        "reasoning": 0.15,
    }
    assert RelationalBuildConfig(
        **base,
        artifact_mode="development",
        development_mixture_index=1,
    ).component_shares == {
        "bed": 0.65,
        "graph": 0.15,
        "reasoning": 0.20,
    }
    assert RelationalBuildConfig(
        **base,
        artifact_mode="development",
        development_mixture_index=2,
    ).component_shares == {
        "bed": 0.65,
        "graph": 0.20,
        "reasoning": 0.15,
    }
    with pytest.raises(ValueError, match="development|frozen study"):
        RelationalBuildConfig(
            **base,
            artifact_mode="fixture",
            development_mixture_index=1,
        )
    assert RelationalBuildConfig(
        **base,
        artifact_mode="protected",
        development_mixture_index=1,
    ).component_shares == {
        "bed": 0.65,
        "graph": 0.15,
        "reasoning": 0.20,
    }


def test_bed_scan_rejects_reserved_protected_handles_and_payload_prefixes():
    plan = build_split_plan(_fixture_schema(), seed=17)
    for protected in (plan.protected_seen, plan.protected_heldout):
        with pytest.raises(LeakageError, match="protected entity handle"):
            relational.scan_bed_text_for_reserved_values(
                f"Leaked Q{protected.entity_id_range[0]} handle.",
                plan,
            )
        with pytest.raises(LeakageError, match="protected payload prefix"):
            relational.scan_bed_text_for_reserved_values(
                f"Leaked {protected.payload_namespace}:SYN_L0 value.",
                plan,
            )

    relational.scan_bed_text_for_reserved_values(
        "A normal passage mentions Q42 and no synthetic values.",
        plan,
    )


def _bed_stream():
    passages = (
        "Glaciers carved the valley and left long ridges of gravel behind.",
        "Wind turbines convert moving air into electricity for the local grid.",
        "The old observatory records the path of each comet across the sky.",
        "Bees communicate the location of food through a sequence of movements.",
    )
    index = 0
    while True:
        yield f"{passages[index % len(passages)]} Passage {index}."
        index += 1


def _fixture_schema():
    entity_specs = tuple(
        RelationSpec(
            relation_id=f"P{index}",
            aliases=(f"relation {index}", f"property {index}"),
            target_kind="entity",
            support=84,
            distinct_subjects=80,
            distinct_objects=80 if index % 2 else 40,
            entity_count=100,
        )
        for index in range(1, 33)
    )
    literal_specs = tuple(
        RelationSpec(relation_id, aliases, target_kind)
        for relation_id, aliases, target_kind in LITERAL_RELATIONS
    )
    return RelationSchema(
        catalog=entity_specs + literal_specs,
        path_relation_ids=tuple(spec.relation_id for spec in entity_specs)
        + tuple(spec.relation_id for spec in literal_specs),
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("relational-corpus") / "published"
    cfg = RelationalBuildConfig(
        n_entities=64,
        total_tokens=60_000,
        data_seed=17,
        world_size=64,
        eval_pairs_per_task=10,
        guardrail_items=8,
        shared_text_eval_count=4,
    )
    schema = _fixture_schema()
    report = build_relational_corpus(
        cfg,
        get_tok(),
        _bed_stream(),
        out,
        relation_schema=schema,
    )
    return out, cfg, schema, report


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )


def _refresh_post_emission_expectation(root: Path) -> None:
    path = root / "published-artifact-expectations.json"
    expectation = json.loads(path.read_text())
    for relative in expectation["artifacts"]:
        artifact = root / relative
        expectation["artifacts"][relative] = {
            "bytes": artifact.stat().st_size,
            "sha256": _sha256(artifact),
        }
    for relative in expectation["jsonl"]:
        digest = hashlib.sha256()
        rows = 0
        for row in _read_jsonl(root / relative):
            digest.update(
                (
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode()
            )
            rows += 1
        expectation["jsonl"][relative] = {
            "rows": rows,
            "canonical_sha256": digest.hexdigest(),
        }
    path.write_text(
        json.dumps(
            expectation,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    expectation_sha256 = _sha256(path)
    audit_path = root / "published-artifact-audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        audit["expectation_sha256"] = expectation_sha256
        audit_path.write_text(
            json.dumps(
                audit,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["published_expectation_sha256"] = expectation_sha256
    if audit_path.exists():
        manifest["audit_sha256"]["published"] = _sha256(audit_path)
    manifest["artifacts"] = [
        {
            **artifact,
            "bytes": (root / artifact["path"]).stat().st_size,
            "sha256": _sha256(root / artifact["path"]),
        }
        for artifact in manifest["artifacts"]
    ]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def test_shared_stream_sidecars_are_aligned_and_checks_pass(built):
    out, _, _, report = built
    tokens = np.memmap(out / "train.bin", dtype=np.uint16, mode="r")
    dense = np.memmap(out / "dense.weights.bin", dtype=np.uint8, mode="r")
    split = np.memmap(out / "split.weights.bin", dtype=np.uint8, mode="r")
    random_control = np.memmap(
        out / "random.weights.bin",
        dtype=np.uint8,
        mode="r",
    )
    selective = np.memmap(
        out / "selective.weights.bin",
        dtype=np.uint8,
        mode="r",
    )

    assert (
        len(tokens)
        == len(dense)
        == len(split)
        == len(random_control)
        == len(selective)
    )
    assert len(tokens) >= 60_000
    assert (dense == 1).all()
    assert all(report["checks"].values())
    assert {
        "all_payload_ranges_exact",
        "selective_routes_exact",
        "random_mass_exact",
        "random_histogram_exact",
        "independent_mask_audit",
        "split_disjointness",
    } <= report["checks"].keys()
    assert not {
        "random_mass_within_1pct",
        "random_span_histogram_within_1pct",
        "random_position_histogram_within_1pct",
    } & report["checks"].keys()


def test_schedule_preserves_mixtures_curriculum_and_six_step_traces(built):
    out, cfg, _, _ = built
    tok = get_tok()
    tokens = np.memmap(out / "train.bin", dtype=np.uint16, mode="r")
    schedule = _read_jsonl(out / "schedule.jsonl")
    components = Counter(row["component"] for row in schedule)
    graph_kinds = Counter(
        row["graph_subcomponent"]
        for row in schedule
        if row["component"] == "graph"
        and "graph_subcomponent" in row
    )

    assert components.keys() == {"bed", "graph", "reasoning"}
    assert graph_kinds == {
        "peripheral": graph_kinds.total() * 7 // 10,
        "central": graph_kinds.total() * 2 // 10,
        "rule": graph_kinds.total() // 10,
    }
    mixture = json.loads((out / "mixture-manifest.json").read_text())
    assert mixture["selected"] == {
        "bed": 0.70,
        "graph": 0.15,
        "reasoning": 0.15,
    }
    assert mixture["ordered_development_fallbacks"] == [
        {"bed": 0.70, "graph": 0.15, "reasoning": 0.15},
        {"bed": 0.65, "graph": 0.15, "reasoning": 0.20},
        {"bed": 0.65, "graph": 0.20, "reasoning": 0.15},
    ]
    assert mixture["actual"]["total_records"] == len(schedule)
    assert mixture["actual"]["total_tokens"] == len(tokens)
    assert mixture["record_rounding"]["validated"]
    assert set(mixture["record_rounding"]["tolerance_tokens"]) == {
        "bed",
        "graph",
        "reasoning",
    }
    assert not any(
        row["record_id"].startswith("random-control-reservoir")
        for row in schedule
    )
    assert "random_control_reservoir" not in json.loads(
        (out / "schedule-manifest.json").read_text()
    )
    assert "random_control_reservoir" not in json.loads(
        (out / "mask-manifest.json").read_text()
    )

    for row in schedule:
        if row["component"] != "reasoning":
            continue
        record_tokens = tokens[row["token_start"] : row["token_end"]]
        assert int((record_tokens == tok.GRAPH_START).sum()) == 6
        assert int((record_tokens == tok.ANSWER_STATE).sum()) == 6
        position = row["token_start"] / cfg.total_tokens
        if position < 0.20:
            assert row["curriculum_band"] == 1
        elif position < 0.50:
            assert row["curriculum_band"] in {1, 2}
        else:
            assert row["curriculum_band"] in {1, 2, 4}


def test_mask_ledger_proves_coverage_and_matched_random_histogram(built):
    out, _, _, _ = built
    split = np.memmap(out / "split.weights.bin", dtype=np.uint8, mode="r")
    random_control = np.memmap(
        out / "random.weights.bin",
        dtype=np.uint8,
        mode="r",
    )
    ledger = _read_jsonl(out / "mask-ledger.jsonl")
    expected_rows = [
        row for row in ledger if row["condition"] == "expected_split"
    ]
    split_rows = [row for row in ledger if row["condition"] == "split"]
    random_rows = [row for row in ledger if row["condition"] == "random"]

    assert split_rows
    assert [
        (row["start"], row["end"], row["fact_id"])
        for row in expected_rows
    ] == [
        (row["start"], row["end"], row["fact_id"])
        for row in split_rows
    ]
    assert all("position_bin" in row for row in split_rows + random_rows)
    split_histogram = Counter(
        (row["length"], row["position_bin"]) for row in split_rows
    )
    random_histogram = Counter(
        (row["length"], row["position_bin"]) for row in random_rows
    )
    assert split_histogram == random_histogram
    assert int((split == 0).sum()) == int((random_control == 0).sum())
    for row in split_rows:
        assert row["role"] == "payload"
        assert not split[row["start"] : row["end"]].any()
    for row in random_rows:
        assert row["role"] == "random_control"
        assert not random_control[row["start"] : row["end"]].any()
    selected_random_positions = np.zeros(len(random_control), dtype=bool)
    for row in random_rows:
        selected_random_positions[row["start"] : row["end"]] = True
    assert np.array_equal(random_control == 0, selected_random_positions)


def test_graph_policy_eval_and_manifests_are_portable(built):
    out, cfg, schema, _ = built
    graph = _read_jsonl(out / "graph.jsonl")
    policy = json.loads((out / "route-policy.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    eval_manifest = json.loads((out / "eval-manifest.json").read_text())
    published_audit = json.loads(
        (out / "published-artifact-audit.json").read_text()
    )
    split_audit = json.loads((out / "split-audit.json").read_text())
    originals = _read_jsonl(out / "eval" / "original.jsonl")
    counterfactuals = _read_jsonl(out / "eval" / "counterfactual.jsonl")
    eval_graph = _read_jsonl(out / "eval" / "graph.jsonl")

    assert len(graph) > cfg.n_entities * 6
    assert {row["relation_id"] for row in graph} <= set(schema.codec_catalog)
    assert any(row["relation_id"].startswith("P") for row in graph)
    assert RelationSchema.from_path(out / "relation-schema.json") == schema
    with PackedGraphStore.load(out / "graph.store", schema.codec) as store:
        assert len(store) == len(graph)
    with PackedGraphStore.load(
        out / "eval" / "graph.store",
        schema.codec,
    ) as eval_store:
        assert len(eval_store) == len(eval_graph)
    with PackedGraphStore.load(
        out / "eval" / "factual-graph.store",
        schema.codec,
    ) as factual_store:
        assert len(factual_store) > 0
    assert policy["calibration"]["split_name"] == "development"
    assert 0.40 <= policy["calibration"]["route_rate"] <= 0.60
    policy_text = json.dumps(policy, sort_keys=True)
    for forbidden in ("audit_class", "answer", "target", "task", "outcome"):
        assert forbidden not in policy_text

    assert len(originals) == len(counterfactuals) == 3 * cfg.eval_pairs_per_task
    assert {row["meta"]["pair_id"] for row in originals} == {
        row["meta"]["pair_id"] for row in counterfactuals
    }
    assert all(
        original["answer"] != counterfactual["answer"]
        for original, counterfactual in zip(originals, counterfactuals)
    )
    assert {row["source_id"] for row in graph}.isdisjoint(
        {row["source_id"] for row in eval_graph}
    )
    assert eval_manifest["checks"] == {
        "exact_task_counts": True,
        "two_variants_per_pair": True,
        "answer_flips": True,
        "changed_supporting_row": True,
        "explicit_gold_actions": True,
        "fresh_sources_disjoint": True,
    }
    assert set(manifest["expectation_sha256"]) == {
        "development",
        "train",
        "protected_seen",
        "protected_heldout",
    }
    assert set(manifest["audit_sha256"]) == {"mask", "published", "split"}
    for name, digest in manifest["expectation_sha256"].items():
        assert digest == _sha256(out / "expectations" / f"{name}.json")
    assert manifest["audit_sha256"]["mask"] == _sha256(
        out / "mask-audit.json"
    )
    assert manifest["audit_sha256"]["split"] == _sha256(
        out / "split-audit.json"
    )
    assert manifest["audit_sha256"]["published"] == _sha256(
        out / "published-artifact-audit.json"
    )
    assert manifest["payload_inventory_sha256"] == _sha256(
        out / "payload-inventory.json"
    )
    assert manifest["generation_expectation_sha256"] == _sha256(
        out / "generation-expectations.json"
    )
    assert all(published_audit["checks"].values())
    assert all(published_audit["split_checks"].values())
    assert (
        published_audit["split_expectation_sha256"]
        == manifest["expectation_sha256"]
    )
    assert split_audit["phase"] == "post_publication_reopen"
    assert split_audit["expectation_sha256"] == manifest["expectation_sha256"]

    for artifact in manifest["artifacts"]:
        relative = Path(artifact["path"])
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        assert _sha256(out / relative) == artifact["sha256"]


def test_alias_exposure_manifest_proves_exact_catalog_coverage(built):
    out, _, schema, _ = built
    manifest = json.loads((out / "manifest.json").read_text())
    schedule = _read_jsonl(out / "schedule.jsonl")
    alias_rows = [
        row
        for row in schedule
        if row["record_id"].startswith("relation-alias:")
    ]
    observed = [
        (row["metadata"]["relation_id"], row["metadata"]["alias"])
        for row in alias_rows
    ]
    expected = [
        (relation.relation_id, alias)
        for relation in schema.catalog
        for alias in relation.aliases
    ]
    receipt = manifest["alias_exposure_receipt"]

    assert observed == expected
    assert len(observed) == len(set(observed))
    assert receipt == {
        "expected_pairs": len(expected),
        "observed_pairs": len(observed),
        "pairs_sha256": hashlib.sha256(
            _canonical_bytes(sorted(observed))
        ).hexdigest(),
        "complete": True,
    }
    assert manifest["alias_exposure_sha256"] == hashlib.sha256(
        _canonical_bytes(receipt)
    ).hexdigest()


def test_frozen_split_expectations_describe_production_plan_outputs(built):
    out, _, _, _ = built
    train = SplitArtifactExpectations.from_path(
        out / "expectations" / "train.json"
    )
    seen = SplitArtifactExpectations.from_path(
        out / "expectations" / "protected_seen.json"
    )
    heldout = SplitArtifactExpectations.from_path(
        out / "expectations" / "protected_heldout.json"
    )
    train_fact_ids = {
        stable_fact_id(GraphRow.from_json(row))
        for row in _read_jsonl(out / "graph.jsonl")
    }
    eval_fact_ids = {
        stable_fact_id(GraphRow.from_json(row))
        for row in _read_jsonl(out / "eval" / "graph.jsonl")
    }
    reasoning_record_ids = {
        row["record_id"]
        for row in _read_jsonl(out / "schedule.jsonl")
        if row["component"] == "reasoning"
    }
    heldout_question_ids = {
        row["qid"]
        for name in ("original.jsonl", "counterfactual.jsonl")
        for row in _read_jsonl(out / "eval" / name)
    }

    assert {
        fact_id
        for signature in train.world_signatures
        for fact_id in signature.fact_ids
    } == train_fact_ids
    assert {
        signature.artifact_id for signature in train.rendered_signatures
    } == reasoning_record_ids
    assert {
        fact_id
        for signature in seen.world_signatures
        for fact_id in signature.fact_ids
    } == eval_fact_ids
    assert {
        fact_id
        for signature in heldout.world_signatures
        for fact_id in signature.fact_ids
    } == eval_fact_ids
    assert {
        signature.artifact_id for signature in heldout.qa_signatures
    } == heldout_question_ids


def test_pre_emission_expectation_rejects_consistent_train_relation_mutation(
    built,
    tmp_path,
):
    out, _, schema, _ = built
    copied = tmp_path / "wrong-planned-train-relation"
    shutil.copytree(out, copied)
    frozen = json.loads((out / "manifest.json").read_text())[
        "expectation_sha256"
    ]
    schedule_path = copied / "schedule.jsonl"
    schedule = _read_jsonl(schedule_path)
    row = next(item for item in schedule if item["component"] == "reasoning")
    metadata = row["metadata"]
    replacement = next(
        relation
        for relation in schema.codec.relation_ids
        if relation != metadata["relations"][0]
    )
    metadata["relations"][0] = replacement
    metadata["relation_path_hash"] = composition_hash(metadata["relations"])
    _write_jsonl(schedule_path, schedule)
    schedule_manifest_path = copied / "schedule-manifest.json"
    schedule_manifest = json.loads(schedule_manifest_path.read_text())
    schedule_manifest["sha256"] = _sha256(schedule_path)
    schedule_manifest["bytes"] = schedule_path.stat().st_size
    schedule_manifest_path.write_text(
        json.dumps(schedule_manifest, indent=2, sort_keys=True) + "\n"
    )
    _refresh_post_emission_expectation(copied)

    with pytest.raises(
        ValueError,
        match="pre-emission generation artifact mismatch",
    ):
        verify_published_artifacts(
            copied,
            schema.codec,
            frozen_split_expectation_sha256=frozen,
        )


def test_pre_emission_expectation_rejects_consistent_heldout_leak(
    built,
    tmp_path,
):
    out, _, schema, _ = built
    copied = tmp_path / "heldout-leak"
    shutil.copytree(out, copied)
    frozen = json.loads((out / "manifest.json").read_text())[
        "expectation_sha256"
    ]
    train = SplitArtifactExpectations.from_path(
        out / "expectations" / "train.json"
    )
    by_hop = {
        signature.hop_count: signature.relations
        for signature in train.rendered_signatures
    }
    heldout_dir = copied / "eval" / "protected_heldout"
    originals = _read_jsonl(heldout_dir / "original.jsonl")
    leaked = next(
        row
        for row in originals
        if row["meta"]["composition_split"] == "heldout"
        and row["meta"]["hop_count"] in by_hop
    )
    leaked_relations = list(by_hop[leaked["meta"]["hop_count"]])
    pair_id = leaked["meta"]["pair_id"]
    for directory in (heldout_dir, copied / "eval"):
        for name in ("original.jsonl", "counterfactual.jsonl"):
            path = directory / name
            rows = _read_jsonl(path)
            for row in rows:
                if row["meta"]["pair_id"] == pair_id:
                    row["meta"]["relations"] = leaked_relations
                    row["meta"]["relation_path_hash"] = composition_hash(
                        leaked_relations
                    )
            _write_jsonl(path, rows)
    eval_manifest_path = copied / "eval-manifest.json"
    eval_manifest = json.loads(eval_manifest_path.read_text())
    eval_manifest["sha256"]["original"] = _sha256(
        copied / "eval" / "original.jsonl"
    )
    eval_manifest["sha256"]["counterfactual"] = _sha256(
        copied / "eval" / "counterfactual.jsonl"
    )
    eval_manifest_path.write_text(
        json.dumps(eval_manifest, indent=2, sort_keys=True) + "\n"
    )
    _refresh_post_emission_expectation(copied)

    with pytest.raises(
        ValueError,
        match="pre-emission generation artifact mismatch",
    ):
        verify_published_artifacts(
            copied,
            schema.codec,
            frozen_split_expectation_sha256=frozen,
        )


def test_pre_emission_expectation_rejects_consistent_eval_item_mutation(
    built,
    tmp_path,
):
    out, _, schema, _ = built
    copied = tmp_path / "wrong-published-eval-item"
    shutil.copytree(out, copied)
    frozen = json.loads((out / "manifest.json").read_text())[
        "expectation_sha256"
    ]
    path = copied / "eval" / "protected_seen" / "original.jsonl"
    rows = _read_jsonl(path)
    rows[0]["meta"]["template_id"] += ":mutated"
    _write_jsonl(path, rows)
    _refresh_post_emission_expectation(copied)

    with pytest.raises(
        ValueError,
        match="pre-emission generation artifact mismatch",
    ):
        verify_published_artifacts(
            copied,
            schema.codec,
            frozen_split_expectation_sha256=frozen,
        )


@pytest.mark.parametrize("role", ["plain", "action", "random_control"])
def test_final_audit_rejects_frozen_replay_token_mutation(
    built,
    tmp_path,
    role,
):
    out, _, schema, _ = built
    copied = tmp_path / f"mutated-{role}-token"
    shutil.copytree(out, copied)
    connection = sqlite3.connect(copied / "mask-occurrences.sqlite3")
    try:
        start, = connection.execute(
            """
            SELECT start
            FROM spans
            WHERE role = ?
            ORDER BY span_id
            LIMIT 1
            """,
            (role,),
        ).fetchone()
    finally:
        connection.close()
    train_path = copied / "train.bin"
    with train_path.open("r+b") as stream:
        stream.seek(start * np.dtype(np.uint16).itemsize)
        token = int.from_bytes(stream.read(2), "little")
        stream.seek(start * np.dtype(np.uint16).itemsize)
        stream.write(((token + 1) % (1 << 16)).to_bytes(2, "little"))
    _refresh_post_emission_expectation(copied)
    manifest = json.loads((out / "manifest.json").read_text())

    with pytest.raises(
        ValueError,
        match=r"train\.bin differs from frozen schedule plan",
    ):
        write_published_artifact_audit(
            copied,
            schema.codec,
            frozen_split_expectation_sha256=manifest[
                "expectation_sha256"
            ],
            frozen_generation_expectation_sha256=manifest[
                "generation_expectation_sha256"
            ],
        )


@pytest.mark.parametrize("mutation", ["role", "fact_id", "range"])
def test_final_audit_rejects_frozen_nonpayload_span_mutation(
    built,
    tmp_path,
    mutation,
):
    out, _, schema, _ = built
    copied = tmp_path / f"mutated-plain-span-{mutation}"
    shutil.copytree(out, copied)
    occurrence_path = copied / "mask-occurrences.sqlite3"
    connection = sqlite3.connect(occurrence_path)
    try:
        span_id, = connection.execute(
            """
            SELECT span_id
            FROM spans
            WHERE role = 'plain' AND end - start > 2
            ORDER BY span_id
            LIMIT 1
            """
        ).fetchone()
        if mutation == "role":
            connection.execute(
                "UPDATE spans SET role = 'rule' WHERE span_id = ?",
                (span_id,),
            )
        elif mutation == "fact_id":
            connection.execute(
                "UPDATE spans SET fact_id = 'forged' WHERE span_id = ?",
                (span_id,),
            )
        else:
            connection.execute(
                "UPDATE spans SET start = start + 1 WHERE span_id = ?",
                (span_id,),
            )
        connection.commit()
    finally:
        connection.close()
    _refresh_post_emission_expectation(copied)
    manifest = json.loads((out / "manifest.json").read_text())

    with pytest.raises(
        ValueError,
        match="replay span differs from frozen schedule plan",
    ):
        write_published_artifact_audit(
            copied,
            schema.codec,
            frozen_split_expectation_sha256=manifest[
                "expectation_sha256"
            ],
            frozen_generation_expectation_sha256=manifest[
                "generation_expectation_sha256"
            ],
        )


@pytest.mark.parametrize(
    ("relative_path", "mutate"),
    [
        (
            Path("graph.jsonl"),
            lambda rows: [{**rows[0], "target": rows[0]["target"] + "9"}]
            + rows[1:],
        ),
        (
            Path("eval/original.jsonl"),
            lambda rows: [{**rows[0], "answer": "mutated"}] + rows[1:],
        ),
        (
            Path("schedule.jsonl"),
            lambda rows: [
                {
                    **rows[0],
                    "curriculum_band": rows[0]["curriculum_band"] + 1,
                }
            ]
            + rows[1:],
        ),
    ],
)
def test_final_audit_rejects_published_row_mutation(
    built,
    tmp_path,
    relative_path,
    mutate,
):
    out, _, schema, _ = built
    copied = tmp_path / "mutated-published"
    shutil.copytree(out, copied)
    path = copied / relative_path
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in mutate(_read_jsonl(path))
        )
    )

    with pytest.raises(ValueError, match="published artifact"):
        verify_published_artifacts(copied, schema.codec)


def test_final_audit_rejects_published_packed_graph_mutation(
    built,
    tmp_path,
):
    out, _, schema, _ = built
    copied = tmp_path / "mutated-packed"
    shutil.copytree(out, copied)
    rows_path = copied / "graph.store" / "rows.bin"
    with rows_path.open("r+b") as stream:
        original = stream.read(1)
        stream.seek(0)
        stream.write(bytes((original[0] ^ 1,)))

    with pytest.raises(ValueError, match="published artifact"):
        verify_published_artifacts(copied, schema.codec)


def test_payload_inventory_is_frozen_from_canonical_rows(built):
    out, _, _, _ = built
    inventory = json.loads((out / "payload-inventory.json").read_text())
    train_rows = _read_jsonl(out / "graph.jsonl")

    assert inventory["version"] == 2
    assert {entry["scope"] for entry in inventory["entries"]} == {
        "train",
        "protected_seen",
        "protected_heldout",
    }
    assert {
        entry["fact_id"]
        for entry in inventory["entries"]
        if entry["scope"] == "train"
    } == {
        stable_fact_id(GraphRow.from_json(row))
        for row in train_rows
    }
    assert all(
        isinstance(entry["expected_occurrences"], int)
        and entry["expected_occurrences"] >= 0
        for entry in inventory["entries"]
    )
    assert all(
        entry["expected_occurrences"] == 0
        for entry in inventory["entries"]
        if entry["scope"].startswith("protected_")
    )
    assert sum(
        entry["expected_occurrences"]
        for entry in inventory["entries"]
        if entry["scope"] == "train"
    ) == sum(
        row["condition"] == "expected_split"
        for row in _read_jsonl(out / "mask-ledger.jsonl")
    )


def test_protected_inventory_covers_every_eval_world_beyond_train_capacity(
    tmp_path,
):
    out = tmp_path / "multi-world-eval"
    cfg = RelationalBuildConfig(
        n_entities=64,
        total_tokens=60_000,
        data_seed=17,
        world_size=64,
        eval_pairs_per_task=33,
        eval_pairs_per_world=8,
        guardrail_items=2,
        shared_text_eval_count=1,
    )

    build_relational_corpus(
        cfg,
        get_tok(),
        _bed_stream(),
        out,
        relation_schema=_fixture_schema(),
    )

    inventory = json.loads((out / "payload-inventory.json").read_text())
    eval_rows = _read_jsonl(out / "eval" / "graph.jsonl")
    eval_fact_ids = {
        stable_fact_id(GraphRow.from_json(row)) for row in eval_rows
    }
    protected_fact_ids = {
        scope: {
            entry["fact_id"]
            for entry in inventory["entries"]
            if entry["scope"] == scope
        }
        for scope in ("protected_seen", "protected_heldout")
    }

    assert json.loads((out / "eval-manifest.json").read_text())["worlds"] == 5
    assert protected_fact_ids == {
        "protected_seen": eval_fact_ids,
        "protected_heldout": eval_fact_ids,
    }


def test_malformed_renderer_fails_atomically_and_destination_is_retryable(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "retryable-corpus"
    cfg = RelationalBuildConfig(
        n_entities=64,
        total_tokens=20_000,
        data_seed=17,
        world_size=64,
        eval_pairs_per_task=1,
        guardrail_items=2,
        shared_text_eval_count=1,
    )
    original = relational.iter_graph_records

    def malformed_records(tok, worlds_factory):
        world = next(iter(worlds_factory()))
        fact = world.facts[0]
        while True:
            yield RenderedRecord(
                segments=(
                    TaggedSegment(
                        json.dumps(fact.row.target),
                        "payload",
                        fact.fact_id,
                        "target",
                    ),
                ),
                schedule=ScheduleEntry(
                    "graph",
                    fact.fact_id,
                    0,
                    0,
                ),
            )

    monkeypatch.setattr(relational, "iter_graph_records", malformed_records)
    with pytest.raises(RandomMaskUndersupplyError):
        build_relational_corpus(
            cfg,
            get_tok(),
            _bed_stream(),
            destination,
        )
    assert not destination.exists()

    monkeypatch.setattr(relational, "iter_graph_records", original)
    report = build_relational_corpus(
        cfg,
        get_tok(),
        _bed_stream(),
        destination,
    )
    assert destination.is_dir()
    assert all(report["checks"].values())


def test_failed_final_publication_audit_never_exposes_destination(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "audit-failed-corpus"
    cfg = RelationalBuildConfig(
        n_entities=64,
        total_tokens=20_000,
        data_seed=17,
        world_size=64,
        eval_pairs_per_task=1,
        guardrail_items=2,
        shared_text_eval_count=1,
    )

    def fail_audit(*_args, **_kwargs):
        raise RuntimeError("injected final publication audit failure")

    monkeypatch.setattr(
        relational,
        "write_published_artifact_audit",
        fail_audit,
    )
    with pytest.raises(RuntimeError, match="injected final"):
        build_relational_corpus(
            cfg,
            get_tok(),
            _bed_stream(),
            destination,
        )

    assert not destination.exists()


def test_builder_commits_all_guardrail_eval_inputs(built):
    out, cfg, _, report = built
    recognition = _read_jsonl(out / "eval" / "recognition.jsonl")
    factual = _read_jsonl(out / "eval" / "factual.jsonl")
    factual_graph = _read_jsonl(out / "eval" / "factual-graph.jsonl")
    internal = _read_jsonl(out / "eval" / "internal.jsonl")
    shared_text = _read_jsonl(out / "eval" / "shared_text.jsonl")
    route = json.loads((out / "eval" / "route-audit.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    artifact_names = {item["path"] for item in manifest["artifacts"]}

    assert len(recognition) == cfg.guardrail_items
    assert len(factual) == cfg.guardrail_items
    assert len(internal) == cfg.guardrail_items
    assert len(shared_text) == cfg.shared_text_eval_count
    assert all(
        len(item["choices"]) == 4
        and 0 <= item["answer_index"] < 4
        for item in recognition + internal
    )
    assert all(
        item["task"] == "factual_recall"
        and len(item["meta"]["gold_actions"]) == 6
        and item["meta"]["route"] in {"internal", "external"}
        and item["meta"]["target_kind"] in {"entity", "literal"}
        for item in factual
    )
    factual_addresses = {
        (
            row["source_id"],
            row["relation_id"],
            row["direction"],
        )
        for row in factual_graph
    }
    assert all(
        tuple(item["meta"]["gold_addresses"][0]) in factual_addresses
        for item in factual
    )
    assert {item["kind"] for item in internal} == {
        "rule",
        "central_fact",
    }
    assert set(route) == {
        "route_rate",
        "route_total",
        "low_use_high_entropy_external_rate",
        "low_use_high_entropy_total",
        "rules_top_centrality_internal_rate",
        "rules_top_centrality_total",
    }
    assert 0.40 <= route["route_rate"] <= 0.60
    assert route["low_use_high_entropy_external_rate"] >= 0.80
    assert route["rules_top_centrality_internal_rate"] >= 0.80
    assert report["eval"]["guardrail_items"] == cfg.guardrail_items
    assert {
        "eval/recognition.jsonl",
        "eval/factual.jsonl",
        "eval/factual-graph.jsonl",
        "eval/internal.jsonl",
        "eval/shared_text.jsonl",
        "eval/route-audit.json",
    } <= artifact_names


def test_factual_guardrail_samples_all_fact_payloads_not_route_subset(built):
    out, cfg, _, report = built
    factual = _read_jsonl(out / "eval" / "factual.jsonl")
    factual_rows = {
        stable_fact_id(row := GraphRow.from_json(raw)): row
        for raw in _read_jsonl(out / "eval" / "factual-graph.jsonl")
    }
    inventory = json.loads((out / "payload-inventory.json").read_text())
    inventory_fact_ids = {
        entry["fact_id"]
        for entry in inventory["entries"]
        if entry["scope"] == "train"
    }
    sampled = {item["meta"]["gold_fact_ids"][0] for item in factual}
    routes = {
        item["meta"]["gold_fact_ids"][0]: item["meta"]["route"]
        for item in factual
    }
    strata = {
        (item["meta"]["target_kind"], item["meta"]["route"])
        for item in factual
    }

    assert len(factual) == cfg.guardrail_items
    assert sampled
    assert sampled <= inventory_fact_ids
    assert set(routes.values()) == {"internal", "external"}
    assert strata == {
        ("entity", "internal"),
        ("entity", "external"),
        ("literal", "internal"),
        ("literal", "external"),
    }
    assert all(
        item["answer"]
        == (
            f"Q{factual_rows[item['meta']['gold_fact_ids'][0]].target}"
            if item["meta"]["target_kind"] == "entity"
            else factual_rows[item["meta"]["gold_fact_ids"][0]].target
        )
        for item in factual
    )
    assert report["guardrail_audit"]["factual_fact_ids"] == [
        item["meta"]["gold_fact_ids"][0] for item in factual
    ]
    assert report["fact_routes"] == routes
    assert report["payload_inventory"] == "payload-inventory.json"


def test_real_eval_answer_choices_are_token_prefix_free(built):
    out, _, _, _ = built
    tok = get_tok()
    choice_sets = [
        item["choices"]
        for name in ("recognition.jsonl", "internal.jsonl")
        for item in _read_jsonl(out / "eval" / name)
    ]
    choice_sets.extend(
        item["meta"]["answer_choices"]
        for name in ("original.jsonl", "counterfactual.jsonl", "factual.jsonl")
        for item in _read_jsonl(out / "eval" / name)
    )

    for choices in choice_sets:
        encoded = [tuple(tok.encode(choice)) for choice in choices]
        assert all(encoded)
        assert len(set(encoded)) == len(encoded)
        assert all(
            not (
                len(left) < len(right)
                and right[: len(left)] == left
            )
            for left in encoded
            for right in encoded
            if left != right
        )


def test_evaluator_guardrails_require_strict_analysis_artifact(built):
    from scripts.freeze_relational_study import make_fixture_freeze
    from scripts.make_relational_manifest import build_manifest
    from scripts.analyze_relational import analyze_runs, expected_run_keys
    from scripts.run_relational_evals import (
        _build_guardrail_source,
        produce_guardrail_measurements,
    )
    from train.model import GPT, GPTConfig

    out, _, schema, _ = built
    torch.manual_seed(5)
    model = GPT(
        GPTConfig(n_layer=1, n_head=1, d_model=32, ctx=1024)
    ).eval()
    tok = get_tok()
    factual_store = AtomicGraphStore.load(
        out / "eval" / "factual-graph.jsonl"
    )
    produced = {
        condition: produce_guardrail_measurements(
            model,
            tok,
            out,
            condition=condition,
            device="cpu",
            batch_size=4,
            factual_store=factual_store,
            codec=schema.codec,
        )
        for condition in ("dense", "split", "random")
    }

    assert all(
        set(value)
        == {
            "within_run_guardrails",
            "recognition_store_off",
            "factual_recall",
            "internal_accuracy",
            "language",
        }
        for value in produced.values()
    )
    assert produced["split"]["within_run_guardrails"]["mask"]["passed"]
    assert produced["dense"]["within_run_guardrails"]["mask"]["passed"]
    assert not produced["dense"]["within_run_guardrails"]["mask"][
        "external_mask_applicable"
    ]
    assert produced["random"]["within_run_guardrails"]["mask"]["passed"]
    assert set(produced["split"]["internal_accuracy"]["per_kind"]) == {
        "rule",
        "central_fact",
    }
    for index, condition in enumerate(("dense", "split", "random"), start=1):
        source = _build_guardrail_source(
            produced[condition],
            {
                "model_id": "d160m",
                "arm": condition,
                "seed": 1001,
                "checkpoint_sha256": str(index) * 64,
                "raw_token_count": 100,
                "evaluator_sha256": "4" * 64,
                "data_sha256": "5" * 64,
                "relation_schema_sha256": "6" * 64,
                "configuration_sha256": "8" * 64,
                "result_schema_sha256": "9" * 64,
                "provenance_sha256": "7" * 64,
            },
        )
        assert source["analysis_role"] == "confirmatory_source_only"
        assert "route" not in source

    def mode_summary(mode, score):
        return {
            "memory": mode,
            "tasks": {
                task: {
                    "counterfactual_pair_accuracy": score,
                    "n_pairs": 10_000,
                }
                for task in (
                    "path_composition",
                    "date_ordering",
                    "balanced_equality",
                )
            },
            "primary_composite": score,
        }

    manifest = build_manifest(make_fixture_freeze())
    runs = {}
    for key in expected_run_keys(manifest):
        _, condition, _, _ = key
        runs[key] = {
            "on": mode_summary("on", 0.5),
            "off": mode_summary("off", 0.5),
            "guardrails": produced[condition],
        }

    with pytest.raises(ValueError, match="strict GuardrailReport"):
        analyze_runs(runs, run_manifest=manifest)


def test_guardrail_producer_raises_on_missing_committed_artifact(
    built,
    tmp_path,
):
    from scripts.run_relational_evals import produce_guardrail_measurements
    from train.model import GPT, GPTConfig

    out, _, schema, _ = built
    copied = tmp_path / "incomplete"
    shutil.copytree(out, copied)
    (copied / "eval" / "recognition.jsonl").unlink()
    model = GPT(
        GPTConfig(n_layer=1, n_head=1, d_model=16, ctx=1024)
    ).eval()

    with pytest.raises(FileNotFoundError, match="recognition.jsonl"):
        produce_guardrail_measurements(
            model,
            get_tok(),
            copied,
            condition="dense",
            device="cpu",
            batch_size=4,
            factual_store=AtomicGraphStore.load(
                copied / "eval" / "factual-graph.jsonl"
            ),
            codec=schema.codec,
        )


def test_eval_validator_rejects_a_nonflipping_twin(built, tmp_path):
    out, cfg, _, _ = built
    copied = tmp_path / "eval-copy"
    copied.mkdir()
    for name in ("graph.jsonl", "original.jsonl", "counterfactual.jsonl"):
        shutil.copy(out / "eval" / name, copied / name)
    training_graph = tmp_path / "graph.jsonl"
    shutil.copy(out / "graph.jsonl", training_graph)

    counterfactuals = _read_jsonl(copied / "counterfactual.jsonl")
    originals = _read_jsonl(copied / "original.jsonl")
    counterfactuals[0]["answer"] = originals[0]["answer"]
    (copied / "counterfactual.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in counterfactuals)
    )

    with pytest.raises(ValueError, match="answers must flip"):
        relational.validate_eval_sets(
            cfg,
            training_graph,
            copied / "graph.jsonl",
            copied / "original.jsonl",
            copied / "counterfactual.jsonl",
        )


def test_eval_validator_rejects_missing_explicit_gold_actions(
    built,
    tmp_path,
):
    out, cfg, _, _ = built
    copied = tmp_path / "eval-gold-copy"
    copied.mkdir()
    for name in ("graph.jsonl", "original.jsonl", "counterfactual.jsonl"):
        shutil.copy(out / "eval" / name, copied / name)
    training_graph = tmp_path / "graph.jsonl"
    shutil.copy(out / "graph.jsonl", training_graph)
    originals = _read_jsonl(copied / "original.jsonl")
    del originals[0]["meta"]["gold_actions"]
    (copied / "original.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in originals)
    )

    with pytest.raises(ValueError, match="gold actions"):
        relational.validate_eval_sets(
            cfg,
            training_graph,
            copied / "graph.jsonl",
            copied / "original.jsonl",
            copied / "counterfactual.jsonl",
        )


def test_eval_validator_accepts_six_reads_with_implicit_budget_termination(
    built,
    tmp_path,
):
    out, cfg, _, _ = built
    copied = tmp_path / "eval-six-read-copy"
    copied.mkdir()
    for name in ("graph.jsonl", "original.jsonl", "counterfactual.jsonl"):
        shutil.copy(out / "eval" / name, copied / name)
    training_graph = tmp_path / "graph.jsonl"
    shutil.copy(out / "graph.jsonl", training_graph)
    originals = _read_jsonl(copied / "original.jsonl")
    counterfactuals = _read_jsonl(copied / "counterfactual.jsonl")
    for item in (originals[0], counterfactuals[0]):
        meta = item["meta"]
        reads = [action for action in meta["gold_actions"] if action["read"]]
        steps = list(
            zip(meta["gold_addresses"], meta["gold_fact_ids"], reads)
        )
        expanded = [steps[index % len(steps)] for index in range(6)]
        meta["gold_addresses"] = [address for address, _, _ in expanded]
        meta["gold_fact_ids"] = [fact_id for _, fact_id, _ in expanded]
        meta["gold_actions"] = [
            {**action, "read": True, "halt": False}
            for _, _, action in expanded
        ]
    (copied / "original.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in originals)
    )
    (copied / "counterfactual.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in counterfactuals)
    )

    checks = relational.validate_eval_sets(
        cfg,
        training_graph,
        copied / "graph.jsonl",
        copied / "original.jsonl",
        copied / "counterfactual.jsonl",
    )

    assert checks["explicit_gold_actions"]


def test_same_seed_rebuild_has_identical_artifact_hashes(built, tmp_path):
    first_out, cfg, schema, _ = built
    second_out = tmp_path / "rerun"
    build_relational_corpus(
        cfg,
        get_tok(),
        _bed_stream(),
        second_out,
        relation_schema=schema,
    )

    first = json.loads((first_out / "manifest.json").read_text())
    second = json.loads((second_out / "manifest.json").read_text())
    first_hashes = {
        artifact["path"]: artifact["sha256"]
        for artifact in first["artifacts"]
    }
    second_hashes = {
        artifact["path"]: artifact["sha256"]
        for artifact in second["artifacts"]
    }
    assert first_hashes == second_hashes


def test_bed_jsonl_stream_rewinds_deterministically(tmp_path):
    from scripts.build_relational_corpus import iter_bed_jsonl

    source = tmp_path / "bed.jsonl"
    source.write_text(
        '{"text":"first pinned passage"}\n'
        '{"text":"second pinned passage"}\n'
    )

    assert list(islice(iter_bed_jsonl(source), 5)) == [
        "first pinned passage",
        "second pinned passage",
        "first pinned passage",
        "second pinned passage",
        "first pinned passage",
    ]


def test_protected_bed_stream_verifies_snapshot_lock_before_yielding(tmp_path):
    from scripts.build_relational_corpus import iter_bed_jsonl

    source = tmp_path / "bed.jsonl"
    source.write_text('{"text":"pinned passage"}\n')
    lock = lock_bed_snapshot(
        source,
        repo_id="HuggingFaceFW/fineweb-edu",
        revision="a" * 40,
        config="sample-10BT",
        split="train",
    )
    source.write_text('{"text":"drifted passage"}\n')

    with pytest.raises(SourceDriftError, match="BED snapshot drift"):
        next(iter_bed_jsonl(source, lock=lock))


def test_corpus_command_runs_as_a_repo_relative_script():
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_relational_corpus.py",
            "--help",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--bed-jsonl" in completed.stdout
    assert "--bed-lock" in completed.stdout
    assert "--relation-schema" in completed.stdout
    assert "--artifact-mode" in completed.stdout
    assert "--guardrail-items" in completed.stdout
    assert "--shared-text-eval-count" in completed.stdout
