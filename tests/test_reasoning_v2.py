from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from corpusgen.reasoning import (
    AnswerPointer,
    CompositionPremise,
    EqualityPremise,
    FactMetadata,
    SemanticFact,
    SemanticLeakageError,
    SupervisedField,
    audit_occurrence_closure,
    build_route_manifest,
    plan_occurrence_closure,
    route_score,
    serialize_answer_state,
    solve_graph_composition,
    solve_slot_equality,
    verify_proof,
)


def _routing_fact(index: int) -> FactMetadata:
    return FactMetadata(
        fact_id=f"fact-{index:02d}",
        source="fixture",
        record_type="graph",
        payload_entropy_bits=100 - index,
        scheduled_exposures=1,
        expected_reads=0,
        expected_hops=0,
        surfaces=(f"value-{index:02d}",),
    )


def _fact_with(**overrides) -> FactMetadata:
    values = {
        "fact_id": "fact",
        "source": "fixture",
        "record_type": "graph",
        "payload_entropy_bits": 10,
        "scheduled_exposures": 1,
        "expected_reads": 0,
        "expected_hops": 0,
        "surfaces": ("value",),
    }
    values.update(overrides)
    return FactMetadata(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"fact_id": 7},
        {"source": 7},
        {"record_type": 7},
        {"surfaces": ["value"]},
        {"surfaces": {"value"}},
    ],
)
def test_fact_metadata_rejects_nonstring_ids_and_unordered_surfaces(overrides):
    with pytest.raises(TypeError):
        _fact_with(**overrides)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SemanticFact(7, ("value",)),
        lambda: SemanticFact("fact", ["value"]),
        lambda: SemanticFact("fact", {"value"}),
        lambda: SupervisedField(7, "text"),
        lambda: SupervisedField("field", 7),
        lambda: SupervisedField("field", "text", supervised=1),
    ],
)
def test_semantic_records_require_string_fields_and_ordered_tuple_inputs(factory):
    with pytest.raises(TypeError):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CompositionPremise(7, hop=0, compose_code=1),
        lambda: EqualityPremise(7, slot=0, value="value"),
        lambda: EqualityPremise("fact", slot=0, value=7),
    ],
)
def test_proof_premises_require_string_ids_and_values(factory):
    with pytest.raises(TypeError):
        factory()


def test_route_score_uses_only_frozen_training_metadata():
    fact = FactMetadata(
        fact_id="fact-a",
        source="synthetic",
        record_type="graph",
        payload_entropy_bits=12,
        scheduled_exposures=2,
        expected_reads=4,
        expected_hops=2,
        surfaces=("cerulean",),
    )

    assert route_score(fact) == Fraction(7, 2)


@pytest.mark.parametrize(
    ("split", "expected_external"),
    [("Split50", 5), ("Split90", 9)],
)
def test_score_ranked_route_manifests_hit_exact_dose_without_hash_coin_flips(
    split,
    expected_external,
):
    facts = [_routing_fact(index) for index in range(10)]

    forward = build_route_manifest(facts, split)
    reverse = build_route_manifest(reversed(facts), split)

    assert forward.to_bytes() == reverse.to_bytes()
    assert forward.policy == "train-score-ranked-quota-v1"
    assert forward.metadata_scope == "training-only"
    assert forward.external_count == expected_external
    assert forward.quota_count == expected_external
    assert forward.rounding_error == Fraction(0)
    assert forward.external_fact_ids == tuple(
        f"fact-{index:02d}" for index in range(expected_external)
    )
    assert all("hash" not in decision.reason for decision in forward.decisions)


@pytest.mark.parametrize(
    ("split", "expected_external"),
    [("Split50", 2), ("Split90", 3)],
)
def test_route_quota_uses_the_unique_minimally_rounded_integer(
    split,
    expected_external,
):
    manifest = build_route_manifest(
        [_routing_fact(index) for index in range(3)],
        split,
    )

    assert manifest.external_count == expected_external
    assert abs(manifest.rounding_error) <= Fraction(1, 2)


def test_route_quota_also_meets_train_only_information_burden_dose():
    facts = [
        FactMetadata(
            fact_id=f"frequent-{index}",
            source="fixture",
            record_type="graph",
            payload_entropy_bits=10,
            scheduled_exposures=1,
            expected_reads=0,
            expected_hops=0,
            surfaces=(f"frequent-value-{index}",),
        )
        for index in range(9)
    ]
    facts.append(
        FactMetadata(
            fact_id="burden-heavy",
            source="fixture",
            record_type="graph",
            payload_entropy_bits=9,
            scheduled_exposures=100,
            expected_reads=0,
            expected_hops=0,
            surfaces=("burden-heavy-value",),
        )
    )

    manifest = build_route_manifest(facts, "Split90")

    assert manifest.external_count == 9
    assert "burden-heavy" in manifest.external_fact_ids
    assert manifest.information_burden_fraction >= Fraction(9, 10)
    assert manifest.information_burden_quota_met is True


def test_split90_rounds_up_to_preserve_its_minimum_fact_dose():
    facts = [
        FactMetadata(
            fact_id=f"equal-{index}",
            source="fixture",
            record_type="graph",
            payload_entropy_bits=10,
            scheduled_exposures=1,
            expected_reads=0,
            expected_hops=0,
            surfaces=(f"equal-value-{index}",),
        )
        for index in range(6)
    ]

    manifest = build_route_manifest(facts, "Split90")

    assert manifest.quota_count == 6
    assert manifest.external_count == 6
    assert manifest.information_burden_quota_met is True


def test_large_burden_repair_is_deterministic_with_bounded_fact_accesses():
    class AccessBoundFact(FactMetadata):
        burden_accesses = 0
        burden_access_limit = 100_000

        @property
        def information_burden_bits(self):
            type(self).burden_accesses += 1
            if type(self).burden_accesses > type(self).burden_access_limit:
                raise AssertionError("routing materialized a pair-list explosion")
            return super().information_burden_bits

    facts = tuple(
        AccessBoundFact(
            fact_id=f"fact-{index:05d}",
            source="large-fixture",
            record_type="graph",
            payload_entropy_bits=10 if index < 1_800 else 9,
            scheduled_exposures=1 if index < 1_800 else 100,
            expected_reads=0,
            expected_hops=0,
            surfaces=(f"value-{index:05d}",),
        )
        for index in range(2_000)
    )

    forward = build_route_manifest(facts, "Split90")
    AccessBoundFact.burden_accesses = 0
    reverse = build_route_manifest(reversed(facts), "Split90")

    assert forward.to_bytes() == reverse.to_bytes()
    assert forward.external_count == 1_800
    assert forward.information_burden_quota_met is True
    heavy_ids = {fact.fact_id for fact in facts[1_800:]}
    assert len(heavy_ids.intersection(forward.external_fact_ids)) == 180


def test_occurrence_closure_masks_every_supervised_declared_surface():
    facts = (SemanticFact("mars", ("Mars", "Red Planet")),)
    fields = (
        SupervisedField("payload", "Mars"),
        SupervisedField("return", "value=Mars; alias=Red Planet"),
        SupervisedField("state", "candidate Mars; final Mars"),
        SupervisedField("prompt", "Mars is named in an unsupervised prompt", False),
    )

    plan = plan_occurrence_closure(facts, fields)
    masks = plan.mask_for_routes({"mars"})
    report = audit_occurrence_closure(
        facts,
        fields,
        {"mars"},
        masks,
    )

    assert report.passed is True
    assert report.supervised_occurrences == 5
    assert report.masked_occurrences == 5
    assert report.unmasked_occurrences == ()
    assert masks["prompt"] == (1,) * len(fields[-1].text)


def test_occurrence_closure_fails_closed_on_one_leaked_copy():
    facts = (SemanticFact("mars", ("Mars",)),)
    fields = (
        SupervisedField("payload", "Mars"),
        SupervisedField("state", "candidate Mars"),
    )
    plan = plan_occurrence_closure(facts, fields)
    masks = {key: list(value) for key, value in plan.mask_for_routes({"mars"}).items()}
    masks["state"][-1] = 1

    with pytest.raises(SemanticLeakageError) as caught:
        audit_occurrence_closure(facts, fields, {"mars"}, masks)

    assert caught.value.report.passed is False
    assert len(caught.value.report.unmasked_occurrences) == 1


def test_occurrence_closure_fails_closed_on_missing_routed_fact_metadata():
    with pytest.raises(SemanticLeakageError) as caught:
        audit_occurrence_closure(
            (SemanticFact("known", ("value",)),),
            (SupervisedField("payload", "value"),),
            {"unknown"},
            {"payload": (1,) * len("value")},
        )

    assert caught.value.report.metadata_errors == (
        "routed fact lacks semantic metadata: unknown",
    )


@pytest.mark.parametrize(
    ("facts", "text"),
    [
        (
            (
                SemanticFact("planet", ("Mars",)),
                SemanticFact("deity", ("Mars",)),
            ),
            "Mars",
        ),
        (
            (
                SemanticFact("short", ("Mars",)),
                SemanticFact("long", ("Mars City",)),
            ),
            "Mars City",
        ),
    ],
)
def test_occurrence_closure_rejects_cross_fact_overlapping_surfaces(facts, text):
    with pytest.raises(ValueError, match="cross-fact overlapping"):
        plan_occurrence_closure(
            facts,
            (SupervisedField("payload", text),),
        )


def test_pointer_answer_state_is_canonical_and_cannot_repeat_surface_values():
    state = serialize_answer_state(
        AnswerPointer(slot=2, read_index=5, member_index=1),
        phase="candidate",
    )

    assert state == (
        '<|answer_state|>{"member_index":1,"phase":"candidate",'
        '"read_index":5,"slot":"<|slot_2|>"}'
    )
    assert "cerulean" not in state
    assert state == serialize_answer_state(
        AnswerPointer(slot=2, read_index=5, member_index=1),
        phase="candidate",
    )


def test_finite_domain_solver_emits_canonical_graph_composition_proof():
    premises = (
        CompositionPremise("edge-b", hop=1, compose_code=3),
        CompositionPremise("edge-a", hop=0, compose_code=2),
    )

    proof = solve_graph_composition(premises)

    assert proof.as_dict() == {
        "format": "memorysplit-canonical-proof-v1",
        "family": "graph_composition_mod4",
        "solver": "finite-domain-enumeration-v1",
        "premise_ids": ["edge-a", "edge-b"],
        "steps": [
            {
                "premise_id": "edge-a",
                "rule": "accumulate_mod4",
                "result": "r2",
            },
            {
                "premise_id": "edge-b",
                "rule": "accumulate_mod4",
                "result": "r1",
            },
        ],
        "conclusion": {"relation": "r1"},
    }
    assert verify_proof(proof, premises) is True
    assert verify_proof(
        replace(proof, conclusion=(("relation", "r0"),)),
        premises,
    ) is False


def test_finite_domain_solver_emits_value_free_non_path_equality_proof():
    premises = (
        EqualityPremise("category-b", slot=1, value="secret-category"),
        EqualityPremise("category-a", slot=0, value="secret-category"),
    )

    proof = solve_slot_equality(premises)

    assert proof.family == "slot_equality"
    assert proof.conclusion == (("equal", True),)
    assert proof.premise_ids == ("category-a", "category-b")
    assert "secret-category" not in proof.to_bytes().decode("utf-8")
    assert verify_proof(proof, premises) is True
    assert verify_proof(
        proof,
        (
            premises[0],
            replace(premises[1], value="different-category"),
        ),
    ) is False
