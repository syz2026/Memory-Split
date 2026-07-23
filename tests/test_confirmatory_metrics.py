from __future__ import annotations

from dataclasses import replace

import pytest

from evals.confirmatory.contracts import Arm, Control, MemoryMode, Stratum, Twin
from evals.confirmatory.metrics import (
    ItemOutcome,
    balanced_counterfactual_pair_metric,
)
from evals.confirmatory.status import (
    STATUS_PRECEDENCE,
    classify_status,
    resolve_status,
)


def _pair(
    stratum: Stratum,
    index: int,
    *,
    original: bool = True,
    counterfactual: bool = True,
) -> list[ItemOutcome]:
    common = {
        "pair_id": f"{stratum.value}-pair-{index}",
        "stratum": stratum,
        "seed": 1001,
        "world_id": f"{stratum.value}-world-{index // 2}",
        "checkpoint_sha256": "a" * 64,
        "arm": Arm.SPLIT,
        "memory_mode": MemoryMode.MEMORY_ON,
        "control": Control.CORRECT,
        "complete": True,
        "valid": True,
    }
    return [
        ItemOutcome(
            item_id=f"{common['pair_id']}-original",
            twin=Twin.ORIGINAL,
            proof_valid=original,
            answer_valid=original,
            **common,
        ),
        ItemOutcome(
            item_id=f"{common['pair_id']}-counterfactual",
            twin=Twin.COUNTERFACTUAL,
            proof_valid=counterfactual,
            answer_valid=counterfactual,
            **common,
        ),
    ]


def test_balanced_pair_metric_requires_both_twins_and_equal_weights_strata():
    rows = [
        *_pair(Stratum.IID, 0),
        *_pair(Stratum.COMPOSITION, 0),
        *_pair(Stratum.COMPOSITION, 1, counterfactual=False),
        *_pair(Stratum.LENGTH, 0, original=False),
        *_pair(Stratum.LENGTH, 1, counterfactual=False),
        *_pair(Stratum.LENGTH, 2, original=False, counterfactual=False),
        *_pair(Stratum.JOINT, 0),
        *_pair(Stratum.JOINT, 1),
        *_pair(Stratum.JOINT, 2),
        *_pair(Stratum.JOINT, 3),
    ]

    result = balanced_counterfactual_pair_metric(rows)

    assert result.by_stratum[Stratum.IID].value == 1.0
    assert result.by_stratum[Stratum.COMPOSITION].value == 0.5
    assert result.by_stratum[Stratum.LENGTH].value == 0.0
    assert result.by_stratum[Stratum.JOINT].value == 1.0
    assert result.balanced_accuracy == pytest.approx(0.625)
    assert result.overall_pair_accuracy.value == pytest.approx(0.6)
    assert result.overall_pair_accuracy.numerator == 6
    assert result.overall_pair_accuracy.denominator == 10
    assert result.arm is Arm.SPLIT
    assert result.memory_mode is MemoryMode.MEMORY_ON
    assert result.control is Control.CORRECT


def test_pair_metric_rejects_missing_duplicate_crossed_or_incomplete_twins():
    complete = [
        row
        for stratum in Stratum
        for row in _pair(stratum, 0)
    ]

    with pytest.raises(ValueError, match="both twins"):
        balanced_counterfactual_pair_metric(complete[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        balanced_counterfactual_pair_metric([*complete, complete[0]])

    crossed = list(complete)
    crossed[1] = replace(crossed[1], world_id="crossed-world")
    with pytest.raises(ValueError, match="metadata"):
        balanced_counterfactual_pair_metric(crossed)

    incomplete = list(complete)
    incomplete[0] = replace(incomplete[0], complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        balanced_counterfactual_pair_metric(incomplete)


def test_pair_metric_requires_all_four_named_strata_and_one_explicit_cell():
    rows = [
        row
        for stratum in (Stratum.IID, Stratum.COMPOSITION, Stratum.LENGTH)
        for row in _pair(stratum, 0)
    ]
    with pytest.raises(ValueError, match="strata"):
        balanced_counterfactual_pair_metric(rows)

    rows.extend(_pair(Stratum.JOINT, 0))
    rows[-1] = replace(rows[-1], memory_mode=MemoryMode.MEMORY_OFF)
    with pytest.raises(ValueError, match="cell"):
        balanced_counterfactual_pair_metric(rows)


def test_memory_modes_and_controls_are_explicit_closed_enums():
    assert {mode.value for mode in MemoryMode} == {"memory_off", "memory_on"}
    assert {arm.value for arm in Arm} == {"dense", "split", "random"}
    assert {control.value for control in Control} == {
        "correct",
        "shuffled_returns",
        "relevant_edge",
        "irrelevant_edge",
        "gold_path",
        "gold_returns",
        "no_query",
        "explicit_miss",
        "handle_swap",
        "entity_rename",
        "graph_isomorphism",
        "page_order_permutation",
    }


def test_status_precedence_is_frozen_and_resolves_worst_status():
    assert STATUS_PRECEDENCE == (
        "incomplete",
        "invalid",
        "directional_only",
        "sign_consistent_only",
        "inconclusive",
        "supports_effect",
        "supports_practical_null",
    )
    assert resolve_status(["supports_effect", "invalid"]) == "invalid"
    assert (
        resolve_status(["supports_practical_null", "directional_only"])
        == "directional_only"
    )
    with pytest.raises(ValueError, match="status"):
        resolve_status(["positive"])


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"observed_seeds": 0}, "incomplete"),
        ({"complete": False, "valid": False}, "incomplete"),
        ({"valid": False, "supports_effect": True}, "invalid"),
        ({"observed_seeds": 1}, "directional_only"),
        (
            {"observed_seeds": 3, "sign_consistent": True},
            "sign_consistent_only",
        ),
        ({"sign_consistent": False}, "inconclusive"),
        ({"supports_effect": True}, "supports_effect"),
        ({"supports_practical_null": True}, "supports_practical_null"),
    ],
)
def test_status_classification_honors_precedence(changes, expected):
    values = {
        "complete": True,
        "valid": True,
        "observed_seeds": 5,
        "required_seeds": 5,
        "sign_consistent": True,
        "supports_effect": False,
        "supports_practical_null": False,
    }
    values.update(changes)
    assert classify_status(**values) == expected
