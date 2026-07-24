from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from corpusgen.reasoning_v2 import (
    BuildGeometry,
    LaneContract,
    ReasoningV2Recipe,
)
from corpusgen.reasoning_v2.contracts import (
    LANE_ORDER,
    balanced_record_lengths,
    geometry_for,
    hamilton_quotas,
    load_recipe,
)


ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "configs" / "reasoning-dataset-v2.json"


def _recipe_value() -> dict[str, object]:
    value = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_recipe(tmp_path: Path, value: object) -> Path:
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _nested_object(value: dict[str, object], path: tuple[str | int, ...]) -> dict:
    current: object = value
    for part in path:
        assert isinstance(current, (dict, list))
        current = current[part]  # type: ignore[index]
    assert isinstance(current, dict)
    return current


def test_full_recipe_recomputes_exact_hamilton_quotas():
    recipe = load_recipe(RECIPE_PATH)

    assert isinstance(recipe, ReasoningV2Recipe)
    assert recipe.dataset_id == "memorysplit-v2-20x-reasoning-max-cohort"
    assert recipe.total_targets == 7_120_879_616
    assert recipe.targets_per_update == 524_288
    assert recipe.optimizer_updates == 13_582
    assert recipe.context_length == 1_024
    assert recipe.shard_count == 32
    assert tuple(lane.lane_id for lane in recipe.lanes) == LANE_ORDER
    assert all(isinstance(lane, LaneContract) for lane in recipe.lanes)
    assert recipe.lane_quotas == (
        ("fineweb_edu", 1_780_219_904),
        ("finemath", 1_068_131_943),
        ("wikidata_graph", 1_424_175_923),
        ("synthetic_graph", 712_087_962),
        ("verified_synthetic_multihop", 1_068_131_942),
        ("wikidata_path_reasoning", 534_065_971),
        ("relational_refinement", 178_021_990),
        ("objective_auxiliary", 356_043_981),
    )
    assert hamilton_quotas(recipe.total_targets, recipe.lane_shares) == (
        recipe.lane_quotas
    )


def test_geometry_copies_full_recipe_and_builds_two_update_canary():
    recipe = load_recipe(RECIPE_PATH)
    full = geometry_for(recipe, "full")
    canary = geometry_for(recipe, "canary")

    assert full == BuildGeometry(
        profile="full",
        total_targets=7_120_879_616,
        targets_per_update=524_288,
        context_length=1_024,
        shard_count=32,
        allow_fewer_shards=False,
        lane_quotas=recipe.lane_quotas,
    )
    assert canary.total_targets == 1_048_576
    assert canary.total_targets // canary.targets_per_update == 2
    assert canary.context_length == recipe.context_length
    assert canary.shard_count == recipe.shard_count
    assert canary.allow_fewer_shards is True
    assert canary.lane_quotas == tuple(
        zip(
            LANE_ORDER,
            (262_144, 157_287, 209_715, 104_858, 157_286, 78_643, 26_214, 52_429),
            strict=True,
        )
    )


def test_geometry_rejects_unknown_profile():
    with pytest.raises(ValueError, match="profile"):
        geometry_for(load_recipe(RECIPE_PATH), "smoke")  # type: ignore[arg-type]


def test_balanced_lengths_close_quota_without_short_tail():
    lengths = balanced_record_lengths(1_068_131_943, 1_024)

    assert sum(lengths) == 1_068_131_943
    assert max(lengths) == 1_024
    assert min(lengths) >= 1_023


@pytest.mark.parametrize("bad", [True, 7_120_879_616.0, 0, -1])
def test_hamilton_rejects_non_integer_or_non_positive_totals(bad):
    shares = (("fineweb_edu", Fraction(1, 1)),)
    with pytest.raises((TypeError, ValueError)):
        hamilton_quotas(bad, shares)  # type: ignore[arg-type]


def test_hamilton_uses_frozen_lane_order_to_break_equal_remainders():
    shares = tuple((lane, Fraction(1, len(LANE_ORDER))) for lane in LANE_ORDER)

    assert hamilton_quotas(3, shares) == tuple(
        (lane, 1 if index < 3 else 0)
        for index, lane in enumerate(LANE_ORDER)
    )


@pytest.mark.parametrize(
    "shares",
    [
        tuple(
            (lane, Fraction(1, len(LANE_ORDER)))
            for lane in reversed(LANE_ORDER)
        ),
        tuple((lane, Fraction(1, 16)) for lane in LANE_ORDER),
        tuple((lane, 0.125) for lane in LANE_ORDER),
    ],
)
def test_hamilton_rejects_noncanonical_shares(shares):
    with pytest.raises((TypeError, ValueError)):
        hamilton_quotas(8, shares)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("targets", "context_length"),
    [(True, 1_024), (0, 1_024), (1_024.0, 1_024), (1_024, True), (1_024, 0)],
)
def test_balanced_lengths_rejects_non_positive_integer_inputs(
    targets,
    context_length,
):
    with pytest.raises((TypeError, ValueError)):
        balanced_record_lengths(targets, context_length)


def test_recipe_rejects_duplicate_json_key(tmp_path):
    text = RECIPE_PATH.read_text(encoding="utf-8")
    needle = '  "schema_version": 2,\n'
    assert text.count(needle) == 1
    path = tmp_path / "duplicate.json"
    path.write_text(text.replace(needle, needle + needle, 1), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_recipe(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_recipe_rejects_non_finite_json_number(tmp_path, constant):
    text = RECIPE_PATH.read_text(encoding="utf-8")
    needle = '"share_percent": 25.0'
    assert text.count(needle) == 1
    path = tmp_path / "non-finite.json"
    path.write_text(text.replace(needle, f'"share_percent": {constant}'), "utf-8")

    with pytest.raises(ValueError, match="finite"):
        load_recipe(path)


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("sprint_recipe",),
        ("sprint_recipe", "lanes", 0),
        ("sprint_recipe", "realized_token_allocation"),
        ("sprint_recipe", "source_policy"),
        ("sprint_recipe", "publication_requirements"),
        ("intervention",),
        ("intervention", "split90_minimum_percent"),
    ],
    ids=[
        "top-level",
        "sprint-recipe",
        "lane",
        "realized-allocation",
        "source-policy",
        "publication-requirements",
        "intervention",
        "split90-dose",
    ],
)
def test_recipe_rejects_unknown_contract_field(tmp_path, path):
    value = _recipe_value()
    _nested_object(value, path)["unknown_contract_field"] = None

    with pytest.raises(ValueError, match="unknown"):
        load_recipe(_write_recipe(tmp_path, value))


@pytest.mark.parametrize(
    ("path", "field"),
    [
        ((), "schema_version"),
        (("sprint_recipe",), "targets_per_update"),
        (("sprint_recipe", "lanes", 0), "verification"),
        (("sprint_recipe", "realized_token_allocation"), "token_quotas"),
        (("sprint_recipe", "source_policy"), "fineweb_source_lock"),
        (
            ("sprint_recipe", "publication_requirements"),
            "deterministic_rebuild_required",
        ),
        (("intervention",), "split90_minimum_percent"),
        (
            ("intervention", "split90_minimum_percent"),
            "distinct_offloadable_atomic_facts",
        ),
    ],
    ids=[
        "top-level",
        "sprint-recipe",
        "lane",
        "realized-allocation",
        "source-policy",
        "publication-requirements",
        "intervention",
        "split90-dose",
    ],
)
def test_recipe_rejects_missing_contract_field(tmp_path, path, field):
    value = _recipe_value()
    del _nested_object(value, path)[field]

    with pytest.raises(ValueError, match="missing"):
        load_recipe(_write_recipe(tmp_path, value))


@pytest.mark.parametrize("location", ["lanes", "allocation"])
def test_recipe_rejects_changed_lane_order(tmp_path, location):
    value = _recipe_value()
    sprint = _nested_object(value, ("sprint_recipe",))
    if location == "lanes":
        lanes = sprint["lanes"]
        assert isinstance(lanes, list)
        lanes[0], lanes[1] = lanes[1], lanes[0]
    else:
        allocation = _nested_object(
            value,
            ("sprint_recipe", "realized_token_allocation"),
        )
        lane_order = allocation["lane_order"]
        assert isinstance(lane_order, list)
        lane_order[0], lane_order[1] = lane_order[1], lane_order[0]

    with pytest.raises(ValueError, match="lane order"):
        load_recipe(_write_recipe(tmp_path, value))


def test_recipe_rejects_changed_committed_quota(tmp_path):
    value = _recipe_value()
    allocation = _nested_object(
        value,
        ("sprint_recipe", "realized_token_allocation"),
    )
    quotas = allocation["token_quotas"]
    assert isinstance(quotas, dict)
    quotas["fineweb_edu"] += 1

    with pytest.raises(ValueError, match="quota"):
        load_recipe(_write_recipe(tmp_path, value))


def test_recipe_rejects_changed_share_percent(tmp_path):
    value = _recipe_value()
    sprint = _nested_object(value, ("sprint_recipe",))
    lanes = sprint["lanes"]
    assert isinstance(lanes, list)
    assert isinstance(lanes[0], dict)
    lanes[0]["share_percent"] = 24.5

    with pytest.raises(ValueError, match="share"):
        load_recipe(_write_recipe(tmp_path, value))


def test_recipe_rejects_total_update_product_mismatch(tmp_path):
    value = _recipe_value()
    sprint = _nested_object(value, ("sprint_recipe",))
    sprint["optimizer_steps"] += 1

    with pytest.raises(ValueError, match="product"):
        load_recipe(_write_recipe(tmp_path, value))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("method", "round_robin"),
        ("tie_break", "alphabetical"),
    ],
)
def test_recipe_rejects_changed_allocation_contract(tmp_path, field, replacement):
    value = _recipe_value()
    allocation = _nested_object(
        value,
        ("sprint_recipe", "realized_token_allocation"),
    )
    allocation[field] = replacement

    with pytest.raises(ValueError, match=field.replace("_", " ")):
        load_recipe(_write_recipe(tmp_path, value))
