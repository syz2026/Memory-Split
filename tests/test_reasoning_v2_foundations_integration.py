from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from corpusgen.reasoning_v2 import load_recipe
from corpusgen.reasoning_v2 import source_lock as source_lock_module
from corpusgen.reasoning_v2.source_lock import resolve_source_lock
from reasoning_v2_fixtures import (
    fake_public_resolver,
    fixed_contract_environment,
)


ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "configs/reasoning-dataset-v2.json"
GENERATOR_COMMIT = "a" * 40

EXPECTED_POLICY_KEYS = (
    "fineweb_source_lock",
    "wikidata_source_lock",
    "finemath_repository",
    "finemath_subsets_in_order",
    "cross_deduplicate_finemath_against_fineweb",
    "objective_auxiliary_sources",
    "arc_conceptarc_total_max_percent",
    "teacher_generated_cot_total_max_percent",
    "teacher_generated_cot_requires",
    "excluded_from_claim_bearing_core",
)


class _RecordingResolver:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.requests = []

    def resolve(self, request, download_root):
        self.requests.append(request)
        return self.delegate.resolve(request, download_root)


class _NeverResolver:
    def resolve(self, request, download_root):
        pytest.fail(f"invalid recipe reached source resolver: {request.source_id}")


class _ForgedKey(str):
    pass


def _real_recipe():
    return load_recipe(RECIPE_PATH)


def _forged_policy(case: str) -> Mapping[str, object]:
    items = list(_real_recipe().source_policy.items())
    if case == "missing-key":
        items.pop()
    elif case == "extra-key":
        items.append(("unreviewed_source", "forged"))
    elif case == "wrong-key-order":
        items[0], items[1] = items[1], items[0]
    elif case == "wrong-key-type":
        key, value = items[2]
        items[2] = (_ForgedKey(key), value)
    elif case == "wrong-tuple-type":
        key, value = items[3]
        items[3] = (key, list(value))
    elif case == "wrong-scalar-type":
        key, _value = items[4]
        items[4] = (key, 1)
    elif case == "wrong-value":
        key, _value = items[2]
        items[2] = (key, "HuggingFaceTB/unreviewed")
    elif case == "wrong-tuple-order":
        key, value = items[5]
        items[5] = (key, tuple(reversed(value)))
    else:  # pragma: no cover - protects the test table itself
        raise AssertionError(f"unknown forged-policy case: {case}")
    return MappingProxyType(dict(items))


def test_real_immutable_recipe_drives_exact_request_catalog_and_lock(
    tmp_path,
    fake_public_resolver,
):
    recipe = _real_recipe()
    assert isinstance(recipe.source_policy, Mapping)
    assert not isinstance(recipe.source_policy, MutableMapping)
    assert tuple(recipe.source_policy) == EXPECTED_POLICY_KEYS
    assert isinstance(recipe.source_policy["finemath_subsets_in_order"], tuple)
    assert isinstance(recipe.source_policy["objective_auxiliary_sources"], tuple)
    assert isinstance(recipe.source_policy["teacher_generated_cot_requires"], tuple)
    assert isinstance(recipe.source_policy["excluded_from_claim_bearing_core"], tuple)

    resolver = _RecordingResolver(fake_public_resolver)
    lock = resolve_source_lock(
        recipe,
        resolver,
        tmp_path / "downloads",
        generator_commit=GENERATOR_COMMIT,
    )

    assert tuple(resolver.requests) == tuple(
        source_lock_module._REQUESTS_BY_ID[source_id]
        for source_id in source_lock_module._RESOLUTION_ORDER
    )
    assert lock.dataset_id == recipe.dataset_id
    assert lock.generator_commit == GENERATOR_COMMIT
    assert lock.source_catalog_sha256 == (
        source_lock_module.reviewed_source_catalog_sha256()
    )


def test_recipe_boundary_rejects_mutable_policy_mapping(
    tmp_path,
    fixed_contract_environment,
):
    recipe = _real_recipe()
    mutable = dict(recipe.source_policy)

    with pytest.raises(ValueError, match="immutable mapping"):
        resolve_source_lock(
            replace(recipe, source_policy=mutable),
            _NeverResolver(),
            tmp_path / "downloads",
            generator_commit=GENERATOR_COMMIT,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("missing-key", "keys and order"),
        ("extra-key", "keys and order"),
        ("wrong-key-order", "keys and order"),
        ("wrong-key-type", "type"),
        ("wrong-tuple-type", "type"),
        ("wrong-scalar-type", "type"),
        ("wrong-value", "drift"),
        ("wrong-tuple-order", "drift"),
    ),
)
def test_recipe_boundary_rejects_forged_immutable_policy(
    tmp_path,
    fixed_contract_environment,
    case,
    message,
):
    recipe = _real_recipe()

    with pytest.raises(ValueError, match=message):
        resolve_source_lock(
            replace(recipe, source_policy=_forged_policy(case)),
            _NeverResolver(),
            tmp_path / case,
            generator_commit=GENERATOR_COMMIT,
        )
