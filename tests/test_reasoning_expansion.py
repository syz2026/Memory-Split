from __future__ import annotations

import hashlib
import sys
from dataclasses import replace

import numpy as np
import pytest

from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.reasoning_expansion import (
    DEFAULT_RECIPE_PATH,
    DEFAULT_SOURCE_STAGE,
    ExpansionRecipe,
    GeneratedRecord,
    ReasoningGymGenerator,
    TaskSpec,
    _compile_extension,
    _ExactSubsetAccumulator,
    _probe_generator,
    load_expansion_recipe,
)


class _FixtureGenerator:
    _LENGTHS = (11, 13, 17, 19, 23, 1)

    def generate(self, task: str, index: int) -> GeneratedRecord:
        offset = 0 if task == "logic" else 3
        length = self._LENGTHS[(index + offset) % len(self._LENGTHS)]
        tokens = tuple((index + offset + position) % 997 for position in range(length))
        digest = hashlib.sha256(
            canonical_json_bytes({"index": index, "task": task, "tokens": tokens})
        ).hexdigest()
        return GeneratedRecord(
            task=task,
            source_index=index,
            token_ids=tokens,
            token_count=length,
            record_sha256=digest,
        )


def _tiny_recipe() -> ExpansionRecipe:
    recipe = load_expansion_recipe()
    return replace(
        recipe,
        extension_tokens=1000,
        targets_per_update=100,
        extension_updates=10,
        composite_tokens=recipe.base_tokens + 1000,
        composite_updates=(recipe.base_tokens + 1000) // 100,
        max_record_tokens=32,
        tasks=(
            TaskSpec("logic", "logic.fixture", {}, 1),
            TaskSpec("graph", "graphs.fixture", {}, 1),
        ),
    )


def test_reasoning_v3_recipe_is_integral_and_additive():
    recipe = load_expansion_recipe(DEFAULT_RECIPE_PATH)

    assert recipe.extension_tokens == 1_048_576_000
    assert recipe.extension_updates == 2000
    assert recipe.composite_tokens == recipe.base_tokens + recipe.extension_tokens
    assert recipe.composite_updates == 15_582
    assert len(recipe.tasks) == 16
    assert len({task.dataset for task in recipe.tasks}) == 16


def test_exact_subset_returns_source_ordered_solution():
    accumulator = _ExactSubsetAccumulator(31)
    selected = None
    lengths = [7, 12, 5, 19]
    for length in lengths:
        selected = accumulator.add(length)
        if selected is not None:
            break

    assert selected is not None
    assert tuple(sorted(selected)) == selected
    assert sum(lengths[index] for index in selected) == 31


def test_tiny_extension_hits_exact_horizon_without_padding(tmp_path):
    path = tmp_path / "targets.bin"
    report = _compile_extension(
        _tiny_recipe(),
        _FixtureGenerator(),
        path,
        finish_window=80,
        max_finish_candidates=256,
    )

    tokens = np.fromfile(path, dtype="<u2")
    assert len(tokens) == 1000
    assert report["record_count"] > 0
    assert sum(item["emitted_tokens"] for item in report["task_stats"]) == 1000
    assert all(item["emitted_records"] > 0 for item in report["task_stats"])
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest() == report["packed_stream_sha256"]
    )


@pytest.mark.skipif(
    not DEFAULT_SOURCE_STAGE.is_dir(),
    reason="frozen Reasoning Gym source stage is not present",
)
def test_reasoning_gym_new_task_probes_are_exact_and_order_independent():
    recipe = load_expansion_recipe()
    generator = ReasoningGymGenerator(
        DEFAULT_SOURCE_STAGE / recipe.source_relative_path,
        recipe,
    )

    probes = _probe_generator(generator, recipe)

    assert set(probes) == {task.dataset for task in recipe.tasks}
    assert all(
        len(task_probes) == 4 and all(probe["record_sha256"] for probe in task_probes)
        for task_probes in probes.values()
    )
    assert sys.modules["reasoning_gym"].__memorysplit_source__ == str(
        (DEFAULT_SOURCE_STAGE / recipe.source_relative_path).resolve()
    )
