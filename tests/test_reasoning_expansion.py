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
    _PrefetchingRecordGenerator,
    _probe_generator,
    _verify_record_manifest,
    load_expansion_recipe,
)
from train.tokenizer import get_tok


class _FixtureGenerator:
    _LENGTHS = (11, 13, 17, 19, 23, 1)

    def generate(self, task: str, index: int) -> GeneratedRecord:
        offset = 0 if task == "logic" else 3
        length = self._LENGTHS[(index + offset) % len(self._LENGTHS)]
        tokens = (
            *((index + offset + position) % 997 for position in range(length - 1)),
            get_tok().EOT,
        )
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
    assert len(recipe.tasks) == 14
    assert len({task.dataset for task in recipe.tasks}) == 14
    assert {"ransom_note", "largest_island", "count_primes"} <= {
        task.dataset for task in recipe.tasks
    }


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
    manifest_path = tmp_path / "manifest.bin"
    report = _compile_extension(
        _tiny_recipe(),
        _FixtureGenerator(),
        path,
        manifest_path,
        finish_window=80,
        max_finish_candidates=256,
    )

    tokens = np.fromfile(path, dtype="<u2")
    assert len(tokens) == 1000
    assert report["record_count"] > 0
    assert sum(item["emitted_tokens"] for item in report["task_stats"]) == 1000
    assert all(item["emitted_records"] > 0 for item in report["task_stats"])
    assert {item["target_quota"] for item in report["task_stats"]} == {500}
    assert manifest_path.stat().st_size == report["manifest_bytes"]
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest() == report["packed_stream_sha256"]
    )


def test_prefetch_preserves_serial_task_streams():
    recipe = _tiny_recipe()
    expected = {
        task.dataset: [
            _FixtureGenerator().generate(task.dataset, index) for index in range(37)
        ]
        for task in recipe.tasks
    }
    with _PrefetchingRecordGenerator(
        _FixtureGenerator(),
        recipe.tasks,
        batch_size=8,
    ) as generator:
        actual = {
            task.dataset: [
                generator.generate(task.dataset, index) for index in range(37)
            ]
            for task in reversed(recipe.tasks)
        }

    assert actual == expected


def test_record_manifest_replays_every_emitted_record(tmp_path):
    recipe = _tiny_recipe()
    packed_path = tmp_path / "targets.bin"
    manifest_path = tmp_path / "manifest.bin"
    compiled = _compile_extension(
        recipe,
        _FixtureGenerator(),
        packed_path,
        manifest_path,
        finish_window=80,
        max_finish_candidates=256,
    )
    artifacts = {
        "packed_targets": {
            "bytes": packed_path.stat().st_size,
            "path": packed_path.name,
            "sha256": compiled["packed_stream_sha256"],
        },
        "record_manifest": {
            "bytes": manifest_path.stat().st_size,
            "path": manifest_path.name,
            "sha256": compiled["manifest_sha256"],
        },
    }
    report = _verify_record_manifest(
        tmp_path,
        recipe,
        artifacts,
        {
            "packed_stream_sha256": compiled["packed_stream_sha256"],
            "record_count": compiled["record_count"],
            "record_stream_sha256": compiled["record_stream_sha256"],
            "task_stats": compiled["task_stats"],
        },
        _FixtureGenerator(),
    )

    assert report["record_count"] == compiled["record_count"]
    assert report["replayed_records"] == compiled["record_count"]


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
