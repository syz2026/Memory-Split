import json
from pathlib import Path

import yaml

from msctl.cohort import (
    ARMS,
    CHECKPOINT_UPDATES,
    COHORT_ID,
    EXPECTED_CONFIG_PATHS,
    MODEL_PARAMETERS,
    RAW_TARGETS,
    ROLES,
    SEEDS,
    TARGETS_PER_UPDATE,
    TERMINAL_UPDATES,
    load_cohort_assignment,
    validate_run_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_assignment_owns_every_pair_once():
    assignment = load_cohort_assignment(
        ROOT / "configs" / "cohort-assignment-135m-n10.json"
    )
    assert assignment["cohort_id"] == COHORT_ID
    assert assignment["model_parameters"] == MODEL_PARAMETERS
    assert assignment["provider_seeds"] == {
        "farmshare-l40s": [0, 1, 2, 5, 6, 7],
        "mit-slurm": [3, 4, 8, 9],
    }
    owned = [
        seed
        for role in assignment["operators"]
        for seed in role["seeds"]
    ]
    assert sorted(owned) == list(SEEDS)
    assert len(owned) == len(set(owned))
    assert {role["id"] for role in assignment["operators"]} == set(ROLES)


def test_twenty_configs_cover_the_frozen_matrix():
    paths = sorted((ROOT / "configs" / "135m-v2").glob("*.yaml"))
    assert {path.relative_to(ROOT).as_posix() for path in paths} == set(
        EXPECTED_CONFIG_PATHS
    )
    cells = set()
    for path in paths:
        raw = yaml.safe_load(path.read_text())
        cfg = validate_run_config(raw, relative_path=path.relative_to(ROOT).as_posix())
        cells.add((cfg["arm"], cfg["seed"]))
        assert cfg["model_parameters"] == MODEL_PARAMETERS
        assert cfg["total_tokens"] == RAW_TARGETS
        assert cfg["tokens_per_step"] == TARGETS_PER_UPDATE
        assert cfg["max_steps"] == TERMINAL_UPDATES
        assert tuple(cfg["checkpoint_updates"]) == CHECKPOINT_UPDATES
        assert cfg["dataset"]["complete_dataset"] is True
    assert cells == {(arm, seed) for arm in ARMS for seed in SEEDS}


def test_preregistration_matches_frozen_cohort():
    prereg = yaml.safe_load(
        (ROOT / "configs" / "preregistration-135m-v1.yaml").read_text()
    )
    protected = prereg["protected_cohort"]
    assert prereg["cohort_id"] == COHORT_ID
    assert protected["seeds"] == list(SEEDS)
    assert protected["terminal_n_pairs"] == 10
    assert protected["model_parameters"] == MODEL_PARAMETERS
    assert protected["raw_target_tokens"] == RAW_TARGETS
    assert protected["terminal_updates"] == TERMINAL_UPDATES
    assert prereg["analysis"]["bootstrap"]["n_resamples"] == 20_000
    assert prereg["analysis"]["practical_equivalence"][
        "margin_absolute_pair_accuracy"
    ] == 0.01


def test_assignment_is_canonical_json():
    path = ROOT / "configs" / "cohort-assignment-135m-n10.json"
    parsed = json.loads(path.read_text())
    assert path.read_text() == json.dumps(parsed, indent=2, sort_keys=True) + "\n"
