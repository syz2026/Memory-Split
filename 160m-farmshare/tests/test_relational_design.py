from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from evals.relational_design import (
    DESIGN_SIMULATION_SEED,
    ProtectedIdentityRegistry,
    commit_arm_label_permutation,
    load_blinded_development,
    run_prospective_design,
    simulate_design_power,
    validate_design_receipt,
)
from evals.relational_metrics import EXPECTED_TASKS
from tests.task8_helpers import (
    DEVELOPMENT_SEEDS,
    PROTECTED_SEEDS,
    make_rows,
    replace_row,
)


def _prepare_inputs(tmp_path, *, permutation_seed=71):
    split_path = tmp_path / "development-split.rows"
    dense_path = tmp_path / "development-dense.rows"
    split_path.write_text("opaque split outcomes\n")
    dense_path.write_text("opaque dense outcomes\n")
    commitment = commit_arm_label_permutation(
        tmp_path / "commitment",
        planned_inputs={"split": split_path, "dense": dense_path},
        rng_seed=permutation_seed,
    )
    rows = {
        split_path: make_rows(
            "development-split",
            "split",
            seeds=DEVELOPMENT_SEEDS,
            namespace="development",
            success=lambda *_: True,
        ),
        dense_path: make_rows(
            "development-dense",
            "dense",
            seeds=DEVELOPMENT_SEEDS,
            namespace="development",
            success=lambda *_: False,
        ),
    }
    return commitment, rows


def test_arm_permutation_is_persisted_before_any_outcome_load(tmp_path):
    commitment, rows = _prepare_inputs(tmp_path)
    calls = []

    def loader(path):
        assert commitment.path.is_dir()
        assert (commitment.path / "commitment.json").is_file()
        assert (commitment.path / "permutation-key.json").is_file()
        calls.append(path)
        return rows[path]

    blinded = load_blinded_development(commitment, loader)

    assert calls == list(commitment.planned_input_paths)
    assert set(blinded.rows_by_label) == {"arm_a", "arm_b"}
    assert set(blinded.source_arms) == {"arm_a", "arm_b"}
    assert set(blinded.source_arms.values()) == {"split", "dense"}
    for displayed, arm_rows in blinded.rows_by_label.items():
        assert {row.arm for row in arm_rows} == {
            blinded.source_arms[displayed]
        }
    assert len({blinded.permutation_commitment}) == 1


def test_temporal_api_rejects_unpersisted_or_removed_commitment(tmp_path):
    with pytest.raises((TypeError, ValueError), match="persisted|commitment"):
        load_blinded_development(object(), lambda _path: ())

    commitment, rows = _prepare_inputs(tmp_path)
    shutil.rmtree(commitment.path)
    with pytest.raises(ValueError, match="persisted|missing|commitment"):
        load_blinded_development(commitment, rows.__getitem__)


def test_commitment_rejects_overwrite_symlink_and_path_mutation(tmp_path):
    commitment, _ = _prepare_inputs(tmp_path)
    with pytest.raises(FileExistsError, match="exists"):
        commit_arm_label_permutation(
            commitment.path,
            planned_inputs={
                "split": tmp_path / "development-split.rows",
                "dense": tmp_path / "development-dense.rows",
            },
            rng_seed=71,
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|travers"):
        commit_arm_label_permutation(
            link / "commitment",
            planned_inputs={
                "split": tmp_path / "development-split.rows",
                "dense": tmp_path / "development-dense.rows",
            },
            rng_seed=71,
        )


def test_gate5_uses_exact_three_disjoint_development_seed_pairs(tmp_path):
    commitment, rows = _prepare_inputs(tmp_path)
    blinded = load_blinded_development(commitment, rows.__getitem__)
    protected = ProtectedIdentityRegistry.from_rows(
        make_rows("protected", "split")
    )

    receipt = run_prospective_design(blinded, protected=protected)

    assert receipt.development_seeds == DEVELOPMENT_SEEDS
    assert receipt.effect == 0.02
    assert receipt.pairs == 5
    assert receipt.studies == 10_000
    assert receipt.successes == 10_000
    assert receipt.power == 1.0
    assert receipt.variance_estimate == 0.0
    assert receipt.passed
    assert receipt.simulation_seed == DESIGN_SIMULATION_SEED
    assert receipt.permutation_commitment == (
        blinded.permutation_commitment
    )
    assert set(receipt.blinded_input_hashes) == {"arm_a", "arm_b"}
    assert validate_design_receipt(receipt.to_dict()) == receipt
    assert len(receipt.receipt_sha256) == 64
    assert len(receipt.decision_sha256) == 64


@pytest.mark.parametrize("studies", [9_999, 10_001, True])
def test_gate5_rejects_any_nonfrozen_study_count(tmp_path, studies):
    commitment, rows = _prepare_inputs(tmp_path)
    blinded = load_blinded_development(commitment, rows.__getitem__)
    with pytest.raises(ValueError, match="10,000|studies"):
        simulate_design_power(
            blinded,
            studies=studies,
            pairs=5,
            effect=0.02,
        )


@pytest.mark.parametrize("pairs", [4, 6, True])
def test_gate5_rejects_any_nonfrozen_pair_count(tmp_path, pairs):
    commitment, rows = _prepare_inputs(tmp_path)
    blinded = load_blinded_development(commitment, rows.__getitem__)
    with pytest.raises(ValueError, match="five|pairs"):
        simulate_design_power(
            blinded,
            studies=10_000,
            pairs=pairs,
            effect=0.02,
        )


def test_gate5_rejects_effect_or_seed_adaptation(tmp_path):
    commitment, rows = _prepare_inputs(tmp_path)
    blinded = load_blinded_development(commitment, rows.__getitem__)
    with pytest.raises(ValueError, match="0.02|effect"):
        simulate_design_power(blinded, effect=0.021)
    with pytest.raises(ValueError, match="simulation seed|frozen"):
        simulate_design_power(
            blinded,
            effect=0.02,
            rng_seed=DESIGN_SIMULATION_SEED + 1,
        )


def test_gate5_rejects_protected_seed_world_path_and_id_leakage(tmp_path):
    protected_rows = make_rows("protected", "split")
    protected = ProtectedIdentityRegistry.from_rows(
        protected_rows,
        paths=(tmp_path / "protected-results",),
    )

    for overlap in (
        "seed",
        "world",
        "path",
        "template",
        "qid",
        "filesystem",
    ):
        case = tmp_path / overlap
        case.mkdir()
        commitment, rows = _prepare_inputs(case)
        if overlap == "seed":
            rows = {
                path: make_rows(
                    f"development-{arm}",
                    arm,
                    seeds=PROTECTED_SEEDS[:3],
                    namespace="development",
                )
                for arm, path in zip(("split", "dense"), rows)
            }
        elif overlap != "filesystem":
            target = protected_rows[0]
            changed_rows = {}
            for path, arm_rows in rows.items():
                changed = list(arm_rows)
                pair_id = changed[0].pair_id
                for index, row in enumerate(changed):
                    if row.pair_id != pair_id:
                        continue
                    changes = {
                        "world": {"world_id": target.world_id},
                        "path": {
                            "relation_path_hash": target.relation_path_hash
                        },
                        "template": {
                            "template_id": target.template_id
                        },
                        "qid": {
                            "qid": target.qid
                            if row.variant == target.variant
                            else protected_rows[1].qid
                        },
                    }[overlap]
                    changed[index] = replace_row(row, **changes)
                changed_rows[path] = tuple(changed)
            rows = changed_rows
        else:
            protected_path = tmp_path / "protected-results"
            protected_path.mkdir()
            source = next(iter(rows))
            source.replace(protected_path / source.name)
            with pytest.raises(ValueError, match="protected|path"):
                commit_arm_label_permutation(
                    case / "path-commitment",
                    planned_inputs={
                        "split": protected_path / source.name,
                        "dense": next(iter(rows.keys() - {source})),
                    },
                    rng_seed=7,
                    protected=protected,
                )
            continue

        blinded = load_blinded_development(commitment, rows.__getitem__)
        with pytest.raises(ValueError, match=overlap if overlap != "qid" else "ID"):
            run_prospective_design(blinded, protected=protected)


def test_gate5_rejects_twin_arm_incompleteness_and_selective_rows(tmp_path):
    commitment, rows = _prepare_inputs(tmp_path)
    path = next(iter(rows))
    incomplete = dict(rows)
    incomplete[path] = incomplete[path][:-1]
    blinded = load_blinded_development(commitment, incomplete.__getitem__)
    with pytest.raises(ValueError, match="both variants|complete"):
        run_prospective_design(
            blinded,
            protected=ProtectedIdentityRegistry.empty(),
        )

    case = tmp_path / "selective-case"
    case.mkdir()
    commitment, rows = _prepare_inputs(case)
    path = next(iter(rows))
    rows[path] = tuple(
        replace_row(row, arm="selective") for row in rows[path]
    )
    blinded = load_blinded_development(commitment, rows.__getitem__)
    with pytest.raises(ValueError, match="arm"):
        run_prospective_design(
            blinded,
            protected=ProtectedIdentityRegistry.empty(),
        )


def test_gate5_receipt_is_deterministic_from_frozen_seed(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_commitment, first_rows = _prepare_inputs(first_dir)
    second_commitment, second_rows = _prepare_inputs(second_dir)
    first = run_prospective_design(
        load_blinded_development(first_commitment, first_rows.__getitem__),
        protected=ProtectedIdentityRegistry.empty(),
    )
    second = run_prospective_design(
        load_blinded_development(second_commitment, second_rows.__getitem__),
        protected=ProtectedIdentityRegistry.empty(),
    )

    assert first.to_dict() == second.to_dict()


def test_design_cli_has_separate_commit_and_analyze_phases():
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/analyze_relational_design.py", "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "commit" in completed.stdout
    assert "analyze" in completed.stdout
