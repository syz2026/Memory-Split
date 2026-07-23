"""Small deterministic positive, practical-null, and invalid study fixtures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from types import MappingProxyType

from evals.confirmatory.contracts import (
    CHECKPOINT_SCHEMA,
    CONTRACT_VERSION,
    ITEM_SCHEMA,
    Arm,
    CheckpointRecord,
    ItemRecord,
    canonical_json_bytes,
)
from evals.confirmatory.inference import PairedObservation
from evals.confirmatory.inference import exact_sign_flip_test
from evals.confirmatory.metrics import (
    ItemOutcome,
    balanced_counterfactual_pair_metric,
)
from evals.confirmatory.reporting import (
    INFERENCE_EVIDENCE_SCHEMA,
    METRICS_SCHEMA,
    PRIMARY_CONTRAST_ID,
    PRIMARY_TEST_METHOD,
    REQUIRED_ARTIFACTS,
)
from evals.confirmatory.status import StatusAxes


@dataclass(frozen=True)
class DeterministicFixture:
    name: str
    observations: tuple[PairedObservation, ...]
    complete: bool
    valid: bool
    sign_consistent: bool
    expected_status: StatusAxes
    artifacts: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if self.name not in {"positive", "null", "invalid"}:
            raise ValueError("unknown deterministic fixture name")
        if not isinstance(self.expected_status, StatusAxes):
            raise ValueError("fixture expected_status must contain all three axes")
        if not self.observations:
            raise ValueError("deterministic fixture requires observations")
        if set(self.artifacts) != set(REQUIRED_ARTIFACTS) or any(
            not isinstance(content, bytes) or not content
            for content in self.artifacts.values()
        ):
            raise ValueError("deterministic fixture artifacts are not exact")
        object.__setattr__(
            self,
            "artifacts",
            MappingProxyType(dict(self.artifacts)),
        )
        for field in (
            "complete",
            "valid",
            "sign_consistent",
        ):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"{field} must be Boolean")


def _rows(*, treatment: float, control: float) -> tuple[PairedObservation, ...]:
    return tuple(
        PairedObservation(
            seed=seed,
            world_id=f"fixture-world-{world}",
            pair_id=f"seed-{seed}-world-{world}-pair-{pair}",
            treatment=treatment,
            control=control,
        )
        for seed in range(1001, 1006)
        for world in range(2)
        for pair in range(2)
    )


def _artifact_items() -> tuple[ItemRecord, ...]:
    shapes = {
        "composition_ood": (2, "heldout"),
        "joint_ood": (7, "heldout"),
    }
    records = []
    for family in ("graph", "non_path"):
        for stratum, (path_length, composition_split) in shapes.items():
            pair_id = f"{family}-{stratum}"
            for twin in ("original", "counterfactual"):
                records.append(
                    ItemRecord.from_dict(
                        {
                            "record_type": ITEM_SCHEMA,
                            "schema_version": CONTRACT_VERSION,
                            "item_id": f"{pair_id}-{twin}",
                            "pair_id": pair_id,
                            "twin": twin,
                            "stratum": stratum,
                            "family": family,
                            "world_id": f"world-{pair_id}",
                            "task": "fixture_task",
                            "path_length": path_length,
                            "composition_split": composition_split,
                            "composition_id": f"composition-{pair_id}",
                            "prompt": "fixture prompt",
                            "initial_slots": ["Q1", None, None, None],
                            "store_id": f"store-{pair_id}",
                            "memory_mode": "memory_on",
                            "control": "correct",
                        }
                    )
                )
    return tuple(sorted(records, key=lambda record: record.item_id))


def _artifact_checkpoints() -> tuple[CheckpointRecord, ...]:
    records = []
    for seed in range(5):
        for arm in (Arm.DENSE, Arm.SPLIT):
            records.append(
                CheckpointRecord.from_dict(
                    {
                        "record_type": CHECKPOINT_SCHEMA,
                        "schema_version": CONTRACT_VERSION,
                        "checkpoint_sha256": hashlib.sha256(
                            f"{seed}:{arm.value}".encode()
                        ).hexdigest(),
                        "model_id": "fixture-model",
                        "arm": arm.value,
                        "seed": seed,
                        "raw_token_count": 1,
                        "configuration_sha256": "b" * 64,
                        "corpus_sha256": "c" * 64,
                        "code_sha256": "d" * 64,
                    }
                )
            )
    return tuple(sorted(records, key=lambda record: (record.seed, record.arm.value)))


def _jsonl(records) -> bytes:
    return b"".join(canonical_json_bytes(record) for record in records)


def _artifact_bundle(
    *,
    split_correct: bool,
    dense_correct: bool,
    measured_validity_failure: bool,
) -> Mapping[str, bytes]:
    items = _artifact_items()
    checkpoints = _artifact_checkpoints()
    item_map = {item.item_id: item for item in items}
    outcomes = []
    summaries = []
    for checkpoint in checkpoints:
        correct = (
            split_correct
            if checkpoint.arm is Arm.SPLIT
            else dense_correct
        )
        rows = [
            ItemOutcome(
                item_id=item.item_id,
                pair_id=item.pair_id,
                twin=item.twin,
                stratum=item.stratum,
                family=item.family,
                seed=checkpoint.seed,
                world_id=item.world_id,
                checkpoint_sha256=checkpoint.checkpoint_sha256,
                arm=checkpoint.arm,
                memory_mode=item.memory_mode,
                control=item.control,
                proof_valid=correct,
                answer_valid=correct,
                complete=True,
                valid=True,
            )
            for item in items
        ]
        outcomes.extend(rows)
        summaries.append(
            (
                checkpoint.seed,
                checkpoint.arm,
                balanced_counterfactual_pair_metric(
                    rows,
                    items=item_map,
                    checkpoints={
                        checkpoint.checkpoint_sha256: checkpoint,
                    },
                ),
            )
        )
    summaries.sort(key=lambda entry: (entry[0], entry[1].value))
    deltas = []
    for seed in range(5):
        by_arm = {
            arm: summary.primary_accuracy
            for summary_seed, arm, summary in summaries
            if summary_seed == seed
        }
        deltas.append(by_arm[Arm.SPLIT] - by_arm[Arm.DENSE])
    exact = exact_sign_flip_test(deltas, alternative="greater")
    decision = exact.statistic > 0.0 and exact.p_value <= 0.05
    inference = {
        "record_type": INFERENCE_EVIDENCE_SCHEMA,
        "schema_version": CONTRACT_VERSION,
        "measured_validity_failure": measured_validity_failure,
        "primary_test": {
            "contrast_id": PRIMARY_CONTRAST_ID,
            "method": PRIMARY_TEST_METHOD,
            "alternative": "greater",
            "alpha": 0.05,
            "n_pairs": 5,
            "sign_assignments": 32,
            "equality_counted": True,
        },
        "paired_seed_bundle_deltas": deltas,
        "exact_test_result": {
            "statistic": exact.statistic,
            "extreme_count": exact.extreme_count,
            "p_value": exact.p_value,
            "reject_null": decision,
        },
    }
    ordered_outcomes = sorted(
        outcomes,
        key=lambda row: (row.seed, row.arm.value, row.item_id),
    )
    return {
        "checkpoints.jsonl": _jsonl(
            checkpoint.to_dict() for checkpoint in checkpoints
        ),
        "inference.json": canonical_json_bytes(inference),
        "items.jsonl": _jsonl(item.to_dict() for item in items),
        "metrics.json": canonical_json_bytes(
            {
                "record_type": METRICS_SCHEMA,
                "schema_version": CONTRACT_VERSION,
                "summaries": [
                    summary.to_dict() for _, _, summary in summaries
                ],
            }
        ),
        "outcomes.jsonl": _jsonl(
            outcome.to_dict() for outcome in ordered_outcomes
        ),
        "sealed-gold.jsonl": canonical_json_bytes(
            {"fixture": "sealed-gold"}
        ),
        "stores.jsonl": canonical_json_bytes({"fixture": "stores"}),
    }


def positive_fixture() -> DeterministicFixture:
    return DeterministicFixture(
        name="positive",
        observations=_rows(treatment=0.75, control=0.50),
        complete=True,
        valid=True,
        sign_consistent=True,
        expected_status=StatusAxes(
            "complete",
            "none",
            "supports_effect",
        ),
        artifacts=_artifact_bundle(
            split_correct=True,
            dense_correct=False,
            measured_validity_failure=False,
        ),
    )


def null_fixture() -> DeterministicFixture:
    return DeterministicFixture(
        name="null",
        observations=_rows(treatment=0.50, control=0.50),
        complete=True,
        valid=True,
        sign_consistent=False,
        expected_status=StatusAxes(
            "complete",
            "none",
            "inconclusive",
        ),
        artifacts=_artifact_bundle(
            split_correct=True,
            dense_correct=True,
            measured_validity_failure=False,
        ),
    )


def invalid_fixture() -> DeterministicFixture:
    return DeterministicFixture(
        name="invalid",
        observations=_rows(treatment=0.75, control=0.50),
        complete=True,
        valid=False,
        sign_consistent=True,
        expected_status=StatusAxes(
            "invalid",
            "none",
            "not_evaluated",
        ),
        artifacts=_artifact_bundle(
            split_correct=True,
            dense_correct=False,
            measured_validity_failure=True,
        ),
    )


def fixture_by_name(name: str) -> DeterministicFixture:
    fixtures = {
        "positive": positive_fixture,
        "null": null_fixture,
        "invalid": invalid_fixture,
    }
    try:
        factory = fixtures[name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown deterministic fixture: {name!r}") from exc
    return factory()


practical_null_fixture = null_fixture
