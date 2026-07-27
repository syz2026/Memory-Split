"""Complete-matrix replay, frozen verdict assignment, and immutable reporting."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

from corpusgen.parallel.canonical import canonical_json_bytes
from evals.reasoning_v3.aws_authority import (
    TASK3_REPORT_KEY,
    S3ObjectVersion,
    VerifiedAwsAuthority,
)
from evals.reasoning_v3.inference import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_RNG_SEED,
    PRACTICAL_NULL_MARGIN,
    PRIMARY_ALPHA,
    PairedItemOutcome,
    _as_fraction,
    _fraction_dict,
    _hierarchical_paired_bootstrap,
    exact_paired_sign_test,
    right_step_aulc,
)
from evals.reasoning_v3.runner import (
    CHECKPOINT_STEPS,
    REQUIRED_VALIDITY_GATES,
    CheckpointBinding,
    CheckpointManifest,
    GateEvidence,
    ReleaseBinding,
    _checkpoint_result_key,
    _load_fixed_release,
    _load_signed_manifest,
    _release_dict,
    _replay_manifest_validity,
    _task3_code_bindings,
    _validate_gate_evidence,
    _validate_checkpoint_result,
)
from msctl.reasoning_cohort import ARMS, COHORT_ID, MODEL_PARAMETERS, SEEDS


LEARNABILITY_THRESHOLD = Fraction(1, 20)
REPORT_FORMAT = "memorysplit-reasoning-v3-scientific-report-v1"
REPORT_KEY = TASK3_REPORT_KEY
CLAIM_LIMITATION = (
    "No equal-mass protected control arm exists; Split90 has less active "
    "target-gradient mass. The conclusion is limited to the complete "
    "Dense-versus-Split90 intervention and does not identify a fact-specific "
    "mechanism."
)
_HEX = frozenset("0123456789abcdef")


class ReportingError(ValueError):
    """The collected checkpoint matrix or frozen report is invalid."""


@dataclass(frozen=True)
class PairingIdentity:
    initialization_sha256: str
    data_order_sha256: str
    runtime_sha256: str
    corpus_sha256: str
    config_sha256: str


@dataclass(frozen=True)
class ValiditySummary:
    passed: bool
    failures: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"failures": list(self.failures), "passed": self.passed}


@dataclass(frozen=True)
class ReplayedCheckpoint:
    arm: str
    seed: int
    step: int
    family_items: Mapping[str, tuple[tuple[str, bool], ...]]
    pairing: PairingIdentity
    validity: ValiditySummary
    release_identity_sha256: str
    evaluator_code_sha256: str
    result_key: str
    result_version_id: str
    result_sha256: str


@dataclass(frozen=True)
class VerdictInputs:
    validity: ValiditySummary
    observed_mean_delta: int | float | Fraction
    exact_p_value: int | float | Fraction
    interval_95: tuple[int | float | Fraction, int | float | Fraction]
    interval_90: tuple[int | float | Fraction, int | float | Fraction]
    dense_mean_accuracy: int | float | Fraction
    split90_mean_accuracy: int | float | Fraction


@dataclass(frozen=True)
class VerdictDecision:
    label: str
    supports_effect: bool
    supports_practical_null: bool
    learnable: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


@dataclass(frozen=True)
class PublishedScientificReport:
    report: Mapping[str, Any]
    object_ref: S3ObjectVersion


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _HEX
    )


def _finite_fraction(value: object, label: str) -> Fraction:
    try:
        return _as_fraction(value, label)
    except ValueError as error:
        raise ReportingError(f"{label} must be finite") from error


def assign_frozen_verdict(inputs: VerdictInputs) -> VerdictDecision:
    """Assign one mutually exclusive frozen exploratory label."""

    if not isinstance(inputs, VerdictInputs):
        raise ReportingError("verdict inputs are malformed")
    observed = _finite_fraction(inputs.observed_mean_delta, "observed mean delta")
    p_value = _finite_fraction(inputs.exact_p_value, "exact p-value")
    lower_95, upper_95 = (
        _finite_fraction(inputs.interval_95[0], "95% lower bound"),
        _finite_fraction(inputs.interval_95[1], "95% upper bound"),
    )
    lower_90, upper_90 = (
        _finite_fraction(inputs.interval_90[0], "90% lower bound"),
        _finite_fraction(inputs.interval_90[1], "90% upper bound"),
    )
    dense = _finite_fraction(inputs.dense_mean_accuracy, "dense mean accuracy")
    split90 = _finite_fraction(inputs.split90_mean_accuracy, "split90 mean accuracy")
    if (
        not 0 <= p_value <= 1
        or lower_95 > upper_95
        or lower_90 > upper_90
        or not 0 <= dense <= 1
        or not 0 <= split90 <= 1
        or not isinstance(inputs.validity, ValiditySummary)
        or not isinstance(inputs.validity.passed, bool)
    ):
        raise ReportingError("verdict inputs lie outside the frozen domain")
    learnable = max(dense, split90) >= LEARNABILITY_THRESHOLD
    effect = (
        inputs.validity.passed
        and observed > 0
        and p_value <= PRIMARY_ALPHA
        and lower_95 > 0
    )
    practical_null = (
        inputs.validity.passed
        and learnable
        and lower_90 > -PRACTICAL_NULL_MARGIN
        and upper_90 < PRACTICAL_NULL_MARGIN
        and not effect
    )
    if not inputs.validity.passed:
        return VerdictDecision(
            label="invalid",
            supports_effect=False,
            supports_practical_null=False,
            learnable=learnable,
            reasons=inputs.validity.failures
            or ("one or more frozen validity gates failed",),
        )
    if effect:
        return VerdictDecision(
            label="exploratory_supports_effect",
            supports_effect=True,
            supports_practical_null=False,
            learnable=learnable,
            reasons=(
                "observed mean is positive",
                "exact upper-tail p-value is at most 0.05",
                "95% nearest-rank lower bound is strictly positive",
            ),
        )
    if not learnable:
        return VerdictDecision(
            label="inconclusive_floor",
            supports_effect=False,
            supports_practical_null=False,
            learnable=False,
            reasons=(
                "both arm mean accuracies are below the frozen 0.05 "
                "absolute learnability threshold",
            ),
        )
    if practical_null:
        return VerdictDecision(
            label="exploratory_supports_practical_null",
            supports_effect=False,
            supports_practical_null=True,
            learnable=True,
            reasons=(
                "90% nearest-rank interval lies strictly inside (-0.01, +0.01)",
            ),
        )
    return VerdictDecision(
        label="inconclusive",
        supports_effect=False,
        supports_practical_null=False,
        learnable=True,
        reasons=("neither frozen support rule passed",),
    )


def _validate_complete_matrix(
    checkpoints: Sequence[ReplayedCheckpoint],
) -> dict[tuple[str, int, int], ReplayedCheckpoint]:
    if isinstance(checkpoints, (str, bytes, bytearray)):
        raise ReportingError("checkpoint matrix must be a sequence")
    expected = {
        (arm, seed, step)
        for seed in SEEDS
        for arm in ARMS
        for step in CHECKPOINT_STEPS
    }
    by_cell: dict[tuple[str, int, int], ReplayedCheckpoint] = {}
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, ReplayedCheckpoint):
            raise ReportingError("matrix rows must be replayed checkpoints")
        cell = (checkpoint.arm, checkpoint.seed, checkpoint.step)
        if cell not in expected:
            raise ReportingError(f"unexpected checkpoint matrix cell: {cell}")
        if cell in by_cell:
            raise ReportingError(f"duplicate checkpoint matrix cell: {cell}")
        by_cell[cell] = checkpoint
    missing = sorted(expected - set(by_cell))
    if missing:
        raise ReportingError(
            f"checkpoint matrix is incomplete: missing {len(missing)} cells"
        )
    if len(by_cell) != 100:
        raise ReportingError("checkpoint matrix must contain exactly 100 cells")
    return by_cell


def _replay_checkpoint_scores(
    checkpoint: ReplayedCheckpoint,
    *,
    family_order: Sequence[str],
    items_per_family: int,
    expected_item_ids: Mapping[str, tuple[str, ...]] | None,
    ) -> tuple[
    Fraction,
    dict[str, Fraction],
    dict[str, tuple[tuple[str, bool], ...]],
    dict[str, tuple[str, ...]],
]:
    if (
        not isinstance(checkpoint.family_items, Mapping)
        or set(checkpoint.family_items) != set(family_order)
    ):
        raise ReportingError("replayed checkpoint family set differs")
    scores: dict[str, Fraction] = {}
    normalized: dict[str, tuple[tuple[str, bool], ...]] = {}
    identities: dict[str, tuple[str, ...]] = {}
    for family in family_order:
        rows = checkpoint.family_items[family]
        if (
            not isinstance(rows, tuple)
            or len(rows) != items_per_family
            or any(
                not isinstance(row, tuple)
                or len(row) != 2
                or not isinstance(row[0], str)
                or not row[0]
                or type(row[1]) is not bool
                for row in rows
            )
        ):
            raise ReportingError(
                f"replayed item rows differ for {checkpoint.seed}/{family}"
            )
        item_ids = tuple(row[0] for row in rows)
        if len(item_ids) != len(set(item_ids)):
            raise ReportingError(f"duplicate replayed item identity for {family}")
        if expected_item_ids is not None and expected_item_ids[family] != item_ids:
            raise ReportingError(
                f"registry item order differs for family {family}"
            )
        normalized[family] = rows
        identities[family] = item_ids
        scores[family] = Fraction(
            sum(int(row[1]) for row in rows),
            items_per_family,
        )
    macro = sum(
        (scores[family] for family in family_order),
        start=Fraction(),
    ) / len(family_order)
    return macro, scores, normalized, identities


def _pairing_failures(
    by_cell: Mapping[tuple[str, int, int], ReplayedCheckpoint],
) -> set[str]:
    failures: set[str] = set()
    for seed in SEEDS:
        identities = {
            by_cell[(arm, seed, step)].pairing
            for arm in ARMS
            for step in CHECKPOINT_STEPS
        }
        if len(identities) != 1:
            failures.add(f"paired_identity_seed_{seed}")
    if (
        len({row.release_identity_sha256 for row in by_cell.values()}) != 1
        or len({row.evaluator_code_sha256 for row in by_cell.values()}) != 1
    ):
        failures.add("no_substitution")
    return failures


def _validate_replayed_metadata(checkpoint: ReplayedCheckpoint) -> None:
    if (
        not isinstance(checkpoint.pairing, PairingIdentity)
        or not isinstance(checkpoint.validity, ValiditySummary)
        or not isinstance(checkpoint.validity.passed, bool)
        or not isinstance(checkpoint.validity.failures, tuple)
        or any(not isinstance(item, str) or not item for item in checkpoint.validity.failures)
        or not all(_is_sha256(value) for value in asdict(checkpoint.pairing).values())
        or not _is_sha256(checkpoint.release_identity_sha256)
        or not _is_sha256(checkpoint.evaluator_code_sha256)
        or not isinstance(checkpoint.result_key, str)
        or not checkpoint.result_key
        or not isinstance(checkpoint.result_version_id, str)
        or not checkpoint.result_version_id
        or not _is_sha256(checkpoint.result_sha256)
    ):
        raise ReportingError("replayed checkpoint metadata is malformed")


def _build_scientific_report(
    checkpoints: Sequence[ReplayedCheckpoint],
    *,
    family_order: Sequence[str],
    items_per_family: int,
) -> dict[str, Any]:
    """Analyze a structurally complete matrix after raw-item/gold replay."""

    families = tuple(family_order)
    if (
        not families
        or len(families) != len(set(families))
        or any(not isinstance(family, str) or not family for family in families)
        or isinstance(items_per_family, bool)
        or not isinstance(items_per_family, int)
        or items_per_family <= 0
    ):
        raise ReportingError("report registry dimensions are malformed")
    by_cell = _validate_complete_matrix(checkpoints)
    expected_item_ids: dict[str, tuple[str, ...]] | None = None
    macro_scores: dict[tuple[str, int, int], Fraction] = {}
    family_scores: dict[tuple[str, int, int], dict[str, Fraction]] = {}
    normalized_items: dict[
        tuple[str, int, int],
        dict[str, tuple[tuple[str, bool], ...]],
    ] = {}
    failures: set[str] = set()
    for cell in sorted(by_cell, key=lambda item: (item[1], ARMS.index(item[0]), item[2])):
        checkpoint = by_cell[cell]
        _validate_replayed_metadata(checkpoint)
        macro, scores, items, identities = _replay_checkpoint_scores(
            checkpoint,
            family_order=families,
            items_per_family=items_per_family,
            expected_item_ids=expected_item_ids,
        )
        if expected_item_ids is None:
            expected_item_ids = identities
        macro_scores[cell] = macro
        family_scores[cell] = scores
        normalized_items[cell] = items
        if not checkpoint.validity.passed:
            failures.update(
                checkpoint.validity.failures
                or ("measured_validity_gate_failed",)
            )
    failures.update(_pairing_failures(by_cell))
    validity = ValiditySummary(
        passed=not failures,
        failures=tuple(sorted(failures)),
    )

    terminal = CHECKPOINT_STEPS[-1]
    seed_deltas = []
    paired_items: list[PairedItemOutcome] = []
    dense_terminal = []
    split_terminal = []
    for seed in SEEDS:
        dense_cell = ("dense", seed, terminal)
        split_cell = ("split90", seed, terminal)
        dense_macro = macro_scores[dense_cell]
        split_macro = macro_scores[split_cell]
        dense_terminal.append(dense_macro)
        split_terminal.append(split_macro)
        seed_deltas.append(split_macro - dense_macro)
        for family in families:
            dense_rows = normalized_items[dense_cell][family]
            split_rows = normalized_items[split_cell][family]
            for dense_row, split_row in zip(dense_rows, split_rows, strict=True):
                if dense_row[0] != split_row[0]:
                    raise ReportingError("terminal paired item identities differ")
                paired_items.append(
                    PairedItemOutcome(
                        seed=seed,
                        family=family,
                        item_id=dense_row[0],
                        dense_correct=dense_row[1],
                        split90_correct=split_row[1],
                    )
                )
    exact = exact_paired_sign_test(seed_deltas)
    bootstrap = _hierarchical_paired_bootstrap(
        paired_items,
        expected_seeds=SEEDS,
        family_order=families,
        items_per_family=items_per_family,
        n_draws=BOOTSTRAP_DRAWS,
        rng_seed=BOOTSTRAP_RNG_SEED,
    )
    arm_means = {
        "dense": sum(dense_terminal) / len(dense_terminal),
        "split90": sum(split_terminal) / len(split_terminal),
    }
    decision = assign_frozen_verdict(
        VerdictInputs(
            validity=validity,
            observed_mean_delta=exact.statistic,
            exact_p_value=exact.p_value,
            interval_95=(
                bootstrap.interval_95.lower,
                bootstrap.interval_95.upper,
            ),
            interval_90=(
                bootstrap.interval_90.lower,
                bootstrap.interval_90.upper,
            ),
            dense_mean_accuracy=arm_means["dense"],
            split90_mean_accuracy=arm_means["split90"],
        )
    )

    seed_macro_accuracy: dict[str, Any] = {}
    seed_macro_deltas: dict[str, list[float]] = {}
    seed_macro_aulc: dict[str, Any] = {}
    seed_macro_aulc_deltas: dict[str, float] = {}
    family_accuracy: dict[str, Any] = {family: {} for family in families}
    family_aulc: dict[str, Any] = {family: {} for family in families}
    for seed in SEEDS:
        dense_values = [
            macro_scores[("dense", seed, step)] for step in CHECKPOINT_STEPS
        ]
        split_values = [
            macro_scores[("split90", seed, step)] for step in CHECKPOINT_STEPS
        ]
        deltas = [
            split - dense
            for dense, split in zip(dense_values, split_values, strict=True)
        ]
        seed_macro_accuracy[str(seed)] = {
            "dense": [float(value) for value in dense_values],
            "split90": [float(value) for value in split_values],
        }
        seed_macro_deltas[str(seed)] = [float(value) for value in deltas]
        dense_aulc = right_step_aulc(CHECKPOINT_STEPS, dense_values)
        split_aulc = right_step_aulc(CHECKPOINT_STEPS, split_values)
        seed_macro_aulc[str(seed)] = {
            "dense": float(dense_aulc),
            "split90": float(split_aulc),
        }
        seed_macro_aulc_deltas[str(seed)] = float(split_aulc - dense_aulc)
        for family in families:
            dense_family = [
                family_scores[("dense", seed, step)][family]
                for step in CHECKPOINT_STEPS
            ]
            split_family = [
                family_scores[("split90", seed, step)][family]
                for step in CHECKPOINT_STEPS
            ]
            family_delta = [
                split - dense
                for dense, split in zip(
                    dense_family,
                    split_family,
                    strict=True,
                )
            ]
            family_accuracy[family][str(seed)] = {
                "dense": [float(value) for value in dense_family],
                "delta": [float(value) for value in family_delta],
                "split90": [float(value) for value in split_family],
            }
            dense_family_aulc = right_step_aulc(
                CHECKPOINT_STEPS,
                dense_family,
            )
            split_family_aulc = right_step_aulc(
                CHECKPOINT_STEPS,
                split_family,
            )
            family_aulc[family][str(seed)] = {
                "dense": float(dense_family_aulc),
                "delta": float(split_family_aulc - dense_family_aulc),
                "split90": float(split_family_aulc),
            }

    ordered_sources = [
        {
            "arm": row.arm,
            "key": row.result_key,
            "seed": row.seed,
            "sha256": row.result_sha256,
            "step": row.step,
            "version_id": row.result_version_id,
        }
        for row in sorted(
            by_cell.values(),
            key=lambda item: (
                item.seed,
                ARMS.index(item.arm),
                CHECKPOINT_STEPS.index(item.step),
            ),
        )
    ]
    return {
        "claim_limitation": CLAIM_LIMITATION,
        "cohort": {
            "arms": list(ARMS),
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "cohort_id": COHORT_ID,
            "model": "d135m",
            "model_parameters": MODEL_PARAMETERS,
            "seeds": list(SEEDS),
            "terminal_primary_step": terminal,
        },
        "decision": decision.as_dict(),
        "format": REPORT_FORMAT,
        "learnability_rule": {
            "arm_symmetric": True,
            "learnable": decision.learnable,
            "operator": "max",
            "scope": "terminal cohort-wide mean macro accuracy by arm",
            "threshold_inclusive": float(LEARNABILITY_THRESHOLD),
            "threshold_inclusive_fraction": _fraction_dict(
                LEARNABILITY_THRESHOLD
            ),
        },
        "matrix": {
            "checkpoint_count": len(by_cell),
            "checkpoints_per_run": len(CHECKPOINT_STEPS),
            "run_count": len(ARMS) * len(SEEDS),
        },
        "primary": {
            "arm_mean_accuracies": {
                arm: float(value) for arm, value in arm_means.items()
            },
            "arm_mean_accuracy_fractions": {
                arm: _fraction_dict(value) for arm, value in arm_means.items()
            },
            "bootstrap": bootstrap.as_dict(),
            "exact": exact.as_dict(),
            "seed_deltas": {
                str(seed): float(seed_deltas[index])
                for index, seed in enumerate(SEEDS)
            },
            "seed_delta_fractions": {
                str(seed): _fraction_dict(seed_deltas[index])
                for index, seed in enumerate(SEEDS)
            },
        },
        "schema_version": 1,
        "secondary_trajectory": {
            "can_affect_primary_verdict": False,
            "family_accuracy": family_accuracy,
            "family_aulc": family_aulc,
            "integration": "no_interpolation_right_step",
            "seed_macro_accuracy": seed_macro_accuracy,
            "seed_macro_aulc": seed_macro_aulc,
            "seed_macro_aulc_deltas": seed_macro_aulc_deltas,
            "seed_macro_deltas": seed_macro_deltas,
            "steps": list(CHECKPOINT_STEPS),
        },
        "source_results": ordered_sources,
        "validity": validity.as_dict(),
    }


def _replayed_checkpoint_from_result(
    result: Mapping[str, Any],
    ref: S3ObjectVersion,
    *,
    derived_validity: Mapping[str, GateEvidence],
    expected_release: Mapping[str, Any],
    expected_code: Sequence[Mapping[str, str]],
    expected_code_sha256: str,
) -> ReplayedCheckpoint:
    run = result["run"]
    checkpoint = result["checkpoint"]
    evaluator = result["evaluator"]
    if (
        result["release"] != expected_release
        or evaluator.get("code") != list(expected_code)
        or evaluator.get("code_sha256") != expected_code_sha256
        or ref.key
        != _checkpoint_result_key(
            run.get("arm"),
            run.get("seed"),
            checkpoint.get("step"),
        )
    ):
        raise ReportingError("checkpoint release/code/result authority differs")
    family_items: dict[str, list[tuple[str, bool]]] = {}
    for row in result["items"]:
        family_items.setdefault(row["task"], []).append(
            (row["item_id"], row["correct"])
        )
    pairing = run["pairing"]
    failures = tuple(
        name
        for name in REQUIRED_VALIDITY_GATES
        if not derived_validity[name].passed
    )
    return ReplayedCheckpoint(
        arm=run["arm"],
        seed=run["seed"],
        step=checkpoint["step"],
        family_items={
            family: tuple(rows) for family, rows in family_items.items()
        },
        pairing=PairingIdentity(
            initialization_sha256=pairing["initialization_sha256"],
            data_order_sha256=pairing["data_order_sha256"],
            runtime_sha256=pairing["runtime_sha256"],
            corpus_sha256=pairing["corpus_sha256"],
            config_sha256=pairing["config_sha256"],
        ),
        validity=ValiditySummary(
            passed=not failures,
            failures=failures,
        ),
        release_identity_sha256=hashlib.sha256(
            canonical_json_bytes(result["release"])
        ).hexdigest(),
        evaluator_code_sha256=evaluator["code_sha256"],
        result_key=ref.key,
        result_version_id=ref.version_id,
        result_sha256=ref.sha256,
    )


def _binding_from_result_manifest(
    result: Mapping[str, Any],
    manifest: CheckpointManifest,
) -> CheckpointBinding:
    run = result["run"]
    checkpoint = result["checkpoint"]
    pairing = run["pairing"]
    cell = manifest.cells.get((run["arm"], run["seed"], checkpoint["step"]))
    if cell is None:
        raise ReportingError("checkpoint result cell is absent from signed manifest")
    return CheckpointBinding(
        arm=run["arm"],
        seed=run["seed"],
        step=checkpoint["step"],
        sha256=checkpoint["sha256"],
        object_key=checkpoint["object_key"],
        version_id=checkpoint["version_id"],
        bytes=checkpoint["bytes"],
        run_config_path=run["config_path"],
        run_config_sha256=run["config_sha256"],
        initialization_sha256=pairing["initialization_sha256"],
        data_order_sha256=pairing["data_order_sha256"],
        paired_runtime_sha256=pairing["runtime_sha256"],
        paired_corpus_sha256=pairing["corpus_sha256"],
        paired_config_sha256=pairing["config_sha256"],
        checkpoint_kms_key_arn=cell.checkpoint.kms_key_arn or "",
        manifest_key=checkpoint["manifest_key"],
        manifest_version_id=checkpoint["manifest_version_id"],
        manifest_sha256=checkpoint["manifest_sha256"],
        evidence_key=checkpoint["evidence_key"],
        evidence_version_id=checkpoint["evidence_version_id"],
        evidence_sha256=checkpoint["evidence_sha256"],
        runtime_lock_key=checkpoint["runtime_lock_key"],
        runtime_lock_version_id=checkpoint["runtime_lock_version_id"],
        runtime_lock_object_sha256=checkpoint["runtime_lock_object_sha256"],
        corpus_receipt_key=checkpoint["corpus_receipt_key"],
        corpus_receipt_version_id=checkpoint["corpus_receipt_version_id"],
        corpus_receipt_object_sha256=checkpoint[
            "corpus_receipt_object_sha256"
        ],
    )


def _read_checkpoint_results(
    authority,
    contract,
    public_release: Mapping[str, Any],
    sealed_release: Mapping[str, Any],
    release: ReleaseBinding,
) -> list[ReplayedCheckpoint]:
    code_bindings = _task3_code_bindings(contract.evaluator_code)
    code = [asdict(binding) for binding in code_bindings]
    code_sha256 = hashlib.sha256(canonical_json_bytes(code)).hexdigest()
    expected_release = _release_dict(release)
    manifest = _load_signed_manifest(authority, release)
    results = []
    for seed in SEEDS:
        for arm in ARMS:
            for step in CHECKPOINT_STEPS:
                payload, ref = authority.read_checkpoint_result(
                    arm,
                    seed,
                    step,
                    release.authority,
                )
                result = _parse_result_bytes(payload)
                _validate_checkpoint_result(
                    result,
                    public_release=public_release,
                    sealed_release=sealed_release,
                    family_order=contract.family_names,
                    items_per_family=contract.accepted_items_per_family,
                    scorer_id=contract.scorer_id,
                )
                binding = _binding_from_result_manifest(result, manifest)
                try:
                    derived_validity = _replay_manifest_validity(
                        authority,
                        release,
                        manifest,
                        binding,
                        expected_item_count=contract.accepted_items_per_family
                        * len(contract.family_names),
                    )
                except Exception as error:
                    raise ReportingError(
                        "checkpoint raw validity evidence replay failed"
                    ) from error
                if result["validity"] != _validate_gate_evidence(
                    derived_validity
                ):
                    raise ReportingError(
                        "checkpoint derived validity differs from raw evidence"
                    )
                results.append(
                    _replayed_checkpoint_from_result(
                        result,
                        ref,
                        derived_validity=derived_validity,
                        expected_release=expected_release,
                        expected_code=code,
                        expected_code_sha256=code_sha256,
                    )
                )
    return results


def _parse_result_bytes(payload: bytes) -> Mapping[str, Any]:
    from evals.reasoning_v3.sealing import _parse_canonical_bytes

    return _parse_canonical_bytes(payload, "checkpoint evaluation result")


def _publish_scientific_report(
    report: Mapping[str, Any],
    verified: VerifiedAwsAuthority,
    authority,
) -> PublishedScientificReport:
    payload = canonical_json_bytes(report)
    publication = authority.put_scientific_report(payload, verified)
    if publication.payload_bytes != payload:
        raise ReportingError("published scientific report payload differs")
    return PublishedScientificReport(
        report=report,
        object_ref=publication.object_ref,
    )


def run_frozen_scientific_inference() -> PublishedScientificReport:
    """Read all exact S3 versions, replay raw rows, and publish one fixed report."""

    authority, contract, public, sealed, release = _load_fixed_release()
    checkpoints = _read_checkpoint_results(
        authority,
        contract,
        public,
        sealed,
        release,
    )
    report = _build_scientific_report(
        checkpoints,
        family_order=contract.family_names,
        items_per_family=contract.accepted_items_per_family,
    )
    return _publish_scientific_report(
        report,
        release.authority,
        authority,
    )


__all__ = [
    "CLAIM_LIMITATION",
    "LEARNABILITY_THRESHOLD",
    "PublishedScientificReport",
    "ReportingError",
    "assign_frozen_verdict",
    "run_frozen_scientific_inference",
]
