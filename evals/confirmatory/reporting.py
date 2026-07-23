"""Frozen verdict assignment and confirmatory report construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from evals.confirmatory.inference import (
    BootstrapEstimate,
    ExactTestResult,
    PairedObservation,
    exact_sign_flip_test,
    hierarchical_paired_bootstrap,
    holm_adjust,
    practical_equivalence,
    provider_mean_deltas,
    seed_mean_differences,
)
from evals.confirmatory.study_lock import StudyLock


@dataclass(frozen=True)
class ValidityGates:
    protocol_valid: bool = True
    provenance_valid: bool = True
    evaluation_valid: bool = True
    failures: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return (
            self.protocol_valid
            and self.provenance_valid
            and self.evaluation_valid
            and not self.failures
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["passed"] = self.passed
        result["failures"] = list(self.failures)
        return result


@dataclass(frozen=True)
class ConfirmatoryEvidence:
    cohort_complete: bool
    validity: ValidityGates
    primary_exact: ExactTestResult
    primary_interval_95: BootstrapEstimate
    equivalence_interval_90: BootstrapEstimate
    provider_means: dict[str, float]
    family_tests: dict[str, dict[str, float | bool]]


@dataclass(frozen=True)
class VerdictDecision:
    verdict: str
    supports_effect: bool
    supports_practical_null: bool
    broader_family_claim: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


def assign_verdict(
    evidence: ConfirmatoryEvidence,
    *,
    alpha: float = 0.05,
    practical_margin: float = 0.01,
) -> VerdictDecision:
    """Apply the mutually exclusive preregistered effect/null/other rules."""

    if not evidence.validity.passed:
        return VerdictDecision(
            verdict="invalid",
            supports_effect=False,
            supports_practical_null=False,
            broader_family_claim=False,
            reasons=tuple(evidence.validity.failures)
            or ("one or more measured validity gates failed",),
        )
    if not evidence.cohort_complete:
        return VerdictDecision(
            verdict="inconclusive",
            supports_effect=False,
            supports_practical_null=False,
            broader_family_claim=False,
            reasons=("protected cohort is incomplete",),
        )
    platform_passed = bool(evidence.provider_means) and all(
        value > 0 for value in evidence.provider_means.values()
    )
    effect = (
        evidence.primary_exact.p_value <= alpha
        and evidence.primary_interval_95.lower > 0
        and platform_passed
    )
    equivalent = practical_equivalence(
        evidence.equivalence_interval_90,
        margin=practical_margin,
    ).equivalent
    family_complete = (
        set(evidence.family_tests) == {"graph", "non_path"}
        and all(
            item.get("reject") is True
            for item in evidence.family_tests.values()
        )
    )
    if effect:
        return VerdictDecision(
            verdict="supports_effect",
            supports_effect=True,
            supports_practical_null=False,
            broader_family_claim=family_complete,
            reasons=(
                "primary exact test and 95% interval pass",
                "all provider-specific mean deltas are positive",
            ),
        )
    if equivalent:
        return VerdictDecision(
            verdict="supports_practical_null",
            supports_effect=False,
            supports_practical_null=True,
            broader_family_claim=False,
            reasons=(
                "frozen 90% interval lies strictly inside the practical margin",
            ),
        )
    reasons = ["neither protected claim rule passed"]
    if not platform_passed:
        reasons.append("provider-specific positive-mean guardrail did not pass")
    return VerdictDecision(
        verdict="inconclusive",
        supports_effect=False,
        supports_practical_null=False,
        broader_family_claim=False,
        reasons=tuple(reasons),
    )


def _with_protocol_failure(validity: ValidityGates, failure: str) -> ValidityGates:
    return replace(
        validity,
        protocol_valid=False,
        failures=tuple(validity.failures) + (failure,),
    )


def build_confirmatory_report(
    observations: list[PairedObservation],
    *,
    study_lock: StudyLock,
    validity: ValidityGates,
    n_resamples: int | None = None,
    rng_seed: int | None = None,
    family_seed_differences: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Run the frozen analysis against an explicit study seed/provider lock."""

    observed_seeds = {row.seed for row in observations}
    complete, missing, unexpected = study_lock.validate_observed_seeds(
        observed_seeds
    )
    if unexpected:
        validity = _with_protocol_failure(
            validity,
            f"unexpected protected seeds: {list(unexpected)}",
        )
    provider_mismatches = sorted(
        {
            row.seed
            for row in observations
            if row.seed in study_lock.seeds
            and row.provider != study_lock.provider_for_seed(row.seed)
        }
    )
    if provider_mismatches:
        validity = _with_protocol_failure(
            validity,
            f"provider assignment mismatch for seeds: {provider_mismatches}",
        )
    if not complete or not validity.passed:
        verdict = "invalid" if not validity.passed else "inconclusive"
        reasons = (
            list(validity.failures)
            if verdict == "invalid"
            else [f"protected cohort is incomplete; missing={list(missing)}"]
        )
        return {
            "cohort_complete": complete,
            "cohort_id": study_lock.cohort_id,
            "decision": {
                "broader_family_claim": False,
                "reasons": reasons,
                "supports_effect": False,
                "supports_practical_null": False,
                "verdict": verdict,
            },
            "families": {},
            "primary": None,
            "protected_replay": False,
            "schema_version": 1,
            "seeds": sorted(observed_seeds),
            "validity": validity.as_dict(),
        }

    seed_estimates = seed_mean_differences(
        observations,
        expected_seeds=study_lock.seeds,
    )
    differences = [seed_estimates[seed] for seed in study_lock.seeds]
    exact = exact_sign_flip_test(
        differences,
        alternative=study_lock.primary_alternative,
    )
    draws = study_lock.bootstrap_resamples if n_resamples is None else n_resamples
    random_seed = study_lock.bootstrap_rng_seed if rng_seed is None else rng_seed
    interval_95 = hierarchical_paired_bootstrap(
        observations,
        expected_seeds=study_lock.seeds,
        n_resamples=draws,
        rng_seed=random_seed,
        confidence=0.95,
    )
    interval_90 = hierarchical_paired_bootstrap(
        observations,
        expected_seeds=study_lock.seeds,
        n_resamples=draws,
        rng_seed=random_seed + 1,
        confidence=0.90,
    )
    family_results = {}
    if family_seed_differences is not None:
        if set(family_seed_differences) != {"graph", "non_path"}:
            raise ValueError("family tests must contain graph and non_path")
        raw_family = {}
        exact_family = {}
        for name, values in family_seed_differences.items():
            if len(values) != len(study_lock.seeds):
                raise ValueError(f"{name} family must contain one value per seed")
            result = exact_sign_flip_test(
                values,
                alternative=study_lock.primary_alternative,
            )
            raw_family[name] = result.p_value
            exact_family[name] = result
        adjusted = holm_adjust(raw_family, alpha=study_lock.primary_alpha)
        family_results = {
            name: {
                **adjusted[name],
                "statistic": exact_family[name].statistic,
            }
            for name in ("graph", "non_path")
        }
    evidence = ConfirmatoryEvidence(
        cohort_complete=complete,
        validity=validity,
        primary_exact=exact,
        primary_interval_95=interval_95,
        equivalence_interval_90=interval_90,
        provider_means=provider_mean_deltas(
            observations,
            expected_seeds=study_lock.seeds,
        ),
        family_tests=family_results,
    )
    decision = assign_verdict(
        evidence,
        alpha=study_lock.primary_alpha,
        practical_margin=study_lock.practical_null_margin,
    )
    return {
        "cohort_complete": complete,
        "cohort_id": study_lock.cohort_id,
        "decision": decision.as_dict(),
        "families": family_results,
        "primary": {
            "equivalence_90": {
                **interval_90.as_dict(),
                **practical_equivalence(
                    interval_90,
                    margin=study_lock.practical_null_margin,
                ).as_dict(),
            },
            "exact": exact.as_dict(),
            "interval_95": interval_95.as_dict(),
            "provider_means": evidence.provider_means,
            "seed_differences": {
                str(seed): seed_estimates[seed]
                for seed in study_lock.seeds
            },
        },
        "protected_replay": (
            draws == study_lock.bootstrap_resamples
            and random_seed == study_lock.bootstrap_rng_seed
        ),
        "schema_version": 1,
        "seeds": list(study_lock.seeds),
        "validity": validity.as_dict(),
    }
