"""Explicit frozen seed/provider locks for confirmatory replay."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from msctl.cohort import (
    COHORT_ID,
    MODEL_PARAMETERS,
    SEEDS,
    load_cohort_assignment,
)


LEGACY_360M_SEEDS = tuple(range(5))


@dataclass(frozen=True)
class StudyLock:
    cohort_id: str
    model_parameters: int
    seeds: tuple[int, ...]
    providers: dict[str, tuple[int, ...]]
    terminal_n_pairs: int
    primary_alternative: str
    primary_alpha: float
    bootstrap_resamples: int
    bootstrap_rng_seed: int
    practical_null_margin: float
    preregistration_sha256: str | None = None
    assignment_sha256: str | None = None

    @classmethod
    def legacy_360m(cls) -> "StudyLock":
        """Reproduce the historical N=5 lock without changing its seed set."""

        return cls(
            cohort_id="memorysplit-confirmatory-v2-360m-n5",
            model_parameters=356_033_536,
            seeds=LEGACY_360M_SEEDS,
            providers={"legacy-protected": LEGACY_360M_SEEDS},
            terminal_n_pairs=5,
            primary_alternative="two-sided",
            primary_alpha=0.05,
            bootstrap_resamples=20_000,
            bootstrap_rng_seed=360_005,
            practical_null_margin=0.01,
        )

    def validate_observed_seeds(
        self,
        observed: set[int] | tuple[int, ...] | list[int],
    ) -> tuple[bool, tuple[int, ...], tuple[int, ...]]:
        values = set(observed)
        missing = tuple(sorted(set(self.seeds) - values))
        unexpected = tuple(sorted(values - set(self.seeds)))
        return not missing and not unexpected, missing, unexpected

    def provider_for_seed(self, seed: int) -> str:
        providers = [
            provider
            for provider, seeds in self.providers.items()
            if seed in seeds
        ]
        if len(providers) != 1:
            raise ValueError(f"seed {seed} does not have one frozen provider")
        return providers[0]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_135m_study_lock(repository_root: Path | str) -> StudyLock:
    root = Path(repository_root)
    prereg_path = root / "configs" / "preregistration-135m-v1.yaml"
    assignment_path = root / "configs" / "cohort-assignment-135m-n10.json"
    if not prereg_path.is_file() or prereg_path.is_symlink():
        raise ValueError("135M preregistration is missing or unsafe")
    try:
        prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("135M preregistration must be valid UTF-8 YAML") from error
    if not isinstance(prereg, dict):
        raise ValueError("135M preregistration must be a mapping")
    assignment = load_cohort_assignment(assignment_path)
    protected = prereg.get("protected_cohort")
    analysis = prereg.get("analysis")
    if (
        prereg.get("contract_status") != "frozen"
        or prereg.get("cohort_id") != COHORT_ID
        or not isinstance(protected, dict)
        or not isinstance(analysis, dict)
    ):
        raise ValueError("135M preregistration is not the frozen contract")
    primary = analysis.get("primary")
    bootstrap = analysis.get("bootstrap")
    practical = analysis.get("practical_equivalence")
    if not all(isinstance(item, dict) for item in (primary, bootstrap, practical)):
        raise ValueError("135M preregistration analysis lock is incomplete")
    providers = {
        provider: tuple(seeds)
        for provider, seeds in assignment["provider_seeds"].items()
    }
    lock = StudyLock(
        cohort_id=COHORT_ID,
        model_parameters=protected["model_parameters"],
        seeds=tuple(protected["seeds"]),
        providers=providers,
        terminal_n_pairs=protected["terminal_n_pairs"],
        primary_alternative=primary["alternative"],
        primary_alpha=float(primary["alpha"]),
        bootstrap_resamples=bootstrap["n_resamples"],
        bootstrap_rng_seed=bootstrap["rng_seed"],
        practical_null_margin=float(
            practical["margin_absolute_pair_accuracy"]
        ),
        preregistration_sha256=_sha(prereg_path),
        assignment_sha256=_sha(assignment_path),
    )
    if (
        lock.model_parameters != MODEL_PARAMETERS
        or lock.seeds != SEEDS
        or lock.terminal_n_pairs != 10
        or lock.primary_alternative != "greater"
        or lock.primary_alpha != 0.05
        or lock.bootstrap_resamples != 20_000
        or lock.practical_null_margin != 0.01
    ):
        raise ValueError("135M study lock differs from the protected constants")
    owned = [seed for seeds in lock.providers.values() for seed in seeds]
    if sorted(owned) != list(SEEDS) or len(owned) != len(set(owned)):
        raise ValueError("135M provider assignment is incomplete or overlapping")
    return lock
