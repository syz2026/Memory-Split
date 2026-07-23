"""Frozen orthogonal confirmatory status axes and classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ScientificStatus(StrEnum):
    INCOMPLETE = "incomplete"
    INVALID = "invalid"
    COMPLETE = "complete"


class InterimEvidenceLabel(StrEnum):
    NONE = "none"
    DIRECTIONAL_ONLY = "directional_only"
    SIGN_CONSISTENT_ONLY = "sign_consistent_only"


class FinalInferenceConclusion(StrEnum):
    NOT_EVALUATED = "not_evaluated"
    INCONCLUSIVE = "inconclusive"
    SUPPORTS_EFFECT = "supports_effect"
    SUPPORTS_PRACTICAL_NULL = "supports_practical_null"


def _enum(value: object, enum_type: type[StrEnum], name: str) -> StrEnum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not an approved value") from exc


@dataclass(frozen=True)
class StatusAxes:
    scientific_status: ScientificStatus
    interim_evidence_label: InterimEvidenceLabel
    final_inference_conclusion: FinalInferenceConclusion

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scientific_status",
            _enum(
                self.scientific_status,
                ScientificStatus,
                "scientific_status",
            ),
        )
        object.__setattr__(
            self,
            "interim_evidence_label",
            _enum(
                self.interim_evidence_label,
                InterimEvidenceLabel,
                "interim_evidence_label",
            ),
        )
        object.__setattr__(
            self,
            "final_inference_conclusion",
            _enum(
                self.final_inference_conclusion,
                FinalInferenceConclusion,
                "final_inference_conclusion",
            ),
        )
        if self.scientific_status is ScientificStatus.COMPLETE:
            if self.interim_evidence_label is not InterimEvidenceLabel.NONE:
                raise ValueError("complete evidence cannot retain an interim label")
            if (
                self.final_inference_conclusion
                is FinalInferenceConclusion.NOT_EVALUATED
            ):
                raise ValueError("complete evidence requires a final conclusion")
        elif (
            self.final_inference_conclusion
            is not FinalInferenceConclusion.NOT_EVALUATED
        ):
            raise ValueError(
                "non-complete evidence requires final conclusion not_evaluated"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "scientific_status": self.scientific_status.value,
            "interim_evidence_label": self.interim_evidence_label.value,
            "final_inference_conclusion": self.final_inference_conclusion.value,
        }


def classify_status(
    *,
    complete: bool,
    valid: bool,
    observed_seeds: int,
    required_seeds: int,
    sign_consistent: bool,
    supports_effect: bool,
    supports_practical_null: bool,
) -> StatusAxes:
    """Derive all three frozen status axes from measured study evidence."""

    for name, value in (
        ("complete", complete),
        ("valid", valid),
        ("sign_consistent", sign_consistent),
        ("supports_effect", supports_effect),
        ("supports_practical_null", supports_practical_null),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be Boolean")
    if (
        isinstance(observed_seeds, bool)
        or not isinstance(observed_seeds, int)
        or observed_seeds < 0
    ):
        raise ValueError("observed_seeds must be a non-negative integer")
    if (
        isinstance(required_seeds, bool)
        or not isinstance(required_seeds, int)
        or required_seeds < 1
    ):
        raise ValueError("required_seeds must be a positive integer")
    if observed_seeds > required_seeds:
        raise ValueError("observed_seeds cannot exceed required_seeds")
    if supports_effect and supports_practical_null:
        raise ValueError("effect and practical-null conclusions are exclusive")

    if not valid:
        scientific = ScientificStatus.INVALID
    elif not complete or observed_seeds < required_seeds:
        scientific = ScientificStatus.INCOMPLETE
    else:
        scientific = ScientificStatus.COMPLETE

    if (
        scientific in {ScientificStatus.COMPLETE, ScientificStatus.INVALID}
        or observed_seeds == 0
    ):
        interim = InterimEvidenceLabel.NONE
    elif sign_consistent and observed_seeds >= 3:
        interim = InterimEvidenceLabel.SIGN_CONSISTENT_ONLY
    else:
        interim = InterimEvidenceLabel.DIRECTIONAL_ONLY

    if scientific is not ScientificStatus.COMPLETE:
        conclusion = FinalInferenceConclusion.NOT_EVALUATED
    elif supports_effect:
        conclusion = FinalInferenceConclusion.SUPPORTS_EFFECT
    elif supports_practical_null:
        conclusion = FinalInferenceConclusion.SUPPORTS_PRACTICAL_NULL
    else:
        conclusion = FinalInferenceConclusion.INCONCLUSIVE

    return StatusAxes(scientific, interim, conclusion)
