"""Semantic occurrence closure over every supervised text field."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticFact:
    fact_id: str
    surfaces: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.fact_id:
            raise ValueError("semantic fact_id must be non-empty")
        surfaces = tuple(self.surfaces)
        if not surfaces:
            raise ValueError("semantic facts require at least one surface")
        if any(not isinstance(surface, str) or not surface for surface in surfaces):
            raise ValueError("semantic surfaces must be non-empty strings")
        if len(surfaces) != len(set(surfaces)):
            raise ValueError("semantic surfaces must be distinct")
        object.__setattr__(self, "surfaces", surfaces)


@dataclass(frozen=True)
class SupervisedField:
    field_id: str
    text: str
    supervised: bool = True

    def __post_init__(self) -> None:
        if not self.field_id:
            raise ValueError("field_id must be non-empty")
        if not isinstance(self.text, str):
            raise TypeError("field text must be a string")
        if not isinstance(self.supervised, bool):
            raise TypeError("supervised must be boolean")


@dataclass(frozen=True, order=True)
class SemanticOccurrence:
    field_id: str
    start: int
    end: int
    fact_id: str
    surface: str

    def as_dict(self) -> dict:
        return {
            "field_id": self.field_id,
            "start": self.start,
            "end": self.end,
            "fact_id": self.fact_id,
            "surface": self.surface,
        }


@dataclass(frozen=True)
class ClosureSlice:
    field_id: str
    start: int
    end: int
    text: str
    fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class OccurrenceClosurePlan:
    facts: tuple[SemanticFact, ...]
    fields: tuple[SupervisedField, ...]
    occurrences: tuple[SemanticOccurrence, ...]

    def mask_for_routes(
        self,
        routed_fact_ids: Iterable[str],
    ) -> dict[str, tuple[int, ...]]:
        routed = frozenset(routed_fact_ids)
        known = {fact.fact_id for fact in self.facts}
        unknown = routed - known
        if unknown:
            raise ValueError(
                f"routed facts lack semantic metadata: {sorted(unknown)}"
            )
        masks = {
            field.field_id: [1] * len(field.text)
            for field in self.fields
        }
        for occurrence in self.occurrences:
            if occurrence.fact_id in routed:
                masks[occurrence.field_id][occurrence.start : occurrence.end] = (
                    [0] * (occurrence.end - occurrence.start)
                )
        return {field_id: tuple(mask) for field_id, mask in masks.items()}

    def partition_field(self, field_id: str) -> tuple[ClosureSlice, ...]:
        try:
            field = next(field for field in self.fields if field.field_id == field_id)
        except StopIteration as error:
            raise KeyError(field_id) from error
        relevant = tuple(
            occurrence
            for occurrence in self.occurrences
            if occurrence.field_id == field_id
        )
        boundaries = {0, len(field.text)}
        for occurrence in relevant:
            boundaries.update((occurrence.start, occurrence.end))
        ordered = sorted(boundaries)
        slices = []
        for start, end in zip(ordered, ordered[1:]):
            if start == end:
                continue
            fact_ids = tuple(
                sorted(
                    {
                        occurrence.fact_id
                        for occurrence in relevant
                        if occurrence.start <= start and end <= occurrence.end
                    }
                )
            )
            slices.append(
                ClosureSlice(
                    field_id=field_id,
                    start=start,
                    end=end,
                    text=field.text[start:end],
                    fact_ids=fact_ids,
                )
            )
        return tuple(slices)


def _find_all(text: str, surface: str):
    start = 0
    while True:
        index = text.find(surface, start)
        if index < 0:
            return
        yield index
        start = index + 1


def plan_occurrence_closure(
    facts: Iterable[SemanticFact],
    fields: Iterable[SupervisedField],
) -> OccurrenceClosurePlan:
    fact_rows = tuple(facts)
    field_rows = tuple(fields)
    if any(not isinstance(fact, SemanticFact) for fact in fact_rows):
        raise TypeError("facts must contain SemanticFact rows")
    if any(not isinstance(field, SupervisedField) for field in field_rows):
        raise TypeError("fields must contain SupervisedField rows")
    fact_ids = [fact.fact_id for fact in fact_rows]
    field_ids = [field.field_id for field in field_rows]
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError("semantic fact ids must be unique")
    if len(field_ids) != len(set(field_ids)):
        raise ValueError("supervised field ids must be unique")

    occurrences = []
    for field in field_rows:
        if not field.supervised:
            continue
        for fact in fact_rows:
            for surface in fact.surfaces:
                occurrences.extend(
                    SemanticOccurrence(
                        field_id=field.field_id,
                        start=start,
                        end=start + len(surface),
                        fact_id=fact.fact_id,
                        surface=surface,
                    )
                    for start in _find_all(field.text, surface)
                )
    return OccurrenceClosurePlan(
        facts=fact_rows,
        fields=field_rows,
        occurrences=tuple(sorted(occurrences)),
    )


@dataclass(frozen=True)
class LeakageReport:
    routed_fact_ids: tuple[str, ...]
    supervised_occurrences: int
    masked_occurrences: int
    unmasked_occurrences: tuple[SemanticOccurrence, ...]
    metadata_errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.unmasked_occurrences and not self.metadata_errors

    def as_dict(self) -> dict:
        return {
            "format": "memorysplit-semantic-leakage-report-v2",
            "passed": self.passed,
            "routed_fact_ids": list(self.routed_fact_ids),
            "supervised_occurrences": self.supervised_occurrences,
            "masked_occurrences": self.masked_occurrences,
            "unmasked_supervised_occurrences": len(self.unmasked_occurrences),
            "leaks": [
                occurrence.as_dict()
                for occurrence in self.unmasked_occurrences
            ],
            "metadata_errors": list(self.metadata_errors),
        }


class SemanticLeakageError(ValueError):
    def __init__(self, report: LeakageReport) -> None:
        self.report = report
        super().__init__(
            "semantic occurrence closure failed: "
            f"{len(report.unmasked_occurrences)} unmasked occurrences, "
            f"{len(report.metadata_errors)} metadata errors"
        )


def audit_occurrence_closure(
    facts: Iterable[SemanticFact],
    fields: Iterable[SupervisedField],
    routed_fact_ids: Iterable[str],
    masks: Mapping[str, Sequence[int]],
    *,
    fail_closed: bool = True,
) -> LeakageReport:
    """Rescan fields and prove every routed supervised occurrence is masked."""

    fact_rows = tuple(facts)
    field_rows = tuple(fields)
    routed = tuple(sorted(set(routed_fact_ids)))
    plan = plan_occurrence_closure(fact_rows, field_rows)
    known = {fact.fact_id for fact in fact_rows}
    errors = [
        f"routed fact lacks semantic metadata: {fact_id}"
        for fact_id in routed
        if fact_id not in known
    ]
    expected_fields = {field.field_id: field for field in field_rows}
    for field_id in sorted(set(masks) - set(expected_fields)):
        errors.append(f"mask references unknown field: {field_id}")
    normalized_masks: dict[str, tuple[int, ...]] = {}
    for field_id, field in expected_fields.items():
        if field_id not in masks:
            errors.append(f"mask missing field: {field_id}")
            continue
        value = tuple(masks[field_id])
        if len(value) != len(field.text):
            errors.append(f"mask length mismatch for field: {field_id}")
            continue
        if any(item not in (0, 1) for item in value):
            errors.append(f"mask is not binary for field: {field_id}")
            continue
        normalized_masks[field_id] = value

    selected = tuple(
        occurrence
        for occurrence in plan.occurrences
        if occurrence.fact_id in routed
    )
    masked = 0
    leaked = []
    for occurrence in selected:
        mask = normalized_masks.get(occurrence.field_id)
        if (
            mask is not None
            and all(value == 0 for value in mask[occurrence.start : occurrence.end])
        ):
            masked += 1
        else:
            leaked.append(occurrence)
    report = LeakageReport(
        routed_fact_ids=routed,
        supervised_occurrences=len(selected),
        masked_occurrences=masked,
        unmasked_occurrences=tuple(leaked),
        metadata_errors=tuple(errors),
    )
    if fail_closed and not report.passed:
        raise SemanticLeakageError(report)
    return report
