"""Small finite-domain solvers with canonical, replay-verifiable proofs."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TypeVar


_T = TypeVar("_T")
PROOF_FORMAT = "memorysplit-canonical-proof-v1"
SOLVER_ID = "finite-domain-enumeration-v1"


class ProofSolveError(ValueError):
    pass


def _unique_solution(
    domain: Iterable[_T],
    constraint: Callable[[_T], bool],
) -> _T:
    solutions = tuple(candidate for candidate in domain if constraint(candidate))
    if len(solutions) != 1:
        raise ProofSolveError(
            f"constraint must have exactly one solution, found {len(solutions)}"
        )
    return solutions[0]


@dataclass(frozen=True, order=True)
class CompositionPremise:
    fact_id: str
    hop: int
    compose_code: int

    def __post_init__(self) -> None:
        if not isinstance(self.fact_id, str):
            raise TypeError("composition fact_id must be a string")
        if not self.fact_id:
            raise ValueError("composition fact_id must be non-empty")
        if isinstance(self.hop, bool) or not isinstance(self.hop, int) or self.hop < 0:
            raise ValueError("hop must be a non-negative integer")
        if (
            isinstance(self.compose_code, bool)
            or not isinstance(self.compose_code, int)
            or self.compose_code not in range(4)
        ):
            raise ValueError("compose_code must be in [0, 3]")


@dataclass(frozen=True, order=True)
class EqualityPremise:
    fact_id: str
    slot: int
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.fact_id, str):
            raise TypeError("equality fact_id must be a string")
        if not self.fact_id:
            raise ValueError("equality fact_id must be non-empty")
        if (
            isinstance(self.slot, bool)
            or not isinstance(self.slot, int)
            or self.slot not in range(4)
        ):
            raise ValueError("slot must be in [0, 3]")
        if not isinstance(self.value, str):
            raise TypeError("equality value must be a string")
        if not self.value:
            raise ValueError("equality value must be a non-empty string")


@dataclass(frozen=True)
class ProofStep:
    premise_id: str
    rule: str
    result: str

    def as_dict(self) -> dict:
        return {
            "premise_id": self.premise_id,
            "rule": self.rule,
            "result": self.result,
        }


@dataclass(frozen=True)
class ProofObject:
    family: str
    premise_ids: tuple[str, ...]
    steps: tuple[ProofStep, ...]
    conclusion: tuple[tuple[str, object], ...]
    format: str = PROOF_FORMAT
    solver: str = SOLVER_ID

    def as_dict(self) -> dict:
        return {
            "format": self.format,
            "family": self.family,
            "solver": self.solver,
            "premise_ids": list(self.premise_ids),
            "steps": [step.as_dict() for step in self.steps],
            "conclusion": dict(self.conclusion),
        }

    def to_bytes(self) -> bytes:
        return (
            json.dumps(
                self.as_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )


def _composition_rows(
    premises: Iterable[CompositionPremise],
) -> tuple[CompositionPremise, ...]:
    rows = tuple(premises)
    if not rows:
        raise ValueError("graph composition requires at least one premise")
    if any(not isinstance(row, CompositionPremise) for row in rows):
        raise TypeError("composition premises have the wrong type")
    rows = tuple(sorted(rows, key=lambda row: (row.hop, row.fact_id)))
    if [row.hop for row in rows] != list(range(len(rows))):
        raise ValueError("composition hops must be unique and contiguous from zero")
    ids = [row.fact_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("composition premise ids must be unique")
    return rows


def solve_graph_composition(
    premises: Iterable[CompositionPremise],
) -> ProofObject:
    rows = _composition_rows(premises)
    expected = sum(row.compose_code for row in rows) % 4
    solution = _unique_solution(
        range(4),
        lambda candidate: candidate == expected,
    )
    accumulator = 0
    steps = []
    for row in rows:
        accumulator = (accumulator + row.compose_code) % 4
        steps.append(
            ProofStep(
                premise_id=row.fact_id,
                rule="accumulate_mod4",
                result=f"r{accumulator}",
            )
        )
    if accumulator != solution:
        raise AssertionError("composition solver and proof replay disagree")
    return ProofObject(
        family="graph_composition_mod4",
        premise_ids=tuple(row.fact_id for row in rows),
        steps=tuple(steps),
        conclusion=(("relation", f"r{solution}"),),
    )


def _equality_rows(
    premises: Iterable[EqualityPremise],
) -> tuple[EqualityPremise, EqualityPremise]:
    rows = tuple(premises)
    if len(rows) != 2:
        raise ValueError("slot equality requires exactly two premises")
    if any(not isinstance(row, EqualityPremise) for row in rows):
        raise TypeError("equality premises have the wrong type")
    rows = tuple(sorted(rows, key=lambda row: (row.slot, row.fact_id)))
    if rows[0].slot == rows[1].slot:
        raise ValueError("slot equality requires two distinct slots")
    if rows[0].fact_id == rows[1].fact_id:
        raise ValueError("equality premise ids must be unique")
    return rows[0], rows[1]


def solve_slot_equality(
    premises: Iterable[EqualityPremise],
) -> ProofObject:
    left, right = _equality_rows(premises)
    expected = left.value == right.value
    solution = _unique_solution(
        (False, True),
        lambda candidate: candidate is expected,
    )
    return ProofObject(
        family="slot_equality",
        premise_ids=(left.fact_id, right.fact_id),
        steps=(
            ProofStep(
                premise_id=left.fact_id,
                rule="compare_slots",
                result="equal" if solution else "not_equal",
            ),
            ProofStep(
                premise_id=right.fact_id,
                rule="compare_slots",
                result="equal" if solution else "not_equal",
            ),
        ),
        conclusion=(("equal", solution),),
    )


def verify_proof(
    proof: ProofObject,
    premises: Iterable[CompositionPremise] | Iterable[EqualityPremise],
) -> bool:
    """Deterministically re-solve premises and require canonical byte identity."""

    if not isinstance(proof, ProofObject):
        return False
    rows = tuple(premises)
    try:
        if proof.family == "graph_composition_mod4":
            expected = solve_graph_composition(rows)
        elif proof.family == "slot_equality":
            expected = solve_slot_equality(rows)
        else:
            return False
    except (TypeError, ValueError):
        return False
    return proof.to_bytes() == expected.to_bytes()
