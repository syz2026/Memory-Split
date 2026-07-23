"""Fail-closed proof and answer verification against trusted solvers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from evals.confirmatory.actions import (
    ActionOp,
    ActionSlot,
    validate_action_slots,
)
from evals.confirmatory.contracts import (
    ItemRecord,
    SealedGoldRecord,
    StoreRecord,
)


class ProofSolver(Protocol):
    solver_id: str

    def solve(
        self,
        item: ItemRecord,
        store: StoreRecord,
        proof: tuple[ActionSlot, ...],
    ) -> str: ...


class SolverError(ValueError):
    """A proof could not be executed under the trusted solver."""


@dataclass(frozen=True)
class ProofAnswerVerification:
    valid: bool
    proof_valid: bool
    answer_valid: bool
    derived_answer: str | None
    reason: str | None


class LookupChainSolver:
    """Execute exact reads and use the final returned target as the answer."""

    solver_id = "lookup-chain-v1"

    def solve(
        self,
        item: ItemRecord,
        store: StoreRecord,
        proof: tuple[ActionSlot, ...],
    ) -> str:
        actions = validate_action_slots(proof)
        if item.store_id != store.store_id or item.world_id != store.world_id:
            raise SolverError("item/store binding mismatch")
        slots = list(item.initial_slots)
        returned: list[str] = []
        halted = False
        for action in actions:
            if action.op is ActionOp.HALT:
                halted = True
                break
            if action.op is not ActionOp.READ:
                continue
            source_slot = action.source_slot
            if source_slot is None:
                raise SolverError("read has no source slot")
            source_id = slots[source_slot]
            if source_id is None:
                raise SolverError(f"source slot {source_slot} is unbound")
            row = store.lookup(
                source_id,
                str(action.relation_id),
                str(action.direction),
            )
            if row is None:
                raise SolverError(
                    "MISS at "
                    f"({source_id}, {action.relation_id}, {action.direction})"
                )
            returned.append(row.target)
            if row.target_kind == "entity":
                slots[source_slot] = row.target
        if not halted:
            raise SolverError("proof does not HALT")
        if not returned:
            raise SolverError("proof contains no successful read")
        return returned[-1]


def _binding_error(
    item: ItemRecord,
    store: StoreRecord,
    gold: SealedGoldRecord,
    solver: ProofSolver,
) -> str | None:
    if not isinstance(item, ItemRecord):
        return "item must be a validated ItemRecord"
    if not isinstance(store, StoreRecord):
        return "store must be a validated StoreRecord"
    if not isinstance(gold, SealedGoldRecord):
        return "gold must be a validated SealedGoldRecord"
    if (
        item.item_id,
        item.pair_id,
        item.twin,
    ) != (
        gold.item_id,
        gold.pair_id,
        gold.twin,
    ):
        return "item/sealed-gold identity mismatch"
    if item.store_id != store.store_id or item.world_id != store.world_id:
        return "item/store identity mismatch"
    if gold.store_sha256 != store.content_sha256:
        return "sealed-gold store hash mismatch"
    solver_id = getattr(solver, "solver_id", None)
    if solver_id != gold.solver_id:
        return "solver_id does not match sealed gold"
    if not callable(getattr(solver, "solve", None)):
        return "solver must expose solve"
    return None


def verify_proof_and_answer(
    *,
    item: ItemRecord,
    store: StoreRecord,
    gold: SealedGoldRecord,
    proof: Sequence[ActionSlot | Mapping[str, Any]],
    answer: str,
    solver: ProofSolver,
) -> ProofAnswerVerification:
    """Verify a model proof with no gold passed into the solver itself."""

    try:
        binding_error = _binding_error(item, store, gold, solver)
    except Exception as exc:
        return ProofAnswerVerification(
            False,
            False,
            False,
            None,
            f"solver binding failure: {exc}",
        )
    if binding_error is not None:
        return ProofAnswerVerification(False, False, False, None, binding_error)
    if not isinstance(answer, str):
        return ProofAnswerVerification(
            False,
            False,
            False,
            None,
            "answer must be a string",
        )
    try:
        actions = validate_action_slots(proof)
        derived = solver.solve(item, store, actions)
        if not isinstance(derived, str):
            raise SolverError("solver answer must be a string")
    except Exception as exc:
        return ProofAnswerVerification(False, False, False, None, str(exc))

    proof_valid = derived == gold.answer
    answer_valid = answer == gold.answer
    if not proof_valid:
        reason = "solver-derived answer disagrees with sealed gold"
    elif not answer_valid:
        reason = "submitted answer disagrees with solver-derived answer"
    else:
        reason = None
    return ProofAnswerVerification(
        proof_valid and answer_valid,
        proof_valid,
        answer_valid,
        derived,
        reason,
    )


def verify_sealed_gold(
    item: ItemRecord,
    store: StoreRecord,
    gold: SealedGoldRecord,
    solver: ProofSolver,
) -> ProofAnswerVerification:
    """Verify that a sealed fixture's own proof and answer are executable."""

    return verify_proof_and_answer(
        item=item,
        store=store,
        gold=gold,
        proof=gold.proof,
        answer=gold.answer,
        solver=solver,
    )
