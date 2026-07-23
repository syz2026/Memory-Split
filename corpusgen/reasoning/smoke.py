"""Deterministic non-scientific smoke compiler for the v2 local core."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from corpusgen.reasoning.closure import (
    SemanticFact,
    SupervisedField,
    audit_occurrence_closure,
    plan_occurrence_closure,
)
from corpusgen.reasoning.proofs import (
    CompositionPremise,
    EqualityPremise,
    ProofObject,
    solve_graph_composition,
    solve_slot_equality,
    verify_proof,
)
from corpusgen.reasoning.routing import (
    FactMetadata,
    RouteManifest,
    build_route_manifests,
)
from corpusgen.reasoning.state import AnswerPointer, serialize_answer_state
from train.tokenizer import get_tok


_FORMAT = "memorysplit-reasoning-v2-smoke-v1"
_VALUES = (
    ("value-00-cerulean", "alias-00-blue"),
    ("value-01-umber", "alias-01-brown"),
    ("value-02-saffron", "alias-02-yellow"),
    ("value-03-viridian", "alias-03-green"),
    ("value-04-carmine", "alias-04-red"),
    ("value-05-indigo", "alias-05-violet"),
    ("value-06-silver", "alias-06-gray"),
    ("value-07-ochre", "alias-07-earth"),
    ("value-08-coral", "alias-08-orange"),
    ("value-09-ebony", "alias-09-black"),
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture_facts() -> tuple[FactMetadata, ...]:
    return tuple(
        FactMetadata(
            fact_id=f"fact-{index:02d}",
            source="reasoning-v2-smoke",
            record_type="graph_return",
            payload_entropy_bits=100 - index,
            scheduled_exposures=1,
            expected_reads=0,
            expected_hops=0,
            surfaces=surfaces,
        )
        for index, surfaces in enumerate(_VALUES)
    )


def _semantic_facts(
    facts: tuple[FactMetadata, ...],
) -> tuple[SemanticFact, ...]:
    return tuple(SemanticFact(fact.fact_id, fact.surfaces) for fact in facts)


def _fixture_states() -> tuple[str, ...]:
    return tuple(
        serialize_answer_state(
            AnswerPointer(
                slot=index % 4,
                read_index=index % 12,
                member_index=0,
            ),
            phase=phase,
        )
        for index in range(len(_VALUES))
        for phase in ("candidate", "final")
    )


def _fixture_fields(states: tuple[str, ...]) -> tuple[SupervisedField, ...]:
    return tuple(
        SupervisedField(
            field_id=f"record-{index:02d}",
            text=(
                f"Declaration {index}: target={value}; alias={alias}. "
                f"Memory return target={value}; alias={alias}. "
                f"{states[2 * index]} {states[2 * index + 1]}"
            ),
        )
        for index, (value, alias) in enumerate(_VALUES)
    )


def _fixture_proofs() -> tuple[ProofObject, ...]:
    composition = solve_graph_composition(
        (
            CompositionPremise("fact-00", hop=0, compose_code=2),
            CompositionPremise("fact-01", hop=1, compose_code=3),
            CompositionPremise("fact-02", hop=2, compose_code=1),
        )
    )
    equality = solve_slot_equality(
        (
            EqualityPremise("fact-08", slot=0, value=_VALUES[8][0]),
            EqualityPremise("fact-09", slot=1, value=_VALUES[9][0]),
        )
    )
    return composition, equality


@dataclass(frozen=True)
class _EncodedSlice:
    field_id: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    fact_ids: tuple[str, ...]


def _encode_fields(
    facts: tuple[SemanticFact, ...],
    fields: tuple[SupervisedField, ...],
) -> tuple[np.ndarray, tuple[_EncodedSlice, ...]]:
    tok = get_tok()
    plan = plan_occurrence_closure(facts, fields)
    token_ids: list[int] = []
    slices = []
    for field in fields:
        for part in plan.partition_field(field.field_id):
            start = len(token_ids)
            token_ids.extend(tok.encode(part.text))
            end = len(token_ids)
            if end <= start:
                raise ValueError("non-empty closure slice encoded to no tokens")
            slices.append(
                _EncodedSlice(
                    field_id=field.field_id,
                    char_start=part.start,
                    char_end=part.end,
                    token_start=start,
                    token_end=end,
                    fact_ids=part.fact_ids,
                )
            )
        token_ids.append(tok.EOT)
    if any(token < 0 or token >= 1 << 16 for token in token_ids):
        raise ValueError("v2 smoke token id does not fit uint16")
    return np.asarray(token_ids, dtype=np.uint16), tuple(slices)


def _mask_for_manifest(
    token_count: int,
    slices: tuple[_EncodedSlice, ...],
    manifest: RouteManifest,
) -> bytes:
    external = frozenset(manifest.external_fact_ids)
    mask = bytearray([1]) * token_count
    for part in slices:
        if external.intersection(part.fact_ids):
            mask[part.token_start : part.token_end] = bytes(
                part.token_end - part.token_start
            )
    return bytes(mask)


def _token_leaks(
    facts: tuple[SemanticFact, ...],
    fields: tuple[SupervisedField, ...],
    external: frozenset[str],
    slices: tuple[_EncodedSlice, ...],
    weights: bytes,
) -> list[dict[str, Any]]:
    plan = plan_occurrence_closure(facts, fields)
    leaks = []
    for occurrence in plan.occurrences:
        if occurrence.fact_id not in external:
            continue
        matching = [
            part
            for part in slices
            if part.field_id == occurrence.field_id
            and part.char_start < occurrence.end
            and occurrence.start < part.char_end
        ]
        if not matching or any(
            any(weights[part.token_start : part.token_end])
            for part in matching
        ):
            leaks.append(occurrence.as_dict())
    return leaks


def _write_jsonl(path: Path, rows) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical_bytes(row))


def _build_private(root: Path) -> dict:
    facts = _fixture_facts()
    semantic_facts = _semantic_facts(facts)
    states = _fixture_states()
    fields = _fixture_fields(states)
    proofs = _fixture_proofs()
    if not all(verify_proof(proof, premises) for proof, premises in (
        (
            proofs[0],
            (
                CompositionPremise("fact-00", 0, 2),
                CompositionPremise("fact-01", 1, 3),
                CompositionPremise("fact-02", 2, 1),
            ),
        ),
        (
            proofs[1],
            (
                EqualityPremise("fact-08", 0, _VALUES[8][0]),
                EqualityPremise("fact-09", 1, _VALUES[9][0]),
            ),
        ),
    )):
        raise AssertionError("fixture proof verification failed")

    manifests = build_route_manifests(facts)
    token_ids, encoded_slices = _encode_fields(semantic_facts, fields)
    token_ids.tofile(root / "train.bin")
    dense = bytes([1]) * len(token_ids)
    (root / "dense.weights.bin").write_bytes(dense)
    _write_jsonl(
        root / "records.jsonl",
        (
            {
                "field_id": field.field_id,
                "text": field.text,
                "supervised": field.supervised,
            }
            for field in fields
        ),
    )
    _write_jsonl(
        root / "answer-states.jsonl",
        ({"state": state} for state in states),
    )
    with (root / "proofs.jsonl").open("wb") as handle:
        for proof in proofs:
            handle.write(proof.to_bytes())

    route_dose = {}
    closure_reports = {}
    plan = plan_occurrence_closure(semantic_facts, fields)
    for split, manifest in manifests.items():
        stem = split.lower()
        (root / f"{stem}-route-manifest.json").write_bytes(manifest.to_bytes())
        weights = _mask_for_manifest(len(token_ids), encoded_slices, manifest)
        (root / f"{stem}.weights.bin").write_bytes(weights)
        char_masks = plan.mask_for_routes(manifest.external_fact_ids)
        leakage = audit_occurrence_closure(
            semantic_facts,
            fields,
            manifest.external_fact_ids,
            char_masks,
        )
        token_leaks = _token_leaks(
            semantic_facts,
            fields,
            frozenset(manifest.external_fact_ids),
            encoded_slices,
            weights,
        )
        if token_leaks:
            raise ValueError(f"{split} left supervised token copies unmasked")
        leakage_json = {
            **leakage.as_dict(),
            "unmasked_token_occurrences": len(token_leaks),
        }
        _write_json(root / f"{stem}-semantic-leakage.json", leakage_json)
        route_dose[split] = {
            "target_fraction": (
                f"{manifest.target_fraction.numerator}/"
                f"{manifest.target_fraction.denominator}"
            ),
            "total_facts": manifest.total_facts,
            "quota_facts": manifest.quota_count,
            "external_facts": manifest.external_count,
            "minimally_rounded": True,
            "information_burden_fraction": (
                f"{manifest.information_burden_fraction.numerator}/"
                f"{manifest.information_burden_fraction.denominator}"
            ),
            "information_burden_quota_met": (
                manifest.information_burden_quota_met
            ),
        }
        closure_reports[split] = {
            "passed": leakage.passed and not token_leaks,
            "supervised_occurrences": leakage.supervised_occurrences,
            "masked_occurrences": leakage.masked_occurrences,
            "unmasked_supervised_occurrences": len(
                leakage.unmasked_occurrences
            ),
            "unmasked_token_occurrences": len(token_leaks),
        }

    surfaces = tuple(
        surface
        for fact in semantic_facts
        for surface in fact.surfaces
    )
    state_surface_copies = sum(
        surface in state
        for state in states
        for surface in surfaces
    )
    report = {
        "format": _FORMAT,
        "profile": "smoke",
        "scientific_result": False,
        "scientific_readiness": False,
        "status": "non-scientific implementation smoke only",
        "facts": len(facts),
        "tokens": len(token_ids),
        "route_dose": route_dose,
        "semantic_closure": closure_reports,
        "answer_states": {
            "format": "pointer-slot",
            "states": len(states),
            "phases": ["candidate", "final"],
            "surface_value_copies": state_surface_copies,
        },
        "proofs": {
            "families": [proof.family for proof in proofs],
            "verified": True,
        },
    }
    if state_surface_copies:
        raise ValueError("pointer answer state repeated a factual surface")
    _write_json(root / "report.json", report)
    return report


def _write_manifest(root: Path) -> None:
    artifacts = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    _write_json(
        root / "manifest.json",
        {
            "format": _FORMAT,
            "profile": "smoke",
            "scientific_result": False,
            "scientific_readiness": False,
            "report": "report.json",
            "artifacts": artifacts,
        },
    )


def _read_records(path: Path) -> tuple[SupervisedField, ...]:
    return tuple(
        SupervisedField(
            field_id=row["field_id"],
            text=row["text"],
            supervised=row["supervised"],
        )
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    )


def _read_semantic_facts(route_manifest: dict) -> tuple[SemanticFact, ...]:
    return tuple(
        SemanticFact(
            decision["fact_id"],
            tuple(decision["surfaces"]),
        )
        for decision in route_manifest["decisions"]
    )


def verify_v2_smoke_fixture(out_dir: Path | str) -> dict:
    root = Path(out_dir)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("missing regular v2 smoke directory")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("v2 smoke manifest is missing or unsafe")
    manifest = json.loads(manifest_path.read_bytes())
    if (
        set(manifest)
        != {
            "format",
            "profile",
            "scientific_result",
            "scientific_readiness",
            "report",
            "artifacts",
        }
        or manifest["format"] != _FORMAT
        or manifest["profile"] != "smoke"
        or manifest["scientific_result"] is not False
        or manifest["scientific_readiness"] is not False
        or manifest["report"] != "report.json"
    ):
        raise ValueError("v2 smoke manifest identity mismatch")

    expected_files = {"manifest.json"}
    for artifact in manifest["artifacts"]:
        if set(artifact) != {"path", "bytes", "sha256"}:
            raise ValueError("invalid v2 smoke artifact record")
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("v2 smoke artifact path is unsafe")
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != artifact["bytes"]
            or _sha256(path) != artifact["sha256"]
        ):
            raise ValueError(f"v2 smoke artifact drift: {relative}")
        expected_files.add(relative.as_posix())
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("v2 smoke manifest is not hash-complete")

    report = json.loads((root / manifest["report"]).read_bytes())
    if (
        report.get("format") != _FORMAT
        or report.get("profile") != "smoke"
        or report.get("scientific_result") is not False
        or report.get("scientific_readiness") is not False
        or report.get("status") != "non-scientific implementation smoke only"
        or report.get("answer_states", {}).get("surface_value_copies") != 0
        or report.get("answer_states", {}).get("phases")
        != ["candidate", "final"]
        or report.get("proofs", {}).get("verified") is not True
    ):
        raise ValueError("v2 smoke report failed non-scientific core checks")

    tokens = int(report["tokens"])
    train = np.fromfile(root / "train.bin", dtype=np.uint16)
    dense = (root / "dense.weights.bin").read_bytes()
    if len(train) != tokens or dense != bytes([1]) * tokens:
        raise ValueError("v2 smoke train stream and Dense sidecar disagree")
    fields = _read_records(root / "records.jsonl")

    for split in ("Split50", "Split90"):
        stem = split.lower()
        route = json.loads((root / f"{stem}-route-manifest.json").read_bytes())
        semantic_facts = _read_semantic_facts(route)
        rebuilt_ids, slices = _encode_fields(semantic_facts, fields)
        if not np.array_equal(rebuilt_ids, train):
            raise ValueError(f"{split} semantic reconstruction changed train.bin")
        external = frozenset(
            decision["fact_id"]
            for decision in route["decisions"]
            if decision["route"] == "external"
        )
        weights = (root / f"{stem}.weights.bin").read_bytes()
        if len(weights) != tokens or set(weights) - {0, 1} or 0 not in weights:
            raise ValueError(f"{split} sidecar is invalid")
        leaks = _token_leaks(
            semantic_facts,
            fields,
            external,
            slices,
            weights,
        )
        leakage = json.loads(
            (root / f"{stem}-semantic-leakage.json").read_bytes()
        )
        dose = report["route_dose"][split]
        closure = report["semantic_closure"][split]
        if (
            route["metadata_scope"] != "training-only"
            or route["policy"] != "train-score-ranked-quota-v1"
            or route["external_facts"] != route["quota_facts"]
            or route["information_burden_quota_met"] is not True
            or len(external) != dose["external_facts"]
            or dose["external_facts"] != dose["quota_facts"]
            or dose["information_burden_quota_met"] is not True
            or leaks
            or leakage["passed"] is not True
            or leakage["unmasked_supervised_occurrences"] != 0
            or leakage["unmasked_token_occurrences"] != 0
            or closure["passed"] is not True
            or closure["unmasked_supervised_occurrences"] != 0
            or closure["unmasked_token_occurrences"] != 0
        ):
            raise ValueError(f"{split} route or semantic closure proof failed")
    return report


def build_v2_smoke_fixture(out_dir: Path | str) -> dict:
    destination = Path(out_dir)
    if (
        destination.is_dir()
        and not destination.is_symlink()
        and not any(destination.iterdir())
    ):
        destination.rmdir()
    if destination.exists() or destination.is_symlink():
        return verify_v2_smoke_fixture(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.parent / f".{destination.name}.v2-partial-{os.getpid()}"
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(f"stale v2 smoke partial output: {partial}")
    partial.mkdir()
    try:
        _build_private(partial)
        _write_manifest(partial)
        verify_v2_smoke_fixture(partial)
        partial.rename(destination)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return verify_v2_smoke_fixture(destination)
