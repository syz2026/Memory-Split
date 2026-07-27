#!/usr/bin/env python
"""Bind Gates 0--5 and source provenance into one immutable study freeze."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.relational_gates import (
    COMMON_INPUT_HASH_FIELDS,
    CONFIRMATION_CANDIDATES,
    LOAD_CANDIDATES,
    ORDERED_MIXTURES,
    validate_gate_receipt,
)
from experiment.artifacts import (
    atomic_write_json,
    canonical_sha256,
    load_canonical_json,
    validate_sha256,
)
from experiment.provenance import SourceProvenance, verify_source_provenance


FIXTURE_WATERMARK = "NONLAUNCHABLE_FIXTURE"
PROTECTED_SEEDS = (1001, 1002, 1003, 1004, 1005)
MODEL_PARAMETERS = {
    "d160m": 162_220_800,
    "d360m": 356_033_536,
}
DEFAULT_TOKENS_PER_STEP = 524_288
DECODE_BUDGET = 6
_REQUIRED_ARTIFACTS = {
    "source_lock",
    "relation_schema",
    "preregistration",
    "evaluator",
    "analysis",
    "corpus_recipe",
    "dense_sidecar_recipe",
    "split_sidecar_recipe",
    "random_sidecar_recipe",
}
_FIELDS = {
    "record_type",
    "schema_version",
    "status",
    "watermark",
    "launchable",
    "source_provenance",
    "gate_receipts",
    "selected_mixture",
    "low_entities",
    "high_entities",
    "confirmation_entities",
    "tokens_per_parameter",
    "tokens_per_step",
    "seeds",
    "model_parameters",
    "decode_budget",
    "decision_sha256",
    "freeze_sha256",
}


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _deep_freeze(item)
                for key, item in sorted(value.items())
            }
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class FreezeManifest:
    status: str
    watermark: str | None
    launchable: bool
    source_provenance: SourceProvenance
    gate_receipts: Mapping[str, Mapping[str, Any]]
    selected_mixture: tuple[float, float, float]
    low_entities: int
    high_entities: int
    confirmation_entities: int
    tokens_per_parameter: int
    tokens_per_step: int
    seeds: tuple[int, ...]
    model_parameters: Mapping[str, int]
    decode_budget: int
    decision_sha256: str
    freeze_sha256: str

    def __post_init__(self) -> None:
        if self.status not in {"fixture", "frozen"}:
            raise ValueError("freeze status must be fixture or frozen")
        if not isinstance(self.launchable, bool):
            raise ValueError("freeze launchable must be Boolean")
        if self.status == "fixture":
            if self.watermark != FIXTURE_WATERMARK or self.launchable:
                raise ValueError(
                    "fixture freeze must be watermarked and nonlaunchable"
                )
        elif self.watermark is not None or not self.launchable:
            raise ValueError(
                "real frozen study must be launchable without a watermark"
            )
        if not isinstance(self.source_provenance, SourceProvenance):
            raise TypeError("freeze requires typed source provenance")
        if not isinstance(self.gate_receipts, Mapping):
            raise ValueError("gate_receipts must be an object")
        object.__setattr__(
            self,
            "gate_receipts",
            _deep_freeze(_deep_thaw(self.gate_receipts)),
        )
        mixture = tuple(self.selected_mixture)
        if mixture not in ORDERED_MIXTURES:
            raise ValueError("selected mixture is not in the frozen grid")
        object.__setattr__(self, "selected_mixture", mixture)
        low = _integer(self.low_entities, "low entities", minimum=1)
        high = _integer(self.high_entities, "high entities", minimum=1)
        confirmation = _integer(
            self.confirmation_entities,
            "confirmation entities",
            minimum=1,
        )
        if (
            low not in LOAD_CANDIDATES
            or high not in LOAD_CANDIDATES
            or low == high
            or confirmation not in CONFIRMATION_CANDIDATES
        ):
            raise ValueError("freeze load selection is invalid")
        if self.tokens_per_parameter not in {10, 20}:
            raise ValueError("tokens per parameter must be 10 or 20")
        _integer(self.tokens_per_step, "tokens per step", minimum=1)
        seeds = tuple(self.seeds)
        if seeds != PROTECTED_SEEDS:
            raise ValueError("protected seeds must be exactly 1001 through 1005")
        object.__setattr__(self, "seeds", seeds)
        if dict(self.model_parameters) != MODEL_PARAMETERS:
            raise ValueError("protected model parameter counts drifted")
        object.__setattr__(
            self,
            "model_parameters",
            MappingProxyType(dict(MODEL_PARAMETERS)),
        )
        if self.decode_budget != DECODE_BUDGET:
            raise ValueError("decode budget must remain six action slots")
        validate_sha256(self.decision_sha256, "freeze decision SHA-256")
        validate_sha256(self.freeze_sha256, "freeze SHA-256")

    @property
    def source_tree_sha256(self) -> str:
        return self.source_provenance.source_tree_sha256

    @property
    def git_revision(self) -> str:
        return self.source_provenance.git_revision

    @property
    def artifact_sha256(self) -> Mapping[str, str]:
        return self.source_provenance.artifact_sha256

    def _decision_material(self) -> dict[str, Any]:
        return {
            "record_type": "relational_freeze_decision",
            "schema_version": 1,
            "status": self.status,
            "watermark": self.watermark,
            "source_tree_sha256": self.source_tree_sha256,
            "git_revision": self.git_revision,
            "artifact_sha256": dict(self.artifact_sha256),
            "gate_receipt_sha256": {
                name: receipt["receipt_sha256"]
                for name, receipt in sorted(self.gate_receipts.items())
            },
            "selected_mixture": list(self.selected_mixture),
            "low_entities": self.low_entities,
            "high_entities": self.high_entities,
            "confirmation_entities": self.confirmation_entities,
            "tokens_per_parameter": self.tokens_per_parameter,
            "tokens_per_step": self.tokens_per_step,
            "seeds": list(self.seeds),
            "model_parameters": dict(self.model_parameters),
            "decode_budget": self.decode_budget,
        }

    def _without_freeze_hash(self) -> dict[str, Any]:
        return {
            "record_type": "relational_study_freeze",
            "schema_version": 1,
            "status": self.status,
            "watermark": self.watermark,
            "launchable": self.launchable,
            "source_provenance": self.source_provenance.to_dict(),
            "gate_receipts": _deep_thaw(self.gate_receipts),
            "selected_mixture": list(self.selected_mixture),
            "low_entities": self.low_entities,
            "high_entities": self.high_entities,
            "confirmation_entities": self.confirmation_entities,
            "tokens_per_parameter": self.tokens_per_parameter,
            "tokens_per_step": self.tokens_per_step,
            "seeds": list(self.seeds),
            "model_parameters": dict(self.model_parameters),
            "decode_budget": self.decode_budget,
            "decision_sha256": self.decision_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._without_freeze_hash(),
            "freeze_sha256": self.freeze_sha256,
        }

    as_dict = to_dict

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FreezeManifest":
        return validate_freeze_manifest(raw)


def _provenance_input_hashes(
    provenance: SourceProvenance,
) -> dict[str, str]:
    artifacts = provenance.artifact_sha256
    return {
        "source_lock_sha256": artifacts["source_lock"],
        "relation_schema_sha256": artifacts["relation_schema"],
        "preregistration_sha256": artifacts["preregistration"],
        "evaluator_sha256": artifacts["evaluator"],
        "analysis_sha256": artifacts["analysis"],
        "source_tree_sha256": provenance.source_tree_sha256,
    }


def _validate_real_bindings(
    provenance: SourceProvenance,
    gate_receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not provenance.clean_tree:
        raise ValueError("real study freeze requires a clean Git revision")
    missing_artifacts = _REQUIRED_ARTIFACTS - set(
        provenance.artifact_sha256
    )
    if missing_artifacts:
        raise ValueError(
            "source provenance is missing required artifact hashes: "
            f"{sorted(missing_artifacts)}"
        )
    expected_names = {f"gate_{number}" for number in range(6)}
    if set(gate_receipts) != expected_names:
        raise ValueError("real freeze requires exactly Gate 0 through Gate 5")
    validated = {
        name: validate_gate_receipt(
            gate_receipts[name],
            expected_gate=int(name.removeprefix("gate_")),
        )
        for name in sorted(gate_receipts)
    }
    for name, receipt in validated.items():
        if not receipt["passed"]:
            raise ValueError(f"{name} must pass before a real freeze")
    development_inputs = {
        validated[f"gate_{number}"]["development_input_sha256"]
        for number in range(1, 5)
    }
    if len(development_inputs) != 1:
        raise ValueError("Gate 1-4 development input hashes must agree")
    identities = [
        receipt["input_hashes"] for receipt in validated.values()
    ]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Gate 0-5 input hashes must agree")
    expected_hashes = _provenance_input_hashes(provenance)
    if identities[0] != expected_hashes:
        differing = sorted(
            name
            for name in COMMON_INPUT_HASH_FIELDS
            if identities[0].get(name) != expected_hashes.get(name)
        )
        raise ValueError(
            "gate input hashes do not match source provenance: "
            f"{differing}"
        )
    gate_chain = (
        ("gate_2", "gate_1_receipt_sha256", "gate_1"),
        ("gate_3", "gate_2_receipt_sha256", "gate_2"),
        ("gate_4", "gate_3_receipt_sha256", "gate_3"),
        ("gate_5", "gate_3_receipt_sha256", "gate_3"),
        ("gate_5", "gate_4_receipt_sha256", "gate_4"),
    )
    for consumer, field, predecessor in gate_chain:
        if (
            validated[consumer][field]
            != validated[predecessor]["receipt_sha256"]
        ):
            raise ValueError(
                f"{consumer} predecessor binding does not match {predecessor}"
            )
    gate_3 = validated["gate_3"]
    gate_4 = validated["gate_4"]
    gate_5 = validated["gate_5"]
    if gate_5["high_entities"] != gate_3["high_entities"]:
        raise ValueError("Gate 5 development does not use the Gate 3 high load")
    if (
        gate_5["tokens_per_parameter"]
        != gate_4["tokens_per_parameter"]
    ):
        raise ValueError(
            "Gate 5 development does not use the Gate 4 token budget"
        )
    high_manifest_hashes = {
        item["manifest_sha256"]
        for item in gate_3["measurements"]
        if item["entities"] == gate_3["high_entities"]
    }
    bound_data_hashes = {
        item["data_sha256"]
        for item in gate_5["development_binding"]["inputs"].values()
    }
    if len(high_manifest_hashes) != 1 or bound_data_hashes != high_manifest_hashes:
        raise ValueError(
            "Gate 5 data does not match the selected Gate 3 high-load manifest"
        )
    return validated


def _finish_manifest(
    *,
    status: str,
    watermark: str | None,
    launchable: bool,
    source_provenance: SourceProvenance,
    gate_receipts: Mapping[str, Mapping[str, Any]],
    selected_mixture: tuple[float, float, float],
    low_entities: int,
    high_entities: int,
    confirmation_entities: int,
    tokens_per_parameter: int,
    tokens_per_step: int,
) -> FreezeManifest:
    provisional = FreezeManifest(
        status=status,
        watermark=watermark,
        launchable=launchable,
        source_provenance=source_provenance,
        gate_receipts=gate_receipts,
        selected_mixture=selected_mixture,
        low_entities=low_entities,
        high_entities=high_entities,
        confirmation_entities=confirmation_entities,
        tokens_per_parameter=tokens_per_parameter,
        tokens_per_step=tokens_per_step,
        seeds=PROTECTED_SEEDS,
        model_parameters=MODEL_PARAMETERS,
        decode_budget=DECODE_BUDGET,
        decision_sha256="0" * 64,
        freeze_sha256="0" * 64,
    )
    decision_sha256 = canonical_sha256(provisional._decision_material())
    with_decision = FreezeManifest(
        **{
            field: getattr(provisional, field)
            for field in (
                "status",
                "watermark",
                "launchable",
                "source_provenance",
                "gate_receipts",
                "selected_mixture",
                "low_entities",
                "high_entities",
                "confirmation_entities",
                "tokens_per_parameter",
                "tokens_per_step",
                "seeds",
                "model_parameters",
                "decode_budget",
            )
        },
        decision_sha256=decision_sha256,
        freeze_sha256="0" * 64,
    )
    return FreezeManifest(
        **{
            field: getattr(with_decision, field)
            for field in (
                "status",
                "watermark",
                "launchable",
                "source_provenance",
                "gate_receipts",
                "selected_mixture",
                "low_entities",
                "high_entities",
                "confirmation_entities",
                "tokens_per_parameter",
                "tokens_per_step",
                "seeds",
                "model_parameters",
                "decode_budget",
                "decision_sha256",
            )
        },
        freeze_sha256=canonical_sha256(
            with_decision._without_freeze_hash()
        ),
    )


def build_freeze_manifest(
    source_provenance: SourceProvenance,
    gate_receipts: Mapping[str, Mapping[str, Any]],
    *,
    tokens_per_step: int = DEFAULT_TOKENS_PER_STEP,
) -> FreezeManifest:
    if not isinstance(source_provenance, SourceProvenance):
        raise TypeError("real freeze requires SourceProvenance")
    provenance = SourceProvenance.from_dict(source_provenance.to_dict())
    validated = _validate_real_bindings(provenance, gate_receipts)
    gate_2 = validated["gate_2"]
    gate_3 = validated["gate_3"]
    gate_4 = validated["gate_4"]
    manifest = _finish_manifest(
        status="frozen",
        watermark=None,
        launchable=True,
        source_provenance=provenance,
        gate_receipts=validated,
        selected_mixture=tuple(gate_2["selected_mixture"]),
        low_entities=gate_3["low_entities"],
        high_entities=gate_3["high_entities"],
        confirmation_entities=gate_3["confirmation_entities"],
        tokens_per_parameter=gate_4["tokens_per_parameter"],
        tokens_per_step=_integer(
            tokens_per_step,
            "tokens per step",
            minimum=1,
        ),
    )
    return validate_freeze_manifest(manifest.to_dict())


def _fixture_provenance() -> SourceProvenance:
    return SourceProvenance(
        git_revision="0" * 40,
        source_tree_sha256=canonical_sha256("fixture-source-tree"),
        clean_tree=False,
        python_version="fixture",
        python_implementation="fixture",
        platform="fixture",
        artifact_sha256={
            name: canonical_sha256(["fixture", name])
            for name in sorted(_REQUIRED_ARTIFACTS)
        },
    )


def make_fixture_freeze(
    *,
    low_entities: int = 50_000,
    high_entities: int = 800_000,
    confirmation_entities: int = 1_800_000,
    tokens_per_parameter: int = 10,
    tokens_per_step: int = DEFAULT_TOKENS_PER_STEP,
) -> FreezeManifest:
    """Build the deterministic config-only fixture, never a launch receipt."""

    manifest = _finish_manifest(
        status="fixture",
        watermark=FIXTURE_WATERMARK,
        launchable=False,
        source_provenance=_fixture_provenance(),
        gate_receipts={},
        selected_mixture=ORDERED_MIXTURES[0],
        low_entities=low_entities,
        high_entities=high_entities,
        confirmation_entities=confirmation_entities,
        tokens_per_parameter=tokens_per_parameter,
        tokens_per_step=tokens_per_step,
    )
    return validate_freeze_manifest(manifest.to_dict())


fixture_freeze = make_fixture_freeze


def validate_freeze_manifest(
    raw: FreezeManifest | Mapping[str, Any],
) -> FreezeManifest:
    value = raw.to_dict() if isinstance(raw, FreezeManifest) else raw
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise ValueError("freeze manifest fields are not exact")
    if (
        value["record_type"] != "relational_study_freeze"
        or value["schema_version"] != 1
    ):
        raise ValueError("freeze manifest protocol mismatch")
    if (
        not isinstance(value["selected_mixture"], list)
        or not isinstance(value["seeds"], list)
        or not isinstance(value["model_parameters"], Mapping)
        or not isinstance(value["gate_receipts"], Mapping)
    ):
        raise ValueError("freeze manifest collection fields are invalid")
    provenance = SourceProvenance.from_dict(value["source_provenance"])
    manifest = FreezeManifest(
        status=value["status"],
        watermark=value["watermark"],
        launchable=value["launchable"],
        source_provenance=provenance,
        gate_receipts=value["gate_receipts"],
        selected_mixture=tuple(value["selected_mixture"]),
        low_entities=value["low_entities"],
        high_entities=value["high_entities"],
        confirmation_entities=value["confirmation_entities"],
        tokens_per_parameter=value["tokens_per_parameter"],
        tokens_per_step=value["tokens_per_step"],
        seeds=tuple(value["seeds"]),
        model_parameters=value["model_parameters"],
        decode_budget=value["decode_budget"],
        decision_sha256=value["decision_sha256"],
        freeze_sha256=value["freeze_sha256"],
    )
    if manifest.status == "fixture":
        if manifest.gate_receipts:
            raise ValueError("fixture freeze cannot contain real gate receipts")
    else:
        validated = _validate_real_bindings(
            manifest.source_provenance,
            manifest.gate_receipts,
        )
        if _deep_thaw(manifest.gate_receipts) != validated:
            raise ValueError("freeze gate receipt normalization mismatch")
        gate_2 = validated["gate_2"]
        gate_3 = validated["gate_3"]
        gate_4 = validated["gate_4"]
        if (
            manifest.selected_mixture
            != tuple(gate_2["selected_mixture"])
            or manifest.low_entities != gate_3["low_entities"]
            or manifest.high_entities != gate_3["high_entities"]
            or manifest.confirmation_entities
            != gate_3["confirmation_entities"]
            or manifest.tokens_per_parameter
            != gate_4["tokens_per_parameter"]
        ):
            raise ValueError("freeze selections do not match gate decisions")
    if manifest.decision_sha256 != canonical_sha256(
        manifest._decision_material()
    ):
        raise ValueError("freeze decision hash mismatch")
    if manifest.freeze_sha256 != canonical_sha256(
        manifest._without_freeze_hash()
    ):
        raise ValueError("freeze hash mismatch")
    return manifest


def require_launchable_freeze(
    freeze: FreezeManifest | Mapping[str, Any],
) -> FreezeManifest:
    manifest = validate_freeze_manifest(freeze)
    if not manifest.launchable:
        raise ValueError("nonlaunchable fixture cannot authorize protected work")
    return manifest


def write_freeze_manifest(
    path: str | Path,
    freeze: FreezeManifest | Mapping[str, Any],
) -> Path:
    destination = Path(path)
    if destination.is_symlink():
        raise ValueError("freeze destination cannot be a symlink")
    if os.path.lexists(destination):
        raise FileExistsError(f"freeze destination already exists: {destination}")
    manifest = validate_freeze_manifest(freeze)
    return atomic_write_json(destination, manifest.to_dict())


def load_freeze_manifest(path: str | Path) -> FreezeManifest:
    return validate_freeze_manifest(load_canonical_json(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a fixture or real relational study freeze."
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--provenance")
    parser.add_argument(
        "--repo-root",
        help="repository whose live revision must match real provenance",
    )
    parser.add_argument(
        "--gate-receipt",
        action="append",
        default=[],
        help="canonical Gate 0-5 receipt JSON; repeat six times",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fixture:
        if args.provenance or args.gate_receipt or args.repo_root:
            raise ValueError("fixture freeze cannot consume real receipts")
        freeze = make_fixture_freeze()
    else:
        if (
            not args.provenance
            or not args.repo_root
            or len(args.gate_receipt) != 6
        ):
            raise ValueError(
                "real freeze requires a live repository, provenance, and "
                "exactly six gate receipts"
            )
        provenance = SourceProvenance.from_dict(
            load_canonical_json(args.provenance)
        )
        verify_source_provenance(
            args.repo_root,
            provenance,
            require_clean=True,
        )
        receipts = {}
        for path in args.gate_receipt:
            receipt = validate_gate_receipt(load_canonical_json(path))
            name = f"gate_{receipt['gate']}"
            if name in receipts:
                raise ValueError(f"duplicate receipt for {name}")
            receipts[name] = receipt
        freeze = build_freeze_manifest(provenance, receipts)
    write_freeze_manifest(args.out, freeze)
    print(json.dumps(freeze.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
