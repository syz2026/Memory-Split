"""Fail-closed orchestration for sealed confirmatory evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Protocol

from evals.confirmatory import metrics, reporting, solver as solver_module
from evals.confirmatory.actions import ActionSlot, validate_action_slots
from evals.confirmatory.contracts import (
    CONTRACT_VERSION,
    CheckpointRecord,
    ItemRecord,
    SealedGoldRecord,
    StoreRecord,
    canonical_json_bytes,
    canonical_sha256,
    validate_contract_bundle,
)


RUN_BINDING_SCHEMA = "memorysplit.confirmatory.run-binding.v2"
OUTCOME_SCHEMA = "memorysplit.confirmatory.outcome.v2"
METRICS_SCHEMA = "memorysplit.confirmatory.metrics.v2"
STUDY_LOCK_SCHEMA = "memorysplit.confirmatory.study-lock.v2"
INFERENCE_EVIDENCE_SCHEMA = "memorysplit.confirmatory.inference-evidence.v2"
VALIDITY_EVIDENCE_SCHEMA = "memorysplit.confirmatory.validity-evidence.v2"
PRIMARY_CONTRAST_ID = (
    "primary_omnibus_pair_and_proof__graph_non_path__"
    "composition_joint_ood__split90_minus_dense"
)
PRIMARY_TEST_METHOD = "exact_one_sided_exhaustive_sign_flip"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RUN_FIELDS = frozenset(
    {
        "record_type",
        "schema_version",
        "run_id",
        "checkpoint_path",
        "checkpoint_sha256",
        "configuration_path",
        "configuration_sha256",
        "route_dose_sha256",
        "corpus_sha256",
        "code_sha256",
        "seed",
        "condition_id",
    }
)
_STUDY_LOCK_FIELDS = frozenset(
    {
        "record_type",
        "schema_version",
        "preregistration_sha256",
        "release",
        "checkpoints",
        "validity_receipts",
    }
)
_RELEASE_FIELDS = frozenset(
    {
        "items_sha256",
        "sealed_gold_sha256",
        "stores_sha256",
        "checkpoints_sha256",
        "item_ids",
        "pair_ids",
        "world_ids",
        "evaluation_cells",
        "item_count",
        "pair_count",
        "world_count",
        "evaluation_cell_count",
        "required_families",
        "required_strata",
        "required_memory_modes",
        "required_controls",
    }
)
_EVALUATION_CELL_FIELDS = frozenset(
    {"item_id", "checkpoint_sha256", "seed", "condition_id"}
)
_CHECKPOINT_APPROVAL_FIELDS = frozenset(
    {
        "checkpoint_sha256",
        "seed",
        "condition_id",
        "configuration_sha256",
        "route_dose_sha256",
    }
)
_HARDENED_ARTIFACTS = (
    "checkpoints.jsonl",
    "inference.json",
    "items.jsonl",
    "metrics.json",
    "outcomes.jsonl",
    "sealed-gold.jsonl",
    "study-lock.json",
    "stores.jsonl",
    "validity.json",
)


class ReportingInterfaceUnavailable(RuntimeError):
    """The installed reporting core cannot safely replay runner evidence."""


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    study_lock_sha256: str
    checkpoint_sha256: str
    condition_id: str
    seed: int
    item_count: int
    report_sha256: str


@dataclass(frozen=True)
class PreflightResult:
    study_lock_sha256: str
    checkpoint_sha256: str
    condition_id: str
    seed: int
    item_count: int


@dataclass(frozen=True)
class RunBinding:
    run_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    configuration_path: str
    configuration_sha256: str
    route_dose_sha256: str
    corpus_sha256: str
    code_sha256: str
    seed: int
    condition_id: str


@dataclass(frozen=True)
class _StudyLockView:
    raw: Mapping[str, Any]
    release: Mapping[str, Any]
    approvals: tuple[Mapping[str, Any], ...]
    content: bytes
    sha256: str


@dataclass(frozen=True)
class _CheckpointView:
    typed: CheckpointRecord
    raw: Mapping[str, Any]
    content: bytes


@dataclass(frozen=True)
class _PreparedEvaluation:
    release: Path
    lock: _StudyLockView
    binding: RunBinding
    checkpoint: _CheckpointView
    selected_ids: tuple[str, ...]
    items_content: bytes
    items: Mapping[str, ItemRecord]


@dataclass(frozen=True)
class Submission:
    """One model answer and its fixed-width action trace."""

    item_id: str
    answer: str
    actions: tuple[ActionSlot, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id:
            raise ValueError("submission item_id must be a non-empty string")
        if not isinstance(self.answer, str):
            raise ValueError("submission answer must be a string")
        object.__setattr__(self, "actions", validate_action_slots(self.actions))

    @property
    def proof(self) -> tuple[ActionSlot, ...]:
        return self.actions


class ModelAdapter(Protocol):
    """Injected model boundary; sealed records never cross this interface."""

    def generate(self, item: ItemRecord) -> Submission: ...


class DeterministicFixtureAdapter:
    """Deterministic item-keyed adapter for CPU-only runner tests."""

    def __init__(
        self,
        submissions: Mapping[
            str,
            Submission,
        ],
    ) -> None:
        if not isinstance(submissions, Mapping) or not submissions:
            raise ValueError("fixture adapter requires submissions")
        copied: dict[str, Submission] = {}
        for item_id, submission in submissions.items():
            if (
                not isinstance(item_id, str)
                or not item_id
                or not isinstance(submission, Submission)
                or submission.item_id != item_id
            ):
                raise ValueError("fixture submission binding is invalid")
            copied[item_id] = submission
        self._submissions = MappingProxyType(copied)

    def generate(self, item: ItemRecord) -> Submission:
        item_id = getattr(item, "item_id", None)
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("model-visible item has no item_id")
        try:
            return self._submissions[item_id]
        except KeyError as exc:
            raise ValueError(f"fixture adapter missing item: {item_id}") from exc


def _strict_mapping(
    value: object,
    expected: frozenset[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} fields are not exact")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _condition(value: object, name: str) -> str:
    if value not in {"dense", "split90"}:
        raise ValueError(
            f"{name} must be explicit dense or split90; generic split is invalid"
        )
    return str(value)


def _canonical_object(content: bytes, name: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be canonical UTF-8 JSON") from exc
    if not isinstance(raw, Mapping) or canonical_json_bytes(raw) != content:
        raise ValueError(f"{name} must be a canonical JSON object")
    return raw


def _read_regular_file(path: Path, name: str) -> bytes:
    try:
        status = path.lstat()
    except FileNotFoundError:
        raise ValueError(f"{name} is missing") from None
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")
    return path.read_bytes()


def _directory(path: str | Path, name: str) -> Path:
    result = Path(path)
    try:
        status = result.lstat()
    except FileNotFoundError:
        raise ValueError(f"{name} directory is missing") from None
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink directory")
    return result


def _relative_file(root: Path, value: object, name: str) -> Path:
    raw = _string(value, name)
    relative = Path(raw)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"{name} must be a traversal-free relative path")
    return root.joinpath(relative)


def _canonical_jsonl(
    content: bytes,
    *,
    name: str,
    parser,
    identity,
) -> tuple[Any, ...]:
    lines = content.splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") for line in lines):
        raise ValueError(f"{name} must be non-empty canonical JSONL")
    result = []
    identities = []
    for index, line in enumerate(lines):
        try:
            raw = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{name} line {index} is invalid JSON") from exc
        if not isinstance(raw, Mapping) or canonical_json_bytes(raw) != line:
            raise ValueError(f"{name} line {index} is not canonical")
        parsed = parser(raw)
        result.append(parsed)
        identities.append(identity(parsed))
    if len(set(identities)) != len(identities):
        raise ValueError(f"{name} contains a duplicate identity")
    if tuple(identities) != tuple(sorted(identities)):
        raise ValueError(f"{name} is not canonically ordered")
    return tuple(result)


def _load_run_binding(run: Path) -> RunBinding:
    raw = _strict_mapping(
        _canonical_object(
            _read_regular_file(run / "run.json", "run binding"),
            "run.json",
        ),
        _RUN_FIELDS,
        "run binding",
    )
    if raw["record_type"] != RUN_BINDING_SCHEMA:
        raise ValueError("run binding record_type is invalid")
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != CONTRACT_VERSION
    ):
        raise ValueError("run binding schema_version is invalid")
    return RunBinding(
        run_id=_string(raw["run_id"], "run_id"),
        checkpoint_path=_string(raw["checkpoint_path"], "checkpoint_path"),
        checkpoint_sha256=_sha256(
            raw["checkpoint_sha256"],
            "checkpoint_sha256",
        ),
        configuration_path=_string(
            raw["configuration_path"],
            "configuration_path",
        ),
        configuration_sha256=_sha256(
            raw["configuration_sha256"],
            "configuration_sha256",
        ),
        route_dose_sha256=_sha256(
            raw["route_dose_sha256"],
            "route_dose_sha256",
        ),
        corpus_sha256=_sha256(raw["corpus_sha256"], "corpus_sha256"),
        code_sha256=_sha256(raw["code_sha256"], "code_sha256"),
        seed=_integer(raw["seed"], "seed"),
        condition_id=_condition(raw["condition_id"], "run condition_id"),
    )


def _validate_study_lock_fallback(raw: Mapping[str, Any]) -> None:
    value = _strict_mapping(raw, _STUDY_LOCK_FIELDS, "study lock")
    if value["record_type"] != STUDY_LOCK_SCHEMA:
        raise ValueError("study lock record_type is invalid")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != CONTRACT_VERSION
    ):
        raise ValueError("study lock schema_version is invalid")
    _sha256(value["preregistration_sha256"], "preregistration_sha256")
    release = _strict_mapping(value["release"], _RELEASE_FIELDS, "release binding")
    for field in (
        "items_sha256",
        "sealed_gold_sha256",
        "stores_sha256",
        "checkpoints_sha256",
    ):
        _sha256(release[field], f"release {field}")
    for field in ("item_ids", "pair_ids", "world_ids"):
        values = release[field]
        if (
            not isinstance(values, list)
            or any(not isinstance(item, str) or not item for item in values)
            or len(set(values)) != len(values)
            or values != sorted(values)
        ):
            raise ValueError(f"release {field} is not a unique ordered registry")
    for registry, count in (
        ("item_ids", "item_count"),
        ("pair_ids", "pair_count"),
        ("world_ids", "world_count"),
        ("evaluation_cells", "evaluation_cell_count"),
    ):
        if type(release[count]) is not int or release[count] != len(release[registry]):
            raise ValueError(f"release {count} disagrees with {registry}")
    cells = release["evaluation_cells"]
    if not isinstance(cells, list):
        raise ValueError("release evaluation_cells must be ordered")
    cell_keys = []
    for cell in cells:
        typed = _strict_mapping(cell, _EVALUATION_CELL_FIELDS, "evaluation cell")
        cell_keys.append(
            (
                _integer(typed["seed"], "evaluation cell seed"),
                _condition(typed["condition_id"], "evaluation cell condition_id"),
                _string(typed["item_id"], "evaluation cell item_id"),
                _sha256(
                    typed["checkpoint_sha256"],
                    "evaluation cell checkpoint_sha256",
                ),
            )
        )
    if len(set(cell_keys)) != len(cell_keys) or cell_keys != sorted(cell_keys):
        raise ValueError("release evaluation_cells are duplicated or unordered")
    approvals = value["checkpoints"]
    if not isinstance(approvals, list) or not approvals:
        raise ValueError("study lock requires checkpoint approvals")
    approval_keys = []
    for approval in approvals:
        typed = _strict_mapping(
            approval,
            _CHECKPOINT_APPROVAL_FIELDS,
            "checkpoint approval",
        )
        approval_keys.append(
            (
                _integer(typed["seed"], "checkpoint approval seed"),
                _condition(
                    typed["condition_id"],
                    "checkpoint approval condition_id",
                ),
            )
        )
        for field in (
            "checkpoint_sha256",
            "configuration_sha256",
            "route_dose_sha256",
        ):
            _sha256(typed[field], f"checkpoint approval {field}")
    if len(set(approval_keys)) != len(approval_keys) or approval_keys != sorted(
        approval_keys
    ):
        raise ValueError("study lock checkpoint approvals are duplicated or unordered")
    if not isinstance(value["validity_receipts"], list):
        raise ValueError("study lock validity_receipts must be ordered")


def _load_study_lock(
    release: Path,
    expected_study_lock_sha256: str,
) -> _StudyLockView:
    expected = _sha256(
        expected_study_lock_sha256,
        "expected_study_lock_sha256",
    )
    content = _read_regular_file(release / "study-lock.json", "study-lock.json")
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise ValueError("study lock disagrees with external commitment")
    raw = _canonical_object(content, "study-lock.json")
    try:
        from evals.confirmatory.study_lock import StudyLock
    except ImportError:
        _validate_study_lock_fallback(raw)
    else:
        StudyLock.from_dict(raw)
    release_binding = _strict_mapping(
        raw["release"],
        _RELEASE_FIELDS,
        "release binding",
    )
    approvals = raw["checkpoints"]
    return _StudyLockView(
        raw=MappingProxyType(dict(raw)),
        release=MappingProxyType(dict(release_binding)),
        approvals=tuple(approvals),
        content=content,
        sha256=actual,
    )


def _read_bound_artifact(
    release: Path,
    name: str,
    expected_sha256: object,
) -> bytes:
    content = _read_regular_file(release / name, name)
    expected = _sha256(expected_sha256, f"{name} sealed hash")
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"{name} disagrees with sealed release commitment")
    return content


def _checkpoint_record(raw: Mapping[str, Any]) -> CheckpointRecord:
    current_fields = set(CheckpointRecord.FIELDS)
    if set(raw) == current_fields:
        return CheckpointRecord.from_dict(raw)
    compatibility_fields = current_fields | {"condition_id", "route_dose_sha256"}
    if set(raw) != compatibility_fields:
        raise ValueError("checkpoint fields are not exact")
    return CheckpointRecord.from_dict(
        {key: raw[key] for key in CheckpointRecord.FIELDS}
    )


def _load_checkpoints(
    release: Path,
    lock: _StudyLockView,
    binding: RunBinding,
) -> _CheckpointView:
    content = _read_bound_artifact(
        release,
        "checkpoints.jsonl",
        lock.release["checkpoints_sha256"],
    )
    lines = content.splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") for line in lines):
        raise ValueError("checkpoints.jsonl must be non-empty canonical JSONL")
    records: list[tuple[Mapping[str, Any], CheckpointRecord]] = []
    hashes: list[str] = []
    order: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        try:
            raw = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"checkpoints.jsonl line {index} is invalid JSON") from exc
        if not isinstance(raw, Mapping) or canonical_json_bytes(raw) != line:
            raise ValueError(f"checkpoints.jsonl line {index} is not canonical")
        condition_id = _condition(
            raw.get("condition_id"),
            "checkpoint condition_id",
        )
        expected_arm = "dense" if condition_id == "dense" else "split"
        if raw.get("arm") != expected_arm:
            raise ValueError("checkpoint condition_id disagrees with arm")
        typed = _checkpoint_record(raw)
        checkpoint_hash = _sha256(
            raw.get("checkpoint_sha256"),
            "checkpoint checkpoint_sha256",
        )
        hashes.append(checkpoint_hash)
        order.append((_integer(raw.get("seed"), "checkpoint seed"), condition_id))
        records.append((raw, typed))
    if len(set(hashes)) != len(hashes):
        raise ValueError("checkpoints.jsonl contains a duplicate hash")
    if order != sorted(order):
        raise ValueError("checkpoints.jsonl is not ordered by seed and condition")
    matches = [
        record
        for record in records
        if record[0]["checkpoint_sha256"] == binding.checkpoint_sha256
    ]
    if len(matches) != 1:
        raise ValueError(
            "run checkpoint hash is not uniquely bound by the sealed release"
        )
    raw, typed = matches[0]
    for field in (
        "checkpoint_sha256",
        "configuration_sha256",
        "route_dose_sha256",
        "corpus_sha256",
        "code_sha256",
        "seed",
        "condition_id",
    ):
        if raw[field] != getattr(binding, field):
            raise ValueError(f"run/checkpoint binding mismatch: {field}")
    approvals = [
        approval
        for approval in lock.approvals
        if approval["checkpoint_sha256"] == binding.checkpoint_sha256
    ]
    if len(approvals) != 1:
        raise ValueError("checkpoint approval is missing or duplicated")
    approval = approvals[0]
    for field in (
        "checkpoint_sha256",
        "seed",
        "condition_id",
        "configuration_sha256",
        "route_dose_sha256",
    ):
        if approval[field] != getattr(binding, field):
            raise ValueError(f"study-lock checkpoint binding mismatch: {field}")
    return _CheckpointView(
        typed=typed, raw=MappingProxyType(dict(raw)), content=content
    )


def _parse_config(content: bytes) -> Mapping[str, Any]:
    try:
        raw = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        try:
            import yaml

            raw = yaml.safe_load(content)
        except Exception as exc:
            raise ValueError("configuration must be valid JSON or YAML") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("configuration must be an object")
    return raw


def _validate_run_files(run: Path, binding: RunBinding) -> None:
    checkpoint_path = _relative_file(
        run,
        binding.checkpoint_path,
        "checkpoint_path",
    )
    checkpoint_bytes = _read_regular_file(checkpoint_path, "checkpoint")
    if hashlib.sha256(checkpoint_bytes).hexdigest() != binding.checkpoint_sha256:
        raise ValueError("checkpoint file hash disagrees with run binding")
    configuration_path = _relative_file(
        run,
        binding.configuration_path,
        "configuration_path",
    )
    configuration_bytes = _read_regular_file(configuration_path, "configuration")
    if hashlib.sha256(configuration_bytes).hexdigest() != binding.configuration_sha256:
        raise ValueError("configuration file hash disagrees with run binding")
    configuration = _parse_config(configuration_bytes)
    if configuration.get("seed") != binding.seed:
        raise ValueError("configuration seed disagrees with run binding")
    if (
        _condition(
            configuration.get("condition"),
            "configuration condition",
        )
        != binding.condition_id
    ):
        raise ValueError("configuration condition disagrees with run binding")


def _selected_item_ids(
    lock: _StudyLockView,
    binding: RunBinding,
) -> tuple[str, ...]:
    selected = [
        cell["item_id"]
        for cell in lock.release["evaluation_cells"]
        if cell["checkpoint_sha256"] == binding.checkpoint_sha256
        and cell["seed"] == binding.seed
        and cell["condition_id"] == binding.condition_id
    ]
    if len(set(selected)) != len(selected):
        raise ValueError("study lock contains duplicate selected evaluation cells")
    if tuple(selected) != tuple(lock.release["item_ids"]):
        raise ValueError(
            "selected evaluation-cell registry is missing model-visible items"
        )
    return tuple(selected)


def _registered_solver(solver_id: str):
    registry = getattr(solver_module, "registered_solver", None)
    if callable(registry):
        return registry(solver_id)
    if solver_id != solver_module.LookupChainSolver.solver_id:
        raise ValueError(f"sealed gold references an unregistered solver: {solver_id}")
    return solver_module.LookupChainSolver()


def _submission_outcome(
    *,
    item: ItemRecord,
    checkpoint: _CheckpointView,
    submission: Submission,
    binding: RunBinding,
) -> dict[str, Any]:
    return {
        "record_type": getattr(metrics, "OUTCOME_SCHEMA", OUTCOME_SCHEMA),
        "schema_version": CONTRACT_VERSION,
        "item_id": item.item_id,
        "pair_id": item.pair_id,
        "twin": item.twin.value,
        "stratum": item.stratum.value,
        "family": item.family.value,
        "seed": binding.seed,
        "world_id": item.world_id,
        "checkpoint_sha256": binding.checkpoint_sha256,
        "arm": checkpoint.raw["arm"],
        "condition_id": binding.condition_id,
        "memory_mode": item.memory_mode.value,
        "control": item.control.value,
        "submitted_answer": submission.answer,
        "submitted_proof": [action.to_dict() for action in submission.actions],
    }


def _score_and_summarize(
    *,
    items: Mapping[str, ItemRecord],
    gold: Mapping[str, SealedGoldRecord],
    stores: Mapping[str, StoreRecord],
    checkpoint: _CheckpointView,
    binding: RunBinding,
    submissions: Mapping[str, Submission],
) -> tuple[bytes, bytes]:
    outcome_dicts = []
    scored = []
    hardened = callable(getattr(metrics, "score_item_outcome", None)) and hasattr(
        metrics.ItemOutcome,
        "from_dict",
    )
    for item_id in sorted(submissions):
        item = items[item_id]
        sealed = gold[item_id]
        store = stores[item.store_id]
        validate_contract_bundle(item, sealed, store, checkpoint.typed)
        solver = _registered_solver(sealed.solver_id)
        verification = solver_module.verify_proof_and_answer(
            item=item,
            store=store,
            gold=sealed,
            proof=submissions[item_id].actions,
            answer=submissions[item_id].answer,
            solver=solver,
        )
        outcome_dict = _submission_outcome(
            item=item,
            checkpoint=checkpoint,
            submission=submissions[item_id],
            binding=binding,
        )
        outcome_dicts.append(outcome_dict)
        if hardened:
            persisted = metrics.ItemOutcome.from_dict(outcome_dict)
            scored.append(
                metrics.score_item_outcome(
                    outcome=persisted,
                    item=item,
                    checkpoint=checkpoint.typed,
                    gold=sealed,
                    store=store,
                )
            )
        else:
            scored.append(
                metrics.ItemOutcome(
                    item_id=item.item_id,
                    pair_id=item.pair_id,
                    twin=item.twin,
                    stratum=item.stratum,
                    family=item.family,
                    seed=binding.seed,
                    world_id=item.world_id,
                    checkpoint_sha256=binding.checkpoint_sha256,
                    arm=checkpoint.typed.arm,
                    memory_mode=item.memory_mode,
                    control=item.control,
                    proof_valid=verification.proof_valid,
                    answer_valid=verification.answer_valid,
                    complete=True,
                    valid=True,
                )
            )

    grouped: dict[tuple[Any, Any], list[Any]] = defaultdict(list)
    for row in scored:
        grouped[(row.memory_mode, row.control)].append(row)
    summaries = []
    for key in sorted(grouped, key=lambda value: (value[0].value, value[1].value)):
        rows = grouped[key]
        row_items = {row.item_id: items[row.item_id] for row in rows}
        summary = metrics.balanced_counterfactual_pair_metric(
            rows,
            items=row_items,
            checkpoints={
                binding.checkpoint_sha256: checkpoint.typed,
            },
        ).to_dict()
        if "condition_id" not in summary:
            summary["condition_id"] = binding.condition_id
        summaries.append(summary)
    outcomes_bytes = b"".join(canonical_json_bytes(row) for row in outcome_dicts)
    metrics_bytes = canonical_json_bytes(
        {
            "record_type": getattr(metrics, "METRICS_SCHEMA", METRICS_SCHEMA),
            "schema_version": CONTRACT_VERSION,
            "summaries": summaries,
        }
    )
    return outcomes_bytes, metrics_bytes


def _inference_bytes() -> bytes:
    return canonical_json_bytes(
        {
            "record_type": getattr(
                reporting,
                "INFERENCE_EVIDENCE_SCHEMA",
                INFERENCE_EVIDENCE_SCHEMA,
            ),
            "schema_version": CONTRACT_VERSION,
            "primary_test": {
                "contrast_id": getattr(
                    reporting,
                    "PRIMARY_CONTRAST_ID",
                    PRIMARY_CONTRAST_ID,
                ),
                "method": getattr(
                    reporting,
                    "PRIMARY_TEST_METHOD",
                    PRIMARY_TEST_METHOD,
                ),
                "alternative": "greater",
                "alpha": 0.05,
                "n_pairs": 5,
                "sign_assignments": 32,
                "equality_counted": True,
            },
            "paired_seed_bundle_deltas": [],
            "exact_test_result": None,
        }
    )


def _require_hardened_reporting() -> None:
    required = set(getattr(reporting, "REQUIRED_ARTIFACTS", ()))
    build = getattr(reporting, "build_artifact_report", None)
    publish = getattr(reporting, "publish_artifact_report", None)
    build_parameters = inspect.signature(build).parameters if callable(build) else {}
    publish_parameters = (
        inspect.signature(publish).parameters if callable(publish) else {}
    )
    if (
        required != set(_HARDENED_ARTIFACTS)
        or "expected_study_lock_sha256" not in build_parameters
        or "expected_study_lock_sha256" not in publish_parameters
    ):
        raise ReportingInterfaceUnavailable(
            "hardened reporting interface unavailable; required interface: "
            "build_artifact_report(*, artifacts, expected_study_lock_sha256); "
            "publish_artifact_report(path, report, artifacts, *, "
            "expected_study_lock_sha256); REQUIRED_ARTIFACTS must include "
            "study-lock.json and validity.json. The current API cannot safely "
            "replay submission-only outcomes or derive inference."
        )


def _publish_file(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _publish_evidence(
    output: Path,
    artifacts: Mapping[str, bytes],
    report: Any,
    expected_study_lock_sha256: str,
) -> None:
    try:
        output.mkdir(mode=0o700)
    except FileExistsError:
        raise FileExistsError(f"output directory already exists: {output}") from None
    for name in _HARDENED_ARTIFACTS:
        _publish_file(output / name, artifacts[name])
    reporting.publish_artifact_report(
        output / "artifact-report.json",
        report,
        artifacts,
        expected_study_lock_sha256=expected_study_lock_sha256,
    )


def _prepare_evaluation(
    *,
    run: str | Path,
    sealed_release: str | Path,
    expected_study_lock_sha256: str,
) -> _PreparedEvaluation:
    release = _directory(sealed_release, "sealed release")
    run_root = _directory(run, "run")
    lock = _load_study_lock(release, expected_study_lock_sha256)
    binding = _load_run_binding(run_root)
    _validate_run_files(run_root, binding)
    checkpoint = _load_checkpoints(release, lock, binding)
    selected_ids = _selected_item_ids(lock, binding)
    items_content = _read_bound_artifact(
        release,
        "items.jsonl",
        lock.release["items_sha256"],
    )
    items_sequence = _canonical_jsonl(
        items_content,
        name="items.jsonl",
        parser=ItemRecord.from_dict,
        identity=lambda item: item.item_id,
    )
    items = {item.item_id: item for item in items_sequence}
    if tuple(items) != selected_ids:
        raise ValueError("model-visible item registry is missing or reordered")
    return _PreparedEvaluation(
        release=release,
        lock=lock,
        binding=binding,
        checkpoint=checkpoint,
        selected_ids=selected_ids,
        items_content=items_content,
        items=MappingProxyType(items),
    )


def preflight(
    *,
    run: str | Path,
    sealed_release: str | Path,
    expected_study_lock_sha256: str,
) -> PreflightResult:
    """Verify trust roots and model-visible registry without opening gold."""

    prepared = _prepare_evaluation(
        run=run,
        sealed_release=sealed_release,
        expected_study_lock_sha256=expected_study_lock_sha256,
    )
    return PreflightResult(
        study_lock_sha256=prepared.lock.sha256,
        checkpoint_sha256=prepared.binding.checkpoint_sha256,
        condition_id=prepared.binding.condition_id,
        seed=prepared.binding.seed,
        item_count=len(prepared.items),
    )


def evaluate(
    *,
    run: str | Path,
    sealed_release: str | Path,
    expected_study_lock_sha256: str,
    model_adapter: ModelAdapter,
    output_dir: str | Path,
) -> EvaluationResult:
    """Evaluate one hash-bound checkpoint without exposing sealed gold."""

    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output directory already exists: {output}")
    if not callable(getattr(model_adapter, "generate", None)):
        raise TypeError("model_adapter must expose generate(item)")

    prepared = _prepare_evaluation(
        run=run,
        sealed_release=sealed_release,
        expected_study_lock_sha256=expected_study_lock_sha256,
    )
    release = prepared.release
    lock = prepared.lock
    binding = prepared.binding
    checkpoint = prepared.checkpoint
    selected_ids = prepared.selected_ids
    items_content = prepared.items_content
    items = prepared.items

    submissions: dict[str, Submission] = {}
    for item_id in selected_ids:
        submission = model_adapter.generate(items[item_id])
        if not isinstance(submission, Submission):
            raise ValueError("ModelAdapter.generate must return a Submission")
        if submission.item_id != item_id:
            if submission.item_id in submissions:
                raise ValueError("model adapter returned a duplicate submission")
            raise ValueError("model adapter returned a submission item mismatch")
        if item_id in submissions:
            raise ValueError("model adapter returned a duplicate submission")
        submissions[item_id] = submission
    if tuple(submissions) != selected_ids:
        raise ValueError("model adapter omitted a required item submission")

    gold_content = _read_bound_artifact(
        release,
        "sealed-gold.jsonl",
        lock.release["sealed_gold_sha256"],
    )
    gold_sequence = _canonical_jsonl(
        gold_content,
        name="sealed-gold.jsonl",
        parser=SealedGoldRecord.from_dict,
        identity=lambda record: record.item_id,
    )
    gold = {record.item_id: record for record in gold_sequence}
    if tuple(gold) != selected_ids:
        raise ValueError("sealed-gold registry disagrees with model-visible items")
    stores_content = _read_bound_artifact(
        release,
        "stores.jsonl",
        lock.release["stores_sha256"],
    )
    stores_sequence = _canonical_jsonl(
        stores_content,
        name="stores.jsonl",
        parser=StoreRecord.from_dict,
        identity=lambda record: record.store_id,
    )
    stores = {record.store_id: record for record in stores_sequence}
    if set(stores) != {item.store_id for item in items.values()}:
        raise ValueError("sealed store registry is not exactly item-bound")

    outcomes_content, metrics_content = _score_and_summarize(
        items=items,
        gold=gold,
        stores=stores,
        checkpoint=checkpoint,
        binding=binding,
        submissions=submissions,
    )
    validity_content = _read_regular_file(
        release / "validity.json",
        "validity.json",
    )
    validity = _canonical_object(validity_content, "validity.json")
    if validity.get("record_type") != VALIDITY_EVIDENCE_SCHEMA:
        raise ValueError("validity.json record_type is invalid")
    if validity.get("study_lock_sha256") != lock.sha256:
        raise ValueError("validity evidence is unbound from study lock")
    artifacts = MappingProxyType(
        {
            "checkpoints.jsonl": checkpoint.content,
            "inference.json": _inference_bytes(),
            "items.jsonl": items_content,
            "metrics.json": metrics_content,
            "outcomes.jsonl": outcomes_content,
            "sealed-gold.jsonl": gold_content,
            "study-lock.json": lock.content,
            "stores.jsonl": stores_content,
            "validity.json": validity_content,
        }
    )

    _require_hardened_reporting()
    report = reporting.build_artifact_report(
        artifacts=artifacts,
        expected_study_lock_sha256=lock.sha256,
    )
    _publish_evidence(output, artifacts, report, lock.sha256)
    report_hash = getattr(report, "report_sha256", None)
    if not isinstance(report_hash, str) or _SHA256_RE.fullmatch(report_hash) is None:
        report_hash = canonical_sha256(report)
    return EvaluationResult(
        output_dir=output,
        study_lock_sha256=lock.sha256,
        checkpoint_sha256=binding.checkpoint_sha256,
        condition_id=binding.condition_id,
        seed=binding.seed,
        item_count=len(items),
        report_sha256=report_hash,
    )
