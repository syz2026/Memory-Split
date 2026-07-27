"""Frozen checkpoint decoding, exact scoring, and evaluator-only publication."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import io
import math
import os
import platform
import stat
import unicodedata
from contextlib import nullcontext
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from corpusgen.parallel.canonical import canonical_json_bytes
from evals.reasoning_v3.aws_authority import (
    ACTIVATION_KEY,
    STORAGE_BUCKET,
    TASK3_CHECKPOINT_MANIFEST_KEY,
    TASK3_CHECKPOINT_PREFIX,
    TASK3_EVIDENCE_PREFIX,
    TASK3_RESULT_PREFIX,
    S3ObjectVersion,
    Task3Publication,
    VerifiedAwsAuthority,
)
from evals.reasoning_v3.inference import _fraction_dict
from evals.reasoning_v3.sealing import (
    ReleaseBundle,
    _load_authorized_contract,
    _new_evaluator_aws_authority,
    _parse_canonical_bytes,
    _read_signed_activation,
    _validate_object_payload,
    _validate_release_bundle,
)
from msctl.reasoning_cohort import (
    ARMS,
    CHECKPOINT_UPDATES,
    COHORT_ID,
    MODEL_PARAMETERS,
    PROVIDER,
    SEEDS,
    config_path,
    pair_id,
    run_id,
)


CHECKPOINT_STEPS = CHECKPOINT_UPDATES
TERMINAL_PRIMARY_STEP = CHECKPOINT_STEPS[-1]
MODEL_ID = "d135m"
ROOT = Path(__file__).resolve().parents[2]
RESULT_PREFIX = TASK3_RESULT_PREFIX.rstrip("/")
SCORING_FORMAT = "memorysplit-reasoning-v3-checkpoint-result-v1"
CHECKPOINT_MANIFEST_FORMAT = (
    "memorysplit-reasoning-v3-checkpoint-evidence-manifest-v1"
)
RAW_EVIDENCE_FORMAT = "memorysplit-reasoning-v3-raw-validity-evidence-v1"
RUNTIME_LOCK_FORMAT = "memorysplit-reasoning-v3-task3-runtime-lock-v2"
PAIRING_EVIDENCE_FORMAT = "memorysplit-reasoning-v3-pairing-evidence-v1"
RUN_CONFIG_EVIDENCE_FORMAT = "memorysplit-reasoning-v3-run-config-evidence-v1"
CORPUS_EVIDENCE_FORMAT = "memorysplit-reasoning-v3-corpus-evidence-v1"
INFERENCE_BATCH_SIZE = 16
REQUIRED_VALIDITY_GATES = (
    "factual_burden",
    "memory_on_recall",
    "memory_off_leakage",
    "exact_resume",
    "evaluator_authority",
    "complete_registry",
    "no_substitution",
    "no_outcome_dependent_stopping",
    "no_replacement",
    "no_exclusion",
    "provider_fixed",
    "no_missing_seed_imputation",
)
_TASK3_EVALUATOR_PATHS = (
    "evals/reasoning_v3/runner.py",
    "evals/reasoning_v3/inference.py",
    "evals/reasoning_v3/reporting.py",
)
_RUNTIME_MODULE_PATHS = (
    (
        "corpusgen.parallel.canonical",
        "corpusgen/parallel/canonical.py",
    ),
    (
        "corpusgen.reasoning_expansion",
        "corpusgen/reasoning_expansion.py",
    ),
    ("evals.reasoning_v3.aws_authority", "evals/reasoning_v3/aws_authority.py"),
    ("evals.reasoning_v3.contracts", "evals/reasoning_v3/contracts.py"),
    ("evals.reasoning_v3.generate", "evals/reasoning_v3/generate.py"),
    ("evals.reasoning_v3.inference", "evals/reasoning_v3/inference.py"),
    ("evals.reasoning_v3.reporting", "evals/reasoning_v3/reporting.py"),
    ("evals.reasoning_v3.runner", "evals/reasoning_v3/runner.py"),
    ("evals.reasoning_v3.sealing", "evals/reasoning_v3/sealing.py"),
    ("msctl.reasoning_cohort", "msctl/reasoning_cohort.py"),
    ("train.model", "train/model.py"),
    ("train.tokenizer", "train/tokenizer.py"),
)
_PAIRING_KINDS = (
    "config",
    "corpus",
    "data_order",
    "initialization",
    "runtime",
)
_HEX = frozenset("0123456789abcdef")
_ITEM_FIELDS = {
    "correct",
    "error",
    "generated_tokens",
    "item_id",
    "max_new_tokens",
    "position",
    "prediction",
    "prediction_sha256",
    "raw_prediction",
    "source_index",
    "status",
    "stop_reason",
    "task",
    "valid",
}
_TOP_FIELDS = {
    "checkpoint",
    "cohort",
    "evaluator",
    "family_scores",
    "format",
    "items",
    "macro_accuracy",
    "macro_accuracy_fraction",
    "release",
    "run",
    "schema_version",
    "validity",
}
_CHECKPOINT_FIELDS = {
    "bytes",
    "corpus_receipt_key",
    "corpus_receipt_object_sha256",
    "corpus_receipt_version_id",
    "evidence_key",
    "evidence_sha256",
    "evidence_version_id",
    "manifest_key",
    "manifest_sha256",
    "manifest_version_id",
    "object_key",
    "runtime_lock_key",
    "runtime_lock_object_sha256",
    "runtime_lock_version_id",
    "sha256",
    "step",
    "version_id",
}
_COHORT_FIELDS = {
    "arms",
    "checkpoint_steps",
    "cohort_id",
    "model",
    "model_parameters",
    "seeds",
    "terminal_primary_step",
}
_EVALUATOR_FIELDS = {"code", "code_sha256", "scorer_id", "scoring"}
_SCORING_FIELDS = {
    "comparison",
    "decoding",
    "output_encoding",
    "output_normalization",
    "stop_token",
    "temperature",
    "whitespace_policy",
}
_RELEASE_FIELDS = {
    "activation_key",
    "activation_sha256",
    "activation_signature_sha256",
    "activation_signing_algorithm",
    "activation_version_id",
    "contract_authority_record_sha256",
    "contract_authority_record_version_id",
    "contract_authority_signature_version_id",
    "contract_id",
    "contract_sha256",
    "corpus_receipt_sha256",
    "model_visible_key",
    "model_visible_sha256",
    "model_visible_version_id",
    "registry_sha256",
    "runtime_lock_sha256",
    "sealed_gold_key",
    "sealed_gold_sha256",
    "sealed_gold_version_id",
    "signer_key_arn",
}
_RUN_FIELDS = {
    "arm",
    "config_path",
    "config_sha256",
    "pair_id",
    "pairing",
    "provider",
    "run_id",
    "seed",
}
_PAIRING_FIELDS = {
    "config_sha256",
    "corpus_sha256",
    "data_order_sha256",
    "initialization_sha256",
    "runtime_sha256",
}
_MODEL_CONFIG = {
    "ctx": 1024,
    "d_model": 720,
    "n_head": 12,
    "n_layer": 10,
    "rope_base": 10000.0,
    "vocab_size": 50304,
}


class RunnerError(ValueError):
    """A checkpoint, prediction, score, or publication violated the frozen rules."""


@dataclass(frozen=True)
class GateEvidence:
    passed: bool
    evidence_sha256: str


@dataclass(frozen=True)
class CodeBinding:
    path: str
    sha256: str


@dataclass(frozen=True)
class RawPrediction:
    item_id: str
    task: str
    source_index: int
    raw_prediction: str | None
    generated_tokens: int
    stop_reason: str


@dataclass(frozen=True)
class ExactOutputScore:
    raw_prediction: str | None
    prediction: str | None
    prediction_sha256: str | None
    valid: bool
    error: str | None
    correct: bool


@dataclass(frozen=True)
class CheckpointBinding:
    arm: str
    seed: int
    step: int
    sha256: str
    object_key: str
    version_id: str
    bytes: int
    run_config_path: str
    run_config_sha256: str
    initialization_sha256: str
    data_order_sha256: str
    paired_runtime_sha256: str
    paired_corpus_sha256: str
    paired_config_sha256: str
    checkpoint_kms_key_arn: str
    manifest_key: str
    manifest_version_id: str
    manifest_sha256: str
    evidence_key: str
    evidence_version_id: str
    evidence_sha256: str
    runtime_lock_key: str
    runtime_lock_version_id: str
    runtime_lock_object_sha256: str
    corpus_receipt_key: str
    corpus_receipt_version_id: str
    corpus_receipt_object_sha256: str


@dataclass(frozen=True)
class ReleaseBinding:
    contract_id: str
    contract_sha256: str
    authority: VerifiedAwsAuthority
    activation: S3ObjectVersion
    activation_sha256: str
    activation_signature_sha256: str
    activation_signing_algorithm: str
    model_visible: S3ObjectVersion
    sealed_gold: S3ObjectVersion
    registry_sha256: str
    corpus_receipt_sha256: str
    runtime_lock_sha256: str


@dataclass(frozen=True)
class _ValidatedCheckpoint:
    binding: CheckpointBinding
    validity: Mapping[str, GateEvidence]
    state: Mapping[str, Any]


@dataclass(frozen=True)
class ManifestCell:
    arm: str
    seed: int
    step: int
    checkpoint: S3ObjectVersion
    run_config: S3ObjectVersion
    evidence: S3ObjectVersion
    pairing: Mapping[str, S3ObjectVersion]


@dataclass(frozen=True)
class CheckpointManifest:
    ref: S3ObjectVersion
    runtime_lock: S3ObjectVersion
    corpus_receipt: S3ObjectVersion
    cells: Mapping[tuple[str, int, int], ManifestCell]


@dataclass(frozen=True)
class EvidenceContext:
    checkpoint: CheckpointBinding
    expected_item_count: int
    manifest_complete: bool
    registry_sha256: str


@dataclass(frozen=True)
class PublishedCheckpointResult:
    result: Mapping[str, Any]
    object_ref: S3ObjectVersion


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _HEX
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise RunnerError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RunnerError(f"{label} must be a positive integer")
    return value


def _exact_mapping(
    value: object,
    fields: set[str] | frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise RunnerError(f"{label} fields differ")
    return value


def _score_exact_output(
    raw_prediction: object,
    canonical_answer: str,
    *,
    max_new_tokens: int,
    generated_tokens: int,
    stop_reason: str,
) -> ExactOutputScore:
    """Apply UTF-8/NFC/outer-strip exact scoring without answer heuristics."""

    if (
        not isinstance(canonical_answer, str)
        or not canonical_answer
        or canonical_answer != canonical_answer.strip()
        or unicodedata.normalize("NFC", canonical_answer) != canonical_answer
    ):
        raise RunnerError("canonical answer is malformed")
    limit = _positive_int(max_new_tokens, "max_new_tokens")
    if (
        isinstance(generated_tokens, bool)
        or not isinstance(generated_tokens, int)
        or generated_tokens < 0
        or stop_reason not in {"eot", "max_new_tokens", "error"}
    ):
        raise RunnerError("prediction generation metadata is malformed")
    if raw_prediction is None:
        return ExactOutputScore(None, None, None, False, "missing", False)
    if not isinstance(raw_prediction, str):
        return ExactOutputScore(None, None, None, False, "malformed", False)
    try:
        encoded = raw_prediction.encode("utf-8")
    except UnicodeEncodeError:
        return ExactOutputScore(
            raw_prediction,
            None,
            None,
            False,
            "encoding",
            False,
        )
    digest = hashlib.sha256(encoded).hexdigest()
    prediction = unicodedata.normalize("NFC", raw_prediction).strip()
    if (
        stop_reason == "max_new_tokens"
        or generated_tokens > limit
        or (generated_tokens == limit and stop_reason != "eot")
    ):
        return ExactOutputScore(
            raw_prediction,
            prediction,
            digest,
            False,
            "over_limit",
            False,
        )
    if stop_reason == "error":
        return ExactOutputScore(
            raw_prediction,
            prediction,
            digest,
            False,
            "generation_error",
            False,
        )
    if "<|eot|>" in raw_prediction:
        return ExactOutputScore(
            raw_prediction,
            prediction,
            digest,
            False,
            "malformed",
            False,
        )
    return ExactOutputScore(
        raw_prediction=raw_prediction,
        prediction=prediction,
        prediction_sha256=digest,
        valid=True,
        error=None,
        correct=prediction == canonical_answer,
    )


def _validate_gate_evidence(
    validity: Mapping[str, GateEvidence],
) -> dict[str, Any]:
    if not isinstance(validity, Mapping) or set(validity) != set(
        REQUIRED_VALIDITY_GATES
    ):
        raise RunnerError("validity gate set differs")
    gates: dict[str, Any] = {}
    for name in REQUIRED_VALIDITY_GATES:
        evidence = validity[name]
        if (
            not isinstance(evidence, GateEvidence)
            or not isinstance(evidence.passed, bool)
            or not _is_sha256(evidence.evidence_sha256)
        ):
            raise RunnerError(f"validity evidence is malformed: {name}")
        gates[name] = {
            "evidence_sha256": evidence.evidence_sha256,
            "passed": evidence.passed,
        }
    return {"gates": gates, "passed": all(item.passed for item in validity.values())}


def _validate_checkpoint_binding(binding: CheckpointBinding) -> None:
    if not isinstance(binding, CheckpointBinding):
        raise RunnerError("checkpoint binding is malformed")
    if (
        binding.arm not in ARMS
        or isinstance(binding.seed, bool)
        or binding.seed not in SEEDS
        or isinstance(binding.step, bool)
        or binding.step not in CHECKPOINT_STEPS
        or binding.run_config_path != config_path(binding.arm, binding.seed)
        or binding.object_key
        != (
            f"{TASK3_CHECKPOINT_PREFIX}{run_id(binding.arm, binding.seed)}/"
            f"step{binding.step:07d}.pt"
        )
        or binding.manifest_key != TASK3_CHECKPOINT_MANIFEST_KEY
        or binding.evidence_key
        != (
            f"{TASK3_EVIDENCE_PREFIX}{run_id(binding.arm, binding.seed)}/"
            f"step{binding.step:07d}/gates.json"
        )
        or binding.runtime_lock_key
        != f"{TASK3_EVIDENCE_PREFIX}runtime/runtime-lock.json"
        or binding.corpus_receipt_key
        != f"{TASK3_EVIDENCE_PREFIX}corpus/receipt.json"
        or not isinstance(binding.version_id, str)
        or not binding.version_id
        or not isinstance(binding.checkpoint_kms_key_arn, str)
        or not binding.checkpoint_kms_key_arn
        or isinstance(binding.bytes, bool)
        or not isinstance(binding.bytes, int)
        or binding.bytes <= 0
    ):
        raise RunnerError("checkpoint binding identity differs")
    for label, value in (
        ("checkpoint", binding.sha256),
        ("run config", binding.run_config_sha256),
        ("initialization", binding.initialization_sha256),
        ("data order", binding.data_order_sha256),
        ("paired runtime", binding.paired_runtime_sha256),
        ("paired corpus", binding.paired_corpus_sha256),
        ("paired config", binding.paired_config_sha256),
        ("manifest", binding.manifest_sha256),
        ("evidence", binding.evidence_sha256),
        ("runtime lock object", binding.runtime_lock_object_sha256),
        ("corpus receipt object", binding.corpus_receipt_object_sha256),
    ):
        _require_sha256(value, label)
    for label, value in (
        ("manifest", binding.manifest_version_id),
        ("evidence", binding.evidence_version_id),
        ("runtime lock", binding.runtime_lock_version_id),
        ("corpus receipt", binding.corpus_receipt_version_id),
    ):
        if not isinstance(value, str) or not value:
            raise RunnerError(f"{label} exact object version is missing")


def _validate_s3_ref(ref: S3ObjectVersion, label: str) -> None:
    if (
        not isinstance(ref, S3ObjectVersion)
        or ref.bucket != STORAGE_BUCKET
        or not isinstance(ref.key, str)
        or not ref.key
        or not isinstance(ref.version_id, str)
        or not ref.version_id
        or isinstance(ref.bytes, bool)
        or not isinstance(ref.bytes, int)
        or ref.bytes <= 0
        or not _is_sha256(ref.sha256)
        or ref.server_side_encryption not in {"AES256", "aws:kms"}
    ):
        raise RunnerError(f"{label} object reference is malformed")


def _validate_release_binding(release: ReleaseBinding) -> None:
    if not isinstance(release, ReleaseBinding):
        raise RunnerError("release binding is malformed")
    for ref, label in (
        (release.activation, "activation"),
        (release.model_visible, "model-visible"),
        (release.sealed_gold, "sealed-gold"),
    ):
        _validate_s3_ref(ref, label)
    if (
        release.contract_id != "memorysplit-reasoning-v3-eval-v1"
        or release.contract_sha256 != release.authority.contract_sha256
        or release.activation.key != ACTIVATION_KEY
        or release.activation_sha256 != release.activation.sha256
        or release.activation_signing_algorithm != "ECDSA_SHA_384"
        or release.sealed_gold.server_side_encryption != "aws:kms"
        or release.sealed_gold.kms_key_arn
        != release.authority.sealed_gold_kms_key_arn
        or not isinstance(release.authority.checkpoint_kms_key_arn, str)
        or not release.authority.checkpoint_kms_key_arn
        or release.authority.checkpoint_kms_key_arn
        == release.authority.sealed_gold_kms_key_arn
        or release.model_visible.server_side_encryption != "AES256"
    ):
        raise RunnerError("release authority binding differs")
    for label, value in (
        ("contract", release.contract_sha256),
        ("authority record", release.authority.record_sha256),
        ("activation", release.activation_sha256),
        ("activation signature", release.activation_signature_sha256),
        ("registry", release.registry_sha256),
        ("corpus receipt", release.corpus_receipt_sha256),
        ("runtime lock", release.runtime_lock_sha256),
    ):
        _require_sha256(value, label)
    if (
        not release.authority.record_version_id
        or not release.authority.signature_version_id
        or not release.authority.signer_key_arn
    ):
        raise RunnerError("release authority versions are missing")


def _validate_code_bindings(
    evaluator_code: Sequence[CodeBinding],
) -> tuple[list[dict[str, str]], str]:
    if (
        not isinstance(evaluator_code, Sequence)
        or isinstance(evaluator_code, (str, bytes, bytearray))
        or not evaluator_code
    ):
        raise RunnerError("evaluator code bindings are missing")
    rows = []
    seen = set()
    for binding in evaluator_code:
        if (
            not isinstance(binding, CodeBinding)
            or not isinstance(binding.path, str)
            or not binding.path
            or binding.path in seen
            or not _is_sha256(binding.sha256)
        ):
            raise RunnerError("evaluator code binding is malformed")
        seen.add(binding.path)
        rows.append(asdict(binding))
    return rows, hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def _release_dict(release: ReleaseBinding) -> dict[str, Any]:
    return {
        "activation_key": release.activation.key,
        "activation_sha256": release.activation_sha256,
        "activation_signature_sha256": release.activation_signature_sha256,
        "activation_signing_algorithm": release.activation_signing_algorithm,
        "activation_version_id": release.activation.version_id,
        "contract_authority_record_sha256": release.authority.record_sha256,
        "contract_authority_record_version_id": (
            release.authority.record_version_id
        ),
        "contract_authority_signature_version_id": (
            release.authority.signature_version_id
        ),
        "contract_id": release.contract_id,
        "contract_sha256": release.contract_sha256,
        "corpus_receipt_sha256": release.corpus_receipt_sha256,
        "model_visible_key": release.model_visible.key,
        "model_visible_sha256": release.model_visible.sha256,
        "model_visible_version_id": release.model_visible.version_id,
        "registry_sha256": release.registry_sha256,
        "runtime_lock_sha256": release.runtime_lock_sha256,
        "sealed_gold_key": release.sealed_gold.key,
        "sealed_gold_sha256": release.sealed_gold.sha256,
        "sealed_gold_version_id": release.sealed_gold.version_id,
        "signer_key_arn": release.authority.signer_key_arn,
    }


def _checkpoint_dict(binding: CheckpointBinding) -> dict[str, Any]:
    return {
        "bytes": binding.bytes,
        "corpus_receipt_key": binding.corpus_receipt_key,
        "corpus_receipt_object_sha256": binding.corpus_receipt_object_sha256,
        "corpus_receipt_version_id": binding.corpus_receipt_version_id,
        "evidence_key": binding.evidence_key,
        "evidence_sha256": binding.evidence_sha256,
        "evidence_version_id": binding.evidence_version_id,
        "manifest_key": binding.manifest_key,
        "manifest_sha256": binding.manifest_sha256,
        "manifest_version_id": binding.manifest_version_id,
        "object_key": binding.object_key,
        "runtime_lock_key": binding.runtime_lock_key,
        "runtime_lock_object_sha256": binding.runtime_lock_object_sha256,
        "runtime_lock_version_id": binding.runtime_lock_version_id,
        "sha256": binding.sha256,
        "step": binding.step,
        "version_id": binding.version_id,
    }


def _run_dict(binding: CheckpointBinding) -> dict[str, Any]:
    return {
        "arm": binding.arm,
        "config_path": binding.run_config_path,
        "config_sha256": binding.run_config_sha256,
        "pair_id": pair_id(binding.seed),
        "pairing": {
            "config_sha256": binding.paired_config_sha256,
            "corpus_sha256": binding.paired_corpus_sha256,
            "data_order_sha256": binding.data_order_sha256,
            "initialization_sha256": binding.initialization_sha256,
            "runtime_sha256": binding.paired_runtime_sha256,
        },
        "provider": PROVIDER,
        "run_id": run_id(binding.arm, binding.seed),
        "seed": binding.seed,
    }


def _build_checkpoint_result(
    *,
    public_release: Mapping[str, Any],
    sealed_release: Mapping[str, Any],
    predictions: Sequence[RawPrediction],
    checkpoint: CheckpointBinding,
    release: ReleaseBinding,
    validity: Mapping[str, GateEvidence],
    evaluator_code: Sequence[CodeBinding],
    family_order: Sequence[str],
    items_per_family: int,
    scorer_id: str,
) -> dict[str, Any]:
    """Private pure scorer; production supplies only authenticated release bytes."""

    _validate_checkpoint_binding(checkpoint)
    _validate_release_binding(release)
    validity_value = _validate_gate_evidence(validity)
    code_rows, code_sha256 = _validate_code_bindings(evaluator_code)
    if (
        not isinstance(family_order, Sequence)
        or isinstance(family_order, (str, bytes, bytearray))
        or not family_order
        or len(family_order) != len(set(family_order))
        or any(not isinstance(name, str) or not name for name in family_order)
    ):
        raise RunnerError("family order is malformed")
    family_names = tuple(family_order)
    count = _positive_int(items_per_family, "items_per_family")
    if not isinstance(scorer_id, str) or not scorer_id:
        raise RunnerError("scorer identity is malformed")
    if not isinstance(public_release, Mapping) or not isinstance(
        sealed_release, Mapping
    ):
        raise RunnerError("evaluation releases must be mappings")
    public_items = public_release.get("items")
    sealed_items = sealed_release.get("items")
    expected_count = len(family_names) * count
    if (
        public_release.get("registry_sha256") != release.registry_sha256
        or sealed_release.get("registry_sha256") != release.registry_sha256
        or not isinstance(public_items, list)
        or not isinstance(sealed_items, list)
        or len(public_items) != expected_count
        or len(sealed_items) != expected_count
        or not isinstance(predictions, Sequence)
        or isinstance(predictions, (str, bytes, bytearray))
        or len(predictions) != expected_count
    ):
        raise RunnerError("registry, gold, or prediction count differs")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, (visible, gold, raw) in enumerate(
        zip(public_items, sealed_items, predictions, strict=True)
    ):
        if (
            not isinstance(visible, Mapping)
            or not isinstance(gold, Mapping)
            or not isinstance(raw, RawPrediction)
        ):
            raise RunnerError(f"checkpoint item {position} is malformed")
        family = family_names[position // count]
        item_id = visible.get("item_id")
        source_index = visible.get("source_index")
        max_new_tokens = visible.get("max_new_tokens")
        if (
            visible.get("task") != family
            or visible.get("scorer_id") != scorer_id
            or gold.get("item_id") != item_id
            or gold.get("task") != family
            or gold.get("source_index") != source_index
            or raw.item_id != item_id
            or raw.task != family
            or raw.source_index != source_index
            or not isinstance(item_id, str)
            or not item_id
            or item_id in seen
            or isinstance(source_index, bool)
            or not isinstance(source_index, int)
        ):
            raise RunnerError(f"checkpoint item identity differs at {position}")
        seen.add(item_id)
        score = _score_exact_output(
            raw.raw_prediction,
            gold.get("canonical_answer"),
            max_new_tokens=max_new_tokens,
            generated_tokens=raw.generated_tokens,
            stop_reason=raw.stop_reason,
        )
        rows.append(
            {
                "correct": score.correct,
                "error": score.error,
                "generated_tokens": raw.generated_tokens,
                "item_id": item_id,
                "max_new_tokens": max_new_tokens,
                "position": position,
                "prediction": score.prediction,
                "prediction_sha256": score.prediction_sha256,
                "raw_prediction": score.raw_prediction,
                "source_index": source_index,
                "status": (
                    "correct"
                    if score.correct
                    else "incorrect"
                    if score.valid
                    else "invalid"
                ),
                "stop_reason": raw.stop_reason,
                "task": family,
                "valid": score.valid,
            }
        )

    family_scores = []
    family_fractions: list[Fraction] = []
    for family_index, family in enumerate(family_names):
        family_rows = rows[family_index * count : (family_index + 1) * count]
        numerator = sum(int(row["correct"]) for row in family_rows)
        accuracy = Fraction(numerator, count)
        family_fractions.append(accuracy)
        family_scores.append(
            {
                "accuracy": float(accuracy),
                "accuracy_fraction": _fraction_dict(accuracy),
                "denominator": count,
                "numerator": numerator,
                "task": family,
            }
        )
    macro = sum(family_fractions, start=Fraction()) / len(family_fractions)
    result = {
        "checkpoint": _checkpoint_dict(checkpoint),
        "cohort": {
            "arms": list(ARMS),
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "cohort_id": COHORT_ID,
            "model": MODEL_ID,
            "model_parameters": MODEL_PARAMETERS,
            "seeds": list(SEEDS),
            "terminal_primary_step": TERMINAL_PRIMARY_STEP,
        },
        "evaluator": {
            "code": code_rows,
            "code_sha256": code_sha256,
            "scorer_id": scorer_id,
            "scoring": {
                "comparison": "exact",
                "decoding": "greedy",
                "output_encoding": "UTF-8",
                "output_normalization": "NFC",
                "stop_token": "<|eot|>",
                "temperature": 0,
                "whitespace_policy": "strip_outer_only",
            },
        },
        "family_scores": family_scores,
        "format": SCORING_FORMAT,
        "items": rows,
        "macro_accuracy": float(macro),
        "macro_accuracy_fraction": _fraction_dict(macro),
        "release": _release_dict(release),
        "run": _run_dict(checkpoint),
        "schema_version": 1,
        "validity": validity_value,
    }
    _validate_checkpoint_result(
        result,
        public_release=public_release,
        sealed_release=sealed_release,
        family_order=family_names,
        items_per_family=count,
        scorer_id=scorer_id,
    )
    return result


def _validate_checkpoint_result(
    result: Mapping[str, Any],
    *,
    public_release: Mapping[str, Any],
    sealed_release: Mapping[str, Any],
    family_order: Sequence[str],
    items_per_family: int,
    scorer_id: str,
) -> None:
    """Replay every correctness bit and aggregate against exact sealed gold."""

    if not isinstance(result, Mapping) or set(result) != _TOP_FIELDS:
        raise RunnerError("checkpoint result fields differ")
    if (
        result["schema_version"] != 1
        or isinstance(result["schema_version"], bool)
        or result["format"] != SCORING_FORMAT
    ):
        raise RunnerError("checkpoint result identity differs")
    checkpoint = _exact_mapping(
        result["checkpoint"],
        _CHECKPOINT_FIELDS,
        "checkpoint result checkpoint",
    )
    cohort = _exact_mapping(
        result["cohort"],
        _COHORT_FIELDS,
        "checkpoint result cohort",
    )
    evaluator = _exact_mapping(
        result["evaluator"],
        _EVALUATOR_FIELDS,
        "checkpoint result evaluator",
    )
    release = _exact_mapping(
        result["release"],
        _RELEASE_FIELDS,
        "checkpoint result release",
    )
    run = _exact_mapping(
        result["run"],
        _RUN_FIELDS,
        "checkpoint result run",
    )
    pairing = _exact_mapping(
        run["pairing"],
        _PAIRING_FIELDS,
        "checkpoint result pairing",
    )
    scoring = _exact_mapping(
        evaluator["scoring"],
        _SCORING_FIELDS,
        "checkpoint result scoring",
    )
    if cohort != {
        "arms": list(ARMS),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "cohort_id": COHORT_ID,
        "model": MODEL_ID,
        "model_parameters": MODEL_PARAMETERS,
        "seeds": list(SEEDS),
        "terminal_primary_step": TERMINAL_PRIMARY_STEP,
    }:
        raise RunnerError("checkpoint result cohort identity differs")
    arm, seed, step = run["arm"], run["seed"], checkpoint["step"]
    if (
        arm not in ARMS
        or isinstance(seed, bool)
        or seed not in SEEDS
        or isinstance(step, bool)
        or step not in CHECKPOINT_STEPS
        or run["config_path"] != config_path(arm, seed)
        or run["pair_id"] != pair_id(seed)
        or run["provider"] != PROVIDER
        or run["run_id"] != run_id(arm, seed)
        or checkpoint["object_key"]
        != (
            f"{TASK3_CHECKPOINT_PREFIX}{run_id(arm, seed)}/"
            f"step{step:07d}.pt"
        )
        or checkpoint["manifest_key"] != TASK3_CHECKPOINT_MANIFEST_KEY
        or checkpoint["evidence_key"]
        != (
            f"{TASK3_EVIDENCE_PREFIX}{run_id(arm, seed)}/"
            f"step{step:07d}/gates.json"
        )
        or checkpoint["runtime_lock_key"]
        != f"{TASK3_EVIDENCE_PREFIX}runtime/runtime-lock.json"
        or checkpoint["corpus_receipt_key"]
        != f"{TASK3_EVIDENCE_PREFIX}corpus/receipt.json"
        or not isinstance(checkpoint["version_id"], str)
        or not checkpoint["version_id"]
        or isinstance(checkpoint["bytes"], bool)
        or not isinstance(checkpoint["bytes"], int)
        or checkpoint["bytes"] <= 0
    ):
        raise RunnerError("checkpoint result run/checkpoint identity differs")
    for label, value in (
        ("checkpoint", checkpoint["sha256"]),
        ("run config", run["config_sha256"]),
        ("manifest", checkpoint["manifest_sha256"]),
        ("evidence", checkpoint["evidence_sha256"]),
        ("runtime lock object", checkpoint["runtime_lock_object_sha256"]),
        ("corpus receipt object", checkpoint["corpus_receipt_object_sha256"]),
        *(
            (f"pairing {name}", value)
            for name, value in pairing.items()
        ),
    ):
        _require_sha256(value, label)
    for label, value in (
        ("checkpoint", checkpoint["version_id"]),
        ("manifest", checkpoint["manifest_version_id"]),
        ("evidence", checkpoint["evidence_version_id"]),
        ("runtime lock", checkpoint["runtime_lock_version_id"]),
        ("corpus receipt", checkpoint["corpus_receipt_version_id"]),
    ):
        if not isinstance(value, str) or not value:
            raise RunnerError(f"checkpoint result {label} version differs")
    if (
        release["activation_key"] != ACTIVATION_KEY
        or release["activation_signing_algorithm"] != "ECDSA_SHA_384"
        or release["contract_id"] != "memorysplit-reasoning-v3-eval-v1"
        or release["registry_sha256"]
        != public_release.get("registry_sha256")
        or release["registry_sha256"]
        != sealed_release.get("registry_sha256")
        or not isinstance(release["activation_version_id"], str)
        or not release["activation_version_id"]
        or not isinstance(release["model_visible_version_id"], str)
        or not release["model_visible_version_id"]
        or not isinstance(release["sealed_gold_version_id"], str)
        or not release["sealed_gold_version_id"]
        or not isinstance(release["signer_key_arn"], str)
        or not release["signer_key_arn"]
        or not isinstance(release["model_visible_key"], str)
        or "/model-visible/" not in f"/{release['model_visible_key']}"
        or not isinstance(release["sealed_gold_key"], str)
        or "/evaluator-only/" not in f"/{release['sealed_gold_key']}"
    ):
        raise RunnerError("checkpoint result release identity differs")
    for name, value in release.items():
        if name.endswith("sha256"):
            _require_sha256(value, f"release {name}")
    code = evaluator["code"]
    if not isinstance(code, list) or not code:
        raise RunnerError("checkpoint result evaluator code fields differ")
    code_paths = set()
    for raw_binding in code:
        binding = _exact_mapping(
            raw_binding,
            {"path", "sha256"},
            "checkpoint result evaluator code",
        )
        if (
            not isinstance(binding["path"], str)
            or not binding["path"]
            or binding["path"] in code_paths
            or not _is_sha256(binding["sha256"])
        ):
            raise RunnerError("checkpoint result evaluator code identity differs")
        code_paths.add(binding["path"])
    if (
        evaluator["code_sha256"]
        != hashlib.sha256(canonical_json_bytes(code)).hexdigest()
        or evaluator["scorer_id"] != scorer_id
        or scoring
        != {
            "comparison": "exact",
            "decoding": "greedy",
            "output_encoding": "UTF-8",
            "output_normalization": "NFC",
            "stop_token": "<|eot|>",
            "temperature": 0,
            "whitespace_policy": "strip_outer_only",
        }
        or isinstance(scoring["temperature"], bool)
    ):
        raise RunnerError("checkpoint result evaluator/scoring identity differs")
    count = _positive_int(items_per_family, "items_per_family")
    families = tuple(family_order)
    public_items = public_release.get("items")
    sealed_items = sealed_release.get("items")
    rows = result["items"]
    expected_count = len(families) * count
    if (
        not isinstance(public_items, list)
        or not isinstance(sealed_items, list)
        or not isinstance(rows, list)
        or len(public_items) != expected_count
        or len(sealed_items) != expected_count
        or len(rows) != expected_count
    ):
        raise RunnerError("checkpoint result replay count differs")
    replayed_correct: list[bool] = []
    seen: set[str] = set()
    for position, (row, visible, gold) in enumerate(
        zip(rows, public_items, sealed_items, strict=True)
    ):
        if (
            not isinstance(row, Mapping)
            or set(row) != _ITEM_FIELDS
            or not isinstance(visible, Mapping)
            or not isinstance(gold, Mapping)
        ):
            raise RunnerError(f"checkpoint result item fields differ at {position}")
        family = families[position // count]
        if (
            row["position"] != position
            or isinstance(row["position"], bool)
            or row["item_id"] != visible.get("item_id")
            or row["item_id"] != gold.get("item_id")
            or row["item_id"] in seen
            or row["task"] != family
            or row["task"] != visible.get("task")
            or row["task"] != gold.get("task")
            or row["source_index"] != visible.get("source_index")
            or row["source_index"] != gold.get("source_index")
            or row["max_new_tokens"] != visible.get("max_new_tokens")
            or visible.get("scorer_id") != scorer_id
        ):
            raise RunnerError(f"checkpoint result item identity differs at {position}")
        seen.add(row["item_id"])
        replay = _score_exact_output(
            row["raw_prediction"],
            gold.get("canonical_answer"),
            max_new_tokens=row["max_new_tokens"],
            generated_tokens=row["generated_tokens"],
            stop_reason=row["stop_reason"],
        )
        expected = {
            "correct": replay.correct,
            "error": replay.error,
            "prediction": replay.prediction,
            "prediction_sha256": replay.prediction_sha256,
            "raw_prediction": replay.raw_prediction,
            "status": (
                "correct"
                if replay.correct
                else "incorrect"
                if replay.valid
                else "invalid"
            ),
            "valid": replay.valid,
        }
        if any(row[name] != value for name, value in expected.items()):
            raise RunnerError(f"checkpoint result correctness differs at {position}")
        replayed_correct.append(replay.correct)

    expected_scores = []
    for family_index, family in enumerate(families):
        values = replayed_correct[
            family_index * count : (family_index + 1) * count
        ]
        numerator = sum(int(value) for value in values)
        accuracy = Fraction(numerator, count)
        expected_scores.append(
            {
                "accuracy": float(accuracy),
                "accuracy_fraction": _fraction_dict(accuracy),
                "denominator": count,
                "numerator": numerator,
                "task": family,
            }
        )
    expected_macro = sum(
        (
            Fraction(item["numerator"], item["denominator"])
            for item in expected_scores
        ),
        start=Fraction(),
    ) / len(
        expected_scores
    )
    if (
        result["family_scores"] != expected_scores
        or result["macro_accuracy"] != float(expected_macro)
        or result["macro_accuracy_fraction"] != _fraction_dict(expected_macro)
        or isinstance(result["macro_accuracy"], bool)
        or not isinstance(result["macro_accuracy"], (int, float))
        or not math.isfinite(float(result["macro_accuracy"]))
    ):
        raise RunnerError("checkpoint aggregates differ from raw-item replay")
    validity = _exact_mapping(result["validity"], {"gates", "passed"}, "validity")
    gates = validity["gates"]
    if not isinstance(gates, Mapping) or set(gates) != set(REQUIRED_VALIDITY_GATES):
        raise RunnerError("checkpoint validity gates differ")
    passed = True
    for name in REQUIRED_VALIDITY_GATES:
        gate = _exact_mapping(
            gates[name],
            {"evidence_sha256", "passed"},
            f"validity gate {name}",
        )
        if not isinstance(gate["passed"], bool) or not _is_sha256(
            gate["evidence_sha256"]
        ):
            raise RunnerError(f"validity gate is malformed: {name}")
        passed = passed and gate["passed"]
    if validity["passed"] is not passed:
        raise RunnerError("checkpoint validity summary differs")


def _checkpoint_result_key(arm: str, seed: int, step: int) -> str:
    if arm not in ARMS or seed not in SEEDS or step not in CHECKPOINT_STEPS:
        raise RunnerError("checkpoint result cell is outside the frozen matrix")
    return f"{RESULT_PREFIX}/{arm}/seed-{seed}/step-{step:07d}.json"


def _publish_checkpoint_result(
    result: Mapping[str, Any],
    release: ReleaseBinding,
    authority,
) -> Task3Publication:
    """Publish once under the evaluator-only KMS boundary."""

    _validate_release_binding(release)
    run = result.get("run") if isinstance(result, Mapping) else None
    checkpoint = result.get("checkpoint") if isinstance(result, Mapping) else None
    if not isinstance(run, Mapping) or not isinstance(checkpoint, Mapping):
        raise RunnerError("checkpoint result publication identity is missing")
    arm = run.get("arm")
    seed = run.get("seed")
    step = checkpoint.get("step")
    _checkpoint_result_key(arm, seed, step)
    payload = canonical_json_bytes(result)
    return authority.put_checkpoint_result(
        arm,
        seed,
        step,
        payload,
        release.authority,
    )


def _s3_ref_dict(ref: S3ObjectVersion) -> dict[str, Any]:
    value = {
        "bucket": ref.bucket,
        "bytes": ref.bytes,
        "key": ref.key,
        "server_side_encryption": ref.server_side_encryption,
        "sha256": ref.sha256,
        "version_id": ref.version_id,
    }
    if ref.kms_key_arn is not None:
        value["kms_key_arn"] = ref.kms_key_arn
    return value


def _s3_ref_from_mapping(value: object, label: str) -> S3ObjectVersion:
    if not isinstance(value, Mapping):
        raise RunnerError(f"{label} fields differ")
    fields = {
        "bucket",
        "bytes",
        "key",
        "server_side_encryption",
        "sha256",
        "version_id",
    }
    if "kms_key_arn" in value:
        fields.add("kms_key_arn")
    _exact_mapping(value, fields, label)
    ref = S3ObjectVersion(
        bucket=value["bucket"],
        key=value["key"],
        version_id=value["version_id"],
        bytes=value["bytes"],
        sha256=value["sha256"],
        server_side_encryption=value["server_side_encryption"],
        kms_key_arn=value.get("kms_key_arn"),
    )
    _validate_s3_ref(ref, label)
    return ref


def _require_task3_ref(
    ref: S3ObjectVersion,
    release: ReleaseBinding,
    *,
    label: str,
    expected_key: str,
    expected_kms_key_arn: str,
) -> None:
    _validate_s3_ref(ref, label)
    if (
        ref.key != expected_key
        or ref.server_side_encryption != "aws:kms"
        or ref.kms_key_arn != expected_kms_key_arn
    ):
        raise RunnerError(f"{label} exact AWS authority differs")


def _parse_checkpoint_manifest(
    payload: Mapping[str, Any],
    manifest_ref: S3ObjectVersion,
    release: ReleaseBinding,
) -> CheckpointManifest:
    _validate_release_binding(release)
    _require_task3_ref(
        manifest_ref,
        release,
        label="signed checkpoint manifest",
        expected_key=TASK3_CHECKPOINT_MANIFEST_KEY,
        expected_kms_key_arn=release.authority.sealed_gold_kms_key_arn,
    )
    top_fields = {
        "activation",
        "authority_record_sha256",
        "authority_record_version_id",
        "cells",
        "cohort_id",
        "contract_sha256",
        "corpus_receipt",
        "format",
        "runtime_lock",
        "schema_version",
    }
    _exact_mapping(payload, top_fields, "checkpoint manifest")
    if (
        payload["format"] != CHECKPOINT_MANIFEST_FORMAT
        or payload["schema_version"] != 1
        or isinstance(payload["schema_version"], bool)
        or payload["cohort_id"] != COHORT_ID
        or payload["contract_sha256"] != release.contract_sha256
        or payload["authority_record_sha256"] != release.authority.record_sha256
        or payload["authority_record_version_id"]
        != release.authority.record_version_id
        or payload["activation"] != _s3_ref_dict(release.activation)
    ):
        raise RunnerError("checkpoint manifest authority identity differs")
    runtime_lock = _s3_ref_from_mapping(
        payload["runtime_lock"],
        "checkpoint manifest runtime lock",
    )
    corpus_receipt = _s3_ref_from_mapping(
        payload["corpus_receipt"],
        "checkpoint manifest corpus receipt",
    )
    _require_task3_ref(
        runtime_lock,
        release,
        label="runtime lock",
        expected_key=f"{TASK3_EVIDENCE_PREFIX}runtime/runtime-lock.json",
        expected_kms_key_arn=release.authority.checkpoint_kms_key_arn,
    )
    _require_task3_ref(
        corpus_receipt,
        release,
        label="corpus receipt",
        expected_key=f"{TASK3_EVIDENCE_PREFIX}corpus/receipt.json",
        expected_kms_key_arn=release.authority.checkpoint_kms_key_arn,
    )
    raw_cells = payload["cells"]
    if not isinstance(raw_cells, list) or len(raw_cells) != 100:
        raise RunnerError("checkpoint manifest must contain exactly 100 cells")
    expected_order = [
        (arm, seed, step)
        for seed in SEEDS
        for arm in ARMS
        for step in CHECKPOINT_STEPS
    ]
    cells: dict[tuple[str, int, int], ManifestCell] = {}
    for position, (raw, expected) in enumerate(
        zip(raw_cells, expected_order, strict=True)
    ):
        cell = _exact_mapping(
            raw,
            {
                "arm",
                "checkpoint",
                "evidence",
                "pairing",
                "run_config",
                "seed",
                "step",
            },
            f"checkpoint manifest cell {position}",
        )
        arm, seed, step = expected
        if (
            cell["arm"] != arm
            or cell["seed"] != seed
            or isinstance(cell["seed"], bool)
            or cell["step"] != step
            or isinstance(cell["step"], bool)
        ):
            raise RunnerError("checkpoint manifest cell order differs")
        run = run_id(arm, seed)
        checkpoint = _s3_ref_from_mapping(
            cell["checkpoint"],
            f"checkpoint manifest checkpoint {position}",
        )
        evidence = _s3_ref_from_mapping(
            cell["evidence"],
            f"checkpoint manifest evidence {position}",
        )
        run_config = _s3_ref_from_mapping(
            cell["run_config"],
            f"checkpoint manifest run config {position}",
        )
        _require_task3_ref(
            checkpoint,
            release,
            label="checkpoint",
            expected_key=f"{TASK3_CHECKPOINT_PREFIX}{run}/step{step:07d}.pt",
            expected_kms_key_arn=release.authority.checkpoint_kms_key_arn,
        )
        _require_task3_ref(
            evidence,
            release,
            label="raw validity evidence",
            expected_key=(
                f"{TASK3_EVIDENCE_PREFIX}{run}/step{step:07d}/gates.json"
            ),
            expected_kms_key_arn=release.authority.checkpoint_kms_key_arn,
        )
        _require_task3_ref(
            run_config,
            release,
            label="run config evidence",
            expected_key=f"{TASK3_EVIDENCE_PREFIX}{run}/config.json",
            expected_kms_key_arn=release.authority.checkpoint_kms_key_arn,
        )
        pairing_value = _exact_mapping(
            cell["pairing"],
            set(_PAIRING_KINDS),
            f"checkpoint manifest pairing {position}",
        )
        pairing: dict[str, S3ObjectVersion] = {}
        for kind in _PAIRING_KINDS:
            ref = _s3_ref_from_mapping(
                pairing_value[kind],
                f"checkpoint manifest pairing {kind}",
            )
            _require_task3_ref(
                ref,
                release,
                label=f"pairing {kind} evidence",
                expected_key=(
                    f"{TASK3_EVIDENCE_PREFIX}{run}/pairing/{kind}.json"
                ),
                expected_kms_key_arn=release.authority.checkpoint_kms_key_arn,
            )
            pairing[kind] = ref
        identity = (arm, seed, step)
        if identity in cells:
            raise RunnerError("checkpoint manifest repeats a matrix cell")
        cells[identity] = ManifestCell(
            arm=arm,
            seed=seed,
            step=step,
            checkpoint=checkpoint,
            run_config=run_config,
            evidence=evidence,
            pairing=pairing,
        )
    return CheckpointManifest(
        ref=manifest_ref,
        runtime_lock=runtime_lock,
        corpus_receipt=corpus_receipt,
        cells=cells,
    )


def _tokenizer_runtime_identity() -> dict[str, Any]:
    tokenizer_module = importlib.import_module("train.tokenizer")
    expansion_module = importlib.import_module("corpusgen.reasoning_expansion")
    cache_root = getattr(tokenizer_module, "_CACHE_DIR", None)
    expected_root = ROOT / ".tiktoken_cache"
    configured_root = os.environ.get("TIKTOKEN_CACHE_DIR")
    if not isinstance(cache_root, Path) or not isinstance(configured_root, str):
        raise RunnerError("frozen tokenizer cache identity is unavailable")
    try:
        actual_root = cache_root.resolve(strict=True)
        expected = expected_root.resolve(strict=True)
        configured = Path(configured_root).resolve(strict=True)
    except OSError as error:
        raise RunnerError("frozen tokenizer cache is unavailable") from error
    if (
        cache_root.is_symlink()
        or expected_root.is_symlink()
        or actual_root != expected
        or configured != expected
        or not actual_root.is_dir()
    ):
        raise RunnerError("frozen tokenizer cache root differs")
    cache_files = []
    try:
        children = sorted(actual_root.iterdir(), key=lambda path: path.name)
    except OSError as error:
        raise RunnerError("frozen tokenizer cache cannot be enumerated") from error
    for path in children:
        try:
            metadata = path.lstat()
            payload = path.read_bytes()
        except OSError as error:
            raise RunnerError("frozen tokenizer cache file is unavailable") from error
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
            or metadata.st_size != len(payload)
        ):
            raise RunnerError("frozen tokenizer cache file is unsafe")
        cache_files.append(
            {
                "bytes": len(payload),
                "name": path.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    effective_fingerprint = expansion_module.effective_tokenizer_sha256()
    vocab_size = getattr(tokenizer_module, "VOCAB_SIZE", None)
    if (
        not cache_files
        or not isinstance(effective_fingerprint, str)
        or len(effective_fingerprint) != 64
        or set(effective_fingerprint) - _HEX
        or isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size <= 0
    ):
        raise RunnerError("effective tokenizer identity differs")
    return {
        "cache_files": cache_files,
        "cache_root": ".tiktoken_cache",
        "effective_sha256": effective_fingerprint,
        "vocab_size": vocab_size,
    }


def _actual_runtime_identity() -> dict[str, Any]:
    modules = []
    for module_name, relative in _RUNTIME_MODULE_PATHS:
        module = importlib.import_module(module_name)
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            raise RunnerError(f"actual imported runtime module is missing: {module_name}")
        source_path = Path(raw_path)
        expected_source = ROOT / relative
        try:
            actual = source_path.resolve(strict=True)
            expected = expected_source.resolve(strict=True)
        except OSError as error:
            raise RunnerError(
                f"actual imported runtime module is unavailable: {module_name}"
            ) from error
        if (
            source_path.is_symlink()
            or expected_source.is_symlink()
            or actual != expected
            or not actual.is_file()
        ):
            raise RunnerError(
                f"actual imported runtime module path differs: {module_name}"
            )
        try:
            module_bytes = actual.read_bytes()
        except OSError as error:
            raise RunnerError(
                f"actual imported runtime module could not be read: {module_name}"
            ) from error
        modules.append(
            {
                "module": module_name,
                "path": relative,
                "sha256": hashlib.sha256(module_bytes).hexdigest(),
            }
        )
    try:
        import numpy
        import torch
    except ImportError as error:
        raise RunnerError("frozen runtime dependencies are unavailable") from error
    cuda = {
        "available": bool(torch.cuda.is_available()),
        "cudnn_version": torch.backends.cudnn.version(),
        "device_capabilities": [
            list(torch.cuda.get_device_capability(index))
            for index in range(torch.cuda.device_count())
        ],
        "device_count": torch.cuda.device_count(),
        "device_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "runtime_version": torch.version.cuda,
    }
    return {
        "cuda": cuda,
        "format": RUNTIME_LOCK_FORMAT,
        "modules": modules,
        "packages": {
            "numpy": numpy.__version__,
            "tiktoken": importlib.metadata.version("tiktoken"),
            "torch": torch.__version__,
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "schema_version": 2,
        "tokenizer": _tokenizer_runtime_identity(),
    }


def _verify_runtime_lock(lock: Mapping[str, Any]) -> None:
    if not isinstance(lock, Mapping) or dict(lock) != _actual_runtime_identity():
        raise RunnerError("signed runtime lock differs from actual imported runtime")


def _deserialize_checkpoint(
    payload: bytes,
    *,
    expected_step: int,
) -> Mapping[str, Any]:
    if not isinstance(payload, bytes) or not payload:
        raise RunnerError("checkpoint bytes are missing")
    try:
        import torch

        state = torch.load(
            io.BytesIO(payload),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise RunnerError(
            "checkpoint rejected by safe weights-only deserialization"
        ) from error
    if (
        not isinstance(state, Mapping)
        or set(state) != {"model", "model_cfg", "step"}
        or state["step"] != expected_step
        or isinstance(state["step"], bool)
        or state["model_cfg"] != _MODEL_CONFIG
        or not isinstance(state["model"], Mapping)
    ):
        raise RunnerError("checkpoint snapshot schema or model identity differs")
    return state


def _load_verified_checkpoint(
    binding: CheckpointBinding,
    authority,
    verified: VerifiedAwsAuthority | None = None,
) -> Mapping[str, Any]:
    _validate_checkpoint_binding(binding)
    expected = S3ObjectVersion(
        bucket=STORAGE_BUCKET,
        key=binding.object_key,
        version_id=binding.version_id,
        bytes=binding.bytes,
        sha256=binding.sha256,
        server_side_encryption="aws:kms",
        kms_key_arn=binding.checkpoint_kms_key_arn,
    )
    if verified is None:
        payload, actual = authority.read_checkpoint(expected)
    else:
        payload, actual = authority.read_checkpoint(expected, verified)
    if (
        not isinstance(payload, bytes)
        or actual != expected
        or len(payload) != binding.bytes
        or hashlib.sha256(payload).hexdigest() != binding.sha256
    ):
        raise RunnerError(
            "checkpoint differs from the exact signed manifest object version"
        )
    return _deserialize_checkpoint(payload, expected_step=binding.step)


def _evidence_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _derive_validity_from_evidence(
    evidence: Mapping[str, Any],
    context: EvidenceContext,
) -> dict[str, GateEvidence]:
    if not isinstance(context, EvidenceContext):
        raise RunnerError("evidence context is malformed")
    top_fields = {
        "arm",
        "evaluator",
        "exact_resume",
        "exclusion",
        "factual_burden",
        "format",
        "identity",
        "imputation",
        "memory_off_leakage",
        "memory_on_recall",
        "provider",
        "registry",
        "replacement",
        "schema_version",
        "seed",
        "step",
        "stopping",
        "substitution",
    }
    _exact_mapping(evidence, top_fields, "raw validity evidence")
    binding = context.checkpoint
    if (
        evidence["format"] != RAW_EVIDENCE_FORMAT
        or evidence["schema_version"] != 1
        or isinstance(evidence["schema_version"], bool)
        or evidence["arm"] != binding.arm
        or evidence["seed"] != binding.seed
        or isinstance(evidence["seed"], bool)
        or evidence["step"] != binding.step
        or isinstance(evidence["step"], bool)
    ):
        raise RunnerError("raw validity evidence identity differs")

    def section(name: str, fields: set[str]) -> Mapping[str, Any]:
        return _exact_mapping(evidence[name], fields, f"{name} evidence")

    factual = section(
        "factual_burden",
        {"observed_fact_tokens", "required_min_fact_tokens"},
    )
    recall = section(
        "memory_on_recall",
        {"expected_recalled_records", "observed_recalled_records"},
    )
    leakage = section("memory_off_leakage", {"leaked_record_count"})
    resume = section(
        "exact_resume",
        {"resumed_state_sha256", "uninterrupted_state_sha256"},
    )
    evaluator = section("evaluator", {"observed_role_arn"})
    registry = section(
        "registry",
        {
            "expected_item_count",
            "expected_registry_sha256",
            "observed_item_count",
            "observed_registry_sha256",
        },
    )
    substitution = section(
        "substitution",
        {"loaded_checkpoint_sha256", "manifest_checkpoint_sha256"},
    )
    stopping = section(
        "stopping",
        {"completed_steps", "continuation_decisions", "planned_steps"},
    )
    replacement = section(
        "replacement",
        {"expected_run_id", "observed_run_id", "replacement_run_ids"},
    )
    exclusion = section(
        "exclusion",
        {"excluded_item_ids", "expected_item_count", "observed_item_count"},
    )
    provider = section("provider", {"expected", "observed"})
    imputation = section(
        "imputation",
        {"expected_seeds", "imputed_seeds", "observed_seeds"},
    )
    identity = section(
        "identity",
        {
            "config_sha256",
            "corpus_sha256",
            "data_order_sha256",
            "initialization_sha256",
            "run_config_sha256",
            "runtime_sha256",
        },
    )
    integer_values = (
        factual["observed_fact_tokens"],
        factual["required_min_fact_tokens"],
        recall["expected_recalled_records"],
        recall["observed_recalled_records"],
        leakage["leaked_record_count"],
        registry["expected_item_count"],
        registry["observed_item_count"],
        exclusion["expected_item_count"],
        exclusion["observed_item_count"],
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in integer_values
    ):
        raise RunnerError("raw validity evidence counts are malformed")
    for name, value in (
        ("resumed state", resume["resumed_state_sha256"]),
        ("uninterrupted state", resume["uninterrupted_state_sha256"]),
        ("expected registry", registry["expected_registry_sha256"]),
        ("observed registry", registry["observed_registry_sha256"]),
        ("loaded checkpoint", substitution["loaded_checkpoint_sha256"]),
        ("manifest checkpoint", substitution["manifest_checkpoint_sha256"]),
        *((f"identity {name}", value) for name, value in identity.items()),
    ):
        _require_sha256(value, name)
    expected_continuations = [
        {"reason": "precommitted", "step": step} for step in CHECKPOINT_STEPS
    ]
    expected_run = run_id(binding.arm, binding.seed)
    identity_matches = identity == {
        "config_sha256": binding.paired_config_sha256,
        "corpus_sha256": binding.paired_corpus_sha256,
        "data_order_sha256": binding.data_order_sha256,
        "initialization_sha256": binding.initialization_sha256,
        "run_config_sha256": binding.run_config_sha256,
        "runtime_sha256": binding.paired_runtime_sha256,
    }
    passed = {
        "factual_burden": (
            factual["required_min_fact_tokens"] > 0
            and factual["observed_fact_tokens"]
            >= factual["required_min_fact_tokens"]
        ),
        "memory_on_recall": (
            recall["expected_recalled_records"] > 0
            and recall["observed_recalled_records"]
            == recall["expected_recalled_records"]
        ),
        "memory_off_leakage": leakage["leaked_record_count"] == 0,
        "exact_resume": (
            resume["resumed_state_sha256"]
            == resume["uninterrupted_state_sha256"]
        ),
        "evaluator_authority": (
            evaluator["observed_role_arn"]
            == "arn:aws:iam::${AWS_ACCOUNT_ID}:role/memorysplit-reasoning-v3-evaluator"
        ),
        "complete_registry": (
            context.manifest_complete
            and registry["expected_item_count"] == context.expected_item_count
            and registry["observed_item_count"] == context.expected_item_count
            and registry["expected_registry_sha256"] == context.registry_sha256
            and registry["observed_registry_sha256"] == context.registry_sha256
        ),
        "no_substitution": (
            context.manifest_complete
            and substitution["loaded_checkpoint_sha256"] == binding.sha256
            and substitution["manifest_checkpoint_sha256"] == binding.sha256
            and identity_matches
        ),
        "no_outcome_dependent_stopping": (
            stopping["planned_steps"] == list(CHECKPOINT_STEPS)
            and stopping["completed_steps"] == list(CHECKPOINT_STEPS)
            and stopping["continuation_decisions"] == expected_continuations
        ),
        "no_replacement": (
            replacement["expected_run_id"] == expected_run
            and replacement["observed_run_id"] == expected_run
            and replacement["replacement_run_ids"] == []
        ),
        "no_exclusion": (
            exclusion["expected_item_count"] == context.expected_item_count
            and exclusion["observed_item_count"] == context.expected_item_count
            and exclusion["excluded_item_ids"] == []
        ),
        "provider_fixed": (
            provider["expected"] == PROVIDER
            and provider["observed"] == PROVIDER
        ),
        "no_missing_seed_imputation": (
            imputation["expected_seeds"] == list(SEEDS)
            and imputation["observed_seeds"] == list(SEEDS)
            and imputation["imputed_seeds"] == []
        ),
    }
    sources = {
        "factual_burden": factual,
        "memory_on_recall": recall,
        "memory_off_leakage": leakage,
        "exact_resume": resume,
        "evaluator_authority": evaluator,
        "complete_registry": registry,
        "no_substitution": {
            "identity": dict(identity),
            "substitution": dict(substitution),
        },
        "no_outcome_dependent_stopping": stopping,
        "no_replacement": replacement,
        "no_exclusion": exclusion,
        "provider_fixed": provider,
        "no_missing_seed_imputation": imputation,
    }
    return {
        name: GateEvidence(
            passed=passed[name],
            evidence_sha256=_evidence_digest(sources[name]),
        )
        for name in REQUIRED_VALIDITY_GATES
    }


def _parse_pairing_evidence(
    payload: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    kind: str,
) -> str:
    _exact_mapping(
        payload,
        {"arm", "format", "identity_sha256", "kind", "schema_version", "seed"},
        f"pairing {kind} evidence",
    )
    if (
        payload["format"] != PAIRING_EVIDENCE_FORMAT
        or payload["schema_version"] != 1
        or isinstance(payload["schema_version"], bool)
        or payload["arm"] != arm
        or payload["seed"] != seed
        or isinstance(payload["seed"], bool)
        or payload["kind"] != kind
    ):
        raise RunnerError(f"pairing {kind} evidence identity differs")
    return _require_sha256(payload["identity_sha256"], f"pairing {kind}")


def _load_authenticated_checkpoint(
    authority,
    release: ReleaseBinding,
    manifest: CheckpointManifest,
    arm: str,
    seed: int,
    checkpoint_step: int,
) -> _ValidatedCheckpoint:
    cell = manifest.cells.get((arm, seed, checkpoint_step))
    if cell is None:
        raise RunnerError("requested checkpoint is outside the signed manifest")
    runtime_bytes, _ = authority.read_task3_evidence(
        manifest.runtime_lock,
        release.authority,
    )
    runtime_lock = _parse_canonical_bytes(runtime_bytes, "Task 3 runtime lock")
    _verify_runtime_lock(runtime_lock)
    corpus_bytes, _ = authority.read_task3_evidence(
        manifest.corpus_receipt,
        release.authority,
    )
    corpus = _parse_canonical_bytes(corpus_bytes, "Task 3 corpus evidence")
    _exact_mapping(
        corpus,
        {
            "corpus_receipt_sha256",
            "format",
            "schema_version",
        },
        "Task 3 corpus evidence",
    )
    if (
        corpus["format"] != CORPUS_EVIDENCE_FORMAT
        or corpus["schema_version"] != 1
        or isinstance(corpus["schema_version"], bool)
        or corpus["corpus_receipt_sha256"] != release.corpus_receipt_sha256
    ):
        raise RunnerError("Task 3 corpus evidence identity differs")
    config_bytes, _ = authority.read_task3_evidence(
        cell.run_config,
        release.authority,
    )
    config_evidence = _parse_canonical_bytes(
        config_bytes,
        "Task 3 run config evidence",
    )
    _exact_mapping(
        config_evidence,
        {
            "arm",
            "config_path",
            "config_sha256",
            "format",
            "schema_version",
            "seed",
        },
        "Task 3 run config evidence",
    )
    if (
        config_evidence["format"] != RUN_CONFIG_EVIDENCE_FORMAT
        or config_evidence["schema_version"] != 1
        or isinstance(config_evidence["schema_version"], bool)
        or config_evidence["arm"] != arm
        or config_evidence["seed"] != seed
        or isinstance(config_evidence["seed"], bool)
        or config_evidence["config_path"] != config_path(arm, seed)
    ):
        raise RunnerError("Task 3 run config evidence identity differs")
    run_config_sha256 = _require_sha256(
        config_evidence["config_sha256"],
        "run config",
    )
    pairing: dict[str, str] = {}
    for kind in _PAIRING_KINDS:
        raw, _ = authority.read_task3_evidence(
            cell.pairing[kind],
            release.authority,
        )
        pairing[kind] = _parse_pairing_evidence(
            _parse_canonical_bytes(raw, f"Task 3 pairing {kind} evidence"),
            arm=arm,
            seed=seed,
            kind=kind,
        )
    binding = CheckpointBinding(
        arm=arm,
        seed=seed,
        step=checkpoint_step,
        sha256=cell.checkpoint.sha256,
        object_key=cell.checkpoint.key,
        version_id=cell.checkpoint.version_id,
        bytes=cell.checkpoint.bytes,
        run_config_path=config_path(arm, seed),
        run_config_sha256=run_config_sha256,
        initialization_sha256=pairing["initialization"],
        data_order_sha256=pairing["data_order"],
        paired_runtime_sha256=pairing["runtime"],
        paired_corpus_sha256=pairing["corpus"],
        paired_config_sha256=pairing["config"],
        checkpoint_kms_key_arn=cell.checkpoint.kms_key_arn or "",
        manifest_key=manifest.ref.key,
        manifest_version_id=manifest.ref.version_id,
        manifest_sha256=manifest.ref.sha256,
        evidence_key=cell.evidence.key,
        evidence_version_id=cell.evidence.version_id,
        evidence_sha256=cell.evidence.sha256,
        runtime_lock_key=manifest.runtime_lock.key,
        runtime_lock_version_id=manifest.runtime_lock.version_id,
        runtime_lock_object_sha256=manifest.runtime_lock.sha256,
        corpus_receipt_key=manifest.corpus_receipt.key,
        corpus_receipt_version_id=manifest.corpus_receipt.version_id,
        corpus_receipt_object_sha256=manifest.corpus_receipt.sha256,
    )
    state = _load_verified_checkpoint(binding, authority, release.authority)
    evidence_bytes, _ = authority.read_task3_evidence(
        cell.evidence,
        release.authority,
    )
    evidence = _parse_canonical_bytes(
        evidence_bytes,
        "Task 3 raw validity evidence",
    )
    validity = _derive_validity_from_evidence(
        evidence,
        EvidenceContext(
            checkpoint=binding,
            expected_item_count=7_168,
            manifest_complete=len(manifest.cells) == 100,
            registry_sha256=release.registry_sha256,
        ),
    )
    return _ValidatedCheckpoint(
        binding=binding,
        validity=validity,
        state=state,
    )


def _replay_manifest_validity(
    authority,
    release: ReleaseBinding,
    manifest: CheckpointManifest,
    binding: CheckpointBinding,
    *,
    expected_item_count: int,
) -> dict[str, GateEvidence]:
    _validate_checkpoint_binding(binding)
    cell = manifest.cells.get((binding.arm, binding.seed, binding.step))
    if cell is None:
        raise RunnerError("result cell is absent from signed checkpoint manifest")
    if (
        binding.object_key != cell.checkpoint.key
        or binding.version_id != cell.checkpoint.version_id
        or binding.bytes != cell.checkpoint.bytes
        or binding.sha256 != cell.checkpoint.sha256
        or binding.checkpoint_kms_key_arn != cell.checkpoint.kms_key_arn
        or binding.manifest_key != manifest.ref.key
        or binding.manifest_version_id != manifest.ref.version_id
        or binding.manifest_sha256 != manifest.ref.sha256
        or binding.evidence_key != cell.evidence.key
        or binding.evidence_version_id != cell.evidence.version_id
        or binding.evidence_sha256 != cell.evidence.sha256
        or binding.runtime_lock_key != manifest.runtime_lock.key
        or binding.runtime_lock_version_id != manifest.runtime_lock.version_id
        or binding.runtime_lock_object_sha256 != manifest.runtime_lock.sha256
        or binding.corpus_receipt_key != manifest.corpus_receipt.key
        or binding.corpus_receipt_version_id != manifest.corpus_receipt.version_id
        or binding.corpus_receipt_object_sha256 != manifest.corpus_receipt.sha256
    ):
        raise RunnerError("checkpoint result differs from signed manifest versions")
    config_bytes, _ = authority.read_task3_evidence(
        cell.run_config,
        release.authority,
    )
    config_evidence = _parse_canonical_bytes(
        config_bytes,
        "Task 3 run config evidence",
    )
    _exact_mapping(
        config_evidence,
        {
            "arm",
            "config_path",
            "config_sha256",
            "format",
            "schema_version",
            "seed",
        },
        "Task 3 run config evidence",
    )
    if (
        config_evidence["format"] != RUN_CONFIG_EVIDENCE_FORMAT
        or config_evidence["schema_version"] != 1
        or isinstance(config_evidence["schema_version"], bool)
        or config_evidence["arm"] != binding.arm
        or config_evidence["seed"] != binding.seed
        or isinstance(config_evidence["seed"], bool)
        or config_evidence["config_path"] != binding.run_config_path
        or config_evidence["config_sha256"] != binding.run_config_sha256
    ):
        raise RunnerError("checkpoint result run config evidence differs")
    expected_pairing = {
        "config": binding.paired_config_sha256,
        "corpus": binding.paired_corpus_sha256,
        "data_order": binding.data_order_sha256,
        "initialization": binding.initialization_sha256,
        "runtime": binding.paired_runtime_sha256,
    }
    for kind in _PAIRING_KINDS:
        raw, _ = authority.read_task3_evidence(
            cell.pairing[kind],
            release.authority,
        )
        actual = _parse_pairing_evidence(
            _parse_canonical_bytes(raw, f"Task 3 pairing {kind} evidence"),
            arm=binding.arm,
            seed=binding.seed,
            kind=kind,
        )
        if actual != expected_pairing[kind]:
            raise RunnerError(f"checkpoint result pairing {kind} evidence differs")
    evidence_bytes, _ = authority.read_task3_evidence(
        cell.evidence,
        release.authority,
    )
    evidence = _parse_canonical_bytes(
        evidence_bytes,
        "Task 3 raw validity evidence",
    )
    return _derive_validity_from_evidence(
        evidence,
        EvidenceContext(
            checkpoint=binding,
            expected_item_count=expected_item_count,
            manifest_complete=len(manifest.cells) == 100,
            registry_sha256=release.registry_sha256,
        ),
    )


def _load_signed_manifest(
    authority,
    release: ReleaseBinding,
) -> CheckpointManifest:
    payload, ref = authority.read_checkpoint_manifest(release.authority)
    parsed = _parse_canonical_bytes(payload, "signed checkpoint manifest payload")
    manifest = _parse_checkpoint_manifest(parsed, ref, release)
    runtime_bytes, _ = authority.read_task3_evidence(
        manifest.runtime_lock,
        release.authority,
    )
    _verify_runtime_lock(
        _parse_canonical_bytes(runtime_bytes, "Task 3 runtime lock")
    )
    corpus_bytes, _ = authority.read_task3_evidence(
        manifest.corpus_receipt,
        release.authority,
    )
    corpus = _parse_canonical_bytes(corpus_bytes, "Task 3 corpus evidence")
    _exact_mapping(
        corpus,
        {"corpus_receipt_sha256", "format", "schema_version"},
        "Task 3 corpus evidence",
    )
    if (
        corpus["format"] != CORPUS_EVIDENCE_FORMAT
        or corpus["schema_version"] != 1
        or isinstance(corpus["schema_version"], bool)
        or corpus["corpus_receipt_sha256"] != release.corpus_receipt_sha256
    ):
        raise RunnerError("Task 3 corpus evidence identity differs")
    return manifest


def validate_frozen_checkpoint_inputs(
    arm: str,
    seed: int,
    checkpoint_step: int,
) -> Mapping[str, Any]:
    """Validate one authenticated cell without decoding or publication."""

    authority, _contract, _public, _sealed, release = _load_fixed_release()
    manifest = _load_signed_manifest(authority, release)
    loaded = _load_authenticated_checkpoint(
        authority,
        release,
        manifest,
        arm,
        seed,
        checkpoint_step,
    )
    failed = [
        name for name in REQUIRED_VALIDITY_GATES if not loaded.validity[name].passed
    ]
    return {
        "arm": arm,
        "checkpoint": _checkpoint_dict(loaded.binding),
        "failed_gates": failed,
        "mode": "invalid" if failed else "valid",
        "run": _run_dict(loaded.binding),
        "seed": seed,
        "validity": _validate_gate_evidence(loaded.validity),
    }


def _task3_code_bindings(
    task2_bindings: Sequence[tuple[str, str]],
) -> tuple[CodeBinding, ...]:
    runtime = _actual_runtime_identity()
    actual = {
        row["path"]: row["sha256"]
        for row in runtime["modules"]
    }
    rows = []
    for path, digest in task2_bindings:
        if actual.get(path) != digest:
            raise RunnerError(
                f"signed Task 2 evaluator code differs from actual import: {path}"
            )
        rows.append(CodeBinding(path=path, sha256=digest))
    existing = {row.path for row in rows}
    for relative in _TASK3_EVALUATOR_PATHS:
        digest = actual.get(relative)
        if digest is None:
            raise RunnerError(
                f"Task 3 evaluator code is not an actual imported module: {relative}"
            )
        if relative not in existing:
            rows.append(CodeBinding(path=relative, sha256=digest))
    _validate_code_bindings(rows)
    return tuple(rows)


def _load_fixed_release():
    authority = _new_evaluator_aws_authority()
    authority.require_evaluator_role()
    contract, verified = _load_authorized_contract(ROOT, authority)
    metadata = _read_signed_activation(contract, verified, authority)
    public_bytes = authority.read_model_visible(metadata.model_visible)
    sealed_bytes = authority.read_sealed_gold(metadata.sealed_gold)
    _validate_object_payload(
        public_bytes,
        metadata.model_visible,
        "model-visible release",
    )
    _validate_object_payload(
        sealed_bytes,
        metadata.sealed_gold,
        "sealed-gold release",
    )
    bundle = ReleaseBundle(
        model_visible_bytes=public_bytes,
        sealed_gold_bytes=sealed_bytes,
        registry_sha256=metadata.registry_sha256,
    )
    _validate_release_bundle(bundle, contract, metadata.provenance)
    public = _parse_canonical_bytes(public_bytes, "model-visible release")
    sealed = _parse_canonical_bytes(sealed_bytes, "sealed-gold release")
    envelope = _parse_canonical_bytes(
        metadata.envelope_bytes,
        "signed activation",
    )
    signature_document = envelope.get("signature")
    if not isinstance(signature_document, Mapping):
        raise RunnerError("signed activation signature is missing")
    signature_document_bytes = canonical_json_bytes(signature_document)
    release = ReleaseBinding(
        contract_id=contract.contract_id,
        contract_sha256=contract.sha256,
        authority=verified,
        activation=metadata.activation_object,
        activation_sha256=hashlib.sha256(metadata.envelope_bytes).hexdigest(),
        activation_signature_sha256=hashlib.sha256(
            signature_document_bytes
        ).hexdigest(),
        activation_signing_algorithm=signature_document.get(
            "signing_algorithm"
        ),
        model_visible=metadata.model_visible,
        sealed_gold=metadata.sealed_gold,
        registry_sha256=metadata.registry_sha256,
        corpus_receipt_sha256=metadata.provenance.corpus_receipt_sha256,
        runtime_lock_sha256=metadata.provenance.runtime_lock_sha256,
    )
    _validate_release_binding(release)
    return authority, contract, public, sealed, release


def _greedy_decode_model(
    model,
    tokenizer,
    public_items: Sequence[Mapping[str, Any]],
    *,
    device: str,
    batch_size: int,
) -> list[RawPrediction]:
    try:
        import torch
    except ImportError as error:
        raise RunnerError("frozen inference runtime dependencies are unavailable") from error
    if (
        device not in {"cpu", "cuda"}
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise RunnerError("greedy decoding runtime is invalid")
    context_limit = getattr(getattr(model, "cfg", None), "ctx", None)
    if (
        isinstance(context_limit, bool)
        or not isinstance(context_limit, int)
        or context_limit <= 0
    ):
        raise RunnerError("greedy decoding model context is invalid")
    encoded: list[list[int]] = []
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for position, item in enumerate(public_items):
        if not isinstance(item, Mapping):
            raise RunnerError(f"model-visible item {position} is malformed")
        prompt = item.get("prompt")
        limit = item.get("max_new_tokens")
        if not isinstance(prompt, str) or not prompt or not isinstance(limit, int):
            raise RunnerError(f"model-visible item {position} decoding fields differ")
        ids = tokenizer.encode(prompt)
        if (
            not ids
            or len(ids) + limit > context_limit
            or tokenizer.EOT in ids
        ):
            raise RunnerError(
                f"model-visible item {position} exceeds frozen decoding context"
            )
        encoded.append(ids)
        groups[(len(ids), limit)].append(position)

    predictions: list[RawPrediction | None] = [None] * len(public_items)
    try:
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            for (prompt_length, limit), positions in sorted(groups.items()):
                for start in range(0, len(positions), batch_size):
                    selected = positions[start : start + batch_size]
                    prompt_tensor = torch.tensor(
                        [encoded[index] for index in selected],
                        dtype=torch.long,
                        device=device,
                    )
                    if prompt_tensor.shape[1] != prompt_length:
                        raise RunnerError("prompt-length grouping drifted")
                    logits, cache = model.forward_step(prompt_tensor, None)
                    generated = [[] for _ in selected]
                    consumed = [0 for _ in selected]
                    done = [False for _ in selected]
                    reasons = ["max_new_tokens" for _ in selected]
                    for _ in range(limit):
                        choices = logits[:, -1, :].argmax(dim=-1).tolist()
                        next_ids = []
                        for row_index, token_id in enumerate(choices):
                            if done[row_index]:
                                next_ids.append(tokenizer.EOT)
                                continue
                            token = int(token_id)
                            consumed[row_index] += 1
                            if token == tokenizer.EOT:
                                done[row_index] = True
                                reasons[row_index] = "eot"
                                next_ids.append(tokenizer.EOT)
                                continue
                            generated[row_index].append(token)
                            next_ids.append(token)
                        if all(done):
                            break
                        next_tensor = torch.tensor(
                            next_ids,
                            dtype=torch.long,
                            device=device,
                        ).unsqueeze(1)
                        logits, cache = model.forward_step(next_tensor, cache)
                    for row_index, item_index in enumerate(selected):
                        item = public_items[item_index]
                        predictions[item_index] = RawPrediction(
                            item_id=item["item_id"],
                            task=item["task"],
                            source_index=item["source_index"],
                            raw_prediction=tokenizer.decode(generated[row_index]),
                            generated_tokens=consumed[row_index],
                            stop_reason=reasons[row_index],
                        )
    except RunnerError:
        raise
    except Exception as error:
        raise RunnerError("frozen greedy checkpoint decoding failed") from error
    if any(item is None for item in predictions):
        raise RunnerError("frozen inference omitted one or more registry items")
    return [item for item in predictions if item is not None]


def _greedy_predictions(
    state: Mapping[str, Any],
    public_items: Sequence[Mapping[str, Any]],
) -> list[RawPrediction]:
    try:
        import torch

        from train.model import GPT, GPTConfig
        from train.tokenizer import get_tok
    except ImportError as error:
        raise RunnerError("frozen inference runtime dependencies are unavailable") from error
    if not torch.cuda.is_available():
        raise RunnerError("frozen checkpoint inference requires CUDA")
    model = GPT(GPTConfig(**_MODEL_CONFIG))
    if model.num_params() != MODEL_PARAMETERS:
        raise RunnerError("inference model parameter count differs")
    try:
        model.load_state_dict(state["model"], strict=True)
        model.to("cuda").eval()
        tokenizer = get_tok()
    except Exception as error:
        raise RunnerError("frozen model/tokenizer loading failed") from error
    return _greedy_decode_model(
        model,
        tokenizer,
        public_items,
        device="cuda",
        batch_size=INFERENCE_BATCH_SIZE,
    )


def run_frozen_checkpoint_evaluation(
    arm: str,
    seed: int,
    checkpoint_step: int,
) -> PublishedCheckpointResult:
    """Score one fixed cell through signed exact-version evaluator authority."""

    authority, contract, public, sealed, release = _load_fixed_release()
    manifest = _load_signed_manifest(authority, release)
    loaded = _load_authenticated_checkpoint(
        authority,
        release,
        manifest,
        arm,
        seed,
        checkpoint_step,
    )
    predictions = _greedy_predictions(loaded.state, public["items"])
    result = _build_checkpoint_result(
        public_release=public,
        sealed_release=sealed,
        predictions=predictions,
        checkpoint=loaded.binding,
        release=release,
        validity=loaded.validity,
        evaluator_code=_task3_code_bindings(contract.evaluator_code),
        family_order=contract.family_names,
        items_per_family=contract.accepted_items_per_family,
        scorer_id=contract.scorer_id,
    )
    publication = _publish_checkpoint_result(result, release, authority)
    if publication.payload_bytes != canonical_json_bytes(result):
        raise RunnerError("published checkpoint result unsigned payload differs")
    return PublishedCheckpointResult(
        result=result,
        object_ref=publication.object_ref,
    )


__all__ = [
    "CHECKPOINT_STEPS",
    "PublishedCheckpointResult",
    "RunnerError",
    "run_frozen_checkpoint_evaluation",
    "validate_frozen_checkpoint_inputs",
]
