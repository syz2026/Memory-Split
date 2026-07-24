"""Fail-closed orchestration for sealed confirmatory evaluation."""

from __future__ import annotations

import base64
from collections import defaultdict
from collections.abc import Mapping
import ctypes
from dataclasses import asdict, dataclass, replace
import errno
import hashlib
import importlib
import inspect
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Protocol

from cluster.aws.gpu_profile import (
    parse_aws_gpu_profile_bytes,
    read_secure_regular_file,
)
from msctl.aws_contracts import (
    AWS_ENVIRONMENT_RECEIPT_V2_FIELDS,
    AWS_RUNTIME_VERSION_FIELDS,
)
from msctl.fsutil import open_directory, rename_noreplace_at
from msctl.aws_hardware import (
    parse_aws_runtime_lock_bytes,
    verify_aws_instance_identity_pkcs7,
)

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from evals.confirmatory.__main__ import main

    raise SystemExit(main())

from evals.confirmatory import (
    metrics,
    reporting,
    sealing,
    solver as solver_module,
)
from evals.confirmatory.actions import ActionOp, ActionSlot, validate_action_slots
from evals.confirmatory.aggregate import (
    RUN_BINDING_SCHEMA_V3,
    RunBindingV3,
)
from evals.confirmatory.contracts import (
    CONTRACT_VERSION,
    STUDY_CONTRACT_VERSION,
    STUDY_TARGETS_PER_UPDATE,
    CheckpointRecord,
    ItemRecord,
    MemoryMode,
    SealedGoldRecord,
    StoreRecord,
    Twin,
    canonical_json_bytes,
    canonical_sha256,
    validate_contract_bundle,
)


MSCTL_EVALUATOR_CONTRACT = "memorysplit-confirmatory-evaluator-v1"
RUN_BINDING_SCHEMA = "memorysplit.confirmatory.run-binding.v2"
OUTCOME_SCHEMA = "memorysplit.confirmatory.outcome.v2"
METRICS_SCHEMA = "memorysplit.confirmatory.metrics.v2"
INFERENCE_EVIDENCE_SCHEMA = "memorysplit.confirmatory.inference-evidence.v2"
VALIDITY_EVIDENCE_SCHEMA = "memorysplit.confirmatory.validity-evidence.v2"
SNAPSHOT_OUTPUT_SCHEMA_V3 = (
    "memorysplit.confirmatory.snapshot-evaluation-output.v3"
)
SNAPSHOT_METRICS_SCHEMA_V3 = (
    "memorysplit.confirmatory.snapshot-evaluation-metrics.v3"
)
SNAPSHOT_INFERENCE_SCHEMA_V3 = (
    "memorysplit.confirmatory.snapshot-inference-evidence.v3"
)
_V3_OUTPUT_ARTIFACTS = (
    "inference.json",
    "items.jsonl",
    "metrics.json",
    "outcomes.jsonl",
    "run.json",
    "sealed-gold.jsonl",
    "sealed-release.json",
    "stores.jsonl",
    "study-lock.json",
)
_MODEL_SNAPSHOT_FIELDS_V2 = frozenset(
    {
        "config_fingerprint",
        "data_provenance",
        "model",
        "model_cfg",
        "snapshot_version",
        "step",
        "study_identity",
        "world_size",
    }
)
_MODEL_SNAPSHOT_IDENTITY_FIELDS_V2 = frozenset(
    {
        "arm",
        "cohort_id",
        "data_provenance_sha256",
        "model_cfg_sha256",
        "run_id",
        "seed",
        "tokens_per_step",
    }
)
_MAX_RUN_BINDING_BYTES = 262_144
_MAX_STUDY_LOCK_BYTES = 2 * 1024 * 1024
_MAX_MODEL_SNAPSHOT_BYTES = 16 * 1024 * 1024 * 1024
_MAX_EXECUTION_EVIDENCE_BYTES = 2 * 1024 * 1024
_AWS_IDENTITY_REQUIRED_FIELDS = frozenset(
    {
        "accountId",
        "architecture",
        "imageId",
        "instanceId",
        "privateIp",
        "region",
    }
)
_AWS_IDENTITY_OPTIONAL_FIELDS = frozenset(
    {
        "availabilityZone",
        "billingProducts",
        "devpayProductCodes",
        "instanceType",
        "kernelId",
        "marketplaceProductCodes",
        "pendingTime",
        "ramdiskId",
        "version",
    }
)
_AWS_ACCOUNT_ID_RE = re.compile(r"[0-9]{12}")
_AWS_INSTANCE_ID_RE = re.compile(r"i-[0-9a-f]{8,17}")
_AWS_REGION_RE = re.compile(r"[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+")
_BOOT_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
PRIMARY_CONTRAST_ID = (
    "primary_omnibus_pair_and_proof__graph_non_path__"
    "composition_joint_ood__split90_minus_dense"
)
PRIMARY_TEST_METHOD = "exact_one_sided_exhaustive_sign_flip"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CANONICAL_CANDIDATE_VALUE = r"(?:[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}|<\|slot_[0-3]\|>)"
_CANDIDATE_RESPONSE_RE = re.compile(rf" candidate=({_CANONICAL_CANDIDATE_VALUE})")
_FINAL_RESPONSE_RE = re.compile(
    rf" candidate=({_CANONICAL_CANDIDATE_VALUE})"
    rf" final=({_CANONICAL_CANDIDATE_VALUE})"
)
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
_FROZEN_RELATION_TOKENS = MappingProxyType(
    {f"r{index}": 50276 + index for index in range(16)}
)
_FROZEN_SLOT_TOKENS = (50267, 50268, 50269, 50270)
_FROZEN_GRAPH_SPECIAL_TOKENS = MappingProxyType(
    {
        "<|graph_start|>": 50261,
        "<|graph_read|>": 50262,
        "<|graph_return|>": 50263,
        "<|graph_end|>": 50264,
        "<|graph_halt|>": 50265,
        "<|graph_noop|>": 50266,
        "<|slot_0|>": 50267,
        "<|slot_1|>": 50268,
        "<|slot_2|>": 50269,
        "<|slot_3|>": 50270,
        "<|dir_out|>": 50271,
        "<|dir_in|>": 50272,
        "<|graph_step|>": 50273,
        "<|answer_state|>": 50274,
        "<|graph_miss|>": 50275,
        **{f"<|rel_{index}|>": 50276 + index for index in range(16)},
        "<|relation_start|>": 50292,
        "<|relation_end|>": 50293,
        "<|page_start|>": 50294,
        "<|page_end|>": 50295,
    }
)
_FROZEN_PROTOCOL_ATTRIBUTES = MappingProxyType(
    {
        "DB_START": 50257,
        "DB_RETRIEVE": 50258,
        "DB_END": 50259,
        "EOT": 50260,
        "GRAPH_START": 50261,
        "GRAPH_READ": 50262,
        "GRAPH_RETURN": 50263,
        "GRAPH_END": 50264,
        "GRAPH_HALT": 50265,
        "GRAPH_NOOP": 50266,
        "DIR_OUT": 50271,
        "DIR_IN": 50272,
        "GRAPH_STEP": 50273,
        "ANSWER_STATE": 50274,
        "GRAPH_MISS": 50275,
        "RELATION_START": 50292,
        "RELATION_END": 50293,
        "PAGE_START": 50294,
        "PAGE_END": 50295,
    }
)


class ReportingInterfaceUnavailable(RuntimeError):
    """The installed reporting core cannot safely replay runner evidence."""


class ValidationInterfaceUnavailable(RuntimeError):
    """The installed validation core lacks the externally rooted contract."""


def _require_frozen_repository_protocol(tokenizer) -> None:
    if (
        getattr(tokenizer, "VOCAB_SIZE", None) != 50304
        or tuple(getattr(tokenizer, "SLOTS", ())) != _FROZEN_SLOT_TOKENS
        or getattr(tokenizer, "RELATIONS", None) != _FROZEN_RELATION_TOKENS
        or getattr(tokenizer, "graph_special_tokens", None)
        != _FROZEN_GRAPH_SPECIAL_TOKENS
        or any(
            getattr(tokenizer, attribute, None) != token_id
            for attribute, token_id in _FROZEN_PROTOCOL_ATTRIBUTES.items()
        )
    ):
        raise ValidationInterfaceUnavailable(
            "repository tokenizer graph action protocol and relation "
            "vocabulary are not globally frozen"
        )


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    study_lock_sha256: str
    checkpoint_sha256: str
    condition_id: str
    seed: int
    item_count: int
    report_sha256: str
    optimizer_step: int | None = None
    output_id: str | None = None
    selected_provider: str | None = None
    production_qualified: bool = False
    snapshot_sha256: str | None = None


@dataclass(frozen=True)
class PreflightResult:
    study_lock_sha256: str
    checkpoint_sha256: str
    condition_id: str
    seed: int
    item_count: int
    optimizer_step: int | None = None
    output_id: str | None = None
    selected_provider: str | None = None
    snapshot_sha256: str | None = None


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
    typed: Any
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
    run: Path
    release: Path
    lock: _StudyLockView
    validity_content: bytes
    binding: RunBinding
    checkpoint: _CheckpointView
    selected_ids: tuple[str, ...]
    items_content: bytes
    items: Mapping[str, ItemRecord]
    stores_content: bytes
    stores: Mapping[str, StoreRecord]


@dataclass(frozen=True)
class _PreparedEvaluationV3:
    run: Path
    release: Path
    binding: RunBindingV3
    run_content: bytes
    lock_content: bytes
    snapshot_content: bytes
    release_manifest_content: bytes
    release_manifest: Mapping[str, Any]
    selected_ids: tuple[str, ...]
    items_content: bytes
    items: Mapping[str, ItemRecord]
    stores_content: bytes
    stores: Mapping[str, StoreRecord]


@dataclass(frozen=True)
class Submission:
    """One model answer and its fixed-width action trace."""

    item_id: str
    answer: str
    actions: tuple[ActionSlot, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id:
            raise ValueError("submission item_id must be a non-empty string")
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("submission answer must be a non-empty string")
        object.__setattr__(self, "answer", self.answer.strip())
        object.__setattr__(self, "actions", validate_action_slots(self.actions))

    @property
    def proof(self) -> tuple[ActionSlot, ...]:
        return self.actions


class ModelAdapter(Protocol):
    """Injected model boundary; sealed records never cross this interface."""

    def generate(
        self,
        item: ItemRecord,
        store: StoreRecord | None,
    ) -> Submission: ...


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

    def generate(
        self,
        item: ItemRecord,
        store: StoreRecord | None,
    ) -> Submission:
        del store
        item_id = getattr(item, "item_id", None)
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("model-visible item has no item_id")
        try:
            return self._submissions[item_id]
        except KeyError as exc:
            raise ValueError(f"fixture adapter missing item: {item_id}") from exc


class RepositoryGPTAdapter:
    """Checkpoint-backed adapter for the repository's frozen graph protocol."""

    _MAX_ANSWER_TOKENS = 128

    def __init__(self, model, tokenizer, device) -> None:
        import torch

        _require_frozen_repository_protocol(tokenizer)
        if model.cfg.vocab_size != 50304:
            raise ValueError(
                "repository model/tokenizer vocabulary architecture mismatch"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self._torch = torch
        terminators = {50256, tokenizer.GRAPH_START, tokenizer.EOT}
        self._answer_token_ids = torch.tensor(
            [
                *(token_id for token_id in range(50296) if token_id not in terminators),
                50256,
                tokenizer.GRAPH_START,
                tokenizer.EOT,
            ],
            dtype=torch.long,
            device=self.device,
        )

    @staticmethod
    def _model_config(config: Mapping[str, Any]):
        from train.model import GPTConfig, PRESETS

        model_value = config.get("model")
        if isinstance(model_value, str):
            try:
                model_config = replace(PRESETS[model_value])
            except KeyError as exc:
                raise ValueError(
                    f"unknown repository model architecture: {model_value}"
                ) from exc
        elif isinstance(model_value, Mapping):
            try:
                model_config = GPTConfig(**dict(model_value))
            except (AssertionError, TypeError, ValueError) as exc:
                raise ValueError("repository model architecture is invalid") from exc
        else:
            raise ValueError("repository model architecture must be a preset or object")
        if "ctx" in config:
            ctx = config["ctx"]
            if isinstance(ctx, bool) or not isinstance(ctx, int) or ctx <= 0:
                raise ValueError("repository model context must be a positive integer")
            model_config = replace(model_config, ctx=ctx)
        try:
            _ = model_config.head_dim
        except AssertionError as exc:
            raise ValueError(
                "repository model architecture head size mismatch"
            ) from exc
        if (
            model_config.n_layer < 0
            or model_config.n_head <= 0
            or model_config.d_model <= 0
            or model_config.ctx <= 0
        ):
            raise ValueError("repository model architecture values are invalid")
        return model_config

    @staticmethod
    def _device(requested: str):
        import torch

        if requested not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be cpu, cuda, or mps")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("requested MPS device is unavailable")
        return torch.device(requested)

    @classmethod
    def from_bound_run(
        cls,
        run: str | Path,
        binding: RunBinding | RunBindingV3,
        device: str,
    ) -> "RepositoryGPTAdapter":
        """Load only hash-bound config/checkpoint bytes into the repository GPT."""

        if isinstance(binding, RunBindingV3):
            return cls._from_bound_snapshot(run, binding, device)
        import torch
        from train.model import GPT, GPTConfig
        from train.tokenizer import get_tok

        if not isinstance(binding, RunBinding):
            raise TypeError("repository adapter requires a validated RunBinding")
        run_root = _directory(run, "run")
        config_path = _relative_file(
            run_root,
            binding.configuration_path,
            "configuration_path",
        )
        checkpoint_path = _relative_file(
            run_root,
            binding.checkpoint_path,
            "checkpoint_path",
        )
        config_content = _read_regular_file(config_path, "configuration")
        checkpoint_content = _read_regular_file(checkpoint_path, "checkpoint")
        if hashlib.sha256(config_content).hexdigest() != binding.configuration_sha256:
            raise ValueError("configuration hash mismatch while loading model")
        if hashlib.sha256(checkpoint_content).hexdigest() != binding.checkpoint_sha256:
            raise ValueError("checkpoint hash mismatch while loading model")
        config = _parse_config(config_content)
        if _integer(config.get("seed"), "configuration seed") != binding.seed:
            raise ValueError("configuration seed does not match bound run")
        if (
            _condition(config.get("condition"), "configuration condition")
            != binding.condition_id
        ):
            raise ValueError("configuration condition does not match bound run")
        expected_config = cls._model_config(config)

        try:
            state = torch.load(
                io.BytesIO(checkpoint_content),
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise ValueError(
                "checkpoint state could not be safely loaded"
            ) from exc
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint state must be an object")
        state_dict = state.get("model")
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ValueError("checkpoint model state is missing")

        checkpoint_architectures = []
        checkpoint_config = state.get("cfg")
        if checkpoint_config is not None:
            if not isinstance(checkpoint_config, Mapping):
                raise ValueError("checkpoint config state must be an object")
            if (
                _integer(checkpoint_config.get("seed"), "checkpoint config seed")
                != binding.seed
            ):
                raise ValueError("checkpoint config seed mismatch")
            if (
                _condition(
                    checkpoint_config.get("condition"),
                    "checkpoint config condition",
                )
                != binding.condition_id
            ):
                raise ValueError("checkpoint config condition mismatch")
            checkpoint_architectures.append(cls._model_config(checkpoint_config))
        raw_model_config = state.get("model_cfg")
        if raw_model_config is not None:
            if not isinstance(raw_model_config, Mapping):
                raise ValueError("checkpoint model_cfg state must be an object")
            try:
                checkpoint_architectures.append(GPTConfig(**dict(raw_model_config)))
            except (AssertionError, TypeError, ValueError) as exc:
                raise ValueError(
                    "checkpoint model_cfg architecture is invalid"
                ) from exc
        if not checkpoint_architectures:
            raise ValueError("checkpoint lacks bound architecture metadata")
        expected_architecture = asdict(expected_config)
        if any(
            asdict(checkpoint_architecture) != expected_architecture
            for checkpoint_architecture in checkpoint_architectures
        ):
            raise ValueError("checkpoint/config architecture mismatch")

        try:
            model = GPT(expected_config)
            model.load_state_dict(dict(state_dict), strict=True)
        except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("checkpoint model state mismatch") from exc
        resolved_device = cls._device(device)
        try:
            model.to(resolved_device).eval()
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "checkpoint could not be loaded on selected device"
            ) from exc
        return cls(model, get_tok(), resolved_device)

    @classmethod
    def _from_bound_snapshot(
        cls,
        run: str | Path,
        binding: RunBindingV3,
        device: str,
    ) -> "RepositoryGPTAdapter":
        """Load one exact model-only Trainer snapshot."""

        from train.model import GPT, GPTConfig
        from train.tokenizer import get_tok

        run_root = _directory(run, "run")
        model_snapshot_path = _relative_file(
            run_root,
            binding.snapshot_path,
            "snapshot_path",
        )
        snapshot_content = _read_v3_input(
            model_snapshot_path,
            "model snapshot",
            max_bytes=_MAX_MODEL_SNAPSHOT_BYTES,
        )
        if (
            hashlib.sha256(snapshot_content).hexdigest()
            != binding.snapshot_sha256
        ):
            raise ValueError("model snapshot hash mismatch while loading model")
        state = _parse_bound_model_snapshot(snapshot_content, binding)
        raw_model_config = state["model_cfg"]
        try:
            expected_config = GPTConfig(**dict(raw_model_config))
            _ = expected_config.head_dim
        except (AssertionError, TypeError, ValueError) as exc:
            raise ValueError(
                "model snapshot model_cfg architecture is invalid"
            ) from exc
        state_dict = state.get("model")
        try:
            model = GPT(expected_config)
            model.load_state_dict(dict(state_dict), strict=True)
        except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("model snapshot state mismatch") from exc
        resolved_device = cls._device(device)
        try:
            model.to(resolved_device).eval()
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "model snapshot could not be loaded on selected device"
            ) from exc
        return cls(model, get_tok(), resolved_device)

    def _step(self, token_id: int, cache):
        if not 0 <= token_id < self.model.cfg.vocab_size:
            raise ValueError("generated token is outside the frozen vocabulary")
        if cache is not None and getattr(cache, "pos", 0) >= self.model.cfg.ctx:
            raise ValueError("model interaction exceeds bound context")
        value = self._torch.tensor(
            [[token_id]],
            dtype=self._torch.long,
            device=self.device,
        )
        try:
            logits, cache = self.model.forward_step(value, cache)
        except (AssertionError, RuntimeError) as exc:
            raise ValueError("model interaction exceeds bound architecture") from exc
        return logits[0, -1], cache

    def _force(self, token_ids, cache):
        logits = None
        for token_id in token_ids:
            logits, cache = self._step(int(token_id), cache)
        if logits is None:
            raise ValueError("cannot force an empty protocol sequence")
        return logits, cache

    def _choose(self, logits, allowed) -> int:
        choices = tuple(int(token_id) for token_id in allowed)
        if not choices:
            raise ValueError("frozen token class is empty")
        index = self._torch.tensor(
            choices,
            dtype=self._torch.long,
            device=self.device,
        )
        return choices[int(logits.index_select(0, index).argmax())]

    def _generate_action(self, logits, cache, *, start_ready: bool, reads: int):
        tok = self.tokenizer
        if not start_ready:
            logits, cache = self._step(tok.GRAPH_START, cache)
        source_token = self._choose(logits, tok.SLOTS)
        logits, cache = self._step(source_token, cache)
        relation_token = self._choose(logits, tok.RELATIONS.values())
        logits, cache = self._step(relation_token, cache)
        direction_token = self._choose(logits, (tok.DIR_OUT, tok.DIR_IN))
        logits, cache = self._step(direction_token, cache)
        operations = (
            (tok.GRAPH_HALT, tok.GRAPH_NOOP)
            if reads >= 10
            else (tok.GRAPH_READ, tok.GRAPH_NOOP, tok.GRAPH_HALT)
        )
        operation_token = self._choose(logits, operations)
        logits, cache = self._step(operation_token, cache)
        end_token = self._choose(logits, (tok.GRAPH_END,))
        logits, cache = self._step(end_token, cache)

        source_slot = tok.SLOTS.index(source_token)
        relation_id = next(
            relation
            for relation, token_id in tok.RELATIONS.items()
            if token_id == relation_token
        )
        direction = "out" if direction_token == tok.DIR_OUT else "in"
        if operation_token == tok.GRAPH_READ:
            action = ActionSlot(source_slot, relation_id, direction, ActionOp.READ)
        elif operation_token == tok.GRAPH_HALT:
            action = ActionSlot(None, None, None, ActionOp.HALT)
        else:
            action = ActionSlot(None, None, None, ActionOp.NOOP)
        return action, source_slot, relation_id, direction, logits, cache

    def _force_noop(self, logits, cache, *, start_ready: bool):
        del logits
        tok = self.tokenizer
        token_ids = [
            tok.GRAPH_START,
            tok.SLOTS[0],
            tok.RELATIONS["r0"],
            tok.DIR_OUT,
            tok.GRAPH_NOOP,
            tok.GRAPH_END,
        ]
        if start_ready:
            token_ids = token_ids[1:]
        return self._force(token_ids, cache)

    def _return_tokens(
        self,
        item: ItemRecord,
        store: StoreRecord | None,
        slots: list[str | None],
        action: ActionSlot,
    ) -> list[int]:
        from corpusgen.graph_records import GraphRow
        from corpusgen.graph_trace import serialize_return

        returned = None
        fact_id = None
        if action.op is ActionOp.READ:
            source_slot = action.source_slot
            if source_slot is None:
                raise AssertionError("validated read lacks source slot")
            source_id = slots[source_slot]
            row = (
                None
                if source_id is None or store is None
                else store.lookup(
                    source_id,
                    str(action.relation_id),
                    str(action.direction),
                )
            )
            if row is not None:
                returned = GraphRow(
                    source_id=row.source_id,
                    relation_id=row.relation_id,
                    direction=row.direction,
                    target_kind=row.target_kind,
                    target=row.target,
                    qualifiers=tuple(sorted(row.qualifiers.items())),
                    provenance_id=store.store_id,
                )
                fact_id = (
                    f"{item.world_id}:{source_id}:"
                    f"{action.relation_id}:{action.direction}"
                )
                if row.target_kind == "entity":
                    slots[source_slot] = row.target
        segments = serialize_return(returned, fact_id)
        token_ids, _, _ = self.tokenizer.encode_tagged_segments(segments)
        if not token_ids:
            raise ValueError("repository return protocol encoded no tokens")
        return token_ids

    def _decode_answer(self, logits, cache, *, final: bool):
        tok = self.tokenizer
        answer_tokens: list[int] = []
        terminators = {50256, tok.GRAPH_START, tok.EOT}
        for _ in range(self._MAX_ANSWER_TOKENS):
            selected_logits = logits.index_select(0, self._answer_token_ids)
            token_id = int(self._answer_token_ids[int(selected_logits.argmax())].item())
            logits, cache = self._step(token_id, cache)
            if token_id in terminators:
                if not answer_tokens:
                    raise ValueError("model produced an empty answer")
                if not final and token_id != tok.GRAPH_START:
                    raise ValueError(
                        "model ended before completing twelve action slots"
                    )
                answer = _parse_candidate_response(
                    tok.decode(answer_tokens),
                    final=final,
                )
                return answer, logits, cache, token_id == tok.GRAPH_START
            answer_tokens.append(token_id)
        raise ValueError("model answer exceeded the frozen token budget")

    def generate(
        self,
        item: ItemRecord,
        store: StoreRecord | None,
    ) -> Submission:
        if not isinstance(item, ItemRecord):
            raise TypeError("repository adapter requires a validated ItemRecord")
        if item.memory_mode is MemoryMode.MEMORY_OFF:
            if store is not None:
                raise ValueError("memory_off adapter boundary must receive no store")
        elif not isinstance(store, StoreRecord):
            raise TypeError(
                "memory_on adapter boundary requires a validated StoreRecord"
            )
        if store is not None and (
            store.store_id != item.store_id or store.world_id != item.world_id
        ):
            raise ValueError("model-visible item/store identity mismatch")
        if store is not None and any(
            row.relation_id not in self.tokenizer.RELATIONS for row in store.rows
        ):
            raise ValueError("store uses an unsupported frozen relation vocabulary")

        prompt_ids = self.tokenizer.encode(item.prompt)
        if not prompt_ids:
            raise ValueError("model-visible prompt encodes to no tokens")
        if len(prompt_ids) >= self.model.cfg.ctx:
            raise ValueError("model-visible prompt exceeds bound context")
        prompt = self._torch.tensor(
            [prompt_ids],
            dtype=self._torch.long,
            device=self.device,
        )
        try:
            with self._torch.no_grad():
                raw_logits, cache = self.model.forward_step(prompt, None)
        except (AssertionError, RuntimeError) as exc:
            raise ValueError("model-visible prompt exceeds bound architecture") from exc
        logits = raw_logits[0, -1]
        slots = list(item.initial_slots)
        actions = []
        answers = []
        reads = 0
        halted = False
        start_ready = False
        with self._torch.no_grad():
            for slot_index in range(12):
                if halted:
                    action = ActionSlot(None, None, None, ActionOp.NOOP)
                    logits, cache = self._force_noop(
                        logits,
                        cache,
                        start_ready=start_ready,
                    )
                else:
                    (
                        action,
                        _source_slot,
                        _relation_id,
                        _direction,
                        logits,
                        cache,
                    ) = self._generate_action(
                        logits,
                        cache,
                        start_ready=start_ready,
                        reads=reads,
                    )
                    if action.op is ActionOp.READ:
                        reads += 1
                    elif action.op is ActionOp.HALT:
                        halted = True
                actions.append(action)
                logits, cache = self._force(
                    self._return_tokens(item, store, slots, action),
                    cache,
                )
                logits, cache = self._step(
                    self.tokenizer.ANSWER_STATE,
                    cache,
                )
                answer, logits, cache, start_ready = self._decode_answer(
                    logits,
                    cache,
                    final=slot_index == 11,
                )
                answers.append(answer)
        if reads > 10:
            raise AssertionError("repository adapter exceeded read cap")
        return Submission(item.item_id, answers[-1], tuple(actions))


def _parse_candidate_response(value: str, *, final: bool) -> str:
    if final:
        final_match = _FINAL_RESPONSE_RE.fullmatch(value)
        if final_match is not None:
            candidate, final_answer = final_match.groups()
            if candidate != final_answer:
                raise ValueError(
                    "model candidate response final value disagrees with candidate"
                )
            return candidate
    match = _CANDIDATE_RESPONSE_RE.fullmatch(value)
    if match is None:
        raise ValueError("model candidate response has invalid framing")
    return match.group(1)


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


def _canonical_structure_sha256(value: object, name: str) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, UnicodeError, ValueError) as exc:
        raise ValueError(f"{name} is not canonical JSON data") from exc
    return hashlib.sha256(payload).hexdigest()


def _parse_bound_model_snapshot(
    content: bytes,
    binding: RunBindingV3,
) -> Mapping[str, Any]:
    import torch

    try:
        state = torch.load(
            io.BytesIO(content),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ValueError(
            "model snapshot could not be safely loaded with weights_only"
        ) from exc
    if type(state) is not dict:
        raise ValueError("model snapshot must be an exact dictionary")
    fields = set(state)
    if fields != _MODEL_SNAPSHOT_FIELDS_V2:
        if fields & {"cfg", "data", "opt", "rng_by_rank"}:
            raise ValueError(
                "full optimizer/RNG checkpoint is not a model-only snapshot"
            )
        raise ValueError("model snapshot fields are not exact")
    if (
        type(state["snapshot_version"]) is not int
        or state["snapshot_version"] != binding.snapshot_version
    ):
        raise ValueError("model snapshot version binding mismatch")
    if type(state["step"]) is not int or state["step"] != binding.optimizer_step:
        raise ValueError("model snapshot step binding mismatch")
    if (
        type(state["world_size"]) is not int
        or state["world_size"] != binding.world_size
    ):
        raise ValueError("model snapshot world_size binding mismatch")
    config_fingerprint = _sha256(
        state["config_fingerprint"],
        "model snapshot config fingerprint",
    )
    if config_fingerprint != binding.config_fingerprint:
        raise ValueError("model snapshot config fingerprint mismatch")
    model_cfg = state["model_cfg"]
    if type(model_cfg) is not dict:
        raise ValueError("model snapshot model_cfg must be an exact dictionary")
    model_cfg_sha256 = _canonical_structure_sha256(
        model_cfg,
        "model snapshot model_cfg",
    )
    if model_cfg_sha256 != binding.model_config_sha256:
        raise ValueError("model snapshot model config binding mismatch")
    data_provenance = state["data_provenance"]
    if type(data_provenance) is not dict:
        raise ValueError(
            "model snapshot data_provenance must be an exact dictionary"
        )
    data_provenance_sha256 = _canonical_structure_sha256(
        data_provenance,
        "model snapshot data_provenance",
    )
    if data_provenance_sha256 != binding.data_provenance_sha256:
        raise ValueError("model snapshot data provenance binding mismatch")
    identity = _strict_mapping(
        state["study_identity"],
        _MODEL_SNAPSHOT_IDENTITY_FIELDS_V2,
        "model snapshot study identity",
    )
    expected_identity = {
        "arm": binding.arm.value,
        "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
        "data_provenance_sha256": data_provenance_sha256,
        "model_cfg_sha256": model_cfg_sha256,
        "run_id": binding.training_run_id,
        "seed": binding.seed,
        "tokens_per_step": binding.tokens_per_step,
    }
    if identity != expected_identity:
        raise ValueError("model snapshot study arm/provenance identity mismatch")
    model = state["model"]
    if not isinstance(model, Mapping) or not model:
        raise ValueError("model snapshot state is missing")
    return state


def _default_execution_probe(device: str) -> Mapping[str, Any]:
    import torch

    device_type = torch.device(device).type
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    if device_type == "cuda" and cuda_available:
        selected = torch.cuda.current_device()
        device_name = str(torch.cuda.get_device_name(selected))
        capability = list(torch.cuda.get_device_capability(selected))
    else:
        device_name = ""
        capability = []
    return {
        "device_type": device_type,
        "cuda_available": cuda_available,
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda or "unavailable"),
        "device_count": device_count,
        "device_name": device_name,
        "device_capability": capability,
    }


def _execution_probe_value(
    device: str,
    execution_probe,
) -> dict[str, Any]:
    probe = _strict_mapping(
        (execution_probe or _default_execution_probe)(device),
        frozenset(
            {
                "device_type",
                "cuda_available",
                "torch_version",
                "cuda_version",
                "device_count",
                "device_name",
                "device_capability",
            }
        ),
        "evaluator execution probe",
    )
    if (
        not isinstance(probe["device_type"], str)
        or type(probe["cuda_available"]) is not bool
        or not isinstance(probe["torch_version"], str)
        or not probe["torch_version"]
        or not isinstance(probe["cuda_version"], str)
        or not probe["cuda_version"]
        or type(probe["device_count"]) is not int
        or probe["device_count"] < 0
        or not isinstance(probe["device_name"], str)
        or not isinstance(probe["device_capability"], list)
        or any(
            type(part) is not int or part < 0
            for part in probe["device_capability"]
        )
    ):
        raise ValueError("evaluator execution probe values are invalid")
    return dict(probe)


def _nonproduction_execution_identity(
    *,
    binding: RunBindingV3,
    device: str,
    model_adapter: ModelAdapter | None,
    execution_probe,
) -> dict[str, Any]:
    return {
        "production_qualified": False,
        "adapter_kind": (
            "repository"
            if model_adapter is None
            or isinstance(model_adapter, RepositoryGPTAdapter)
            else "fixture"
        ),
        "selected_provider": binding.selected_provider,
        "profile_sha256": binding.evaluator_profile_sha256,
        "runtime_lock_sha256": binding.evaluator_runtime_lock_sha256,
        "environment_receipt_sha256": (
            binding.evaluator_environment_receipt_sha256
        ),
        **_execution_probe_value(device, execution_probe),
    }


def _authenticate_evaluator_execution(
    *,
    run: Path,
    binding: RunBindingV3,
    device: str,
    identity_verifier,
    execution_probe,
) -> dict[str, Any]:
    if device != "cuda":
        raise ValueError(
            "provider-qualified publication requires actual cuda execution"
        )
    profile_path = _relative_file(
        run,
        binding.evaluator_profile_path,
        "evaluator_profile_path",
    )
    runtime_lock_path = _relative_file(
        run,
        binding.evaluator_runtime_lock_path,
        "evaluator_runtime_lock_path",
    )
    environment_path = _relative_file(
        run,
        binding.evaluator_environment_receipt_path,
        "evaluator_environment_receipt_path",
    )
    profile_data = _read_v3_input(
        profile_path,
        "evaluator profile",
        max_bytes=_MAX_EXECUTION_EVIDENCE_BYTES,
    )
    runtime_lock_data = _read_v3_input(
        runtime_lock_path,
        "evaluator runtime lock",
        max_bytes=_MAX_EXECUTION_EVIDENCE_BYTES,
    )
    environment_data = _read_v3_input(
        environment_path,
        "evaluator environment receipt",
        max_bytes=_MAX_EXECUTION_EVIDENCE_BYTES,
    )
    if hashlib.sha256(profile_data).hexdigest() != (
        binding.evaluator_profile_sha256
    ):
        raise ValueError("evaluator profile hash binding mismatch")
    if hashlib.sha256(runtime_lock_data).hexdigest() != (
        binding.evaluator_runtime_lock_sha256
    ):
        raise ValueError("evaluator runtime-lock hash binding mismatch")
    if hashlib.sha256(environment_data).hexdigest() != (
        binding.evaluator_environment_receipt_sha256
    ):
        raise ValueError("evaluator environment-receipt hash mismatch")
    profile = parse_aws_gpu_profile_bytes(profile_data)
    runtime_lock = parse_aws_runtime_lock_bytes(runtime_lock_data)
    if (
        profile.profile_id != binding.evaluator_profile_id
        or profile.provider != binding.selected_provider
        or profile.sha256 != binding.evaluator_profile_sha256
        or runtime_lock.profile_sha256 != profile.sha256
        or runtime_lock.sha256 != binding.evaluator_runtime_lock_sha256
    ):
        raise ValueError("evaluator profile/runtime identity is crossed")
    environment = _strict_mapping(
        _canonical_object(
            environment_data,
            "evaluator environment receipt",
        ),
        frozenset(AWS_ENVIRONMENT_RECEIPT_V2_FIELDS),
        "evaluator environment receipt",
    )
    if (
        type(environment["schema_version"]) is not int
        or environment["schema_version"] != 2
        or environment["receipt_type"] != "memorysplit-aws-environment-v2"
        or environment["provider"] != profile.provider
        or environment["profile_sha256"] != profile.sha256
        or environment["runtime_lock_sha256"] != runtime_lock.sha256
        or environment["control_bundle_sha256"]
        != runtime_lock.control_bundle_sha256
        or environment["source_commit"] != runtime_lock.source_commit
        or environment["source_tree"] != runtime_lock.source_tree
        or environment["ami_id"] != runtime_lock.ami_id
        or environment["container_image"] != runtime_lock.container_image
        or environment["container_image_digest"]
        != runtime_lock.container_image_digest
    ):
        raise ValueError("evaluator environment differs from runtime authority")
    runtime_facts = _strict_mapping(
        environment["runtime_facts"],
        frozenset(AWS_RUNTIME_VERSION_FIELDS),
        "evaluator environment runtime facts",
    )
    if runtime_facts != dict(runtime_lock.versions):
        raise ValueError("evaluator runtime facts differ from runtime lock")
    identity = environment["aws_instance_identity_document"]
    pkcs7 = environment["aws_instance_identity_pkcs7"]
    if not isinstance(identity, Mapping) or not isinstance(pkcs7, str):
        raise ValueError("evaluator AWS identity evidence is invalid")
    identity_fields = set(identity)
    if (
        not _AWS_IDENTITY_REQUIRED_FIELDS <= identity_fields
        or not identity_fields
        <= _AWS_IDENTITY_REQUIRED_FIELDS | _AWS_IDENTITY_OPTIONAL_FIELDS
        or any(
            not isinstance(identity[field], str) or not identity[field]
            for field in _AWS_IDENTITY_REQUIRED_FIELDS
        )
        or _AWS_ACCOUNT_ID_RE.fullmatch(identity["accountId"]) is None
        or _AWS_INSTANCE_ID_RE.fullmatch(identity["instanceId"]) is None
        or _AWS_REGION_RE.fullmatch(identity["region"]) is None
        or not isinstance(environment["boot_id"], str)
        or _BOOT_ID_RE.fullmatch(environment["boot_id"]) is None
    ):
        raise ValueError("evaluator AWS identity fields are invalid")
    try:
        decoded_pkcs7 = base64.b64decode(pkcs7, validate=True)
    except ValueError as exc:
        raise ValueError("evaluator AWS identity signature is invalid") from exc
    if (
        not decoded_pkcs7
        or base64.b64encode(decoded_pkcs7).decode("ascii") != pkcs7
        or environment["account_id"] != identity.get("accountId")
        or environment["instance_id"] != identity.get("instanceId")
        or environment["region"] != identity.get("region")
        or environment["ami_id"] != identity.get("imageId")
        or identity.get("architecture") != profile.architecture
    ):
        raise ValueError("evaluator AWS identity fields are crossed")
    if not callable(identity_verifier):
        raise TypeError("evaluator execution identity verifier is required")
    try:
        verified = identity_verifier(
            identity,
            pkcs7,
            environment["region"],
        )
    except Exception as exc:
        raise ValueError(
            "evaluator execution identity verification failed"
        ) from exc
    if verified is not True:
        raise ValueError("evaluator execution identity signature is invalid")
    probe = _execution_probe_value(device, execution_probe)
    if (
        probe["device_type"] != "cuda"
        or probe["cuda_available"] is not True
        or probe["device_count"] != profile.allocated_gpus
        or not probe["device_name"]
        or profile.gpu_model.lower() not in probe["device_name"].lower()
        or len(probe["device_capability"]) != 2
        or probe["torch_version"] != runtime_facts["pytorch"]
        or probe["cuda_version"] != runtime_facts["cuda"]
    ):
        raise ValueError(
            "actual cuda/torch execution differs from evaluator authority"
        )
    return {
        "production_qualified": True,
        "adapter_kind": "repository",
        "selected_provider": binding.selected_provider,
        "profile_id": profile.profile_id,
        "profile_sha256": profile.sha256,
        "runtime_lock_sha256": runtime_lock.sha256,
        "qualification_evidence_sha256": (
            binding.evaluator_qualification_evidence_sha256
        ),
        "environment_receipt_sha256": (
            binding.evaluator_environment_receipt_sha256
        ),
        "canary_receipt_sha256": (
            binding.evaluator_canary_receipt_sha256
        ),
        "approval_receipt_sha256": (
            binding.evaluator_approval_receipt_sha256
        ),
        "approval_public_key_sha256": (
            binding.evaluator_approval_public_key_sha256
        ),
        "account_id": environment["account_id"],
        "instance_id": environment["instance_id"],
        "boot_id": environment["boot_id"],
        "region": environment["region"],
        **probe,
    }


def _read_regular_file(path: Path, name: str) -> bytes:
    try:
        status = path.lstat()
    except FileNotFoundError:
        raise ValueError(f"{name} is missing") from None
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")
    return path.read_bytes()


def _read_v3_input(
    path: Path,
    name: str,
    *,
    max_bytes: int,
) -> bytes:
    return read_secure_regular_file(
        path,
        label=name,
        max_bytes=max_bytes,
    )


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


def _parse_run_binding_content(content: bytes) -> RunBinding | RunBindingV3:
    parsed = _canonical_object(content, "run.json")
    if parsed.get("record_type") == RUN_BINDING_SCHEMA_V3:
        return RunBindingV3.from_dict(parsed)
    if parsed.get("record_type") != RUN_BINDING_SCHEMA:
        raise ValueError("run binding record_type is invalid")
    raw = _strict_mapping(parsed, _RUN_FIELDS, "run binding")
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


def _load_run_binding(run: Path) -> RunBinding | RunBindingV3:
    return _parse_run_binding_content(
        _read_regular_file(run / "run.json", "run binding")
    )


def _hardened_study_lock_api():
    try:
        module = importlib.import_module("evals.confirmatory.study_lock")
    except ImportError as exc:
        raise ValidationInterfaceUnavailable(
            "hardened study-lock validation API unavailable; require "
            "evals.confirmatory.study_lock.StudyLock from f934124"
        ) from exc
    required = (
        "StudyLock",
        "ValidityEvidence",
        "ReadinessResult",
        "evaluate_readiness",
        "VALIDITY_EVIDENCE_SCHEMA",
        "FROZEN_PREREGISTRATION_SHA256",
        "REQUIRED_CONTROL_IDS",
        "REQUIRED_RECEIPTS",
    )
    if any(not hasattr(module, name) for name in required):
        raise ValidationInterfaceUnavailable(
            "hardened study-lock validation API unavailable; require "
            "StudyLock plus frozen preregistration/control/receipt contracts"
        )
    if (
        not callable(getattr(module.StudyLock, "from_dict", None))
        or not callable(getattr(module.ValidityEvidence, "from_dict", None))
        or not callable(module.evaluate_readiness)
    ):
        raise ValidationInterfaceUnavailable(
            "hardened study-lock/readiness validation API unavailable"
        )
    return module


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
    api = _hardened_study_lock_api()
    typed = api.StudyLock.from_dict(raw)
    wrong_lock_sha256 = "0" * 64 if actual != "0" * 64 else "1" * 64
    readiness_probe = api.ValidityEvidence.from_dict(
        {
            "record_type": api.VALIDITY_EVIDENCE_SCHEMA,
            "schema_version": CONTRACT_VERSION,
            "study_lock_sha256": wrong_lock_sha256,
            "preregistration_sha256": typed.preregistration_sha256,
            "receipts": [],
        }
    )
    try:
        api.evaluate_readiness(typed, readiness_probe)
    except ValueError:
        pass
    else:
        raise ValidationInterfaceUnavailable(
            "hardened externally rooted readiness validation from f934124 "
            "is unavailable"
        )
    release_binding = typed.release.to_dict()
    approvals = tuple(approval.to_dict() for approval in typed.checkpoints)
    return _StudyLockView(
        typed=typed,
        raw=MappingProxyType(dict(raw)),
        release=MappingProxyType(dict(release_binding)),
        approvals=tuple(approvals),
        content=content,
        sha256=actual,
    )


def _load_complete_readiness(release: Path, lock: _StudyLockView) -> bytes:
    content = _read_regular_file(
        release / "validity.json",
        "validity.json",
    )
    raw = _canonical_object(content, "validity.json")
    api = _hardened_study_lock_api()
    typed = api.ValidityEvidence.from_dict(raw)
    readiness = api.evaluate_readiness(lock.typed, typed)
    if (
        not isinstance(readiness, api.ReadinessResult)
        or type(readiness.complete) is not bool
        or type(readiness.valid) is not bool
    ):
        raise ValidationInterfaceUnavailable(
            "hardened study-lock readiness result contract is unavailable"
        )
    if not readiness.complete or not readiness.valid:
        raise ValueError(
            "study-lock readiness must be complete and valid before evaluation"
        )
    return content


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
    scored_type = getattr(metrics, "_ScoredItemOutcome", None)
    from_solver_replay = getattr(scored_type, "_from_solver_replay", None)
    aggregate = getattr(metrics, "_aggregate_scored_pair_metric", None)
    if (
        not callable(getattr(metrics.ItemOutcome, "from_dict", None))
        or not callable(from_solver_replay)
        or not callable(aggregate)
    ):
        raise ValidationInterfaceUnavailable(
            "hardened internal metrics replay API unavailable; require f934124 "
            "_ScoredItemOutcome._from_solver_replay and "
            "_aggregate_scored_pair_metric"
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
        persisted = metrics.ItemOutcome.from_dict(outcome_dict)
        scored.append(from_solver_replay(persisted, verification))

    grouped: dict[tuple[Any, Any], list[Any]] = defaultdict(list)
    for row in scored:
        grouped[(row.memory_mode, row.control)].append(row)
    summaries = []
    for key in sorted(grouped, key=lambda value: (value[0].value, value[1].value)):
        rows = grouped[key]
        row_items = {row.item_id: items[row.item_id] for row in rows}
        summary = aggregate(
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


def _study_submission_outcome(
    *,
    item: ItemRecord,
    submission: Submission,
    binding: RunBindingV3,
) -> dict[str, Any]:
    return {
        "record_type": metrics.STUDY_OUTCOME_SCHEMA,
        "schema_version": STUDY_CONTRACT_VERSION,
        "item_id": item.item_id,
        "pair_id": item.pair_id,
        "twin": item.twin.value,
        "stratum": item.stratum.value,
        "family": item.family.value,
        "seed": binding.seed,
        "world_id": item.world_id,
        "checkpoint_sha256": binding.snapshot_sha256,
        "arm": binding.arm.value,
        "condition_id": binding.condition_id,
        "optimizer_step": binding.optimizer_step,
        "raw_token_count": binding.raw_token_count,
        "memory_mode": item.memory_mode.value,
        "control": item.control.value,
        "submitted_answer": submission.answer,
        "submitted_proof": [
            action.to_dict() for action in submission.actions
        ],
    }


def _aggregate_scored_study_pairs(
    rows,
    *,
    binding: RunBindingV3,
):
    values = tuple(rows)
    if not values:
        raise ValueError("v3 pair metric requires outcomes")
    grouped: dict[tuple[int, str, str], dict[Twin, tuple[Any, Any]]] = (
        defaultdict(dict)
    )
    seen_items = set()
    for outcome, verification in values:
        if outcome.item_id in seen_items:
            raise ValueError("v3 pair metric contains a duplicate item")
        seen_items.add(outcome.item_id)
        key = outcome.seed, outcome.world_id, outcome.pair_id
        if outcome.twin in grouped[key]:
            raise ValueError("v3 pair metric contains a duplicate twin")
        grouped[key][outcome.twin] = outcome, verification

    stratum_successes: dict[Any, int] = defaultdict(int)
    stratum_totals: dict[Any, int] = defaultdict(int)
    family_successes: dict[Any, int] = defaultdict(int)
    family_totals: dict[Any, int] = defaultdict(int)
    primary_successes = {cell: 0 for cell in metrics.PRIMARY_CELLS}
    primary_totals = {cell: 0 for cell in metrics.PRIMARY_CELLS}
    pair_metadata = (
        "pair_id",
        "stratum",
        "family",
        "seed",
        "world_id",
        "checkpoint_sha256",
        "arm",
        "condition_id",
        "optimizer_step",
        "raw_token_count",
        "memory_mode",
        "control",
    )
    for pair in grouped.values():
        if set(pair) != {Twin.ORIGINAL, Twin.COUNTERFACTUAL}:
            raise ValueError("v3 pair metric requires both twins")
        original, original_verification = pair[Twin.ORIGINAL]
        counterfactual, counterfactual_verification = pair[
            Twin.COUNTERFACTUAL
        ]
        if any(
            getattr(original, field) != getattr(counterfactual, field)
            for field in pair_metadata
        ):
            raise ValueError("v3 pair metric twins have crossed metadata")
        success = (
            original_verification.proof_valid
            and original_verification.answer_valid
            and counterfactual_verification.proof_valid
            and counterfactual_verification.answer_valid
        )
        stratum_totals[original.stratum] += 1
        stratum_successes[original.stratum] += success
        family_totals[original.family] += 1
        family_successes[original.family] += success
        cell = original.family, original.stratum
        if cell in primary_totals:
            primary_totals[cell] += 1
            primary_successes[cell] += success
    missing = [
        f"{family.value}__{stratum.value}"
        for family, stratum in metrics.PRIMARY_CELLS
        if primary_totals[(family, stratum)] == 0
    ]
    if missing:
        raise ValueError(
            f"v3 pair metric is missing required primary cells: {missing}"
        )
    primary_rates = {
        f"{family.value}__{stratum.value}": metrics.Rate(
            primary_successes[(family, stratum)],
            primary_totals[(family, stratum)],
        )
        for family, stratum in metrics.PRIMARY_CELLS
    }
    return metrics.StudyMetricsRecord(
        primary_accuracy=(
            sum(rate.value for rate in primary_rates.values())
            / len(metrics.PRIMARY_CELLS)
        ),
        primary_cells=primary_rates,
        overall_pair_accuracy=metrics.Rate(
            sum(stratum_successes.values()),
            sum(stratum_totals.values()),
        ),
        by_stratum={
            stratum: metrics.Rate(
                stratum_successes[stratum],
                total,
            )
            for stratum, total in stratum_totals.items()
        },
        by_family={
            family: metrics.Rate(
                family_successes[family],
                total,
            )
            for family, total in family_totals.items()
        },
        checkpoint_sha256=binding.snapshot_sha256,
        seed=binding.seed,
        arm=binding.arm,
        condition_id=binding.condition_id,
        optimizer_step=binding.optimizer_step,
        raw_token_count=binding.raw_token_count,
        memory_mode=values[0][0].memory_mode,
        control=values[0][0].control,
    )


def _score_and_summarize_v3(
    *,
    items: Mapping[str, ItemRecord],
    gold: Mapping[str, SealedGoldRecord],
    stores: Mapping[str, StoreRecord],
    binding: RunBindingV3,
    submissions: Mapping[str, Submission],
) -> tuple[bytes, bytes]:
    outcome_dicts = []
    scored = []
    for item_id in submissions:
        item = items[item_id]
        sealed = gold[item_id]
        store = stores[item.store_id]
        if (
            (item.item_id, item.pair_id, item.twin)
            != (sealed.item_id, sealed.pair_id, sealed.twin)
            or item.store_id != store.store_id
            or item.world_id != store.world_id
            or sealed.store_sha256 != store.content_sha256
        ):
            raise ValueError("v3 item/gold/store binding mismatch")
        study_dict = _study_submission_outcome(
            item=item,
            submission=submissions[item_id],
            binding=binding,
        )
        study_outcome = metrics.StudyOutcomeRecord.from_dict(study_dict)
        for field in (
            "item_id",
            "pair_id",
            "twin",
            "stratum",
            "family",
            "world_id",
            "memory_mode",
            "control",
        ):
            if getattr(study_outcome, field) != getattr(item, field):
                raise ValueError(f"v3 output item binding mismatch: {field}")
        for field in (
            "seed",
            "optimizer_step",
            "raw_token_count",
        ):
            if getattr(study_outcome, field) != getattr(binding, field):
                raise ValueError(
                    f"v3 output checkpoint binding mismatch: {field}"
                )
        if study_outcome.checkpoint_sha256 != binding.snapshot_sha256:
            raise ValueError(
                "v3 output model snapshot hash binding mismatch"
            )
        if (
            study_outcome.arm != binding.arm
            or study_outcome.condition_id.value != binding.condition_id
        ):
            raise ValueError("v3 output arm binding mismatch")
        verification = solver_module.verify_proof_and_answer(
            item=item,
            store=store,
            gold=sealed,
            proof=submissions[item_id].actions,
            answer=submissions[item_id].answer,
            solver=_registered_solver(sealed.solver_id),
        )
        scored.append((study_outcome, verification))
        outcome_dicts.append(study_outcome.to_dict())

    grouped: dict[tuple[Any, Any], list[Any]] = defaultdict(list)
    for row, verification in scored:
        grouped[(row.memory_mode, row.control)].append((row, verification))
    summaries = []
    for key in sorted(
        grouped,
        key=lambda value: (value[0].value, value[1].value),
    ):
        rows = grouped[key]
        study_summary = _aggregate_scored_study_pairs(
            rows,
            binding=binding,
        )
        summaries.append(study_summary.to_dict())
    return (
        b"".join(canonical_json_bytes(row) for row in outcome_dicts),
        canonical_json_bytes(
            {
                "record_type": SNAPSHOT_METRICS_SCHEMA_V3,
                "schema_version": STUDY_CONTRACT_VERSION,
                "scope": "single_snapshot",
                "seed": binding.seed,
                "arm": binding.arm.value,
                "optimizer_step": binding.optimizer_step,
                "snapshot_sha256": binding.snapshot_sha256,
                "summaries": summaries,
            }
        ),
    )


def _inference_bytes_v3(binding: RunBindingV3) -> bytes:
    return canonical_json_bytes(
        {
            "record_type": SNAPSHOT_INFERENCE_SCHEMA_V3,
            "schema_version": STUDY_CONTRACT_VERSION,
            "scope": "cohort",
            "snapshot_scope": "single_snapshot",
            "output_id": binding.output_id,
            "seed": binding.seed,
            "arm": binding.arm.value,
            "optimizer_step": binding.optimizer_step,
            "snapshot_sha256": binding.snapshot_sha256,
            "cohort_aggregation_status": "not_implemented",
            "final_conclusion": None,
        }
    )


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
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
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


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish_directory(staging: Path, output: Path) -> None:
    """Atomically rename a directory while refusing every destination collision."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(staging)
    output_bytes = os.fsencode(output)
    if sys.platform == "darwin":
        rename = getattr(libc, "renamex_np", None)
        if rename is None:
            raise RuntimeError("atomic no-replace directory publication is unsupported")
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, output_bytes, 0x00000004)
    elif sys.platform.startswith("linux"):
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise RuntimeError("atomic no-replace directory publication is unsupported")
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, output_bytes, 1)
    else:
        raise RuntimeError("atomic no-replace directory publication is unsupported")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            f"output directory already exists: {output}",
            str(output),
        )
    raise OSError(error_number, os.strerror(error_number), str(output))


def _quarantine_or_clean_staging(staging: Path, parent: Path) -> None:
    if not staging.exists() and not staging.is_symlink():
        return
    try:
        shutil.rmtree(staging)
        _fsync_directory(parent)
        return
    except OSError:
        pass
    quarantine = parent / (
        f".{staging.name.lstrip('.')}.quarantine-{os.getpid()}-{id(staging):x}"
    )
    try:
        _atomic_publish_directory(staging, quarantine)
        _fsync_directory(parent)
    except OSError:
        pass


def _publish_evidence(
    output: Path,
    artifacts: Mapping[str, bytes],
    report: Any,
    expected_study_lock_sha256: str,
) -> None:
    if output.name in {"", ".", ".."}:
        raise ValueError("output must name a directory")
    parent = output.parent
    try:
        parent_status = parent.lstat()
    except FileNotFoundError:
        raise ValueError("output parent directory is missing") from None
    if stat.S_ISLNK(parent_status.st_mode) or not stat.S_ISDIR(parent_status.st_mode):
        raise ValueError("output parent must be a regular non-symlink directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output directory already exists: {output}")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.confirmatory-",
            dir=parent,
        )
    )
    published = False
    try:
        for name in _HARDENED_ARTIFACTS:
            _publish_file(staging / name, artifacts[name])
        reporting.publish_artifact_report(
            staging / "artifact-report.json",
            report,
            artifacts,
            expected_study_lock_sha256=expected_study_lock_sha256,
        )
        _fsync_directory(staging)
        _atomic_publish_directory(staging, output)
        published = True
        try:
            _fsync_directory(parent)
        except OSError:
            quarantine = parent / (
                f".{output.name}.confirmatory-quarantine-{os.getpid()}-{id(output):x}"
            )
            _atomic_publish_directory(output, quarantine)
            _fsync_directory(parent)
            raise
    finally:
        if not published:
            _quarantine_or_clean_staging(staging, parent)


def _snapshot_output_manifest_bytes(
    *,
    binding: RunBindingV3,
    artifacts: Mapping[str, bytes],
    execution_identity_before: Mapping[str, Any],
    execution_identity_after: Mapping[str, Any],
    production_qualified: bool,
) -> bytes:
    if tuple(artifacts) != _V3_OUTPUT_ARTIFACTS:
        raise ValueError("v3 snapshot output artifacts are not exact")
    return canonical_json_bytes(
        {
            "record_type": SNAPSHOT_OUTPUT_SCHEMA_V3,
            "schema_version": STUDY_CONTRACT_VERSION,
            "output_id": binding.output_id,
            "run_binding_sha256": hashlib.sha256(
                artifacts["run.json"]
            ).hexdigest(),
            "study_lock_sha256": binding.study_lock_sha256,
            "sealed_evaluation_release_sha256": (
                binding.sealed_evaluation_release_sha256
            ),
            "provider_selection_sha256": (
                binding.provider_selection_sha256
            ),
            "provider_selection_s3_version_id": (
                binding.provider_selection_s3_version_id
            ),
            "snapshot_sha256": binding.snapshot_sha256,
            "seed": binding.seed,
            "arm": binding.arm.value,
            "optimizer_step": binding.optimizer_step,
            "scope": "single_snapshot",
            "inference_scope": "cohort",
            "final_conclusion": None,
            "production_qualified": production_qualified,
            "publication_class": (
                "provider_qualified"
                if production_qualified
                else "test_only"
            ),
            "execution_identity_before": execution_identity_before,
            "execution_identity_after": execution_identity_after,
            "artifacts": [
                {
                    "path": name,
                    "sha256": hashlib.sha256(artifacts[name]).hexdigest(),
                    "bytes": len(artifacts[name]),
                }
                for name in sorted(artifacts)
            ],
        }
    )


def _output_parent_identity(
    details: os.stat_result,
    *,
    label: str,
) -> tuple[int, int, int, int, int]:
    mode = stat.S_IMODE(details.st_mode)
    if (
        not stat.S_ISDIR(details.st_mode)
        or details.st_uid != os.geteuid()
        or mode & 0o022
        or details.st_nlink < 1
    ):
        raise ValueError(f"{label} is not an owned safe directory")
    return (
        details.st_dev,
        details.st_ino,
        mode,
        details.st_uid,
        details.st_gid,
    )


def _assert_output_parent(
    path: Path,
    descriptor: int,
    expected: tuple[int, int, int, int, int],
) -> None:
    pinned = _output_parent_identity(
        os.fstat(descriptor),
        label="output parent",
    )
    try:
        named = _output_parent_identity(
            os.stat(path, follow_symlinks=False),
            label="output parent",
        )
    except OSError as exc:
        raise ValueError("output parent changed or was replaced") from exc
    if pinned != expected or named != expected:
        raise ValueError("output parent changed or was replaced")


def _publish_file_at(
    directory_fd: int,
    name: str,
    content: bytes,
) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing v3 evidence")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_v3_staging(
    parent_fd: int,
    staging_fd: int,
    staging_name: str,
    written_names: tuple[str, ...],
) -> None:
    for name in reversed(written_names):
        try:
            os.unlink(name, dir_fd=staging_fd)
        except FileNotFoundError:
            pass
    os.close(staging_fd)
    try:
        os.rmdir(staging_name, dir_fd=parent_fd)
    except FileNotFoundError:
        pass


def _publish_v3_evidence(
    *,
    output: Path,
    binding: RunBindingV3,
    artifacts: Mapping[str, bytes],
    execution_identity_before: Mapping[str, Any],
    execution_identity_after: Mapping[str, Any],
    production_qualified: bool,
) -> str:
    if output.name != binding.output_id:
        raise ValueError("output directory name disagrees with output identity")
    if output.name in {"", ".", ".."}:
        raise ValueError("output must name a directory")
    parent = output.parent
    try:
        parent_fd = open_directory(parent, label="output parent")
    except (OSError, ValueError) as exc:
        raise ValueError("output parent is missing or unsafe") from exc
    expected_parent = _output_parent_identity(
        os.fstat(parent_fd),
        label="output parent",
    )
    _assert_output_parent(parent, parent_fd, expected_parent)
    try:
        try:
            os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                f"output directory already exists: {output}"
            )
        manifest = _snapshot_output_manifest_bytes(
            binding=binding,
            artifacts=artifacts,
            execution_identity_before=execution_identity_before,
            execution_identity_after=execution_identity_after,
            production_qualified=production_qualified,
        )
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.confirmatory-v3-",
                dir=parent,
            )
        )
        staging_fd: int | None = None
        written: list[str] = []
        installed = False
        try:
            _assert_output_parent(parent, parent_fd, expected_parent)
            staging_fd = os.open(
                staging.name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            staging_identity = _output_parent_identity(
                os.fstat(staging_fd),
                label="v3 output staging",
            )
            for name in _V3_OUTPUT_ARTIFACTS:
                _publish_file_at(staging_fd, name, artifacts[name])
                written.append(name)
            _publish_file_at(staging_fd, "output.json", manifest)
            written.append("output.json")
            os.fsync(staging_fd)
            _assert_output_parent(parent, parent_fd, expected_parent)
            rename_noreplace_at(
                parent_fd,
                staging.name,
                parent_fd,
                output.name,
            )
            installed = True
            installed = _output_parent_identity(
                os.stat(
                    output.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                ),
                label="installed v3 output",
            )
            if installed != staging_identity:
                raise ValueError("installed v3 output identity was replaced")
            os.fsync(parent_fd)
            _assert_output_parent(parent, parent_fd, expected_parent)
            os.close(staging_fd)
            staging_fd = None
        except BaseException:
            if installed:
                quarantine_name = (
                    f".{output.name}.confirmatory-v3-quarantine-"
                    f"{os.getpid()}-{id(output):x}"
                )
                rename_noreplace_at(
                    parent_fd,
                    output.name,
                    parent_fd,
                    quarantine_name,
                )
                os.fsync(parent_fd)
                if staging_fd is not None:
                    os.close(staging_fd)
                    staging_fd = None
            raise
        finally:
            if staging_fd is not None and not installed:
                _cleanup_v3_staging(
                    parent_fd,
                    staging_fd,
                    staging.name,
                    tuple(written),
                )
    finally:
        os.close(parent_fd)
    return hashlib.sha256(manifest).hexdigest()


def _validate_model_visible_protocol(
    items: Mapping[str, ItemRecord],
    stores: Mapping[str, StoreRecord],
) -> None:
    from train.tokenizer import get_tok

    tok = get_tok()
    _require_frozen_repository_protocol(tok)
    expected_stores = {item.store_id for item in items.values()}
    if set(stores) != expected_stores:
        raise ValueError("sealed store registry is not exactly item-bound")
    for item in items.values():
        store = stores[item.store_id]
        if store.world_id != item.world_id:
            raise ValueError("model-visible item/store world identity mismatch")
    if any(
        row.relation_id not in _FROZEN_RELATION_TOKENS
        for store in stores.values()
        for row in store.rows
    ):
        raise ValueError("store uses an unsupported frozen relation vocabulary")


def _v3_snapshot_for_binding(lock, binding: RunBindingV3):
    matches = [
        snapshot
        for snapshot in lock.snapshots
        if (
            snapshot.seed,
            snapshot.arm,
            snapshot.optimizer_step,
        )
        == (
            binding.seed,
            binding.arm,
            binding.optimizer_step,
        )
    ]
    if len(matches) != 1:
        raise ValueError("study lock does not contain one exact snapshot slot")
    snapshot = matches[0]
    for run_field, snapshot_field in (
        ("snapshot_sha256", "checkpoint_sha256"),
        ("snapshot_s3_object_key", "s3_object_key"),
        ("snapshot_s3_version_id", "s3_version_id"),
        ("checkpoint_receipt_sha256", "checkpoint_receipt_sha256"),
        (
            "checkpoint_receipt_s3_object_key",
            "checkpoint_receipt_s3_object_key",
        ),
        (
            "checkpoint_receipt_s3_version_id",
            "checkpoint_receipt_s3_version_id",
        ),
        ("provider_selection_sha256", "provider_selection_sha256"),
        (
            "provider_selection_s3_version_id",
            "provider_selection_s3_version_id",
        ),
        ("snapshot_version", "snapshot_version"),
        ("training_run_id", "training_run_id"),
        ("config_fingerprint", "config_fingerprint"),
        ("model_config_sha256", "model_config_sha256"),
        ("data_provenance_sha256", "data_provenance_sha256"),
        ("world_size", "world_size"),
        ("tokens_per_step", "tokens_per_step"),
    ):
        if getattr(binding, run_field) != getattr(snapshot, snapshot_field):
            raise ValueError(
                f"run/study-lock snapshot mismatch: {run_field}"
            )
    return snapshot


def _validate_v3_selection_binding(lock, binding: RunBindingV3) -> None:
    selection = lock.provider_selection
    for run_field, selection_field in (
        ("provider_selection_s3_key", "provider_selection_s3_key"),
        ("provider_selection_sha256", "provider_selection_sha256"),
        (
            "provider_selection_s3_version_id",
            "provider_selection_s3_version_id",
        ),
        ("hardware_amendment_sha256", "hardware_amendment_sha256"),
        ("selected_provider", "selected_provider"),
        ("evaluator_profile_id", "profile_id"),
        ("evaluator_profile_sha256", "profile_sha256"),
        ("evaluator_runtime_lock_sha256", "runtime_lock_sha256"),
        (
            "evaluator_qualification_evidence_sha256",
            "qualification_evidence_sha256",
        ),
        (
            "evaluator_environment_receipt_sha256",
            "environment_receipt_sha256",
        ),
        ("evaluator_canary_receipt_sha256", "canary_receipt_sha256"),
        (
            "evaluator_approval_receipt_sha256",
            "approval_receipt_sha256",
        ),
        (
            "evaluator_approval_public_key_sha256",
            "approval_public_key_sha256",
        ),
    ):
        if getattr(binding, run_field) != getattr(selection, selection_field):
            raise ValueError(
                f"run/provider selection mismatch: {run_field}"
            )


def _prepare_evaluation_v3(
    *,
    run_root: Path,
    release: Path,
    expected_study_lock_sha256: str,
    binding: RunBindingV3,
    run_content: bytes,
) -> _PreparedEvaluationV3:
    from evals.confirmatory.study_lock import StudyLockV3

    if run_content != canonical_json_bytes(binding.to_dict()):
        raise ValueError("run.json changed or is not the canonical binding")
    expected_lock = _sha256(
        expected_study_lock_sha256,
        "expected_study_lock_sha256",
    )
    if binding.study_lock_sha256 != expected_lock:
        raise ValueError("run binding study-lock commitment mismatch")
    lock_path = _relative_file(
        run_root,
        binding.study_lock_path,
        "study_lock_path",
    )
    lock_content = _read_v3_input(
        lock_path,
        "study-lock.json",
        max_bytes=_MAX_STUDY_LOCK_BYTES,
    )
    if hashlib.sha256(lock_content).hexdigest() != expected_lock:
        raise ValueError("study lock disagrees with run/external commitment")
    lock = StudyLockV3.from_dict(
        _canonical_object(lock_content, "study-lock.json")
    )
    if lock.sealed_evaluation_release_sha256 != (
        binding.sealed_evaluation_release_sha256
    ):
        raise ValueError("run/study-lock sealed release mismatch")
    _validate_v3_selection_binding(lock, binding)
    _v3_snapshot_for_binding(lock, binding)

    model_snapshot_path = _relative_file(
        run_root,
        binding.snapshot_path,
        "snapshot_path",
    )
    snapshot_content = _read_v3_input(
        model_snapshot_path,
        "model snapshot",
        max_bytes=_MAX_MODEL_SNAPSHOT_BYTES,
    )
    if (
        hashlib.sha256(snapshot_content).hexdigest()
        != binding.snapshot_sha256
    ):
        raise ValueError("model snapshot hash disagrees with run binding")
    _parse_bound_model_snapshot(snapshot_content, binding)

    visible_preflight = sealing.preflight_model_visible_release(
        release_dir=release,
        expected_release_sha256=binding.sealed_evaluation_release_sha256,
    )
    manifest_content = _read_regular_file(
        release / sealing.SEALED_RELEASE_MANIFEST,
        sealing.SEALED_RELEASE_MANIFEST,
    )
    if (
        hashlib.sha256(manifest_content).hexdigest()
        != binding.sealed_evaluation_release_sha256
    ):
        raise ValueError("sealed-release manifest commitment mismatch")
    manifest = _canonical_object(
        manifest_content,
        sealing.SEALED_RELEASE_MANIFEST,
    )
    model_visible = manifest.get("model_visible")
    if not isinstance(model_visible, Mapping):
        raise ValueError("sealed release has no model-visible binding")
    items_binding = model_visible.get("items")
    stores_binding = model_visible.get("stores")
    if not isinstance(items_binding, Mapping) or not isinstance(
        stores_binding,
        Mapping,
    ):
        raise ValueError(
            "sealed release model-visible artifact binding is invalid"
        )
    items_content = _read_bound_artifact(
        release,
        sealing.ITEMS_NAME,
        items_binding.get("sha256"),
    )
    items_sequence = _canonical_jsonl(
        items_content,
        name=sealing.ITEMS_NAME,
        parser=ItemRecord.from_dict,
        identity=lambda item: item.item_id,
    )
    items = {item.item_id: item for item in items_sequence}
    selected_ids_raw = model_visible.get("item_ids")
    if (
        not isinstance(selected_ids_raw, list)
        or any(not isinstance(item_id, str) for item_id in selected_ids_raw)
    ):
        raise ValueError("sealed release item registry is invalid")
    selected_ids = tuple(selected_ids_raw)
    if tuple(items) != selected_ids:
        raise ValueError("model-visible item registry is missing or reordered")
    stores_content = _read_bound_artifact(
        release,
        sealing.STORES_NAME,
        stores_binding.get("sha256"),
    )
    stores_sequence = _canonical_jsonl(
        stores_content,
        name=sealing.STORES_NAME,
        parser=StoreRecord.from_dict,
        identity=lambda record: record.store_id,
    )
    stores = {record.store_id: record for record in stores_sequence}
    _validate_model_visible_protocol(items, stores)
    if (
        visible_preflight.item_count != len(items)
        or visible_preflight.store_count != len(stores)
    ):
        raise ValueError("sealed release preflight count binding mismatch")
    final_preflight = sealing.preflight_model_visible_release(
        release_dir=release,
        expected_release_sha256=binding.sealed_evaluation_release_sha256,
    )
    if final_preflight != visible_preflight:
        raise ValueError("sealed release changed during evaluator preflight")
    if (
        _read_v3_input(
            run_root / "run.json",
            "run.json",
            max_bytes=_MAX_RUN_BINDING_BYTES,
        )
        != run_content
    ):
        raise ValueError("run.json changed during evaluator preflight")
    if (
        _read_v3_input(
            lock_path,
            "study-lock.json",
            max_bytes=_MAX_STUDY_LOCK_BYTES,
        )
        != lock_content
    ):
        raise ValueError("study-lock.json changed during evaluator preflight")
    return _PreparedEvaluationV3(
        run=run_root,
        release=release,
        binding=binding,
        run_content=run_content,
        lock_content=lock_content,
        snapshot_content=snapshot_content,
        release_manifest_content=manifest_content,
        release_manifest=MappingProxyType(dict(manifest)),
        selected_ids=selected_ids,
        items_content=items_content,
        items=MappingProxyType(items),
        stores_content=stores_content,
        stores=MappingProxyType(stores),
    )


def _prepare_evaluation(
    *,
    run: str | Path,
    sealed_release: str | Path,
    expected_study_lock_sha256: str,
) -> _PreparedEvaluation | _PreparedEvaluationV3:
    release = _directory(sealed_release, "sealed release")
    run_root = _directory(run, "run")
    run_content = _read_v3_input(
        run_root / "run.json",
        "run.json",
        max_bytes=_MAX_RUN_BINDING_BYTES,
    )
    parsed_binding = _parse_run_binding_content(run_content)
    if isinstance(parsed_binding, RunBindingV3):
        return _prepare_evaluation_v3(
            run_root=run_root,
            release=release,
            expected_study_lock_sha256=expected_study_lock_sha256,
            binding=parsed_binding,
            run_content=run_content,
        )
    lock = _load_study_lock(release, expected_study_lock_sha256)
    validity_content = _load_complete_readiness(release, lock)
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
    _validate_model_visible_protocol(items, stores)
    return _PreparedEvaluation(
        run=run_root,
        release=release,
        lock=lock,
        validity_content=validity_content,
        binding=binding,
        checkpoint=checkpoint,
        selected_ids=selected_ids,
        items_content=items_content,
        items=MappingProxyType(items),
        stores_content=stores_content,
        stores=MappingProxyType(stores),
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
    if isinstance(prepared, _PreparedEvaluationV3):
        return PreflightResult(
            study_lock_sha256=prepared.binding.study_lock_sha256,
            checkpoint_sha256=prepared.binding.snapshot_sha256,
            condition_id=prepared.binding.condition_id,
            seed=prepared.binding.seed,
            item_count=len(prepared.items),
            optimizer_step=prepared.binding.optimizer_step,
            output_id=prepared.binding.output_id,
            selected_provider=prepared.binding.selected_provider,
            snapshot_sha256=prepared.binding.snapshot_sha256,
        )
    return PreflightResult(
        study_lock_sha256=prepared.lock.sha256,
        checkpoint_sha256=prepared.binding.checkpoint_sha256,
        condition_id=prepared.binding.condition_id,
        seed=prepared.binding.seed,
        item_count=len(prepared.items),
    )


def _evaluate_prepared_v3(
    *,
    prepared: _PreparedEvaluationV3,
    output: Path,
    model_adapter: ModelAdapter | None,
    device: str,
    provider_qualified: bool,
    execution_identity_verifier,
    execution_probe,
) -> EvaluationResult:
    binding = prepared.binding
    if output.name != binding.output_id:
        raise ValueError("output directory name disagrees with output identity")
    if provider_qualified:
        if model_adapter is not None:
            raise ValueError(
                "fixture or injected adapters cannot produce "
                "provider-qualified output"
            )
        execution_identity_before = _authenticate_evaluator_execution(
            run=prepared.run,
            binding=binding,
            device=device,
            identity_verifier=execution_identity_verifier,
            execution_probe=execution_probe,
        )
    else:
        execution_identity_before = _nonproduction_execution_identity(
            binding=binding,
            device=device,
            model_adapter=model_adapter,
            execution_probe=execution_probe,
        )
    if model_adapter is None:
        model_adapter = RepositoryGPTAdapter.from_bound_run(
            prepared.run,
            binding,
            device,
        )
    if not callable(getattr(model_adapter, "generate", None)):
        raise TypeError("model_adapter must expose generate(item, store)")
    if provider_qualified and not isinstance(
        model_adapter,
        RepositoryGPTAdapter,
    ):
        raise ValueError(
            "provider-qualified output requires RepositoryGPTAdapter"
        )

    submissions: dict[str, Submission] = {}
    for item_id in prepared.selected_ids:
        item = prepared.items[item_id]
        visible_store = (
            None
            if item.memory_mode is MemoryMode.MEMORY_OFF
            else prepared.stores[item.store_id]
        )
        submission = model_adapter.generate(item, visible_store)
        if not isinstance(submission, Submission):
            raise ValueError("ModelAdapter.generate must return a Submission")
        if submission.item_id != item_id:
            if submission.item_id in submissions:
                raise ValueError("model adapter returned a duplicate submission")
            raise ValueError("model adapter returned a submission item mismatch")
        if item_id in submissions:
            raise ValueError("model adapter returned a duplicate submission")
        submissions[item_id] = submission
    if tuple(submissions) != prepared.selected_ids:
        raise ValueError("model adapter omitted a required item submission")

    if provider_qualified:
        execution_identity_after = _authenticate_evaluator_execution(
            run=prepared.run,
            binding=binding,
            device=device,
            identity_verifier=execution_identity_verifier,
            execution_probe=execution_probe,
        )
        if execution_identity_after != execution_identity_before:
            raise ValueError(
                "evaluator execution identity changed during evaluation"
            )
    else:
        execution_identity_after = _nonproduction_execution_identity(
            binding=binding,
            device=device,
            model_adapter=model_adapter,
            execution_probe=execution_probe,
        )
    model_snapshot_path = _relative_file(
        prepared.run,
        binding.snapshot_path,
        "snapshot_path",
    )
    current_snapshot_content = _read_v3_input(
        model_snapshot_path,
        "model snapshot",
        max_bytes=_MAX_MODEL_SNAPSHOT_BYTES,
    )
    if (
        current_snapshot_content != prepared.snapshot_content
        or hashlib.sha256(current_snapshot_content).hexdigest()
        != binding.snapshot_sha256
    ):
        raise ValueError("model snapshot changed after evaluator preflight")
    if (
        _read_v3_input(
            prepared.run / "run.json",
            "run.json",
            max_bytes=_MAX_RUN_BINDING_BYTES,
        )
        != prepared.run_content
    ):
        raise ValueError("run.json changed before scoring")
    lock_path = _relative_file(
        prepared.run,
        binding.study_lock_path,
        "study_lock_path",
    )
    if (
        _read_v3_input(
            lock_path,
            "study-lock.json",
            max_bytes=_MAX_STUDY_LOCK_BYTES,
        )
        != prepared.lock_content
    ):
        raise ValueError("study-lock.json changed before scoring")
    verified = sealing.verify_release(
        release_dir=prepared.release,
        expected_release_sha256=binding.sealed_evaluation_release_sha256,
    )
    if (
        verified.item_count != len(prepared.items)
        or verified.store_count != len(prepared.stores)
    ):
        raise ValueError("verified sealed-release counts changed")
    sealed_gold_binding = prepared.release_manifest.get("sealed_gold")
    if not isinstance(sealed_gold_binding, Mapping):
        raise ValueError("sealed release has no sealed-gold binding")
    gold_content = _read_bound_artifact(
        prepared.release,
        sealing.SEALED_GOLD_NAME,
        sealed_gold_binding.get("sha256"),
    )
    gold_sequence = _canonical_jsonl(
        gold_content,
        name=sealing.SEALED_GOLD_NAME,
        parser=SealedGoldRecord.from_dict,
        identity=lambda record: record.item_id,
    )
    gold = {record.item_id: record for record in gold_sequence}
    if tuple(gold) != prepared.selected_ids:
        raise ValueError(
            "sealed-gold registry disagrees with model-visible items"
        )

    outcomes_content, metrics_content = _score_and_summarize_v3(
        items=prepared.items,
        gold=gold,
        stores=prepared.stores,
        binding=binding,
        submissions=submissions,
    )
    artifacts = MappingProxyType(
        {
            "inference.json": _inference_bytes_v3(binding),
            "items.jsonl": prepared.items_content,
            "metrics.json": metrics_content,
            "outcomes.jsonl": outcomes_content,
            "run.json": prepared.run_content,
            "sealed-gold.jsonl": gold_content,
            "sealed-release.json": prepared.release_manifest_content,
            "stores.jsonl": prepared.stores_content,
            "study-lock.json": prepared.lock_content,
        }
    )
    report_hash = _publish_v3_evidence(
        output=output,
        binding=binding,
        artifacts=artifacts,
        execution_identity_before=execution_identity_before,
        execution_identity_after=execution_identity_after,
        production_qualified=provider_qualified,
    )
    return EvaluationResult(
        output_dir=output,
        study_lock_sha256=binding.study_lock_sha256,
        checkpoint_sha256=binding.snapshot_sha256,
        condition_id=binding.condition_id,
        seed=binding.seed,
        item_count=len(prepared.items),
        report_sha256=report_hash,
        optimizer_step=binding.optimizer_step,
        output_id=binding.output_id,
        selected_provider=binding.selected_provider,
        production_qualified=provider_qualified,
        snapshot_sha256=binding.snapshot_sha256,
    )


def evaluate(
    *,
    run: str | Path,
    sealed_release: str | Path,
    expected_study_lock_sha256: str,
    output_dir: str | Path,
    model_adapter: ModelAdapter | None = None,
    device: str = "cpu",
    provider_qualified: bool = False,
    execution_identity_verifier=verify_aws_instance_identity_pkcs7,
    execution_probe=None,
) -> EvaluationResult:
    """Evaluate one hash-bound checkpoint without exposing sealed gold."""

    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output directory already exists: {output}")
    prepared = _prepare_evaluation(
        run=run,
        sealed_release=sealed_release,
        expected_study_lock_sha256=expected_study_lock_sha256,
    )
    if isinstance(prepared, _PreparedEvaluationV3):
        return _evaluate_prepared_v3(
            prepared=prepared,
            output=output,
            model_adapter=model_adapter,
            device=device,
            provider_qualified=provider_qualified,
            execution_identity_verifier=execution_identity_verifier,
            execution_probe=execution_probe,
        )
    if provider_qualified:
        raise ValueError(
            "provider-qualified publication requires a v3 run binding"
        )
    release = prepared.release
    lock = prepared.lock
    binding = prepared.binding
    checkpoint = prepared.checkpoint
    selected_ids = prepared.selected_ids
    items_content = prepared.items_content
    items = prepared.items
    stores_content = prepared.stores_content
    stores = prepared.stores
    validity_content = prepared.validity_content
    if model_adapter is None:
        model_adapter = RepositoryGPTAdapter.from_bound_run(
            prepared.run,
            binding,
            device,
        )
    if not callable(getattr(model_adapter, "generate", None)):
        raise TypeError("model_adapter must expose generate(item, store)")

    submissions: dict[str, Submission] = {}
    for item_id in selected_ids:
        item = items[item_id]
        visible_store = (
            None if item.memory_mode is MemoryMode.MEMORY_OFF else stores[item.store_id]
        )
        submission = model_adapter.generate(item, visible_store)
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

    outcomes_content, metrics_content = _score_and_summarize(
        items=items,
        gold=gold,
        stores=stores,
        checkpoint=checkpoint,
        binding=binding,
        submissions=submissions,
    )
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
