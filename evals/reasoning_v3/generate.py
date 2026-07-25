"""Deterministic held-out item generation for the reasoning-v3 evaluator."""

from __future__ import annotations

import hashlib
import os
import stat
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.reasoning_expansion import (
    GeneratedRecord,
    ReasoningExpansionError,
    ReasoningGymGenerator,
    _MANIFEST_HEADER,
    _MANIFEST_MAGIC,
    _MANIFEST_RECORD,
    _generator_artifacts,
    _reasoning_gym_tree_commitment,
    _require_runtime_lock,
    load_expansion_recipe,
)
from corpusgen.reasoning_oracles import (
    ReasoningOracleError,
    ReasoningOracleRejection,
    canonical_reasoning_answer,
)
from evals.reasoning_v3.contracts import (
    EvaluationContract,
    EvaluationContractError,
    FamilyContract,
    _validate_corpus_receipt,
)


_HEX = frozenset("0123456789abcdef")
_MAX_MANIFEST_BYTES = 512 << 20
_SPECIAL_TOKEN_LITERALS = (
    "<|db_start|>",
    "<|db_retrieve|>",
    "<|db_end|>",
    "<|eot|>",
)


class EvaluationGenerationError(ValueError):
    """Evaluation item generation violated its prospective contract."""


@dataclass(frozen=True)
class FrozenEvaluationPaths:
    """Relocatable roots whose internal identities remain frozen by the contract."""

    repository_root: Path
    dataset_root: Path
    source_stage_root: Path

    def __post_init__(self) -> None:
        for field in ("repository_root", "dataset_root", "source_stage_root"):
            object.__setattr__(self, field, Path(getattr(self, field)))


@dataclass(frozen=True)
class ProvenanceCommitment:
    corpus_receipt_sha256: str
    source_stage_receipt_sha256: str
    source_tree_commitment_sha256: str
    record_manifest_sha256: str
    record_count: int
    generator_artifacts_sha256: str
    runtime_lock_sha256: str


@dataclass(frozen=True)
class ManifestReplay:
    overlap_keys: frozenset[tuple[str, int]]
    record_count: int
    sha256: str


@dataclass(frozen=True)
class _AuthenticatedRegistry:
    contract: EvaluationContract
    items: tuple[EvaluationItem, ...]
    provenance: ProvenanceCommitment


@dataclass(frozen=True)
class CandidateRow:
    task: str
    source_index: int
    question: str | None
    native_answer: str | None
    oracle_answer: str | None
    token_count: int
    prompt_token_count: int
    answer_token_count: int
    record_sha256: str
    rejection_reason: str | None = None


@dataclass(frozen=True)
class OracleReplayEvidence:
    record_sha256: str
    question_sha256: str
    native_answer_sha256: str
    independent_answer_sha256: str
    task_config_sha256: str


@dataclass(frozen=True)
class EvaluationItem:
    item_id: str
    task: str
    source_index: int
    prompt: str
    max_new_tokens: int
    scorer_id: str
    canonical_answer: str
    oracle_replay: OracleReplayEvidence


class EvaluationSource(Protocol):
    def generate(self, task: str, source_index: int) -> CandidateRow: ...


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _HEX
    )


def _answer_sha256(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def _item_id(task: str, source_index: int) -> str:
    return f"memorysplit-reasoning-v3-eval-v1/{task}/{source_index}"


class _ReasoningGymEvaluationSource:
    """Expose prompt/gold rows while delegating native generation and oracle logic."""

    def __init__(self, generator: ReasoningGymGenerator):
        self.generator = generator

    def generate(self, task: str, source_index: int) -> CandidateRow:
        try:
            generated = self.generator.generate(task, source_index)
        except (KeyError, ReasoningOracleError, RuntimeError, ValueError) as error:
            raise EvaluationGenerationError(
                f"reasoning generator failed for {task}[{source_index}]"
            ) from error
        if (
            not isinstance(generated, GeneratedRecord)
            or generated.task != task
            or generated.source_index != source_index
            or not _is_sha256(generated.record_sha256)
        ):
            raise EvaluationGenerationError(
                f"reasoning generator identity drifted for {task}[{source_index}]"
            )
        if generated.token_ids is None:
            return CandidateRow(
                task=task,
                source_index=source_index,
                question=None,
                native_answer=None,
                oracle_answer=None,
                token_count=generated.token_count,
                prompt_token_count=0,
                answer_token_count=0,
                record_sha256=generated.record_sha256,
                rejection_reason=generated.rejection_reason,
            )

        try:
            row = self.generator.datasets[task][source_index]
            task_spec = self.generator.task_specs[task]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise EvaluationGenerationError(
                f"reasoning row replay failed for {task}[{source_index}]"
            ) from error
        if not isinstance(row, Mapping):
            raise EvaluationGenerationError(
                f"reasoning row is not an object for {task}[{source_index}]"
            )
        metadata = row.get("metadata")
        question = row.get("question")
        native_answer = row.get("answer")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("source_dataset") != task
            or metadata.get("source_index") != source_index
            or not isinstance(question, str)
            or not question
            or not isinstance(native_answer, str)
            or not native_answer
        ):
            raise EvaluationGenerationError(
                f"reasoning row provenance/content drifted for {task}[{source_index}]"
            )
        try:
            oracle_answer = canonical_reasoning_answer(
                task,
                row,
                task_spec.config,
            )
        except ReasoningOracleRejection as error:
            raise EvaluationGenerationError(
                f"accepted native row was independently rejected for "
                f"{task}[{source_index}]"
            ) from error
        except ReasoningOracleError as error:
            raise EvaluationGenerationError(
                f"independent oracle failed for {task}[{source_index}]"
            ) from error
        commitment = {
            "answer": oracle_answer,
            "question": question,
            "source_index": source_index,
            "task": task,
        }
        if (
            native_answer != oracle_answer
            or hashlib.sha256(canonical_json_bytes(commitment)).hexdigest()
            != generated.record_sha256
        ):
            raise EvaluationGenerationError(
                f"native and independent oracle disagrees for {task}[{source_index}]"
            )
        prompt = (
            f"Reasoning task={task}\n"
            f"Question: {question}\n"
            "Answer:"
        )
        prompt_token_count = len(self.generator.tok.encode(prompt))
        answer_token_count = len(self.generator.tok.encode(f" {oracle_answer}"))
        if generated.token_count != prompt_token_count + answer_token_count + 1:
            raise EvaluationGenerationError(
                f"reasoning token replay differs for {task}[{source_index}]"
            )
        return CandidateRow(
            task=task,
            source_index=source_index,
            question=question,
            native_answer=native_answer,
            oracle_answer=oracle_answer,
            token_count=generated.token_count,
            prompt_token_count=prompt_token_count,
            answer_token_count=answer_token_count,
            record_sha256=generated.record_sha256,
        )


def _validate_training_keys(
    keys: frozenset[tuple[str, int]] | set[tuple[str, int]],
    contract: EvaluationContract,
) -> frozenset[tuple[str, int]]:
    if not isinstance(keys, (frozenset, set)):
        raise EvaluationGenerationError("training record keys must be a set")
    family_names = set(contract.family_names)
    for key in keys:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or key[0] not in family_names
            or isinstance(key[1], bool)
            or not isinstance(key[1], int)
            or key[1] < 0
            or key[1] >= (1 << 32)
        ):
            raise EvaluationGenerationError("training record key is invalid")
    return frozenset(keys)


def _accepted_item(
    contract: EvaluationContract,
    family: FamilyContract,
    candidate: CandidateRow,
) -> EvaluationItem:
    if (
        candidate.task != family.task
        or isinstance(candidate.source_index, bool)
        or not isinstance(candidate.source_index, int)
        or not family.index_start <= candidate.source_index < family.index_stop
        or not isinstance(candidate.question, str)
        or not candidate.question
        or not isinstance(candidate.native_answer, str)
        or not candidate.native_answer
        or not isinstance(candidate.oracle_answer, str)
        or not candidate.oracle_answer
        or isinstance(candidate.token_count, bool)
        or not isinstance(candidate.token_count, int)
        or candidate.token_count <= 0
        or candidate.token_count > contract.max_record_tokens
        or isinstance(candidate.prompt_token_count, bool)
        or not isinstance(candidate.prompt_token_count, int)
        or candidate.prompt_token_count <= 0
        or isinstance(candidate.answer_token_count, bool)
        or not isinstance(candidate.answer_token_count, int)
        or candidate.answer_token_count <= 0
        or not _is_sha256(candidate.record_sha256)
    ):
        raise EvaluationGenerationError(
            f"accepted candidate is malformed for "
            f"{family.task}[{candidate.source_index!r}]"
        )
    if (
        unicodedata.normalize("NFC", candidate.question) != candidate.question
        or unicodedata.normalize("NFC", candidate.native_answer)
        != candidate.native_answer
        or unicodedata.normalize("NFC", candidate.oracle_answer)
        != candidate.oracle_answer
    ):
        raise EvaluationGenerationError(
            f"candidate text is not NFC for {family.task}[{candidate.source_index}]"
        )
    if any(
        special in text
        for special in _SPECIAL_TOKEN_LITERALS
        for text in (
            candidate.question,
            candidate.native_answer,
            candidate.oracle_answer,
        )
    ):
        raise EvaluationGenerationError(
            f"candidate contains a literal special token for "
            f"{family.task}[{candidate.source_index}]"
        )
    if (
        candidate.native_answer != candidate.native_answer.strip()
        or candidate.oracle_answer != candidate.oracle_answer.strip()
    ):
        raise EvaluationGenerationError(
            f"candidate answer has outer whitespace for "
            f"{family.task}[{candidate.source_index}]"
        )
    if candidate.native_answer != candidate.oracle_answer:
        raise EvaluationGenerationError(
            f"native and independent oracle disagrees for "
            f"{family.task}[{candidate.source_index}]"
        )
    prompt = (
        f"Reasoning task={family.task}\n"
        f"Question: {candidate.question}\n"
        "Answer:"
    )
    if (
        candidate.answer_token_count + 1 > family.max_new_tokens
        or candidate.prompt_token_count + family.max_new_tokens
        > contract.max_record_tokens
    ):
        raise EvaluationGenerationError(
            f"frozen decoding limit does not fit {family.task}"
            f"[{candidate.source_index}]"
        )
    answer_digest = _answer_sha256(candidate.oracle_answer)
    return EvaluationItem(
        item_id=_item_id(family.task, candidate.source_index),
        task=family.task,
        source_index=candidate.source_index,
        prompt=prompt,
        max_new_tokens=family.max_new_tokens,
        scorer_id=contract.scorer_id,
        canonical_answer=candidate.oracle_answer,
        oracle_replay=OracleReplayEvidence(
            record_sha256=candidate.record_sha256,
            question_sha256=hashlib.sha256(
                candidate.question.encode("utf-8")
            ).hexdigest(),
            native_answer_sha256=_answer_sha256(candidate.native_answer),
            independent_answer_sha256=answer_digest,
            task_config_sha256=hashlib.sha256(
                canonical_json_bytes(dict(family.config))
            ).hexdigest(),
        ),
    )


def _build_evaluation_registry(
    contract: EvaluationContract,
    source: EvaluationSource,
    *,
    training_record_keys: frozenset[tuple[str, int]] | set[tuple[str, int]],
) -> tuple[EvaluationItem, ...]:
    """Build every family in frozen order from its disjoint reserved window."""

    training_keys = _validate_training_keys(training_record_keys, contract)
    items: list[EvaluationItem] = []
    for family in contract.families:
        accepted = 0
        for source_index in range(family.index_start, family.index_stop):
            try:
                candidate = source.generate(family.task, source_index)
            except EvaluationGenerationError:
                raise
            except Exception as error:
                raise EvaluationGenerationError(
                    f"candidate generation failed for "
                    f"{family.task}[{source_index}]"
                ) from error
            if (
                not isinstance(candidate, CandidateRow)
                or candidate.task != family.task
                or candidate.source_index != source_index
                or not _is_sha256(candidate.record_sha256)
            ):
                raise EvaluationGenerationError(
                    f"candidate identity drifted for {family.task}[{source_index}]"
                )
            if candidate.rejection_reason is not None:
                if candidate.rejection_reason not in contract.skip_reasons:
                    raise EvaluationGenerationError(
                        f"candidate rejection reason is not frozen: "
                        f"{candidate.rejection_reason!r}"
                    )
                if (
                    candidate.question is not None
                    or candidate.native_answer is not None
                    or candidate.oracle_answer is not None
                    or candidate.prompt_token_count != 0
                    or candidate.answer_token_count != 0
                    or (
                        candidate.rejection_reason == "independent_oracle"
                        and candidate.token_count != 0
                    )
                    or (
                        candidate.rejection_reason == "overlength"
                        and (
                            isinstance(candidate.token_count, bool)
                            or not isinstance(candidate.token_count, int)
                            or candidate.token_count <= contract.max_record_tokens
                        )
                    )
                ):
                    raise EvaluationGenerationError(
                        f"candidate rejection payload differs for "
                        f"{family.task}[{source_index}]"
                    )
                continue
            key = (family.task, source_index)
            if key in training_keys:
                raise EvaluationGenerationError(
                    f"evaluation item overlaps training: {family.task}[{source_index}]"
                )
            items.append(_accepted_item(contract, family, candidate))
            accepted += 1
            if accepted == contract.accepted_items_per_family:
                break
        if accepted != contract.accepted_items_per_family:
            raise EvaluationGenerationError(
                f"reserved window lacks accepted items for {family.task}: "
                f"{accepted}/{contract.accepted_items_per_family}"
            )
    result = tuple(items)
    _validate_registry(contract, result)
    return result


def _validate_registry(
    contract: EvaluationContract,
    items: Sequence[EvaluationItem],
) -> None:
    """Validate complete ordering, identity, closure, and oracle commitments."""

    if (
        not isinstance(items, Sequence)
        or isinstance(items, (str, bytes, bytearray))
        or len(items) != contract.total_items
    ):
        raise EvaluationGenerationError(
            f"evaluation registry item count differs: "
            f"{len(items) if isinstance(items, Sequence) else 'invalid'}"
        )
    seen_ids: set[str] = set()
    seen_keys: set[tuple[str, int]] = set()
    offset = 0
    for family in contract.families:
        previous_index = family.index_start - 1
        for item in items[
            offset : offset + contract.accepted_items_per_family
        ]:
            if (
                not isinstance(item, EvaluationItem)
                or item.task != family.task
                or isinstance(item.source_index, bool)
                or not isinstance(item.source_index, int)
                or not family.index_start <= item.source_index < family.index_stop
                or item.source_index <= previous_index
                or item.item_id != _item_id(item.task, item.source_index)
                or item.max_new_tokens != family.max_new_tokens
                or item.scorer_id != contract.scorer_id
                or not isinstance(item.prompt, str)
                or not item.prompt.startswith(
                    f"Reasoning task={family.task}\nQuestion: "
                )
                or not item.prompt.endswith("\nAnswer:")
                or not isinstance(item.canonical_answer, str)
                or not item.canonical_answer
                or item.canonical_answer != item.canonical_answer.strip()
                or unicodedata.normalize("NFC", item.canonical_answer)
                != item.canonical_answer
                or unicodedata.normalize("NFC", item.prompt) != item.prompt
                or any(
                    special in text
                    for special in _SPECIAL_TOKEN_LITERALS
                    for text in (item.prompt, item.canonical_answer)
                )
                or not isinstance(item.oracle_replay, OracleReplayEvidence)
            ):
                raise EvaluationGenerationError(
                    f"evaluation registry is not ordered/closed for {family.task}"
                )
            key = (item.task, item.source_index)
            if item.item_id in seen_ids or key in seen_keys:
                raise EvaluationGenerationError(
                    "evaluation registry contains duplicate items"
                )
            evidence = item.oracle_replay
            question = item.prompt.split("\nQuestion: ", 1)[1].removesuffix(
                "\nAnswer:"
            )
            expected_record_sha256 = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "answer": item.canonical_answer,
                        "question": question,
                        "source_index": item.source_index,
                        "task": item.task,
                    }
                )
            ).hexdigest()
            if (
                not question
                or evidence.question_sha256
                != hashlib.sha256(question.encode("utf-8")).hexdigest()
                or evidence.record_sha256 != expected_record_sha256
                or not all(
                    _is_sha256(value)
                    for value in (
                        evidence.record_sha256,
                        evidence.question_sha256,
                        evidence.native_answer_sha256,
                        evidence.independent_answer_sha256,
                        evidence.task_config_sha256,
                    )
                )
                or evidence.native_answer_sha256
                != evidence.independent_answer_sha256
                or evidence.independent_answer_sha256
                != _answer_sha256(item.canonical_answer)
                or evidence.task_config_sha256
                != hashlib.sha256(
                    canonical_json_bytes(dict(family.config))
                ).hexdigest()
            ):
                raise EvaluationGenerationError(
                    f"oracle evidence differs for {item.item_id}"
                )
            seen_ids.add(item.item_id)
            seen_keys.add(key)
            previous_index = item.source_index
        offset += contract.accepted_items_per_family


def _replay_training_record_manifest(
    path: Path | str,
    contract: EvaluationContract,
) -> ManifestReplay:
    """Replay every entry and retain all keys that touch reserved eval windows."""

    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise EvaluationGenerationError(
            "training record manifest is missing or unsafe"
        ) from error
    digest = hashlib.sha256()
    keys: set[tuple[str, int]] = set()
    last_indices = [-1] * len(contract.families)
    record_count = 0
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > _MAX_MANIFEST_BYTES
        ):
            raise EvaluationGenerationError(
                "training record manifest must be a safe regular file"
            )
        with os.fdopen(descriptor, "rb", buffering=8 << 20, closefd=False) as handle:
            header = handle.read(_MANIFEST_HEADER.size)
            digest.update(header)
            expected_header = _MANIFEST_HEADER.pack(
                _MANIFEST_MAGIC,
                bytes.fromhex(contract.recipe_sha256),
                len(contract.families),
                contract.max_record_tokens,
            )
            if header != expected_header:
                raise EvaluationGenerationError(
                    "training record manifest header differs"
                )
            while True:
                entry = handle.read(_MANIFEST_RECORD.size)
                if not entry:
                    break
                digest.update(entry)
                if len(entry) != _MANIFEST_RECORD.size:
                    raise EvaluationGenerationError(
                        "training record manifest has a partial entry"
                    )
                task_id, token_count, source_index = _MANIFEST_RECORD.unpack(entry)
                if (
                    task_id >= len(contract.families)
                    or token_count <= 0
                    or token_count > contract.max_record_tokens
                    or source_index <= last_indices[task_id]
                ):
                    raise EvaluationGenerationError(
                        "training record manifest entry/order is invalid"
                    )
                key = (contract.families[task_id].task, source_index)
                family = contract.families[task_id]
                if family.index_start <= source_index < family.index_stop:
                    keys.add(key)
                last_indices[task_id] = source_index
                record_count += 1
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise EvaluationGenerationError(
            "training record manifest changed during replay"
        )
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != contract.record_manifest_sha256:
        raise EvaluationGenerationError(
            "training record manifest SHA-256 differs"
        )
    if record_count == 0:
        raise EvaluationGenerationError("training record manifest is empty")
    return ManifestReplay(
        overlap_keys=frozenset(keys),
        record_count=record_count,
        sha256=actual_sha256,
    )


def _safe_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise EvaluationGenerationError(f"{label} is missing") from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise EvaluationGenerationError(f"{label} is unsafe")
    return path.resolve(strict=True)


def _authenticate_generation_inputs(
    paths: FrozenEvaluationPaths,
    contract: EvaluationContract,
) -> tuple[
    EvaluationContract,
    object,
    ManifestReplay,
    ProvenanceCommitment,
    Path,
]:
    """Authenticate every external production input before source admission."""

    try:
        repository_root = _safe_directory(paths.repository_root, "repository root")
        if contract.repository_root != repository_root:
            raise EvaluationGenerationError(
                "authorized contract repository root differs"
            )
        dataset_root = _safe_directory(paths.dataset_root, "production dataset root")
        receipt = _validate_corpus_receipt(
            dataset_root / contract.corpus_receipt_path,
            contract,
        )
        replay = _replay_training_record_manifest(
            dataset_root / contract.record_manifest_path,
            contract,
        )
        if replay.record_count != receipt["extension"]["record_count"]:
            raise EvaluationGenerationError(
                "training record manifest count differs from corpus receipt"
            )

        source_stage_root = _safe_directory(
            paths.source_stage_root,
            "frozen source stage",
        )
        recipe = load_expansion_recipe(repository_root / contract.recipe_path)
        if recipe.source_relative_path != contract.source_relative_path:
            raise EvaluationGenerationError(
                "Reasoning Gym source path differs from the frozen contract"
            )
        runtime_lock = _require_runtime_lock(recipe)
        source_tree_sha256 = _reasoning_gym_tree_commitment(
            source_stage_root,
            recipe,
        )
        if (
            receipt["source"]["reasoning_gym_tree_commitment_sha256"]
            != source_tree_sha256
            or receipt["source"]["source_stage_receipt_sha256"]
            != contract.source_stage_receipt_sha256
        ):
            raise EvaluationGenerationError(
                "Reasoning Gym source provenance differs from corpus receipt"
            )
        generator_artifacts = _generator_artifacts(recipe)
        if generator_artifacts != receipt["generator_artifacts"]:
            raise EvaluationGenerationError(
                "generator/oracle provenance differs from corpus receipt"
            )
    except EvaluationGenerationError:
        raise
    except (EvaluationContractError, ReasoningExpansionError, OSError) as error:
        raise EvaluationGenerationError(
            "production receipt/source/manifest authentication failed"
        ) from error

    provenance = ProvenanceCommitment(
        corpus_receipt_sha256=contract.corpus_receipt_sha256,
        source_stage_receipt_sha256=contract.source_stage_receipt_sha256,
        source_tree_commitment_sha256=source_tree_sha256,
        record_manifest_sha256=replay.sha256,
        record_count=replay.record_count,
        generator_artifacts_sha256=hashlib.sha256(
            canonical_json_bytes(generator_artifacts)
        ).hexdigest(),
        runtime_lock_sha256=hashlib.sha256(
            canonical_json_bytes(runtime_lock)
        ).hexdigest(),
    )
    source_root = source_stage_root / contract.source_relative_path
    return contract, recipe, replay, provenance, source_root


def _generate_authenticated_registry(
    paths: FrozenEvaluationPaths,
    contract: EvaluationContract,
) -> _AuthenticatedRegistry:
    """Generate only after frozen receipt, manifest, source, runtime, and code replay."""

    contract, recipe, replay, provenance, source_root = (
        _authenticate_generation_inputs(paths, contract)
    )
    generator = ReasoningGymGenerator(source_root, recipe)
    items = _build_evaluation_registry(
        contract,
        _ReasoningGymEvaluationSource(generator),
        training_record_keys=replay.overlap_keys,
    )
    return _AuthenticatedRegistry(
        contract=contract,
        items=items,
        provenance=provenance,
    )
