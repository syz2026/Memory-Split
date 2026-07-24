"""Build and verify an append-only, exact-answer reasoning corpus extension.

The frozen MemorySplit v2 corpus is an immutable component.  This module adds a
new objective-reasoning suffix and publishes a composite receipt; it never
rewrites, reschedules, or aliases the v2 bytes.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import stat
import sys
import tempfile
import types
from array import array
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from corpusgen.parallel import atomic_rename_noreplace
from corpusgen.parallel.canonical import canonical_json_bytes
from train.tokenizer import get_tok

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE_PATH = ROOT / "configs" / "reasoning-dataset-v3.json"
DEFAULT_SOURCE_STAGE = ROOT / "corpus-build" / "memorysplit-v2-frozen-upstream-sources"
DEFAULT_BASE_CORPUS = ROOT / "corpus-build" / "memorysplit-parallel-corpus-v2"
DEFAULT_OUTPUT = ROOT / "corpus-build" / "memorysplit-reasoning-corpus-v3"

FORMAT = "memorysplit-reasoning-composite-v3"
EXTENSION_FORMAT = "memorysplit-reasoning-extension-v1"
SCHEMA_VERSION = 1
FINISH_WINDOW = 1 << 18
MAX_FINISH_CANDIDATES = 32_768
PROBE_INDICES = (0, 1, 17, 127)
_SEED_MULTIPLIER = 0x9E3779B1
_HEX = frozenset("0123456789abcdef")
_EXTENSION_PATHS = {
    "packed_targets": "packed/targets.bin",
    "shared_target_weights": "sidecars/shared_target_weights.bin",
}


class ReasoningExpansionError(RuntimeError):
    """The successor corpus cannot be built without violating its contract."""


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    module: str
    config: Mapping[str, Any]
    weight: int


@dataclass(frozen=True)
class ExpansionRecipe:
    path: Path
    sha256: str
    contract_id: str
    base_receipt_sha256: str
    base_ordered_sha256: str
    base_packed_sha256: str
    base_tokens: int
    extension_tokens: int
    targets_per_update: int
    extension_updates: int
    composite_tokens: int
    composite_updates: int
    max_record_tokens: int
    source_stage_receipt_sha256: str
    source_relative_path: str
    reasoning_gym_version: str
    tasks: tuple[TaskSpec, ...]


@dataclass(frozen=True)
class GeneratedRecord:
    task: str
    source_index: int
    token_ids: tuple[int, ...] | None
    token_count: int
    record_sha256: str


class RecordGenerator(Protocol):
    def generate(self, task: str, index: int) -> GeneratedRecord: ...


@dataclass
class _TaskStats:
    emitted_records: int = 0
    emitted_tokens: int = 0
    examined_records: int = 0
    overlength_rejections: int = 0


@dataclass(frozen=True)
class _Candidate:
    offset: int
    record: GeneratedRecord


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ReasoningExpansionError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReasoningExpansionError(f"{label} must be a positive integer")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ReasoningExpansionError(f"JSON repeats key: {key}")
        value[key] = item
    return value


def _read_json(path: Path, label: str) -> tuple[Any, bytes]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ReasoningExpansionError(f"{label} is missing") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ReasoningExpansionError(f"{label} must be a singly linked regular file")
    data = path.read_bytes()
    try:
        return (
            json.loads(
                data,
                object_pairs_hook=_unique_object,
                parse_constant=lambda item: (_ for _ in ()).throw(
                    ReasoningExpansionError(f"{label} contains non-finite JSON: {item}")
                ),
            ),
            data,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReasoningExpansionError(f"{label} is not valid UTF-8 JSON") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    _stream_regular(path, label=str(path), consumers=(digest.update,))
    return digest.hexdigest()


def load_expansion_recipe(
    path: Path | str = DEFAULT_RECIPE_PATH,
) -> ExpansionRecipe:
    recipe_path = Path(path)
    raw, data = _read_json(recipe_path, "reasoning expansion recipe")
    if not isinstance(raw, Mapping) or set(raw) != {
        "base_corpus",
        "composite",
        "contract_id",
        "extension",
        "reasoning_gym",
        "schema_version",
        "scientific_scope",
    }:
        raise ReasoningExpansionError("reasoning expansion recipe fields differ")
    if raw["schema_version"] != 1:
        raise ReasoningExpansionError("reasoning expansion recipe version differs")
    scope = raw["scientific_scope"]
    if scope != {
        "may_replace_frozen_v2_n10_dataset": False,
        "status": "successor_exploratory_unpreregistered",
        "target_weight_policy": (
            "all_extension_reasoning_targets_are_internal_in_both_arms"
        ),
    }:
        raise ReasoningExpansionError("reasoning expansion scientific scope differs")

    base = raw["base_corpus"]
    extension = raw["extension"]
    composite = raw["composite"]
    source = raw["reasoning_gym"]
    if not all(
        isinstance(item, Mapping) for item in (base, extension, composite, source)
    ):
        raise ReasoningExpansionError("reasoning expansion sections must be objects")
    if base.get("contract_id") != "memorysplit-parallel-corpus-v2":
        raise ReasoningExpansionError("reasoning expansion base contract differs")
    if extension.get("format") != EXTENSION_FORMAT:
        raise ReasoningExpansionError("reasoning extension format differs")
    if composite.get("ordering") != ("frozen-v2-prefix-then-reasoning-extension"):
        raise ReasoningExpansionError("reasoning composite ordering differs")
    if source.get("version") != "0.1.19":
        raise ReasoningExpansionError("Reasoning Gym version differs")

    base_tokens = _positive_int(base.get("raw_target_tokens"), "base tokens")
    extension_tokens = _positive_int(
        extension.get("raw_target_tokens"),
        "extension tokens",
    )
    targets_per_update = _positive_int(
        extension.get("targets_per_update"),
        "targets per update",
    )
    extension_updates = _positive_int(
        extension.get("terminal_updates"),
        "extension updates",
    )
    composite_tokens = _positive_int(
        composite.get("raw_target_tokens"),
        "composite tokens",
    )
    composite_updates = _positive_int(
        composite.get("terminal_updates"),
        "composite updates",
    )
    if (
        extension_tokens != targets_per_update * extension_updates
        or composite_tokens != base_tokens + extension_tokens
        or composite_updates * targets_per_update != composite_tokens
    ):
        raise ReasoningExpansionError(
            "reasoning expansion token/update geometry is not integral"
        )

    raw_tasks = extension.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) < 2:
        raise ReasoningExpansionError(
            "reasoning extension needs multiple task families"
        )
    tasks: list[TaskSpec] = []
    for index, item in enumerate(raw_tasks):
        if not isinstance(item, Mapping) or set(item) != {
            "config",
            "dataset",
            "module",
            "weight",
        }:
            raise ReasoningExpansionError(f"reasoning task {index} fields differ")
        dataset = item["dataset"]
        module = item["module"]
        config = item["config"]
        if (
            not isinstance(dataset, str)
            or not dataset
            or not isinstance(module, str)
            or not module
            or not isinstance(config, Mapping)
            or any(key in config for key in ("seed", "size"))
        ):
            raise ReasoningExpansionError(
                f"reasoning task {index} identity/config is invalid"
            )
        tasks.append(
            TaskSpec(
                dataset=dataset,
                module=module,
                config=dict(config),
                weight=_positive_int(item["weight"], f"task {index} weight"),
            )
        )
    names = [task.dataset for task in tasks]
    if len(names) != len(set(names)):
        raise ReasoningExpansionError("reasoning task ids must be unique")

    source_relative = source.get("staged_relative_path")
    if (
        not isinstance(source_relative, str)
        or PurePosixPath(source_relative).is_absolute()
        or ".." in PurePosixPath(source_relative).parts
    ):
        raise ReasoningExpansionError("Reasoning Gym staged path is unsafe")
    return ExpansionRecipe(
        path=recipe_path.resolve(strict=True),
        sha256=hashlib.sha256(data).hexdigest(),
        contract_id=str(raw["contract_id"]),
        base_receipt_sha256=_require_sha256(
            base.get("receipt_sha256"),
            "base receipt",
        ),
        base_ordered_sha256=_require_sha256(
            base.get("ordered_stream_sha256"),
            "base ordered stream",
        ),
        base_packed_sha256=_require_sha256(
            base.get("packed_stream_sha256"),
            "base packed stream",
        ),
        base_tokens=base_tokens,
        extension_tokens=extension_tokens,
        targets_per_update=targets_per_update,
        extension_updates=extension_updates,
        composite_tokens=composite_tokens,
        composite_updates=composite_updates,
        max_record_tokens=_positive_int(
            extension.get("max_record_tokens"),
            "maximum record tokens",
        ),
        source_stage_receipt_sha256=_require_sha256(
            source.get("source_stage_receipt_sha256"),
            "source-stage receipt",
        ),
        source_relative_path=source_relative,
        reasoning_gym_version=str(source["version"]),
        tasks=tuple(tasks),
    )


def _task_quotas(recipe: ExpansionRecipe) -> dict[str, int]:
    total_weight = sum(task.weight for task in recipe.tasks)
    ideals = [
        (
            task.dataset,
            Fraction(recipe.extension_tokens * task.weight, total_weight),
        )
        for task in recipe.tasks
    ]
    quotas = {name: int(ideal) for name, ideal in ideals}
    remaining = recipe.extension_tokens - sum(quotas.values())
    order = sorted(
        range(len(ideals)),
        key=lambda index: (
            -(ideals[index][1] - int(ideals[index][1])),
            index,
        ),
    )
    for index in order[:remaining]:
        quotas[ideals[index][0]] += 1
    return quotas


def _reasoning_gym_tree_commitment(
    source_stage: Path,
    recipe: ExpansionRecipe,
) -> str:
    receipt_path = source_stage / "source-stage-receipt.json"
    if sha256_file(receipt_path) != recipe.source_stage_receipt_sha256:
        raise ReasoningExpansionError("source-stage receipt SHA-256 mismatch")
    raw, _data = _read_json(receipt_path, "source-stage receipt")
    files = raw.get("files") if isinstance(raw, Mapping) else None
    if not isinstance(files, list):
        raise ReasoningExpansionError("source-stage receipt has no file inventory")
    prefix = f"{recipe.source_relative_path}/"
    selected = []
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {
            "bytes",
            "path",
            "sha256",
        }:
            raise ReasoningExpansionError("source-stage receipt file inventory differs")
        if str(item["path"]).startswith(prefix):
            selected.append(dict(item))
    if not selected:
        raise ReasoningExpansionError(
            "source-stage receipt does not bind Reasoning Gym"
        )
    selected.sort(key=lambda item: item["path"])
    for item in selected:
        relative = item["path"]
        if (
            not isinstance(relative, str)
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
        ):
            raise ReasoningExpansionError(
                "Reasoning Gym source inventory contains an unsafe path"
            )
        size, digest = _stream_regular(
            source_stage / relative,
            label=f"Reasoning Gym source {relative}",
        )
        if size != _positive_int(
            item["bytes"], "Reasoning Gym source bytes"
        ) or digest != _require_sha256(
            item["sha256"],
            "Reasoning Gym source digest",
        ):
            raise ReasoningExpansionError(
                f"Reasoning Gym source differs from its stage receipt: {relative}"
            )
    return hashlib.sha256(canonical_json_bytes(selected)).hexdigest()


def _install_reasoning_gym(
    source_root: Path,
    tasks: Sequence[TaskSpec],
):
    package_root = source_root.resolve(strict=True)
    marker = str(package_root)
    existing = sys.modules.get("reasoning_gym")
    if existing is not None:
        if getattr(existing, "__memorysplit_source__", None) != marker:
            raise ReasoningExpansionError(
                "another Reasoning Gym source is already imported"
            )
    else:
        package = types.ModuleType("reasoning_gym")
        package.__path__ = [str(package_root / "reasoning_gym")]
        package.__package__ = "reasoning_gym"
        package.__memorysplit_source__ = marker
        sys.modules["reasoning_gym"] = package

        coaching = types.ModuleType("reasoning_gym.coaching")
        coaching.__path__ = [str(package_root / "reasoning_gym" / "coaching")]
        coaching.__package__ = "reasoning_gym.coaching"
        sys.modules[coaching.__package__] = coaching
        attributes = importlib.import_module("reasoning_gym.coaching.attributes")
        base_curriculum = importlib.import_module(
            "reasoning_gym.coaching.base_curriculum"
        )
        for name in (
            "AttributeDefinition",
            "RangeAttributeDefinition",
            "ScalarAttributeDefinition",
        ):
            setattr(coaching, name, getattr(attributes, name))
        coaching.BaseCurriculum = base_curriculum.BaseCurriculum

    categories = sorted({task.module.split(".", 1)[0] for task in tasks})
    for category in categories:
        qualified = f"reasoning_gym.{category}"
        if qualified not in sys.modules:
            module = types.ModuleType(qualified)
            module.__path__ = [str(package_root / "reasoning_gym" / category)]
            module.__package__ = qualified
            sys.modules[qualified] = module
    factory = importlib.import_module("reasoning_gym.factory")
    for task in tasks:
        importlib.import_module(f"reasoning_gym.{task.module}")
        if task.dataset not in factory.DATASETS:
            raise ReasoningExpansionError(
                f"Reasoning Gym task is not registered: {task.dataset}"
            )
    return factory


def _task_seed(task: str) -> int:
    return int.from_bytes(hashlib.sha256(task.encode()).digest()[:4], "big")


class ReasoningGymGenerator:
    def __init__(self, source_root: Path, recipe: ExpansionRecipe):
        self.recipe = recipe
        self.tok = get_tok()
        factory = _install_reasoning_gym(source_root, recipe.tasks)
        self.datasets = {}
        for task in recipe.tasks:
            self.datasets[task.dataset] = factory.create_dataset(
                task.dataset,
                **dict(task.config),
                seed=_task_seed(task.dataset),
                size=(1 << 31) - 1,
            )

    def generate(self, task: str, index: int) -> GeneratedRecord:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= (1 << 31)
        ):
            raise ReasoningExpansionError(
                "reasoning task index exceeds its collision-free domain"
            )
        dataset = self.datasets[task]
        row = dataset[index]
        if not isinstance(row, Mapping):
            raise ReasoningExpansionError(f"Reasoning Gym {task} returned a non-object")
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not question or answer is None:
            raise ReasoningExpansionError(
                f"Reasoning Gym {task} returned no exact question/answer"
            )
        score = dataset.score_answer(str(answer), row)
        if score != 1.0:
            raise ReasoningExpansionError(
                f"Reasoning Gym {task} rejected its own oracle: {score!r}"
            )
        answer_text = json.dumps(
            answer,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        text = f"Reasoning task={task}\nQuestion: {question}\nAnswer: {answer_text}"
        token_ids = (*self.tok.encode(text), self.tok.EOT)
        commitment = {
            "answer": answer,
            "question": question,
            "source_index": index,
            "task": task,
        }
        digest = hashlib.sha256(canonical_json_bytes(commitment)).hexdigest()
        if len(token_ids) > self.recipe.max_record_tokens:
            return GeneratedRecord(
                task=task,
                source_index=index,
                token_ids=None,
                token_count=len(token_ids),
                record_sha256=digest,
            )
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            or token >= (1 << 16)
            for token in token_ids
        ):
            raise ReasoningExpansionError(
                f"Reasoning Gym {task} emitted an invalid token id"
            )
        return GeneratedRecord(
            task=task,
            source_index=index,
            token_ids=tuple(token_ids),
            token_count=len(token_ids),
            record_sha256=digest,
        )


def _probe_generator(
    generator: RecordGenerator,
    recipe: ExpansionRecipe,
) -> dict[str, list[dict[str, Any]]]:
    expected: dict[tuple[str, int], dict[str, Any]] = {}
    for tasks, indices in (
        (recipe.tasks, PROBE_INDICES),
        (tuple(reversed(recipe.tasks)), tuple(reversed(PROBE_INDICES))),
    ):
        for task in tasks:
            for index in indices:
                record = generator.generate(task.dataset, index)
                value = {
                    "accepted": record.token_ids is not None,
                    "record_sha256": record.record_sha256,
                    "source_index": index,
                    "token_count": record.token_count,
                }
                key = (task.dataset, index)
                prior = expected.setdefault(key, value)
                if prior != value:
                    raise ReasoningExpansionError(
                        f"Reasoning Gym task is order-dependent: {task.dataset}"
                    )
    return {
        task.dataset: [expected[(task.dataset, index)] for index in PROBE_INDICES]
        for task in recipe.tasks
    }


class _ExactSubsetAccumulator:
    def __init__(self, target: int) -> None:
        if target < 0:
            raise ValueError("subset target must be non-negative")
        self.target = target
        self.reachable = 1
        self.limit_mask = (1 << (target + 1)) - 1
        self.predecessor_sum = array("i", [-1]) * (target + 1)
        self.predecessor_index = array("i", [-1]) * (target + 1)
        self.count = 0

    def add(self, length: int) -> tuple[int, ...] | None:
        index = self.count
        self.count += 1
        if self.target == 0:
            return ()
        if length <= 0 or length > self.target:
            return None
        new = ((self.reachable << length) & self.limit_mask) & ~self.reachable
        bits = new
        while bits:
            bit = bits & -bits
            total = bit.bit_length() - 1
            self.predecessor_sum[total] = total - length
            self.predecessor_index[total] = index
            bits ^= bit
        self.reachable |= new
        if not ((self.reachable >> self.target) & 1):
            return None
        selected = []
        total = self.target
        while total:
            selected.append(self.predecessor_index[total])
            total = self.predecessor_sum[total]
        return tuple(reversed(selected))


def _u16_bytes(tokens: Sequence[int]) -> bytes:
    values = array("H", tokens)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


class _TokenWriter:
    def __init__(self, path: Path):
        self.handle = path.open("xb")
        self.digest = hashlib.sha256()
        self.records_digest = hashlib.sha256()
        self.tokens = 0
        self.records = 0
        self.buffer = bytearray()

    def add(self, record: GeneratedRecord, payload: bytes | None = None) -> None:
        if record.token_ids is None:
            raise ValueError("cannot write a rejected record")
        data = _u16_bytes(record.token_ids) if payload is None else payload
        if len(data) != record.token_count * 2:
            raise ReasoningExpansionError("record byte/token count differs")
        self.buffer.extend(data)
        self.digest.update(data)
        self.records_digest.update(
            canonical_json_bytes(
                {
                    "record_sha256": record.record_sha256,
                    "source_index": record.source_index,
                    "task": record.task,
                    "token_count": record.token_count,
                }
            )
        )
        self.tokens += record.token_count
        self.records += 1
        if len(self.buffer) >= (8 << 20):
            self._flush()

    def _flush(self) -> None:
        if self.buffer:
            self.handle.write(self.buffer)
            self.buffer.clear()

    def close(self) -> None:
        self._flush()
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


def _choose_task(
    recipe: ExpansionRecipe,
    quotas: Mapping[str, int],
    task_tokens: Mapping[str, int],
    written: int,
) -> str:
    return max(
        (task.dataset for task in recipe.tasks),
        key=lambda task: (
            quotas[task] * written - task_tokens[task] * recipe.extension_tokens
        ),
    )


def _next_accepted(
    generator: RecordGenerator,
    task: str,
    next_indices: dict[str, int],
    stats: dict[str, _TaskStats],
) -> GeneratedRecord:
    while True:
        index = next_indices[task]
        next_indices[task] += 1
        stats[task].examined_records += 1
        record = generator.generate(task, index)
        if record.task != task or record.source_index != index:
            raise ReasoningExpansionError(
                f"reasoning generator identity drifted for {task}"
            )
        if record.token_ids is None:
            stats[task].overlength_rejections += 1
            continue
        if record.token_count != len(record.token_ids) or record.token_count <= 0:
            raise ReasoningExpansionError(
                f"reasoning generator token count drifted for {task}"
            )
        return record


def _compile_extension(
    recipe: ExpansionRecipe,
    generator: RecordGenerator,
    destination: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
    finish_window: int = FINISH_WINDOW,
    max_finish_candidates: int = MAX_FINISH_CANDIDATES,
) -> dict[str, Any]:
    if finish_window <= 0 or finish_window >= recipe.extension_tokens:
        raise ValueError("finish window must be inside the extension horizon")
    quotas = _task_quotas(recipe)
    stats = {task.dataset: _TaskStats() for task in recipe.tasks}
    next_indices = {task.dataset: 0 for task in recipe.tasks}
    task_tokens = {task.dataset: 0 for task in recipe.tasks}
    writer = _TokenWriter(destination)
    next_progress = 64 << 20
    try:
        while recipe.extension_tokens - writer.tokens > finish_window:
            task = _choose_task(recipe, quotas, task_tokens, writer.tokens)
            record = _next_accepted(generator, task, next_indices, stats)
            if record.token_count > recipe.extension_tokens - writer.tokens:
                raise ReasoningExpansionError(
                    "non-final reasoning record exceeds the remaining horizon"
                )
            writer.add(record)
            stats[task].emitted_records += 1
            stats[task].emitted_tokens += record.token_count
            task_tokens[task] += record.token_count
            if progress is not None and writer.tokens >= next_progress:
                progress(writer.tokens, recipe.extension_tokens)
                next_progress += 64 << 20

        target = recipe.extension_tokens - writer.tokens
        subset = _ExactSubsetAccumulator(target)
        candidates: list[_Candidate] = []
        selected: tuple[int, ...] | None = () if target == 0 else None
        virtual_written = writer.tokens
        virtual_task_tokens = dict(task_tokens)
        with tempfile.TemporaryFile(
            mode="w+b",
            prefix=".reasoning-finish-",
            dir=destination.parent,
        ) as spool:
            for _ in range(max_finish_candidates):
                if selected is not None:
                    break
                task = _choose_task(
                    recipe,
                    quotas,
                    virtual_task_tokens,
                    virtual_written,
                )
                record = _next_accepted(generator, task, next_indices, stats)
                virtual_written += record.token_count
                virtual_task_tokens[task] += record.token_count
                if record.token_count > target:
                    continue
                assert record.token_ids is not None
                offset = spool.tell()
                spool.write(_u16_bytes(record.token_ids))
                candidates.append(_Candidate(offset=offset, record=record))
                selected = subset.add(record.token_count)
            if selected is None:
                raise ReasoningExpansionError(
                    f"could not exactly fill the final {target} tokens from "
                    f"{len(candidates)} fresh reasoning records"
                )
            for index in selected:
                candidate = candidates[index]
                record = candidate.record
                spool.seek(candidate.offset)
                payload = spool.read(record.token_count * 2)
                if len(payload) != record.token_count * 2:
                    raise ReasoningExpansionError(
                        "reasoning finish spool was truncated"
                    )
                writer.add(record, payload)
                stats[record.task].emitted_records += 1
                stats[record.task].emitted_tokens += record.token_count
                task_tokens[record.task] += record.token_count
        if writer.tokens != recipe.extension_tokens:
            raise ReasoningExpansionError(
                "reasoning extension did not meet its exact token horizon"
            )
    finally:
        writer.close()
    if progress is not None:
        progress(writer.tokens, recipe.extension_tokens)
    return {
        "packed_stream_sha256": writer.digest.hexdigest(),
        "record_count": writer.records,
        "records_commitment_sha256": writer.records_digest.hexdigest(),
        "task_stats": [
            {
                "dataset": task.dataset,
                "emitted_records": stats[task.dataset].emitted_records,
                "emitted_tokens": stats[task.dataset].emitted_tokens,
                "examined_records": stats[task.dataset].examined_records,
                "overlength_rejections": (stats[task.dataset].overlength_rejections),
                "source_cursor": next_indices[task.dataset],
                "target_quota": quotas[task.dataset],
            }
            for task in recipe.tasks
        ],
    }


def _write_ones(path: Path, count: int) -> str:
    digest = hashlib.sha256()
    block = b"\x01" * (8 << 20)
    remaining = count
    with path.open("xb") as handle:
        while remaining:
            chunk = block[: min(len(block), remaining)]
            handle.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    return digest.hexdigest()


def _stream_regular(
    path: Path,
    *,
    label: str,
    consumers: Sequence[Callable[[bytes], None]] = (),
) -> tuple[int, str]:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise ReasoningExpansionError(
            f"{label} is missing, symlinked, or unsafe"
        ) from error
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ReasoningExpansionError(
                f"{label} must be a singly linked regular file"
            )
        while True:
            chunk = os.read(descriptor, 8 << 20)
            if not chunk:
                break
            digest.update(chunk)
            for consumer in consumers:
                consumer(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or size != after.st_size:
            raise ReasoningExpansionError(f"{label} changed while being read")
    finally:
        os.close(descriptor)
    return size, digest.hexdigest()


def _artifact_records(value: object, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ReasoningExpansionError(f"{label} artifacts are missing")
    records = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "bytes",
            "path",
            "sha256",
        }:
            raise ReasoningExpansionError(f"{label} artifact fields differ")
        records.append(item)
    return records


def _stream_base_group(
    base_root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
    expected_stream_sha256: str,
    composite: hashlib._Hash,
) -> None:
    stream = hashlib.sha256()
    for item in records:
        relative = item["path"]
        if (
            not isinstance(relative, str)
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
        ):
            raise ReasoningExpansionError(f"{label} artifact path is unsafe")
        expected_bytes = _positive_int(
            item["bytes"],
            f"{label} artifact bytes",
        )
        expected_sha = _require_sha256(
            item["sha256"],
            f"{label} artifact digest",
        )
        size, digest = _stream_regular(
            base_root / relative,
            label=f"{label} artifact {relative}",
            consumers=(stream.update, composite.update),
        )
        if size != expected_bytes or digest != expected_sha:
            raise ReasoningExpansionError(
                f"{label} artifact differs from the base receipt"
            )
    if stream.hexdigest() != expected_stream_sha256:
        raise ReasoningExpansionError(f"{label} stream differs from its receipt")


def _base_receipt(
    base_root: Path,
    recipe: ExpansionRecipe,
) -> Mapping[str, Any]:
    raw, data = _read_json(base_root / "receipt.json", "base corpus receipt")
    if hashlib.sha256(data).hexdigest() != recipe.base_receipt_sha256:
        raise ReasoningExpansionError("base corpus receipt SHA-256 mismatch")
    if (
        not isinstance(raw, Mapping)
        or raw.get("format") != "memorysplit-parallel-corpus-v2"
        or raw.get("logical_tokens") != recipe.base_tokens
        or raw.get("packed_tokens") != recipe.base_tokens
        or raw.get("padding_tokens") != 0
        or raw.get("ordered_stream_sha256") != recipe.base_ordered_sha256
        or raw.get("packed_stream_sha256") != recipe.base_packed_sha256
    ):
        raise ReasoningExpansionError("base corpus identity/geometry differs")
    return raw


def _composite_stream_hashes(
    base_root: Path,
    base: Mapping[str, Any],
    extension_root: Path,
    extension_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    groups = {
        "packed_targets": (
            [
                item
                for item in _artifact_records(
                    base.get("artifacts"),
                    "base primary",
                )
                if str(item["path"]).startswith("shards/")
            ],
            _require_sha256(
                base.get("packed_stream_sha256"),
                "base packed stream",
            ),
        )
    }
    sidecar_sets = base.get("sidecar_sets")
    if not isinstance(sidecar_sets, list):
        raise ReasoningExpansionError("base sidecar sets are missing")
    for name in ("dense_target_weights", "split90_target_weights"):
        matches = [
            item
            for item in sidecar_sets
            if isinstance(item, Mapping) and item.get("name") == name
        ]
        if len(matches) != 1:
            raise ReasoningExpansionError(f"base {name} sidecar set differs")
        sidecar = matches[0]
        groups[name] = (
            _artifact_records(sidecar.get("artifacts"), f"base {name}"),
            _require_sha256(
                sidecar.get("stream_sha256"),
                f"base {name} stream",
            ),
        )

    result = {}
    for name, (records, expected_stream) in groups.items():
        composite = hashlib.sha256()
        _stream_base_group(
            base_root,
            records,
            label=f"base {name}",
            expected_stream_sha256=expected_stream,
            composite=composite,
        )
        extension_name = (
            "packed_targets" if name == "packed_targets" else "shared_target_weights"
        )
        artifact = extension_artifacts[extension_name]
        extension_path = extension_root / artifact["path"]
        size, digest = _stream_regular(
            extension_path,
            label=f"extension {extension_name}",
            consumers=(composite.update,),
        )
        if size != artifact["bytes"] or digest != artifact["sha256"]:
            raise ReasoningExpansionError(
                f"extension {extension_name} differs from its receipt"
            )
        result[name] = composite.hexdigest()
    return result


def _generator_artifacts(recipe: ExpansionRecipe) -> dict[str, str]:
    paths = {
        "corpusgen/reasoning_expansion.py": Path(__file__).resolve(),
        "train/tokenizer.py": ROOT / "train" / "tokenizer.py",
    }
    return {name: sha256_file(path) for name, path in sorted(paths.items())} | {
        "configs/reasoning-dataset-v3.json": recipe.sha256
    }


def _extension_artifacts(
    root: Path,
    *,
    packed_sha256: str,
    shared_sha256: str,
    extension_tokens: int,
) -> dict[str, dict[str, Any]]:
    return {
        "packed_targets": {
            "bytes": extension_tokens * 2,
            "path": _EXTENSION_PATHS["packed_targets"],
            "sha256": packed_sha256,
        },
        "shared_target_weights": {
            "bytes": extension_tokens,
            "path": _EXTENSION_PATHS["shared_target_weights"],
            "sha256": shared_sha256,
        },
    }


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> str:
    payload = canonical_json_bytes(receipt)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _publish_no_replace(staging: Path, destination: Path) -> None:
    parent = destination.parent
    descriptor = os.open(
        parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        atomic_rename_noreplace(
            descriptor,
            staging.name,
            descriptor,
            destination.name,
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_reasoning_corpus(
    *,
    recipe_path: Path | str = DEFAULT_RECIPE_PATH,
    source_stage: Path | str = DEFAULT_SOURCE_STAGE,
    base_corpus: Path | str = DEFAULT_BASE_CORPUS,
    destination: Path | str = DEFAULT_OUTPUT,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    recipe = load_expansion_recipe(recipe_path)
    source_stage_root = Path(source_stage)
    base_root = Path(base_corpus)
    output = Path(destination)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"reasoning corpus destination exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = recipe.extension_tokens * 3
    if shutil.disk_usage(output.parent).free < expected_bytes + (1 << 30):
        raise ReasoningExpansionError(
            "insufficient disk for reasoning extension and safety margin"
        )
    tree_commitment = _reasoning_gym_tree_commitment(
        source_stage_root,
        recipe,
    )
    source_root = source_stage_root / recipe.source_relative_path
    probes = _probe_generator(
        ReasoningGymGenerator(source_root, recipe),
        recipe,
    )
    generator = ReasoningGymGenerator(source_root, recipe)
    base = _base_receipt(base_root, recipe)

    staging = output.with_name(f".{output.name}.building-{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"reasoning corpus staging path exists: {staging}")
    try:
        (staging / "packed").mkdir(parents=True)
        (staging / "sidecars").mkdir()
        compiled = _compile_extension(
            recipe,
            generator,
            staging / _EXTENSION_PATHS["packed_targets"],
            progress=progress,
        )
        shared_sha = _write_ones(
            staging / _EXTENSION_PATHS["shared_target_weights"],
            recipe.extension_tokens,
        )
        artifacts = _extension_artifacts(
            staging,
            packed_sha256=compiled["packed_stream_sha256"],
            shared_sha256=shared_sha,
            extension_tokens=recipe.extension_tokens,
        )
        composite_hashes = _composite_stream_hashes(
            base_root,
            base,
            staging,
            artifacts,
        )
        receipt = {
            "base_corpus": {
                "contract_id": "memorysplit-parallel-corpus-v2",
                "logical_tokens": recipe.base_tokens,
                "ordered_stream_sha256": recipe.base_ordered_sha256,
                "packed_stream_sha256": recipe.base_packed_sha256,
                "receipt_sha256": recipe.base_receipt_sha256,
            },
            "composite": {
                "ordering": "frozen-v2-prefix-then-reasoning-extension",
                "raw_target_tokens": recipe.composite_tokens,
                "stream_sha256": composite_hashes,
                "terminal_updates": recipe.composite_updates,
            },
            "contract_id": recipe.contract_id,
            "extension": {
                "artifacts": [artifacts[name] for name in sorted(artifacts)],
                "format": EXTENSION_FORMAT,
                "max_record_tokens": recipe.max_record_tokens,
                "packed_stream_sha256": compiled["packed_stream_sha256"],
                "probes": probes,
                "raw_target_tokens": recipe.extension_tokens,
                "record_count": compiled["record_count"],
                "records_commitment_sha256": (compiled["records_commitment_sha256"]),
                "shared_target_weights_sha256": shared_sha,
                "target_weight_policy": (
                    "all_extension_reasoning_targets_are_internal_in_both_arms"
                ),
                "targets_per_update": recipe.targets_per_update,
                "task_stats": compiled["task_stats"],
                "terminal_updates": recipe.extension_updates,
            },
            "format": FORMAT,
            "generator_artifacts": _generator_artifacts(recipe),
            "recipe_sha256": recipe.sha256,
            "schema_version": SCHEMA_VERSION,
            "source": {
                "reasoning_gym_tree_commitment_sha256": tree_commitment,
                "reasoning_gym_version": recipe.reasoning_gym_version,
                "source_stage_receipt_sha256": (recipe.source_stage_receipt_sha256),
            },
        }
        receipt_sha = _write_receipt(staging / "receipt.json", receipt)
        _publish_no_replace(staging, output)
    except BaseException:
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    verified = verify_reasoning_corpus(
        publication=output,
        recipe_path=recipe.path,
        source_stage=source_stage_root,
        base_corpus=base_root,
        expected_receipt_sha256=receipt_sha,
    )
    return verified


def _publication_namespace(root: Path) -> None:
    expected_files = {
        "receipt.json",
        *_EXTENSION_PATHS.values(),
    }
    expected_directories = {"packed", "sidecars"}
    files: set[str] = set()
    directories: set[str] = set()
    for member in root.rglob("*"):
        relative = member.relative_to(root).as_posix()
        metadata = member.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ReasoningExpansionError(
                f"reasoning corpus contains a symlink: {relative}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            directories.add(relative)
        elif stat.S_ISREG(metadata.st_mode):
            files.add(relative)
        else:
            raise ReasoningExpansionError(
                f"reasoning corpus contains a special entry: {relative}"
            )
    if files != expected_files or directories != expected_directories:
        raise ReasoningExpansionError(
            "reasoning corpus namespace contains missing or extra entries"
        )


def verify_reasoning_corpus(
    *,
    publication: Path | str,
    recipe_path: Path | str = DEFAULT_RECIPE_PATH,
    source_stage: Path | str = DEFAULT_SOURCE_STAGE,
    base_corpus: Path | str = DEFAULT_BASE_CORPUS,
    expected_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    recipe = load_expansion_recipe(recipe_path)
    root = Path(publication)
    if root.is_symlink() or not root.is_dir():
        raise ReasoningExpansionError("reasoning corpus root is missing or unsafe")
    _publication_namespace(root)
    raw, receipt_bytes = _read_json(root / "receipt.json", "reasoning receipt")
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    if expected_receipt_sha256 is not None and receipt_sha != _require_sha256(
        expected_receipt_sha256, "expected reasoning receipt"
    ):
        raise ReasoningExpansionError("reasoning receipt SHA-256 mismatch")
    if (
        not isinstance(raw, Mapping)
        or canonical_json_bytes(raw) != receipt_bytes
        or set(raw)
        != {
            "base_corpus",
            "composite",
            "contract_id",
            "extension",
            "format",
            "generator_artifacts",
            "recipe_sha256",
            "schema_version",
            "source",
        }
        or raw["schema_version"] != SCHEMA_VERSION
        or raw["format"] != FORMAT
        or raw["contract_id"] != recipe.contract_id
        or raw["recipe_sha256"] != recipe.sha256
    ):
        raise ReasoningExpansionError("reasoning receipt identity differs")
    if raw["generator_artifacts"] != _generator_artifacts(recipe):
        raise ReasoningExpansionError("reasoning generator artifact hashes differ")
    source_stage_root = Path(source_stage)
    tree_commitment = _reasoning_gym_tree_commitment(
        source_stage_root,
        recipe,
    )
    if raw["source"] != {
        "reasoning_gym_tree_commitment_sha256": tree_commitment,
        "reasoning_gym_version": recipe.reasoning_gym_version,
        "source_stage_receipt_sha256": recipe.source_stage_receipt_sha256,
    }:
        raise ReasoningExpansionError("reasoning source identity differs")

    base = _base_receipt(Path(base_corpus), recipe)
    if raw["base_corpus"] != {
        "contract_id": "memorysplit-parallel-corpus-v2",
        "logical_tokens": recipe.base_tokens,
        "ordered_stream_sha256": recipe.base_ordered_sha256,
        "packed_stream_sha256": recipe.base_packed_sha256,
        "receipt_sha256": recipe.base_receipt_sha256,
    }:
        raise ReasoningExpansionError("reasoning base receipt binding differs")

    extension = raw["extension"]
    if not isinstance(extension, Mapping):
        raise ReasoningExpansionError("reasoning extension receipt is missing")
    artifacts_list = extension.get("artifacts")
    if not isinstance(artifacts_list, list) or len(artifacts_list) != 2:
        raise ReasoningExpansionError("reasoning extension artifacts differ")
    artifacts = {}
    for item in artifacts_list:
        if not isinstance(item, Mapping) or set(item) != {
            "bytes",
            "path",
            "sha256",
        }:
            raise ReasoningExpansionError("reasoning extension artifact fields differ")
        name = next(
            (
                key
                for key, relative in _EXTENSION_PATHS.items()
                if item["path"] == relative
            ),
            None,
        )
        if name is None or name in artifacts:
            raise ReasoningExpansionError("reasoning extension artifact paths differ")
        artifacts[name] = item
    expected_artifacts = _extension_artifacts(
        root,
        packed_sha256=_require_sha256(
            extension.get("packed_stream_sha256"),
            "extension packed stream",
        ),
        shared_sha256=_require_sha256(
            extension.get("shared_target_weights_sha256"),
            "extension target weights",
        ),
        extension_tokens=recipe.extension_tokens,
    )
    if artifacts != expected_artifacts:
        raise ReasoningExpansionError(
            "reasoning extension artifact declarations differ"
        )
    for name, artifact in artifacts.items():
        validate: Callable[[bytes], None] | None = None
        if name == "shared_target_weights":

            def require_ones(chunk: bytes) -> None:
                if chunk.strip(b"\x01"):
                    raise ReasoningExpansionError(
                        "reasoning extension target weights are not all one"
                    )

            validate = require_ones
        consumers = () if validate is None else (validate,)
        size, digest = _stream_regular(
            root / artifact["path"],
            label=f"reasoning extension {name}",
            consumers=consumers,
        )
        if size != artifact["bytes"] or digest != artifact["sha256"]:
            raise ReasoningExpansionError(
                f"reasoning extension {name} differs from its receipt"
            )

    probes = _probe_generator(
        ReasoningGymGenerator(
            source_stage_root / recipe.source_relative_path,
            recipe,
        ),
        recipe,
    )
    task_stats = extension.get("task_stats")
    if (
        extension.get("format") != EXTENSION_FORMAT
        or extension.get("raw_target_tokens") != recipe.extension_tokens
        or extension.get("targets_per_update") != recipe.targets_per_update
        or extension.get("terminal_updates") != recipe.extension_updates
        or extension.get("max_record_tokens") != recipe.max_record_tokens
        or extension.get("target_weight_policy")
        != "all_extension_reasoning_targets_are_internal_in_both_arms"
        or extension.get("probes") != probes
        or not isinstance(task_stats, list)
        or [item.get("dataset") for item in task_stats]
        != [task.dataset for task in recipe.tasks]
        or sum(item.get("emitted_tokens", 0) for item in task_stats)
        != recipe.extension_tokens
        or sum(item.get("emitted_records", 0) for item in task_stats)
        != extension.get("record_count")
        or not _is_sha256(extension.get("records_commitment_sha256"))
    ):
        raise ReasoningExpansionError(
            "reasoning extension geometry, probes, or task accounting differs"
        )
    composite_hashes = _composite_stream_hashes(
        Path(base_corpus),
        base,
        root,
        artifacts,
    )
    expected_composite = {
        "ordering": "frozen-v2-prefix-then-reasoning-extension",
        "raw_target_tokens": recipe.composite_tokens,
        "stream_sha256": composite_hashes,
        "terminal_updates": recipe.composite_updates,
    }
    if raw["composite"] != expected_composite:
        raise ReasoningExpansionError("reasoning composite stream hashes differ")
    return {
        "base_receipt_sha256": recipe.base_receipt_sha256,
        "composite_stream_sha256": composite_hashes,
        "contract_id": recipe.contract_id,
        "extension_record_count": extension["record_count"],
        "extension_tokens": recipe.extension_tokens,
        "publication": str(root.resolve(strict=True)),
        "receipt_sha256": receipt_sha,
        "task_count": len(recipe.tasks),
        "total_tokens": recipe.composite_tokens,
    }
