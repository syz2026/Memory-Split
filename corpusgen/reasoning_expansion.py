"""Build and verify an append-only, exact-answer reasoning corpus extension.

The frozen MemorySplit v2 corpus is an immutable component.  This module adds a
new objective-reasoning suffix and publishes a composite receipt; it never
rewrites, reschedules, or aliases the v2 bytes.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import stat
import struct
import sys
import tempfile
import types
from array import array
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from multiprocessing import get_context
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, Self

import numpy as np

from corpusgen.parallel import atomic_rename_noreplace
from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.reasoning_oracles import (
    ReasoningOracleError,
    ReasoningOracleRejection,
    canonical_reasoning_answer,
)
from train.tokenizer import VOCAB_SIZE, get_tok

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE_PATH = ROOT / "configs" / "reasoning-dataset-v3.json"
DEFAULT_SOURCE_STAGE = ROOT / "corpus-build" / "memorysplit-v2-frozen-upstream-sources"
DEFAULT_BASE_CORPUS = ROOT / "corpus-build" / "memorysplit-parallel-corpus-v2"
DEFAULT_OUTPUT = ROOT / "corpus-build" / "memorysplit-reasoning-corpus-v3"
DEFAULT_POINTER = ROOT / "corpus-build" / "memorysplit-reasoning-corpus-v3.pointer.json"

FORMAT = "memorysplit-reasoning-composite-v3"
EXTENSION_FORMAT = "memorysplit-reasoning-extension-v2"
POINTER_FORMAT = "memorysplit-reasoning-pointer-v1"
SCHEMA_VERSION = 2
FINISH_WINDOW = 1 << 18
MAX_FINISH_CANDIDATES = 32_768
PREFETCH_BATCH_SIZE = 256
PROBE_INDICES = (0, 1, 17, 127)
_HEX = frozenset("0123456789abcdef")
_MANIFEST_MAGIC = b"MSR3REC2"
_MANIFEST_HEADER = struct.Struct("<8s32sHH")
_MANIFEST_RECORD = struct.Struct("<HHI")
_EXTENSION_PATHS = {
    "packed_targets": "packed/targets.bin",
    "record_manifest": "records/manifest.bin",
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
    runtime_lock: Mapping[str, str]
    tasks: tuple[TaskSpec, ...]


@dataclass(frozen=True)
class GeneratedRecord:
    task: str
    source_index: int
    token_ids: tuple[int, ...] | None
    token_count: int
    record_sha256: str
    rejection_reason: str | None = None


class RecordGenerator(Protocol):
    def generate(self, task: str, index: int) -> GeneratedRecord: ...


class RecordReplayer(Protocol):
    def replay(
        self,
        requests: Sequence[tuple[str, int]],
    ) -> tuple[GeneratedRecord, ...]: ...


@dataclass
class _PrefetchState:
    next_index: int
    batch_start: int
    rows: tuple[GeneratedRecord, ...]
    future: Future[tuple[GeneratedRecord, ...]] | None


@dataclass
class _TaskStats:
    emitted_records: int = 0
    emitted_tokens: int = 0
    examined_records: int = 0
    oracle_rejections: int = 0
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


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReasoningExpansionError(f"{label} must be a non-negative integer")
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


def effective_tokenizer_sha256() -> str:
    """Fingerprint the effective BPE ranks and special-token assignment."""

    tokenizer = get_tok()
    encoding = tokenizer._enc
    digest = hashlib.sha256(b"memorysplit-effective-tokenizer-v1\0")
    pattern = encoding._pat_str.encode("utf-8")
    digest.update(struct.pack("<I", len(pattern)))
    digest.update(pattern)
    ranks = sorted(
        encoding._mergeable_ranks.items(),
        key=lambda item: (item[1], item[0]),
    )
    digest.update(struct.pack("<I", len(ranks)))
    for token, rank in ranks:
        digest.update(struct.pack("<II", rank, len(token)))
        digest.update(token)
    specials = sorted(encoding._special_tokens.items())
    digest.update(struct.pack("<I", len(specials)))
    for token, rank in specials:
        encoded = token.encode("utf-8")
        digest.update(struct.pack("<II", rank, len(encoded)))
        digest.update(encoded)
    digest.update(struct.pack("<I", VOCAB_SIZE))
    return digest.hexdigest()


def effective_runtime_lock() -> dict[str, str]:
    return {
        "byteorder": sys.byteorder,
        "python_cache_tag": str(sys.implementation.cache_tag),
        "python_hash_seed": str(os.environ.get("PYTHONHASHSEED", "")),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "tiktoken_version": importlib.metadata.version("tiktoken"),
        "tokenizer_sha256": effective_tokenizer_sha256(),
    }


def _require_runtime_lock(recipe: ExpansionRecipe) -> dict[str, str]:
    expected_keys = {
        "byteorder",
        "python_cache_tag",
        "python_hash_seed",
        "python_implementation",
        "python_version",
        "tiktoken_version",
        "tokenizer_sha256",
    }
    if set(recipe.runtime_lock) != expected_keys:
        raise ReasoningExpansionError("reasoning runtime lock fields differ")
    if (
        recipe.runtime_lock["python_hash_seed"] == "0"
        and sys.flags.hash_randomization != 0
    ):
        raise ReasoningExpansionError(
            "reasoning runtime requires hash randomization disabled at startup"
        )
    actual = effective_runtime_lock()
    if dict(recipe.runtime_lock) != actual:
        raise ReasoningExpansionError(
            f"reasoning runtime differs: expected={dict(recipe.runtime_lock)!r}, "
            f"actual={actual!r}"
        )
    return actual


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
        "runtime_lock",
        "schema_version",
        "scientific_scope",
    }:
        raise ReasoningExpansionError("reasoning expansion recipe fields differ")
    if raw["schema_version"] != SCHEMA_VERSION:
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
    runtime_lock = raw["runtime_lock"]
    if not all(
        isinstance(item, Mapping)
        for item in (base, extension, composite, source, runtime_lock)
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
        runtime_lock={
            str(key): str(value) for key, value in sorted(runtime_lock.items())
        },
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


def _reasoning_gym_source_inventory(
    source_stage: Path,
    recipe: ExpansionRecipe,
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]], set[str]]:
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
    expected_files = {}
    expected_directories: set[str] = set()
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
        source_relative = PurePosixPath(relative).relative_to(
            PurePosixPath(recipe.source_relative_path)
        )
        source_name = source_relative.as_posix()
        if source_name in expected_files:
            raise ReasoningExpansionError(
                "Reasoning Gym source inventory repeats a path"
            )
        expected_files[source_name] = item
        parent = source_relative.parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    return selected, expected_files, expected_directories


def _actual_source_namespace(source_root: Path) -> tuple[set[str], set[str]]:
    if source_root.is_symlink() or not source_root.is_dir():
        raise ReasoningExpansionError("Reasoning Gym source root is unsafe")
    files: set[str] = set()
    directories: set[str] = set()
    for directory, names, filenames in os.walk(source_root, followlinks=False):
        current = Path(directory)
        for name in names:
            path = current / name
            relative = path.relative_to(source_root).as_posix()
            metadata = path.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise ReasoningExpansionError(
                    f"Reasoning Gym source directory is unsafe: {relative}"
                )
            directories.add(relative)
        for name in filenames:
            path = current / name
            relative = path.relative_to(source_root).as_posix()
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise ReasoningExpansionError(
                    f"Reasoning Gym source file is unsafe: {relative}"
                )
            files.add(relative)
    return files, directories


def _reasoning_gym_tree_commitment(
    source_stage: Path,
    recipe: ExpansionRecipe,
) -> str:
    selected, expected_files, expected_directories = _reasoning_gym_source_inventory(
        source_stage, recipe
    )
    source_root = source_stage / recipe.source_relative_path
    actual_files, actual_directories = _actual_source_namespace(source_root)
    if (
        actual_files != set(expected_files)
        or actual_directories != expected_directories
    ):
        extra = sorted(
            (actual_files - set(expected_files))
            | (actual_directories - expected_directories)
        )
        missing = sorted(
            (set(expected_files) - actual_files)
            | (expected_directories - actual_directories)
        )
        raise ReasoningExpansionError(
            "Reasoning Gym source namespace differs from its receipt; "
            f"extra={extra[:8]!r}, missing={missing[:8]!r}"
        )
    for relative, item in sorted(expected_files.items()):
        size, digest = _stream_regular(
            source_root / relative,
            label=f"Reasoning Gym source {relative}",
        )
        if size != _nonnegative_int(
            item["bytes"], "Reasoning Gym source bytes"
        ) or digest != _require_sha256(
            item["sha256"],
            "Reasoning Gym source digest",
        ):
            raise ReasoningExpansionError(
                f"Reasoning Gym source differs from its stage receipt: {relative}"
            )
    return hashlib.sha256(canonical_json_bytes(selected)).hexdigest()


def clean_reasoning_gym_bytecode(
    source_stage: Path | str,
    recipe_path: Path | str = DEFAULT_RECIPE_PATH,
) -> int:
    """Remove only unreceipted Python bytecode, then require an exact tree."""

    recipe = load_expansion_recipe(recipe_path)
    source_stage_root = Path(source_stage)
    _selected, expected_files, expected_directories = _reasoning_gym_source_inventory(
        source_stage_root, recipe
    )
    source_root = source_stage_root / recipe.source_relative_path
    actual_files, actual_directories = _actual_source_namespace(source_root)
    missing = (set(expected_files) - actual_files) | (
        expected_directories - actual_directories
    )
    if missing:
        raise ReasoningExpansionError(
            f"Reasoning Gym source is missing receipted entries: {sorted(missing)[:8]!r}"
        )
    extra_files = actual_files - set(expected_files)
    extra_directories = actual_directories - expected_directories
    for relative in extra_files:
        parts = PurePosixPath(relative).parts
        if "__pycache__" not in parts or not relative.endswith((".pyc", ".pyo")):
            raise ReasoningExpansionError(
                f"refusing to remove non-bytecode source extra: {relative}"
            )
    for relative in extra_directories:
        if PurePosixPath(relative).name != "__pycache__":
            raise ReasoningExpansionError(
                f"refusing to remove non-bytecode source directory: {relative}"
            )
    for relative in sorted(extra_files):
        (source_root / relative).unlink()
    for relative in sorted(
        extra_directories,
        key=lambda item: (len(PurePosixPath(item).parts), item),
        reverse=True,
    ):
        (source_root / relative).rmdir()
    _reasoning_gym_tree_commitment(source_stage_root, recipe)
    return len(extra_files)


def _install_reasoning_gym(
    source_root: Path,
    tasks: Sequence[TaskSpec],
):
    sys.dont_write_bytecode = True
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
        self.special_tokens = tuple(sorted(self.tok._enc._special_tokens))
        factory = _install_reasoning_gym(source_root, recipe.tasks)
        self.datasets = {}
        self.task_specs = {task.dataset: task for task in recipe.tasks}
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
        metadata = row.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("source_dataset") != task
            or metadata.get("source_index") != index
        ):
            raise ReasoningExpansionError(
                f"Reasoning Gym {task} returned incorrect source provenance"
            )
        question = row.get("question")
        native_answer = row.get("answer")
        if (
            not isinstance(question, str)
            or not question
            or not isinstance(native_answer, str)
            or not native_answer
        ):
            raise ReasoningExpansionError(
                f"Reasoning Gym {task} returned no exact question/answer"
            )
        try:
            answer = canonical_reasoning_answer(
                task,
                row,
                self.task_specs[task].config,
            )
        except ReasoningOracleRejection as error:
            commitment = {
                "rejection": str(error),
                "source_index": index,
                "task": task,
            }
            return GeneratedRecord(
                task=task,
                source_index=index,
                token_ids=None,
                token_count=0,
                record_sha256=hashlib.sha256(
                    canonical_json_bytes(commitment)
                ).hexdigest(),
                rejection_reason="independent_oracle",
            )
        except ReasoningOracleError as error:
            raise ReasoningExpansionError(
                f"independent {task} oracle failed: {error}"
            ) from error
        if answer != native_answer:
            raise ReasoningExpansionError(
                f"Reasoning Gym {task} oracle disagrees with independent replay"
            )
        score = dataset.score_answer(answer, row)
        if score != 1.0:
            raise ReasoningExpansionError(
                f"Reasoning Gym {task} rejected its own oracle: {score!r}"
            )
        if any(
            special in question or special in answer for special in self.special_tokens
        ):
            raise ReasoningExpansionError(
                f"Reasoning Gym {task} contains a literal special token"
            )
        text = f"Reasoning task={task}\nQuestion: {question}\nAnswer: {answer}"
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
                rejection_reason="overlength",
            )
        if (
            any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                or token >= VOCAB_SIZE
                for token in token_ids
            )
            or token_ids[-1] != self.tok.EOT
            or self.tok.EOT in token_ids[:-1]
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


_PROCESS_GENERATOR: ReasoningGymGenerator | None = None


def _initialize_process_generator(
    source_root: str,
    recipe: ExpansionRecipe,
) -> None:
    global _PROCESS_GENERATOR
    sys.dont_write_bytecode = True
    _PROCESS_GENERATOR = ReasoningGymGenerator(Path(source_root), recipe)


def _process_generate_serial(
    task: str,
    start: int,
    count: int,
) -> tuple[GeneratedRecord, ...]:
    if _PROCESS_GENERATOR is None:
        raise RuntimeError("reasoning process generator is not initialized")
    return tuple(
        _PROCESS_GENERATOR.generate(task, index)
        for index in range(start, start + count)
    )


def _process_generate_indices(
    task: str,
    indices: tuple[int, ...],
) -> tuple[GeneratedRecord, ...]:
    if _PROCESS_GENERATOR is None:
        raise RuntimeError("reasoning process generator is not initialized")
    return tuple(_PROCESS_GENERATOR.generate(task, index) for index in indices)


def _process_pool(
    source_root: Path,
    recipe: ExpansionRecipe,
) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=min(len(recipe.tasks), max(1, os.cpu_count() or 1)),
        mp_context=get_context("spawn"),
        initializer=_initialize_process_generator,
        initargs=(str(source_root.resolve(strict=True)), recipe),
    )


class _ProcessPrefetchingRecordGenerator:
    """Materialize independent task batches in isolated deterministic workers."""

    def __init__(
        self,
        source_root: Path,
        recipe: ExpansionRecipe,
        *,
        batch_size: int = PREFETCH_BATCH_SIZE,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("prefetch batch size must be positive")
        self.batch_size = batch_size
        self.executor = _process_pool(source_root, recipe)
        self.states = {
            task.dataset: _PrefetchState(
                next_index=0,
                batch_start=0,
                rows=(),
                future=self._submit(task.dataset, 0),
            )
            for task in recipe.tasks
        }
        self.closed = False

    def _submit(
        self,
        task: str,
        start: int,
    ) -> Future[tuple[GeneratedRecord, ...]]:
        return self.executor.submit(
            _process_generate_serial,
            task,
            start,
            self.batch_size,
        )

    def generate(self, task: str, index: int) -> GeneratedRecord:
        if self.closed:
            raise RuntimeError("reasoning process prefetch generator is closed")
        state = self.states[task]
        if index != state.next_index:
            raise ReasoningExpansionError(
                f"reasoning process prefetch index is non-serial for {task}: "
                f"expected {state.next_index}, got {index}"
            )
        if not state.rows or index >= state.batch_start + len(state.rows):
            if state.future is None:
                raise RuntimeError("reasoning process prefetch future is missing")
            state.rows = state.future.result()
            state.batch_start = index
            if len(state.rows) != self.batch_size:
                raise ReasoningExpansionError(
                    f"reasoning process prefetch batch size differs for {task}"
                )
            next_start = state.batch_start + self.batch_size
            state.future = self._submit(task, next_start)
        record = state.rows[index - state.batch_start]
        state.next_index += 1
        return record

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for state in self.states.values():
            if state.future is not None:
                state.future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class _GeneratorRecordReplayer:
    def __init__(self, generator: RecordGenerator):
        self.generator = generator

    def replay(
        self,
        requests: Sequence[tuple[str, int]],
    ) -> tuple[GeneratedRecord, ...]:
        return tuple(self.generator.generate(task, index) for task, index in requests)


class _ProcessRecordReplayer:
    """Replay arbitrary manifest indices in task-grouped process batches."""

    def __init__(self, source_root: Path, recipe: ExpansionRecipe):
        self.executor = _process_pool(source_root, recipe)
        self.closed = False

    def replay(
        self,
        requests: Sequence[tuple[str, int]],
    ) -> tuple[GeneratedRecord, ...]:
        if self.closed:
            raise RuntimeError("reasoning process replayer is closed")
        grouped: dict[str, list[tuple[int, int]]] = {}
        for position, (task, index) in enumerate(requests):
            grouped.setdefault(task, []).append((position, index))
        futures = {
            task: self.executor.submit(
                _process_generate_indices,
                task,
                tuple(index for _position, index in positions),
            )
            for task, positions in grouped.items()
        }
        output: list[GeneratedRecord | None] = [None] * len(requests)
        for task, positions in grouped.items():
            records = futures[task].result()
            if len(records) != len(positions):
                raise ReasoningExpansionError(
                    f"reasoning process replay batch size differs for {task}"
                )
            for (position, _index), record in zip(positions, records):
                output[position] = record
        if any(record is None for record in output):
            raise ReasoningExpansionError("reasoning process replay omitted a record")
        return tuple(record for record in output if record is not None)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class _PrefetchingRecordGenerator:
    """Prefetch one serial batch per task without changing record order."""

    def __init__(
        self,
        inner: RecordGenerator,
        tasks: Sequence[TaskSpec],
        *,
        batch_size: int = PREFETCH_BATCH_SIZE,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("prefetch batch size must be positive")
        self.inner = inner
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(
            max_workers=min(len(tasks), max(1, os.cpu_count() or 1)),
            thread_name_prefix="reasoning-prefetch",
        )
        self.states = {
            task.dataset: _PrefetchState(
                next_index=0,
                batch_start=0,
                rows=(),
                future=self._submit(task.dataset, 0),
            )
            for task in tasks
        }
        self.closed = False

    def _generate_batch(
        self,
        task: str,
        start: int,
    ) -> tuple[GeneratedRecord, ...]:
        return tuple(
            self.inner.generate(task, index)
            for index in range(start, start + self.batch_size)
        )

    def _submit(
        self,
        task: str,
        start: int,
    ) -> Future[tuple[GeneratedRecord, ...]]:
        return self.executor.submit(self._generate_batch, task, start)

    def generate(self, task: str, index: int) -> GeneratedRecord:
        if self.closed:
            raise RuntimeError("reasoning prefetch generator is closed")
        state = self.states[task]
        if index != state.next_index:
            raise ReasoningExpansionError(
                f"reasoning prefetch index is non-serial for {task}: "
                f"expected {state.next_index}, got {index}"
            )
        if not state.rows or index >= state.batch_start + len(state.rows):
            if state.future is None:
                raise RuntimeError("reasoning prefetch future is missing")
            state.rows = state.future.result()
            state.batch_start = index
            if len(state.rows) != self.batch_size:
                raise ReasoningExpansionError(
                    f"reasoning prefetch batch size differs for {task}"
                )
            next_start = state.batch_start + self.batch_size
            state.future = self._submit(task, next_start)
        record = state.rows[index - state.batch_start]
        state.next_index += 1
        return record

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for state in self.states.values():
            if state.future is not None:
                state.future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


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
                    "rejection_reason": record.rejection_reason,
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
    def __init__(self, path: Path, manifest_path: Path, recipe: ExpansionRecipe):
        self.handle = path.open("xb")
        try:
            self.manifest_handle = manifest_path.open("xb")
        except BaseException:
            self.handle.close()
            path.unlink(missing_ok=True)
            raise
        self.digest = hashlib.sha256()
        self.manifest_digest = hashlib.sha256()
        self.record_stream_digest = hashlib.sha256(
            b"memorysplit-reasoning-record-stream-v2\0"
        )
        self.task_ids = {task.dataset: index for index, task in enumerate(recipe.tasks)}
        header = _MANIFEST_HEADER.pack(
            _MANIFEST_MAGIC,
            bytes.fromhex(recipe.sha256),
            len(recipe.tasks),
            recipe.max_record_tokens,
        )
        self.manifest_handle.write(header)
        self.manifest_digest.update(header)
        self.record_stream_digest.update(header)
        self.tokens = 0
        self.records = 0
        self.buffer = bytearray()
        self.manifest_buffer = bytearray()

    def add(self, record: GeneratedRecord, payload: bytes | None = None) -> None:
        if record.token_ids is None:
            raise ValueError("cannot write a rejected record")
        data = _u16_bytes(record.token_ids) if payload is None else payload
        if len(data) != record.token_count * 2:
            raise ReasoningExpansionError("record byte/token count differs")
        entry = _MANIFEST_RECORD.pack(
            self.task_ids[record.task],
            record.token_count,
            record.source_index,
        )
        self.buffer.extend(data)
        self.manifest_buffer.extend(entry)
        self.digest.update(data)
        self.manifest_digest.update(entry)
        self.record_stream_digest.update(entry)
        self.record_stream_digest.update(data)
        self.tokens += record.token_count
        self.records += 1
        if len(self.buffer) >= (8 << 20):
            self._flush()

    def _flush(self) -> None:
        if self.buffer:
            self.handle.write(self.buffer)
            self.buffer.clear()
        if self.manifest_buffer:
            self.manifest_handle.write(self.manifest_buffer)
            self.manifest_buffer.clear()

    def close(self) -> None:
        self._flush()
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.manifest_handle.flush()
        os.fsync(self.manifest_handle.fileno())
        self.manifest_handle.close()


def _choose_task(
    recipe: ExpansionRecipe,
    quotas: Mapping[str, int],
    task_tokens: Mapping[str, int],
    written: int,
    eligible: set[str] | None = None,
) -> str:
    return max(
        (
            task.dataset
            for task in recipe.tasks
            if eligible is None or task.dataset in eligible
        ),
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
            if record.rejection_reason == "overlength":
                stats[task].overlength_rejections += 1
            elif record.rejection_reason == "independent_oracle":
                stats[task].oracle_rejections += 1
            else:
                raise ReasoningExpansionError(
                    f"reasoning generator rejection reason differs for {task}"
                )
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
    manifest_destination: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
    finish_window: int = FINISH_WINDOW,
    max_finish_candidates: int = MAX_FINISH_CANDIDATES,
) -> dict[str, Any]:
    if finish_window <= 0 or finish_window >= recipe.extension_tokens:
        raise ValueError("finish window must be inside the extension horizon")
    if finish_window < recipe.max_record_tokens:
        raise ValueError("finish window must cover the maximum record length")
    quotas = _task_quotas(recipe)
    if min(quotas.values()) <= finish_window:
        raise ValueError("finish window must be smaller than every task quota")
    stats = {task.dataset: _TaskStats() for task in recipe.tasks}
    next_indices = {task.dataset: 0 for task in recipe.tasks}
    task_tokens = {task.dataset: 0 for task in recipe.tasks}
    writer = _TokenWriter(destination, manifest_destination, recipe)
    next_progress = 64 << 20
    try:
        active = {
            task.dataset
            for task in recipe.tasks
            if quotas[task.dataset] - task_tokens[task.dataset] > finish_window
        }
        while active:
            task = _choose_task(
                recipe,
                quotas,
                task_tokens,
                writer.tokens,
                active,
            )
            record = _next_accepted(generator, task, next_indices, stats)
            if record.token_count > quotas[task] - task_tokens[task]:
                raise ReasoningExpansionError(
                    f"non-final {task} record exceeds its remaining quota"
                )
            writer.add(record)
            stats[task].emitted_records += 1
            stats[task].emitted_tokens += record.token_count
            task_tokens[task] += record.token_count
            if quotas[task] - task_tokens[task] <= finish_window:
                active.remove(task)
            if progress is not None and writer.tokens >= next_progress:
                progress(writer.tokens, recipe.extension_tokens)
                next_progress += 64 << 20

        for task_spec in recipe.tasks:
            task = task_spec.dataset
            target = quotas[task] - task_tokens[task]
            subset = _ExactSubsetAccumulator(target)
            candidates: list[_Candidate] = []
            selected: tuple[int, ...] | None = () if target == 0 else None
            with tempfile.TemporaryFile(
                mode="w+b",
                prefix=f".reasoning-finish-{task}-",
                dir=destination.parent,
            ) as spool:
                for _ in range(max_finish_candidates):
                    if selected is not None:
                        break
                    record = _next_accepted(
                        generator,
                        task,
                        next_indices,
                        stats,
                    )
                    if record.token_count > target:
                        continue
                    assert record.token_ids is not None
                    offset = spool.tell()
                    spool.write(_u16_bytes(record.token_ids))
                    candidates.append(_Candidate(offset=offset, record=record))
                    selected = subset.add(record.token_count)
                if selected is None:
                    raise ReasoningExpansionError(
                        f"could not exactly fill the final {target} {task} tokens "
                        f"from {len(candidates)} fresh records"
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
                    stats[task].emitted_records += 1
                    stats[task].emitted_tokens += record.token_count
                    task_tokens[task] += record.token_count
            if task_tokens[task] != quotas[task]:
                raise ReasoningExpansionError(
                    f"{task} did not meet its exact Hamilton quota"
                )
        if writer.tokens != recipe.extension_tokens or task_tokens != quotas:
            raise ReasoningExpansionError(
                "reasoning extension did not meet its exact token horizon"
            )
    finally:
        writer.close()
    if progress is not None:
        progress(writer.tokens, recipe.extension_tokens)
    return {
        "manifest_bytes": _MANIFEST_HEADER.size
        + writer.records * _MANIFEST_RECORD.size,
        "manifest_sha256": writer.manifest_digest.hexdigest(),
        "packed_stream_sha256": writer.digest.hexdigest(),
        "record_count": writer.records,
        "record_stream_sha256": writer.record_stream_digest.hexdigest(),
        "task_stats": [
            {
                "dataset": task.dataset,
                "emitted_records": stats[task.dataset].emitted_records,
                "emitted_tokens": stats[task.dataset].emitted_tokens,
                "examined_records": stats[task.dataset].examined_records,
                "oracle_rejections": stats[task.dataset].oracle_rejections,
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
        "corpusgen/parallel/canonical.py": ROOT
        / "corpusgen"
        / "parallel"
        / "canonical.py",
        "corpusgen/reasoning_expansion.py": Path(__file__).resolve(),
        "corpusgen/reasoning_oracles.py": ROOT / "corpusgen" / "reasoning_oracles.py",
        "train/tokenizer.py": ROOT / "train" / "tokenizer.py",
    }
    return {name: sha256_file(path) for name, path in sorted(paths.items())} | {
        "configs/reasoning-dataset-v3.json": recipe.sha256
    }


def _extension_artifacts(
    *,
    manifest_bytes: int,
    manifest_sha256: str,
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
        "record_manifest": {
            "bytes": manifest_bytes,
            "path": _EXTENSION_PATHS["record_manifest"],
            "sha256": manifest_sha256,
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publication_snapshot(
    root: Path,
) -> dict[str, tuple[int, int, int, int, int, int, int]]:
    paths = [root, *root.rglob("*")]
    snapshot = {}
    for path in paths:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ReasoningExpansionError(
                "reasoning publication snapshot encountered a symlink"
            )
        relative = "." if path == root else path.relative_to(root).as_posix()
        snapshot[relative] = (
            metadata.st_mode,
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
    return snapshot


def _publish_no_replace(
    staging: Path,
    destination: Path,
    expected_snapshot: Mapping[str, tuple[int, int, int, int, int, int, int]],
) -> None:
    parent = destination.parent
    descriptor = os.open(
        parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        metadata = os.stat(
            staging.name,
            dir_fd=descriptor,
            follow_symlinks=False,
        )
        root_snapshot = expected_snapshot.get(".")
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or root_snapshot is None
            or (metadata.st_dev, metadata.st_ino)
            != (root_snapshot[1], root_snapshot[2])
            or _publication_snapshot(staging) != dict(expected_snapshot)
        ):
            raise ReasoningExpansionError(
                "reasoning staging publication changed after verification"
            )
        final_metadata = os.stat(
            staging.name,
            dir_fd=descriptor,
            follow_symlinks=False,
        )
        if (final_metadata.st_dev, final_metadata.st_ino) != (
            root_snapshot[1],
            root_snapshot[2],
        ):
            raise ReasoningExpansionError(
                "reasoning staging directory identity changed before publication"
            )
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
    pointer_destination: Path | str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    recipe = load_expansion_recipe(recipe_path)
    source_stage_root = Path(source_stage)
    base_root = Path(base_corpus)
    output = Path(destination)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"reasoning corpus destination exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pointer = None if pointer_destination is None else Path(pointer_destination)
    if pointer is not None:
        if pointer.exists() or pointer.is_symlink():
            raise FileExistsError(f"reasoning corpus pointer exists: {pointer}")
        pointer.parent.mkdir(parents=True, exist_ok=True)
        if pointer.parent.resolve(strict=True) != output.parent.resolve(strict=True):
            raise ReasoningExpansionError(
                "reasoning corpus and pointer must share a parent directory"
            )
    clean_reasoning_gym_bytecode(source_stage_root, recipe.path)
    runtime_lock = _require_runtime_lock(recipe)
    generator_artifacts = _generator_artifacts(recipe)
    expected_bytes = recipe.extension_tokens * 3 + (512 << 20)
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
    base = _base_receipt(base_root, recipe)

    staging = output.with_name(f".{output.name}.building-{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"reasoning corpus staging path exists: {staging}")
    try:
        (staging / "packed").mkdir(parents=True)
        (staging / "records").mkdir()
        (staging / "sidecars").mkdir()
        with _ProcessPrefetchingRecordGenerator(
            source_root,
            recipe,
        ) as generator:
            compiled = _compile_extension(
                recipe,
                generator,
                staging / _EXTENSION_PATHS["packed_targets"],
                staging / _EXTENSION_PATHS["record_manifest"],
                progress=progress,
            )
        shared_sha = _write_ones(
            staging / _EXTENSION_PATHS["shared_target_weights"],
            recipe.extension_tokens,
        )
        artifacts = _extension_artifacts(
            manifest_bytes=compiled["manifest_bytes"],
            manifest_sha256=compiled["manifest_sha256"],
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
                "record_stream_sha256": compiled["record_stream_sha256"],
                "shared_target_weights_sha256": shared_sha,
                "target_weight_policy": (
                    "all_extension_reasoning_targets_are_internal_in_both_arms"
                ),
                "targets_per_update": recipe.targets_per_update,
                "task_stats": compiled["task_stats"],
                "terminal_updates": recipe.extension_updates,
            },
            "format": FORMAT,
            "generator_artifacts": generator_artifacts,
            "recipe_sha256": recipe.sha256,
            "runtime_lock": runtime_lock,
            "schema_version": SCHEMA_VERSION,
            "source": {
                "reasoning_gym_tree_commitment_sha256": tree_commitment,
                "reasoning_gym_version": recipe.reasoning_gym_version,
                "source_stage_receipt_sha256": (recipe.source_stage_receipt_sha256),
            },
        }
        if (
            _require_runtime_lock(recipe) != runtime_lock
            or _generator_artifacts(recipe) != generator_artifacts
            or _reasoning_gym_tree_commitment(source_stage_root, recipe)
            != tree_commitment
        ):
            raise ReasoningExpansionError(
                "reasoning generation inputs changed during materialization"
            )
        receipt_sha = _write_receipt(staging / "receipt.json", receipt)
        for directory in (
            staging / "packed",
            staging / "records",
            staging / "sidecars",
            staging,
        ):
            _fsync_directory(directory)
        staging_snapshot = _publication_snapshot(staging)
        verified = verify_reasoning_corpus(
            publication=staging,
            recipe_path=recipe.path,
            source_stage=source_stage_root,
            base_corpus=base_root,
            expected_receipt_sha256=receipt_sha,
        )
        _publish_no_replace(staging, output, staging_snapshot)
        verified["publication"] = str(output.resolve(strict=True))
    except BaseException:
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    if pointer is not None:
        pointer_report = _publish_reasoning_pointer(
            pointer,
            output,
            recipe,
            verified,
        )
        verified.update(pointer_report)
    return verified


def _publication_namespace(root: Path) -> None:
    expected_files = {
        "receipt.json",
        *_EXTENSION_PATHS.values(),
    }
    expected_directories = {"packed", "records", "sidecars"}
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


def _verify_record_manifest(
    root: Path,
    recipe: ExpansionRecipe,
    artifacts: Mapping[str, Mapping[str, Any]],
    extension: Mapping[str, Any],
    replayer: RecordReplayer | None = None,
) -> dict[str, Any]:
    manifest_path = root / artifacts["record_manifest"]["path"]
    packed_path = root / artifacts["packed_targets"]["path"]
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    manifest_descriptor: int | None = None
    packed_descriptor: int | None = None
    try:
        manifest_descriptor = os.open(manifest_path, flags)
        packed_descriptor = os.open(packed_path, flags)
    except OSError as error:
        if manifest_descriptor is not None:
            os.close(manifest_descriptor)
        raise ReasoningExpansionError(
            "reasoning record streams are missing, symlinked, or unsafe"
        ) from error
    assert manifest_descriptor is not None
    assert packed_descriptor is not None
    manifest_before = os.fstat(manifest_descriptor)
    packed_before = os.fstat(packed_descriptor)
    if (
        not stat.S_ISREG(manifest_before.st_mode)
        or manifest_before.st_nlink != 1
        or not stat.S_ISREG(packed_before.st_mode)
        or packed_before.st_nlink != 1
    ):
        os.close(manifest_descriptor)
        os.close(packed_descriptor)
        raise ReasoningExpansionError(
            "reasoning record streams must be singly linked regular files"
        )

    manifest_digest = hashlib.sha256()
    packed_digest = hashlib.sha256()
    record_stream_digest = hashlib.sha256(b"memorysplit-reasoning-record-stream-v2\0")
    expected_header = _MANIFEST_HEADER.pack(
        _MANIFEST_MAGIC,
        bytes.fromhex(recipe.sha256),
        len(recipe.tasks),
        recipe.max_record_tokens,
    )
    record_count = 0
    task_records = [0] * len(recipe.tasks)
    task_tokens = [0] * len(recipe.tasks)
    last_source_indices = [-1] * len(recipe.tasks)
    eot = get_tok().EOT
    eot_bytes = struct.pack("<H", eot)
    eot_count = 0
    maximum_token = 0
    validation_buffer = bytearray()
    replay_queue: list[tuple[int, int, int, bytes]] = []

    def validate_buffer() -> None:
        nonlocal eot_count, maximum_token
        if not validation_buffer:
            return
        values = np.frombuffer(validation_buffer, dtype="<u2")
        if values.size:
            maximum_token = max(maximum_token, int(values.max()))
            eot_count += int(np.count_nonzero(values == eot))
        del values
        validation_buffer.clear()

    def validate_replays() -> None:
        if not replay_queue:
            return
        if replayer is None:
            raise RuntimeError("reasoning record replayer is missing")
        generated_records = replayer.replay(
            tuple(
                (recipe.tasks[task_id].dataset, source_index)
                for task_id, source_index, _token_count, _payload in replay_queue
            )
        )
        if len(generated_records) != len(replay_queue):
            raise ReasoningExpansionError("reasoning record replay count differs")
        for generated, (
            task_id,
            source_index,
            token_count,
            payload,
        ) in zip(generated_records, replay_queue):
            task = recipe.tasks[task_id].dataset
            if (
                generated.task != task
                or generated.source_index != source_index
                or generated.token_ids is None
                or generated.token_count != token_count
                or _u16_bytes(generated.token_ids) != payload
            ):
                raise ReasoningExpansionError(
                    f"reasoning record replay differs for {task}[{source_index}]"
                )
        replay_queue.clear()

    try:
        with (
            os.fdopen(
                manifest_descriptor,
                "rb",
                buffering=8 << 20,
                closefd=False,
            ) as manifest_handle,
            os.fdopen(
                packed_descriptor,
                "rb",
                buffering=8 << 20,
                closefd=False,
            ) as packed_handle,
        ):
            header = manifest_handle.read(_MANIFEST_HEADER.size)
            if header != expected_header:
                raise ReasoningExpansionError(
                    "reasoning record manifest header differs"
                )
            manifest_digest.update(header)
            record_stream_digest.update(header)
            while True:
                entry = manifest_handle.read(_MANIFEST_RECORD.size)
                if not entry:
                    break
                if len(entry) != _MANIFEST_RECORD.size:
                    raise ReasoningExpansionError(
                        "reasoning record manifest has a partial entry"
                    )
                task_id, token_count, source_index = _MANIFEST_RECORD.unpack(entry)
                if task_id >= len(recipe.tasks):
                    raise ReasoningExpansionError(
                        "reasoning record manifest task id is invalid"
                    )
                if (
                    token_count <= 0
                    or token_count > recipe.max_record_tokens
                    or source_index <= last_source_indices[task_id]
                ):
                    raise ReasoningExpansionError(
                        "reasoning record manifest ordering/length is invalid"
                    )
                payload = packed_handle.read(token_count * 2)
                if len(payload) != token_count * 2 or payload[-2:] != eot_bytes:
                    raise ReasoningExpansionError(
                        "reasoning packed record is truncated or lacks terminal EOT"
                    )
                manifest_digest.update(entry)
                packed_digest.update(payload)
                record_stream_digest.update(entry)
                record_stream_digest.update(payload)
                validation_buffer.extend(payload)
                if len(validation_buffer) >= (8 << 20):
                    validate_buffer()
                if replayer is not None:
                    replay_queue.append(
                        (
                            task_id,
                            source_index,
                            token_count,
                            payload,
                        )
                    )
                    if len(replay_queue) >= 4096:
                        validate_replays()
                record_count += 1
                task_records[task_id] += 1
                task_tokens[task_id] += token_count
                last_source_indices[task_id] = source_index
            if packed_handle.read(1):
                raise ReasoningExpansionError(
                    "reasoning packed stream has unbound trailing bytes"
                )
        validate_buffer()
        validate_replays()
        manifest_after = os.fstat(manifest_descriptor)
        packed_after = os.fstat(packed_descriptor)
    finally:
        os.close(manifest_descriptor)
        os.close(packed_descriptor)
    if (
        manifest_before.st_dev,
        manifest_before.st_ino,
        manifest_before.st_size,
        manifest_before.st_mtime_ns,
        manifest_before.st_ctime_ns,
    ) != (
        manifest_after.st_dev,
        manifest_after.st_ino,
        manifest_after.st_size,
        manifest_after.st_mtime_ns,
        manifest_after.st_ctime_ns,
    ) or (
        packed_before.st_dev,
        packed_before.st_ino,
        packed_before.st_size,
        packed_before.st_mtime_ns,
        packed_before.st_ctime_ns,
    ) != (
        packed_after.st_dev,
        packed_after.st_ino,
        packed_after.st_size,
        packed_after.st_mtime_ns,
        packed_after.st_ctime_ns,
    ):
        raise ReasoningExpansionError(
            "reasoning record streams changed during verification"
        )
    if maximum_token >= VOCAB_SIZE or eot_count != record_count:
        raise ReasoningExpansionError(
            "reasoning packed stream has invalid token ids or embedded EOT"
        )
    if (
        manifest_before.st_size != artifacts["record_manifest"]["bytes"]
        or manifest_digest.hexdigest() != artifacts["record_manifest"]["sha256"]
        or packed_before.st_size != artifacts["packed_targets"]["bytes"]
        or packed_digest.hexdigest() != artifacts["packed_targets"]["sha256"]
        or packed_digest.hexdigest() != extension.get("packed_stream_sha256")
        or record_stream_digest.hexdigest() != extension.get("record_stream_sha256")
        or record_count != extension.get("record_count")
    ):
        raise ReasoningExpansionError(
            "reasoning manifest/packed stream digests or counts differ"
        )

    raw_stats = extension.get("task_stats")
    if not isinstance(raw_stats, list) or len(raw_stats) != len(recipe.tasks):
        raise ReasoningExpansionError("reasoning task accounting is missing")
    quotas = _task_quotas(recipe)
    for task_id, (task, item) in enumerate(zip(recipe.tasks, raw_stats)):
        if not isinstance(item, Mapping):
            raise ReasoningExpansionError("reasoning task accounting is invalid")
        if (
            item.get("dataset") != task.dataset
            or item.get("emitted_records") != task_records[task_id]
            or item.get("emitted_tokens") != task_tokens[task_id]
            or item.get("emitted_tokens") != quotas[task.dataset]
            or item.get("target_quota") != quotas[task.dataset]
            or item.get("source_cursor") != item.get("examined_records")
            or not isinstance(item.get("source_cursor"), int)
            or item["source_cursor"] <= last_source_indices[task_id]
            or not isinstance(item.get("oracle_rejections"), int)
            or item["oracle_rejections"] < 0
            or not isinstance(item.get("overlength_rejections"), int)
            or item["overlength_rejections"] < 0
        ):
            raise ReasoningExpansionError(
                f"reasoning task accounting differs for {task.dataset}"
            )
    return {
        "manifest_sha256": manifest_digest.hexdigest(),
        "packed_sha256": packed_digest.hexdigest(),
        "record_count": record_count,
        "record_stream_sha256": record_stream_digest.hexdigest(),
        "replayed_records": record_count if replayer is not None else 0,
    }


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
            "runtime_lock",
            "schema_version",
            "source",
        }
        or raw["schema_version"] != SCHEMA_VERSION
        or raw["format"] != FORMAT
        or raw["contract_id"] != recipe.contract_id
        or raw["recipe_sha256"] != recipe.sha256
    ):
        raise ReasoningExpansionError("reasoning receipt identity differs")
    runtime_lock = _require_runtime_lock(recipe)
    if raw["runtime_lock"] != runtime_lock:
        raise ReasoningExpansionError("reasoning receipt runtime lock differs")
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
    expected_extension_fields = {
        "artifacts",
        "format",
        "max_record_tokens",
        "packed_stream_sha256",
        "probes",
        "raw_target_tokens",
        "record_count",
        "record_stream_sha256",
        "shared_target_weights_sha256",
        "target_weight_policy",
        "targets_per_update",
        "task_stats",
        "terminal_updates",
    }
    if (
        not isinstance(extension, Mapping)
        or set(extension) != expected_extension_fields
    ):
        raise ReasoningExpansionError("reasoning extension receipt is missing")
    record_count = _positive_int(
        extension.get("record_count"),
        "extension record count",
    )
    packed_sha = _require_sha256(
        extension.get("packed_stream_sha256"),
        "extension packed stream",
    )
    _require_sha256(
        extension.get("record_stream_sha256"),
        "extension record stream",
    )
    shared_sha = _require_sha256(
        extension.get("shared_target_weights_sha256"),
        "extension target weights",
    )
    artifacts_list = extension.get("artifacts")
    if not isinstance(artifacts_list, list) or len(artifacts_list) != len(
        _EXTENSION_PATHS
    ):
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
    if set(artifacts) != set(_EXTENSION_PATHS):
        raise ReasoningExpansionError("reasoning extension artifact set differs")
    expected_artifacts = _extension_artifacts(
        manifest_bytes=(_MANIFEST_HEADER.size + record_count * _MANIFEST_RECORD.size),
        manifest_sha256=_require_sha256(
            artifacts["record_manifest"].get("sha256"),
            "extension record manifest",
        ),
        packed_sha256=packed_sha,
        shared_sha256=shared_sha,
        extension_tokens=recipe.extension_tokens,
    )
    if artifacts != expected_artifacts:
        raise ReasoningExpansionError(
            "reasoning extension artifact declarations differ"
        )
    if artifacts_list != [
        expected_artifacts[name] for name in sorted(expected_artifacts)
    ]:
        raise ReasoningExpansionError("reasoning extension artifact order differs")

    def require_ones(chunk: bytes) -> None:
        if chunk.strip(b"\x01"):
            raise ReasoningExpansionError(
                "reasoning extension target weights are not all one"
            )

    weight_artifact = artifacts["shared_target_weights"]
    size, digest = _stream_regular(
        root / weight_artifact["path"],
        label="reasoning extension shared_target_weights",
        consumers=(require_ones,),
    )
    if size != weight_artifact["bytes"] or digest != weight_artifact["sha256"]:
        raise ReasoningExpansionError(
            "reasoning extension shared_target_weights differs from its receipt"
        )

    generator = ReasoningGymGenerator(
        source_stage_root / recipe.source_relative_path,
        recipe,
    )
    probes = _probe_generator(generator, recipe)
    if (
        extension.get("format") != EXTENSION_FORMAT
        or extension.get("raw_target_tokens") != recipe.extension_tokens
        or extension.get("targets_per_update") != recipe.targets_per_update
        or extension.get("terminal_updates") != recipe.extension_updates
        or extension.get("max_record_tokens") != recipe.max_record_tokens
        or extension.get("target_weight_policy")
        != "all_extension_reasoning_targets_are_internal_in_both_arms"
        or extension.get("probes") != probes
    ):
        raise ReasoningExpansionError(
            "reasoning extension geometry, probes, or task accounting differs"
        )
    with _ProcessRecordReplayer(
        source_stage_root / recipe.source_relative_path,
        recipe,
    ) as replayer:
        manifest_report = _verify_record_manifest(
            root,
            recipe,
            artifacts,
            extension,
            replayer,
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
        "extension_manifest_sha256": manifest_report["manifest_sha256"],
        "extension_packed_sha256": manifest_report["packed_sha256"],
        "extension_record_count": record_count,
        "extension_record_stream_sha256": manifest_report["record_stream_sha256"],
        "extension_tokens": recipe.extension_tokens,
        "publication": str(root.resolve(strict=True)),
        "receipt_sha256": receipt_sha,
        "replayed_records": manifest_report["replayed_records"],
        "task_count": len(recipe.tasks),
        "total_tokens": recipe.composite_tokens,
    }


def _pointer_payload(
    publication: Path,
    recipe: ExpansionRecipe,
    verified: Mapping[str, Any],
) -> dict[str, Any]:
    composite = verified.get("composite_stream_sha256")
    if (
        verified.get("contract_id") != recipe.contract_id
        or verified.get("total_tokens") != recipe.composite_tokens
        or verified.get("task_count") != len(recipe.tasks)
        or not _is_sha256(verified.get("receipt_sha256"))
        or not _is_sha256(verified.get("extension_manifest_sha256"))
        or not _is_sha256(verified.get("extension_packed_sha256"))
        or not _is_sha256(verified.get("extension_record_stream_sha256"))
        or not isinstance(composite, Mapping)
        or set(composite)
        != {
            "dense_target_weights",
            "packed_targets",
            "split90_target_weights",
        }
        or any(not _is_sha256(value) for value in composite.values())
    ):
        raise ReasoningExpansionError(
            "verified reasoning report cannot freeze a pointer"
        )
    return {
        "contract_id": recipe.contract_id,
        "expected_base_receipt_sha256": recipe.base_receipt_sha256,
        "expected_composite_stream_sha256": dict(composite),
        "expected_extension_manifest_sha256": verified["extension_manifest_sha256"],
        "expected_extension_packed_sha256": verified["extension_packed_sha256"],
        "expected_extension_record_stream_sha256": verified[
            "extension_record_stream_sha256"
        ],
        "expected_receipt_sha256": verified["receipt_sha256"],
        "format": POINTER_FORMAT,
        "launch_gate_status": "frozen",
        "raw_target_tokens": recipe.composite_tokens,
        "receipt_relative_path": "receipt.json",
        "recipe_sha256": recipe.sha256,
        "relative_path": publication.name,
        "schema_version": 1,
        "scientific_scope": "successor_exploratory_unpreregistered",
        "source_stage_receipt_sha256": recipe.source_stage_receipt_sha256,
        "task_count": len(recipe.tasks),
    }


def _publish_reasoning_pointer(
    pointer: Path,
    publication: Path,
    recipe: ExpansionRecipe,
    verified: Mapping[str, Any],
) -> dict[str, Any]:
    if pointer.exists() or pointer.is_symlink():
        raise FileExistsError(f"reasoning corpus pointer exists: {pointer}")
    if pointer.parent.resolve(strict=True) != publication.parent.resolve(strict=True):
        raise ReasoningExpansionError(
            "reasoning corpus and pointer must share a parent directory"
        )
    payload = _pointer_payload(publication, recipe, verified)
    temporary = pointer.with_name(f".{pointer.name}.building-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"reasoning pointer staging path exists: {temporary}")
    try:
        pointer_sha = _write_receipt(temporary, payload)
        metadata = temporary.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise ReasoningExpansionError("reasoning pointer staging file is unsafe")
        descriptor = os.open(
            pointer.parent,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            atomic_rename_noreplace(
                descriptor,
                temporary.name,
                descriptor,
                pointer.name,
            )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "pointer": str(pointer.resolve(strict=True)),
        "pointer_sha256": pointer_sha,
    }


def verify_reasoning_pointer(
    *,
    pointer: Path | str = DEFAULT_POINTER,
    recipe_path: Path | str = DEFAULT_RECIPE_PATH,
    source_stage: Path | str = DEFAULT_SOURCE_STAGE,
    base_corpus: Path | str = DEFAULT_BASE_CORPUS,
    expected_pointer_sha256: str | None = None,
) -> dict[str, Any]:
    pointer_path = Path(pointer)
    raw, pointer_bytes = _read_json(pointer_path, "reasoning corpus pointer")
    pointer_sha = hashlib.sha256(pointer_bytes).hexdigest()
    if expected_pointer_sha256 is not None and pointer_sha != _require_sha256(
        expected_pointer_sha256,
        "expected reasoning pointer",
    ):
        raise ReasoningExpansionError("reasoning pointer SHA-256 mismatch")
    expected_fields = {
        "contract_id",
        "expected_base_receipt_sha256",
        "expected_composite_stream_sha256",
        "expected_extension_manifest_sha256",
        "expected_extension_packed_sha256",
        "expected_extension_record_stream_sha256",
        "expected_receipt_sha256",
        "format",
        "launch_gate_status",
        "raw_target_tokens",
        "receipt_relative_path",
        "recipe_sha256",
        "relative_path",
        "schema_version",
        "scientific_scope",
        "source_stage_receipt_sha256",
        "task_count",
    }
    recipe = load_expansion_recipe(recipe_path)
    relative = raw.get("relative_path") if isinstance(raw, Mapping) else None
    if (
        not isinstance(raw, Mapping)
        or canonical_json_bytes(raw) != pointer_bytes
        or set(raw) != expected_fields
        or raw.get("format") != POINTER_FORMAT
        or raw.get("schema_version") != 1
        or raw.get("launch_gate_status") != "frozen"
        or raw.get("scientific_scope") != "successor_exploratory_unpreregistered"
        or raw.get("contract_id") != recipe.contract_id
        or raw.get("recipe_sha256") != recipe.sha256
        or raw.get("expected_base_receipt_sha256") != recipe.base_receipt_sha256
        or raw.get("source_stage_receipt_sha256") != recipe.source_stage_receipt_sha256
        or raw.get("raw_target_tokens") != recipe.composite_tokens
        or raw.get("task_count") != len(recipe.tasks)
        or raw.get("receipt_relative_path") != "receipt.json"
        or not isinstance(relative, str)
        or len(PurePosixPath(relative).parts) != 1
        or relative in ("", ".", "..")
    ):
        raise ReasoningExpansionError("reasoning corpus pointer identity differs")
    publication = pointer_path.parent / relative
    verified = verify_reasoning_corpus(
        publication=publication,
        recipe_path=recipe.path,
        source_stage=source_stage,
        base_corpus=base_corpus,
        expected_receipt_sha256=raw["expected_receipt_sha256"],
    )
    if (
        verified["base_receipt_sha256"] != raw["expected_base_receipt_sha256"]
        or verified["composite_stream_sha256"]
        != raw["expected_composite_stream_sha256"]
        or verified["extension_manifest_sha256"]
        != raw["expected_extension_manifest_sha256"]
        or verified["extension_packed_sha256"]
        != raw["expected_extension_packed_sha256"]
        or verified["extension_record_stream_sha256"]
        != raw["expected_extension_record_stream_sha256"]
    ):
        raise ReasoningExpansionError(
            "reasoning corpus differs from its frozen pointer"
        )
    verified.update(
        {
            "pointer": str(pointer_path.resolve(strict=True)),
            "pointer_sha256": pointer_sha,
        }
    )
    return verified
