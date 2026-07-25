"""Strict parser for the prospectively frozen reasoning-v3 evaluation contract."""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.reasoning_expansion import (
    _MANIFEST_HEADER,
    _MANIFEST_RECORD,
    load_expansion_recipe,
)
from evals.reasoning_v3.aws_authority import (
    ACTIVATION_KEY,
    AUTHORITY_RECORD_KEY,
    AUTHORITY_SIGNATURE_KEY,
    AWS_BOUNDARY_CONFIG_PATH,
    AWS_REGION,
    EVALUATOR_ROLE_ARN,
    SEALED_GOLD_KMS_KEY_ALIAS,
    SIGNER_KEY_ALIAS,
    STORAGE_BUCKET,
    STORAGE_PREFIX,
    _parse_aws_boundary_record,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH = (
    ROOT / "configs" / "preregistration-135m-reasoning-v3-eval-v1.yaml"
)
CONTRACT_ID = "memorysplit-reasoning-v3-eval-v1"
DATASET_CONTRACT_ID = "memorysplit-reasoning-dataset-v3"
FAMILY_ORDER = (
    "course_schedule",
    "binary_matrix",
    "rotten_oranges",
    "futoshiki",
    "ransom_note",
    "dice",
    "string_splitting",
    "path_star",
    "largest_island",
    "rotate_matrix",
    "spiral_matrix",
    "prime_factorization",
    "base_conversion",
    "count_primes",
)
FAMILY_MAX_NEW_TOKENS = {
    "course_schedule": 8,
    "binary_matrix": 512,
    "rotten_oranges": 16,
    "futoshiki": 128,
    "ransom_note": 8,
    "dice": 32,
    "string_splitting": 32,
    "path_star": 128,
    "largest_island": 16,
    "rotate_matrix": 512,
    "spiral_matrix": 512,
    "prime_factorization": 128,
    "base_conversion": 64,
    "count_primes": 16,
}
INDEX_BASE = 2_000_000_000
INDEX_WINDOW_SIZE = 10_000
ACCEPTED_ITEMS_PER_FAMILY = 512
MAX_RECORD_TOKENS = 1_024
SCORER_ID = "memorysplit-independent-exact-v1"
SKIP_REASONS = ("independent_oracle", "overlength")

_HEX = frozenset("0123456789abcdef")
_CORPUS_RECEIPT_PATH = "extension/receipt.json"
_SOURCE_STAGE_RECEIPT_PATH = "source-stage-receipt.json"
_SOURCE_RELATIVE_PATH = (
    "objective_auxiliary/reasoning_gym_exact_answer/reasoning_gym"
)
_SOURCE_COMMIT = "4fbdd59860198f2ccc623dc2cdd1aeb5af254afa"
_SOURCE_TREE = "a6f4f51213acaf57f63371e878c247e0179c345e"
_CORPUS_RECEIPT = "b1eabb1719f66876ab54cc0791b857ccdbbbddb0ffb8c5986ac2aaa7bf33b80d"
_RECIPE_SHA256 = "276c0f85e2551af5df66b866b2c424118e7a8ddbf9efe2eff805196ba4d60a2f"
_TRANSFER_MANIFEST_SHA256 = (
    "84142597cebd96e041d47c7c22dd4b42285b71a213b01265728042cb1a8f6fbb"
)
_RECORD_MANIFEST_SHA256 = (
    "558b70fd0ad55ba2bf91b1efeea1b49583c9fe3d0286c71571fa868686a8acc8"
)
_SOURCE_STAGE_RECEIPT_SHA256 = (
    "ba83813d73ebccd4812c8297be58176efca8f9e36368dfa04058408fd658ea0c"
)
_FROZEN_LOCK_SHA256 = (
    "dd4f5083c90dd4b75e9b3c2da4db34c7f6299c7b845d15f960a13a3f9f3e849f"
)
_RUNTIME_LOCK = {
    "byteorder": "little",
    "python_cache_tag": "cpython-312",
    "python_hash_seed": "0",
    "python_implementation": "CPython",
    "python_version": "3.12.3",
    "tiktoken_version": "0.13.0",
    "tokenizer_sha256": (
        "96edc08ae20a0e7d545d3fab086eefe17e9908db6c8722085ff2521074c54126"
    ),
}
_EVALUATOR_PATHS = (
    "evals/reasoning_v3/aws_authority.py",
    "evals/reasoning_v3/contracts.py",
    "evals/reasoning_v3/generate.py",
    "evals/reasoning_v3/sealing.py",
)


class EvaluationContractError(ValueError):
    """The evaluation contract or one of its frozen bindings is invalid."""


class _StrictLoader(yaml.SafeLoader):
    def compose_node(self, parent: Any, index: Any) -> yaml.nodes.Node:
        if self.check_event(yaml.events.AliasEvent):
            raise EvaluationContractError(
                "evaluation contract YAML aliases are forbidden"
            )
        return super().compose_node(parent, index)


def _construct_mapping(
    loader: _StrictLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    loader.flatten_mapping(node)
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise EvaluationContractError("evaluation contract keys must be strings")
        if key in result:
            raise EvaluationContractError(f"evaluation contract repeats key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


@dataclass(frozen=True)
class FamilyContract:
    task: str
    module: str
    config: Mapping[str, Any]
    index_start: int
    index_stop: int
    max_new_tokens: int


@dataclass(frozen=True)
class EvaluationContract:
    path: Path
    sha256: str
    repository_root: Path
    contract_id: str
    corpus_receipt_path: str
    corpus_receipt_sha256: str
    recipe_path: str
    recipe_sha256: str
    transfer_manifest_path: str
    transfer_manifest_sha256: str
    record_manifest_path: str
    record_manifest_sha256: str
    source_commit: str
    source_tree: str
    source_stage_receipt_path: str
    source_stage_receipt_sha256: str
    source_relative_path: str
    runtime_lock: Mapping[str, str]
    families: tuple[FamilyContract, ...]
    accepted_items_per_family: int
    max_record_tokens: int
    skip_reasons: tuple[str, ...]
    scorer_id: str
    model_visible_fields: tuple[str, ...]
    sealed_gold_fields: tuple[str, ...]
    no_inspection_attestation: Mapping[str, bool]
    evaluator_code: tuple[tuple[str, str], ...]
    aws_boundary_path: str
    aws_boundary_sha256: str
    aws_region: str
    evaluator_role_arn: str
    signer_key_alias: str
    sealed_gold_kms_key_alias: str
    storage_bucket: str
    storage_prefix: str
    authority_record_key: str
    authority_signature_key: str
    activation_key: str

    @property
    def family_names(self) -> tuple[str, ...]:
        return tuple(family.task for family in self.families)

    @property
    def total_items(self) -> int:
        return len(self.families) * self.accepted_items_per_family

    @property
    def evaluator_code_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                [
                    {"path": path, "sha256": digest}
                    for path, digest in self.evaluator_code
                ]
            )
        ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _HEX
    )


def _sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise EvaluationContractError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvaluationContractError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationContractError(f"{label} must be a non-negative integer")
    return value


def _exact_fields(
    value: object,
    expected: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise EvaluationContractError(f"{label} fields differ")
    if any(not isinstance(key, str) for key in value):
        raise EvaluationContractError(f"{label} keys must be strings")
    return value


def _safe_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise EvaluationContractError(f"{label} path is unsafe")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise EvaluationContractError(f"{label} path is unsafe")
    if path.as_posix() != value:
        raise EvaluationContractError(f"{label} path is unsafe")
    return value


def _regular_file_bytes(path: Path, label: str, *, maximum: int) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise EvaluationContractError(f"{label} is missing") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > maximum
    ):
        raise EvaluationContractError(
            f"{label} must be a safe, singly linked regular file"
        )
    try:
        return path.read_bytes()
    except OSError as error:
        raise EvaluationContractError(f"{label} could not be read") from error


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationContractError(f"JSON repeats key: {key}")
        result[key] = value
    return result


def read_canonical_json(path: Path | str, *, label: str) -> Any:
    """Read the repository's sorted, two-space, newline-terminated JSON form."""

    source = Path(path)
    data = _regular_file_bytes(source, label, maximum=16 << 20)
    try:
        value = json.loads(
            data,
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                EvaluationContractError(
                    f"{label} contains non-finite JSON: {item}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationContractError(f"{label} is not valid UTF-8 JSON") from error
    try:
        expected = (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise EvaluationContractError(f"{label} is not canonical JSON") from error
    if data != expected:
        raise EvaluationContractError(f"{label} is not canonical JSON")
    return value


def _read_strict_json(path: Path, label: str) -> tuple[Any, bytes]:
    data = _regular_file_bytes(path, label, maximum=16 << 20)
    try:
        value = json.loads(
            data,
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                EvaluationContractError(
                    f"{label} contains non-finite JSON: {item}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationContractError(f"{label} is not valid UTF-8 JSON") from error
    return value, data


def _load_yaml_bytes(data: bytes) -> Mapping[str, Any]:
    try:
        raw = yaml.load(data.decode("utf-8"), Loader=_StrictLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise EvaluationContractError(
            "evaluation contract must be valid UTF-8 YAML"
        ) from error
    if not isinstance(raw, Mapping):
        raise EvaluationContractError("evaluation contract must be a mapping")
    return raw


def _load_yaml(path: Path) -> tuple[Mapping[str, Any], bytes]:
    data = _regular_file_bytes(path, "evaluation contract", maximum=1 << 20)
    raw = _load_yaml_bytes(data)
    return raw, data


def _require_bound_json(
    root: Path,
    relative: str,
    digest: str,
    label: str,
) -> Any:
    path = root / relative
    value, data = _read_strict_json(path, label)
    if hashlib.sha256(data).hexdigest() != digest:
        raise EvaluationContractError(f"{label} SHA-256 differs")
    return value


def _validate_corpus_receipt(
    path: Path | str,
    contract: EvaluationContract,
) -> Mapping[str, Any]:
    """Validate an injectable canonical corpus receipt without reading corpus bytes."""

    source_path = Path(path)
    raw, data = _read_strict_json(source_path, "reasoning corpus receipt")
    if data != canonical_json_bytes(raw):
        raise EvaluationContractError(
            "reasoning corpus receipt is not canonical JSON"
        )
    if hashlib.sha256(data).hexdigest() != contract.corpus_receipt_sha256:
        raise EvaluationContractError("reasoning corpus receipt SHA-256 differs")
    receipt = _exact_fields(
        raw,
        {
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
        },
        "reasoning corpus receipt",
    )
    if (
        receipt["schema_version"] != 2
        or isinstance(receipt["schema_version"], bool)
        or receipt["format"] != "memorysplit-reasoning-composite-v3"
        or receipt["contract_id"] != DATASET_CONTRACT_ID
        or receipt["recipe_sha256"] != contract.recipe_sha256
        or receipt["runtime_lock"] != dict(contract.runtime_lock)
    ):
        raise EvaluationContractError("reasoning corpus receipt identity differs")

    base = _exact_fields(
        receipt["base_corpus"],
        {
            "contract_id",
            "logical_tokens",
            "ordered_stream_sha256",
            "packed_stream_sha256",
            "receipt_sha256",
        },
        "reasoning corpus base receipt",
    )
    if (
        base["contract_id"] != "memorysplit-parallel-corpus-v2"
        or base["logical_tokens"] != 7_120_879_616
        or isinstance(base["logical_tokens"], bool)
        or _sha256(base["ordered_stream_sha256"], "base ordered stream")
        != "ae7a334cd2d73689d7ff439d9b736ebf06e5f992ae59b52ebdd32a1426bac40f"
        or _sha256(base["packed_stream_sha256"], "base packed stream")
        != "dc0134131c57ec339997f9cee9c22f14a7414200671805c63d7cd7a7a3d5738d"
        or _sha256(base["receipt_sha256"], "base receipt")
        != "65eda881f59acac356df0c3fbbb62df46d60bc013df05b72e8baacba7640b027"
    ):
        raise EvaluationContractError("reasoning corpus base receipt differs")

    composite = _exact_fields(
        receipt["composite"],
        {"ordering", "raw_target_tokens", "stream_sha256", "terminal_updates"},
        "reasoning corpus composite receipt",
    )
    streams = _exact_fields(
        composite["stream_sha256"],
        {"dense_target_weights", "packed_targets", "split90_target_weights"},
        "reasoning corpus composite streams",
    )
    expected_streams = {
        "dense_target_weights": (
            "917768b13ec169728cec51dc8294d118a113aee3c370ecd8c16ef0529f63f56e"
        ),
        "packed_targets": (
            "035ee111c329eb615c642eae9b9a7075314932ff8175e989aabb3317d6a4ef6f"
        ),
        "split90_target_weights": (
            "8a9c84c900e503d1742342b6a21092292c2968313087d0873e429b4268757144"
        ),
    }
    if (
        composite["ordering"] != "frozen-v2-prefix-then-reasoning-extension"
        or composite["raw_target_tokens"] != 8_169_455_616
        or isinstance(composite["raw_target_tokens"], bool)
        or composite["terminal_updates"] != 15_582
        or isinstance(composite["terminal_updates"], bool)
        or dict(streams) != expected_streams
    ):
        raise EvaluationContractError("reasoning corpus composite receipt differs")

    generator_artifacts = _exact_fields(
        receipt["generator_artifacts"],
        {
            "configs/reasoning-dataset-v3.json",
            "corpusgen/parallel/canonical.py",
            "corpusgen/reasoning_expansion.py",
            "corpusgen/reasoning_oracles.py",
            "train/tokenizer.py",
        },
        "reasoning corpus generator artifacts",
    )
    if (
        generator_artifacts["configs/reasoning-dataset-v3.json"]
        != contract.recipe_sha256
        or not all(_is_sha256(value) for value in generator_artifacts.values())
    ):
        raise EvaluationContractError(
            "reasoning corpus generator artifact binding differs"
        )

    source = _exact_fields(
        receipt["source"],
        {
            "reasoning_gym_tree_commitment_sha256",
            "reasoning_gym_version",
            "source_stage_receipt_sha256",
        },
        "reasoning corpus source receipt",
    )
    if (
        source["reasoning_gym_version"] != "0.1.19"
        or source["source_stage_receipt_sha256"]
        != contract.source_stage_receipt_sha256
        or not _is_sha256(source["reasoning_gym_tree_commitment_sha256"])
    ):
        raise EvaluationContractError("reasoning corpus source receipt differs")

    extension = _exact_fields(
        receipt["extension"],
        {
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
        },
        "reasoning corpus extension receipt",
    )
    record_count = _positive_int(
        extension["record_count"], "reasoning receipt record count"
    )
    packed_sha256 = _sha256(
        extension["packed_stream_sha256"], "reasoning packed stream"
    )
    shared_sha256 = _sha256(
        extension["shared_target_weights_sha256"],
        "reasoning shared target weights",
    )
    _sha256(extension["record_stream_sha256"], "reasoning record stream")
    if (
        extension["format"] != "memorysplit-reasoning-extension-v2"
        or extension["max_record_tokens"] != contract.max_record_tokens
        or isinstance(extension["max_record_tokens"], bool)
        or extension["raw_target_tokens"] != 1_048_576_000
        or isinstance(extension["raw_target_tokens"], bool)
        or record_count != 7_530_527
        or extension["target_weight_policy"]
        != "all_extension_reasoning_targets_are_internal_in_both_arms"
        or extension["targets_per_update"] != 524_288
        or isinstance(extension["targets_per_update"], bool)
        or extension["terminal_updates"] != 2_000
        or isinstance(extension["terminal_updates"], bool)
    ):
        raise EvaluationContractError("reasoning corpus extension identity differs")

    artifacts = extension["artifacts"]
    expected_artifacts = (
        (2_097_152_000, "packed/targets.bin", packed_sha256),
        (
            _MANIFEST_HEADER.size + record_count * _MANIFEST_RECORD.size,
            "records/manifest.bin",
            contract.record_manifest_sha256,
        ),
        (1_048_576_000, "sidecars/shared_target_weights.bin", shared_sha256),
    )
    if not isinstance(artifacts, list) or len(artifacts) != len(expected_artifacts):
        raise EvaluationContractError("reasoning corpus artifact count differs")
    for index, (artifact, expected) in enumerate(zip(artifacts, expected_artifacts)):
        item = _exact_fields(
            artifact,
            {"bytes", "path", "sha256"},
            f"reasoning corpus artifact {index}",
        )
        actual = (
            _positive_int(item["bytes"], f"reasoning artifact {index} bytes"),
            _safe_relative_path(
                item["path"], f"reasoning artifact {index}"
            ),
            _sha256(item["sha256"], f"reasoning artifact {index}"),
        )
        if actual != expected:
            raise EvaluationContractError(
                f"reasoning corpus artifact {index} differs"
            )

    probes = extension["probes"]
    if not isinstance(probes, Mapping) or set(probes) != set(contract.family_names):
        raise EvaluationContractError("reasoning corpus probe families differ")
    probe_fields = {
        "accepted",
        "record_sha256",
        "rejection_reason",
        "source_index",
        "token_count",
    }
    for task in contract.family_names:
        task_probes = probes[task]
        if not isinstance(task_probes, list) or len(task_probes) != 4:
            raise EvaluationContractError(f"reasoning corpus probes differ for {task}")
        for expected_index, probe in zip((0, 1, 17, 127), task_probes):
            item = _exact_fields(probe, probe_fields, f"{task} probe")
            accepted = item["accepted"]
            token_count = _nonnegative_int(item["token_count"], f"{task} probe tokens")
            if (
                not isinstance(accepted, bool)
                or item["source_index"] != expected_index
                or isinstance(item["source_index"], bool)
                or not _is_sha256(item["record_sha256"])
                or item["rejection_reason"] not in (None, *SKIP_REASONS)
                or (accepted and item["rejection_reason"] is not None)
                or (accepted and not 0 < token_count <= contract.max_record_tokens)
                or (not accepted and item["rejection_reason"] is None)
            ):
                raise EvaluationContractError(f"reasoning corpus probe differs for {task}")

    task_stats = extension["task_stats"]
    stat_fields = {
        "dataset",
        "emitted_records",
        "emitted_tokens",
        "examined_records",
        "oracle_rejections",
        "overlength_rejections",
        "source_cursor",
        "target_quota",
    }
    if not isinstance(task_stats, list) or len(task_stats) != len(contract.families):
        raise EvaluationContractError("reasoning corpus task accounting differs")
    for family, raw_stat in zip(contract.families, task_stats):
        item = _exact_fields(raw_stat, stat_fields, f"{family.task} accounting")
        if item["dataset"] != family.task:
            raise EvaluationContractError(
                f"reasoning corpus task order differs for {family.task}"
            )
        for field in stat_fields - {"dataset"}:
            _nonnegative_int(item[field], f"{family.task} {field}")
    return receipt


def _parse_families(
    raw_families: object,
    recipe_tasks: tuple[Any, ...] | None,
) -> tuple[FamilyContract, ...]:
    if not isinstance(raw_families, list) or len(raw_families) != len(FAMILY_ORDER):
        raise EvaluationContractError("evaluation family count differs")
    result: list[FamilyContract] = []
    for index, raw_family in enumerate(raw_families):
        family = _exact_fields(
            raw_family,
            {
                "config",
                "index_start",
                "index_stop",
                "max_new_tokens",
                "module",
                "task",
            },
            f"evaluation family {index}",
        )
        task = family["task"]
        module = family["module"]
        config = family["config"]
        if task != FAMILY_ORDER[index]:
            raise EvaluationContractError("evaluation family order differs")
        if (
            not isinstance(module, str)
            or not module
            or not isinstance(config, Mapping)
            or any(not isinstance(key, str) for key in config)
        ):
            raise EvaluationContractError(
                f"evaluation family {task} identity/config differs"
            )
        index_start = _positive_int(
            family["index_start"], f"{task} index start"
        )
        index_stop = _positive_int(family["index_stop"], f"{task} index stop")
        max_new_tokens = _positive_int(
            family["max_new_tokens"], f"{task} max_new_tokens"
        )
        if (
            index_start != INDEX_BASE + index * INDEX_WINDOW_SIZE
            or index_stop != index_start + INDEX_WINDOW_SIZE
            or max_new_tokens != FAMILY_MAX_NEW_TOKENS[task]
            or max_new_tokens > MAX_RECORD_TOKENS
        ):
            raise EvaluationContractError(f"evaluation family {task} window/limit differs")
        if recipe_tasks is not None:
            recipe_task = recipe_tasks[index]
            if (
                recipe_task.dataset != task
                or recipe_task.module != module
                or dict(recipe_task.config) != dict(config)
            ):
                raise EvaluationContractError(
                    f"evaluation family {task} differs from the frozen recipe"
                )
        result.append(
            FamilyContract(
                task=task,
                module=module,
                config=dict(config),
                index_start=index_start,
                index_stop=index_stop,
                max_new_tokens=max_new_tokens,
            )
        )
    return tuple(result)


def _load_evaluation_contract_from_bytes(
    data: bytes,
    *,
    path: Path | str,
    repository_root: Path | str = ROOT,
) -> EvaluationContract:
    """Parse bytes already read and authenticated by a production authority."""

    contract_path = Path(path)
    root = Path(repository_root).resolve(strict=True)
    raw = _load_yaml_bytes(data)
    top = _exact_fields(
        raw,
        {
            "authority",
            "contract_id",
            "contract_status",
            "corpus",
            "decoding",
            "evaluator_code",
            "families",
            "generation",
            "no_inspection_attestation",
            "release",
            "schema_version",
            "scientific_scope",
            "source",
        },
        "evaluation contract",
    )
    if (
        top["schema_version"] != 1
        or isinstance(top["schema_version"], bool)
        or top["contract_status"] != "frozen"
        or top["contract_id"] != CONTRACT_ID
        or top["scientific_scope"] != "prospectively_frozen_exploratory_n10"
    ):
        raise EvaluationContractError("evaluation contract identity differs")

    corpus = _exact_fields(
        top["corpus"],
        {
            "contract_id",
            "frozen_lock",
            "raw_target_tokens",
            "receipt",
            "recipe",
            "record_manifest",
            "transfer_manifest",
        },
        "evaluation corpus",
    )
    frozen_lock = _exact_fields(
        corpus["frozen_lock"], {"path", "sha256"}, "frozen corpus lock"
    )
    corpus_receipt = _exact_fields(
        corpus["receipt"], {"path", "sha256"}, "reasoning corpus receipt"
    )
    recipe = _exact_fields(corpus["recipe"], {"path", "sha256"}, "corpus recipe")
    transfer = _exact_fields(
        corpus["transfer_manifest"], {"path", "sha256"}, "transfer manifest"
    )
    records = _exact_fields(
        corpus["record_manifest"], {"path", "sha256"}, "record manifest"
    )
    frozen_path = _safe_relative_path(frozen_lock["path"], "frozen corpus lock")
    corpus_receipt_path = _safe_relative_path(
        corpus_receipt["path"], "reasoning corpus receipt"
    )
    recipe_path = _safe_relative_path(recipe["path"], "corpus recipe")
    transfer_path = _safe_relative_path(transfer["path"], "transfer manifest")
    record_path = _safe_relative_path(records["path"], "record manifest")
    corpus_receipt_sha256 = _sha256(
        corpus_receipt["sha256"], "reasoning corpus receipt"
    )
    recipe_sha256 = _sha256(recipe["sha256"], "corpus recipe")
    transfer_sha256 = _sha256(transfer["sha256"], "transfer manifest")
    record_sha256 = _sha256(records["sha256"], "record manifest")
    if (
        corpus["contract_id"] != DATASET_CONTRACT_ID
        or isinstance(corpus["raw_target_tokens"], bool)
        or corpus["raw_target_tokens"] != 8_169_455_616
        or corpus_receipt_path != _CORPUS_RECEIPT_PATH
        or corpus_receipt_sha256 != _CORPUS_RECEIPT
        or frozen_path != "artifacts/reasoning-corpus-v3/FROZEN.json"
        or _sha256(frozen_lock["sha256"], "frozen corpus lock")
        != _FROZEN_LOCK_SHA256
        or recipe_path != "configs/reasoning-dataset-v3.json"
        or recipe_sha256 != _RECIPE_SHA256
        or transfer_path != "cluster/aws/reasoning-v3-corpus-manifest.json"
        or transfer_sha256 != _TRANSFER_MANIFEST_SHA256
        or record_path != "extension/records/manifest.bin"
        or record_sha256 != _RECORD_MANIFEST_SHA256
    ):
        raise EvaluationContractError("evaluation corpus binding differs")

    source = _exact_fields(
        top["source"],
        {
            "reasoning_gym_version",
            "repository_commit",
            "repository_tree",
            "runtime_lock",
            "source_stage_receipt",
            "staged_relative_path",
        },
        "evaluation source",
    )
    source_stage_receipt = _exact_fields(
        source["source_stage_receipt"],
        {"path", "sha256"},
        "source-stage receipt",
    )
    source_stage_receipt_path = _safe_relative_path(
        source_stage_receipt["path"], "source-stage receipt"
    )
    source_relative_path = _safe_relative_path(
        source["staged_relative_path"], "Reasoning Gym source"
    )
    runtime_lock = _exact_fields(
        source["runtime_lock"], set(_RUNTIME_LOCK), "evaluation runtime lock"
    )
    if (
        source["repository_commit"] != _SOURCE_COMMIT
        or source["repository_tree"] != _SOURCE_TREE
        or source["reasoning_gym_version"] != "0.1.19"
        or _sha256(
            source_stage_receipt["sha256"],
            "source-stage receipt",
        )
        != _SOURCE_STAGE_RECEIPT_SHA256
        or source_stage_receipt_path != _SOURCE_STAGE_RECEIPT_PATH
        or source_relative_path != _SOURCE_RELATIVE_PATH
        or dict(runtime_lock) != _RUNTIME_LOCK
    ):
        raise EvaluationContractError("evaluation source binding differs")

    generation = _exact_fields(
        top["generation"],
        {
            "accepted_items_per_family",
            "family_order",
            "index_base",
            "index_window_size",
            "max_record_tokens",
            "skip_reasons",
        },
        "evaluation generation",
    )
    accepted = _positive_int(
        generation["accepted_items_per_family"],
        "accepted items per family",
    )
    index_base = _positive_int(generation["index_base"], "evaluation index base")
    window_size = _positive_int(
        generation["index_window_size"], "evaluation index window"
    )
    max_record_tokens = _positive_int(
        generation["max_record_tokens"], "evaluation record limit"
    )
    if (
        generation["family_order"] != list(FAMILY_ORDER)
        or accepted != ACCEPTED_ITEMS_PER_FAMILY
        or index_base != INDEX_BASE
        or window_size != INDEX_WINDOW_SIZE
        or max_record_tokens != MAX_RECORD_TOKENS
        or generation["skip_reasons"] != list(SKIP_REASONS)
    ):
        raise EvaluationContractError("evaluation generation constants differ")

    _require_bound_json(root, frozen_path, _FROZEN_LOCK_SHA256, "frozen corpus lock")
    _require_bound_json(root, recipe_path, recipe_sha256, "corpus recipe")
    _require_bound_json(
        root,
        transfer_path,
        transfer_sha256,
        "transfer manifest",
    )
    try:
        recipe_object = load_expansion_recipe(root / recipe_path)
    except (OSError, ValueError, RuntimeError) as error:
        raise EvaluationContractError(
            "frozen reasoning recipe validation failed"
        ) from error
    if (
        recipe_object.sha256 != recipe_sha256
        or recipe_object.max_record_tokens != max_record_tokens
        or dict(recipe_object.runtime_lock) != _RUNTIME_LOCK
    ):
        raise EvaluationContractError("frozen reasoning recipe binding differs")
    families = _parse_families(
        top["families"],
        recipe_object.tasks,
    )

    decoding = _exact_fields(
        top["decoding"],
        {
            "comparison",
            "malformed_output",
            "output_encoding",
            "output_normalization",
            "prompt_suffix",
            "scorer_id",
            "stop_token",
            "strategy",
            "temperature",
            "whitespace_policy",
        },
        "evaluation decoding",
    )
    if decoding != {
        "comparison": "exact",
        "malformed_output": "incorrect",
        "output_encoding": "UTF-8",
        "output_normalization": "NFC",
        "prompt_suffix": "Answer:",
        "scorer_id": SCORER_ID,
        "stop_token": "<|eot|>",
        "strategy": "greedy",
        "temperature": 0,
        "whitespace_policy": "strip_outer_only",
    } or isinstance(decoding["temperature"], bool):
        raise EvaluationContractError("evaluation decoding contract differs")

    release = _exact_fields(
        top["release"],
        {
            "format",
            "model_visible_authorization",
            "model_visible_fields",
            "sealed_gold_authorization",
            "sealed_gold_fields",
        },
        "evaluation release",
    )
    model_fields = (
        "item_id",
        "task",
        "source_index",
        "prompt",
        "max_new_tokens",
        "scorer_id",
    )
    sealed_fields = (
        "item_id",
        "task",
        "source_index",
        "canonical_answer",
        "oracle_replay",
    )
    if (
        release["format"] != "memorysplit-reasoning-v3-eval-release-v1"
        or release["model_visible_authorization"] != "trainer_and_evaluator"
        or release["sealed_gold_authorization"] != "evaluator_only"
        or release["model_visible_fields"] != list(model_fields)
        or release["sealed_gold_fields"] != list(sealed_fields)
    ):
        raise EvaluationContractError("evaluation release contract differs")

    attestation = _exact_fields(
        top["no_inspection_attestation"],
        {
            "evaluation_items_or_gold_inspected",
            "protected_training_started",
            "selection_depends_on_observed_results",
        },
        "no-inspection attestation",
    )
    expected_attestation = {
        "evaluation_items_or_gold_inspected": False,
        "protected_training_started": False,
        "selection_depends_on_observed_results": False,
    }
    if dict(attestation) != expected_attestation or any(
        not isinstance(value, bool) for value in attestation.values()
    ):
        raise EvaluationContractError("no-inspection attestation differs")

    authority = _exact_fields(
        top["authority"],
        {
            "activation_key",
            "authority_record_key",
            "authority_signature_key",
            "aws_boundary_path",
            "aws_boundary_sha256",
            "aws_region",
            "evaluator_role_arn",
            "sealed_gold_kms_key_alias",
            "signer_key_alias",
            "storage_bucket",
            "storage_prefix",
        },
        "evaluation authority",
    )
    aws_boundary_path = _safe_relative_path(
        authority["aws_boundary_path"],
        "AWS evaluator boundary",
    )
    aws_boundary_sha256 = _sha256(
        authority["aws_boundary_sha256"],
        "AWS evaluator boundary",
    )
    if authority != {
        "activation_key": ACTIVATION_KEY,
        "authority_record_key": AUTHORITY_RECORD_KEY,
        "authority_signature_key": AUTHORITY_SIGNATURE_KEY,
        "aws_boundary_path": AWS_BOUNDARY_CONFIG_PATH.relative_to(ROOT).as_posix(),
        "aws_boundary_sha256": aws_boundary_sha256,
        "aws_region": AWS_REGION,
        "evaluator_role_arn": EVALUATOR_ROLE_ARN,
        "sealed_gold_kms_key_alias": SEALED_GOLD_KMS_KEY_ALIAS,
        "signer_key_alias": SIGNER_KEY_ALIAS,
        "storage_bucket": STORAGE_BUCKET,
        "storage_prefix": STORAGE_PREFIX,
    }:
        raise EvaluationContractError("evaluation authority differs")
    aws_boundary_bytes = _regular_file_bytes(
        root / aws_boundary_path,
        "AWS evaluator boundary",
        maximum=1 << 20,
    )
    if hashlib.sha256(aws_boundary_bytes).hexdigest() != aws_boundary_sha256:
        raise EvaluationContractError("AWS evaluator boundary SHA-256 differs")
    try:
        _parse_aws_boundary_record(aws_boundary_bytes)
    except Exception as error:
        raise EvaluationContractError(
            "AWS evaluator boundary policy differs"
        ) from error

    raw_code = top["evaluator_code"]
    if not isinstance(raw_code, list) or len(raw_code) != len(_EVALUATOR_PATHS):
        raise EvaluationContractError("evaluator code binding count differs")
    code_bindings: list[tuple[str, str]] = []
    for index, raw_binding in enumerate(raw_code):
        binding = _exact_fields(
            raw_binding,
            {"path", "sha256"},
            f"evaluator code binding {index}",
        )
        relative = _safe_relative_path(binding["path"], "evaluator code")
        digest = _sha256(binding["sha256"], "evaluator code")
        if relative != _EVALUATOR_PATHS[index]:
            raise EvaluationContractError("evaluator code path order differs")
        payload = _regular_file_bytes(
            root / relative,
            f"evaluator code {relative}",
            maximum=1 << 20,
        )
        if hashlib.sha256(payload).hexdigest() != digest:
            raise EvaluationContractError(
                f"evaluator code SHA-256 differs: {relative}"
            )
        code_bindings.append((relative, digest))

    return EvaluationContract(
        path=contract_path.resolve(strict=True),
        sha256=hashlib.sha256(data).hexdigest(),
        repository_root=root,
        contract_id=CONTRACT_ID,
        corpus_receipt_path=corpus_receipt_path,
        corpus_receipt_sha256=corpus_receipt_sha256,
        recipe_path=recipe_path,
        recipe_sha256=recipe_sha256,
        transfer_manifest_path=transfer_path,
        transfer_manifest_sha256=transfer_sha256,
        record_manifest_path=record_path,
        record_manifest_sha256=record_sha256,
        source_commit=_SOURCE_COMMIT,
        source_tree=_SOURCE_TREE,
        source_stage_receipt_path=source_stage_receipt_path,
        source_stage_receipt_sha256=_SOURCE_STAGE_RECEIPT_SHA256,
        source_relative_path=source_relative_path,
        runtime_lock=dict(runtime_lock),
        families=families,
        accepted_items_per_family=accepted,
        max_record_tokens=max_record_tokens,
        skip_reasons=SKIP_REASONS,
        scorer_id=SCORER_ID,
        model_visible_fields=model_fields,
        sealed_gold_fields=sealed_fields,
        no_inspection_attestation=expected_attestation,
        evaluator_code=tuple(code_bindings),
        aws_boundary_path=aws_boundary_path,
        aws_boundary_sha256=aws_boundary_sha256,
        aws_region=AWS_REGION,
        evaluator_role_arn=EVALUATOR_ROLE_ARN,
        signer_key_alias=SIGNER_KEY_ALIAS,
        sealed_gold_kms_key_alias=SEALED_GOLD_KMS_KEY_ALIAS,
        storage_bucket=STORAGE_BUCKET,
        storage_prefix=STORAGE_PREFIX,
        authority_record_key=AUTHORITY_RECORD_KEY,
        authority_signature_key=AUTHORITY_SIGNATURE_KEY,
        activation_key=ACTIVATION_KEY,
    )


def load_evaluation_contract(
    path: Path | str = DEFAULT_CONTRACT_PATH,
    *,
    repository_root: Path | str = ROOT,
) -> EvaluationContract:
    """Load the semantic contract for private validation and test fixtures."""

    contract_path = Path(path)
    data = _regular_file_bytes(
        contract_path,
        "evaluation contract",
        maximum=1 << 20,
    )
    return _load_evaluation_contract_from_bytes(
        data,
        path=contract_path,
        repository_root=repository_root,
    )
