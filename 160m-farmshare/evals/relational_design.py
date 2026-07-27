"""Blinded, prospective Gate-5 design analysis.

Arm-label commitment and outcome loading are deliberately separate APIs.  The
only supported loader accepts a persisted commitment and invokes the caller's
outcome loader after revalidating that commitment on disk.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from evals.relational_contracts import (
    EvalRow,
    _rename_directory_noreplace_between,
    canonical_json_bytes,
    validate_eval_rows,
)
from evals.relational_stats import (
    BOOTSTRAP_VERSION,
    CONFIRMATORY_SEEDS,
    FROZEN_N_BOOT,
    FROZEN_PERCENTILE_INDICES,
    PERCENTILE_CONVENTION,
    _build_panel,
    _point_seed_deltas,
)


DESIGN_SIMULATION_SEED = 202607225
DESIGN_SIMULATION_VERSION = "prospective-gate5-parametric-v1"
DESIGN_EFFECT = 0.02
DESIGN_STUDIES = 10_000
DESIGN_PAIRS = 5
DEVELOPMENT_PAIR_COUNT = 3
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SOURCE_ARMS = ("split", "dense")
_DISPLAY_LABELS = ("arm_a", "arm_b")


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validate_hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_int(
    value: object,
    name: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} is out of range")
    return value


def _validate_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _contains_parent_reference(path: Path) -> bool:
    return any(part == ".." for part in path.parts)


def _require_regular_parent(path: Path) -> Path:
    if _contains_parent_reference(path):
        raise ValueError("output path cannot contain path traversal")
    parent = path.parent
    if parent.is_symlink():
        raise ValueError("output path cannot use a symlink parent")
    if not parent.is_dir():
        raise ValueError("output parent must be a regular directory")
    if parent.resolve(strict=True) != parent.absolute():
        raise ValueError("output path cannot traverse symlink components")
    return parent


def _require_regular_input(path: Path, name: str) -> Path:
    if _contains_parent_reference(path):
        raise ValueError(f"{name} cannot contain path traversal")
    absolute = path.absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular file")
    if path.resolve(strict=True) != absolute:
        raise ValueError(f"{name} cannot traverse symlink components")
    return absolute


def _write_synced(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_directory(destination: Path, files: Mapping[str, bytes]) -> None:
    parent = _require_regular_parent(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"output already exists: {destination}")
    stage = Path(
        tempfile.mkdtemp(
            dir=parent,
            prefix=f".{destination.name}.stage-",
        )
    )
    try:
        for name, content in sorted(files.items()):
            if (
                not name
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
            ):
                raise ValueError("invalid staged commitment filename")
            _write_synced(stage / name, content)
        stage_fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            _rename_directory_noreplace_between(
                parent_fd,
                stage.name,
                parent_fd,
                destination.name,
            )
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


@dataclass(frozen=True)
class ProtectedIdentityRegistry:
    seeds: frozenset[int]
    qids: frozenset[str]
    pair_ids: frozenset[str]
    world_ids: frozenset[int]
    relation_path_hashes: frozenset[str]
    template_ids: frozenset[str]
    cluster_ids: frozenset[str]
    provenance_ids: frozenset[str]
    paths: frozenset[Path]

    @classmethod
    def empty(cls) -> "ProtectedIdentityRegistry":
        return cls(
            seeds=frozenset(CONFIRMATORY_SEEDS),
            qids=frozenset(),
            pair_ids=frozenset(),
            world_ids=frozenset(),
            relation_path_hashes=frozenset(),
            template_ids=frozenset(),
            cluster_ids=frozenset(),
            provenance_ids=frozenset(),
            paths=frozenset(),
        )

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[EvalRow],
        *,
        paths: Sequence[str | Path] = (),
    ) -> "ProtectedIdentityRegistry":
        if any(not isinstance(row, EvalRow) for row in rows):
            raise TypeError(
                "protected registry rows must be validated EvalRow values"
            )
        materialized = validate_eval_rows(
            EvalRow.from_dict(row.to_dict()) for row in rows
        )
        normalized_paths = frozenset(Path(path).absolute() for path in paths)
        return cls(
            seeds=frozenset(row.seed for row in materialized)
            | frozenset(CONFIRMATORY_SEEDS),
            qids=frozenset(row.qid for row in materialized),
            pair_ids=frozenset(row.pair_id for row in materialized),
            world_ids=frozenset(row.world_id for row in materialized),
            relation_path_hashes=frozenset(
                row.relation_path_hash for row in materialized
            ),
            template_ids=frozenset(row.template_id for row in materialized),
            cluster_ids=frozenset(row.cluster_id for row in materialized),
            provenance_ids=frozenset(
                row.provenance_id for row in materialized
            ),
            paths=normalized_paths,
        )


def _paths_overlap(
    path: Path,
    protected: ProtectedIdentityRegistry,
) -> bool:
    absolute = path.absolute()
    return any(
        absolute == item
        or item in absolute.parents
        or absolute in item.parents
        for item in protected.paths
    )


@dataclass(frozen=True)
class PersistedPermutationCommitment:
    path: Path
    planned_input_paths: tuple[Path, ...]
    planned_input_arms: tuple[str, ...]
    permutation_commitment: str
    commitment_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).absolute())
        object.__setattr__(
            self,
            "planned_input_paths",
            tuple(Path(path).absolute() for path in self.planned_input_paths),
        )
        object.__setattr__(
            self, "planned_input_arms", tuple(self.planned_input_arms)
        )
        _validate_hash(
            self.permutation_commitment, "permutation commitment"
        )
        _validate_hash(self.commitment_sha256, "commitment SHA-256")


def _permutation_secret(
    *,
    rng_seed: int,
    planned_ids: Sequence[str],
) -> dict[str, Any]:
    seed = _validate_int(rng_seed, "permutation rng seed")
    seed_material = {
        "protocol": "global-arm-label-permutation-v1",
        "rng_seed": seed,
        "planned_input_ids": list(planned_ids),
    }
    digest = hashlib.sha256(canonical_json_bytes(seed_material)).digest()
    source = _SOURCE_ARMS if digest[0] % 2 == 0 else tuple(reversed(_SOURCE_ARMS))
    nonce = hashlib.sha256(
        canonical_json_bytes(["permutation-nonce", seed_material])
    ).hexdigest()
    return {
        "record_type": "arm_permutation_key",
        "schema_version": 1,
        "protocol_version": "global-arm-label-permutation-v1",
        "mapping": dict(zip(_DISPLAY_LABELS, source, strict=True)),
        "nonce": nonce,
    }


def commit_arm_label_permutation(
    output: str | Path,
    *,
    planned_inputs: Mapping[str, str | Path],
    rng_seed: int,
    protected: ProtectedIdentityRegistry | None = None,
) -> PersistedPermutationCommitment:
    """Persist the global arm permutation before any outcome loader runs."""

    destination = Path(output)
    _require_regular_parent(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"commitment output already exists: {destination}")
    if not isinstance(planned_inputs, Mapping) or set(planned_inputs) != set(
        _SOURCE_ARMS
    ):
        raise ValueError("planned inputs must contain exactly split and dense")
    registry = (
        ProtectedIdentityRegistry.empty()
        if protected is None
        else protected
    )
    if not isinstance(registry, ProtectedIdentityRegistry):
        raise TypeError("protected identities must use the strict registry")
    paths = tuple(
        _require_regular_input(
            Path(planned_inputs[arm]),
            f"planned {arm} input",
        )
        for arm in _SOURCE_ARMS
    )
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("planned input filenames must be unique")
    if any(_paths_overlap(path, registry) for path in paths):
        raise ValueError("planned development path overlaps protected paths")
    planned_ids = tuple(path.name for path in paths)
    secret = _permutation_secret(
        rng_seed=rng_seed,
        planned_ids=planned_ids,
    )
    secret_content = canonical_json_bytes(secret)
    permutation_commitment = hashlib.sha256(secret_content).hexdigest()
    public = {
        "record_type": "arm_permutation_commitment",
        "schema_version": 1,
        "protocol_version": "global-arm-label-permutation-v1",
        "state": "committed_before_outcomes",
        "planned_inputs": [
            {"source_arm": arm, "path_id": path.name}
            for arm, path in zip(_SOURCE_ARMS, paths, strict=True)
        ],
        "display_labels": list(_DISPLAY_LABELS),
        "permutation_commitment_sha256": permutation_commitment,
        "permutation_key_sha256": hashlib.sha256(secret_content).hexdigest(),
    }
    public_content = canonical_json_bytes(public)
    _publish_directory(
        destination,
        {
            "commitment.json": public_content,
            "permutation-key.json": secret_content,
        },
    )
    return PersistedPermutationCommitment(
        path=destination,
        planned_input_paths=paths,
        planned_input_arms=_SOURCE_ARMS,
        permutation_commitment=permutation_commitment,
        commitment_sha256=hashlib.sha256(public_content).hexdigest(),
    )


def _read_commitment(
    commitment: PersistedPermutationCommitment,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(commitment, PersistedPermutationCommitment):
        raise TypeError("a persisted permutation commitment is required")
    root = commitment.path
    if root.is_symlink() or not root.is_dir():
        raise ValueError("persisted permutation commitment is missing")
    if root.resolve(strict=True) != root.absolute():
        raise ValueError("commitment path traverses a symlink")
    if {entry.name for entry in root.iterdir()} != {
        "commitment.json",
        "permutation-key.json",
    }:
        raise ValueError("persisted commitment file set is not exact")
    public_path = root / "commitment.json"
    key_path = root / "permutation-key.json"
    for path in (public_path, key_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("commitment contains a non-regular file")
    public_content = public_path.read_bytes()
    key_content = key_path.read_bytes()
    try:
        public = json.loads(public_content)
        secret = json.loads(key_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("persisted commitment JSON is invalid") from exc
    if (
        public_content != canonical_json_bytes(public)
        or key_content != canonical_json_bytes(secret)
    ):
        raise ValueError("persisted commitment is not canonical")
    expected_public_fields = {
        "record_type",
        "schema_version",
        "protocol_version",
        "state",
        "planned_inputs",
        "display_labels",
        "permutation_commitment_sha256",
        "permutation_key_sha256",
    }
    expected_secret_fields = {
        "record_type",
        "schema_version",
        "protocol_version",
        "mapping",
        "nonce",
    }
    if (
        not isinstance(public, dict)
        or set(public) != expected_public_fields
        or public["record_type"] != "arm_permutation_commitment"
        or public["schema_version"] != 1
        or public["protocol_version"] != "global-arm-label-permutation-v1"
        or public["state"] != "committed_before_outcomes"
        or public["display_labels"] != list(_DISPLAY_LABELS)
        or not isinstance(secret, dict)
        or set(secret) != expected_secret_fields
        or secret["record_type"] != "arm_permutation_key"
        or secret["schema_version"] != 1
        or secret["protocol_version"] != "global-arm-label-permutation-v1"
        or not isinstance(secret["mapping"], dict)
        or set(secret["mapping"]) != set(_DISPLAY_LABELS)
        or set(secret["mapping"].values()) != set(_SOURCE_ARMS)
    ):
        raise ValueError("persisted commitment contract is invalid")
    key_hash = hashlib.sha256(key_content).hexdigest()
    if (
        public["permutation_commitment_sha256"] != key_hash
        or public["permutation_key_sha256"] != key_hash
        or commitment.permutation_commitment != key_hash
        or commitment.commitment_sha256
        != hashlib.sha256(public_content).hexdigest()
    ):
        raise ValueError("persisted permutation commitment hash mismatch")
    planned = public["planned_inputs"]
    if (
        not isinstance(planned, list)
        or len(planned) != 2
        or planned
        != [
            {"source_arm": arm, "path_id": path.name}
            for arm, path in zip(
                commitment.planned_input_arms,
                commitment.planned_input_paths,
                strict=True,
            )
        ]
    ):
        raise ValueError("persisted commitment planned inputs changed")
    return public, secret


def open_arm_label_permutation(
    path: str | Path,
    *,
    planned_inputs: Mapping[str, str | Path],
) -> PersistedPermutationCommitment:
    """Open and verify an existing commitment in a later process."""

    root = Path(path).absolute()
    if not isinstance(planned_inputs, Mapping) or set(planned_inputs) != set(
        _SOURCE_ARMS
    ):
        raise ValueError("planned inputs must contain exactly split and dense")
    paths = tuple(
        _require_regular_input(
            Path(planned_inputs[arm]),
            f"planned {arm} input",
        )
        for arm in _SOURCE_ARMS
    )
    public_path = root / "commitment.json"
    key_path = root / "permutation-key.json"
    if not public_path.is_file() or not key_path.is_file():
        raise ValueError("persisted permutation commitment is missing")
    public_content = public_path.read_bytes()
    key_content = key_path.read_bytes()
    commitment = PersistedPermutationCommitment(
        path=root,
        planned_input_paths=paths,
        planned_input_arms=_SOURCE_ARMS,
        permutation_commitment=hashlib.sha256(key_content).hexdigest(),
        commitment_sha256=hashlib.sha256(public_content).hexdigest(),
    )
    _read_commitment(commitment)
    return commitment


def _rows_hash(rows: Sequence[EvalRow]) -> str:
    digest = hashlib.sha256()
    for row in sorted(
        rows,
        key=lambda value: (
            value.checkpoint_sha256,
            value.paired_join_key(),
        ),
    ):
        digest.update(canonical_json_bytes(row))
    return digest.hexdigest()


@dataclass(frozen=True)
class BlindedDevelopmentRows:
    rows_by_label: Mapping[str, tuple[EvalRow, ...]]
    source_arms: Mapping[str, str]
    source_paths: Mapping[str, Path]
    permutation_commitment: str
    commitment_sha256: str
    blinded_input_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rows_by_label",
            MappingProxyType(
                {
                    label: tuple(rows)
                    for label, rows in self.rows_by_label.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "source_arms",
            MappingProxyType(dict(self.source_arms)),
        )
        object.__setattr__(
            self,
            "source_paths",
            MappingProxyType(
                {
                    label: Path(path).absolute()
                    for label, path in self.source_paths.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "blinded_input_hashes",
            MappingProxyType(dict(self.blinded_input_hashes)),
        )


def load_blinded_development(
    commitment: PersistedPermutationCommitment,
    loader: Callable[[Path], Sequence[EvalRow]],
) -> BlindedDevelopmentRows:
    """Load outcomes only after verifying the persisted global permutation."""

    _public, secret = _read_commitment(commitment)
    if not callable(loader):
        raise TypeError("development outcome loader must be callable")
    loaded_by_arm: dict[str, tuple[EvalRow, ...]] = {}
    path_by_arm = dict(
        zip(
            commitment.planned_input_arms,
            commitment.planned_input_paths,
            strict=True,
        )
    )
    # The invocation of loader is intentionally below commitment validation.
    for arm in _SOURCE_ARMS:
        path = _require_regular_input(path_by_arm[arm], f"{arm} input")
        raw_rows = loader(path)
        if isinstance(raw_rows, (str, bytes)) or not isinstance(
            raw_rows, Sequence
        ):
            raise TypeError("development loader must return EvalRow sequences")
        if any(not isinstance(row, EvalRow) for row in raw_rows):
            raise TypeError(
                "development outcomes must be strict EvalRow artifacts"
            )
        loaded_by_arm[arm] = tuple(
            EvalRow.from_dict(row.to_dict()) for row in raw_rows
        )
    mapping = dict(secret["mapping"])
    rows_by_label = {
        label: loaded_by_arm[arm] for label, arm in mapping.items()
    }
    source_paths = {
        label: path_by_arm[arm] for label, arm in mapping.items()
    }
    return BlindedDevelopmentRows(
        rows_by_label=rows_by_label,
        source_arms=mapping,
        source_paths=source_paths,
        permutation_commitment=commitment.permutation_commitment,
        commitment_sha256=commitment.commitment_sha256,
        blinded_input_hashes={
            label: _rows_hash(rows)
            for label, rows in rows_by_label.items()
        },
    )


def _development_seed_deltas(
    development: BlindedDevelopmentRows,
    protected: ProtectedIdentityRegistry,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if not isinstance(development, BlindedDevelopmentRows):
        raise TypeError(
            "prospective design requires temporally blinded development rows"
        )
    if not isinstance(protected, ProtectedIdentityRegistry):
        raise TypeError("protected identities must use the strict registry")
    if (
        set(development.rows_by_label) != set(_DISPLAY_LABELS)
        or set(development.source_arms) != set(_DISPLAY_LABELS)
        or set(development.source_arms.values()) != set(_SOURCE_ARMS)
    ):
        raise ValueError("blinded development arm labels are invalid")
    all_rows = tuple(
        row
        for label in _DISPLAY_LABELS
        for row in development.rows_by_label[label]
    )
    seeds = tuple(sorted({row.seed for row in all_rows}))
    if len(seeds) != DEVELOPMENT_PAIR_COUNT:
        raise ValueError(
            "Gate 5 requires exactly three development seed pairs"
        )
    if set(seeds) & protected.seeds:
        raise ValueError("development seed overlaps protected seeds")
    if any(
        not row.provenance_id.startswith("development")
        for row in all_rows
    ):
        raise ValueError(
            "development rows must use the disjoint development namespace"
        )
    axes = {
        "ID": (
            {row.qid for row in all_rows}
            | {row.pair_id for row in all_rows}
            | {row.provenance_id for row in all_rows}
        ),
        "world": {row.world_id for row in all_rows},
        "path": {row.relation_path_hash for row in all_rows},
        "template": {row.template_id for row in all_rows},
        "cluster": {row.cluster_id for row in all_rows},
    }
    protected_axes = {
        "ID": (
            set(protected.qids)
            | set(protected.pair_ids)
            | set(protected.provenance_ids)
        ),
        "world": set(protected.world_ids),
        "path": set(protected.relation_path_hashes),
        "template": set(protected.template_ids),
        "cluster": set(protected.cluster_ids),
    }
    for name in axes:
        if axes[name] & protected_axes[name]:
            raise ValueError(
                f"development {name} overlaps protected identities"
            )
    if any(
        _paths_overlap(path, protected)
        for path in development.source_paths.values()
    ):
        raise ValueError("development filesystem path overlaps protected paths")
    panel = _build_panel(development.rows_by_label, seeds=seeds)
    weights = {"arm_a": 1.0, "arm_b": -1.0}
    return seeds, _point_seed_deltas(panel, weights)


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return center - radius, center + radius


@dataclass(frozen=True)
class DesignPowerSimulation:
    development_seeds: tuple[int, ...]
    blinded_seed_deltas: tuple[float, ...]
    variance_estimate: float
    effect: float
    pairs: int
    studies: int
    successes: int
    power: float
    power_ci_lo: float
    power_ci_hi: float
    passed: bool
    simulation_seed: int
    simulation_version: str


def simulate_design_power(
    development_rows: BlindedDevelopmentRows,
    *,
    effect: float = DESIGN_EFFECT,
    studies: int = DESIGN_STUDIES,
    pairs: int = DESIGN_PAIRS,
    rng_seed: int = DESIGN_SIMULATION_SEED,
    protected: ProtectedIdentityRegistry | None = None,
) -> DesignPowerSimulation:
    """Simulate exactly 10,000 frozen five-pair studies for Gate 5."""

    effect_value = _validate_number(effect, "effect")
    if effect_value != DESIGN_EFFECT:
        raise ValueError("Gate 5 effect must remain frozen at additive 0.02")
    studies_value = _validate_int(studies, "studies", minimum=1)
    if studies_value != DESIGN_STUDIES:
        raise ValueError("Gate 5 requires exactly 10,000 studies")
    pairs_value = _validate_int(pairs, "pairs", minimum=1)
    if pairs_value != DESIGN_PAIRS:
        raise ValueError("Gate 5 studies require exactly five seed pairs")
    random_seed = _validate_int(rng_seed, "simulation seed")
    if random_seed != DESIGN_SIMULATION_SEED:
        raise ValueError("Gate 5 simulation seed is frozen")
    registry = (
        ProtectedIdentityRegistry.empty()
        if protected is None
        else protected
    )
    seeds, seed_deltas = _development_seed_deltas(
        development_rows, registry
    )
    mean = sum(seed_deltas) / len(seed_deltas)
    variance = sum((value - mean) ** 2 for value in seed_deltas) / (
        len(seed_deltas) - 1
    )
    variance = max(0.0, variance)
    rng = np.random.Generator(np.random.PCG64(random_seed))
    planned = rng.normal(
        loc=effect_value,
        scale=math.sqrt(variance),
        size=(studies_value, pairs_value),
    )
    indices = rng.integers(
        0,
        pairs_value,
        size=(FROZEN_N_BOOT, pairs_value),
    )
    multiplicities = np.zeros(
        (FROZEN_N_BOOT, pairs_value),
        dtype=np.int8,
    )
    rows = np.repeat(np.arange(FROZEN_N_BOOT), pairs_value)
    np.add.at(multiplicities, (rows, indices.reshape(-1)), 1)
    lower_index = FROZEN_PERCENTILE_INDICES[0]
    successes = 0
    for start in range(0, studies_value, 100):
        block = planned[start : start + 100]
        bootstrap_means = (
            block @ multiplicities.astype(np.float64).T
        ) / pairs_value
        bootstrap_means.sort(axis=1)
        successes += int(
            np.count_nonzero(bootstrap_means[:, lower_index] > 0.0)
        )
    power = successes / studies_value
    power_low, power_high = _wilson_interval(successes, studies_value)
    return DesignPowerSimulation(
        development_seeds=seeds,
        blinded_seed_deltas=seed_deltas,
        variance_estimate=variance,
        effect=effect_value,
        pairs=pairs_value,
        studies=studies_value,
        successes=successes,
        power=power,
        power_ci_lo=power_low,
        power_ci_hi=power_high,
        passed=power >= 0.80,
        simulation_seed=random_seed,
        simulation_version=DESIGN_SIMULATION_VERSION,
    )


_RECEIPT_FIELDS = {
    "record_type",
    "schema_version",
    "gate",
    "permutation_commitment",
    "commitment_sha256",
    "blinded_input_hashes",
    "development_seeds",
    "blinded_seed_deltas",
    "variance_estimate",
    "effect",
    "pairs",
    "studies",
    "successes",
    "power",
    "power_ci_lo",
    "power_ci_hi",
    "passed",
    "simulation_seed",
    "simulation_version",
    "estimator_version",
    "bootstrap_replicates",
    "percentile_convention",
    "percentile_indices",
    "decision_sha256",
    "receipt_sha256",
}


@dataclass(frozen=True)
class DesignPowerReceipt:
    permutation_commitment: str
    commitment_sha256: str
    blinded_input_hashes: Mapping[str, str]
    development_seeds: tuple[int, ...]
    blinded_seed_deltas: tuple[float, ...]
    variance_estimate: float
    effect: float
    pairs: int
    studies: int
    successes: int
    power: float
    power_ci_lo: float
    power_ci_hi: float
    passed: bool
    simulation_seed: int
    simulation_version: str
    decision_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "blinded_input_hashes",
            MappingProxyType(dict(self.blinded_input_hashes)),
        )
        object.__setattr__(
            self, "development_seeds", tuple(self.development_seeds)
        )
        object.__setattr__(
            self, "blinded_seed_deltas", tuple(self.blinded_seed_deltas)
        )

    def _base_dict(self) -> dict[str, Any]:
        return {
            "record_type": "gate5_power_receipt",
            "schema_version": 1,
            "gate": 5,
            "permutation_commitment": self.permutation_commitment,
            "commitment_sha256": self.commitment_sha256,
            "blinded_input_hashes": {
                label: self.blinded_input_hashes[label]
                for label in sorted(self.blinded_input_hashes)
            },
            "development_seeds": list(self.development_seeds),
            "blinded_seed_deltas": list(self.blinded_seed_deltas),
            "variance_estimate": self.variance_estimate,
            "effect": self.effect,
            "pairs": self.pairs,
            "studies": self.studies,
            "successes": self.successes,
            "power": self.power,
            "power_ci_lo": self.power_ci_lo,
            "power_ci_hi": self.power_ci_hi,
            "passed": self.passed,
            "simulation_seed": self.simulation_seed,
            "simulation_version": self.simulation_version,
            "estimator_version": BOOTSTRAP_VERSION,
            "bootstrap_replicates": FROZEN_N_BOOT,
            "percentile_convention": PERCENTILE_CONVENTION,
            "percentile_indices": list(FROZEN_PERCENTILE_INDICES),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._base_dict(),
            "decision_sha256": self.decision_sha256,
            "receipt_sha256": self.receipt_sha256,
        }


def _make_receipt(
    development: BlindedDevelopmentRows,
    simulation: DesignPowerSimulation,
) -> DesignPowerReceipt:
    provisional = DesignPowerReceipt(
        permutation_commitment=development.permutation_commitment,
        commitment_sha256=development.commitment_sha256,
        blinded_input_hashes=development.blinded_input_hashes,
        development_seeds=simulation.development_seeds,
        blinded_seed_deltas=simulation.blinded_seed_deltas,
        variance_estimate=simulation.variance_estimate,
        effect=simulation.effect,
        pairs=simulation.pairs,
        studies=simulation.studies,
        successes=simulation.successes,
        power=simulation.power,
        power_ci_lo=simulation.power_ci_lo,
        power_ci_hi=simulation.power_ci_hi,
        passed=simulation.passed,
        simulation_seed=simulation.simulation_seed,
        simulation_version=simulation.simulation_version,
        decision_sha256="0" * 64,
        receipt_sha256="0" * 64,
    )
    base = provisional._base_dict()
    decision = {
        "record_type": "gate5_decision",
        "schema_version": 1,
        "gate": 5,
        "passed": simulation.passed,
        "successes": simulation.successes,
        "studies": simulation.studies,
        "power": simulation.power,
        "threshold": 0.80,
        "effect": simulation.effect,
        "pairs": simulation.pairs,
        "permutation_commitment": development.permutation_commitment,
        "blinded_input_hashes": dict(development.blinded_input_hashes),
        "simulation_seed": simulation.simulation_seed,
        "simulation_version": simulation.simulation_version,
    }
    decision_hash = _hash(decision)
    receipt_without_hash = {
        **base,
        "decision_sha256": decision_hash,
    }
    return DesignPowerReceipt(
        **{
            field: getattr(provisional, field)
            for field in (
                "permutation_commitment",
                "commitment_sha256",
                "blinded_input_hashes",
                "development_seeds",
                "blinded_seed_deltas",
                "variance_estimate",
                "effect",
                "pairs",
                "studies",
                "successes",
                "power",
                "power_ci_lo",
                "power_ci_hi",
                "passed",
                "simulation_seed",
                "simulation_version",
            )
        },
        decision_sha256=decision_hash,
        receipt_sha256=_hash(receipt_without_hash),
    )


def run_prospective_design(
    development: BlindedDevelopmentRows,
    *,
    protected: ProtectedIdentityRegistry,
) -> DesignPowerReceipt:
    simulation = simulate_design_power(
        development,
        protected=protected,
    )
    return _make_receipt(development, simulation)


def validate_design_receipt(
    raw: Mapping[str, Any],
) -> DesignPowerReceipt:
    if not isinstance(raw, Mapping) or set(raw) != _RECEIPT_FIELDS:
        raise ValueError("Gate 5 receipt fields are not exact")
    if (
        raw["record_type"] != "gate5_power_receipt"
        or raw["schema_version"] != 1
        or raw["gate"] != 5
        or raw["estimator_version"] != BOOTSTRAP_VERSION
        or raw["bootstrap_replicates"] != FROZEN_N_BOOT
        or raw["percentile_convention"] != PERCENTILE_CONVENTION
        or raw["percentile_indices"] != list(FROZEN_PERCENTILE_INDICES)
    ):
        raise ValueError("Gate 5 receipt frozen protocol mismatch")
    receipt = DesignPowerReceipt(
        permutation_commitment=_validate_hash(
            raw["permutation_commitment"], "permutation commitment"
        ),
        commitment_sha256=_validate_hash(
            raw["commitment_sha256"], "commitment SHA-256"
        ),
        blinded_input_hashes={
            label: _validate_hash(value, f"blinded input {label}")
            for label, value in raw["blinded_input_hashes"].items()
        },
        development_seeds=tuple(raw["development_seeds"]),
        blinded_seed_deltas=tuple(raw["blinded_seed_deltas"]),
        variance_estimate=raw["variance_estimate"],
        effect=raw["effect"],
        pairs=raw["pairs"],
        studies=raw["studies"],
        successes=raw["successes"],
        power=raw["power"],
        power_ci_lo=raw["power_ci_lo"],
        power_ci_hi=raw["power_ci_hi"],
        passed=raw["passed"],
        simulation_seed=raw["simulation_seed"],
        simulation_version=raw["simulation_version"],
        decision_sha256=_validate_hash(
            raw["decision_sha256"], "decision SHA-256"
        ),
        receipt_sha256=_validate_hash(
            raw["receipt_sha256"], "receipt SHA-256"
        ),
    )
    if set(receipt.blinded_input_hashes) != set(_DISPLAY_LABELS):
        raise ValueError("Gate 5 receipt blinded input labels mismatch")
    if (
        len(receipt.development_seeds) != DEVELOPMENT_PAIR_COUNT
        or len(set(receipt.development_seeds)) != DEVELOPMENT_PAIR_COUNT
        or any(seed in CONFIRMATORY_SEEDS for seed in receipt.development_seeds)
        or len(receipt.blinded_seed_deltas) != DEVELOPMENT_PAIR_COUNT
        or receipt.effect != DESIGN_EFFECT
        or receipt.pairs != DESIGN_PAIRS
        or receipt.studies != DESIGN_STUDIES
        or receipt.simulation_seed != DESIGN_SIMULATION_SEED
        or receipt.simulation_version != DESIGN_SIMULATION_VERSION
        or receipt.successes < 0
        or receipt.successes > DESIGN_STUDIES
        or receipt.power != receipt.successes / DESIGN_STUDIES
        or receipt.passed != (receipt.power >= 0.80)
    ):
        raise ValueError("Gate 5 receipt values violate the frozen design")
    for name, value in (
        ("variance estimate", receipt.variance_estimate),
        ("power", receipt.power),
        ("power lower bound", receipt.power_ci_lo),
        ("power upper bound", receipt.power_ci_hi),
        *(
            ("blinded seed delta", value)
            for value in receipt.blinded_seed_deltas
        ),
    ):
        _validate_number(value, name)
    base = receipt._base_dict()
    decision = {
        "record_type": "gate5_decision",
        "schema_version": 1,
        "gate": 5,
        "passed": receipt.passed,
        "successes": receipt.successes,
        "studies": receipt.studies,
        "power": receipt.power,
        "threshold": 0.80,
        "effect": receipt.effect,
        "pairs": receipt.pairs,
        "permutation_commitment": receipt.permutation_commitment,
        "blinded_input_hashes": dict(receipt.blinded_input_hashes),
        "simulation_seed": receipt.simulation_seed,
        "simulation_version": receipt.simulation_version,
    }
    if receipt.decision_sha256 != _hash(decision):
        raise ValueError("Gate 5 decision hash mismatch")
    if receipt.receipt_sha256 != _hash(
        {**base, "decision_sha256": receipt.decision_sha256}
    ):
        raise ValueError("Gate 5 receipt hash mismatch")
    return receipt


def write_design_receipt(
    path: str | Path,
    receipt: DesignPowerReceipt,
) -> Path:
    destination = Path(path)
    _require_regular_parent(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"Gate 5 receipt already exists: {destination}")
    validated = validate_design_receipt(receipt.to_dict())
    _write_synced(destination, canonical_json_bytes(validated.to_dict()))
    parent_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return destination
