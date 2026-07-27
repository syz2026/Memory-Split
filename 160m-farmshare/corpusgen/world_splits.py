from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from corpusgen.relation_schema import RelationSchema


SplitName = Literal[
    "development",
    "train",
    "protected_seen",
    "protected_heldout",
]

TYPE_METADATA_REASON = "pinned Wikidata5M source provides no type metadata"
_SPLIT_NAMES = (
    "development",
    "train",
    "protected_seen",
    "protected_heldout",
)
_WORLD_NAMESPACE_SIZE = 1 << 20
_ENTITY_NAMESPACE_SIZE = 1 << 32
_MAX_COMPOSITION_CANDIDATES = 100_000
_REQUIRED_RENDERED_METADATA = {
    "world_id",
    "relation_path_hash",
    "template_id",
    "composition_split",
    "hop_count",
}
_REQUIRED_HOPS_BY_SPLIT = {
    "development": frozenset(range(1, 5)),
    "train": frozenset(range(1, 5)),
    "protected_seen": frozenset(range(1, 7)),
    "protected_heldout": frozenset(range(1, 7)),
}


def composition_hash(relations: Sequence[str]) -> str:
    payload = json.dumps(list(relations), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def composition_bucket(relations: Sequence[str]) -> int:
    return int(composition_hash(relations)[:8], 16) % 100


@dataclass(frozen=True)
class SplitPartition:
    name: SplitName
    namespace: str
    world_seed_range: tuple[int, int]
    entity_id_range: tuple[int, int]
    payload_namespace: str
    paraphrase_namespace: str
    question_namespace: str
    compositions: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if self.name not in _SPLIT_NAMES:
            raise ValueError(f"invalid split name: {self.name!r}")
        if not self.namespace:
            raise ValueError("split namespace must be nonempty")
        for label, value in (
            ("world seed", self.world_seed_range),
            ("entity ID", self.entity_id_range),
        ):
            if (
                len(value) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in value
                )
                or value[0] < 0
                or value[0] >= value[1]
            ):
                raise ValueError(f"invalid {label} namespace range")
        for label, value in (
            ("payload", self.payload_namespace),
            ("paraphrase", self.paraphrase_namespace),
            ("question", self.question_namespace),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} namespace must be nonempty")
        frozen_compositions = tuple(
            tuple(relation_ids) for relation_ids in self.compositions
        )
        if not frozen_compositions or any(
            not relation_ids
            or any(
                not isinstance(relation_id, str) or not relation_id
                for relation_id in relation_ids
            )
            for relation_ids in frozen_compositions
        ):
            raise ValueError(
                "split partitions require nonempty relation compositions"
            )
        if len(frozen_compositions) != len(set(frozen_compositions)):
            raise ValueError("relation compositions must be unique")
        object.__setattr__(self, "compositions", frozen_compositions)

    @property
    def composition_hashes(self) -> frozenset[str]:
        return frozenset(composition_hash(path) for path in self.compositions)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "namespace": self.namespace,
            "world_seed_range": list(self.world_seed_range),
            "entity_id_range": list(self.entity_id_range),
            "payload_namespace": self.payload_namespace,
            "paraphrase_namespace": self.paraphrase_namespace,
            "question_namespace": self.question_namespace,
            "compositions": [list(path) for path in self.compositions],
            "composition_hashes": sorted(self.composition_hashes),
        }


@dataclass(frozen=True)
class SplitPlan:
    seed: int
    schema_sha256: str
    development: SplitPartition
    train: SplitPartition
    protected_seen: SplitPartition
    protected_heldout: SplitPartition

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("split seed must be a nonnegative integer")
        if not re.fullmatch(r"[0-9a-f]{64}", self.schema_sha256):
            raise ValueError("schema_sha256 must be lowercase SHA-256")
        for expected_name, partition in zip(
            _SPLIT_NAMES,
            self.partitions,
        ):
            if partition.name != expected_name:
                raise ValueError("split partition names are out of order")

    @property
    def partitions(self) -> tuple[SplitPartition, ...]:
        return (
            self.development,
            self.train,
            self.protected_seen,
            self.protected_heldout,
        )

    def partition(self, name: SplitName | str) -> SplitPartition:
        if name not in _SPLIT_NAMES:
            raise ValueError(f"invalid split name: {name!r}")
        return cast(SplitPartition, getattr(self, name))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "seed": self.seed,
            "schema_sha256": self.schema_sha256,
            "type_metadata": {
                "available": False,
                "reason": TYPE_METADATA_REASON,
            },
            "partitions": {
                partition.name: partition.to_dict()
                for partition in self.partitions
            },
        }

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def require_static_disjointness(self) -> None:
        failed = [
            axis
            for axis, passed in audit_split_plan_disjointness(self).items()
            if not passed
        ]
        if failed:
            raise ValueError(
                "split plan static disjointness failure: " + ", ".join(failed)
            )


@dataclass(frozen=True)
class WorldArtifactSignature:
    world_id: int
    world_seed: int
    fact_ids: tuple[str, ...]
    row_address_sha256: str
    fact_count: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in (self.world_id, self.world_seed)
        ):
            raise ValueError(
                "world signature IDs and seeds must be nonnegative integers"
            )
        fact_ids = tuple(self.fact_ids)
        if (
            not fact_ids
            or any(
                not isinstance(fact_id, str) or not fact_id
                for fact_id in fact_ids
            )
            or len(fact_ids) != len(set(fact_ids))
        ):
            raise ValueError(
                "world signature fact IDs must be nonempty and unique"
            )
        if (
            isinstance(self.fact_count, bool)
            or not isinstance(self.fact_count, int)
            or self.fact_count != len(fact_ids)
        ):
            raise ValueError(
                "world signature fact count must match ordered fact IDs"
            )
        if not isinstance(self.row_address_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}",
            self.row_address_sha256,
        ):
            raise ValueError(
                "world signature row/address digest must be lowercase SHA-256"
            )
        object.__setattr__(self, "fact_ids", fact_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "world_id": self.world_id,
            "world_seed": self.world_seed,
            "fact_ids": list(self.fact_ids),
            "row_address_sha256": self.row_address_sha256,
            "fact_count": self.fact_count,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> WorldArtifactSignature:
        expected = {
            "world_id",
            "world_seed",
            "fact_ids",
            "row_address_sha256",
            "fact_count",
        }
        if set(value) != expected or not isinstance(value["fact_ids"], list):
            raise ValueError("invalid world artifact signature fields")
        return cls(
            world_id=value["world_id"],
            world_seed=value["world_seed"],
            fact_ids=tuple(value["fact_ids"]),
            row_address_sha256=value["row_address_sha256"],
            fact_count=value["fact_count"],
        )


@dataclass(frozen=True)
class ReasoningArtifactSignature:
    artifact_id: str
    world_id: int
    relation_path_hash: str
    template_id: str
    composition_split: str
    hop_count: int
    relations: tuple[str, ...]

    def __post_init__(self) -> None:
        relations = tuple(self.relations)
        if any(
            not isinstance(value, str) or not value
            for value in (self.artifact_id, self.template_id)
        ):
            raise ValueError(
                "reasoning signature IDs and templates must be nonempty"
            )
        if (
            isinstance(self.world_id, bool)
            or not isinstance(self.world_id, int)
            or self.world_id < 0
        ):
            raise ValueError(
                "reasoning signature world ID must be a nonnegative integer"
            )
        if (
            not relations
            or any(
                not isinstance(relation_id, str) or not relation_id
                for relation_id in relations
            )
        ):
            raise ValueError(
                "reasoning signature relations must be nonempty strings"
            )
        if (
            isinstance(self.hop_count, bool)
            or not isinstance(self.hop_count, int)
            or self.hop_count != len(relations)
            or self.hop_count not in range(1, 7)
        ):
            raise ValueError(
                "reasoning signature hop count must match 1-6 relations"
            )
        if self.composition_split not in {"seen", "heldout"}:
            raise ValueError(
                "reasoning signature composition split must be seen or heldout"
            )
        if (
            not isinstance(self.relation_path_hash, str)
            or composition_hash(relations) != self.relation_path_hash
        ):
            raise ValueError(
                "reasoning signature requires a canonical relation path hash"
            )
        object.__setattr__(self, "relations", relations)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "world_id": self.world_id,
            "relation_path_hash": self.relation_path_hash,
            "template_id": self.template_id,
            "composition_split": self.composition_split,
            "hop_count": self.hop_count,
            "relations": list(self.relations),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> ReasoningArtifactSignature:
        expected = {
            "artifact_id",
            "world_id",
            "relation_path_hash",
            "template_id",
            "composition_split",
            "hop_count",
            "relations",
        }
        if set(value) != expected or not isinstance(value["relations"], list):
            raise ValueError("invalid reasoning artifact signature fields")
        return cls(
            artifact_id=value["artifact_id"],
            world_id=value["world_id"],
            relation_path_hash=value["relation_path_hash"],
            template_id=value["template_id"],
            composition_split=value["composition_split"],
            hop_count=value["hop_count"],
            relations=tuple(value["relations"]),
        )


@dataclass(frozen=True)
class SplitArtifactExpectations:
    name: SplitName
    world_signatures: tuple[WorldArtifactSignature, ...]
    qa_signatures: tuple[ReasoningArtifactSignature, ...]
    rendered_signatures: tuple[ReasoningArtifactSignature, ...]
    required_hops: frozenset[int]

    def __post_init__(self) -> None:
        if self.name not in _SPLIT_NAMES:
            raise ValueError(f"invalid split name: {self.name!r}")
        world_signatures = tuple(self.world_signatures)
        qa_signatures = tuple(self.qa_signatures)
        rendered_signatures = tuple(self.rendered_signatures)
        required_hops = frozenset(self.required_hops)
        for label, signatures, expected_type in (
            (
                "world signatures",
                world_signatures,
                WorldArtifactSignature,
            ),
            (
                "QA signatures",
                qa_signatures,
                ReasoningArtifactSignature,
            ),
            (
                "rendered signatures",
                rendered_signatures,
                ReasoningArtifactSignature,
            ),
        ):
            if not signatures:
                raise ValueError(f"expected {label} must be nonempty")
            if any(
                not isinstance(signature, expected_type)
                for signature in signatures
            ):
                raise ValueError(f"expected {label} have invalid types")
            identifiers = tuple(
                (
                    signature.world_id
                    if isinstance(signature, WorldArtifactSignature)
                    else signature.artifact_id
                )
                for signature in signatures
            )
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"expected {label} must have unique IDs")
        if required_hops != _REQUIRED_HOPS_BY_SPLIT[self.name]:
            raise ValueError(
                f"required hops for {self.name} must be "
                f"{sorted(_REQUIRED_HOPS_BY_SPLIT[self.name])}"
            )
        object.__setattr__(
            self,
            "world_signatures",
            world_signatures,
        )
        object.__setattr__(self, "qa_signatures", qa_signatures)
        object.__setattr__(
            self,
            "rendered_signatures",
            rendered_signatures,
        )
        object.__setattr__(self, "required_hops", required_hops)

    @property
    def world_ids(self) -> tuple[int, ...]:
        return tuple(signature.world_id for signature in self.world_signatures)

    @property
    def qa_question_ids(self) -> tuple[str, ...]:
        return tuple(
            signature.artifact_id for signature in self.qa_signatures
        )

    @property
    def rendered_record_ids(self) -> tuple[str, ...]:
        return tuple(
            signature.artifact_id for signature in self.rendered_signatures
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "name": self.name,
            "world_signatures": [
                signature.to_dict() for signature in self.world_signatures
            ],
            "qa_signatures": [
                signature.to_dict() for signature in self.qa_signatures
            ],
            "rendered_signatures": [
                signature.to_dict() for signature in self.rendered_signatures
            ],
            "required_hops": sorted(self.required_hops),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> SplitArtifactExpectations:
        expected = {
            "version",
            "name",
            "world_signatures",
            "qa_signatures",
            "rendered_signatures",
            "required_hops",
        }
        if set(value) != expected:
            raise ValueError("invalid split artifact expectation fields")
        version = value["version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != 1
        ):
            raise ValueError(
                f"unsupported split artifact expectation version: {version!r}"
            )
        collection_names = (
            "world_signatures",
            "qa_signatures",
            "rendered_signatures",
            "required_hops",
        )
        if any(not isinstance(value[name], list) for name in collection_names):
            raise ValueError(
                "split artifact expectation collections must be lists"
            )
        return cls(
            name=value["name"],
            world_signatures=tuple(
                WorldArtifactSignature.from_dict(item)
                for item in value["world_signatures"]
            ),
            qa_signatures=tuple(
                ReasoningArtifactSignature.from_dict(item)
                for item in value["qa_signatures"]
            ),
            rendered_signatures=tuple(
                ReasoningArtifactSignature.from_dict(item)
                for item in value["rendered_signatures"]
            ),
            required_hops=frozenset(value["required_hops"]),
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
    ) -> SplitArtifactExpectations:
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid split artifact expectation JSON: {source}"
            ) from exc
        if not isinstance(value, Mapping):
            raise ValueError("split artifact expectations must be an object")
        return cls.from_dict(value)

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(self.canonical_bytes())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def _require_exact_ids(
    label: str,
    actual: Sequence[Any],
    expected: Sequence[Any],
) -> None:
    if len(actual) != len(set(actual)):
        raise ValueError(f"duplicate {label} in observed artifacts")
    actual_ids = set(actual)
    expected_ids = set(expected)
    if actual_ids == expected_ids and len(actual) == len(expected):
        return
    details = []
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    if missing:
        details.append(f"missing {sorted(missing)!r}")
    if extra:
        details.append(f"extra {sorted(extra)!r}")
    if not details:
        details.append("count mismatch")
    raise ValueError(f"{label} mismatch: " + "; ".join(details))


def _canonical_row_address_sha256(facts: Sequence[Any]) -> str:
    canonical = []
    for fact in facts:
        row = getattr(fact, "row", None)
        as_json = getattr(row, "as_json", None)
        address = getattr(row, "address", None)
        if not callable(as_json) or address is None:
            raise ValueError("generated facts require canonical graph rows")
        canonical.append(
            {
                "address": {
                    "source_id": getattr(address, "source_id", None),
                    "relation_id": getattr(address, "relation_id", None),
                    "direction": getattr(address, "direction", None),
                },
                "row": as_json(),
            }
        )
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _reasoning_signature(
    artifact_id: str,
    metadata: Mapping[str, Any],
) -> ReasoningArtifactSignature:
    return ReasoningArtifactSignature(
        artifact_id=artifact_id,
        world_id=metadata.get("world_id"),
        relation_path_hash=metadata.get("relation_path_hash"),
        template_id=metadata.get("template_id"),
        composition_split=metadata.get("composition_split"),
        hop_count=metadata.get("hop_count"),
        relations=metadata.get("relations", ()),
    )


@dataclass(frozen=True)
class ObservedSplitArtifacts:
    name: SplitName
    world_ids: frozenset[int]
    world_seeds: frozenset[int]
    entity_ids: frozenset[int]
    payload_values: frozenset[str]
    paraphrase_assignments: frozenset[str]
    question_ids: frozenset[str]
    qa_question_ids: frozenset[str]
    rendered_record_ids: frozenset[str]
    relation_path_hashes: frozenset[str]
    qa_relation_path_hashes: frozenset[str]
    seen_relation_path_hashes: frozenset[str]
    heldout_relation_path_hashes: frozenset[str]
    rendered_relation_path_hashes: frozenset[str]
    rendered_seen_relation_path_hashes: frozenset[str]
    rendered_heldout_relation_path_hashes: frozenset[str]
    relation_compositions: frozenset[tuple[str, ...]]
    qa_relation_compositions: frozenset[tuple[str, ...]]
    seen_relation_compositions: frozenset[tuple[str, ...]]
    heldout_relation_compositions: frozenset[tuple[str, ...]]
    rendered_relation_compositions: frozenset[tuple[str, ...]]
    rendered_seen_relation_compositions: frozenset[tuple[str, ...]]
    rendered_heldout_relation_compositions: frozenset[tuple[str, ...]]
    qa_hops: frozenset[int]
    rendered_hops: frozenset[int]
    relation_hashes_canonical: bool

    @classmethod
    def from_generated(
        cls,
        name: SplitName | str,
        *,
        expectations: SplitArtifactExpectations,
        worlds: Iterable[Any],
        qa_items: Iterable[Any],
        rendered_records: Iterable[Any],
    ) -> ObservedSplitArtifacts:
        if name not in _SPLIT_NAMES:
            raise ValueError(f"invalid split name: {name!r}")
        split_name = cast(SplitName, name)
        if (
            not isinstance(expectations, SplitArtifactExpectations)
            or expectations.name != split_name
        ):
            raise ValueError("artifact expectations do not match split")
        frozen_worlds = tuple(worlds)
        frozen_items = tuple(qa_items)
        frozen_records = tuple(rendered_records)
        expected_world_signatures = {
            signature.world_id: signature
            for signature in expectations.world_signatures
        }
        expected_qa_signatures = {
            signature.artifact_id: signature
            for signature in expectations.qa_signatures
        }
        expected_rendered_signatures = {
            signature.artifact_id: signature
            for signature in expectations.rendered_signatures
        }

        world_ids: set[int] = set()
        world_seeds: set[int] = set()
        entity_ids: set[int] = set()
        payload_values: set[str] = set()
        paraphrase_assignments: set[str] = set()
        question_ids: set[str] = set()
        qa_question_ids: set[str] = set()
        rendered_record_ids: set[str] = set()
        relation_path_hashes: set[str] = set()
        qa_relation_path_hashes: set[str] = set()
        seen_relation_path_hashes: set[str] = set()
        heldout_relation_path_hashes: set[str] = set()
        rendered_relation_path_hashes: set[str] = set()
        rendered_seen_relation_path_hashes: set[str] = set()
        rendered_heldout_relation_path_hashes: set[str] = set()
        relation_compositions: set[tuple[str, ...]] = set()
        qa_relation_compositions: set[tuple[str, ...]] = set()
        seen_relation_compositions: set[tuple[str, ...]] = set()
        heldout_relation_compositions: set[tuple[str, ...]] = set()
        rendered_relation_compositions: set[tuple[str, ...]] = set()
        rendered_seen_relation_compositions: set[tuple[str, ...]] = set()
        rendered_heldout_relation_compositions: set[tuple[str, ...]] = set()
        qa_hops: set[int] = set()
        rendered_hops: set[int] = set()
        relation_hashes_canonical = True

        def add_row(row: Any) -> None:
            if isinstance(row, Mapping):
                source_id = row.get("source_id")
                target_kind = row.get("target_kind")
                target = row.get("target")
                qualifiers = row.get("qualifiers", ())
            else:
                source_id = getattr(row, "source_id", None)
                target_kind = getattr(row, "target_kind", None)
                target = getattr(row, "target", None)
                qualifiers = getattr(row, "qualifiers", ())
            if (
                isinstance(source_id, bool)
                or not isinstance(source_id, int)
                or source_id < 0
            ):
                raise ValueError("observed graph rows require integer source IDs")
            if not isinstance(target, str) or not target:
                raise ValueError("observed graph rows require nonempty targets")
            entity_ids.add(source_id)
            payload_values.add(target)
            if target_kind == "entity":
                try:
                    target_id = int(target)
                except ValueError as exc:
                    raise ValueError(
                        "entity targets must contain integer IDs"
                    ) from exc
                if target_id < 0:
                    raise ValueError("entity target IDs must be nonnegative")
                entity_ids.add(target_id)
            for qualifier in qualifiers:
                if (
                    not isinstance(qualifier, (list, tuple))
                    or len(qualifier) != 2
                    or not isinstance(qualifier[1], str)
                ):
                    raise ValueError("graph qualifiers must be string pairs")
                payload_values.add(qualifier[1])

        def add_relation_metadata(
            metadata: Mapping[str, Any],
            *,
            rendered: bool,
        ) -> None:
            nonlocal relation_hashes_canonical
            path_hash = metadata.get("relation_path_hash")
            composition_split = metadata.get("composition_split")
            relations = metadata.get("relations")
            if (
                not isinstance(path_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", path_hash)
            ):
                raise ValueError(
                    "emitted relation_path_hash must be lowercase SHA-256"
                )
            if composition_split not in {"seen", "heldout"}:
                raise ValueError(
                    "emitted composition_split must be seen or heldout"
                )
            if (
                not isinstance(relations, (list, tuple))
                or not relations
                or any(
                    not isinstance(relation_id, str) or not relation_id
                    for relation_id in relations
                )
            ):
                raise ValueError(
                    "emitted relation metadata requires a relation composition"
                )
            composition = tuple(relations)
            relation_hashes_canonical &= (
                composition_hash(composition) == path_hash
            )
            relation_path_hashes.add(path_hash)
            relation_compositions.add(composition)
            if composition_split == "seen":
                seen_relation_path_hashes.add(path_hash)
                seen_relation_compositions.add(composition)
            else:
                heldout_relation_path_hashes.add(path_hash)
                heldout_relation_compositions.add(composition)
            if not rendered:
                qa_relation_path_hashes.add(path_hash)
                qa_relation_compositions.add(composition)
                return
            rendered_relation_path_hashes.add(path_hash)
            rendered_relation_compositions.add(composition)
            if composition_split == "seen":
                rendered_seen_relation_path_hashes.add(path_hash)
                rendered_seen_relation_compositions.add(composition)
            else:
                rendered_heldout_relation_path_hashes.add(path_hash)
                rendered_heldout_relation_compositions.add(composition)

        def metadata_hop(metadata: Mapping[str, Any]) -> int:
            hop_count = metadata.get("hop_count")
            if (
                isinstance(hop_count, bool)
                or not isinstance(hop_count, int)
                or hop_count < 1
                or hop_count > 6
            ):
                raise ValueError(
                    "emitted hop_count must be an integer in [1, 6]"
                )
            relations = metadata.get("relations")
            if not isinstance(relations, (list, tuple)) or hop_count != len(
                relations
            ):
                raise ValueError(
                    "emitted hop_count must match relation composition"
                )
            return hop_count

        raw_world_ids = []
        for world in frozen_worlds:
            world_id = getattr(world, "world_id", None)
            if (
                isinstance(world_id, bool)
                or not isinstance(world_id, int)
                or world_id < 0
            ):
                raise ValueError(
                    "generated worlds require nonnegative integer IDs"
                )
            raw_world_ids.append(world_id)
        _require_exact_ids(
            "world IDs",
            raw_world_ids,
            expectations.world_ids,
        )

        raw_question_ids = [getattr(item, "qid", None) for item in frozen_items]
        if any(
            not isinstance(question_id, str) or not question_id
            for question_id in raw_question_ids
        ):
            raise ValueError("generated QA items require question IDs")
        _require_exact_ids(
            "QA question IDs",
            raw_question_ids,
            expectations.qa_question_ids,
        )

        raw_record_ids = []
        for record in frozen_records:
            schedule = getattr(record, "schedule", None)
            if getattr(schedule, "component", None) != "reasoning":
                raise ValueError(
                    "observed split artifacts require reasoning records"
                )
            record_id = getattr(schedule, "record_id", None)
            if not isinstance(record_id, str) or not record_id:
                raise ValueError("final rendered records require record IDs")
            raw_record_ids.append(record_id)
        _require_exact_ids(
            "rendered record IDs",
            raw_record_ids,
            expectations.rendered_record_ids,
        )

        fact_ids: set[str] = set()
        graph_addresses: set[Any] = set()
        for world in frozen_worlds:
            if getattr(world, "split_name", None) != split_name:
                raise ValueError("generated world does not match observed split")
            world_id = getattr(world, "world_id")
            seed = getattr(world, "world_seed", None)
            if (
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or seed < 0
            ):
                raise ValueError(
                    "schema-shaped generated worlds require integer seeds"
                )
            facts = getattr(world, "facts", None)
            if not isinstance(facts, tuple) or not facts:
                raise ValueError("observed generated worlds require graph facts")
            manifest = getattr(world, "manifest", None)
            if not isinstance(manifest, Mapping):
                raise ValueError("generated world manifest must be an object")
            manifest_world_id = manifest.get("world_id")
            if manifest_world_id != world_id:
                raise ValueError("world ID does not match manifest world ID")
            manifest_seed = manifest.get("world_seed")
            if (
                isinstance(manifest_seed, bool)
                or not isinstance(manifest_seed, int)
                or manifest_seed < 0
            ):
                raise ValueError(
                    "world manifests require nonnegative integer seeds"
                )
            if seed != manifest_seed:
                raise ValueError("world seed does not match manifest world seed")
            manifest_fact_count = manifest.get("fact_count")
            if (
                isinstance(manifest_fact_count, bool)
                or not isinstance(manifest_fact_count, int)
                or manifest_fact_count != len(facts)
            ):
                raise ValueError(
                    "manifest fact count does not match generated facts"
                )
            for fact in facts:
                fact_id = getattr(fact, "fact_id", None)
                if not isinstance(fact_id, str) or not fact_id:
                    raise ValueError("generated facts require fact IDs")
                if fact_id in fact_ids:
                    raise ValueError(
                        "duplicate fact ID in observed generated worlds"
                    )
                fact_ids.add(fact_id)
                row = getattr(fact, "row", None)
                address = getattr(row, "address", None)
                if address in graph_addresses:
                    raise ValueError(
                        "duplicate graph address in observed generated worlds"
                    )
                graph_addresses.add(address)
            observed_world_signature = WorldArtifactSignature(
                world_id=world_id,
                world_seed=seed,
                fact_ids=tuple(fact.fact_id for fact in facts),
                row_address_sha256=_canonical_row_address_sha256(facts),
                fact_count=len(facts),
            )
            if (
                observed_world_signature
                != expected_world_signatures[world_id]
            ):
                raise ValueError(
                    "world artifact signature mismatch for world ID "
                    f"{world_id}"
                )
            world_ids.add(world_id)
            world_seeds.add(seed)
            for fact in facts:
                row = getattr(fact, "row", None)
                add_row(row)
            raw_assignments = manifest.get("paraphrase_assignment_ids")
            if not isinstance(raw_assignments, Mapping) or not raw_assignments:
                raise ValueError(
                    "world manifest requires paraphrase assignment IDs"
                )
            for assignment_id in raw_assignments.values():
                if not isinstance(assignment_id, str) or not assignment_id:
                    raise ValueError(
                        "paraphrase assignment IDs must be nonempty strings"
                    )
                paraphrase_assignments.add(assignment_id)

        for item in frozen_items:
            qid = getattr(item, "qid", None)
            metadata = getattr(item, "meta", None)
            if not isinstance(metadata, Mapping):
                raise ValueError("generated QA metadata must be an object")
            observed_qa_signature = _reasoning_signature(qid, metadata)
            if observed_qa_signature != expected_qa_signatures[qid]:
                raise ValueError(
                    "QA artifact signature mismatch for question ID "
                    f"{qid!r}"
                )
            question_ids.add(qid)
            qa_question_ids.add(qid)
            add_relation_metadata(metadata, rendered=False)
            qa_hops.add(metadata_hop(metadata))
            changed_row = metadata.get("changed_row")
            if changed_row is not None:
                if not isinstance(changed_row, Mapping):
                    raise ValueError("changed_row metadata must be an object")
                add_row(changed_row)
            entity_slots = metadata.get("entity_slots", ())
            if not isinstance(entity_slots, (list, tuple)):
                raise ValueError("entity_slots metadata must be a sequence")
            for entity_id in entity_slots:
                if entity_id is None:
                    continue
                if isinstance(entity_id, bool) or not isinstance(entity_id, int):
                    raise ValueError("entity slots must contain integer IDs")
                entity_ids.add(entity_id)

        for record in frozen_records:
            schedule = getattr(record, "schedule", None)
            metadata = getattr(record, "metadata", None)
            if not isinstance(metadata, Mapping):
                raise ValueError("final reasoning records require metadata")
            missing = _REQUIRED_RENDERED_METADATA - metadata.keys()
            if missing:
                raise ValueError(
                    "final reasoning record metadata is missing fields: "
                    + ", ".join(sorted(missing))
                )
            question_id = metadata.get("question_id")
            if not isinstance(question_id, str) or not question_id:
                raise ValueError("final reasoning records require question IDs")
            if question_id != schedule.record_id:
                raise ValueError(
                    "rendered metadata question ID does not match "
                    "schedule record ID"
                )
            observed_rendered_signature = _reasoning_signature(
                schedule.record_id,
                metadata,
            )
            if (
                observed_rendered_signature
                != expected_rendered_signatures[schedule.record_id]
            ):
                raise ValueError(
                    "rendered artifact signature mismatch for record ID "
                    f"{schedule.record_id!r}"
                )
            question_ids.add(question_id)
            rendered_record_ids.add(schedule.record_id)
            add_relation_metadata(metadata, rendered=True)
            rendered_hops.add(metadata_hop(metadata))
        if qa_hops != expectations.required_hops:
            raise ValueError(
                "QA hop coverage mismatch: "
                f"expected {sorted(expectations.required_hops)!r}, "
                f"observed {sorted(qa_hops)!r}"
            )
        if rendered_hops != expectations.required_hops:
            raise ValueError(
                "rendered hop coverage mismatch: "
                f"expected {sorted(expectations.required_hops)!r}, "
                f"observed {sorted(rendered_hops)!r}"
            )

        return cls(
            name=split_name,
            world_ids=frozenset(world_ids),
            world_seeds=frozenset(world_seeds),
            entity_ids=frozenset(entity_ids),
            payload_values=frozenset(payload_values),
            paraphrase_assignments=frozenset(paraphrase_assignments),
            question_ids=frozenset(question_ids),
            qa_question_ids=frozenset(qa_question_ids),
            rendered_record_ids=frozenset(rendered_record_ids),
            relation_path_hashes=frozenset(relation_path_hashes),
            qa_relation_path_hashes=frozenset(qa_relation_path_hashes),
            seen_relation_path_hashes=frozenset(
                seen_relation_path_hashes
            ),
            heldout_relation_path_hashes=frozenset(
                heldout_relation_path_hashes
            ),
            rendered_relation_path_hashes=frozenset(
                rendered_relation_path_hashes
            ),
            rendered_seen_relation_path_hashes=frozenset(
                rendered_seen_relation_path_hashes
            ),
            rendered_heldout_relation_path_hashes=frozenset(
                rendered_heldout_relation_path_hashes
            ),
            relation_compositions=frozenset(relation_compositions),
            qa_relation_compositions=frozenset(
                qa_relation_compositions
            ),
            seen_relation_compositions=frozenset(
                seen_relation_compositions
            ),
            heldout_relation_compositions=frozenset(
                heldout_relation_compositions
            ),
            rendered_relation_compositions=frozenset(
                rendered_relation_compositions
            ),
            rendered_seen_relation_compositions=frozenset(
                rendered_seen_relation_compositions
            ),
            rendered_heldout_relation_compositions=frozenset(
                rendered_heldout_relation_compositions
            ),
            qa_hops=frozenset(qa_hops),
            rendered_hops=frozenset(rendered_hops),
            relation_hashes_canonical=relation_hashes_canonical,
        )


def _compositions_for(
    relation_ids: tuple[str, ...],
    *,
    hops: int,
    bucket_start: int,
    bucket_stop: int,
    count: int,
) -> tuple[tuple[str, ...], ...]:
    found: list[tuple[str, ...]] = []
    candidates = itertools.islice(
        itertools.product(relation_ids, repeat=hops),
        _MAX_COMPOSITION_CANDIDATES,
    )
    for relations in candidates:
        if hops > 1 and not any(
            relations.count(relation_id) == 1
            for relation_id in set(relations)
        ):
            continue
        if bucket_start <= composition_bucket(relations) < bucket_stop:
            found.append(relations)
            if len(found) == count:
                return tuple(found)
    raise ValueError(
        f"schema cannot supply {count} length-{hops} compositions "
        f"in buckets {bucket_start}-{bucket_stop - 1}"
    )


def _range(base: int, ordinal: int, stride: int) -> tuple[int, int]:
    start = base + ordinal * stride
    return start, start + stride


def build_split_plan(schema: RelationSchema, seed: int) -> SplitPlan:
    if not isinstance(schema, RelationSchema):
        raise TypeError("schema must be a RelationSchema")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("split seed must be a nonnegative integer")

    entity_relations = tuple(
        spec.relation_id
        for spec in schema.path_relations
        if spec.target_kind == "entity"
    )
    if not entity_relations:
        raise ValueError("split planning requires entity-valued path relations")

    train_by_hop = {
        hops: _compositions_for(
            entity_relations,
            hops=hops,
            bucket_start=0,
            bucket_stop=80,
            count=2,
        )
        for hops in range(1, 5)
    }
    protected_seen = tuple(
        train_by_hop[hops][0] for hops in range(1, 5)
    )
    train = protected_seen + tuple(
        train_by_hop[hops][1] for hops in range(1, 5)
    )
    development = tuple(
        _compositions_for(
            entity_relations,
            hops=hops,
            bucket_start=80,
            bucket_stop=90,
            count=1,
        )[0]
        for hops in range(1, 5)
    )
    protected_heldout = tuple(
        _compositions_for(
            entity_relations,
            hops=hops,
            bucket_start=90,
            bucket_stop=100,
            count=1,
        )[0]
        for hops in range(1, 7)
    )

    world_base = seed * 3 * _WORLD_NAMESPACE_SIZE
    entity_base = seed * 3 * _ENTITY_NAMESPACE_SIZE

    def partition(
        name: SplitName,
        namespace: str,
        namespace_ordinal: int,
        compositions: tuple[tuple[str, ...], ...],
    ) -> SplitPartition:
        namespace_tag = f"{namespace}-{seed}"
        return SplitPartition(
            name=name,
            namespace=namespace,
            world_seed_range=_range(
                world_base,
                namespace_ordinal,
                _WORLD_NAMESPACE_SIZE,
            ),
            entity_id_range=_range(
                entity_base,
                namespace_ordinal,
                _ENTITY_NAMESPACE_SIZE,
            ),
            payload_namespace=f"{namespace_tag}:payload",
            paraphrase_namespace=f"{namespace_tag}:paraphrase",
            question_namespace=f"{namespace_tag}:question",
            compositions=compositions,
        )

    plan = SplitPlan(
        seed=seed,
        schema_sha256=schema.sha256(),
        development=partition(
            "development",
            "development",
            0,
            development,
        ),
        train=partition("train", "train", 1, train),
        protected_seen=partition(
            "protected_seen",
            "protected",
            2,
            protected_seen,
        ),
        protected_heldout=partition(
            "protected_heldout",
            "protected",
            2,
            protected_heldout,
        ),
    )
    plan.require_static_disjointness()
    return plan


def _ranges_disjoint(values: tuple[tuple[int, int], ...]) -> bool:
    ordered = sorted(values)
    return all(
        left[1] <= right[0]
        for left, right in zip(ordered, ordered[1:])
    )


def audit_split_plan_disjointness(plan: SplitPlan) -> dict[str, bool]:
    physical = (
        plan.development,
        plan.train,
        plan.protected_seen,
    )
    protected_namespaces_match = all(
        getattr(plan.protected_seen, field)
        == getattr(plan.protected_heldout, field)
        for field in (
            "namespace",
            "world_seed_range",
            "entity_id_range",
            "payload_namespace",
            "paraphrase_namespace",
            "question_namespace",
        )
    )
    development_hashes = plan.development.composition_hashes
    train_hashes = plan.train.composition_hashes
    seen_hashes = plan.protected_seen.composition_hashes
    heldout_hashes = plan.protected_heldout.composition_hashes
    relation_paths_disjoint = (
        development_hashes.isdisjoint(train_hashes)
        and development_hashes.isdisjoint(heldout_hashes)
        and heldout_hashes.isdisjoint(train_hashes)
        and seen_hashes <= train_hashes
    )

    return {
        "world_seeds": (
            protected_namespaces_match
            and _ranges_disjoint(
                tuple(item.world_seed_range for item in physical)
            )
        ),
        "entity_ids": (
            protected_namespaces_match
            and _ranges_disjoint(
                tuple(item.entity_id_range for item in physical)
            )
        ),
        "payload_values": (
            protected_namespaces_match
            and len({item.payload_namespace for item in physical})
            == len(physical)
        ),
        "paraphrase_assignments": (
            protected_namespaces_match
            and len({item.paraphrase_namespace for item in physical})
            == len(physical)
        ),
        "question_ids": (
            protected_namespaces_match
            and len({item.question_namespace for item in physical})
            == len(physical)
        ),
        "relation_path_hashes": relation_paths_disjoint,
        "heldout_relation_compositions": (
            heldout_hashes.isdisjoint(
                development_hashes | train_hashes | seen_hashes
            )
        ),
    }


def _sets_pairwise_disjoint(values: Sequence[frozenset[Any]]) -> bool:
    return all(
        left.isdisjoint(right)
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    )


def audit_disjointness(
    observed: Mapping[str, ObservedSplitArtifacts],
) -> dict[str, bool]:
    if set(observed) != set(_SPLIT_NAMES):
        raise ValueError(
            "artifact audit requires development, train, "
            "protected_seen, and protected_heldout"
        )
    partitions = tuple(observed[name] for name in _SPLIT_NAMES)
    if any(
        not isinstance(partition, ObservedSplitArtifacts)
        or partition.name != expected
        for expected, partition in zip(_SPLIT_NAMES, partitions)
    ):
        raise ValueError("observed artifact partitions are mislabeled")
    development, train, protected_seen, protected_heldout = partitions

    def physical_axis(field: str) -> bool:
        values = tuple(
            cast(frozenset[Any], getattr(partition, field))
            for partition in (
                development,
                train,
                protected_seen,
            )
        )
        protected_value = cast(
            frozenset[Any],
            getattr(protected_heldout, field),
        )
        return (
            all(values)
            and protected_value == values[-1]
            and _sets_pairwise_disjoint(values)
        )

    question_sets = tuple(partition.question_ids for partition in partitions)
    development_hashes = development.relation_path_hashes
    train_rendered_hashes = train.rendered_relation_path_hashes
    train_qa_hashes = train.qa_relation_path_hashes
    train_hashes = train_rendered_hashes | train_qa_hashes
    protected_seen_hashes = (
        protected_seen.seen_relation_path_hashes
        | protected_heldout.seen_relation_path_hashes
    )
    protected_heldout_hashes = (
        protected_seen.heldout_relation_path_hashes
        | protected_heldout.heldout_relation_path_hashes
    )
    hashes_canonical = all(
        partition.relation_hashes_canonical for partition in partitions
    )
    relation_path_hashes = (
        hashes_canonical
        and all(
            (
                development_hashes,
                train_rendered_hashes,
                train_qa_hashes,
                protected_seen_hashes,
                protected_heldout_hashes,
            )
        )
        and development_hashes.isdisjoint(train_hashes)
        and development_hashes.isdisjoint(protected_heldout_hashes)
        and protected_heldout_hashes.isdisjoint(train_hashes)
        and protected_seen_hashes <= train_rendered_hashes
    )

    development_compositions = development.relation_compositions
    train_rendered_compositions = train.rendered_relation_compositions
    train_qa_compositions = train.qa_relation_compositions
    train_compositions = train_rendered_compositions | train_qa_compositions
    protected_seen_compositions = (
        protected_seen.seen_relation_compositions
        | protected_heldout.seen_relation_compositions
    )
    protected_heldout_compositions = (
        protected_seen.heldout_relation_compositions
        | protected_heldout.heldout_relation_compositions
    )

    return {
        "world_seeds": (
            physical_axis("world_ids") and physical_axis("world_seeds")
        ),
        "entity_ids": physical_axis("entity_ids"),
        "payload_values": physical_axis("payload_values"),
        "paraphrase_assignments": physical_axis(
            "paraphrase_assignments"
        ),
        "question_ids": (
            all(question_sets) and _sets_pairwise_disjoint(question_sets)
        ),
        "relation_path_hashes": relation_path_hashes,
        "heldout_relation_compositions": (
            all(
                (
                    development_compositions,
                    train_rendered_compositions,
                    train_qa_compositions,
                    protected_seen_compositions,
                    protected_heldout_compositions,
                )
            )
            and protected_seen_compositions <= train_rendered_compositions
            and protected_heldout_compositions.isdisjoint(
                development_compositions
                | train_compositions
                | protected_seen_compositions
            )
        ),
    }


def require_disjointness(
    observed: Mapping[str, ObservedSplitArtifacts],
) -> None:
    failed = [
        axis
        for axis, passed in audit_disjointness(observed).items()
        if not passed
    ]
    if failed:
        raise ValueError(
            "observed artifact disjointness failure: " + ", ".join(failed)
        )
