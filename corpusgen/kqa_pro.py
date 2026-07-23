"""KQA Pro preparation for answer-only memory-transfer experiments.

KoPL is used only offline here: programs are executed to recover the lookup
facts a question depends on and to form a structure-only skeleton.  Programs
and SPARQL never enter the emitted continuation-training or evaluation text.

The fact unit is an organizer lookup entry rather than a raw Wikidata triple.
That matches the native MemorySplit query interface: ``name, relation`` maps
to one textual value (possibly a short `` | ``-separated list).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from corpusgen.records import (
    DB_END,
    DB_RETRIEVE,
    DB_START,
    Doc,
    QAItem,
    Segment,
    lookup_segments,
)
from organizer.store import Organizer, normalize


class KQAExecutionError(RuntimeError):
    """Raised when a labeled KoPL item cannot be executed faithfully."""


_DATE_RE = re.compile(r"^(-?\d+)[/-](\d+)[/-](\d+)$")


def _parse_date(text: str) -> tuple[int, int, int]:
    match = _DATE_RE.fullmatch(text.strip())
    if match is None:
        raise KQAExecutionError(f"invalid date value: {text!r}")
    year, month, day = match.groups()
    return int(year), int(month), int(day)


@dataclass(frozen=True)
class LookupFact:
    key: str
    name: str
    relation: str
    value: str
    kind: str
    value_count: int = 1
    raw_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceResult:
    answer: str
    skeleton: str
    support_keys: tuple[str, ...]
    support_atoms: tuple[str, ...]
    program_length: int


@dataclass(frozen=True)
class SplitConfig:
    train_limit: int = 20_000
    dev_limit: int = 1_000
    test_limit: int = 2_000
    min_train_per_skeleton: int = 8
    max_support_facts: int = 16
    max_values_per_lookup: int = 16
    noise_limit: int | None = None
    exclude_find_all: bool = True
    seed: int = 42


@dataclass
class PreparedKQA:
    train: list[dict]
    dev: list[dict]
    test: list[dict]
    fact_buckets: dict[str, list[str]]
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _TypedValue:
    kind: str
    value: Any
    unit: str = ""

    @classmethod
    def from_raw(cls, raw: dict) -> "_TypedValue":
        kind = raw["type"]
        value = raw["value"]
        if kind == "date":
            value = _parse_date(str(value))
        elif kind == "year":
            value = int(value)
        elif kind == "quantity":
            value = float(value)
        elif kind == "string":
            value = str(value)
        else:
            raise KQAExecutionError(f"unsupported KQA value type: {kind!r}")
        return cls(kind, value, str(raw.get("unit", "")))

    def __str__(self) -> str:
        if self.kind == "date":
            year, month, day = self.value
            year_text = f"{year:04d}" if year >= 0 else f"-{abs(year):04d}"
            return f"{year_text}-{month:02d}-{day:02d}"
        if self.kind == "quantity":
            number = int(self.value) if abs(self.value - int(self.value)) < 1e-5 else self.value
            return f"{number} {self.unit}" if self.unit and self.unit != "1" else str(number)
        return str(self.value)

    @property
    def is_time(self) -> bool:
        return self.kind in {"year", "date"}

    def can_compare(self, other: "_TypedValue") -> bool:
        if self.kind == "string":
            return other.kind == "string"
        if self.kind == "quantity":
            return other.kind == "quantity" and self.unit == other.unit
        return other.is_time

    def contains(self, other: "_TypedValue") -> bool:
        if self.kind == "year":
            other_year = other.value if other.kind == "year" else other.value[0]
            return self.value == other_year
        if self.kind == "date":
            return other.kind == "date" and self.value == other.value
        raise KQAExecutionError(f"contains is undefined for {self.kind}")


def _semantic_atom(kind: str, *parts: object) -> str:
    """Stable identity for one semantic KB fact, independent of storage alias."""
    return f"{kind}:{json.dumps(parts, ensure_ascii=False, separators=(',', ':'))}"


def _attribute_atom(entity_id: str, key: str, value: _TypedValue) -> str:
    return _semantic_atom(
        "attribute",
        entity_id,
        key,
        value.kind,
        value.value,
        value.unit,
    )


def _relation_atom(owner_id: str, relation: dict) -> str:
    direction = relation["direction"]
    if direction == "forward":
        subject_id, object_id = owner_id, relation["object"]
    elif direction == "backward":
        subject_id, object_id = relation["object"], owner_id
    else:
        raise KQAExecutionError(f"invalid relation direction: {direction!r}")
    return _semantic_atom(
        "relation",
        subject_id,
        relation["predicate"],
        object_id,
    )


@dataclass(frozen=True)
class _Handle:
    kind: str
    owner_id: str
    index: int


@dataclass
class _Entities:
    ids: list[str]
    handles: list[_Handle] | None
    supports: set[str]


@dataclass
class _Scalar:
    value: Any
    supports: set[str]


_CONTROL_INPUTS = {
    "Relate": ((1, "direction"),),
    "FilterNum": ((2, "operator"),),
    "FilterYear": ((2, "operator"),),
    "FilterDate": ((2, "operator"),),
    "QFilterNum": ((2, "operator"),),
    "QFilterYear": ((2, "operator"),),
    "QFilterDate": ((2, "operator"),),
    "VerifyNum": ((1, "operator"),),
    "VerifyYear": ((1, "operator"),),
    "VerifyDate": ((1, "operator"),),
    "SelectBetween": ((1, "selection"),),
    "SelectAmong": ((1, "selection"),),
}


def program_skeleton(program: list[dict]) -> str:
    """Structure signature retaining control flow but removing fact arguments."""
    parts = []
    for step in program:
        function = step["function"]
        dependencies = ",".join(str(i) for i in step.get("dependencies", []))
        controls = []
        inputs = step.get("inputs", [])
        for index, label in _CONTROL_INPUTS.get(function, ()):
            if index < len(inputs):
                controls.append(f"{label}={inputs[index]}")
        suffix = "{" + ",".join(controls) + "}" if controls else ""
        parts.append(f"{function}[{dependencies}]{suffix}")
    return "|".join(parts)


def _compare(a: _TypedValue, b: _TypedValue, op: str) -> bool:
    if not a.can_compare(b):
        return False
    if b.is_time and op in {"=", "!="}:
        equal = b.contains(a)
        return equal if op == "=" else not equal
    if op == "=":
        return a.kind == b.kind and a.value == b.value and (
            a.kind != "quantity" or a.unit == b.unit
        )
    if op == "!=":
        return not _compare(a, b, "=")
    if a.kind == "string":
        raise KQAExecutionError("ordered comparison between strings")
    av = a.value
    bv = b.value
    if a.is_time and b.is_time and a.kind != b.kind:
        av = av if a.kind == "year" else av[0]
        bv = bv if b.kind == "year" else bv[0]
    return av < bv if op == "<" else av > bv


def _normalize_answer(text: Any) -> str:
    return " ".join(str(text).lower().split()).removesuffix(".").strip()


class KQAKnowledgeBase:
    """KQA KB plus a provenance-producing executor and organizer projection."""

    def __init__(self, kb: dict):
        self.concepts: dict[str, dict] = kb["concepts"]
        self.entities: dict[str, dict] = kb["entities"]
        for info in list(self.concepts.values()) + list(self.entities.values()):
            info["name"] = " ".join(info["name"].split())

        self.entity_name_to_ids: dict[str, list[str]] = defaultdict(list)
        self.concept_name_to_ids: dict[str, list[str]] = defaultdict(list)
        for eid, info in self.entities.items():
            self.entity_name_to_ids[info["name"]].append(eid)
        for cid, info in self.concepts.items():
            self.concept_name_to_ids[info["name"]].append(cid)

        counts = Counter(
            info["name"] for info in list(self.entities.values()) + list(self.concepts.values())
        )
        self.ambiguous_names = {name for name, count in counts.items() if count > 1}

        self.key_type: dict[str, str] = {}
        self._attrs: dict[str, list[dict]] = {}
        self._relations: dict[str, list[dict]] = {
            item_id: [] for item_id in (*self.entities.keys(), *self.concepts.keys())
        }
        for eid, info in self.entities.items():
            attrs = []
            for attr in info.get("attributes", []):
                value = _TypedValue.from_raw(attr["value"])
                parsed = {
                    **attr,
                    "_raw_id": _attribute_atom(eid, attr["key"], value),
                    "value": value,
                    "qualifiers": {
                        key: [_TypedValue.from_raw(value) for value in values]
                        for key, values in attr.get("qualifiers", {}).items()
                    },
                }
                attrs.append(parsed)
                self.key_type[attr["key"]] = parsed["value"].kind
                for key, values in parsed["qualifiers"].items():
                    if values:
                        self.key_type[key] = "date" if values[0].kind == "year" else values[0].kind
            self._attrs[eid] = attrs

            for relation in info.get("relations", []):
                parsed = {
                    **relation,
                    "_raw_id": _relation_atom(eid, relation),
                    "qualifiers": {
                        key: [_TypedValue.from_raw(value) for value in values]
                        for key, values in relation.get("qualifiers", {}).items()
                    },
                }
                self._relations[eid].append(parsed)
                for key, values in parsed["qualifiers"].items():
                    if values:
                        self.key_type[key] = "date" if values[0].kind == "year" else values[0].kind
                if relation["object"] in self.concepts:
                    self._relations[relation["object"]].append(
                        {
                            **parsed,
                            "object": eid,
                            "direction": (
                                "forward"
                                if relation["direction"] == "backward"
                                else "backward"
                            ),
                        }
                    )

        self._concept_cache: dict[str, tuple[str, ...]] = {}
        self.lookup_facts: dict[str, LookupFact] = {}
        self.ambiguous_keys: set[str] = set()
        self.blocked_keys: set[str] = set()
        self._build_lookup_projection()

    @classmethod
    def load(cls, path: str | Path) -> "KQAKnowledgeBase":
        with open(path) as handle:
            return cls(json.load(handle))

    def _name(self, item_id: str) -> str:
        if item_id in self.entities:
            return self.entities[item_id]["name"]
        if item_id in self.concepts:
            return self.concepts[item_id]["name"]
        raise KQAExecutionError(f"unknown entity/concept id: {item_id}")

    def _all_concepts(self, item_id: str) -> tuple[str, ...]:
        if item_id in self._concept_cache:
            return self._concept_cache[item_id]
        info = self.entities.get(item_id) or self.concepts.get(item_id)
        if info is None:
            raise KQAExecutionError(f"unknown item for concept traversal: {item_id}")
        found: list[str] = []
        seen: set[str] = set()
        queue = list(info.get("instanceOf", []))
        while queue:
            cid = queue.pop(0)
            if cid in seen:
                continue
            seen.add(cid)
            found.append(cid)
            queue.extend(self.concepts[cid].get("instanceOf", []))
        result = tuple(found)
        self._concept_cache[item_id] = result
        return result

    @staticmethod
    def _relation_label(predicate: str, direction: str) -> str:
        return f"{predicate} ({direction})"

    def _concept_key(self, item_id: str) -> str:
        return normalize(f"{self._name(item_id)}, instance of")

    def _attr_key(self, entity_id: str, key: str) -> str:
        return normalize(f"{self._name(entity_id)}, {key}")

    def _attr_qualifier_key(self, entity_id: str, index: int, qkey: str) -> str:
        attr = self._attrs[entity_id][index]
        relation = f"{attr['key']}, qualifier {qkey}"
        return normalize(f"{self._name(entity_id)}, {relation}")

    def _relation_key(self, owner_id: str, predicate: str, direction: str) -> str:
        relation = self._relation_label(predicate, direction)
        return normalize(f"{self._name(owner_id)}, {relation}")

    def _relation_qualifier_key(self, owner_id: str, index: int, qkey: str) -> str:
        relation = self._relations[owner_id][index]
        label = (
            f"{self._relation_label(relation['predicate'], relation['direction'])}, "
            f"qualifier {qkey}"
        )
        return normalize(f"{self._name(owner_id)}, {label}")

    def _relation_between_key(self, subject_id: str, object_id: str) -> str:
        return normalize(
            f"{self._name(subject_id)}, relation to {self._name(object_id)}"
        )

    def _register(
        self,
        name: str,
        relation: str,
        values: Iterable[str],
        kind: str,
        raw_ids: Iterable[str],
    ) -> None:
        unique_values = tuple(dict.fromkeys(str(value) for value in values))
        if not unique_values:
            return
        key = normalize(f"{name}, {relation}")
        raw_ids = tuple(raw_ids)
        incoming = LookupFact(
            key=key,
            name=name,
            relation=relation,
            value=" | ".join(unique_values),
            kind=kind,
            value_count=len(unique_values),
            raw_ids=raw_ids,
        )
        old = self.lookup_facts.get(key)
        if old is None:
            self.lookup_facts[key] = incoming
            return
        self.ambiguous_keys.add(key)
        merged_values = tuple(dict.fromkeys(old.value.split(" | ") + list(unique_values)))
        self.lookup_facts[key] = LookupFact(
            key=key,
            name=name,
            relation=relation,
            value=" | ".join(merged_values),
            kind=kind,
            value_count=len(merged_values),
            raw_ids=tuple(dict.fromkeys(old.raw_ids + raw_ids)),
        )

    def _build_lookup_projection(self) -> None:
        for eid in self.entities:
            concept_ids = self._all_concepts(eid)
            concepts = [self._name(cid) for cid in concept_ids]
            self._register(
                self._name(eid),
                "instance of",
                concepts,
                "concept",
                (f"instance:{eid}:{cid}" for cid in concept_ids),
            )

            by_key: dict[str, list[tuple[int, _TypedValue]]] = defaultdict(list)
            qualifier_groups: dict[
                tuple[str, str], list[tuple[str, str]]
            ] = defaultdict(list)
            for index, attr in enumerate(self._attrs[eid]):
                by_key[attr["key"]].append((index, attr["value"]))
                for qkey, values in attr["qualifiers"].items():
                    qualifier_groups[(attr["key"], qkey)].append(
                        (
                            f"{attr['value']} => {'; '.join(map(str, values))}",
                            f"{attr['_raw_id']}:qualifier:{qkey}",
                        )
                    )
            for key, indexed_values in by_key.items():
                self._register(
                    self._name(eid),
                    key,
                    (str(value) for _, value in indexed_values),
                    "attribute",
                    (self._attrs[eid][index]["_raw_id"] for index, _ in indexed_values),
                )
            for (key, qkey), mappings in qualifier_groups.items():
                self._register(
                    self._name(eid),
                    f"{key}, qualifier {qkey}",
                    (mapping for mapping, _ in mappings),
                    "attribute_qualifier",
                    (raw_id for _, raw_id in mappings),
                )

        for owner_id, owner_relations in self._relations.items():
            relation_groups: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
            between_groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
            qualifier_groups: dict[
                tuple[str, str, str], list[tuple[str, str]]
            ] = defaultdict(list)
            for index, relation in enumerate(owner_relations):
                obj_name = self._name(relation["object"])
                relation_groups[(relation["predicate"], relation["direction"])].append(
                    (index, obj_name)
                )
                if relation["direction"] == "forward":
                    between_groups[relation["object"]].append((index, relation["predicate"]))
                for qkey, values in relation["qualifiers"].items():
                    qualifier_groups[
                        (relation["predicate"], relation["direction"], qkey)
                    ].append(
                        (
                            f"{obj_name} => {'; '.join(map(str, values))}",
                            f"{relation['_raw_id']}:qualifier:{qkey}",
                        )
                    )
            for (predicate, direction), indexed_values in relation_groups.items():
                self._register(
                    self._name(owner_id),
                    self._relation_label(predicate, direction),
                    (value for _, value in indexed_values),
                    "relation",
                    (
                        owner_relations[index]["_raw_id"]
                        for index, _ in indexed_values
                    ),
                )
            for object_id, indexed_values in between_groups.items():
                self._register(
                    self._name(owner_id),
                    f"relation to {self._name(object_id)}",
                    (predicate for _, predicate in indexed_values),
                    "relation_name",
                    (
                        owner_relations[index]["_raw_id"]
                        for index, _ in indexed_values
                    ),
                )
            for (predicate, direction, qkey), mappings in qualifier_groups.items():
                self._register(
                    self._name(owner_id),
                    f"{self._relation_label(predicate, direction)}, qualifier {qkey}",
                    (mapping for mapping, _ in mappings),
                    "relation_qualifier",
                    (raw_id for _, raw_id in mappings),
                )

    def _parse_input(self, key: str | None, text: str, kind: str | None = None) -> _TypedValue:
        kind = kind or self.key_type[key]  # type: ignore[index]
        if kind == "string":
            return _TypedValue("string", text)
        if kind == "quantity":
            pieces = text.split()
            try:
                number = float(pieces[0])
            except (IndexError, ValueError) as exc:
                raise KQAExecutionError(f"invalid quantity input: {text!r}") from exc
            return _TypedValue("quantity", number, " ".join(pieces[1:]) or "1")
        if _DATE_RE.fullmatch(text.strip()):
            return _TypedValue("date", _parse_date(text))
        return _TypedValue("year", int(text))

    @staticmethod
    def _supports(*outputs: _Entities | _Scalar) -> set[str]:
        merged: set[str] = set()
        for output in outputs:
            merged.update(output.supports)
        return merged

    def _find(self, name: str) -> _Entities:
        ids = list(self.entity_name_to_ids.get(name, ()))
        ids.extend(self.concept_name_to_ids.get(name, ()))
        return _Entities(ids, None, set())

    def _execute_step(
        self,
        function: str,
        dependencies: list[_Entities | _Scalar],
        inputs: list[str],
    ) -> _Entities | _Scalar:
        if function == "FindAll":
            return _Entities(list(self.entities), None, set())
        if function == "Find":
            return self._find(inputs[0])
        if function == "FilterConcept":
            source = dependencies[0]
            assert isinstance(source, _Entities)
            wanted: set[str] = set()
            for cid in self.concept_name_to_ids.get(inputs[0], ()):
                wanted.add(cid)
            kept = []
            supports = self._supports(source)
            for item_id in source.ids:
                if item_id not in self.entities:
                    continue
                supports.add(self._concept_key(item_id))
                if wanted.intersection(self._all_concepts(item_id)):
                    kept.append(item_id)
            return _Entities(kept, None, supports)
        if function in {"FilterStr", "FilterNum", "FilterYear", "FilterDate"}:
            source = dependencies[0]
            assert isinstance(source, _Entities)
            key, text = inputs[0], inputs[1]
            op = inputs[2] if len(inputs) > 2 else "="
            kind = {
                "FilterStr": "string",
                "FilterNum": "quantity",
                "FilterYear": "year",
                "FilterDate": "date",
            }[function]
            target = self._parse_input(key, text, kind)
            ids: list[str] = []
            handles: list[_Handle] = []
            supports = self._supports(source)
            for entity_id in source.ids:
                if entity_id not in self.entities:
                    continue
                supports.add(self._attr_key(entity_id, key))
                for index, attr in enumerate(self._attrs[entity_id]):
                    if attr["key"] == key and _compare(attr["value"], target, op):
                        ids.append(entity_id)
                        handles.append(_Handle("attr", entity_id, index))
            return _Entities(ids, handles, supports)
        if function in {"QFilterStr", "QFilterNum", "QFilterYear", "QFilterDate"}:
            source = dependencies[0]
            assert isinstance(source, _Entities)
            if source.handles is None:
                raise KQAExecutionError(f"{function} requires provenance-bearing facts")
            key, text = inputs[0], inputs[1]
            op = inputs[2] if len(inputs) > 2 else "="
            kind = {
                "QFilterStr": "string",
                "QFilterNum": "quantity",
                "QFilterYear": "year",
                "QFilterDate": "date",
            }[function]
            target = self._parse_input(key, text, kind)
            ids: list[str] = []
            handles: list[_Handle] = []
            supports = self._supports(source)
            for entity_id, handle in zip(source.ids, source.handles):
                if handle.kind == "attr":
                    fact = self._attrs[handle.owner_id][handle.index]
                    support = self._attr_qualifier_key(handle.owner_id, handle.index, key)
                else:
                    fact = self._relations[handle.owner_id][handle.index]
                    support = self._relation_qualifier_key(handle.owner_id, handle.index, key)
                supports.add(support)
                if any(_compare(value, target, op) for value in fact["qualifiers"].get(key, ())):
                    ids.append(entity_id)
                    handles.append(handle)
            return _Entities(ids, handles, supports)
        if function == "Relate":
            source = dependencies[0]
            assert isinstance(source, _Entities)
            predicate, direction = inputs
            ids: list[str] = []
            handles: list[_Handle] = []
            supports = self._supports(source)
            for owner_id in source.ids:
                if owner_id not in self._relations:
                    continue
                supports.add(self._relation_key(owner_id, predicate, direction))
                for index, relation in enumerate(self._relations[owner_id]):
                    if (
                        relation["predicate"] == predicate
                        and relation["direction"] == direction
                    ):
                        ids.append(relation["object"])
                        handles.append(_Handle("rel", owner_id, index))
            return _Entities(ids, handles, supports)
        if function in {"And", "Or"}:
            left, right = dependencies
            assert isinstance(left, _Entities) and isinstance(right, _Entities)
            left_ids, right_ids = set(left.ids), set(right.ids)
            result = left_ids & right_ids if function == "And" else left_ids | right_ids
            return _Entities(sorted(result), None, self._supports(left, right))
        if function in {"What", "QueryName"}:
            source = dependencies[0]
            assert isinstance(source, _Entities)
            if not source.ids:
                raise KQAExecutionError(f"{function} received an empty entity set")
            return _Scalar(self._name(source.ids[0]), self._supports(source))
        if function == "Count":
            source = dependencies[0]
            assert isinstance(source, _Entities)
            return _Scalar(len(source.ids), self._supports(source))
        if function in {"SelectBetween", "SelectAmong"}:
            if function == "SelectBetween":
                candidates = []
                supports = self._supports(*dependencies)
                for dep in dependencies:
                    assert isinstance(dep, _Entities)
                    candidates.extend(dep.ids[:1])
            else:
                dep = dependencies[0]
                assert isinstance(dep, _Entities)
                candidates = list(dep.ids)
                supports = self._supports(dep)
            key, op = inputs
            ranked: list[tuple[str, _TypedValue]] = []
            for entity_id in candidates:
                supports.add(self._attr_key(entity_id, key))
                for attr in self._attrs.get(entity_id, ()):
                    if attr["key"] == key:
                        ranked.append((entity_id, attr["value"]))
                        break
            if not ranked:
                raise KQAExecutionError(f"{function} found no values for {key!r}")
            ranked.sort(key=lambda pair: _sort_key(pair[1]))
            chosen = ranked[0 if op in {"less", "smallest"} else -1][0]
            return _Scalar(self._name(chosen), supports)
        if function == "QueryAttr":
            source = dependencies[0]
            assert isinstance(source, _Entities)
            entity_id, key = source.ids[0], inputs[0]
            supports = self._supports(source)
            supports.add(self._attr_key(entity_id, key))
            for attr in self._attrs[entity_id]:
                if attr["key"] == key:
                    return _Scalar(attr["value"], supports)
            raise KQAExecutionError(f"attribute {key!r} missing for {entity_id}")
        if function == "QueryAttrUnderCondition":
            source = dependencies[0]
            assert isinstance(source, _Entities)
            entity_id = source.ids[0]
            key, qkey, qtext = inputs
            target = self._parse_input(qkey, qtext)
            supports = self._supports(source)
            supports.add(self._attr_key(entity_id, key))
            for index, attr in enumerate(self._attrs[entity_id]):
                if attr["key"] != key:
                    continue
                supports.add(self._attr_qualifier_key(entity_id, index, qkey))
                if any(
                    _compare(value, target, "=")
                    for value in attr["qualifiers"].get(qkey, ())
                ):
                    return _Scalar(attr["value"], supports)
            raise KQAExecutionError("no attribute satisfies qualifier condition")
        if function in {"VerifyStr", "VerifyNum", "VerifyYear", "VerifyDate"}:
            source = dependencies[0]
            assert isinstance(source, _Scalar)
            if not isinstance(source.value, _TypedValue):
                raise KQAExecutionError(f"{function} requires a typed value")
            kind = {
                "VerifyStr": "string",
                "VerifyNum": "quantity",
                "VerifyYear": "year",
                "VerifyDate": "date",
            }[function]
            target = self._parse_input(None, inputs[0], kind)
            op = inputs[1] if len(inputs) > 1 else "="
            return _Scalar("yes" if _compare(source.value, target, op) else "no", set(source.supports))
        if function == "QueryRelation":
            left, right = dependencies
            assert isinstance(left, _Entities) and isinstance(right, _Entities)
            subject, obj = left.ids[0], right.ids[0]
            supports = self._supports(left, right)
            supports.add(self._relation_between_key(subject, obj))
            for relation in self._relations.get(subject, ()):
                if relation["object"] == obj and relation["direction"] == "forward":
                    return _Scalar(relation["predicate"], supports)
            raise KQAExecutionError("no forward relation between entities")
        if function == "QueryAttrQualifier":
            source = dependencies[0]
            assert isinstance(source, _Entities)
            entity_id = source.ids[0]
            key, value_text, qkey = inputs
            target = self._parse_input(key, value_text)
            supports = self._supports(source)
            supports.add(self._attr_key(entity_id, key))
            for index, attr in enumerate(self._attrs[entity_id]):
                if attr["key"] != key or not _compare(attr["value"], target, "="):
                    continue
                supports.add(self._attr_qualifier_key(entity_id, index, qkey))
                values = attr["qualifiers"].get(qkey, ())
                if values:
                    return _Scalar(values[0], supports)
            raise KQAExecutionError("attribute qualifier not found")
        if function == "QueryRelationQualifier":
            left, right = dependencies
            assert isinstance(left, _Entities) and isinstance(right, _Entities)
            subject, obj = left.ids[0], right.ids[0]
            predicate, qkey = inputs
            supports = self._supports(left, right)
            for index, relation in enumerate(self._relations.get(subject, ())):
                if (
                    relation["object"] == obj
                    and relation["direction"] == "forward"
                    and relation["predicate"] == predicate
                ):
                    supports.add(self._relation_key(subject, predicate, "forward"))
                    supports.add(self._relation_qualifier_key(subject, index, qkey))
                    values = relation["qualifiers"].get(qkey, ())
                    if values:
                        return _Scalar(values[0], supports)
            raise KQAExecutionError("relation qualifier not found")
        raise KQAExecutionError(f"unsupported KoPL function: {function}")

    def trace(self, item: dict) -> TraceResult:
        program = item["program"]
        memory: list[_Entities | _Scalar] = []
        for index, step in enumerate(program):
            try:
                dependencies = [memory[i] for i in step.get("dependencies", ())]
            except IndexError as exc:
                raise KQAExecutionError(
                    f"invalid dependency at program step {index}: {step}"
                ) from exc
            memory.append(
                self._execute_step(step["function"], dependencies, step.get("inputs", []))
            )
        if not memory or not isinstance(memory[-1], _Scalar):
            raise KQAExecutionError("program did not produce a scalar answer")
        answer = str(memory[-1].value)
        if "answer" in item and _normalize_answer(answer) != _normalize_answer(item["answer"]):
            raise KQAExecutionError(
                f"executor answer mismatch: expected {item['answer']!r}, got {answer!r}"
            )
        return TraceResult(
            answer=answer,
            skeleton=program_skeleton(program),
            support_keys=tuple(sorted(memory[-1].supports)),
            support_atoms=tuple(
                sorted(
                    {
                        raw_id
                        for key in memory[-1].supports
                        if key in self.lookup_facts
                        for raw_id in self.lookup_facts[key].raw_ids
                    }
                )
            ),
            program_length=len(program),
        )


def _sort_key(value: _TypedValue) -> tuple:
    if value.kind == "string":
        return (0, value.value)
    if value.kind == "quantity":
        return (1, value.unit, value.value)
    if value.kind == "year":
        return (2, value.value, 0, 0)
    return (2, *value.value)


def _annotate_candidates(
    kb: KQAKnowledgeBase,
    items: list[dict],
    source: str,
    cfg: SplitConfig,
) -> tuple[list[dict], Counter]:
    accepted: list[dict] = []
    rejected: Counter = Counter()
    for index, item in enumerate(items):
        functions = {step["function"] for step in item.get("program", ())}
        if cfg.exclude_find_all and "FindAll" in functions:
            rejected["find_all"] += 1
            continue
        try:
            trace = kb.trace(item)
        except (KQAExecutionError, KeyError, IndexError, ValueError, TypeError):
            rejected["execution"] += 1
            continue
        support = set(trace.support_keys)
        if not support:
            rejected["no_support"] += 1
            continue
        if len(support) > cfg.max_support_facts:
            rejected["too_many_supports"] += 1
            continue
        if support & kb.ambiguous_keys:
            rejected["ambiguous_key"] += 1
            continue
        if support & kb.blocked_keys:
            rejected["blocked_lookup"] += 1
            continue
        if any(key not in kb.lookup_facts for key in support):
            rejected["missing_lookup"] += 1
            continue
        if any(
            kb.lookup_facts[key].value_count > cfg.max_values_per_lookup
            for key in support
        ):
            rejected["wide_lookup"] += 1
            continue
        accepted.append(
            {
                "qid": f"kqa-{source}-{index}",
                "question": item["question"],
                "answer": item["answer"],
                "skeleton": trace.skeleton,
                "support_keys": list(trace.support_keys),
                "support_atoms": list(trace.support_atoms),
                "support_count": len(trace.support_keys),
                "program_length": trace.program_length,
                "skill": item["program"][-1]["function"],
                "source": source,
            }
        )
    return accepted, rejected


def build_transfer_split(
    kb: KQAKnowledgeBase,
    train_items: list[dict],
    validation_items: list[dict],
    cfg: SplitConfig | None = None,
) -> PreparedKQA:
    """Build a fact-disjoint split with matched, previously seen skeletons."""
    cfg = cfg or SplitConfig()
    train_candidates, train_rejected = _annotate_candidates(
        kb, train_items, "train", cfg
    )
    val_candidates, val_rejected = _annotate_candidates(
        kb, validation_items, "val", cfg
    )
    rng = random.Random(cfg.seed)
    rng.shuffle(train_candidates)
    rng.shuffle(val_candidates)

    train_indices_by_atom: dict[str, set[int]] = defaultdict(set)
    eligible_indices = set(range(len(train_candidates)))
    eligible_counts = Counter(item["skeleton"] for item in train_candidates)
    for index, item in enumerate(train_candidates):
        for atom in item["support_atoms"]:
            train_indices_by_atom[atom].add(index)

    test: list[dict] = []
    required_skeletons: set[str] = set()
    for item in val_candidates:
        if len(test) >= cfg.test_limit:
            break
        removed: set[int] = set()
        for atom in item["support_atoms"]:
            removed.update(train_indices_by_atom.get(atom, set()) & eligible_indices)
        removed_counts = Counter(
            train_candidates[index]["skeleton"] for index in removed
        )
        would_require = required_skeletons | {item["skeleton"]}
        if any(
            eligible_counts[skeleton] - removed_counts[skeleton]
            < cfg.min_train_per_skeleton
            for skeleton in would_require
        ):
            continue
        test.append(item)
        required_skeletons.add(item["skeleton"])
        eligible_indices.difference_update(removed)
        eligible_counts.subtract(removed_counts)

    if len(test) != cfg.test_limit:
        raise ValueError(
            f"could construct only {len(test)} of {cfg.test_limit} requested "
            "semantic-fact-disjoint transfer questions"
        )

    transfer_keys = {key for item in test for key in item["support_keys"]}
    transfer_atoms = {atom for item in test for atom in item["support_atoms"]}
    eligible_train = [train_candidates[index] for index in sorted(eligible_indices)]
    eligible_by_skeleton: dict[str, list[dict]] = defaultdict(list)
    for item in eligible_train:
        eligible_by_skeleton[item["skeleton"]].append(item)

    selected_train: list[dict] = []
    selected_ids: set[str] = set()
    for skeleton in sorted(required_skeletons):
        choices = eligible_by_skeleton[skeleton]
        rng.shuffle(choices)
        for item in choices[: cfg.min_train_per_skeleton]:
            selected_train.append(item)
            selected_ids.add(item["qid"])
    if len(selected_train) > cfg.train_limit:
        raise ValueError(
            "train_limit is too small for min_train_per_skeleton across test skeletons"
        )
    remaining = [item for item in eligible_train if item["qid"] not in selected_ids]
    rng.shuffle(remaining)
    selected_train.extend(remaining[: cfg.train_limit - len(selected_train)])
    if len(selected_train) != cfg.train_limit:
        raise ValueError(
            f"could construct only {len(selected_train)} of {cfg.train_limit} "
            "requested training questions"
        )
    selected_ids = {item["qid"] for item in selected_train}

    seen_keys = {key for item in selected_train for key in item["support_keys"]}
    seen_atoms = {atom for item in selected_train for atom in item["support_atoms"]}
    if seen_atoms & transfer_atoms:
        raise AssertionError("semantic support leakage between train and transfer QA")
    train_skeletons = {item["skeleton"] for item in selected_train}
    test_ids = {test_item["qid"] for test_item in test}
    dev_candidates = [
        item
        for item in val_candidates
        if item["qid"] not in test_ids
        and item["skeleton"] in train_skeletons
        and set(item["support_keys"]) <= seen_keys
        and set(item["support_atoms"]) <= seen_atoms
    ]
    rng.shuffle(dev_candidates)
    dev = dev_candidates[: cfg.dev_limit]
    if len(dev) < cfg.dev_limit:
        dev_ids = {item["qid"] for item in dev}
        train_dev_candidates = [
            item
            for item in eligible_train
            if item["qid"] not in selected_ids
            and item["qid"] not in dev_ids
            and item["skeleton"] in train_skeletons
            and set(item["support_keys"]) <= seen_keys
            and set(item["support_atoms"]) <= seen_atoms
        ]
        rng.shuffle(train_dev_candidates)
        dev.extend(train_dev_candidates[: cfg.dev_limit - len(dev)])
    if len(dev) != cfg.dev_limit:
        raise ValueError(
            f"could construct only {len(dev)} of {cfg.dev_limit} requested "
            "seen-fact development questions"
        )

    all_keys = {
        key
        for key, fact in kb.lookup_facts.items()
        if key not in kb.blocked_keys
        and key not in kb.ambiguous_keys
        and fact.value_count <= cfg.max_values_per_lookup
    }
    all_noise_keys = all_keys - seen_keys - transfer_keys
    used_atoms = seen_atoms | transfer_atoms
    all_noise_keys = {
        key
        for key in all_noise_keys
        if set(kb.lookup_facts[key].raw_ids).isdisjoint(used_atoms)
    }
    noise_keys = set(all_noise_keys)
    if cfg.noise_limit is not None and len(noise_keys) > cfg.noise_limit:
        ordered_noise = sorted(noise_keys)
        rng.shuffle(ordered_noise)
        noise_keys = set(ordered_noise[: cfg.noise_limit])
    stats = {
        "config": asdict(cfg),
        "candidates": {
            "train_accepted": len(train_candidates),
            "validation_accepted": len(val_candidates),
            "train_rejected": dict(train_rejected),
            "validation_rejected": dict(val_rejected),
        },
        "questions": {
            "train": len(selected_train),
            "dev": len(dev),
            "test": len(test),
        },
        "skeletons": {
            "train": len(train_skeletons),
            "test": len({item["skeleton"] for item in test}),
        },
        "facts": {
            "seen": len(seen_keys),
            "transfer": len(transfer_keys),
            "noise": len(noise_keys),
            "noise_available": len(all_noise_keys),
            "seen_atoms": len(seen_atoms),
            "transfer_atoms": len(transfer_atoms),
        },
    }
    return PreparedKQA(
        train=selected_train,
        dev=dev,
        test=test,
        fact_buckets={
            "seen": sorted(seen_keys),
            "transfer": sorted(transfer_keys),
            "noise": sorted(noise_keys),
        },
        stats=stats,
    )


def render_fact_doc(fact: LookupFact, bucket: str) -> Doc:
    prefix = f"Knowledge: {fact.name}'s {fact.relation} is"
    dense = [(f"{prefix} {fact.value}.", False)]
    split: list[Segment] = [(prefix, False)]
    split.extend(lookup_segments(fact.name, fact.relation, fact.value))
    split.append((".", False))
    return Doc(
        kind="kqa_fact",
        dense_segments=dense,
        split_segments=split,
        meta={"fact_id": fact.key, "bucket": bucket, "fact_kind": fact.kind},
    )


def render_qa_doc(item: dict) -> Doc:
    text = f"Question: {item['question']}\nAnswer: {item['answer']}"
    return Doc(
        kind="kqa_qa",
        dense_segments=[(text, False)],
        split_segments=[(text, False)],
        meta={
            "qid": item["qid"],
            "skeleton": item["skeleton"],
            "support_keys": item["support_keys"],
            "support_atoms": item.get("support_atoms", []),
        },
    )


def make_recall_probe(fact: LookupFact, bucket: str) -> QAItem:
    return QAItem(
        qid=f"kqa-recall-{hashlib.sha1(fact.key.encode()).hexdigest()[:16]}",
        task="kqa_recall",
        prompt=f"Knowledge: {fact.name}'s {fact.relation} is",
        answer=fact.value,
        meta={
            "relation": fact.relation,
            "fact_id": fact.key,
            "query": f"{fact.name}, {fact.relation}",
            "bucket": bucket,
            "fact_kind": fact.kind,
        },
    )


def make_qa_item(item: dict, bucket: str) -> QAItem:
    return QAItem(
        qid=item["qid"],
        task="kqa",
        prompt=f"Question: {item['question']}\n",
        answer=item["answer"],
        meta={
            "template": item["skeleton"],
            "skeleton": item["skeleton"],
            "support_keys": item["support_keys"],
            "support_atoms": item["support_atoms"],
            "support_count": item["support_count"],
            "program_length": item["program_length"],
            "skill": item["skill"],
            "fact_bucket": bucket,
        },
    )


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_prepared_data(
    prepared: PreparedKQA,
    kb: KQAKnowledgeBase,
    out_dir: str | Path,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket_for_key = {
        key: bucket
        for bucket, keys in prepared.fact_buckets.items()
        for key in keys
    }
    fact_rows = []
    organizer = Organizer()
    for key, fact in sorted(kb.lookup_facts.items()):
        if key not in bucket_for_key:
            continue
        bucket = bucket_for_key[key]
        fact_rows.append({**asdict(fact), "bucket": bucket})
        organizer.add(fact.name, fact.relation, fact.value)
    _write_jsonl(out_dir / "facts.jsonl", fact_rows)
    for bucket in ("seen", "transfer", "noise"):
        _write_jsonl(
            out_dir / f"facts_{bucket}.jsonl",
            (row for row in fact_rows if row["bucket"] == bucket),
        )
    organizer.save(out_dir / "organizer.jsonl")

    _write_jsonl(out_dir / "qa_train.jsonl", prepared.train)
    _write_jsonl(out_dir / "qa_dev.jsonl", prepared.dev)
    _write_jsonl(out_dir / "qa_test.jsonl", prepared.test)
    _write_jsonl(
        out_dir / "recall_transfer.jsonl",
        (
            asdict(make_recall_probe(kb.lookup_facts[key], "transfer"))
            for key in prepared.fact_buckets["transfer"]
        ),
    )
    _write_jsonl(
        out_dir / "eval_transfer.jsonl",
        (asdict(make_qa_item(item, "transfer")) for item in prepared.test),
    )
    _write_jsonl(
        out_dir / "eval_dev.jsonl",
        (asdict(make_qa_item(item, "seen")) for item in prepared.dev),
    )

    seen = set(prepared.fact_buckets["seen"])
    transfer = set(prepared.fact_buckets["transfer"])
    seen_atoms = {atom for item in prepared.train for atom in item["support_atoms"]}
    transfer_atoms = {atom for item in prepared.test for atom in item["support_atoms"]}
    config = prepared.stats["config"]
    report = {
        **prepared.stats,
        "checks": {
            "seen_transfer_disjoint": seen.isdisjoint(transfer),
            "semantic_support_disjoint": seen_atoms.isdisjoint(transfer_atoms),
            "test_requires_transfer": bool(prepared.test)
            and all(
                set(item["support_keys"]) <= transfer and item["support_keys"]
                for item in prepared.test
            ),
            "test_skeletons_seen": {
                item["skeleton"] for item in prepared.test
            }
            <= {item["skeleton"] for item in prepared.train},
            "no_kopl_in_emitted_qa": all(
                "program" not in item and "sparql" not in item
                for item in prepared.train + prepared.dev + prepared.test
            ),
            "question_splits_disjoint": (
                {
                    item["qid"] for item in prepared.train
                }.isdisjoint(item["qid"] for item in prepared.dev)
                and {
                    item["qid"] for item in prepared.train
                }.isdisjoint(item["qid"] for item in prepared.test)
                and {
                    item["qid"] for item in prepared.dev
                }.isdisjoint(item["qid"] for item in prepared.test)
            ),
            "requested_train_size_met": len(prepared.train) == config["train_limit"],
            "requested_dev_size_met": len(prepared.dev) == config["dev_limit"],
            "requested_test_size_met": len(prepared.test) == config["test_limit"],
        },
    }
    with open(out_dir / "report.json", "w") as handle:
        json.dump(report, handle, indent=2)
    return report


def load_questions(path: str | Path) -> list[dict]:
    with open(path) as handle:
        return json.load(handle)

