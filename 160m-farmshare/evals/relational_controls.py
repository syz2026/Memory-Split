"""Deterministic causal controls for relational graph evaluation.

Controls never use Python's process-randomized ``hash``.  Every ordering and
choice is derived from canonical JSON and SHA-256 so a seed has the same
meaning across processes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from types import MappingProxyType
from typing import Any

from corpusgen.graph_records import (
    GraphAction,
    GraphAddress,
    GraphRow,
    stable_fact_id,
)
from corpusgen.records import QAItem
from organizer.graph_store import GraphStore, StoreStats


class ControlID(StrEnum):
    CORRECT = "correct"
    SHUFFLED_RETURNS = "shuffled_returns"
    RELEVANT_EDGE = "relevant_edge"
    IRRELEVANT_EDGE = "irrelevant_edge"
    GOLD_PATH = "gold_path"
    GOLD_RETURNS = "gold_returns"
    NO_QUERY = "no_query"
    EXPLICIT_MISS = "explicit_miss"
    HANDLE_SWAP = "handle_swap"
    ENTITY_RENAME = "entity_rename"
    GRAPH_ISOMORPHISM = "graph_isomorphism"


class EvalMode(StrEnum):
    MEMORY_OFF = "memory_off"
    MEMORY_ON = "memory_on"


_BINDING_RE = re.compile(r"Slot ([0-3]) refers to ([^.]+)\.")
_STORE_ROWS_CACHE: dict[int, tuple[GraphStore, tuple[GraphRow, ...]]] = {}
_STORE_GROUP_CACHE: dict[
    int,
    tuple[GraphStore, dict[str, tuple[GraphRow, ...]]],
] = {}
_STORE_ENTITY_CACHE: dict[int, tuple[GraphStore, tuple[int, ...]]] = {}
_STORE_STATS_CACHE: dict[int, tuple[GraphStore, StoreStats]] = {}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _address_json(address: GraphAddress) -> list[Any]:
    return [address.source_id, address.relation_id, address.direction]


def _stable_key(seed: int, label: str, value: Any) -> bytes:
    return hashlib.sha256(
        _canonical_bytes([seed, label, value])
    ).digest()


def _seeded_order(
    values: Iterable[Any],
    *,
    seed: int,
    label: str,
    serializer=lambda value: value,
) -> tuple[Any, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: (
                _stable_key(seed, label, serializer(value)),
                _canonical_bytes(serializer(value)),
            ),
        )
    )


def _validate_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("control seed must be a non-negative integer")
    return seed


def _item_value(item: QAItem | Mapping[str, Any], name: str) -> Any:
    return item[name] if isinstance(item, Mapping) else getattr(item, name)


def _item_meta(item: QAItem | Mapping[str, Any]) -> dict[str, Any]:
    value = _item_value(item, "meta")
    if not isinstance(value, dict):
        raise ValueError("control item meta must be a mutable mapping")
    return value


def _copy_item(item: QAItem | Mapping[str, Any]) -> QAItem | dict[str, Any]:
    if isinstance(item, QAItem):
        return QAItem(
            qid=item.qid,
            task=item.task,
            prompt=item.prompt,
            answer=item.answer,
            meta=copy.deepcopy(item.meta),
        )
    if isinstance(item, Mapping):
        required = {"qid", "task", "prompt", "answer", "meta"}
        if not required.issubset(item):
            raise ValueError("control item is missing required fields")
        return copy.deepcopy(dict(item))
    raise TypeError("control item must be a QAItem or mapping")


def _set_item_value(
    item: QAItem | dict[str, Any],
    name: str,
    value: Any,
) -> None:
    if isinstance(item, dict):
        item[name] = value
    else:
        setattr(item, name, value)


def _row_payload(row: GraphRow) -> tuple[Any, ...]:
    return row.target_kind, row.target, row.qualifiers, row.provenance_id


def _rows_from_store(store: GraphStore) -> tuple[GraphRow, ...]:
    cached = _STORE_ROWS_CACHE.get(id(store))
    if cached is not None and cached[0] is store:
        return cached[1]
    rows_method = getattr(store, "rows", None)
    if callable(rows_method):
        rows = tuple(rows_method())
    else:
        decode_row = getattr(store, "_decode_row", None)
        if callable(decode_row):
            rows = tuple(decode_row(index) for index in range(len(store)))
        else:
            base = getattr(store, "base", None)
            replacements = getattr(store, "_replacements", None)
            if base is None or not isinstance(replacements, Mapping):
                raise TypeError(
                    "control stores must expose deterministic row iteration"
                )
            rows = tuple(
                replacements.get(row.address, row)
                for row in _rows_from_store(base)
            )
    if len(rows) != len(store):
        raise ValueError("store row iteration count does not match len(store)")
    if any(not isinstance(row, GraphRow) for row in rows):
        raise TypeError("store rows must be GraphRow values")
    ordered = tuple(sorted(rows, key=lambda row: row.address))
    if len({row.address for row in ordered}) != len(ordered):
        raise ValueError("control store contains duplicate graph addresses")
    _STORE_ROWS_CACHE[id(store)] = (store, ordered)
    return ordered


def _base_store_and_replacements(
    store: GraphStore,
) -> tuple[GraphStore, dict[GraphAddress, GraphRow]]:
    replacements: dict[GraphAddress, GraphRow] = {}
    current = store
    seen: set[int] = set()
    while True:
        if id(current) in seen:
            raise ValueError("control store base chain contains a cycle")
        seen.add(id(current))
        base = getattr(current, "base", None)
        current_replacements = getattr(current, "_replacements", None)
        if base is None or not isinstance(current_replacements, Mapping):
            break
        for address, row in current_replacements.items():
            replacements.setdefault(address, row)
        current = base
    return current, replacements


def _source_store_stats(store: GraphStore) -> StoreStats:
    base, _ = _base_store_and_replacements(store)
    cached = _STORE_STATS_CACHE.get(id(base))
    if cached is not None and cached[0] is base:
        return cached[1]
    stats = base.stats()
    _STORE_STATS_CACHE[id(base)] = (base, stats)
    return stats


def _control_rows(
    store: GraphStore,
    item: QAItem | Mapping[str, Any],
) -> tuple[GraphRow, ...]:
    base, replacements = _base_store_and_replacements(store)
    meta = _item_meta(item)
    world_id = meta.get("world_id")
    if (
        isinstance(world_id, bool)
        or not isinstance(world_id, int)
        or world_id < 0
    ):
        raise ValueError("control items require a non-negative world_id")
    provenance = meta.get("provenance_id")
    if not isinstance(provenance, str) or not provenance:
        raise ValueError("control items require an exact provenance_id")
    provenance_rows = getattr(base, "rows_for_provenance", None)
    selected: tuple[GraphRow, ...]
    if callable(provenance_rows):
        selected = tuple(provenance_rows(provenance))
    else:
        selected = ()
    if not selected:
        rows = _rows_from_store(base)
        cached = _STORE_GROUP_CACHE.get(id(base))
        if cached is not None and cached[0] is base:
            groups = cached[1]
        else:
            mutable: dict[str, list[GraphRow]] = defaultdict(list)
            for row in rows:
                mutable[row.provenance_id].append(row)
            groups = {
                row_provenance: tuple(group)
                for row_provenance, group in mutable.items()
            }
            _STORE_GROUP_CACHE[id(base)] = (base, groups)
        selected = groups.get(provenance, ())
    if not selected:
        raise ValueError(
            f"store has no exact provenance partition {provenance}"
        )
    expected_rows = meta.get("graph_rows")
    if (
        isinstance(expected_rows, bool)
        or not isinstance(expected_rows, int)
        or expected_rows <= 0
        or expected_rows != len(selected)
    ):
        raise ValueError(
            "item graph_rows does not match its provenance partition"
        )
    materialized = tuple(
        replacements.get(row.address, row) for row in selected
    )
    if any(
        replacement.provenance_id != row.provenance_id
        or replacement.target_kind != row.target_kind
        for row, replacement in zip(selected, materialized)
    ):
        raise ValueError(
            "item overlays cannot cross provenance or target-kind partitions"
        )
    return materialized


class _PatchStore:
    """Lazy read-only replacements/deletions over a graph store."""

    def __init__(
        self,
        base: GraphStore,
        *,
        replacements: Mapping[GraphAddress, GraphRow] = MappingProxyType({}),
        deletions: Iterable[GraphAddress] = (),
    ) -> None:
        if not isinstance(base, GraphStore):
            raise TypeError("patch base must implement GraphStore")
        self.base = base
        self._replacements = dict(replacements)
        self._deletions = frozenset(deletions)
        if set(self._replacements) & self._deletions:
            raise ValueError("patch addresses cannot be replaced and deleted")
        for address, row in self._replacements.items():
            if (
                not isinstance(address, GraphAddress)
                or not isinstance(row, GraphRow)
                or row.address != address
            ):
                raise TypeError(
                    "patch replacements must map addresses to matching rows"
                )
            previous = base.lookup(address)
            if previous is None:
                raise ValueError("replacement address is absent from the store")
            if previous == row:
                raise ValueError("replacement row must change its payload")
        for address in self._deletions:
            if not isinstance(address, GraphAddress):
                raise TypeError("patch deletions must be graph addresses")
            if base.lookup(address) is None:
                raise ValueError("deleted address is absent from the store")
        self.hits = 0
        self.misses = 0

    def lookup(self, address: GraphAddress) -> GraphRow | None:
        if not isinstance(address, GraphAddress):
            raise TypeError("graph lookup requires a GraphAddress")
        if address in self._deletions:
            self.misses += 1
            return None
        row = self._replacements.get(address)
        if row is None:
            row = self.base.lookup(address)
        if row is None:
            self.misses += 1
        else:
            self.hits += 1
        return row

    def __len__(self) -> int:
        return len(self.base) - len(self._deletions)

    def snapshot_sha256(self) -> str:
        payload = {
            "base_snapshot_sha256": self.base.snapshot_sha256(),
            "replacements": [
                self._replacements[address].as_json()
                for address in sorted(self._replacements)
            ],
            "deletions": [
                _address_json(address)
                for address in sorted(self._deletions)
            ],
        }
        return hashlib.sha256(_canonical_bytes(payload)).hexdigest()

    def stats(self) -> StoreStats:
        stats = self.base.stats()
        return StoreStats(
            rows=len(self),
            index_bytes=stats.index_bytes,
            row_bytes=stats.row_bytes,
            blob_bytes=stats.blob_bytes,
        )

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0


class _ShuffledReturnStore:
    """Lazy seeded payload permutation within one world partition."""

    def __init__(
        self,
        base: GraphStore,
        donors: Mapping[GraphAddress, GraphAddress],
    ) -> None:
        if not isinstance(base, GraphStore):
            raise TypeError("shuffle base must implement GraphStore")
        self.base = base
        self._donors = dict(donors)
        if not self._donors or set(self._donors) != set(self._donors.values()):
            raise ValueError("shuffle donors must form an address permutation")
        if any(source == donor for source, donor in self._donors.items()):
            raise ValueError("shuffle donors must be a derangement")
        self.hits = 0
        self.misses = 0

    def lookup(self, address: GraphAddress) -> GraphRow | None:
        if not isinstance(address, GraphAddress):
            raise TypeError("graph lookup requires a GraphAddress")
        donor = self._donors.get(address)
        if donor is None:
            row = self.base.lookup(address)
        else:
            payload = self.base.lookup(donor)
            row = (
                None
                if payload is None
                else GraphRow(
                    source_id=address.source_id,
                    relation_id=address.relation_id,
                    direction=address.direction,
                    target_kind=payload.target_kind,
                    target=payload.target,
                    qualifiers=payload.qualifiers,
                    provenance_id=payload.provenance_id,
                )
            )
        if row is None:
            self.misses += 1
        else:
            self.hits += 1
        return row

    def __len__(self) -> int:
        return len(self.base)

    def snapshot_sha256(self) -> str:
        payload = {
            "base_snapshot_sha256": self.base.snapshot_sha256(),
            "donors": [
                [_address_json(source), _address_json(donor)]
                for source, donor in sorted(self._donors.items())
            ],
        }
        return hashlib.sha256(_canonical_bytes(payload)).hexdigest()

    def stats(self) -> StoreStats:
        return self.base.stats()

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0


class _EntityRemapStore:
    """Lazy entity-address bijection scoped to one world partition."""

    def __init__(
        self,
        base: GraphStore,
        mapping: Mapping[int, int],
        scoped_rows: Iterable[GraphRow],
    ) -> None:
        if not isinstance(base, GraphStore):
            raise TypeError("entity-remap base must implement GraphStore")
        self.base = base
        self._mapping = dict(mapping)
        self._inverse = {
            target: source for source, target in self._mapping.items()
        }
        if len(self._inverse) != len(self._mapping):
            raise ValueError("entity remapping must be injective")
        rows = tuple(scoped_rows)
        self._scoped_sources = frozenset(row.source_id for row in rows)
        if not self._scoped_sources.issubset(self._mapping):
            raise ValueError("entity remapping omits scoped source entities")
        self._scope_addresses = tuple(row.address for row in rows)
        self._scope_address_set = frozenset(self._scope_addresses)
        self.hits = 0
        self.misses = 0

    def lookup(self, address: GraphAddress) -> GraphRow | None:
        if not isinstance(address, GraphAddress):
            raise TypeError("graph lookup requires a GraphAddress")
        old_source = self._inverse.get(address.source_id)
        if old_source is not None:
            row = self.base.lookup(
                GraphAddress(
                    old_source,
                    address.relation_id,
                    address.direction,
                )
            )
            if row is not None and row.source_id in self._scoped_sources:
                row = _remap_row(row, self._mapping)
            else:
                row = None
        elif address.source_id in self._scoped_sources:
            row = None
        else:
            row = self.base.lookup(address)
        if row is None:
            self.misses += 1
        else:
            self.hits += 1
        return row

    def __len__(self) -> int:
        return len(self.base)

    def rows(self) -> tuple[GraphRow, ...]:
        return tuple(
            (
                _remap_row(row, self._mapping)
                if row.address in self._scope_address_set
                else row
            )
            for row in _rows_from_store(self.base)
        )

    def snapshot_sha256(self) -> str:
        payload = {
            "base_snapshot_sha256": self.base.snapshot_sha256(),
            "mapping": [
                [source, target]
                for source, target in sorted(self._mapping.items())
            ],
            "scope": [
                _address_json(address)
                for address in sorted(self._scope_addresses)
            ],
        }
        return hashlib.sha256(_canonical_bytes(payload)).hexdigest()

    def stats(self) -> StoreStats:
        return self.base.stats()

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0


def _replace_row(
    row: GraphRow,
    *,
    target: str | None = None,
    qualifiers: tuple[tuple[str, str], ...] | None = None,
    payload: GraphRow | None = None,
) -> GraphRow:
    if payload is not None:
        target = payload.target
        qualifiers = payload.qualifiers
        target_kind = payload.target_kind
        provenance_id = payload.provenance_id
    else:
        target_kind = row.target_kind
        provenance_id = row.provenance_id
    return GraphRow(
        source_id=row.source_id,
        relation_id=row.relation_id,
        direction=row.direction,
        target_kind=target_kind,
        target=row.target if target is None else target,
        qualifiers=row.qualifiers if qualifiers is None else qualifiers,
        provenance_id=provenance_id,
    )


def _store_with_replacement(
    store: GraphStore,
    replacement: GraphRow,
) -> _PatchStore:
    return _PatchStore(
        store,
        replacements={replacement.address: replacement},
    )


def _parse_gold_actions(item: QAItem | Mapping[str, Any]) -> tuple[GraphAction, ...]:
    raw_actions = _item_meta(item).get("gold_actions")
    if not isinstance(raw_actions, list) or len(raw_actions) != 6:
        raise ValueError("gold_actions must contain exactly six actions")
    required = {
        "source_slot",
        "relation_id",
        "direction",
        "read",
        "halt",
    }
    actions: list[GraphAction] = []
    for raw in raw_actions:
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("gold action fields do not match the contract")
        if not isinstance(raw["read"], bool) or not isinstance(
            raw["halt"], bool
        ):
            raise ValueError("gold action terminals must be Boolean")
        actions.append(GraphAction(**raw))
    halts = [index for index, action in enumerate(actions) if action.halt]
    if len(halts) > 1:
        raise ValueError("gold action trace allows at most one HALT")
    if halts:
        halt = halts[0]
        if halt == 0 or any(
            not action.read or action.halt for action in actions[:halt]
        ):
            raise ValueError("gold trace must contain reads before HALT")
        if any(
            action.read or action.halt for action in actions[halt + 1 :]
        ):
            raise ValueError("gold trace must contain NOOPs after HALT")
    elif any(not action.read or action.halt for action in actions):
        raise ValueError("a halt-free gold trace must contain six reads")
    return tuple(actions)


def _oracle_trace(
    item: QAItem | Mapping[str, Any],
    store: GraphStore,
) -> tuple[tuple[GraphAddress, ...], tuple[GraphRow, ...]] | None:
    meta = _item_meta(item)
    slots = meta.get("entity_slots")
    if (
        not isinstance(slots, list)
        or len(slots) != 4
        or any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            )
            for value in slots
        )
    ):
        raise ValueError("entity_slots must contain four entity ids or nulls")
    working = list(slots)
    addresses: list[GraphAddress] = []
    returned: list[GraphRow] = []
    for action in _parse_gold_actions(item):
        if not action.read:
            continue
        source_id = working[action.source_slot]
        if source_id is None:
            return None
        address = GraphAddress(
            source_id,
            action.relation_id,
            action.direction,
        )
        row = store.lookup(address)
        if row is None:
            return None
        addresses.append(address)
        returned.append(row)
        if row.target_kind == "entity":
            if not row.target.isascii() or not row.target.isdecimal():
                raise ValueError(
                    "entity returns must contain canonical non-negative ids"
                )
            working[action.source_slot] = int(row.target)
    return tuple(addresses), tuple(returned)


def oracle_answer(
    item: QAItem | Mapping[str, Any],
    store: GraphStore,
) -> str | None:
    """Evaluate the frozen task oracle against a concrete graph view."""

    if not isinstance(store, GraphStore):
        raise TypeError("oracle store must implement GraphStore")
    traced = _oracle_trace(item, store)
    if traced is None:
        return None
    _, rows = traced
    task = _item_value(item, "task")
    if task in {"path_composition", "factual_recall"}:
        try:
            value = sum(
                int(dict(row.qualifiers)["compose"]) for row in rows
            ) % 4
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "composition rows require integer compose qualifiers"
            ) from exc
        return f"r{value}"
    if task == "date_ordering":
        if len(rows) != 2 or any(row.target_kind != "literal" for row in rows):
            raise ValueError("date ordering requires two literal returns")
        return "<|slot_0|>" if rows[0].target < rows[1].target else "<|slot_1|>"
    if task == "balanced_equality":
        if len(rows) != 2 or any(row.target_kind != "literal" for row in rows):
            raise ValueError("balanced equality requires two literal returns")
        return "yes" if rows[0].target == rows[1].target else "no"
    raise ValueError(f"unsupported relational oracle task: {task}")


def _oracle_effect(before: str, after: str | None) -> str:
    if after is None:
        return "miss"
    return "unchanged" if before == after else "changed"


def _semantic_candidates(
    row: GraphRow,
    rows: tuple[GraphRow, ...],
    *,
    seed: int,
    label: str,
) -> tuple[GraphRow, ...]:
    candidates: dict[bytes, GraphRow] = {}

    def add(candidate: GraphRow) -> None:
        if candidate != row and candidate.address == row.address:
            candidates[_canonical_bytes(candidate.as_json())] = candidate

    qualifiers = dict(row.qualifiers)
    if "compose" in qualifiers:
        try:
            current = int(qualifiers["compose"])
        except ValueError as exc:
            raise ValueError("compose qualifiers must be integers") from exc
        for residue in range(4):
            value = current - current % 4 + residue
            changed = tuple(
                (key, str(value) if key == "compose" else item)
                for key, item in row.qualifiers
            )
            add(_replace_row(row, qualifiers=changed))

    for donor in rows:
        if donor.target_kind == row.target_kind:
            add(_replace_row(row, payload=donor))

    if row.target_kind == "literal":
        for target in (
            "!" + row.target,
            row.target + "~control-edit",
            "\uffff" + row.target,
            "0000-00-00",
            "9999-99-99",
        ):
            add(_replace_row(row, target=target))
    else:
        entity_ids = {
            candidate.source_id for candidate in rows
        } | {
            int(candidate.target)
            for candidate in rows
            if candidate.target_kind == "entity"
            and candidate.target.isascii()
            and candidate.target.isdecimal()
        }
        for entity_id in sorted(entity_ids):
            add(_replace_row(row, target=str(entity_id)))

    return _seeded_order(
        candidates.values(),
        seed=seed,
        label=label,
        serializer=lambda candidate: candidate.as_json(),
    )


def _edit_view(
    item: QAItem | Mapping[str, Any],
    store: GraphStore,
    rows: tuple[GraphRow, ...],
    *,
    seed: int,
    relevant: bool,
    before: str,
) -> tuple[GraphStore, tuple[GraphAddress, ...], str]:
    traced = _oracle_trace(item, store)
    if traced is None:
        raise ValueError("cannot edit an item whose oracle path misses")
    path_addresses = set(traced[0])
    eligible = (
        list(traced[1])
        if relevant
        else [
            row
            for row in rows
            if row.address not in path_addresses
        ]
    )
    ordered = _seeded_order(
        eligible,
        seed=seed,
        label="relevant-row" if relevant else "irrelevant-row",
        serializer=lambda row: row.as_json(),
    )
    for row in ordered:
        for replacement in _semantic_candidates(
            row,
            rows,
            seed=seed,
            label=(
                "relevant-payload"
                if relevant
                else "irrelevant-payload"
            ),
        ):
            candidate = _store_with_replacement(store, replacement)
            after = oracle_answer(item, candidate)
            if relevant and after is not None and after != before:
                return candidate, (row.address,), after
            if not relevant and after == before:
                return candidate, (row.address,), after
    kind = "relevant" if relevant else "irrelevant"
    raise ValueError(f"no valid one-row {kind} edit exists")


def _shuffled_store(
    store: GraphStore,
    rows: tuple[GraphRow, ...],
    *,
    seed: int,
) -> tuple[GraphStore, tuple[tuple[GraphAddress, GraphAddress], ...]]:
    by_kind: dict[str, list[GraphRow]] = defaultdict(list)
    for row in rows:
        by_kind[row.target_kind].append(row)
    donors: dict[GraphAddress, GraphAddress] = {}
    sources: list[tuple[GraphAddress, GraphAddress]] = []
    for kind in sorted(by_kind):
        kind_rows = by_kind[kind]
        if len(kind_rows) < 2:
            raise ValueError(
                f"seeded derangement undersupplied for target kind {kind}"
            )
        payload_groups: dict[tuple[Any, ...], list[GraphRow]] = defaultdict(
            list
        )
        for row in kind_rows:
            payload_groups[_row_payload(row)].append(row)
        maximum_count = max(map(len, payload_groups.values()))
        if maximum_count * 2 > len(kind_rows):
            raise ValueError(
                "no true payload derangement exists for target kind "
                f"{kind}"
            )
        ordered_payloads = sorted(
            payload_groups,
            key=lambda payload: (
                -len(payload_groups[payload]),
                _stable_key(
                    seed,
                    f"shuffle-payload-{kind}",
                    payload,
                ),
                _canonical_bytes(payload),
            ),
        )
        recipients = tuple(
            row
            for payload in ordered_payloads
            for row in _seeded_order(
                payload_groups[payload],
                seed=seed,
                label=f"shuffle-recipient-{kind}",
                serializer=lambda candidate: candidate.as_json(),
            )
        )
        donor_order = (
            recipients[maximum_count:] + recipients[:maximum_count]
        )
        for recipient, donor in zip(recipients, donor_order, strict=True):
            if _row_payload(recipient) == _row_payload(donor):
                raise AssertionError(
                    "bounded payload derangement produced a fixed payload"
                )
            donors[recipient.address] = donor.address
            sources.append((recipient.address, donor.address))
    return (
        _ShuffledReturnStore(store, donors),
        tuple(sorted(sources)),
    )


def _all_entity_ids(
    rows: tuple[GraphRow, ...],
    item: QAItem | Mapping[str, Any],
) -> tuple[int, ...]:
    entities = {row.source_id for row in rows}
    for row in rows:
        if row.target_kind == "entity":
            if not row.target.isascii() or not row.target.isdecimal():
                raise ValueError("entity targets must be canonical integers")
            entities.add(int(row.target))
    for value in _item_meta(item).get("entity_slots", []):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("entity slots must be non-negative integers")
            entities.add(value)
    return tuple(sorted(entities))


def _store_entity_ids(store: GraphStore) -> tuple[int, ...]:
    base, _ = _base_store_and_replacements(store)
    cached = _STORE_ENTITY_CACHE.get(id(base))
    if cached is not None and cached[0] is base:
        return cached[1]
    entities = _all_entity_ids(_rows_from_store(base), {"meta": {}})
    _STORE_ENTITY_CACHE[id(base)] = (base, entities)
    return entities


def _store_max_entity_id(store: GraphStore) -> int:
    base, _ = _base_store_and_replacements(store)
    method = getattr(base, "max_entity_id", None)
    if callable(method):
        value = method()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("store max entity id must be non-negative")
        return value
    return max(_store_entity_ids(base))


def _entity_mapping(
    entities: tuple[int, ...],
    *,
    seed: int,
    isomorphism: bool,
    reserved_max_entity: int = -1,
) -> dict[int, int]:
    if len(entities) < 2:
        raise ValueError("entity bijections require at least two entities")
    ordered = _seeded_order(
        entities,
        seed=seed,
        label="graph-isomorphism" if isomorphism else "entity-rename",
    )
    if isomorphism:
        shift = 1 + int.from_bytes(
            _stable_key(seed, "isomorphism-shift", len(ordered))[:8],
            "big",
        ) % (len(ordered) - 1)
        mapping = {
            value: ordered[(index + shift) % len(ordered)]
            for index, value in enumerate(ordered)
        }
    else:
        offset = (
            max(
                max(entities),
                reserved_max_entity,
            )
            + 1
            + int.from_bytes(
                _stable_key(seed, "rename-offset", list(entities))[:8],
                "big",
            )
        )
        mapping = {
            value: offset + index for index, value in enumerate(ordered)
        }
    if len(set(mapping.values())) != len(mapping):
        raise AssertionError("entity mapping is not injective")
    if any(source == target for source, target in mapping.items()):
        raise AssertionError("entity mapping contains a fixed point")
    return mapping


def _replace_binding_handles(
    prompt: str,
    slots: list[int | None],
    mapping: Mapping[int, int],
) -> str:
    matches = list(_BINDING_RE.finditer(prompt))
    if not matches:
        raise ValueError("prompt contains no canonical slot bindings")
    output: list[str] = []
    cursor = 0
    seen_slots: set[int] = set()
    for match in matches:
        slot = int(match.group(1))
        entity_id = slots[slot]
        if entity_id is None:
            raise ValueError("prompt binds a slot with no entity id")
        if entity_id not in mapping:
            raise ValueError("prompt entity is absent from the bijection")
        output.append(prompt[cursor : match.start(2)])
        output.append(f"Q{mapping[entity_id]}")
        cursor = match.end(2)
        seen_slots.add(slot)
    output.append(prompt[cursor:])
    if not seen_slots:
        raise ValueError("prompt entity bijection changed no handles")
    return "".join(output)


def _remap_row(row: GraphRow, mapping: Mapping[int, int]) -> GraphRow:
    return GraphRow(
        source_id=mapping[row.source_id],
        relation_id=row.relation_id,
        direction=row.direction,
        target_kind=row.target_kind,
        target=(
            str(mapping[int(row.target)])
            if row.target_kind == "entity"
            else row.target
        ),
        qualifiers=row.qualifiers,
        provenance_id=row.provenance_id,
    )


def _remap_item(
    item: QAItem | Mapping[str, Any],
    mapping: Mapping[int, int],
    remapped_store: GraphStore,
) -> QAItem | dict[str, Any]:
    transformed = _copy_item(item)
    meta = _item_meta(transformed)
    slots = meta.get("entity_slots")
    if not isinstance(slots, list) or len(slots) != 4:
        raise ValueError("entity_slots must contain exactly four values")
    original_slots = list(slots)
    meta["entity_slots"] = [
        None if value is None else mapping[value] for value in slots
    ]
    addresses = meta.get("gold_addresses")
    if not isinstance(addresses, list):
        raise ValueError("gold_addresses must be a list")
    meta["gold_addresses"] = [
        [mapping[int(source)], str(relation), direction]
        for source, relation, direction in addresses
    ]
    changed = meta.get("changed_row")
    if changed is not None:
        if not isinstance(changed, dict):
            raise ValueError("changed_row must be an object or null")
        meta["changed_row"] = _remap_row(
            GraphRow.from_json(changed),
            mapping,
        ).as_json()
    meta["gold_fact_ids"] = [
        stable_fact_id(row)
        for source, relation, direction in meta["gold_addresses"]
        if (
            row := remapped_store.lookup(
                GraphAddress(int(source), str(relation), direction)
            )
        )
        is not None
    ]
    if len(meta["gold_fact_ids"]) != len(meta["gold_addresses"]):
        raise ValueError("remapped gold addresses must all resolve")
    prompt = str(_item_value(transformed, "prompt"))
    _set_item_value(
        transformed,
        "prompt",
        _replace_binding_handles(prompt, original_slots, mapping),
    )
    return transformed


def _no_query_item(
    item: QAItem | Mapping[str, Any],
) -> QAItem | dict[str, Any]:
    transformed = _copy_item(item)
    prompt = str(_item_value(transformed, "prompt"))
    matches = list(_BINDING_RE.finditer(prompt))
    if not matches:
        raise ValueError("no-query control requires slot bindings")
    binding_prefix = prompt[: matches[-1].end()]
    if not prompt[matches[-1].end() :].strip():
        raise ValueError("no-query control requires a query to remove")
    _set_item_value(transformed, "prompt", binding_prefix + " ")
    return transformed


@dataclass(frozen=True)
class ControlView:
    item: QAItem | dict[str, Any]
    store: GraphStore
    control_id: ControlID
    seed: int
    changed_addresses: tuple[GraphAddress, ...]
    oracle_before: str
    oracle_after: str | None
    oracle_effect: str
    provenance_id: str
    oracle_addresses: tuple[GraphAddress, ...]
    oracle_rows: tuple[GraphRow, ...]
    oracle_fact_ids: tuple[str, ...]
    source_store_rows: int
    source_store_bytes: int
    forced_actions: tuple[GraphAction, ...] | None = None
    forced_returns: tuple[GraphRow | None, ...] | None = None
    return_sources: tuple[
        tuple[GraphAddress, GraphAddress], ...
    ] = ()
    entity_bijection: Mapping[int, int] = MappingProxyType({})

    @cached_property
    def _compact_transformation_record(
        self,
    ) -> Mapping[str, Any] | None:
        if self.control_id not in {
            ControlID.SHUFFLED_RETURNS,
            ControlID.ENTITY_RENAME,
            ControlID.GRAPH_ISOMORPHISM,
        }:
            return None
        changed_sha256 = hashlib.sha256(
            _canonical_bytes(
                [
                    _address_json(address)
                    for address in sorted(self.changed_addresses)
                ]
            )
        ).hexdigest()
        return_sources_sha256 = hashlib.sha256(
            _canonical_bytes(
                [
                    [_address_json(source), _address_json(donor)]
                    for source, donor in sorted(self.return_sources)
                ]
            )
        ).hexdigest()
        entity_bijection_sha256 = hashlib.sha256(
            _canonical_bytes(
                [
                    [source, target]
                    for source, target in sorted(
                        self.entity_bijection.items()
                    )
                ]
            )
        ).hexdigest()
        transformation_metadata = {
            "changed_address_count": len(self.changed_addresses),
            "changed_addresses_sha256": changed_sha256,
            "return_sources_sha256": return_sources_sha256,
            "entity_bijection_sha256": entity_bijection_sha256,
        }
        payload = {
            "record_type": "control_transformation",
            "schema_version": 1,
            "control_id": self.control_id.value,
            "seed": self.seed,
            "provenance_id": self.provenance_id,
            "source_store_sha256": (
                getattr(self.store, "base", self.store).snapshot_sha256()
            ),
            "transformed_store_sha256": self.store.snapshot_sha256(),
            **transformation_metadata,
            "transformation_metadata_sha256": hashlib.sha256(
                _canonical_bytes(transformation_metadata)
            ).hexdigest(),
        }
        return MappingProxyType(
            {
                **payload,
                "transformation_id": hashlib.sha256(
                    _canonical_bytes(payload)
                ).hexdigest(),
            }
        )

    def transformation_record(self) -> dict[str, Any] | None:
        record = self._compact_transformation_record
        return None if record is None else dict(record)

    @property
    def transformation_id(self) -> str | None:
        record = self._compact_transformation_record
        return None if record is None else str(record["transformation_id"])

    def fingerprint(self) -> str:
        payload = {
            "control_id": self.control_id.value,
            "seed": self.seed,
            "qid": str(_item_value(self.item, "qid")),
            "prompt": str(_item_value(self.item, "prompt")),
            "store_sha256": self.store.snapshot_sha256(),
            "changed_addresses": [
                _address_json(value) for value in self.changed_addresses
            ],
            "oracle_before": self.oracle_before,
            "oracle_after": self.oracle_after,
            "oracle_effect": self.oracle_effect,
            "provenance_id": self.provenance_id,
            "oracle_addresses": [
                _address_json(value) for value in self.oracle_addresses
            ],
            "oracle_fact_ids": list(self.oracle_fact_ids),
            "forced_actions": (
                None
                if self.forced_actions is None
                else [
                    [
                        action.source_slot,
                        action.relation_id,
                        action.direction,
                        action.read,
                        action.halt,
                    ]
                    for action in self.forced_actions
                ]
            ),
            "forced_returns": (
                None
                if self.forced_returns is None
                else [
                    None if row is None else row.as_json()
                    for row in self.forced_returns
                ]
            ),
            "return_sources": [
                [_address_json(source), _address_json(donor)]
                for source, donor in self.return_sources
            ],
            "entity_bijection": [
                [source, target]
                for source, target in sorted(self.entity_bijection.items())
            ],
        }
        return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def build_control_view(
    item: QAItem | Mapping[str, Any],
    store: GraphStore,
    control_id: ControlID | str,
    seed: int,
    *,
    transformation_cache: MutableMapping[tuple, tuple] | None = None,
) -> ControlView:
    """Build one fail-closed control view without mutating its inputs."""

    seed = _validate_seed(seed)
    try:
        control = ControlID(control_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown relational control: {control_id!r}") from exc
    if not isinstance(store, GraphStore):
        raise TypeError("control store must implement GraphStore")
    if transformation_cache is not None and not isinstance(
        transformation_cache, MutableMapping
    ):
        raise TypeError("transformation_cache must be mutable mapping")
    source_stats = _source_store_stats(store)
    source_store_bytes = (
        source_stats.index_bytes
        + source_stats.row_bytes
        + source_stats.blob_bytes
    )
    original_item = _copy_item(item)
    original_trace = _oracle_trace(original_item, store)
    if original_trace is None:
        raise ValueError("item gold trace does not resolve before the control")
    before = oracle_answer(original_item, store)
    expected = str(_item_value(original_item, "answer"))
    if before is None or before != expected:
        raise ValueError("item answer does not match the pre-control oracle")

    transformed_item = original_item
    transformed_store: GraphStore = store
    changed_addresses: tuple[GraphAddress, ...] = ()
    forced_actions: tuple[GraphAction, ...] | None = None
    forced_returns: tuple[GraphRow | None, ...] | None = None
    return_sources: tuple[tuple[GraphAddress, GraphAddress], ...] = ()
    entity_bijection: Mapping[int, int] = MappingProxyType({})
    rows = _control_rows(store, original_item)

    if control == ControlID.SHUFFLED_RETURNS:
        cache_key = (
            control.value,
            seed,
            str(_item_meta(original_item)["provenance_id"]),
            store.snapshot_sha256(),
        )
        cached = (
            transformation_cache.get(cache_key)
            if transformation_cache is not None
            else None
        )
        if cached is None:
            transformed_store, return_sources = _shuffled_store(
                store,
                rows,
                seed=seed,
            )
            changed_addresses = tuple(row.address for row in rows)
            if transformation_cache is not None:
                transformation_cache[cache_key] = (
                    transformed_store,
                    return_sources,
                    changed_addresses,
                )
        else:
            (
                transformed_store,
                return_sources,
                changed_addresses,
            ) = cached
    elif control == ControlID.RELEVANT_EDGE:
        transformed_store, changed_addresses, _ = _edit_view(
            original_item,
            store,
            rows,
            seed=seed,
            relevant=True,
            before=before,
        )
    elif control == ControlID.IRRELEVANT_EDGE:
        transformed_store, changed_addresses, _ = _edit_view(
            original_item,
            store,
            rows,
            seed=seed,
            relevant=False,
            before=before,
        )
    elif control == ControlID.GOLD_PATH:
        forced_actions = _parse_gold_actions(original_item)
        traced = _oracle_trace(original_item, store)
        if traced is None:
            raise ValueError("gold-path control requires a complete path")
        returned = iter(traced[1])
        forced_returns = tuple(
            next(returned) if action.read else None
            for action in forced_actions
        )
        try:
            next(returned)
        except StopIteration:
            pass
        else:
            raise AssertionError("gold return trace exceeds action budget")
    elif control == ControlID.GOLD_RETURNS:
        gold_actions = _parse_gold_actions(original_item)
        traced = _oracle_trace(original_item, store)
        if traced is None:
            raise ValueError("gold-return control requires a complete path")
        returned = iter(traced[1])
        forced_returns = tuple(
            next(returned) if action.read else None
            for action in gold_actions
        )
        try:
            next(returned)
        except StopIteration:
            pass
        else:
            raise AssertionError("gold return trace exceeds action budget")
        forced_actions = None
    elif control == ControlID.NO_QUERY:
        transformed_item = _no_query_item(original_item)
    elif control == ControlID.EXPLICIT_MISS:
        traced = _oracle_trace(original_item, store)
        if traced is None:
            raise ValueError("explicit-miss control requires a complete path")
        unique_addresses = _seeded_order(
            set(traced[0]),
            seed=seed,
            label="explicit-miss",
            serializer=_address_json,
        )
        if not unique_addresses:
            raise ValueError("explicit-miss control requires a read address")
        removed = unique_addresses[0]
        transformed_store = _PatchStore(
            store,
            deletions=(removed,),
        )
        changed_addresses = (removed,)
    elif control == ControlID.HANDLE_SWAP:
        entities = _all_entity_ids(rows, original_item)
        handle_mapping = _entity_mapping(
            entities,
            seed=seed,
            isomorphism=True,
        )
        transformed_item = _copy_item(original_item)
        meta = _item_meta(transformed_item)
        slots = meta["entity_slots"]
        _set_item_value(
            transformed_item,
            "prompt",
            _replace_binding_handles(
                str(_item_value(transformed_item, "prompt")),
                slots,
                handle_mapping,
            ),
        )
    elif control in {
        ControlID.ENTITY_RENAME,
        ControlID.GRAPH_ISOMORPHISM,
    }:
        entities = _all_entity_ids(rows, original_item)
        cache_key = (
            control.value,
            seed,
            str(_item_meta(original_item)["provenance_id"]),
            store.snapshot_sha256(),
            tuple(entities),
        )
        cached = (
            transformation_cache.get(cache_key)
            if transformation_cache is not None
            else None
        )
        if cached is None:
            mapping = _entity_mapping(
                entities,
                seed=seed,
                isomorphism=control == ControlID.GRAPH_ISOMORPHISM,
                reserved_max_entity=_store_max_entity_id(store),
            )
            scoped_addresses = {row.address for row in rows}
            for row in rows:
                remapped_address = _remap_row(row, mapping).address
                collision = store.lookup(remapped_address)
                if (
                    collision is not None
                    and remapped_address not in scoped_addresses
                ):
                    raise ValueError(
                        "entity remapping collides with an unchanged graph row"
                    )
            transformed_store = _EntityRemapStore(
                store,
                mapping,
                rows,
            )
            changed_addresses = tuple(row.address for row in rows)
            entity_bijection = MappingProxyType(
                dict(sorted(mapping.items()))
            )
            if transformation_cache is not None:
                transformation_cache[cache_key] = (
                    transformed_store,
                    entity_bijection,
                    changed_addresses,
                )
        else:
            (
                transformed_store,
                entity_bijection,
                changed_addresses,
            ) = cached
        mapping = entity_bijection
        transformed_item = _remap_item(
            original_item,
            mapping,
            transformed_store,
        )

    transformed_trace = _oracle_trace(transformed_item, transformed_store)
    if transformed_trace is None:
        oracle_addresses: tuple[GraphAddress, ...] = ()
        oracle_rows: tuple[GraphRow, ...] = ()
    else:
        oracle_addresses, oracle_rows = transformed_trace
        if transformed_trace != original_trace:
            transformed_meta = _item_meta(transformed_item)
            transformed_meta["gold_addresses"] = [
                _address_json(address) for address in oracle_addresses
            ]
            transformed_meta["gold_fact_ids"] = [
                stable_fact_id(row) for row in oracle_rows
            ]
    oracle_fact_ids = tuple(stable_fact_id(row) for row in oracle_rows)
    after = oracle_answer(transformed_item, transformed_store)
    effect = _oracle_effect(before, after)
    expected_effects = {
        ControlID.CORRECT: "unchanged",
        ControlID.RELEVANT_EDGE: "changed",
        ControlID.IRRELEVANT_EDGE: "unchanged",
        ControlID.GOLD_PATH: "unchanged",
        ControlID.GOLD_RETURNS: "unchanged",
        ControlID.NO_QUERY: "unchanged",
        ControlID.EXPLICIT_MISS: "miss",
        ControlID.HANDLE_SWAP: "unchanged",
        ControlID.ENTITY_RENAME: "unchanged",
        ControlID.GRAPH_ISOMORPHISM: "unchanged",
    }
    expected_effect = expected_effects.get(control)
    if expected_effect is not None and effect != expected_effect:
        raise ValueError(
            f"{control.value} produced oracle effect {effect}, "
            f"expected {expected_effect}"
        )
    return ControlView(
        item=transformed_item,
        store=transformed_store,
        control_id=control,
        seed=seed,
        changed_addresses=changed_addresses,
        oracle_before=before,
        oracle_after=after,
        oracle_effect=effect,
        provenance_id=str(_item_meta(transformed_item)["provenance_id"]),
        oracle_addresses=oracle_addresses,
        oracle_rows=oracle_rows,
        oracle_fact_ids=oracle_fact_ids,
        source_store_rows=source_stats.rows,
        source_store_bytes=source_store_bytes,
        forced_actions=forced_actions,
        forced_returns=forced_returns,
        return_sources=return_sources,
        entity_bijection=entity_bijection,
    )
