"""Independent replay checks for emitted Wikidata path twins."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

from corpusgen.graph_records import GraphAddress, GraphRow, stable_fact_id
from evals.relational_generate import OverlayStore
from organizer.packed_graph_store import PackedGraphStore


_QID = re.compile(r"Q(0|[1-9][0-9]*)\Z")


def _meta(item: Mapping[str, Any]) -> Mapping[str, Any]:
    value = item.get("meta")
    if not isinstance(value, Mapping):
        raise ValueError("path item metadata must be a mapping")
    return value


def _qid(value: object, name: str) -> int:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a Q ID")
    match = _QID.fullmatch(value)
    if match is None:
        raise ValueError(f"{name} must be a Q ID")
    return int(match.group(1))


def _replay(rows: Sequence[GraphRow]) -> int:
    if not rows:
        raise ValueError("cannot replay an empty Wikidata path")
    endpoint = rows[0].source_id
    for row in rows:
        if (
            row.source_id != endpoint
            or row.direction != "out"
            or row.target_kind != "entity"
            or row.qualifiers
        ):
            raise ValueError("Wikidata path rows do not replay exactly")
        try:
            endpoint = int(row.target)
        except ValueError as exc:
            raise ValueError("Wikidata entity target must be numeric") from exc
        if endpoint < 0 or str(endpoint) != row.target:
            raise ValueError("Wikidata entity target is not canonical")
    return endpoint


def _assert_item_rows(
    item: Mapping[str, Any],
    rows: Sequence[GraphRow],
    *,
    changed: GraphRow | None,
) -> None:
    meta = _meta(item)
    expected_addresses = [
        [row.source_id, row.relation_id, row.direction] for row in rows
    ]
    expected_facts = [stable_fact_id(row) for row in rows]
    if meta.get("gold_addresses") != expected_addresses:
        raise ValueError("declared path addresses do not match replay rows")
    if meta.get("gold_fact_ids") != expected_facts:
        raise ValueError("declared path facts do not match replay rows")
    if meta.get("changed_row") != (
        None if changed is None else changed.as_json()
    ):
        raise ValueError("declared overlay does not match replay row")
    if meta.get("counterfactual_changed_rows") != int(changed is not None):
        raise ValueError("declared counterfactual row count is invalid")


def replay_and_validate_path_twins(
    original_rows: Sequence[GraphRow],
    original_items: Sequence[Mapping[str, Any]],
    counterfactual_items: Sequence[Mapping[str, Any]],
) -> None:
    """Replay both path variants and verify all declared oracles before write."""

    rows = tuple(original_rows)
    if len(rows) not in range(1, 7):
        raise ValueError("Wikidata replay requires one through six rows")
    addresses = tuple(row.address for row in rows)
    if addresses[-1] in addresses[:-1]:
        raise ValueError("counterfactual changed address repeats in prefix")
    if len(set(addresses)) != len(addresses):
        raise ValueError("Wikidata path address repeats")
    original_endpoint = _replay(rows)

    original_by_pair = {
        str(_meta(item).get("pair_id")): item for item in original_items
    }
    counterfactual_by_pair = {
        str(_meta(item).get("pair_id")): item
        for item in counterfactual_items
    }
    if (
        len(original_items) != 2
        or len(counterfactual_items) != 2
        or len(original_by_pair) != 2
        or set(original_by_pair) != set(counterfactual_by_pair)
    ):
        raise ValueError("path replay requires exactly two matching task pairs")

    changed_rows = []
    for item in counterfactual_items:
        raw_changed = _meta(item).get("changed_row")
        if not isinstance(raw_changed, dict):
            raise ValueError("counterfactual item requires one changed row")
        changed_rows.append(GraphRow.from_json(raw_changed))
    changed = changed_rows[0]
    if any(row != changed for row in changed_rows[1:]):
        raise ValueError("counterfactual task twins disagree on changed row")
    final = rows[-1]
    if (
        changed.address != final.address
        or changed.target_kind != final.target_kind
        or changed.qualifiers != final.qualifiers
        or changed.target == final.target
    ):
        raise ValueError("overlay must change exactly the final returned row")
    counterfactual_rows = (*rows[:-1], changed)
    counterfactual_endpoint = _replay(counterfactual_rows)
    if original_endpoint == counterfactual_endpoint:
        raise ValueError("counterfactual endpoint must change")

    tasks = set()
    for pair_id, original in original_by_pair.items():
        counterfactual = counterfactual_by_pair[pair_id]
        original_meta = _meta(original)
        counterfactual_meta = _meta(counterfactual)
        task = original.get("task")
        if (
            task not in {"endpoint_traversal", "endpoint_equality"}
            or counterfactual.get("task") != task
            or original_meta.get("task") != task
            or counterfactual_meta.get("task") != task
        ):
            raise ValueError("path task pair metadata is inconsistent")
        tasks.add(task)
        if (
            original_meta.get("variant") != "original"
            or counterfactual_meta.get("variant") != "counterfactual"
            or original.get("prompt") != counterfactual.get("prompt")
        ):
            raise ValueError("path variants do not form a prompt twin")
        _assert_item_rows(original, rows, changed=None)
        _assert_item_rows(
            counterfactual,
            counterfactual_rows,
            changed=changed,
        )
        for meta, endpoint in (
            (original_meta, original_endpoint),
            (counterfactual_meta, counterfactual_endpoint),
        ):
            if _qid(meta.get("oracle_endpoint"), "oracle endpoint") != endpoint:
                raise ValueError("declared oracle endpoint does not replay")
            if (
                _qid(meta.get("original_endpoint"), "original endpoint")
                != original_endpoint
                or _qid(
                    meta.get("counterfactual_endpoint"),
                    "counterfactual endpoint",
                )
                != counterfactual_endpoint
            ):
                raise ValueError("declared twin endpoints do not replay")

        if task == "endpoint_traversal":
            expected_original = f"Q{original_endpoint}"
            expected_counterfactual = f"Q{counterfactual_endpoint}"
        else:
            comparison = _qid(
                original_meta.get("comparison_entity"),
                "comparison entity",
            )
            if (
                counterfactual_meta.get("comparison_entity")
                != original_meta.get("comparison_entity")
            ):
                raise ValueError("equality twins use different comparisons")
            expected_original = (
                "yes" if original_endpoint == comparison else "no"
            )
            expected_counterfactual = (
                "yes" if counterfactual_endpoint == comparison else "no"
            )
        if (
            original.get("answer") != expected_original
            or counterfactual.get("answer") != expected_counterfactual
            or expected_original == expected_counterfactual
        ):
            raise ValueError("declared twin answers do not flip under replay")
    if tasks != {"endpoint_traversal", "endpoint_equality"}:
        raise ValueError("path replay is missing a required task")


def _iter_jsonl_items(path: Path) -> Iterator[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            f"materialized path artifact is not a regular file: {path}"
        )
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid materialized path JSON at line {line_number}"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError("materialized path item must be an object")
            yield item


def _lookup_materialized_path(
    store,
    item: Mapping[str, Any],
    *,
    changed: GraphRow | None,
) -> tuple[tuple[GraphRow, ...], int]:
    meta = _meta(item)
    slots = meta.get("entity_slots")
    addresses = meta.get("gold_addresses")
    if (
        not isinstance(slots, list)
        or not slots
        or isinstance(slots[0], bool)
        or not isinstance(slots[0], int)
        or slots[0] < 0
        or not isinstance(addresses, list)
        or not 1 <= len(addresses) <= 6
    ):
        raise ValueError("materialized path metadata is invalid")

    endpoint = slots[0]
    rows = []
    for raw_address in addresses:
        if (
            not isinstance(raw_address, list)
            or len(raw_address) != 3
            or isinstance(raw_address[0], bool)
            or not isinstance(raw_address[0], int)
            or not isinstance(raw_address[1], str)
            or raw_address[2] not in {"out", "in"}
        ):
            raise ValueError("materialized path address is invalid")
        declared = GraphAddress(
            raw_address[0],
            raw_address[1],
            raw_address[2],
        )
        actual = GraphAddress(endpoint, declared.relation_id, declared.direction)
        if actual != declared:
            raise ValueError(
                "materialized lookup endpoint diverges from declared path"
            )
        row = store.lookup(actual)
        if row is None or row.address != actual:
            raise ValueError("materialized lookup did not return the exact row")
        if row.target_kind != "entity" or row.qualifiers:
            raise ValueError("materialized lookup returned an unsupported row")
        try:
            next_endpoint = int(row.target)
        except ValueError as exc:
            raise ValueError(
                "materialized lookup returned a nonnumeric entity"
            ) from exc
        if next_endpoint < 0 or str(next_endpoint) != row.target:
            raise ValueError("materialized lookup returned a noncanonical entity")
        rows.append(row)
        endpoint = next_endpoint

    materialized_rows = tuple(rows)
    _assert_item_rows(item, materialized_rows, changed=changed)
    if _qid(meta.get("oracle_endpoint"), "oracle endpoint") != endpoint:
        raise ValueError("materialized endpoint does not match declared oracle")
    task = item.get("task")
    if task == "endpoint_traversal":
        expected_answer = f"Q{endpoint}"
    elif task == "endpoint_equality":
        comparison = _qid(
            meta.get("comparison_entity"),
            "comparison entity",
        )
        expected_answer = "yes" if endpoint == comparison else "no"
    else:
        raise ValueError("materialized path has an unsupported task")
    if item.get("answer") != expected_answer:
        raise ValueError("materialized endpoint does not match declared answer")
    return materialized_rows, endpoint


def audit_materialized_path_artifacts(
    store: PackedGraphStore,
    original_path: str | Path,
    counterfactual_path: str | Path,
) -> int:
    """Stream every item through serialized base and one-row overlay stores."""

    if not isinstance(store, PackedGraphStore):
        raise TypeError("materialized audit requires a PackedGraphStore")
    sentinel = object()
    audited_items = 0
    originals = _iter_jsonl_items(Path(original_path))
    counterfactuals = _iter_jsonl_items(Path(counterfactual_path))
    for original, counterfactual in zip_longest(
        originals,
        counterfactuals,
        fillvalue=sentinel,
    ):
        try:
            if original is sentinel or counterfactual is sentinel:
                raise ValueError(
                    "original and counterfactual artifact counts differ"
                )
            if not isinstance(original, dict) or not isinstance(
                counterfactual,
                dict,
            ):
                raise TypeError("materialized path twins must be objects")
            original_meta = _meta(original)
            counterfactual_meta = _meta(counterfactual)
            if (
                original_meta.get("variant") != "original"
                or counterfactual_meta.get("variant") != "counterfactual"
                or original_meta.get("pair_id")
                != counterfactual_meta.get("pair_id")
                or original.get("task") != counterfactual.get("task")
                or original.get("prompt") != counterfactual.get("prompt")
            ):
                raise ValueError("materialized path twins are inconsistent")

            original_rows, original_endpoint = _lookup_materialized_path(
                store,
                original,
                changed=None,
            )
            raw_changed = counterfactual_meta.get("changed_row")
            if not isinstance(raw_changed, dict):
                raise ValueError(
                    "materialized counterfactual requires one changed row"
                )
            changed = GraphRow.from_json(raw_changed)
            if changed.address != original_rows[-1].address:
                raise ValueError(
                    "materialized overlay does not target the final address"
                )
            base_before = store.lookup(changed.address)
            if base_before != original_rows[-1]:
                raise ValueError(
                    "materialized base changed address does not match original"
                )
            overlay = OverlayStore(store, changed)
            if overlay.lookup(changed.address) != changed:
                raise ValueError(
                    "materialized overlay did not replace the changed address"
                )
            if store.lookup(changed.address) != base_before:
                raise ValueError("materialized overlay mutated the base store")
            counterfactual_rows, counterfactual_endpoint = (
                _lookup_materialized_path(
                    overlay,
                    counterfactual,
                    changed=changed,
                )
            )
            if (
                counterfactual_rows[:-1] != original_rows[:-1]
                or counterfactual_rows[-1] != changed
                or original_endpoint == counterfactual_endpoint
                or original.get("answer") == counterfactual.get("answer")
            ):
                raise ValueError(
                    "materialized overlay does not produce an answer-flipping twin"
                )
            for meta in (original_meta, counterfactual_meta):
                if (
                    _qid(meta.get("original_endpoint"), "original endpoint")
                    != original_endpoint
                    or _qid(
                        meta.get("counterfactual_endpoint"),
                        "counterfactual endpoint",
                    )
                    != counterfactual_endpoint
                ):
                    raise ValueError(
                        "materialized twin endpoint declarations do not replay"
                    )
            audited_items += 2
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "materialized packed store audit failed"
            ) from exc
    return audited_items
