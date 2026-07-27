from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import nullcontext
from itertools import zip_longest
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np

from corpusgen.graph_records import (
    GraphRow,
    relative_position_bin,
    stable_fact_id,
)
from corpusgen.mask_ledger import verify_weight_sidecars
from corpusgen.payload_inventory import PayloadInventory
from corpusgen.relation_codec import RelationCodec
from corpusgen.world_splits import (
    ObservedSplitArtifacts,
    ReasoningArtifactSignature,
    SplitArtifactExpectations,
    WorldArtifactSignature,
    audit_disjointness,
    require_disjointness,
)
from organizer.packed_graph_store import PackedGraphStore


EXPECTATION_NAME = "published-artifact-expectations.json"
AUDIT_NAME = "published-artifact-audit.json"
_EXCLUDED = frozenset(
    ("manifest.json", "split-audit.json", EXPECTATION_NAME, AUDIT_NAME)
)
_SPLIT_NAMES = (
    "development",
    "train",
    "protected_seen",
    "protected_heldout",
)
_REPLAY_SPAN_ROLES = frozenset(
    {
        "plain",
        "relation_alias",
        "random_control",
        "rule",
        "action",
        "provisional_answer",
        "final_answer",
        "boundary",
        "payload",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and relative not in _EXCLUDED:
            yield path


def _jsonl_signature(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    rows = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid published JSONL row: {path.name}:{line_number}"
                ) from exc
            digest.update(_canonical_bytes(row))
            rows += 1
    return {"rows": rows, "canonical_sha256": digest.hexdigest()}


def _jsonl_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid split plan JSONL: {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError("split plan rows must be objects")
            yield row


def _world_signature(world) -> WorldArtifactSignature:
    canonical = [
        {
            "address": {
                "source_id": fact.row.source_id,
                "relation_id": fact.row.relation_id,
                "direction": fact.row.direction,
            },
            "row": fact.row.as_json(),
        }
        for fact in world.facts
    ]
    digest = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return WorldArtifactSignature(
        world_id=world.world_id,
        world_seed=world.world_seed,
        fact_ids=tuple(fact.fact_id for fact in world.facts),
        row_address_sha256=digest,
        fact_count=len(world.facts),
    )


def _reasoning_signature(
    artifact_id: str,
    metadata: dict[str, object],
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


def _capture_split_expectations(
    name: str,
    worlds,
    qa_items,
    rendered_records,
) -> SplitArtifactExpectations:
    return SplitArtifactExpectations(
        name=name,
        world_signatures=tuple(_world_signature(world) for world in worlds),
        qa_signatures=tuple(
            _reasoning_signature(item.qid, item.meta) for item in qa_items
        ),
        rendered_signatures=tuple(
            _reasoning_signature(
                record.schedule.record_id,
                record.metadata,
            )
            for record in rendered_records
        ),
        required_hops=(
            frozenset(range(1, 7))
            if name.startswith("protected_")
            else frozenset(range(1, 5))
        ),
    )


def _dedicated_split_snapshot(root: Path, name: str):
    plan_dir = root / "split-plans" / name
    worlds = []
    for raw in _jsonl_rows(plan_dir / "worlds.jsonl"):
        facts = tuple(
            SimpleNamespace(
                fact_id=str(fact["fact_id"]),
                row=GraphRow.from_json(fact["row"]),
            )
            for fact in raw["facts"]
        )
        worlds.append(
            SimpleNamespace(
                world_id=raw["world_id"],
                world_seed=raw["world_seed"],
                split_name=raw["split_name"],
                manifest=raw["manifest"],
                facts=facts,
            )
        )
    qa_items = tuple(
        SimpleNamespace(qid=raw["qid"], meta=raw["meta"])
        for raw in _jsonl_rows(plan_dir / "qa.jsonl")
    )
    rendered = tuple(
        SimpleNamespace(
            schedule=SimpleNamespace(
                component="reasoning",
                record_id=raw["record_id"],
            ),
            metadata=raw["metadata"],
        )
        for raw in _jsonl_rows(plan_dir / "rendered.jsonl")
    )
    return tuple(worlds), qa_items, rendered


def _production_worlds(
    graph_path: Path,
    world_plan_path: Path,
    name: str,
):
    facts_by_provenance: dict[str, list[object]] = {}
    for raw in _jsonl_rows(graph_path):
        row = GraphRow.from_json(raw)
        facts_by_provenance.setdefault(row.provenance_id, []).append(
            SimpleNamespace(fact_id=stable_fact_id(row), row=row)
        )
    worlds = []
    used: set[str] = set()
    for raw in _jsonl_rows(world_plan_path):
        provenance = raw.get("provenance_id")
        if (
            not isinstance(provenance, str)
            or provenance in used
            or provenance not in facts_by_provenance
        ):
            raise ValueError("production world plan provenance mismatch")
        used.add(provenance)
        worlds.append(
            SimpleNamespace(
                world_id=raw["world_id"],
                world_seed=raw["world_seed"],
                split_name=name,
                manifest=raw["manifest"],
                facts=tuple(facts_by_provenance[provenance]),
            )
        )
    if used != set(facts_by_provenance):
        raise ValueError("production graph has unplanned world rows")
    return tuple(worlds)


def _qa_items(*paths: Path):
    return tuple(
        SimpleNamespace(qid=raw["qid"], meta=raw["meta"])
        for path in paths
        for raw in _jsonl_rows(path)
    )


def _rendered_from_qa(items):
    return tuple(
        SimpleNamespace(
            schedule=SimpleNamespace(
                component="reasoning",
                record_id=item.qid,
            ),
            metadata={**item.meta, "question_id": item.qid},
        )
        for item in items
    )


def _production_split_snapshots(
    root: Path,
    *,
    verify_replay_evidence: bool = True,
):
    _verify_schedule_plan_replay(
        root,
        verify_emitted_evidence=verify_replay_evidence,
    )
    development = _dedicated_split_snapshot(root, "development")
    dedicated_train = _dedicated_split_snapshot(root, "train")
    train_worlds = _production_worlds(
        root / "graph.jsonl",
        root
        / "split-plans"
        / "production"
        / "train-worlds.jsonl",
        "train",
    )
    train_rendered = tuple(
        SimpleNamespace(
            schedule=SimpleNamespace(
                component="reasoning",
                record_id=raw["record_id"],
            ),
            metadata=raw["metadata"],
        )
        for raw in _jsonl_rows(root / "schedule.jsonl")
        if raw.get("component") == "reasoning"
    )
    snapshots = {
        "development": development,
        "train": (train_worlds, dedicated_train[1], train_rendered),
    }
    for name in ("protected_seen", "protected_heldout"):
        eval_dir = root / "eval" / name
        items = _qa_items(
            eval_dir / "original.jsonl",
            eval_dir / "counterfactual.jsonl",
        )
        snapshots[name] = (
            _production_worlds(
                eval_dir / "graph.jsonl",
                eval_dir / "worlds.jsonl",
                name,
            ),
            items,
            _rendered_from_qa(items),
        )
    return snapshots


def _verify_schedule_plan_replay(
    root: Path,
    *,
    verify_emitted_evidence: bool = True,
) -> None:
    plan_path = root / "schedule-plan.sqlite3"
    occurrence_path = root / "mask-occurrences.sqlite3"
    plan_connection = None
    occurrence_connection = None
    try:
        plan_connection = sqlite3.connect(
            f"{plan_path.resolve().as_uri()}?mode=ro",
            uri=True,
        )
        if verify_emitted_evidence:
            occurrence_connection = sqlite3.connect(
                f"{occurrence_path.resolve().as_uri()}?mode=ro",
                uri=True,
            )
    except sqlite3.Error as exc:
        if plan_connection is not None:
            plan_connection.close()
        raise ValueError("invalid frozen schedule plan") from exc
    try:
        if dict(
            plan_connection.execute("SELECT key, value FROM metadata")
        ) != {"sealed": 1}:
            raise ValueError("frozen schedule plan is not sealed")
        if occurrence_connection is not None:
            if dict(
                occurrence_connection.execute(
                    "SELECT key, value FROM metadata"
                )
            ) != {"schema_version": "2", "finalized": "1"}:
                raise ValueError("replay span table is not finalized")
        planned = plan_connection.execute(
            """
            SELECT
              record_index, component, token_start, token_end,
              token_bytes, spans_json, schedule_json
            FROM records
            ORDER BY record_index
            """
        )
        replay_spans = (
            iter(
                occurrence_connection.execute(
                    """
                    SELECT
                      span_id, record_index, component, start, end, length,
                      position_bin, role, fact_id, payload_field, payload_text
                    FROM spans
                    ORDER BY span_id
                    """
                )
            )
            if occurrence_connection is not None
            else None
        )
        emitted = _jsonl_rows(root / "schedule.jsonl")
        sentinel = object()
        expected_record_index = 0
        expected_token_start = 0
        expected_span_id = 1
        train_context = (
            (root / "train.bin").open("rb")
            if verify_emitted_evidence
            else nullcontext()
        )
        with train_context as train:
            for planned_record, emitted_row in zip_longest(
                planned,
                emitted,
                fillvalue=sentinel,
            ):
                if (
                    planned_record is sentinel
                    or emitted_row is sentinel
                ):
                    raise ValueError(
                        "published schedule differs from frozen "
                        "disk-backed plan"
                    )
                (
                    record_index,
                    component,
                    token_start,
                    token_end,
                    token_bytes,
                    spans_json,
                    schedule_json,
                ) = planned_record
                if (
                    record_index != expected_record_index
                    or token_start != expected_token_start
                    or not isinstance(component, str)
                    or not component
                    or token_end <= token_start
                ):
                    raise ValueError("invalid frozen schedule record offsets")
                planned_schedule = json.loads(schedule_json)
                if (
                    planned_schedule != emitted_row
                    or planned_schedule.get("component") != component
                    or planned_schedule.get("token_start") != token_start
                    or planned_schedule.get("token_end") != token_end
                ):
                    raise ValueError(
                        "published schedule differs from frozen "
                        "disk-backed plan"
                    )
                planned_bytes = bytes(token_bytes)
                expected_bytes = (
                    (token_end - token_start)
                    * np.dtype(np.uint16).itemsize
                )
                if len(planned_bytes) != expected_bytes:
                    raise ValueError(
                        "frozen schedule token bytes do not match offsets"
                    )
                if train is not None and train.read(
                    expected_bytes
                ) != planned_bytes:
                    raise ValueError(
                        "train.bin differs from frozen schedule plan"
                    )
                planned_spans = json.loads(spans_json)
                if not isinstance(planned_spans, list) or not planned_spans:
                    raise ValueError(
                        "frozen schedule record has no span evidence"
                    )
                local_end = 0
                record_length = token_end - token_start
                for raw_span in planned_spans:
                    if not isinstance(raw_span, dict) or set(raw_span) != {
                        "start",
                        "end",
                        "role",
                        "fact_id",
                        "fact_cost",
                        "payload_field",
                        "payload_text",
                    }:
                        raise ValueError(
                            "invalid frozen schedule span evidence"
                        )
                    start = raw_span["start"]
                    end = raw_span["end"]
                    role = raw_span["role"]
                    fact_id = raw_span["fact_id"]
                    if (
                        isinstance(start, bool)
                        or not isinstance(start, int)
                        or isinstance(end, bool)
                        or not isinstance(end, int)
                        or start != local_end
                        or not start < end <= record_length
                        or role not in _REPLAY_SPAN_ROLES
                        or (
                            role == "payload"
                            and (
                                not isinstance(fact_id, str)
                                or not fact_id
                            )
                        )
                        or (role != "payload" and fact_id is not None)
                    ):
                        raise ValueError(
                            "invalid frozen schedule span evidence"
                        )
                    if replay_spans is not None:
                        replay_span = next(replay_spans, sentinel)
                        expected_span = (
                            expected_span_id,
                            record_index,
                            component,
                            token_start + start,
                            token_start + end,
                            end - start,
                            relative_position_bin(
                                start,
                                end,
                                record_length,
                            ),
                            role,
                            fact_id,
                            raw_span["payload_field"],
                            raw_span["payload_text"],
                        )
                        if (
                            replay_span is sentinel
                            or replay_span != expected_span
                        ):
                            raise ValueError(
                                "replay span differs from frozen "
                                "schedule plan"
                            )
                    local_end = end
                    expected_span_id += 1
                if local_end != record_length:
                    raise ValueError(
                        "frozen schedule spans do not cover their record"
                    )
                expected_record_index += 1
                expected_token_start = token_end
            if expected_record_index == 0:
                raise ValueError("frozen schedule plan contains no records")
            if train is not None and train.read(1):
                raise ValueError(
                    "train.bin extends beyond frozen schedule plan"
                )
        if (
            replay_spans is not None
            and next(replay_spans, sentinel) is not sentinel
        ):
            raise ValueError(
                "replay span table contains extra span evidence"
            )
    except (sqlite3.Error, json.JSONDecodeError) as exc:
        raise ValueError("invalid frozen schedule plan") from exc
    finally:
        assert plan_connection is not None
        plan_connection.close()
        if occurrence_connection is not None:
            occurrence_connection.close()


def freeze_production_split_expectations(
    root_dir: str | Path,
) -> dict[str, str]:
    """Freeze Task 4 signatures from the exact disk-backed production plans."""

    root = Path(root_dir)
    snapshots = _production_split_snapshots(
        root,
        verify_replay_evidence=False,
    )
    expectation_dir = root / "expectations"
    expectation_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in _SPLIT_NAMES:
        expectation = _capture_split_expectations(
            name,
            *snapshots[name],
        )
        expectation.write(expectation_dir / f"{name}.json")
        hashes[name] = expectation.sha256()
    return hashes


def _observe_production_splits(
    root: Path,
    frozen_sha256: dict[str, str] | None,
) -> tuple[dict[str, ObservedSplitArtifacts], dict[str, str]]:
    snapshots = _production_split_snapshots(
        root,
        verify_replay_evidence=False,
    )
    observed = {}
    hashes = {}
    for name in _SPLIT_NAMES:
        path = root / "expectations" / f"{name}.json"
        expectation = SplitArtifactExpectations.from_path(path)
        digest = expectation.sha256()
        if frozen_sha256 is not None and digest != frozen_sha256.get(name):
            raise ValueError(
                f"persisted pre-emission expectation changed: {name}"
            )
        hashes[name] = digest
        observed[name] = ObservedSplitArtifacts.from_generated(
            name,
            expectations=expectation,
            worlds=snapshots[name][0],
            qa_items=snapshots[name][1],
            rendered_records=snapshots[name][2],
        )
    require_disjointness(observed)
    return observed, hashes


def _verify_payload_inventory(root: Path, tok) -> None:
    inventory_path = root / "payload-inventory.json"
    inventory = PayloadInventory.from_path(inventory_path)
    rebuilt = PayloadInventory.from_rows(
        tok,
        {
            "train": (
                GraphRow.from_json(raw)
                for raw in _jsonl_rows(root / "graph.jsonl")
            ),
            "protected_seen": (
                GraphRow.from_json(raw)
                for raw in _jsonl_rows(
                    root / "eval" / "protected_seen" / "graph.jsonl"
                )
            ),
            "protected_heldout": (
                GraphRow.from_json(raw)
                for raw in _jsonl_rows(
                    root / "eval" / "protected_heldout" / "graph.jsonl"
                )
            ),
        },
    )
    def identity(entry):
        return (
            entry.scope,
            entry.fact_id,
            entry.field,
            entry.text,
            entry.token_ids,
        )
    if {identity(entry) for entry in inventory.entries} != {
        identity(entry) for entry in rebuilt.entries
    }:
        raise ValueError(
            "payload inventory differs from reopened production graph rows"
        )
    if any(
        entry.expected_occurrences is None
        or (
            entry.scope != "train"
            and entry.expected_occurrences != 0
        )
        for entry in inventory.entries
    ):
        raise ValueError("payload inventory occurrence counts are not frozen")
    mask_audit = verify_weight_sidecars(
        root / "train.bin",
        {
            condition: root / f"{condition}.weights.bin"
            for condition in ("dense", "split", "random", "selective")
        },
        root / "mask-occurrences.sqlite3",
        payload_inventory=inventory_path,
        record_schedule=root / "schedule.jsonl",
    )
    try:
        persisted = json.loads((root / "mask-audit.json").read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid persisted mask audit") from exc
    if persisted != mask_audit.to_dict():
        raise ValueError("persisted mask audit differs from independent replay")


def freeze_published_artifact_expectations(
    source_dir: str | Path,
    destination: str | Path | None = None,
) -> dict[str, object]:
    """Freeze exact plan outputs before replay into the publication directory."""

    root = Path(source_dir)
    if not root.is_dir():
        raise ValueError("publication plan directory does not exist")
    artifacts = {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in _artifact_paths(root)
    }
    required = {
        "train.bin",
        "dense.weights.bin",
        "split.weights.bin",
        "random.weights.bin",
        "selective.weights.bin",
        "graph.jsonl",
        "eval/graph.jsonl",
        "eval/original.jsonl",
        "eval/counterfactual.jsonl",
        "schedule.jsonl",
        "schedule-manifest.json",
        "mixture-manifest.json",
    }
    if "relation-schema.json" in artifacts:
        required.update(
            {
                "eval/protected_seen/graph.jsonl",
                "eval/protected_seen/original.jsonl",
                "eval/protected_seen/counterfactual.jsonl",
                "eval/protected_heldout/graph.jsonl",
                "eval/protected_heldout/original.jsonl",
                "eval/protected_heldout/counterfactual.jsonl",
                *{
                    f"expectations/{name}.json"
                    for name in _SPLIT_NAMES
                },
            }
        )
    missing = sorted(required - artifacts.keys())
    if missing:
        raise ValueError(f"publication plan lacks required artifacts: {missing}")
    jsonl = {
        relative: _jsonl_signature(root / relative)
        for relative in sorted(artifacts)
        if relative.endswith(".jsonl")
    }
    expectation = {
        "version": 1,
        "artifacts": artifacts,
        "jsonl": jsonl,
    }
    output = root / EXPECTATION_NAME if destination is None else Path(destination)
    output.write_bytes(_canonical_bytes(expectation))
    return expectation


def _load_expectation(root: Path) -> tuple[dict[str, object], str]:
    path = root / EXPECTATION_NAME
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid published artifact expectations") from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "artifacts", "jsonl"}
        or value["version"] != 1
        or not isinstance(value["artifacts"], dict)
        or not isinstance(value["jsonl"], dict)
        or raw != _canonical_bytes(value)
    ):
        raise ValueError("invalid published artifact expectations")
    return value, hashlib.sha256(raw).hexdigest()


def _verify_generation_expectations(
    root: Path,
    frozen_sha256: str | None,
) -> str:
    path = root / "generation-expectations.json"
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid pre-emission generation expectations") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if frozen_sha256 is not None and digest != frozen_sha256:
        raise ValueError("pre-emission generation expectations changed")
    if (
        not isinstance(value, dict)
        or value.get("version") != 1
        or value.get("phase") != "before_train_replay"
        or not isinstance(value.get("artifacts"), dict)
        or not isinstance(value.get("split_expectation_sha256"), dict)
        or not isinstance(value.get("payload_inventory_sha256"), str)
    ):
        raise ValueError("invalid pre-emission generation expectations")
    for relative in value["artifacts"]:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("pre-emission artifact path is not portable")
    try:
        _verify_file_signatures(root, value["artifacts"])
    except ValueError as exc:
        raise ValueError(
            "pre-emission generation artifact mismatch: "
            "published artifact changed"
        ) from exc
    split_hashes = (
        {
            name: SplitArtifactExpectations.from_path(
                root / "expectations" / f"{name}.json"
            ).sha256()
            for name in _SPLIT_NAMES
        }
        if value["split_expectation_sha256"]
        else {}
    )
    if split_hashes != value["split_expectation_sha256"]:
        raise ValueError(
            "pre-emission split expectation digest mismatch"
        )
    inventory = PayloadInventory.from_path(root / "payload-inventory.json")
    if inventory.sha256() != value["payload_inventory_sha256"]:
        raise ValueError(
            "pre-emission payload inventory digest mismatch"
        )
    return digest


def _verify_file_signatures(root: Path, expected: dict[str, object]) -> None:
    for relative, signature in expected.items():
        path = root / relative
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ValueError(
                f"published artifact is missing: {relative}"
            ) from exc
        if (
            not isinstance(signature, dict)
            or signature.get("bytes") != size
            or signature.get("sha256") != _sha256_file(path)
        ):
            raise ValueError(
                f"published artifact signature mismatch: {relative}"
            )


def _verify_jsonl_signatures(root: Path, expected: dict[str, object]) -> None:
    for relative, signature in expected.items():
        if _jsonl_signature(root / relative) != signature:
            raise ValueError(
                f"published artifact row signature mismatch: {relative}"
            )


def _verify_graph_store_pair(
    jsonl_path: Path,
    store_path: Path,
    codec: RelationCodec,
) -> None:
    with PackedGraphStore.load(store_path, codec) as store:
        rows = 0
        with jsonl_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = GraphRow.from_json(json.loads(line))
                if store.lookup(row.address) != row:
                    raise ValueError(
                        "published artifact graph row differs from packed store: "
                        f"{jsonl_path.relative_to(jsonl_path.parents[1])}"
                    )
                rows += 1
        if rows != len(store):
            raise ValueError(
                "published artifact graph row count differs from packed store"
            )


def _graph_fact_ids(path: Path) -> set[str]:
    fact_ids = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                fact_ids.add(stable_fact_id(GraphRow.from_json(json.loads(line))))
    if not fact_ids:
        raise ValueError("published artifact graph contains no facts")
    return fact_ids


def _verify_schedule(root: Path) -> None:
    token_count = (
        (root / "train.bin").stat().st_size
        // np.dtype(np.uint16).itemsize
    )
    previous_end = 0
    records = 0
    train_fact_ids = _graph_fact_ids(root / "graph.jsonl")
    with (root / "schedule.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "component",
                "record_id",
                "exposure",
                "curriculum_band",
                "token_start",
                "token_end",
            }
            if not isinstance(row, dict) or not required <= row.keys():
                raise ValueError(
                    f"published artifact schedule row {line_number} is invalid"
                )
            if (
                row["token_start"] != previous_end
                or not isinstance(row["token_end"], int)
                or row["token_end"] <= previous_end
            ):
                raise ValueError(
                    "published artifact schedule token ranges are not contiguous"
                )
            previous_end = row["token_end"]
            if (
                row["component"] == "graph"
                and not str(row["record_id"]).startswith("rule-")
                    and not str(row["record_id"]).startswith(
                        "relation-alias:"
                    )
                and row["record_id"] not in train_fact_ids
            ):
                raise ValueError(
                    "published artifact schedule graph fact is absent from "
                    "the training graph"
                )
            if row["component"] == "reasoning":
                metadata = row.get("metadata")
                fact_ids = (
                    metadata.get("gold_fact_ids")
                    if isinstance(metadata, dict)
                    else None
                )
                if (
                    not isinstance(fact_ids, list)
                    or not fact_ids
                    or any(fact_id not in train_fact_ids for fact_id in fact_ids)
                ):
                    raise ValueError(
                        "published artifact reasoning schedule is not "
                        "reconciled with the training graph"
                    )
            records += 1
    if previous_end != token_count:
        raise ValueError(
            "published artifact schedule does not cover train.bin"
        )
    manifest = json.loads((root / "schedule-manifest.json").read_bytes())
    if (
        manifest.get("records") != records
        or manifest.get("tokens") != token_count
        or manifest.get("sha256")
        != _sha256_file(root / "schedule.jsonl")
    ):
        raise ValueError(
            "published artifact schedule manifest does not match records"
        )


def _verify_eval_items(eval_dir: Path) -> None:
    eval_fact_ids = _graph_fact_ids(eval_dir / "graph.jsonl")
    for name in ("original.jsonl", "counterfactual.jsonl"):
        path = eval_dir / name
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                metadata = item.get("meta")
                fact_ids = (
                    metadata.get("gold_fact_ids")
                    if isinstance(metadata, dict)
                    else None
                )
                if (
                    not isinstance(fact_ids, list)
                    or not fact_ids
                    or any(fact_id not in eval_fact_ids for fact_id in fact_ids)
                ):
                    raise ValueError(
                        "published artifact eval item is not reconciled with "
                        "the protected graph"
                    )


def verify_published_artifacts(
    root_dir: str | Path,
    codec: RelationCodec | None = None,
    *,
    frozen_split_expectation_sha256: dict[str, str] | None = None,
    frozen_generation_expectation_sha256: str | None = None,
    tok=None,
) -> dict[str, object]:
    """Independently reopen exact published graph, eval, and schedule outputs."""

    root = Path(root_dir)
    if frozen_generation_expectation_sha256 is None:
        try:
            manifest = json.loads((root / "manifest.json").read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid published manifest") from exc
        candidate = manifest.get("generation_expectation_sha256")
        if isinstance(candidate, str):
            frozen_generation_expectation_sha256 = candidate
    generation_expectation_sha256 = _verify_generation_expectations(
        root,
        frozen_generation_expectation_sha256,
    )
    expectation, expectation_sha256 = _load_expectation(root)
    _verify_file_signatures(root, expectation["artifacts"])
    _verify_jsonl_signatures(root, expectation["jsonl"])
    _verify_schedule_plan_replay(root)
    _verify_schedule(root)
    _verify_eval_items(root / "eval")
    split_hashes: dict[str, str] = {}
    split_checks: dict[str, bool] = {}
    if codec is not None:
        pairs = (
            ("graph.jsonl", "graph.store"),
            ("eval/graph.jsonl", "eval/graph.store"),
            ("eval/factual-graph.jsonl", "eval/factual-graph.store"),
            (
                "eval/protected_seen/graph.jsonl",
                "eval/protected_seen/graph.store",
            ),
            (
                "eval/protected_heldout/graph.jsonl",
                "eval/protected_heldout/graph.store",
            ),
        )
        for jsonl, store in pairs:
            _verify_graph_store_pair(root / jsonl, root / store, codec)
        for name in ("protected_seen", "protected_heldout"):
            _verify_eval_items(root / "eval" / name)
        observed, split_hashes = _observe_production_splits(
            root,
            frozen_split_expectation_sha256,
        )
        split_checks = audit_disjointness(observed)
    if tok is not None:
        _verify_payload_inventory(root, tok)
    return {
        "version": 3,
        "expectation_sha256": expectation_sha256,
        "generation_expectation_sha256": (
            generation_expectation_sha256
        ),
        "split_expectation_sha256": dict(sorted(split_hashes.items())),
        "split_checks": dict(sorted(split_checks.items())),
        "checks": {
            "exact_file_signatures": True,
            "pre_emission_generation_reconciliation": True,
            "canonical_jsonl_signatures": True,
            "graph_store_reconciliation": codec is not None,
            "schedule_reconciliation": True,
            "schedule_plan_token_replay": True,
            "all_span_replay_reconciliation": True,
            "eval_item_reconciliation": True,
            "split_expectation_reconciliation": (
                codec is None or bool(split_hashes)
            ),
            "split_disjointness": (
                codec is None or all(split_checks.values())
            ),
            "payload_inventory_reconciliation": (
                tok is not None or codec is None
            ),
        },
    }


def write_published_artifact_audit(
    root_dir: str | Path,
    codec: RelationCodec | None = None,
    *,
    frozen_split_expectation_sha256: dict[str, str] | None = None,
    frozen_generation_expectation_sha256: str | None = None,
    tok=None,
) -> Path:
    root = Path(root_dir)
    audit = verify_published_artifacts(
        root,
        codec,
        frozen_split_expectation_sha256=(
            frozen_split_expectation_sha256
        ),
        frozen_generation_expectation_sha256=(
            frozen_generation_expectation_sha256
        ),
        tok=tok,
    )
    path = root / AUDIT_NAME
    if path.exists():
        try:
            persisted = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid persisted publication audit") from exc
        if (
            persisted != audit
            or path.read_bytes() != _canonical_bytes(persisted)
        ):
            raise ValueError("persisted publication audit changed")
    path.write_bytes(_canonical_bytes(audit))
    return path
