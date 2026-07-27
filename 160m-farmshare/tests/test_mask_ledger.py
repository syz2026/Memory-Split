from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from corpusgen.mask_ledger import (
    LeakageError,
    OccurrenceSpool,
    RandomMaskUndersupplyError,
    _compare_zero_runs,
    _iter_weight_zero_runs,
    derive_weight_sidecars,
    verify_weight_sidecars,
)
from corpusgen.payload_inventory import (
    PayloadInventory,
    PayloadInventoryEntry,
)
from corpusgen.relational_build import EncodedSpan, FactCost, RoutePolicy


def _costs():
    internal = FactCost(
        "internal",
        entropy=1,
        exposures=32,
        expected_reads=20,
        expected_hops=4,
    )
    external = FactCost(
        "external",
        entropy=16,
        exposures=1,
        expected_reads=0,
        expected_hops=0,
    )
    return internal, external


def _spans():
    internal, external = _costs()
    return [
        EncodedSpan(0, 10, "action"),
        EncodedSpan(
            10,
            12,
            "payload",
            "internal",
            internal,
            payload_field="target",
            payload_text='"internal"',
        ),
        EncodedSpan(12, 13, "relation_alias"),
        EncodedSpan(13, 15, "random_control"),
        EncodedSpan(15, 30, "plain"),
        EncodedSpan(
            30,
            33,
            "payload",
            "external",
            external,
            payload_field="target",
            payload_text='"external"',
        ),
        EncodedSpan(33, 34, "rule"),
        EncodedSpan(34, 37, "random_control"),
        EncodedSpan(37, 70, "plain"),
        EncodedSpan(70, 80, "provisional_answer"),
        EncodedSpan(80, 90, "final_answer"),
        EncodedSpan(90, 100, "plain"),
        EncodedSpan(100, 101, "boundary"),
    ]


def _zero_fact_ids(weights, spans):
    return {
        span.fact_id
        for span in spans
        if span.fact_id is not None
        and not weights[span.start : span.end].any()
    }


def test_split_masks_every_fact_but_selective_preserves_cost_routing():
    spans = _spans()

    outputs = derive_weight_sidecars(
        spans,
        policy=RoutePolicy(write_cost=1),
        seed=7,
    )

    assert _zero_fact_ids(outputs["split"], spans) == {
        "internal",
        "external",
    }
    assert _zero_fact_ids(outputs["selective"], spans) == {"external"}
    assert _zero_fact_ids(outputs["dense"], spans) == set()
    assert outputs["random"][12] == 1
    assert outputs["random"][33] == 1
    assert outputs["random"][70:90].all()


def _build_fixture(
    tmp_path,
    *,
    duplicate_plain_payload=False,
    token_overrides=(),
):
    spans = _spans()
    policy = RoutePolicy(write_cost=1)
    tokens = np.arange(101, dtype=np.uint16) + 1000
    tokens[10:12] = [401, 402]
    tokens[30:33] = [501, 502, 503]
    if duplicate_plain_payload:
        tokens[50:52] = tokens[10:12]
    for start, values in token_overrides:
        tokens[start : start + len(values)] = values

    outputs = derive_weight_sidecars(spans, policy=policy, seed=23)
    train_path = tmp_path / "train.bin"
    train_path.write_bytes(tokens.tobytes())
    sidecars = {
        condition: tmp_path / f"{condition}.weights.bin"
        for condition in ("dense", "split", "random", "selective")
    }
    for condition in ("dense", "split", "selective"):
        sidecars[condition].write_bytes(outputs[condition].tobytes())
    sidecars["random"].write_bytes(
        np.ones(len(tokens), dtype=np.uint8).tobytes()
    )

    spool_path = tmp_path / "occurrences.sqlite3"
    with OccurrenceSpool(spool_path) as spool:
        spool.add_record(
            component="fixture",
            record_index=0,
            global_start=0,
            token_ids=tokens,
            spans=spans,
            policy=policy,
        )
        spool.finalize_random(sidecars["random"], seed=23)

    return train_path, sidecars, spool_path


def _inventory(tmp_path, *entries):
    path = tmp_path / "payload-inventory.json"
    PayloadInventory(entries=tuple(entries)).write(path)
    return path


def _canonical_fixture_entries():
    return (
        PayloadInventoryEntry(
            scope="train",
            fact_id="internal",
            field="target",
            text='"internal"',
            token_ids=(401, 402),
        ),
        PayloadInventoryEntry(
            scope="train",
            fact_id="external",
            field="target",
            text='"external"',
            token_ids=(501, 502, 503),
        ),
    )


def test_random_matches_split_exactly_by_mass_length_and_position(tmp_path):
    train_path, sidecars, spool_path = _build_fixture(tmp_path)

    audit = verify_weight_sidecars(train_path, sidecars, spool_path)

    assert audit.masked_tokens["random"] == audit.masked_tokens["split"]
    assert audit.histograms["random"] == audit.histograms["split"]
    assert audit.histograms["split"] == Counter({(2, 1): 1, (3, 3): 1})
    assert audit.pending_random_demands == 0


def test_independent_scan_rejects_untagged_payload_occurrence(tmp_path):
    train_path, sidecars, spool_path = _build_fixture(
        tmp_path,
        duplicate_plain_payload=True,
    )

    with pytest.raises(LeakageError, match="untagged direct payload"):
        verify_weight_sidecars(train_path, sidecars, spool_path)


def test_canonical_inventory_rejects_unique_mislabeled_payload(tmp_path):
    train_path, sidecars, spool_path = _build_fixture(
        tmp_path,
        token_overrides=((50, (701, 702)),),
    )
    inventory = _inventory(
        tmp_path,
        *_canonical_fixture_entries(),
        PayloadInventoryEntry(
            scope="train",
            fact_id="omitted-fact",
            field="target",
            text='"omitted"',
            token_ids=(701, 702),
        ),
    )

    with pytest.raises(LeakageError, match="untagged direct payload"):
        verify_weight_sidecars(
            train_path,
            sidecars,
            spool_path,
            payload_inventory=inventory,
        )


def test_canonical_inventory_rejects_unknown_tagged_payload(tmp_path):
    train_path, sidecars, spool_path = _build_fixture(tmp_path)
    inventory = _inventory(
        tmp_path,
        PayloadInventoryEntry(
            scope="train",
            fact_id="internal",
            field="target",
            text='"internal"',
            token_ids=(401, 402),
        ),
    )

    with pytest.raises(LeakageError, match="inventory-unknown tagged payload"):
        verify_weight_sidecars(
            train_path,
            sidecars,
            spool_path,
            payload_inventory=inventory,
        )


def test_canonical_inventory_rejects_missing_expected_payload(tmp_path):
    train_path, sidecars, spool_path = _build_fixture(tmp_path)
    inventory = _inventory(
        tmp_path,
        PayloadInventoryEntry(
            scope="train",
            fact_id="internal",
            field="target",
            text='"internal"',
            token_ids=(401, 402),
            expected_occurrences=1,
        ),
        PayloadInventoryEntry(
            scope="train",
            fact_id="external",
            field="target",
            text='"external"',
            token_ids=(501, 502, 503),
            expected_occurrences=1,
        ),
        PayloadInventoryEntry(
            scope="train",
            fact_id="missing",
            field="target",
            text='"missing"',
            token_ids=(701, 702),
            expected_occurrences=1,
        ),
    )

    with pytest.raises(LeakageError, match="payload occurrence count mismatch"):
        verify_weight_sidecars(
            train_path,
            sidecars,
            spool_path,
            payload_inventory=inventory,
        )


def test_payload_inventory_binds_frozen_schedule_occurrence_counts():
    inventory = PayloadInventory(
        entries=(
            PayloadInventoryEntry(
                scope="train",
                fact_id="fact",
                field="target",
                text='"value"',
                token_ids=(7, 8),
                expected_occurrences=0,
            ),
            PayloadInventoryEntry(
                scope="protected_seen",
                fact_id="protected",
                field="target",
                text='"protected"',
                token_ids=(9,),
                expected_occurrences=0,
            ),
        )
    )

    bound = inventory.bind_expected_occurrences(
        {("fact", "target", '"value"', (7, 8)): 3}
    )

    assert {
        (entry.scope, entry.fact_id): entry.expected_occurrences
        for entry in bound.entries
    } == {
        ("train", "fact"): 3,
        ("protected_seen", "protected"): 0,
    }


def test_canonical_inventory_requires_matching_field_and_text(tmp_path):
    train_path, sidecars, spool_path = _build_fixture(tmp_path)
    inventory = _inventory(
        tmp_path,
        PayloadInventoryEntry(
            scope="train",
            fact_id="internal",
            field="qualifier:wrong",
            text='"wrong"',
            token_ids=(401, 402),
        ),
        PayloadInventoryEntry(
            scope="train",
            fact_id="external",
            field="target",
            text='"external"',
            token_ids=(501, 502, 503),
        ),
    )

    with pytest.raises(LeakageError, match="inventory-unknown tagged payload"):
        verify_weight_sidecars(
            train_path,
            sidecars,
            spool_path,
            payload_inventory=inventory,
        )


def test_canonical_inventory_scans_overlapping_prefix_patterns(tmp_path):
    train_path, sidecars, spool_path = _build_fixture(tmp_path)
    inventory = _inventory(
        tmp_path,
        *_canonical_fixture_entries(),
        PayloadInventoryEntry(
            scope="train",
            fact_id="internal",
            field="target",
            text='"prefix"',
            token_ids=(401,),
        ),
    )

    audit = verify_weight_sidecars(
        train_path,
        sidecars,
        spool_path,
        payload_inventory=inventory,
    )

    assert audit.pending_random_demands == 0


def test_canonical_inventory_rejects_protected_payload_pattern(tmp_path):
    train_path, sidecars, spool_path = _build_fixture(
        tmp_path,
        token_overrides=((50, (701, 702)),),
    )
    inventory = _inventory(
        tmp_path,
        *_canonical_fixture_entries(),
        PayloadInventoryEntry(
            scope="protected_seen",
            fact_id="protected-fact",
            field="target",
            text='"protected"',
            token_ids=(701, 702),
        ),
    )

    with pytest.raises(LeakageError, match="protected payload"):
        verify_weight_sidecars(
            train_path,
            sidecars,
            spool_path,
            payload_inventory=inventory,
        )


def test_payload_scanner_resets_at_frozen_record_boundaries(tmp_path):
    train_path, sidecars, spool_path = _build_fixture(
        tmp_path,
        token_overrides=((49, (65534,)), (50, (65533,))),
    )
    inventory = _inventory(
        tmp_path,
        *_canonical_fixture_entries(),
        PayloadInventoryEntry(
            scope="protected_seen",
            fact_id="cross-record-only",
            field="target",
            text='"special-boundary-pieces"',
            token_ids=(65534, 65533),
        ),
    )
    schedule = tmp_path / "schedule.jsonl"
    schedule.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in (
                {"token_start": 0, "token_end": 50},
                {"token_start": 50, "token_end": 101},
            )
        )
    )

    audit = verify_weight_sidecars(
        train_path,
        sidecars,
        spool_path,
        payload_inventory=inventory,
        record_schedule=schedule,
    )

    assert audit.pending_random_demands == 0


def test_mask_audit_summarizes_instead_of_materializing_zero_runs(tmp_path):
    train_path, sidecars, spool_path = _build_fixture(tmp_path)

    audit = verify_weight_sidecars(train_path, sidecars, spool_path)

    assert audit.zero_runs["split"].count == 2
    assert audit.zero_runs["split"].masked_tokens == 5
    assert not hasattr(audit.zero_runs["split"], "__iter__")


def test_zero_run_audit_streams_many_occurrences(tmp_path):
    run_count = 50_000
    path = tmp_path / "many-runs.weights.bin"
    path.write_bytes(bytes((0, 1)) * run_count)

    summary = _compare_zero_runs(
        condition="stress",
        actual=_iter_weight_zero_runs(path, run_count * 2),
        expected=((offset, offset + 1) for offset in range(0, run_count * 2, 2)),
    )

    assert summary.count == run_count
    assert summary.masked_tokens == run_count


def test_sqlite_matcher_fails_when_an_exact_key_is_undersupplied(tmp_path):
    spans = [
        EncodedSpan(0, 10, "plain"),
        EncodedSpan(10, 12, "payload", "internal", _costs()[0]),
        EncodedSpan(12, 70, "plain"),
        EncodedSpan(70, 72, "random_control"),
        EncodedSpan(72, 101, "plain"),
    ]
    tokens = np.arange(101, dtype=np.uint16)
    random_path = tmp_path / "random.weights.bin"
    random_path.write_bytes(np.ones(len(tokens), dtype=np.uint8).tobytes())

    with OccurrenceSpool(tmp_path / "occurrences.sqlite3") as spool:
        spool.add_record(
            component="fixture",
            record_index=0,
            global_start=0,
            token_ids=tokens,
            spans=spans,
            policy=RoutePolicy(write_cost=1),
        )
        assert spool.random_deficits() == {(2, 1): 1}
        with pytest.raises(
            RandomMaskUndersupplyError,
            match=r"exact key \(2, 1\)",
        ):
            spool.finalize_random(random_path, seed=7)
