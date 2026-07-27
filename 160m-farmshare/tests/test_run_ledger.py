from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment.artifacts import (
    canonical_json_bytes,
    canonical_sha256,
    load_canonical_json,
)
from experiment.ledger import (
    LedgerError,
    LedgerEvent,
    RunLedger,
    materialize_summary,
)


def _append_started(ledger: RunLedger) -> None:
    ledger.append("planned", event_id="plan")
    ledger.append("preflight_passed", event_id="preflight")
    ledger.append("launch_requested", event_id="launch")
    ledger.append("started", event_id="start")


def _tree_snapshot(root: Path) -> tuple:
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        rows.append(
            (
                relative,
                "dir" if path.is_dir() else "file",
                None if path.is_dir() else path.read_bytes(),
            )
        )
    return tuple(rows)


def test_legal_resume_lifecycle_retains_failure_history(tmp_path):
    ledger = RunLedger(tmp_path, "run-1")
    _append_started(ledger)
    ledger.append("checkpointed", event_id="checkpoint-1", details={"step": 2})
    ledger.append(
        "failed",
        event_id="failure-1",
        details={"reason": "worker_lost", "step": 2},
    )
    ledger.append("resumed", event_id="resume-1", details={"step": 2})
    ledger.append("checkpointed", event_id="checkpoint-2", details={"step": 3})
    ledger.append("completed", event_id="complete", details={"step": 3})

    events = ledger.events()
    assert [event.sequence for event in events] == list(range(len(events)))
    assert [event.event_type for event in events] == [
        "planned",
        "preflight_passed",
        "launch_requested",
        "started",
        "checkpointed",
        "failed",
        "resumed",
        "checkpointed",
        "completed",
    ]
    assert all(
        path.name == f"{event.sequence}-{event.event_id}.json"
        for path, event in zip(
            sorted(
                ledger.path.glob("*.json"),
                key=lambda item: int(item.name.split("-", 1)[0]),
            ),
            events,
        )
    )

    summary = ledger.summary()
    assert summary == ledger.summary()
    assert summary["status"] == "completed"
    assert summary["failure_count"] == 1
    assert summary["failures"] == [
        {
            "event_id": "failure-1",
            "sequence": 5,
            "details": {"reason": "worker_lost", "step": 2},
        }
    ]
    assert summary["exit_status"] == 0

    for path in ledger.path.glob("*.json"):
        assert path.read_bytes() == canonical_json_bytes(load_canonical_json(path))


@pytest.mark.parametrize(
    ("events", "rejected"),
    [
        ((), "started"),
        (("planned",), "started"),
        (("planned", "excluded"), "resumed"),
        (
            (
                "planned",
                "preflight_passed",
                "launch_requested",
                "started",
                "completed",
            ),
            "checkpointed",
        ),
    ],
)
def test_illegal_ledger_transitions_do_not_publish(tmp_path, events, rejected):
    ledger = RunLedger(tmp_path, "run-1")
    for index, event_type in enumerate(events):
        ledger.append(event_type, event_id=f"valid-{index}")
    before = _tree_snapshot(tmp_path)

    with pytest.raises(LedgerError, match="transition|initial"):
        ledger.append(rejected, event_id="rejected")

    assert _tree_snapshot(tmp_path) == before


def test_duplicate_event_id_is_rejected_without_overwrite(tmp_path):
    ledger = RunLedger(tmp_path, "run-1")
    ledger.append("planned", event_id="same-id", details={"attempt": 1})
    before = _tree_snapshot(tmp_path)

    with pytest.raises(LedgerError, match="duplicate.*event"):
        ledger.append(
            "preflight_passed",
            event_id="same-id",
            details={"attempt": 2},
        )

    assert _tree_snapshot(tmp_path) == before


def test_loader_rejects_tampering_and_duplicate_sequences(tmp_path):
    ledger = RunLedger(tmp_path, "run-1")
    _append_started(ledger)
    first = ledger.path / "0-plan.json"

    raw = load_canonical_json(first)
    raw["details"] = {"tampered": True}
    first.write_bytes(canonical_json_bytes(raw))
    with pytest.raises(LedgerError, match="hash|tamper"):
        ledger.events()

    first.write_bytes(canonical_json_bytes({**raw, "details": {}}))
    repaired = json.loads(first.read_bytes())
    repaired["event_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in repaired.items()
            if key != "event_sha256"
        }
    )
    first.write_bytes(canonical_json_bytes(repaired))
    duplicate = ledger.path / "0-copy.json"
    duplicate.write_bytes(first.read_bytes())
    with pytest.raises(LedgerError, match="duplicate.*sequence|duplicate.*event"):
        ledger.events()


def test_loader_rejects_duplicate_event_ids_even_with_valid_hash(tmp_path):
    ledger = RunLedger(tmp_path, "run-1")
    ledger.append("planned", event_id="duplicate")
    ledger.append("preflight_passed", event_id="second")
    second = ledger.path / "1-second.json"
    raw = load_canonical_json(second)
    raw["event_id"] = "duplicate"
    raw["event_sha256"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "event_sha256"}
    )
    replacement = ledger.path / "1-duplicate.json"
    replacement.write_bytes(canonical_json_bytes(raw))
    second.unlink()

    with pytest.raises(LedgerError, match="duplicate.*event"):
        ledger.events()


def test_terminal_failure_and_exclusion_have_nonzero_exit_status(tmp_path):
    failed = RunLedger(tmp_path, "failed-run")
    _append_started(failed)
    failed.append("failed", event_id="failed", details={"reason": "oom"})
    assert failed.summary()["status"] == "failed"
    assert failed.summary()["exit_status"] != 0

    excluded = RunLedger(tmp_path, "excluded-run")
    excluded.append("planned", event_id="planned")
    excluded.append(
        "excluded",
        event_id="excluded",
        details={"reason": "preflight_policy"},
    )
    summary = excluded.summary()
    assert summary["status"] == "excluded"
    assert summary["exclusion_count"] == 1
    assert summary["exclusions"][0]["details"]["reason"] == "preflight_policy"
    assert summary["exit_status"] != 0


def test_event_publication_uses_atomic_staging_and_leaves_no_partial_files(
    tmp_path,
    monkeypatch,
):
    import experiment.ledger as ledger_module

    staged: list[Path] = []
    real_atomic_write = ledger_module.atomic_write_json

    def recording_atomic_write(path, value):
        staged.append(Path(path))
        return real_atomic_write(path, value)

    monkeypatch.setattr(
        ledger_module,
        "atomic_write_json",
        recording_atomic_write,
    )
    ledger = RunLedger(tmp_path, "run-1")
    ledger.append("planned", event_id="plan")

    assert len(staged) == 1
    assert staged[0].parent == ledger.path
    assert staged[0].name.startswith(".")
    assert not staged[0].exists()
    assert [path.name for path in ledger.path.glob("*.json")] == ["0-plan.json"]


def test_reader_observes_previous_ledger_while_event_is_staged(
    tmp_path,
    monkeypatch,
):
    import experiment.ledger as ledger_module

    real_atomic_write = ledger_module.atomic_write_json
    observed: list[tuple[LedgerEvent, ...]] = []
    ledger = RunLedger(tmp_path, "run-1")

    def observe_staged_write(path, value):
        result = real_atomic_write(path, value)
        observed.append(ledger.events())
        return result

    monkeypatch.setattr(
        ledger_module,
        "atomic_write_json",
        observe_staged_write,
    )

    ledger.append("planned", event_id="plan")

    assert observed == [()]
    assert [event.event_type for event in ledger.events()] == ["planned"]


def test_summary_revalidates_supplied_event_transitions():
    planned = LedgerEvent.create(
        run_id="run-1",
        sequence=0,
        event_id="plan",
        event_type="planned",
        previous_event_sha256=None,
    )
    illegally_completed = LedgerEvent.create(
        run_id="run-1",
        sequence=1,
        event_id="complete",
        event_type="completed",
        previous_event_sha256=planned.event_sha256,
    )

    with pytest.raises(LedgerError, match="transition"):
        materialize_summary((planned, illegally_completed))
