"""Immutable, hash-chained run lifecycle ledgers."""

from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from experiment.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    load_canonical_json,
    validate_sha256,
)


EVENT_TYPES = frozenset(
    {
        "planned",
        "preflight_passed",
        "launch_requested",
        "started",
        "checkpointed",
        "failed",
        "resumed",
        "excluded",
        "completed",
    }
)

_TRANSITIONS = {
    "planned": {"preflight_passed", "failed", "excluded"},
    "preflight_passed": {"launch_requested", "failed", "excluded"},
    "launch_requested": {"started", "failed", "excluded"},
    "started": {"checkpointed", "failed", "completed"},
    "checkpointed": {"checkpointed", "failed", "resumed", "completed"},
    "failed": {
        "preflight_passed",
        "launch_requested",
        "checkpointed",
        "resumed",
        "excluded",
    },
    "resumed": {"started", "checkpointed", "failed", "completed"},
    "excluded": set(),
    "completed": set(),
}

_IDENTIFIER_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_FILENAME_RE = re.compile(r"(0|[1-9][0-9]*)-([a-z0-9][a-z0-9._-]{0,127})\.json")
_EVENT_FIELDS = {
    "record_type",
    "schema_version",
    "run_id",
    "sequence",
    "event_id",
    "event_type",
    "previous_event_sha256",
    "details",
    "event_sha256",
}


class LedgerError(ValueError):
    """Raised when a run ledger is invalid or cannot be extended safely."""


def _validate_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise LedgerError(f"{name} is not a canonical portable identifier")
    return value


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise LedgerError("ledger details must be canonical JSON") from exc


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _event_material(
    *,
    run_id: str,
    sequence: int,
    event_id: str,
    event_type: str,
    previous_event_sha256: str | None,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "record_type": "run_ledger_event",
        "schema_version": 1,
        "run_id": run_id,
        "sequence": sequence,
        "event_id": event_id,
        "event_type": event_type,
        "previous_event_sha256": previous_event_sha256,
        "details": _thaw(details),
    }


@dataclass(frozen=True)
class LedgerEvent:
    run_id: str
    sequence: int
    event_id: str
    event_type: str
    previous_event_sha256: str | None
    details: Mapping[str, Any]
    event_sha256: str

    def to_dict(self) -> dict[str, Any]:
        material = _event_material(
            run_id=self.run_id,
            sequence=self.sequence,
            event_id=self.event_id,
            event_type=self.event_type,
            previous_event_sha256=self.previous_event_sha256,
            details=self.details,
        )
        return {**material, "event_sha256": self.event_sha256}

    as_dict = to_dict

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        sequence: int,
        event_id: str,
        event_type: str,
        previous_event_sha256: str | None,
        details: Mapping[str, Any] | None = None,
    ) -> "LedgerEvent":
        run = _validate_identifier(run_id, "run ID")
        identifier = _validate_identifier(event_id, "event ID")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
        ):
            raise LedgerError("event sequence must be a nonnegative integer")
        if event_type not in EVENT_TYPES:
            raise LedgerError("ledger event type is invalid")
        if previous_event_sha256 is not None:
            try:
                validate_sha256(
                    previous_event_sha256,
                    "previous event SHA-256",
                )
            except ValueError as exc:
                raise LedgerError(str(exc)) from exc
        copied = _json_copy({} if details is None else details)
        if not isinstance(copied, dict):
            raise LedgerError("ledger event details must be an object")
        frozen = _freeze(copied)
        material = _event_material(
            run_id=run,
            sequence=sequence,
            event_id=identifier,
            event_type=event_type,
            previous_event_sha256=previous_event_sha256,
            details=frozen,
        )
        return cls(
            run_id=run,
            sequence=sequence,
            event_id=identifier,
            event_type=event_type,
            previous_event_sha256=previous_event_sha256,
            details=frozen,
            event_sha256=canonical_sha256(material),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LedgerEvent":
        if not isinstance(raw, Mapping) or set(raw) != _EVENT_FIELDS:
            raise LedgerError("ledger event fields are not exact")
        if (
            raw["record_type"] != "run_ledger_event"
            or raw["schema_version"] != 1
        ):
            raise LedgerError("ledger event protocol is invalid")
        event = cls.create(
            run_id=raw["run_id"],
            sequence=raw["sequence"],
            event_id=raw["event_id"],
            event_type=raw["event_type"],
            previous_event_sha256=raw["previous_event_sha256"],
            details=raw["details"],
        )
        try:
            supplied_hash = validate_sha256(
                raw["event_sha256"],
                "event SHA-256",
            )
        except ValueError as exc:
            raise LedgerError(str(exc)) from exc
        if supplied_hash != event.event_sha256:
            raise LedgerError("ledger event hash indicates tampering")
        return event


def _validate_transition(
    previous: LedgerEvent | None,
    event_type: str,
) -> None:
    if previous is None:
        if event_type != "planned":
            raise LedgerError("initial ledger event must be planned")
        return
    if event_type not in _TRANSITIONS[previous.event_type]:
        raise LedgerError(
            "illegal ledger transition "
            f"{previous.event_type!r} -> {event_type!r}"
        )


def _validate_root(root: str | Path) -> Path:
    candidate = Path(root)
    if ".." in candidate.parts:
        raise LedgerError("ledger root cannot contain traversal")
    absolute = candidate if candidate.is_absolute() else Path.cwd() / candidate
    if candidate.is_symlink() or not candidate.is_dir():
        raise LedgerError("ledger root must be a regular non-symlink directory")
    if candidate.resolve(strict=True) != absolute:
        raise LedgerError("ledger root is not canonical or traverses a symlink")
    return absolute


def _validate_ledger_path(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise LedgerError("run ledger must be a regular non-symlink directory")
    if path.resolve(strict=True) != path:
        raise LedgerError("run ledger path traverses a symlink")


def _load_events(path: Path, run_id: str) -> tuple[LedgerEvent, ...]:
    if not os.path.lexists(path):
        return ()
    _validate_ledger_path(path)
    candidates: list[tuple[int, str, Path]] = []
    for entry in path.iterdir():
        if entry.name == ".append.lock":
            if entry.is_symlink() or not entry.is_file():
                raise LedgerError("ledger append lock is not a regular file")
            continue
        if entry.name.startswith("."):
            if entry.is_symlink() or not entry.is_file():
                raise LedgerError(
                    "ledger staging entry is not a regular file"
                )
            continue
        match = _FILENAME_RE.fullmatch(entry.name)
        if match is None or entry.is_symlink() or not entry.is_file():
            raise LedgerError("run ledger contains a partial or unexpected entry")
        candidates.append((int(match.group(1)), match.group(2), entry))

    sequence_names = [sequence for sequence, _, _ in candidates]
    if len(sequence_names) != len(set(sequence_names)):
        raise LedgerError("duplicate ledger event sequence")
    filename_ids = [event_id for _, event_id, _ in candidates]
    if len(filename_ids) != len(set(filename_ids)):
        raise LedgerError("duplicate ledger event ID")

    events: list[LedgerEvent] = []
    for sequence, filename_id, entry in sorted(candidates):
        try:
            event = LedgerEvent.from_dict(load_canonical_json(entry))
        except (OSError, TypeError, ValueError) as exc:
            if isinstance(exc, LedgerError):
                raise
            raise LedgerError(
                "ledger event canonical hash indicates tampering"
            ) from exc
        if event.sequence != sequence or event.event_id != filename_id:
            raise LedgerError("ledger event filename does not match its identity")
        if event.run_id != run_id:
            raise LedgerError("ledger event run ID mismatch")
        if event.sequence != len(events):
            raise LedgerError("ledger event sequences must be contiguous")
        previous = events[-1] if events else None
        _validate_transition(previous, event.event_type)
        expected_previous = (
            previous.event_sha256 if previous is not None else None
        )
        if event.previous_event_sha256 != expected_previous:
            raise LedgerError("ledger event hash chain is broken")
        events.append(event)

    content_ids = [event.event_id for event in events]
    if len(content_ids) != len(set(content_ids)):
        raise LedgerError("duplicate ledger event ID")
    return tuple(events)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class RunLedger:
    """Append-only canonical event ledger for one run ID."""

    def __init__(self, out_root: str | Path, run_id: str):
        self.out_root = _validate_root(out_root)
        self.run_id = _validate_identifier(run_id, "run ID")
        self.path = self.out_root / "ledger" / self.run_id

    def _ensure_directory(self) -> None:
        ledger_root = self.out_root / "ledger"
        ledger_root_existed = os.path.lexists(ledger_root)
        if not ledger_root_existed:
            ledger_root.mkdir(exist_ok=True)
        _validate_ledger_path(ledger_root)
        if not ledger_root_existed:
            _fsync_directory(self.out_root)
        run_path_existed = os.path.lexists(self.path)
        if not run_path_existed:
            self.path.mkdir(exist_ok=True)
        _validate_ledger_path(self.path)
        if not run_path_existed:
            _fsync_directory(ledger_root)

    def events(self) -> tuple[LedgerEvent, ...]:
        return _load_events(self.path, self.run_id)

    def append(
        self,
        event_type: str,
        *,
        event_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> LedgerEvent:
        if not os.path.lexists(self.path):
            _validate_transition(None, event_type)
            if event_id is not None:
                _validate_identifier(event_id, "event ID")
            copied = _json_copy({} if details is None else details)
            if not isinstance(copied, dict):
                raise LedgerError("ledger event details must be an object")
        self._ensure_directory()
        lock_path = self.path / ".append.lock"
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise LedgerError("cannot open ledger append lock") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise LedgerError("ledger append lock must be a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            events = _load_events(self.path, self.run_id)
            previous = events[-1] if events else None
            _validate_transition(previous, event_type)
            sequence = len(events)
            copied_details = _json_copy({} if details is None else details)
            if not isinstance(copied_details, dict):
                raise LedgerError("ledger event details must be an object")
            if event_id is None:
                event_id = canonical_sha256(
                    {
                        "record_type": "generated_run_event_id",
                        "schema_version": 1,
                        "run_id": self.run_id,
                        "sequence": sequence,
                        "event_type": event_type,
                        "previous_event_sha256": (
                            previous.event_sha256
                            if previous is not None
                            else None
                        ),
                        "details": copied_details,
                    }
                )[:24]
            identifier = _validate_identifier(event_id, "event ID")
            if any(event.event_id == identifier for event in events):
                raise LedgerError(f"duplicate ledger event ID: {identifier}")
            event = LedgerEvent.create(
                run_id=self.run_id,
                sequence=sequence,
                event_id=identifier,
                event_type=event_type,
                previous_event_sha256=(
                    previous.event_sha256 if previous is not None else None
                ),
                details=copied_details,
            )
            destination = (
                self.path / f"{event.sequence}-{event.event_id}.json"
            )
            staging = self.path / (
                f".{event.sequence}-{event.event_id}."
                f"{os.getpid()}.{secrets.token_hex(8)}.json"
            )
            try:
                atomic_write_json(staging, event.to_dict())
                try:
                    os.link(staging, destination, follow_symlinks=False)
                except FileExistsError as exc:
                    raise LedgerError(
                        "ledger event destination already exists; "
                        "immutable event was not overwritten"
                    ) from exc
                _fsync_directory(self.path)
            finally:
                staging.unlink(missing_ok=True)
                _fsync_directory(self.path)
            return event
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def summary(self) -> dict[str, Any]:
        return materialize_summary(self.events(), run_id=self.run_id)

    def exit_status(self) -> int:
        return int(self.summary()["exit_status"])


def materialize_summary(
    events: Sequence[LedgerEvent],
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    materialized = tuple(events)
    seen_ids: set[str] = set()
    previous: LedgerEvent | None = None
    for index, event in enumerate(materialized):
        if not isinstance(event, LedgerEvent):
            raise LedgerError("summary requires validated LedgerEvent values")
        validated = LedgerEvent.from_dict(event.to_dict())
        if validated.sequence != index:
            raise LedgerError("summary event sequences must be contiguous")
        if validated.event_id in seen_ids:
            raise LedgerError("duplicate ledger event ID")
        _validate_transition(previous, validated.event_type)
        expected_previous = (
            previous.event_sha256 if previous is not None else None
        )
        if validated.previous_event_sha256 != expected_previous:
            raise LedgerError("summary event hash chain is broken")
        seen_ids.add(validated.event_id)
        previous = validated
    if materialized:
        selected_run_id = materialized[0].run_id
        if any(event.run_id != selected_run_id for event in materialized):
            raise LedgerError("summary events contain multiple run IDs")
        if run_id is not None and run_id != selected_run_id:
            raise LedgerError("summary run ID mismatch")
        status = materialized[-1].event_type
    else:
        if run_id is None:
            raise LedgerError("empty ledger summary requires a run ID")
        selected_run_id = _validate_identifier(run_id, "run ID")
        status = "unplanned"
    failures = [
        {
            "event_id": event.event_id,
            "sequence": event.sequence,
            "details": _thaw(event.details),
        }
        for event in materialized
        if event.event_type == "failed"
    ]
    exclusions = [
        {
            "event_id": event.event_id,
            "sequence": event.sequence,
            "details": _thaw(event.details),
        }
        for event in materialized
        if event.event_type == "excluded"
    ]
    terminal = status in {"failed", "excluded", "completed"}
    if status == "completed":
        exit_status = 0
    elif status in {"failed", "excluded"}:
        exit_status = 1
    else:
        exit_status = 2
    return {
        "record_type": "run_ledger_summary",
        "schema_version": 1,
        "run_id": selected_run_id,
        "status": status,
        "terminal": terminal,
        "event_count": len(materialized),
        "latest_sequence": (
            materialized[-1].sequence if materialized else None
        ),
        "event_chain_sha256": canonical_sha256(
            [event.event_sha256 for event in materialized]
        ),
        "checkpoint_count": sum(
            event.event_type == "checkpointed" for event in materialized
        ),
        "failure_count": len(failures),
        "failures": failures,
        "exclusion_count": len(exclusions),
        "exclusions": exclusions,
        "exit_status": exit_status,
    }


def load_run_ledger(
    out_root: str | Path,
    run_id: str,
) -> tuple[LedgerEvent, ...]:
    return RunLedger(out_root, run_id).events()


def validate_run_ledger(
    out_root: str | Path,
    run_id: str,
) -> tuple[LedgerEvent, ...]:
    return load_run_ledger(out_root, run_id)


def append_event(
    out_root: str | Path,
    run_id: str,
    event_type: str,
    *,
    event_id: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> LedgerEvent:
    return RunLedger(out_root, run_id).append(
        event_type,
        event_id=event_id,
        details=details,
    )
