"""Descriptor-pinned atomic lifecycle state for idempotent run IDs."""

from __future__ import annotations

import fcntl
import math
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path

from .errors import MsctlError
from .fsutil import (
    atomic_write_json_at,
    load_json_at,
    open_directory,
    open_directory_at,
)
from .jsonutil import (
    RUN_ID_RE,
    require_exact_keys,
    require_nonnegative_int,
    require_object,
    require_schema_version,
    require_sha256,
)

RUN_STATE_KEYS = {
    "schema_version",
    "run_id",
    "arm",
    "seed",
    "provider",
    "release_sha256",
    "run_manifest_sha256",
    "config_sha256",
    "dataset_sha256",
    "dataset_verification_sha256",
    "environment_receipt_sha256",
    "operation",
    "submission_key",
    "resource_request",
    "job_id",
    "status",
    "attempt",
    "created_at",
    "updated_at",
}
RESUME_STATE_KEYS = RUN_STATE_KEYS | {
    "checkpoint_receipt_sha256",
    "prior_job_ids",
}
EVALUATION_STATE_KEYS = {
    "schema_version",
    "provider",
    "release_sha256",
    "run_manifest_sha256",
    "dataset_sha256",
    "dataset_verification_sha256",
    "environment_receipt_sha256",
    "operation",
    "submission_key",
    "resource_request",
    "job_id",
    "status",
    "created_at",
    "updated_at",
}
INTENT_KEYS = {
    "schema_version",
    "submission_key",
    "provider",
    "release_sha256",
    "run_manifest_sha256",
    "dataset_sha256",
    "dataset_verification_sha256",
    "environment_receipt_sha256",
    "operation",
    "resource_request",
    "run_ids",
    "attempt",
    "checkpoint_receipt_sha256",
    "phase",
    "job_id",
    "created_at",
    "updated_at",
}
RESOURCE_KEYS = {
    "schema_version",
    "operation",
    "jobs",
    "allocated_gpus",
    "wall_minutes",
    "gpu_hours",
    "gres",
    "script",
}
STATUS_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MsctlError("STATE_CORRUPT", f"{label} must be a non-empty string")
    return value


def _validate_job_id(value: object, *, label: str) -> None:
    if value is not None and (
        not isinstance(value, str) or not value.isdigit()
    ):
        raise MsctlError(
            "STATE_CORRUPT",
            f"{label} must be null or a decimal Slurm job ID",
        )


def _validate_resource(value: object, *, operation: str) -> None:
    resource = require_object(value, label="state resource request")
    require_exact_keys(resource, RESOURCE_KEYS, label="state resource request")
    require_schema_version(
        resource["schema_version"],
        label="state resource request.schema_version",
    )
    if resource["operation"] != operation:
        raise MsctlError(
            "STATE_CORRUPT",
            "state resource request operation does not match",
        )
    for field in ("jobs", "allocated_gpus", "wall_minutes"):
        require_nonnegative_int(
            resource[field],
            label=f"state resource request.{field}",
        )
    gpu_hours = resource["gpu_hours"]
    if (
        isinstance(gpu_hours, bool)
        or not isinstance(gpu_hours, (int, float))
        or not math.isfinite(gpu_hours)
        or gpu_hours < 0
    ):
        raise MsctlError(
            "STATE_CORRUPT",
            "state resource request GPU hours are invalid",
        )
    for field in ("gres", "script"):
        if resource[field] is not None and (
            not isinstance(resource[field], str) or not resource[field]
        ):
            raise MsctlError(
                "STATE_CORRUPT",
                f"state resource request {field} is invalid",
            )


def _validate_common(value: dict[str, object], *, operation: str) -> None:
    require_schema_version(
        value["schema_version"],
        label="lifecycle state.schema_version",
    )
    _require_string(value["provider"], label="state provider")
    for field in (
        "release_sha256",
        "run_manifest_sha256",
        "dataset_sha256",
        "dataset_verification_sha256",
        "environment_receipt_sha256",
        "submission_key",
    ):
        require_sha256(value[field], label=f"state {field}")
    if value["operation"] != operation:
        raise MsctlError("STATE_CORRUPT", "state operation is invalid")
    _validate_resource(value["resource_request"], operation=operation)
    _validate_job_id(value["job_id"], label="state job_id")
    status = _require_string(value["status"], label="state status")
    if STATUS_RE.fullmatch(status) is None:
        raise MsctlError("STATE_CORRUPT", "state status is invalid")
    for field in ("created_at", "updated_at"):
        _require_string(value[field], label=f"state {field}")


def _validate_run_state(value: dict[str, object], run_id: str) -> None:
    operation = value.get("operation")
    keys = RESUME_STATE_KEYS if operation == "resume" else RUN_STATE_KEYS
    require_exact_keys(value, keys, label="run state")
    if operation not in {"submit", "resume"}:
        raise MsctlError("STATE_CORRUPT", "run state operation is invalid")
    _validate_common(value, operation=operation)
    if value["run_id"] != run_id or RUN_ID_RE.fullmatch(run_id) is None:
        raise MsctlError("STATE_CORRUPT", "run state has the wrong run ID")
    if value["arm"] not in {"dense", "split90"}:
        raise MsctlError("STATE_CORRUPT", "run state arm is invalid")
    require_nonnegative_int(value["seed"], label="run state seed")
    require_sha256(value["config_sha256"], label="run state config hash")
    attempt = require_nonnegative_int(value["attempt"], label="run state attempt")
    if attempt < 1:
        raise MsctlError("STATE_CORRUPT", "run state attempt is invalid")
    if operation == "resume":
        require_sha256(
            value["checkpoint_receipt_sha256"],
            label="run state checkpoint receipt hash",
        )
        prior = value["prior_job_ids"]
        if (
            not isinstance(prior, list)
            or not prior
            or any(not isinstance(item, str) or not item.isdigit() for item in prior)
            or len(prior) != len(set(prior))
        ):
            raise MsctlError(
                "STATE_CORRUPT",
                "run state prior job IDs are invalid",
            )


def _validate_evaluation_state(
    value: dict[str, object],
    manifest_sha256: str,
) -> None:
    require_exact_keys(value, EVALUATION_STATE_KEYS, label="evaluation state")
    _validate_common(value, operation="evaluate")
    if value["run_manifest_sha256"] != manifest_sha256:
        raise MsctlError(
            "STATE_CORRUPT",
            "evaluation state has the wrong manifest binding",
        )


def _validate_intent(value: dict[str, object], submission_key: str) -> None:
    require_exact_keys(value, INTENT_KEYS, label="pair intent")
    operation = value.get("operation")
    if operation not in {"submit", "resume"}:
        raise MsctlError("STATE_CORRUPT", "pair intent operation is invalid")
    require_schema_version(
        value["schema_version"],
        label="pair intent.schema_version",
    )
    if value["submission_key"] != submission_key:
        raise MsctlError("STATE_CORRUPT", "pair intent key does not match")
    _require_string(value["provider"], label="pair intent provider")
    for field in (
        "submission_key",
        "release_sha256",
        "run_manifest_sha256",
        "dataset_sha256",
        "dataset_verification_sha256",
        "environment_receipt_sha256",
    ):
        require_sha256(value[field], label=f"pair intent {field}")
    _validate_resource(value["resource_request"], operation=operation)
    run_ids = value["run_ids"]
    if not isinstance(run_ids, list) or len(run_ids) != 2:
        raise MsctlError("STATE_CORRUPT", "pair intent run IDs are invalid")
    if (
        any(
            not isinstance(run_id, str) or RUN_ID_RE.fullmatch(run_id) is None
            for run_id in run_ids
        )
        or len(set(run_ids)) != 2
    ):
        raise MsctlError("STATE_CORRUPT", "pair intent run IDs are invalid")
    attempt = require_nonnegative_int(value["attempt"], label="pair intent attempt")
    if attempt < 1:
        raise MsctlError("STATE_CORRUPT", "pair intent attempt is invalid")
    checkpoint = value["checkpoint_receipt_sha256"]
    if operation == "resume":
        require_sha256(checkpoint, label="pair intent checkpoint hash")
    elif checkpoint is not None:
        raise MsctlError("STATE_CORRUPT", "submit intent has a checkpoint hash")
    if value["phase"] not in {"PREPARED", "SUBMITTING", "SUBMITTED"}:
        raise MsctlError("STATE_CORRUPT", "pair intent phase is invalid")
    _validate_job_id(value["job_id"], label="pair intent job_id")
    if (value["phase"] == "SUBMITTED") != (value["job_id"] is not None):
        raise MsctlError(
            "STATE_CORRUPT",
            "pair intent phase and job ID are inconsistent",
        )
    for field in ("created_at", "updated_at"):
        _require_string(value[field], label=f"pair intent {field}")


class StateStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._root_fd: int | None = None
        self._runs_fd: int | None = None
        self._evaluations_fd: int | None = None
        self._intents_fd: int | None = None

    def _state_error(self, error: Exception) -> MsctlError:
        return MsctlError(
            "UNSAFE_STATE",
            "state directories must be pinned regular directories without symlinks",
            details={"root": str(self.root)},
        )

    def _require_locked(self) -> tuple[int, int, int, int]:
        if (
            self._root_fd is None
            or self._runs_fd is None
            or self._evaluations_fd is None
            or self._intents_fd is None
        ):
            raise MsctlError(
                "UNSAFE_STATE",
                "state access requires the pinned state lock",
            )
        return (
            self._root_fd,
            self._runs_fd,
            self._evaluations_fd,
            self._intents_fd,
        )

    @contextmanager
    def locked(self):
        if self._root_fd is not None:
            raise MsctlError("UNSAFE_STATE", "state lock is not reentrant")
        root_fd: int | None = None
        runs_fd: int | None = None
        evaluations_fd: int | None = None
        intents_fd: int | None = None
        try:
            root_fd = open_directory(
                self.root,
                label="state root",
                create=True,
            )
            runs_fd = open_directory_at(
                root_fd,
                "runs",
                label="state runs",
                create=True,
            )
            evaluations_fd = open_directory_at(
                root_fd,
                "evaluations",
                label="state evaluations",
                create=True,
            )
            intents_fd = open_directory_at(
                root_fd,
                "intents",
                label="state intents",
                create=True,
            )
        except MsctlError as error:
            for descriptor in (intents_fd, evaluations_fd, runs_fd, root_fd):
                if descriptor is not None:
                    os.close(descriptor)
            raise self._state_error(error) from error
        assert root_fd is not None
        assert runs_fd is not None
        assert evaluations_fd is not None
        assert intents_fd is not None
        lock_flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        lock_fd: int | None = None
        try:
            lock_fd = os.open(".lock", lock_flags, 0o600, dir_fd=root_fd)
            if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
                raise self._state_error(ValueError("non-regular lock"))
        except (OSError, MsctlError) as error:
            if lock_fd is not None:
                os.close(lock_fd)
            os.close(intents_fd)
            os.close(evaluations_fd)
            os.close(runs_fd)
            os.close(root_fd)
            raise self._state_error(error) from error
        assert lock_fd is not None
        self._root_fd = root_fd
        self._runs_fd = runs_fd
        self._evaluations_fd = evaluations_fd
        self._intents_fd = intents_fd
        lock_acquired = False
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            lock_acquired = True
            yield self
        finally:
            if lock_acquired:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            os.close(intents_fd)
            os.close(evaluations_fd)
            os.close(runs_fd)
            os.close(root_fd)
            self._root_fd = None
            self._runs_fd = None
            self._evaluations_fd = None
            self._intents_fd = None

    def _run_name(self, run_id: str) -> str:
        if RUN_ID_RE.fullmatch(run_id) is None:
            raise MsctlError("UNSAFE_STATE", "invalid run ID for state path")
        return f"{run_id}.json"

    def read_run(self, run_id: str) -> dict[str, object] | None:
        _, runs_fd, _, _ = self._require_locked()
        name = self._run_name(run_id)
        try:
            raw = load_json_at(runs_fd, name, label="run state")
        except MsctlError as error:
            if error.code == "FILE_NOT_FOUND":
                return None
            if error.code in {"INVALID_JSON"}:
                raise MsctlError(
                    "STATE_CORRUPT",
                    "run state is not valid JSON",
                    details={"run_id": run_id},
                ) from error
            raise MsctlError(
                "UNSAFE_STATE",
                "run state must be a regular file",
                details={"run_id": run_id},
            ) from error
        try:
            value = require_object(raw, label="run state")
            _validate_run_state(value, run_id)
        except MsctlError as error:
            raise MsctlError(
                "STATE_CORRUPT",
                "run state schema is invalid",
                details={"run_id": run_id},
            ) from error
        return value

    def write_run(self, run_id: str, value: dict[str, object]) -> None:
        _, runs_fd, _, _ = self._require_locked()
        try:
            _validate_run_state(value, run_id)
        except MsctlError as error:
            raise MsctlError(
                "STATE_CORRUPT",
                "state write schema is invalid",
            ) from error
        atomic_write_json_at(
            runs_fd,
            self._run_name(run_id),
            value,
            label="run state",
        )

    def _evaluation_name(self, manifest_sha256: str) -> str:
        try:
            require_sha256(
                manifest_sha256,
                label="evaluation manifest hash",
            )
        except MsctlError as error:
            raise MsctlError(
                "UNSAFE_STATE",
                "invalid manifest hash for evaluation state",
            ) from error
        return f"{manifest_sha256}.json"

    def read_evaluation(
        self, manifest_sha256: str
    ) -> dict[str, object] | None:
        _, _, evaluations_fd, _ = self._require_locked()
        name = self._evaluation_name(manifest_sha256)
        try:
            raw = load_json_at(
                evaluations_fd,
                name,
                label="evaluation state",
            )
        except MsctlError as error:
            if error.code == "FILE_NOT_FOUND":
                return None
            if error.code == "INVALID_JSON":
                raise MsctlError(
                    "STATE_CORRUPT",
                    "evaluation state is not valid JSON",
                ) from error
            raise MsctlError(
                "UNSAFE_STATE",
                "evaluation state must be a regular file",
            ) from error
        try:
            value = require_object(raw, label="evaluation state")
            _validate_evaluation_state(value, manifest_sha256)
        except MsctlError as error:
            raise MsctlError(
                "STATE_CORRUPT",
                "evaluation state schema is invalid",
            ) from error
        return value

    def write_evaluation(
        self,
        manifest_sha256: str,
        value: dict[str, object],
    ) -> None:
        _, _, evaluations_fd, _ = self._require_locked()
        try:
            _validate_evaluation_state(value, manifest_sha256)
        except MsctlError as error:
            raise MsctlError(
                "STATE_CORRUPT",
                "evaluation state write schema is invalid",
            ) from error
        atomic_write_json_at(
            evaluations_fd,
            self._evaluation_name(manifest_sha256),
            value,
            label="evaluation state",
        )

    def _intent_name(self, submission_key: str) -> str:
        try:
            require_sha256(submission_key, label="pair intent key")
        except MsctlError as error:
            raise MsctlError(
                "UNSAFE_STATE",
                "invalid submission key for pair intent",
            ) from error
        return f"{submission_key}.json"

    def read_intent(
        self,
        submission_key: str,
    ) -> dict[str, object] | None:
        _, _, _, intents_fd = self._require_locked()
        name = self._intent_name(submission_key)
        try:
            raw = load_json_at(intents_fd, name, label="pair intent")
        except MsctlError as error:
            if error.code == "FILE_NOT_FOUND":
                return None
            if error.code == "INVALID_JSON":
                raise MsctlError(
                    "STATE_CORRUPT",
                    "pair intent is not valid JSON",
                ) from error
            raise MsctlError(
                "UNSAFE_STATE",
                "pair intent must be a regular file",
            ) from error
        try:
            value = require_object(raw, label="pair intent")
            _validate_intent(value, submission_key)
        except MsctlError as error:
            raise MsctlError(
                "STATE_CORRUPT",
                "pair intent schema is invalid",
            ) from error
        return value

    def write_intent(
        self,
        submission_key: str,
        value: dict[str, object],
    ) -> None:
        _, _, _, intents_fd = self._require_locked()
        try:
            _validate_intent(value, submission_key)
        except MsctlError as error:
            raise MsctlError(
                "STATE_CORRUPT",
                "pair intent write schema is invalid",
            ) from error
        atomic_write_json_at(
            intents_fd,
            self._intent_name(submission_key),
            value,
            label="pair intent",
        )
