"""Descriptor-pinned atomic lifecycle state for idempotent run IDs."""

from __future__ import annotations

import fcntl
import os
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
    require_object,
    require_schema_version,
    require_sha256,
)


class StateStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._root_fd: int | None = None
        self._runs_fd: int | None = None
        self._evaluations_fd: int | None = None

    def _state_error(self, error: Exception) -> MsctlError:
        return MsctlError(
            "UNSAFE_STATE",
            "state directories must be pinned regular directories without symlinks",
            details={"root": str(self.root)},
        )

    def _require_locked(self) -> tuple[int, int, int]:
        if (
            self._root_fd is None
            or self._runs_fd is None
            or self._evaluations_fd is None
        ):
            raise MsctlError(
                "UNSAFE_STATE",
                "state access requires the pinned state lock",
            )
        return self._root_fd, self._runs_fd, self._evaluations_fd

    @contextmanager
    def locked(self):
        if self._root_fd is not None:
            raise MsctlError("UNSAFE_STATE", "state lock is not reentrant")
        root_fd: int | None = None
        runs_fd: int | None = None
        evaluations_fd: int | None = None
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
        except MsctlError as error:
            for descriptor in (evaluations_fd, runs_fd, root_fd):
                if descriptor is not None:
                    os.close(descriptor)
            raise self._state_error(error) from error
        assert root_fd is not None
        assert runs_fd is not None
        assert evaluations_fd is not None
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
            os.close(evaluations_fd)
            os.close(runs_fd)
            os.close(root_fd)
            raise self._state_error(error) from error
        assert lock_fd is not None
        self._root_fd = root_fd
        self._runs_fd = runs_fd
        self._evaluations_fd = evaluations_fd
        lock_acquired = False
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            lock_acquired = True
            yield self
        finally:
            if lock_acquired:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            os.close(evaluations_fd)
            os.close(runs_fd)
            os.close(root_fd)
            self._root_fd = None
            self._runs_fd = None
            self._evaluations_fd = None

    def _run_name(self, run_id: str) -> str:
        if RUN_ID_RE.fullmatch(run_id) is None:
            raise MsctlError("UNSAFE_STATE", "invalid run ID for state path")
        return f"{run_id}.json"

    def read_run(self, run_id: str) -> dict[str, object] | None:
        _, runs_fd, _ = self._require_locked()
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
        value = require_object(raw, label="run state")
        try:
            require_schema_version(
                value.get("schema_version"),
                label="run state.schema_version",
            )
        except MsctlError as error:
            raise MsctlError(
                "STATE_CORRUPT",
                "run state has an invalid schema version",
                details={"run_id": run_id},
            ) from error
        if value.get("run_id") != run_id:
            raise MsctlError(
                "STATE_CORRUPT",
                "run state does not bind its run ID",
                details={"run_id": run_id},
            )
        return value

    def write_run(self, run_id: str, value: dict[str, object]) -> None:
        _, runs_fd, _ = self._require_locked()
        if value.get("run_id") != run_id:
            raise MsctlError("STATE_CORRUPT", "state write has the wrong run ID")
        try:
            require_schema_version(
                value.get("schema_version"),
                label="run state.schema_version",
            )
            atomic_write_json_at(
                runs_fd,
                self._run_name(run_id),
                value,
                label="run state",
            )
        except MsctlError as error:
            if error.code in {"SCHEMA_INVALID"}:
                raise MsctlError(
                    "STATE_CORRUPT",
                    "state write has an invalid schema version",
                ) from error
            raise

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
        _, _, evaluations_fd = self._require_locked()
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
        value = require_object(raw, label="evaluation state")
        try:
            require_schema_version(
                value.get("schema_version"),
                label="evaluation state.schema_version",
            )
        except MsctlError as error:
            raise MsctlError(
                "STATE_CORRUPT",
                "evaluation state has an invalid schema version",
            ) from error
        if value.get("run_manifest_sha256") != manifest_sha256:
            raise MsctlError(
                "STATE_CORRUPT",
                "evaluation state has the wrong manifest binding",
            )
        return value

    def write_evaluation(
        self,
        manifest_sha256: str,
        value: dict[str, object],
    ) -> None:
        _, _, evaluations_fd = self._require_locked()
        if value.get("run_manifest_sha256") != manifest_sha256:
            raise MsctlError(
                "STATE_CORRUPT",
                "evaluation state write has the wrong manifest binding",
            )
        try:
            require_schema_version(
                value.get("schema_version"),
                label="evaluation state.schema_version",
            )
            atomic_write_json_at(
                evaluations_fd,
                self._evaluation_name(manifest_sha256),
                value,
                label="evaluation state",
            )
        except MsctlError as error:
            if error.code == "SCHEMA_INVALID":
                raise MsctlError(
                    "STATE_CORRUPT",
                    "evaluation state write has an invalid schema version",
                ) from error
            raise
