"""Atomic local lifecycle state for idempotent run IDs."""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path

from .errors import MsctlError
from .jsonutil import RUN_ID_RE, atomic_write_json


class StateStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _check_root(self) -> None:
        if self.root.is_symlink():
            raise MsctlError(
                "UNSAFE_STATE",
                "state root must not be a symlink",
            )

    @contextmanager
    def locked(self):
        self._check_root()
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / ".lock"
        if lock.is_symlink():
            raise MsctlError("UNSAFE_STATE", "state lock must not be a symlink")
        with lock.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield self
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _run_path(self, run_id: str) -> Path:
        if RUN_ID_RE.fullmatch(run_id) is None:
            raise MsctlError("UNSAFE_STATE", "invalid run ID for state path")
        return self.root / "runs" / f"{run_id}.json"

    def read_run(self, run_id: str) -> dict[str, object] | None:
        path = self._run_path(run_id)
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise MsctlError(
                "UNSAFE_STATE",
                "run state must be a regular file",
                details={"run_id": run_id},
            )
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise MsctlError(
                "STATE_CORRUPT",
                "run state is not valid JSON",
                details={"run_id": run_id},
            ) from error
        if not isinstance(value, dict) or value.get("run_id") != run_id:
            raise MsctlError(
                "STATE_CORRUPT",
                "run state does not bind its run ID",
                details={"run_id": run_id},
            )
        return value

    def write_run(self, run_id: str, value: dict[str, object]) -> None:
        if value.get("run_id") != run_id:
            raise MsctlError("STATE_CORRUPT", "state write has the wrong run ID")
        atomic_write_json(self._run_path(run_id), value)

    def _evaluation_path(self, manifest_sha256: str) -> Path:
        if (
            len(manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in manifest_sha256)
        ):
            raise MsctlError(
                "UNSAFE_STATE",
                "invalid manifest hash for evaluation state",
            )
        return self.root / "evaluations" / f"{manifest_sha256}.json"

    def read_evaluation(
        self, manifest_sha256: str
    ) -> dict[str, object] | None:
        path = self._evaluation_path(manifest_sha256)
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise MsctlError(
                "UNSAFE_STATE",
                "evaluation state must be a regular file",
            )
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise MsctlError(
                "STATE_CORRUPT",
                "evaluation state is not valid JSON",
            ) from error
        if (
            not isinstance(value, dict)
            or value.get("run_manifest_sha256") != manifest_sha256
        ):
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
        if value.get("run_manifest_sha256") != manifest_sha256:
            raise MsctlError(
                "STATE_CORRUPT",
                "evaluation state write has the wrong manifest binding",
            )
        atomic_write_json(self._evaluation_path(manifest_sha256), value)
