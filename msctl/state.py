"""Descriptor-pinned atomic lifecycle state for idempotent run IDs."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import stat
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

from .errors import MsctlError
from .fsutil import (
    atomic_write_at,
    atomic_write_json_at,
    load_json_at,
    open_directory,
    open_directory_at,
    open_regular_at,
    read_fd,
)
from .jsonutil import (
    RUN_ID_RE,
    canonical_json,
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
AWS_RUN_STATE_KEYS = {
    "schema_version",
    "run_id",
    "arm",
    "seed",
    "provider",
    "release_sha256",
    "run_manifest_sha256",
    "config_sha256",
    "dataset_sha256",
    "dataset_pointer_sha256",
    "dataset_verification_sha256",
    "environment_receipt_sha256",
    "cohort_assignment_sha256",
    "study_lock_sha256",
    "source_commit",
    "profile_sha256",
    "runtime_sha256",
    "ami_id",
    "container_digest",
    "instance_id",
    "terminate_at",
    "operation_id",
    "intent_sha256",
    "intent_uri",
    "command_id",
    "operation",
    "status",
    "attempt",
    "send_attempted",
    "created_at",
    "updated_at",
}
AWS_RESUME_STATE_KEYS = AWS_RUN_STATE_KEYS | {
    "checkpoint_receipt_sha256",
    "prior_command_ids",
}
AWS_V3_RUN_STATE_KEYS = {
    "schema_version",
    "run_id",
    "arm",
    "seed",
    "provider",
    "release_sha256",
    "release_receipt_sha256",
    "run_manifest_sha256",
    "config_sha256",
    "dataset_pointer_sha256",
    "dataset_receipt_sha256",
    "dataset_build_id",
    "ordered_stream_sha256",
    "dataset_verification_sha256",
    "environment_receipt_sha256",
    "cohort_assignment_sha256",
    "preregistration_sha256",
    "source_commit",
    "source_tree",
    "profile_sha256",
    "runtime_sha256",
    "ami_id",
    "container_digest",
    "instance_id",
    "terminate_at",
    "operation_id",
    "intent_sha256",
    "intent_uri",
    "command_id",
    "operation",
    "status",
    "attempt",
    "send_attempted",
    "created_at",
    "updated_at",
}
AWS_V3_RESUME_STATE_KEYS = AWS_V3_RUN_STATE_KEYS | {
    "checkpoint_receipt",
    "checkpoint_objects",
    "prior_command_ids",
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
AWS_EVALUATION_STATE_KEYS = {
    "schema_version",
    "provider",
    "seed",
    "release_sha256",
    "run_manifest_sha256",
    "dataset_sha256",
    "dataset_pointer_sha256",
    "dataset_verification_sha256",
    "environment_receipt_sha256",
    "profile_sha256",
    "runtime_sha256",
    "ami_id",
    "container_digest",
    "instance_id",
    "terminate_at",
    "operation_id",
    "intent_sha256",
    "intent_uri",
    "command_id",
    "operation",
    "status",
    "send_attempted",
    "created_at",
    "updated_at",
}
AWS_PAIR_INTENT_KEYS = {
    "schema_version",
    "provider",
    "run_manifest_sha256",
    "operation_id",
    "states",
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
_AWS_PROVIDER = "aws-p5.48xlarge"
_AWS_INSTANCE_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
_AWS_COMMAND_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{7,127}$")
_AWS_AMI_RE = re.compile(r"^ami-[0-9a-f]{8,17}$")
_AWS_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_AWS_STATUSES = {
    "INTENT_PUBLISHED",
    "SENDING",
    "Pending",
    "InProgress",
    "Delayed",
    "Success",
    "Failed",
    "Cancelled",
    "TimedOut",
    "Cancelling",
    "Terminating",
    "Submitting",
}
_AWS_PAIR_RUN_FIELDS = {
    "run_id",
    "arm",
    "config_sha256",
    "created_at",
    "updated_at",
}
_AWS_PAIR_SAME_OPERATION_MUTABLE_FIELDS = {
    "command_id",
    "send_attempted",
    "status",
    "updated_at",
}
_AWS_PAIR_RESUME_TRANSITION_FIELDS = {
    "attempt",
    "checkpoint_objects",
    "checkpoint_receipt",
    "command_id",
    "intent_sha256",
    "intent_uri",
    "operation",
    "operation_id",
    "prior_command_ids",
    "send_attempted",
    "status",
    "updated_at",
}


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
    if value.get("provider") == _AWS_PROVIDER:
        _validate_aws_run_state(value, run_id)
        return
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


def _validate_aws_run_state(value: dict[str, object], run_id: str) -> None:
    if value.get("schema_version") == 2:
        _validate_aws_v3_run_state(value, run_id)
        return
    operation = value.get("operation")
    keys = AWS_RESUME_STATE_KEYS if operation == "resume" else AWS_RUN_STATE_KEYS
    require_exact_keys(value, keys, label="AWS run state")
    require_schema_version(
        value["schema_version"],
        label="AWS run state.schema_version",
    )
    if (
        operation not in {"submit", "resume"}
        or value["run_id"] != run_id
        or RUN_ID_RE.fullmatch(run_id) is None
        or value["arm"] not in {"dense", "split90"}
        or isinstance(value["seed"], bool)
        or value["seed"] not in {1, 2, 3, 4}
    ):
        raise MsctlError("STATE_CORRUPT", "AWS run identity is invalid")
    for field in (
        "release_sha256",
        "run_manifest_sha256",
        "config_sha256",
        "dataset_sha256",
        "dataset_pointer_sha256",
        "dataset_verification_sha256",
        "environment_receipt_sha256",
        "cohort_assignment_sha256",
        "study_lock_sha256",
        "profile_sha256",
        "runtime_sha256",
        "operation_id",
        "intent_sha256",
    ):
        require_sha256(value[field], label=f"AWS run state {field}")
    if (
        not isinstance(value["source_commit"], str)
        or _COMMIT_RE.fullmatch(value["source_commit"]) is None
        or not isinstance(value["ami_id"], str)
        or _AWS_AMI_RE.fullmatch(value["ami_id"]) is None
        or not isinstance(value["container_digest"], str)
        or _AWS_DIGEST_RE.fullmatch(value["container_digest"]) is None
        or not isinstance(value["instance_id"], str)
        or _AWS_INSTANCE_RE.fullmatch(value["instance_id"]) is None
        or not isinstance(value["terminate_at"], str)
        or not value["terminate_at"].endswith("Z")
        or not isinstance(value["intent_uri"], str)
        or not value["intent_uri"].startswith("s3://")
    ):
        raise MsctlError("STATE_CORRUPT", "AWS execution binding is invalid")
    command_id = value["command_id"]
    if command_id is not None and (
        not isinstance(command_id, str)
        or _AWS_COMMAND_RE.fullmatch(command_id) is None
    ):
        raise MsctlError("STATE_CORRUPT", "AWS command ID is invalid")
    if value["status"] not in _AWS_STATUSES:
        raise MsctlError("STATE_CORRUPT", "AWS run status is invalid")
    attempt = require_nonnegative_int(
        value["attempt"],
        label="AWS run attempt",
    )
    if attempt < 1 or not isinstance(value["send_attempted"], bool):
        raise MsctlError("STATE_CORRUPT", "AWS send attempt is invalid")
    for field in ("created_at", "updated_at"):
        _require_string(value[field], label=f"AWS run state {field}")
    if operation == "resume":
        require_sha256(
            value["checkpoint_receipt_sha256"],
            label="AWS checkpoint receipt SHA-256",
        )
        prior = value["prior_command_ids"]
        if (
            not isinstance(prior, list)
            or not prior
            or any(
                not isinstance(item, str)
                or _AWS_COMMAND_RE.fullmatch(item) is None
                for item in prior
            )
            or len(prior) != len(set(prior))
        ):
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS prior command IDs are invalid",
            )


def _validate_aws_v3_run_state(
    value: dict[str, object],
    run_id: str,
) -> None:
    operation = value.get("operation")
    keys = (
        AWS_V3_RESUME_STATE_KEYS
        if operation == "resume"
        else AWS_V3_RUN_STATE_KEYS
    )
    require_exact_keys(value, keys, label="AWS v3 run state")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 2
        or operation not in {"submit", "resume"}
        or value["run_id"] != run_id
        or RUN_ID_RE.fullmatch(run_id) is None
        or value["arm"] not in {"dense", "split90"}
        or type(value["seed"]) is not int
        or value["seed"] not in range(10)
        or value["provider"] != _AWS_PROVIDER
    ):
        raise MsctlError("STATE_CORRUPT", "AWS v3 run identity is invalid")
    for field in (
        "release_sha256",
        "release_receipt_sha256",
        "run_manifest_sha256",
        "config_sha256",
        "dataset_pointer_sha256",
        "dataset_receipt_sha256",
        "dataset_build_id",
        "ordered_stream_sha256",
        "dataset_verification_sha256",
        "environment_receipt_sha256",
        "cohort_assignment_sha256",
        "preregistration_sha256",
        "profile_sha256",
        "runtime_sha256",
        "operation_id",
        "intent_sha256",
    ):
        require_sha256(value[field], label=f"AWS v3 run state {field}")
    if (
        not isinstance(value["source_commit"], str)
        or _COMMIT_RE.fullmatch(value["source_commit"]) is None
        or not isinstance(value["source_tree"], str)
        or _COMMIT_RE.fullmatch(value["source_tree"]) is None
        or not isinstance(value["ami_id"], str)
        or _AWS_AMI_RE.fullmatch(value["ami_id"]) is None
        or not isinstance(value["container_digest"], str)
        or _AWS_DIGEST_RE.fullmatch(value["container_digest"]) is None
        or not isinstance(value["instance_id"], str)
        or _AWS_INSTANCE_RE.fullmatch(value["instance_id"]) is None
        or not isinstance(value["terminate_at"], str)
        or not value["terminate_at"].endswith("Z")
        or not isinstance(value["intent_uri"], str)
        or not value["intent_uri"].startswith("s3://")
    ):
        raise MsctlError(
            "STATE_CORRUPT",
            "AWS v3 execution binding is invalid",
        )
    command_id = value["command_id"]
    if command_id is not None and (
        not isinstance(command_id, str)
        or _AWS_COMMAND_RE.fullmatch(command_id) is None
    ):
        raise MsctlError("STATE_CORRUPT", "AWS v3 command ID is invalid")
    if value["status"] not in _AWS_STATUSES:
        raise MsctlError("STATE_CORRUPT", "AWS v3 run status is invalid")
    attempt = require_nonnegative_int(
        value["attempt"],
        label="AWS v3 run attempt",
    )
    if attempt < 1 or not isinstance(value["send_attempted"], bool):
        raise MsctlError("STATE_CORRUPT", "AWS v3 send attempt is invalid")
    for field in ("created_at", "updated_at"):
        _require_string(value[field], label=f"AWS v3 state {field}")
    if operation != "resume":
        return
    receipt = require_object(
        value["checkpoint_receipt"],
        label="AWS v3 checkpoint receipt state",
    )
    require_exact_keys(
        receipt,
        {"sha256", "uri", "version_id"},
        label="AWS v3 checkpoint receipt state",
    )
    require_sha256(
        receipt["sha256"],
        label="AWS v3 checkpoint receipt state hash",
    )
    if (
        not isinstance(receipt["uri"], str)
        or not receipt["uri"].startswith("s3://")
        or not isinstance(receipt["version_id"], str)
        or receipt["version_id"] in {"", "null"}
    ):
        raise MsctlError(
            "STATE_CORRUPT",
            "AWS v3 checkpoint receipt state is invalid",
        )
    objects = value["checkpoint_objects"]
    if not isinstance(objects, list) or len(objects) != 2:
        raise MsctlError(
            "STATE_CORRUPT",
            "AWS v3 checkpoint object state is incomplete",
        )
    arms = set()
    for raw in objects:
        row = require_object(raw, label="AWS v3 checkpoint object state")
        require_exact_keys(
            row,
            {"arm", "bytes", "sha256", "uri", "version_id"},
            label="AWS v3 checkpoint object state",
        )
        require_sha256(
            row["sha256"],
            label="AWS v3 checkpoint object state hash",
        )
        if (
            row["arm"] not in {"dense", "split90"}
            or row["arm"] in arms
            or type(row["bytes"]) is not int
            or row["bytes"] <= 0
            or not isinstance(row["uri"], str)
            or not row["uri"].startswith("s3://")
            or not isinstance(row["version_id"], str)
            or row["version_id"] in {"", "null"}
        ):
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS v3 checkpoint object state is invalid",
            )
        arms.add(row["arm"])
    if arms != {"dense", "split90"}:
        raise MsctlError(
            "STATE_CORRUPT",
            "AWS v3 checkpoint object state arms are incomplete",
        )
    prior = value["prior_command_ids"]
    if (
        not isinstance(prior, list)
        or not prior
        or any(
            not isinstance(item, str)
            or _AWS_COMMAND_RE.fullmatch(item) is None
            for item in prior
        )
        or len(prior) != len(set(prior))
    ):
        raise MsctlError(
            "STATE_CORRUPT",
            "AWS v3 prior command IDs are invalid",
        )


def _validate_evaluation_state(
    value: dict[str, object],
    manifest_sha256: str,
) -> None:
    if value.get("provider") == _AWS_PROVIDER:
        require_exact_keys(
            value,
            AWS_EVALUATION_STATE_KEYS,
            label="AWS evaluation state",
        )
        require_schema_version(
            value["schema_version"],
            label="AWS evaluation state.schema_version",
        )
        for field in (
            "release_sha256",
            "run_manifest_sha256",
            "dataset_sha256",
            "dataset_pointer_sha256",
            "dataset_verification_sha256",
            "environment_receipt_sha256",
            "profile_sha256",
            "runtime_sha256",
            "operation_id",
            "intent_sha256",
        ):
            require_sha256(
                value[field],
                label=f"AWS evaluation state {field}",
            )
        if (
            value["run_manifest_sha256"] != manifest_sha256
            or isinstance(value["seed"], bool)
            or value["seed"] not in {1, 2, 3, 4}
            or not isinstance(value["ami_id"], str)
            or _AWS_AMI_RE.fullmatch(value["ami_id"]) is None
            or not isinstance(value["container_digest"], str)
            or _AWS_DIGEST_RE.fullmatch(value["container_digest"]) is None
            or not isinstance(value["instance_id"], str)
            or _AWS_INSTANCE_RE.fullmatch(value["instance_id"]) is None
            or not isinstance(value["terminate_at"], str)
            or not value["terminate_at"].endswith("Z")
            or not isinstance(value["intent_uri"], str)
            or not value["intent_uri"].startswith("s3://")
            or value["operation"] != "evaluate"
            or not isinstance(value["send_attempted"], bool)
            or value["status"] not in _AWS_STATUSES
        ):
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS evaluation identity is invalid",
            )
        command_id = value["command_id"]
        if command_id is not None and (
            not isinstance(command_id, str)
            or _AWS_COMMAND_RE.fullmatch(command_id) is None
        ):
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS evaluation command ID is invalid",
            )
        for field in ("created_at", "updated_at"):
            _require_string(
                value[field],
                label=f"AWS evaluation state {field}",
            )
        return
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

    def _aws_pair_name(self, manifest_sha256: str) -> str:
        try:
            require_sha256(manifest_sha256, label="AWS pair manifest")
        except MsctlError as error:
            raise MsctlError(
                "UNSAFE_STATE",
                "invalid AWS pair manifest identity",
            ) from error
        return f"aws-{manifest_sha256}.json"

    def _aws_pair_failure_name(self, manifest_sha256: str) -> str:
        self._aws_pair_name(manifest_sha256)
        return f"aws-{manifest_sha256}.rollback-failed"

    def _require_aws_pair_not_failed(
        self,
        intents_fd: int,
        manifest_sha256: str,
    ) -> None:
        name = self._aws_pair_failure_name(manifest_sha256)
        try:
            metadata = os.stat(
                name,
                dir_fd=intents_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        if not stat.S_ISREG(metadata.st_mode):
            raise MsctlError(
                "UNSAFE_STATE",
                "AWS pair rollback marker is not a regular file",
            )
        raise MsctlError(
            "STATE_ROLLBACK_FAILED",
            "AWS pair state is blocked after an incomplete rollback",
            details={"manifest_sha256": manifest_sha256},
        )

    def _mark_aws_pair_failed(
        self,
        intents_fd: int,
        manifest_sha256: str,
    ) -> None:
        name = self._aws_pair_failure_name(manifest_sha256)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=intents_fd)
        except FileExistsError:
            self._require_aws_pair_not_failed(
                intents_fd,
                manifest_sha256,
            )
            raise AssertionError("unreachable")
        try:
            payload = b"paired state rollback incomplete\n"
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(intents_fd)

    def _validate_aws_pair(
        self,
        value: dict[str, object],
        manifest_sha256: str,
    ) -> None:
        require_exact_keys(value, AWS_PAIR_INTENT_KEYS, label="AWS pair intent")
        schema_version = value["schema_version"]
        if type(schema_version) is not int or schema_version not in {1, 2}:
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS pair intent schema version is invalid",
            )
        if (
            value["provider"] != _AWS_PROVIDER
            or value["run_manifest_sha256"] != manifest_sha256
        ):
            raise MsctlError("STATE_CORRUPT", "AWS pair intent identity is invalid")
        require_sha256(value["operation_id"], label="AWS pair operation")
        states = value["states"]
        if not isinstance(states, list) or len(states) != 2:
            raise MsctlError("STATE_CORRUPT", "AWS pair intent is incomplete")
        run_ids: set[str] = set()
        arms: set[str] = set()
        shared_binding: dict[str, object] | None = None
        for raw in states:
            state = require_object(raw, label="AWS pair state")
            run_id = state.get("run_id")
            if not isinstance(run_id, str):
                raise MsctlError("STATE_CORRUPT", "AWS pair run ID is invalid")
            _validate_aws_run_state(state, run_id)
            if (
                state["schema_version"] != schema_version
                or state["run_manifest_sha256"] != manifest_sha256
                or state["operation_id"] != value["operation_id"]
                or run_id in run_ids
            ):
                raise MsctlError(
                    "STATE_CORRUPT",
                    "AWS pair state does not match its durable intent",
                )
            run_ids.add(run_id)
            arms.add(str(state["arm"]))
            current_shared = {
                key: field
                for key, field in state.items()
                if key not in _AWS_PAIR_RUN_FIELDS
            }
            if shared_binding is None:
                shared_binding = current_shared
            elif current_shared != shared_binding:
                raise MsctlError(
                    "STATE_CORRUPT",
                    "AWS pair states have divergent shared bindings",
                )
        if arms != {"dense", "split90"}:
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS pair intent must contain both exact arms",
            )

    def read_aws_pair(
        self,
        manifest_sha256: str,
    ) -> dict[str, object] | None:
        _, _, _, intents_fd = self._require_locked()
        self._require_aws_pair_not_failed(intents_fd, manifest_sha256)
        try:
            raw = load_json_at(
                intents_fd,
                self._aws_pair_name(manifest_sha256),
                label="AWS pair intent",
            )
        except MsctlError as error:
            if error.code == "FILE_NOT_FOUND":
                return None
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS pair intent is unreadable",
            ) from error
        value = require_object(raw, label="AWS pair intent")
        self._validate_aws_pair(value, manifest_sha256)
        return value

    def write_aws_pair(
        self,
        manifest_sha256: str,
        value: dict[str, object],
    ) -> None:
        _, _, _, intents_fd = self._require_locked()
        self._require_aws_pair_not_failed(intents_fd, manifest_sha256)
        self._validate_aws_pair(value, manifest_sha256)
        atomic_write_json_at(
            intents_fd,
            self._aws_pair_name(manifest_sha256),
            value,
            label="AWS pair intent",
        )

    @staticmethod
    def _read_state_bytes(
        directory_fd: int,
        name: str,
        *,
        label: str,
    ) -> bytes | None:
        try:
            descriptor, parent_fd, _ = open_regular_at(
                directory_fd,
                name,
                label=label,
            )
        except MsctlError as error:
            if error.code == "FILE_NOT_FOUND":
                return None
            raise
        try:
            before = os.fstat(descriptor)
            data = read_fd(descriptor)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
            os.close(parent_fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or len(data) != after.st_size:
            raise MsctlError(
                "FILE_CHANGED",
                f"{label} changed while being snapshotted",
                details={"path": name},
            )
        return data

    @staticmethod
    def _remove_state_file(
        directory_fd: int,
        name: str,
        *,
        label: str,
    ) -> None:
        try:
            metadata = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        if not stat.S_ISREG(metadata.st_mode):
            raise MsctlError(
                "UNSAFE_STATE",
                f"{label} rollback target is not a regular file",
            )
        os.unlink(name, dir_fd=directory_fd)
        os.fsync(directory_fd)

    @staticmethod
    def _validate_aws_pair_transition(
        current_by_id: dict[str, dict[str, object]],
        next_by_id: dict[str, dict[str, object]],
    ) -> None:
        if set(current_by_id) != set(next_by_id):
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS pair transition changes the exact run set",
            )
        current_operations = {
            state["operation"] for state in current_by_id.values()
        }
        next_operations = {state["operation"] for state in next_by_id.values()}
        if len(current_operations) != 1 or len(next_operations) != 1:
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS pair transition has divergent operations",
            )
        current_operation = str(next(iter(current_operations)))
        next_operation = str(next(iter(next_operations)))
        semantic_changed = False
        if current_operation == next_operation:
            for run_id, current in current_by_id.items():
                following = next_by_id[run_id]
                for field in set(current) | set(following):
                    if (
                        field not in _AWS_PAIR_SAME_OPERATION_MUTABLE_FIELDS
                        and current.get(field) != following.get(field)
                    ):
                        raise MsctlError(
                            "STATE_CORRUPT",
                            "AWS pair transition changes immutable provenance",
                        )
                current_command = current.get("command_id")
                next_command = following.get("command_id")
                if (
                    current_command is not None
                    and next_command != current_command
                ):
                    raise MsctlError(
                        "STATE_CORRUPT",
                        "AWS pair transition changes its durable command",
                    )
                if (
                    current.get("send_attempted") is True
                    and following.get("send_attempted") is not True
                ):
                    raise MsctlError(
                        "STATE_CORRUPT",
                        "AWS pair transition clears its durable send marker",
                    )
                semantic_changed = semantic_changed or any(
                    current.get(field) != following.get(field)
                    for field in (
                        "command_id",
                        "send_attempted",
                        "status",
                    )
                )
        elif current_operation == "submit" and next_operation == "resume":
            current_commands = {
                state.get("command_id") for state in current_by_id.values()
            }
            if (
                len(current_commands) != 1
                or not isinstance(next(iter(current_commands)), str)
            ):
                raise MsctlError(
                    "STATE_CORRUPT",
                    "AWS resume transition lacks one prior command",
                )
            previous_command = str(next(iter(current_commands)))
            for run_id, current in current_by_id.items():
                following = next_by_id[run_id]
                for field in set(current) | set(following):
                    if (
                        field not in _AWS_PAIR_RESUME_TRANSITION_FIELDS
                        and current.get(field) != following.get(field)
                    ):
                        raise MsctlError(
                            "STATE_CORRUPT",
                            "AWS resume transition changes immutable provenance",
                        )
                if (
                    following.get("operation_id")
                    == current.get("operation_id")
                    or following.get("intent_sha256")
                    == current.get("intent_sha256")
                    or following.get("intent_uri") == current.get("intent_uri")
                    or following.get("command_id") is not None
                    or following.get("status") != "INTENT_PUBLISHED"
                    or following.get("send_attempted") is not False
                    or following.get("attempt") != current.get("attempt", 0) + 1
                    or following.get("prior_command_ids")
                    != [previous_command]
                ):
                    raise MsctlError(
                        "STATE_CORRUPT",
                        "AWS resume transition binding is invalid",
                    )
            semantic_changed = True
        else:
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS pair operation transition is invalid",
            )

        current_updated = {
            state["updated_at"] for state in current_by_id.values()
        }
        next_updated = {state["updated_at"] for state in next_by_id.values()}
        if semantic_changed:
            if len(next_updated) != 1:
                raise MsctlError(
                    "STATE_CORRUPT",
                    "AWS pair transition must use one update timestamp",
                )
        elif next_updated != current_updated:
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS pair replay changes only its update timestamp",
            )

    def write_aws_pair_transaction(
        self,
        manifest_sha256: str,
        value: dict[str, object],
        states: Sequence[dict[str, object]],
    ) -> None:
        """Install one AWS journal and its two run files as one rollback unit."""

        _, runs_fd, _, intents_fd = self._require_locked()
        self._validate_aws_pair(value, manifest_sha256)
        state_by_id: dict[str, dict[str, object]] = {}
        for state in states:
            run_id = state.get("run_id")
            if not isinstance(run_id, str) or run_id in state_by_id:
                raise MsctlError(
                    "STATE_CORRUPT",
                    "AWS pair transaction run IDs are invalid",
                )
            _validate_run_state(state, run_id)
            state_by_id[run_id] = state
        journal_by_id = {
            str(state["run_id"]): state for state in value["states"]
        }
        if (
            len(state_by_id) != 2
            or set(state_by_id) != set(journal_by_id)
            or any(
                state_by_id[run_id] != journal_by_id[run_id]
                for run_id in state_by_id
            )
        ):
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS pair transaction journal and run states differ",
            )

        current_journal = self.read_aws_pair(manifest_sha256)
        current_runs = {
            run_id: self.read_run(run_id) for run_id in sorted(state_by_id)
        }
        if current_journal is None:
            if any(state is not None for state in current_runs.values()):
                raise MsctlError(
                    "STATE_INCOMPLETE",
                    "AWS pair transaction lacks its existing journal",
                )
        else:
            current_by_id = {
                str(state["run_id"]): state
                for state in current_journal["states"]
            }
            if (
                set(current_by_id) != set(state_by_id)
                or any(
                    current_runs[run_id] is None
                    or current_runs[run_id] != current_by_id[run_id]
                    for run_id in state_by_id
                )
            ):
                raise MsctlError(
                    "STATE_CORRUPT",
                    "AWS pair transaction found a mixed existing generation",
                )
            self._validate_aws_pair_transition(
                current_by_id,
                state_by_id,
            )

        encoded_journal = canonical_json(value) + b"\n"
        parsed_journal = require_object(
            json.loads(encoded_journal),
            label="AWS pair transaction journal bytes",
        )
        self._validate_aws_pair(parsed_journal, manifest_sha256)
        targets: list[tuple[int, str, bytes, str]] = [
            (
                intents_fd,
                self._aws_pair_name(manifest_sha256),
                encoded_journal,
                "AWS pair intent",
            )
        ]
        for run_id in sorted(state_by_id):
            encoded_state = canonical_json(state_by_id[run_id]) + b"\n"
            parsed_state = require_object(
                json.loads(encoded_state),
                label="AWS pair transaction run bytes",
            )
            _validate_run_state(parsed_state, run_id)
            targets.append(
                (
                    runs_fd,
                    self._run_name(run_id),
                    encoded_state,
                    "run state",
                )
            )

        snapshots = [
            self._read_state_bytes(directory_fd, name, label=label)
            for directory_fd, name, _data, label in targets
        ]
        if all(
            previous == data
            for previous, (_directory_fd, _name, data, _label) in zip(
                snapshots,
                targets,
                strict=True,
            )
        ):
            return

        attempted: list[int] = []
        try:
            for index, (directory_fd, name, data, label) in enumerate(targets):
                attempted.append(index)
                atomic_write_at(
                    directory_fd,
                    name,
                    data,
                    label=label,
                )
        except Exception as error:
            rollback_errors: list[str] = []
            for index in reversed(attempted):
                directory_fd, name, _data, label = targets[index]
                previous = snapshots[index]
                try:
                    if previous is None:
                        self._remove_state_file(
                            directory_fd,
                            name,
                            label=label,
                        )
                    else:
                        atomic_write_at(
                            directory_fd,
                            name,
                            previous,
                            label=f"{label} rollback",
                        )
                except Exception as rollback_error:
                    rollback_errors.append(
                        f"{name}: {type(rollback_error).__name__}"
                    )
            if rollback_errors:
                try:
                    self._mark_aws_pair_failed(
                        intents_fd,
                        manifest_sha256,
                    )
                except Exception as marker_error:
                    rollback_errors.append(
                        "rollback marker: "
                        f"{type(marker_error).__name__}"
                    )
                raise MsctlError(
                    "STATE_ROLLBACK_FAILED",
                    "AWS pair state rollback failed closed",
                    details={"failures": rollback_errors},
                ) from error
            raise MsctlError(
                "STATE_WRITE_FAILED",
                "AWS pair state transaction failed; prior bytes restored",
            ) from error
