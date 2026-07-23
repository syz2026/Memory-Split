"""High-level dry-run-first lifecycle operations."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from .approval import verify_approval
from .contracts import (
    RunManifest,
    bind_release,
    load_release,
    load_run_manifest,
    verify_checkpoint_receipt,
)
from .errors import MsctlError
from .profile import IlluminaProfile
from .slurm import (
    ACTIVE_STATES,
    capacity_check as slurm_capacity_check,
    query_states,
    render_evaluate_command,
    render_seed0_command,
    run_command,
    submit as slurm_submit,
)
from .state import StateStore


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_bound_inputs(
    *,
    release_path: Path | str,
    manifest_path: Path | str,
    repo_root: Path | str,
):
    release = load_release(release_path)
    manifest = load_run_manifest(manifest_path, repo_root=repo_root)
    bind_release(release, manifest)
    return release, manifest


def render_runs(
    *,
    profile: IlluminaProfile,
    release_path: Path | str,
    manifest_path: Path | str,
    repo_root: Path | str,
) -> dict[str, object]:
    release, manifest = load_bound_inputs(
        release_path=release_path,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    return {
        "provider": profile.provider,
        "profile_sha256": profile.sha256,
        "release_sha256": release.archive_sha256,
        "run_manifest_sha256": manifest.sha256,
        "layout": {
            "allocated_gpus": profile.allocated_gpus,
            "train_groups": list(profile.train_groups),
            "evaluation_gpus": profile.evaluation_gpus,
        },
        "commands": [
            render_seed0_command(profile, release, manifest),
        ],
    }


def _same_binding(state: dict[str, object], manifest: RunManifest) -> bool:
    return (
        state.get("run_manifest_sha256") == manifest.sha256
        and state.get("release_sha256") == manifest.release_sha256
        and state.get("provider") == manifest.provider
    )


def submit_runs(
    *,
    profile: IlluminaProfile,
    release_path: Path | str,
    manifest_path: Path | str,
    repo_root: Path | str,
    state_root: Path | str,
    approval_path: Path | str | None,
    apply: bool,
    environ: dict[str, str] | None = None,
) -> dict[str, object]:
    release, manifest = load_bound_inputs(
        release_path=release_path,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    command = render_seed0_command(profile, release, manifest)
    if not apply:
        return {
            "provider": profile.provider,
            "release_sha256": release.archive_sha256,
            "run_manifest_sha256": manifest.sha256,
            "commands": [command],
            "submitted": 0,
            "idempotent": False,
        }
    verify_approval(
        approval_path,
        operation="submit",
        release_sha256=release.archive_sha256,
        run_manifest=manifest,
        profile=profile,
        environ=(dict(os.environ) if environ is None else environ),
    )
    store = StateStore(state_root)
    with store.locked():
        states = [store.read_run(run.run_id) for run in manifest.runs]
        existing = [state for state in states if state is not None]
        if existing:
            if len(existing) != len(manifest.runs):
                raise MsctlError(
                    "STATE_INCOMPLETE",
                    "only part of the paired run has lifecycle state",
                )
            if not all(_same_binding(state, manifest) for state in existing):
                raise MsctlError(
                    "RUN_ID_CONFLICT",
                    "run ID already exists with different provenance",
                )
            job_ids = {state.get("job_id") for state in existing}
            if None in job_ids or len(job_ids) != 1:
                raise MsctlError(
                    "SUBMISSION_UNCERTAIN",
                    "prior submission intent lacks one authoritative job ID",
                )
            job_id = str(next(iter(job_ids)))
            reconciled = query_states([job_id], environ=environ)
            status = reconciled[job_id]
            for run, state in zip(manifest.runs, existing):
                updated = dict(state)
                updated["status"] = status
                updated["updated_at"] = _timestamp()
                store.write_run(run.run_id, updated)
            return {
                "provider": profile.provider,
                "release_sha256": release.archive_sha256,
                "run_manifest_sha256": manifest.sha256,
                "job_id": job_id,
                "status": status,
                "submitted": 0,
                "idempotent": True,
                "active": status in ACTIVE_STATES,
            }

        intent_time = _timestamp()
        for run in manifest.runs:
            store.write_run(
                run.run_id,
                {
                    "schema_version": 1,
                    "run_id": run.run_id,
                    "arm": run.arm,
                    "seed": run.seed,
                    "provider": profile.provider,
                    "release_sha256": release.archive_sha256,
                    "run_manifest_sha256": manifest.sha256,
                    "config_sha256": run.config_sha256,
                    "dataset_sha256": manifest.dataset_sha256,
                    "job_id": None,
                    "status": "SUBMITTING",
                    "attempt": 1,
                    "created_at": intent_time,
                    "updated_at": intent_time,
                },
            )
        job_id = slurm_submit(command, environ=environ)
        submitted_at = _timestamp()
        for run in manifest.runs:
            state = store.read_run(run.run_id)
            assert state is not None
            state["job_id"] = job_id
            state["status"] = "SUBMITTED"
            state["updated_at"] = submitted_at
            store.write_run(run.run_id, state)
        return {
            "provider": profile.provider,
            "release_sha256": release.archive_sha256,
            "run_manifest_sha256": manifest.sha256,
            "job_id": job_id,
            "status": "SUBMITTED",
            "submitted": 1,
            "idempotent": False,
        }


def check_capacity(
    profile: IlluminaProfile,
    *,
    environ: dict[str, str] | None = None,
) -> dict[str, object]:
    return slurm_capacity_check(profile, environ=environ)


def _paired_states(
    store: StateStore,
    manifest: RunManifest,
) -> list[dict[str, object]]:
    states = [store.read_run(run.run_id) for run in manifest.runs]
    if any(state is None for state in states):
        raise MsctlError(
            "RUN_STATE_MISSING",
            "paired run lifecycle state is incomplete",
        )
    present = [state for state in states if state is not None]
    if not all(_same_binding(state, manifest) for state in present):
        raise MsctlError(
            "RUN_ID_CONFLICT",
            "run lifecycle state has different provenance",
        )
    return present


def _aggregate_status(states: list[dict[str, object]]) -> str:
    values = {str(state.get("status", "UNKNOWN")) for state in states}
    return next(iter(values)) if len(values) == 1 else "MIXED"


def status_runs(
    *,
    manifest_path: Path | str,
    repo_root: Path | str,
    state_root: Path | str,
    cached: bool,
    environ: dict[str, str] | None = None,
) -> dict[str, object]:
    manifest = load_run_manifest(manifest_path, repo_root=repo_root)
    store = StateStore(state_root)
    with store.locked():
        states = _paired_states(store, manifest)
        if cached:
            return {
                "run_manifest_sha256": manifest.sha256,
                "status": _aggregate_status(states),
                "authoritative": False,
                "runs": states,
            }
        job_ids = {
            str(state["job_id"])
            for state in states
            if state.get("job_id") is not None
        }
        if len(job_ids) != 1:
            raise MsctlError(
                "SUBMISSION_UNCERTAIN",
                "paired runs do not bind one Slurm job ID",
            )
        reconciled = query_states(list(job_ids), environ=environ)
        status = reconciled[next(iter(job_ids))]
        now = _timestamp()
        for run, state in zip(manifest.runs, states):
            state["status"] = status
            state["updated_at"] = now
            store.write_run(run.run_id, state)
        return {
            "run_manifest_sha256": manifest.sha256,
            "status": status,
            "authoritative": True,
            "runs": states,
        }


def resume_runs(
    *,
    profile: IlluminaProfile,
    release_path: Path | str,
    manifest_path: Path | str,
    checkpoint_receipt: Path | str,
    repo_root: Path | str,
    state_root: Path | str,
    approval_path: Path | str | None,
    apply: bool,
    environ: dict[str, str] | None = None,
) -> dict[str, object]:
    release, manifest = load_bound_inputs(
        release_path=release_path,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    verify_checkpoint_receipt(
        checkpoint_receipt,
        release=release,
        manifest=manifest,
    )
    command = render_seed0_command(profile, release, manifest, resume=True)
    if not apply:
        return {
            "command": command,
            "submitted": 0,
            "run_manifest_sha256": manifest.sha256,
        }
    verify_approval(
        approval_path,
        operation="resume",
        release_sha256=release.archive_sha256,
        run_manifest=manifest,
        profile=profile,
        environ=(dict(os.environ) if environ is None else environ),
    )
    store = StateStore(state_root)
    with store.locked():
        states = _paired_states(store, manifest)
        previous_ids = {
            str(state["job_id"])
            for state in states
            if state.get("job_id") is not None
        }
        if len(previous_ids) != 1:
            raise MsctlError(
                "SUBMISSION_UNCERTAIN",
                "resume requires one previous paired Slurm job",
            )
        previous_id = next(iter(previous_ids))
        previous_status = query_states([previous_id], environ=environ)[previous_id]
        if previous_status in ACTIVE_STATES:
            raise MsctlError(
                "RUN_ALREADY_ACTIVE",
                "refusing to resume an active Slurm job",
                details={"job_id": previous_id, "status": previous_status},
            )
        if previous_status == "COMPLETED":
            raise MsctlError(
                "RUN_ALREADY_COMPLETE",
                "refusing to resume a completed run",
            )
        attempt = max(int(state.get("attempt", 1)) for state in states) + 1
        intent_time = _timestamp()
        for run, state in zip(manifest.runs, states):
            prior = list(state.get("prior_job_ids", []))
            prior.append(previous_id)
            state.update(
                {
                    "job_id": None,
                    "status": "RESUBMITTING",
                    "attempt": attempt,
                    "prior_job_ids": prior,
                    "updated_at": intent_time,
                }
            )
            store.write_run(run.run_id, state)
        job_id = slurm_submit(command, environ=environ)
        submitted_at = _timestamp()
        for run in manifest.runs:
            state = store.read_run(run.run_id)
            assert state is not None
            state.update(
                {
                    "job_id": job_id,
                    "status": "SUBMITTED",
                    "updated_at": submitted_at,
                }
            )
            store.write_run(run.run_id, state)
        return {
            "job_id": job_id,
            "status": "SUBMITTED",
            "attempt": attempt,
            "submitted": 1,
            "run_manifest_sha256": manifest.sha256,
        }


def cancel_runs(
    *,
    profile: IlluminaProfile,
    release_path: Path | str,
    manifest_path: Path | str,
    repo_root: Path | str,
    state_root: Path | str,
    approval_path: Path | str | None,
    apply: bool,
    environ: dict[str, str] | None = None,
) -> dict[str, object]:
    release, manifest = load_bound_inputs(
        release_path=release_path,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    if not apply:
        return {
            "operation": "cancel",
            "run_ids": [run.run_id for run in manifest.runs],
            "cancelled": 0,
        }
    verify_approval(
        approval_path,
        operation="cancel",
        release_sha256=release.archive_sha256,
        run_manifest=manifest,
        profile=profile,
        environ=(dict(os.environ) if environ is None else environ),
    )
    store = StateStore(state_root)
    with store.locked():
        states = _paired_states(store, manifest)
        job_ids = {
            str(state["job_id"])
            for state in states
            if state.get("job_id") is not None
        }
        if len(job_ids) != 1:
            raise MsctlError(
                "SUBMISSION_UNCERTAIN",
                "cancel requires one paired Slurm job",
            )
        job_id = next(iter(job_ids))
        status = query_states([job_id], environ=environ)[job_id]
        if status not in ACTIVE_STATES:
            raise MsctlError(
                "RUN_NOT_ACTIVE",
                "refusing to cancel a terminal Slurm job",
                details={"status": status},
            )
        run_command(
            ["scancel", job_id],
            operation="cancel",
            environ=environ,
        )
        now = _timestamp()
        for run, state in zip(manifest.runs, states):
            state["status"] = "CANCEL_REQUESTED"
            state["updated_at"] = now
            store.write_run(run.run_id, state)
        return {
            "job_id": job_id,
            "status": "CANCEL_REQUESTED",
            "cancelled": 1,
        }


def evaluate_runs(
    *,
    profile: IlluminaProfile,
    release_path: Path | str,
    manifest_path: Path | str,
    repo_root: Path | str,
    state_root: Path | str,
    approval_path: Path | str | None,
    apply: bool,
    environ: dict[str, str] | None = None,
) -> dict[str, object]:
    release, manifest = load_bound_inputs(
        release_path=release_path,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    command = render_evaluate_command(profile, release, manifest)
    if not apply:
        return {
            "command": command,
            "submitted": 0,
            "run_manifest_sha256": manifest.sha256,
        }
    verify_approval(
        approval_path,
        operation="evaluate",
        release_sha256=release.archive_sha256,
        run_manifest=manifest,
        profile=profile,
        environ=(dict(os.environ) if environ is None else environ),
    )
    store = StateStore(state_root)
    with store.locked():
        existing = store.read_evaluation(manifest.sha256)
        if existing is not None:
            if (
                existing.get("release_sha256") != release.archive_sha256
                or existing.get("job_id") is None
            ):
                raise MsctlError(
                    "RUN_ID_CONFLICT",
                    "evaluation state has different provenance",
                )
            job_id = str(existing["job_id"])
            status = query_states([job_id], environ=environ)[job_id]
            existing["status"] = status
            existing["updated_at"] = _timestamp()
            store.write_evaluation(manifest.sha256, existing)
            return {
                "command": command,
                "job_id": job_id,
                "status": status,
                "submitted": 0,
                "idempotent": True,
            }
        now = _timestamp()
        store.write_evaluation(
            manifest.sha256,
            {
                "schema_version": 1,
                "provider": profile.provider,
                "release_sha256": release.archive_sha256,
                "run_manifest_sha256": manifest.sha256,
                "job_id": None,
                "status": "SUBMITTING",
                "created_at": now,
                "updated_at": now,
            },
        )
        job_id = slurm_submit(command, environ=environ)
        state = store.read_evaluation(manifest.sha256)
        assert state is not None
        state["job_id"] = job_id
        state["status"] = "SUBMITTED"
        state["updated_at"] = _timestamp()
        store.write_evaluation(manifest.sha256, state)
        return {
            "command": command,
            "job_id": job_id,
            "status": "SUBMITTED",
            "submitted": 1,
            "idempotent": False,
        }
