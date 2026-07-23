"""High-level dry-run-first lifecycle operations."""

from __future__ import annotations

import ast
import os
from datetime import UTC, datetime
from pathlib import Path

from .approval import verify_approval
from .contracts import (
    RunManifest,
    bind_release,
    load_release,
    load_run_manifest,
    read_release_member,
    verify_release_extraction,
    verify_release_member,
    verify_checkpoint_receipt,
)
from .dataset import load_dataset_verification, verify_dataset
from .environment import verify_environment_receipt
from .errors import MsctlError
from .profile import IlluminaProfile
from .slurm import (
    ACTIVE_STATES,
    EVALUATE_SCRIPT,
    EVALUATOR_ENTRYPOINT,
    RESUMABLE_TERMINAL_STATES,
    SEED0_SCRIPT,
    TRAIN_ENTRYPOINT,
    capacity_check as slurm_capacity_check,
    discover_jobs,
    query_states,
    render_evaluate_command,
    render_seed0_command,
    resource_request,
    runtime_binding_sha256,
    run_command,
    submission_identity,
    submit as slurm_submit,
)
from .state import StateStore


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_bound_inputs(
    *,
    profile: IlluminaProfile,
    release_path: Path | str,
    manifest_path: Path | str,
    repo_root: Path | str,
):
    release = load_release(release_path)
    release_root = verify_release_extraction(release, repo_root)
    manifest = load_run_manifest(manifest_path, repo_root=release_root)
    bind_release(release, manifest)
    if release.metadata.get("profile_sha256") != profile.source_sha256:
        raise MsctlError(
            "PROFILE_RELEASE_MISMATCH",
            "local provider profile differs from the verified release",
        )
    for script in (SEED0_SCRIPT, EVALUATE_SCRIPT):
        verify_release_member(
            release,
            member_path=script,
            local_path=release_root / script,
            label="Slurm entrypoint",
        )
    for run in manifest.runs:
        verify_release_member(
            release,
            member_path=run.config,
            local_path=release_root / run.config,
            label="run config",
        )
    return release, manifest, release_root


def _preflight_entrypoint(
    *,
    release,
    release_root: Path,
    member: str,
    marker_name: str,
    marker_value: str,
    required_interface: set[str],
) -> None:
    try:
        data = read_release_member(
            release,
            member_path=member,
            release_root=release_root,
            label="runtime entrypoint",
        )
        tree = ast.parse(data, filename=member)
        marker = None
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id == marker_name
                for target in node.targets
            ):
                marker = ast.literal_eval(node.value)
        strings = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
    except (OSError, SyntaxError, TypeError, ValueError) as error:
        raise MsctlError(
            "RUNTIME_PREFLIGHT_FAILED",
            "authenticated runtime entrypoint could not be inspected",
            details={"member": member},
        ) from error
    if marker != marker_value or not required_interface <= strings:
        raise MsctlError(
            "RUNTIME_PREFLIGHT_FAILED",
            "authenticated runtime entrypoint does not implement its contract",
            details={"member": member},
        )


def _resolve_dataset_verification(
    *,
    profile: IlluminaProfile,
    dataset_pointer: Path | str,
    dataset_root: Path | str | None,
    dataset_verification: Path | str | None,
    release,
    manifest: RunManifest,
    repo_root: Path | str,
) -> dict[str, object]:
    if (dataset_root is None) == (dataset_verification is None):
        raise MsctlError(
            "DATASET_INPUT_INVALID",
            "provide exactly one of dataset root or prior verification",
        )
    if dataset_verification is not None:
        return load_dataset_verification(
            dataset_verification,
            profile=profile,
            pointer_path=dataset_pointer,
            release=release,
            manifest=manifest,
            repo_root=repo_root,
        )
    assert dataset_root is not None
    return verify_dataset(
        profile=profile,
        pointer_path=dataset_pointer,
        dataset_root=dataset_root,
        release=release,
        manifest=manifest,
        repo_root=repo_root,
    )


def render_runs(
    *,
    profile: IlluminaProfile,
    release_path: Path | str,
    manifest_path: Path | str,
    dataset_pointer: Path | str,
    dataset_root: Path | str | None,
    dataset_verification: Path | str | None,
    environment_receipt: Path | str | None,
    repo_root: Path | str,
) -> dict[str, object]:
    release, manifest, release_root = load_bound_inputs(
        profile=profile,
        release_path=release_path,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    dataset = _resolve_dataset_verification(
        profile=profile,
        dataset_pointer=dataset_pointer,
        dataset_root=dataset_root,
        dataset_verification=dataset_verification,
        release=release,
        manifest=manifest,
        repo_root=repo_root,
    )
    environment = verify_environment_receipt(
        environment_receipt,
        profile=profile,
        release=release,
    )
    resources = resource_request(profile, "submit")
    return {
        "provider": profile.provider,
        "profile_sha256": profile.sha256,
        "release_sha256": release.archive_sha256,
        "run_manifest_sha256": manifest.sha256,
        "dataset_verification": dataset,
        "resource_request": resources,
        "layout": {
            "allocated_gpus": profile.allocated_gpus,
            "train_groups": list(profile.train_groups),
            "evaluation_gpus": profile.evaluation_gpus,
        },
        "commands": [
            render_seed0_command(
                profile,
                release,
                manifest,
                release_root=release_root,
                dataset=dataset,
                environment=environment,
            ),
        ],
    }


def _same_binding(state: dict[str, object], manifest: RunManifest) -> bool:
    return (
        state.get("run_manifest_sha256") == manifest.sha256
        and state.get("release_sha256") == manifest.release_sha256
        and state.get("provider") == manifest.provider
        and state.get("dataset_sha256") == manifest.dataset_sha256
    )


def _new_pair_intent(
    *,
    profile: IlluminaProfile,
    manifest: RunManifest,
    submission_key: str,
    dataset: dict[str, object],
    environment: dict[str, object],
    resources: dict[str, object],
    operation: str,
    attempt: int,
    checkpoint_receipt_sha256: str | None,
) -> dict[str, object]:
    now = _timestamp()
    return {
        "schema_version": 1,
        "submission_key": submission_key,
        "provider": profile.provider,
        "release_sha256": manifest.release_sha256,
        "run_manifest_sha256": manifest.sha256,
        "dataset_sha256": manifest.dataset_sha256,
        "dataset_verification_sha256": dataset["verification_sha256"],
        "environment_receipt_sha256": environment["receipt_sha256"],
        "operation": operation,
        "resource_request": resources,
        "run_ids": [run.run_id for run in manifest.runs],
        "attempt": attempt,
        "checkpoint_receipt_sha256": checkpoint_receipt_sha256,
        "phase": "PREPARED",
        "job_id": None,
        "created_at": now,
        "updated_at": now,
    }


def _intent_matches(
    intent: dict[str, object],
    *,
    profile: IlluminaProfile,
    manifest: RunManifest,
    dataset: dict[str, object],
    environment: dict[str, object],
    operation: str,
    attempt: int,
    checkpoint_receipt_sha256: str | None,
) -> bool:
    return (
        intent["provider"] == profile.provider
        and intent["release_sha256"] == manifest.release_sha256
        and intent["run_manifest_sha256"] == manifest.sha256
        and intent["dataset_sha256"] == manifest.dataset_sha256
        and intent["dataset_verification_sha256"]
        == dataset["verification_sha256"]
        and intent["environment_receipt_sha256"]
        == environment["receipt_sha256"]
        and intent["operation"] == operation
        and intent["resource_request"] == resource_request(profile, operation)
        and intent["attempt"] == attempt
        and intent["checkpoint_receipt_sha256"]
        == checkpoint_receipt_sha256
        and intent["run_ids"] == [run.run_id for run in manifest.runs]
    )


def _initial_submit_state(
    *,
    run,
    manifest: RunManifest,
    intent: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run.run_id,
        "arm": run.arm,
        "seed": run.seed,
        "provider": intent["provider"],
        "release_sha256": intent["release_sha256"],
        "run_manifest_sha256": manifest.sha256,
        "config_sha256": run.config_sha256,
        "dataset_sha256": intent["dataset_sha256"],
        "dataset_verification_sha256": intent[
            "dataset_verification_sha256"
        ],
        "environment_receipt_sha256": intent[
            "environment_receipt_sha256"
        ],
        "operation": "submit",
        "submission_key": intent["submission_key"],
        "resource_request": intent["resource_request"],
        "job_id": intent["job_id"],
        "status": (
            "SUBMITTED" if intent["phase"] == "SUBMITTED" else "SUBMITTING"
        ),
        "attempt": 1,
        "created_at": intent["created_at"],
        "updated_at": intent["updated_at"],
    }


def _reconcile_pair_intent(
    *,
    store: StateStore,
    intent: dict[str, object],
    submission_key: str,
    job_name: str,
    environ: dict[str, str] | None,
) -> tuple[str, str] | None:
    if intent["phase"] == "PREPARED":
        return None
    if intent["phase"] == "SUBMITTED":
        job_id = str(intent["job_id"])
        return job_id, query_states([job_id], environ=environ)[job_id]
    discovered = discover_jobs(
        submission_key,
        job_name,
        environ=environ,
    )
    if not discovered:
        raise MsctlError(
            "SUBMISSION_UNCERTAIN",
            "Slurm cannot prove whether the durable pair intent was submitted",
            details={
                "recoverable": True,
                "submission_key": submission_key,
                "action": "reconcile the exact key; do not resubmit",
            },
        )
    if len(discovered) != 1:
        raise MsctlError(
            "SUBMISSION_MULTIPLE_MATCHES",
            "multiple Slurm jobs match one durable pair intent",
            details={
                "submission_key": submission_key,
                "job_ids": sorted(discovered, key=int),
            },
        )
    job_id, status = next(iter(discovered.items()))
    intent["phase"] = "SUBMITTED"
    intent["job_id"] = job_id
    intent["updated_at"] = _timestamp()
    store.write_intent(submission_key, intent)
    return job_id, status


def submit_runs(
    *,
    profile: IlluminaProfile,
    release_path: Path | str,
    manifest_path: Path | str,
    dataset_pointer: Path | str,
    dataset_root: Path | str | None,
    dataset_verification: Path | str | None,
    environment_receipt: Path | str | None,
    repo_root: Path | str,
    state_root: Path | str,
    approval_path: Path | str | None,
    apply: bool,
    environ: dict[str, str] | None = None,
) -> dict[str, object]:
    release, manifest, release_root = load_bound_inputs(
        profile=profile,
        release_path=release_path,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    dataset = _resolve_dataset_verification(
        profile=profile,
        dataset_pointer=dataset_pointer,
        dataset_root=dataset_root,
        dataset_verification=dataset_verification,
        release=release,
        manifest=manifest,
        repo_root=repo_root,
    )
    environment = verify_environment_receipt(
        environment_receipt,
        profile=profile,
        release=release,
    )
    command = render_seed0_command(
        profile,
        release,
        manifest,
        release_root=release_root,
        dataset=dataset,
        environment=environment,
    )
    resources = resource_request(profile, "submit")
    runtime_hash = runtime_binding_sha256(
        release_root=release_root,
        dataset=dataset,
        environment=environment,
    )
    submission_key, job_name, _ = submission_identity(
        profile,
        release,
        manifest,
        operation="submit",
        runtime_binding_sha256=runtime_hash,
    )
    if not apply:
        return {
            "provider": profile.provider,
            "release_sha256": release.archive_sha256,
            "run_manifest_sha256": manifest.sha256,
            "dataset_verification": dataset,
            "resource_request": resources,
            "submission_key": submission_key,
            "commands": [command],
            "submitted": 0,
            "idempotent": False,
        }
    _preflight_entrypoint(
        release=release,
        release_root=release_root,
        member=TRAIN_ENTRYPOINT,
        marker_name="MSCTL_DDP_CONTRACT",
        marker_value="memorysplit-ddp-v1",
        required_interface={"--config", "--resume-path"},
    )
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
        intent = store.read_intent(submission_key)
        states = [store.read_run(run.run_id) for run in manifest.runs]
        if intent is None:
            if any(state is not None for state in states):
                raise MsctlError(
                    "STATE_INCOMPLETE",
                    "paired run state exists without its durable intent",
                )
            intent = _new_pair_intent(
                profile=profile,
                manifest=manifest,
                submission_key=submission_key,
                dataset=dataset,
                environment=environment,
                resources=resources,
                operation="submit",
                attempt=1,
                checkpoint_receipt_sha256=None,
            )
            store.write_intent(submission_key, intent)
        elif not _intent_matches(
            intent,
            profile=profile,
            manifest=manifest,
            dataset=dataset,
            environment=environment,
            operation="submit",
            attempt=1,
            checkpoint_receipt_sha256=None,
        ):
            raise MsctlError(
                "RUN_ID_CONFLICT",
                "durable pair intent has different provenance",
            )

        repaired: list[dict[str, object]] = []
        for run, state in zip(manifest.runs, states):
            if state is None:
                state = _initial_submit_state(
                    run=run,
                    manifest=manifest,
                    intent=intent,
                )
                store.write_run(run.run_id, state)
            elif (
                not _same_binding(state, manifest)
                or state.get("submission_key") != submission_key
                or state.get("environment_receipt_sha256")
                != environment["receipt_sha256"]
            ):
                raise MsctlError(
                    "RUN_ID_CONFLICT",
                    "run ID already exists with different provenance",
                )
            repaired.append(state)

        recovered = _reconcile_pair_intent(
            store=store,
            intent=intent,
            submission_key=submission_key,
            job_name=job_name,
            environ=environ,
        )
        if recovered is not None:
            job_id, status = recovered
            now = _timestamp()
            for run, state in zip(manifest.runs, repaired):
                state["job_id"] = job_id
                state["status"] = status
                state["updated_at"] = now
                store.write_run(run.run_id, state)
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

        intent["phase"] = "SUBMITTING"
        intent["updated_at"] = _timestamp()
        store.write_intent(submission_key, intent)
        job_id = slurm_submit(command, environ=environ)
        submitted_at = _timestamp()
        intent["phase"] = "SUBMITTED"
        intent["job_id"] = job_id
        intent["updated_at"] = submitted_at
        store.write_intent(submission_key, intent)
        for run, state in zip(manifest.runs, repaired):
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
    profile: IlluminaProfile,
    release_path: Path | str,
    manifest_path: Path | str,
    repo_root: Path | str,
    state_root: Path | str,
    cached: bool,
    environ: dict[str, str] | None = None,
) -> dict[str, object]:
    release, manifest, _ = load_bound_inputs(
        profile=profile,
        release_path=release_path,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    store = StateStore(state_root)
    with store.locked():
        states = _paired_states(store, manifest)
        if cached:
            return {
                "run_manifest_sha256": manifest.sha256,
                "release_sha256": release.archive_sha256,
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
            "release_sha256": release.archive_sha256,
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
    dataset_pointer: Path | str,
    dataset_root: Path | str | None,
    dataset_verification: Path | str | None,
    environment_receipt: Path | str | None,
    repo_root: Path | str,
    state_root: Path | str,
    approval_path: Path | str | None,
    apply: bool,
    environ: dict[str, str] | None = None,
) -> dict[str, object]:
    release, manifest, release_root = load_bound_inputs(
        profile=profile,
        release_path=release_path,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    dataset = _resolve_dataset_verification(
        profile=profile,
        dataset_pointer=dataset_pointer,
        dataset_root=dataset_root,
        dataset_verification=dataset_verification,
        release=release,
        manifest=manifest,
        repo_root=repo_root,
    )
    environment = verify_environment_receipt(
        environment_receipt,
        profile=profile,
        release=release,
    )
    checkpoint = verify_checkpoint_receipt(
        checkpoint_receipt,
        release=release,
        manifest=manifest,
    )
    resources = resource_request(profile, "resume")
    runtime_hash = runtime_binding_sha256(
        release_root=release_root,
        dataset=dataset,
        environment=environment,
    )
    command = render_seed0_command(
        profile,
        release,
        manifest,
        release_root=release_root,
        dataset=dataset,
        environment=environment,
        checkpoint_receipt=checkpoint,
        attempt=2,
    )
    if not apply:
        return {
            "command": command,
            "submitted": 0,
            "run_manifest_sha256": manifest.sha256,
            "dataset_verification": dataset,
            "checkpoint_receipt_sha256": checkpoint.sha256,
            "resource_request": resources,
        }
    _preflight_entrypoint(
        release=release,
        release_root=release_root,
        member=TRAIN_ENTRYPOINT,
        marker_name="MSCTL_DDP_CONTRACT",
        marker_value="memorysplit-ddp-v1",
        required_interface={"--config", "--resume-path"},
    )
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
        if not all(
            state.get("dataset_verification_sha256")
            == dataset["verification_sha256"]
            and state.get("environment_receipt_sha256")
            == environment["receipt_sha256"]
            for state in states
        ):
            raise MsctlError(
                "DATASET_STATE_MISMATCH",
                "resume state was not created from this verified dataset",
            )
        candidate_attempt = max(int(state["attempt"]) for state in states)
        candidate_key, _, _ = submission_identity(
            profile,
            release,
            manifest,
            operation="resume",
            attempt=candidate_attempt,
            checkpoint_receipt_sha256=checkpoint.sha256,
            runtime_binding_sha256=runtime_hash,
        )
        candidate_intent = store.read_intent(candidate_key)
        if candidate_intent is not None:
            if not _intent_matches(
                candidate_intent,
                profile=profile,
                manifest=manifest,
                dataset=dataset,
                environment=environment,
                operation="resume",
                attempt=candidate_attempt,
                checkpoint_receipt_sha256=checkpoint.sha256,
            ):
                raise MsctlError(
                    "RUN_ID_CONFLICT",
                    "resume pair intent has different provenance",
                )
            previous_ids = {
                str(state["job_id"])
                for state in states
                if state["operation"] == "submit" and state["job_id"] is not None
            }
            for state in states:
                previous_ids.update(
                    str(job_id)
                    for job_id in state.get("prior_job_ids", [])
                )
            if len(previous_ids) != 1:
                raise MsctlError(
                    "SUBMISSION_UNCERTAIN",
                    "partial resume intent lacks one previous Slurm job",
                )
            previous_id = next(iter(previous_ids))
            repaired_at = _timestamp()
            for run, state in zip(manifest.runs, states):
                if state["operation"] == "resume":
                    continue
                state.update(
                    {
                        "job_id": None,
                        "status": "RESUBMITTING",
                        "attempt": candidate_attempt,
                        "operation": "resume",
                        "submission_key": candidate_key,
                        "resource_request": resources,
                        "checkpoint_receipt_sha256": checkpoint.sha256,
                        "prior_job_ids": [previous_id],
                        "updated_at": repaired_at,
                    }
                )
                store.write_run(run.run_id, state)
        existing_resume = all(
            state.get("operation") == "resume"
            and state.get("checkpoint_receipt_sha256") == checkpoint.sha256
            for state in states
        )
        if existing_resume:
            attempt = max(int(state.get("attempt", 1)) for state in states)
            submission_key, job_name, _ = submission_identity(
                profile,
                release,
                manifest,
                operation="resume",
                attempt=attempt,
                checkpoint_receipt_sha256=checkpoint.sha256,
                runtime_binding_sha256=runtime_hash,
            )
            intent = store.read_intent(submission_key)
            if intent is None or not _intent_matches(
                intent,
                profile=profile,
                manifest=manifest,
                dataset=dataset,
                environment=environment,
                operation="resume",
                attempt=attempt,
                checkpoint_receipt_sha256=checkpoint.sha256,
            ):
                raise MsctlError(
                    "STATE_INCOMPLETE",
                    "resume state lacks its matching durable pair intent",
                )
            recovered = _reconcile_pair_intent(
                store=store,
                intent=intent,
                submission_key=submission_key,
                job_name=job_name,
                environ=environ,
            )
            if recovered is None:
                command = render_seed0_command(
                    profile,
                    release,
                    manifest,
                    release_root=release_root,
                    dataset=dataset,
                    environment=environment,
                    checkpoint_receipt=checkpoint,
                    attempt=attempt,
                )
                intent["phase"] = "SUBMITTING"
                intent["updated_at"] = _timestamp()
                store.write_intent(submission_key, intent)
                job_id = slurm_submit(command, environ=environ)
                status = "SUBMITTED"
                intent["phase"] = "SUBMITTED"
                intent["job_id"] = job_id
                intent["updated_at"] = _timestamp()
                store.write_intent(submission_key, intent)
                submitted = 1
                idempotent = False
            else:
                job_id, status = recovered
                submitted = 0
                idempotent = True
            now = _timestamp()
            for run, state in zip(manifest.runs, states):
                state["job_id"] = job_id
                state["status"] = status
                state["updated_at"] = now
                store.write_run(run.run_id, state)
            return {
                "job_id": job_id,
                "status": status,
                "attempt": attempt,
                "submitted": submitted,
                "idempotent": idempotent,
                "run_manifest_sha256": manifest.sha256,
            }
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
        if previous_status not in RESUMABLE_TERMINAL_STATES:
            raise MsctlError(
                "RESUME_STATE_UNCERTAIN",
                "Slurm did not report an explicit resumable terminal state",
                details={
                    "job_id": previous_id,
                    "status": previous_status,
                },
            )
        attempt = max(int(state.get("attempt", 1)) for state in states) + 1
        submission_key, job_name, _ = submission_identity(
            profile,
            release,
            manifest,
            operation="resume",
            attempt=attempt,
            checkpoint_receipt_sha256=checkpoint.sha256,
            runtime_binding_sha256=runtime_hash,
        )
        command = render_seed0_command(
            profile,
            release,
            manifest,
            release_root=release_root,
            dataset=dataset,
            environment=environment,
            checkpoint_receipt=checkpoint,
            attempt=attempt,
        )
        intent = store.read_intent(submission_key)
        if intent is None:
            intent = _new_pair_intent(
                profile=profile,
                manifest=manifest,
                submission_key=submission_key,
                dataset=dataset,
                environment=environment,
                resources=resources,
                operation="resume",
                attempt=attempt,
                checkpoint_receipt_sha256=checkpoint.sha256,
            )
            store.write_intent(submission_key, intent)
        elif not _intent_matches(
            intent,
            profile=profile,
            manifest=manifest,
            dataset=dataset,
            environment=environment,
            operation="resume",
            attempt=attempt,
            checkpoint_receipt_sha256=checkpoint.sha256,
        ):
            raise MsctlError(
                "RUN_ID_CONFLICT",
                "resume intent key already exists with incompatible state",
            )
        intent_time = _timestamp()
        for run, state in zip(manifest.runs, states):
            if (
                state["operation"] == "resume"
                and state["submission_key"] == submission_key
            ):
                continue
            state.update(
                {
                    "job_id": None,
                    "status": "RESUBMITTING",
                    "attempt": attempt,
                    "operation": "resume",
                    "submission_key": submission_key,
                    "resource_request": resources,
                    "checkpoint_receipt_sha256": checkpoint.sha256,
                    "dataset_verification_sha256": dataset[
                        "verification_sha256"
                    ],
                    "prior_job_ids": [previous_id],
                    "updated_at": intent_time,
                }
            )
            store.write_run(run.run_id, state)
        recovered = _reconcile_pair_intent(
            store=store,
            intent=intent,
            submission_key=submission_key,
            job_name=job_name,
            environ=environ,
        )
        if recovered is not None:
            job_id, status = recovered
            reconciled_at = _timestamp()
            for run, state in zip(manifest.runs, states):
                state["job_id"] = job_id
                state["status"] = status
                state["updated_at"] = reconciled_at
                store.write_run(run.run_id, state)
            return {
                "job_id": job_id,
                "status": status,
                "attempt": attempt,
                "submitted": 0,
                "idempotent": True,
                "run_manifest_sha256": manifest.sha256,
            }
        intent["phase"] = "SUBMITTING"
        intent["updated_at"] = _timestamp()
        store.write_intent(submission_key, intent)
        job_id = slurm_submit(command, environ=environ)
        submitted_at = _timestamp()
        intent["phase"] = "SUBMITTED"
        intent["job_id"] = job_id
        intent["updated_at"] = submitted_at
        store.write_intent(submission_key, intent)
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
            "idempotent": False,
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
    release, manifest, _ = load_bound_inputs(
        profile=profile,
        release_path=release_path,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    if not apply:
        return {
            "operation": "cancel",
            "run_ids": [run.run_id for run in manifest.runs],
            "resource_request": resource_request(profile, "cancel"),
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
    dataset_pointer: Path | str,
    dataset_root: Path | str | None,
    dataset_verification: Path | str | None,
    environment_receipt: Path | str | None,
    repo_root: Path | str,
    state_root: Path | str,
    approval_path: Path | str | None,
    apply: bool,
    environ: dict[str, str] | None = None,
) -> dict[str, object]:
    release, manifest, release_root = load_bound_inputs(
        profile=profile,
        release_path=release_path,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    dataset = _resolve_dataset_verification(
        profile=profile,
        dataset_pointer=dataset_pointer,
        dataset_root=dataset_root,
        dataset_verification=dataset_verification,
        release=release,
        manifest=manifest,
        repo_root=repo_root,
    )
    environment = verify_environment_receipt(
        environment_receipt,
        profile=profile,
        release=release,
    )
    command = render_evaluate_command(
        profile,
        release,
        manifest,
        release_root=release_root,
        dataset=dataset,
        environment=environment,
    )
    resources = resource_request(profile, "evaluate")
    runtime_hash = runtime_binding_sha256(
        release_root=release_root,
        dataset=dataset,
        environment=environment,
    )
    submission_key, job_name, _ = submission_identity(
        profile,
        release,
        manifest,
        operation="evaluate",
        runtime_binding_sha256=runtime_hash,
    )
    if not apply:
        return {
            "command": command,
            "submitted": 0,
            "run_manifest_sha256": manifest.sha256,
            "dataset_verification": dataset,
            "resource_request": resources,
            "submission_key": submission_key,
        }
    _preflight_entrypoint(
        release=release,
        release_root=release_root,
        member=EVALUATOR_ENTRYPOINT,
        marker_name="MSCTL_EVALUATOR_CONTRACT",
        marker_value="memorysplit-confirmatory-evaluator-v1",
        required_interface={"evaluate", "--run", "--sealed-release", "--device"},
    )
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
                or existing.get("dataset_sha256") != manifest.dataset_sha256
                or existing.get("dataset_verification_sha256")
                != dataset["verification_sha256"]
                or existing.get("environment_receipt_sha256")
                != environment["receipt_sha256"]
                or existing.get("submission_key") != submission_key
            ):
                raise MsctlError(
                    "RUN_ID_CONFLICT",
                    "evaluation state has different provenance",
                )
            if existing.get("job_id") is None:
                discovered = discover_jobs(
                    submission_key,
                    job_name,
                    environ=environ,
                )
                if not discovered:
                    raise MsctlError(
                        "SUBMISSION_UNCERTAIN",
                        "Slurm cannot prove whether the evaluation intent submitted",
                        details={
                            "recoverable": True,
                            "submission_key": submission_key,
                        },
                    )
                if len(discovered) != 1:
                    raise MsctlError(
                        "SUBMISSION_MULTIPLE_MATCHES",
                        "multiple Slurm jobs match one evaluation intent",
                        details={
                            "job_ids": sorted(discovered, key=int),
                        },
                    )
                job_id, status = next(iter(discovered.items()))
            else:
                job_id = str(existing["job_id"])
                status = query_states([job_id], environ=environ)[job_id]
            existing["job_id"] = job_id
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
                "dataset_sha256": manifest.dataset_sha256,
                "dataset_verification_sha256": dataset[
                    "verification_sha256"
                ],
                "environment_receipt_sha256": environment["receipt_sha256"],
                "operation": "evaluate",
                "submission_key": submission_key,
                "resource_request": resources,
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
