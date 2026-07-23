"""Closed Illumina Slurm command rendering and bounded execution."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from .bootstrap import BOOTSTRAP_PAYLOAD, BOOTSTRAP_SHA256
from .contracts import CheckpointReceipt, Release, RunManifest
from .errors import MsctlError
from .jsonutil import canonical_sha256
from .profile import IlluminaProfile


SEED0_SCRIPT = "cluster/slurm/v2_seed0.sbatch"
EVALUATE_SCRIPT = "cluster/slurm/v2_evaluate.sbatch"
TRAIN_ENTRYPOINT = "scripts/run_train.py"
EVALUATOR_ENTRYPOINT = "evals/confirmatory/runner.py"
MAX_CAPTURE = 16_384
JOB_ID_RE = re.compile(r"^(?P<job>[0-9]+)(?:;(?P<cluster>[A-Za-z0-9._-]+))?$")
ACTIVE_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "RUNNING",
    "SUSPENDED",
}
RESUMABLE_TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
_SAFE_CHILD_ENV = {
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "TMPDIR",
    "USER",
}
_REDACT_PATTERNS = (
    re.compile(r"(?i)(token|secret|password|key)=([^,\s]+)"),
    re.compile(r"\b(?:hf|ghp|github_pat)_[A-Za-z0-9_-]{8,}\b"),
)


def _wall_time(minutes: int) -> str:
    hours, minute = divmod(minutes, 60)
    days, hour = divmod(hours, 24)
    clock = f"{hour:02d}:{minute:02d}:00"
    return f"{days}-{clock}" if days else clock


def _safe_export(name: str, value: object, profile: IlluminaProfile) -> str:
    if name not in profile.job_env_allowlist:
        raise MsctlError(
            "ENV_NOT_ALLOWED",
            "job environment key is not allowlisted",
            details={"name": name},
        )
    text = str(value)
    if (
        not text
        or any(character in text for character in ",=\n\r\0")
        or len(text) > 4096
    ):
        raise MsctlError(
            "ENV_VALUE_INVALID",
            "job environment value is unsafe",
            details={"name": name},
        )
    return f"{name}={text}"


def _resource_args(profile: IlluminaProfile) -> list[str]:
    result: list[str] = []
    if profile.partition is not None:
        result.append(f"--partition={profile.partition}")
    if profile.account is not None:
        result.append(f"--account={profile.account}")
    if profile.qos is not None:
        result.append(f"--qos={profile.qos}")
    return result


def _job_shape_args(*, operation: str, shared_root: str) -> list[str]:
    if operation in {"submit", "resume"}:
        cpus = 32
        memory = "0"
        output = "ms-v2-seed0-%j.out"
    elif operation == "evaluate":
        cpus = 8
        memory = "96G"
        output = "ms-v2-evaluate-%j.out"
    else:  # pragma: no cover - callers are closed over known operations.
        raise MsctlError(
            "RESOURCE_REQUEST_INVALID",
            "unsupported Slurm job shape",
        )
    return [
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cpus}",
        f"--mem={memory}",
        "--open-mode=append",
        f"--output={Path(shared_root) / output}",
        "--chdir=/",
    ]


def resource_request(
    profile: IlluminaProfile,
    operation: str,
) -> dict[str, object]:
    if operation in {"submit", "resume"}:
        gpus = profile.allocated_gpus
        wall_minutes = profile.seed0_wall_minutes
        script: str | None = SEED0_SCRIPT
        gres: str | None = f"{profile.gres}:{gpus}"
        jobs = 1
    elif operation == "evaluate":
        gpus = profile.evaluation_gpus
        wall_minutes = profile.evaluation_wall_minutes
        script = EVALUATE_SCRIPT
        gres = f"{profile.gres}:{gpus}"
        jobs = 1
    elif operation in {"cancel", "cleanup"}:
        gpus = 0
        wall_minutes = 0
        script = None
        gres = None
        jobs = 0
    else:
        raise MsctlError(
            "RESOURCE_REQUEST_INVALID",
            "unsupported operation for resource accounting",
            details={"operation": operation},
        )
    return {
        "schema_version": 1,
        "operation": operation,
        "jobs": jobs,
        "allocated_gpus": gpus,
        "wall_minutes": wall_minutes,
        "gpu_hours": gpus * wall_minutes / 60.0,
        "gres": gres,
        "script": script,
    }


def submission_identity(
    profile: IlluminaProfile,
    release: Release,
    manifest: RunManifest,
    *,
    operation: str,
    attempt: int = 1,
    checkpoint_receipt_sha256: str | None = None,
    runtime_binding_sha256: str | None = None,
) -> tuple[str, str, str]:
    value = {
        "schema_version": 1,
        "bootstrap_sha256": BOOTSTRAP_SHA256,
        "provider": profile.provider,
        "profile_sha256": profile.sha256,
        "release_sha256": release.archive_sha256,
        "run_manifest_sha256": manifest.sha256,
        "operation": operation,
        "attempt": attempt,
        "checkpoint_receipt_sha256": checkpoint_receipt_sha256,
        "runtime_binding_sha256": runtime_binding_sha256,
        "resource_request": resource_request(profile, operation),
    }
    key = canonical_sha256(value)
    return key, f"ms-v2-{operation}-{key[:12]}", f"msctl:{key}"


def runtime_binding_sha256(
    *,
    release_root: Path,
    dataset: Mapping[str, object],
    environment: Mapping[str, object],
) -> str:
    return canonical_sha256(
        {
            "schema_version": 1,
            "release_root": str(release_root),
            "dataset_root": dataset["dataset_root"],
            "publication_root": dataset["publication_root"],
            "dataset_verification_sha256": dataset["verification_sha256"],
            "environment_root": environment["root"],
            "environment_receipt_sha256": environment["receipt_sha256"],
        }
    )


def render_seed0_command(
    profile: IlluminaProfile,
    release: Release,
    manifest: RunManifest,
    *,
    release_root: Path,
    dataset: Mapping[str, object],
    environment: Mapping[str, object],
    checkpoint_receipt: CheckpointReceipt | None = None,
    attempt: int = 1,
) -> list[str]:
    operation = "resume" if checkpoint_receipt is not None else "submit"
    by_arm = {run.arm: run for run in manifest.runs}
    runtime_hash = runtime_binding_sha256(
        release_root=release_root,
        dataset=dataset,
        environment=environment,
    )
    key, job_name, comment = submission_identity(
        profile,
        release,
        manifest,
        operation=operation,
        attempt=attempt,
        checkpoint_receipt_sha256=(
            checkpoint_receipt.sha256
            if checkpoint_receipt is not None
            else None
        ),
        runtime_binding_sha256=runtime_hash,
    )
    dataset_root = str(dataset["dataset_root"])
    publication_root = str(dataset["publication_root"])
    environment_root = str(environment["root"])
    output_root = str(Path(publication_root) / "memorysplit" / "runs")
    dataset_relative = (
        Path(dataset_root).relative_to(Path(publication_root)).as_posix()
    )
    exports = [
        _safe_export("MS_SHARED_ROOT", publication_root, profile),
        _safe_export(
            "MS_SHARED_ROOT_PREFIX",
            profile.shared_root_prefix,
            profile,
        ),
        _safe_export("MS_ENV_ROOT", environment_root, profile),
        _safe_export("MS_DATA_ROOT", dataset_root, profile),
        _safe_export("MS_DATA_RELATIVE_PATH", dataset_relative, profile),
        _safe_export("MS_OUT_ROOT", output_root, profile),
        _safe_export(
            "MS_RELEASE_ARCHIVE",
            release.archive_path,
            profile,
        ),
        _safe_export(
            "MS_RELEASE_ARCHIVE_SHA256",
            release.archive_sha256,
            profile,
        ),
        _safe_export(
            "MS_RELEASE_ARCHIVE_BYTES",
            release.archive_bytes,
            profile,
        ),
        _safe_export(
            "MS_RELEASE_MEMBERS_SHA256",
            release.members_sha256,
            profile,
        ),
        _safe_export("MS_JOB_SCRIPT_REL", SEED0_SCRIPT, profile),
        _safe_export(
            "MS_TRAIN_ENTRYPOINT_REL",
            TRAIN_ENTRYPOINT,
            profile,
        ),
        _safe_export(
            "MS_TRAIN_ENTRYPOINT_SHA256",
            release.members[TRAIN_ENTRYPOINT]["sha256"],
            profile,
        ),
        _safe_export("MS_PROVIDER", profile.provider, profile),
        _safe_export("MS_PROFILE_SHA256", profile.sha256, profile),
        _safe_export("MS_RELEASE_ID", release.release_id, profile),
        _safe_export(
            "MS_RELEASE_SHA256", release.archive_sha256, profile
        ),
        _safe_export(
            "MS_RUN_MANIFEST_SHA256", manifest.sha256, profile
        ),
        _safe_export("MS_DATASET_SHA256", manifest.dataset_sha256, profile),
        _safe_export("MS_DENSE_RUN_ID", by_arm["dense"].run_id, profile),
        _safe_export(
            "MS_DENSE_CONFIG_REL",
            by_arm["dense"].config,
            profile,
        ),
        _safe_export(
            "MS_DENSE_CONFIG_SHA256",
            by_arm["dense"].config_sha256,
            profile,
        ),
        _safe_export("MS_SPLIT_RUN_ID", by_arm["split90"].run_id, profile),
        _safe_export(
            "MS_SPLIT_CONFIG_REL",
            by_arm["split90"].config,
            profile,
        ),
        _safe_export(
            "MS_SPLIT_CONFIG_SHA256",
            by_arm["split90"].config_sha256,
            profile,
        ),
        _safe_export("MS_WORLD_SIZE", 3, profile),
        _safe_export("MS_EVAL_GPU", 6, profile),
        _safe_export("MS_RESUME", int(checkpoint_receipt is not None), profile),
    ]
    if checkpoint_receipt is not None:
        by_checkpoint_arm = {
            checkpoint.arm: checkpoint
            for checkpoint in checkpoint_receipt.checkpoints
        }
        for prefix, arm in (("DENSE", "dense"), ("SPLIT", "split90")):
            checkpoint = by_checkpoint_arm[arm]
            exports.extend(
                [
                    _safe_export(
                        f"MS_{prefix}_RESUME_PATH",
                        checkpoint.path,
                        profile,
                    ),
                    _safe_export(
                        f"MS_{prefix}_RESUME_SHA256",
                        checkpoint.sha256,
                        profile,
                    ),
                ]
            )
    return [
        "sbatch",
        "--parsable",
        *_resource_args(profile),
        *_job_shape_args(
            operation=operation,
            shared_root=publication_root,
        ),
        f"--gres={profile.gres}:{profile.allocated_gpus}",
        f"--time={_wall_time(profile.seed0_wall_minutes)}",
        f"--job-name={job_name}",
        f"--comment={comment}",
        f"--export={','.join(exports)}",
    ]


def render_evaluate_command(
    profile: IlluminaProfile,
    release: Release,
    manifest: RunManifest,
    *,
    release_root: Path,
    dataset: Mapping[str, object],
    environment: Mapping[str, object],
) -> list[str]:
    _, job_name, comment = submission_identity(
        profile,
        release,
        manifest,
        operation="evaluate",
        runtime_binding_sha256=runtime_binding_sha256(
            release_root=release_root,
            dataset=dataset,
            environment=environment,
        ),
    )
    dataset_root = str(dataset["dataset_root"])
    publication_root = str(dataset["publication_root"])
    environment_root = str(environment["root"])
    output_root = str(Path(publication_root) / "memorysplit" / "runs")
    sealed_root = str(Path(publication_root) / "memorysplit" / "sealed-eval")
    dataset_relative = (
        Path(dataset_root).relative_to(Path(publication_root)).as_posix()
    )
    exports = [
        _safe_export("MS_SHARED_ROOT", publication_root, profile),
        _safe_export(
            "MS_SHARED_ROOT_PREFIX",
            profile.shared_root_prefix,
            profile,
        ),
        _safe_export("MS_ENV_ROOT", environment_root, profile),
        _safe_export("MS_DATA_ROOT", dataset_root, profile),
        _safe_export("MS_DATA_RELATIVE_PATH", dataset_relative, profile),
        _safe_export("MS_OUT_ROOT", output_root, profile),
        _safe_export("MS_SEALED_ROOT", sealed_root, profile),
        _safe_export(
            "MS_RELEASE_ARCHIVE",
            release.archive_path,
            profile,
        ),
        _safe_export(
            "MS_RELEASE_ARCHIVE_SHA256",
            release.archive_sha256,
            profile,
        ),
        _safe_export(
            "MS_RELEASE_ARCHIVE_BYTES",
            release.archive_bytes,
            profile,
        ),
        _safe_export(
            "MS_RELEASE_MEMBERS_SHA256",
            release.members_sha256,
            profile,
        ),
        _safe_export("MS_JOB_SCRIPT_REL", EVALUATE_SCRIPT, profile),
        _safe_export(
            "MS_EVALUATOR_ENTRYPOINT_REL",
            EVALUATOR_ENTRYPOINT,
            profile,
        ),
        _safe_export(
            "MS_EVALUATOR_ENTRYPOINT_SHA256",
            release.members[EVALUATOR_ENTRYPOINT]["sha256"],
            profile,
        ),
        _safe_export("MS_PROVIDER", profile.provider, profile),
        _safe_export("MS_PROFILE_SHA256", profile.sha256, profile),
        _safe_export(
            "MS_RELEASE_SHA256", release.archive_sha256, profile
        ),
        _safe_export(
            "MS_RUN_MANIFEST_SHA256", manifest.sha256, profile
        ),
        _safe_export("MS_DATASET_SHA256", manifest.dataset_sha256, profile),
        _safe_export(
            "MS_RUN_IDS",
            ":".join(run.run_id for run in manifest.runs),
            profile,
        ),
    ]
    return [
        "sbatch",
        "--parsable",
        *_resource_args(profile),
        *_job_shape_args(
            operation="evaluate",
            shared_root=publication_root,
        ),
        f"--gres={profile.gres}:{profile.evaluation_gpus}",
        f"--time={_wall_time(profile.evaluation_wall_minutes)}",
        f"--job-name={job_name}",
        f"--comment={comment}",
        f"--export={','.join(exports)}",
    ]


def redact(value: str, *, secrets: Sequence[str] = ()) -> str:
    result = value[:MAX_CAPTURE]
    for secret in secrets:
        if secret:
            result = result.replace(secret, "[REDACTED]")
    for pattern in _REDACT_PATTERNS:
        if pattern.groups:
            result = pattern.sub(
                lambda match: f"{match.group(1)}=[REDACTED]",
                result,
            )
        else:
            result = pattern.sub("[REDACTED]", result)
    return result


def _child_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    result = {
        key: value for key, value in source.items() if key in _SAFE_CHILD_ENV
    }
    supplied_path = result.get("PATH", "")
    path_parts = [part for part in supplied_path.split(os.pathsep) if part]
    for part in os.defpath.split(os.pathsep):
        if part not in path_parts:
            path_parts.append(part)
    result["PATH"] = os.pathsep.join(path_parts)
    result.setdefault("LANG", "C.UTF-8")
    return result


def require_tools(
    tools: Sequence[str],
    *,
    operation: str,
    environ: Mapping[str, str] | None = None,
) -> None:
    path = (os.environ if environ is None else environ).get("PATH")
    missing = [tool for tool in tools if shutil.which(tool, path=path) is None]
    if missing:
        raise MsctlError(
            "EXTERNAL_UNAVAILABLE",
            "required external operation is unavailable",
            details={"operation": operation, "missing": missing},
        )


def run_command(
    command: Sequence[str],
    *,
    operation: str,
    environ: Mapping[str, str] | None = None,
    timeout: int = 30,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    require_tools([command[0]], operation=operation, environ=environ)
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=_child_environment(environ),
            input=input_text,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise MsctlError(
            "EXTERNAL_FAILED",
            "external operation did not complete safely",
            details={"operation": operation},
        ) from error
    if completed.returncode != 0:
        raise MsctlError(
            "EXTERNAL_FAILED",
            "external operation returned a failure",
            details={
                "operation": operation,
                "returncode": completed.returncode,
                "stdout": redact(completed.stdout),
                "stderr": redact(completed.stderr),
            },
        )
    return completed


def submit(
    command: Sequence[str],
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    completed = run_command(
        command,
        operation="sbatch submission",
        environ=environ,
        input_text=BOOTSTRAP_PAYLOAD.decode("ascii"),
    )
    output = completed.stdout.strip()
    match = JOB_ID_RE.fullmatch(output)
    if match is None:
        raise MsctlError(
            "SBATCH_RESPONSE_INVALID",
            "sbatch --parsable returned an invalid job ID",
            details={"stdout": redact(completed.stdout)},
        )
    return match.group("job")


def query_states(
    job_ids: Sequence[str],
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if any(
        not isinstance(job_id, str) or not job_id.isdigit()
        for job_id in job_ids
    ):
        raise MsctlError(
            "STATUS_RESPONSE_INVALID",
            "recorded Slurm job IDs must be decimal integers",
        )
    unique = sorted(set(job_ids), key=int)
    if not unique:
        return {}
    require_tools(
        ["squeue", "sacct"],
        operation="status reconciliation",
        environ=environ,
    )
    joined = ",".join(unique)
    queue = run_command(
        ["squeue", "-h", "-j", joined, "-o", "%i|%T"],
        operation="status reconciliation",
        environ=environ,
    )
    states: dict[str, str] = {}
    for line in queue.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.strip().split("|")
        if len(parts) != 2 or parts[0] not in unique:
            raise MsctlError(
                "STATUS_RESPONSE_INVALID",
                "squeue returned an unexpected row",
            )
        states[parts[0]] = parts[1].upper()
    missing = [job_id for job_id in unique if job_id not in states]
    if missing:
        accounting = run_command(
            [
                "sacct",
                "-n",
                "-P",
                "-j",
                ",".join(missing),
                "--format=JobIDRaw,State",
            ],
            operation="status reconciliation",
            environ=environ,
        )
        for line in accounting.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.strip().split("|")
            if len(parts) < 2:
                raise MsctlError(
                    "STATUS_RESPONSE_INVALID",
                    "sacct returned an unexpected row",
                )
            job_id = parts[0].split(".", 1)[0]
            if job_id in missing and job_id not in states:
                states[job_id] = parts[1].split()[0].upper()
    if any(job_id not in states for job_id in unique):
        raise MsctlError(
            "STATUS_UNKNOWN",
            "Slurm could not authoritatively reconcile every job",
            details={"job_ids": unique},
        )
    return states


def discover_jobs(
    submission_key: str,
    job_name: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Find existing Slurm jobs by the exact deterministic msctl comment."""

    require_tools(
        ["squeue", "sacct"],
        operation="submission recovery",
        environ=environ,
    )
    source = os.environ if environ is None else environ
    user = source.get("USER", "")
    comment = f"msctl:{submission_key}"
    queue = run_command(
        ["squeue", "-h", "-u", user, "-o", "%i|%k|%T"],
        operation="submission recovery",
        environ=environ,
    )
    jobs: dict[str, str] = {}

    def consume(text: str, *, source_name: str) -> None:
        for raw_line in text.splitlines():
            if not raw_line.strip():
                continue
            parts = raw_line.strip().split("|")
            if len(parts) < 3:
                raise MsctlError(
                    "RECOVERY_RESPONSE_INVALID",
                    f"{source_name} returned a malformed recovery row",
                )
            job_id, row_comment, state = parts[:3]
            if row_comment != comment:
                continue
            if not job_id.isdigit() or not state:
                raise MsctlError(
                    "RECOVERY_RESPONSE_INVALID",
                    f"{source_name} returned an invalid matching row",
                )
            jobs[job_id] = state.split()[0].upper()

    consume(queue.stdout, source_name="squeue")
    accounting = run_command(
        [
            "sacct",
            "-n",
            "-P",
            "-X",
            "--name",
            job_name,
            "--format=JobIDRaw,Comment,State",
        ],
        operation="submission recovery",
        environ=environ,
    )
    consume(accounting.stdout, source_name="sacct")
    return jobs


def capacity_check(
    profile: IlluminaProfile,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    require_tools(
        ["sinfo", "squeue", "sacct"],
        operation="capacity check",
        environ=environ,
    )
    partitions = run_command(
        ["sinfo", "-h", "-o", "%P|%G|%D|%c"],
        operation="capacity check",
        environ=environ,
    )
    queue = run_command(
        ["squeue", "-h", "-u", (environ or os.environ).get("USER", ""), "-o", "%i|%T|%b"],
        operation="capacity check",
        environ=environ,
    )
    cpu_nodes = 0
    a100_gpus = 0
    for raw_line in partitions.stdout.splitlines():
        if not raw_line.strip():
            continue
        fields = raw_line.strip().split("|")
        if len(fields) != 4:
            raise MsctlError(
                "CAPACITY_RESPONSE_INVALID",
                "sinfo returned a malformed capacity row",
            )
        _, gres, raw_nodes, raw_cpus = fields
        try:
            nodes = int(raw_nodes)
            cpus = int(raw_cpus)
        except ValueError as error:
            raise MsctlError(
                "CAPACITY_RESPONSE_INVALID",
                "sinfo returned non-integer node capacity",
            ) from error
        if nodes < 0 or cpus < 0:
            raise MsctlError(
                "CAPACITY_RESPONSE_INVALID",
                "sinfo returned negative capacity",
            )
        matches = re.findall(
            r"(?:^|,)gpu:a100:(\d+)(?:\([^)]*\))?(?:,|$)",
            gres,
            flags=re.IGNORECASE,
        )
        if matches:
            a100_gpus += nodes * sum(int(count) for count in matches)
        elif gres in {"(null)", "N/A", "none"} and cpus >= 56:
            cpu_nodes += nodes
    observed = {
        "cpu_nodes": cpu_nodes,
        "cpus_per_node_floor": 56 if cpu_nodes else 0,
        "a100_gpus": a100_gpus,
    }
    if cpu_nodes < 35 or a100_gpus < profile.allocated_gpus:
        raise MsctlError(
            "CAPACITY_INSUFFICIENT",
            "observed Slurm capacity is below the frozen Illumina floor",
            details={
                "observed": observed,
                "required": {
                    "cpu_nodes": 35,
                    "cpus_per_node": 56,
                    "a100_gpus": profile.allocated_gpus,
                },
            },
        )
    return {
        "profile_id": profile.profile_id,
        "expected": {
            "cpu_nodes": 35,
            "cpus_per_node": 56,
            "gpu_model": "NVIDIA A100 80GB",
            "allocated_gpus": 7,
        },
        "observed": observed,
        "sufficient": True,
        "sinfo": redact(partitions.stdout),
        "squeue": redact(queue.stdout),
    }
