"""Closed Illumina Slurm command rendering and bounded execution."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence

from .contracts import Release, RunManifest
from .errors import MsctlError
from .profile import IlluminaProfile


SEED0_SCRIPT = "cluster/slurm/v2_seed0.sbatch"
EVALUATE_SCRIPT = "cluster/slurm/v2_evaluate.sbatch"
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
_SAFE_CHILD_ENV = {
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "TMPDIR",
    "USER",
    "MS_SHARED_ROOT",
    "MS_ENV_ROOT",
    "MS_DATA_ROOT",
    "MS_OUT_ROOT",
    "MS_SEALED_ROOT",
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


def _forward_exports(
    profile: IlluminaProfile,
    names: Sequence[str],
) -> list[str]:
    for name in names:
        if name not in profile.job_env_allowlist:
            raise MsctlError(
                "ENV_NOT_ALLOWED",
                "runtime root is not allowlisted for Slurm",
                details={"name": name},
            )
    return list(names)


def _resource_args(profile: IlluminaProfile) -> list[str]:
    result: list[str] = []
    if profile.partition is not None:
        result.append(f"--partition={profile.partition}")
    if profile.account is not None:
        result.append(f"--account={profile.account}")
    if profile.qos is not None:
        result.append(f"--qos={profile.qos}")
    return result


def render_seed0_command(
    profile: IlluminaProfile,
    release: Release,
    manifest: RunManifest,
    *,
    resume: bool = False,
) -> list[str]:
    by_arm = {run.arm: run for run in manifest.runs}
    exports = _forward_exports(
        profile,
        ("MS_SHARED_ROOT", "MS_ENV_ROOT", "MS_DATA_ROOT", "MS_OUT_ROOT"),
    ) + [
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
        _safe_export("MS_DENSE_CONFIG", by_arm["dense"].config, profile),
        _safe_export(
            "MS_DENSE_CONFIG_SHA256",
            by_arm["dense"].config_sha256,
            profile,
        ),
        _safe_export("MS_SPLIT_RUN_ID", by_arm["split90"].run_id, profile),
        _safe_export("MS_SPLIT_CONFIG", by_arm["split90"].config, profile),
        _safe_export(
            "MS_SPLIT_CONFIG_SHA256",
            by_arm["split90"].config_sha256,
            profile,
        ),
        _safe_export("MS_WORLD_SIZE", 3, profile),
        _safe_export("MS_EVAL_GPU", 6, profile),
        _safe_export("MS_RESUME", int(resume), profile),
    ]
    return [
        "sbatch",
        "--parsable",
        *_resource_args(profile),
        f"--gres={profile.gres}:{profile.allocated_gpus}",
        f"--time={_wall_time(profile.seed0_wall_minutes)}",
        "--job-name=ms-v2-seed0",
        f"--export=NONE,{','.join(exports)}",
        SEED0_SCRIPT,
    ]


def render_evaluate_command(
    profile: IlluminaProfile,
    release: Release,
    manifest: RunManifest,
) -> list[str]:
    exports = _forward_exports(
        profile,
        (
            "MS_SHARED_ROOT",
            "MS_ENV_ROOT",
            "MS_DATA_ROOT",
            "MS_OUT_ROOT",
            "MS_SEALED_ROOT",
        ),
    ) + [
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
        f"--gres={profile.gres}:{profile.evaluation_gpus}",
        f"--time={_wall_time(profile.evaluation_wall_minutes)}",
        "--job-name=ms-v2-evaluate",
        f"--export=NONE,{','.join(exports)}",
        EVALUATE_SCRIPT,
    ]


def redact(value: str, *, secrets: Sequence[str] = ()) -> str:
    result = value[:MAX_CAPTURE]
    for secret in secrets:
        if secret:
            result = result.replace(secret, "[REDACTED]")
    for pattern in _REDACT_PATTERNS:
        result = pattern.sub(lambda match: f"{match.group(1)}=[REDACTED]", result)
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
