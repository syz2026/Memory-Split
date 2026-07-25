"""Strict, dry-run-first backend shared by closed AWS GPU profiles."""

from __future__ import annotations

import base64
import configparser
import hashlib
import importlib
import json
import os
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from cluster.aws.p5.terminal_artifacts import (
    EVALUATION_ARTIFACTS,
    verify_checkpoint_receipt_bytes,
    verify_evaluation_receipt_bytes,
    verify_materialized_terminal,
)
from evals.confirmatory.sealing import (
    SEALED_FIXTURE_MEMBERS,
    sealed_fixture_sha256,
)

from . import aws_identity
from .approval import verify_scope_approval
from .aws_control_bundle import (
    ControlBundle,
    build_control_bundle_bytes,
    render_control_install_command,
    verify_control_bundle,
)
from .aws_fleet import (
    FleetManifestBinding,
    FleetPlan,
    create_fleet_advance,
    fleet_transition_for_target,
    load_fleet_advance,
    load_fleet_plan,
    validate_fleet_manifest,
    write_fleet_advance,
)
from .aws_identity import (
    AWS_INSTANCE_IDENTITY_CERTIFICATES,
    AwsIdentityError,
    verify_instance_identity_pkcs7,
)
from .aws_argv import (
    ARGV_DOCUMENT_CONTENT,
    ARGV_DOCUMENT_NAME,
    ARGV_DOCUMENT_SHA256,
)
from .aws_canary import (
    ORCHESTRATION_PLAN_TYPE,
    build_canary_intent,
    canary_plan_uri,
    load_canary_plan,
    plan_canary,
)
from .aws_selection import (
    HardwareAmendment,
    ProviderSelection,
    load_hardware_amendment,
    load_provider_selection,
)
from .aws_readiness import (
    DIAGNOSTIC_IDS,
    LaunchReadiness,
    load_launch_readiness,
)
from .aws_sealed_evaluation import (
    REQUIRED_SEALED_MEMBERS,
    SealedEvaluationFixture,
    SealedEvaluationRelease,
    load_sealed_evaluation_fixture,
    load_sealed_evaluation_release,
)
from .contracts import (
    bind_release,
    load_release,
    load_run_manifest,
    validate_runtime_attested_contract,
    verify_checkpoint_receipt,
    verify_release_member,
)
from .errors import MsctlError
from .fsutil import atomic_write_at, open_directory, rename_noreplace_at
from .jsonutil import (
    canonical_json,
    canonical_sha256,
    load_json,
    require_exact_keys,
    require_object,
    require_sha256,
    sha256_file,
)
from .profile import (
    AWS_GPU_PROFILES,
    AWS_P5_PROFILE,
    AWS_P5_V3_PROFILE,
    AWS_P6_B300_V3_PROFILE,
)
from .state import StateStore


INSTANCE_TYPE = "p5.48xlarge"
_PROFILE_INSTANCE_TYPES = {
    "aws-p5.48xlarge": "p5.48xlarge",
    "aws-p5.48xlarge-v3": "p5.48xlarge",
    "aws-p6-b300.48xlarge-v3": "p6-b300.48xlarge",
}
_PROFILE_GRES = {
    "aws-p5.48xlarge": "gpu:h100:8",
    "aws-p5.48xlarge-v3": "gpu:h100:8",
    "aws-p6-b300.48xlarge-v3": "gpu:b300:8",
}
_PROFILE_SEEDS = {
    "aws-p5.48xlarge": (1, 2, 3, 4),
    "aws-p5.48xlarge-v3": tuple(range(10)),
    "aws-p6-b300.48xlarge-v3": tuple(range(10)),
}
_PROFILE_PURCHASE_MODELS = {
    "aws-p5.48xlarge": "on_demand",
    "aws-p5.48xlarge-v3": "on_demand",
    "aws-p6-b300.48xlarge-v3": "capacity_block",
}
_V3_PROFILES = frozenset({AWS_P5_V3_PROFILE, AWS_P6_B300_V3_PROFILE})
_V3_PROVENANCE_FIELDS = (
    "cohort_assignment_sha256",
    "preregistration_sha256",
    "hardware_amendment_sha256",
    "provider_selection_sha256",
    "profile_sha256",
    "sealed_fixture_sha256",
)
_INSTANCE_ID_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
_COMMAND_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{7,127}$")
_REGION_RE = re.compile(r"^[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+$")
_AMI_RE = re.compile(r"^ami-[0-9a-f]{8,17}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_BUCKET_RE = re.compile(
    r"^(?![0-9]+(?:\.[0-9]+){3}$)(?!-)(?!.*\.\.)(?!.*\.-)(?!.*-\.)"
    r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$"
)
_INSTANCE_PROFILE_RE = re.compile(
    r"^arn:aws(?:-[a-z]+)?:iam::[0-9]{12}:instance-profile/"
    r"[A-Za-z0-9+=,.@_/-]{1,128}$"
)
_ACTIVE_INSTANCE_STATES = {"pending", "running", "stopping"}
_ACTIVE_COMMAND_STATES = {"Pending", "InProgress", "Delayed"}
_INSTANCE_FIELDS = {
    "instance_id",
    "instance_type",
    "profile_instance_type",
    "state",
    "instance_profile_arn",
    "provider",
    "seed",
    "cohort_sha256",
    "release_sha256",
    "dataset_sha256",
    "run_manifest_sha256",
    "profile_sha256",
    "gres",
}
_SELECTED_INSTANCE_FIELDS = _INSTANCE_FIELDS | {
    "ami_id",
    "container_digest",
    "runtime_sha256",
    "terminate_at",
}
_V3_INSTANCE_FIELDS = {
    "preregistration_sha256",
    "hardware_amendment_sha256",
    "provider_selection_sha256",
    "sealed_fixture_sha256",
    "fleet_plan_sha256",
    "fleet_wave",
    "launch_readiness_sha256",
    "control_bundle_sha256",
}
_SAFE_PATH = "/usr/local/bin:/usr/bin:/bin"
_MAX_PAID_RUNTIME = timedelta(minutes=1440)
_AWS_PRIVATE_HOME = "/var/lib/memorysplit/aws-private-home"
_AWS_DSA_CERTIFICATES = AWS_INSTANCE_IDENTITY_CERTIFICATES
_OPERATOR_CREDENTIAL_VARIABLES = frozenset(
    {
        "MSCTL_AWS_CONFIG_FILE",
        "MSCTL_AWS_CONFIG_SHA256",
        "MSCTL_AWS_CREDENTIAL_PROCESS_SHA256",
        "MSCTL_AWS_PROFILE",
    }
)
_MAX_AWS_CONFIG_BYTES = 64 * 1024
_MAX_CREDENTIAL_PROCESS_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class V3LifecycleContext:
    """Validated amendment, selection, and fleet binding for one seed pair."""

    amendment: HardwareAmendment
    selection: ProviderSelection
    fleet_plan: FleetPlan
    fleet_binding: FleetManifestBinding
    readiness: LaunchReadiness | None = None
    sealed_fixture: SealedEvaluationFixture | None = None
    sealed_evaluation: SealedEvaluationRelease | None = None

    @property
    def instance_id(self) -> str:
        return self.fleet_binding.instance_id


@dataclass(frozen=True)
class OperatorCredentialProcess:
    """One reviewed AWS CLI credential_process configuration."""

    config_file: Path
    config_sha256: str
    profile: str
    executable: Path
    executable_sha256: str

    def environment(self) -> tuple[str, ...]:
        return (
            f"AWS_CONFIG_FILE={self.config_file}",
            f"AWS_PROFILE={self.profile}",
            "AWS_SHARED_CREDENTIALS_FILE=/dev/null",
            "AWS_EC2_METADATA_DISABLED=true",
            "AWS_SDK_LOAD_CONFIG=1",
        )


def _reviewed_regular_bytes(
    path: Path,
    *,
    label: str,
    max_bytes: int | None = None,
    executable: bool = False,
) -> bytes:
    descriptor: int | None = None
    try:
        if not path.is_absolute():
            raise MsctlError(
                "AWS_CREDENTIAL_PROCESS_INVALID",
                f"{label} is not one safe reviewed regular file",
            )
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid not in {0, os.geteuid()}
            or before.st_mode & 0o022
            or before.st_size <= 0
            or (max_bytes is not None and before.st_size > max_bytes)
            or (executable and (before.st_mode & 0o111) == 0)
        ):
            raise MsctlError(
                "AWS_CREDENTIAL_PROCESS_INVALID",
                f"{label} is not one safe reviewed regular file",
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as error:
        raise MsctlError(
            "AWS_CREDENTIAL_PROCESS_INVALID",
            f"{label} cannot be read safely",
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
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
    ) or len(payload) != after.st_size:
        raise MsctlError(
            "AWS_CREDENTIAL_PROCESS_INVALID",
            f"{label} changed while being read",
        )
    return payload


def load_operator_credential_process(
    environment: Mapping[str, str],
    *,
    region: str,
) -> OperatorCredentialProcess:
    """Load a hash-pinned credential_process-only AWS CLI configuration."""

    missing = sorted(
        name
        for name in _OPERATOR_CREDENTIAL_VARIABLES
        if not isinstance(environment.get(name), str) or not environment[name]
    )
    if missing:
        raise MsctlError(
            "AWS_CREDENTIAL_PROCESS_REQUIRED",
            "AWS CLI access requires one explicit reviewed credential_process",
            details={"missing": missing},
        )
    config_path = Path(environment["MSCTL_AWS_CONFIG_FILE"])
    config_digest = environment["MSCTL_AWS_CONFIG_SHA256"]
    executable_digest = environment[
        "MSCTL_AWS_CREDENTIAL_PROCESS_SHA256"
    ]
    profile = environment["MSCTL_AWS_PROFILE"]
    if (
        _SHA256_RE.fullmatch(config_digest) is None
        or _SHA256_RE.fullmatch(executable_digest) is None
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", profile) is None
    ):
        raise MsctlError(
            "AWS_CREDENTIAL_PROCESS_INVALID",
            "credential process hashes or profile name are invalid",
        )
    config_payload = _reviewed_regular_bytes(
        config_path,
        label="AWS credential_process config",
        max_bytes=_MAX_AWS_CONFIG_BYTES,
    )
    if hashlib.sha256(config_payload).hexdigest() != config_digest:
        raise MsctlError(
            "AWS_CREDENTIAL_PROCESS_INVALID",
            "AWS credential_process config SHA-256 does not match review",
        )
    parser = configparser.ConfigParser(
        interpolation=None,
        strict=True,
        delimiters=("=",),
    )
    try:
        parser.read_string(config_payload.decode("utf-8"))
    except (UnicodeDecodeError, configparser.Error) as error:
        raise MsctlError(
            "AWS_CREDENTIAL_PROCESS_INVALID",
            "AWS credential_process config is not strict UTF-8 INI",
        ) from error
    section_name = f"profile {profile}"
    if (
        parser.defaults()
        or parser.sections() != [section_name]
        or set(parser[section_name])
        != {"credential_process", "output", "region"}
        or parser[section_name]["region"] != region
        or parser[section_name]["output"] != "json"
    ):
        raise MsctlError(
            "AWS_CREDENTIAL_PROCESS_INVALID",
            "AWS config must contain only the reviewed credential_process profile",
        )
    command = parser[section_name]["credential_process"]
    try:
        process_argv = shlex.split(command, posix=True)
    except ValueError as error:
        raise MsctlError(
            "AWS_CREDENTIAL_PROCESS_INVALID",
            "credential_process command cannot be parsed",
        ) from error
    if (
        len(process_argv) != 1
        or not process_argv[0]
        or any(c in process_argv[0] for c in "\x00\n\r")
        or not Path(process_argv[0]).is_absolute()
    ):
        raise MsctlError(
            "AWS_CREDENTIAL_PROCESS_INVALID",
            "credential_process must be exactly one absolute executable",
        )
    executable = Path(process_argv[0])
    executable_payload = _reviewed_regular_bytes(
        executable,
        label="AWS credential_process executable",
        max_bytes=_MAX_CREDENTIAL_PROCESS_BYTES,
        executable=True,
    )
    if hashlib.sha256(executable_payload).hexdigest() != executable_digest:
        raise MsctlError(
            "AWS_CREDENTIAL_PROCESS_INVALID",
            "AWS credential_process executable SHA-256 does not match review",
        )
    return OperatorCredentialProcess(
        config_file=config_path,
        config_sha256=config_digest,
        profile=profile,
        executable=executable,
        executable_sha256=executable_digest,
    )


def _manifest_schema(manifest: object) -> int | None:
    value = getattr(manifest, "schema_version", None)
    return value if type(value) is int else None


def _is_v3_manifest(manifest: object) -> bool:
    return _manifest_schema(manifest) == 3


def _diagnostic_receipt_map(
    values: Sequence[str] | None,
) -> dict[str, Path] | None:
    if values is None:
        return None
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if (
            separator != "="
            or name not in DIAGNOSTIC_IDS
            or name in result
            or not path
        ):
            raise MsctlError(
                "CLI_USAGE",
                "--diagnostic-receipt must be one unique NAME=PATH for each gate",
            )
        result[name] = Path(path)
    if set(result) != set(DIAGNOSTIC_IDS):
        raise MsctlError(
            "CLI_USAGE",
            "all six named diagnostic receipts are required",
        )
    return result


def _verify_instance_identity_pkcs7(
    identity: Mapping[str, object],
    pkcs7: str,
    region: str,
) -> bool:
    try:
        return verify_instance_identity_pkcs7(identity, pkcs7, region)
    except AwsIdentityError as error:
        raise MsctlError(
            "ENVIRONMENT_RECEIPT_INVALID",
            str(error),
        ) from error


class AwsJsonRunner(Protocol):
    def run_json(
        self,
        argv: Sequence[str],
        *,
        operation: str,
    ) -> object: ...


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


class SubprocessAwsJsonRunner:
    """Run explicit argv with an empty inherited environment and parse JSON."""

    def run_json(
        self,
        argv: Sequence[str],
        *,
        operation: str,
    ) -> object:
        try:
            completed = subprocess.run(
                list(argv),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
                env={},
                timeout=60,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as error:
            raise MsctlError(
                "EXTERNAL_UNAVAILABLE",
                "AWS CLI is unavailable",
                details={"operation": operation},
            ) from error
        if completed.returncode != 0:
            raise MsctlError(
                "AWS_COMMAND_FAILED",
                "AWS CLI operation failed",
                details={"operation": operation},
            )
        try:
            return json.loads(
                completed.stdout,
                object_pairs_hook=_strict_json_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON value: {value}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise MsctlError(
                "AWS_OUTPUT_INVALID",
                "AWS CLI did not return one valid JSON value",
                details={"operation": operation},
            ) from error


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _aws_output_object(
    value: object,
    fields: set[str],
    *,
    label: str,
) -> dict[str, object]:
    try:
        row = require_object(value, label=label)
        require_exact_keys(row, fields, label=label)
    except MsctlError as error:
        raise MsctlError(
            "AWS_OUTPUT_INVALID",
            f"{label} does not match the closed AWS output schema",
        ) from error
    return row


def _aws_output_list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise MsctlError(
            "AWS_OUTPUT_INVALID",
            f"{label} must be a JSON array",
        )
    return value


def _validate_profile(profile: object) -> None:
    provider = getattr(profile, "provider", None)
    expected_instance_type = _PROFILE_INSTANCE_TYPES.get(provider)
    if (
        provider not in AWS_GPU_PROFILES
        or getattr(profile, "profile_id", None) != provider
        or getattr(profile, "instance_type", None) != expected_instance_type
        or not isinstance(getattr(profile, "sha256", None), str)
        or _SHA256_RE.fullmatch(profile.sha256) is None
        or _profile_gres(profile) != _PROFILE_GRES.get(provider)
        or getattr(profile, "purchase_model", None)
        != _PROFILE_PURCHASE_MODELS.get(provider)
        or getattr(profile, "allocated_gpus", None) != 8
        or getattr(profile, "train_groups", None) != (4, 4)
        or getattr(profile, "assigned_seeds", None)
        != _PROFILE_SEEDS.get(provider)
        or (
            hasattr(profile, "vcpus")
            and getattr(profile, "vcpus") != 192
        )
        or (
            hasattr(profile, "memory_gib")
            and getattr(profile, "memory_gib")
            != (
                4096
                if provider == "aws-p6-b300.48xlarge-v3"
                else 2048
            )
        )
    ):
        raise MsctlError(
            "PROFILE_INVALID",
            "AWS backend requires one exact known 4+4 GPU provider profile",
        )


def _profile_gres(profile: object) -> str | None:
    value = getattr(profile, "gres", None)
    if value is None and getattr(profile, "provider", None) == AWS_P5_PROFILE:
        # Compatibility for legacy in-process profile doubles.
        return "gpu:h100:8"
    return value if isinstance(value, str) else None


def _validate_runtime(runtime: object) -> None:
    s3_root = getattr(runtime, "s3_root", None)
    container_image = getattr(runtime, "container_image", None)
    container_digest = getattr(runtime, "container_digest", None)
    try:
        parsed = urlsplit(s3_root) if isinstance(s3_root, str) else None
        parsed_port = parsed.port if parsed is not None else None
    except ValueError:
        parsed = None
        parsed_port = None
    path_parts = (
        parsed.path.removeprefix("/").split("/") if parsed is not None else []
    )
    if (
        not isinstance(getattr(runtime, "region", None), str)
        or _REGION_RE.fullmatch(runtime.region) is None
        or parsed is None
        or parsed.scheme != "s3"
        or parsed.username is not None
        or parsed.password is not None
        or parsed_port is not None
        or parsed.query
        or parsed.fragment
        or parsed.hostname is None
        or parsed.netloc != parsed.hostname
        or _BUCKET_RE.fullmatch(parsed.hostname) is None
        or not path_parts
        or any(part in {"", ".", ".."} for part in path_parts)
        or "\\" in s3_root
        or not isinstance(getattr(runtime, "ami_id", None), str)
        or _AMI_RE.fullmatch(runtime.ami_id) is None
        or not isinstance(getattr(runtime, "container_digest", None), str)
        or _DIGEST_RE.fullmatch(runtime.container_digest) is None
        or not isinstance(container_image, str)
        or container_image != (
            container_image.partition("@")[0] + "@" + str(container_digest)
        )
        or container_image.count("@") != 1
        or "/" not in container_image.partition("@")[0]
        or "." not in container_image.partition("/")[0]
        or any(character.isspace() for character in container_image)
    ):
        raise MsctlError(
            "AWS_RUNTIME_INVALID",
            "AWS runtime identities are incomplete or mutable",
        )


def aws_resource_request(
    operation: str,
    *,
    bindings: Mapping[str, object] | None = None,
    profile: object | None = None,
) -> dict[str, object]:
    if profile is not None:
        _validate_profile(profile)
    policy = {
        "submit": (1, 8, 1440, "scripts/run_train.py"),
        "resume": (1, 8, 1440, "scripts/run_train.py"),
        "evaluate": (1, 8, 360, "evals/confirmatory/runner.py"),
        "cancel": (1, 0, 0, "aws:ssm:cancel-command"),
        "cleanup": (1, 0, 0, "aws:ec2:terminate-instances"),
        "canary": (1, 8, 60, "cluster/aws/p5/canary_orchestrator.py"),
    }
    if operation not in policy:
        raise MsctlError(
            "APPROVAL_INVALID",
            "AWS operation has no approval resource policy",
        )
    jobs, requested_gpus, wall_minutes, script = policy[operation]
    gpus = (
        int(getattr(profile, "allocated_gpus", requested_gpus))
        if requested_gpus
        else 0
    )
    gres = (
        _profile_gres(profile)
        if profile is not None and gpus
        else ("gpu:h100:8" if gpus else "none")
    )
    if (
        type(gpus) is not int
        or gpus != requested_gpus
        or not isinstance(gres, str)
        or (gpus and not gres.endswith(f":{gpus}"))
    ):
        raise MsctlError(
            "PROFILE_INVALID",
            "AWS GPU resource identity does not match the lifecycle policy",
        )
    request: dict[str, object] = {
        "schema_version": 1,
        "operation": operation,
        "jobs": jobs,
        "allocated_gpus": gpus,
        "wall_minutes": wall_minutes,
        "gpu_hours": gpus * wall_minutes / 60.0,
        "gres": gres,
        "script": script,
    }
    if bindings is not None:
        request.update(bindings)
    return request


class AwsP5Backend:
    """Compatibility-named backend with one injected JSON process boundary."""

    def __init__(
        self,
        *,
        profile: object,
        runtime: object,
        instance_profile_arn: str,
        state_root: Path | str,
        runner: AwsJsonRunner | None = None,
        approval_verifier: Callable[..., object] = verify_scope_approval,
        corpus_verifier: Callable[..., object] | None = None,
        identity_verifier: Callable[
            [Mapping[str, object], str, str],
            bool,
        ] = _verify_instance_identity_pkcs7,
        environ: Mapping[str, str] | None = None,
        operator_credentials: OperatorCredentialProcess | None = None,
    ) -> None:
        _validate_profile(profile)
        _validate_runtime(runtime)
        if (
            not isinstance(instance_profile_arn, str)
            or _INSTANCE_PROFILE_RE.fullmatch(instance_profile_arn) is None
        ):
            raise MsctlError(
                "AWS_RUNTIME_INVALID",
                "an exact IAM instance profile ARN is required",
            )
        self.profile = profile
        self.runtime = runtime
        self.instance_profile_arn = instance_profile_arn
        self.state_root = Path(state_root)
        self.runner = runner or SubprocessAwsJsonRunner()
        self.approval_verifier = approval_verifier
        self.corpus_verifier = corpus_verifier
        self.identity_verifier = identity_verifier
        self.environ = dict(os.environ if environ is None else environ)
        self.operator_credentials = operator_credentials
        self.control_bundle = build_control_bundle_bytes(
            Path(__file__).resolve().parents[1]
        )

    def _aws_argv(
        self,
        service: str,
        action: str,
        *arguments: str,
        query: str,
    ) -> list[str]:
        credential_environment = (
            list(self.operator_credentials.environment())
            if self.operator_credentials is not None
            else []
        )
        return [
            "env",
            "-i",
            f"AWS_REGION={self.runtime.region}",
            "HOME=/tmp",
            f"PATH={_SAFE_PATH}",
            *credential_environment,
            "aws",
            "--no-cli-pager",
            "--region",
            self.runtime.region,
            service,
            action,
            *arguments,
            "--output",
            "json",
            "--query",
            query,
        ]

    def _run(
        self,
        argv: Sequence[str],
        *,
        operation: str,
    ) -> object:
        return self.runner.run_json(argv, operation=operation)

    def auth_check(self) -> dict[str, object]:
        argv = self._aws_argv(
            "sts",
            "get-caller-identity",
            query="{account:Account,arn:Arn,user_id:UserId}",
        )
        output = _aws_output_object(
            self._run(argv, operation="auth check"),
            {"account", "arn", "user_id"},
            label="AWS caller identity",
        )
        if not all(
            isinstance(output[field], str) and output[field]
            for field in ("account", "arn", "user_id")
        ):
            raise MsctlError(
                "AWS_OUTPUT_INVALID",
                "AWS caller identity fields must be non-empty strings",
            )
        return {
            "provider": self.profile.provider,
            "instance_type": self.profile.instance_type,
            "profile_sha256": self.profile.sha256,
            "gres": _profile_gres(self.profile),
            "region": self.runtime.region,
            **output,
        }

    def _validate_manifest(self, manifest: object) -> None:
        schema_version = _manifest_schema(manifest)
        expected_schema = 2 if self.profile.provider == AWS_P5_PROFILE else 3
        if schema_version != expected_schema:
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "AWS lifecycle manifest schema does not match the selected profile",
                details={
                    "provider": self.profile.provider,
                    "expected_schema_version": expected_schema,
                    "actual_schema_version": schema_version,
                },
            )
        runs = getattr(manifest, "runs", ())
        if (
            getattr(manifest, "provider", None) != self.profile.provider
            or getattr(manifest, "seed", None)
            not in getattr(self.profile, "assigned_seeds", ())
            or len(runs) != 2
            or {getattr(run, "arm", None) for run in runs}
            != {"dense", "split90"}
            or {getattr(run, "seed", None) for run in runs}
            != {getattr(manifest, "seed", None)}
        ):
            raise MsctlError(
                "SEED_OWNERSHIP_VIOLATION",
                "AWS lifecycle requires one owned Dense/Split90 seed pair",
            )
        if (
            not isinstance(getattr(manifest, "source_commit", None), str)
            or _COMMIT_RE.fullmatch(manifest.source_commit) is None
        ):
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "AWS lifecycle requires a source-bound run manifest",
            )
        hash_fields = [
            "release_sha256",
            "dataset_sha256",
            "cohort_assignment_sha256",
            "sha256",
        ]
        if schema_version == 2:
            hash_fields.append("study_lock_sha256")
        else:
            hash_fields.extend(_V3_PROVENANCE_FIELDS[1:])
        for field in hash_fields:
            try:
                require_sha256(
                    getattr(manifest, field, None),
                    label=f"run manifest.{field}",
                )
            except MsctlError as error:
                raise MsctlError(
                    "RUN_MANIFEST_INVALID",
                    "AWS run manifest is missing a hash binding",
                ) from error
        for run in runs:
            try:
                require_sha256(
                    getattr(run, "config_sha256", None),
                    label="run config hash",
                )
            except MsctlError as error:
                raise MsctlError(
                    "RUN_MANIFEST_INVALID",
                    "AWS run manifest has an invalid config binding",
                ) from error
        if schema_version == 3:
            instance_hours = getattr(manifest, "estimated_instance_hours", None)
            gpu_hours = getattr(manifest, "estimated_gpu_hours", None)
            if (
                getattr(manifest, "profile_sha256", None) != self.profile.sha256
                or isinstance(instance_hours, bool)
                or not isinstance(instance_hours, (int, float))
                or instance_hours <= 0
                or isinstance(gpu_hours, bool)
                or not isinstance(gpu_hours, (int, float))
                or gpu_hours != instance_hours * self.profile.allocated_gpus
            ):
                raise MsctlError(
                    "RUN_MANIFEST_INVALID",
                    "v3 manifest profile or estimated-hour binding is invalid",
                )

    def _validate_v3_context(
        self,
        manifest: object,
        context: V3LifecycleContext | None,
        *,
        instance_id: str | None = None,
    ) -> V3LifecycleContext | None:
        if not _is_v3_manifest(manifest):
            if context is not None:
                raise MsctlError(
                    "FLEET_PLAN_INVALID",
                    "v2 lifecycle cannot consume v3 amendment or fleet bindings",
                )
            return None
        if context is None:
            raise MsctlError(
                "FLEET_PLAN_REQUIRED",
                "v3 lifecycle requires a validated provider selection and fleet plan",
            )
        selection = context.selection
        plan = context.fleet_plan
        fixture_sha256 = (
            context.sealed_fixture.sha256
            if context.sealed_fixture is not None
            else (
                context.sealed_evaluation.fixture_sha256
                if context.sealed_evaluation is not None
                else None
            )
        )
        binding = validate_fleet_manifest(
            plan,
            manifest,
            instance_id=instance_id,
        )
        if (
            context.fleet_binding != binding
            or context.amendment.sha256
            != getattr(manifest, "hardware_amendment_sha256", None)
            or selection.sha256
            != getattr(manifest, "provider_selection_sha256", None)
            or selection.profile_sha256 != self.profile.sha256
            or selection.provider != self.profile.provider
            or selection.region != self.runtime.region
            or selection.ami_id != self.runtime.ami_id
            or selection.container_image != self.runtime.container_image
            or selection.container_digest != self.runtime.container_digest
            or (
                fixture_sha256 is not None
                and fixture_sha256
                != getattr(manifest, "sealed_fixture_sha256", None)
            )
            or (
                context.readiness is not None
                and (
                    fixture_sha256 is None
                    or context.readiness.bindings[
                        "sealed_fixture_sha256"
                    ]
                    != fixture_sha256
                )
            )
        ):
            raise MsctlError(
                "PROVIDER_SELECTION_MISMATCH",
                "v3 lifecycle context conflicts with the selected runtime",
            )
        return context

    def _require_protected_launch_context(
        self,
        manifest: object,
        context: V3LifecycleContext | None,
    ) -> V3LifecycleContext | None:
        validated = self._validate_v3_context(manifest, context)
        if not _is_v3_manifest(manifest):
            return None
        if (
            validated is None
            or validated.readiness is None
            or (
                validated.sealed_fixture is None
                and validated.sealed_evaluation is None
            )
            or validated.readiness.bindings.get("release_sha256")
            != getattr(manifest, "release_sha256", None)
            or validated.readiness.bindings.get(
                "provider_selection_sha256"
            )
            != validated.selection.sha256
            or validated.readiness.bindings.get(
                "hardware_amendment_sha256"
            )
            != validated.amendment.sha256
            or validated.readiness.decision.get(
                "protected_launch_allowed"
            )
            is not True
        ):
            raise MsctlError(
                "LAUNCH_READINESS_REQUIRED",
                "v3 protected submit, resume, and evaluation require one "
                "validated affirmative launch-readiness receipt",
            )
        return validated

    def _require_fleet_advance(
        self,
        manifest: object,
        context: V3LifecycleContext | None,
    ) -> None:
        if not _is_v3_manifest(manifest):
            return
        validated = self._validate_v3_context(manifest, context)
        assert validated is not None
        binding = validated.fleet_binding
        if binding.wave == 0:
            return
        wave_bindings = tuple(
            candidate
            for candidate in validated.fleet_plan.manifests
            if candidate.wave == binding.wave
        )
        missing = [
            {
                "instance_id": candidate.instance_id,
                "seed": candidate.seed,
                "wave": candidate.wave,
            }
            for candidate in wave_bindings
            if load_fleet_advance(
                self.state_root,
                plan=validated.fleet_plan,
                to_binding=candidate,
            )
            is None
        ]
        if missing:
            raise MsctlError(
                "FLEET_ADVANCE_REQUIRED",
                "later fleet waves require completed local fleet-advance "
                "receipts for every instance in the wave; absent AWS tags "
                "are not proof",
                details={
                    "instance_id": binding.instance_id,
                    "seed": binding.seed,
                    "wave": binding.wave,
                    "missing": missing,
                },
            )

    def _load_v3_context(
        self,
        manifest: object,
        *,
        release: object,
        amendment_path: Path | str | None,
        provider_selection_path: Path | str | None,
        fleet_plan_path: Path | str | None,
        readiness_path: Path | str | None,
        environment_receipt_path: Path | str | None,
        qualification_receipt_path: Path | str | None,
        diagnostic_receipts: Mapping[str, Path | str] | None,
        sealed_evaluation_fixture_path: Path | str | None,
        sealed_evaluation_release_path: Path | str | None,
        expected_sealed_evaluation_sha256: str | None,
        expected_study_lock_sha256: str | None,
        require_readiness: bool,
        require_finalized_evaluation: bool,
        instance_id: str | None = None,
    ) -> V3LifecycleContext | None:
        if not _is_v3_manifest(manifest):
            if any(
                value is not None
                for value in (
                    amendment_path,
                    provider_selection_path,
                    fleet_plan_path,
                    readiness_path,
                    qualification_receipt_path,
                    diagnostic_receipts,
                    sealed_evaluation_fixture_path,
                    sealed_evaluation_release_path,
                    expected_sealed_evaluation_sha256,
                    expected_study_lock_sha256,
                )
            ):
                raise MsctlError(
                    "CLI_USAGE",
                    "v3 lifecycle bindings cannot be supplied to a v2 manifest",
                )
            return None
        if (
            amendment_path is None
            or provider_selection_path is None
            or fleet_plan_path is None
            or (
                require_readiness
                and (
                    readiness_path is None
                    or environment_receipt_path is None
                    or qualification_receipt_path is None
                    or diagnostic_receipts is None
                    or (
                        require_finalized_evaluation
                        and (
                            sealed_evaluation_release_path is None
                            or expected_sealed_evaluation_sha256 is None
                            or expected_study_lock_sha256 is None
                        )
                    )
                    or (
                        not require_finalized_evaluation
                        and sealed_evaluation_fixture_path is None
                    )
                )
            )
        ):
            raise MsctlError(
                "CLI_USAGE",
                "v3 lifecycle requires --hardware-amendment, "
                "--provider-selection, --fleet-plan, --launch-readiness, "
                "--qualification-receipt, six --diagnostic-receipt values, "
                "and the operation's sealed evaluator fixture/release bindings",
            )
        amendment = load_hardware_amendment(amendment_path)
        selection = load_provider_selection(
            provider_selection_path,
            amendment=amendment,
            profile=self.profile,
        )
        plan = load_fleet_plan(
            fleet_plan_path,
            profile=self.profile,
            selection=selection,
        )
        binding = validate_fleet_manifest(
            plan,
            manifest,
            instance_id=instance_id,
        )
        sealed_fixture = None
        sealed_evaluation = None
        readiness = None
        if require_readiness:
            assert readiness_path is not None
            assert environment_receipt_path is not None
            assert qualification_receipt_path is not None
            assert diagnostic_receipts is not None
            if require_finalized_evaluation:
                assert sealed_evaluation_release_path is not None
                sealed_evaluation = load_sealed_evaluation_release(
                    sealed_evaluation_release_path,
                    expected_release_sha256=expected_sealed_evaluation_sha256,
                    expected_study_lock_sha256=expected_study_lock_sha256,
                    expected_preregistration_sha256=(
                        manifest.preregistration_sha256
                    ),
                )
            else:
                assert sealed_evaluation_fixture_path is not None
                sealed_fixture = load_sealed_evaluation_fixture(
                    sealed_evaluation_fixture_path,
                    expected_fixture_sha256=manifest.sealed_fixture_sha256,
                )
            readiness = load_launch_readiness(
                readiness_path,
                profile=self.profile,
                amendment=amendment,
                selection=selection,
                release=release,
                manifest=manifest,
                environment_receipt=environment_receipt_path,
                qualification_receipt=qualification_receipt_path,
                diagnostic_receipts=diagnostic_receipts,
                sealed_evaluation_fixture=(
                    sealed_evaluation_fixture_path
                    if sealed_fixture is not None
                    else None
                ),
                expected_instance_id=binding.instance_id,
                environ=self.environ,
            )
        context = V3LifecycleContext(
            amendment=amendment,
            selection=selection,
            fleet_plan=plan,
            fleet_binding=binding,
            readiness=readiness,
            sealed_fixture=sealed_fixture,
            sealed_evaluation=sealed_evaluation,
        )
        return self._validate_v3_context(
            manifest,
            context,
            instance_id=instance_id,
        )

    def _v3_bindings(
        self,
        manifest: object,
        context: V3LifecycleContext | None,
    ) -> dict[str, object]:
        if not _is_v3_manifest(manifest):
            return {}
        validated = self._validate_v3_context(manifest, context)
        assert validated is not None
        bindings = {
            field: getattr(manifest, field)
            for field in _V3_PROVENANCE_FIELDS
        } | {
            "fleet_plan_sha256": validated.fleet_plan.sha256,
            "fleet_wave": validated.fleet_binding.wave,
            "control_bundle_sha256": self.control_bundle.sha256,
        }
        if validated.readiness is not None:
            bindings["launch_readiness_sha256"] = validated.readiness.sha256
        return bindings

    def _v3_evaluation_bindings(
        self,
        manifest: object,
        context: V3LifecycleContext | None,
    ) -> dict[str, object]:
        if not _is_v3_manifest(manifest):
            return {}
        validated = self._validate_v3_context(manifest, context)
        if validated is None or validated.sealed_evaluation is None:
            raise MsctlError(
                "SEALED_EVALUATION_REQUIRED",
                "v3 evaluation requires a finalized externally hash-bound release",
            )
        final = validated.sealed_evaluation
        if final.fixture_sha256 != manifest.sealed_fixture_sha256:
            raise MsctlError(
                "SEALED_EVALUATION_INVALID",
                "finalized evaluation release changes the launch fixture",
            )
        return {
            "sealed_evaluation_sha256": final.sha256,
            "study_lock_sha256": final.study_lock_sha256,
        }

    def _evaluation_checkpoint_binding(
        self,
        manifest: object,
        checkpoint_receipt: object | None,
    ) -> dict[str, object] | None:
        if not _is_v3_manifest(manifest):
            if checkpoint_receipt is not None:
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    "legacy evaluation cannot consume a v3 terminal bundle",
                )
            return None
        if (
            checkpoint_receipt is None
            or getattr(checkpoint_receipt, "schema_version", None) != 3
            or not isinstance(getattr(checkpoint_receipt, "sha256", None), str)
        ):
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                "v3 evaluation requires the canonical terminal checkpoint "
                "receipt identity",
            )
        checkpoints = tuple(
            getattr(checkpoint_receipt, "checkpoints", ())
        )
        by_arm = {
            str(getattr(checkpoint, "arm", "")): checkpoint
            for checkpoint in checkpoints
        }
        if len(checkpoints) != 2 or set(by_arm) != {"dense", "split90"}:
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                "v3 evaluation checkpoint receipt is not a complete pair",
            )
        receipt_sha256 = require_sha256(
            checkpoint_receipt.sha256,
            label="evaluation checkpoint receipt",
        )
        receipt_uri = (
            f"{self.runtime.s3_root.rstrip('/')}/checkpoints/"
            f"seed-{manifest.seed}/receipts/{receipt_sha256}.json"
        )
        rows: list[dict[str, object]] = []
        for arm in ("dense", "split90"):
            checkpoint = by_arm[arm]
            durable_fields = {
                field: getattr(checkpoint, field, None)
                for field in (
                    "checkpoint_uri",
                    "configuration_uri",
                    "run_binding_sha256",
                    "run_binding_uri",
                    "checkpoint_record_sha256",
                    "checkpoint_record_uri",
                )
            }
            if any(value is None for value in durable_fields.values()):
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    "v3 evaluation receipt omits durable terminal artifacts",
                )
            for field in (
                "checkpoint_uri",
                "configuration_uri",
                "run_binding_uri",
                "checkpoint_record_uri",
            ):
                self._runtime_relative_uri(
                    durable_fields[field],
                    label=f"evaluation {arm} {field}",
                )
            rows.append(
                {
                    "run_id": checkpoint.run_id,
                    "arm": arm,
                    "checkpoint_sha256": checkpoint.sha256,
                    "checkpoint_uri": durable_fields["checkpoint_uri"],
                    "configuration_sha256": checkpoint.config_sha256,
                    "configuration_uri": durable_fields[
                        "configuration_uri"
                    ],
                    "run_binding_sha256": durable_fields[
                        "run_binding_sha256"
                    ],
                    "run_binding_uri": durable_fields["run_binding_uri"],
                    "checkpoint_record_sha256": durable_fields[
                        "checkpoint_record_sha256"
                    ],
                    "checkpoint_record_uri": durable_fields[
                        "checkpoint_record_uri"
                    ],
                }
            )
        return {
            "sha256": receipt_sha256,
            "uri": receipt_uri,
            "checkpoints": rows,
        }

    def _validate_release(self, release: object, manifest: object) -> None:
        if (
            getattr(release, "provider", None) != self.profile.provider
            or getattr(release, "archive_sha256", None)
            != manifest.release_sha256
            or getattr(release, "source_commit", None)
            != manifest.source_commit
        ):
            raise MsctlError(
                "RELEASE_RUN_MISMATCH",
                "AWS release and run manifest provenance differ",
            )

    def _release_root(self, release: object) -> str:
        scratch_root = getattr(self.profile, "scratch_root", "/mnt/memorysplit")
        return f"{scratch_root}/releases/{release.archive_sha256}"

    def _validated_lifecycle_evidence(
        self,
        *,
        operation: str,
        manifest: object,
        evidence: Mapping[str, str] | None,
        context: V3LifecycleContext | None,
    ) -> dict[str, str]:
        protected_v3 = _is_v3_manifest(manifest) and operation in {
            "submit",
            "resume",
            "evaluate",
        }
        if evidence is None and protected_v3:
            raise MsctlError(
                "LIFECYCLE_EVIDENCE_REQUIRED",
                "v3 submit, resume, and evaluate require verified dataset "
                "and environment lifecycle evidence",
            )
        lifecycle_evidence = (
            dict(evidence)
            if evidence is not None
            else {
                "dataset_pointer_sha256": manifest.dataset_sha256,
                "dataset_verification_sha256": manifest.dataset_sha256,
                "environment_receipt_sha256": self._runtime_sha256(),
            }
        )
        expected_fields = {
            "dataset_pointer_sha256",
            "dataset_verification_sha256",
            "environment_receipt_sha256",
        }
        if set(lifecycle_evidence) != expected_fields:
            raise MsctlError(
                "LIFECYCLE_EVIDENCE_INVALID",
                "AWS lifecycle evidence fields do not match the contract",
            )
        for field, digest in lifecycle_evidence.items():
            require_sha256(digest, label=field)
        if protected_v3:
            readiness = context.readiness if context is not None else None
            expected_environment = (
                readiness.bindings.get("environment_receipt_sha256")
                if readiness is not None
                else None
            )
            try:
                expected_environment = require_sha256(
                    expected_environment,
                    label="launch readiness environment receipt",
                )
            except MsctlError as error:
                raise MsctlError(
                    "LAUNCH_READINESS_REQUIRED",
                    "v3 lifecycle readiness lacks its signed environment binding",
                ) from error
            if (
                lifecycle_evidence["environment_receipt_sha256"]
                != expected_environment
            ):
                raise MsctlError(
                    "ENVIRONMENT_RECEIPT_MISMATCH",
                    "v3 lifecycle evidence does not match the signed "
                    "launch-readiness environment binding",
                )
        return lifecycle_evidence

    def _training_operation_intent(
        self,
        *,
        operation: str,
        release: object,
        manifest: object,
        terminate_at: str | None = None,
        checkpoints: Mapping[str, object] | None = None,
        checkpoint_receipt_sha256: str | None = None,
        evidence: Mapping[str, str] | None = None,
        context: V3LifecycleContext | None = None,
    ) -> dict[str, object]:
        v3_bindings = self._v3_bindings(manifest, context)
        lifecycle_evidence = self._validated_lifecycle_evidence(
            operation=operation,
            manifest=manifest,
            evidence=evidence,
            context=context,
        )
        release_root = self._release_root(release)
        scratch_root = getattr(self.profile, "scratch_root", "/mnt/memorysplit")
        staging = f"{scratch_root}/staging"
        dataset_root = f"{scratch_root}/dataset"
        control_root = (
            f"/opt/memorysplit/control/{self.control_bundle.sha256}"
            if _is_v3_manifest(manifest)
            else "/opt/memorysplit"
        )
        profile_name = f"{self.profile.profile_id}.json"
        cohort_version = "v3" if _is_v3_manifest(manifest) else "v2"
        cohort_name = f"cohort-assignment-{cohort_version}.json"
        steps: list[dict[str, object]] = []
        if operation == "submit":
            assert terminate_at is not None
            steps.append(
                {
                    "name": "auto-termination",
                    "argv": [
                        "/usr/bin/systemd-run",
                        "--unit",
                        "memorysplit-auto-terminate",
                        "--on-calendar",
                        terminate_at,
                        "/sbin/shutdown",
                        "-h",
                        "now",
                    ],
                }
            )
        if operation in {"render", "submit"}:
            release_bucket, release_key = self._s3_location(
                f"releases/{release.archive_sha256}/release.zip"
            )
            receipt_bucket, receipt_key = self._s3_location(
                f"releases/{release.archive_sha256}/RELEASE.json"
            )
            cohort_bucket, cohort_key = self._s3_location(
                f"releases/{release.archive_sha256}/{cohort_name}"
            )
            steps.extend(
                [
                    {
                        "name": "prepare-aws-private-home",
                        "argv": [
                            "/usr/bin/install",
                            "-d",
                            "-m",
                            "0700",
                            "-o",
                            "0",
                            "-g",
                            "0",
                            _AWS_PRIVATE_HOME,
                        ],
                    },
                    {
                        "name": "prepare-staging",
                        "argv": [
                            "/usr/bin/install",
                            "-d",
                            "-m",
                            "0700",
                            f"{staging}/releases/{release.archive_sha256}",
                            dataset_root,
                        ],
                    },
                    {
                        "name": "materialize-release-archive",
                        "argv": [
                            "/usr/bin/env",
                            "aws",
                            "--no-cli-pager",
                            "--region",
                            self.runtime.region,
                            "s3api",
                            "get-object",
                            "--bucket",
                            release_bucket,
                            "--key",
                            release_key,
                            "--checksum-mode",
                            "ENABLED",
                            (
                                f"{staging}/releases/{release.archive_sha256}/"
                                "release.zip"
                            ),
                        ],
                    },
                    {
                        "name": "materialize-release-receipt",
                        "argv": [
                            "/usr/bin/env",
                            "aws",
                            "--no-cli-pager",
                            "--region",
                            self.runtime.region,
                            "s3api",
                            "get-object",
                            "--bucket",
                            receipt_bucket,
                            "--key",
                            receipt_key,
                            "--checksum-mode",
                            "ENABLED",
                            (
                                f"{staging}/releases/{release.archive_sha256}/"
                                "RELEASE.json"
                            ),
                        ],
                    },
                    {
                        "name": "materialize-cohort-assignment",
                        "argv": [
                            "/usr/bin/env",
                            "aws",
                            "--no-cli-pager",
                            "--region",
                            self.runtime.region,
                            "s3api",
                            "get-object",
                            "--bucket",
                            cohort_bucket,
                            "--key",
                            cohort_key,
                            "--checksum-mode",
                            "ENABLED",
                            (
                                f"{staging}/releases/{release.archive_sha256}/"
                                f"{cohort_name}"
                            ),
                        ],
                    },
                    {
                        "name": "materialize-dataset",
                        "argv": [
                            "/usr/bin/env",
                            "aws",
                            "--no-cli-pager",
                            "--region",
                            self.runtime.region,
                            "s3",
                            "sync",
                            f"{self.runtime.s3_root}/dataset",
                            dataset_root,
                            "--no-follow-symlinks",
                            "--only-show-errors",
                        ],
                    },
                    {
                        "name": "bootstrap",
                        "argv": [
                            "/usr/bin/python3",
                            f"{control_root}/cluster/aws/p5/bootstrap.py",
                            "--profile",
                            f"{control_root}/cluster/profiles/{profile_name}",
                            "--container-image",
                            self.runtime.container_image,
                            "--release-archive",
                            (
                                f"{staging}/releases/{release.archive_sha256}/"
                                "release.zip"
                            ),
                            "--release-sha256",
                            release.archive_sha256,
                            "--release-receipt",
                            (
                                f"{staging}/releases/{release.archive_sha256}/"
                                "RELEASE.json"
                            ),
                            "--release-receipt-sha256",
                            getattr(
                                release,
                                "receipt_sha256",
                                manifest.release_sha256,
                            ),
                            "--dataset-receipt",
                            f"{dataset_root}/receipt.json",
                            "--dataset-receipt-sha256",
                            manifest.dataset_sha256,
                            "--cohort-assignment",
                            (
                                f"{staging}/releases/{release.archive_sha256}/"
                                f"{cohort_name}"
                            ),
                            "--cohort-assignment-sha256",
                            manifest.cohort_assignment_sha256,
                            "--code-commit",
                            manifest.source_commit,
                            "--receipt",
                            f"{staging}/bootstrap-receipt.json",
                            "--owner-uid",
                            str(getattr(self.runtime, "uid", 1000)),
                            "--owner-gid",
                            str(getattr(self.runtime, "gid", 1000)),
                            "--aws-private-home",
                            _AWS_PRIVATE_HOME,
                            "--authorize-destructive-instance-store",
                            "--apply",
                        ],
                    },
                ]
            )
            steps.append(
                {
                    "name": "build-launcher-manifest",
                    "argv": [
                        "/usr/bin/python3",
                        f"{release_root}/msctl/aws_launch_manifest.py",
                        "--profile",
                        f"{release_root}/cluster/profiles/{profile_name}",
                        "--out",
                        f"{staging}/launcher-manifest-{manifest.sha256}.json",
                        "--scratch-root",
                        scratch_root,
                        "--seed",
                        str(manifest.seed),
                        "--profile-sha256",
                        self.profile.sha256,
                        "--release-sha256",
                        release.archive_sha256,
                        "--release-members-sha256",
                        getattr(release, "members_sha256", ""),
                        "--cohort-assignment-sha256",
                        manifest.cohort_assignment_sha256,
                        "--code-commit",
                        manifest.source_commit,
                        *(
                            argument
                            for field, option in (
                                (
                                    "preregistration_sha256",
                                    "--preregistration-sha256",
                                ),
                                (
                                    "hardware_amendment_sha256",
                                    "--hardware-amendment-sha256",
                                ),
                                (
                                    "provider_selection_sha256",
                                    "--provider-selection-sha256",
                                ),
                                (
                                    "sealed_fixture_sha256",
                                    "--sealed-fixture-sha256",
                                ),
                                (
                                    "fleet_plan_sha256",
                                    "--fleet-plan-sha256",
                                ),
                                (
                                    "launch_readiness_sha256",
                                    "--launch-readiness-sha256",
                                ),
                                (
                                    "control_bundle_sha256",
                                    "--control-bundle-sha256",
                                ),
                            )
                            if field in v3_bindings
                            for argument in (option, str(v3_bindings[field]))
                        ),
                        *(
                            [
                                "--run-manifest-sha256",
                                manifest.sha256,
                                "--fleet-wave",
                                str(v3_bindings["fleet_wave"]),
                            ]
                            if v3_bindings
                            else []
                        ),
                        "--bootstrap-receipt",
                        f"{staging}/bootstrap-receipt.json",
                        "--corpus-receipt",
                        f"{dataset_root}/receipt.json",
                        *(
                            argument
                            for run in sorted(
                                manifest.runs,
                                key=lambda row: row.arm,
                            )
                            for argument in (
                                "--run",
                                canonical_json(
                                    {
                                        "arm": run.arm,
                                        "config": run.config,
                                        "config_sha256": run.config_sha256,
                                    }
                                ).decode("ascii"),
                            )
                        ),
                    ],
                }
            )
        checkpoint_binding = None
        if checkpoints is not None:
            receipt_sha256 = require_sha256(
                checkpoint_receipt_sha256,
                label="checkpoint receipt",
            )
            checkpoint_prefix = (
                f"checkpoints/seed-{manifest.seed}"
                if _is_v3_manifest(manifest)
                else "checkpoints"
            )
            resume_root = f"{staging}/resume/{receipt_sha256}"
            receipt_bucket, receipt_key = self._s3_location(
                f"{checkpoint_prefix}/receipts/{receipt_sha256}.json"
            )
            checkpoint_rows = [
                {
                    "arm": arm,
                    "resume_path": f"{resume_root}/{arm}.pt",
                    "resume_sha256": checkpoints[arm].sha256,
                    "world_size": checkpoints[arm].world_size,
                }
                for arm in ("dense", "split90")
            ]
            checkpoint_binding = {
                "sha256": receipt_sha256,
                "checkpoints": checkpoint_rows,
            }
            steps.append(
                {
                    "name": "prepare-resume-staging",
                    "argv": [
                        "/usr/bin/install",
                        "-d",
                        "-m",
                        "0700",
                        resume_root,
                    ],
                }
            )
            steps.append(
                {
                    "name": "materialize-resume-receipt",
                    "argv": [
                        "/usr/bin/env",
                        "aws",
                        "--no-cli-pager",
                        "--region",
                        self.runtime.region,
                        "s3api",
                        "get-object",
                        "--bucket",
                        receipt_bucket,
                        "--key",
                        receipt_key,
                        "--checksum-mode",
                        "ENABLED",
                        f"{resume_root}/receipt.json",
                    ],
                }
            )
            for row in checkpoint_rows:
                bucket, key = self._s3_location(
                    f"{checkpoint_prefix}/{row['arm']}/sha256/"
                    f"{row['resume_sha256']}.pt"
                )
                steps.append(
                    {
                        "name": f"materialize-resume-{row['arm']}",
                        "argv": [
                            "/usr/bin/env",
                            "aws",
                            "--no-cli-pager",
                            "--region",
                            self.runtime.region,
                            "s3api",
                            "get-object",
                            "--bucket",
                            bucket,
                            "--key",
                            key,
                            "--checksum-mode",
                            "ENABLED",
                            row["resume_path"],
                        ],
                    }
                )
            launcher_argv = [
                "/usr/bin/python3",
                f"{release_root}/msctl/aws_resume_launch.py",
                "--seed",
                str(manifest.seed),
                "--manifest",
                f"{staging}/launcher-manifest-{manifest.sha256}.json",
                "--profile",
                f"{release_root}/cluster/profiles/{profile_name}",
                "--repo-root",
                release_root,
                "--scratch-root",
                scratch_root,
                "--launcher",
                f"{release_root}/cluster/aws/p5/launch_seed_pair.py",
                "--checkpoint-receipt",
                f"{resume_root}/receipt.json",
                "--checkpoint-receipt-sha256",
                receipt_sha256,
                "--run-manifest-sha256",
                manifest.sha256,
                *(
                    argument
                    for row in checkpoint_rows
                    for argument in (
                        "--checkpoint",
                        canonical_json(row).decode("ascii"),
                    )
                ),
                "--apply",
            ]
        else:
            launcher_argv = [
                "/usr/bin/python3",
                f"{release_root}/cluster/aws/p5/launch_seed_pair.py",
                "--seed",
                str(manifest.seed),
                "--manifest",
                f"{staging}/launcher-manifest-{manifest.sha256}.json",
                "--profile",
                f"{release_root}/cluster/profiles/{profile_name}",
                "--repo-root",
                release_root,
                "--scratch-root",
                scratch_root,
                "--apply",
            ]
        steps.append({"name": "paired-launch", "argv": launcher_argv})
        return {
            "schema_version": 3 if v3_bindings else 1,
            "operation": operation,
            "provider": self.profile.provider,
            "instance_type": self.profile.instance_type,
            "profile_sha256": self.profile.sha256,
            "gres": _profile_gres(self.profile),
            "seed": manifest.seed,
            "release_sha256": release.archive_sha256,
            "run_manifest_sha256": manifest.sha256,
            "dataset_sha256": manifest.dataset_sha256,
            **v3_bindings,
            **lifecycle_evidence,
            "runtime_sha256": self._runtime_sha256(),
            "environment": {
                "AWS_REGION": self.runtime.region,
                "MS_AWS_AMI_ID": self.runtime.ami_id,
                "MS_CONTAINER_DIGEST": self.runtime.container_digest,
                "MS_RUNTIME_GID": str(getattr(self.runtime, "gid", 1000)),
                "MS_RUNTIME_UID": str(getattr(self.runtime, "uid", 1000)),
                "MS_S3_ROOT": self.runtime.s3_root,
                **(
                    {
                        "MS_S3_KMS_KEY_ID": str(
                            getattr(self.runtime, "kms_key_id", "")
                        )
                    }
                    if v3_bindings
                    else {}
                ),
            },
            "checkpoint_receipt": checkpoint_binding,
            "steps": steps,
        }

    def _operation_envelope(
        self,
        core: Mapping[str, object],
        *,
        instance_id: str,
        terminate_at: str | None,
    ) -> dict[str, object]:
        identity = {
            **dict(core),
            "instance_id": instance_id,
            "terminate_at": terminate_at,
        }
        operation_id = canonical_sha256(identity)
        receipt_root = (
            f"{self.runtime.s3_root}/operations/{operation_id}/receipts"
        )
        return {
            **identity,
            "operation_id": operation_id,
            "ssm_document": {
                "name": ARGV_DOCUMENT_NAME,
                "sha256": ARGV_DOCUMENT_SHA256,
            },
            "started_receipt_uri": f"{receipt_root}/started.json",
            "terminal_receipt_uri": f"{receipt_root}/terminal.json",
        }

    def _verify_published_object(
        self,
        output: object,
        *,
        digest: str,
        payload_size: int,
        operation_id: str,
    ) -> dict[str, object]:
        root = _aws_output_object(
            output,
            {"object"},
            label="S3 immutable object output",
        )
        row = _aws_output_object(
            root["object"],
            {
                "checksum_sha256",
                "content_length",
                "metadata",
                "version_id",
            },
            label="S3 immutable object",
        )
        metadata = _aws_output_object(
            row["metadata"],
            {"operation-id", "sha256"},
            label="S3 immutable object metadata",
        )
        expected_checksum = base64.b64encode(
            bytes.fromhex(digest)
        ).decode("ascii")
        if (
            row["checksum_sha256"] != expected_checksum
            or row["content_length"] != payload_size
            or metadata
            != {"operation-id": operation_id, "sha256": digest}
            or not isinstance(row["version_id"], str)
            or not row["version_id"]
        ):
            raise MsctlError(
                "S3_OBJECT_MISMATCH",
                "immutable S3 operation intent does not match local bytes",
            )
        return row

    def _publish_operation_intent(
        self,
        intent: Mapping[str, object],
    ) -> dict[str, object]:
        payload = canonical_json(intent)
        digest = hashlib.sha256(payload).hexdigest()
        operation_id = str(intent.get("operation_id"))
        if require_sha256(operation_id, label="operation intent ID") != operation_id:
            raise AssertionError("unreachable")
        bucket, key = self._s3_location(
            f"operations/intents/sha256/{digest}.json"
        )
        directory_fd = open_directory(
            self.state_root,
            label="AWS state root",
            create=True,
        )
        name = f"intent-{digest}.json"
        try:
            atomic_write_at(
                directory_fd,
                name,
                payload,
                label="operation intent",
            )
        finally:
            os.close(directory_fd)
        body = str((self.state_root / name).absolute())
        checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
        put = self._aws_argv(
            "s3api",
            "put-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--body",
            body,
            "--checksum-algorithm",
            "SHA256",
            "--checksum-sha256",
            checksum,
            "--metadata",
            f"operation-id={operation_id},sha256={digest}",
            "--if-none-match",
            "*",
            query=(
                "{object:{checksum_sha256:ChecksumSHA256,"
                "version_id:VersionId}}"
            ),
        )
        try:
            put_output = _aws_output_object(
                self._run(put, operation="publish operation intent"),
                {"object"},
                label="S3 put-object output",
            )
        except MsctlError:
            put_output = None
        if put_output is not None:
            put_row = _aws_output_object(
                put_output["object"],
                {"checksum_sha256", "version_id"},
                label="S3 put-object",
            )
            if (
                put_row["checksum_sha256"] != checksum
                or not isinstance(put_row["version_id"], str)
                or not put_row["version_id"]
            ):
                raise MsctlError(
                    "S3_OBJECT_MISMATCH",
                    "S3 did not confirm the immutable intent checksum",
                )
        head = self._aws_argv(
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--checksum-mode",
            "ENABLED",
            query=(
                "{object:{checksum_sha256:ChecksumSHA256,"
                "content_length:ContentLength,metadata:Metadata,"
                "version_id:VersionId}}"
            ),
        )
        row = self._verify_published_object(
            self._run(head, operation="verify operation intent"),
            digest=digest,
            payload_size=len(payload),
            operation_id=operation_id,
        )
        return {
            "intent_sha256": digest,
            "intent_uri": f"s3://{bucket}/{key}",
            "version_id": row["version_id"],
        }

    def _publish_control_bundle(
        self,
        bundle: ControlBundle | None = None,
    ) -> dict[str, object]:
        """Publish and independently verify the content-addressed bootstrap."""

        bundle = bundle or self.control_bundle
        if (
            hashlib.sha256(bundle.payload).hexdigest() != bundle.sha256
            or len(bundle.payload) != bundle.bytes
        ):
            raise MsctlError(
                "CONTROL_BUNDLE_INVALID",
                "in-memory control bundle identity changed before publication",
            )
        bucket, key = self._s3_location(
            f"control/{bundle.sha256}.tar"
        )
        directory_fd = open_directory(
            self.state_root,
            label="AWS state root",
            create=True,
        )
        name = f"control-{bundle.sha256}.tar"
        try:
            atomic_write_at(
                directory_fd,
                name,
                bundle.payload,
                label="control bundle",
            )
        finally:
            os.close(directory_fd)
        checksum = base64.b64encode(
            bytes.fromhex(bundle.sha256)
        ).decode("ascii")
        encryption = (
            [
                "--server-side-encryption",
                "aws:kms",
                "--ssekms-key-id",
                self.runtime.kms_key_id,
            ]
            if getattr(self.runtime, "kms_key_id", None) is not None
            else []
        )
        put = self._aws_argv(
            "s3api",
            "put-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--body",
            str((self.state_root / name).absolute()),
            "--checksum-algorithm",
            "SHA256",
            "--checksum-sha256",
            checksum,
            "--metadata",
            f"bundle-type=memorysplit-aws-control-bundle-v1,sha256={bundle.sha256}",
            "--if-none-match",
            "*",
            *encryption,
            query=(
                "{object:{checksum_sha256:ChecksumSHA256,"
                "version_id:VersionId}}"
            ),
        )
        try:
            put_output = _aws_output_object(
                self._run(put, operation="publish control bundle"),
                {"object"},
                label="S3 control bundle put",
            )
        except MsctlError:
            put_output = None
        if put_output is not None:
            put_row = _aws_output_object(
                put_output["object"],
                {"checksum_sha256", "version_id"},
                label="S3 control bundle put",
            )
            if (
                put_row["checksum_sha256"] != checksum
                or not isinstance(put_row["version_id"], str)
                or not put_row["version_id"]
            ):
                raise MsctlError(
                    "S3_OBJECT_MISMATCH",
                    "S3 did not confirm the immutable control bundle",
                )
        head = self._aws_argv(
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--checksum-mode",
            "ENABLED",
            query=(
                "{object:{checksum_sha256:ChecksumSHA256,"
                "content_length:ContentLength,metadata:Metadata,"
                "server_side_encryption:ServerSideEncryption,"
                "sse_kms_key_id:SSEKMSKeyId,version_id:VersionId}}"
            ),
        )
        output = _aws_output_object(
            self._run(head, operation="verify control bundle"),
            {"object"},
            label="S3 control bundle head",
        )
        row = _aws_output_object(
            output["object"],
            {
                "checksum_sha256",
                "content_length",
                "metadata",
                "server_side_encryption",
                "sse_kms_key_id",
                "version_id",
            },
            label="S3 control bundle",
        )
        expected_metadata = {
            "bundle-type": "memorysplit-aws-control-bundle-v1",
            "sha256": bundle.sha256,
        }
        kms_key_id = getattr(self.runtime, "kms_key_id", None)
        if (
            row["checksum_sha256"] != checksum
            or row["content_length"] != bundle.bytes
            or row["metadata"] != expected_metadata
            or not isinstance(row["version_id"], str)
            or not row["version_id"]
            or (
                kms_key_id is not None
                and (
                    row["server_side_encryption"] != "aws:kms"
                    or row["sse_kms_key_id"] != kms_key_id
                )
            )
        ):
            raise MsctlError(
                "S3_OBJECT_MISMATCH",
                "published control bundle bytes, metadata, or encryption differ",
            )
        return {
            "sha256": bundle.sha256,
            "uri": f"s3://{bucket}/{key}",
            "version_id": row["version_id"],
        }

    def _control_install_plan(
        self,
        instance_id: str,
        *,
        bundle: ControlBundle,
    ) -> dict[str, object]:
        if (
            getattr(self.profile, "profile_id", None) not in _V3_PROFILES
            or _INSTANCE_ID_RE.fullmatch(instance_id) is None
            or not isinstance(getattr(self.runtime, "kms_key_id", None), str)
        ):
            raise MsctlError(
                "CONTROL_INSTALL_INVALID",
                "control install requires one explicit v3 instance and KMS key",
            )
        bucket, key = self._s3_location(f"control/{bundle.sha256}.tar")
        uri = f"s3://{bucket}/{key}"
        local_path = (
            self.state_root / f"control-{bundle.sha256}.tar"
        ).absolute()
        checksum = base64.b64encode(
            bytes.fromhex(bundle.sha256)
        ).decode("ascii")
        put = self._aws_argv(
            "s3api",
            "put-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--body",
            str(local_path),
            "--checksum-algorithm",
            "SHA256",
            "--checksum-sha256",
            checksum,
            "--metadata",
            (
                "bundle-type=memorysplit-aws-control-bundle-v1,"
                f"sha256={bundle.sha256}"
            ),
            "--if-none-match",
            "*",
            "--server-side-encryption",
            "aws:kms",
            "--ssekms-key-id",
            str(self.runtime.kms_key_id),
            query=(
                "{object:{checksum_sha256:ChecksumSHA256,"
                "version_id:VersionId}}"
            ),
        )
        head = self._aws_argv(
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--checksum-mode",
            "ENABLED",
            query=(
                "{object:{checksum_sha256:ChecksumSHA256,"
                "content_length:ContentLength,metadata:Metadata,"
                "server_side_encryption:ServerSideEncryption,"
                "sse_kms_key_id:SSEKMSKeyId,version_id:VersionId}}"
            ),
        )
        install = render_control_install_command(
            bundle_sha256=bundle.sha256,
            bundle_uri=uri,
            region=self.runtime.region,
        )
        parameters = canonical_json({"commands": [install]}).decode("ascii")
        send = self._aws_argv(
            "ssm",
            "send-command",
            "--document-name",
            "AWS-RunShellScript",
            "--instance-ids",
            instance_id,
            "--comment",
            f"memorysplit-control-{bundle.sha256}",
            "--timeout-seconds",
            "900",
            "--parameters",
            parameters,
            query=(
                "{command:{command_id:Command.CommandId,"
                "status:Command.Status}}"
            ),
        )
        return {
            "schema_version": 1,
            "instance_id": instance_id,
            "control_bundle_sha256": bundle.sha256,
            "control_bundle_uri": uri,
            "installer_sha256": hashlib.sha256(
                install.encode("ascii")
            ).hexdigest(),
            "ssm_document": "AWS-RunShellScript",
            "commands": [put, head, send],
            "installed": 0,
        }

    def control_install(
        self,
        *,
        instance_id: str,
        bundle_path: Path | str,
        bundle_sha256: str,
        apply: bool,
    ) -> dict[str, object]:
        """Install reviewed control bytes before any custom SSM document use."""

        bundle = verify_control_bundle(
            bundle_path,
            expected_sha256=bundle_sha256,
        )
        if (
            bundle.sha256 != self.control_bundle.sha256
            or bundle.payload != self.control_bundle.payload
        ):
            raise MsctlError(
                "CONTROL_BUNDLE_INVALID",
                "reviewed control bundle differs from the running msctl source",
            )
        plan = {
            **self._control_install_plan(instance_id, bundle=bundle),
            "reviewed_control_bundle": str(Path(bundle_path).resolve()),
        }
        if not apply:
            return plan
        self._require_ssm_online(instance_id)
        published = self._publish_control_bundle(bundle)
        output = _aws_output_object(
            self._run(plan["commands"][2], operation="install control bundle"),
            {"command"},
            label="SSM control install",
        )
        command = _aws_output_object(
            output["command"],
            {"command_id", "status"},
            label="SSM control install",
        )
        if (
            not isinstance(command["command_id"], str)
            or _COMMAND_ID_RE.fullmatch(command["command_id"]) is None
            or command["status"]
            not in {"Pending", "InProgress", "Delayed", "Success"}
            or published["sha256"] != plan["control_bundle_sha256"]
            or published["uri"] != plan["control_bundle_uri"]
        ):
            raise MsctlError(
                "CONTROL_INSTALL_INVALID",
                "AWS did not accept the exact control install command",
            )
        command_id = str(command["command_id"])
        status = str(command["status"])
        deadline = time.monotonic() + 900.0
        while status != "Success":
            if status not in _ACTIVE_COMMAND_STATES:
                raise MsctlError(
                    "CONTROL_INSTALL_FAILED",
                    "verified control-bundle installation did not succeed",
                    details={"command_id": command_id, "status": status},
                )
            if time.monotonic() >= deadline:
                raise MsctlError(
                    "CONTROL_INSTALL_TIMEOUT",
                    "verified control-bundle installation did not finish in time",
                    details={"command_id": command_id, "status": status},
                )
            time.sleep(2.0)
            status = self._command_status(instance_id, command_id)
        return {
            **plan,
            "command_id": command_id,
            "status": status,
            "control_bundle_version_id": published["version_id"],
            "installed": 1,
        }

    def canary_plan(
        self,
        *,
        release_path: Path | str,
        amendment_path: Path | str,
        provider_selection_path: Path | str,
        environment_receipt: Path | str,
        instance_id: str,
        out: Path | str,
        apply: bool,
    ) -> dict[str, object]:
        """Render or exclusively publish one authenticated canary plan."""

        if self.profile.provider not in _V3_PROFILES:
            raise MsctlError(
                "CANARY_PLAN_INVALID",
                "executable qualification is available only to AWS v3 profiles",
            )
        release = load_release(release_path)
        amendment = load_hardware_amendment(amendment_path)
        selection = load_provider_selection(
            provider_selection_path,
            amendment=amendment,
            profile=self.profile,
        )
        required_release_members = {
            "cluster/aws/p5/canary.py",
            "cluster/aws/p5/canary_orchestrator.py",
            "cluster/aws/p5/canary_runtime.py",
            "cluster/aws/p5/interruption_checkpoint.py",
            "cluster/aws/p5/profile.py",
            "msctl/aws_identity.py",
            "msctl/jsonutil.py",
        }
        if (
            release.provider != self.profile.provider
            or not required_release_members <= set(release.members)
        ):
            raise MsctlError(
                "CANARY_PLAN_INVALID",
                "release does not contain the closed executable canary",
            )
        try:
            validate_runtime_attested_contract(
                release.metadata.get("environment"),
                profile_sha256=self.profile.sha256,
            )
        except MsctlError as error:
            raise MsctlError(
                "CANARY_PLAN_INVALID",
                "release does not bind the runtime-attested environment contract",
            ) from error
        return plan_canary(
            out=out,
            apply=apply,
            profile=self.profile,
            runtime=self.runtime,
            release=release,
            selection=selection,
            environment_receipt=environment_receipt,
            instance_id=instance_id,
        )

    def _canary_instance_argv(self, instance_id: str) -> list[str]:
        if _INSTANCE_ID_RE.fullmatch(instance_id) is None:
            raise MsctlError(
                "INSTANCE_BINDING_MISMATCH",
                "canary requires one valid explicit EC2 instance ID",
            )
        return self._aws_argv(
            "ec2",
            "describe-instances",
            "--instance-ids",
            instance_id,
            query=(
                "{instances:Reservations[].Instances[]."
                "{instance_id:InstanceId,instance_type:InstanceType,"
                "state:State.Name,ami_id:ImageId,"
                "instance_profile_arn:IamInstanceProfile.Arn}}"
            ),
        )

    def _require_canary_instance(self, instance_id: str) -> None:
        output = _aws_output_object(
            self._run(
                self._canary_instance_argv(instance_id),
                operation="validate canary instance",
            ),
            {"instances"},
            label="canary EC2 instance output",
        )
        rows = _aws_output_list(
            output["instances"],
            label="canary EC2 instances",
        )
        if len(rows) != 1:
            raise MsctlError(
                "INSTANCE_BINDING_MISMATCH",
                "explicit canary instance did not resolve exactly once",
            )
        row = _aws_output_object(
            rows[0],
            {
                "instance_id",
                "instance_type",
                "state",
                "ami_id",
                "instance_profile_arn",
            },
            label="canary EC2 instance",
        )
        if row != {
            "instance_id": instance_id,
            "instance_type": self.profile.instance_type,
            "state": "running",
            "ami_id": self.runtime.ami_id,
            "instance_profile_arn": self.instance_profile_arn,
        }:
            raise MsctlError(
                "INSTANCE_BINDING_MISMATCH",
                "explicit canary instance has the wrong immutable runtime",
            )

    def _publish_canary_plan(
        self,
        plan: Mapping[str, object],
        *,
        plan_sha256: str,
        plan_uri: str,
    ) -> dict[str, object]:
        digest = require_sha256(plan_sha256, label="canary plan")
        payload = canonical_json(plan) + b"\n"
        expected_uri = canary_plan_uri(self.runtime.s3_root, digest)
        if (
            hashlib.sha256(payload).hexdigest() != digest
            or plan_uri != expected_uri
        ):
            raise MsctlError(
                "CANARY_PLAN_INVALID",
                "canary plan bytes or content address changed before publication",
            )
        relative = plan_uri.removeprefix(self.runtime.s3_root.rstrip("/") + "/")
        bucket, key = self._s3_location(relative)
        directory_fd = open_directory(
            self.state_root,
            label="AWS state root",
            create=True,
        )
        name = f"canary-plan-{digest}.json"
        try:
            atomic_write_at(
                directory_fd,
                name,
                payload,
                label="canary plan",
            )
        finally:
            os.close(directory_fd)
        checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
        kms_key_id = getattr(self.runtime, "kms_key_id", None)
        if not isinstance(kms_key_id, str) or not kms_key_id:
            raise MsctlError(
                "CANARY_PLAN_INVALID",
                "canary publication requires the pinned runtime KMS key",
            )
        metadata = (
            f"plan-type={ORCHESTRATION_PLAN_TYPE},sha256={digest}"
        )
        put = self._aws_argv(
            "s3api",
            "put-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--body",
            str((self.state_root / name).absolute()),
            "--checksum-algorithm",
            "SHA256",
            "--checksum-sha256",
            checksum,
            "--metadata",
            metadata,
            "--if-none-match",
            "*",
            "--server-side-encryption",
            "aws:kms",
            "--ssekms-key-id",
            kms_key_id,
            query=(
                "{object:{checksum_sha256:ChecksumSHA256,"
                "version_id:VersionId}}"
            ),
        )
        try:
            put_output = _aws_output_object(
                self._run(put, operation="publish canary plan"),
                {"object"},
                label="S3 canary plan put",
            )
        except MsctlError:
            put_output = None
        if put_output is not None:
            put_row = _aws_output_object(
                put_output["object"],
                {"checksum_sha256", "version_id"},
                label="S3 canary plan put",
            )
            if (
                put_row["checksum_sha256"] != checksum
                or not isinstance(put_row["version_id"], str)
                or not put_row["version_id"]
            ):
                raise MsctlError(
                    "S3_OBJECT_MISMATCH",
                    "S3 did not confirm the immutable canary plan",
                )
        head = self._aws_argv(
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--checksum-mode",
            "ENABLED",
            query=(
                "{object:{checksum_sha256:ChecksumSHA256,"
                "content_length:ContentLength,metadata:Metadata,"
                "server_side_encryption:ServerSideEncryption,"
                "sse_kms_key_id:SSEKMSKeyId,version_id:VersionId}}"
            ),
        )
        head_output = _aws_output_object(
            self._run(head, operation="verify canary plan"),
            {"object"},
            label="S3 canary plan head",
        )
        row = _aws_output_object(
            head_output["object"],
            {
                "checksum_sha256",
                "content_length",
                "metadata",
                "server_side_encryption",
                "sse_kms_key_id",
                "version_id",
            },
            label="S3 canary plan",
        )
        if (
            row["checksum_sha256"] != checksum
            or row["content_length"] != len(payload)
            or row["metadata"]
            != {
                "plan-type": ORCHESTRATION_PLAN_TYPE,
                "sha256": digest,
            }
            or row["server_side_encryption"] != "aws:kms"
            or row["sse_kms_key_id"] != kms_key_id
            or not isinstance(row["version_id"], str)
            or not row["version_id"]
        ):
            raise MsctlError(
                "S3_OBJECT_MISMATCH",
                "published canary plan bytes, metadata, or encryption differ",
            )
        return {
            "plan_sha256": digest,
            "plan_uri": plan_uri,
            "version_id": row["version_id"],
        }

    def canary_run(
        self,
        *,
        plan_path: Path | str,
        instance_id: str,
        approval_path: Path | str | None = None,
        apply: bool,
    ) -> dict[str, object]:
        """Plan or send one exact canary intent to one explicit instance."""

        if self.profile.provider not in _V3_PROFILES:
            raise MsctlError(
                "CANARY_RUN_INVALID",
                "executable qualification is available only to AWS v3 profiles",
            )
        plan, plan_sha256 = load_canary_plan(
            plan_path,
            profile=self.profile,
            runtime=self.runtime,
            expected_instance_id=instance_id,
        )
        plan_uri = canary_plan_uri(self.runtime.s3_root, plan_sha256)
        intent = build_canary_intent(
            plan=plan,
            plan_sha256=plan_sha256,
            plan_uri=plan_uri,
            runtime=self.runtime,
            control_bundle_sha256=self.control_bundle.sha256,
        )
        intent_sha256 = hashlib.sha256(canonical_json(intent)).hexdigest()
        approval_resources = aws_resource_request(
            "canary",
            profile=self.profile,
            bindings={
                "ami_id": self.runtime.ami_id,
                "container_image": self.runtime.container_image,
                "container_digest": self.runtime.container_digest,
                "instance_id": instance_id,
                "instance_type": self.profile.instance_type,
                "profile_sha256": self.profile.sha256,
                "provider": self.profile.provider,
                "release_sha256": plan["release_sha256"],
                "runtime_sha256": self._runtime_sha256(),
                "provider_selection_sha256": plan[
                    "provider_selection"
                ]["provider_selection_sha256"],
                "environment_receipt_sha256": plan[
                    "environment_receipt_sha256"
                ],
                "command_plan_sha256": plan["command_plan_sha256"],
                "orchestration_plan_sha256": plan_sha256,
                "control_bundle_sha256": self.control_bundle.sha256,
            },
        )
        result = {
            "schema_version": 3,
            "operation": "canary",
            "instance_id": instance_id,
            "plan_sha256": plan_sha256,
            "plan_uri": plan_uri,
            "intent": intent,
            "intent_sha256": intent_sha256,
            "approval_resources": approval_resources,
            "submitted": 0,
            "idempotent": False,
        }
        if not apply:
            return result

        if approval_path is None:
            raise MsctlError(
                "APPROVAL_REQUIRED",
                "canary run --apply requires an explicit signed approval receipt",
            )
        self.approval_verifier(
            path=approval_path,
            operation="canary",
            release_sha256=str(plan["release_sha256"]),
            scope_sha256=plan_sha256,
            resources=approval_resources,
            profile=self.profile,
            environ=self.environ,
        )
        self._require_canary_instance(instance_id)
        self._require_ssm_online(instance_id)
        self._ensure_argv_document()
        control = self._publish_control_bundle()
        published_plan = self._publish_canary_plan(
            plan,
            plan_sha256=plan_sha256,
            plan_uri=plan_uri,
        )
        published_intent = self._publish_operation_intent(intent)
        existing = self._find_operation_command(
            operation_id=str(intent["operation_id"]),
            instance_id=instance_id,
        )
        if existing is not None:
            return {
                **result,
                "command_id": existing["command_id"],
                "status": existing["status"],
                "control_bundle_version_id": control["version_id"],
                "plan_version_id": published_plan["version_id"],
                "intent_version_id": published_intent["version_id"],
                "idempotent": True,
            }
        command_id = self._send_operation_intent(
            instance_id=instance_id,
            intent=intent,
            published=published_intent,
            operation="run canary",
        )
        return {
            **result,
            "command_id": command_id,
            "status": "Pending",
            "control_bundle_version_id": control["version_id"],
            "plan_version_id": published_plan["version_id"],
            "intent_version_id": published_intent["version_id"],
            "submitted": 1,
        }

    def _ensure_argv_document(self) -> None:
        listing_argv = self._aws_argv(
            "ssm",
            "list-documents",
            "--filters",
            f"Key=Name,Values={ARGV_DOCUMENT_NAME}",
            query=(
                "{documents:DocumentIdentifiers[]."
                "{name:Name,hash:Hash,status:Status}}"
            ),
        )
        output = _aws_output_object(
            self._run(listing_argv, operation="verify SSM document"),
            {"documents"},
            label="SSM document listing",
        )
        rows = _aws_output_list(
            output["documents"],
            label="SSM documents",
        )
        if len(rows) > 1:
            raise MsctlError(
                "SSM_DOCUMENT_MISMATCH",
                "SSM returned duplicate argv documents",
            )
        if rows:
            row = _aws_output_object(
                rows[0],
                {"name", "hash", "status"},
                label="SSM argv document",
            )
        else:
            create_argv = self._aws_argv(
                "ssm",
                "create-document",
                "--name",
                ARGV_DOCUMENT_NAME,
                "--document-type",
                "Command",
                "--document-format",
                "JSON",
                "--content",
                ARGV_DOCUMENT_CONTENT,
                query=(
                    "{document:{name:DocumentDescription.Name,"
                    "hash:DocumentDescription.Hash,"
                    "status:DocumentDescription.Status}}"
                ),
            )
            created = _aws_output_object(
                self._run(create_argv, operation="provision SSM document"),
                {"document"},
                label="SSM create-document output",
            )
            row = _aws_output_object(
                created["document"],
                {"name", "hash", "status"},
                label="created SSM argv document",
            )
        if row != {
            "name": ARGV_DOCUMENT_NAME,
            "hash": ARGV_DOCUMENT_SHA256,
            "status": "Active",
        }:
            raise MsctlError(
                "SSM_DOCUMENT_MISMATCH",
                "SSM argv document does not match the fixed reviewed hash",
            )

    def _discover_argv(
        self,
        manifest: object,
        *,
        instance_id: str | None = None,
    ) -> list[str]:
        arguments: list[str] = []
        if instance_id is not None:
            if _INSTANCE_ID_RE.fullmatch(instance_id) is None:
                raise MsctlError(
                    "INSTANCE_BINDING_MISMATCH",
                    "EC2 instance ID is invalid",
                )
            arguments.extend(["--instance-ids", instance_id])
        else:
            arguments.extend(
                [
                    "--filters",
                    (
                        "Name=tag:MemorySplitProvider,"
                        f"Values={self.profile.provider}"
                    ),
                    f"Name=tag:MemorySplitSeed,Values={manifest.seed}",
                    "Name=instance-state-name,Values=pending,running,stopping",
                ]
            )
        v3_query = (
            "preregistration_sha256:"
            "Tags[?Key=='MemorySplitPreregistrationSHA256']|[0].Value,"
            "hardware_amendment_sha256:"
            "Tags[?Key=='MemorySplitHardwareAmendmentSHA256']|[0].Value,"
            "provider_selection_sha256:"
            "Tags[?Key=='MemorySplitProviderSelectionSHA256']|[0].Value,"
            "sealed_fixture_sha256:"
            "Tags[?Key=='MemorySplitSealedFixtureSHA256']|[0].Value,"
            "fleet_plan_sha256:"
            "Tags[?Key=='MemorySplitFleetPlanSHA256']|[0].Value,"
            "fleet_wave:to_number(Tags[?Key=='MemorySplitFleetWave']|[0].Value),"
            "launch_readiness_sha256:"
            "Tags[?Key=='MemorySplitLaunchReadinessSHA256']|[0].Value,"
            "control_bundle_sha256:"
            "Tags[?Key=='MemorySplitControlBundleSHA256']|[0].Value,"
            if _is_v3_manifest(manifest)
            else ""
        )
        query = (
            "{instances:Reservations[].Instances[]."
            "{instance_id:InstanceId,instance_type:InstanceType,"
            "state:State.Name,instance_profile_arn:IamInstanceProfile.Arn,"
            "provider:Tags[?Key=='MemorySplitProvider']|[0].Value,"
            "profile_instance_type:"
            "Tags[?Key=='MemorySplitInstanceType']|[0].Value,"
            "seed:to_number(Tags[?Key=='MemorySplitSeed']|[0].Value),"
            "cohort_sha256:Tags[?Key=='MemorySplitCohortSHA256']|[0].Value,"
            "release_sha256:Tags[?Key=='MemorySplitReleaseSHA256']|[0].Value,"
            "dataset_sha256:Tags[?Key=='MemorySplitDatasetSHA256']|[0].Value,"
            "run_manifest_sha256:"
            "Tags[?Key=='MemorySplitRunManifestSHA256']|[0].Value,"
            "profile_sha256:Tags[?Key=='MemorySplitProfileSHA256']|[0].Value,"
            f"{v3_query}"
            "gres:Tags[?Key=='MemorySplitGRES']|[0].Value}}"
        )
        return self._aws_argv(
            "ec2",
            "describe-instances",
            *arguments,
            query=query,
        )

    def _selected_instance_argv(
        self,
        instance_id: str,
        manifest: object | None = None,
    ) -> list[str]:
        if _INSTANCE_ID_RE.fullmatch(instance_id) is None:
            raise MsctlError(
                "INSTANCE_BINDING_MISMATCH",
                "EC2 instance ID is invalid",
            )
        v3_query = (
            "preregistration_sha256:"
            "Tags[?Key=='MemorySplitPreregistrationSHA256']|[0].Value,"
            "hardware_amendment_sha256:"
            "Tags[?Key=='MemorySplitHardwareAmendmentSHA256']|[0].Value,"
            "provider_selection_sha256:"
            "Tags[?Key=='MemorySplitProviderSelectionSHA256']|[0].Value,"
            "sealed_fixture_sha256:"
            "Tags[?Key=='MemorySplitSealedFixtureSHA256']|[0].Value,"
            "fleet_plan_sha256:"
            "Tags[?Key=='MemorySplitFleetPlanSHA256']|[0].Value,"
            "fleet_wave:to_number(Tags[?Key=='MemorySplitFleetWave']|[0].Value),"
            "launch_readiness_sha256:"
            "Tags[?Key=='MemorySplitLaunchReadinessSHA256']|[0].Value,"
            "control_bundle_sha256:"
            "Tags[?Key=='MemorySplitControlBundleSHA256']|[0].Value,"
            if manifest is not None and _is_v3_manifest(manifest)
            else ""
        )
        query = (
            "{instances:Reservations[].Instances[]."
            "{instance_id:InstanceId,instance_type:InstanceType,"
            "state:State.Name,instance_profile_arn:IamInstanceProfile.Arn,"
            "ami_id:ImageId,"
            "provider:Tags[?Key=='MemorySplitProvider']|[0].Value,"
            "profile_instance_type:"
            "Tags[?Key=='MemorySplitInstanceType']|[0].Value,"
            "seed:to_number(Tags[?Key=='MemorySplitSeed']|[0].Value),"
            "cohort_sha256:Tags[?Key=='MemorySplitCohortSHA256']|[0].Value,"
            "release_sha256:Tags[?Key=='MemorySplitReleaseSHA256']|[0].Value,"
            "dataset_sha256:Tags[?Key=='MemorySplitDatasetSHA256']|[0].Value,"
            "run_manifest_sha256:"
            "Tags[?Key=='MemorySplitRunManifestSHA256']|[0].Value,"
            "profile_sha256:Tags[?Key=='MemorySplitProfileSHA256']|[0].Value,"
            f"{v3_query}"
            "runtime_sha256:Tags[?Key=='MemorySplitRuntimeSHA256']|[0].Value,"
            "container_digest:"
            "Tags[?Key=='MemorySplitContainerDigest']|[0].Value,"
            "gres:Tags[?Key=='MemorySplitGRES']|[0].Value,"
            "terminate_at:Tags[?Key=='MemorySplitTerminateAt']|[0].Value}}"
        )
        return self._aws_argv(
            "ec2",
            "describe-instances",
            "--instance-ids",
            instance_id,
            query=query,
        )

    def _selected_binding(
        self,
        manifest: object,
        *,
        terminate_at: str,
        context: V3LifecycleContext | None = None,
    ) -> dict[str, object]:
        v3_bindings = self._v3_bindings(manifest, context)
        # The EC2 tag contract uses cohort_sha256 for the manifest's
        # cohort-assignment identity; do not introduce a second unqueryable
        # cohort_assignment_sha256 tag field.
        v3_tag_bindings = {
            field: value
            for field, value in v3_bindings.items()
            if field != "cohort_assignment_sha256"
        }
        return {
            "provider": self.profile.provider,
            "profile_instance_type": self.profile.instance_type,
            "seed": manifest.seed,
            "cohort_sha256": manifest.cohort_assignment_sha256,
            "release_sha256": manifest.release_sha256,
            "dataset_sha256": manifest.dataset_sha256,
            "run_manifest_sha256": manifest.sha256,
            "profile_sha256": self.profile.sha256,
            "runtime_sha256": self._runtime_sha256(),
            "container_digest": self.runtime.container_digest,
            "gres": _profile_gres(self.profile),
            "terminate_at": terminate_at,
            **v3_tag_bindings,
        }

    def _parse_selected_instance(
        self,
        output: object,
        manifest: object,
        *,
        instance_id: str,
        terminate_at: str,
        require_bound: bool,
        context: V3LifecycleContext | None = None,
        expected_binding: Mapping[str, object] | None = None,
        allow_historical_binding: bool = False,
    ) -> dict[str, object]:
        root = _aws_output_object(
            output,
            {"instances"},
            label="selected EC2 instance output",
        )
        rows = _aws_output_list(
            root["instances"],
            label="selected EC2 instances",
        )
        if len(rows) != 1:
            raise MsctlError(
                "INSTANCE_BINDING_MISMATCH",
                "operator selection must resolve to exactly one instance",
            )
        row = _aws_output_object(
            rows[0],
            (
                _SELECTED_INSTANCE_FIELDS | _V3_INSTANCE_FIELDS
                if _is_v3_manifest(manifest)
                else _SELECTED_INSTANCE_FIELDS
            ),
            label="selected EC2 instance",
        )
        if (
            row["instance_id"] != instance_id
            or row["instance_type"] != self.profile.instance_type
            or row["state"] != "running"
            or row["instance_profile_arn"] != self.instance_profile_arn
            or row["ami_id"] != self.runtime.ami_id
        ):
            raise MsctlError(
                "INSTANCE_BINDING_MISMATCH",
                "operator-selected instance has the wrong immutable runtime",
            )
        expected = (
            dict(expected_binding)
            if expected_binding is not None
            else self._selected_binding(
                manifest,
                terminate_at=terminate_at,
                context=context,
            )
        )
        expected_fields = (
            self._instance_tag_names(manifest).keys()
            if _is_v3_manifest(manifest)
            else self._instance_tag_names(manifest).keys()
        )
        if set(expected) != set(expected_fields):
            raise MsctlError(
                "INSTANCE_BINDING_MISMATCH",
                "selected-instance expected tag binding is incomplete",
            )
        observed = {field: row[field] for field in expected}
        if allow_historical_binding:
            valid = True
        elif require_bound:
            valid = observed == expected
        else:
            valid = observed == expected or all(
                value is None for value in observed.values()
            )
        if not valid:
            raise MsctlError(
                "INSTANCE_BINDING_MISMATCH",
                "operator-selected instance has conflicting provenance tags",
            )
        return row

    def _instance_tags(
        self,
        manifest: object,
        *,
        terminate_at: str,
        context: V3LifecycleContext | None = None,
    ) -> str:
        binding = self._selected_binding(
            manifest,
            terminate_at=terminate_at,
            context=context,
        )
        names = self._instance_tag_names(manifest)
        return canonical_json(
            [
                {"Key": names[field], "Value": str(binding[field])}
                for field in names
            ]
        ).decode("ascii")

    def _instance_tag_names(self, manifest: object) -> dict[str, str]:
        names = {
            "provider": "MemorySplitProvider",
            "profile_instance_type": "MemorySplitInstanceType",
            "seed": "MemorySplitSeed",
            "cohort_sha256": "MemorySplitCohortSHA256",
            "release_sha256": "MemorySplitReleaseSHA256",
            "dataset_sha256": "MemorySplitDatasetSHA256",
            "run_manifest_sha256": "MemorySplitRunManifestSHA256",
            "profile_sha256": "MemorySplitProfileSHA256",
            "runtime_sha256": "MemorySplitRuntimeSHA256",
            "container_digest": "MemorySplitContainerDigest",
            "gres": "MemorySplitGRES",
            "terminate_at": "MemorySplitTerminateAt",
        }
        if _is_v3_manifest(manifest):
            names.update(
                {
                    "preregistration_sha256": (
                        "MemorySplitPreregistrationSHA256"
                    ),
                    "hardware_amendment_sha256": (
                        "MemorySplitHardwareAmendmentSHA256"
                    ),
                    "provider_selection_sha256": (
                        "MemorySplitProviderSelectionSHA256"
                    ),
                    "sealed_fixture_sha256": (
                        "MemorySplitSealedFixtureSHA256"
                    ),
                    "fleet_plan_sha256": "MemorySplitFleetPlanSHA256",
                    "fleet_wave": "MemorySplitFleetWave",
                    "launch_readiness_sha256": (
                        "MemorySplitLaunchReadinessSHA256"
                    ),
                    "control_bundle_sha256": (
                        "MemorySplitControlBundleSHA256"
                    ),
                }
            )
        return names

    def _bind_selected_instance(
        self,
        manifest: object,
        *,
        instance_id: str,
        terminate_at: str,
        context: V3LifecycleContext | None = None,
    ) -> dict[str, object]:
        selected_argv = self._selected_instance_argv(instance_id, manifest)
        self._parse_selected_instance(
            self._run(selected_argv, operation="validate selected instance"),
            manifest,
            instance_id=instance_id,
            terminate_at=terminate_at,
            require_bound=False,
            context=context,
        )
        modify = self._aws_argv(
            "ec2",
            "modify-instance-attribute",
            "--instance-id",
            instance_id,
            "--instance-initiated-shutdown-behavior",
            "Value=terminate",
            query="{}",
        )
        _aws_output_object(
            self._run(modify, operation="bind termination behavior"),
            set(),
            label="EC2 modify-instance-attribute output",
        )
        tag = self._aws_argv(
            "ec2",
            "create-tags",
            "--resources",
            instance_id,
            "--tags",
            self._instance_tags(
                manifest,
                terminate_at=terminate_at,
                context=context,
            ),
            query="{}",
        )
        _aws_output_object(
            self._run(tag, operation="bind selected instance"),
            set(),
            label="EC2 create-tags output",
        )
        row = self._parse_selected_instance(
            self._run(selected_argv, operation="verify selected instance"),
            manifest,
            instance_id=instance_id,
            terminate_at=terminate_at,
            require_bound=True,
            context=context,
        )
        attribute = self._aws_argv(
            "ec2",
            "describe-instance-attribute",
            "--instance-id",
            instance_id,
            "--attribute",
            "instanceInitiatedShutdownBehavior",
            query=(
                "{attribute:{instance_id:InstanceId,"
                "shutdown_behavior:InstanceInitiatedShutdownBehavior.Value}}"
            ),
        )
        output = _aws_output_object(
            self._run(attribute, operation="verify termination behavior"),
            {"attribute"},
            label="EC2 instance attribute output",
        )
        exact = _aws_output_object(
            output["attribute"],
            {"instance_id", "shutdown_behavior"},
            label="EC2 shutdown behavior",
        )
        if exact != {
            "instance_id": instance_id,
            "shutdown_behavior": "terminate",
        }:
            raise MsctlError(
                "INSTANCE_BINDING_MISMATCH",
                "instance auto-termination behavior is not enforceable",
            )
        return {**row, "terminate_at": terminate_at}

    def _validate_selected_instance_binding(
        self,
        manifest: object,
        *,
        instance_id: str,
        terminate_at: str,
        operation: str,
        context: V3LifecycleContext | None = None,
        allow_historical_binding: bool = False,
    ) -> dict[str, object]:
        selected_argv = self._selected_instance_argv(instance_id, manifest)
        row = self._parse_selected_instance(
            self._run(
                selected_argv,
                operation=f"verify {operation} instance",
            ),
            manifest,
            instance_id=instance_id,
            terminate_at=terminate_at,
            require_bound=True,
            context=context,
            allow_historical_binding=allow_historical_binding,
        )
        attribute = self._aws_argv(
            "ec2",
            "describe-instance-attribute",
            "--instance-id",
            instance_id,
            "--attribute",
            "instanceInitiatedShutdownBehavior",
            query=(
                "{attribute:{instance_id:InstanceId,"
                "shutdown_behavior:InstanceInitiatedShutdownBehavior.Value}}"
            ),
        )
        output = _aws_output_object(
            self._run(
                attribute,
                operation=f"verify {operation} termination behavior",
            ),
            {"attribute"},
            label="EC2 instance attribute output",
        )
        exact = _aws_output_object(
            output["attribute"],
            {"instance_id", "shutdown_behavior"},
            label="EC2 shutdown behavior",
        )
        if exact != {
            "instance_id": instance_id,
            "shutdown_behavior": "terminate",
        }:
            raise MsctlError(
                "INSTANCE_BINDING_MISMATCH",
                "selected instance no longer has enforced termination behavior",
            )
        return row

    def _validate_evaluation_instance_binding(
        self,
        manifest: object,
        *,
        instance_id: str,
        terminate_at: str,
        checkpoint_receipt_sha256: str,
        context: V3LifecycleContext | None,
    ) -> dict[str, object]:
        validated = self._validate_v3_context(manifest, context)
        if validated is None:
            return self._validate_selected_instance_binding(
                manifest,
                instance_id=instance_id,
                terminate_at=terminate_at,
                operation="evaluation",
                context=context,
            )
        binding = validated.fleet_binding
        successors = [
            candidate
            for candidate in validated.fleet_plan.manifests
            if candidate.instance_id == binding.instance_id
            and candidate.wave == binding.wave + 1
        ]
        if len(successors) > 1:
            raise MsctlError(
                "FLEET_PLAN_INVALID",
                "fleet plan has duplicate same-instance successors",
            )
        advance = (
            load_fleet_advance(
                self.state_root,
                plan=validated.fleet_plan,
                to_binding=successors[0],
            )
            if successors
            else None
        )
        if successors and (
            advance is None
            or advance.from_binding != binding
            or advance.evidence.get("checkpoint_receipt_sha256")
            != checkpoint_receipt_sha256
        ):
            raise MsctlError(
                "FLEET_ADVANCE_REQUIRED",
                "evaluation after training-tag removal requires the matching "
                "closed training-wave advance receipt",
                details={
                    "instance_id": binding.instance_id,
                    "seed": binding.seed,
                    "wave": binding.wave,
                },
            )
        return self._validate_selected_instance_binding(
            manifest,
            instance_id=instance_id,
            terminate_at=terminate_at,
            operation="evaluation",
            context=context,
            allow_historical_binding=advance is not None,
        )

    def _parse_instances(
        self,
        output: object,
        manifest: object,
        context: V3LifecycleContext | None = None,
    ) -> list[dict[str, object]]:
        root = _aws_output_object(
            output,
            {"instances"},
            label="EC2 describe-instances output",
        )
        instances: list[dict[str, object]] = []
        for index, raw in enumerate(
            _aws_output_list(root["instances"], label="EC2 instances")
        ):
            row = _aws_output_object(
                raw,
                (
                    _INSTANCE_FIELDS | _V3_INSTANCE_FIELDS
                    if _is_v3_manifest(manifest)
                    else _INSTANCE_FIELDS
                ),
                label=f"EC2 instance[{index}]",
            )
            if (
                not isinstance(row["instance_id"], str)
                or _INSTANCE_ID_RE.fullmatch(row["instance_id"]) is None
                or row["instance_type"] != self.profile.instance_type
                or row["profile_instance_type"] != self.profile.instance_type
                or row["state"] not in _ACTIVE_INSTANCE_STATES
                or row["instance_profile_arn"] != self.instance_profile_arn
                or row["provider"] != self.profile.provider
                or row["seed"] != manifest.seed
                or row["cohort_sha256"]
                != manifest.cohort_assignment_sha256
                or row["release_sha256"] != manifest.release_sha256
                or row["dataset_sha256"] != manifest.dataset_sha256
                or row["run_manifest_sha256"] != manifest.sha256
                or row["profile_sha256"] != self.profile.sha256
                or row["gres"] != _profile_gres(self.profile)
                or (
                    _is_v3_manifest(manifest)
                    and any(
                        row[field] != expected
                        for field, expected in self._v3_bindings(
                            manifest,
                            context,
                        ).items()
                    )
                )
            ):
                raise MsctlError(
                    "INSTANCE_BINDING_MISMATCH",
                    "EC2 instance does not match the exact run provenance",
                    details={"index": index},
                )
            instances.append(row)
        if len(instances) > 1:
            raise MsctlError(
                "DUPLICATE_ACTIVE_SEED",
                "multiple active AWS GPU instances claim the same seed",
                details={
                    "seed": manifest.seed,
                    "instance_ids": sorted(
                        str(row["instance_id"]) for row in instances
                    ),
                },
            )
        return instances

    def discover_instances(
        self,
        manifest: object,
        context: V3LifecycleContext | None = None,
    ) -> list[dict[str, object]]:
        self._validate_manifest(manifest)
        self._validate_v3_context(manifest, context)
        return self._parse_instances(
            self._run(
                self._discover_argv(manifest),
                operation="instance discovery",
            ),
            manifest,
            context,
        )

    def _validate_exact_instance(
        self,
        manifest: object,
        instance_id: str,
        context: V3LifecycleContext | None = None,
    ) -> dict[str, object]:
        rows = self._parse_instances(
            self._run(
                self._discover_argv(manifest, instance_id=instance_id),
                operation="instance validation",
            ),
            manifest,
            context,
        )
        if len(rows) != 1:
            raise MsctlError(
                "INSTANCE_BINDING_MISMATCH",
                "launched instance cannot be proven from EC2",
            )
        return rows[0]

    def _require_ssm_online(self, instance_id: str) -> None:
        if _INSTANCE_ID_RE.fullmatch(instance_id) is None:
            raise MsctlError(
                "INSTANCE_BINDING_MISMATCH",
                "SSM instance ID is invalid",
            )
        argv = self._aws_argv(
            "ssm",
            "describe-instance-information",
            "--filters",
            f"Key=InstanceIds,Values={instance_id}",
            query=(
                "{managed_instances:InstanceInformationList[]."
                "{instance_id:InstanceId,ping_status:PingStatus}}"
            ),
        )
        output = _aws_output_object(
            self._run(argv, operation="SSM readiness"),
            {"managed_instances"},
            label="SSM instance information",
        )
        rows = _aws_output_list(
            output["managed_instances"],
            label="SSM managed instances",
        )
        if len(rows) != 1:
            raise MsctlError(
                "SSM_UNAVAILABLE",
                "exactly one managed AWS GPU instance must be online",
            )
        row = _aws_output_object(
            rows[0],
            {"instance_id", "ping_status"},
            label="SSM managed instance",
        )
        if row != {"instance_id": instance_id, "ping_status": "Online"}:
            raise MsctlError(
                "SSM_UNAVAILABLE",
                "AWS GPU instance is not online in Systems Manager",
            )

    def _command_status(
        self,
        instance_id: str,
        command_id: str,
    ) -> str:
        if (
            _INSTANCE_ID_RE.fullmatch(instance_id) is None
            or _COMMAND_ID_RE.fullmatch(command_id) is None
        ):
            raise MsctlError(
                "STATE_CORRUPT",
                "SSM command binding contains an invalid identifier",
            )
        argv = self._aws_argv(
            "ssm",
            "get-command-invocation",
            "--instance-id",
            instance_id,
            "--command-id",
            command_id,
            query="{command:{command_id:CommandId,status:Status}}",
        )
        output = _aws_output_object(
            self._run(argv, operation="status"),
            {"command"},
            label="SSM command status output",
        )
        command = _aws_output_object(
            output["command"],
            {"command_id", "status"},
            label="SSM command status",
        )
        if (
            command["command_id"] != command_id
            or not isinstance(command["status"], str)
            or not command["status"]
        ):
            raise MsctlError(
                "AWS_OUTPUT_INVALID",
                "SSM command status does not bind the requested command",
            )
        return str(command["status"])

    def _verify_approval(
        self,
        path: Path | str | None,
        *,
        operation: str,
        release: object,
        manifest: object,
        resources: dict[str, object] | None = None,
    ) -> None:
        if path is None:
            raise MsctlError(
                "APPROVAL_REQUIRED",
                f"{operation} --apply requires explicit signed approval",
            )
        self.approval_verifier(
            path=path,
            operation=operation,
            release_sha256=release.archive_sha256,
            scope_sha256=manifest.sha256,
            resources=resources
            or aws_resource_request(operation, profile=self.profile),
            profile=self.profile,
            environ=self.environ,
        )

    def _same_state_binding(
        self,
        state: dict[str, object],
        manifest: object,
        context: V3LifecycleContext | None = None,
    ) -> bool:
        by_id = {run.run_id: run for run in manifest.runs}
        run = by_id.get(state.get("run_id"))
        instance_id = state.get("instance_id")
        command_id = state.get("command_id")
        return (
            run is not None
            and state.get("provider") == self.profile.provider
            and state.get("instance_type") == self.profile.instance_type
            and state.get("gres") == _profile_gres(self.profile)
            and state.get("seed") == manifest.seed
            and state.get("arm") == run.arm
            and state.get("config_sha256") == run.config_sha256
            and state.get("release_sha256") == manifest.release_sha256
            and state.get("dataset_sha256") == manifest.dataset_sha256
            and state.get("run_manifest_sha256") == manifest.sha256
            and state.get("cohort_assignment_sha256")
            == manifest.cohort_assignment_sha256
            and state.get("source_commit") == manifest.source_commit
            and state.get("profile_sha256") == self.profile.sha256
            and state.get("runtime_sha256") == self._runtime_sha256()
            and state.get("ami_id") == self.runtime.ami_id
            and state.get("container_digest") == self.runtime.container_digest
            and isinstance(instance_id, str)
            and _INSTANCE_ID_RE.fullmatch(instance_id) is not None
            and (
                command_id is None
                or (
                    isinstance(command_id, str)
                    and _COMMAND_ID_RE.fullmatch(command_id) is not None
                )
            )
            and (
                (
                    all(
                        state.get(field) == expected
                        for field, expected in self._v3_bindings(
                            manifest,
                            context,
                        ).items()
                    )
                )
                if _is_v3_manifest(manifest)
                else state.get("study_lock_sha256") == manifest.study_lock_sha256
            )
        )

    def _require_state_intent_binding(
        self,
        states: Sequence[Mapping[str, object]],
        intent: Mapping[str, object],
    ) -> None:
        digest = hashlib.sha256(canonical_json(intent)).hexdigest()
        expected = {
            "operation_id": intent["operation_id"],
            "intent_sha256": digest,
            "intent_uri": (
                f"{self.runtime.s3_root}/operations/intents/"
                f"sha256/{digest}.json"
            ),
        }
        if any(
            any(state.get(field) != value for field, value in expected.items())
            for state in states
        ):
            raise MsctlError(
                "STATE_CORRUPT",
                "paired state does not bind the deterministic operation intent",
            )

    def _submission_plan(
        self,
        manifest: object,
        *,
        release: object,
        instance_id: str,
        terminate_at: str,
        evidence: Mapping[str, str] | None = None,
        context: V3LifecycleContext | None = None,
    ) -> dict[str, object]:
        operation_intent = self._training_operation_intent(
            operation="submit",
            release=release,
            manifest=manifest,
            terminate_at=terminate_at,
            evidence=evidence,
            context=context,
        )
        return {
            "provider": self.profile.provider,
            "instance_type": self.profile.instance_type,
            "profile_sha256": self.profile.sha256,
            "gres": _profile_gres(self.profile),
            "seed": manifest.seed,
            "release_sha256": manifest.release_sha256,
            "run_manifest_sha256": manifest.sha256,
            "instance_id": instance_id,
            "terminate_at": terminate_at,
            "commands": [
                self._discover_argv(manifest),
                self._discover_argv(manifest, instance_id=instance_id),
            ],
            "operation_intent": operation_intent,
            "submitted": 0,
            "idempotent": False,
        }

    def _runtime_sha256(self) -> str:
        return require_sha256(
            hashlib.sha256(
                canonical_json(
                    {
                        "ami_id": self.runtime.ami_id,
                        "container_image": self.runtime.container_image,
                        "container_digest": self.runtime.container_digest,
                        "gid": getattr(self.runtime, "gid", None),
                        "region": self.runtime.region,
                        "s3_root": self.runtime.s3_root,
                        "uid": getattr(self.runtime, "uid", None),
                    }
                )
            ).hexdigest(),
            label="AWS runtime binding",
        )

    def _submit_resources(
        self,
        *,
        release: object,
        manifest: object,
        instance_id: str,
        terminate_at: str,
        context: V3LifecycleContext | None = None,
    ) -> dict[str, object]:
        return aws_resource_request(
            "submit",
            profile=self.profile,
            bindings=self._execution_bindings(
                release=release,
                manifest=manifest,
                instance_id=instance_id,
                terminate_at=terminate_at,
                context=context,
            ),
        )

    def _execution_bindings(
        self,
        *,
        release: object,
        manifest: object,
        instance_id: str,
        terminate_at: str,
        context: V3LifecycleContext | None = None,
    ) -> dict[str, object]:
        return {
            "ami_id": self.runtime.ami_id,
            "container_image": self.runtime.container_image,
            "container_digest": self.runtime.container_digest,
            "instance_id": instance_id,
            "instance_type": self.profile.instance_type,
            "profile_sha256": self.profile.sha256,
            "provider": self.profile.provider,
            "release_sha256": release.archive_sha256,
            "run_manifest_sha256": manifest.sha256,
            "runtime_sha256": self._runtime_sha256(),
            "seed": manifest.seed,
            "terminate_at": terminate_at,
            **self._v3_bindings(manifest, context),
        }

    def _cleanup_resources(
        self,
        *,
        release: object,
        manifest: object,
        instance_id: str,
        terminate_at: str,
        context: V3LifecycleContext | None = None,
    ) -> dict[str, object]:
        return aws_resource_request(
            "cleanup",
            profile=self.profile,
            bindings=self._execution_bindings(
                release=release,
                manifest=manifest,
                instance_id=instance_id,
                terminate_at=terminate_at,
                context=context,
            ),
        )

    def _cancel_resources(
        self,
        *,
        release: object,
        manifest: object,
        instance_id: str,
        terminate_at: str,
        context: V3LifecycleContext | None = None,
    ) -> dict[str, object]:
        return aws_resource_request(
            "cancel",
            profile=self.profile,
            bindings=self._execution_bindings(
                release=release,
                manifest=manifest,
                instance_id=instance_id,
                terminate_at=terminate_at,
                context=context,
            ),
        )

    def _evaluation_resources(
        self,
        *,
        release: object,
        manifest: object,
        instance_id: str,
        terminate_at: str,
        context: V3LifecycleContext | None = None,
        checkpoint_receipt: object | None = None,
    ) -> dict[str, object]:
        bindings = (
            self._execution_bindings(
                release=release,
                manifest=manifest,
                instance_id=instance_id,
                terminate_at=terminate_at,
                context=context,
            )
            | self._v3_evaluation_bindings(manifest, context)
        )
        if _is_v3_manifest(manifest):
            bindings["checkpoint_receipt_sha256"] = require_sha256(
                getattr(checkpoint_receipt, "sha256", None),
                label="evaluation checkpoint receipt",
            )
        return aws_resource_request(
            "evaluate",
            profile=self.profile,
            bindings=bindings,
        )

    def _resume_resources(
        self,
        *,
        release: object,
        manifest: object,
        instance_id: str,
        terminate_at: str,
        checkpoint_receipt_sha256: str,
        context: V3LifecycleContext | None = None,
    ) -> dict[str, object]:
        return aws_resource_request(
            "resume",
            profile=self.profile,
            bindings={
                **self._execution_bindings(
                    release=release,
                    manifest=manifest,
                    instance_id=instance_id,
                    terminate_at=terminate_at,
                    context=context,
                ),
                "checkpoint_receipt_sha256": checkpoint_receipt_sha256,
            },
        )

    def _validate_submit_selection(
        self,
        instance_id: str,
        terminate_at: str,
    ) -> None:
        if not isinstance(instance_id, str) or _INSTANCE_ID_RE.fullmatch(
            instance_id
        ) is None:
            raise MsctlError(
                "INSTANCE_BINDING_MISMATCH",
                "submit requires one explicit EC2 instance ID",
            )
        if not isinstance(terminate_at, str) or not terminate_at.endswith("Z"):
            raise MsctlError(
                "TERMINATION_DEADLINE_INVALID",
                "submit requires an RFC 3339 UTC termination deadline",
            )
        try:
            parsed = datetime.fromisoformat(terminate_at[:-1] + "+00:00")
        except ValueError as error:
            raise MsctlError(
                "TERMINATION_DEADLINE_INVALID",
                "submit requires an RFC 3339 UTC termination deadline",
            ) from error
        if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
            raise MsctlError(
                "TERMINATION_DEADLINE_INVALID",
                "termination deadline must be in UTC",
            )
        now = datetime.now(UTC)
        if parsed <= now:
            raise MsctlError(
                "TERMINATION_DEADLINE_INVALID",
                "termination deadline must be in the future",
            )
        if parsed > now + _MAX_PAID_RUNTIME:
            raise MsctlError(
                "TERMINATION_DEADLINE_INVALID",
                "termination deadline exceeds the approved 1440-minute horizon",
            )

    def capacity_check(self) -> dict[str, object]:
        instance_type = self.profile.instance_type
        argv = self._aws_argv(
            "ec2",
            "describe-instance-type-offerings",
            "--location-type",
            "region",
            "--filters",
            f"Name=instance-type,Values={instance_type}",
            query=(
                "{offerings:InstanceTypeOfferings[]."
                "{instance_type:InstanceType,location:Location,"
                "location_type:LocationType}}"
            ),
        )
        output = _aws_output_object(
            self._run(argv, operation="capacity check"),
            {"offerings"},
            label="EC2 instance offerings output",
        )
        offerings = []
        for index, raw in enumerate(
            _aws_output_list(output["offerings"], label="EC2 offerings")
        ):
            row = _aws_output_object(
                raw,
                {"instance_type", "location", "location_type"},
                label=f"EC2 offering[{index}]",
            )
            if (
                row["instance_type"] != instance_type
                or row["location"] != self.runtime.region
                or row["location_type"] != "region"
            ):
                raise MsctlError(
                    "AWS_OUTPUT_INVALID",
                    "EC2 offering does not match the pinned region and type",
                )
            offerings.append(row)
        if not offerings:
            raise MsctlError(
                "CAPACITY_INSUFFICIENT",
                "AWS GPU instance type is not offered in the selected region",
                details={
                    "region": self.runtime.region,
                    "instance_type": instance_type,
                },
            )
        return {
            "provider": self.profile.provider,
            "profile_sha256": self.profile.sha256,
            "gres": _profile_gres(self.profile),
            "region": self.runtime.region,
            "instance_type": instance_type,
            "offered": True,
            "offerings": offerings,
        }

    def _load_bound_inputs(
        self,
        *,
        release_path: Path | str,
        manifest_path: Path | str,
        repo_root: Path | str,
    ) -> tuple[object, object]:
        release = load_release(release_path)
        manifest = load_run_manifest(manifest_path, repo_root=repo_root)
        bind_release(release, manifest)
        self._validate_manifest(manifest)
        self._validate_release(release, manifest)
        profile_binding = release.metadata.get("profile")
        release_profile_sha256 = (
            profile_binding.get("sha256")
            if isinstance(profile_binding, dict)
            else release.metadata.get("profile_sha256")
        )
        if release_profile_sha256 != getattr(self.profile, "sha256", None):
            raise MsctlError(
                "PROFILE_RELEASE_MISMATCH",
                "local AWS profile differs from the verified release",
            )
        try:
            validate_runtime_attested_contract(
                release.metadata.get("environment"),
                profile_sha256=self.profile.sha256,
            )
        except MsctlError as error:
            raise MsctlError(
                "ENVIRONMENT_CONTRACT_INVALID",
                "AWS release does not carry the final runtime_attested contract",
            ) from error
        for run in manifest.runs:
            verify_release_member(
                release,
                member_path=run.config,
                local_path=Path(repo_root) / run.config,
                label="run config",
            )
        version = "v3" if _is_v3_manifest(manifest) else "v2"
        cohort_member = release.members.get(
            f"configs/cohort-assignment-{version}.json"
        )
        study_member = release.members.get(
            f"configs/preregistration-{version}.yaml"
        )
        if (
            cohort_member is None
            or cohort_member.get("sha256")
            != manifest.cohort_assignment_sha256
            or study_member is None
            or study_member.get("sha256")
            != (
                manifest.preregistration_sha256
                if _is_v3_manifest(manifest)
                else manifest.study_lock_sha256
            )
        ):
            raise MsctlError(
                "RELEASE_COHORT_MISMATCH",
                "AWS release does not bind the manifest cohort and study lock",
            )
        if _is_v3_manifest(manifest):
            amendment_member = release.members.get(
                "configs/hardware-amendment-v3.json"
            )
            if (
                amendment_member is None
                or amendment_member.get("sha256")
                != manifest.hardware_amendment_sha256
            ):
                raise MsctlError(
                    "RELEASE_COHORT_MISMATCH",
                    "AWS v3 training release does not bind the amendment",
                )
        return release, manifest

    def _read_lifecycle_environment_receipt(self, path: Path) -> bytes:
        maximum = 4 * 1024 * 1024
        chunks: list[bytes] = []
        try:
            before = path.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size <= 0
                or before.st_size > maximum
            ):
                raise MsctlError(
                    "ENVIRONMENT_RECEIPT_INVALID",
                    "AWS environment receipt must be one bounded singly "
                    "linked regular file",
                )
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or (
                        opened.st_dev,
                        opened.st_ino,
                        opened.st_size,
                        opened.st_mtime_ns,
                        opened.st_ctime_ns,
                    )
                    != (
                        before.st_dev,
                        before.st_ino,
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                    )
                ):
                    raise MsctlError(
                        "ENVIRONMENT_RECEIPT_INVALID",
                        "AWS environment receipt changed before it was opened",
                    )
                total = 0
                while True:
                    chunk = os.read(descriptor, min(1 << 20, maximum + 1 - total))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > maximum:
                        raise MsctlError(
                            "ENVIRONMENT_RECEIPT_INVALID",
                            "AWS environment receipt exceeds the size bound",
                        )
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
        except MsctlError:
            raise
        except OSError as error:
            raise MsctlError(
                "ENVIRONMENT_RECEIPT_INVALID",
                "AWS environment receipt cannot be read safely",
            ) from error
        data = b"".join(chunks)
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
        ) or (
            not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or len(data) != after.st_size
        ):
            raise MsctlError(
                "ENVIRONMENT_RECEIPT_INVALID",
                "AWS environment receipt changed while it was read",
            )
        return data

    def _load_lifecycle_evidence(
        self,
        *,
        manifest: object,
        dataset_pointer: Path | str,
        dataset_root: Path | str | None,
        dataset_verification: Path | str | None,
        environment_receipt: Path | str,
        selection: ProviderSelection | None = None,
        expected_instance_id: str | None = None,
    ) -> dict[str, str]:
        pointer_path = Path(dataset_pointer)
        pointer = require_object(
            load_json(pointer_path, label="AWS dataset pointer"),
            label="AWS dataset pointer",
        )
        require_exact_keys(
            pointer,
            {
                "schema_version",
                "dataset_id",
                "provider",
                "materialization",
                "durable_uri_env",
                "scratch_root",
                "relative_path",
                "required_receipt",
                "required_sidecars",
                "source_lock_manifest",
                "full_corpus_in_release",
            },
            label="AWS dataset pointer",
        )
        allowed_pointer_providers = {self.profile.provider}
        if _is_v3_manifest(manifest):
            # The v3 amendment supersedes hardware selection, not the frozen
            # Task 4 dataset publication originally qualified on legacy P5.
            allowed_pointer_providers.add(AWS_P5_PROFILE)
        if (
            pointer["schema_version"] != 1
            or pointer["provider"] not in allowed_pointer_providers
            or pointer["materialization"] != "s3"
            or pointer["durable_uri_env"]
            != getattr(self.profile, "durable_uri_env", "MS_S3_ROOT")
            or pointer["scratch_root"]
            != getattr(self.profile, "scratch_root", "/mnt/memorysplit")
            or pointer["required_receipt"] != "dataset/receipt.json"
        ):
            raise MsctlError(
                "DATASET_POINTER_INVALID",
                "AWS lifecycle requires the canonical immutable dataset pointer",
            )
        if (dataset_root is None) == (dataset_verification is None):
            raise MsctlError(
                "CLI_USAGE",
                "AWS lifecycle requires exactly one dataset source",
            )
        if dataset_root is not None:
            dataset_receipt = Path(dataset_root) / "receipt.json"
            self._verify_local_dataset(dataset_receipt)
            receipt_sha256 = sha256_file(dataset_receipt)
            source_sha256 = receipt_sha256
        else:
            verification = require_object(
                load_json(
                    Path(str(dataset_verification)),
                    label="AWS dataset verification",
                ),
                label="AWS dataset verification",
            )
            receipt_sha256 = require_sha256(
                verification.get("receipt_sha256"),
                label="AWS dataset receipt",
            )
            source_sha256 = require_sha256(
                verification.get("verification_sha256"),
                label="AWS dataset verification",
            )
            unsigned = {
                key: value
                for key, value in verification.items()
                if key != "verification_sha256"
            }
            if canonical_sha256(unsigned) != source_sha256:
                raise MsctlError(
                    "DATASET_VERIFICATION_INVALID",
                    "AWS dataset verification identity does not match content",
                )
        if receipt_sha256 != manifest.dataset_sha256:
            raise MsctlError(
                "DATASET_PROVENANCE_MISMATCH",
                "AWS dataset evidence does not match the run manifest",
            )

        environment_path = Path(environment_receipt)
        environment_bytes = self._read_lifecycle_environment_receipt(
            environment_path
        )
        v3_environment = _is_v3_manifest(manifest)
        if v3_environment:
            if (
                selection is None
                or expected_instance_id is None
                or _INSTANCE_ID_RE.fullmatch(expected_instance_id) is None
                or selection.sha256
                != getattr(manifest, "provider_selection_sha256", None)
                or selection.selected_profile_id
                != getattr(self.profile, "profile_id", None)
                or selection.provider != self.profile.provider
                or selection.profile_sha256 != self.profile.sha256
                or selection.instance_type != self.profile.instance_type
                or selection.region != self.runtime.region
                or selection.ami_id != self.runtime.ami_id
                or selection.container_image != self.runtime.container_image
                or selection.container_digest != self.runtime.container_digest
            ):
                raise MsctlError(
                    "ENVIRONMENT_RECEIPT_INVALID",
                    "AWS environment receipt lacks its validated provider "
                    "selection binding",
                )
            try:
                verified_environment = aws_identity.verify_environment_receipt(
                    environment_bytes,
                    expected_profile_sha256=self.profile.sha256,
                    expected_container_digest=selection.container_digest,
                    expected_region=selection.region,
                    expected_ami_id=selection.ami_id,
                    expected_account_id=selection.aws_account_id,
                    expected_instance_id=expected_instance_id,
                    expected_instance_type=selection.instance_type,
                )
            except AwsIdentityError as error:
                raise MsctlError(
                    "ENVIRONMENT_RECEIPT_INVALID",
                    "AWS authenticated receipt does not bind the selected "
                    "account, instance, type, region, AMI, and profile",
                ) from error
            environment_sha256 = verified_environment.receipt_sha256
        else:
            try:
                environment = require_object(
                    json.loads(environment_bytes.decode("utf-8")),
                    label="AWS environment receipt",
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise MsctlError(
                    "ENVIRONMENT_RECEIPT_INVALID",
                    "AWS environment receipt must contain valid UTF-8 JSON",
                ) from error
            environment_sha256 = hashlib.sha256(environment_bytes).hexdigest()
        environment_fields = {
            "schema_version",
            "profile_sha256",
            "container_image_digest",
            "aws_instance_identity_document",
            "aws_instance_identity_pkcs7",
        }
        if not v3_environment:
            require_exact_keys(
                environment,
                environment_fields,
                label="AWS environment receipt",
            )
            identity = require_object(
                environment["aws_instance_identity_document"],
                label="AWS instance identity document",
            )
            pkcs7 = environment["aws_instance_identity_pkcs7"]
            try:
                decoded_pkcs7 = base64.b64decode(
                    "".join(str(pkcs7).split()),
                    validate=True,
                )
            except (ValueError, TypeError) as error:
                raise MsctlError(
                    "ENVIRONMENT_RECEIPT_INVALID",
                    "AWS identity PKCS7 is not valid base64",
                ) from error
            if (
                environment_bytes != canonical_json(environment) + b"\n"
                or environment["schema_version"] != 1
                or environment["profile_sha256"] != self.profile.sha256
                or environment["container_image_digest"]
                != self.runtime.container_digest
                or identity.get("imageId") != self.runtime.ami_id
                or identity.get("region") != self.runtime.region
                or (
                    expected_instance_id is not None
                    and identity.get("instanceId") != expected_instance_id
                )
                or not decoded_pkcs7
                or not self.identity_verifier(
                    identity,
                    "".join(str(pkcs7).split()),
                    self.runtime.region,
                )
            ):
                raise MsctlError(
                    "ENVIRONMENT_RECEIPT_INVALID",
                    "AWS authenticated receipt does not bind the selected runtime",
                )
        return {
            "dataset_pointer_sha256": sha256_file(pointer_path),
            "dataset_verification_sha256": source_sha256,
            "environment_receipt_sha256": environment_sha256,
        }

    def render(
        self,
        *,
        release: object,
        manifest: object,
        evidence: Mapping[str, str] | None = None,
        context: V3LifecycleContext | None = None,
    ) -> dict[str, object]:
        self._validate_manifest(manifest)
        self._validate_release(release, manifest)
        self._validate_v3_context(manifest, context)
        operation_intent = self._training_operation_intent(
            operation="render",
            release=release,
            manifest=manifest,
            evidence=evidence,
            context=context,
        )
        return {
            "provider": self.profile.provider,
            "seed": manifest.seed,
            "release_sha256": release.archive_sha256,
            "run_manifest_sha256": manifest.sha256,
            "operation_intent": operation_intent,
            "layout": {
                "allocated_gpus": 8,
                "train_groups": [4, 4],
            },
        }

    def status(
        self,
        *,
        release: object,
        manifest: object,
        cached: bool,
        context: V3LifecycleContext | None = None,
    ) -> dict[str, object]:
        self._validate_manifest(manifest)
        self._validate_release(release, manifest)
        validated_context = self._validate_v3_context(manifest, context)
        store = StateStore(self.state_root)
        with store.locked():
            states = self._paired_states(store, manifest, context)
            command_ids = {state.get("command_id") for state in states}
            instance_ids = {state.get("instance_id") for state in states}
            if (
                None in command_ids
                or len(command_ids) != 1
                or None in instance_ids
                or len(instance_ids) != 1
            ):
                raise MsctlError(
                    "SUBMISSION_UNCERTAIN",
                    "AWS paired state lacks one command and instance",
                )
            command_id = str(next(iter(command_ids)))
            instance_id = str(next(iter(instance_ids)))
            if (
                validated_context is not None
                and instance_id != validated_context.instance_id
            ):
                raise MsctlError(
                    "FLEET_PLAN_INVALID",
                    "status state is bound to a different fleet instance",
                )
            if cached:
                statuses = {str(state.get("status")) for state in states}
                status = (
                    next(iter(statuses)) if len(statuses) == 1 else "Mixed"
                )
                return {
                    "provider": self.profile.provider,
                    "seed": manifest.seed,
                    "instance_id": instance_id,
                    "command_id": command_id,
                    "status": status,
                    "authoritative": False,
                    "runs": states,
                }
            status = self._command_status(instance_id, command_id)
            now = _timestamp()
            for run, state in zip(manifest.runs, states):
                state["status"] = status
                state["updated_at"] = now
                store.write_run(run.run_id, state)
            return {
                "provider": self.profile.provider,
                "seed": manifest.seed,
                "instance_id": instance_id,
                "command_id": command_id,
                "status": status,
                "authoritative": True,
                "runs": states,
            }

    def _s3_location(self, relative: str) -> tuple[str, str]:
        parsed = urlsplit(self.runtime.s3_root)
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or not relative
            or relative.startswith("/")
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))
        ):
            raise MsctlError(
                "AWS_RUNTIME_INVALID",
                "S3 object path is unsafe",
            )
        prefix = parsed.path.strip("/")
        key = f"{prefix}/{relative}" if prefix else relative
        return parsed.netloc, key

    def _s3_head(
        self,
        relative: str,
        *,
        operation: str,
        apply: bool,
    ) -> dict[str, object]:
        bucket, key = self._s3_location(relative)
        argv = self._aws_argv(
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            query=(
                "{object:{content_length:ContentLength,etag:ETag,"
                "version_id:VersionId}}"
            ),
        )
        if not apply:
            return {
                "provider": self.profile.provider,
                "s3_uri": f"s3://{bucket}/{key}",
                "commands": [argv],
                "verified": False,
            }
        output = _aws_output_object(
            self._run(argv, operation=operation),
            {"object"},
            label="S3 head-object output",
        )
        row = _aws_output_object(
            output["object"],
            {"content_length", "etag", "version_id"},
            label="S3 object metadata",
        )
        if (
            isinstance(row["content_length"], bool)
            or not isinstance(row["content_length"], int)
            or row["content_length"] < 0
            or not isinstance(row["etag"], str)
            or not row["etag"]
            or (
                row["version_id"] is not None
                and not isinstance(row["version_id"], str)
            )
        ):
            raise MsctlError(
                "AWS_OUTPUT_INVALID",
                "S3 object metadata is invalid",
            )
        return {
            "provider": self.profile.provider,
            "s3_uri": f"s3://{bucket}/{key}",
            "verified": True,
            "object": row,
        }

    def _verify_local_dataset(self, receipt_path: Path | str):
        receipt = require_object(
            load_json(receipt_path, label="Task 4 dataset receipt"),
            label="Task 4 dataset receipt",
        )
        try:
            ordered_sha256 = require_sha256(
                receipt.get("ordered_stream_sha256"),
                label="Task 4 dataset ordered stream",
            )
            if self.corpus_verifier is None:
                from cluster.aws.p5.corpus_contract import verify_canonical_corpus

                verifier = verify_canonical_corpus
            else:
                verifier = self.corpus_verifier
            evidence = verifier(
                Path(receipt_path),
                expected_sha256=sha256_file(receipt_path),
                expected_ordered_sha256=ordered_sha256,
            )
            from .operations import _dataset_file_identities

            identities = _dataset_file_identities(evidence)
        except MsctlError:
            raise
        except (ImportError, ModuleNotFoundError) as error:
            raise MsctlError(
                "DATASET_ADAPTER_UNAVAILABLE",
                "the canonical Task 4 dataset verifier is not installed",
            ) from error
        except Exception as error:
            raise MsctlError(
                "DATASET_RECEIPT_INVALID",
                "dataset is not a verified canonical Task 4 publication",
            ) from error
        return evidence, identities

    @staticmethod
    def _hash_regular_nofollow(
        path: Path,
        *,
        label: str,
    ) -> tuple[str, int, tuple[int, int, int, int, int]]:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                f"{label} cannot be opened without following links",
            ) from error
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    f"{label} must be one singly linked regular file",
                )
            digest = hashlib.sha256()
            byte_count = 0
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                byte_count += len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or byte_count != before.st_size:
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                f"{label} changed while read",
            )
        return digest.hexdigest(), byte_count, identity

    def _snapshot_checkpoint(
        self,
        source: Path,
        destination: Path,
        *,
        expected_sha256: str,
        expected_bytes: int,
        arm: str,
    ) -> None:
        """Copy one no-follow checkpoint into a private immutable upload path."""

        if destination.exists() or destination.is_symlink():
            digest, byte_count, _ = self._hash_regular_nofollow(
                destination,
                label=f"{arm} checkpoint snapshot",
            )
            mode = stat.S_IMODE(destination.stat(follow_symlinks=False).st_mode)
            if (
                digest != expected_sha256
                or byte_count != expected_bytes
                or mode != 0o400
            ):
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    f"{arm} checkpoint snapshot identity has drifted",
                )
            return
        directory_fd = open_directory(
            destination.parent,
            label=f"{arm} checkpoint snapshot",
            create=True,
        )
        source_fd: int | None = None
        destination_fd: int | None = None
        temporary = f".{destination.name}.{secrets.token_hex(12)}.tmp"
        try:
            source_fd = os.open(
                source,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            source_before = os.fstat(source_fd)
            if (
                not stat.S_ISREG(source_before.st_mode)
                or source_before.st_nlink != 1
            ):
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    f"{arm} checkpoint source is not one regular file",
                )
            destination_fd = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=directory_fd,
            )
            digest = hashlib.sha256()
            byte_count = 0
            while True:
                chunk = os.read(source_fd, 1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                byte_count += len(chunk)
                view = memoryview(chunk)
                while view:
                    view = view[os.write(destination_fd, view) :]
            source_after = os.fstat(source_fd)
            if (
                (
                    source_before.st_dev,
                    source_before.st_ino,
                    source_before.st_size,
                    source_before.st_mtime_ns,
                    source_before.st_ctime_ns,
                )
                != (
                    source_after.st_dev,
                    source_after.st_ino,
                    source_after.st_size,
                    source_after.st_mtime_ns,
                    source_after.st_ctime_ns,
                )
                or byte_count != expected_bytes
                or digest.hexdigest() != expected_sha256
            ):
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    f"{arm} checkpoint changed during no-follow snapshot",
                )
            os.fchmod(destination_fd, 0o400)
            os.fsync(destination_fd)
            os.close(destination_fd)
            destination_fd = None
            try:
                rename_noreplace_at(
                    directory_fd,
                    temporary,
                    directory_fd,
                    destination.name,
                )
            except FileExistsError as error:
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    f"{arm} checkpoint snapshot appeared concurrently",
                ) from error
            os.fsync(directory_fd)
        except OSError as error:
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                f"{arm} checkpoint snapshot failed",
            ) from error
        finally:
            if destination_fd is not None:
                os.close(destination_fd)
            if source_fd is not None:
                os.close(source_fd)
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            os.close(directory_fd)
        digest, byte_count, _ = self._hash_regular_nofollow(
            destination,
            label=f"{arm} checkpoint snapshot",
        )
        if digest != expected_sha256 or byte_count != expected_bytes:
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                f"{arm} checkpoint snapshot verification failed",
            )

    def _immutable_s3_object_argv(
        self,
        *,
        local_path: Path,
        relative: str,
        sha256: str,
        byte_count: int,
        upload: bool,
        require_kms: bool = False,
    ) -> list[str]:
        bucket, key = self._s3_location(relative)
        kms_key_id = getattr(self.runtime, "kms_key_id", None)
        if require_kms and not isinstance(kms_key_id, str):
            raise MsctlError(
                "AWS_RUNTIME_INVALID",
                "v3 checkpoint publication requires an exact SSE-KMS key",
            )
        if upload:
            checksum = base64.b64encode(bytes.fromhex(sha256)).decode("ascii")
            encryption = (
                [
                    "--server-side-encryption",
                    "aws:kms",
                    "--ssekms-key-id",
                    str(kms_key_id),
                ]
                if require_kms
                else []
            )
            return self._aws_argv(
                "s3api",
                "put-object",
                "--bucket",
                bucket,
                "--key",
                key,
                "--body",
                str(local_path),
                "--content-length",
                str(byte_count),
                "--checksum-algorithm",
                "SHA256",
                "--checksum-sha256",
                checksum,
                "--metadata",
                f"sha256={sha256}",
                "--if-none-match",
                "*",
                *encryption,
                query=(
                    "{object:{checksum_sha256:ChecksumSHA256,"
                    "etag:ETag,version_id:VersionId}}"
                ),
            )
        query = (
            "{object:{bytes:ContentLength,"
            "checksum_sha256:ChecksumSHA256,metadata:Metadata,"
            "server_side_encryption:ServerSideEncryption,"
            "sse_kms_key_id:SSEKMSKeyId,version_id:VersionId}}"
            if require_kms
            else (
                "{object:{bytes:ContentLength,"
                "checksum_sha256:ChecksumSHA256,metadata:Metadata,"
                "version_id:VersionId}}"
            )
        )
        return self._aws_argv(
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--checksum-mode",
            "ENABLED",
            query=query,
        )

    def _checkpoint_publication(
        self,
        *,
        checkpoint_receipt: object,
        checkpoints: Mapping[str, object],
        apply: bool,
    ) -> dict[str, object]:
        receipt_sha256 = require_sha256(
            getattr(checkpoint_receipt, "sha256", None),
            label="checkpoint receipt",
        )
        receipt_value = require_object(
            getattr(checkpoint_receipt, "value", None),
            label="checkpoint receipt",
        )
        v3_checkpoint = getattr(checkpoint_receipt, "schema_version", None) == 3
        checkpoint_seeds = {
            getattr(checkpoint, "seed", None)
            for checkpoint in checkpoints.values()
        }
        if v3_checkpoint and (
            len(checkpoint_seeds) != 1
            or not all(
                type(seed) is int and seed in range(10)
                for seed in checkpoint_seeds
            )
        ):
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                "v3 checkpoint publication requires one exact seed",
            )
        checkpoint_prefix = (
            f"checkpoints/seed-{next(iter(checkpoint_seeds))}"
            if v3_checkpoint
            else "checkpoints"
        )
        receipt_bytes = canonical_json(receipt_value) + (
            b"\n"
            if v3_checkpoint
            else b""
        )
        if hashlib.sha256(receipt_bytes).hexdigest() != receipt_sha256:
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                "checkpoint receipt canonical bytes do not match its identity",
            )
        receipt_path = (
            self.state_root / f"checkpoint-receipt-{receipt_sha256}.json"
        ).absolute()
        objects: list[dict[str, object]] = [
            {
                "path": "receipt.json",
                "local_path": receipt_path,
                "s3_relative": (
                    f"{checkpoint_prefix}/receipts/{receipt_sha256}.json"
                ),
                "sha256": receipt_sha256,
                "bytes": len(receipt_bytes),
                "device": None,
                "inode": None,
            }
        ]
        for arm in ("dense", "split90"):
            checkpoint = checkpoints[arm]
            path = Path(checkpoint.path)
            digest, byte_count, identity = self._hash_regular_nofollow(
                path,
                label=f"{arm} checkpoint",
            )
            if digest != checkpoint.sha256:
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    "checkpoint identity changed before immutable publication",
                    details={"arm": arm},
                )
            upload_path = path
            if v3_checkpoint:
                seed = next(iter(checkpoint_seeds))
                upload_path = (
                    self.state_root.absolute()
                    / "checkpoint-upload-snapshots"
                    / f"seed-{seed}"
                    / digest
                    / f"{arm}.pt"
                )
            objects.append(
                {
                    "path": f"{arm}.pt",
                    "local_path": upload_path,
                    "source_path": path,
                    "s3_relative": (
                        f"{checkpoint_prefix}/{arm}/sha256/{digest}.pt"
                    ),
                    "sha256": digest,
                    "bytes": byte_count,
                    "device": identity[0],
                    "inode": identity[1],
                }
            )
        commands = [
            self._immutable_s3_object_argv(
                local_path=Path(item["local_path"]),
                relative=str(item["s3_relative"]),
                sha256=str(item["sha256"]),
                byte_count=int(item["bytes"]),
                upload=upload,
                require_kms=v3_checkpoint,
            )
            for item in objects
            for upload in (True, False)
        ]
        result = {
            "receipt_sha256": receipt_sha256,
            "objects": [
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"local_path", "source_path"}
                }
                for item in objects
            ],
            "commands": commands,
            "verified": False,
        }
        if not apply:
            return result
        directory_fd = open_directory(
            self.state_root,
            label="AWS checkpoint publication state",
            create=True,
        )
        try:
            try:
                existing = receipt_path.read_bytes()
            except FileNotFoundError:
                atomic_write_at(
                    directory_fd,
                    receipt_path.name,
                    receipt_bytes,
                    label="checkpoint receipt publication",
                )
            else:
                if receipt_path.is_symlink() or existing != receipt_bytes:
                    raise MsctlError(
                        "CHECKPOINT_RECEIPT_CONFLICT",
                        "local checkpoint receipt publication conflicts",
                    )
        finally:
            os.close(directory_fd)
        receipt_stat = receipt_path.stat(follow_symlinks=False)
        objects[0]["device"] = receipt_stat.st_dev
        objects[0]["inode"] = receipt_stat.st_ino
        if v3_checkpoint:
            for item in objects[1:]:
                self._snapshot_checkpoint(
                    Path(item["source_path"]),
                    Path(item["local_path"]),
                    expected_sha256=str(item["sha256"]),
                    expected_bytes=int(item["bytes"]),
                    arm=str(item["path"]).removesuffix(".pt"),
                )
        for index, item in enumerate(objects):
            local_digest, local_bytes, _ = self._hash_regular_nofollow(
                Path(item["local_path"]),
                label=f"checkpoint upload {item['path']}",
            )
            if (
                local_digest != item["sha256"]
                or local_bytes != item["bytes"]
            ):
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    "checkpoint upload source differs from its reviewed bytes",
                    details={"path": item["path"]},
                )
            try:
                self._run(
                    commands[index * 2],
                    operation="materialize checkpoint object",
                )
            except MsctlError:
                pass
            self._verify_dataset_object(
                item,
                self._run(
                    commands[index * 2 + 1],
                    operation="verify checkpoint object",
                ),
                require_kms=v3_checkpoint,
            )
            final_digest, final_bytes, _ = self._hash_regular_nofollow(
                Path(item["local_path"]),
                label=f"checkpoint upload {item['path']}",
            )
            if (
                final_digest != item["sha256"]
                or final_bytes != item["bytes"]
            ):
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    "checkpoint upload source changed during publication",
                    details={"path": item["path"]},
                )
        result["verified"] = True
        result["objects"] = [
            {
                key: value
                for key, value in item.items()
                    if key not in {"local_path", "source_path"}
            }
            for item in objects
        ]
        return result

    def env_ensure(
        self,
        *,
        root: Path | str,
        lock: Path | str,
        apply: bool,
    ) -> dict[str, object]:
        raise MsctlError(
            "STATIC_ENVIRONMENT_FORBIDDEN",
            "AWS runtime identity must come from the package runtime_attested "
            "contract and an authenticated instance receipt",
        )
        lock_path = Path(lock).resolve(strict=True)
        before = lock_path.stat(follow_symlinks=False)
        lock_sha256 = sha256_file(lock_path)
        after = lock_path.stat(follow_symlinks=False)
        if (
            not lock_path.is_file()
            or lock_path.is_symlink()
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino, before.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
        ):
            raise MsctlError(
                "ENVIRONMENT_LOCK_INVALID",
                "AWS environment lock must be one stable regular file",
            )
        receipt = {
            "schema_version": 1,
            "receipt_type": "memorysplit-aws-environment",
            "profile_sha256": self.profile.sha256,
            "runtime_sha256": self._runtime_sha256(),
            "ami_id": self.runtime.ami_id,
            "container_image": self.runtime.container_image,
            "container_digest": self.runtime.container_digest,
            "lock_name": lock_path.name,
            "lock_sha256": lock_sha256,
            "lock_bytes": before.st_size,
        }
        receipt_bytes = canonical_json(receipt) + b"\n"
        receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        receipt_root = Path(root).absolute()
        receipt_path = receipt_root / "environment-receipt.json"
        objects = [
            {
                "path": lock_path,
                "s3_relative": (
                    f"environment/locks/{lock_sha256}/{lock_path.name}"
                ),
                "sha256": lock_sha256,
                "bytes": before.st_size,
                "device": before.st_dev,
                "inode": before.st_ino,
            },
            {
                "path": receipt_path,
                "s3_relative": (
                    f"environment/receipts/{receipt_sha256}.json"
                ),
                "sha256": receipt_sha256,
                "bytes": len(receipt_bytes),
                "device": None,
                "inode": None,
            },
        ]
        commands = [
            self._immutable_s3_object_argv(
                local_path=item["path"],
                relative=item["s3_relative"],
                sha256=item["sha256"],
                byte_count=item["bytes"],
                upload=upload,
            )
            for item in objects
            for upload in (True, False)
        ]
        result = {
            "provider": self.profile.provider,
            "receipt": receipt,
            "receipt_sha256": receipt_sha256,
            "receipt_path": str(receipt_path),
            "objects": [
                {
                    key: value
                    for key, value in item.items()
                    if key != "path"
                }
                for item in objects
            ],
            "commands": commands,
            "verified": False,
        }
        if not apply:
            return result
        directory_fd = open_directory(
            receipt_root,
            label="AWS environment receipt root",
            create=True,
        )
        try:
            try:
                existing = receipt_path.read_bytes()
            except FileNotFoundError:
                atomic_write_at(
                    directory_fd,
                    receipt_path.name,
                    receipt_bytes,
                    label="AWS environment receipt",
                )
            else:
                if receipt_path.is_symlink() or existing != receipt_bytes:
                    raise MsctlError(
                        "ENVIRONMENT_RECEIPT_CONFLICT",
                        "existing AWS environment receipt has different content",
                    )
        finally:
            os.close(directory_fd)
        receipt_stat = receipt_path.stat(follow_symlinks=False)
        objects[1]["device"] = receipt_stat.st_dev
        objects[1]["inode"] = receipt_stat.st_ino
        for index, item in enumerate(objects):
            put_argv = commands[index * 2]
            head_argv = commands[index * 2 + 1]
            try:
                self._run(put_argv, operation="materialize environment object")
            except MsctlError:
                pass
            self._verify_dataset_object(
                item,
                self._run(head_argv, operation="verify environment object"),
            )
        result["verified"] = True
        result["objects"] = [
            {key: value for key, value in item.items() if key != "path"}
            for item in objects
        ]
        return result

    def _dataset_object_argv(
        self,
        identity: Mapping[str, object],
        *,
        upload: bool,
        root: Path,
    ) -> list[str]:
        relative = str(identity["path"])
        bucket, key = self._s3_location(f"dataset/{relative}")
        if upload:
            checksum = base64.b64encode(
                bytes.fromhex(str(identity["sha256"]))
            ).decode("ascii")
            return self._aws_argv(
                "s3api",
                "put-object",
                "--bucket",
                bucket,
                "--key",
                key,
                "--body",
                str(root / relative),
                "--content-length",
                str(identity["bytes"]),
                "--checksum-algorithm",
                "SHA256",
                "--checksum-sha256",
                checksum,
                "--metadata",
                f"sha256={identity['sha256']}",
                "--if-none-match",
                "*",
                query=(
                    "{object:{checksum_sha256:ChecksumSHA256,"
                    "etag:ETag,version_id:VersionId}}"
                ),
            )
        return self._aws_argv(
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--checksum-mode",
            "ENABLED",
            query=(
                "{object:{bytes:ContentLength,"
                "checksum_sha256:ChecksumSHA256,metadata:Metadata,"
                "version_id:VersionId}}"
            ),
        )

    def _verify_dataset_object(
        self,
        identity: Mapping[str, object],
        output: object,
        *,
        require_kms: bool = False,
    ) -> None:
        wrapped = _aws_output_object(
            output,
            {"object"},
            label="S3 dataset object response",
        )
        fields = {
            "bytes",
            "checksum_sha256",
            "metadata",
            "version_id",
        }
        if require_kms:
            fields |= {"server_side_encryption", "sse_kms_key_id"}
        row = _aws_output_object(
            wrapped["object"],
            fields,
            label="S3 dataset object",
        )
        metadata = _aws_output_object(
            row["metadata"],
            {"sha256"},
            label="S3 dataset object metadata",
        )
        expected_checksum = base64.b64encode(
            bytes.fromhex(str(identity["sha256"]))
        ).decode("ascii")
        if (
            row["bytes"] != identity["bytes"]
            or row["checksum_sha256"] != expected_checksum
            or metadata["sha256"] != identity["sha256"]
            or (
                require_kms
                and (
                    row["server_side_encryption"] != "aws:kms"
                    or row["sse_kms_key_id"]
                    != getattr(self.runtime, "kms_key_id", None)
                )
            )
            or (
                row["version_id"] is not None
                and not isinstance(row["version_id"], str)
            )
        ):
            raise MsctlError(
                "DATASET_S3_MISMATCH",
                "S3 dataset object content binding does not match Task 4",
                details={"path": identity["path"]},
            )

    def _dataset_publication(
        self,
        *,
        receipt_path: Path | str,
        upload: bool,
        apply: bool,
    ) -> dict[str, object]:
        evidence, identities = self._verify_local_dataset(receipt_path)
        root = Path(receipt_path).parent.resolve(strict=True)
        commands: list[list[str]] = []
        for identity in identities:
            if upload:
                commands.append(
                    self._dataset_object_argv(
                        identity,
                        upload=True,
                        root=root,
                    )
                )
            commands.append(
                self._dataset_object_argv(
                    identity,
                    upload=False,
                    root=root,
                )
            )
        result = {
            "provider": self.profile.provider,
            "receipt_sha256": sha256_file(receipt_path),
            "ordered_stream_sha256": evidence.receipt[
                "ordered_stream_sha256"
            ],
            "file_identities": identities,
            "commands": commands,
            "verified": False,
        }
        if not apply:
            return result
        offset = 0
        for identity in identities:
            if upload:
                put_argv = commands[offset]
                offset += 1
                try:
                    self._run(put_argv, operation="materialize dataset object")
                except MsctlError:
                    # Conditional creation may report an existing object. The
                    # authoritative checksum-bound HEAD below decides idempotence.
                    pass
            head_argv = commands[offset]
            offset += 1
            self._verify_dataset_object(
                identity,
                self._run(head_argv, operation="verify dataset object"),
            )
        result["verified"] = True
        return result

    def dataset_ensure(
        self,
        *,
        receipt_path: Path | str,
        apply: bool,
    ) -> dict[str, object]:
        return self._dataset_publication(
            receipt_path=receipt_path,
            upload=True,
            apply=apply,
        )

    def dataset_verify(
        self,
        *,
        receipt_path: Path | str,
        apply: bool,
    ) -> dict[str, object]:
        return self._dataset_publication(
            receipt_path=receipt_path,
            upload=False,
            apply=apply,
        )

    @staticmethod
    def _read_collection_regular(
        path: Path,
        *,
        label: str,
        maximum: int = 1 << 30,
    ) -> bytes:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise MsctlError(
                "FLEET_COLLECTION_INVALID",
                f"{label} cannot be opened without following links",
            ) from error
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size <= 0
                or before.st_size > maximum
            ):
                raise MsctlError(
                    "FLEET_COLLECTION_INVALID",
                    f"{label} must be one bounded singly linked regular file",
                )
            chunks: list[bytes] = []
            offset = 0
            while offset < before.st_size:
                chunk = os.pread(
                    descriptor,
                    min(1 << 20, before.st_size - offset),
                    offset,
                )
                if not chunk:
                    break
                chunks.append(chunk)
                offset += len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if (
            (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or offset != after.st_size
        ):
            raise MsctlError(
                "FLEET_COLLECTION_INVALID",
                f"{label} changed while being read",
            )
        return b"".join(chunks)

    def _runtime_relative_uri(self, uri: object, *, label: str) -> str:
        prefix = f"{self.runtime.s3_root.rstrip('/')}/"
        if (
            not isinstance(uri, str)
            or not uri.startswith(prefix)
            or any(character in uri for character in "\n\r\x00")
        ):
            raise MsctlError(
                "FLEET_COLLECTION_INVALID",
                f"{label} is outside the immutable runtime root",
            )
        relative = uri.removeprefix(prefix)
        bucket, key = self._s3_location(relative)
        if f"s3://{bucket}/{key}" != uri:
            raise MsctlError(
                "FLEET_COLLECTION_INVALID",
                f"{label} is not a canonical runtime URI",
            )
        return relative

    def _collection_get_object(
        self,
        *,
        relative: str,
        destination: Path,
        label: str,
        expected_sha256: str | None = None,
        expected_bytes: int | None = None,
    ) -> tuple[list[str], dict[str, object]]:
        bucket, key = self._s3_location(relative)
        argv = self._aws_argv(
            "s3api",
            "get-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--checksum-mode",
            "ENABLED",
            str(destination),
            query=(
                "{object:{checksum_sha256:ChecksumSHA256,"
                "version_id:VersionId}}"
            ),
        )
        destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        output = _aws_output_object(
            self._run(argv, operation=f"collect {label}"),
            {"object"},
            label=f"{label} get-object output",
        )
        row = _aws_output_object(
            output["object"],
            {"checksum_sha256", "version_id"},
            label=f"{label} downloaded object",
        )
        digest, byte_count, _ = self._hash_regular_nofollow(
            destination,
            label=label,
        )
        expected_checksum = base64.b64encode(
            bytes.fromhex(expected_sha256 or digest)
        ).decode("ascii")
        if (
            row["checksum_sha256"] != expected_checksum
            or (
                row["version_id"] is not None
                and (
                    not isinstance(row["version_id"], str)
                    or not row["version_id"]
                )
            )
            or (expected_sha256 is not None and digest != expected_sha256)
            or (expected_bytes is not None and byte_count != expected_bytes)
        ):
            raise MsctlError(
                "COLLECT_INCOMPLETE",
                f"{label} does not match its immutable S3 identity",
            )
        return argv, {
            "path": destination,
            "sha256": digest,
            "bytes": byte_count,
        }

    def _verify_v3_lifecycle_collection(
        self,
        root: Path | str,
        *,
        source: str | None = None,
        manifest: object | None = None,
        fleet_plan_sha256: str | None = None,
        fleet_wave: int | None = None,
    ) -> dict[str, object]:
        collection = Path(root)
        try:
            root_status = collection.stat(follow_symlinks=False)
            resolved = collection.resolve(strict=True)
        except OSError as error:
            raise MsctlError(
                "FLEET_COLLECTION_INVALID",
                "v3 lifecycle collection root is unavailable",
            ) from error
        if collection.is_symlink() or not stat.S_ISDIR(root_status.st_mode):
            raise MsctlError(
                "FLEET_COLLECTION_INVALID",
                "v3 lifecycle collection root must be a real directory",
            )

        evaluation_payload = self._read_collection_regular(
            resolved / "EVALUATION.json",
            label="paired evaluation receipt",
        )
        checkpoint_payload = self._read_collection_regular(
            resolved / "CHECKPOINT.json",
            label="paired checkpoint receipt",
        )
        try:
            checkpoint = verify_checkpoint_receipt_bytes(
                checkpoint_payload,
                expected={"provider": self.profile.provider},
                require_durable=True,
            )
            evaluation = verify_evaluation_receipt_bytes(
                evaluation_payload,
                expected={
                    "provider": self.profile.provider,
                    "profile_sha256": self.profile.sha256,
                },
            )
        except (TypeError, ValueError) as error:
            raise MsctlError(
                "FLEET_COLLECTION_INVALID",
                "paired lifecycle receipts do not satisfy their closed schemas",
            ) from error
        checkpoint_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()
        evaluation_sha256 = hashlib.sha256(evaluation_payload).hexdigest()
        seed = evaluation["seed"]
        canonical_source = f"results/seed-{seed}.json"
        if source is not None and source != canonical_source:
            raise MsctlError(
                "FLEET_COLLECTION_INVALID",
                "collection source does not match the paired evaluation seed",
            )
        checkpoint_uri = (
            f"{self.runtime.s3_root.rstrip('/')}/checkpoints/seed-{seed}/"
            f"receipts/{checkpoint_sha256}.json"
        )
        if evaluation["checkpoint_receipt"] != {
            "sha256": checkpoint_sha256,
            "uri": checkpoint_uri,
        }:
            raise MsctlError(
                "FLEET_COLLECTION_INVALID",
                "evaluation receipt does not bind the collected checkpoint receipt",
            )
        self._runtime_relative_uri(
            evaluation["checkpoint_receipt"]["uri"],
            label="checkpoint receipt URI",
        )
        shared_fields = (
            "provider",
            "release_sha256",
            "run_manifest_sha256",
            "dataset_sha256",
            "source_commit",
            "cohort_assignment_sha256",
            "preregistration_sha256",
            "hardware_amendment_sha256",
            "provider_selection_sha256",
            "profile_sha256",
            "sealed_fixture_sha256",
        )
        if any(evaluation[field] != checkpoint[field] for field in shared_fields):
            raise MsctlError(
                "FLEET_COLLECTION_INVALID",
                "checkpoint and evaluation receipts have divergent provenance",
            )
        checkpoint_rows = {
            str(row["arm"]): row for row in checkpoint["checkpoints"]
        }
        evaluation_rows = {
            str(row["arm"]): row for row in evaluation["arms"]
        }
        if (
            set(checkpoint_rows) != {"dense", "split90"}
            or set(evaluation_rows) != {"dense", "split90"}
            or any(
                evaluation_rows[arm]["run_id"]
                != checkpoint_rows[arm]["run_id"]
                or evaluation_rows[arm]["checkpoint_sha256"]
                != checkpoint_rows[arm]["sha256"]
                or checkpoint_rows[arm]["seed"] != seed
                for arm in ("dense", "split90")
            )
        ):
            raise MsctlError(
                "FLEET_COLLECTION_INVALID",
                "evaluation receipt does not close the checkpoint pair",
            )
        if manifest is not None:
            expected_manifest = {
                "provider": getattr(manifest, "provider", None),
                "release_sha256": getattr(manifest, "release_sha256", None),
                "run_manifest_sha256": getattr(manifest, "sha256", None),
                "dataset_sha256": getattr(manifest, "dataset_sha256", None),
                "source_commit": getattr(manifest, "source_commit", None),
                "cohort_assignment_sha256": getattr(
                    manifest,
                    "cohort_assignment_sha256",
                    None,
                ),
                "preregistration_sha256": getattr(
                    manifest,
                    "preregistration_sha256",
                    None,
                ),
                "hardware_amendment_sha256": getattr(
                    manifest,
                    "hardware_amendment_sha256",
                    None,
                ),
                "provider_selection_sha256": getattr(
                    manifest,
                    "provider_selection_sha256",
                    None,
                ),
                "profile_sha256": getattr(manifest, "profile_sha256", None),
                "sealed_fixture_sha256": getattr(
                    manifest,
                    "sealed_fixture_sha256",
                    None,
                ),
            }
            if any(
                evaluation[field] != expected
                for field, expected in expected_manifest.items()
            ) or (
                evaluation["seed"] != getattr(manifest, "seed", None)
                or evaluation["fleet_plan_sha256"] != fleet_plan_sha256
                or evaluation["fleet_wave"] != fleet_wave
            ):
                raise MsctlError(
                    "FLEET_COLLECTION_INVALID",
                    "lifecycle receipts do not bind the completed fleet wave",
                )
            manifest_runs = {
                str(run.arm): run for run in getattr(manifest, "runs", ())
            }
            if set(manifest_runs) != {"dense", "split90"} or any(
                checkpoint_rows[arm]["run_id"] != manifest_runs[arm].run_id
                or checkpoint_rows[arm]["config_sha256"]
                != manifest_runs[arm].config_sha256
                for arm in ("dense", "split90")
            ):
                raise MsctlError(
                    "FLEET_COLLECTION_INVALID",
                    "checkpoint receipt does not bind the manifest run pair",
                )

        expected_files: dict[str, dict[str, object]] = {}
        for name, payload in (
            ("CHECKPOINT.json", checkpoint_payload),
            ("EVALUATION.json", evaluation_payload),
        ):
            expected_files[name] = {
                "path": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        for arm in ("dense", "split90"):
            arm_row = evaluation_rows[arm]
            run_id = str(arm_row["run_id"])
            artifacts_by_name = {
                str(artifact["path"]): artifact
                for artifact in arm_row["artifacts"]
            }
            for artifact in arm_row["artifacts"]:
                name = str(artifact["path"])
                relative = f"{run_id}/{name}"
                expected_uri = (
                    f"{self.runtime.s3_root.rstrip('/')}/evaluations/seed-{seed}/"
                    f"{run_id}/{name}/sha256/{artifact['sha256']}"
                )
                if artifact["uri"] != expected_uri:
                    raise MsctlError(
                        "FLEET_COLLECTION_INVALID",
                        "evaluation artifact URI is not canonical",
                        details={"path": relative},
                    )
                self._runtime_relative_uri(
                    artifact["uri"],
                    label=f"evaluation artifact {relative}",
                )
                digest, byte_count, _ = self._hash_regular_nofollow(
                    resolved.joinpath(*relative.split("/")),
                    label=f"collected artifact {relative}",
                )
                if (
                    digest != artifact["sha256"]
                    or byte_count != artifact["bytes"]
                ):
                    raise MsctlError(
                        "FLEET_COLLECTION_INVALID",
                        "collected evaluation artifact differs from its receipt",
                        details={"path": relative},
                    )
                expected_files[relative] = {
                    "path": relative,
                    "bytes": byte_count,
                    "sha256": digest,
                }
            sealed_inventory = {
                member: str(artifacts_by_name[member]["sha256"])
                for member in REQUIRED_SEALED_MEMBERS
            }
            fixture_inventory = {
                member: sealed_inventory[member]
                for member in SEALED_FIXTURE_MEMBERS
            }
            if (
                sealed_inventory["study-lock.json"]
                != evaluation["study_lock_sha256"]
                or canonical_sha256(
                    {
                        "schema_version": 1,
                        "members": dict(sorted(sealed_inventory.items())),
                    }
                )
                != evaluation["sealed_evaluation_sha256"]
                or sealed_fixture_sha256(fixture_inventory)
                != evaluation["sealed_fixture_sha256"]
            ):
                raise MsctlError(
                    "FLEET_COLLECTION_INVALID",
                    "collected finalized evaluator provenance does not match "
                    "the paired evaluation receipt",
                    details={"arm": arm},
                )
        collection_payload = self._read_collection_regular(
            resolved / "COLLECTION.json",
            label="collection receipt",
        )
        try:
            collection_value = json.loads(collection_payload.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MsctlError(
                "FLEET_COLLECTION_INVALID",
                "collection receipt is not valid canonical JSON",
            ) from error
        expected_rows = [
            expected_files[path] for path in sorted(expected_files)
        ]
        if (
            not isinstance(collection_value, dict)
            or collection_value
            != {"schema_version": 1, "files": expected_rows}
            or collection_payload != canonical_json(collection_value) + b"\n"
        ):
            raise MsctlError(
                "FLEET_COLLECTION_INVALID",
                "collection receipt does not enumerate the exact paired lifecycle",
            )
        actual_files: set[str] = set()
        for path in resolved.rglob("*"):
            metadata = path.stat(follow_symlinks=False)
            if path.is_symlink() or not (
                stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
            ):
                raise MsctlError(
                    "FLEET_COLLECTION_INVALID",
                    "collection contains a symlink or special member",
                )
            if stat.S_ISREG(metadata.st_mode):
                actual_files.add(path.relative_to(resolved).as_posix())
        if actual_files != set(expected_files) | {"COLLECTION.json"}:
            raise MsctlError(
                "FLEET_COLLECTION_INVALID",
                "collection file inventory is not exact",
            )
        return {
            "collection_receipt_sha256": hashlib.sha256(
                collection_payload
            ).hexdigest(),
            "checkpoint_receipt_sha256": checkpoint_sha256,
            "checkpoint_receipt_uri": checkpoint_uri,
            "evaluation_receipt_sha256": evaluation_sha256,
            "evaluation_receipt_uri": (
                f"{self.runtime.s3_root.rstrip('/')}/evaluations/seed-{seed}/"
                f"receipts/{evaluation_sha256}.json"
            ),
            "evaluation": evaluation,
            "checkpoint": checkpoint,
            "files": expected_rows,
            "source": canonical_source,
        }

    def collect(
        self,
        *,
        source: str,
        out: Path | str,
        apply: bool,
    ) -> dict[str, object]:
        if self.profile.provider in _V3_PROFILES:
            destination = Path(out).absolute()
            if destination.exists() or destination.is_symlink():
                if destination.is_symlink() or not destination.is_dir():
                    raise MsctlError(
                        "COLLECT_DESTINATION_EXISTS",
                        "v3 collection destination must be a real directory",
                    )
                details = self._verify_v3_lifecycle_collection(
                    destination,
                    source=source,
                )
                return {
                    "provider": self.profile.provider,
                    "s3_uri": f"{self.runtime.s3_root.rstrip('/')}/{source}",
                    "out": str(destination),
                    "collection_receipt_sha256": details[
                        "collection_receipt_sha256"
                    ],
                    "evaluation_receipt_sha256": details[
                        "evaluation_receipt_sha256"
                    ],
                    "collected": 0,
                    "idempotent": True,
                }
            bucket, key = self._s3_location(source)
            first_argv = self._aws_argv(
                "s3api",
                "get-object",
                "--bucket",
                bucket,
                "--key",
                key,
                "--checksum-mode",
                "ENABLED",
                str(destination / "EVALUATION.json"),
                query=(
                    "{object:{checksum_sha256:ChecksumSHA256,"
                    "version_id:VersionId}}"
                ),
            )
            if not apply:
                return {
                    "provider": self.profile.provider,
                    "s3_uri": f"s3://{bucket}/{key}",
                    "out": str(destination),
                    "commands": [first_argv],
                    "collected": 0,
                    "idempotent": False,
                }
            parent_fd = open_directory(
                destination.parent,
                label="v3 collection destination",
                create=True,
            )
            os.close(parent_fd)
            staging = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.",
                    dir=destination.parent,
                )
            )
            os.chmod(staging, 0o700)
            commands: list[list[str]] = []
            try:
                argv, evaluation_identity = self._collection_get_object(
                    relative=source,
                    destination=staging / "EVALUATION.json",
                    label="paired evaluation receipt",
                )
                commands.append(argv)
                evaluation_payload = self._read_collection_regular(
                    staging / "EVALUATION.json",
                    label="paired evaluation receipt",
                )
                try:
                    evaluation = verify_evaluation_receipt_bytes(
                        evaluation_payload,
                        expected={
                            "provider": self.profile.provider,
                            "profile_sha256": self.profile.sha256,
                        },
                    )
                except (TypeError, ValueError) as error:
                    raise MsctlError(
                        "COLLECT_INCOMPLETE",
                        "source is not a closed paired evaluation receipt",
                    ) from error
                if source != f"results/seed-{evaluation['seed']}.json":
                    raise MsctlError(
                        "COLLECT_INCOMPLETE",
                        "source path does not match the evaluation seed",
                    )
                checkpoint_binding = evaluation["checkpoint_receipt"]
                checkpoint_relative = self._runtime_relative_uri(
                    checkpoint_binding["uri"],
                    label="checkpoint receipt URI",
                )
                argv, checkpoint_identity = self._collection_get_object(
                    relative=checkpoint_relative,
                    destination=staging / "CHECKPOINT.json",
                    label="paired checkpoint receipt",
                    expected_sha256=str(checkpoint_binding["sha256"]),
                )
                commands.append(argv)
                identities: dict[str, dict[str, object]] = {
                    "EVALUATION.json": {
                        **evaluation_identity,
                        "path": "EVALUATION.json",
                    },
                    "CHECKPOINT.json": {
                        **checkpoint_identity,
                        "path": "CHECKPOINT.json",
                    },
                }
                for arm_row in evaluation["arms"]:
                    run_id = str(arm_row["run_id"])
                    for artifact in arm_row["artifacts"]:
                        name = str(artifact["path"])
                        relative = f"{run_id}/{name}"
                        artifact_relative = self._runtime_relative_uri(
                            artifact["uri"],
                            label=f"evaluation artifact {relative}",
                        )
                        argv, identity = self._collection_get_object(
                            relative=artifact_relative,
                            destination=staging / run_id / name,
                            label=f"evaluation artifact {relative}",
                            expected_sha256=str(artifact["sha256"]),
                            expected_bytes=int(artifact["bytes"]),
                        )
                        commands.append(argv)
                        identities[relative] = {
                            **identity,
                            "path": relative,
                        }
                rows = [
                    {
                        "path": relative,
                        "bytes": int(identities[relative]["bytes"]),
                        "sha256": str(identities[relative]["sha256"]),
                    }
                    for relative in sorted(identities)
                ]
                receipt_payload = (
                    canonical_json({"schema_version": 1, "files": rows}) + b"\n"
                )
                stage_fd = open_directory(
                    staging,
                    label="v3 collection staging",
                )
                try:
                    atomic_write_at(
                        stage_fd,
                        "COLLECTION.json",
                        receipt_payload,
                        label="v3 collection receipt",
                    )
                finally:
                    os.close(stage_fd)
                details = self._verify_v3_lifecycle_collection(
                    staging,
                    source=source,
                )
                parent_fd = open_directory(
                    destination.parent,
                    label="v3 collection destination",
                )
                try:
                    try:
                        rename_noreplace_at(
                            parent_fd,
                            staging.name,
                            parent_fd,
                            destination.name,
                        )
                    except FileExistsError:
                        existing = self._verify_v3_lifecycle_collection(
                            destination,
                            source=source,
                        )
                        if (
                            existing["evaluation_receipt_sha256"]
                            != details["evaluation_receipt_sha256"]
                        ):
                            raise MsctlError(
                                "COLLECT_DESTINATION_EXISTS",
                                "concurrent collection destination conflicts",
                            )
                        shutil.rmtree(staging)
                        return {
                            "provider": self.profile.provider,
                            "s3_uri": f"s3://{bucket}/{key}",
                            "out": str(destination),
                            "collection_receipt_sha256": existing[
                                "collection_receipt_sha256"
                            ],
                            "evaluation_receipt_sha256": existing[
                                "evaluation_receipt_sha256"
                            ],
                            "commands": commands,
                            "collected": 0,
                            "idempotent": True,
                        }
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
                staging = Path()
                return {
                    "provider": self.profile.provider,
                    "s3_uri": f"s3://{bucket}/{key}",
                    "out": str(destination),
                    "collection_receipt_sha256": details[
                        "collection_receipt_sha256"
                    ],
                    "evaluation_receipt_sha256": details[
                        "evaluation_receipt_sha256"
                    ],
                    "commands": commands,
                    "collected": len(rows),
                    "idempotent": False,
                }
            finally:
                if staging != Path() and staging.exists():
                    shutil.rmtree(staging)

        bucket, key = self._s3_location(source)
        destination = Path(out).absolute()
        if destination.exists() or destination.is_symlink():
            raise MsctlError(
                "COLLECT_DESTINATION_EXISTS",
                "refusing to replace an existing collection destination",
            )
        argv = self._aws_argv(
            "s3api",
            "get-object",
            "--bucket",
            bucket,
            "--key",
            key,
            str(destination),
            query="{object:{etag:ETag,version_id:VersionId}}",
        )
        if not apply:
            return {
                "provider": self.profile.provider,
                "s3_uri": f"s3://{bucket}/{key}",
                "out": str(destination),
                "commands": [argv],
                "collected": 0,
            }
        destination.parent.mkdir(parents=True, exist_ok=True)
        output = _aws_output_object(
            self._run(argv, operation="collect"),
            {"object"},
            label="S3 get-object output",
        )
        _aws_output_object(
            output["object"],
            {"etag", "version_id"},
            label="S3 downloaded object",
        )
        if not destination.is_file() or destination.is_symlink():
            raise MsctlError(
                "COLLECT_INCOMPLETE",
                "AWS CLI did not materialize the requested regular file",
            )
        return {
            "provider": self.profile.provider,
            "s3_uri": f"s3://{bucket}/{key}",
            "out": str(destination),
            "collected": 1,
        }

    def _fleet_state_tag_binding(
        self,
        manifest: object,
        state: Mapping[str, object],
    ) -> dict[str, object]:
        values = {
            "provider": state.get("provider"),
            "profile_instance_type": state.get("instance_type"),
            "seed": state.get("seed"),
            "cohort_sha256": getattr(
                manifest,
                "cohort_assignment_sha256",
                None,
            ),
            "release_sha256": state.get("release_sha256"),
            "dataset_sha256": state.get("dataset_sha256"),
            "run_manifest_sha256": state.get("run_manifest_sha256"),
            "profile_sha256": state.get("profile_sha256"),
            "runtime_sha256": state.get("runtime_sha256"),
            "container_digest": state.get("container_digest"),
            "gres": state.get("gres"),
            "terminate_at": state.get("terminate_at"),
        }
        if _is_v3_manifest(manifest):
            values.update(
                {
                    field: state.get(field)
                    for field in (
                        "preregistration_sha256",
                        "hardware_amendment_sha256",
                        "provider_selection_sha256",
                        "sealed_fixture_sha256",
                        "fleet_plan_sha256",
                        "fleet_wave",
                        "launch_readiness_sha256",
                        "control_bundle_sha256",
                    )
                }
            )
        if set(values) != set(self._instance_tag_names(manifest)) or any(
            value is None for value in values.values()
        ):
            raise MsctlError(
                "FLEET_ADVANCE_INVALID",
                "prior local state does not contain one complete AWS tag binding",
            )
        return values

    def _verify_state_terminal_receipt(
        self,
        state: Mapping[str, object],
    ) -> str:
        operation_id = state.get("operation_id")
        intent_sha256 = state.get("intent_sha256")
        require_sha256(operation_id, label="fleet operation ID")
        require_sha256(intent_sha256, label="fleet intent SHA-256")
        uri = (
            f"{self.runtime.s3_root}/operations/{operation_id}/"
            "receipts/terminal.json"
        )
        bucket, key = self._s3_location(
            uri.removeprefix(f"{self.runtime.s3_root}/")
        )
        argv = self._aws_argv(
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--checksum-mode",
            "ENABLED",
            query=(
                "{receipt:{checksum_sha256:ChecksumSHA256,"
                "content_length:ContentLength,metadata:Metadata,"
                "version_id:VersionId}}"
            ),
        )
        output = _aws_output_object(
            self._run(argv, operation="verify fleet terminal receipt"),
            {"receipt"},
            label="fleet terminal receipt",
        )
        receipt = _aws_output_object(
            output["receipt"],
            {"checksum_sha256", "content_length", "metadata", "version_id"},
            label="fleet terminal receipt",
        )
        metadata = _aws_output_object(
            receipt["metadata"],
            {"operation-id", "intent-sha256", "receipt-kind"},
            label="fleet terminal receipt metadata",
        )
        if (
            not isinstance(receipt["checksum_sha256"], str)
            or not receipt["checksum_sha256"]
            or type(receipt["content_length"]) is not int
            or receipt["content_length"] <= 0
            or not isinstance(receipt["version_id"], str)
            or not receipt["version_id"]
            or metadata
            != {
                "operation-id": operation_id,
                "intent-sha256": intent_sha256,
                "receipt-kind": "terminal",
            }
        ):
            raise MsctlError(
                "FLEET_ADVANCE_INVALID",
                "remote terminal receipt does not bind the prior operation",
            )
        return uri

    def _terminal_checkpoint_identity(
        self,
        receipt_path: Path | str,
        *,
        manifest: object,
    ) -> dict[str, object]:
        try:
            payload = self._read_collection_regular(
                Path(receipt_path),
                label="terminal checkpoint receipt",
                maximum=1 << 20,
            )
            receipt = verify_checkpoint_receipt_bytes(
                payload,
                expected={
                    "provider": self.profile.provider,
                    "release_sha256": manifest.release_sha256,
                    "run_manifest_sha256": manifest.sha256,
                    "dataset_sha256": manifest.dataset_sha256,
                    "source_commit": manifest.source_commit,
                    "cohort_assignment_sha256": (
                        manifest.cohort_assignment_sha256
                    ),
                    "preregistration_sha256": (
                        manifest.preregistration_sha256
                    ),
                    "hardware_amendment_sha256": (
                        manifest.hardware_amendment_sha256
                    ),
                    "provider_selection_sha256": (
                        manifest.provider_selection_sha256
                    ),
                    "profile_sha256": manifest.profile_sha256,
                    "sealed_fixture_sha256": manifest.sealed_fixture_sha256,
                },
                require_durable=True,
            )
        except (MsctlError, OSError, TypeError, ValueError) as error:
            raise MsctlError(
                "FLEET_ADVANCE_INVALID",
                "training-wave advance requires one canonical schema-v3 "
                "terminal checkpoint receipt",
            ) from error
        digest = hashlib.sha256(payload).hexdigest()
        seed = manifest.seed
        receipt_uri = (
            f"{self.runtime.s3_root.rstrip('/')}/checkpoints/seed-{seed}/"
            f"receipts/{digest}.json"
        )
        manifest_runs = {
            str(run.arm): run for run in getattr(manifest, "runs", ())
        }
        rows = {
            str(row["arm"]): row for row in receipt["checkpoints"]
        }
        if (
            set(manifest_runs) != {"dense", "split90"}
            or set(rows) != set(manifest_runs)
            or any(
                rows[arm]["run_id"] != manifest_runs[arm].run_id
                or rows[arm]["config_sha256"]
                != manifest_runs[arm].config_sha256
                or rows[arm]["seed"] != seed
                for arm in rows
            )
        ):
            raise MsctlError(
                "FLEET_ADVANCE_INVALID",
                "terminal checkpoint receipt does not bind the manifest pair",
            )
        for row in rows.values():
            for field in (
                "checkpoint_uri",
                "configuration_uri",
                "run_binding_uri",
                "checkpoint_record_uri",
            ):
                try:
                    self._runtime_relative_uri(
                        row[field],
                        label=f"terminal checkpoint {field}",
                    )
                except MsctlError as error:
                    raise MsctlError(
                        "FLEET_ADVANCE_INVALID",
                        "terminal checkpoint artifacts are outside the "
                        "immutable runtime root",
                    ) from error
        records = [
            {
                "run_id": str(rows[arm]["run_id"]),
                "arm": arm,
                "sha256": str(rows[arm]["checkpoint_record_sha256"]),
                "uri": str(rows[arm]["checkpoint_record_uri"]),
            }
            for arm in ("dense", "split90")
        ]
        return {
            "payload": payload,
            "receipt": receipt,
            "sha256": digest,
            "uri": receipt_uri,
            "records": records,
        }

    def _materialize_terminal_checkpoint_identity(
        self,
        identity: Mapping[str, object],
        *,
        manifest: object,
    ) -> list[list[str]]:
        receipt = identity["receipt"]
        if not isinstance(receipt, dict):
            raise MsctlError(
                "FLEET_ADVANCE_INVALID",
                "terminal checkpoint identity is unavailable",
            )
        commands: list[list[str]] = []
        with tempfile.TemporaryDirectory(
            prefix="memorysplit-terminal-verify-"
        ) as temporary:
            root = Path(temporary)
            try:
                receipt_relative = self._runtime_relative_uri(
                    identity["uri"],
                    label="terminal checkpoint receipt URI",
                )
                argv, _ = self._collection_get_object(
                    relative=receipt_relative,
                    destination=root / "receipt.json",
                    label="terminal checkpoint receipt",
                    expected_sha256=str(identity["sha256"]),
                )
                commands.append(argv)
                for raw in receipt["checkpoints"]:
                    if not isinstance(raw, dict):
                        raise ValueError("checkpoint row is unavailable")
                    arm = str(raw["arm"])
                    run_root = root / "runs" / arm / "run"
                    artifacts = (
                        (
                            "checkpoint_uri",
                            "sha256",
                            run_root / "ckpt.pt",
                            "checkpoint",
                        ),
                        (
                            "configuration_uri",
                            "config_sha256",
                            run_root / "configuration.yaml",
                            "configuration",
                        ),
                        (
                            "run_binding_uri",
                            "run_binding_sha256",
                            run_root / "run.json",
                            "run binding",
                        ),
                        (
                            "checkpoint_record_uri",
                            "checkpoint_record_sha256",
                            root
                            / "records"
                            / f"{raw['checkpoint_record_sha256']}.json",
                            "checkpoint record",
                        ),
                    )
                    for uri_field, digest_field, destination, label in artifacts:
                        relative = self._runtime_relative_uri(
                            raw[uri_field],
                            label=f"{arm} {label} URI",
                        )
                        argv, _ = self._collection_get_object(
                            relative=relative,
                            destination=destination,
                            label=f"{arm} terminal {label}",
                            expected_sha256=str(raw[digest_field]),
                        )
                        commands.append(argv)
                verified = verify_materialized_terminal(
                    receipt_path=root / "receipt.json",
                    run_root=root / "runs",
                    record_root=root / "records",
                    expected_receipt_sha256=str(identity["sha256"]),
                    expected={
                        "provider": self.profile.provider,
                        "run_manifest_sha256": manifest.sha256,
                        "sealed_fixture_sha256": (
                            manifest.sealed_fixture_sha256
                        ),
                    },
                )
            except (MsctlError, OSError, TypeError, ValueError) as error:
                raise MsctlError(
                    "FLEET_ADVANCE_INVALID",
                    "immutable terminal checkpoint bundle could not be "
                    "rematerialized and verified",
                ) from error
        if verified["sha256"] != identity["sha256"]:
            raise MsctlError(
                "FLEET_ADVANCE_INVALID",
                "rematerialized terminal checkpoint receipt changed",
            )
        return commands

    def fleet_advance(
        self,
        *,
        amendment_path: Path | str,
        provider_selection_path: Path | str,
        fleet_plan_path: Path | str,
        target_manifest_path: Path | str,
        repo_root: Path | str,
        instance_id: str,
        checkpoint_receipt_path: Path | str,
        approval_path: Path | str | None,
        apply: bool,
    ) -> dict[str, object]:
        """Authorize and record one safe same-instance wave transition."""

        amendment = load_hardware_amendment(amendment_path)
        selection = load_provider_selection(
            provider_selection_path,
            amendment=amendment,
            profile=self.profile,
        )
        plan = load_fleet_plan(
            fleet_plan_path,
            profile=self.profile,
            selection=selection,
        )
        target = load_run_manifest(target_manifest_path, repo_root=repo_root)
        from_binding, to_binding = fleet_transition_for_target(
            plan,
            instance_id=instance_id,
            target=target,
        )
        previous = load_run_manifest(from_binding.path, repo_root=repo_root)
        validate_fleet_manifest(
            plan,
            previous,
            instance_id=instance_id,
        )
        checkpoint_identity = self._terminal_checkpoint_identity(
            checkpoint_receipt_path,
            manifest=previous,
        )
        existing = load_fleet_advance(
            self.state_root,
            plan=plan,
            to_binding=to_binding,
        )
        if existing is not None:
            if (
                existing.evidence.get("checkpoint_receipt_sha256")
                != checkpoint_identity["sha256"]
                or existing.evidence.get("checkpoint_receipt_uri")
                != checkpoint_identity["uri"]
                or existing.evidence.get("checkpoint_records")
                != checkpoint_identity["records"]
            ):
                raise MsctlError(
                    "FLEET_ADVANCE_INVALID",
                    "existing training-wave advance binds a different "
                    "terminal checkpoint bundle",
                )
            return {
                "provider": self.profile.provider,
                "instance_id": instance_id,
                "from_seed": from_binding.seed,
                "to_seed": to_binding.seed,
                "fleet_plan_sha256": plan.sha256,
                "advance_receipt_sha256": existing.sha256,
                "advance_receipt": str(existing.path),
                "advanced": 0,
                "idempotent": True,
            }
        previous_context = V3LifecycleContext(
            amendment=amendment,
            selection=selection,
            fleet_plan=plan,
            fleet_binding=from_binding,
        )
        store = StateStore(self.state_root)
        with store.locked():
            self._repair_paired_states(store, previous, previous_context)
            states = [store.read_run(run.run_id) for run in previous.runs]
            if any(state is None for state in states):
                raise MsctlError(
                    "FLEET_ADVANCE_INVALID",
                    "fleet advance requires complete prior paired local state",
                )
            paired = [state for state in states if state is not None]
            if (
                not all(
                    self._same_state_binding(
                        state,
                        previous,
                        previous_context,
                    )
                    for state in paired
                )
                or {state.get("status") for state in paired} != {"Success"}
                or len({state.get("command_id") for state in paired}) != 1
                or None in {state.get("command_id") for state in paired}
                or len({state.get("operation_id") for state in paired}) != 1
                or len({state.get("intent_sha256") for state in paired}) != 1
                or len(
                    {
                        state.get("launch_readiness_sha256")
                        for state in paired
                    }
                )
                != 1
            ):
                raise MsctlError(
                    "FLEET_ADVANCE_INVALID",
                    "prior paired state is not terminal or has divergent provenance",
                )
            training_state_sha256 = canonical_sha256(
                {
                    "states": sorted(
                        paired,
                        key=lambda state: str(state["run_id"]),
                    )
                }
            )
            bound_tags = self._fleet_state_tag_binding(
                previous,
                paired[0],
            )
        unbound_tags = {field: None for field in bound_tags}
        delete_tags = canonical_json(
            [
                {
                    "Key": self._instance_tag_names(previous)[field],
                    "Value": str(bound_tags[field]),
                }
                for field in self._instance_tag_names(previous)
            ]
        ).decode("ascii")
        delete_argv = self._aws_argv(
            "ec2",
            "delete-tags",
            "--resources",
            instance_id,
            "--tags",
            delete_tags,
            query="{}",
        )
        resources = {
            "schema_version": 1,
            "operation": "fleet-advance",
            "provider": self.profile.provider,
            "instance_type": self.profile.instance_type,
            "profile_sha256": self.profile.sha256,
            "gres": _profile_gres(self.profile),
            "fleet_plan_sha256": plan.sha256,
            "provider_selection_sha256": selection.sha256,
            "instance_id": instance_id,
            "from_seed": from_binding.seed,
            "from_wave": from_binding.wave,
            "from_manifest_sha256": from_binding.sha256,
            "to_seed": to_binding.seed,
            "to_wave": to_binding.wave,
            "to_manifest_sha256": to_binding.sha256,
            "training_state_sha256": training_state_sha256,
            "checkpoint_receipt_sha256": checkpoint_identity["sha256"],
            "checkpoint_records_sha256": canonical_sha256(
                checkpoint_identity["records"]
            ),
            "jobs": 0,
            "allocated_gpus": 0,
            "wall_minutes": 0,
            "gpu_hours": 0,
            "script": "aws-fleet-advance",
        }
        result = {
            "provider": self.profile.provider,
            "instance_id": instance_id,
            "from_seed": from_binding.seed,
            "to_seed": to_binding.seed,
            "fleet_plan_sha256": plan.sha256,
            "resources": resources,
            "commands": [delete_argv],
            "advanced": 0,
            "idempotent": False,
        }
        if not apply:
            return result
        if approval_path is None:
            raise MsctlError(
                "APPROVAL_REQUIRED",
                "fleet advance --apply requires explicit signed approval",
            )
        self.approval_verifier(
            path=approval_path,
            operation="fleet-advance",
            release_sha256=plan.release_sha256,
            scope_sha256=plan.sha256,
            resources=resources,
            profile=self.profile,
            environ=self.environ,
        )
        training_command_id = str(paired[0]["command_id"])
        if self._command_status(instance_id, training_command_id) != "Success":
            raise MsctlError(
                "FLEET_ADVANCE_INVALID",
                "authoritative AWS training is not successfully terminal",
            )
        training_terminal_uri = self._verify_state_terminal_receipt(paired[0])
        terminate_at = str(paired[0]["terminate_at"])
        selected_argv = self._selected_instance_argv(instance_id, previous)
        observed = self._parse_selected_instance(
            self._run(
                selected_argv,
                operation="verify fleet advance binding",
            ),
            previous,
            instance_id=instance_id,
            terminate_at=terminate_at,
            require_bound=True,
            expected_binding=bound_tags,
        )
        del observed
        materialization_commands = (
            self._materialize_terminal_checkpoint_identity(
                checkpoint_identity,
                manifest=previous,
            )
        )
        _aws_output_object(
            self._run(delete_argv, operation="unbind completed fleet wave"),
            set(),
            label="EC2 delete-tags output",
        )
        self._parse_selected_instance(
            self._run(
                selected_argv,
                operation="verify fleet wave unbound",
            ),
            previous,
            instance_id=instance_id,
            terminate_at=terminate_at,
            require_bound=True,
            expected_binding=unbound_tags,
        )
        evidence = {
            "training_state_sha256": training_state_sha256,
            "training_command_id": training_command_id,
            "training_terminal_receipt_uri": training_terminal_uri,
            "checkpoint_receipt_sha256": checkpoint_identity["sha256"],
            "checkpoint_receipt_uri": checkpoint_identity["uri"],
            "checkpoint_records": checkpoint_identity["records"],
            "aws_bound_tags_sha256": canonical_sha256(bound_tags),
            "aws_unbound_tags_sha256": canonical_sha256(unbound_tags),
        }
        receipt = create_fleet_advance(
            plan=plan,
            instance_id=instance_id,
            from_binding=from_binding,
            to_binding=to_binding,
            evidence=evidence,
            approval_sha256=sha256_file(approval_path),
            advanced_at=_timestamp(),
        )
        receipt_path = write_fleet_advance(
            self.state_root,
            plan=plan,
            to_binding=to_binding,
            value=receipt,
        )
        loaded = load_fleet_advance(
            self.state_root,
            plan=plan,
            to_binding=to_binding,
        )
        assert loaded is not None
        return {
            **result,
            "commands": [*materialization_commands, delete_argv],
            "advance_receipt": str(receipt_path),
            "advance_receipt_sha256": loaded.sha256,
            "advanced": 1,
        }

    def _new_aws_run_state(
        self,
        *,
        run: object,
        manifest: object,
        operation: str,
        instance_id: str,
        terminate_at: str,
        intent: Mapping[str, object],
        published: Mapping[str, object],
        attempt: int,
        checkpoint_receipt_sha256: str | None = None,
        prior_command_ids: list[str] | None = None,
        context: V3LifecycleContext | None = None,
    ) -> dict[str, object]:
        now = _timestamp()
        state: dict[str, object] = {
            "schema_version": 3 if _is_v3_manifest(manifest) else 1,
            "run_id": run.run_id,
            "arm": run.arm,
            "seed": run.seed,
            "provider": self.profile.provider,
            "instance_type": self.profile.instance_type,
            "gres": _profile_gres(self.profile),
            "release_sha256": manifest.release_sha256,
            "run_manifest_sha256": manifest.sha256,
            "config_sha256": run.config_sha256,
            "dataset_sha256": manifest.dataset_sha256,
            "dataset_pointer_sha256": intent["dataset_pointer_sha256"],
            "dataset_verification_sha256": intent[
                "dataset_verification_sha256"
            ],
            "environment_receipt_sha256": intent[
                "environment_receipt_sha256"
            ],
            "cohort_assignment_sha256": manifest.cohort_assignment_sha256,
            "source_commit": manifest.source_commit,
            "profile_sha256": self.profile.sha256,
            "runtime_sha256": self._runtime_sha256(),
            "ami_id": self.runtime.ami_id,
            "container_digest": self.runtime.container_digest,
            "instance_id": instance_id,
            "terminate_at": terminate_at,
            "operation_id": intent["operation_id"],
            "intent_sha256": published["intent_sha256"],
            "intent_uri": published["intent_uri"],
            "command_id": None,
            "operation": operation,
            "status": "INTENT_PUBLISHED",
            "attempt": attempt,
            "send_attempted": False,
            "created_at": now,
            "updated_at": now,
        }
        if _is_v3_manifest(manifest):
            state.update(self._v3_bindings(manifest, context))
        else:
            state["study_lock_sha256"] = manifest.study_lock_sha256
        if operation == "resume":
            state["checkpoint_receipt_sha256"] = checkpoint_receipt_sha256
            state["prior_command_ids"] = list(prior_command_ids or [])
        return state

    def _write_paired_states(
        self,
        store: StateStore,
        manifest: object,
        states: Sequence[dict[str, object]],
        context: V3LifecycleContext | None = None,
    ) -> None:
        if len(states) != 2 or len({state["run_id"] for state in states}) != 2:
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS pair update must contain exactly two distinct arms",
            )
        operation_ids = {state["operation_id"] for state in states}
        if len(operation_ids) != 1:
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS pair update must bind one operation",
            )
        pair = {
            "schema_version": 3 if _is_v3_manifest(manifest) else 1,
            "provider": self.profile.provider,
            "instance_type": self.profile.instance_type,
            "profile_sha256": self.profile.sha256,
            "gres": _profile_gres(self.profile),
            "run_manifest_sha256": manifest.sha256,
            "operation_id": next(iter(operation_ids)),
            "states": [dict(state) for state in states],
        }
        if _is_v3_manifest(manifest):
            pair.update(self._v3_bindings(manifest, context))
        store.write_aws_pair(
            manifest.sha256,
            pair,
        )
        for state in states:
            store.write_run(str(state["run_id"]), state)

    def _repair_paired_states(
        self,
        store: StateStore,
        manifest: object,
        context: V3LifecycleContext | None = None,
    ) -> None:
        journal = store.read_aws_pair(manifest.sha256)
        if journal is None:
            return
        if (
            journal.get("provider") != self.profile.provider
            or journal.get("instance_type") != self.profile.instance_type
            or journal.get("profile_sha256") != self.profile.sha256
            or journal.get("gres") != _profile_gres(self.profile)
            or (
                _is_v3_manifest(manifest)
                and any(
                    journal.get(field) != expected
                    for field, expected in self._v3_bindings(
                        manifest,
                        context,
                    ).items()
                )
            )
        ):
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS pair journal belongs to a different provider profile",
            )
        expected = {
            str(state["run_id"]): state
            for state in journal["states"]
            if isinstance(state, dict)
        }
        if set(expected) != {run.run_id for run in manifest.runs}:
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS pair intent names the wrong run pair",
            )
        for run_id, state in expected.items():
            current = store.read_run(run_id)
            if current is None:
                store.write_run(run_id, dict(state))
            elif current != state:
                raise MsctlError(
                    "STATE_CORRUPT",
                    "AWS pair state conflicts with its durable repair intent",
                )

    def _operation_receipt_exists(
        self,
        intent: Mapping[str, object],
        *,
        kind: str,
    ) -> bool:
        uri = intent.get(f"{kind}_receipt_uri")
        prefix = f"{self.runtime.s3_root}/"
        if not isinstance(uri, str) or not uri.startswith(prefix):
            raise MsctlError(
                "STATE_CORRUPT",
                "operation receipt URI is not under the pinned S3 root",
            )
        bucket, key = self._s3_location(uri.removeprefix(prefix))
        argv = self._aws_argv(
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--checksum-mode",
            "ENABLED",
            query=(
                "{receipt:{checksum_sha256:ChecksumSHA256,"
                "content_length:ContentLength,metadata:Metadata,"
                "version_id:VersionId}}"
            ),
        )
        try:
            output = _aws_output_object(
                self._run(argv, operation=f"reconcile {kind} receipt"),
                {"receipt"},
                label=f"S3 {kind} receipt",
            )
        except MsctlError as error:
            if error.code == "AWS_COMMAND_FAILED":
                return False
            raise
        receipt = _aws_output_object(
            output["receipt"],
            {"checksum_sha256", "content_length", "metadata", "version_id"},
            label=f"S3 {kind} receipt metadata",
        )
        metadata = _aws_output_object(
            receipt["metadata"],
            {"operation-id", "intent-sha256", "receipt-kind"},
            label=f"S3 {kind} receipt binding",
        )
        expected_intent_sha256 = hashlib.sha256(
            canonical_json(intent)
        ).hexdigest()
        if (
            not isinstance(receipt["checksum_sha256"], str)
            or not receipt["checksum_sha256"]
            or isinstance(receipt["content_length"], bool)
            or not isinstance(receipt["content_length"], int)
            or receipt["content_length"] <= 0
            or not isinstance(receipt["version_id"], str)
            or not receipt["version_id"]
            or metadata
            != {
                "operation-id": intent["operation_id"],
                "intent-sha256": expected_intent_sha256,
                "receipt-kind": kind,
            }
        ):
            raise MsctlError(
                "REMOTE_RECEIPT_INVALID",
                "remote operation receipt metadata is incomplete",
            )
        return True

    def _find_operation_command(
        self,
        *,
        operation_id: str,
        instance_id: str,
    ) -> dict[str, str] | None:
        argv = self._aws_argv(
            "ssm",
            "list-commands",
            query=(
                "{commands:Commands[?Comment=='"
                + operation_id
                + "']."
                "{command_id:CommandId,status:Status,comment:Comment,"
                "instance_ids:InstanceIds}}"
            ),
        )
        output = _aws_output_object(
            self._run(argv, operation="reconcile SSM operation"),
            {"commands"},
            label="SSM operation commands",
        )
        rows = _aws_output_list(
            output["commands"],
            label="SSM operation commands",
        )
        if len(rows) > 1:
            raise MsctlError(
                "DUPLICATE_REMOTE_OPERATION",
                "multiple SSM commands claim one deterministic operation",
            )
        if not rows:
            return None
        row = _aws_output_object(
            rows[0],
            {"command_id", "status", "comment", "instance_ids"},
            label="SSM operation command",
        )
        command_id = row["command_id"]
        if (
            not isinstance(command_id, str)
            or _COMMAND_ID_RE.fullmatch(command_id) is None
            or row["comment"] != operation_id
            or row["instance_ids"] != [instance_id]
            or not isinstance(row["status"], str)
            or not row["status"]
        ):
            raise MsctlError(
                "AWS_OUTPUT_INVALID",
                "SSM operation reconciliation returned a mismatched command",
            )
        return {"command_id": command_id, "status": str(row["status"])}

    def _send_operation_intent(
        self,
        *,
        instance_id: str,
        intent: Mapping[str, object],
        published: Mapping[str, object],
        operation: str,
    ) -> str:
        operation_id = str(intent["operation_id"])
        parameters = canonical_json(
            {
                "BootstrapMode": [
                    (
                        "verified-control-bundle"
                        if intent.get("schema_version") == 3
                        else "installed"
                    )
                ],
                "ControlBundleSHA256": [self.control_bundle.sha256],
                "ControlBundleURI": [
                    f"{self.runtime.s3_root}/control/"
                    f"{self.control_bundle.sha256}.tar"
                ],
                "IntentSHA256": [published["intent_sha256"]],
                "IntentUri": [published["intent_uri"]],
                "Region": [self.runtime.region],
            }
        ).decode("ascii")
        argv = self._aws_argv(
            "ssm",
            "send-command",
            "--instance-ids",
            instance_id,
            "--document-name",
            ARGV_DOCUMENT_NAME,
            "--document-hash",
            ARGV_DOCUMENT_SHA256,
            "--document-hash-type",
            "Sha256",
            "--comment",
            operation_id,
            "--parameters",
            parameters,
            query="{command:{command_id:Command.CommandId}}",
        )
        output = _aws_output_object(
            self._run(argv, operation=operation),
            {"command"},
            label="SSM send-command output",
        )
        command = _aws_output_object(
            output["command"],
            {"command_id"},
            label="SSM command",
        )
        command_id = command["command_id"]
        if (
            not isinstance(command_id, str)
            or _COMMAND_ID_RE.fullmatch(command_id) is None
        ):
            raise MsctlError(
                "AWS_OUTPUT_INVALID",
                "SSM returned an invalid command ID",
            )
        return command_id

    def submit(
        self,
        *,
        release: object,
        manifest: object,
        instance_id: str | None,
        terminate_at: str,
        approval_path: Path | str | None,
        apply: bool,
        evidence: Mapping[str, str] | None = None,
        context: V3LifecycleContext | None = None,
    ) -> dict[str, object]:
        self._validate_manifest(manifest)
        self._validate_release(release, manifest)
        validated_context = self._require_protected_launch_context(
            manifest,
            context,
        )
        if validated_context is not None:
            if (
                instance_id is not None
                and instance_id != validated_context.instance_id
            ):
                raise MsctlError(
                    "FLEET_PLAN_INVALID",
                    "submit instance differs from the protected fleet binding",
                )
            instance_id = validated_context.instance_id
            self._require_fleet_advance(manifest, validated_context)
        if not isinstance(instance_id, str):
            raise MsctlError(
                "INSTANCE_BINDING_MISMATCH",
                "submit requires one explicit or fleet-bound EC2 instance ID",
            )
        self._validate_submit_selection(instance_id, terminate_at)
        core = self._training_operation_intent(
            operation="submit",
            release=release,
            manifest=manifest,
            terminate_at=terminate_at,
            evidence=evidence,
            context=context,
        )
        operation_intent = self._operation_envelope(
            core,
            instance_id=instance_id,
            terminate_at=terminate_at,
        )
        if not apply:
            plan = self._submission_plan(
                manifest,
                release=release,
                instance_id=instance_id,
                terminate_at=terminate_at,
                evidence=evidence,
                context=context,
            )
            plan["operation_intent"] = operation_intent
            plan["operation_id"] = operation_intent["operation_id"]
            return plan
        resources = self._submit_resources(
            release=release,
            manifest=manifest,
            instance_id=instance_id,
            terminate_at=terminate_at,
            context=context,
        )
        resources.update(
            {
                field: core[field]
                for field in (
                    "dataset_pointer_sha256",
                    "dataset_verification_sha256",
                    "environment_receipt_sha256",
                )
            }
        )
        self._verify_approval(
            approval_path,
            operation="submit",
            release=release,
            manifest=manifest,
            resources=resources,
        )

        store = StateStore(self.state_root)
        with store.locked():
            self._repair_paired_states(store, manifest, context)
            states = [store.read_run(run.run_id) for run in manifest.runs]
            existing = [state for state in states if state is not None]
            if existing:
                if len(existing) != 2 or not all(
                    self._same_state_binding(state, manifest, context)
                    for state in existing
                ):
                    raise MsctlError(
                        "STATE_INCOMPLETE",
                        "AWS paired lifecycle state is incomplete or conflicting",
                    )
                instance_ids = {state.get("instance_id") for state in existing}
                deadlines = {state.get("terminate_at") for state in existing}
                operation_ids = {state.get("operation_id") for state in existing}
                intent_hashes = {state.get("intent_sha256") for state in existing}
                intent_uris = {state.get("intent_uri") for state in existing}
                if instance_ids != {instance_id} or deadlines != {terminate_at}:
                    raise MsctlError(
                        "INSTANCE_BINDING_MISMATCH",
                        "paired state differs from the operator selection",
                    )
                self._require_state_intent_binding(
                    existing,
                    operation_intent,
                )
                if (
                    len(operation_ids) != 1
                    or len(intent_hashes) != 1
                    or len(intent_uris) != 1
                ):
                    raise MsctlError(
                        "SUBMISSION_UNCERTAIN",
                        "paired state does not bind one immutable operation",
                    )
                recovered_intent = {
                    **operation_intent,
                    "operation_id": next(iter(operation_ids)),
                    "started_receipt_uri": (
                        f"{self.runtime.s3_root}/operations/"
                        f"{next(iter(operation_ids))}/receipts/started.json"
                    ),
                    "terminal_receipt_uri": (
                        f"{self.runtime.s3_root}/operations/"
                        f"{next(iter(operation_ids))}/receipts/terminal.json"
                    ),
                }
                command_ids = {state.get("command_id") for state in existing}
                if None in command_ids or len(command_ids) != 1:
                    if command_ids != {None}:
                        raise MsctlError(
                            "SUBMISSION_UNCERTAIN",
                            "paired state has divergent SSM command IDs",
                        )
                    started = self._operation_receipt_exists(
                        recovered_intent,
                        kind="started",
                    )
                    terminal = self._operation_receipt_exists(
                        recovered_intent,
                        kind="terminal",
                    )
                    recovered = self._find_operation_command(
                        operation_id=str(next(iter(operation_ids))),
                        instance_id=instance_id,
                    )
                    if recovered is None:
                        if terminal and not started:
                            raise MsctlError(
                                "REMOTE_RECEIPT_INVALID",
                                "remote terminal receipt exists without its "
                                "started acquisition receipt",
                            )
                        if terminal:
                            return {
                                "provider": self.profile.provider,
                                "seed": manifest.seed,
                                "instance_id": instance_id,
                                "operation_id": next(iter(operation_ids)),
                                "status": "REMOTE_TERMINAL",
                                "submitted": 0,
                                "idempotent": True,
                            }
                        if started:
                            now = _timestamp()
                            for state in existing:
                                state["status"] = "RECOVERY_REQUIRED"
                                state["updated_at"] = now
                            self._write_paired_states(
                                store,
                                manifest,
                                existing,
                                context,
                            )
                            raise MsctlError(
                                "REMOTE_RECOVERY_REQUIRED",
                                "remote execution was acquired but has no "
                                "terminal receipt; automatic success or resend "
                                "is forbidden",
                                details={
                                    "operation_id": next(iter(operation_ids)),
                                },
                            )
                        raise MsctlError(
                            "SUBMISSION_UNCERTAIN",
                            "send was attempted but no safe resend proof exists",
                        )
                    command_id = recovered["command_id"]
                    status = recovered["status"]
                    now = _timestamp()
                    for run, state in zip(manifest.runs, existing):
                        state["command_id"] = command_id
                        state["status"] = status
                        state["updated_at"] = now
                        store.write_run(run.run_id, state)
                    return {
                        "provider": self.profile.provider,
                        "seed": manifest.seed,
                        "instance_id": instance_id,
                        "command_id": command_id,
                        "operation_id": next(iter(operation_ids)),
                        "status": status,
                        "submitted": 0,
                        "idempotent": True,
                    }
                if len(command_ids) != 1:
                    raise MsctlError(
                        "SUBMISSION_UNCERTAIN",
                        "AWS paired intent lacks one command ID",
                    )
                command_id = str(next(iter(command_ids)))
                status = self._command_status(instance_id, command_id)
                now = _timestamp()
                for run, state in zip(manifest.runs, existing):
                    state["status"] = status
                    state["updated_at"] = now
                    store.write_run(run.run_id, state)
                return {
                    "provider": self.profile.provider,
                    "seed": manifest.seed,
                    "instance_id": instance_id,
                    "command_id": command_id,
                    "status": status,
                    "submitted": 0,
                    "idempotent": True,
                    "active": status in _ACTIVE_COMMAND_STATES,
                }

            discovered = self.discover_instances(manifest, context)
            if discovered:
                if discovered[0]["instance_id"] != instance_id:
                    raise MsctlError(
                        "INSTANCE_SELECTION_CONFLICT",
                        "an active seed instance differs from the operator selection",
                        details={
                            "selected_instance_id": instance_id,
                            "discovered_instance_id": discovered[0]["instance_id"],
                        },
                    )
                raise MsctlError(
                    "RUN_ALREADY_ACTIVE",
                    "an active instance already claims this seed without paired state",
                    details={"instance_id": discovered[0]["instance_id"]},
                )
            selected_argv = self._selected_instance_argv(instance_id, manifest)
            self._parse_selected_instance(
                self._run(
                    selected_argv,
                    operation="preflight selected instance",
                ),
                manifest,
                instance_id=instance_id,
                terminate_at=terminate_at,
                require_bound=False,
                context=context,
            )
            self._require_ssm_online(instance_id)
            self._ensure_argv_document()
            if _is_v3_manifest(manifest):
                self._publish_control_bundle()
            self._bind_selected_instance(
                manifest,
                instance_id=instance_id,
                terminate_at=terminate_at,
                context=context,
            )
            published = self._publish_operation_intent(operation_intent)
            new_states = [
                self._new_aws_run_state(
                        run=run,
                        manifest=manifest,
                        operation="submit",
                        instance_id=instance_id,
                        terminate_at=terminate_at,
                        intent=operation_intent,
                        published=published,
                        attempt=1,
                        context=context,
                )
                for run in manifest.runs
            ]
            self._write_paired_states(store, manifest, new_states, context)
            now = _timestamp()
            for state in new_states:
                state["status"] = "SENDING"
                state["send_attempted"] = True
                state["updated_at"] = now
            self._write_paired_states(store, manifest, new_states, context)
            command_id = self._send_operation_intent(
                instance_id=instance_id,
                intent=operation_intent,
                published=published,
                operation="submit",
            )
            now = _timestamp()
            for state in new_states:
                state["command_id"] = command_id
                state["status"] = "Pending"
                state["updated_at"] = now
            self._write_paired_states(store, manifest, new_states, context)
            return {
                "provider": self.profile.provider,
                "seed": manifest.seed,
                "instance_id": instance_id,
                "command_id": command_id,
                "operation_id": operation_intent["operation_id"],
                "status": "Pending",
                "submitted": 1,
                "idempotent": False,
            }

    def _checkpoint_map(
        self,
        manifest: object,
        receipt: object,
    ) -> dict[str, object]:
        checkpoints = getattr(receipt, "checkpoints", ())
        expected_schema = 3 if _is_v3_manifest(manifest) else 2
        by_run = {run.run_id: run for run in manifest.runs}
        if (
            getattr(receipt, "schema_version", None) != expected_schema
            or len(checkpoints) != 2
            or {checkpoint.run_id for checkpoint in checkpoints} != set(by_run)
        ):
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                "AWS resume requires one complete schema-matched paired receipt",
            )
        if _is_v3_manifest(manifest) and any(
            getattr(receipt, field, None) != getattr(manifest, field, None)
            for field in (
                "hardware_amendment_sha256",
                "provider_selection_sha256",
                "profile_sha256",
                "preregistration_sha256",
                "sealed_fixture_sha256",
            )
        ):
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                "v3 checkpoint receipt drops immutable lifecycle provenance",
            )
        by_arm: dict[str, object] = {}
        for checkpoint in checkpoints:
            run = by_run[checkpoint.run_id]
            if (
                checkpoint.arm != run.arm
                or checkpoint.seed != manifest.seed
                or checkpoint.world_size != 4
                or checkpoint.config_sha256 != run.config_sha256
                or checkpoint.dataset_sha256 != manifest.dataset_sha256
                or checkpoint.source_commit != manifest.source_commit
                or not isinstance(checkpoint.sha256, str)
            ):
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    "AWS checkpoint does not match its seed arm provenance",
                )
            by_arm[checkpoint.arm] = checkpoint
        if set(by_arm) != {"dense", "split90"}:
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                "AWS checkpoint receipt is not a complete pair",
            )
        return by_arm

    def resume(
        self,
        *,
        release: object,
        manifest: object,
        checkpoint_receipt: object,
        approval_path: Path | str | None,
        apply: bool,
        evidence: Mapping[str, str] | None = None,
        context: V3LifecycleContext | None = None,
    ) -> dict[str, object]:
        self._validate_manifest(manifest)
        self._validate_release(release, manifest)
        validated_context = self._require_protected_launch_context(
            manifest,
            context,
        )
        if validated_context is not None:
            self._require_fleet_advance(manifest, validated_context)
        checkpoints = self._checkpoint_map(manifest, checkpoint_receipt)
        checkpoint_publication = self._checkpoint_publication(
            checkpoint_receipt=checkpoint_receipt,
            checkpoints=checkpoints,
            apply=False,
        )
        operation_intent = self._training_operation_intent(
            operation="resume",
            release=release,
            manifest=manifest,
            checkpoints=checkpoints,
            checkpoint_receipt_sha256=checkpoint_receipt.sha256,
            evidence=evidence,
            context=context,
        )
        if not apply:
            return {
                "provider": self.profile.provider,
                "seed": manifest.seed,
                "run_manifest_sha256": manifest.sha256,
                "checkpoint_receipt_sha256": checkpoint_receipt.sha256,
                "checkpoint_commands": checkpoint_publication["commands"],
                "checkpoint_objects": checkpoint_publication["objects"],
                "operation_intent": operation_intent,
                "submitted": 0,
                "idempotent": False,
            }
        if approval_path is None:
            raise MsctlError(
                "APPROVAL_REQUIRED",
                "resume apply requires an explicit signed approval receipt",
            )

        store = StateStore(self.state_root)
        with store.locked():
            self._repair_paired_states(store, manifest, context)
            states = [store.read_run(run.run_id) for run in manifest.runs]
            if any(state is None for state in states):
                raise MsctlError(
                    "RUN_STATE_MISSING",
                    "AWS resume requires complete paired submit state",
                )
            present = [state for state in states if state is not None]
            if not all(
                self._same_state_binding(state, manifest, context)
                for state in present
            ):
                raise MsctlError(
                    "RUN_ID_CONFLICT",
                    "AWS resume state has different provenance",
                )
            if {
                state.get("environment_receipt_sha256") for state in present
            } != {operation_intent["environment_receipt_sha256"]}:
                raise MsctlError(
                    "ENVIRONMENT_RECEIPT_MISMATCH",
                    "AWS resume must reuse the authenticated launch receipt",
                )
            instance_ids = {state.get("instance_id") for state in present}
            if None in instance_ids or len(instance_ids) != 1:
                raise MsctlError(
                    "SUBMISSION_UNCERTAIN",
                    "AWS paired state does not bind one instance",
                )
            instance_id = str(next(iter(instance_ids)))
            if (
                validated_context is not None
                and instance_id != validated_context.instance_id
            ):
                raise MsctlError(
                    "FLEET_PLAN_INVALID",
                    "resume state is bound to a different fleet instance",
                )
            deadlines = {state.get("terminate_at") for state in present}
            if len(deadlines) != 1 or not isinstance(
                next(iter(deadlines)),
                str,
            ):
                raise MsctlError(
                    "SUBMISSION_UNCERTAIN",
                    "AWS resume requires one bound termination deadline",
                )
            terminate_at = str(next(iter(deadlines)))
            self._validate_submit_selection(instance_id, terminate_at)
            self._validate_selected_instance_binding(
                manifest,
                instance_id=instance_id,
                terminate_at=terminate_at,
                operation="resume",
                context=context,
            )
            resume_resources = self._resume_resources(
                release=release,
                manifest=manifest,
                instance_id=instance_id,
                terminate_at=terminate_at,
                checkpoint_receipt_sha256=checkpoint_receipt.sha256,
                context=context,
            )
            resume_resources.update(
                {
                    field: operation_intent[field]
                    for field in (
                        "dataset_pointer_sha256",
                        "dataset_verification_sha256",
                        "environment_receipt_sha256",
                    )
                }
            )
            self._verify_approval(
                approval_path,
                operation="resume",
                release=release,
                manifest=manifest,
                resources=resume_resources,
            )
            operation_intent = self._operation_envelope(
                operation_intent,
                instance_id=instance_id,
                terminate_at=terminate_at,
            )

            repeated = all(
                state.get("operation") == "resume"
                and state.get("checkpoint_receipt_sha256")
                == checkpoint_receipt.sha256
                for state in present
            )
            if repeated:
                self._require_state_intent_binding(
                    present,
                    operation_intent,
                )
                command_ids = {state.get("command_id") for state in present}
                if command_ids == {None}:
                    started = self._operation_receipt_exists(
                        operation_intent,
                        kind="started",
                    )
                    terminal = self._operation_receipt_exists(
                        operation_intent,
                        kind="terminal",
                    )
                    recovered = self._find_operation_command(
                        operation_id=str(operation_intent["operation_id"]),
                        instance_id=instance_id,
                    )
                    if recovered is None:
                        if terminal and not started:
                            raise MsctlError(
                                "REMOTE_RECEIPT_INVALID",
                                "remote terminal receipt exists without its "
                                "started acquisition receipt",
                            )
                        if terminal:
                            return {
                                "provider": self.profile.provider,
                                "seed": manifest.seed,
                                "instance_id": instance_id,
                                "operation_id": operation_intent[
                                    "operation_id"
                                ],
                                "status": "REMOTE_TERMINAL",
                                "submitted": 0,
                                "idempotent": True,
                            }
                        if started:
                            now = _timestamp()
                            for state in present:
                                state["status"] = "RECOVERY_REQUIRED"
                                state["updated_at"] = now
                            self._write_paired_states(
                                store,
                                manifest,
                                present,
                                context,
                            )
                            raise MsctlError(
                                "REMOTE_RECOVERY_REQUIRED",
                                "remote resume was acquired but has no terminal "
                                "receipt; automatic success or resend is forbidden",
                                details={
                                    "operation_id": operation_intent[
                                        "operation_id"
                                    ],
                                },
                            )
                        raise MsctlError(
                            "SUBMISSION_UNCERTAIN",
                            "resume send was attempted without safe resend proof",
                        )
                    command_id = str(recovered["command_id"])
                    status = str(recovered["status"])
                    now = _timestamp()
                    for run, state in zip(manifest.runs, present):
                        state["command_id"] = command_id
                        state["status"] = status
                        state["updated_at"] = now
                        store.write_run(run.run_id, state)
                    return {
                        "provider": self.profile.provider,
                        "seed": manifest.seed,
                        "instance_id": instance_id,
                        "command_id": command_id,
                        "operation_id": operation_intent["operation_id"],
                        "status": status,
                        "submitted": 0,
                        "idempotent": True,
                    }
                if None in command_ids or len(command_ids) != 1:
                    raise MsctlError(
                        "SUBMISSION_UNCERTAIN",
                        "AWS resume intent lacks one command ID",
                    )
                command_id = str(next(iter(command_ids)))
                status = self._command_status(instance_id, command_id)
                now = _timestamp()
                for run, state in zip(manifest.runs, present):
                    state["status"] = status
                    state["updated_at"] = now
                    store.write_run(run.run_id, state)
                return {
                    "provider": self.profile.provider,
                    "seed": manifest.seed,
                    "instance_id": instance_id,
                    "command_id": command_id,
                    "status": status,
                    "submitted": 0,
                    "idempotent": True,
                }

            previous_commands = {
                state.get("command_id") for state in present
            }
            if None in previous_commands or len(previous_commands) != 1:
                raise MsctlError(
                    "SUBMISSION_UNCERTAIN",
                    "AWS resume requires one previous paired command",
                )
            previous_command = str(next(iter(previous_commands)))
            previous_status = self._command_status(
                instance_id,
                previous_command,
            )
            if previous_status in _ACTIVE_COMMAND_STATES:
                raise MsctlError(
                    "RUN_ALREADY_ACTIVE",
                    "refusing to resume an active AWS command",
                )
            if previous_status == "Success":
                raise MsctlError(
                    "RUN_ALREADY_COMPLETE",
                    "refusing to resume a successful AWS command",
                )
            if previous_status not in {"Failed", "Cancelled", "TimedOut"}:
                raise MsctlError(
                    "RESUME_STATE_UNCERTAIN",
                    "SSM did not report an explicit resumable terminal state",
                    details={"status": previous_status},
                )
            self._require_ssm_online(instance_id)
            self._ensure_argv_document()
            if _is_v3_manifest(manifest):
                self._publish_control_bundle()
            attempt = max(int(state.get("attempt", 1)) for state in present) + 1
            self._checkpoint_publication(
                checkpoint_receipt=checkpoint_receipt,
                checkpoints=checkpoints,
                apply=True,
            )
            published = self._publish_operation_intent(operation_intent)
            now = _timestamp()
            for state in present:
                prior = list(state.get("prior_command_ids", []))
                prior.append(previous_command)
                state.update(
                    {
                        "command_id": None,
                        "operation": "resume",
                        "status": "INTENT_PUBLISHED",
                        "attempt": attempt,
                        "operation_id": operation_intent["operation_id"],
                        "intent_sha256": published["intent_sha256"],
                        "intent_uri": published["intent_uri"],
                        "dataset_pointer_sha256": operation_intent[
                            "dataset_pointer_sha256"
                        ],
                        "dataset_verification_sha256": operation_intent[
                            "dataset_verification_sha256"
                        ],
                        "environment_receipt_sha256": operation_intent[
                            "environment_receipt_sha256"
                        ],
                        "send_attempted": False,
                        "checkpoint_receipt_sha256": (
                            checkpoint_receipt.sha256
                        ),
                        "prior_command_ids": prior,
                        "updated_at": now,
                    }
                )
            self._write_paired_states(store, manifest, present, context)
            now = _timestamp()
            for state in present:
                state["status"] = "SENDING"
                state["send_attempted"] = True
                state["updated_at"] = now
            self._write_paired_states(store, manifest, present, context)
            command_id = self._send_operation_intent(
                instance_id=instance_id,
                intent=operation_intent,
                published=published,
                operation="resume",
            )
            now = _timestamp()
            for state in present:
                state["command_id"] = command_id
                state["status"] = "Pending"
                state["updated_at"] = now
            self._write_paired_states(store, manifest, present, context)
            return {
                "provider": self.profile.provider,
                "seed": manifest.seed,
                "instance_id": instance_id,
                "command_id": command_id,
                "status": "Pending",
                "attempt": attempt,
                "submitted": 1,
                "idempotent": False,
            }

    def _paired_states(
        self,
        store: StateStore,
        manifest: object,
        context: V3LifecycleContext | None = None,
    ) -> list[dict[str, object]]:
        states = [store.read_run(run.run_id) for run in manifest.runs]
        if any(state is None for state in states):
            raise MsctlError(
                "RUN_STATE_MISSING",
                "AWS lifecycle state is missing one paired arm",
            )
        present = [state for state in states if state is not None]
        if not all(
            self._same_state_binding(state, manifest, context)
            for state in present
        ):
            raise MsctlError(
                "RUN_ID_CONFLICT",
                "AWS lifecycle state has different provenance",
            )
        paired_fields = (
            "instance_id",
            "terminate_at",
            "operation",
            "operation_id",
            "intent_sha256",
            "intent_uri",
            "attempt",
            "send_attempted",
        )
        if any(
            len({state.get(field) for state in present}) != 1
            for field in paired_fields
        ):
            raise MsctlError(
                "STATE_INCOMPLETE",
                "AWS paired lifecycle state diverges between arms",
            )
        if present[0].get("operation") == "resume" and len(
            {
                state.get("checkpoint_receipt_sha256")
                for state in present
            }
        ) != 1:
            raise MsctlError(
                "STATE_INCOMPLETE",
                "AWS paired resume state binds different checkpoints",
            )
        return present

    def cancel(
        self,
        *,
        release: object,
        manifest: object,
        approval_path: Path | str | None,
        apply: bool,
        context: V3LifecycleContext | None = None,
    ) -> dict[str, object]:
        self._validate_manifest(manifest)
        self._validate_release(release, manifest)
        validated_context = self._validate_v3_context(manifest, context)
        if not apply:
            return {
                "provider": self.profile.provider,
                "seed": manifest.seed,
                "operation": "cancel",
                "run_ids": [run.run_id for run in manifest.runs],
                **self._v3_bindings(manifest, context),
                "cancelled": 0,
            }
        if validated_context is None:
            self._verify_approval(
                approval_path,
                operation="cancel",
                release=release,
                manifest=manifest,
            )
        store = StateStore(self.state_root)
        with store.locked():
            states = self._paired_states(store, manifest, context)
            command_ids = {state.get("command_id") for state in states}
            if (
                validated_context is not None
                and {state.get("instance_id") for state in states}
                != {validated_context.instance_id}
            ):
                raise MsctlError(
                    "FLEET_PLAN_INVALID",
                    "cancel state is bound to a different fleet instance",
                )
            if validated_context is not None:
                deadline = states[0].get("terminate_at")
                selected_instance = states[0].get("instance_id")
                if not isinstance(deadline, str) or not isinstance(
                    selected_instance,
                    str,
                ):
                    raise MsctlError(
                        "STATE_CORRUPT",
                        "v3 cancel state lacks its execution binding",
                    )
                self._verify_approval(
                    approval_path,
                    operation="cancel",
                    release=release,
                    manifest=manifest,
                    resources=self._cancel_resources(
                        release=release,
                        manifest=manifest,
                        instance_id=selected_instance,
                        terminate_at=deadline,
                        context=context,
                    ),
                )
            if {state.get("status") for state in states} == {"Cancelling"}:
                return {
                    "provider": self.profile.provider,
                    "seed": manifest.seed,
                    "command_id": states[0]["command_id"],
                    "status": "Cancelling",
                    "cancelled": 0,
                    "idempotent": True,
                }
            if None in command_ids or len(command_ids) != 1:
                raise MsctlError(
                    "SUBMISSION_UNCERTAIN",
                    "AWS cancel requires one paired SSM command",
                )
            command_id = str(next(iter(command_ids)))
            status = self._command_status(
                str(states[0].get("instance_id")),
                command_id,
            )
            if status not in _ACTIVE_COMMAND_STATES:
                raise MsctlError(
                    "RUN_NOT_ACTIVE",
                    "refusing to cancel a terminal AWS command",
                    details={"status": status},
                )
            argv = self._aws_argv(
                "ssm",
                "cancel-command",
                "--command-id",
                command_id,
                query="{}",
            )
            _aws_output_object(
                self._run(argv, operation="cancel"),
                set(),
                label="SSM cancel-command output",
            )
            now = _timestamp()
            for run, state in zip(manifest.runs, states):
                state["status"] = "Cancelling"
                state["updated_at"] = now
                store.write_run(run.run_id, state)
            return {
                "provider": self.profile.provider,
                "seed": manifest.seed,
                "command_id": command_id,
                "status": "Cancelling",
                "cancelled": 1,
                "idempotent": False,
            }

    def _evaluation_argv(
        self,
        release: object,
        manifest: object,
        context: V3LifecycleContext | None = None,
    ) -> list[list[str]]:
        release_root = self._release_root(release)
        scratch_root = getattr(self.profile, "scratch_root", "/mnt/memorysplit")
        run_root = f"{scratch_root}/runs/seed-{manifest.seed}"
        evaluation_root = f"{scratch_root}/evaluations"
        final_bindings = self._v3_evaluation_bindings(manifest, context)
        sealed_root = (
            f"{scratch_root}/sealed-evaluation/"
            f"{final_bindings['sealed_evaluation_sha256']}"
            if _is_v3_manifest(manifest)
            else release_root
        )
        return [
            [
                "/usr/bin/docker",
                "run",
                "--rm",
                "--read-only",
                "--network",
                "none",
                "--gpus",
                "all",
                "--user",
                (
                    f"{getattr(self.runtime, 'uid', 1000)}:"
                    f"{getattr(self.runtime, 'gid', 1000)}"
                ),
                "--security-opt",
                "no-new-privileges",
                "--cap-drop",
                "ALL",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,noexec,size=1073741824",
                "--env",
                "HOME=/tmp",
                "--env",
                "PYTHONNOUSERSITE=1",
                "--mount",
                f"type=bind,src={release_root},dst=/workspace,readonly",
                "--mount",
                f"type=bind,src={sealed_root},dst=/sealed,readonly",
                "--mount",
                f"type=bind,src={run_root},dst={run_root},readonly",
                "--mount",
                (
                    f"type=bind,src={evaluation_root},"
                    f"dst={evaluation_root}"
                ),
                "--workdir",
                "/workspace",
                self.runtime.container_image,
                "/opt/venv/bin/python",
                "-m",
                "evals.confirmatory",
                "evaluate",
                "--run",
                f"{run_root}/{run.arm}/run",
                "--sealed-release",
                "/sealed",
                "--expected-study-lock-sha256",
                (
                    final_bindings["study_lock_sha256"]
                    if _is_v3_manifest(manifest)
                    else manifest.study_lock_sha256
                ),
                "--device",
                "cuda",
                "--output-dir",
                f"{evaluation_root}/{run.run_id}",
            ]
            for run in sorted(manifest.runs, key=lambda row: row.arm)
        ]

    def _evaluation_operation_intent(
        self,
        release: object,
        manifest: object,
        evidence: Mapping[str, str] | None = None,
        context: V3LifecycleContext | None = None,
        checkpoint_receipt: object | None = None,
    ) -> dict[str, object]:
        v3_bindings = self._v3_bindings(manifest, context)
        final_bindings = self._v3_evaluation_bindings(manifest, context)
        checkpoint_binding = self._evaluation_checkpoint_binding(
            manifest,
            checkpoint_receipt,
        )
        remote_argv = self._evaluation_argv(release, manifest, context)
        sealed_materialization: list[dict[str, object]] = []
        terminal_materialization: list[dict[str, object]] = []
        receipt_root = ""
        record_root = ""
        run_root = ""
        if _is_v3_manifest(manifest):
            scratch_root = getattr(
                self.profile,
                "scratch_root",
                "/mnt/memorysplit",
            )
            sealed_root = (
                f"{scratch_root}/sealed-evaluation/"
                f"{final_bindings['sealed_evaluation_sha256']}"
            )
            for member in sorted(REQUIRED_SEALED_MEMBERS):
                bucket, key = self._s3_location(
                    "sealed-evaluation/"
                    f"{final_bindings['sealed_evaluation_sha256']}/{member}"
                )
                sealed_materialization.append(
                    {
                        "name": (
                            "materialize-sealed-evaluation-"
                            f"{member.replace('.', '-')}"
                        ),
                        "argv": [
                            "/usr/bin/env",
                            "aws",
                            "--no-cli-pager",
                            "--region",
                            self.runtime.region,
                            "s3api",
                            "get-object",
                            "--bucket",
                            bucket,
                            "--key",
                            key,
                            "--checksum-mode",
                            "ENABLED",
                            f"{sealed_root}/{member}",
                        ],
                    }
                )
            assert checkpoint_binding is not None
            receipt_root = (
                f"{scratch_root}/receipts/checkpoints/seed-{manifest.seed}"
            )
            record_root = f"{receipt_root}/records"
            run_root = f"{scratch_root}/runs/seed-{manifest.seed}"
            terminal_sources: list[tuple[str, object, str]] = [
                (
                    "receipt",
                    checkpoint_binding["uri"],
                    f"{receipt_root}/receipt.json",
                )
            ]
            raw_checkpoints = checkpoint_binding["checkpoints"]
            assert isinstance(raw_checkpoints, list)
            for raw in raw_checkpoints:
                assert isinstance(raw, dict)
                arm = str(raw["arm"])
                arm_root = f"{run_root}/{arm}/run"
                terminal_sources.extend(
                    [
                        (
                            f"{arm}-checkpoint",
                            raw["checkpoint_uri"],
                            f"{arm_root}/ckpt.pt",
                        ),
                        (
                            f"{arm}-configuration",
                            raw["configuration_uri"],
                            f"{arm_root}/configuration.yaml",
                        ),
                        (
                            f"{arm}-run-binding",
                            raw["run_binding_uri"],
                            f"{arm_root}/run.json",
                        ),
                        (
                            f"{arm}-checkpoint-record",
                            raw["checkpoint_record_uri"],
                            (
                                f"{record_root}/"
                                f"{raw['checkpoint_record_sha256']}.json"
                            ),
                        ),
                    ]
                )
            for name, uri, destination in terminal_sources:
                relative = self._runtime_relative_uri(
                    uri,
                    label=f"evaluation terminal artifact {name}",
                )
                bucket, key = self._s3_location(relative)
                terminal_materialization.append(
                    {
                        "name": f"materialize-terminal-{name}",
                        "argv": [
                            "/usr/bin/env",
                            "aws",
                            "--no-cli-pager",
                            "--region",
                            self.runtime.region,
                            "s3api",
                            "get-object",
                            "--bucket",
                            bucket,
                            "--key",
                            key,
                            "--checksum-mode",
                            "ENABLED",
                            destination,
                        ],
                    }
                )
        lifecycle_evidence = self._validated_lifecycle_evidence(
            operation="evaluate",
            manifest=manifest,
            evidence=evidence,
            context=context,
        )
        return {
            "schema_version": 3 if v3_bindings else 1,
            "operation": "evaluate",
            "provider": self.profile.provider,
            "instance_type": self.profile.instance_type,
            "profile_sha256": self.profile.sha256,
            "gres": _profile_gres(self.profile),
            "seed": manifest.seed,
            "release_sha256": release.archive_sha256,
            "run_manifest_sha256": manifest.sha256,
            "dataset_sha256": manifest.dataset_sha256,
            **v3_bindings,
            **final_bindings,
            **lifecycle_evidence,
            "runtime_sha256": self._runtime_sha256(),
            "environment": {
                "AWS_REGION": self.runtime.region,
                "MS_AWS_AMI_ID": self.runtime.ami_id,
                "MS_CONTAINER_DIGEST": self.runtime.container_digest,
                "MS_RUNTIME_GID": str(getattr(self.runtime, "gid", 1000)),
                "MS_RUNTIME_UID": str(getattr(self.runtime, "uid", 1000)),
                "MS_S3_ROOT": self.runtime.s3_root,
                **(
                    {
                        "MS_S3_KMS_KEY_ID": str(
                            getattr(self.runtime, "kms_key_id", "")
                        )
                    }
                    if v3_bindings
                    else {}
                ),
            },
            "checkpoint_receipt": checkpoint_binding,
            "steps": [
                *(
                    [
                        {
                            "name": "prepare-sealed-evaluation",
                            "argv": [
                                "/usr/bin/install",
                                "-d",
                                "-m",
                                "0700",
                                "-o",
                                str(getattr(self.runtime, "uid", 1000)),
                                "-g",
                                str(getattr(self.runtime, "gid", 1000)),
                                (
                                    f"{getattr(self.profile, 'scratch_root', '/mnt/memorysplit')}"
                                    "/sealed-evaluation/"
                                    f"{final_bindings['sealed_evaluation_sha256']}"
                                ),
                            ],
                        },
                        {
                            "name": "prepare-evaluation-output",
                            "argv": [
                                "/usr/bin/install",
                                "-d",
                                "-m",
                                "0700",
                                "-o",
                                str(getattr(self.runtime, "uid", 1000)),
                                "-g",
                                str(getattr(self.runtime, "gid", 1000)),
                                (
                                    f"{getattr(self.profile, 'scratch_root', '/mnt/memorysplit')}"
                                    "/evaluations"
                                ),
                            ],
                        },
                        {
                            "name": "prepare-terminal-checkpoint-bundle",
                            "argv": [
                                "/usr/bin/install",
                                "-d",
                                "-m",
                                "0700",
                                "-o",
                                str(getattr(self.runtime, "uid", 1000)),
                                "-g",
                                str(getattr(self.runtime, "gid", 1000)),
                                receipt_root,
                                record_root,
                                *[
                                    f"{run_root}/{arm}/run"
                                    for arm in ("dense", "split90")
                                ],
                            ],
                        },
                    ]
                    + [
                        {
                            **step,
                        }
                        for step in sealed_materialization
                    ]
                    + [
                        {
                            **step,
                        }
                        for step in terminal_materialization
                    ]
                    + [
                        {
                            "name": "verify-sealed-evaluation",
                            "argv": [
                                "/usr/bin/docker",
                                "run",
                                "--rm",
                                "--network",
                                "none",
                                "--read-only",
                                "--user",
                                (
                                    f"{getattr(self.runtime, 'uid', 1000)}:"
                                    f"{getattr(self.runtime, 'gid', 1000)}"
                                ),
                                "--mount",
                                (
                                    f"type=bind,src={self._release_root(release)},"
                                    "dst=/workspace,readonly"
                                ),
                                "--mount",
                                (
                                    "type=bind,src="
                                    f"{getattr(self.profile, 'scratch_root', '/mnt/memorysplit')}"
                                    "/sealed-evaluation/"
                                    f"{final_bindings['sealed_evaluation_sha256']},"
                                    "dst=/sealed,readonly"
                                ),
                                "--workdir",
                                "/workspace",
                                self.runtime.container_image,
                                "/opt/venv/bin/python",
                                "-m",
                                "msctl.aws_sealed_evaluation",
                                "--root",
                                "/sealed",
                                "--expected-release-sha256",
                                final_bindings["sealed_evaluation_sha256"],
                                "--expected-study-lock-sha256",
                                final_bindings["study_lock_sha256"],
                            ],
                        },
                        {
                            "name": "verify-terminal-checkpoints",
                            "argv": [
                                "/usr/bin/python3",
                                (
                                    f"{self._release_root(release)}/"
                                    "cluster/aws/p5/terminal_artifacts.py"
                                ),
                                "verify-materialized",
                                "--receipt",
                                f"{receipt_root}/receipt.json",
                                "--run-root",
                                run_root,
                                "--record-root",
                                record_root,
                                "--expected-receipt-sha256",
                                checkpoint_binding["sha256"],
                                "--provider",
                                self.profile.provider,
                                "--run-manifest-sha256",
                                manifest.sha256,
                                "--sealed-fixture-sha256",
                                manifest.sealed_fixture_sha256,
                            ],
                        },
                    ]
                    if _is_v3_manifest(manifest)
                    else []
                ),
                *[
                    {"name": f"evaluate-{run.arm}", "argv": argv}
                for run, argv in zip(
                    sorted(manifest.runs, key=lambda row: row.arm),
                    remote_argv,
                    strict=True,
                )
                ],
                *(
                    [
                        {
                            "name": "publish-paired-evaluation",
                            "argv": [
                                "/usr/bin/python3",
                                (
                                    f"{self._release_root(release)}/"
                                    "cluster/aws/p5/terminal_artifacts.py"
                                ),
                                "publish-evaluation",
                                "--evaluation-root",
                                (
                                    f"{getattr(self.profile, 'scratch_root', '/mnt/memorysplit')}"
                                    "/evaluations"
                                ),
                                "--checkpoint-receipt",
                                (
                                    f"{getattr(self.profile, 'scratch_root', '/mnt/memorysplit')}"
                                    f"/receipts/checkpoints/seed-{manifest.seed}/"
                                    "receipt.json"
                                ),
                                "--s3-root",
                                self.runtime.s3_root,
                                "--provider",
                                self.profile.provider,
                                "--run-manifest-sha256",
                                manifest.sha256,
                                "--sealed-fixture-sha256",
                                manifest.sealed_fixture_sha256,
                                "--sealed-evaluation-sha256",
                                final_bindings["sealed_evaluation_sha256"],
                                "--study-lock-sha256",
                                final_bindings["study_lock_sha256"],
                                "--fleet-plan-sha256",
                                v3_bindings["fleet_plan_sha256"],
                                "--fleet-wave",
                                str(v3_bindings["fleet_wave"]),
                                "--launch-readiness-sha256",
                                v3_bindings["launch_readiness_sha256"],
                            ],
                        }
                    ]
                    if _is_v3_manifest(manifest)
                    else []
                ),
            ],
        }

    def evaluate(
        self,
        *,
        release: object,
        manifest: object,
        approval_path: Path | str | None,
        apply: bool,
        evidence: Mapping[str, str] | None = None,
        context: V3LifecycleContext | None = None,
        checkpoint_receipt: object | None = None,
    ) -> dict[str, object]:
        self._validate_manifest(manifest)
        self._validate_release(release, manifest)
        validated_context = self._require_protected_launch_context(
            manifest,
            context,
        )
        if validated_context is not None:
            self._require_fleet_advance(manifest, validated_context)
        operation_intent = self._evaluation_operation_intent(
            release,
            manifest,
            evidence,
            context,
            checkpoint_receipt,
        )
        if not apply:
            return {
                "provider": self.profile.provider,
                "seed": manifest.seed,
                "operation_intent": operation_intent,
                "submitted": 0,
                "idempotent": False,
            }
        if approval_path is None:
            raise MsctlError(
                "APPROVAL_REQUIRED",
                "evaluate apply requires an explicit signed approval receipt",
            )
        store = StateStore(self.state_root)
        with store.locked():
            states = self._paired_states(store, manifest, context)
            if {
                state.get("environment_receipt_sha256") for state in states
            } != {operation_intent["environment_receipt_sha256"]}:
                raise MsctlError(
                    "ENVIRONMENT_RECEIPT_MISMATCH",
                    "AWS evaluation must reuse the authenticated launch receipt",
                )
            instance_ids = {state.get("instance_id") for state in states}
            if None in instance_ids or len(instance_ids) != 1:
                raise MsctlError(
                    "SUBMISSION_UNCERTAIN",
                    "AWS evaluation requires one paired instance",
                )
            instance_id = str(next(iter(instance_ids)))
            if (
                validated_context is not None
                and instance_id != validated_context.instance_id
            ):
                raise MsctlError(
                    "FLEET_PLAN_INVALID",
                    "evaluation state is bound to a different fleet instance",
                )
            deadlines = {state.get("terminate_at") for state in states}
            if len(deadlines) != 1 or not isinstance(
                next(iter(deadlines)),
                str,
            ):
                raise MsctlError(
                    "SUBMISSION_UNCERTAIN",
                    "AWS evaluation requires one bound termination deadline",
                )
            terminate_at = str(next(iter(deadlines)))
            self._validate_submit_selection(instance_id, terminate_at)
            evaluation_resources = self._evaluation_resources(
                release=release,
                manifest=manifest,
                instance_id=instance_id,
                terminate_at=terminate_at,
                context=context,
                checkpoint_receipt=checkpoint_receipt,
            )
            evaluation_resources.update(
                {
                    field: operation_intent[field]
                    for field in (
                        "dataset_pointer_sha256",
                        "dataset_verification_sha256",
                        "environment_receipt_sha256",
                    )
                }
            )
            self._verify_approval(
                approval_path,
                operation="evaluate",
                release=release,
                manifest=manifest,
                resources=evaluation_resources,
            )
            operation_intent = self._operation_envelope(
                operation_intent,
                instance_id=instance_id,
                terminate_at=terminate_at,
            )
            if _is_v3_manifest(manifest):
                self._validate_evaluation_instance_binding(
                    manifest,
                    instance_id=instance_id,
                    terminate_at=terminate_at,
                    checkpoint_receipt_sha256=require_sha256(
                        getattr(checkpoint_receipt, "sha256", None),
                        label="evaluation checkpoint receipt",
                    ),
                    context=context,
                )
            else:
                self._validate_selected_instance_binding(
                    manifest,
                    instance_id=instance_id,
                    terminate_at=terminate_at,
                    operation="evaluation",
                    context=context,
                )
            training_commands = {
                state.get("command_id") for state in states
            }
            if None in training_commands or len(training_commands) != 1:
                raise MsctlError(
                    "SUBMISSION_UNCERTAIN",
                    "AWS evaluation requires one paired training command",
                )
            training_status = self._command_status(
                instance_id,
                str(next(iter(training_commands))),
            )
            if training_status in _ACTIVE_COMMAND_STATES:
                raise MsctlError(
                    "RUN_ALREADY_ACTIVE",
                    "refusing to evaluate an active AWS training pair",
                )
            if training_status != "Success":
                raise MsctlError(
                    "RUN_NOT_COMPLETE",
                    "AWS evaluation requires successful paired training",
                    details={"status": training_status},
                )
            existing = store.read_evaluation(manifest.sha256)
            if existing is not None:
                if (
                    existing.get("provider") != self.profile.provider
                    or existing.get("instance_type")
                    != self.profile.instance_type
                    or existing.get("gres") != _profile_gres(self.profile)
                    or existing.get("release_sha256")
                    != manifest.release_sha256
                    or existing.get("dataset_sha256")
                    != manifest.dataset_sha256
                    or existing.get("run_manifest_sha256") != manifest.sha256
                    or existing.get("profile_sha256") != self.profile.sha256
                    or existing.get("runtime_sha256")
                    != self._runtime_sha256()
                    or existing.get("ami_id") != self.runtime.ami_id
                    or existing.get("container_digest")
                    != self.runtime.container_digest
                    or existing.get("instance_id") != instance_id
                    or existing.get("terminate_at") != terminate_at
                    or existing.get("operation") != "evaluate"
                    or (
                        _is_v3_manifest(manifest)
                        and existing.get("checkpoint_receipt_sha256")
                        != getattr(checkpoint_receipt, "sha256", None)
                    )
                    or (
                        _is_v3_manifest(manifest)
                        and any(
                            existing.get(field) != expected
                            for field, expected in (
                                self._v3_bindings(manifest, context)
                                | self._v3_evaluation_bindings(
                                    manifest,
                                    context,
                                )
                            ).items()
                        )
                    )
                ):
                    raise MsctlError(
                        "RUN_ID_CONFLICT",
                        "AWS evaluation state has different provenance",
                    )
                self._require_state_intent_binding(
                    [existing],
                    operation_intent,
                )
                command_value = existing.get("command_id")
                if command_value is None:
                    if existing.get("send_attempted") is not True:
                        raise MsctlError(
                            "SUBMISSION_UNCERTAIN",
                            "evaluation intent was not durably marked as sent",
                        )
                    started = self._operation_receipt_exists(
                        operation_intent,
                        kind="started",
                    )
                    terminal = self._operation_receipt_exists(
                        operation_intent,
                        kind="terminal",
                    )
                    recovered = self._find_operation_command(
                        operation_id=str(operation_intent["operation_id"]),
                        instance_id=instance_id,
                    )
                    if recovered is None:
                        if terminal and not started:
                            raise MsctlError(
                                "REMOTE_RECEIPT_INVALID",
                                "remote terminal receipt exists without its "
                                "started acquisition receipt",
                            )
                        if terminal:
                            return {
                                "provider": self.profile.provider,
                                "seed": manifest.seed,
                                "instance_id": instance_id,
                                "operation_id": operation_intent[
                                    "operation_id"
                                ],
                                "status": "REMOTE_TERMINAL",
                                "submitted": 0,
                                "idempotent": True,
                            }
                        if started:
                            existing["status"] = "RECOVERY_REQUIRED"
                            existing["updated_at"] = _timestamp()
                            store.write_evaluation(manifest.sha256, existing)
                            raise MsctlError(
                                "REMOTE_RECOVERY_REQUIRED",
                                "remote evaluation was acquired but has no "
                                "terminal receipt; automatic success or resend "
                                "is forbidden",
                                details={
                                    "operation_id": operation_intent[
                                        "operation_id"
                                    ],
                                },
                            )
                        raise MsctlError(
                            "SUBMISSION_UNCERTAIN",
                            "evaluation send has no safe resend proof",
                        )
                    command_id = str(recovered["command_id"])
                    status = str(recovered["status"])
                    existing["command_id"] = command_id
                    existing["status"] = status
                    existing["updated_at"] = _timestamp()
                    store.write_evaluation(manifest.sha256, existing)
                    return {
                        "provider": self.profile.provider,
                        "seed": manifest.seed,
                        "instance_id": instance_id,
                        "command_id": command_id,
                        "operation_id": operation_intent["operation_id"],
                        "status": status,
                        "submitted": 0,
                        "idempotent": True,
                    }
                command_id = str(command_value)
                status = self._command_status(instance_id, command_id)
                existing["status"] = status
                existing["updated_at"] = _timestamp()
                store.write_evaluation(manifest.sha256, existing)
                return {
                    "provider": self.profile.provider,
                    "seed": manifest.seed,
                    "instance_id": instance_id,
                    "command_id": command_id,
                    "status": status,
                    "submitted": 0,
                    "idempotent": True,
                }
            self._require_ssm_online(instance_id)
            self._ensure_argv_document()
            if _is_v3_manifest(manifest):
                self._publish_control_bundle()
            published = self._publish_operation_intent(operation_intent)
            now = _timestamp()
            evaluation_state = {
                    "schema_version": (
                        3 if _is_v3_manifest(manifest) else 1
                    ),
                    "provider": self.profile.provider,
                    "instance_type": self.profile.instance_type,
                    "gres": _profile_gres(self.profile),
                    "seed": manifest.seed,
                    "release_sha256": manifest.release_sha256,
                    "dataset_sha256": manifest.dataset_sha256,
                    "dataset_pointer_sha256": operation_intent[
                        "dataset_pointer_sha256"
                    ],
                    "dataset_verification_sha256": operation_intent[
                        "dataset_verification_sha256"
                    ],
                    "environment_receipt_sha256": operation_intent[
                        "environment_receipt_sha256"
                    ],
                    "run_manifest_sha256": manifest.sha256,
                    "profile_sha256": self.profile.sha256,
                    "runtime_sha256": self._runtime_sha256(),
                    "ami_id": self.runtime.ami_id,
                    "container_digest": self.runtime.container_digest,
                    "instance_id": instance_id,
                    "terminate_at": terminate_at,
                    "operation_id": operation_intent["operation_id"],
                    "intent_sha256": published["intent_sha256"],
                    "intent_uri": published["intent_uri"],
                    "command_id": None,
                    "operation": "evaluate",
                    "status": "INTENT_PUBLISHED",
                    "send_attempted": False,
                    "created_at": now,
                    "updated_at": now,
                }
            if _is_v3_manifest(manifest):
                evaluation_state.update(self._v3_bindings(manifest, context))
                evaluation_state.update(
                    self._v3_evaluation_bindings(manifest, context)
                )
                evaluation_state["checkpoint_receipt_sha256"] = (
                    checkpoint_receipt.sha256
                )
            store.write_evaluation(manifest.sha256, evaluation_state)
            state = store.read_evaluation(manifest.sha256)
            assert state is not None
            state["status"] = "SENDING"
            state["send_attempted"] = True
            state["updated_at"] = _timestamp()
            store.write_evaluation(manifest.sha256, state)
            command_id = self._send_operation_intent(
                instance_id=instance_id,
                intent=operation_intent,
                published=published,
                operation="evaluate",
            )
            state = store.read_evaluation(manifest.sha256)
            assert state is not None
            state["command_id"] = command_id
            state["status"] = "Pending"
            state["updated_at"] = _timestamp()
            store.write_evaluation(manifest.sha256, state)
            return {
                "provider": self.profile.provider,
                "seed": manifest.seed,
                "instance_id": instance_id,
                "command_id": command_id,
                "status": "Pending",
                "submitted": 1,
                "idempotent": False,
            }

    def cleanup(
        self,
        *,
        release: object,
        manifest: object,
        approval_path: Path | str | None,
        apply: bool,
        context: V3LifecycleContext | None = None,
    ) -> dict[str, object]:
        self._validate_manifest(manifest)
        self._validate_release(release, manifest)
        validated_context = self._validate_v3_context(manifest, context)
        if not apply:
            return {
                "provider": self.profile.provider,
                "seed": manifest.seed,
                "release_sha256": release.archive_sha256,
                "run_manifest_sha256": manifest.sha256,
                "operation": "cleanup",
                **self._v3_bindings(manifest, context),
                "terminated": 0,
            }
        if approval_path is None:
            raise MsctlError(
                "APPROVAL_REQUIRED",
                "cleanup apply requires an explicit signed approval receipt",
            )
        store = StateStore(self.state_root)
        with store.locked():
            states = self._paired_states(store, manifest, context)
            instance_ids = {state.get("instance_id") for state in states}
            if None in instance_ids or len(instance_ids) != 1:
                raise MsctlError(
                    "SUBMISSION_UNCERTAIN",
                    "AWS cleanup requires one paired instance",
                )
            instance_id = str(next(iter(instance_ids)))
            if (
                validated_context is not None
                and instance_id != validated_context.instance_id
            ):
                raise MsctlError(
                    "FLEET_PLAN_INVALID",
                    "cleanup state is bound to a different fleet instance",
                )
            deadlines = {state.get("terminate_at") for state in states}
            if len(deadlines) != 1 or not isinstance(
                next(iter(deadlines)),
                str,
            ):
                raise MsctlError(
                    "SUBMISSION_UNCERTAIN",
                    "AWS cleanup requires one bound termination deadline",
                )
            terminate_at = str(next(iter(deadlines)))
            self._verify_approval(
                approval_path,
                operation="cleanup",
                release=release,
                manifest=manifest,
                resources=self._cleanup_resources(
                    release=release,
                    manifest=manifest,
                    instance_id=instance_id,
                    terminate_at=terminate_at,
                    context=context,
                ),
            )
            if {state.get("status") for state in states} == {"Terminating"}:
                return {
                    "provider": self.profile.provider,
                    "seed": manifest.seed,
                    "instance_id": instance_id,
                    "status": "Terminating",
                    "terminated": 0,
                    "idempotent": True,
                }
            self._validate_selected_instance_binding(
                manifest,
                instance_id=instance_id,
                terminate_at=terminate_at,
                operation="cleanup",
                context=context,
            )
            command_ids = {state.get("command_id") for state in states}
            if None in command_ids or len(command_ids) != 1:
                raise MsctlError(
                    "SUBMISSION_UNCERTAIN",
                    "AWS cleanup requires one paired command",
                )
            status = self._command_status(
                instance_id,
                str(next(iter(command_ids))),
            )
            if status in _ACTIVE_COMMAND_STATES:
                raise MsctlError(
                    "RUN_ALREADY_ACTIVE",
                    "refusing to clean up an active AWS training pair",
                )
            evaluation = store.read_evaluation(manifest.sha256)
            if evaluation is not None:
                evaluation_command = evaluation.get("command_id")
                if not isinstance(evaluation_command, str):
                    raise MsctlError(
                        "SUBMISSION_UNCERTAIN",
                        "AWS evaluation state lacks one command ID",
                    )
                evaluation_status = self._command_status(
                    instance_id,
                    evaluation_command,
                )
                if evaluation_status in _ACTIVE_COMMAND_STATES:
                    raise MsctlError(
                        "RUN_ALREADY_ACTIVE",
                        "refusing to clean up an active AWS evaluation",
                    )
            argv = self._aws_argv(
                "ec2",
                "terminate-instances",
                "--instance-ids",
                instance_id,
                query=(
                    "{terminating_instances:TerminatingInstances[]."
                    "{instance_id:InstanceId,current_state:CurrentState.Name,"
                    "previous_state:PreviousState.Name}}"
                ),
            )
            output = _aws_output_object(
                self._run(argv, operation="cleanup"),
                {"terminating_instances"},
                label="EC2 terminate-instances output",
            )
            rows = _aws_output_list(
                output["terminating_instances"],
                label="EC2 terminating instances",
            )
            if len(rows) != 1:
                raise MsctlError(
                    "AWS_OUTPUT_INVALID",
                    "EC2 must confirm one terminating instance",
                )
            row = _aws_output_object(
                rows[0],
                {"instance_id", "current_state", "previous_state"},
                label="EC2 terminating instance",
            )
            if row["instance_id"] != instance_id:
                raise MsctlError(
                    "INSTANCE_BINDING_MISMATCH",
                    "termination response has the wrong instance ID",
                )
            now = _timestamp()
            for run, state in zip(manifest.runs, states):
                state["status"] = "Terminating"
                state["updated_at"] = now
                store.write_run(run.run_id, state)
            return {
                "provider": self.profile.provider,
                "seed": manifest.seed,
                "instance_id": instance_id,
                "status": row["current_state"],
                "terminated": 1,
                "idempotent": False,
            }

    def dispatch(
        self,
        command: str,
        args: object,
    ) -> tuple[bool, dict[str, object]]:
        if command == "auth check":
            return False, self.auth_check()
        if command == "capacity check":
            return False, self.capacity_check()
        if command == "control install":
            apply = bool(getattr(args, "apply", False))
            return not apply, self.control_install(
                instance_id=str(args.instance_id),
                bundle_path=args.bundle,
                bundle_sha256=args.bundle_sha256,
                apply=apply,
            )
        if command == "canary plan":
            apply = bool(getattr(args, "apply", False))
            return not apply, self.canary_plan(
                release_path=args.release,
                amendment_path=args.amendment,
                provider_selection_path=args.provider_selection,
                environment_receipt=args.environment_receipt,
                instance_id=args.instance_id,
                out=args.out,
                apply=apply,
            )
        if command == "canary run":
            apply = bool(getattr(args, "apply", False))
            return not apply, self.canary_run(
                plan_path=args.canary_plan,
                instance_id=args.instance_id,
                approval_path=args.approval,
                apply=apply,
            )
        if command == "env ensure" and self.profile.provider in _V3_PROFILES:
            raise MsctlError(
                "EXTERNAL_OPERATION_UNSUPPORTED",
                "v3 AWS runtime environments are externally selected and attested",
                details={"operation": command},
            )
        if command == "env ensure":
            apply = bool(getattr(args, "apply", False))
            lock_value = getattr(args, "lock", None)
            root_value = getattr(args, "root", None)
            if lock_value is None or root_value is None:
                raise MsctlError(
                    "CLI_USAGE",
                    "AWS env ensure requires --lock and --root",
                )
            return not apply, self.env_ensure(
                root=root_value,
                lock=lock_value,
                apply=apply,
            )
        if command in {"dataset ensure", "dataset verify"}:
            apply = bool(getattr(args, "apply", False))
            receipt_path = getattr(args, "receipt", None)
            if receipt_path is None:
                raise MsctlError(
                    "CLI_USAGE",
                    f"AWS {command} requires --receipt",
                )
            operation = (
                self.dataset_ensure
                if command == "dataset ensure"
                else self.dataset_verify
            )
            return not apply, operation(
                receipt_path=receipt_path,
                apply=apply,
            )
        if command == "fleet advance":
            if self.profile.provider not in _V3_PROFILES:
                raise MsctlError(
                    "EXTERNAL_OPERATION_UNSUPPORTED",
                    "fleet advance is available only to closed AWS v3 profiles",
                )
            apply = bool(getattr(args, "apply", False))
            return not apply, self.fleet_advance(
                amendment_path=args.amendment,
                provider_selection_path=args.provider_selection,
                fleet_plan_path=args.fleet_plan,
                target_manifest_path=args.to_manifest,
                repo_root=args.repo_root,
                instance_id=args.instance_id,
                checkpoint_receipt_path=args.checkpoint_receipt,
                approval_path=args.approval,
                apply=apply,
            )
        if command == "collect":
            apply = bool(getattr(args, "apply", False))
            return not apply, self.collect(
                source=str(args.source),
                out=args.out,
                apply=apply,
            )

        if command in {
            "runs render",
            "submit",
            "status",
            "resume",
            "cancel",
            "evaluate",
            "cleanup plan",
            "cleanup apply",
        }:
            release_path = getattr(args, "release", None)
            manifest_path = getattr(args, "manifest", None)
            if release_path is None or manifest_path is None:
                raise MsctlError(
                    "CLI_USAGE",
                    f"AWS {command} requires --release and --manifest",
                )
            release, manifest = self._load_bound_inputs(
                release_path=release_path,
                manifest_path=manifest_path,
                repo_root=args.repo_root,
            )
            requested_instance_id = (
                getattr(args, "instance_id", None)
                if command == "submit"
                else None
            )
            context = self._load_v3_context(
                manifest,
                release=release,
                amendment_path=getattr(args, "hardware_amendment", None),
                provider_selection_path=getattr(
                    args,
                    "provider_selection",
                    None,
                ),
                fleet_plan_path=getattr(args, "fleet_plan", None),
                readiness_path=getattr(args, "launch_readiness", None),
                environment_receipt_path=getattr(
                    args,
                    "environment_receipt",
                    None,
                ),
                qualification_receipt_path=getattr(
                    args,
                    "qualification_receipt",
                    None,
                ),
                diagnostic_receipts=_diagnostic_receipt_map(
                    getattr(args, "diagnostic_receipt", None)
                ),
                sealed_evaluation_fixture_path=getattr(
                    args,
                    "sealed_evaluation_fixture",
                    None,
                ),
                sealed_evaluation_release_path=getattr(
                    args,
                    "sealed_evaluation_release",
                    None,
                ),
                expected_sealed_evaluation_sha256=getattr(
                    args,
                    "expected_sealed_evaluation_sha256",
                    None,
                ),
                expected_study_lock_sha256=getattr(
                    args,
                    "expected_study_lock_sha256",
                    None,
                ),
                require_readiness=command in {"submit", "resume", "evaluate"},
                require_finalized_evaluation=command == "evaluate",
                instance_id=requested_instance_id,
            )
            if command == "submit" and context is not None:
                requested_instance_id = context.instance_id
            evidence = None
            if command in {"runs render", "submit", "resume", "evaluate"}:
                dataset_pointer = getattr(args, "dataset_pointer", None)
                dataset_root = getattr(args, "dataset_root", None)
                dataset_verification = getattr(
                    args,
                    "dataset_verification",
                    None,
                )
                environment_receipt = getattr(
                    args,
                    "environment_receipt",
                    None,
                )
                if (
                    dataset_pointer is None
                    or environment_receipt is None
                    or (dataset_root is None) == (dataset_verification is None)
                ):
                    raise MsctlError(
                        "CLI_USAGE",
                        "AWS lifecycle requires dataset pointer/source and "
                        "environment receipt",
                    )
                evidence = self._load_lifecycle_evidence(
                    manifest=manifest,
                    dataset_pointer=dataset_pointer,
                    dataset_root=dataset_root,
                    dataset_verification=dataset_verification,
                    environment_receipt=environment_receipt,
                    selection=(
                        context.selection if context is not None else None
                    ),
                    expected_instance_id=(
                        context.instance_id
                        if context is not None
                        else (
                            str(requested_instance_id)
                            if command == "submit"
                            else None
                        )
                    ),
                )
            if command == "runs render":
                return True, self.render(
                    release=release,
                    manifest=manifest,
                    evidence=evidence,
                    context=context,
                )
            if command == "submit":
                return not args.apply, self.submit(
                    release=release,
                    manifest=manifest,
                    instance_id=requested_instance_id,
                    terminate_at=args.terminate_at,
                    approval_path=args.approval,
                    apply=args.apply,
                    evidence=evidence,
                    context=context,
                )
            if command == "status":
                return False, self.status(
                    release=release,
                    manifest=manifest,
                    cached=args.cached,
                    context=context,
                )
            if command == "resume":
                receipt = verify_checkpoint_receipt(
                    args.checkpoint_receipt,
                    release=release,
                    manifest=manifest,
                )
                return not args.apply, self.resume(
                    release=release,
                    manifest=manifest,
                    checkpoint_receipt=receipt,
                    approval_path=args.approval,
                    apply=args.apply,
                    evidence=evidence,
                    context=context,
                )
            if command == "cancel":
                return not args.apply, self.cancel(
                    release=release,
                    manifest=manifest,
                    approval_path=args.approval,
                    apply=args.apply,
                    context=context,
                )
            if command == "evaluate":
                checkpoint_receipt = None
                if _is_v3_manifest(manifest):
                    if args.checkpoint_receipt is None:
                        raise MsctlError(
                            "CLI_USAGE",
                            "v3 evaluation requires --checkpoint-receipt",
                        )
                    checkpoint_receipt = verify_checkpoint_receipt(
                        args.checkpoint_receipt,
                        release=release,
                        manifest=manifest,
                        require_checkpoint_files=False,
                        require_durable_terminal=True,
                    )
                elif args.checkpoint_receipt is not None:
                    raise MsctlError(
                        "CLI_USAGE",
                        "legacy evaluation does not accept "
                        "--checkpoint-receipt",
                    )
                return not args.apply, self.evaluate(
                    release=release,
                    manifest=manifest,
                    approval_path=args.approval,
                    apply=args.apply,
                    evidence=evidence,
                    context=context,
                    checkpoint_receipt=checkpoint_receipt,
                )
            apply = bool(getattr(args, "apply", False))
            return not apply, self.cleanup(
                release=release,
                manifest=manifest,
                approval_path=getattr(args, "approval", None),
                apply=apply,
                context=context,
            )
        raise MsctlError(
            "EXTERNAL_OPERATION_UNSUPPORTED",
            "AWS provider does not support this lifecycle command",
            details={"operation": command},
        )


def build_aws_backend(
    *,
    profile: object,
    state_root: Path | str,
    environ: Mapping[str, str] | None = None,
    runner: AwsJsonRunner | None = None,
    approval_verifier: Callable[..., object] = verify_scope_approval,
) -> AwsP5Backend:
    environment = dict(os.environ if environ is None else environ)
    supplied_credential_variables = (
        set(environment) & _OPERATOR_CREDENTIAL_VARIABLES
    )
    runtime_environment = {
        name: value
        for name, value in environment.items()
        if name not in _OPERATOR_CREDENTIAL_VARIABLES
    }
    module_name = "cluster.aws.p5.profile"
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as error:
        raise MsctlError(
            "PROVIDER_ADAPTER_UNAVAILABLE",
            "the AWS GPU runtime adapter is not installed",
            details={"adapter": module_name},
        ) from error
    validator = getattr(
        module,
        "validate_aws_gpu_runtime_environment",
        getattr(module, "validate_runtime_environment", None),
    )
    if not callable(validator):
        raise MsctlError(
            "PROVIDER_ADAPTER_UNAVAILABLE",
            "the AWS GPU runtime adapter has no validator",
            details={"adapter": module_name},
        )
    try:
        runtime = validator(profile, runtime_environment)
    except MsctlError:
        raise
    except (TypeError, ValueError) as error:
        raise MsctlError(
            "AWS_RUNTIME_INVALID",
            "AWS runtime environment validation failed",
        ) from error
    instance_profile_arn = environment.get("MS_AWS_INSTANCE_PROFILE_ARN")
    if instance_profile_arn is None:
        raise MsctlError(
            "AWS_RUNTIME_INVALID",
            "MS_AWS_INSTANCE_PROFILE_ARN is required",
        )
    operator_credentials = None
    if runner is None or supplied_credential_variables:
        operator_credentials = load_operator_credential_process(
            environment,
            region=runtime.region,
        )
    return AwsP5Backend(
        profile=profile,
        runtime=runtime,
        instance_profile_arn=instance_profile_arn,
        state_root=state_root,
        runner=runner,
        approval_verifier=approval_verifier,
        environ=environment,
        operator_credentials=operator_credentials,
    )


AwsGpuBackend = AwsP5Backend
build_aws_gpu_backend = build_aws_backend
