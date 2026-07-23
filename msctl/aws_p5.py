"""Strict, dry-run-first AWS P5 lifecycle backend."""

from __future__ import annotations

import importlib
import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from cluster.aws.p5.attest_environment import (
    AttestationError,
    parse_environment_receipt_bytes,
    parse_runtime_lock_bytes,
    read_regular_input,
)

from .approval import verify_scope_approval
from .aws_argv import (
    ARGV_DOCUMENT_CONTENT,
    ARGV_DOCUMENT_NAME,
    ARGV_DOCUMENT_SHA256,
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
from .fsutil import atomic_write_at, open_directory
from .jsonutil import (
    canonical_json,
    canonical_sha256,
    load_json,
    require_exact_keys,
    require_object,
    require_sha256,
    sha256_file,
)
from .profile import AWS_P5_PROFILE
from .state import StateStore


INSTANCE_TYPE = "p5.48xlarge"
_INSTANCE_ID_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
_COMMAND_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{7,127}$")
_REGION_RE = re.compile(r"^[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+$")
_AMI_RE = re.compile(r"^ami-[0-9a-f]{8,17}$")
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
    "state",
    "instance_profile_arn",
    "provider",
    "seed",
    "cohort_sha256",
    "release_sha256",
    "dataset_sha256",
    "run_manifest_sha256",
}
_SELECTED_INSTANCE_FIELDS = _INSTANCE_FIELDS | {
    "ami_id",
    "container_digest",
    "profile_sha256",
    "runtime_sha256",
    "terminate_at",
}
_SAFE_PATH = "/usr/local/bin:/usr/bin:/bin"
_MAX_PAID_RUNTIME = timedelta(minutes=1440)
_AWS_PRIVATE_HOME = "/var/lib/memorysplit/aws-private-home"
_V3_PROFILE_ID = "aws-p5.48xlarge-v3"
_LOCAL_AWS_CONFIG = (
    "AWS_PROFILE",
    "AWS_CONFIG_FILE",
    "AWS_SHARED_CREDENTIALS_FILE",
)
_STATIC_AWS_CREDENTIALS = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
}
_AWS_US_EAST_1_DSA_CERTIFICATE = """-----BEGIN CERTIFICATE-----
MIIC7TCCAq0CCQCWukjZ5V4aZzAJBgcqhkjOOAQDMFwxCzAJBgNVBAYTAlVTMRkw
FwYDVQQIExBXYXNoaW5ndG9uIFN0YXRlMRAwDgYDVQQHEwdTZWF0dGxlMSAwHgYD
VQQKExdBbWF6b24gV2ViIFNlcnZpY2VzIExMQzAeFw0xMjAxMDUxMjU2MTJaFw0z
ODAxMDUxMjU2MTJaMFwxCzAJBgNVBAYTAlVTMRkwFwYDVQQIExBXYXNoaW5ndG9u
IFN0YXRlMRAwDgYDVQQHEwdTZWF0dGxlMSAwHgYDVQQKExdBbWF6b24gV2ViIFNl
cnZpY2VzIExMQzCCAbcwggEsBgcqhkjOOAQBMIIBHwKBgQCjkvcS2bb1VQ4yt/5e
ih5OO6kK/n1Lzllr7D8ZwtQP8fOEpp5E2ng+D6Ud1Z1gYipr58Kj3nssSNpI6bX3
VyIQzK7wLclnd/YozqNNmgIyZecN7EglK9ITHJLP+x8FtUpt3QbyYXJdmVMegN6P
hviYt5JH/nYl4hh3Pa1HJdskgQIVALVJ3ER11+Ko4tP6nwvHwh6+ERYRAoGBAI1j
k+tkqMVHuAFcvAGKocTgsjJem6/5qomzJuKDmbJNu9Qxw3rAotXau8Qe+MBcJl/U
hhy1KHVpCGl9fueQ2s6IL0CaO/buycU1CiYQk40KNHCcHfNiZbdlx1E9rpUp7bnF
lRa2v1ntMX3caRVDdbtPEWmdxSCYsYFDk4mZrOLBA4GEAAKBgEbmeve5f8LIE/Gf
MNmP9CM5eovQOGx5ho8WqD+aTebs+k2tn92BBPqeZqpWRa5P/+jrdKml1qx4llHW
MXrs3IgIb6+hUIB+S8dz8/mmO0bpr76RoZVCXYab2CZedFut7qc3WUH9+EUAH5mw
vSeDCOUMYQR7R9LINYwouHIziqQYMAkGByqGSM44BAMDLwAwLAIUWXBlk40xTwSw
7HX32MxXYruse9ACFBNGmdX2ZBrVNGrN9N2f6ROk0k9K
-----END CERTIFICATE-----
"""


def _verify_instance_identity_pkcs7(
    identity: Mapping[str, object],
    pkcs7: str,
    region: str,
) -> bool:
    if region != "us-east-1":
        raise MsctlError(
            "ENVIRONMENT_RECEIPT_INVALID",
            "no pinned AWS PKCS7 trust anchor exists for the selected region",
        )
    wrapped = (
        "-----BEGIN PKCS7-----\n"
        + "\n".join(
            pkcs7[index : index + 64] for index in range(0, len(pkcs7), 64)
        )
        + "\n-----END PKCS7-----\n"
    )
    with tempfile.TemporaryDirectory(prefix="msctl-iid-") as directory:
        root = Path(directory)
        signature = root / "identity.pkcs7"
        certificate = root / "aws-dsa.pem"
        signature.write_text(wrapped, encoding="ascii")
        certificate.write_text(
            _AWS_US_EAST_1_DSA_CERTIFICATE,
            encoding="ascii",
        )
        completed = subprocess.run(
            [
                "/usr/bin/openssl",
                "smime",
                "-verify",
                "-in",
                str(signature),
                "-inform",
                "PEM",
                "-certfile",
                str(certificate),
                "-nointern",
                "-noverify",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            shell=False,
            env={},
            timeout=10,
        )
    if completed.returncode != 0:
        return False
    try:
        signed = json.loads(
            completed.stdout.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False
    return signed == dict(identity)


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
    profile_id = getattr(profile, "profile_id", None)
    expected_seeds = (
        (1, 2, 3, 4)
        if profile_id == AWS_P5_PROFILE
        else tuple(range(10))
        if profile_id == _V3_PROFILE_ID
        else None
    )
    if (
        getattr(profile, "provider", None) != AWS_P5_PROFILE
        or expected_seeds is None
        or getattr(profile, "instance_type", None) != INSTANCE_TYPE
        or getattr(profile, "purchase_model", None) != "on_demand"
        or getattr(profile, "allocated_gpus", None) != 8
        or getattr(profile, "train_groups", None) != (4, 4)
        or getattr(profile, "assigned_seeds", None) != expected_seeds
    ):
        raise MsctlError(
            "PROFILE_INVALID",
            "AWS backend requires one exact frozen P5 provider profile",
        )


def _validate_runtime(runtime: object) -> None:
    s3_root = getattr(runtime, "s3_root", None)
    container_image = getattr(runtime, "container_image", None)
    container_digest = getattr(runtime, "container_digest", None)
    image_name = (
        container_image.partition("@")[0]
        if isinstance(container_image, str)
        else ""
    )
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
        or "/" not in image_name
        or "." not in image_name.partition("/")[0]
        or ":" in image_name.rsplit("/", 1)[-1]
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
) -> dict[str, object]:
    policy = {
        "submit": (1, 8, 1440, "scripts/run_train.py"),
        "resume": (1, 8, 1440, "scripts/run_train.py"),
        "evaluate": (1, 8, 360, "evals/confirmatory/runner.py"),
        "cancel": (1, 0, 0, "aws:ssm:cancel-command"),
        "cleanup": (1, 0, 0, "aws:ec2:terminate-instances"),
    }
    if operation not in policy:
        raise MsctlError(
            "APPROVAL_INVALID",
            "AWS operation has no approval resource policy",
        )
    jobs, gpus, wall_minutes, script = policy[operation]
    request: dict[str, object] = {
        "schema_version": 1,
        "operation": operation,
        "jobs": jobs,
        "allocated_gpus": gpus,
        "wall_minutes": wall_minutes,
        "gpu_hours": gpus * wall_minutes / 60.0,
        "gres": "gpu:h100:8" if gpus else "none",
        "script": script,
    }
    if bindings is not None:
        request.update(bindings)
    return request


class AwsP5Backend:
    """Provider backend whose only process boundary is an injected JSON runner."""

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
        inherited_static = sorted(
            name
            for name in _STATIC_AWS_CREDENTIALS
            if self.environ.get(name)
        )
        if inherited_static:
            raise MsctlError(
                "AWS_RUNTIME_INVALID",
                "local AWS controller must not inherit static key credentials",
                details={"variables": inherited_static},
            )
        self.local_aws_config: dict[str, str] = {}
        for name in _LOCAL_AWS_CONFIG:
            value = self.environ.get(name)
            if value is None or value == "":
                continue
            if not isinstance(value, str) or "\x00" in value:
                raise MsctlError(
                    "AWS_RUNTIME_INVALID",
                    "local AWS controller configuration is invalid",
                    details={"variable": name},
                )
            self.local_aws_config[name] = value
        configured_home = self.environ.get("HOME")
        if self.local_aws_config and configured_home:
            if (
                not isinstance(configured_home, str)
                or "\x00" in configured_home
                or not Path(configured_home).is_absolute()
            ):
                raise MsctlError(
                    "AWS_RUNTIME_INVALID",
                    "local AWS controller HOME is invalid",
                )
            self.controller_home = configured_home
        else:
            self.controller_home = "/tmp"

    def _aws_argv(
        self,
        service: str,
        action: str,
        *arguments: str,
        query: str,
    ) -> list[str]:
        return [
            "env",
            "-i",
            *[
                f"{name}={self.local_aws_config[name]}"
                for name in _LOCAL_AWS_CONFIG
                if name in self.local_aws_config
            ],
            f"AWS_REGION={self.runtime.region}",
            f"HOME={self.controller_home}",
            f"PATH={_SAFE_PATH}",
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
            "provider": AWS_P5_PROFILE,
            "region": self.runtime.region,
            **output,
        }

    def _validate_manifest(self, manifest: object) -> None:
        runs = getattr(manifest, "runs", ())
        if (
            getattr(manifest, "provider", None) != AWS_P5_PROFILE
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
            getattr(manifest, "schema_version", None) != 2
            or not isinstance(getattr(manifest, "source_commit", None), str)
            or _COMMIT_RE.fullmatch(manifest.source_commit) is None
        ):
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "AWS lifecycle requires a v2 source-bound run manifest",
            )
        for field in (
            "release_sha256",
            "dataset_sha256",
            "cohort_assignment_sha256",
            "study_lock_sha256",
            "sha256",
        ):
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

    def _validate_release(self, release: object, manifest: object) -> None:
        if (
            getattr(release, "provider", None) != AWS_P5_PROFILE
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
        return f"/mnt/memorysplit/releases/{release.archive_sha256}"

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
    ) -> dict[str, object]:
        lifecycle_evidence = dict(
            evidence
            or {
                "dataset_pointer_sha256": manifest.dataset_sha256,
                "dataset_verification_sha256": manifest.dataset_sha256,
                "environment_receipt_sha256": self._runtime_sha256(),
            }
        )
        if set(lifecycle_evidence) != {
            "dataset_pointer_sha256",
            "dataset_verification_sha256",
            "environment_receipt_sha256",
        }:
            raise MsctlError(
                "LIFECYCLE_EVIDENCE_INVALID",
                "AWS lifecycle evidence fields do not match the contract",
            )
        for field, digest in lifecycle_evidence.items():
            require_sha256(digest, label=field)
        release_root = self._release_root(release)
        staging = "/mnt/memorysplit/staging"
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
            release_bucket, release_prefix = self._s3_location(
                f"releases/{release.archive_sha256}/release.zip"
            )
            receipt_bucket, receipt_key = self._s3_location(
                f"releases/{release.archive_sha256}/RELEASE.json"
            )
            cohort_bucket, cohort_key = self._s3_location(
                f"releases/{release.archive_sha256}/cohort-assignment-v2.json"
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
                            "/mnt/memorysplit/dataset",
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
                            release_prefix,
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
                                "cohort-assignment-v2.json"
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
                            "/mnt/memorysplit/dataset",
                            "--no-follow-symlinks",
                            "--only-show-errors",
                        ],
                    },
                    {
                        "name": "bootstrap",
                        "argv": [
                        "/usr/bin/python3",
                        "/opt/memorysplit/cluster/aws/p5/bootstrap.py",
                        "--profile",
                        "/opt/memorysplit/cluster/profiles/aws-p5.48xlarge.json",
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
                        "/mnt/memorysplit/dataset/receipt.json",
                        "--dataset-receipt-sha256",
                        manifest.dataset_sha256,
                        "--cohort-assignment",
                        (
                            f"{staging}/releases/{release.archive_sha256}/"
                            "cohort-assignment-v2.json"
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
                        "/opt/memorysplit/msctl/aws_launch_manifest.py",
                        "--out",
                        f"{staging}/launcher-manifest-{manifest.sha256}.json",
                        "--scratch-root",
                        "/mnt/memorysplit",
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
                        "--bootstrap-receipt",
                        f"{staging}/bootstrap-receipt.json",
                        "--corpus-receipt",
                        "/mnt/memorysplit/dataset/receipt.json",
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
            resume_root = f"{staging}/resume/{receipt_sha256}"
            receipt_bucket, receipt_key = self._s3_location(
                f"checkpoints/receipts/{receipt_sha256}.json"
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
                    f"checkpoints/sha256/{row['resume_sha256']}.pt"
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
                f"{release_root}/cluster/profiles/aws-p5.48xlarge.json",
                "--repo-root",
                release_root,
                "--scratch-root",
                "/mnt/memorysplit",
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
                f"{release_root}/cluster/profiles/aws-p5.48xlarge.json",
                "--repo-root",
                release_root,
                "--scratch-root",
                "/mnt/memorysplit",
                "--apply",
            ]
        steps.append({"name": "paired-launch", "argv": launcher_argv})
        return {
            "schema_version": 1,
            "operation": operation,
            "provider": AWS_P5_PROFILE,
            "seed": manifest.seed,
            "release_sha256": release.archive_sha256,
            "run_manifest_sha256": manifest.sha256,
            "dataset_sha256": manifest.dataset_sha256,
            **lifecycle_evidence,
            "runtime_sha256": self._runtime_sha256(),
            "environment": {
                "AWS_REGION": self.runtime.region,
                "MS_AWS_AMI_ID": self.runtime.ami_id,
                "MS_CONTAINER_DIGEST": self.runtime.container_digest,
                "MS_CONTAINER_IMAGE": self.runtime.container_image,
                "MS_RUNTIME_GID": str(getattr(self.runtime, "gid", 1000)),
                "MS_RUNTIME_UID": str(getattr(self.runtime, "uid", 1000)),
                "MS_S3_ROOT": self.runtime.s3_root,
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
                    "Name=tag:MemorySplitProvider,Values=aws-p5.48xlarge",
                    f"Name=tag:MemorySplitSeed,Values={manifest.seed}",
                    "Name=instance-state-name,Values=pending,running,stopping",
                ]
            )
        query = (
            "{instances:Reservations[].Instances[]."
            "{instance_id:InstanceId,instance_type:InstanceType,"
            "state:State.Name,instance_profile_arn:IamInstanceProfile.Arn,"
            "provider:Tags[?Key=='MemorySplitProvider']|[0].Value,"
            "seed:to_number(Tags[?Key=='MemorySplitSeed']|[0].Value),"
            "cohort_sha256:Tags[?Key=='MemorySplitCohortSHA256']|[0].Value,"
            "release_sha256:Tags[?Key=='MemorySplitReleaseSHA256']|[0].Value,"
            "dataset_sha256:Tags[?Key=='MemorySplitDatasetSHA256']|[0].Value,"
            "run_manifest_sha256:"
            "Tags[?Key=='MemorySplitRunManifestSHA256']|[0].Value}}"
        )
        return self._aws_argv(
            "ec2",
            "describe-instances",
            *arguments,
            query=query,
        )

    def _selected_instance_argv(self, instance_id: str) -> list[str]:
        if _INSTANCE_ID_RE.fullmatch(instance_id) is None:
            raise MsctlError(
                "INSTANCE_BINDING_MISMATCH",
                "EC2 instance ID is invalid",
            )
        query = (
            "{instances:Reservations[].Instances[]."
            "{instance_id:InstanceId,instance_type:InstanceType,"
            "state:State.Name,instance_profile_arn:IamInstanceProfile.Arn,"
            "ami_id:ImageId,"
            "provider:Tags[?Key=='MemorySplitProvider']|[0].Value,"
            "seed:to_number(Tags[?Key=='MemorySplitSeed']|[0].Value),"
            "cohort_sha256:Tags[?Key=='MemorySplitCohortSHA256']|[0].Value,"
            "release_sha256:Tags[?Key=='MemorySplitReleaseSHA256']|[0].Value,"
            "dataset_sha256:Tags[?Key=='MemorySplitDatasetSHA256']|[0].Value,"
            "run_manifest_sha256:"
            "Tags[?Key=='MemorySplitRunManifestSHA256']|[0].Value,"
            "profile_sha256:Tags[?Key=='MemorySplitProfileSHA256']|[0].Value,"
            "runtime_sha256:Tags[?Key=='MemorySplitRuntimeSHA256']|[0].Value,"
            "container_digest:"
            "Tags[?Key=='MemorySplitContainerDigest']|[0].Value,"
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
    ) -> dict[str, object]:
        return {
            "provider": AWS_P5_PROFILE,
            "seed": manifest.seed,
            "cohort_sha256": manifest.cohort_assignment_sha256,
            "release_sha256": manifest.release_sha256,
            "dataset_sha256": manifest.dataset_sha256,
            "run_manifest_sha256": manifest.sha256,
            "profile_sha256": self.profile.sha256,
            "runtime_sha256": self._runtime_sha256(),
            "container_digest": self.runtime.container_digest,
            "terminate_at": terminate_at,
        }

    def _parse_selected_instance(
        self,
        output: object,
        manifest: object,
        *,
        instance_id: str,
        terminate_at: str,
        require_bound: bool,
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
            _SELECTED_INSTANCE_FIELDS,
            label="selected EC2 instance",
        )
        if (
            row["instance_id"] != instance_id
            or row["instance_type"] != INSTANCE_TYPE
            or row["state"] != "running"
            or row["instance_profile_arn"] != self.instance_profile_arn
            or row["ami_id"] != self.runtime.ami_id
        ):
            raise MsctlError(
                "INSTANCE_BINDING_MISMATCH",
                "operator-selected instance has the wrong immutable runtime",
            )
        expected = self._selected_binding(
            manifest,
            terminate_at=terminate_at,
        )
        observed = {field: row[field] for field in expected}
        if require_bound:
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
    ) -> str:
        binding = self._selected_binding(
            manifest,
            terminate_at=terminate_at,
        )
        names = {
            "provider": "MemorySplitProvider",
            "seed": "MemorySplitSeed",
            "cohort_sha256": "MemorySplitCohortSHA256",
            "release_sha256": "MemorySplitReleaseSHA256",
            "dataset_sha256": "MemorySplitDatasetSHA256",
            "run_manifest_sha256": "MemorySplitRunManifestSHA256",
            "profile_sha256": "MemorySplitProfileSHA256",
            "runtime_sha256": "MemorySplitRuntimeSHA256",
            "container_digest": "MemorySplitContainerDigest",
            "terminate_at": "MemorySplitTerminateAt",
        }
        return canonical_json(
            [
                {"Key": names[field], "Value": str(binding[field])}
                for field in names
            ]
        ).decode("ascii")

    def _bind_selected_instance(
        self,
        manifest: object,
        *,
        instance_id: str,
        terminate_at: str,
    ) -> dict[str, object]:
        selected_argv = self._selected_instance_argv(instance_id)
        self._parse_selected_instance(
            self._run(selected_argv, operation="validate selected instance"),
            manifest,
            instance_id=instance_id,
            terminate_at=terminate_at,
            require_bound=False,
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
            self._instance_tags(manifest, terminate_at=terminate_at),
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
    ) -> dict[str, object]:
        selected_argv = self._selected_instance_argv(instance_id)
        row = self._parse_selected_instance(
            self._run(
                selected_argv,
                operation=f"verify {operation} instance",
            ),
            manifest,
            instance_id=instance_id,
            terminate_at=terminate_at,
            require_bound=True,
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

    def _parse_instances(
        self,
        output: object,
        manifest: object,
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
                _INSTANCE_FIELDS,
                label=f"EC2 instance[{index}]",
            )
            if (
                not isinstance(row["instance_id"], str)
                or _INSTANCE_ID_RE.fullmatch(row["instance_id"]) is None
                or row["instance_type"] != INSTANCE_TYPE
                or row["state"] not in _ACTIVE_INSTANCE_STATES
                or row["instance_profile_arn"] != self.instance_profile_arn
                or row["provider"] != AWS_P5_PROFILE
                or row["seed"] != manifest.seed
                or row["cohort_sha256"]
                != manifest.cohort_assignment_sha256
                or row["release_sha256"] != manifest.release_sha256
                or row["dataset_sha256"] != manifest.dataset_sha256
                or row["run_manifest_sha256"] != manifest.sha256
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
                "multiple active P5 instances claim the same seed",
                details={
                    "seed": manifest.seed,
                    "instance_ids": sorted(
                        str(row["instance_id"]) for row in instances
                    ),
                },
            )
        return instances

    def discover_instances(self, manifest: object) -> list[dict[str, object]]:
        self._validate_manifest(manifest)
        return self._parse_instances(
            self._run(
                self._discover_argv(manifest),
                operation="instance discovery",
            ),
            manifest,
        )

    def _validate_exact_instance(
        self,
        manifest: object,
        instance_id: str,
    ) -> dict[str, object]:
        rows = self._parse_instances(
            self._run(
                self._discover_argv(manifest, instance_id=instance_id),
                operation="instance validation",
            ),
            manifest,
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
                "exactly one managed P5 instance must be online",
            )
        row = _aws_output_object(
            rows[0],
            {"instance_id", "ping_status"},
            label="SSM managed instance",
        )
        if row != {"instance_id": instance_id, "ping_status": "Online"}:
            raise MsctlError(
                "SSM_UNAVAILABLE",
                "P5 instance is not online in Systems Manager",
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
            resources=resources or aws_resource_request(operation),
            profile=self.profile,
            environ=self.environ,
        )

    def _same_state_binding(
        self,
        state: dict[str, object],
        manifest: object,
    ) -> bool:
        by_id = {run.run_id: run for run in manifest.runs}
        run = by_id.get(state.get("run_id"))
        instance_id = state.get("instance_id")
        command_id = state.get("command_id")
        return (
            run is not None
            and state.get("provider") == AWS_P5_PROFILE
            and state.get("seed") == manifest.seed
            and state.get("arm") == run.arm
            and state.get("config_sha256") == run.config_sha256
            and state.get("release_sha256") == manifest.release_sha256
            and state.get("dataset_sha256") == manifest.dataset_sha256
            and state.get("run_manifest_sha256") == manifest.sha256
            and state.get("cohort_assignment_sha256")
            == manifest.cohort_assignment_sha256
            and state.get("study_lock_sha256") == manifest.study_lock_sha256
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
    ) -> dict[str, object]:
        operation_intent = self._training_operation_intent(
            operation="submit",
            release=release,
            manifest=manifest,
            terminate_at=terminate_at,
        )
        return {
            "provider": AWS_P5_PROFILE,
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
    ) -> dict[str, object]:
        return aws_resource_request(
            "submit",
            bindings=self._execution_bindings(
                release=release,
                manifest=manifest,
                instance_id=instance_id,
                terminate_at=terminate_at,
            ),
        )

    def _execution_bindings(
        self,
        *,
        release: object,
        manifest: object,
        instance_id: str,
        terminate_at: str,
    ) -> dict[str, object]:
        return {
            "ami_id": self.runtime.ami_id,
            "container_image": self.runtime.container_image,
            "container_digest": self.runtime.container_digest,
            "instance_id": instance_id,
            "profile_sha256": self.profile.sha256,
            "release_sha256": release.archive_sha256,
            "run_manifest_sha256": manifest.sha256,
            "runtime_sha256": self._runtime_sha256(),
            "seed": manifest.seed,
            "terminate_at": terminate_at,
        }

    def _cleanup_resources(
        self,
        *,
        release: object,
        manifest: object,
        instance_id: str,
        terminate_at: str,
    ) -> dict[str, object]:
        return aws_resource_request(
            "cleanup",
            bindings=self._execution_bindings(
                release=release,
                manifest=manifest,
                instance_id=instance_id,
                terminate_at=terminate_at,
            ),
        )

    def _evaluation_resources(
        self,
        *,
        release: object,
        manifest: object,
        instance_id: str,
        terminate_at: str,
    ) -> dict[str, object]:
        return aws_resource_request(
            "evaluate",
            bindings=self._execution_bindings(
                release=release,
                manifest=manifest,
                instance_id=instance_id,
                terminate_at=terminate_at,
            ),
        )

    def _resume_resources(
        self,
        *,
        release: object,
        manifest: object,
        instance_id: str,
        terminate_at: str,
        checkpoint_receipt_sha256: str,
    ) -> dict[str, object]:
        return aws_resource_request(
            "resume",
            bindings={
                **self._execution_bindings(
                    release=release,
                    manifest=manifest,
                    instance_id=instance_id,
                    terminate_at=terminate_at,
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
        argv = self._aws_argv(
            "ec2",
            "describe-instance-type-offerings",
            "--location-type",
            "region",
            "--filters",
            f"Name=instance-type,Values={INSTANCE_TYPE}",
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
                row["instance_type"] != INSTANCE_TYPE
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
                "P5 is not offered in the selected region",
                details={
                    "region": self.runtime.region,
                    "instance_type": INSTANCE_TYPE,
                },
            )
        return {
            "provider": AWS_P5_PROFILE,
            "region": self.runtime.region,
            "instance_type": INSTANCE_TYPE,
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
        cohort_member = release.members.get(
            "configs/cohort-assignment-v2.json"
        )
        study_member = release.members.get("configs/preregistration-v2.yaml")
        if (
            cohort_member is None
            or cohort_member.get("sha256")
            != manifest.cohort_assignment_sha256
            or study_member is None
            or study_member.get("sha256") != manifest.study_lock_sha256
        ):
            raise MsctlError(
                "RELEASE_COHORT_MISMATCH",
                "AWS release does not bind the manifest cohort and study lock",
            )
        return release, manifest

    def _load_lifecycle_evidence(
        self,
        *,
        manifest: object,
        dataset_pointer: Path | str,
        dataset_root: Path | str | None,
        dataset_verification: Path | str | None,
        environment_receipt: Path | str,
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
        if (
            pointer["schema_version"] != 1
            or pointer["provider"] != AWS_P5_PROFILE
            or pointer["materialization"] != "s3"
            or pointer["durable_uri_env"] != "MS_S3_ROOT"
            or pointer["scratch_root"] != "/mnt/memorysplit"
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
        environment_bytes = environment_path.read_bytes()
        environment = require_object(
            load_json(environment_path, label="AWS environment receipt"),
            label="AWS environment receipt",
        )
        require_exact_keys(
            environment,
            {
                "schema_version",
                "profile_sha256",
                "container_image_digest",
                "aws_instance_identity_document",
                "aws_instance_identity_pkcs7",
            },
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
            "environment_receipt_sha256": hashlib.sha256(
                environment_bytes
            ).hexdigest(),
        }

    def render(
        self,
        *,
        release: object,
        manifest: object,
        evidence: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        self._validate_manifest(manifest)
        self._validate_release(release, manifest)
        operation_intent = self._training_operation_intent(
            operation="render",
            release=release,
            manifest=manifest,
            evidence=evidence,
        )
        return {
            "provider": AWS_P5_PROFILE,
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
    ) -> dict[str, object]:
        self._validate_manifest(manifest)
        self._validate_release(release, manifest)
        store = StateStore(self.state_root)
        with store.locked():
            states = self._paired_states(store, manifest)
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
            if cached:
                statuses = {str(state.get("status")) for state in states}
                status = (
                    next(iter(statuses)) if len(statuses) == 1 else "Mixed"
                )
                return {
                    "provider": AWS_P5_PROFILE,
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
                "provider": AWS_P5_PROFILE,
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
                "provider": AWS_P5_PROFILE,
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
            "provider": AWS_P5_PROFILE,
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

    def _immutable_s3_object_argv(
        self,
        *,
        local_path: Path,
        relative: str,
        sha256: str,
        byte_count: int,
        upload: bool,
    ) -> list[str]:
        bucket, key = self._s3_location(relative)
        if upload:
            checksum = base64.b64encode(bytes.fromhex(sha256)).decode("ascii")
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
        receipt_bytes = canonical_json(receipt_value)
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
                "s3_relative": f"checkpoints/receipts/{receipt_sha256}.json",
                "sha256": receipt_sha256,
                "bytes": len(receipt_bytes),
                "device": None,
                "inode": None,
            }
        ]
        for arm in ("dense", "split90"):
            checkpoint = checkpoints[arm]
            path = Path(checkpoint.path)
            try:
                before = path.stat(follow_symlinks=False)
                digest = sha256_file(path)
                after = path.stat(follow_symlinks=False)
            except OSError as error:
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    "checkpoint file is unavailable before publication",
                    details={"arm": arm},
                ) from error
            if (
                path.is_symlink()
                or before.st_nlink != 1
                or (
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
                or digest != checkpoint.sha256
            ):
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    "checkpoint identity changed before immutable publication",
                    details={"arm": arm},
                )
            objects.append(
                {
                    "path": f"{arm}.pt",
                    "local_path": path,
                    "s3_relative": f"checkpoints/sha256/{digest}.pt",
                    "sha256": digest,
                    "bytes": before.st_size,
                    "device": before.st_dev,
                    "inode": before.st_ino,
                }
            )
        commands = [
            self._immutable_s3_object_argv(
                local_path=Path(item["local_path"]),
                relative=str(item["s3_relative"]),
                sha256=str(item["sha256"]),
                byte_count=int(item["bytes"]),
                upload=upload,
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
                    if key != "local_path"
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
        for index, item in enumerate(objects):
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
            )
        result["verified"] = True
        result["objects"] = [
            {
                key: value
                for key, value in item.items()
                if key != "local_path"
            }
            for item in objects
        ]
        return result

    def env_ensure(
        self,
        *,
        root: Path | str,
        apply: bool,
        runtime_lock: Path | str | None = None,
        control_bundle: Path | str | None = None,
        instance_id: str | None = None,
        receipt: Path | str | None = None,
        lock: Path | str | None = None,
    ) -> dict[str, object]:
        if getattr(self.profile, "profile_id", None) != _V3_PROFILE_ID:
            raise MsctlError(
                "STATIC_ENVIRONMENT_FORBIDDEN",
                "AWS runtime identity must come from the package "
                "runtime_attested contract and an authenticated instance receipt",
            )
        if (
            lock is not None
            or runtime_lock is None
            or control_bundle is None
            or instance_id is None
            or receipt is None
        ):
            raise MsctlError(
                "CLI_USAGE",
                "v3 environment attestation requires explicit runtime lock, "
                "control bundle, instance ID, receipt, and output root",
            )
        if (
            not isinstance(instance_id, str)
            or _INSTANCE_ID_RE.fullmatch(instance_id) is None
        ):
            raise MsctlError(
                "ENVIRONMENT_RECEIPT_INVALID",
                "environment attestation requires one immutable instance ID",
            )
        try:
            lock_bytes = read_regular_input(
                runtime_lock,
                label="AWS runtime lock",
                maximum_bytes=1024 * 1024,
            )
            control_bytes = read_regular_input(
                control_bundle,
                label="AWS control bundle",
            )
            receipt_bytes = read_regular_input(
                receipt,
                label="AWS environment receipt",
                maximum_bytes=1024 * 1024,
            )
            lock = parse_runtime_lock_bytes(lock_bytes)
            environment_receipt = parse_environment_receipt_bytes(
                receipt_bytes
            )
        except (AttestationError, OSError, TypeError, ValueError) as error:
            raise MsctlError(
                "ENVIRONMENT_RECEIPT_INVALID",
                "AWS environment attestation inputs are invalid",
            ) from error

        lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
        control_sha256 = hashlib.sha256(control_bytes).hexdigest()
        identity = environment_receipt["aws_instance_identity_document"]
        if (
            lock["profile_sha256"] != self.profile.sha256
            or lock["control_bundle_sha256"] != control_sha256
            or lock["ami_id"] != self.runtime.ami_id
            or lock["container_image"] != self.runtime.container_image
            or lock["container_image_digest"] != self.runtime.container_digest
            or environment_receipt["profile_sha256"] != self.profile.sha256
            or environment_receipt["runtime_lock_sha256"] != lock_sha256
            or environment_receipt["control_bundle_sha256"] != control_sha256
            or environment_receipt["source_commit"] != lock["source_commit"]
            or environment_receipt["source_tree"] != lock["source_tree"]
            or environment_receipt["container_image"]
            != self.runtime.container_image
            or environment_receipt["container_image_digest"]
            != self.runtime.container_digest
            or environment_receipt["account_id"] != identity["accountId"]
            or environment_receipt["instance_id"] != instance_id
            or environment_receipt["region"] != self.runtime.region
            or environment_receipt["ami_id"] != self.runtime.ami_id
            or environment_receipt["runtime_facts"] != lock["versions"]
        ):
            raise MsctlError(
                "ENVIRONMENT_RECEIPT_INVALID",
                "AWS environment receipt does not bind the selected runtime",
            )
        try:
            signature_valid = self.identity_verifier(
                identity,
                str(environment_receipt["aws_instance_identity_pkcs7"]),
                self.runtime.region,
            )
        except MsctlError:
            raise
        except Exception as error:
            raise MsctlError(
                "ENVIRONMENT_RECEIPT_INVALID",
                "AWS instance identity signature verification failed",
            ) from error
        if signature_valid is not True:
            raise MsctlError(
                "ENVIRONMENT_RECEIPT_INVALID",
                "AWS instance identity signature is not valid",
            )

        remote_root = Path(os.path.abspath(os.fspath(root)))
        receipt_path = Path(os.path.abspath(os.fspath(receipt)))
        attestation_argv = [
            "/usr/bin/python3",
            str(
                remote_root
                / "cluster"
                / "aws"
                / "p5"
                / "attest_environment.py"
            ),
            "--profile",
            str(
                remote_root
                / "cluster"
                / "profiles"
                / "aws-p5.48xlarge-v3.json"
            ),
            "--runtime-lock",
            str(Path(runtime_lock)),
            "--control-bundle",
            str(Path(control_bundle)),
            "--out",
            str(receipt_path),
            "--apply",
        ]
        ssm_intent = {
            "schema_version": 1,
            "operation": "attest-environment",
            "instance_id": instance_id,
            "environment": {
                "AWS_REGION": self.runtime.region,
                "MS_AWS_AMI_ID": self.runtime.ami_id,
                "MS_CONTAINER_DIGEST": self.runtime.container_digest,
                "MS_CONTAINER_IMAGE": self.runtime.container_image,
            },
            "steps": [
                {
                    "name": "attest-environment",
                    "argv": attestation_argv,
                }
            ],
        }
        receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        receipt_key = (
            f"environments/{receipt_sha256}/receipt.json"
        )
        result = {
            "provider": AWS_P5_PROFILE,
            "profile_id": _V3_PROFILE_ID,
            "environment_receipt_sha256": receipt_sha256,
            "receipt_key": receipt_key,
            "receipt_uri": (
                f"{self.runtime.s3_root.rstrip('/')}/{receipt_key}"
            ),
            "receipt_path": str(receipt_path),
            "remote_attestation_argv": attestation_argv,
            "ssm_intent": ssm_intent,
            "published": False,
            "verified": True,
        }
        if not apply:
            return result

        instance_output = _aws_output_object(
            self._run(
                self._aws_argv(
                    "ec2",
                    "describe-instances",
                    "--instance-ids",
                    instance_id,
                    query=(
                        "{instance:{account_id:Reservations[0].OwnerId,"
                        "instance_id:Reservations[0].Instances[0].InstanceId,"
                        "image_id:Reservations[0].Instances[0].ImageId,"
                        "instance_type:Reservations[0].Instances[0].InstanceType,"
                        "state:Reservations[0].Instances[0].State.Name,"
                        "architecture:Reservations[0].Instances[0].Architecture,"
                        "private_ip:Reservations[0].Instances[0].PrivateIpAddress}}"
                    ),
                ),
                operation="verify selected environment instance",
            ),
            {"instance"},
            label="selected environment instance output",
        )
        selected_instance = _aws_output_object(
            instance_output["instance"],
            {
                "account_id",
                "instance_id",
                "image_id",
                "instance_type",
                "state",
                "architecture",
                "private_ip",
            },
            label="selected environment instance",
        )
        image_output = _aws_output_object(
            self._run(
                self._aws_argv(
                    "ec2",
                    "describe-images",
                    "--image-ids",
                    self.runtime.ami_id,
                    "--owners",
                    str(lock["ami_owner_id"]),
                    query=(
                        "{image:{image_id:Images[0].ImageId,"
                        "owner_id:Images[0].OwnerId,"
                        "state:Images[0].State,"
                        "architecture:Images[0].Architecture}}"
                    ),
                ),
                operation="verify selected environment AMI",
            ),
            {"image"},
            label="selected environment AMI output",
        )
        selected_image = _aws_output_object(
            image_output["image"],
            {"image_id", "owner_id", "state", "architecture"},
            label="selected environment AMI",
        )
        if (
            selected_instance["account_id"]
            != environment_receipt["account_id"]
            or selected_instance["instance_id"] != instance_id
            or selected_instance["image_id"] != self.runtime.ami_id
            or selected_instance["instance_type"] != INSTANCE_TYPE
            or selected_instance["state"] != "running"
            or selected_instance["architecture"] != identity["architecture"]
            or selected_instance["private_ip"] != identity["privateIp"]
            or selected_image["image_id"] != self.runtime.ami_id
            or selected_image["owner_id"] != lock["ami_owner_id"]
            or selected_image["state"] != "available"
            or selected_image["architecture"] != identity["architecture"]
        ):
            raise MsctlError(
                "ENVIRONMENT_RECEIPT_INVALID",
                "selected AWS instance or AMI differs from authenticated receipt",
            )

        checksum = base64.b64encode(
            bytes.fromhex(receipt_sha256)
        ).decode("ascii")
        bucket, key = self._s3_location(receipt_key)
        put_version: str | None = None
        with tempfile.TemporaryDirectory(
            prefix="msctl-environment-receipt-"
        ) as staging_directory:
            staged_receipt = (
                Path(staging_directory) / "environment-receipt.json"
            )
            staged_receipt.write_bytes(receipt_bytes)
            staged_receipt.chmod(0o600)
            put_argv = self._aws_argv(
                "s3api",
                "put-object",
                "--bucket",
                bucket,
                "--key",
                key,
                "--body",
                str(staged_receipt),
                "--content-length",
                str(len(receipt_bytes)),
                "--checksum-algorithm",
                "SHA256",
                "--checksum-sha256",
                checksum,
                "--metadata",
                f"environment-receipt-sha256={receipt_sha256}",
                "--if-none-match",
                "*",
                query=(
                    "{object:{checksum_sha256:ChecksumSHA256,"
                    "version_id:VersionId}}"
                ),
            )
            try:
                put_output = _aws_output_object(
                    self._run(
                        put_argv,
                        operation="publish environment receipt",
                    ),
                    {"object"},
                    label="environment receipt publication output",
                )
                put_object = _aws_output_object(
                    put_output["object"],
                    {"checksum_sha256", "version_id"},
                    label="environment receipt publication",
                )
                if (
                    put_object["checksum_sha256"] != checksum
                    or not isinstance(put_object["version_id"], str)
                    or put_object["version_id"] in {"", "null"}
                ):
                    raise MsctlError(
                        "S3_OBJECT_MISMATCH",
                        "published environment receipt lacks checksum or version",
                    )
                put_version = put_object["version_id"]
            except MsctlError as error:
                if error.code != "AWS_COMMAND_FAILED":
                    raise

        head_output = _aws_output_object(
            self._run(
                self._aws_argv(
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
                ),
                operation="verify environment receipt publication",
            ),
            {"object"},
            label="environment receipt head output",
        )
        head_object = _aws_output_object(
            head_output["object"],
            {
                "checksum_sha256",
                "content_length",
                "metadata",
                "version_id",
            },
            label="environment receipt object",
        )
        version_id = head_object["version_id"]
        if (
            head_object["checksum_sha256"] != checksum
            or type(head_object["content_length"]) is not int
            or head_object["content_length"] != len(receipt_bytes)
            or head_object["metadata"]
            != {"environment-receipt-sha256": receipt_sha256}
            or not isinstance(version_id, str)
            or version_id in {"", "null"}
            or (put_version is not None and version_id != put_version)
        ):
            raise MsctlError(
                "S3_OBJECT_MISMATCH",
                "environment receipt publication does not match local bytes",
            )
        return {
            **result,
            "published": True,
            "version_id": version_id,
        }

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
    ) -> None:
        wrapped = _aws_output_object(
            output,
            {"object"},
            label="S3 dataset object response",
        )
        row = _aws_output_object(
            wrapped["object"],
            {"bytes", "checksum_sha256", "metadata", "version_id"},
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
            "provider": AWS_P5_PROFILE,
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

    def collect(
        self,
        *,
        source: str,
        out: Path | str,
        apply: bool,
    ) -> dict[str, object]:
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
                "provider": AWS_P5_PROFILE,
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
            "provider": AWS_P5_PROFILE,
            "s3_uri": f"s3://{bucket}/{key}",
            "out": str(destination),
            "collected": 1,
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
    ) -> dict[str, object]:
        now = _timestamp()
        state: dict[str, object] = {
            "schema_version": 1,
            "run_id": run.run_id,
            "arm": run.arm,
            "seed": run.seed,
            "provider": AWS_P5_PROFILE,
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
            "study_lock_sha256": manifest.study_lock_sha256,
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
        if operation == "resume":
            state["checkpoint_receipt_sha256"] = checkpoint_receipt_sha256
            state["prior_command_ids"] = list(prior_command_ids or [])
        return state

    def _write_paired_states(
        self,
        store: StateStore,
        manifest: object,
        states: Sequence[dict[str, object]],
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
        store.write_aws_pair(
            manifest.sha256,
            {
                "schema_version": 1,
                "provider": AWS_P5_PROFILE,
                "run_manifest_sha256": manifest.sha256,
                "operation_id": next(iter(operation_ids)),
                "states": [dict(state) for state in states],
            },
        )
        for state in states:
            store.write_run(str(state["run_id"]), state)

    def _repair_paired_states(
        self,
        store: StateStore,
        manifest: object,
    ) -> None:
        journal = store.read_aws_pair(manifest.sha256)
        if journal is None:
            return
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
                "IntentSHA256": [published["intent_sha256"]],
                "IntentUri": [published["intent_uri"]],
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
        instance_id: str,
        terminate_at: str,
        approval_path: Path | str | None,
        apply: bool,
        evidence: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        self._validate_manifest(manifest)
        self._validate_release(release, manifest)
        self._validate_submit_selection(instance_id, terminate_at)
        core = self._training_operation_intent(
            operation="submit",
            release=release,
            manifest=manifest,
            terminate_at=terminate_at,
            evidence=evidence,
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
            )
            plan["operation_intent"] = operation_intent
            plan["operation_id"] = operation_intent["operation_id"]
            return plan
        resources = self._submit_resources(
            release=release,
            manifest=manifest,
            instance_id=instance_id,
            terminate_at=terminate_at,
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
            self._repair_paired_states(store, manifest)
            states = [store.read_run(run.run_id) for run in manifest.runs]
            existing = [state for state in states if state is not None]
            if existing:
                if len(existing) != 2 or not all(
                    self._same_state_binding(state, manifest)
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
                        if started or terminal:
                            return {
                                "provider": AWS_P5_PROFILE,
                                "seed": manifest.seed,
                                "instance_id": instance_id,
                                "operation_id": next(iter(operation_ids)),
                                "status": (
                                    "REMOTE_TERMINAL"
                                    if terminal
                                    else "REMOTE_STARTED"
                                ),
                                "submitted": 0,
                                "idempotent": True,
                            }
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
                        "provider": AWS_P5_PROFILE,
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
                    "provider": AWS_P5_PROFILE,
                    "seed": manifest.seed,
                    "instance_id": instance_id,
                    "command_id": command_id,
                    "status": status,
                    "submitted": 0,
                    "idempotent": True,
                    "active": status in _ACTIVE_COMMAND_STATES,
                }

            discovered = self.discover_instances(manifest)
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
            selected_argv = self._selected_instance_argv(instance_id)
            self._parse_selected_instance(
                self._run(
                    selected_argv,
                    operation="preflight selected instance",
                ),
                manifest,
                instance_id=instance_id,
                terminate_at=terminate_at,
                require_bound=False,
            )
            self._require_ssm_online(instance_id)
            self._ensure_argv_document()
            self._bind_selected_instance(
                manifest,
                instance_id=instance_id,
                terminate_at=terminate_at,
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
                )
                for run in manifest.runs
            ]
            self._write_paired_states(store, manifest, new_states)
            now = _timestamp()
            for state in new_states:
                state["status"] = "SENDING"
                state["send_attempted"] = True
                state["updated_at"] = now
            self._write_paired_states(store, manifest, new_states)
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
            self._write_paired_states(store, manifest, new_states)
            return {
                "provider": AWS_P5_PROFILE,
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
        by_run = {run.run_id: run for run in manifest.runs}
        if (
            getattr(receipt, "schema_version", None) != 2
            or len(checkpoints) != 2
            or {checkpoint.run_id for checkpoint in checkpoints} != set(by_run)
        ):
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                "AWS resume requires one complete v2 paired receipt",
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
    ) -> dict[str, object]:
        self._validate_manifest(manifest)
        self._validate_release(release, manifest)
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
        )
        if not apply:
            return {
                "provider": AWS_P5_PROFILE,
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
            self._repair_paired_states(store, manifest)
            states = [store.read_run(run.run_id) for run in manifest.runs]
            if any(state is None for state in states):
                raise MsctlError(
                    "RUN_STATE_MISSING",
                    "AWS resume requires complete paired submit state",
                )
            present = [state for state in states if state is not None]
            if not all(
                self._same_state_binding(state, manifest) for state in present
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
            )
            resume_resources = self._resume_resources(
                release=release,
                manifest=manifest,
                instance_id=instance_id,
                terminate_at=terminate_at,
                checkpoint_receipt_sha256=checkpoint_receipt.sha256,
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
                        if started or terminal:
                            return {
                                "provider": AWS_P5_PROFILE,
                                "seed": manifest.seed,
                                "instance_id": instance_id,
                                "operation_id": operation_intent[
                                    "operation_id"
                                ],
                                "status": (
                                    "REMOTE_TERMINAL"
                                    if terminal
                                    else "REMOTE_STARTED"
                                ),
                                "submitted": 0,
                                "idempotent": True,
                            }
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
                        "provider": AWS_P5_PROFILE,
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
                    "provider": AWS_P5_PROFILE,
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
            self._write_paired_states(store, manifest, present)
            now = _timestamp()
            for state in present:
                state["status"] = "SENDING"
                state["send_attempted"] = True
                state["updated_at"] = now
            self._write_paired_states(store, manifest, present)
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
            self._write_paired_states(store, manifest, present)
            return {
                "provider": AWS_P5_PROFILE,
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
    ) -> list[dict[str, object]]:
        states = [store.read_run(run.run_id) for run in manifest.runs]
        if any(state is None for state in states):
            raise MsctlError(
                "RUN_STATE_MISSING",
                "AWS lifecycle state is missing one paired arm",
            )
        present = [state for state in states if state is not None]
        if not all(
            self._same_state_binding(state, manifest) for state in present
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
    ) -> dict[str, object]:
        self._validate_manifest(manifest)
        self._validate_release(release, manifest)
        if not apply:
            return {
                "provider": AWS_P5_PROFILE,
                "seed": manifest.seed,
                "operation": "cancel",
                "run_ids": [run.run_id for run in manifest.runs],
                "cancelled": 0,
            }
        self._verify_approval(
            approval_path,
            operation="cancel",
            release=release,
            manifest=manifest,
        )
        store = StateStore(self.state_root)
        with store.locked():
            states = self._paired_states(store, manifest)
            if {state.get("status") for state in states} == {"Cancelling"}:
                return {
                    "provider": AWS_P5_PROFILE,
                    "seed": manifest.seed,
                    "command_id": states[0]["command_id"],
                    "status": "Cancelling",
                    "cancelled": 0,
                    "idempotent": True,
                }
            command_ids = {state.get("command_id") for state in states}
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
                "provider": AWS_P5_PROFILE,
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
    ) -> list[list[str]]:
        release_root = self._release_root(release)
        run_root = f"/mnt/memorysplit/runs/seed-{manifest.seed}"
        evaluation_root = "/mnt/memorysplit/evaluations"
        return [
            [
                "/usr/bin/docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--gpus",
                "all",
                "--user",
                (
                    f"{getattr(self.runtime, 'uid', 1000)}:"
                    f"{getattr(self.runtime, 'gid', 1000)}"
                ),
                "--mount",
                f"type=bind,src={release_root},dst={release_root},readonly",
                "--mount",
                f"type=bind,src={run_root},dst={run_root},readonly",
                "--mount",
                (
                    f"type=bind,src={evaluation_root},"
                    f"dst={evaluation_root}"
                ),
                "--workdir",
                release_root,
                self.runtime.container_image,
                "/usr/bin/python3",
                "-m",
                "evals.confirmatory",
                "evaluate",
                "--run",
                f"{run_root}/{run.arm}/run",
                "--sealed-release",
                release_root,
                "--expected-study-lock-sha256",
                manifest.study_lock_sha256,
                "--device",
                "cuda",
                "--output-dir",
                f"/mnt/memorysplit/evaluations/{run.run_id}",
            ]
            for run in sorted(manifest.runs, key=lambda row: row.arm)
        ]

    def _evaluation_operation_intent(
        self,
        release: object,
        manifest: object,
        evidence: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        remote_argv = self._evaluation_argv(release, manifest)
        lifecycle_evidence = dict(
            evidence
            or {
                "dataset_pointer_sha256": manifest.dataset_sha256,
                "dataset_verification_sha256": manifest.dataset_sha256,
                "environment_receipt_sha256": self._runtime_sha256(),
            }
        )
        for field, digest in lifecycle_evidence.items():
            require_sha256(digest, label=field)
        return {
            "schema_version": 1,
            "operation": "evaluate",
            "provider": AWS_P5_PROFILE,
            "seed": manifest.seed,
            "release_sha256": release.archive_sha256,
            "run_manifest_sha256": manifest.sha256,
            "dataset_sha256": manifest.dataset_sha256,
            **lifecycle_evidence,
            "runtime_sha256": self._runtime_sha256(),
            "environment": {
                "AWS_REGION": self.runtime.region,
                "MS_AWS_AMI_ID": self.runtime.ami_id,
                "MS_CONTAINER_DIGEST": self.runtime.container_digest,
                "MS_CONTAINER_IMAGE": self.runtime.container_image,
                "MS_RUNTIME_GID": str(getattr(self.runtime, "gid", 1000)),
                "MS_RUNTIME_UID": str(getattr(self.runtime, "uid", 1000)),
                "MS_S3_ROOT": self.runtime.s3_root,
            },
            "checkpoint_receipt": None,
            "steps": [
                {"name": f"evaluate-{run.arm}", "argv": argv}
                for run, argv in zip(
                    sorted(manifest.runs, key=lambda row: row.arm),
                    remote_argv,
                    strict=True,
                )
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
    ) -> dict[str, object]:
        self._validate_manifest(manifest)
        self._validate_release(release, manifest)
        operation_intent = self._evaluation_operation_intent(
            release,
            manifest,
            evidence,
        )
        if not apply:
            return {
                "provider": AWS_P5_PROFILE,
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
            states = self._paired_states(store, manifest)
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
            self._validate_selected_instance_binding(
                manifest,
                instance_id=instance_id,
                terminate_at=terminate_at,
                operation="evaluation",
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
                    existing.get("provider") != AWS_P5_PROFILE
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
                        if started or terminal:
                            return {
                                "provider": AWS_P5_PROFILE,
                                "seed": manifest.seed,
                                "instance_id": instance_id,
                                "operation_id": operation_intent[
                                    "operation_id"
                                ],
                                "status": (
                                    "REMOTE_TERMINAL"
                                    if terminal
                                    else "REMOTE_STARTED"
                                ),
                                "submitted": 0,
                                "idempotent": True,
                            }
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
                        "provider": AWS_P5_PROFILE,
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
                    "provider": AWS_P5_PROFILE,
                    "seed": manifest.seed,
                    "instance_id": instance_id,
                    "command_id": command_id,
                    "status": status,
                    "submitted": 0,
                    "idempotent": True,
                }
            self._require_ssm_online(instance_id)
            self._ensure_argv_document()
            published = self._publish_operation_intent(operation_intent)
            now = _timestamp()
            store.write_evaluation(
                manifest.sha256,
                {
                    "schema_version": 1,
                    "provider": AWS_P5_PROFILE,
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
                },
            )
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
                "provider": AWS_P5_PROFILE,
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
    ) -> dict[str, object]:
        self._validate_manifest(manifest)
        self._validate_release(release, manifest)
        if not apply:
            return {
                "provider": AWS_P5_PROFILE,
                "seed": manifest.seed,
                "release_sha256": release.archive_sha256,
                "run_manifest_sha256": manifest.sha256,
                "operation": "cleanup",
                "terminated": 0,
            }
        if approval_path is None:
            raise MsctlError(
                "APPROVAL_REQUIRED",
                "cleanup apply requires an explicit signed approval receipt",
            )
        store = StateStore(self.state_root)
        with store.locked():
            states = self._paired_states(store, manifest)
            instance_ids = {state.get("instance_id") for state in states}
            if None in instance_ids or len(instance_ids) != 1:
                raise MsctlError(
                    "SUBMISSION_UNCERTAIN",
                    "AWS cleanup requires one paired instance",
                )
            instance_id = str(next(iter(instance_ids)))
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
                ),
            )
            if {state.get("status") for state in states} == {"Terminating"}:
                return {
                    "provider": AWS_P5_PROFILE,
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
                "provider": AWS_P5_PROFILE,
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
        if command == "env ensure":
            apply = bool(getattr(args, "apply", False))
            root_value = getattr(args, "root", None)
            if getattr(self.profile, "profile_id", None) == _V3_PROFILE_ID:
                runtime_lock = getattr(args, "runtime_lock", None)
                control_bundle = getattr(args, "control_bundle", None)
                instance_id = getattr(args, "instance_id", None)
                receipt = getattr(args, "receipt", None)
                if (
                    root_value is None
                    or runtime_lock is None
                    or control_bundle is None
                    or instance_id is None
                    or receipt is None
                ):
                    raise MsctlError(
                        "CLI_USAGE",
                        "AWS v3 env ensure requires explicit runtime lock, "
                        "control bundle, instance ID, receipt, and root",
                    )
                return not apply, self.env_ensure(
                    root=root_value,
                    runtime_lock=runtime_lock,
                    control_bundle=control_bundle,
                    instance_id=instance_id,
                    receipt=receipt,
                    apply=apply,
                )
            lock_value = getattr(args, "lock", None)
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
                    expected_instance_id=(
                        str(getattr(args, "instance_id"))
                        if command == "submit"
                        else None
                    ),
                )
            if command == "runs render":
                return True, self.render(
                    release=release,
                    manifest=manifest,
                    evidence=evidence,
                )
            if command == "submit":
                return not args.apply, self.submit(
                    release=release,
                    manifest=manifest,
                    instance_id=args.instance_id,
                    terminate_at=args.terminate_at,
                    approval_path=args.approval,
                    apply=args.apply,
                    evidence=evidence,
                )
            if command == "status":
                return False, self.status(
                    release=release,
                    manifest=manifest,
                    cached=args.cached,
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
                )
            if command == "cancel":
                return not args.apply, self.cancel(
                    release=release,
                    manifest=manifest,
                    approval_path=args.approval,
                    apply=args.apply,
                )
            if command == "evaluate":
                return not args.apply, self.evaluate(
                    release=release,
                    manifest=manifest,
                    approval_path=args.approval,
                    apply=args.apply,
                    evidence=evidence,
                )
            apply = bool(getattr(args, "apply", False))
            return not apply, self.cleanup(
                release=release,
                manifest=manifest,
                approval_path=getattr(args, "approval", None),
                apply=apply,
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
    module_name = "cluster.aws.p5.profile"
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as error:
        raise MsctlError(
            "PROVIDER_ADAPTER_UNAVAILABLE",
            "the AWS P5 runtime adapter is not installed",
            details={"adapter": module_name},
        ) from error
    validator = getattr(module, "validate_runtime_environment", None)
    if not callable(validator):
        raise MsctlError(
            "PROVIDER_ADAPTER_UNAVAILABLE",
            "the AWS P5 runtime adapter has no validator",
            details={"adapter": module_name},
        )
    remote_environment = {
        name: value
        for name, value in environment.items()
        if name not in _LOCAL_AWS_CONFIG
    }
    try:
        runtime = validator(profile, remote_environment)
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
    return AwsP5Backend(
        profile=profile,
        runtime=runtime,
        instance_profile_arn=instance_profile_arn,
        state_root=state_root,
        runner=runner,
        approval_verifier=approval_verifier,
        environ=environment,
    )
