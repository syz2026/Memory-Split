"""Strict, dry-run-first AWS P5 lifecycle backend."""

from __future__ import annotations

import importlib
import base64
import hashlib
import json
import os
import re
import secrets
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace as dataclass_replace
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
from cluster.aws.p5.canary import (
    CanaryError,
    load_canary_plan,
    parse_qualification_receipt_bytes,
    qualification_roundtrip_blob,
)
from cluster.aws.p5.checkpoint_mirror import _canonical_json
from cluster.aws.p5.run_finalization import (
    _RECEIPT_BINDING_FIELDS,
    FinalizationError,
    parse_run_finalization_receipt_bytes,
)
from cluster.aws.gpu_profile import (
    validate_runtime_environment as validate_selected_runtime_environment,
)

from .approval import verify_scope_approval
from .aws_collect import (
    COLLECTION_RECEIPT_FIELDS,
    COLLECTION_RECEIPT_TYPE,
    SEED_COLLECTION_OBJECT_COUNT,
    CollectionError,
    CollectionReceiptRef,
    SubprocessAwsDownloadRunner,
    admit_prior_seed_collection,
    download_timeout_seconds,
    parse_seed_collection_receipt_bytes,
)
from .aws_cohort_collect import (
    COHORT_COLLECTION_OBJECT_COUNT,
    COHORT_COLLECTION_RECEIPT_TYPE,
    CohortCollectionError,
    _strict_json as _strict_cohort_json,
    parse_cohort_collection_receipt_bytes,
    parse_cohort_evidence_index,
)
from .aws_contracts import (
    EVALUATION_OUTPUT_MEMBERS,
    bootstrap_receipt_key,
    cohort_collection_receipt_key,
    cohort_report_object_key,
    collection_receipt_key,
    run_receipt_key,
)
from .aws_lifecycle import (
    AuthenticatedProviderLifecycle,
    LIFECYCLE_BINDING_FIELDS,
    ProviderLifecycleBinding,
    admit_provider_lifecycle,
)
from .aws_argv import (
    ARGV_DOCUMENT_CONTENT,
    ARGV_DOCUMENT_NAME,
    ARGV_DOCUMENT_SHA256,
)
from .aws_seed_transition import (
    PriorRunReceiptRef,
    SeedTransitionError,
    admit_prior_seed_finalization,
)
from .contracts import (
    bind_release,
    load_release,
    load_run_manifest,
    parse_paired_checkpoint_receipt_v3,
    validate_aws_dataset_pointer_contract,
    validate_runtime_attested_contract,
    verify_aws_checkpoint_receipt_v3,
    verify_checkpoint_receipt,
    verify_release_member,
)
from .errors import MsctlError
from .fsutil import (
    atomic_write_at,
    hash_fd,
    open_directory,
    open_directory_at,
    open_parent_at,
    open_regular_at,
    read_fd,
    remove_tree_at,
    rename_noreplace_at,
)
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
_TERMINAL_COMMAND_STATES = {"Success", "Failed", "Cancelled", "TimedOut"}
_BOOT_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
# Cohort-wide identity-only EC2 tags for selected manifests: never seed,
# run-manifest, or terminate-at, which live in signed approval, immutable
# operation intents, and atomic paired state instead.
_SELECTED_IDENTITY_TAG_NAMES = {
    "provider": "MemorySplitProvider",
    "cohort_sha256": "MemorySplitCohortSHA256",
    "release_sha256": "MemorySplitReleaseSHA256",
    "dataset_sha256": "MemorySplitDatasetSHA256",
    "profile_sha256": "MemorySplitProfileSHA256",
    "runtime_sha256": "MemorySplitRuntimeSHA256",
    "container_digest": "MemorySplitContainerDigest",
    "selection_sha256": "MemorySplitSelectionSHA256",
    "selection_version_id": "MemorySplitSelectionVersionId",
}
_SELECTED_IDENTITY_INSTANCE_FIELDS = {
    "instance_id",
    "instance_type",
    "state",
    "instance_profile_arn",
    "ami_id",
    *_SELECTED_IDENTITY_TAG_NAMES,
}
_LEASE_RESET_ARGV = (
    "/usr/bin/systemctl",
    "stop",
    "memorysplit-auto-terminate*.timer",
    "memorysplit-auto-terminate*.service",
)
# Frozen copy of cluster.aws.p5.bootstrap.BOOTSTRAP_RECEIPT_FIELDS: the
# controller must not import the on-instance bootstrap module, so the field
# closure is mirrored here and pinned by a cross-check test.
_BOOTSTRAP_RECEIPT_FIELDS = (
    "account_id",
    "ami_id",
    "boot_id",
    "code_commit",
    "cohort_assignment_sha256",
    "container_image",
    "container_digest",
    "corpus_build_id",
    "corpus_ordered_stream_sha256",
    "corpus_receipt_sha256",
    "durable_upload_verified",
    "instance_id",
    "instance_store",
    "instance_type",
    "profile_sha256",
    "provider",
    "receipt_type",
    "region",
    "release_members_sha256",
    "release_root",
    "release_sha256",
    "role_arn",
    "role_name",
    "runtime_gid",
    "runtime_uid",
    "schema_version",
    "scratch_root",
)
_SAFE_PATH = "/usr/local/bin:/usr/bin:/bin"
_MAX_PAID_RUNTIME = timedelta(minutes=1440)
_AWS_PRIVATE_HOME = "/var/lib/memorysplit/aws-private-home"
_V3_PROFILE_ID = "aws-p5.48xlarge-v3"
_CANARY_REMOTE_TIMEOUT_SECONDS = 172_800.0
_CANARY_POLL_SECONDS = 5.0
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
        download_runner: object | None = None,
        approval_verifier: Callable[..., object] = verify_scope_approval,
        corpus_verifier: Callable[..., object] | None = None,
        identity_verifier: Callable[
            [Mapping[str, object], str, str],
            bool,
        ] = _verify_instance_identity_pkcs7,
        environ: Mapping[str, str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        _authenticated_lifecycle: AuthenticatedProviderLifecycle | None = None,
        _lifecycle_authority_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        if _authenticated_lifecycle is None:
            _validate_profile(profile)
        elif (
            not isinstance(
                _authenticated_lifecycle,
                AuthenticatedProviderLifecycle,
            )
            or _authenticated_lifecycle.profile != profile
        ):
            raise MsctlError(
                "PROFILE_INVALID",
                "authenticated provider lifecycle differs from profile",
            )
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
        self.lifecycle_binding = (
            _authenticated_lifecycle.binding
            if _authenticated_lifecycle is not None
            else None
        )
        self._lifecycle_authority_kwargs = (
            dict(_lifecycle_authority_kwargs)
            if _lifecycle_authority_kwargs is not None
            else None
        )
        self.instance_profile_arn = instance_profile_arn
        self.state_root = Path(state_root)
        self.runner = runner or SubprocessAwsJsonRunner()
        self.download_runner = download_runner or (
            SubprocessAwsDownloadRunner()
        )
        self.approval_verifier = approval_verifier
        self.corpus_verifier = corpus_verifier
        self.identity_verifier = identity_verifier
        self.environ = dict(os.environ if environ is None else environ)
        if not callable(sleep) or not callable(monotonic):
            raise MsctlError(
                "AWS_RUNTIME_INVALID",
                "AWS controller time boundaries must be callable",
            )
        self.sleep = sleep
        self.monotonic = monotonic
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

    @classmethod
    def from_authenticated_selection(
        cls,
        *,
        authority_root: Path | str,
        repo_root: Path | str,
        runtime_lock_path: Path | str,
        runtime_evidence_path: Path | str,
        runtime_sbom_path: Path | str,
        objective_controls_amendment_path: Path | str,
        selection_store: object,
        account_id: str,
        instance_id: str,
        boot_id: str,
        seed: int,
        expected_selection_version_id: str,
        selection_identity_verifier: object,
        qualification_approval_verifier: object,
        trusted_qualification_public_key_sha256: str,
        runtime_environment: Mapping[str, str],
        instance_profile_arn: str,
        state_root: Path | str,
        runner: AwsJsonRunner | None = None,
        download_runner: object | None = None,
        scope_approval_verifier: Callable[..., object] = verify_scope_approval,
        corpus_verifier: Callable[..., object] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> "AwsP5Backend":
        """Construct a selected-provider controller from fixed authority."""

        authority_kwargs = {
            "authority_root": authority_root,
            "repo_root": repo_root,
            "runtime_lock_path": runtime_lock_path,
            "runtime_evidence_path": runtime_evidence_path,
            "runtime_sbom_path": runtime_sbom_path,
            "objective_controls_amendment_path": (
                objective_controls_amendment_path
            ),
            "store": selection_store,
            "account_id": account_id,
            "instance_id": instance_id,
            "boot_id": boot_id,
            "expected_selection_version_id": (
                expected_selection_version_id
            ),
            "identity_verifier": selection_identity_verifier,
            "approval_verifier": qualification_approval_verifier,
            "trusted_public_key_sha256": (
                trusted_qualification_public_key_sha256
            ),
        }
        lifecycle = admit_provider_lifecycle(
            **authority_kwargs,
            seed=seed,
        )
        try:
            runtime = validate_selected_runtime_environment(
                lifecycle.profile,
                runtime_environment,
            )
        except ValueError as error:
            raise MsctlError(
                "AWS_RUNTIME_INVALID",
                "selected provider runtime environment is invalid",
            ) from error
        return cls(
            profile=lifecycle.profile,
            runtime=runtime,
            instance_profile_arn=instance_profile_arn,
            state_root=state_root,
            runner=runner,
            download_runner=download_runner,
            approval_verifier=scope_approval_verifier,
            corpus_verifier=corpus_verifier,
            identity_verifier=selection_identity_verifier,
            environ=runtime_environment,
            sleep=sleep,
            monotonic=monotonic,
            _authenticated_lifecycle=lifecycle,
            _lifecycle_authority_kwargs={
                **authority_kwargs,
                "seed": seed,
            },
        )

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
            "provider": self.profile.provider,
            "region": self.runtime.region,
            **output,
        }

    def _require_provider_lifecycle(self, manifest: object) -> None:
        selected = getattr(
            manifest,
            "provider_selection_sha256",
            None,
        ) is not None
        if not selected:
            if self.lifecycle_binding is not None:
                raise MsctlError(
                    "RUN_MANIFEST_INVALID",
                    "authenticated controller requires a selected manifest",
                )
            return
        if self._lifecycle_authority_kwargs is None:
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "selected manifest requires fixed provider authority paths",
            )
        lifecycle = admit_provider_lifecycle(
            **{
                **self._lifecycle_authority_kwargs,
                "seed": manifest.seed,
            }
        )
        expected = lifecycle.binding.to_dict()
        if (
            lifecycle.profile != self.profile
            or self.lifecycle_binding != lifecycle.binding
            or any(
                (
                    tuple(getattr(manifest, field, ()))
                    if field == "arms"
                    and isinstance(
                        getattr(manifest, field, None),
                        (list, tuple),
                    )
                    else getattr(manifest, field, None)
                )
                != (
                    tuple(value)
                    if field == "arms"
                    and isinstance(value, (list, tuple))
                    else value
                )
                for field, value in expected.items()
            )
        ):
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "selected manifest differs from authenticated provider authority",
            )

    @staticmethod
    def _selected_manifest_lifecycle(manifest: object) -> bool:
        return (
            getattr(manifest, "schema_version", None) == 3
            and getattr(
                manifest,
                "provider_selection_sha256",
                None,
            )
            is not None
        )

    def _validate_manifest(self, manifest: object) -> None:
        self._require_provider_lifecycle(manifest)
        runs = getattr(manifest, "runs", ())
        is_v3_profile = (
            getattr(self.profile, "profile_id", None)
            in {_V3_PROFILE_ID, "aws-p6-b300.48xlarge-v3"}
        )
        expected_schema = 3 if is_v3_profile else 2
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
            getattr(manifest, "schema_version", None) != expected_schema
            or not isinstance(getattr(manifest, "source_commit", None), str)
            or _COMMIT_RE.fullmatch(manifest.source_commit) is None
        ):
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "AWS lifecycle requires its exact source-bound run manifest",
            )
        hash_fields = (
            (
                "release_sha256",
                "release_receipt_sha256",
                "profile_sha256",
                "dataset_pointer_sha256",
                "dataset_receipt_sha256",
                "dataset_build_id",
                "ordered_stream_sha256",
                "cohort_assignment_sha256",
                "preregistration_sha256",
                "sealed_evaluation_release_sha256",
                "sha256",
            )
            if is_v3_profile
            else (
                "release_sha256",
                "dataset_sha256",
                "cohort_assignment_sha256",
                "study_lock_sha256",
                "sha256",
            )
        )
        if is_v3_profile and getattr(
            manifest,
            "provider_selection_sha256",
            None,
        ) is not None:
            hash_fields = (
                *hash_fields,
                "hardware_amendment_sha256",
                "provider_selection_sha256",
                "runtime_lock_sha256",
                "runtime_sbom_sha256",
                "qualification_evidence_sha256",
                "qualification_environment_receipt_sha256",
                "qualification_canary_receipt_sha256",
                "qualification_approval_receipt_sha256",
                "qualification_approval_public_key_sha256",
                "objective_controls_contract_sha256",
            )
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
        if is_v3_profile and (
            getattr(manifest, "profile_sha256", None)
            != getattr(self.profile, "sha256", None)
            or getattr(manifest, "cohort_id", None)
            != "memorysplit-confirmatory-v3-360m-n10-aws"
            or not isinstance(getattr(manifest, "source_tree", None), str)
            or _COMMIT_RE.fullmatch(manifest.source_tree) is None
            or getattr(manifest, "profile_id", None)
            != getattr(self.profile, "profile_id", None)
            or (
                getattr(
                    manifest,
                    "provider_selection_sha256",
                    None,
                )
                is not None
                and (
                    not isinstance(
                        getattr(
                            manifest,
                            "provider_selection_version_id",
                            None,
                        ),
                        str,
                    )
                    or getattr(
                        manifest,
                        "provider_selection_version_id",
                        None,
                    )
                    in {"", "null"}
                )
            )
        ):
            raise MsctlError(
                "RUN_MANIFEST_INVALID",
                "AWS v3 manifest identity does not match the frozen profile",
            )
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
        is_v3 = getattr(manifest, "schema_version", None) == 3
        if (
            getattr(release, "provider", None) != self.profile.provider
            or getattr(release, "archive_sha256", None)
            != manifest.release_sha256
            or getattr(release, "source_commit", None)
            != manifest.source_commit
            or (
                is_v3
                and (
                    getattr(release, "receipt_sha256", None)
                    != manifest.release_receipt_sha256
                    or getattr(release, "source_tree", None)
                    != manifest.source_tree
                )
            )
        ):
            raise MsctlError(
                "RELEASE_RUN_MISMATCH",
                "AWS release and run manifest provenance differ",
            )

    def _release_root(self, release: object) -> str:
        return f"/mnt/memorysplit/releases/{release.archive_sha256}"

    def _launcher_manifest_step(
        self,
        *,
        release: object,
        manifest: object,
        lifecycle_evidence: Mapping[str, object],
        is_v3: bool,
        staging: str,
    ) -> dict[str, object]:
        return {
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
                *(
                    [
                        "--release-receipt-sha256",
                        release.receipt_sha256,
                        "--environment-receipt-sha256",
                        lifecycle_evidence[
                            "environment_receipt_sha256"
                        ],
                        "--run-manifest-sha256",
                        manifest.sha256,
                        "--source-tree",
                        manifest.source_tree,
                    ]
                    if is_v3
                    else []
                ),
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

    def _training_operation_intent(
        self,
        *,
        operation: str,
        release: object,
        manifest: object,
        terminate_at: str | None = None,
        checkpoints: Mapping[str, object] | None = None,
        checkpoint_receipt_sha256: str | None = None,
        checkpoint_receipt_uri: str | None = None,
        checkpoint_receipt_version_id: str | None = None,
        checkpoint_receipt_bytes: int | None = None,
        evidence: Mapping[str, str] | None = None,
        attempt: int | None = None,
        bootstrap_mode: str | None = None,
        bootstrap_receipt_sha256: str | None = None,
    ) -> dict[str, object]:
        is_v3 = getattr(manifest, "schema_version", None) == 3
        dataset_receipt_sha256 = (
            manifest.dataset_receipt_sha256
            if is_v3
            else manifest.dataset_sha256
        )
        lifecycle_evidence = dict(
            evidence
            or {
                "dataset_pointer_sha256": (
                    manifest.dataset_pointer_sha256
                    if is_v3
                    else manifest.dataset_sha256
                ),
                "dataset_verification_sha256": dataset_receipt_sha256,
                "environment_receipt_sha256": self._runtime_sha256(),
            }
        )
        expected_evidence_fields = {
            "dataset_pointer_sha256",
            "dataset_verification_sha256",
            "environment_receipt_sha256",
            *({"instance_id", "boot_id"} if is_v3 else set()),
        }
        if set(lifecycle_evidence) != expected_evidence_fields:
            raise MsctlError(
                "LIFECYCLE_EVIDENCE_INVALID",
                "AWS lifecycle evidence fields do not match the contract",
            )
        for field in (
            "dataset_pointer_sha256",
            "dataset_verification_sha256",
            "environment_receipt_sha256",
        ):
            require_sha256(lifecycle_evidence[field], label=field)
        if is_v3 and (
            not isinstance(lifecycle_evidence["instance_id"], str)
            or _INSTANCE_ID_RE.fullmatch(
                lifecycle_evidence["instance_id"]
            )
            is None
            or not isinstance(lifecycle_evidence["boot_id"], str)
            or not lifecycle_evidence["boot_id"]
        ):
            raise MsctlError(
                "LIFECYCLE_EVIDENCE_INVALID",
                "AWS v3 lifecycle instance and boot identities are invalid",
            )
        release_root = self._release_root(release)
        staging = "/mnt/memorysplit/staging"
        selected_lifecycle = (
            is_v3
            and getattr(
                manifest,
                "provider_selection_sha256",
                None,
            )
            is not None
        )
        launch_profile_name = (
            f"{self.profile.profile_id}.json"
            if selected_lifecycle
            else ("aws-p5.48xlarge-v3.json" if is_v3 else "aws-p5.48xlarge.json")
        )
        steps: list[dict[str, object]] = []
        if selected_lifecycle:
            if operation not in {"submit", "resume"}:
                raise MsctlError(
                    "OPERATION_UNSUPPORTED",
                    "selected intents exist only for submit and resume",
                )
            if (
                not isinstance(terminate_at, str)
                or type(attempt) is not int
                or attempt < 1
                or bootstrap_mode not in {"bootstrap", "reuse"}
                or (bootstrap_mode == "reuse")
                != (bootstrap_receipt_sha256 is not None)
            ):
                raise MsctlError(
                    "BOOTSTRAP_REUSE_INVALID",
                    "selected intents require one exact lease and bootstrap binding",
                )
            if bootstrap_receipt_sha256 is not None:
                require_sha256(
                    bootstrap_receipt_sha256,
                    label="bootstrap reuse receipt",
                )
            lease_unit = "memorysplit-auto-terminate-" + canonical_sha256(
                {
                    "attempt": attempt,
                    "operation": operation,
                    "run_manifest_sha256": manifest.sha256,
                    "seed": manifest.seed,
                    "terminate_at": terminate_at,
                }
            )
            steps.append(
                {
                    "name": "reset-termination-leases",
                    "argv": list(_LEASE_RESET_ARGV),
                }
            )
            steps.append(
                {
                    "name": "auto-termination",
                    "argv": [
                        "/usr/bin/systemd-run",
                        "--unit",
                        lease_unit,
                        "--on-calendar",
                        terminate_at,
                        "/sbin/shutdown",
                        "-h",
                        "now",
                    ],
                }
            )
            if bootstrap_mode == "bootstrap":
                steps.append(
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
                    }
                )
                # Full bootstrap syncs the release and dataset from S3 by
                # itself after mounting the fresh RAID scratch root, so the
                # legacy controller pre-staging steps (which a later mount
                # would shadow) are intentionally absent.
                steps.append(
                    {
                        "name": "bootstrap",
                        "argv": [
                            "/usr/bin/python3",
                            "/opt/memorysplit/cluster/aws/p5/bootstrap.py",
                            "--profile",
                            (
                                "/opt/memorysplit/cluster/profiles/"
                                + launch_profile_name
                            ),
                            "--container-image",
                            self.runtime.container_image,
                            "--release-archive",
                            (
                                f"{staging}/releases/"
                                f"{release.archive_sha256}/release.zip"
                            ),
                            "--release-sha256",
                            release.archive_sha256,
                            "--release-receipt",
                            (
                                f"{staging}/releases/"
                                f"{release.archive_sha256}/RELEASE.json"
                            ),
                            "--release-receipt-sha256",
                            release.receipt_sha256,
                            "--dataset-receipt",
                            "/mnt/memorysplit/dataset/receipt.json",
                            "--dataset-receipt-sha256",
                            dataset_receipt_sha256,
                            "--cohort-assignment",
                            (
                                f"{staging}/releases/"
                                f"{release.archive_sha256}/"
                                "cohort-assignment-v3.json"
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
                    }
                )
            else:
                steps.append(
                    {
                        "name": "verify-bootstrap-reuse",
                        "argv": [
                            "/usr/bin/python3",
                            "/opt/memorysplit/cluster/aws/p5/bootstrap.py",
                            "--profile",
                            (
                                "/opt/memorysplit/cluster/profiles/"
                                + launch_profile_name
                            ),
                            "--verify-reuse",
                            "--receipt",
                            f"{staging}/bootstrap-receipt.json",
                            "--expected-receipt-sha256",
                            str(bootstrap_receipt_sha256),
                        ],
                    }
                )
            if bootstrap_mode == "bootstrap" or operation == "submit":
                steps.append(
                    self._launcher_manifest_step(
                        release=release,
                        manifest=manifest,
                        lifecycle_evidence=lifecycle_evidence,
                        is_v3=is_v3,
                        staging=staging,
                    )
                )
        elif operation == "submit":
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
        if not selected_lifecycle and operation in {"render", "submit"}:
            release_bucket, release_prefix = self._s3_location(
                f"releases/{release.archive_sha256}/release.zip"
            )
            receipt_bucket, receipt_key = self._s3_location(
                f"releases/{release.archive_sha256}/RELEASE.json"
            )
            cohort_bucket, cohort_key = self._s3_location(
                f"releases/{release.archive_sha256}/"
                f"cohort-assignment-{'v3' if is_v3 else 'v2'}.json"
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
                                f"cohort-assignment-{'v3' if is_v3 else 'v2'}.json"
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
                        (
                            "/opt/memorysplit/cluster/profiles/"
                            + (
                                "aws-p5.48xlarge-v3.json"
                                if is_v3
                                else "aws-p5.48xlarge.json"
                            )
                        ),
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
                        dataset_receipt_sha256,
                        "--cohort-assignment",
                        (
                            f"{staging}/releases/{release.archive_sha256}/"
                            f"cohort-assignment-{'v3' if is_v3 else 'v2'}.json"
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
                self._launcher_manifest_step(
                    release=release,
                    manifest=manifest,
                    lifecycle_evidence=lifecycle_evidence,
                    is_v3=is_v3,
                    staging=staging,
                )
            )
        checkpoint_binding = None
        if checkpoints is not None:
            receipt_sha256 = require_sha256(
                checkpoint_receipt_sha256,
                label="checkpoint receipt",
            )
            resume_root = f"{staging}/resume/{receipt_sha256}"
            if is_v3:
                if (
                    not isinstance(checkpoint_receipt_uri, str)
                    or not checkpoint_receipt_uri.startswith(
                        self.runtime.s3_root.rstrip("/") + "/"
                    )
                    or not isinstance(
                        checkpoint_receipt_version_id,
                        str,
                    )
                    or checkpoint_receipt_version_id in {"", "null"}
                    or type(checkpoint_receipt_bytes) is not int
                    or checkpoint_receipt_bytes <= 0
                ):
                    raise MsctlError(
                        "CHECKPOINT_PROVENANCE_MISMATCH",
                        "v3 resume receipt identity is incomplete",
                    )
                relative_receipt = checkpoint_receipt_uri.removeprefix(
                    self.runtime.s3_root.rstrip("/") + "/"
                )
                receipt_bucket, receipt_key = self._s3_location(
                    relative_receipt
                )
                checkpoint_rows = []
                for arm in ("dense", "split90"):
                    checkpoint = checkpoints[arm]
                    checkpoint_rows.append(
                        {
                            "arm": arm,
                            "bytes": checkpoint.object.bytes,
                            "config_fingerprint": (
                                checkpoint.config_fingerprint
                            ),
                            "config_sha256": checkpoint.config_sha256,
                            "resume_path": f"{resume_root}/{arm}.pt",
                            "resume_sha256": checkpoint.object.sha256,
                            "step": checkpoint.step,
                            "uri": checkpoint.object.uri,
                            "version_id": checkpoint.object.version_id,
                            "world_size": checkpoint.world_size,
                        }
                    )
                checkpoint_binding = {
                    "bytes": checkpoint_receipt_bytes,
                    "checkpoints": checkpoint_rows,
                    "sha256": receipt_sha256,
                    "uri": checkpoint_receipt_uri,
                    "version_id": checkpoint_receipt_version_id,
                }
            else:
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
                        *(
                            [
                                "--version-id",
                                str(checkpoint_receipt_version_id),
                            ]
                            if is_v3
                            else []
                        ),
                        "--checksum-mode",
                        "ENABLED",
                        f"{resume_root}/receipt.json",
                    ],
                }
            )
            for row in checkpoint_rows:
                if is_v3:
                    relative_checkpoint = str(row["uri"]).removeprefix(
                        self.runtime.s3_root.rstrip("/") + "/"
                    )
                    bucket, key = self._s3_location(relative_checkpoint)
                else:
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
                            *(
                                [
                                    "--version-id",
                                    str(row["version_id"]),
                                ]
                                if is_v3
                                else []
                            ),
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
                f"{release_root}/cluster/profiles/{launch_profile_name}",
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
                *(
                    [
                        "--checkpoint-receipt-uri",
                        str(checkpoint_receipt_uri),
                        "--checkpoint-receipt-version-id",
                        str(checkpoint_receipt_version_id),
                    ]
                    if is_v3
                    else []
                ),
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
                f"{release_root}/cluster/profiles/{launch_profile_name}",
                "--repo-root",
                release_root,
                "--scratch-root",
                "/mnt/memorysplit",
                "--apply",
            ]
        steps.append({"name": "paired-launch", "argv": launcher_argv})
        lifecycle_fields = (
            {
                field: (
                    list(manifest.arms)
                    if field == "arms"
                    else getattr(manifest, field)
                )
                for field in LIFECYCLE_BINDING_FIELDS
            }
            if selected_lifecycle
            else {}
        )
        selected_environment = (
            {
                "MS_HARDWARE_AMENDMENT_SHA256": (
                    manifest.hardware_amendment_sha256
                ),
                "MS_OBJECTIVE_CONTROLS_SHA256": (
                    manifest.objective_controls_contract_sha256
                ),
                "MS_PROFILE_ID": manifest.profile_id,
                "MS_PROFILE_SHA256": manifest.profile_sha256,
                "MS_PROVIDER": manifest.provider,
                "MS_PROVIDER_SELECTION_SHA256": (
                    manifest.provider_selection_sha256
                ),
                "MS_PROVIDER_SELECTION_VERSION_ID": (
                    manifest.provider_selection_version_id
                ),
                "MS_QUALIFICATION_EVIDENCE_SHA256": (
                    manifest.qualification_evidence_sha256
                ),
                "MS_RUNTIME_LOCK_SHA256": manifest.runtime_lock_sha256,
                "MS_RUNTIME_SBOM_SHA256": manifest.runtime_sbom_sha256,
            }
            if selected_lifecycle
            else {}
        )
        selected_bindings = (
            {
                "bootstrap": {
                    "mode": bootstrap_mode,
                    "receipt_sha256": bootstrap_receipt_sha256,
                },
                "lease_unit": lease_unit,
            }
            if selected_lifecycle
            else {}
        )
        return {
            **lifecycle_fields,
            **selected_bindings,
            "schema_version": 3 if selected_lifecycle else 2 if is_v3 else 1,
            "operation": operation,
            "provider": getattr(
                manifest,
                "provider",
                self.profile.provider,
            ),
            "seed": manifest.seed,
            "release_sha256": release.archive_sha256,
            "run_manifest_sha256": manifest.sha256,
            "dataset_sha256": dataset_receipt_sha256,
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
                **selected_environment,
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
        dataset_sha256 = getattr(
            manifest,
            "dataset_receipt_sha256",
            getattr(manifest, "dataset_sha256", None),
        )
        return {
            "provider": getattr(
                manifest,
                "provider",
                self.profile.provider,
            ),
            "seed": manifest.seed,
            "cohort_sha256": manifest.cohort_assignment_sha256,
            "release_sha256": manifest.release_sha256,
            "dataset_sha256": dataset_sha256,
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

    def _selected_identity_binding(self, manifest: object) -> dict[str, object]:
        return {
            "provider": manifest.provider,
            "cohort_sha256": manifest.cohort_assignment_sha256,
            "release_sha256": manifest.release_sha256,
            "dataset_sha256": manifest.dataset_receipt_sha256,
            "profile_sha256": self.profile.sha256,
            "runtime_sha256": self._runtime_sha256(),
            "container_digest": self.runtime.container_digest,
            "selection_sha256": manifest.provider_selection_sha256,
            "selection_version_id": (
                manifest.provider_selection_version_id
            ),
        }

    def _selected_identity_tags(self, manifest: object) -> str:
        binding = self._selected_identity_binding(manifest)
        return canonical_json(
            [
                {"Key": name, "Value": str(binding[field])}
                for field, name in _SELECTED_IDENTITY_TAG_NAMES.items()
            ]
        ).decode("ascii")

    @staticmethod
    def _selected_identity_query() -> str:
        tag_selectors = ",".join(
            f"{field}:Tags[?Key=='{name}']|[0].Value"
            for field, name in _SELECTED_IDENTITY_TAG_NAMES.items()
        )
        return (
            "{instances:Reservations[].Instances[]."
            "{instance_id:InstanceId,instance_type:InstanceType,"
            "state:State.Name,instance_profile_arn:IamInstanceProfile.Arn,"
            "ami_id:ImageId," + tag_selectors + "}}"
        )

    def _selected_identity_discover_argv(self, manifest: object) -> list[str]:
        return self._aws_argv(
            "ec2",
            "describe-instances",
            "--filters",
            f"Name=tag:MemorySplitProvider,Values={manifest.provider}",
            (
                "Name=tag:MemorySplitReleaseSHA256,"
                f"Values={manifest.release_sha256}"
            ),
            (
                "Name=tag:MemorySplitSelectionSHA256,"
                f"Values={manifest.provider_selection_sha256}"
            ),
            "Name=instance-state-name,Values=pending,running,stopping",
            query=self._selected_identity_query(),
        )

    def _selected_identity_instance_argv(self, instance_id: str) -> list[str]:
        if _INSTANCE_ID_RE.fullmatch(instance_id) is None:
            raise MsctlError(
                "INSTANCE_BINDING_MISMATCH",
                "EC2 instance ID is invalid",
            )
        return self._aws_argv(
            "ec2",
            "describe-instances",
            "--instance-ids",
            instance_id,
            query=self._selected_identity_query(),
        )

    def _parse_selected_identity_instance(
        self,
        output: object,
        manifest: object,
        *,
        instance_id: str,
        require_bound: bool,
    ) -> dict[str, object]:
        root = _aws_output_object(
            output,
            {"instances"},
            label="selected identity instance output",
        )
        rows = _aws_output_list(
            root["instances"],
            label="selected identity instances",
        )
        if len(rows) != 1:
            raise MsctlError(
                "INSTANCE_BINDING_MISMATCH",
                "selected cohort must resolve to exactly one instance",
            )
        row = _aws_output_object(
            rows[0],
            _SELECTED_IDENTITY_INSTANCE_FIELDS,
            label="selected identity instance",
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
                "selected instance has the wrong immutable runtime",
            )
        expected = self._selected_identity_binding(manifest)
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
                "selected instance has conflicting identity tags",
            )
        return row

    def _discover_selected_identity_instances(
        self,
        manifest: object,
    ) -> list[dict[str, object]]:
        root = _aws_output_object(
            self._run(
                self._selected_identity_discover_argv(manifest),
                operation="selected instance discovery",
            ),
            {"instances"},
            label="selected identity discovery output",
        )
        expected = self._selected_identity_binding(manifest)
        instances: list[dict[str, object]] = []
        for index, raw in enumerate(
            _aws_output_list(
                root["instances"],
                label="selected identity instances",
            )
        ):
            row = _aws_output_object(
                raw,
                _SELECTED_IDENTITY_INSTANCE_FIELDS,
                label=f"selected identity instance[{index}]",
            )
            if (
                not isinstance(row["instance_id"], str)
                or _INSTANCE_ID_RE.fullmatch(row["instance_id"]) is None
                or row["instance_type"] != self.profile.instance_type
                or row["state"] not in _ACTIVE_INSTANCE_STATES
                or row["instance_profile_arn"] != self.instance_profile_arn
                or any(
                    row[field] != value for field, value in expected.items()
                )
            ):
                raise MsctlError(
                    "INSTANCE_BINDING_MISMATCH",
                    "discovered instance does not match the cohort identity",
                    details={"index": index},
                )
            instances.append(row)
        if len(instances) > 1:
            raise MsctlError(
                "DUPLICATE_ACTIVE_SEED",
                "multiple active instances claim one selected cohort",
                details={
                    "instance_ids": sorted(
                        str(row["instance_id"]) for row in instances
                    ),
                },
            )
        return instances

    def _bind_selected_identity_instance(
        self,
        manifest: object,
        *,
        instance_id: str,
    ) -> dict[str, object]:
        selected_argv = self._selected_identity_instance_argv(instance_id)
        self._parse_selected_identity_instance(
            self._run(selected_argv, operation="validate selected instance"),
            manifest,
            instance_id=instance_id,
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
            self._selected_identity_tags(manifest),
            query="{}",
        )
        _aws_output_object(
            self._run(tag, operation="bind selected instance"),
            set(),
            label="EC2 create-tags output",
        )
        row = self._parse_selected_identity_instance(
            self._run(selected_argv, operation="verify selected instance"),
            manifest,
            instance_id=instance_id,
            require_bound=True,
        )
        self._require_terminate_shutdown_behavior(instance_id)
        return row

    def _require_terminate_shutdown_behavior(self, instance_id: str) -> None:
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

    def _validate_selected_identity_instance(
        self,
        manifest: object,
        *,
        instance_id: str,
        operation: str,
    ) -> dict[str, object]:
        row = self._parse_selected_identity_instance(
            self._run(
                self._selected_identity_instance_argv(instance_id),
                operation=f"verify {operation} instance",
            ),
            manifest,
            instance_id=instance_id,
            require_bound=True,
        )
        self._require_terminate_shutdown_behavior(instance_id)
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
                or row["instance_type"] != self.profile.instance_type
                or row["state"] not in _ACTIVE_INSTANCE_STATES
                or row["instance_profile_arn"] != self.instance_profile_arn
                or row["provider"] != self.profile.provider
                or row["seed"] != manifest.seed
                or row["cohort_sha256"]
                != manifest.cohort_assignment_sha256
                or row["release_sha256"] != manifest.release_sha256
                or row["dataset_sha256"]
                != getattr(
                    manifest,
                    "dataset_receipt_sha256",
                    getattr(manifest, "dataset_sha256", None),
                )
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
        if self._selected_manifest_lifecycle(manifest):
            return self._discover_selected_identity_instances(manifest)
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
        is_v3 = getattr(manifest, "schema_version", None) == 3
        selected_lifecycle = (
            is_v3
            and getattr(
                manifest,
                "provider_selection_sha256",
                None,
            )
            is not None
        )
        lifecycle_matches = (
            all(
                state.get(field)
                == (
                    list(manifest.arms)
                    if field == "arms"
                    else getattr(manifest, field)
                )
                for field in LIFECYCLE_BINDING_FIELDS
            )
            if selected_lifecycle
            else True
        )
        provenance_matches = (
            (
                state.get("schema_version")
                == 2
                and state.get("release_receipt_sha256")
                == manifest.release_receipt_sha256
                and state.get("dataset_pointer_sha256")
                == manifest.dataset_pointer_sha256
                and state.get("dataset_receipt_sha256")
                == manifest.dataset_receipt_sha256
                and state.get("dataset_build_id")
                == manifest.dataset_build_id
                and state.get("ordered_stream_sha256")
                == manifest.ordered_stream_sha256
                and state.get("preregistration_sha256")
                == manifest.preregistration_sha256
                and state.get("source_tree") == manifest.source_tree
                and lifecycle_matches
            )
            if is_v3
            else (
                state.get("schema_version") == 1
                and state.get("dataset_sha256") == manifest.dataset_sha256
                and state.get("study_lock_sha256")
                == manifest.study_lock_sha256
            )
        )
        return (
            run is not None
            and state.get("provider") == manifest.provider
            and state.get("seed") == manifest.seed
            and state.get("arm") == run.arm
            and state.get("config_sha256") == run.config_sha256
            and state.get("release_sha256") == manifest.release_sha256
            and provenance_matches
            and state.get("run_manifest_sha256") == manifest.sha256
            and state.get("cohort_assignment_sha256")
            == manifest.cohort_assignment_sha256
            and state.get("source_commit") == manifest.source_commit
            and state.get("profile_sha256")
            == (
                manifest.profile_sha256
                if is_v3
                else self.profile.sha256
            )
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
        operation_intent: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        if operation_intent is None:
            operation_intent = self._training_operation_intent(
                operation="submit",
                release=release,
                manifest=manifest,
                terminate_at=terminate_at,
            )
        return {
            "provider": getattr(
                manifest,
                "provider",
                self.profile.provider,
            ),
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
        checkpoint_receipt: object | None = None,
        checkpoints: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        versioned_bindings: dict[str, object] = {}
        if getattr(manifest, "schema_version", None) == 3:
            if checkpoint_receipt is None or checkpoints is None:
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    "v3 resume approval lacks versioned checkpoint bindings",
                )
            versioned_bindings = {
                "checkpoint_receipt_uri": checkpoint_receipt.uri,
                "checkpoint_receipt_version_id": (
                    checkpoint_receipt.version_id
                ),
                "checkpoint_objects": [
                    {
                        "arm": arm,
                        "sha256": checkpoints[arm].object.sha256,
                        "uri": checkpoints[arm].object.uri,
                        "version_id": checkpoints[arm].object.version_id,
                    }
                    for arm in ("dense", "split90")
                ],
            }
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
                **versioned_bindings,
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
        is_v3 = getattr(manifest, "schema_version", None) == 3
        cohort_member = release.members.get(
            (
                "configs/cohort-assignment-v3.json"
                if is_v3
                else "configs/cohort-assignment-v2.json"
            )
        )
        study_member = release.members.get(
            (
                "configs/preregistration-v3.yaml"
                if is_v3
                else "configs/preregistration-v2.yaml"
            )
        )
        expected_study_sha256 = (
            manifest.preregistration_sha256
            if is_v3
            else manifest.study_lock_sha256
        )
        if (
            cohort_member is None
            or cohort_member.get("sha256")
            != manifest.cohort_assignment_sha256
            or study_member is None
            or study_member.get("sha256") != expected_study_sha256
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
        is_v3 = getattr(manifest, "schema_version", None) == 3
        if is_v3:
            validate_aws_dataset_pointer_contract(pointer)
            if sha256_file(pointer_path) != manifest.dataset_pointer_sha256:
                raise MsctlError(
                    "DATASET_POINTER_INVALID",
                    "AWS v3 dataset pointer does not match the run manifest",
                )
        else:
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
        expected_dataset_sha256 = (
            manifest.dataset_receipt_sha256
            if is_v3
            else manifest.dataset_sha256
        )
        if receipt_sha256 != expected_dataset_sha256:
            raise MsctlError(
                "DATASET_PROVENANCE_MISMATCH",
                "AWS dataset evidence does not match the run manifest",
            )

        environment_path = Path(environment_receipt)
        environment_bytes = environment_path.read_bytes()
        if is_v3:
            try:
                environment = parse_environment_receipt_bytes(
                    environment_bytes
                )
            except AttestationError as error:
                raise MsctlError(
                    "ENVIRONMENT_RECEIPT_INVALID",
                    "AWS v3 environment receipt is invalid",
                ) from error
            if (
                environment["profile_sha256"] != self.profile.sha256
                or environment["source_commit"] != manifest.source_commit
                or environment["source_tree"] != manifest.source_tree
                or environment["container_image_digest"]
                != self.runtime.container_digest
                or environment["ami_id"] != self.runtime.ami_id
                or environment["region"] != self.runtime.region
                or (
                    expected_instance_id is not None
                    and environment["instance_id"] != expected_instance_id
                )
                or not self.identity_verifier(
                    environment["aws_instance_identity_document"],
                    "".join(
                        str(
                            environment[
                                "aws_instance_identity_pkcs7"
                            ]
                        ).split()
                    ),
                    self.runtime.region,
                )
            ):
                raise MsctlError(
                    "ENVIRONMENT_RECEIPT_INVALID",
                    "AWS v3 environment receipt does not bind the selected runtime",
                )
            return {
                "dataset_pointer_sha256": sha256_file(pointer_path),
                "dataset_verification_sha256": source_sha256,
                "environment_receipt_sha256": hashlib.sha256(
                    environment_bytes
                ).hexdigest(),
                "instance_id": str(environment["instance_id"]),
                "boot_id": str(environment["boot_id"]),
            }
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
            states = self._refresh_paired_states(
                store,
                manifest,
                states,
                {"status": status},
            )
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

    def canary_run(
        self,
        *,
        release_root: Path | str,
        release_receipt: Path | str,
        run_manifest: Path | str,
        dataset_receipt: Path | str,
        environment_receipt: Path | str,
        runtime_lock: Path | str,
        instance_id: str,
        boot_id: str,
        output: Path | str,
        apply: bool,
    ) -> dict[str, object]:
        """Validate and render one v3-only qualification operation."""

        if getattr(self.profile, "profile_id", None) != _V3_PROFILE_ID:
            raise MsctlError(
                "CANARY_UNSUPPORTED",
                "P5 qualification is enabled only for the exact v3 profile",
            )
        if type(apply) is not bool:
            raise MsctlError("CLI_USAGE", "canary apply flag must be boolean")
        output_path = Path(os.path.abspath(os.fspath(output)))
        try:
            plan = load_canary_plan(
                release_root=release_root,
                release_receipt_path=release_receipt,
                run_manifest_path=run_manifest,
                dataset_receipt_path=dataset_receipt,
                environment_receipt_path=environment_receipt,
                runtime_lock_path=runtime_lock,
                instance_id=instance_id,
                boot_id=boot_id,
                scratch_root=output_path.parent,
                output_path=output_path,
                s3_root=self.runtime.s3_root,
                dataset_verifier=self.corpus_verifier,
            )
            environment_bytes = read_regular_input(
                environment_receipt,
                label="AWS environment receipt",
                maximum_bytes=1024 * 1024,
            )
            environment = parse_environment_receipt_bytes(environment_bytes)
        except (CanaryError, AttestationError, OSError, TypeError, ValueError) as error:
            raise MsctlError(
                "CANARY_INPUT_INVALID",
                "P5 qualification inputs do not form one authenticated tuple",
            ) from error
        identity = environment["aws_instance_identity_document"]
        try:
            signature_valid = self.identity_verifier(
                identity,
                str(environment["aws_instance_identity_pkcs7"]),
                self.runtime.region,
            )
        except MsctlError:
            raise
        except Exception as error:
            raise MsctlError(
                "CANARY_INPUT_INVALID",
                "P5 qualification environment signature verification failed",
            ) from error
        if signature_valid is not True:
            raise MsctlError(
                "CANARY_INPUT_INVALID",
                "P5 qualification environment signature is invalid",
            )

        remote_argv = [
            "/usr/bin/python3",
            str(plan.release_root / "cluster" / "aws" / "p5" / "canary.py"),
            "--release-root",
            str(plan.release_root),
            "--release-receipt",
            str(plan.release_receipt_path),
            "--manifest",
            str(plan.run_manifest_path),
            "--dataset-receipt",
            str(plan.dataset_receipt_path),
            "--environment-receipt",
            str(plan.environment_receipt_path),
            "--runtime-lock",
            str(plan.runtime_lock_path),
            "--instance-id",
            instance_id,
            "--boot-id",
            boot_id,
            "--scratch-root",
            str(plan.scratch_root),
            "--output",
            str(plan.output_path),
            "--s3-root",
            self.runtime.s3_root,
            "--apply",
        ]
        core = {
            "schema_version": 1,
            "operation": "canary",
            "provider": AWS_P5_PROFILE,
            "seed": plan.seed,
            "release_sha256": plan.release_sha256,
            "run_manifest_sha256": plan.run_manifest_sha256,
            "dataset_sha256": plan.dataset_receipt_sha256,
            "dataset_pointer_sha256": plan.dataset_pointer_sha256,
            "dataset_verification_sha256": plan.dataset_receipt_sha256,
            "environment_receipt_sha256": plan.environment_receipt_sha256,
            "runtime_sha256": plan.runtime_lock_sha256,
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
                {
                    "name": "qualification-canary",
                    "argv": remote_argv,
                }
            ],
        }
        operation_intent = self._operation_envelope(
            core,
            instance_id=instance_id,
            terminate_at=None,
        )
        result = {
            "provider": AWS_P5_PROFILE,
            "profile_id": _V3_PROFILE_ID,
            "qualification_tuple_sha256": plan.tuple_sha256,
            "receipt_key_prefix": f"canaries/{plan.tuple_sha256}/",
            "remote_canary_argv": remote_argv,
            "operation_intent": operation_intent,
            "operation_id": operation_intent["operation_id"],
            "output": str(output_path),
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
                operation="verify canary instance",
            ),
            {"instance"},
            label="canary instance output",
        )
        selected = _aws_output_object(
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
            label="canary instance",
        )
        if (
            selected["account_id"] != environment["account_id"]
            or selected["instance_id"] != instance_id
            or selected["image_id"] != environment["ami_id"]
            or selected["instance_type"] != INSTANCE_TYPE
            or selected["state"] != "running"
            or selected["architecture"] != identity["architecture"]
            or selected["private_ip"] != identity["privateIp"]
        ):
            raise MsctlError(
                "CANARY_INPUT_INVALID",
                "selected P5 differs from the authenticated environment",
            )
        self._require_ssm_online(instance_id)
        self._ensure_argv_document()
        published_intent = self._publish_operation_intent(operation_intent)
        command_id = self._send_operation_intent(
            instance_id=instance_id,
            intent=operation_intent,
            published=published_intent,
            operation="send canary operation",
        )
        invocation_argv = self._aws_argv(
            "ssm",
            "get-command-invocation",
            "--instance-id",
            instance_id,
            "--command-id",
            command_id,
            query=(
                "{command:{command_id:CommandId,status:Status,"
                "stdout:StandardOutputContent,"
                "stderr:StandardErrorContent}}"
            ),
        )
        deadline = self.monotonic() + _CANARY_REMOTE_TIMEOUT_SECONDS
        while True:
            try:
                invocation_output = _aws_output_object(
                    self._run(
                        invocation_argv,
                        operation="consume canary operation",
                    ),
                    {"command"},
                    label="canary command output",
                )
            except MsctlError as error:
                if (
                    error.code != "AWS_COMMAND_FAILED"
                    or self.monotonic() >= deadline
                ):
                    raise
                self.sleep(_CANARY_POLL_SECONDS)
                continue
            invocation = _aws_output_object(
                invocation_output["command"],
                {"command_id", "status", "stdout", "stderr"},
                label="canary command",
            )
            if (
                invocation["command_id"] != command_id
                or not isinstance(invocation["status"], str)
                or not isinstance(invocation["stdout"], str)
                or not isinstance(invocation["stderr"], str)
                or invocation["stderr"] != ""
            ):
                raise MsctlError(
                    "CANARY_REMOTE_FAILED",
                    "remote P5 qualification returned invalid command evidence",
                )
            if invocation["status"] == "Success":
                break
            if invocation["status"] not in _ACTIVE_COMMAND_STATES:
                raise MsctlError(
                    "CANARY_REMOTE_FAILED",
                    "remote P5 qualification failed before receipt production",
                )
            if self.monotonic() >= deadline:
                raise MsctlError(
                    "CANARY_REMOTE_FAILED",
                    "remote P5 qualification exceeded its bounded deadline",
                )
            self.sleep(_CANARY_POLL_SECONDS)
        candidates = []
        for line in invocation["stdout"].splitlines():
            try:
                payload = (line + "\n").encode("ascii")
                parsed = parse_qualification_receipt_bytes(
                    payload,
                    plan=plan,
                )
            except (UnicodeEncodeError, CanaryError):
                continue
            candidates.append((parsed, payload))
        if len(candidates) != 1:
            raise MsctlError(
                "CANARY_RECEIPT_INVALID",
                "remote command did not emit one exact qualification receipt",
            )
        qualification, receipt_bytes = candidates[0]
        roundtrip = qualification["s3_roundtrip"]
        roundtrip_uri = str(roundtrip["object_uri"])
        prefix = self.runtime.s3_root.rstrip("/") + "/"
        if not roundtrip_uri.startswith(prefix):
            raise MsctlError(
                "CANARY_RECEIPT_INVALID",
                "roundtrip object is outside the pinned S3 root",
            )
        roundtrip_bucket, roundtrip_key = self._s3_location(
            roundtrip_uri.removeprefix(prefix)
        )
        roundtrip_sha256 = str(roundtrip["sha256"])
        roundtrip_checksum = base64.b64encode(
            bytes.fromhex(roundtrip_sha256)
        ).decode("ascii")
        roundtrip_version = str(roundtrip["version_id"])
        head_output = _aws_output_object(
            self._run(
                self._aws_argv(
                    "s3api",
                    "head-object",
                    "--bucket",
                    roundtrip_bucket,
                    "--key",
                    roundtrip_key,
                    "--version-id",
                    roundtrip_version,
                    "--checksum-mode",
                    "ENABLED",
                    query=(
                        "{object:{checksum_sha256:ChecksumSHA256,"
                        "content_length:ContentLength,version_id:VersionId}}"
                    ),
                ),
                operation="verify canary roundtrip object",
            ),
            {"object"},
            label="canary roundtrip head output",
        )
        head = _aws_output_object(
            head_output["object"],
            {"checksum_sha256", "content_length", "version_id"},
            label="canary roundtrip object",
        )
        if (
            head["checksum_sha256"] != roundtrip_checksum
            or type(head["content_length"]) is not int
            or head["content_length"] != roundtrip["bytes"]
            or head["version_id"] != roundtrip_version
        ):
            raise MsctlError(
                "S3_OBJECT_MISMATCH",
                "canary roundtrip object metadata does not match receipt",
            )
        with tempfile.TemporaryDirectory(
            prefix="msctl-canary-roundtrip-",
            dir=self.state_root,
        ) as roundtrip_directory:
            downloaded_path = Path(roundtrip_directory) / "roundtrip.json"
            download_output = _aws_output_object(
                self._run(
                    self._aws_argv(
                        "s3api",
                        "get-object",
                        "--bucket",
                        roundtrip_bucket,
                        "--key",
                        roundtrip_key,
                        "--version-id",
                        roundtrip_version,
                        "--checksum-mode",
                        "ENABLED",
                        str(downloaded_path),
                        query=(
                            "{object:{checksum_sha256:ChecksumSHA256,"
                            "version_id:VersionId}}"
                        ),
                    ),
                    operation="download canary roundtrip object",
                ),
                {"object"},
                label="canary roundtrip download output",
            )
            downloaded = _aws_output_object(
                download_output["object"],
                {"checksum_sha256", "version_id"},
                label="downloaded canary roundtrip object",
            )
            try:
                downloaded_bytes = read_regular_input(
                    downloaded_path,
                    label="downloaded canary roundtrip object",
                    maximum_bytes=1024 * 1024,
                )
            except (AttestationError, OSError) as error:
                raise MsctlError(
                    "S3_OBJECT_MISMATCH",
                    "downloaded canary roundtrip object is unsafe",
                ) from error
        if (
            downloaded["checksum_sha256"] != roundtrip_checksum
            or downloaded["version_id"] != roundtrip_version
            or downloaded_bytes != qualification_roundtrip_blob(plan)
            or hashlib.sha256(downloaded_bytes).hexdigest()
            != roundtrip_sha256
        ):
            raise MsctlError(
                "S3_OBJECT_MISMATCH",
                "downloaded canary roundtrip bytes do not match receipt",
            )

        receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        receipt_key = (
            f"canaries/{plan.tuple_sha256}/{receipt_sha256}.json"
        )
        receipt_bucket, receipt_object_key = self._s3_location(receipt_key)
        receipt_checksum = base64.b64encode(
            bytes.fromhex(receipt_sha256)
        ).decode("ascii")
        put_version = None
        with tempfile.TemporaryDirectory(
            prefix="msctl-canary-receipt-",
            dir=self.state_root,
        ) as receipt_directory:
            staged_receipt = Path(receipt_directory) / "qualification.json"
            staged_receipt.write_bytes(receipt_bytes)
            staged_receipt.chmod(0o600)
            put_argv = self._aws_argv(
                "s3api",
                "put-object",
                "--bucket",
                receipt_bucket,
                "--key",
                receipt_object_key,
                "--body",
                str(staged_receipt),
                "--content-length",
                str(len(receipt_bytes)),
                "--checksum-algorithm",
                "SHA256",
                "--checksum-sha256",
                receipt_checksum,
                "--metadata",
                (
                    f"qualification-tuple-sha256={plan.tuple_sha256},"
                    f"receipt-sha256={receipt_sha256}"
                ),
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
                        operation="publish canary receipt",
                    ),
                    {"object"},
                    label="canary receipt publication output",
                )
            except MsctlError as error:
                if error.code != "AWS_COMMAND_FAILED":
                    raise
            else:
                put_object = _aws_output_object(
                    put_output["object"],
                    {"checksum_sha256", "version_id"},
                    label="canary receipt publication",
                )
                if (
                    put_object["checksum_sha256"] != receipt_checksum
                    or not isinstance(put_object["version_id"], str)
                    or put_object["version_id"] in {"", "null"}
                ):
                    raise MsctlError(
                        "S3_OBJECT_MISMATCH",
                        "published canary receipt lacks checksum or version",
                    )
                put_version = put_object["version_id"]
        final_head_output = _aws_output_object(
            self._run(
                self._aws_argv(
                    "s3api",
                    "head-object",
                    "--bucket",
                    receipt_bucket,
                    "--key",
                    receipt_object_key,
                    "--checksum-mode",
                    "ENABLED",
                    query=(
                        "{object:{checksum_sha256:ChecksumSHA256,"
                        "content_length:ContentLength,metadata:Metadata,"
                        "version_id:VersionId}}"
                    ),
                ),
                operation="verify canary receipt publication",
            ),
            {"object"},
            label="canary receipt head output",
        )
        final_head = _aws_output_object(
            final_head_output["object"],
            {
                "checksum_sha256",
                "content_length",
                "metadata",
                "version_id",
            },
            label="published canary receipt",
        )
        version_id = final_head["version_id"]
        if (
            final_head["checksum_sha256"] != receipt_checksum
            or type(final_head["content_length"]) is not int
            or final_head["content_length"] != len(receipt_bytes)
            or final_head["metadata"]
            != {
                "qualification-tuple-sha256": plan.tuple_sha256,
                "receipt-sha256": receipt_sha256,
            }
            or not isinstance(version_id, str)
            or version_id in {"", "null"}
            or (put_version is not None and version_id != put_version)
        ):
            raise MsctlError(
                "S3_OBJECT_MISMATCH",
                "published canary receipt does not match verified bytes",
            )
        return {
            **result,
            "command_id": command_id,
            "intent_sha256": published_intent["intent_sha256"],
            "intent_uri": published_intent["intent_uri"],
            "receipt_sha256": receipt_sha256,
            "receipt_key": receipt_key,
            "receipt_uri": (
                f"{self.runtime.s3_root.rstrip('/')}/{receipt_key}"
            ),
            "published": True,
            "version_id": version_id,
        }

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

    def _collect_fetch_receipt(
        self,
        *,
        uri: str,
        sha256: str,
        version_id: str,
        operation: str,
        label: str,
    ) -> bytes:
        """GET one exact receipt version and independently hash its bytes."""

        prefix = self.runtime.s3_root.rstrip("/") + "/"
        if not isinstance(uri, str) or not uri.startswith(prefix):
            raise MsctlError(
                "COLLECT_RECEIPT_INVALID",
                f"{label} is outside the pinned S3 root",
            )
        bucket, key = self._s3_location(uri.removeprefix(prefix))
        expected_checksum = base64.b64encode(
            bytes.fromhex(sha256)
        ).decode("ascii")
        with tempfile.TemporaryDirectory(
            prefix="memorysplit-collect-receipt-",
            dir=Path(tempfile.gettempdir()).resolve(),
        ) as temporary:
            destination = Path(temporary) / "receipt.json"
            try:
                output = _aws_output_object(
                    self._run(
                        self._aws_argv(
                            "s3api",
                            "get-object",
                            "--bucket",
                            bucket,
                            "--key",
                            key,
                            "--version-id",
                            version_id,
                            "--checksum-mode",
                            "ENABLED",
                            str(destination),
                            query=(
                                "{receipt:{checksum_sha256:ChecksumSHA256,"
                                "version_id:VersionId}}"
                            ),
                        ),
                        operation=operation,
                    ),
                    {"receipt"},
                    label=f"{label} download",
                )
            except MsctlError as error:
                if error.code == "AWS_COMMAND_FAILED":
                    raise MsctlError(
                        "COLLECT_INCOMPLETE",
                        f"{label} is unavailable",
                    ) from error
                raise
            row = _aws_output_object(
                output["receipt"],
                {"checksum_sha256", "version_id"},
                label=f"{label} object",
            )
            try:
                payload = read_regular_input(
                    destination,
                    label=label,
                    maximum_bytes=16 * 1024 * 1024,
                )
            except (AttestationError, OSError) as error:
                raise MsctlError(
                    "COLLECT_RECEIPT_INVALID",
                    f"downloaded {label} is unsafe",
                ) from error
        if (
            row["checksum_sha256"] != expected_checksum
            or row["version_id"] != version_id
            or hashlib.sha256(payload).hexdigest() != sha256
        ):
            raise MsctlError(
                "COLLECT_RECEIPT_INVALID",
                f"downloaded {label} identity differs",
            )
        return payload

    def _collect_run_receipt_value(
        self,
        manifest: object,
        *,
        uri: str,
        sha256: str,
        version_id: str,
    ) -> tuple[dict[str, object], bytes, ProviderLifecycleBinding]:
        """Fetch and authenticate this seed's exact finalization receipt."""

        payload = self._collect_fetch_receipt(
            uri=uri,
            sha256=sha256,
            version_id=version_id,
            operation="fetch seed run finalization receipt",
            label="run finalization receipt",
        )
        try:
            value = parse_run_finalization_receipt_bytes(
                payload,
                receipt_uri=uri,
                receipt_sha256=sha256,
                receipt_version_id=version_id,
            )
        except FinalizationError as error:
            raise MsctlError(
                "COLLECT_RECEIPT_INVALID",
                f"run finalization receipt is invalid: {error}",
            ) from error
        if value["seed"] != manifest.seed:
            raise MsctlError(
                "COLLECT_RECEIPT_INVALID",
                "run finalization receipt seed differs from the manifest",
            )
        # The finalization may have happened on an earlier boot; every other
        # lifecycle commitment must match the authenticated binding exactly.
        try:
            expected_binding = dataclass_replace(
                self.lifecycle_binding,
                boot_id=str(value["boot_id"]),
            )
        except (TypeError, ValueError) as error:
            raise MsctlError(
                "COLLECT_RECEIPT_INVALID",
                "run finalization receipt boot identity is invalid",
            ) from error
        try:
            parse_run_finalization_receipt_bytes(
                payload,
                receipt_uri=uri,
                receipt_sha256=sha256,
                receipt_version_id=version_id,
                expected_binding=expected_binding,
            )
        except FinalizationError as error:
            raise MsctlError(
                "COLLECT_RECEIPT_INVALID",
                f"run finalization lifecycle authority differs: {error}",
            ) from error
        for field, expected_value in (
            ("release_sha256", manifest.release_sha256),
            ("release_receipt_sha256", manifest.release_receipt_sha256),
            ("run_manifest_sha256", manifest.sha256),
            ("dataset_receipt_sha256", manifest.dataset_receipt_sha256),
            ("dataset_build_id", manifest.dataset_build_id),
            ("ordered_stream_sha256", manifest.ordered_stream_sha256),
            ("source_commit", manifest.source_commit),
            ("source_tree", manifest.source_tree),
        ):
            if value[field] != expected_value:
                raise MsctlError(
                    "COLLECT_RECEIPT_INVALID",
                    f"run finalization {field} differs from the manifest",
                )
        by_run = {run.run_id: run for run in manifest.runs}
        for row in value["arms"]:
            run = by_run.get(row["run_id"])
            if run is None or row["config_sha256"] != run.config_sha256:
                raise MsctlError(
                    "COLLECT_RECEIPT_INVALID",
                    "run finalization arms do not bind the manifest runs",
                )
        return value, payload, expected_binding

    def _collect_checkpoint_receipt(
        self,
        manifest: object,
        expected_binding: ProviderLifecycleBinding,
        checkpoint_reference: Mapping[str, object],
    ) -> tuple[object, bytes]:
        """Fetch and cross-bind the exact Task 3C checkpoint receipt."""

        payload = self._collect_fetch_receipt(
            uri=str(checkpoint_reference["uri"]),
            sha256=str(checkpoint_reference["sha256"]),
            version_id=str(checkpoint_reference["version_id"]),
            operation="fetch seed checkpoint receipt",
            label="checkpoint receipt",
        )
        try:
            receipt = parse_paired_checkpoint_receipt_v3(
                payload,
                receipt_uri=str(checkpoint_reference["uri"]),
                receipt_sha256=str(checkpoint_reference["sha256"]),
                receipt_version_id=str(
                    checkpoint_reference["version_id"]
                ),
            )
        except MsctlError as error:
            raise MsctlError(
                "COLLECT_RECEIPT_INVALID",
                "checkpoint receipt is invalid",
            ) from error
        if receipt.provider_selection_sha256 is None:
            raise MsctlError(
                "COLLECT_RECEIPT_INVALID",
                "collection requires a provider-aware checkpoint receipt",
            )
        expected = expected_binding.to_dict()
        for field in LIFECYCLE_BINDING_FIELDS:
            actual = getattr(receipt, field)
            if field == "arms":
                actual = list(actual)
            if actual != expected[field]:
                raise MsctlError(
                    "COLLECT_RECEIPT_INVALID",
                    "checkpoint receipt lifecycle differs from the "
                    "authenticated provider authority",
                )
        by_run = {run.run_id: run for run in manifest.runs}
        if (
            receipt.seed != manifest.seed
            or receipt.release_sha256 != manifest.release_sha256
            or receipt.release_receipt_sha256
            != manifest.release_receipt_sha256
            or receipt.run_manifest_sha256 != manifest.sha256
            or receipt.dataset_receipt_sha256
            != manifest.dataset_receipt_sha256
            or receipt.dataset_build_id != manifest.dataset_build_id
            or receipt.ordered_stream_sha256
            != manifest.ordered_stream_sha256
            or receipt.source_commit != manifest.source_commit
            or receipt.source_tree != manifest.source_tree
            or any(
                by_run.get(checkpoint.run_id) is None
                or checkpoint.config_sha256
                != by_run[checkpoint.run_id].config_sha256
                for checkpoint in receipt.checkpoints
            )
        ):
            raise MsctlError(
                "COLLECT_RECEIPT_INVALID",
                "checkpoint receipt does not bind the selected manifest",
            )
        return receipt, payload

    @staticmethod
    def _collection_rows(
        run_value: Mapping[str, object],
        run_payload: bytes,
        run_reference: Mapping[str, object],
        checkpoint_receipt: object,
        checkpoint_payload: bytes,
    ) -> list[dict[str, object]]:
        """Enumerate the sixteen canonical collection rows."""

        rows: list[dict[str, object]] = []
        for arm_row in run_value["arms"]:
            for item in arm_row["snapshots"]:
                snapshot = item["object"]
                rows.append(
                    {
                        "arm": arm_row["arm"],
                        "bytes": snapshot["bytes"],
                        "kind": "snapshot",
                        "sha256": snapshot["sha256"],
                        "step": item["step"],
                        "uri": snapshot["uri"],
                        "version_id": snapshot["version_id"],
                    }
                )
            log = arm_row["log"]
            rows.append(
                {
                    "arm": arm_row["arm"],
                    "bytes": log["bytes"],
                    "kind": "log",
                    "sha256": log["sha256"],
                    "step": None,
                    "uri": log["uri"],
                    "version_id": log["version_id"],
                }
            )
        for checkpoint in checkpoint_receipt.checkpoints:
            rows.append(
                {
                    "arm": checkpoint.arm,
                    "bytes": checkpoint.object.bytes,
                    "kind": "checkpoint",
                    "sha256": checkpoint.object.sha256,
                    "step": None,
                    "uri": checkpoint.object.uri,
                    "version_id": checkpoint.object.version_id,
                }
            )
        reference = run_value["checkpoint_receipt"]
        rows.append(
            {
                "arm": None,
                "bytes": len(checkpoint_payload),
                "kind": "checkpoint_receipt",
                "sha256": reference["sha256"],
                "step": None,
                "uri": reference["uri"],
                "version_id": reference["version_id"],
            }
        )
        rows.append(
            {
                "arm": None,
                "bytes": len(run_payload),
                "kind": "run_receipt",
                "sha256": run_reference["sha256"],
                "step": None,
                "uri": run_reference["uri"],
                "version_id": run_reference["version_id"],
            }
        )
        return rows

    @staticmethod
    def _stage_collection_bytes(
        staging_fd: int,
        relative: str,
        payload: bytes,
    ) -> None:
        parent_fd, name = open_parent_at(
            staging_fd,
            relative,
            label="collection staging entry",
            create=True,
        )
        try:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            descriptor = os.open(name, flags, 0o600, dir_fd=parent_fd)
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            os.close(parent_fd)

    def _rehash_staged_collection_object(
        self,
        staging_fd: int,
        relative: str,
        row: Mapping[str, object],
    ) -> None:
        """Stream-hash the descriptor-pinned staged bytes independently."""

        try:
            descriptor, parent_fd, _name = open_regular_at(
                staging_fd,
                relative,
                label="collected object",
            )
        except MsctlError as error:
            raise MsctlError(
                "COLLECT_OBJECT_MISMATCH",
                "collected object was not materialized as a regular file",
                details={"uri": row["uri"]},
            ) from error
        try:
            size, digest = hash_fd(descriptor)
        finally:
            os.close(descriptor)
            os.close(parent_fd)
        if size != row["bytes"] or digest != row["sha256"]:
            raise MsctlError(
                "COLLECT_OBJECT_MISMATCH",
                "collected object bytes differ from the receipt identity",
                details={"uri": row["uri"]},
            )

    def _download_collection_body(
        self,
        row: Mapping[str, object],
        *,
        local_path: str,
    ) -> None:
        prefix = self.runtime.s3_root.rstrip("/") + "/"
        bucket, key = self._s3_location(
            str(row["uri"]).removeprefix(prefix)
        )
        argv = self._aws_argv(
            "s3api",
            "get-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--version-id",
            str(row["version_id"]),
            "--checksum-mode",
            "ENABLED",
            local_path,
            query=(
                "{object:{checksum_sha256:ChecksumSHA256,"
                "content_length:ContentLength,version_id:VersionId}}"
            ),
        )
        try:
            timeout_seconds = download_timeout_seconds(row["bytes"])
        except CollectionError as error:
            raise MsctlError(
                "COLLECT_OBJECT_MISMATCH",
                "collected object size is invalid",
            ) from error
        try:
            output = _aws_output_object(
                self.download_runner.run_json(
                    argv,
                    operation=f"collect {row['kind']} body",
                    timeout_seconds=timeout_seconds,
                ),
                {"object"},
                label="collection body download",
            )
        except MsctlError as error:
            if error.code in {"AWS_COMMAND_FAILED", "EXTERNAL_UNAVAILABLE"}:
                raise MsctlError(
                    "COLLECT_INCOMPLETE",
                    "collection evidence object is unavailable",
                    details={"uri": row["uri"]},
                ) from error
            raise
        downloaded = _aws_output_object(
            output["object"],
            {"checksum_sha256", "content_length", "version_id"},
            label="collection body object",
        )
        if (
            downloaded["checksum_sha256"]
            != base64.b64encode(
                bytes.fromhex(str(row["sha256"]))
            ).decode("ascii")
            or downloaded["content_length"] != row["bytes"]
            or downloaded["version_id"] != row["version_id"]
        ):
            raise MsctlError(
                "COLLECT_OBJECT_MISMATCH",
                "collection evidence object version or checksum differs",
                details={"uri": row["uri"]},
            )

    def _publish_collection_receipt(
        self,
        *,
        payload: bytes,
        digest: str,
        request_id: str,
        seed: int,
        body_path: str,
    ) -> str:
        """Publish the receipt no-replace and verify it by exact HEAD."""

        bucket, key = self._s3_location(
            collection_receipt_key(seed, digest)
        )
        checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
        expected_metadata = {
            "receipt-sha256": digest,
            "receipt-type": COLLECTION_RECEIPT_TYPE,
            "request-id": request_id,
            "seed": str(seed),
        }
        put = self._aws_argv(
            "s3api",
            "put-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--body",
            body_path,
            "--checksum-algorithm",
            "SHA256",
            "--checksum-sha256",
            checksum,
            "--metadata",
            ",".join(
                f"{name}={value}"
                for name, value in expected_metadata.items()
            ),
            "--if-none-match",
            "*",
            query=(
                "{object:{checksum_sha256:ChecksumSHA256,"
                "version_id:VersionId}}"
            ),
        )
        put_version: str | None = None
        try:
            put_output = _aws_output_object(
                self._run(put, operation="publish seed collection receipt"),
                {"object"},
                label="S3 collection receipt put",
            )
        except MsctlError as error:
            if error.code != "AWS_COMMAND_FAILED":
                raise
            # A lost or conflicting no-replace PUT recovers only through
            # the exact HEAD verification below.
            put_output = None
        if put_output is not None:
            put_row = _aws_output_object(
                put_output["object"],
                {"checksum_sha256", "version_id"},
                label="S3 collection receipt put object",
            )
            if (
                put_row["checksum_sha256"] != checksum
                or not isinstance(put_row["version_id"], str)
                or not put_row["version_id"]
            ):
                raise MsctlError(
                    "COLLECT_CONFLICT",
                    "S3 did not confirm the immutable collection receipt",
                )
            put_version = str(put_row["version_id"])
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
        try:
            head_output = _aws_output_object(
                self._run(head, operation="verify seed collection receipt"),
                {"object"},
                label="S3 collection receipt head",
            )
        except MsctlError as error:
            if error.code == "AWS_COMMAND_FAILED":
                raise MsctlError(
                    "COLLECT_CONFLICT",
                    "published collection receipt cannot be verified",
                ) from error
            raise
        row = _aws_output_object(
            head_output["object"],
            {"checksum_sha256", "content_length", "metadata", "version_id"},
            label="S3 collection receipt object",
        )
        if (
            row["checksum_sha256"] != checksum
            or row["content_length"] != len(payload)
            or row["metadata"] != expected_metadata
            or not isinstance(row["version_id"], str)
            or not row["version_id"]
            or (
                put_version is not None
                and row["version_id"] != put_version
            )
        ):
            raise MsctlError(
                "COLLECT_CONFLICT",
                "published collection receipt does not match local bytes",
            )
        return str(row["version_id"])

    def _verify_published_collection_head(
        self,
        reference: Mapping[str, object],
    ) -> None:
        """Exact-HEAD verify one durably recorded collection receipt."""

        prefix = self.runtime.s3_root.rstrip("/") + "/"
        uri = str(reference["uri"])
        if not uri.startswith(prefix):
            raise MsctlError(
                "COLLECT_INCOMPLETE",
                "recorded collection receipt is outside the pinned S3 root",
            )
        bucket, key = self._s3_location(uri.removeprefix(prefix))
        try:
            output = _aws_output_object(
                self._run(
                    self._aws_argv(
                        "s3api",
                        "head-object",
                        "--bucket",
                        bucket,
                        "--key",
                        key,
                        "--version-id",
                        str(reference["version_id"]),
                        "--checksum-mode",
                        "ENABLED",
                        query=(
                            "{object:{checksum_sha256:ChecksumSHA256,"
                            "content_length:ContentLength,"
                            "version_id:VersionId}}"
                        ),
                    ),
                    operation="verify published seed collection receipt",
                ),
                {"object"},
                label="published collection receipt",
            )
        except MsctlError as error:
            if error.code == "AWS_COMMAND_FAILED":
                raise MsctlError(
                    "COLLECT_INCOMPLETE",
                    "published collection receipt is unavailable",
                ) from error
            raise
        row = _aws_output_object(
            output["object"],
            {"checksum_sha256", "content_length", "version_id"},
            label="published collection receipt object",
        )
        if (
            row["checksum_sha256"]
            != base64.b64encode(
                bytes.fromhex(str(reference["sha256"]))
            ).decode("ascii")
            or row["content_length"] != reference["bytes"]
            or row["version_id"] != reference["version_id"]
        ):
            raise MsctlError(
                "COLLECT_INCOMPLETE",
                "published collection receipt differs from durable state",
            )

    def collect_seed_evidence(
        self,
        *,
        release: object,
        manifest: object,
        run_receipt: Mapping[str, object] | None,
        out: Path | str | None,
        apply: bool,
    ) -> dict[str, object]:
        """Collect one seed's sixteen training-evidence objects durably.

        Collection requires no paid-instance approval: it creates no
        capacity, sends no remote command, and only publishes one
        content-addressed collection receipt. Every evidence body stays
        opaque; only the finalization and checkpoint receipts are parsed.
        """

        # 1) Authenticate the manifest, release, provider lifecycle, and
        #    the exact operator-declared run receipt triple.
        self._validate_manifest(manifest)
        self._validate_release(release, manifest)
        if (
            not self._selected_manifest_lifecycle(manifest)
            or self.lifecycle_binding is None
        ):
            raise MsctlError(
                "OPERATION_UNSUPPORTED",
                "seed evidence collection requires an authenticated "
                "selected manifest",
            )
        if (
            not isinstance(run_receipt, Mapping)
            or set(run_receipt) != {"uri", "sha256", "version_id"}
        ):
            raise MsctlError(
                "COLLECT_RECEIPT_INVALID",
                "collection requires the exact run receipt triple",
            )
        try:
            receipt_sha256 = require_sha256(
                run_receipt["sha256"],
                label="run receipt",
            )
        except MsctlError as error:
            raise MsctlError(
                "COLLECT_RECEIPT_INVALID",
                "run receipt SHA-256 is invalid",
            ) from error
        receipt_version = run_receipt["version_id"]
        if (
            not isinstance(receipt_version, str)
            or receipt_version in {"", "null"}
        ):
            raise MsctlError(
                "COLLECT_RECEIPT_INVALID",
                "run receipt version ID must be a real non-null version",
            )
        expected_key = run_receipt_key(manifest.seed, receipt_sha256)
        receipt_uri = run_receipt["uri"]
        if receipt_uri != f"{self.runtime.s3_root}/{expected_key}":
            raise MsctlError(
                "COLLECT_RECEIPT_INVALID",
                "run receipt URI must use its canonical key under the "
                "pinned S3 root",
            )
        if out is None:
            raise MsctlError(
                "CLI_USAGE",
                "seed evidence collection requires --out",
            )
        # 2) Validate the nonexisting, nonsymlink output destination.
        destination = Path(out).absolute()
        if destination.exists() or destination.is_symlink():
            raise MsctlError(
                "COLLECT_DESTINATION_EXISTS",
                "refusing to replace an existing collection destination",
            )
        bucket, key = self._s3_location(expected_key)
        first_get = self._aws_argv(
            "s3api",
            "get-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--version-id",
            receipt_version,
            "--checksum-mode",
            "ENABLED",
            str(destination / expected_key),
            query=(
                "{receipt:{checksum_sha256:ChecksumSHA256,"
                "version_id:VersionId}}"
            ),
        )
        if not apply:
            # Later commands depend on the fetched receipt contents, so the
            # dry run performs zero AWS calls and renders only the first
            # exact finalization receipt GET.
            return {
                "provider": self.profile.provider,
                "seed": manifest.seed,
                "run_manifest_sha256": manifest.sha256,
                "run_receipt": dict(run_receipt),
                "out": str(destination),
                "commands": [first_get],
                "collected": 0,
                "idempotent": False,
            }

        store = StateStore(self.state_root)
        with store.locked():
            # 3) Idempotent replay or conflict, before any AWS call.
            existing = store.read_collection(manifest.sha256)
            if existing is not None:
                stored = existing["run_receipt"]
                if (
                    stored["uri"] != receipt_uri
                    or stored["sha256"] != receipt_sha256
                    or stored["version_id"] != receipt_version
                ):
                    raise MsctlError(
                        "COLLECT_CONFLICT",
                        "existing collection state binds a different run "
                        "receipt",
                    )
                self._verify_published_collection_head(
                    existing["collection_receipt"]
                )
                return {
                    "provider": self.profile.provider,
                    "seed": manifest.seed,
                    "run_manifest_sha256": manifest.sha256,
                    "run_receipt": dict(stored),
                    "collection_receipt": dict(
                        existing["collection_receipt"]
                    ),
                    "objects_collected": existing["objects_collected"],
                    "bytes_collected": existing["bytes_collected"],
                    "out": str(destination),
                    "collected": 0,
                    "idempotent": True,
                }
            # 4) GET and authenticate the exact finalization receipt.
            run_value, run_payload, expected_binding = (
                self._collect_run_receipt_value(
                    manifest,
                    uri=receipt_uri,
                    sha256=receipt_sha256,
                    version_id=receipt_version,
                )
            )
            # 5) GET and cross-bind the exact Task 3C checkpoint receipt.
            checkpoint_receipt, checkpoint_payload = (
                self._collect_checkpoint_receipt(
                    manifest,
                    expected_binding,
                    run_value["checkpoint_receipt"],
                )
            )
            run_reference = {
                "bytes": len(run_payload),
                "sha256": receipt_sha256,
                "uri": receipt_uri,
                "version_id": receipt_version,
            }
            rows = self._collection_rows(
                run_value,
                run_payload,
                run_reference,
                checkpoint_receipt,
                checkpoint_payload,
            )
            if len(rows) != SEED_COLLECTION_OBJECT_COUNT:
                raise MsctlError(
                    "COLLECT_INCOMPLETE",
                    "collection did not enumerate all sixteen objects",
                )
            prefix = self.runtime.s3_root.rstrip("/") + "/"
            if any(
                not str(row["uri"]).startswith(prefix) for row in rows
            ):
                raise MsctlError(
                    "COLLECT_OBJECT_MISMATCH",
                    "collection object is outside the pinned S3 root",
                )
            # 6) GET every body by exact version into a private no-follow
            #    staging tree and independently stream-rehash each file.
            request_id = secrets.token_hex(16)
            parent_fd = open_directory(
                destination.parent,
                label="collection destination parent",
                create=True,
            )
            staging_name = f".{destination.name}.collect-{request_id}"
            staging_path = destination.parent / staging_name
            published = False
            staging_fd: int | None = None
            try:
                os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
                staging_fd = open_directory_at(
                    parent_fd,
                    staging_name,
                    label="collection staging",
                )
                bytes_collected = 0
                for row in rows:
                    relative = str(row["uri"]).removeprefix(prefix)
                    if row["kind"] in {"checkpoint_receipt", "run_receipt"}:
                        self._stage_collection_bytes(
                            staging_fd,
                            relative,
                            (
                                checkpoint_payload
                                if row["kind"] == "checkpoint_receipt"
                                else run_payload
                            ),
                        )
                    else:
                        directory_fd, _name = open_parent_at(
                            staging_fd,
                            relative,
                            label="collection staging entry",
                            create=True,
                        )
                        os.close(directory_fd)
                        self._download_collection_body(
                            row,
                            local_path=str(staging_path / relative),
                        )
                    self._rehash_staged_collection_object(
                        staging_fd,
                        relative,
                        row,
                    )
                    bytes_collected += int(row["bytes"])
                # 7) Build and self-parse the canonical collection receipt.
                shared_fields = COLLECTION_RECEIPT_FIELDS - {
                    "checkpoint_receipt",
                    "collected_at",
                    "objects",
                    "receipt_type",
                    "request_id",
                    "run_receipt",
                }
                collection_value: dict[str, object] = {
                    field: run_value[field] for field in shared_fields
                }
                collection_value.update(
                    {
                        "receipt_type": COLLECTION_RECEIPT_TYPE,
                        "request_id": request_id,
                        "collected_at": datetime.now(UTC).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "run_receipt": dict(run_reference),
                        "checkpoint_receipt": {
                            "bytes": len(checkpoint_payload),
                            "sha256": str(
                                run_value["checkpoint_receipt"]["sha256"]
                            ),
                            "uri": str(
                                run_value["checkpoint_receipt"]["uri"]
                            ),
                            "version_id": str(
                                run_value["checkpoint_receipt"][
                                    "version_id"
                                ]
                            ),
                        },
                        "objects": [dict(sorted(row.items())) for row in rows],
                    }
                )
                collection_bytes = _canonical_json(collection_value)
                collection_sha256 = hashlib.sha256(
                    collection_bytes
                ).hexdigest()
                collection_uri = f"{self.runtime.s3_root}/" + (
                    collection_receipt_key(manifest.seed, collection_sha256)
                )
                try:
                    parse_seed_collection_receipt_bytes(
                        collection_bytes,
                        receipt_uri=collection_uri,
                        receipt_sha256=collection_sha256,
                        receipt_version_id="unpublished",
                        expected_binding=expected_binding,
                    )
                except CollectionError as error:
                    raise MsctlError(
                        "COLLECT_RECEIPT_INVALID",
                        f"built collection receipt failed self-parse: "
                        f"{error}",
                    ) from error
                receipt_relative = collection_receipt_key(
                    manifest.seed,
                    collection_sha256,
                )
                self._stage_collection_bytes(
                    staging_fd,
                    receipt_relative,
                    collection_bytes,
                )
                # 8) Publish no-replace, checksum-bound, then exact HEAD.
                version_id = self._publish_collection_receipt(
                    payload=collection_bytes,
                    digest=collection_sha256,
                    request_id=request_id,
                    seed=manifest.seed,
                    body_path=str(staging_path / receipt_relative),
                )
                collection_reference = {
                    "bytes": len(collection_bytes),
                    "sha256": collection_sha256,
                    "uri": collection_uri,
                    "version_id": version_id,
                }
                # 9) Atomically write collection state, then rename the
                #    staging tree no-replace.
                now = _timestamp()
                store.write_collection(
                    manifest.sha256,
                    {
                        **self.lifecycle_binding.to_dict(),
                        "schema_version": 2,
                        "operation": "collect",
                        "run_manifest_sha256": manifest.sha256,
                        "release_sha256": manifest.release_sha256,
                        "release_receipt_sha256": (
                            manifest.release_receipt_sha256
                        ),
                        "run_receipt": dict(run_reference),
                        "collection_receipt": dict(collection_reference),
                        "objects_collected": SEED_COLLECTION_OBJECT_COUNT,
                        "bytes_collected": bytes_collected,
                        "status": "Published",
                        "created_at": now,
                        "updated_at": now,
                    },
                )
                try:
                    rename_noreplace_at(
                        parent_fd,
                        staging_name,
                        parent_fd,
                        destination.name,
                    )
                except FileExistsError as error:
                    raise MsctlError(
                        "COLLECT_DESTINATION_EXISTS",
                        "refusing to replace an existing collection "
                        "destination",
                    ) from error
                published = True
            finally:
                if staging_fd is not None:
                    os.close(staging_fd)
                if not published:
                    try:
                        remove_tree_at(
                            parent_fd,
                            staging_name,
                            label="collection staging",
                        )
                    except MsctlError:
                        pass
                    except FileNotFoundError:
                        pass
                os.close(parent_fd)
            # 10) Report the durable identities.
            return {
                "provider": self.profile.provider,
                "seed": manifest.seed,
                "run_manifest_sha256": manifest.sha256,
                "run_receipt": dict(run_reference),
                "collection_receipt": dict(collection_reference),
                "objects_collected": SEED_COLLECTION_OBJECT_COUNT,
                "bytes_collected": bytes_collected,
                "out": str(destination),
                "collected": SEED_COLLECTION_OBJECT_COUNT,
                "idempotent": False,
            }

    def _download_cohort_object(
        self,
        row: Mapping[str, object],
        *,
        kind: str,
        local_path: str,
    ) -> None:
        """GET one exact-version evidence body through the bounded runner."""

        prefix = self.runtime.s3_root.rstrip("/") + "/"
        bucket, key = self._s3_location(
            str(row["uri"]).removeprefix(prefix)
        )
        argv = self._aws_argv(
            "s3api",
            "get-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--version-id",
            str(row["version_id"]),
            "--checksum-mode",
            "ENABLED",
            local_path,
            query=(
                "{object:{checksum_sha256:ChecksumSHA256,"
                "content_length:ContentLength,version_id:VersionId}}"
            ),
        )
        try:
            timeout_seconds = download_timeout_seconds(row["bytes"])
        except CollectionError as error:
            raise MsctlError(
                "COHORT_COLLECT_OBJECT_MISMATCH",
                "cohort evidence object size is invalid",
            ) from error
        try:
            output = _aws_output_object(
                self.download_runner.run_json(
                    argv,
                    operation=f"collect cohort {kind} body",
                    timeout_seconds=timeout_seconds,
                ),
                {"object"},
                label="cohort evidence body download",
            )
        except MsctlError as error:
            if error.code in {"AWS_COMMAND_FAILED", "EXTERNAL_UNAVAILABLE"}:
                raise MsctlError(
                    "COHORT_COLLECT_INCOMPLETE",
                    "cohort evidence object is unavailable",
                    details={"uri": row["uri"]},
                ) from error
            raise
        downloaded = _aws_output_object(
            output["object"],
            {"checksum_sha256", "content_length", "version_id"},
            label="cohort evidence body object",
        )
        if (
            downloaded["checksum_sha256"]
            != base64.b64encode(
                bytes.fromhex(str(row["sha256"]))
            ).decode("ascii")
            or downloaded["content_length"] != row["bytes"]
            or downloaded["version_id"] != row["version_id"]
        ):
            raise MsctlError(
                "COHORT_COLLECT_OBJECT_MISMATCH",
                "cohort evidence object version or checksum differs",
                details={"uri": row["uri"]},
            )

    def _rehash_staged_cohort_object(
        self,
        staging_fd: int,
        relative: str,
        *,
        row: Mapping[str, object],
        mode: int,
        read: bool,
    ) -> bytes | None:
        """Independently stream-hash the descriptor-pinned staged bytes."""

        try:
            descriptor, parent_fd, _name = open_regular_at(
                staging_fd,
                relative,
                label="collected cohort object",
            )
        except MsctlError as error:
            raise MsctlError(
                "COHORT_COLLECT_OBJECT_MISMATCH",
                "cohort evidence object was not materialized as a regular "
                "file",
                details={"uri": row["uri"]},
            ) from error
        try:
            size, digest = hash_fd(descriptor)
            payload = read_fd(descriptor) if read else None
            os.fchmod(descriptor, mode)
        finally:
            os.close(descriptor)
            os.close(parent_fd)
        if size != row["bytes"] or digest != row["sha256"]:
            raise MsctlError(
                "COHORT_COLLECT_OBJECT_MISMATCH",
                "cohort evidence bytes differ from their declared identity",
                details={"uri": row["uri"]},
            )
        return payload

    def _fetch_cohort_evidence(
        self,
        staging_fd: int,
        staging_path: Path,
        relative: str,
        *,
        row: Mapping[str, object],
        kind: str,
        mode: int,
        read: bool,
    ) -> bytes | None:
        directory_fd, _name = open_parent_at(
            staging_fd,
            relative,
            label="cohort collection staging entry",
            create=True,
        )
        os.close(directory_fd)
        self._download_cohort_object(
            row,
            kind=kind,
            local_path=str(staging_path / relative),
        )
        return self._rehash_staged_cohort_object(
            staging_fd,
            relative,
            row=row,
            mode=mode,
            read=read,
        )

    def _require_cohort_selection_authority(
        self,
        selection: Mapping[str, object],
        *,
        label: str,
    ) -> None:
        """Require exact equality with the sealed evaluator authority."""

        binding = self.lifecycle_binding
        expected = {
            "cohort_id": binding.cohort_id,
            "provider_selection_sha256": binding.provider_selection_sha256,
            "provider_selection_s3_version_id": (
                binding.provider_selection_version_id
            ),
            "hardware_amendment_sha256": binding.hardware_amendment_sha256,
            "selected_provider": binding.provider,
            "profile_id": binding.profile_id,
            "profile_sha256": binding.profile_sha256,
            "runtime_lock_sha256": binding.runtime_lock_sha256,
            "qualification_evidence_sha256": (
                binding.qualification_evidence_sha256
            ),
            "environment_receipt_sha256": (
                binding.qualification_environment_receipt_sha256
            ),
            "canary_receipt_sha256": (
                binding.qualification_canary_receipt_sha256
            ),
            "approval_receipt_sha256": (
                binding.qualification_approval_receipt_sha256
            ),
            "approval_public_key_sha256": (
                binding.qualification_approval_public_key_sha256
            ),
        }
        if any(
            selection.get(field) != value
            for field, value in expected.items()
        ):
            raise MsctlError(
                "COHORT_COLLECT_EVIDENCE_INVALID",
                f"{label} provider selection differs from the "
                "authenticated cohort selection authority",
            )

    def _publish_cohort_collection_receipt(
        self,
        *,
        payload: bytes,
        digest: str,
        request_id: str,
        body_path: str,
    ) -> str:
        """Publish the cohort receipt no-replace and verify by exact HEAD."""

        bucket, key = self._s3_location(
            cohort_collection_receipt_key(digest)
        )
        checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
        expected_metadata = {
            "receipt-sha256": digest,
            "receipt-type": COHORT_COLLECTION_RECEIPT_TYPE,
            "request-id": request_id,
        }
        put = self._aws_argv(
            "s3api",
            "put-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--body",
            body_path,
            "--checksum-algorithm",
            "SHA256",
            "--checksum-sha256",
            checksum,
            "--metadata",
            ",".join(
                f"{name}={value}"
                for name, value in expected_metadata.items()
            ),
            "--if-none-match",
            "*",
            query=(
                "{object:{checksum_sha256:ChecksumSHA256,"
                "version_id:VersionId}}"
            ),
        )
        put_version: str | None = None
        try:
            put_output = _aws_output_object(
                self._run(
                    put,
                    operation="publish cohort collection receipt",
                ),
                {"object"},
                label="S3 cohort collection receipt put",
            )
        except MsctlError as error:
            if error.code != "AWS_COMMAND_FAILED":
                raise
            # A lost or conflicting no-replace PUT recovers only through
            # the exact HEAD verification below.
            put_output = None
        if put_output is not None:
            put_row = _aws_output_object(
                put_output["object"],
                {"checksum_sha256", "version_id"},
                label="S3 cohort collection receipt put object",
            )
            if (
                put_row["checksum_sha256"] != checksum
                or not isinstance(put_row["version_id"], str)
                or not put_row["version_id"]
            ):
                raise MsctlError(
                    "COHORT_COLLECT_CONFLICT",
                    "S3 did not confirm the immutable cohort collection "
                    "receipt",
                )
            put_version = str(put_row["version_id"])
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
        try:
            head_output = _aws_output_object(
                self._run(
                    head,
                    operation="verify cohort collection receipt",
                ),
                {"object"},
                label="S3 cohort collection receipt head",
            )
        except MsctlError as error:
            if error.code == "AWS_COMMAND_FAILED":
                raise MsctlError(
                    "COHORT_COLLECT_CONFLICT",
                    "published cohort collection receipt cannot be verified",
                ) from error
            raise
        row = _aws_output_object(
            head_output["object"],
            {"checksum_sha256", "content_length", "metadata", "version_id"},
            label="S3 cohort collection receipt object",
        )
        if (
            row["checksum_sha256"] != checksum
            or row["content_length"] != len(payload)
            or row["metadata"] != expected_metadata
            or not isinstance(row["version_id"], str)
            or not row["version_id"]
            or (
                put_version is not None
                and row["version_id"] != put_version
            )
        ):
            raise MsctlError(
                "COHORT_COLLECT_CONFLICT",
                "published cohort collection receipt does not match local "
                "bytes",
            )
        return str(row["version_id"])

    def _verify_published_cohort_collection_head(
        self,
        reference: Mapping[str, object],
    ) -> None:
        """Exact-HEAD verify the durably recorded cohort receipt."""

        prefix = self.runtime.s3_root.rstrip("/") + "/"
        uri = str(reference["uri"])
        if not uri.startswith(prefix):
            raise MsctlError(
                "COHORT_COLLECT_INCOMPLETE",
                "recorded cohort collection receipt is outside the pinned "
                "S3 root",
            )
        bucket, key = self._s3_location(uri.removeprefix(prefix))
        try:
            output = _aws_output_object(
                self._run(
                    self._aws_argv(
                        "s3api",
                        "head-object",
                        "--bucket",
                        bucket,
                        "--key",
                        key,
                        "--version-id",
                        str(reference["version_id"]),
                        "--checksum-mode",
                        "ENABLED",
                        query=(
                            "{object:{checksum_sha256:ChecksumSHA256,"
                            "content_length:ContentLength,"
                            "version_id:VersionId}}"
                        ),
                    ),
                    operation="verify published cohort collection receipt",
                ),
                {"object"},
                label="published cohort collection receipt",
            )
        except MsctlError as error:
            if error.code == "AWS_COMMAND_FAILED":
                raise MsctlError(
                    "COHORT_COLLECT_INCOMPLETE",
                    "published cohort collection receipt is unavailable",
                ) from error
            raise
        row = _aws_output_object(
            output["object"],
            {"checksum_sha256", "content_length", "version_id"},
            label="published cohort collection receipt object",
        )
        if (
            row["checksum_sha256"]
            != base64.b64encode(
                bytes.fromhex(str(reference["sha256"]))
            ).decode("ascii")
            or row["content_length"] != reference["bytes"]
            or row["version_id"] != reference["version_id"]
        ):
            raise MsctlError(
                "COHORT_COLLECT_INCOMPLETE",
                "published cohort collection receipt differs from durable "
                "state",
            )

    def collect_cohort_evidence(
        self,
        *,
        release: object,
        manifest: object,
        cohort_report: Mapping[str, object] | None,
        evidence_index: Path | str | None,
        out: Path | str | None,
        apply: bool,
    ) -> dict[str, object]:
        """Collect the 1,012 cohort evaluation-evidence objects durably.

        Like per-seed collection, cohort collection requires no
        paid-instance approval: it creates no capacity, sends no remote
        command, and only publishes one content-addressed cohort collection
        receipt. Every evidence body other than the report, the lock, the
        ten seed collection receipts, and the 100 ``output.json`` manifests
        stays bytes-opaque; outcome interpretation happens only inside the
        function-locally imported Task 4C report replay.
        """

        # 1) Authenticate the manifest, release, and provider lifecycle,
        #    then the exact operator anchor triple and evidence index.
        self._validate_manifest(manifest)
        self._validate_release(release, manifest)
        if (
            not self._selected_manifest_lifecycle(manifest)
            or self.lifecycle_binding is None
        ):
            raise MsctlError(
                "OPERATION_UNSUPPORTED",
                "cohort evidence collection requires an authenticated "
                "selected manifest",
            )
        binding = self.lifecycle_binding
        if manifest.seed != 9:
            # The terminal seed's instance is the one cleanup will later
            # terminate; every other seed fails before any AWS call.
            raise MsctlError(
                "COHORT_COLLECT_EVIDENCE_INVALID",
                "cohort evidence collection requires the terminal seed-9 "
                "manifest",
            )
        if (
            not isinstance(cohort_report, Mapping)
            or set(cohort_report) != {"uri", "sha256", "version_id"}
        ):
            raise MsctlError(
                "COHORT_COLLECT_EVIDENCE_INVALID",
                "cohort collection requires the exact cohort report triple",
            )
        try:
            report_sha256 = require_sha256(
                cohort_report["sha256"],
                label="cohort report",
            )
        except MsctlError as error:
            raise MsctlError(
                "COHORT_COLLECT_EVIDENCE_INVALID",
                "cohort report SHA-256 is invalid",
            ) from error
        report_version = cohort_report["version_id"]
        if (
            not isinstance(report_version, str)
            or report_version in {"", "null"}
        ):
            raise MsctlError(
                "COHORT_COLLECT_EVIDENCE_INVALID",
                "cohort report version ID must be a real non-null version",
            )
        report_key = cohort_report_object_key(report_sha256)
        report_uri = cohort_report["uri"]
        if report_uri != f"{self.runtime.s3_root}/{report_key}":
            raise MsctlError(
                "COHORT_COLLECT_EVIDENCE_INVALID",
                "cohort report URI must use its canonical key under the "
                "pinned S3 root",
            )
        if evidence_index is None:
            raise MsctlError(
                "CLI_USAGE",
                "cohort evidence collection requires --evidence-index",
            )
        try:
            index_payload = read_regular_input(
                Path(evidence_index),
                label="cohort evidence index",
                maximum_bytes=64 * 1024 * 1024,
            )
        except (AttestationError, OSError) as error:
            raise MsctlError(
                "COHORT_COLLECT_EVIDENCE_INVALID",
                "cohort evidence index is unreadable",
            ) from error
        try:
            index = parse_cohort_evidence_index(
                index_payload,
                s3_root=self.runtime.s3_root,
            )
        except CohortCollectionError as error:
            raise MsctlError(
                "COHORT_COLLECT_EVIDENCE_INVALID",
                f"cohort evidence index is invalid: {error}",
            ) from error
        report_row = dict(index["cohort_report"])
        if (
            report_row["uri"] != report_uri
            or report_row["sha256"] != report_sha256
            or report_row["version_id"] != report_version
        ):
            raise MsctlError(
                "COHORT_COLLECT_EVIDENCE_INVALID",
                "evidence index cohort report differs from the operator "
                "anchor triple",
            )
        lock_row = dict(index["study_lock"])
        if out is None:
            raise MsctlError(
                "CLI_USAGE",
                "cohort evidence collection requires --out",
            )
        # 2) Validate the nonexisting, nonsymlink output destination.
        destination = Path(out).absolute()
        if destination.exists() or destination.is_symlink():
            raise MsctlError(
                "COHORT_COLLECT_DESTINATION_EXISTS",
                "refusing to replace an existing cohort collection "
                "destination",
            )
        report_relative = f"cohort-report-{report_sha256}/cohort-report.json"
        bucket, key = self._s3_location(report_key)
        first_get = self._aws_argv(
            "s3api",
            "get-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--version-id",
            report_version,
            "--checksum-mode",
            "ENABLED",
            str(destination / report_relative),
            query=(
                "{object:{checksum_sha256:ChecksumSHA256,"
                "content_length:ContentLength,version_id:VersionId}}"
            ),
        )
        if not apply:
            # Every later command depends on the fetched report contents,
            # so the dry run performs zero AWS calls and renders only the
            # first exact cohort-report GET.
            return {
                "provider": self.profile.provider,
                "seed": manifest.seed,
                "run_manifest_sha256": manifest.sha256,
                "cohort_report": dict(cohort_report),
                "out": str(destination),
                "commands": [first_get],
                "collected": 0,
                "idempotent": False,
            }

        store = StateStore(self.state_root)
        with store.locked():
            # 3) Idempotent replay or conflict on the singleton cohort
            #    state, before any AWS call.
            existing = store.read_cohort_collection()
            if existing is not None:
                recorded_report = existing["cohort_report"]
                if (
                    any(
                        recorded_report[field] != cohort_report[field]
                        for field in ("uri", "sha256", "version_id")
                    )
                    or existing["study_lock"] != lock_row
                    or existing["run_manifest_sha256"] != manifest.sha256
                ):
                    raise MsctlError(
                        "COHORT_COLLECT_CONFLICT",
                        "existing cohort collection state binds different "
                        "anchors",
                    )
                self._verify_published_cohort_collection_head(
                    existing["cohort_collection_receipt"]
                )
                return {
                    "provider": self.profile.provider,
                    "seed": manifest.seed,
                    "run_manifest_sha256": manifest.sha256,
                    "study_lock": dict(existing["study_lock"]),
                    "cohort_report": dict(existing["cohort_report"]),
                    "cohort_collection_receipt": dict(
                        existing["cohort_collection_receipt"]
                    ),
                    "objects_collected": existing["objects_collected"],
                    "bytes_collected": existing["bytes_collected"],
                    "out": str(destination),
                    "collected": 0,
                    "idempotent": True,
                }
            request_id = secrets.token_hex(16)
            parent_fd = open_directory(
                destination.parent,
                label="cohort collection destination parent",
                create=True,
            )
            staging_name = (
                f".{destination.name}.collect-cohort-{request_id}"
            )
            staging_path = destination.parent / staging_name
            published = False
            staging_fd: int | None = None
            try:
                os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
                staging_fd = open_directory_at(
                    parent_fd,
                    staging_name,
                    label="cohort collection staging",
                )
                # 4) GET the cohort report by exact version, rehash, and
                #    authenticate it against the index and the controller
                #    authority.
                report_bytes = self._fetch_cohort_evidence(
                    staging_fd,
                    staging_path,
                    report_relative,
                    row=report_row,
                    kind="report",
                    mode=0o444,
                    read=True,
                )
                from evals.confirmatory.aggregate import CohortReport
                from evals.confirmatory.contracts import (
                    canonical_json_bytes,
                )

                try:
                    report_value = CohortReport.from_dict(
                        _strict_cohort_json(
                            report_bytes,
                            label="cohort report",
                        )
                    )
                    if report_value.canonical_bytes != report_bytes:
                        raise ValueError(
                            "cohort report bytes are not canonical"
                        )
                except (
                    CohortCollectionError,
                    TypeError,
                    ValueError,
                ) as error:
                    raise MsctlError(
                        "COHORT_COLLECT_EVIDENCE_INVALID",
                        f"cohort report is invalid: {error}",
                    ) from error
                for seed_row, receipt_row in zip(
                    index["seed_collections"],
                    report_value.collection_receipts,
                    strict=True,
                ):
                    if any(
                        receipt_row[field] != seed_row[field]
                        for field in ("seed", "uri", "sha256", "version_id")
                    ):
                        raise MsctlError(
                            "COHORT_COLLECT_EVIDENCE_INVALID",
                            "cohort report collection receipts differ from "
                            "the evidence index",
                        )
                if report_value.study_lock_sha256 != lock_row["sha256"]:
                    raise MsctlError(
                        "COHORT_COLLECT_EVIDENCE_INVALID",
                        "cohort report study-lock commitment differs from "
                        "the evidence index",
                    )
                if (
                    report_value.sealed_evaluation_release_sha256
                    != manifest.sealed_evaluation_release_sha256
                    or report_value.preregistration_sha256
                    != manifest.preregistration_sha256
                ):
                    raise MsctlError(
                        "COHORT_COLLECT_EVIDENCE_INVALID",
                        "cohort report sealed-evaluation commitments differ "
                        "from the authenticated manifest",
                    )
                self._require_cohort_selection_authority(
                    report_value.provider_selection,
                    label="cohort report",
                )
                # 5) GET the study lock by exact version; its content hash
                #    is the report's commitment; bind it to the controller
                #    authority with boot rebound only.
                from evals.confirmatory.study_lock import StudyLockV3

                lock_relative = (
                    f"study-lock-{lock_row['sha256']}/study-lock.json"
                )
                lock_bytes = self._fetch_cohort_evidence(
                    staging_fd,
                    staging_path,
                    lock_relative,
                    row=lock_row,
                    kind="study lock",
                    mode=0o444,
                    read=True,
                )
                try:
                    lock_value = StudyLockV3.from_dict(
                        _strict_cohort_json(
                            lock_bytes,
                            label="study lock",
                        )
                    )
                    if canonical_json_bytes(lock_value.to_dict()) != (
                        lock_bytes
                    ):
                        raise ValueError(
                            "study lock bytes are not canonical"
                        )
                except (
                    CohortCollectionError,
                    TypeError,
                    ValueError,
                ) as error:
                    raise MsctlError(
                        "COHORT_COLLECT_EVIDENCE_INVALID",
                        f"study lock is invalid: {error}",
                    ) from error
                self._require_cohort_selection_authority(
                    lock_value.provider_selection.to_dict(),
                    label="study lock",
                )
                lock_binding = lock_value.lifecycle_binding(manifest.seed)
                # A reboot between training and collection is legitimate;
                # the exact instance and every other lifecycle commitment
                # must match the authenticated controller binding.
                if lock_binding != dataclass_replace(
                    binding,
                    boot_id=lock_binding.boot_id,
                ):
                    raise MsctlError(
                        "COHORT_COLLECT_EVIDENCE_INVALID",
                        "study lock seed-9 lifecycle differs from the "
                        "authenticated provider authority",
                    )
                for lifecycle, seed_row in zip(
                    lock_value.seed_lifecycles,
                    index["seed_collections"],
                    strict=True,
                ):
                    if (
                        lifecycle.collection_receipt_s3_uri
                        != seed_row["uri"]
                        or lifecycle.collection_receipt_sha256
                        != seed_row["sha256"]
                        or lifecycle.collection_receipt_s3_version_id
                        != seed_row["version_id"]
                    ):
                        raise MsctlError(
                            "COHORT_COLLECT_EVIDENCE_INVALID",
                            "study lock collection receipts differ from "
                            "the evidence index",
                        )
                if (
                    lock_value.sealed_evaluation_release_sha256
                    != manifest.sealed_evaluation_release_sha256
                ):
                    raise MsctlError(
                        "COHORT_COLLECT_EVIDENCE_INVALID",
                        "study lock sealed-evaluation release differs from "
                        "the authenticated manifest",
                    )
                uniform = lock_value.snapshots[0]
                if (
                    uniform.data_receipt_sha256
                    != manifest.dataset_receipt_sha256
                    or uniform.data_build_id != manifest.dataset_build_id
                    or uniform.ordered_stream_sha256
                    != manifest.ordered_stream_sha256
                ):
                    raise MsctlError(
                        "COHORT_COLLECT_EVIDENCE_INVALID",
                        "study lock dataset identity differs from the "
                        "authenticated manifest",
                    )
                # 6) GET the ten per-seed collection receipts by exact
                #    version; their full Task 3F parse happens inside the
                #    replay's snapshot planning.
                seed_payloads: dict[int, bytes] = {}
                for seed_row in index["seed_collections"]:
                    seed_relative = (
                        f"seed-collections/seed-{seed_row['seed']}/"
                        f"sha256/{seed_row['sha256']}.json"
                    )
                    seed_payloads[seed_row["seed"]] = (
                        self._fetch_cohort_evidence(
                            staging_fd,
                            staging_path,
                            seed_relative,
                            row=seed_row,
                            kind="seed collection receipt",
                            mode=0o600,
                            read=True,
                        )
                    )
                # 7) For each of the 100 slots in frozen order: GET and
                #    authenticate output.json, then GET its nine artifacts
                #    bytes-opaquely into the replay-shaped staging tree.
                output_ids: list[str] = []
                for output in index["outputs"]:
                    output_id = output["output_id"]
                    output_ids.append(output_id)
                    members = {
                        row["member"]: row for row in output["members"]
                    }
                    manifest_row = members["output.json"]
                    slot_index = output["slot_index"]
                    manifest_bytes = self._fetch_cohort_evidence(
                        staging_fd,
                        staging_path,
                        f"outputs/{output_id}/output.json",
                        row=manifest_row,
                        kind="output manifest",
                        mode=0o600,
                        read=True,
                    )
                    input_row = report_value.inputs[slot_index]
                    if manifest_row["sha256"] != (
                        input_row["output_commitment"]
                    ):
                        raise MsctlError(
                            "COHORT_COLLECT_OBJECT_MISMATCH",
                            "output manifest differs from the report "
                            "commitment",
                            details={"uri": manifest_row["uri"]},
                        )
                    try:
                        manifest_value = _strict_cohort_json(
                            manifest_bytes,
                            label="output manifest",
                        )
                    except CohortCollectionError as error:
                        raise MsctlError(
                            "COHORT_COLLECT_OBJECT_MISMATCH",
                            f"output manifest is invalid: {error}",
                            details={"uri": manifest_row["uri"]},
                        ) from error
                    artifacts = manifest_value.get("artifacts")
                    declared: dict[str, tuple[object, object]] = {}
                    if isinstance(artifacts, list):
                        for artifact_row in artifacts:
                            if not isinstance(
                                artifact_row,
                                Mapping,
                            ) or set(artifact_row) != {
                                "path",
                                "sha256",
                                "bytes",
                            }:
                                declared = {}
                                break
                            declared[str(artifact_row["path"])] = (
                                artifact_row["sha256"],
                                artifact_row["bytes"],
                            )
                    expected_rows = {
                        name: (
                            members[name]["sha256"],
                            members[name]["bytes"],
                        )
                        for name in EVALUATION_OUTPUT_MEMBERS
                        if name != "output.json"
                    }
                    if declared != expected_rows:
                        raise MsctlError(
                            "COHORT_COLLECT_OBJECT_MISMATCH",
                            "output artifact rows differ from the evidence "
                            "index",
                            details={"uri": manifest_row["uri"]},
                        )
                    if (
                        manifest_value.get("output_id") != output_id
                        or input_row["output_id"] != output_id
                        or manifest_value.get("study_lock_sha256")
                        != report_value.study_lock_sha256
                        or manifest_value.get(
                            "sealed_evaluation_release_sha256"
                        )
                        != report_value.sealed_evaluation_release_sha256
                        or manifest_value.get("provider_selection_sha256")
                        != binding.provider_selection_sha256
                        or manifest_value.get(
                            "provider_selection_s3_version_id"
                        )
                        != binding.provider_selection_version_id
                        or manifest_value.get("seed") != output["seed"]
                        or manifest_value.get("arm") != output["arm"]
                        or manifest_value.get("optimizer_step")
                        != output["optimizer_step"]
                    ):
                        raise MsctlError(
                            "COHORT_COLLECT_OBJECT_MISMATCH",
                            "output manifest lock, release, selection, or "
                            "slot identity is crossed",
                            details={"uri": manifest_row["uri"]},
                        )
                    for member in EVALUATION_OUTPUT_MEMBERS:
                        if member == "output.json":
                            continue
                        self._fetch_cohort_evidence(
                            staging_fd,
                            staging_path,
                            f"outputs/{output_id}/{member}",
                            row=members[member],
                            kind=f"{member} artifact",
                            mode=0o600,
                            read=False,
                        )
                # 8) Outcome replay: the complete Task 4C recomputation is
                #    the only outcome authority.
                from evals.confirmatory.aggregate import (
                    CollectionReceiptEvidence,
                    SnapshotOutputReference,
                    validate_cohort_report,
                )

                try:
                    references = tuple(
                        SnapshotOutputReference(
                            output_dir=(
                                staging_path / "outputs" / output_id
                            ),
                            output_commitment=(
                                report_value.inputs[slot_index][
                                    "output_commitment"
                                ]
                            ),
                        )
                        for slot_index, output_id in enumerate(output_ids)
                    )
                    receipt_evidence = tuple(
                        CollectionReceiptEvidence(
                            payload=seed_payloads[seed_row["seed"]],
                            uri=seed_row["uri"],
                            sha256=seed_row["sha256"],
                            version_id=seed_row["version_id"],
                        )
                        for seed_row in index["seed_collections"]
                    )
                    validate_cohort_report(
                        report_value,
                        outputs=references,
                        collection_receipts=receipt_evidence,
                        expected_study_lock_sha256=(
                            report_value.study_lock_sha256
                        ),
                    )
                except (TypeError, ValueError) as error:
                    raise MsctlError(
                        "COHORT_COLLECT_REPLAY_FAILED",
                        f"cohort report replay diverged: {error}",
                    ) from error
                # 9) Build and self-parse the canonical cohort collection
                #    receipt before any publication.
                binding_values = binding.to_dict()
                collection_value: dict[str, object] = {
                    receipt_field: binding_values[binding_field]
                    for receipt_field, binding_field in (
                        _RECEIPT_BINDING_FIELDS.items()
                    )
                }
                objects: list[dict[str, object]] = []
                for output in index["outputs"]:
                    for member_row in output["members"]:
                        objects.append(
                            {
                                "arm": output["arm"],
                                "bytes": member_row["bytes"],
                                "kind": "output_member",
                                "member": member_row["member"],
                                "seed": output["seed"],
                                "sha256": member_row["sha256"],
                                "step": output["optimizer_step"],
                                "uri": member_row["uri"],
                                "version_id": member_row["version_id"],
                            }
                        )
                for seed_row in index["seed_collections"]:
                    objects.append(
                        {
                            "arm": None,
                            "bytes": seed_row["bytes"],
                            "kind": "seed_collection",
                            "member": None,
                            "seed": seed_row["seed"],
                            "sha256": seed_row["sha256"],
                            "step": None,
                            "uri": seed_row["uri"],
                            "version_id": seed_row["version_id"],
                        }
                    )
                for kind, reference in (
                    ("study_lock", lock_row),
                    ("cohort_report", report_row),
                ):
                    objects.append(
                        {
                            "arm": None,
                            "bytes": reference["bytes"],
                            "kind": kind,
                            "member": None,
                            "seed": None,
                            "sha256": reference["sha256"],
                            "step": None,
                            "uri": reference["uri"],
                            "version_id": reference["version_id"],
                        }
                    )
                if len(objects) != COHORT_COLLECTION_OBJECT_COUNT:
                    raise MsctlError(
                        "COHORT_COLLECT_INCOMPLETE",
                        "cohort collection did not enumerate all 1012 "
                        "objects",
                    )
                bytes_collected = sum(
                    int(row["bytes"]) for row in objects
                )
                collection_value.update(
                    {
                        "receipt_type": COHORT_COLLECTION_RECEIPT_TYPE,
                        "schema_version": 3,
                        "complete": True,
                        "request_id": request_id,
                        "collected_at": datetime.now(UTC).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "dataset_build_id": manifest.dataset_build_id,
                        "dataset_receipt_sha256": (
                            manifest.dataset_receipt_sha256
                        ),
                        "ordered_stream_sha256": (
                            manifest.ordered_stream_sha256
                        ),
                        "release_sha256": manifest.release_sha256,
                        "release_receipt_sha256": (
                            manifest.release_receipt_sha256
                        ),
                        "run_manifest_sha256": manifest.sha256,
                        "source_commit": manifest.source_commit,
                        "source_tree": manifest.source_tree,
                        "sealed_evaluation_release_sha256": (
                            manifest.sealed_evaluation_release_sha256
                        ),
                        "preregistration_sha256": (
                            manifest.preregistration_sha256
                        ),
                        "study_lock": dict(lock_row),
                        "cohort_report": dict(report_row),
                        "seed_collections": [
                            dict(seed_row)
                            for seed_row in index["seed_collections"]
                        ],
                        "objects": objects,
                    }
                )
                collection_bytes = _canonical_json(collection_value)
                collection_sha256 = hashlib.sha256(
                    collection_bytes
                ).hexdigest()
                collection_uri = f"{self.runtime.s3_root}/" + (
                    cohort_collection_receipt_key(collection_sha256)
                )
                try:
                    parse_cohort_collection_receipt_bytes(
                        collection_bytes,
                        receipt_uri=collection_uri,
                        receipt_sha256=collection_sha256,
                        receipt_version_id="unpublished",
                        expected_binding=binding,
                    )
                except CohortCollectionError as error:
                    raise MsctlError(
                        "COHORT_COLLECT_EVIDENCE_INVALID",
                        f"built cohort collection receipt failed "
                        f"self-parse: {error}",
                    ) from error
                receipt_relative = cohort_collection_receipt_key(
                    collection_sha256
                )
                self._stage_collection_bytes(
                    staging_fd,
                    receipt_relative,
                    collection_bytes,
                )
                # 10) Publish no-replace, checksum- and metadata-bound,
                #     then verify by exact HEAD.
                version_id = self._publish_cohort_collection_receipt(
                    payload=collection_bytes,
                    digest=collection_sha256,
                    request_id=request_id,
                    body_path=str(staging_path / receipt_relative),
                )
                collection_reference = {
                    "uri": collection_uri,
                    "sha256": collection_sha256,
                    "version_id": version_id,
                    "bytes": len(collection_bytes),
                }
                # 11) Atomically write the singleton cohort state, then
                #     rename the staging tree no-replace.
                now = _timestamp()
                store.write_cohort_collection(
                    {
                        **binding_values,
                        "schema_version": 2,
                        "operation": "collect-cohort",
                        "run_manifest_sha256": manifest.sha256,
                        "release_sha256": manifest.release_sha256,
                        "release_receipt_sha256": (
                            manifest.release_receipt_sha256
                        ),
                        "sealed_evaluation_release_sha256": (
                            manifest.sealed_evaluation_release_sha256
                        ),
                        "preregistration_sha256": (
                            manifest.preregistration_sha256
                        ),
                        "study_lock": dict(lock_row),
                        "cohort_report": dict(report_row),
                        "cohort_collection_receipt": dict(
                            collection_reference
                        ),
                        "objects_collected": (
                            COHORT_COLLECTION_OBJECT_COUNT
                        ),
                        "bytes_collected": bytes_collected,
                        "status": "Published",
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                try:
                    rename_noreplace_at(
                        parent_fd,
                        staging_name,
                        parent_fd,
                        destination.name,
                    )
                except FileExistsError as error:
                    raise MsctlError(
                        "COHORT_COLLECT_DESTINATION_EXISTS",
                        "refusing to replace an existing cohort collection "
                        "destination",
                    ) from error
                published = True
            finally:
                if staging_fd is not None:
                    os.close(staging_fd)
                if not published:
                    try:
                        remove_tree_at(
                            parent_fd,
                            staging_name,
                            label="cohort collection staging",
                        )
                    except MsctlError:
                        pass
                    except FileNotFoundError:
                        pass
                os.close(parent_fd)
            # 12) Report the durable identities.
            return {
                "provider": self.profile.provider,
                "seed": manifest.seed,
                "run_manifest_sha256": manifest.sha256,
                "study_lock": dict(lock_row),
                "cohort_report": dict(report_row),
                "cohort_collection_receipt": dict(collection_reference),
                "objects_collected": COHORT_COLLECTION_OBJECT_COUNT,
                "bytes_collected": bytes_collected,
                "out": str(destination),
                "collected": COHORT_COLLECTION_OBJECT_COUNT,
                "idempotent": False,
            }

    def _selected_prior_run_receipt(
        self,
        manifest: object,
        prior_run_receipt: Mapping[str, object] | None,
    ) -> PriorRunReceiptRef | None:
        """Validate the operator prior-receipt triple before any AWS call."""

        if manifest.seed == 0:
            if prior_run_receipt is not None:
                raise MsctlError(
                    "SEED_TRANSITION_BLOCKED",
                    "seed 0 forbids a prior-seed finalization receipt",
                )
            return None
        if (
            not isinstance(prior_run_receipt, Mapping)
            or set(prior_run_receipt) != {"uri", "sha256", "version_id"}
        ):
            raise MsctlError(
                "SEED_TRANSITION_BLOCKED",
                "seeds 1 through 9 require the exact prior receipt triple",
            )
        try:
            return PriorRunReceiptRef(
                uri=prior_run_receipt["uri"],
                sha256=prior_run_receipt["sha256"],
                version_id=prior_run_receipt["version_id"],
            )
        except SeedTransitionError as error:
            raise MsctlError(
                "SEED_TRANSITION_BLOCKED",
                f"prior receipt triple is invalid: {error}",
            ) from error

    def _require_sequential_seed_transition(
        self,
        store: StateStore,
        manifest: object,
    ) -> None:
        """Enforce one active pair and strictly increasing seeds."""

        for manifest_sha256, journal in sorted(
            store.read_all_aws_pairs().items()
        ):
            if manifest_sha256 == manifest.sha256:
                continue
            states = [
                state
                for state in journal["states"]
                if isinstance(state, dict)
            ]
            if any(
                state.get("status") not in _TERMINAL_COMMAND_STATES
                for state in states
            ):
                raise MsctlError(
                    "SEQUENTIAL_PAIR_ACTIVE",
                    "another AWS pair is still active on this controller",
                    details={"manifest_sha256": manifest_sha256},
                )
            if any(
                isinstance(state.get("seed"), int)
                and not isinstance(state.get("seed"), bool)
                and state["seed"] >= manifest.seed
                for state in states
            ):
                raise MsctlError(
                    "SEED_TRANSITION_BLOCKED",
                    "a later or equal seed already holds durable pair state",
                    details={"manifest_sha256": manifest_sha256},
                )

    def _admit_selected_prior_finalization(
        self,
        manifest: object,
        ref: PriorRunReceiptRef,
    ) -> object:
        """GET and authenticate the prior receipt, then HEAD all 14 objects."""

        prefix = self.runtime.s3_root.rstrip("/") + "/"
        if not ref.uri.startswith(prefix):
            raise MsctlError(
                "SEED_TRANSITION_BLOCKED",
                "prior receipt is outside the pinned S3 root",
            )
        bucket, key = self._s3_location(ref.uri.removeprefix(prefix))
        expected_checksum = base64.b64encode(
            bytes.fromhex(ref.sha256)
        ).decode("ascii")
        with tempfile.TemporaryDirectory(
            prefix="memorysplit-prior-receipt-",
            dir=Path(tempfile.gettempdir()).resolve(),
        ) as temporary:
            destination = Path(temporary) / "receipt.json"
            try:
                output = _aws_output_object(
                    self._run(
                        self._aws_argv(
                            "s3api",
                            "get-object",
                            "--bucket",
                            bucket,
                            "--key",
                            key,
                            "--version-id",
                            ref.version_id,
                            "--checksum-mode",
                            "ENABLED",
                            str(destination),
                            query=(
                                "{receipt:{checksum_sha256:ChecksumSHA256,"
                                "version_id:VersionId}}"
                            ),
                        ),
                        operation="fetch prior run finalization",
                    ),
                    {"receipt"},
                    label="prior run receipt download",
                )
            except MsctlError as error:
                if error.code == "AWS_COMMAND_FAILED":
                    raise MsctlError(
                        "SEED_TRANSITION_BLOCKED",
                        "prior run finalization receipt is unavailable",
                    ) from error
                raise
            row = _aws_output_object(
                output["receipt"],
                {"checksum_sha256", "version_id"},
                label="prior run receipt object",
            )
            try:
                payload = read_regular_input(
                    destination,
                    label="prior run receipt",
                    maximum_bytes=16 * 1024 * 1024,
                )
            except (AttestationError, OSError) as error:
                raise MsctlError(
                    "SEED_TRANSITION_BLOCKED",
                    "downloaded prior run receipt is unsafe",
                ) from error
        if (
            row["checksum_sha256"] != expected_checksum
            or row["version_id"] != ref.version_id
            or hashlib.sha256(payload).hexdigest() != ref.sha256
        ):
            raise MsctlError(
                "SEED_TRANSITION_BLOCKED",
                "downloaded prior run receipt identity differs",
            )
        try:
            admitted = admit_prior_seed_finalization(
                payload,
                ref=ref,
                binding=self.lifecycle_binding,
                release_sha256=manifest.release_sha256,
                release_receipt_sha256=manifest.release_receipt_sha256,
                dataset_receipt_sha256=manifest.dataset_receipt_sha256,
                dataset_build_id=manifest.dataset_build_id,
                ordered_stream_sha256=manifest.ordered_stream_sha256,
                source_commit=manifest.source_commit,
                source_tree=manifest.source_tree,
                instance_id=manifest.instance_id,
            )
        except SeedTransitionError as error:
            raise MsctlError(
                "SEED_TRANSITION_BLOCKED",
                str(error),
            ) from error
        for item in admitted.evidence:
            if not item.uri.startswith(prefix):
                raise MsctlError(
                    "SEED_TRANSITION_BLOCKED",
                    "prior evidence object is outside the pinned S3 root",
                )
            bucket, key = self._s3_location(item.uri.removeprefix(prefix))
            try:
                head_output = _aws_output_object(
                    self._run(
                        self._aws_argv(
                            "s3api",
                            "head-object",
                            "--bucket",
                            bucket,
                            "--key",
                            key,
                            "--version-id",
                            item.version_id,
                            "--checksum-mode",
                            "ENABLED",
                            query=(
                                "{object:{checksum_sha256:ChecksumSHA256,"
                                "content_length:ContentLength,"
                                "version_id:VersionId}}"
                            ),
                        ),
                        operation="verify prior evidence object",
                    ),
                    {"object"},
                    label="prior evidence head",
                )
            except MsctlError as error:
                if error.code == "AWS_COMMAND_FAILED":
                    raise MsctlError(
                        "SEED_TRANSITION_BLOCKED",
                        "prior evidence object is unavailable",
                    ) from error
                raise
            head = _aws_output_object(
                head_output["object"],
                {"checksum_sha256", "content_length", "version_id"},
                label="prior evidence object",
            )
            if (
                head["checksum_sha256"]
                != base64.b64encode(bytes.fromhex(item.sha256)).decode(
                    "ascii"
                )
                or (
                    item.bytes is not None
                    and head["content_length"] != item.bytes
                )
                or head["version_id"] != item.version_id
            ):
                raise MsctlError(
                    "SEED_TRANSITION_BLOCKED",
                    "prior evidence object version or checksum differs",
                    details={"uri": item.uri},
                )
        return admitted

    def _selected_prior_collection_receipt(
        self,
        manifest: object,
        prior_collection_receipt: Mapping[str, object] | None,
    ) -> CollectionReceiptRef | None:
        """Validate the operator collection triple before any AWS call."""

        if manifest.seed == 0:
            if prior_collection_receipt is not None:
                raise MsctlError(
                    "SEED_TRANSITION_BLOCKED",
                    "seed 0 forbids a prior-seed collection receipt",
                )
            return None
        if (
            not isinstance(prior_collection_receipt, Mapping)
            or set(prior_collection_receipt)
            != {"uri", "sha256", "version_id"}
        ):
            raise MsctlError(
                "SEED_TRANSITION_BLOCKED",
                "seeds 1 through 9 require the exact prior collection "
                "receipt triple",
            )
        try:
            return CollectionReceiptRef(
                uri=prior_collection_receipt["uri"],
                sha256=prior_collection_receipt["sha256"],
                version_id=prior_collection_receipt["version_id"],
            )
        except CollectionError as error:
            raise MsctlError(
                "SEED_TRANSITION_BLOCKED",
                f"prior collection receipt triple is invalid: {error}",
            ) from error

    def _admit_selected_prior_collection(
        self,
        ref: CollectionReceiptRef,
        admitted: object,
    ) -> None:
        """GET and authenticate the prior collection, then HEAD both
        checkpoints."""

        prefix = self.runtime.s3_root.rstrip("/") + "/"
        if not ref.uri.startswith(prefix):
            raise MsctlError(
                "SEED_TRANSITION_BLOCKED",
                "prior collection receipt is outside the pinned S3 root",
            )
        bucket, key = self._s3_location(ref.uri.removeprefix(prefix))
        expected_checksum = base64.b64encode(
            bytes.fromhex(ref.sha256)
        ).decode("ascii")
        with tempfile.TemporaryDirectory(
            prefix="memorysplit-prior-collection-",
            dir=Path(tempfile.gettempdir()).resolve(),
        ) as temporary:
            destination = Path(temporary) / "collection.json"
            try:
                output = _aws_output_object(
                    self._run(
                        self._aws_argv(
                            "s3api",
                            "get-object",
                            "--bucket",
                            bucket,
                            "--key",
                            key,
                            "--version-id",
                            ref.version_id,
                            "--checksum-mode",
                            "ENABLED",
                            str(destination),
                            query=(
                                "{receipt:{checksum_sha256:ChecksumSHA256,"
                                "version_id:VersionId}}"
                            ),
                        ),
                        operation="fetch prior seed collection",
                    ),
                    {"receipt"},
                    label="prior collection receipt download",
                )
            except MsctlError as error:
                if error.code == "AWS_COMMAND_FAILED":
                    raise MsctlError(
                        "SEED_TRANSITION_BLOCKED",
                        "prior seed collection receipt is unavailable",
                    ) from error
                raise
            row = _aws_output_object(
                output["receipt"],
                {"checksum_sha256", "version_id"},
                label="prior collection receipt object",
            )
            try:
                payload = read_regular_input(
                    destination,
                    label="prior collection receipt",
                    maximum_bytes=16 * 1024 * 1024,
                )
            except (AttestationError, OSError) as error:
                raise MsctlError(
                    "SEED_TRANSITION_BLOCKED",
                    "downloaded prior collection receipt is unsafe",
                ) from error
        if (
            row["checksum_sha256"] != expected_checksum
            or row["version_id"] != ref.version_id
            or hashlib.sha256(payload).hexdigest() != ref.sha256
        ):
            raise MsctlError(
                "SEED_TRANSITION_BLOCKED",
                "downloaded prior collection receipt identity differs",
            )
        try:
            admitted_collection = admit_prior_seed_collection(
                payload,
                ref=ref,
                admitted=admitted,
                binding=self.lifecycle_binding,
            )
        except CollectionError as error:
            raise MsctlError(
                "SEED_TRANSITION_BLOCKED",
                str(error),
            ) from error
        for checkpoint in admitted_collection.checkpoints:
            if not checkpoint.uri.startswith(prefix):
                raise MsctlError(
                    "SEED_TRANSITION_BLOCKED",
                    "prior collection checkpoint is outside the pinned "
                    "S3 root",
                )
            bucket, key = self._s3_location(
                checkpoint.uri.removeprefix(prefix)
            )
            try:
                head_output = _aws_output_object(
                    self._run(
                        self._aws_argv(
                            "s3api",
                            "head-object",
                            "--bucket",
                            bucket,
                            "--key",
                            key,
                            "--version-id",
                            checkpoint.version_id,
                            "--checksum-mode",
                            "ENABLED",
                            query=(
                                "{object:{checksum_sha256:ChecksumSHA256,"
                                "content_length:ContentLength,"
                                "version_id:VersionId}}"
                            ),
                        ),
                        operation="verify prior collection checkpoint",
                    ),
                    {"object"},
                    label="prior collection checkpoint head",
                )
            except MsctlError as error:
                if error.code == "AWS_COMMAND_FAILED":
                    raise MsctlError(
                        "SEED_TRANSITION_BLOCKED",
                        "prior collection checkpoint is unavailable",
                    ) from error
                raise
            head = _aws_output_object(
                head_output["object"],
                {"checksum_sha256", "content_length", "version_id"},
                label="prior collection checkpoint object",
            )
            if (
                head["checksum_sha256"]
                != base64.b64encode(
                    bytes.fromhex(checkpoint.sha256)
                ).decode("ascii")
                or head["content_length"] != checkpoint.bytes
                or head["version_id"] != checkpoint.version_id
            ):
                raise MsctlError(
                    "SEED_TRANSITION_BLOCKED",
                    "prior collection checkpoint version or checksum "
                    "differs",
                    details={"uri": checkpoint.uri},
                )

    def _resolve_selected_bootstrap_mode(
        self,
        release: object,
        manifest: object,
        *,
        boot_id: str,
    ) -> tuple[str, str | None]:
        """Resolve bootstrap-once-per-boot from the durable S3 receipt."""

        bucket, key = self._s3_location(
            bootstrap_receipt_key(manifest.instance_id)
        )
        with tempfile.TemporaryDirectory(
            prefix="memorysplit-bootstrap-receipt-",
            dir=Path(tempfile.gettempdir()).resolve(),
        ) as temporary:
            destination = Path(temporary) / "bootstrap-receipt.json"
            try:
                self._run(
                    self._aws_argv(
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
                            "{receipt:{content_length:ContentLength,"
                            "version_id:VersionId}}"
                        ),
                    ),
                    operation="resolve bootstrap receipt",
                )
            except MsctlError as error:
                if error.code == "AWS_COMMAND_FAILED":
                    return "bootstrap", None
                raise
            try:
                payload = read_regular_input(
                    destination,
                    label="durable bootstrap receipt",
                    maximum_bytes=1024 * 1024,
                )
            except (AttestationError, OSError) as error:
                raise MsctlError(
                    "BOOTSTRAP_REUSE_INVALID",
                    "downloaded bootstrap receipt is unsafe",
                ) from error
        try:
            value = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=_strict_json_object,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON value: {constant}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise MsctlError(
                "BOOTSTRAP_REUSE_INVALID",
                "durable bootstrap receipt is not valid JSON",
            ) from error
        if (
            not isinstance(value, dict)
            or set(value) != set(_BOOTSTRAP_RECEIPT_FIELDS)
        ):
            raise MsctlError(
                "BOOTSTRAP_REUSE_INVALID",
                "durable bootstrap receipt fields do not match the contract",
            )
        expected = {
            "schema_version": 2,
            "receipt_type": "aws-p5-bootstrap",
            "instance_id": manifest.instance_id,
            "profile_sha256": self.profile.sha256,
            "provider": self.profile.provider,
            "instance_type": self.profile.instance_type,
            "scratch_root": self.profile.scratch_root,
            "region": self.runtime.region,
            "ami_id": self.runtime.ami_id,
            "container_image": self.runtime.container_image,
            "container_digest": self.runtime.container_digest,
            "runtime_uid": getattr(self.runtime, "uid", 1000),
            "runtime_gid": getattr(self.runtime, "gid", 1000),
            "release_sha256": manifest.release_sha256,
            "release_members_sha256": getattr(
                release,
                "members_sha256",
                None,
            ),
            "release_root": f"releases/{manifest.release_sha256}",
            "cohort_assignment_sha256": manifest.cohort_assignment_sha256,
            "code_commit": manifest.source_commit,
            "corpus_receipt_sha256": manifest.dataset_receipt_sha256,
            "corpus_build_id": manifest.dataset_build_id,
            "corpus_ordered_stream_sha256": manifest.ordered_stream_sha256,
            "durable_upload_verified": True,
        }
        mismatched = sorted(
            field
            for field, expected_value in expected.items()
            if value.get(field) != expected_value
        )
        if mismatched:
            raise MsctlError(
                "BOOTSTRAP_REUSE_INVALID",
                "durable bootstrap receipt does not match this cohort",
                details={"fields": mismatched},
            )
        if (
            not isinstance(value["boot_id"], str)
            or _BOOT_ID_RE.fullmatch(value["boot_id"]) is None
        ):
            raise MsctlError(
                "BOOTSTRAP_REUSE_INVALID",
                "durable bootstrap receipt boot identity is invalid",
            )
        if value["boot_id"] != boot_id:
            return "bootstrap", None
        return "reuse", hashlib.sha256(payload).hexdigest()

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
        checkpoint_receipt: object | None = None,
        checkpoints: Mapping[str, object] | None = None,
        prior_command_ids: list[str] | None = None,
        bootstrap_mode: str | None = None,
        bootstrap_receipt_sha256: str | None = None,
        prior_run_receipt: Mapping[str, object] | None = None,
        prior_collection_receipt: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        now = _timestamp()
        is_v3 = getattr(manifest, "schema_version", None) == 3
        if is_v3:
            selected_lifecycle = (
                getattr(
                    manifest,
                    "provider_selection_sha256",
                    None,
                )
                is not None
            )
            state = {
                **(
                    {
                        field: (
                            list(manifest.arms)
                            if field == "arms"
                            else getattr(manifest, field)
                        )
                        for field in LIFECYCLE_BINDING_FIELDS
                    }
                    if selected_lifecycle
                    else {}
                ),
                **(
                    {
                        "bootstrap_mode": bootstrap_mode,
                        "bootstrap_receipt_sha256": (
                            bootstrap_receipt_sha256
                        ),
                        "prior_run_receipt": (
                            dict(prior_run_receipt)
                            if prior_run_receipt is not None
                            else None
                        ),
                        "prior_collection_receipt": (
                            dict(prior_collection_receipt)
                            if prior_collection_receipt is not None
                            else None
                        ),
                    }
                    if selected_lifecycle
                    else {}
                ),
                "schema_version": 2,
                "run_id": run.run_id,
                "arm": run.arm,
                "seed": run.seed,
                "provider": getattr(
                    manifest,
                    "provider",
                    self.profile.provider,
                ),
                "release_sha256": manifest.release_sha256,
                "release_receipt_sha256": (
                    manifest.release_receipt_sha256
                ),
                "run_manifest_sha256": manifest.sha256,
                "config_sha256": run.config_sha256,
                "dataset_pointer_sha256": (
                    manifest.dataset_pointer_sha256
                ),
                "dataset_receipt_sha256": (
                    manifest.dataset_receipt_sha256
                ),
                "dataset_build_id": manifest.dataset_build_id,
                "ordered_stream_sha256": (
                    manifest.ordered_stream_sha256
                ),
                "dataset_verification_sha256": intent[
                    "dataset_verification_sha256"
                ],
                "environment_receipt_sha256": intent[
                    "environment_receipt_sha256"
                ],
                "cohort_assignment_sha256": (
                    manifest.cohort_assignment_sha256
                ),
                "preregistration_sha256": (
                    manifest.preregistration_sha256
                ),
                "source_commit": manifest.source_commit,
                "source_tree": manifest.source_tree,
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
                if checkpoint_receipt is None or checkpoints is None:
                    raise MsctlError(
                        "STATE_INCOMPLETE",
                        "v3 resume state requires versioned checkpoint bindings",
                    )
                state["checkpoint_receipt"] = {
                    "sha256": checkpoint_receipt.sha256,
                    "uri": checkpoint_receipt.uri,
                    "version_id": checkpoint_receipt.version_id,
                }
                state["checkpoint_objects"] = [
                    {
                        "arm": arm,
                        "bytes": checkpoints[arm].object.bytes,
                        "sha256": checkpoints[arm].object.sha256,
                        "uri": checkpoints[arm].object.uri,
                        "version_id": checkpoints[arm].object.version_id,
                    }
                    for arm in ("dense", "split90")
                ]
                state["prior_command_ids"] = list(
                    prior_command_ids or []
                )
            return state
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

    def _canonical_paired_state_inputs(
        self,
        manifest: object,
        states: Sequence[dict[str, object]],
    ) -> list[dict[str, object]]:
        manifest_runs = tuple(getattr(manifest, "runs", ()))
        supplied = tuple(states)
        manifest_run_ids = [
            getattr(run, "run_id", None) for run in manifest_runs
        ]
        manifest_arms = [getattr(run, "arm", None) for run in manifest_runs]
        supplied_run_ids = [
            state.get("run_id") if isinstance(state, dict) else None
            for state in supplied
        ]
        supplied_arms = [
            state.get("arm") if isinstance(state, dict) else None
            for state in supplied
        ]
        if (
            getattr(manifest, "provider", None) != self.profile.provider
            or len(manifest_runs) != 2
            or any(not isinstance(run_id, str) for run_id in manifest_run_ids)
            or len(set(manifest_run_ids)) != 2
            or manifest_arms.count("dense") != 1
            or manifest_arms.count("split90") != 1
            or len(supplied) != 2
            or any(not isinstance(state, dict) for state in supplied)
            or any(not isinstance(run_id, str) for run_id in supplied_run_ids)
            or len(set(supplied_run_ids)) != 2
            or supplied_arms.count("dense") != 1
            or supplied_arms.count("split90") != 1
            or set(supplied_run_ids) != set(manifest_run_ids)
        ):
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS pair update must bind exactly two unique manifest runs",
            )
        state_by_id = {
            str(state["run_id"]): state for state in supplied
        }
        if not all(
            self._same_state_binding(state, manifest)
            for state in state_by_id.values()
        ):
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS pair update has invalid manifest provenance",
            )
        return [
            dict(state_by_id[str(run_id)]) for run_id in manifest_run_ids
        ]

    def _write_paired_states(
        self,
        store: StateStore,
        manifest: object,
        states: Sequence[dict[str, object]],
    ) -> None:
        ordered_states = self._canonical_paired_state_inputs(
            manifest,
            states,
        )
        operation_ids = {state.get("operation_id") for state in ordered_states}
        if (
            len(operation_ids) != 1
            or not isinstance(next(iter(operation_ids)), str)
        ):
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS pair update must bind one operation",
            )
        journal = {
            "schema_version": (
                2 if getattr(manifest, "schema_version", None) == 3 else 1
            ),
            "provider": getattr(
                manifest,
                "provider",
                self.profile.provider,
            ),
            "run_manifest_sha256": manifest.sha256,
            "operation_id": next(iter(operation_ids)),
            "states": ordered_states,
        }
        if getattr(manifest, "schema_version", None) == 3:
            store.write_aws_pair_transaction(
                manifest.sha256,
                journal,
                ordered_states,
            )
            return
        store.write_aws_pair(manifest.sha256, journal)
        for state in ordered_states:
            store.write_run(str(state["run_id"]), state)

    def _refresh_paired_states(
        self,
        store: StateStore,
        manifest: object,
        states: Sequence[dict[str, object]],
        updates: Mapping[str, object],
    ) -> list[dict[str, object]]:
        """Validate and uniformly refresh one already-durable run pair."""

        if set(updates) - {"command_id", "send_attempted", "status"}:
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS pair refresh contains unsupported fields",
            )
        canonical_states = self._canonical_paired_state_inputs(
            manifest,
            states,
        )
        state_by_id = {
            str(state["run_id"]): state for state in canonical_states
        }
        manifest_run_ids = tuple(state_by_id)
        current = {
            run_id: store.read_run(run_id)
            for run_id in manifest_run_ids
        }
        if any(
            current[run_id] is None
            or current[run_id] != state_by_id[run_id]
            for run_id in manifest_run_ids
        ):
            raise MsctlError(
                "STATE_CORRUPT",
                "AWS pair refresh input differs from durable run state",
            )
        journal = store.read_aws_pair(manifest.sha256)
        if journal is not None:
            journal_by_id = {
                str(state["run_id"]): state for state in journal["states"]
            }
            if journal_by_id != state_by_id:
                raise MsctlError(
                    "STATE_CORRUPT",
                    "AWS pair refresh conflicts with its durable journal",
                )
        elif getattr(manifest, "schema_version", None) == 3:
            raise MsctlError(
                "STATE_INCOMPLETE",
                "AWS v3 pair refresh requires its durable journal",
            )

        if all(
            all(state.get(field) == value for field, value in updates.items())
            for state in state_by_id.values()
        ):
            return canonical_states

        now = _timestamp()
        refreshed = []
        for state in canonical_states:
            next_state = dict(state)
            next_state.update(updates)
            next_state["updated_at"] = now
            refreshed.append(next_state)
        if journal is not None or getattr(manifest, "schema_version", None) == 3:
            self._write_paired_states(store, manifest, refreshed)
        else:
            for state in refreshed:
                store.write_run(str(state["run_id"]), state)
        return refreshed

    def _repair_paired_states(
        self,
        store: StateStore,
        manifest: object,
    ) -> None:
        journal = store.read_aws_pair(manifest.sha256)
        if journal is None:
            if (
                getattr(manifest, "schema_version", None) == 3
                and any(
                    store.read_run(run.run_id) is not None
                    for run in manifest.runs
                )
            ):
                raise MsctlError(
                    "STATE_INCOMPLETE",
                    "AWS v3 run files exist without their pair journal",
                )
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
                if getattr(manifest, "schema_version", None) == 3:
                    raise MsctlError(
                        "STATE_INCOMPLETE",
                        "v3 paired state is partial and cannot be repaired",
                    )
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
        bootstrap_mode: str | None = None,
        prior_run_receipt: Mapping[str, object] | None = None,
        prior_collection_receipt: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        self._validate_manifest(manifest)
        self._validate_release(release, manifest)
        selected = self._selected_manifest_lifecycle(manifest)
        if not selected and (
            bootstrap_mode is not None
            or prior_run_receipt is not None
            or prior_collection_receipt is not None
        ):
            raise MsctlError(
                "CLI_USAGE",
                "bootstrap modes and prior receipts are selected-manifest "
                "arguments",
            )
        self._validate_submit_selection(instance_id, terminate_at)
        prior_ref: PriorRunReceiptRef | None = None
        prior_collection_ref: CollectionReceiptRef | None = None
        if selected:
            if instance_id != manifest.instance_id:
                raise MsctlError(
                    "INSTANCE_BINDING_MISMATCH",
                    "selected submit must target the authenticated cohort "
                    "instance",
                )
            if bootstrap_mode not in {"bootstrap", "reuse"}:
                raise MsctlError(
                    "CLI_USAGE",
                    "selected submit requires --bootstrap-mode bootstrap "
                    "or reuse",
                )
            if evidence is None:
                raise MsctlError(
                    "LIFECYCLE_EVIDENCE_INVALID",
                    "selected submit requires complete lifecycle evidence",
                )
            prior_ref = self._selected_prior_run_receipt(
                manifest,
                prior_run_receipt,
            )
            prior_collection_ref = self._selected_prior_collection_receipt(
                manifest,
                prior_collection_receipt,
            )
        core: dict[str, object] | None = None
        operation_intent: dict[str, object] | None = None
        if not selected:
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
            if selected:
                return {
                    "provider": manifest.provider,
                    "seed": manifest.seed,
                    "release_sha256": manifest.release_sha256,
                    "run_manifest_sha256": manifest.sha256,
                    "instance_id": instance_id,
                    "terminate_at": terminate_at,
                    "bootstrap_mode": bootstrap_mode,
                    "prior_run_receipt": (
                        dict(prior_run_receipt)
                        if prior_run_receipt is not None
                        else None
                    ),
                    "prior_collection_receipt": (
                        dict(prior_collection_receipt)
                        if prior_collection_receipt is not None
                        else None
                    ),
                    "commands": [
                        self._selected_identity_discover_argv(manifest),
                        self._selected_identity_instance_argv(instance_id),
                    ],
                    "submitted": 0,
                    "idempotent": False,
                }
            plan = self._submission_plan(
                manifest,
                release=release,
                instance_id=instance_id,
                terminate_at=terminate_at,
                operation_intent=operation_intent,
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
        if selected:
            resources.update(
                {
                    "dataset_pointer_sha256": evidence[
                        "dataset_pointer_sha256"
                    ],
                    "dataset_verification_sha256": evidence[
                        "dataset_verification_sha256"
                    ],
                    "environment_receipt_sha256": evidence[
                        "environment_receipt_sha256"
                    ],
                    **{
                        field: (
                            list(manifest.arms)
                            if field == "arms"
                            else getattr(manifest, field)
                        )
                        for field in LIFECYCLE_BINDING_FIELDS
                    },
                    "bootstrap_mode": bootstrap_mode,
                    "prior_run_receipt": (
                        dict(prior_run_receipt)
                        if prior_run_receipt is not None
                        else None
                    ),
                    "prior_collection_receipt": (
                        dict(prior_collection_receipt)
                        if prior_collection_receipt is not None
                        else None
                    ),
                }
            )
        else:
            resources.update(
                {
                    field: core[field]
                    for field in (
                        "dataset_pointer_sha256",
                        "dataset_verification_sha256",
                        "environment_receipt_sha256",
                        *LIFECYCLE_BINDING_FIELDS,
                    )
                    if field in core
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
                if selected:
                    stored_modes = {
                        state.get("bootstrap_mode") for state in existing
                    }
                    stored_priors = {
                        canonical_json(state.get("prior_run_receipt"))
                        for state in existing
                    }
                    stored_collections = {
                        canonical_json(
                            state.get("prior_collection_receipt")
                        )
                        for state in existing
                    }
                    declared_prior = (
                        dict(prior_run_receipt)
                        if prior_run_receipt is not None
                        else None
                    )
                    declared_collection = (
                        dict(prior_collection_receipt)
                        if prior_collection_receipt is not None
                        else None
                    )
                    # Idempotent replay compares the exact stored receipt
                    # triples and never re-resolves the prior seed.
                    if (
                        stored_modes != {bootstrap_mode}
                        or stored_priors
                        != {canonical_json(declared_prior)}
                        or stored_collections
                        != {canonical_json(declared_collection)}
                    ):
                        raise MsctlError(
                            "BOOTSTRAP_REUSE_INVALID",
                            "paired state binds a different bootstrap or "
                            "prior receipt",
                        )
                    stored_receipt_hashes = {
                        state.get("bootstrap_receipt_sha256")
                        for state in existing
                    }
                    if len(stored_receipt_hashes) != 1:
                        raise MsctlError(
                            "STATE_INCOMPLETE",
                            "paired state binds divergent bootstrap receipts",
                        )
                    core = self._training_operation_intent(
                        operation="submit",
                        release=release,
                        manifest=manifest,
                        terminate_at=terminate_at,
                        evidence=evidence,
                        attempt=1,
                        bootstrap_mode=bootstrap_mode,
                        bootstrap_receipt_sha256=next(
                            iter(stored_receipt_hashes)
                        ),
                    )
                    operation_intent = self._operation_envelope(
                        core,
                        instance_id=instance_id,
                        terminate_at=terminate_at,
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
                                "provider": manifest.provider,
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
                    existing = self._refresh_paired_states(
                        store,
                        manifest,
                        existing,
                        {"command_id": command_id, "status": status},
                    )
                    return {
                        "provider": manifest.provider,
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
                existing = self._refresh_paired_states(
                    store,
                    manifest,
                    existing,
                    {"status": status},
                )
                return {
                    "provider": manifest.provider,
                    "seed": manifest.seed,
                    "instance_id": instance_id,
                    "command_id": command_id,
                    "status": status,
                    "submitted": 0,
                    "idempotent": True,
                    "active": status in _ACTIVE_COMMAND_STATES,
                }

            bootstrap_receipt_sha256: str | None = None
            if selected:
                self._require_sequential_seed_transition(store, manifest)
                if prior_ref is not None:
                    admitted_prior = self._admit_selected_prior_finalization(
                        manifest,
                        prior_ref,
                    )
                    self._admit_selected_prior_collection(
                        prior_collection_ref,
                        admitted_prior,
                    )
                resolved_mode, bootstrap_receipt_sha256 = (
                    self._resolve_selected_bootstrap_mode(
                        release,
                        manifest,
                        boot_id=str(evidence["boot_id"]),
                    )
                )
                if resolved_mode != bootstrap_mode:
                    raise MsctlError(
                        "BOOTSTRAP_REUSE_INVALID",
                        "declared bootstrap mode differs from the resolved "
                        "durable receipt",
                        details={
                            "declared": bootstrap_mode,
                            "resolved": resolved_mode,
                        },
                    )
                core = self._training_operation_intent(
                    operation="submit",
                    release=release,
                    manifest=manifest,
                    terminate_at=terminate_at,
                    evidence=evidence,
                    attempt=1,
                    bootstrap_mode=bootstrap_mode,
                    bootstrap_receipt_sha256=bootstrap_receipt_sha256,
                )
                operation_intent = self._operation_envelope(
                    core,
                    instance_id=instance_id,
                    terminate_at=terminate_at,
                )
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
                if not selected:
                    raise MsctlError(
                        "RUN_ALREADY_ACTIVE",
                        "an active instance already claims this seed without paired state",
                        details={"instance_id": discovered[0]["instance_id"]},
                    )
            if selected:
                self._parse_selected_identity_instance(
                    self._run(
                        self._selected_identity_instance_argv(instance_id),
                        operation="preflight selected instance",
                    ),
                    manifest,
                    instance_id=instance_id,
                    require_bound=False,
                )
            else:
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
            if selected:
                self._bind_selected_identity_instance(
                    manifest,
                    instance_id=instance_id,
                )
            else:
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
                        bootstrap_mode=bootstrap_mode,
                        bootstrap_receipt_sha256=bootstrap_receipt_sha256,
                        prior_run_receipt=prior_run_receipt,
                        prior_collection_receipt=prior_collection_receipt,
                )
                for run in manifest.runs
            ]
            self._write_paired_states(store, manifest, new_states)
            new_states = self._refresh_paired_states(
                store,
                manifest,
                new_states,
                {"send_attempted": True, "status": "SENDING"},
            )
            command_id = self._send_operation_intent(
                instance_id=instance_id,
                intent=operation_intent,
                published=published,
                operation="submit",
            )
            new_states = self._refresh_paired_states(
                store,
                manifest,
                new_states,
                {"command_id": command_id, "status": "Pending"},
            )
            return {
                "provider": manifest.provider,
                "seed": manifest.seed,
                "instance_id": instance_id,
                "command_id": command_id,
                "operation_id": operation_intent["operation_id"],
                "status": "Pending",
                "submitted": 1,
                "idempotent": False,
            }

    @staticmethod
    def _v3_checkpoint_metadata(
        receipt: object,
        checkpoint: object,
    ) -> dict[str, str]:
        # Checkpoint object metadata is content addressed and carries no
        # request_id: an identical checkpoint published under an earlier
        # request remains recoverable and verifiable by exact HEAD.
        return {
            "arm": checkpoint.arm,
            "checkpoint-version": str(checkpoint.checkpoint_version),
            "config-fingerprint": checkpoint.config_fingerprint,
            "config-sha256": checkpoint.config_sha256,
            "data-build-id": checkpoint.data.build_id,
            "data-receipt-sha256": checkpoint.data.receipt_sha256,
            "global-cursor": str(checkpoint.data.global_cursor),
            "ordered-stream-sha256": (
                checkpoint.data.ordered_stream_sha256
            ),
            "provider": receipt.provider,
            "profile-id": receipt.profile_id,
            "profile-sha256": receipt.profile_sha256,
            "hardware-amendment-sha256": (
                receipt.hardware_amendment_sha256
            ),
            "provider-selection-sha256": (
                receipt.provider_selection_sha256
            ),
            "provider-selection-version-id": (
                receipt.provider_selection_version_id
            ),
            "runtime-lock-sha256": receipt.runtime_lock_sha256,
            "runtime-sbom-sha256": receipt.runtime_sbom_sha256,
            "qualification-evidence-sha256": (
                receipt.qualification_evidence_sha256
            ),
            "qualification-environment-sha256": (
                receipt.environment_receipt_sha256
            ),
            "qualification-canary-sha256": (
                receipt.qualification_canary_receipt_sha256
            ),
            "qualification-approval-sha256": (
                receipt.qualification_approval_receipt_sha256
            ),
            "objective-controls-sha256": (
                receipt.objective_controls_contract_sha256
            ),
            "run-id": checkpoint.run_id,
            "seed": str(checkpoint.seed),
            "sha256": checkpoint.object.sha256,
            "sidecar-name": checkpoint.data.sidecar_name,
            "step": str(checkpoint.step),
            "world-size": str(checkpoint.world_size),
        }

    def _fetch_checkpoint_receipt_v3(
        self,
        *,
        receipt_uri: str,
        receipt_sha256: str,
        receipt_version_id: str,
        release: object,
        manifest: object,
        evidence: Mapping[str, str],
    ):
        """GET one exact receipt version and HEAD both exact checkpoints."""

        expected_evidence = {
            "environment_receipt_sha256",
            "instance_id",
            "boot_id",
        }
        if set(evidence) != expected_evidence:
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                "v3 checkpoint environment evidence is incomplete",
            )
        receipt_digest = require_sha256(
            receipt_sha256,
            label="v3 checkpoint receipt",
        )
        if (
            not isinstance(receipt_version_id, str)
            or receipt_version_id in {"", "null"}
        ):
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                "v3 checkpoint receipt version ID is invalid",
            )
        prefix = self.runtime.s3_root.rstrip("/") + "/"
        if (
            not isinstance(receipt_uri, str)
            or not receipt_uri.startswith(prefix)
        ):
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                "v3 checkpoint receipt is outside the pinned S3 root",
            )
        receipt_bucket, receipt_key = self._s3_location(
            receipt_uri.removeprefix(prefix)
        )
        expected_checksum = base64.b64encode(
            bytes.fromhex(receipt_digest)
        ).decode("ascii")
        with tempfile.TemporaryDirectory(
            prefix="memorysplit-checkpoint-receipt-",
            dir=Path(tempfile.gettempdir()).resolve(),
        ) as temporary:
            destination = Path(temporary) / "receipt.json"
            output = _aws_output_object(
                self._run(
                    self._aws_argv(
                        "s3api",
                        "get-object",
                        "--bucket",
                        receipt_bucket,
                        "--key",
                        receipt_key,
                        "--version-id",
                        receipt_version_id,
                        "--checksum-mode",
                        "ENABLED",
                        str(destination),
                        query=(
                            "{receipt:{checksum_sha256:ChecksumSHA256,"
                            "version_id:VersionId}}"
                        ),
                    ),
                    operation="fetch v3 checkpoint receipt",
                ),
                {"receipt"},
                label="v3 checkpoint receipt download",
            )
            receipt_output = _aws_output_object(
                output["receipt"],
                {"checksum_sha256", "version_id"},
                label="v3 checkpoint receipt object",
            )
            try:
                payload = read_regular_input(
                    destination,
                    label="v3 checkpoint receipt",
                    maximum_bytes=16 * 1024 * 1024,
                )
            except (AttestationError, OSError) as error:
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    "downloaded v3 checkpoint receipt is unsafe",
                ) from error
        if (
            receipt_output["checksum_sha256"] != expected_checksum
            or receipt_output["version_id"] != receipt_version_id
            or hashlib.sha256(payload).hexdigest() != receipt_digest
        ):
            raise MsctlError(
                "CHECKPOINT_PROVENANCE_MISMATCH",
                "downloaded v3 checkpoint receipt identity differs",
            )
        receipt = parse_paired_checkpoint_receipt_v3(
            payload,
            receipt_uri=receipt_uri,
            receipt_sha256=receipt_digest,
            receipt_version_id=receipt_version_id,
        )
        verify_aws_checkpoint_receipt_v3(
            receipt,
            release=release,
            manifest=manifest,
            environment_receipt_sha256=evidence[
                "environment_receipt_sha256"
            ],
            instance_id=evidence["instance_id"],
            boot_id=evidence["boot_id"],
        )
        for checkpoint in receipt.checkpoints:
            checkpoint_uri = checkpoint.object.uri
            if not checkpoint_uri.startswith(prefix):
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    "v3 checkpoint object is outside the pinned S3 root",
                )
            bucket, key = self._s3_location(
                checkpoint_uri.removeprefix(prefix)
            )
            head_output = _aws_output_object(
                self._run(
                    self._aws_argv(
                        "s3api",
                        "head-object",
                        "--bucket",
                        bucket,
                        "--key",
                        key,
                        "--version-id",
                        checkpoint.object.version_id,
                        "--checksum-mode",
                        "ENABLED",
                        query=(
                            "{object:{checksum_sha256:ChecksumSHA256,"
                            "content_length:ContentLength,"
                            "metadata:Metadata,version_id:VersionId}}"
                        ),
                    ),
                    operation=f"verify v3 {checkpoint.arm} checkpoint",
                ),
                {"object"},
                label=f"v3 {checkpoint.arm} checkpoint head",
            )
            head = _aws_output_object(
                head_output["object"],
                {
                    "checksum_sha256",
                    "content_length",
                    "metadata",
                    "version_id",
                },
                label=f"v3 {checkpoint.arm} checkpoint object",
            )
            checkpoint_checksum = base64.b64encode(
                bytes.fromhex(checkpoint.object.sha256)
            ).decode("ascii")
            if (
                head["checksum_sha256"] != checkpoint_checksum
                or head["content_length"] != checkpoint.object.bytes
                or head["metadata"]
                != self._v3_checkpoint_metadata(receipt, checkpoint)
                or head["version_id"] != checkpoint.object.version_id
            ):
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    f"v3 {checkpoint.arm} checkpoint version differs",
                )
        return receipt

    def _checkpoint_map(
        self,
        manifest: object,
        receipt: object,
    ) -> dict[str, object]:
        checkpoints = getattr(receipt, "checkpoints", ())
        by_run = {run.run_id: run for run in manifest.runs}
        if getattr(manifest, "schema_version", None) == 3:
            if (
                getattr(receipt, "schema_version", None) != 3
                or len(checkpoints) != 2
                or {checkpoint.run_id for checkpoint in checkpoints}
                != set(by_run)
            ):
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    "AWS v3 resume requires one complete versioned pair",
                )
            by_arm = {}
            for checkpoint in checkpoints:
                run = by_run[checkpoint.run_id]
                if (
                    checkpoint.arm != run.arm
                    or checkpoint.seed != manifest.seed
                    or checkpoint.world_size != 4
                    or checkpoint.config_sha256 != run.config_sha256
                    or checkpoint.object.version_id in {"", "null"}
                ):
                    raise MsctlError(
                        "CHECKPOINT_PROVENANCE_MISMATCH",
                        "AWS v3 checkpoint does not match its seed arm provenance",
                    )
                by_arm[checkpoint.arm] = checkpoint
            if set(by_arm) != {"dense", "split90"}:
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    "AWS v3 checkpoint pair is incomplete",
                )
            return by_arm
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
        terminate_at: str | None = None,
        bootstrap_mode: str | None = None,
    ) -> dict[str, object]:
        self._validate_manifest(manifest)
        self._validate_release(release, manifest)
        selected = self._selected_manifest_lifecycle(manifest)
        if not selected and (
            terminate_at is not None or bootstrap_mode is not None
        ):
            raise MsctlError(
                "CLI_USAGE",
                "fresh deadlines and bootstrap modes are selected-manifest "
                "arguments",
            )
        fresh_deadline: str | None = None
        if selected:
            if not isinstance(terminate_at, str):
                raise MsctlError(
                    "TERMINATION_DEADLINE_INVALID",
                    "selected resume requires a fresh operator termination "
                    "deadline",
                )
            if bootstrap_mode not in {"bootstrap", "reuse"}:
                raise MsctlError(
                    "CLI_USAGE",
                    "selected resume requires --bootstrap-mode bootstrap "
                    "or reuse",
                )
            self._validate_submit_selection(
                manifest.instance_id,
                terminate_at,
            )
            fresh_deadline = terminate_at
        is_v3 = getattr(manifest, "schema_version", None) == 3
        if is_v3:
            if evidence is None or set(evidence) != {
                "dataset_pointer_sha256",
                "dataset_verification_sha256",
                "environment_receipt_sha256",
                "instance_id",
                "boot_id",
            }:
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    "AWS v3 resume evidence is incomplete",
                )
            verify_aws_checkpoint_receipt_v3(
                checkpoint_receipt,
                release=release,
                manifest=manifest,
                environment_receipt_sha256=evidence[
                    "environment_receipt_sha256"
                ],
                instance_id=evidence["instance_id"],
                boot_id=evidence["boot_id"],
            )
        checkpoints = self._checkpoint_map(manifest, checkpoint_receipt)
        checkpoint_publication = (
            {
                "commands": [],
                "objects": [
                    {
                        "arm": arm,
                        "sha256": checkpoints[arm].object.sha256,
                        "uri": checkpoints[arm].object.uri,
                        "version_id": checkpoints[arm].object.version_id,
                    }
                    for arm in ("dense", "split90")
                ],
            }
            if is_v3
            else self._checkpoint_publication(
                checkpoint_receipt=checkpoint_receipt,
                checkpoints=checkpoints,
                apply=False,
            )
        )
        operation_intent: dict[str, object] | None = None
        if not selected:
            operation_intent = self._training_operation_intent(
                operation="resume",
                release=release,
                manifest=manifest,
                checkpoints=checkpoints,
                checkpoint_receipt_sha256=checkpoint_receipt.sha256,
                checkpoint_receipt_uri=(
                    checkpoint_receipt.uri if is_v3 else None
                ),
                checkpoint_receipt_version_id=(
                    checkpoint_receipt.version_id if is_v3 else None
                ),
                checkpoint_receipt_bytes=(
                    checkpoint_receipt.bytes if is_v3 else None
                ),
                evidence=evidence,
            )
        if not apply:
            if selected:
                return {
                    "provider": manifest.provider,
                    "seed": manifest.seed,
                    "run_manifest_sha256": manifest.sha256,
                    "checkpoint_receipt_sha256": checkpoint_receipt.sha256,
                    "checkpoint_receipt_uri": checkpoint_receipt.uri,
                    "checkpoint_receipt_version_id": (
                        checkpoint_receipt.version_id
                    ),
                    "checkpoint_objects": checkpoint_publication["objects"],
                    "terminate_at": fresh_deadline,
                    "bootstrap_mode": bootstrap_mode,
                    "submitted": 0,
                    "idempotent": False,
                }
            return {
                "provider": AWS_P5_PROFILE,
                "seed": manifest.seed,
                "run_manifest_sha256": manifest.sha256,
                "checkpoint_receipt_sha256": checkpoint_receipt.sha256,
                **(
                    {
                        "checkpoint_receipt_uri": checkpoint_receipt.uri,
                        "checkpoint_receipt_version_id": (
                            checkpoint_receipt.version_id
                        ),
                    }
                    if is_v3
                    else {}
                ),
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
        if selected:
            resume_resources = self._resume_resources(
                release=release,
                manifest=manifest,
                instance_id=manifest.instance_id,
                terminate_at=str(fresh_deadline),
                checkpoint_receipt_sha256=checkpoint_receipt.sha256,
                checkpoint_receipt=checkpoint_receipt,
                checkpoints=checkpoints,
            )
            resume_resources.update(
                {
                    "dataset_pointer_sha256": evidence[
                        "dataset_pointer_sha256"
                    ],
                    "dataset_verification_sha256": evidence[
                        "dataset_verification_sha256"
                    ],
                    "environment_receipt_sha256": evidence[
                        "environment_receipt_sha256"
                    ],
                    **{
                        field: (
                            list(manifest.arms)
                            if field == "arms"
                            else getattr(manifest, field)
                        )
                        for field in LIFECYCLE_BINDING_FIELDS
                    },
                    "bootstrap_mode": bootstrap_mode,
                }
            )
            self._verify_approval(
                approval_path,
                operation="resume",
                release=release,
                manifest=manifest,
                resources=resume_resources,
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
            expected_environment_receipt = (
                str(evidence["environment_receipt_sha256"])
                if selected
                else operation_intent["environment_receipt_sha256"]
            )
            if {
                state.get("environment_receipt_sha256") for state in present
            } != {expected_environment_receipt}:
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
            if selected:
                if instance_id != manifest.instance_id:
                    raise MsctlError(
                        "INSTANCE_BINDING_MISMATCH",
                        "paired state binds a different cohort instance",
                    )
                terminate_at = str(fresh_deadline)
            else:
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
                    checkpoint_receipt=(
                        checkpoint_receipt if is_v3 else None
                    ),
                    checkpoints=checkpoints if is_v3 else None,
                )
                resume_resources.update(
                    {
                        field: operation_intent[field]
                        for field in (
                            "dataset_pointer_sha256",
                            "dataset_verification_sha256",
                            "environment_receipt_sha256",
                            *LIFECYCLE_BINDING_FIELDS,
                        )
                        if field in operation_intent
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

            v3_receipt_binding = (
                {
                    "sha256": checkpoint_receipt.sha256,
                    "uri": checkpoint_receipt.uri,
                    "version_id": checkpoint_receipt.version_id,
                }
                if is_v3
                else None
            )
            v3_object_bindings = (
                [
                    {
                        "arm": arm,
                        "bytes": checkpoints[arm].object.bytes,
                        "sha256": checkpoints[arm].object.sha256,
                        "uri": checkpoints[arm].object.uri,
                        "version_id": checkpoints[arm].object.version_id,
                    }
                    for arm in ("dense", "split90")
                ]
                if is_v3
                else None
            )
            if is_v3 and any(
                state.get("operation") == "resume"
                and (
                    state.get("checkpoint_receipt")
                    != v3_receipt_binding
                    or state.get("checkpoint_objects")
                    != v3_object_bindings
                )
                for state in present
            ):
                raise MsctlError(
                    "CHECKPOINT_PROVENANCE_MISMATCH",
                    "v3 resume state binds a different receipt version",
                )
            repeated = all(
                state.get("operation") == "resume"
                and (
                    (
                        state.get("checkpoint_receipt")
                        == v3_receipt_binding
                        and state.get("checkpoint_objects")
                        == v3_object_bindings
                    )
                    if is_v3
                    else state.get("checkpoint_receipt_sha256")
                    == checkpoint_receipt.sha256
                )
                for state in present
            )
            if selected:
                repeated = repeated and all(
                    state.get("terminate_at") == terminate_at
                    and state.get("bootstrap_mode") == bootstrap_mode
                    for state in present
                )
            if repeated:
                if selected:
                    attempts = {state.get("attempt") for state in present}
                    receipt_hashes = {
                        state.get("bootstrap_receipt_sha256")
                        for state in present
                    }
                    if (
                        len(attempts) != 1
                        or len(receipt_hashes) != 1
                        or type(next(iter(attempts))) is not int
                    ):
                        raise MsctlError(
                            "STATE_INCOMPLETE",
                            "paired resume state binds divergent replays",
                        )
                    core = self._training_operation_intent(
                        operation="resume",
                        release=release,
                        manifest=manifest,
                        terminate_at=terminate_at,
                        checkpoints=checkpoints,
                        checkpoint_receipt_sha256=checkpoint_receipt.sha256,
                        checkpoint_receipt_uri=checkpoint_receipt.uri,
                        checkpoint_receipt_version_id=(
                            checkpoint_receipt.version_id
                        ),
                        checkpoint_receipt_bytes=checkpoint_receipt.bytes,
                        evidence=evidence,
                        attempt=next(iter(attempts)),
                        bootstrap_mode=bootstrap_mode,
                        bootstrap_receipt_sha256=next(iter(receipt_hashes)),
                    )
                    operation_intent = self._operation_envelope(
                        core,
                        instance_id=instance_id,
                        terminate_at=terminate_at,
                    )
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
                                "provider": manifest.provider,
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
                    present = self._refresh_paired_states(
                        store,
                        manifest,
                        present,
                        {"command_id": command_id, "status": status},
                    )
                    return {
                        "provider": manifest.provider,
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
                present = self._refresh_paired_states(
                    store,
                    manifest,
                    present,
                    {"status": status},
                )
                return {
                    "provider": manifest.provider,
                    "seed": manifest.seed,
                    "instance_id": instance_id,
                    "command_id": command_id,
                    "status": status,
                    "submitted": 0,
                    "idempotent": True,
                }

            resolved_receipt_sha256: str | None = None
            if selected:
                self._require_sequential_seed_transition(store, manifest)
                stored_prior = present[0].get("prior_run_receipt")
                stored_collection = present[0].get(
                    "prior_collection_receipt"
                )
                if manifest.seed > 0:
                    stored_ref = self._selected_prior_run_receipt(
                        manifest,
                        stored_prior,
                    )
                    admitted_prior = self._admit_selected_prior_finalization(
                        manifest,
                        stored_ref,
                    )
                    stored_collection_ref = (
                        self._selected_prior_collection_receipt(
                            manifest,
                            stored_collection,
                        )
                    )
                    self._admit_selected_prior_collection(
                        stored_collection_ref,
                        admitted_prior,
                    )
                elif stored_prior is not None:
                    raise MsctlError(
                        "SEED_TRANSITION_BLOCKED",
                        "seed 0 state must not bind a prior run receipt",
                    )
                elif stored_collection is not None:
                    raise MsctlError(
                        "SEED_TRANSITION_BLOCKED",
                        "seed 0 state must not bind a prior collection "
                        "receipt",
                    )
                resolved_mode, resolved_receipt_sha256 = (
                    self._resolve_selected_bootstrap_mode(
                        release,
                        manifest,
                        boot_id=str(evidence["boot_id"]),
                    )
                )
                if resolved_mode != bootstrap_mode:
                    raise MsctlError(
                        "BOOTSTRAP_REUSE_INVALID",
                        "declared bootstrap mode differs from the resolved "
                        "durable receipt",
                        details={
                            "declared": bootstrap_mode,
                            "resolved": resolved_mode,
                        },
                    )
                self._validate_selected_identity_instance(
                    manifest,
                    instance_id=instance_id,
                    operation="resume",
                )
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
            if selected:
                core = self._training_operation_intent(
                    operation="resume",
                    release=release,
                    manifest=manifest,
                    terminate_at=terminate_at,
                    checkpoints=checkpoints,
                    checkpoint_receipt_sha256=checkpoint_receipt.sha256,
                    checkpoint_receipt_uri=checkpoint_receipt.uri,
                    checkpoint_receipt_version_id=(
                        checkpoint_receipt.version_id
                    ),
                    checkpoint_receipt_bytes=checkpoint_receipt.bytes,
                    evidence=evidence,
                    attempt=attempt,
                    bootstrap_mode=bootstrap_mode,
                    bootstrap_receipt_sha256=resolved_receipt_sha256,
                )
                operation_intent = self._operation_envelope(
                    core,
                    instance_id=instance_id,
                    terminate_at=terminate_at,
                )
            if not is_v3:
                self._checkpoint_publication(
                    checkpoint_receipt=checkpoint_receipt,
                    checkpoints=checkpoints,
                    apply=True,
                )
            published = self._publish_operation_intent(operation_intent)
            now = _timestamp()
            transitioned = []
            for state in present:
                prior = list(state.get("prior_command_ids", []))
                prior.append(previous_command)
                checkpoint_state = (
                    {
                        "checkpoint_receipt": v3_receipt_binding,
                        "checkpoint_objects": v3_object_bindings,
                    }
                    if is_v3
                    else {
                        "checkpoint_receipt_sha256": (
                            checkpoint_receipt.sha256
                        )
                    }
                )
                next_state = dict(state)
                next_state.update(
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
                        **checkpoint_state,
                        **(
                            {
                                "terminate_at": terminate_at,
                                "bootstrap_mode": bootstrap_mode,
                                "bootstrap_receipt_sha256": (
                                    resolved_receipt_sha256
                                ),
                            }
                            if selected
                            else {}
                        ),
                        "prior_command_ids": prior,
                        "updated_at": now,
                    }
                )
                transitioned.append(next_state)
            self._write_paired_states(store, manifest, transitioned)
            present = transitioned
            present = self._refresh_paired_states(
                store,
                manifest,
                present,
                {"send_attempted": True, "status": "SENDING"},
            )
            command_id = self._send_operation_intent(
                instance_id=instance_id,
                intent=operation_intent,
                published=published,
                operation="resume",
            )
            present = self._refresh_paired_states(
                store,
                manifest,
                present,
                {"command_id": command_id, "status": "Pending"},
            )
            return {
                "provider": manifest.provider,
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
        journal = store.read_aws_pair(manifest.sha256)
        states = [store.read_run(run.run_id) for run in manifest.runs]
        if any(state is None for state in states):
            raise MsctlError(
                "RUN_STATE_MISSING",
                "AWS lifecycle state is missing one paired arm",
            )
        present = [state for state in states if state is not None]
        if journal is None:
            if getattr(manifest, "schema_version", None) == 3:
                raise MsctlError(
                    "STATE_INCOMPLETE",
                    "AWS v3 lifecycle state lacks its durable pair journal",
                )
        else:
            journal_by_id = {
                str(state["run_id"]): state for state in journal["states"]
            }
            present_by_id = {
                str(state["run_id"]): state for state in present
            }
            if (
                set(journal_by_id) != {run.run_id for run in manifest.runs}
                or present_by_id != journal_by_id
            ):
                raise MsctlError(
                    "STATE_CORRUPT",
                    "AWS lifecycle state conflicts with its pair journal",
                )
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
        checkpoint_bindings = {
            canonical_json(
                {
                    "receipt": state.get("checkpoint_receipt"),
                    "objects": state.get("checkpoint_objects"),
                }
                if state.get("schema_version") == 2
                else {
                    "sha256": state.get(
                        "checkpoint_receipt_sha256"
                    )
                }
            )
            for state in present
        }
        if (
            present[0].get("operation") == "resume"
            and len(checkpoint_bindings) != 1
        ):
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
            states = self._refresh_paired_states(
                store,
                manifest,
                states,
                {"status": "Cancelling"},
            )
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
                        *LIFECYCLE_BINDING_FIELDS,
                    )
                    if field in operation_intent
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
            states = self._refresh_paired_states(
                store,
                manifest,
                states,
                {"status": "Terminating"},
            )
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
        if command == "canary run":
            apply = bool(getattr(args, "apply", False))
            required = {
                "release": getattr(args, "release", None),
                "manifest": getattr(args, "manifest", None),
                "dataset_receipt": getattr(args, "dataset_receipt", None),
                "environment_receipt": getattr(
                    args,
                    "environment_receipt",
                    None,
                ),
                "runtime_lock": getattr(args, "runtime_lock", None),
                "instance_id": getattr(args, "instance_id", None),
                "boot_id": getattr(args, "boot_id", None),
                "output": getattr(args, "output", None),
                "repo_root": getattr(args, "repo_root", None),
            }
            if any(value is None for value in required.values()):
                raise MsctlError(
                    "CLI_USAGE",
                    "AWS v3 canary requires every explicit qualification input",
                    details={
                        "missing": [
                            f"--{name.replace('_', '-')}"
                            for name, value in required.items()
                            if value is None
                        ]
                    },
                )
            return not apply, self.canary_run(
                release_root=required["repo_root"],
                release_receipt=required["release"],
                run_manifest=required["manifest"],
                dataset_receipt=required["dataset_receipt"],
                environment_receipt=required["environment_receipt"],
                runtime_lock=required["runtime_lock"],
                instance_id=required["instance_id"],
                boot_id=required["boot_id"],
                output=required["output"],
                apply=apply,
            )
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
            selected_arguments = {
                "release": getattr(args, "release", None),
                "manifest": getattr(args, "manifest", None),
                "run_receipt_uri": getattr(args, "run_receipt_uri", None),
                "run_receipt_sha256": getattr(
                    args,
                    "run_receipt_sha256",
                    None,
                ),
                "run_receipt_version_id": getattr(
                    args,
                    "run_receipt_version_id",
                    None,
                ),
            }
            source = getattr(args, "source", None)
            if any(
                value is not None for value in selected_arguments.values()
            ):
                if source is not None:
                    raise MsctlError(
                        "CLI_USAGE",
                        "selected collect forbids the legacy --source "
                        "argument",
                    )
                missing = [
                    f"--{name.replace('_', '-')}"
                    for name, value in selected_arguments.items()
                    if value is None
                ]
                if getattr(args, "out", None) is None:
                    missing.append("--out")
                if missing:
                    raise MsctlError(
                        "CLI_USAGE",
                        "selected collect requires its complete argument "
                        "set",
                        details={"missing": missing},
                    )
                release, manifest = self._load_bound_inputs(
                    release_path=selected_arguments["release"],
                    manifest_path=selected_arguments["manifest"],
                    repo_root=args.repo_root,
                )
                # Collection needs no paid-instance approval: it creates no
                # capacity and only publishes one content-addressed receipt.
                return not apply, self.collect_seed_evidence(
                    release=release,
                    manifest=manifest,
                    run_receipt={
                        "uri": str(
                            selected_arguments["run_receipt_uri"]
                        ),
                        "sha256": str(
                            selected_arguments["run_receipt_sha256"]
                        ),
                        "version_id": str(
                            selected_arguments["run_receipt_version_id"]
                        ),
                    },
                    out=args.out,
                    apply=apply,
                )
            if source is None:
                raise MsctlError(
                    "CLI_USAGE",
                    "AWS collect requires --source",
                )
            return not apply, self.collect(
                source=str(source),
                out=args.out,
                apply=apply,
            )
        if command == "collect-cohort":
            apply = bool(getattr(args, "apply", False))
            required = {
                "release": getattr(args, "release", None),
                "manifest": getattr(args, "manifest", None),
                "cohort_report_uri": getattr(
                    args,
                    "cohort_report_uri",
                    None,
                ),
                "cohort_report_sha256": getattr(
                    args,
                    "cohort_report_sha256",
                    None,
                ),
                "cohort_report_version_id": getattr(
                    args,
                    "cohort_report_version_id",
                    None,
                ),
                "evidence_index": getattr(args, "evidence_index", None),
                "out": getattr(args, "out", None),
            }
            missing = [
                f"--{name.replace('_', '-')}"
                for name, value in required.items()
                if value is None
            ]
            if missing:
                raise MsctlError(
                    "CLI_USAGE",
                    "selected collect-cohort requires its complete "
                    "argument set",
                    details={"missing": missing},
                )
            release, manifest = self._load_bound_inputs(
                release_path=required["release"],
                manifest_path=required["manifest"],
                repo_root=args.repo_root,
            )
            # Like selected collect, cohort collection needs no
            # paid-instance approval: it creates no capacity and only
            # publishes one content-addressed receipt.
            return not apply, self.collect_cohort_evidence(
                release=release,
                manifest=manifest,
                cohort_report={
                    "uri": str(required["cohort_report_uri"]),
                    "sha256": str(required["cohort_report_sha256"]),
                    "version_id": str(
                        required["cohort_report_version_id"]
                    ),
                },
                evidence_index=required["evidence_index"],
                out=required["out"],
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
            if (
                getattr(manifest, "schema_version", None) == 3
                and (
                    command == "evaluate"
                    or command.startswith("cleanup")
                )
            ):
                if command == "evaluate":
                    raise MsctlError(
                        "OPERATION_UNSUPPORTED",
                        "AWS v3 evaluation awaits Task 4E "
                        "selected-evaluation enablement",
                    )
                # Per-seed collection receipts and the cohort
                # evaluation-evidence collection receipt are necessary but
                # never sufficient: even ten collection states plus a
                # valid cohort collection state cannot reach
                # terminate-instances before cleanup enablement.
                raise MsctlError(
                    "OPERATION_UNSUPPORTED",
                    "AWS v3 cleanup requires all ten per-seed collection "
                    "receipts, the cohort evaluation-evidence collection "
                    "receipt, and cleanup enablement itself, which "
                    "follows selected evaluation (Task 4E)",
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
                selected = self._selected_manifest_lifecycle(manifest)
                prior_values = (
                    getattr(args, "prior_run_receipt_uri", None),
                    getattr(args, "prior_run_receipt_sha256", None),
                    getattr(args, "prior_run_receipt_version_id", None),
                )
                collection_values = (
                    getattr(
                        args,
                        "prior_collection_receipt_uri",
                        None,
                    ),
                    getattr(
                        args,
                        "prior_collection_receipt_sha256",
                        None,
                    ),
                    getattr(
                        args,
                        "prior_collection_receipt_version_id",
                        None,
                    ),
                )
                declared_mode = getattr(args, "bootstrap_mode", None)
                if not selected and (
                    declared_mode is not None
                    or any(value is not None for value in prior_values)
                    or any(
                        value is not None for value in collection_values
                    )
                ):
                    raise MsctlError(
                        "CLI_USAGE",
                        "bootstrap and prior-receipt arguments require a "
                        "selected manifest",
                    )
                if selected and declared_mode is None:
                    raise MsctlError(
                        "CLI_USAGE",
                        "selected submit requires --bootstrap-mode",
                    )
                prior_run_receipt = None
                if any(value is not None for value in prior_values):
                    if any(value is None for value in prior_values):
                        raise MsctlError(
                            "CLI_USAGE",
                            "the prior receipt triple requires URI, SHA-256, "
                            "and version ID together",
                        )
                    prior_run_receipt = {
                        "uri": str(prior_values[0]),
                        "sha256": str(prior_values[1]),
                        "version_id": str(prior_values[2]),
                    }
                prior_collection_receipt = None
                if any(value is not None for value in collection_values):
                    if any(value is None for value in collection_values):
                        raise MsctlError(
                            "CLI_USAGE",
                            "the prior collection triple requires URI, "
                            "SHA-256, and version ID together",
                        )
                    prior_collection_receipt = {
                        "uri": str(collection_values[0]),
                        "sha256": str(collection_values[1]),
                        "version_id": str(collection_values[2]),
                    }
                if selected and manifest.seed > 0 and (
                    prior_run_receipt is None
                    or prior_collection_receipt is None
                ):
                    raise MsctlError(
                        "CLI_USAGE",
                        "selected seeds 1 through 9 require the prior "
                        "receipt and prior collection triples",
                        details={
                            "missing": [
                                *(
                                    [
                                        "--prior-run-receipt-uri",
                                        "--prior-run-receipt-sha256",
                                        "--prior-run-receipt-version-id",
                                    ]
                                    if prior_run_receipt is None
                                    else []
                                ),
                                *(
                                    [
                                        "--prior-collection-receipt-uri",
                                        "--prior-collection-receipt-sha256",
                                        (
                                            "--prior-collection-receipt-"
                                            "version-id"
                                        ),
                                    ]
                                    if prior_collection_receipt is None
                                    else []
                                ),
                            ]
                        },
                    )
                return not args.apply, self.submit(
                    release=release,
                    manifest=manifest,
                    instance_id=args.instance_id,
                    terminate_at=args.terminate_at,
                    approval_path=args.approval,
                    apply=args.apply,
                    evidence=evidence,
                    bootstrap_mode=declared_mode,
                    prior_run_receipt=prior_run_receipt,
                    prior_collection_receipt=prior_collection_receipt,
                )
            if command == "status":
                return False, self.status(
                    release=release,
                    manifest=manifest,
                    cached=args.cached,
                )
            if command == "resume":
                remote_values = (
                    getattr(args, "checkpoint_receipt_uri", None),
                    getattr(args, "checkpoint_receipt_sha256", None),
                    getattr(args, "checkpoint_receipt_version_id", None),
                )
                if getattr(manifest, "schema_version", None) == 3:
                    if getattr(args, "checkpoint_receipt", None) is not None:
                        raise MsctlError(
                            "CLI_USAGE",
                            "schema-3 v3 resume rejects legacy local checkpoint receipts",
                        )
                    if any(value is None for value in remote_values):
                        raise MsctlError(
                            "CLI_USAGE",
                            "v3 resume requires the complete checkpoint receipt URI/hash/version triple",
                        )
                    assert evidence is not None
                    receipt = self._fetch_checkpoint_receipt_v3(
                        receipt_uri=str(remote_values[0]),
                        receipt_sha256=str(remote_values[1]),
                        receipt_version_id=str(remote_values[2]),
                        release=release,
                        manifest=manifest,
                        evidence={
                            field: evidence[field]
                            for field in (
                                "environment_receipt_sha256",
                                "instance_id",
                                "boot_id",
                            )
                        },
                    )
                else:
                    if (
                        getattr(args, "checkpoint_receipt", None) is None
                        or any(value is not None for value in remote_values)
                    ):
                        raise MsctlError(
                            "CLI_USAGE",
                            "legacy AWS resume requires only --checkpoint-receipt",
                        )
                    receipt = verify_checkpoint_receipt(
                        args.checkpoint_receipt,
                        release=release,
                        manifest=manifest,
                    )
                selected = self._selected_manifest_lifecycle(manifest)
                resume_deadline = getattr(args, "terminate_at", None)
                resume_mode = getattr(args, "bootstrap_mode", None)
                if not selected and (
                    resume_deadline is not None or resume_mode is not None
                ):
                    raise MsctlError(
                        "CLI_USAGE",
                        "fresh deadlines and bootstrap modes require a "
                        "selected manifest",
                    )
                if selected and (
                    resume_deadline is None or resume_mode is None
                ):
                    raise MsctlError(
                        "CLI_USAGE",
                        "selected resume requires fresh --terminate-at and "
                        "--bootstrap-mode",
                        details={
                            "missing": [
                                name
                                for name, value in (
                                    ("--terminate-at", resume_deadline),
                                    ("--bootstrap-mode", resume_mode),
                                )
                                if value is None
                            ]
                        },
                    )
                return not args.apply, self.resume(
                    release=release,
                    manifest=manifest,
                    checkpoint_receipt=receipt,
                    approval_path=args.approval,
                    apply=args.apply,
                    evidence=evidence,
                    terminate_at=resume_deadline,
                    bootstrap_mode=resume_mode,
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
