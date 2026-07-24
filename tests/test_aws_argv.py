from __future__ import annotations

import hashlib
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from msctl import aws_contracts
from msctl.aws_argv import (
    ARGV_DOCUMENT_NAME,
    ARGV_DOCUMENT_SHA256,
    RemoteIntentError,
    _validate_intent,
    execute_intent,
)
from msctl.jsonutil import canonical_json, canonical_sha256


DIGEST = "sha256:" + "a" * 64
IMAGE = (
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/memorysplit"
    f"@{DIGEST}"
)


def _intent() -> dict[str, object]:
    terminate_at = (
        datetime.now(UTC) + timedelta(hours=1)
    ).isoformat().replace("+00:00", "Z")
    identity: dict[str, object] = {
        "schema_version": 1,
        "operation": "submit",
        "provider": "aws-p5.48xlarge",
        "seed": 1,
        "release_sha256": "1" * 64,
        "run_manifest_sha256": "2" * 64,
        "dataset_sha256": "3" * 64,
        "dataset_pointer_sha256": "4" * 64,
        "dataset_verification_sha256": "5" * 64,
        "environment_receipt_sha256": "6" * 64,
        "runtime_sha256": "7" * 64,
        "environment": {
            "AWS_REGION": "us-east-1",
            "MS_AWS_AMI_ID": "ami-0123456789abcdef0",
            "MS_CONTAINER_DIGEST": DIGEST,
            "MS_CONTAINER_IMAGE": IMAGE,
            "MS_RUNTIME_GID": "1000",
            "MS_RUNTIME_UID": "1000",
            "MS_S3_ROOT": "s3://memorysplit-prod/confirmatory-v3",
        },
        "checkpoint_receipt": None,
        "steps": [
            {
                "name": "auto-termination",
                "argv": ["/usr/bin/systemd-run", "--version"],
            },
            {
                "name": "prepare-aws-private-home",
                "argv": ["/usr/bin/install", "-d", "/var/lib/memorysplit"],
            },
            {
                "name": "bootstrap",
                "argv": ["/usr/bin/python3", "/opt/memorysplit/bootstrap.py"],
            },
        ],
        "instance_id": "i-0123456789abcdef0",
        "terminate_at": terminate_at,
    }
    operation_id = canonical_sha256(identity)
    receipt_root = (
        "s3://memorysplit-prod/confirmatory-v3/"
        f"operations/{operation_id}/receipts"
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


def _payload(value: dict[str, object]) -> tuple[bytes, str]:
    payload = canonical_json(value)
    return payload, hashlib.sha256(payload).hexdigest()


def _rebind(value: dict[str, object]) -> tuple[bytes, str]:
    identity = {
        key: item
        for key, item in value.items()
        if key
        not in {
            "operation_id",
            "ssm_document",
            "started_receipt_uri",
            "terminal_receipt_uri",
        }
    }
    value["operation_id"] = canonical_sha256(identity)
    receipt_root = (
        "s3://memorysplit-prod/confirmatory-v3/"
        f"operations/{value['operation_id']}/receipts"
    )
    value["started_receipt_uri"] = f"{receipt_root}/started.json"
    value["terminal_receipt_uri"] = f"{receipt_root}/terminal.json"
    return _payload(value)


def test_remote_intent_requires_exact_digest_pinned_container_image():
    intent = _intent()
    payload, digest = _payload(intent)

    parsed = _validate_intent(payload, expected_sha256=digest)

    assert parsed["environment"]["MS_CONTAINER_IMAGE"] == IMAGE
    for mutation in ("missing", "mismatch", "mutable", "tagged"):
        value = deepcopy(intent)
        environment = value["environment"]
        if mutation == "missing":
            environment.pop("MS_CONTAINER_IMAGE")
        elif mutation == "mismatch":
            environment["MS_CONTAINER_IMAGE"] = IMAGE.replace("a" * 64, "b" * 64)
        else:
            environment["MS_CONTAINER_IMAGE"] = (
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/"
                + (
                    f"memorysplit:latest@{DIGEST}"
                    if mutation == "tagged"
                    else "memorysplit:latest"
                )
            )
        mutated_payload, mutated_digest = _rebind(value)
        with pytest.raises(RemoteIntentError, match="environment|container|image"):
            _validate_intent(
                mutated_payload,
                expected_sha256=mutated_digest,
            )


@pytest.mark.parametrize(
    "image",
    [
        "registry.example/@sha256:" + "a" * 64,
        "registry.example/repo//@sha256:" + "a" * 64,
        "registry.example/repo/./part@sha256:" + "a" * 64,
        "registry.example/repo/../part@sha256:" + "a" * 64,
        "registry.example/repo:tag@sha256:" + "a" * 64,
        "registry.example/repo@@sha256:" + "a" * 64,
        "registry.example/repo@sha256:" + "A" * 64,
        "registry.example/re po@sha256:" + "a" * 64,
        "-registry.example/repo@sha256:" + "a" * 64,
    ],
)
def test_remote_intent_rejects_malformed_oci_image_references(image):
    value = _intent()
    value["environment"]["MS_CONTAINER_IMAGE"] = image
    payload, digest = _rebind(value)

    with pytest.raises(RemoteIntentError, match="container|image"):
        _validate_intent(payload, expected_sha256=digest)


def test_shared_oci_validator_is_closed_and_digest_matched():
    validator = getattr(
        aws_contracts,
        "validate_digest_pinned_oci_image",
        None,
    )
    assert callable(validator)
    assert validator(IMAGE, DIGEST) == (IMAGE, DIGEST)

    with pytest.raises(ValueError, match="container image|digest"):
        validator(IMAGE, "sha256:" + "b" * 64)


@pytest.mark.parametrize(
    "credential_name",
    [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    ],
)
def test_remote_intent_rejects_every_credential_and_config_source(
    credential_name,
):
    value = _intent()
    value["environment"][credential_name] = "must-not-cross-boundary"
    identity = {
        key: item
        for key, item in value.items()
        if key
        not in {
            "operation_id",
            "ssm_document",
            "started_receipt_uri",
            "terminal_receipt_uri",
        }
    }
    value["operation_id"] = canonical_sha256(identity)
    receipt_root = (
        "s3://memorysplit-prod/confirmatory-v3/"
        f"operations/{value['operation_id']}/receipts"
    )
    value["started_receipt_uri"] = f"{receipt_root}/started.json"
    value["terminal_receipt_uri"] = f"{receipt_root}/terminal.json"
    payload, digest = _payload(value)

    with pytest.raises(RemoteIntentError, match="environment|credential"):
        _validate_intent(payload, expected_sha256=digest)


S3_ROOT = "s3://memorysplit-prod/confirmatory-v3"
_LEASE_RESET_ARGV = [
    "/usr/bin/systemctl",
    "stop",
    "memorysplit-auto-terminate*.timer",
    "memorysplit-auto-terminate*.service",
]


def _selected_lifecycle_fields(provider: str = "aws-p5.48xlarge") -> dict:
    return {
        "account_id": "123456789012",
        "arms": ["dense", "split90"],
        "availability_zone": "us-east-1d",
        "cohort_id": "memorysplit-confirmatory-v3-360m-n10-aws",
        "hardware_amendment_sha256": "b" * 64,
        "objective_controls_contract_sha256": "c" * 64,
        "profile_id": f"{provider}-v3",
        "profile_sha256": "d" * 64,
        "provider_selection_sha256": "e" * 64,
        "provider_selection_version_id": "selection-version-1",
        "purchase_model": "on_demand",
        "qualification_approval_public_key_sha256": "f" * 64,
        "qualification_approval_receipt_sha256": "0" * 64,
        "qualification_canary_receipt_sha256": "9" * 64,
        "qualification_environment_receipt_sha256": "8" * 64,
        "qualification_evidence_sha256": "7" * 64,
        "region": "us-east-1",
        "runtime_lock_sha256": "3" * 64,
        "runtime_sbom_sha256": "2" * 64,
    }


def _v3_checkpoint_receipt(seed: int = 0) -> dict:
    receipt_sha256 = "a" * 64
    checkpoints = []
    for arm, digest_character in (("dense", "1"), ("split90", "2")):
        digest = digest_character * 64
        checkpoints.append(
            {
                "arm": arm,
                "bytes": 100,
                "config_fingerprint": "3" * 64,
                "config_sha256": "4" * 64,
                "resume_path": (
                    "/mnt/memorysplit/staging/resume/"
                    f"{receipt_sha256}/{arm}.pt"
                ),
                "resume_sha256": digest,
                "step": 1_358,
                "uri": (
                    f"{S3_ROOT}/checkpoints/seed-{seed}/{arm}/"
                    f"sha256/{digest}.pt"
                ),
                "version_id": f"{arm}-version-1",
                "world_size": 4,
            }
        )
    return {
        "bytes": 1_000,
        "checkpoints": checkpoints,
        "sha256": receipt_sha256,
        "uri": (
            f"{S3_ROOT}/receipts/checkpoints/seed-{seed}/"
            f"sha256/{receipt_sha256}.json"
        ),
        "version_id": "receipt-version-1",
    }


def _selected_steps(
    *,
    operation: str,
    bootstrap_mode: str,
    lease_unit: str,
    terminate_at: str,
) -> list[dict]:
    steps = [
        {
            "name": "reset-termination-leases",
            "argv": list(_LEASE_RESET_ARGV),
        },
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
        },
    ]
    if bootstrap_mode == "bootstrap":
        steps.extend(
            [
                {
                    "name": "prepare-aws-private-home",
                    "argv": [
                        "/usr/bin/install",
                        "-d",
                        "-m",
                        "0700",
                        "/var/lib/memorysplit/aws-private-home",
                    ],
                },
                {
                    "name": "bootstrap",
                    "argv": [
                        "/usr/bin/python3",
                        "/opt/memorysplit/cluster/aws/p5/bootstrap.py",
                        "--apply",
                    ],
                },
                {
                    "name": "build-launcher-manifest",
                    "argv": [
                        "/usr/bin/python3",
                        "/opt/memorysplit/msctl/aws_launch_manifest.py",
                    ],
                },
            ]
        )
    else:
        steps.append(
            {
                "name": "verify-bootstrap-reuse",
                "argv": [
                    "/usr/bin/python3",
                    "/opt/memorysplit/cluster/aws/p5/bootstrap.py",
                    "--verify-reuse",
                ],
            }
        )
        if operation == "submit":
            steps.append(
                {
                    "name": "build-launcher-manifest",
                    "argv": [
                        "/usr/bin/python3",
                        "/opt/memorysplit/msctl/aws_launch_manifest.py",
                    ],
                }
            )
    if operation == "resume":
        steps.extend(
            [
                {
                    "name": "prepare-resume-staging",
                    "argv": ["/usr/bin/install", "-d", "-m", "0700"],
                },
                {
                    "name": "materialize-resume-receipt",
                    "argv": ["/usr/bin/env", "aws", "s3api", "get-object"],
                },
                {
                    "name": "materialize-resume-dense",
                    "argv": ["/usr/bin/env", "aws", "s3api", "get-object"],
                },
                {
                    "name": "materialize-resume-split90",
                    "argv": ["/usr/bin/env", "aws", "s3api", "get-object"],
                },
            ]
        )
    steps.append(
        {
            "name": "paired-launch",
            "argv": ["/usr/bin/python3", "/opt/memorysplit/launch.py"],
        }
    )
    return steps


def _selected_intent(
    *,
    operation: str = "submit",
    bootstrap_mode: str = "bootstrap",
    provider: str = "aws-p5.48xlarge",
    seed: int = 0,
    attempt: int = 1,
) -> dict:
    terminate_at = (
        datetime.now(UTC) + timedelta(hours=1)
    ).isoformat().replace("+00:00", "Z")
    run_manifest_sha256 = "2" * 64
    lease_unit = "memorysplit-auto-terminate-" + canonical_sha256(
        {
            "attempt": attempt,
            "operation": operation,
            "run_manifest_sha256": run_manifest_sha256,
            "seed": seed,
            "terminate_at": terminate_at,
        }
    )
    lifecycle = _selected_lifecycle_fields(provider)
    identity: dict[str, object] = {
        **lifecycle,
        "schema_version": 3,
        "operation": operation,
        "provider": provider,
        "seed": seed,
        "release_sha256": "1" * 64,
        "run_manifest_sha256": run_manifest_sha256,
        "dataset_sha256": "3" * 64,
        "dataset_pointer_sha256": "4" * 64,
        "dataset_verification_sha256": "5" * 64,
        "environment_receipt_sha256": "6" * 64,
        "runtime_sha256": "7" * 64,
        "boot_id": "12345678-1234-4abc-8def-1234567890ab",
        "environment": {
            "AWS_REGION": "us-east-1",
            "MS_AWS_AMI_ID": "ami-0123456789abcdef0",
            "MS_CONTAINER_DIGEST": DIGEST,
            "MS_CONTAINER_IMAGE": IMAGE,
            "MS_RUNTIME_GID": "1000",
            "MS_RUNTIME_UID": "1000",
            "MS_S3_ROOT": S3_ROOT,
            "MS_HARDWARE_AMENDMENT_SHA256": (
                lifecycle["hardware_amendment_sha256"]
            ),
            "MS_OBJECTIVE_CONTROLS_SHA256": (
                lifecycle["objective_controls_contract_sha256"]
            ),
            "MS_PROFILE_ID": lifecycle["profile_id"],
            "MS_PROFILE_SHA256": lifecycle["profile_sha256"],
            "MS_PROVIDER": provider,
            "MS_PROVIDER_SELECTION_SHA256": (
                lifecycle["provider_selection_sha256"]
            ),
            "MS_PROVIDER_SELECTION_VERSION_ID": (
                lifecycle["provider_selection_version_id"]
            ),
            "MS_QUALIFICATION_EVIDENCE_SHA256": (
                lifecycle["qualification_evidence_sha256"]
            ),
            "MS_RUNTIME_LOCK_SHA256": lifecycle["runtime_lock_sha256"],
            "MS_RUNTIME_SBOM_SHA256": lifecycle["runtime_sbom_sha256"],
        },
        "bootstrap": {
            "mode": bootstrap_mode,
            "receipt_sha256": (
                None if bootstrap_mode == "bootstrap" else "b" * 64
            ),
        },
        "lease_unit": lease_unit,
        "checkpoint_receipt": (
            _v3_checkpoint_receipt(seed) if operation == "resume" else None
        ),
        "steps": _selected_steps(
            operation=operation,
            bootstrap_mode=bootstrap_mode,
            lease_unit=lease_unit,
            terminate_at=terminate_at,
        ),
        "instance_id": "i-0123456789abcdef0",
        "terminate_at": terminate_at,
    }
    operation_id = canonical_sha256(identity)
    receipt_root = f"{S3_ROOT}/operations/{operation_id}/receipts"
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


def test_wrapper_lifecycle_field_contract_matches_authority_module():
    from msctl.aws_argv import _SELECTED_LIFECYCLE_FIELDS
    from msctl.aws_lifecycle import LIFECYCLE_BINDING_FIELDS

    assert set(_SELECTED_LIFECYCLE_FIELDS) == set(LIFECYCLE_BINDING_FIELDS)


@pytest.mark.parametrize("provider", ["aws-p5.48xlarge", "aws-p6-b300.48xlarge"])
@pytest.mark.parametrize(
    ("operation", "bootstrap_mode"),
    [
        ("submit", "bootstrap"),
        ("submit", "reuse"),
        ("resume", "reuse"),
        ("resume", "bootstrap"),
    ],
)
def test_selected_intent_accepts_all_four_exact_shapes(
    provider,
    operation,
    bootstrap_mode,
):
    intent = _selected_intent(
        operation=operation,
        bootstrap_mode=bootstrap_mode,
        provider=provider,
        seed=4,
        attempt=2 if operation == "resume" else 1,
    )
    payload, digest = _payload(intent)

    parsed = _validate_intent(payload, expected_sha256=digest)

    assert parsed["schema_version"] == 3
    assert parsed["provider"] == provider
    assert parsed["bootstrap"]["mode"] == bootstrap_mode
    assert parsed["lease_unit"].startswith("memorysplit-auto-terminate-")
    names = [step["name"] for step in parsed["steps"]]
    assert names[:2] == ["reset-termination-leases", "auto-termination"]
    assert names[-1] == "paired-launch"


def test_selected_intent_requires_closed_fields_and_selected_environment():
    base = _selected_intent()

    for mutation in (
        "drop-lifecycle",
        "drop-bootstrap",
        "drop-lease",
        "unknown-field",
        "foreign-provider",
        "profile-provider-mismatch",
        "environment-missing-selected",
        "environment-extra",
        "environment-value-drift",
        "seed-out-of-range",
        "arms-drift",
    ):
        value = deepcopy(base)
        if mutation == "drop-lifecycle":
            value.pop("provider_selection_sha256")
        elif mutation == "drop-bootstrap":
            value.pop("bootstrap")
        elif mutation == "drop-lease":
            value.pop("lease_unit")
        elif mutation == "unknown-field":
            value["surprise"] = "field"
        elif mutation == "foreign-provider":
            value["provider"] = "gcp-a3.highgpu"
            value["environment"]["MS_PROVIDER"] = "gcp-a3.highgpu"
        elif mutation == "profile-provider-mismatch":
            value["profile_id"] = "aws-p6-b300.48xlarge-v3"
            value["environment"]["MS_PROFILE_ID"] = (
                "aws-p6-b300.48xlarge-v3"
            )
        elif mutation == "environment-missing-selected":
            value["environment"].pop("MS_PROVIDER_SELECTION_SHA256")
        elif mutation == "environment-extra":
            value["environment"]["MS_EXTRA"] = "value"
        elif mutation == "environment-value-drift":
            value["environment"]["MS_RUNTIME_SBOM_SHA256"] = "a" * 64
        elif mutation == "seed-out-of-range":
            value["seed"] = 10
        elif mutation == "arms-drift":
            value["arms"] = ["dense"]
        payload, digest = _rebind(value)
        with pytest.raises(RemoteIntentError):
            _validate_intent(payload, expected_sha256=digest)


def test_selected_bootstrap_binding_is_mode_consistent():
    for mutation in (
        "unknown-mode",
        "bootstrap-with-receipt",
        "reuse-without-receipt",
        "extra-field",
    ):
        value = deepcopy(_selected_intent(bootstrap_mode="reuse"))
        if mutation == "unknown-mode":
            value["bootstrap"]["mode"] = "refresh"
        elif mutation == "bootstrap-with-receipt":
            value["bootstrap"] = {
                "mode": "bootstrap",
                "receipt_sha256": "b" * 64,
            }
        elif mutation == "reuse-without-receipt":
            value["bootstrap"]["receipt_sha256"] = None
        elif mutation == "extra-field":
            value["bootstrap"]["extra"] = True
        payload, digest = _rebind(value)
        with pytest.raises(RemoteIntentError, match="bootstrap"):
            _validate_intent(payload, expected_sha256=digest)


def test_selected_lease_binds_reset_then_exact_systemd_unit():
    for mutation in (
        "missing-reset",
        "reset-after-lease",
        "unit-mismatch",
        "malformed-unit",
        "reset-argv-drift",
        "lease-deadline-drift",
    ):
        value = deepcopy(_selected_intent())
        steps = value["steps"]
        if mutation == "missing-reset":
            value["steps"] = steps[1:]
        elif mutation == "reset-after-lease":
            value["steps"] = [steps[1], steps[0], *steps[2:]]
        elif mutation == "unit-mismatch":
            value["lease_unit"] = (
                "memorysplit-auto-terminate-" + "a" * 64
            )
        elif mutation == "malformed-unit":
            value["lease_unit"] = "memorysplit-auto-terminate"
            steps[1]["argv"][2] = "memorysplit-auto-terminate"
        elif mutation == "reset-argv-drift":
            steps[0]["argv"] = [
                "/usr/bin/systemctl",
                "stop",
                "memorysplit-auto-terminate*.timer",
            ]
        elif mutation == "lease-deadline-drift":
            steps[1]["argv"][4] = "2099-01-01T00:00:00Z"
        payload, digest = _rebind(value)
        with pytest.raises(RemoteIntentError, match="lease|step|termination"):
            _validate_intent(payload, expected_sha256=digest)


def test_selected_step_orders_are_exact_per_operation_and_mode():
    for operation, bootstrap_mode, mutation in (
        ("submit", "bootstrap", "insert-legacy-staging"),
        ("submit", "reuse", "add-bootstrap"),
        ("resume", "reuse", "drop-materialization"),
        ("resume", "bootstrap", "drop-launcher-manifest"),
        ("submit", "bootstrap", "verify-in-bootstrap-mode"),
    ):
        value = deepcopy(
            _selected_intent(
                operation=operation,
                bootstrap_mode=bootstrap_mode,
                attempt=2 if operation == "resume" else 1,
            )
        )
        steps = value["steps"]
        if mutation == "insert-legacy-staging":
            steps.insert(
                2,
                {
                    "name": "materialize-dataset",
                    "argv": ["/usr/bin/env", "aws", "s3", "sync"],
                },
            )
        elif mutation == "add-bootstrap":
            steps.insert(
                3,
                {
                    "name": "bootstrap",
                    "argv": [
                        "/usr/bin/python3",
                        "/opt/memorysplit/cluster/aws/p5/bootstrap.py",
                    ],
                },
            )
        elif mutation == "drop-materialization":
            value["steps"] = [
                step
                for step in steps
                if step["name"] != "materialize-resume-receipt"
            ]
        elif mutation == "drop-launcher-manifest":
            value["steps"] = [
                step
                for step in steps
                if step["name"] != "build-launcher-manifest"
            ]
        elif mutation == "verify-in-bootstrap-mode":
            steps[2] = {
                "name": "verify-bootstrap-reuse",
                "argv": [
                    "/usr/bin/python3",
                    "/opt/memorysplit/cluster/aws/p5/bootstrap.py",
                    "--verify-reuse",
                ],
            }
        payload, digest = _rebind(value)
        with pytest.raises(RemoteIntentError, match="step"):
            _validate_intent(payload, expected_sha256=digest)


def test_legacy_schemas_reject_selected_fields_and_stay_byte_compatible():
    legacy = _intent()
    payload, digest = _payload(legacy)
    assert _validate_intent(payload, expected_sha256=digest) == legacy

    for field, value in (
        ("bootstrap", {"mode": "bootstrap", "receipt_sha256": None}),
        ("lease_unit", "memorysplit-auto-terminate-" + "a" * 64),
    ):
        mutated = deepcopy(legacy)
        mutated[field] = value
        mutated_payload, mutated_digest = _rebind(mutated)
        with pytest.raises(RemoteIntentError, match="field"):
            _validate_intent(mutated_payload, expected_sha256=mutated_digest)


def test_executor_propagates_image_without_ambient_credentials():
    intent = _intent()
    payload, digest = _payload(intent)

    class Store:
        def __init__(self):
            self.objects = {"s3://memorysplit-prod/intent.json": payload}

        def read(self, uri, *, expected_sha256=None):
            value = self.objects[uri]
            assert hashlib.sha256(value).hexdigest() == expected_sha256
            return value

        def read_if_exists(self, uri):
            return self.objects.get(uri)

        def put_if_absent(self, uri, payload, *, metadata):
            if uri in self.objects:
                return False
            self.objects[uri] = payload
            return True

    class Executor:
        def __init__(self):
            self.environments = []

        def run(self, argv, *, environment):
            self.environments.append(dict(environment))
            return 0

    executor = Executor()
    result = execute_intent(
        intent_uri="s3://memorysplit-prod/intent.json",
        intent_sha256=digest,
        store=Store(),
        executor=executor,
    )

    assert result["returncode"] == 0
    assert executor.environments
    assert all(environment["MS_CONTAINER_IMAGE"] == IMAGE for environment in executor.environments)
    assert all(environment["MS_CONTAINER_DIGEST"] == DIGEST for environment in executor.environments)
    assert all(
        not {
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_PROFILE",
            "AWS_CONFIG_FILE",
            "AWS_SHARED_CREDENTIALS_FILE",
        }
        & set(environment)
        for environment in executor.environments
    )
