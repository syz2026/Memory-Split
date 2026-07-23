from __future__ import annotations

import hashlib
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

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
        mutated_payload, mutated_digest = _payload(value)
        with pytest.raises(RemoteIntentError, match="environment|container|image"):
            _validate_intent(
                mutated_payload,
                expected_sha256=mutated_digest,
            )


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
