"""Execute one content-addressed AWS operation intent without a shell."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlsplit

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msctl.jsonutil import canonical_json


ARGV_DOCUMENT_NAME = "MemorySplit-ArgvV1"
_ARGV_DOCUMENT = {
    "description": "Execute one content-addressed MemorySplit argv intent.",
    "mainSteps": [
        {
            "action": "aws:runShellScript",
            "inputs": {
                "runCommand": [
                    (
                        "set -eu; umask 077; "
                        "R='/opt/memorysplit'; "
                        "set -- --intent-uri '{{ IntentUri }}' "
                        "--intent-sha256 '{{ IntentSHA256 }}'; "
                        "if [ '{{ BootstrapMode }}' = 'verified-control-bundle' ]; "
                        "then "
                        "B='/var/lib/memorysplit/control-"
                        "{{ ControlBundleSHA256 }}.tar'; "
                        "R='/opt/memorysplit/control/{{ ControlBundleSHA256 }}'; "
                        "/usr/bin/install -d -m 0700 /var/lib/memorysplit "
                        "/opt/memorysplit/control; "
                        "if [ -e \"$B\" ]; then "
                        "test -f \"$B\" && test ! -L \"$B\"; "
                        "else BT=\"$B.tmp.$$\"; test ! -e \"$BT\"; "
                        "/usr/bin/env aws --region '{{ Region }}' s3 cp "
                        "'{{ ControlBundleURI }}' \"$BT\" "
                        "--only-show-errors --no-progress; "
                        "printf '%s  %s\\n' '{{ ControlBundleSHA256 }}' \"$BT\" "
                        "| /usr/bin/sha256sum -c -; "
                        "/usr/bin/chmod 0400 \"$BT\"; "
                        "/usr/bin/ln \"$BT\" \"$B\"; /usr/bin/rm \"$BT\"; fi; "
                        "printf '%s  %s\\n' '{{ ControlBundleSHA256 }}' \"$B\" "
                        "| /usr/bin/sha256sum -c -; "
                        "if [ -e \"$R\" ]; then "
                        "test -d \"$R\" && test ! -L \"$R\"; "
                        "/usr/bin/tar --extract --to-stdout --file \"$B\" "
                        "CONTROL-BUNDLE.json | /usr/bin/cmp - "
                        "\"$R/CONTROL-BUNDLE.json\"; "
                        "/usr/bin/tar --extract --to-stdout --file \"$B\" "
                        "CONTROL-SHA256SUMS | /usr/bin/cmp - "
                        "\"$R/CONTROL-SHA256SUMS\"; "
                        "(cd \"$R\" && /usr/bin/sha256sum -c CONTROL-SHA256SUMS); "
                        "E=$(cd \"$R\" && { /usr/bin/awk '{print $2}' "
                        "CONTROL-SHA256SUMS; printf '%s\\n' CONTROL-BUNDLE.json "
                        "CONTROL-SHA256SUMS; } | /usr/bin/sort); "
                        "A=$(cd \"$R\" && /usr/bin/find . -type f -printf '%P\\n' "
                        "| /usr/bin/sort); test \"$A\" = \"$E\"; "
                        "test -z \"$(cd \"$R\" && /usr/bin/find . "
                        "! -type d ! -type f -print -quit)\"; "
                        "test -z \"$(cd \"$R\" && /usr/bin/find . -mindepth 1 "
                        "-perm /222 -print -quit)\"; "
                        "else T=\"$R.tmp.$$\"; test ! -e \"$T\"; "
                        "/usr/bin/install -d -m 0700 \"$T\"; "
                        "/usr/bin/tar --extract --file \"$B\" --directory \"$T\" "
                        "--no-same-owner --no-same-permissions; "
                        "(cd \"$T\" && /usr/bin/sha256sum -c CONTROL-SHA256SUMS); "
                        "/usr/bin/chmod -R a-w \"$T\"; "
                        "/usr/bin/mv -T -n \"$T\" \"$R\"; "
                        "test ! -e \"$T\"; fi; "
                        "set -- \"$@\" --control-bundle-sha256 "
                        "'{{ ControlBundleSHA256 }}'; fi; "
                        "test -f \"$R/msctl/aws_argv.py\" && "
                        "test ! -L \"$R/msctl/aws_argv.py\"; "
                        "PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 "
                        "\"$R/msctl/aws_argv.py\" \"$@\""
                    )
                ],
                "timeoutSeconds": "172800",
            },
            "name": "runContentAddressedArgv",
        }
    ],
    "parameters": {
        "BootstrapMode": {
            "allowedValues": ["installed", "verified-control-bundle"],
            "type": "String",
        },
        "ControlBundleSHA256": {
            "allowedPattern": "^[0-9a-f]{64}$",
            "type": "String",
        },
        "ControlBundleURI": {
            "allowedPattern": (
                "^s3://[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]/"
                "[A-Za-z0-9._/-]+$"
            ),
            "type": "String",
        },
        "IntentSHA256": {
            "allowedPattern": "^[0-9a-f]{64}$",
            "type": "String",
        },
        "IntentUri": {
            "allowedPattern": (
                "^s3://[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]/"
                "[A-Za-z0-9._/-]+$"
            ),
            "type": "String",
        },
        "Region": {
            "allowedValues": ["us-east-1", "us-west-2"],
            "type": "String",
        },
    },
    "schemaVersion": "2.2",
}
ARGV_DOCUMENT_CONTENT = canonical_json(_ARGV_DOCUMENT).decode("ascii")
ARGV_DOCUMENT_SHA256 = hashlib.sha256(
    ARGV_DOCUMENT_CONTENT.encode("ascii")
).hexdigest()
CANARY_INTENT_TYPE = "memorysplit-aws-gpu-canary-intent-v3"


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INSTANCE_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
_REGION_RE = re.compile(r"^[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+$")
_BUCKET_RE = re.compile(
    r"^(?![0-9]+(?:\.[0-9]+){3}$)(?!-)(?!.*\.\.)(?!.*\.-)(?!.*-\.)"
    r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$"
)
_NONCE_RE = re.compile(r"^[0-9a-f]{32}$")
_SAFE_PATH = "/usr/local/bin:/usr/bin:/bin"
_PROFILE_CONTRACTS = {
    "aws-p5.48xlarge": {
        "instance_type": "p5.48xlarge",
        "gres": "gpu:h100:8",
        "assigned_seeds": frozenset({1, 2, 3, 4}),
    },
    "aws-p5.48xlarge-v3": {
        "instance_type": "p5.48xlarge",
        "gres": "gpu:h100:8",
        "assigned_seeds": frozenset(range(10)),
    },
    "aws-p6-b300.48xlarge-v3": {
        "instance_type": "p6-b300.48xlarge",
        "gres": "gpu:b300:8",
        "assigned_seeds": frozenset(range(10)),
    },
}
_BASE_FIELDS = {
    "schema_version",
    "operation",
    "provider",
    "instance_type",
    "profile_sha256",
    "gres",
    "seed",
    "release_sha256",
    "run_manifest_sha256",
    "dataset_sha256",
    "dataset_pointer_sha256",
    "dataset_verification_sha256",
    "environment_receipt_sha256",
    "runtime_sha256",
    "environment",
    "checkpoint_receipt",
    "steps",
    "operation_id",
    "instance_id",
    "terminate_at",
    "ssm_document",
    "started_receipt_uri",
    "terminal_receipt_uri",
}
_CANARY_FIELDS = {
    "schema_version",
    "intent_type",
    "operation",
    "provider",
    "instance_type",
    "profile_sha256",
    "gres",
    "instance_id",
    "release_sha256",
    "provider_selection_sha256",
    "environment_receipt_sha256",
    "command_plan_sha256",
    "orchestration_plan_sha256",
    "orchestration_plan_uri",
    "qualification_receipt_uri",
    "control_bundle_sha256",
    "environment",
    "steps",
    "operation_id",
    "ssm_document",
    "started_receipt_uri",
    "terminal_receipt_uri",
}
_CANARY_RECEIPT_BINDING_FIELDS = {
    "intent_type",
    "release_sha256",
    "provider_selection_sha256",
    "environment_receipt_sha256",
    "command_plan_sha256",
    "orchestration_plan_sha256",
    "orchestration_plan_uri",
    "qualification_receipt_uri",
    "control_bundle_sha256",
}
_V3_PROVENANCE_FIELDS = {
    "cohort_assignment_sha256",
    "preregistration_sha256",
    "hardware_amendment_sha256",
    "provider_selection_sha256",
    "sealed_fixture_sha256",
    "fleet_plan_sha256",
    "fleet_wave",
    "launch_readiness_sha256",
    "control_bundle_sha256",
}
_V3_FINAL_EVALUATION_FIELDS = {
    "sealed_evaluation_sha256",
    "study_lock_sha256",
}
_V3_EXECUTION_FIELDS = {
    "container_image",
    "release_receipt_sha256",
    "release_members_sha256",
    "source_commit",
    "runs",
}
_ENVIRONMENT_FIELDS = {
    "AWS_REGION",
    "MS_AWS_AMI_ID",
    "MS_CONTAINER_DIGEST",
    "MS_RUNTIME_GID",
    "MS_RUNTIME_UID",
    "MS_S3_ROOT",
}
_V3_ENVIRONMENT_FIELDS = _ENVIRONMENT_FIELDS | {"MS_S3_KMS_KEY_ID"}
_CHECKPOINT_RECEIPT_FIELDS = {"sha256", "checkpoints"}
_CHECKPOINT_FIELDS = {
    "arm",
    "resume_path",
    "resume_sha256",
    "world_size",
}
_EVALUATION_CHECKPOINT_RECEIPT_FIELDS = {"sha256", "uri", "checkpoints"}
_EVALUATION_CHECKPOINT_FIELDS = {
    "run_id",
    "arm",
    "checkpoint_sha256",
    "checkpoint_uri",
    "configuration_sha256",
    "configuration_uri",
    "run_binding_sha256",
    "run_binding_uri",
    "checkpoint_record_sha256",
    "checkpoint_record_uri",
}
_RUN_FIELDS = {"run_id", "arm", "config", "config_sha256"}
_FORBIDDEN_ENVIRONMENT = {
    "AWS_ACCESS_KEY_ID",
    "AWS_CONFIG_FILE",
    "AWS_PROFILE",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
}
_SHELL_INTERPRETERS = {
    "/bin/bash",
    "/bin/dash",
    "/bin/sh",
    "/usr/bin/bash",
    "/usr/bin/dash",
    "/usr/bin/sh",
}
_SEALED_EVALUATION_MEMBERS = (
    "checkpoints.jsonl",
    "items.jsonl",
    "sealed-gold.jsonl",
    "stores.jsonl",
    "study-lock.json",
    "validity.json",
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ECR_IMAGE_RE = re.compile(
    r"^[0-9]{12}\.dkr\.ecr\."
    r"[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+"
    r"\.amazonaws\.com(?:\.cn)?/"
    r"[a-z0-9]+(?:[._/-][a-z0-9]+)*@sha256:[0-9a-f]{64}$"
)


class RemoteIntentError(ValueError):
    """A remote intent or its execution boundary is invalid."""


def _v3_provenance_fields(value: Mapping[str, object]) -> set[str]:
    fields = set(_V3_PROVENANCE_FIELDS)
    if value.get("operation") == "evaluate":
        fields |= _V3_FINAL_EVALUATION_FIELDS
    return fields


class ImmutableObjectStore(Protocol):
    def read(self, uri: str, *, expected_sha256: str | None = None) -> bytes: ...

    def read_if_exists(self, uri: str) -> bytes | None: ...

    def put_if_absent(
        self,
        uri: str,
        payload: bytes,
        *,
        metadata: Mapping[str, str],
    ) -> bool: ...


class ArgvExecutor(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        environment: Mapping[str, str],
    ) -> int: ...


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RemoteIntentError(f"JSON repeats field: {key}")
        result[key] = value
    return result


def _decode_object(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                RemoteIntentError(f"{label} contains non-finite {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteIntentError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise RemoteIntentError(f"{label} must be a JSON object")
    return value


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RemoteIntentError(f"{label} must be lowercase SHA-256")
    return value


def _s3_uri(value: object, *, root: str | None = None) -> str:
    if not isinstance(value, str):
        raise RemoteIntentError("S3 URI must be a string")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise RemoteIntentError("S3 URI is invalid") from error
    parts = parsed.path.removeprefix("/").split("/")
    if (
        parsed.scheme != "s3"
        or parsed.hostname is None
        or parsed.netloc != parsed.hostname
        or port is not None
        or parsed.query
        or parsed.fragment
        or _BUCKET_RE.fullmatch(parsed.hostname) is None
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
        or "\\" in value
        or (root is not None and not value.startswith(root.rstrip("/") + "/"))
    ):
        raise RemoteIntentError("S3 URI is outside the immutable object namespace")
    return value


def _validate_checkpoint_receipt(value: object) -> None:
    if not isinstance(value, dict) or set(value) != _CHECKPOINT_RECEIPT_FIELDS:
        raise RemoteIntentError(
            "checkpoint receipt fields do not match the contract"
        )
    receipt_sha256 = _sha256(
        value["sha256"],
        label="checkpoint receipt",
    )
    checkpoints = value["checkpoints"]
    if not isinstance(checkpoints, list) or len(checkpoints) != 2:
        raise RemoteIntentError(
            "checkpoint receipt must bind one complete pair"
        )
    arms: set[str] = set()
    ordered_arms: list[str] = []
    for row in checkpoints:
        if not isinstance(row, dict) or set(row) != _CHECKPOINT_FIELDS:
            raise RemoteIntentError(
                "checkpoint binding fields do not match the contract"
            )
        arm = row["arm"]
        digest = _sha256(
            row["resume_sha256"],
            label=f"{arm} checkpoint",
        )
        expected_path = (
            "/mnt/memorysplit/staging/resume/"
            f"{receipt_sha256}/{arm}.pt"
        )
        if (
            arm not in {"dense", "split90"}
            or arm in arms
            or row["resume_path"] != expected_path
            or type(row["world_size"]) is not int
            or row["world_size"] != 4
            or not digest
        ):
            raise RemoteIntentError(
                "checkpoint binding identity is invalid"
            )
        arms.add(str(arm))
        ordered_arms.append(str(arm))
    if arms != {"dense", "split90"} or ordered_arms != ["dense", "split90"]:
        raise RemoteIntentError("checkpoint binding pair is incomplete")


def _validate_evaluation_checkpoint_receipt(
    value: object,
    *,
    seed: int,
    s3_root: str,
    runs: Mapping[str, Mapping[str, object]],
) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != _EVALUATION_CHECKPOINT_RECEIPT_FIELDS
    ):
        raise RemoteIntentError(
            "evaluation checkpoint receipt fields do not match the contract"
        )
    receipt_sha256 = _sha256(
        value["sha256"],
        label="evaluation checkpoint receipt",
    )
    expected_receipt_uri = (
        f"{s3_root}/checkpoints/seed-{seed}/receipts/"
        f"{receipt_sha256}.json"
    )
    if (
        _s3_uri(value["uri"], root=s3_root) != expected_receipt_uri
    ):
        raise RemoteIntentError(
            "evaluation checkpoint receipt URI is not canonical"
        )
    checkpoints = value["checkpoints"]
    if not isinstance(checkpoints, list) or len(checkpoints) != 2:
        raise RemoteIntentError(
            "evaluation checkpoint receipt must bind one complete pair"
        )
    arms: set[str] = set()
    ordered_arms: list[str] = []
    run_ids: set[str] = set()
    for row in checkpoints:
        if (
            not isinstance(row, dict)
            or set(row) != _EVALUATION_CHECKPOINT_FIELDS
        ):
            raise RemoteIntentError(
                "evaluation checkpoint artifact fields do not match"
            )
        arm = row["arm"]
        run_id = row["run_id"]
        if (
            arm not in {"dense", "split90"}
            or arm in arms
            or not isinstance(run_id, str)
            or _RUN_ID_RE.fullmatch(run_id) is None
            or run_id in run_ids
            or arm not in runs
            or run_id != runs[arm]["run_id"]
        ):
            raise RemoteIntentError(
                "evaluation checkpoint artifact identity is invalid"
            )
        checkpoint_sha256 = _sha256(
            row["checkpoint_sha256"],
            label=f"{arm} evaluation checkpoint",
        )
        configuration_sha256 = _sha256(
            row["configuration_sha256"],
            label=f"{arm} evaluation configuration",
        )
        if configuration_sha256 != runs[arm]["config_sha256"]:
            raise RemoteIntentError(
                "evaluation configuration does not match the run manifest binding"
            )
        run_binding_sha256 = _sha256(
            row["run_binding_sha256"],
            label=f"{arm} evaluation run binding",
        )
        record_sha256 = _sha256(
            row["checkpoint_record_sha256"],
            label=f"{arm} evaluation checkpoint record",
        )
        prefix = f"{s3_root}/checkpoints/seed-{seed}/{arm}"
        expected_uris = {
            "checkpoint_uri": (
                f"{prefix}/sha256/{checkpoint_sha256}.pt"
            ),
            "configuration_uri": (
                f"{prefix}/configuration/sha256/"
                f"{configuration_sha256}.yaml"
            ),
            "run_binding_uri": (
                f"{prefix}/run-binding/sha256/"
                f"{run_binding_sha256}.json"
            ),
            "checkpoint_record_uri": (
                f"{prefix}/records/{record_sha256}.json"
            ),
        }
        if any(
            _s3_uri(row[field], root=s3_root) != expected_uri
            for field, expected_uri in expected_uris.items()
        ):
            raise RemoteIntentError(
                "evaluation checkpoint artifact URI is not canonical"
            )
        arms.add(str(arm))
        ordered_arms.append(str(arm))
        run_ids.add(run_id)
    if arms != {"dense", "split90"} or ordered_arms != ["dense", "split90"]:
        raise RemoteIntentError(
            "evaluation checkpoint artifact pair is incomplete"
        )


def _validate_canary_intent(
    intent: dict[str, object],
    *,
    expected_control_bundle_sha256: str | None,
) -> dict[str, object]:
    if set(intent) != _CANARY_FIELDS:
        raise RemoteIntentError("canary intent fields do not match the contract")
    provider = intent["provider"]
    profile_contract = (
        _PROFILE_CONTRACTS.get(provider)
        if isinstance(provider, str)
        else None
    )
    if (
        intent["schema_version"] != 3
        or intent["intent_type"] != CANARY_INTENT_TYPE
        or intent["operation"] != "canary"
        or provider
        not in {"aws-p5.48xlarge-v3", "aws-p6-b300.48xlarge-v3"}
        or profile_contract is None
        or intent["instance_type"] != profile_contract["instance_type"]
        or intent["gres"] != profile_contract["gres"]
        or not isinstance(intent["profile_sha256"], str)
        or _SHA256_RE.fullmatch(intent["profile_sha256"]) is None
        or not isinstance(intent["instance_id"], str)
        or _INSTANCE_RE.fullmatch(intent["instance_id"]) is None
    ):
        raise RemoteIntentError("canary intent identity is invalid")
    for field in (
        "release_sha256",
        "provider_selection_sha256",
        "environment_receipt_sha256",
        "command_plan_sha256",
        "orchestration_plan_sha256",
        "control_bundle_sha256",
        "operation_id",
    ):
        _sha256(intent[field], label=f"canary intent {field}")
    if (
        expected_control_bundle_sha256 is None
        or intent["control_bundle_sha256"]
        != _sha256(
            expected_control_bundle_sha256,
            label="installed control bundle",
        )
    ):
        raise RemoteIntentError(
            "canary intent control bundle does not match the installed bytes"
        )

    identity = {
        key: value
        for key, value in intent.items()
        if key
        not in {
            "operation_id",
            "ssm_document",
            "started_receipt_uri",
            "terminal_receipt_uri",
        }
    }
    if hashlib.sha256(canonical_json(identity)).hexdigest() != intent["operation_id"]:
        raise RemoteIntentError("canary operation ID does not match its identity")

    environment = intent["environment"]
    if (
        not isinstance(environment, dict)
        or set(environment) != _ENVIRONMENT_FIELDS
        or set(environment) & _FORBIDDEN_ENVIRONMENT
        or not all(isinstance(value, str) and value for value in environment.values())
        or environment["AWS_REGION"] not in {"us-east-1", "us-west-2"}
        or re.fullmatch(r"^ami-[0-9a-f]{8,17}$", environment["MS_AWS_AMI_ID"])
        is None
        or re.fullmatch(
            r"^sha256:[0-9a-f]{64}$",
            environment["MS_CONTAINER_DIGEST"],
        )
        is None
        or any(
            not environment[field].isdigit() or int(environment[field]) <= 0
            for field in ("MS_RUNTIME_GID", "MS_RUNTIME_UID")
        )
    ):
        raise RemoteIntentError(
            "canary environment is not closed, pinned, and credential-free"
        )
    s3_root = _s3_uri(environment["MS_S3_ROOT"])
    plan_uri = _s3_uri(intent["orchestration_plan_uri"], root=s3_root)
    qualification_uri = _s3_uri(
        intent["qualification_receipt_uri"],
        root=s3_root,
    )
    expected_plan_uri = (
        f"{s3_root}/qualification/plans/sha256/"
        f"{intent['orchestration_plan_sha256']}.json"
    )
    expected_qualification_uri = (
        f"{s3_root}/qualification/{provider}/{intent['instance_id']}/"
        f"environment-{intent['environment_receipt_sha256']}/"
        f"plan-{intent['command_plan_sha256']}.json"
    )
    if (
        plan_uri != expected_plan_uri
        or qualification_uri != expected_qualification_uri
    ):
        raise RemoteIntentError("canary object paths are not deterministic")

    receipt_root = (
        f"{s3_root}/operations/{intent['operation_id']}/receipts"
    )
    if (
        _s3_uri(intent["started_receipt_uri"], root=s3_root)
        != f"{receipt_root}/started.json"
        or _s3_uri(intent["terminal_receipt_uri"], root=s3_root)
        != f"{receipt_root}/terminal.json"
    ):
        raise RemoteIntentError("canary receipt paths do not match the operation")
    if intent["ssm_document"] != {
        "name": ARGV_DOCUMENT_NAME,
        "sha256": ARGV_DOCUMENT_SHA256,
    }:
        raise RemoteIntentError("canary intent names the wrong SSM document")

    release_root = f"/mnt/memorysplit/releases/{intent['release_sha256']}"
    expected_argv = [
        "/usr/bin/python3",
        f"{release_root}/cluster/aws/p5/canary_orchestrator.py",
        "--plan-uri",
        plan_uri,
        "--plan-sha256",
        str(intent["orchestration_plan_sha256"]),
        "--region",
        str(environment["AWS_REGION"]),
    ]
    if intent["steps"] != [
        {
            "name": "execute-canary-orchestration",
            "argv": expected_argv,
        }
    ]:
        raise RemoteIntentError(
            "canary intent does not contain the exact release-mounted argv"
        )
    return intent


def _required_option(argv: list[str], option: str, expected: str) -> None:
    if argv.count(option) != 1:
        raise RemoteIntentError(f"v3 operation step must bind {option} exactly once")
    index = argv.index(option)
    if index + 1 >= len(argv) or argv[index + 1] != expected:
        raise RemoteIntentError(f"v3 operation step has a stale {option} binding")


def _s3_location(root: str, suffix: str) -> tuple[str, str]:
    parsed = urlsplit(root)
    prefix = parsed.path.removeprefix("/").rstrip("/")
    key = f"{prefix}/{suffix}" if prefix else suffix
    return str(parsed.hostname), key


def _aws_get_argv(
    *,
    region: str,
    s3_root: str,
    suffix: str,
    destination: str,
) -> list[str]:
    bucket, key = _s3_location(s3_root, suffix)
    return [
        "/usr/bin/env",
        "aws",
        "--no-cli-pager",
        "--region",
        region,
        "s3api",
        "get-object",
        "--bucket",
        bucket,
        "--key",
        key,
        "--checksum-mode",
        "ENABLED",
        destination,
    ]


def build_v3_evaluation_steps(
    intent: Mapping[str, object],
) -> list[dict[str, object]]:
    """Build the single canonical schema-v3 remote evaluation plan."""

    environment = cast(Mapping[str, object], intent["environment"])
    checkpoint = cast(Mapping[str, object], intent["checkpoint_receipt"])
    checkpoint_rows = cast(list[Mapping[str, object]], checkpoint["checkpoints"])
    rows = {str(row["arm"]): row for row in checkpoint_rows}
    run_rows = {
        str(row["arm"]): row
        for row in cast(list[Mapping[str, object]], intent["runs"])
    }
    scratch = "/mnt/memorysplit"
    release_root = f"{scratch}/releases/{intent['release_sha256']}"
    sealed_root = (
        f"{scratch}/sealed-evaluation/{intent['sealed_evaluation_sha256']}"
    )
    evaluation_root = f"{scratch}/evaluations"
    receipt_root = f"{scratch}/receipts/checkpoints/seed-{intent['seed']}"
    record_root = f"{receipt_root}/records"
    run_root = f"{scratch}/runs/seed-{intent['seed']}"
    region = str(environment["AWS_REGION"])
    s3_root = str(environment["MS_S3_ROOT"])
    uid = str(environment["MS_RUNTIME_UID"])
    gid = str(environment["MS_RUNTIME_GID"])
    image = str(intent["container_image"])

    steps: list[dict[str, object]] = [
        {
            "name": "prepare-sealed-evaluation",
            "argv": [
                "/usr/bin/install",
                "-d",
                "-m",
                "0700",
                "-o",
                uid,
                "-g",
                gid,
                sealed_root,
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
                uid,
                "-g",
                gid,
                evaluation_root,
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
                uid,
                "-g",
                gid,
                receipt_root,
                record_root,
                f"{run_root}/dense/run",
                f"{run_root}/split90/run",
            ],
        },
    ]
    steps.extend(
        {
            "name": f"materialize-sealed-evaluation-{member.replace('.', '-')}",
            "argv": _aws_get_argv(
                region=region,
                s3_root=s3_root,
                suffix=(
                    f"sealed-evaluation/{intent['sealed_evaluation_sha256']}/"
                    f"{member}"
                ),
                destination=f"{sealed_root}/{member}",
            ),
        }
        for member in _SEALED_EVALUATION_MEMBERS
    )

    terminal_objects: list[tuple[str, str, str]] = [
        (
            "receipt",
            (
                f"checkpoints/seed-{intent['seed']}/receipts/"
                f"{checkpoint['sha256']}.json"
            ),
            f"{receipt_root}/receipt.json",
        )
    ]
    for arm in ("dense", "split90"):
        row = rows[arm]
        arm_root = f"{run_root}/{arm}/run"
        terminal_objects.extend(
            [
                (
                    f"{arm}-checkpoint",
                    (
                        f"checkpoints/seed-{intent['seed']}/{arm}/sha256/"
                        f"{row['checkpoint_sha256']}.pt"
                    ),
                    f"{arm_root}/ckpt.pt",
                ),
                (
                    f"{arm}-configuration",
                    (
                        f"checkpoints/seed-{intent['seed']}/{arm}/"
                        f"configuration/sha256/{row['configuration_sha256']}.yaml"
                    ),
                    f"{arm_root}/configuration.yaml",
                ),
                (
                    f"{arm}-run-binding",
                    (
                        f"checkpoints/seed-{intent['seed']}/{arm}/"
                        f"run-binding/sha256/{row['run_binding_sha256']}.json"
                    ),
                    f"{arm_root}/run.json",
                ),
                (
                    f"{arm}-checkpoint-record",
                    (
                        f"checkpoints/seed-{intent['seed']}/{arm}/records/"
                        f"{row['checkpoint_record_sha256']}.json"
                    ),
                    (
                        f"{record_root}/"
                        f"{row['checkpoint_record_sha256']}.json"
                    ),
                ),
            ]
        )
    steps.extend(
        {
            "name": f"materialize-terminal-{name}",
            "argv": _aws_get_argv(
                region=region,
                s3_root=s3_root,
                suffix=suffix,
                destination=destination,
            ),
        }
        for name, suffix, destination in terminal_objects
    )
    steps.extend(
        [
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
                    f"{uid}:{gid}",
                    "--mount",
                    f"type=bind,src={release_root},dst=/workspace,readonly",
                    "--mount",
                    f"type=bind,src={sealed_root},dst=/sealed,readonly",
                    "--workdir",
                    "/workspace",
                    image,
                    "/opt/venv/bin/python",
                    "-m",
                    "msctl.aws_sealed_evaluation",
                    "--root",
                    "/sealed",
                    "--expected-release-sha256",
                    str(intent["sealed_evaluation_sha256"]),
                    "--expected-study-lock-sha256",
                    str(intent["study_lock_sha256"]),
                ],
            },
            {
                "name": "verify-terminal-checkpoints",
                "argv": [
                    "/usr/bin/python3",
                    f"{release_root}/cluster/aws/p5/terminal_artifacts.py",
                    "verify-materialized",
                    "--receipt",
                    f"{receipt_root}/receipt.json",
                    "--run-root",
                    run_root,
                    "--record-root",
                    record_root,
                    "--expected-receipt-sha256",
                    str(checkpoint["sha256"]),
                    "--provider",
                    str(intent["provider"]),
                    "--run-manifest-sha256",
                    str(intent["run_manifest_sha256"]),
                    "--sealed-fixture-sha256",
                    str(intent["sealed_fixture_sha256"]),
                ],
            },
        ]
    )
    for arm in ("dense", "split90"):
        row = run_rows[arm]
        steps.append(
            {
                "name": f"evaluate-{arm}",
                "argv": [
                    "/usr/bin/docker",
                    "run",
                    "--rm",
                    "--read-only",
                    "--network",
                    "none",
                    "--gpus",
                    "all",
                    "--user",
                    f"{uid}:{gid}",
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
                    image,
                    "/opt/venv/bin/python",
                    "-m",
                    "evals.confirmatory",
                    "evaluate",
                    "--run",
                    f"{run_root}/{arm}/run",
                    "--sealed-release",
                    "/sealed",
                    "--expected-study-lock-sha256",
                    str(intent["study_lock_sha256"]),
                    "--device",
                    "cuda",
                    "--output-dir",
                    f"{evaluation_root}/{row['run_id']}",
                ],
            }
        )
    steps.append(
        {
            "name": "publish-paired-evaluation",
            "argv": [
                "/usr/bin/python3",
                f"{release_root}/cluster/aws/p5/terminal_artifacts.py",
                "publish-evaluation",
                "--evaluation-root",
                evaluation_root,
                "--checkpoint-receipt",
                f"{receipt_root}/receipt.json",
                "--s3-root",
                s3_root,
                "--provider",
                str(intent["provider"]),
                "--run-manifest-sha256",
                str(intent["run_manifest_sha256"]),
                "--sealed-fixture-sha256",
                str(intent["sealed_fixture_sha256"]),
                "--sealed-evaluation-sha256",
                str(intent["sealed_evaluation_sha256"]),
                "--study-lock-sha256",
                str(intent["study_lock_sha256"]),
                "--fleet-plan-sha256",
                str(intent["fleet_plan_sha256"]),
                "--fleet-wave",
                str(intent["fleet_wave"]),
                "--launch-readiness-sha256",
                str(intent["launch_readiness_sha256"]),
            ],
        }
    )
    return steps


def build_v3_training_steps(
    intent: Mapping[str, object],
) -> list[dict[str, object]]:
    """Build the single canonical schema-v3 submit or resume plan."""

    operation = str(intent["operation"])
    if operation not in {"submit", "resume"}:
        raise RemoteIntentError("v3 training plan operation is invalid")
    environment = cast(Mapping[str, object], intent["environment"])
    scratch = "/mnt/memorysplit"
    staging = f"{scratch}/staging"
    release_root = f"{scratch}/releases/{intent['release_sha256']}"
    control_root = f"/opt/memorysplit/control/{intent['control_bundle_sha256']}"
    provider = str(intent["provider"])
    profile_path = f"{release_root}/cluster/profiles/{provider}.json"
    region = str(environment["AWS_REGION"])
    s3_root = str(environment["MS_S3_ROOT"])
    run_rows = cast(list[Mapping[str, object]], intent["runs"])

    if operation == "resume":
        checkpoint = cast(Mapping[str, object], intent["checkpoint_receipt"])
        receipt_sha256 = str(checkpoint["sha256"])
        resume_root = f"{staging}/resume/{receipt_sha256}"
        steps: list[dict[str, object]] = [
            {
                "name": "prepare-resume-staging",
                "argv": [
                    "/usr/bin/install",
                    "-d",
                    "-m",
                    "0700",
                    resume_root,
                ],
            },
            {
                "name": "materialize-resume-receipt",
                "argv": _aws_get_argv(
                    region=region,
                    s3_root=s3_root,
                    suffix=(
                        f"checkpoints/seed-{intent['seed']}/receipts/"
                        f"{receipt_sha256}.json"
                    ),
                    destination=f"{resume_root}/receipt.json",
                ),
            },
        ]
        checkpoint_rows = cast(
            list[Mapping[str, object]],
            checkpoint["checkpoints"],
        )
        for row in checkpoint_rows:
            arm = str(row["arm"])
            steps.append(
                {
                    "name": f"materialize-resume-{arm}",
                    "argv": _aws_get_argv(
                        region=region,
                        s3_root=s3_root,
                        suffix=(
                            f"checkpoints/seed-{intent['seed']}/{arm}/sha256/"
                            f"{row['resume_sha256']}.pt"
                        ),
                        destination=str(row["resume_path"]),
                    ),
                }
            )
        launcher_argv = [
            "/usr/bin/python3",
            f"{release_root}/msctl/aws_resume_launch.py",
            "--seed",
            str(intent["seed"]),
            "--manifest",
            f"{staging}/launcher-manifest-{intent['run_manifest_sha256']}.json",
            "--profile",
            profile_path,
            "--repo-root",
            release_root,
            "--scratch-root",
            scratch,
            "--launcher",
            f"{release_root}/cluster/aws/p5/launch_seed_pair.py",
            "--checkpoint-receipt",
            f"{resume_root}/receipt.json",
            "--checkpoint-receipt-sha256",
            receipt_sha256,
            "--run-manifest-sha256",
            str(intent["run_manifest_sha256"]),
        ]
        for row in checkpoint_rows:
            launcher_argv.extend(
                ["--checkpoint", canonical_json(row).decode("ascii")]
            )
        launcher_argv.append("--apply")
        steps.append({"name": "paired-launch", "argv": launcher_argv})
        return steps

    steps = [
        {
            "name": "auto-termination",
            "argv": [
                "/usr/bin/systemd-run",
                "--unit",
                "memorysplit-auto-terminate",
                "--on-calendar",
                str(intent["terminate_at"]),
                "/sbin/shutdown",
                "-h",
                "now",
            ],
        },
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
                "/var/lib/memorysplit/aws-private-home",
            ],
        },
        {
            "name": "prepare-staging",
            "argv": [
                "/usr/bin/install",
                "-d",
                "-m",
                "0700",
                f"{staging}/releases/{intent['release_sha256']}",
                f"{scratch}/dataset",
            ],
        },
        {
            "name": "materialize-release-archive",
            "argv": _aws_get_argv(
                region=region,
                s3_root=s3_root,
                suffix=f"releases/{intent['release_sha256']}/release.zip",
                destination=(
                    f"{staging}/releases/{intent['release_sha256']}/release.zip"
                ),
            ),
        },
        {
            "name": "materialize-release-receipt",
            "argv": _aws_get_argv(
                region=region,
                s3_root=s3_root,
                suffix=f"releases/{intent['release_sha256']}/RELEASE.json",
                destination=(
                    f"{staging}/releases/{intent['release_sha256']}/RELEASE.json"
                ),
            ),
        },
        {
            "name": "materialize-cohort-assignment",
            "argv": _aws_get_argv(
                region=region,
                s3_root=s3_root,
                suffix=(
                    f"releases/{intent['release_sha256']}/"
                    "cohort-assignment-v3.json"
                ),
                destination=(
                    f"{staging}/releases/{intent['release_sha256']}/"
                    "cohort-assignment-v3.json"
                ),
            ),
        },
        {
            "name": "materialize-dataset",
            "argv": [
                "/usr/bin/env",
                "aws",
                "--no-cli-pager",
                "--region",
                region,
                "s3",
                "sync",
                f"{s3_root}/dataset",
                f"{scratch}/dataset",
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
                f"{control_root}/cluster/profiles/{provider}.json",
                "--container-image",
                str(intent["container_image"]),
                "--release-archive",
                f"{staging}/releases/{intent['release_sha256']}/release.zip",
                "--release-sha256",
                str(intent["release_sha256"]),
                "--release-receipt",
                f"{staging}/releases/{intent['release_sha256']}/RELEASE.json",
                "--release-receipt-sha256",
                str(intent["release_receipt_sha256"]),
                "--dataset-receipt",
                f"{scratch}/dataset/receipt.json",
                "--dataset-receipt-sha256",
                str(intent["dataset_sha256"]),
                "--cohort-assignment",
                (
                    f"{staging}/releases/{intent['release_sha256']}/"
                    "cohort-assignment-v3.json"
                ),
                "--cohort-assignment-sha256",
                str(intent["cohort_assignment_sha256"]),
                "--code-commit",
                str(intent["source_commit"]),
                "--receipt",
                f"{staging}/bootstrap-receipt.json",
                "--owner-uid",
                str(environment["MS_RUNTIME_UID"]),
                "--owner-gid",
                str(environment["MS_RUNTIME_GID"]),
                "--aws-private-home",
                "/var/lib/memorysplit/aws-private-home",
                "--authorize-destructive-instance-store",
                "--apply",
            ],
        },
    ]
    launcher_manifest_argv = [
        "/usr/bin/python3",
        f"{release_root}/msctl/aws_launch_manifest.py",
        "--profile",
        profile_path,
        "--out",
        f"{staging}/launcher-manifest-{intent['run_manifest_sha256']}.json",
        "--scratch-root",
        scratch,
        "--seed",
        str(intent["seed"]),
        "--profile-sha256",
        str(intent["profile_sha256"]),
        "--release-sha256",
        str(intent["release_sha256"]),
        "--release-members-sha256",
        str(intent["release_members_sha256"]),
        "--cohort-assignment-sha256",
        str(intent["cohort_assignment_sha256"]),
        "--code-commit",
        str(intent["source_commit"]),
        "--preregistration-sha256",
        str(intent["preregistration_sha256"]),
        "--hardware-amendment-sha256",
        str(intent["hardware_amendment_sha256"]),
        "--provider-selection-sha256",
        str(intent["provider_selection_sha256"]),
        "--sealed-fixture-sha256",
        str(intent["sealed_fixture_sha256"]),
        "--fleet-plan-sha256",
        str(intent["fleet_plan_sha256"]),
        "--launch-readiness-sha256",
        str(intent["launch_readiness_sha256"]),
        "--control-bundle-sha256",
        str(intent["control_bundle_sha256"]),
        "--run-manifest-sha256",
        str(intent["run_manifest_sha256"]),
        "--fleet-wave",
        str(intent["fleet_wave"]),
        "--bootstrap-receipt",
        f"{staging}/bootstrap-receipt.json",
        "--corpus-receipt",
        f"{scratch}/dataset/receipt.json",
    ]
    for row in run_rows:
        launcher_manifest_argv.extend(
            [
                "--run",
                canonical_json(
                    {
                        "arm": row["arm"],
                        "config": row["config"],
                        "config_sha256": row["config_sha256"],
                    }
                ).decode("ascii"),
            ]
        )
    steps.extend(
        [
            {
                "name": "build-launcher-manifest",
                "argv": launcher_manifest_argv,
            },
            {
                "name": "paired-launch",
                "argv": [
                    "/usr/bin/python3",
                    f"{release_root}/cluster/aws/p5/launch_seed_pair.py",
                    "--seed",
                    str(intent["seed"]),
                    "--manifest",
                    (
                        f"{staging}/launcher-manifest-"
                        f"{intent['run_manifest_sha256']}.json"
                    ),
                    "--profile",
                    profile_path,
                    "--repo-root",
                    release_root,
                    "--scratch-root",
                    scratch,
                    "--apply",
                ],
            },
        ]
    )
    return steps


def build_v3_operation_steps(
    intent: Mapping[str, object],
) -> list[dict[str, object]]:
    """Build the canonical remote argv plan for one schema-v3 operation."""

    if intent.get("operation") == "evaluate":
        return build_v3_evaluation_steps(intent)
    return build_v3_training_steps(intent)


def _validate_v3_python_step(
    *,
    name: str,
    argv: list[str],
    intent: Mapping[str, object],
    release_root: str,
    control_root: str,
) -> None:
    operation = str(intent["operation"])
    scratch = "/mnt/memorysplit"
    staging = f"{scratch}/staging"
    provider = str(intent["provider"])
    profile_path = f"{release_root}/cluster/profiles/{provider}.json"
    checkpoint = cast(dict[str, object] | None, intent["checkpoint_receipt"])
    run_rows = cast(list[dict[str, object]], intent["runs"])
    expected: list[str]
    if name == "bootstrap":
        expected = [
            "/usr/bin/python3",
            f"{control_root}/cluster/aws/p5/bootstrap.py",
            "--profile",
            f"{control_root}/cluster/profiles/{provider}.json",
            "--container-image",
            str(intent["container_image"]),
            "--release-archive",
            f"{staging}/releases/{intent['release_sha256']}/release.zip",
            "--release-sha256",
            str(intent["release_sha256"]),
            "--release-receipt",
            f"{staging}/releases/{intent['release_sha256']}/RELEASE.json",
            "--release-receipt-sha256",
            str(intent["release_receipt_sha256"]),
            "--dataset-receipt",
            f"{scratch}/dataset/receipt.json",
            "--dataset-receipt-sha256",
            str(intent["dataset_sha256"]),
            "--cohort-assignment",
            (
                f"{staging}/releases/{intent['release_sha256']}/"
                "cohort-assignment-v3.json"
            ),
            "--cohort-assignment-sha256",
            str(intent["cohort_assignment_sha256"]),
            "--code-commit",
            str(intent["source_commit"]),
            "--receipt",
            f"{staging}/bootstrap-receipt.json",
            "--owner-uid",
            str(cast(dict[str, object], intent["environment"])["MS_RUNTIME_UID"]),
            "--owner-gid",
            str(cast(dict[str, object], intent["environment"])["MS_RUNTIME_GID"]),
            "--aws-private-home",
            "/var/lib/memorysplit/aws-private-home",
            "--authorize-destructive-instance-store",
            "--apply",
        ]
    elif name == "build-launcher-manifest":
        expected = [
            "/usr/bin/python3",
            f"{release_root}/msctl/aws_launch_manifest.py",
            "--profile",
            profile_path,
            "--out",
            f"{staging}/launcher-manifest-{intent['run_manifest_sha256']}.json",
            "--scratch-root",
            scratch,
            "--seed",
            str(intent["seed"]),
            "--profile-sha256",
            str(intent["profile_sha256"]),
            "--release-sha256",
            str(intent["release_sha256"]),
            "--release-members-sha256",
            str(intent["release_members_sha256"]),
            "--cohort-assignment-sha256",
            str(intent["cohort_assignment_sha256"]),
            "--code-commit",
            str(intent["source_commit"]),
            "--preregistration-sha256",
            str(intent["preregistration_sha256"]),
            "--hardware-amendment-sha256",
            str(intent["hardware_amendment_sha256"]),
            "--provider-selection-sha256",
            str(intent["provider_selection_sha256"]),
            "--sealed-fixture-sha256",
            str(intent["sealed_fixture_sha256"]),
            "--fleet-plan-sha256",
            str(intent["fleet_plan_sha256"]),
            "--launch-readiness-sha256",
            str(intent["launch_readiness_sha256"]),
            "--control-bundle-sha256",
            str(intent["control_bundle_sha256"]),
            "--run-manifest-sha256",
            str(intent["run_manifest_sha256"]),
            "--fleet-wave",
            str(intent["fleet_wave"]),
            "--bootstrap-receipt",
            f"{staging}/bootstrap-receipt.json",
            "--corpus-receipt",
            f"{scratch}/dataset/receipt.json",
        ]
        for row in run_rows:
            expected.extend(
                [
                    "--run",
                    canonical_json(
                        {
                            "arm": row["arm"],
                            "config": row["config"],
                            "config_sha256": row["config_sha256"],
                        }
                    ).decode("ascii"),
                ]
            )
    elif name == "paired-launch":
        expected = [
            "/usr/bin/python3",
            (
                f"{release_root}/msctl/aws_resume_launch.py"
                if operation == "resume"
                else f"{release_root}/cluster/aws/p5/launch_seed_pair.py"
            ),
            "--seed",
            str(intent["seed"]),
            "--manifest",
            f"{staging}/launcher-manifest-{intent['run_manifest_sha256']}.json",
            "--profile",
            profile_path,
            "--repo-root",
            release_root,
            "--scratch-root",
            scratch,
        ]
        if operation == "resume":
            assert checkpoint is not None
            expected.extend(
                [
                    "--launcher",
                    f"{release_root}/cluster/aws/p5/launch_seed_pair.py",
                    "--checkpoint-receipt",
                    f"{staging}/resume/{checkpoint['sha256']}/receipt.json",
                    "--checkpoint-receipt-sha256",
                    str(checkpoint["sha256"]),
                    "--run-manifest-sha256",
                    str(intent["run_manifest_sha256"]),
                ]
            )
            for row in cast(list[dict[str, object]], checkpoint["checkpoints"]):
                expected.extend(
                    ["--checkpoint", canonical_json(row).decode("ascii")]
                )
        expected.append("--apply")
    else:
        raise RemoteIntentError("v3 operation contains an unexpected Python step")
    if argv != expected:
        raise RemoteIntentError("v3 Python argv differs from the closed operation plan")


def _validate_v3_docker_step(
    *,
    name: str,
    argv: list[str],
    intent: Mapping[str, object],
    release_root: str,
) -> None:
    environment = cast(dict[str, object], intent["environment"])
    scratch = "/mnt/memorysplit"
    sealed_root = (
        f"{scratch}/sealed-evaluation/{intent['sealed_evaluation_sha256']}"
    )
    image_indexes = [
        index
        for index, item in enumerate(argv)
        if item.endswith("@" + str(environment["MS_CONTAINER_DIGEST"]))
    ]
    if len(image_indexes) != 1:
        raise RemoteIntentError("v3 Docker step does not use the pinned image digest")
    image_index = image_indexes[0]
    image = argv[image_index]
    if (
        argv[0:2] != ["/usr/bin/docker", "run"]
        or image_index + 3 >= len(argv)
        or argv[image_index + 1 : image_index + 3]
        != ["/opt/venv/bin/python", "-m"]
    ):
        raise RemoteIntentError("v3 Docker execution boundary is not closed")
    module = argv[image_index + 3]
    if name == "verify-sealed-evaluation":
        if module != "msctl.aws_sealed_evaluation":
            raise RemoteIntentError("v3 sealed verifier module is invalid")
        _required_option(argv, "--root", "/sealed")
        _required_option(
            argv,
            "--expected-release-sha256",
            str(intent["sealed_evaluation_sha256"]),
        )
        _required_option(
            argv,
            "--expected-study-lock-sha256",
            str(intent["study_lock_sha256"]),
        )
        expected_argv = [
            "/usr/bin/docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--user",
            f"{environment['MS_RUNTIME_UID']}:{environment['MS_RUNTIME_GID']}",
            "--mount",
            f"type=bind,src={release_root},dst=/workspace,readonly",
            "--mount",
            f"type=bind,src={sealed_root},dst=/sealed,readonly",
            "--workdir",
            "/workspace",
            image,
            "/opt/venv/bin/python",
            "-m",
            "msctl.aws_sealed_evaluation",
            "--root",
            "/sealed",
            "--expected-release-sha256",
            str(intent["sealed_evaluation_sha256"]),
            "--expected-study-lock-sha256",
            str(intent["study_lock_sha256"]),
        ]
    elif name in {"evaluate-dense", "evaluate-split90"}:
        arm = name.removeprefix("evaluate-")
        if module != "evals.confirmatory":
            raise RemoteIntentError("v3 evaluator module is invalid")
        _required_option(
            argv,
            "--run",
            f"{scratch}/runs/seed-{intent['seed']}/{arm}/run",
        )
        _required_option(argv, "--sealed-release", "/sealed")
        _required_option(
            argv,
            "--expected-study-lock-sha256",
            str(intent["study_lock_sha256"]),
        )
        output_index = argv.index("--output-dir") if "--output-dir" in argv else -1
        if (
            output_index < 0
            or output_index + 1 >= len(argv)
            or not argv[output_index + 1].startswith(f"{scratch}/evaluations/")
            or not argv[output_index + 1].removeprefix(
                f"{scratch}/evaluations/"
            )
            or "/" in argv[output_index + 1].removeprefix(
                f"{scratch}/evaluations/"
            )
        ):
            raise RemoteIntentError("v3 evaluation output path is outside its root")
        run_root = f"{scratch}/runs/seed-{intent['seed']}"
        evaluation_root = f"{scratch}/evaluations"
        expected_argv = [
            "/usr/bin/docker",
            "run",
            "--rm",
            "--read-only",
            "--network",
            "none",
            "--gpus",
            "all",
            "--user",
            f"{environment['MS_RUNTIME_UID']}:{environment['MS_RUNTIME_GID']}",
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
            f"type=bind,src={evaluation_root},dst={evaluation_root}",
            "--workdir",
            "/workspace",
            image,
            "/opt/venv/bin/python",
            "-m",
            "evals.confirmatory",
            "evaluate",
            "--run",
            f"{run_root}/{arm}/run",
            "--sealed-release",
            "/sealed",
            "--expected-study-lock-sha256",
            str(intent["study_lock_sha256"]),
            "--device",
            "cuda",
            "--output-dir",
            argv[output_index + 1],
        ]
    else:
        raise RemoteIntentError("v3 operation contains an unexpected Docker step")
    if argv != expected_argv:
        raise RemoteIntentError("v3 Docker argv differs from the closed operation plan")


def _validate_v3_steps(
    intent: Mapping[str, object],
    steps: list[dict[str, object]],
) -> None:
    if steps != build_v3_operation_steps(intent):
        raise RemoteIntentError(
            "v3 operation steps differ from the closed operation plan"
        )
    operation = str(intent["operation"])
    if operation == "evaluate":
        return
    release_root = f"/mnt/memorysplit/releases/{intent['release_sha256']}"
    control_root = f"/opt/memorysplit/control/{intent['control_bundle_sha256']}"
    environment = cast(dict[str, object], intent["environment"])
    region = str(environment["AWS_REGION"])
    s3_root = str(environment["MS_S3_ROOT"])
    scratch = "/mnt/memorysplit"
    staging = f"{scratch}/staging"
    submit_names = [
        "auto-termination",
        "prepare-aws-private-home",
        "prepare-staging",
        "materialize-release-archive",
        "materialize-release-receipt",
        "materialize-cohort-assignment",
        "materialize-dataset",
        "bootstrap",
        "build-launcher-manifest",
        "paired-launch",
    ]
    resume_names = [
        "prepare-resume-staging",
        "materialize-resume-receipt",
        "materialize-resume-dense",
        "materialize-resume-split90",
        "paired-launch",
    ]
    expected_names = {
        "submit": submit_names,
        "resume": resume_names,
    }[operation]
    names = [str(step["name"]) for step in steps]
    if names != expected_names:
        raise RemoteIntentError(
            "v3 operation steps do not match the closed operation plan"
        )
    checkpoint = (
        cast(dict[str, object], intent["checkpoint_receipt"])
        if operation == "resume"
        else None
    )
    checkpoint_rows = (
        {
            str(row["arm"]): cast(dict[str, object], row)
            for row in cast(list[object], checkpoint["checkpoints"])
            if isinstance(row, dict)
        }
        if checkpoint is not None
        else {}
    )
    for step in steps:
        name = str(step["name"])
        argv = cast(list[str], step["argv"])
        if any(item in _SHELL_INTERPRETERS for item in argv):
            raise RemoteIntentError("v3 operation must not invoke a shell interpreter")
        if name == "auto-termination":
            expected = [
                "/usr/bin/systemd-run",
                "--unit",
                "memorysplit-auto-terminate",
                "--on-calendar",
                str(intent["terminate_at"]),
                "/sbin/shutdown",
                "-h",
                "now",
            ]
            if argv != expected:
                raise RemoteIntentError("v3 auto-termination argv is not exact")
        elif name == "prepare-aws-private-home":
            if argv != [
                "/usr/bin/install",
                "-d",
                "-m",
                "0700",
                "-o",
                "0",
                "-g",
                "0",
                "/var/lib/memorysplit/aws-private-home",
            ]:
                raise RemoteIntentError("v3 private HOME preparation is not exact")
        elif name == "prepare-staging":
            if argv != [
                "/usr/bin/install",
                "-d",
                "-m",
                "0700",
                f"{staging}/releases/{intent['release_sha256']}",
                f"{scratch}/dataset",
            ]:
                raise RemoteIntentError("v3 staging preparation is not exact")
        elif name == "prepare-resume-staging":
            assert checkpoint is not None
            if argv != [
                "/usr/bin/install",
                "-d",
                "-m",
                "0700",
                f"{staging}/resume/{checkpoint['sha256']}",
            ]:
                raise RemoteIntentError("v3 resume staging path is not exact")
        elif name in {"prepare-sealed-evaluation", "prepare-evaluation-output"}:
            destination = (
                f"{scratch}/sealed-evaluation/"
                f"{intent['sealed_evaluation_sha256']}"
                if name == "prepare-sealed-evaluation"
                else f"{scratch}/evaluations"
            )
            if argv != [
                "/usr/bin/install",
                "-d",
                "-m",
                "0700",
                "-o",
                str(environment["MS_RUNTIME_UID"]),
                "-g",
                str(environment["MS_RUNTIME_GID"]),
                destination,
            ]:
                raise RemoteIntentError("v3 evaluation directory setup is not exact")
        elif name == "materialize-release-archive":
            if argv != _aws_get_argv(
                region=region,
                s3_root=s3_root,
                suffix=f"releases/{intent['release_sha256']}/release.zip",
                destination=(
                    f"{staging}/releases/{intent['release_sha256']}/release.zip"
                ),
            ):
                raise RemoteIntentError("v3 release materialization is not exact")
        elif name == "materialize-release-receipt":
            if argv != _aws_get_argv(
                region=region,
                s3_root=s3_root,
                suffix=f"releases/{intent['release_sha256']}/RELEASE.json",
                destination=(
                    f"{staging}/releases/{intent['release_sha256']}/RELEASE.json"
                ),
            ):
                raise RemoteIntentError(
                    "v3 release receipt materialization is not exact"
                )
        elif name == "materialize-cohort-assignment":
            if argv != _aws_get_argv(
                region=region,
                s3_root=s3_root,
                suffix=(
                    f"releases/{intent['release_sha256']}/"
                    "cohort-assignment-v3.json"
                ),
                destination=(
                    f"{staging}/releases/{intent['release_sha256']}/"
                    "cohort-assignment-v3.json"
                ),
            ):
                raise RemoteIntentError("v3 cohort materialization is not exact")
        elif name == "materialize-dataset":
            if argv != [
                "/usr/bin/env",
                "aws",
                "--no-cli-pager",
                "--region",
                region,
                "s3",
                "sync",
                f"{s3_root}/dataset",
                f"{scratch}/dataset",
                "--no-follow-symlinks",
                "--only-show-errors",
            ]:
                raise RemoteIntentError("v3 dataset materialization is not exact")
        elif name == "materialize-resume-receipt":
            assert checkpoint is not None
            if argv != _aws_get_argv(
                region=region,
                s3_root=s3_root,
                suffix=(
                    f"checkpoints/seed-{intent['seed']}/receipts/"
                    f"{checkpoint['sha256']}.json"
                ),
                destination=f"{staging}/resume/{checkpoint['sha256']}/receipt.json",
            ):
                raise RemoteIntentError(
                    "v3 checkpoint receipt materialization is not exact"
                )
        elif name in {"materialize-resume-dense", "materialize-resume-split90"}:
            assert checkpoint is not None
            arm = name.removeprefix("materialize-resume-")
            row = checkpoint_rows.get(arm)
            if row is None or argv != _aws_get_argv(
                region=region,
                s3_root=s3_root,
                suffix=(
                    f"checkpoints/seed-{intent['seed']}/{arm}/sha256/"
                    f"{row['resume_sha256']}.pt"
                ),
                destination=str(row["resume_path"]),
            ):
                raise RemoteIntentError("v3 checkpoint materialization is not exact")
        elif name.startswith("materialize-sealed-evaluation-"):
            member = next(
                (
                    candidate
                    for candidate in _SEALED_EVALUATION_MEMBERS
                    if name
                    == "materialize-sealed-evaluation-"
                    + candidate.replace(".", "-")
                ),
                None,
            )
            if member is None or argv != _aws_get_argv(
                region=region,
                s3_root=s3_root,
                suffix=(
                    f"sealed-evaluation/{intent['sealed_evaluation_sha256']}/"
                    f"{member}"
                ),
                destination=(
                    f"{scratch}/sealed-evaluation/"
                    f"{intent['sealed_evaluation_sha256']}/{member}"
                ),
            ):
                raise RemoteIntentError(
                    "v3 sealed evaluation materialization is not exact"
                )
        elif argv[0] == "/usr/bin/python3":
            _validate_v3_python_step(
                name=name,
                argv=argv,
                intent=intent,
                release_root=release_root,
                control_root=control_root,
            )
        elif argv[0] == "/usr/bin/docker":
            _validate_v3_docker_step(
                name=name,
                argv=argv,
                intent=intent,
                release_root=release_root,
            )
        else:
            raise RemoteIntentError(
                "v3 operation executable is outside the closed allowlist"
            )


def _validate_v3_execution_bindings(
    intent: Mapping[str, object],
    *,
    environment: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    for field in ("release_receipt_sha256", "release_members_sha256"):
        _sha256(intent[field], label=f"operation intent {field}")
    source_commit = intent["source_commit"]
    container_image = intent["container_image"]
    if (
        not isinstance(source_commit, str)
        or _COMMIT_RE.fullmatch(source_commit) is None
        or not isinstance(container_image, str)
        or _ECR_IMAGE_RE.fullmatch(container_image) is None
        or not container_image.endswith("@" + str(environment["MS_CONTAINER_DIGEST"]))
    ):
        raise RemoteIntentError(
            "v3 release and container execution bindings are invalid"
        )
    expected_runtime_sha256 = hashlib.sha256(
        canonical_json(
            {
                "ami_id": environment["MS_AWS_AMI_ID"],
                "container_image": container_image,
                "container_digest": environment["MS_CONTAINER_DIGEST"],
                "gid": int(str(environment["MS_RUNTIME_GID"])),
                "region": environment["AWS_REGION"],
                "s3_root": environment["MS_S3_ROOT"],
                "uid": int(str(environment["MS_RUNTIME_UID"])),
            }
        )
    ).hexdigest()
    if intent["runtime_sha256"] != expected_runtime_sha256:
        raise RemoteIntentError(
            "v3 runtime SHA-256 does not match its closed execution binding"
        )

    raw_runs = intent["runs"]
    if not isinstance(raw_runs, list) or len(raw_runs) != 2:
        raise RemoteIntentError("v3 operation must bind one exact run pair")
    runs: dict[str, dict[str, object]] = {}
    ordered_arms: list[str] = []
    run_ids: set[str] = set()
    for raw in raw_runs:
        if not isinstance(raw, dict) or set(raw) != _RUN_FIELDS:
            raise RemoteIntentError("v3 run binding fields do not match")
        arm = raw["arm"]
        run_id = raw["run_id"]
        config = raw["config"]
        if (
            arm not in {"dense", "split90"}
            or arm in runs
            or not isinstance(run_id, str)
            or _RUN_ID_RE.fullmatch(run_id) is None
            or run_id != f"memorysplit-v3-360m-s{intent['seed']}-{arm}"
            or run_id in run_ids
            or not isinstance(config, str)
            or not config.startswith("configs/")
            or not config.endswith((".yaml", ".yml"))
            or "\\" in config
            or any(part in {"", ".", ".."} for part in config.split("/"))
            or config != f"configs/360m-v3/{arm}-s{intent['seed']}.yaml"
        ):
            raise RemoteIntentError("v3 run binding identity is invalid")
        _sha256(raw["config_sha256"], label=f"{arm} run configuration")
        runs[str(arm)] = raw
        ordered_arms.append(str(arm))
        run_ids.add(run_id)
    if ordered_arms != ["dense", "split90"]:
        raise RemoteIntentError("v3 run bindings are not canonically ordered")
    return runs


def _validate_intent(
    payload: bytes,
    *,
    expected_sha256: str,
    expected_control_bundle_sha256: str | None = None,
) -> dict[str, object]:
    if hashlib.sha256(payload).hexdigest() != _sha256(
        expected_sha256,
        label="expected intent",
    ):
        raise RemoteIntentError("operation intent SHA-256 mismatch")
    intent = _decode_object(payload, label="operation intent")
    if intent.get("operation") == "canary":
        return _validate_canary_intent(
            intent,
            expected_control_bundle_sha256=expected_control_bundle_sha256,
        )
    schema_version = intent.get("schema_version")
    expected_fields = (
        _BASE_FIELDS | _v3_provenance_fields(intent) | _V3_EXECUTION_FIELDS
        if schema_version == 3
        else _BASE_FIELDS
    )
    if set(intent) != expected_fields:
        raise RemoteIntentError("operation intent fields do not match the contract")
    provider = intent["provider"]
    profile_contract = (
        _PROFILE_CONTRACTS.get(provider)
        if isinstance(provider, str)
        else None
    )
    if (
        schema_version not in {1, 3}
        or intent["operation"] not in {"submit", "resume", "evaluate"}
        or profile_contract is None
        or intent["instance_type"] != profile_contract["instance_type"]
        or intent["gres"] != profile_contract["gres"]
        or not isinstance(intent["profile_sha256"], str)
        or _SHA256_RE.fullmatch(intent["profile_sha256"]) is None
        or isinstance(intent["seed"], bool)
        or intent["seed"] not in profile_contract["assigned_seeds"]
        or not isinstance(intent["instance_id"], str)
        or _INSTANCE_RE.fullmatch(intent["instance_id"]) is None
        or not isinstance(intent["terminate_at"], str)
        or not intent["terminate_at"].endswith("Z")
        or (
            intent["operation"] == "resume"
            and not isinstance(intent["checkpoint_receipt"], dict)
        )
        or (
            intent["operation"] == "evaluate"
            and schema_version == 3
            and not isinstance(intent["checkpoint_receipt"], dict)
        )
        or (
            intent["operation"] not in {"resume", "evaluate"}
            and intent["checkpoint_receipt"] is not None
        )
        or (
            intent["operation"] == "evaluate"
            and schema_version != 3
            and intent["checkpoint_receipt"] is not None
        )
    ):
        raise RemoteIntentError("operation intent identity is invalid")
    try:
        deadline = datetime.fromisoformat(
            str(intent["terminate_at"])[:-1] + "+00:00"
        )
    except ValueError as error:
        raise RemoteIntentError(
            "operation termination deadline is invalid"
        ) from error
    now = datetime.now(UTC)
    if (
        deadline.tzinfo is None
        or deadline.utcoffset() != UTC.utcoffset(deadline)
        or deadline <= now
        or deadline > now + timedelta(minutes=1440)
    ):
        raise RemoteIntentError(
            "operation termination deadline is expired or non-UTC"
        )
    if intent["operation"] == "resume":
        _validate_checkpoint_receipt(intent["checkpoint_receipt"])
    for field in (
        "release_sha256",
        "run_manifest_sha256",
        "dataset_sha256",
        "dataset_pointer_sha256",
        "dataset_verification_sha256",
        "environment_receipt_sha256",
        "runtime_sha256",
        "operation_id",
    ):
        _sha256(intent[field], label=f"operation intent {field}")
    if schema_version == 3:
        if provider not in {
            "aws-p5.48xlarge-v3",
            "aws-p6-b300.48xlarge-v3",
        }:
            raise RemoteIntentError(
                "schema-v3 operation intent requires one v3 AWS profile"
            )
        provenance_fields = _v3_provenance_fields(intent)
        for field in provenance_fields - {"fleet_wave"}:
            _sha256(intent[field], label=f"operation intent {field}")
        if type(intent["fleet_wave"]) is not int or intent["fleet_wave"] < 0:
            raise RemoteIntentError("operation intent fleet wave is invalid")
        if (
            expected_control_bundle_sha256 is None
            or intent["control_bundle_sha256"]
            != _sha256(
                expected_control_bundle_sha256,
                label="installed control bundle",
            )
        ):
            raise RemoteIntentError(
                "operation intent control bundle does not match the installed bytes"
            )
    elif provider != "aws-p5.48xlarge":
        raise RemoteIntentError(
            "legacy operation intent requires the legacy AWS P5 profile"
        )
    identity = {
        key: value
        for key, value in intent.items()
        if key
        not in {
            "operation_id",
            "ssm_document",
            "started_receipt_uri",
            "terminal_receipt_uri",
        }
    }
    if hashlib.sha256(canonical_json(identity)).hexdigest() != intent["operation_id"]:
        raise RemoteIntentError("operation ID does not match canonical intent identity")
    environment = intent["environment"]
    expected_environment_fields = (
        _V3_ENVIRONMENT_FIELDS if schema_version == 3 else _ENVIRONMENT_FIELDS
    )
    if (
        not isinstance(environment, dict)
        or set(environment) != expected_environment_fields
        or set(environment) & _FORBIDDEN_ENVIRONMENT
        or not all(isinstance(value, str) and value for value in environment.values())
        or _REGION_RE.fullmatch(str(environment["AWS_REGION"])) is None
    ):
        raise RemoteIntentError(
            "operation environment is not closed and credential-free"
        )
    if schema_version == 3 and (
        environment["AWS_REGION"] not in {"us-east-1", "us-west-2"}
        or re.fullmatch(r"^ami-[0-9a-f]{8,17}$", environment["MS_AWS_AMI_ID"])
        is None
        or re.fullmatch(
            r"^sha256:[0-9a-f]{64}$",
            environment["MS_CONTAINER_DIGEST"],
        )
        is None
        or any(
            not environment[field].isdigit() or int(environment[field]) <= 0
            for field in ("MS_RUNTIME_GID", "MS_RUNTIME_UID")
        )
    ):
        raise RemoteIntentError(
            "v3 operation environment is not pinned to a supported runtime"
        )
    v3_runs = (
        _validate_v3_execution_bindings(intent, environment=environment)
        if schema_version == 3
        else {}
    )
    s3_root = _s3_uri(environment["MS_S3_ROOT"])
    if intent["operation"] == "evaluate" and schema_version == 3:
        _validate_evaluation_checkpoint_receipt(
            intent["checkpoint_receipt"],
            seed=intent["seed"],
            s3_root=s3_root,
            runs=v3_runs,
        )
    started_receipt_uri = _s3_uri(intent["started_receipt_uri"], root=s3_root)
    terminal_receipt_uri = _s3_uri(intent["terminal_receipt_uri"], root=s3_root)
    receipt_root = (
        f"{s3_root}/operations/{intent['operation_id']}/receipts"
    )
    if (
        started_receipt_uri != f"{receipt_root}/started.json"
        or terminal_receipt_uri != f"{receipt_root}/terminal.json"
    ):
        raise RemoteIntentError(
            "operation receipt paths do not match the deterministic identity"
        )
    document = intent["ssm_document"]
    if (
        not isinstance(document, dict)
        or set(document) != {"name", "sha256"}
        or document
        != {
            "name": ARGV_DOCUMENT_NAME,
            "sha256": ARGV_DOCUMENT_SHA256,
        }
    ):
        raise RemoteIntentError("operation intent names the wrong SSM document")
    steps = intent["steps"]
    if not isinstance(steps, list) or not steps:
        raise RemoteIntentError("operation intent must contain ordered argv steps")
    names: list[str] = []
    for step in steps:
        if not isinstance(step, dict) or set(step) != {"name", "argv"}:
            raise RemoteIntentError("operation step fields do not match")
        name = step["name"]
        argv = step["argv"]
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(argv, list)
            or not argv
            or any(not isinstance(item, str) or "\x00" in item for item in argv)
            or not str(argv[0]).startswith("/")
        ):
            raise RemoteIntentError("operation step is not one safe absolute argv")
        names.append(name)
    if intent["operation"] == "submit":
        if (
            names[0] != "auto-termination"
            or "prepare-aws-private-home" not in names
            or "bootstrap" not in names
            or names.index("prepare-aws-private-home") > names.index("bootstrap")
        ):
            raise RemoteIntentError(
                "submit must establish expiry and private HOME before bootstrap"
            )
    if schema_version == 3:
        _validate_v3_steps(intent, cast(list[dict[str, object]], steps))
    return intent


def _receipt(
    intent: Mapping[str, object],
    *,
    intent_sha256: str,
    kind: str,
    nonce: str,
    returncode: int | None = None,
) -> bytes:
    schema_version = 3 if intent.get("schema_version") == 3 else 1
    value: dict[str, object] = {
        "schema_version": schema_version,
        "receipt_type": (
            "memorysplit-aws-operation-v3"
            if schema_version == 3
            else "memorysplit-aws-operation-v1"
        ),
        "kind": kind,
        "provider": intent["provider"],
        "instance_type": intent["instance_type"],
        "profile_sha256": intent["profile_sha256"],
        "gres": intent["gres"],
        "operation_id": intent["operation_id"],
        "intent_sha256": intent_sha256,
        "instance_id": intent["instance_id"],
        "execution_nonce": nonce,
    }
    binding_fields = (
        _CANARY_RECEIPT_BINDING_FIELDS
        if intent.get("operation") == "canary"
        else _v3_provenance_fields(intent)
    )
    if schema_version == 3:
        value.update(
            {
                field: intent[field]
                for field in binding_fields
            }
        )
    if returncode is not None:
        value["returncode"] = returncode
        value["status"] = "success" if returncode == 0 else "failed"
    return canonical_json(value) + b"\n"


def _validate_receipt(
    payload: bytes,
    intent: Mapping[str, object],
    *,
    intent_sha256: str,
    kind: str,
) -> dict[str, object]:
    value = _decode_object(payload, label=f"{kind} receipt")
    fields = {
        "schema_version",
        "receipt_type",
        "kind",
        "provider",
        "instance_type",
        "profile_sha256",
        "gres",
        "operation_id",
        "intent_sha256",
        "instance_id",
        "execution_nonce",
    }
    schema_version = 3 if intent.get("schema_version") == 3 else 1
    binding_fields = (
        _CANARY_RECEIPT_BINDING_FIELDS
        if intent.get("operation") == "canary"
        else _v3_provenance_fields(intent)
    )
    if schema_version == 3:
        fields |= binding_fields
    if kind == "terminal":
        fields |= {"returncode", "status"}
    if set(value) != fields:
        raise RemoteIntentError(f"{kind} receipt fields do not match")
    nonce = value["execution_nonce"]
    if (
        value["schema_version"] != schema_version
        or value["receipt_type"]
        != (
            "memorysplit-aws-operation-v3"
            if schema_version == 3
            else "memorysplit-aws-operation-v1"
        )
        or value["kind"] != kind
        or value["provider"] != intent["provider"]
        or value["instance_type"] != intent["instance_type"]
        or value["profile_sha256"] != intent["profile_sha256"]
        or value["gres"] != intent["gres"]
        or value["operation_id"] != intent["operation_id"]
        or value["intent_sha256"] != intent_sha256
        or value["instance_id"] != intent["instance_id"]
        or (
            schema_version == 3
            and any(
                value[field] != intent[field]
                for field in binding_fields
            )
        )
        or not isinstance(nonce, str)
        or _NONCE_RE.fullmatch(nonce) is None
    ):
        raise RemoteIntentError(f"{kind} receipt identity is invalid")
    if kind == "terminal":
        returncode = value["returncode"]
        if (
            isinstance(returncode, bool)
            or not isinstance(returncode, int)
            or value["status"]
            != ("success" if returncode == 0 else "failed")
        ):
            raise RemoteIntentError("terminal receipt result is invalid")
    expected_payload = _receipt(
        intent,
        intent_sha256=intent_sha256,
        kind=kind,
        nonce=str(nonce),
        returncode=(
            int(value["returncode"]) if kind == "terminal" else None
        ),
    )
    if payload != expected_payload:
        raise RemoteIntentError(f"{kind} receipt is not canonical")
    return value


def execute_intent(
    *,
    intent_uri: str,
    intent_sha256: str,
    store: ImmutableObjectStore,
    executor: ArgvExecutor,
    control_bundle_sha256: str | None = None,
) -> dict[str, object]:
    """Acquire one operation receipt, execute once, and terminally receipt it."""

    payload = store.read(_s3_uri(intent_uri), expected_sha256=intent_sha256)
    intent = _validate_intent(
        payload,
        expected_sha256=intent_sha256,
        expected_control_bundle_sha256=control_bundle_sha256,
    )
    nonce = secrets.token_hex(16)
    metadata = {
        "operation-id": str(intent["operation_id"]),
        "intent-sha256": intent_sha256,
    }
    started = _receipt(
        intent,
        intent_sha256=intent_sha256,
        kind="started",
        nonce=nonce,
    )
    acquired = store.put_if_absent(
        str(intent["started_receipt_uri"]),
        started,
        metadata={**metadata, "receipt-kind": "started"},
    )
    if not acquired:
        existing_started = store.read_if_exists(
            str(intent["started_receipt_uri"])
        )
        if existing_started is None:
            raise RemoteIntentError(
                "started receipt acquisition failed without durable evidence"
            )
        started_value = _validate_receipt(
            existing_started,
            intent,
            intent_sha256=intent_sha256,
            kind="started",
        )
        terminal = store.read_if_exists(str(intent["terminal_receipt_uri"]))
        terminal_value: dict[str, object] | None = None
        if terminal is not None:
            terminal_value = _validate_receipt(
                terminal,
                intent,
                intent_sha256=intent_sha256,
                kind="terminal",
            )
            if (
                terminal_value["execution_nonce"]
                != started_value["execution_nonce"]
            ):
                raise RemoteIntentError(
                    "terminal receipt does not match the acquired execution"
                )
        return {
            "schema_version": 1,
            "operation_id": intent["operation_id"],
            "executed": False,
            "idempotent": True,
            "terminal": terminal is not None,
            "status": (
                terminal_value["status"]
                if terminal_value is not None
                else "recovery-required"
            ),
            "recovery_required": terminal is None,
            "returncode": (
                terminal_value["returncode"]
                if terminal_value is not None
                else 75
            ),
        }
    intent_environment = cast(dict[str, object], intent["environment"])
    environment = {
        **{key: str(value) for key, value in intent_environment.items()},
        "HOME": "/var/lib/memorysplit/aws-home",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MS_OPERATION_ID": str(intent["operation_id"]),
        "PATH": _SAFE_PATH,
    }
    returncode = 0
    steps = cast(list[dict[str, object]], intent["steps"])
    for step in steps:
        step_argv = cast(list[str], step["argv"])
        returncode = executor.run(step_argv, environment=environment)
        if returncode != 0:
            break
    terminal = _receipt(
        intent,
        intent_sha256=intent_sha256,
        kind="terminal",
        nonce=nonce,
        returncode=returncode,
    )
    if not store.put_if_absent(
        str(intent["terminal_receipt_uri"]),
        terminal,
        metadata={**metadata, "receipt-kind": "terminal"},
    ):
        existing = store.read_if_exists(str(intent["terminal_receipt_uri"]))
        if existing != terminal:
            raise RemoteIntentError("terminal receipt conflicts with this execution")
    return {
        "schema_version": 1,
        "operation_id": intent["operation_id"],
        "executed": True,
        "idempotent": False,
        "returncode": returncode,
        "terminal": True,
    }


class SubprocessArgvExecutor:
    def run(
        self,
        argv: Sequence[str],
        *,
        environment: Mapping[str, str],
    ) -> int:
        try:
            completed = subprocess.run(
                list(argv),
                stdin=subprocess.DEVNULL,
                stdout=None,
                stderr=None,
                check=False,
                shell=False,
                env=dict(environment),
                timeout=172_800,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RemoteIntentError("operation argv could not execute") from error
        return completed.returncode


class AwsCliObjectStore:
    """Minimal instance-role-only S3 boundary for the remote wrapper."""

    def __init__(self, *, region: str) -> None:
        if _REGION_RE.fullmatch(region) is None:
            raise RemoteIntentError("AWS region is invalid")
        self.region = region
        self.environment = {
            "AWS_REGION": region,
            "HOME": "/var/lib/memorysplit/aws-home",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": _SAFE_PATH,
        }

    def _location(self, uri: str) -> tuple[str, str]:
        parsed = urlsplit(_s3_uri(uri))
        return str(parsed.hostname), parsed.path.removeprefix("/")

    def _run(self, arguments: Sequence[str]) -> dict[str, object]:
        argv = [
            "aws",
            "--no-cli-pager",
            "--region",
            self.region,
            *arguments,
            "--output",
            "json",
        ]
        try:
            completed = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
                env=self.environment,
                timeout=300,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RemoteIntentError("AWS CLI is unavailable") from error
        if completed.returncode != 0:
            raise RemoteIntentError("AWS S3 operation failed")
        try:
            value = json.loads(
                completed.stdout or "{}",
                object_pairs_hook=_strict_object,
            )
        except json.JSONDecodeError as error:
            raise RemoteIntentError("AWS CLI returned invalid JSON") from error
        if not isinstance(value, dict):
            raise RemoteIntentError("AWS CLI output must be an object")
        return value

    def read(self, uri: str, *, expected_sha256: str | None = None) -> bytes:
        bucket, key = self._location(uri)
        descriptor, temporary = tempfile.mkstemp(prefix="msctl-aws-object-")
        os.close(descriptor)
        try:
            self._run(
                [
                    "s3api",
                    "get-object",
                    "--bucket",
                    bucket,
                    "--key",
                    key,
                    "--checksum-mode",
                    "ENABLED",
                    temporary,
                ]
            )
            payload = Path(temporary).read_bytes()
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        if (
            expected_sha256 is not None
            and hashlib.sha256(payload).hexdigest() != expected_sha256
        ):
            raise RemoteIntentError("downloaded S3 object SHA-256 mismatch")
        return payload

    def read_if_exists(self, uri: str) -> bytes | None:
        try:
            return self.read(uri)
        except RemoteIntentError:
            return None

    def put_if_absent(
        self,
        uri: str,
        payload: bytes,
        *,
        metadata: Mapping[str, str],
    ) -> bool:
        bucket, key = self._location(uri)
        digest = hashlib.sha256(payload).hexdigest()
        checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
        descriptor, temporary = tempfile.mkstemp(prefix="msctl-aws-receipt-")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                self._run(
                    [
                        "s3api",
                        "put-object",
                        "--bucket",
                        bucket,
                        "--key",
                        key,
                        "--body",
                        temporary,
                        "--checksum-algorithm",
                        "SHA256",
                        "--checksum-sha256",
                        checksum,
                        "--metadata",
                        ",".join(
                            f"{name}={value}"
                            for name, value in sorted(metadata.items())
                        ),
                        "--if-none-match",
                        "*",
                    ]
                )
                return True
            except RemoteIntentError:
                return self.read_if_exists(uri) == payload
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent-uri", required=True)
    parser.add_argument("--intent-sha256", required=True)
    parser.add_argument("--control-bundle-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        region = os.environ.get("AWS_REGION", "")
        result = execute_intent(
            intent_uri=arguments.intent_uri,
            intent_sha256=arguments.intent_sha256,
            store=AwsCliObjectStore(region=region),
            executor=SubprocessArgvExecutor(),
            control_bundle_sha256=arguments.control_bundle_sha256,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        returncode = result.get("returncode", 0)
        return returncode if type(returncode) is int else 0
    except (RemoteIntentError, OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": False,
                    "error": str(error),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
