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

from msctl.aws_contracts import validate_digest_pinned_oci_image
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
                        "/usr/bin/python3 /opt/memorysplit/msctl/aws_argv.py "
                        "--intent-uri '{{ IntentUri }}' "
                        "--intent-sha256 '{{ IntentSHA256 }}'"
                    )
                ],
                "timeoutSeconds": "172800",
            },
            "name": "runContentAddressedArgv",
        }
    ],
    "parameters": {
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
    },
    "schemaVersion": "2.2",
}
ARGV_DOCUMENT_CONTENT = canonical_json(_ARGV_DOCUMENT).decode("ascii")
ARGV_DOCUMENT_SHA256 = hashlib.sha256(
    ARGV_DOCUMENT_CONTENT.encode("ascii")
).hexdigest()


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INSTANCE_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
_REGION_RE = re.compile(r"^[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+$")
_BUCKET_RE = re.compile(
    r"^(?![0-9]+(?:\.[0-9]+){3}$)(?!-)(?!.*\.\.)(?!.*\.-)(?!.*-\.)"
    r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$"
)
_NONCE_RE = re.compile(r"^[0-9a-f]{32}$")
_SAFE_PATH = "/usr/local/bin:/usr/bin:/bin"
_BASE_FIELDS = {
    "schema_version",
    "operation",
    "provider",
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
_ENVIRONMENT_FIELDS = {
    "AWS_REGION",
    "MS_AWS_AMI_ID",
    "MS_CONTAINER_DIGEST",
    "MS_CONTAINER_IMAGE",
    "MS_RUNTIME_GID",
    "MS_RUNTIME_UID",
    "MS_S3_ROOT",
}
_CHECKPOINT_RECEIPT_FIELDS = {"sha256", "checkpoints"}
_CHECKPOINT_FIELDS = {
    "arm",
    "resume_path",
    "resume_sha256",
    "world_size",
}
_FORBIDDEN_ENVIRONMENT = {
    "AWS_ACCESS_KEY_ID",
    "AWS_CONFIG_FILE",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_DEFAULT_PROFILE",
    "AWS_EC2_METADATA_DISABLED",
    "AWS_PROFILE",
    "AWS_ROLE_ARN",
    "AWS_ROLE_SESSION_NAME",
    "AWS_SECURITY_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
}


class RemoteIntentError(ValueError):
    """A remote intent or its execution boundary is invalid."""


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
    if arms != {"dense", "split90"}:
        raise RemoteIntentError("checkpoint binding pair is incomplete")


def _validate_intent(
    payload: bytes,
    *,
    expected_sha256: str,
) -> dict[str, object]:
    if hashlib.sha256(payload).hexdigest() != _sha256(
        expected_sha256,
        label="expected intent",
    ):
        raise RemoteIntentError("operation intent SHA-256 mismatch")
    intent = _decode_object(payload, label="operation intent")
    if set(intent) != _BASE_FIELDS:
        raise RemoteIntentError("operation intent fields do not match the contract")
    if (
        intent["schema_version"] != 1
        or intent["operation"] not in {"submit", "resume", "evaluate"}
        or intent["provider"] != "aws-p5.48xlarge"
        or isinstance(intent["seed"], bool)
        or intent["seed"] not in {1, 2, 3, 4}
        or not isinstance(intent["instance_id"], str)
        or _INSTANCE_RE.fullmatch(intent["instance_id"]) is None
        or not isinstance(intent["terminate_at"], str)
        or not intent["terminate_at"].endswith("Z")
        or (
            intent["operation"] == "resume"
            and not isinstance(intent["checkpoint_receipt"], dict)
        )
        or (
            intent["operation"] != "resume"
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
    if (
        not isinstance(environment, dict)
        or set(environment) != _ENVIRONMENT_FIELDS
        or set(environment) & _FORBIDDEN_ENVIRONMENT
        or not all(isinstance(value, str) and value for value in environment.values())
        or _REGION_RE.fullmatch(str(environment["AWS_REGION"])) is None
    ):
        raise RemoteIntentError("operation environment is not closed and credential-free")
    image = str(environment["MS_CONTAINER_IMAGE"])
    digest = str(environment["MS_CONTAINER_DIGEST"])
    try:
        validate_digest_pinned_oci_image(image, digest)
    except ValueError as error:
        raise RemoteIntentError(
            "operation container image is not pinned to its digest"
        ) from error
    s3_root = _s3_uri(environment["MS_S3_ROOT"])
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
    return intent


def _receipt(
    intent: Mapping[str, object],
    *,
    intent_sha256: str,
    kind: str,
    nonce: str,
    returncode: int | None = None,
) -> bytes:
    value: dict[str, object] = {
        "schema_version": 1,
        "receipt_type": "memorysplit-aws-operation-v1",
        "kind": kind,
        "operation_id": intent["operation_id"],
        "intent_sha256": intent_sha256,
        "instance_id": intent["instance_id"],
        "execution_nonce": nonce,
    }
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
        "operation_id",
        "intent_sha256",
        "instance_id",
        "execution_nonce",
    }
    if kind == "terminal":
        fields |= {"returncode", "status"}
    if set(value) != fields:
        raise RemoteIntentError(f"{kind} receipt fields do not match")
    nonce = value["execution_nonce"]
    if (
        value["schema_version"] != 1
        or value["receipt_type"] != "memorysplit-aws-operation-v1"
        or value["kind"] != kind
        or value["operation_id"] != intent["operation_id"]
        or value["intent_sha256"] != intent_sha256
        or value["instance_id"] != intent["instance_id"]
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
    return value


def execute_intent(
    *,
    intent_uri: str,
    intent_sha256: str,
    store: ImmutableObjectStore,
    executor: ArgvExecutor,
) -> dict[str, object]:
    """Acquire one operation receipt, execute once, and terminally receipt it."""

    payload = store.read(_s3_uri(intent_uri), expected_sha256=intent_sha256)
    intent = _validate_intent(payload, expected_sha256=intent_sha256)
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
        }
    intent_environment = cast(dict[str, object], intent["environment"])
    environment = {
        **{key: str(value) for key, value in intent_environment.items()},
        "HOME": "/var/lib/memorysplit/aws-home",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
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
