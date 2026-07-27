"""Fixed AWS IAM/KMS/S3 authority for the reasoning-v3 evaluator."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from corpusgen.parallel.canonical import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[2]
AWS_ACCOUNT_ID = "${AWS_ACCOUNT_ID}"
AWS_REGION = "us-east-1"
EVALUATOR_ROLE_ARN = (
    "arn:aws:iam::${AWS_ACCOUNT_ID}:role/memorysplit-reasoning-v3-evaluator"
)
TRAINER_ROLE_ARN = (
    "arn:aws:iam::${AWS_ACCOUNT_ID}:role/memorysplit-reasoning-v3-trainer"
)
SIGNER_KEY_ALIAS = "alias/memorysplit-reasoning-v3-evaluator-v1"
# S3 SSE-KMS cannot use the approved asymmetric signing key. This fixed,
# evaluator-policy-only alias must resolve to a symmetric encryption key.
SEALED_GOLD_KMS_KEY_ALIAS = (
    "alias/memorysplit-reasoning-v3-evaluator-sealed-v1"
)
CHECKPOINT_KMS_KEY_ALIAS = "alias/memorysplit-reasoning-v3-checkpoints-v1"
STORAGE_BUCKET = "${MS_S3_BUCKET}"
STORAGE_PREFIX = "evaluations/reasoning-v3/v1"
CHECKPOINT_STORAGE_PREFIX = "checkpoints/reasoning-v3/v1/"
AUTHORITY_RECORD_KEY = f"{STORAGE_PREFIX}/authority/contract-authority.json"
AUTHORITY_SIGNATURE_KEY = (
    f"{STORAGE_PREFIX}/authority/contract-authority.signature.json"
)
ACTIVATION_KEY = f"{STORAGE_PREFIX}/activation/ACTIVE.json"
TASK3_CHECKPOINT_MANIFEST_KEY = (
    f"{STORAGE_PREFIX}/evaluator-only/manifests/checkpoint-evidence.json"
)
TASK3_CHECKPOINT_PREFIX = f"{CHECKPOINT_STORAGE_PREFIX}checkpoints/"
TASK3_EVIDENCE_PREFIX = f"{CHECKPOINT_STORAGE_PREFIX}evidence/"
TASK3_RESULT_PREFIX = f"{STORAGE_PREFIX}/evaluator-only/results/"
TASK3_REPORT_KEY = (
    f"{STORAGE_PREFIX}/evaluator-only/reports/frozen-scientific-inference.json"
)
AUTHORITY_CONFIG_PATH = (
    ROOT
    / "configs"
    / "preregistration-135m-reasoning-v3-eval-v1-authority.json"
)
AWS_BOUNDARY_CONFIG_PATH = (
    ROOT
    / "configs"
    / "preregistration-135m-reasoning-v3-eval-v1-aws-boundary.json"
)

_CONTRACT_PATH = "configs/preregistration-135m-reasoning-v3-eval-v1.yaml"
_AWS_BOUNDARY_PATH = (
    "configs/preregistration-135m-reasoning-v3-eval-v1-aws-boundary.json"
)
_CONTRACT_ID = "memorysplit-reasoning-v3-eval-v1"
_AUTHORITY_FORMAT = "memorysplit-reasoning-v3-contract-authority-v1"
_AWS_BOUNDARY_FORMAT = "memorysplit-reasoning-v3-aws-boundary-v1"
_SIGNATURE_FORMAT = "memorysplit-aws-kms-signature-v1"
_TASK3_SIGNATURE_FORMAT = "memorysplit-task3-aws-kms-signature-v1"
_SIGNING_ALGORITHM = "ECDSA_SHA_384"
_MESSAGE_TYPE = "RAW"
_TASK3_MESSAGE_TYPE = "DIGEST"
_TASK3_DIGEST_ALGORITHM = "SHA_384"
_MAX_AUTHORITY_BYTES = 64 << 10
_MAX_SIGNATURE_DOCUMENT_BYTES = 16 << 10
_MAX_ACTIVATION_BYTES = 1 << 20
_MAX_RELEASE_BYTES = 256 << 20
_MAX_TASK3_MANIFEST_BYTES = 32 << 20
_MAX_TASK3_CHECKPOINT_BYTES = 4 << 30
_MAX_TASK3_EVIDENCE_BYTES = 16 << 20
_MAX_TASK3_RESULT_BYTES = 256 << 20
_MAX_TASK3_REPORT_BYTES = 64 << 20
_KMS_RAW_MESSAGE_LIMIT = 4_096
_HEX = frozenset("0123456789abcdef")
_TASK3_ARMS = ("dense", "split90")
_TASK3_SEEDS = tuple(range(10))
_TASK3_STEPS = (1558, 3896, 7791, 11687, 15582)
_TASK3_PAIRING_KINDS = (
    "config",
    "corpus",
    "data_order",
    "initialization",
    "runtime",
)


class AwsAuthorityError(PermissionError):
    """The fixed AWS evaluator authority is absent, invalid, or inaccessible."""


@dataclass(frozen=True)
class ContractAuthorityRecord:
    contract_id: str
    contract_path: str
    contract_sha256: str
    aws_boundary_path: str
    aws_boundary_sha256: str
    aws_region: str
    evaluator_role_arn: str
    trainer_role_arn: str
    signer_key_alias: str
    sealed_gold_kms_key_alias: str
    checkpoint_kms_key_alias: str
    storage_bucket: str
    storage_prefix: str
    checkpoint_storage_prefix: str
    authority_record_key: str
    authority_signature_key: str
    activation_key: str
    task3_checkpoint_manifest_key: str
    task3_checkpoint_prefix: str
    task3_evidence_prefix: str
    task3_result_prefix: str
    task3_report_key: str


@dataclass(frozen=True)
class VerifiedAwsAuthority:
    contract_sha256: str
    record_sha256: str
    record_version_id: str
    signature_version_id: str
    signer_key_arn: str
    sealed_gold_kms_key_arn: str
    checkpoint_kms_key_arn: str


@dataclass(frozen=True)
class S3ObjectVersion:
    bucket: str
    key: str
    version_id: str
    bytes: int
    sha256: str
    server_side_encryption: str
    kms_key_arn: str | None = None


@dataclass(frozen=True)
class ActivationPublication:
    envelope_bytes: bytes
    object_ref: S3ObjectVersion
    created: bool


@dataclass(frozen=True)
class Task3Publication:
    payload_bytes: bytes
    envelope_bytes: bytes
    object_ref: S3ObjectVersion
    created: bool


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _HEX
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AwsAuthorityError(f"AWS authority JSON repeats key: {key}")
        result[key] = value
    return result


def _parse_canonical_json(
    payload: bytes,
    *,
    label: str,
    maximum: int,
) -> Mapping[str, Any]:
    if not isinstance(payload, bytes) or not payload or len(payload) > maximum:
        raise AwsAuthorityError(f"{label} is missing or oversized")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                AwsAuthorityError(f"{label} contains non-finite JSON: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AwsAuthorityError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, Mapping) or payload != canonical_json_bytes(value):
        raise AwsAuthorityError(f"{label} is not canonical JSON")
    return value


def _read_regular_file(path: Path, *, label: str, maximum: int) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise AwsAuthorityError(f"{label} is unavailable") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size <= 0
        or metadata.st_size > maximum
    ):
        raise AwsAuthorityError(f"{label} is unsafe or oversized")
    try:
        return path.read_bytes()
    except OSError as error:
        raise AwsAuthorityError(f"{label} could not be read") from error


def _parse_aws_boundary_record(payload: bytes) -> Mapping[str, Any]:
    record = _parse_canonical_json(
        payload,
        label="AWS evaluator boundary record",
        maximum=_MAX_AUTHORITY_BYTES,
    )
    expected = {
        "account_id": AWS_ACCOUNT_ID,
        "authorization": {
            "activation": "trainer_and_evaluator_read_evaluator_write",
            "authority": "trainer_and_evaluator_read_evaluator_write",
            "model_visible": "trainer_and_evaluator_read_evaluator_write",
            "sealed_gold": "evaluator_only",
        },
        "evaluator_role_arn": EVALUATOR_ROLE_ARN,
        "trainer_role_arn": TRAINER_ROLE_ARN,
        "format": _AWS_BOUNDARY_FORMAT,
        "kms": {
            "checkpoint_key_alias": CHECKPOINT_KMS_KEY_ALIAS,
            "checkpoint_key_spec": "SYMMETRIC_DEFAULT",
            "sealed_gold_key_alias": SEALED_GOLD_KMS_KEY_ALIAS,
            "sealed_gold_key_spec": "SYMMETRIC_DEFAULT",
            "signer_key_alias": SIGNER_KEY_ALIAS,
            "signer_key_spec": "ECC_NIST_P384",
            "signing_algorithm": _SIGNING_ALGORITHM,
        },
        "region": AWS_REGION,
        "required_controls": {
            "bucket_versioning": "Enabled",
            "deny_non_evaluator_sealed_gold_kms": True,
            "deny_non_evaluator_sealed_gold_s3": True,
            "immutable_activation": "s3_if_none_match_and_version_bound",
            "sealed_gold_encryption": "aws:kms",
        },
        "schema_version": 1,
        "storage": {
            "activation_key": ACTIVATION_KEY,
            "authority_prefix": f"{STORAGE_PREFIX}/authority/",
            "bucket": STORAGE_BUCKET,
            "model_visible_prefix": f"{STORAGE_PREFIX}/model-visible/",
            "checkpoint_storage_prefix": CHECKPOINT_STORAGE_PREFIX,
            "sealed_gold_prefix": f"{STORAGE_PREFIX}/evaluator-only/",
        },
        "task3": {
            "authorization": {
                "checkpoint": "trainer_write_evaluator_read_exact_version",
                "evidence": "trainer_write_evaluator_read_exact_version",
                "manifest": "evaluator_only",
                "report": "evaluator_only",
                "result": "evaluator_only",
            },
            "key_patterns": {
                "checkpoint": (
                    f"{TASK3_CHECKPOINT_PREFIX}"
                    "{run_id}/step{checkpoint_step:07d}.pt"
                ),
                "evidence": (
                    f"{TASK3_EVIDENCE_PREFIX}"
                    "{run_id}/step{checkpoint_step:07d}/gates.json"
                ),
                "run_config_evidence": (
                    f"{TASK3_EVIDENCE_PREFIX}" "{run_id}/config.json"
                ),
                "pairing_evidence": (
                    f"{TASK3_EVIDENCE_PREFIX}"
                    "{run_id}/pairing/{pairing_kind}.json"
                ),
                "runtime_lock_evidence": (
                    f"{TASK3_EVIDENCE_PREFIX}runtime/runtime-lock.json"
                ),
                "corpus_receipt_evidence": (
                    f"{TASK3_EVIDENCE_PREFIX}corpus/receipt.json"
                ),
                "result": (
                    f"{TASK3_RESULT_PREFIX}"
                    "{arm}/seed-{seed}/step-{checkpoint_step:07d}.json"
                ),
            },
            "kms": {
                "checkpoint_encryption": "aws:kms",
                "checkpoint_key_alias": CHECKPOINT_KMS_KEY_ALIAS,
                "checkpoint_key_spec": "SYMMETRIC_DEFAULT",
                "evaluator_artifact_encryption": "aws:kms",
                "evaluator_artifact_key_alias": SEALED_GOLD_KMS_KEY_ALIAS,
                "evaluator_artifact_key_spec": "SYMMETRIC_DEFAULT",
                "signing_algorithm": _SIGNING_ALGORITHM,
                "signing_key_alias": SIGNER_KEY_ALIAS,
            },
            "operations": {
                "checkpoint_reader": [
                    "kms:Decrypt",
                    "kms:DescribeKey",
                    "s3:GetObjectVersion",
                ],
                "checkpoint_writer": [
                    "kms:DescribeKey",
                    "kms:GenerateDataKey",
                    "s3:PutObject",
                ],
                "evidence_reader": [
                    "kms:Decrypt",
                    "kms:DescribeKey",
                    "s3:GetObjectVersion",
                ],
                "evidence_writer": [
                    "kms:DescribeKey",
                    "kms:GenerateDataKey",
                    "s3:PutObject",
                ],
                "manifest": [
                    "kms:Sign",
                    "kms:Verify",
                    "s3:GetObjectVersion",
                    "s3:PutObject",
                ],
                "report": [
                    "kms:Sign",
                    "kms:Verify",
                    "s3:GetObjectVersion",
                    "s3:PutObject",
                ],
                "result": [
                    "kms:Sign",
                    "kms:Verify",
                    "s3:GetObjectVersion",
                    "s3:PutObject",
                ],
            },
            "required_controls": {
                "deny_non_evaluator_checkpoint_reads": True,
                "deny_non_evaluator_evidence_reads": True,
                "deny_non_evaluator_report_reads": True,
                "deny_non_evaluator_result_reads": True,
                "exact_version_reads": True,
                "manifest_signature": "ECDSA_SHA_384",
                "no_delete": True,
                "no_overwrite": True,
                "publication": "s3_if_none_match_and_version_bound",
                "result_report_signatures": "ECDSA_SHA_384",
                "checkpoint_evidence_publication": (
                    "s3_if_none_match_and_version_bound"
                ),
                "trainer_denied_evaluator_only_prefix": True,
                "trainer_denied_sealed_gold_kms": True,
                "trainer_denied_signer_kms": True,
            },
            "roles": {
                "checkpoint_reader_role_arn": EVALUATOR_ROLE_ARN,
                "checkpoint_writer_role_arn": TRAINER_ROLE_ARN,
                "evaluator_artifact_writer_role_arn": EVALUATOR_ROLE_ARN,
            },
            "storage": {
                "checkpoint_storage_prefix": CHECKPOINT_STORAGE_PREFIX,
                "checkpoint_manifest_key": TASK3_CHECKPOINT_MANIFEST_KEY,
                "checkpoint_prefix": TASK3_CHECKPOINT_PREFIX,
                "evidence_prefix": TASK3_EVIDENCE_PREFIX,
                "report_key": TASK3_REPORT_KEY,
                "result_prefix": TASK3_RESULT_PREFIX,
            },
        },
    }
    if dict(record) != expected:
        raise AwsAuthorityError("AWS evaluator boundary record differs")
    return record


def _parse_contract_authority_record(
    payload: bytes,
    contract_bytes: bytes,
    aws_boundary_bytes: bytes,
) -> ContractAuthorityRecord:
    _parse_aws_boundary_record(aws_boundary_bytes)
    record = _parse_canonical_json(
        payload,
        label="contract authority record",
        maximum=_MAX_AUTHORITY_BYTES,
    )
    expected_fields = {
        "activation_key",
        "authority_record_key",
        "authority_signature_key",
        "aws_boundary_path",
        "aws_boundary_sha256",
        "aws_region",
        "contract_id",
        "contract_path",
        "contract_sha256",
        "checkpoint_kms_key_alias",
        "checkpoint_storage_prefix",
        "evaluator_role_arn",
        "format",
        "schema_version",
        "sealed_gold_kms_key_alias",
        "signer_key_alias",
        "storage_bucket",
        "storage_prefix",
        "trainer_role_arn",
        "task3_checkpoint_manifest_key",
        "task3_checkpoint_prefix",
        "task3_evidence_prefix",
        "task3_report_key",
        "task3_result_prefix",
    }
    if set(record) != expected_fields:
        raise AwsAuthorityError("contract authority record fields differ")
    actual_contract_sha256 = hashlib.sha256(contract_bytes).hexdigest()
    actual_boundary_sha256 = hashlib.sha256(aws_boundary_bytes).hexdigest()
    expected = {
        "activation_key": ACTIVATION_KEY,
        "authority_record_key": AUTHORITY_RECORD_KEY,
        "authority_signature_key": AUTHORITY_SIGNATURE_KEY,
        "aws_boundary_path": _AWS_BOUNDARY_PATH,
        "aws_boundary_sha256": actual_boundary_sha256,
        "aws_region": AWS_REGION,
        "contract_id": _CONTRACT_ID,
        "contract_path": _CONTRACT_PATH,
        "contract_sha256": actual_contract_sha256,
        "checkpoint_kms_key_alias": CHECKPOINT_KMS_KEY_ALIAS,
        "checkpoint_storage_prefix": CHECKPOINT_STORAGE_PREFIX,
        "evaluator_role_arn": EVALUATOR_ROLE_ARN,
        "format": _AUTHORITY_FORMAT,
        "schema_version": 1,
        "sealed_gold_kms_key_alias": SEALED_GOLD_KMS_KEY_ALIAS,
        "signer_key_alias": SIGNER_KEY_ALIAS,
        "storage_bucket": STORAGE_BUCKET,
        "storage_prefix": STORAGE_PREFIX,
        "trainer_role_arn": TRAINER_ROLE_ARN,
        "task3_checkpoint_manifest_key": TASK3_CHECKPOINT_MANIFEST_KEY,
        "task3_checkpoint_prefix": TASK3_CHECKPOINT_PREFIX,
        "task3_evidence_prefix": TASK3_EVIDENCE_PREFIX,
        "task3_report_key": TASK3_REPORT_KEY,
        "task3_result_prefix": TASK3_RESULT_PREFIX,
    }
    if dict(record) != expected or not _is_sha256(record["contract_sha256"]):
        raise AwsAuthorityError(
            "contract authority record or exact contract digest differs"
        )
    return ContractAuthorityRecord(
        contract_id=_CONTRACT_ID,
        contract_path=_CONTRACT_PATH,
        contract_sha256=actual_contract_sha256,
        aws_boundary_path=_AWS_BOUNDARY_PATH,
        aws_boundary_sha256=actual_boundary_sha256,
        aws_region=AWS_REGION,
        evaluator_role_arn=EVALUATOR_ROLE_ARN,
        trainer_role_arn=TRAINER_ROLE_ARN,
        signer_key_alias=SIGNER_KEY_ALIAS,
        sealed_gold_kms_key_alias=SEALED_GOLD_KMS_KEY_ALIAS,
        checkpoint_kms_key_alias=CHECKPOINT_KMS_KEY_ALIAS,
        storage_bucket=STORAGE_BUCKET,
        storage_prefix=STORAGE_PREFIX,
        checkpoint_storage_prefix=CHECKPOINT_STORAGE_PREFIX,
        authority_record_key=AUTHORITY_RECORD_KEY,
        authority_signature_key=AUTHORITY_SIGNATURE_KEY,
        activation_key=ACTIVATION_KEY,
        task3_checkpoint_manifest_key=TASK3_CHECKPOINT_MANIFEST_KEY,
        task3_checkpoint_prefix=TASK3_CHECKPOINT_PREFIX,
        task3_evidence_prefix=TASK3_EVIDENCE_PREFIX,
        task3_result_prefix=TASK3_RESULT_PREFIX,
        task3_report_key=TASK3_REPORT_KEY,
    )


def _valid_key_arn(value: object) -> bool:
    prefix = f"arn:aws:kms:{AWS_REGION}:{AWS_ACCOUNT_ID}:key/"
    return (
        isinstance(value, str)
        and value.startswith(prefix)
        and len(value) > len(prefix)
        and all(character.isalnum() or character == "-" for character in value[len(prefix):])
    )


def _signature_document_bytes(
    signature: bytes,
    message: bytes,
    key_arn: str,
) -> bytes:
    if (
        not isinstance(signature, bytes)
        or not signature
        or len(signature) > 1_024
        or not isinstance(message, bytes)
        or not message
        or len(message) > _KMS_RAW_MESSAGE_LIMIT
        or not _valid_key_arn(key_arn)
    ):
        raise AwsAuthorityError("KMS signature document inputs are invalid")
    return canonical_json_bytes(
        {
            "format": _SIGNATURE_FORMAT,
            "key_id": key_arn,
            "message_sha256": hashlib.sha256(message).hexdigest(),
            "message_type": _MESSAGE_TYPE,
            "schema_version": 1,
            "signature_base64": base64.b64encode(signature).decode("ascii"),
            "signing_algorithm": _SIGNING_ALGORITHM,
        }
    )


def _parse_signature_document(
    payload: bytes,
    message: bytes,
    key_arn: str,
) -> bytes:
    document = _parse_canonical_json(
        payload,
        label="KMS signature document",
        maximum=_MAX_SIGNATURE_DOCUMENT_BYTES,
    )
    if set(document) != {
        "format",
        "key_id",
        "message_sha256",
        "message_type",
        "schema_version",
        "signature_base64",
        "signing_algorithm",
    }:
        raise AwsAuthorityError("KMS signature document fields differ")
    if (
        document["format"] != _SIGNATURE_FORMAT
        or document["schema_version"] != 1
        or isinstance(document["schema_version"], bool)
        or document["key_id"] != key_arn
        or document["message_sha256"] != hashlib.sha256(message).hexdigest()
        or document["message_type"] != _MESSAGE_TYPE
        or document["signing_algorithm"] != _SIGNING_ALGORITHM
        or not isinstance(document["signature_base64"], str)
    ):
        raise AwsAuthorityError("KMS signature document identity differs")
    try:
        signature = base64.b64decode(
            document["signature_base64"],
            validate=True,
        )
    except (ValueError, binascii.Error) as error:
        raise AwsAuthorityError("KMS signature encoding differs") from error
    if not signature or len(signature) > 1_024:
        raise AwsAuthorityError("KMS signature length differs")
    return signature


def _task3_signature_document_bytes(
    signature: bytes,
    payload: bytes,
    key_arn: str,
) -> bytes:
    if (
        not isinstance(signature, bytes)
        or not signature
        or len(signature) > 1_024
        or not isinstance(payload, bytes)
        or not payload
        or not _valid_key_arn(key_arn)
    ):
        raise AwsAuthorityError("Task 3 KMS signature document inputs are invalid")
    return canonical_json_bytes(
        {
            "digest_algorithm": _TASK3_DIGEST_ALGORITHM,
            "format": _TASK3_SIGNATURE_FORMAT,
            "key_id": key_arn,
            "message_type": _TASK3_MESSAGE_TYPE,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "schema_version": 1,
            "signature_base64": base64.b64encode(signature).decode("ascii"),
            "signing_algorithm": _SIGNING_ALGORITHM,
        }
    )


def _parse_task3_signature_document(
    document: object,
    payload: bytes,
    key_arn: str,
) -> bytes:
    if not isinstance(document, Mapping):
        raise AwsAuthorityError("Task 3 KMS signature document is missing")
    expected_fields = {
        "digest_algorithm",
        "format",
        "key_id",
        "message_type",
        "payload_sha256",
        "schema_version",
        "signature_base64",
        "signing_algorithm",
    }
    if set(document) != expected_fields:
        raise AwsAuthorityError("Task 3 KMS signature document fields differ")
    if (
        document["digest_algorithm"] != _TASK3_DIGEST_ALGORITHM
        or document["format"] != _TASK3_SIGNATURE_FORMAT
        or document["key_id"] != key_arn
        or document["message_type"] != _TASK3_MESSAGE_TYPE
        or document["payload_sha256"] != hashlib.sha256(payload).hexdigest()
        or document["schema_version"] != 1
        or isinstance(document["schema_version"], bool)
        or document["signing_algorithm"] != _SIGNING_ALGORITHM
        or not isinstance(document["signature_base64"], str)
    ):
        raise AwsAuthorityError("Task 3 KMS signature identity differs")
    try:
        signature = base64.b64decode(
            document["signature_base64"],
            validate=True,
        )
    except (ValueError, binascii.Error) as error:
        raise AwsAuthorityError("Task 3 KMS signature encoding differs") from error
    if not signature or len(signature) > 1_024:
        raise AwsAuthorityError("Task 3 KMS signature length differs")
    return signature


def _role_arn_from_identity(identity: Mapping[str, Any]) -> str:
    expected_prefix = (
        f"arn:aws:sts::{AWS_ACCOUNT_ID}:assumed-role/"
        "memorysplit-reasoning-v3-evaluator/"
    )
    account = identity.get("Account") if isinstance(identity, Mapping) else None
    arn = identity.get("Arn") if isinstance(identity, Mapping) else None
    valid_session = (
        isinstance(arn, str)
        and arn.startswith(expected_prefix)
        and len(arn) > len(expected_prefix)
        and "/" not in arn[len(expected_prefix):]
    )
    if account != AWS_ACCOUNT_ID or not valid_session:
        raise AwsAuthorityError(
            "operation requires the dedicated evaluator role at the AWS boundary"
        )
    return EVALUATOR_ROLE_ARN


def _trainer_role_arn_from_identity(identity: Mapping[str, Any]) -> str:
    expected_prefix = (
        f"arn:aws:sts::{AWS_ACCOUNT_ID}:assumed-role/"
        "memorysplit-reasoning-v3-trainer/"
    )
    account = identity.get("Account") if isinstance(identity, Mapping) else None
    arn = identity.get("Arn") if isinstance(identity, Mapping) else None
    valid_session = (
        isinstance(arn, str)
        and arn.startswith(expected_prefix)
        and len(arn) > len(expected_prefix)
        and "/" not in arn[len(expected_prefix):]
    )
    if account != AWS_ACCOUNT_ID or not valid_session:
        raise AwsAuthorityError(
            "operation requires the dedicated trainer role at the AWS boundary"
        )
    return TRAINER_ROLE_ARN


def _object_dict(value: S3ObjectVersion) -> dict[str, Any]:
    result: dict[str, Any] = {
        "bucket": value.bucket,
        "bytes": value.bytes,
        "key": value.key,
        "server_side_encryption": value.server_side_encryption,
        "sha256": value.sha256,
        "version_id": value.version_id,
    }
    if value.kms_key_arn is not None:
        result["kms_key_arn"] = value.kms_key_arn
    return result


class _FixedAwsEvaluatorAuthority:
    """Non-injectable production adapter pinned to one account, region, and bucket."""

    def __init__(self) -> None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as error:
            raise AwsAuthorityError(
                "boto3/botocore are required for fixed AWS evaluator authority"
            ) from error
        try:
            session = boto3.Session(region_name=AWS_REGION)
            config = Config(
                region_name=AWS_REGION,
                retries={"max_attempts": 3, "mode": "standard"},
                signature_version="v4",
            )
            common = {"config": config, "region_name": AWS_REGION, "verify": True}
            self._sts = session.client(
                "sts",
                endpoint_url=f"https://sts.{AWS_REGION}.amazonaws.com",
                **common,
            )
            self._kms = session.client(
                "kms",
                endpoint_url=f"https://kms.{AWS_REGION}.amazonaws.com",
                **common,
            )
            self._s3 = session.client(
                "s3",
                endpoint_url=f"https://s3.{AWS_REGION}.amazonaws.com",
                **common,
            )
        except Exception as error:
            raise AwsAuthorityError(
                "fixed AWS evaluator clients could not be initialized"
            ) from error

    @staticmethod
    def _aws_call(label: str, operation, **kwargs):
        try:
            return operation(**kwargs)
        except Exception as error:
            raise AwsAuthorityError(f"{label} failed closed") from error

    @staticmethod
    def _version_id(response: Mapping[str, Any], label: str) -> str:
        version = response.get("VersionId")
        if not isinstance(version, str) or not version or version == "null":
            raise AwsAuthorityError(
                f"{label} requires enabled S3 object versioning"
            )
        return version

    @staticmethod
    def _read_body(
        response: Mapping[str, Any],
        *,
        label: str,
        maximum: int,
    ) -> bytes:
        body = response.get("Body")
        if body is None or not hasattr(body, "read"):
            raise AwsAuthorityError(f"{label} response body is unavailable")
        try:
            payload = body.read(maximum + 1)
            extra = body.read(1)
        except Exception as error:
            raise AwsAuthorityError(f"{label} response body failed") from error
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        if (
            not isinstance(payload, bytes)
            or not payload
            or len(payload) > maximum
            or extra
        ):
            raise AwsAuthorityError(f"{label} response body is invalid")
        return payload

    def require_evaluator_role(self) -> None:
        identity = self._aws_call(
            "AWS evaluator identity verification",
            self._sts.get_caller_identity,
        )
        _role_arn_from_identity(identity)

    def _resolve_signer_key(self) -> str:
        response = self._aws_call(
            "KMS signer-key resolution",
            self._kms.describe_key,
            KeyId=SIGNER_KEY_ALIAS,
        )
        metadata = response.get("KeyMetadata")
        if not isinstance(metadata, Mapping):
            raise AwsAuthorityError("KMS signer metadata is unavailable")
        arn = metadata.get("Arn")
        if (
            not _valid_key_arn(arn)
            or metadata.get("Enabled") is not True
            or metadata.get("KeyState") != "Enabled"
            or metadata.get("KeyUsage") != "SIGN_VERIFY"
            or metadata.get("KeySpec") != "ECC_NIST_P384"
            or _SIGNING_ALGORITHM not in metadata.get("SigningAlgorithms", ())
        ):
            raise AwsAuthorityError("KMS signer key does not match frozen policy")
        return str(arn)

    def _resolve_sealed_key(self) -> str:
        response = self._aws_call(
            "KMS sealed-gold key resolution",
            self._kms.describe_key,
            KeyId=SEALED_GOLD_KMS_KEY_ALIAS,
        )
        metadata = response.get("KeyMetadata")
        if not isinstance(metadata, Mapping):
            raise AwsAuthorityError("KMS sealed-gold metadata is unavailable")
        arn = metadata.get("Arn")
        if (
            not _valid_key_arn(arn)
            or metadata.get("Enabled") is not True
            or metadata.get("KeyState") != "Enabled"
            or metadata.get("KeyUsage") != "ENCRYPT_DECRYPT"
            or metadata.get("KeySpec") != "SYMMETRIC_DEFAULT"
        ):
            raise AwsAuthorityError(
                "KMS sealed-gold key does not match frozen policy"
            )
        return str(arn)

    def _resolve_checkpoint_key(self) -> str:
        response = self._aws_call(
            "KMS checkpoint key resolution",
            self._kms.describe_key,
            KeyId=CHECKPOINT_KMS_KEY_ALIAS,
        )
        metadata = response.get("KeyMetadata")
        if not isinstance(metadata, Mapping):
            raise AwsAuthorityError("KMS checkpoint metadata is unavailable")
        arn = metadata.get("Arn")
        if (
            not _valid_key_arn(arn)
            or metadata.get("Enabled") is not True
            or metadata.get("KeyState") != "Enabled"
            or metadata.get("KeyUsage") != "ENCRYPT_DECRYPT"
            or metadata.get("KeySpec") != "SYMMETRIC_DEFAULT"
        ):
            raise AwsAuthorityError(
                "KMS checkpoint key does not match frozen policy"
            )
        return str(arn)

    def _get_object(
        self,
        key: str,
        *,
        label: str,
        maximum: int,
        version_id: str | None = None,
    ) -> tuple[bytes, S3ObjectVersion]:
        kwargs: dict[str, Any] = {"Bucket": STORAGE_BUCKET, "Key": key}
        if version_id is not None:
            if not isinstance(version_id, str) or not version_id:
                raise AwsAuthorityError(f"{label} version is invalid")
            kwargs["VersionId"] = version_id
        response = self._aws_call(
            f"S3 {label} read",
            self._s3.get_object,
            **kwargs,
        )
        version = self._version_id(response, label)
        if version_id is not None and version != version_id:
            raise AwsAuthorityError(f"{label} returned the wrong object version")
        payload = self._read_body(response, label=label, maximum=maximum)
        metadata = response.get("Metadata", {})
        if not isinstance(metadata, Mapping):
            raise AwsAuthorityError(f"{label} metadata is invalid")
        digest = hashlib.sha256(payload).hexdigest()
        metadata_digest = metadata.get("sha256")
        if metadata_digest is not None and metadata_digest != digest:
            raise AwsAuthorityError(f"{label} metadata digest differs")
        encryption = response.get("ServerSideEncryption")
        if encryption is None:
            encryption = ""
        if not isinstance(encryption, str):
            raise AwsAuthorityError(f"{label} encryption metadata differs")
        kms_key = response.get("SSEKMSKeyId")
        if kms_key is not None and not isinstance(kms_key, str):
            raise AwsAuthorityError(f"{label} KMS metadata differs")
        return payload, S3ObjectVersion(
            bucket=STORAGE_BUCKET,
            key=key,
            version_id=version,
            bytes=len(payload),
            sha256=digest,
            server_side_encryption=encryption,
            kms_key_arn=kms_key,
        )

    def _get_stable_current_object(
        self,
        key: str,
        *,
        label: str,
        maximum: int,
    ) -> tuple[bytes, S3ObjectVersion]:
        first_payload, first_ref = self._get_object(
            key,
            label=label,
            maximum=maximum,
        )
        exact_payload, exact_ref = self._get_object(
            key,
            label=label,
            maximum=maximum,
            version_id=first_ref.version_id,
        )
        final_payload, final_ref = self._get_object(
            key,
            label=label,
            maximum=maximum,
        )
        if (
            first_payload != exact_payload
            or first_ref != exact_ref
            or final_payload != exact_payload
            or final_ref != exact_ref
        ):
            raise AwsAuthorityError(
                f"{label} current object/version is ambiguous or mutable"
            )
        return exact_payload, exact_ref

    def _verify_kms(
        self,
        message: bytes,
        signature: bytes,
        key_arn: str,
    ) -> None:
        if not message or len(message) > _KMS_RAW_MESSAGE_LIMIT:
            raise AwsAuthorityError("KMS verification message is invalid")
        response = self._aws_call(
            "KMS ECDSA signature verification",
            self._kms.verify,
            KeyId=key_arn,
            Message=message,
            MessageType=_MESSAGE_TYPE,
            Signature=signature,
            SigningAlgorithm=_SIGNING_ALGORITHM,
        )
        if (
            response.get("SignatureValid") is not True
            or response.get("KeyId") != key_arn
            or response.get("SigningAlgorithm") != _SIGNING_ALGORITHM
        ):
            raise AwsAuthorityError("KMS ECDSA signature is invalid")

    def _sign_task3_payload(
        self,
        payload: bytes,
        verified: VerifiedAwsAuthority,
    ) -> bytes:
        if not isinstance(payload, bytes) or not payload:
            raise AwsAuthorityError("Task 3 signing payload is invalid")
        digest = hashlib.sha384(payload).digest()
        response = self._aws_call(
            "KMS Task 3 ECDSA signing",
            self._kms.sign,
            KeyId=verified.signer_key_arn,
            Message=digest,
            MessageType=_TASK3_MESSAGE_TYPE,
            SigningAlgorithm=_SIGNING_ALGORITHM,
        )
        signature = response.get("Signature")
        if (
            not isinstance(signature, bytes)
            or not signature
            or response.get("KeyId") != verified.signer_key_arn
            or response.get("SigningAlgorithm") != _SIGNING_ALGORITHM
        ):
            raise AwsAuthorityError("KMS Task 3 signing response differs")
        return signature

    def _verify_task3_signature(
        self,
        payload: bytes,
        signature: bytes,
        verified: VerifiedAwsAuthority,
    ) -> None:
        response = self._aws_call(
            "KMS Task 3 ECDSA signature verification",
            self._kms.verify,
            KeyId=verified.signer_key_arn,
            Message=hashlib.sha384(payload).digest(),
            MessageType=_TASK3_MESSAGE_TYPE,
            Signature=signature,
            SigningAlgorithm=_SIGNING_ALGORITHM,
        )
        if (
            response.get("SignatureValid") is not True
            or response.get("KeyId") != verified.signer_key_arn
            or response.get("SigningAlgorithm") != _SIGNING_ALGORITHM
        ):
            raise AwsAuthorityError("Task 3 KMS ECDSA signature is invalid")

    def _task3_signed_envelope(
        self,
        payload: bytes,
        verified: VerifiedAwsAuthority,
    ) -> bytes:
        parsed = _parse_canonical_json(
            payload,
            label="Task 3 unsigned payload",
            maximum=_MAX_TASK3_RESULT_BYTES,
        )
        signature = self._sign_task3_payload(payload, verified)
        signature_document = _parse_canonical_json(
            _task3_signature_document_bytes(
                signature,
                payload,
                verified.signer_key_arn,
            ),
            label="Task 3 KMS signature document",
            maximum=_MAX_SIGNATURE_DOCUMENT_BYTES,
        )
        return canonical_json_bytes(
            {"payload": dict(parsed), "signature": dict(signature_document)}
        )

    def _verify_task3_envelope(
        self,
        envelope: bytes,
        verified: VerifiedAwsAuthority,
        *,
        label: str,
        maximum: int,
    ) -> bytes:
        parsed = _parse_canonical_json(
            envelope,
            label=f"signed {label}",
            maximum=maximum,
        )
        if set(parsed) != {"payload", "signature"} or not isinstance(
            parsed["payload"], Mapping
        ):
            raise AwsAuthorityError(f"signed {label} fields differ")
        payload = canonical_json_bytes(parsed["payload"])
        signature = _parse_task3_signature_document(
            parsed["signature"],
            payload,
            verified.signer_key_arn,
        )
        self._verify_task3_signature(payload, signature, verified)
        return payload

    def verify_contract_authority(
        self,
        repository_root: Path,
        contract_bytes: bytes,
    ) -> VerifiedAwsAuthority:
        boundary_path = (
            repository_root / AWS_BOUNDARY_CONFIG_PATH.relative_to(ROOT)
        )
        boundary_record = _read_regular_file(
            boundary_path,
            label="local AWS evaluator boundary record",
            maximum=_MAX_AUTHORITY_BYTES,
        )
        _parse_aws_boundary_record(boundary_record)
        local_path = repository_root / AUTHORITY_CONFIG_PATH.relative_to(ROOT)
        local_record = _read_regular_file(
            local_path,
            label="local contract authority record",
            maximum=_MAX_AUTHORITY_BYTES,
        )
        parsed = _parse_contract_authority_record(
            local_record,
            contract_bytes,
            boundary_record,
        )
        remote_record, remote_ref = self._get_object(
            parsed.authority_record_key,
            label="contract authority record",
            maximum=_MAX_AUTHORITY_BYTES,
        )
        if remote_record != local_record:
            raise AwsAuthorityError(
                "signed AWS contract authority record differs from local policy"
            )
        signer_key_arn = self._resolve_signer_key()
        signature_document, signature_ref = self._get_object(
            parsed.authority_signature_key,
            label="contract authority signature",
            maximum=_MAX_SIGNATURE_DOCUMENT_BYTES,
        )
        signature = _parse_signature_document(
            signature_document,
            remote_record,
            signer_key_arn,
        )
        self._verify_kms(remote_record, signature, signer_key_arn)
        sealed_key_arn = self._resolve_sealed_key()
        checkpoint_key_arn = self._resolve_checkpoint_key()
        if checkpoint_key_arn == sealed_key_arn:
            raise AwsAuthorityError(
                "checkpoint and evaluator sealed-gold KMS keys must be distinct"
            )
        return VerifiedAwsAuthority(
            contract_sha256=parsed.contract_sha256,
            record_sha256=hashlib.sha256(remote_record).hexdigest(),
            record_version_id=remote_ref.version_id,
            signature_version_id=signature_ref.version_id,
            signer_key_arn=signer_key_arn,
            sealed_gold_kms_key_arn=sealed_key_arn,
            checkpoint_kms_key_arn=checkpoint_key_arn,
        )

    @staticmethod
    def _precondition_failed(error: Exception) -> bool:
        response = getattr(error, "response", None)
        if not isinstance(response, Mapping):
            return False
        details = response.get("Error")
        return (
            isinstance(details, Mapping)
            and details.get("Code") in {"PreconditionFailed", "412"}
        )

    def _put_immutable(
        self,
        key: str,
        payload: bytes,
        *,
        label: str,
        encryption: str,
        kms_key_arn: str | None = None,
    ) -> S3ObjectVersion:
        if not isinstance(payload, bytes) or not payload:
            raise AwsAuthorityError(f"{label} payload is invalid")
        digest = hashlib.sha256(payload).hexdigest()
        kwargs: dict[str, Any] = {
            "Body": payload,
            "Bucket": STORAGE_BUCKET,
            "ChecksumAlgorithm": "SHA256",
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(payload).digest()
            ).decode("ascii"),
            "ContentType": "application/json",
            "IfNoneMatch": "*",
            "Key": key,
            "Metadata": {"sha256": digest},
            "ServerSideEncryption": encryption,
        }
        if kms_key_arn is not None:
            kwargs["SSEKMSKeyId"] = kms_key_arn
        try:
            response = self._s3.put_object(**kwargs)
        except Exception as error:
            if not self._precondition_failed(error):
                raise AwsAuthorityError(f"S3 {label} write failed closed") from error
            existing, ref = self._get_object(
                key,
                label=label,
                maximum=max(len(payload), 1),
            )
            if existing != payload:
                raise AwsAuthorityError(
                    f"existing immutable {label} object differs"
                ) from error
            return self._validate_stored_ref(
                ref,
                label=label,
                encryption=encryption,
                kms_key_arn=kms_key_arn,
            )
        version_id = self._version_id(response, label)
        stored, ref = self._get_object(
            key,
            label=label,
            maximum=max(len(payload), 1),
            version_id=version_id,
        )
        if stored != payload:
            raise AwsAuthorityError(f"stored {label} bytes differ")
        return self._validate_stored_ref(
            ref,
            label=label,
            encryption=encryption,
            kms_key_arn=kms_key_arn,
        )

    @staticmethod
    def _validate_stored_ref(
        ref: S3ObjectVersion,
        *,
        label: str,
        encryption: str,
        kms_key_arn: str | None,
    ) -> S3ObjectVersion:
        if (
            ref.bucket != STORAGE_BUCKET
            or ref.server_side_encryption != encryption
            or ref.kms_key_arn != kms_key_arn
        ):
            raise AwsAuthorityError(f"{label} encryption/storage identity differs")
        return ref

    def put_model_visible(
        self,
        transaction_id: str,
        payload: bytes,
    ) -> S3ObjectVersion:
        self.require_evaluator_role()
        key = (
            f"{STORAGE_PREFIX}/model-visible/quarantine/"
            f"{transaction_id}/release.json"
        )
        return self._put_immutable(
            key,
            payload,
            label="model-visible release",
            encryption="AES256",
        )

    def put_sealed_gold(
        self,
        transaction_id: str,
        payload: bytes,
        kms_key_arn: str,
    ) -> S3ObjectVersion:
        self.require_evaluator_role()
        if not _valid_key_arn(kms_key_arn):
            raise AwsAuthorityError("sealed-gold KMS key ARN is invalid")
        key = (
            f"{STORAGE_PREFIX}/evaluator-only/quarantine/"
            f"{transaction_id}/gold.json"
        )
        return self._put_immutable(
            key,
            payload,
            label="sealed-gold release",
            encryption="aws:kms",
            kms_key_arn=kms_key_arn,
        )

    def sign(
        self,
        message: bytes,
        verified: VerifiedAwsAuthority,
    ) -> bytes:
        self.require_evaluator_role()
        if (
            not isinstance(message, bytes)
            or not message
            or len(message) > _KMS_RAW_MESSAGE_LIMIT
            or not _valid_key_arn(verified.signer_key_arn)
        ):
            raise AwsAuthorityError("activation signing request is invalid")
        response = self._aws_call(
            "KMS activation signing",
            self._kms.sign,
            KeyId=verified.signer_key_arn,
            Message=message,
            MessageType=_MESSAGE_TYPE,
            SigningAlgorithm=_SIGNING_ALGORITHM,
        )
        signature = response.get("Signature")
        if (
            not isinstance(signature, bytes)
            or not signature
            or response.get("KeyId") != verified.signer_key_arn
            or response.get("SigningAlgorithm") != _SIGNING_ALGORITHM
        ):
            raise AwsAuthorityError("KMS activation signature response differs")
        return signature

    def verify(
        self,
        message: bytes,
        signature: bytes,
        verified: VerifiedAwsAuthority,
    ) -> None:
        self._verify_kms(message, signature, verified.signer_key_arn)

    def put_activation(self, payload: bytes) -> ActivationPublication:
        self.require_evaluator_role()
        if (
            not isinstance(payload, bytes)
            or not payload
            or len(payload) > _MAX_ACTIVATION_BYTES
        ):
            raise AwsAuthorityError("signed activation payload is invalid")
        digest = hashlib.sha256(payload).hexdigest()
        kwargs = {
            "Body": payload,
            "Bucket": STORAGE_BUCKET,
            "ChecksumAlgorithm": "SHA256",
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(payload).digest()
            ).decode("ascii"),
            "ContentType": "application/json",
            "IfNoneMatch": "*",
            "Key": ACTIVATION_KEY,
            "Metadata": {"sha256": digest},
            "ServerSideEncryption": "AES256",
        }
        try:
            response = self._s3.put_object(**kwargs)
            version_id = self._version_id(response, "signed activation")
            stored, ref = self._get_object(
                ACTIVATION_KEY,
                label="signed activation",
                maximum=_MAX_ACTIVATION_BYTES,
                version_id=version_id,
            )
            if stored != payload:
                raise AwsAuthorityError(
                    "stored signed activation bytes differ"
                )
            ref = self._validate_stored_ref(
                ref,
                label="signed activation",
                encryption="AES256",
                kms_key_arn=None,
            )
        except Exception as write_error:
            try:
                existing, existing_ref = self._get_stable_current_object(
                    ACTIVATION_KEY,
                    label="signed activation",
                    maximum=_MAX_ACTIVATION_BYTES,
                )
                existing_ref = self._validate_stored_ref(
                    existing_ref,
                    label="signed activation",
                    encryption="AES256",
                    kms_key_arn=None,
                )
            except Exception as existing_error:
                raise AwsAuthorityError(
                    "S3 signed activation write failed without a stable "
                    "immutable existing object"
                ) from existing_error
            return ActivationPublication(
                envelope_bytes=existing,
                object_ref=existing_ref,
                created=False,
            )
        return ActivationPublication(
            envelope_bytes=stored,
            object_ref=ref,
            created=True,
        )

    def read_activation(self) -> tuple[bytes, S3ObjectVersion]:
        return self._get_stable_current_object(
            ACTIVATION_KEY,
            label="signed activation",
            maximum=_MAX_ACTIVATION_BYTES,
        )

    @staticmethod
    def _require_ref(
        ref: S3ObjectVersion,
        *,
        label: str,
        prefix: str,
        encryption: str,
        kms_key_arn: str | None,
    ) -> None:
        if (
            not isinstance(ref, S3ObjectVersion)
            or ref.bucket != STORAGE_BUCKET
            or not ref.key.startswith(prefix)
            or not ref.version_id
            or ref.bytes <= 0
            or not _is_sha256(ref.sha256)
            or ref.server_side_encryption != encryption
            or ref.kms_key_arn != kms_key_arn
        ):
            raise AwsAuthorityError(f"{label} reference differs")

    def read_model_visible(self, ref: S3ObjectVersion) -> bytes:
        self._require_ref(
            ref,
            label="model-visible release",
            prefix=f"{STORAGE_PREFIX}/model-visible/quarantine/",
            encryption="AES256",
            kms_key_arn=None,
        )
        payload, actual = self._get_object(
            ref.key,
            label="model-visible release",
            maximum=_MAX_RELEASE_BYTES,
            version_id=ref.version_id,
        )
        if actual != ref:
            raise AwsAuthorityError("model-visible object differs from activation")
        return payload

    def read_sealed_gold(self, ref: S3ObjectVersion) -> bytes:
        self.require_evaluator_role()
        self._require_ref(
            ref,
            label="sealed-gold release",
            prefix=f"{STORAGE_PREFIX}/evaluator-only/quarantine/",
            encryption="aws:kms",
            kms_key_arn=ref.kms_key_arn,
        )
        payload, actual = self._get_object(
            ref.key,
            label="sealed-gold release",
            maximum=_MAX_RELEASE_BYTES,
            version_id=ref.version_id,
        )
        if actual != ref:
            raise AwsAuthorityError("sealed-gold object differs from activation")
        return payload

    @staticmethod
    def _task3_run_id(arm: str, seed: int) -> str:
        if arm not in _TASK3_ARMS or seed not in _TASK3_SEEDS:
            raise AwsAuthorityError("Task 3 cell is outside the frozen matrix")
        return f"d135m_{arm}_reasoning_v3_s{seed}"

    @classmethod
    def _task3_key(
        cls,
        operation: str,
        arm: str,
        seed: int,
        step: int,
    ) -> str:
        if operation not in {"checkpoint", "evidence", "result"}:
            raise AwsAuthorityError("Task 3 operation is not approved")
        run = cls._task3_run_id(arm, seed)
        if step not in _TASK3_STEPS:
            raise AwsAuthorityError("Task 3 cell is outside the frozen matrix")
        if operation == "checkpoint":
            return f"{TASK3_CHECKPOINT_PREFIX}{run}/step{step:07d}.pt"
        if operation == "evidence":
            return f"{TASK3_EVIDENCE_PREFIX}{run}/step{step:07d}/gates.json"
        return f"{TASK3_RESULT_PREFIX}{arm}/seed-{seed}/step-{step:07d}.json"

    @classmethod
    def _allowed_task3_checkpoint_keys(cls) -> frozenset[str]:
        return frozenset(
            cls._task3_key("checkpoint", arm, seed, step)
            for arm in _TASK3_ARMS
            for seed in _TASK3_SEEDS
            for step in _TASK3_STEPS
        )

    @classmethod
    def _allowed_task3_evidence_keys(cls) -> frozenset[str]:
        keys = {
            f"{TASK3_EVIDENCE_PREFIX}runtime/runtime-lock.json",
            f"{TASK3_EVIDENCE_PREFIX}corpus/receipt.json",
        }
        for arm in _TASK3_ARMS:
            for seed in _TASK3_SEEDS:
                run = cls._task3_run_id(arm, seed)
                keys.add(f"{TASK3_EVIDENCE_PREFIX}{run}/config.json")
                for kind in _TASK3_PAIRING_KINDS:
                    keys.add(
                        f"{TASK3_EVIDENCE_PREFIX}{run}/pairing/{kind}.json"
                    )
                for step in _TASK3_STEPS:
                    keys.add(cls._task3_key("evidence", arm, seed, step))
        return frozenset(keys)

    @staticmethod
    def _validate_task3_authority(
        verified: VerifiedAwsAuthority,
    ) -> None:
        if (
            not isinstance(verified, VerifiedAwsAuthority)
            or not _valid_key_arn(verified.signer_key_arn)
            or not _valid_key_arn(verified.sealed_gold_kms_key_arn)
            or not _valid_key_arn(verified.checkpoint_kms_key_arn)
            or verified.checkpoint_kms_key_arn
            == verified.sealed_gold_kms_key_arn
            or not verified.record_version_id
            or not verified.signature_version_id
            or not _is_sha256(verified.record_sha256)
            or not _is_sha256(verified.contract_sha256)
        ):
            raise AwsAuthorityError("Task 3 verified authority differs")

    @classmethod
    def _validate_task3_ref(
        cls,
        ref: S3ObjectVersion,
        verified: VerifiedAwsAuthority,
        *,
        label: str,
        expected_kms_key_arn: str,
        exact_key: str | None = None,
        allowed_keys: frozenset[str] | None = None,
    ) -> None:
        cls._validate_task3_authority(verified)
        valid_key = (
            isinstance(ref, S3ObjectVersion)
            and isinstance(ref.key, str)
            and (
                (exact_key is not None and ref.key == exact_key)
                or (allowed_keys is not None and ref.key in allowed_keys)
            )
        )
        if (
            not valid_key
            or ref.bucket != STORAGE_BUCKET
            or not ref.version_id
            or ref.bytes <= 0
            or not _is_sha256(ref.sha256)
            or ref.server_side_encryption != "aws:kms"
            or ref.kms_key_arn != expected_kms_key_arn
        ):
            raise AwsAuthorityError(f"Task 3 {label} reference differs")

    def _read_exact_task3_object(
        self,
        ref: S3ObjectVersion,
        verified: VerifiedAwsAuthority,
        *,
        label: str,
        maximum: int,
        expected_kms_key_arn: str,
        exact_key: str | None = None,
        allowed_keys: frozenset[str] | None = None,
    ) -> tuple[bytes, S3ObjectVersion]:
        self.require_evaluator_role()
        self._validate_task3_ref(
            ref,
            verified,
            label=label,
            expected_kms_key_arn=expected_kms_key_arn,
            exact_key=exact_key,
            allowed_keys=allowed_keys,
        )
        payload, actual = self._get_object(
            ref.key,
            label=f"Task 3 {label}",
            maximum=maximum,
            version_id=ref.version_id,
        )
        if actual != ref:
            raise AwsAuthorityError(
                f"Task 3 {label} exact object version differs"
            )
        return payload, actual

    def read_checkpoint(
        self,
        ref: S3ObjectVersion,
        verified: VerifiedAwsAuthority,
    ) -> tuple[bytes, S3ObjectVersion]:
        return self._read_exact_task3_object(
            ref,
            verified,
            label="checkpoint",
            maximum=_MAX_TASK3_CHECKPOINT_BYTES,
            expected_kms_key_arn=verified.checkpoint_kms_key_arn,
            allowed_keys=self._allowed_task3_checkpoint_keys(),
        )

    def read_task3_evidence(
        self,
        ref: S3ObjectVersion,
        verified: VerifiedAwsAuthority,
    ) -> tuple[bytes, S3ObjectVersion]:
        return self._read_exact_task3_object(
            ref,
            verified,
            label="evidence",
            maximum=_MAX_TASK3_EVIDENCE_BYTES,
            expected_kms_key_arn=verified.checkpoint_kms_key_arn,
            allowed_keys=self._allowed_task3_evidence_keys(),
        )

    def _put_signed_task3(
        self,
        key: str,
        payload: bytes,
        verified: VerifiedAwsAuthority,
        *,
        label: str,
        maximum: int,
    ) -> Task3Publication:
        self.require_evaluator_role()
        self._validate_task3_authority(verified)
        if not isinstance(payload, bytes) or not payload or len(payload) > maximum:
            raise AwsAuthorityError(f"Task 3 {label} payload is invalid")
        _parse_canonical_json(
            payload,
            label=f"Task 3 {label} payload",
            maximum=maximum,
        )
        envelope = self._task3_signed_envelope(payload, verified)
        digest = hashlib.sha256(envelope).hexdigest()
        kwargs = {
            "Body": envelope,
            "Bucket": STORAGE_BUCKET,
            "ChecksumAlgorithm": "SHA256",
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(envelope).digest()
            ).decode("ascii"),
            "ContentType": "application/json",
            "IfNoneMatch": "*",
            "Key": key,
            "Metadata": {"sha256": digest},
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": verified.sealed_gold_kms_key_arn,
        }
        try:
            response = self._s3.put_object(**kwargs)
            version_id = self._version_id(response, f"Task 3 {label}")
            stored, ref = self._get_object(
                key,
                label=f"Task 3 {label}",
                maximum=maximum + _MAX_SIGNATURE_DOCUMENT_BYTES,
                version_id=version_id,
            )
            self._validate_task3_ref(
                ref,
                verified,
                label=label,
                expected_kms_key_arn=verified.sealed_gold_kms_key_arn,
                exact_key=key,
            )
            stored_payload = self._verify_task3_envelope(
                stored,
                verified,
                label=label,
                maximum=maximum + _MAX_SIGNATURE_DOCUMENT_BYTES,
            )
            if stored != envelope or stored_payload != payload:
                raise AwsAuthorityError(
                    f"stored Task 3 {label} bytes differ"
                )
            return Task3Publication(payload, stored, ref, True)
        except Exception as write_error:
            try:
                existing, ref = self._get_stable_current_object(
                    key,
                    label=f"Task 3 {label}",
                    maximum=maximum + _MAX_SIGNATURE_DOCUMENT_BYTES,
                )
                self._validate_task3_ref(
                    ref,
                    verified,
                    label=label,
                    expected_kms_key_arn=verified.sealed_gold_kms_key_arn,
                    exact_key=key,
                )
                existing_payload = self._verify_task3_envelope(
                    existing,
                    verified,
                    label=label,
                    maximum=maximum + _MAX_SIGNATURE_DOCUMENT_BYTES,
                )
            except Exception as existing_error:
                raise AwsAuthorityError(
                    f"Task 3 {label} write failed without a stable "
                    "signed immutable object"
                ) from existing_error
            if existing_payload != payload:
                raise AwsAuthorityError(
                    f"existing immutable Task 3 {label} unsigned payload differs"
                ) from write_error
            return Task3Publication(
                existing_payload,
                existing,
                ref,
                False,
            )

    def _read_signed_task3(
        self,
        key: str,
        verified: VerifiedAwsAuthority,
        *,
        label: str,
        maximum: int,
    ) -> tuple[bytes, S3ObjectVersion]:
        self.require_evaluator_role()
        self._validate_task3_authority(verified)
        envelope, ref = self._get_stable_current_object(
            key,
            label=f"Task 3 {label}",
            maximum=maximum + _MAX_SIGNATURE_DOCUMENT_BYTES,
        )
        self._validate_task3_ref(
            ref,
            verified,
            label=label,
            expected_kms_key_arn=verified.sealed_gold_kms_key_arn,
            exact_key=key,
        )
        payload = self._verify_task3_envelope(
            envelope,
            verified,
            label=label,
            maximum=maximum + _MAX_SIGNATURE_DOCUMENT_BYTES,
        )
        return payload, ref

    def put_checkpoint_manifest(
        self,
        payload: bytes,
        verified: VerifiedAwsAuthority,
    ) -> Task3Publication:
        return self._put_signed_task3(
            TASK3_CHECKPOINT_MANIFEST_KEY,
            payload,
            verified,
            label="checkpoint manifest",
            maximum=_MAX_TASK3_MANIFEST_BYTES,
        )

    def read_checkpoint_manifest(
        self,
        verified: VerifiedAwsAuthority,
    ) -> tuple[bytes, S3ObjectVersion]:
        return self._read_signed_task3(
            TASK3_CHECKPOINT_MANIFEST_KEY,
            verified,
            label="checkpoint manifest",
            maximum=_MAX_TASK3_MANIFEST_BYTES,
        )

    def put_checkpoint_result(
        self,
        arm: str,
        seed: int,
        step: int,
        payload: bytes,
        verified: VerifiedAwsAuthority,
    ) -> Task3Publication:
        key = self._task3_key("result", arm, seed, step)
        return self._put_signed_task3(
            key,
            payload,
            verified,
            label="checkpoint result",
            maximum=_MAX_TASK3_RESULT_BYTES,
        )

    def read_checkpoint_result(
        self,
        arm: str,
        seed: int,
        step: int,
        verified: VerifiedAwsAuthority | None = None,
    ) -> tuple[bytes, S3ObjectVersion]:
        self.require_evaluator_role()
        if verified is None:
            raise AwsAuthorityError(
                "Task 3 checkpoint result read requires verified authority"
            )
        key = self._task3_key("result", arm, seed, step)
        return self._read_signed_task3(
            key,
            verified,
            label="checkpoint result",
            maximum=_MAX_TASK3_RESULT_BYTES,
        )

    def put_scientific_report(
        self,
        payload: bytes,
        verified: VerifiedAwsAuthority,
    ) -> Task3Publication:
        return self._put_signed_task3(
            TASK3_REPORT_KEY,
            payload,
            verified,
            label="scientific report",
            maximum=_MAX_TASK3_REPORT_BYTES,
        )

    def read_scientific_report(
        self,
        verified: VerifiedAwsAuthority | None = None,
    ) -> tuple[bytes, S3ObjectVersion]:
        self.require_evaluator_role()
        if verified is None:
            raise AwsAuthorityError(
                "Task 3 scientific report read requires verified authority"
            )
        return self._read_signed_task3(
            TASK3_REPORT_KEY,
            verified,
            label="scientific report",
            maximum=_MAX_TASK3_REPORT_BYTES,
        )


class _FixedAwsTrainerAuthority:
    """Fixed write-only checkpoint/evidence authority for the trainer role."""

    def __init__(self) -> None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as error:
            raise AwsAuthorityError(
                "boto3/botocore are required for fixed AWS trainer authority"
            ) from error
        try:
            session = boto3.Session(region_name=AWS_REGION)
            config = Config(
                region_name=AWS_REGION,
                retries={"max_attempts": 3, "mode": "standard"},
                signature_version="v4",
            )
            common = {"config": config, "region_name": AWS_REGION, "verify": True}
            self._sts = session.client(
                "sts",
                endpoint_url=f"https://sts.{AWS_REGION}.amazonaws.com",
                **common,
            )
            self._kms = session.client(
                "kms",
                endpoint_url=f"https://kms.{AWS_REGION}.amazonaws.com",
                **common,
            )
            self._s3 = session.client(
                "s3",
                endpoint_url=f"https://s3.{AWS_REGION}.amazonaws.com",
                **common,
            )
        except Exception as error:
            raise AwsAuthorityError(
                "fixed AWS trainer clients could not be initialized"
            ) from error

    @staticmethod
    def _aws_call(label: str, operation, **kwargs):
        try:
            return operation(**kwargs)
        except Exception as error:
            raise AwsAuthorityError(f"{label} failed closed") from error

    def require_trainer_role(self) -> None:
        identity = self._aws_call(
            "AWS trainer identity verification",
            self._sts.get_caller_identity,
        )
        _trainer_role_arn_from_identity(identity)

    def _resolve_checkpoint_key(self) -> str:
        response = self._aws_call(
            "KMS trainer checkpoint key resolution",
            self._kms.describe_key,
            KeyId=CHECKPOINT_KMS_KEY_ALIAS,
        )
        metadata = response.get("KeyMetadata")
        if not isinstance(metadata, Mapping):
            raise AwsAuthorityError("KMS checkpoint metadata is unavailable")
        arn = metadata.get("Arn")
        if (
            not _valid_key_arn(arn)
            or metadata.get("Enabled") is not True
            or metadata.get("KeyState") != "Enabled"
            or metadata.get("KeyUsage") != "ENCRYPT_DECRYPT"
            or metadata.get("KeySpec") != "SYMMETRIC_DEFAULT"
        ):
            raise AwsAuthorityError(
                "KMS checkpoint key does not match frozen trainer policy"
            )
        return str(arn)

    @staticmethod
    def _trainer_run_id(arm: str, seed: int) -> str:
        return _FixedAwsEvaluatorAuthority._task3_run_id(arm, seed)

    @classmethod
    def _trainer_key(
        cls,
        operation: str,
        arm: str | None = None,
        seed: int | None = None,
        step: int | None = None,
        *,
        pairing_kind: str | None = None,
    ) -> str:
        if operation == "runtime_lock":
            return f"{TASK3_EVIDENCE_PREFIX}runtime/runtime-lock.json"
        if operation == "corpus_receipt":
            return f"{TASK3_EVIDENCE_PREFIX}corpus/receipt.json"
        if operation not in {"checkpoint", "gate", "run_config", "pairing"}:
            raise AwsAuthorityError("trainer operation is not approved")
        if not isinstance(arm, str) or not isinstance(seed, int):
            raise AwsAuthorityError("trainer run identity is invalid")
        run = cls._trainer_run_id(arm, seed)
        if operation == "run_config":
            return f"{TASK3_EVIDENCE_PREFIX}{run}/config.json"
        if operation == "pairing":
            if pairing_kind not in _TASK3_PAIRING_KINDS:
                raise AwsAuthorityError("trainer pairing evidence kind is invalid")
            return (
                f"{TASK3_EVIDENCE_PREFIX}{run}/pairing/{pairing_kind}.json"
            )
        if step not in _TASK3_STEPS:
            raise AwsAuthorityError("trainer cell is outside the frozen matrix")
        if operation == "checkpoint":
            return f"{TASK3_CHECKPOINT_PREFIX}{run}/step{step:07d}.pt"
        return f"{TASK3_EVIDENCE_PREFIX}{run}/step{step:07d}/gates.json"

    def _put_checkpoint_artifact(
        self,
        key: str,
        payload: bytes,
        *,
        label: str,
        maximum: int,
        canonical_json: bool,
    ) -> S3ObjectVersion:
        self.require_trainer_role()
        if (
            not isinstance(payload, bytes)
            or not payload
            or len(payload) > maximum
            or not (
                key.startswith(TASK3_CHECKPOINT_PREFIX)
                or key.startswith(TASK3_EVIDENCE_PREFIX)
            )
        ):
            raise AwsAuthorityError(f"trainer {label} payload or key is invalid")
        if canonical_json:
            _parse_canonical_json(payload, label=label, maximum=maximum)
        checkpoint_key_arn = self._resolve_checkpoint_key()
        digest = hashlib.sha256(payload).hexdigest()
        response = self._aws_call(
            f"S3 trainer {label} write",
            self._s3.put_object,
            Body=payload,
            Bucket=STORAGE_BUCKET,
            ChecksumAlgorithm="SHA256",
            ChecksumSHA256=base64.b64encode(
                hashlib.sha256(payload).digest()
            ).decode("ascii"),
            ContentType=(
                "application/json"
                if canonical_json
                else "application/octet-stream"
            ),
            IfNoneMatch="*",
            Key=key,
            Metadata={"sha256": digest},
            ServerSideEncryption="aws:kms",
            SSEKMSKeyId=checkpoint_key_arn,
        )
        if (
            response.get("ServerSideEncryption") != "aws:kms"
            or response.get("SSEKMSKeyId") != checkpoint_key_arn
        ):
            raise AwsAuthorityError(f"trainer {label} encryption differs")
        version_id = _FixedAwsEvaluatorAuthority._version_id(response, label)
        return S3ObjectVersion(
            bucket=STORAGE_BUCKET,
            key=key,
            version_id=version_id,
            bytes=len(payload),
            sha256=digest,
            server_side_encryption="aws:kms",
            kms_key_arn=checkpoint_key_arn,
        )

    def put_checkpoint(
        self,
        arm: str,
        seed: int,
        step: int,
        payload: bytes,
    ) -> S3ObjectVersion:
        return self._put_checkpoint_artifact(
            self._trainer_key("checkpoint", arm, seed, step),
            payload,
            label="checkpoint",
            maximum=_MAX_TASK3_CHECKPOINT_BYTES,
            canonical_json=False,
        )

    def put_gate_evidence(
        self,
        arm: str,
        seed: int,
        step: int,
        payload: bytes,
    ) -> S3ObjectVersion:
        return self._put_checkpoint_artifact(
            self._trainer_key("gate", arm, seed, step),
            payload,
            label="gate evidence",
            maximum=_MAX_TASK3_EVIDENCE_BYTES,
            canonical_json=True,
        )

    def put_run_config_evidence(
        self,
        arm: str,
        seed: int,
        payload: bytes,
    ) -> S3ObjectVersion:
        return self._put_checkpoint_artifact(
            self._trainer_key("run_config", arm, seed),
            payload,
            label="run-config evidence",
            maximum=_MAX_TASK3_EVIDENCE_BYTES,
            canonical_json=True,
        )

    def put_pairing_evidence(
        self,
        arm: str,
        seed: int,
        pairing_kind: str,
        payload: bytes,
    ) -> S3ObjectVersion:
        return self._put_checkpoint_artifact(
            self._trainer_key(
                "pairing",
                arm,
                seed,
                pairing_kind=pairing_kind,
            ),
            payload,
            label="pairing evidence",
            maximum=_MAX_TASK3_EVIDENCE_BYTES,
            canonical_json=True,
        )

    def put_runtime_lock(self, payload: bytes) -> S3ObjectVersion:
        return self._put_checkpoint_artifact(
            self._trainer_key("runtime_lock"),
            payload,
            label="runtime-lock evidence",
            maximum=_MAX_TASK3_EVIDENCE_BYTES,
            canonical_json=True,
        )

    def put_corpus_receipt(self, payload: bytes) -> S3ObjectVersion:
        return self._put_checkpoint_artifact(
            self._trainer_key("corpus_receipt"),
            payload,
            label="corpus-receipt evidence",
            maximum=_MAX_TASK3_EVIDENCE_BYTES,
            canonical_json=True,
        )


def _new_fixed_aws_authority() -> _FixedAwsEvaluatorAuthority:
    return _FixedAwsEvaluatorAuthority()


def _new_fixed_trainer_authority() -> _FixedAwsTrainerAuthority:
    return _FixedAwsTrainerAuthority()


__all__ = [
    "ACTIVATION_KEY",
    "AUTHORITY_CONFIG_PATH",
    "AUTHORITY_RECORD_KEY",
    "AUTHORITY_SIGNATURE_KEY",
    "AWS_BOUNDARY_CONFIG_PATH",
    "AWS_REGION",
    "ActivationPublication",
    "CHECKPOINT_KMS_KEY_ALIAS",
    "CHECKPOINT_STORAGE_PREFIX",
    "EVALUATOR_ROLE_ARN",
    "SEALED_GOLD_KMS_KEY_ALIAS",
    "SIGNER_KEY_ALIAS",
    "STORAGE_BUCKET",
    "STORAGE_PREFIX",
    "TASK3_CHECKPOINT_MANIFEST_KEY",
    "TASK3_CHECKPOINT_PREFIX",
    "TASK3_EVIDENCE_PREFIX",
    "TASK3_REPORT_KEY",
    "TASK3_RESULT_PREFIX",
    "AwsAuthorityError",
    "S3ObjectVersion",
    "Task3Publication",
    "TRAINER_ROLE_ARN",
    "VerifiedAwsAuthority",
]
