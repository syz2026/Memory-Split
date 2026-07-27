from __future__ import annotations

import base64
import builtins
import hashlib
import io
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

import evals.reasoning_v3.contracts as contracts_module
import evals.reasoning_v3.sealing as sealing_module
from corpusgen.parallel.canonical import canonical_json_bytes
from evals.reasoning_v3.aws_authority import (
    ACTIVATION_KEY,
    AUTHORITY_CONFIG_PATH,
    AUTHORITY_RECORD_KEY,
    AUTHORITY_SIGNATURE_KEY,
    AWS_BOUNDARY_CONFIG_PATH,
    AWS_REGION,
    EVALUATOR_ROLE_ARN,
    SEALED_GOLD_KMS_KEY_ALIAS,
    SIGNER_KEY_ALIAS,
    STORAGE_BUCKET,
    STORAGE_PREFIX,
    ActivationPublication,
    AwsAuthorityError,
    S3ObjectVersion,
    VerifiedAwsAuthority,
    _FixedAwsEvaluatorAuthority,
    _parse_aws_boundary_record,
    _parse_contract_authority_record,
    _parse_signature_document,
    _role_arn_from_identity,
    _signature_document_bytes,
)
from evals.reasoning_v3.contracts import (
    DEFAULT_CONTRACT_PATH,
    load_evaluation_contract,
)
from evals.reasoning_v3.generate import (
    CandidateRow,
    FrozenEvaluationPaths,
    ProvenanceCommitment,
    _build_evaluation_registry,
)
from evals.reasoning_v3.sealing import (
    EvaluationSealingError,
    _build_release_bundle,
    _load_activated_bundle_from_aws,
    _publish_release_to_aws,
    load_model_visible_release,
    materialize_frozen_evaluation_release,
    validate_frozen_evaluation_release,
)


ROOT = Path(__file__).resolve().parents[1]
_SIGNER_ARN = (
    "arn:aws:kms:us-east-1:${AWS_ACCOUNT_ID}:key/"
    "11111111-1111-4111-8111-111111111111"
)
_SEALED_KEY_ARN = (
    "arn:aws:kms:us-east-1:${AWS_ACCOUNT_ID}:key/"
    "22222222-2222-4222-8222-222222222222"
)
_CHECKPOINT_KEY_ARN = (
    "arn:aws:kms:us-east-1:${AWS_ACCOUNT_ID}:key/"
    "33333333-3333-4333-8333-333333333333"
)


class _FixtureSource:
    def generate(self, task: str, source_index: int) -> CandidateRow:
        question = f"Fixture question {source_index}?"
        answer = "1"
        record = {
            "answer": answer,
            "question": question,
            "source_index": source_index,
            "task": task,
        }
        return CandidateRow(
            task=task,
            source_index=source_index,
            question=question,
            native_answer=answer,
            oracle_answer=answer,
            token_count=20,
            prompt_token_count=10,
            answer_token_count=1,
            record_sha256=hashlib.sha256(canonical_json_bytes(record)).hexdigest(),
        )


def _fixture():
    contract = replace(
        load_evaluation_contract(DEFAULT_CONTRACT_PATH, repository_root=ROOT),
        accepted_items_per_family=2,
    )
    items = _build_evaluation_registry(
        contract,
        _FixtureSource(),
        training_record_keys=frozenset(),
    )
    provenance = ProvenanceCommitment(
        corpus_receipt_sha256=contract.corpus_receipt_sha256,
        source_stage_receipt_sha256=contract.source_stage_receipt_sha256,
        source_tree_commitment_sha256="a" * 64,
        record_manifest_sha256=contract.record_manifest_sha256,
        record_count=7_530_527,
        generator_artifacts_sha256="b" * 64,
        runtime_lock_sha256="c" * 64,
    )
    verified = VerifiedAwsAuthority(
        contract_sha256=contract.sha256,
        record_sha256="d" * 64,
        record_version_id="authority-record-version",
        signature_version_id="authority-signature-version",
        signer_key_arn=_SIGNER_ARN,
        sealed_gold_kms_key_arn=_SEALED_KEY_ARN,
        checkpoint_kms_key_arn=_CHECKPOINT_KEY_ARN,
    )
    return contract, _build_release_bundle(contract, items, provenance), verified


class _FakeAwsAuthority:
    """Private in-memory transport for exercising private orchestration only."""

    def __init__(self, *, crash_once: bool = False):
        self.objects: dict[tuple[str, str], tuple[bytes, S3ObjectVersion]] = {}
        self.activation: tuple[bytes, S3ObjectVersion] | None = None
        self.crash_once = crash_once
        self.failed_requests: list[str] = []
        self.issued_signatures: dict[bytes, bytes] = {}
        self.signature_history: list[bytes] = []

    def _signature(self, message: bytes) -> bytes:
        nonce = len(self.signature_history).to_bytes(8, "big")
        signature = (
            b"fixture-nondeterministic-ecdsa\x00"
            + nonce
            + hashlib.sha384(message).digest()
        )
        self.issued_signatures[signature] = message
        self.signature_history.append(signature)
        return signature

    @staticmethod
    def _ref(
        key: str,
        payload: bytes,
        *,
        version: str,
        encryption: str,
        kms_key_arn: str | None = None,
    ) -> S3ObjectVersion:
        return S3ObjectVersion(
            bucket=STORAGE_BUCKET,
            key=key,
            version_id=version,
            bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            server_side_encryption=encryption,
            kms_key_arn=kms_key_arn,
        )

    def put_model_visible(self, transaction_id: str, payload: bytes) -> S3ObjectVersion:
        key = (
            f"{STORAGE_PREFIX}/model-visible/quarantine/"
            f"{transaction_id}/release.json"
        )
        ref = self._ref(
            key,
            payload,
            version="model-visible-version",
            encryption="AES256",
        )
        self.objects[(key, ref.version_id)] = (payload, ref)
        return ref

    def put_sealed_gold(
        self,
        transaction_id: str,
        payload: bytes,
        kms_key_arn: str,
    ) -> S3ObjectVersion:
        key = (
            f"{STORAGE_PREFIX}/evaluator-only/quarantine/"
            f"{transaction_id}/gold.json"
        )
        if self.crash_once:
            self.crash_once = False
            self.failed_requests.append(key)
            raise RuntimeError("simulated crash during S3 request body")
        ref = self._ref(
            key,
            payload,
            version="sealed-gold-version",
            encryption="aws:kms",
            kms_key_arn=kms_key_arn,
        )
        self.objects[(key, ref.version_id)] = (payload, ref)
        return ref

    def sign(self, message: bytes, verified: VerifiedAwsAuthority) -> bytes:
        assert verified.signer_key_arn == _SIGNER_ARN
        return self._signature(message)

    def verify(
        self,
        message: bytes,
        signature: bytes,
        verified: VerifiedAwsAuthority,
    ) -> None:
        assert verified.signer_key_arn == _SIGNER_ARN
        if self.issued_signatures.get(signature) != message:
            raise AwsAuthorityError("fixture KMS signature differs")

    def put_activation(self, payload: bytes) -> ActivationPublication:
        ref = self._ref(
            ACTIVATION_KEY,
            payload,
            version="activation-version",
            encryption="AES256",
        )
        if self.activation is None:
            self.activation = (payload, ref)
            return ActivationPublication(
                envelope_bytes=payload,
                object_ref=ref,
                created=True,
            )
        return ActivationPublication(
            envelope_bytes=self.activation[0],
            object_ref=self.activation[1],
            created=False,
        )

    def read_activation(self) -> tuple[bytes, S3ObjectVersion]:
        if self.activation is None:
            raise AwsAuthorityError("activation unavailable")
        return self.activation

    def read_model_visible(self, ref: S3ObjectVersion) -> bytes:
        return self.objects[(ref.key, ref.version_id)][0]

    def read_sealed_gold(self, ref: S3ObjectVersion) -> bytes:
        return self.objects[(ref.key, ref.version_id)][0]


def test_fixed_authority_config_binds_contract_and_approved_aws_boundary():
    assert AWS_REGION == "us-east-1"
    assert EVALUATOR_ROLE_ARN == (
        "arn:aws:iam::${AWS_ACCOUNT_ID}:role/memorysplit-reasoning-v3-evaluator"
    )
    assert SIGNER_KEY_ALIAS == "alias/memorysplit-reasoning-v3-evaluator-v1"
    assert STORAGE_BUCKET == "${MS_S3_BUCKET}"
    assert STORAGE_PREFIX
    assert "/evaluator-only/" in f"/{STORAGE_PREFIX}/evaluator-only/"
    assert SEALED_GOLD_KMS_KEY_ALIAS != SIGNER_KEY_ALIAS

    boundary_payload = AWS_BOUNDARY_CONFIG_PATH.read_bytes()
    boundary = _parse_aws_boundary_record(boundary_payload)
    assert boundary_payload == canonical_json_bytes(json.loads(boundary_payload))
    assert boundary["evaluator_role_arn"] == EVALUATOR_ROLE_ARN
    assert boundary["required_controls"] == {
        "bucket_versioning": "Enabled",
        "deny_non_evaluator_sealed_gold_kms": True,
        "deny_non_evaluator_sealed_gold_s3": True,
        "immutable_activation": "s3_if_none_match_and_version_bound",
        "sealed_gold_encryption": "aws:kms",
    }

    payload = AUTHORITY_CONFIG_PATH.read_bytes()
    record = _parse_contract_authority_record(
        payload,
        DEFAULT_CONTRACT_PATH.read_bytes(),
        boundary_payload,
    )
    assert payload == canonical_json_bytes(json.loads(payload))
    assert record.contract_sha256 == hashlib.sha256(
        DEFAULT_CONTRACT_PATH.read_bytes()
    ).hexdigest()
    assert record.evaluator_role_arn == EVALUATOR_ROLE_ARN
    assert record.signer_key_alias == SIGNER_KEY_ALIAS
    assert record.storage_bucket == STORAGE_BUCKET
    assert record.storage_prefix == STORAGE_PREFIX
    assert record.authority_record_key == AUTHORITY_RECORD_KEY
    assert record.authority_signature_key == AUTHORITY_SIGNATURE_KEY
    assert record.activation_key == ACTIVATION_KEY
    assert record.aws_boundary_sha256 == hashlib.sha256(
        boundary_payload
    ).hexdigest()

    semantically_identical = DEFAULT_CONTRACT_PATH.read_bytes() + b"# drift\n"
    with pytest.raises(AwsAuthorityError, match="contract.*digest"):
        _parse_contract_authority_record(
            payload,
            semantically_identical,
            boundary_payload,
        )


def test_kms_signature_document_is_closed_and_ecdsa_sha384_only():
    message = canonical_json_bytes({"authority": "fixture"})
    document = _signature_document_bytes(
        b"DER fixture signature",
        message,
        _SIGNER_ARN,
    )
    parsed = _parse_signature_document(document, message, _SIGNER_ARN)
    assert parsed == b"DER fixture signature"
    assert json.loads(document)["signing_algorithm"] == "ECDSA_SHA_384"

    changed = json.loads(document)
    changed["signing_algorithm"] = "RSASSA_PSS_SHA_256"
    with pytest.raises(AwsAuthorityError, match="signature"):
        _parse_signature_document(
            canonical_json_bytes(changed),
            message,
            _SIGNER_ARN,
        )


def test_assumed_role_identity_normalizes_only_the_dedicated_evaluator_role():
    assert _role_arn_from_identity(
        {
            "Account": "${AWS_ACCOUNT_ID}",
            "Arn": (
                "arn:aws:sts::${AWS_ACCOUNT_ID}:assumed-role/"
                "memorysplit-reasoning-v3-evaluator/session-1"
            ),
        }
    ) == EVALUATOR_ROLE_ARN
    with pytest.raises(AwsAuthorityError, match="evaluator role"):
        _role_arn_from_identity(
            {
                "Account": "${AWS_ACCOUNT_ID}",
                "Arn": (
                    "arn:aws:sts::${AWS_ACCOUNT_ID}:assumed-role/"
                    "memorysplit-reasoning-v3-trainer/session-1"
                ),
            }
        )


def test_fixed_adapter_uses_conditional_versioned_sse_kms_puts():
    payload = canonical_json_bytes({"sealed": "fixture"})

    class _Sts:
        @staticmethod
        def get_caller_identity():
            return {
                "Account": "${AWS_ACCOUNT_ID}",
                "Arn": (
                    "arn:aws:sts::${AWS_ACCOUNT_ID}:assumed-role/"
                    "memorysplit-reasoning-v3-evaluator/unit-test"
                ),
            }

    class _S3:
        def __init__(self):
            self.put: dict[str, object] | None = None

        def put_object(self, **kwargs):
            self.put = kwargs
            return {"VersionId": "fixture-version"}

        def get_object(self, **kwargs):
            assert self.put is not None
            assert kwargs["VersionId"] == "fixture-version"
            return {
                "Body": io.BytesIO(payload),
                "Metadata": self.put["Metadata"],
                "ServerSideEncryption": "aws:kms",
                "SSEKMSKeyId": _SEALED_KEY_ARN,
                "VersionId": "fixture-version",
            }

    adapter = object.__new__(_FixedAwsEvaluatorAuthority)
    adapter._sts = _Sts()
    adapter._s3 = _S3()
    ref = adapter.put_sealed_gold("a" * 64, payload, _SEALED_KEY_ARN)

    request = adapter._s3.put
    assert request is not None
    assert request["Bucket"] == STORAGE_BUCKET
    assert request["IfNoneMatch"] == "*"
    assert request["ServerSideEncryption"] == "aws:kms"
    assert request["SSEKMSKeyId"] == _SEALED_KEY_ARN
    assert request["ChecksumAlgorithm"] == "SHA256"
    assert request["Body"] == payload
    assert ref.version_id == "fixture-version"
    assert ref.sha256 == hashlib.sha256(payload).hexdigest()


def test_activation_put_recovers_exact_version_after_lost_success_response():
    first_envelope = canonical_json_bytes(
        {"activation": {"value": 1}, "signature": "first"}
    )
    retry_envelope = canonical_json_bytes(
        {"activation": {"value": 1}, "signature": "second"}
    )

    class _Sts:
        @staticmethod
        def get_caller_identity():
            return {
                "Account": "${AWS_ACCOUNT_ID}",
                "Arn": (
                    "arn:aws:sts::${AWS_ACCOUNT_ID}:assumed-role/"
                    "memorysplit-reasoning-v3-evaluator/unit-test"
                ),
            }

    class _PreconditionFailed(Exception):
        response = {"Error": {"Code": "PreconditionFailed"}}

    class _S3:
        def __init__(self):
            self.payload: bytes | None = None
            self.metadata: dict[str, str] | None = None
            self.put_calls = 0
            self.get_calls: list[str | None] = []

        def put_object(self, **kwargs):
            self.put_calls += 1
            if self.payload is None:
                self.payload = kwargs["Body"]
                self.metadata = kwargs["Metadata"]
                raise RuntimeError("simulated lost successful response")
            raise _PreconditionFailed()

        def get_object(self, **kwargs):
            assert self.payload is not None
            assert self.metadata is not None
            version = kwargs.get("VersionId")
            self.get_calls.append(version)
            return {
                "Body": io.BytesIO(self.payload),
                "Metadata": self.metadata,
                "ServerSideEncryption": "AES256",
                "VersionId": "existing-activation-version",
            }

    adapter = object.__new__(_FixedAwsEvaluatorAuthority)
    adapter._sts = _Sts()
    adapter._s3 = _S3()

    recovered = adapter.put_activation(first_envelope)
    retried = adapter.put_activation(retry_envelope)

    assert recovered.created is False
    assert recovered.envelope_bytes == first_envelope
    assert recovered.object_ref.version_id == "existing-activation-version"
    assert retried.created is False
    assert retried.envelope_bytes == first_envelope
    assert retried.object_ref == recovered.object_ref
    assert adapter._s3.get_calls == [
        None,
        "existing-activation-version",
        None,
        None,
        "existing-activation-version",
        None,
    ]


@pytest.mark.parametrize("mode", ["missing_version", "mutable_current"])
def test_activation_put_rejects_unversioned_or_mutable_existing_object(mode: str):
    intended = canonical_json_bytes({"activation": "intended"})
    original = canonical_json_bytes({"activation": "original"})
    changed = canonical_json_bytes({"activation": "changed"})

    class _Sts:
        @staticmethod
        def get_caller_identity():
            return {
                "Account": "${AWS_ACCOUNT_ID}",
                "Arn": (
                    "arn:aws:sts::${AWS_ACCOUNT_ID}:assumed-role/"
                    "memorysplit-reasoning-v3-evaluator/unit-test"
                ),
            }

    class _PreconditionFailed(Exception):
        response = {"Error": {"Code": "PreconditionFailed"}}

    class _S3:
        def __init__(self):
            self.reads = 0

        @staticmethod
        def put_object(**_kwargs):
            raise _PreconditionFailed()

        def get_object(self, **kwargs):
            self.reads += 1
            payload = original
            version = "original-version"
            if mode == "mutable_current" and self.reads == 3:
                payload = changed
                version = "changed-version"
            response = {
                "Body": io.BytesIO(payload),
                "Metadata": {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
                "ServerSideEncryption": "AES256",
            }
            if mode != "missing_version":
                response["VersionId"] = version
            return response

    adapter = object.__new__(_FixedAwsEvaluatorAuthority)
    adapter._sts = _Sts()
    adapter._s3 = _S3()

    with pytest.raises(AwsAuthorityError, match="stable immutable"):
        adapter.put_activation(intended)


def test_public_production_signatures_have_no_authority_or_plaintext_injection():
    assert set(inspect.signature(materialize_frozen_evaluation_release).parameters) == {
        "paths"
    }
    assert set(inspect.signature(validate_frozen_evaluation_release).parameters) == {
        "paths"
    }
    assert not inspect.signature(load_model_visible_release).parameters

    for forbidden in (
        "authenticate_evaluator_access",
        "AuthorityVerification",
        "EvaluatorAccessContext",
        "EvaluatorAuthorityVerifier",
    ):
        assert not hasattr(contracts_module, forbidden)
    assert not hasattr(sealing_module, "load_sealed_gold_release")


def test_public_entry_points_fail_closed_when_fixed_aws_authority_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = FrozenEvaluationPaths(ROOT, tmp_path / "dataset", tmp_path / "source")
    real_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "boto3" or name.startswith("botocore"):
            raise ImportError("simulated unavailable AWS SDK")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    for call in (
        lambda: materialize_frozen_evaluation_release(paths),
        lambda: validate_frozen_evaluation_release(paths),
        load_model_visible_release,
    ):
        with pytest.raises(AwsAuthorityError, match="required"):
            call()


def test_private_aws_publish_signs_versioned_activation_and_never_returns_gold():
    contract, bundle, verified = _fixture()
    authority = _FakeAwsAuthority()

    activated = _publish_release_to_aws(contract, bundle, verified, authority)

    assert not hasattr(activated, "sealed_gold_bytes")
    assert activated.model_visible.version_id == "model-visible-version"
    assert activated.sealed_gold.version_id == "sealed-gold-version"
    assert activated.sealed_gold.server_side_encryption == "aws:kms"
    assert activated.sealed_gold.kms_key_arn == _SEALED_KEY_ARN
    assert "/evaluator-only/" in f"/{activated.sealed_gold.key}"
    assert authority.activation is not None

    envelope = json.loads(authority.activation[0])
    activation = envelope["activation"]
    assert envelope["signature"]["signing_algorithm"] == "ECDSA_SHA_384"
    assert activation["contract_sha256"] == contract.sha256
    assert activation["provenance"]
    assert activation["registry_sha256"] == bundle.registry_sha256
    assert activation["artifacts"]["model_visible"]["version_id"]
    assert activation["artifacts"]["sealed_gold"]["version_id"]
    assert activation["storage"] == {
        "activation_key": ACTIVATION_KEY,
        "bucket": STORAGE_BUCKET,
        "prefix": STORAGE_PREFIX,
    }
    assert activation["authority"] == {
        "contract_authority_record_sha256": verified.record_sha256,
        "contract_authority_record_version_id": verified.record_version_id,
        "contract_authority_signature_version_id": verified.signature_version_id,
        "evaluator_role_arn": EVALUATOR_ROLE_ARN,
        "signer_key_alias": SIGNER_KEY_ALIAS,
        "signer_key_arn": _SIGNER_ARN,
    }

    restored = _load_activated_bundle_from_aws(
        contract,
        verified,
        authority,
        include_sealed=True,
    )
    assert restored == bundle


def test_activation_rerun_accepts_same_payload_with_new_valid_ecdsa_signature():
    contract, bundle, verified = _fixture()
    authority = _FakeAwsAuthority()

    first = _publish_release_to_aws(contract, bundle, verified, authority)
    original_envelope = authority.activation
    second = _publish_release_to_aws(contract, bundle, verified, authority)

    assert len(authority.signature_history) == 2
    assert authority.signature_history[0] != authority.signature_history[1]
    assert authority.activation == original_envelope
    assert second == first


def test_activation_rerun_rejects_same_payload_with_invalid_existing_signature():
    contract, bundle, verified = _fixture()
    authority = _FakeAwsAuthority()
    _publish_release_to_aws(contract, bundle, verified, authority)
    assert authority.activation is not None

    envelope = json.loads(authority.activation[0])
    envelope["signature"]["signature_base64"] = base64.b64encode(
        b"invalid existing signature"
    ).decode("ascii")
    invalid = canonical_json_bytes(envelope)
    authority.activation = (
        invalid,
        authority._ref(
            ACTIVATION_KEY,
            invalid,
            version="invalid-signature-version",
            encryption="AES256",
        ),
    )

    with pytest.raises(AwsAuthorityError, match="signature"):
        _publish_release_to_aws(contract, bundle, verified, authority)


def test_activation_rerun_rejects_different_payload_with_valid_signature():
    contract, bundle, verified = _fixture()
    authority = _FakeAwsAuthority()
    _publish_release_to_aws(contract, bundle, verified, authority)
    assert authority.activation is not None

    envelope = json.loads(authority.activation[0])
    activation = envelope["activation"]
    activation["artifacts"]["model_visible"]["version_id"] = (
        "different-model-visible-version"
    )
    activation_bytes = canonical_json_bytes(activation)
    envelope["signature"] = json.loads(
        _signature_document_bytes(
            authority.sign(activation_bytes, verified),
            activation_bytes,
            _SIGNER_ARN,
        )
    )
    changed = canonical_json_bytes(envelope)
    authority.activation = (
        changed,
        authority._ref(
            ACTIVATION_KEY,
            changed,
            version="different-payload-version",
            encryption="AES256",
        ),
    )

    with pytest.raises(EvaluationSealingError, match="payload differs"):
        _publish_release_to_aws(contract, bundle, verified, authority)


def test_signed_activation_rejects_tampering_before_object_reads():
    contract, bundle, verified = _fixture()
    authority = _FakeAwsAuthority()
    _publish_release_to_aws(contract, bundle, verified, authority)
    assert authority.activation is not None
    envelope = json.loads(authority.activation[0])
    envelope["activation"]["registry_sha256"] = "0" * 64
    tampered = canonical_json_bytes(envelope)
    authority.activation = (
        tampered,
        authority._ref(
            ACTIVATION_KEY,
            tampered,
            version="attacker-replaced-version",
            encryption="AES256",
        ),
    )

    with pytest.raises(AwsAuthorityError, match="signature"):
        _load_activated_bundle_from_aws(
            contract,
            verified,
            authority,
            include_sealed=True,
        )


def test_crash_during_s3_body_write_leaves_no_activation_and_retry_recovers():
    contract, bundle, verified = _fixture()
    authority = _FakeAwsAuthority(crash_once=True)

    with pytest.raises(RuntimeError, match="simulated crash"):
        _publish_release_to_aws(contract, bundle, verified, authority)

    assert authority.activation is None
    assert authority.failed_requests
    assert not any(key.endswith(".partial") for key, _ in authority.objects)

    activated = _publish_release_to_aws(contract, bundle, verified, authority)
    assert authority.activation is not None
    assert activated.activation.version_id == "activation-version"


def test_activation_with_wrong_object_version_or_encryption_is_rejected():
    contract, bundle, verified = _fixture()
    authority = _FakeAwsAuthority()
    _publish_release_to_aws(contract, bundle, verified, authority)
    assert authority.activation is not None

    envelope = json.loads(authority.activation[0])
    activation = envelope["activation"]
    activation["artifacts"]["sealed_gold"]["version_id"] = ""
    activation_bytes = canonical_json_bytes(activation)
    envelope["signature"] = json.loads(
        _signature_document_bytes(
            authority._signature(activation_bytes),
            activation_bytes,
            _SIGNER_ARN,
        )
    )
    changed = canonical_json_bytes(envelope)
    authority.activation = (
        changed,
        authority._ref(
            ACTIVATION_KEY,
            changed,
            version="resigned-fixture-version",
            encryption="AES256",
        ),
    )
    with pytest.raises(EvaluationSealingError, match="artifact"):
        _load_activated_bundle_from_aws(
            contract,
            verified,
            authority,
            include_sealed=True,
        )
