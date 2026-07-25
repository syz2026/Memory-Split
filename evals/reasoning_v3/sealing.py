"""KMS-signed, S3-versioned release sealing for the reasoning-v3 evaluator."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpusgen.parallel.canonical import canonical_json_bytes
from evals.reasoning_v3.aws_authority import (
    ACTIVATION_KEY,
    AWS_REGION,
    EVALUATOR_ROLE_ARN,
    SIGNER_KEY_ALIAS,
    STORAGE_BUCKET,
    STORAGE_PREFIX,
    ActivationPublication,
    AwsAuthorityError,
    S3ObjectVersion,
    VerifiedAwsAuthority,
    _new_fixed_aws_authority,
    _parse_signature_document,
    _signature_document_bytes,
)
from evals.reasoning_v3.contracts import (
    DEFAULT_CONTRACT_PATH,
    ROOT,
    EvaluationContract,
    EvaluationContractError,
    _load_evaluation_contract_from_bytes,
    _regular_file_bytes,
)
from evals.reasoning_v3.generate import (
    EvaluationGenerationError,
    EvaluationItem,
    FrozenEvaluationPaths,
    OracleReplayEvidence,
    ProvenanceCommitment,
    _generate_authenticated_registry,
    _validate_registry,
)


_HEX = frozenset("0123456789abcdef")
_RELEASE_FORMAT = "memorysplit-reasoning-v3-eval-release-v1"
_ACTIVATION_FORMAT = "memorysplit-reasoning-v3-eval-activation-v2"
_SIGNED_ACTIVATION_FORMAT = "memorysplit-reasoning-v3-signed-activation-v1"
_MAX_RELEASE_BYTES = 256 << 20
_MAX_ACTIVATION_BYTES = 1 << 20
_TOP_FIELDS = {
    "authorization",
    "contract_id",
    "contract_sha256",
    "family_order",
    "format",
    "item_count",
    "items",
    "provenance",
    "registry_sha256",
    "release_kind",
    "schema_version",
}
_PROVENANCE_FIELDS = {
    "corpus_receipt_sha256",
    "generator_artifacts_sha256",
    "record_count",
    "record_manifest_sha256",
    "runtime_lock_sha256",
    "source_stage_receipt_sha256",
    "source_tree_commitment_sha256",
}
_ORACLE_FIELDS = {
    "independent_answer_sha256",
    "native_answer_sha256",
    "question_sha256",
    "record_sha256",
    "task_config_sha256",
}
_ENVELOPE_FIELDS = {"activation", "format", "schema_version", "signature"}
_ACTIVATION_FIELDS = {
    "artifacts",
    "authority",
    "contract_id",
    "contract_sha256",
    "format",
    "provenance",
    "provenance_sha256",
    "registry_sha256",
    "schema_version",
    "storage",
    "transaction_id",
}
_AUTHORITY_FIELDS = {
    "contract_authority_record_sha256",
    "contract_authority_record_version_id",
    "contract_authority_signature_version_id",
    "evaluator_role_arn",
    "signer_key_alias",
    "signer_key_arn",
}
_STORAGE_FIELDS = {"activation_key", "bucket", "prefix"}
_MODEL_OBJECT_FIELDS = {
    "bucket",
    "bytes",
    "key",
    "server_side_encryption",
    "sha256",
    "version_id",
}
_SEALED_OBJECT_FIELDS = _MODEL_OBJECT_FIELDS | {"kms_key_arn"}


class EvaluationSealingError(ValueError):
    """A release is unauthenticated, nondeterministic, partial, or malformed."""


@dataclass(frozen=True)
class ReleaseBundle:
    model_visible_bytes: bytes
    sealed_gold_bytes: bytes
    registry_sha256: str


@dataclass(frozen=True)
class ActivatedRelease:
    transaction_id: str
    registry_sha256: str
    model_visible: S3ObjectVersion
    sealed_gold: S3ObjectVersion
    activation: S3ObjectVersion
    activation_sha256: str


@dataclass(frozen=True)
class _ActivationMetadata:
    transaction_id: str
    registry_sha256: str
    provenance: ProvenanceCommitment
    activation_bytes: bytes
    model_visible: S3ObjectVersion
    sealed_gold: S3ObjectVersion
    envelope_bytes: bytes
    activation_object: S3ObjectVersion


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _provenance_dict(value: ProvenanceCommitment) -> dict[str, Any]:
    return {
        "corpus_receipt_sha256": value.corpus_receipt_sha256,
        "generator_artifacts_sha256": value.generator_artifacts_sha256,
        "record_count": value.record_count,
        "record_manifest_sha256": value.record_manifest_sha256,
        "runtime_lock_sha256": value.runtime_lock_sha256,
        "source_stage_receipt_sha256": value.source_stage_receipt_sha256,
        "source_tree_commitment_sha256": value.source_tree_commitment_sha256,
    }


def _parse_provenance(value: object) -> ProvenanceCommitment:
    if not isinstance(value, Mapping) or set(value) != _PROVENANCE_FIELDS:
        raise EvaluationSealingError("release provenance fields differ")
    if (
        isinstance(value["record_count"], bool)
        or not isinstance(value["record_count"], int)
        or value["record_count"] <= 0
        or not all(
            _is_sha256(item)
            for name, item in value.items()
            if name != "record_count"
        )
    ):
        raise EvaluationSealingError("release provenance is invalid")
    return ProvenanceCommitment(**dict(value))


def _public_item(item: EvaluationItem) -> dict[str, Any]:
    return {
        "item_id": item.item_id,
        "max_new_tokens": item.max_new_tokens,
        "prompt": item.prompt,
        "scorer_id": item.scorer_id,
        "source_index": item.source_index,
        "task": item.task,
    }


def _sealed_item(item: EvaluationItem) -> dict[str, Any]:
    evidence = item.oracle_replay
    return {
        "canonical_answer": item.canonical_answer,
        "item_id": item.item_id,
        "oracle_replay": {
            "independent_answer_sha256": evidence.independent_answer_sha256,
            "native_answer_sha256": evidence.native_answer_sha256,
            "question_sha256": evidence.question_sha256,
            "record_sha256": evidence.record_sha256,
            "task_config_sha256": evidence.task_config_sha256,
        },
        "source_index": item.source_index,
        "task": item.task,
    }


def _registry_sha256(
    public: Sequence[Mapping[str, Any]],
    sealed: Sequence[Mapping[str, Any]],
) -> str:
    rows = [
        {"model_visible": left, "sealed_gold": right}
        for left, right in zip(public, sealed, strict=True)
    ]
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def _header(
    contract: EvaluationContract,
    provenance: ProvenanceCommitment,
    *,
    kind: str,
    authorization: str,
    registry: str,
) -> dict[str, Any]:
    return {
        "authorization": authorization,
        "contract_id": contract.contract_id,
        "contract_sha256": contract.sha256,
        "family_order": list(contract.family_names),
        "format": _RELEASE_FORMAT,
        "item_count": contract.total_items,
        "provenance": _provenance_dict(provenance),
        "registry_sha256": registry,
        "release_kind": kind,
        "schema_version": 1,
    }


def _build_release_bundle(
    contract: EvaluationContract,
    items: Sequence[EvaluationItem],
    provenance: ProvenanceCommitment,
) -> ReleaseBundle:
    """Private fixture primitive; production regenerates authenticated inputs."""

    try:
        _validate_registry(contract, items)
    except EvaluationGenerationError as error:
        raise EvaluationSealingError("cannot seal an invalid registry") from error
    if (
        provenance.corpus_receipt_sha256 != contract.corpus_receipt_sha256
        or provenance.source_stage_receipt_sha256
        != contract.source_stage_receipt_sha256
        or provenance.record_manifest_sha256 != contract.record_manifest_sha256
    ):
        raise EvaluationSealingError("release provenance differs from contract")
    _parse_provenance(_provenance_dict(provenance))
    public_items = [_public_item(item) for item in items]
    sealed_items = [_sealed_item(item) for item in items]
    registry = _registry_sha256(public_items, sealed_items)
    public = {
        **_header(
            contract,
            provenance,
            kind="model_visible",
            authorization="trainer_and_evaluator",
            registry=registry,
        ),
        "items": public_items,
    }
    sealed = {
        **_header(
            contract,
            provenance,
            kind="sealed_gold",
            authorization="evaluator_only",
            registry=registry,
        ),
        "items": sealed_items,
    }
    return ReleaseBundle(
        canonical_json_bytes(public),
        canonical_json_bytes(sealed),
        registry,
    )


def _double_generate_release(factory: Callable[[], ReleaseBundle]) -> ReleaseBundle:
    first = factory()
    second = factory()
    if first != second:
        raise EvaluationSealingError(
            "independent generation produced different release bundles"
        )
    return first


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationSealingError(f"JSON repeats key: {key}")
        result[key] = value
    return result


def _parse_canonical_bytes(data: bytes, label: str) -> Mapping[str, Any]:
    if not isinstance(data, bytes) or not data:
        raise EvaluationSealingError(f"{label} is missing")
    try:
        value = json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                EvaluationSealingError(
                    f"{label} has non-finite JSON: {item}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationSealingError(
            f"{label} is not valid UTF-8 JSON"
        ) from error
    if not isinstance(value, Mapping) or data != canonical_json_bytes(value):
        raise EvaluationSealingError(f"{label} is not canonical JSON")
    return value


def _read_file_bytes(path: Path, label: str, maximum: int) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise EvaluationSealingError(
            f"{label} is missing or unsafe"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise EvaluationSealingError(f"{label} is unsafe or oversized")
        chunks: list[bytes] = []
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while chunk := handle.read(8 << 20):
                chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
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
    ):
        raise EvaluationSealingError(f"{label} changed while reading")
    return b"".join(chunks)


def _read_canonical_file(
    path: Path | str,
    label: str,
    maximum: int,
) -> Mapping[str, Any]:
    return _parse_canonical_bytes(
        _read_file_bytes(Path(path), label, maximum),
        label,
    )


def _validate_header(
    release: Mapping[str, Any],
    contract: EvaluationContract,
    *,
    kind: str,
    authorization: str,
    provenance: ProvenanceCommitment | None = None,
) -> tuple[list[Any], ProvenanceCommitment]:
    if set(release) != _TOP_FIELDS:
        raise EvaluationSealingError(f"{kind} release fields differ")
    parsed = _parse_provenance(release["provenance"])
    if (
        release["schema_version"] != 1
        or isinstance(release["schema_version"], bool)
        or release["format"] != _RELEASE_FORMAT
        or release["release_kind"] != kind
        or release["authorization"] != authorization
        or release["contract_id"] != contract.contract_id
        or release["contract_sha256"] != contract.sha256
        or release["family_order"] != list(contract.family_names)
        or isinstance(release["item_count"], bool)
        or release["item_count"] != contract.total_items
        or not _is_sha256(release["registry_sha256"])
        or (provenance is not None and parsed != provenance)
    ):
        raise EvaluationSealingError(f"{kind} release identity differs")
    items = release["items"]
    if not isinstance(items, list) or len(items) != contract.total_items:
        raise EvaluationSealingError(f"{kind} release item count differs")
    return items, parsed


def _validate_public_items(
    items: Sequence[Any],
    contract: EvaluationContract,
) -> None:
    fields = set(contract.model_visible_fields)
    offset = 0
    seen: set[str] = set()
    for family in contract.families:
        previous = family.index_start - 1
        family_items = items[
            offset : offset + contract.accepted_items_per_family
        ]
        for raw in family_items:
            if not isinstance(raw, Mapping) or set(raw) != fields:
                raise EvaluationSealingError(
                    "model_visible item fields differ or contain gold"
                )
            index = raw["source_index"]
            if (
                raw["task"] != family.task
                or isinstance(index, bool)
                or not isinstance(index, int)
                or not family.index_start <= index < family.index_stop
                or index <= previous
                or raw["item_id"]
                != f"memorysplit-reasoning-v3-eval-v1/{family.task}/{index}"
                or raw["item_id"] in seen
                or raw["max_new_tokens"] != family.max_new_tokens
                or isinstance(raw["max_new_tokens"], bool)
                or raw["scorer_id"] != contract.scorer_id
                or not isinstance(raw["prompt"], str)
                or not raw["prompt"].startswith(
                    f"Reasoning task={family.task}\nQuestion: "
                )
                or not raw["prompt"].endswith("\nAnswer:")
            ):
                raise EvaluationSealingError(
                    f"model_visible item identity/order differs for {family.task}"
                )
            seen.add(str(raw["item_id"]))
            previous = index
        offset += contract.accepted_items_per_family


def _validate_sealed_items(
    items: Sequence[Any],
    contract: EvaluationContract,
) -> None:
    fields = set(contract.sealed_gold_fields)
    offset = 0
    seen: set[str] = set()
    for family in contract.families:
        previous = family.index_start - 1
        family_items = items[
            offset : offset + contract.accepted_items_per_family
        ]
        for raw in family_items:
            if not isinstance(raw, Mapping) or set(raw) != fields:
                raise EvaluationSealingError("sealed_gold item fields differ")
            index = raw["source_index"]
            evidence = raw["oracle_replay"]
            answer = raw["canonical_answer"]
            if (
                raw["task"] != family.task
                or isinstance(index, bool)
                or not isinstance(index, int)
                or not family.index_start <= index < family.index_stop
                or index <= previous
                or raw["item_id"]
                != f"memorysplit-reasoning-v3-eval-v1/{family.task}/{index}"
                or raw["item_id"] in seen
                or not isinstance(answer, str)
                or not answer
                or answer != answer.strip()
                or not isinstance(evidence, Mapping)
                or set(evidence) != _ORACLE_FIELDS
                or not all(_is_sha256(value) for value in evidence.values())
            ):
                raise EvaluationSealingError(
                    f"sealed_gold item identity/evidence differs for {family.task}"
                )
            seen.add(str(raw["item_id"]))
            previous = index
        offset += contract.accepted_items_per_family


def _parse_model_visible_release(
    path: Path | str,
    contract: EvaluationContract,
) -> Mapping[str, Any]:
    release = _read_canonical_file(
        path,
        "model-visible release",
        _MAX_RELEASE_BYTES,
    )
    items, _ = _validate_header(
        release,
        contract,
        kind="model_visible",
        authorization="trainer_and_evaluator",
    )
    _validate_public_items(items, contract)
    return release


def _parse_sealed_gold_release(
    path: Path | str,
    contract: EvaluationContract,
) -> Mapping[str, Any]:
    release = _read_canonical_file(
        path,
        "sealed-gold release",
        _MAX_RELEASE_BYTES,
    )
    items, _ = _validate_header(
        release,
        contract,
        kind="sealed_gold",
        authorization="evaluator_only",
    )
    _validate_sealed_items(items, contract)
    return release


def _validate_release_bundle(
    bundle: ReleaseBundle,
    contract: EvaluationContract,
    provenance: ProvenanceCommitment,
) -> str:
    public = _parse_canonical_bytes(
        bundle.model_visible_bytes,
        "model-visible release",
    )
    sealed = _parse_canonical_bytes(
        bundle.sealed_gold_bytes,
        "sealed-gold release",
    )
    public_items, public_provenance = _validate_header(
        public,
        contract,
        kind="model_visible",
        authorization="trainer_and_evaluator",
        provenance=provenance,
    )
    sealed_items, sealed_provenance = _validate_header(
        sealed,
        contract,
        kind="sealed_gold",
        authorization="evaluator_only",
        provenance=provenance,
    )
    _validate_public_items(public_items, contract)
    _validate_sealed_items(sealed_items, contract)
    if (
        public_provenance != sealed_provenance
        or public["registry_sha256"] != sealed["registry_sha256"]
        or bundle.registry_sha256 != public["registry_sha256"]
    ):
        raise EvaluationSealingError("release registry commitments differ")
    reconstructed: list[EvaluationItem] = []
    for visible, gold in zip(public_items, sealed_items, strict=True):
        if (
            visible["item_id"] != gold["item_id"]
            or visible["task"] != gold["task"]
            or visible["source_index"] != gold["source_index"]
        ):
            raise EvaluationSealingError(
                "public and sealed item identities differ"
            )
        question = visible["prompt"].split(
            "\nQuestion: ",
            1,
        )[1].removesuffix("\nAnswer:")
        evidence = gold["oracle_replay"]
        if (
            not question
            or hashlib.sha256(question.encode()).hexdigest()
            != evidence["question_sha256"]
        ):
            raise EvaluationSealingError(
                "public prompt differs from sealed replay"
            )
        reconstructed.append(
            EvaluationItem(
                item_id=visible["item_id"],
                task=visible["task"],
                source_index=visible["source_index"],
                prompt=visible["prompt"],
                max_new_tokens=visible["max_new_tokens"],
                scorer_id=visible["scorer_id"],
                canonical_answer=gold["canonical_answer"],
                oracle_replay=OracleReplayEvidence(**dict(evidence)),
            )
        )
    try:
        _validate_registry(contract, reconstructed)
    except EvaluationGenerationError as error:
        raise EvaluationSealingError(
            "release oracle evidence differs"
        ) from error
    actual = _registry_sha256(public_items, sealed_items)
    if actual != bundle.registry_sha256:
        raise EvaluationSealingError("release registry commitment differs")
    return actual


def _new_evaluator_aws_authority():
    """Construct the fixed production adapter; no caller object is accepted."""

    return _new_fixed_aws_authority()


def _load_authorized_contract(
    repository_root: Path | str,
    authority,
) -> tuple[EvaluationContract, VerifiedAwsAuthority]:
    try:
        root = Path(repository_root).resolve(strict=True)
        contract_path = root / DEFAULT_CONTRACT_PATH.relative_to(ROOT)
        contract_bytes = _regular_file_bytes(
            contract_path,
            "evaluation contract",
            maximum=1 << 20,
        )
        verified = authority.verify_contract_authority(
            root,
            contract_bytes,
        )
        if (
            not isinstance(verified, VerifiedAwsAuthority)
            or verified.contract_sha256
            != hashlib.sha256(contract_bytes).hexdigest()
        ):
            raise AwsAuthorityError(
                "signed contract authority did not bind exact contract bytes"
            )
        contract = _load_evaluation_contract_from_bytes(
            contract_bytes,
            path=contract_path,
            repository_root=root,
        )
    except (EvaluationContractError, OSError) as error:
        raise EvaluationSealingError(
            "signed production contract loading failed"
        ) from error
    if (
        contract.sha256 != verified.contract_sha256
        or contract.aws_region != AWS_REGION
        or contract.evaluator_role_arn != EVALUATOR_ROLE_ARN
        or contract.signer_key_alias != SIGNER_KEY_ALIAS
        or contract.storage_bucket != STORAGE_BUCKET
        or contract.storage_prefix != STORAGE_PREFIX
        or contract.activation_key != ACTIVATION_KEY
    ):
        raise EvaluationSealingError(
            "signed production contract authority differs"
        )
    return contract, verified


def _double_generate_authenticated(
    paths: FrozenEvaluationPaths,
    contract: EvaluationContract,
) -> tuple[ProvenanceCommitment, ReleaseBundle]:
    generated = []

    def factory() -> ReleaseBundle:
        registry = _generate_authenticated_registry(paths, contract)
        generated.append(registry)
        return _build_release_bundle(
            registry.contract,
            registry.items,
            registry.provenance,
        )

    bundle = _double_generate_release(factory)
    first, second = generated
    if (
        first.contract.sha256 != contract.sha256
        or second.contract.sha256 != contract.sha256
        or first.provenance != second.provenance
    ):
        raise EvaluationSealingError(
            "independent generation provenance differs"
        )
    return first.provenance, bundle


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


def _transaction_id(
    contract: EvaluationContract,
    bundle: ReleaseBundle,
    provenance: ProvenanceCommitment,
    verified: VerifiedAwsAuthority,
) -> str:
    core = {
        "contract_authority_record_sha256": verified.record_sha256,
        "contract_sha256": contract.sha256,
        "model_visible_sha256": hashlib.sha256(
            bundle.model_visible_bytes
        ).hexdigest(),
        "provenance_sha256": hashlib.sha256(
            canonical_json_bytes(_provenance_dict(provenance))
        ).hexdigest(),
        "registry_sha256": bundle.registry_sha256,
        "sealed_gold_sha256": hashlib.sha256(
            bundle.sealed_gold_bytes
        ).hexdigest(),
        "storage_bucket": STORAGE_BUCKET,
        "storage_prefix": STORAGE_PREFIX,
    }
    return hashlib.sha256(canonical_json_bytes(core)).hexdigest()


def _publish_release_to_aws(
    contract: EvaluationContract,
    bundle: ReleaseBundle,
    verified: VerifiedAwsAuthority,
    authority,
) -> ActivatedRelease:
    """Private orchestration seam used by production and in-memory tests."""

    public = _parse_canonical_bytes(
        bundle.model_visible_bytes,
        "model-visible release",
    )
    provenance = _parse_provenance(public["provenance"])
    _validate_release_bundle(bundle, contract, provenance)
    if verified.contract_sha256 != contract.sha256:
        raise EvaluationSealingError(
            "AWS authority contract digest differs"
        )
    transaction = _transaction_id(
        contract,
        bundle,
        provenance,
        verified,
    )
    model_ref = authority.put_model_visible(
        transaction,
        bundle.model_visible_bytes,
    )
    sealed_ref = authority.put_sealed_gold(
        transaction,
        bundle.sealed_gold_bytes,
        verified.sealed_gold_kms_key_arn,
    )
    provenance_value = _provenance_dict(provenance)
    activation = {
        "artifacts": {
            "model_visible": _object_dict(model_ref),
            "sealed_gold": _object_dict(sealed_ref),
        },
        "authority": {
            "contract_authority_record_sha256": verified.record_sha256,
            "contract_authority_record_version_id": verified.record_version_id,
            "contract_authority_signature_version_id": (
                verified.signature_version_id
            ),
            "evaluator_role_arn": EVALUATOR_ROLE_ARN,
            "signer_key_alias": SIGNER_KEY_ALIAS,
            "signer_key_arn": verified.signer_key_arn,
        },
        "contract_id": contract.contract_id,
        "contract_sha256": contract.sha256,
        "format": _ACTIVATION_FORMAT,
        "provenance": provenance_value,
        "provenance_sha256": hashlib.sha256(
            canonical_json_bytes(provenance_value)
        ).hexdigest(),
        "registry_sha256": bundle.registry_sha256,
        "schema_version": 2,
        "storage": {
            "activation_key": ACTIVATION_KEY,
            "bucket": STORAGE_BUCKET,
            "prefix": STORAGE_PREFIX,
        },
        "transaction_id": transaction,
    }
    activation_bytes = canonical_json_bytes(activation)
    signature = authority.sign(activation_bytes, verified)
    signature_document = _parse_canonical_bytes(
        _signature_document_bytes(
            signature,
            activation_bytes,
            verified.signer_key_arn,
        ),
        "activation signature",
    )
    envelope_bytes = canonical_json_bytes(
        {
            "activation": activation,
            "format": _SIGNED_ACTIVATION_FORMAT,
            "schema_version": 1,
            "signature": signature_document,
        }
    )
    publication = authority.put_activation(envelope_bytes)
    if not isinstance(publication, ActivationPublication):
        raise EvaluationSealingError(
            "AWS activation publication result is invalid"
        )
    published = _parse_signed_activation_envelope(
        publication.envelope_bytes,
        publication.object_ref,
        contract,
        verified,
        authority,
    )
    if (
        published.activation_bytes != activation_bytes
        or published.transaction_id != transaction
        or published.registry_sha256 != bundle.registry_sha256
        or published.provenance != provenance
        or published.model_visible != model_ref
        or published.sealed_gold != sealed_ref
    ):
        raise EvaluationSealingError(
            "existing signed activation payload differs from intended payload"
        )
    return ActivatedRelease(
        transaction_id=transaction,
        registry_sha256=bundle.registry_sha256,
        model_visible=model_ref,
        sealed_gold=sealed_ref,
        activation=published.activation_object,
        activation_sha256=hashlib.sha256(
            published.envelope_bytes
        ).hexdigest(),
    )


def _parse_object_ref(
    raw: object,
    *,
    label: str,
    transaction_id: str,
    verified: VerifiedAwsAuthority,
) -> S3ObjectVersion:
    sealed = label == "sealed_gold"
    fields = _SEALED_OBJECT_FIELDS if sealed else _MODEL_OBJECT_FIELDS
    if not isinstance(raw, Mapping) or set(raw) != fields:
        raise EvaluationSealingError(
            f"activation {label} artifact fields differ"
        )
    expected_key = (
        f"{STORAGE_PREFIX}/evaluator-only/quarantine/"
        f"{transaction_id}/gold.json"
        if sealed
        else (
            f"{STORAGE_PREFIX}/model-visible/quarantine/"
            f"{transaction_id}/release.json"
        )
    )
    expected_encryption = "aws:kms" if sealed else "AES256"
    kms_key_arn = raw.get("kms_key_arn")
    if (
        raw["bucket"] != STORAGE_BUCKET
        or raw["key"] != expected_key
        or not isinstance(raw["version_id"], str)
        or not raw["version_id"]
        or isinstance(raw["bytes"], bool)
        or not isinstance(raw["bytes"], int)
        or raw["bytes"] <= 0
        or not _is_sha256(raw["sha256"])
        or raw["server_side_encryption"] != expected_encryption
        or (
            sealed
            and kms_key_arn != verified.sealed_gold_kms_key_arn
        )
    ):
        raise EvaluationSealingError(
            f"activation {label} artifact identity differs"
        )
    return S3ObjectVersion(
        bucket=raw["bucket"],
        key=raw["key"],
        version_id=raw["version_id"],
        bytes=raw["bytes"],
        sha256=raw["sha256"],
        server_side_encryption=raw["server_side_encryption"],
        kms_key_arn=kms_key_arn,
    )


def _parse_signed_activation_envelope(
    envelope_bytes: bytes,
    activation_object: S3ObjectVersion,
    contract: EvaluationContract,
    verified: VerifiedAwsAuthority,
    authority,
) -> _ActivationMetadata:
    if (
        not isinstance(activation_object, S3ObjectVersion)
        or activation_object.bucket != STORAGE_BUCKET
        or activation_object.key != ACTIVATION_KEY
        or not activation_object.version_id
        or activation_object.server_side_encryption != "AES256"
        or activation_object.kms_key_arn is not None
        or activation_object.bytes != len(envelope_bytes)
        or activation_object.sha256
        != hashlib.sha256(envelope_bytes).hexdigest()
    ):
        raise EvaluationSealingError(
            "signed activation S3 object identity differs"
        )
    envelope = _parse_canonical_bytes(
        envelope_bytes,
        "signed activation",
    )
    if (
        set(envelope) != _ENVELOPE_FIELDS
        or envelope["format"] != _SIGNED_ACTIVATION_FORMAT
        or envelope["schema_version"] != 1
        or isinstance(envelope["schema_version"], bool)
        or not isinstance(envelope["activation"], Mapping)
        or not isinstance(envelope["signature"], Mapping)
    ):
        raise EvaluationSealingError(
            "signed activation envelope fields differ"
        )
    activation = envelope["activation"]
    activation_bytes = canonical_json_bytes(activation)
    signature_bytes = canonical_json_bytes(envelope["signature"])
    signature = _parse_signature_document(
        signature_bytes,
        activation_bytes,
        verified.signer_key_arn,
    )
    authority.verify(activation_bytes, signature, verified)
    if set(activation) != _ACTIVATION_FIELDS:
        raise EvaluationSealingError("evaluation activation fields differ")
    provenance = _parse_provenance(activation["provenance"])
    authority_value = activation["authority"]
    storage = activation["storage"]
    transaction = activation["transaction_id"]
    if (
        activation["schema_version"] != 2
        or isinstance(activation["schema_version"], bool)
        or activation["format"] != _ACTIVATION_FORMAT
        or activation["contract_id"] != contract.contract_id
        or activation["contract_sha256"] != contract.sha256
        or not _is_sha256(transaction)
        or not _is_sha256(activation["registry_sha256"])
        or activation["provenance_sha256"]
        != hashlib.sha256(
            canonical_json_bytes(_provenance_dict(provenance))
        ).hexdigest()
        or not isinstance(authority_value, Mapping)
        or set(authority_value) != _AUTHORITY_FIELDS
        or authority_value
        != {
            "contract_authority_record_sha256": verified.record_sha256,
            "contract_authority_record_version_id": (
                verified.record_version_id
            ),
            "contract_authority_signature_version_id": (
                verified.signature_version_id
            ),
            "evaluator_role_arn": EVALUATOR_ROLE_ARN,
            "signer_key_alias": SIGNER_KEY_ALIAS,
            "signer_key_arn": verified.signer_key_arn,
        }
        or not isinstance(storage, Mapping)
        or set(storage) != _STORAGE_FIELDS
        or storage
        != {
            "activation_key": ACTIVATION_KEY,
            "bucket": STORAGE_BUCKET,
            "prefix": STORAGE_PREFIX,
        }
        or not isinstance(activation["artifacts"], Mapping)
        or set(activation["artifacts"])
        != {"model_visible", "sealed_gold"}
    ):
        raise EvaluationSealingError(
            "evaluation activation identity differs"
        )
    model_ref = _parse_object_ref(
        activation["artifacts"]["model_visible"],
        label="model_visible",
        transaction_id=transaction,
        verified=verified,
    )
    sealed_ref = _parse_object_ref(
        activation["artifacts"]["sealed_gold"],
        label="sealed_gold",
        transaction_id=transaction,
        verified=verified,
    )
    core = {
        "contract_authority_record_sha256": verified.record_sha256,
        "contract_sha256": contract.sha256,
        "model_visible_sha256": model_ref.sha256,
        "provenance_sha256": activation["provenance_sha256"],
        "registry_sha256": activation["registry_sha256"],
        "sealed_gold_sha256": sealed_ref.sha256,
        "storage_bucket": STORAGE_BUCKET,
        "storage_prefix": STORAGE_PREFIX,
    }
    if hashlib.sha256(canonical_json_bytes(core)).hexdigest() != transaction:
        raise EvaluationSealingError(
            "evaluation activation transaction differs"
        )
    return _ActivationMetadata(
        transaction_id=transaction,
        registry_sha256=activation["registry_sha256"],
        provenance=provenance,
        activation_bytes=activation_bytes,
        model_visible=model_ref,
        sealed_gold=sealed_ref,
        envelope_bytes=envelope_bytes,
        activation_object=activation_object,
    )


def _read_signed_activation(
    contract: EvaluationContract,
    verified: VerifiedAwsAuthority,
    authority,
) -> _ActivationMetadata:
    envelope_bytes, activation_object = authority.read_activation()
    return _parse_signed_activation_envelope(
        envelope_bytes,
        activation_object,
        contract,
        verified,
        authority,
    )


def _validate_object_payload(
    payload: bytes,
    ref: S3ObjectVersion,
    label: str,
) -> None:
    if (
        not isinstance(payload, bytes)
        or len(payload) != ref.bytes
        or hashlib.sha256(payload).hexdigest() != ref.sha256
    ):
        raise EvaluationSealingError(
            f"{label} differs from signed activation"
        )


def _load_activated_bundle_from_aws(
    contract: EvaluationContract,
    verified: VerifiedAwsAuthority,
    authority,
    *,
    include_sealed: bool,
) -> ReleaseBundle:
    if include_sealed is not True:
        raise EvaluationSealingError(
            "private bundle loading requires evaluator-only sealed access"
        )
    metadata = _read_signed_activation(contract, verified, authority)
    public = authority.read_model_visible(metadata.model_visible)
    sealed = authority.read_sealed_gold(metadata.sealed_gold)
    _validate_object_payload(
        public,
        metadata.model_visible,
        "model-visible release",
    )
    _validate_object_payload(
        sealed,
        metadata.sealed_gold,
        "sealed-gold release",
    )
    return ReleaseBundle(public, sealed, metadata.registry_sha256)


def materialize_frozen_evaluation_release(
    paths: FrozenEvaluationPaths,
) -> ActivatedRelease:
    """Generate and publish only through the fixed evaluator AWS authority."""

    authority = _new_evaluator_aws_authority()
    authority.require_evaluator_role()
    contract, verified = _load_authorized_contract(
        paths.repository_root,
        authority,
    )
    _, bundle = _double_generate_authenticated(paths, contract)
    return _publish_release_to_aws(
        contract,
        bundle,
        verified,
        authority,
    )


def validate_frozen_evaluation_release(
    paths: FrozenEvaluationPaths,
) -> str:
    """Regenerate and validate the exact signed, versioned AWS release."""

    authority = _new_evaluator_aws_authority()
    authority.require_evaluator_role()
    contract, verified = _load_authorized_contract(
        paths.repository_root,
        authority,
    )
    provenance, expected = _double_generate_authenticated(paths, contract)
    actual = _load_activated_bundle_from_aws(
        contract,
        verified,
        authority,
        include_sealed=True,
    )
    _validate_release_bundle(actual, contract, provenance)
    if actual != expected:
        raise EvaluationSealingError(
            "activated release differs from authenticated regeneration"
        )
    return actual.registry_sha256


def load_model_visible_release() -> Mapping[str, Any]:
    """Load only the signed model-visible object from the fixed AWS location."""

    authority = _new_evaluator_aws_authority()
    contract, verified = _load_authorized_contract(ROOT, authority)
    metadata = _read_signed_activation(contract, verified, authority)
    payload = authority.read_model_visible(metadata.model_visible)
    _validate_object_payload(
        payload,
        metadata.model_visible,
        "model-visible release",
    )
    release = _parse_canonical_bytes(payload, "model-visible release")
    items, provenance = _validate_header(
        release,
        contract,
        kind="model_visible",
        authorization="trainer_and_evaluator",
    )
    _validate_public_items(items, contract)
    if (
        release["registry_sha256"] != metadata.registry_sha256
        or provenance != metadata.provenance
    ):
        raise EvaluationSealingError(
            "model-visible release differs from signed activation"
        )
    return release


__all__ = [
    "ActivatedRelease",
    "EvaluationSealingError",
    "load_model_visible_release",
    "materialize_frozen_evaluation_release",
    "validate_frozen_evaluation_release",
]
