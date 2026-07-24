from __future__ import annotations

import json
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest

from cluster.aws.corpus_builder.contracts import (
    CORPUS_KMS_KEY_ARN,
    LAUNCH_INTENT_FORMAT,
    MAX_COMPUTE_USD,
    MAX_HOURLY_USD,
    PHASE_RECEIPT_FORMAT,
    CorpusBuilderProfile,
    LaunchIntent,
    PhaseReceipt,
    S3ObjectVersion,
    corpus_builder_profile_to_bytes,
    launch_intent_from_bytes,
    launch_intent_to_bytes,
    load_corpus_builder_profile,
    phase_receipt_from_bytes,
    phase_receipt_to_bytes,
    s3_object_version_from_dict,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "cluster" / "profiles" / "aws-i4i.16xlarge-corpus-v1.json"

_BUCKET = "memorysplit-corpus-056956104102-us-east-1"
_KMS_ARN = CORPUS_KMS_KEY_ARN
_SHA256_A = "a" * 64
_SHA256_B = "b" * 64
_SHA256_C = "c" * 64
_SHA256_D = "d" * 64


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _valid_s3_object(
    *,
    build_id: str = _SHA256_B,
    object_name: str = "receipt.json",
) -> dict[str, object]:
    uri_suffix = f"v2/builds/{build_id}/{object_name}"
    return {
        "bytes": 1024,
        "etag": "0123456789abcdef0123456789abcdef",
        "kms_key_arn": _KMS_ARN,
        "sha256": _SHA256_A,
        "sse_algorithm": "aws:kms",
        "uri": f"s3://{_BUCKET}/{uri_suffix}",
        "version_id": "version-id-001",
    }


def _valid_phase_receipt_dict(*, objects: list[dict[str, object]] | None = None) -> dict[str, object]:
    if objects is None:
        objects = [_valid_s3_object()]
    return {
        "build_id": _SHA256_B,
        "format": PHASE_RECEIPT_FORMAT,
        "objects": objects,
        "package_sha256": _SHA256_C,
        "phase": "source-verify",
        "schema_version": 1,
        "source_lock_sha256": _SHA256_D,
    }


def _valid_launch_intent_dict() -> dict[str, object]:
    return {
        "ami_id": "ami-0123456789abcdef0",
        "ami_owner_id": "056956104102",
        "format": LAUNCH_INTENT_FORMAT,
        "hourly_usd": "5.491",
        "instance_profile_arn": (
            "arn:aws:iam::056956104102:instance-profile/memorysplit-corpus-builder"
        ),
        "launch_template_id": "lt-0123456789abcdef0",
        "launch_template_version": "3",
        "max_compute_usd": "131.78",
        "not_after": "2026-07-25T12:00:00Z",
        "package": _valid_s3_object(build_id=_SHA256_B, object_name="package.zip"),
        "profile_sha256": _SHA256_A,
        "schema_version": 1,
        "security_group_id": "sg-0123456789abcdef0",
        "source_manifest": _valid_s3_object(
            build_id=_SHA256_B, object_name="source-manifest.json"
        ),
        "subnet_id": "subnet-0123456789abcdef0",
    }


def _mutated_phase_bytes(mutator) -> bytes:
    value = deepcopy(_valid_phase_receipt_dict())
    mutator(value)
    return _canonical_bytes(value)


def malformed_contract_payloads() -> list[bytes]:
    payloads: list[bytes] = []

    def add(mutator) -> None:
        payloads.append(_mutated_phase_bytes(mutator))

    add(lambda value: value.pop("phase"))
    add(lambda value: value.update(extra="field"))
    add(lambda value: value.update(format="other-format"))
    add(lambda value: value.update(schema_version=2))
    add(lambda value: value.update(package_sha256="A" * 64))
    add(lambda value: value["objects"].append(_valid_s3_object()))
    add(
        lambda value: value["objects"].append(
            _valid_s3_object(object_name="other.json")
        )
    )
    add(
        lambda value: value["objects"][0].update(
            uri="s3://user:pass@memorysplit-corpus-056956104102-us-east-1/key"
        )
    )
    add(lambda value: value["objects"][0].update(sse_algorithm="AES256"))
    add(lambda value: value.update(max_compute_usd="131.78"))

    duplicate = _canonical_bytes(_valid_phase_receipt_dict()).decode("utf-8")
    duplicate = duplicate.replace(
        '"schema_version":1',
        '"schema_version":1,"schema_version":1',
        1,
    )
    payloads.append(duplicate.encode("utf-8"))

    noncanonical = json.dumps(
        _valid_phase_receipt_dict(),
        ensure_ascii=False,
        sort_keys=False,
    ).encode("utf-8") + b"\n"
    payloads.append(noncanonical)

    no_newline = _canonical_bytes(_valid_phase_receipt_dict())[:-1]
    payloads.append(no_newline)

    return payloads


def test_builder_profile_is_exact_and_cost_bounded():
    profile = load_corpus_builder_profile(PROFILE_PATH)

    assert isinstance(profile, CorpusBuilderProfile)
    assert profile.profile_id == "aws-i4i.16xlarge-corpus-v1"
    assert profile.instance_type == "i4i.16xlarge"
    assert profile.region == "us-east-1"
    assert profile.vcpus == 64
    assert profile.memory_mib == 524_288
    assert profile.nvme_devices == 4
    assert profile.nvme_total_gib == 15_000
    assert profile.root_volume_gib == 200
    assert profile.max_runtime_seconds == 86_400
    assert profile.watchdog_shutdown_seconds == 84_600
    assert profile.max_hourly_usd == Decimal("5.491")
    assert profile.max_compute_usd == Decimal("131.78")
    assert profile.bucket_name == _BUCKET
    assert profile.key_prefix == "v2/builds"


def test_profile_round_trip_is_canonical():
    profile = load_corpus_builder_profile(PROFILE_PATH)
    payload = corpus_builder_profile_to_bytes(profile)

    assert payload == PROFILE_PATH.read_bytes()
    assert payload.endswith(b"\n")
    assert load_corpus_builder_profile(PROFILE_PATH) == profile


def test_contract_parsers_reject_unknown_missing_duplicate_and_noncanonical_fields():
    for payload in malformed_contract_payloads():
        with pytest.raises(ValueError):
            phase_receipt_from_bytes(payload)


def test_phase_receipt_round_trip_is_canonical():
    receipt = phase_receipt_from_bytes(_canonical_bytes(_valid_phase_receipt_dict()))
    payload = phase_receipt_to_bytes(receipt)

    assert payload == _canonical_bytes(_valid_phase_receipt_dict())
    assert receipt == PhaseReceipt(
        format=PHASE_RECEIPT_FORMAT,
        schema_version=1,
        build_id=_SHA256_B,
        phase="source-verify",
        package_sha256=_SHA256_C,
        source_lock_sha256=_SHA256_D,
        objects=(
            S3ObjectVersion(
                uri=f"s3://{_BUCKET}/v2/builds/{_SHA256_B}/receipt.json",
                version_id="version-id-001",
                bytes=1024,
                sha256=_SHA256_A,
                etag="0123456789abcdef0123456789abcdef",
                sse_algorithm="aws:kms",
                kms_key_arn=_KMS_ARN,
            ),
        ),
    )


def test_launch_intent_round_trip_is_canonical():
    intent = launch_intent_from_bytes(_canonical_bytes(_valid_launch_intent_dict()))
    payload = launch_intent_to_bytes(intent)

    assert payload == _canonical_bytes(_valid_launch_intent_dict())
    assert intent.hourly_usd == Decimal("5.491")
    assert intent.max_compute_usd == Decimal("131.78")
    assert intent.not_after == "2026-07-25T12:00:00Z"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.pop("not_after"),
        lambda value: value.update(not_after="2026-07-25T12:00:00+00:00"),
        lambda value: value.update(hourly_usd=5.491),
        lambda value: value.update(max_compute_usd="131.780"),
        lambda value: value.update(ami_id="ami-latest"),
        lambda value: value.update(launch_template_version="$Latest"),
        lambda value: value.update(extra="field"),
        lambda value: value["package"].update(uri="https://example.com/package.zip"),
    ],
)
def test_launch_intent_rejects_contract_drift(mutator):
    value = deepcopy(_valid_launch_intent_dict())
    mutator(value)

    with pytest.raises(ValueError):
        launch_intent_from_bytes(_canonical_bytes(value))


def test_s3_object_version_requires_safe_uri_and_lowercase_sha256():
    with pytest.raises(ValueError):
        s3_object_version_from_dict(
            {
                **_valid_s3_object(),
                "sha256": "A" * 64,
            }
        )

    with pytest.raises(ValueError):
        s3_object_version_from_dict(
            {
                **_valid_s3_object(),
                "uri": "s3://memorysplit-corpus-056956104102-us-east-1",
            }
        )


@pytest.mark.parametrize(
    ("hourly_usd", "max_compute_usd"),
    [
        ("5.492", "131.78"),
        ("5.491", "131.79"),
        ("6", "100"),
    ],
)
def test_launch_intent_rejects_prices_above_profile_ceiling(hourly_usd, max_compute_usd):
    value = _valid_launch_intent_dict()
    value["hourly_usd"] = hourly_usd
    value["max_compute_usd"] = max_compute_usd

    with pytest.raises(ValueError):
        launch_intent_from_bytes(_canonical_bytes(value))


def test_launch_intent_accepts_prices_at_profile_ceiling():
    value = _valid_launch_intent_dict()
    value["hourly_usd"] = str(MAX_HOURLY_USD)
    value["max_compute_usd"] = str(MAX_COMPUTE_USD)

    intent = launch_intent_from_bytes(_canonical_bytes(value))
    assert intent.hourly_usd == MAX_HOURLY_USD
    assert intent.max_compute_usd == MAX_COMPUTE_USD


@pytest.mark.parametrize(
    "mutator",
    [
        lambda obj: obj.update(
            uri=f"s3://other-bucket/v2/builds/{_SHA256_B}/receipt.json"
        ),
        lambda obj: obj.update(
            uri=f"s3://{_BUCKET}/v2/builds/wrong-build-id/receipt.json"
        ),
        lambda obj: obj.update(uri=f"s3://{_BUCKET}/other-prefix/{_SHA256_B}/x.json"),
        lambda obj: obj.update(sse_algorithm="AES256", kms_key_arn=""),
        lambda obj: obj.update(
            kms_key_arn=(
                "arn:aws:kms:us-east-1:056956104102:key/"
                "fedcba98-7654-3210-fedc-ba9876543210"
            )
        ),
    ],
)
def test_s3_object_rejects_non_corpus_bucket_prefix_or_encryption(mutator):
    value = _valid_s3_object()
    mutator(value)

    with pytest.raises(ValueError):
        s3_object_version_from_dict(value)


def test_phase_receipt_requires_objects_under_receipt_build_id():
    value = _valid_phase_receipt_dict(
        objects=[_valid_s3_object(build_id=_SHA256_C, object_name="receipt.json")]
    )

    with pytest.raises(ValueError):
        phase_receipt_from_bytes(_canonical_bytes(value))


def test_launch_intent_requires_package_and_manifest_under_same_build_id():
    value = _valid_launch_intent_dict()
    value["source_manifest"] = _valid_s3_object(
        build_id=_SHA256_C, object_name="source-manifest.json"
    )

    with pytest.raises(ValueError):
        launch_intent_from_bytes(_canonical_bytes(value))


@pytest.mark.parametrize(
    "not_after",
    [
        "2026-02-30T12:00:00Z",
        "2021-02-29T12:00:00Z",
        "2026-04-31T00:00:00Z",
    ],
)
def test_launch_intent_rejects_impossible_utc_timestamps(not_after):
    value = _valid_launch_intent_dict()
    value["not_after"] = not_after

    with pytest.raises(ValueError):
        launch_intent_from_bytes(_canonical_bytes(value))


def test_public_serializers_reject_invalid_dataclass_inputs():
    profile = load_corpus_builder_profile(PROFILE_PATH)
    bad_profile = CorpusBuilderProfile(
        profile_id=profile.profile_id,
        region=profile.region,
        instance_type=profile.instance_type,
        vcpus=profile.vcpus,
        memory_mib=profile.memory_mib,
        nvme_devices=profile.nvme_devices,
        nvme_total_gib=profile.nvme_total_gib,
        root_volume_gib=profile.root_volume_gib,
        max_runtime_seconds=profile.max_runtime_seconds,
        watchdog_shutdown_seconds=profile.watchdog_shutdown_seconds,
        max_hourly_usd=Decimal("6"),
        max_compute_usd=profile.max_compute_usd,
        bucket_name=profile.bucket_name,
        key_prefix=profile.key_prefix,
    )
    with pytest.raises(ValueError):
        corpus_builder_profile_to_bytes(bad_profile)

    valid_receipt = phase_receipt_from_bytes(_canonical_bytes(_valid_phase_receipt_dict()))
    bad_object = S3ObjectVersion(
        uri=f"s3://{_BUCKET}/v2/builds/{_SHA256_B}/receipt.json",
        version_id="version-id-001",
        bytes=1024,
        sha256=_SHA256_A,
        etag="0123456789abcdef0123456789abcdef",
        sse_algorithm="AES256",
        kms_key_arn="",
    )
    bad_receipt = PhaseReceipt(
        format=valid_receipt.format,
        schema_version=valid_receipt.schema_version,
        build_id=valid_receipt.build_id,
        phase=valid_receipt.phase,
        package_sha256=valid_receipt.package_sha256,
        source_lock_sha256=valid_receipt.source_lock_sha256,
        objects=(bad_object,),
    )
    with pytest.raises(ValueError):
        phase_receipt_to_bytes(bad_receipt)

    valid_intent = launch_intent_from_bytes(_canonical_bytes(_valid_launch_intent_dict()))
    bad_intent = LaunchIntent(
        format=valid_intent.format,
        schema_version=valid_intent.schema_version,
        profile_sha256=valid_intent.profile_sha256,
        package=valid_intent.package,
        source_manifest=valid_intent.source_manifest,
        ami_id=valid_intent.ami_id,
        ami_owner_id=valid_intent.ami_owner_id,
        launch_template_id=valid_intent.launch_template_id,
        launch_template_version=valid_intent.launch_template_version,
        subnet_id=valid_intent.subnet_id,
        security_group_id=valid_intent.security_group_id,
        instance_profile_arn=valid_intent.instance_profile_arn,
        hourly_usd=Decimal("5.492"),
        max_compute_usd=valid_intent.max_compute_usd,
        not_after=valid_intent.not_after,
    )
    with pytest.raises(ValueError):
        launch_intent_to_bytes(bad_intent)
