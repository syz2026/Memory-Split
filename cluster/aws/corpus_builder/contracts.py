"""Closed AWS corpus builder profile and receipt contracts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit


PHASE_RECEIPT_FORMAT = "memorysplit-aws-corpus-phase-v1"
LAUNCH_INTENT_FORMAT = "memorysplit-aws-corpus-launch-v1"

CORPUS_BUCKET = "memorysplit-corpus-056956104102-us-east-1"
CORPUS_KEY_PREFIX = "v2/builds"
MAX_HOURLY_USD = Decimal("5.491")
MAX_COMPUTE_USD = Decimal("131.78")

_CORPUS_ACCOUNT = "056956104102"
_CORPUS_REGION = "us-east-1"

_PROFILE_ID = "aws-i4i.16xlarge-corpus-v1"
_MAX_PROFILE_BYTES = 65_536

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REGION_RE = re.compile(r"^[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+$")
_BUCKET_RE = re.compile(
    r"^(?![0-9]+(?:\.[0-9]+){3}$)(?!-)(?!.*\.\.)(?!.*\.-)(?!.*-\.)"
    r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$"
)
_DECIMAL_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
_UTC_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
)
_AMI_RE = re.compile(r"^ami-[0-9a-f]{8,17}$")
_ACCOUNT_RE = re.compile(r"^[0-9]{12}$")
_LAUNCH_TEMPLATE_RE = re.compile(r"^lt-[0-9a-f]{8,17}$")
_SUBNET_RE = re.compile(r"^subnet-[0-9a-f]{8,17}$")
_SECURITY_GROUP_RE = re.compile(r"^sg-[0-9a-f]{8,17}$")
_TEMPLATE_VERSION_RE = re.compile(r"^[1-9][0-9]*$")
_ETAG_RE = re.compile(r"^[0-9a-f]{32}(?:-[0-9]+)?$")
_KMS_KEY_ARN_RE = re.compile(
    rf"^arn:aws:kms:{_CORPUS_REGION}:{_CORPUS_ACCOUNT}:key/"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_INSTANCE_PROFILE_ARN_RE = re.compile(
    r"^arn:aws:iam::[0-9]{12}:instance-profile/[A-Za-z0-9+=,.@_-]+$"
)
_PHASE_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")

_PROFILE_FIELDS = frozenset(
    {
        "bucket_name",
        "instance_type",
        "key_prefix",
        "max_compute_usd",
        "max_hourly_usd",
        "max_runtime_seconds",
        "memory_mib",
        "nvme_devices",
        "nvme_total_gib",
        "profile_id",
        "region",
        "root_volume_gib",
        "vcpus",
        "watchdog_shutdown_seconds",
    }
)
_S3_OBJECT_FIELDS = frozenset(
    {
        "bytes",
        "etag",
        "kms_key_arn",
        "sha256",
        "sse_algorithm",
        "uri",
        "version_id",
    }
)
_PHASE_RECEIPT_FIELDS = frozenset(
    {
        "build_id",
        "format",
        "objects",
        "package_sha256",
        "phase",
        "schema_version",
        "source_lock_sha256",
    }
)
_LAUNCH_INTENT_FIELDS = frozenset(
    {
        "ami_id",
        "ami_owner_id",
        "format",
        "hourly_usd",
        "instance_profile_arn",
        "launch_template_id",
        "launch_template_version",
        "max_compute_usd",
        "not_after",
        "package",
        "profile_sha256",
        "schema_version",
        "security_group_id",
        "source_manifest",
        "subnet_id",
    }
)


@dataclass(frozen=True)
class CorpusBuilderProfile:
    profile_id: str
    region: str
    instance_type: str
    vcpus: int
    memory_mib: int
    nvme_devices: int
    nvme_total_gib: int
    root_volume_gib: int
    max_runtime_seconds: int
    watchdog_shutdown_seconds: int
    max_hourly_usd: Decimal
    max_compute_usd: Decimal
    bucket_name: str
    key_prefix: str


@dataclass(frozen=True)
class S3ObjectVersion:
    uri: str
    version_id: str
    bytes: int
    sha256: str
    etag: str
    sse_algorithm: str
    kms_key_arn: str


@dataclass(frozen=True)
class PhaseReceipt:
    format: str
    schema_version: int
    build_id: str
    phase: str
    package_sha256: str
    source_lock_sha256: str
    objects: tuple[S3ObjectVersion, ...]


@dataclass(frozen=True)
class LaunchIntent:
    format: str
    schema_version: int
    profile_sha256: str
    package: S3ObjectVersion
    source_manifest: S3ObjectVersion
    ami_id: str
    ami_owner_id: str
    launch_template_id: str
    launch_template_version: str
    subnet_id: str
    security_group_id: str
    instance_profile_arn: str
    hourly_usd: Decimal
    max_compute_usd: Decimal
    not_after: str


def _canonical_json_bytes(value: object) -> bytes:
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


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON repeats field: {key}")
        result[key] = value
    return result


def _object(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{label} fields do not match schema; "
            f"missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )
    return value


def _exact_int(value: object, expected: int, *, label: str) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(f"{label} must be exactly {expected}")
    return value


def _exact_string(value: object, expected: str, *, label: str) -> str:
    if not isinstance(value, str) or value != expected:
        raise ValueError(f"{label} must be exactly {expected!r}")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_decimal_string(value: object, expected: str, *, label: str) -> Decimal:
    if not isinstance(value, str) or value != expected:
        raise ValueError(f"{label} must be exactly {expected!r}")
    if _DECIMAL_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical decimal string")
    return Decimal(value)


def _decimal_string(value: object, *, label: str) -> Decimal:
    if not isinstance(value, str) or _DECIMAL_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical decimal string")
    if "." in value and value.endswith("0"):
        raise ValueError(f"{label} must be a canonical decimal string")
    return Decimal(value)


def _canonical_decimal_from_decimal(
    value: Decimal,
    *,
    label: str,
    maximum: Decimal | None = None,
    exact: str | None = None,
) -> str:
    if value.is_nan() or value.is_infinite():
        raise ValueError(f"{label} must be a finite decimal")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    text = format(value, "f")
    if _DECIMAL_RE.fullmatch(text) is None:
        raise ValueError(f"{label} must be a canonical decimal string")
    if "." in text and text.endswith("0"):
        raise ValueError(f"{label} must be a canonical decimal string")
    if value != Decimal(text):
        raise ValueError(f"{label} must be a canonical decimal string")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} exceeds the approved profile ceiling")
    if exact is not None and text != exact:
        raise ValueError(f"{label} must be exactly {exact!r}")
    return text


def _kms_key_arn(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _KMS_KEY_ARN_RE.fullmatch(value) is None:
        raise ValueError(
            f"{label} must be arn:aws:kms:{_CORPUS_REGION}:{_CORPUS_ACCOUNT}:key/<uuid>"
        )
    return value


def _bounded_price(value: object, *, label: str, maximum: Decimal) -> Decimal:
    parsed = _decimal_string(value, label=label)
    if parsed > maximum:
        raise ValueError(f"{label} exceeds the approved profile ceiling")
    return parsed


def _utc_timestamp(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a UTC timestamp ending with Z")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ValueError(f"{label} must be a valid UTC timestamp") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError(f"{label} must be a valid UTC timestamp")
    return value


def _validate_s3_uri(value: object, *, label: str) -> str:
    if not isinstance(value, str) or "\\" in value:
        raise ValueError(f"{label} must be a safe S3 URI")
    parsed = urlsplit(value)
    path_parts = parsed.path.removeprefix("/").split("/")
    if (
        parsed.scheme != "s3"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
        or parsed.hostname is None
        or _BUCKET_RE.fullmatch(parsed.hostname) is None
        or len(path_parts) < 1
        or any(part in {"", ".", ".."} for part in path_parts)
    ):
        raise ValueError(f"{label} must be a safe S3 URI")
    return value


def _extract_build_id(uri: str, *, label: str) -> str:
    parsed = urlsplit(uri)
    if parsed.hostname != CORPUS_BUCKET:
        raise ValueError(f"{label} must use the corpus bucket")
    parts = parsed.path.removeprefix("/").split("/")
    prefix_parts = CORPUS_KEY_PREFIX.split("/")
    if len(parts) < len(prefix_parts) + 2:
        raise ValueError(f"{label} must be under {CORPUS_KEY_PREFIX}/{{build_id}}/")
    if parts[: len(prefix_parts)] != prefix_parts:
        raise ValueError(f"{label} must be under {CORPUS_KEY_PREFIX}/{{build_id}}/")
    build_id = parts[len(prefix_parts)]
    if _SHA256_RE.fullmatch(build_id) is None:
        raise ValueError(f"{label} build_id must be a lowercase SHA-256")
    if not parts[len(prefix_parts) + 1 :]:
        raise ValueError(f"{label} must name an object under the build prefix")
    return build_id


def _validate_s3_object_version(
    obj: S3ObjectVersion,
    *,
    label: str,
    expected_build_id: str | None = None,
) -> None:
    uri = _validate_s3_uri(obj.uri, label=f"{label}.uri")
    build_id = _extract_build_id(uri, label=f"{label}.uri")
    if expected_build_id is not None and build_id != expected_build_id:
        raise ValueError(f"{label}.uri build_id does not match the receipt build_id")
    if not isinstance(obj.version_id, str) or not obj.version_id:
        raise ValueError(f"{label}.version_id must be non-empty")
    if isinstance(obj.bytes, bool) or not isinstance(obj.bytes, int) or obj.bytes <= 0:
        raise ValueError(f"{label}.bytes must be a positive integer")
    _sha256(obj.sha256, label=f"{label}.sha256")
    if not isinstance(obj.etag, str) or _ETAG_RE.fullmatch(obj.etag) is None:
        raise ValueError(f"{label}.etag must be a lowercase S3 ETag")
    if obj.sse_algorithm != "aws:kms":
        raise ValueError(f"{label}.sse_algorithm must be aws:kms")
    _kms_key_arn(obj.kms_key_arn, label=f"{label}.kms_key_arn")


def _validate_profile(profile: CorpusBuilderProfile) -> None:
    if profile.profile_id != _PROFILE_ID:
        raise ValueError("profile.profile_id drift")
    if profile.region != "us-east-1":
        raise ValueError("profile.region drift")
    if profile.instance_type != "i4i.16xlarge":
        raise ValueError("profile.instance_type drift")
    for field, expected in (
        ("vcpus", 64),
        ("memory_mib", 524_288),
        ("nvme_devices", 4),
        ("nvme_total_gib", 15_000),
        ("root_volume_gib", 200),
        ("max_runtime_seconds", 86_400),
        ("watchdog_shutdown_seconds", 84_600),
    ):
        if getattr(profile, field) != expected:
            raise ValueError(f"profile.{field} drift")
    if profile.max_hourly_usd != MAX_HOURLY_USD:
        raise ValueError("profile.max_hourly_usd drift")
    if profile.max_compute_usd != MAX_COMPUTE_USD:
        raise ValueError("profile.max_compute_usd drift")
    _canonical_decimal_from_decimal(
        profile.max_hourly_usd,
        label="profile.max_hourly_usd",
        exact="5.491",
    )
    _canonical_decimal_from_decimal(
        profile.max_compute_usd,
        label="profile.max_compute_usd",
        exact="131.78",
    )
    if profile.bucket_name != CORPUS_BUCKET:
        raise ValueError("profile.bucket_name drift")
    if profile.key_prefix != CORPUS_KEY_PREFIX:
        raise ValueError("profile.key_prefix drift")


def _validate_phase_receipt(receipt: PhaseReceipt) -> None:
    if receipt.format != PHASE_RECEIPT_FORMAT:
        raise ValueError("phase receipt.format drift")
    if receipt.schema_version != 1:
        raise ValueError("phase receipt.schema_version drift")
    _sha256(receipt.build_id, label="phase receipt.build_id")
    if not _PHASE_RE.fullmatch(receipt.phase):
        raise ValueError("phase receipt.phase drift")
    _sha256(receipt.package_sha256, label="phase receipt.package_sha256")
    _sha256(receipt.source_lock_sha256, label="phase receipt.source_lock_sha256")
    if not receipt.objects:
        raise ValueError("phase receipt.objects must be non-empty")
    uris = [obj.uri for obj in receipt.objects]
    if len(uris) != len(set(uris)):
        raise ValueError("phase receipt.objects repeats an object URI")
    if uris != sorted(uris):
        raise ValueError("phase receipt.objects must be sorted by URI")
    for index, obj in enumerate(receipt.objects):
        _validate_s3_object_version(
            obj,
            label=f"phase receipt.objects[{index}]",
            expected_build_id=receipt.build_id,
        )
    kms_arns = {obj.kms_key_arn for obj in receipt.objects}
    if len(kms_arns) != 1:
        raise ValueError("phase receipt.objects must share one KMS key ARN")


def _validate_launch_intent(intent: LaunchIntent) -> None:
    if intent.format != LAUNCH_INTENT_FORMAT:
        raise ValueError("launch intent.format drift")
    if intent.schema_version != 1:
        raise ValueError("launch intent.schema_version drift")
    _sha256(intent.profile_sha256, label="launch intent.profile_sha256")
    package_build_id = _extract_build_id(intent.package.uri, label="launch intent.package.uri")
    manifest_build_id = _extract_build_id(
        intent.source_manifest.uri,
        label="launch intent.source_manifest.uri",
    )
    if package_build_id != manifest_build_id:
        raise ValueError("launch intent package and source manifest build_id mismatch")
    _validate_s3_object_version(intent.package, label="launch intent.package")
    _validate_s3_object_version(intent.source_manifest, label="launch intent.source_manifest")
    if intent.package.kms_key_arn != intent.source_manifest.kms_key_arn:
        raise ValueError(
            "launch intent package and source manifest KMS key ARN mismatch"
        )
    if not _AMI_RE.fullmatch(intent.ami_id):
        raise ValueError("launch intent.ami_id drift")
    if not _ACCOUNT_RE.fullmatch(intent.ami_owner_id):
        raise ValueError("launch intent.ami_owner_id drift")
    if not _LAUNCH_TEMPLATE_RE.fullmatch(intent.launch_template_id):
        raise ValueError("launch intent.launch_template_id drift")
    if not _TEMPLATE_VERSION_RE.fullmatch(intent.launch_template_version):
        raise ValueError("launch intent.launch_template_version drift")
    if not _SUBNET_RE.fullmatch(intent.subnet_id):
        raise ValueError("launch intent.subnet_id drift")
    if not _SECURITY_GROUP_RE.fullmatch(intent.security_group_id):
        raise ValueError("launch intent.security_group_id drift")
    if not _INSTANCE_PROFILE_ARN_RE.fullmatch(intent.instance_profile_arn):
        raise ValueError("launch intent.instance_profile_arn drift")
    if intent.hourly_usd > MAX_HOURLY_USD:
        raise ValueError("launch intent.hourly_usd exceeds the approved profile ceiling")
    if intent.max_compute_usd > MAX_COMPUTE_USD:
        raise ValueError("launch intent.max_compute_usd exceeds the approved profile ceiling")
    _canonical_decimal_from_decimal(
        intent.hourly_usd,
        label="launch intent.hourly_usd",
        maximum=MAX_HOURLY_USD,
    )
    _canonical_decimal_from_decimal(
        intent.max_compute_usd,
        label="launch intent.max_compute_usd",
        maximum=MAX_COMPUTE_USD,
    )
    _utc_timestamp(intent.not_after, label="launch intent.not_after")


def _s3_object_dict(obj: S3ObjectVersion) -> dict[str, object]:
    return {
        "bytes": obj.bytes,
        "etag": obj.etag,
        "kms_key_arn": obj.kms_key_arn,
        "sha256": obj.sha256,
        "sse_algorithm": obj.sse_algorithm,
        "uri": obj.uri,
        "version_id": obj.version_id,
    }


def s3_object_version_from_dict(
    value: object,
    *,
    expected_build_id: str | None = None,
) -> S3ObjectVersion:
    obj = _object(value, fields=_S3_OBJECT_FIELDS, label="S3 object version")
    uri = _validate_s3_uri(obj["uri"], label="S3 object version.uri")
    build_id = _extract_build_id(uri, label="S3 object version.uri")
    if expected_build_id is not None and build_id != expected_build_id:
        raise ValueError("S3 object version.uri build_id does not match the receipt build_id")
    version_id = obj["version_id"]
    if not isinstance(version_id, str) or not version_id:
        raise ValueError("S3 object version.version_id must be non-empty")
    byte_count = _positive_int(obj["bytes"], label="S3 object version.bytes")
    digest = _sha256(obj["sha256"], label="S3 object version.sha256")
    etag = obj["etag"]
    if not isinstance(etag, str) or _ETAG_RE.fullmatch(etag) is None:
        raise ValueError("S3 object version.etag must be a lowercase S3 ETag")
    sse_algorithm = obj["sse_algorithm"]
    kms_key_arn = _kms_key_arn(obj["kms_key_arn"], label="S3 object version.kms_key_arn")
    if sse_algorithm != "aws:kms":
        raise ValueError("S3 object version.sse_algorithm must be aws:kms")
    parsed = S3ObjectVersion(
        uri=uri,
        version_id=version_id,
        bytes=byte_count,
        sha256=digest,
        etag=etag,
        sse_algorithm="aws:kms",
        kms_key_arn=kms_key_arn,
    )
    _validate_s3_object_version(parsed, label="S3 object version", expected_build_id=expected_build_id)
    return parsed


def _parse_s3_objects(
    value: object,
    *,
    label: str,
    expected_build_id: str,
) -> tuple[S3ObjectVersion, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    objects = tuple(
        s3_object_version_from_dict(item, expected_build_id=expected_build_id)
        for item in value
    )
    uris = [obj.uri for obj in objects]
    if len(uris) != len(set(uris)):
        raise ValueError(f"{label} repeats an object URI")
    if uris != sorted(uris):
        raise ValueError(f"{label} object URIs must be sorted and unique")
    kms_arns = {obj.kms_key_arn for obj in objects}
    if len(kms_arns) != 1:
        raise ValueError(f"{label} must share one KMS key ARN")
    return objects


def _parse_profile(raw: object) -> CorpusBuilderProfile:
    value = _object(raw, fields=_PROFILE_FIELDS, label="profile")
    profile_id = _exact_string(
        value["profile_id"], _PROFILE_ID, label="profile.profile_id"
    )
    region = _exact_string(value["region"], "us-east-1", label="profile.region")
    instance_type = _exact_string(
        value["instance_type"], "i4i.16xlarge", label="profile.instance_type"
    )
    vcpus = _exact_int(value["vcpus"], 64, label="profile.vcpus")
    memory_mib = _exact_int(value["memory_mib"], 524_288, label="profile.memory_mib")
    nvme_devices = _exact_int(value["nvme_devices"], 4, label="profile.nvme_devices")
    nvme_total_gib = _exact_int(
        value["nvme_total_gib"], 15_000, label="profile.nvme_total_gib"
    )
    root_volume_gib = _exact_int(
        value["root_volume_gib"], 200, label="profile.root_volume_gib"
    )
    max_runtime_seconds = _exact_int(
        value["max_runtime_seconds"], 86_400, label="profile.max_runtime_seconds"
    )
    watchdog_shutdown_seconds = _exact_int(
        value["watchdog_shutdown_seconds"],
        84_600,
        label="profile.watchdog_shutdown_seconds",
    )
    max_hourly_usd = _exact_decimal_string(
        value["max_hourly_usd"], "5.491", label="profile.max_hourly_usd"
    )
    max_compute_usd = _exact_decimal_string(
        value["max_compute_usd"], "131.78", label="profile.max_compute_usd"
    )
    bucket_name = _exact_string(
        value["bucket_name"],
        CORPUS_BUCKET,
        label="profile.bucket_name",
    )
    key_prefix = _exact_string(
        value["key_prefix"], CORPUS_KEY_PREFIX, label="profile.key_prefix"
    )
    if _REGION_RE.fullmatch(region) is None:
        raise ValueError("profile.region is not a valid explicit region")
    if _BUCKET_RE.fullmatch(bucket_name) is None:
        raise ValueError("profile.bucket_name is not a valid S3 bucket name")
    profile = CorpusBuilderProfile(
        profile_id=profile_id,
        region=region,
        instance_type=instance_type,
        vcpus=vcpus,
        memory_mib=memory_mib,
        nvme_devices=nvme_devices,
        nvme_total_gib=nvme_total_gib,
        root_volume_gib=root_volume_gib,
        max_runtime_seconds=max_runtime_seconds,
        watchdog_shutdown_seconds=watchdog_shutdown_seconds,
        max_hourly_usd=max_hourly_usd,
        max_compute_usd=max_compute_usd,
        bucket_name=bucket_name,
        key_prefix=key_prefix,
    )
    _validate_profile(profile)
    return profile


def _profile_dict(profile: CorpusBuilderProfile) -> dict[str, object]:
    return {
        "bucket_name": profile.bucket_name,
        "instance_type": profile.instance_type,
        "key_prefix": profile.key_prefix,
        "max_compute_usd": _canonical_decimal_from_decimal(
            profile.max_compute_usd,
            label="profile.max_compute_usd",
            exact="131.78",
        ),
        "max_hourly_usd": _canonical_decimal_from_decimal(
            profile.max_hourly_usd,
            label="profile.max_hourly_usd",
            exact="5.491",
        ),
        "max_runtime_seconds": profile.max_runtime_seconds,
        "memory_mib": profile.memory_mib,
        "nvme_devices": profile.nvme_devices,
        "nvme_total_gib": profile.nvme_total_gib,
        "profile_id": profile.profile_id,
        "region": profile.region,
        "root_volume_gib": profile.root_volume_gib,
        "vcpus": profile.vcpus,
        "watchdog_shutdown_seconds": profile.watchdog_shutdown_seconds,
    }


def load_corpus_builder_profile(path: Path | str) -> CorpusBuilderProfile:
    profile_path = Path(path)
    if profile_path.is_symlink():
        raise ValueError(f"profile must not be a symlink: {profile_path}")
    if not profile_path.is_file():
        raise ValueError(f"profile is not a regular file: {profile_path}")
    data = profile_path.read_bytes()
    if len(data) > _MAX_PROFILE_BYTES:
        raise ValueError("profile exceeds 64 KiB")
    try:
        raw = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"profile contains non-finite value: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("profile must contain valid UTF-8 JSON") from error
    profile = _parse_profile(raw)
    if _canonical_json_bytes(_profile_dict(profile)) != data:
        raise ValueError("profile is not canonical JSON")
    return profile


def corpus_builder_profile_to_bytes(profile: CorpusBuilderProfile) -> bytes:
    _validate_profile(profile)
    return _canonical_json_bytes(_profile_dict(profile))


def _parse_phase_receipt(raw: object) -> PhaseReceipt:
    value = _object(raw, fields=_PHASE_RECEIPT_FIELDS, label="phase receipt")
    receipt_format = _exact_string(
        value["format"], PHASE_RECEIPT_FORMAT, label="phase receipt.format"
    )
    schema_version = _exact_int(
        value["schema_version"], 1, label="phase receipt.schema_version"
    )
    build_id = _sha256(value["build_id"], label="phase receipt.build_id")
    phase = value["phase"]
    if not isinstance(phase, str) or _PHASE_RE.fullmatch(phase) is None:
        raise ValueError("phase receipt.phase must be a lowercase phase name")
    package_sha256 = _sha256(
        value["package_sha256"], label="phase receipt.package_sha256"
    )
    source_lock_sha256 = _sha256(
        value["source_lock_sha256"], label="phase receipt.source_lock_sha256"
    )
    objects = _parse_s3_objects(
        value["objects"],
        label="phase receipt.objects",
        expected_build_id=build_id,
    )
    receipt = PhaseReceipt(
        format=receipt_format,
        schema_version=schema_version,
        build_id=build_id,
        phase=phase,
        package_sha256=package_sha256,
        source_lock_sha256=source_lock_sha256,
        objects=objects,
    )
    _validate_phase_receipt(receipt)
    return receipt


def _phase_receipt_dict(receipt: PhaseReceipt) -> dict[str, object]:
    return {
        "build_id": receipt.build_id,
        "format": receipt.format,
        "objects": [_s3_object_dict(obj) for obj in receipt.objects],
        "package_sha256": receipt.package_sha256,
        "phase": receipt.phase,
        "schema_version": receipt.schema_version,
        "source_lock_sha256": receipt.source_lock_sha256,
    }


def phase_receipt_from_bytes(payload: bytes) -> PhaseReceipt:
    if not payload.endswith(b"\n"):
        raise ValueError("phase receipt must end with one terminal newline")
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"phase receipt contains non-finite value: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("phase receipt must contain valid UTF-8 JSON") from error
    receipt = _parse_phase_receipt(raw)
    if _canonical_json_bytes(_phase_receipt_dict(receipt)) != payload:
        raise ValueError("phase receipt is not canonical JSON")
    return receipt


def phase_receipt_to_bytes(receipt: PhaseReceipt) -> bytes:
    _validate_phase_receipt(receipt)
    return _canonical_json_bytes(_phase_receipt_dict(receipt))


def _parse_launch_intent(raw: object) -> LaunchIntent:
    value = _object(raw, fields=_LAUNCH_INTENT_FIELDS, label="launch intent")
    intent_format = _exact_string(
        value["format"], LAUNCH_INTENT_FORMAT, label="launch intent.format"
    )
    schema_version = _exact_int(
        value["schema_version"], 1, label="launch intent.schema_version"
    )
    profile_sha256 = _sha256(
        value["profile_sha256"], label="launch intent.profile_sha256"
    )
    package = s3_object_version_from_dict(value["package"])
    source_manifest = s3_object_version_from_dict(value["source_manifest"])
    package_build_id = _extract_build_id(package.uri, label="launch intent.package.uri")
    manifest_build_id = _extract_build_id(
        source_manifest.uri,
        label="launch intent.source_manifest.uri",
    )
    if package_build_id != manifest_build_id:
        raise ValueError("launch intent package and source manifest build_id mismatch")
    ami_id = value["ami_id"]
    if not isinstance(ami_id, str) or _AMI_RE.fullmatch(ami_id) is None:
        raise ValueError("launch intent.ami_id must be an immutable AMI ID")
    ami_owner_id = value["ami_owner_id"]
    if not isinstance(ami_owner_id, str) or _ACCOUNT_RE.fullmatch(ami_owner_id) is None:
        raise ValueError("launch intent.ami_owner_id must be a 12-digit account ID")
    launch_template_id = value["launch_template_id"]
    if (
        not isinstance(launch_template_id, str)
        or _LAUNCH_TEMPLATE_RE.fullmatch(launch_template_id) is None
    ):
        raise ValueError("launch intent.launch_template_id must be an explicit template ID")
    launch_template_version = value["launch_template_version"]
    if (
        not isinstance(launch_template_version, str)
        or _TEMPLATE_VERSION_RE.fullmatch(launch_template_version) is None
    ):
        raise ValueError(
            "launch intent.launch_template_version must be an explicit numeric version"
        )
    subnet_id = value["subnet_id"]
    if not isinstance(subnet_id, str) or _SUBNET_RE.fullmatch(subnet_id) is None:
        raise ValueError("launch intent.subnet_id must be an explicit subnet ID")
    security_group_id = value["security_group_id"]
    if (
        not isinstance(security_group_id, str)
        or _SECURITY_GROUP_RE.fullmatch(security_group_id) is None
    ):
        raise ValueError(
            "launch intent.security_group_id must be an explicit security group ID"
        )
    instance_profile_arn = value["instance_profile_arn"]
    if (
        not isinstance(instance_profile_arn, str)
        or _INSTANCE_PROFILE_ARN_RE.fullmatch(instance_profile_arn) is None
    ):
        raise ValueError(
            "launch intent.instance_profile_arn must be an instance-profile ARN"
        )
    hourly_usd = _bounded_price(
        value["hourly_usd"],
        label="launch intent.hourly_usd",
        maximum=MAX_HOURLY_USD,
    )
    max_compute_usd = _bounded_price(
        value["max_compute_usd"],
        label="launch intent.max_compute_usd",
        maximum=MAX_COMPUTE_USD,
    )
    not_after = _utc_timestamp(value["not_after"], label="launch intent.not_after")
    intent = LaunchIntent(
        format=intent_format,
        schema_version=schema_version,
        profile_sha256=profile_sha256,
        package=package,
        source_manifest=source_manifest,
        ami_id=ami_id,
        ami_owner_id=ami_owner_id,
        launch_template_id=launch_template_id,
        launch_template_version=launch_template_version,
        subnet_id=subnet_id,
        security_group_id=security_group_id,
        instance_profile_arn=instance_profile_arn,
        hourly_usd=hourly_usd,
        max_compute_usd=max_compute_usd,
        not_after=not_after,
    )
    _validate_launch_intent(intent)
    return intent


def _launch_intent_dict(intent: LaunchIntent) -> dict[str, object]:
    return {
        "ami_id": intent.ami_id,
        "ami_owner_id": intent.ami_owner_id,
        "format": intent.format,
        "hourly_usd": _canonical_decimal_from_decimal(
            intent.hourly_usd,
            label="launch intent.hourly_usd",
            maximum=MAX_HOURLY_USD,
        ),
        "instance_profile_arn": intent.instance_profile_arn,
        "launch_template_id": intent.launch_template_id,
        "launch_template_version": intent.launch_template_version,
        "max_compute_usd": _canonical_decimal_from_decimal(
            intent.max_compute_usd,
            label="launch intent.max_compute_usd",
            maximum=MAX_COMPUTE_USD,
        ),
        "not_after": intent.not_after,
        "package": _s3_object_dict(intent.package),
        "profile_sha256": intent.profile_sha256,
        "security_group_id": intent.security_group_id,
        "source_manifest": _s3_object_dict(intent.source_manifest),
        "subnet_id": intent.subnet_id,
        "schema_version": intent.schema_version,
    }


def launch_intent_from_bytes(payload: bytes) -> LaunchIntent:
    if not payload.endswith(b"\n"):
        raise ValueError("launch intent must end with one terminal newline")
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"launch intent contains non-finite value: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("launch intent must contain valid UTF-8 JSON") from error
    intent = _parse_launch_intent(raw)
    if _canonical_json_bytes(_launch_intent_dict(intent)) != payload:
        raise ValueError("launch intent is not canonical JSON")
    return intent


def launch_intent_to_bytes(intent: LaunchIntent) -> bytes:
    _validate_launch_intent(intent)
    return _canonical_json_bytes(_launch_intent_dict(intent))
