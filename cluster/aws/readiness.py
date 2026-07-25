"""Read-only AWS inspection and exact P6-B200 hardware admission.

Admission is deliberately not a regex over GPU marketing names. A protected
135M cell may only run on a node whose complete attested identity equals the
frozen ``aws-p6-b200.48xlarge-135m-v1`` profile, so P5, H100, H200, B300, and
mixed-hardware evidence can never be substituted under B200 authority.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INSTANCE_TYPES = ("p5.48xlarge", "p5en.48xlarge", "p6-b200.48xlarge")
_REGION_RE = re.compile(r"^[a-z]{2}(?:-gov)?-[a-z]+-\d$")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
B200_PROFILE_PATH = (
    REPOSITORY_ROOT / "cluster" / "profiles" / "aws-p6-b200.48xlarge-135m-v1.json"
)
B200_AUTHORITY_ID = "memorysplit-p6-b200-135m-v1"
B200_INSTANCE_TYPE = "p6-b200.48xlarge"
B200_GPU_COUNT = 8
B200_GPUS_PER_PAIR = 2
QUALIFICATION_GATES = (
    "device_query",
    "nvidia_smi_attestation",
    "nccl_all_reduce",
    "one_step_training",
    "throughput_100_updates",
    "memory_headroom",
    "checkpoint_write",
    "exact_resume_equality",
)
SOFTWARE_FLOOR_FIELDS = (
    "cuda_version",
    "driver_version",
    "efa_installer_version",
    "kernel_version",
    "nccl_version",
    "ofi_nccl_version",
    "torch_version",
)

_HEX = frozenset("0123456789abcdef")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_PREFIX_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
_AMI_RE = re.compile(r"^ami-(?:[0-9a-f]{8}|[0-9a-f]{17})$")
_EC2_ID_RE = re.compile(r"^i-(?:[0-9a-f]{8}|[0-9a-f]{17})$")
_ACCOUNT_RE = re.compile(r"^\d{12}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GPU_UUID_RE = re.compile(
    r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_GPU_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")
_AMI_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ()\[\]./_-]{2,127}$")
_REPOSITORY_RE = re.compile(
    r"^\d{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[a-z0-9][a-z0-9._/-]{0,205}$"
)
_VERSION_RE = re.compile(r"^(\d+(?:\.\d+){0,3})([A-Za-z0-9._+-]*)$")
_PAIR_ID_RE = re.compile(r"^(?P<prefix>[a-z0-9_]+?)(?P<seed>\d)$")

_PROFILE_FIELDS = frozenset(
    {
        "authority_id",
        "cohort_id",
        "container_runtime",
        "dataset_contract_id",
        "hardware",
        "image",
        "pair_geometry",
        "pair_id_prefix",
        "pending_confirmation",
        "placement",
        "profile_id",
        "qualification",
        "rejected_authorities",
        "schema_version",
        "software_floors",
    }
)
_HARDWARE_FIELDS = frozenset(
    {
        "gpu_compute_capability",
        "gpu_count",
        "gpu_memory_mib",
        "gpu_name",
        "instance_type",
        "local_storage_gb_floor",
        "memory_gib_floor",
        "vcpus",
    }
)
_IMAGE_FIELDS = frozenset({"ami_id", "ami_name", "ami_owner"})
_RUNTIME_FIELDS = frozenset({"digest", "repository"})
_PLACEMENT_FIELDS = frozenset({"availability_zone", "region"})
_GEOMETRY_FIELDS = frozenset({"gpus_per_pair", "pairs_per_node"})
_QUALIFICATION_FIELDS = frozenset({"memory_headroom_fraction_floor"})
_PENDING_FIELDS = frozenset({"field", "reason", "resolve_with"})

_ATTESTATION_FIELDS = frozenset(
    {
        "ami_id",
        "ami_owner",
        "availability_zone",
        "container_image_digest",
        "container_image_repository",
        "cuda_version",
        "driver_version",
        "efa_installer_version",
        "gpus",
        "instance_id",
        "instance_type",
        "kernel_version",
        "local_storage_gb",
        "memory_gib",
        "nccl_version",
        "ofi_nccl_version",
        "profile_id",
        "region",
        "schema_version",
        "torch_version",
        "vcpus",
    }
)
_GPU_FIELDS = frozenset(
    {"compute_capability", "index", "memory_mib", "name", "uuid"}
)
_ALLOCATION_FIELDS = frozenset({"arms", "pair_id", "seed"})
_ALLOCATION_ARM_FIELDS = frozenset({"gpu_uuid"})
_SITE_FIELDS = frozenset(
    {
        "attestation",
        "authority_id",
        "cohort_id",
        "pair_allocations",
        "qualification_gates",
        "schema_version",
    }
)
_GATE_FIELDS = {
    "checkpoint_write": frozenset(
        {"bytes", "checkpoint_sha256", "evidence_sha256", "fsync", "status"}
    ),
    "device_query": frozenset(
        {"devices_passed", "evidence_sha256", "result", "status"}
    ),
    "exact_resume_equality": frozenset(
        {
            "continuous_state_sha256",
            "evidence_sha256",
            "resume_exact",
            "resumed_state_sha256",
            "status",
            "updates",
        }
    ),
    "memory_headroom": frozenset(
        {
            "device_total_mib",
            "evidence_sha256",
            "headroom_fraction",
            "oom_detected",
            "peak_allocated_mib",
            "status",
        }
    ),
    "nccl_all_reduce": frozenset(
        {
            "algorithm_bandwidth_gbps",
            "errors",
            "evidence_sha256",
            "gpu_count",
            "pair_all_reduce_passed",
            "status",
        }
    ),
    "nvidia_smi_attestation": frozenset(
        {
            "attestation_sha256",
            "ecc_uncorrected_errors",
            "evidence_sha256",
            "gpu_count",
            "status",
        }
    ),
    "one_step_training": frozenset(
        {"arms", "evidence_sha256", "oom_detected", "status", "updates"}
    ),
    "throughput_100_updates": frozenset(
        {"evidence_sha256", "oom_detected", "status", "tokens_per_second", "updates"}
    ),
}
_ARMS = ("dense", "split90")


class HardwareAdmissionError(ValueError):
    """Raised when node evidence is not the exact frozen hardware profile."""


def canonical_digest(value: object) -> str:
    """Digest a JSON document under one canonical serialization."""

    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise HardwareAdmissionError("evidence is not canonically encodable") from error
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _unique_object(pairs):
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HardwareAdmissionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _exact_mapping(
    value: object,
    required: frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HardwareAdmissionError(f"{label} must be a JSON object")  # noqa: TRY004
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing or unknown:
        raise HardwareAdmissionError(
            f"{label} fields do not match the frozen schema; "
            f"missing={missing}, unknown={unknown}"
        )
    return value


def _text(value: object, *, field: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise HardwareAdmissionError(f"{field} is missing, unknown, or malformed")
    return value


def _integer(
    value: object,
    *,
    field: str,
    minimum: int = 1,
    maximum: int = 1 << 40,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise HardwareAdmissionError(
            f"{field} must be an integer between {minimum} and {maximum}"
        )
    return value


def _number(
    value: object,
    *,
    field: str,
    minimum: float,
    maximum: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value != value
        or value < minimum
        or value > maximum
    ):
        raise HardwareAdmissionError(
            f"{field} must be a real number in [{minimum}, {maximum}]"
        )
    return float(value)


def _flag(value: object, *, field: str, expected: bool) -> bool:
    if value is not expected:
        raise HardwareAdmissionError(f"{field} must be exactly {expected}")
    return expected


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX:
        raise HardwareAdmissionError(f"{field} is not a SHA-256 digest")
    return value


def _version(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise HardwareAdmissionError(f"{field} is missing or not a version string")
    match = _VERSION_RE.fullmatch(value)
    if match is None:
        raise HardwareAdmissionError(f"{field} {value!r} is not a parsable version")
    return tuple(int(part) for part in match.group(1).split("."))


def _version_text(value: object, *, field: str) -> str:
    _version(value, field=field)
    return str(value)


def _version_floor(
    value: object,
    floor: str | None,
    *,
    field: str,
) -> str:
    if floor is None:
        raise HardwareAdmissionError(f"{field} floor is pending confirmation")
    if _version(value, field=field) < _version(floor, field=f"{field} floor"):
        raise HardwareAdmissionError(
            f"{field} {value!r} is below the frozen floor {floor!r}"
        )
    return str(value)


def _exact(value: object, expected: object, *, field: str) -> Any:
    if expected is None:
        raise HardwareAdmissionError(f"{field} is pending confirmation")
    if value != expected or type(value) is not type(expected):
        raise HardwareAdmissionError(
            f"{field} {value!r} is not the frozen value {expected!r}"
        )
    return value


@dataclass(frozen=True)
class B200HardwareProfile:
    """One exact, immutable P6-B200 node identity."""

    schema_version: int
    profile_id: str
    authority_id: str
    cohort_id: str
    dataset_contract_id: str
    pair_id_prefix: str
    rejected_authorities: tuple[str, ...]
    instance_type: str
    gpu_count: int
    gpu_name: str | None
    gpu_memory_mib: int | None
    gpu_compute_capability: str | None
    vcpus: int
    memory_gib_floor: int
    local_storage_gb_floor: int
    software_floors: Mapping[str, str | None]
    ami_id: str | None
    ami_name: str | None
    ami_owner: str | None
    container_repository: str | None
    container_digest: str | None
    region: str
    availability_zone: str | None
    gpus_per_pair: int
    pairs_per_node: int
    memory_headroom_fraction_floor: float | None
    pending_confirmation: tuple[str, ...]
    sha256: str
    source_path: Path

    @property
    def launch_ready(self) -> bool:
        return not self.pending_confirmation


def _declared_pending(document: Mapping[str, Any]) -> tuple[str, ...]:
    entries = document["pending_confirmation"]
    if not isinstance(entries, list):
        raise HardwareAdmissionError(  # noqa: TRY004
            "pending_confirmation must be a list"
        )
    fields = []
    for entry in entries:
        record = _exact_mapping(entry, _PENDING_FIELDS, label="pending_confirmation")
        for key in sorted(_PENDING_FIELDS):
            if not isinstance(record[key], str) or not record[key]:
                raise HardwareAdmissionError(
                    f"pending_confirmation {key} must be a non-empty string"
                )
        fields.append(record["field"])
    if fields != sorted(fields) or len(fields) != len(set(fields)):
        raise HardwareAdmissionError(
            "pending_confirmation entries must be unique and sorted by field"
        )
    return tuple(fields)


def _confirmable(
    section: Mapping[str, Any],
    *,
    name: str,
    key: str,
    validator: Callable[[object, str], Any],
    observed: set[str],
) -> Any:
    dotted = f"{name}.{key}"
    value = section[key]
    if value is None:
        observed.add(dotted)
        return None
    return validator(value, dotted)


def load_hardware_profile(
    path: Path | str = B200_PROFILE_PATH,
) -> B200HardwareProfile:
    """Load one exact hardware profile, failing closed on any deviation."""

    source = Path(path)
    if not source.is_file() or source.is_symlink() or source.stat().st_size > 1 << 20:
        raise HardwareAdmissionError(
            f"hardware profile is missing, unsafe, or oversized: {source}"
        )
    data = source.read_bytes()
    try:
        document = json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                HardwareAdmissionError(f"non-finite JSON value: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HardwareAdmissionError("hardware profile is not valid UTF-8 JSON") from error
    document = _exact_mapping(document, _PROFILE_FIELDS, label="hardware profile")
    if document["schema_version"] != 1 or isinstance(document["schema_version"], bool):
        raise HardwareAdmissionError("hardware profile schema_version must be exactly 1")

    profile_id = _text(document["profile_id"], field="profile_id", pattern=_ID_RE)
    authority_id = _text(document["authority_id"], field="authority_id", pattern=_ID_RE)
    if authority_id != B200_AUTHORITY_ID:
        raise HardwareAdmissionError(
            f"authority {authority_id!r} is not the frozen B200 authority "
            f"{B200_AUTHORITY_ID!r}"
        )
    rejected = document["rejected_authorities"]
    if (
        not isinstance(rejected, list)
        or not rejected
        or rejected != sorted(rejected)
        or len(rejected) != len(set(rejected))
        or authority_id in rejected
        or any(
            not isinstance(item, str) or not _ID_RE.fullmatch(item) for item in rejected
        )
    ):
        raise HardwareAdmissionError(
            "rejected_authorities must be unique, sorted, and exclude this authority"
        )

    hardware = _exact_mapping(
        document["hardware"],
        _HARDWARE_FIELDS,
        label="hardware profile hardware",
    )
    image = _exact_mapping(document["image"], _IMAGE_FIELDS, label="hardware profile image")
    runtime = _exact_mapping(
        document["container_runtime"],
        _RUNTIME_FIELDS,
        label="hardware profile container_runtime",
    )
    placement = _exact_mapping(
        document["placement"],
        _PLACEMENT_FIELDS,
        label="hardware profile placement",
    )
    geometry = _exact_mapping(
        document["pair_geometry"],
        _GEOMETRY_FIELDS,
        label="hardware profile pair_geometry",
    )
    qualification = _exact_mapping(
        document["qualification"],
        _QUALIFICATION_FIELDS,
        label="hardware profile qualification",
    )
    floors = _exact_mapping(
        document["software_floors"],
        frozenset(SOFTWARE_FLOOR_FIELDS),
        label="hardware profile software_floors",
    )

    instance_type = hardware["instance_type"]
    if instance_type != B200_INSTANCE_TYPE:
        raise HardwareAdmissionError(
            f"hardware.instance_type {instance_type!r} is not {B200_INSTANCE_TYPE!r}"
        )
    gpu_count = _integer(hardware["gpu_count"], field="hardware.gpu_count", maximum=8)
    gpus_per_pair = _integer(
        geometry["gpus_per_pair"],
        field="pair_geometry.gpus_per_pair",
        maximum=B200_GPUS_PER_PAIR,
    )
    pairs_per_node = _integer(
        geometry["pairs_per_node"],
        field="pair_geometry.pairs_per_node",
        maximum=4,
    )
    if (
        gpu_count != B200_GPU_COUNT
        or gpus_per_pair != B200_GPUS_PER_PAIR
        or gpus_per_pair * pairs_per_node != gpu_count
    ):
        raise HardwareAdmissionError(
            "pair geometry must be four disjoint two-GPU pairs on eight devices"
        )

    observed: set[str] = set()
    profile = B200HardwareProfile(
        schema_version=1,
        profile_id=profile_id,
        authority_id=authority_id,
        cohort_id=_text(document["cohort_id"], field="cohort_id", pattern=_ID_RE),
        dataset_contract_id=_text(
            document["dataset_contract_id"],
            field="dataset_contract_id",
            pattern=_ID_RE,
        ),
        pair_id_prefix=_text(
            document["pair_id_prefix"],
            field="pair_id_prefix",
            pattern=_PREFIX_RE,
        ),
        rejected_authorities=tuple(rejected),
        instance_type=instance_type,
        gpu_count=gpu_count,
        gpu_name=_confirmable(
            hardware,
            name="hardware",
            key="gpu_name",
            validator=lambda value, field: _text(
                value,
                field=field,
                pattern=_GPU_NAME_RE,
            ),
            observed=observed,
        ),
        gpu_memory_mib=_confirmable(
            hardware,
            name="hardware",
            key="gpu_memory_mib",
            validator=lambda value, field: _integer(
                value,
                field=field,
                minimum=1_024,
                maximum=1 << 22,
            ),
            observed=observed,
        ),
        gpu_compute_capability=_confirmable(
            hardware,
            name="hardware",
            key="gpu_compute_capability",
            validator=lambda value, field: _version_text(value, field=field),
            observed=observed,
        ),
        vcpus=_integer(hardware["vcpus"], field="hardware.vcpus", maximum=1_024),
        memory_gib_floor=_integer(
            hardware["memory_gib_floor"],
            field="hardware.memory_gib_floor",
            maximum=1 << 20,
        ),
        local_storage_gb_floor=_integer(
            hardware["local_storage_gb_floor"],
            field="hardware.local_storage_gb_floor",
            maximum=1 << 22,
        ),
        software_floors={
            key: _confirmable(
                floors,
                name="software_floors",
                key=key,
                validator=lambda value, field: _version_text(value, field=field),
                observed=observed,
            )
            for key in SOFTWARE_FLOOR_FIELDS
        },
        ami_id=_confirmable(
            image,
            name="image",
            key="ami_id",
            validator=lambda value, field: _text(value, field=field, pattern=_AMI_RE),
            observed=observed,
        ),
        ami_name=_confirmable(
            image,
            name="image",
            key="ami_name",
            validator=lambda value, field: _text(
                value,
                field=field,
                pattern=_AMI_NAME_RE,
            ),
            observed=observed,
        ),
        ami_owner=_confirmable(
            image,
            name="image",
            key="ami_owner",
            validator=lambda value, field: _text(
                value,
                field=field,
                pattern=_ACCOUNT_RE,
            ),
            observed=observed,
        ),
        container_repository=_confirmable(
            runtime,
            name="container_runtime",
            key="repository",
            validator=lambda value, field: _text(
                value,
                field=field,
                pattern=_REPOSITORY_RE,
            ),
            observed=observed,
        ),
        container_digest=_confirmable(
            runtime,
            name="container_runtime",
            key="digest",
            validator=lambda value, field: _text(
                value,
                field=field,
                pattern=_DIGEST_RE,
            ),
            observed=observed,
        ),
        region=_text(
            placement["region"],
            field="placement.region",
            pattern=_REGION_RE,
        ),
        availability_zone=_confirmable(
            placement,
            name="placement",
            key="availability_zone",
            validator=lambda value, field: _text(
                value,
                field=field,
                pattern=re.compile(
                    rf"^{re.escape(str(placement['region']))}[a-z]$"
                ),
            ),
            observed=observed,
        ),
        gpus_per_pair=gpus_per_pair,
        pairs_per_node=pairs_per_node,
        memory_headroom_fraction_floor=_confirmable(
            qualification,
            name="qualification",
            key="memory_headroom_fraction_floor",
            validator=lambda value, field: _number(
                value,
                field=field,
                minimum=0.0,
                maximum=1.0,
            ),
            observed=observed,
        ),
        pending_confirmation=tuple(sorted(observed)),
        sha256=hashlib.sha256(data).hexdigest(),
        source_path=source.resolve(),
    )
    declared = _declared_pending(document)
    if tuple(sorted(observed)) != declared:
        raise HardwareAdmissionError(
            "pending_confirmation does not enumerate exactly the unresolved "
            f"fields; declared={list(declared)}, unresolved={sorted(observed)}"
        )
    return profile


def _require_authority(profile: B200HardwareProfile, asserted: object) -> str:
    if not isinstance(asserted, str) or not asserted:
        raise HardwareAdmissionError("asserted authority must be a non-empty string")
    if asserted in profile.rejected_authorities:
        raise HardwareAdmissionError(
            f"authority {asserted!r} is explicitly rejected by "
            f"{profile.profile_id}; B200 evidence cannot be reused under it"
        )
    if asserted != profile.authority_id:
        raise HardwareAdmissionError(
            f"asserted authority {asserted!r} is not the frozen B200 authority "
            f"{profile.authority_id!r}"
        )
    if not profile.launch_ready:
        raise HardwareAdmissionError(
            f"{profile.profile_id} still has fields pending confirmation: "
            f"{list(profile.pending_confirmation)}"
        )
    return asserted


def _admit_gpu_inventory(
    profile: B200HardwareProfile,
    gpus: object,
) -> tuple[dict[str, int], list[str]]:
    if not isinstance(gpus, list) or len(gpus) != profile.gpu_count:
        raise HardwareAdmissionError(
            f"gpus must attest exactly {profile.gpu_count} devices"
        )
    by_uuid: dict[str, int] = {}
    ordered: list[str] = []
    for position, device in enumerate(gpus):
        record = _exact_mapping(device, _GPU_FIELDS, label=f"gpus[{position}]")
        index = _integer(
            record["index"],
            field=f"gpus[{position}].index",
            minimum=0,
            maximum=profile.gpu_count - 1,
        )
        if index != position:
            raise HardwareAdmissionError(
                f"gpus must be attested in ascending index order at {position}"
            )
        name = record["name"]
        if name != profile.gpu_name:
            raise HardwareAdmissionError(
                f"gpu {index} name {name!r} is not the frozen device "
                f"{profile.gpu_name!r}"
            )
        _exact(
            record["memory_mib"],
            profile.gpu_memory_mib,
            field=f"gpu {index} memory_mib",
        )
        _exact(
            record["compute_capability"],
            profile.gpu_compute_capability,
            field=f"gpu {index} compute_capability",
        )
        uuid = _text(
            record["uuid"],
            field=f"gpus[{position}].uuid",
            pattern=_GPU_UUID_RE,
        )
        if uuid in by_uuid:
            raise HardwareAdmissionError(f"gpu uuid {uuid} is attested twice")
        by_uuid[uuid] = index
        ordered.append(uuid)
    return by_uuid, ordered


def admit_b200_node(
    *,
    profile: B200HardwareProfile,
    attestation: object,
    asserted_authority: object,
) -> dict[str, Any]:
    """Admit one node only when every attested fact equals the frozen profile."""

    authority = _require_authority(profile, asserted_authority)
    record = _exact_mapping(attestation, _ATTESTATION_FIELDS, label="node attestation")
    if record["schema_version"] != 1 or isinstance(record["schema_version"], bool):
        raise HardwareAdmissionError("node attestation schema_version must be exactly 1")
    if record["profile_id"] != profile.profile_id:
        raise HardwareAdmissionError(
            f"attestation profile_id {record['profile_id']!r} does not carry the "
            f"frozen B200 authority {profile.profile_id!r}"
        )
    _exact(record["instance_type"], profile.instance_type, field="instance_type")
    _exact(record["region"], profile.region, field="region")
    _exact(
        record["availability_zone"],
        profile.availability_zone,
        field="availability_zone",
    )
    _exact(record["ami_id"], profile.ami_id, field="ami_id")
    _exact(record["ami_owner"], profile.ami_owner, field="ami_owner")
    _exact(
        record["container_image_repository"],
        profile.container_repository,
        field="container_image_repository",
    )
    _exact(
        record["container_image_digest"],
        profile.container_digest,
        field="container_image_digest",
    )
    instance_id = _text(
        record["instance_id"],
        field="instance_id",
        pattern=_EC2_ID_RE,
    )
    _exact(record["vcpus"], profile.vcpus, field="vcpus")
    if _integer(record["memory_gib"], field="memory_gib") < profile.memory_gib_floor:
        raise HardwareAdmissionError(
            f"memory_gib is below the frozen floor {profile.memory_gib_floor}"
        )
    if (
        _integer(record["local_storage_gb"], field="local_storage_gb")
        < profile.local_storage_gb_floor
    ):
        raise HardwareAdmissionError(
            "local_storage_gb is below the frozen floor "
            f"{profile.local_storage_gb_floor}"
        )
    for field in SOFTWARE_FLOOR_FIELDS:
        _version_floor(record[field], profile.software_floors[field], field=field)
    by_uuid, ordered = _admit_gpu_inventory(profile, record["gpus"])
    return {
        "admitted": True,
        "asserted_authority": authority,
        "attestation_sha256": canonical_digest(dict(record)),
        "authority_id": profile.authority_id,
        "availability_zone": profile.availability_zone,
        "gpu_index_by_uuid": dict(by_uuid),
        "gpu_uuids": list(ordered),
        "instance_id": instance_id,
        "instance_type": profile.instance_type,
        "profile_id": profile.profile_id,
        "profile_sha256": profile.sha256,
        "region": profile.region,
        "schema_version": 1,
    }


def admit_pair_allocations(
    *,
    profile: B200HardwareProfile,
    attestation: object,
    allocations: object,
) -> tuple[dict[str, Any], ...]:
    """Derive and admit exactly four disjoint two-GPU pair allocations."""

    by_uuid, _ = _admit_gpu_inventory(
        profile,
        _exact_mapping(
            attestation,
            _ATTESTATION_FIELDS,
            label="node attestation",
        )["gpus"],
    )
    if not isinstance(allocations, list) or len(allocations) != profile.pairs_per_node:
        raise HardwareAdmissionError(
            f"a qualified node must run exactly {profile.pairs_per_node} "
            "independent two-GPU pairs"
        )
    claimed: dict[str, str] = {}
    parsed = []
    for position, allocation in enumerate(allocations):
        record = _exact_mapping(
            allocation,
            _ALLOCATION_FIELDS,
            label=f"pair allocation {position}",
        )
        seed = _integer(
            record["seed"],
            field=f"pair allocation {position} seed",
            minimum=0,
            maximum=9,
        )
        pair_id = record["pair_id"]
        if pair_id != f"{profile.pair_id_prefix}{seed}":
            raise HardwareAdmissionError(
                f"pair allocation {position} pair_id {pair_id!r} is not the "
                f"canonical identity for seed {seed}"
            )
        arms = _exact_mapping(
            record["arms"],
            frozenset(_ARMS),
            label=f"pair allocation {position} arms",
        )
        derived: dict[str, dict[str, Any]] = {}
        for arm in _ARMS:
            arm_record = _exact_mapping(
                arms[arm],
                _ALLOCATION_ARM_FIELDS,
                label=(
                    f"pair allocation {position} {arm}; GPU ids must be derived "
                    "from the live allocation, never declared"
                ),
            )
            uuid = _text(
                arm_record["gpu_uuid"],
                field=f"pair allocation {position} {arm} gpu_uuid",
                pattern=_GPU_UUID_RE,
            )
            if uuid not in by_uuid:
                raise HardwareAdmissionError(
                    f"pair allocation {position} {arm} claims GPU {uuid} that "
                    "the node did not attest"
                )
            if uuid in claimed:
                raise HardwareAdmissionError(
                    f"GPU {uuid} is allocated to both {claimed[uuid]} and {pair_id}"
                )
            claimed[uuid] = pair_id
            derived[arm] = {"gpu_index": by_uuid[uuid], "gpu_uuid": uuid}
        indices = tuple(derived[arm]["gpu_index"] for arm in _ARMS)
        parsed.append(
            {
                "arms": derived,
                "cuda_visible_devices": ",".join(str(index) for index in indices),
                "gpu_indices": indices,
                "pair_id": pair_id,
                "seed": seed,
            }
        )
    seeds = [record["seed"] for record in parsed]
    if len(set(seeds)) != len(seeds):
        raise HardwareAdmissionError("a node cannot run one seed on two pairs")
    if len(claimed) != profile.gpu_count:
        raise HardwareAdmissionError(
            "pair allocations must cover every attested GPU exactly once"
        )
    parsed.sort(key=lambda record: record["seed"])
    return tuple(
        {**record, "pair_index": index} for index, record in enumerate(parsed)
    )


def _admit_gate(
    name: str,
    record: Mapping[str, Any],
    *,
    profile: B200HardwareProfile,
    attestation_sha256: str,
) -> None:
    if record["status"] != "passed":
        raise HardwareAdmissionError(
            f"qualification gate {name} status is {record['status']!r}, not 'passed'"
        )
    _sha256(record["evidence_sha256"], field=f"{name} evidence_sha256")
    if name == "device_query":
        _exact(record["devices_passed"], profile.gpu_count, field=f"{name} devices")
        _exact(record["result"], "PASS", field=f"{name} result")
    elif name == "nvidia_smi_attestation":
        _exact(record["gpu_count"], profile.gpu_count, field=f"{name} gpu_count")
        if _sha256(record["attestation_sha256"], field=f"{name} attestation") != (
            attestation_sha256
        ):
            raise HardwareAdmissionError(
                f"{name} is bound to a different node attestation"
            )
        _exact(record["ecc_uncorrected_errors"], 0, field=f"{name} ecc errors")
    elif name == "nccl_all_reduce":
        _exact(record["gpu_count"], profile.gpu_count, field=f"{name} gpu_count")
        _exact(record["errors"], 0, field=f"{name} errors")
        _flag(record["pair_all_reduce_passed"], field=f"{name} pairs", expected=True)
        _number(
            record["algorithm_bandwidth_gbps"],
            field=f"{name} bandwidth",
            minimum=1e-9,
            maximum=1e6,
        )
    elif name == "one_step_training":
        _exact(record["updates"], 1, field=f"{name} updates")
        _exact(record["arms"], list(_ARMS), field=f"{name} arms")
        _flag(record["oom_detected"], field=f"{name} oom", expected=False)
    elif name == "throughput_100_updates":
        _exact(record["updates"], 100, field=f"{name} updates")
        _flag(record["oom_detected"], field=f"{name} oom", expected=False)
        measured = _exact_mapping(
            record["tokens_per_second"],
            frozenset(_ARMS),
            label=f"{name} tokens_per_second",
        )
        for arm in _ARMS:
            _number(
                measured[arm],
                field=f"{name} {arm} tokens_per_second",
                minimum=1e-9,
                maximum=1e12,
            )
    elif name == "memory_headroom":
        _flag(record["oom_detected"], field=f"{name} oom", expected=False)
        total = _exact(
            record["device_total_mib"],
            profile.gpu_memory_mib,
            field=f"{name} device_total_mib",
        )
        peak = _integer(record["peak_allocated_mib"], field=f"{name} peak_allocated_mib")
        if peak >= total:
            raise HardwareAdmissionError(f"{name} left no measured device headroom")
        floor = profile.memory_headroom_fraction_floor
        if floor is None:
            raise HardwareAdmissionError(f"{name} floor is pending confirmation")
        fraction = _number(
            record["headroom_fraction"],
            field=f"{name} headroom_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        if fraction < floor:
            raise HardwareAdmissionError(
                f"{name} fraction {fraction} is below the frozen floor {floor}"
            )
        if fraction > 1.0 - peak / total + 1e-9:
            raise HardwareAdmissionError(
                f"{name} claims more headroom than the measurement supports"
            )
    elif name == "checkpoint_write":
        _sha256(record["checkpoint_sha256"], field=f"{name} checkpoint_sha256")
        _integer(record["bytes"], field=f"{name} bytes")
        _flag(record["fsync"], field=f"{name} fsync", expected=True)
    elif name == "exact_resume_equality":
        _exact(record["updates"], 2, field=f"{name} updates")
        _flag(record["resume_exact"], field=f"{name} resume_exact", expected=True)
        continuous = _sha256(
            record["continuous_state_sha256"],
            field=f"{name} continuous_state_sha256",
        )
        resumed = _sha256(
            record["resumed_state_sha256"],
            field=f"{name} resumed_state_sha256",
        )
        if continuous != resumed:
            raise HardwareAdmissionError(f"{name} resumed state is not bit-identical")
    else:  # pragma: no cover - the gate table and tuple are kept in sync
        raise HardwareAdmissionError(f"unknown qualification gate: {name}")


def admit_qualification_gates(
    *,
    profile: B200HardwareProfile,
    gates: object,
    attestation_sha256: str,
) -> dict[str, Any]:
    """Require every frozen site canary to be present, bound, and passing."""

    _sha256(attestation_sha256, field="attestation_sha256")
    if not isinstance(gates, Mapping):
        raise HardwareAdmissionError(  # noqa: TRY004
            "qualification gates must be a JSON object"
        )
    missing = sorted(set(QUALIFICATION_GATES) - set(gates))
    unknown = sorted(set(gates) - set(QUALIFICATION_GATES))
    if missing or unknown:
        raise HardwareAdmissionError(
            "qualification gates do not match the frozen set; "
            f"missing={missing}, unknown={unknown}"
        )
    for name in QUALIFICATION_GATES:
        record = _exact_mapping(
            gates[name],
            _GATE_FIELDS[name],
            label=f"qualification gate {name}",
        )
        _admit_gate(
            name,
            record,
            profile=profile,
            attestation_sha256=attestation_sha256,
        )
    return {
        "passed": list(QUALIFICATION_GATES),
        "profile_sha256": profile.sha256,
        "schema_version": 1,
    }


def admit_protected_site(
    *,
    profile: B200HardwareProfile,
    site_evidence: object,
    asserted_authority: object,
) -> dict[str, Any]:
    """Compose node, pair-geometry, and canary admission into one receipt."""

    authority = _require_authority(profile, asserted_authority)
    evidence = _exact_mapping(site_evidence, _SITE_FIELDS, label="site evidence")
    if evidence["schema_version"] != 1 or isinstance(evidence["schema_version"], bool):
        raise HardwareAdmissionError("site evidence schema_version must be exactly 1")
    if evidence["authority_id"] != authority:
        raise HardwareAdmissionError(
            f"site evidence authority {evidence['authority_id']!r} differs from "
            f"the asserted authority {authority!r}"
        )
    _exact(evidence["cohort_id"], profile.cohort_id, field="site evidence cohort_id")
    node = admit_b200_node(
        profile=profile,
        attestation=evidence["attestation"],
        asserted_authority=authority,
    )
    allocations = admit_pair_allocations(
        profile=profile,
        attestation=evidence["attestation"],
        allocations=evidence["pair_allocations"],
    )
    admit_qualification_gates(
        profile=profile,
        gates=evidence["qualification_gates"],
        attestation_sha256=node["attestation_sha256"],
    )
    return {
        "admitted": True,
        "attestation_sha256": node["attestation_sha256"],
        "authority_id": authority,
        "availability_zone": profile.availability_zone,
        "cohort_id": profile.cohort_id,
        "instance_id": node["instance_id"],
        "instance_type": profile.instance_type,
        "pair_allocations": [
            {
                "cuda_visible_devices": record["cuda_visible_devices"],
                "gpu_indices": list(record["gpu_indices"]),
                "pair_id": record["pair_id"],
                "pair_index": record["pair_index"],
                "seed": record["seed"],
            }
            for record in allocations
        ],
        "profile_id": profile.profile_id,
        "profile_sha256": profile.sha256,
        "qualification_gates": list(QUALIFICATION_GATES),
        "region": profile.region,
        "schema_version": 1,
        "site_evidence_sha256": canonical_digest(dict(evidence)),
    }


def hardware_profile_report(
    path: Path | str = B200_PROFILE_PATH,
) -> dict[str, Any]:
    """Summarize one hardware profile without contacting AWS."""

    profile = load_hardware_profile(path)
    return {
        "authority_id": profile.authority_id,
        "gpu_count": profile.gpu_count,
        "instance_type": profile.instance_type,
        "launch_ready": profile.launch_ready,
        "pairs_per_node": profile.pairs_per_node,
        "pending_confirmation": list(profile.pending_confirmation),
        "profile_id": profile.profile_id,
        "profile_sha256": profile.sha256,
        "protected_launch_admissible": profile.launch_ready,
        "region": profile.region,
        "schema_version": 1,
    }


def _runner(command: Sequence[str]):
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )


def _json_command(
    command: list[str],
    *,
    runner: Callable[[Sequence[str]], object],
) -> Any:
    result = runner(command)
    returncode = int(result.returncode)
    stdout = str(getattr(result, "stdout", "") or "")
    stderr = str(getattr(result, "stderr", "") or "")
    if returncode:
        raise RuntimeError(
            f"AWS inspection command failed ({returncode}): {stderr.strip()}"
        )
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("AWS inspection returned invalid JSON") from error


def inspect_aws_readiness(
    *,
    region: str,
    runner: Callable[[Sequence[str]], object] = _runner,
    hardware_profile_path: Path | str = B200_PROFILE_PATH,
) -> dict[str, Any]:
    """Inspect account reachability; this never creates or changes resources."""

    if not isinstance(region, str) or not _REGION_RE.fullmatch(region):
        raise ValueError("region must be a canonical AWS region name")
    identity = _json_command(
        ["aws", "sts", "get-caller-identity", "--output", "json"],
        runner=runner,
    )
    offerings = _json_command(
        [
            "aws",
            "ec2",
            "describe-instance-type-offerings",
            "--location-type",
            "availability-zone",
            "--filters",
            f"Name=instance-type,Values={','.join(INSTANCE_TYPES)}",
            "--region",
            region,
            "--output",
            "json",
        ],
        runner=runner,
    )
    quotas = _json_command(
        [
            "aws",
            "service-quotas",
            "list-service-quotas",
            "--service-code",
            "ec2",
            "--region",
            region,
            "--query",
            "Quotas[?contains(QuotaName, `Running On-Demand P`)]",
            "--output",
            "json",
        ],
        runner=runner,
    )
    if not isinstance(identity, dict) or not {
        "Account",
        "Arn",
        "UserId",
    } <= set(identity):
        raise RuntimeError("AWS identity response is malformed")
    values = offerings.get("InstanceTypeOfferings") if isinstance(offerings, dict) else None
    if not isinstance(values, list) or not isinstance(quotas, list):
        raise RuntimeError(  # noqa: TRY004
            "AWS offering or quota response is malformed"
        )
    by_type = {instance_type: [] for instance_type in INSTANCE_TYPES}
    for value in values:
        if (
            isinstance(value, dict)
            and value.get("InstanceType") in by_type
            and isinstance(value.get("Location"), str)
        ):
            by_type[value["InstanceType"]].append(value["Location"])
    by_type = {
        key: sorted(set(value))
        for key, value in by_type.items()
    }
    quota_values = [
        {
            "adjustable": value.get("Adjustable"),
            "name": value.get("QuotaName"),
            "quota_code": value.get("QuotaCode"),
            "unit": value.get("Unit"),
            "value": value.get("Value"),
        }
        for value in quotas
        if isinstance(value, dict)
    ]
    exact_profile = hardware_profile_report(hardware_profile_path)
    offered = by_type.get(exact_profile["instance_type"], [])
    exact_profile.update(
        {
            "availability_zones_offering_instance_type": offered,
            "protected_launch_admissible": (
                exact_profile["launch_ready"]
                and region == exact_profile["region"]
                and bool(offered)
            ),
            "region_matches_profile": region == exact_profile["region"],
        }
    )
    return {
        "account": identity["Account"],
        "caller_arn": identity["Arn"],
        "exact_hardware_profile": exact_profile,
        "instance_type_availability_zones": by_type,
        "note": (
            "Offerings and quotas do not guarantee immediate capacity; "
            "an offered instance type is not an admitted node. Protected work "
            "additionally requires exact-profile admission and the measured "
            "Slurm GPU canaries."
        ),
        "on_demand_p_quotas": quota_values,
        "region": region,
        "schema_version": 1,
    }
