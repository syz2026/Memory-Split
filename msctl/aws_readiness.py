"""Closed protected-launch readiness receipts for AWS GPU v3."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

from cluster.aws.p5.canary import (
    load_qualification_receipt,
    render_canary_command_plan,
)
from cluster.aws.p5.profile import AwsGpuRuntime

from .aws_sealed_evaluation import (
    SealedEvaluationRelease,
    load_sealed_evaluation_release,
)
from .aws_selection import HardwareAmendment, ProviderSelection
from .errors import MsctlError
from .fsutil import open_directory, rename_noreplace_at
from .jsonutil import canonical_json, canonical_sha256, require_sha256


READINESS_RECEIPT_TYPE = "memorysplit-protected-launch-readiness-v3"
DIAGNOSTIC_RECEIPT_TYPE = "memorysplit-29m-diagnostic-v3"
DIAGNOSTIC_IDS = (
    "full_corpus_dense",
    "full_corpus_split90",
    "no_arc_conceptarc_dense",
    "no_arc_conceptarc_split90",
    "no_refinement_dense",
    "no_refinement_split90",
)
_ROOT_FIELDS = {
    "schema_version",
    "receipt_type",
    "profile_id",
    "bindings",
    "capacity",
    "decision",
}
_BINDING_FIELDS = {
    "hardware_amendment_sha256",
    "provider_selection_sha256",
    "environment_receipt_sha256",
    "qualification_receipt_sha256",
    "diagnostic_receipt_sha256",
    "sealed_evaluation_release_sha256",
    "study_lock_sha256",
    "profile_sha256",
    "release_sha256",
    "cohort_assignment_sha256",
    "preregistration_sha256",
}
_CAPACITY_FIELDS = {
    "purchase_model",
    "capacity_reservation_id",
    "capacity_block_offering_id",
}
_DECISION_FIELDS = {
    "protected_launch_allowed",
    "reviewer",
    "reviewed_at",
}
_DIAGNOSTIC_FIELDS = {
    "schema_version",
    "receipt_type",
    "diagnostic_id",
    "model_parameters",
    "targets_per_update",
    "optimizer_steps",
    "raw_target_tokens",
    "artifact_sha256",
    "passed",
    "reviewer",
    "completed_at",
}
_MAX_JSON_BYTES = 4 * 1024 * 1024
_REVIEWER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._@+-]{1,127}$")


@dataclass(frozen=True)
class LaunchReadiness:
    profile_id: str
    sha256: str
    bindings: Mapping[str, object]
    decision: Mapping[str, object]
    path: Path | None
    value: Mapping[str, object]


def _fail(message: str, *, code: str = "LAUNCH_READINESS_INVALID") -> None:
    raise MsctlError(code, message)


def _exact(value: object, fields: set[str], *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        _fail(f"{label} must be an object")
    actual = set(value)
    if actual != fields:
        _fail(
            f"{label} fields are not closed; "
            f"missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )
    return dict(value)


def _timestamp(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z") or len(value) > 40:
        _fail(f"{label} must be an RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise MsctlError(
            "LAUNCH_READINESS_INVALID",
            f"{label} must be an RFC 3339 UTC timestamp",
        ) from error
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != UTC.utcoffset(parsed)
        or parsed.year < 2020
    ):
        _fail(f"{label} must be an RFC 3339 UTC timestamp")
    return value


def _reviewer(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _REVIEWER_RE.fullmatch(value) is None:
        _fail(f"{label} is invalid")
    return value


def _read_regular(path: Path, *, label: str) -> bytes:
    try:
        before = path.stat(follow_symlinks=False)
        if (
            path.is_symlink()
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_JSON_BYTES
        ):
            _fail(f"{label} must be one bounded singly linked regular file")
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise MsctlError(
            "LAUNCH_READINESS_INVALID",
            f"{label} cannot be read safely",
        ) from error
    data = b"".join(chunks)
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
    ) or len(data) != after.st_size:
        _fail(f"{label} changed while being read")
    return data


def _decode(data: bytes, *, label: str) -> dict[str, object]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                _fail(f"{label} repeats field {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda constant: _fail(
                f"{label} contains non-finite {constant}"
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MsctlError(
            "LAUNCH_READINESS_INVALID",
            f"{label} must contain valid UTF-8 JSON",
        ) from error
    if not isinstance(value, dict):
        _fail(f"{label} must contain one JSON object")
    return value


def _file_sha256(path: Path, *, label: str) -> str:
    return hashlib.sha256(_read_regular(path, label=label)).hexdigest()


def _environment_identity(
    path: Path,
    *,
    profile: object,
    selection: ProviderSelection,
) -> tuple[str, str, str]:
    data = _read_regular(path, label="environment receipt")
    value = _exact(
        _decode(data, label="environment receipt"),
        {
            "schema_version",
            "profile_sha256",
            "container_image_digest",
            "boot_id",
            "aws_instance_identity_document",
            "aws_instance_identity_pkcs7",
        },
        label="environment receipt",
    )
    identity = value["aws_instance_identity_document"]
    if not isinstance(identity, dict):
        _fail("environment receipt identity document must be an object")
    instance_id = identity.get("instanceId")
    boot_id = value["boot_id"]
    if (
        value["schema_version"] != 3
        or value["profile_sha256"] != getattr(profile, "sha256", None)
        or value["container_image_digest"] != selection.container_digest
        or identity.get("imageId") != selection.ami_id
        or identity.get("region") != selection.region
        or identity.get("accountId") != selection.aws_account_id
        or not isinstance(instance_id, str)
        or re.fullmatch(r"^i-[0-9a-f]{8,17}$", instance_id) is None
        or not isinstance(boot_id, str)
        or re.fullmatch(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
            r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            boot_id,
        )
        is None
        or not isinstance(value["aws_instance_identity_pkcs7"], str)
        or not value["aws_instance_identity_pkcs7"]
        or data != canonical_json(value) + b"\n"
    ):
        _fail("environment receipt is incomplete or cross-runtime")
    return instance_id, str(boot_id), hashlib.sha256(data).hexdigest()


def _expected_command_plan_sha256(
    *,
    profile: object,
    selection: ProviderSelection,
    release: object,
) -> str:
    release_sha256 = getattr(release, "archive_sha256", None)
    require_sha256(release_sha256, label="readiness release")
    runtime = AwsGpuRuntime(
        region=selection.region,
        s3_root="s3://readiness.invalid/immutable",
        ami_id=selection.ami_id,
        container_image=selection.container_image,
        container_digest=selection.container_digest,
        uid=10001,
        gid=10001,
    )
    scratch = getattr(profile, "scratch_root", "/mnt/memorysplit")
    return canonical_sha256(
        render_canary_command_plan(
            profile,
            runtime,
            release_sha256=release_sha256,
            release_root=f"{scratch}/releases/{release_sha256}",
        )
    )


def load_diagnostic_receipt(path: Path | str, *, diagnostic_id: str) -> str:
    candidate = Path(path)
    data = _read_regular(candidate, label=f"diagnostic {diagnostic_id}")
    value = _decode(data, label=f"diagnostic {diagnostic_id}")
    receipt = _exact(value, _DIAGNOSTIC_FIELDS, label="diagnostic receipt")
    if (
        diagnostic_id not in DIAGNOSTIC_IDS
        or receipt["schema_version"] != 3
        or receipt["receipt_type"] != DIAGNOSTIC_RECEIPT_TYPE
        or receipt["diagnostic_id"] != diagnostic_id
        or receipt["model_parameters"] != 28_969_216
        or receipt["targets_per_update"] != 524_288
        or receipt["optimizer_steps"] != 1_106
        or receipt["raw_target_tokens"] != 579_862_528
        or receipt["passed"] is not True
    ):
        _fail(f"diagnostic {diagnostic_id} is incomplete or stale")
    require_sha256(
        receipt["artifact_sha256"],
        label=f"diagnostic {diagnostic_id} artifact",
    )
    _reviewer(receipt["reviewer"], label="diagnostic reviewer")
    _timestamp(receipt["completed_at"], label="diagnostic completed_at")
    if data != canonical_json(receipt) + b"\n":
        _fail(f"diagnostic {diagnostic_id} is not canonical JSON")
    return hashlib.sha256(data).hexdigest()


def _capacity(selection: ProviderSelection, profile: object) -> dict[str, object]:
    return {
        "purchase_model": getattr(profile, "purchase_model", None),
        "capacity_reservation_id": getattr(
            selection, "capacity_reservation_id", None
        ),
        "capacity_block_offering_id": getattr(
            selection, "capacity_block_offering_id", None
        ),
    }


def _validate_capacity(
    value: Mapping[str, object],
    *,
    selection: ProviderSelection,
    profile: object,
) -> None:
    expected = _capacity(selection, profile)
    if dict(value) != expected:
        _fail("launch readiness capacity identity is stale or cross-profile")
    model = getattr(profile, "purchase_model", None)
    reservation = value["capacity_reservation_id"]
    offering = value["capacity_block_offering_id"]
    if model == "on_demand":
        if reservation is not None or offering is not None:
            _fail("on-demand readiness must not claim Capacity Block identity")
    elif model == "capacity_block":
        if not all(isinstance(item, str) and item for item in (reservation, offering)):
            _fail("Capacity Block readiness requires reservation and offering IDs")
    else:
        _fail("launch readiness purchase model is unsupported")


def create_launch_readiness(
    *,
    profile: object,
    amendment: HardwareAmendment,
    selection: ProviderSelection,
    release: object,
    environment_receipt: Path | str,
    qualification_receipt: Path | str,
    diagnostic_receipts: Mapping[str, Path | str],
    sealed_evaluation_release: Path | str,
    reviewer: str,
    reviewed_at: str,
) -> dict[str, object]:
    """Create a decision only after validating every bound artifact."""

    if set(diagnostic_receipts) != set(DIAGNOSTIC_IDS):
        _fail("readiness requires exactly all six named 29M diagnostics")
    environment_path = Path(environment_receipt)
    qualification_path = Path(qualification_receipt)
    instance_id, boot_id, environment_sha256 = _environment_identity(
        environment_path,
        profile=profile,
        selection=selection,
    )
    qualification_sha256 = _file_sha256(qualification_path, label="qualification receipt")
    try:
        report = load_qualification_receipt(
            qualification_path,
            profile,
            expected_instance_id=instance_id,
            expected_boot_id=boot_id,
            expected_provider_selection_sha256=selection.sha256,
            expected_environment_receipt_sha256=environment_sha256,
            expected_region=selection.region,
            expected_ami_id=selection.ami_id,
            expected_ami_owner_id=getattr(selection, "ami_owner_id", None),
            expected_release_sha256=getattr(release, "archive_sha256", None),
            expected_container_image=selection.container_image,
            expected_container_digest=selection.container_digest,
            expected_command_plan_sha256=_expected_command_plan_sha256(
                profile=profile,
                selection=selection,
                release=release,
            ),
        )
    except (TypeError, ValueError) as error:
        raise MsctlError(
            "LAUNCH_READINESS_INVALID",
            "qualification receipt is incomplete or cross-profile",
        ) from error
    if report.provenance is None:
        _fail("protected launch requires a provenance-complete v3 qualification")
    diagnostic_hashes = {
        diagnostic_id: load_diagnostic_receipt(
            diagnostic_receipts[diagnostic_id],
            diagnostic_id=diagnostic_id,
        )
        for diagnostic_id in DIAGNOSTIC_IDS
    }
    # The evaluator's study lock has its own frozen preregistration.  The
    # training preregistration and the actual study-lock bytes are independent
    # readiness bindings and must not be forced to share a digest.
    sealed = load_sealed_evaluation_release(sealed_evaluation_release)
    profile_id = getattr(profile, "profile_id", None)
    profile_sha256 = getattr(profile, "sha256", None)
    release_sha256 = getattr(release, "archive_sha256", None)
    if (
        profile_id != selection.selected_profile_id
        or getattr(profile, "provider", None) != selection.provider
        or profile_sha256 != selection.profile_sha256
        or amendment.sha256 != selection.amendment_sha256
    ):
        _fail("readiness inputs are cross-profile or stale")
    for label, digest in (
        ("profile", profile_sha256),
        ("release", release_sha256),
    ):
        require_sha256(digest, label=f"readiness {label}")
    return {
        "schema_version": 3,
        "receipt_type": READINESS_RECEIPT_TYPE,
        "profile_id": profile_id,
        "bindings": {
            "hardware_amendment_sha256": amendment.sha256,
            "provider_selection_sha256": selection.sha256,
            "environment_receipt_sha256": environment_sha256,
            "qualification_receipt_sha256": qualification_sha256,
            "diagnostic_receipt_sha256": diagnostic_hashes,
            "sealed_evaluation_release_sha256": sealed.sha256,
            "study_lock_sha256": sealed.study_lock_sha256,
            "profile_sha256": profile_sha256,
            "release_sha256": release_sha256,
            "cohort_assignment_sha256": amendment.cohort_assignment_sha256,
            "preregistration_sha256": amendment.preregistration_sha256,
        },
        "capacity": _capacity(selection, profile),
        "decision": {
            "protected_launch_allowed": True,
            "reviewer": _reviewer(reviewer, label="readiness reviewer"),
            "reviewed_at": _timestamp(
                reviewed_at,
                label="readiness reviewed_at",
            ),
        },
    }


def validate_launch_readiness(
    value: object,
    *,
    profile: object,
    amendment: HardwareAmendment,
    selection: ProviderSelection,
    release: object,
    manifest: object | None = None,
    environment_receipt: Path | str | None = None,
    qualification_receipt: Path | str | None = None,
    diagnostic_receipts: Mapping[str, Path | str] | None = None,
    sealed_evaluation_release: Path | str | None = None,
    expected_instance_id: str | None = None,
    path: Path | None = None,
    sha256: str | None = None,
) -> LaunchReadiness:
    receipt = _exact(value, _ROOT_FIELDS, label="launch readiness")
    bindings = _exact(
        receipt["bindings"],
        _BINDING_FIELDS,
        label="launch readiness bindings",
    )
    capacity = _exact(
        receipt["capacity"],
        _CAPACITY_FIELDS,
        label="launch readiness capacity",
    )
    decision = _exact(
        receipt["decision"],
        _DECISION_FIELDS,
        label="launch readiness decision",
    )
    diagnostics = bindings["diagnostic_receipt_sha256"]
    if not isinstance(diagnostics, dict) or set(diagnostics) != set(DIAGNOSTIC_IDS):
        _fail("readiness diagnostic hash set is not exact")
    for name in _BINDING_FIELDS - {"diagnostic_receipt_sha256"}:
        digest = bindings[name]
        require_sha256(digest, label=f"readiness {name}")
    for name in DIAGNOSTIC_IDS:
        require_sha256(diagnostics[name], label=f"readiness diagnostic {name}")
    profile_id = getattr(profile, "profile_id", None)
    expected = {
        "hardware_amendment_sha256": amendment.sha256,
        "provider_selection_sha256": selection.sha256,
        "profile_sha256": getattr(profile, "sha256", None),
        "release_sha256": getattr(release, "archive_sha256", None),
        "cohort_assignment_sha256": amendment.cohort_assignment_sha256,
        "preregistration_sha256": amendment.preregistration_sha256,
    }
    if (
        receipt["schema_version"] != 3
        or receipt["receipt_type"] != READINESS_RECEIPT_TYPE
        or receipt["profile_id"] != profile_id
        or any(bindings[field] != digest for field, digest in expected.items())
        or decision["protected_launch_allowed"] is not True
        or selection.selected_profile_id != profile_id
    ):
        _fail("launch readiness is false, stale, or cross-profile")
    _validate_capacity(capacity, selection=selection, profile=profile)
    _reviewer(decision["reviewer"], label="readiness reviewer")
    _timestamp(decision["reviewed_at"], label="readiness reviewed_at")
    if manifest is not None and (
        getattr(manifest, "profile_sha256", None) != bindings["profile_sha256"]
        or getattr(manifest, "release_sha256", None) != bindings["release_sha256"]
        or getattr(manifest, "cohort_assignment_sha256", None)
        != bindings["cohort_assignment_sha256"]
        or getattr(manifest, "preregistration_sha256", None)
        != bindings["preregistration_sha256"]
        or getattr(manifest, "hardware_amendment_sha256", None)
        != bindings["hardware_amendment_sha256"]
        or getattr(manifest, "provider_selection_sha256", None)
        != bindings["provider_selection_sha256"]
        or getattr(manifest, "sealed_evaluation_sha256", None)
        != bindings["sealed_evaluation_release_sha256"]
        or getattr(manifest, "study_lock_sha256", None)
        != bindings["study_lock_sha256"]
    ):
        _fail("launch readiness does not bind this run manifest")
    environment_instance_id = None
    environment_boot_id = None
    if environment_receipt is not None:
        (
            environment_instance_id,
            environment_boot_id,
            environment_digest,
        ) = _environment_identity(
            Path(environment_receipt),
            profile=profile,
            selection=selection,
        )
        if environment_digest != bindings["environment_receipt_sha256"]:
            _fail("launch readiness environment receipt binding is stale")
    if qualification_receipt is not None and _file_sha256(
        Path(qualification_receipt),
        label="qualification receipt",
    ) != bindings["qualification_receipt_sha256"]:
        _fail("launch readiness qualification receipt binding is stale")
    if (
        expected_instance_id is not None
        and environment_instance_id is not None
        and environment_instance_id != expected_instance_id
    ):
        _fail("launch readiness environment is bound to another instance")
    if qualification_receipt is not None:
        try:
            report = load_qualification_receipt(
                qualification_receipt,
                profile,
                expected_instance_id=(
                    expected_instance_id or environment_instance_id
                ),
                expected_boot_id=environment_boot_id,
                expected_provider_selection_sha256=selection.sha256,
                expected_environment_receipt_sha256=str(
                    bindings["environment_receipt_sha256"]
                ),
                expected_region=selection.region,
                expected_ami_id=selection.ami_id,
                expected_ami_owner_id=getattr(selection, "ami_owner_id", None),
                expected_release_sha256=getattr(
                    release,
                    "archive_sha256",
                    None,
                ),
                expected_container_image=selection.container_image,
                expected_container_digest=selection.container_digest,
                expected_command_plan_sha256=_expected_command_plan_sha256(
                    profile=profile,
                    selection=selection,
                    release=release,
                ),
            )
        except (TypeError, ValueError) as error:
            raise MsctlError(
                "LAUNCH_READINESS_INVALID",
                "qualification receipt is incomplete or cross-profile",
            ) from error
        if report.provenance is None:
            _fail("protected launch requires a provenance-complete v3 qualification")
    if diagnostic_receipts is not None:
        if set(diagnostic_receipts) != set(DIAGNOSTIC_IDS):
            _fail("readiness diagnostic receipt set is not exact")
        for diagnostic_id in DIAGNOSTIC_IDS:
            digest = load_diagnostic_receipt(
                diagnostic_receipts[diagnostic_id],
                diagnostic_id=diagnostic_id,
            )
            if digest != diagnostics[diagnostic_id]:
                _fail(
                    f"launch readiness diagnostic {diagnostic_id} binding is stale"
                )
    if sealed_evaluation_release is not None:
        load_sealed_evaluation_release(
            sealed_evaluation_release,
            expected_release_sha256=str(
                bindings["sealed_evaluation_release_sha256"]
            ),
            expected_study_lock_sha256=str(bindings["study_lock_sha256"]),
        )
    digest = sha256 or hashlib.sha256(canonical_json(receipt) + b"\n").hexdigest()
    require_sha256(digest, label="launch readiness receipt")
    return LaunchReadiness(
        profile_id=str(profile_id),
        sha256=digest,
        bindings=bindings,
        decision=decision,
        path=path,
        value=receipt,
    )


def load_launch_readiness(
    path: Path | str,
    **kwargs: object,
) -> LaunchReadiness:
    candidate = Path(path)
    data = _read_regular(candidate, label="launch readiness receipt")
    value = _decode(data, label="launch readiness receipt")
    if data != canonical_json(value) + b"\n":
        _fail("launch readiness must use canonical JSON plus one newline")
    return validate_launch_readiness(
        value,
        path=candidate,
        sha256=hashlib.sha256(data).hexdigest(),
        **kwargs,
    )


def write_launch_readiness(path: Path | str, value: Mapping[str, object]) -> Path:
    destination = Path(path)
    directory_fd = open_directory(
        destination.parent,
        label="launch readiness output",
        create=True,
    )
    temporary = f".{destination.name}.{secrets.token_hex(12)}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        payload = canonical_json(dict(value)) + b"\n"
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            rename_noreplace_at(
                directory_fd,
                temporary,
                directory_fd,
                destination.name,
            )
        except FileExistsError as error:
            raise MsctlError(
                "LAUNCH_READINESS_EXISTS",
                "refusing to replace an existing launch readiness receipt",
            ) from error
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)
    return destination


def plan_launch_readiness(
    *,
    out: Path | str,
    apply: bool,
    **kwargs: object,
) -> dict[str, object]:
    receipt = create_launch_readiness(**kwargs)
    # Revalidate the in-memory decision before exposing or publishing it.
    validated = validate_launch_readiness(
        receipt,
        profile=kwargs["profile"],
        amendment=kwargs["amendment"],
        selection=kwargs["selection"],
        release=kwargs["release"],
        environment_receipt=kwargs["environment_receipt"],
        qualification_receipt=kwargs["qualification_receipt"],
        diagnostic_receipts=kwargs["diagnostic_receipts"],
        sealed_evaluation_release=kwargs["sealed_evaluation_release"],
    )
    result = {
        "receipt": receipt,
        "receipt_sha256": validated.sha256,
        "out": str(Path(out)),
        "published": False,
    }
    if apply:
        write_launch_readiness(out, receipt)
        result["published"] = True
    return result
