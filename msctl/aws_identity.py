"""Pinned AWS instance-identity verification shared by launch gates."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .jsonutil import canonical_json


_AWS_US_EAST_1_DSA_CERTIFICATE = """-----BEGIN CERTIFICATE-----
MIIC7TCCAq0CCQCWukjZ5V4aZzAJBgcqhkjOOAQDMFwxCzAJBgNVBAYTAlVTMRkw
FwYDVQQIExBXYXNoaW5ndG9uIFN0YXRlMRAwDgYDVQQHEwdTZWF0dGxlMSAwHgYD
VQQKExdBbWF6b24gV2ViIFNlcnZpY2VzIExMQzAeFw0xMjAxMDUxMjU2MTJaFw0z
ODAxMDUxMjU2MTJaMFwxCzAJBgNVBAYTAlVTMRkwFwYDVQQIExBXYXNoaW5ndG9u
IFN0YXRlMRAwDgYDVQQHEwdTZWF0dGxlMSAwHgYDVQQKExdBbWF6b24gV2ViIFNl
cnZpY2VzIExMQzCCAbcwggEsBgcqhkjOOAQBMIIBHwKBgQCjkvcS2bb1VQ4yt/5e
ih5OO6kK/n1Lzllr7D8ZwtQP8fOEpp5E2ng+D6Ud1Z1gYipr58Kj3nssSNpI6bX3
VyIQzK7wLclnd/YozqNNmgIyZecN7EglK9ITHJLP+x8FtUpt3QbyYXJdmVMegN6P
hviYt5JH/nYl4hh3Pa1HJdskgQIVALVJ3ER11+Ko4tP6nwvHwh6+ERYRAoGBAI1j
k+tkqMVHuAFcvAGKocTgsjJem6/5qomzJuKDmbJNu9Qxw3rAotXau8Qe+MBcJl/U
hhy1KHVpCGl9fueQ2s6IL0CaO/buycU1CiYQk40KNHCcHfNiZbdlx1E9rpUp7bnF
lRa2v1ntMX3caRVDdbtPEWmdxSCYsYFDk4mZrOLBA4GEAAKBgEbmeve5f8LIE/Gf
MNmP9CM5eovQOGx5ho8WqD+aTebs+k2tn92BBPqeZqpWRa5P/+jrdKml1qx4llHW
MXrs3IgIb6+hUIB+S8dz8/mmO0bpr76RoZVCXYab2CZedFut7qc3WUH9+EUAH5mw
vSeDCOUMYQR7R9LINYwouHIziqQYMAkGByqGSM44BAMDLwAwLAIUWXBlk40xTwSw
7HX32MxXYruse9ACFBNGmdX2ZBrVNGrN9N2f6ROk0k9K
-----END CERTIFICATE-----
"""

# AWS publishes this DSA trust anchor for both supported Regions. Keep the
# allowlist explicit so adding a Region requires an intentional trust decision.
AWS_INSTANCE_IDENTITY_CERTIFICATES = {
    "us-east-1": _AWS_US_EAST_1_DSA_CERTIFICATE,
    "us-west-2": _AWS_US_EAST_1_DSA_CERTIFICATE,
}

_ENVIRONMENT_FIELDS = {
    "schema_version",
    "profile_sha256",
    "container_image_digest",
    "boot_id",
    "aws_instance_identity_document",
    "aws_instance_identity_pkcs7",
}


class AwsIdentityError(ValueError):
    """AWS identity evidence is malformed, unverifiable, or cross-runtime."""


@dataclass(frozen=True)
class VerifiedEnvironmentIdentity:
    instance_id: str
    boot_id: str
    region: str
    ami_id: str
    account_id: str
    receipt_sha256: str
    receipt: Mapping[str, object]


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise AwsIdentityError("AWS identity document repeats a JSON field")
        value[key] = item
    return value


def verify_instance_identity_pkcs7(
    identity: Mapping[str, object],
    pkcs7: str,
    region: str,
) -> bool:
    """Verify that the pinned AWS Region anchor signed exactly ``identity``."""

    certificate_text = AWS_INSTANCE_IDENTITY_CERTIFICATES.get(region)
    if certificate_text is None:
        raise AwsIdentityError(
            "no pinned AWS PKCS7 trust anchor exists for the selected region"
        )
    if not isinstance(pkcs7, str):
        return False
    compact = "".join(pkcs7.split())
    try:
        decoded = base64.b64decode(compact, validate=True)
    except (TypeError, ValueError):
        return False
    if not decoded:
        return False
    wrapped = (
        "-----BEGIN PKCS7-----\n"
        + "\n".join(
            compact[index : index + 64]
            for index in range(0, len(compact), 64)
        )
        + "\n-----END PKCS7-----\n"
    )
    try:
        with tempfile.TemporaryDirectory(prefix="msctl-iid-") as directory:
            root = Path(directory)
            signature = root / "identity.pkcs7"
            certificate = root / "aws-dsa.pem"
            signature.write_text(wrapped, encoding="ascii")
            certificate.write_text(certificate_text, encoding="ascii")
            completed = subprocess.run(
                [
                    "/usr/bin/openssl",
                    "smime",
                    "-verify",
                    "-in",
                    str(signature),
                    "-inform",
                    "PEM",
                    "-certfile",
                    str(certificate),
                    "-nointern",
                    "-noverify",
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                shell=False,
                env={},
                timeout=10,
            )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if completed.returncode != 0:
        return False
    try:
        signed = json.loads(
            completed.stdout.decode("utf-8"),
            object_pairs_hook=_strict_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, AwsIdentityError):
        return False
    return signed == dict(identity)


def verify_environment_receipt(
    data: bytes,
    *,
    expected_profile_sha256: str,
    expected_container_digest: str,
    expected_region: str,
    expected_ami_id: str,
    expected_account_id: str,
    expected_instance_id: str | None = None,
) -> VerifiedEnvironmentIdentity:
    """Verify one canonical v3 receipt and its AWS PKCS7 identity signature."""

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                AwsIdentityError(
                    f"environment receipt contains non-finite {constant}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AwsIdentityError(
            "environment receipt must contain valid UTF-8 JSON"
        ) from error
    if not isinstance(value, dict) or set(value) != _ENVIRONMENT_FIELDS:
        raise AwsIdentityError(
            "environment receipt fields do not match the v3 contract"
        )
    if data != canonical_json(value) + b"\n":
        raise AwsIdentityError(
            "environment receipt must be canonical JSON plus one newline"
        )
    identity = value["aws_instance_identity_document"]
    if not isinstance(identity, dict):
        raise AwsIdentityError(
            "environment receipt identity document must be an object"
        )
    instance_id = identity.get("instanceId")
    boot_id = value["boot_id"]
    pkcs7 = value["aws_instance_identity_pkcs7"]
    if (
        value["schema_version"] != 3
        or value["profile_sha256"] != expected_profile_sha256
        or value["container_image_digest"] != expected_container_digest
        or identity.get("imageId") != expected_ami_id
        or identity.get("region") != expected_region
        or identity.get("accountId") != expected_account_id
        or identity.get("instanceType") is None
        or not isinstance(instance_id, str)
        or re.fullmatch(r"i-[0-9a-f]{8,17}", instance_id) is None
        or (
            expected_instance_id is not None
            and instance_id != expected_instance_id
        )
        or not isinstance(boot_id, str)
        or re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
            r"[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            boot_id,
        )
        is None
        or not isinstance(pkcs7, str)
        or not verify_instance_identity_pkcs7(identity, pkcs7, expected_region)
    ):
        raise AwsIdentityError(
            "environment receipt is incomplete, unsigned, or cross-runtime"
        )
    return VerifiedEnvironmentIdentity(
        instance_id=instance_id,
        boot_id=boot_id,
        region=expected_region,
        ami_id=expected_ami_id,
        account_id=expected_account_id,
        receipt_sha256=hashlib.sha256(data).hexdigest(),
        receipt=value,
    )
