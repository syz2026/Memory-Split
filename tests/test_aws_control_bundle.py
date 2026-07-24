from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

import pytest

from msctl.aws_argv import ARGV_DOCUMENT_CONTENT
from msctl.aws_control_bundle import (
    CONTROL_BUNDLE_MEMBERS,
    CONTROL_MANIFEST,
    CONTROL_SUMS,
    build_control_bundle_bytes,
    render_control_install_command,
    verify_control_bundle,
    write_control_bundle,
)
from msctl.errors import MsctlError


ROOT = Path(__file__).resolve().parents[1]


def test_control_bundle_is_deterministic_exact_and_exclusively_published(tmp_path):
    first = build_control_bundle_bytes(ROOT)
    second = build_control_bundle_bytes(ROOT)
    assert first.payload == second.payload
    assert first.sha256 == hashlib.sha256(first.payload).hexdigest()
    assert set(first.members) == set(CONTROL_BUNDLE_MEMBERS)
    assert "msctl/fsutil.py" in first.members

    destination = tmp_path / "control.tar"
    write_control_bundle(destination, first)
    verified = verify_control_bundle(
        destination,
        expected_sha256=first.sha256,
    )
    assert verified.payload == first.payload
    with tarfile.open(destination, "r:") as archive:
        assert archive.getnames() == sorted(
            [*CONTROL_BUNDLE_MEMBERS, CONTROL_MANIFEST, CONTROL_SUMS]
        )

    with pytest.raises(MsctlError, match="replace"):
        write_control_bundle(destination, first)
    with pytest.raises(MsctlError, match="SHA-256"):
        verify_control_bundle(destination, expected_sha256="0" * 64)


def test_stock_ssm_installer_is_hash_bound_idempotent_and_precedes_custom_doc():
    digest = "a" * 64
    command = render_control_install_command(
        bundle_sha256=digest,
        bundle_uri=f"s3://memorysplit-prod/control/{digest}.tar",
        region="us-east-1",
    )

    assert f"/opt/memorysplit/control/{digest}" in command
    assert f"control-{digest}.tar" in command
    assert "/usr/bin/sha256sum -c -" in command
    assert "CONTROL-SHA256SUMS" in command
    assert (
        'CONTROL-BUNDLE.json | /usr/bin/cmp - "$R/CONTROL-BUNDLE.json"'
        in command
    )
    assert (
        'CONTROL-SHA256SUMS | /usr/bin/cmp - "$R/CONTROL-SHA256SUMS"'
        in command
    )
    assert "-perm /222" in command
    assert "/usr/bin/mv -T -n" in command
    assert "AWS-RunShellScript" not in command
    assert '"BootstrapMode"' in ARGV_DOCUMENT_CONTENT
    assert "verified-control-bundle" in ARGV_DOCUMENT_CONTENT
    assert "/opt/memorysplit/control/{{ ControlBundleSHA256 }}" in (
        ARGV_DOCUMENT_CONTENT
    )
    assert "/usr/bin/cmp -" in ARGV_DOCUMENT_CONTENT
    assert "-perm /222" in ARGV_DOCUMENT_CONTENT

    with pytest.raises(MsctlError):
        render_control_install_command(
            bundle_sha256=digest,
            bundle_uri="s3://memorysplit-prod/control/../escape.tar",
            region="us-east-1",
        )
