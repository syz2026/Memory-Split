from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

import cluster.aws.corpus_builder.bootstrap as bootstrap_module
from cluster.aws.corpus_builder.bootstrap import (
    BootstrapConfig,
    download_bootstrap_inputs,
    render_bootstrap,
)
from cluster.aws.corpus_builder.contracts import CORPUS_BUCKET, S3ObjectVersion


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SH = ROOT / "cluster" / "aws" / "corpus_builder" / "bootstrap.sh"

_BUILD_ID = "b" * 64
_PACKAGE_SHA256 = "a" * 64
_SOURCE_MANIFEST_SHA256 = "c" * 64
_PROFILE_SHA256 = "d" * 64
_LAUNCH_INTENT_SHA256 = "e" * 64
_KMS_ARN = (
    "arn:aws:kms:us-east-1:056956104102:"
    "key/01234567-89ab-cdef-0123-456789abcdef"
)
_OTHER_KMS_ARN = (
    "arn:aws:kms:us-east-1:056956104102:"
    "key/fedcba98-7654-3210-fedc-ba9876543210"
)


def _object(
    name: str,
    *,
    sha256: str,
    version_id: str,
    kms_key_arn: str = _KMS_ARN,
) -> S3ObjectVersion:
    return S3ObjectVersion(
        uri=f"s3://{CORPUS_BUCKET}/v2/builds/{_BUILD_ID}/bootstrap/{name}",
        version_id=version_id,
        bytes=4096,
        sha256=sha256,
        etag="0123456789abcdef0123456789abcdef",
        sse_algorithm="aws:kms",
        kms_key_arn=kms_key_arn,
    )


def fixture_config() -> BootstrapConfig:
    return BootstrapConfig(
        build_id=_BUILD_ID,
        package=_object(
            "memorysplit-corpus-builder.tar.gz",
            sha256=_PACKAGE_SHA256,
            version_id="package-version-0001",
        ),
        source_manifest=_object(
            "source-manifest.json",
            sha256=_SOURCE_MANIFEST_SHA256,
            version_id="manifest-version-0002",
        ),
        kms_key_arn=_KMS_ARN,
        profile_sha256=_PROFILE_SHA256,
        launch_intent_sha256=_LAUNCH_INTENT_SHA256,
        workers=32,
    )


def test_bootstrap_mounts_exactly_four_nvmes_as_raid0_xfs_and_starts_watchdog():
    text = render_bootstrap(fixture_config())

    assert "mdadm --create /dev/md/memorysplit" in text
    assert "--level=0" in text
    assert "--raid-devices=4" in text
    assert "--chunk=512" in text
    assert "mkfs.xfs" in text
    assert "mount -t xfs -o noatime,nodiratime" in text
    assert "OnActiveSec=23h30m" in text
    assert "Persistent=true" in text
    assert "/usr/sbin/shutdown -h now" in text


def test_bootstrap_discovers_only_blank_unmounted_unique_instance_store_nvmes():
    text = render_bootstrap(fixture_config())

    assert "Amazon EC2 NVMe Instance Storage" in text
    assert "exactly four instance-store NVMe devices" in text
    assert "duplicate instance-store NVMe serial" in text
    assert "instance-store NVMe has an existing filesystem" in text
    assert "instance-store NVMe is mounted" in text
    assert "instance-store NVMe has child devices" in text
    assert "wipefs --no-act" in text
    assert "findmnt --source" in text


def test_bootstrap_records_device_serials_uuid_and_owner_only_directories():
    text = render_bootstrap(fixture_config())

    assert "bootstrap-receipt.json" in text
    assert '"serial"' in text
    assert "blkid -s UUID -o value /dev/md/memorysplit" in text
    assert 'filesystem_uuid' in text
    assert "install -d -m 0700" in text
    assert "/mnt/memorysplit-builder/work" in text
    assert "/mnt/memorysplit-builder/output" in text
    assert "/mnt/memorysplit-builder/cleanroom" in text


def test_watchdog_is_enabled_before_any_package_download_and_uploads_timeout_marker():
    text = render_bootstrap(fixture_config())

    enabled_at = text.index(
        "systemctl enable --now memorysplit-corpus-watchdog.timer"
    )
    downloaded_at = text.index("download-inputs")
    assert enabled_at < downloaded_at
    assert "memorysplit-corpus-watchdog.service" in text
    assert "memorysplit-corpus-watchdog.timer" in text
    assert "operational/timeout.json" in text
    assert "ServerSideEncryption" in text
    assert "SSEKMSKeyId" in text


def test_watchdog_shutdown_is_unconditional_when_timeout_upload_fails():
    text = render_bootstrap(fixture_config())

    assert "trap '/usr/sbin/shutdown -h now' EXIT" in text
    assert "TimeoutStartSec=3min" in text
    assert "--key \"v2/builds/${MEMORYSPLIT_BUILD_ID}/operational/timeout.json\"" in text
    assert "--output json >/dev/null 2>&1 || true" in text


def test_rendered_bootstrap_pins_every_input_authority_without_mutable_reads():
    config = fixture_config()
    text = render_bootstrap(config)

    for expected in (
        config.build_id,
        config.package.uri,
        config.package.version_id,
        config.package.sha256,
        str(config.package.bytes),
        config.package.etag,
        config.source_manifest.uri,
        config.source_manifest.version_id,
        config.source_manifest.sha256,
        config.profile_sha256,
        config.launch_intent_sha256,
        config.kms_key_arn,
    ):
        assert expected in text
    assert "download_exact_object" in text
    assert "VersionId" in text
    assert "aws s3 cp" not in text
    assert "git clone" not in text
    assert "git checkout" not in text


def test_download_bootstrap_inputs_delegates_both_objects_to_task3_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    calls: list[tuple[object, S3ObjectVersion, Path]] = []

    def fake_download(
        s3: object,
        expected: S3ObjectVersion,
        destination: Path,
    ) -> None:
        calls.append((s3, expected, destination))

    monkeypatch.setattr(bootstrap_module, "download_exact_object", fake_download)
    client = object()
    config = fixture_config()
    package_path = tmp_path / "package.tar.gz"
    manifest_path = tmp_path / "source-manifest.json"

    download_bootstrap_inputs(
        client,
        config,
        package_destination=package_path,
        source_manifest_destination=manifest_path,
    )

    assert calls == [
        (client, config.package, package_path),
        (client, config.source_manifest, manifest_path),
    ]


@pytest.mark.parametrize(
    "config",
    (
        replace(fixture_config(), build_id="B" * 64),
        replace(fixture_config(), profile_sha256="short"),
        replace(fixture_config(), launch_intent_sha256="f" * 63),
        replace(fixture_config(), workers=0),
        replace(fixture_config(), workers=65),
        replace(
            fixture_config(),
            package=replace(fixture_config().package, kms_key_arn=_OTHER_KMS_ARN),
        ),
        replace(
            fixture_config(),
            source_manifest=replace(
                fixture_config().source_manifest,
                uri=(
                    f"s3://{CORPUS_BUCKET}/v2/builds/"
                    f"{'f' * 64}/bootstrap/source-manifest.json"
                ),
            ),
        ),
    ),
)
def test_render_bootstrap_rejects_invalid_or_cross_authority_config(
    config: BootstrapConfig,
):
    with pytest.raises(ValueError):
        render_bootstrap(config)


def test_render_bootstrap_is_deterministic_cwd_independent_and_bash_syntax_valid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = fixture_config()
    first = render_bootstrap(config)
    monkeypatch.chdir(tmp_path)
    second = render_bootstrap(config)

    assert first == second
    assert first.startswith("#!/usr/bin/env bash\n")
    completed = subprocess.run(
        ["bash", "-n"],
        input=first,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_runtime_script_uses_strict_shell_and_allowlisted_driver_environment():
    text = BOOTSTRAP_SH.read_text(encoding="utf-8")

    assert "set -Eeuo pipefail" in text
    assert "umask 077" in text
    assert "env -i" in text
    for name in (
        "HOME=",
        "LANG=",
        "LC_ALL=",
        "PATH=",
        "PYTHONHASHSEED=",
        "MEMORYSPLIT_BUILD_ID=",
        "MEMORYSPLIT_SOURCE_MANIFEST=",
        "MEMORYSPLIT_WORKERS=",
    ):
        assert name in text
    assert "upload_phase_log" in text
    assert "trap " in text


def test_runtime_script_rejects_unsafe_archive_members_before_final_extraction():
    text = BOOTSTRAP_SH.read_text(encoding="utf-8")

    assert "verify-package" in text
    assert "unsafe package archive member" in text
    assert "tar -xzf" in text
    assert text.index("verify-package") < text.index("tar -xzf")
    assert "/opt/memorysplit" in text
