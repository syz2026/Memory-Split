from __future__ import annotations

import gzip
import os
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


def alternate_config() -> BootstrapConfig:
    build_id = "f" * 64
    config = fixture_config()
    return replace(
        config,
        build_id=build_id,
        package=replace(
            config.package,
            uri=(
                f"s3://{CORPUS_BUCKET}/v2/builds/{build_id}/"
                "bootstrap/other-package.tar.gz"
            ),
            version_id="other-package-version",
            sha256="1" * 64,
        ),
        source_manifest=replace(
            config.source_manifest,
            uri=(
                f"s3://{CORPUS_BUCKET}/v2/builds/{build_id}/"
                "bootstrap/other-source-manifest.json"
            ),
            version_id="other-manifest-version",
            sha256="2" * 64,
        ),
        profile_sha256="3" * 64,
        launch_intent_sha256="4" * 64,
        workers=64,
    )


def _builder_entrypoint_source(text: str) -> str:
    marker = "<<'BUILDER_ENTRYPOINT'\n"
    assert text.count(marker) == 1
    source = text.split(marker, 1)[1].split("\nBUILDER_ENTRYPOINT", 1)[0]
    compile(source, "memorysplit-corpus-builder", "exec", dont_inherit=True)
    return source


def _write_stub(path: Path, body: str) -> None:
    path.write_text(
        "#!/usr/bin/env bash\nset -u\n" + body,
        encoding="utf-8",
    )
    path.chmod(0o755)


def _run_sandboxed_bootstrap(
    text: str,
    tmp_path: Path,
    *,
    name: str,
) -> tuple[subprocess.CompletedProcess[str], list[str], Path, dict[str, str], Path]:
    sandbox = tmp_path / name
    command_dir = sandbox / "bin"
    state_root = sandbox / "root"
    call_log = sandbox / "calls.log"
    command_dir.mkdir(parents=True)
    state_root.mkdir()

    _write_stub(
        command_dir / "systemctl",
        'printf "systemctl %s\\n" "$*" >>"${CALL_LOG}"\nexit 0\n',
    )
    _write_stub(
        command_dir / "udevadm",
        'printf "udevadm %s\\n" "$*" >>"${CALL_LOG}"\nexit 73\n',
    )
    _write_stub(
        command_dir / "shutdown",
        'printf "shutdown %s\\n" "$*" >>"${CALL_LOG}"\nexit 0\n',
    )
    _write_stub(
        command_dir / "timeout",
        'printf "timeout %s\\n" "$*" >>"${CALL_LOG}"\nexit 124\n',
    )
    _write_stub(
        command_dir / "aws",
        'printf "aws %s\\n" "$*" >>"${CALL_LOG}"\nexit 1\n',
    )
    for command in (
        "blkid",
        "findmnt",
        "lsblk",
        "mdadm",
        "mkfs.xfs",
        "mount",
        "wipefs",
    ):
        _write_stub(command_dir / command, "exit 0\n")

    script_path = sandbox / "bootstrap.sh"
    script_path.write_text(text, encoding="utf-8")
    script_path.chmod(0o700)
    environment = dict(os.environ)
    environment.update(
        {
            "CALL_LOG": str(call_log),
            "PATH": f"{command_dir}:{environment['PATH']}",
        }
    )
    completed = subprocess.run(
        ["bash", str(script_path), str(state_root)],
        cwd=command_dir,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    calls = (
        call_log.read_text(encoding="utf-8").splitlines()
        if call_log.exists()
        else []
    )
    return completed, calls, state_root, environment, call_log


def _call_index(calls: list[str], prefix: str) -> int:
    for index, call in enumerate(calls):
        if call.startswith(prefix):
            return index
    raise AssertionError(f"missing call {prefix!r}; recorded calls: {calls!r}")


def test_bootstrap_mounts_exactly_four_nvmes_as_raid0_xfs_and_starts_watchdog():
    text = render_bootstrap(fixture_config())

    assert "mdadm --create /dev/md/memorysplit" in text
    assert "--level=0" in text
    assert "--raid-devices=4" in text
    assert "--chunk=512" in text
    assert "mkfs.xfs" in text
    assert "mount -t xfs -o noatime,nodiratime" in text
    assert "OnBootSec=23h30m" in text
    assert "OnActiveSec=" not in text
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


def test_watchdog_uploads_timeout_marker():
    text = render_bootstrap(fixture_config())

    assert "memorysplit-corpus-watchdog.service" in text
    assert "memorysplit-corpus-watchdog.timer" in text
    assert "operational/timeout.json" in text
    assert "ServerSideEncryption" in text
    assert "SSEKMSKeyId" in text


def test_rendered_bootstrap_activates_watchdog_before_any_storage_command(
    tmp_path: Path,
):
    text = render_bootstrap(fixture_config())
    completed, calls, _root, _environment, _call_log = _run_sandboxed_bootstrap(
        text,
        tmp_path,
        name="ordered",
    )

    assert completed.returncode != 0
    assert _call_index(
        calls,
        "systemctl enable --now memorysplit-corpus-watchdog.timer",
    ) < _call_index(calls, "udevadm settle")

    regressed = text.replace(
        "    install_watchdog\n"
        "    trap on_exit EXIT\n"
        "    initialize_runtime\n"
        "    discover_instance_store_nvmes\n",
        "    trap on_exit EXIT\n"
        "    initialize_runtime\n"
        "    discover_instance_store_nvmes\n"
        "    install_watchdog\n",
        1,
    )
    assert regressed != text
    _failed, regressed_calls, *_rest = _run_sandboxed_bootstrap(
        regressed,
        tmp_path,
        name="regressed",
    )
    assert not any(
        call.startswith(
            "systemctl enable --now memorysplit-corpus-watchdog.timer"
        )
        for call in regressed_calls
    )


def test_watchdog_initiates_shutdown_before_timed_out_marker_upload(
    tmp_path: Path,
):
    completed, _calls, root, environment, call_log = _run_sandboxed_bootstrap(
        render_bootstrap(fixture_config()),
        tmp_path,
        name="timeout",
    )
    watchdog = (
        root / "usr" / "local" / "sbin" / "memorysplit-corpus-watchdog"
    )
    assert watchdog.is_file(), completed.stderr
    (root / "etc" / "memorysplit-corpus-build.env").write_text(
        (
            f"MEMORYSPLIT_BUILD_ID={_BUILD_ID}\n"
            f"MEMORYSPLIT_KMS_KEY_ARN={_KMS_ARN}\n"
        ),
        encoding="utf-8",
    )
    call_log.write_text("", encoding="utf-8")

    watchdog_result = subprocess.run(
        [str(watchdog)],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    watchdog_calls = call_log.read_text(encoding="utf-8").splitlines()

    assert watchdog_result.returncode == 0
    assert watchdog_calls[0] == "shutdown -h now"
    assert watchdog_calls[1].startswith("timeout 5s ")
    assert watchdog_calls[-1] == "shutdown -h now"


def test_rendered_bootstrap_is_build_invariant_and_contains_no_build_values():
    first_config = fixture_config()
    second_config = alternate_config()
    first = render_bootstrap(first_config)
    second = render_bootstrap(second_config)

    assert first == second
    assert first == render_bootstrap()
    for config in (first_config, second_config):
        for forbidden in (
            config.build_id,
            config.package.uri,
            config.package.version_id,
            config.package.sha256,
            config.source_manifest.uri,
            config.source_manifest.version_id,
            config.source_manifest.sha256,
            config.profile_sha256,
            config.launch_intent_sha256,
            config.kms_key_arn,
        ):
            assert forbidden not in first
    assert "/usr/local/bin/memorysplit-corpus-builder" in first
    assert "--build-id" in first
    assert "--package-uri" in first
    assert "--source-manifest-uri" in first
    assert "aws s3 cp" not in first
    assert "git clone" not in first
    assert "git checkout" not in first


def test_invariant_bootstrap_gzip_fits_ec2_user_data_limit():
    payload = gzip.compress(
        render_bootstrap(fixture_config()).encode("utf-8"),
        compresslevel=9,
        mtime=0,
    )

    assert len(payload) <= 16_384


def test_fixed_entrypoint_resolves_one_immutable_version_before_download():
    source = _builder_entrypoint_source(render_bootstrap(fixture_config()))
    namespace: dict[str, object] = {"__name__": "memorysplit_entrypoint_test"}
    exec(
        compile(
            source,
            "memorysplit-corpus-builder",
            "exec",
            dont_inherit=True,
        ),
        namespace,
    )
    calls: list[list[str]] = []
    sha256 = "a" * 64

    def fake_aws(arguments: list[str]) -> dict[str, object]:
        calls.append(arguments)
        if arguments[0] == "list-object-versions":
            return {
                "DeleteMarkers": [],
                "IsTruncated": False,
                "Versions": [
                    {
                        "IsLatest": True,
                        "Key": "v2/packages/package.tar.gz",
                        "VersionId": "version-001",
                    }
                ],
            }
        assert arguments[0] == "head-object"
        assert arguments[-2:] == ["--version-id", "version-001"]
        return {
            "ContentLength": 123,
            "ETag": '"0123456789abcdef0123456789abcdef"',
            "Metadata": {"sha256": sha256},
            "SSEKMSKeyId": _KMS_ARN,
            "ServerSideEncryption": "aws:kms",
            "VersionId": "version-001",
        }

    resolve = namespace["_resolve_exact_object"]
    assert callable(resolve)
    authority = resolve(
        f"s3://{CORPUS_BUCKET}/v2/packages/package.tar.gz",
        sha256,
        "v2/packages/",
        fake_aws,
    )

    assert authority.version_id == "version-001"
    assert authority.sha256 == sha256
    assert [call[0] for call in calls] == [
        "list-object-versions",
        "head-object",
    ]


def test_fixed_entrypoint_rejects_ambiguous_object_version_history():
    source = _builder_entrypoint_source(render_bootstrap(fixture_config()))
    namespace: dict[str, object] = {"__name__": "memorysplit_entrypoint_test"}
    exec(
        compile(
            source,
            "memorysplit-corpus-builder",
            "exec",
            dont_inherit=True,
        ),
        namespace,
    )

    def ambiguous_history(_arguments: list[str]) -> dict[str, object]:
        return {
            "DeleteMarkers": [],
            "IsTruncated": False,
            "Versions": [
                {
                    "Key": "v2/sources/manifest.json",
                    "VersionId": "version-001",
                },
                {
                    "Key": "v2/sources/manifest.json",
                    "VersionId": "version-002",
                },
            ],
        }

    error = namespace["BuilderError"]
    resolve = namespace["_resolve_exact_object"]
    assert isinstance(error, type)
    assert callable(resolve)
    with pytest.raises(error):
        resolve(
            f"s3://{CORPUS_BUCKET}/v2/sources/manifest.json",
            "b" * 64,
            "v2/sources/",
            ambiguous_history,
        )


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
    tmp_path: Path,
):
    with pytest.raises(ValueError):
        download_bootstrap_inputs(
            object(),
            config,
            package_destination=tmp_path / "package.tar.gz",
            source_manifest_destination=tmp_path / "manifest.json",
        )


def test_render_bootstrap_is_deterministic_cwd_independent_and_bash_syntax_valid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    first = render_bootstrap(fixture_config())
    monkeypatch.chdir(tmp_path)
    second = render_bootstrap(alternate_config())

    assert first == second
    assert first == render_bootstrap()
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
    entrypoint = _builder_entrypoint_source(render_bootstrap(fixture_config()))

    assert "set -Eeuo pipefail" in text
    assert "umask 077" in text
    for name in (
        '"HOME":',
        '"LANG":',
        '"LC_ALL":',
        '"PATH":',
        '"PYTHONHASHSEED":',
        '"MEMORYSPLIT_BUILD_ID":',
        '"MEMORYSPLIT_SOURCE_MANIFEST":',
        '"MEMORYSPLIT_WORKERS":',
    ):
        assert name in entrypoint
    assert "launch_intent" not in entrypoint.lower()
    assert "finally:" in entrypoint
    assert "_request_shutdown()" in entrypoint
    assert "trap " in text


def test_runtime_script_rejects_unsafe_archive_members_before_final_extraction():
    entrypoint = _builder_entrypoint_source(render_bootstrap(fixture_config()))

    assert "unsafe package archive member" in entrypoint
    assert "tarfile.open" in entrypoint
    assert "extractall" not in entrypoint
    assert entrypoint.index("_safe_extract(") < entrypoint.index("_run_driver(")
    assert "/opt/memorysplit" in entrypoint
