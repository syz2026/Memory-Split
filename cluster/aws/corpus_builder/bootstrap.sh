#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# __MEMORYSPLIT_RENDERED_CONFIG__

readonly CORPUS_BUCKET="memorysplit-corpus-056956104102-us-east-1"
readonly CORPUS_REGION="us-east-1"
readonly SAFE_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

RAID_DEVICE=""
SCRATCH_ROOT=""
WORK_ROOT=""
OUTPUT_ROOT=""
CLEANROOM_ROOT=""
BOOTSTRAP_RECEIPT=""
LOG_PATH=""
RUNTIME_ROOT=""
WATCHDOG_ENV=""
WATCHDOG_PROGRAM=""
WATCHDOG_STATE_DIR=""
SYSTEMD_UNIT_DIR=""
BUILDER_ENTRYPOINT=""
BUILD_CONTEXT_ENV=""
LSBLK_JSON=""
NVME_RECORDS=""
AWS_BIN=""
SHUTDOWN_BIN=""
TIMEOUT_BIN=""
PYTHON_BIN=""
SHA256_BIN=""
AWK_BIN=""
INSTALL_BIN=""
SYNC_BIN=""
SYSTEMCTL_BIN=""
CURRENT_PHASE="initializing"
SHUTDOWN_REQUESTED=0
NVME_DEVICES=()
NVME_SERIALS=()

die() {
    printf 'bootstrap error [%s]: %s\n' "${CURRENT_PHASE}" "$*" >&2
    exit 1
}

configure_paths() {
    local requested_root="${1:-/}"
    [[ "${requested_root}" == /* ]] \
        || die "bootstrap state root must be absolute"
    [[ -d "${requested_root}" && ! -L "${requested_root}" ]] \
        || die "bootstrap state root must be a real directory"
    local prefix
    if [[ "${requested_root}" == "/" ]]; then
        prefix=""
        RAID_DEVICE="/dev/md/memorysplit"
        SCRATCH_ROOT="/mnt/memorysplit-builder"
        WORK_ROOT="/mnt/memorysplit-builder/work"
        OUTPUT_ROOT="/mnt/memorysplit-builder/output"
        CLEANROOM_ROOT="/mnt/memorysplit-builder/cleanroom"
        BOOTSTRAP_RECEIPT="/mnt/memorysplit-builder/bootstrap-receipt.json"
    else
        prefix="${requested_root%/}"
        RAID_DEVICE="${prefix}/dev/md/memorysplit"
        SCRATCH_ROOT="${prefix}/mnt/memorysplit-builder"
        WORK_ROOT="${SCRATCH_ROOT}/work"
        OUTPUT_ROOT="${SCRATCH_ROOT}/output"
        CLEANROOM_ROOT="${SCRATCH_ROOT}/cleanroom"
        BOOTSTRAP_RECEIPT="${SCRATCH_ROOT}/bootstrap-receipt.json"
    fi
    LOG_PATH="${prefix}/var/log/memorysplit-corpus-bootstrap.log"
    RUNTIME_ROOT="${prefix}/run/memorysplit-corpus-bootstrap"
    WATCHDOG_ENV="${prefix}/etc/memorysplit-corpus-watchdog.env"
    WATCHDOG_PROGRAM="${prefix}/usr/local/sbin/memorysplit-corpus-watchdog"
    WATCHDOG_STATE_DIR="${prefix}/var/lib/memorysplit-corpus-watchdog"
    SYSTEMD_UNIT_DIR="${prefix}/etc/systemd/system"
    BUILDER_ENTRYPOINT="${prefix}/usr/local/bin/memorysplit-corpus-builder"
    BUILD_CONTEXT_ENV="${prefix}/etc/memorysplit-corpus-build.env"
    LSBLK_JSON="${RUNTIME_ROOT}/lsblk.json"
    NVME_RECORDS="${RUNTIME_ROOT}/nvme-records.tsv"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command is unavailable: $1"
}

request_shutdown() {
    SHUTDOWN_REQUESTED=1
    if [[ -n "${SHUTDOWN_BIN}" ]]; then
        "${SHUTDOWN_BIN}" -h now >/dev/null 2>&1 || true
    else
        /usr/sbin/shutdown -h now >/dev/null 2>&1 || true
    fi
}

emergency_exit() {
    local status=$?
    trap - EXIT
    if (( status != 0 )); then
        request_shutdown
    fi
    exit "${status}"
}

on_exit() {
    local status=$?
    trap - EXIT ERR
    set +e
    if (( status != 0 )); then
        if (( SHUTDOWN_REQUESTED == 0 )); then
            request_shutdown
        fi
        if [[ -n "${LOG_PATH}" ]]; then
            printf 'bootstrap failed in phase %s with status %d\n' \
                "${CURRENT_PHASE}" "${status}" >>"${LOG_PATH}"
        fi
    fi
    if [[ -n "${RUNTIME_ROOT}" ]]; then
        rm -rf -- "${RUNTIME_ROOT}"
    fi
    exit "${status}"
}

initialize_runtime() {
    CURRENT_PHASE="runtime-initialize"
    for required in \
        awk \
        aws \
        blkid \
        findmnt \
        install \
        lsblk \
        mdadm \
        mkfs.xfs \
        mount \
        python3 \
        sha256sum \
        systemctl \
        tar \
        timeout \
        udevadm \
        wipefs; do
        require_command "${required}"
    done

    [[ ! -e "${RUNTIME_ROOT}" && ! -L "${RUNTIME_ROOT}" ]] \
        || die "bootstrap runtime path already exists"
    "${INSTALL_BIN}" -d -m 0700 "${RUNTIME_ROOT}"
    "${INSTALL_BIN}" -d -m 0755 "${LOG_PATH%/*}"
    "${INSTALL_BIN}" -m 0600 /dev/null "${LOG_PATH}"
    exec >>"${LOG_PATH}" 2>&1
    printf 'starting build-invariant MemorySplit corpus bootstrap\n'
}

discover_instance_store_nvmes() {
    CURRENT_PHASE="nvme-discovery"
    udevadm settle
    lsblk --json --paths \
        --output NAME,MODEL,SERIAL,FSTYPE,MOUNTPOINTS,TYPE >"${LSBLK_JSON}"

    python3 - "${LSBLK_JSON}" >"${NVME_RECORDS}" <<'PY'
import json
import re
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as stream:
    value = json.load(stream)
devices = []
serials = set()
for item in value.get("blockdevices", []):
    model = item.get("model")
    if isinstance(model, str):
        model = model.strip()
    if model != "Amazon EC2 NVMe Instance Storage":
        continue
    name = item.get("name")
    serial = item.get("serial")
    if item.get("type") != "disk" or not isinstance(name, str):
        raise SystemExit("instance-store NVMe is not a whole disk")
    if re.fullmatch(r"/dev/nvme[0-9]+n[0-9]+", name) is None:
        raise SystemExit("instance-store NVMe has an unsafe device name")
    if not isinstance(serial, str) or not serial or any(c.isspace() for c in serial):
        raise SystemExit("instance-store NVMe has an unsafe serial")
    if serial in serials:
        raise SystemExit("duplicate instance-store NVMe serial")
    fstype = item.get("fstype")
    if fstype not in (None, ""):
        raise SystemExit("instance-store NVMe has an existing filesystem")
    mountpoints = item.get("mountpoints")
    if mountpoints is None:
        mountpoints = []
    if not isinstance(mountpoints, list) or any(mountpoints):
        raise SystemExit("instance-store NVMe is mounted")
    if item.get("children"):
        raise SystemExit("instance-store NVMe has child devices")
    serials.add(serial)
    devices.append((name, serial))
if len(devices) != 4:
    raise SystemExit("expected exactly four instance-store NVMe devices")
for name, serial in sorted(devices):
    print(f"{name}\t{serial}")
PY

    mapfile -t records <"${NVME_RECORDS}"
    [[ "${#records[@]}" -eq 4 ]] \
        || die "expected exactly four instance-store NVMe devices"

    local root_source root_parent line device serial signatures
    root_source="$(findmnt --noheadings --output SOURCE /)"
    root_parent="$(lsblk --noheadings --output PKNAME "${root_source}" 2>/dev/null \
        | awk 'NF {print "/dev/" $1; exit}')"
    for line in "${records[@]}"; do
        IFS=$'\t' read -r device serial <<<"${line}"
        [[ -b "${device}" ]] || die "instance-store NVMe is not a block device"
        [[ "${device}" != "${root_source}" && "${device}" != "${root_parent}" ]] \
            || die "root EBS device was selected as instance storage"
        if findmnt --source "${device}" >/dev/null 2>&1; then
            die "instance-store NVMe is mounted"
        fi
        signatures="$(wipefs --no-act --output TYPE --noheadings "${device}")"
        [[ -z "${signatures//[[:space:]]/}" ]] \
            || die "instance-store NVMe has an existing filesystem"
        if mdadm --examine "${device}" >/dev/null 2>&1; then
            die "instance-store NVMe has existing RAID metadata"
        fi
        if compgen -G "/sys/class/block/${device##*/}/holders/*" >/dev/null; then
            die "instance-store NVMe is already held by another device"
        fi
        NVME_DEVICES+=("${device}")
        NVME_SERIALS+=("${serial}")
    done
}

build_scratch_array() {
    CURRENT_PHASE="nvme-raid"
    install -d -m 0755 /dev/md
    mdadm --create /dev/md/memorysplit \
        --run \
        --level=0 \
        --raid-devices=4 \
        --chunk=512 \
        "${NVME_DEVICES[@]}"
    udevadm settle
    [[ -b "${RAID_DEVICE}" ]] || die "RAID0 device was not created"
    mkfs.xfs -f -L memorysplit-builder "${RAID_DEVICE}"
    install -d -m 0700 "${SCRATCH_ROOT}"
    mount -t xfs -o noatime,nodiratime \
        "${RAID_DEVICE}" "${SCRATCH_ROOT}"
    findmnt --mountpoint "${SCRATCH_ROOT}" --source "${RAID_DEVICE}" >/dev/null \
        || die "scratch XFS mount cannot be verified"
    chmod 0700 "${SCRATCH_ROOT}"
    install -d -m 0700 \
        "${WORK_ROOT}" \
        "${OUTPUT_ROOT}" \
        "${CLEANROOM_ROOT}"
}

write_bootstrap_receipt() {
    CURRENT_PHASE="bootstrap-receipt"
    local filesystem_uuid
    filesystem_uuid="$(blkid -s UUID -o value /dev/md/memorysplit)"
    [[ -n "${filesystem_uuid}" ]] || die "XFS filesystem UUID is unavailable"
    export MEMORYSPLIT_FILESYSTEM_UUID="${filesystem_uuid}"
    export MEMORYSPLIT_NVME_RECORDS="${NVME_RECORDS}"
    export MEMORYSPLIT_BOOTSTRAP_RECEIPT="${BOOTSTRAP_RECEIPT}"
    python3 <<'PY'
import datetime
import json
import os

devices = []
with open(os.environ["MEMORYSPLIT_NVME_RECORDS"], "r", encoding="utf-8") as stream:
    for line in stream:
        device, serial = line.rstrip("\n").split("\t", 1)
        devices.append({"device": device, "serial": serial})
receipt = {
    "filesystem_type": "xfs",
    "filesystem_uuid": os.environ["MEMORYSPLIT_FILESYSTEM_UUID"],
    "format": "memorysplit-aws-corpus-bootstrap-v1",
    "mount_options": ["noatime", "nodiratime"],
    "mount_path": "/mnt/memorysplit-builder",
    "nvme_devices": devices,
    "raid_chunk_kib": 512,
    "raid_device": "/dev/md/memorysplit",
    "raid_level": 0,
    "schema_version": 1,
    "started_at": datetime.datetime.now(
        datetime.timezone.utc
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
}
path = os.environ["MEMORYSPLIT_BOOTSTRAP_RECEIPT"]
with open(path, "x", encoding="utf-8") as stream:
    json.dump(
        receipt,
        stream,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    stream.write("\n")
PY
    chmod 0600 "${BOOTSTRAP_RECEIPT}"
}

install_watchdog() {
    CURRENT_PHASE="watchdog-install"
    SHUTDOWN_BIN="$(command -v shutdown || true)"
    TIMEOUT_BIN="$(command -v timeout || true)"
    AWS_BIN="$(command -v aws || true)"
    PYTHON_BIN="$(command -v python3 || true)"
    SHA256_BIN="$(command -v sha256sum || true)"
    AWK_BIN="$(command -v awk || true)"
    INSTALL_BIN="$(command -v install || true)"
    SYNC_BIN="$(command -v sync || true)"
    SYSTEMCTL_BIN="$(command -v systemctl || true)"
    for resolved in \
        "${SHUTDOWN_BIN}" \
        "${TIMEOUT_BIN}" \
        "${AWS_BIN}" \
        "${PYTHON_BIN}" \
        "${SHA256_BIN}" \
        "${AWK_BIN}" \
        "${INSTALL_BIN}" \
        "${SYNC_BIN}" \
        "${SYSTEMCTL_BIN}"; do
        [[ -n "${resolved}" && -x "${resolved}" ]] \
            || die "watchdog prerequisite is unavailable"
    done

    "${INSTALL_BIN}" -d -m 0755 \
        "${SYSTEMD_UNIT_DIR}" \
        "${WATCHDOG_ENV%/*}" \
        "${WATCHDOG_PROGRAM%/*}"
    "${INSTALL_BIN}" -d -m 0700 "${WATCHDOG_STATE_DIR}"
    {
        printf 'MEMORYSPLIT_BUILD_CONTEXT_ENV=%q\n' "${BUILD_CONTEXT_ENV}"
        printf 'MEMORYSPLIT_SHUTDOWN_BIN=%q\n' "${SHUTDOWN_BIN}"
        printf 'MEMORYSPLIT_TIMEOUT_BIN=%q\n' "${TIMEOUT_BIN}"
        printf 'MEMORYSPLIT_AWS_BIN=%q\n' "${AWS_BIN}"
        printf 'MEMORYSPLIT_PYTHON_BIN=%q\n' "${PYTHON_BIN}"
        printf 'MEMORYSPLIT_SHA256_BIN=%q\n' "${SHA256_BIN}"
        printf 'MEMORYSPLIT_AWK_BIN=%q\n' "${AWK_BIN}"
        printf 'MEMORYSPLIT_SYNC_BIN=%q\n' "${SYNC_BIN}"
        printf 'MEMORYSPLIT_TIMEOUT_MARKER=%q\n' \
            "${WATCHDOG_STATE_DIR}/timeout.json"
    } >"${WATCHDOG_ENV}"
    chmod 0600 "${WATCHDOG_ENV}"

    {
        cat <<'WATCHDOG_HEADER'
#!/usr/bin/env bash
set +e
umask 077
WATCHDOG_HEADER
        printf 'source %q || true\n' "${WATCHDOG_ENV}"
        cat <<'WATCHDOG_BODY'
: "${MEMORYSPLIT_SHUTDOWN_BIN:=/usr/sbin/shutdown}"
: "${MEMORYSPLIT_TIMEOUT_BIN:=/usr/bin/timeout}"
: "${MEMORYSPLIT_AWS_BIN:=/usr/bin/aws}"
: "${MEMORYSPLIT_PYTHON_BIN:=/usr/bin/python3}"
: "${MEMORYSPLIT_SHA256_BIN:=/usr/bin/sha256sum}"
: "${MEMORYSPLIT_AWK_BIN:=/usr/bin/awk}"
: "${MEMORYSPLIT_SYNC_BIN:=/usr/bin/sync}"
: "${MEMORYSPLIT_TIMEOUT_MARKER:=/var/lib/memorysplit-corpus-watchdog/timeout.json}"
: "${MEMORYSPLIT_BUILD_CONTEXT_ENV:=/etc/memorysplit-corpus-build.env}"

terminate() {
    if [[ -n "${MEMORYSPLIT_SHUTDOWN_BIN}" ]]; then
        "${MEMORYSPLIT_SHUTDOWN_BIN}" -h now >/dev/null 2>&1 || true
    else
        /usr/sbin/shutdown -h now >/dev/null 2>&1 || true
    fi
}

trap terminate EXIT
terminate

if [[ -f "${MEMORYSPLIT_BUILD_CONTEXT_ENV}" \
    && ! -L "${MEMORYSPLIT_BUILD_CONTEXT_ENV}" ]]; then
    source "${MEMORYSPLIT_BUILD_CONTEXT_ENV}" || true
fi

export MEMORYSPLIT_TIMEOUT_MARKER
"${MEMORYSPLIT_PYTHON_BIN}" <<'PY'
import datetime
import json
import os

value = {
    "format": "memorysplit-aws-corpus-timeout-v1",
    "reason": "watchdog-timeout",
    "schema_version": 1,
    "shutdown_requested_at": datetime.datetime.now(
        datetime.timezone.utc
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
}
build_id = os.environ.get("MEMORYSPLIT_BUILD_ID")
if build_id:
    value["build_id"] = build_id
with open(os.environ["MEMORYSPLIT_TIMEOUT_MARKER"], "w", encoding="utf-8") as stream:
    json.dump(value, stream, separators=(",", ":"), sort_keys=True)
    stream.write("\n")
PY
digest="$(
    "${MEMORYSPLIT_SHA256_BIN}" "${MEMORYSPLIT_TIMEOUT_MARKER}" \
        | "${MEMORYSPLIT_AWK_BIN}" '{print $1}'
)"
if [[ "${MEMORYSPLIT_BUILD_ID:-}" =~ ^[0-9a-f]{64}$ \
    && -n "${MEMORYSPLIT_KMS_KEY_ARN:-}" ]]; then
    "${MEMORYSPLIT_TIMEOUT_BIN}" 5s "${MEMORYSPLIT_AWS_BIN}" s3api put-object \
        --region us-east-1 \
        --bucket memorysplit-corpus-056956104102-us-east-1 \
        --key "v2/builds/${MEMORYSPLIT_BUILD_ID}/operational/timeout.json" \
        --body "${MEMORYSPLIT_TIMEOUT_MARKER}" \
        --server-side-encryption aws:kms \
        --ssekms-key-id "${MEMORYSPLIT_KMS_KEY_ARN}" \
        --metadata "sha256=${digest},build-id=${MEMORYSPLIT_BUILD_ID}" \
        --no-cli-pager \
        --output json >/dev/null 2>&1 || true
fi
"${MEMORYSPLIT_SYNC_BIN}" || true
WATCHDOG_BODY
    } >"${WATCHDOG_PROGRAM}"
    chmod 0700 "${WATCHDOG_PROGRAM}"

    cat >"${SYSTEMD_UNIT_DIR}/memorysplit-corpus-watchdog.service" <<SERVICE
[Unit]
Description=Terminate MemorySplit corpus builder before 24-hour ceiling

[Service]
Type=oneshot
TimeoutStartSec=15s
ExecStart=${WATCHDOG_PROGRAM}
SERVICE

    cat >"${SYSTEMD_UNIT_DIR}/memorysplit-corpus-watchdog.timer" <<'TIMER'
[Unit]
Description=Terminate MemorySplit corpus builder before 24-hour ceiling

[Timer]
OnBootSec=23h30m
Persistent=true
AccuracySec=1s
Unit=memorysplit-corpus-watchdog.service

[Install]
WantedBy=timers.target
TIMER
    chmod 0644 \
        "${SYSTEMD_UNIT_DIR}/memorysplit-corpus-watchdog.service" \
        "${SYSTEMD_UNIT_DIR}/memorysplit-corpus-watchdog.timer"
    "${SYSTEMCTL_BIN}" daemon-reload
    "${SYSTEMCTL_BIN}" enable --now memorysplit-corpus-watchdog.timer
    "${SYSTEMCTL_BIN}" is-active --quiet memorysplit-corpus-watchdog.timer \
        || die "termination watchdog timer did not start"
}

install_fixed_entrypoint() {
    CURRENT_PHASE="entrypoint-install"
    "${INSTALL_BIN}" -d -m 0755 "${BUILDER_ENTRYPOINT%/*}"
    cat >"${BUILDER_ENTRYPOINT}" <<'BUILDER_ENTRYPOINT'
#!/usr/bin/env python3
import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
from dataclasses import dataclass


AWS_PATH = "/usr/bin/aws"
BUCKET = "memorysplit-corpus-056956104102-us-east-1"
REGION = "us-east-1"
SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
KMS_ARN_RE = re.compile(
    r"^arn:aws:kms:us-east-1:056956104102:key/[A-Za-z0-9-]+$"
)
WORK_ROOT = pathlib.Path("/mnt/memorysplit-builder/work")
OUTPUT_ROOT = pathlib.Path("/mnt/memorysplit-builder/output")
CLEANROOM_ROOT = pathlib.Path("/mnt/memorysplit-builder/cleanroom")
BOOTSTRAP_RECEIPT = pathlib.Path(
    "/mnt/memorysplit-builder/bootstrap-receipt.json"
)
BUILD_CONTEXT_ENV = pathlib.Path("/etc/memorysplit-corpus-build.env")
INSTALL_ROOT = pathlib.Path("/opt/memorysplit")
PROFILE_RELATIVE_PATH = pathlib.Path(
    "cluster/profiles/aws-i4i.16xlarge-corpus-v1.json"
)
DRIVER_RELATIVE_PATH = pathlib.Path(
    "scripts/aws_corpus_builder_driver.py"
)


class BuilderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ObjectAuthority:
    uri: str
    bucket: str
    key: str
    version_id: str
    bytes: int
    sha256: str
    etag: str
    kms_key_arn: str


def _aws_environment():
    return {
        "AWS_CONFIG_FILE": "/dev/null",
        "AWS_DEFAULT_REGION": REGION,
        "AWS_EC2_METADATA_DISABLED": "false",
        "AWS_PAGER": "",
        "AWS_REGION": REGION,
        "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
        "HOME": "/root",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": SAFE_PATH,
    }


def _run_aws_json(arguments):
    timeout = 900 if arguments and arguments[0] == "get-object" else 120
    try:
        completed = subprocess.run(
            [
                AWS_PATH,
                "s3api",
                *arguments,
                "--no-cli-pager",
                "--output",
                "json",
            ],
            cwd="/",
            env=_aws_environment(),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BuilderError("AWS CLI exact-version request failed") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        raise BuilderError(
            "AWS CLI exact-version request failed: "
            + (detail or "unknown error")
        )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise BuilderError("AWS CLI returned malformed JSON") from error
    if not isinstance(response, dict):
        raise BuilderError("AWS CLI response must be a JSON object")
    return response


def _parse_uri(uri, namespace):
    if not isinstance(uri, str):
        raise BuilderError("input URI must be a string")
    parsed = urllib.parse.urlsplit(uri)
    if (
        parsed.scheme != "s3"
        or parsed.netloc != BUCKET
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise BuilderError("input URI is outside the approved S3 authority")
    key = parsed.path[1:]
    if (
        not key.startswith(namespace)
        or key == namespace
        or re.fullmatch(r"[A-Za-z0-9._/-]+", key) is None
        or any(part in {"", ".", ".."} for part in key.split("/"))
    ):
        raise BuilderError("input URI has an unsafe or unexpected key")
    return key


def _etag(response):
    value = response.get("ETag")
    if not isinstance(value, str):
        raise BuilderError("S3 authority lacks an ETag")
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    if re.fullmatch(r"[A-Za-z0-9-]{1,128}", value) is None:
        raise BuilderError("S3 authority has an unsafe ETag")
    return value


def _authority_from_response(uri, key, version_id, expected_sha256, response):
    content_length = response.get("ContentLength")
    metadata = response.get("Metadata")
    kms_key_arn = response.get("SSEKMSKeyId")
    if response.get("VersionId") != version_id:
        raise BuilderError("S3 VersionId drift")
    if (
        isinstance(content_length, bool)
        or not isinstance(content_length, int)
        or content_length <= 0
    ):
        raise BuilderError("S3 authority has an invalid byte count")
    if response.get("ServerSideEncryption") != "aws:kms":
        raise BuilderError("S3 authority is not SSE-KMS encrypted")
    if not isinstance(kms_key_arn, str) or KMS_ARN_RE.fullmatch(kms_key_arn) is None:
        raise BuilderError("S3 authority has an unexpected KMS key")
    if not isinstance(metadata, dict) or metadata.get("sha256") != expected_sha256:
        raise BuilderError("S3 metadata SHA-256 does not match the command")
    return ObjectAuthority(
        uri=uri,
        bucket=BUCKET,
        key=key,
        version_id=version_id,
        bytes=content_length,
        sha256=expected_sha256,
        etag=_etag(response),
        kms_key_arn=kms_key_arn,
    )


def _head_exact(uri, key, version_id, expected_sha256, runner):
    response = runner(
        [
            "head-object",
            "--region",
            REGION,
            "--bucket",
            BUCKET,
            "--key",
            key,
            "--version-id",
            version_id,
        ]
    )
    return _authority_from_response(
        uri,
        key,
        version_id,
        expected_sha256,
        response,
    )


def _resolve_exact_object(uri, expected_sha256, namespace, runner=_run_aws_json):
    if not isinstance(expected_sha256, str) or SHA256_RE.fullmatch(
        expected_sha256
    ) is None:
        raise BuilderError("input SHA-256 must be lowercase hexadecimal")
    key = _parse_uri(uri, namespace)
    versions = []
    delete_markers = []
    key_marker = None
    version_marker = None
    seen_markers = set()
    while True:
        arguments = [
            "list-object-versions",
            "--region",
            REGION,
            "--bucket",
            BUCKET,
            "--prefix",
            key,
        ]
        if key_marker is not None:
            arguments.extend(["--key-marker", key_marker])
        if version_marker is not None:
            arguments.extend(["--version-id-marker", version_marker])
        response = runner(arguments)
        listed_versions = response.get("Versions", [])
        listed_markers = response.get("DeleteMarkers", [])
        if not isinstance(listed_versions, list) or not isinstance(
            listed_markers, list
        ):
            raise BuilderError("S3 version history is malformed")
        versions.extend(
            item
            for item in listed_versions
            if isinstance(item, dict) and item.get("Key") == key
        )
        delete_markers.extend(
            item
            for item in listed_markers
            if isinstance(item, dict) and item.get("Key") == key
        )
        if response.get("IsTruncated") is not True:
            break
        key_marker = response.get("NextKeyMarker")
        version_marker = response.get("NextVersionIdMarker")
        marker = (key_marker, version_marker)
        if (
            not isinstance(key_marker, str)
            or not key_marker
            or not isinstance(version_marker, str)
            or not version_marker
            or marker in seen_markers
        ):
            raise BuilderError("S3 version-history pagination is unsafe")
        seen_markers.add(marker)
    if delete_markers or len(versions) != 1:
        raise BuilderError(
            "input key must have exactly one immutable object version"
        )
    version_id = versions[0].get("VersionId")
    if not isinstance(version_id, str) or not version_id or version_id == "null":
        raise BuilderError("input key lacks a versioned S3 authority")
    return _head_exact(
        uri,
        key,
        version_id,
        expected_sha256,
        runner,
    )


def _hash_regular(path):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BuilderError("download is not a regular file")
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            byte_count += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or byte_count != before.st_size:
        raise BuilderError("download changed while hashing")
    return byte_count, digest.hexdigest()


def _validate_get_response(authority, response):
    observed = _authority_from_response(
        authority.uri,
        authority.key,
        authority.version_id,
        authority.sha256,
        response,
    )
    if observed != authority:
        raise BuilderError("S3 GET response drifted from exact HEAD authority")


def _download_exact(authority, destination):
    destination = pathlib.Path(destination)
    if destination.exists() or destination.is_symlink():
        raise BuilderError("download destination already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".exact-s3-",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    try:
        response = _run_aws_json(
            [
                "get-object",
                "--region",
                REGION,
                "--bucket",
                authority.bucket,
                "--key",
                authority.key,
                "--version-id",
                authority.version_id,
                str(temporary),
            ]
        )
        _validate_get_response(authority, response)
        byte_count, sha256 = _hash_regular(temporary)
        if byte_count != authority.bytes or sha256 != authority.sha256:
            raise BuilderError("download bytes do not match exact authority")
        final_head = _head_exact(
            authority.uri,
            authority.key,
            authority.version_id,
            authority.sha256,
            _run_aws_json,
        )
        if final_head != authority:
            raise BuilderError("S3 authority changed after exact download")
        os.link(temporary, destination, follow_symlinks=False)
        destination.chmod(0o600)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise BuilderError("source manifest repeats a JSON field")
        value[key] = item
    return value


def _verify_manifest(path):
    if os.lstat(path).st_size > 64 * 1024 * 1024:
        raise BuilderError("source manifest exceeds 64 MiB")
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_unique_pairs,
                parse_constant=lambda item: (_ for _ in ()).throw(
                    BuilderError(
                        "source manifest has a non-finite value: " + item
                    )
                ),
            )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuilderError("source manifest is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise BuilderError("source manifest must be a JSON object")


def _safe_extract(archive_path, destination):
    names = set()
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                pure = pathlib.PurePosixPath(member.name)
                if (
                    not member.name
                    or pure.is_absolute()
                    or pure.as_posix() != member.name
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or member.name in names
                    or not (member.isdir() or member.isreg())
                ):
                    raise BuilderError(
                        "unsafe package archive member: " + repr(member.name)
                    )
                names.add(member.name)
            if not names:
                raise BuilderError("package archive is empty")
            for member in members:
                pure = pathlib.PurePosixPath(member.name)
                target = destination.joinpath(*pure.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=False, mode=0o755)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                source = archive.extractfile(member)
                if source is None:
                    raise BuilderError("safe archive file has no body")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
    except BuilderError:
        raise
    except (OSError, tarfile.TarError) as error:
        raise BuilderError("package is not a safe gzip tar archive") from error


def _require_owner_directory(path):
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise BuilderError("bootstrap owner-only directory is unavailable")


def _write_build_context(build_id, kms_key_arn):
    if BUILD_CONTEXT_ENV.is_symlink():
        raise BuilderError("watchdog build context path is a symlink")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".memorysplit-build.",
        dir=BUILD_CONTEXT_ENV.parent,
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                "MEMORYSPLIT_BUILD_ID=" + shlex.quote(build_id) + "\n"
            )
            stream.write(
                "MEMORYSPLIT_KMS_KEY_ARN="
                + shlex.quote(kms_key_arn)
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, BUILD_CONTEXT_ENV)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _profile_sha256(profile_path):
    byte_count, sha256 = _hash_regular(profile_path)
    if byte_count <= 0:
        raise BuilderError("frozen builder profile is empty")
    return sha256


def _run_driver(args, source_manifest, package_sha256, kms_key_arn):
    driver = INSTALL_ROOT / DRIVER_RELATIVE_PATH
    profile = INSTALL_ROOT / PROFILE_RELATIVE_PATH
    if (
        driver.is_symlink()
        or not driver.is_file()
        or profile.is_symlink()
        or not profile.is_file()
    ):
        raise BuilderError("verified package lacks fixed builder files")
    environment = {
        "AWS_CONFIG_FILE": "/dev/null",
        "AWS_DEFAULT_REGION": REGION,
        "AWS_EC2_METADATA_DISABLED": "false",
        "AWS_REGION": REGION,
        "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
        "HOME": "/root",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": SAFE_PATH,
        "PYTHONHASHSEED": "0",
        "MEMORYSPLIT_BUILD_ID": args.build_id,
        "MEMORYSPLIT_CLEANROOM_ROOT": str(CLEANROOM_ROOT),
        "MEMORYSPLIT_KMS_KEY_ARN": kms_key_arn,
        "MEMORYSPLIT_OUTPUT_ROOT": str(OUTPUT_ROOT),
        "MEMORYSPLIT_PACKAGE_SHA256": package_sha256,
        "MEMORYSPLIT_PROFILE_SHA256": _profile_sha256(profile),
        "MEMORYSPLIT_SOURCE_MANIFEST": str(source_manifest),
        "MEMORYSPLIT_SOURCE_MANIFEST_SHA256": args.source_manifest_sha256,
        "MEMORYSPLIT_WORK_ROOT": str(WORK_ROOT),
        "MEMORYSPLIT_WORKERS": str(max(1, (os.cpu_count() or 1) // 2)),
    }
    completed = subprocess.run(
        ["/usr/bin/python3", str(driver)],
        cwd=str(INSTALL_ROOT),
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise BuilderError(
            "corpus builder driver failed with status "
            + str(completed.returncode)
        )


def _request_shutdown():
    try:
        subprocess.run(
            ["/usr/sbin/shutdown", "-h", "now"],
            cwd="/",
            env={
                "HOME": "/root",
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": SAFE_PATH,
            },
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def _parser():
    parser = argparse.ArgumentParser(
        description="Run one exact MemorySplit corpus-builder package."
    )
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--package-uri", required=True)
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--source-manifest-uri", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    return parser


def _run(args):
    if SHA256_RE.fullmatch(args.build_id) is None:
        raise BuilderError("build ID must be a lowercase SHA-256")
    for path in (WORK_ROOT, OUTPUT_ROOT, CLEANROOM_ROOT):
        _require_owner_directory(path)
    receipt_metadata = os.lstat(BOOTSTRAP_RECEIPT)
    if not stat.S_ISREG(receipt_metadata.st_mode):
        raise BuilderError("bootstrap receipt is unavailable")

    lock_path = pathlib.Path("/run/memorysplit-corpus-builder.lock")
    lock = lock_path.open("x+b")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        package = _resolve_exact_object(
            args.package_uri,
            args.package_sha256,
            "v2/packages/",
        )
        manifest = _resolve_exact_object(
            args.source_manifest_uri,
            args.source_manifest_sha256,
            "v2/sources/",
        )
        if package.kms_key_arn != manifest.kms_key_arn:
            raise BuilderError("package and manifest use different KMS keys")
        _write_build_context(args.build_id, package.kms_key_arn)

        input_root = WORK_ROOT / ("inputs-" + args.build_id)
        input_root.mkdir(mode=0o700)
        package_path = input_root / "memorysplit-corpus-builder.tar.gz"
        manifest_path = input_root / "source-manifest.json"
        _download_exact(package, package_path)
        _download_exact(manifest, manifest_path)
        _verify_manifest(manifest_path)

        install_stage = pathlib.Path(
            "/opt/.memorysplit.install-" + args.build_id
        )
        if (
            INSTALL_ROOT.exists()
            or INSTALL_ROOT.is_symlink()
            or install_stage.exists()
            or install_stage.is_symlink()
        ):
            raise BuilderError("package install path already exists")
        install_stage.mkdir(mode=0o700)
        try:
            _safe_extract(package_path, install_stage)
            os.replace(install_stage, INSTALL_ROOT)
        finally:
            if install_stage.exists():
                shutil.rmtree(install_stage)
        _run_driver(
            args,
            manifest_path,
            package.sha256,
            package.kms_key_arn,
        )
    finally:
        lock.close()


def main(argv=None):
    try:
        args = _parser().parse_args(argv)
        _run(args)
    except (BuilderError, OSError) as error:
        print("corpus builder failed: " + str(error), file=sys.stderr)
        return 1
    finally:
        _request_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
BUILDER_ENTRYPOINT
    chmod 0700 "${BUILDER_ENTRYPOINT}"
    "${PYTHON_BIN}" - "${BUILDER_ENTRYPOINT}" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
compile(path.read_text(encoding="utf-8"), str(path), "exec")
PY
}

main() {
    configure_paths "${1:-/}"
    SHUTDOWN_BIN="$(command -v shutdown || true)"
    trap emergency_exit EXIT
    install_watchdog
    trap on_exit EXIT
    initialize_runtime
    discover_instance_store_nvmes
    build_scratch_array
    write_bootstrap_receipt
    install_fixed_entrypoint
    CURRENT_PHASE="bootstrap-complete"
    "${SYNC_BIN}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "${1:-/}"
fi
