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
DRIVER_PATH=""
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
    DRIVER_PATH="${prefix}/opt/memorysplit/scripts/aws_corpus_builder_driver.py"
    LSBLK_JSON="${RUNTIME_ROOT}/lsblk.json"
    NVME_RECORDS="${RUNTIME_ROOT}/nvme-records.tsv"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command is unavailable: $1"
}

object_key_from_uri() {
    local uri="$1"
    local prefix="s3://${CORPUS_BUCKET}/"
    [[ "${uri}" == "${prefix}"* ]] || die "S3 URI is outside the corpus bucket"
    printf '%s' "${uri#"${prefix}"}"
}

upload_file() {
    local key="$1"
    local path="$2"
    local digest
    [[ -f "${path}" && ! -L "${path}" ]] || die "upload source is not a regular file"
    digest="$(sha256sum "${path}" | awk '{print $1}')"
    "${TIMEOUT_BIN}" 120s "${AWS_BIN}" s3api put-object \
        --region "${CORPUS_REGION}" \
        --bucket "${CORPUS_BUCKET}" \
        --key "${key}" \
        --body "${path}" \
        --server-side-encryption "aws:kms" \
        --ssekms-key-id "${MEMORYSPLIT_KMS_KEY_ARN}" \
        --metadata "sha256=${digest},build-id=${MEMORYSPLIT_BUILD_ID}" \
        --no-cli-pager \
        --output json >/dev/null
}

upload_phase_log() {
    local phase="$1"
    upload_file \
        "v2/builds/${MEMORYSPLIT_BUILD_ID}/operational/logs/${phase}.log" \
        "${LOG_PATH}"
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
        if [[ -f "${LOG_PATH}" && ! -L "${LOG_PATH}" ]]; then
            upload_phase_log "bootstrap-failed"
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
    printf 'starting MemorySplit corpus bootstrap for %s\n' \
        "${MEMORYSPLIT_BUILD_ID}"
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
    "build_id": os.environ["MEMORYSPLIT_BUILD_ID"],
    "filesystem_type": "xfs",
    "filesystem_uuid": os.environ["MEMORYSPLIT_FILESYSTEM_UUID"],
    "format": "memorysplit-aws-corpus-bootstrap-v1",
    "launch_intent_sha256": os.environ["MEMORYSPLIT_LAUNCH_INTENT_SHA256"],
    "mount_options": ["noatime", "nodiratime"],
    "mount_path": "/mnt/memorysplit-builder",
    "nvme_devices": devices,
    "package_sha256": os.environ["MEMORYSPLIT_PACKAGE_SHA256"],
    "profile_sha256": os.environ["MEMORYSPLIT_PROFILE_SHA256"],
    "raid_chunk_kib": 512,
    "raid_device": "/dev/md/memorysplit",
    "raid_level": 0,
    "schema_version": 1,
    "source_manifest_sha256": os.environ["MEMORYSPLIT_SOURCE_MANIFEST_SHA256"],
    "started_at": datetime.datetime.now(
        datetime.timezone.utc
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "workers": int(os.environ["MEMORYSPLIT_WORKERS"]),
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
        printf 'MEMORYSPLIT_BUILD_ID=%q\n' "${MEMORYSPLIT_BUILD_ID}"
        printf 'MEMORYSPLIT_KMS_KEY_ARN=%q\n' "${MEMORYSPLIT_KMS_KEY_ARN}"
        printf 'MEMORYSPLIT_LAUNCH_INTENT_SHA256=%q\n' \
            "${MEMORYSPLIT_LAUNCH_INTENT_SHA256}"
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

terminate() {
    if [[ -n "${MEMORYSPLIT_SHUTDOWN_BIN}" ]]; then
        "${MEMORYSPLIT_SHUTDOWN_BIN}" -h now >/dev/null 2>&1 || true
    else
        /usr/sbin/shutdown -h now >/dev/null 2>&1 || true
    fi
}

trap terminate EXIT
terminate

export MEMORYSPLIT_TIMEOUT_MARKER
"${MEMORYSPLIT_PYTHON_BIN}" <<'PY'
import datetime
import json
import os

value = {
    "build_id": os.environ["MEMORYSPLIT_BUILD_ID"],
    "format": "memorysplit-aws-corpus-timeout-v1",
    "launch_intent_sha256": os.environ["MEMORYSPLIT_LAUNCH_INTENT_SHA256"],
    "reason": "watchdog-timeout",
    "schema_version": 1,
    "shutdown_requested_at": datetime.datetime.now(
        datetime.timezone.utc
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
}
with open(os.environ["MEMORYSPLIT_TIMEOUT_MARKER"], "w", encoding="utf-8") as stream:
    json.dump(value, stream, separators=(",", ":"), sort_keys=True)
    stream.write("\n")
PY
digest="$(
    "${MEMORYSPLIT_SHA256_BIN}" "${MEMORYSPLIT_TIMEOUT_MARKER}" \
        | "${MEMORYSPLIT_AWK_BIN}" '{print $1}'
)"
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

seed_exact_package() {
    CURRENT_PHASE="bootstrap-seed"
    local package_key response_path
    readonly SEED_ARCHIVE="${RUNTIME_ROOT}/seed-package.tar.gz"
    readonly SEED_CODE_ROOT="${RUNTIME_ROOT}/seed-code"
    response_path="${RUNTIME_ROOT}/seed-get-response.json"
    package_key="$(object_key_from_uri "${MEMORYSPLIT_PACKAGE_URI}")"
    "${TIMEOUT_BIN}" 900s "${AWS_BIN}" s3api get-object \
        --region "${CORPUS_REGION}" \
        --bucket "${CORPUS_BUCKET}" \
        --key "${package_key}" \
        --version-id "${MEMORYSPLIT_PACKAGE_VERSION_ID}" \
        --no-cli-pager \
        --output json \
        "${SEED_ARCHIVE}" >"${response_path}"
    chmod 0600 "${SEED_ARCHIVE}" "${response_path}"

    export MEMORYSPLIT_SEED_ARCHIVE="${SEED_ARCHIVE}"
    export MEMORYSPLIT_SEED_RESPONSE="${response_path}"
    python3 <<'PY'
import hashlib
import json
import os

with open(os.environ["MEMORYSPLIT_SEED_RESPONSE"], "r", encoding="utf-8") as stream:
    response = json.load(stream)
etag = response.get("ETag")
if isinstance(etag, str) and etag.startswith('"') and etag.endswith('"'):
    etag = etag[1:-1]
expected = {
    "ContentLength": int(os.environ["MEMORYSPLIT_PACKAGE_BYTES"]),
    "VersionId": os.environ["MEMORYSPLIT_PACKAGE_VERSION_ID"],
    "ServerSideEncryption": os.environ["MEMORYSPLIT_PACKAGE_SSE_ALGORITHM"],
    "SSEKMSKeyId": os.environ["MEMORYSPLIT_PACKAGE_KMS_KEY_ARN"],
}
for key, value in expected.items():
    if response.get(key) != value:
        raise SystemExit(f"seed package {key} drift")
if etag != os.environ["MEMORYSPLIT_PACKAGE_ETAG"]:
    raise SystemExit("seed package ETag drift")
metadata = response.get("Metadata")
if not isinstance(metadata, dict) or metadata.get("sha256") != os.environ[
    "MEMORYSPLIT_PACKAGE_SHA256"
]:
    raise SystemExit("seed package metadata SHA-256 drift")
digest = hashlib.sha256()
byte_count = 0
with open(os.environ["MEMORYSPLIT_SEED_ARCHIVE"], "rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        byte_count += len(chunk)
        digest.update(chunk)
if byte_count != int(os.environ["MEMORYSPLIT_PACKAGE_BYTES"]):
    raise SystemExit("seed package byte count drift")
if digest.hexdigest() != os.environ["MEMORYSPLIT_PACKAGE_SHA256"]:
    raise SystemExit("seed package SHA-256 drift")
PY

    install -d -m 0700 "${SEED_CODE_ROOT}"
    export MEMORYSPLIT_SEED_CODE_ROOT="${SEED_CODE_ROOT}"
    python3 <<'PY'
import os
import pathlib
import shutil
import tarfile

archive_path = os.environ["MEMORYSPLIT_SEED_ARCHIVE"]
destination = pathlib.Path(os.environ["MEMORYSPLIT_SEED_CODE_ROOT"])
names = set()
with tarfile.open(archive_path, "r:gz") as archive:
    for member in archive.getmembers():
        pure = pathlib.PurePosixPath(member.name)
        if (
            not member.name
            or pure.is_absolute()
            or pure.as_posix() != member.name
            or any(part in {"", ".", ".."} for part in pure.parts)
            or member.name in names
            or not (member.isdir() or member.isreg())
        ):
            raise SystemExit(f"unsafe package archive member: {member.name!r}")
        names.add(member.name)
        target = destination.joinpath(*pure.parts)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=False, mode=0o755)
            continue
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        source = archive.extractfile(member)
        if source is None:
            raise SystemExit("unsafe package archive member has no body")
        with source, target.open("xb") as output:
            shutil.copyfileobj(source, output)
        target.chmod(0o755 if member.mode & 0o111 else 0o644)
if not names:
    raise SystemExit("seed package archive is empty")
PY
    [[ -f "${SEED_CODE_ROOT}/cluster/aws/corpus_builder/bootstrap.py" ]] \
        || die "seed package lacks the bootstrap runtime"
    [[ -f "${SEED_CODE_ROOT}/cluster/aws/corpus_builder/s3.py" ]] \
        || die "seed package lacks the Task 3 S3 helper"
}

download_and_verify_inputs() {
    CURRENT_PHASE="exact-input-download"
    readonly INPUT_ROOT="${WORK_ROOT}/bootstrap-inputs"
    readonly PACKAGE_PATH="${INPUT_ROOT}/memorysplit-corpus-builder.tar.gz"
    readonly SOURCE_MANIFEST_PATH="${INPUT_ROOT}/source-manifest.json"
    install -d -m 0700 "${INPUT_ROOT}"

    # The runtime subcommand delegates both records to download_exact_object.
    # Task 3 verifies VersionId, ServerSideEncryption, SSEKMSKeyId, ETag,
    # byte count, metadata SHA-256, streamed SHA-256, and a final exact HEAD.
    PYTHONPATH="${SEED_CODE_ROOT}" python3 \
        -m cluster.aws.corpus_builder.bootstrap \
        download-inputs \
        --scratch-directory "${INPUT_ROOT}" \
        --package-destination "${PACKAGE_PATH}" \
        --source-manifest-destination "${SOURCE_MANIFEST_PATH}"

    PYTHONPATH="${SEED_CODE_ROOT}" python3 \
        -m cluster.aws.corpus_builder.bootstrap \
        verify-package \
        --archive "${PACKAGE_PATH}" \
        --source-manifest "${SOURCE_MANIFEST_PATH}"
}

install_verified_package() {
    CURRENT_PHASE="package-install"
    local install_stage="/opt/.memorysplit.install"
    [[ ! -e /opt/memorysplit && ! -L /opt/memorysplit ]] \
        || die "/opt/memorysplit already exists"
    [[ ! -e "${install_stage}" && ! -L "${install_stage}" ]] \
        || die "package install staging path already exists"
    install -d -m 0755 "${install_stage}"
    tar -xzf "${PACKAGE_PATH}" \
        --directory "${install_stage}" \
        --no-same-owner
    mv -- "${install_stage}" /opt/memorysplit
    [[ -f "${DRIVER_PATH}" && ! -L "${DRIVER_PATH}" ]] \
        || die "verified package lacks the corpus builder driver"
}

run_driver() {
    CURRENT_PHASE="corpus-driver"
    env -i \
        HOME="/root" \
        LANG="C" \
        LC_ALL="C" \
        PATH="${SAFE_PATH}" \
        PYTHONHASHSEED="0" \
        AWS_CONFIG_FILE="/dev/null" \
        AWS_SHARED_CREDENTIALS_FILE="/dev/null" \
        AWS_DEFAULT_REGION="${CORPUS_REGION}" \
        AWS_REGION="${CORPUS_REGION}" \
        AWS_EC2_METADATA_DISABLED="false" \
        MEMORYSPLIT_BUILD_ID="${MEMORYSPLIT_BUILD_ID}" \
        MEMORYSPLIT_CLEANROOM_ROOT="${CLEANROOM_ROOT}" \
        MEMORYSPLIT_KMS_KEY_ARN="${MEMORYSPLIT_KMS_KEY_ARN}" \
        MEMORYSPLIT_LAUNCH_INTENT_SHA256="${MEMORYSPLIT_LAUNCH_INTENT_SHA256}" \
        MEMORYSPLIT_OUTPUT_ROOT="${OUTPUT_ROOT}" \
        MEMORYSPLIT_PACKAGE_SHA256="${MEMORYSPLIT_PACKAGE_SHA256}" \
        MEMORYSPLIT_PROFILE_SHA256="${MEMORYSPLIT_PROFILE_SHA256}" \
        MEMORYSPLIT_SOURCE_MANIFEST="${SOURCE_MANIFEST_PATH}" \
        MEMORYSPLIT_SOURCE_MANIFEST_SHA256="${MEMORYSPLIT_SOURCE_MANIFEST_SHA256}" \
        MEMORYSPLIT_WORK_ROOT="${WORK_ROOT}" \
        MEMORYSPLIT_WORKERS="${MEMORYSPLIT_WORKERS}" \
        /usr/bin/python3 "${DRIVER_PATH}"
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

    upload_file \
        "v2/builds/${MEMORYSPLIT_BUILD_ID}/operational/bootstrap-receipt.json" \
        "${BOOTSTRAP_RECEIPT}"
    upload_phase_log "nvme-mounted"
    upload_phase_log "watchdog-installed"

    seed_exact_package
    upload_phase_log "bootstrap-seed-verified"

    download_and_verify_inputs
    upload_phase_log "inputs-verified"

    install_verified_package
    upload_phase_log "package-installed"

    run_driver
    CURRENT_PHASE="driver-complete"
    upload_phase_log "driver-complete"

    CURRENT_PHASE="normal-shutdown"
    "${SYNC_BIN}"
    request_shutdown
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "${1:-/}"
fi
