"""Deterministic fresh-instance control bundle for the AWS SSM bootstrap."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import secrets
import stat
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping

from .errors import MsctlError
from .fsutil import open_directory, rename_noreplace_at
from .jsonutil import canonical_json, require_sha256


CONTROL_BUNDLE_TYPE = "memorysplit-aws-control-bundle-v1"
CONTROL_BUNDLE_MEMBERS = (
    "cluster/aws/p5/bootstrap.py",
    "cluster/aws/p5/interruption_checkpoint.py",
    "cluster/aws/p5/profile.py",
    "cluster/profiles/aws-p5.48xlarge.json",
    "cluster/profiles/aws-p5.48xlarge-v3.json",
    "cluster/profiles/aws-p6-b300.48xlarge-v3.json",
    "msctl/__init__.py",
    "msctl/aws_argv.py",
    "msctl/errors.py",
    "msctl/fsutil.py",
    "msctl/jsonutil.py",
)
CONTROL_MANIFEST = "CONTROL-BUNDLE.json"
CONTROL_SUMS = "CONTROL-SHA256SUMS"
_MAX_MEMBER_BYTES = 4 * 1024 * 1024
_S3_URI_RE = re.compile(
    r"^s3://[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]/"
    r"[A-Za-z0-9._/-]+$"
)


@dataclass(frozen=True)
class ControlBundle:
    sha256: str
    bytes: int
    members: Mapping[str, str]
    payload: bytes


def render_control_install_command(
    *,
    bundle_sha256: str,
    bundle_uri: str,
    region: str,
) -> str:
    """Render the fixed stock-SSM installer for one reviewed control bundle."""

    digest = require_sha256(bundle_sha256, label="control bundle")
    if region not in {"us-east-1", "us-west-2"}:
        _fail("control install region is outside the closed scope")
    if (
        not isinstance(bundle_uri, str)
        or _S3_URI_RE.fullmatch(bundle_uri) is None
        or any(part in {"", ".", ".."} for part in bundle_uri.split("/")[3:])
    ):
        _fail("control install URI is not one safe S3 object")
    return (
        "set -eu; umask 077; "
        f"B='/var/lib/memorysplit/control-{digest}.tar'; "
        f"R='/opt/memorysplit/control/{digest}'; "
        "/usr/bin/install -d -m 0700 /var/lib/memorysplit "
        "/opt/memorysplit/control; "
        'if [ -e "$B" ]; then test -f "$B" && test ! -L "$B"; '
        'else T="$B.tmp.$$"; test ! -e "$T"; '
        f"/usr/bin/env aws --region '{region}' s3 cp '{bundle_uri}' "
        '"$T" --only-show-errors --no-progress; '
        f"printf '%s  %s\\n' '{digest}' \"$T\" | "
        "/usr/bin/sha256sum -c -; "
        '/usr/bin/chmod 0400 "$T"; /usr/bin/ln "$T" "$B"; '
        '/usr/bin/rm "$T"; fi; '
        f"printf '%s  %s\\n' '{digest}' \"$B\" | "
        "/usr/bin/sha256sum -c -; "
        'if [ -e "$R" ]; then test -d "$R" && test ! -L "$R"; '
        '/usr/bin/tar --extract --to-stdout --file "$B" '
        'CONTROL-BUNDLE.json | /usr/bin/cmp - "$R/CONTROL-BUNDLE.json"; '
        '/usr/bin/tar --extract --to-stdout --file "$B" '
        'CONTROL-SHA256SUMS | /usr/bin/cmp - "$R/CONTROL-SHA256SUMS"; '
        '(cd "$R" && /usr/bin/sha256sum -c CONTROL-SHA256SUMS); '
        'E=$(cd "$R" && { /usr/bin/awk \'{print $2}\' '
        "CONTROL-SHA256SUMS; printf '%s\\n' CONTROL-BUNDLE.json "
        "CONTROL-SHA256SUMS; } | /usr/bin/sort); "
        'A=$(cd "$R" && /usr/bin/find . -type f -printf \'%P\\n\' '
        '| /usr/bin/sort); test "$A" = "$E"; '
        'test -z "$(cd "$R" && /usr/bin/find . '
        '! -type d ! -type f -print -quit)"; '
        'test -z "$(cd "$R" && /usr/bin/find . -mindepth 1 '
        '-perm /222 -print -quit)"; '
        'else X="$R.tmp.$$"; test ! -e "$X"; '
        '/usr/bin/install -d -m 0700 "$X"; '
        '/usr/bin/tar --extract --file "$B" --directory "$X" '
        "--no-same-owner --no-same-permissions; "
        '(cd "$X" && /usr/bin/sha256sum -c CONTROL-SHA256SUMS); '
        '/usr/bin/chmod -R a-w "$X"; '
        '/usr/bin/mv -T -n "$X" "$R"; test ! -e "$X"; fi; '
        'test -f "$R/msctl/aws_argv.py" && '
        'test ! -L "$R/msctl/aws_argv.py"'
    )


def _fail(message: str) -> None:
    raise MsctlError("CONTROL_BUNDLE_INVALID", message)


def _source_member(root: Path, relative: str) -> bytes:
    path = root.joinpath(*PurePosixPath(relative).parts)
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
        before = path.stat(follow_symlinks=False)
    except (OSError, ValueError) as error:
        raise MsctlError(
            "CONTROL_BUNDLE_INVALID",
            f"control source member is unavailable: {relative}",
        ) from error
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > _MAX_MEMBER_BYTES
    ):
        _fail(f"control source member is unsafe: {relative}")
    payload = path.read_bytes()
    after = path.stat(follow_symlinks=False)
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
    ) or len(payload) != after.st_size:
        _fail(f"control source member changed while read: {relative}")
    return payload


def _tar_entry(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mode = 0o444
    info.uid = 0
    info.gid = 0
    info.mtime = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(payload))


def build_control_bundle_bytes(source_root: Path | str) -> ControlBundle:
    """Build byte-identical tar bytes from the exact reviewed control subset."""

    root = Path(source_root)
    payloads = {
        relative: _source_member(root, relative)
        for relative in CONTROL_BUNDLE_MEMBERS
    }
    hashes = {
        relative: hashlib.sha256(payload).hexdigest()
        for relative, payload in payloads.items()
    }
    sums = "".join(
        f"{hashes[relative]}  {relative}\n"
        for relative in CONTROL_BUNDLE_MEMBERS
    ).encode("ascii")
    manifest = canonical_json(
        {
            "schema_version": 1,
            "bundle_type": CONTROL_BUNDLE_TYPE,
            "members": hashes,
            "sums_sha256": hashlib.sha256(sums).hexdigest(),
        }
    ) + b"\n"
    all_payloads = {
        **payloads,
        CONTROL_MANIFEST: manifest,
        CONTROL_SUMS: sums,
    }
    output = io.BytesIO()
    with tarfile.open(
        fileobj=output,
        mode="w",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        for name in sorted(all_payloads):
            _tar_entry(archive, name, all_payloads[name])
    data = output.getvalue()
    return ControlBundle(
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        members=dict(sorted(hashes.items())),
        payload=data,
    )


def verify_control_bundle(
    path: Path | str,
    *,
    expected_sha256: str,
) -> ControlBundle:
    """Verify archive hash, exact member set, metadata, and member hashes."""

    expected = require_sha256(expected_sha256, label="control bundle")
    candidate = Path(path)
    try:
        before = candidate.stat(follow_symlinks=False)
        if (
            candidate.is_symlink()
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            _fail("control bundle must be one singly linked regular file")
        data = candidate.read_bytes()
        after = candidate.stat(follow_symlinks=False)
    except OSError as error:
        raise MsctlError(
            "CONTROL_BUNDLE_INVALID",
            "control bundle cannot be read",
        ) from error
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
    ):
        _fail("control bundle changed while read")
    if hashlib.sha256(data).hexdigest() != expected:
        _fail("control bundle SHA-256 does not match")
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
            infos = archive.getmembers()
            names = [info.name for info in infos]
            if (
                names != sorted(names)
                or len(names) != len(set(names))
                or set(names)
                != set(CONTROL_BUNDLE_MEMBERS)
                | {CONTROL_MANIFEST, CONTROL_SUMS}
                or any(
                    not info.isfile()
                    or info.issym()
                    or info.islnk()
                    or info.mode != 0o444
                    or info.uid != 0
                    or info.gid != 0
                    or info.mtime != 0
                    for info in infos
                )
            ):
                _fail("control bundle tar inventory is not exact")
            payloads = {
                info.name: archive.extractfile(info).read()
                for info in infos
            }
    except (OSError, tarfile.TarError, AttributeError) as error:
        raise MsctlError(
            "CONTROL_BUNDLE_INVALID",
            "control bundle tar is invalid",
        ) from error
    try:
        manifest = json.loads(payloads[CONTROL_MANIFEST].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MsctlError(
            "CONTROL_BUNDLE_INVALID",
            "control bundle manifest is invalid",
        ) from error
    hashes = {
        relative: hashlib.sha256(payloads[relative]).hexdigest()
        for relative in CONTROL_BUNDLE_MEMBERS
    }
    sums = "".join(
        f"{hashes[relative]}  {relative}\n"
        for relative in CONTROL_BUNDLE_MEMBERS
    ).encode("ascii")
    expected_manifest = {
        "schema_version": 1,
        "bundle_type": CONTROL_BUNDLE_TYPE,
        "members": hashes,
        "sums_sha256": hashlib.sha256(sums).hexdigest(),
    }
    if (
        payloads[CONTROL_SUMS] != sums
        or payloads[CONTROL_MANIFEST] != canonical_json(expected_manifest) + b"\n"
        or manifest != expected_manifest
    ):
        _fail("control bundle member manifest does not match")
    return ControlBundle(
        sha256=expected,
        bytes=len(data),
        members=dict(sorted(hashes.items())),
        payload=data,
    )


def write_control_bundle(path: Path | str, bundle: ControlBundle) -> Path:
    destination = Path(path)
    directory_fd = open_directory(
        destination.parent,
        label="control bundle output",
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
        view = memoryview(bundle.payload)
        while view:
            view = view[os.write(descriptor, view) :]
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
                "CONTROL_BUNDLE_EXISTS",
                "refusing to replace an existing control bundle",
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


def plan_control_bundle(
    *,
    source_root: Path | str,
    out: Path | str,
    apply: bool,
) -> dict[str, object]:
    bundle = build_control_bundle_bytes(source_root)
    result = {
        "schema_version": 1,
        "bundle_type": CONTROL_BUNDLE_TYPE,
        "bundle_sha256": bundle.sha256,
        "bundle_bytes": bundle.bytes,
        "members": dict(bundle.members),
        "out": str(Path(out)),
        "published": False,
    }
    if apply:
        write_control_bundle(out, bundle)
        verify_control_bundle(out, expected_sha256=bundle.sha256)
        result["published"] = True
    return result
