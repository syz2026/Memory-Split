"""Pinned no-follow reads and durable directory-relative writes."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import BinaryIO, Callable


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | os.O_DIRECTORY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_CHECKPOINT_REQUEST_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_CHECKPOINT_REQUEST_ARMS = frozenset({"dense", "split90"})
_CHECKPOINT_REQUEST_FIELDS = {"arm", "request_id", "schema_version"}
CHECKPOINT_REQUEST_TOKEN_FILENAME = "checkpoint-request.json"
CHECKPOINT_REQUEST_TOKEN_FILE_ENV = "MS_CHECKPOINT_REQUEST_TOKEN_FILE"
CHECKPOINT_REQUEST_ARM_ENV = "MS_CHECKPOINT_REQUEST_ARM"


def _safe_name(name: str, *, label: str) -> None:
    if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name:
        raise ValueError(f"{label} name is unsafe")


def _open_root() -> int:
    return os.open("/", os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0))


def _absolute_parts(path: Path) -> tuple[str, ...]:
    absolute = path.absolute()
    if absolute.name in {"", ".", ".."}:
        raise ValueError(f"path is unsafe: {path}")
    return absolute.parts


def _walk_parent(path: Path, *, create: bool) -> tuple[int, str]:
    parts = _absolute_parts(path)
    current = _open_root()
    try:
        for component in parts[1:-1]:
            try:
                next_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
            except OSError as error:
                if error.errno == errno.ENOENT and not create:
                    raise FileNotFoundError(
                        f"path parent is missing: {path}"
                    ) from error
                if error.errno != errno.ENOENT:
                    raise ValueError(
                        f"path has a missing, symlinked, or unsafe component: {path}"
                    ) from error
                try:
                    os.mkdir(component, mode=0o755, dir_fd=current)
                    os.fsync(current)
                    next_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
                except OSError as create_error:
                    raise ValueError(
                        f"path component could not be created safely: {path}"
                    ) from create_error
            os.close(current)
            current = next_fd
        return current, parts[-1]
    except BaseException:
        os.close(current)
        raise


def require_absent_path(path: str | Path, *, label: str) -> None:
    candidate = Path(path)
    parts = _absolute_parts(candidate)
    current = _open_root()
    try:
        for component in parts[1:-1]:
            try:
                next_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
            except OSError as error:
                if error.errno == errno.ENOENT:
                    return
                raise ValueError(
                    f"{label} has a symlinked or unsafe parent component"
                ) from error
            os.close(current)
            current = next_fd
        try:
            os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise FileExistsError(f"fresh {label} path already exists: {candidate}")
    finally:
        os.close(current)


def _read_and_hash_fd(fd: int) -> tuple[bytes, str]:
    chunks = []
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(fd, 1 << 20, offset)
        if not chunk:
            break
        chunks.append(chunk)
        digest.update(chunk)
        offset += len(chunk)
    return b"".join(chunks), digest.hexdigest()


@dataclass(frozen=True)
class PinnedBytes:
    payload: bytes
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class PublishedCheckpointRequestToken:
    path: Path
    request_id: str
    arm: str
    owner_uid: int
    owner_gid: int
    device: int
    inode: int


def _checkpoint_request_identity(
    request_id: object,
    arm: object,
) -> tuple[str, str]:
    if (
        not isinstance(request_id, str)
        or _CHECKPOINT_REQUEST_ID_RE.fullmatch(request_id) is None
    ):
        raise ValueError(
            "checkpoint request ID must be 32 lowercase hexadecimal characters"
        )
    if arm not in _CHECKPOINT_REQUEST_ARMS:
        raise ValueError("checkpoint request arm must be dense or split90")
    return request_id, str(arm)


def checkpoint_request_token_bytes(request_id: str, *, arm: str) -> bytes:
    request_id, arm = _checkpoint_request_identity(request_id, arm)
    return (
        json.dumps(
            {
                "arm": arm,
                "request_id": request_id,
                "schema_version": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def parse_checkpoint_request_token(
    payload: bytes,
    *,
    expected_arm: str,
) -> str:
    _request_id, expected_arm = _checkpoint_request_identity(
        "0" * 32,
        expected_arm,
    )

    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(
                    f"checkpoint request token repeats field: {key}"
                )
            value[key] = item
        return value

    if not isinstance(payload, bytes) or len(payload) > 256:
        raise ValueError("checkpoint request token bytes are invalid")
    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(
                    "checkpoint request token contains non-finite "
                    f"{constant}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "checkpoint request token is not canonical JSON"
        ) from error
    if (
        not isinstance(value, dict)
        or set(value) != _CHECKPOINT_REQUEST_FIELDS
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("checkpoint request token fields are invalid")
    request_id, arm = _checkpoint_request_identity(
        value["request_id"],
        value["arm"],
    )
    if arm != expected_arm:
        raise ValueError("checkpoint request token arm does not match")
    if checkpoint_request_token_bytes(request_id, arm=arm) != payload:
        raise ValueError("checkpoint request token is not canonical")
    return request_id


def _checkpoint_request_owner(
    owner_uid: object,
    owner_gid: object,
) -> tuple[int, int]:
    if type(owner_uid) is not int or owner_uid < 0:
        raise ValueError("checkpoint request token owner UID is invalid")
    if type(owner_gid) is not int or owner_gid < 0:
        raise ValueError("checkpoint request token owner GID is invalid")
    return owner_uid, owner_gid


def _validate_checkpoint_request_parent(
    parent_fd: int,
    *,
    owner_uid: int,
    owner_gid: int,
) -> None:
    metadata = os.fstat(parent_fd)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != owner_uid
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise ValueError(
            "checkpoint request token parent is foreign or unsafe"
        )


def _validate_checkpoint_request_file(
    metadata: os.stat_result,
    *,
    owner_uid: int,
    owner_gid: int,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != owner_uid
        or metadata.st_gid != owner_gid
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or metadata.st_nlink != 1
        or (
            expected_identity is not None
            and (metadata.st_dev, metadata.st_ino) != expected_identity
        )
    ):
        raise ValueError(
            "checkpoint request token identity, owner, mode, or hard-link "
            "state is unsafe"
        )


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def publish_checkpoint_request_token(
    path: str | Path,
    *,
    request_id: str,
    arm: str,
    owner_uid: int,
    owner_gid: int,
) -> PublishedCheckpointRequestToken:
    request_id, arm = _checkpoint_request_identity(request_id, arm)
    owner_uid, owner_gid = _checkpoint_request_owner(owner_uid, owner_gid)
    payload = checkpoint_request_token_bytes(request_id, arm=arm)
    candidate = Path(path)
    parent_fd, name = _walk_parent(candidate, create=False)
    temporary = f".{name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    descriptor = -1
    linked = False
    metadata = None
    try:
        _validate_checkpoint_request_parent(
            parent_fd,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
        )
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                f"pending checkpoint request token already exists: {candidate}"
            )
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        installed = os.fstat(descriptor)
        if (installed.st_uid, installed.st_gid) != (owner_uid, owner_gid):
            os.fchown(descriptor, owner_uid, owner_gid)
        os.fchmod(descriptor, 0o400)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        _validate_checkpoint_request_file(
            metadata,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
        )
        if metadata.st_size != len(payload):
            raise ValueError(
                "checkpoint request token changed while being published"
            )
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"pending checkpoint request token already exists: {candidate}"
            ) from error
        linked = True
        os.unlink(temporary, dir_fd=parent_fd)
        os.fsync(parent_fd)
        final = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        _validate_checkpoint_request_file(
            final,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
            expected_identity=(metadata.st_dev, metadata.st_ino),
        )
        return PublishedCheckpointRequestToken(
            path=candidate,
            request_id=request_id,
            arm=arm,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
            device=metadata.st_dev,
            inode=metadata.st_ino,
        )
    except BaseException:
        if linked and metadata is not None:
            try:
                current = os.stat(
                    name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if (
                    current.st_dev,
                    current.st_ino,
                ) == (metadata.st_dev, metadata.st_ino):
                    os.unlink(name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
            except (FileNotFoundError, OSError):
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)


def _restore_claimed_token(
    parent_fd: int,
    *,
    claimed: str,
    name: str,
) -> None:
    try:
        os.link(
            claimed,
            name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileExistsError:
        return
    os.unlink(claimed, dir_fd=parent_fd)
    os.fsync(parent_fd)


def _claim_checkpoint_request_token(
    path: Path,
    *,
    expected_arm: str,
    owner_uid: int,
    owner_gid: int,
    expected_request_id: str | None,
    expected_identity: tuple[int, int] | None,
    missing_ok: bool,
) -> str | None:
    owner_uid, owner_gid = _checkpoint_request_owner(owner_uid, owner_gid)
    if expected_request_id is not None:
        expected_request_id, expected_arm = _checkpoint_request_identity(
            expected_request_id,
            expected_arm,
        )
    else:
        _ignored, expected_arm = _checkpoint_request_identity(
            "0" * 32,
            expected_arm,
        )
    parent_fd, name = _walk_parent(path, create=False)
    claimed = f".{name}.claimed-{os.getpid()}-{secrets.token_hex(8)}"
    descriptor = -1
    claimed_exists = False
    consumed = False
    try:
        _validate_checkpoint_request_parent(
            parent_fd,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
        )
        try:
            os.rename(
                name,
                claimed,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        except FileNotFoundError:
            if missing_ok:
                return None
            raise FileNotFoundError(
                f"checkpoint request token is missing: {path}"
            )
        claimed_exists = True
        try:
            descriptor = os.open(claimed, _FILE_FLAGS, dir_fd=parent_fd)
        except OSError as error:
            raise ValueError(
                "checkpoint request token is symlinked or unsafe"
            ) from error
        metadata = os.fstat(descriptor)
        _validate_checkpoint_request_file(
            metadata,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
            expected_identity=expected_identity,
        )
        payload, _digest = _read_and_hash_fd(descriptor)
        if len(payload) != metadata.st_size:
            raise ValueError(
                "checkpoint request token changed while being consumed"
            )
        request_id = parse_checkpoint_request_token(
            payload,
            expected_arm=expected_arm,
        )
        if (
            expected_request_id is not None
            and not hmac.compare_digest(request_id, expected_request_id)
        ):
            raise ValueError(
                "checkpoint request token belongs to another request"
            )
        os.unlink(claimed, dir_fd=parent_fd)
        claimed_exists = False
        consumed = True
        os.fsync(parent_fd)
        if os.fstat(descriptor).st_nlink != 0:
            raise ValueError(
                "checkpoint request token changed while being removed"
            )
        return request_id
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if claimed_exists and not consumed:
            _restore_claimed_token(
                parent_fd,
                claimed=claimed,
                name=name,
            )
        os.close(parent_fd)


def consume_checkpoint_request_token(
    path: str | Path,
    *,
    expected_arm: str,
    expected_uid: int,
    expected_gid: int,
) -> str:
    request_id = _claim_checkpoint_request_token(
        Path(path),
        expected_arm=expected_arm,
        owner_uid=expected_uid,
        owner_gid=expected_gid,
        expected_request_id=None,
        expected_identity=None,
        missing_ok=False,
    )
    assert request_id is not None
    return request_id


def cleanup_checkpoint_request_token(
    token: PublishedCheckpointRequestToken,
) -> bool:
    if not isinstance(token, PublishedCheckpointRequestToken):
        raise ValueError("published checkpoint request token is invalid")
    removed = _claim_checkpoint_request_token(
        token.path,
        expected_arm=token.arm,
        owner_uid=token.owner_uid,
        owner_gid=token.owner_gid,
        expected_request_id=token.request_id,
        expected_identity=(token.device, token.inode),
        missing_ok=True,
    )
    return removed is not None


def cleanup_checkpoint_request_token_path(
    path: str | Path,
    *,
    request_id: str,
    arm: str,
    owner_uid: int,
    owner_gid: int,
) -> bool:
    removed = _claim_checkpoint_request_token(
        Path(path),
        expected_arm=arm,
        owner_uid=owner_uid,
        owner_gid=owner_gid,
        expected_request_id=request_id,
        expected_identity=None,
        missing_ok=True,
    )
    return removed is not None


def read_regular_at(parent_fd: int, name: str, *, label: str) -> PinnedBytes:
    _safe_name(name, label=label)
    try:
        fd = os.open(name, _FILE_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} is missing") from error
    except OSError as error:
        raise ValueError(f"{label} is symlinked or unsafe") from error
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} is not a regular file")
        payload, digest = _read_and_hash_fd(fd)
        if len(payload) != metadata.st_size:
            raise ValueError(f"{label} changed length while open")
        return PinnedBytes(
            payload=payload,
            sha256=digest,
            byte_count=metadata.st_size,
        )
    finally:
        os.close(fd)


def read_regular_path(
    path: str | Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> PinnedBytes:
    candidate = Path(path)
    parent_fd, name = _walk_parent(candidate, create=False)
    try:
        pinned = read_regular_at(parent_fd, name, label=label)
    finally:
        os.close(parent_fd)
    if expected_sha256 is not None:
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError("resume_sha256 must be 64 lowercase hexadecimal characters")
        if not hmac.compare_digest(pinned.sha256, expected_sha256):
            raise ValueError(f"{label} SHA-256 does not match resume_sha256")
    return pinned


class PinnedDirectory:
    def __init__(
        self,
        *,
        path: Path,
        parent_fd: int,
        name: str,
        fd: int,
    ) -> None:
        self.path = path
        self._parent_fd = parent_fd
        self._name = name
        self.fd = fd
        metadata = os.fstat(fd)
        if not stat.S_ISDIR(metadata.st_mode):
            self.close()
            raise ValueError(f"directory is unsafe: {path}")
        self._identity = (metadata.st_dev, metadata.st_ino)

    @classmethod
    def create(cls, path: str | Path) -> PinnedDirectory:
        candidate = Path(path)
        parent_fd, name = _walk_parent(candidate, create=True)
        try:
            os.mkdir(name, mode=0o755, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError as error:
            os.close(parent_fd)
            raise FileExistsError(
                f"fresh output path already exists: {candidate}"
            ) from error
        except OSError as error:
            os.close(parent_fd)
            raise ValueError(f"output directory could not be created safely: {candidate}") from error
        try:
            fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
            return cls(
                path=candidate,
                parent_fd=parent_fd,
                name=name,
                fd=fd,
            )
        except BaseException:
            os.close(parent_fd)
            raise

    @classmethod
    def open_existing(cls, path: str | Path, *, label: str) -> PinnedDirectory:
        candidate = Path(path)
        parent_fd, name = _walk_parent(candidate, create=False)
        try:
            fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        except FileNotFoundError as error:
            os.close(parent_fd)
            raise FileNotFoundError(f"{label} is missing") from error
        except OSError as error:
            os.close(parent_fd)
            raise ValueError(f"{label} is symlinked or unsafe") from error
        try:
            return cls(
                path=candidate,
                parent_fd=parent_fd,
                name=name,
                fd=fd,
            )
        except BaseException:
            os.close(parent_fd)
            raise

    def _verify_identity(self) -> None:
        try:
            metadata = os.stat(
                self._name,
                dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
        except OSError as error:
            raise ValueError(f"pinned directory path changed or became unsafe: {self.path}") from error
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != self._identity
        ):
            raise ValueError(f"pinned directory path changed or became unsafe: {self.path}")
        current_parent = -1
        current_fd = -1
        try:
            current_parent, current_name = _walk_parent(self.path, create=False)
            current_fd = os.open(
                current_name,
                _DIRECTORY_FLAGS,
                dir_fd=current_parent,
            )
            current_metadata = os.fstat(current_fd)
        except OSError as error:
            raise ValueError(
                f"pinned directory path changed or became unsafe: {self.path}"
            ) from error
        finally:
            if current_fd >= 0:
                os.close(current_fd)
            if current_parent >= 0:
                os.close(current_parent)
        if (
            not stat.S_ISDIR(current_metadata.st_mode)
            or (current_metadata.st_dev, current_metadata.st_ino) != self._identity
        ):
            raise ValueError(f"pinned directory path changed or became unsafe: {self.path}")

    def open_child(self, name: str, *, create: bool) -> PinnedDirectory:
        self._verify_identity()
        _safe_name(name, label="directory")
        if create:
            try:
                os.mkdir(name, mode=0o755, dir_fd=self.fd)
                os.fsync(self.fd)
            except FileExistsError:
                pass
            except OSError as error:
                raise ValueError(f"child directory could not be created safely: {name}") from error
        parent_fd = os.dup(self.fd)
        try:
            fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        except FileNotFoundError as error:
            os.close(parent_fd)
            raise FileNotFoundError(f"required directory is missing: {name}") from error
        except OSError as error:
            os.close(parent_fd)
            raise ValueError(f"child directory is symlinked or unsafe: {name}") from error
        try:
            return PinnedDirectory(
                path=self.path / name,
                parent_fd=parent_fd,
                name=name,
                fd=fd,
            )
        except BaseException:
            os.close(parent_fd)
            raise

    def read_regular(self, name: str, *, label: str) -> PinnedBytes:
        self._verify_identity()
        return read_regular_at(self.fd, name, label=label)

    def entries(self) -> tuple[str, ...]:
        self._verify_identity()
        return tuple(sorted(os.listdir(self.fd)))

    def entry_metadata(self, name: str, *, label: str) -> os.stat_result:
        self._verify_identity()
        _safe_name(name, label=label)
        try:
            return os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"{label} is missing") from error

    def quarantine_regular(self, name: str, quarantine_name: str) -> None:
        """Atomically rename one pinned regular artifact out of its live name."""

        self._verify_identity()
        _safe_name(name, label="artifact")
        _safe_name(quarantine_name, label="quarantine")
        try:
            source = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"artifact is missing: {name}") from error
        if not stat.S_ISREG(source.st_mode) or source.st_nlink != 1:
            raise ValueError(f"artifact is not an owned regular file: {name}")
        try:
            os.stat(quarantine_name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                f"quarantine artifact already exists: {quarantine_name}"
            )
        descriptor = -1
        try:
            descriptor = os.open(name, _FILE_FLAGS, dir_fd=self.fd)
            pinned = os.fstat(descriptor)
            if (
                not stat.S_ISREG(pinned.st_mode)
                or pinned.st_nlink != 1
                or (pinned.st_dev, pinned.st_ino)
                != (source.st_dev, source.st_ino)
            ):
                raise ValueError(f"artifact identity changed: {name}")
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        os.rename(
            name,
            quarantine_name,
            src_dir_fd=self.fd,
            dst_dir_fd=self.fd,
        )
        os.fsync(self.fd)

    def unlink_regular(
        self,
        name: str,
        *,
        expected: os.stat_result,
        label: str,
    ) -> None:
        """Remove one inspected regular entry only if its identity is unchanged."""

        self._verify_identity()
        _safe_name(name, label=label)
        current = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        expected_identity = (expected.st_dev, expected.st_ino)
        if (
            not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != expected_identity
            or current.st_nlink != expected.st_nlink
        ):
            raise ValueError(f"{label} identity changed: {name}")
        descriptor = -1
        try:
            descriptor = os.open(name, _FILE_FLAGS, dir_fd=self.fd)
            pinned = os.fstat(descriptor)
            if (
                not stat.S_ISREG(pinned.st_mode)
                or (pinned.st_dev, pinned.st_ino) != expected_identity
                or pinned.st_nlink != expected.st_nlink
            ):
                raise ValueError(f"{label} identity changed: {name}")
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        os.unlink(name, dir_fd=self.fd)
        os.fsync(self.fd)

    def write_atomic(
        self,
        name: str,
        writer: Callable[[BinaryIO], None],
        *,
        replace: bool,
    ) -> None:
        self._verify_identity()
        _safe_name(name, label="artifact")
        try:
            existing = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if not stat.S_ISREG(existing.st_mode):
                raise ValueError(f"output artifact is symlinked or unsafe: {name}")
            if not replace:
                raise FileExistsError(f"output artifact already exists: {name}")

        temporary = f".{name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
        temp_fd = -1
        try:
            temp_fd = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=self.fd,
            )
            with os.fdopen(temp_fd, "wb", closefd=True) as handle:
                temp_fd = -1
                writer(handle)
                handle.flush()
                os.fsync(handle.fileno())
            if replace:
                os.replace(
                    temporary,
                    name,
                    src_dir_fd=self.fd,
                    dst_dir_fd=self.fd,
                )
            else:
                os.link(
                    temporary,
                    name,
                    src_dir_fd=self.fd,
                    dst_dir_fd=self.fd,
                    follow_symlinks=False,
                )
                os.unlink(temporary, dir_fd=self.fd)
            os.fsync(self.fd)
        except BaseException:
            if temp_fd >= 0:
                os.close(temp_fd)
            try:
                os.unlink(temporary, dir_fd=self.fd)
            except FileNotFoundError:
                pass
            raise

    def write_bytes(self, name: str, payload: bytes, *, replace: bool) -> None:
        self.write_atomic(name, lambda handle: handle.write(payload), replace=replace)

    def append_bytes(self, name: str, payload: bytes) -> None:
        self._verify_identity()
        _safe_name(name, label="log")
        try:
            fd = os.open(
                name,
                os.O_WRONLY
                | os.O_APPEND
                | os.O_CREAT
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=self.fd,
            )
        except OSError as error:
            raise ValueError(f"log path is symlinked or unsafe: {name}") from error
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"log path is not a regular file: {name}")
            if metadata.st_nlink != 1:
                raise ValueError(
                    f"log path is hard-linked or not owned: {name}"
                )
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(self.fd)

    def close(self) -> None:
        for attribute in ("fd", "_parent_fd"):
            fd = getattr(self, attribute, -1)
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, attribute, -1)


class DurableOutput:
    def __init__(self, root: PinnedDirectory, snapshots: PinnedDirectory) -> None:
        self.root = root
        self.snapshots = snapshots

    @classmethod
    def create(cls, path: str | Path) -> DurableOutput:
        root = PinnedDirectory.create(path)
        try:
            snapshots = root.open_child("snapshots", create=True)
            return cls(root, snapshots)
        except BaseException:
            root.close()
            raise

    @classmethod
    def open_existing(cls, path: str | Path) -> DurableOutput:
        root = PinnedDirectory.open_existing(path, label="resume output directory")
        try:
            snapshots = root.open_child("snapshots", create=False)
            return cls(root, snapshots)
        except BaseException:
            root.close()
            raise

    def close(self) -> None:
        self.snapshots.close()
        self.root.close()


def write_atomic_path(
    path: str | Path,
    payload: bytes,
    *,
    label: str,
) -> None:
    candidate = Path(path)
    parent = PinnedDirectory.open_existing(
        candidate.parent,
        label=f"{label} parent directory",
    )
    try:
        parent.write_bytes(candidate.name, payload, replace=True)
    finally:
        parent.close()
