"""Deterministic, closed-world packaging for the AWS corpus builder."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import secrets
import stat
import subprocess
import tarfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml


ARCHIVE_NAME = "memorysplit-corpus-builder.tar.gz"
MANIFEST_NAME = "memorysplit-corpus-builder.manifest.json"
SHA256_NAME = f"{ARCHIVE_NAME}.sha256"
PACKAGE_FORMAT = "memorysplit-corpus-builder-package-v1"

REQUIRED_PREFIXES = (
    "cluster/aws/corpus_builder/",
    "cluster/profiles/",
    "configs/",
    "corpusgen/parallel/",
    "corpusgen/reasoning_v2/",
    "scripts/",
    "sources/",
    "vendor/tiktoken/",
)
REQUIRED_FILES = (
    "pyproject.toml",
    "requirements.txt",
    "scripts/build_parallel_corpus.py",
    "scripts/package_aws_corpus_builder.py",
)

_REQUIRED_PACKAGE_PATHS = frozenset(
    {
        "cluster/aws/corpus_builder/contracts.py",
        "cluster/aws/corpus_builder/package.py",
        "cluster/profiles/aws-i4i.16xlarge-corpus-v1.json",
        "configs/current-dataset-lock.json",
        "configs/reasoning-dataset-v2.json",
        "corpusgen/parallel/adapters.py",
        "corpusgen/parallel/catalog.py",
        "corpusgen/parallel/publication.py",
        "corpusgen/reasoning_v2/catalog.py",
        "corpusgen/reasoning_v2/source_lock.py",
        "requirements.txt",
        "scripts/build_parallel_corpus.py",
        "scripts/package_aws_corpus_builder.py",
        "sources/Wikidata-CC0-1.0.txt",
        "sources/current-dataset-licenses.json",
        "sources/wikidata5m.lock.json",
        "vendor/tiktoken/6c7ea1a7e38e3a7f062df639a5b80947f075ffe6",
        "vendor/tiktoken/6d1cbeee0f20b3d9449abfede4726ed8212e3aee",
    }
)
_REQUIRED_TEST_PATHS = frozenset(
    {
        "tests/test_aws_corpus_builder_contracts.py",
        "tests/test_parallel_corpus.py",
        "tests/test_reasoning_v2_catalog.py",
        "tests/test_reasoning_v2_source_lock.py",
    }
)

_DISPOSABLE_COMPONENTS = frozenset(
    {
        ".cache",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tiktoken_cache",
        "__pycache__",
        "cache",
        "caches",
        "checkpoint",
        "checkpoints",
        "output",
        "outputs",
        "pilot",
        "pilot-artifact",
        "pilot-artifacts",
    }
)
_PRIVATE_KEY_PATTERN = re.compile(
    rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY"
    rb"(?: BLOCK)?-----"
)
_AWS_ACCESS_KEY_PATTERN = re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
_SERVICE_TOKEN_PATTERN = re.compile(
    rb"\b(?:hf_[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{16,}|"
    rb"github_pat_[A-Za-z0-9_]{16,}|glpat-[A-Za-z0-9_-]{16,}|"
    rb"npm_[A-Za-z0-9]{16,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}|"
    rb"sk-ant-[A-Za-z0-9_-]{20,})\b"
)
_ASSIGNMENT_NAME_PATTERN = (
    rb"(?:AWS_(?:ACCESS_KEY_ID|SECRET_ACCESS_KEY|SESSION_TOKEN)|"
    rb"(?:HF|HUGGINGFACE|GITHUB|GH)_(?:TOKEN|API_KEY|API_TOKEN|PAT)|"
    rb"(?:OPENAI|ANTHROPIC|WANDB)_(?:API_KEY|TOKEN)|"
    rb"SLACK_(?:TOKEN|BOT_TOKEN)|"
    rb"(?:API|ACCESS|AUTH)_(?:KEY|TOKEN|SECRET)|"
    rb"CLIENT_SECRET|PRIVATE_KEY|PASSWORD)"
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    rb"(?i)\b"
    + _ASSIGNMENT_NAME_PATTERN
    + rb"\s*=\s*(?:[\"'][^\"'\r\n]{8,}[\"']|[^\s#;\"']{8,})"
)
_GENERIC_SECRET_ASSIGNMENT_PATTERN = re.compile(
    rb"\b(?:TOKEN|SECRET|PASSWORD|"
    rb"[A-Z][A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY|PRIVATE_KEY))\s*=\s*"
    rb"(?:[\"'][^\"'\r\n]{8,}[\"']|[^\s#;\"']{8,})"
)
_BEARER_TOKEN_PATTERN = re.compile(
    rb"(?i)\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._~+/-]{16,}"
)
_SECRET_PATTERNS = (
    _PRIVATE_KEY_PATTERN,
    _AWS_ACCESS_KEY_PATTERN,
    _SERVICE_TOKEN_PATTERN,
    _SECRET_ASSIGNMENT_PATTERN,
    _GENERIC_SECRET_ASSIGNMENT_PATTERN,
    _BEARER_TOKEN_PATTERN,
)
_SENSITIVE_FIELD_EXACT = frozenset(
    {"credential", "credentials", "secret", "token"}
)
_SENSITIVE_FIELD_AFFIXES = (
    "accesskey",
    "accesstoken",
    "apikey",
    "authtoken",
    "clientsecret",
    "credential",
    "credentials",
    "passphrase",
    "password",
    "privatekey",
    "secretkey",
    "sessiontoken",
)
_SENSITIVE_FIELD_VALUE_FORMS = (
    "apitokenvalue",
    "secretvalue",
    "tokenvalue",
)
_TEXT_SUFFIXES = frozenset(
    {
        ".json",
        ".lock",
        ".md",
        ".py",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
_OBJECT_ID_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


class PackageError(ValueError):
    """A fail-closed corpus builder packaging error."""


@dataclass(frozen=True)
class BuilderPackage:
    archive: Path
    manifest: Path
    sha256_file: Path
    revision: str
    sha256: str
    bytes: int
    members: tuple[str, ...]


@dataclass(frozen=True)
class _Tracked:
    path: str
    git_mode: str
    object_id: str

    @property
    def archive_mode(self) -> int:
        return 0o755 if self.git_mode == "100755" else 0o644

    @property
    def manifest_mode(self) -> str:
        return "0755" if self.git_mode == "100755" else "0644"


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _sanitized_git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
    )
    return environment


def _run_git(
    source_root: Path,
    *arguments: str,
    input_data: bytes | None = None,
) -> bytes:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(source_root),
                "--no-pager",
                "--no-replace-objects",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                *arguments,
            ],
            input=input_data,
            capture_output=True,
            check=False,
            cwd="/",
            env=_sanitized_git_environment(),
        )
    except OSError as error:
        raise PackageError("Git repository verification failed") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise PackageError(
            f"Git repository verification failed: {detail or arguments[0]}"
        )
    return completed.stdout


def _repository_root(source_root: Path) -> Path:
    requested = Path(os.path.abspath(os.fspath(source_root)))
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise PackageError("source root must be an existing directory") from error
    if requested != resolved or not resolved.is_dir():
        raise PackageError(
            "source root must be a real directory without symlink components"
        )
    try:
        reported = Path(
            _run_git(resolved, "rev-parse", "--show-toplevel")
            .decode("utf-8", errors="strict")
            .strip()
        )
    except UnicodeDecodeError as error:
        raise PackageError("Git repository root is not UTF-8") from error
    if Path(os.path.abspath(os.fspath(reported))) != resolved:
        raise PackageError("source root must be the exact Git worktree root")
    return resolved


def _head_revision(source_root: Path) -> str:
    try:
        revision = (
            _run_git(source_root, "rev-parse", "--verify", "HEAD^{commit}")
            .decode("ascii", errors="strict")
            .strip()
        )
    except UnicodeDecodeError as error:
        raise PackageError("Git HEAD is not an ASCII object ID") from error
    if _OBJECT_ID_PATTERN.fullmatch(revision) is None:
        raise PackageError("Git HEAD is not a full commit object ID")
    return revision


def _require_clean(source_root: Path) -> str:
    before = _head_revision(source_root)
    status = _run_git(
        source_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    after = _head_revision(source_root)
    if status:
        raise PackageError("corpus builder packages require a clean Git tree")
    if before != after:
        raise PackageError("Git HEAD changed during clean-tree verification")
    return before


def _assert_unchanged(source_root: Path, revision: str) -> None:
    before = _head_revision(source_root)
    status = _run_git(
        source_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    after = _head_revision(source_root)
    if before != revision or after != revision or status:
        raise PackageError("source Git tree changed during packaging")


def _portable_path(encoded: bytes) -> str:
    try:
        path = encoded.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PackageError("Git tree contains a non-UTF-8 path") from error
    pure = PurePosixPath(path)
    if (
        not path
        or path != unicodedata.normalize("NFC", path)
        or "\\" in path
        or pure.is_absolute()
        or pure.as_posix() != path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise PackageError(f"Git tree contains an unsafe path: {path!r}")
    return path


def _is_disposable(path: str) -> bool:
    return any(
        component.lower() in _DISPOSABLE_COMPONENTS
        for component in PurePosixPath(path).parts
    )


def _is_allowlisted(path: str) -> bool:
    return path in REQUIRED_FILES or path.startswith(REQUIRED_PREFIXES)


def _tracked_files(source_root: Path, revision: str) -> tuple[list[_Tracked], set[str]]:
    try:
        tree_id = (
            _run_git(
                source_root,
                "rev-parse",
                "--verify",
                f"{revision}^{{tree}}",
            )
            .decode("ascii", errors="strict")
            .strip()
        )
    except UnicodeDecodeError as error:
        raise PackageError("Git tree ID is not ASCII") from error
    if _OBJECT_ID_PATTERN.fullmatch(tree_id) is None:
        raise PackageError("revision does not identify a full Git tree")
    output = _run_git(
        source_root,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        tree_id,
    )
    selected: list[_Tracked] = []
    all_paths: set[str] = set()
    for raw in output.split(b"\0"):
        if not raw:
            continue
        try:
            metadata, encoded_path = raw.split(b"\t", 1)
            mode, kind, encoded_object_id = metadata.split()
            git_mode = mode.decode("ascii", errors="strict")
            object_kind = kind.decode("ascii", errors="strict")
            object_id = encoded_object_id.decode("ascii", errors="strict")
        except (UnicodeDecodeError, ValueError) as error:
            raise PackageError("Git tree listing is malformed") from error
        path = _portable_path(encoded_path)
        if path in all_paths:
            raise PackageError(f"Git tree repeats path: {path}")
        all_paths.add(path)
        if not _is_allowlisted(path) or _is_disposable(path):
            continue
        if git_mode == "120000":
            raise PackageError(f"tracked symlink is forbidden: {path}")
        if object_kind != "blob" or git_mode not in {"100644", "100755"}:
            raise PackageError(f"package member is not a regular file: {path}")
        if _OBJECT_ID_PATTERN.fullmatch(object_id) is None:
            raise PackageError(f"package member has an invalid object ID: {path}")
        selected.append(
            _Tracked(path=path, git_mode=git_mode, object_id=object_id)
        )
    selected.sort(key=lambda item: item.path.encode("utf-8"))
    return selected, all_paths


def _require_authorities(
    selected: list[_Tracked],
    all_paths: set[str],
) -> None:
    selected_paths = {item.path for item in selected}
    missing_package = sorted(_REQUIRED_PACKAGE_PATHS - selected_paths)
    if missing_package:
        raise PackageError(
            f"required package authority is missing: {missing_package[0]}"
        )
    missing_tests = sorted(_REQUIRED_TEST_PATHS - all_paths)
    if missing_tests:
        raise PackageError(f"required package test is missing: {missing_tests[0]}")


def _read_git_blobs(
    source_root: Path,
    tracked: list[_Tracked],
) -> dict[str, bytes]:
    queries = b"".join(
        item.object_id.encode("ascii") + b"\n" for item in tracked
    )
    output = _run_git(
        source_root,
        "cat-file",
        "--batch",
        input_data=queries,
    )
    cursor = 0
    blobs: dict[str, bytes] = {}
    try:
        for item in tracked:
            header_end = output.index(b"\n", cursor)
            header = output[cursor:header_end].decode("ascii").split()
            if (
                len(header) != 3
                or header[0] != item.object_id
                or header[1] != "blob"
            ):
                raise ValueError
            size = int(header[2])
            data_start = header_end + 1
            data_end = data_start + size
            if size < 0 or output[data_end : data_end + 1] != b"\n":
                raise ValueError
            blobs[item.path] = output[data_start:data_end]
            cursor = data_end + 1
    except (UnicodeDecodeError, ValueError) as error:
        raise PackageError("Git blob snapshot response is malformed") from error
    if cursor != len(output) or len(blobs) != len(tracked):
        raise PackageError("Git blob snapshot response is incomplete")
    return blobs


def _canonical_security_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _is_sensitive_field_name(value: str) -> bool:
    canonical = _canonical_security_name(value)
    if canonical in _SENSITIVE_FIELD_EXACT:
        return True
    if canonical.endswith(("secret", "token")):
        return True
    if any(form in canonical for form in _SENSITIVE_FIELD_VALUE_FORMS):
        return True
    return any(concept in canonical for concept in _SENSITIVE_FIELD_AFFIXES)


def _reject_sensitive_fields(value: object, *, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise PackageError(f"structured package member has non-string key: {path}")
            if _is_sensitive_field_name(key):
                raise PackageError(f"secret field is forbidden in package member: {path}")
            _reject_sensitive_fields(child, path=path)
    elif isinstance(value, list):
        for child in value:
            _reject_sensitive_fields(child, path=path)


def _unique_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PackageError(f"JSON package member repeats field: {key}")
        result[key] = value
    return result


def _scan_secret(path: str, data: bytes) -> None:
    lower_name = PurePosixPath(path).name.lower()
    security_components = (
        _canonical_security_name(component)
        for component in PurePosixPath(path).parts
    )
    if (
        lower_name == ".env"
        or lower_name.startswith("id_rsa")
        or lower_name.endswith((".key", ".pem", ".p12"))
        or "credential" in lower_name
        or any(
            "credential" in component
            or component in {"privatekey", "privatekeys", "secret", "secrets"}
            for component in security_components
        )
    ):
        raise PackageError(f"secret-like file is forbidden: {path}")
    if any(pattern.search(data) for pattern in _SECRET_PATTERNS):
        raise PackageError(f"secret material detected in package member: {path}")
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        try:
            text = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise PackageError(f"text package member is not UTF-8: {path}") from error
        if "\x00" in text:
            raise PackageError(f"text package member contains a NUL byte: {path}")
    else:
        text = ""
    if suffix == ".json":
        try:
            value = json.loads(text, object_pairs_hook=_unique_json_pairs)
        except PackageError:
            raise
        except json.JSONDecodeError as error:
            raise PackageError(f"JSON package member is malformed: {path}") from error
        _reject_sensitive_fields(value, path=path)
    elif suffix in {".yaml", ".yml"}:
        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise PackageError(f"YAML package member is malformed: {path}") from error
        _reject_sensitive_fields(value, path=path)


def _archive_bytes(
    tracked: list[_Tracked],
    blobs: dict[str, bytes],
) -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=raw,
        compresslevel=9,
        mtime=0,
    ) as compressed:
        with tarfile.open(
            fileobj=compressed,
            mode="w",
            format=tarfile.PAX_FORMAT,
        ) as archive:
            for item in tracked:
                data = blobs[item.path]
                info = tarfile.TarInfo(item.path)
                info.size = len(data)
                info.mode = item.archive_mode
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.type = tarfile.REGTYPE
                info.pax_headers = {}
                archive.addfile(info, io.BytesIO(data))
    return raw.getvalue()


def _verify_archive(
    archive_bytes: bytes,
    tracked: list[_Tracked],
    blobs: dict[str, bytes],
) -> None:
    expected = {item.path: item for item in tracked}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            infos = archive.getmembers()
            names = [info.name for info in infos]
            if names != [item.path for item in tracked] or len(names) != len(set(names)):
                raise PackageError("staged archive member list is not canonical")
            for info in infos:
                item = expected[info.name]
                extracted = archive.extractfile(info)
                data = None if extracted is None else extracted.read()
                if (
                    not info.isreg()
                    or stat.S_IMODE(info.mode) != item.archive_mode
                    or info.mtime != 0
                    or info.uid != 0
                    or info.gid != 0
                    or info.uname != ""
                    or info.gname != ""
                    or data != blobs[info.name]
                ):
                    raise PackageError("staged archive metadata or content is invalid")
    except PackageError:
        raise
    except (KeyError, OSError, tarfile.TarError) as error:
        raise PackageError("staged archive verification failed") from error


def _external_output(source_root: Path, output_dir: Path) -> Path:
    output = Path(os.path.abspath(os.fspath(output_dir)))
    try:
        within_source = os.path.commonpath((str(source_root), str(output))) == str(
            source_root
        )
    except ValueError:
        within_source = False
    if within_source:
        raise PackageError("package output must be outside the source worktree")
    return output


def _write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, 0o600)
        view = memoryview(data)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise PackageError("short write while staging package artifact")
            written += count
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
    except PackageError:
        raise
    except OSError as error:
        raise PackageError(f"cannot publish package artifact: {path.name}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_corpus_builder_package(
    source_root: Path,
    output_dir: Path,
) -> BuilderPackage:
    """Build deterministic artifacts from one clean committed Git snapshot."""

    source = _repository_root(Path(source_root))
    output = _external_output(source, Path(output_dir))
    revision = _require_clean(source)
    tracked, all_paths = _tracked_files(source, revision)
    _require_authorities(tracked, all_paths)
    blobs = _read_git_blobs(source, tracked)
    for item in tracked:
        _scan_secret(item.path, blobs[item.path])
    archive_payload = _archive_bytes(tracked, blobs)
    _verify_archive(archive_payload, tracked, blobs)
    archive_sha256 = hashlib.sha256(archive_payload).hexdigest()
    member_rows = [
        {
            "bytes": len(blobs[item.path]),
            "mode": item.manifest_mode,
            "object_id": item.object_id,
            "path": item.path,
            "sha256": hashlib.sha256(blobs[item.path]).hexdigest(),
        }
        for item in tracked
    ]
    manifest_payload = _canonical_json_bytes(
        {
            "archive": {
                "bytes": len(archive_payload),
                "path": ARCHIVE_NAME,
                "sha256": archive_sha256,
            },
            "format": PACKAGE_FORMAT,
            "members": member_rows,
            "revision": revision,
            "schema_version": 1,
        }
    )
    checksum_payload = f"{archive_sha256}  {ARCHIVE_NAME}\n".encode("ascii")
    _assert_unchanged(source, revision)

    try:
        output.mkdir(parents=True, exist_ok=True, mode=0o755)
    except OSError as error:
        raise PackageError("package output directory cannot be created") from error
    if output.is_symlink() or not output.is_dir():
        raise PackageError("package output must be a real directory")
    archive = output / ARCHIVE_NAME
    manifest = output / MANIFEST_NAME
    sha256_file = output / SHA256_NAME
    _write_atomic(archive, archive_payload)
    _write_atomic(manifest, manifest_payload)
    _write_atomic(sha256_file, checksum_payload)
    return BuilderPackage(
        archive=archive,
        manifest=manifest,
        sha256_file=sha256_file,
        revision=revision,
        sha256=archive_sha256,
        bytes=len(archive_payload),
        members=tuple(item.path for item in tracked),
    )
