"""Resumable, exact-version orchestration for production corpus builds."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Protocol, cast
from urllib.parse import urlsplit

from cluster.aws.corpus_builder.contracts import (
    CORPUS_BUCKET,
    CORPUS_KEY_PREFIX,
    PHASE_RECEIPT_FORMAT,
    PhaseReceipt,
    S3ObjectVersion,
    phase_receipt_from_bytes,
    phase_receipt_to_bytes,
)
from cluster.aws.corpus_builder.s3 import (
    PublicationError,
    S3Client,
    _head,
    _list_exact_key_history,
    _record_from_head,
    _response_metadata,
    download_exact_object,
    publish_exact_file,
    publish_phase_receipt,
    verify_exact_object,
)
from corpusgen.parallel import (
    InputCatalog,
    ParallelBuildConfig,
    Renderer,
    build_parallel_corpus,
    verify_parallel_corpus,
)
from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.parallel.safeio import atomic_rename_noreplace, fsync_directory
from corpusgen.reasoning_v2.contracts import load_recipe
from corpusgen.reasoning_v2.source_lock import (
    SourceLock,
    load_source_lock,
    stage_source_lock,
    verify_source_tree,
)
from corpusgen.reasoning_v2.wikidata_source import (
    WikidataDerivedView,
    build_wikidata_derived_view,
    verify_wikidata_derived_view,
)


PHASES = (
    "source-stage",
    "wikidata-view",
    "catalog",
    "render-pack",
    "local-verify",
    "s3-publish",
    "cleanroom-verify",
)
_LOCAL_OUTPUT_PHASES = frozenset(PHASES[:5])

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SIDECAR_NAME_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_KMS_KEY_ARN_RE = re.compile(
    r"arn:aws:kms:us-east-1:056956104102:key/"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_RECIPE_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "reasoning-dataset-v2.json"
)


@dataclass(frozen=True)
class VerifiedProductionInputs:
    catalog: InputCatalog
    renderer: Renderer
    sidecars: Mapping[str, Path]

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, InputCatalog):
            raise TypeError("catalog must be a parallel InputCatalog")
        renderer_id = getattr(self.renderer, "renderer_id", None)
        if not isinstance(renderer_id, str) or not renderer_id:
            raise TypeError("renderer must expose a non-empty renderer_id")
        if not isinstance(self.sidecars, Mapping):
            raise TypeError("sidecars must be a mapping")
        frozen: dict[str, Path] = {}
        for name, path in sorted(self.sidecars.items()):
            if not isinstance(name, str) or _SIDECAR_NAME_RE.fullmatch(name) is None:
                raise ValueError("sidecar names must be canonical lowercase identifiers")
            if not isinstance(path, Path):
                raise TypeError("sidecar paths must be pathlib.Path values")
            frozen[name] = path
        object.__setattr__(self, "sidecars", MappingProxyType(frozen))


class ProductionInputsUnavailable(RuntimeError):
    """The verified reasoning-v2 to parallel adapter is not installed."""


def _production_inputs_from_verified_view(
    *,
    source_lock: SourceLock,
    source_root: Path,
    derived_root: Path,
    view: WikidataDerivedView,
    expected_generator_commit: str,
) -> VerifiedProductionInputs:
    """Keep the evolving production catalog adapter behind one boundary."""

    from corpusgen.parallel import adapters

    factory = getattr(adapters, "verified_reasoning_v2_inputs", None)
    if not callable(factory):
        raise ProductionInputsUnavailable(
            "corpusgen.parallel.adapters must provide "
            "verified_reasoning_v2_inputs after all production lane adapters land"
        )
    value = factory(
        source_lock=source_lock,
        source_root=source_root,
        derived_root=derived_root,
        wikidata_view=view,
        expected_generator_commit=expected_generator_commit,
    )
    if isinstance(value, VerifiedProductionInputs):
        return value
    try:
        return VerifiedProductionInputs(
            catalog=value.catalog,
            renderer=value.renderer,
            sidecars=value.sidecars,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ProductionInputsUnavailable(
            "verified reasoning-v2 adapter returned an invalid production input set"
        ) from error


def load_verified_production_inputs(
    *,
    source_lock_path: Path,
    source_root: Path,
    derived_root: Path,
    expected_generator_commit: str,
) -> VerifiedProductionInputs:
    """Verify source and derived authorities before constructing adapters."""

    for name, value in (
        ("source_lock_path", source_lock_path),
        ("source_root", source_root),
        ("derived_root", derived_root),
    ):
        if not isinstance(value, Path):
            raise TypeError(f"{name} must be a pathlib.Path")
    if (
        not isinstance(expected_generator_commit, str)
        or _COMMIT_RE.fullmatch(expected_generator_commit) is None
    ):
        raise ValueError("expected_generator_commit must be lowercase 40-hex")

    lock = load_source_lock(
        source_lock_path,
        expected_generator_commit=expected_generator_commit,
    )
    verify_source_tree(
        lock,
        source_root,
        expected_generator_commit=expected_generator_commit,
    )
    built = build_wikidata_derived_view(
        source_lock_path,
        source_root,
        derived_root,
        expected_generator_commit=expected_generator_commit,
    )
    view = verify_wikidata_derived_view(
        source_lock_path,
        source_root,
        built.root,
        expected_generator_commit=expected_generator_commit,
    )
    if (
        view.receipt.source_lock_sha256 != lock.sha256
        or view.receipt.generator_commit != expected_generator_commit
    ):
        raise ValueError("verified Wikidata view disagrees with source authority")
    result = _production_inputs_from_verified_view(
        source_lock=lock,
        source_root=source_root,
        derived_root=derived_root,
        view=view,
        expected_generator_commit=expected_generator_commit,
    )
    for name, path in result.sidecars.items():
        try:
            metadata = path.lstat()
        except OSError as error:
            raise ValueError(f"production sidecar is missing: {name}") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"production sidecar is unsafe: {name}")
    return result


class CommandRunner(Protocol):
    def run(self, phase: str, action: Callable[[], object]) -> object: ...


@dataclass(frozen=True)
class InlineCommandRunner:
    """Execute phase actions in the current process."""

    def run(self, phase: str, action: Callable[[], object]) -> object:
        if phase not in PHASES:
            raise ValueError(f"unknown corpus build phase: {phase}")
        return action()


@dataclass(frozen=True)
class CorpusBuildRequest:
    build_id: str
    package_sha256: str
    source_lock_path: Path
    source_lock_sha256: str
    work_root: Path
    output_root: Path
    workers: int
    shard_count: int
    bucket: str
    prefix: str
    kms_key_arn: str

    def __post_init__(self) -> None:
        for name in ("build_id", "package_sha256", "source_lock_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256")
        for name in ("source_lock_path", "work_root", "output_root"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                raise TypeError(f"{name} must be a pathlib.Path")
            if not value.is_absolute():
                raise ValueError(f"{name} must be absolute")
        for name in ("workers", "shard_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.bucket != CORPUS_BUCKET:
            raise ValueError("bucket must equal the frozen corpus bucket")
        if self.prefix != CORPUS_KEY_PREFIX:
            raise ValueError("prefix must equal the frozen corpus key prefix")
        if (
            not isinstance(self.kms_key_arn, str)
            or _KMS_KEY_ARN_RE.fullmatch(self.kms_key_arn) is None
        ):
            raise ValueError("kms_key_arn must be the dedicated corpus KMS key ARN")


@dataclass(frozen=True)
class _Artifact:
    path: Path
    relative_key: str


@dataclass(frozen=True)
class _PhaseOutcome:
    artifacts: tuple[_Artifact, ...] = ()
    objects: tuple[S3ObjectVersion, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.artifacts, tuple) or not isinstance(self.objects, tuple):
            raise TypeError("phase outcome collections must be tuples")
        if bool(self.artifacts) == bool(self.objects):
            raise ValueError("phase outcome must contain artifacts or exact objects")


@dataclass
class _BuildContext:
    request: CorpusBuildRequest
    s3: S3Client
    receipts: dict[str, PhaseReceipt] = field(default_factory=dict)
    receipt_versions: dict[str, S3ObjectVersion] = field(default_factory=dict)
    production_inputs: VerifiedProductionInputs | None = None
    source_lock: SourceLock | None = None
    attempt_phase: str | None = None
    attempt_container: Path | None = None
    attempt_output: Path | None = None


def _file_commitment(path: Path, description: str) -> tuple[int, str]:
    try:
        before = path.lstat()
    except OSError as error:
        raise PublicationError(f"{description} is missing") from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise PublicationError(f"{description} is not a regular file")
    descriptor = -1
    digest = hashlib.sha256()
    byte_count = 0
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        opened = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            byte_count += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        named = path.lstat()
    except OSError as error:
        raise PublicationError(f"{description} cannot be read") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    def identity(row: os.stat_result) -> tuple[int, ...]:
        return (
            row.st_dev,
            row.st_ino,
            row.st_mode,
            row.st_nlink,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
        )

    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or identity(before) != identity(opened)
        or identity(opened) != identity(after)
        or identity(after) != identity(named)
        or byte_count != before.st_size
    ):
        raise PublicationError(f"{description} changed while reading")
    return byte_count, digest.hexdigest()


def _validate_request_authority(request: CorpusBuildRequest) -> None:
    count, digest = _file_commitment(request.source_lock_path, "source lock")
    if count <= 0 or digest != request.source_lock_sha256:
        raise PublicationError("source lock SHA-256 or byte count differs")
    try:
        work_metadata = request.work_root.lstat()
    except OSError as error:
        raise PublicationError("private work root is missing") from error
    if (
        stat.S_ISLNK(work_metadata.st_mode)
        or not stat.S_ISDIR(work_metadata.st_mode)
        or stat.S_IMODE(work_metadata.st_mode) != 0o700
    ):
        raise PublicationError("private work root must be a real owner-only directory")
    if request.output_root.is_symlink():
        raise PublicationError("output root must not be a symlink")


def _build_prefix(request: CorpusBuildRequest) -> str:
    return f"{request.prefix}/{request.build_id}"


def _request_seed(request: CorpusBuildRequest) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "build_id": request.build_id,
                "bucket": request.bucket,
                "kms_key_arn": request.kms_key_arn,
                "package_sha256": request.package_sha256,
                "prefix": request.prefix,
                "schema": "memorysplit-aws-corpus-driver-v2",
                "shard_count": request.shard_count,
                "source_lock_sha256": request.source_lock_sha256,
            }
        )
    ).hexdigest()


def _phase_receipt_key(
    request: CorpusBuildRequest,
    phase: str,
    dependency_sha256: str,
) -> str:
    index = PHASES.index(phase)
    return (
        f"{_build_prefix(request)}/receipts/phases/"
        f"{index:02d}-{phase}-{dependency_sha256}.json"
    )


def _safe_relative(value: str, description: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise PublicationError(f"{description} is not a safe relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PublicationError(f"{description} is not a safe relative path")
    return path


def _phase_local_root(request: CorpusBuildRequest, phase: str) -> Path:
    if phase == "render-pack":
        return request.output_root
    return request.work_root / "phases" / phase


def _phase_action_root(context: _BuildContext, phase: str) -> Path:
    if phase in _LOCAL_OUTPUT_PHASES:
        if context.attempt_phase != phase or context.attempt_output is None:
            raise PublicationError(f"{phase} lacks an active private phase attempt")
        return context.attempt_output
    return _phase_local_root(context.request, phase)


def _begin_phase_attempt(context: _BuildContext, phase: str) -> None:
    if phase not in _LOCAL_OUTPUT_PHASES:
        return
    if (
        context.attempt_phase is not None
        or context.attempt_container is not None
        or context.attempt_output is not None
    ):
        raise PublicationError("another private phase attempt is already active")
    stable = _phase_local_root(context.request, phase)
    stable.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    container = Path(
        tempfile.mkdtemp(
            prefix=f".{stable.name}.{phase}.attempt-",
            dir=stable.parent,
        )
    )
    os.chmod(container, 0o700)
    context.attempt_phase = phase
    context.attempt_container = container
    context.attempt_output = container / "output"


def _clear_phase_attempt(context: _BuildContext) -> None:
    context.attempt_phase = None
    context.attempt_container = None
    context.attempt_output = None


def _tree_commitments(
    root: Path,
    *,
    description: str,
) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (
            path.relative_to(root).as_posix(),
            *_file_commitment(path, description),
        )
        for path in _walk_regular_files(root)
    )


def _promote_phase_output(
    context: _BuildContext,
    phase: str,
    outcome: _PhaseOutcome,
) -> _PhaseOutcome:
    if phase not in _LOCAL_OUTPUT_PHASES:
        return outcome
    source = context.attempt_output
    container = context.attempt_container
    if (
        context.attempt_phase != phase
        or source is None
        or container is None
    ):
        raise PublicationError(f"{phase} lacks a private phase attempt")
    if outcome.objects:
        raise PublicationError(f"{phase} returned exact objects instead of local output")

    expected_artifacts: set[Path] = set()
    for artifact in outcome.artifacts:
        relative = _safe_relative(artifact.relative_key, "phase artifact key")
        expected_prefix = ("phase-artifacts", phase)
        if relative.parts[:2] != expected_prefix or len(relative.parts) < 3:
            raise PublicationError(
                f"{phase} artifact must use phase-artifacts/{phase}/<path>"
            )
        suffix = PurePosixPath(*relative.parts[2:])
        expected = source.joinpath(*suffix.parts)
        if artifact.path != expected:
            raise PublicationError(
                f"{phase} artifact path is outside its private attempt"
            )
        count, _digest = _file_commitment(artifact.path, f"{phase} attempt artifact")
        if count <= 0:
            raise PublicationError(f"{phase} artifact must not be empty")
        expected_artifacts.add(artifact.path)

    actual_artifacts = set(_walk_regular_files(source))
    if actual_artifacts != expected_artifacts:
        raise PublicationError(f"{phase} private attempt inventory differs")

    stable = _phase_local_root(context.request, phase)
    destination_parent = stable.parent
    source_parent_fd = -1
    destination_parent_fd = -1
    promoted = False
    try:
        source_parent_fd = os.open(
            container,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        destination_parent_fd = os.open(
            destination_parent,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            atomic_rename_noreplace(
                source_parent_fd,
                source.name,
                destination_parent_fd,
                stable.name,
            )
        except FileExistsError:
            pass
        else:
            fsync_directory(destination_parent_fd)
            promoted = True
    except OSError as error:
        raise PublicationError(f"{phase} output cannot be atomically published") from error
    finally:
        if source_parent_fd >= 0:
            os.close(source_parent_fd)
        if destination_parent_fd >= 0:
            os.close(destination_parent_fd)

    if promoted:
        try:
            shutil.rmtree(container)
        except OSError as error:
            raise PublicationError(
                f"{phase} private attempt container cannot be removed"
            ) from error
    else:
        attempted = _tree_commitments(
            source,
            description=f"{phase} private attempt artifact",
        )
        persisted = _tree_commitments(
            stable,
            description=f"{phase} persistent artifact",
        )
        if attempted != persisted:
            raise PublicationError(
                f"{phase} persistent output conflicts with verified attempt"
            )
        try:
            shutil.rmtree(container)
        except OSError as error:
            raise PublicationError(
                f"{phase} duplicate private attempt cannot be removed"
            ) from error

    remapped = tuple(
        _Artifact(
            path=stable / artifact.path.relative_to(source),
            relative_key=artifact.relative_key,
        )
        for artifact in outcome.artifacts
    )
    return _PhaseOutcome(artifacts=remapped)


def _artifact_key(request: CorpusBuildRequest, phase: str, artifact: _Artifact) -> str:
    relative = _safe_relative(artifact.relative_key, "phase artifact key")
    if phase == "s3-publish":
        expected_prefix = ("corpus",)
        if relative.parts[:1] != expected_prefix or len(relative.parts) < 2:
            raise PublicationError("S3 publication artifact must use corpus/<path>")
        suffix = PurePosixPath(*relative.parts[1:]).as_posix()
        if relative.parts[1] in {"receipts", "phase-artifacts"}:
            raise PublicationError("corpus artifact collides with control namespace")
        return f"{_build_prefix(request)}/{suffix}"
    expected = ("phase-artifacts", phase)
    if relative.parts[:2] != expected or len(relative.parts) < 3:
        raise PublicationError(
            f"{phase} artifact must use phase-artifacts/{phase}/<path>"
        )
    return f"{_build_prefix(request)}/{relative.as_posix()}"


def _artifact_expected_local_path(
    request: CorpusBuildRequest,
    phase: str,
    obj: S3ObjectVersion,
) -> Path | None:
    key = urlsplit(obj.uri).path.removeprefix("/")
    base = f"{_build_prefix(request)}/"
    if not key.startswith(base):
        raise PublicationError("phase object escapes the build prefix")
    relative = _safe_relative(key[len(base) :], "phase object key")
    if phase in {"s3-publish", "cleanroom-verify"}:
        if relative.parts[0] in {"receipts", "phase-artifacts"}:
            raise PublicationError("published corpus object uses a control namespace")
        return None
    expected = ("phase-artifacts", phase)
    if relative.parts[:2] != expected or len(relative.parts) < 3:
        raise PublicationError("phase receipt object uses the wrong phase namespace")
    suffix = PurePosixPath(*relative.parts[2:])
    return _phase_local_root(request, phase).joinpath(*suffix.parts)


def _walk_regular_files(root: Path) -> tuple[Path, ...]:
    try:
        metadata = root.lstat()
    except OSError as error:
        raise PublicationError(f"phase output root is missing: {root}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise PublicationError(f"phase output root is unsafe: {root}")
    files: list[Path] = []
    try:
        for current, directories, names in os.walk(root, followlinks=False):
            current_path = Path(current)
            directories.sort()
            names.sort()
            for name in tuple(directories):
                entry = current_path / name
                entry_metadata = entry.lstat()
                if (
                    stat.S_ISLNK(entry_metadata.st_mode)
                    or not stat.S_ISDIR(entry_metadata.st_mode)
                ):
                    raise PublicationError(
                        f"phase output contains an unsafe directory: {entry}"
                    )
            for name in names:
                entry = current_path / name
                entry_metadata = entry.lstat()
                if (
                    stat.S_ISLNK(entry_metadata.st_mode)
                    or not stat.S_ISREG(entry_metadata.st_mode)
                ):
                    raise PublicationError(
                        f"phase output contains an unsafe file: {entry}"
                    )
                files.append(entry)
    except OSError as error:
        raise PublicationError(f"phase output inventory cannot be read: {root}") from error
    return tuple(files)


def _tree_outcome(root: Path, *, phase: str, corpus: bool = False) -> _PhaseOutcome:
    files = _walk_regular_files(root)
    if not files:
        raise PublicationError(f"{phase} produced no files")
    prefix = "corpus" if corpus else f"phase-artifacts/{phase}"
    return _PhaseOutcome(
        artifacts=tuple(
            _Artifact(
                path=path,
                relative_key=f"{prefix}/{path.relative_to(root).as_posix()}",
            )
            for path in files
        )
    )


def _read_existing_receipt(
    context: _BuildContext,
    *,
    key: str,
) -> tuple[PhaseReceipt, S3ObjectVersion, bytes] | None:
    request = context.request
    versions, delete_markers = _list_exact_key_history(
        context.s3,
        bucket=request.bucket,
        key=key,
    )
    if delete_markers:
        raise PublicationError("phase receipt key has a conflicting delete marker")
    if not versions:
        return None
    if len(versions) != 1:
        raise PublicationError("phase receipt key has conflicting version history")
    version_id = versions[0].get("VersionId")
    if not isinstance(version_id, str) or not version_id:
        raise PublicationError("phase receipt history has an invalid version ID")
    head = _head(
        context.s3,
        bucket=request.bucket,
        key=key,
        version_id=version_id,
    )
    metadata = _response_metadata(head.get("Metadata"))
    digest = metadata.get("sha256")
    if (
        set(metadata) != {"sha256"}
        or not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
    ):
        raise PublicationError("phase receipt exact metadata is malformed")
    reference = _record_from_head(
        uri=f"s3://{request.bucket}/{key}",
        requested_version_id=version_id,
        expected_sha256=digest,
        response=head,
    )
    if reference.kms_key_arn != request.kms_key_arn:
        raise PublicationError("phase receipt KMS key differs from request")
    verify_exact_object(context.s3, reference)

    cache = request.work_root / ".receipt-cache"
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = cache / f"{secrets.token_hex(16)}.json"
    try:
        download_exact_object(context.s3, reference, destination)
        payload = destination.read_bytes()
    except OSError as error:
        raise PublicationError("downloaded phase receipt cannot be read") from error
    finally:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
    if (
        len(payload) != reference.bytes
        or hashlib.sha256(payload).hexdigest() != reference.sha256
    ):
        raise PublicationError("downloaded phase receipt bytes changed")
    try:
        receipt = phase_receipt_from_bytes(payload)
    except ValueError as error:
        raise PublicationError("downloaded phase receipt is invalid") from error
    return receipt, reference, payload


def _validate_receipt_identity(
    request: CorpusBuildRequest,
    phase: str,
    receipt: PhaseReceipt,
) -> None:
    if (
        receipt.build_id != request.build_id
        or receipt.phase != phase
        or receipt.package_sha256 != request.package_sha256
        or receipt.source_lock_sha256 != request.source_lock_sha256
    ):
        raise PublicationError(f"{phase} receipt identity differs from request")
    for obj in receipt.objects:
        if obj.kms_key_arn != request.kms_key_arn:
            raise PublicationError(f"{phase} receipt object KMS key differs")
        _artifact_expected_local_path(request, phase, obj)


def _restore_or_verify_local_phase(
    context: _BuildContext,
    phase: str,
    objects: tuple[S3ObjectVersion, ...],
) -> None:
    if phase in {"s3-publish", "cleanroom-verify"}:
        return
    root = _phase_local_root(context.request, phase)
    expected_paths: set[Path] = set()
    for obj in objects:
        destination = _artifact_expected_local_path(context.request, phase, obj)
        assert destination is not None
        expected_paths.add(destination)
        verify_exact_object(context.s3, obj)
        if destination.exists() or destination.is_symlink():
            count, digest = _file_commitment(
                destination,
                f"local {phase} dependency",
            )
            if count != obj.bytes or digest != obj.sha256:
                raise PublicationError(f"local {phase} dependency drift")
        else:
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            download_exact_object(context.s3, obj, destination)
    actual_paths = set(_walk_regular_files(root))
    if actual_paths != expected_paths:
        raise PublicationError(f"local {phase} dependency inventory drift")


def _publish_artifacts(
    context: _BuildContext,
    phase: str,
    outcome: _PhaseOutcome,
) -> tuple[S3ObjectVersion, ...]:
    request = context.request
    if outcome.objects:
        if phase != "cleanroom-verify":
            raise PublicationError(
                "only clean-room verification may reuse authoritative objects"
            )
        for obj in outcome.objects:
            if obj.kms_key_arn != request.kms_key_arn:
                raise PublicationError(f"{phase} object KMS key differs")
            _artifact_expected_local_path(request, phase, obj)
            verify_exact_object(context.s3, obj)
        return tuple(sorted(outcome.objects, key=lambda item: item.uri))

    keyed: list[tuple[str, _Artifact]] = []
    for artifact in outcome.artifacts:
        if not isinstance(artifact, _Artifact):
            raise TypeError("phase artifacts must be _Artifact values")
        key = _artifact_key(request, phase, artifact)
        expected_local = (
            request.output_root
            / PurePosixPath(artifact.relative_key).relative_to("corpus")
            if phase == "s3-publish"
            else _phase_local_root(request, phase)
            / PurePosixPath(artifact.relative_key).relative_to(
                PurePosixPath("phase-artifacts") / phase,
            )
        )
        if artifact.path != expected_local:
            raise PublicationError(f"{phase} artifact local path is not canonical")
        count, _digest = _file_commitment(artifact.path, f"{phase} artifact")
        if count <= 0:
            raise PublicationError(f"{phase} artifact must not be empty")
        keyed.append((key, artifact))
    keys = [key for key, _artifact in keyed]
    if len(keys) != len(set(keys)):
        raise PublicationError(f"{phase} artifacts repeat an S3 key")
    records = tuple(
        publish_exact_file(
            context.s3,
            bucket=request.bucket,
            key=key,
            path=artifact.path,
            kms_key_arn=request.kms_key_arn,
            metadata={"phase": phase},
        )
        for key, artifact in sorted(keyed, key=lambda item: item[0])
    )
    return tuple(sorted(records, key=lambda item: item.uri))


def _load_context_source_lock(context: _BuildContext) -> SourceLock:
    if context.source_lock is not None:
        return context.source_lock
    try:
        value = json.loads(context.request.source_lock_path.read_bytes())
        expected_commit = value["generator_commit"]
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise PublicationError("source lock cannot provide generator authority") from error
    if (
        not isinstance(expected_commit, str)
        or _COMMIT_RE.fullmatch(expected_commit) is None
    ):
        raise PublicationError("source lock generator commit is invalid")
    try:
        lock = load_source_lock(
            context.request.source_lock_path,
            expected_generator_commit=expected_commit,
        )
    except ValueError as error:
        raise PublicationError("source lock verification failed") from error
    if lock.sha256 != context.request.source_lock_sha256:
        raise PublicationError("source lock object differs from request authority")
    context.source_lock = lock
    return lock


def _source_root(context: _BuildContext) -> Path:
    lock = _load_context_source_lock(context)
    return _phase_local_root(context.request, "source-stage") / "sources" / lock.sha256


def _load_context_production_inputs(
    context: _BuildContext,
) -> VerifiedProductionInputs:
    if context.production_inputs is not None:
        return context.production_inputs
    lock = _load_context_source_lock(context)
    inputs = load_verified_production_inputs(
        source_lock_path=context.request.source_lock_path,
        source_root=_source_root(context),
        derived_root=_phase_local_root(context.request, "wikidata-view"),
        expected_generator_commit=lock.generator_commit,
    )
    context.production_inputs = inputs
    return inputs


def _production_inputs_manifest(
    inputs: VerifiedProductionInputs,
) -> dict[str, object]:
    return {
        "catalog_sha256": inputs.catalog.sha256,
        "renderer_id": inputs.renderer.renderer_id,
        "sidecars": {
            name: _file_commitment(path, f"production sidecar {name}")[1]
            for name, path in inputs.sidecars.items()
        },
    }


def _authenticate_catalog_inputs(
    context: _BuildContext,
    *,
    root: Path | None = None,
) -> VerifiedProductionInputs:
    catalog_root = (
        _phase_local_root(context.request, "catalog")
        if root is None
        else root
    )
    path = catalog_root / "production-inputs.json"
    try:
        payload = path.read_bytes()
        recorded = json.loads(payload)
        canonical = canonical_json_bytes(recorded)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise PublicationError("catalog production input manifest cannot be read") from error
    if (
        not isinstance(recorded, dict)
        or set(recorded) != {"catalog_sha256", "renderer_id", "sidecars"}
        or canonical != payload
    ):
        raise PublicationError("catalog production input manifest is invalid")
    current = _load_context_production_inputs(context)
    if recorded != _production_inputs_manifest(current):
        raise PublicationError("catalog production inputs differ from recorded authority")
    return current


def _integer_lane_weights() -> tuple[tuple[str, int], ...]:
    recipe = load_recipe(_RECIPE_PATH)
    denominator = math.lcm(*(share.denominator for _lane, share in recipe.lane_shares))
    return tuple(
        (lane, share.numerator * (denominator // share.denominator))
        for lane, share in recipe.lane_shares
    )


def _parallel_config(request: CorpusBuildRequest) -> ParallelBuildConfig:
    recipe = load_recipe(_RECIPE_PATH)
    return ParallelBuildConfig(
        lane_weights=_integer_lane_weights(),
        update_tokens=recipe.targets_per_update,
        shard_count=request.shard_count,
        allow_fewer_shards=False,
    )


def _write_local_receipt(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = canonical_json_bytes(dict(value))
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise PublicationError(f"conflicting local verification receipt: {path}")


def _execute_cleanroom(context: _BuildContext) -> _PhaseOutcome:
    publish_receipt = context.receipts.get("s3-publish")
    if publish_receipt is None:
        raise PublicationError("clean-room verification lacks S3 publication receipt")
    request = context.request
    parent = request.work_root / "cleanrooms"
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    room = Path(tempfile.mkdtemp(prefix="cleanroom-", dir=parent))
    os.chmod(room, 0o700)
    expected_paths: dict[Path, S3ObjectVersion] = {}
    base = f"{_build_prefix(request)}/"
    for obj in publish_receipt.objects:
        key = urlsplit(obj.uri).path.removeprefix("/")
        if not key.startswith(base):
            raise PublicationError("clean-room object escapes build prefix")
        relative = _safe_relative(key[len(base) :], "clean-room object path")
        if relative.parts[0] in {"receipts", "phase-artifacts"}:
            raise PublicationError("clean-room object uses a control namespace")
        destination = room.joinpath(*relative.parts)
        if destination in expected_paths:
            raise PublicationError("clean-room receipt repeats a destination")
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        download_exact_object(context.s3, obj, destination)
        expected_paths[destination] = obj
    if room / "receipt.json" not in expected_paths:
        raise PublicationError("S3 publication omits canonical receipt.json")
    if set(_walk_regular_files(room)) != set(expected_paths):
        raise PublicationError("clean-room download inventory differs from receipt")
    verified = verify_parallel_corpus(room, expected_build_id=request.build_id)
    if not isinstance(verified, Mapping) or verified.get("build_id") != request.build_id:
        raise PublicationError("clean-room verifier returned the wrong build ID")
    for path, obj in expected_paths.items():
        count, digest = _file_commitment(path, "clean-room corpus object")
        if count != obj.bytes or digest != obj.sha256:
            raise PublicationError("clean-room verifier changed a corpus object")
        verify_exact_object(context.s3, obj)
    if set(_walk_regular_files(room)) != set(expected_paths):
        raise PublicationError("clean-room verifier changed the corpus inventory")
    return _PhaseOutcome(objects=publish_receipt.objects)


def _execute_phase(context: _BuildContext, phase: str) -> _PhaseOutcome:
    request = context.request
    if phase == "source-stage":
        lock = _load_context_source_lock(context)
        output = _phase_action_root(context, phase)
        staged = stage_source_lock(
            lock,
            request.work_root / "source-input",
            output,
            expected_generator_commit=lock.generator_commit,
        )
        verify_source_tree(
            lock,
            staged,
            expected_generator_commit=lock.generator_commit,
        )
        return _tree_outcome(output, phase=phase)

    if phase == "wikidata-view":
        lock = _load_context_source_lock(context)
        output = _phase_action_root(context, phase)
        view = build_wikidata_derived_view(
            request.source_lock_path,
            _source_root(context),
            output,
            expected_generator_commit=lock.generator_commit,
        )
        verify_wikidata_derived_view(
            request.source_lock_path,
            _source_root(context),
            view.root,
            expected_generator_commit=lock.generator_commit,
        )
        return _tree_outcome(output, phase=phase)

    if phase == "catalog":
        inputs = _load_context_production_inputs(context)
        output = _phase_action_root(context, phase)
        _write_local_receipt(
            output / "production-inputs.json",
            _production_inputs_manifest(inputs),
        )
        return _tree_outcome(output, phase=phase)

    if phase == "render-pack":
        inputs = _authenticate_catalog_inputs(context)
        output = _phase_action_root(context, phase)
        receipt = build_parallel_corpus(
            inputs.catalog,
            inputs.renderer,
            _parallel_config(request),
            output,
            workers=request.workers,
            sidecar_paths=dict(inputs.sidecars),
        )
        if receipt.get("build_id") != request.build_id:
            raise PublicationError("rendered corpus build ID differs from request")
        return _tree_outcome(output, phase=phase)

    if phase == "local-verify":
        receipt = verify_parallel_corpus(
            request.output_root,
            expected_build_id=request.build_id,
        )
        if receipt.get("build_id") != request.build_id:
            raise PublicationError("local verifier returned the wrong build ID")
        output = _phase_action_root(context, phase)
        _write_local_receipt(
            output / "verification.json",
            cast(Mapping[str, object], receipt),
        )
        return _tree_outcome(output, phase=phase)

    if phase == "s3-publish":
        receipt = verify_parallel_corpus(
            request.output_root,
            expected_build_id=request.build_id,
        )
        if receipt.get("build_id") != request.build_id:
            raise PublicationError("pre-publication verifier returned the wrong build ID")
        return _tree_outcome(request.output_root, phase=phase, corpus=True)

    if phase == "cleanroom-verify":
        return _execute_cleanroom(context)

    raise ValueError(f"unknown corpus build phase: {phase}")


def _receipt(
    request: CorpusBuildRequest,
    phase: str,
    objects: tuple[S3ObjectVersion, ...],
) -> PhaseReceipt:
    return PhaseReceipt(
        format=PHASE_RECEIPT_FORMAT,
        schema_version=1,
        build_id=request.build_id,
        phase=phase,
        package_sha256=request.package_sha256,
        source_lock_sha256=request.source_lock_sha256,
        objects=tuple(sorted(objects, key=lambda item: item.uri)),
    )


def _verify_receipt_and_objects(
    context: _BuildContext,
    phase: str,
) -> None:
    reference = context.receipt_versions[phase]
    receipt = context.receipts[phase]
    verify_exact_object(context.s3, reference)
    for obj in receipt.objects:
        verify_exact_object(context.s3, obj)


def run_corpus_build(
    request: CorpusBuildRequest,
    *,
    s3: S3Client,
    runner: CommandRunner,
) -> S3ObjectVersion:
    """Run or exactly resume all phases and publish the final receipt last."""

    if not isinstance(request, CorpusBuildRequest):
        raise TypeError("request must be a CorpusBuildRequest")
    if not callable(getattr(runner, "run", None)):
        raise TypeError("runner must implement run(phase, action)")
    _validate_request_authority(request)
    context = _BuildContext(request=request, s3=s3)
    dependency_sha256 = _request_seed(request)

    for phase in PHASES:
        key = _phase_receipt_key(request, phase, dependency_sha256)
        existing = _read_existing_receipt(context, key=key)
        if existing is not None:
            receipt, receipt_version, payload = existing
            _validate_receipt_identity(request, phase, receipt)
            for obj in receipt.objects:
                verify_exact_object(s3, obj)
            if phase == "cleanroom-verify":
                published = context.receipts.get("s3-publish")
                if published is None or receipt.objects != published.objects:
                    raise PublicationError(
                        "clean-room receipt does not agree with S3 publication"
                    )
            _restore_or_verify_local_phase(context, phase, receipt.objects)
            if phase == "catalog":
                _authenticate_catalog_inputs(context)
        else:
            _begin_phase_attempt(context, phase)
            try:
                value = runner.run(
                    phase,
                    lambda phase=phase: _execute_phase(context, phase),
                )
                if not isinstance(value, _PhaseOutcome):
                    raise TypeError(f"{phase} runner returned an invalid phase outcome")
                if phase == "catalog":
                    _authenticate_catalog_inputs(
                        context,
                        root=_phase_action_root(context, phase),
                    )
                if phase == "cleanroom-verify":
                    published = context.receipts.get("s3-publish")
                    if published is None or value.objects != published.objects:
                        raise PublicationError(
                            "clean-room verification does not agree with S3 publication"
                        )
                value = _promote_phase_output(context, phase, value)
            finally:
                _clear_phase_attempt(context)
            objects = _publish_artifacts(context, phase, value)
            receipt = _receipt(request, phase, objects)
            receipt_version = publish_phase_receipt(
                s3,
                bucket=request.bucket,
                key=key,
                receipt=receipt,
                kms_key_arn=request.kms_key_arn,
            )
            payload = phase_receipt_to_bytes(receipt)
        context.receipts[phase] = receipt
        context.receipt_versions[phase] = receipt_version
        dependency_sha256 = hashlib.sha256(payload).hexdigest()

    published = context.receipts["s3-publish"]
    cleanroom = context.receipts["cleanroom-verify"]
    if (
        published.build_id != cleanroom.build_id
        or published.objects != cleanroom.objects
    ):
        raise PublicationError(
            "clean-room receipt does not agree on build ID and exact corpus objects"
        )
    _verify_receipt_and_objects(context, "s3-publish")
    _verify_receipt_and_objects(context, "cleanroom-verify")

    final = _receipt(request, "final", published.objects)
    final_reference = publish_phase_receipt(
        s3,
        bucket=request.bucket,
        key=f"{_build_prefix(request)}/receipts/final.json",
        receipt=final,
        kms_key_arn=request.kms_key_arn,
    )
    verify_exact_object(s3, final_reference)
    return final_reference


__all__ = (
    "CommandRunner",
    "CorpusBuildRequest",
    "InlineCommandRunner",
    "PHASES",
    "ProductionInputsUnavailable",
    "VerifiedProductionInputs",
    "load_verified_production_inputs",
    "run_corpus_build",
)
