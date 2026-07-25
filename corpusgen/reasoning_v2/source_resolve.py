"""Freeze the reviewed source catalog from upstream metadata, refuse the rest.

The reasoning-v2 source lock binds each source to an immutable revision and to
a SHA-256 over every materialized byte. Some of that is provable from metadata
alone: ``git ls-remote`` names a commit without a clone, and the Hugging Face
tree API reports, for LFS-backed files, the same SHA-256 the lock demands. The
rest is not provable that way, and this module never pretends otherwise -- an
unprovable binding becomes an itemized refusal carrying its byte cost, never a
default.

Every byte that crosses the network arrives through an injected
:class:`SourceMetadataClient`, so the whole resolution replays offline from a
recorded transcript.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from corpusgen.parallel.canonical import canonical_json_bytes, sha256_hex
from corpusgen.reasoning_v2 import source_lock


PIN_PLAN_FORMAT = "memorysplit-reasoning-v2-source-pin-plan-v1"
METADATA_RECORD_FORMAT = "memorysplit-reasoning-v2-source-metadata-record-v1"
TRANSCRIPT_PROVENANCE = "source-pin-resolver-transcript"
USER_AGENT = "MemorySplit-source-freezer/1"
HUGGINGFACE_API = "https://huggingface.co/api/datasets"
RULETAKER_ARTIFACT_HOST = "aristo-data-public.s3-us-west-2.amazonaws.com"
DEFAULT_HUGGINGFACE_REF = "main"
DEFAULT_GIT_REF = "HEAD"
MAX_METADATA_BYTES = 64 << 20

_HF_DATASET_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_HF_REVISION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_GIT_REF_RE = re.compile(r"HEAD\Z|refs/[A-Za-z0-9][A-Za-z0-9._/-]*\Z")
_SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_RECORD_FIELDS = frozenset(
    {
        "format",
        "git_ls_remote",
        "huggingface_revision",
        "huggingface_tree",
        "provenance",
        "schema_version",
    }
)
_PLAN_FIELDS = frozenset(
    {
        "complete",
        "dataset_id",
        "evidence_sha256",
        "format",
        "schema_version",
        "source_catalog_sha256",
        "sources",
        "unpinnable",
    }
)
_PINNED_SOURCE_FIELDS = frozenset(
    {
        "candidate_files",
        "files",
        "inventory_complete",
        "license_files",
        "license_spdx",
        "materialized_path",
        "repository",
        "revision",
        "revision_provenance",
        "source_id",
        "transport",
    }
)
_PINNED_FILE_FIELDS = frozenset({"bytes", "path", "provenance", "sha256"})
_BINDING_FIELDS = frozenset(
    {"binding", "byte_estimate", "bytes_required", "path", "reason", "source_id"}
)

_FILE_PROVENANCE = frozenset({"huggingface_lfs", "local_contract"})
_REVISION_PROVENANCE = frozenset(
    {"git_ls_remote", "huggingface_revision", "reviewed_catalog"}
)
_BINDING_KINDS = frozenset({"file_digest", "file_inventory", "finemath_selection"})
_BYTE_ESTIMATES = frozenset({"exact", "unknown", "upper_bound"})

_GIT_DIGEST_REFUSAL = (
    "git names objects by SHA-1 over a length-prefixed blob header, so no git "
    "metadata endpoint can prove the SHA-256 over file content that the lock "
    "requires; the whole tree at this commit must be fetched and hashed"
)
_NON_LFS_DIGEST_REFUSAL = (
    "the file is stored as a plain git blob, so the tree API reports only its "
    "SHA-1 object name and byte size; its SHA-256 requires the bytes"
)
_FINEMATH_SELECTION_REFUSAL = (
    "the selected shard prefix, usable target count, and duplicate-row count "
    "come from exact NFC cross-deduplication against FineWeb followed by "
    "tokenization, so they are unknowable until both corpora are read"
)


class UnpinnedSourceError(ValueError):
    """A caller asked for a lock while at least one binding stayed unproven."""


class SourceMetadataClient(Protocol):
    """The single network seam. Implementations do I/O and nothing else.

    Every method returns the upstream answer verbatim; all parsing, validation,
    and refusal happens in this module so that the transport stays auditable.
    """

    def git_ls_remote(self, repository: str, ref: str) -> str:
        """Return raw ``git ls-remote`` stdout for one ref."""

    def huggingface_dataset_revision(self, repository: str, revision: str) -> object:
        """Return the decoded ``/api/datasets/{id}/revision/{rev}`` body."""

    def huggingface_dataset_tree(self, repository: str, revision: str) -> object:
        """Return every decoded ``/api/datasets/{id}/tree/{rev}`` entry.

        Implementations must follow pagination and return the complete
        recursive listing; a truncated tree would look like a missing file.
        """


def _byte_key(value: str) -> bytes:
    return value.encode("utf-8")


def _require_int(value: object, description: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{description} must be a non-negative integer")
    return value


def _require_text(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a nonempty string")
    return value


def huggingface_dataset_id(value: object) -> str:
    if not isinstance(value, str) or _HF_DATASET_ID_RE.fullmatch(value) is None:
        raise ValueError(f"Hugging Face dataset identifier is unsafe: {value!r}")
    return value


def huggingface_revision(value: object) -> str:
    if not isinstance(value, str) or _HF_REVISION_RE.fullmatch(value) is None:
        raise ValueError(f"Hugging Face revision is unsafe: {value!r}")
    return value


def huggingface_tree_url(repository: str, revision: str) -> str:
    return (
        f"{HUGGINGFACE_API}/{huggingface_dataset_id(repository)}"
        f"/tree/{huggingface_revision(revision)}?expand=1&recursive=1"
    )


def huggingface_revision_url(repository: str, revision: str) -> str:
    return (
        f"{HUGGINGFACE_API}/{huggingface_dataset_id(repository)}"
        f"/revision/{huggingface_revision(revision)}"
    )


def git_remote_host(repository: object) -> str:
    if not isinstance(repository, str) or not repository:
        raise ValueError(f"git remote is unsafe: {repository!r}")
    parts = urllib.parse.urlsplit(repository)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError(f"git remote is unsafe: {repository!r}")
    return parts.hostname


def git_ls_remote_argv(repository: str, ref: str) -> tuple[str, ...]:
    git_remote_host(repository)
    if not isinstance(ref, str) or _GIT_REF_RE.fullmatch(ref) is None:
        raise ValueError(f"git ref is unsafe: {ref!r}")
    return ("git", "ls-remote", "--exit-code", repository, ref)


def _reviewed_request(source_id: str) -> source_lock.SourceRequest:
    request = source_lock._REQUESTS_BY_ID.get(source_id)
    if request is None:
        raise ValueError(f"source is not in the reviewed source catalog: {source_id}")
    return request


def _needs_git_metadata(source_id: str) -> bool:
    request = _reviewed_request(source_id)
    return (
        request.transport == "git"
        and source_lock._FIXED_IDENTITIES.get(source_id) is None
    )


def metadata_egress_hosts() -> tuple[str, ...]:
    hosts = set()
    for source_id in source_lock._RESOLUTION_ORDER:
        request = _reviewed_request(source_id)
        if request.transport == "huggingface_dataset":
            hosts.add(urllib.parse.urlsplit(HUGGINGFACE_API).hostname)
        elif _needs_git_metadata(source_id):
            hosts.add(git_remote_host(request.repository))
    return tuple(sorted(hosts))


@dataclass(frozen=True)
class PinnedFile:
    path: str
    bytes: int
    sha256: str
    provenance: Literal["huggingface_lfs", "local_contract"]

    def __post_init__(self) -> None:
        source_lock._safe_relative_path(self.path, "pinned file path")
        _require_int(self.bytes, "pinned file bytes")
        if not isinstance(self.sha256, str) or _SHA256_RE.fullmatch(self.sha256) is None:
            raise ValueError("pinned file sha256 must be a lowercase SHA-256")
        if self.provenance not in _FILE_PROVENANCE:
            raise ValueError(f"unsupported pin provenance: {self.provenance!r}")

    def as_dict(self) -> dict[str, object]:
        return {
            "bytes": self.bytes,
            "path": self.path,
            "provenance": self.provenance,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "PinnedFile":
        if not isinstance(value, dict) or set(value) != _PINNED_FILE_FIELDS:
            raise ValueError("pinned file fields do not match the contract")
        return cls(
            path=value["path"],
            bytes=value["bytes"],
            sha256=value["sha256"],
            provenance=value["provenance"],
        )

    def as_source_file(self) -> source_lock.SourceFile:
        return source_lock.SourceFile(
            path=self.path,
            bytes=self.bytes,
            sha256=self.sha256,
        )


@dataclass(frozen=True)
class UnpinnableBinding:
    source_id: str
    binding: Literal["file_digest", "file_inventory", "finemath_selection"]
    path: str | None
    reason: str
    bytes_required: int | None
    byte_estimate: Literal["exact", "unknown", "upper_bound"]

    def __post_init__(self) -> None:
        _reviewed_request(self.source_id)
        if self.binding not in _BINDING_KINDS:
            raise ValueError(f"unsupported unpinnable binding: {self.binding!r}")
        if self.path is not None:
            source_lock._safe_relative_path(self.path, "unpinnable binding path")
        _require_text(self.reason, "unpinnable binding reason")
        if self.byte_estimate not in _BYTE_ESTIMATES:
            raise ValueError(f"unsupported byte estimate: {self.byte_estimate!r}")
        if self.byte_estimate == "unknown":
            if self.bytes_required is not None:
                raise ValueError("an unknown byte estimate must carry no byte count")
        else:
            _require_int(self.bytes_required, "unpinnable binding bytes_required")

    @property
    def sort_key(self) -> tuple[bytes, bytes, bytes]:
        return (
            _byte_key(self.source_id),
            _byte_key(self.binding),
            _byte_key(self.path or ""),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "binding": self.binding,
            "byte_estimate": self.byte_estimate,
            "bytes_required": self.bytes_required,
            "path": self.path,
            "reason": self.reason,
            "source_id": self.source_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> "UnpinnableBinding":
        if not isinstance(value, dict) or set(value) != _BINDING_FIELDS:
            raise ValueError("unpinnable binding fields do not match the contract")
        return cls(
            source_id=value["source_id"],
            binding=value["binding"],
            path=value["path"],
            reason=value["reason"],
            bytes_required=value["bytes_required"],
            byte_estimate=value["byte_estimate"],
        )


@dataclass(frozen=True)
class PinnedSource:
    source_id: str
    transport: Literal["huggingface_dataset", "git"]
    repository: str
    revision: str
    revision_provenance: Literal[
        "git_ls_remote", "huggingface_revision", "reviewed_catalog"
    ]
    license_spdx: str
    license_files: tuple[str, ...]
    materialized_path: str
    files: tuple[PinnedFile, ...]
    candidate_files: tuple[PinnedFile, ...]
    inventory_complete: bool

    def __post_init__(self) -> None:
        request = _reviewed_request(self.source_id)
        if self.transport != request.transport or self.repository != request.repository:
            raise ValueError(f"reviewed source identity drift: {self.source_id}")
        source_lock._validate_commit(self.revision, f"{self.source_id} revision")
        if self.revision_provenance not in _REVISION_PROVENANCE:
            raise ValueError(
                f"unsupported revision provenance: {self.revision_provenance!r}"
            )
        if self.license_spdx != source_lock._REVIEWED_LICENSES[self.source_id]:
            raise ValueError(f"reviewed license drift: {self.source_id}")
        if self.license_files != source_lock._reviewed_license_paths(request):
            raise ValueError(f"reviewed license file drift: {self.source_id}")
        if self.materialized_path != self.source_id:
            raise ValueError(f"reviewed materialization drift: {self.source_id}")
        for group, description in (
            (self.files, "pinned files"),
            (self.candidate_files, "candidate files"),
        ):
            paths = tuple(row.path for row in group)
            if len(paths) != len(set(paths)):
                raise ValueError(f"{description} contain a duplicate path")
            if paths != tuple(sorted(paths, key=_byte_key)):
                raise ValueError(f"{description} must use bytewise path order")
        if set(row.path for row in self.files) & set(
            row.path for row in self.candidate_files
        ):
            raise ValueError(f"a path is both pinned and candidate: {self.source_id}")
        if type(self.inventory_complete) is not bool:
            raise ValueError("inventory_complete must be a boolean")
        if self.inventory_complete and (self.candidate_files or not self.files):
            raise ValueError(
                f"a complete inventory cannot hold candidates: {self.source_id}"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_files": [row.as_dict() for row in self.candidate_files],
            "files": [row.as_dict() for row in self.files],
            "inventory_complete": self.inventory_complete,
            "license_files": list(self.license_files),
            "license_spdx": self.license_spdx,
            "materialized_path": self.materialized_path,
            "repository": self.repository,
            "revision": self.revision,
            "revision_provenance": self.revision_provenance,
            "source_id": self.source_id,
            "transport": self.transport,
        }

    @classmethod
    def from_dict(cls, value: object) -> "PinnedSource":
        if not isinstance(value, dict) or set(value) != _PINNED_SOURCE_FIELDS:
            raise ValueError("pinned source fields do not match the contract")
        for field in ("candidate_files", "files", "license_files"):
            if not isinstance(value[field], list):
                raise ValueError(f"pinned source {field} must be a JSON list")
        return cls(
            source_id=value["source_id"],
            transport=value["transport"],
            repository=value["repository"],
            revision=value["revision"],
            revision_provenance=value["revision_provenance"],
            license_spdx=value["license_spdx"],
            license_files=tuple(value["license_files"]),
            materialized_path=value["materialized_path"],
            files=tuple(PinnedFile.from_dict(row) for row in value["files"]),
            candidate_files=tuple(
                PinnedFile.from_dict(row) for row in value["candidate_files"]
            ),
            inventory_complete=value["inventory_complete"],
        )


@dataclass(frozen=True)
class SourcePinPlan:
    schema_version: Literal[1]
    format: Literal["memorysplit-reasoning-v2-source-pin-plan-v1"]
    dataset_id: str
    source_catalog_sha256: str
    evidence_sha256: str
    sources: tuple[PinnedSource, ...]
    unpinnable: tuple[UnpinnableBinding, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("source pin plan schema_version must be integer 1")
        if self.format != PIN_PLAN_FORMAT:
            raise ValueError("source pin plan format identity mismatch")
        if self.dataset_id != source_lock.DATASET_ID:
            raise ValueError("source pin plan dataset identity mismatch")
        source_lock._validate_digest(self.source_catalog_sha256, "source_catalog_sha256")
        source_lock._validate_digest(self.evidence_sha256, "evidence_sha256")
        source_ids = tuple(row.source_id for row in self.sources)
        if set(source_ids) != set(source_lock._EXPECTED_SOURCE_IDS):
            raise ValueError("source pin plan does not cover the reviewed catalog")
        if source_ids != tuple(sorted(source_ids, key=_byte_key)):
            raise ValueError("source pin plan sources must use bytewise order")
        keys = [row.sort_key for row in self.unpinnable]
        if keys != sorted(keys):
            raise ValueError("source pin plan bindings must use bytewise order")
        if len(keys) != len(set(keys)):
            raise ValueError("source pin plan contains a duplicate binding")
        unpinned = {row.source_id for row in self.unpinnable}
        for row in self.sources:
            if row.inventory_complete and row.source_id in unpinned:
                raise ValueError(
                    f"a source cannot be complete and unpinned: {row.source_id}"
                )

    @property
    def complete(self) -> bool:
        return not self.unpinnable

    @property
    def unpinned_source_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted({row.source_id for row in self.unpinnable}, key=_byte_key)
        )

    @property
    def sha256(self) -> str:
        return sha256_hex(self.to_bytes())

    def byte_budget(self) -> dict[str, int]:
        pinned = sum(
            row.bytes
            for source in self.sources
            for row in (*source.files, *source.candidate_files)
        )
        exact = sum(
            row.bytes_required or 0
            for row in self.unpinnable
            if row.byte_estimate == "exact"
        )
        bounded = sum(
            row.bytes_required or 0
            for row in self.unpinnable
            if row.byte_estimate in {"exact", "upper_bound"}
        )
        return {
            "freeze_bindings_without_estimate": sum(
                1 for row in self.unpinnable if row.byte_estimate == "unknown"
            ),
            "freeze_bytes_exact": exact,
            "freeze_bytes_upper_bound": bounded,
            "pinned_bytes": pinned,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "complete": self.complete,
            "dataset_id": self.dataset_id,
            "evidence_sha256": self.evidence_sha256,
            "format": self.format,
            "schema_version": self.schema_version,
            "source_catalog_sha256": self.source_catalog_sha256,
            "sources": [row.as_dict() for row in self.sources],
            "unpinnable": [row.as_dict() for row in self.unpinnable],
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    @classmethod
    def from_dict(cls, value: object) -> "SourcePinPlan":
        if not isinstance(value, dict) or set(value) != _PLAN_FIELDS:
            raise ValueError("source pin plan fields do not match the contract")
        for field in ("sources", "unpinnable"):
            if not isinstance(value[field], list):
                raise ValueError(f"source pin plan {field} must be a JSON list")
        plan = cls(
            schema_version=value["schema_version"],
            format=value["format"],
            dataset_id=value["dataset_id"],
            source_catalog_sha256=value["source_catalog_sha256"],
            evidence_sha256=value["evidence_sha256"],
            sources=tuple(PinnedSource.from_dict(row) for row in value["sources"]),
            unpinnable=tuple(
                UnpinnableBinding.from_dict(row) for row in value["unpinnable"]
            ),
        )
        if value["complete"] is not plan.complete:
            raise ValueError("source pin plan completeness claim drift")
        return plan


def source_lock_entries(plan: SourcePinPlan) -> tuple[source_lock.SourceEntry, ...]:
    """Convert a plan into lock entries, or refuse if anything stayed unproven."""

    if not isinstance(plan, SourcePinPlan):
        raise TypeError("plan must be a SourcePinPlan")
    if not plan.complete:
        raise UnpinnedSourceError(
            "refusing to emit source lock entries while bindings are unproven: "
            + ", ".join(plan.unpinned_source_ids)
        )
    return tuple(
        source_lock.SourceEntry(
            source_id=row.source_id,
            transport=row.transport,
            repository=row.repository,
            revision_kind="git_commit",
            revision=row.revision,
            license_spdx=row.license_spdx,
            license_files=row.license_files,
            materialized_path=row.materialized_path,
            files=tuple(pinned.as_source_file() for pinned in row.files),
        )
        for row in plan.sources
    )


class RecordedMetadataClient:
    """Replay a recorded metadata transcript; never answer what was not recorded."""

    __slots__ = ("_git", "_revision", "_tree", "_provenance")

    def __init__(self, document: object) -> None:
        if not isinstance(document, dict) or set(document) != _RECORD_FIELDS:
            raise ValueError("source metadata record fields do not match the contract")
        if document["format"] != METADATA_RECORD_FORMAT:
            raise ValueError("source metadata record format identity mismatch")
        if type(document["schema_version"]) is not int or (
            document["schema_version"] != 1
        ):
            raise ValueError("source metadata record schema_version must be integer 1")
        self._provenance = _require_text(
            document["provenance"],
            "source metadata record provenance",
        )
        self._git = _validated_record_map(document["git_ls_remote"], "git_ls_remote")
        self._revision = _validated_record_map(
            document["huggingface_revision"],
            "huggingface_revision",
        )
        self._tree = _validated_record_map(
            document["huggingface_tree"],
            "huggingface_tree",
        )

    @property
    def provenance(self) -> str:
        return self._provenance

    def _answer(self, table: dict, kind: str, repository: str, ref: str) -> object:
        answers = table.get(repository)
        if not isinstance(answers, dict) or ref not in answers:
            raise ValueError(
                f"metadata record has no {kind} answer for {repository} at {ref}"
            )
        return copy.deepcopy(answers[ref])

    def git_ls_remote(self, repository: str, ref: str) -> str:
        answer = self._answer(self._git, "git_ls_remote", repository, ref)
        if not isinstance(answer, str):
            raise ValueError(f"recorded git_ls_remote answer is not text: {repository}")
        return answer

    def huggingface_dataset_revision(self, repository: str, revision: str) -> object:
        return self._answer(
            self._revision,
            "huggingface_dataset_revision",
            repository,
            revision,
        )

    def huggingface_dataset_tree(self, repository: str, revision: str) -> object:
        return self._answer(
            self._tree,
            "huggingface_dataset_tree",
            repository,
            revision,
        )


def _validated_record_map(value: object, description: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError(f"source metadata record {description} must be an object")
    table: dict[str, dict[str, Any]] = {}
    for key, answers in value.items():
        _require_text(key, f"{description} repository")
        if not isinstance(answers, dict):
            raise ValueError(f"source metadata record {description} entry is invalid")
        for ref in answers:
            _require_text(ref, f"{description} ref")
        table[key] = copy.deepcopy(answers)
    return table


def load_metadata_record(path: Path) -> RecordedMetadataClient:
    payload = source_lock._read_regular_bytes(Path(path), "source metadata record")
    return RecordedMetadataClient(
        source_lock._strict_json_bytes(payload, "source metadata record")
    )


class HttpsSourceMetadataClient:
    """The only place this tool opens a transport; kept thin enough to audit."""

    __slots__ = ("_timeout",)

    def __init__(self, *, timeout: float = 30.0) -> None:
        self._timeout = timeout

    def git_ls_remote(self, repository: str, ref: str) -> str:
        argv = git_ls_remote_argv(repository, ref)
        environment = {
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": "",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
        }
        return subprocess.run(
            list(argv),
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout

    def huggingface_dataset_revision(self, repository: str, revision: str) -> object:
        return self._get_json(huggingface_revision_url(repository, revision))

    def huggingface_dataset_tree(self, repository: str, revision: str) -> object:
        return self._get_json(huggingface_tree_url(repository, revision))

    def _get_json(self, url: str) -> object:
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            if response.geturl() != url:
                raise ValueError(f"source metadata URL redirected unexpectedly: {url}")
            payload = response.read(MAX_METADATA_BYTES + 1)
        if len(payload) > MAX_METADATA_BYTES:
            raise ValueError(f"source metadata response is too large: {url}")
        return json.loads(payload.decode("utf-8"))


@dataclass(frozen=True)
class _TreeFile:
    path: str
    size: int
    lfs_sha256: str | None


def _tree_files(value: object, source_id: str) -> dict[str, _TreeFile]:
    if not isinstance(value, list):
        raise ValueError(f"repository tree must be a JSON list: {source_id}")
    files: dict[str, _TreeFile] = {}
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(f"repository tree entry must be an object: {source_id}")
        kind = row.get("type")
        if kind == "directory":
            continue
        if kind != "file":
            raise ValueError(
                f"repository tree holds an unsupported entry type: {source_id}"
            )
        path = source_lock._safe_relative_path(
            row.get("path"),
            f"{source_id} repository path",
        )
        if path in files:
            raise ValueError(f"repository tree repeats a path: {source_id}:{path}")
        size = _require_int(row.get("size"), f"{source_id}:{path} byte count")
        lfs = row.get("lfs")
        digest = None
        if lfs is not None:
            if not isinstance(lfs, dict):
                raise ValueError(f"repository LFS metadata is invalid: {source_id}:{path}")
            digest = lfs.get("oid")
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(
                    f"repository LFS oid is not a SHA-256: {source_id}:{path}"
                )
            if _require_int(lfs.get("size"), f"{source_id}:{path} LFS size") != size:
                raise ValueError(
                    f"repository LFS byte count disagrees with the tree: "
                    f"{source_id}:{path}"
                )
        files[path] = _TreeFile(path=path, size=size, lfs_sha256=digest)
    return files


def _required_tree_file(
    files: dict[str, _TreeFile],
    source_id: str,
    path: str,
) -> _TreeFile:
    row = files.get(path)
    if row is None:
        raise ValueError(f"required file is absent from the tree: {source_id}:{path}")
    return row


def _pin_or_refuse(
    row: _TreeFile,
    source_id: str,
) -> tuple[PinnedFile | None, UnpinnableBinding | None]:
    if row.lfs_sha256 is None:
        return None, UnpinnableBinding(
            source_id=source_id,
            binding="file_digest",
            path=row.path,
            reason=_NON_LFS_DIGEST_REFUSAL,
            bytes_required=row.size,
            byte_estimate="exact",
        )
    return (
        PinnedFile(
            path=row.path,
            bytes=row.size,
            sha256=row.lfs_sha256,
            provenance="huggingface_lfs",
        ),
        None,
    )


def _require_reviewed_digests(source_id: str, pinned: tuple[PinnedFile, ...]) -> None:
    by_path = {row.path: row for row in pinned}
    for path, expected in source_lock._reviewed_fixed_files(source_id).items():
        row = by_path.get(path)
        if (
            row is None
            or row.bytes != expected["bytes"]
            or row.sha256 != expected["sha256"]
        ):
            raise ValueError(f"reviewed file digest drift: {source_id}:{path}")


def _reviewed_fixed_bytes(source_id: str) -> int:
    return sum(
        int(row["bytes"])
        for row in source_lock._reviewed_fixed_files(source_id).values()
    )


def _local_contract_pin(path: str, contract_path: Path) -> PinnedFile:
    payload = source_lock._read_regular_bytes(
        Path(contract_path),
        f"local source contract {path}",
    )
    return PinnedFile(
        path=path,
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        provenance="local_contract",
    )


def _sorted_pins(rows: list[PinnedFile]) -> tuple[PinnedFile, ...]:
    return tuple(sorted(rows, key=lambda row: _byte_key(row.path)))


class _Transcript:
    """Record exactly what the resolver consumed, so a plan can be replayed."""

    __slots__ = ("git_ls_remote", "huggingface_revision", "huggingface_tree")

    def __init__(self) -> None:
        self.git_ls_remote: dict[str, dict[str, Any]] = {}
        self.huggingface_revision: dict[str, dict[str, Any]] = {}
        self.huggingface_tree: dict[str, dict[str, Any]] = {}

    def note(self, table: str, repository: str, ref: str, answer: object) -> None:
        getattr(self, table).setdefault(repository, {})[ref] = answer

    def document(self) -> dict[str, object]:
        return {
            "format": METADATA_RECORD_FORMAT,
            "git_ls_remote": self.git_ls_remote,
            "huggingface_revision": self.huggingface_revision,
            "huggingface_tree": self.huggingface_tree,
            "provenance": TRANSCRIPT_PROVENANCE,
            "schema_version": 1,
        }


def _git_commit_from_ls_remote(stdout: object, source_id: str, ref: str) -> str:
    if not isinstance(stdout, str):
        raise ValueError(f"git ls-remote answer is not text: {source_id}")
    rows = [line.split() for line in stdout.splitlines() if line.strip()]
    if (
        len(rows) != 1
        or len(rows[0]) != 2
        or rows[0][1] != ref
        or _SHA1_RE.fullmatch(rows[0][0]) is None
    ):
        raise ValueError(
            f"git {ref} did not resolve to one immutable commit: {source_id}"
        )
    return rows[0][0]


def _finemath_revision(answer: object, source_id: str) -> str:
    if not isinstance(answer, dict):
        raise ValueError(f"Hugging Face revision answer must be an object: {source_id}")
    card = answer.get("cardData")
    license_value = card.get("license") if isinstance(card, dict) else None
    if license_value not in {"odc-by", "ODC-By-1.0"}:
        raise ValueError("FineMath metadata does not prove ODC-By-1.0")
    revision = answer.get("sha")
    if not isinstance(revision, str) or _SHA1_RE.fullmatch(revision) is None:
        raise ValueError(
            f"Hugging Face source did not resolve to an immutable commit: {source_id}"
        )
    return revision


def _resolve_huggingface_fixed(
    source_id: str,
    client: SourceMetadataClient,
    transcript: _Transcript,
    *,
    tree_paths: tuple[str, ...],
    local_files: tuple[tuple[str, Path], ...] = (),
) -> tuple[PinnedSource, list[UnpinnableBinding]]:
    request = _reviewed_request(source_id)
    revision = source_lock._FIXED_IDENTITIES[source_id][2]
    answer = client.huggingface_dataset_tree(request.repository, revision)
    transcript.note("huggingface_tree", request.repository, revision, answer)
    files = _tree_files(answer, source_id)
    pins: list[PinnedFile] = []
    refusals: list[UnpinnableBinding] = []
    for path in tree_paths:
        pin, refusal = _pin_or_refuse(
            _required_tree_file(files, source_id, path),
            source_id,
        )
        if pin is not None:
            pins.append(pin)
        if refusal is not None:
            refusals.append(refusal)
    for path, contract_path in local_files:
        pins.append(_local_contract_pin(path, contract_path))
    _require_reviewed_digests(source_id, tuple(pins))
    source = PinnedSource(
        source_id=source_id,
        transport=request.transport,
        repository=request.repository,
        revision=revision,
        revision_provenance="reviewed_catalog",
        license_spdx=source_lock._REVIEWED_LICENSES[source_id],
        license_files=source_lock._reviewed_license_paths(request),
        materialized_path=source_id,
        files=_sorted_pins(pins),
        candidate_files=(),
        inventory_complete=not refusals,
    )
    return source, refusals


def _resolve_finemath(
    client: SourceMetadataClient,
    transcript: _Transcript,
) -> tuple[PinnedSource, list[UnpinnableBinding]]:
    source_id = "finemath"
    request = _reviewed_request(source_id)
    answer = client.huggingface_dataset_revision(
        request.repository,
        DEFAULT_HUGGINGFACE_REF,
    )
    transcript.note(
        "huggingface_revision",
        request.repository,
        DEFAULT_HUGGINGFACE_REF,
        answer,
    )
    revision = _finemath_revision(answer, source_id)
    tree = client.huggingface_dataset_tree(request.repository, revision)
    transcript.note("huggingface_tree", request.repository, revision, tree)
    files = _tree_files(tree, source_id)

    refusals: list[UnpinnableBinding] = []
    license_path = source_lock._reviewed_license_paths(request)[0]
    pin, refusal = _pin_or_refuse(
        _required_tree_file(files, source_id, license_path),
        source_id,
    )
    pins = [pin] if pin is not None else []
    if refusal is not None:
        refusals.append(refusal)

    four_plus = tuple(
        sorted(
            (
                path
                for path in files
                if source_lock._is_finemath_train_file(path, "finemath-4plus")
            ),
            key=_byte_key,
        )
    )
    three_plus = tuple(
        sorted(
            (
                path
                for path in files
                if source_lock._is_finemath_train_file(path, "finemath-3plus")
            ),
            key=_byte_key,
        )
    )
    source_lock._validate_finemath_shard_sequence(four_plus, "finemath-4plus")
    source_lock._validate_finemath_shard_sequence(three_plus, "finemath-3plus")

    candidates: list[PinnedFile] = []
    for path in (*four_plus, *three_plus):
        shard_pin, shard_refusal = _pin_or_refuse(files[path], source_id)
        if shard_pin is None:
            raise ValueError(
                f"FineMath shard is not LFS-backed, so no metadata proves its "
                f"SHA-256: {path}"
            )
        del shard_refusal
        candidates.append(shard_pin)

    refusals.append(
        UnpinnableBinding(
            source_id=source_id,
            binding="finemath_selection",
            path=None,
            reason=_FINEMATH_SELECTION_REFUSAL,
            bytes_required=(
                _reviewed_fixed_bytes("fineweb_edu")
                + sum(row.bytes for row in candidates)
            ),
            byte_estimate="upper_bound",
        )
    )
    source = PinnedSource(
        source_id=source_id,
        transport=request.transport,
        repository=request.repository,
        revision=revision,
        revision_provenance="huggingface_revision",
        license_spdx=source_lock._REVIEWED_LICENSES[source_id],
        license_files=source_lock._reviewed_license_paths(request),
        materialized_path=source_id,
        files=_sorted_pins(pins),
        candidate_files=_sorted_pins(candidates),
        inventory_complete=False,
    )
    return source, refusals


def _resolve_git(
    source_id: str,
    client: SourceMetadataClient,
    transcript: _Transcript,
) -> tuple[PinnedSource, list[UnpinnableBinding]]:
    request = _reviewed_request(source_id)
    fixed = source_lock._FIXED_IDENTITIES.get(source_id)
    if fixed is None:
        answer = client.git_ls_remote(request.repository, DEFAULT_GIT_REF)
        transcript.note(
            "git_ls_remote",
            request.repository,
            DEFAULT_GIT_REF,
            answer,
        )
        revision = _git_commit_from_ls_remote(answer, source_id, DEFAULT_GIT_REF)
        provenance = "git_ls_remote"
    else:
        revision = fixed[2]
        provenance = "reviewed_catalog"
    refusals = [
        UnpinnableBinding(
            source_id=source_id,
            binding="file_inventory",
            path=None,
            reason=_GIT_DIGEST_REFUSAL,
            bytes_required=None,
            byte_estimate="unknown",
        )
    ]
    if source_id == "ruletaker":
        refusals.append(
            UnpinnableBinding(
                source_id=source_id,
                binding="file_digest",
                path=source_lock._RULETAKER_ARCHIVE,
                reason=(
                    "the reviewed dataset archive is served by "
                    f"{RULETAKER_ARTIFACT_HOST}, which publishes no SHA-256; the "
                    "archive must be downloaded and hashed"
                ),
                bytes_required=None,
                byte_estimate="unknown",
            )
        )
    source = PinnedSource(
        source_id=source_id,
        transport=request.transport,
        repository=request.repository,
        revision=revision,
        revision_provenance=provenance,
        license_spdx=source_lock._REVIEWED_LICENSES[source_id],
        license_files=source_lock._reviewed_license_paths(request),
        materialized_path=source_id,
        files=(),
        candidate_files=(),
        inventory_complete=False,
    )
    return source, refusals


def resolve_source_pins(
    client: SourceMetadataClient,
    *,
    evidence_sink: list[dict[str, object]] | None = None,
) -> SourcePinPlan:
    """Pin every reviewed source that metadata can prove, and itemize the rest."""

    transcript = _Transcript()
    sources: list[PinnedSource] = []
    unpinnable: list[UnpinnableBinding] = []
    for source_id in source_lock._RESOLUTION_ORDER:
        request = _reviewed_request(source_id)
        if source_id == "fineweb_edu":
            resolved, refusals = _resolve_huggingface_fixed(
                source_id,
                client,
                transcript,
                tree_paths=tuple(
                    sorted(
                        {
                            *source_lock._reviewed_license_paths(request),
                            *request.required_data_paths,
                        },
                        key=_byte_key,
                    )
                ),
            )
        elif source_id == "wikidata5m":
            resolved, refusals = _resolve_huggingface_fixed(
                source_id,
                client,
                transcript,
                tree_paths=request.required_data_paths,
                local_files=(
                    (
                        source_lock._reviewed_license_paths(request)[0],
                        source_lock.WIKIDATA_NOTICE_PATH,
                    ),
                ),
            )
        elif source_id == "finemath":
            resolved, refusals = _resolve_finemath(client, transcript)
        elif request.transport == "git":
            resolved, refusals = _resolve_git(source_id, client, transcript)
        else:
            raise ValueError(f"no reviewed pin policy for source: {source_id}")
        _require_license_coverage(resolved)
        sources.append(resolved)
        unpinnable.extend(refusals)

    document = transcript.document()
    if evidence_sink is not None:
        evidence_sink.append(document)
    return SourcePinPlan(
        schema_version=1,
        format=PIN_PLAN_FORMAT,
        dataset_id=source_lock.DATASET_ID,
        source_catalog_sha256=source_lock.reviewed_source_catalog_sha256(),
        evidence_sha256=sha256_hex(canonical_json_bytes(document)),
        sources=tuple(sorted(sources, key=lambda row: _byte_key(row.source_id))),
        unpinnable=tuple(sorted(unpinnable, key=lambda row: row.sort_key)),
    )


def _require_license_coverage(source: PinnedSource) -> None:
    request = _reviewed_request(source.source_id)
    known = {row.path for row in source.files}
    for path in request.required_data_paths:
        if source.inventory_complete and path not in known:
            raise ValueError(
                f"a complete inventory omits a required data path: "
                f"{source.source_id}:{path}"
            )
    for path in source.license_files:
        PurePosixPath(path)


__all__ = (
    "DEFAULT_GIT_REF",
    "DEFAULT_HUGGINGFACE_REF",
    "HUGGINGFACE_API",
    "HttpsSourceMetadataClient",
    "METADATA_RECORD_FORMAT",
    "PIN_PLAN_FORMAT",
    "PinnedFile",
    "PinnedSource",
    "RULETAKER_ARTIFACT_HOST",
    "RecordedMetadataClient",
    "SourceMetadataClient",
    "SourcePinPlan",
    "UnpinnableBinding",
    "UnpinnedSourceError",
    "git_ls_remote_argv",
    "git_remote_host",
    "huggingface_dataset_id",
    "huggingface_revision",
    "huggingface_revision_url",
    "huggingface_tree_url",
    "load_metadata_record",
    "metadata_egress_hosts",
    "resolve_source_pins",
    "source_lock_entries",
)
