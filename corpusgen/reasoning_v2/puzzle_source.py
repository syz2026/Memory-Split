"""V2-native direct puzzle discovery and per-record lookup.

Puzzle bytes stay in the immutable, content-addressed :class:`SourceLock` tree.
This module never adds a legacy ``source-manifest.json`` or ``git/`` namespace
and never enumerates the filesystem for discovery: it classifies only the paths
already committed in ``SourceEntry.files``.

``scan_v2_puzzle_sources`` authenticates the complete source root once and emits
a compact, recomputable :class:`PuzzleSourceScan` of training/evaluation and
deduplication decisions. It holds only hashes and compact locators, never raw
task or answer bytes. ``read_v2_puzzle_task`` authenticates the source lock and
reopens exactly one selected JSON file with descriptor-relative, no-follow opens
and pre/post identity replay, so a single lookup never scans another source.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from corpusgen.current_sources import (
    _PUZZLE_SOURCE_ORDER as _REVIEWED_PUZZLE_SOURCE_ORDER,
    _strict_json_bytes as _strict_json,
    canonical_task_sha256,
    load_dataset_lock,
)
from corpusgen.parallel.canonical import canonical_json_bytes, sha256_hex
from corpusgen.parallel.safeio import (
    entry_lstat,
    open_directory_at,
    open_parent_directory,
    open_regular_file_at,
)
from corpusgen.reasoning_v2.source_lock import (
    SourceEntry,
    SourceFile,
    SourceLock,
    _authorize_source_lock,
    verify_source_tree,
)


_ROOT = Path(__file__).resolve().parents[2]
_CURRENT_DATASET_LOCK_PATH = _ROOT / "configs/current-dataset-lock.json"

_POLICY_VERSION = "memorysplit-reasoning-v2-puzzle-policy-v1"
_SCAN_VERSION = "memorysplit-reasoning-v2-puzzle-scan-v1"
_SUFFIX = ".json"
_INCLUDE_GLOB = "**/*.json"
_STRICT_JSON_RULE = "utf8-strict; reject duplicate object keys and non-finite constants"
_CANONICAL_TASK_HASH_RULE = (
    "sha256 of compact sorted-key utf-8 JSON without a trailing newline"
)
_EVALUATION_RULE = (
    "reject any training task whose canonical hash appears in any evaluation set"
)
_DEDUP_RULE = (
    "deduplicate accepted tasks by canonical hash; the first source-order then "
    "bytewise path-order task wins"
)
_RAW_AUDIT_RULE = (
    "record exact-byte duplicates; one raw hash maps to exactly one canonical hash"
)
_CANONICAL_HASH_CONTRACT = {
    "algorithm": "sha256",
    "encoding": "utf-8",
    "json": {"sort_keys": True, "separators": [",", ":"], "ensure_ascii": False},
}

PUZZLE_SOURCE_ORDER: tuple[str, ...] = ("arc_agi_1", "arc_agi_2", "conceptarc")


@dataclass(frozen=True)
class PuzzleSourcePolicy:
    source_id: str
    training_prefixes: tuple[str, ...]
    evaluation_prefixes: tuple[str, ...]
    suffix: str


_PUZZLE_POLICIES: tuple[PuzzleSourcePolicy, ...] = (
    PuzzleSourcePolicy("arc_agi_1", ("data/training",), ("data/evaluation",), _SUFFIX),
    PuzzleSourcePolicy("arc_agi_2", ("data/training",), ("data/evaluation",), _SUFFIX),
    PuzzleSourcePolicy("conceptarc", ("corpus",), (), _SUFFIX),
)


@dataclass(frozen=True)
class PuzzleTaskLocator:
    source_id: str
    path: str
    source_bytes: int
    source_sha256: str
    canonical_task_sha256: str
    test_count: int
    policy_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "canonical_task_sha256": self.canonical_task_sha256,
            "path": self.path,
            "policy_sha256": self.policy_sha256,
            "source_bytes": self.source_bytes,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "test_count": self.test_count,
        }


@dataclass(frozen=True)
class PuzzleDuplicateRecord:
    source_id: str
    path: str
    duplicate_of_source_id: str
    duplicate_of_path: str
    raw_sha256: str
    canonical_task_sha256: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "canonical_task_sha256": self.canonical_task_sha256,
            "duplicate_of_path": self.duplicate_of_path,
            "duplicate_of_source_id": self.duplicate_of_source_id,
            "path": self.path,
            "raw_sha256": self.raw_sha256,
            "reason": self.reason,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class PuzzleSourceScan:
    policy_sha256: str
    accepted: tuple[PuzzleTaskLocator, ...]
    evaluation_task_sha256s: tuple[str, ...]
    duplicates: tuple[PuzzleDuplicateRecord, ...]
    sha256: str


@dataclass(frozen=True)
class V2PuzzleTask:
    locator: PuzzleTaskLocator
    repository: str
    revision: str
    license_spdx: str
    task: dict[str, object]


# ---------------------------------------------------------------------------
# Frozen policy: cross-checked against the reviewed dataset lock at runtime.
# ---------------------------------------------------------------------------


def _policy_document() -> dict[str, object]:
    return {
        "canonical_task_hash": _CANONICAL_HASH_CONTRACT,
        "canonical_task_hash_rule": _CANONICAL_TASK_HASH_RULE,
        "dedup_rule": _DEDUP_RULE,
        "evaluation_rule": _EVALUATION_RULE,
        "raw_duplicate_rule": _RAW_AUDIT_RULE,
        "sources": [
            {
                "evaluation_prefixes": list(policy.evaluation_prefixes),
                "source_id": policy.source_id,
                "suffix": policy.suffix,
                "training_prefixes": list(policy.training_prefixes),
            }
            for policy in _PUZZLE_POLICIES
        ],
        "strict_json_rule": _STRICT_JSON_RULE,
        "suffix": _SUFFIX,
        "version": _POLICY_VERSION,
    }


def _verify_policy_matches_reviewed_lock() -> None:
    if PUZZLE_SOURCE_ORDER != tuple(_REVIEWED_PUZZLE_SOURCE_ORDER):
        raise ValueError("puzzle source order drift from the reviewed dataset lock")
    if tuple(policy.source_id for policy in _PUZZLE_POLICIES) != PUZZLE_SOURCE_ORDER:
        raise ValueError("frozen puzzle policy order drift")
    lock = load_dataset_lock(_CURRENT_DATASET_LOCK_PATH)
    for policy in _PUZZLE_POLICIES:
        source = lock.sources[policy.source_id]
        if (
            tuple(source["training_paths"]) != policy.training_prefixes
            or tuple(source["evaluation_paths"]) != policy.evaluation_prefixes
            or source["include_glob"] != _INCLUDE_GLOB
        ):
            raise ValueError(
                f"puzzle path policy drift from the reviewed dataset lock: "
                f"{policy.source_id}"
            )
    if lock.implementation["arc_tasks"]["canonical_hash"] != _CANONICAL_HASH_CONTRACT:
        raise ValueError(
            "puzzle canonical-hash policy drift from the reviewed dataset lock"
        )


def _policy_sha256() -> str:
    _verify_policy_matches_reviewed_lock()
    return sha256_hex(canonical_json_bytes(_policy_document()))


def _is_under(prefix: str, path: str) -> bool:
    return path.startswith(prefix + "/")


def _classify(policy: PuzzleSourcePolicy, path: str) -> str | None:
    if not path.endswith(policy.suffix):
        return None
    if any(_is_under(prefix, path) for prefix in policy.training_prefixes):
        return "training"
    if any(_is_under(prefix, path) for prefix in policy.evaluation_prefixes):
        return "evaluation"
    return None


# ---------------------------------------------------------------------------
# Strict parsing and the exact-answer training schema.
# ---------------------------------------------------------------------------


def _strict_task_object(content: bytes, description: str) -> dict[str, object]:
    task = _strict_json(content, description=description)
    if not isinstance(task, dict):
        raise ValueError(f"puzzle task must be a JSON object: {description}")
    return task


def _validate_exact_answer_task(task: dict[str, object], description: str) -> int:
    for split in ("train", "test"):
        if split not in task:
            raise ValueError(
                f"puzzle task is missing the {split} split: {description}"
            )
        examples = task[split]
        if not isinstance(examples, list) or not examples:
            raise ValueError(
                f"puzzle task {split} split must be a nonempty list: {description}"
            )
        for index, example in enumerate(examples):
            if not isinstance(example, dict):
                raise ValueError(
                    f"puzzle task {split}[{index}] must be an object: {description}"
                )
            if "input" not in example or "output" not in example:
                raise ValueError(
                    f"puzzle task {split}[{index}] must have input and output: "
                    f"{description}"
                )
    return len(task["test"])


# ---------------------------------------------------------------------------
# Descriptor-relative, no-follow selected-file read with identity replay.
# ---------------------------------------------------------------------------


def _selected_read_hook(phase: str, directory_fd: int, name: str) -> None:
    """Test seam fired during a selected read; a no-op in production."""

    del phase, directory_fd, name


def _namespace_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        stat.S_IMODE(metadata.st_mode),
    )


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_nlink,
    )


def _require_owned_directory(metadata: os.stat_result, description: str) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{description} is not a directory")
    if metadata.st_uid != os.geteuid():
        raise ValueError(f"{description} owner drift")
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise ValueError(f"{description} is group/world writable")


def _require_owned_regular(metadata: os.stat_result, description: str) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{description} is not a regular file")
    if metadata.st_uid != os.geteuid():
        raise ValueError(f"{description} owner drift")
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise ValueError(f"{description} is group/world writable")
    if metadata.st_nlink != 1:
        raise ValueError(f"{description} is a hardlink")


def _descend_directory(parent_fd: int, name: str, description: str) -> int:
    try:
        named_before = entry_lstat(parent_fd, name)
    except FileNotFoundError as error:
        raise ValueError(f"{description} is missing") from error
    if stat.S_ISLNK(named_before.st_mode):
        raise ValueError(f"{description} is a symlink")
    directory_fd, _created = open_directory_at(parent_fd, name)
    try:
        opened = os.fstat(directory_fd)
        _require_owned_directory(opened, description)
        try:
            named_after = entry_lstat(parent_fd, name)
        except FileNotFoundError as error:
            raise ValueError(f"{description} identity drift") from error
        identity = _namespace_identity(opened)
        if (
            _namespace_identity(named_before) != identity
            or _namespace_identity(named_after) != identity
        ):
            raise ValueError(f"{description} identity drift")
        return directory_fd
    except BaseException:
        os.close(directory_fd)
        raise


def _read_regular_no_follow(
    directory_fd: int,
    name: str,
    description: str,
) -> tuple[int, str, bytes]:
    try:
        named_before = entry_lstat(directory_fd, name)
    except FileNotFoundError as error:
        raise ValueError(f"selected puzzle file is missing: {description}") from error
    if stat.S_ISLNK(named_before.st_mode):
        raise ValueError(f"selected puzzle file is a symlink: {description}")
    if not stat.S_ISREG(named_before.st_mode):
        raise ValueError(f"selected puzzle file is not a regular file: {description}")
    if named_before.st_nlink != 1:
        raise ValueError(f"selected puzzle file is a hardlink: {description}")
    descriptor, opened = open_regular_file_at(directory_fd, name)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    read_bytes = 0
    try:
        _require_owned_regular(opened, f"selected puzzle file: {description}")
        _selected_read_hook("after_open", directory_fd, name)
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
            read_bytes += len(chunk)
        after = os.fstat(descriptor)
        _selected_read_hook("after_read", directory_fd, name)
        try:
            named_after = entry_lstat(directory_fd, name)
        except FileNotFoundError:
            named_after = None
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(after.st_mode)
        or named_after is None
        or _file_identity(named_before) != _file_identity(opened)
        or _file_identity(opened) != _file_identity(after)
        or _file_identity(after) != _file_identity(named_after)
        or read_bytes != opened.st_size
    ):
        raise ValueError(f"selected puzzle file identity drift: {description}")
    return read_bytes, digest.hexdigest(), b"".join(chunks)


def _read_selected_task_bytes(
    root_dir: Path,
    materialized_path: str,
    relative: str,
) -> tuple[int, str, bytes]:
    directory_names = (
        *PurePosixPath(materialized_path).parts,
        *PurePosixPath(relative).parts[:-1],
    )
    file_name = PurePosixPath(relative).name
    parent_fd, root_name = open_parent_directory(root_dir)
    open_fds: list[int] = []
    try:
        current = _descend_directory(parent_fd, root_name, "puzzle source root")
        open_fds.append(current)
        for name in directory_names:
            open_fds.append(
                _descend_directory(open_fds[-1], name, f"puzzle directory: {name}")
            )
        return _read_regular_no_follow(
            open_fds[-1],
            file_name,
            f"{materialized_path}/{relative}",
        )
    finally:
        for descriptor in reversed(open_fds):
            os.close(descriptor)
        os.close(parent_fd)


# ---------------------------------------------------------------------------
# Lock and root binding helpers.
# ---------------------------------------------------------------------------


def _require_content_addressed_root(
    source_root: Path,
    source_lock: SourceLock,
) -> Path:
    resolved = Path(source_root)
    if resolved.name != source_lock.sha256 or resolved.parent.name != "sources":
        raise ValueError(
            "puzzle source root is not a content-addressed "
            "<sources>/<source-lock-sha256> directory"
        )
    return resolved


def _entry_by_id(source_lock: SourceLock, source_id: str) -> SourceEntry:
    for entry in source_lock.sources:
        if entry.source_id == source_id:
            return entry
    raise ValueError(f"source lock has no source: {source_id}")


def _source_file_by_path(entry: SourceEntry, path: str) -> SourceFile:
    for row in entry.files:
        if row.path == path:
            return row
    raise ValueError(
        f"puzzle locator path is not in the source inventory: {entry.source_id}:{path}"
    )


def _require_exact_source_file(
    source_id: str,
    row: SourceFile,
    size: int,
    raw_sha256: str,
) -> None:
    if size != row.bytes or raw_sha256 != row.sha256:
        raise ValueError(f"selected puzzle file drift: {source_id}:{row.path}")


# ---------------------------------------------------------------------------
# Compact scan commitment.
# ---------------------------------------------------------------------------


def _scan_digest(
    policy_sha256: str,
    accepted: tuple[PuzzleTaskLocator, ...],
    evaluation_task_sha256s: tuple[str, ...],
    duplicates: tuple[PuzzleDuplicateRecord, ...],
) -> str:
    return sha256_hex(
        canonical_json_bytes(
            {
                "accepted": [locator.as_dict() for locator in accepted],
                "duplicates": [record.as_dict() for record in duplicates],
                "evaluation_task_sha256s": list(evaluation_task_sha256s),
                "policy_sha256": policy_sha256,
                "version": _SCAN_VERSION,
            }
        )
    )


def _scan_commitment(scan: PuzzleSourceScan) -> str:
    return _scan_digest(
        scan.policy_sha256,
        scan.accepted,
        scan.evaluation_task_sha256s,
        scan.duplicates,
    )


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def scan_v2_puzzle_sources(
    source_lock: SourceLock,
    source_root: Path,
    *,
    expected_generator_commit: str,
) -> PuzzleSourceScan:
    policy_sha256 = _policy_sha256()
    root_dir = _require_content_addressed_root(source_root, source_lock)
    verify_source_tree(
        source_lock,
        root_dir,
        expected_generator_commit=expected_generator_commit,
    )
    entries = {entry.source_id: entry for entry in source_lock.sources}

    evaluation_hashes: set[str] = set()
    for source_id in PUZZLE_SOURCE_ORDER:
        policy = _POLICY_BY_ID[source_id]
        entry = entries[source_id]
        for row in entry.files:
            if _classify(policy, row.path) != "evaluation":
                continue
            size, raw_sha256, content = _read_selected_task_bytes(
                root_dir, entry.materialized_path, row.path
            )
            _require_exact_source_file(source_id, row, size, raw_sha256)
            task = _strict_task_object(content, f"{source_id}:{row.path}")
            evaluation_hashes.add(canonical_task_sha256(task))

    accepted: list[PuzzleTaskLocator] = []
    duplicates: list[PuzzleDuplicateRecord] = []
    first_by_canonical: dict[str, PuzzleTaskLocator] = {}
    raw_to_canonical: dict[str, str] = {}
    for source_id in PUZZLE_SOURCE_ORDER:
        policy = _POLICY_BY_ID[source_id]
        entry = entries[source_id]
        for row in entry.files:
            if _classify(policy, row.path) != "training":
                continue
            identity = f"{source_id}:{row.path}"
            size, raw_sha256, content = _read_selected_task_bytes(
                root_dir, entry.materialized_path, row.path
            )
            _require_exact_source_file(source_id, row, size, raw_sha256)
            task = _strict_task_object(content, identity)
            test_count = _validate_exact_answer_task(task, identity)
            canonical = canonical_task_sha256(task)
            prior_canonical = raw_to_canonical.get(raw_sha256)
            if prior_canonical is not None and prior_canonical != canonical:
                raise ValueError(
                    f"raw hash maps to conflicting canonical task: {identity}"
                )
            raw_to_canonical[raw_sha256] = canonical
            if canonical in evaluation_hashes:
                raise ValueError(
                    f"training task overlaps an evaluation task: {identity}"
                )
            winner = first_by_canonical.get(canonical)
            if winner is not None:
                reason = (
                    "raw_duplicate"
                    if raw_sha256 == winner.source_sha256
                    else "canonical_duplicate"
                )
                duplicates.append(
                    PuzzleDuplicateRecord(
                        source_id=source_id,
                        path=row.path,
                        duplicate_of_source_id=winner.source_id,
                        duplicate_of_path=winner.path,
                        raw_sha256=raw_sha256,
                        canonical_task_sha256=canonical,
                        reason=reason,
                    )
                )
                continue
            locator = PuzzleTaskLocator(
                source_id=source_id,
                path=row.path,
                source_bytes=size,
                source_sha256=raw_sha256,
                canonical_task_sha256=canonical,
                test_count=test_count,
                policy_sha256=policy_sha256,
            )
            accepted.append(locator)
            first_by_canonical[canonical] = locator

    accepted_tuple = tuple(accepted)
    duplicates_tuple = tuple(duplicates)
    evaluation_tuple = tuple(sorted(evaluation_hashes))
    digest = _scan_digest(
        policy_sha256,
        accepted_tuple,
        evaluation_tuple,
        duplicates_tuple,
    )
    return PuzzleSourceScan(
        policy_sha256=policy_sha256,
        accepted=accepted_tuple,
        evaluation_task_sha256s=evaluation_tuple,
        duplicates=duplicates_tuple,
        sha256=digest,
    )


def _read_locator(
    root_dir: Path,
    source_lock: SourceLock,
    locator: PuzzleTaskLocator,
    policy_sha256: str,
) -> V2PuzzleTask:
    if not isinstance(locator, PuzzleTaskLocator):
        raise TypeError("locator must be a PuzzleTaskLocator")
    if locator.policy_sha256 != policy_sha256:
        raise ValueError("puzzle locator policy commitment drift")
    if locator.source_id not in _POLICY_BY_ID:
        raise ValueError(
            f"puzzle locator source is not a puzzle source: {locator.source_id}"
        )
    entry = _entry_by_id(source_lock, locator.source_id)
    if _classify(_POLICY_BY_ID[locator.source_id], locator.path) != "training":
        raise ValueError(
            f"puzzle locator path is not a training puzzle path: {locator.path}"
        )
    source_file = _source_file_by_path(entry, locator.path)
    if (
        source_file.bytes != locator.source_bytes
        or source_file.sha256 != locator.source_sha256
    ):
        raise ValueError("puzzle locator source-byte commitment drift")
    size, raw_sha256, content = _read_selected_task_bytes(
        root_dir, entry.materialized_path, locator.path
    )
    if size != locator.source_bytes or raw_sha256 != locator.source_sha256:
        raise ValueError("selected puzzle file raw-byte drift")
    identity = f"{locator.source_id}:{locator.path}"
    task = _strict_task_object(content, identity)
    test_count = _validate_exact_answer_task(task, identity)
    if canonical_task_sha256(task) != locator.canonical_task_sha256:
        raise ValueError("selected puzzle task canonical commitment drift")
    if test_count != locator.test_count:
        raise ValueError("selected puzzle task test-example count drift")
    return V2PuzzleTask(
        locator=locator,
        repository=entry.repository,
        revision=entry.revision,
        license_spdx=entry.license_spdx,
        task=task,
    )


def read_v2_puzzle_task(
    source_lock: SourceLock,
    source_root: Path,
    locator: PuzzleTaskLocator,
    *,
    expected_generator_commit: str,
) -> V2PuzzleTask:
    policy_sha256 = _policy_sha256()
    _authorize_source_lock(
        source_lock,
        expected_generator_commit=expected_generator_commit,
    )
    root_dir = _require_content_addressed_root(source_root, source_lock)
    return _read_locator(root_dir, source_lock, locator, policy_sha256)


def iter_v2_puzzle_tasks(
    source_lock: SourceLock,
    source_root: Path,
    scan: PuzzleSourceScan,
    *,
    expected_generator_commit: str,
) -> Iterator[V2PuzzleTask]:
    if not isinstance(scan, PuzzleSourceScan):
        raise TypeError("scan must be a PuzzleSourceScan")
    policy_sha256 = _policy_sha256()
    if scan.policy_sha256 != policy_sha256:
        raise ValueError("puzzle scan policy commitment drift")
    if scan.sha256 != _scan_commitment(scan):
        raise ValueError("puzzle scan commitment drift")
    root_dir = _require_content_addressed_root(source_root, source_lock)
    verify_source_tree(
        source_lock,
        root_dir,
        expected_generator_commit=expected_generator_commit,
    )
    for locator in scan.accepted:
        yield _read_locator(root_dir, source_lock, locator, policy_sha256)


_POLICY_BY_ID: dict[str, PuzzleSourcePolicy] = {
    policy.source_id: policy for policy in _PUZZLE_POLICIES
}
