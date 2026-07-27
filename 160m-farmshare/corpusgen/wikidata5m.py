from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any


_QID_RE = re.compile(r"Q[0-9]+")
_PID_RE = re.compile(r"P[0-9]+")
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
DEFAULT_WIKIDATA_LOCK_PATH = (
    Path(__file__).resolve().parents[1] / "sources" / "wikidata5m.lock.json"
)


class SourceDriftError(RuntimeError):
    """Raised when local bytes do not match a frozen source lock."""


class UnsafeArchiveError(ValueError):
    """Raised when an archive cannot be extracted as regular local files."""


@dataclass(frozen=True)
class Triple:
    subject: int
    relation: str
    object: int


@dataclass(frozen=True)
class ArchiveLock:
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.bytes, bool)
            or not isinstance(self.bytes, int)
            or self.bytes < 0
        ):
            raise ValueError("archive byte count must be a nonnegative integer")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(
            self.sha256
        ):
            raise ValueError("archive sha256 must be 64 lowercase hex characters")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArchiveLock:
        if set(value) != {"bytes", "sha256"}:
            raise ValueError("archive lock must contain only bytes and sha256")
        return cls(bytes=value["bytes"], sha256=value["sha256"])

    def to_dict(self) -> dict[str, int | str]:
        return {"bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class WikidataLock:
    repo_id: str
    repo_type: str
    revision: str
    files: Mapping[str, ArchiveLock]

    def __init__(
        self,
        repo_id: str,
        repo_type: str,
        revision: str,
        files: Mapping[str, ArchiveLock],
    ) -> None:
        if not isinstance(repo_id, str) or not repo_id:
            raise ValueError("repo_id must be a nonempty string")
        if not isinstance(repo_type, str) or not repo_type:
            raise ValueError("repo_type must be a nonempty string")
        if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
            raise ValueError("revision must be a 40-hex commit")
        if not files:
            raise ValueError("Wikidata lock must contain at least one file")

        locked_files: dict[str, ArchiveLock] = {}
        for name in sorted(files):
            if (
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                or "/" in name
                or "\\" in name
            ):
                raise ValueError(f"invalid locked archive name: {name!r}")
            item = files[name]
            if not isinstance(item, ArchiveLock):
                raise TypeError("WikidataLock files must contain ArchiveLock values")
            locked_files[name] = item

        object.__setattr__(self, "repo_id", repo_id)
        object.__setattr__(self, "repo_type", repo_type)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "files", MappingProxyType(locked_files))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WikidataLock:
        if set(value) != {"repo_id", "repo_type", "revision", "files"}:
            raise ValueError(
                "Wikidata lock must contain repo_id, repo_type, revision, and files"
            )
        raw_files = value["files"]
        if not isinstance(raw_files, Mapping):
            raise ValueError("Wikidata lock files must be an object")
        return cls(
            repo_id=value["repo_id"],
            repo_type=value["repo_type"],
            revision=value["revision"],
            files={
                name: ArchiveLock.from_dict(item)
                for name, item in raw_files.items()
            },
        )

    @classmethod
    def from_path(cls, path: str | Path) -> WikidataLock:
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Wikidata lock JSON: {path}") from exc
        if not isinstance(value, Mapping):
            raise ValueError("Wikidata lock must be a JSON object")
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "revision": self.revision,
            "files": {
                name: item.to_dict() for name, item in self.files.items()
            },
        }

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            + b"\n"
        )


def _load_default_lock() -> WikidataLock:
    return WikidataLock.from_path(DEFAULT_WIKIDATA_LOCK_PATH)


@dataclass(frozen=True)
class AliasCatalog(Mapping[str, tuple[str, ...]]):
    _items: tuple[tuple[str, tuple[str, ...]], ...]
    _ambiguous_items: tuple[tuple[str, tuple[str, ...]], ...]
    _lookup: Mapping[str, tuple[str, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_lookup", MappingProxyType(dict(self._items)))

    def __getitem__(self, key: str) -> tuple[str, ...]:
        return self._lookup[key]

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def ambiguous_normalized(self) -> dict[str, tuple[str, ...]]:
        return dict(self._ambiguous_items)


def parse_qid(value: str) -> int:
    if not isinstance(value, str) or not _QID_RE.fullmatch(value):
        raise ValueError(f"invalid Q id: {value!r}")
    return int(value[1:])


def parse_pid(value: str) -> str:
    if not isinstance(value, str) or not _PID_RE.fullmatch(value):
        raise ValueError(f"invalid P id: {value!r}")
    return value


def normalize_alias(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("alias must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _canonical_id_key(value: str) -> tuple[str, int, str]:
    if _PID_RE.fullmatch(value):
        return ("P", int(value[1:]), value)
    if _QID_RE.fullmatch(value):
        return ("Q", int(value[1:]), value)
    raise ValueError(f"invalid canonical ID: {value!r}")


def canonicalize_aliases(
    aliases: Mapping[str, Sequence[str]],
) -> AliasCatalog:
    deduplicated: dict[str, tuple[tuple[str, str], ...]] = {}
    owners: dict[str, set[str]] = {}

    for canonical_id in sorted(aliases, key=_canonical_id_key):
        raw_aliases = aliases[canonical_id]
        if isinstance(raw_aliases, (str, bytes)):
            raise TypeError("aliases for one canonical ID must be a sequence")
        first_seen: dict[str, str] = {}
        for raw_alias in raw_aliases:
            normalized = normalize_alias(raw_alias)
            if not normalized:
                raise ValueError(f"empty alias for {canonical_id}")
            if normalized not in first_seen:
                first_seen[normalized] = raw_alias
                owners.setdefault(normalized, set()).add(canonical_id)
        deduplicated[canonical_id] = tuple(first_seen.items())

    ambiguous = {
        normalized: tuple(sorted(relation_ids, key=_canonical_id_key))
        for normalized, relation_ids in owners.items()
        if len(relation_ids) > 1
    }
    items = tuple(
        (
            canonical_id,
            tuple(
                raw_alias
                for normalized, raw_alias in deduplicated[canonical_id]
                if normalized not in ambiguous
            ),
        )
        for canonical_id in sorted(deduplicated, key=_canonical_id_key)
    )
    return AliasCatalog(
        items,
        tuple((key, ambiguous[key]) for key in sorted(ambiguous)),
    )


def iter_triples(path: str | Path) -> Iterator[Triple]:
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            row = line.rstrip("\r\n")
            if not row:
                continue
            fields = row.split("\t")
            if len(fields) != 3:
                raise ValueError(
                    f"{source.name}:{line_number}: "
                    "expected 3 tab-separated fields"
                )
            try:
                subject = parse_qid(fields[0])
                relation = parse_pid(fields[1])
                object_id = parse_qid(fields[2])
            except ValueError as exc:
                raise ValueError(
                    f"{source.name}:{line_number}: {exc}"
                ) from exc
            yield Triple(subject, relation, object_id)


def read_aliases(
    path: str | Path,
    prefix: str,
) -> dict[str, tuple[str, ...]]:
    if prefix not in {"P", "Q"}:
        raise ValueError("alias prefix must be P or Q")
    parse_id = parse_pid if prefix == "P" else parse_qid
    source = Path(path)
    aliases: dict[str, tuple[str, ...]] = {}

    with source.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            row = line.rstrip("\r\n")
            if not row:
                continue
            fields = row.split("\t")
            if len(fields) < 2:
                raise ValueError(
                    f"{source.name}:{line_number}: "
                    "expected a canonical ID and at least one alias"
                )
            canonical_id = fields[0]
            try:
                parse_id(canonical_id)
            except ValueError as exc:
                raise ValueError(
                    f"{source.name}:{line_number}: {exc}"
                ) from exc
            if canonical_id in aliases:
                raise ValueError(
                    f"{source.name}:{line_number}: "
                    f"duplicate canonical ID: {canonical_id}"
                )

            first_seen: dict[str, str] = {}
            for raw_alias in fields[1:]:
                normalized = normalize_alias(raw_alias)
                if not normalized:
                    raise ValueError(
                        f"{source.name}:{line_number}: empty alias"
                    )
                first_seen.setdefault(normalized, raw_alias)
            aliases[canonical_id] = tuple(first_seen.values())

    return aliases


def _size_and_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def verify_archives(
    root: str | Path,
    lock: WikidataLock,
) -> tuple[Path, ...]:
    source_root = Path(root)
    staged_names = {
        path.name for path in source_root.glob("*.tar.gz")
    }
    extra_names = sorted(staged_names - set(lock.files))
    if extra_names:
        raise SourceDriftError(
            "unexpected unpinned archive(s): " + ", ".join(extra_names)
        )

    verified: list[Path] = []
    for name, expected in lock.files.items():
        path = source_root / name
        if not path.is_file() or path.is_symlink():
            raise SourceDriftError(f"missing locked archive: {name}")
        size, sha256 = _size_and_sha256(path)
        if size != expected.bytes or sha256 != expected.sha256:
            raise SourceDriftError(
                f"locked archive drift for {name}: "
                f"expected {expected.bytes} bytes/{expected.sha256}, "
                f"got {size} bytes/{sha256}"
            )
        verified.append(path)
    return tuple(verified)


def _validated_member_path(destination: Path, member: tarfile.TarInfo) -> Path:
    name = member.name
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if (
        not name
        or "\x00" in name
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
        or posix == PurePosixPath(".")
    ):
        raise UnsafeArchiveError(f"unsafe archive member: {name!r}")
    if not (member.isdir() or member.isreg()):
        raise UnsafeArchiveError(
            f"unsafe archive member type for {name!r}"
        )

    target = destination.joinpath(*posix.parts)
    destination_resolved = destination.resolve(strict=False)
    target_resolved = target.resolve(strict=False)
    try:
        target_resolved.relative_to(destination_resolved)
    except ValueError as exc:
        raise UnsafeArchiveError(f"archive member escapes output: {name!r}") from exc

    current = target
    while current != destination and current != current.parent:
        if current.is_symlink():
            raise UnsafeArchiveError(
                f"archive member crosses symlink: {name!r}"
            )
        current = current.parent
    return target


def safe_extract_archives(
    root: str | Path,
    out: str | Path,
    *,
    lock: WikidataLock | None = None,
) -> tuple[Path, ...]:
    archive_root = Path(root)
    destination = Path(out)
    if lock is None:
        lock = _load_default_lock()
    archives = verify_archives(archive_root, lock)

    planned: dict[Path, str] = {}
    for archive_path in archives:
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                target = _validated_member_path(destination, member)
                kind = "directory" if member.isdir() else "file"
                previous = planned.get(target)
                if previous is not None and not (
                    previous == "directory" and kind == "directory"
                ):
                    raise UnsafeArchiveError(
                        f"duplicate archive output path: {member.name!r}"
                    )
                planned[target] = kind

    extracted: list[Path] = []
    destination.mkdir(parents=True, exist_ok=True)
    for archive_path in archives:
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                target = _validated_member_path(destination, member)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise UnsafeArchiveError(
                        f"could not read regular archive member: {member.name!r}"
                    )
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted.append(target)

    return tuple(sorted(extracted))
