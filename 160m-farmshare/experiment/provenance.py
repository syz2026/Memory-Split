"""Source-tree and artifact identity for a study freeze."""

from __future__ import annotations

import platform as platform_module
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from experiment.artifacts import (
    canonical_sha256,
    sha256_file,
    validate_sha256,
)


_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_FIELDS = {
    "record_type",
    "schema_version",
    "git_revision",
    "source_tree_sha256",
    "clean_tree",
    "python_version",
    "python_implementation",
    "platform",
    "artifact_sha256",
}


def _require_repository(root: str | Path) -> Path:
    path = Path(root)
    if ".." in path.parts:
        raise ValueError("repository path cannot contain traversal")
    absolute = path if path.is_absolute() else Path.cwd() / path
    if path.is_symlink() or not path.is_dir():
        raise ValueError("repository must be a regular non-symlink directory")
    if path.resolve(strict=True) != absolute:
        raise ValueError("repository path is not canonical or uses a symlink")
    return absolute


def _git(root: Path, *arguments: str, text: bool = True):
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("cannot inspect Git repository") from exc
    return completed.stdout


def _tracked_paths(root: Path) -> tuple[Path, ...]:
    raw = _git(root, "ls-files", "-z", "--cached", text=False)
    entries = tuple(item for item in raw.split(b"\0") if item)
    decoded: list[Path] = []
    for entry in entries:
        try:
            value = entry.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("tracked source path is not valid UTF-8") from exc
        relative = Path(value)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != value
        ):
            raise ValueError("tracked source path is not canonical")
        source = root / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError(
                f"tracked source must be a regular file: {value}"
            )
        if source.resolve(strict=True) != source:
            raise ValueError(f"tracked source traverses a symlink: {value}")
        decoded.append(relative)
    return tuple(sorted(decoded, key=lambda item: item.as_posix()))


def tracked_source_tree_sha256(root: str | Path) -> str:
    repository = _require_repository(root)
    members = {
        relative.as_posix(): sha256_file(
            repository / relative,
        )
        for relative in _tracked_paths(repository)
    }
    return canonical_sha256(
        {
            "record_type": "tracked_source_tree",
            "schema_version": 1,
            "members": members,
        }
    )


def _is_clean(root: Path) -> bool:
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        text=False,
    )
    return status == b""


@dataclass(frozen=True)
class SourceProvenance:
    git_revision: str
    source_tree_sha256: str
    clean_tree: bool
    python_version: str
    python_implementation: str
    platform: str
    artifact_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.git_revision, str)
            or _REVISION_RE.fullmatch(self.git_revision) is None
        ):
            raise ValueError("git revision must be a lowercase 40-hex value")
        validate_sha256(self.source_tree_sha256, "source tree SHA-256")
        if not isinstance(self.clean_tree, bool):
            raise ValueError("clean_tree must be Boolean")
        for name, value in (
            ("python_version", self.python_version),
            ("python_implementation", self.python_implementation),
            ("platform", self.platform),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.artifact_sha256, Mapping):
            raise ValueError("artifact_sha256 must be an object")
        artifacts: dict[str, str] = {}
        for name, digest in self.artifact_sha256.items():
            if not isinstance(name, str) or not name:
                raise ValueError("artifact hash names must be nonempty strings")
            artifacts[name] = validate_sha256(
                digest,
                f"artifact {name} SHA-256",
            )
        object.__setattr__(
            self,
            "artifact_sha256",
            MappingProxyType(dict(sorted(artifacts.items()))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "source_provenance",
            "schema_version": 1,
            "git_revision": self.git_revision,
            "source_tree_sha256": self.source_tree_sha256,
            "clean_tree": self.clean_tree,
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "platform": self.platform,
            "artifact_sha256": dict(self.artifact_sha256),
        }

    as_dict = to_dict

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SourceProvenance":
        if not isinstance(raw, Mapping) or set(raw) != _FIELDS:
            raise ValueError("source provenance fields are not exact")
        if (
            raw["record_type"] != "source_provenance"
            or raw["schema_version"] != 1
        ):
            raise ValueError("source provenance protocol is invalid")
        return cls(
            git_revision=raw["git_revision"],
            source_tree_sha256=raw["source_tree_sha256"],
            clean_tree=raw["clean_tree"],
            python_version=raw["python_version"],
            python_implementation=raw["python_implementation"],
            platform=raw["platform"],
            artifact_sha256=raw["artifact_sha256"],
        )


def verify_source_provenance(
    repo_root: str | Path,
    provenance: SourceProvenance | Mapping[str, Any],
    *,
    require_clean: bool = False,
) -> SourceProvenance:
    """Require the current revision and tracked bytes to match a freeze."""

    root = _require_repository(repo_root)
    validated = (
        SourceProvenance.from_dict(provenance.to_dict())
        if isinstance(provenance, SourceProvenance)
        else SourceProvenance.from_dict(provenance)
    )
    if not isinstance(require_clean, bool):
        raise ValueError("require_clean must be Boolean")
    if require_clean and not _is_clean(root):
        raise ValueError(
            "live source provenance requires a clean tree with no untracked files"
        )
    revision = _git(root, "rev-parse", "--verify", "HEAD").strip()
    if revision != validated.git_revision:
        raise ValueError("current Git revision does not match source provenance")
    if tracked_source_tree_sha256(root) != validated.source_tree_sha256:
        raise ValueError("current source tree drifted from source provenance")
    return validated


def collect_source_provenance(
    repo_root: str | Path,
    *,
    artifact_paths: Mapping[str, str | Path],
    require_clean: bool = True,
) -> SourceProvenance:
    root = _require_repository(repo_root)
    if not isinstance(require_clean, bool):
        raise ValueError("require_clean must be Boolean")
    clean = _is_clean(root)
    if require_clean and not clean:
        raise ValueError(
            "real study provenance requires a clean Git tree, including "
            "no untracked files"
        )
    revision = _git(root, "rev-parse", "--verify", "HEAD").strip()
    if _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("Git HEAD is not a canonical revision")
    if not isinstance(artifact_paths, Mapping) or not artifact_paths:
        raise ValueError("artifact paths must be a nonempty mapping")
    source_tree_sha256 = tracked_source_tree_sha256(root)
    artifact_hashes: dict[str, str] = {}
    for name, path in artifact_paths.items():
        if not isinstance(name, str) or not name:
            raise ValueError("artifact names must be nonempty strings")
        if name in artifact_hashes:
            raise ValueError("artifact names must be unique")
        artifact_hashes[name] = sha256_file(path)
    revision_after = _git(root, "rev-parse", "--verify", "HEAD").strip()
    clean_after = _is_clean(root)
    if revision_after != revision or clean_after != clean:
        raise ValueError("Git revision or tree changed during provenance capture")
    if require_clean and not clean_after:
        raise ValueError("real study provenance became dirty during capture")
    return SourceProvenance(
        git_revision=revision,
        source_tree_sha256=source_tree_sha256,
        clean_tree=clean,
        python_version=platform_module.python_version(),
        python_implementation=platform_module.python_implementation(),
        platform=platform_module.platform(),
        artifact_sha256=artifact_hashes,
    )


collect_provenance = collect_source_provenance
