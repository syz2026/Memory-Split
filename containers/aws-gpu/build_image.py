#!/usr/bin/env python3
"""Render, and only with explicit approval execute, the AWS GPU image build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol


PLATFORM = "linux/amd64"
DEFAULT_DOCKER_BINARY = "/usr/bin/docker"
DEFAULT_GIT_BINARY = "/usr/bin/git"
DLC_PYTHON = "/opt/conda/bin/python"
BASE_REGISTRY = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training"
)
BASE_DIGEST = (
    "sha256:1414a836532f22b271c03b7ccdbdff3d"
    "aa0591975b3bd9a3cf51601a45b37f4f"
)
BASE_IMAGE = f"{BASE_REGISTRY}@{BASE_DIGEST}"
MINIMAL_COMMAND_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/local/bin:/usr/bin:/bin",
}
_PLAN_FIELDS = {
    "schema_version",
    "operation",
    "platform",
    "source_commit",
    "source_tree",
    "repository_root",
    "dockerfile",
    "repository_uri",
    "staging_tag",
    "inputs",
    "repository_transcript_sha256",
    "environment",
    "commands",
    "push_requires_apply",
}
_COMMAND_FIELDS = {
    "build",
    "python_base",
    "python_local",
    "inspect_base",
    "inspect_local",
    "facts_base",
    "facts_local",
    "inspection_local",
    "push",
}
_INPUT_FIELDS = {
    "dockerfile_sha256",
    "dockerignore_sha256",
    "dependency_lock_sha256",
    "inspection_script_sha256",
}
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PRIVATE_ECR_RE = re.compile(
    r"^(?P<account>[0-9]{12})\.dkr\.ecr\."
    r"(?P<region>[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+)"
    r"\.amazonaws\.com/"
    r"(?P<repository>[a-z0-9]+(?:(?:[._-]|/)[a-z0-9]+)*)$"
)
_PUSH_DIGEST_RE = re.compile(
    rb"(?m)^[^\r\n]*\bdigest:\s*"
    rb"(sha256:[0-9a-f]{64})\s+size:\s*[0-9]+\s*$"
)
_CONTAINER_FACT_FIELDS = {"python", "pytorch", "cuda", "cudnn", "nccl"}
_BINDING_FIELDS = {
    "schema_version",
    "binding_type",
    "source_commit",
    "source_tree",
    "repository_uri",
    "base_image",
    "base_image_digest",
    "container_image",
    "container_image_digest",
    "build_inputs",
    "repository_transcript_sha256",
    "command_transcript_sha256",
    "container_facts_sha256",
    "inherited_entrypoint",
    "entrypoint_sha256",
    "inspection_artifact_sha256",
    "inspection_artifact",
}
_CONTAINER_FACT_SCRIPT = (
    "import json,platform,torch;"
    "v=torch.cuda.nccl.version();"
    "n='.'.join(map(str,v)) if isinstance(v,tuple) else str(v or '');"
    "print(json.dumps({"
    "'cuda':str(torch.version.cuda or ''),"
    "'cudnn':str(torch.backends.cudnn.version() or ''),"
    "'nccl':n,"
    "'python':platform.python_version(),"
    "'pytorch':str(torch.__version__)"
    "},sort_keys=True,separators=(',',':')))"
)


class BuildPlanError(ValueError):
    """The requested build cannot produce an immutable reviewed image binding."""


class CommandRunner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        environment: Mapping[str, str],
    ) -> bytes:
        """Execute one exact argv and return its combined output."""


class RepositoryReader(Protocol):
    def inspect(self, repository_root: Path | str) -> dict[str, object]:
        """Return measured commit, tree, cleanliness, and command hashes."""


class SubprocessRunner:
    """Run exact Docker argv without a shell or inherited environment."""

    def run(
        self,
        argv: Sequence[str],
        *,
        environment: Mapping[str, str],
    ) -> bytes:
        try:
            completed = subprocess.run(
                list(argv),
                cwd="/",
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False,
                check=False,
                timeout=6 * 60 * 60,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise BuildPlanError("Docker command could not complete") from error
        if completed.returncode != 0:
            raise BuildPlanError(
                f"Docker command failed with exit code {completed.returncode}"
            )
        return bytes(completed.stdout)


class GitRepositoryReader:
    """Measure source identity with fixed local Git commands and no network."""

    def __init__(self, git_binary: str = DEFAULT_GIT_BINARY) -> None:
        self.git_binary = _absolute_path(git_binary, label="Git binary")

    def _run(self, root: str, arguments: Sequence[str]) -> tuple[bytes, str]:
        argv = [self.git_binary, "-C", root, *arguments]
        try:
            completed = subprocess.run(
                argv,
                cwd="/",
                env=dict(MINIMAL_COMMAND_ENVIRONMENT),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise BuildPlanError("Git source identity command failed") from error
        if completed.returncode != 0 or completed.stderr:
            raise BuildPlanError("Git source identity command did not succeed cleanly")
        output = bytes(completed.stdout)
        transcript = _command_transcript_sha256(
            argv,
            MINIMAL_COMMAND_ENVIRONMENT,
            output,
        )
        return output, transcript

    def inspect(self, repository_root: Path | str) -> dict[str, object]:
        root = _absolute_path(repository_root, label="repository root")
        head, head_hash = self._run(root, ("rev-parse", "--verify", "HEAD"))
        tree, tree_hash = self._run(
            root,
            ("rev-parse", "--verify", "HEAD^{tree}"),
        )
        status, status_hash = self._run(
            root,
            ("status", "--porcelain=v1", "--untracked-files=all"),
        )
        try:
            commit = head.decode("ascii").strip()
            source_tree = tree.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise BuildPlanError("Git source identity is not ASCII") from error
        return {
            "source_commit": commit,
            "source_tree": source_tree,
            "clean": status == b"",
            "command_transcript_sha256": {
                "head": head_hash,
                "tree": tree_hash,
                "status": status_hash,
            },
        }


def canonical_json(value: object) -> bytes:
    """Serialize one deterministic operator artifact."""

    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise BuildPlanError("build artifact is not canonical JSON") from error


def _command_transcript_sha256(
    argv: Sequence[str],
    environment: Mapping[str, str],
    output: bytes,
) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "argv": list(argv),
                "environment": dict(environment),
                "output_sha256": hashlib.sha256(output).hexdigest(),
            }
        )
    ).hexdigest()


def _absolute_path(value: Path | str, *, label: str) -> str:
    try:
        path = Path(os.path.abspath(os.fspath(value)))
    except (TypeError, ValueError, OSError) as error:
        raise BuildPlanError(f"{label} must be a filesystem path") from error
    if not path.is_absolute() or "\x00" in str(path):
        raise BuildPlanError(f"{label} must be an absolute filesystem path")
    return str(path)


def _repository(value: object) -> str:
    if not isinstance(value, str) or _PRIVATE_ECR_RE.fullmatch(value) is None:
        raise BuildPlanError(
            "repository URI must be one untagged private Amazon ECR repository"
        )
    return value


def _source_commit(value: object) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise BuildPlanError("source commit must be 40 lowercase hexadecimal digits")
    return value


def _docker_binary(value: Path | str) -> str:
    rendered = _absolute_path(value, label="Docker binary")
    if Path(rendered).name != "docker":
        raise BuildPlanError("Docker binary must name the docker executable")
    return rendered


def image_inspect_argv(
    image: str,
    *,
    docker_binary: str = DEFAULT_DOCKER_BINARY,
) -> tuple[str, ...]:
    return (
        docker_binary,
        "image",
        "inspect",
        "--format",
        "{{json .}}",
        image,
    )


def final_image_inspect_argv(
    image: str,
    *,
    docker_binary: str = DEFAULT_DOCKER_BINARY,
) -> tuple[str, ...]:
    return image_inspect_argv(image, docker_binary=docker_binary)


def container_python_exists_argv(
    image: str,
    *,
    docker_binary: str = DEFAULT_DOCKER_BINARY,
) -> tuple[str, ...]:
    return (
        docker_binary,
        "run",
        "--rm",
        "--network",
        "none",
        "--pull",
        "never",
        "--user",
        "10001:10001",
        "--read-only",
        "--entrypoint",
        "/usr/bin/test",
        image,
        "-x",
        DLC_PYTHON,
    )


def container_facts_argv(
    image: str,
    *,
    docker_binary: str = DEFAULT_DOCKER_BINARY,
) -> tuple[str, ...]:
    return (
        docker_binary,
        "run",
        "--rm",
        "--network",
        "none",
        "--pull",
        "never",
        "--user",
        "10001:10001",
        "--gpus",
        "all",
        "--read-only",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTHONNOUSERSITE=1",
        "--workdir",
        "/",
        "--entrypoint",
        DLC_PYTHON,
        image,
        "-I",
        "-P",
        "-c",
        _CONTAINER_FACT_SCRIPT,
    )


def container_inspection_argv(
    image: str,
    *,
    docker_binary: str = DEFAULT_DOCKER_BINARY,
    container_facts_json: str = "__MEASURED_CONTAINER_FACTS__",
) -> tuple[str, ...]:
    return (
        docker_binary,
        "run",
        "--rm",
        "--network",
        "none",
        "--pull",
        "never",
        "--user",
        "0:0",
        "--read-only",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--cap-add",
        "DAC_READ_SEARCH",
        "--entrypoint",
        DLC_PYTHON,
        image,
        "-I",
        "-P",
        "/opt/memorysplit/inspect_container.py",
        "--container-facts-json",
        container_facts_json,
    )


def _regular_file_sha256(path: Path, *, label: str) -> str:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        before = path.stat(follow_symlinks=False)
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BuildPlanError(f"{label} cannot be opened safely") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or before.st_nlink != 1
            or opened.st_nlink != 1
            or before.st_size != opened.st_size
            or before.st_size > 16 * 1024 * 1024
        ):
            raise BuildPlanError(f"{label} must be one bounded regular file")
        digest = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise BuildPlanError(f"{label} changed while being read")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise BuildPlanError(f"{label} changed while being read")
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        identity = lambda details: (
            details.st_dev,
            details.st_ino,
            details.st_size,
            details.st_mtime_ns,
            details.st_ctime_ns,
            details.st_nlink,
        )
        if identity(before) != identity(after) or identity(after) != identity(
            current
        ):
            raise BuildPlanError(f"{label} changed while being read")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


_BUILD_INPUT_PATHS = {
    "dockerfile_sha256": Path("containers/aws-gpu/Dockerfile"),
    "dockerignore_sha256": Path(
        "containers/aws-gpu/Dockerfile.dockerignore"
    ),
    "dependency_lock_sha256": Path(
        "containers/aws-gpu/requirements.lock"
    ),
    "inspection_script_sha256": Path(
        "containers/aws-gpu/inspect_container.py"
    ),
}


class _PinnedBuildInputs:
    """Hold every build input descriptor and verify its path around commands."""

    def __init__(
        self,
        repository_root: Path | str,
        expected: Mapping[str, str],
    ) -> None:
        self.root = Path(_absolute_path(repository_root, label="repository root"))
        self.expected = dict(expected)
        self.descriptors: dict[str, int] = {}
        self.identities: dict[str, tuple[int, int, int, int, int, int]] = {}

    @staticmethod
    def _identity(details: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            details.st_dev,
            details.st_ino,
            details.st_size,
            details.st_mtime_ns,
            details.st_ctime_ns,
            details.st_nlink,
        )

    @staticmethod
    def _descriptor_bytes(descriptor: int) -> bytes:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)

    def __enter__(self) -> "_PinnedBuildInputs":
        if set(self.expected) != set(_BUILD_INPUT_PATHS):
            raise BuildPlanError("build input hash fields do not match schema")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            for field, relative in _BUILD_INPUT_PATHS.items():
                path = self.root / relative
                descriptor = os.open(path, flags)
                details = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(details.st_mode)
                    or details.st_nlink != 1
                    or details.st_size > 16 * 1024 * 1024
                ):
                    os.close(descriptor)
                    raise BuildPlanError("build input is not one bounded regular file")
                self.descriptors[field] = descriptor
                self.identities[field] = self._identity(details)
            self.verify()
            return self
        except Exception:
            self.close()
            raise

    def bytes_for(self, field: str) -> bytes:
        if field not in self.descriptors:
            raise BuildPlanError("build input descriptor is unavailable")
        return self._descriptor_bytes(self.descriptors[field])

    def verify(self) -> None:
        for field, relative in _BUILD_INPUT_PATHS.items():
            descriptor = self.descriptors.get(field)
            if descriptor is None:
                raise BuildPlanError("build input descriptor is unavailable")
            opened = os.fstat(descriptor)
            try:
                current = (self.root / relative).stat(follow_symlinks=False)
            except OSError as error:
                raise BuildPlanError("build input path changed") from error
            data = self._descriptor_bytes(descriptor)
            if (
                self._identity(opened) != self.identities[field]
                or self._identity(current) != self.identities[field]
                or hashlib.sha256(data).hexdigest() != self.expected[field]
            ):
                raise BuildPlanError("build input changed while applying plan")

    def close(self) -> None:
        for descriptor in self.descriptors.values():
            os.close(descriptor)
        self.descriptors.clear()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def render_build_plan(
    *,
    repository_uri: str,
    source_commit: str,
    repository_root: Path | str,
    docker_config: Path | str,
    docker_binary: Path | str = DEFAULT_DOCKER_BINARY,
    repository_reader: RepositoryReader | None = None,
) -> dict[str, object]:
    """Return exact build and push argv without executing either command."""

    repository = _repository(repository_uri)
    commit = _source_commit(source_commit)
    root = _absolute_path(repository_root, label="repository root")
    inspect_repository = (
        repository_reader if repository_reader is not None else GitRepositoryReader()
    )
    identity = inspect_repository.inspect(root)
    if not isinstance(identity, dict) or set(identity) != {
        "source_commit",
        "source_tree",
        "clean",
        "command_transcript_sha256",
    }:
        raise BuildPlanError("repository identity fields do not match schema")
    source_tree = _source_commit(identity["source_tree"])
    repository_transcripts = identity["command_transcript_sha256"]
    if (
        identity["source_commit"] != commit
        or identity["clean"] is not True
        or not isinstance(repository_transcripts, dict)
        or set(repository_transcripts) != {"head", "tree", "status"}
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in repository_transcripts.values()
        )
    ):
        raise BuildPlanError(
            "repository must be clean and match the requested source commit"
        )
    runtime_root = Path(root) / "containers" / "aws-gpu"
    dockerfile_path = runtime_root / "Dockerfile"
    dockerignore_path = runtime_root / "Dockerfile.dockerignore"
    dependency_lock_path = runtime_root / "requirements.lock"
    inspection_script_path = runtime_root / "inspect_container.py"
    dockerfile = str(dockerfile_path)
    if not Path(root).is_dir():
        raise BuildPlanError("repository root does not contain the AWS GPU Dockerfile")
    inputs = {
        "dockerfile_sha256": _regular_file_sha256(
            dockerfile_path,
            label="Dockerfile",
        ),
        "dockerignore_sha256": _regular_file_sha256(
            dockerignore_path,
            label="Docker context policy",
        ),
        "dependency_lock_sha256": _regular_file_sha256(
            dependency_lock_path,
            label="dependency lock",
        ),
        "inspection_script_sha256": _regular_file_sha256(
            inspection_script_path,
            label="container inspection script",
        ),
    }
    config = _absolute_path(docker_config, label="Docker configuration directory")
    executable = _docker_binary(docker_binary)
    staging_tag = f"{repository}:source-{commit}"
    environment = {
        **MINIMAL_COMMAND_ENVIRONMENT,
        "DOCKER_CONFIG": config,
    }
    binding = {
        "schema_version": 1,
        "operation": "memorysplit-aws-gpu-image-build-v1",
        "platform": PLATFORM,
        "source_commit": commit,
        "source_tree": source_tree,
        "repository_root": root,
        "dockerfile": dockerfile,
        "repository_uri": repository,
        "staging_tag": staging_tag,
        "inputs": inputs,
        "repository_transcript_sha256": dict(repository_transcripts),
        "environment": environment,
        "commands": {
            "build": [
                executable,
                "build",
                "--file",
                dockerfile,
                "--platform",
                PLATFORM,
                "--pull=false",
                "--tag",
                staging_tag,
                root,
            ],
            "python_base": list(
                container_python_exists_argv(
                    BASE_IMAGE,
                    docker_binary=executable,
                )
            ),
            "python_local": list(
                container_python_exists_argv(
                    staging_tag,
                    docker_binary=executable,
                )
            ),
            "inspect_base": list(
                image_inspect_argv(BASE_IMAGE, docker_binary=executable)
            ),
            "inspect_local": list(
                image_inspect_argv(staging_tag, docker_binary=executable)
            ),
            "facts_base": list(
                container_facts_argv(BASE_IMAGE, docker_binary=executable)
            ),
            "facts_local": list(
                container_facts_argv(staging_tag, docker_binary=executable)
            ),
            "inspection_local": list(
                container_inspection_argv(
                    staging_tag,
                    docker_binary=executable,
                )
            ),
            "push": [executable, "push", staging_tag],
        },
        "push_requires_apply": True,
    }
    return binding


def _validated_plan(plan: object) -> dict[str, object]:
    if not isinstance(plan, dict) or set(plan) != _PLAN_FIELDS:
        raise BuildPlanError("image build plan fields do not match schema version 1")
    if (
        type(plan["schema_version"]) is not int
        or plan["schema_version"] != 1
        or plan["operation"] != "memorysplit-aws-gpu-image-build-v1"
        or plan["platform"] != PLATFORM
        or plan["push_requires_apply"] is not True
    ):
        raise BuildPlanError("image build plan identity is invalid")
    environment = plan["environment"]
    commands = plan["commands"]
    inputs = plan["inputs"]
    repository_transcripts = plan["repository_transcript_sha256"]
    if (
        not isinstance(environment, dict)
        or set(environment) != {*MINIMAL_COMMAND_ENVIRONMENT, "DOCKER_CONFIG"}
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment.items()
        )
        or not isinstance(commands, dict)
        or set(commands) != _COMMAND_FIELDS
        or not isinstance(inputs, dict)
        or set(inputs) != _INPUT_FIELDS
        or any(
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in inputs.values()
        )
        or not isinstance(repository_transcripts, dict)
        or set(repository_transcripts) != {"head", "tree", "status"}
        or any(
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in repository_transcripts.values()
        )
        or any(
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(item, str) or not item for item in argv)
            for argv in commands.values()
        )
    ):
        raise BuildPlanError("image build plan command boundary is invalid")
    repository = _repository(plan["repository_uri"])
    commit = _source_commit(plan["source_commit"])
    source_tree = _source_commit(plan["source_tree"])
    root = _absolute_path(plan["repository_root"], label="repository root")
    dockerfile = str(Path(root) / "containers" / "aws-gpu" / "Dockerfile")
    executable = _docker_binary(commands["build"][0])
    staging_tag = f"{repository}:source-{commit}"
    expected_commands = {
        "build": [
            executable,
            "build",
            "--file",
            dockerfile,
            "--platform",
            PLATFORM,
            "--pull=false",
            "--tag",
            staging_tag,
            root,
        ],
        "python_base": list(
            container_python_exists_argv(BASE_IMAGE, docker_binary=executable)
        ),
        "python_local": list(
            container_python_exists_argv(staging_tag, docker_binary=executable)
        ),
        "inspect_base": list(
            image_inspect_argv(BASE_IMAGE, docker_binary=executable)
        ),
        "inspect_local": list(
            image_inspect_argv(staging_tag, docker_binary=executable)
        ),
        "facts_base": list(
            container_facts_argv(BASE_IMAGE, docker_binary=executable)
        ),
        "facts_local": list(
            container_facts_argv(staging_tag, docker_binary=executable)
        ),
        "inspection_local": list(
            container_inspection_argv(staging_tag, docker_binary=executable)
        ),
        "push": [executable, "push", staging_tag],
    }
    expected_inputs = {
        field: _regular_file_sha256(
            Path(root) / relative,
            label=field,
        )
        for field, relative in _BUILD_INPUT_PATHS.items()
    }
    if (
        plan["dockerfile"] != dockerfile
        or plan["staging_tag"] != staging_tag
        or source_tree != plan["source_tree"]
        or commands != expected_commands
        or inputs != expected_inputs
    ):
        raise BuildPlanError("image build input changed after deterministic rendering")
    return plan


def _push_digest(output: object) -> str:
    if not isinstance(output, bytes):
        raise BuildPlanError("Docker push output must be bytes")
    matches = _PUSH_DIGEST_RE.findall(output)
    if len(matches) != 1:
        raise BuildPlanError("Docker push must report exactly one immutable digest")
    digest = matches[0].decode("ascii")
    if _DIGEST_RE.fullmatch(digest) is None:
        raise BuildPlanError("Docker push reported a malformed image digest")
    return digest


def _json_object(data: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuildPlanError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise BuildPlanError(f"{label} must be one JSON object")
    return value


def _image_inspection(data: bytes, *, label: str) -> dict[str, object]:
    value = _json_object(data, label=label)
    config = value.get("Config")
    repo_digests = value.get("RepoDigests")
    image_id = value.get("Id")
    if (
        not isinstance(config, dict)
        or not isinstance(repo_digests, list)
        or any(not isinstance(item, str) for item in repo_digests)
        or not isinstance(image_id, str)
        or _DIGEST_RE.fullmatch(image_id) is None
    ):
        raise BuildPlanError(f"{label} lacks closed image identity")
    entrypoint = config.get("Entrypoint")
    command = config.get("Cmd")
    for field, item in (("Entrypoint", entrypoint), ("Cmd", command)):
        if item is not None and (
            not isinstance(item, list)
            or any(not isinstance(value, str) for value in item)
        ):
            raise BuildPlanError(f"{label} {field} is malformed")
    binding = {
        "image_id": image_id,
        "repo_digests": list(repo_digests),
        "entrypoint": entrypoint,
        "command": command,
    }
    return binding


def _container_facts(data: bytes, *, label: str) -> dict[str, str]:
    value = _json_object(data, label=label)
    if data != canonical_json(value) or set(value) != _CONTAINER_FACT_FIELDS:
        raise BuildPlanError(f"{label} does not match the canonical fact schema")
    facts: dict[str, str] = {}
    for field in sorted(_CONTAINER_FACT_FIELDS):
        item = value[field]
        if not isinstance(item, str) or not item or item != item.strip():
            raise BuildPlanError(f"{label} {field} is not one exact version")
        facts[field] = item
    if (
        not re.fullmatch(r"3\.12(?:\.[0-9]+)+", facts["python"])
        or facts["pytorch"].split("+", 1)[0] != "2.9.0"
        or facts["cuda"] != "13.0"
        or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", facts["cudnn"])
        or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", facts["nccl"])
    ):
        raise BuildPlanError(f"{label} differs from the reviewed framework runtime")
    return facts


def _dependency_lock_packages(data: bytes) -> dict[str, tuple[str, set[str]]]:
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as error:
        raise BuildPlanError("dependency lock is not ASCII") from error
    logical: list[str] = []
    pending = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pending = f"{pending} {line}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        logical.append(pending)
        pending = ""
    if pending or not logical:
        raise BuildPlanError("dependency lock is empty or truncated")
    packages: dict[str, tuple[str, set[str]]] = {}
    for line in logical:
        requirement, *options = line.split()
        if "==" not in requirement:
            raise BuildPlanError("dependency lock contains a floating requirement")
        name, version = requirement.split("==", 1)
        name = name.lower().replace("_", "-")
        hashes = {
            option.removeprefix("--hash=sha256:")
            for option in options
            if option.startswith("--hash=sha256:")
        }
        if (
            not name
            or not version
            or name in packages
            or not hashes
            or len(hashes) != len(options)
            or any(re.fullmatch(r"[0-9a-f]{64}", item) is None for item in hashes)
        ):
            raise BuildPlanError("dependency lock package is malformed")
        packages[name] = (version, hashes)
    return packages


def _file_commitments(value: object, *, label: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise BuildPlanError(f"{label} must be a nonempty file commitment list")
    paths: list[str] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "path",
            "type",
            "commitment_sha256",
        }:
            raise BuildPlanError(f"{label} file fields are not closed")
        if (
            not isinstance(item["path"], str)
            or not item["path"].startswith("/")
            or item["type"] not in {"regular", "symlink", "directory"}
            or not isinstance(item["commitment_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", item["commitment_sha256"])
            is None
        ):
            raise BuildPlanError(f"{label} file commitment is invalid")
        paths.append(item["path"])
    if paths != sorted(set(paths)):
        raise BuildPlanError(f"{label} files are not unique and sorted")
    return value


def _os_package_inventory(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "manager",
        "database_root",
        "database_files",
        "database_tree_sha256",
        "packages",
        "package_count",
    }:
        raise BuildPlanError("OS package inventory fields are not closed")
    if (
        value["manager"] != "dpkg"
        or value["database_root"] != "/var/lib/dpkg"
        or not isinstance(value["packages"], list)
        or not value["packages"]
        or type(value["package_count"]) is not int
        or value["package_count"] != len(value["packages"])
    ):
        raise BuildPlanError("OS package inventory identity is invalid")
    database_files = _file_commitments(
        value["database_files"],
        label="dpkg database",
    )
    if value["database_tree_sha256"] != hashlib.sha256(
        canonical_json(database_files)
    ).hexdigest():
        raise BuildPlanError("dpkg database commitment is invalid")
    names: list[str] = []
    for package in value["packages"]:
        if not isinstance(package, dict) or set(package) != {
            "name",
            "version",
            "architecture",
            "status",
            "installed_files",
            "installed_files_sha256",
        }:
            raise BuildPlanError("OS package fields are not closed")
        name = package["name"]
        if (
            not isinstance(name, str)
            or not name
            or package["status"] != "ii"
        ):
            raise BuildPlanError("OS package name is invalid")
        names.append(name)
        files = _file_commitments(
            package["installed_files"],
            label=f"OS package {name}",
        )
        if package["installed_files_sha256"] != hashlib.sha256(
            canonical_json(files)
        ).hexdigest():
            raise BuildPlanError("OS package file commitment is invalid")
    if names != sorted(set(names)):
        raise BuildPlanError("OS packages are not unique and sorted")


def _inspection_artifact(
    data: bytes,
    *,
    dependency_lock_bytes: bytes,
    expected_facts: Mapping[str, str],
) -> dict[str, object]:
    value = _json_object(data, label="container inspection artifact")
    if data != canonical_json(value) or set(value) != {
        "schema_version",
        "artifact_type",
        "os_release",
        "os_packages",
        "python",
        "container_facts",
        "installed_python_packages",
        "inventory_method",
        "installed_distribution_count",
        "project_install_report_sha256",
    }:
        raise BuildPlanError("container inspection artifact is not closed canonical JSON")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 2
        or value["artifact_type"] != "memorysplit-container-inspection-v2"
        or value["container_facts"] != dict(expected_facts)
        or not isinstance(value["os_release"], dict)
        or set(value["os_release"]) != {"id", "version_id", "pretty_name"}
        or not isinstance(value["python"], dict)
        or set(value["python"]) != {"implementation", "version", "executable"}
        or value["python"]["version"] != expected_facts["python"]
        or value["python"]["executable"] != DLC_PYTHON
        or not isinstance(value["installed_python_packages"], list)
        or value["inventory_method"] != "importlib.metadata.distributions"
        or type(value["installed_distribution_count"]) is not int
        or value["installed_distribution_count"]
        != len(value["installed_python_packages"])
        or not isinstance(value["project_install_report_sha256"], str)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            value["project_install_report_sha256"],
        )
        is None
    ):
        raise BuildPlanError("container inspection artifact identity is invalid")
    _os_package_inventory(value["os_packages"])
    installed: dict[str, dict[str, object]] = {}
    for package in value["installed_python_packages"]:
        if not isinstance(package, dict) or set(package) != {
            "name",
            "version",
            "installer",
            "provenance",
            "metadata_file_sha256",
            "record_file_sha256",
            "wheel_file_sha256",
            "installed_files",
            "installed_files_sha256",
        }:
            raise BuildPlanError("installed package inventory is not closed")
        name = package["name"]
        if not isinstance(name, str) or name in installed:
            raise BuildPlanError("installed package inventory repeats a name")
        provenance = package["provenance"]
        if not isinstance(provenance, dict) or provenance.get("kind") not in {
            "project-wheel",
            "inherited-base-image",
        }:
            raise BuildPlanError("installed package provenance is malformed")
        if provenance["kind"] == "project-wheel":
            if (
                set(provenance) != {"kind", "archive_sha256"}
                or not isinstance(provenance["archive_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", provenance["archive_sha256"])
                is None
            ):
                raise BuildPlanError("project wheel archive provenance is invalid")
        elif (
            set(provenance) != {"kind", "base_image_digest"}
            or provenance["base_image_digest"] != BASE_DIGEST
        ):
            raise BuildPlanError("inherited package base provenance is invalid")
        metadata_hashes = {
            package[field]
            for field in (
                "metadata_file_sha256",
                "record_file_sha256",
                "wheel_file_sha256",
            )
            if isinstance(package[field], str)
            and re.fullmatch(r"[0-9a-f]{64}", package[field]) is not None
        }
        if len(metadata_hashes) != 3:
            raise BuildPlanError("installed package metadata hashes are malformed")
        files = _file_commitments(
            package["installed_files"],
            label=f"installed package {name}",
        )
        if (
            package["installed_files_sha256"]
            != hashlib.sha256(canonical_json(files)).hexdigest()
            or not metadata_hashes
            <= {item["commitment_sha256"] for item in files}
        ):
            raise BuildPlanError("installed package file commitments are invalid")
        installed[name] = package
    if (
        list(installed) != sorted(installed)
        or "torch" not in installed
        or installed["torch"]["provenance"]
        != {
            "kind": "inherited-base-image",
            "base_image_digest": BASE_DIGEST,
        }
    ):
        raise BuildPlanError(
            "installed inventory is incomplete, unsorted, or lacks inherited Torch"
        )
    locked = _dependency_lock_packages(dependency_lock_bytes)
    for name, (version, allowed_hashes) in locked.items():
        package = installed.get(name)
        if (
            package is None
            or package["version"] != version
            or package["provenance"].get("kind") != "project-wheel"
            or package["provenance"].get("archive_sha256") not in allowed_hashes
        ):
            raise BuildPlanError(
                "installed project package is not bound to its selected lock hash"
            )
    return value


def parse_image_binding_bytes(
    data: bytes,
    *,
    dependency_lock_bytes: bytes,
) -> dict[str, object]:
    """Parse the complete build authority and its measured inspection artifact."""

    value = _json_object(data, label="image binding")
    if data != canonical_json(value) or set(value) != _BINDING_FIELDS:
        raise BuildPlanError("image binding is not closed canonical JSON")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 2
        or value["binding_type"] != "memorysplit-aws-gpu-image-binding-v2"
        or value["base_image"] != BASE_IMAGE
        or value["base_image_digest"] != BASE_DIGEST
    ):
        raise BuildPlanError("image binding identity is invalid")
    _source_commit(value["source_commit"])
    _source_commit(value["source_tree"])
    repository = _repository(value["repository_uri"])
    digest = value["container_image_digest"]
    if (
        not isinstance(digest, str)
        or _DIGEST_RE.fullmatch(digest) is None
        or value["container_image"] != f"{repository}@{digest}"
    ):
        raise BuildPlanError("image binding final digest is invalid")
    build_inputs = value["build_inputs"]
    if (
        not isinstance(build_inputs, dict)
        or set(build_inputs) != _INPUT_FIELDS
        or any(
            not isinstance(item, str)
            or re.fullmatch(r"[0-9a-f]{64}", item) is None
            for item in build_inputs.values()
        )
        or build_inputs["dependency_lock_sha256"]
        != hashlib.sha256(dependency_lock_bytes).hexdigest()
    ):
        raise BuildPlanError("image binding build inputs are invalid")
    repository_transcripts = value["repository_transcript_sha256"]
    if (
        not isinstance(repository_transcripts, dict)
        or set(repository_transcripts) != {"plan", "pre_apply", "final"}
        or any(
            not isinstance(group, dict)
            or set(group) != {"head", "tree", "status"}
            or any(
                not isinstance(item, str)
                or re.fullmatch(r"[0-9a-f]{64}", item) is None
                for item in group.values()
            )
            for group in repository_transcripts.values()
        )
    ):
        raise BuildPlanError("repository transcript binding is invalid")
    command_transcripts = value["command_transcript_sha256"]
    expected_commands = {
        "build",
        "python_base",
        "python_local",
        "inspect_base",
        "inspect_local",
        "facts_base",
        "facts_local",
        "inspection_local",
        "push",
        "python_final",
        "inspect_final",
        "facts_final",
        "inspection_final",
    }
    if (
        not isinstance(command_transcripts, dict)
        or set(command_transcripts) != expected_commands
        or any(
            not isinstance(item, str)
            or re.fullmatch(r"[0-9a-f]{64}", item) is None
            for item in command_transcripts.values()
        )
    ):
        raise BuildPlanError("Docker command transcript binding is invalid")
    artifact = value["inspection_artifact"]
    if not isinstance(artifact, dict):
        raise BuildPlanError("image binding lacks inspection artifact")
    facts = _container_facts(
        canonical_json(artifact.get("container_facts")),
        label="binding container facts",
    )
    parsed_artifact = _inspection_artifact(
        canonical_json(artifact),
        dependency_lock_bytes=dependency_lock_bytes,
        expected_facts=facts,
    )
    hash_fields = {
        "container_facts_sha256": hashlib.sha256(
            canonical_json(facts)
        ).hexdigest(),
        "inspection_artifact_sha256": hashlib.sha256(
            canonical_json(parsed_artifact)
        ).hexdigest(),
    }
    if any(value[field] != expected for field, expected in hash_fields.items()):
        raise BuildPlanError("image binding measured artifact hash is inconsistent")
    inherited_entrypoint = value["inherited_entrypoint"]
    if (
        not isinstance(inherited_entrypoint, dict)
        or set(inherited_entrypoint) != {"entrypoint", "command"}
        or value["entrypoint_sha256"]
        != hashlib.sha256(canonical_json(inherited_entrypoint)).hexdigest()
    ):
        raise BuildPlanError("image binding entrypoint hash is invalid")
    return value


def execute_build_plan(
    plan: dict[str, object],
    *,
    apply: bool,
    runner: CommandRunner | None = None,
    repository_reader: RepositoryReader | None = None,
) -> dict[str, object]:
    """Execute build and push only when ``apply`` is the exact boolean true."""

    checked = _validated_plan(plan)
    if type(apply) is not bool:
        raise BuildPlanError("apply must be an exact boolean")
    if not apply:
        return checked
    command_runner = runner if runner is not None else SubprocessRunner()
    inspect_repository = (
        repository_reader if repository_reader is not None else GitRepositoryReader()
    )
    pre_apply_repository = inspect_repository.inspect(checked["repository_root"])
    if (
        not isinstance(pre_apply_repository, dict)
        or pre_apply_repository.get("source_commit") != checked["source_commit"]
        or pre_apply_repository.get("source_tree") != checked["source_tree"]
        or pre_apply_repository.get("clean") is not True
        or not isinstance(
            pre_apply_repository.get("command_transcript_sha256"),
            dict,
        )
    ):
        raise BuildPlanError("source repository changed before image apply")
    commands = checked["commands"]
    environment = checked["environment"]
    repository = str(checked["repository_uri"])
    transcripts: dict[str, str] = {}
    with _PinnedBuildInputs(
        checked["repository_root"],
        checked["inputs"],
    ) as pinned:

        def run_bound(name: str, argv: Sequence[str]) -> bytes:
            pinned.verify()
            output = command_runner.run(argv, environment=environment)
            pinned.verify()
            transcripts[name] = _command_transcript_sha256(
                argv,
                environment,
                output,
            )
            return output

        run_bound("build", commands["build"])
        run_bound("python_base", commands["python_base"])
        run_bound("python_local", commands["python_local"])
        base_inspect = _image_inspection(
            run_bound("inspect_base", commands["inspect_base"]),
            label="base image inspection",
        )
        local_inspect = _image_inspection(
            run_bound("inspect_local", commands["inspect_local"]),
            label="local image inspection",
        )
        if BASE_IMAGE not in base_inspect["repo_digests"]:
            raise BuildPlanError("local base image does not carry the reviewed digest")
        if (
            local_inspect["entrypoint"] != base_inspect["entrypoint"]
            or local_inspect["command"] != base_inspect["command"]
        ):
            raise BuildPlanError("built image changed inherited entrypoint behavior")

        base_facts = _container_facts(
            run_bound("facts_base", commands["facts_base"]),
            label="base container facts",
        )
        local_facts = _container_facts(
            run_bound("facts_local", commands["facts_local"]),
            label="local container facts",
        )
        if local_facts != base_facts:
            raise BuildPlanError("built image changed inherited framework facts")
        measured_facts_json = canonical_json(local_facts).decode("ascii").rstrip(
            "\n"
        )
        local_inspection_argv = container_inspection_argv(
            checked["staging_tag"],
            docker_binary=commands["build"][0],
            container_facts_json=measured_facts_json,
        )
        local_artifact = _inspection_artifact(
            run_bound("inspection_local", local_inspection_argv),
            dependency_lock_bytes=pinned.bytes_for("dependency_lock_sha256"),
            expected_facts=local_facts,
        )

        push_output = run_bound("push", commands["push"])
        digest = _push_digest(push_output)
        final_image = f"{repository}@{digest}"
        docker_binary = commands["build"][0]
        final_python_argv = container_python_exists_argv(
            final_image,
            docker_binary=docker_binary,
        )
        final_inspect_argv = final_image_inspect_argv(
            final_image,
            docker_binary=docker_binary,
        )
        final_facts_argv = container_facts_argv(
            final_image,
            docker_binary=docker_binary,
        )
        run_bound("python_final", final_python_argv)
        final_inspect = _image_inspection(
            run_bound("inspect_final", final_inspect_argv),
            label="final digest image inspection",
        )
        if (
            final_image not in final_inspect["repo_digests"]
            or final_inspect["image_id"] != local_inspect["image_id"]
            or final_inspect["entrypoint"] != base_inspect["entrypoint"]
            or final_inspect["command"] != base_inspect["command"]
        ):
            raise BuildPlanError(
                "final digest does not preserve local image identity and entrypoint"
            )
        final_facts = _container_facts(
            run_bound("facts_final", final_facts_argv),
            label="final digest container facts",
        )
        if final_facts != local_facts:
            raise BuildPlanError("final digest framework facts differ after push")
        final_artifact_argv = container_inspection_argv(
            final_image,
            docker_binary=docker_binary,
            container_facts_json=canonical_json(final_facts)
            .decode("ascii")
            .rstrip("\n"),
        )
        final_artifact = _inspection_artifact(
            run_bound("inspection_final", final_artifact_argv),
            dependency_lock_bytes=pinned.bytes_for("dependency_lock_sha256"),
            expected_facts=final_facts,
        )
        if final_artifact != local_artifact:
            raise BuildPlanError("final digest inspection differs after push")
        pinned.verify()
        dependency_lock_bytes = pinned.bytes_for("dependency_lock_sha256")

    final_repository = inspect_repository.inspect(checked["repository_root"])
    if (
        not isinstance(final_repository, dict)
        or final_repository.get("source_commit") != checked["source_commit"]
        or final_repository.get("source_tree") != checked["source_tree"]
        or final_repository.get("clean") is not True
        or not isinstance(
            final_repository.get("command_transcript_sha256"),
            dict,
        )
    ):
        raise BuildPlanError("source repository changed during image apply")
    entrypoint_binding = {
        "entrypoint": base_inspect["entrypoint"],
        "command": base_inspect["command"],
    }
    binding = {
        "schema_version": 2,
        "binding_type": "memorysplit-aws-gpu-image-binding-v2",
        "source_commit": str(checked["source_commit"]),
        "source_tree": str(checked["source_tree"]),
        "repository_uri": repository,
        "base_image": BASE_IMAGE,
        "base_image_digest": BASE_DIGEST,
        "container_image": final_image,
        "container_image_digest": digest,
        "build_inputs": dict(checked["inputs"]),
        "repository_transcript_sha256": {
            "plan": dict(checked["repository_transcript_sha256"]),
            "pre_apply": dict(
                pre_apply_repository["command_transcript_sha256"]
            ),
            "final": dict(final_repository["command_transcript_sha256"]),
        },
        "command_transcript_sha256": transcripts,
        "container_facts_sha256": hashlib.sha256(
            canonical_json(final_facts)
        ).hexdigest(),
        "entrypoint_sha256": hashlib.sha256(
            canonical_json(entrypoint_binding)
        ).hexdigest(),
        "inherited_entrypoint": entrypoint_binding,
        "inspection_artifact_sha256": hashlib.sha256(
            canonical_json(final_artifact)
        ).hexdigest(),
        "inspection_artifact": final_artifact,
    }
    return parse_image_binding_bytes(
        canonical_json(binding),
        dependency_lock_bytes=dependency_lock_bytes,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a digest-pinned AWS GPU image build. Build and push occur "
            "only when --apply is supplied."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--repository-uri", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    parser.add_argument("--docker-config", required=True)
    parser.add_argument("--docker-binary", default=DEFAULT_DOCKER_BINARY)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: CommandRunner | None = None,
    repository_reader: RepositoryReader | None = None,
) -> int:
    arguments = _parser().parse_args(argv)
    try:
        plan = render_build_plan(
            repository_uri=arguments.repository_uri,
            source_commit=arguments.source_commit,
            repository_root=arguments.repository_root,
            docker_config=arguments.docker_config,
            docker_binary=arguments.docker_binary,
            repository_reader=repository_reader,
        )
        artifact = execute_build_plan(
            plan,
            apply=arguments.apply,
            runner=runner,
            repository_reader=repository_reader,
        )
    except BuildPlanError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_json(artifact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
