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
    "repository_root",
    "dockerfile",
    "repository_uri",
    "staging_tag",
    "inputs",
    "environment",
    "commands",
    "push_requires_apply",
}
_COMMAND_FIELDS = {"build", "push"}
_INPUT_FIELDS = {
    "dockerfile_sha256",
    "dockerignore_sha256",
    "dependency_lock_sha256",
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


def render_build_plan(
    *,
    repository_uri: str,
    source_commit: str,
    repository_root: Path | str,
    docker_config: Path | str,
    docker_binary: Path | str = DEFAULT_DOCKER_BINARY,
) -> dict[str, object]:
    """Return exact build and push argv without executing either command."""

    repository = _repository(repository_uri)
    commit = _source_commit(source_commit)
    root = _absolute_path(repository_root, label="repository root")
    runtime_root = Path(root) / "containers" / "aws-gpu"
    dockerfile_path = runtime_root / "Dockerfile"
    dockerignore_path = runtime_root / "Dockerfile.dockerignore"
    dependency_lock_path = runtime_root / "requirements.lock"
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
    }
    config = _absolute_path(docker_config, label="Docker configuration directory")
    executable = _docker_binary(docker_binary)
    staging_tag = f"{repository}:source-{commit}"
    environment = {
        **MINIMAL_COMMAND_ENVIRONMENT,
        "DOCKER_CONFIG": config,
    }
    return {
        "schema_version": 1,
        "operation": "memorysplit-aws-gpu-image-build-v1",
        "platform": PLATFORM,
        "source_commit": commit,
        "repository_root": root,
        "dockerfile": dockerfile,
        "repository_uri": repository,
        "staging_tag": staging_tag,
        "inputs": inputs,
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
            "push": [executable, "push", staging_tag],
        },
        "push_requires_apply": True,
    }


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
        or any(
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(item, str) or not item for item in argv)
            for argv in commands.values()
        )
    ):
        raise BuildPlanError("image build plan command boundary is invalid")
    expected = render_build_plan(
        repository_uri=_repository(plan["repository_uri"]),
        source_commit=_source_commit(plan["source_commit"]),
        repository_root=str(plan["repository_root"]),
        docker_config=environment["DOCKER_CONFIG"],
        docker_binary=commands["build"][0],
    )
    if plan != expected:
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


def execute_build_plan(
    plan: dict[str, object],
    *,
    apply: bool,
    runner: CommandRunner | None = None,
) -> dict[str, object]:
    """Execute build and push only when ``apply`` is the exact boolean true."""

    checked = _validated_plan(plan)
    if type(apply) is not bool:
        raise BuildPlanError("apply must be an exact boolean")
    if not apply:
        return checked
    command_runner = runner if runner is not None else SubprocessRunner()
    commands = checked["commands"]
    environment = checked["environment"]
    command_runner.run(commands["build"], environment=environment)
    push_output = command_runner.run(commands["push"], environment=environment)
    digest = _push_digest(push_output)
    repository = str(checked["repository_uri"])
    return {
        "schema_version": 1,
        "source_commit": str(checked["source_commit"]),
        "repository_uri": repository,
        "container_image": f"{repository}@{digest}",
        "container_image_digest": digest,
    }


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
) -> int:
    arguments = _parser().parse_args(argv)
    try:
        plan = render_build_plan(
            repository_uri=arguments.repository_uri,
            source_commit=arguments.source_commit,
            repository_root=arguments.repository_root,
            docker_config=arguments.docker_config,
            docker_binary=arguments.docker_binary,
        )
        artifact = execute_build_plan(
            plan,
            apply=arguments.apply,
            runner=runner,
        )
    except BuildPlanError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_json(artifact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
