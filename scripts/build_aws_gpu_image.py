#!/usr/bin/env python3
"""Render or apply the locked AWS GPU container build and push commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTEXT = ROOT / "containers" / "aws-gpu"
EXPECTED_BASE_IMAGE = (
    "public.ecr.aws/deep-learning-containers/"
    "pytorch:2.12.1-cu130-amzn2023"
)
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ECR_REPOSITORY_PATTERN = (
    r"[0-9]{12}\.dkr\.ecr\."
    r"[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+"
    r"\.amazonaws\.com(?:\.cn)?/"
    r"[a-z0-9]+(?:[._/-][a-z0-9]+)*"
)
_ECR_REPOSITORY_RE = re.compile(rf"^{_ECR_REPOSITORY_PATTERN}$")
_IMMUTABLE_ECR_RE = re.compile(
    rf"^(?P<repository>{_ECR_REPOSITORY_PATTERN})"
    r":build-(?P<digest>[0-9a-f]{64})$"
)
_LOCK_ROOT_FIELDS = {"schema_version", "base", "runtime"}
_LOCK_BASE_FIELDS = {"image", "digest"}
_LOCK_RUNTIME_FIELDS = {
    "pytorch",
    "cuda",
    "operating_system",
    "python",
    "requirements_sha256",
    "user",
    "uid",
    "gid",
}


class ImageBuildError(ValueError):
    """The local image build contract is incomplete or mutable."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ImageBuildError(f"image lock repeats field: {key}")
        value[key] = item
    return value


def _context_path(value: Path | str) -> Path:
    candidate = Path(value)
    if candidate.is_symlink() or not candidate.is_dir():
        raise ImageBuildError("build context must be a real directory")
    return candidate.resolve(strict=True)


def _regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        before = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ImageBuildError(f"{label} is unavailable") from error
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise ImageBuildError(f"{label} must be a singly linked regular file")
    payload = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ImageBuildError(f"{label} changed while being read")
    return payload


def _load_lock(context: Path) -> tuple[dict[str, object], bytes]:
    payload = _regular_bytes(context / "image.lock.json", label="image lock")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ImageBuildError(f"image lock contains non-finite {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ImageBuildError("image lock must contain valid UTF-8 JSON") from error
    if not isinstance(value, dict) or set(value) != _LOCK_ROOT_FIELDS:
        raise ImageBuildError("image lock root fields do not match")
    base = value["base"]
    runtime = value["runtime"]
    if (
        value["schema_version"] != 1
        or type(value["schema_version"]) is not int
        or not isinstance(base, dict)
        or set(base) != _LOCK_BASE_FIELDS
        or base["image"] != EXPECTED_BASE_IMAGE
        or not isinstance(base["digest"], str)
        or _DIGEST_RE.fullmatch(base["digest"]) is None
        or not isinstance(runtime, dict)
        or set(runtime) != _LOCK_RUNTIME_FIELDS
        or runtime
        != {
            "pytorch": "2.12.1",
            "cuda": "13.0",
            "operating_system": "Amazon Linux 2023",
            "python": "3.11",
            "requirements_sha256": hashlib.sha256(
                _regular_bytes(
                    context / "requirements.lock",
                    label="runtime dependency lock",
                )
            ).hexdigest(),
            "user": "memorysplit",
            "uid": 10001,
            "gid": 10001,
        }
    ):
        raise ImageBuildError("image lock does not match the reviewed AWS GPU base")
    return value, payload


def build_context_sha256(context_dir: Path | str = DEFAULT_CONTEXT) -> str:
    """Hash the only three files admitted to the image build context."""

    context = _context_path(context_dir)
    names = {entry.name for entry in context.iterdir()}
    if names != {"Dockerfile", "image.lock.json", "requirements.lock"}:
        raise ImageBuildError(
            "AWS GPU build context may contain only Dockerfile and its two locks"
        )
    dockerfile = _regular_bytes(context / "Dockerfile", label="Dockerfile")
    requirements = _regular_bytes(
        context / "requirements.lock",
        label="runtime dependency lock",
    )
    _lock, lock_payload = _load_lock(context)
    digest = hashlib.sha256()
    for name, payload in (
        ("Dockerfile", dockerfile),
        ("image.lock.json", lock_payload),
        ("requirements.lock", requirements),
    ):
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def immutable_ecr_destination(
    value: str,
    *,
    context_sha256: str,
) -> str:
    """Derive or verify the content-bound immutable ECR build tag."""

    if not isinstance(value, str):
        raise ImageBuildError("ECR destination must be a string")
    if re.fullmatch(r"[0-9a-f]{64}", context_sha256) is None:
        raise ImageBuildError("build context SHA-256 is invalid")
    if _ECR_REPOSITORY_RE.fullmatch(value) is not None:
        return f"{value}:build-{context_sha256}"
    match = _IMMUTABLE_ECR_RE.fullmatch(value)
    if match is None or match.group("digest") != context_sha256:
        raise ImageBuildError(
            "ECR destination must be an exact repository or its matching "
            "build-<context-sha256> immutable tag"
        )
    return value


def build_image_plan(
    destination: str,
    *,
    context_dir: Path | str = DEFAULT_CONTEXT,
) -> dict[str, object]:
    """Return exact argv for a local Docker build followed by one push."""

    context = _context_path(context_dir)
    lock, _payload = _load_lock(context)
    context_digest = build_context_sha256(context)
    target = immutable_ecr_destination(
        destination,
        context_sha256=context_digest,
    )
    base = lock["base"]
    runtime = lock["runtime"]
    commands = [
        [
            "docker",
            "build",
            "--pull=false",
            "--file",
            str(context / "Dockerfile"),
            "--build-arg",
            f"BASE_IMAGE={base['image']}",
            "--build-arg",
            f"BASE_DIGEST={base['digest']}",
            "--build-arg",
            f"RUNTIME_UID={runtime['uid']}",
            "--build-arg",
            f"RUNTIME_GID={runtime['gid']}",
            "--label",
            f"org.opencontainers.image.base.digest={base['digest']}",
            "--label",
            f"com.memorysplit.build-context-sha256={context_digest}",
            "--tag",
            target,
            str(context),
        ],
        ["docker", "push", target],
    ]
    return {
        "schema_version": 1,
        "base_image": f"{base['image']}@{base['digest']}",
        "build_context_sha256": context_digest,
        "destination": target,
        "commands": commands,
    }


def apply_image_plan(
    plan: dict[str, object],
    *,
    runner: Callable[..., object] = subprocess.run,
) -> None:
    """Execute a previously rendered plan without a shell."""

    commands = plan.get("commands")
    if not isinstance(commands, list) or len(commands) != 2:
        raise ImageBuildError("image plan must contain build and push commands")
    for command in commands:
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) for item in command)
        ):
            raise ImageBuildError("image plan contains invalid argv")
        runner(
            command,
            check=True,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr,
            stderr=sys.stderr,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render the locked AWS GPU Docker build and immutable ECR push. "
            "No command executes unless --apply is present."
        )
    )
    parser.add_argument(
        "--destination",
        required=True,
        help=(
            "private ECR repository, or its exact build-<context-sha256> tag"
        ),
    )
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        plan = build_image_plan(arguments.destination)
        if arguments.apply:
            apply_image_plan(plan)
        report = {
            **plan,
            "dry_run": not arguments.apply,
            "applied": arguments.apply,
        }
        print(
            json.dumps(
                report,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
        return 0
    except (ImageBuildError, OSError, subprocess.SubprocessError) as error:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": False,
                    "error": str(error),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
