from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from scripts import build_aws_gpu_image as image_build


ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "containers" / "aws-gpu"
DESTINATION_REPOSITORY = (
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/memorysplit/aws-gpu"
)


def test_dockerfile_uses_digest_pinned_base_and_nonroot_runtime_only():
    dockerfile = (CONTEXT / "Dockerfile").read_text(encoding="utf-8")
    lock = json.loads((CONTEXT / "image.lock.json").read_text(encoding="utf-8"))

    assert lock["base"]["image"] == (
        "public.ecr.aws/deep-learning-containers/"
        "pytorch:2.12.1-cu130-amzn2023"
    )
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", lock["base"]["digest"])
    assert "FROM ${BASE_IMAGE}@${BASE_DIGEST}" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert lock["runtime"]["pytorch"] == "2.12.1"
    assert lock["runtime"]["cuda"] == "13.0"
    copy_lines = [
        line.strip()
        for line in dockerfile.splitlines()
        if line.strip().startswith(("COPY ", "ADD "))
    ]
    assert copy_lines == [
        "COPY image.lock.json requirements.lock /opt/memorysplit-build/"
    ]
    assert "/opt/venv/bin/python -m pip install" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "COPY . " not in dockerfile


def test_build_plan_renders_exact_build_then_immutable_ecr_push():
    plan = image_build.build_image_plan(DESTINATION_REPOSITORY)
    digest = plan["build_context_sha256"]
    destination = f"{DESTINATION_REPOSITORY}:build-{digest}"

    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert plan["destination"] == destination
    build, push = plan["commands"]
    assert build[:3] == ["docker", "build", "--pull=false"]
    assert (
        f"BASE_DIGEST={json.loads((CONTEXT / 'image.lock.json').read_text())['base']['digest']}"
        in build
    )
    assert build[build.index("--tag") + 1] == destination
    assert push == ["docker", "push", destination]


def test_dry_run_main_never_executes_docker(monkeypatch, capsys):
    called = False

    def forbidden(_plan):
        nonlocal called
        called = True
        raise AssertionError("dry run executed Docker")

    monkeypatch.setattr(image_build, "apply_image_plan", forbidden)

    assert (
        image_build.main(["--destination", DESTINATION_REPOSITORY]) == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True
    assert report["applied"] is False
    assert called is False


def test_apply_executes_only_the_previously_rendered_argv():
    plan = image_build.build_image_plan(DESTINATION_REPOSITORY)
    calls = []

    def runner(argv, **kwargs):
        calls.append((list(argv), kwargs))

    image_build.apply_image_plan(plan, runner=runner)

    assert [call[0] for call in calls] == plan["commands"]
    assert all(call[1]["shell"] is False for call in calls)
    assert all(call[1]["check"] is True for call in calls)


@pytest.mark.parametrize(
    "destination",
    [
        "public.ecr.aws/memorysplit/aws-gpu",
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/aws-gpu:latest",
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/aws-gpu@sha256:"
        + "a" * 64,
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/aws-gpu:build-"
        + "0" * 64,
    ],
)
def test_build_plan_rejects_nonimmutable_or_nonprivate_ecr_destination(
    destination,
):
    with pytest.raises(image_build.ImageBuildError):
        image_build.build_image_plan(destination)


def test_build_plan_rejects_unpinned_base_lock(tmp_path):
    context = tmp_path / "aws-gpu"
    context.mkdir()
    shutil.copy2(CONTEXT / "Dockerfile", context / "Dockerfile")
    lock = json.loads((CONTEXT / "image.lock.json").read_text(encoding="utf-8"))
    lock["base"]["digest"] = "latest"
    (context / "image.lock.json").write_text(
        json.dumps(lock),
        encoding="utf-8",
    )

    with pytest.raises(image_build.ImageBuildError, match="lock"):
        image_build.build_image_plan(DESTINATION_REPOSITORY, context_dir=context)
