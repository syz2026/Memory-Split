from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import sys
import uuid
from pathlib import Path

import pytest

from cluster.aws.p5.attest_environment import parse_runtime_lock_bytes


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = ROOT / "containers" / "aws-gpu"
DOCKERFILE = RUNTIME_ROOT / "Dockerfile"
REQUIREMENTS_INPUT = RUNTIME_ROOT / "requirements.in"
REQUIREMENTS_LOCK = RUNTIME_ROOT / "requirements.lock"
BUILD_SCRIPT = RUNTIME_ROOT / "build_image.py"
RUNTIME_LOCK_SCRIPT = RUNTIME_ROOT / "runtime_lock.py"
HOST_CANDIDATE = RUNTIME_ROOT / "host-candidate.json"
DOCKERIGNORE = RUNTIME_ROOT / "Dockerfile.dockerignore"

BASE_REGISTRY = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training"
)
BASE_DIGEST = (
    "sha256:1414a836532f22b271c03b7ccdbdff3d"
    "aa0591975b3bd9a3cf51601a45b37f4f"
)
BASE_IMAGE = f"{BASE_REGISTRY}@{BASE_DIGEST}"
PRIVATE_REPOSITORY = (
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/memorysplit/aws-gpu"
)
SOURCE_COMMIT = "b3471e0969ca2a997d33acf60d2e777720afa1c4"


def _locked_requirements(data: str) -> dict[str, tuple[str, ...]]:
    logical_lines: list[str] = []
    pending = ""
    for raw_line in data.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pending = f"{pending} {line}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        logical_lines.append(pending)
        pending = ""
    assert not pending, "requirements lock ends with a continuation"

    locked: dict[str, tuple[str, ...]] = {}
    for line in logical_lines:
        requirement, *hash_parts = line.split()
        assert re.fullmatch(
            r"[a-z0-9][a-z0-9._-]*==[0-9][0-9A-Za-z.!+_-]*",
            requirement,
        ), requirement
        name = requirement.split("==", 1)[0].lower().replace("_", "-")
        assert name not in locked, f"duplicate locked package: {name}"
        hashes = tuple(
            part.removeprefix("--hash=sha256:")
            for part in hash_parts
            if part.startswith("--hash=sha256:")
        )
        assert hashes, f"missing hashes for {requirement}"
        assert all(re.fullmatch(r"[0-9a-f]{64}", item) for item in hashes)
        assert len(hashes) == len(hash_parts), f"unlocked option in {line}"
        locked[name] = hashes
    return locked


def _load_script(path: Path):
    assert path.is_file(), f"missing script: {path}"
    name = f"aws_gpu_runtime_test_{path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _canonical(value: object) -> bytes:
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


def _runtime_spec() -> dict[str, object]:
    digest = "sha256:" + "9" * 64
    return {
        "schema_version": 1,
        "source": {
            "commit": SOURCE_COMMIT,
            "tree": "c" * 40,
        },
        "control_bundle_sha256": "d" * 64,
        "profile_sha256": "e" * 64,
        "container": {
            "base_image": BASE_IMAGE,
            "platform": "linux/amd64",
            "image_binding": {
                "schema_version": 1,
                "source_commit": SOURCE_COMMIT,
                "repository_uri": PRIVATE_REPOSITORY,
                "container_image": f"{PRIVATE_REPOSITORY}@{digest}",
                "container_image_digest": digest,
            },
            "versions": {
                "python": "3.12.11",
                "pytorch": "2.9.0+cu130",
                "cuda": "13.0",
                "cudnn": "9.10.2",
                "nccl": "2.28.3",
            },
        },
        "host_runtime_versions": {
            "python": "3.12.3",
            "pytorch": "2.9.0+cu132",
            "cuda": "13.2",
            "cudnn": "9.10.1",
            "nccl": "2.28.1",
            "nvidia_driver": "595.71.05",
            "fabric_manager": "595.71.05",
            "docker": "28.5.1",
            "nvidia_container_runtime": "1.18.0",
            "aws_cli": "2.31.7",
        },
    }


class _ForbiddenRunner:
    def run(self, argv, *, environment):
        raise AssertionError(f"unexpected command execution: {argv!r}")


class _RecordingRunner:
    def __init__(self, outputs: list[bytes]) -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def run(self, argv, *, environment):
        self.calls.append((tuple(argv), dict(environment)))
        return self.outputs.pop(0)


def test_dockerfile_uses_only_the_approved_digest_pinned_base():
    text = DOCKERFILE.read_text(encoding="utf-8")
    instructions = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    from_lines = [line for line in instructions if line.upper().startswith("FROM ")]

    assert from_lines == [f"FROM {BASE_IMAGE}"]
    assert "linux/amd64" in text
    assert "--require-hashes" in text
    assert "--no-deps" in text
    assert "--only-binary=:all:" in text
    assert "--no-compile" in text
    assert "requirements.lock" in text
    assert re.search(r"(?m)^USER 10001:10001$", text)
    assert "# syntax=" not in text
    assert "ARG APP_UID" not in text
    assert "ARG APP_GID" not in text
    assert not re.search(r"(?im)^\s*ADD\s+https?://", text)
    assert not re.search(r"(?i)\b(curl|wget)\b", text)
    assert "apt-get" not in text
    assert not (ROOT / "runtime" / "aws-p5").exists()


def test_project_dependency_input_is_fully_hash_locked():
    source = REQUIREMENTS_INPUT.read_text(encoding="utf-8")
    lock = REQUIREMENTS_LOCK.read_text(encoding="utf-8")
    locked = _locked_requirements(lock)

    assert "torch" not in locked
    assert "torch" in source
    assert "digest-pinned DLC" in source
    assert {
        "numpy",
        "tiktoken",
        "pyyaml",
        "tqdm",
        "matplotlib",
        "datasets",
        "huggingface-hub",
        "pytest",
    } <= set(locked)
    assert "latest" not in lock.lower()
    assert "http://" not in lock
    assert "https://" not in lock
    assert "--index-url" not in lock
    assert "--extra-index-url" not in lock
    assert "-e " not in lock


def test_docker_context_is_a_closed_dependency_only_allowlist():
    rules = DOCKERIGNORE.read_text(encoding="utf-8").splitlines()

    assert rules == [
        "**",
        "!containers/",
        "!containers/aws-gpu/",
        "!containers/aws-gpu/requirements.lock",
    ]


def test_build_plan_is_dry_run_and_does_not_inherit_secrets(monkeypatch, tmp_path):
    module = _load_script(BUILD_SCRIPT)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "must-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-leak")
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-leak")
    docker_config = tmp_path / "docker"

    plan = module.render_build_plan(
        repository_uri=PRIVATE_REPOSITORY,
        source_commit=SOURCE_COMMIT,
        repository_root=ROOT,
        docker_config=docker_config,
    )
    result = module.execute_build_plan(
        plan,
        apply=False,
        runner=_ForbiddenRunner(),
    )

    assert result == plan
    assert plan["schema_version"] == 1
    assert plan["platform"] == "linux/amd64"
    assert plan["push_requires_apply"] is True
    assert plan["repository_uri"] == PRIVATE_REPOSITORY
    assert plan["staging_tag"].startswith(PRIVATE_REPOSITORY + ":source-")
    assert plan["commands"]["build"][0] == "/usr/bin/docker"
    assert "--pull=false" in plan["commands"]["build"]
    assert plan["commands"]["push"] == [
        "/usr/bin/docker",
        "push",
        plan["staging_tag"],
    ]
    assert plan["environment"] == {
        "DOCKER_CONFIG": str(docker_config.resolve()),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    }
    serialized = module.canonical_json(plan)
    assert serialized == (
        json.dumps(
            plan,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    assert b"must-not-leak" not in serialized
    assert b"AWS_ACCESS_KEY_ID" not in serialized
    assert b"GITHUB_TOKEN" not in serialized


def test_explicit_apply_builds_then_pushes_and_emits_digest_binding(tmp_path):
    module = _load_script(BUILD_SCRIPT)
    digest = "sha256:" + "9" * 64
    runner = _RecordingRunner(
        [
            b"image built\n",
            f"source-b3471e0969ca: digest: {digest} size: 2048\n".encode(
                "ascii"
            ),
        ]
    )
    plan = module.render_build_plan(
        repository_uri=PRIVATE_REPOSITORY,
        source_commit=SOURCE_COMMIT,
        repository_root=ROOT,
        docker_config=tmp_path / "docker",
    )

    binding = module.execute_build_plan(plan, apply=True, runner=runner)

    assert [call[0] for call in runner.calls] == [
        tuple(plan["commands"]["build"]),
        tuple(plan["commands"]["push"]),
    ]
    assert all(call[1] == plan["environment"] for call in runner.calls)
    assert binding == {
        "schema_version": 1,
        "source_commit": SOURCE_COMMIT,
        "repository_uri": PRIVATE_REPOSITORY,
        "container_image": f"{PRIVATE_REPOSITORY}@{digest}",
        "container_image_digest": digest,
    }
    assert ":" not in binding["container_image"].split("@", 1)[0].rsplit("/", 1)[-1]
    assert module.canonical_json(binding).endswith(b"\n")


def test_build_plan_binds_inputs_and_rejects_post_plan_drift(tmp_path):
    module = _load_script(BUILD_SCRIPT)
    repository = tmp_path / "repository"
    runtime = repository / "containers" / "aws-gpu"
    runtime.mkdir(parents=True)
    for source in (DOCKERFILE, REQUIREMENTS_LOCK, DOCKERIGNORE):
        shutil.copyfile(source, runtime / source.name)
    plan = module.render_build_plan(
        repository_uri=PRIVATE_REPOSITORY,
        source_commit=SOURCE_COMMIT,
        repository_root=repository,
        docker_config=tmp_path / "docker",
    )

    assert plan["inputs"] == {
        "dockerfile_sha256": hashlib.sha256(
            (runtime / "Dockerfile").read_bytes()
        ).hexdigest(),
        "dockerignore_sha256": hashlib.sha256(
            (runtime / "Dockerfile.dockerignore").read_bytes()
        ).hexdigest(),
        "dependency_lock_sha256": hashlib.sha256(
            (runtime / "requirements.lock").read_bytes()
        ).hexdigest(),
    }

    (runtime / "requirements.lock").write_bytes(
        (runtime / "requirements.lock").read_bytes() + b"# drift\n"
    )
    with pytest.raises(module.BuildPlanError, match="changed|drift|input"):
        module.execute_build_plan(
            plan,
            apply=True,
            runner=_ForbiddenRunner(),
        )


@pytest.mark.parametrize(
    "repository",
    [
        "memorysplit/aws-gpu",
        "public.ecr.aws/example/aws-gpu",
        PRIVATE_REPOSITORY + ":latest",
        PRIVATE_REPOSITORY + "@sha256:" + "a" * 64,
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/../escape",
    ],
)
def test_build_renderer_rejects_nonprivate_or_mutable_repository(repository, tmp_path):
    module = _load_script(BUILD_SCRIPT)

    with pytest.raises(module.BuildPlanError):
        module.render_build_plan(
            repository_uri=repository,
            source_commit=SOURCE_COMMIT,
            repository_root=ROOT,
            docker_config=tmp_path / "docker",
        )


def test_runtime_lock_and_sbom_are_deterministic_and_parser_compatible():
    module = _load_script(RUNTIME_LOCK_SCRIPT)
    host_bytes = HOST_CANDIDATE.read_bytes()
    host = json.loads(host_bytes)
    dependency_bytes = REQUIREMENTS_LOCK.read_bytes()
    spec_bytes = _canonical(_runtime_spec())

    first = module.produce_runtime_artifacts(
        spec_bytes=spec_bytes,
        host_candidate_bytes=host_bytes,
        dependency_lock_bytes=dependency_bytes,
    )
    second = module.produce_runtime_artifacts(
        spec_bytes=spec_bytes,
        host_candidate_bytes=host_bytes,
        dependency_lock_bytes=dependency_bytes,
    )

    assert first == second
    assert host_bytes == _canonical(host)
    assert host == {
        "schema_version": 1,
        "ami_id": "ami-0260c4d597dcc8641",
        "ami_owner_id": "898082745236",
        "ami_name": (
            "Deep Learning Base AMI with Single CUDA "
            "Ubuntu 24.04 20260523"
        ),
        "architecture": "x86_64",
        "versions": {
            "cuda": "13.2",
            "nvidia_driver": "595.71.05",
            "kernel": "6.17",
            "efa": "1.47.0",
            "ofi_nccl": "1.18.0",
        },
        "minimum_versions": {
            "cuda": "13.0",
            "nvidia_driver": "580.0",
            "kernel": "6.1",
            "efa": "1.44.0",
            "ofi_nccl": "1.17.1",
        },
    }

    lock = parse_runtime_lock_bytes(first.runtime_lock_bytes)
    assert first.runtime_lock_bytes == _canonical(lock)
    assert set(lock) == {
        "schema_version",
        "source_commit",
        "source_tree",
        "control_bundle_sha256",
        "profile_sha256",
        "ami_id",
        "ami_owner_id",
        "container_image",
        "container_image_digest",
        "versions",
    }
    assert lock["ami_id"] == host["ami_id"]
    assert lock["ami_owner_id"] == host["ami_owner_id"]
    assert lock["versions"] == _runtime_spec()["host_runtime_versions"]

    sbom = json.loads(first.sbom_bytes)
    assert first.sbom_bytes == _canonical(sbom)
    assert sbom["document_type"] == "memorysplit-aws-gpu-sbom-v1"
    assert sbom["runtime_lock_sha256"] == hashlib.sha256(
        first.runtime_lock_bytes
    ).hexdigest()
    assert sbom["dependency_lock_sha256"] == hashlib.sha256(
        dependency_bytes
    ).hexdigest()
    assert sbom["host"]["versions"]["cuda"] == "13.2"
    assert sbom["container"]["versions"]["cuda"] == "13.0"
    assert sbom["host"]["versions"]["pytorch"] == "2.9.0+cu132"
    assert sbom["container"]["versions"]["pytorch"] == "2.9.0+cu130"
    assert sbom["host"]["ami_id"] == "ami-0260c4d597dcc8641"
    assert sbom["container"]["base_image"] == BASE_IMAGE
    assert sbom["container"]["image"] == lock["container_image"]
    packages = sbom["python_packages"]
    assert [item["name"] for item in packages] == sorted(
        _locked_requirements(REQUIREMENTS_LOCK.read_text(encoding="utf-8"))
    )
    assert all(item["allowed_distribution_sha256"] for item in packages)
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", digest)
        for item in packages
        for digest in item["allowed_distribution_sha256"]
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("container", "versions", "python"), "latest"),
        (("container", "versions", "pytorch"), "2.9.0+nightly20260724"),
        (("host_runtime_versions", "docker"), "28+rolling"),
        (("host_runtime_versions", "aws_cli"), "${AWS_PROFILE}"),
    ],
)
def test_runtime_producer_rejects_floating_or_environment_versions(path, value):
    module = _load_script(RUNTIME_LOCK_SCRIPT)
    spec = _runtime_spec()
    target = spec
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value

    with pytest.raises(module.RuntimeArtifactError):
        module.produce_runtime_artifacts(
            spec_bytes=_canonical(spec),
            host_candidate_bytes=HOST_CANDIDATE.read_bytes(),
            dependency_lock_bytes=REQUIREMENTS_LOCK.read_bytes(),
        )


def test_runtime_producer_rejects_open_schemas_mutable_images_and_fake_hashes():
    module = _load_script(RUNTIME_LOCK_SCRIPT)
    cases: list[tuple[dict[str, object], bytes, bytes]] = []

    extra = _runtime_spec()
    extra["unexpected"] = True
    cases.append((extra, HOST_CANDIDATE.read_bytes(), REQUIREMENTS_LOCK.read_bytes()))

    tagged = _runtime_spec()
    tagged["container"]["image_binding"]["container_image"] = (
        PRIVATE_REPOSITORY
        + ":latest@"
        + tagged["container"]["image_binding"]["container_image_digest"]
    )
    cases.append((tagged, HOST_CANDIDATE.read_bytes(), REQUIREMENTS_LOCK.read_bytes()))

    changed_host = json.loads(HOST_CANDIDATE.read_bytes())
    changed_host["versions"]["nvidia_driver"] = "579.99"
    cases.append(
        (
            _runtime_spec(),
            _canonical(changed_host),
            REQUIREMENTS_LOCK.read_bytes(),
        )
    )

    unhashed = b"numpy==2.3.2\n"
    cases.append((_runtime_spec(), HOST_CANDIDATE.read_bytes(), unhashed))

    for spec, host_bytes, dependency_bytes in cases:
        with pytest.raises(module.RuntimeArtifactError):
            module.produce_runtime_artifacts(
                spec_bytes=_canonical(spec),
                host_candidate_bytes=host_bytes,
                dependency_lock_bytes=dependency_bytes,
            )


def test_operator_clis_default_to_offline_dry_run(
    capsys,
    tmp_path,
):
    build_module = _load_script(BUILD_SCRIPT)
    build_status = build_module.main(
        [
            "--repository-uri",
            PRIVATE_REPOSITORY,
            "--source-commit",
            SOURCE_COMMIT,
            "--repository-root",
            str(ROOT),
            "--docker-config",
            str(tmp_path / "docker"),
        ],
        runner=_ForbiddenRunner(),
    )
    build_output = json.loads(capsys.readouterr().out)

    assert build_status == 0
    assert build_output["push_requires_apply"] is True
    assert build_output["repository_uri"] == PRIVATE_REPOSITORY

    runtime_module = _load_script(RUNTIME_LOCK_SCRIPT)
    runtime_input = tmp_path / "runtime-input.json"
    runtime_input.write_bytes(_canonical(_runtime_spec()))
    lock_output = tmp_path / "runtime-lock.json"
    sbom_output = tmp_path / "sbom.json"
    runtime_status = runtime_module.main(
        [
            "--input",
            str(runtime_input),
            "--host-candidate",
            str(HOST_CANDIDATE),
            "--dependency-lock",
            str(REQUIREMENTS_LOCK),
            "--runtime-lock-out",
            str(lock_output),
            "--sbom-out",
            str(sbom_output),
        ]
    )
    runtime_output = json.loads(capsys.readouterr().out)

    assert runtime_status == 0
    assert runtime_output["applied"] is False
    assert not lock_output.exists()
    assert not sbom_output.exists()


def test_operator_tree_contains_no_embedded_secret_or_network_fetcher():
    source_paths = [
        DOCKERFILE,
        REQUIREMENTS_INPUT,
        BUILD_SCRIPT,
        RUNTIME_LOCK_SCRIPT,
        HOST_CANDIDATE,
    ]
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in source_paths
    )

    assert re.search(r"AKIA[0-9A-Z]{16}", combined) is None
    assert "-----BEGIN PRIVATE KEY-----" not in combined
    assert re.search(
        r"(?i)(?:password|secret|token)\s*=\s*['\"][^'\"]+['\"]",
        combined,
    ) is None
    assert "urllib" not in combined
    assert "requests." not in combined
    assert "boto3" not in combined
    assert re.search(r"(?i)\b(curl|wget)\b", DOCKERFILE.read_text()) is None


def test_runtime_apply_rejects_identical_outputs_before_any_write(
    monkeypatch,
    capsys,
    tmp_path,
):
    module = _load_script(RUNTIME_LOCK_SCRIPT)
    runtime_input = tmp_path / "runtime-input.json"
    runtime_input.write_bytes(_canonical(_runtime_spec()))
    output = tmp_path / "artifact.json"
    writes: list[tuple[object, bytes]] = []

    def record_write(path, data):
        writes.append((path, data))

    monkeypatch.setattr(module, "_write_no_replace", record_write)
    status = module.main(
        [
            "--input",
            str(runtime_input),
            "--host-candidate",
            str(HOST_CANDIDATE),
            "--dependency-lock",
            str(REQUIREMENTS_LOCK),
            "--runtime-lock-out",
            str(output),
            "--sbom-out",
            str(output),
            "--apply",
        ]
    )

    assert status == 2
    assert writes == []
    assert capsys.readouterr().out == ""
