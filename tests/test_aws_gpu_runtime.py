from __future__ import annotations

import copy
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
INSPECT_SCRIPT = RUNTIME_ROOT / "inspect_container.py"
QUALIFICATION_WORKER = ROOT / "cluster" / "aws" / "qualification_worker.py"

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
SOURCE_TREE = "f" * 40
CONTAINER_FACTS = {
    "python": "3.12.11",
    "pytorch": "2.9.0+cu130",
    "cuda": "13.0",
    "cudnn": "9.10.2",
    "nccl": "2.28.3",
}
DLC_PYTHON = "/opt/conda/bin/python"


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
    return {
        "schema_version": 1,
        "source": {
            "commit": SOURCE_COMMIT,
            "tree": SOURCE_TREE,
        },
        "control_bundle_sha256": "d" * 64,
        "profile_sha256": "e" * 64,
        "host_runtime_versions": {
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


class _RepositoryReader:
    def __init__(
        self,
        *,
        commit: str = SOURCE_COMMIT,
        tree: str = SOURCE_TREE,
        clean: bool = True,
    ) -> None:
        self.commit = commit
        self.tree = tree
        self.clean = clean
        self.calls: list[Path] = []

    def inspect(self, repository_root):
        self.calls.append(Path(repository_root))
        return {
            "source_commit": self.commit,
            "source_tree": self.tree,
            "clean": self.clean,
            "command_transcript_sha256": {
                "head": "1" * 64,
                "tree": "2" * 64,
                "status": "3" * 64,
            },
        }


def _locked_packages(data: str) -> list[dict[str, object]]:
    packages = []
    pending = ""
    for raw_line in data.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pending = f"{pending} {line}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        requirement, *hashes = pending.split()
        name, version = requirement.split("==", 1)
        normalized_name = name.replace("_", "-").lower()
        files = [
            {
                "path": f"/opt/conda/lib/python3.12/site-packages/{normalized_name}.py",
                "type": "regular",
                "commitment_sha256": "9" * 64,
            },
            {
                "path": f"/opt/conda/lib/python3.12/site-packages/{normalized_name}.dist-info/METADATA",
                "type": "regular",
                "commitment_sha256": "4" * 64,
            },
            {
                "path": f"/opt/conda/lib/python3.12/site-packages/{normalized_name}.dist-info/RECORD",
                "type": "regular",
                "commitment_sha256": "5" * 64,
            },
            {
                "path": f"/opt/conda/lib/python3.12/site-packages/{normalized_name}.dist-info/WHEEL",
                "type": "regular",
                "commitment_sha256": "6" * 64,
            },
        ]
        files.sort(key=lambda item: item["path"])
        packages.append(
            {
                "name": normalized_name,
                "version": version,
                "installer": "pip",
                "provenance": {
                    "kind": "project-wheel",
                    "archive_sha256": hashes[0].removeprefix("--hash=sha256:"),
                },
                "metadata_file_sha256": "4" * 64,
                "record_file_sha256": "5" * 64,
                "wheel_file_sha256": "6" * 64,
                "installed_files": files,
                "installed_files_sha256": hashlib.sha256(
                    _canonical(files)
                ).hexdigest(),
            }
        )
        pending = ""
    packages.append(
        {
            "name": "torch",
            "version": "2.9.0+cu130",
            "installer": "pip",
            "provenance": {
                "kind": "inherited-base-image",
                "base_image_digest": BASE_DIGEST,
            },
            "metadata_file_sha256": "a" * 64,
            "record_file_sha256": "b" * 64,
            "wheel_file_sha256": "c" * 64,
            "installed_files": [
                {
                    "path": "/opt/conda/lib/python3.12/site-packages/torch.dist-info/METADATA",
                    "type": "regular",
                    "commitment_sha256": "a" * 64,
                },
                {
                    "path": "/opt/conda/lib/python3.12/site-packages/torch.dist-info/RECORD",
                    "type": "regular",
                    "commitment_sha256": "b" * 64,
                },
                {
                    "path": "/opt/conda/lib/python3.12/site-packages/torch.dist-info/WHEEL",
                    "type": "regular",
                    "commitment_sha256": "c" * 64,
                },
                {
                    "path": "/opt/conda/lib/python3.12/site-packages/torch.py",
                    "type": "regular",
                    "commitment_sha256": "d" * 64,
                },
            ],
            "installed_files_sha256": hashlib.sha256(
                _canonical(
                    [
                        {
                            "path": "/opt/conda/lib/python3.12/site-packages/torch.dist-info/METADATA",
                            "type": "regular",
                            "commitment_sha256": "a" * 64,
                        },
                        {
                            "path": "/opt/conda/lib/python3.12/site-packages/torch.dist-info/RECORD",
                            "type": "regular",
                            "commitment_sha256": "b" * 64,
                        },
                        {
                            "path": "/opt/conda/lib/python3.12/site-packages/torch.dist-info/WHEEL",
                            "type": "regular",
                            "commitment_sha256": "c" * 64,
                        },
                        {
                            "path": (
                                "/opt/conda/lib/python3.12/site-packages/torch.py"
                            ),
                            "type": "regular",
                            "commitment_sha256": "d" * 64,
                        },
                    ]
                )
            ).hexdigest(),
        }
    )
    return sorted(packages, key=lambda item: item["name"])


def _os_package_inventory() -> dict[str, object]:
    database_files = [
        {
            "path": "/var/lib/dpkg/status",
            "type": "regular",
            "commitment_sha256": "e" * 64,
        }
    ]
    installed_files = [
        {
            "path": "/usr/bin/bash",
            "type": "regular",
            "commitment_sha256": "f" * 64,
        }
    ]
    packages = [
        {
            "name": "bash",
            "version": "5.2.21-2ubuntu4",
            "architecture": "amd64",
            "status": "ii",
            "installed_files": installed_files,
            "installed_files_sha256": hashlib.sha256(
                _canonical(installed_files)
            ).hexdigest(),
        }
    ]
    return {
        "manager": "dpkg",
        "database_root": "/var/lib/dpkg",
        "database_files": database_files,
        "database_tree_sha256": hashlib.sha256(
            _canonical(database_files)
        ).hexdigest(),
        "packages": packages,
        "package_count": len(packages),
    }


def _inspection_artifact() -> dict[str, object]:
    return {
        "schema_version": 2,
        "artifact_type": "memorysplit-container-inspection-v2",
        "os_release": {
            "id": "ubuntu",
            "version_id": "24.04",
            "pretty_name": "Ubuntu 24.04.4 LTS",
        },
        "os_packages": _os_package_inventory(),
        "python": {
            "implementation": "CPython",
            "version": "3.12.11",
            "executable": DLC_PYTHON,
        },
        "container_facts": dict(CONTAINER_FACTS),
        "installed_python_packages": _locked_packages(
            REQUIREMENTS_LOCK.read_text(encoding="utf-8")
        ),
        "inventory_method": "importlib.metadata.distributions",
        "installed_distribution_count": len(
            _locked_packages(REQUIREMENTS_LOCK.read_text(encoding="utf-8"))
        ),
        "project_install_report_sha256": "8" * 64,
    }


def _docker_inspect(
    *,
    repo_digests: list[str],
    entrypoint: list[str] | None = None,
    command: list[str] | None = None,
    image_id: str = "sha256:" + "b" * 64,
) -> bytes:
    return _canonical(
        {
            "Id": image_id,
            "RepoDigests": repo_digests,
            "Config": {
                "Entrypoint": (
                    ["/usr/local/bin/dlc-entrypoint"] if entrypoint is None else entrypoint
                ),
                "Cmd": ["bash"] if command is None else command,
            },
        }
    )


def _apply_outputs(digest: str) -> list[bytes]:
    facts = _canonical(CONTAINER_FACTS)
    inspection = _canonical(_inspection_artifact())
    return [
        b"image built\n",
        b"",
        b"",
        _docker_inspect(repo_digests=[BASE_IMAGE]),
        _docker_inspect(repo_digests=[]),
        facts,
        facts,
        inspection,
        f"source: digest: {digest} size: 2048\n".encode("ascii"),
        b"",
        _docker_inspect(repo_digests=[f"{PRIVATE_REPOSITORY}@{digest}"]),
        facts,
        inspection,
    ]


def _image_binding() -> dict[str, object]:
    digest = "sha256:" + "9" * 64
    inspection = _inspection_artifact()
    entrypoint = {
        "entrypoint": ["/usr/local/bin/dlc-entrypoint"],
        "command": ["bash"],
    }
    return {
        "schema_version": 2,
        "binding_type": "memorysplit-aws-gpu-image-binding-v2",
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "repository_uri": PRIVATE_REPOSITORY,
        "base_image": BASE_IMAGE,
        "base_image_digest": BASE_DIGEST,
        "container_image": f"{PRIVATE_REPOSITORY}@{digest}",
        "container_image_digest": digest,
        "build_inputs": {
            "dockerfile_sha256": "a" * 64,
            "dockerignore_sha256": "b" * 64,
            "dependency_lock_sha256": hashlib.sha256(
                REQUIREMENTS_LOCK.read_bytes()
            ).hexdigest(),
            "inspection_script_sha256": "c" * 64,
            "qualification_worker_sha256": "d" * 64,
        },
        "repository_transcript_sha256": {
            "plan": {"head": "1" * 64, "tree": "2" * 64, "status": "3" * 64},
            "pre_apply": {
                "head": "1" * 64,
                "tree": "2" * 64,
                "status": "3" * 64,
            },
            "final": {"head": "1" * 64, "tree": "2" * 64, "status": "3" * 64},
        },
        "command_transcript_sha256": {
            name: f"{index:x}" * 64
            for index, name in enumerate(
                (
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
                ),
                start=1,
            )
        },
        "container_facts_sha256": hashlib.sha256(
            _canonical(CONTAINER_FACTS)
        ).hexdigest(),
        "entrypoint_sha256": hashlib.sha256(_canonical(entrypoint)).hexdigest(),
        "inherited_entrypoint": entrypoint,
        "inspection_artifact_sha256": hashlib.sha256(
            _canonical(inspection)
        ).hexdigest(),
        "inspection_artifact": inspection,
    }


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
    assert "--force-reinstall" in text
    assert "--ignore-installed" not in text
    assert "--report=/opt/memorysplit/project-install-report.json" in text
    assert "inspect_container.py" in text
    assert "qualification_worker.py" in text
    assert "RUN /opt/conda/bin/python -m pip install" in text
    assert "/opt/conda/bin/python -c" in text
    assert re.search(r"(?m)^RUN python\b", text) is None
    assert re.search(r"(?m)^USER 10001:10001$", text)
    assert "# syntax=" not in text
    assert "ARG APP_UID" not in text
    assert "ARG APP_GID" not in text
    assert not re.search(r"(?im)^\s*ADD\s+https?://", text)
    assert not re.search(r"(?i)\b(curl|wget)\b", text)
    assert "apt-get" not in text
    assert re.search(r"(?im)^\s*(ENTRYPOINT|CMD)\b", text) is None
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
        "!containers/aws-gpu/inspect_container.py",
        "!containers/aws-gpu/requirements.lock",
        "!cluster/",
        "!cluster/aws/",
        "!cluster/aws/qualification_worker.py",
    ]


def test_container_inspection_contract_includes_inherited_and_selected_packages(
    tmp_path,
):
    assert INSPECT_SCRIPT.read_text(encoding="utf-8").startswith(
        "#!/opt/conda/bin/python\n"
    )
    module = _load_script(INSPECT_SCRIPT)

    class Distribution:
        def __init__(self, name, version, texts, files):
            self.metadata = {"Name": name}
            self.version = version
            self.texts = texts
            self.files = tuple(files)

        def read_text(self, name):
            return self.texts.get(name)

        def locate_file(self, name):
            return tmp_path / str(name)

    for relative, content in {
        "numpy.py": b"numpy installed bytes\n",
        "torch.py": b"torch installed bytes\n",
        "numpy-2.5.1.dist-info/METADATA": b"Name: numpy\nVersion: 2.5.1\n",
        "numpy-2.5.1.dist-info/RECORD": b"numpy.py,sha256=x,1\n",
        "numpy-2.5.1.dist-info/WHEEL": b"Wheel-Version: 1.0\n",
        "torch-2.9.0.dist-info/METADATA": (
            b"Name: torch\nVersion: 2.9.0+cu130\n"
        ),
        "torch-2.9.0.dist-info/RECORD": b"torch.py,sha256=y,1\n",
        "torch-2.9.0.dist-info/WHEEL": b"Wheel-Version: 1.0\n",
    }.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    selected_hash = "a" * 64
    install_report = _canonical(
        {
            "install": [
                {
                    "metadata": {"name": "numpy", "version": "2.5.1"},
                    "download_info": {
                        "archive_info": {"hashes": {"sha256": selected_hash}}
                    },
                }
            ]
        }
    )
    artifact = module.build_inspection_artifact(
        os_release_bytes=(
            b'ID=ubuntu\nVERSION_ID="24.04"\n'
            b'PRETTY_NAME="Ubuntu 24.04.4 LTS"\n'
        ),
        install_report_bytes=install_report,
        distributions=(
            Distribution(
                "numpy",
                "2.5.1",
                {
                    "INSTALLER": "pip\n",
                    "METADATA": "Name: numpy\nVersion: 2.5.1\n",
                    "RECORD": "numpy.py,sha256=x,1\n",
                    "WHEEL": "Wheel-Version: 1.0\n",
                },
                [
                    "numpy.py",
                    "numpy-2.5.1.dist-info/METADATA",
                    "numpy-2.5.1.dist-info/RECORD",
                    "numpy-2.5.1.dist-info/WHEEL",
                ],
            ),
            Distribution(
                "torch",
                "2.9.0+cu130",
                {
                    "INSTALLER": "pip\n",
                    "METADATA": "Name: torch\nVersion: 2.9.0+cu130\n",
                    "RECORD": "torch.py,sha256=y,1\n",
                    "WHEEL": "Wheel-Version: 1.0\n",
                },
                [
                    "torch.py",
                    "torch-2.9.0.dist-info/METADATA",
                    "torch-2.9.0.dist-info/RECORD",
                    "torch-2.9.0.dist-info/WHEEL",
                ],
            ),
        ),
        os_package_inventory=_os_package_inventory(),
        base_image_digest=BASE_DIGEST,
        container_facts=CONTAINER_FACTS,
        python_version=CONTAINER_FACTS["python"],
        python_implementation="CPython",
        python_executable=DLC_PYTHON,
    )

    assert artifact["os_release"]["version_id"] == "24.04"
    assert artifact["python"] == {
        "implementation": "CPython",
        "version": CONTAINER_FACTS["python"],
        "executable": DLC_PYTHON,
    }
    packages = {
        package["name"]: package
        for package in artifact["installed_python_packages"]
    }
    assert set(packages) == {"numpy", "torch"}
    assert artifact["os_packages"] == _os_package_inventory()
    assert artifact["os_packages"]["database_files"] == [
        {
            "path": "/var/lib/dpkg/status",
            "type": "regular",
            "commitment_sha256": "e" * 64,
        }
    ]
    assert artifact["inventory_method"] == "importlib.metadata.distributions"
    assert artifact["installed_distribution_count"] == 2
    assert packages["numpy"]["provenance"] == {
        "kind": "project-wheel",
        "archive_sha256": selected_hash,
    }
    assert packages["torch"]["provenance"] == {
        "kind": "inherited-base-image",
        "base_image_digest": BASE_DIGEST,
    }
    assert packages["torch"]["record_file_sha256"] == hashlib.sha256(
        b"torch.py,sha256=y,1\n"
    ).hexdigest()
    assert packages["torch"]["metadata_file_sha256"] == hashlib.sha256(
        b"Name: torch\nVersion: 2.9.0+cu130\n"
    ).hexdigest()
    torch_files = {
        item["path"]: item
        for item in packages["torch"]["installed_files"]
    }
    assert torch_files[str(tmp_path / "torch.py")] == {
        "path": str(tmp_path / "torch.py"),
        "type": "regular",
        "commitment_sha256": hashlib.sha256(
            b"torch installed bytes\n"
        ).hexdigest(),
    }
    assert all(
        value is not None
        for package in packages.values()
        for value in package.values()
    )
    assert module.parse_inspection_artifact_bytes(
        module.canonical_json(artifact)
    ) == artifact
    assert module.canonical_json(artifact).endswith(b"\n")
    assert "--out" not in INSPECT_SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in INSPECT_SCRIPT.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "mutation",
    ["missing-os-package", "missing-python-file", "forged-inherited-base"],
)
def test_container_inspection_rejects_missing_or_forged_inventory(mutation):
    module = _load_script(INSPECT_SCRIPT)
    artifact = _inspection_artifact()
    if mutation == "missing-os-package":
        artifact["os_packages"]["packages"] = []
        artifact["os_packages"]["package_count"] = 0
    elif mutation == "missing-python-file":
        package = artifact["installed_python_packages"][0]
        package["installed_files"] = []
        package["installed_files_sha256"] = hashlib.sha256(
            _canonical([])
        ).hexdigest()
    elif mutation == "forged-inherited-base":
        torch = next(
            package
            for package in artifact["installed_python_packages"]
            if package["name"] == "torch"
        )
        torch["provenance"]["base_image_digest"] = "sha256:" + "0" * 64
    else:
        raise AssertionError(mutation)

    with pytest.raises(module.InspectionError, match="package|file|base|inventory"):
        module.parse_inspection_artifact_bytes(_canonical(artifact))


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
        repository_reader=_RepositoryReader(),
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
    runner = _RecordingRunner(_apply_outputs(digest))
    plan = module.render_build_plan(
        repository_uri=PRIVATE_REPOSITORY,
        source_commit=SOURCE_COMMIT,
        repository_root=ROOT,
        docker_config=tmp_path / "docker",
        repository_reader=_RepositoryReader(),
    )

    binding = module.execute_build_plan(
        plan,
        apply=True,
        runner=runner,
        repository_reader=_RepositoryReader(),
    )

    assert [call[0] for call in runner.calls] == [
        tuple(plan["commands"][name])
        for name in (
            "build",
            "python_base",
            "python_local",
            "inspect_base",
            "inspect_local",
            "facts_base",
            "facts_local",
        )
    ] + [
        tuple(
            module.container_inspection_argv(
                plan["staging_tag"],
                container_facts_json=_canonical(CONTAINER_FACTS)
                .decode("ascii")
                .rstrip("\n"),
            )
        ),
        tuple(plan["commands"]["push"]),
        tuple(module.container_python_exists_argv(f"{PRIVATE_REPOSITORY}@{digest}")),
        tuple(module.final_image_inspect_argv(f"{PRIVATE_REPOSITORY}@{digest}")),
        tuple(module.container_facts_argv(f"{PRIVATE_REPOSITORY}@{digest}")),
        tuple(
            module.container_inspection_argv(
                f"{PRIVATE_REPOSITORY}@{digest}",
                container_facts_json=_canonical(CONTAINER_FACTS)
                .decode("ascii")
                .rstrip("\n"),
            )
        ),
    ]
    assert all(call[1] == plan["environment"] for call in runner.calls)
    assert binding["schema_version"] == 2
    assert binding["binding_type"] == "memorysplit-aws-gpu-image-binding-v2"
    assert binding["source_commit"] == SOURCE_COMMIT
    assert binding["source_tree"] == SOURCE_TREE
    assert binding["repository_uri"] == PRIVATE_REPOSITORY
    assert binding["base_image"] == BASE_IMAGE
    assert binding["base_image_digest"] == BASE_DIGEST
    assert binding["container_image"] == f"{PRIVATE_REPOSITORY}@{digest}"
    assert binding["container_image_digest"] == digest
    assert binding["build_inputs"] == plan["inputs"]
    assert set(binding["command_transcript_sha256"]) == {
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
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", value)
        for value in binding["command_transcript_sha256"].values()
    )
    assert binding["inspection_artifact"] == _inspection_artifact()
    assert binding["inherited_entrypoint"] == {
        "entrypoint": ["/usr/local/bin/dlc-entrypoint"],
        "command": ["bash"],
    }
    assert ":" not in binding["container_image"].split("@", 1)[0].rsplit("/", 1)[-1]
    assert module.canonical_json(binding).endswith(b"\n")


def test_inventory_probe_is_restricted_root_and_framework_probe_is_nonroot():
    build_module = _load_script(BUILD_SCRIPT)
    attestation_module = _load_script(
        ROOT / "cluster" / "aws" / "p5" / "attest_environment.py"
    )
    image = f"{PRIVATE_REPOSITORY}@{'sha256:' + '9' * 64}"

    inventory_argv = build_module.container_inspection_argv(image)
    assert inventory_argv == (
        "/usr/bin/docker",
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
        "__MEASURED_CONTAINER_FACTS__",
    )
    assert "--gpus" not in inventory_argv
    assert "--mount" not in inventory_argv
    assert "--volume" not in inventory_argv
    assert "-v" not in inventory_argv
    assert [item for item in inventory_argv if item == "--cap-add"] == ["--cap-add"]

    for facts_argv in (
        build_module.container_facts_argv(image),
        attestation_module.container_facts_argv(image),
    ):
        assert facts_argv[facts_argv.index("--user") + 1] == "10001:10001"
        assert facts_argv[facts_argv.index("--network") + 1] == "none"
        assert facts_argv[facts_argv.index("--entrypoint") + 1] == DLC_PYTHON
        assert "--gpus" in facts_argv and "all" in facts_argv
        assert "--cap-add" not in facts_argv
        assert "--mount" not in facts_argv
    for exists_argv in (
        build_module.container_python_exists_argv(image),
        attestation_module.container_python_exists_argv(image),
    ):
        assert exists_argv[exists_argv.index("--user") + 1] == "10001:10001"


@pytest.mark.parametrize(
    "unsafe_argument",
    ["--network=host", "--cap-add=SYS_ADMIN", "--mount", "--volume"],
)
def test_build_plan_rejects_extra_inventory_privilege(unsafe_argument, tmp_path):
    module = _load_script(BUILD_SCRIPT)
    plan = module.render_build_plan(
        repository_uri=PRIVATE_REPOSITORY,
        source_commit=SOURCE_COMMIT,
        repository_root=ROOT,
        docker_config=tmp_path / "docker",
        repository_reader=_RepositoryReader(),
    )
    plan["commands"]["inspection_local"].insert(2, unsafe_argument)

    with pytest.raises(module.BuildPlanError, match="plan|command|input|changed"):
        module.execute_build_plan(plan, apply=False, runner=_ForbiddenRunner())


@pytest.mark.parametrize(
    "reader",
    [
        _RepositoryReader(clean=False),
        _RepositoryReader(commit="0" * 40),
    ],
)
def test_build_plan_requires_actual_clean_expected_repository(reader, tmp_path):
    module = _load_script(BUILD_SCRIPT)

    with pytest.raises(module.BuildPlanError, match="clean|commit|source"):
        module.render_build_plan(
            repository_uri=PRIVATE_REPOSITORY,
            source_commit=SOURCE_COMMIT,
            repository_root=ROOT,
            docker_config=tmp_path / "docker",
            repository_reader=reader,
        )


def test_apply_rechecks_repository_before_first_docker_command(tmp_path):
    module = _load_script(BUILD_SCRIPT)
    plan = module.render_build_plan(
        repository_uri=PRIVATE_REPOSITORY,
        source_commit=SOURCE_COMMIT,
        repository_root=ROOT,
        docker_config=tmp_path / "docker",
        repository_reader=_RepositoryReader(),
    )

    with pytest.raises(module.BuildPlanError, match="clean|source|repository"):
        module.execute_build_plan(
            plan,
            apply=True,
            runner=_ForbiddenRunner(),
            repository_reader=_RepositoryReader(clean=False),
        )


def test_apply_holds_and_rehashes_build_inputs_before_push(tmp_path):
    module = _load_script(BUILD_SCRIPT)
    repository = tmp_path / "repository"
    runtime = repository / "containers" / "aws-gpu"
    runtime.mkdir(parents=True)
    for source in (DOCKERFILE, REQUIREMENTS_LOCK, DOCKERIGNORE, INSPECT_SCRIPT):
        shutil.copyfile(source, runtime / source.name)
    worker = repository / "cluster" / "aws" / "qualification_worker.py"
    worker.parent.mkdir(parents=True)
    shutil.copyfile(QUALIFICATION_WORKER, worker)
    plan = module.render_build_plan(
        repository_uri=PRIVATE_REPOSITORY,
        source_commit=SOURCE_COMMIT,
        repository_root=repository,
        docker_config=tmp_path / "docker",
        repository_reader=_RepositoryReader(),
    )

    class MutatingRunner(_RecordingRunner):
        def run(self, argv, *, environment):
            output = super().run(argv, environment=environment)
            if len(self.calls) == 1:
                with (runtime / "requirements.lock").open("ab") as stream:
                    stream.write(b"# post-build drift\n")
            return output

    runner = MutatingRunner(_apply_outputs("sha256:" + "9" * 64))
    with pytest.raises(module.BuildPlanError, match="changed|drift|input"):
        module.execute_build_plan(
            plan,
            apply=True,
            runner=runner,
            repository_reader=_RepositoryReader(),
        )

    assert len(runner.calls) == 1
    assert "push" not in runner.calls[0][0]


@pytest.mark.parametrize(
    "mutation",
    [
        "entrypoint",
        "local-facts",
        "final-facts",
        "final-image-id",
        "system-python-inspection",
    ],
)
def test_apply_rejects_unmeasured_or_changed_container_runtime(
    tmp_path,
    mutation,
):
    module = _load_script(BUILD_SCRIPT)
    digest = "sha256:" + "9" * 64
    outputs = _apply_outputs(digest)
    if mutation == "entrypoint":
        outputs[4] = _docker_inspect(
            repo_digests=[],
            entrypoint=["/unreviewed-entrypoint"],
        )
    elif mutation == "local-facts":
        facts = dict(CONTAINER_FACTS)
        facts["cuda"] = "12.9"
        outputs[6] = _canonical(facts)
    elif mutation == "final-facts":
        facts = dict(CONTAINER_FACTS)
        facts["pytorch"] = "2.9.1+cu130"
        outputs[11] = _canonical(facts)
    elif mutation == "final-image-id":
        outputs[10] = _docker_inspect(
            repo_digests=[f"{PRIVATE_REPOSITORY}@{digest}"],
            image_id="sha256:" + "c" * 64,
        )
    elif mutation == "system-python-inspection":
        inspection = _inspection_artifact()
        inspection["python"]["executable"] = "/usr/bin/python3"
        outputs[7] = _canonical(inspection)
    else:
        raise AssertionError(mutation)
    plan = module.render_build_plan(
        repository_uri=PRIVATE_REPOSITORY,
        source_commit=SOURCE_COMMIT,
        repository_root=ROOT,
        docker_config=tmp_path / "docker",
        repository_reader=_RepositoryReader(),
    )

    with pytest.raises(
        module.BuildPlanError,
        match="entrypoint|container|framework|fact|runtime",
    ):
        module.execute_build_plan(
            plan,
            apply=True,
            runner=_RecordingRunner(outputs),
            repository_reader=_RepositoryReader(),
        )


def test_build_plan_binds_inputs_and_rejects_post_plan_drift(tmp_path):
    module = _load_script(BUILD_SCRIPT)
    repository = tmp_path / "repository"
    runtime = repository / "containers" / "aws-gpu"
    runtime.mkdir(parents=True)
    for source in (DOCKERFILE, REQUIREMENTS_LOCK, DOCKERIGNORE, INSPECT_SCRIPT):
        shutil.copyfile(source, runtime / source.name)
    worker = repository / "cluster" / "aws" / "qualification_worker.py"
    worker.parent.mkdir(parents=True)
    shutil.copyfile(QUALIFICATION_WORKER, worker)
    plan = module.render_build_plan(
        repository_uri=PRIVATE_REPOSITORY,
        source_commit=SOURCE_COMMIT,
        repository_root=repository,
        docker_config=tmp_path / "docker",
        repository_reader=_RepositoryReader(),
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
        "inspection_script_sha256": hashlib.sha256(
            (runtime / "inspect_container.py").read_bytes()
        ).hexdigest(),
        "qualification_worker_sha256": hashlib.sha256(
            worker.read_bytes()
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
            repository_reader=_RepositoryReader(),
        )


def test_runtime_lock_and_sbom_are_deterministic_and_parser_compatible():
    module = _load_script(RUNTIME_LOCK_SCRIPT)
    host_bytes = HOST_CANDIDATE.read_bytes()
    host = json.loads(host_bytes)
    dependency_bytes = REQUIREMENTS_LOCK.read_bytes()
    spec_bytes = _canonical(_runtime_spec())
    image_binding_bytes = _canonical(_image_binding())

    first = module.produce_runtime_artifacts(
        spec_bytes=spec_bytes,
        host_candidate_bytes=host_bytes,
        dependency_lock_bytes=dependency_bytes,
        image_binding_bytes=image_binding_bytes,
    )
    second = module.produce_runtime_artifacts(
        spec_bytes=spec_bytes,
        host_candidate_bytes=host_bytes,
        dependency_lock_bytes=dependency_bytes,
        image_binding_bytes=image_binding_bytes,
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
            "nvlsm": "595.71.05",
        },
        "minimum_versions": {
            "cuda": "13.0",
            "nvidia_driver": "580.0",
            "kernel": "6.1",
            "efa": "1.44.0",
            "ofi_nccl": "1.17.1",
            "nvlsm": "580.0",
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
    assert lock["versions"] == {
        **CONTAINER_FACTS,
        **_runtime_spec()["host_runtime_versions"],
    }

    sbom = json.loads(first.sbom_bytes)
    assert first.sbom_bytes == _canonical(sbom)
    assert module.parse_runtime_sbom_bytes(first.sbom_bytes) == sbom
    assert sbom["document_type"] == "memorysplit-aws-gpu-sbom-v2"
    assert sbom["runtime_lock_sha256"] == hashlib.sha256(
        first.runtime_lock_bytes
    ).hexdigest()
    assert sbom["project_dependency_lock"]["sha256"] == hashlib.sha256(
        dependency_bytes
    ).hexdigest()
    assert sbom["host"]["versions"]["cuda"] == "13.2"
    assert sbom["container"]["versions"]["cuda"] == "13.0"
    assert sbom["host"]["ami_id"] == "ami-0260c4d597dcc8641"
    assert sbom["container"]["base_image"] == BASE_IMAGE
    assert sbom["container"]["base_image_digest"] == BASE_DIGEST
    assert sbom["container"]["image"] == lock["container_image"]
    assert sbom["container"]["os_release"]["id"] == "ubuntu"
    assert sbom["container"]["os_packages"] == _os_package_inventory()
    assert sbom["container"]["python"]["version"] == CONTAINER_FACTS["python"]
    assert sbom["container"]["python"]["executable"] == DLC_PYTHON
    assert sbom["container"]["inherited_entrypoint"] == {
        "entrypoint": ["/usr/local/bin/dlc-entrypoint"],
        "command": ["bash"],
    }
    packages = sbom["container"]["installed_python_packages"]
    assert sbom["container"]["inventory_method"] == (
        "importlib.metadata.distributions"
    )
    assert sbom["container"]["installed_distribution_count"] == len(packages)
    assert "torch" in {item["name"] for item in packages}
    assert set(_locked_requirements(REQUIREMENTS_LOCK.read_text())) < {
        item["name"] for item in packages
    }
    assert all(
        item["provenance"]["kind"]
        in {"project-wheel", "inherited-base-image"}
        for item in packages
    )
    assert all(item["installed_files"] for item in packages)
    assert sbom["container"]["inspection_artifact_sha256"] == hashlib.sha256(
        _canonical(_inspection_artifact())
    ).hexdigest()
    assert "python_packages" not in sbom
    open_sbom = copy.deepcopy(sbom)
    open_sbom["unexpected"] = True
    with pytest.raises(module.RuntimeArtifactError, match="schema|field|closed"):
        module.parse_runtime_sbom_bytes(_canonical(open_sbom))
    wrong_host_sbom = copy.deepcopy(sbom)
    wrong_host_sbom["host"]["ami_id"] = "ami-0b39828e6910b0bb8"
    with pytest.raises(module.RuntimeArtifactError, match="host|AMI|reviewed"):
        module.parse_runtime_sbom_bytes(_canonical(wrong_host_sbom))
    wrong_archive_sbom = copy.deepcopy(sbom)
    project_package = next(
        package
        for package in wrong_archive_sbom["container"][
            "installed_python_packages"
        ]
        if package["provenance"]["kind"] == "project-wheel"
    )
    project_package["provenance"]["archive_sha256"] = "0" * 64
    with pytest.raises(module.RuntimeArtifactError, match="archive|lock|project"):
        module.parse_runtime_sbom_bytes(_canonical(wrong_archive_sbom))


@pytest.mark.parametrize(
    ("path", "value"),
    [
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
            image_binding_bytes=_canonical(_image_binding()),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("python", "latest"),
        ("pytorch", "2.9.0+nightly20260724"),
    ],
)
def test_runtime_producer_rejects_unmeasured_container_version_claims(field, value):
    module = _load_script(RUNTIME_LOCK_SCRIPT)
    binding = _image_binding()
    binding["inspection_artifact"]["container_facts"][field] = value
    binding["inspection_artifact_sha256"] = hashlib.sha256(
        _canonical(binding["inspection_artifact"])
    ).hexdigest()

    with pytest.raises(module.RuntimeArtifactError):
        module.produce_runtime_artifacts(
            spec_bytes=_canonical(_runtime_spec()),
            host_candidate_bytes=HOST_CANDIDATE.read_bytes(),
            dependency_lock_bytes=REQUIREMENTS_LOCK.read_bytes(),
            image_binding_bytes=_canonical(binding),
        )


def test_runtime_producer_rejects_open_schemas_mutable_images_and_fake_hashes():
    module = _load_script(RUNTIME_LOCK_SCRIPT)
    cases: list[tuple[dict[str, object], bytes, bytes, dict[str, object]]] = []

    extra = _runtime_spec()
    extra["unexpected"] = True
    cases.append(
        (
            extra,
            HOST_CANDIDATE.read_bytes(),
            REQUIREMENTS_LOCK.read_bytes(),
            _image_binding(),
        )
    )

    tagged = _image_binding()
    tagged["container_image"] = (
        PRIVATE_REPOSITORY
        + ":latest@"
        + tagged["container_image_digest"]
    )
    cases.append(
        (
            _runtime_spec(),
            HOST_CANDIDATE.read_bytes(),
            REQUIREMENTS_LOCK.read_bytes(),
            tagged,
        )
    )

    changed_host = json.loads(HOST_CANDIDATE.read_bytes())
    changed_host["versions"]["nvidia_driver"] = "579.99"
    cases.append(
        (
            _runtime_spec(),
            _canonical(changed_host),
            REQUIREMENTS_LOCK.read_bytes(),
            _image_binding(),
        )
    )

    unhashed = b"numpy==2.3.2\n"
    cases.append(
        (
            _runtime_spec(),
            HOST_CANDIDATE.read_bytes(),
            unhashed,
            _image_binding(),
        )
    )

    dependency_only = _image_binding()
    dependency_only["inspection_artifact"]["installed_python_packages"] = [
        package
        for package in dependency_only["inspection_artifact"][
            "installed_python_packages"
        ]
        if package["name"] != "torch"
    ]
    dependency_only["inspection_artifact_sha256"] = hashlib.sha256(
        _canonical(dependency_only["inspection_artifact"])
    ).hexdigest()
    cases.append(
        (
            _runtime_spec(),
            HOST_CANDIDATE.read_bytes(),
            REQUIREMENTS_LOCK.read_bytes(),
            dependency_only,
        )
    )

    for spec, host_bytes, dependency_bytes, binding in cases:
        with pytest.raises(module.RuntimeArtifactError):
            module.produce_runtime_artifacts(
                spec_bytes=_canonical(spec),
                host_candidate_bytes=host_bytes,
                dependency_lock_bytes=dependency_bytes,
                image_binding_bytes=_canonical(binding),
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
        repository_reader=_RepositoryReader(),
    )
    build_output = json.loads(capsys.readouterr().out)

    assert build_status == 0
    assert build_output["push_requires_apply"] is True
    assert build_output["repository_uri"] == PRIVATE_REPOSITORY

    runtime_module = _load_script(RUNTIME_LOCK_SCRIPT)
    runtime_input = tmp_path / "runtime-input.json"
    runtime_input.write_bytes(_canonical(_runtime_spec()))
    image_binding = tmp_path / "image-binding.json"
    image_binding.write_bytes(_canonical(_image_binding()))
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
            "--image-binding",
            str(image_binding),
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
        INSPECT_SCRIPT,
        QUALIFICATION_WORKER,
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
    image_binding = tmp_path / "image-binding.json"
    image_binding.write_bytes(_canonical(_image_binding()))
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
            "--image-binding",
            str(image_binding),
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
