from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "cluster" / "aws" / "p5" / "attest_environment.py"
PROFILE = ROOT / "cluster" / "profiles" / "aws-p5.48xlarge-v3.json"
IMAGE_DIGEST = "sha256:" + "a" * 64
IMAGE = (
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/memorysplit"
    f"@{IMAGE_DIGEST}"
)
VERSIONS = {
    "python": "3.12.4",
    "pytorch": "2.7.1+cu128",
    "cuda": "12.8",
    "cudnn": "9.7.1",
    "nccl": "2.26.2",
    "nvidia_driver": "570.133.20",
    "fabric_manager": "570.133.20",
    "docker": "28.0.4",
    "nvidia_container_runtime": "1.17.8",
    "aws_cli": "2.27.49",
}
IDENTITY = {
    "accountId": "123456789012",
    "architecture": "x86_64",
    "imageId": "ami-0123456789abcdef0",
    "instanceId": "i-0123456789abcdef0",
    "privateIp": "10.23.45.67",
    "region": "us-east-1",
}
PKCS7 = base64.b64encode(b"synthetic-signed-identity").decode("ascii")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def _load_module():
    assert SCRIPT.is_file(), "AWS environment attestation producer is missing"
    name = f"attest_environment_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _runtime_lock(profile_sha256: str, control_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_commit": "b" * 40,
        "source_tree": "c" * 40,
        "control_bundle_sha256": control_sha256,
        "profile_sha256": profile_sha256,
        "ami_id": IDENTITY["imageId"],
        "ami_owner_id": "210987654321",
        "container_image": IMAGE,
        "container_image_digest": IMAGE_DIGEST,
        "versions": dict(VERSIONS),
    }


class _FakeImds:
    def __init__(
        self,
        *,
        identity: dict[str, object] | None = None,
        identity_data: bytes | None = None,
        pkcs7_data: bytes | None = None,
        token: str = "imds-v2-token",
    ) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.identity_data = (
            identity_data
            if identity_data is not None
            else _canonical(IDENTITY if identity is None else identity)
        )
        self.pkcs7_data = (
            pkcs7_data
            if pkcs7_data is not None
            else (PKCS7 + "\n").encode("ascii")
        )
        self.token_value = token

    def token(self) -> str:
        self.calls.append(("token", None))
        return self.token_value

    def read(self, path: str, *, token: str) -> bytes:
        self.calls.append((path, token))
        assert token == "imds-v2-token"
        if path.endswith("/document"):
            return self.identity_data
        if path.endswith("/pkcs7"):
            return self.pkcs7_data
        raise AssertionError(f"unexpected IMDS path: {path}")


class _FakeCommands:
    def __init__(
        self,
        module,
        *,
        versions: dict[str, str] | None = None,
        repo_digests: list[str] | None = None,
        fail_field: str | None = None,
    ) -> None:
        self.module = module
        self.calls: list[tuple[tuple[str, ...], dict[str, str]]] = []
        self.versions = dict(VERSIONS if versions is None else versions)
        self.repo_digests = [IMAGE] if repo_digests is None else repo_digests
        self.fail_field = fail_field

    def run(self, argv, *, environment) -> bytes:
        rendered = tuple(argv)
        self.calls.append((rendered, dict(environment)))
        for field, expected_argv in self.module.VERSION_COMMANDS.items():
            if rendered == tuple(expected_argv):
                if field == self.fail_field:
                    raise OSError("synthetic command failure must not leak")
                return (self.versions[field] + "\n").encode("ascii")
        if rendered == tuple(self.module.container_inspect_argv(IMAGE)):
            if self.fail_field == "container":
                raise OSError("synthetic command failure must not leak")
            return json.dumps(self.repo_digests).encode("ascii") + b"\n"
        raise AssertionError(f"unexpected command: {rendered!r}")


def _case(tmp_path: Path, module, *, name: str = "case") -> dict[str, object]:
    root = tmp_path / name
    root.mkdir()
    control = root / "control-bundle.zip"
    control.write_bytes(b"synthetic immutable control bundle")
    lock = root / "runtime-lock.json"
    lock_value = _runtime_lock(
        hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
        hashlib.sha256(control.read_bytes()).hexdigest(),
    )
    lock.write_bytes(_canonical(lock_value))
    boot = root / "boot_id"
    boot.write_text("12345678-1234-4abc-8def-1234567890ab\n", encoding="ascii")
    return {
        "module": module,
        "profile": PROFILE,
        "control": control,
        "lock": lock,
        "lock_value": lock_value,
        "boot": boot,
        "output": root / "receipt.json",
    }


def _attest(
    case: dict[str, object],
    *,
    apply: bool = False,
    imds=None,
    commands=None,
):
    module = case["module"]
    return module.attest_environment(
        profile_path=case["profile"],
        runtime_lock_path=case["lock"],
        control_bundle_path=case["control"],
        output_path=case["output"],
        apply=apply,
        imds_reader=imds or _FakeImds(),
        command_reader=commands or _FakeCommands(module),
        boot_id_path=case["boot"],
    )


def test_producer_emits_exact_canonical_v2_receipt_and_dry_run_writes_nothing(
    tmp_path,
):
    module = _load_module()
    case = _case(tmp_path, module)
    lock = case["lock"]
    lock_value = case["lock_value"]
    output = case["output"]
    imds = _FakeImds()
    commands = _FakeCommands(module)

    receipt = _attest(case, imds=imds, commands=commands)

    expected = {
        "schema_version": 2,
        "receipt_type": "memorysplit-aws-environment-v2",
        "provider": "aws-p5.48xlarge",
        "profile_sha256": lock_value["profile_sha256"],
        "runtime_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "control_bundle_sha256": lock_value["control_bundle_sha256"],
        "source_commit": lock_value["source_commit"],
        "source_tree": lock_value["source_tree"],
        "container_image": IMAGE,
        "container_image_digest": IMAGE_DIGEST,
        "aws_instance_identity_document": IDENTITY,
        "aws_instance_identity_pkcs7": PKCS7,
        "account_id": IDENTITY["accountId"],
        "instance_id": IDENTITY["instanceId"],
        "region": IDENTITY["region"],
        "ami_id": IDENTITY["imageId"],
        "boot_id": "12345678-1234-4abc-8def-1234567890ab",
        "runtime_facts": VERSIONS,
    }
    assert receipt == expected
    assert module.canonical_receipt(receipt) == _canonical(expected)
    assert not output.exists()
    assert imds.calls == [
        ("token", None),
        (
            "/latest/dynamic/instance-identity/document",
            "imds-v2-token",
        ),
        (
            "/latest/dynamic/instance-identity/pkcs7",
            "imds-v2-token",
        ),
    ]
    assert len(commands.calls) == len(VERSIONS) + 1
    assert module.VERSION_COMMANDS["fabric_manager"][0] == (
        "/usr/bin/nv-fabricmanager"
    )
    assert all(argv and argv[0].startswith("/") for argv, _ in commands.calls)
    assert all(
        environment == module.MINIMAL_COMMAND_ENVIRONMENT
        for _, environment in commands.calls
    )


def test_python_probes_use_exact_isolation_and_trusted_working_directory(
    monkeypatch,
):
    module = _load_module()
    for field in ("python", "pytorch", "cuda", "cudnn", "nccl"):
        argv = module.VERSION_COMMANDS[field]
        assert argv[0] == "/usr/bin/python3"
        assert argv[1:4] == ("-I", "-P", "-c")

    trusted = Path(module.TRUSTED_COMMAND_WORKING_DIRECTORY)
    details = trusted.stat(follow_symlinks=False)
    assert trusted.is_absolute()
    assert stat.S_ISDIR(details.st_mode)
    assert stat.S_IMODE(details.st_mode) & 0o022 == 0

    observed = {}

    def run(argv, **kwargs):
        observed["argv"] = list(argv)
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=b"3.12.4\n", stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", run)
    output = module.SubprocessCommandReader().run(
        module.VERSION_COMMANDS["python"],
        environment=module.MINIMAL_COMMAND_ENVIRONMENT,
    )

    assert output == b"3.12.4\n"
    assert observed["kwargs"]["cwd"] == str(trusted)
    assert observed["kwargs"]["env"] == module.MINIMAL_COMMAND_ENVIRONMENT
    assert observed["kwargs"]["shell"] is False


def test_writable_platform_and_torch_modules_cannot_spoof_runtime_facts(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    attack = tmp_path / "attacker-cwd"
    attack.mkdir()
    (attack / "platform.py").write_text(
        "def python_version():\n"
        f"    return {VERSIONS['python']!r}\n",
        encoding="utf-8",
    )
    (attack / "torch.py").write_text(
        f"__version__ = {VERSIONS['pytorch']!r}\n"
        "class version:\n"
        f"    cuda = {VERSIONS['cuda']!r}\n"
        "class _Cudnn:\n"
        "    @staticmethod\n"
        f"    def version(): return {VERSIONS['cudnn']!r}\n"
        "class _Backends:\n"
        "    cudnn = _Cudnn()\n"
        "backends = _Backends()\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(attack)
    reader = module.SubprocessCommandReader()

    for field in ("python", "pytorch"):
        try:
            output = reader.run(
                module.VERSION_COMMANDS[field],
                environment=module.MINIMAL_COMMAND_ENVIRONMENT,
            )
        except module.AttestationError:
            continue
        assert output.decode("utf-8").strip() != VERSIONS[field]


def test_producer_parses_the_complete_v3_profile_not_only_identity_fields(
    tmp_path,
):
    module = _load_module()
    case = _case(tmp_path, module)
    profile_value = json.loads(PROFILE.read_text(encoding="utf-8"))
    profile_value["gpu"]["allocated"] = 1
    profile = tmp_path / "mutated-profile.json"
    profile.write_bytes(_canonical(profile_value))
    case["profile"] = profile
    lock_value = copy.deepcopy(case["lock_value"])
    lock_value["profile_sha256"] = hashlib.sha256(profile.read_bytes()).hexdigest()
    case["lock"].write_bytes(_canonical(lock_value))

    with pytest.raises(module.AttestationError, match="profile|P5|v3"):
        _attest(case)


def test_producer_accepts_only_closed_canonical_runtime_lock_schema(tmp_path):
    module = _load_module()
    mutations = [
        ("schema-float", lambda value: value.update(schema_version=1.0)),
        ("unknown-field", lambda value: value.update(unexpected=True)),
        ("missing-field", lambda value: value.pop("source_tree")),
        ("uppercase-commit", lambda value: value.update(source_commit="B" * 40)),
        ("bad-tree", lambda value: value.update(source_tree="c" * 39)),
        (
            "bad-control-hash",
            lambda value: value.update(control_bundle_sha256="D" * 64),
        ),
        ("mutable-ami", lambda value: value.update(ami_id="ami-latest")),
        ("bad-owner", lambda value: value.update(ami_owner_id="1234")),
        (
            "mutable-image",
            lambda value: value.update(container_image="registry.example/x:latest"),
        ),
        (
            "tagged-pinned-image",
            lambda value: value.update(
                container_image=(
                    "registry.example/x:latest@" + str(IMAGE_DIGEST)
                )
            ),
        ),
        (
            "digest-mismatch",
            lambda value: value.update(
                container_image_digest="sha256:" + "d" * 64
            ),
        ),
        ("missing-version", lambda value: value["versions"].pop("nccl")),
        (
            "unknown-version",
            lambda value: value["versions"].update(extra="1.0"),
        ),
        ("empty-version", lambda value: value["versions"].update(cuda="")),
        ("floating-version", lambda value: value["versions"].update(cuda="latest")),
        (
            "embedded-floating-version",
            lambda value: value["versions"].update(cuda="12+latest"),
        ),
        (
            "snapshot-version",
            lambda value: value["versions"].update(cuda="12+snapshot"),
        ),
        (
            "rolling-version",
            lambda value: value["versions"].update(cuda="12+rolling"),
        ),
    ]
    rejected: list[str] = []
    for index, (name, mutate) in enumerate(mutations):
        case = _case(tmp_path, module, name=f"lock-{index}")
        value = copy.deepcopy(case["lock_value"])
        mutate(value)
        case["lock"].write_bytes(_canonical(value))
        try:
            _attest(case)
        except module.AttestationError:
            rejected.append(name)
    assert rejected == [name for name, _ in mutations]

    for index, floating in enumerate(
        ("12+latest", "12+snapshot", "12+rolling")
    ):
        direct = _case(
            tmp_path,
            module,
            name=f"direct-floating-lock-{index}",
        )
        value = copy.deepcopy(direct["lock_value"])
        value["versions"]["cuda"] = floating
        direct["lock"].write_bytes(_canonical(value))
        with pytest.raises(module.AttestationError, match="non-floating"):
            module.parse_runtime_lock_bytes(direct["lock"].read_bytes())

    duplicate = _case(tmp_path, module, name="duplicate-lock")
    duplicate["lock"].write_bytes(
        duplicate["lock"].read_bytes().replace(
            b'{"ami_id":',
            b'{"ami_id":"ami-0123456789abcdef0","ami_id":',
            1,
        )
    )
    with pytest.raises(module.AttestationError, match="repeat|duplicate"):
        _attest(duplicate)

    noncanonical = _case(tmp_path, module, name="noncanonical-lock")
    noncanonical["lock"].write_text(
        json.dumps(noncanonical["lock_value"], indent=2) + "\n",
        encoding="ascii",
    )
    with pytest.raises(module.AttestationError, match="canonical"):
        _attest(noncanonical)


@pytest.mark.parametrize(
    "floating",
    [
        "1.0-latest2026",
        "1.0-La-TeSt.2026",
        "1.0-MAIN7",
        "1.0-master_8",
        "1.0-he.ad9",
        "1.0-dev2026",
        "1.0-NIGHTLY.20260723",
        "1.0-snap_shot9",
        "1.0-UnKnown2",
    ],
)
def test_runtime_versions_reject_embedded_floating_markers(floating):
    module = _load_module()

    with pytest.raises(module.AttestationError, match="non-floating"):
        module._fixed_version(floating, label="test version")


@pytest.mark.parametrize(
    "concrete",
    [
        "3.12.4",
        "2.7.1+cu128",
        "580.159.03",
        "2.0.0-rc1",
        "1.2.3+cpu",
    ],
)
def test_runtime_versions_preserve_concrete_tool_outputs(concrete):
    module = _load_module()

    assert module._fixed_version(concrete, label="test version") == concrete


@pytest.mark.parametrize(
    "image",
    [
        "registry.example/@sha256:" + "a" * 64,
        "registry.example/repo//@sha256:" + "a" * 64,
        "registry.example/repo/./part@sha256:" + "a" * 64,
        "registry.example/repo/../part@sha256:" + "a" * 64,
        "registry.example/repo:latest@sha256:" + "a" * 64,
        "registry.example/repo@@sha256:" + "a" * 64,
        "registry.example/repo@sha256:" + "A" * 64,
        "registry.example/re po@sha256:" + "a" * 64,
        "-registry.example/repo@sha256:" + "a" * 64,
    ],
)
def test_runtime_lock_rejects_malformed_oci_image_references(tmp_path, image):
    module = _load_module()
    case = _case(tmp_path, module)
    value = copy.deepcopy(case["lock_value"])
    value["container_image"] = image
    case["lock"].write_bytes(_canonical(value))

    with pytest.raises(module.AttestationError, match="container image"):
        module.parse_runtime_lock_bytes(case["lock"].read_bytes())


def test_producer_accepts_documented_iid_fields_but_rejects_identity_ambiguity(
    tmp_path,
):
    module = _load_module()
    case = _case(tmp_path, module)
    full_identity = {
        **IDENTITY,
        "availabilityZone": "us-east-1a",
        "billingProducts": None,
        "instanceType": "p5.48xlarge",
        "marketplaceProductCodes": None,
        "pendingTime": "2026-07-23T20:00:00Z",
        "version": "2017-09-30",
    }

    receipt = _attest(case, imds=_FakeImds(identity=full_identity))

    assert receipt["aws_instance_identity_document"] == full_identity
    assert receipt["instance_id"] == full_identity["instanceId"]

    ambiguous = _case(tmp_path, module, name="ambiguous-iid")
    identity_data = _canonical(IDENTITY).replace(
        b'{"accountId":',
        b'{"accountId":"999999999999","accountId":',
        1,
    )
    with pytest.raises(module.AttestationError, match="repeat|duplicate"):
        _attest(ambiguous, imds=_FakeImds(identity_data=identity_data))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accountId", "123"),
        ("instanceId", "instance-latest"),
        ("region", "global"),
        ("imageId", "ami-latest"),
        ("architecture", "mips"),
        ("privateIp", "203.0.113.8"),
    ],
)
def test_producer_rejects_malformed_authenticated_identity(
    tmp_path,
    field,
    value,
):
    module = _load_module()
    case = _case(tmp_path, module)
    identity = dict(IDENTITY)
    identity[field] = value

    with pytest.raises(module.AttestationError, match="identity|private IP"):
        _attest(case, imds=_FakeImds(identity=identity))


def test_producer_rejects_hash_fact_command_and_container_drift(tmp_path):
    module = _load_module()

    control_case = _case(tmp_path, module, name="control")
    control_case["control"].write_bytes(b"changed control bundle")
    with pytest.raises(module.AttestationError, match="control|lock"):
        _attest(control_case)

    fact_case = _case(tmp_path, module, name="fact")
    drifted = dict(VERSIONS)
    drifted["pytorch"] = "2.7.2+cu128"
    with pytest.raises(module.AttestationError, match="version|drift"):
        _attest(fact_case, commands=_FakeCommands(module, versions=drifted))

    command_case = _case(tmp_path, module, name="command")
    with pytest.raises(module.AttestationError, match="command|runtime"):
        _attest(
            command_case,
            commands=_FakeCommands(module, fail_field="nvidia_driver"),
        )

    container_case = _case(tmp_path, module, name="container")
    with pytest.raises(module.AttestationError, match="container|digest"):
        _attest(
            container_case,
            commands=_FakeCommands(
                module,
                repo_digests=[
                    IMAGE.replace("a" * 64, "f" * 64),
                ],
            ),
        )


def test_version_parser_accepts_real_fabric_manager_output():
    module = _load_module()

    assert module._command_version(
        "fabric_manager",
        b"Fabric Manager version is : 580.159.03\n",
    ) == "580.159.03"


def test_producer_rejects_noncanonical_pkcs7_token_and_boot_id(tmp_path):
    module = _load_module()

    pkcs7_case = _case(tmp_path, module, name="pkcs7")
    with pytest.raises(module.AttestationError, match="PKCS7|base64"):
        _attest(pkcs7_case, imds=_FakeImds(pkcs7_data=b"not base64!\n"))

    token_case = _case(tmp_path, module, name="token")
    with pytest.raises(module.AttestationError, match="token|IMDS"):
        _attest(token_case, imds=_FakeImds(token=""))

    boot_case = _case(tmp_path, module, name="boot")
    boot_case["boot"].write_text(
        "12345678-1234-4ABC-8DEF-1234567890AB\n",
        encoding="ascii",
    )
    with pytest.raises(module.AttestationError, match="boot|UUID"):
        _attest(boot_case)


def test_regular_reader_supports_stable_zero_size_virtual_files(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    virtual = tmp_path / "virtual-boot-id"
    expected = b"12345678-1234-4abc-8def-1234567890ab\n"
    virtual.write_bytes(expected)
    original_fstat = module.os.fstat
    original_stat = module.os.stat

    def zero_size(details):
        return SimpleNamespace(
            st_mode=details.st_mode,
            st_nlink=details.st_nlink,
            st_size=0,
            st_dev=details.st_dev,
            st_ino=details.st_ino,
            st_mtime_ns=details.st_mtime_ns,
            st_ctime_ns=details.st_ctime_ns,
        )

    monkeypatch.setattr(
        module.os,
        "fstat",
        lambda descriptor: zero_size(original_fstat(descriptor)),
    )
    monkeypatch.setattr(
        module.os,
        "stat",
        lambda *args, **kwargs: zero_size(original_stat(*args, **kwargs)),
    )

    assert module.read_regular_input(
        virtual,
        label="virtual boot ID",
        maximum_bytes=128,
    ) == expected


def test_producer_rejects_symlink_and_hardlink_inputs_including_parent_links(
    tmp_path,
):
    module = _load_module()

    symlink_case = _case(tmp_path, module, name="symlink")
    linked_lock = symlink_case["lock"].with_name("lock-link.json")
    linked_lock.symlink_to(symlink_case["lock"])
    symlink_case["lock"] = linked_lock
    with pytest.raises(module.AttestationError, match="safe|link|regular"):
        _attest(symlink_case)

    hardlink_case = _case(tmp_path, module, name="hardlink")
    os.link(
        hardlink_case["control"],
        hardlink_case["control"].with_name("second-control-link.zip"),
    )
    with pytest.raises(module.AttestationError, match="link|regular"):
        _attest(hardlink_case)

    parent_case = _case(tmp_path, module, name="parent-link")
    real_parent = parent_case["control"].parent
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    parent_case["control"] = linked_parent / parent_case["control"].name
    with pytest.raises(module.AttestationError, match="symlink|safe|directory"):
        _attest(parent_case)


def test_apply_is_owner_only_atomic_and_no_replace(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    case = _case(tmp_path, module)
    output = case["output"]
    original_write = module.os.write
    destination_visibility: list[bool] = []

    def observing_write(descriptor, data):
        destination_visibility.append(output.exists())
        return original_write(descriptor, data)

    monkeypatch.setattr(module.os, "write", observing_write)
    receipt = _attest(case, apply=True)

    assert destination_visibility
    assert destination_visibility == [False] * len(destination_visibility)
    assert output.read_bytes() == module.canonical_receipt(receipt)
    details = output.stat(follow_symlinks=False)
    assert stat.S_IMODE(details.st_mode) == 0o600
    assert details.st_nlink == 1
    before = output.read_bytes()
    with pytest.raises(module.AttestationError, match="exist|replace|conflict"):
        _attest(case, apply=True)
    assert output.read_bytes() == before


def test_help_works_without_site_packages():
    completed = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
        env={},
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert "--runtime-lock" in completed.stdout
    assert "--control-bundle" in completed.stdout
    assert "--apply" in completed.stdout


SELECTED_CONTAINER_FACTS = {
    "python": "3.12.11",
    "pytorch": "2.9.0+cu130",
    "cuda": "13.0",
    "cudnn": "9.10.2",
    "nccl": "2.28.3",
}
SELECTED_HOST_FACTS = {
    "cuda": "13.2",
    "nvidia_driver": "595.71.05",
    "fabric_manager": "595.71.05",
    "docker": "28.5.1",
    "nvidia_container_runtime": "1.18.0",
    "aws_cli": "2.31.7",
    "kernel": "6.17",
    "efa": "1.47.0",
    "ofi_nccl": "1.18.0",
    "nvlsm": "595.71.05",
}
P6_AMI_ID = "ami-0260c4d597dcc8641"
P6_AMI_OWNER = "898082745236"
UNSUPPORTED_FRAMEWORK_AMI = "ami-0b39828e6910b0bb8"


def _selected_profile(kind: str):
    if kind == "p6":
        return SimpleNamespace(
            profile_id="aws-p6-b300.48xlarge-v3",
            provider="aws-p6-b300.48xlarge",
            instance_type="p6-b300.48xlarge",
            gpu_model="NVIDIA B300",
            allocated_gpus=8,
            architecture="x86_64",
            software_floors=(
                ("cuda", "13.0"),
                ("efa", "1.44.0"),
                ("kernel", "6.1"),
                ("nvidia_driver", "R580"),
                ("nvlink", "R580"),
                ("ofi_nccl", "1.17.1"),
            ),
            sha256=hashlib.sha256(b"p6-profile").hexdigest(),
        )
    if kind == "p5":
        return SimpleNamespace(
            profile_id="aws-p5.48xlarge-v3",
            provider="aws-p5.48xlarge",
            instance_type="p5.48xlarge",
            gpu_model="NVIDIA H100 80GB",
            allocated_gpus=8,
            architecture="x86_64",
            software_floors=(),
            sha256=hashlib.sha256(b"p5-profile").hexdigest(),
        )
    raise AssertionError(kind)


def _selected_lock(
    profile,
    control_sha256: str,
    *,
    ami_id: str = P6_AMI_ID,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_commit": "b" * 40,
        "source_tree": "c" * 40,
        "control_bundle_sha256": control_sha256,
        "profile_sha256": profile.sha256,
        "ami_id": ami_id,
        "ami_owner_id": (
            P6_AMI_OWNER if profile.profile_id.startswith("aws-p6") else "210987654321"
        ),
        "container_image": IMAGE,
        "container_image_digest": IMAGE_DIGEST,
        "versions": {
            **SELECTED_CONTAINER_FACTS,
            "nvidia_driver": SELECTED_HOST_FACTS["nvidia_driver"],
            "fabric_manager": SELECTED_HOST_FACTS["fabric_manager"],
            "docker": SELECTED_HOST_FACTS["docker"],
            "nvidia_container_runtime": SELECTED_HOST_FACTS[
                "nvidia_container_runtime"
            ],
            "aws_cli": SELECTED_HOST_FACTS["aws_cli"],
        },
    }


class _SelectedCommands:
    def __init__(
        self,
        module,
        *,
        host_facts: dict[str, str] | None = None,
        container_facts: dict[str, str] | None = None,
        gpu_names: list[str] | None = None,
        missing_host_field: str | None = None,
    ) -> None:
        self.module = module
        self.host_facts = dict(
            SELECTED_HOST_FACTS if host_facts is None else host_facts
        )
        self.container_facts = dict(
            SELECTED_CONTAINER_FACTS
            if container_facts is None
            else container_facts
        )
        self.gpu_names = list(
            ["NVIDIA B300"] * 8 if gpu_names is None else gpu_names
        )
        self.missing_host_field = missing_host_field
        self.calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def run(self, argv, *, environment) -> bytes:
        rendered = tuple(argv)
        self.calls.append((rendered, dict(environment)))
        for field, expected in self.module.SELECTED_HOST_VERSION_COMMANDS.items():
            if rendered == tuple(expected):
                if field == self.missing_host_field:
                    raise OSError("synthetic missing host measurement")
                return (self.host_facts[field] + "\n").encode("ascii")
        if rendered == tuple(self.module.gpu_names_argv()):
            return ("\n".join(self.gpu_names) + "\n").encode("utf-8")
        if rendered == tuple(self.module.container_inspect_argv(IMAGE)):
            return _canonical([IMAGE])
        if rendered == tuple(self.module.container_python_exists_argv(IMAGE)):
            return b""
        if rendered == tuple(self.module.container_facts_argv(IMAGE)):
            return _canonical(self.container_facts)
        raise AssertionError(f"unexpected selected-GPU command: {rendered!r}")


def _selected_case(
    tmp_path: Path,
    module,
    *,
    profile_kind: str = "p6",
    name: str | None = None,
):
    profile = _selected_profile(profile_kind)
    root = tmp_path / (name or f"selected-{profile_kind}")
    root.mkdir()
    control = root / "control.zip"
    control.write_bytes(b"selected GPU control bundle")
    lock = root / "runtime-lock.json"
    lock_value = _selected_lock(
        profile,
        hashlib.sha256(control.read_bytes()).hexdigest(),
        ami_id=P6_AMI_ID if profile_kind == "p6" else IDENTITY["imageId"],
    )
    lock.write_bytes(_canonical(lock_value))
    boot = root / "boot_id"
    boot.write_text("12345678-1234-4abc-8def-1234567890ab\n", encoding="ascii")
    identity = {
        **IDENTITY,
        "imageId": lock_value["ami_id"],
        "instanceType": profile.instance_type,
    }
    return {
        "module": module,
        "profile": profile,
        "control": control,
        "lock": lock,
        "lock_value": lock_value,
        "boot": boot,
        "identity": identity,
        "output": root / "gpu-evidence.json",
    }


def _attest_selected(case, *, commands=None, identity=None):
    module = case["module"]
    return module.attest_selected_gpu_environment(
        selected_profile=case["profile"],
        runtime_lock_path=case["lock"],
        control_bundle_path=case["control"],
        output_path=case["output"],
        apply=False,
        imds_reader=_FakeImds(
            identity=case["identity"] if identity is None else identity
        ),
        command_reader=commands or _SelectedCommands(module),
        boot_id_path=case["boot"],
    )


def test_selected_gpu_attestation_measures_framework_only_in_locked_container(
    tmp_path,
):
    module = _load_module()
    case = _selected_case(tmp_path, module)
    commands = _SelectedCommands(module)

    evidence = _attest_selected(case, commands=commands)

    assert evidence["evidence_type"] == "memorysplit-aws-gpu-attestation-v1"
    assert evidence["profile_id"] == case["profile"].profile_id
    assert evidence["provider"] == case["profile"].provider
    assert evidence["instance_type"] == "p6-b300.48xlarge"
    assert evidence["gpu_model"] == "NVIDIA B300"
    assert evidence["gpu_count"] == 8
    assert evidence["host_facts"] == SELECTED_HOST_FACTS
    assert module.SELECTED_HOST_VERSION_COMMANDS["nvlsm"] == (
        "/usr/bin/nvlsm",
        "--version",
    )
    assert evidence["container_facts"] == SELECTED_CONTAINER_FACTS
    assert module.parse_gpu_evidence_bytes(
        module.canonical_gpu_evidence(evidence)
    ) == evidence
    assert module.attest_legacy_p5_environment is module.attest_environment

    container_argv = module.container_facts_argv(IMAGE)
    assert container_argv in [argv for argv, _ in commands.calls]
    assert container_argv[:7] == (
        "/usr/bin/docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--pull",
        "never",
    )
    assert container_argv[container_argv.index("--user") + 1] == "10001:10001"
    assert container_argv[container_argv.index("--gpus") + 1] == "all"
    assert "all" in container_argv
    assert "--entrypoint" in container_argv
    assert container_argv[container_argv.index("--entrypoint") + 1] == (
        "/opt/conda/bin/python"
    )
    assert IMAGE in container_argv
    assert tuple(module.container_inspect_argv(IMAGE)) in [
        argv for argv, _ in commands.calls
    ]
    assert tuple(module.container_python_exists_argv(IMAGE)) in [
        argv for argv, _ in commands.calls
    ]
    host_commands = {
        tuple(argv) for argv in module.SELECTED_HOST_VERSION_COMMANDS.values()
    } | {tuple(module.gpu_names_argv()), tuple(module.container_inspect_argv(IMAGE))}
    for argv, _ in commands.calls:
        if argv in host_commands:
            assert "torch" not in " ".join(argv)
            assert argv[0] != "/usr/bin/python3"
    assert not case["output"].exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cuda", "12.9"),
        ("nvidia_driver", "579.99"),
        ("kernel", "6.0"),
        ("efa", "1.43.9"),
        ("ofi_nccl", "1.17.0"),
        ("nvlsm", "579.99"),
    ],
)
def test_p6_attestation_rejects_measured_host_facts_below_official_floor(
    tmp_path,
    field,
    value,
):
    module = _load_module()
    case = _selected_case(tmp_path, module)
    host_facts = dict(SELECTED_HOST_FACTS)
    host_facts[field] = value

    with pytest.raises(module.AttestationError, match="floor|P6|B300"):
        _attest_selected(
            case,
            commands=_SelectedCommands(module, host_facts=host_facts),
        )


def test_p6_attestation_requires_exact_b300_identity_and_supported_base_ami(tmp_path):
    module = _load_module()

    wrong_gpu = _selected_case(tmp_path, module, profile_kind="p6")
    with pytest.raises(module.AttestationError, match="B300|GPU|profile"):
        _attest_selected(
            wrong_gpu,
            commands=_SelectedCommands(module, gpu_names=["NVIDIA B200"] * 8),
        )

    wrong_ami = _selected_case(
        tmp_path,
        module,
        profile_kind="p6",
        name="selected-p6-wrong-ami",
    )
    value = copy.deepcopy(wrong_ami["lock_value"])
    value["ami_id"] = UNSUPPORTED_FRAMEWORK_AMI
    wrong_ami["lock"].write_bytes(_canonical(value))
    identity = {
        **wrong_ami["identity"],
        "imageId": UNSUPPORTED_FRAMEWORK_AMI,
    }
    with pytest.raises(module.AttestationError, match="AMI|P6|supported"):
        _attest_selected(wrong_ami, identity=identity)


def test_p6_attestation_rejects_missing_or_forged_nvlsm_evidence(tmp_path):
    module = _load_module()
    missing = _selected_case(tmp_path, module, profile_kind="p6")
    with pytest.raises(module.AttestationError, match="nvlsm|NVLink|command"):
        _attest_selected(
            missing,
            commands=_SelectedCommands(module, missing_host_field="nvlsm"),
        )

    forged = _selected_case(
        tmp_path,
        module,
        profile_kind="p6",
        name="selected-p6-forged-nvlsm",
    )
    host_facts = dict(SELECTED_HOST_FACTS)
    host_facts["nvlsm"] = "600.1.1"
    with pytest.raises(module.AttestationError, match="nvlsm|NVLink|driver|coherent"):
        _attest_selected(
            forged,
            commands=_SelectedCommands(module, host_facts=host_facts),
        )


def test_selected_gpu_attestation_rejects_cross_profile_evidence(tmp_path):
    module = _load_module()

    p6 = _selected_case(tmp_path, module, profile_kind="p6")
    p6_identity_as_p5 = {**p6["identity"], "instanceType": "p5.48xlarge"}
    with pytest.raises(module.AttestationError, match="instance|profile"):
        _attest_selected(p6, identity=p6_identity_as_p5)

    p5 = _selected_case(tmp_path, module, profile_kind="p5")
    with pytest.raises(module.AttestationError, match="H100|GPU|profile"):
        _attest_selected(
            p5,
            commands=_SelectedCommands(module, gpu_names=["NVIDIA B300"] * 8),
        )


@pytest.mark.parametrize(
    "gpu_name",
    ["NVIDIA H100 80GB", "NVIDIA H100 80GB HBM3"],
)
def test_p5_attestation_accepts_only_established_exact_h100_names(
    tmp_path,
    gpu_name,
):
    from msctl.aws_contracts import (
        allowed_gpu_product_names,
        validate_gpu_product_names,
    )

    module = _load_module()
    case = _selected_case(
        tmp_path,
        module,
        profile_kind="p5",
        name="p5-" + gpu_name.replace(" ", "-"),
    )
    evidence = _attest_selected(
        case,
        commands=_SelectedCommands(module, gpu_names=[gpu_name] * 8),
    )

    assert evidence["gpu_model"] == gpu_name
    assert allowed_gpu_product_names("aws-p5.48xlarge-v3") == (
        "NVIDIA H100 80GB",
        "NVIDIA H100 80GB HBM3",
    )
    assert validate_gpu_product_names(
        "aws-p5.48xlarge-v3",
        [gpu_name] * 8,
        expected_count=8,
    ) == gpu_name


@pytest.mark.parametrize(
    "gpu_names",
    [
        ["NVIDIA H100"] * 8,
        ["NVIDIA H100 80GB HBM3 Engineering"] * 8,
        ["NVIDIA H100 80GB"] * 7 + ["NVIDIA H100 80GB HBM3"],
    ],
)
def test_p5_attestation_rejects_prefixes_and_mixed_h100_names(tmp_path, gpu_names):
    module = _load_module()
    case = _selected_case(tmp_path, module, profile_kind="p5")

    with pytest.raises(module.AttestationError, match="GPU|profile|product"):
        _attest_selected(
            case,
            commands=_SelectedCommands(module, gpu_names=gpu_names),
        )


def test_gpu_evidence_parser_revalidates_profile_identity_and_p6_floors(tmp_path):
    module = _load_module()
    p6 = _selected_case(tmp_path, module, profile_kind="p6")
    p6_evidence = _attest_selected(p6)
    p6_evidence["host_facts"]["efa"] = "1.43.9"
    with pytest.raises(module.AttestationError, match="floor|P6|B300"):
        module.parse_gpu_evidence_bytes(_canonical(p6_evidence))

    p5 = _selected_case(tmp_path, module, profile_kind="p5")
    p5_evidence = _attest_selected(
        p5,
        commands=_SelectedCommands(
            module,
            gpu_names=["NVIDIA H100 80GB"] * 8,
        ),
    )
    p5_evidence["provider"] = "aws-p6-b300.48xlarge"
    with pytest.raises(module.AttestationError, match="profile|identity|provider"):
        module.parse_gpu_evidence_bytes(_canonical(p5_evidence))


def test_selected_host_fact_commands_parse_real_efa_and_ofi_version_markers():
    module = _load_module()

    assert module.SELECTED_HOST_VERSION_COMMANDS["efa"] == (
        "/usr/bin/cat",
        "/opt/amazon/efa_installed_packages",
    )
    assert module.SELECTED_HOST_VERSION_COMMANDS["ofi_nccl"] == (
        "/usr/bin/strings",
        "/opt/amazon/ofi-nccl/lib/libnccl-net.so",
    )
    assert module._selected_host_version(
        "efa",
        b"# EFA installer version: 1.47.0\n",
    ) == "1.47.0"
    assert module._selected_host_version(
        "ofi_nccl",
        b"NET/OFI Initializing aws-ofi-nccl 1.18.0-aws\n",
    ) == "1.18.0"


class _ControllerRunner:
    def __init__(self, *outputs: object) -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[list[str], str]] = []

    def run_json(self, argv, *, operation: str):
        self.calls.append((list(argv), operation))
        if not self.outputs:
            raise AssertionError(f"unexpected AWS call: {argv}")
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def _controller_backend(
    tmp_path: Path,
    *,
    runner: _ControllerRunner | None = None,
    identity_verifier=None,
    environ: dict[str, str] | None = None,
):
    from cluster.aws.p5.profile import load_aws_p5_profile
    from msctl.aws_p5 import AwsP5Backend

    profile = load_aws_p5_profile(PROFILE)
    runtime = SimpleNamespace(
        region=IDENTITY["region"],
        s3_root="s3://memorysplit-prod/confirmatory-v3",
        ami_id=IDENTITY["imageId"],
        container_image=IMAGE,
        container_digest=IMAGE_DIGEST,
        uid=1000,
        gid=1000,
    )
    return AwsP5Backend(
        profile=profile,
        runtime=runtime,
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "controller-state",
        runner=runner or _ControllerRunner(),
        identity_verifier=(
            identity_verifier
            if identity_verifier is not None
            else (lambda identity, pkcs7, region: True)
        ),
        environ={} if environ is None else environ,
    )


def _selected_instance(
    *,
    account_id: str = str(IDENTITY["accountId"]),
    instance_id: str = str(IDENTITY["instanceId"]),
    image_id: str = str(IDENTITY["imageId"]),
    instance_type: str = "p5.48xlarge",
    state: str = "running",
    architecture: str = str(IDENTITY["architecture"]),
    private_ip: str = str(IDENTITY["privateIp"]),
) -> dict[str, object]:
    return {
        "instance": {
            "account_id": account_id,
            "instance_id": instance_id,
            "image_id": image_id,
            "instance_type": instance_type,
            "state": state,
            "architecture": architecture,
            "private_ip": private_ip,
        }
    }


def _selected_image(
    *,
    image_id: str = str(IDENTITY["imageId"]),
    owner_id: str = "210987654321",
    state: str = "available",
    architecture: str = str(IDENTITY["architecture"]),
) -> dict[str, object]:
    return {
        "image": {
            "image_id": image_id,
            "owner_id": owner_id,
            "state": state,
            "architecture": architecture,
        }
    }


def _published_outputs(receipt_bytes: bytes) -> tuple[dict[str, object], ...]:
    digest = hashlib.sha256(receipt_bytes).hexdigest()
    checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
    return (
        {
            "object": {
                "checksum_sha256": checksum,
                "version_id": "environment-version-1",
            }
        },
        {
            "object": {
                "checksum_sha256": checksum,
                "content_length": len(receipt_bytes),
                "metadata": {"environment-receipt-sha256": digest},
                "version_id": "environment-version-1",
            }
        },
    )


def _controller_call(
    backend,
    case: dict[str, object],
    *,
    remote_root: Path,
    apply: bool,
):
    return backend.env_ensure(
        root=remote_root,
        runtime_lock=case["lock"],
        control_bundle=case["control"],
        instance_id=IDENTITY["instanceId"],
        receipt=case["output"],
        apply=apply,
    )


def test_controller_dry_run_returns_exact_remote_intent_and_apply_publishes_v2(
    tmp_path,
):
    module = _load_module()
    case = _case(tmp_path, module)
    receipt = _attest(case, apply=True)
    receipt_bytes = case["output"].read_bytes()
    verifier_calls: list[tuple[dict[str, object], str, str]] = []

    def verify(identity, pkcs7, region):
        verifier_calls.append((dict(identity), pkcs7, region))
        return True

    runner = _ControllerRunner(
        _selected_instance(),
        _selected_image(),
        *_published_outputs(receipt_bytes),
    )
    backend = _controller_backend(
        tmp_path,
        runner=runner,
        identity_verifier=verify,
    )
    remote_root = tmp_path / "remote-root"

    planned = _controller_call(
        backend,
        case,
        remote_root=remote_root,
        apply=False,
    )

    digest = hashlib.sha256(receipt_bytes).hexdigest()
    expected_key = f"environments/{digest}/receipt.json"
    assert planned["environment_receipt_sha256"] == digest
    assert planned["receipt_key"] == expected_key
    assert planned["receipt_uri"] == (
        "s3://memorysplit-prod/confirmatory-v3/" + expected_key
    )
    assert planned["published"] is False
    assert planned["verified"] is True
    assert runner.calls == []
    assert not remote_root.exists()
    argv = planned["remote_attestation_argv"]
    assert argv[0] == "/usr/bin/python3"
    assert argv[1] == str(
        remote_root / "cluster" / "aws" / "p5" / "attest_environment.py"
    )
    assert argv[argv.index("--profile") + 1] == str(
        remote_root
        / "cluster"
        / "profiles"
        / "aws-p5.48xlarge-v3.json"
    )
    assert argv[argv.index("--runtime-lock") + 1] == str(case["lock"])
    assert argv[argv.index("--control-bundle") + 1] == str(case["control"])
    assert argv[argv.index("--out") + 1] == str(case["output"])
    assert argv[-1] == "--apply"
    intent = planned["ssm_intent"]
    assert intent["operation"] == "attest-environment"
    assert intent["steps"] == [{"name": "attest-environment", "argv": argv}]
    assert intent["environment"] == {
        "AWS_REGION": IDENTITY["region"],
        "MS_AWS_AMI_ID": IDENTITY["imageId"],
        "MS_CONTAINER_DIGEST": IMAGE_DIGEST,
        "MS_CONTAINER_IMAGE": IMAGE,
    }
    serialized_intent = json.dumps(intent, sort_keys=True)
    for forbidden in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_CONFIG_FILE",
        "AWS_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
    ):
        assert forbidden not in serialized_intent

    applied = _controller_call(
        backend,
        case,
        remote_root=remote_root,
        apply=True,
    )

    assert applied == {
        **planned,
        "published": True,
        "version_id": "environment-version-1",
    }
    assert len(verifier_calls) == 2
    assert verifier_calls[0] == (
        receipt["aws_instance_identity_document"],
        PKCS7,
        "us-east-1",
    )
    assert [operation for _, operation in runner.calls] == [
        "verify selected environment instance",
        "verify selected environment AMI",
        "publish environment receipt",
        "verify environment receipt publication",
    ]
    put, head = (argv for argv, _ in runner.calls[-2:])
    assert "--if-none-match" in put and put[put.index("--if-none-match") + 1] == "*"
    assert "--checksum-algorithm" in put
    assert "--checksum-sha256" in put
    assert "--checksum-mode" in head


def test_controller_rejects_every_receipt_binding_and_signature_drift(tmp_path):
    module = _load_module()
    baseline = _case(tmp_path, module, name="baseline-controller")
    _attest(baseline, apply=True)
    receipt = json.loads(baseline["output"].read_text(encoding="ascii"))
    mutations = [
        ("schema", lambda value: value.update(schema_version=2.0)),
        ("type", lambda value: value.update(receipt_type="other")),
        ("provider", lambda value: value.update(provider="aws")),
        ("profile", lambda value: value.update(profile_sha256="0" * 64)),
        ("lock", lambda value: value.update(runtime_lock_sha256="0" * 64)),
        ("control", lambda value: value.update(control_bundle_sha256="0" * 64)),
        ("commit", lambda value: value.update(source_commit="0" * 40)),
        ("tree", lambda value: value.update(source_tree="0" * 40)),
        ("image", lambda value: value.update(container_image=IMAGE + "-other")),
        (
            "digest",
            lambda value: value.update(
                container_image_digest="sha256:" + "0" * 64
            ),
        ),
        ("account", lambda value: value.update(account_id="000000000000")),
        ("instance", lambda value: value.update(instance_id="i-00000000")),
        ("region", lambda value: value.update(region="us-west-2")),
        ("ami", lambda value: value.update(ami_id="ami-00000000")),
        ("boot", lambda value: value.update(boot_id="not-a-uuid")),
        (
            "facts",
            lambda value: value["runtime_facts"].update(cuda="12.7"),
        ),
        ("unknown", lambda value: value.update(unexpected=True)),
        ("missing", lambda value: value.pop("source_tree")),
    ]
    rejected: list[str] = []
    for index, (name, mutate) in enumerate(mutations):
        case = _case(tmp_path, module, name=f"receipt-drift-{index}")
        value = copy.deepcopy(receipt)
        mutate(value)
        case["output"].write_bytes(_canonical(value))
        backend = _controller_backend(tmp_path / f"backend-{index}")
        try:
            _controller_call(
                backend,
                case,
                remote_root=tmp_path / f"remote-{index}",
                apply=False,
            )
        except Exception as error:
            if getattr(error, "code", None) == "ENVIRONMENT_RECEIPT_INVALID":
                rejected.append(name)
    assert rejected == [name for name, _ in mutations]

    signature_case = _case(tmp_path, module, name="signature")
    _attest(signature_case, apply=True)
    backend = _controller_backend(
        tmp_path / "signature-backend",
        identity_verifier=lambda identity, pkcs7, region: False,
    )
    with pytest.raises(Exception) as caught:
        _controller_call(
            backend,
            signature_case,
            remote_root=tmp_path / "signature-remote",
            apply=False,
        )
    assert getattr(caught.value, "code", None) == "ENVIRONMENT_RECEIPT_INVALID"


@pytest.mark.parametrize(
    "mutation",
    [
        "account",
        "instance",
        "ami",
        "instance-type",
        "state",
        "architecture",
        "private-ip",
        "ami-owner",
        "ami-state",
    ],
)
def test_controller_rejects_selected_instance_and_ami_drift(
    tmp_path,
    mutation,
):
    module = _load_module()
    case = _case(tmp_path, module)
    _attest(case, apply=True)
    instance = _selected_instance()
    image = _selected_image()
    if mutation == "account":
        instance["instance"]["account_id"] = "000000000000"
    elif mutation == "instance":
        instance["instance"]["instance_id"] = "i-00000000"
    elif mutation == "ami":
        instance["instance"]["image_id"] = "ami-00000000"
    elif mutation == "instance-type":
        instance["instance"]["instance_type"] = "p5.4xlarge"
    elif mutation == "state":
        instance["instance"]["state"] = "terminated"
    elif mutation == "architecture":
        instance["instance"]["architecture"] = "arm64"
    elif mutation == "private-ip":
        instance["instance"]["private_ip"] = "10.99.99.99"
    elif mutation == "ami-owner":
        image["image"]["owner_id"] = "000000000000"
    elif mutation == "ami-state":
        image["image"]["state"] = "failed"
    else:
        raise AssertionError(mutation)
    runner = _ControllerRunner(instance, image)
    backend = _controller_backend(tmp_path, runner=runner)

    with pytest.raises(Exception) as caught:
        _controller_call(
            backend,
            case,
            remote_root=tmp_path / "remote",
            apply=True,
        )

    assert getattr(caught.value, "code", None) == "ENVIRONMENT_RECEIPT_INVALID"
    assert all("put-object" not in argv for argv, _ in runner.calls)


@pytest.mark.parametrize(
    "mutation",
    [
        "put-checksum",
        "put-version",
        "put-null-version",
        "head-checksum",
        "head-version",
        "head-null-version",
    ],
)
def test_s3_environment_publication_requires_checksum_version_and_no_replace(
    tmp_path,
    mutation,
):
    from msctl.errors import MsctlError

    module = _load_module()
    case = _case(tmp_path, module)
    _attest(case, apply=True)
    receipt_bytes = case["output"].read_bytes()
    put, head = copy.deepcopy(_published_outputs(receipt_bytes))
    if mutation == "put-checksum":
        put["object"]["checksum_sha256"] = "bad"
    elif mutation == "put-version":
        put["object"]["version_id"] = ""
    elif mutation == "put-null-version":
        put["object"]["version_id"] = "null"
    elif mutation == "head-checksum":
        head["object"]["checksum_sha256"] = "bad"
    elif mutation == "head-version":
        head["object"]["version_id"] = ""
    elif mutation == "head-null-version":
        head["object"]["version_id"] = "null"
    runner = _ControllerRunner(_selected_instance(), _selected_image(), put, head)
    backend = _controller_backend(tmp_path, runner=runner)

    with pytest.raises(MsctlError) as caught:
        _controller_call(
            backend,
            case,
            remote_root=tmp_path / "remote",
            apply=True,
        )

    assert caught.value.code == "S3_OBJECT_MISMATCH"
    put_argv = next(argv for argv, _ in runner.calls if "put-object" in argv)
    assert put_argv[put_argv.index("--if-none-match") + 1] == "*"


def test_existing_identical_s3_receipt_is_verified_without_replacement(tmp_path):
    from msctl.errors import MsctlError

    module = _load_module()
    case = _case(tmp_path, module)
    _attest(case, apply=True)
    _, head = _published_outputs(case["output"].read_bytes())
    runner = _ControllerRunner(
        _selected_instance(),
        _selected_image(),
        MsctlError("AWS_COMMAND_FAILED", "precondition failed"),
        head,
    )
    backend = _controller_backend(tmp_path, runner=runner)

    result = _controller_call(
        backend,
        case,
        remote_root=tmp_path / "remote",
        apply=True,
    )

    assert result["published"] is True
    assert result["version_id"] == "environment-version-1"
    assert sum("put-object" in argv for argv, _ in runner.calls) == 1


def test_local_profile_config_reaches_aws_argv_but_not_remote_intent(tmp_path):
    module = _load_module()
    case = _case(tmp_path, module)
    _attest(case, apply=True)
    receipt_bytes = case["output"].read_bytes()
    controller_environment = {
        "AWS_PROFILE": "memorysplit-sso",
        "AWS_CONFIG_FILE": "/controller/aws/config",
        "AWS_SHARED_CREDENTIALS_FILE": "/controller/aws/credentials",
        "HOME": "/controller/home",
    }
    runner = _ControllerRunner(
        _selected_instance(),
        _selected_image(),
        *_published_outputs(receipt_bytes),
    )
    backend = _controller_backend(
        tmp_path,
        runner=runner,
        environ=controller_environment,
    )

    planned = _controller_call(
        backend,
        case,
        remote_root=tmp_path / "remote",
        apply=False,
    )
    assert not set(controller_environment) & set(planned["ssm_intent"]["environment"])

    _controller_call(
        backend,
        case,
        remote_root=tmp_path / "remote",
        apply=True,
    )
    for argv, _ in runner.calls:
        assert "AWS_PROFILE=memorysplit-sso" in argv
        assert "AWS_CONFIG_FILE=/controller/aws/config" in argv
        assert (
            "AWS_SHARED_CREDENTIALS_FILE=/controller/aws/credentials" in argv
        )
        assert "HOME=/controller/home" in argv


@pytest.mark.parametrize(
    "name",
    [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
    ],
)
def test_controller_rejects_static_aws_keys(name, tmp_path):
    from msctl.errors import MsctlError

    with pytest.raises(MsctlError) as caught:
        _controller_backend(
            tmp_path,
            environ={name: "must-not-be-inherited"},
        )

    assert caught.value.code == "AWS_RUNTIME_INVALID"


def test_default_pkcs7_verifier_fails_closed_without_reviewed_region_anchor():
    from msctl.aws_p5 import _verify_instance_identity_pkcs7
    from msctl.errors import MsctlError

    with pytest.raises(MsctlError) as caught:
        _verify_instance_identity_pkcs7(IDENTITY, PKCS7, "us-west-2")

    assert caught.value.code == "ENVIRONMENT_RECEIPT_INVALID"
