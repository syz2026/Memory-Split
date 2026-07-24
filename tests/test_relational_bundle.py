from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

import pytest

import scripts.package_relational_run as package_module
from scripts.make_relational_manifest import write_manifests
from scripts.package_relational_run import main, package_run


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COUNTS = {"160m": 15, "360m": 6}
TOKENIZER_ASSETS = {
    "vendor/tiktoken/6c7ea1a7e38e3a7f062df639a5b80947f075ffe6",
    "vendor/tiktoken/6d1cbeee0f20b3d9449abfede4726ed8212e3aee",
}
TASK3C_CHECKPOINT_TEST_MODULES = (
    "tests/test_aws_checkpoint_mirror.py",
    "tests/test_checkpoint_mirror_attempt_cleanup.py",
)


def _portable_inputs(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "route-policy.json").write_text(
        '{"policy":{"hop_cost":0.25,"read_cost":0.25,"write_cost":1.0}}\n'
    )
    configs = {}
    manifests = {}
    for scale, count in EXPECTED_COUNTS.items():
        config_paths = []
        config_dir = root / "configs" / scale
        config_dir.mkdir(parents=True)
        for index in range(count):
            relative = Path("configs") / scale / f"run-{index:02d}.yaml"
            (root / relative).write_text(
                f"model: d{scale}\n"
                f"data_rel: corpora/{scale}/seed-{index % 3}\n"
                f"out_rel: runs/{scale}/run-{index:02d}\n"
            )
            config_paths.append(relative.as_posix())
        manifest = Path("configs") / f"{scale}.tsv"
        (root / manifest).write_text(
            "".join(f"{path}\n" for path in config_paths)
        )
        configs[scale] = config_paths
        manifests[scale] = manifest.as_posix()
    return root, configs, manifests


def _archive_files(path):
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        files = {
            member.name: archive.extractfile(member).read()
            for member in members
        }
    return members, files


def test_bundle_is_deterministic_relative_and_hash_complete(tmp_path):
    input_root, configs, manifests = _portable_inputs(tmp_path)
    first = package_run(
        tmp_path / "first.tar.gz",
        source_root=REPO_ROOT,
        input_root=input_root,
        config_inputs=configs,
        manifest_inputs=manifests,
        route_policy="route-policy.json",
    )
    second = package_run(
        tmp_path / "second.tar.gz",
        source_root=REPO_ROOT,
        input_root=input_root,
        config_inputs=configs,
        manifest_inputs=manifests,
        route_policy="route-policy.json",
    )

    assert first.read_bytes() == second.read_bytes()
    members, files = _archive_files(first)
    names = set(files)
    assert {
        "manifest.json",
        "source-revision.txt",
        "route-policy.json",
        "fixtures/relational-smoke.json",
        "tests/fixtures/relational-smoke-route-policy.json",
        "configs/160m.tsv",
        "configs/360m.tsv",
        "scripts/relational_smoke_test.py",
        "scripts/run_relational_evals.py",
        "evals/relational_generate.py",
        "tests/test_relational_smoke.py",
        "tests/test_relational_bundle.py",
        *TOKENIZER_ASSETS,
    } <= names
    assert sum(name.startswith("configs/160m/run-") for name in names) == 15
    assert sum(name.startswith("configs/360m/run-") for name in names) == 6
    assert all(
        not Path(member.name).is_absolute()
        and ".." not in Path(member.name).parts
        and "\\" not in member.name
        for member in members
    )
    assert not any(
        name.endswith((".bin", ".pt", ".pth"))
        or "checkpoint" in name.lower()
        or "ckpt" in name.lower()
        for name in names
    )
    for module in TASK3C_CHECKPOINT_TEST_MODULES:
        assert module not in names

    manifest = json.loads(files["manifest.json"])
    assert manifest["expected_run_counts"] == EXPECTED_COUNTS
    assert manifest["required_environment"] == ["DATA_ROOT", "OUT_ROOT"]
    assert files["source-revision.txt"].decode().strip() == manifest[
        "source_revision"
    ]
    assert json.loads(files["fixtures/relational-smoke.json"]) == {
        "data_seed": 1,
        "eval_pairs_per_task": 4,
        "n_entities": 32,
        "steps": 2,
        "total_tokens": 40_000,
    }
    indexed = {item["path"]: item for item in manifest["members"]}
    assert set(indexed) == names - {"manifest.json"}
    assert TOKENIZER_ASSETS <= set(indexed)
    for name, item in indexed.items():
        assert item["bytes"] == len(files[name])
        assert item["sha256"] == hashlib.sha256(files[name]).hexdigest()
    for member in members:
        assert member.pax_headers["SHA256"] == hashlib.sha256(
            files[member.name]
        ).hexdigest()


def test_bundle_integrates_production_manifests_and_real_smoke_report(tmp_path):
    assert hasattr(package_module, "production_inputs")
    input_root = tmp_path / "inputs"
    generated = write_manifests(input_root)
    (input_root / "route-policy.json").write_text(
        '{"policy":{"hop_cost":0.25,"read_cost":0.25,"write_cost":1.0},'
        '"policy_sha256":'
        '"0214cd5dd63e7534dc786569f8b789b6c614ffbe219c84887bd3a71b57bcf058"}\n'
    )
    smoke_report = {
        "shared_stream": True,
        "dense_steps": 2,
        "split_steps": 2,
        "resume_exact": True,
        "memory_modes": ["off", "on"],
        "pairs_complete": True,
    }
    (input_root / "smoke-report.json").write_text(
        json.dumps(smoke_report, sort_keys=True) + "\n"
    )
    production = package_module.production_inputs(input_root)

    archive = package_run(
        tmp_path / "production.tar.gz",
        source_root=REPO_ROOT,
        input_root=input_root,
        config_inputs=production["configs"],
        manifest_inputs=production["manifests"],
        route_policy="route-policy.json",
        smoke_report="smoke-report.json",
    )
    _, files = _archive_files(archive)

    assert json.loads(
        files["fixtures/relational-smoke-report.json"]
    ) == smoke_report
    for scale, result in generated.items():
        manifest = result["manifest"].relative_to(input_root).as_posix()
        assert files[manifest] == result["manifest"].read_bytes()
        for config in result["configs"]:
            relative = config.relative_to(input_root).as_posix()
            assert files[relative] == config.read_bytes()


@pytest.mark.parametrize(
    "bad_path",
    [
        "/Users/example/corpus",
        "/scratch/relational",
        "s3://example-bucket/corpus",
        "../outside",
        r"C:\datasets\relational",
    ],
)
def test_bundle_rejects_nonportable_paths_in_supplied_files(
    tmp_path,
    bad_path,
):
    input_root, configs, manifests = _portable_inputs(tmp_path)
    (input_root / configs["160m"][0]).write_text(f"data_rel: {bad_path}\n")

    with pytest.raises(ValueError, match="portable"):
        package_run(
            tmp_path / "bad.tar.gz",
            source_root=REPO_ROOT,
            input_root=input_root,
            config_inputs=configs,
            manifest_inputs=manifests,
            route_policy="route-policy.json",
        )


@pytest.mark.parametrize(
    "bad_input",
    ["/absolute/run.yaml", "../escape.yaml", r"configs\run.yaml"],
)
def test_bundle_rejects_nonportable_supplied_member_names(
    tmp_path,
    bad_input,
):
    input_root, configs, manifests = _portable_inputs(tmp_path)
    configs["160m"][0] = bad_input

    with pytest.raises(ValueError, match="portable|traversal"):
        package_run(
            tmp_path / "bad-name.tar.gz",
            source_root=REPO_ROOT,
            input_root=input_root,
            config_inputs=configs,
            manifest_inputs=manifests,
            route_policy="route-policy.json",
        )


def test_bundle_rejects_symlink_escape_from_input_root(tmp_path):
    input_root, configs, manifests = _portable_inputs(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "route-policy.json").write_text('{"policy":{}}\n')
    (input_root / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="input root|symlink"):
        package_run(
            tmp_path / "symlink.tar.gz",
            source_root=REPO_ROOT,
            input_root=input_root,
            config_inputs=configs,
            manifest_inputs=manifests,
            route_policy="linked/route-policy.json",
        )


def test_bundle_rejects_checkpoint_suffix_as_a_config(tmp_path):
    input_root, configs, manifests = _portable_inputs(tmp_path)
    original = input_root / configs["160m"][0]
    disguised = Path("configs/160m/checkpoint.pt")
    (input_root / disguised).write_bytes(original.read_bytes())
    configs["160m"][0] = disguised.as_posix()

    with pytest.raises(ValueError, match="YAML"):
        package_run(
            tmp_path / "checkpoint.tar.gz",
            source_root=REPO_ROOT,
            input_root=input_root,
            config_inputs=configs,
            manifest_inputs=manifests,
            route_policy="route-policy.json",
        )


def test_production_cli_rejects_a_dirty_source_tree(tmp_path):
    input_root, configs, manifests = _portable_inputs(tmp_path)
    with tempfile.NamedTemporaryFile(
        prefix=".task5-dirty-probe-",
        dir=REPO_ROOT,
        delete=False,
    ) as handle:
        dirty_probe = Path(handle.name)
    try:
        args = [
            "--out",
            str(tmp_path / "bundle.tar.gz"),
            "--source-root",
            str(REPO_ROOT),
            "--input-root",
            str(input_root),
            "--route-policy",
            "route-policy.json",
            "--manifest-160m",
            manifests["160m"],
            "--manifest-360m",
            manifests["360m"],
        ]
        for scale in ("160m", "360m"):
            for config in configs[scale]:
                args.extend([f"--config-{scale}", config])

        with pytest.raises(ValueError, match="clean source tree"):
            main(args)
    finally:
        dirty_probe.unlink(missing_ok=True)
