from __future__ import annotations

import importlib
import importlib.util

import pytest


def _contracts():
    spec = importlib.util.find_spec("msctl.aws_contracts")
    assert spec is not None, "canonical AWS contract module is missing"
    return importlib.import_module("msctl.aws_contracts")


def test_v3_aws_contract_constants_are_exact_and_immutable_values():
    contract = _contracts()

    assert contract.PROVIDER == "aws-p5.48xlarge"
    assert contract.COHORT_ID == "memorysplit-confirmatory-v3-360m-n10-aws"
    assert contract.PREREGISTRATION_ID == "memorysplit-confirmatory-v3"
    assert contract.PREREGISTRATION_PATH == "configs/preregistration-v3.yaml"
    assert contract.COHORT_ASSIGNMENT_PATH == "configs/cohort-assignment-v3.json"
    assert contract.CONFIG_ROOT == "configs/360m-v3"
    assert contract.PROFILE_PATH == "cluster/profiles/aws-p5.48xlarge-v3.json"
    assert contract.DATASET_POINTER_PATH == "DATASET-POINTER-AWS.json"
    assert contract.DATASET_RECEIPT_PATH == "dataset/receipt.json"
    assert contract.SEEDS == tuple(range(10))
    assert contract.ARMS == ("dense", "split90")
    assert contract.SNAPSHOT_STEPS == (1_358, 3_396, 6_791, 10_187, 13_582)
    assert contract.PACKAGE_FORMAT_VERSION == 2
    assert type(contract.PACKAGE_FORMAT_VERSION) is int
    assert all(
        isinstance(value, tuple)
        for value in (contract.SEEDS, contract.ARMS, contract.SNAPSHOT_STEPS)
    )


def test_expected_config_paths_are_the_exact_twenty_v3_pairs():
    contract = _contracts()
    expected = tuple(
        f"configs/360m-v3/{arm}-s{seed}.yaml"
        for seed in range(10)
        for arm in ("dense", "split90")
    )

    paths = contract.expected_config_paths()

    assert paths == expected
    assert len(paths) == len(set(paths)) == 20
    assert not any("360m-v2" in path or "illumina" in path for path in paths)


def test_sha256_validation_and_content_addressed_keys_are_exact():
    contract = _contracts()
    digest = "0123456789abcdef" * 4

    assert contract.validate_sha256(digest) == digest
    assert contract.release_archive_key(digest) == f"releases/{digest}/archive.zip"
    assert contract.release_receipt_key(digest) == (
        f"releases/{digest}/release.json"
    )
    assert contract.release_checksum_key(digest) == (
        f"releases/{digest}/archive.sha256"
    )
    assert contract.dataset_receipt_key() == "dataset/receipt.json"


@pytest.mark.parametrize(
    "value",
    [
        "A" * 64,
        "a" * 63,
        "a" * 65,
        "g" * 64,
        "../" + "a" * 64,
        "a" * 32 + "/" + "a" * 31,
        "sha256:" + "a" * 64,
        "a" * 64 + "/archive.zip",
        " a" * 32,
        b"a" * 64,
        None,
    ],
    ids=[
        "uppercase",
        "short",
        "long",
        "non-hex",
        "parent-path",
        "slash",
        "algorithm-prefix",
        "path-suffix",
        "whitespace",
        "bytes",
        "none",
    ],
)
def test_sha256_validation_rejects_malformed_or_path_like_values(value):
    contract = _contracts()

    with pytest.raises(ValueError, match="SHA-256"):
        contract.validate_sha256(value)

    for derive in (
        contract.release_archive_key,
        contract.release_receipt_key,
        contract.release_checksum_key,
    ):
        with pytest.raises(ValueError, match="SHA-256"):
            derive(value)


def test_snapshot_object_key_has_exactly_one_live_unpadded_definition():
    import inspect

    contract = _contracts()
    source = inspect.getsource(contract)

    assert source.count("def snapshot_object_key(") == 1
    assert "step-{optimizer_step:07d}" not in source
    assert contract.snapshot_object_key(0, "dense", 1_358, "a" * 64) == (
        f"snapshots/seed-0/dense/step-1358/sha256/{'a' * 64}.pt"
    )


def test_run_finalization_keys_are_exact_for_every_seed_arm_and_step():
    contract = _contracts()
    digest = "0123456789abcdef" * 4

    for seed in contract.SEEDS:
        for arm in contract.ARMS:
            for step in contract.SNAPSHOT_STEPS:
                assert contract.snapshot_object_key(seed, arm, step, digest) == (
                    f"snapshots/seed-{seed}/{arm}/step-{step}"
                    f"/sha256/{digest}.pt"
                )
            assert contract.log_object_key(seed, arm, digest) == (
                f"logs/seed-{seed}/{arm}/sha256/{digest}.jsonl"
            )
        assert contract.run_receipt_key(seed, digest) == (
            f"receipts/runs/seed-{seed}/sha256/{digest}.json"
        )
    assert "snapshot_object_key" in contract.__all__
    assert "log_object_key" in contract.__all__
    assert "run_receipt_key" in contract.__all__


@pytest.mark.parametrize(
    ("seed", "arm", "step"),
    [
        (10, "dense", 1_358),
        (-1, "dense", 1_358),
        (True, "dense", 1_358),
        (1.0, "dense", 1_358),
        ("1", "dense", 1_358),
        (None, "dense", 1_358),
        (1, "Dense", 1_358),
        (1, "split", 1_358),
        (1, None, 1_358),
        (1, "dense", 1_359),
        (1, "dense", 0),
        (1, "dense", -1_358),
        (1, "dense", True),
        (1, "dense", "1358"),
        (1, "dense", 13_582.0),
    ],
    ids=[
        "seed-ten",
        "seed-negative",
        "seed-bool",
        "seed-float",
        "seed-text",
        "seed-none",
        "arm-case",
        "arm-foreign",
        "arm-none",
        "step-off-schedule",
        "step-zero",
        "step-negative",
        "step-bool",
        "step-text",
        "step-float",
    ],
)
def test_run_finalization_keys_reject_foreign_seed_arm_or_step(seed, arm, step):
    contract = _contracts()
    digest = "a" * 64

    with pytest.raises(ValueError):
        contract.snapshot_object_key(seed, arm, step, digest)
    if not (type(seed) is int and seed in contract.SEEDS and arm in contract.ARMS):
        with pytest.raises(ValueError):
            contract.log_object_key(seed, arm, digest)
    if not (type(seed) is int and seed in contract.SEEDS):
        with pytest.raises(ValueError):
            contract.run_receipt_key(seed, digest)


@pytest.mark.parametrize(
    "value",
    ["A" * 64, "a" * 63, "g" * 64, "sha256:" + "a" * 64, b"a" * 64, None],
    ids=["uppercase", "short", "non-hex", "algorithm-prefix", "bytes", "none"],
)
def test_run_finalization_keys_reject_malformed_sha256(value):
    contract = _contracts()

    with pytest.raises(ValueError, match="SHA-256"):
        contract.snapshot_object_key(0, "dense", 1_358, value)
    with pytest.raises(ValueError, match="SHA-256"):
        contract.log_object_key(0, "dense", value)
    with pytest.raises(ValueError, match="SHA-256"):
        contract.run_receipt_key(0, value)


def test_collection_receipt_key_is_exact_for_every_seed():
    contract = _contracts()
    digest = "0123456789abcdef" * 4

    for seed in contract.SEEDS:
        assert contract.collection_receipt_key(seed, digest) == (
            f"receipts/collections/seed-{seed}/sha256/{digest}.json"
        )
    assert "collection_receipt_key" in contract.__all__


@pytest.mark.parametrize(
    "seed",
    [10, -1, True, 1.0, "1", None],
    ids=["ten", "negative", "bool", "float", "text", "none"],
)
def test_collection_receipt_key_rejects_foreign_seeds(seed):
    contract = _contracts()

    with pytest.raises(ValueError, match="seed"):
        contract.collection_receipt_key(seed, "a" * 64)


@pytest.mark.parametrize(
    "value",
    ["A" * 64, "a" * 63, "g" * 64, "sha256:" + "a" * 64, b"a" * 64, None],
    ids=["uppercase", "short", "non-hex", "algorithm-prefix", "bytes", "none"],
)
def test_collection_receipt_key_rejects_malformed_sha256(value):
    contract = _contracts()

    with pytest.raises(ValueError, match="SHA-256"):
        contract.collection_receipt_key(0, value)


def test_bootstrap_receipt_key_is_exact_and_instance_scoped():
    contract = _contracts()

    assert contract.bootstrap_receipt_key("i-0123456789abcdef0") == (
        "receipts/bootstrap/i-0123456789abcdef0.json"
    )
    assert "bootstrap_receipt_key" in contract.__all__


@pytest.mark.parametrize(
    "value",
    [
        "i-XYZ",
        "i-0123456",
        "0123456789abcdef0",
        "i-0123456789abcdef0/../escape",
        "i-0123456789ABCDEF0",
        "",
        None,
        b"i-0123456789abcdef0",
        7,
    ],
    ids=[
        "non-hex",
        "short",
        "unprefixed",
        "path-traversal",
        "uppercase",
        "empty",
        "none",
        "bytes",
        "int",
    ],
)
def test_bootstrap_receipt_key_rejects_foreign_instance_ids(value):
    contract = _contracts()

    with pytest.raises(ValueError, match="instance"):
        contract.bootstrap_receipt_key(value)


def test_evaluation_output_members_are_the_closed_sorted_ten():
    contract = _contracts()

    assert contract.EVALUATION_OUTPUT_MEMBERS == (
        "inference.json",
        "items.jsonl",
        "metrics.json",
        "outcomes.jsonl",
        "output.json",
        "run.json",
        "sealed-gold.jsonl",
        "sealed-release.json",
        "stores.jsonl",
        "study-lock.json",
    )
    assert contract.EVALUATION_OUTPUT_MEMBERS == tuple(
        sorted(contract.EVALUATION_OUTPUT_MEMBERS)
    )
    assert "EVALUATION_OUTPUT_MEMBERS" in contract.__all__


def test_evaluation_output_members_mirror_aggregate_and_runner_registries():
    import evals.confirmatory.aggregate as aggregate_module
    import evals.confirmatory.runner as runner_module

    contract = _contracts()

    assert contract.EVALUATION_OUTPUT_MEMBERS == aggregate_module._OUTPUT_NAMES
    assert contract.EVALUATION_OUTPUT_MEMBERS == tuple(
        sorted((*runner_module._V3_OUTPUT_ARTIFACTS, "output.json"))
    )


def test_cohort_evaluation_object_keys_are_exact():
    contract = _contracts()
    digest = "0123456789abcdef" * 4

    assert contract.study_lock_object_key(digest) == (
        f"evaluations/study-lock/sha256/{digest}.json"
    )
    assert contract.cohort_report_object_key(digest) == (
        f"evaluations/cohort-report/sha256/{digest}.json"
    )
    assert contract.cohort_collection_receipt_key(digest) == (
        f"receipts/cohort-collections/sha256/{digest}.json"
    )
    for name in (
        "study_lock_object_key",
        "cohort_report_object_key",
        "cohort_collection_receipt_key",
        "evaluation_output_member_key",
    ):
        assert name in contract.__all__


def test_evaluation_output_member_key_splits_stem_at_the_final_dot():
    contract = _contracts()
    digest = "0123456789abcdef" * 4

    for seed in contract.SEEDS:
        for arm in contract.ARMS:
            for step in contract.SNAPSHOT_STEPS:
                for member in contract.EVALUATION_OUTPUT_MEMBERS:
                    stem, _, extension = member.rpartition(".")
                    assert contract.evaluation_output_member_key(
                        seed,
                        arm,
                        step,
                        member,
                        digest,
                    ) == (
                        f"evaluations/outputs/seed-{seed}/{arm}/step-{step}/"
                        f"{stem}/sha256/{digest}.{extension}"
                    )
    assert contract.evaluation_output_member_key(
        0,
        "dense",
        1_358,
        "sealed-gold.jsonl",
        digest,
    ) == (
        "evaluations/outputs/seed-0/dense/step-1358/sealed-gold/"
        f"sha256/{digest}.jsonl"
    )


@pytest.mark.parametrize(
    ("seed", "arm", "step", "member"),
    [
        (10, "dense", 1_358, "output.json"),
        (-1, "dense", 1_358, "output.json"),
        (True, "dense", 1_358, "output.json"),
        ("1", "dense", 1_358, "output.json"),
        (None, "dense", 1_358, "output.json"),
        (1, "Dense", 1_358, "output.json"),
        (1, "split", 1_358, "output.json"),
        (1, None, 1_358, "output.json"),
        (1, "dense", 1_359, "output.json"),
        (1, "dense", True, "output.json"),
        (1, "dense", "1358", "output.json"),
        (1, "dense", 1_358, "weights.pt"),
        (1, "dense", 1_358, "OUTPUT.JSON"),
        (1, "dense", 1_358, "output"),
        (1, "dense", 1_358, ""),
        (1, "dense", 1_358, None),
        (1, "dense", 1_358, b"output.json"),
    ],
    ids=[
        "seed-ten",
        "seed-negative",
        "seed-bool",
        "seed-text",
        "seed-none",
        "arm-case",
        "arm-foreign",
        "arm-none",
        "step-off-schedule",
        "step-bool",
        "step-text",
        "member-foreign",
        "member-case",
        "member-no-extension",
        "member-empty",
        "member-none",
        "member-bytes",
    ],
)
def test_evaluation_output_member_key_rejects_foreign_slots_or_members(
    seed,
    arm,
    step,
    member,
):
    contract = _contracts()

    with pytest.raises(ValueError):
        contract.evaluation_output_member_key(seed, arm, step, member, "a" * 64)


@pytest.mark.parametrize(
    "value",
    ["A" * 64, "a" * 63, "g" * 64, "sha256:" + "a" * 64, b"a" * 64, None],
    ids=["uppercase", "short", "non-hex", "algorithm-prefix", "bytes", "none"],
)
def test_cohort_evaluation_keys_reject_malformed_sha256(value):
    contract = _contracts()

    with pytest.raises(ValueError, match="SHA-256"):
        contract.study_lock_object_key(value)
    with pytest.raises(ValueError, match="SHA-256"):
        contract.cohort_report_object_key(value)
    with pytest.raises(ValueError, match="SHA-256"):
        contract.cohort_collection_receipt_key(value)
    with pytest.raises(ValueError, match="SHA-256"):
        contract.evaluation_output_member_key(
            0,
            "dense",
            1_358,
            "output.json",
            value,
        )
