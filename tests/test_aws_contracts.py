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
