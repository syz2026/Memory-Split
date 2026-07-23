import hashlib
import json
from pathlib import Path

import pytest

from cluster.corpus_contract import (
    CorpusContractError,
    load_dataset_pointer,
    stage_dataset_no_replace,
    verify_canonical_corpus,
    verify_dataset_root,
)
from msctl.cohort import DATASET_CONTRACT_ID, RAW_TARGETS
from msctl.manifest import build_role_manifest, create_role_manifest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = ROOT / "configs" / "reasoning-dataset-v2.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_fixture(root: Path) -> tuple[Path, str, str]:
    (root / "packed").mkdir(parents=True)
    (root / "sidecars").mkdir()
    packed = b"\x01\x00\x02\x00\x03\x00"
    dense = b"\x01\x01\x01"
    split = b"\x01\x00\x01"
    semantic = {
        "contract_id": DATASET_CONTRACT_ID,
        "lane_ids": [
            "fineweb_edu",
            "finemath",
            "wikidata_graph",
            "srgm_worlds",
            "proof_composition",
            "graph_non_path",
            "counterfactual_pairs",
            "general_reasoning",
        ],
        "ordered_token_stream_sha256": _sha(packed),
        "packed_target_count": RAW_TARGETS,
        "passed": True,
        "schema_version": 1,
        "sidecar_alignment": {
            "dense_target_weights": RAW_TARGETS,
            "split90_target_weights": RAW_TARGETS,
        },
        "source_locks_verified": True,
        "wikidata_complete_once": True,
    }
    semantic_bytes = (
        json.dumps(semantic, indent=2, sort_keys=True) + "\n"
    ).encode()
    (root / "packed" / "targets.bin").write_bytes(packed)
    (root / "sidecars" / "dense_target_weights.bin").write_bytes(dense)
    (root / "sidecars" / "split90_target_weights.bin").write_bytes(split)
    (root / "semantic-verification.json").write_bytes(semantic_bytes)

    source = json.loads(SOURCE_LOCK.read_text())
    lanes = [
        {
            "id": lane["id"],
            "share_percent": lane["share_percent"],
            "source_lock_sha256": lane["source_lock_sha256"],
        }
        for lane in source["sprint_recipe"]["lanes"]
    ]
    receipt = {
        "complete": True,
        "contract_id": DATASET_CONTRACT_ID,
        "frozen": True,
        "lanes": lanes,
        "ordered_token_stream_sha256": _sha(packed),
        "production": True,
        "raw_target_tokens": RAW_TARGETS,
        "schema_version": 2,
        "semantic_verification": {
            "algorithm": "memorysplit-v2-semantic-v1",
            "evidence_path": "semantic-verification.json",
            "evidence_sha256": _sha(semantic_bytes),
            "passed": True,
        },
        "streams": {
            "dense_target_weights": {
                "path": "sidecars/dense_target_weights.bin",
                "sha256": _sha(dense),
                "targets": RAW_TARGETS,
            },
            "packed_targets": {
                "path": "packed/targets.bin",
                "sha256": _sha(packed),
                "targets": RAW_TARGETS,
            },
            "split90_target_weights": {
                "path": "sidecars/split90_target_weights.bin",
                "sha256": _sha(split),
                "targets": RAW_TARGETS,
            },
        },
        "wikidata": {
            "complete_once": True,
            "duplicate_entities": 0,
            "lane_id": "wikidata_graph",
        },
    }
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    receipt_path = root / "receipt.json"
    receipt_path.write_bytes(receipt_bytes)
    return receipt_path, _sha(receipt_bytes), _sha(packed)


def _write_pointer(
    path: Path,
    *,
    receipt_sha: str | None,
    ordered_sha: str | None,
) -> None:
    pointer = {
        "contract_id": DATASET_CONTRACT_ID,
        "dataset_id": "memorysplit-v2-20x-reasoning-max-cohort",
        "expected_ordered_token_stream_sha256": ordered_sha,
        "expected_receipt_sha256": receipt_sha,
        "launch_gate_status": "frozen" if receipt_sha and ordered_sha else "unfrozen",
        "materialization": "filesystem-mirror",
        "receipt_relative_path": "receipt.json",
        "relative_path": "dataset",
        "required_sidecars": [
            "dense_target_weights",
            "split90_target_weights",
        ],
        "schema_version": 1,
        "source_lock_manifest": "configs/reasoning-dataset-v2.json",
    }
    path.write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n")


def test_complete_v2_fixture_verifies(tmp_path):
    receipt, receipt_sha, ordered_sha = _write_fixture(tmp_path)
    evidence = verify_canonical_corpus(
        receipt,
        expected_sha256=receipt_sha,
        expected_ordered_sha256=ordered_sha,
        source_lock_path=SOURCE_LOCK,
    )
    assert evidence.contract_id == DATASET_CONTRACT_ID
    assert evidence.raw_target_tokens == RAW_TARGETS
    assert evidence.receipt_sha256 == receipt_sha
    assert evidence.ordered_token_stream_sha256 == ordered_sha
    assert evidence.semantic_verification_passed is True
    assert len(evidence.lane_ids) == 8


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("production", False, "production"),
        ("complete", False, "complete"),
        ("raw_target_tokens", 100, "target"),
        ("frozen", False, "frozen"),
    ],
)
def test_fixture_or_incomplete_receipt_cannot_satisfy_contract(
    tmp_path, field, value, match
):
    receipt, receipt_sha, ordered_sha = _write_fixture(tmp_path)
    raw = json.loads(receipt.read_text())
    raw[field] = value
    receipt.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    with pytest.raises(CorpusContractError, match=match):
        verify_canonical_corpus(
            receipt,
            expected_sha256=_sha(receipt.read_bytes()),
            expected_ordered_sha256=ordered_sha,
            source_lock_path=SOURCE_LOCK,
        )


def test_missing_lane_and_misaligned_sidecar_fail_closed(tmp_path):
    receipt, _, ordered_sha = _write_fixture(tmp_path)
    raw = json.loads(receipt.read_text())
    raw["lanes"].pop()
    raw["streams"]["split90_target_weights"]["targets"] -= 1
    receipt.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    with pytest.raises(CorpusContractError):
        verify_canonical_corpus(
            receipt,
            expected_sha256=_sha(receipt.read_bytes()),
            expected_ordered_sha256=ordered_sha,
            source_lock_path=SOURCE_LOCK,
        )


def test_unfrozen_checked_in_pointer_is_not_launchable():
    with pytest.raises(CorpusContractError, match="unfrozen"):
        load_dataset_pointer(ROOT / "DATASET-POINTER-SLURM-135M.json")


def test_pointer_verification_and_no_replace_staging(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _, receipt_sha, ordered_sha = _write_fixture(source)
    pointer = tmp_path / "pointer.json"
    _write_pointer(
        pointer,
        receipt_sha=receipt_sha,
        ordered_sha=ordered_sha,
    )
    verified = verify_dataset_root(
        source,
        pointer_path=pointer,
        source_lock_path=SOURCE_LOCK,
    )
    assert verified.receipt_sha256 == receipt_sha

    destination = tmp_path / "farmshare-mirror"
    staged = stage_dataset_no_replace(
        source,
        destination,
        pointer_path=pointer,
        source_lock_path=SOURCE_LOCK,
    )
    assert staged.receipt_sha256 == receipt_sha
    with pytest.raises(FileExistsError):
        stage_dataset_no_replace(
            source,
            destination,
            pointer_path=pointer,
            source_lock_path=SOURCE_LOCK,
        )


def test_symlinked_stream_is_rejected(tmp_path):
    receipt, receipt_sha, ordered_sha = _write_fixture(tmp_path)
    target = tmp_path / "packed" / "targets.bin"
    target.unlink()
    target.symlink_to(tmp_path / "sidecars" / "dense_target_weights.bin")
    with pytest.raises(CorpusContractError, match="symlink"):
        verify_canonical_corpus(
            receipt,
            expected_sha256=receipt_sha,
            expected_ordered_sha256=ordered_sha,
            source_lock_path=SOURCE_LOCK,
        )


def test_role_manifest_requires_verified_dataset_and_is_no_replace(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _, receipt_sha, ordered_sha = _write_fixture(dataset)
    pointer = tmp_path / "pointer.json"
    _write_pointer(
        pointer,
        receipt_sha=receipt_sha,
        ordered_sha=ordered_sha,
    )
    manifest = build_role_manifest(
        "farmshare-lead",
        dataset_root=dataset,
        pointer_path=pointer,
        source_lock_path=SOURCE_LOCK,
        repository_root=ROOT,
    )
    assert [(item["arm"], item["seed"]) for item in manifest["configs"]] == [
        ("dense", 0),
        ("split90", 0),
        ("dense", 5),
        ("split90", 5),
    ]
    assert manifest["dataset"]["receipt_sha256"] == receipt_sha
    output = tmp_path / "role-manifest.json"
    create_role_manifest(
        "farmshare-lead",
        output,
        dataset_root=dataset,
        pointer_path=pointer,
        source_lock_path=SOURCE_LOCK,
        repository_root=ROOT,
    )
    with pytest.raises(FileExistsError):
        create_role_manifest(
            "farmshare-lead",
            output,
            dataset_root=dataset,
            pointer_path=pointer,
            source_lock_path=SOURCE_LOCK,
            repository_root=ROOT,
        )
