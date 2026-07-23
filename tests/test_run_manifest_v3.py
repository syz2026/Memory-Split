from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

from cluster.aws.p5.corpus_contract import (
    CorpusEvidence,
    PinnedCorpusFile,
    verify_canonical_corpus,
)
from cluster.aws.p5.profile import load_aws_p5_profile
from msctl.aws_contracts import (
    COHORT_ASSIGNMENT_PATH,
    COHORT_ID,
    CONFIG_ROOT,
    DATASET_POINTER_PATH,
    PREREGISTRATION_PATH,
    PROFILE_PATH,
)
from msctl.cohort import load_cohort_assignment
from msctl.contracts import bind_release, load_release, load_run_manifest
from msctl.errors import MsctlError
from msctl.operations import instantiate_run_manifest
from tests.test_aws_p5_launcher import _launcher_fixture
from tests.test_package_aws_p5_handoff import (
    _build,
    _commit,
    _git,
    _load_module,
    _minimal_repo,
)


ROOT = Path(__file__).resolve().parents[1]
SEALED_EVALUATION_RELEASE_SHA256 = "e" * 64
ESTIMATED_INSTANCE_HOURS = 2.5
V3_FIELDS = {
    "schema_version",
    "provider",
    "cohort_id",
    "seed",
    "release_sha256",
    "release_receipt_sha256",
    "profile_sha256",
    "dataset_pointer_sha256",
    "dataset_receipt_sha256",
    "dataset_build_id",
    "ordered_stream_sha256",
    "cohort_assignment_sha256",
    "preregistration_sha256",
    "sealed_evaluation_release_sha256",
    "source_commit",
    "source_tree",
    "runs",
}
V3_RUN_FIELDS = {
    "run_id",
    "arm",
    "seed",
    "config",
    "config_sha256",
    "estimated_gpu_hours",
}
V3_SHA256_FIELDS = {
    "release_sha256",
    "release_receipt_sha256",
    "profile_sha256",
    "dataset_pointer_sha256",
    "dataset_receipt_sha256",
    "dataset_build_id",
    "ordered_stream_sha256",
    "cohort_assignment_sha256",
    "preregistration_sha256",
    "sealed_evaluation_release_sha256",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")
    return path


def _different_sha256(value: str) -> str:
    candidate = "0" * 64
    return "1" * 64 if value == candidate else candidate


def _copy_real_v3_contract(source: Path) -> None:
    members = [
        COHORT_ASSIGNMENT_PATH,
        PREREGISTRATION_PATH,
        PROFILE_PATH,
        DATASET_POINTER_PATH,
        *(
            f"{CONFIG_ROOT}/{arm}-s{seed}.yaml"
            for seed in range(10)
            for arm in ("dense", "split90")
        ),
    ]
    for relative in members:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    _commit(source, "use canonical AWS v3 scientific contract")


def _build_default_verified_corpus(root: Path) -> dict[str, object]:
    from corpusgen.parallel import (
        FixtureRenderer,
        ParallelBuildConfig,
        build_parallel_corpus,
        fixture_catalog,
        render_metadata,
    )

    catalog = fixture_catalog(record_count=6)
    renderer = FixtureRenderer()
    logical_tokens = sum(
        record.token_length for record in render_metadata(catalog, renderer)
    )
    sidecar_root = root.parent / f"{root.name}-sidecar-inputs"
    sidecar_root.mkdir(parents=True)
    dense = sidecar_root / "dense_target_weights.bin"
    split90 = sidecar_root / "split90_target_weights.bin"
    dense.write_bytes(b"\x01" * logical_tokens)
    split90.write_bytes(
        bytes(
            0 if index % 10 == 0 else 1
            for index in range(logical_tokens)
        )
    )
    receipt = build_parallel_corpus(
        catalog,
        renderer,
        ParallelBuildConfig(
            lane_weights=(
                ("natural", 1),
                ("facts", 1),
                ("reasoning", 1),
            ),
            update_tokens=64,
            allow_fewer_shards=True,
        ),
        root,
        sidecar_paths={
            "dense_target_weights": dense,
            "split90_target_weights": split90,
        },
    )
    return {
        "corpus_path": root / "receipt.json",
        "corpus": receipt,
    }


@pytest.fixture()
def v3_case(tmp_path: Path) -> dict[str, object]:
    source = _minimal_repo(tmp_path, name="release-source")
    _copy_real_v3_contract(source)
    packaged = _build(_load_module(), source, tmp_path / "published")
    dataset = _launcher_fixture(tmp_path / "task4", seed=0)
    profile = load_aws_p5_profile(source / PROFILE_PATH)

    def verifier(
        receipt_path: Path,
        *,
        expected_sha256: str,
        expected_ordered_sha256: str,
    ):
        return verify_canonical_corpus(
            receipt_path,
            expected_sha256=expected_sha256,
            expected_ordered_sha256=expected_ordered_sha256,
            semantic_verifier=lambda _root: dataset["corpus"],
        )

    return {
        "source": source,
        "packaged": packaged,
        "dataset": dataset,
        "profile": profile,
        "verifier": verifier,
    }


def _expected_manifest(
    case: dict[str, object],
    seed: int,
    *,
    estimated_instance_hours: float = ESTIMATED_INSTANCE_HOURS,
) -> dict[str, object]:
    source = Path(case["source"])
    packaged = case["packaged"]
    dataset = case["dataset"]
    profile = case["profile"]
    release = load_release(packaged.release)
    cohort = load_cohort_assignment(source / COHORT_ASSIGNMENT_PATH)
    receipt_path = Path(dataset["corpus_path"])
    receipt = dataset["corpus"]
    return {
        "schema_version": 3,
        "provider": "aws-p5.48xlarge",
        "cohort_id": COHORT_ID,
        "seed": seed,
        "release_sha256": release.archive_sha256,
        "release_receipt_sha256": release.receipt_sha256,
        "profile_sha256": profile.sha256,
        "dataset_pointer_sha256": _sha256(source / DATASET_POINTER_PATH),
        "dataset_receipt_sha256": _sha256(receipt_path),
        "dataset_build_id": receipt["build_id"],
        "ordered_stream_sha256": receipt["ordered_stream_sha256"],
        "cohort_assignment_sha256": cohort.assignment_sha256,
        "preregistration_sha256": cohort.preregistration_sha256,
        "sealed_evaluation_release_sha256": (
            SEALED_EVALUATION_RELEASE_SHA256
        ),
        "source_commit": _git(source, "rev-parse", "HEAD"),
        "source_tree": _git(source, "rev-parse", "HEAD^{tree}"),
        "runs": [
            {
                "run_id": f"memorysplit-v3-360m-s{seed}-{arm}",
                "arm": arm,
                "seed": seed,
                "config": f"{CONFIG_ROOT}/{arm}-s{seed}.yaml",
                "config_sha256": _sha256(
                    source / CONFIG_ROOT / f"{arm}-s{seed}.yaml"
                ),
                "estimated_gpu_hours": 4 * estimated_instance_hours,
            }
            for arm in ("dense", "split90")
        ],
    }


def _instantiate(
    case: dict[str, object],
    *,
    seed: int,
    out: Path,
    apply: bool = False,
    sealed_evaluation_release_sha256: str | None = (
        SEALED_EVALUATION_RELEASE_SHA256
    ),
    estimated_instance_hours: object = ESTIMATED_INSTANCE_HOURS,
    dataset_verifier=None,
) -> dict[str, object]:
    packaged = case["packaged"]
    dataset = case["dataset"]
    return instantiate_run_manifest(
        profile=case["profile"],
        release_path=packaged.release,
        dataset_receipt=dataset["corpus_path"],
        seed=seed,
        out=out,
        repo_root=case["source"],
        apply=apply,
        sealed_evaluation_release_sha256=sealed_evaluation_release_sha256,
        estimated_instance_hours=estimated_instance_hours,
        dataset_verifier=(
            case["verifier"]
            if dataset_verifier is None
            else dataset_verifier
        ),
    )


def test_real_v3_release_instantiates_exact_seed_zero_and_nine_manifests(
    v3_case,
    tmp_path,
):
    for seed in (0, 9):
        out = tmp_path / f"runs-s{seed}.json"

        result = _instantiate(v3_case, seed=seed, out=out)

        expected = _expected_manifest(v3_case, seed)
        assert result["manifest"] == expected
        assert set(result["manifest"]) == V3_FIELDS
        assert all(set(run) == V3_RUN_FIELDS for run in result["manifest"]["runs"])
        assert "dataset_sha256" not in result["manifest"]
        assert "study_lock_sha256" not in result["manifest"]
        assert result["manifest_sha256"] == hashlib.sha256(
            _canonical(expected)
        ).hexdigest()
        assert result["published"] is False
        assert not out.exists()
        verification = result["dataset_verification"]
        assert verification["receipt_sha256"] == expected[
            "dataset_receipt_sha256"
        ]
        assert verification["build_id"] == expected["dataset_build_id"]
        assert verification["ordered_stream_sha256"] == expected[
            "ordered_stream_sha256"
        ]
        assert len(verification["file_identities"]) > 5


def test_v3_instantiates_through_unmodified_default_corpus_verifier(
    v3_case,
    tmp_path,
):
    packaged = v3_case["packaged"]
    dataset = _build_default_verified_corpus(
        tmp_path / "default-verified-corpus"
    )

    result = instantiate_run_manifest(
        profile=v3_case["profile"],
        release_path=packaged.release,
        dataset_receipt=dataset["corpus_path"],
        seed=0,
        out=tmp_path / "default-verifier.json",
        repo_root=v3_case["source"],
        apply=False,
        sealed_evaluation_release_sha256=(
            SEALED_EVALUATION_RELEASE_SHA256
        ),
        estimated_instance_hours=ESTIMATED_INSTANCE_HOURS,
    )

    assert result["manifest"]["dataset_receipt_sha256"] == _sha256(
        Path(dataset["corpus_path"])
    )
    assert result["dataset_verification"]["build_id"] == dataset["corpus"][
        "build_id"
    ]


def test_v3_load_reload_binding_and_no_replace_preserve_every_identity(
    v3_case,
    tmp_path,
):
    out = tmp_path / "runs-s0.json"
    expected = _expected_manifest(v3_case, 0)

    result = _instantiate(v3_case, seed=0, out=out, apply=True)

    assert result["published"] is True
    assert out.read_bytes() == _canonical(expected) + b"\n"
    manifest = load_run_manifest(out, repo_root=v3_case["source"])
    release = load_release(v3_case["packaged"].release)
    bind_release(release, manifest)
    assert manifest.value == expected
    assert manifest.schema_version == 3
    assert manifest.cohort_id == COHORT_ID
    assert manifest.dataset_receipt_sha256 == expected[
        "dataset_receipt_sha256"
    ]
    assert manifest.dataset_build_id == expected["dataset_build_id"]
    assert manifest.ordered_stream_sha256 == expected["ordered_stream_sha256"]
    assert manifest.gpu_hours == 8 * ESTIMATED_INSTANCE_HOURS
    assert not hasattr(manifest, "dataset_sha256")
    assert not hasattr(manifest, "study_lock_sha256")

    reloaded_path = _write_json(tmp_path / "reloaded.json", manifest.value)
    reloaded = load_run_manifest(reloaded_path, repo_root=v3_case["source"])
    bind_release(release, reloaded)
    assert reloaded == manifest

    original = out.read_bytes()
    with pytest.raises(MsctlError) as caught:
        _instantiate(v3_case, seed=0, out=out, apply=True)
    assert caught.value.code == "RUN_MANIFEST_EXISTS"
    assert out.read_bytes() == original


def test_v3_missing_sealed_evaluation_hash_blocks_publication(v3_case, tmp_path):
    out = tmp_path / "runs-s0.json"

    with pytest.raises(MsctlError) as caught:
        _instantiate(
            v3_case,
            seed=0,
            out=out,
            apply=True,
            sealed_evaluation_release_sha256=None,
        )

    assert caught.value.code == "RUN_MANIFEST_INVALID"
    assert not out.exists()


@pytest.mark.parametrize(
    "estimated_instance_hours",
    [
        None,
        True,
        "2.5",
        0,
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
        1e308,
        pytest.param(10**1000, id="overflowing-int"),
    ],
)
def test_v3_rejects_invalid_estimated_instance_hours(
    v3_case,
    tmp_path,
    estimated_instance_hours,
):
    out = tmp_path / "invalid-instance-hours.json"

    with pytest.raises(MsctlError) as caught:
        _instantiate(
            v3_case,
            seed=0,
            out=out,
            apply=True,
            estimated_instance_hours=estimated_instance_hours,
        )

    assert caught.value.code == "RUN_MANIFEST_INVALID"
    assert not out.exists()


def test_v3_cli_requires_and_forwards_instantiation_inputs(
    v3_case,
    tmp_path,
    monkeypatch,
):
    import msctl.operations as operations
    from msctl.cli import build_parser, dispatch

    monkeypatch.setattr(
        operations,
        "_load_task4_dataset_verifier",
        v3_case["verifier"],
    )
    packaged = v3_case["packaged"]
    dataset = v3_case["dataset"]
    common = [
        "--profile",
        str(Path(v3_case["source"]) / PROFILE_PATH),
        "--repo-root",
        str(v3_case["source"]),
        "runs",
        "instantiate",
        "--release",
        str(packaged.release),
        "--dataset-receipt",
        str(dataset["corpus_path"]),
        "--seed",
        "0",
        "--out",
        str(tmp_path / "cli-runs.json"),
    ]
    missing = build_parser().parse_args(common)

    with pytest.raises(MsctlError) as caught:
        dispatch(missing)
    assert caught.value.code == "CLI_USAGE"
    assert set(caught.value.details["missing"]) == {
        "--estimated-instance-hours",
        "--sealed-evaluation-release-sha256",
    }
    assert not (tmp_path / "cli-runs.json").exists()

    missing_hours = build_parser().parse_args(
        [
            *common,
            "--sealed-evaluation-release-sha256",
            SEALED_EVALUATION_RELEASE_SHA256,
        ]
    )
    with pytest.raises(MsctlError) as caught:
        dispatch(missing_hours)
    assert caught.value.code == "CLI_USAGE"
    assert caught.value.details["missing"] == ["--estimated-instance-hours"]

    supplied = build_parser().parse_args(
        [
            *common,
            "--sealed-evaluation-release-sha256",
            SEALED_EVALUATION_RELEASE_SHA256,
            "--estimated-instance-hours",
            str(ESTIMATED_INSTANCE_HOURS),
        ]
    )
    dry_run, result = dispatch(supplied)

    assert dry_run is True
    assert result["manifest"]["sealed_evaluation_release_sha256"] == (
        SEALED_EVALUATION_RELEASE_SHA256
    )
    assert {
        run["estimated_gpu_hours"] for run in result["manifest"]["runs"]
    } == {4 * ESTIMATED_INSTANCE_HOURS}
    assert result["published"] is False


def test_v3_cli_profile_support_is_limited_to_manifest_instantiation(
    v3_case,
):
    from msctl.cli import build_parser, dispatch

    args = build_parser().parse_args(
        [
            "--profile",
            str(Path(v3_case["source"]) / PROFILE_PATH),
            "capacity",
            "check",
        ]
    )

    with pytest.raises(MsctlError) as caught:
        dispatch(args)

    assert caught.value.code == "PROFILE_INVALID"


@pytest.mark.parametrize(
    "mutation",
    ["receipt-bytes", "build-id", "ordered-stream"],
)
def test_v3_instantiation_rejects_dataset_identity_mutations(
    v3_case,
    tmp_path,
    mutation,
):
    dataset = v3_case["dataset"]
    receipt_path = Path(dataset["corpus_path"])
    if mutation == "receipt-bytes":
        receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
    else:
        receipt = json.loads(receipt_path.read_text())
        field = (
            "build_id"
            if mutation == "build-id"
            else "ordered_stream_sha256"
        )
        receipt[field] = _different_sha256(receipt[field])
        _write_json(receipt_path, receipt)
    out = tmp_path / "runs-s0.json"

    with pytest.raises(MsctlError) as caught:
        _instantiate(v3_case, seed=0, out=out, apply=True)

    assert caught.value.code == "DATASET_RECEIPT_INVALID"
    assert not out.exists()


def test_v3_accepts_reordered_pins_from_real_canonical_verifier(
    v3_case,
    tmp_path,
):
    dataset = v3_case["dataset"]
    calls = 0

    def verifier(
        receipt_path: Path,
        *,
        expected_sha256: str,
        expected_ordered_sha256: str,
    ):
        nonlocal calls
        calls += 1
        evidence = verify_canonical_corpus(
            receipt_path,
            expected_sha256=expected_sha256,
            expected_ordered_sha256=expected_ordered_sha256,
            semantic_verifier=lambda _root: dataset["corpus"],
        )
        return CorpusEvidence(
            receipt=evidence.receipt,
            files=tuple(reversed(evidence.files)),
        )

    result = _instantiate(
        v3_case,
        seed=0,
        out=tmp_path / "reordered-pins.json",
        dataset_verifier=verifier,
    )

    assert calls == 1
    assert result["manifest"]["dataset_receipt_sha256"] == _sha256(
        Path(dataset["corpus_path"])
    )


@pytest.mark.parametrize(
    "fault",
    [
        "different-content",
        "numeric-alias",
        "non-object-content",
        "missing-receipt",
        "other-receipt",
        "wrong-receipt-hash",
    ],
)
def test_v3_rejects_faulty_dataset_verifier_evidence(
    v3_case,
    tmp_path,
    fault,
):
    dataset = v3_case["dataset"]
    requested_receipt = Path(dataset["corpus_path"]).resolve()

    def verifier(
        receipt_path: Path,
        *,
        expected_sha256: str,
        expected_ordered_sha256: str,
    ):
        evidence = v3_case["verifier"](
            receipt_path,
            expected_sha256=expected_sha256,
            expected_ordered_sha256=expected_ordered_sha256,
        )
        receipt = dict(evidence.receipt)
        files = list(evidence.files)
        receipt_index = next(
            index
            for index, pinned in enumerate(files)
            if Path(pinned.path).resolve() == requested_receipt
        )
        if fault == "different-content":
            receipt["build_id"] = _different_sha256(receipt["build_id"])
        elif fault == "numeric-alias":
            receipt["logical_tokens"] = float(receipt["logical_tokens"])
        elif fault == "non-object-content":
            receipt = []
        elif fault == "missing-receipt":
            files.pop(receipt_index)
        elif fault == "other-receipt":
            alternate = requested_receipt.with_name("alternate-receipt.json")
            alternate.write_bytes(requested_receipt.read_bytes())
            files[receipt_index] = PinnedCorpusFile(
                path=alternate,
                sha256=expected_sha256,
            )
        else:
            files[receipt_index] = PinnedCorpusFile(
                path=requested_receipt,
                sha256=_different_sha256(expected_sha256),
            )
        return CorpusEvidence(receipt=receipt, files=tuple(files))

    out = tmp_path / f"faulty-{fault}.json"
    with pytest.raises(MsctlError) as caught:
        _instantiate(
            v3_case,
            seed=0,
            out=out,
            apply=True,
            dataset_verifier=verifier,
        )

    assert caught.value.code == "DATASET_RECEIPT_INVALID"
    assert not out.exists()


@pytest.mark.parametrize(
    "field",
    [
        "release_sha256",
        "release_receipt_sha256",
        "profile_sha256",
        "dataset_pointer_sha256",
        "cohort_assignment_sha256",
        "preregistration_sha256",
        "source_commit",
        "source_tree",
    ],
)
def test_v3_binding_rejects_release_identity_mutations(
    v3_case,
    tmp_path,
    field,
):
    value = _expected_manifest(v3_case, 0)
    width = 40 if field in {"source_commit", "source_tree"} else 64
    value[field] = "0" * width
    path = _write_json(tmp_path / f"mutated-{field}.json", value)
    manifest = load_run_manifest(path, repo_root=v3_case["source"])

    with pytest.raises(MsctlError) as caught:
        bind_release(load_release(v3_case["packaged"].release), manifest)

    assert caught.value.code == "RELEASE_RUN_MISMATCH"


@pytest.mark.parametrize("field", sorted(V3_SHA256_FIELDS))
def test_v3_loader_rejects_malformed_sha256_fields(
    v3_case,
    tmp_path,
    field,
):
    value = _expected_manifest(v3_case, 0)
    value[field] = "A" * 64
    path = _write_json(tmp_path / f"malformed-{field}.json", value)

    with pytest.raises(MsctlError):
        load_run_manifest(path, repo_root=v3_case["source"])


@pytest.mark.parametrize(
    ("target", "numeric_alias"),
    [
        ("schema_version", 3.0),
        ("seed", False),
        ("seed", 0.0),
        ("run_seed", False),
        ("run_seed", 0.0),
    ],
)
def test_v3_loader_rejects_schema_and_seed_numeric_aliases(
    v3_case,
    tmp_path,
    target,
    numeric_alias,
):
    value = _expected_manifest(v3_case, 0)
    if target == "run_seed":
        value["runs"][0]["seed"] = numeric_alias
    else:
        value[target] = numeric_alias
    path = _write_json(
        tmp_path / f"{target}-{type(numeric_alias).__name__}.json",
        value,
    )

    with pytest.raises(MsctlError):
        load_run_manifest(path, repo_root=v3_case["source"])


@pytest.mark.parametrize(
    "estimated_gpu_hours",
    [True, 0, -1, float("nan"), float("inf"), float("-inf")],
)
def test_v3_loader_rejects_nonpositive_or_nonfinite_gpu_hours(
    v3_case,
    tmp_path,
    estimated_gpu_hours,
):
    value = _expected_manifest(v3_case, 0)
    value["runs"][0]["estimated_gpu_hours"] = estimated_gpu_hours
    path = tmp_path / "invalid-gpu-hours.json"
    path.write_text(json.dumps(value, allow_nan=True) + "\n")

    with pytest.raises(MsctlError):
        load_run_manifest(path, repo_root=v3_case["source"])


@pytest.mark.parametrize(
    ("mutation", "target"),
    [
        ("missing", "root"),
        ("unknown", "root"),
        ("missing", "run"),
        ("unknown", "run"),
    ],
)
def test_v3_loader_rejects_open_or_incomplete_objects(
    v3_case,
    tmp_path,
    mutation,
    target,
):
    value = _expected_manifest(v3_case, 0)
    selected = value if target == "root" else value["runs"][0]
    if mutation == "missing":
        selected.pop(
            "dataset_build_id" if target == "root" else "estimated_gpu_hours"
        )
    else:
        selected["unexpected"] = "forbidden"
    path = _write_json(tmp_path / f"{target}-{mutation}.json", value)

    with pytest.raises(MsctlError):
        load_run_manifest(path, repo_root=v3_case["source"])


@pytest.mark.parametrize("target", ["provider", "arm"])
def test_v3_loader_rejects_nonscalar_enum_fields(v3_case, tmp_path, target):
    value = _expected_manifest(v3_case, 0)
    if target == "provider":
        value["provider"] = []
    else:
        value["runs"][0]["arm"] = []
    path = _write_json(tmp_path / f"nonscalar-{target}.json", value)

    with pytest.raises(MsctlError):
        load_run_manifest(path, repo_root=v3_case["source"])


def test_v3_loader_rejects_duplicate_and_nonfinite_json(v3_case, tmp_path):
    value = _expected_manifest(v3_case, 0)
    canonical = _canonical(value).decode("ascii")
    duplicate = (
        '{"schema_version":3,' + canonical.removeprefix("{")
    )
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(duplicate + "\n")
    nonfinite = deepcopy(value)
    nonfinite["runs"][0]["estimated_gpu_hours"] = float("nan")
    nonfinite_path = tmp_path / "nonfinite.json"
    nonfinite_path.write_text(json.dumps(nonfinite, allow_nan=True) + "\n")

    for path in (duplicate_path, nonfinite_path):
        with pytest.raises(MsctlError):
            load_run_manifest(path, repo_root=v3_case["source"])


@pytest.mark.parametrize(
    "mutation",
    [
        "seed-10",
        "partial-pair",
        "duplicate-arm",
        "v2-run-id",
        "v2-config",
        "wrong-config-hash",
        "wrong-cohort",
    ],
)
def test_v3_loader_rejects_seed_pair_and_cross_version_mutations(
    v3_case,
    tmp_path,
    mutation,
):
    value = _expected_manifest(v3_case, 0)
    source = Path(v3_case["source"])
    if mutation == "seed-10":
        value["seed"] = 10
        for run in value["runs"]:
            run["seed"] = 10
    elif mutation == "partial-pair":
        value["runs"].pop()
    elif mutation == "duplicate-arm":
        value["runs"][1] = deepcopy(value["runs"][0])
    elif mutation == "v2-run-id":
        value["runs"][0]["run_id"] = "memorysplit-v2-360m-s0-dense"
    elif mutation == "v2-config":
        relative = "configs/360m-v2/dense-s0.yaml"
        value["runs"][0]["config"] = relative
        value["runs"][0]["config_sha256"] = _sha256(source / relative)
    elif mutation == "wrong-config-hash":
        value["runs"][0]["config_sha256"] = _different_sha256(
            value["runs"][0]["config_sha256"]
        )
    else:
        value["cohort_id"] = "memorysplit-confirmatory-v2-360m-n5"
    path = _write_json(tmp_path / f"{mutation}.json", value)

    with pytest.raises(MsctlError):
        load_run_manifest(path, repo_root=source)


def _legacy_schema2_manifest(root: Path, release) -> dict[str, object]:
    seed = 1
    return {
        "schema_version": 2,
        "provider": "aws-p5.48xlarge",
        "seed": seed,
        "release_sha256": release.archive_sha256,
        "dataset_sha256": "d" * 64,
        "cohort_assignment_sha256": _sha256(
            root / "configs" / "cohort-assignment-v2.json"
        ),
        "study_lock_sha256": _sha256(
            root / "configs" / "preregistration-v2.yaml"
        ),
        "source_commit": release.source_commit,
        "runs": [
            {
                "run_id": f"memorysplit-v2-360m-s{seed}-{arm}",
                "arm": arm,
                "seed": seed,
                "config": f"configs/360m-v2/{arm}-s{seed}.yaml",
                "config_sha256": _sha256(
                    root / "configs" / "360m-v2" / f"{arm}-s{seed}.yaml"
                ),
            }
            for arm in ("dense", "split90")
        ],
    }


def test_schema2_legacy_loading_and_binding_remain_explicit(tmp_path):
    from tests.test_msctl import _cohort_release

    legacy_root = tmp_path / "legacy"
    release = load_release(
        _cohort_release(legacy_root, "aws-p5.48xlarge")
    )
    value = _legacy_schema2_manifest(legacy_root, release)
    path = _write_json(legacy_root / "runs-s1.json", value)

    manifest = load_run_manifest(path, repo_root=legacy_root)
    bind_release(release, manifest)

    assert manifest.schema_version == 2
    assert manifest.dataset_sha256 == "d" * 64
    assert manifest.study_lock_sha256 == value["study_lock_sha256"]
    assert not hasattr(manifest, "dataset_receipt_sha256")


def test_schema2_and_schema3_releases_cannot_cross_bind(
    v3_case,
    tmp_path,
):
    from tests.test_msctl import _cohort_release

    v3_release = load_release(v3_case["packaged"].release)
    legacy_root = tmp_path / "legacy-cross-version"
    legacy_release = load_release(
        _cohort_release(legacy_root, "aws-p5.48xlarge")
    )

    schema2_value = _legacy_schema2_manifest(
        Path(v3_case["source"]),
        v3_release,
    )
    schema2 = load_run_manifest(
        _write_json(tmp_path / "schema2-v3-release.json", schema2_value),
        repo_root=v3_case["source"],
    )
    with pytest.raises(MsctlError) as schema2_error:
        bind_release(v3_release, schema2)
    assert schema2_error.value.code == "RELEASE_RUN_MISMATCH"

    schema3_value = _expected_manifest(v3_case, 0)
    schema3_value["release_sha256"] = legacy_release.archive_sha256
    schema3_value["release_receipt_sha256"] = legacy_release.receipt_sha256
    schema3_value["source_commit"] = legacy_release.source_commit
    schema3 = load_run_manifest(
        _write_json(tmp_path / "schema3-v2-release.json", schema3_value),
        repo_root=v3_case["source"],
    )
    with pytest.raises(MsctlError) as schema3_error:
        bind_release(legacy_release, schema3)
    assert schema3_error.value.code == "RELEASE_RUN_MISMATCH"


def test_v3_manifest_hash_and_gpu_hour_types_are_canonical(v3_case, tmp_path):
    value = _expected_manifest(v3_case, 0)
    path = _write_json(tmp_path / "runs.json", value)

    manifest = load_run_manifest(path, repo_root=v3_case["source"])

    assert manifest.sha256 == hashlib.sha256(_canonical(value)).hexdigest()
    assert all(type(run.seed) is int for run in manifest.runs)
    assert all(
        isinstance(run.estimated_gpu_hours, float)
        and math.isfinite(run.estimated_gpu_hours)
        and run.estimated_gpu_hours > 0
        for run in manifest.runs
    )
