from __future__ import annotations

import builtins
import inspect
import json
import os
from types import SimpleNamespace
from pathlib import Path

import pytest

import scripts.run_reasoning_v3_evals as eval_cli
import scripts.run_reasoning_v3_inference as inference_cli
from evals.reasoning_v3.aws_authority import (
    AWS_REGION,
    AwsAuthorityError,
    S3ObjectVersion,
    STORAGE_BUCKET,
    TASK3_REPORT_KEY,
    TASK3_RESULT_PREFIX,
)
from evals.reasoning_v3.generate import FrozenEvaluationPaths
from evals.reasoning_v3.reporting import run_frozen_scientific_inference
from evals.reasoning_v3.runner import (
    CHECKPOINT_STEPS,
    run_frozen_checkpoint_evaluation,
)
from evals.reasoning_v3.sealing import validate_frozen_evaluation_release


ROOT = Path(__file__).resolve().parents[1]


def test_cli_defaults_are_non_mutating_canonical_dry_runs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        eval_cli,
        "run_frozen_checkpoint_evaluation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry run mutated")
        ),
    )
    assert (
        eval_cli.main(
            [
                "--arm",
                "dense",
                "--seed",
                "0",
                "--checkpoint-step",
                "15582",
            ]
        )
        == 0
    )
    eval_output = capsys.readouterr().out
    assert eval_output.endswith("\n") and not eval_output.endswith("\n\n")
    assert json.loads(eval_output)["mode"] == "dry_run"

    monkeypatch.setattr(
        inference_cli,
        "run_frozen_scientific_inference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry run mutated")
        ),
    )
    assert inference_cli.main([]) == 0
    inference_output = capsys.readouterr().out
    assert inference_output.endswith("\n") and not inference_output.endswith("\n\n")
    assert json.loads(inference_output)["mode"] == "dry_run"


def test_production_statistical_inference_fails_closed_without_fixed_aws_sdk(
    monkeypatch: pytest.MonkeyPatch,
):
    real_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "boto3" or name.startswith("botocore"):
            raise ImportError("simulated unavailable AWS SDK")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(AwsAuthorityError, match="required"):
        run_frozen_scientific_inference()


def _external_paths() -> FrozenEvaluationPaths:
    if os.environ.get("MEMORYSPLIT_REASONING_V3_TASK3_AWS_INTEGRATION") != "1":
        pytest.skip(
            "set MEMORYSPLIT_REASONING_V3_TASK3_AWS_INTEGRATION=1 "
            "to use real evaluator IAM/KMS/S3 authority"
        )
    dataset = os.environ.get("MEMORYSPLIT_REASONING_V3_EVAL_DATASET_ROOT")
    source = os.environ.get("MEMORYSPLIT_REASONING_V3_EVAL_SOURCE_STAGE_ROOT")
    if not dataset or not source:
        pytest.skip(
            "external complete checkpoints, corpus, and source stage were not provided"
        )
    return FrozenEvaluationPaths(
        repository_root=ROOT,
        dataset_root=Path(dataset),
        source_stage_root=Path(source),
    )


def test_external_complete_matrix_scores_and_reports_through_real_aws():
    """Opt-in: evaluate all 100 checkpoints and publish the frozen report."""

    evaluation_paths = _external_paths()
    validated_registry = validate_frozen_evaluation_release(evaluation_paths)
    refs = [
        run_frozen_checkpoint_evaluation(arm, seed, step)
        for seed in range(10)
        for arm in ("dense", "split90")
        for step in CHECKPOINT_STEPS
    ]
    assert len(refs) == 100
    assert all(result.object_ref.version_id for result in refs)
    published = run_frozen_scientific_inference()
    assert published.report["primary"]
    assert validated_registry
    assert published.object_ref.version_id
    assert published.report["matrix"]["checkpoint_count"] == 100


def test_external_non_evaluator_cannot_publish_or_read_task3_results():
    if os.environ.get("MEMORYSPLIT_REASONING_V3_TASK3_AWS_INTEGRATION") != "1":
        pytest.skip(
            "set MEMORYSPLIT_REASONING_V3_TASK3_AWS_INTEGRATION=1 "
            "to use real non-evaluator AWS credentials"
        )
    profile = os.environ.get(
        "MEMORYSPLIT_REASONING_V3_TASK3_NON_EVALUATOR_AWS_PROFILE"
    )
    if not profile:
        pytest.skip("explicit non-evaluator AWS profile was not provided")
    result_version = os.environ.get(
        "MEMORYSPLIT_REASONING_V3_TASK3_RESULT_VERSION_ID"
    )
    report_version = os.environ.get(
        "MEMORYSPLIT_REASONING_V3_TASK3_REPORT_VERSION_ID"
    )
    if not result_version or not report_version:
        pytest.skip("exact external result/report version IDs were not provided")
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError

    session = boto3.Session(profile_name=profile, region_name=AWS_REGION)
    s3 = session.client(
        "s3",
        config=Config(
            region_name=AWS_REGION,
            retries={"max_attempts": 1, "mode": "standard"},
            signature_version="v4",
        ),
        endpoint_url=f"https://s3.{AWS_REGION}.amazonaws.com",
        region_name=AWS_REGION,
        verify=True,
    )
    result_key = (
        f"{TASK3_RESULT_PREFIX}dense/seed-0/"
        f"step-{CHECKPOINT_STEPS[-1]:07d}.json"
    )
    with pytest.raises(ClientError) as result_denial:
        s3.get_object(
            Bucket=STORAGE_BUCKET,
            Key=result_key,
            VersionId=result_version,
        )
    assert result_denial.value.response["Error"]["Code"] == "AccessDenied"
    with pytest.raises(ClientError) as report_denial:
        s3.get_object(
            Bucket=STORAGE_BUCKET,
            Key=TASK3_REPORT_KEY,
            VersionId=report_version,
        )
    assert report_denial.value.response["Error"]["Code"] == "AccessDenied"


def test_non_evaluator_external_check_uses_direct_exact_version_s3_reads():
    source = inspect.getsource(
        test_external_non_evaluator_cannot_publish_or_read_task3_results
    )
    assert "boto3.Session" in source
    assert "s3.get_object" in source
    assert source.count("VersionId=") >= 2
    assert source.count('"AccessDenied"') >= 2
    assert "_new_fixed_aws_authority" not in source


def test_invalid_checkpoint_apply_returns_nonzero_and_names_failed_gates(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    failed = ["memory_off_leakage", "exact_resume"]
    ref = S3ObjectVersion(
        bucket="bucket",
        bytes=1,
        key="result",
        kms_key_arn="kms",
        server_side_encryption="aws:kms",
        sha256="a" * 64,
        version_id="version",
    )
    monkeypatch.setattr(
        eval_cli,
        "run_frozen_checkpoint_evaluation",
        lambda *_args: SimpleNamespace(
            object_ref=ref,
            result={
                "validity": {
                    "gates": {
                        name: {"passed": name not in failed}
                        for name in (
                            "factual_burden",
                            "memory_on_recall",
                            "memory_off_leakage",
                            "exact_resume",
                            "evaluator_authority",
                            "complete_registry",
                            "no_substitution",
                            "no_outcome_dependent_stopping",
                            "no_replacement",
                            "no_exclusion",
                            "provider_fixed",
                            "no_missing_seed_imputation",
                        )
                    },
                    "passed": False,
                }
            },
        ),
    )
    status = eval_cli.main(
        [
            "--arm",
            "dense",
            "--seed",
            "0",
            "--checkpoint-step",
            "15582",
            "--apply",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert status == 2
    assert output["mode"] == "invalid"
    assert output["failed_gates"] == failed
    assert "validated" not in json.dumps(output).lower()
