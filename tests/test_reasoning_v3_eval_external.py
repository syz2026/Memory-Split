from __future__ import annotations

import os
import json
from pathlib import Path

import pytest

from evals.reasoning_v3.aws_authority import (
    ACTIVATION_KEY,
    AWS_REGION,
    STORAGE_BUCKET,
)
from evals.reasoning_v3.generate import FrozenEvaluationPaths
from evals.reasoning_v3.sealing import (
    load_model_visible_release,
    materialize_frozen_evaluation_release,
    validate_frozen_evaluation_release,
)


ROOT = Path(__file__).resolve().parents[1]


def _external_paths() -> FrozenEvaluationPaths:
    if os.environ.get("MEMORYSPLIT_REASONING_V3_EVAL_AWS_INTEGRATION") != "1":
        pytest.skip(
            "set MEMORYSPLIT_REASONING_V3_EVAL_AWS_INTEGRATION=1 "
            "to use real AWS authority"
        )
    dataset = os.environ.get("MEMORYSPLIT_REASONING_V3_EVAL_DATASET_ROOT")
    source = os.environ.get("MEMORYSPLIT_REASONING_V3_EVAL_SOURCE_STAGE_ROOT")
    if not dataset or not source:
        pytest.skip("external dataset/source-stage inputs were not provided")
    return FrozenEvaluationPaths(
        repository_root=ROOT,
        dataset_root=Path(dataset),
        source_stage_root=Path(source),
    )


def test_external_full_manifest_replay_and_real_aws_authority():
    """Opt-in: replay 7,530,527 records and exercise IAM/KMS/S3 end to end."""

    paths = _external_paths()
    activated = materialize_frozen_evaluation_release(paths)
    assert activated.registry_sha256 == validate_frozen_evaluation_release(paths)
    visible = load_model_visible_release()
    assert visible["item_count"] == 14 * 512
    assert visible["registry_sha256"] == activated.registry_sha256


@pytest.mark.parametrize(
    "profile_variable",
    [
        "MEMORYSPLIT_REASONING_V3_EVAL_TRAINER_AWS_PROFILE",
        "MEMORYSPLIT_REASONING_V3_EVAL_OPERATOR_AWS_PROFILE",
    ],
)
def test_external_non_evaluator_is_denied_sealed_gold_at_aws_boundary(
    profile_variable: str,
    monkeypatch: pytest.MonkeyPatch,
):
    if os.environ.get("MEMORYSPLIT_REASONING_V3_EVAL_AWS_INTEGRATION") != "1":
        pytest.skip(
            "set MEMORYSPLIT_REASONING_V3_EVAL_AWS_INTEGRATION=1 "
            "to use real AWS authority"
        )
    profile = os.environ.get(profile_variable)
    if not profile:
        pytest.skip(f"explicit non-evaluator profile absent: {profile_variable}")

    monkeypatch.setenv("AWS_PROFILE", profile)
    for variable in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        monkeypatch.delenv(variable, raising=False)

    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError

    session = boto3.Session(region_name=AWS_REGION, profile_name=profile)
    s3 = session.client(
        "s3",
        config=Config(
            region_name=AWS_REGION,
            signature_version="v4",
        ),
        endpoint_url=f"https://s3.{AWS_REGION}.amazonaws.com",
        region_name=AWS_REGION,
        verify=True,
    )
    activation_response = s3.get_object(
        Bucket=STORAGE_BUCKET,
        Key=ACTIVATION_KEY,
    )
    activation = json.loads(activation_response["Body"].read())["activation"]
    sealed = activation["artifacts"]["sealed_gold"]
    with pytest.raises(ClientError) as denied:
        s3.get_object(
            Bucket=sealed["bucket"],
            Key=sealed["key"],
            VersionId=sealed["version_id"],
        )
    assert denied.value.response["Error"]["Code"] in {
        "AccessDenied",
        "403",
        "KMS.AccessDeniedException",
    }
