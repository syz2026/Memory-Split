"""Backward-compatible P5 names for the neutral AWS GPU profile contract."""

from cluster.aws.gpu_profile import (
    AwsGpuProfile,
    AwsGpuRuntime,
    P5_PROFILE_ID,
    P5_PROFILE_ID_V3,
    load_aws_gpu_profile,
    parse_aws_gpu_profile_bytes,
    validate_runtime_environment,
)


PROFILE_ID = P5_PROFILE_ID
PROFILE_ID_V3 = P5_PROFILE_ID_V3
AwsP5Profile = AwsGpuProfile
AwsP5Runtime = AwsGpuRuntime
load_aws_p5_profile = load_aws_gpu_profile
parse_aws_p5_profile_bytes = parse_aws_gpu_profile_bytes


__all__ = [
    "AwsP5Profile",
    "AwsP5Runtime",
    "PROFILE_ID",
    "PROFILE_ID_V3",
    "load_aws_p5_profile",
    "parse_aws_p5_profile_bytes",
    "validate_runtime_environment",
]
