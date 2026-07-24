"""Read-only AWS identity, P-instance offering, and quota inspection."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Sequence
from typing import Any

INSTANCE_TYPES = ("p5.48xlarge", "p5en.48xlarge", "p6-b200.48xlarge")
_REGION_RE = re.compile(r"^[a-z]{2}(?:-gov)?-[a-z]+-\d$")


def _runner(command: Sequence[str]):
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )


def _json_command(
    command: list[str],
    *,
    runner: Callable[[Sequence[str]], object],
) -> Any:
    result = runner(command)
    returncode = int(result.returncode)
    stdout = str(getattr(result, "stdout", "") or "")
    stderr = str(getattr(result, "stderr", "") or "")
    if returncode:
        raise RuntimeError(
            f"AWS inspection command failed ({returncode}): {stderr.strip()}"
        )
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("AWS inspection returned invalid JSON") from error


def inspect_aws_readiness(
    *,
    region: str,
    runner: Callable[[Sequence[str]], object] = _runner,
) -> dict[str, Any]:
    """Inspect account reachability; this never creates or changes resources."""

    if not isinstance(region, str) or not _REGION_RE.fullmatch(region):
        raise ValueError("region must be a canonical AWS region name")
    identity = _json_command(
        ["aws", "sts", "get-caller-identity", "--output", "json"],
        runner=runner,
    )
    offerings = _json_command(
        [
            "aws",
            "ec2",
            "describe-instance-type-offerings",
            "--location-type",
            "availability-zone",
            "--filters",
            f"Name=instance-type,Values={','.join(INSTANCE_TYPES)}",
            "--region",
            region,
            "--output",
            "json",
        ],
        runner=runner,
    )
    quotas = _json_command(
        [
            "aws",
            "service-quotas",
            "list-service-quotas",
            "--service-code",
            "ec2",
            "--region",
            region,
            "--query",
            "Quotas[?contains(QuotaName, `Running On-Demand P`)]",
            "--output",
            "json",
        ],
        runner=runner,
    )
    if not isinstance(identity, dict) or not {
        "Account",
        "Arn",
        "UserId",
    } <= set(identity):
        raise RuntimeError("AWS identity response is malformed")
    values = offerings.get("InstanceTypeOfferings") if isinstance(offerings, dict) else None
    if not isinstance(values, list) or not isinstance(quotas, list):
        raise RuntimeError(  # noqa: TRY004
            "AWS offering or quota response is malformed"
        )
    by_type = {instance_type: [] for instance_type in INSTANCE_TYPES}
    for value in values:
        if (
            isinstance(value, dict)
            and value.get("InstanceType") in by_type
            and isinstance(value.get("Location"), str)
        ):
            by_type[value["InstanceType"]].append(value["Location"])
    by_type = {
        key: sorted(set(value))
        for key, value in by_type.items()
    }
    quota_values = [
        {
            "adjustable": value.get("Adjustable"),
            "name": value.get("QuotaName"),
            "quota_code": value.get("QuotaCode"),
            "unit": value.get("Unit"),
            "value": value.get("Value"),
        }
        for value in quotas
        if isinstance(value, dict)
    ]
    return {
        "account": identity["Account"],
        "caller_arn": identity["Arn"],
        "instance_type_availability_zones": by_type,
        "note": (
            "Offerings and quotas do not guarantee immediate capacity; "
            "the measured Slurm GPU canaries remain required."
        ),
        "on_demand_p_quotas": quota_values,
        "region": region,
        "schema_version": 1,
    }
