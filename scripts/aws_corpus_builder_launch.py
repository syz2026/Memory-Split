#!/usr/bin/env python3
"""Launch one explicitly approved AWS corpus builder."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cluster.aws.corpus_builder.contracts import (  # noqa: E402
    launch_intent_from_bytes,
)
from cluster.aws.corpus_builder.launch import (  # noqa: E402
    BuilderLaunch,
    LaunchError,
    launch_approved_builder,
)


_MAX_INTENT_BYTES = 128 * 1024


class _AwsLaunchClient:
    def __init__(self, ec2: object, pricing: object, sts: object) -> None:
        self._ec2 = ec2
        self._pricing = pricing
        self._sts = sts

    def get_caller_identity(self, **kwargs: object) -> Mapping[str, object]:
        return self._sts.get_caller_identity(**kwargs)

    def describe_launch_template_versions(
        self,
        **kwargs: object,
    ) -> Mapping[str, object]:
        return self._ec2.describe_launch_template_versions(**kwargs)

    def get_products(self, **kwargs: object) -> Mapping[str, object]:
        return self._pricing.get_products(**kwargs)

    def describe_security_groups(
        self,
        **kwargs: object,
    ) -> Mapping[str, object]:
        return self._ec2.describe_security_groups(**kwargs)

    def run_instances(self, **kwargs: object) -> Mapping[str, object]:
        return self._ec2.run_instances(**kwargs)

    def describe_instances(self, **kwargs: object) -> Mapping[str, object]:
        return self._ec2.describe_instances(**kwargs)

    def describe_instance_attribute(
        self,
        **kwargs: object,
    ) -> Mapping[str, object]:
        return self._ec2.describe_instance_attribute(**kwargs)

    def terminate_instances(self, **kwargs: object) -> Mapping[str, object]:
        return self._ec2.terminate_instances(**kwargs)


def _new_session(*, profile_name: str, region_name: str):
    import boto3

    return boto3.Session(
        profile_name=profile_name,
        region_name=region_name,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--approve-intent-sha256", required=True)
    parser.add_argument(
        "--profile",
        choices=("sbsandbox",),
        required=True,
    )
    parser.add_argument(
        "--region",
        choices=("us-east-1",),
        required=True,
    )
    return parser


def _read_intent(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise LaunchError("launch intent must be a regular file")
    payload = path.read_bytes()
    if len(payload) > _MAX_INTENT_BYTES:
        raise LaunchError("launch intent exceeds 128 KiB")
    return payload


def _success_report(
    launch: BuilderLaunch,
    *,
    max_compute_usd: str,
) -> dict[str, str]:
    return {
        "instance_id": launch.instance_id,
        "launch_intent_sha256": launch.launch_intent_sha256,
        "launch_time": launch.launch_time,
        "max_compute_usd": max_compute_usd,
        "terminate_at": launch.terminate_at,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        intent_bytes = _read_intent(arguments.intent)
        intent = launch_intent_from_bytes(intent_bytes)
        session = _new_session(
            profile_name=arguments.profile,
            region_name=arguments.region,
        )
        ec2 = session.client("ec2", region_name=arguments.region)
        pricing = session.client("pricing", region_name="us-east-1")
        sts = session.client("sts", region_name=arguments.region)
        launch = launch_approved_builder(
            intent_bytes,
            approved_intent_sha256=arguments.approve_intent_sha256,
            ec2=_AwsLaunchClient(ec2, pricing, sts),
            now=_utc_now(),
            clock=_utc_now,
        )
    except (LaunchError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    report = _success_report(
        launch,
        max_compute_usd=format(intent.max_compute_usd, "f"),
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
