"""Check the exact SageMaker Training quota for the selected billing mode."""

from __future__ import annotations

import argparse
import json
import os

import boto3

from lava.observability import verify_training_quota


def main() -> int:
    """Read one immutable quota code and fail closed if capacity is insufficient."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-type", default="ml.g6e.2xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--spot", action="store_true")
    args = parser.parse_args()
    region = os.environ.get("AWS_REGION", "us-west-2")
    client = boto3.session.Session(region_name=region).client("service-quotas")
    quota = verify_training_quota(
        service_quotas=client,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        managed_spot=args.spot,
    )
    print(json.dumps(quota, indent=2, sort_keys=True))
    print("EXACT_SAGEMAKER_TRAINING_QUOTA_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
