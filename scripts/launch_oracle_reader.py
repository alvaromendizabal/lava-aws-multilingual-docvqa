"""Preview or explicitly submit a charge-bounded SageMaker oracle job."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import boto3

from lava.observability import verify_training_quota
from lava.readers.sagemaker import build_job_plan, submit_or_preview_job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", default="qwen35_4b_fused_direct")
    parser.add_argument("--instance-type")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--acknowledge-charges", default="NO")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    bucket = os.environ.get("S3_BUCKET")
    region = os.environ.get("AWS_REGION", "us-west-2")
    os.environ.setdefault("AWS_DEFAULT_REGION", region)
    logging.getLogger("sagemaker.core.helper.iam_role_resolver").setLevel(logging.ERROR)
    if not bucket:
        raise SystemExit("S3_BUCKET is required; run `set -a; source .env; set +a`")
    plan = build_job_plan(
        repo_root=repo_root,
        config_path=repo_root / "configs/oracle_reader_benchmark.yaml",
        model_lock_path=repo_root / "configs/oracle_reader_models.lock.json",
        model_key=args.model_key,
        instance_type=args.instance_type,
        bucket=bucket,
        limit=args.limit,
    )
    print(json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True))
    quota_client = boto3.session.Session(region_name=region).client("service-quotas")
    exact_quota = verify_training_quota(
        service_quotas=quota_client,
        instance_type=str(plan.instance_type),
        instance_count=int(plan.instance_count),
        managed_spot=bool(plan.managed_spot),
    )
    result = submit_or_preview_job(
        plan=plan,
        repo_root=repo_root,
        region=region,
        submit=args.submit,
        wait=args.wait,
        acknowledgement=args.acknowledge_charges,
    )
    result["quota"] = exact_quota
    print(json.dumps(result, indent=2, sort_keys=True))
    print("SAGEMAKER_JOB_SUBMITTED" if result["submitted"] else "SAGEMAKER_PLAN_PREVIEWED")


if __name__ == "__main__":
    main()
