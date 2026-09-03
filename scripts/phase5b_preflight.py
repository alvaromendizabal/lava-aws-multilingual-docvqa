"""No-cost preflight for the observable one-question SageMaker smoke job."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from lava.notebook_support import find_repo_root, git_snapshot, public_metadata
from lava.observability import (
    EventLogger,
    enforce_cost_cap,
    estimate_maximum_cost,
    list_training_job_names,
    validate_first_smoke_plan,
)
from lava.readers.sagemaker import build_job_plan, validate_sagemaker_sdk_contract

_INSTANCE_QUOTA_FRAGMENT = "ml.g6e.2xlarge"


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"Expected a JSON object in {path}."
        raise TypeError(message)
    return payload


def _find_training_quota(service_quotas: Any) -> dict[str, object]:
    paginator = service_quotas.get_paginator("list_service_quotas")
    for page in paginator.paginate(ServiceCode="sagemaker"):
        for quota in page.get("Quotas", []):
            quota_name = quota.get("QuotaName", "")
            if _INSTANCE_QUOTA_FRAGMENT in quota_name and "training job" in quota_name.lower():
                return {
                    "found": True,
                    "quota_name": quota_name,
                    "quota_code": quota.get("QuotaCode"),
                    "value": quota.get("Value"),
                    "adjustable": quota.get("Adjustable"),
                }
    return {"found": False, "quota_name": None, "value": None}


def _head_required_s3_objects(*, s3: Any, bucket: str, keys: list[str]) -> None:
    for key in keys:
        s3.head_object(Bucket=bucket, Key=key)


def main() -> int:
    """Run a no-cost validation of code, data, quotas, and charge guards."""
    root = find_repo_root(Path(__file__).resolve())
    os.chdir(root)
    load_dotenv(root / ".env", override=False)
    run_id = f"phase5b-preflight-{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    logger = EventLogger.to_stdout(
        run_id=run_id,
        component="phase5b.preflight",
        jsonl_path=root / "artifacts" / "oracle_reader" / "runtime" / f"{run_id}.jsonl",
    )

    with logger.stage("preflight", heartbeat_seconds=15.0):
        environment = {
            "AWS_REGION": os.environ.get("AWS_REGION", "us-west-2"),
            "S3_BUCKET": os.environ.get("S3_BUCKET"),
        }
        if not environment["S3_BUCKET"]:
            message = "S3_BUCKET is missing. Load the project .env file before preflight."
            raise RuntimeError(message)

        protocol_lock = _load_json(root / "configs" / "evaluation_protocol.lock.json")
        model_lock = _load_json(root / "configs" / "oracle_reader_models.lock.json")
        asset_summary = _load_json(
            root / "reports" / "oracle_reader" / "oracle_assets_summary.json"
        )
        config_path = root / "configs" / "oracle_reader_benchmark.yaml"

        protocol_lock_id = protocol_lock.get("protocol_lock_id")
        if model_lock.get("protocol_lock_id") != protocol_lock_id:
            message = "Model lock does not match the frozen evaluation protocol."
            raise RuntimeError(message)
        if asset_summary.get("protocol_lock_id") != protocol_lock_id:
            message = "Oracle asset summary does not match the frozen evaluation protocol."
            raise RuntimeError(message)
        if asset_summary.get("question_count") != 16:
            message = "Oracle asset summary must contain exactly 16 labeled questions."
            raise RuntimeError(message)

        validate_sagemaker_sdk_contract("3.21.0")
        plan_model = build_job_plan(
            repo_root=root,
            config_path=config_path,
            model_lock_path=root / "configs" / "oracle_reader_models.lock.json",
            model_key="qwen35_4b_fused_direct",
            bucket=str(environment["S3_BUCKET"]),
            limit=1,
        )
        plan = plan_model.model_dump(mode="json")
        validate_first_smoke_plan(plan)

        estimate = estimate_maximum_cost(
            hourly_usd_ceiling=10.0,
            max_runtime_seconds=int(plan["max_runtime_seconds"]),
            instance_count=int(plan["instance_count"]),
            contingency_factor=1.25,
        )
        enforce_cost_cap(estimate, maximum_allowed_usd=12.50)

        session = boto3.session.Session(region_name=str(environment["AWS_REGION"]))
        identity = session.client("sts").get_caller_identity()
        s3 = session.client("s3")
        sagemaker = session.client("sagemaker")
        service_quotas = session.client("service-quotas")
        bucket = str(environment["S3_BUCKET"])
        _head_required_s3_objects(
            s3=s3,
            bucket=bucket,
            keys=[
                "splits/evaluation-protocol/latest/protocol_lock.json",
                "processed/oracle-reader/v3/manifests/latest/oracle_examples.jsonl",
            ],
        )

        active_jobs = sorted(
            name
            for name in list_training_job_names(sagemaker_client=sagemaker)
            if "oracle" in name.lower() or "lava" in name.lower()
        )
        if active_jobs:
            message = "A LAVA/Oracle SageMaker training job is already active."
            raise RuntimeError(message)

        try:
            quota = _find_training_quota(service_quotas)
        except ClientError as exc:
            quota = {
                "found": False,
                "warning": "Unable to read Service Quotas with the current role.",
                "error_code": exc.response.get("Error", {}).get("Code"),
            }

        snapshot: dict[str, object] = {
            **git_snapshot(root),
            "aws_arn": identity["Arn"],
            "aws_region": environment["AWS_REGION"],
            "question_count": asset_summary["question_count"],
            "document_count": asset_summary["document_count"],
            "unique_evidence_page_count": asset_summary["unique_evidence_page_count"],
            "protocol_lock_id": protocol_lock_id,
            "model_count": len(model_lock.get("resolved_models", [])),
            "quota": quota,
            "active_lava_training_jobs": active_jobs,
            "cost_guard": estimate.as_dict(),
            "job_plan": public_metadata(plan),
            "paid_resource_created": False,
        }
        logger.emit("preflight.verified", **snapshot)
        print(json.dumps(snapshot, indent=2, sort_keys=True))

    print("PHASE_5B_OBSERVABLE_SMOKE_PREFLIGHT_COMPLETE")
    print("NO_PAID_SAGEMAKER_RESOURCE_WAS_CREATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
