"""No-cost readiness gate for a configured oracle-reader SageMaker job."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import yaml
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from lava.notebook_support import find_repo_root, git_snapshot, public_metadata
from lava.observability import (
    EventLogger,
    enforce_cost_cap,
    estimate_maximum_cost,
    list_training_job_names,
    validate_first_smoke_plan,
    verify_training_quota,
)
from lava.readers.evaluation_contract import validate_evaluation_plan
from lava.readers.sagemaker import build_job_plan, validate_sagemaker_sdk_contract


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return payload


def _head_required_s3_objects(*, s3: Any, bucket: str, keys: list[str]) -> None:
    for key in keys:
        s3.head_object(Bucket=bucket, Key=key)


def _resolved_model_count(model_lock: Mapping[str, object]) -> int:
    resolved = model_lock.get("resolved_models")
    if not isinstance(resolved, list) or not resolved:
        raise RuntimeError("Model lock contains no resolved model revisions.")
    return len(resolved)


def main() -> int:
    """Validate local state, private assets, quota, and cost without paid compute."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", default="qwen35_9b_fused_direct")
    parser.add_argument("--mode", choices=("smoke", "benchmark"), default="smoke")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--hourly-usd-ceiling", type=float, default=10.0)
    parser.add_argument("--maximum-total-usd", type=float, default=12.5)
    args = parser.parse_args()

    root = find_repo_root(Path(__file__).resolve())
    os.chdir(root)
    load_dotenv(root / ".env", override=False)

    run_id = f"preflight-{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    logger = EventLogger.to_stdout(
        run_id=run_id,
        component="oracle_reader.preflight",
        jsonl_path=root / "artifacts" / "oracle_reader" / "runtime" / f"{run_id}.jsonl",
    )

    with logger.stage("preflight", heartbeat_seconds=15.0):
        region = os.environ.get("AWS_REGION", "us-west-2")
        bucket_value = os.environ.get("S3_BUCKET")
        if not bucket_value:
            raise RuntimeError("S3_BUCKET is missing from the project environment.")
        bucket = str(bucket_value)

        protocol_lock = _load_json(root / "configs" / "evaluation_protocol.lock.json")
        model_lock = _load_json(root / "configs" / "oracle_reader_models.lock.json")
        asset_summary = _load_json(
            root / "reports" / "oracle_reader" / "oracle_assets_summary.json"
        )
        config_path = root / "configs" / "oracle_reader_benchmark.yaml"
        benchmark_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(benchmark_config, Mapping):
            raise TypeError("Oracle-reader benchmark configuration must be a mapping.")

        protocol_lock_id = protocol_lock.get("protocol_lock_id")
        if not isinstance(protocol_lock_id, str) or not protocol_lock_id:
            raise RuntimeError("Frozen evaluation protocol has no valid protocol_lock_id.")
        if benchmark_config.get("protocol_lock_id") != protocol_lock_id:
            raise RuntimeError("Benchmark configuration conflicts with the protocol lock.")
        model_protocol_lock_id = model_lock.get("protocol_lock_id")
        if model_protocol_lock_id is not None and model_protocol_lock_id != protocol_lock_id:
            raise RuntimeError("Model lock conflicts with the evaluation protocol.")
        if asset_summary.get("protocol_lock_id") != protocol_lock_id:
            raise RuntimeError("Oracle asset summary conflicts with the protocol lock.")

        model_count = _resolved_model_count(model_lock)
        runtime_config = benchmark_config.get("training_runtime")
        if not isinstance(runtime_config, Mapping):
            raise TypeError("training_runtime must be a mapping.")
        sdk_version = runtime_config.get("sdk_version")
        if not isinstance(sdk_version, str):
            raise TypeError("training_runtime.sdk_version must be a string.")
        validate_sagemaker_sdk_contract(sdk_version)

        plan_model = build_job_plan(
            repo_root=root,
            config_path=config_path,
            model_lock_path=root / "configs" / "oracle_reader_models.lock.json",
            model_key=args.model_key,
            bucket=bucket,
            limit=args.limit if args.limit is not None else (16 if args.mode == "benchmark" else 1),
            mode=args.mode,
        )
        plan = plan_model.model_dump(mode="json")
        if plan.get("protocol_lock_id") != protocol_lock_id:
            raise RuntimeError("Constructed SageMaker plan conflicts with protocol lock.")
        if args.mode == "benchmark":
            validate_evaluation_plan(plan_model, root)
        else:
            validate_first_smoke_plan(plan)

        max_runtime_seconds = plan.get("max_runtime_seconds")
        instance_count = plan.get("instance_count")
        managed_spot = plan.get("managed_spot")
        instance_type = plan.get("instance_type")
        if not isinstance(max_runtime_seconds, int):
            raise TypeError("Plan max_runtime_seconds must be an integer.")
        if not isinstance(instance_count, int):
            raise TypeError("Plan instance_count must be an integer.")
        if not isinstance(managed_spot, bool):
            raise TypeError("Plan managed_spot must be boolean.")
        if not isinstance(instance_type, str):
            raise TypeError("Plan instance_type must be a string.")

        estimate = estimate_maximum_cost(
            hourly_usd_ceiling=args.hourly_usd_ceiling,
            max_runtime_seconds=max_runtime_seconds,
            instance_count=instance_count,
            contingency_factor=1.25,
        )
        enforce_cost_cap(estimate, maximum_allowed_usd=args.maximum_total_usd)

        session = boto3.session.Session(region_name=region)
        identity = session.client("sts").get_caller_identity()
        s3 = session.client("s3")
        sagemaker = session.client("sagemaker")
        service_quotas = session.client("service-quotas")
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
            raise RuntimeError(
                "A LAVA/Oracle SageMaker training job is already active: " + ", ".join(active_jobs)
            )

        try:
            quota = verify_training_quota(
                service_quotas=service_quotas,
                instance_type=instance_type,
                instance_count=instance_count,
                managed_spot=managed_spot,
            )
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            raise RuntimeError(
                "Unable to verify exact SageMaker Training quota; "
                f"Service Quotas returned {error_code!r}."
            ) from exc

        snapshot: dict[str, object] = {
            **git_snapshot(root),
            "aws_arn": identity["Arn"],
            "aws_region": region,
            "model_key": args.model_key,
            "instance_type": instance_type,
            "protocol_lock_id": protocol_lock_id,
            "model_count": model_count,
            "quota": quota,
            "active_lava_training_jobs": active_jobs,
            "cost_guard": estimate.as_dict(),
            "job_plan": public_metadata(plan),
            "paid_resource_created": False,
        }
        logger.emit("preflight.verified", event_fields=snapshot)
        print(json.dumps(public_metadata(snapshot), indent=2, sort_keys=True))

    print("ORACLE_READER_PREFLIGHT_VERIFIED")
    print("NO_PAID_SAGEMAKER_RESOURCE_WAS_CREATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
