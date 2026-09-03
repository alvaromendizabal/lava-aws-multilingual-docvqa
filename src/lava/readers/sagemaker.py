"""Charge-gated SageMaker SDK V3 orchestration for oracle reader jobs."""

from __future__ import annotations

import inspect
import json
import re
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

import boto3
import yaml
from botocore.exceptions import ClientError

from lava.readers.model_registry import load_resolved_model
from lava.readers.schemas import SageMakerJobPlan


def _git_sha(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_clean(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def validate_sagemaker_sdk_contract(expected_version: str) -> dict[str, Any]:
    """Validate exact SDK interfaces without constructing or submitting a job."""
    from sagemaker.train import ModelTrainer
    from sagemaker.train.configs import (
        Compute,
        OutputDataConfig,
        SourceCode,
        StoppingCondition,
        Tag,
    )

    observed_version = version("sagemaker")
    if observed_version != expected_version:
        raise RuntimeError(f"Expected sagemaker=={expected_version}, observed {observed_version}")
    required_signatures = {
        "SourceCode": (
            inspect.signature(SourceCode).parameters,
            {"source_dir", "requirements", "entry_script", "ignore_patterns"},
        ),
        "Compute": (
            inspect.signature(Compute).parameters,
            {
                "instance_type",
                "instance_count",
                "volume_size_in_gb",
                "enable_managed_spot_training",
            },
        ),
        "StoppingCondition": (
            inspect.signature(StoppingCondition).parameters,
            {"max_runtime_in_seconds", "max_wait_time_in_seconds"},
        ),
        "OutputDataConfig": (
            inspect.signature(OutputDataConfig).parameters,
            {"s3_output_path", "compression_type"},
        ),
        "Tag": (inspect.signature(Tag).parameters, {"key", "value"}),
        "ModelTrainer.train": (
            inspect.signature(ModelTrainer.train).parameters,
            {"input_data_config", "wait", "logs"},
        ),
    }
    for label, (observed, required) in required_signatures.items():
        missing = sorted(required - set(observed))
        if missing:
            raise RuntimeError(f"SageMaker SDK contract changed for {label}; missing={missing}")
    return {
        "sdk_version": observed_version,
        "validated_interfaces": sorted(required_signatures),
    }


def find_training_quota(
    *,
    region: str,
    instance_type: str,
    spot: bool,
) -> dict[str, Any]:
    """Locate exactly one quota for the selected SageMaker training billing mode."""
    client = boto3.client("service-quotas", region_name=region)

    suffix = "spot training job usage" if spot else "training job usage"
    expected_name = f"{instance_type} for {suffix}"
    expected_normalized = " ".join(re.findall(r"[a-z0-9.]+", expected_name.casefold()))

    token: str | None = None
    matches: list[dict[str, Any]] = []

    try:
        while True:
            kwargs: dict[str, Any] = {
                "ServiceCode": "sagemaker",
                "MaxResults": 100,
            }
            if token:
                kwargs["NextToken"] = token

            response = client.list_service_quotas(**kwargs)

            for quota in response.get("Quotas", []):
                quota_name = str(quota.get("QuotaName", ""))
                normalized_name = " ".join(re.findall(r"[a-z0-9.]+", quota_name.casefold()))

                if normalized_name == expected_normalized:
                    matches.append(quota)

            token = response.get("NextToken")
            if not token:
                break

    except ClientError as error:
        return {
            "found": False,
            "status": "permission_or_api_error",
            "error_code": error.response.get("Error", {}).get("Code"),
            "instance_type": instance_type,
            "spot": spot,
            "value": None,
            "quota_name": None,
            "quota_code": None,
        }

    if not matches:
        return {
            "found": False,
            "status": "not_found",
            "error_code": None,
            "instance_type": instance_type,
            "spot": spot,
            "value": None,
            "quota_name": None,
            "quota_code": None,
        }

    if len(matches) != 1:
        return {
            "found": False,
            "status": "ambiguous",
            "error_code": None,
            "instance_type": instance_type,
            "spot": spot,
            "value": None,
            "quota_name": None,
            "quota_code": None,
            "match_count": len(matches),
        }

    quota = matches[0]

    return {
        "found": True,
        "status": "verified",
        "error_code": None,
        "instance_type": instance_type,
        "spot": spot,
        "value": float(quota.get("Value", 0.0)),
        "quota_name": quota.get("QuotaName"),
        "quota_code": quota.get("QuotaCode"),
        "adjustable": bool(quota.get("Adjustable", False)),
    }


def build_job_plan(
    *,
    repo_root: Path,
    config_path: Path,
    model_lock_path: Path,
    model_key: str,
    bucket: str,
    limit: int,
) -> SageMakerJobPlan:
    """Build a fully specified, non-submitting SageMaker job plan."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = load_resolved_model(model_lock_path, model_key)
    runtime = config["training_runtime"]
    git_sha = _git_sha(repo_root)
    manifest_key = config["benchmark"]["private_manifest_s3_key"]
    output_prefix = (
        f"s3://{bucket}/{config['benchmark']['results_prefix']}/"
        f"{config['protocol_lock_id']}/{model_key}/{git_sha}"
    )
    return SageMakerJobPlan(
        sdk_version=runtime["sdk_version"],
        model_key=model_key,
        model_id=model.model_id,
        model_revision=model.revision,
        protocol_lock_id=config["protocol_lock_id"],
        git_commit_sha=git_sha,
        bucket=bucket,
        manifest_s3_uri=f"s3://{bucket}/{manifest_key}",
        output_s3_prefix=output_prefix,
        training_image=runtime["image_uri"],
        training_image_digest=runtime["image_digest"],
        instance_type=model.instance_type,
        instance_count=int(runtime["instance_count"]),
        volume_size_gb=int(runtime["volume_size_gb"]),
        max_runtime_seconds=int(runtime["max_runtime_seconds"]),
        max_wait_seconds=int(runtime["max_wait_seconds"]),
        managed_spot=bool(runtime["managed_spot_for_smoke"]),
        limit=limit,
        input_mode=model.input_mode,
        generation=model.generation,
        creates_endpoint=False,
    )


def validate_submission_guardrails(plan: SageMakerJobPlan, *, repo_root: Path) -> None:
    """Refuse mutable, oversized, endpoint-backed, or uncommitted smoke jobs."""
    if not _git_clean(repo_root):
        raise RuntimeError("Commit all code and lock files before submitting a paid job")
    if plan.instance_count != 1:
        raise ValueError("Only one SageMaker instance is permitted")
    if plan.max_runtime_seconds > 3600 or plan.max_wait_seconds > 3600:
        raise ValueError("Smoke jobs are capped at one hour")
    if plan.limit != 1:
        raise ValueError("The first paid smoke job must run exactly one question")
    if plan.creates_endpoint:
        raise ValueError("Oracle reader jobs must never create a persistent endpoint")
    bucket_prefix = f"s3://{plan.bucket}/"
    if not plan.manifest_s3_uri.startswith(bucket_prefix):
        raise ValueError("Manifest must be inside the private project bucket")
    if not plan.output_s3_prefix.startswith(bucket_prefix):
        raise ValueError("Outputs must be inside the private project bucket")


def create_model_trainer(*, plan: SageMakerJobPlan, repo_root: Path, region: str) -> Any:
    """Construct a SageMaker SDK V3 ModelTrainer after all guardrails pass."""
    from sagemaker.core.helper.session_helper import Session, get_execution_role
    from sagemaker.train import ModelTrainer
    from sagemaker.train.configs import (
        Compute,
        OutputDataConfig,
        SourceCode,
        StoppingCondition,
        Tag,
    )

    validate_sagemaker_sdk_contract(plan.sdk_version)
    session = Session()
    role = get_execution_role()
    source = SourceCode(
        source_dir=str(repo_root),
        entry_script="pipelines/oracle_reader/job_entry.py",
        requirements="pipelines/oracle_reader/requirements-gpu.txt",
        ignore_patterns=[
            ".env",
            ".git",
            ".venv",
            ".cache",
            ".ipynb_checkpoints",
            "artifacts",
            "data",
            "notebooks",
            "reports",
        ],
    )
    compute = Compute(
        instance_type=plan.instance_type,
        instance_count=plan.instance_count,
        volume_size_in_gb=plan.volume_size_gb,
        enable_managed_spot_training=plan.managed_spot,
    )
    stopping_kwargs: dict[str, int] = {"max_runtime_in_seconds": plan.max_runtime_seconds}
    if plan.managed_spot:
        stopping_kwargs["max_wait_time_in_seconds"] = plan.max_wait_seconds
    stopping = StoppingCondition(**stopping_kwargs)
    output = OutputDataConfig(
        s3_output_path=f"{plan.output_s3_prefix}/sagemaker-output",
        compression_type="NONE",
    )
    experiment_id = f"smoke-{plan.model_key}-{plan.git_commit_sha[:8]}"
    hyperparameters = {
        "bucket": plan.bucket,
        "region": region,
        "manifest_s3_uri": plan.manifest_s3_uri,
        "output_s3_prefix": plan.output_s3_prefix,
        "protocol_lock_id": plan.protocol_lock_id,
        "model_key": plan.model_key,
        "limit": str(plan.limit),
        "experiment_id": experiment_id,
    }
    return ModelTrainer(
        training_image=plan.training_image,
        source_code=source,
        role=role,
        compute=compute,
        stopping_condition=stopping,
        output_data_config=output,
        hyperparameters=hyperparameters,
        environment={
            "HF_HOME": "/tmp/huggingface",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": str(plan.generation.seed),
            "LAVA_GIT_COMMIT_SHA": plan.git_commit_sha,
            "AWS_DEFAULT_REGION": region,
        },
        tags=[
            Tag(key="Project", value="lava-docvqa"),
            Tag(key="Phase", value="oracle-reader"),
            Tag(key="ProtocolLock", value=plan.protocol_lock_id[:16]),
            Tag(key="ModelKey", value=plan.model_key),
        ],
        base_job_name=f"lava-oracle-{plan.model_key.replace('_', '-')}",
        sagemaker_session=session,
    )


def submit_or_preview_job(
    *,
    plan: SageMakerJobPlan,
    repo_root: Path,
    region: str,
    submit: bool,
    wait: bool,
    acknowledgement: str,
) -> dict[str, Any]:
    """Preview by default; submit only after explicit cost acknowledgement."""
    if not submit:
        return {
            "submitted": False,
            "charged_resources_created": False,
            "plan": plan.model_dump(mode="json"),
        }
    if acknowledgement != "YES":
        raise RuntimeError("Paid submission requires --acknowledge-charges YES")
    validate_submission_guardrails(plan, repo_root=repo_root)
    quota = find_training_quota(
        region=region,
        instance_type=plan.instance_type,
        spot=plan.managed_spot,
    )
    if not quota["found"]:
        raise RuntimeError(
            f"Unable to verify the SageMaker training quota before paid submission: {quota}"
        )
    if float(quota["value"] or 0.0) < 1.0:
        raise RuntimeError(
            "SageMaker training quota is below one instance; request a quota increase first: "
            f"{quota}"
        )
    trainer = create_model_trainer(plan=plan, repo_root=repo_root, region=region)
    trainer.train(wait=wait, logs=wait)
    latest = trainer._latest_training_job
    job_name = getattr(latest, "training_job_name", None) or getattr(latest, "name", None)
    if not job_name:
        raise RuntimeError("SageMaker submitted a job but did not expose its name")
    result = {
        "submitted": True,
        "waited": wait,
        "training_job_name": str(job_name),
        "quota": quota,
        "plan": plan.model_dump(mode="json"),
    }
    state_path = repo_root / "artifacts/oracle_reader/latest_job.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
