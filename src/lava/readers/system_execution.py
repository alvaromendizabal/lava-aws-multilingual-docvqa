"""Charge-gated, idempotent SageMaker execution for the frozen final pilot."""

from __future__ import annotations

import json
import math
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from botocore.exceptions import BotoCoreError, ClientError

from lava.evaluation.semantic import digest, encode
from lava.evaluation.submission_store import atomic_write
from lava.observability.smoke_guard import verify_training_quota
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.readers.system import put_blob, store_for, system_contract, validate_manifest

TERMINAL = {"Completed", "Failed", "Stopped"}
SYSTEM_INSTANCES = {"ml.g6e.2xlarge", "ml.g6e.8xlarge"}


def job_name(identity: str, attempt: int) -> str:
    """A lost API response or disconnected monitor must not create a duplicate job."""
    if (
        not re.fullmatch(r"[0-9a-f]{64}", identity)
        or type(attempt) is not int
        or not 1 <= attempt <= 99
    ):
        raise ValueError("Invalid contract identity or bounded attempt number")
    return f"lava-system-9b-{identity[:32]}-a{attempt}"


def bootstrap(archive_sha: str, bucket: str, region: str, identity: str) -> str:
    """Keep each SageMaker container argument within the service's 256-character limit."""
    if not re.fullmatch(r"[0-9a-f]{64}", archive_sha):
        raise ValueError("Invalid source archive checksum")
    return (
        "cd /opt/ml/input/data/source && "
        'echo "$LAVA_SOURCE_SHA256  source.tar.gz" | sha256sum -c - && '
        "mkdir -p /opt/ml/code && tar -xzf source.tar.gz -C /opt/ml/code --no-same-owner && "
        "exec bash /opt/ml/code/pipelines/oracle_reader/run.sh"
    )


def training_request(
    root: Path,
    manifest: dict[str, Any],
    bucket: str,
    region: str,
    role: str,
    attempt: int,
    source_key: str,
    source_sha: str,
    commit: str,
    *,
    manifest_validator: Callable[..., Any] = validate_manifest,
) -> dict[str, Any]:
    """One pinned image, one GPU, a server-enforced runtime cap, and no endpoint."""
    contract = manifest["contract"]
    manifest_validator(manifest, contract)
    config = contract["config"]
    runtime = yaml.safe_load((root / "configs/oracle_reader_benchmark.yaml").read_bytes())[
        "training_runtime"
    ]
    if not re.fullmatch(r"arn:aws:iam::[0-9]{12}:role/.+", role):
        raise ValueError("A SageMaker execution role ARN is required")
    if region != "us-west-2" or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("The frozen run requires us-west-2 and an immutable code revision")
    identity = contract["contract_id"]
    image = runtime["image_uri"].rsplit(":", 1)[0] + "@" + runtime["image_digest"]
    return {
        "TrainingJobName": job_name(identity, attempt),
        "RoleArn": role,
        "AlgorithmSpecification": {
            "TrainingImage": image,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["/bin/bash"],
            "ContainerArguments": ["-c", bootstrap(source_sha, bucket, region, identity)],
        },
        "InputDataConfig": [
            {
                "ChannelName": "source",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{bucket}/{source_key}",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "CompressionType": "None",
                "InputMode": "File",
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{bucket}/experiments/oracle-reader/system-runs/{identity}/attempt-{attempt}",
            "CompressionType": "NONE",
        },
        "ResourceConfig": {
            "InstanceType": "ml.g6e.2xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 100,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": config["max_runtime_seconds"],
            "MaxPendingTimeInSeconds": config["max_pending_seconds"],
        },
        "EnableManagedSpotTraining": False,
        "Environment": {
            "HF_HOME": "/tmp/huggingface",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "20260902",
            "LAVA_GIT_COMMIT_SHA": commit,
            "AWS_DEFAULT_REGION": region,
            "LAVA_BUCKET": bucket,
            "LAVA_SYSTEM_CONTRACT": identity,
            "LAVA_SOURCE_SHA256": source_sha,
        },
        "Tags": [
            {"Key": "Project", "Value": "lava-docvqa"},
            {"Key": "Phase", "Value": "retrieved-evidence"},
        ],
    }


def describe_or_none(client: Any, name: str) -> dict[str, Any] | None:
    """Only genuine missing resources are misses; auth/network failures must propagate."""
    try:
        return client.describe_training_job(TrainingJobName=name)
    except ClientError as error:
        details = error.response.get("Error", {})
        message = details.get("Message", "").casefold()
        if details.get("Code") in {"ResourceNotFound", "ResourceNotFoundException"} or (
            details.get("Code") == "ValidationException"
            and (
                message.strip().rstrip(".") == "requested resource not found"
                or "could not find" in message
                or "does not exist" in message
            )
        ):
            return None
        raise


def validate_remote(description: dict[str, Any], request: dict[str, Any]) -> None:
    """Reattachment is allowed only to this exact computation, never a name-only match."""
    for name in (
        "TrainingJobName",
        "RoleArn",
        "ResourceConfig",
        "StoppingCondition",
        "Environment",
    ):
        observed = description.get(name, {})
        expected = request[name]
        if isinstance(expected, dict):
            if any(observed.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Existing training job differs in {name}")
        elif observed != expected:
            raise ValueError(f"Existing training job differs in {name}")
    for key in ("TrainingImage", "TrainingInputMode", "ContainerEntrypoint", "ContainerArguments"):
        if description["AlgorithmSpecification"].get(key) != request["AlgorithmSpecification"][key]:
            raise ValueError("Existing training job has different source or image")
    if (
        description["InputDataConfig"][0]["DataSource"]
        != request["InputDataConfig"][0]["DataSource"]
    ):
        raise ValueError("Existing training job has different source data")


def ensure_job(
    client: Any,
    request: dict[str, Any],
    *,
    acknowledge_charges: str,
    allow_retry: bool,
    attempt: int,
    logger: RuntimeEventLogger,
) -> dict[str, Any]:
    """Reattach safely; creating another paid attempt always requires explicit opt-in."""
    name = request["TrainingJobName"]
    existing = describe_or_none(client, name)
    if existing is not None:
        validate_remote(existing, request)
        logger.emit("system.job.reattached", job_name=name, status=existing["TrainingJobStatus"])
        return existing
    if acknowledge_charges != "YES":
        raise ValueError("A new GPU job requires CHARGES=YES")
    if attempt > 1 and not allow_retry:
        raise ValueError("A new paid retry also requires RETRY=YES")
    # Paginate the complete active project inventory, not just the first page.
    for state in ("InProgress", "Stopping"):
        for page in client.get_paginator("list_training_jobs").paginate(StatusEquals=state):
            if any(
                "lava" in row["TrainingJobName"].lower() for row in page["TrainingJobSummaries"]
            ):
                raise ValueError("Another LAVA training job is active; no new job created")
    try:
        client.create_training_job(**request)
    except (BotoCoreError, ClientError):
        # A timeout does not establish that AWS rejected the request. Never change its name.
        existing = describe_or_none(client, name)
        if existing is None:
            raise
        validate_remote(existing, request)
        logger.emit("system.submission.recovered", job_name=name)
        return existing
    logger.emit("system.job.submitted", job_name=name)
    return {"TrainingJobName": name, "TrainingJobStatus": "InProgress"}


def stream_progress(logs: Any, name: str, logger: RuntimeEventLogger, seen: set[str]) -> None:
    """Expose safe model progress, not raw generations, prompts or private PDF text."""
    try:
        pages = logs.get_paginator("filter_log_events").paginate(
            logGroupName="/aws/sagemaker/TrainingJobs",
            logStreamNamePrefix=name,
            filterPattern='?"system.question" ?"system.job" ?"DEPENDENCIES_"',
        )
        for page in pages:
            for event in page.get("events", []):
                if event["eventId"] in seen:
                    continue
                seen.add(event["eventId"])
                message = event["message"].strip()
                try:
                    record = json.loads(message)
                except json.JSONDecodeError:
                    if "DEPENDENCIES_" in message and message.startswith("["):
                        logger.emit("system.dependencies.progress", message=message[:256])
                    continue
                if not isinstance(record, dict) or not str(record.get("event", "")).startswith(
                    "system."
                ):
                    continue
                safe = {
                    key: record[key]
                    for key in (
                        "completed",
                        "total",
                        "reused_count",
                        "new_count",
                        "schema_valid",
                        "stage_elapsed_seconds",
                    )
                    if key in record
                }
                logger.emit(
                    "system.remote.progress",
                    remote_event=record["event"],
                    remote_timestamp_utc=record.get("timestamp_utc"),
                    remote_elapsed_seconds=record.get("elapsed_seconds"),
                    **safe,
                )
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code")
        if code == "ResourceNotFoundException":
            return
        if code == "AccessDeniedException":
            logger.emit(
                "system.progress.status_only",
                reason="CloudWatch read permission denied; job status monitoring continues",
            )
            return
        raise


ARCHIVE_PATHS = ("src", "configs", "pipelines", "scripts", "uv.lock", "pyproject.toml")


def verify_committed_source(root: Path) -> None:
    """Reject uncommitted inference source, while allowing local notebook controls/outputs."""
    tracked = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD", "--", *ARCHIVE_PATHS], cwd=root, text=True
    )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *ARCHIVE_PATHS],
        cwd=root,
        text=True,
    )
    if tracked.strip() or untracked.strip():
        raise ValueError("Commit inference source/configuration changes before immutable GPU work")


def execute_system(
    root: Path,
    session: Any,
    bucket: str,
    region: str,
    logger: RuntimeEventLogger,
    *,
    acknowledge_charges: str,
    attempt: int = 1,
    allow_retry: bool = False,
    hourly_usd_ceiling: float = 5.0,
    maximum_training_usd: float = 5.0,
    instance_type: str = "ml.g6e.2xlarge",
) -> dict[str, Any]:
    """Run the original frozen 16-question pilot without changing its saved contract."""
    return execute_prepared(
        root,
        session,
        bucket,
        region,
        logger,
        contract=system_contract(root),
        acknowledge_charges=acknowledge_charges,
        attempt=attempt,
        allow_retry=allow_retry,
        hourly_usd_ceiling=hourly_usd_ceiling,
        maximum_training_usd=maximum_training_usd,
        instance_type=instance_type,
    )


def execute_prepared(
    root: Path,
    session: Any,
    bucket: str,
    region: str,
    logger: RuntimeEventLogger,
    *,
    contract: dict[str, Any],
    request_factory: Callable[..., dict[str, Any]] = training_request,
    manifest_validator: Callable[..., Any] = validate_manifest,
    receipt_directory: str = "reports/system",
    acknowledge_charges: str,
    attempt: int = 1,
    allow_retry: bool = False,
    hourly_usd_ceiling: float = 5.0,
    maximum_training_usd: float = 5.0,
    instance_type: str = "ml.g6e.2xlarge",
) -> dict[str, Any]:
    """Submit or reattach, monitor visibly, then independently verify durable inference."""
    identity = contract["contract_id"]
    if instance_type not in SYSTEM_INSTANCES:
        raise ValueError("System execution requires an approved single 48-GB GPU instance")
    s3, client = session.client("s3"), session.client("sagemaker")
    store = store_for(s3, bucket, identity)
    manifest = store.read("inputs.json")
    if manifest is None:
        raise ValueError("Prepare reader inputs before submitting")
    manifest_validator(manifest, contract)
    if store.read("inference.json") is not None:
        logger.emit("system.inference.already_complete", new_gpu_job=False)
        return {"new_gpu_job": False, "status": "durable_inference_complete"}
    for value in (hourly_usd_ceiling, maximum_training_usd):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("Compute budget limits must be finite and positive")
    estimate = hourly_usd_ceiling * contract["config"]["max_runtime_seconds"] / 3600 * 1.25
    if estimate > maximum_training_usd:
        raise ValueError("Estimated training compute exceeds the acknowledged per-attempt budget")
    logger.emit(
        "system.cost_guard",
        hourly_usd_ceiling=hourly_usd_ceiling,
        estimated_training_usd=estimate,
        maximum_training_usd=maximum_training_usd,
        scope="Per attempt only; not an AWS account billing cap",
    )
    verify_committed_source(root)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    source = subprocess.check_output(
        ["git", "archive", "--format=tar.gz", "HEAD", *ARCHIVE_PATHS],
        cwd=root,
    )
    source_sha = digest(source)
    source_key = f"experiments/oracle-reader/system-source/{source_sha}/source.tar.gz"
    # This prefix already allows ListBucket for the SageMaker input channel.
    put_blob(s3, bucket, source_key, source, "application/gzip")
    from sagemaker.core.helper.session_helper import (  # type: ignore[import-untyped]
        get_execution_role,
    )

    role = get_execution_role()
    request = request_factory(
        root, manifest, bucket, region, role, attempt, source_key, source_sha, commit
    )
    request["ResourceConfig"]["InstanceType"] = instance_type
    request_key = f"jobs/attempt-{attempt}/request.json"
    stored_request = store.read(request_key)
    if stored_request is not None:
        # A report-only commit must reattach using the original source archive/commit.
        expected_name = request["TrainingJobName"]
        request = stored_request
        if request["TrainingJobName"] != expected_name or request["RoleArn"] != role:
            raise ValueError("Stored launch identity does not match this execution")
        if request["ResourceConfig"]["InstanceType"] != instance_type:
            raise ValueError("A saved attempt cannot change instance type; use an explicit retry")
    else:
        verify_training_quota(
            service_quotas=session.client("service-quotas"),
            instance_type=instance_type,
            instance_count=1,
            managed_spot=False,
        )
        store.write(request_key, request)
        if store.read(request_key) != request:
            raise ValueError("Launch intent failed durable read-back")
    description = ensure_job(
        client,
        request,
        acknowledge_charges=acknowledge_charges,
        allow_retry=allow_retry,
        attempt=attempt,
        logger=logger,
    )
    started = time.monotonic()
    monitor_ceiling = (
        sum(contract["config"][key] for key in ("max_pending_seconds", "max_runtime_seconds")) + 900
    )
    seen_events: set[str] = set()
    logs = session.client("logs")
    while description["TrainingJobStatus"] not in TERMINAL:
        if time.monotonic() - started > monitor_ceiling:
            raise TimeoutError("Monitor timed out; rerun to reattach. AWS job was not cancelled.")
        logger.emit(
            "system.job.heartbeat",
            job_name=request["TrainingJobName"],
            status=description["TrainingJobStatus"],
            secondary_status=description.get("SecondaryStatus"),
            monitor_elapsed_seconds=round(time.monotonic() - started, 3),
        )
        stream_progress(logs, request["TrainingJobName"], logger, seen_events)
        time.sleep(15)
        description = client.describe_training_job(TrainingJobName=request["TrainingJobName"])
    validate_remote(description, request)
    stream_progress(logs, request["TrainingJobName"], logger, seen_events)
    receipt = {
        "job_name": request["TrainingJobName"],
        "status": description["TrainingJobStatus"],
        "instance_type": description["ResourceConfig"]["InstanceType"],
        "billable_seconds": description.get("BillableTimeInSeconds"),
        "training_seconds": description.get("TrainingTimeInSeconds"),
        "creation_time": str(description.get("CreationTime")),
        "completion_time": str(description.get("TrainingEndTime")),
        "source_commit": request["Environment"]["LAVA_GIT_COMMIT_SHA"],
        "request_sha256": digest(encode(request)),
    }
    store.write(f"jobs/attempt-{attempt}/receipt.json", receipt)
    if store.read(f"jobs/attempt-{attempt}/receipt.json") != receipt:
        raise ValueError("Job receipt failed durable read-back")
    receipt_folder = receipt_directory if receipt["status"] == "Completed" else "artifacts/system"
    atomic_write(root / receipt_folder / f"attempt-{attempt}.json", encode(receipt))
    logger.emit("system.job.terminal", **receipt)
    if receipt["status"] != "Completed":
        raise RuntimeError(
            "GPU attempt did not complete. Saved answers remain reusable; a new attempt needs ATTEMPT and RETRY=YES. No automatic paid retry."
        )
    inference = store.read("inference.json")
    if inference is None:
        raise ValueError("AWS completion alone is insufficient: missing durable inference")
    return receipt
