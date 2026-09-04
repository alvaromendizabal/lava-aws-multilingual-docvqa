"""Preview or submit one charge-bounded, capacity-safe oracle-reader GPU smoke job."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

from lava.notebook_support import find_repo_root, git_snapshot, public_metadata
from lava.observability import (
    EventLogger,
    SageMakerTrainingMonitor,
    TrainingRunState,
    enforce_cost_cap,
    estimate_maximum_cost,
    latest_state_path,
    list_training_job_names,
    validate_first_smoke_plan,
    verify_training_quota,
    write_state_atomic,
)
from lava.observability.events import format_utc
from lava.readers.artifact_gate import verify_training_model_artifact
from lava.readers.sagemaker import build_job_plan, submit_or_preview_job

_MONITOR_SAFETY_MARGIN_SECONDS = 900.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", default="qwen35_4b_fused_direct")
    parser.add_argument("--instance-type")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--acknowledge-charges", default="NO")
    parser.add_argument("--hourly-usd-ceiling", type=float, default=10.0)
    parser.add_argument("--maximum-total-usd", type=float, default=12.50)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument(
        "--max-monitor-seconds",
        type=float,
        help=(
            "Local monitor ceiling. Defaults to the cloud pending limit plus the "
            "runtime limit and a 15-minute cleanup margin."
        ),
    )
    return parser.parse_args()


def _assert_output_prefix_empty(*, s3_client: object, bucket: str, prefix: str) -> None:
    key_prefix = prefix.split(f"s3://{bucket}/", maxsplit=1)[-1].rstrip("/") + "/"
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=key_prefix, MaxKeys=1)  # type: ignore[attr-defined]
    if int(response.get("KeyCount", 0)) > 0:
        message = (
            "The immutable output prefix already contains objects. Commit a new code revision before "
            "submitting another smoke job."
        )
        raise RuntimeError(message)


def _save_state(
    *,
    root: Path,
    run_id: str,
    job_name: str | None,
    plan: dict[str, object],
    status: str,
    created_at: datetime,
) -> None:
    now = datetime.now(tz=UTC)
    state = TrainingRunState(
        schema_version=1,
        run_id=run_id,
        job_name=job_name,
        model_key=str(plan["model_key"]),
        git_commit_sha=str(plan["git_commit_sha"]),
        protocol_lock_id=str(plan["protocol_lock_id"]),
        output_s3_prefix=str(plan["output_s3_prefix"]),
        created_at_utc=format_utc(created_at),
        updated_at_utc=format_utc(now),
        status=status,
    )
    write_state_atomic(latest_state_path(root), state)
    write_state_atomic(
        root / "artifacts" / "oracle_reader" / "runtime" / run_id / "state.json",
        state,
    )


def _monitor_ceiling_seconds(
    *,
    requested: float | None,
    cloud_pending_seconds: int,
    runtime_seconds: int,
) -> float:
    minimum = float(cloud_pending_seconds + runtime_seconds) + _MONITOR_SAFETY_MARGIN_SECONDS
    if requested is None:
        return minimum
    if requested < minimum:
        message = (
            "--max-monitor-seconds is too short for the server-enforced bounds: "
            f"requested={requested}, required_at_least={minimum}. A short local timeout would "
            "re-create the premature capacity cancellation this command is designed to prevent."
        )
        raise ValueError(message)
    return requested


def _reconnect_command(job_name: str) -> str:
    return f"uv run python scripts/monitor_oracle_reader_job.py --job-name {job_name}"


def main() -> int:
    """Preview or submit one bounded GPU job and monitor it visibly."""
    args = parse_args()
    root = find_repo_root(Path(__file__).resolve())
    os.chdir(root)
    load_dotenv(root / ".env", override=False)
    bucket = os.environ.get("S3_BUCKET")
    region = os.environ.get("AWS_REGION", "us-west-2")
    if not bucket:
        message = "S3_BUCKET is missing from the project environment."
        raise RuntimeError(message)

    run_id = f"oracle-smoke-{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    logger = EventLogger.to_stdout(
        run_id=run_id,
        component="oracle_reader.smoke",
        jsonl_path=root / "artifacts" / "oracle_reader" / "runtime" / run_id / "events.jsonl",
        static_context={"model_key": args.model_key},
    )

    with logger.stage("smoke_command", heartbeat_seconds=args.heartbeat_seconds):
        plan_model = build_job_plan(
            repo_root=root,
            config_path=root / "configs" / "oracle_reader_benchmark.yaml",
            model_lock_path=root / "configs" / "oracle_reader_models.lock.json",
            model_key=args.model_key,
            instance_type=args.instance_type,
            bucket=bucket,
            limit=args.limit,
        )
        plan = plan_model.model_dump(mode="json")
        validate_first_smoke_plan(plan)

        monitor_ceiling = _monitor_ceiling_seconds(
            requested=args.max_monitor_seconds,
            cloud_pending_seconds=plan_model.max_pending_seconds,
            runtime_seconds=plan_model.max_runtime_seconds,
        )
        estimate = estimate_maximum_cost(
            hourly_usd_ceiling=args.hourly_usd_ceiling,
            max_runtime_seconds=plan_model.max_runtime_seconds,
            instance_count=plan_model.instance_count,
            contingency_factor=1.25,
        )
        enforce_cost_cap(estimate, maximum_allowed_usd=args.maximum_total_usd)
        logger.emit(
            "smoke.plan.verified",
            cost_guard=estimate.as_dict(),
            plan=public_metadata(plan),
            git=git_snapshot(root),
            monitor_ceiling_seconds=monitor_ceiling,
        )
        print(json.dumps(public_metadata(plan), indent=2, sort_keys=True))

        if not args.submit:
            logger.emit("smoke.preview.complete", paid_resource_created=False)
            print("SMOKE_PLAN_PREVIEWED")
            print("NO_PAID_SAGEMAKER_RESOURCE_WAS_CREATED")
            return 0
        if not args.wait:
            message = "Paid submission requires --wait so the bounded monitor remains attached."
            raise RuntimeError(message)
        if args.acknowledge_charges != "YES":
            message = "Paid submission requires --acknowledge-charges YES."
            raise RuntimeError(message)
        if not git_snapshot(root)["working_tree_clean"]:
            message = "Commit all code and lock files before paid submission."
            raise RuntimeError(message)

        session = boto3.session.Session(region_name=region)
        sagemaker_client = session.client("sagemaker")
        s3_client = session.client("s3")
        service_quotas = session.client("service-quotas")
        quota = verify_training_quota(
            service_quotas=service_quotas,
            instance_type=plan_model.instance_type,
            instance_count=plan_model.instance_count,
            managed_spot=plan_model.managed_spot,
        )
        logger.emit("smoke.quota.verified", quota=quota)

        active = sorted(
            name
            for name in list_training_job_names(sagemaker_client=sagemaker_client)
            if "oracle" in name.lower() or "lava" in name.lower()
        )
        if active:
            message = f"Another LAVA/Oracle training job is active: {active}"
            raise RuntimeError(message)
        _assert_output_prefix_empty(
            s3_client=s3_client,
            bucket=bucket,
            prefix=plan_model.output_s3_prefix,
        )

        created_at = datetime.now(tz=UTC)
        _save_state(
            root=root,
            run_id=run_id,
            job_name=None,
            plan=plan,
            status="launching",
            created_at=created_at,
        )

        submission = submit_or_preview_job(
            plan=plan_model,
            repo_root=root,
            region=region,
            submit=True,
            wait=False,
            acknowledgement="YES",
        )
        raw_job_name = submission.get("training_job_name")
        if not isinstance(raw_job_name, str) or not raw_job_name:
            raise RuntimeError("SageMaker submitted a job but returned no training job name.")
        job_name = raw_job_name
        logger.emit(
            "sagemaker.job.submitted",
            job_name=job_name,
            instance_type=plan_model.instance_type,
            cloud_max_pending_seconds=plan_model.max_pending_seconds,
            cloud_max_runtime_seconds=plan_model.max_runtime_seconds,
            reconnect_command=_reconnect_command(job_name),
        )
        _save_state(
            root=root,
            run_id=run_id,
            job_name=job_name,
            plan=plan,
            status="InProgress",
            created_at=created_at,
        )

        monitor = SageMakerTrainingMonitor(
            sagemaker_client=sagemaker_client,
            logger=logger,
            poll_seconds=args.poll_seconds,
            heartbeat_seconds=args.heartbeat_seconds,
            max_monitor_seconds=monitor_ceiling,
            # SageMaker enforces MaxPendingTimeInSeconds in the job itself. Do not
            # maintain a second, shorter local pending clock that throws away queue position.
            max_pending_seconds=None,
        )
        try:
            snapshot = monitor.wait(job_name, stop_on_timeout=False)
        except KeyboardInterrupt:
            reconnect = _reconnect_command(job_name)
            _save_state(
                root=root,
                run_id=run_id,
                job_name=job_name,
                plan=plan,
                status="detached",
                created_at=created_at,
            )
            logger.emit(
                "monitor.detached",
                level="WARNING",
                job_name=job_name,
                cloud_job_continues=True,
                reconnect_command=reconnect,
            )
            print("MONITOR_DETACHED_CLOUD_JOB_CONTINUES_WITH_SERVER_SIDE_LIMITS")
            print(reconnect)
            return 130
        except Exception as monitor_error:
            reconnect = _reconnect_command(job_name)
            print("LOCAL_MONITOR_FAILED_CLOUD_JOB_MAY_STILL_BE_RUNNING")
            print(reconnect)
            try:
                failure_snapshot = monitor.describe(job_name)
            except (BotoCoreError, ClientError, TypeError) as describe_error:
                logger.emit(
                    "monitor.failure_state.unavailable",
                    level="ERROR",
                    job_name=job_name,
                    monitor_exception_type=type(monitor_error).__name__,
                    describe_exception_type=type(describe_error).__name__,
                )
            else:
                _save_state(
                    root=root,
                    run_id=run_id,
                    job_name=job_name,
                    plan=plan,
                    status=failure_snapshot.status,
                    created_at=created_at,
                )
                logger.emit(
                    "monitor.failure_state.saved",
                    level="ERROR",
                    monitor_exception_type=type(monitor_error).__name__,
                    snapshot=failure_snapshot.as_dict(),
                )
            raise

        _save_state(
            root=root,
            run_id=run_id,
            job_name=job_name,
            plan=plan,
            status=snapshot.status,
            created_at=created_at,
        )
        if snapshot.status != "Completed":
            message = f"SageMaker smoke ended with non-success status {snapshot.status!r}."
            raise RuntimeError(message)

        artifact_gate = verify_training_model_artifact(
            sagemaker_client=sagemaker_client,
            s3_client=s3_client,
            job_name=job_name,
            expected_output_s3_prefix=plan_model.output_s3_prefix,
        )
        logger.emit("smoke.artifact.verified", artifact_gate=artifact_gate.as_dict())
        logger.emit("smoke.submit.complete", snapshot=snapshot.as_dict())
        print("ORACLE_READER_ONE_QUESTION_SMOKE_COMPLETED")
        print("ORACLE_READER_ONE_QUESTION_SMOKE_VERIFIED")
        return 0

    # EventLogger.Stage never suppresses exceptions at runtime, but its generic context-manager
    # return type is intentionally broad. Keep an explicit terminal guard so static analysis and
    # future refactors cannot create an implicit None return from this command.
    raise RuntimeError("Smoke command exited its telemetry stage without a terminal result.")


if __name__ == "__main__":
    raise SystemExit(main())
