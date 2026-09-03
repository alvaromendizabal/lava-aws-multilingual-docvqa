"""Preview or submit the first charge-bounded oracle-reader GPU smoke job."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import boto3
from dotenv import load_dotenv

from lava.notebook_support import find_repo_root, git_snapshot, public_metadata
from lava.observability import (
    EventLogger,
    SageMakerTrainingMonitor,
    TrainingRunState,
    discover_new_training_job,
    enforce_cost_cap,
    estimate_maximum_cost,
    latest_state_path,
    list_training_job_names,
    validate_first_smoke_plan,
    write_state_atomic,
)
from lava.observability.events import format_utc, redact_string
from lava.readers.sagemaker import build_job_plan


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", default="qwen35_4b_fused_direct")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--acknowledge-charges", default="NO")
    parser.add_argument("--hourly-usd-ceiling", type=float, default=10.0)
    parser.add_argument("--maximum-total-usd", type=float, default=12.50)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--max-monitor-seconds", type=float, default=3900.0)
    return parser.parse_args()


def _stream_process_output(process: subprocess.Popen[str], logger: EventLogger) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        logger.emit("launcher.output", message=redact_string(line.rstrip()))


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
            bucket=bucket,
            limit=args.limit,
        )
        plan = plan_model.model_dump(mode="json")
        validate_first_smoke_plan(plan)
        estimate = estimate_maximum_cost(
            hourly_usd_ceiling=args.hourly_usd_ceiling,
            max_runtime_seconds=int(plan["max_runtime_seconds"]),
            instance_count=int(plan["instance_count"]),
            contingency_factor=1.25,
        )
        enforce_cost_cap(estimate, maximum_allowed_usd=args.maximum_total_usd)
        logger.emit(
            "smoke.plan.verified",
            cost_guard=estimate.as_dict(),
            plan=public_metadata(plan),
            git=git_snapshot(root),
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
            prefix=str(plan["output_s3_prefix"]),
        )

        created_at = datetime.now(tz=UTC)
        known_names = list_training_job_names(
            sagemaker_client=sagemaker_client,
            statuses=("InProgress", "Completed", "Failed", "Stopped", "Stopping"),
        )
        _save_state(
            root=root,
            run_id=run_id,
            job_name=None,
            plan=plan,
            status="launching",
            created_at=created_at,
        )

        command = [
            sys.executable,
            str(root / "scripts" / "launch_oracle_reader.py"),
            "--model-key",
            args.model_key,
            "--limit",
            str(args.limit),
            "--submit",
            "--wait",
            "--acknowledge-charges",
            "YES",
        ]
        logger.emit(
            "launcher.process.started", command=["python", "scripts/launch_oracle_reader.py", "..."]
        )
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_thread = threading.Thread(
            target=_stream_process_output,
            args=(process, logger),
            name="lava-launch-output",
            daemon=True,
        )
        output_thread.start()

        job_name: str | None = None
        discovery_deadline = time.monotonic() + 300.0
        while job_name is None and process.poll() is None and time.monotonic() < discovery_deadline:
            job_name = discover_new_training_job(
                sagemaker_client=sagemaker_client,
                created_after=created_at,
                known_job_names=known_names,
                name_contains="",
            )
            if job_name is None:
                logger.emit("launcher.job_discovery.heartbeat")
                time.sleep(5.0)

        if job_name is None:
            return_code = process.wait(timeout=30)
            message = (
                f"Unable to discover submitted SageMaker job; launcher exit code={return_code}."
            )
            raise RuntimeError(message)

        logger.emit("launcher.job.discovered", job_name=job_name)
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
            max_monitor_seconds=args.max_monitor_seconds,
        )
        try:
            snapshot = monitor.wait(job_name, stop_on_timeout=True)
        except KeyboardInterrupt:
            logger.emit(
                "monitor.detached",
                level="WARNING",
                job_name=job_name,
                reconnect_command=(
                    f"uv run python scripts/monitor_oracle_reader_job.py --job-name {job_name}"
                ),
            )
            raise
        finally:
            output_thread.join(timeout=5.0)

        return_code = process.wait(timeout=60)
        if return_code != 0:
            message = f"Underlying launcher exited with code {return_code}."
            raise RuntimeError(message)
        _save_state(
            root=root,
            run_id=run_id,
            job_name=job_name,
            plan=plan,
            status=snapshot.status,
            created_at=created_at,
        )
        logger.emit("smoke.submit.complete", **snapshot.as_dict())
        print("ORACLE_READER_ONE_QUESTION_SMOKE_COMPLETED")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
