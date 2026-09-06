"""Reconnect to a previously submitted SageMaker oracle-reader job."""

from __future__ import annotations

import argparse
import os
import uuid
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import boto3
from dotenv import load_dotenv

from lava.notebook_support import find_repo_root
from lava.observability import (
    EventLogger,
    SageMakerTrainingMonitor,
    latest_state_path,
    read_state,
    write_state_atomic,
)
from lava.observability.events import format_utc

_MONITOR_SAFETY_MARGIN_SECONDS = 900.0


def parse_args() -> argparse.Namespace:
    """Parse monitor arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--max-monitor-seconds", type=float)
    return parser.parse_args()


def _positive_int(value: object, *, fallback: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return fallback
    parsed = int(value)
    return parsed if parsed > 0 else fallback


def _derived_monitor_ceiling(
    description: Mapping[str, object],
    *,
    requested: float | None,
) -> float:
    stopping = description.get("StoppingCondition")
    stopping_mapping = stopping if isinstance(stopping, Mapping) else {}
    pending = _positive_int(stopping_mapping.get("MaxPendingTimeInSeconds"), fallback=86400)
    runtime = _positive_int(stopping_mapping.get("MaxRuntimeInSeconds"), fallback=3600)
    minimum = float(pending + runtime) + _MONITOR_SAFETY_MARGIN_SECONDS
    if requested is None:
        return minimum
    if requested < minimum:
        raise ValueError(
            "--max-monitor-seconds is shorter than the job's cloud-side pending and runtime "
            f"bounds: requested={requested}, required_at_least={minimum}."
        )
    return requested


def main() -> int:
    """Monitor a named job or the latest saved state until terminal."""
    args = parse_args()
    root = find_repo_root(Path(__file__).resolve())
    os.chdir(root)
    load_dotenv(root / ".env", override=False)
    region = os.environ.get("AWS_REGION", "us-west-2")
    state_path = latest_state_path(root)
    state = read_state(state_path) if args.job_name is None and state_path.is_file() else None
    job_name = args.job_name or (state.job_name if state is not None else None)
    if not job_name:
        message = "No job name was supplied and the latest state has no submitted job."
        raise RuntimeError(message)

    run_id = (
        state.run_id
        if state is not None
        else (f"reconnect-{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}")
    )
    logger = EventLogger.to_stdout(
        run_id=run_id,
        component="oracle_reader.monitor",
        jsonl_path=root / "artifacts" / "oracle_reader" / "runtime" / run_id / "monitor.jsonl",
    )
    client = boto3.session.Session(region_name=region).client("sagemaker")
    description = client.describe_training_job(TrainingJobName=job_name)
    monitor_ceiling = _derived_monitor_ceiling(
        description,
        requested=args.max_monitor_seconds,
    )
    monitor = SageMakerTrainingMonitor(
        sagemaker_client=client,
        logger=logger,
        poll_seconds=args.poll_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
        max_monitor_seconds=monitor_ceiling,
        max_pending_seconds=None,
    )
    try:
        snapshot = monitor.wait(job_name, stop_on_timeout=False)
    except KeyboardInterrupt:
        logger.emit(
            "monitor.detached",
            level="WARNING",
            job_name=job_name,
            cloud_job_continues=True,
        )
        print("MONITOR_DETACHED_CLOUD_JOB_CONTINUES_WITH_SERVER_SIDE_LIMITS")
        print(f"uv run python scripts/monitor_oracle_reader_job.py --job-name {job_name}")
        return 130
    if state is not None and state.job_name == job_name:
        write_state_atomic(
            state_path,
            replace(
                state,
                status=snapshot.status,
                updated_at_utc=format_utc(datetime.now(tz=UTC)),
            ),
        )
    logger.emit("monitor.reconnect.complete", snapshot=snapshot.as_dict())
    if snapshot.status != "Completed":
        raise RuntimeError(
            f"SageMaker job {job_name!r} ended with non-success status {snapshot.status!r}."
        )
    print("SAGEMAKER_JOB_MONITOR_COMPLETE")
    print("SAGEMAKER_JOB_COMPLETED_SUCCESSFULLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
