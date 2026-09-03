"""Reconnect to a previously submitted SageMaker oracle-reader job."""

from __future__ import annotations

import argparse
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import boto3
from dotenv import load_dotenv

from lava.notebook_support import find_repo_root
from lava.observability import EventLogger, SageMakerTrainingMonitor, latest_state_path, read_state


def parse_args() -> argparse.Namespace:
    """Parse monitor arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--max-monitor-seconds", type=float, default=3900.0)
    parser.add_argument("--max-pending-seconds", type=float)
    return parser.parse_args()


def main() -> int:
    """Monitor a named job or the latest saved state until terminal."""
    args = parse_args()
    root = find_repo_root(Path(__file__).resolve())
    os.chdir(root)
    load_dotenv(root / ".env", override=False)
    region = os.environ.get("AWS_REGION", "us-west-2")
    state = read_state(latest_state_path(root)) if args.job_name is None else None
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
    monitor = SageMakerTrainingMonitor(
        sagemaker_client=client,
        logger=logger,
        poll_seconds=args.poll_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
        max_monitor_seconds=args.max_monitor_seconds,
        max_pending_seconds=args.max_pending_seconds,
    )
    snapshot = monitor.wait(job_name, stop_on_timeout=False)
    logger.emit("monitor.reconnect.complete", **snapshot.as_dict())
    print("SAGEMAKER_JOB_MONITOR_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
