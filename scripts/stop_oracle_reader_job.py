"""Safely stop the latest or a named SageMaker oracle-reader training job."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import boto3
from dotenv import load_dotenv

from lava.notebook_support import find_repo_root
from lava.observability import latest_state_path, read_state, write_state_atomic
from lava.observability.events import format_utc
from lava.observability.sagemaker_monitor import TrainingJobSnapshot, snapshot_from_description


def parse_args() -> argparse.Namespace:
    """Parse stop-command arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name")
    parser.add_argument("--confirm", default="NO")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    parser.add_argument("--max-wait-seconds", type=float, default=300.0)
    return parser.parse_args()


def _print_snapshot(
    event: str,
    snapshot: TrainingJobSnapshot,
    *,
    started_monotonic: float,
) -> None:
    payload = {
        "timestamp_utc": format_utc(datetime.now(tz=UTC)),
        "elapsed_seconds": round(max(0.0, time.monotonic() - started_monotonic), 3),
        "event": event,
        **snapshot.as_dict(),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def main() -> int:
    """Request a stop and wait for the existing job to become terminal."""
    command_started = time.monotonic()
    args = parse_args()
    if args.confirm != "YES":
        raise RuntimeError("Stopping a job requires --confirm YES.")
    for name, value in (
        ("poll_seconds", args.poll_seconds),
        ("heartbeat_seconds", args.heartbeat_seconds),
        ("max_wait_seconds", args.max_wait_seconds),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")

    root = find_repo_root(Path(__file__).resolve())
    os.chdir(root)
    load_dotenv(root / ".env", override=False)
    region = os.environ.get("AWS_REGION", "us-west-2")
    state_path = latest_state_path(root)
    state = read_state(state_path) if state_path.is_file() else None
    job_name = args.job_name or (state.job_name if state is not None else None)
    if not job_name:
        raise RuntimeError("No job name was supplied and no submitted latest job was found.")
    if not job_name.startswith("lava-oracle-"):
        raise RuntimeError(
            f"Refusing to stop non-project SageMaker job {job_name!r}; expected prefix 'lava-oracle-'."
        )

    client = boto3.session.Session(region_name=region).client("sagemaker")
    before = snapshot_from_description(client.describe_training_job(TrainingJobName=job_name))
    _print_snapshot("stop.before", before, started_monotonic=command_started)

    if before.terminal:
        if state is not None and state.job_name == job_name:
            write_state_atomic(
                state_path,
                replace(
                    state,
                    status=before.status,
                    updated_at_utc=format_utc(datetime.now(tz=UTC)),
                ),
            )
        print("STOP_NOT_REQUIRED_JOB_ALREADY_TERMINAL")
        print(f"STOP_TOTAL_ELAPSED_SECONDS={time.monotonic() - command_started:.3f}")
        return 0
    if before.status == "InProgress":
        client.stop_training_job(TrainingJobName=job_name)
        print("SAGEMAKER_STOP_REQUEST_ACCEPTED", flush=True)
    elif before.status != "Stopping":
        raise RuntimeError(f"Refusing to stop unexpected job status {before.status!r}.")

    started = time.monotonic()
    next_heartbeat = started
    last_state: tuple[str, str | None] | None = None
    try:
        while True:
            now = time.monotonic()
            elapsed = now - started
            if elapsed > args.max_wait_seconds:
                raise TimeoutError(
                    f"Job {job_name!r} did not become terminal within "
                    f"{args.max_wait_seconds:.1f} seconds."
                )
            snapshot = snapshot_from_description(
                client.describe_training_job(TrainingJobName=job_name)
            )
            current_state = (snapshot.status, snapshot.secondary_status)
            if current_state != last_state:
                _print_snapshot("stop.status.changed", snapshot, started_monotonic=command_started)
                last_state = current_state
            if now >= next_heartbeat:
                _print_snapshot("stop.heartbeat", snapshot, started_monotonic=command_started)
                next_heartbeat = now + args.heartbeat_seconds
            if snapshot.terminal:
                if state is not None and state.job_name == job_name:
                    write_state_atomic(
                        state_path,
                        replace(
                            state,
                            status=snapshot.status,
                            updated_at_utc=format_utc(datetime.now(tz=UTC)),
                        ),
                    )
                _print_snapshot("stop.complete", snapshot, started_monotonic=command_started)
                print("SAGEMAKER_JOB_TERMINAL_AFTER_STOP_REQUEST")
                print(f"STOP_TOTAL_ELAPSED_SECONDS={time.monotonic() - command_started:.3f}")
                return 0
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("STOP_MONITOR_DETACHED_STOP_REQUEST_REMAINS_ACTIVE")
        print(f"STOP_TOTAL_ELAPSED_SECONDS={time.monotonic() - command_started:.3f}")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
