"""Reconnectable SageMaker training-job monitoring with safe heartbeats."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from botocore.exceptions import ClientError  # type: ignore[import-untyped]

from lava.observability.events import EventLogger, sanitize_value

_TERMINAL = frozenset({"Completed", "Failed", "Stopped"})


class TrainingJobFailedError(RuntimeError):
    """Raised when a SageMaker training job ends unsuccessfully."""


class TrainingJobMonitorTimeoutError(TimeoutError):
    """Raised when the local monitor exceeds its bounded wait time."""


class TrainingJobPendingTimeoutError(TrainingJobMonitorTimeoutError):
    """Raised when SageMaker remains Pending beyond the capacity-wait limit."""


@dataclass(frozen=True, slots=True)
class TrainingJobSnapshot:
    """Sanitized subset of a SageMaker training-job description."""

    job_name: str
    status: str
    secondary_status: str | None
    training_time_seconds: int | None
    billable_time_seconds: int | None
    failure_reason: str | None

    @property
    def terminal(self) -> bool:
        """Return whether the job reached a terminal state."""
        return self.status in _TERMINAL

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "job_name": self.job_name,
            "status": self.status,
            "secondary_status": self.secondary_status,
            "training_time_seconds": self.training_time_seconds,
            "billable_time_seconds": self.billable_time_seconds,
            "failure_reason": self.failure_reason,
            "terminal": self.terminal,
        }


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def snapshot_from_description(description: Mapping[str, object]) -> TrainingJobSnapshot:
    """Create a safe snapshot from ``DescribeTrainingJob`` output."""
    job_name = description.get("TrainingJobName")
    status = description.get("TrainingJobStatus")
    if not isinstance(job_name, str) or not isinstance(status, str):
        message = "Training-job description lacks a valid name or status."
        raise TypeError(message)
    secondary = description.get("SecondaryStatus")
    failure = description.get("FailureReason")
    return TrainingJobSnapshot(
        job_name=job_name,
        status=status,
        secondary_status=secondary if isinstance(secondary, str) else None,
        training_time_seconds=_optional_int(description.get("TrainingTimeInSeconds")),
        billable_time_seconds=_optional_int(description.get("BillableTimeInSeconds")),
        failure_reason=failure if isinstance(failure, str) else None,
    )


def discover_new_training_job(
    *,
    sagemaker_client: Any,
    created_after: datetime,
    known_job_names: set[str],
    name_contains: str = "oracle",
) -> str | None:
    """Discover the newest matching training job created after a known boundary."""
    response = sagemaker_client.list_training_jobs(
        CreationTimeAfter=created_after.astimezone(UTC),
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=100,
    )
    summaries = response.get("TrainingJobSummaries", [])
    if not isinstance(summaries, list):
        return None
    for summary in summaries:
        if not isinstance(summary, Mapping):
            continue
        name = summary.get("TrainingJobName")
        if not isinstance(name, str):
            continue
        if name in known_job_names:
            continue
        if name_contains and name_contains.lower() not in name.lower():
            continue
        return name
    return None


def list_training_job_names(
    *,
    sagemaker_client: Any,
    statuses: Iterable[str] = ("InProgress", "Stopping"),
) -> set[str]:
    """List current training-job names for the supplied statuses."""
    names: set[str] = set()
    for status in statuses:
        response = sagemaker_client.list_training_jobs(
            StatusEquals=status,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=100,
        )
        summaries = response.get("TrainingJobSummaries", [])
        if not isinstance(summaries, list):
            continue
        for summary in summaries:
            if isinstance(summary, Mapping):
                name = summary.get("TrainingJobName")
                if isinstance(name, str):
                    names.add(name)
    return names


@dataclass(slots=True)
class SageMakerTrainingMonitor:
    """Poll SageMaker safely and emit status changes plus visible heartbeats."""

    sagemaker_client: Any
    logger: EventLogger
    poll_seconds: float = 15.0
    heartbeat_seconds: float = 30.0
    max_monitor_seconds: float = 3900.0
    max_pending_seconds: float | None = None
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.perf_counter

    def __post_init__(self) -> None:
        """Validate bounded monitor timing."""
        for name, value in (
            ("poll_seconds", self.poll_seconds),
            ("heartbeat_seconds", self.heartbeat_seconds),
            ("max_monitor_seconds", self.max_monitor_seconds),
        ):
            if value <= 0:
                message = f"{name} must be greater than zero."
                raise ValueError(message)
        if self.max_pending_seconds is not None and self.max_pending_seconds <= 0:
            message = "max_pending_seconds must be greater than zero when supplied."
            raise ValueError(message)

    def describe(self, job_name: str) -> TrainingJobSnapshot:
        """Return a sanitized training-job snapshot."""
        description = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)
        return snapshot_from_description(description)

    def wait(self, job_name: str, *, stop_on_timeout: bool = True) -> TrainingJobSnapshot:
        """Wait for completion while emitting status and heartbeat events."""
        started = self.monotonic()
        next_heartbeat = started
        last_status: tuple[str, str | None] | None = None
        pending_started = started
        pending_observed = False
        pending_cleared = False
        self.logger.emit(
            "sagemaker.monitor.started",
            job_name=job_name,
            max_monitor_seconds=self.max_monitor_seconds,
            max_pending_seconds=self.max_pending_seconds,
        )
        while True:
            now = self.monotonic()
            elapsed = max(0.0, now - started)
            if elapsed > self.max_monitor_seconds:
                if stop_on_timeout:
                    self._stop_job(job_name)
                self.logger.emit(
                    "sagemaker.monitor.timeout",
                    level="ERROR",
                    job_name=job_name,
                    monitor_elapsed_seconds=round(elapsed, 3),
                    stop_requested=stop_on_timeout,
                )
                message = f"Monitoring timed out for training job {job_name!r}."
                raise TrainingJobMonitorTimeoutError(message)

            snapshot = self.describe(job_name)

            if not pending_cleared:
                if snapshot.secondary_status == "Pending":
                    pending_observed = True
                    pending_elapsed = max(0.0, now - pending_started)
                    if (
                        self.max_pending_seconds is not None
                        and pending_elapsed > self.max_pending_seconds
                    ):
                        if stop_on_timeout:
                            self._stop_job(job_name)
                        self.logger.emit(
                            "sagemaker.pending.timeout",
                            level="ERROR",
                            job_name=job_name,
                            pending_elapsed_seconds=round(pending_elapsed, 3),
                            max_pending_seconds=self.max_pending_seconds,
                            stop_requested=stop_on_timeout,
                        )
                        message = (
                            f"Training job {job_name!r} remained Pending beyond "
                            f"{self.max_pending_seconds:.3f} seconds."
                        )
                        raise TrainingJobPendingTimeoutError(message)
                else:
                    pending_cleared = True
                    if pending_observed:
                        pending_elapsed = max(0.0, now - pending_started)
                        self.logger.emit(
                            "sagemaker.pending.cleared",
                            job_name=job_name,
                            pending_elapsed_seconds=round(pending_elapsed, 3),
                            secondary_status=snapshot.secondary_status,
                        )

            state = (snapshot.status, snapshot.secondary_status)
            if state != last_status:
                self.logger.emit(
                    "sagemaker.status.changed",
                    job_name=snapshot.job_name,
                    status=snapshot.status,
                    secondary_status=snapshot.secondary_status,
                    training_time_seconds=snapshot.training_time_seconds,
                    billable_time_seconds=snapshot.billable_time_seconds,
                    failure_reason=snapshot.failure_reason,
                    terminal=snapshot.terminal,
                )
                last_status = state
            if now >= next_heartbeat:
                self.logger.emit(
                    "sagemaker.status.heartbeat",
                    job_name=job_name,
                    status=snapshot.status,
                    secondary_status=snapshot.secondary_status,
                    monitor_elapsed_seconds=round(elapsed, 3),
                )
                next_heartbeat = now + self.heartbeat_seconds
            if snapshot.terminal:
                level = "INFO" if snapshot.status == "Completed" else "ERROR"
                self.logger.emit(
                    "sagemaker.monitor.finished",
                    level=level,
                    job_name=snapshot.job_name,
                    status=snapshot.status,
                    secondary_status=snapshot.secondary_status,
                    training_time_seconds=snapshot.training_time_seconds,
                    billable_time_seconds=snapshot.billable_time_seconds,
                    failure_reason=snapshot.failure_reason,
                    terminal=snapshot.terminal,
                )
                if snapshot.status != "Completed":
                    message = (
                        f"Training job {job_name!r} ended with status {snapshot.status!r}. "
                        f"Reason: {snapshot.failure_reason or 'not supplied'}"
                    )
                    raise TrainingJobFailedError(message)
                return snapshot
            self.sleep(self.poll_seconds)

    def _stop_job(self, job_name: str) -> None:
        try:
            self.sagemaker_client.stop_training_job(TrainingJobName=job_name)
        except ClientError as exc:
            self.logger.emit(
                "sagemaker.stop.failed",
                level="ERROR",
                job_name=job_name,
                error_code=exc.response.get("Error", {}).get("Code"),
            )


def sanitize_cloudwatch_line(line: str) -> dict[str, object] | None:
    """Return only structured LAVA events; suppress arbitrary model output."""
    stripped = line.strip()
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if not {"event", "component", "timestamp_utc"}.issubset(payload):
        return None
    sanitized = sanitize_value(payload)
    return sanitized if isinstance(sanitized, dict) else None
