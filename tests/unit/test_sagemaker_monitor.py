"""Tests for reconnectable SageMaker job discovery and monitoring."""

from __future__ import annotations

import io
from datetime import UTC, datetime

import pytest

from lava.observability.events import EventLogger
from lava.observability.sagemaker_monitor import (
    SageMakerTrainingMonitor,
    TrainingJobFailedError,
    TrainingJobMonitorTimeoutError,
    discover_new_training_job,
    sanitize_cloudwatch_line,
    snapshot_from_description,
)


class FakeSageMakerClient:
    """Minimal deterministic SageMaker client for unit tests."""

    def __init__(self, descriptions: list[dict[str, object]]) -> None:
        self.descriptions = descriptions
        self.index = 0
        self.stopped: list[str] = []

    def describe_training_job(self, *, TrainingJobName: str) -> dict[str, object]:
        result = dict(self.descriptions[min(self.index, len(self.descriptions) - 1)])
        result.setdefault("TrainingJobName", TrainingJobName)
        self.index += 1
        return result

    def list_training_jobs(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "TrainingJobSummaries": [
                {"TrainingJobName": "lava-oracle-new"},
                {"TrainingJobName": "unrelated-job"},
            ]
        }

    def stop_training_job(self, *, TrainingJobName: str) -> None:
        self.stopped.append(TrainingJobName)


def _logger() -> tuple[EventLogger, io.StringIO]:
    stream = io.StringIO()
    return EventLogger(run_id="test", component="monitor", stream=stream), stream


def test_snapshot_parses_terminal_timing() -> None:
    """Training and billable time must survive snapshot conversion."""
    snapshot = snapshot_from_description(
        {
            "TrainingJobName": "job",
            "TrainingJobStatus": "Completed",
            "SecondaryStatus": "Completed",
            "TrainingTimeInSeconds": 12,
            "BillableTimeInSeconds": 14,
        }
    )
    assert snapshot.terminal is True
    assert snapshot.training_time_seconds == 12
    assert snapshot.billable_time_seconds == 14


def test_monitor_emits_status_and_completes() -> None:
    """The monitor must handle an in-progress to completed transition."""
    client = FakeSageMakerClient(
        [
            {"TrainingJobStatus": "InProgress", "SecondaryStatus": "Training"},
            {
                "TrainingJobStatus": "Completed",
                "SecondaryStatus": "Completed",
                "TrainingTimeInSeconds": 10,
                "BillableTimeInSeconds": 11,
            },
        ]
    )
    logger, stream = _logger()
    ticks = iter([0.0, 0.0, 1.0, 1.0])
    monitor = SageMakerTrainingMonitor(
        sagemaker_client=client,
        logger=logger,
        poll_seconds=0.01,
        heartbeat_seconds=0.01,
        max_monitor_seconds=10.0,
        sleep=lambda _: None,
        monotonic=lambda: next(ticks),
    )
    result = monitor.wait("job")
    assert result.status == "Completed"
    assert "sagemaker.status.changed" in stream.getvalue()
    assert "sagemaker.monitor.finished" in stream.getvalue()


def test_monitor_raises_on_failed_job() -> None:
    """A failed SageMaker job must fail the local command."""
    client = FakeSageMakerClient(
        [
            {
                "TrainingJobStatus": "Failed",
                "SecondaryStatus": "Failed",
                "FailureReason": "synthetic failure",
            }
        ]
    )
    logger, _ = _logger()
    monitor = SageMakerTrainingMonitor(
        sagemaker_client=client,
        logger=logger,
        poll_seconds=0.01,
        heartbeat_seconds=0.01,
        max_monitor_seconds=10.0,
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )
    with pytest.raises(TrainingJobFailedError, match="synthetic failure"):
        monitor.wait("job")


def test_monitor_requests_stop_on_timeout() -> None:
    """The cost guard must request stop when bounded monitoring times out."""
    client = FakeSageMakerClient(
        [{"TrainingJobStatus": "InProgress", "SecondaryStatus": "Training"}]
    )
    logger, _ = _logger()
    ticks = iter([0.0, 0.0, 2.0])
    monitor = SageMakerTrainingMonitor(
        sagemaker_client=client,
        logger=logger,
        poll_seconds=0.01,
        heartbeat_seconds=0.01,
        max_monitor_seconds=1.0,
        sleep=lambda _: None,
        monotonic=lambda: next(ticks),
    )
    with pytest.raises(TrainingJobMonitorTimeoutError):
        monitor.wait("job", stop_on_timeout=True)
    assert client.stopped == ["job"]


def test_discover_new_training_job_excludes_known_names() -> None:
    """Discovery must return only a newly created matching job."""
    client = FakeSageMakerClient([])
    result = discover_new_training_job(
        sagemaker_client=client,
        created_after=datetime(2026, 9, 2, tzinfo=UTC),
        known_job_names={"old-job"},
        name_contains="oracle",
    )
    assert result == "lava-oracle-new"


def test_cloudwatch_filter_suppresses_unstructured_private_output() -> None:
    """Only structured LAVA event lines may be surfaced publicly."""
    assert sanitize_cloudwatch_line("raw answer and private question") is None
    line = '{"timestamp_utc":"2026-09-02T00:00:00Z","component":"job","event":"heartbeat","question":"secret"}'
    parsed = sanitize_cloudwatch_line(line)
    assert parsed is not None
    assert parsed["question"] == "<redacted>"
