"""Tests for structured runtime events and progress heartbeats."""

from __future__ import annotations

import io
import json
import time
from datetime import UTC, datetime

from lava.observability.events import EventLogger, ProgressReporter, redact_string, stable_hash


def test_event_logger_redacts_private_values(tmp_path) -> None:
    """Logs must redact account IDs, bucket names, and raw questions."""
    stream = io.StringIO()
    jsonl = tmp_path / "events.jsonl"
    logger = EventLogger(
        run_id="run-1",
        component="unit",
        stream=stream,
        jsonl_path=jsonl,
        clock=lambda: datetime(2026, 9, 2, tzinfo=UTC),
        monotonic=lambda: 10.0,
    )
    payload = logger.emit(
        "test.event",
        aws_arn="arn:aws:iam::123456789012:user/example",
        uri="s3://private-bucket/path/file.json",
        question="private question",
        question_count=16,
    )
    assert "123456789012" not in stream.getvalue()
    assert "private-bucket" not in stream.getvalue()
    assert payload["question"] == "<redacted>"
    assert payload["question_count"] == 16
    parsed = json.loads(jsonl.read_text().splitlines()[0])
    assert parsed["aws_arn"] == "arn:aws:iam::<redacted-account>:user/example"
    assert parsed["uri"] == "s3://<redacted-bucket>/path/file.json"


def test_stage_emits_heartbeat_and_completion() -> None:
    """A long-enough stage must emit visible heartbeat and terminal events."""
    stream = io.StringIO()
    logger = EventLogger.to_stdout(run_id="run-2", component="unit")
    logger.stream = stream
    with logger.stage("slow", heartbeat_seconds=0.01):
        time.sleep(0.035)
    output = stream.getvalue()
    assert "slow.started" in output
    assert "slow.heartbeat" in output
    assert "slow.completed" in output
    assert "stage_elapsed_seconds" in output


def test_progress_reporter_emits_percent_and_eta() -> None:
    """Progress events must include completed count, percent, and ETA fields."""
    stream = io.StringIO()
    values = iter([0.0, 0.0, 2.0, 2.0])
    logger = EventLogger(
        run_id="run-3",
        component="unit",
        stream=stream,
        monotonic=lambda: next(values),
    )
    reporter = ProgressReporter(logger=logger, total=4, event_prefix="items")
    reporter.advance(increment=2)
    output = stream.getvalue()
    assert "completed=2" in output
    assert "percent=50.0" in output
    assert "eta_seconds=2.0" in output


def test_redaction_and_hash_are_deterministic() -> None:
    """Sanitized hashes must be stable and independent of private bucket names."""
    assert redact_string("s3://one-private-bucket/a") == "s3://<redacted-bucket>/a"
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
