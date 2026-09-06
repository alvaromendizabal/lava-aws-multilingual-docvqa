from __future__ import annotations

import json
import time

import pytest

from lava.readers.runtime_logging import RuntimeEventLogger


def test_logger_emits_utc_total_stage_and_heartbeat(capsys) -> None:
    logger = RuntimeEventLogger("test.component")
    with logger.stage("work", heartbeat_seconds=0.01):
        time.sleep(0.035)
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    events = [row["event"] for row in lines]
    assert events[0] == "work.started"
    assert "work.heartbeat" in events
    assert events[-1] == "work.completed"
    for row in lines:
        assert row["timestamp_utc"].endswith("+00:00")
        assert row["elapsed_seconds"] >= 0
    heartbeat = next(row for row in lines if row["event"] == "work.heartbeat")
    assert heartbeat["stage_elapsed_seconds"] > 0


def test_log_file_appends_durable_events_across_logger_instances(tmp_path, capsys):
    path = tmp_path / "run/events.jsonl"
    RuntimeEventLogger("first", jsonl_path=path).emit("accepted")
    original = path.read_bytes()
    RuntimeEventLogger("second", jsonl_path=path).emit("resumed")
    assert path.read_bytes().startswith(original)
    assert path.read_text() == capsys.readouterr().out


@pytest.mark.parametrize("exception", [RuntimeError("failure"), KeyboardInterrupt()])
def test_log_file_records_failed_and_interrupted_stages(tmp_path, exception):
    path = tmp_path / "events.jsonl"
    logger = RuntimeEventLogger("test", jsonl_path=path)
    with pytest.raises(type(exception)), logger.stage("work", heartbeat_seconds=0.01):
        raise exception
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[-1]["event"] == "work.failed"
    assert rows[-1]["stage_elapsed_seconds"] >= 0
    assert rows[-1]["exception_type"] == type(exception).__name__
