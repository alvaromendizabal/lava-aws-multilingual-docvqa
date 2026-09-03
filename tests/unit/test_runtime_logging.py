from __future__ import annotations

import json
import time

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
