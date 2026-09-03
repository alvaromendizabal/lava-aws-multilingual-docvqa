"""Small dependency-free UTC event logger with stage timing and heartbeat support."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any


class RuntimeEventLogger:
    """Emit machine-readable one-line events suitable for CloudWatch and terminal tails."""

    def __init__(self, component: str) -> None:
        self.component = component
        self.started = time.monotonic()
        self._lock = threading.Lock()

    def emit(self, event: str, *, level: str = "INFO", **fields: Any) -> None:
        """Emit one event with UTC time and total elapsed seconds."""
        payload = {
            "timestamp_utc": datetime.now(tz=UTC).isoformat(timespec="milliseconds"),
            "level": level,
            "component": self.component,
            "event": event,
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
            **fields,
        }
        with self._lock:
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)

    @contextmanager
    def stage(self, name: str, *, heartbeat_seconds: float = 15.0) -> Iterator[None]:
        """Log start/completion/failure plus periodic stage heartbeats."""
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        stage_started = time.monotonic()
        stop = threading.Event()

        def heartbeat() -> None:
            while not stop.wait(heartbeat_seconds):
                self.emit(
                    f"{name}.heartbeat",
                    stage_elapsed_seconds=round(time.monotonic() - stage_started, 3),
                )

        thread = threading.Thread(target=heartbeat, name=f"{name}-heartbeat", daemon=True)
        self.emit(f"{name}.started", stage_elapsed_seconds=0.0)
        thread.start()
        try:
            yield
        except Exception as exc:
            self.emit(
                f"{name}.failed",
                level="ERROR",
                stage_elapsed_seconds=round(time.monotonic() - stage_started, 3),
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
            raise
        else:
            self.emit(
                f"{name}.completed",
                stage_elapsed_seconds=round(time.monotonic() - stage_started, 3),
            )
        finally:
            stop.set()
            thread.join(timeout=min(heartbeat_seconds, 1.0))
