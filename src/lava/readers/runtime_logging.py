"""Small dependency-free UTC event logger with stage timing and heartbeat support."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class RuntimeEventLogger:
    """Emit machine-readable one-line events suitable for CloudWatch and terminal tails."""

    def __init__(self, component: str, *, jsonl_path: Path | None = None) -> None:
        self.component = component
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self.jsonl_path = jsonl_path
        if jsonl_path is not None:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)

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
            line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            if self.jsonl_path is not None:
                with self.jsonl_path.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            print(line, flush=True)

    @contextmanager
    def stage(self, name: str, *, heartbeat_seconds: float = 15.0) -> Iterator[None]:
        """Log start/completion/failure plus periodic stage heartbeats."""
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        stage_started = time.monotonic()
        stop = threading.Event()
        lifecycle_lock = threading.Lock()

        def heartbeat() -> None:
            while not stop.wait(heartbeat_seconds):
                with lifecycle_lock:
                    # The timer may have expired just before the stage stopped.
                    if stop.is_set():
                        return
                    self.emit(
                        f"{name}.heartbeat",
                        stage_elapsed_seconds=round(time.monotonic() - stage_started, 3),
                    )

        def finish(event: str, **fields: Any) -> None:
            stop.set()
            # Serialize the terminal event after any heartbeat already writing.
            with lifecycle_lock:
                self.emit(
                    event,
                    stage_elapsed_seconds=round(time.monotonic() - stage_started, 3),
                    **fields,
                )

        thread = threading.Thread(target=heartbeat, name=f"{name}-heartbeat", daemon=True)
        self.emit(f"{name}.started", stage_elapsed_seconds=0.0)
        thread.start()
        try:
            yield
        except BaseException as exc:
            finish(
                f"{name}.failed",
                level="ERROR",
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
            raise
        else:
            finish(f"{name}.completed")
        finally:
            stop.set()
            thread.join(timeout=min(heartbeat_seconds, 1.0))
