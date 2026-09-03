"""Structured, privacy-aware runtime telemetry with progress heartbeats."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self, TextIO

_ACCOUNT_ID = re.compile(r"(?<!\d)\d{12}(?!\d)")
_S3_URI = re.compile(r"s3://[^/\s]+")
_SECRET_LIKE = re.compile(
    r"(?i)(bearer\s+[a-z0-9._~+/=-]+|aws_secret_access_key\s*[=:]\s*\S+|"
    r"aws_access_key_id\s*[=:]\s*\S+)"
)
_SENSITIVE_KEYS = frozenset(
    {
        "access_token",
        "answer",
        "aws_account_id",
        "credential",
        "credentials",
        "native_text",
        "page_text",
        "password",
        "private_key",
        "prompt",
        "question",
        "raw_answer",
        "raw_content",
        "raw_prompt",
        "raw_question",
        "raw_response",
        "reference_answer",
        "secret",
        "secret_access_key",
        "token",
    }
)


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(tz=UTC)


def format_utc(value: datetime) -> str:
    """Format a UTC timestamp using an ISO-8601 ``Z`` suffix."""
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def redact_string(value: str) -> str:
    """Redact account IDs, bucket names, and credential-like substrings."""
    redacted = _ACCOUNT_ID.sub("<redacted-account>", value)
    redacted = _S3_URI.sub("s3://<redacted-bucket>", redacted)
    return _SECRET_LIKE.sub("<redacted-secret>", redacted)


def _is_sensitive_key(key: str | None) -> bool:
    if key is None:
        return False
    normalized = key.strip().lower()
    if normalized in _SENSITIVE_KEYS:
        return True
    return normalized.endswith(("_secret", "_password", "_private_key", "_access_token"))


def sanitize_value(value: object, *, key: str | None = None) -> object:
    """Recursively sanitize a value before it is written to logs or reports."""
    if _is_sensitive_key(key):
        return "<redacted>"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return redact_string(value)
    if isinstance(value, Path):
        return redact_string(str(value))
    if isinstance(value, Mapping):
        return {
            str(item_key): sanitize_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [sanitize_value(item) for item in value]
    if isinstance(value, set | frozenset):
        return [sanitize_value(item) for item in sorted(value, key=repr)]
    return redact_string(str(value))


def stable_hash(value: object) -> str:
    """Return a deterministic SHA-256 hash of a sanitized JSON-compatible value."""
    payload = json.dumps(
        sanitize_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(slots=True)
class EventLogger:
    """Emit human-readable and JSONL events with timestamps and elapsed time."""

    run_id: str
    component: str
    stream: TextIO = field(default_factory=lambda: io.TextIOWrapper(io.BytesIO()))
    jsonl_path: Path | None = None
    static_context: Mapping[str, object] = field(default_factory=dict)
    clock: Callable[[], datetime] = utc_now
    monotonic: Callable[[], float] = time.perf_counter
    _start_monotonic: float = field(init=False, repr=False)
    _lock: threading.Lock = field(init=False, repr=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        """Initialize runtime state and ensure the JSONL parent exists."""
        self._start_monotonic = self.monotonic()
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def to_stdout(
        cls,
        *,
        run_id: str,
        component: str,
        jsonl_path: Path | None = None,
        static_context: Mapping[str, object] | None = None,
    ) -> EventLogger:
        """Create a logger that emits immediately to stdout."""
        import sys

        return cls(
            run_id=run_id,
            component=component,
            stream=sys.stdout,
            jsonl_path=jsonl_path,
            static_context=static_context or {},
        )

    def bind(self, **context: object) -> EventLogger:
        """Create a child logger with additional static context."""
        merged = dict(self.static_context)
        merged.update(context)
        return EventLogger(
            run_id=self.run_id,
            component=self.component,
            stream=self.stream,
            jsonl_path=self.jsonl_path,
            static_context=merged,
            clock=self.clock,
            monotonic=self.monotonic,
        )

    @property
    def elapsed_seconds(self) -> float:
        """Return elapsed seconds since logger creation."""
        return max(0.0, self.monotonic() - self._start_monotonic)

    def emit(
        self,
        event: str,
        *,
        level: str = "INFO",
        message: str | None = None,
        **fields: object,
    ) -> dict[str, object]:
        """Emit one redacted structured event and return its payload."""
        payload: dict[str, object] = {
            "schema_version": 1,
            "timestamp_utc": format_utc(self.clock()),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "level": level.upper(),
            "component": self.component,
            "run_id": self.run_id,
            "event": event,
        }
        payload.update(self.static_context)
        if message is not None:
            payload["message"] = message
        payload.update(fields)
        sanitized = sanitize_value(payload)
        if not isinstance(sanitized, dict):
            message_text = "Sanitized event payload must remain a mapping."
            raise TypeError(message_text)

        visible_fields = [
            f"{key}={json.dumps(value, ensure_ascii=False, sort_keys=True)}"
            for key, value in sorted(sanitized.items())
            if key not in {"timestamp_utc", "level", "component", "event", "message"}
        ]
        visible_message = sanitized.get("message")
        message_segment = f" {visible_message}" if visible_message else ""
        line = (
            f"[{sanitized['timestamp_utc']}] [{sanitized['level']}] "
            f"[{sanitized['component']}] {sanitized['event']}{message_segment}"
        )
        if visible_fields:
            line += " | " + " ".join(visible_fields)

        with self._lock:
            print(line, file=self.stream, flush=True)
            if self.jsonl_path is not None:
                with self.jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(sanitized, ensure_ascii=False, sort_keys=True))
                    handle.write("\n")
                    handle.flush()
        return sanitized

    def stage(
        self,
        name: str,
        *,
        heartbeat_seconds: float = 30.0,
        **fields: object,
    ) -> Stage:
        """Create a timed stage that emits start, heartbeat, and terminal events."""
        return Stage(
            logger=self,
            name=name,
            heartbeat_seconds=heartbeat_seconds,
            fields=fields,
        )


@dataclass(slots=True)
class Stage(contextlib.AbstractContextManager["Stage"]):
    """Context manager for visible, heartbeat-enabled stage execution."""

    logger: EventLogger
    name: str
    heartbeat_seconds: float = 30.0
    fields: Mapping[str, object] = field(default_factory=dict)
    _start: float = field(init=False, default=0.0, repr=False)
    _stop_event: threading.Event = field(init=False, default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)

    def __enter__(self) -> Self:
        """Start the stage and its heartbeat thread."""
        if self.heartbeat_seconds <= 0:
            message = "heartbeat_seconds must be greater than zero."
            raise ValueError(message)
        self._start = self.logger.monotonic()
        self.logger.emit(f"{self.name}.started", **dict(self.fields))
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"lava-heartbeat-{self.name}",
            daemon=True,
        )
        self._thread.start()
        return self

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_seconds):
            duration = max(0.0, self.logger.monotonic() - self._start)
            self.logger.emit(
                f"{self.name}.heartbeat",
                stage_elapsed_seconds=round(duration, 3),
                **dict(self.fields),
            )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Stop heartbeats and emit a completion or failure event."""
        del traceback
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.heartbeat_seconds + 1.0))
        duration = max(0.0, self.logger.monotonic() - self._start)
        terminal_fields = {
            **dict(self.fields),
            "stage_elapsed_seconds": round(duration, 3),
        }
        if exc_type is None:
            self.logger.emit(f"{self.name}.completed", **terminal_fields)
            return False
        self.logger.emit(
            f"{self.name}.failed",
            level="ERROR",
            exception_type=exc_type.__name__,
            exception_message=str(exc_value) if exc_value is not None else None,
            **terminal_fields,
        )
        return False


@dataclass(slots=True)
class ProgressReporter:
    """Emit monotonic progress events with rate and estimated time remaining."""

    logger: EventLogger
    total: int
    event_prefix: str
    emit_every: int = 1
    _completed: int = field(init=False, default=0)
    _started: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate configuration and initialize the progress clock."""
        if self.total <= 0:
            message = "total must be greater than zero."
            raise ValueError(message)
        if self.emit_every <= 0:
            message = "emit_every must be greater than zero."
            raise ValueError(message)
        self._started = self.logger.monotonic()

    def advance(self, *, increment: int = 1, **fields: object) -> None:
        """Advance progress and emit when the configured cadence is reached."""
        if increment <= 0:
            message = "increment must be greater than zero."
            raise ValueError(message)
        self._completed = min(self.total, self._completed + increment)
        if self._completed % self.emit_every != 0 and self._completed != self.total:
            return
        elapsed = max(0.0, self.logger.monotonic() - self._started)
        rate = self._completed / elapsed if elapsed > 0 else None
        remaining = self.total - self._completed
        eta = remaining / rate if rate and rate > 0 else None
        self.logger.emit(
            f"{self.event_prefix}.progress",
            completed=self._completed,
            total=self.total,
            percent=round(100.0 * self._completed / self.total, 2),
            elapsed_seconds=round(elapsed, 3),
            rate_per_second=round(rate, 6) if rate is not None else None,
            eta_seconds=round(eta, 3) if eta is not None else None,
            **fields,
        )


@contextlib.contextmanager
def total_runtime(logger: EventLogger, *, name: str = "run") -> Iterator[None]:
    """Emit a visible total-runtime boundary around a complete command."""
    with logger.stage(name, heartbeat_seconds=30.0):
        yield
