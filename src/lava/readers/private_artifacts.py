"""Atomic private retention of exact model generations inside SageMaker model artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

_COUNTER_LOCK = threading.Lock()
_COUNTER = 0
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class RawResponseArtifact:
    """Metadata for one exact private response retained before parsing."""

    sequence: int
    response_path: str
    metadata_path: str
    sha256: str
    byte_count: int
    saved_at_utc: str

    def as_dict(self) -> dict[str, object]:
        """Return JSON-safe metadata without exposing response content."""
        return asdict(self)


def _next_sequence() -> int:
    global _COUNTER
    with _COUNTER_LOCK:
        _COUNTER += 1
        return _COUNTER


def _safe_identifier(value: str | None, sequence: int) -> str:
    if value:
        cleaned = _SAFE_ID.sub("-", value).strip("-._")
        if cleaned:
            return cleaned[:80]
    return f"response-{sequence:06d}"


def _atomic_private_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temporary)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise


def persist_raw_response(
    raw_text: str,
    *,
    question_id: str | None = None,
    root: Path | None = None,
) -> RawResponseArtifact:
    """Persist exact UTF-8 generation before any stripping, parsing, or normalization."""
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    sequence = _next_sequence()
    destination = root or Path(os.environ.get("LAVA_PRIVATE_MODEL_DIR", "/opt/ml/model/private"))
    response_dir = destination / "raw_responses"
    stem = _safe_identifier(question_id, sequence)
    response_path = response_dir / f"{stem}.txt"
    metadata_path = response_dir / f"{stem}.json"
    encoded = raw_text.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    saved_at = datetime.now(tz=UTC).isoformat()
    metadata = {
        "schema_version": 1,
        "sequence": sequence,
        "question_id": question_id,
        "sha256": digest,
        "byte_count": len(encoded),
        "saved_at_utc": saved_at,
        "response_filename": response_path.name,
    }
    _atomic_private_write(response_path, encoded)
    _atomic_private_write(
        metadata_path,
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return RawResponseArtifact(
        sequence=sequence,
        response_path=str(response_path),
        metadata_path=str(metadata_path),
        sha256=digest,
        byte_count=len(encoded),
        saved_at_utc=saved_at,
    )
