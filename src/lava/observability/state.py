"""Atomic local state for reconnectable SageMaker experiment monitoring."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainingRunState:
    """Minimal state required to reconnect to a submitted training job."""

    schema_version: int
    run_id: str
    job_name: str | None
    model_key: str
    git_commit_sha: str
    protocol_lock_id: str
    output_s3_prefix: str
    created_at_utc: str
    updated_at_utc: str
    status: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def write_state_atomic(path: Path, state: TrainingRunState) -> None:
    """Persist state atomically with restrictive permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(state.as_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_state(path: Path) -> TrainingRunState:
    """Read and validate a saved training-run state file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TrainingRunState(**payload)


def latest_state_path(root: Path) -> Path:
    """Return the conventional latest-state path below the repository."""
    return root / "artifacts" / "oracle_reader" / "runtime" / "latest.json"
