"""Tests for atomic reconnectable experiment state."""

from __future__ import annotations

from lava.observability.state import TrainingRunState, read_state, write_state_atomic


def test_state_round_trip_is_atomic_and_restrictive(tmp_path) -> None:
    """Saved state must round-trip and use owner-only permissions."""
    path = tmp_path / "runtime" / "latest.json"
    state = TrainingRunState(
        schema_version=1,
        run_id="run-1",
        job_name="job-1",
        model_key="qwen35_4b_fused_direct",
        git_commit_sha="a" * 40,
        protocol_lock_id="b" * 64,
        output_s3_prefix="s3://bucket/prefix",
        created_at_utc="2026-09-02T00:00:00.000Z",
        updated_at_utc="2026-09-02T00:00:01.000Z",
        status="InProgress",
    )
    write_state_atomic(path, state)
    assert read_state(path) == state
    assert path.stat().st_mode & 0o777 == 0o600
