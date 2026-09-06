"""Regression contracts for the capacity-safe SageMaker reader workflow."""

from __future__ import annotations

import importlib.util
import stat
from pathlib import Path
from types import ModuleType

import pytest


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_script(name: str, relative_path: str) -> ModuleType:
    path = _root() / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load test module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_submits_directly_without_launcher_process_race() -> None:
    source = (_root() / "scripts" / "run_oracle_reader_smoke.py").read_text(encoding="utf-8")
    assert "submit_or_preview_job(" in source
    assert "wait=False" in source
    assert "subprocess.Popen" not in source
    assert "threading.Thread" not in source
    assert "discover_new_training_job" not in source


def test_smoke_uses_server_side_pending_limit() -> None:
    sagemaker_source = (_root() / "src" / "lava" / "readers" / "sagemaker.py").read_text(
        encoding="utf-8"
    )
    schema_source = (_root() / "src" / "lava" / "readers" / "schemas.py").read_text(
        encoding="utf-8"
    )
    smoke_source = (_root() / "scripts" / "run_oracle_reader_smoke.py").read_text(encoding="utf-8")
    assert '"max_pending_time_in_seconds"' in sagemaker_source
    assert '"max_pending_time_in_seconds": plan.max_pending_seconds' in sagemaker_source
    assert "ge=7200, le=2419200" in schema_source
    assert "max_pending_seconds=None" in smoke_source
    assert "monitor.wait(job_name, stop_on_timeout=False)" in smoke_source


def test_local_monitor_cannot_undercut_cloud_bounds() -> None:
    module = _load_script(
        "capacity_safe_smoke_runner",
        "scripts/run_oracle_reader_smoke.py",
    )
    calculate = module._monitor_ceiling_seconds

    assert (
        calculate(
            requested=None,
            cloud_pending_seconds=86400,
            runtime_seconds=3600,
        )
        == 90900.0
    )
    with pytest.raises(ValueError, match="too short"):
        calculate(
            requested=90899.0,
            cloud_pending_seconds=86400,
            runtime_seconds=3600,
        )


def test_reconnect_monitor_derives_actual_server_bounds() -> None:
    module = _load_script(
        "capacity_safe_reconnect",
        "scripts/monitor_oracle_reader_job.py",
    )
    calculate = module._derived_monitor_ceiling
    description = {
        "StoppingCondition": {
            "MaxPendingTimeInSeconds": 86400,
            "MaxRuntimeInSeconds": 3600,
        }
    }
    assert calculate(description, requested=None) == 90900.0
    with pytest.raises(ValueError, match="shorter"):
        calculate(description, requested=3900.0)


def test_canonical_makefile_has_no_instance_override() -> None:
    source = (_root() / "Makefile").read_text(encoding="utf-8")
    assert "--instance-type" not in source
    assert "INSTANCE ?=" not in source
    assert "--model-key $(MODEL)" in source
    assert "CHARGES ?= NO" in source
    assert 'test "$(CHARGES)" = "YES"' in source


def test_canonical_stop_command_is_explicitly_guarded() -> None:
    root = _root()
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    stopper = (root / "scripts" / "stop_oracle_reader_job.py").read_text(encoding="utf-8")

    assert "stop:" in makefile
    assert "CONFIRM ?= NO" in makefile
    assert 'test "$(CONFIRM)" = "YES"' in makefile
    assert "--confirm YES" in makefile
    assert "stop_training_job" in stopper
    assert 'job_name.startswith("lava-oracle-")' in stopper
    assert '"timestamp_utc"' in stopper
    assert '"elapsed_seconds"' in stopper
    assert '"stop.heartbeat"' in stopper
    assert "STOP_TOTAL_ELAPSED_SECONDS" in stopper


def test_entrypoint_modes_are_lint_safe() -> None:
    root = _root()
    for relative in (
        "scripts/monitor_oracle_reader_job.py",
        "scripts/run_oracle_reader_smoke.py",
        "scripts/stop_oracle_reader_job.py",
        "scripts/preflight.py",
    ):
        assert stat.S_IMODE((root / relative).stat().st_mode) == 0o644, relative

    quality_gate = root / "scripts" / "quality_gate.sh"
    assert stat.S_IMODE(quality_gate.stat().st_mode) == 0o755
    assert quality_gate.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash\n")


def test_monitor_failure_fallback_catches_only_expected_failures() -> None:
    source = (_root() / "scripts" / "run_oracle_reader_smoke.py").read_text(encoding="utf-8")
    assert "except (BotoCoreError, ClientError, TypeError) as describe_error:" in source
    assert "except Exception as describe_error:" not in source


def test_event_logger_calls_are_mypy_safe() -> None:
    runner = (_root() / "scripts" / "run_oracle_reader_smoke.py").read_text(encoding="utf-8")
    monitor = (_root() / "scripts" / "monitor_oracle_reader_job.py").read_text(encoding="utf-8")
    preflight = (_root() / "scripts" / "preflight.py").read_text(encoding="utf-8")

    assert "**snapshot.as_dict()" not in runner
    assert "**failure_snapshot.as_dict()" not in runner
    assert 'logger.emit("smoke.submit.complete", snapshot=snapshot.as_dict())' in runner
    assert "snapshot=failure_snapshot.as_dict()" in runner
    assert "**snapshot.as_dict()" not in monitor
    assert 'logger.emit("monitor.reconnect.complete", snapshot=snapshot.as_dict())' in monitor
    assert 'logger.emit("preflight.verified", event_fields=snapshot)' in preflight


def test_smoke_main_has_explicit_terminal_guard() -> None:
    source = (_root() / "scripts" / "run_oracle_reader_smoke.py").read_text(encoding="utf-8")
    assert "Smoke command exited its telemetry stage without a terminal result." in source


def test_model_specific_qwen_wrapper_is_retired() -> None:
    assert not (_root() / "scripts" / "run_qwen38_smoke.sh").exists()
