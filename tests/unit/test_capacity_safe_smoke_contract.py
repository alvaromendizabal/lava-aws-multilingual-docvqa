"""Regression contracts for the capacity-safe SageMaker reader workflow."""

from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
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
    source = (_root() / "scripts" / "run_oracle_reader.py").read_text(encoding="utf-8")
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
    smoke_source = (_root() / "scripts" / "run_oracle_reader.py").read_text(encoding="utf-8")
    assert '"max_pending_time_in_seconds"' in sagemaker_source
    assert '"max_pending_time_in_seconds": plan.max_pending_seconds' in sagemaker_source
    assert "ge=7200, le=2419200" in schema_source
    assert "max_pending_seconds=None" in smoke_source
    assert "monitor.wait(job_name, stop_on_timeout=False)" in smoke_source


def test_local_monitor_cannot_undercut_cloud_bounds() -> None:
    module = _load_script(
        "capacity_safe_smoke_runner",
        "scripts/run_oracle_reader.py",
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


_PYTHON_ENTRYPOINTS = (
    "scripts/monitor_oracle_reader_job.py",
    "scripts/run_oracle_reader.py",
    "scripts/stop_oracle_reader_job.py",
    "scripts/preflight.py",
)
_QUALITY_GATE = "scripts/quality_gate.sh"


def _assert_entrypoint_modes(root: Path) -> None:
    """Check execution semantics, not group/other bits controlled by the user's umask."""
    for relative in _PYTHON_ENTRYPOINTS:
        mode = (root / relative).lstat().st_mode
        assert stat.S_ISREG(mode), f"{relative} must be a regular file"
        assert mode & stat.S_IRUSR, f"{relative} must be owner-readable"
        assert not mode & 0o111, f"{relative} must not be executable"

    quality_gate = root / _QUALITY_GATE
    mode = quality_gate.lstat().st_mode
    assert stat.S_ISREG(mode), f"{_QUALITY_GATE} must be a regular file"
    required = stat.S_IRUSR | stat.S_IXUSR
    assert mode & required == required, f"{_QUALITY_GATE} must be owner-readable and executable"
    assert quality_gate.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash\n")


def test_entrypoint_modes_are_lint_safe() -> None:
    _assert_entrypoint_modes(_root())


def test_entrypoint_git_modes_are_canonical() -> None:
    """Git records executable intent; a private checkout need not be world-readable."""
    expected = {relative: "100644" for relative in _PYTHON_ENTRYPOINTS}
    expected[_QUALITY_GATE] = "100755"
    result = subprocess.run(
        ["git", "ls-files", "--stage", "-z", "--", *expected],
        cwd=_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    observed = {}
    for entry in result.stdout.split("\0"):
        if entry:
            metadata, relative = entry.split("\t", 1)
            mode, _object_id, stage = metadata.split()
            assert stage == "0", f"Unmerged entrypoint: {relative}"
            observed[relative] = mode
    assert observed == expected


def _write_entrypoint_fixture(root: Path) -> None:
    """Small inert scripts: no project imports, credentials, network or cloud calls."""
    (root / "scripts").mkdir(parents=True)
    for relative in _PYTHON_ENTRYPOINTS:
        path = root / relative
        path.write_text("# Non-executable Python entrypoint fixture.\n", encoding="utf-8")
        path.chmod(0o644)
    gate = root / _QUALITY_GATE
    gate.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    gate.chmod(0o755)


@pytest.mark.skipif(os.name != "posix", reason="POSIX checkout permissions")
@pytest.mark.parametrize("mask", [0o022, 0o002, 0o077], ids=["022", "002", "077"])
def test_entrypoint_contract_survives_real_git_checkout(tmp_path: Path, mask: int) -> None:
    """Reproduce Git's actual checkout behavior, with umask isolated to the child process."""
    source = tmp_path / "source"
    _write_entrypoint_fixture(source)
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    subprocess.run(["git", "add", "--", "scripts"], cwd=source, check=True)
    checkout = tmp_path / "checkout"
    subprocess.run(
        ["git", "checkout-index", "--all", f"--prefix={checkout}/"],
        cwd=source,
        umask=mask,
        check=True,
        capture_output=True,
        text=True,
    )
    assert stat.S_IMODE((checkout / _QUALITY_GATE).stat().st_mode) == 0o777 & ~mask
    for relative in _PYTHON_ENTRYPOINTS:
        assert stat.S_IMODE((checkout / relative).stat().st_mode) == 0o666 & ~mask
    _assert_entrypoint_modes(checkout)
    subprocess.run([str(checkout / _QUALITY_GATE)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize(
    ("relative", "mode", "message"),
    [
        (_PYTHON_ENTRYPOINTS[0], 0o744, "must not be executable"),
        (_PYTHON_ENTRYPOINTS[0], 0o200, "must be owner-readable"),
        (_QUALITY_GATE, 0o600, "must be owner-readable and executable"),
        (_QUALITY_GATE, 0o100, "must be owner-readable and executable"),
    ],
)
def test_entrypoint_contract_rejects_unusable_modes(
    tmp_path: Path, relative: str, mode: int, message: str
) -> None:
    """Accepting private checkouts must not hide genuine read/execute regressions."""
    _write_entrypoint_fixture(tmp_path)
    (tmp_path / relative).chmod(mode)
    with pytest.raises(AssertionError, match=message):
        _assert_entrypoint_modes(tmp_path)


def test_entrypoint_contract_requires_shell_shebang(tmp_path: Path) -> None:
    _write_entrypoint_fixture(tmp_path)
    (tmp_path / _QUALITY_GATE).write_text("exit 0\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        _assert_entrypoint_modes(tmp_path)


def test_monitor_failure_fallback_catches_only_expected_failures() -> None:
    source = (_root() / "scripts" / "run_oracle_reader.py").read_text(encoding="utf-8")
    assert "except (BotoCoreError, ClientError, TypeError) as describe_error:" in source
    assert "except Exception as describe_error:" not in source


def test_event_logger_calls_are_mypy_safe() -> None:
    runner = (_root() / "scripts" / "run_oracle_reader.py").read_text(encoding="utf-8")
    monitor = (_root() / "scripts" / "monitor_oracle_reader_job.py").read_text(encoding="utf-8")
    preflight = (_root() / "scripts" / "preflight.py").read_text(encoding="utf-8")

    assert "**snapshot.as_dict()" not in runner
    assert "**failure_snapshot.as_dict()" not in runner
    assert 'logger.emit(f"{args.mode}.submit.complete", snapshot=snapshot.as_dict())' in runner
    assert "snapshot=failure_snapshot.as_dict()" in runner
    assert "**snapshot.as_dict()" not in monitor
    assert 'logger.emit("monitor.reconnect.complete", snapshot=snapshot.as_dict())' in monitor
    assert 'logger.emit("preflight.verified", event_fields=snapshot)' in preflight


def test_smoke_main_has_explicit_terminal_guard() -> None:
    source = (_root() / "scripts" / "run_oracle_reader.py").read_text(encoding="utf-8")
    assert "Reader command exited its telemetry stage without a terminal result." in source


def test_model_specific_qwen_wrapper_is_retired() -> None:
    assert not (_root() / "scripts" / "run_qwen38_smoke.sh").exists()
