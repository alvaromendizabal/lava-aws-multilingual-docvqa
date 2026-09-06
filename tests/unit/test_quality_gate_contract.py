from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_quality_gate_is_fail_closed_and_observable() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "quality_gate.sh").read_text(encoding="utf-8")

    assert "set -Eeuo pipefail" in script
    assert "QUALITY_GATE_PASSED" in script
    assert "QUALITY_GATE_FAILED" in script
    assert "HEARTBEAT:" in script
    assert "total_elapsed_seconds=" in script
    assert "uv sync --frozen" in script
    assert "ruff format --check" in script
    assert "ruff check" in script
    assert "mypy src scripts" in script
    assert "pytest -q" in script
    assert "compileall" in script
    assert "validate_public_notebooks.py" in script
    assert "git diff --check" in script


def test_ci_runs_the_canonical_quality_gate() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "bash scripts/quality_gate.sh" in workflow


def test_type_checks_survive_a_corrupt_sqlite_cache_and_still_detect_errors(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    cache = tmp_path / "cache"
    version_cache = cache / f"{sys.version_info.major}.{sys.version_info.minor}"
    version_cache.mkdir(parents=True)
    (version_cache / "cache.db").write_bytes(b"deliberately malformed database")
    source = tmp_path / "example.py"
    source.write_text('value: int = "incorrect"\n')
    command = [
        sys.executable,
        "-m",
        "mypy",
        "--config-file",
        str(root / "pyproject.toml"),
        "--cache-dir",
        str(cache),
        str(source),
    ]
    invalid = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    assert invalid.returncode == 1, invalid.stdout + invalid.stderr
    assert "Incompatible types" in invalid.stdout
    source.write_text("value: int = 1\n")
    valid = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    assert valid.returncode == 0, valid.stdout + valid.stderr
