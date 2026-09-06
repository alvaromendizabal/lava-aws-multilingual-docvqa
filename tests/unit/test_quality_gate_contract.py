from __future__ import annotations

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
