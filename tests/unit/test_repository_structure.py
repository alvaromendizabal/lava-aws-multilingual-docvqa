from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY_PATHS = (
    "scripts/phase5a_preflight.sh",
    "scripts/phase5b_preflight.py",
    "scripts/phase5b_preflight.sh",
    "scripts/phase5c_preflight.py",
    "scripts/phase5c_preflight.sh",
    "scripts/validate_phase5a.py",
    "scripts/run_qwen38_smoke.sh",
    "scripts/launch_oracle_reader.py",
    "scripts/.gitkeep",
    "notebooks/.gitkeep",
    "docs/.gitkeep",
    "docs/PHASE5B_OBSERVABLE_GPU_SMOKE.md",
    "docs/ORACLE_READER_BENCHMARK.md",
    "VERIFICATION.md",
    "tests/unit/test_phase5b_notebook_hygiene.py",
    "tests/unit/test_phase5c_source_contract.py",
    "scripts/execute_notebook_smoke.py",
)


def test_historical_phase_wrappers_are_not_part_of_the_public_interface() -> None:
    for relative in LEGACY_PATHS:
        assert not (ROOT / relative).exists(), relative


def test_public_workflow_has_one_canonical_interface() -> None:
    required = (
        "Makefile",
        "scripts/quality_gate.sh",
        "scripts/preflight.py",
        "scripts/run_oracle_reader.py",
        "scripts/monitor_oracle_reader_job.py",
        "scripts/inspect_oracle_reader_artifact.py",
        "scripts/sync_oracle_reader_results.py",
        "scripts/stop_oracle_reader_job.py",
        "scripts/validate_public_notebooks.py",
        "scripts/execute_notebooks.py",
    )
    for relative in required:
        assert (ROOT / relative).is_file(), relative


def test_git_does_not_track_python_cache_artifacts() -> None:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = completed.stdout.splitlines()
    forbidden = [
        path
        for path in tracked
        if "__pycache__/" in path or path.endswith((".pyc", ".pyo", ".pyd"))
    ]
    assert forbidden == []


def test_gitignore_excludes_python_runtime_cache() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "__pycache__/" in ignore
    assert "*.py[cod]" in ignore


def test_empty_placeholder_capabilities_are_not_advertised_as_implemented() -> None:
    removed = (
        "app/.gitkeep",
        "docker/.gitkeep",
        "infra/terraform/.gitkeep",
        "src/lava/agents/__init__.py",
        "src/lava/models/__init__.py",
        "src/lava/reranking/__init__.py",
        "src/lava/serving/__init__.py",
    )
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    assert not set(removed).intersection(tracked)


def test_project_entrance_links_every_canonical_notebook_in_order() -> None:
    from lava.notebook_execution import NOTEBOOK_STEMS

    readme = (ROOT / "README.md").read_text()
    catalog = "\n".join(line for line in readme.splitlines() if line.startswith("| ["))
    positions = [catalog.index(f"notebooks/{stem}.ipynb") for stem in NOTEBOOK_STEMS]
    assert positions == sorted(positions)
    assert "reports/notebooks/" not in readme
    assert "## Scope and limitations" in readme
    assert "Completed applied ML research system" in readme
