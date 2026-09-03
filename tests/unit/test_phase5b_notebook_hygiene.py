"""Regression tests for public Phase 5 notebook hygiene."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

_EXPECTED = (
    "00_environment_and_cost_guardrails",
    "01_oracle_evidence_reader_benchmark",
    "02_first_gpu_smoke_run",
)
_ACCOUNT_ID = re.compile(r"(?<!\d)\d{12}(?!\d)")


def test_phase5_notebook_pairs_exist_and_are_output_free() -> None:
    """Every public notebook must have a reviewable source pair and no stored output."""
    root = Path(__file__).resolve().parents[2]
    for stem in _EXPECTED:
        notebook_path = root / "notebooks" / f"{stem}.ipynb"
        source_path = root / "notebooks" / f"{stem}.py"
        assert notebook_path.is_file()
        assert source_path.is_file()
        notebook = nbformat.read(notebook_path, as_version=4)
        assert notebook.metadata.kernelspec.name == "lava"
        serialized = nbformat.writes(notebook)
        assert _ACCOUNT_ID.search(serialized) is None
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.get("execution_count") is None
                assert not cell.get("outputs", [])
