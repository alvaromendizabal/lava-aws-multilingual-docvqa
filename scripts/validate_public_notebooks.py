"""Validate public notebook hygiene and Jupytext pairing."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

_ACCOUNT_ID = re.compile(r"(?<!\d)\d{12}(?!\d)")
_EXPECTED = (
    "00_environment_and_cost_guardrails",
    "01_oracle_evidence_reader_benchmark",
    "02_first_gpu_smoke_run",
)


def main() -> int:
    """Fail when a public notebook stores output, secrets, or unmatched source."""
    root = Path(__file__).resolve().parents[1]
    for stem in _EXPECTED:
        notebook_path = root / "notebooks" / f"{stem}.ipynb"
        source_path = root / "notebooks" / f"{stem}.py"
        if not notebook_path.is_file() or not source_path.is_file():
            message = f"Missing notebook pair for {stem}."
            raise FileNotFoundError(message)
        notebook = nbformat.read(notebook_path, as_version=4)
        serialized = nbformat.writes(notebook)
        if _ACCOUNT_ID.search(serialized):
            message = f"Notebook contains a 12-digit account identifier: {notebook_path}"
            raise RuntimeError(message)
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            if cell.get("outputs", []):
                message = f"Notebook contains stored outputs: {notebook_path}"
                raise RuntimeError(message)
            if cell.get("execution_count") is not None:
                message = f"Notebook contains execution counts: {notebook_path}"
                raise RuntimeError(message)
    print("PUBLIC_NOTEBOOK_HYGIENE_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
