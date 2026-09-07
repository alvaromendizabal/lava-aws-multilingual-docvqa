from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import nbformat

EXPECTED = (
    "00_reproducibility_and_protocol",
    "01_oracle_reader_benchmark_design",
    "02_verified_gpu_execution",
    "03_model_scaling_and_cost",
    "04_evidence_retrieval",
)
ACCOUNT_ID = re.compile(r"(?<!\d)\d{12}(?!\d)")


def test_notebook_clean_filter_preserves_ids_and_removes_execution_outputs() -> None:
    """Prevent ID renumbering from making restored notebooks appear modified."""
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("Results", id="results-heading"),
            nbformat.v4.new_code_cell(
                "print('example')",
                id="results-code",
                execution_count=7,
                outputs=[nbformat.v4.new_output("stream", name="stdout", text="example\n")],
            ),
        ]
    )
    command = [sys.executable, "-m", "nbstripout", "--keep-id"]
    first = subprocess.run(
        command, input=nbformat.writes(notebook), text=True, capture_output=True, check=True
    ).stdout
    cleaned = nbformat.reads(first, as_version=4)
    assert [cell.id for cell in cleaned.cells] == [cell.id for cell in notebook.cells]
    assert [cell.source for cell in cleaned.cells] == [cell.source for cell in notebook.cells]
    assert cleaned.cells[1].outputs == []
    assert cleaned.cells[1].execution_count is None
    second = subprocess.run(command, input=first, text=True, capture_output=True, check=True).stdout
    assert second == first


def test_public_notebooks_are_paired_output_free_and_sanitized() -> None:
    root = Path(__file__).resolve().parents[2]
    notebook_dir = root / "notebooks"

    ipynb_paths = sorted(notebook_dir.glob("*.ipynb"))
    py_paths = sorted(notebook_dir.glob("*.py"))

    assert [path.stem for path in ipynb_paths] == list(EXPECTED)
    assert [path.stem for path in py_paths] == list(EXPECTED)

    for path in ipynb_paths:
        notebook = nbformat.read(path, as_version=4)
        assert notebook.metadata.kernelspec.name == "lava"

        serialized = nbformat.writes(notebook)
        assert ACCOUNT_ID.search(serialized) is None

        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            assert not cell.get("outputs", [])
            assert cell.get("execution_count") is None

    for path in py_paths:
        assert ACCOUNT_ID.search(path.read_text(encoding="utf-8")) is None
