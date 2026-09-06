from __future__ import annotations

import re
from pathlib import Path

import nbformat

EXPECTED = (
    "00_reproducibility_and_protocol",
    "01_oracle_reader_benchmark_design",
    "02_verified_gpu_execution",
    "03_model_scaling_and_cost",
)
ACCOUNT_ID = re.compile(r"(?<!\d)\d{12}(?!\d)")


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
