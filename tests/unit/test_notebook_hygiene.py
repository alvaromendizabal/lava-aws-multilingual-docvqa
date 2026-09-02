from pathlib import Path

import nbformat


def test_phase5_notebooks_have_no_saved_outputs() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "notebooks").glob("0[01]_*.ipynb"))
    assert len(paths) == 2
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            assert not cell.get("outputs", [])
            assert cell.get("execution_count") is None
