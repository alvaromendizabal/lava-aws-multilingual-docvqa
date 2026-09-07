"""The notebook folder is the complete, readable public interface."""

import subprocess
from pathlib import Path

import nbformat

from lava.notebook_execution import NOTEBOOK_STEMS

ROOT = Path(__file__).resolve().parents[2]


def test_one_canonical_notebook_sequence_has_no_duplicate_sources():
    assert sorted(path.stem for path in (ROOT / "notebooks").glob("*.ipynb")) == list(
        NOTEBOOK_STEMS
    )
    assert not list((ROOT / "notebooks").glob("*.py"))
    assert not list((ROOT / "reports").rglob("*.ipynb"))
    assert not (ROOT / ".jupytext.toml").exists()


def test_git_preserves_outputs_and_notebooks_have_no_pairing_metadata():
    for stem in NOTEBOOK_STEMS:
        path = ROOT / "notebooks" / f"{stem}.ipynb"
        notebook = nbformat.read(path, 4)
        assert notebook.metadata.kernelspec.name == "lava"
        assert "jupytext" not in notebook.metadata
        attribute = subprocess.check_output(
            ["git", "check-attr", "filter", "--", str(path.relative_to(ROOT))], cwd=ROOT, text=True
        )
        assert attribute.strip().endswith(": unset")
