"""Current source-bound outputs and actual retained-system metric claims."""

from pathlib import Path

import pytest

from lava.evaluation.system import load_summary
from lava.evaluation.system_reporting import verified_system_figure
from lava.notebook_execution import NOTEBOOK_STEMS, validate_public_notebook
from lava.readers.refinement import load_refinement

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("stem", NOTEBOOK_STEMS)
def test_current_source_bound_publication(stem):
    validate_public_notebook(ROOT, ROOT / "notebooks" / f"{stem}.ipynb")


def test_retained_reader_comparison_preserves_exact_metrics_and_scope():
    text = (ROOT / "README.md").read_text()
    for loader in (load_summary, load_refinement):
        result = loader(ROOT)
        assert result is not None
        assert result["metrics"]["question_count"] == 16
        assert result["official_server_parity"] is False
        assert f"{result['metrics']['question_micro']['overall']:.2%}" in text
    for name in ("quality.svg", "documents.svg", "questions.svg"):
        assert "<svg" in verified_system_figure(ROOT, name)
    assert "two unresolved answers" in text
