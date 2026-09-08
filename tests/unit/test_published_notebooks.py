"""Published notebooks have verified outputs and an explicit project position."""

from pathlib import Path

import nbformat
import pytest

from lava.evaluation.reporting import load_report
from lava.evaluation.system import load_summary
from lava.evaluation.system_reporting import verified_system_figure
from lava.notebook_execution import NOTEBOOK_STEMS, validate_public_notebook
from lava.readers.refinement import load_refinement
from lava.retrieval.pipeline import load_public_report

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("stem", NOTEBOOK_STEMS)
def test_canonical_publication_matches_current_sources_and_verified_inputs(stem):
    path = ROOT / "notebooks" / f"{stem}.ipynb"
    validate_public_notebook(ROOT, path)
    expected = {
        "00": "Data audit complete",
        "01": "Three reader pilots scored",
        "02": "Full pilot scored",
        "03": "Full pilot scored",
        "04": "Full-document retrieval evaluated",
        "05": "Current end-to-end status",
    }[stem[:2]]
    assert expected in path.read_text()
    assert '"contract_current": false' not in path.read_text()


def test_employer_headline_scores_match_the_verified_experiments():
    """The readable entrance must not drift from measured scores or their scope."""
    notebook = nbformat.read(ROOT / "notebooks/00_reproducibility_and_protocol.ipynb", 4)
    rendered = "\n".join(
        output.data["text/html"]
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if "text/html" in output.get("data", {})
    )
    readme = (ROOT / "README.md").read_text()
    report = load_report(ROOT)
    for run in report["current_models"]:
        assert run["complete"] and run["semantic_summary"]["contract_current"]
        metrics = run["semantic_summary"]["metrics"]["question_micro"]
        for metric in ("answer", "overall"):
            score = f"{metrics[metric]:.2%}"
            assert score in readme and score in rendered
    retrieval = load_public_report(ROOT)
    recall = retrieval["methods"]["bm25"]["question_average"]["5"]["recall_at_k"]
    assert f"{recall:.2%}" in readme and f"{recall:.2%}" in rendered
    assert retrieval["reader_evaluated"] is False
    assert retrieval["local_lava_overall"] is None
    assert "submission optional" in rendered
    for loader in (load_summary, load_refinement):
        system = loader(ROOT)
        assert system is not None and system["metrics"]["question_count"] == 16
        assert system["official_server_parity"] is False
        for metric in ("answer", "grounding", "overall"):
            score = f"{system['metrics']['question_micro'][metric]:.2%}"
            assert score in readme
        assert f"{system['metrics']['question_micro']['overall']:.2%}" in rendered
    for figure in ("quality.svg", "documents.svg", "questions.svg"):
        assert "<svg" in verified_system_figure(ROOT, figure)
    assert "post-hoc development finding" in readme
    assert "not official server scores" in readme
