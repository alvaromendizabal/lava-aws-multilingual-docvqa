"""Semantic report regressions use verified aggregates and fresh offline imports."""

from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from lava.evaluation.analysis import (
    compare_semantic_runs,
    score_gap_profile,
    validate_semantic_metrics,
)
from lava.evaluation.reporting import load_report, render_report

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def pilots():
    return [run for run in load_report(ROOT)["runs"] if run["complete"]]


def test_offline_report_does_not_import_model_download_or_widget_libraries():
    code = """
import importlib.abc
import sys
from pathlib import Path
class OfflineImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'huggingface_hub', 'torch', 'transformers', 'tqdm'}:
            raise AssertionError('Unexpected model dependency: ' + fullname)
sys.meta_path.insert(0, OfflineImports())
from lava.evaluation.reporting import load_report, render_report
assert 'Semantic model comparisons' in render_report(load_report(Path.cwd()))
print('OFFLINE_REPORT_VERIFIED')
"""
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "OFFLINE_REPORT_VERIFIED"
    assert result.stderr == ""


def test_real_semantic_comparison_uses_official_overall_not_exact_diagnostics(pilots):
    pairs = compare_semantic_runs(*pilots)
    overall, answer, grounding = pairs
    assert [pair["metric"] for pair in pairs] == ["overall", "answer", "grounding"]
    assert overall["question_mean_delta"] == pytest.approx(0.13199404761904765)
    assert overall["mean_delta"] == pytest.approx(0.10023809523809528)
    assert overall["documents_improved"] == 3
    assert overall["documents_tied"] == 1
    assert overall["documents_regressed"] == 1
    assert overall["exact_two_sided_sign_flip_p_value"] == 0.375
    assert answer["question_mean_delta"] == pytest.approx(0.2952380952380953)
    assert grounding["question_mean_delta"] == -0.03125


@pytest.mark.parametrize("change", ["stale", "judge", "scope", "manifest", "generation"])
def test_incompatible_semantic_runs_are_not_compared(pilots, change):
    left, right = deepcopy(pilots)
    semantic = right["semantic_summary"]
    if change == "stale":
        semantic["contract_current"] = False
    elif change == "judge":
        semantic["judge_contract"]["contract_id"] = "different"
    elif change == "scope":
        semantic["evidence_scope"] = "retrieved_full_document_pages"
    elif change == "manifest":
        semantic["source"]["asset_manifest_sha256"] = "different"
    else:
        right["summary"]["generation"] = {"different": True}
    assert compare_semantic_runs(left, right) == []


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_semantic_values_are_rejected(pilots, value):
    run = deepcopy(pilots[0])
    run["semantic_summary"]["metrics"]["question_micro"]["answer"] = value
    with pytest.raises(ValueError, match="finite"):
        score_gap_profile(run)


@pytest.mark.parametrize("change", ["coverage", "missing_slice", "formula", "weighting"])
def test_inconsistent_semantic_aggregates_are_rejected(pilots, change):
    run = deepcopy(pilots[0])
    metrics = run["semantic_summary"]["metrics"]
    if change == "coverage":
        metrics["question_count"] -= 1
    elif change == "missing_slice":
        metrics["by_document"].pop("doc-01")
    elif change == "formula":
        metrics["question_micro"]["overall"] = 0.5
    else:
        metrics["document_macro"] = metrics["question_micro"].copy()
    with pytest.raises(ValueError):
        validate_semantic_metrics(metrics, run["summary"])


def test_score_gap_components_add_to_exact_overall_gap(pilots):
    for run in pilots:
        profile = score_gap_profile(run)
        assert profile["answer_gap"] + profile["grounding_gap"] == pytest.approx(
            profile["overall_gap"]
        )
        assert sum(
            row["overall_gap_contribution"] for row in profile["by_answer_format"]
        ) == pytest.approx(profile["overall_gap"])
    nine = score_gap_profile(pilots[1])
    assert nine["by_answer_format"][0]["answer_format"] == "string"
    assert nine["by_answer_format"][0]["overall_gap_contribution"] == pytest.approx(0.078125)


def test_report_labels_semantic_and_diagnostic_comparisons_separately():
    html = render_report(load_report(ROOT))
    assert "Semantic model comparisons" in html
    assert "Normalized-exact diagnostic comparisons" in html
    assert "+13.20 pp" in html
    assert "Where the remaining score is lost" in html
    assert "7.81 pp" in html
    assert '<th scope="col">Semantic VQA</th>' in html
