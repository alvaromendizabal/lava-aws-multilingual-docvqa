"""Keep published research counts and cloud-derived visual results auditable."""

import hashlib
import json
import math
from pathlib import Path

from lava.retrieval.feature_research import bm25_feature_grid, exploration_grid, fusion_grid
from lava.retrieval.research import research_contract

ROOT = Path(__file__).resolve().parents[2]


def test_research_candidate_accounting_matches_executable_catalog():
    path = ROOT / "reports/retrieval/feature_search.json"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == path.with_suffix(".sha256").read_text().strip()
    )
    report = json.loads(path.read_bytes())
    counts = report["candidate_space"]
    assert counts["scalar_bm25_features_generated"] == len(bm25_feature_grid())
    assert counts["fusion_and_exploration_policies_generated"] == len(fusion_grid()) + len(
        exploration_grid()
    )
    assert (
        counts["scalar_bm25_features_generated"]
        == counts["unique_bm25_ranking_signatures"] + counts["duplicate_bm25_rankings_rejected"]
    )
    assert (
        counts["fusion_and_exploration_policies_generated"]
        == counts["unique_fusion_ranking_signatures"] + counts["duplicate_fusion_rankings_rejected"]
    )
    assert (
        counts["total_candidate_configurations_generated"]
        == counts["scalar_bm25_features_generated"]
        + counts["fusion_and_exploration_policies_generated"]
    )
    assert not report["decision"]["changed"]
    assert report["contract"] == research_contract(
        ROOT, report["contract"]["source_retrieval_contract"]
    )
    assert counts["total_candidate_configurations_generated"] == (
        counts["globally_unique_ranking_signatures"] + counts["global_duplicate_rankings"]
    )
    assert counts["global_duplicate_rankings"] == (
        counts["duplicate_bm25_rankings_rejected"]
        + counts["duplicate_fusion_rankings_rejected"]
        + counts["cross_stage_duplicates"]
    )
    assert len(report["document_folds"]) == 10
    for fold in report["document_folds"]:
        audit = fold["selection_audit"]
        assert audit["training_documents"] == 4
        assert audit["training_questions"] + fold["held_out_questions"] == 16
        assert audit["unique_training_rankings"] <= audit["generated"]
    assert len(report["feature_family_diagnostics_pooled_only"]) == 11


def test_visual_result_matches_cloud_provenance_and_baseline():
    path = ROOT / "reports/retrieval/visual_search.json"
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == path.with_suffix(".sha256").read_text().strip()
    closeout = json.loads((ROOT / "reports/portfolio/closeout.json").read_bytes())
    assert digest == closeout["visual_retrieval"]["summary_sha256"]
    report = json.loads(payload)
    internal_digest = report.pop("summary_sha256")
    assert (
        internal_digest
        == hashlib.sha256(
            json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert report["model"]["revision"] == closeout["visual_retrieval"]["model_revision"]
    assert (report["question_count"], report["document_count"], report["physical_page_count"]) == (
        16,
        5,
        74,
    )
    baseline = json.loads((ROOT / "reports/retrieval/feature_search.json").read_bytes())["baseline"]
    for metric, value in report["systems"]["bm25"].items():
        assert math.isclose(value, baseline[metric], abs_tol=1e-12)
    assert all(
        0 <= value <= 1 for metrics in report["systems"].values() for value in metrics.values()
    )
    assert not report["selection"]["hidden_test_feedback_used"]
    assert not report["selection"]["production_change_automatic"]
