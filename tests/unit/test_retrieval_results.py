"""Public retrieval evidence stays complete, distinct from reader scores and resumable."""

from __future__ import annotations

import json
from pathlib import Path

from lava.retrieval.pipeline import load_public_report

ROOT = Path(__file__).resolve().parents[2]


def test_verified_public_retrieval_report_keeps_full_denominators_and_score_boundaries():
    summary = load_public_report(ROOT)
    assert summary["question_count"] == 16 and summary["document_count"] == 5
    assert len(summary["document_coverage"]) == 5
    assert len({row["question"] for row in summary["per_question"]}) == 16
    assert summary["reader_evaluated"] is False and summary["local_lava_overall"] is None
    budgets = summary["implementation"]["config"]["budgets"]
    for method in ("page_order", "bm25"):
        assert len([row for row in summary["per_question"] if row["method"] == method]) == 16 * len(
            budgets
        )
        values = summary["methods"][method]["question_average"]
        recalls = [values[str(k)]["recall_at_k"] for k in budgets]
        assert recalls == sorted(recalls)
        for metrics in values.values():
            assert all(0 <= value <= 1 for value in metrics.values())
            assert metrics["all_evidence_at_k"] <= metrics["recall_at_k"]
    serialized = json.dumps(summary)
    assert '"question": "q-' in serialized
    assert '"answer":' not in serialized and '"text":' not in serialized


def test_live_resume_record_matches_completed_contract_and_reuses_every_stage():
    summary = load_public_report(ROOT)
    validation = json.loads((ROOT / "reports/retrieval/validation.json").read_bytes())
    first, second = validation["attempts"]
    for row in (first, second):
        assert row["contract_id"] == summary["contract_id"]
        assert row["total_elapsed_seconds"] > 0
        assert row["timestamp_utc"].endswith("+00:00")
    assert first["documents_computed"] == 5 and first["queries_computed"] == 16
    assert second["documents_computed"] == second["queries_computed"] == 0
    assert second["documents_reused"] == 5 and second["queries_reused"] == 16
    assert validation["log_archives_verified"] is True
    assert validation["new_gpu_jobs"] == 0
