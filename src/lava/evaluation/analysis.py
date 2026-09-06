"""Validated semantic comparisons and additive score-gap analysis."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any

from lava.evaluation.statistics import compare_document_scores

METRICS = {"overall": "Local LAVA overall", "answer": "Semantic VQA", "grounding": "Evidence F1"}


def validate_semantic_metrics(metrics: dict[str, Any], summary: dict[str, Any]) -> None:
    """Check coverage, ranges, formula identities and weighting before comparing scores."""
    documents = summary["document_question_counts"]
    questions = summary["record_count"]
    if metrics["question_count"] != questions or metrics["document_count"] != len(documents):
        raise ValueError("Semantic evaluation coverage differs from reader coverage")
    if metrics["answer_format_counts"] != summary["answer_format_counts"]:
        raise ValueError("Semantic answer-format counts differ from reader coverage")
    groups = {
        "by_document": documents,
        "by_language": summary["language_counts"],
        "by_answer_format": summary["answer_format_counts"],
    }
    if any(sum(counts.values()) != questions for counts in groups.values()):
        raise ValueError("Semantic slice counts must cover every question")
    rows = [metrics["question_micro"], metrics["document_macro"]]
    for group, counts in groups.items():
        if set(metrics[group]) != set(counts):
            raise ValueError("Semantic slices differ from reader coverage")
        rows.extend(metrics[group].values())
    for row in rows:
        if set(row) != set(METRICS):
            raise ValueError("Semantic score row has unexpected metrics")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in row.values()
        ):
            raise ValueError("Semantic scores must be finite numbers between zero and one")
        if not math.isclose(row["overall"], (row["answer"] + row["grounding"]) / 2, abs_tol=1e-12):
            raise ValueError("Semantic overall score violates the published formula")
    for metric in METRICS:
        if not math.isclose(
            metrics["document_macro"][metric],
            fmean(row[metric] for row in metrics["by_document"].values()),
            abs_tol=1e-12,
        ):
            raise ValueError("Semantic document weighting is inconsistent")
        for group, counts in groups.items():
            weighted = sum(metrics[group][key][metric] * n for key, n in counts.items()) / questions
            if not math.isclose(metrics["question_micro"][metric], weighted, abs_tol=1e-12):
                raise ValueError("Semantic question weighting is inconsistent")


def compare_semantic_runs(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare compatible, current judgments; never mix prompts or evidence regimes."""
    a, b = left.get("semantic_summary"), right.get("semantic_summary")
    if not a or not b or not a["contract_current"] or not b["contract_current"]:
        return []
    if a["judge_contract"] != b["judge_contract"] or a["evidence_scope"] != b["evidence_scope"]:
        return []
    compatible = (
        "protocol_lock_id",
        "asset_manifest_sha256",
        "prompt_version",
        "input_mode",
        "generation",
        "metric_boundary",
        "document_question_counts",
    )
    if any(left["summary"].get(key) != right["summary"].get(key) for key in compatible):
        return []
    for field in ("protocol_lock_id", "asset_manifest_sha256"):
        if a["source"][field] != b["source"][field]:
            return []
    if left["summary"]["document_question_counts"] != right["summary"]["document_question_counts"]:
        return []
    for run in (left, right):
        validate_semantic_metrics(run["semantic_summary"]["metrics"], run["summary"])
    results = []
    for metric, label in METRICS.items():
        baseline = {doc: row[metric] for doc, row in a["metrics"]["by_document"].items()}
        challenger = {doc: row[metric] for doc, row in b["metrics"]["by_document"].items()}
        results.append(
            {
                "metric": metric,
                "metric_label": label,
                "baseline_job": left["job_name"],
                "challenger_job": right["job_name"],
                "baseline_document_scores": baseline,
                "challenger_document_scores": challenger,
                "question_mean_delta": b["metrics"]["question_micro"][metric]
                - a["metrics"]["question_micro"][metric],
                "judge_contract_id": a["judge_contract"]["contract_id"],
                **compare_document_scores(baseline, challenger, seed=20260902),
            }
        )
    return results


def score_gap_profile(run: dict[str, Any]) -> dict[str, Any] | None:
    """Decompose the distance from a perfect score without inventing error causes."""
    semantic = run.get("semantic_summary")
    if not semantic or not semantic["contract_current"]:
        return None
    metrics, summary = semantic["metrics"], run["summary"]
    validate_semantic_metrics(metrics, summary)
    total = metrics["question_count"]
    rows = []
    for answer_format, scores in metrics["by_answer_format"].items():
        count = metrics["answer_format_counts"][answer_format]
        answer_gap = (1 - scores["answer"]) * count / (2 * total)
        evidence_gap = (1 - scores["grounding"]) * count / (2 * total)
        rows.append(
            {
                "answer_format": answer_format,
                "question_count": count,
                "answer_score": scores["answer"],
                "grounding_score": scores["grounding"],
                "answer_gap_contribution": answer_gap,
                "grounding_gap_contribution": evidence_gap,
                "overall_gap_contribution": answer_gap + evidence_gap,
            }
        )
    return {
        "job_name": run["job_name"],
        "label": run["label"],
        "question_count": total,
        "overall_score": metrics["question_micro"]["overall"],
        "overall_gap": 1 - metrics["question_micro"]["overall"],
        "answer_gap": (1 - metrics["question_micro"]["answer"]) / 2,
        "grounding_gap": (1 - metrics["question_micro"]["grounding"]) / 2,
        "by_answer_format": sorted(
            rows, key=lambda row: (-row["overall_gap_contribution"], row["answer_format"])
        ),
        "interpretation": "Measured score deficits, not causal diagnoses or promised improvements.",
    }
