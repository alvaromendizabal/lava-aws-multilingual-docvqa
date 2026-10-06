"""Check the public closeout contract and recompute aggregate arithmetic offline.

This verifier deliberately does not authenticate private receipts or rerun inference.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any


class EvidenceError(ValueError):
    """Published evidence violates its scope or arithmetic contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def count(value: Any, name: str) -> int:
    require(type(value) is int and value >= 0, f"{name} must be a nonnegative integer")
    return int(value)


def score(value: Any) -> Decimal:
    require(type(value) in (int, float), "Scores must be numeric")
    result = Decimal(str(value))
    require(result.is_finite() and 0 <= result <= 1, "Scores must be finite and within [0, 1]")
    return result


def verify(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Validate explicit boundaries and derive auditable aggregate measurements."""
    require(evidence["schema_version"] == 1, "Unsupported schema version")
    identity = evidence["source_identity"]
    require(
        re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) is not None,
        "Source identity must contain a SHA-256 digest",
    )
    require(
        identity["kind"] == "private_owner_execution_archive",
        "Private source identity must not be presented as public inference proof",
    )
    official = evidence["official_results"]
    require(official["metric"] == "official_competition_score", "Official score scope mismatch")
    require(official["private_scores_used_for_tuning"] is False, "Private-score tuning unsupported")
    best, diagnostic = official["historical_best"], official["later_diagnostic"]
    contract = evidence["prediction_contract"]
    records = count(contract["records"], "records")
    accepted = count(contract["structurally_accepted_non_abstaining"], "structural acceptance")
    unresolved = count(contract["unresolved"], "unresolved")
    require(records > 0 and accepted + unresolved == records, "Coverage accounting mismatch")
    require(contract["strict_complete"] is False, "Closeout is not strict submission completion")
    require(
        contract["factual_correctness_verified"] is False,
        "Structural acceptance must not be presented as factual correctness",
    )
    for candidate in (best, diagnostic):
        require(candidate["rows"] == records, "Candidate row count mismatch")
        require(
            candidate["strict_complete"] is False, "Historical candidate is not strict complete"
        )
    require(best["template_fallback_rows"] == unresolved, "Template-fallback accounting mismatch")
    require(
        diagnostic["compatibility_abstention_rows"] == unresolved,
        "Diagnostic abstention accounting mismatch",
    )
    deltas = {
        split: float(score(diagnostic[split]) - score(best[split]))
        for split in ("public", "private")
    }
    comparison = evidence["comparison"]
    changed = count(comparison["rows_changed"], "changed rows")
    unchanged = count(comparison["rows_unchanged"], "unchanged rows")
    answer = count(comparison["answer_changed"], "answer changes")
    pages = count(comparison["evidence_changed"], "evidence changes")
    require(changed + unchanged == records, "Changed/unchanged accounting mismatch")
    require(max(answer, pages) <= changed <= answer + pages, "Impossible overlapping change counts")
    development = evidence["development_context"]
    score(development["local_score"])
    require(
        development["metric"] == "local_published_formula_score"
        and development["applies_to"] == "original_frozen_development_route_only"
        and development["applies_to_recovered_composite"] is False
        and development["recomputed_by_closeout"] is False,
        "Development metric scope must remain separate from full-test submissions",
    )
    require(
        development["questions"] == 16 and development["documents"] == 5,
        "Published development scope must disclose 16 questions across five documents",
    )
    execution = evidence["execution"]
    planned = count(execution["planned_tasks"], "planned tasks")
    completed = count(execution["completed_tasks"], "completed tasks")
    accounted = sum(count(n, "stage tasks") for n in execution["stage_task_counts"].values())
    require(planned == completed == accounted == 14, "Closeout task accounting mismatch")
    require(execution["outcome"] == "reconciliation_completed", "Unsupported execution outcome")
    require(
        all(execution[key] == 0 for key in ("new_model_calls", "new_answers", "new_submissions")),
        "Closeout must not claim new inference, answers or submissions",
    )
    notebook = evidence["owner_notebook"]
    require(
        notebook["browser_rendering_asserted"] is False,
        "Persisted notebook outputs do not prove browser rendering",
    )
    require(
        evidence["reproduction_boundary"]["private_evidence_not_reproduced"],
        "Private evidence reproduction limits must be disclosed",
    )
    return {
        "status": "PUBLIC_AGGREGATE_CONTRACT_VERIFIED",
        "diagnostic_minus_best": deltas,
        "structural_coverage_percent": round(100 * accepted / records, 4),
        "answer_and_evidence_changed": answer + pages - changed,
        "answer_only_changed": changed - pages,
        "evidence_only_changed": changed - answer,
        "completed_closeout_tasks": accounted,
        "strict_complete": False,
        "factual_correctness_established": False,
        "private_inference_reproduced": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "research" / "submission_closeout.json",
    )
    args = parser.parse_args()
    try:
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
        print(json.dumps(verify(evidence), indent=2, sort_keys=True))
    except (EvidenceError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Closeout evidence invalid: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
