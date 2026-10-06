"""Guard against overstated completion and metric-scope regressions."""

from __future__ import annotations

import importlib.util
import json
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "verify_submission_closeout", ROOT / "examples" / "verify_submission_closeout.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
EVIDENCE = json.loads((ROOT / "research" / "submission_closeout.json").read_text())


class SubmissionCloseoutTests(unittest.TestCase):
    def test_contract_reports_remaining_gap_and_comparable_deltas(self) -> None:
        result = MODULE.verify(EVIDENCE)
        self.assertEqual(result["diagnostic_minus_best"], {"public": -0.01, "private": -0.04})
        self.assertAlmostEqual(result["structural_coverage_percent"], 99.6795)
        self.assertEqual(result["answer_and_evidence_changed"], 98)
        self.assertEqual(result["answer_only_changed"], 278)
        self.assertEqual(result["evidence_only_changed"], 13)
        self.assertIs(result["strict_complete"], False)
        self.assertIs(result["factual_correctness_established"], False)
        self.assertIs(result["private_inference_reproduced"], False)

    def test_rejects_misleading_or_inconsistent_claims(self) -> None:
        mutations = [
            ("prediction_contract", "strict_complete", True, "not strict submission completion"),
            ("prediction_contract", "factual_correctness_verified", True, "factual correctness"),
            ("prediction_contract", "structurally_accepted_non_abstaining", 624, "Coverage"),
            ("development_context", "applies_to_recovered_composite", True, "metric scope"),
            ("development_context", "metric", "official_competition_score", "metric scope"),
            ("development_context", "questions", 624, "development scope"),
            ("comparison", "evidence_changed", 400, "overlapping"),
            ("execution", "new_submissions", 1, "new inference"),
            ("execution", "completed_tasks", 13, "task accounting"),
            ("owner_notebook", "browser_rendering_asserted", True, "browser rendering"),
        ]
        for section, field, value, message in mutations:
            with self.subTest(section=section, field=field):
                tampered = deepcopy(EVIDENCE)
                tampered[section][field] = value
                with self.assertRaisesRegex(MODULE.EvidenceError, message):
                    MODULE.verify(tampered)

    def test_development_context_matches_existing_aggregate_source(self) -> None:
        frontier = json.loads((ROOT / "research" / "frontier_status.json").read_text())
        context = EVIDENCE["development_context"]
        self.assertEqual(context["questions"], frontier["evaluation"]["panel_questions"])
        self.assertEqual(context["documents"], frontier["evaluation"]["panel_documents"])
        self.assertEqual(
            context["local_score"], frontier["validated_routed_challenger"]["local_lava"]
        )

    def test_aggregate_excludes_private_records_and_cloud_locations(self) -> None:
        text = json.dumps(EVIDENCE)
        for forbidden in ("q_0", "s3://", "/home/", "AKIA", "answer_text", "submission_ref"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)

    def test_rejects_nonfinite_scores(self) -> None:
        tampered = deepcopy(EVIDENCE)
        tampered["official_results"]["later_diagnostic"]["public"] = float("nan")
        with self.assertRaisesRegex(MODULE.EvidenceError, "finite"):
            MODULE.verify(tampered)


if __name__ == "__main__":
    unittest.main()
