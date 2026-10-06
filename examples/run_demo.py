"""Replay authored multilingual fixtures through the public validation contracts.

Standard library only. No inference, credentials, network requests, or file writes.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = Path(__file__).with_name("fixture.json")


def load_public_module(name: str, relative_path: str) -> ModuleType:
    """Load the actual stdlib-only module without optional package dependencies."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise ImportError(relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


submission = load_public_module("lava_demo_submission", "src/lava/evaluation/submission.py")
structured = load_public_module("lava_demo_structured", "src/lava/readers/structured_output.py")


def csv_payload(rows: list[dict[str, str]], columns: tuple[str, ...]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def run_demo(fixture_path: Path = FIXTURE_PATH) -> dict[str, Any]:
    """Return validation evidence; fixture equality is not a semantic judge."""
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    documents = {document["id"]: document for document in fixture["documents"]}
    questions = fixture["questions"]
    page_counts = {key: len(document["pages"]) for key, document in documents.items()}
    question_rows = [
        {key: question[key] for key in submission.TEST_COLUMNS} for question in questions
    ]
    inputs = submission.SubmissionInputs(
        order=tuple(question["id"] for question in questions),
        questions={question["id"]: row for question, row in zip(questions, question_rows)},
        source_sha256={"synthetic_fixture.json": submission.sha256(fixture_path.read_bytes())},
    )

    rows = []
    accepted = []
    for question in questions:
        allowed = [page["page_number"] for page in documents[question["file_id"]]["pages"]]
        parsed = structured.parse_structured_output(
            json.dumps(question["provided_response"], ensure_ascii=False),
            valid_page_numbers=allowed,
        )
        if not parsed.valid:
            raise ValueError(f"Fixture response was rejected: {parsed.error}")
        response = parsed.parsed
        answer_matches = response.answer == question["expected_answer"]
        evidence_matches = list(response.evidence_pages) == question["expected_evidence_pages"]
        if not answer_matches or not evidence_matches:
            raise ValueError("Provided response differs from the authored fixture expectation")
        rows.append(
            {
                "id": question["id"],
                "answer": response.answer,
                "evidence_page_number": json.dumps(list(response.evidence_pages)),
            }
        )
        accepted.append(
            {
                "id": question["id"],
                "answer": response.answer,
                "evidence_pages": list(response.evidence_pages),
                "matches_authored_fixture": True,
            }
        )

    validation = submission.validate_submission_csv(
        csv_payload(rows, submission.SUBMISSION_COLUMNS), inputs, page_counts
    )
    invalid_page = page_counts[questions[0]["file_id"]] + 1
    malformed = [dict(row) for row in rows]
    malformed[0]["evidence_page_number"] = json.dumps([invalid_page])
    try:
        submission.validate_submission_csv(
            csv_payload(malformed, submission.SUBMISSION_COLUMNS), inputs, page_counts
        )
    except ValueError as error:
        rejected = {"case": "out_of_bounds_citation", "page": invalid_page, "reason": str(error)}
    else:
        raise AssertionError("The validator accepted an out-of-bounds citation")

    # A valid schema cannot establish whether an answer agrees with the document.
    wrong_answer = [dict(row) for row in rows]
    wrong_answer[0]["answer"] = "9000"
    wrong_check = submission.validate_submission_csv(
        csv_payload(wrong_answer, submission.SUBMISSION_COLUMNS), inputs, page_counts
    )
    return {
        "scope": fixture["scope"],
        "accepted": accepted,
        "submission_contract": validation,
        "rejected": rejected,
        "semantic_limit": {
            "wrong_answer": "9000",
            "schema_valid": wrong_check["schema_valid"],
            "matches_authored_fixture": False,
            "explanation": "Schema and page-range checks alone do not verify answer correctness.",
        },
        "model_inference_performed": False,
        "confidence_is_calibrated": False,
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2))
