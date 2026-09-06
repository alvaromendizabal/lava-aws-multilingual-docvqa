"""Strict LAVA submission validation, without inference or competition upload."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

SUBMISSION_COLUMNS = ("id", "answer", "evidence_page_number")
TEST_COLUMNS = ("id", "file_id", "question", "answer_format", "language")
FORMATS = {"string", "number", "ordered_list", "unordered_list"}


def sha256(payload: bytes) -> str:
    """Identify exact file bytes, including CSV quoting and row order."""
    return hashlib.sha256(payload).hexdigest()


def _csv_rows(payload: bytes, columns: tuple[str, ...]) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
    if tuple(reader.fieldnames or ()) != columns:
        raise ValueError("CSV columns or column order do not match the competition schema")
    rows = list(reader)
    if not rows or any(set(row) != set(columns) or None in row.values() for row in rows):
        raise ValueError("CSV is empty or contains a malformed row")
    ids = [row["id"] for row in rows]
    if any(not value or value.strip() != value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("CSV IDs must be nonempty, unique, and free of surrounding whitespace")
    return rows


@dataclass(frozen=True)
class SubmissionInputs:
    """Verified competition inputs; template order is authoritative."""

    order: tuple[str, ...]
    questions: dict[str, dict[str, str]]
    source_sha256: dict[str, str]


def load_submission_inputs(
    test: bytes, template: bytes, contract: dict[str, Any]
) -> SubmissionInputs:
    """Bind input content, complete ID coverage, formats, languages and document counts."""
    payloads = {"test.csv": test, "sample_submission.csv": template}
    hashes = {name: sha256(payload) for name, payload in payloads.items()}
    if any(hashes[name] != contract["source_files"][name]["sha256"] for name in payloads):
        raise ValueError("Submission input checksum differs from the frozen source")
    questions = _csv_rows(test, TEST_COLUMNS)
    samples = _csv_rows(template, SUBMISSION_COLUMNS)
    by_id = {row["id"]: row for row in questions}
    order = tuple(row["id"] for row in samples)
    if set(order) != set(by_id) or len(order) != contract["expected_question_count"]:
        raise ValueError("Submission template and test IDs do not have complete matching coverage")
    if len({row["file_id"] for row in questions}) != contract["expected_document_count"]:
        raise ValueError("Test document coverage differs from the frozen source")
    for column, expected in (
        ("answer_format", contract["answer_format_counts"]),
        ("language", contract["language_counts"]),
    ):
        if dict(Counter(row[column] for row in questions)) != expected:
            raise ValueError("Test format or language coverage differs from the frozen source")
    if any(
        row["answer_format"] not in FORMATS
        or not row["file_id"].strip()
        or not row["question"].strip()
        for row in questions
    ):
        raise ValueError("Test inputs contain an unsupported format or an empty required field")
    return SubmissionInputs(order, by_id, hashes)


def _list_value(value: str, label: str) -> list[Any]:
    """Read JSON or Python list literals; never execute source text."""
    if len(value) > 1_000_000:
        raise ValueError(f"{label} exceeds the supported serialization size")
    try:
        result = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError, RecursionError):
        raise ValueError(f"{label} must contain a list literal") from None
    if not isinstance(result, list):
        raise TypeError(f"{label} must contain a list")
    return result


def _answer(value: Any, answer_format: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Every question requires a nonempty string answer")
    if answer_format in {"ordered_list", "unordered_list"}:
        items = _list_value(value, "List answer")
        if not items or any(
            isinstance(item, bool)
            or not isinstance(item, (str, int, float))
            or (isinstance(item, str) and not item.strip())
            or (isinstance(item, float) and not math.isfinite(item))
            for item in items
        ):
            raise ValueError("List answers require nonempty, finite scalar items")
        return json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    return value


def _pages(value: Any, page_count: int) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError("Predicted evidence must be a nonempty list of page numbers")
    if any(type(page) is not int or not 1 <= page <= page_count for page in value):
        raise ValueError("Evidence pages must be integers within the source PDF page range")
    if len(set(value)) != len(value):
        raise ValueError("Duplicate evidence pages are not allowed")
    return sorted(value)


def validate_provenance(value: dict[str, Any], inputs: SubmissionInputs) -> None:
    """Reject oracle or training predictions presented as competition test results."""
    required = {
        "split": "test",
        "evidence_scope": "retrieved_full_document_pages",
        "test_csv_sha256": inputs.source_sha256["test.csv"],
        "question_count": len(inputs.order),
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("Submission requires complete test inference without oracle evidence")
    if not isinstance(value.get("model_id"), str) or not value["model_id"].strip():
        raise ValueError("Submission provenance requires a model identity")
    for key in ("model_revision", "code_commit"):
        if not isinstance(value.get(key), str) or not re.fullmatch(r"[0-9a-f]{40}", value[key]):
            raise ValueError("Submission provenance requires pinned model and code revisions")


def build_submission(
    inputs: SubmissionInputs,
    predictions: bytes,
    page_counts: dict[str, int],
    provenance: dict[str, Any],
) -> bytes:
    """Create the exact CSV only after every prediction and its provenance pass validation."""
    validate_provenance(provenance, inputs)
    count_bytes = (
        json.dumps(page_counts, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    if provenance.get("predictions_sha256") != sha256(predictions) or provenance.get(
        "page_counts_sha256"
    ) != sha256(count_bytes):
        raise ValueError("Prediction or page-count content differs from its recorded provenance")
    documents = {row["file_id"] for row in inputs.questions.values()}
    if set(page_counts) != documents or any(
        type(count) is not int or count < 1 for count in page_counts.values()
    ):
        raise ValueError("Verified positive PDF page counts are required for every test document")
    by_id = {}
    for line in predictions.splitlines():
        if not line.strip():
            raise ValueError("Prediction JSONL contains an empty record")
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != {"question_id", "answer", "evidence_pages"}:
            raise ValueError("Prediction record does not match the submission schema")
        question_id = row["question_id"]
        if not isinstance(question_id, str) or question_id in by_id:
            raise ValueError("Prediction IDs must be unique strings")
        by_id[question_id] = row
    if set(by_id) != set(inputs.order):
        raise ValueError(
            "Predictions must cover every test ID exactly once; no missing or extra rows"
        )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=SUBMISSION_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for question_id in inputs.order:
        prediction, question = by_id[question_id], inputs.questions[question_id]
        pages = _pages(prediction["evidence_pages"], page_counts[question["file_id"]])
        writer.writerow(
            {
                "id": question_id,
                "answer": _answer(prediction["answer"], question["answer_format"]),
                "evidence_page_number": json.dumps(pages, separators=(",", ":")),
            }
        )
    return output.getvalue().encode("utf-8")


def validate_submission_csv(
    payload: bytes, inputs: SubmissionInputs, page_counts: dict[str, int]
) -> dict[str, Any]:
    """Independently validate CSV coverage, order, list syntax and PDF page boundaries."""
    rows = _csv_rows(payload, SUBMISSION_COLUMNS)
    if tuple(row["id"] for row in rows) != inputs.order:
        raise ValueError("Submission IDs or row order differ from the frozen template")
    for row in rows:
        question = inputs.questions[row["id"]]
        count = page_counts.get(question["file_id"])
        if type(count) is not int or count < 1:
            raise ValueError("Verified PDF page counts are missing or invalid")
        _answer(row["answer"], question["answer_format"])
        _pages(_list_value(row["evidence_page_number"], "Evidence pages"), count)
    return {
        "row_count": len(rows),
        "columns": list(SUBMISSION_COLUMNS),
        "sha256": sha256(payload),
        "schema_valid": True,
        "model_quality_verified": False,
        "organizer_runtime_verified": False,
        "uploaded_to_kaggle": False,
    }
