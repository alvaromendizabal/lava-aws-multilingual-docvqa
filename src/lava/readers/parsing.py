"""Robust structured-output parsing without retaining hidden reasoning text."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
)

from lava.evaluation.normalization import parse_evidence_pages
from lava.evaluation.schemas import AnswerFormat
from lava.readers.schemas import ReaderPrediction
from lava.readers.structured_output import extract_json_candidate

_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE)
ScalarAnswer = StrictStr | StrictInt | StrictFloat


class ReaderOutputError(ValueError):
    """Raised when a model response cannot be converted to the required schema."""

    def __init__(self, message: str, *, code: str = "invalid_output") -> None:
        super().__init__(message)
        self.code = code


class _ReaderPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer: ScalarAnswer | list[ScalarAnswer]
    evidence_pages: list[StrictInt]
    confidence: StrictFloat
    abstain: StrictBool

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        """Reject non-finite or out-of-range confidence values."""
        if not math.isfinite(value):
            raise ValueError("Confidence must be finite")
        if not 0.0 <= value <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        return value


def strip_hidden_reasoning(value: str) -> str:
    """Remove Qwen-style thinking blocks before parsing or persistence."""
    return _THINK_BLOCK.sub("", value).strip()


def _first_json_object(value: str) -> str:
    text = _FENCE.sub("", value.strip())
    start = text.find("{")
    if start < 0:
        raise ReaderOutputError(
            "Reader response did not contain a JSON object",
            code="missing_json_object",
        )
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ReaderOutputError(
        "Reader response contained an unterminated JSON object",
        code="unterminated_json_object",
    )


def _canonical_answer(value: Any, answer_format: AnswerFormat) -> str:
    if answer_format in {AnswerFormat.ORDERED_LIST, AnswerFormat.UNORDERED_LIST}:
        if isinstance(value, str):
            try:
                parsed = json.loads(extract_json_candidate(value))
            except json.JSONDecodeError:
                parsed = [value] if value.strip() else []
        else:
            parsed = value
        if not isinstance(parsed, list):
            raise ReaderOutputError(
                "List answer formats require a JSON array",
                code="wrong_answer_type",
            )
        normalized = [str(item).strip() for item in parsed if str(item).strip()]
        return json.dumps(normalized, ensure_ascii=False)
    if isinstance(value, (dict, list)):
        raise ReaderOutputError(
            "Scalar answer formats require a string or number",
            code="wrong_answer_type",
        )
    return str(value).strip()


def parse_reader_response(
    *,
    question_id: str,
    answer_format: AnswerFormat,
    raw_response: str,
    allowed_pages: tuple[int, ...],
) -> ReaderPrediction:
    """Parse one model response into the strict public prediction contract."""
    digest = hashlib.sha256(raw_response.encode()).hexdigest()
    cleaned = strip_hidden_reasoning(raw_response)
    try:
        decoded = json.loads(extract_json_candidate(_first_json_object(cleaned)))
    except json.JSONDecodeError as error:
        raise ReaderOutputError(str(error), code="invalid_json") from error
    if not isinstance(decoded, dict):
        raise ReaderOutputError("Reader output must be a JSON object", code="wrong_root_type")
    try:
        payload = _ReaderPayload.model_validate(decoded)
    except ValidationError as error:
        raise ReaderOutputError(str(error), code="schema_validation_failed") from error
    try:
        evidence_pages = parse_evidence_pages(payload.evidence_pages)
    except (TypeError, ValueError) as error:
        raise ReaderOutputError(f"Invalid evidence pages: {error}", code="invalid_pages") from error
    if not set(evidence_pages).issubset(set(allowed_pages)):
        raise ReaderOutputError(
            "Reader cited a page outside the supplied oracle evidence set",
            code="out_of_scope_page",
        )
    answer = _canonical_answer(payload.answer, answer_format)
    if payload.abstain and (answer not in {"", "[]"} or evidence_pages):
        raise ReaderOutputError(
            "Abstentions must return an empty answer and no evidence pages",
            code="invalid_abstention",
        )
    try:
        return ReaderPrediction(
            question_id=question_id,
            answer_format=answer_format,
            answer=answer,
            evidence_pages=evidence_pages,
            confidence=payload.confidence,
            abstain=payload.abstain,
            schema_valid=True,
            parser_error_code=None,
            raw_response_sha256=digest,
        )
    except ValidationError as error:
        raise ReaderOutputError(str(error), code="prediction_validation_failed") from error
