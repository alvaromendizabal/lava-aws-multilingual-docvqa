"""Regression coverage for lossless model-output page normalization."""

from __future__ import annotations

import json

import pytest

from lava.evaluation.schemas import AnswerFormat
from lava.readers.output_normalization import normalize_model_evidence_pages
from lava.readers.parsing import ReaderOutputError, parse_reader_response


def test_exact_decimal_strings_are_normalized() -> None:
    result = normalize_model_evidence_pages(["11", "12"], allowed_pages=(11, 12, 13))

    assert result.pages == (11, 12)
    assert result.converted_decimal_strings == 2
    assert result.changed is True


def test_model_answer_content_is_not_changed() -> None:
    raw = json.dumps(
        {
            "answer": "ドイツ",
            "evidence_pages": ["11", "12"],
            "confidence": 0.8,
            "abstain": False,
        },
        ensure_ascii=False,
    )

    prediction = parse_reader_response(
        question_id="q-01",
        answer_format=AnswerFormat.STRING,
        raw_response=raw,
        allowed_pages=(11, 12),
    )

    assert prediction.answer == "ドイツ"
    assert prediction.evidence_pages == (11, 12)


@pytest.mark.parametrize(
    "value",
    [
        [11.0],
        [True],
        ["+11"],
        ["011"],
        ["11-12"],
        ["page 11"],
        [0],
        ["0"],
        [11, "11"],
    ],
)
def test_ambiguous_or_duplicate_values_are_rejected(value: list[object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_model_evidence_pages(value)


def test_out_of_context_page_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside the supplied context"):
        normalize_model_evidence_pages([3], allowed_pages=(1, 2))


def test_reader_parser_keeps_ambiguous_string_invalid() -> None:
    raw = json.dumps(
        {
            "answer": "ドイツ",
            "evidence_pages": ["11-12"],
            "confidence": 0.8,
            "abstain": False,
        },
        ensure_ascii=False,
    )

    with pytest.raises(ReaderOutputError) as error:
        parse_reader_response(
            question_id="q-01",
            answer_format=AnswerFormat.STRING,
            raw_response=raw,
            allowed_pages=(11, 12),
        )

    assert error.value.code == "invalid_pages"
