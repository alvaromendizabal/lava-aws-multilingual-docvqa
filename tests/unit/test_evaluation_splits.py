"""Tests for nested leave-one-document-out fold construction."""

import pytest

from lava.evaluation.schemas import AnswerFormat, ReferenceRecord
from lava.evaluation.splits import (
    build_nested_leave_one_document_out_folds,
)


def _record(question_id: str, document_id: str) -> ReferenceRecord:
    return ReferenceRecord(
        question_id=question_id,
        document_id=document_id,
        question="質問",
        answer_format=AnswerFormat.STRING,
        answer="答え",
        evidence_pages=(1,),
        language="ja",
    )


def test_nested_folds_isolate_outer_and_inner_documents() -> None:
    """Every outer and inner validation document must remain fully isolated."""
    records = tuple(_record(f"q{index}", f"d{index}") for index in range(1, 6))
    folds = build_nested_leave_one_document_out_folds(records)
    assert len(folds) == 5
    assert {fold.validation_document_id for fold in folds} == {
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
    }
    for outer in folds:
        assert len(outer.inner_folds) == 4
        assert outer.validation_document_id not in outer.training_document_ids
        assert {inner.validation_document_id for inner in outer.inner_folds} == set(
            outer.training_document_ids
        )
        for inner in outer.inner_folds:
            assert outer.validation_document_id not in inner.training_document_ids
            assert inner.validation_document_id not in inner.training_document_ids


def test_duplicate_question_ids_fail_closed() -> None:
    """A duplicated label record cannot enter the evaluation protocol."""
    with pytest.raises(ValueError, match="Duplicate question ID"):
        build_nested_leave_one_document_out_folds(
            (_record("q1", "d1"), _record("q1", "d2"), _record("q3", "d3"))
        )
